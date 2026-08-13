/*
 * Copyright (c) Kia Shakiba
 *
 * This source code is licensed under the GNU AGPLv3 license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! Integration tests for the
//! `s3_fifo_lazy_demotion_reprieve_hybrid_cache`
//! feature.
//!
//! Run with nightly (required for `allocator_api` via `key_value_pmem`):
//!   cargo +nightly test --test s3_fifo_lazy_demotion_reprieve_hybrid_cache_integration --features s3_fifo_lazy_demotion_reprieve_hybrid_cache
//!
//! Same one-`PaperCache<K, TieredBuffer>` architecture, fast-tier one-access
//! queue, demotion-time reprieve, mid-segment checkpoint, and eviction-time
//! second-chance mechanic as
//! `s3_fifo_ghost_lazy_demotion_fast_admission_midpoint_hybrid_cache` — see
//! that feature's integration test file for the shared coverage; this file
//! mirrors it end to end, minus the ghost-queue tests (there is no ghost
//! queue in this variant), plus tests specific to the new behavior: a
//! one-access-queue key that ages out lands directly in the main queue's
//! slow tier instead of being evicted
//! (`a_key_that_ages_out_lands_directly_in_the_main_queues_slow_tier`), can
//! still be promoted later by a real access
//! (`a_reprieved_key_can_be_promoted_by_a_later_access`), and never causes a
//! terminal (real) eviction on its own
//! (`one_access_pressure_alone_never_causes_a_terminal_eviction`).

#[cfg(feature = "s3_fifo_lazy_demotion_reprieve_hybrid_cache")]
mod s3_fifo_lazy_demotion_reprieve_hybrid_cache_tests {
    use paper_cache::{PaperCache, TieredBuffer, CacheTierSize, Tier, CacheError};

    fn wait_until(timeout: std::time::Duration, mut predicate: impl FnMut() -> bool) -> bool {
        let deadline = std::time::Instant::now() + timeout;
        loop {
            if predicate() {
                return true;
            }
            if std::time::Instant::now() > deadline {
                return false;
            }
            std::thread::sleep(std::time::Duration::from_millis(20));
        }
    }

    /// Admission here is always Fast, so a warm-up key just sitting in the
    /// one-access queue never touches PMEM at all. Force a real demotion
    /// instead (a tiny effective main-fast budget makes a single promoted
    /// key self-demote immediately), which is what actually allocates
    /// through the UMF/TBB pool for the first time in this process.
    /// Forces the one-time UMF/PMEM pool init before any timing-sensitive
    /// assertion runs.
    ///
    /// Much simpler than the fast-admission variants' equivalent: admission
    /// here is *already* a real PMEM write (the one-access queue is
    /// slow-tier), so a single `set()` allocates through the pool on the
    /// calling thread. Those variants had to manufacture a demotion to reach
    /// PMEM at all, which is what made their warm-up racy.
    fn ensure_pmem_allocator_warm() {
        let cache = PaperCache::<u32, TieredBuffer>::new(1_000_000, CacheTierSize::Bytes(1_000), 0.5)
            .expect("warm-up cache should construct");

        cache.set(0u32, b"warm", None).expect("warm-up set should succeed");

        let slow = wait_until(
            std::time::Duration::from_secs(90),
            || cache.tier_of(&0u32) == Some(Tier::Slow),
        );
        assert!(slow, "admission must land in the slow tier in this variant");
    }

    const MIGRATION_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(10);

    /// Returns true if `key` is still in the ONE-ACCESS queue (as opposed to
    /// the main queue's slow segment).
    ///
    /// `tier_of` reads the physical buffer, and in this variant *both*
    /// PMEM-resident structures report `Tier::Slow` -- so `tier_of(k) ==
    /// Slow` is true from the instant a key is admitted and, on its own,
    /// proves nothing happened. The stats gauges cannot separate them either
    /// (`slow_objects` sums both).
    ///
    /// The discriminator has to be behavioural, and conveniently the
    /// behaviour *is* the semantic difference: a one-access resident is
    /// promoted to DRAM by a single access (eager promotion out of
    /// probation), whereas a main-queue slow resident is not -- under lazy
    /// promotion an access only sets its reference bit, and it returns to
    /// DRAM solely via the demotion-boundary reprieve or the eviction-time
    /// second chance.
    ///
    /// Destructive: performs a real access, and promotes the key if it was in
    /// the one-access queue. Use it as the final assertion about that key.
    fn probe_is_in_one_access_queue(
        cache: &PaperCache<u32, TieredBuffer>,
        key: u32,
    ) -> bool {
        cache.get(&key).expect("probe get should succeed");
        // Migration latency is ~1ms p50 (measured); 300ms is a 300x margin,
        // and a negative result genuinely means "no promotion happened".
        std::thread::sleep(std::time::Duration::from_millis(300));
        cache.tier_of(&key) == Some(Tier::Fast)
    }

    #[test]
    fn admission_always_lands_in_slow_tier() {
        ensure_pmem_allocator_warm();

        let cache = PaperCache::<u32, TieredBuffer>::new(
            1_000_000,
            CacheTierSize::Bytes(1_000_000),
            1.0,
        ).expect("cache should construct");

        cache.set(1u32, b"hello world", None).expect("set should succeed");

        // The defining property of this variant: the one-access queue is in
        // PMEM, so every admission is a slow-tier write. The fast-admission
        // variants assert Fast here.
        assert_eq!(cache.tier_of(&1u32), Some(Tier::Slow));

        // Stronger than the tier read alone: nothing may be occupying DRAM.
        //
        // This MUST be checked before any get(). A get is an access, and an
        // access promotes the key out of the one-access queue into DRAM --
        // reading these gauges afterwards races that promotion (which is
        // exactly how this assertion was flaky when first written).
        std::thread::sleep(std::time::Duration::from_millis(300));
        let stats = cache.s3_fifo_lazy_demotion_reprieve_hybrid_stats();
        assert_eq!(stats.fast_objects, 0, "a fresh admission must not occupy the DRAM tier");
        assert_eq!(stats.fast_bytes_used, 0);
        assert_eq!(stats.demotions, 0, "admission is not a demotion -- no bytes moved");

        // Value intact. Deliberately last: this access does promote the key.
        assert_eq!(cache.get(&1u32).unwrap(), b"hello world");
    }

    // ── the signature new mechanic: reprieve instead of eviction ───────────

    #[test]
    fn a_key_that_ages_out_lands_directly_in_the_main_queues_slow_tier() {
        ensure_pmem_allocator_warm();

        let cache = PaperCache::<u32, TieredBuffer>::new(
            1_000_000,
            CacheTierSize::Bytes(1_000_000),
            0.00004, // one_access_capacity = 40 bytes, fits one ~15-byte value
        ).expect("cache should construct");

        cache.set(1u32, b"first value 123", None).expect("set should succeed");
        // Sitting in the one-access queue, which is slow-tier in this variant.
        assert_eq!(cache.tier_of(&1u32), Some(Tier::Slow));

        cache.set(2u32, b"second value 45", None).expect("set should succeed");

        // Unlike the ghost-queue predecessor (where key 1 would be evicted
        // here), key 1 must remain alive -- and must have moved OUT of the
        // one-access queue into the main queue's slow segment.
        std::thread::sleep(std::time::Duration::from_millis(300));
        assert!(cache.has(&1u32), "key 1 must still be a live entry, not evicted");
        assert_eq!(cache.get(&1u32).unwrap(), b"first value 123");
        assert_eq!(cache.tier_of(&1u32), Some(Tier::Slow));

        // The load-bearing assertion. `tier_of` alone cannot show this: key 1
        // read Slow the moment it was admitted. Being in the MAIN queue means
        // a plain access no longer promotes it (lazy promotion), which is
        // precisely what distinguishes it from still sitting in probation.
        assert!(
            !probe_is_in_one_access_queue(&cache, 1u32),
            "key 1 should be in the main queue's slow segment, not still in the one-access queue",
        );
    }

    #[test]
    fn a_reprieved_key_can_be_promoted_by_a_later_access() {
        ensure_pmem_allocator_warm();

        let cache = PaperCache::<u32, TieredBuffer>::new(
            1_000_000,
            CacheTierSize::Bytes(1_000_000),
            0.00004,
        ).expect("cache should construct");

        cache.set(1u32, b"first value 123", None).expect("set should succeed");
        cache.set(2u32, b"second value 45", None).expect("set should succeed");
        // Precondition: key 1 has been spliced into the main queue's slow
        // segment. Asserting `tier_of == Slow` here would be vacuous (it was
        // Slow from admission), so wait on the gauge that actually moves --
        // the main queue gaining its first resident.
        assert!(
            wait_until(MIGRATION_TIMEOUT, || {
                cache.has(&1u32) && cache.tier_of(&1u32) == Some(Tier::Slow)
            }),
            "key 1 must survive the splice",
        );

        // Re-access the reprieved key. With one-access-queue pressure no
        // longer routed through eviction at all (see the module doc), the
        // ONLY thing that ever calls evict_one() -- and hence
        // check_slow_midpoint(), which is what actually checks this
        // reference bit -- is real max_size pressure. Force a real
        // eviction pass via a deterministic resize, not a filler set()
        // (same trigger the other main-queue-tail tests in this family
        // use), so the reference bit actually gets checked.
        cache.get(&1u32).expect("get should succeed");
        std::thread::sleep(std::time::Duration::from_millis(300));

        cache.resize(180).expect("resize should succeed");

        let promoted = wait_until(MIGRATION_TIMEOUT, || cache.tier_of(&1u32) == Some(Tier::Fast));
        assert!(promoted, "a reprieved key should still be promotable via the ordinary second chance");
        assert_eq!(cache.get(&1u32).unwrap(), b"first value 123");
    }

    #[test]
    fn one_access_pressure_alone_never_causes_a_terminal_eviction() {
        ensure_pmem_allocator_warm();

        // max_size comfortably larger than anything this test admits, so
        // the ONLY pressure driving anything is the tiny one-access
        // capacity -- if that pressure incorrectly triggered a terminal
        // eviction (the bug this variant's design doc explains was caught
        // and fixed), the evictions counter would move; it must not.
        let cache = PaperCache::<u32, TieredBuffer>::new(
            1_000_000,
            CacheTierSize::Bytes(1_000_000),
            0.00004,
        ).expect("cache should construct");

        for key in 1u32..=20 {
            cache.set(key, b"payload bytes A", None).expect("set should succeed");
        }

        // Give every reprieve a chance to settle.
        std::thread::sleep(std::time::Duration::from_millis(500));

        let stats = cache.s3_fifo_lazy_demotion_reprieve_hybrid_stats();
        assert_eq!(stats.evictions, 0, "one-access capacity pressure must only ever reprieve, never terminally evict");

        for key in 1u32..=20 {
            assert!(cache.has(&key), "key {key} should still be alive, just possibly reprieved to slow");
        }
    }

    // ── main-queue behavior (unaffected by ghost removal / reprieve) ───────

    #[test]
    fn a_plain_access_on_a_fast_main_queue_key_does_not_migrate_or_reorder() {
        ensure_pmem_allocator_warm();

        // The ratio only needs to be large enough that set() does not age the
        // key straight out of the one-access queue, so the get() below can
        // promote it. Note this variant does NOT subtract one_access_capacity
        // from fast_capacity (the one-access queue is PMEM and competes for
        // nothing) -- the main queue gets the whole 1_000_000, so the
        // promotion sticks regardless of the ratio chosen here.
        let cache = PaperCache::<u32, TieredBuffer>::new(
            1_000_000,
            CacheTierSize::Bytes(1_000_000),
            0.5,
        ).expect("cache should construct");

        cache.set(1u32, b"first value 123", None).expect("set should succeed");
        cache.get(&1u32).expect("get should succeed");
        // Asynchronous here, unlike the fast-admission variants: promotion out
        // of the one-access queue is a real PMEM->DRAM move applied by the
        // worker, not a key that was already sitting in DRAM.
        assert!(
            wait_until(MIGRATION_TIMEOUT, || cache.tier_of(&1u32) == Some(Tier::Fast)),
            "the get should have promoted key 1 into the main queue's fast segment",
        );

        let promotions_before = cache.s3_fifo_lazy_demotion_reprieve_hybrid_stats().promotions;

        cache.get(&1u32).expect("get should succeed");
        std::thread::sleep(std::time::Duration::from_millis(300));

        assert_eq!(cache.tier_of(&1u32), Some(Tier::Fast));
        assert_eq!(
            cache.s3_fifo_lazy_demotion_reprieve_hybrid_stats().promotions,
            promotions_before,
        );
    }

    #[test]
    fn an_accessed_key_at_the_main_queue_tail_gets_a_second_chance_instead_of_eviction() {
        ensure_pmem_allocator_warm();

        // one_access_capacity = 0.00004 * 1_000_000 = 40, comfortably above
        // one payload's stack-level size, so a set()+get() in immediate
        // succession promotes normally via touch() instead of racing
        // settle_one_access's synchronous reprieve.
        //
        // fast_capacity is stated directly as the main queue's budget (40 --
        // one key). This variant does NOT subtract one_access_capacity from
        // it the way the fast-admission variants do, because the one-access
        // queue is PMEM here and competes for nothing the main queue's fast
        // segment wants. Passing 80 would leave room for both keys and
        // nothing would demote.
        let cache = PaperCache::<u32, TieredBuffer>::new(
            1_000_000,
            CacheTierSize::Bytes(40),
            0.00004,
        ).expect("cache should construct");

        cache.set(1u32, b"payload bytes A", None).expect("set should succeed");
        cache.get(&1u32).expect("get should succeed");
        // Asynchronous here, unlike the fast-admission variants: promotion out
        // of the one-access queue is a real PMEM->DRAM move applied by the
        // worker, not a key that was already sitting in DRAM.
        assert!(
            wait_until(MIGRATION_TIMEOUT, || cache.tier_of(&1u32) == Some(Tier::Fast)),
            "the get should have promoted key 1 into the main queue's fast segment",
        );

        cache.set(2u32, b"payload bytes B", None).expect("set should succeed");
        cache.get(&2u32).expect("get should succeed");
        assert!(wait_until(MIGRATION_TIMEOUT, || cache.tier_of(&1u32) == Some(Tier::Slow)));

        cache.get(&1u32).expect("get should succeed");
        std::thread::sleep(std::time::Duration::from_millis(300));
        assert_eq!(cache.tier_of(&1u32), Some(Tier::Slow));

        // Deterministic trigger, not a filler set() -- see
        // s3_fifo_hybrid_cache_integration.rs's equivalent test for why.
        cache.resize(180).expect("resize should succeed");

        let survived_and_promoted = wait_until(MIGRATION_TIMEOUT, || {
            cache.has(&1u32) && cache.tier_of(&1u32) == Some(Tier::Fast)
        });
        assert!(
            survived_and_promoted,
            "key 1 should have been given a second chance and promoted back to fast",
        );
        assert_eq!(cache.get(&1u32).unwrap(), b"payload bytes A");
    }

    // ── the inherited signature mechanic: reprieve at DEMOTION time ────────

    #[test]
    fn an_accessed_fast_boundary_key_is_reprieved_at_demotion_time_instead_of_the_newcomer() {
        ensure_pmem_allocator_warm();

        // one_access_capacity = 0.00004 * 1_000_000 = 40, comfortably above
        // one payload's stack-level size, so a set()+get() in immediate
        // succession promotes normally via touch() instead of racing
        // settle_one_access's synchronous reprieve.
        //
        // fast_capacity is stated directly as the main queue's budget (40 --
        // one key). This variant does NOT subtract one_access_capacity from
        // it the way the fast-admission variants do, because the one-access
        // queue is PMEM here and competes for nothing the main queue's fast
        // segment wants. Passing 80 would leave room for both keys and
        // nothing would demote.
        let cache = PaperCache::<u32, TieredBuffer>::new(
            1_000_000,
            CacheTierSize::Bytes(40),
            0.00004,
        ).expect("cache should construct");

        cache.set(1u32, b"payload bytes A", None).expect("set should succeed");
        cache.get(&1u32).expect("get should succeed");
        // Asynchronous here, unlike the fast-admission variants: promotion out
        // of the one-access queue is a real PMEM->DRAM move applied by the
        // worker, not a key that was already sitting in DRAM.
        assert!(
            wait_until(MIGRATION_TIMEOUT, || cache.tier_of(&1u32) == Some(Tier::Fast)),
            "the get should have promoted key 1 into the main queue's fast segment",
        );

        let promotions_before = cache.s3_fifo_lazy_demotion_reprieve_hybrid_stats().promotions;
        let demotions_before = cache.s3_fifo_lazy_demotion_reprieve_hybrid_stats().demotions;

        cache.get(&1u32).expect("get should succeed");
        std::thread::sleep(std::time::Duration::from_millis(300));
        assert_eq!(cache.tier_of(&1u32), Some(Tier::Fast));

        cache.set(2u32, b"payload bytes B", None).expect("set should succeed");
        cache.get(&2u32).expect("get should succeed");

        // Wait on the DEMOTIONS COUNTER, not on `tier_of(&2)`. Key 2 reads
        // Slow from the moment it is admitted (the one-access queue is PMEM
        // here), so `wait_until(tier_of(2) == Slow)` returns instantly and
        // proves nothing -- it was letting this test read the stats before
        // the worker had processed key 2's access at all. The counter only
        // moves on a genuine main-queue demotion, which is the event under
        // test.
        let demoted = wait_until(
            MIGRATION_TIMEOUT,
            || cache.s3_fifo_lazy_demotion_reprieve_hybrid_stats().demotions == demotions_before + 1,
        );
        assert!(
            demoted,
            "key 2 should have been demoted in key 1's place (demotions {} -> {})",
            demotions_before,
            cache.s3_fifo_lazy_demotion_reprieve_hybrid_stats().demotions,
        );
        assert_eq!(cache.tier_of(&2u32), Some(Tier::Slow));

        std::thread::sleep(std::time::Duration::from_millis(300));
        assert_eq!(
            cache.tier_of(&1u32), Some(Tier::Fast),
            "key 1 should have been reprieved at the demotion boundary, not demoted",
        );

        let stats = cache.s3_fifo_lazy_demotion_reprieve_hybrid_stats();
        assert_eq!(stats.promotions, promotions_before, "reprieve must not count as a promotion");
        assert_eq!(stats.demotions, demotions_before + 1, "exactly key 2 should have been demoted");

        assert_eq!(cache.get(&1u32).unwrap(), b"payload bytes A");
        assert_eq!(cache.get(&2u32).unwrap(), b"payload bytes B");
    }

    #[test]
    fn an_unaccessed_key_deep_in_the_slow_segment_is_eventually_evicted() {
        ensure_pmem_allocator_warm();

        let cache = PaperCache::<u32, TieredBuffer>::new(
            2_000,
            CacheTierSize::Bytes(470),
            0.2,
        ).expect("cache should construct");

        for key in 1u32..=6 {
            cache.set(key, b"payload bytes A", None).expect("set should succeed");
            cache.get(&key).expect("get should succeed");
        }
        assert!(wait_until(MIGRATION_TIMEOUT, || cache.tier_of(&1u32) == Some(Tier::Slow)));

        // Key 1 is never reaccessed -- with no mid-tier checkpoint at all,
        // it simply ages to the tail and is evicted there.
        for key in 7u32..=30 {
            cache.set(key, b"payload bytes A", None).expect("set should succeed");
            cache.get(&key).expect("get should succeed");
        }

        let evicted = wait_until(MIGRATION_TIMEOUT, || !cache.has(&1u32));
        assert!(evicted, "an unaccessed key should eventually be evicted for real, not kept alive forever");
    }

    // ── TTL ───────────────────────────────────────────────────────────────

    const TTL_FAST_TIER: u64 = 200;

    #[test]
    fn ttl_survives_a_demotion() {
        ensure_pmem_allocator_warm();

        let cache = PaperCache::<u32, TieredBuffer>::new(
            1_000_000,
            CacheTierSize::Bytes(TTL_FAST_TIER),
            // Must actually hold the ttl'd key: at ratio 0.0 it is spliced
            // straight into the main queue and the get() below can no longer
            // promote it (a main-queue hit only sets the reference bit).
            0.5,
        ).expect("cache should construct");

        let ttl_secs = 5u32;
        let set_at = std::time::Instant::now();
        cache.set(1u32, b"first value 123", Some(ttl_secs)).expect("set should succeed");

        for key in 2u32..=6 {
            cache.set(key, b"filler bytes", None).expect("set should succeed");
        }

        cache.get(&1u32).expect("get should succeed");
        // Asynchronous here, unlike the fast-admission variants: promotion out
        // of the one-access queue is a real PMEM->DRAM move applied by the
        // worker, not a key that was already sitting in DRAM.
        assert!(
            wait_until(MIGRATION_TIMEOUT, || cache.tier_of(&1u32) == Some(Tier::Fast)),
            "the get should have promoted key 1 into the main queue's fast segment",
        );

        for key in 2u32..=6 {
            cache.get(&key).expect("get should succeed");
        }

        assert!(wait_until(MIGRATION_TIMEOUT, || cache.tier_of(&1u32) == Some(Tier::Slow)));
        assert!(cache.has(&1u32), "key should still be alive right after migrating");

        let remaining = std::time::Duration::from_millis(ttl_secs as u64 * 1000 + 500)
            .saturating_sub(set_at.elapsed());
        std::thread::sleep(remaining);

        assert!(matches!(cache.get(&1u32), Err(CacheError::KeyNotFound)));
        assert!(!cache.has(&1u32));
    }

    #[test]
    fn ttl_survives_a_reprieve() {
        ensure_pmem_allocator_warm();

        let cache = PaperCache::<u32, TieredBuffer>::new(
            1_000_000,
            CacheTierSize::Bytes(1_000_000),
            0.00004, // one_access_capacity = 40 bytes, fits one ~15-byte value
        ).expect("cache should construct");

        let ttl_secs = 5u32;
        let set_at = std::time::Instant::now();
        cache.set(1u32, b"first value 123", Some(ttl_secs)).expect("set should succeed");
        // Slow from the outset -- the one-access queue is PMEM in this variant.
        assert_eq!(cache.tier_of(&1u32), Some(Tier::Slow));

        // Pushes key 1 past the one-access budget, reprieving it into the main
        // queue's slow tier. Note `tier_of` cannot observe this transition
        // here: both the one-access queue and the main queue's slow segment
        // are PMEM, so the splice is slow->slow and moves no bytes (that is
        // the point of this design). What must hold is that the key survives
        // the splice with its TTL intact, which is what the rest of this test
        // checks -- still alive immediately after, and expiring on the
        // original schedule rather than being reset or dropped early.
        cache.set(2u32, b"second value 45", None).expect("set should succeed");
        assert_eq!(cache.tier_of(&1u32), Some(Tier::Slow));
        assert!(cache.has(&1u32), "key should still be alive right after the reprieve");

        let remaining = std::time::Duration::from_millis(ttl_secs as u64 * 1000 + 500)
            .saturating_sub(set_at.elapsed());
        std::thread::sleep(remaining);

        assert!(matches!(cache.get(&1u32), Err(CacheError::KeyNotFound)));
        assert!(!cache.has(&1u32));
    }

    // ── runtime resize ────────────────────────────────────────────────────

    #[test]
    fn set_fast_tier_size_takes_effect_at_runtime() {
        ensure_pmem_allocator_warm();

        let cache = PaperCache::<u32, TieredBuffer>::new(
            1_000_000,
            // Not 0.0: at that ratio the one-access queue holds nothing, so a
            // key is spliced straight into the main queue and a plain get()
            // can no longer promote it -- under lazy promotion a main-queue
            // hit only sets the reference bit. A ratio that actually holds the
            // key is what lets the get() below promote it to the fast tier.
            CacheTierSize::Bytes(1_000_000),
            0.5,
        ).expect("cache should construct");

        cache.set(1u32, b"first value 123", None).expect("set should succeed");
        cache.get(&1u32).expect("get should succeed");
        // Asynchronous here, unlike the fast-admission variants: promotion out
        // of the one-access queue is a real PMEM->DRAM move applied by the
        // worker, not a key that was already sitting in DRAM.
        assert!(
            wait_until(MIGRATION_TIMEOUT, || cache.tier_of(&1u32) == Some(Tier::Fast)),
            "the get should have promoted key 1 into the main queue's fast segment",
        );

        cache.set_fast_tier_size(CacheTierSize::Bytes(1)).expect("resize should succeed");

        let demoted = wait_until(MIGRATION_TIMEOUT, || cache.tier_of(&1u32) == Some(Tier::Slow));
        assert!(demoted, "shrinking the fast tier should demote the promoted key");
    }

    // ── edge cases ────────────────────────────────────────────────────────

    #[test]
    fn zero_fast_tier_size_is_rejected() {
        let result = PaperCache::<u32, TieredBuffer>::new(1_000, CacheTierSize::Bytes(0), 0.5);
        assert!(matches!(result, Err(CacheError::InvalidFastTierSize)));
    }

    #[test]
    fn invalid_one_access_ratio_is_rejected() {
        assert!(matches!(
            PaperCache::<u32, TieredBuffer>::new(1_000, CacheTierSize::Bytes(500), 1.5),
            Err(CacheError::InvalidPolicy),
        ));
    }

    #[test]
    fn del_removes_key_from_whichever_tier_it_is_in() {
        ensure_pmem_allocator_warm();

        let cache = PaperCache::<u32, TieredBuffer>::new(
            1_000_000,
            // Not 0.0: at that ratio the one-access queue holds nothing, so a
            // key is spliced straight into the main queue and a plain get()
            // can no longer promote it -- under lazy promotion a main-queue
            // hit only sets the reference bit. A ratio that actually holds the
            // key is what lets the get() below promote it to the fast tier.
            CacheTierSize::Bytes(1_000_000),
            0.5,
        ).expect("cache should construct");

        cache.set(1u32, b"first value 123", None).expect("set should succeed");
        cache.get(&1u32).expect("get should succeed");
        // Asynchronous here, unlike the fast-admission variants: promotion out
        // of the one-access queue is a real PMEM->DRAM move applied by the
        // worker, not a key that was already sitting in DRAM.
        assert!(
            wait_until(MIGRATION_TIMEOUT, || cache.tier_of(&1u32) == Some(Tier::Fast)),
            "the get should have promoted key 1 into the main queue's fast segment",
        );

        cache.del(&1u32).expect("del should succeed");
        assert!(!cache.has(&1u32));
        assert_eq!(cache.tier_of(&1u32), None);
    }

    #[test]
    fn wipe_clears_both_tiers() {
        ensure_pmem_allocator_warm();

        let cache = PaperCache::<u32, TieredBuffer>::new(
            1_000_000,
            // Not 0.0: at that ratio the one-access queue holds nothing, so a
            // key is spliced straight into the main queue and a plain get()
            // can no longer promote it -- under lazy promotion a main-queue
            // hit only sets the reference bit. A ratio that actually holds the
            // key is what lets the get() below promote it to the fast tier.
            CacheTierSize::Bytes(1_000_000),
            0.5,
        ).expect("cache should construct");

        cache.set(1u32, b"first value 123", None).expect("set should succeed");
        cache.set(2u32, b"second value 45", None).expect("set should succeed");
        cache.get(&1u32).expect("get should succeed");
        // Asynchronous here, unlike the fast-admission variants: promotion out
        // of the one-access queue is a real PMEM->DRAM move applied by the
        // worker, not a key that was already sitting in DRAM.
        assert!(
            wait_until(MIGRATION_TIMEOUT, || cache.tier_of(&1u32) == Some(Tier::Fast)),
            "the get should have promoted key 1 into the main queue's fast segment",
        );

        cache.wipe().expect("wipe should succeed");

        assert!(!cache.has(&1u32));
        assert!(!cache.has(&2u32));
        assert_eq!(cache.tier_of(&1u32), None);
        assert_eq!(cache.tier_of(&2u32), None);
    }
}
