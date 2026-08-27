/*
 * Copyright (c) Kia Shakiba
 *
 * This source code is licensed under the GNU AGPLv3 license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! Integration tests for the `s3_fifo_ghost_compact_hybrid_cache` feature.
//!
//! A deliberate near-copy of the baseline suite. This stack is a compaction of
//! that one and must be behaviourally indistinguishable from it, so it answers
//! the same behavioural questions rather than a reduced set.
//!
//! Unit tests cannot substitute: they drive the stack directly and never place
//! any bytes, so they cannot see a placement or validation bug. Porting these
//! suites to the first four conversions immediately found four real defects
//! that every unit and fidelity test had passed.
//!
//! Run with nightly (required for `allocator_api` via `key_value_pmem`):
//!   cargo +nightly test --test s3_fifo_ghost_compact_hybrid_cache_integration --features s3_fifo_ghost_compact_hybrid_cache
//!
//! Same one-`PaperCache<K, TieredBuffer>` architecture and admission/
//! demotion/promotion/eviction rules as `s3_fifo_hybrid_cache` — see that
//! feature's integration test file for the shared coverage (the
//! lazy/reference-bit second-chance mechanic, TTL survival, runtime
//! resize, edge cases); this file focuses on what's actually new here: the
//! ghost queue.

#[cfg(feature = "s3_fifo_ghost_compact_hybrid_cache")]
mod hybrid_cache_tests {
    use paper_cache::{PaperPolicy, PaperCache, TieredBuffer, CacheTierSize, Tier, CacheError};

    // ~1 KB values for the two fast-tier-pressure fixtures below. Those need a
    // fast tier that holds ONE object and not two, and only an object's
    // *migrating* bytes are charged to a tier -- the value buffer alone, since
    // the key and the expiry stay in DRAM whichever tier the value is in. A
    // 15-byte value therefore migrates ~16 bytes, making the window between
    // "one fits" and "two fit" a few bytes wide; at ~1 KB it is hundreds of
    // bytes wide. Same idiom as the `s3_fifo_hybrid_cache` suite this file is
    // the ghost variant of.
    //
    // Deliberately NOT applied file-wide, by function only:
    // `terminal_eviction_prefers_one_access_queue_over_main_queue` runs a
    // 256-byte TOTAL cache with nine fillers, and the ghost fixtures turn on
    // their own small-value byte budgets. Those already pass.
    const VALUE_LEN: usize = 1024;

    fn value(seed: u8) -> Vec<u8> {
        vec![seed; VALUE_LEN]
    }

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

    fn ensure_pmem_allocator_warm() {
        // Mechanics tests at toy scales: metadata reservation off (see
        // `get_hybrid_dram_shared_overhead`).
        unsafe { std::env::set_var("PAPER_DISABLE_SHARED_OVERHEAD", "1") };
        // 0.5 * 1_048_576 = 524_288 B for each of the one-access and main
        // queues -- both far past this one 24 B object. (Ratio 1.0 is
        // rejected now: it leaves the main queue 0 B, and `used >= max`
        // reads an empty 0-byte queue as full.)
        let cache = PaperCache::<u32, TieredBuffer>::new(1_048_576, CacheTierSize::Bytes(1_048_576), PaperPolicy::S3FifoGhostCompactHybrid(0.5))
            .expect("warm-up cache should construct");

        cache.set(0u32, b"warm", None).expect("warm-up set should succeed");
        assert_eq!(cache.tier_of(&0u32), Some(Tier::Slow));
    }

    const MIGRATION_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(10);

    #[test]
    fn admission_always_lands_in_slow_tier() {
        ensure_pmem_allocator_warm();

        // 0.5 * 1_048_576 = 524_288 B one-access / 524_288 B main; the
        // single 31 B object (11 B payload + 20 B key/expiry) is nowhere
        // near either budget.
        let cache = PaperCache::<u32, TieredBuffer>::new(
            1_048_576,
            CacheTierSize::Bytes(1_048_576), PaperPolicy::S3FifoGhostCompactHybrid(0.5)).expect("cache should construct");

        cache.set(1u32, b"hello world", None).expect("set should succeed");

        assert_eq!(cache.tier_of(&1u32), Some(Tier::Slow));
        assert_eq!(cache.get(&1u32).unwrap(), b"hello world");
    }

    #[test]
    fn reaccessing_a_one_access_key_promotes_it_eagerly_to_fast_tier() {
        ensure_pmem_allocator_warm();

        // 0.5 * 1_048_576 = 524_288 B per queue; the one 31 B object
        // crosses from the one-access queue to the main queue with both
        // budgets untouched.
        let cache = PaperCache::<u32, TieredBuffer>::new(
            1_048_576,
            CacheTierSize::Bytes(1_048_576), PaperPolicy::S3FifoGhostCompactHybrid(0.5)).expect("cache should construct");

        cache.set(1u32, b"hello world", None).expect("set should succeed");
        cache.get(&1u32).expect("get should succeed");

        let promoted = wait_until(MIGRATION_TIMEOUT, || cache.tier_of(&1u32) == Some(Tier::Fast));
        assert!(promoted, "key should have promoted to the fast tier after a re-access");

        let stats = cache.hybrid_stats();
        assert!(stats.promotions >= 1);
    }

    // ── ghost queue: the new behavior this feature adds ────────────────────

    #[test]
    fn a_key_that_ages_out_and_is_readmitted_lands_directly_in_fast_tier() {
        ensure_pmem_allocator_warm();

        // Everything here rests on the one-access queue holding exactly one
        // object, so that admitting the second ages the first out into the
        // ghost queue. 0.00002 * 1_048_576 = 20 bytes: one value (~16 migrating
        // bytes) fits and the second forces the eviction under test. It was
        // 0.00004 -> 41 bytes, which was one object's worth back when the queue
        // was charged the whole ~36-byte object; now only the migrating bytes
        // count, so 41 holds BOTH, nothing ever aged out, and the wait for
        // key 1 to disappear timed out.
        //
        // The small values stay: this fixture is about the ghost queue, and the
        // budget -- not the payload -- is the knob that drives it.
        let cache = PaperCache::<u32, TieredBuffer>::new(1_048_576, CacheTierSize::Bytes(1_048_576), PaperPolicy::S3FifoGhostCompactHybrid(0.00002)).expect("cache should construct");

        cache.set(1u32, b"first value 123", None).expect("set should succeed");
        assert_eq!(cache.tier_of(&1u32), Some(Tier::Slow));

        cache.set(2u32, b"second value 45", None).expect("set should succeed");

        let evicted = wait_until(MIGRATION_TIMEOUT, || !cache.has(&1u32));
        assert!(evicted, "key 1 should have aged out of the one-access queue");

        cache.set(1u32, b"first value 123", None).expect("re-set should succeed");

        let promoted_directly = wait_until(MIGRATION_TIMEOUT, || cache.tier_of(&1u32) == Some(Tier::Fast));
        assert!(
            promoted_directly,
            "a ghost-queue hit on re-admission should land directly in the fast tier, \
             got tier {:?}",
            cache.tier_of(&1u32),
        );
        assert_eq!(cache.get(&1u32).unwrap(), b"first value 123");
    }

    #[test]
    fn a_key_with_no_ghost_history_still_lands_in_the_one_access_queue_slow() {
        ensure_pmem_allocator_warm();

        // `admission_tier` answers `Slow` for *every* key absent from the
        // object map, ghost record or not, so the tier of a lone brand-new
        // key says nothing about the ghost queue -- which is all the old
        // version of this test asserted. What separates a ghost MISS from a
        // ghost HIT is what the worker does next: a hit is spliced into the
        // main queue and promoted, a miss goes into the one-access queue and
        // stays there. Both cases run through one cache below.
        //
        // The one-access queue gets 0.001 * 1_048_576 = 1_048 bytes: several
        // objects, so an admission never immediately self-evicts (contrast
        // the near-zero ratio in the aging test above, where one object
        // already exceeds the budget), but a burst of fat fillers ages its
        // tail out.
        let cache = PaperCache::<u32, TieredBuffer>::new(
            1_048_576,
            CacheTierSize::Bytes(1_048_576), PaperPolicy::S3FifoGhostCompactHybrid(0.001)).expect("cache should construct");

        // Fat on purpose: ten of these overflow the 1000-byte one-access
        // budget outright, so the aging below doesn't depend on this policy's
        // exact per-object accounting.
        const FILLER: &[u8; 200] = &[b'f'; 200];

        // Give key 1 -- and only key 1 -- a ghost record.
        cache.set(1u32, b"first value 123", None).expect("set should succeed");
        for filler in 100u32..110 {
            cache.set(filler, FILLER, None).expect("filler set should succeed");
        }
        assert!(
            wait_until(MIGRATION_TIMEOUT, || !cache.has(&1u32)),
            "key 1 should have aged out of the one-access queue into the ghost queue",
        );

        // Key 9 (never seen -> ghost MISS) is admitted BEFORE key 1 (aged out
        // above -> ghost HIT) so that waiting on key 1's placement also
        // proves key 9's has already been applied: the worker drains its
        // events, and `apply_tier_migrations` its batch, in order. That is
        // what makes the assertion about key 9 sound without a blind sleep.
        cache.set(9u32, b"brand new value", None).expect("set should succeed");
        cache.set(1u32, b"first value 123", None).expect("re-set should succeed");

        assert!(
            wait_until(MIGRATION_TIMEOUT, || cache.tier_of(&1u32) == Some(Tier::Fast)),
            "the ghost-hit key should have been spliced into the main queue and promoted",
        );
        assert_eq!(
            cache.tier_of(&9u32), Some(Tier::Slow),
            "a key with no ghost record must not be given the ghost-hit treatment",
        );

        // "One-access queue", not merely "slow tier", is the claim: key 9 is
        // subject to exactly the aging that produced key 1's ghost record,
        // while key 1 -- a main-queue resident now -- is immune to it.
        for filler in 110u32..120 {
            cache.set(filler, FILLER, None).expect("filler set should succeed");
        }
        assert!(
            wait_until(MIGRATION_TIMEOUT, || !cache.has(&9u32)),
            "the ghost-miss key should have aged out of the one-access queue",
        );

        assert!(cache.has(&1u32), "a main-queue key should survive one-access aging");
        assert_eq!(cache.tier_of(&1u32), Some(Tier::Fast));
        assert_eq!(cache.get(&1u32).unwrap(), b"first value 123");
    }

    // ── main-queue behavior (unaffected by the ghost queue) ────────────────

    #[test]
    fn a_plain_access_on_a_fast_main_queue_key_does_not_migrate_or_reorder() {
        ensure_pmem_allocator_warm();

        // 0.5 * 1_048_576 = 524_288 B per queue; the single 35 B object
        // sits far under the main budget `main_is_full` reads, so the
        // eviction gate never fires here.
        let cache = PaperCache::<u32, TieredBuffer>::new(
            1_048_576,
            CacheTierSize::Bytes(1_048_576), PaperPolicy::S3FifoGhostCompactHybrid(0.5)).expect("cache should construct");

        cache.set(1u32, b"first value 123", None).expect("set should succeed");
        cache.get(&1u32).expect("get should succeed");
        assert!(wait_until(MIGRATION_TIMEOUT, || cache.tier_of(&1u32) == Some(Tier::Fast)));

        let promotions_before = cache.hybrid_stats().promotions;

        cache.get(&1u32).expect("get should succeed");
        std::thread::sleep(std::time::Duration::from_millis(300));

        assert_eq!(cache.tier_of(&1u32), Some(Tier::Fast));
        assert_eq!(cache.hybrid_stats().promotions, promotions_before);
    }

    #[test]
    fn an_accessed_key_at_the_main_queue_tail_gets_a_second_chance_instead_of_eviction() {
        ensure_pmem_allocator_warm();

        // 0.5 * 1_048_576 = 524_288 B per queue -- and 700 B each after the
        // resize(1_400) below, under the ~2 KB the two promoted keys park in
        // the main queue, so `main_is_full()` is true and `evict_one` drops
        // straight into the main-queue CLOCK sweep. The one-access queue is
        // empty by then either way, so the resize's trigger is unchanged.
        //
        // 1_600 holds one ~1 KB value and not two, which is what makes key 1
        // demote once key 2 is promoted. It was Bytes(40) against 15-byte
        // values: only each object's ~16 migrating bytes are charged to the
        // tier, so BOTH fitted in 40, nothing ever demoted, and the wait for
        // key 1 to reach Slow timed out.
        let cache = PaperCache::<u32, TieredBuffer>::new(
            1_048_576,
            CacheTierSize::Bytes(1_600), PaperPolicy::S3FifoGhostCompactHybrid(0.5)).expect("cache should construct");

        cache.set(1u32, &value(0xA4), None).expect("set should succeed");
        cache.get(&1u32).expect("get should succeed");
        assert!(wait_until(MIGRATION_TIMEOUT, || cache.tier_of(&1u32) == Some(Tier::Fast)));

        cache.set(2u32, &value(0xB5), None).expect("set should succeed");
        cache.get(&2u32).expect("get should succeed");
        assert!(wait_until(MIGRATION_TIMEOUT, || cache.tier_of(&2u32) == Some(Tier::Fast)));
        assert!(wait_until(MIGRATION_TIMEOUT, || cache.tier_of(&1u32) == Some(Tier::Slow)));

        cache.get(&1u32).expect("get should succeed");
        std::thread::sleep(std::time::Duration::from_millis(300));
        assert_eq!(cache.tier_of(&1u32), Some(Tier::Slow));

        // Deterministic trigger, not a filler set() -- see
        // hybrid_cache_integration.rs's equivalent test for why.
        //
        // Scaled with the payloads, which it has to be: one ~1 KB object
        // accounts for ~1.1 KB (1_044 B of object plus 87 B of policy
        // overhead), so 1_400 holds one and evicts the other -- exactly one
        // eviction, which is the pressure the second chance has to be observed
        // against. Was 180, sized for 15-byte values; left there against 1 KB
        // values it would evict everything and there would be no surviving key
        // left to check.
        cache.resize(1_400).expect("resize should succeed");

        let survived_and_promoted = wait_until(MIGRATION_TIMEOUT, || {
            cache.has(&1u32) && cache.tier_of(&1u32) == Some(Tier::Fast)
        });
        assert!(
            survived_and_promoted,
            "key 1 should have been given a second chance and promoted back to fast",
        );
        assert_eq!(cache.get(&1u32).unwrap(), value(0xA4));
    }

    // ── TTL ───────────────────────────────────────────────────────────────

    // Holds one ~1 KB value but not two, so promoting the first filler demotes
    // key 1 -- the demotion this test needs the TTL to survive. Was 200 against
    // 15-byte values: each of the six objects migrates only ~16 bytes, 96 in
    // total, so the tier never filled and key 1 never left it.
    const TTL_FAST_TIER: u64 = 1_600;

    #[test]
    fn ttl_survives_a_demotion() {
        ensure_pmem_allocator_warm();

        // 0.5 * 1_048_576 = 524_288 B per queue; the six ~1 KB objects total
        // ~6 KB, so neither queue budget ever binds and only the
        // TTL_FAST_TIER-byte fast tier drives the demotion under test.
        let cache = PaperCache::<u32, TieredBuffer>::new(
            1_048_576,
            CacheTierSize::Bytes(TTL_FAST_TIER), PaperPolicy::S3FifoGhostCompactHybrid(0.5)).expect("cache should construct");

        let ttl_secs = 5u32;
        let set_at = std::time::Instant::now();
        cache.set(1u32, &value(0xA1), Some(ttl_secs)).expect("set should succeed");

        for key in 2u32..=6 {
            cache.set(key, &value(0xC3), None).expect("set should succeed");
        }

        cache.get(&1u32).expect("get should succeed");
        assert!(wait_until(MIGRATION_TIMEOUT, || cache.tier_of(&1u32) == Some(Tier::Fast)));

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

    // ── eviction priority ─────────────────────────────────────────────────

    #[test]
    fn terminal_eviction_prefers_one_access_queue_over_main_queue() {
        ensure_pmem_allocator_warm();

        // This fixture evicts off the GLOBAL trigger (`used_size >
        // max_size`), which also charges 87 B of policy overhead per
        // object: key 1 is 122 B on its own and any second object puts the
        // cache over 200 B. So max_size stays 200 and only the ratio moves.
        // 0.5 * 200 = 100 B one-access / 100 B main; key 1 occupies 35 B of
        // the main queue, so `main_is_full` stays false and `evict_one`
        // keeps preferring the one-access tail -- which is the claim below.
        // (Ratio 1.0 gives the main queue 0 B, i.e. permanently "full",
        // and would evict key 1 instead.)
        let cache = PaperCache::<u32, TieredBuffer>::new(
            256,
            CacheTierSize::Bytes(256), PaperPolicy::S3FifoGhostCompactHybrid(0.5)).expect("cache should construct");

        cache.set(1u32, b"payload bytes 1", None).expect("set should succeed");
        cache.get(&1u32).expect("get should succeed");
        assert!(wait_until(MIGRATION_TIMEOUT, || cache.tier_of(&1u32) == Some(Tier::Fast)));

        for key in 2u32..=10 {
            let _ = cache.set(key, b"payload bytes", None);
        }

        let evicted = wait_until(MIGRATION_TIMEOUT, || {
            cache.hybrid_stats().evictions >= 1
        });
        assert!(evicted, "at least one terminal eviction should have occurred");

        std::thread::sleep(std::time::Duration::from_millis(200));

        assert!(
            cache.has(&1u32),
            "the proven main-queue key should not be evicted while one-access objects remain",
        );
    }

    // ── runtime resize ────────────────────────────────────────────────────

    #[test]
    fn set_fast_tier_size_takes_effect_at_runtime() {
        ensure_pmem_allocator_warm();

        // 0.5 * 1_048_576 = 524_288 B per queue; the single 35 B object is
        // far under both, so only the fast-tier resize below moves it.
        let cache = PaperCache::<u32, TieredBuffer>::new(
            1_048_576,
            CacheTierSize::Bytes(1_048_576), PaperPolicy::S3FifoGhostCompactHybrid(0.5)).expect("cache should construct");

        cache.set(1u32, b"first value 123", None).expect("set should succeed");
        cache.get(&1u32).expect("get should succeed");
        assert!(wait_until(MIGRATION_TIMEOUT, || cache.tier_of(&1u32) == Some(Tier::Fast)));

        cache.set_fast_tier_size(CacheTierSize::Bytes(1)).expect("resize should succeed");

        let demoted = wait_until(MIGRATION_TIMEOUT, || cache.tier_of(&1u32) == Some(Tier::Slow));
        assert!(demoted, "shrinking the fast tier should demote the promoted key");
    }

    // ── edge cases ────────────────────────────────────────────────────────

    #[test]
    fn zero_fast_tier_size_is_rejected() {
        let result = PaperCache::<u32, TieredBuffer>::new(1_024, CacheTierSize::Bytes(0), PaperPolicy::S3FifoGhostCompactHybrid(0.5));
        assert!(matches!(result, Err(CacheError::InvalidFastTierSize)));
    }

    #[test]
    fn invalid_one_access_ratio_is_rejected() {
        assert!(matches!(
            PaperCache::<u32, TieredBuffer>::new(1_024, CacheTierSize::Bytes(512), PaperPolicy::S3FifoGhostCompactHybrid(1.5)),
            Err(CacheError::InvalidPolicy),
        ));
    }

    #[test]
    fn del_removes_key_from_whichever_tier_it_is_in() {
        ensure_pmem_allocator_warm();

        // 0.5 * 1_048_576 = 524_288 B per queue; the one 35 B object is
        // nowhere near either budget.
        let cache = PaperCache::<u32, TieredBuffer>::new(
            1_048_576,
            CacheTierSize::Bytes(1_048_576), PaperPolicy::S3FifoGhostCompactHybrid(0.5)).expect("cache should construct");

        cache.set(1u32, b"first value 123", None).expect("set should succeed");
        cache.get(&1u32).expect("get should succeed");
        assert!(wait_until(MIGRATION_TIMEOUT, || cache.tier_of(&1u32) == Some(Tier::Fast)));

        cache.del(&1u32).expect("del should succeed");
        assert!(!cache.has(&1u32));
        assert_eq!(cache.tier_of(&1u32), None);
    }

    #[test]
    fn wipe_clears_both_tiers() {
        ensure_pmem_allocator_warm();

        // 0.5 * 1_048_576 = 524_288 B per queue; the two 35 B objects are
        // nowhere near either budget.
        let cache = PaperCache::<u32, TieredBuffer>::new(
            1_048_576,
            CacheTierSize::Bytes(1_048_576), PaperPolicy::S3FifoGhostCompactHybrid(0.5)).expect("cache should construct");

        cache.set(1u32, b"first value 123", None).expect("set should succeed");
        cache.set(2u32, b"second value 45", None).expect("set should succeed");
        cache.get(&1u32).expect("get should succeed");
        assert!(wait_until(MIGRATION_TIMEOUT, || cache.tier_of(&1u32) == Some(Tier::Fast)));

        cache.wipe().expect("wipe should succeed");

        assert!(!cache.has(&1u32));
        assert!(!cache.has(&2u32));
        assert_eq!(cache.tier_of(&1u32), None);
        assert_eq!(cache.tier_of(&2u32), None);
    }
}
