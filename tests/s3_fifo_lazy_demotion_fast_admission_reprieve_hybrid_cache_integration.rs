/*
 * Copyright (c) Kia Shakiba
 *
 * This source code is licensed under the GNU AGPLv3 license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! Integration tests for the
//! `s3_fifo_lazy_demotion_fast_admission_reprieve_hybrid_cache`
//! feature.
//!
//! Run with nightly (required for `allocator_api` via `key_value_pmem`):
//!   cargo +nightly test --test s3_fifo_lazy_demotion_fast_admission_reprieve_hybrid_cache_integration --features s3_fifo_lazy_demotion_fast_admission_reprieve_hybrid_cache
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

#[cfg(feature = "s3_fifo_lazy_demotion_fast_admission_reprieve_hybrid_cache")]
mod hybrid_cache_tests {
    use paper_cache::{PaperPolicy, PaperCache, TieredBuffer, CacheTierSize, Tier, CacheError};

    /// What these fixtures used to write as a `one_access_ratio` of 0.0: no
    /// meaningful one-access reservation, so an admitted key is reprieved
    /// straight into the main queue and `effective_main_fast_capacity`
    /// (`fast_capacity - one_access_capacity`) is the whole fast tier.
    ///
    /// A literal 0.0 is no longer constructible: `PaperCache::new` now rejects
    /// any s3-fifo ratio whose `ratio * max_size` truncates to zero bytes. At
    /// the `max_size` of 1_048_576 every use site below passes, this is the
    /// smallest ratio that clears that check -- 0.000001 * 1_048_576 = 1 byte,
    /// still far below one ~84-byte accounted object, so the one-access queue
    /// holds nothing exactly as at 0.0. The only arithmetic that moves is
    /// `effective_main_fast_capacity`, now one byte lower.
    const NO_ONE_ACCESS_RATIO: f64 = 0.000001;

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
    fn ensure_pmem_allocator_warm() {
        // Mechanics tests at toy scales: metadata reservation off (see
        // `get_hybrid_dram_shared_overhead`).
        unsafe { std::env::set_var("PAPER_DISABLE_SHARED_OVERHEAD", "1") };
        let cache = PaperCache::<u32, TieredBuffer>::new(1_048_576, CacheTierSize::Bytes(1), PaperPolicy::S3FifoLazyDemotionFastAdmissionReprieveHybrid(NO_ONE_ACCESS_RATIO))
            .expect("warm-up cache should construct");

        cache.set(0u32, b"warm", None).expect("warm-up set should succeed");
        cache.get(&0u32).expect("warm-up get should succeed");

        let demoted = wait_until(
            std::time::Duration::from_secs(90),
            || cache.tier_of(&0u32) == Some(Tier::Slow),
        );
        assert!(demoted, "warm-up key should have self-demoted to force a real PMEM allocation");
    }

    const MIGRATION_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(10);

    #[test]
    fn admission_always_lands_in_fast_tier() {
        ensure_pmem_allocator_warm();

        // 0.5, not 1.0: 1.0 is now rejected by the parser and by
        // `PaperCache::new` for the whole s3-fifo family. This is a REPRIEVE
        // variant, which has no main-queue budget at all, so the ratio only
        // sizes the one-access queue and that rejection is the *only* reason
        // 1.0 fails here. 0.5 * 1_048_576 = 524_288 bytes still holds this
        // test's single key many times over, so the change is behaviour-neutral.
        let cache = PaperCache::<u32, TieredBuffer>::new(
            1_048_576,
            CacheTierSize::Bytes(1_048_576), PaperPolicy::S3FifoLazyDemotionFastAdmissionReprieveHybrid(0.5)).expect("cache should construct");

        cache.set(1u32, b"hello world", None).expect("set should succeed");

        assert_eq!(cache.tier_of(&1u32), Some(Tier::Fast));
        assert_eq!(cache.get(&1u32).unwrap(), b"hello world");
    }

    // ── the signature new mechanic: reprieve instead of eviction ───────────

    #[test]
    fn a_key_that_ages_out_lands_directly_in_the_main_queues_slow_tier() {
        ensure_pmem_allocator_warm();

        let cache = PaperCache::<u32, TieredBuffer>::new(1_048_576, CacheTierSize::Bytes(1_048_576), PaperPolicy::S3FifoLazyDemotionFastAdmissionReprieveHybrid(0.00004)).expect("cache should construct");

        cache.set(1u32, b"first value 123", None).expect("set should succeed");
        assert_eq!(cache.tier_of(&1u32), Some(Tier::Fast));

        cache.set(2u32, b"second value 45", None).expect("set should succeed");

        // Unlike the ghost-queue predecessor (where key 1 would be evicted
        // here), key 1 must remain alive throughout, migrating straight to
        // Slow -- no eviction, no need to re-set it.
        let reprieved = wait_until(MIGRATION_TIMEOUT, || cache.tier_of(&1u32) == Some(Tier::Slow));
        assert!(reprieved, "key 1 should have been spliced into the slow tier, not evicted");
        assert!(cache.has(&1u32), "key 1 must still be a live entry");
        assert_eq!(cache.get(&1u32).unwrap(), b"first value 123");
    }

    #[test]
    fn a_reprieved_key_can_be_promoted_by_a_later_access() {
        ensure_pmem_allocator_warm();

        // max_size 4_000 at ratio 0.01, not 1_048_576 at 0.00004: the
        // one-access budget is `ratio * max_size` either way, and 0.01 * 4_000
        // is the same 40 bytes this fixture has always sized against. What
        // changed is `resize()`, which re-derives that budget against the NEW
        // size and now rejects a config that rounds it to zero -- 0.00004 * 180
        // is 0, so the resize below failed with `InvalidPolicy`; 0.01 * 180 is
        // 1 byte, which it accepts. 4_000 still dwarfs the two keys this test
        // admits, so the global `used_size() > max_size` trigger stays quiet
        // until the resize fires it, exactly as before.
        let cache = PaperCache::<u32, TieredBuffer>::new(
            4_096,
            CacheTierSize::Bytes(4_096), PaperPolicy::S3FifoLazyDemotionFastAdmissionReprieveHybrid(0.01)).expect("cache should construct");

        cache.set(1u32, b"first value 123", None).expect("set should succeed");
        cache.set(2u32, b"second value 45", None).expect("set should succeed");
        assert!(wait_until(MIGRATION_TIMEOUT, || cache.tier_of(&1u32) == Some(Tier::Slow)));

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
            1_048_576,
            CacheTierSize::Bytes(1_048_576), PaperPolicy::S3FifoLazyDemotionFastAdmissionReprieveHybrid(0.00004)).expect("cache should construct");

        for key in 1u32..=20 {
            cache.set(key, b"payload bytes A", None).expect("set should succeed");
        }

        // Give every reprieve a chance to settle.
        std::thread::sleep(std::time::Duration::from_millis(500));

        let stats = cache.hybrid_stats();
        assert_eq!(stats.evictions, 0, "one-access capacity pressure must only ever reprieve, never terminally evict");

        for key in 1u32..=20 {
            assert!(cache.has(&key), "key {key} should still be alive, just possibly reprieved to slow");
        }
    }

    // ── main-queue behavior (unaffected by ghost removal / reprieve) ───────

    #[test]
    fn a_plain_access_on_a_fast_main_queue_key_does_not_migrate_or_reorder() {
        ensure_pmem_allocator_warm();

        // `one_access_ratio` must leave the MAIN queue real fast-tier room:
        // `effective_main_fast_capacity` is `fast_capacity - one_access_capacity`,
        // so a ratio of 1.0 (with fast_tier_size == max_size) zeroes it out and
        // `promote_from_one_access` demotes the key straight back to slow inside
        // the very same worker event. This test is about a key sitting *in* the
        // main queue's fast segment, so it needs a ratio that leaves headroom on
        // both sides: 0.5 gives the one-access queue 524_288 bytes (far more than
        // one payload, so set() never ages it out) and leaves the main queue the
        // other 524_288 (so the first get()'s promotion sticks).
        let cache = PaperCache::<u32, TieredBuffer>::new(
            1_048_576,
            CacheTierSize::Bytes(1_048_576), PaperPolicy::S3FifoLazyDemotionFastAdmissionReprieveHybrid(0.5)).expect("cache should construct");

        cache.set(1u32, b"first value 123", None).expect("set should succeed");
        cache.get(&1u32).expect("get should succeed");
        assert_eq!(cache.tier_of(&1u32), Some(Tier::Fast));

        let promotions_before = cache.hybrid_stats().promotions;

        cache.get(&1u32).expect("get should succeed");
        std::thread::sleep(std::time::Duration::from_millis(300));

        assert_eq!(cache.tier_of(&1u32), Some(Tier::Fast));
        assert_eq!(
            cache.hybrid_stats().promotions,
            promotions_before,
        );
    }

    #[test]
    fn an_accessed_key_at_the_main_queue_tail_gets_a_second_chance_instead_of_eviction() {
        ensure_pmem_allocator_warm();

        // one_access_capacity = 0.01 * 4_000 = 40, comfortably above
        // one payload's stack-level size, so a set()+get() in immediate
        // succession promotes normally via touch() instead of racing
        // settle_one_access's synchronous reprieve (which would otherwise
        // fire from within the set() event itself, before the get() event
        // is even processed -- see the module doc). fast_capacity is
        // bumped by that same 40 so the MAIN queue's effective budget
        // (fast_capacity - one_access_capacity) stays 40, matching the
        // dynamics this test was originally built around.
        // max_size 4_000 at ratio 0.01, not 1_048_576 at 0.00004: the
        // one-access budget is `ratio * max_size` either way, and 0.01 * 4_000
        // is the same 40 bytes this fixture has always sized against. What
        // changed is `resize()`, which re-derives that budget against the NEW
        // size and now rejects a config that rounds it to zero -- 0.00004 * 180
        // is 0, so the resize below failed with `InvalidPolicy`; 0.01 * 180 is
        // 1 byte, which it accepts. 4_000 still dwarfs the two keys this test
        // admits, so the global `used_size() > max_size` trigger stays quiet
        // until the resize fires it, exactly as before.
        let cache = PaperCache::<u32, TieredBuffer>::new(
            4_096,
            CacheTierSize::Bytes(80), PaperPolicy::S3FifoLazyDemotionFastAdmissionReprieveHybrid(0.01)).expect("cache should construct");

        cache.set(1u32, b"payload bytes A", None).expect("set should succeed");
        cache.get(&1u32).expect("get should succeed");
        assert_eq!(cache.tier_of(&1u32), Some(Tier::Fast));

        cache.set(2u32, b"payload bytes B", None).expect("set should succeed");
        cache.get(&2u32).expect("get should succeed");
        assert!(wait_until(MIGRATION_TIMEOUT, || cache.tier_of(&1u32) == Some(Tier::Slow)));

        cache.get(&1u32).expect("get should succeed");
        std::thread::sleep(std::time::Duration::from_millis(300));
        assert_eq!(cache.tier_of(&1u32), Some(Tier::Slow));

        // Deterministic trigger, not a filler set() -- see
        // hybrid_cache_integration.rs's equivalent test for why.
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

        // one_access_capacity = 0.00004 * 1_048_576 = 41, comfortably above
        // one payload's stack-level size, so a set()+get() in immediate
        // succession promotes normally via touch() instead of racing
        // settle_one_access's synchronous reprieve (which would otherwise
        // fire from within the set() event itself, before the get() event
        // is even processed -- see the module doc). fast_capacity is
        // bumped by that same 40 so the MAIN queue's effective budget
        // (fast_capacity - one_access_capacity) stays 40, matching the
        // dynamics this test was originally built around.
        let cache = PaperCache::<u32, TieredBuffer>::new(
            1_048_576,
            CacheTierSize::Bytes(80), PaperPolicy::S3FifoLazyDemotionFastAdmissionReprieveHybrid(0.00004)).expect("cache should construct");

        cache.set(1u32, b"payload bytes A", None).expect("set should succeed");
        cache.get(&1u32).expect("get should succeed");
        assert_eq!(cache.tier_of(&1u32), Some(Tier::Fast));

        let promotions_before = cache.hybrid_stats().promotions;
        let demotions_before = cache.hybrid_stats().demotions;

        cache.get(&1u32).expect("get should succeed");
        std::thread::sleep(std::time::Duration::from_millis(300));
        assert_eq!(cache.tier_of(&1u32), Some(Tier::Fast));

        cache.set(2u32, b"payload bytes B", None).expect("set should succeed");
        cache.get(&2u32).expect("get should succeed");

        let demoted = wait_until(MIGRATION_TIMEOUT, || cache.tier_of(&2u32) == Some(Tier::Slow));
        assert!(demoted, "key 2 should have been demoted in key 1's place, got tier {:?}", cache.tier_of(&2u32));

        std::thread::sleep(std::time::Duration::from_millis(300));
        assert_eq!(
            cache.tier_of(&1u32), Some(Tier::Fast),
            "key 1 should have been reprieved at the demotion boundary, not demoted",
        );

        let stats = cache.hybrid_stats();
        assert_eq!(stats.promotions, promotions_before, "reprieve must not count as a promotion");
        assert_eq!(stats.demotions, demotions_before + 1, "exactly key 2 should have been demoted");

        assert_eq!(cache.get(&1u32).unwrap(), b"payload bytes A");
        assert_eq!(cache.get(&2u32).unwrap(), b"payload bytes B");
    }

    #[test]
    fn an_unaccessed_key_deep_in_the_slow_segment_is_eventually_evicted() {
        ensure_pmem_allocator_warm();

        let cache = PaperCache::<u32, TieredBuffer>::new(
            2_048,
            CacheTierSize::Bytes(470), PaperPolicy::S3FifoLazyDemotionFastAdmissionReprieveHybrid(0.2)).expect("cache should construct");

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
            1_048_576,
            CacheTierSize::Bytes(TTL_FAST_TIER), PaperPolicy::S3FifoLazyDemotionFastAdmissionReprieveHybrid(NO_ONE_ACCESS_RATIO)).expect("cache should construct");

        let ttl_secs = 5u32;
        let set_at = std::time::Instant::now();
        cache.set(1u32, b"first value 123", Some(ttl_secs)).expect("set should succeed");

        for key in 2u32..=6 {
            cache.set(key, b"filler bytes", None).expect("set should succeed");
        }

        cache.get(&1u32).expect("get should succeed");
        assert_eq!(cache.tier_of(&1u32), Some(Tier::Fast));

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

        let cache = PaperCache::<u32, TieredBuffer>::new(1_048_576, CacheTierSize::Bytes(1_048_576), PaperPolicy::S3FifoLazyDemotionFastAdmissionReprieveHybrid(0.00004)).expect("cache should construct");

        let ttl_secs = 5u32;
        let set_at = std::time::Instant::now();
        cache.set(1u32, b"first value 123", Some(ttl_secs)).expect("set should succeed");
        assert_eq!(cache.tier_of(&1u32), Some(Tier::Fast));

        // Pushes key 1 past the one-access budget, reprieving it into the
        // main queue's slow tier -- must not disturb its TTL.
        cache.set(2u32, b"second value 45", None).expect("set should succeed");
        assert!(wait_until(MIGRATION_TIMEOUT, || cache.tier_of(&1u32) == Some(Tier::Slow)));
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
            1_048_576,
            CacheTierSize::Bytes(1_048_576), PaperPolicy::S3FifoLazyDemotionFastAdmissionReprieveHybrid(NO_ONE_ACCESS_RATIO)).expect("cache should construct");

        cache.set(1u32, b"first value 123", None).expect("set should succeed");
        cache.get(&1u32).expect("get should succeed");
        assert_eq!(cache.tier_of(&1u32), Some(Tier::Fast));

        cache.set_fast_tier_size(CacheTierSize::Bytes(1)).expect("resize should succeed");

        let demoted = wait_until(MIGRATION_TIMEOUT, || cache.tier_of(&1u32) == Some(Tier::Slow));
        assert!(demoted, "shrinking the fast tier should demote the promoted key");
    }

    // ── edge cases ────────────────────────────────────────────────────────

    #[test]
    fn zero_fast_tier_size_is_rejected() {
        let result = PaperCache::<u32, TieredBuffer>::new(1_024, CacheTierSize::Bytes(0), PaperPolicy::S3FifoLazyDemotionFastAdmissionReprieveHybrid(0.5));
        assert!(matches!(result, Err(CacheError::InvalidFastTierSize)));
    }

    #[test]
    fn invalid_one_access_ratio_is_rejected() {
        assert!(matches!(
            PaperCache::<u32, TieredBuffer>::new(1_024, CacheTierSize::Bytes(512), PaperPolicy::S3FifoLazyDemotionFastAdmissionReprieveHybrid(1.5)),
            Err(CacheError::InvalidPolicy),
        ));
    }

    #[test]
    fn del_removes_key_from_whichever_tier_it_is_in() {
        ensure_pmem_allocator_warm();

        let cache = PaperCache::<u32, TieredBuffer>::new(
            1_048_576,
            CacheTierSize::Bytes(1_048_576), PaperPolicy::S3FifoLazyDemotionFastAdmissionReprieveHybrid(NO_ONE_ACCESS_RATIO)).expect("cache should construct");

        cache.set(1u32, b"first value 123", None).expect("set should succeed");
        cache.get(&1u32).expect("get should succeed");
        assert_eq!(cache.tier_of(&1u32), Some(Tier::Fast));

        cache.del(&1u32).expect("del should succeed");
        assert!(!cache.has(&1u32));
        assert_eq!(cache.tier_of(&1u32), None);
    }

    #[test]
    fn wipe_clears_both_tiers() {
        ensure_pmem_allocator_warm();

        let cache = PaperCache::<u32, TieredBuffer>::new(
            1_048_576,
            CacheTierSize::Bytes(1_048_576), PaperPolicy::S3FifoLazyDemotionFastAdmissionReprieveHybrid(NO_ONE_ACCESS_RATIO)).expect("cache should construct");

        cache.set(1u32, b"first value 123", None).expect("set should succeed");
        cache.set(2u32, b"second value 45", None).expect("set should succeed");
        cache.get(&1u32).expect("get should succeed");
        assert_eq!(cache.tier_of(&1u32), Some(Tier::Fast));

        cache.wipe().expect("wipe should succeed");

        assert!(!cache.has(&1u32));
        assert!(!cache.has(&2u32));
        assert_eq!(cache.tier_of(&1u32), None);
        assert_eq!(cache.tier_of(&2u32), None);
    }
}
