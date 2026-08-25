/*
 * Copyright (c) Kia Shakiba
 *
 * This source code is licensed under the GNU AGPLv3 license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! Integration tests for the `s3_fifo_ghost_lazy_demotion_hybrid_cache`
//! feature.
//!
//! Run with nightly (required for `allocator_api` via `key_value_pmem`):
//!   cargo +nightly test --test s3_fifo_ghost_lazy_demotion_hybrid_cache_integration --features s3_fifo_ghost_lazy_demotion_hybrid_cache
//!
//! Same one-`PaperCache<K, TieredBuffer>` architecture, ghost-queue
//! lifecycle, and eviction-time second-chance mechanic as
//! `s3_fifo_ghost_hybrid_cache` — see that feature's integration test file
//! for the shared coverage; this file mirrors it end to end and adds one
//! test specific to the new behavior: a demotion-time reference-bit gate
//! (`an_accessed_fast_boundary_key_is_reprieved_at_demotion_time_instead_of_the_newcomer`).

#[cfg(feature = "s3_fifo_ghost_lazy_demotion_hybrid_cache")]
mod hybrid_cache_tests {
    use paper_cache::{PaperPolicy, PaperCache, TieredBuffer, CacheTierSize, Tier, CacheError};

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
        // 0.5, not 1.0: ratio 1.0 is now rejected by the parser and by
        // `PaperCache::new` for the whole s3-fifo family, since
        // `main_capacity` is `(1 - ratio) * max_size` and would be zero. The
        // warm-up admits one 4-byte value (24 accounted bytes); 0.5 * 1_000_000
        // = 500_000 bytes on each side dwarfs that, so the one-access queue is
        // still effectively unbounded and the main-queue gate is transparent.
        let cache = PaperCache::<u32, TieredBuffer>::new(1_000_000, CacheTierSize::Bytes(1_000_000), PaperPolicy::S3FifoGhostLazyDemotionHybrid(0.5))
            .expect("warm-up cache should construct");

        cache.set(0u32, b"warm", None).expect("warm-up set should succeed");
        assert_eq!(cache.tier_of(&0u32), Some(Tier::Slow));
    }

    const MIGRATION_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(10);

    #[test]
    fn admission_always_lands_in_slow_tier() {
        ensure_pmem_allocator_warm();

        // 0.5, not 1.0: 1.0 is now rejected, `main_capacity` being
        // `(1 - ratio) * max_size`. This fixture needs nothing but room for its
        // single "hello world" key (31 accounted bytes), and 0.5 * 1_000_000 =
        // 500_000 bytes per side dwarfs it, so the change is behaviour-neutral.
        let cache = PaperCache::<u32, TieredBuffer>::new(
            1_000_000,
            CacheTierSize::Bytes(1_000_000), PaperPolicy::S3FifoGhostLazyDemotionHybrid(0.5)).expect("cache should construct");

        cache.set(1u32, b"hello world", None).expect("set should succeed");

        assert_eq!(cache.tier_of(&1u32), Some(Tier::Slow));
        assert_eq!(cache.get(&1u32).unwrap(), b"hello world");
    }

    #[test]
    fn reaccessing_a_one_access_key_promotes_it_eagerly_to_fast_tier() {
        ensure_pmem_allocator_warm();

        // 0.5, not 1.0 (rejected now: `main_capacity` would be 0). One
        // 31-accounted-byte key moves one-access -> main here; both budgets are
        // 0.5 * 1_000_000 = 500_000 bytes, so neither queue is ever near full
        // and the eviction path is never entered at all.
        let cache = PaperCache::<u32, TieredBuffer>::new(
            1_000_000,
            CacheTierSize::Bytes(1_000_000), PaperPolicy::S3FifoGhostLazyDemotionHybrid(0.5)).expect("cache should construct");

        cache.set(1u32, b"hello world", None).expect("set should succeed");
        cache.get(&1u32).expect("get should succeed");

        let promoted = wait_until(MIGRATION_TIMEOUT, || cache.tier_of(&1u32) == Some(Tier::Fast));
        assert!(promoted, "key should have promoted to the fast tier after a re-access");

        let stats = cache.hybrid_stats();
        assert!(stats.promotions >= 1);
    }

    // ── ghost queue: unchanged from s3_fifo_ghost_hybrid_cache ─────────────

    #[test]
    fn a_key_that_ages_out_and_is_readmitted_lands_directly_in_fast_tier() {
        ensure_pmem_allocator_warm();

        let cache = PaperCache::<u32, TieredBuffer>::new(1_000_000, CacheTierSize::Bytes(1_000_000), PaperPolicy::S3FifoGhostLazyDemotionHybrid(0.00004)).expect("cache should construct");

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

        // 0.5, not 1.0 (rejected now: `main_capacity` would be 0). The single
        // "brand new value" key accounts for 35 bytes against a 500_000-byte
        // one-access budget, so it stays put exactly as before.
        let cache = PaperCache::<u32, TieredBuffer>::new(
            1_000_000,
            CacheTierSize::Bytes(1_000_000), PaperPolicy::S3FifoGhostLazyDemotionHybrid(0.5)).expect("cache should construct");

        cache.set(9u32, b"brand new value", None).expect("set should succeed");

        assert_eq!(cache.tier_of(&9u32), Some(Tier::Slow));
        assert_eq!(cache.get(&9u32).unwrap(), b"brand new value");
    }

    // ── main-queue behavior (unaffected by the ghost queue) ────────────────

    #[test]
    fn a_plain_access_on_a_fast_main_queue_key_does_not_migrate_or_reorder() {
        ensure_pmem_allocator_warm();

        // 0.5, not 1.0 (rejected now: `main_capacity` would be 0). One
        // 35-accounted-byte key sits in the main queue against a 0.5 * 1_000_000
        // = 500_000-byte main budget, so `main_is_full()` is never true and the
        // no-migration/no-reorder behaviour under test is untouched.
        let cache = PaperCache::<u32, TieredBuffer>::new(
            1_000_000,
            CacheTierSize::Bytes(1_000_000), PaperPolicy::S3FifoGhostLazyDemotionHybrid(0.5)).expect("cache should construct");

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

        // 0.5, not 1.0 (rejected now: `main_capacity` would be 0). The ratio is
        // load-bearing here because `cache.resize(180)` below recomputes both
        // budgets against the new max size: 0.5 * 180 = 90 bytes each. The two
        // promoted "payload bytes N" keys account for 35 bytes apiece, so the
        // main queue holds 70 < 90, `main_is_full()` stays false, and
        // `evict_one` falls through the (empty) one-access tail into the
        // main-queue sweep -- the same path it took before the gate existed.
        let cache = PaperCache::<u32, TieredBuffer>::new(
            1_000_000,
            CacheTierSize::Bytes(40), PaperPolicy::S3FifoGhostLazyDemotionHybrid(0.5)).expect("cache should construct");

        cache.set(1u32, b"payload bytes A", None).expect("set should succeed");
        cache.get(&1u32).expect("get should succeed");
        assert!(wait_until(MIGRATION_TIMEOUT, || cache.tier_of(&1u32) == Some(Tier::Fast)));

        cache.set(2u32, b"payload bytes B", None).expect("set should succeed");
        cache.get(&2u32).expect("get should succeed");
        assert!(wait_until(MIGRATION_TIMEOUT, || cache.tier_of(&2u32) == Some(Tier::Fast)));
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

    // ── the signature new mechanic: reprieve at DEMOTION time ───────────────

    #[test]
    fn an_accessed_fast_boundary_key_is_reprieved_at_demotion_time_instead_of_the_newcomer() {
        ensure_pmem_allocator_warm();

        // Fast tier fits comfortably one "payload bytes N"-sized value but
        // not two (same 40-byte budget the eviction-time second-chance test
        // above already relies on for "exactly one slot").
        //
        // 0.5, not 1.0 (rejected now: `main_capacity` would be 0). Nothing is
        // ever evicted in this test -- 2 x 35 accounted bytes against a
        // 1_000_000-byte max size -- so both 500_000-byte budgets are pure
        // headroom and only the demotion-time reprieve is exercised.
        let cache = PaperCache::<u32, TieredBuffer>::new(
            1_000_000,
            CacheTierSize::Bytes(40), PaperPolicy::S3FifoGhostLazyDemotionHybrid(0.5)).expect("cache should construct");

        cache.set(1u32, b"payload bytes A", None).expect("set should succeed");
        cache.get(&1u32).expect("get should succeed");
        assert!(wait_until(MIGRATION_TIMEOUT, || cache.tier_of(&1u32) == Some(Tier::Fast)));

        let promotions_before = cache.hybrid_stats().promotions;
        let demotions_before = cache.hybrid_stats().demotions;
        let slow_objects_before = cache.hybrid_stats().slow_objects;

        // Touch key 1 again while it's still Fast -- sets its reference bit,
        // no reorder, no migration (same lazy-bit convention proven by
        // `a_plain_access_on_a_fast_main_queue_key_does_not_migrate_or_reorder`
        // above).
        cache.get(&1u32).expect("get should succeed");
        std::thread::sleep(std::time::Duration::from_millis(300));
        assert_eq!(cache.tier_of(&1u32), Some(Tier::Fast));

        // Promoting key 2 forces fast-tier pressure. In
        // `s3_fifo_ghost_hybrid_cache` (unconditional demotion) this would
        // demote key 1. Here, key 1's bit is set, so it must be reprieved
        // (stay Fast) and key 2 -- the only other candidate, with a clear
        // bit -- must be demoted in its place instead.
        //
        // Note: `tier_of(&2u32)` is NOT a usable signal for "key 2 was
        // demoted" here -- admission for this design always builds a
        // brand-new key's bytes as Slow at the API layer to begin with (see
        // `hybrid_policy::admission_tier`), and key
        // 2's promotion-then-immediate-re-demotion happens entirely within
        // one worker batch, so its physical buffer never visibly passes
        // through Fast at all -- `tier_of(&2u32) == Some(Tier::Slow)` would
        // already be true the instant `set()` returns, well before the real
        // demotion this test means to observe. Wait on the demotion counter
        // itself instead, which only increments once the real
        // `settle_fast_tier` demotion has been physically applied.
        cache.set(2u32, b"payload bytes B", None).expect("set should succeed");
        cache.get(&2u32).expect("get should succeed");

        // Wait on the slow-tier GAUGE, not on the demotions counter. Since
        // the counters were narrowed to count only migrations that are
        // physically completed, this event does not move `demotions` at all:
        // key 2's bytes are built Slow at admission, and its promotion and
        // re-demotion happen inside one worker batch, so the buffer never
        // physically becomes Fast and the Fast->Slow demotion is declined as
        // already-in-tier. The gauge does move -- 0 slow objects to 1 -- and
        // it is the state being asserted rather than a proxy for it.
        let demoted = wait_until(MIGRATION_TIMEOUT, || {
            cache.hybrid_stats().slow_objects > slow_objects_before
        });
        assert!(demoted, "key 2 should have been demoted in key 1's place");

        // Confirm key 1 was never moved off the fast tier at any point
        // during this, and key 2 settled back to (still) Slow.
        std::thread::sleep(std::time::Duration::from_millis(300));
        assert_eq!(
            cache.tier_of(&1u32), Some(Tier::Fast),
            "key 1 should have been reprieved at the demotion boundary, not demoted",
        );
        assert_eq!(cache.tier_of(&2u32), Some(Tier::Slow));

        // A reprieve is not a promotion (key 1 was already Fast) and key 2's
        // own promotion-then-immediate-demotion nets out to exactly one
        // real demotion, zero new promotions.
        // Neither counter moves, and that is the correct outcome rather than
        // a missed event. The counters record migrations that were physically
        // COMPLETED. A reprieve copies nothing (key 1 was already Fast, and
        // stays), and key 2's demotion copies nothing either: its bytes were
        // built Slow at admission and its promotion never physically landed,
        // so the Fast->Slow migration is declined as already-in-tier. The
        // move that did happen is visible in the gauge, not the counters.
        let stats = cache.hybrid_stats();
        assert_eq!(stats.promotions, promotions_before, "a reprieve copies nothing, so it cannot count as a promotion");
        assert_eq!(
            stats.demotions, demotions_before,
            "key 2's bytes were already Slow, so its demotion moves nothing and must not be counted",
        );
        assert_eq!(
            stats.slow_objects, slow_objects_before + 1,
            "the demotion itself is real and must show up in the slow-tier gauge",
        );

        assert_eq!(cache.get(&1u32).unwrap(), b"payload bytes A");
        assert_eq!(cache.get(&2u32).unwrap(), b"payload bytes B");
    }

    // ── TTL ───────────────────────────────────────────────────────────────

    const TTL_FAST_TIER: u64 = 200;

    #[test]
    fn ttl_survives_a_demotion() {
        ensure_pmem_allocator_warm();

        // 0.5, not 1.0 (rejected now: `main_capacity` would be 0). The six keys
        // account for 35 + 5 * 32 = 195 bytes, all of which end up in the main
        // queue; against 0.5 * 1_000_000 = 500_000 bytes per side neither budget
        // is approached, so the fast-tier demotion this test turns on is driven
        // by `TTL_FAST_TIER` alone, exactly as before.
        let cache = PaperCache::<u32, TieredBuffer>::new(
            1_000_000,
            CacheTierSize::Bytes(TTL_FAST_TIER), PaperPolicy::S3FifoGhostLazyDemotionHybrid(0.5)).expect("cache should construct");

        let ttl_secs = 5u32;
        let set_at = std::time::Instant::now();
        cache.set(1u32, b"first value 123", Some(ttl_secs)).expect("set should succeed");

        for key in 2u32..=6 {
            cache.set(key, b"filler bytes", None).expect("set should succeed");
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

        // 0.5 over the SAME 200-byte max size, not 1.0. max_size has to stay
        // 200 because this fixture drives the global eviction trigger
        // (`used_size() > max_size`); only the split moves. 0.5 * 200 = 100
        // bytes for the one-access queue and 100 for the main queue. Key 1 is
        // the main queue's only resident (35 accounted bytes), so main_used =
        // 35 < 100, `main_is_full()` is false, and `evict_one` keeps preferring
        // the one-access tail -- which is the priority this test asserts. At
        // ratio 1.0 `main_capacity` would be 0, the main queue would read full
        // from the outset, and the sweep would evict key 1 instead.
        let cache = PaperCache::<u32, TieredBuffer>::new(
            200,
            CacheTierSize::Bytes(200), PaperPolicy::S3FifoGhostLazyDemotionHybrid(0.5)).expect("cache should construct");

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

        // 0.5, not 1.0 (rejected now: `main_capacity` would be 0). One
        // 35-accounted-byte key against two 500_000-byte budgets: only
        // `set_fast_tier_size` drives the demotion under test.
        let cache = PaperCache::<u32, TieredBuffer>::new(
            1_000_000,
            CacheTierSize::Bytes(1_000_000), PaperPolicy::S3FifoGhostLazyDemotionHybrid(0.5)).expect("cache should construct");

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
        let result = PaperCache::<u32, TieredBuffer>::new(1_000, CacheTierSize::Bytes(0), PaperPolicy::S3FifoGhostLazyDemotionHybrid(0.5));
        assert!(matches!(result, Err(CacheError::InvalidFastTierSize)));
    }

    #[test]
    fn invalid_one_access_ratio_is_rejected() {
        assert!(matches!(
            PaperCache::<u32, TieredBuffer>::new(1_000, CacheTierSize::Bytes(500), PaperPolicy::S3FifoGhostLazyDemotionHybrid(1.5)),
            Err(CacheError::InvalidPolicy),
        ));
    }

    #[test]
    fn del_removes_key_from_whichever_tier_it_is_in() {
        ensure_pmem_allocator_warm();

        // 0.5, not 1.0 (rejected now: `main_capacity` would be 0). One
        // 35-accounted-byte key against two 500_000-byte budgets, so `del` is
        // the only thing that ever removes it.
        let cache = PaperCache::<u32, TieredBuffer>::new(
            1_000_000,
            CacheTierSize::Bytes(1_000_000), PaperPolicy::S3FifoGhostLazyDemotionHybrid(0.5)).expect("cache should construct");

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

        // 0.5, not 1.0 (rejected now: `main_capacity` would be 0). Two keys,
        // 35 accounted bytes each, against two 500_000-byte budgets -- nothing
        // is evicted before `wipe`.
        let cache = PaperCache::<u32, TieredBuffer>::new(
            1_000_000,
            CacheTierSize::Bytes(1_000_000), PaperPolicy::S3FifoGhostLazyDemotionHybrid(0.5)).expect("cache should construct");

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
