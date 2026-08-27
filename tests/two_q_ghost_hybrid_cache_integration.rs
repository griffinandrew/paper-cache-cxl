/*
 * Copyright (c) Kia Shakiba
 *
 * This source code is licensed under the GNU AGPLv3 license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! Integration tests for the `two_q_ghost_hybrid_cache` feature.
//!
//! Run with nightly (required for `allocator_api` via `key_value_pmem`):
//!   cargo +nightly test --test two_q_ghost_hybrid_cache_integration --features two_q_ghost_hybrid_cache
//!
//! Same one-`PaperCache<K, TieredBuffer>` architecture and admission/
//! demotion/promotion/eviction rules as `two_q_hybrid_cache` — see that
//! feature's integration test file for the shared coverage (fifo-queue
//! demotion cascades, TTL survival, runtime resize, edge cases); this file
//! focuses on what's actually new here: the ghost queue.

#[cfg(feature = "two_q_ghost_hybrid_cache")]
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
        let cache = PaperCache::<u32, TieredBuffer>::new(1_048_576, CacheTierSize::Bytes(1_048_576), PaperPolicy::TwoQGhostHybrid(1.0))
            .expect("warm-up cache should construct");

        cache.set(0u32, b"warm", None).expect("warm-up set should succeed");
        assert_eq!(cache.tier_of(&0u32), Some(Tier::Slow));
    }

    const MIGRATION_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(10);

    #[test]
    fn admission_always_lands_in_slow_tier() {
        ensure_pmem_allocator_warm();

        let cache = PaperCache::<u32, TieredBuffer>::new(
            1_048_576,
            CacheTierSize::Bytes(1_048_576), PaperPolicy::TwoQGhostHybrid(1.0)).expect("cache should construct");

        cache.set(1u32, b"hello world", None).expect("set should succeed");

        assert_eq!(cache.tier_of(&1u32), Some(Tier::Slow));
        assert_eq!(cache.get(&1u32).unwrap(), b"hello world");
    }

    #[test]
    fn reaccessing_a_fifo_key_promotes_it_to_fast_tier() {
        ensure_pmem_allocator_warm();

        let cache = PaperCache::<u32, TieredBuffer>::new(
            1_048_576,
            CacheTierSize::Bytes(1_048_576), PaperPolicy::TwoQGhostHybrid(1.0)).expect("cache should construct");

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

        // FIFO capacity fits exactly one small value, so a second, distinct
        // key's admission evicts the first (into the ghost queue) before it
        // is ever re-accessed.
        //
        // k_in gives fifo_capacity = 0.00002 * 1_048_576 = 20 bytes, so one
        // object (~16 migrating bytes) fits and the second forces the ageing
        // out this test is about. It was 0.00004 -> 41 bytes, which holds
        // BOTH 15-byte values, so key 1 never aged out and never reached the
        // ghost queue -- there was no ghost hit left to observe.
        //
        // Small values are deliberate here and must stay: a FIFO budget this
        // tight is the whole source of the pressure, and a ~1 KB value would
        // not fit in it at all.
        let cache = PaperCache::<u32, TieredBuffer>::new(1_048_576, CacheTierSize::Bytes(1_048_576), PaperPolicy::TwoQGhostHybrid(0.00002)).expect("cache should construct");

        cache.set(1u32, b"first value 123", None).expect("set should succeed");
        assert_eq!(cache.tier_of(&1u32), Some(Tier::Slow));

        // Key 2's admission overflows fifo_capacity -> key 1 ages out,
        // unaccessed, into the ghost queue.
        cache.set(2u32, b"second value 45", None).expect("set should succeed");

        let evicted = wait_until(MIGRATION_TIMEOUT, || !cache.has(&1u32));
        assert!(evicted, "key 1 should have aged out of the FIFO queue");

        // Re-admitting key 1: a ghost-queue hit should land it directly in
        // the main queue's fast tier -- no second trip through the FIFO
        // queue needed.
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
    fn a_key_with_no_ghost_history_still_lands_in_the_fifo_queue_slow() {
        ensure_pmem_allocator_warm();

        let cache = PaperCache::<u32, TieredBuffer>::new(
            1_048_576,
            CacheTierSize::Bytes(1_048_576), PaperPolicy::TwoQGhostHybrid(1.0)).expect("cache should construct");

        // Key 9 has never been seen before -- no ghost entry to hit.
        cache.set(9u32, b"brand new value", None).expect("set should succeed");

        assert_eq!(cache.tier_of(&9u32), Some(Tier::Slow));
        assert_eq!(cache.get(&9u32).unwrap(), b"brand new value");
    }

    // ── main-queue behavior (unaffected by the ghost queue) ────────────────

    #[test]
    fn fast_tier_pressure_within_main_queue_demotes_lru_tail() {
        ensure_pmem_allocator_warm();

        let cache = PaperCache::<u32, TieredBuffer>::new(
            1_048_576,
            // 1_600 holds one ~1 KB value, not two. Was Bytes(40) with 15-byte
            // values, where TWO objects (~16 migrating bytes each, size minus
            // the key and expiry that never migrate) fit and nothing ever
            // demoted, so the wait below simply timed out.
            CacheTierSize::Bytes(1_600), PaperPolicy::TwoQGhostHybrid(1.0)).expect("cache should construct");

        cache.set(1u32, &value(0xA1), None).expect("set should succeed");
        cache.set(2u32, &value(0xB2), None).expect("set should succeed");

        cache.get(&1u32).expect("get should succeed");
        assert!(wait_until(MIGRATION_TIMEOUT, || cache.tier_of(&1u32) == Some(Tier::Fast)));

        // The fast tier only fits one ~1 KB value, so promoting key 2 must
        // demote key 1 back down.
        cache.get(&2u32).expect("get should succeed");
        let demoted = wait_until(MIGRATION_TIMEOUT, || cache.tier_of(&1u32) == Some(Tier::Slow));
        assert!(demoted, "key 1 should have been demoted once key 2 was promoted");
        assert_eq!(cache.tier_of(&2u32), Some(Tier::Fast));
    }

    // ── TTL ───────────────────────────────────────────────────────────────

    // ~1 KB payloads for the tests that need one object to fill the fast
    // tier on its own, now that the fast tier also carries per-object DRAM
    // metadata. At 1 KB that reservation is a small fraction and the budgets
    // have a wide margin. Same idiom as the `two_q_`, `lru_` and `lfu_`
    // suites.
    //
    // Deliberately NOT applied to every test in this file: the ghost-queue
    // ageing test runs a 20-byte FIFO budget and
    // `terminal_eviction_prefers_fifo_queue_over_main_queue` runs a 256-byte
    // TOTAL cache with nine fillers, so both need small values and both
    // already pass.
    const VALUE_LEN: usize = 1024;

    fn value(seed: u8) -> Vec<u8> {
        vec![seed; VALUE_LEN]
    }

    // Holds one ~1 KB value but not two, so promoting a filler demotes key 1.
    // Was 200, sized for 15-byte values: at ~16 migrating bytes per object a
    // dozen of them fit, so key 1 was never pushed back out of the fast tier.
    const TTL_FAST_TIER: u64 = 1_600;

    #[test]
    fn ttl_survives_a_demotion() {
        ensure_pmem_allocator_warm();

        let cache = PaperCache::<u32, TieredBuffer>::new(
            1_048_576,
            CacheTierSize::Bytes(TTL_FAST_TIER), PaperPolicy::TwoQGhostHybrid(1.0)).expect("cache should construct");

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
    fn terminal_eviction_prefers_fifo_queue_over_main_queue() {
        ensure_pmem_allocator_warm();

        let cache = PaperCache::<u32, TieredBuffer>::new(
            256,
            CacheTierSize::Bytes(256), PaperPolicy::TwoQGhostHybrid(1.0)).expect("cache should construct");

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
            "the proven main-queue key should not be evicted while FIFO objects remain",
        );
    }

    // ── runtime resize ────────────────────────────────────────────────────

    #[test]
    fn set_fast_tier_size_takes_effect_at_runtime() {
        ensure_pmem_allocator_warm();

        let cache = PaperCache::<u32, TieredBuffer>::new(
            1_048_576,
            CacheTierSize::Bytes(1_048_576), PaperPolicy::TwoQGhostHybrid(1.0)).expect("cache should construct");

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
        let result = PaperCache::<u32, TieredBuffer>::new(1_024, CacheTierSize::Bytes(0), PaperPolicy::TwoQGhostHybrid(0.5));
        assert!(matches!(result, Err(CacheError::InvalidFastTierSize)));
    }

    #[test]
    fn invalid_k_in_is_rejected() {
        assert!(matches!(
            PaperCache::<u32, TieredBuffer>::new(1_024, CacheTierSize::Bytes(512), PaperPolicy::TwoQGhostHybrid(1.5)),
            Err(CacheError::InvalidPolicy),
        ));
    }

    #[test]
    fn del_removes_key_from_whichever_tier_it_is_in() {
        ensure_pmem_allocator_warm();

        let cache = PaperCache::<u32, TieredBuffer>::new(
            1_048_576,
            CacheTierSize::Bytes(1_048_576), PaperPolicy::TwoQGhostHybrid(1.0)).expect("cache should construct");

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

        let cache = PaperCache::<u32, TieredBuffer>::new(
            1_048_576,
            CacheTierSize::Bytes(1_048_576), PaperPolicy::TwoQGhostHybrid(1.0)).expect("cache should construct");

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
