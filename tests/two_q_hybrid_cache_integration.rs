/*
 * Copyright (c) Kia Shakiba
 *
 * This source code is licensed under the GNU AGPLv3 license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! Integration tests for the `two_q_hybrid_cache` feature.
//!
//! Run with nightly (required for `allocator_api` via `key_value_pmem`):
//!   cargo +nightly test --test two_q_hybrid_cache_integration --features two_q_hybrid_cache
//!
//! Same one-`PaperCache<K, TieredBuffer>` architecture as
//! `hybrid_cache_integration.rs`/`hybrid_cache_integration.rs` —
//! `tier_of` reads the tier directly off the single object map.
//!
//! The biggest structural difference from the other two hybrids: admission
//! always lands in the **slow** tier here (the one-access FIFO queue), not
//! the fast tier — `PaperCache::set()` builds `TieredBuffer::new_slow`
//! directly, a real synchronous PMEM write, for every single `set()` call.
//! This means, unlike the other two integration test files, **every** test
//! here pays (or waits out) the one-time PMEM pool warm-up cost — there is
//! no "fast-tier-only path" that avoids touching PMEM at all. See
//! `hybrid_cache_integration.rs`'s module doc for the ~45s warm-up
//! caveat itself; `ensure_pmem_allocator_warm` below is the same pattern.
//!
//! What is tested:
//!   * Admission always lands in the slow tier (the FIFO queue)
//!   * A re-access to a FIFO-queue object promotes it straight to the main
//!     queue's fast tier
//!   * A FIFO object that ages out without a second access is evicted
//!     (both via `k_in`-driven FIFO-capacity pressure and via the global
//!     capacity-exhausted eviction loop)
//!   * Once in the main queue, an object behaves like `lru_hybrid_cache`:
//!     fast-tier pressure demotes the LRU tail; a slow-tier access promotes
//!     it back, possibly cascading a further demotion
//!   * Terminal eviction prefers the FIFO queue over the main queue
//!   * TTL set before a demotion/promotion is still correctly enforced after
//!   * `set_fast_tier_size` / `resize` (which rescales `k_in`'s FIFO budget)
//!     take effect at runtime
//!   * Zero/invalid/tiny fast-tier-size and `k_in` edge cases

#[cfg(feature = "two_q_hybrid_cache")]
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

    /// Forces the one-time PMEM allocator pool init/prewarm to complete
    /// before a test's own timing-sensitive assertions begin. Unlike the
    /// other two hybrids' analog, the very first `set()` call here already
    /// pays this cost synchronously (admission itself calls
    /// `TieredBuffer::new_slow` directly, not via the async worker), so this
    /// helper doesn't need `wait_until` at all -- by the time `set()`
    /// returns, the allocator is warm. Kept as a named helper anyway for
    /// symmetry with the other two integration test files and to make every
    /// test's intent explicit.
    fn ensure_pmem_allocator_warm() {
        // Mechanics tests at toy scales: metadata reservation off (see
        // `get_hybrid_dram_shared_overhead`).
        unsafe { std::env::set_var("PAPER_DISABLE_SHARED_OVERHEAD", "1") };
        let cache = PaperCache::<u32, TieredBuffer>::new(1_048_576, CacheTierSize::Bytes(1_048_576), PaperPolicy::TwoQHybrid(1.0))
            .expect("warm-up cache should construct");

        cache.set(0u32, b"warm", None).expect("warm-up set should succeed");
        assert_eq!(cache.tier_of(&0u32), Some(Tier::Slow));
    }

    const MIGRATION_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(10);

    // ── admission ─────────────────────────────────────────────────────────

    #[test]
    fn admission_always_lands_in_slow_tier() {
        ensure_pmem_allocator_warm();

        let cache = PaperCache::<u32, TieredBuffer>::new(
            1_048_576,
            CacheTierSize::Bytes(1_048_576), PaperPolicy::TwoQHybrid(1.0)).expect("cache should construct");

        cache.set(1u32, b"hello world", None).expect("set should succeed");

        // Admission is synchronous (TieredBuffer::new_slow is built
        // directly inside set(), before the WorkerEvent is even
        // broadcast), so this doesn't need wait_until.
        assert_eq!(cache.tier_of(&1u32), Some(Tier::Slow));
        assert_eq!(cache.get(&1u32).unwrap(), b"hello world");
    }

    // ── promotion ─────────────────────────────────────────────────────────

    #[test]
    fn reaccessing_a_fifo_key_promotes_it_to_fast_tier() {
        ensure_pmem_allocator_warm();

        let cache = PaperCache::<u32, TieredBuffer>::new(
            1_048_576,
            CacheTierSize::Bytes(1_048_576), PaperPolicy::TwoQHybrid(1.0)).expect("cache should construct");

        cache.set(1u32, b"hello world", None).expect("set should succeed");
        assert_eq!(cache.tier_of(&1u32), Some(Tier::Slow));

        cache.get(&1u32).expect("get should succeed");

        let promoted = wait_until(MIGRATION_TIMEOUT, || {
            cache.tier_of(&1u32) == Some(Tier::Fast)
        });
        assert!(promoted, "key should have promoted to the fast tier after a re-access");

        let stats = cache.hybrid_stats();
        assert!(stats.promotions >= 1);
    }

    // ── FIFO eviction ─────────────────────────────────────────────────────

    #[test]
    fn fifo_key_aging_out_without_reaccess_is_evicted() {
        ensure_pmem_allocator_warm();

        let cache = PaperCache::<u32, TieredBuffer>::new(
            256,
            CacheTierSize::Bytes(256), PaperPolicy::TwoQHybrid(1.0)).expect("cache should construct");

        cache.set(1u32, b"first value 123", None).expect("set should succeed");
        assert_eq!(cache.tier_of(&1u32), Some(Tier::Slow));

        // Never re-accessed; admitting more keys should eventually evict it
        // via the global capacity-exhausted eviction loop.
        for key in 2u32..=10 {
            let _ = cache.set(key, b"payload bytes", None);
        }

        let evicted = wait_until(MIGRATION_TIMEOUT, || !cache.has(&1u32));
        assert!(evicted, "key 1 should have aged out of the FIFO queue and been evicted");
        assert_eq!(cache.tier_of(&1u32), None);

        let stats = cache.hybrid_stats();
        assert!(stats.evictions >= 1);
    }

    #[test]
    fn fifo_capacity_pressure_evicts_before_global_max_size_is_reached() {
        ensure_pmem_allocator_warm();

        // Overall cache is huge, but k_in caps the FIFO queue's own byte
        // budget tightly (0.00004 * 1_048_576 = 41 bytes, fitting one
        // ~15-byte value) -- eviction should trigger from fifo_capacity
        // pressure alone, nowhere near the global max_size.
        let cache = PaperCache::<u32, TieredBuffer>::new(
            1_048_576,
            // k_in gives fifo_capacity = 0.00002 * 1_048_576 = 20 bytes, so one
            // object (~16 migrating bytes) fits and the second forces the
            // eviction this test is about. It was 0.00004 -> 41 bytes, which
            // holds BOTH, so nothing was ever evicted.
            //
            // Small values are deliberate here and must stay: the assertion
            // below is that overall usage is still nowhere near max_size, which
            // ~1 KB values would break.
            CacheTierSize::Bytes(1_048_576), PaperPolicy::TwoQHybrid(0.00002)).expect("cache should construct");

        cache.set(1u32, b"first value 123", None).expect("set should succeed");
        assert_eq!(cache.tier_of(&1u32), Some(Tier::Slow));

        cache.set(2u32, b"second value 45", None).expect("set should succeed");

        let evicted = wait_until(MIGRATION_TIMEOUT, || !cache.has(&1u32));
        assert!(evicted, "key 1 should have been evicted by fifo_capacity pressure");

        let status = cache.status().unwrap();
        assert!(
            status.used_size() < 1_000,
            "overall usage should be nowhere near max_size, confirming this was \
             fifo_capacity pressure, not the global eviction loop",
        );
    }

    // ── main-queue behavior (mirrors lru_hybrid_cache once promoted) ──────

    #[test]
    fn fast_tier_pressure_within_main_queue_demotes_lru_tail() {
        ensure_pmem_allocator_warm();

        let cache = PaperCache::<u32, TieredBuffer>::new(
            1_048_576,
            // 1_600 holds one ~1 KB value, not two. Was Bytes(40) with 15-byte
            // values, where TWO objects (~16 migrating bytes each) fit and
            // nothing ever demoted.
            CacheTierSize::Bytes(1_600), PaperPolicy::TwoQHybrid(1.0)).expect("cache should construct");

        cache.set(1u32, &value(0xA1), None).expect("set should succeed");
        cache.set(2u32, &value(0xB2), None).expect("set should succeed");
        assert_eq!(cache.tier_of(&1u32), Some(Tier::Slow));
        assert_eq!(cache.tier_of(&2u32), Some(Tier::Slow));

        cache.get(&1u32).expect("get should succeed");
        assert!(wait_until(MIGRATION_TIMEOUT, || cache.tier_of(&1u32) == Some(Tier::Fast)));

        // Fast tier only fits one ~15-byte value, so promoting key 2 must
        // demote key 1 back down.
        cache.get(&2u32).expect("get should succeed");
        let demoted = wait_until(MIGRATION_TIMEOUT, || cache.tier_of(&1u32) == Some(Tier::Slow));
        assert!(demoted, "key 1 should have been demoted once key 2 was promoted");
        assert_eq!(cache.tier_of(&2u32), Some(Tier::Fast));
    }

    #[test]
    fn promotion_can_cascade_a_demotion_and_values_survive() {
        ensure_pmem_allocator_warm();

        let cache = PaperCache::<u32, TieredBuffer>::new(
            1_048_576,
            // 1_600 holds one ~1 KB value, not two. Was Bytes(40) with 15-byte
            // values, where TWO objects (~16 migrating bytes each) fit and
            // nothing ever demoted.
            CacheTierSize::Bytes(1_600), PaperPolicy::TwoQHybrid(1.0)).expect("cache should construct");

        cache.set(1u32, &value(0xA1), None).expect("set should succeed");
        cache.set(2u32, &value(0xB2), None).expect("set should succeed");

        cache.get(&1u32).expect("get should succeed");
        assert!(wait_until(MIGRATION_TIMEOUT, || cache.tier_of(&1u32) == Some(Tier::Fast)));

        cache.get(&2u32).expect("get should succeed");
        assert!(wait_until(MIGRATION_TIMEOUT, || cache.tier_of(&2u32) == Some(Tier::Fast)));
        assert!(wait_until(MIGRATION_TIMEOUT, || cache.tier_of(&1u32) == Some(Tier::Slow)));

        // Both values remain intact and reachable regardless of tier.
        assert_eq!(cache.get(&1u32).unwrap(), value(0xA1));
        assert_eq!(cache.get(&2u32).unwrap(), value(0xB2));
    }

    // ── TTL ───────────────────────────────────────────────────────────────

    // `overhead_manager.base_size` (internal, not part of the public API)
    // adds a fixed TTL bookkeeping cost on top of key + value + expiry-slot
    // size for any object with `Some` expiry -- tens of bytes on top of what
    // a `None`-ttl object of the same value costs. A fast-tier capacity
    // sized only for `None`-ttl objects (as the other tests in this file
    // use) is too tight for a *single* ttl'd object: promoting it to fast
    // can immediately trip `settle_fast_tier` again and demote the very key
    // just promoted, before the test ever observes it as `Fast` (both
    // migrations land in the same `drain_tier_migrations` batch, so there's
    // no window to observe the intermediate state). Use a capacity
    // comfortably larger than one ttl'd object alone, and force demotion
    // pressure with several small filler keys instead of a second
    // same-sized key. Same lesson already documented for `lru_hybrid_cache`/
    // `lfu_hybrid_cache` in CLAUDE.md.
    // ~1 KB values for the tier-pressure tests. They need a fast tier that
    // holds ONE object and not two, and with 15-byte values that window is a
    // handful of bytes wide -- narrower than the per-object DRAM reservation
    // the fast tier now also carries. At 1 KB the reservation is a small
    // fraction and the budgets have a wide margin. Same idiom as the `lru_`
    // and `lfu_` suites.
    //
    // Deliberately NOT applied to every test in this file: `fifo_key_aging_out`
    // runs a 256-byte TOTAL cache and `fifo_capacity_pressure` asserts
    // `used_size() < 1_000`, so both need small values and both already pass.
    const VALUE_LEN: usize = 1024;

    fn value(seed: u8) -> Vec<u8> {
        vec![seed; VALUE_LEN]
    }

    // Holds one ~1 KB value but not two, so the second promotion demotes the
    // first. Was 200, sized for 15-byte values.
    const TTL_FAST_TIER: u64 = 1_600;

    #[test]
    fn ttl_survives_a_demotion() {
        ensure_pmem_allocator_warm();

        let cache = PaperCache::<u32, TieredBuffer>::new(
            1_048_576,
            CacheTierSize::Bytes(TTL_FAST_TIER), PaperPolicy::TwoQHybrid(1.0)).expect("cache should construct");

        let ttl_secs = 5u32;
        let set_at = std::time::Instant::now();
        cache.set(1u32, &value(0xA1), Some(ttl_secs)).expect("set should succeed");

        for key in 2u32..=6 {
            cache.set(key, &value(0xC3), None).expect("set should succeed");
        }

        // Promote key 1 to fast first.
        cache.get(&1u32).expect("get should succeed");
        assert!(wait_until(MIGRATION_TIMEOUT, || cache.tier_of(&1u32) == Some(Tier::Fast)));

        // Promote the fillers one by one until fast-tier pressure demotes
        // key 1 (the LRU-most fast key) back down.
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
    fn ttl_survives_a_promotion() {
        ensure_pmem_allocator_warm();

        let cache = PaperCache::<u32, TieredBuffer>::new(
            1_048_576,
            CacheTierSize::Bytes(1_048_576), PaperPolicy::TwoQHybrid(1.0)).expect("cache should construct");

        let ttl_secs = 5u32;
        let set_at = std::time::Instant::now();
        cache.set(1u32, b"first value 123", Some(ttl_secs)).expect("set should succeed");
        assert_eq!(cache.tier_of(&1u32), Some(Tier::Slow));

        cache.get(&1u32).expect("get should succeed");
        assert!(wait_until(MIGRATION_TIMEOUT, || cache.tier_of(&1u32) == Some(Tier::Fast)));

        assert!(cache.has(&1u32), "key should still be alive right after promoting");

        let remaining = std::time::Duration::from_millis(ttl_secs as u64 * 1000 + 500)
            .saturating_sub(set_at.elapsed());
        std::thread::sleep(remaining);

        assert!(matches!(cache.get(&1u32), Err(CacheError::KeyNotFound)));
    }

    // ── eviction priority ─────────────────────────────────────────────────

    #[test]
    fn terminal_eviction_prefers_fifo_queue_over_main_queue() {
        ensure_pmem_allocator_warm();

        let cache = PaperCache::<u32, TieredBuffer>::new(
            256,
            CacheTierSize::Bytes(256), PaperPolicy::TwoQHybrid(1.0)).expect("cache should construct");

        // Promote key 1 into the main queue (fast) so it's "proven" and
        // should survive eviction pressure that FIFO objects wouldn't.
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
            CacheTierSize::Bytes(1_048_576), PaperPolicy::TwoQHybrid(1.0)).expect("cache should construct");

        cache.set(1u32, b"first value 123", None).expect("set should succeed");
        cache.get(&1u32).expect("get should succeed");
        assert!(wait_until(MIGRATION_TIMEOUT, || cache.tier_of(&1u32) == Some(Tier::Fast)));
        assert_eq!(cache.fast_tier_size(), 1_048_576);

        cache.set_fast_tier_size(CacheTierSize::Bytes(1)).expect("resize should succeed");
        assert_eq!(cache.fast_tier_size(), 1);

        let demoted = wait_until(MIGRATION_TIMEOUT, || cache.tier_of(&1u32) == Some(Tier::Slow));
        assert!(demoted, "shrinking the fast tier should demote the promoted key");
    }

    #[test]
    fn resize_rescales_fifo_capacity() {
        ensure_pmem_allocator_warm();

        let cache = PaperCache::<u32, TieredBuffer>::new(1_024, CacheTierSize::Bytes(1_024), PaperPolicy::TwoQHybrid(0.5))
            .expect("cache should construct");
        // fifo_capacity = 0.5 * 1_000 = 500

        cache.set(1u32, b"first value 123", None).expect("set should succeed");
        assert!(cache.has(&1u32));

        // Shrink overall max_size drastically -> fifo_capacity rescales
        // (proportionally, via k_in) down to a tiny budget -> key 1 should
        // be evicted.
        cache.resize(10).expect("resize should succeed");

        let evicted = wait_until(MIGRATION_TIMEOUT, || !cache.has(&1u32));
        assert!(evicted, "shrinking max_size should rescale fifo_capacity and evict");
    }

    // ── edge cases ────────────────────────────────────────────────────────

    #[test]
    fn zero_fast_tier_size_is_rejected() {
        let result = PaperCache::<u32, TieredBuffer>::new(1_024, CacheTierSize::Bytes(0), PaperPolicy::TwoQHybrid(0.5));
        assert!(matches!(result, Err(CacheError::InvalidFastTierSize)));
    }

    #[test]
    fn fast_tier_size_exceeding_max_size_is_rejected() {
        let result = PaperCache::<u32, TieredBuffer>::new(1_024, CacheTierSize::Bytes(2_048), PaperPolicy::TwoQHybrid(0.5));
        assert!(matches!(result, Err(CacheError::InvalidFastTierSize)));
    }

    #[test]
    fn zero_max_size_is_rejected() {
        let result = PaperCache::<u32, TieredBuffer>::new(0, CacheTierSize::Bytes(100), PaperPolicy::TwoQHybrid(0.5));
        assert!(matches!(result, Err(CacheError::ZeroCacheSize)));
    }

    #[test]
    fn invalid_k_in_is_rejected() {
        assert!(matches!(
            PaperCache::<u32, TieredBuffer>::new(1_024, CacheTierSize::Bytes(512), PaperPolicy::TwoQHybrid(1.5)),
            Err(CacheError::InvalidPolicy),
        ));

        assert!(matches!(
            PaperCache::<u32, TieredBuffer>::new(1_024, CacheTierSize::Bytes(512), PaperPolicy::TwoQHybrid(-0.1)),
            Err(CacheError::InvalidPolicy),
        ));
    }

    #[test]
    fn tiny_fast_tier_still_allows_fifo_admission() {
        ensure_pmem_allocator_warm();

        // fast_tier_size doesn't affect FIFO admission at all (it only
        // governs the main queue's split), so even a 1-byte fast tier
        // should admit and store a real value fine.
        let cache = PaperCache::<u32, TieredBuffer>::new(
            1_048_576,
            CacheTierSize::Bytes(1), PaperPolicy::TwoQHybrid(1.0)).expect("cache should construct");

        cache.set(1u32, b"a value", None).expect("set should succeed");
        assert_eq!(cache.tier_of(&1u32), Some(Tier::Slow));
        assert_eq!(cache.get(&1u32).unwrap(), b"a value");
    }

    #[test]
    fn del_removes_key_from_whichever_tier_it_is_in() {
        ensure_pmem_allocator_warm();

        let cache = PaperCache::<u32, TieredBuffer>::new(
            1_048_576,
            CacheTierSize::Bytes(1_048_576), PaperPolicy::TwoQHybrid(1.0)).expect("cache should construct");

        cache.set(1u32, b"first value 123", None).expect("set should succeed");
        cache.set(2u32, b"second value 45", None).expect("set should succeed");
        cache.get(&1u32).expect("get should succeed");
        assert!(wait_until(MIGRATION_TIMEOUT, || cache.tier_of(&1u32) == Some(Tier::Fast)));

        cache.del(&1u32).expect("del should succeed");
        assert!(!cache.has(&1u32));
        assert_eq!(cache.tier_of(&1u32), None);

        cache.del(&2u32).expect("del should succeed");
        assert!(!cache.has(&2u32));
        assert_eq!(cache.tier_of(&2u32), None);
    }

    #[test]
    fn wipe_clears_both_tiers() {
        ensure_pmem_allocator_warm();

        let cache = PaperCache::<u32, TieredBuffer>::new(
            1_048_576,
            CacheTierSize::Bytes(1_048_576), PaperPolicy::TwoQHybrid(1.0)).expect("cache should construct");

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
