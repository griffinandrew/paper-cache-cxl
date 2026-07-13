/*
 * Copyright (c) Kia Shakiba
 *
 * This source code is licensed under the GNU AGPLv3 license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! Integration tests for the `lfu_hybrid_cache` feature.
//!
//! Run with nightly (required for `allocator_api` via `key_value_pmem`):
//!   cargo +nightly test --test lfu_hybrid_cache_integration --features lfu_hybrid_cache
//!
//! Same one-`PaperCache<K, TieredBuffer>` architecture as
//! `lru_hybrid_cache_integration.rs` — `tier_of` reads the tier directly off
//! the single object map, no `has_in_dram`/`has_in_pmem` pair needed.
//!
//! One behavioral difference from the LRU-hybrid tests worth calling out:
//! ties within the same frequency bucket break toward whichever key is
//! least-recently-touched (see `LfuHybridStack`'s module doc), so once the
//! fast tier is full, admitting a *new* key does not necessarily demote
//! that new key itself — it may instead demote an older, untouched
//! resident tied at the same frequency. Either way the demoted key is, by
//! construction, tied for the fast tier's lowest frequency.
//!
//! What is tested:
//!   * Admission always lands in the fast tier
//!   * Fast-tier pressure demotes the lowest-frequency resident, and
//!     `tier_of` confirms real data movement (not a copy)
//!   * A slow-tier access promotes the key back to fast once its frequency
//!     *strictly* exceeds the fast tier's minimum — a tie does not promote
//!   * TTL set before a demotion/promotion is still correctly enforced after
//!   * Terminal eviction only ever removes the slow-tier minimum-frequency
//!     resident (falling back to the fast tier only if slow is empty) and
//!     is counted in `lfu_hybrid_stats().evictions`
//!   * `set_fast_tier_size` takes effect at runtime
//!   * Zero/invalid/tiny fast-tier-size edge cases
//!
//! All tier-crossing tests exercise the real `Hybrid`/UMF PMEM allocator —
//! see `lru_hybrid_cache_integration.rs`'s module doc for the one-time
//! ~45s PMEM pool warm-up caveat this shares (`ensure_pmem_allocator_warm`
//! below is the same pattern, backed by the same process-wide `Once`).

#[cfg(feature = "lfu_hybrid_cache")]
mod lfu_hybrid_cache_tests {
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

    /// Forces the one-time PMEM allocator pool init/prewarm to complete
    /// before a test's own timing-sensitive assertions begin. See the module
    /// doc comment above for why this is necessary.
    fn ensure_pmem_allocator_warm() {
        let cache = PaperCache::<u32, TieredBuffer>::new(1_000_000, CacheTierSize::Bytes(1))
            .expect("warm-up cache should construct");

        cache.set(0u32, b"warm", None).expect("warm-up set should succeed");

        let ready = wait_until(std::time::Duration::from_secs(90), || {
            cache.tier_of(&0u32) == Some(Tier::Slow)
        });
        assert!(ready, "PMEM allocator warm-up should complete within 90s");
    }

    const MIGRATION_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(10);

    // A fast tier of this size comfortably fits one ~15-byte value's
    // base_size but not two, matching the two demotion-relevant values used
    // throughout ("first value 123" / "second value 45", both 15 bytes).
    const DEMOTES_ONE_OF_TWO: u64 = 40;

    // ── admission ─────────────────────────────────────────────────────────

    #[test]
    fn admission_always_lands_in_fast_tier() {
        let cache = PaperCache::<u32, TieredBuffer>::new(
            1_000_000,
            CacheTierSize::Bytes(1_000_000),
        ).expect("cache should construct");

        cache.set(1u32, b"hello world", None).expect("set should succeed");

        // Admission is synchronous (the object is inserted as `TieredBuffer::
        // Fast` directly inside `set()`, before the WorkerEvent is even
        // broadcast), so this doesn't need `wait_until`.
        assert_eq!(cache.tier_of(&1u32), Some(Tier::Fast));
        assert_eq!(cache.get(&1u32).unwrap(), b"hello world");
    }

    // ── demotion ──────────────────────────────────────────────────────────

    #[test]
    fn fast_tier_full_admission_demotes_the_lru_tied_resident() {
        ensure_pmem_allocator_warm();

        let cache = PaperCache::<u32, TieredBuffer>::new(
            1_000_000,
            CacheTierSize::Bytes(DEMOTES_ONE_OF_TWO),
        ).expect("cache should construct");

        cache.set(1u32, b"first value 123", None).expect("set should succeed");
        assert_eq!(cache.tier_of(&1u32), Some(Tier::Fast));

        // Key 2 is admitted tied at frequency 1 with key 1; ties break
        // toward the least-recently-touched key in the tied bucket, which
        // is key 1 (already resident, untouched since admission) — not the
        // newcomer.
        cache.set(2u32, b"second value 45", None).expect("set should succeed");

        let demoted = wait_until(MIGRATION_TIMEOUT, || {
            cache.tier_of(&1u32) == Some(Tier::Slow)
        });
        assert!(demoted, "key 1 should have demoted to the slow tier");

        assert_ne!(cache.tier_of(&1u32), Some(Tier::Fast));
        assert_eq!(cache.tier_of(&2u32), Some(Tier::Fast));

        // Value survives the physical move intact.
        assert_eq!(cache.get(&1u32).unwrap(), b"first value 123");

        let stats = cache.lfu_hybrid_stats();
        assert!(stats.demotions >= 1);
    }

    #[test]
    fn fast_tier_pressure_demotes_the_lowest_frequency_key() {
        ensure_pmem_allocator_warm();

        // Fits exactly one value (same capacity proven to do so in
        // `fast_tier_full_admission_demotes_the_lru_tied_resident`).
        let cache = PaperCache::<u32, TieredBuffer>::new(
            1_000_000,
            CacheTierSize::Bytes(DEMOTES_ONE_OF_TWO),
        ).expect("cache should construct");

        cache.set(1u32, b"first value 123", None).expect("set should succeed");
        assert_eq!(cache.tier_of(&1u32), Some(Tier::Fast));

        // Bump key 1's frequency well above 1, so it's protected once
        // something else competes for the single fast-tier slot.
        for _ in 0..5 {
            let _ = cache.get(&1u32);
        }
        std::thread::sleep(std::time::Duration::from_millis(300));

        // Key 2 is admitted at frequency 1 -- strictly the lowest in the
        // fast tier now that key 1's frequency is far higher -- so it
        // demotes immediately instead of key 1.
        cache.set(2u32, b"second value 45", None).expect("set should succeed");

        let demoted = wait_until(MIGRATION_TIMEOUT, || {
            cache.tier_of(&2u32) == Some(Tier::Slow)
        });
        assert!(demoted, "key 2 (lowest frequency) should have demoted");
        assert_eq!(cache.tier_of(&1u32), Some(Tier::Fast));
    }

    // ── promotion ─────────────────────────────────────────────────────────

    #[test]
    fn slow_tier_key_promotes_once_frequency_strictly_exceeds_fast_minimum() {
        ensure_pmem_allocator_warm();

        let cache = PaperCache::<u32, TieredBuffer>::new(
            1_000_000,
            CacheTierSize::Bytes(DEMOTES_ONE_OF_TWO),
        ).expect("cache should construct");

        cache.set(1u32, b"first value 123", None).expect("set should succeed");
        cache.set(2u32, b"second value 45", None).expect("set should succeed");

        assert!(wait_until(MIGRATION_TIMEOUT, || cache.tier_of(&1u32) == Some(Tier::Slow)));

        // Grow the fast tier so the upcoming promotion has headroom and
        // doesn't also need to cascade a demotion (that combined behavior
        // is covered separately below).
        cache.set_fast_tier_size(CacheTierSize::Bytes(1_000_000)).expect("resize should succeed");

        // Accessing the slow-tier key should promote it back to fast: its
        // frequency (now 2) strictly exceeds key 2's (still 1).
        assert_eq!(cache.get(&1u32).unwrap(), b"first value 123");

        let promoted = wait_until(MIGRATION_TIMEOUT, || {
            cache.tier_of(&1u32) == Some(Tier::Fast)
        });
        assert!(promoted, "key 1 should have promoted back to the fast tier");

        assert_ne!(cache.tier_of(&1u32), Some(Tier::Slow));

        let stats = cache.lfu_hybrid_stats();
        assert!(stats.promotions >= 1);
    }

    #[test]
    fn tie_with_fast_minimum_does_not_promote() {
        ensure_pmem_allocator_warm();

        let cache = PaperCache::<u32, TieredBuffer>::new(
            1_000_000,
            CacheTierSize::Bytes(DEMOTES_ONE_OF_TWO),
        ).expect("cache should construct");

        cache.set(1u32, b"first value 123", None).expect("set should succeed");
        cache.set(2u32, b"second value 45", None).expect("set should succeed");

        assert!(wait_until(MIGRATION_TIMEOUT, || cache.tier_of(&1u32) == Some(Tier::Slow)));
        assert_eq!(cache.tier_of(&2u32), Some(Tier::Fast));

        // Bump the fast key's frequency to 2, then give the worker a moment
        // to process it before the next access.
        cache.get(&2u32).expect("get should succeed");
        std::thread::sleep(std::time::Duration::from_millis(300));

        // Bump the slow key from 1 to 2 as well -- this only *ties* the
        // fast minimum (also 2 now), which must not promote.
        cache.get(&1u32).expect("get should succeed");
        std::thread::sleep(std::time::Duration::from_millis(300));

        assert_eq!(cache.tier_of(&1u32), Some(Tier::Slow), "a tie must not promote");
    }

    #[test]
    fn cascading_demotion_on_promotion_is_handled() {
        ensure_pmem_allocator_warm();

        // Promoting a slow key back to a full fast tier can itself demote
        // whatever is now the fast tier's lowest-frequency resident.
        let cache = PaperCache::<u32, TieredBuffer>::new(
            1_000_000,
            CacheTierSize::Bytes(DEMOTES_ONE_OF_TWO),
        ).expect("cache should construct");

        cache.set(1u32, b"first value 123", None).expect("set should succeed");
        cache.set(2u32, b"second value 45", None).expect("set should succeed");

        assert!(wait_until(MIGRATION_TIMEOUT, || cache.tier_of(&1u32) == Some(Tier::Slow)));
        assert_eq!(cache.tier_of(&2u32), Some(Tier::Fast));

        // Promote key 1 back — the fast tier only has room for one object
        // here, so key 2 should now be the one demoted.
        cache.get(&1u32).expect("get should succeed");

        assert!(wait_until(MIGRATION_TIMEOUT, || cache.tier_of(&1u32) == Some(Tier::Fast)));
        assert!(wait_until(MIGRATION_TIMEOUT, || cache.tier_of(&2u32) == Some(Tier::Slow)));

        // Both values remain intact and reachable regardless of tier.
        assert_eq!(cache.get(&1u32).unwrap(), b"first value 123");
        assert_eq!(cache.get(&2u32).unwrap(), b"second value 45");
    }

    // ── TTL ───────────────────────────────────────────────────────────────

    // See `lru_hybrid_cache_integration.rs`'s analogous constant/tests for
    // why this is comfortably larger than one ttl'd object's base_size
    // (which includes fixed TTL bookkeeping overhead on top of key + value)
    // and why several small filler keys (rather than one) are used to
    // create demotion pressure.
    const TTL_FAST_TIER: u64 = 200;

    #[test]
    fn ttl_survives_a_demotion() {
        ensure_pmem_allocator_warm();

        let cache = PaperCache::<u32, TieredBuffer>::new(
            1_000_000,
            CacheTierSize::Bytes(TTL_FAST_TIER),
        ).expect("cache should construct");

        // See `ttl_survives_a_demotion` in `lru_hybrid_cache_integration.rs`
        // for why the TTL must be comfortably longer than any plausible
        // migration latency, not merely comparable to it.
        let ttl_secs = 5u32;
        let set_at = std::time::Instant::now();
        cache.set(1u32, b"first value 123", Some(ttl_secs)).expect("set should succeed");

        // Key 1 is inserted first (oldest, tied at frequency 1 with each
        // filler as it arrives), so it's the first candidate demoted once
        // the fast tier's capacity is exceeded.
        for key in 2u32..=6 {
            cache.set(key, b"filler bytes", None).expect("set should succeed");
        }

        assert!(wait_until(MIGRATION_TIMEOUT, || cache.tier_of(&1u32) == Some(Tier::Slow)));

        // If `Object::set_data` (the migration) had reset or dropped
        // `expiry`, the key would already be gone or immortal here.
        assert!(cache.has(&1u32), "key should still be alive right after migrating");

        // Sleep past the *original* deadline (measured from `set`, not from
        // the migration), proving the original clock kept ticking through
        // the tier move rather than being restarted or cleared.
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
            1_000_000,
            CacheTierSize::Bytes(TTL_FAST_TIER),
        ).expect("cache should construct");

        let ttl_secs = 5u32;
        let set_at = std::time::Instant::now();
        cache.set(1u32, b"first value 123", Some(ttl_secs)).expect("set should succeed");

        for key in 2u32..=6 {
            cache.set(key, b"filler bytes", None).expect("set should succeed");
        }

        assert!(wait_until(MIGRATION_TIMEOUT, || cache.tier_of(&1u32) == Some(Tier::Slow)));

        // Promote key 1; its original TTL should still be in effect
        // afterward. Its frequency (now 2, after this access) strictly
        // exceeds the fillers' (still 1), so it promotes.
        cache.get(&1u32).expect("get should succeed");
        assert!(wait_until(MIGRATION_TIMEOUT, || cache.tier_of(&1u32) == Some(Tier::Fast)));

        assert!(cache.has(&1u32), "key should still be alive right after promoting");

        let remaining = std::time::Duration::from_millis(ttl_secs as u64 * 1000 + 500)
            .saturating_sub(set_at.elapsed());
        std::thread::sleep(remaining);

        assert!(matches!(cache.get(&1u32), Err(CacheError::KeyNotFound)));
    }

    // ── eviction ──────────────────────────────────────────────────────────

    #[test]
    fn terminal_eviction_only_removes_from_slow_tier_and_is_counted() {
        ensure_pmem_allocator_warm();

        // A small overall cache with a tiny fast tier: every object demotes
        // to slow almost immediately, and once total usage exceeds max_size
        // the slow-tier minimum-frequency resident must be evicted.
        let cache = PaperCache::<u32, TieredBuffer>::new(
            200,
            CacheTierSize::Bytes(10),
        ).expect("cache should construct");

        for key in 1u32..=10 {
            let _ = cache.set(key, b"payload bytes", None);
        }

        let evicted = wait_until(MIGRATION_TIMEOUT, || {
            cache.lfu_hybrid_stats().evictions >= 1
        });
        assert!(evicted, "at least one terminal eviction should have occurred");

        // Give the worker a moment to settle so the count below is stable.
        std::thread::sleep(std::time::Duration::from_millis(200));

        let stats = cache.lfu_hybrid_stats();
        let present = (1u32..=10).filter(|key| cache.has(key)).count() as u64;

        // Every key is accounted for exactly once: either still present
        // (in fast or slow — doesn't matter which) or evicted. None should
        // be silently lost, and none double-counted.
        assert_eq!(present + stats.evictions, 10);

        // Every evicted key is fully gone, never left dangling in a tier.
        for key in 1u32..=10 {
            if !cache.has(&key) {
                assert_eq!(cache.tier_of(&key), None);
            }
        }
    }

    #[test]
    fn terminal_eviction_falls_back_to_fast_tier_when_slow_tier_is_empty() {
        ensure_pmem_allocator_warm();

        // Fast tier == whole cache (fast_capacity is tracked in raw
        // base_size bytes, independent of the per-object policy overhead
        // that eviction's `max_size` budget also counts) — so in principle
        // nothing needs to demote. But `fast_capacity` and `max_size` use
        // different accounting units (max_size = base_size + per-object
        // overhead; fast_capacity = base_size only), so admitting several
        // keys before eviction has caught up can transiently push raw bytes
        // past fast_capacity even though max_size (with its much larger
        // per-object overhead) admits far fewer objects at steady state.
        // Waiting for `used_size` to settle back under `max_size` after each
        // `set()` keeps admission and eviction in lockstep, so at most ~2
        // objects' worth of raw bytes ever coexist — comfortably under
        // fast_capacity here — and the slow tier genuinely stays empty for
        // the whole test.
        let cache = PaperCache::<u32, TieredBuffer>::new(
            200,
            CacheTierSize::Bytes(200),
        ).expect("cache should construct");

        for key in 1u32..=10 {
            let _ = cache.set(key, b"payload bytes", None);

            let settled = wait_until(MIGRATION_TIMEOUT, || {
                cache.status().map(|s| s.used_size() <= 200).unwrap_or(false)
            });
            assert!(settled, "eviction should keep used_size at or under max_size");
        }

        let stats = cache.lfu_hybrid_stats();
        assert_eq!(stats.demotions, 0, "lockstep admission should never spike fast_used past fast_capacity");
        assert!(stats.evictions >= 1, "should still have evicted once max_size was exceeded");
        assert_eq!(stats.slow_objects, 0);
    }

    // ── runtime fast-tier resize ─────────────────────────────────────────

    #[test]
    fn set_fast_tier_size_takes_effect_at_runtime() {
        ensure_pmem_allocator_warm();

        let cache = PaperCache::<u32, TieredBuffer>::new(
            1_000_000,
            CacheTierSize::Bytes(1_000_000), // huge: nothing demotes initially
        ).expect("cache should construct");

        cache.set(1u32, b"first value 123", None).expect("set should succeed");
        assert_eq!(cache.tier_of(&1u32), Some(Tier::Fast));
        assert_eq!(cache.fast_tier_size(), 1_000_000);

        // Shrink the fast tier drastically; the existing key should demote
        // even without any further access, once the worker applies the
        // resize (mirrors `LfuHybridStack::resize_fast_tier`'s eager
        // `settle_fast_tier` call).
        cache.set_fast_tier_size(CacheTierSize::Bytes(1)).expect("resize should succeed");
        assert_eq!(cache.fast_tier_size(), 1);

        let demoted = wait_until(MIGRATION_TIMEOUT, || {
            cache.tier_of(&1u32) == Some(Tier::Slow)
        });
        assert!(demoted, "shrinking the fast tier should demote the existing key");
    }

    // ── edge cases ────────────────────────────────────────────────────────

    #[test]
    fn zero_fast_tier_size_is_rejected() {
        let result = PaperCache::<u32, TieredBuffer>::new(1_000, CacheTierSize::Bytes(0));
        assert!(matches!(result, Err(CacheError::InvalidFastTierSize)));
    }

    #[test]
    fn fast_tier_size_exceeding_max_size_is_rejected() {
        let result = PaperCache::<u32, TieredBuffer>::new(1_000, CacheTierSize::Bytes(2_000));
        assert!(matches!(result, Err(CacheError::InvalidFastTierSize)));
    }

    #[test]
    fn zero_max_size_is_rejected() {
        let result = PaperCache::<u32, TieredBuffer>::new(0, CacheTierSize::Bytes(100));
        assert!(matches!(result, Err(CacheError::ZeroCacheSize)));
    }

    #[test]
    fn tiny_fast_tier_demotes_everything_almost_immediately() {
        ensure_pmem_allocator_warm();

        let cache = PaperCache::<u32, TieredBuffer>::new(
            1_000_000,
            CacheTierSize::Bytes(1),
        ).expect("cache should construct");

        cache.set(1u32, b"a value", None).expect("set should succeed");

        let demoted = wait_until(MIGRATION_TIMEOUT, || {
            cache.tier_of(&1u32) == Some(Tier::Slow)
        });
        assert!(demoted, "a 1-byte fast tier should demote any real value almost immediately");
        assert_eq!(cache.get(&1u32).unwrap(), b"a value");
    }

    #[test]
    fn del_removes_key_from_whichever_tier_it_is_in() {
        ensure_pmem_allocator_warm();

        let cache = PaperCache::<u32, TieredBuffer>::new(
            1_000_000,
            CacheTierSize::Bytes(DEMOTES_ONE_OF_TWO),
        ).expect("cache should construct");

        cache.set(1u32, b"first value 123", None).expect("set should succeed");
        cache.set(2u32, b"second value 45", None).expect("set should succeed");
        assert!(wait_until(MIGRATION_TIMEOUT, || cache.tier_of(&1u32) == Some(Tier::Slow)));

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
            1_000_000,
            CacheTierSize::Bytes(DEMOTES_ONE_OF_TWO),
        ).expect("cache should construct");

        cache.set(1u32, b"first value 123", None).expect("set should succeed");
        cache.set(2u32, b"second value 45", None).expect("set should succeed");
        assert!(wait_until(MIGRATION_TIMEOUT, || cache.tier_of(&1u32) == Some(Tier::Slow)));

        cache.wipe().expect("wipe should succeed");

        assert!(!cache.has(&1u32));
        assert!(!cache.has(&2u32));
        assert_eq!(cache.tier_of(&1u32), None);
        assert_eq!(cache.tier_of(&2u32), None);
    }
}
