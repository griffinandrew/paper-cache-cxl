/*
 * Copyright (c) Kia Shakiba
 *
 * This source code is licensed under the GNU AGPLv3 license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! Integration tests for the `lru_lfu_hybrid_cache` feature.
//!
//! Run with nightly (required for `allocator_api` via `key_value_pmem`):
//!   cargo +nightly test --test hybrid_cache_integration --features lru_lfu_hybrid_cache
//!
//! This design is **one** `PaperCache<K, TieredBuffer>` whose two tiers rank
//! by *different* metrics — recency (LRU) in the fast tier, frequency (LFU)
//! in the slow tier. What that buys, and what these tests check end to end
//! against the real `Hybrid`/UMF PMEM allocator:
//!
//!   * Admission always lands fast, at frequency 1
//!   * Fast-tier pressure demotes the LRU tail with real data movement
//!   * A single slow-tier access does NOT promote when `promote_k > 1` —
//!     the property that separates this from `lru_hybrid_cache`
//!   * Crossing `promote_k` promotes, with real data movement back to DRAM
//!   * An overwrite goes through the same frequency gate a read does
//!   * Eviction takes the slow tier's least-frequent object, so an object
//!     that was hot in DRAM outlives a one-hit-wonder demoted alongside it
//!   * TTL survives a demotion and a promotion
//!   * `set_fast_tier_size` takes effect at runtime; `del`/`wipe` across tiers
//!
//! See `tests/hybrid_cache_integration.rs`'s module doc for why every
//! PMEM-touching test calls `ensure_pmem_allocator_warm()` first (a one-time
//! ~45s NUMA pool init that a concurrently running test can otherwise stall
//! this one behind, losing races against short TTLs).

#[cfg(feature = "lru_lfu_hybrid_cache")]
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
    /// before a test's own timing-sensitive assertions begin.
    fn ensure_pmem_allocator_warm() {
        let cache = PaperCache::<u32, TieredBuffer>::new(1_000_000, CacheTierSize::Bytes(1), PaperPolicy::LruLfuHybrid(2))
            .expect("warm-up cache should construct");

        cache.set(0u32, b"warm", None).expect("warm-up set should succeed");

        let ready = wait_until(std::time::Duration::from_secs(90), || {
            cache.tier_of(&0u32) == Some(Tier::Slow)
        });
        assert!(ready, "PMEM allocator warm-up should complete within 90s");
    }

    const MIGRATION_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(10);

    /// ~1 KB values, so the per-object shared-metadata DRAM reservation
    /// (tens of bytes — see `object/overhead.rs`) stays a small fraction of
    /// the byte budgets below rather than dominating them.
    const VALUE_LEN: usize = 1024;
    const MAX_SIZE: u64 = 1_000_000;
    /// Holds ~2 of the values above, so a third admission forces a demotion.
    const FAST_TIER: u64 = 2_600;
    /// The promotion threshold under test.
    ///
    /// `promote_k` is an ABSOLUTE frequency, not a count of accesses since
    /// demotion (see `lru_lfu_hybrid_stack.rs`'s module doc). A key admitted
    /// and never accessed demotes carrying frequency 1, so `2` would be
    /// reached by a single slow access — behaving exactly like
    /// `lru_hybrid_cache` and making the feature under test invisible. `3` is
    /// the smallest value that actually filters, requiring two slow accesses.
    const PROMOTE_K: u16 = 3;
    /// Slow accesses a never-accessed demoted key needs to reach PROMOTE_K.
    const SLOW_ACCESSES_TO_PROMOTE: u16 = PROMOTE_K - 1;

    fn value(seed: u8) -> Vec<u8> {
        vec![seed; VALUE_LEN]
    }

    fn make_cache() -> PaperCache<u32, TieredBuffer> {
        PaperCache::<u32, TieredBuffer>::new(
            MAX_SIZE,
            CacheTierSize::Bytes(FAST_TIER), PaperPolicy::LruLfuHybrid(PROMOTE_K)).expect("cache should construct")
    }

    /// Fills the fast tier and keeps admitting until `key` is observed in the
    /// slow tier, returning the next unused filler key.
    fn force_into_slow(cache: &PaperCache<u32, TieredBuffer>, key: u32, first_filler: u32) -> u32 {
        let mut filler = first_filler;

        for _ in 0..12 {
            if cache.tier_of(&key) == Some(Tier::Slow) {
                return filler;
            }

            cache.set(filler, &value(0xF0), None).expect("filler set should succeed");
            filler += 1;

            wait_until(std::time::Duration::from_millis(300), || {
                cache.tier_of(&key) == Some(Tier::Slow)
            });
        }

        assert_eq!(cache.tier_of(&key), Some(Tier::Slow), "key {key} should have demoted");
        filler
    }

    // ── admission ─────────────────────────────────────────────────────────

    #[test]
    fn admission_always_lands_in_the_fast_tier() {
        let cache = PaperCache::<u32, TieredBuffer>::new(
            MAX_SIZE,
            CacheTierSize::Bytes(MAX_SIZE), PaperPolicy::LruLfuHybrid(PROMOTE_K)).expect("cache should construct");

        cache.set(1u32, b"hello world", None).expect("set should succeed");

        // Admission is synchronous — the object is built as `TieredBuffer::
        // Fast` inside `set()` before the WorkerEvent is even broadcast.
        assert_eq!(cache.tier_of(&1u32), Some(Tier::Fast));
        assert_eq!(cache.get(&1u32).unwrap(), b"hello world");
    }

    // ── demotion ──────────────────────────────────────────────────────────

    #[test]
    fn fast_tier_pressure_demotes_with_real_data_movement() {
        ensure_pmem_allocator_warm();

        let cache = make_cache();

        cache.set(1u32, &value(0xA1), None).expect("set should succeed");
        assert_eq!(cache.tier_of(&1u32), Some(Tier::Fast));

        force_into_slow(&cache, 1u32, 2u32);

        // One object map, so "gone from fast" and "present in slow" are the
        // same fact — and the bytes survived the physical move intact.
        assert_ne!(cache.tier_of(&1u32), Some(Tier::Fast));
        assert_eq!(cache.get(&1u32).unwrap(), value(0xA1));

        let stats = cache.hybrid_stats();
        assert!(stats.demotions > 0, "expected a real demotion; got {stats:?}");
    }

    // ── promotion: the property that separates this from lru_hybrid_cache ──

    #[test]
    fn a_single_slow_access_does_not_promote() {
        ensure_pmem_allocator_warm();

        // The whole point of the frequency gate: under `lru_hybrid_cache`
        // this same access would promote immediately.
        let cache = make_cache();

        cache.set(1u32, &value(0xA1), None).expect("set should succeed");
        force_into_slow(&cache, 1u32, 2u32);

        // The key demoted carrying frequency 1, so it needs
        // SLOW_ACCESSES_TO_PROMOTE accesses. Do one fewer than that.
        for _ in 0..(SLOW_ACCESSES_TO_PROMOTE - 1) {
            assert_eq!(cache.get(&1u32).unwrap(), value(0xA1));
        }

        // Give the worker a real chance to promote it if it were going to.
        std::thread::sleep(std::time::Duration::from_millis(500));

        assert_eq!(
            cache.tier_of(&1u32),
            Some(Tier::Slow),
            "must not reach DRAM below the absolute threshold promote_k = {PROMOTE_K}",
        );
    }

    #[test]
    fn crossing_the_threshold_promotes_with_real_data_movement() {
        ensure_pmem_allocator_warm();

        let cache = make_cache();

        cache.set(1u32, &value(0xA1), None).expect("set should succeed");
        force_into_slow(&cache, 1u32, 2u32);

        // Demoted carrying frequency 1; PROMOTE_K accesses take it over.
        for _ in 0..SLOW_ACCESSES_TO_PROMOTE {
            assert_eq!(cache.get(&1u32).unwrap(), value(0xA1));
        }

        let promoted = wait_until(MIGRATION_TIMEOUT, || {
            cache.tier_of(&1u32) == Some(Tier::Fast)
        });
        assert!(promoted, "key 1 should have promoted after {PROMOTE_K} slow accesses");

        // Real movement back to DRAM, value intact.
        assert_ne!(cache.tier_of(&1u32), Some(Tier::Slow));
        assert_eq!(cache.get(&1u32).unwrap(), value(0xA1));

        let stats = cache.hybrid_stats();
        assert!(stats.promotions > 0, "expected a real promotion; got {stats:?}");
    }

    #[test]
    fn an_overwrite_goes_through_the_same_gate_as_a_read() {
        ensure_pmem_allocator_warm();

        let cache = make_cache();

        cache.set(1u32, &value(0xA1), None).expect("set should succeed");
        force_into_slow(&cache, 1u32, 2u32);

        // A single overwrite is one access — not an automatic re-admission
        // to the fast tier, which is what `lru_hybrid_cache` would do.
        cache.set(1u32, &value(0xB2), None).expect("overwrite should succeed");
        std::thread::sleep(std::time::Duration::from_millis(500));

        assert_eq!(
            cache.tier_of(&1u32),
            Some(Tier::Slow),
            "a set() must not bypass the frequency gate",
        );
        assert_eq!(cache.get(&1u32).unwrap(), value(0xB2), "new bytes, still in PMEM");
    }

    // ── eviction ──────────────────────────────────────────────────────────

    #[test]
    fn eviction_prefers_the_least_frequently_used_slow_object() {
        ensure_pmem_allocator_warm();

        // The payoff of carrying frequency across a demotion: an object that
        // was genuinely hot while in DRAM must outlive a one-hit-wonder that
        // was demoted alongside it, even though LRU order would say
        // otherwise.
        let cache = make_cache();

        cache.set(1u32, &value(0xA1), None).expect("set should succeed");

        // Make key 1 hot while it is still fast. Reads only — a read never
        // changes tier for a fast key, it just bumps the carried counter.
        for _ in 0..6 {
            assert_eq!(cache.get(&1u32).unwrap(), value(0xA1));
        }

        cache.set(2u32, &value(0xB2), None).expect("set should succeed");

        let mut filler = force_into_slow(&cache, 2u32, 3u32);
        filler = force_into_slow(&cache, 1u32, filler);

        assert_eq!(cache.tier_of(&1u32), Some(Tier::Slow));
        assert_eq!(cache.tier_of(&2u32), Some(Tier::Slow));

        // Drive the cache past max_size so terminal eviction runs.
        for _ in 0..1_400 {
            cache.set(filler, &value(0xC3), None).expect("filler set should succeed");
            filler += 1;

            wait_until(std::time::Duration::from_secs(5), || {
                cache.status().expect("status").used_size() <= MAX_SIZE
            });
        }

        let stats = cache.hybrid_stats();
        assert!(stats.evictions > 0, "expected terminal evictions; got {stats:?}");

        // Key 2 (frequency 1) should have been evicted before key 1
        // (frequency 7 when it demoted).
        assert!(
            !cache.has(&2u32) || cache.has(&1u32),
            "the once-hot object should not be evicted while the one-hit-wonder survives",
        );
    }

    // ── ttl across a tier move ────────────────────────────────────────────

    #[test]
    fn ttl_survives_a_demotion() {
        ensure_pmem_allocator_warm();

        // A ttl'd object's base_size carries a fixed TTL bookkeeping cost on
        // top of its value, so the fast tier is sized generously here rather
        // than to a bare multiple of VALUE_LEN.
        let cache = PaperCache::<u32, TieredBuffer>::new(
            MAX_SIZE,
            CacheTierSize::Bytes(4_500), PaperPolicy::LruLfuHybrid(PROMOTE_K)).expect("cache should construct");

        // A TTL comfortably longer than any plausible migration latency:
        // `tier_of` reports an expired object as absent, so a short TTL would
        // make "never migrated" and "migrated then expired" indistinguishable.
        let ttl_secs = 5u32;
        let set_at = std::time::Instant::now();
        cache.set(1u32, &value(0xA1), Some(ttl_secs)).expect("set should succeed");

        force_into_slow(&cache, 1u32, 2u32);

        // If the migration had dropped or reset `expiry`, the key would be
        // either already gone or immortal here.
        assert!(cache.has(&1u32), "key should still be alive right after migrating");
        assert_eq!(cache.get(&1u32).unwrap(), value(0xA1));

        // Sleep past the *original* deadline, measured from `set` rather than
        // from the migration — proving the original clock kept running
        // through the move to PMEM instead of being restarted.
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
            MAX_SIZE,
            CacheTierSize::Bytes(4_500), PaperPolicy::LruLfuHybrid(PROMOTE_K)).expect("cache should construct");

        let ttl_secs = 8u32;
        let set_at = std::time::Instant::now();
        cache.set(1u32, &value(0xA1), Some(ttl_secs)).expect("set should succeed");

        force_into_slow(&cache, 1u32, 2u32);

        // Each read is one access; this many cross the absolute threshold.
        for _ in 0..SLOW_ACCESSES_TO_PROMOTE {
            assert_eq!(cache.get(&1u32).unwrap(), value(0xA1));
        }

        let promoted = wait_until(MIGRATION_TIMEOUT, || {
            cache.tier_of(&1u32) == Some(Tier::Fast)
        });
        assert!(promoted, "key 1 should have promoted");
        assert!(cache.has(&1u32), "key should still be alive right after promoting");

        // The round trip DRAM -> PMEM -> DRAM must not have reset the clock.
        let remaining = std::time::Duration::from_millis(ttl_secs as u64 * 1000 + 500)
            .saturating_sub(set_at.elapsed());
        std::thread::sleep(remaining);

        assert!(matches!(cache.get(&1u32), Err(CacheError::KeyNotFound)));
        assert!(!cache.has(&1u32));
    }

    // ── configuration / lifecycle ─────────────────────────────────────────

    #[test]
    fn set_fast_tier_size_takes_effect_at_runtime() {
        ensure_pmem_allocator_warm();

        let cache = PaperCache::<u32, TieredBuffer>::new(
            MAX_SIZE,
            CacheTierSize::Bytes(MAX_SIZE), PaperPolicy::LruLfuHybrid(PROMOTE_K)).expect("cache should construct");

        cache.set(1u32, &value(0xA1), None).expect("set should succeed");
        cache.set(2u32, &value(0xB2), None).expect("set should succeed");
        assert_eq!(cache.tier_of(&1u32), Some(Tier::Fast));

        // Shrink the budget under the resident set — that alone must demote.
        cache.set_fast_tier_size(CacheTierSize::Bytes(1_200))
            .expect("resize should succeed");

        let demoted = wait_until(MIGRATION_TIMEOUT, || {
            cache.tier_of(&1u32) == Some(Tier::Slow)
        });
        assert!(demoted, "shrinking the fast tier should demote the LRU tail");
        assert_eq!(cache.fast_tier_size(), 1_200);
    }

    #[test]
    fn invalid_fast_tier_sizes_are_rejected() {
        let zero = PaperCache::<u32, TieredBuffer>::new(
            MAX_SIZE,
            CacheTierSize::Bytes(0), PaperPolicy::LruLfuHybrid(PROMOTE_K));
        assert!(matches!(zero, Err(CacheError::InvalidFastTierSize)));

        let too_big = PaperCache::<u32, TieredBuffer>::new(
            MAX_SIZE,
            CacheTierSize::Bytes(MAX_SIZE + 1), PaperPolicy::LruLfuHybrid(PROMOTE_K));
        assert!(matches!(too_big, Err(CacheError::InvalidFastTierSize)));
    }

    #[test]
    fn a_zero_promotion_threshold_is_rejected() {
        // 0 would make every slow object promotable before it was ever
        // accessed, which is not the same policy at any k.
        let bad = PaperCache::<u32, TieredBuffer>::new(
            MAX_SIZE,
            CacheTierSize::Bytes(FAST_TIER), PaperPolicy::LruLfuHybrid(0));
        assert!(matches!(bad, Err(CacheError::InvalidPolicy)));
    }

    #[test]
    fn del_and_wipe_work_across_both_tiers() {
        ensure_pmem_allocator_warm();

        let cache = make_cache();

        cache.set(1u32, &value(0xA1), None).expect("set should succeed");
        let filler = force_into_slow(&cache, 1u32, 2u32);

        // A fast-tier key to delete alongside the slow-tier one.
        cache.set(99u32, &value(0xD4), None).expect("set should succeed");
        assert_eq!(cache.tier_of(&99u32), Some(Tier::Fast));

        cache.del(&1u32).expect("deleting a slow-tier key should succeed");
        assert!(!cache.has(&1u32));

        cache.del(&99u32).expect("deleting a fast-tier key should succeed");
        assert!(!cache.has(&99u32));

        cache.wipe().expect("wipe should succeed");

        for key in 2..filler {
            assert!(!cache.has(&key), "key {key} should be gone after a wipe");
        }
        assert_eq!(cache.status().expect("status").num_objects(), 0);
    }

    #[test]
    fn tier_gauges_account_for_every_object() {
        ensure_pmem_allocator_warm();

        let cache = make_cache();

        for key in 0..10u32 {
            cache.set(key, &value(0xA1), None).expect("set should succeed");
        }

        let settled = wait_until(MIGRATION_TIMEOUT, || {
            let stats = cache.hybrid_stats();
            let objects = cache.status().expect("status").num_objects();
            stats.fast_objects + stats.slow_objects == objects
        });

        let stats = cache.hybrid_stats();
        assert!(
            settled,
            "every object should be accounted to exactly one tier; got fast={} slow={} objects={}",
            stats.fast_objects,
            stats.slow_objects,
            cache.status().expect("status").num_objects(),
        );
        assert!(stats.fast_bytes_used > 0, "fast tier should be holding something");
        assert!(stats.slow_bytes_used > 0, "the fast tier is too small for 10 keys");
    }
}
