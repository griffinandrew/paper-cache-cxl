/*
 * Copyright (c) Kia Shakiba
 *
 * This source code is licensed under the GNU AGPLv3 license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! Integration tests for the `lru_sized_hybrid_cache` feature.
//!
//! Run with nightly (required for `allocator_api` via `key_value_pmem`):
//!   cargo +nightly test --test hybrid_cache_integration --features lru_sized_hybrid_cache
//!
//! Same one-`PaperCache<K, TieredBuffer>` architecture as `lru_hybrid_cache`
//! (see that feature's own integration test file for the base pattern this
//! mirrors) -- `tier_of` reads the tier directly off the single object map,
//! synchronously reflecting `TieredBuffer::Fast`/`Slow`. What's specific to
//! this feature: both the fast tier and the slow tier are each split into
//! two independently-tracked segments ("small"/"large") by a configurable
//! byte threshold, so tests here additionally check the granular
//! `small_fast_objects`/`large_fast_objects`/`small_slow_objects`/
//! `large_slow_objects` gauges on `hybrid_stats()`, not just the
//! combined `fast_objects`/`slow_objects` totals `lru_hybrid_cache` has.
//!
//! What is tested:
//!   * Admission routes a small/large value to its matching fast segment
//!   * Each fast segment's own pressure demotes independently, without
//!     touching the other segment, into its own matching slow list
//!   * A slow-tier hit promotes back into the segment matching the key's
//!     current size
//!   * An overwrite that crosses the threshold reclassifies into the other
//!     fast segment
//!   * TTL set before a demotion/promotion is still correctly enforced after
//!   * Terminal eviction prefers the slow tier and is counted
//!   * The confirmed last-resort fallback: if both slow lists are empty
//!     but the cache is over max_size, eviction falls back to the fast
//!     segment furthest over its own budget
//!   * `set_fast_tier_size`/`set_large_fast_tier_size`/`set_size_threshold`
//!     each take effect at runtime, independently
//!   * Zero/invalid/tiny fast-tier-size edge cases
//!
//! All tier-crossing tests exercise the real `Hybrid`/UMF PMEM allocator --
//! see `hybrid_cache_integration.rs`'s module doc for the one-time
//! ~45s pool warm-up this sandbox pays on first PMEM touch, and why
//! `ensure_pmem_allocator_warm()` below forces that cost to be paid
//! synchronously before any test's own timing-sensitive assertions begin.

#[cfg(feature = "lru_sized_hybrid_cache")]
mod hybrid_cache_tests {
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
        // Mechanics tests at toy scales: metadata reservation off (see
        // `get_hybrid_dram_shared_overhead`).
        unsafe { std::env::set_var("PAPER_DISABLE_SHARED_OVERHEAD", "1") };
        let cache = PaperCache::<u32, TieredBuffer>::new_sized(
            1_000_000,
            CacheTierSize::Bytes(1),
            CacheTierSize::Bytes(1),
            CacheTierSize::Bytes(1),
        ).expect("warm-up cache should construct");

        cache.set(0u32, b"warm", None).expect("warm-up set should succeed");

        let ready = wait_until(std::time::Duration::from_secs(90), || {
            cache.tier_of(&0u32) == Some(Tier::Slow)
        });
        assert!(ready, "PMEM allocator warm-up should complete within 90s");
    }

    const MIGRATION_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(10);

    // Size-classification threshold used throughout: values well below/above
    // this land unambiguously in the small/large segment regardless of the
    // small, fixed key/expiry overhead `overhead_manager.base_size` adds on
    // top of raw value length (see `hybrid_cache_integration.rs`'s
    // `VALUE_LEN` comment for the equivalent reservation-vs-budget
    // reasoning this crate's hybrid-cache tests already established).
    const SIZE_THRESHOLD: u64 = 1_000;
    const SMALL_VALUE_LEN: usize = 100;
    const LARGE_VALUE_LEN: usize = 2_000;

    fn small_value(seed: u8) -> Vec<u8> {
        vec![seed; SMALL_VALUE_LEN]
    }

    fn large_value(seed: u8) -> Vec<u8> {
        vec![seed; LARGE_VALUE_LEN]
    }

    // ── admission ─────────────────────────────────────────────────────────

    #[test]
    fn admission_routes_small_and_large_values_to_their_respective_segments() {
        let cache = PaperCache::<u32, TieredBuffer>::new_sized(
            1_000_000,
            CacheTierSize::Bytes(1_000_000),
            CacheTierSize::Bytes(1_000_000),
            CacheTierSize::Bytes(SIZE_THRESHOLD),
        ).expect("cache should construct");

        cache.set(1u32, &small_value(0xA1), None).expect("set should succeed");
        cache.set(2u32, &large_value(0xB2), None).expect("set should succeed");

        // Admission is synchronous (both objects are inserted as
        // `TieredBuffer::Fast` directly inside `set()`), so this doesn't
        // need `wait_until`.
        assert_eq!(cache.tier_of(&1u32), Some(Tier::Fast));
        assert_eq!(cache.tier_of(&2u32), Some(Tier::Fast));
        assert_eq!(cache.get(&1u32).unwrap(), small_value(0xA1));
        assert_eq!(cache.get(&2u32).unwrap(), large_value(0xB2));
    }

    // ── demotion ──────────────────────────────────────────────────────────

    #[test]
    fn small_segment_pressure_demotes_independently_of_large_segment() {
        ensure_pmem_allocator_warm();

        let cache = PaperCache::<u32, TieredBuffer>::new_sized(
            1_000_000,
            CacheTierSize::Bytes(160),       // fits ~1 of the 100-byte values
            CacheTierSize::Bytes(1_000_000), // large: huge, never demotes
            CacheTierSize::Bytes(SIZE_THRESHOLD),
        ).expect("cache should construct");

        cache.set(1u32, &small_value(0xA1), None).expect("set should succeed"); // small
        cache.set(2u32, &large_value(0xB2), None).expect("set should succeed"); // large
        cache.set(3u32, &small_value(0xC3), None).expect("set should succeed"); // small, demotes 1

        let demoted = wait_until(MIGRATION_TIMEOUT, || {
            cache.tier_of(&1u32) == Some(Tier::Slow)
        });
        assert!(demoted, "key 1 should have demoted to the slow tier");

        // The large segment (key 2) is completely untouched.
        assert_eq!(cache.tier_of(&2u32), Some(Tier::Fast));
        assert_eq!(cache.get(&1u32).unwrap(), small_value(0xA1));

        let stats = cache.hybrid_stats();
        assert!(stats.demotions >= 1);
        assert_eq!(stats.large_slow_objects, 0);
    }

    #[test]
    fn large_segment_pressure_demotes_independently_of_small_segment() {
        ensure_pmem_allocator_warm();

        let cache = PaperCache::<u32, TieredBuffer>::new_sized(
            1_000_000,
            CacheTierSize::Bytes(1_000_000), // small: huge, never demotes
            CacheTierSize::Bytes(3_200),      // fits ~1 of the 2000-byte values
            CacheTierSize::Bytes(SIZE_THRESHOLD),
        ).expect("cache should construct");

        cache.set(1u32, &large_value(0xA1), None).expect("set should succeed"); // large
        cache.set(2u32, &small_value(0xB2), None).expect("set should succeed"); // small
        cache.set(3u32, &large_value(0xC3), None).expect("set should succeed"); // large, demotes 1

        let demoted = wait_until(MIGRATION_TIMEOUT, || {
            cache.tier_of(&1u32) == Some(Tier::Slow)
        });
        assert!(demoted, "key 1 should have demoted to the slow tier");

        // The small segment (key 2) is completely untouched.
        assert_eq!(cache.tier_of(&2u32), Some(Tier::Fast));
        assert_eq!(cache.get(&1u32).unwrap(), large_value(0xA1));

        let stats = cache.hybrid_stats();
        assert!(stats.demotions >= 1);
        assert_eq!(stats.small_slow_objects, 0);
    }

    // ── promotion / reclassification ─────────────────────────────────────

    #[test]
    fn slow_tier_hit_promotes_back_into_the_segment_matching_current_size() {
        ensure_pmem_allocator_warm();

        let cache = PaperCache::<u32, TieredBuffer>::new_sized(
            1_000_000,
            CacheTierSize::Bytes(160),
            CacheTierSize::Bytes(1_000_000),
            CacheTierSize::Bytes(SIZE_THRESHOLD),
        ).expect("cache should construct");

        cache.set(1u32, &small_value(0xA1), None).expect("set should succeed");
        cache.set(2u32, &small_value(0xB2), None).expect("set should succeed"); // demotes 1

        assert!(wait_until(MIGRATION_TIMEOUT, || cache.tier_of(&1u32) == Some(Tier::Slow)));

        assert_eq!(cache.get(&1u32).unwrap(), small_value(0xA1));

        let promoted = wait_until(MIGRATION_TIMEOUT, || cache.tier_of(&1u32) == Some(Tier::Fast));
        assert!(promoted, "key 1 should have promoted back to the fast tier");
        assert_ne!(cache.tier_of(&1u32), Some(Tier::Slow));

        let stats = cache.hybrid_stats();
        assert!(stats.promotions >= 1);
        assert_eq!(stats.small_fast_objects, 1); // back in the SMALL segment specifically
    }

    #[test]
    fn overwrite_reclassifies_an_existing_key_into_the_other_segment() {
        let cache = PaperCache::<u32, TieredBuffer>::new_sized(
            1_000_000,
            CacheTierSize::Bytes(1_000_000),
            CacheTierSize::Bytes(1_000_000),
            CacheTierSize::Bytes(SIZE_THRESHOLD),
        ).expect("cache should construct");

        cache.set(1u32, &small_value(0xA1), None).expect("set should succeed");
        assert_eq!(cache.tier_of(&1u32), Some(Tier::Fast));
        assert!(wait_until(MIGRATION_TIMEOUT, || cache.hybrid_stats().small_fast_objects == 1));

        cache.set(1u32, &large_value(0xB2), None).expect("overwrite should succeed");
        assert_eq!(cache.tier_of(&1u32), Some(Tier::Fast)); // still fast, just a different segment
        assert_eq!(cache.get(&1u32).unwrap(), large_value(0xB2));

        let reclassified = wait_until(MIGRATION_TIMEOUT, || {
            let stats = cache.hybrid_stats();
            stats.small_fast_objects == 0 && stats.large_fast_objects == 1
        });
        assert!(reclassified, "key 1 should have reclassified into the large segment");
    }

    // ── TTL ───────────────────────────────────────────────────────────────

    // See `hybrid_cache_integration.rs`'s `TTL_FAST_TIER` comment for
    // the full rationale (a capacity sized only for `None`-ttl objects is
    // too tight for a single ttl'd object once its bookkeeping overhead is
    // included). Sized comfortably for one ttl'd 100-byte object plus
    // several small filler keys to force demotion pressure.
    const TTL_SMALL_FAST_TIER: u64 = 300;

    #[test]
    fn ttl_survives_a_demotion() {
        ensure_pmem_allocator_warm();

        let cache = PaperCache::<u32, TieredBuffer>::new_sized(
            1_000_000,
            CacheTierSize::Bytes(TTL_SMALL_FAST_TIER),
            CacheTierSize::Bytes(1_000_000),
            CacheTierSize::Bytes(SIZE_THRESHOLD),
        ).expect("cache should construct");

        let ttl_secs = 5u32;
        let set_at = std::time::Instant::now();
        cache.set(1u32, &small_value(0xC1), Some(ttl_secs)).expect("set should succeed");

        for key in 2u32..=4 {
            cache.set(key, &small_value(key as u8), None).expect("set should succeed");
        }

        assert!(wait_until(MIGRATION_TIMEOUT, || cache.tier_of(&1u32) == Some(Tier::Slow)));

        // If `Object::set_data` (the migration) had reset or dropped
        // `expiry`, the key would already be gone or immortal here.
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

        let cache = PaperCache::<u32, TieredBuffer>::new_sized(
            1_000_000,
            CacheTierSize::Bytes(TTL_SMALL_FAST_TIER),
            CacheTierSize::Bytes(1_000_000),
            CacheTierSize::Bytes(SIZE_THRESHOLD),
        ).expect("cache should construct");

        let ttl_secs = 5u32;
        let set_at = std::time::Instant::now();
        cache.set(1u32, &small_value(0xC1), Some(ttl_secs)).expect("set should succeed");

        for key in 2u32..=4 {
            cache.set(key, &small_value(key as u8), None).expect("set should succeed");
        }

        assert!(wait_until(MIGRATION_TIMEOUT, || cache.tier_of(&1u32) == Some(Tier::Slow)));

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

        let cache = PaperCache::<u32, TieredBuffer>::new_sized(
            200,
            CacheTierSize::Bytes(10),
            CacheTierSize::Bytes(10),
            CacheTierSize::Bytes(SIZE_THRESHOLD),
        ).expect("cache should construct");

        for key in 1u32..=10 {
            let _ = cache.set(key, b"payload bytes", None);
        }

        let evicted = wait_until(MIGRATION_TIMEOUT, || {
            cache.hybrid_stats().evictions >= 1
        });
        assert!(evicted, "at least one terminal eviction should have occurred");

        std::thread::sleep(std::time::Duration::from_millis(200));

        let stats = cache.hybrid_stats();
        let present = (1u32..=10).filter(|key| cache.has(key)).count() as u64;

        assert_eq!(present + stats.evictions, 10);

        for key in 1u32..=10 {
            if !cache.has(&key) {
                assert_eq!(cache.tier_of(&key), None);
            }
        }
    }

    #[test]
    fn terminal_eviction_falls_back_to_the_more_over_budget_fast_segment_when_slow_is_empty() {
        // Capacities confirmed via direct measurement, not derived from
        // first principles: with 10 tiny ("payload bytes", 13-byte) objects
        // and equal 1000-byte small/large capacities, real
        // `status().used_size()` settles at 1180 (base bytes plus the
        // per-object policy-overhead charge), comfortably under 1000 in
        // *either* segment's own raw-byte usage (so nothing ever demotes --
        // the shared-metadata DRAM reservation, which scales with total
        // tracked object count, still leaves each segment's effective
        // budget well above what 10 tiny objects actually need at this
        // capacity) but over a max_size of 1150. An earlier version of this
        // test hand-derived the expected numbers from the overhead
        // constants in `object/overhead.rs` and got them measurably wrong
        // twice in a row (500/500/500 under-shot effective_small and 7 of
        // 10 objects demoted for real; a follow-up guess of 1200 for
        // max_size overshot the real used_size of 1180 by just 20) --
        // replaced with the actual measured value here rather than a third
        // guess. This is the confirmed, documented last-resort fallback
        // (see CLAUDE.md/the plan): with both slow lists genuinely empty,
        // `evict_one()` must evict directly from the fast segment furthest
        // over its own budget -- here, the only fast segment with any
        // objects at all.
        let cache = PaperCache::<u32, TieredBuffer>::new_sized(
            1_150,
            CacheTierSize::Bytes(1_000),
            CacheTierSize::Bytes(1_000),
            CacheTierSize::Bytes(SIZE_THRESHOLD),
        ).expect("cache should construct");

        for key in 1u32..=10 {
            let _ = cache.set(key, b"payload bytes", None);
        }

        let evicted = wait_until(MIGRATION_TIMEOUT, || {
            cache.hybrid_stats().evictions >= 1
        });
        assert!(evicted, "at least one terminal eviction should have occurred");

        std::thread::sleep(std::time::Duration::from_millis(200));

        let stats = cache.hybrid_stats();
        // Confirm this genuinely went through the fast-segment fallback,
        // not a normal slow-tier eviction: nothing was ever demoted.
        assert_eq!(stats.demotions, 0, "nothing should have been demoted in this scenario");
        assert_eq!(stats.small_slow_objects, 0);
        assert_eq!(stats.large_slow_objects, 0);
        assert!(stats.evictions >= 1);
    }

    // ── DRAM cap accounts for shared metadata (hashtable + eviction stacks) ──

    #[test]
    fn dram_cap_reserves_shared_metadata_and_demotes_without_evicting() {
        ensure_pmem_allocator_warm();

        // A minimal (but non-zero, since 0 is rejected) large-segment
        // capacity keeps the shared-metadata reservation concentrated
        // almost entirely on the small segment (see
        // `LruSizedHybridStack::reserved_shares` -- the split is
        // proportional to each segment's *capacity*), matching
        // `hybrid_cache_integration.rs`'s equivalent single-tier test.
        let cache = PaperCache::<u32, TieredBuffer>::new_sized(
            1_000_000,
            CacheTierSize::Bytes(2_000),
            CacheTierSize::Bytes(1),
            CacheTierSize::Bytes(SIZE_THRESHOLD),
        ).expect("cache should construct");

        for key in 1u32..=300 {
            cache.set(key, b"payload bytes", None).expect("set should succeed");
        }

        let demoted = wait_until(MIGRATION_TIMEOUT, || {
            cache.hybrid_stats().demotions >= 1
        });
        assert!(demoted, "shared-metadata reservation should force demotions");

        std::thread::sleep(std::time::Duration::from_millis(300));

        let stats = cache.hybrid_stats();
        assert_eq!(stats.evictions, 0, "the DRAM cap must demote, never evict");

        let present = (1u32..=300).filter(|key| cache.has(key)).count();
        assert_eq!(present, 300, "no key should be evicted by the DRAM cap");

        assert!(stats.small_fast_bytes_used <= cache.fast_tier_size());
    }

    // ── runtime resize ────────────────────────────────────────────────────

    #[test]
    fn set_fast_tier_size_resizes_small_segment_at_runtime() {
        ensure_pmem_allocator_warm();

        let cache = PaperCache::<u32, TieredBuffer>::new_sized(
            1_000_000,
            CacheTierSize::Bytes(1_000_000), // huge: nothing demotes initially
            CacheTierSize::Bytes(1_000_000),
            CacheTierSize::Bytes(SIZE_THRESHOLD),
        ).expect("cache should construct");

        cache.set(1u32, &small_value(0xA1), None).expect("set should succeed");
        assert_eq!(cache.tier_of(&1u32), Some(Tier::Fast));
        assert_eq!(cache.fast_tier_size(), 1_000_000);

        cache.set_fast_tier_size(CacheTierSize::Bytes(1)).expect("resize should succeed");
        assert_eq!(cache.fast_tier_size(), 1);

        let demoted = wait_until(MIGRATION_TIMEOUT, || {
            cache.tier_of(&1u32) == Some(Tier::Slow)
        });
        assert!(demoted, "shrinking the small fast segment should demote the existing small key");
    }

    #[test]
    fn set_large_fast_tier_size_resizes_large_segment_at_runtime() {
        ensure_pmem_allocator_warm();

        let cache = PaperCache::<u32, TieredBuffer>::new_sized(
            1_000_000,
            CacheTierSize::Bytes(1_000_000),
            CacheTierSize::Bytes(1_000_000), // huge: nothing demotes initially
            CacheTierSize::Bytes(SIZE_THRESHOLD),
        ).expect("cache should construct");

        cache.set(1u32, &large_value(0xA1), None).expect("set should succeed");
        assert_eq!(cache.tier_of(&1u32), Some(Tier::Fast));
        assert_eq!(cache.large_fast_tier_size(), 1_000_000);

        cache.set_large_fast_tier_size(CacheTierSize::Bytes(1)).expect("resize should succeed");
        assert_eq!(cache.large_fast_tier_size(), 1);

        let demoted = wait_until(MIGRATION_TIMEOUT, || {
            cache.tier_of(&1u32) == Some(Tier::Slow)
        });
        assert!(demoted, "shrinking the large fast segment should demote the existing large key");
    }

    #[test]
    fn set_size_threshold_changes_future_routing_at_runtime() {
        let cache = PaperCache::<u32, TieredBuffer>::new_sized(
            1_000_000,
            CacheTierSize::Bytes(1_000_000),
            CacheTierSize::Bytes(1_000_000),
            CacheTierSize::Bytes(SIZE_THRESHOLD),
        ).expect("cache should construct");

        cache.set(1u32, &small_value(0xA1), None).expect("set should succeed"); // small
        assert!(wait_until(MIGRATION_TIMEOUT, || cache.hybrid_stats().small_fast_objects == 1));

        assert_eq!(cache.size_threshold(), SIZE_THRESHOLD);
        cache.set_size_threshold(CacheTierSize::Bytes(1)).expect("threshold change should succeed");
        assert_eq!(cache.size_threshold(), 1);

        // Existing key 1 is not retroactively reclassified.
        assert_eq!(cache.tier_of(&1u32), Some(Tier::Fast));

        // A brand-new admission of the same small value now routes large
        // under the lowered threshold.
        cache.set(2u32, &small_value(0xB2), None).expect("set should succeed");
        let routed_large = wait_until(MIGRATION_TIMEOUT, || {
            cache.hybrid_stats().large_fast_objects == 1
        });
        assert!(routed_large, "key 2 should route to the large segment under the new threshold");
    }

    // ── edge cases ────────────────────────────────────────────────────────

    #[test]
    fn zero_small_fast_tier_size_is_rejected() {
        let result = PaperCache::<u32, TieredBuffer>::new_sized(
            1_000, CacheTierSize::Bytes(0), CacheTierSize::Bytes(500), CacheTierSize::Bytes(100),
        );
        assert!(matches!(result, Err(CacheError::InvalidFastTierSize)));
    }

    #[test]
    fn zero_large_fast_tier_size_is_rejected() {
        let result = PaperCache::<u32, TieredBuffer>::new_sized(
            1_000, CacheTierSize::Bytes(500), CacheTierSize::Bytes(0), CacheTierSize::Bytes(100),
        );
        assert!(matches!(result, Err(CacheError::InvalidFastTierSize)));
    }

    #[test]
    fn small_fast_tier_size_exceeding_max_size_is_rejected() {
        let result = PaperCache::<u32, TieredBuffer>::new_sized(
            1_000, CacheTierSize::Bytes(2_000), CacheTierSize::Bytes(500), CacheTierSize::Bytes(100),
        );
        assert!(matches!(result, Err(CacheError::InvalidFastTierSize)));
    }

    #[test]
    fn large_fast_tier_size_exceeding_max_size_is_rejected() {
        let result = PaperCache::<u32, TieredBuffer>::new_sized(
            1_000, CacheTierSize::Bytes(500), CacheTierSize::Bytes(2_000), CacheTierSize::Bytes(100),
        );
        assert!(matches!(result, Err(CacheError::InvalidFastTierSize)));
    }

    #[test]
    fn zero_max_size_is_rejected() {
        let result = PaperCache::<u32, TieredBuffer>::new_sized(
            0, CacheTierSize::Bytes(100), CacheTierSize::Bytes(100), CacheTierSize::Bytes(50),
        );
        assert!(matches!(result, Err(CacheError::ZeroCacheSize)));
    }

    #[test]
    fn del_removes_key_from_whichever_segment_or_tier_it_is_in() {
        ensure_pmem_allocator_warm();

        let cache = PaperCache::<u32, TieredBuffer>::new_sized(
            1_000_000,
            CacheTierSize::Bytes(160),
            CacheTierSize::Bytes(1_000_000),
            CacheTierSize::Bytes(SIZE_THRESHOLD),
        ).expect("cache should construct");

        cache.set(1u32, &small_value(0xA1), None).expect("set should succeed");
        cache.set(2u32, &small_value(0xB2), None).expect("set should succeed"); // demotes 1
        cache.set(3u32, &large_value(0xC3), None).expect("set should succeed"); // large, stays fast

        assert!(wait_until(MIGRATION_TIMEOUT, || cache.tier_of(&1u32) == Some(Tier::Slow)));

        cache.del(&1u32).expect("del should succeed");
        assert!(!cache.has(&1u32));
        assert_eq!(cache.tier_of(&1u32), None);

        cache.del(&2u32).expect("del should succeed");
        assert!(!cache.has(&2u32));
        assert_eq!(cache.tier_of(&2u32), None);

        cache.del(&3u32).expect("del should succeed");
        assert!(!cache.has(&3u32));
        assert_eq!(cache.tier_of(&3u32), None);
    }

    #[test]
    fn wipe_clears_every_segment_and_the_slow_tier() {
        ensure_pmem_allocator_warm();

        let cache = PaperCache::<u32, TieredBuffer>::new_sized(
            1_000_000,
            CacheTierSize::Bytes(160),
            CacheTierSize::Bytes(1_000_000),
            CacheTierSize::Bytes(SIZE_THRESHOLD),
        ).expect("cache should construct");

        cache.set(1u32, &small_value(0xA1), None).expect("set should succeed");
        cache.set(2u32, &small_value(0xB2), None).expect("set should succeed"); // demotes 1
        cache.set(3u32, &large_value(0xC3), None).expect("set should succeed"); // large

        assert!(wait_until(MIGRATION_TIMEOUT, || cache.tier_of(&1u32) == Some(Tier::Slow)));

        cache.wipe().expect("wipe should succeed");

        assert!(!cache.has(&1u32));
        assert!(!cache.has(&2u32));
        assert!(!cache.has(&3u32));
        assert_eq!(cache.tier_of(&1u32), None);
        assert_eq!(cache.tier_of(&2u32), None);
        assert_eq!(cache.tier_of(&3u32), None);
    }
}
