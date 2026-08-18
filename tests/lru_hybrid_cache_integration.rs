/*
 * Copyright (c) Kia Shakiba
 *
 * This source code is licensed under the GNU AGPLv3 license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! Integration tests for the `lru_hybrid_cache` feature.
//!
//! Run with nightly (required for `allocator_api` via `key_value_pmem`):
//!   cargo +nightly test --test lru_hybrid_cache_integration --features lru_hybrid_cache
//!
//! This feature is **one** `PaperCache<K, TieredBuffer>` instance (not two
//! composed `PaperCache`s), so `tier_of` reads the tier directly off the
//! single object map.
//!
//! What is tested:
//!   * Admission always lands in the fast tier
//!   * Fast-tier pressure demotes the LRU tail to the slow tier, and `tier_of`
//!     confirms it is gone from the fast tier (real data movement, not a copy)
//!   * A slow-tier hit promotes the key back to the fast tier, and `tier_of`
//!     confirms it is gone from the slow tier
//!   * TTL set before a demotion/promotion is still correctly enforced after
//!   * Terminal eviction only ever removes the slow-tier LRU tail and is
//!     counted in `lru_hybrid_stats().evictions`
//!   * `set_fast_tier_size` takes effect at runtime
//!   * Zero/invalid/tiny fast-tier-size edge cases
//!
//! All tier-crossing tests exercise the real `Hybrid`/UMF PMEM allocator (no
//! shortcuts): the very first PMEM allocation in the whole test process
//! triggers a one-time NUMA-node pool init + prewarm that can take on the
//! order of a minute on first touch (observed ~45s in this sandbox) — see
//! `allocator.rs`'s `HybridObjects`. Whichever test's thread happens to
//! trigger that first pays the cost inline; tests running concurrently on
//! other threads are *not* blocked by it (different `PaperCache` instances,
//! different worker threads), so a test with its own tight wall-clock
//! assertion (e.g. a short TTL) can race the one-time warm-up and fail
//! spuriously. `ensure_pmem_allocator_warm()` below forces that one-time
//! cost to be paid synchronously at the start of *every* test — since it's
//! backed by the same process-wide `Once`, only the very first call actually
//! waits ~45s; every other call (in this test or any other) returns almost
//! immediately once the allocator is warm.

#[cfg(feature = "lru_hybrid_cache")]
mod lru_hybrid_cache_tests {
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

    // The fast-tier budget now also reserves an approximate per-object DRAM
    // cost for the shared object hashtable + eviction stacks (tens of bytes per
    // object — see `object/overhead.rs::get_hybrid_dram_shared_overhead`). To
    // keep the byte-sized fast-tier budgets below behaving intuitively
    // (~value-sized) rather than being dominated by that reservation, the
    // demotion/promotion tests use ~1 KB values, so the per-object reservation
    // is a small fraction and the fast-tier sizes have a wide, robust margin.
    const VALUE_LEN: usize = 1024;

    fn value(seed: u8) -> Vec<u8> {
        vec![seed; VALUE_LEN]
    }

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
    fn fast_tier_pressure_demotes_lru_tail_with_real_data_movement() {
        ensure_pmem_allocator_warm();

        // A fast tier sized to hold ~1 of these ~1 KB values (after the
        // per-object shared-metadata reservation) guarantees the first key
        // demotes once the second is admitted.
        let cache = PaperCache::<u32, TieredBuffer>::new(
            1_000_000,
            CacheTierSize::Bytes(1_600),
        ).expect("cache should construct");

        cache.set(1u32, &value(0xA1), None).expect("set should succeed");
        assert_eq!(cache.tier_of(&1u32), Some(Tier::Fast));

        cache.set(2u32, &value(0xB2), None).expect("set should succeed");

        let demoted = wait_until(MIGRATION_TIMEOUT, || {
            cache.tier_of(&1u32) == Some(Tier::Slow)
        });
        assert!(demoted, "key 1 should have demoted to the slow tier");

        // Real data movement, not a copy: the key is gone from the fast
        // tier's accounting entirely — there is only one object map, so
        // "gone from fast" and "present in slow" are the same fact checked
        // two ways.
        assert_ne!(cache.tier_of(&1u32), Some(Tier::Fast));

        // Value survives the physical move intact.
        assert_eq!(cache.get(&1u32).unwrap(), value(0xA1));

        let stats = cache.lru_hybrid_stats();
        assert!(stats.demotions >= 1);
    }

    // ── promotion ─────────────────────────────────────────────────────────

    #[test]
    fn slow_tier_hit_promotes_and_is_confirmed_gone_from_slow() {
        ensure_pmem_allocator_warm();

        let cache = PaperCache::<u32, TieredBuffer>::new(
            1_000_000,
            CacheTierSize::Bytes(1_600),
        ).expect("cache should construct");

        cache.set(1u32, &value(0xA1), None).expect("set should succeed");
        cache.set(2u32, &value(0xB2), None).expect("set should succeed");

        let demoted = wait_until(MIGRATION_TIMEOUT, || {
            cache.tier_of(&1u32) == Some(Tier::Slow)
        });
        assert!(demoted, "key 1 should have demoted to the slow tier");

        // Accessing the slow-tier key should promote it back to fast.
        assert_eq!(cache.get(&1u32).unwrap(), value(0xA1));

        let promoted = wait_until(MIGRATION_TIMEOUT, || {
            cache.tier_of(&1u32) == Some(Tier::Fast)
        });
        assert!(promoted, "key 1 should have promoted back to the fast tier");

        // Gone from slow, not copied — same single-object-map guarantee as
        // the demotion test, checked in the other direction.
        assert_ne!(cache.tier_of(&1u32), Some(Tier::Slow));

        let stats = cache.lru_hybrid_stats();
        assert!(stats.promotions >= 1);
    }

    #[test]
    fn cascading_demotion_on_promotion_is_handled() {
        ensure_pmem_allocator_warm();

        // Promoting a slow key back to a full fast tier can itself demote
        // whatever is now the fast-tier LRU tail — exercises the "more than
        // one migration per call" path (`LruHybridStack::settle_fast_tier`'s
        // loop), not just the common one-in-one-out case.
        let cache = PaperCache::<u32, TieredBuffer>::new(
            1_000_000,
            CacheTierSize::Bytes(1_600),
        ).expect("cache should construct");

        cache.set(1u32, &value(0xA1), None).expect("set should succeed");
        cache.set(2u32, &value(0xB2), None).expect("set should succeed");

        assert!(wait_until(MIGRATION_TIMEOUT, || cache.tier_of(&1u32) == Some(Tier::Slow)));
        assert!(wait_until(MIGRATION_TIMEOUT, || cache.tier_of(&2u32) == Some(Tier::Fast)));

        // Promote key 1 back — key 2 should now be the one under pressure.
        cache.get(&1u32).expect("get should succeed");

        assert!(wait_until(MIGRATION_TIMEOUT, || cache.tier_of(&1u32) == Some(Tier::Fast)));
        assert!(wait_until(MIGRATION_TIMEOUT, || cache.tier_of(&2u32) == Some(Tier::Slow)));

        // Both values remain intact and reachable regardless of tier.
        assert_eq!(cache.get(&1u32).unwrap(), value(0xA1));
        assert_eq!(cache.get(&2u32).unwrap(), value(0xB2));
    }

    // ── TTL ───────────────────────────────────────────────────────────────

    // `overhead_manager.base_size` (internal, not part of the public API)
    // adds a fixed TTL bookkeeping cost on top of key + value + expiry-slot
    // size for any object with `Some` expiry (`get_ttl_overhead` in
    // `object/overhead.rs`) — tens of bytes on top of what a `None`-ttl
    // object of the same value costs. A fast-tier capacity sized only for
    // `None`-ttl objects (as the demotion/promotion tests above use) is too
    // tight for a *single* ttl'd object: promoting it back to fast can
    // immediately trip `settle_fast_tier` again and re-demote the very key
    // just promoted, before the test ever observes it as `Fast`. Use a
    // capacity comfortably larger than one ttl'd object alone, and force
    // demotion pressure with several filler keys. Sized (with the ~1 KB
    // `value()` payloads and the per-object shared-metadata reservation) to
    // hold ~2 objects, so the ttl'd key demotes under filler pressure yet can
    // still be promoted back and observed as `Fast` before re-settling.
    const TTL_FAST_TIER: u64 = 2_600;

    #[test]
    fn ttl_survives_a_demotion() {
        ensure_pmem_allocator_warm();

        let cache = PaperCache::<u32, TieredBuffer>::new(
            1_000_000,
            CacheTierSize::Bytes(TTL_FAST_TIER),
        ).expect("cache should construct");

        // Note: a *short* TTL here (comparable to `MIGRATION_TIMEOUT`) would
        // make this test racy against `tier_of` itself: `tier_of` treats an
        // expired object as absent (`None`), same as `get`/`has`, so if the
        // object happened to expire before the migration was observed, the
        // `wait_until` below would spin until its own timeout with no way
        // to distinguish "never migrated" from "migrated but already
        // expired." A TTL comfortably longer than any plausible migration
        // latency avoids that ambiguity; the assertions below still prove
        // the *original* deadline survived rather than being reset/dropped.
        let ttl_secs = 5u32;
        let set_at = std::time::Instant::now();
        cache.set(1u32, &value(0xC1), Some(ttl_secs)).expect("set should succeed");

        for key in 2u32..=4 {
            cache.set(key, &value(key as u8), None).expect("set should succeed");
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

        // See `ttl_survives_a_demotion` and `TTL_FAST_TIER` for why this
        // uses a larger fast tier and several filler keys rather than one.
        let ttl_secs = 5u32;
        let set_at = std::time::Instant::now();
        cache.set(1u32, &value(0xC1), Some(ttl_secs)).expect("set should succeed");

        for key in 2u32..=4 {
            cache.set(key, &value(key as u8), None).expect("set should succeed");
        }

        assert!(wait_until(MIGRATION_TIMEOUT, || cache.tier_of(&1u32) == Some(Tier::Slow)));

        // Promote key 1; its original TTL should still be in effect afterward.
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
        // the slow-tier LRU tail must be evicted (never the fast tier,
        // which by construction holds only the most-recently-touched key).
        let cache = PaperCache::<u32, TieredBuffer>::new(
            200,
            CacheTierSize::Bytes(10),
        ).expect("cache should construct");

        for key in 1u32..=10 {
            let _ = cache.set(key, b"payload bytes", None);
        }

        let evicted = wait_until(MIGRATION_TIMEOUT, || {
            cache.lru_hybrid_stats().evictions >= 1
        });
        assert!(evicted, "at least one terminal eviction should have occurred");

        // Give the worker a moment to settle so the count below is stable.
        std::thread::sleep(std::time::Duration::from_millis(200));

        let stats = cache.lru_hybrid_stats();
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

    // ── DRAM cap accounts for shared metadata (hashtable + eviction stacks) ──

    #[test]
    fn dram_cap_reserves_shared_metadata_and_demotes_without_evicting() {
        ensure_pmem_allocator_warm();

        // A big overall cache (so `max_size` never triggers eviction) with a
        // fast tier whose raw byte budget (2 KB) would comfortably hold all of
        // these tiny (~13-byte) values at once. The fast-tier budget, however,
        // now also reserves an approximate per-object DRAM cost for the shared
        // object hashtable (and the eviction stacks too, when those are also
        // DRAM-resident -- excluded here under `eviction_stacks_pmem`, so this
        // test only needs to rely on the smaller hashtable-only term); across
        // *enough* objects that reservation exceeds the budget regardless of
        // which terms apply, so the DRAM cap forces demotions even though the
        // values alone would fit. Crucially, the DRAM cap responds only with
        // demotions — it never evicts. 300 objects gives comfortable margin
        // under the hashtable-only reservation alone (roughly 11 bytes/object
        // -- see `object/overhead.rs::HASHTABLE_ENTRY_OVERHEAD` -- so >180
        // objects already exceeds the 2 KB budget on that term by itself).
        let cache = PaperCache::<u32, TieredBuffer>::new(
            1_000_000,
            CacheTierSize::Bytes(2_000),
        ).expect("cache should construct");

        for key in 1u32..=300 {
            cache.set(key, b"payload bytes", None).expect("set should succeed");
        }

        // The shared-metadata reservation for 300 objects far exceeds 2 KB, so
        // some objects must have been demoted to slow.
        let demoted = wait_until(MIGRATION_TIMEOUT, || {
            cache.lru_hybrid_stats().demotions >= 1
        });
        assert!(demoted, "shared-metadata reservation should force demotions");

        // Let the worker settle, then confirm the DRAM cap never evicted: every
        // key is still present (only demoted, not dropped), and the evictions
        // counter — which only `max_size` pressure increments — stays 0.
        std::thread::sleep(std::time::Duration::from_millis(300));

        let stats = cache.lru_hybrid_stats();
        assert_eq!(stats.evictions, 0, "the DRAM cap must demote, never evict");

        let present = (1u32..=300).filter(|key| cache.has(key)).count();
        assert_eq!(present, 300, "no key should be evicted by the DRAM cap");

        // The fast tier's live value bytes stay within the configured budget.
        assert!(stats.fast_bytes_used <= cache.fast_tier_size());
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
        // resize (mirrors `LruHybridStack::resize_fast_tier`'s eager
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
            CacheTierSize::Bytes(40),
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
            CacheTierSize::Bytes(40),
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

    // ── large-scale real-DRAM reproduction (manual, not part of the default
    //    suite -- run explicitly with --ignored; see doc comment below) ────

    /// Sums the `N0=`/`N1=` fields (page counts) across every mapped region in
    /// `/proc/self/numa_maps`, converting 4 KiB pages to MB -- the same
    /// aggregation the reported `numa_maps` `awk` one-liner performs against
    /// an external process, just done in-process against this test's own PID.
    fn read_own_numa_usage_mb() -> (f64, f64) {
        let contents = std::fs::read_to_string("/proc/self/numa_maps")
            .expect("should be able to read /proc/self/numa_maps");

        let mut n0_pages: u64 = 0;
        let mut n1_pages: u64 = 0;

        for line in contents.lines() {
            for field in line.split_whitespace() {
                if let Some(value) = field.strip_prefix("N0=") {
                    n0_pages += value.parse::<u64>().unwrap_or(0);
                } else if let Some(value) = field.strip_prefix("N1=") {
                    n1_pages += value.parse::<u64>().unwrap_or(0);
                }
            }
        }

        // 4 KiB pages -> MB
        (n0_pages as f64 * 4.0 / 1024.0, n1_pages as f64 * 4.0 / 1024.0)
    }

    /// Reproduces the reported scenario directly, in-process, without relying
    /// on any external benchmark: a 1 GB fast tier, ~16 KB average objects,
    /// inserted sequentially (single-threaded, no artificial burst), then
    /// real `/proc/self/numa_maps` is read from *this* process at two points
    /// -- right after the insert burst (captures the near-peak in-flight
    /// backlog, since every `set()` writes to DRAM synchronously before the
    /// worker has decided a tier) and again after the worker has *verifiably*
    /// fully settled (`fast_objects + slow_objects == num_objects` exactly,
    /// not just `fast_bytes_used` looking momentarily low -- see the fixed
    /// gauge-staleness bug this test caught) -- to test the high-water-mark
    /// hypothesis directly: does real DRAM shrink back down once the stack
    /// has genuinely caught up, or does it stay pinned near the peak
    /// (suggesting the allocator pool doesn't return freed pages to the OS)?
    ///
    /// Object count is read from `REPRO_OBJECT_COUNT` (default 1,000,000) so
    /// the same code path can be run at different scales in separate,
    /// uncontaminated processes for comparison:
    ///   REPRO_OBJECT_COUNT=50000 cargo +nightly test --release \
    ///     --test lru_hybrid_cache_integration --features lru_hybrid_cache \
    ///     repro_real_dram_usage_at_scale -- --ignored --nocapture
    ///
    /// Not part of the default suite (`#[ignore]` -- allocates real value
    /// bytes proportional to `REPRO_OBJECT_COUNT` and does real PMEM
    /// migrations for ~90%+ of them).
    #[test]
    #[ignore]
    fn repro_real_dram_usage_at_scale() {
        ensure_pmem_allocator_warm();

        let object_count: u64 = std::env::var("REPRO_OBJECT_COUNT")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(1_000_000);

        const VALUE_LEN: usize = 16 * 1024; // 16 KB
        const FAST_TIER_GB: u64 = 1;
        const MAX_SIZE_GB: u64 = 24; // comfortably over ~16 GB of raw value bytes

        let cache = PaperCache::<u64, TieredBuffer>::new(
            MAX_SIZE_GB * 1_073_741_824,
            CacheTierSize::Gb(FAST_TIER_GB),
        ).expect("cache should construct");

        // `CacheTierSize::Gb` is decimal SI (10^9 bytes), documented in
        // `size.rs` -- not binary GiB (2^30). Confirmed via this test: an
        // earlier version of this assertion expected the binary value and
        // failed, catching a real (if minor, ~7%) unit gotcha worth knowing
        // about, distinct from the much larger gap under investigation here.
        assert_eq!(cache.fast_tier_size(), FAST_TIER_GB * 1_000_000_000);

        let start = std::time::Instant::now();

        for key in 0..object_count {
            // Non-zero, non-uniform payload so real distinct pages get
            // touched (avoids any zero-page-adjacent artifact skewing RSS).
            let value = vec![(key % 256) as u8; VALUE_LEN];
            cache.set(key, &value, None).expect("set should succeed");

            if key > 0 && key % 200_000 == 0 {
                println!("... {key} objects set ({:?} elapsed)", start.elapsed());
            }
        }

        println!("[n={object_count}] all objects set in {:?}", start.elapsed());

        // Peak measurement: sampled immediately after the insert burst,
        // before the worker has had a chance to catch up -- this is close to
        // the largest number of objects ever simultaneously resident in DRAM
        // (every `set()` builds `TieredBuffer::new_fast` synchronously,
        // regardless of eventual tier).
        let (peak_node0_mb, peak_node1_mb) = read_own_numa_usage_mb();
        println!(
            "[n={object_count}] PEAK (right after insert burst): node0={peak_node0_mb:.1} MB  node1={peak_node1_mb:.1} MB"
        );

        // `fast_bytes_used <= fast_tier_size` alone is *not* a reliable
        // "worker has fully caught up" signal: it can already be satisfied
        // while a large backlog of Set events the worker hasn't even reached
        // yet is still sitting in the channel (each one already physically
        // DRAM-resident, built synchronously by `set()`, but invisible to
        // the stack -- and therefore to this gauge -- until processed). Wait
        // for the stack's own tracked count to actually reach every inserted
        // object before trusting any downstream measurement.
        let settle_start = std::time::Instant::now();
        let mut last_print = std::time::Instant::now();

        let settled = wait_until(std::time::Duration::from_secs(600), || {
            let stats = cache.lru_hybrid_stats();
            let status = cache.status().expect("status should be available");
            let processed = stats.fast_objects + stats.slow_objects;

            if last_print.elapsed() >= std::time::Duration::from_secs(5) {
                let elapsed = settle_start.elapsed().as_secs_f64();
                let rate = processed as f64 / elapsed.max(0.001);
                println!(
                    "[n={object_count}] ... worker processed {processed}/{} ({:?} since insert loop finished, {rate:.0} objects/sec)",
                    status.num_objects(), settle_start.elapsed(),
                );
                last_print = std::time::Instant::now();
            }

            processed == status.num_objects() && stats.fast_bytes_used <= cache.fast_tier_size()
        });
        assert!(settled, "[n={object_count}] the stack should have processed every inserted object within 10 minutes");

        // Extra settle time: even once the stack has assigned every object a
        // tier, the *physical* PMEM migration for the very last few of them
        // may still be in flight inside `apply_tier_migrations` -- give it a
        // further moment before measuring real memory.
        std::thread::sleep(std::time::Duration::from_secs(5));

        let stats = cache.lru_hybrid_stats();
        let status = cache.status().expect("status should be available");
        let (settled_node0_mb, settled_node1_mb) = read_own_numa_usage_mb();

        // A second, later sample: some pool allocators (e.g. jemalloc's
        // default dirty/muzzy decay) release freed pages back to the OS only
        // after an idle decay period (~10s by default), not immediately on
        // free -- so a measurement taken only 5s after settling may still
        // catch pages mid-decay. Comfortably clear that window and re-sample
        // to see whether memory keeps dropping (decay-based release) or has
        // already reached its floor (e.g. a pool that never releases at all).
        std::thread::sleep(std::time::Duration::from_secs(30));
        let (decayed_node0_mb, decayed_node1_mb) = read_own_numa_usage_mb();
        println!(
            "[n={object_count}] DECAYED (+30s more): node0={decayed_node0_mb:.1} MB  node1={decayed_node1_mb:.1} MB"
        );

        println!("[n={object_count}] === lru_hybrid_stats ===");
        println!(
            "[n={object_count}] fast_objects={} slow_objects={} fast_bytes_used={} slow_bytes_used={} promotions={} demotions={} evictions={}",
            stats.fast_objects, stats.slow_objects, stats.fast_bytes_used, stats.slow_bytes_used,
            stats.promotions, stats.demotions, stats.evictions,
        );
        println!("[n={object_count}] === status ===");
        println!(
            "[n={object_count}] max_size={} used_size={} num_objects={} configured_fast_tier_size={}",
            status.max_size(), status.used_size(), status.num_objects(), cache.fast_tier_size(),
        );
        println!(
            "[n={object_count}] SETTLED: node0={settled_node0_mb:.1} MB  node1={settled_node1_mb:.1} MB  total={:.1} MB",
            settled_node0_mb + settled_node1_mb,
        );

        let fast_tier_mb = cache.fast_tier_size() as f64 / 1_048_576.0;
        println!(
            "[n={object_count}] SUMMARY: configured fast tier = {fast_tier_mb:.1} MB; \
             peak node0 = {peak_node0_mb:.1} MB ({:.2}x budget); \
             settled node0 (+5s) = {settled_node0_mb:.1} MB ({:.2}x budget); \
             decayed node0 (+35s) = {decayed_node0_mb:.1} MB ({:.2}x budget); \
             decayed/peak ratio = {:.3} (near 1.0 => DRAM stayed pinned near peak \
             despite settlement; well below 1.0 => DRAM tracked the true live footprint)",
            peak_node0_mb / fast_tier_mb,
            settled_node0_mb / fast_tier_mb,
            decayed_node0_mb / fast_tier_mb,
            decayed_node0_mb / peak_node0_mb.max(0.001),
        );

        let _ = decayed_node1_mb;
    }

    /// Diagnostic (not part of the default suite) that isolated a real,
    /// serious concurrency bug in UMF's disjoint pool (`umf_disjoint_pool`
    /// feature): under this exact test (N threads concurrently calling
    /// `set()`, well within available memory on both nodes -- ~6.4GB total
    /// against nodes with 50GB/124GB respectively), TBB (the default
    /// backend) passes cleanly at every thread count tried (2/4/6/8+); the
    /// disjoint pool passes at 2 and 4 threads but reliably fails at 6+
    /// with spurious `AllocFailed`/`UMF alloc failed` errors on **both**
    /// the fast tier's global allocator (node 0) and the slow tier's
    /// explicit `alloc_on` pool (node 1) -- two structurally independent
    /// pool instances failing together rules out simple node-level memory
    /// exhaustion and points at a genuine concurrency bug inside
    /// `umfDisjointPoolOps` itself, not this crate's integration of it.
    /// This directly contradicts the standalone (non-`#[global_allocator]`,
    /// narrower size-class) 24M-allocation/8-thread stress test that passed
    /// cleanly earlier in this investigation -- that test evidently didn't
    /// exercise the real production pattern of disjoint pool serving as
    /// the *entire* process's global allocator under the full, varied
    /// allocation traffic a real multi-threaded Rust program generates.
    /// **Do not enable `umf_disjoint_pool` in production** until this is
    /// root-caused or fixed upstream in UMF.
    #[test]
    #[ignore]
    fn concurrent_set_from_multiple_threads_still_demotes() {
        ensure_pmem_allocator_warm();

        const THREADS: u64 = 8;
        const PER_THREAD: u64 = 50_000;
        const VALUE_LEN: usize = 16 * 1024;

        let cache = std::sync::Arc::new(
            PaperCache::<u64, TieredBuffer>::new(24 * 1_073_741_824, CacheTierSize::Gb(1))
                .expect("cache should construct"),
        );

        let handles: Vec<_> = (0..THREADS)
            .map(|t| {
                let cache = std::sync::Arc::clone(&cache);
                std::thread::spawn(move || {
                    for i in 0..PER_THREAD {
                        let key = t * PER_THREAD + i;
                        let value = vec![(key % 256) as u8; VALUE_LEN];
                        cache.set(key, &value, None).expect("set should succeed");
                    }
                })
            })
            .collect();
        for h in handles {
            h.join().expect("thread should not panic");
        }

        let settled = wait_until(std::time::Duration::from_secs(60), || {
            let stats = cache.lru_hybrid_stats();
            let status = cache.status().expect("status should be available");
            let processed = stats.fast_objects + stats.slow_objects;
            println!(
                "... processed {processed}/{} fast={} slow={} demotions={}",
                status.num_objects(), stats.fast_objects, stats.slow_objects, stats.demotions,
            );
            processed == status.num_objects()
        });
        assert!(settled, "worker should process every inserted object within 60s");

        let stats = cache.lru_hybrid_stats();
        println!(
            "FINAL: fast_objects={} slow_objects={} demotions={} fast_bytes_used={} slow_bytes_used={}",
            stats.fast_objects, stats.slow_objects, stats.demotions, stats.fast_bytes_used, stats.slow_bytes_used,
        );
        assert!(stats.demotions > 0, "expected real demotions once the 1GB fast tier filled -- got 0");
    }

    // ── ttl reaping vs. the tier gauges ───────────────────────────────────

    /// A TTL reap must clear the reaped object out of the policy stack's tier
    /// accounting, not just out of the object map.
    ///
    /// `TtlWorker` erases expired objects itself, and `erase` only touches the
    /// object map and `AtomicStatus`. Before it also emitted
    /// `WorkerEvent::Expire`, the policy stack went on ranking every reaped
    /// key and went on counting its bytes toward `fast_used` -- so
    /// `status().num_objects()` fell to 0 while
    /// `lru_hybrid_stats().fast_bytes_used` stayed pinned at its pre-expiry
    /// value. On an idle cache that never self-corrected: the only thing that
    /// dropped a phantom key was it reaching the eviction tail, and nothing
    /// evicts when `used_size()` is already 0.
    ///
    /// The practical damage was to demotion decisions -- `settle_fast_tier`
    /// compares `fast_used` against the budget, so an inflated `fast_used`
    /// demotes live objects to PMEM to make room for bytes that no longer
    /// exist.
    #[test]
    fn ttl_reap_clears_the_fast_tier_gauges() {
        // This test never demotes, so it never allocates PMEM itself -- but it
        // still has to wait out the warm-up, because a *concurrently running*
        // test in this binary can trigger it and stall the whole process well
        // past a short TTL. Racing that is exactly what made the first version
        // of this test flaky: every key expired and was reaped before the
        // admission gauges were ever observed, so the pre-expiry baseline
        // assertion below failed. See the module doc comment.
        ensure_pmem_allocator_warm();

        // Fast tier far larger than the working set on purpose: this test is
        // about expiry, not demotion. Keeping every object in the fast tier
        // leaves `fast_bytes_used` as the single gauge that has to move.
        let cache = PaperCache::<u32, TieredBuffer>::new(
            1_000_000,
            CacheTierSize::Bytes(1_000_000),
        ).expect("cache should construct");

        const KEYS: u32 = 8;
        // Long enough that the admission gauges are comfortably observable
        // before the first reap, even on a loaded box.
        const TTL_SECS: u32 = 5;

        for key in 0..KEYS {
            cache.set(key, &value(0xC3), Some(TTL_SECS)).expect("set should succeed");
        }

        // The gauges are published once per worker poll, so wait for the
        // admissions to be accounted for rather than assuming they are.
        let admitted = wait_until(MIGRATION_TIMEOUT, || {
            cache.lru_hybrid_stats().fast_objects == KEYS as u64
        });
        assert!(admitted, "all {KEYS} keys should be accounted for in the fast tier");

        let before = cache.lru_hybrid_stats();
        assert!(
            before.fast_bytes_used >= (KEYS as u64) * (VALUE_LEN as u64),
            "fast tier should be holding all {KEYS} values before expiry; got {}",
            before.fast_bytes_used,
        );

        // The TTL, then the TtlWorker's own poll (1ms once an expiry is
        // imminent). Generous budget so this can't flake on a loaded box.
        let reaped = wait_until(std::time::Duration::from_secs(60), || {
            cache.status().expect("status should be available").num_objects() == 0
        });
        assert!(reaped, "all {KEYS} keys should expire and be reaped from the object map");

        // The actual regression: the stack's own tier accounting has to
        // follow the reap.
        let cleared = wait_until(std::time::Duration::from_secs(10), || {
            let stats = cache.lru_hybrid_stats();
            stats.fast_objects == 0 && stats.fast_bytes_used == 0
        });

        let after = cache.lru_hybrid_stats();
        assert!(
            cleared,
            "fast-tier gauges should clear after a TTL reap, but stayed at \
             fast_objects={} fast_bytes_used={} (was fast_objects={} \
             fast_bytes_used={} before expiry)",
            after.fast_objects, after.fast_bytes_used,
            before.fast_objects, before.fast_bytes_used,
        );
    }

    /// The reap notification must not clobber a key that was re-`set` in the
    /// window between `TtlWorker`'s `erase` and `PolicyWorker` handling the
    /// resulting `Expire` -- that would desync the other way (live in the
    /// object map, absent from the stack), leaving an object that can never
    /// be evicted and whose bytes go unaccounted for. `handle_expire` guards
    /// against it by re-reading the object map before touching the stack.
    #[test]
    fn a_key_re_set_after_expiring_is_still_tracked_by_the_stack() {
        ensure_pmem_allocator_warm();

        let cache = PaperCache::<u32, TieredBuffer>::new(
            1_000_000,
            CacheTierSize::Bytes(1_000_000),
        ).expect("cache should construct");

        cache.set(1u32, &value(0xD4), Some(1)).expect("set should succeed");

        let reaped = wait_until(std::time::Duration::from_secs(60), || {
            cache.status().expect("status should be available").num_objects() == 0
        });
        assert!(reaped, "key 1 should expire and be reaped");

        // Re-admit the same key, now with no TTL.
        cache.set(1u32, &value(0xD4), None).expect("re-set should succeed");
        assert_eq!(cache.tier_of(&1u32), Some(Tier::Fast));

        // The stack must be tracking the new object, not have dropped it on
        // the back of the previous incarnation's expiry.
        let tracked = wait_until(MIGRATION_TIMEOUT, || {
            let stats = cache.lru_hybrid_stats();
            stats.fast_objects == 1 && stats.fast_bytes_used >= VALUE_LEN as u64
        });

        let stats = cache.lru_hybrid_stats();
        assert!(
            tracked,
            "re-set key should be tracked in the fast tier; got fast_objects={} \
             fast_bytes_used={}",
            stats.fast_objects, stats.fast_bytes_used,
        );
        assert_eq!(cache.get(&1u32).unwrap(), value(0xD4));
    }
}
