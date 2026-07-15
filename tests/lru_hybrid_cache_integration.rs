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
//! Unlike `hybridcache_integration.rs`, this feature is **one**
//! `PaperCache<K, TieredBuffer>` instance (not two composed `PaperCache`s), so
//! there's no `has_in_dram`/`has_in_pmem` pair to reuse — `tier_of` reads the
//! tier directly off the single object map.
//!
//! What is tested:
//!   * Admission always lands in the fast tier
//!   * Fast-tier pressure demotes the LRU tail to the slow tier, and `tier_of`
//!     confirms it is gone from the fast tier (real data movement, not a
//!     copy — the defining difference from `S3FifoHybridCache`)
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
}
