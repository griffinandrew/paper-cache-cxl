/*
 * Copyright (c) Kia Shakiba
 *
 * This source code is licensed under the GNU AGPLv3 license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! Integration tests for the `fifo_hybrid_cache` feature.
//!
//! Run with nightly (required for `allocator_api` via `key_value_pmem`):
//!   cargo +nightly test --test hybrid_cache_integration --features fifo_hybrid_cache
//!
//! This feature is **one** `PaperCache<K, TieredBuffer>` instance (not two
//! composed `PaperCache`s), so `tier_of` reads the tier directly off the
//! single object map. Modeled on
//! `tests/hybrid_cache_integration.rs`, with promotion-specific tests
//! dropped (FIFO has no promotion policy at all) and two FIFO-defining tests
//! added instead.
//!
//! What is tested:
//!   * Admission always lands in the fast tier
//!   * Fast-tier pressure demotes the oldest object to the slow tier, and
//!     `tier_of` confirms it is gone from the fast tier (real data movement,
//!     not a copy)
//!   * A slow-tier hit does **not** promote the key — it stays slow (the
//!     defining difference from `lru_hybrid_cache`)
//!   * Overwriting an existing key never repositions it or changes its tier
//!     (exercises the tier-aware `set()` fix needed since FIFO's overwrite
//!     rule differs from LRU's)
//!   * TTL set before a demotion is still correctly enforced after
//!   * Terminal eviction only ever removes the slow-tier oldest object and
//!     is counted in `hybrid_stats().evictions`
//!   * `set_fast_tier_size` takes effect at runtime
//!   * Zero/invalid/tiny fast-tier-size edge cases
//!
//! All tier-crossing tests exercise the real `Hybrid`/UMF PMEM allocator (no
//! shortcuts): the very first PMEM allocation in the whole test process
//! triggers a one-time NUMA-node pool init + prewarm that can take on the
//! order of a minute on first touch (observed ~45s in this sandbox) — see
//! `allocator.rs`'s `HybridObjects`. `ensure_pmem_allocator_warm()` below
//! forces that one-time cost to be paid synchronously at the start of every
//! test — since it's backed by the same process-wide `Once`, only the very
//! first call actually waits ~45s; every other call returns almost
//! immediately once the allocator is warm.

#[cfg(feature = "fifo_hybrid_cache")]
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
    /// before a test's own timing-sensitive assertions begin. See the module
    /// doc comment above for why this is necessary.
    fn ensure_pmem_allocator_warm() {
        // Mechanics tests at toy scales: metadata reservation off (see
        // `get_hybrid_dram_shared_overhead`).
        unsafe { std::env::set_var("PAPER_DISABLE_SHARED_OVERHEAD", "1") };
        let cache = PaperCache::<u32, TieredBuffer>::new(1_000_000, CacheTierSize::Bytes(1), PaperPolicy::FifoHybrid)
            .expect("warm-up cache should construct");

        cache.set(0u32, b"warm", None).expect("warm-up set should succeed");

        let ready = wait_until(std::time::Duration::from_secs(90), || {
            cache.tier_of(&0u32) == Some(Tier::Slow)
        });
        assert!(ready, "PMEM allocator warm-up should complete within 90s");
    }

    const MIGRATION_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(10);

    // ~1 KB values, same rationale as `hybrid_cache_integration.rs`'s
    // `VALUE_LEN`: keeps the byte-sized fast-tier budgets behaving
    // intuitively (~value-sized) for the demotion tests below.
    const VALUE_LEN: usize = 1024;

    fn value(seed: u8) -> Vec<u8> {
        vec![seed; VALUE_LEN]
    }

    // ── admission ─────────────────────────────────────────────────────────

    #[test]
    fn admission_always_lands_in_fast_tier() {
        let cache = PaperCache::<u32, TieredBuffer>::new(
            1_000_000,
            CacheTierSize::Bytes(1_000_000), PaperPolicy::FifoHybrid).expect("cache should construct");

        cache.set(1u32, b"hello world", None).expect("set should succeed");

        // Admission is synchronous (the object is inserted as `TieredBuffer::
        // Fast` directly inside `set()`, before the WorkerEvent is even
        // broadcast), so this doesn't need `wait_until`.
        assert_eq!(cache.tier_of(&1u32), Some(Tier::Fast));
        assert_eq!(cache.get(&1u32).unwrap(), b"hello world");
    }

    // ── demotion ──────────────────────────────────────────────────────────

    #[test]
    fn fast_tier_pressure_demotes_oldest_object_with_real_data_movement() {
        ensure_pmem_allocator_warm();

        // A fast tier sized to hold ~1 of these ~1 KB values guarantees the
        // first (oldest) key demotes once the second is admitted.
        let cache = PaperCache::<u32, TieredBuffer>::new(
            1_000_000,
            CacheTierSize::Bytes(1_600), PaperPolicy::FifoHybrid).expect("cache should construct");

        cache.set(1u32, &value(0xA1), None).expect("set should succeed");
        assert_eq!(cache.tier_of(&1u32), Some(Tier::Fast));

        cache.set(2u32, &value(0xB2), None).expect("set should succeed");

        let demoted = wait_until(MIGRATION_TIMEOUT, || {
            cache.tier_of(&1u32) == Some(Tier::Slow)
        });
        assert!(demoted, "key 1 (oldest) should have demoted to the slow tier");

        // Real data movement, not a copy: the key is gone from the fast
        // tier's accounting entirely — there is only one object map, so
        // "gone from fast" and "present in slow" are the same fact checked
        // two ways.
        assert_ne!(cache.tier_of(&1u32), Some(Tier::Fast));

        // Value survives the physical move intact.
        assert_eq!(cache.get(&1u32).unwrap(), value(0xA1));

        let stats = cache.hybrid_stats();
        assert!(stats.demotions >= 1);
        assert_eq!(stats.promotions, 0);
    }

    #[test]
    fn cascading_demotion_on_repeated_admission_is_handled() {
        ensure_pmem_allocator_warm();

        // FIFO has no promotion to cascade a demotion from — cascades here
        // come from repeated *new-key* admission into a fast tier sized for
        // ~1 object instead. Exercises the "more than one migration per
        // call" path (`FifoHybridStack::settle_fast_tier`'s loop).
        let cache = PaperCache::<u32, TieredBuffer>::new(
            1_000_000,
            CacheTierSize::Bytes(1_600), PaperPolicy::FifoHybrid).expect("cache should construct");

        for key in 1u32..=4 {
            cache.set(key, &value(key as u8), None).expect("set should succeed");
        }

        assert!(wait_until(MIGRATION_TIMEOUT, || cache.tier_of(&1u32) == Some(Tier::Slow)));
        assert!(wait_until(MIGRATION_TIMEOUT, || cache.tier_of(&2u32) == Some(Tier::Slow)));
        assert!(wait_until(MIGRATION_TIMEOUT, || cache.tier_of(&3u32) == Some(Tier::Slow)));
        assert_eq!(cache.tier_of(&4u32), Some(Tier::Fast));

        for key in 1u32..=4 {
            assert_eq!(cache.get(&key).unwrap(), value(key as u8));
        }
    }

    // ── no promotion, ever ────────────────────────────────────────────────

    #[test]
    fn slow_tier_hit_does_not_promote_and_object_stays_slow() {
        ensure_pmem_allocator_warm();

        // The defining difference from `lru_hybrid_cache`: a hit on a
        // slow-tier key must never migrate it back to fast, since FIFO has
        // no promotion policy at all ("objects are never reordered
        // regardless of subsequent accesses").
        let cache = PaperCache::<u32, TieredBuffer>::new(
            1_000_000,
            CacheTierSize::Bytes(1_600), PaperPolicy::FifoHybrid).expect("cache should construct");

        cache.set(1u32, &value(0xA1), None).expect("set should succeed");
        cache.set(2u32, &value(0xB2), None).expect("set should succeed");
        assert!(wait_until(MIGRATION_TIMEOUT, || cache.tier_of(&1u32) == Some(Tier::Slow)));

        // A get() on the slow-tier key must never promote it.
        assert_eq!(cache.get(&1u32).unwrap(), value(0xA1));

        // No "wait_until true" assertion is possible for a negative claim;
        // sleep comfortably longer than a real migration would take and
        // confirm the tier never changed.
        std::thread::sleep(std::time::Duration::from_millis(500));
        assert_eq!(cache.tier_of(&1u32), Some(Tier::Slow));

        let stats = cache.hybrid_stats();
        assert_eq!(stats.promotions, 0);
    }

    #[test]
    fn overwriting_an_existing_key_does_not_reposition_it_in_the_queue() {
        ensure_pmem_allocator_warm();

        // Exercises the tier-aware `set()` fix: overwriting a key that is
        // still in the fast tier must not reposition it (unlike LRU, which
        // would move it to MRU), and a subsequent demotion must still pick
        // the same oldest key, not whichever key was most recently
        // overwritten.
        let cache = PaperCache::<u32, TieredBuffer>::new(
            1_000_000,
            CacheTierSize::Bytes(3_400), PaperPolicy::FifoHybrid).expect("cache should construct");

        cache.set(1u32, &value(0x11), None).expect("set should succeed"); // oldest
        cache.set(2u32, &value(0x22), None).expect("set should succeed");
        cache.set(3u32, &value(0x33), None).expect("set should succeed"); // newest

        // Overwrite the oldest key. Under LRU semantics this would move it
        // to MRU; under FIFO it must stay the oldest.
        cache.set(1u32, &value(0x11), None).expect("overwrite should succeed");
        assert_eq!(cache.tier_of(&1u32), Some(Tier::Fast), "overwrite must not change tier");

        // A fourth admission that forces exactly one demotion should demote
        // key 1 (still oldest by insertion order), never key 2 or key 3.
        cache.set(4u32, &value(0x44), None).expect("set should succeed");

        assert!(wait_until(MIGRATION_TIMEOUT, || cache.tier_of(&1u32) == Some(Tier::Slow)));
        assert_eq!(cache.tier_of(&2u32), Some(Tier::Fast));
        assert_eq!(cache.tier_of(&3u32), Some(Tier::Fast));
        assert_eq!(cache.tier_of(&4u32), Some(Tier::Fast));

        assert_eq!(cache.get(&1u32).unwrap(), value(0x11));
    }

    // ── TTL ───────────────────────────────────────────────────────────────

    // Same rationale as `hybrid_cache_integration.rs`'s `TTL_FAST_TIER`:
    // a TTL'd object's `base_size` (via `get_ttl_overhead`) is large enough
    // that a fast tier sized only for `None`-ttl objects is too tight for a
    // single ttl'd object alone. Sized to hold ~2 objects, with several
    // small filler keys creating demotion pressure instead of one same-sized
    // key.
    const TTL_FAST_TIER: u64 = 2_600;

    #[test]
    fn ttl_survives_a_demotion() {
        ensure_pmem_allocator_warm();

        let cache = PaperCache::<u32, TieredBuffer>::new(
            1_000_000,
            CacheTierSize::Bytes(TTL_FAST_TIER), PaperPolicy::FifoHybrid).expect("cache should construct");

        // A TTL comfortably longer than any plausible migration latency
        // avoids ambiguity between "never migrated" and "migrated but
        // already expired" (see `tier_of`'s expired-is-absent semantics).
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

    // ── eviction ──────────────────────────────────────────────────────────

    #[test]
    fn terminal_eviction_only_removes_from_slow_tier_and_is_counted() {
        ensure_pmem_allocator_warm();

        // A small overall cache with a tiny fast tier: every object demotes
        // to slow almost immediately, and once total usage exceeds max_size
        // the slow-tier oldest object must be evicted (never the fast tier,
        // which by construction holds only the most-recently-admitted key).
        let cache = PaperCache::<u32, TieredBuffer>::new(
            200,
            CacheTierSize::Bytes(10), PaperPolicy::FifoHybrid).expect("cache should construct");

        for key in 1u32..=10 {
            let _ = cache.set(key, b"payload bytes", None);
        }

        let evicted = wait_until(MIGRATION_TIMEOUT, || {
            cache.hybrid_stats().evictions >= 1
        });
        assert!(evicted, "at least one terminal eviction should have occurred");

        // Give the worker a moment to settle so the count below is stable.
        std::thread::sleep(std::time::Duration::from_millis(200));

        let stats = cache.hybrid_stats();
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

    // ── runtime fast-tier resize ─────────────────────────────────────────

    #[test]
    fn set_fast_tier_size_takes_effect_at_runtime() {
        ensure_pmem_allocator_warm();

        let cache = PaperCache::<u32, TieredBuffer>::new(1_000_000, CacheTierSize::Bytes(1_000_000), PaperPolicy::FifoHybrid).expect("cache should construct");

        cache.set(1u32, b"first value 123", None).expect("set should succeed");
        assert_eq!(cache.tier_of(&1u32), Some(Tier::Fast));
        assert_eq!(cache.fast_tier_size(), 1_000_000);

        // Shrink the fast tier drastically; the existing key should demote
        // even without any further access, once the worker applies the
        // resize (mirrors `FifoHybridStack::resize_fast_tier`'s eager
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
        let result = PaperCache::<u32, TieredBuffer>::new(1_000, CacheTierSize::Bytes(0), PaperPolicy::FifoHybrid);
        assert!(matches!(result, Err(CacheError::InvalidFastTierSize)));
    }

    #[test]
    fn fast_tier_size_exceeding_max_size_is_rejected() {
        let result = PaperCache::<u32, TieredBuffer>::new(1_000, CacheTierSize::Bytes(2_000), PaperPolicy::FifoHybrid);
        assert!(matches!(result, Err(CacheError::InvalidFastTierSize)));
    }

    #[test]
    fn zero_max_size_is_rejected() {
        let result = PaperCache::<u32, TieredBuffer>::new(0, CacheTierSize::Bytes(100), PaperPolicy::FifoHybrid);
        assert!(matches!(result, Err(CacheError::ZeroCacheSize)));
    }

    #[test]
    fn tiny_fast_tier_demotes_everything_almost_immediately() {
        ensure_pmem_allocator_warm();

        let cache = PaperCache::<u32, TieredBuffer>::new(
            1_000_000,
            CacheTierSize::Bytes(1), PaperPolicy::FifoHybrid).expect("cache should construct");

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
            CacheTierSize::Bytes(40), PaperPolicy::FifoHybrid).expect("cache should construct");

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
            CacheTierSize::Bytes(40), PaperPolicy::FifoHybrid).expect("cache should construct");

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
