/*
 * Copyright (c) Kia Shakiba
 *
 * This source code is licensed under the GNU AGPLv3 license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! Integration tests for the `s3_fifo_hybrid_cache` feature.
//!
//! Run with nightly (required for `allocator_api` via `key_value_pmem`):
//!   cargo +nightly test --test s3_fifo_hybrid_cache_integration --features s3_fifo_hybrid_cache
//!
//! Same one-`PaperCache<K, TieredBuffer>` architecture as the other
//! hybrids — `tier_of` reads the tier directly off the single object map.
//!
//! Structurally closest to `two_q_hybrid_cache_integration.rs`: admission
//! always lands in the **slow** tier (the one-access queue), a real
//! synchronous PMEM write on every `set()`, so — same as that file — every
//! test here pays (or waits out) the one-time PMEM pool warm-up cost. See
//! that file's module doc for the ~45s warm-up caveat.
//!
//! What is tested, beyond what `two_q_hybrid_cache_integration.rs` already
//! covers structurally:
//!   * Admission always lands in the slow tier (the one-access queue)
//!   * A re-access to a one-access-queue object promotes it EAGERLY (same
//!     as 2Q) straight to the main queue's fast tier
//!   * A one-access object that ages out without a second access is evicted
//!   * Unlike 2Q's LRU main queue: a plain access on an already-Fast main-
//!     queue object does NOT reorder it or produce a migration -- it only
//!     sets a reference bit, invisible until eviction time
//!   * Fast-tier pressure demotes unconditionally (independent of the
//!     reference bit)
//!   * The signature S3-FIFO/CLOCK behavior: an accessed key sitting at the
//!     main queue's tail gets a second chance (promoted back to fast)
//!     instead of being evicted, when real eviction pressure reaches it
//!   * Terminal eviction prefers the one-access queue over the main queue
//!   * TTL set before a demotion/promotion is still correctly enforced after
//!   * `set_fast_tier_size` / `resize` (which rescales the one-access
//!     queue's byte budget) take effect at runtime
//!   * Zero/invalid/tiny fast-tier-size and ratio edge cases

#[cfg(feature = "s3_fifo_hybrid_cache")]
mod s3_fifo_hybrid_cache_tests {
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
    /// before a test's own timing-sensitive assertions begin. Same shape as
    /// `two_q_hybrid_cache_integration.rs`'s helper -- admission itself
    /// calls `TieredBuffer::new_slow` directly (synchronous), so by the
    /// time `set()` returns, the allocator is warm.
    fn ensure_pmem_allocator_warm() {
        let cache = PaperCache::<u32, TieredBuffer>::new(1_000_000, CacheTierSize::Bytes(1_000_000), 1.0)
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
            1_000_000,
            CacheTierSize::Bytes(1_000_000),
            1.0,
        ).expect("cache should construct");

        cache.set(1u32, b"hello world", None).expect("set should succeed");

        // Admission is synchronous (TieredBuffer::new_slow is built
        // directly inside set()), so this doesn't need wait_until.
        assert_eq!(cache.tier_of(&1u32), Some(Tier::Slow));
        assert_eq!(cache.get(&1u32).unwrap(), b"hello world");
    }

    // ── eager one-access promotion ───────────────────────────────────────

    #[test]
    fn reaccessing_a_one_access_key_promotes_it_eagerly_to_fast_tier() {
        ensure_pmem_allocator_warm();

        let cache = PaperCache::<u32, TieredBuffer>::new(
            1_000_000,
            CacheTierSize::Bytes(1_000_000),
            1.0,
        ).expect("cache should construct");

        cache.set(1u32, b"hello world", None).expect("set should succeed");
        assert_eq!(cache.tier_of(&1u32), Some(Tier::Slow));

        cache.get(&1u32).expect("get should succeed");

        let promoted = wait_until(MIGRATION_TIMEOUT, || {
            cache.tier_of(&1u32) == Some(Tier::Fast)
        });
        assert!(promoted, "key should have promoted to the fast tier after a re-access");

        let stats = cache.s3_fifo_hybrid_stats();
        assert!(stats.promotions >= 1);
    }

    // ── one-access eviction ──────────────────────────────────────────────

    #[test]
    fn one_access_key_aging_out_without_reaccess_is_evicted() {
        ensure_pmem_allocator_warm();

        let cache = PaperCache::<u32, TieredBuffer>::new(
            200,
            CacheTierSize::Bytes(200),
            1.0,
        ).expect("cache should construct");

        cache.set(1u32, b"first value 123", None).expect("set should succeed");
        assert_eq!(cache.tier_of(&1u32), Some(Tier::Slow));

        // Never re-accessed; admitting more keys should eventually evict it
        // via the global capacity-exhausted eviction loop.
        for key in 2u32..=10 {
            let _ = cache.set(key, b"payload bytes", None);
        }

        let evicted = wait_until(MIGRATION_TIMEOUT, || !cache.has(&1u32));
        assert!(evicted, "key 1 should have aged out of the one-access queue and been evicted");
        assert_eq!(cache.tier_of(&1u32), None);

        let stats = cache.s3_fifo_hybrid_stats();
        assert!(stats.evictions >= 1);
    }

    #[test]
    fn one_access_capacity_pressure_evicts_before_global_max_size_is_reached() {
        ensure_pmem_allocator_warm();

        // Overall cache is huge, but the ratio caps the one-access queue's
        // own byte budget tightly (0.00004 * 1_000_000 = 40 bytes, fitting
        // one ~15-byte value) -- eviction should trigger from that pressure
        // alone, nowhere near the global max_size.
        let cache = PaperCache::<u32, TieredBuffer>::new(
            1_000_000,
            CacheTierSize::Bytes(1_000_000),
            0.00004,
        ).expect("cache should construct");

        cache.set(1u32, b"first value 123", None).expect("set should succeed");
        assert_eq!(cache.tier_of(&1u32), Some(Tier::Slow));

        cache.set(2u32, b"second value 45", None).expect("set should succeed");

        let evicted = wait_until(MIGRATION_TIMEOUT, || !cache.has(&1u32));
        assert!(evicted, "key 1 should have been evicted by one-access capacity pressure");

        let status = cache.status().unwrap();
        assert!(
            status.used_size() < 1_000,
            "overall usage should be nowhere near max_size, confirming this was \
             one-access-capacity pressure, not the global eviction loop",
        );
    }

    // ── main-queue behavior: unconditional aging, no reorder-on-access ────

    #[test]
    fn fast_tier_pressure_within_main_queue_demotes_the_oldest() {
        ensure_pmem_allocator_warm();

        let cache = PaperCache::<u32, TieredBuffer>::new(
            1_000_000,
            CacheTierSize::Bytes(40),
            1.0,
        ).expect("cache should construct");

        cache.set(1u32, b"first value 123", None).expect("set should succeed");
        cache.set(2u32, b"second value 45", None).expect("set should succeed");
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
    fn a_plain_access_on_a_fast_main_queue_key_does_not_migrate_or_reorder() {
        ensure_pmem_allocator_warm();

        let cache = PaperCache::<u32, TieredBuffer>::new(
            1_000_000,
            CacheTierSize::Bytes(1_000_000),
            1.0,
        ).expect("cache should construct");

        cache.set(1u32, b"first value 123", None).expect("set should succeed");
        cache.get(&1u32).expect("get should succeed");
        assert!(wait_until(MIGRATION_TIMEOUT, || cache.tier_of(&1u32) == Some(Tier::Fast)));

        let promotions_before = cache.s3_fifo_hybrid_stats().promotions;

        // A second access while already Fast should be a pure no-op from
        // the tiering machinery's point of view -- unlike two_q_hybrid_cache
        // (LRU main queue), this must never register as another migration.
        cache.get(&1u32).expect("get should succeed");
        std::thread::sleep(std::time::Duration::from_millis(300));

        assert_eq!(cache.tier_of(&1u32), Some(Tier::Fast));
        assert_eq!(cache.s3_fifo_hybrid_stats().promotions, promotions_before);
    }

    #[test]
    fn promotion_can_cascade_a_demotion_and_values_survive() {
        ensure_pmem_allocator_warm();

        let cache = PaperCache::<u32, TieredBuffer>::new(
            1_000_000,
            CacheTierSize::Bytes(40),
            1.0,
        ).expect("cache should construct");

        cache.set(1u32, b"first value 123", None).expect("set should succeed");
        cache.set(2u32, b"second value 45", None).expect("set should succeed");

        cache.get(&1u32).expect("get should succeed");
        assert!(wait_until(MIGRATION_TIMEOUT, || cache.tier_of(&1u32) == Some(Tier::Fast)));

        cache.get(&2u32).expect("get should succeed");
        assert!(wait_until(MIGRATION_TIMEOUT, || cache.tier_of(&2u32) == Some(Tier::Fast)));
        assert!(wait_until(MIGRATION_TIMEOUT, || cache.tier_of(&1u32) == Some(Tier::Slow)));

        // Both values remain intact and reachable regardless of tier.
        assert_eq!(cache.get(&1u32).unwrap(), b"first value 123");
        assert_eq!(cache.get(&2u32).unwrap(), b"second value 45");
    }

    // ── the signature mechanic: second chance at eviction time ────────────

    #[test]
    fn an_accessed_key_at_the_main_queue_tail_gets_a_second_chance_instead_of_eviction() {
        ensure_pmem_allocator_warm();

        // Generous initial max_size so setting up and promoting both keys
        // carries zero risk of a premature eviction.
        let cache = PaperCache::<u32, TieredBuffer>::new(
            1_000_000,
            CacheTierSize::Bytes(40),
            1.0,
        ).expect("cache should construct");

        cache.set(1u32, b"payload bytes A", None).expect("set should succeed");
        cache.get(&1u32).expect("get should succeed");
        assert!(wait_until(MIGRATION_TIMEOUT, || cache.tier_of(&1u32) == Some(Tier::Fast)));

        cache.set(2u32, b"payload bytes B", None).expect("set should succeed");
        cache.get(&2u32).expect("get should succeed");
        assert!(wait_until(MIGRATION_TIMEOUT, || cache.tier_of(&2u32) == Some(Tier::Fast)));
        assert!(wait_until(MIGRATION_TIMEOUT, || cache.tier_of(&1u32) == Some(Tier::Slow)));

        // Re-access key 1 while it's slow -- lazily sets its reference bit;
        // it stays Slow (no reorder/migration from a mere access).
        cache.get(&1u32).expect("get should succeed");
        std::thread::sleep(std::time::Duration::from_millis(300));
        assert_eq!(cache.tier_of(&1u32), Some(Tier::Slow));

        // Shrink max_size well below both keys' combined accounted size.
        // Deliberately NOT using a new filler `set()` as the trigger here:
        // a filler admits into (and, alone, evicts right back out of) the
        // one-access queue in a single pass, self-cancelling its own
        // pressure before it can ever reach the main queue -- whether it
        // reaches the main queue at all becomes a race on how many `set()`s
        // happen to land in the same worker-loop batch. `resize()` adds no
        // new one-access entry at all: with the one-access queue already
        // empty (both keys left it when promoted), the resulting eviction
        // pressure is forced straight into the main queue's tail
        // deterministically, regardless of event-batching timing.
        cache.resize(180).expect("resize should succeed");

        let survived_and_promoted = wait_until(MIGRATION_TIMEOUT, || {
            cache.has(&1u32) && cache.tier_of(&1u32) == Some(Tier::Fast)
        });
        assert!(
            survived_and_promoted,
            "key 1 should have been given a second chance and promoted back to fast, \
             not evicted -- final tier: {:?}, present: {}",
            cache.tier_of(&1u32), cache.has(&1u32),
        );

        assert_eq!(cache.get(&1u32).unwrap(), b"payload bytes A");
        assert!(!cache.has(&2u32), "key 2 should have been evicted in key 1's place");
    }

    // ── TTL ───────────────────────────────────────────────────────────────

    // Same lesson already documented for the other hybrids in CLAUDE.md: a
    // fast-tier capacity sized only for `None`-ttl objects is too tight for
    // a *single* ttl'd object (fixed TTL bookkeeping overhead). Use a
    // capacity comfortably larger than one ttl'd object, and force demotion
    // pressure with several small filler keys instead of a second
    // same-sized key.
    const TTL_FAST_TIER: u64 = 200;

    #[test]
    fn ttl_survives_a_demotion() {
        ensure_pmem_allocator_warm();

        let cache = PaperCache::<u32, TieredBuffer>::new(
            1_000_000,
            CacheTierSize::Bytes(TTL_FAST_TIER),
            1.0,
        ).expect("cache should construct");

        let ttl_secs = 5u32;
        let set_at = std::time::Instant::now();
        cache.set(1u32, b"first value 123", Some(ttl_secs)).expect("set should succeed");

        for key in 2u32..=6 {
            cache.set(key, b"filler bytes", None).expect("set should succeed");
        }

        // Promote key 1 to fast first.
        cache.get(&1u32).expect("get should succeed");
        assert!(wait_until(MIGRATION_TIMEOUT, || cache.tier_of(&1u32) == Some(Tier::Fast)));

        // Promote the fillers one by one until fast-tier pressure demotes
        // key 1 (the oldest fast key) back down.
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
            1_000_000,
            CacheTierSize::Bytes(1_000_000),
            1.0,
        ).expect("cache should construct");

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
    fn terminal_eviction_prefers_one_access_queue_over_main_queue() {
        ensure_pmem_allocator_warm();

        let cache = PaperCache::<u32, TieredBuffer>::new(
            200,
            CacheTierSize::Bytes(200),
            1.0,
        ).expect("cache should construct");

        // Promote key 1 into the main queue (fast) so it's "proven" and
        // should survive eviction pressure that one-access objects wouldn't.
        cache.set(1u32, b"payload bytes 1", None).expect("set should succeed");
        cache.get(&1u32).expect("get should succeed");
        assert!(wait_until(MIGRATION_TIMEOUT, || cache.tier_of(&1u32) == Some(Tier::Fast)));

        for key in 2u32..=10 {
            let _ = cache.set(key, b"payload bytes", None);
        }

        let evicted = wait_until(MIGRATION_TIMEOUT, || {
            cache.s3_fifo_hybrid_stats().evictions >= 1
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

        let cache = PaperCache::<u32, TieredBuffer>::new(
            1_000_000,
            CacheTierSize::Bytes(1_000_000),
            1.0,
        ).expect("cache should construct");

        cache.set(1u32, b"first value 123", None).expect("set should succeed");
        cache.get(&1u32).expect("get should succeed");
        assert!(wait_until(MIGRATION_TIMEOUT, || cache.tier_of(&1u32) == Some(Tier::Fast)));
        assert_eq!(cache.fast_tier_size(), 1_000_000);

        cache.set_fast_tier_size(CacheTierSize::Bytes(1)).expect("resize should succeed");
        assert_eq!(cache.fast_tier_size(), 1);

        let demoted = wait_until(MIGRATION_TIMEOUT, || cache.tier_of(&1u32) == Some(Tier::Slow));
        assert!(demoted, "shrinking the fast tier should demote the promoted key");
    }

    #[test]
    fn resize_rescales_one_access_capacity() {
        ensure_pmem_allocator_warm();

        let cache = PaperCache::<u32, TieredBuffer>::new(1_000, CacheTierSize::Bytes(1_000), 0.5)
            .expect("cache should construct");
        // one_access_capacity = 0.5 * 1_000 = 500

        cache.set(1u32, b"first value 123", None).expect("set should succeed");
        assert!(cache.has(&1u32));

        // Shrink overall max_size drastically -> one_access_capacity
        // rescales (proportionally, via the ratio) down to a tiny budget
        // -> key 1 should be evicted.
        cache.resize(10).expect("resize should succeed");

        let evicted = wait_until(MIGRATION_TIMEOUT, || !cache.has(&1u32));
        assert!(evicted, "shrinking max_size should rescale one_access_capacity and evict");
    }

    // ── edge cases ────────────────────────────────────────────────────────

    #[test]
    fn zero_fast_tier_size_is_rejected() {
        let result = PaperCache::<u32, TieredBuffer>::new(1_000, CacheTierSize::Bytes(0), 0.5);
        assert!(matches!(result, Err(CacheError::InvalidFastTierSize)));
    }

    #[test]
    fn fast_tier_size_exceeding_max_size_is_rejected() {
        let result = PaperCache::<u32, TieredBuffer>::new(1_000, CacheTierSize::Bytes(2_000), 0.5);
        assert!(matches!(result, Err(CacheError::InvalidFastTierSize)));
    }

    #[test]
    fn zero_max_size_is_rejected() {
        let result = PaperCache::<u32, TieredBuffer>::new(0, CacheTierSize::Bytes(100), 0.5);
        assert!(matches!(result, Err(CacheError::ZeroCacheSize)));
    }

    #[test]
    fn invalid_one_access_ratio_is_rejected() {
        assert!(matches!(
            PaperCache::<u32, TieredBuffer>::new(1_000, CacheTierSize::Bytes(500), 1.5),
            Err(CacheError::InvalidPolicy),
        ));

        assert!(matches!(
            PaperCache::<u32, TieredBuffer>::new(1_000, CacheTierSize::Bytes(500), -0.1),
            Err(CacheError::InvalidPolicy),
        ));
    }

    #[test]
    fn tiny_fast_tier_still_allows_one_access_admission() {
        ensure_pmem_allocator_warm();

        // fast_tier_size doesn't affect one-access admission at all (it
        // only governs the main queue's split), so even a 1-byte fast tier
        // should admit and store a real value fine.
        let cache = PaperCache::<u32, TieredBuffer>::new(
            1_000_000,
            CacheTierSize::Bytes(1),
            1.0,
        ).expect("cache should construct");

        cache.set(1u32, b"a value", None).expect("set should succeed");
        assert_eq!(cache.tier_of(&1u32), Some(Tier::Slow));
        assert_eq!(cache.get(&1u32).unwrap(), b"a value");
    }

    #[test]
    fn del_removes_key_from_whichever_tier_it_is_in() {
        ensure_pmem_allocator_warm();

        let cache = PaperCache::<u32, TieredBuffer>::new(
            1_000_000,
            CacheTierSize::Bytes(1_000_000),
            1.0,
        ).expect("cache should construct");

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
            1_000_000,
            CacheTierSize::Bytes(1_000_000),
            1.0,
        ).expect("cache should construct");

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
