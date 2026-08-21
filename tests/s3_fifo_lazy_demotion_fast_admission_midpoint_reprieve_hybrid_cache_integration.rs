/*
 * Copyright (c) Kia Shakiba
 *
 * This source code is licensed under the GNU AGPLv3 license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! Integration tests for the
//! `s3_fifo_lazy_demotion_fast_admission_midpoint_reprieve_hybrid_cache`
//! feature.
//!
//! Run with nightly (required for `allocator_api` via `key_value_pmem`):
//!   cargo +nightly test --test hybrid_cache_integration --features s3_fifo_lazy_demotion_fast_admission_midpoint_reprieve_hybrid_cache
//!
//! Same one-`PaperCache<K, TieredBuffer>` architecture, fast-tier one-access
//! queue, one-access reprieve, demotion-time reprieve, mid-segment
//! checkpoint, and eviction-tail second chance as
//! `s3_fifo_ghost_lazy_demotion_fast_admission_midpoint_hybrid_cache` — see
//! that feature's integration test file for the shared coverage; this file
//! mirrors it end to end, minus the ghost-queue tests (there is no ghost
//! queue in this variant), plus tests specific to the reprieve behaviour:
//! a one-access-queue key that ages out lands in the main queue's slow tier
//! instead of being evicted, can still be promoted later by a real access,
//! and never causes a terminal eviction on its own.

#[cfg(feature = "s3_fifo_lazy_demotion_fast_admission_midpoint_reprieve_hybrid_cache")]
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

    /// Admission here is always Fast, so a warm-up key just sitting in the
    /// one-access queue never touches PMEM at all. Force a real demotion
    /// instead (a tiny effective main-fast budget makes a single promoted
    /// key self-demote immediately), which is what actually allocates
    /// through the UMF/TBB pool for the first time in this process.
    fn ensure_pmem_allocator_warm() {
        let cache = PaperCache::<u32, TieredBuffer>::new(1_000_000, CacheTierSize::Bytes(1), PaperPolicy::S3FifoLazyDemotionFastAdmissionMidpointReprieveHybrid(0.0))
            .expect("warm-up cache should construct");

        cache.set(0u32, b"warm", None).expect("warm-up set should succeed");
        cache.get(&0u32).expect("warm-up get should succeed");

        let demoted = wait_until(
            std::time::Duration::from_secs(90),
            || cache.tier_of(&0u32) == Some(Tier::Slow),
        );
        assert!(demoted, "warm-up key should have self-demoted to force a real PMEM allocation");
    }

    const MIGRATION_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(10);

    #[test]
    fn admission_always_lands_in_fast_tier() {
        ensure_pmem_allocator_warm();

        let cache = PaperCache::<u32, TieredBuffer>::new(
            1_000_000,
            CacheTierSize::Bytes(1_000_000), PaperPolicy::S3FifoLazyDemotionFastAdmissionMidpointReprieveHybrid(1.0)).expect("cache should construct");

        cache.set(1u32, b"hello world", None).expect("set should succeed");

        assert_eq!(cache.tier_of(&1u32), Some(Tier::Fast));
        assert_eq!(cache.get(&1u32).unwrap(), b"hello world");
    }

    // ── the signature new mechanic: reprieve instead of eviction ───────────

    #[test]
    fn a_key_that_ages_out_lands_directly_in_the_main_queues_slow_tier() {
        ensure_pmem_allocator_warm();

        let cache = PaperCache::<u32, TieredBuffer>::new(1_000_000, CacheTierSize::Bytes(1_000_000), PaperPolicy::S3FifoLazyDemotionFastAdmissionMidpointReprieveHybrid(0.00004)).expect("cache should construct");

        cache.set(1u32, b"first value 123", None).expect("set should succeed");
        assert_eq!(cache.tier_of(&1u32), Some(Tier::Fast));

        cache.set(2u32, b"second value 45", None).expect("set should succeed");

        // Unlike the ghost-queue predecessor (where key 1 would be evicted
        // here), key 1 must remain alive throughout, migrating straight to
        // Slow -- no eviction, no need to re-set it.
        let reprieved = wait_until(MIGRATION_TIMEOUT, || cache.tier_of(&1u32) == Some(Tier::Slow));
        assert!(reprieved, "key 1 should have been spliced into the slow tier, not evicted");
        assert!(cache.has(&1u32), "key 1 must still be a live entry");
        assert_eq!(cache.get(&1u32).unwrap(), b"first value 123");
    }

    #[test]
    fn a_reprieved_key_can_be_promoted_by_a_later_access() {
        ensure_pmem_allocator_warm();

        let cache = PaperCache::<u32, TieredBuffer>::new(
            1_000_000,
            CacheTierSize::Bytes(1_000_000), PaperPolicy::S3FifoLazyDemotionFastAdmissionMidpointReprieveHybrid(0.00004)).expect("cache should construct");

        cache.set(1u32, b"first value 123", None).expect("set should succeed");
        cache.set(2u32, b"second value 45", None).expect("set should succeed");
        assert!(wait_until(MIGRATION_TIMEOUT, || cache.tier_of(&1u32) == Some(Tier::Slow)));

        // Re-access the reprieved key. With one-access-queue pressure no
        // longer routed through eviction at all (see the module doc), the
        // ONLY thing that ever calls evict_one() -- and hence
        // check_slow_midpoint(), which is what actually checks this
        // reference bit -- is real max_size pressure. Force a real
        // eviction pass via a deterministic resize, not a filler set()
        // (same trigger the other main-queue-tail tests in this family
        // use), so the reference bit actually gets checked.
        cache.get(&1u32).expect("get should succeed");
        std::thread::sleep(std::time::Duration::from_millis(300));

        cache.resize(180).expect("resize should succeed");

        let promoted = wait_until(MIGRATION_TIMEOUT, || cache.tier_of(&1u32) == Some(Tier::Fast));
        assert!(promoted, "a reprieved key should still be promotable via the ordinary second chance");
        assert_eq!(cache.get(&1u32).unwrap(), b"first value 123");
    }

    #[test]
    fn one_access_pressure_alone_never_causes_a_terminal_eviction() {
        ensure_pmem_allocator_warm();

        // max_size comfortably larger than anything this test admits, so
        // the ONLY pressure driving anything is the tiny one-access
        // capacity -- if that pressure incorrectly triggered a terminal
        // eviction (the bug this variant's design doc explains was caught
        // and fixed), the evictions counter would move; it must not.
        let cache = PaperCache::<u32, TieredBuffer>::new(
            1_000_000,
            CacheTierSize::Bytes(1_000_000), PaperPolicy::S3FifoLazyDemotionFastAdmissionMidpointReprieveHybrid(0.00004)).expect("cache should construct");

        for key in 1u32..=20 {
            cache.set(key, b"payload bytes A", None).expect("set should succeed");
        }

        // Give every reprieve a chance to settle.
        std::thread::sleep(std::time::Duration::from_millis(500));

        let stats = cache.hybrid_stats();
        assert_eq!(stats.evictions, 0, "one-access capacity pressure must only ever reprieve, never terminally evict");

        for key in 1u32..=20 {
            assert!(cache.has(&key), "key {key} should still be alive, just possibly reprieved to slow");
        }
    }

    // ── main-queue behavior (unaffected by ghost removal / reprieve) ───────

    #[test]
    fn a_plain_access_on_a_fast_main_queue_key_does_not_migrate_or_reorder() {
        ensure_pmem_allocator_warm();

        // `one_access_ratio` must leave the MAIN queue real fast-tier room:
        // `effective_main_fast_capacity` is `fast_capacity - one_access_capacity`,
        // so a ratio of 1.0 (with fast_tier_size == max_size) zeroes it out and
        // `promote_from_one_access` demotes the key straight back to slow inside
        // the very same worker event. This test is about a key sitting *in* the
        // main queue's fast segment, so it needs a ratio that leaves headroom on
        // both sides: 0.5 gives the one-access queue 500_000 bytes (far more than
        // one payload, so set() never ages it out) and leaves the main queue the
        // other 500_000 (so the first get()'s promotion sticks).
        let cache = PaperCache::<u32, TieredBuffer>::new(
            1_000_000,
            CacheTierSize::Bytes(1_000_000), PaperPolicy::S3FifoLazyDemotionFastAdmissionMidpointReprieveHybrid(0.5)).expect("cache should construct");

        cache.set(1u32, b"first value 123", None).expect("set should succeed");
        cache.get(&1u32).expect("get should succeed");
        assert_eq!(cache.tier_of(&1u32), Some(Tier::Fast));

        let promotions_before = cache.hybrid_stats().promotions;

        cache.get(&1u32).expect("get should succeed");
        std::thread::sleep(std::time::Duration::from_millis(300));

        assert_eq!(cache.tier_of(&1u32), Some(Tier::Fast));
        assert_eq!(
            cache.hybrid_stats().promotions,
            promotions_before,
        );
    }

    #[test]
    fn an_accessed_key_at_the_main_queue_tail_gets_a_second_chance_instead_of_eviction() {
        ensure_pmem_allocator_warm();

        // one_access_capacity = 0.00004 * 1_000_000 = 40, comfortably above
        // one payload's stack-level size, so a set()+get() in immediate
        // succession promotes normally via touch() instead of racing
        // settle_one_access's synchronous reprieve (which would otherwise
        // fire from within the set() event itself, before the get() event
        // is even processed -- see the module doc). fast_capacity is
        // bumped by that same 40 so the MAIN queue's effective budget
        // (fast_capacity - one_access_capacity) stays 40, matching the
        // dynamics this test was originally built around.
        let cache = PaperCache::<u32, TieredBuffer>::new(
            1_000_000,
            CacheTierSize::Bytes(80), PaperPolicy::S3FifoLazyDemotionFastAdmissionMidpointReprieveHybrid(0.00004)).expect("cache should construct");

        cache.set(1u32, b"payload bytes A", None).expect("set should succeed");
        cache.get(&1u32).expect("get should succeed");
        assert_eq!(cache.tier_of(&1u32), Some(Tier::Fast));

        cache.set(2u32, b"payload bytes B", None).expect("set should succeed");
        cache.get(&2u32).expect("get should succeed");
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

    // ── the inherited signature mechanic: reprieve at DEMOTION time ────────

    #[test]
    fn an_accessed_fast_boundary_key_is_reprieved_at_demotion_time_instead_of_the_newcomer() {
        ensure_pmem_allocator_warm();

        // one_access_capacity = 0.00004 * 1_000_000 = 40, comfortably above
        // one payload's stack-level size, so a set()+get() in immediate
        // succession promotes normally via touch() instead of racing
        // settle_one_access's synchronous reprieve (which would otherwise
        // fire from within the set() event itself, before the get() event
        // is even processed -- see the module doc). fast_capacity is
        // bumped by that same 40 so the MAIN queue's effective budget
        // (fast_capacity - one_access_capacity) stays 40, matching the
        // dynamics this test was originally built around.
        let cache = PaperCache::<u32, TieredBuffer>::new(
            1_000_000,
            CacheTierSize::Bytes(80), PaperPolicy::S3FifoLazyDemotionFastAdmissionMidpointReprieveHybrid(0.00004)).expect("cache should construct");

        cache.set(1u32, b"payload bytes A", None).expect("set should succeed");
        cache.get(&1u32).expect("get should succeed");
        assert_eq!(cache.tier_of(&1u32), Some(Tier::Fast));

        let promotions_before = cache.hybrid_stats().promotions;
        let demotions_before = cache.hybrid_stats().demotions;

        cache.get(&1u32).expect("get should succeed");
        std::thread::sleep(std::time::Duration::from_millis(300));
        assert_eq!(cache.tier_of(&1u32), Some(Tier::Fast));

        cache.set(2u32, b"payload bytes B", None).expect("set should succeed");
        cache.get(&2u32).expect("get should succeed");

        let demoted = wait_until(MIGRATION_TIMEOUT, || cache.tier_of(&2u32) == Some(Tier::Slow));
        assert!(demoted, "key 2 should have been demoted in key 1's place, got tier {:?}", cache.tier_of(&2u32));

        std::thread::sleep(std::time::Duration::from_millis(300));
        assert_eq!(
            cache.tier_of(&1u32), Some(Tier::Fast),
            "key 1 should have been reprieved at the demotion boundary, not demoted",
        );

        let stats = cache.hybrid_stats();
        assert_eq!(stats.promotions, promotions_before, "reprieve must not count as a promotion");
        assert_eq!(stats.demotions, demotions_before + 1, "exactly key 2 should have been demoted");

        assert_eq!(cache.get(&1u32).unwrap(), b"payload bytes A");
        assert_eq!(cache.get(&2u32).unwrap(), b"payload bytes B");
    }

    // ── the mid-segment checkpoint ─────────────────────────────────────────

    #[test]
    fn a_reaccessed_key_deep_in_the_slow_segment_survives_via_the_midpoint_checkpoint() {
        ensure_pmem_allocator_warm();

        // A small enough max_size that a run of ~30 tiny objects (measured
        // real base_size ~35 bytes each for this payload) forces real
        // terminal evictions (not just fast/slow demotions, which are
        // governed by fast_tier_size alone and would happen regardless of
        // max_size) -- terminal eviction is what drives evict_one(), and
        // hence the mid-segment checkpoint this test is exercising.
        let cache = PaperCache::<u32, TieredBuffer>::new(
            2_000,
            CacheTierSize::Bytes(470), PaperPolicy::S3FifoLazyDemotionFastAdmissionMidpointReprieveHybrid(0.2)).expect("cache should construct");

        // Build a real slow segment: each key is admitted (Fast), then
        // immediately re-accessed to promote it into the main queue,
        // which demotes whatever's currently at the fast/slow boundary.
        for key in 1u32..=6 {
            cache.set(key, b"payload bytes A", None).expect("set should succeed");
            cache.get(&key).expect("get should succeed");
        }

        // wait_until, not a bare sleep -- the worker thread's very first
        // batch of events can lag noticeably behind a freshly constructed
        // cache under load, so a fixed sleep here would be a source of
        // flakiness.
        assert!(
            wait_until(MIGRATION_TIMEOUT, || cache.tier_of(&1u32) == Some(Tier::Slow)),
            "key 1 (the oldest) should have been demoted by now",
        );

        // Reaccess key 1 once, without otherwise touching it -- just sets
        // its reference bit.
        cache.get(&1u32).expect("get should succeed");
        std::thread::sleep(std::time::Duration::from_millis(300));

        // Keep growing the slow segment well past key 1's position and
        // past max_size, forcing a real run of terminal evictions -- each
        // one runs check_slow_midpoint() before evaluating the tail.
        for key in 7u32..=30 {
            cache.set(key, b"payload bytes A", None).expect("set should succeed");
            cache.get(&key).expect("get should succeed");
        }

        let promoted = wait_until(MIGRATION_TIMEOUT, || cache.tier_of(&1u32) == Some(Tier::Fast));
        assert!(
            promoted,
            "a reaccessed key sitting well inside the slow segment should have been promoted \
             early via the mid-segment checkpoint, got tier {:?} (has: {})",
            cache.tier_of(&1u32), cache.has(&1u32),
        );
        assert_eq!(cache.get(&1u32).unwrap(), b"payload bytes A");
    }

    #[test]
    fn an_unaccessed_key_deep_in_the_slow_segment_is_eventually_evicted() {
        ensure_pmem_allocator_warm();

        let cache = PaperCache::<u32, TieredBuffer>::new(
            2_000,
            CacheTierSize::Bytes(470), PaperPolicy::S3FifoLazyDemotionFastAdmissionMidpointReprieveHybrid(0.2)).expect("cache should construct");

        for key in 1u32..=6 {
            cache.set(key, b"payload bytes A", None).expect("set should succeed");
            cache.get(&key).expect("get should succeed");
        }
        assert!(wait_until(MIGRATION_TIMEOUT, || cache.tier_of(&1u32) == Some(Tier::Slow)));

        // Key 1 is never reaccessed this time -- the mid-segment
        // checkpoint must leave it alone (bit clear), and it should
        // eventually be evicted for real once enough pressure builds.
        for key in 7u32..=30 {
            cache.set(key, b"payload bytes A", None).expect("set should succeed");
            cache.get(&key).expect("get should succeed");
        }

        let evicted = wait_until(MIGRATION_TIMEOUT, || !cache.has(&1u32));
        assert!(evicted, "an unaccessed key should eventually be evicted for real, not kept alive forever");
    }

    // ── TTL ───────────────────────────────────────────────────────────────

    const TTL_FAST_TIER: u64 = 200;

    #[test]
    fn ttl_survives_a_demotion() {
        ensure_pmem_allocator_warm();

        let cache = PaperCache::<u32, TieredBuffer>::new(
            1_000_000,
            CacheTierSize::Bytes(TTL_FAST_TIER), PaperPolicy::S3FifoLazyDemotionFastAdmissionMidpointReprieveHybrid(0.0)).expect("cache should construct");

        let ttl_secs = 5u32;
        let set_at = std::time::Instant::now();
        cache.set(1u32, b"first value 123", Some(ttl_secs)).expect("set should succeed");

        for key in 2u32..=6 {
            cache.set(key, b"filler bytes", None).expect("set should succeed");
        }

        cache.get(&1u32).expect("get should succeed");
        assert_eq!(cache.tier_of(&1u32), Some(Tier::Fast));

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
    fn ttl_survives_a_reprieve() {
        ensure_pmem_allocator_warm();

        let cache = PaperCache::<u32, TieredBuffer>::new(1_000_000, CacheTierSize::Bytes(1_000_000), PaperPolicy::S3FifoLazyDemotionFastAdmissionMidpointReprieveHybrid(0.00004)).expect("cache should construct");

        let ttl_secs = 5u32;
        let set_at = std::time::Instant::now();
        cache.set(1u32, b"first value 123", Some(ttl_secs)).expect("set should succeed");
        assert_eq!(cache.tier_of(&1u32), Some(Tier::Fast));

        // Pushes key 1 past the one-access budget, reprieving it into the
        // main queue's slow tier -- must not disturb its TTL.
        cache.set(2u32, b"second value 45", None).expect("set should succeed");
        assert!(wait_until(MIGRATION_TIMEOUT, || cache.tier_of(&1u32) == Some(Tier::Slow)));
        assert!(cache.has(&1u32), "key should still be alive right after the reprieve");

        let remaining = std::time::Duration::from_millis(ttl_secs as u64 * 1000 + 500)
            .saturating_sub(set_at.elapsed());
        std::thread::sleep(remaining);

        assert!(matches!(cache.get(&1u32), Err(CacheError::KeyNotFound)));
        assert!(!cache.has(&1u32));
    }

    // ── runtime resize ────────────────────────────────────────────────────

    #[test]
    fn set_fast_tier_size_takes_effect_at_runtime() {
        ensure_pmem_allocator_warm();

        let cache = PaperCache::<u32, TieredBuffer>::new(
            1_000_000,
            CacheTierSize::Bytes(1_000_000), PaperPolicy::S3FifoLazyDemotionFastAdmissionMidpointReprieveHybrid(0.0)).expect("cache should construct");

        cache.set(1u32, b"first value 123", None).expect("set should succeed");
        cache.get(&1u32).expect("get should succeed");
        assert_eq!(cache.tier_of(&1u32), Some(Tier::Fast));

        cache.set_fast_tier_size(CacheTierSize::Bytes(1)).expect("resize should succeed");

        let demoted = wait_until(MIGRATION_TIMEOUT, || cache.tier_of(&1u32) == Some(Tier::Slow));
        assert!(demoted, "shrinking the fast tier should demote the promoted key");
    }

    // ── edge cases ────────────────────────────────────────────────────────

    #[test]
    fn zero_fast_tier_size_is_rejected() {
        let result = PaperCache::<u32, TieredBuffer>::new(1_000, CacheTierSize::Bytes(0), PaperPolicy::S3FifoLazyDemotionFastAdmissionMidpointReprieveHybrid(0.5));
        assert!(matches!(result, Err(CacheError::InvalidFastTierSize)));
    }

    #[test]
    fn invalid_one_access_ratio_is_rejected() {
        assert!(matches!(
            PaperCache::<u32, TieredBuffer>::new(1_000, CacheTierSize::Bytes(500), PaperPolicy::S3FifoLazyDemotionFastAdmissionMidpointReprieveHybrid(1.5)),
            Err(CacheError::InvalidPolicy),
        ));
    }

    #[test]
    fn del_removes_key_from_whichever_tier_it_is_in() {
        ensure_pmem_allocator_warm();

        let cache = PaperCache::<u32, TieredBuffer>::new(
            1_000_000,
            CacheTierSize::Bytes(1_000_000), PaperPolicy::S3FifoLazyDemotionFastAdmissionMidpointReprieveHybrid(0.0)).expect("cache should construct");

        cache.set(1u32, b"first value 123", None).expect("set should succeed");
        cache.get(&1u32).expect("get should succeed");
        assert_eq!(cache.tier_of(&1u32), Some(Tier::Fast));

        cache.del(&1u32).expect("del should succeed");
        assert!(!cache.has(&1u32));
        assert_eq!(cache.tier_of(&1u32), None);
    }

    #[test]
    fn wipe_clears_both_tiers() {
        ensure_pmem_allocator_warm();

        let cache = PaperCache::<u32, TieredBuffer>::new(
            1_000_000,
            CacheTierSize::Bytes(1_000_000), PaperPolicy::S3FifoLazyDemotionFastAdmissionMidpointReprieveHybrid(0.0)).expect("cache should construct");

        cache.set(1u32, b"first value 123", None).expect("set should succeed");
        cache.set(2u32, b"second value 45", None).expect("set should succeed");
        cache.get(&1u32).expect("get should succeed");
        assert_eq!(cache.tier_of(&1u32), Some(Tier::Fast));

        cache.wipe().expect("wipe should succeed");

        assert!(!cache.has(&1u32));
        assert!(!cache.has(&2u32));
        assert_eq!(cache.tier_of(&1u32), None);
        assert_eq!(cache.tier_of(&2u32), None);
    }

    // ── concurrency: the migration window's stale-write guard ─────────────

    /// Proves the `Arc::ptr_eq` guard in `PolicyWorker::apply_tier_migrations`
    /// is load-bearing.
    ///
    /// That method snapshots an object's `Arc<TieredBuffer>`, releases the
    /// object-map guard, builds the destination-tier buffer *unlocked*, then
    /// re-acquires and swaps. Taking the copy off the lock is what removed it
    /// from the GET critical path (see `PaperCache::get`'s doc comment), but
    /// it opens a window in which a concurrent `set()` replaces the whole
    /// `Object` -- and writing the migrated bytes of the *previous* value over
    /// that replacement would resurrect stale data. The guard re-checks
    /// `Arc::ptr_eq` before swapping and drops the migration if it moved.
    ///
    /// Two things make this detect the bug rather than merely hope to:
    ///
    /// 1. **A wide window.** `migrate()`'s cost scales with value size, so
    ///    multi-megabyte values stretch it from microseconds to milliseconds,
    ///    which a writer can reliably land inside. Earlier drafts using
    ///    kilobyte values could not hit it at any timing.
    /// 2. **A continuous reader.** A stale write can land at *any* point after
    ///    the `set()` that raced it, and it persists until the next write to
    ///    that key. Checking once immediately after writing (as earlier drafts
    ///    did) samples a nanosecond-wide instant and almost always misses. A
    ///    dedicated reader thread polls its key in a tight loop instead, so it
    ///    observes the resurrected value during the whole interval it is live.
    ///
    /// The invariant is per-key monotonicity: exactly one writer touches a
    /// given key and its generations strictly increase (`del` then two `set`s
    /// per round, generations `2r` and `2r+1`), so observing any generation
    /// below one already seen means a migration wrote back a value that had
    /// since been replaced. Verified to fail when the `Arc::ptr_eq` check is
    /// removed from `apply_tier_migrations`.
    #[test]
    fn concurrent_sets_are_never_clobbered_by_an_in_flight_migration() {
        use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

        ensure_pmem_allocator_warm();

        const KEYS: u32 = 2;
        const VALUE_LEN: usize = 4 * 1024 * 1024;
        const ROUNDS: u64 = 600;

        // The value size is the load-bearing part: it sets how long the
        // worker spends inside `migrate()`, which is the window under test.
        // `one_access_ratio` is small enough that a single value overflows the
        // one-access queue, so every admission is immediately reprieved into
        // the slow tier -- one real migration per `set()` of a new key.
        let cache = std::sync::Arc::new(
            PaperCache::<u32, TieredBuffer>::new(
                256 * 1024 * 1024,
                CacheTierSize::Bytes(16 * 1024 * 1024), PaperPolicy::S3FifoLazyDemotionFastAdmissionMidpointReprieveHybrid(0.004)).expect("cache should construct"),
        );

        let done = std::sync::Arc::new(AtomicBool::new(false));
        let stale = std::sync::Arc::new(AtomicU64::new(0));
        let migrations_seen = std::sync::Arc::new(AtomicU64::new(0));

        // One reader per key, polling continuously so a resurrected value is
        // caught for as long as it is live rather than at a single instant.
        let readers: Vec<_> = (0..KEYS)
            .map(|key| {
                let cache = std::sync::Arc::clone(&cache);
                let done = std::sync::Arc::clone(&done);
                let stale = std::sync::Arc::clone(&stale);

                std::thread::spawn(move || {
                    let mut max_seen = 0u64;

                    while !done.load(Ordering::Relaxed) {
                        if let Ok(bytes) = cache.get(&key) {
                            let mut buf = [0u8; 8];
                            buf.copy_from_slice(&bytes[..8]);
                            let observed = u64::from_le_bytes(buf);

                            if observed < max_seen {
                                stale.fetch_max(max_seen - observed, Ordering::Relaxed);
                                panic!(
                                    "key {key}: observed generation {observed} after having \
                                     already seen {max_seen} -- an in-flight migration \
                                     resurrected a stale value",
                                );
                            }

                            max_seen = observed;
                        }
                    }
                })
            })
            .collect();

        let writers: Vec<_> = (0..KEYS)
            .map(|key| {
                let cache = std::sync::Arc::clone(&cache);
                let migrations_seen = std::sync::Arc::clone(&migrations_seen);

                std::thread::spawn(move || {
                    let mut value = vec![0u8; VALUE_LEN];

                    for round in 1..=ROUNDS {
                        // Removing the key makes the next `set()` a *fresh
                        // admission*, which is the only thing that queues a
                        // migration for an already-live working set: a `set()`
                        // on a tracked key records no tier transition.
                        let _ = cache.del(&key);

                        value[..8].copy_from_slice(&(round * 2).to_le_bytes());
                        cache.set(key, &value, None).expect("set should succeed");

                        // Land the replacement inside the worker's copy of the
                        // admission above.
                        std::thread::sleep(std::time::Duration::from_micros(300));

                        value[..8].copy_from_slice(&(round * 2 + 1).to_le_bytes());
                        cache.set(key, &value, None).expect("set should succeed");

                        migrations_seen.store(
                            cache
                                .hybrid_stats()
                                .demotions,
                            Ordering::Relaxed,
                        );
                    }
                })
            })
            .collect();

        for w in writers {
            w.join().expect("writer thread panicked");
        }

        done.store(true, Ordering::Relaxed);

        for r in readers {
            r.join().expect("reader thread observed a stale value");
        }

        // Non-vacuity: the worker must actually have been migrating.
        assert!(
            migrations_seen.load(Ordering::Relaxed) > 0,
            "no migrations occurred -- the window this test targets was never exercised",
        );
    }
}
