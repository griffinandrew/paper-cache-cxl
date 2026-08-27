/*
 * Copyright (c) Kia Shakiba
 *
 * This source code is licensed under the GNU AGPLv3 license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! Integration tests for the
//! `s3_fifo_ghost_lazy_demotion_fast_admission_midpoint_hybrid_cache`
//! feature.
//!
//! Run with nightly (required for `allocator_api` via `key_value_pmem`):
//!   cargo +nightly test --test s3_fifo_ghost_lazy_demotion_fast_admission_midpoint_hybrid_cache_integration --features s3_fifo_ghost_lazy_demotion_fast_admission_midpoint_hybrid_cache
//!
//! Same one-`PaperCache<K, TieredBuffer>` architecture, fast-tier one-access
//! queue, ghost-queue lifecycle, demotion-time reprieve, and eviction-time
//! second-chance mechanic as
//! `s3_fifo_ghost_lazy_demotion_fast_admission_hybrid_cache` — see that
//! feature's integration test file for the shared coverage; this file
//! mirrors it end to end and adds one test specific to the new behavior:
//! a reaccessed key sitting well inside the slow segment (not near the
//! tail) getting promoted early via the mid-segment checkpoint
//! (`a_reaccessed_key_deep_in_the_slow_segment_survives_via_the_midpoint_checkpoint`).
//! The exact positional mechanics of the checkpoint are already covered
//! precisely, with hand-traced cursor positions, by this feature's stack
//! unit tests -- this file's job is to confirm the mechanism works
//! end-to-end through the real API and worker pipeline (real object
//! overhead, real async event processing, real PMEM migrations), not to
//! re-derive the exact cursor arithmetic.

#[cfg(feature = "s3_fifo_ghost_lazy_demotion_fast_admission_midpoint_hybrid_cache")]
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
    ///
    /// The one-access queue must be given enough capacity to actually HOLD the
    /// warm-up key (`one_access_ratio` 0.001 * 1_048_576 = 1_048 bytes, far more
    /// than one payload). A ratio of 0.0 would leave it zero-capacity, making
    /// the key eligible for `needs_capacity_eviction` the instant it is
    /// admitted -- and in *this* variant an aged-out one-access key is
    /// `evict_one_access_tail`'d (removed outright, remembered only as a bare
    /// ghost key), never demoted. Whether the eviction pass or the `get()`
    /// below reached the key first was then a genuine coin flip: if eviction
    /// won, `tier_of` returned `None` forever and this helper burned its full
    /// 90-second budget before failing. The reprieve variants splice an
    /// aged-out key into the main queue's slow tier instead of evicting it, so
    /// they are not exposed to this and keep their own simpler warm-up.
    fn ensure_pmem_allocator_warm() {
        // Mechanics tests at toy scales: metadata reservation off (see
        // `get_hybrid_dram_shared_overhead`).
        unsafe { std::env::set_var("PAPER_DISABLE_SHARED_OVERHEAD", "1") };
        // fast_tier_size == one_access_capacity (1000 == 0.001 * 1_048_576)
        // leaves `effective_main_fast_capacity` at exactly 0, so the promotion
        // triggered by the get() below self-demotes deterministically.
        let cache = PaperCache::<u32, TieredBuffer>::new(1_048_576, CacheTierSize::Bytes(1_024), PaperPolicy::S3FifoGhostLazyDemotionFastAdmissionMidpointHybrid(0.001))
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

    /// ~1 KB values for the tier-pressure tests below. Each of those needs a
    /// main-queue fast segment that holds ONE object and not two, and only a
    /// value's MIGRATING bytes are charged to that segment -- `base_size`
    /// minus the key and the expiry, which live in the object map and so stay
    /// in DRAM whichever tier the object is in. A 15-byte value migrates just
    /// ~16 bytes (jemalloc's 16-byte size class), so the window between "one
    /// fits" and "two fit" was sixteen bytes wide, and the 40-byte budgets
    /// those tests were written against sat above BOTH objects: no demotion
    /// pass ever triggered and every such `wait_until` burned its full
    /// timeout. At `VALUE_LEN` the same window is ~1 KB wide. Same idiom as
    /// the `s3_fifo_`, `lru_` and `lfu_` suites.
    ///
    /// Deliberately NOT applied file-wide.
    /// `terminal_eviction_prefers_one_access_queue_over_main_queue` runs a
    /// 512-byte TOTAL cache sized in 122-byte objects, and
    /// `a_key_that_ages_out_and_is_readmitted_lands_directly_in_main_queue`
    /// closes by asserting that ten fillers do NOT displace a main-queue key.
    /// Both pass precisely because their values are small.
    const VALUE_LEN: usize = 1_024;

    fn value(seed: u8) -> Vec<u8> {
        vec![seed; VALUE_LEN]
    }

    /// A `one_access_ratio` of 0.0 is unusable in this variant, and was the
    /// cause of a whole class of flaky failures across this file.
    ///
    /// Two of this variant's properties combine badly at ratio 0.0. Admission
    /// is Fast, so a brand-new key lands in the DRAM one-access queue; and an
    /// aged-out one-access key is `evict_one_access_tail`'d -- removed from the
    /// cache outright, remembered only as a bare ghost key -- never demoted.
    /// With `one_access_capacity` at 0, `needs_capacity_eviction()` is
    /// therefore true the instant *any* key is admitted, so every `set()` races
    /// the worker's eviction pass against the test's own next observation.
    /// Whichever won decided whether `tier_of` returned `Some(Tier::Fast)` or
    /// `None`, which is exactly the coin-flip these tests were showing.
    ///
    /// `ONE_ACCESS_RATIO * max_size` gives the one-access queue 8_192 bytes
    /// (eight of the ~1 KB payloads below, or ~500 of the 15-byte ones),
    /// enough that admission never self-evicts. Because
    /// `effective_main_fast_capacity` is `fast_capacity - one_access_capacity`,
    /// each test below adds `ONE_ACCESS_RESERVE` to its fast-tier size so the
    /// main queue's own budget stays exactly the value that test intends.
    ///
    /// Sized for the ~1 KB payloads the tier-pressure tests now use (see
    /// `VALUE_LEN`), not for the 15-byte ones they were written against. The
    /// old 1_024 bytes is EXACTLY one such payload's migrating bytes, i.e.
    /// sitting precisely on `needs_capacity_eviction`'s `one_access_used >
    /// capacity` boundary with no room for a second object to be in flight --
    /// and `ttl_survives_a_demotion` parks all six of its objects in the
    /// one-access queue before promoting any of them, so it needs six
    /// payloads' worth at once. Overshooting here is not a demotion but a
    /// deletion (an aged-out one-access key is `evict_one_access_tail`'d, see
    /// `ensure_pmem_allocator_warm`), so the reserve is given real headroom
    /// rather than an exact fit.
    ///
    /// Raising it costs the tests that only want the one-access queue out of
    /// the way nothing: a larger budget can only make an admission LESS
    /// likely to self-evict, and the one budget it is subtracted from,
    /// `effective_main_fast_capacity`, is `fast_capacity - 8_192` rather than
    /// `fast_capacity - 1_024` for them -- still four orders of magnitude
    /// above what any of them stores. The one test in this file that does
    /// assert a one-access eviction --
    /// `a_key_that_ages_out_and_is_readmitted_lands_directly_in_main_queue`
    /// -- states its own tight ratio inline and never reads this one.
    ///
    /// 1/128 exactly, so `ratio * max_size` is 2^20 / 2^7 = 8_192 with no
    /// truncation -- the reserve below is the exact product, not a rounded
    /// stand-in for it.
    const ONE_ACCESS_RATIO: f64 = 0.007_812_5;
    const ONE_ACCESS_RESERVE: u64 = 8_192;

    #[test]
    fn admission_always_lands_in_fast_tier() {
        ensure_pmem_allocator_warm();

        // ONE_ACCESS_RATIO, not 1.0: ratio 1.0 is now rejected outright, since
        // `main_capacity` is `(1 - ratio) * max_size` and would be zero. The
        // fixture only needs the one-access queue to hold the single admitted
        // key: 0.001 * 1_048_576 = 1_048 bytes (~8 objects at this policy's
        // 122-byte accounted size), leaving the main queue the other 999_000 --
        // both budgets dwarf this one-key workload, so the main-queue gate is
        // transparent and the test keeps its original meaning.
        let cache = PaperCache::<u32, TieredBuffer>::new(
            1_048_576,
            CacheTierSize::Bytes(1_048_576), PaperPolicy::S3FifoGhostLazyDemotionFastAdmissionMidpointHybrid(ONE_ACCESS_RATIO)).expect("cache should construct");

        cache.set(1u32, b"hello world", None).expect("set should succeed");

        assert_eq!(cache.tier_of(&1u32), Some(Tier::Fast));
        assert_eq!(cache.get(&1u32).unwrap(), b"hello world");
    }

    // ── ghost queue: still governs Main-vs-one-access placement ────────────

    #[test]
    fn a_key_that_ages_out_and_is_readmitted_lands_directly_in_main_queue() {
        ensure_pmem_allocator_warm();

        // 0.00002 * 1_048_576 = 20 bytes of one-access budget, so ONE 15-byte
        // object fits (~16 migrating bytes -- `base_size` minus the key and
        // expiry, which never leave DRAM) and the second overflows it, which
        // is the ageing-out this test is built on. It was 0.00004 -> 41 bytes,
        // which holds BOTH: `needs_capacity_eviction` never fired, key 1 never
        // left the one-access queue, and the wait below timed out.
        //
        // The BUDGET is what was wrong here, not the values: small values are
        // deliberate and must stay, because the closing assertion is that ten
        // fillers do NOT displace key 1 once the ghost hit has put it in the
        // main queue.
        let cache = PaperCache::<u32, TieredBuffer>::new(1_048_576, CacheTierSize::Bytes(1_048_576), PaperPolicy::S3FifoGhostLazyDemotionFastAdmissionMidpointHybrid(0.00002)).expect("cache should construct");

        cache.set(1u32, b"first value 123", None).expect("set should succeed");
        assert_eq!(cache.tier_of(&1u32), Some(Tier::Fast));

        cache.set(2u32, b"second value 45", None).expect("set should succeed");

        let evicted = wait_until(MIGRATION_TIMEOUT, || !cache.has(&1u32));
        assert!(evicted, "key 1 should have aged out of the one-access queue");

        cache.set(1u32, b"first value 123", None).expect("re-set should succeed");
        assert_eq!(cache.tier_of(&1u32), Some(Tier::Fast));

        for extra_key in 100u32..110u32 {
            cache.set(extra_key, b"filler filler!!", None).expect("set should succeed");
        }
        std::thread::sleep(std::time::Duration::from_millis(300));

        assert!(
            cache.has(&1u32),
            "a ghost-queue hit should land directly in the main queue, protected from one-access aging",
        );
    }

    // ── main-queue behavior (unaffected by the mid-segment checkpoint) ─────

    #[test]
    fn a_plain_access_on_a_fast_main_queue_key_does_not_migrate_or_reorder() {
        ensure_pmem_allocator_warm();

        // `one_access_ratio` must leave the MAIN queue real fast-tier room:
        // `effective_main_fast_capacity` is `fast_capacity - one_access_capacity`,
        // so a ratio of 1.0 (with fast_tier_size == max_size) zeroes it out and
        // `promote_from_one_access` demotes the key straight back to slow inside
        // the very same worker event. This test is about a key sitting *in* the
        // main queue's fast segment, so it needs a ratio that leaves headroom on
        // both sides: 0.5 gives the one-access queue 524_288 bytes (far more than
        // one payload, so set() never ages it out) and leaves the main queue the
        // other 524_288 (so the first get()'s promotion sticks).
        let cache = PaperCache::<u32, TieredBuffer>::new(
            1_048_576,
            CacheTierSize::Bytes(1_048_576), PaperPolicy::S3FifoGhostLazyDemotionFastAdmissionMidpointHybrid(0.5)).expect("cache should construct");

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

        // Leaves 1_600 bytes of effective room for the main queue (this test
        // is specifically about main-queue eviction priority, not the
        // one-access budget) -- see ONE_ACCESS_RATIO for why that reservation
        // is added on top rather than the ratio simply being 0.0.
        // `effective_main_fast_capacity` is `fast_capacity -
        // one_access_capacity`, so the reserve has to be handed back on top or
        // that subtraction eats the budget this test intends.
        //
        // 1_600 holds one ~1 KB value and not two: 1_024 migrating bytes is
        // under the 0.98 high watermark's 1_568, and 2_048 is over it. It was
        // 40 bytes against 15-byte values, whose two objects migrate ~32 bytes
        // between them -- comfortably under the 39-byte trigger -- so
        // promoting key 2 demoted nothing and the wait below timed out.
        //
        // Back on the file-wide max_size/ratio pair, too: the local 131_072
        // pair existed only because `resize(180)` truncated ONE_ACCESS_RATIO's
        // product to zero. The resize target scales with the payloads (see
        // below) and 0.007_812_5 * 1_400 is 10 bytes, so the local pair has
        // nothing left to buy. 1_048_576 is ~460x what this test admits, so
        // the global `used_size() > max_size` trigger stays quiet until the
        // resize fires it, exactly as before.
        let cache = PaperCache::<u32, TieredBuffer>::new(
            1_048_576,
            CacheTierSize::Bytes(1_600 + ONE_ACCESS_RESERVE), PaperPolicy::S3FifoGhostLazyDemotionFastAdmissionMidpointHybrid(ONE_ACCESS_RATIO)).expect("cache should construct");

        cache.set(1u32, &value(0xD4), None).expect("set should succeed");
        cache.get(&1u32).expect("get should succeed");
        assert_eq!(cache.tier_of(&1u32), Some(Tier::Fast));

        cache.set(2u32, &value(0xE5), None).expect("set should succeed");
        cache.get(&2u32).expect("get should succeed");
        assert!(wait_until(MIGRATION_TIMEOUT, || cache.tier_of(&1u32) == Some(Tier::Slow)));

        cache.get(&1u32).expect("get should succeed");
        std::thread::sleep(std::time::Duration::from_millis(300));
        assert_eq!(cache.tier_of(&1u32), Some(Tier::Slow));

        // Deterministic trigger, not a filler set() -- see
        // hybrid_cache_integration.rs's equivalent test for why.
        //
        // Scaled WITH the payloads, not independently of them: one ~1 KB
        // object costs roughly 1.1 KB of accounted size (base bytes plus this
        // policy's 87-byte per-object overhead), so 1_400 leaves room for one
        // of the two and forces exactly the single eviction this test is
        // about. It was 180, sized for 15-byte values; against ~1 KB values a
        // 180-byte cache evicts everything, key 1 included, and the second
        // chance could not be observed at all.
        cache.resize(1_400).expect("resize should succeed");

        let survived_and_promoted = wait_until(MIGRATION_TIMEOUT, || {
            cache.has(&1u32) && cache.tier_of(&1u32) == Some(Tier::Fast)
        });
        assert!(
            survived_and_promoted,
            "key 1 should have been given a second chance and promoted back to fast",
        );
        assert_eq!(cache.get(&1u32).unwrap(), value(0xD4));
    }

    // ── the inherited signature mechanic: reprieve at DEMOTION time ────────

    #[test]
    fn an_accessed_fast_boundary_key_is_reprieved_at_demotion_time_instead_of_the_newcomer() {
        ensure_pmem_allocator_warm();

        // Same 1_600 bytes of effective main-queue room, and the same
        // one-access reservation on top, as the second-chance test above: one
        // ~1 KB value migrates 1_024 bytes, under the 0.98 high watermark's
        // 1_568, and two migrate 2_048, over it. It was 40 bytes against
        // 15-byte values, where BOTH objects' ~32 migrating bytes fit, so no
        // demotion pass ever ran -- which left no demotion boundary for key
        // 1's reference bit to be consulted at, and key 2 simply stayed Fast.
        let cache = PaperCache::<u32, TieredBuffer>::new(
            1_048_576,
            CacheTierSize::Bytes(1_600 + ONE_ACCESS_RESERVE), PaperPolicy::S3FifoGhostLazyDemotionFastAdmissionMidpointHybrid(ONE_ACCESS_RATIO)).expect("cache should construct");

        cache.set(1u32, &value(0xA1), None).expect("set should succeed");
        cache.get(&1u32).expect("get should succeed");
        assert_eq!(cache.tier_of(&1u32), Some(Tier::Fast));

        let promotions_before = cache.hybrid_stats().promotions;
        let demotions_before = cache.hybrid_stats().demotions;

        cache.get(&1u32).expect("get should succeed");
        std::thread::sleep(std::time::Duration::from_millis(300));
        assert_eq!(cache.tier_of(&1u32), Some(Tier::Fast));

        cache.set(2u32, &value(0xB2), None).expect("set should succeed");
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

        assert_eq!(cache.get(&1u32).unwrap(), value(0xA1));
        assert_eq!(cache.get(&2u32).unwrap(), value(0xB2));
    }

    // ── the signature new mechanic: a checkpoint mid-slow-segment ──────────

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
            2_048,
            CacheTierSize::Bytes(470), PaperPolicy::S3FifoGhostLazyDemotionFastAdmissionMidpointHybrid(0.2)).expect("cache should construct");

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
        // one runs check_slow_midpoint() before evaluating the tail, and
        // with this many events the cursor's sweep is virtually certain to
        // pass through key 1's position before key 1 would naturally reach
        // the tail itself.
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
            2_048,
            CacheTierSize::Bytes(470), PaperPolicy::S3FifoGhostLazyDemotionFastAdmissionMidpointHybrid(0.2)).expect("cache should construct");

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

    /// Holds one ~1 KB value but not two: 1_024 migrating bytes is under the
    /// 0.98 high watermark's 1_568, and 2_048 is over it. It was 200, sized
    /// for the 15-byte values this test used to store -- twelve of those fit,
    /// so promoting key 1 and then all five fillers on top of it (six objects,
    /// ~96 migrating bytes in total) never came near the demotion trigger and
    /// key 1 stayed Fast forever.
    const TTL_FAST_TIER: u64 = 1_600;

    #[test]
    fn ttl_survives_a_demotion() {
        ensure_pmem_allocator_warm();

        // Unlike the two tests above, this one `set()`s all six objects
        // before it `get()`s any of them, so all six sit in the one-access
        // queue at once -- 6 * 1_024 = 6_144 migrating bytes. That is what
        // ONE_ACCESS_RESERVE's headroom is for: at the old 1_024 the queue
        // would be over budget from the second `set()` on, and an aged-out
        // one-access key in this variant is removed outright rather than
        // demoted, so key 1 would be deleted before it could ever be promoted.
        let cache = PaperCache::<u32, TieredBuffer>::new(
            1_048_576,
            CacheTierSize::Bytes(TTL_FAST_TIER + ONE_ACCESS_RESERVE), PaperPolicy::S3FifoGhostLazyDemotionFastAdmissionMidpointHybrid(ONE_ACCESS_RATIO)).expect("cache should construct");

        let ttl_secs = 5u32;
        let set_at = std::time::Instant::now();
        cache.set(1u32, &value(0xA1), Some(ttl_secs)).expect("set should succeed");

        // The fillers scale with key 1: promoting one of them has to push the
        // main queue's fast segment past its trigger on its own, which
        // 12-byte fillers (~16 migrating bytes apiece) never could.
        for key in 2u32..=6 {
            cache.set(key, &value(0xC3), None).expect("set should succeed");
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

    // ── eviction priority ─────────────────────────────────────────────────

    #[test]
    fn terminal_eviction_prefers_one_access_queue_over_main_queue() {
        ensure_pmem_allocator_warm();

        // Each object here accounts for 122 bytes (measured, not derived --
        // base bytes plus this policy's fixed per-object overhead). The old
        // config (max_size 200, fast_tier_size 200, ratio 1.0) was wrong three
        // ways: only ONE object fits in 200 bytes, so no one-access key could
        // ever still "remain" for the final assertion to be about; and a ratio
        // of 1.0 zeroed `effective_main_fast_capacity` (fast_capacity minus
        // one_access_capacity), so key 1 self-demoted to slow instead of
        // staying Fast as asserted below.
        //
        // max_size 500 holds four objects; one_access_capacity is 0.4 * 500 =
        // 200 (one key at a time, so sets 2..=10 generate real one-access
        // eviction pressure) and `effective_main_fast_capacity` is 400 - 200 =
        // 200, comfortably holding key 1 in the main queue's FAST segment.
        const MAX_SIZE: u64 = 512;

        let cache = PaperCache::<u32, TieredBuffer>::new(
            MAX_SIZE,
            CacheTierSize::Bytes(400), PaperPolicy::S3FifoGhostLazyDemotionFastAdmissionMidpointHybrid(0.4)).expect("cache should construct");

        cache.set(1u32, b"payload bytes 1", None).expect("set should succeed");
        cache.get(&1u32).expect("get should succeed");
        assert_eq!(cache.tier_of(&1u32), Some(Tier::Fast));

        // Paced, not burst. `apply_evictions` triggers on `status.used_size()`,
        // which `set()` updates synchronously on the API thread, but picks its
        // victim from the policy stack, which the worker updates asynchronously.
        // A tight loop can therefore leave eviction pressure genuinely present
        // while the stack still knows only about key 1 -- so `evict_one` finds
        // nothing in the one-access queue and takes the main-queue key this test
        // is asserting about. Same lockstep fix already documented for
        // `lfu_hybrid_cache`'s equivalent burst-admission test.
        for key in 2u32..=10 {
            let _ = cache.set(key, b"payload bytes", None);
            assert!(
                wait_until(MIGRATION_TIMEOUT, || {
                    cache.status().map(|s| s.used_size() <= MAX_SIZE).unwrap_or(false)
                }),
                "eviction should keep used_size within max_size between admissions",
            );
        }

        let evicted = wait_until(MIGRATION_TIMEOUT, || {
            cache.hybrid_stats().evictions >= 1
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
            1_048_576,
            CacheTierSize::Bytes(1_048_576), PaperPolicy::S3FifoGhostLazyDemotionFastAdmissionMidpointHybrid(ONE_ACCESS_RATIO)).expect("cache should construct");

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
        let result = PaperCache::<u32, TieredBuffer>::new(1_024, CacheTierSize::Bytes(0), PaperPolicy::S3FifoGhostLazyDemotionFastAdmissionMidpointHybrid(0.5));
        assert!(matches!(result, Err(CacheError::InvalidFastTierSize)));
    }

    #[test]
    fn invalid_one_access_ratio_is_rejected() {
        assert!(matches!(
            PaperCache::<u32, TieredBuffer>::new(1_024, CacheTierSize::Bytes(512), PaperPolicy::S3FifoGhostLazyDemotionFastAdmissionMidpointHybrid(1.5)),
            Err(CacheError::InvalidPolicy),
        ));
    }

    #[test]
    fn del_removes_key_from_whichever_tier_it_is_in() {
        ensure_pmem_allocator_warm();

        let cache = PaperCache::<u32, TieredBuffer>::new(
            1_048_576,
            CacheTierSize::Bytes(1_048_576), PaperPolicy::S3FifoGhostLazyDemotionFastAdmissionMidpointHybrid(ONE_ACCESS_RATIO)).expect("cache should construct");

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
            1_048_576,
            CacheTierSize::Bytes(1_048_576), PaperPolicy::S3FifoGhostLazyDemotionFastAdmissionMidpointHybrid(ONE_ACCESS_RATIO)).expect("cache should construct");

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
}
