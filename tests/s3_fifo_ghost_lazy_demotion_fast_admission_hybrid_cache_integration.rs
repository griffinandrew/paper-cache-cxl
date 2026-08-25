/*
 * Copyright (c) Kia Shakiba
 *
 * This source code is licensed under the GNU AGPLv3 license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! Integration tests for the
//! `s3_fifo_ghost_lazy_demotion_fast_admission_hybrid_cache` feature.
//!
//! Run with nightly (required for `allocator_api` via `key_value_pmem`):
//!   cargo +nightly test --test s3_fifo_ghost_lazy_demotion_fast_admission_hybrid_cache_integration --features s3_fifo_ghost_lazy_demotion_fast_admission_hybrid_cache
//!
//! Same one-`PaperCache<K, TieredBuffer>` architecture, ghost-queue
//! lifecycle, demotion-time reprieve, and eviction-time second-chance
//! mechanic as `s3_fifo_ghost_lazy_demotion_hybrid_cache` — see that
//! feature's integration test file for the shared coverage; this file
//! mirrors it end to end with tests adapted for the one change: the
//! one-access queue is now DRAM-resident, so admission is Fast, not Slow.
//! Adds two tests specific to the new behavior: the shared-DRAM-budget
//! accounting (`one_access_ratio_can_reserve_the_entire_fast_budget_forcing_immediate_demotion`)
//! and ghost-hit-lands-in-main-not-one-access-again
//! (`a_key_that_ages_out_and_is_readmitted_lands_directly_in_main_queue`).

#[cfg(feature = "s3_fifo_ghost_lazy_demotion_fast_admission_hybrid_cache")]
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

    /// Unlike the other hybrid designs' equivalent helper, admission here
    /// is always Fast (see `hybrid_policy::admission_tier`),
    /// so a warm-up key just sitting in the one-access queue never touches
    /// PMEM at all. Force a real demotion instead (a tiny effective
    /// main-fast budget makes a single promoted key self-demote
    /// immediately), which is what actually allocates through the
    /// UMF/TBB pool for the first time in this process.
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
        let cache = PaperCache::<u32, TieredBuffer>::new(1_048_576, CacheTierSize::Bytes(1_024), PaperPolicy::S3FifoGhostLazyDemotionFastAdmissionHybrid(0.001))
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
    /// `ONE_ACCESS_RATIO * max_size` gives the one-access queue 1024 bytes
    /// (~8 objects at this policy's measured 122-byte accounted size), enough
    /// that admission never self-evicts. Because
    /// `effective_main_fast_capacity` is `fast_capacity - one_access_capacity`,
    /// each test below adds `ONE_ACCESS_RESERVE` to its fast-tier size so the
    /// main queue's own budget stays exactly the value that test intends.
    ///
    /// Do not raise it to dodge a `resize()` rejection (`resize` re-derives
    /// both budgets against the new size and refuses one that rounds to zero).
    /// `a_key_with_no_ghost_history_still_lands_in_the_one_access_queue_fast`
    /// needs this 1024-byte budget to be overflowed by ~2200 bytes of filler,
    /// so the resizing test below states its own `max_size`/ratio pair instead.
    /// 1/1024 exactly, so `ratio * max_size` is 2^20 / 2^10 = 1024 with
    /// no truncation -- the reserve below is the exact product, not a
    /// rounded stand-in for it.
    const ONE_ACCESS_RATIO: f64 = 0.000_976_562_5;
    const ONE_ACCESS_RESERVE: u64 = 1_024;

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
            CacheTierSize::Bytes(1_048_576), PaperPolicy::S3FifoGhostLazyDemotionFastAdmissionHybrid(ONE_ACCESS_RATIO)).expect("cache should construct");

        cache.set(1u32, b"hello world", None).expect("set should succeed");

        assert_eq!(cache.tier_of(&1u32), Some(Tier::Fast));
        assert_eq!(cache.get(&1u32).unwrap(), b"hello world");
    }

    #[test]
    fn reaccessing_a_one_access_key_stays_fast_without_a_counted_promotion() {
        ensure_pmem_allocator_warm();

        // 0.5, not 1.0: the get() below promotes this key out of the one-access
        // queue into the main queue, and at a ratio of 1.0 the one-access
        // reservation consumes the entire fast tier, leaving
        // `effective_main_fast_capacity` at 0 -- so `settle_fast_tier` would
        // demote the key to Slow inside that same worker event, contradicting
        // the assertion below. 0.5 leaves the main queue 524_288 bytes.
        let cache = PaperCache::<u32, TieredBuffer>::new(
            1_048_576,
            CacheTierSize::Bytes(1_048_576), PaperPolicy::S3FifoGhostLazyDemotionFastAdmissionHybrid(0.5)).expect("cache should construct");

        cache.set(1u32, b"hello world", None).expect("set should succeed");

        let promotions_before = cache.hybrid_stats().promotions;

        cache.get(&1u32).expect("get should succeed");
        std::thread::sleep(std::time::Duration::from_millis(300));

        // Trivially still Fast (admission already put it there) -- the
        // real assertion is that this promotion out of the one-access
        // queue did NOT get counted, since the key's bytes never actually
        // moved (see this feature's stack module doc's "no more redundant
        // Fast→Fast copies" section). Contrast with
        // `s3_fifo_ghost_lazy_demotion_hybrid_cache`'s equivalent test,
        // which asserts `promotions >= 1` for the same access.
        assert_eq!(cache.tier_of(&1u32), Some(Tier::Fast));
        assert_eq!(
            cache.hybrid_stats().promotions,
            promotions_before,
        );
    }

    // ── ghost queue: still governs Main-vs-one-access placement ────────────

    #[test]
    fn a_key_that_ages_out_and_is_readmitted_lands_directly_in_main_queue() {
        ensure_pmem_allocator_warm();

        let cache = PaperCache::<u32, TieredBuffer>::new(1_048_576, CacheTierSize::Bytes(1_048_576), PaperPolicy::S3FifoGhostLazyDemotionFastAdmissionHybrid(0.00004)).expect("cache should construct");

        cache.set(1u32, b"first value 123", None).expect("set should succeed");
        assert_eq!(cache.tier_of(&1u32), Some(Tier::Fast)); // admission is always Fast now

        cache.set(2u32, b"second value 45", None).expect("set should succeed");

        let evicted = wait_until(MIGRATION_TIMEOUT, || !cache.has(&1u32));
        assert!(evicted, "key 1 should have aged out of the one-access queue");

        cache.set(1u32, b"first value 123", None).expect("re-set should succeed");
        assert_eq!(cache.tier_of(&1u32), Some(Tier::Fast)); // trivially true either way

        // The real proof this landed in the MAIN queue (not another
        // one-access admission that also happens to be Fast): overfill the
        // one-access queue again with fresh keys and confirm key 1
        // survives. If it had landed back in the one-access queue, this
        // same pressure would evict it to ghost again exactly like the
        // first time.
        for extra_key in 100u32..110u32 {
            cache.set(extra_key, b"filler filler!!", None).expect("set should succeed");
        }
        std::thread::sleep(std::time::Duration::from_millis(300));

        assert!(
            cache.has(&1u32),
            "a ghost-queue hit should land directly in the main queue, protected from one-access aging",
        );
    }

    #[test]
    fn a_key_with_no_ghost_history_still_lands_in_the_one_access_queue_fast() {
        ensure_pmem_allocator_warm();

        // Admission here is Fast for *every* key absent from the object map
        // (`hybrid_policy::admission_tier`), so both assertions below are
        // "trivially true either way" in the same sense
        // `a_key_that_ages_out_and_is_readmitted_lands_directly_in_main_queue`
        // flags -- and in this variant the tier can never separate a ghost
        // HIT from a ghost MISS at all, since the main queue's fast segment
        // and the one-access queue are both DRAM. What separates them is
        // WHICH DRAM queue the key is in, and one-access aging pressure is
        // what makes that observable.
        let cache = PaperCache::<u32, TieredBuffer>::new(
            1_048_576,
            CacheTierSize::Bytes(1_048_576), PaperPolicy::S3FifoGhostLazyDemotionFastAdmissionHybrid(ONE_ACCESS_RATIO)).expect("cache should construct");

        // Fat on purpose: ten of these overflow the one-access budget
        // (`ONE_ACCESS_RATIO * max_size` = 1024 bytes) outright, so the aging
        // below doesn't lean on this policy's measured 122-byte per-object
        // accounting.
        const FILLER: &[u8; 200] = &[b'f'; 200];

        // Give key 1 -- and only key 1 -- a ghost record. An aged-out
        // one-access key in this variant is removed outright and remembered
        // only as a bare ghost key (see this file's `ONE_ACCESS_RATIO` doc).
        cache.set(1u32, b"first value 123", None).expect("set should succeed");
        for filler in 100u32..110 {
            cache.set(filler, FILLER, None).expect("filler set should succeed");
        }
        assert!(
            wait_until(MIGRATION_TIMEOUT, || !cache.has(&1u32)),
            "key 1 should have aged out of the one-access queue into the ghost queue",
        );

        // Re-admit key 1 (ghost HIT) BEFORE admitting key 9 for the first
        // time (ghost MISS), so key 1 is the older of the two in one-access
        // FIFO order: were the ghost lookup to miss and put key 1 back in the
        // one-access queue, the pressure below would take key 1 out first,
        // and key 1 would not be there for the final assertion.
        cache.set(1u32, b"first value 123", None).expect("re-set should succeed");
        cache.set(9u32, b"brand new value", None).expect("set should succeed");

        assert_eq!(cache.tier_of(&9u32), Some(Tier::Fast)); // trivially true either way
        assert_eq!(cache.tier_of(&1u32), Some(Tier::Fast)); // trivially true either way

        // The discriminating half: a second burst ages the one-access queue
        // exactly as the first one did. Key 9 has no ghost record, so it is a
        // one-access resident and goes; key 1 had one, so it is a main-queue
        // resident and stays.
        for filler in 110u32..120 {
            cache.set(filler, FILLER, None).expect("filler set should succeed");
        }
        assert!(
            wait_until(MIGRATION_TIMEOUT, || !cache.has(&9u32)),
            "a key with no ghost record should be a one-access resident and age out",
        );

        assert!(
            cache.has(&1u32),
            "a ghost-queue hit should land in the main queue, immune to one-access aging",
        );
        assert_eq!(cache.get(&1u32).unwrap(), b"first value 123");
    }

    // ── main-queue behavior (unaffected by the fast-tier one-access queue) ─

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
            CacheTierSize::Bytes(1_048_576), PaperPolicy::S3FifoGhostLazyDemotionFastAdmissionHybrid(0.5)).expect("cache should construct");

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

        // Leaves 40 bytes of effective room for the main queue (this test is
        // specifically about main-queue eviction priority, not the one-access
        // budget) -- see ONE_ACCESS_RATIO for why that reservation is added on
        // top rather than the ratio simply being 0.0.
        //
        // max_size 131_072 with an explicit 1/128, not 1_048_576 with
        // ONE_ACCESS_RATIO: the PRODUCT is what this fixture depends on and it
        // is unchanged -- 0.0078125 * 131_072 is the same 1024-byte one-access
        // reservation `ONE_ACCESS_RESERVE` adds back to the fast tier below, so
        // the main queue's effective fast budget is still exactly 40 bytes, one
        // key. (`effective_main_fast_capacity` is `fast_capacity -
        // one_access_capacity`, so a product ABOVE the reserve would saturate
        // that subtraction to zero rather than merely shrinking it.) The ratio
        // has to move because `resize()` re-derives both budgets against the
        // NEW size and rejects one that rounds to zero: ONE_ACCESS_RATIO * 180
        // is 0, while 0.0078125 * 180 is 1 byte (main 178).
        // Raising ONE_ACCESS_RATIO itself is not an option -- see its doc --
        // hence the local pair. 131_072 is still ~400x what this test admits,
        // so the global `used_size() > max_size` trigger stays quiet until the
        // resize fires it, exactly as before.
        let cache = PaperCache::<u32, TieredBuffer>::new(
            131_072,
            CacheTierSize::Bytes(40 + ONE_ACCESS_RESERVE), PaperPolicy::S3FifoGhostLazyDemotionFastAdmissionHybrid(0.007_812_5)).expect("cache should construct");

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

        // Same 40 bytes of effective main-queue room, and the same one-access
        // reservation on top, as the second-chance test above.
        let cache = PaperCache::<u32, TieredBuffer>::new(
            1_048_576,
            CacheTierSize::Bytes(40 + ONE_ACCESS_RESERVE), PaperPolicy::S3FifoGhostLazyDemotionFastAdmissionHybrid(ONE_ACCESS_RATIO)).expect("cache should construct");

        cache.set(1u32, b"payload bytes A", None).expect("set should succeed");
        cache.get(&1u32).expect("get should succeed");
        assert_eq!(cache.tier_of(&1u32), Some(Tier::Fast));

        let promotions_before = cache.hybrid_stats().promotions;
        let demotions_before = cache.hybrid_stats().demotions;

        // Touch key 1 again while it's still Fast -- sets its reference
        // bit, no reorder, no migration.
        cache.get(&1u32).expect("get should succeed");
        std::thread::sleep(std::time::Duration::from_millis(300));
        assert_eq!(cache.tier_of(&1u32), Some(Tier::Fast));

        // Promoting key 2 forces fast-tier pressure. Key 1's bit is set,
        // so it must be reprieved (stay Fast) and key 2 -- the only other
        // candidate, with a clear bit -- must be demoted in its place
        // instead. Unlike the non-fast-admission variant, `tier_of(&2u32)`
        // IS a usable signal here: key 2's bytes really do start Fast (per
        // this variant's admission rule) and only become Slow once the
        // worker's real demotion physically runs, so waiting on it
        // directly is valid -- still cross-checked against the demotion
        // counter for extra rigor.
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

    // ── the signature new accounting mechanic: shared DRAM budget ──────────

    #[test]
    fn one_access_ratio_can_reserve_the_entire_fast_budget_forcing_immediate_demotion() {
        ensure_pmem_allocator_warm();

        // one_access_capacity = 0.0001 * 1_048_576 = 104, exactly consuming
        // the entire 100-byte fast_capacity and leaving zero effective room
        // for the main queue's fast segment (see this feature's stack
        // module doc's "Accounting" section). A single promoted key must
        // self-demote immediately as a result, even though 100 bytes would
        // otherwise be plenty of room for one small object -- this is
        // exactly the "account for this when sizing the fast tier" concern
        // this feature exists to get right.
        let cache = PaperCache::<u32, TieredBuffer>::new(
            1_048_576,
            CacheTierSize::Bytes(100), PaperPolicy::S3FifoGhostLazyDemotionFastAdmissionHybrid(0.0001)).expect("cache should construct");

        cache.set(1u32, b"payload bytes A", None).expect("set should succeed");
        cache.get(&1u32).expect("get should succeed");

        let demoted = wait_until(MIGRATION_TIMEOUT, || cache.tier_of(&1u32) == Some(Tier::Slow));
        assert!(
            demoted,
            "reserving the entire fast budget for the one-access queue should force every \
             main-queue promotion to self-demote immediately",
        );
        assert_eq!(cache.get(&1u32).unwrap(), b"payload bytes A");
    }

    // ── TTL ───────────────────────────────────────────────────────────────

    const TTL_FAST_TIER: u64 = 200;

    #[test]
    fn ttl_survives_a_demotion() {
        ensure_pmem_allocator_warm();

        let cache = PaperCache::<u32, TieredBuffer>::new(
            1_048_576,
            CacheTierSize::Bytes(TTL_FAST_TIER + ONE_ACCESS_RESERVE), PaperPolicy::S3FifoGhostLazyDemotionFastAdmissionHybrid(ONE_ACCESS_RATIO)).expect("cache should construct");

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
            CacheTierSize::Bytes(400), PaperPolicy::S3FifoGhostLazyDemotionFastAdmissionHybrid(0.4)).expect("cache should construct");

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
            CacheTierSize::Bytes(1_048_576), PaperPolicy::S3FifoGhostLazyDemotionFastAdmissionHybrid(ONE_ACCESS_RATIO)).expect("cache should construct");

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
        let result = PaperCache::<u32, TieredBuffer>::new(1_024, CacheTierSize::Bytes(0), PaperPolicy::S3FifoGhostLazyDemotionFastAdmissionHybrid(0.5));
        assert!(matches!(result, Err(CacheError::InvalidFastTierSize)));
    }

    #[test]
    fn invalid_one_access_ratio_is_rejected() {
        assert!(matches!(
            PaperCache::<u32, TieredBuffer>::new(1_024, CacheTierSize::Bytes(512), PaperPolicy::S3FifoGhostLazyDemotionFastAdmissionHybrid(1.5)),
            Err(CacheError::InvalidPolicy),
        ));
    }

    #[test]
    fn del_removes_key_from_whichever_tier_it_is_in() {
        ensure_pmem_allocator_warm();

        let cache = PaperCache::<u32, TieredBuffer>::new(
            1_048_576,
            CacheTierSize::Bytes(1_048_576), PaperPolicy::S3FifoGhostLazyDemotionFastAdmissionHybrid(ONE_ACCESS_RATIO)).expect("cache should construct");

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
            CacheTierSize::Bytes(1_048_576), PaperPolicy::S3FifoGhostLazyDemotionFastAdmissionHybrid(ONE_ACCESS_RATIO)).expect("cache should construct");

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
