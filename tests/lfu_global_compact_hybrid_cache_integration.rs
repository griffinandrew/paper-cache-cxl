/*
 * Copyright (c) Kia Shakiba
 *
 * This source code is licensed under the GNU AGPLv3 license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! Integration tests for the `lfu_global_compact_hybrid_cache` feature --
//! `PaperPolicy::LfuGlobalCompactHybrid`, `"lfu-global-compact-hybrid"`.
//!
//! The design is `lfu_compact_hybrid_cache` with two rules changed, and this
//! suite is about those two, through a real `PaperCache` whose bytes really
//! move between DRAM and PMEM:
//!
//!   * GLOBAL EVICTION. The victim is the least `(count, stamp)` across both
//!     tiers -- upstream LFU's victim -- so a fast key that was never reused
//!     can be evicted straight out of DRAM ahead of newer slow keys, which
//!     `lfu-compact-hybrid`'s slow-first rule would shield it from. A fast
//!     victim is freed where it is: no demotion, and the latch stays shut.
//!   * THE CREDIT-LIMITED REFILL. The DRAM such an eviction frees is credited,
//!     and a slow key whose hit TIES the fast minimum may take it back -- with
//!     no demotion, and only while credit lasts.
//!
//! Everything else -- admission, the latch and its `admission_tier` mirror,
//! strict promotion, the settle -- is the base design's, and
//! `tests/lfu_compact_hybrid_cache_integration.rs` covers it there. A few of
//! those checks are repeated here against the new policy value, because an
//! `admission_tier` arm that puts this policy under the wrong admission
//! contract fails silently. A MISSING arm no longer compiles, since that match
//! lost its catch-all; that file's module doc records the bug the catch-all
//! let through, when a missing arm still compiled.
//!
//! This suite targets the DashMap object store, like the base design's. Under
//! `merged_object_store` the store admits and places keys on its own terms, so
//! `seat_the_cohort`'s fixture does not hold there and the tests built on it
//! fail, as the base design's fixture-built tests do; the merged store's version
//! of this policy is checked by the unit and fidelity tests in
//! `src/merged_store.rs` and `src/worker/policy/policy_stack/merged_stack.rs`,
//! and constructed end to end by `tests/merged_object_store_policy_refusal.rs`.
//!
//! Run with nightly (required for `allocator_api` via `key_value_pmem`):
//!   cargo +nightly test --test lfu_global_compact_hybrid_cache_integration --features lfu_global_compact_hybrid_cache

#[cfg(feature = "lfu_global_compact_hybrid_cache")]
mod lfu_global_tests {
    use paper_cache::{CacheError, CacheTierSize, PaperCache, PaperPolicy, Tier, TieredBuffer};

    const TIMEOUT: std::time::Duration = std::time::Duration::from_secs(10);

    /// Long enough for the policy worker to drain a handful of events. Used
    /// where the thing to wait for -- a bump -- has no observable effect.
    const SETTLE: std::time::Duration = std::time::Duration::from_millis(300);

    /// ~1 KB, so the per-object bookkeeping is a small fraction of each value
    /// and the fast-tier budgets below have a wide margin.
    const VALUE_LEN: usize = 1024;

    fn value(seed: u8) -> Vec<u8> {
        vec![seed; VALUE_LEN]
    }

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

    /// The one-time PMEM pool warm-up, and the metadata reservation turned off
    /// for this process -- the scenarios below place objects by value bytes
    /// alone. See `lfu_compact_hybrid_cache_integration.rs`, which does the
    /// same for the same reasons.
    fn ensure_pmem_allocator_warm() {
        unsafe { std::env::set_var("PAPER_DISABLE_SHARED_OVERHEAD", "1") };

        let cache = PaperCache::<u32, TieredBuffer>::new(
            1_048_576,
            CacheTierSize::Bytes(1),
            PaperPolicy::LfuGlobalCompactHybrid,
        )
        .expect("warm-up cache should construct");

        cache.set(0u32, b"warm", None).expect("warm-up set should succeed");

        let ready = wait_until(std::time::Duration::from_secs(90), || {
            cache.tier_of(&0u32) == Some(Tier::Slow)
        });
        assert!(ready, "PMEM allocator warm-up should complete within 90s");
    }

    /// Seats the history both eviction scopes are compared on:
    ///
    ///   1, 2   fast at count 1 (2.5 objects of DRAM), then 1 is read once
    ///   3,4,5  slow at count 1 -- 3 finds the tier full and shuts the latch,
    ///          4 and 5 are admitted straight to PMEM behind it
    ///
    /// So fast key 2 is tied at the minimum count with three slow keys, and it
    /// reached that count before any of them. Returns the cache and one
    /// object's stack-accounted bytes.
    fn seat_the_cohort(policy: PaperPolicy) -> (PaperCache<u32, TieredBuffer>, u64) {
        ensure_pmem_allocator_warm();

        let cache = PaperCache::<u32, TieredBuffer>::new(
            1_048_576,
            CacheTierSize::Bytes(1_048_576),
            policy,
        )
        .expect("cache should construct");

        cache.set(1u32, &value(1), None).expect("set should succeed");
        cache.set(2u32, &value(2), None).expect("set should succeed");
        cache.get(&1u32).expect("get should succeed");

        assert!(
            wait_until(TIMEOUT, || cache.hybrid_stats().fast_objects == 2),
            "both keys should be resident in the fast tier",
        );
        std::thread::sleep(SETTLE);

        // Measured, not assumed: what the stack charges one of these objects.
        let per_object = cache.hybrid_stats().fast_bytes_used / 2;

        // Room for 2.5 objects: the two residents stay (2 <= 0.98 x 2.5), and
        // a third cannot be admitted.
        cache
            .set_fast_tier_size(CacheTierSize::Bytes(per_object * 5 / 2))
            .expect("resize should succeed");

        cache.set(3u32, &value(3), None).expect("set should succeed");

        // Wait for key 3's admission-to-slow correction before the next sets:
        // the worker mirrors the latch onto the status first, so after this
        // `set()` builds 4 and 5 in PMEM directly.
        assert!(
            wait_until(TIMEOUT, || cache.tier_of(&3u32) == Some(Tier::Slow)),
            "key 3 should have been admitted to the slow tier",
        );

        cache.set(4u32, &value(4), None).expect("set should succeed");
        cache.set(5u32, &value(5), None).expect("set should succeed");

        assert!(
            wait_until(TIMEOUT, || {
                let stats = cache.hybrid_stats();
                stats.fast_objects == 2 && stats.slow_objects == 3
            }),
            "the cohort should be two fast keys and three slow ones",
        );

        assert_eq!(cache.tier_of(&4u32), Some(Tier::Slow), "admission is latched");
        assert_eq!(cache.tier_of(&5u32), Some(Tier::Slow), "admission is latched");
        assert_eq!(cache.hybrid_stats().demotions, 0, "nothing has been displaced yet");

        (cache, per_object)
    }

    /// Shrinks `max_size` to one byte under what the cache holds, so the
    /// worker evicts exactly one object, and waits for it.
    fn evict_exactly_one(cache: &PaperCache<u32, TieredBuffer>) {
        let before = cache.hybrid_stats().evictions;
        let used = cache.status().expect("status").used_size();

        cache.resize(used - 1).expect("resize should succeed");

        assert!(
            wait_until(TIMEOUT, || cache.hybrid_stats().evictions == before + 1),
            "shrinking max_size under the cache should evict",
        );
        std::thread::sleep(SETTLE);
        assert_eq!(cache.hybrid_stats().evictions, before + 1, "exactly one object should go");
    }

    // ── the policy value ──────────────────────────────────────────────────

    #[test]
    fn the_policy_string_round_trips_and_constructs_a_cache() {
        let policy = "lfu-global-compact-hybrid".parse::<PaperPolicy>().expect("the string parses");

        assert_eq!(policy, PaperPolicy::LfuGlobalCompactHybrid);
        assert_eq!(policy.to_string(), "lfu-global-compact-hybrid");
        assert_ne!(policy, PaperPolicy::LfuCompactHybrid, "a distinct policy, not an alias");
        assert!(policy.is_hybrid());

        let cache = PaperCache::<u32, TieredBuffer>::new(
            1_048_576,
            CacheTierSize::Bytes(1_048_576),
            policy,
        )
        .expect("cache should construct");

        cache.set(1u32, b"hello world", None).expect("set should succeed");

        assert_eq!(cache.tier_of(&1u32), Some(Tier::Fast), "a new key lands fast while there is room");
        assert_eq!(cache.get(&1u32).unwrap(), b"hello world");
        assert_eq!(cache.status().expect("status").policy(), PaperPolicy::LfuGlobalCompactHybrid);
    }

    #[test]
    fn the_constructor_validates_sizes_as_for_every_hybrid() {
        let zero = PaperCache::<u32, TieredBuffer>::new(1_024, CacheTierSize::Bytes(0), PaperPolicy::LfuGlobalCompactHybrid);
        assert!(matches!(zero, Err(CacheError::InvalidFastTierSize)));

        let too_big = PaperCache::<u32, TieredBuffer>::new(1_024, CacheTierSize::Bytes(2_048), PaperPolicy::LfuGlobalCompactHybrid);
        assert!(matches!(too_big, Err(CacheError::InvalidFastTierSize)));

        let empty = PaperCache::<u32, TieredBuffer>::new(0, CacheTierSize::Bytes(100), PaperPolicy::LfuGlobalCompactHybrid);
        assert!(matches!(empty, Err(CacheError::ZeroCacheSize)));
    }

    // ── admission: the base design's contract, under the new policy value ──

    /// `hybrid_policy::admission_tier` must give this policy the latch mirror
    /// `LfuCompactHybrid` has: once admission is latched, `set()` builds a
    /// brand-new key in PMEM synchronously. Read with no `wait_until`: a key
    /// built in DRAM would read back `Fast` here and only turn slow later.
    #[test]
    fn set_places_a_brand_new_key_directly_in_slow_once_admission_is_latched() {
        let (cache, _) = seat_the_cohort(PaperPolicy::LfuGlobalCompactHybrid);

        cache.set(6u32, &value(6), None).expect("set should succeed");
        assert_eq!(cache.tier_of(&6u32), Some(Tier::Slow));

        // An existing fast key's re-set is an access, not an admission.
        cache.set(1u32, &value(9), None).expect("set should succeed");
        assert_eq!(cache.tier_of(&1u32), Some(Tier::Fast));
    }

    // ── global eviction ────────────────────────────────────────────────────

    /// The contract that separates the two policies, on one history. Global
    /// eviction takes fast key 2 -- count 1, and older than 3, 4 and 5 --
    /// straight out of DRAM: it is evicted, not demoted, the slow keys all
    /// survive, and the one remaining fast key is untouched.
    #[test]
    fn a_never_reused_fast_key_is_evicted_ahead_of_newer_slow_keys() {
        let (cache, per_object) = seat_the_cohort(PaperPolicy::LfuGlobalCompactHybrid);

        evict_exactly_one(&cache);

        assert!(!cache.has(&2u32), "the never-reused fast key should have been the victim");
        assert_eq!(cache.tier_of(&2u32), None, "and gone from both tiers");

        for key in [1u32, 3, 4, 5] {
            assert!(cache.has(&key), "key {key} should have survived");
        }

        assert_eq!(cache.tier_of(&1u32), Some(Tier::Fast));

        let stats = cache.hybrid_stats();
        assert_eq!(stats.demotions, 0, "a fast victim is evicted in place, never demoted");
        assert_eq!((stats.fast_objects, stats.slow_objects), (1, 3));
        assert_eq!(stats.fast_bytes_used, per_object, "the victim's bytes left the fast tier");

        // Evicting from DRAM does not reopen admission: a brand-new key still
        // goes straight to PMEM.
        cache.set(6u32, &value(6), None).expect("set should succeed");
        assert_eq!(cache.tier_of(&6u32), Some(Tier::Slow));
    }

    /// The same history under `lfu-compact-hybrid`, whose slow-first rule
    /// shields fast key 2 and evicts slow key 3 instead. Kept beside the test
    /// above so the difference is stated by the suite, not only by the docs.
    #[test]
    fn the_slow_first_policy_shields_the_same_fast_key() {
        let (cache, _) = seat_the_cohort(PaperPolicy::LfuCompactHybrid);

        evict_exactly_one(&cache);

        assert!(!cache.has(&3u32), "slow-first takes the oldest slow key");
        assert!(cache.has(&2u32), "and shields the fast one");
        assert_eq!(cache.tier_of(&2u32), Some(Tier::Fast));
    }

    // ── the refill ─────────────────────────────────────────────────────────

    /// Evicting fast key 2 frees one object of DRAM and credits it. Key 4's
    /// read takes it to count 2, which only TIES key 1 at the fast minimum --
    /// not enough for a promotion under the strict rule -- but the credit
    /// covers it and the tier has the room, so it is promoted into the freed
    /// space, with no demotion. Key 5's read ties too, and finds neither
    /// credit nor room: two objects are fast again and a third would exceed
    /// 0.98 x 2.5 objects, so the room test alone refuses it here. The credit
    /// refusing a tie that DOES have room is
    /// `a_fast_eviction_earns_credit_that_a_tie_spends_without_a_demotion`'s
    /// job, in the stack's own unit tests.
    #[test]
    fn a_tie_refills_the_room_a_fast_eviction_freed() {
        let (cache, per_object) = seat_the_cohort(PaperPolicy::LfuGlobalCompactHybrid);

        evict_exactly_one(&cache);
        assert!(!cache.has(&2u32), "the fixture's victim should be fast key 2");

        let promotions = cache.hybrid_stats().promotions;

        assert_eq!(cache.get(&4u32).unwrap(), value(4));

        assert!(
            wait_until(TIMEOUT, || cache.tier_of(&4u32) == Some(Tier::Fast)),
            "a slow key tied with the fast minimum should refill the freed room",
        );

        let stats = cache.hybrid_stats();
        assert_eq!(stats.demotions, 0, "a refill displaces nothing");
        assert!(stats.promotions > promotions, "the refill is a promotion");
        assert!(
            wait_until(TIMEOUT, || cache.hybrid_stats().fast_bytes_used == 2 * per_object),
            "the refilled key's bytes are charged to the fast tier",
        );

        assert_eq!(cache.get(&5u32).unwrap(), value(5));
        std::thread::sleep(SETTLE);

        assert_eq!(cache.tier_of(&5u32), Some(Tier::Slow), "no credit and no room, so a tie stays slow");
        assert_eq!(cache.tier_of(&1u32), Some(Tier::Fast));
        assert_eq!(cache.hybrid_stats().demotions, 0);

        // The bytes survived both moves intact.
        assert_eq!(cache.get(&4u32).unwrap(), value(4));
        assert_eq!(cache.get(&1u32).unwrap(), value(1));
    }

    /// Without a fast eviction there is no credit, and the strict rule stands:
    /// a read that only ties the fast minimum does not promote, however much
    /// room the budget shows -- here the room left by growing the tier. Key
    /// 2 is read first so the fast minimum is 2 (keys 1 and 2), and key 4's
    /// read then ties it rather than exceeding it.
    #[test]
    fn a_tie_without_credit_does_not_promote() {
        let (cache, _) = seat_the_cohort(PaperPolicy::LfuGlobalCompactHybrid);

        cache
            .set_fast_tier_size(CacheTierSize::Bytes(1_048_576))
            .expect("resize should succeed");
        cache.get(&2u32).expect("get should succeed");
        std::thread::sleep(SETTLE);

        cache.get(&4u32).expect("get should succeed");
        std::thread::sleep(SETTLE);

        assert_eq!(cache.tier_of(&4u32), Some(Tier::Slow), "a tie must not promote without credit");
        assert_eq!(cache.hybrid_stats().demotions, 0);
    }

    // ── accounting ─────────────────────────────────────────────────────────

    /// Churn with evictions from both tiers, reads that promote and refill,
    /// and deletes. Every 50 keys the worker must catch up to a stack whose
    /// two tier counters add up to exactly what the map holds; at the end every
    /// key must be accounted for exactly once -- present, evicted, or deleted
    /// -- every present key must be in a tier, and the fast tier must hold no
    /// more than its budget.
    ///
    /// The checkpoints also PACE the churn, and that is deliberate. Left to
    /// run flat out, the API thread can get further ahead of the policy
    /// worker than the whole cache holds; the worker then evicts every key its
    /// stack has seen, finds itself still over capacity, and falls back to
    /// erasing arbitrary map entries whose `Set` events are still queued
    /// (`ERASE_FALLBACK`) -- which the stack then admits, so it ends up
    /// tracking keys the map no longer has. That is a property of the worker,
    /// not of this policy. Measured on this exact churn, unpaced, three runs
    /// per policy: one of three `lfu-compact-hybrid` runs ended with 305 keys
    /// tracked against 289 present, one of three `lru-compact-hybrid` runs
    /// with 312 against 290, and in both the excess was exactly the run's
    /// `ERASE_FALLBACK` count (16 and 22). Fifty keys is about 22 KB, a
    /// seventh of the cache, so the race cannot open here.
    #[test]
    fn accounting_and_tier_counters_stay_consistent_under_churn() {
        ensure_pmem_allocator_warm();

        const N: u32 = 600;

        let cache = PaperCache::<u32, TieredBuffer>::new(
            160 * 1_024,
            CacheTierSize::Bytes(40 * 1_024),
            PaperPolicy::LfuGlobalCompactHybrid,
        )
        .expect("cache should construct");

        let mut deleted = 0u64;

        for key in 1u32..=N {
            cache.set(key, &vec![key as u8; 256 + (key as usize % 7) * 64], None)
                .expect("set should succeed");

            // Reads of a hot-ish window behind the newest key, so counts climb
            // and ties at the fast minimum occur; misses are fine.
            for back in [3u32, 11, 29] {
                if key > back && key % back != 0 {
                    let _ = cache.get(&(key - back));
                }
            }

            if key % 50 == 0 && cache.del(&(key - 25)).is_ok() {
                deleted += 1;
            }

            if key % 50 == 0 {
                assert!(
                    wait_until(TIMEOUT, || {
                        let stats = cache.hybrid_stats();
                        stats.fast_objects + stats.slow_objects
                            == cache.status().expect("status").num_objects()
                    }),
                    "after key {key}: the stack's tier counters never matched the map",
                );
            }
        }

        assert!(cache.hybrid_stats().evictions > 0, "the churn should have evicted");
        std::thread::sleep(SETTLE);

        let stats = cache.hybrid_stats();
        let present: Vec<u32> = (1u32..=N).filter(|key| cache.has(key)).collect();

        assert_eq!(
            present.len() as u64 + stats.evictions + deleted,
            u64::from(N),
            "a key was lost or counted twice",
        );

        for key in &present {
            assert!(cache.tier_of(key).is_some(), "present key {key} is in no tier");
        }

        for key in (1u32..=N).filter(|key| !cache.has(key)) {
            assert_eq!(cache.tier_of(&key), None, "absent key {key} still has a tier");
        }

        // The stack's counters, not the physical tiers: under a burst of sets
        // the admission-latch mirror lags, and a key the stack admits slow can
        // be built in DRAM -- the base design's known placement gap (see
        // `MergedStore::insert`), not this policy's. The counters are what
        // eviction and the budget run on.
        assert_eq!(
            stats.fast_objects + stats.slow_objects,
            present.len() as u64,
            "the tier counters disagree with what is present",
        );

        assert!(stats.fast_bytes_used <= cache.fast_tier_size(), "the fast tier overran its budget");
        assert!(stats.slow_objects > 0 && stats.fast_objects > 0, "the churn left one tier empty");
    }

    #[test]
    fn wipe_clears_both_tiers() {
        let (cache, _) = seat_the_cohort(PaperPolicy::LfuGlobalCompactHybrid);

        cache.wipe().expect("wipe should succeed");

        for key in 1u32..=5 {
            assert!(!cache.has(&key));
            assert_eq!(cache.tier_of(&key), None);
        }

        assert!(
            wait_until(TIMEOUT, || {
                let stats = cache.hybrid_stats();
                stats.fast_objects == 0 && stats.slow_objects == 0 && stats.fast_bytes_used == 0
            }),
            "the tier gauges should drain to zero after a wipe",
        );
    }
}
