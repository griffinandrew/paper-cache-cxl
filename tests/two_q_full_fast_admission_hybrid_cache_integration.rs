/*
 * Copyright (c) Kia Shakiba
 *
 * This source code is licensed under the GNU AGPLv3 license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! Integration tests for the `two_q_full_fast_admission_hybrid_cache` feature.
//!
//! Run with nightly (required for `allocator_api` via `key_value_pmem`):
//!   cargo +nightly test --test two_q_full_fast_admission_hybrid_cache_integration --features two_q_full_fast_admission_hybrid_cache
//!
//! Same one-`PaperCache<K, TieredBuffer>` architecture as the other hybrid
//! integration suites — `tier_of` reads the tier directly off the single
//! object map.
//!
//! What distinguishes this design from `two_q_fast_admission_hybrid_cache`,
//! and therefore what this suite is actually for:
//!
//!   * Admission lands in `a1_in`, in the **fast** tier, on `set()` itself
//!   * A re-access of an `a1_in` object is a **no-op** — it does NOT promote
//!     (the fidelity point; the Simplified-2Q hybrids promote here)
//!   * An `a1_in` object that ages out is **demoted into `a1_out`**, not
//!     evicted: it stays in the cache, readable, in the slow tier
//!   * A re-access of an `a1_out` object is a genuine PMEM→DRAM promotion
//!     into `am`
//!   * Only `a1_out` overflow (`k_out`) drives capacity eviction, and
//!     eviction drains `a1_out` before `a1_in` and `am`
//!   * Once in `am`, an object behaves like `lru_hybrid_cache`
//!   * TTL survives a tier move; `set_fast_tier_size` / `resize` take effect
//!     at runtime; both ratios are range-checked at construction

#[cfg(feature = "two_q_full_fast_admission_hybrid_cache")]
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

    const MIGRATION_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(10);

    // ── sizing ────────────────────────────────────────────────────────────
    //
    // The arithmetic trap this design inherits from
    // `two_q_fast_admission_hybrid_cache`, and sharpens: `k_in` is
    // denominated in `max_size` but the budget it consumes is
    // `fast_tier_size`, because `a1_in` is DRAM. `effective_am_fast_capacity
    // = fast_tier_size - k_in * max_size`, so a `k_in` that looks tiny
    // against `max_size` can swallow the whole fast tier and leave `am` no
    // fast segment at all.
    //
    // `k_out` has no such interaction: it bounds PMEM, and is the ONLY
    // budget that drives capacity eviction here.
    //
    // The second trap is the UNIT those budgets are denominated in, and it
    // is the one these fixtures were sized against wrongly. Every queue
    // counter here — `a1_in_used`, `a1_out_used`, `am_fast_used` —
    // accumulates an entry's MIGRATING bytes: `base_size` minus the part that
    // stays in DRAM whichever tier the object is in. For a 64-byte value
    // under a `u32` key that is 84 - 20 = 64 bytes, since the key and the
    // expiry field never move and so are never charged to a tier. Only the
    // *incoming* object is measured in full: `settle_a1_in` tests
    // `a1_in_used + base_size(incoming) > a1_in_capacity`. So `a1_in` settles
    // at the largest R satisfying `(R - 1) * 64 + 84 <= a1_in_capacity`.
    //
    // The old constants (FAST_TIER 1_600, K_IN 0.04) counted 84 bytes per
    // object in both places, so every queue held ~30% more objects than the
    // comments claimed: that 819-byte `a1_in` held TWELVE of these rather
    // than nine, 24 sets aged only 12 keys out, and `set_and_age_out(24, 15)`
    // sat out its timeout waiting for a fifteenth that could never arrive.
    //
    //   a1_in reservation   = K_IN  * MAX_SIZE = 614 bytes -> 9 residents
    //                         (8 * 64 + 84 = 596 fits, 9 * 64 + 84 = 660 does
    //                         not), so 24 sets age exactly 15 keys out
    //   effective am fast   = FAST_TIER - 614  = 410 bytes -> 6 objects, well
    //                         short of the 15 * 64 = 960 bytes those aged keys
    //                         carry back into `am` on promotion, which is what
    //                         makes the main queue shed its LRU tail
    //   a1_out budget       = K_OUT * MAX_SIZE = 4_096 bytes (roomy, so the
    //                         demote-don't-evict path is what gets exercised)
    const MAX_SIZE: u64 = 20_480;
    const FAST_TIER: u64 = 1_024;
    const K_IN: f64 = 0.03;
    const K_OUT: f64 = 0.2;

    /// Cache sized by the constants above — use this for any test that needs
    /// a *reachable* fast `am` segment. Tests deliberately exercising a
    /// degenerate configuration construct their own.
    fn make_cache() -> PaperCache<u32, TieredBuffer> {
        PaperCache::<u32, TieredBuffer>::new(
            MAX_SIZE,
            CacheTierSize::Bytes(FAST_TIER),
            PaperPolicy::TwoQFullFastAdmissionHybrid(K_IN, K_OUT),
        ).expect("cache should construct")
    }


    /// Sets keys `1..=count` as 64-byte objects, then blocks until at least
    /// `expect_aged` of them have actually aged out of `a1_in` into `a1_out`,
    /// returning those keys oldest-first (= ascending, since `a1_in` demotes
    /// its tail and the tail is the earliest-admitted key).
    ///
    /// The wait is the whole point: `settle_a1_in` runs synchronously inside
    /// the policy worker's `insert`, but the `(key, Tier::Slow)` migrations it
    /// emits are applied off-thread, so a check made immediately after the
    /// `set` loop sees ZERO aged keys.
    ///
    /// Only the returned keys can be promoted into `am`: an `a1_in` hit is a
    /// complete no-op in this design, so re-`get`ting a key that never aged
    /// out moves nothing.
    fn set_and_age_out(
        cache: &PaperCache<u32, TieredBuffer>,
        count: u32,
        expect_aged: usize,
    ) -> Vec<u32> {
        for key in 1..=count {
            cache.set(key, &[key as u8; 64], None).expect("set should succeed");
        }

        assert!(
            wait_until(MIGRATION_TIMEOUT, || {
                (1..=count).filter(|key| cache.tier_of(key) == Some(Tier::Slow)).count() >= expect_aged
            }),
            "expected at least {expect_aged} of {count} keys to age out of a1_in into a1_out",
        );

        (1..=count)
            .filter(|key| cache.tier_of(key) == Some(Tier::Slow))
            .collect()
    }

    /// Re-accesses each of `keys` — which must currently be sitting in
    /// `a1_out` — and blocks until every one of those promotions has been
    /// applied.
    ///
    /// This is the ONLY route into `am`: an `a1_in` hit is a complete no-op,
    /// so the `set` + `get` + `get` shape borrowed from the Simplified-2Q
    /// suites promotes nothing at all. `keys` is consumed in order, so it is
    /// also `am`'s LRU order afterwards: `keys[0]` ends up deepest and is the
    /// first demotion candidate.
    fn promote_out_of_a1_out(cache: &PaperCache<u32, TieredBuffer>, keys: &[u32]) {
        let before = cache.hybrid_stats().promotions as usize;

        for key in keys {
            cache.get(key).expect("a get on an a1_out key should HIT, not miss");
        }

        assert!(
            wait_until(MIGRATION_TIMEOUT, || {
                cache.hybrid_stats().promotions as usize >= before + keys.len()
            }),
            "every a1_out hit should have promoted its key into am at Tier::Fast",
        );
    }

    /// Forces the one-time PMEM allocator pool init/prewarm to complete
    /// before a test's own timing-sensitive assertions begin.
    ///
    /// Admission is pure DRAM here, so nothing touches the PMEM pool until a
    /// real demotion occurs — and unlike the Simplified-2Q designs, the
    /// cheapest demotion to drive is `a1_in` overflow, which needs no
    /// re-access at all. Backed by the same process-wide `Once` inside the
    /// allocator, so only the first call anywhere in this binary waits.
    fn ensure_pmem_allocator_warm() {
        // Mechanics tests at toy scales: metadata reservation off (see
        // `get_hybrid_dram_shared_overhead`).
        unsafe { std::env::set_var("PAPER_DISABLE_SHARED_OVERHEAD", "1") };

        let cache = PaperCache::<u32, TieredBuffer>::new(
            1_048_576,
            CacheTierSize::Bytes(400),
            PaperPolicy::TwoQFullFastAdmissionHybrid(0.0005, 0.5),
        ).expect("warm-up cache should construct");

        for key in 0..16u32 {
            cache.set(key, &[0u8; 64], None).expect("warm-up set should succeed");
        }

        assert!(
            wait_until(std::time::Duration::from_secs(90), || {
                (0..16u32).any(|key| cache.tier_of(&key) == Some(Tier::Slow))
            }),
            "warm-up should have produced at least one real PMEM demotion",
        );
    }

    // ── admission ─────────────────────────────────────────────────────────

    /// Deliberately asserts with **no** `wait_until`: the object must be
    /// placed in the fast tier by `set()` itself, not started slow and
    /// corrected by the background worker moments later.
    #[test]
    fn admission_lands_in_a1_in_in_the_fast_tier_immediately() {
        ensure_pmem_allocator_warm();

        let cache = PaperCache::<u32, TieredBuffer>::new(
            1_048_576,
            CacheTierSize::Bytes(524_288),
            PaperPolicy::TwoQFullFastAdmissionHybrid(0.1, 0.2),
        ).expect("cache should construct");

        cache.set(1u32, b"hello", None).expect("set should succeed");

        assert_eq!(cache.tier_of(&1u32), Some(Tier::Fast));
        assert_eq!(cache.get(&1u32).unwrap(), b"hello");
    }

    /// The fidelity point, observed from the outside: re-accessing a
    /// probation object must not move it anywhere. It was already fast, and
    /// it must still be sitting in `a1_in` — which is observable by the fact
    /// that it is still the *first* thing sacrificed once `a1_in` overflows,
    /// rather than having been promoted into `am`.
    #[test]
    fn re_accessing_an_a1_in_key_does_not_promote_it() {
        ensure_pmem_allocator_warm();

        let cache = make_cache();

        cache.set(1u32, &[1u8; 64], None).expect("set should succeed");
        cache.get(&1u32).expect("get should hit");
        cache.get(&1u32).expect("second get should hit");

        std::thread::sleep(std::time::Duration::from_millis(200));

        // Still fast (it never left `a1_in`), and no PMEM→DRAM move happened.
        assert_eq!(cache.tier_of(&1u32), Some(Tier::Fast));
        assert_eq!(cache.hybrid_stats().promotions, 0);

        // Age it out of `a1_in` without ever touching it again. Because the
        // repeated hits bought it nothing, it demotes like any other unproven
        // object.
        for key in 2..=40u32 {
            cache.set(key, &[key as u8; 64], None).expect("set should succeed");
        }

        assert!(
            wait_until(MIGRATION_TIMEOUT, || cache.tier_of(&1u32) == Some(Tier::Slow)),
            "an a1_in hit must not buy promotion: key 1 should have aged out to a1_out",
        );
    }

    // ── a1_in overflow demotes rather than evicting ───────────────────────

    /// The structural difference from every Simplified-2Q design here: an
    /// object that ages out of the probation queue is still in the cache.
    #[test]
    fn ageing_out_of_a1_in_demotes_into_a1_out_rather_than_evicting() {
        ensure_pmem_allocator_warm();

        // Roomy `k_out` so nothing is under capacity pressure, and a total
        // cache far larger than the working set.
        let cache = PaperCache::<u32, TieredBuffer>::new(
            1_048_576,
            CacheTierSize::Bytes(131_072),
            PaperPolicy::TwoQFullFastAdmissionHybrid(0.001, 0.5),
        ).expect("cache should construct");

        for key in 1..=30u32 {
            cache.set(key, &[key as u8; 64], None).expect("set should succeed");
        }

        assert!(
            wait_until(MIGRATION_TIMEOUT, || {
                (1..=30u32).any(|key| cache.tier_of(&key) == Some(Tier::Slow))
            }),
            "a1_in overflow should have demoted something into a1_out",
        );

        let demoted: Vec<u32> = (1..=30u32)
            .filter(|key| cache.tier_of(key) == Some(Tier::Slow))
            .collect();

        assert!(!demoted.is_empty());

        // Demoted, not evicted: every one is still present and still reads
        // back correctly out of PMEM.
        for key in &demoted {
            assert!(cache.has(key), "key {key} aged out of a1_in but should still be cached");
            assert_eq!(cache.get(key).unwrap(), vec![*key as u8; 64]);
        }

        // Nothing left the cache at all.
        assert_eq!(cache.hybrid_stats().evictions, 0);
    }

    // ── the 2Q promotion ──────────────────────────────────────────────────

    /// A hit on an `a1_out` object is 2Q's promotion signal, and here it is
    /// also a genuine PMEM→DRAM move.
    #[test]
    fn re_accessing_an_a1_out_key_promotes_it_into_am_at_fast() {
        ensure_pmem_allocator_warm();

        let cache = PaperCache::<u32, TieredBuffer>::new(
            1_048_576,
            CacheTierSize::Bytes(131_072),
            PaperPolicy::TwoQFullFastAdmissionHybrid(0.001, 0.5),
        ).expect("cache should construct");

        for key in 1..=30u32 {
            cache.set(key, &[key as u8; 64], None).expect("set should succeed");
        }

        let mut slow_key = None;

        assert!(
            wait_until(MIGRATION_TIMEOUT, || {
                slow_key = (1..=30u32).find(|key| cache.tier_of(key) == Some(Tier::Slow));
                slow_key.is_some()
            }),
            "some key should have aged into a1_out",
        );

        let slow_key = slow_key.expect("checked above");

        cache.get(&slow_key).expect("get on an a1_out key should HIT, not miss");

        assert!(
            wait_until(MIGRATION_TIMEOUT, || cache.tier_of(&slow_key) == Some(Tier::Fast)),
            "an a1_out hit should promote key {slow_key} into am at fast",
        );

        assert!(cache.hybrid_stats().promotions > 0);
        assert_eq!(cache.get(&slow_key).unwrap(), vec![slow_key as u8; 64]);
    }

    // ── capacity eviction is driven by k_out ──────────────────────────────

    #[test]
    fn a1_out_capacity_pressure_evicts_without_leaking_the_object() {
        ensure_pmem_allocator_warm();

        // Both budgets tiny, so objects race through a1_in into a1_out and
        // out the far end.
        let cache = PaperCache::<u32, TieredBuffer>::new(
            1_048_576,
            CacheTierSize::Bytes(524_288),
            PaperPolicy::TwoQFullFastAdmissionHybrid(0.0002, 0.0002),
        ).expect("cache should construct");

        for key in 1..=40u32 {
            cache.set(key, &[key as u8; 64], None).expect("set should succeed");
        }

        assert!(
            wait_until(MIGRATION_TIMEOUT, || cache.hybrid_stats().evictions > 0),
            "a1_out capacity pressure should have produced evictions",
        );

        assert!(
            wait_until(MIGRATION_TIMEOUT, || (1..=40u32).any(|key| !cache.has(&key))),
            "an evicted key should no longer be present",
        );

        let stats = cache.hybrid_stats();
        let present = (1..=40u32).filter(|key| cache.has(key)).count() as u64;

        assert_eq!(
            present,
            40 - stats.evictions,
            "objects still present should equal admissions minus evictions",
        );
    }

    /// `evict_one` drains `a1_out` before `a1_in` and `am`, so the unproven
    /// objects already in PMEM are sacrificed ahead of a proven one in DRAM.
    #[test]
    fn eviction_prefers_a1_out_over_the_main_queue() {
        ensure_pmem_allocator_warm();

        // a1_in reservation = 0.0002 × 1_048_576 = 200 bytes, which holds two
        // 84-byte objects, so the THIRD set is the one that ages key 1 out —
        // the original two-set fixture never overflowed a1_in at all. a1_out's
        // budget = 0.0005 × 1_048_576 = 500 bytes = 5 objects, so the churn
        // below overruns it repeatedly. The 524_288-byte fast tier leaves am
        // an effective 499_800, so nothing in `am` is ever under pressure:
        // a1_out overflow is the only eviction driver in this fixture.
        let cache = PaperCache::<u32, TieredBuffer>::new(
            1_048_576,
            CacheTierSize::Bytes(524_288),
            PaperPolicy::TwoQFullFastAdmissionHybrid(0.0002, 0.0005),
        ).expect("cache should construct");

        // Key 1 has to AGE OUT and then be hit to reach `am`. Hitting it while
        // it is still in `a1_in` is a complete no-op, which would leave `am`
        // empty and make the eviction-priority claim untestable.
        let aged = set_and_age_out(&cache, 3, 1);

        assert_eq!(aged, vec![1u32], "the a1_in tail — the oldest key — is the one that ages out");

        promote_out_of_a1_out(&cache, &aged);

        assert!(
            wait_until(MIGRATION_TIMEOUT, || cache.tier_of(&1u32) == Some(Tier::Fast)),
            "an a1_out hit should have promoted key 1 into am at Tier::Fast",
        );

        // Everything after this churns through a1_in into a1_out, which
        // overruns its 500-byte budget; `evict_one` drains a1_out first.
        for key in 4..=40u32 {
            cache.set(key, &[key as u8; 64], None).expect("set should succeed");
        }

        assert!(
            wait_until(MIGRATION_TIMEOUT, || cache.hybrid_stats().evictions > 0),
            "a1_out capacity pressure should have produced evictions",
        );

        std::thread::sleep(std::time::Duration::from_millis(300));

        let stats = cache.hybrid_stats();

        // 37 keys are pushed through a 5-object a1_out, so ~32 of them are
        // evicted; the bound is loose, but it has to be well past "one".
        assert!(
            stats.evictions >= 10,
            "the churn should have drained a1_out repeatedly, not once: {} evictions",
            stats.evictions,
        );

        // The proven, DRAM-resident key outlives every unproven a1_out key,
        // even though it is the OLDEST key in the cache.
        assert!(
            cache.has(&1u32),
            "the proven main-queue key should outlive unproven a1_out keys",
        );

        assert_eq!(
            cache.tier_of(&1u32),
            Some(Tier::Fast),
            "and it should still be sitting in am's fast segment",
        );

        // Nothing leaked: every departure is accounted for by an eviction, and
        // every eviction came out of a1_out (key 1, in `am`, is still here).
        let present = (1..=40u32).filter(|key| cache.has(key)).count() as u64;

        assert_eq!(
            present,
            40 - stats.evictions,
            "objects still present should equal admissions minus evictions",
        );

        assert_eq!(cache.get(&1u32).unwrap(), vec![1u8; 64]);
    }

    // ── am behaves like lru_hybrid_cache ──────────────────────────────────

    #[test]
    fn main_queue_pressure_demotes_the_lru_tail_with_real_data_movement() {
        ensure_pmem_allocator_warm();

        let cache = make_cache();

        // An `a1_in` hit is a COMPLETE no-op here, so the `set` + `get` +
        // `get` shape that promotes in the Simplified-2Q siblings would leave
        // all 24 keys in `a1_in` and put NOTHING in `am`. The only route in is
        // to age out of `a1_in` first and take the hit in `a1_out`.
        //
        // 24 sets against a 614-byte `a1_in` (K_IN * MAX_SIZE, 64 MIGRATING
        // bytes per 64-byte object) keeps 9 resident and ages the other 15
        // out. Charged at 84 bytes apiece — which is what the old 819-byte
        // reservation was sized for — the same queue held twelve, only 12
        // keys ever reached a1_out, and this wait timed out.
        let aged = set_and_age_out(&cache, 24, 15);

        assert_eq!(aged.len(), 15, "sizing: 24 sets should age exactly 15 keys into a1_out");

        let ageing_demotions = cache.hybrid_stats().demotions;

        // 15 × 64 = 960 bytes of promoted objects against am's effective
        // fast budget of FAST_TIER - K_IN * MAX_SIZE = 410: `am` has to shed
        // its LRU tail into PMEM. No further `set` happens after this point,
        // so every demotion past `ageing_demotions` came out of `am`.
        promote_out_of_a1_out(&cache, &aged);

        assert!(
            wait_until(MIGRATION_TIMEOUT, || {
                cache.hybrid_stats().demotions > ageing_demotions
                    && aged.iter().any(|key| cache.tier_of(key) == Some(Tier::Slow))
            }),
            "promoting 960 bytes into a 410-byte am fast budget must demote from am",
        );

        std::thread::sleep(std::time::Duration::from_millis(300));

        // Snapshot every tier BEFORE reading any value back: a `get` on a slow
        // `am` key is an `am` hit, which re-promotes it and would erase the
        // very ordering being asserted.
        let tiers: Vec<Option<Tier>> = aged.iter().map(|key| cache.tier_of(key)).collect();

        assert!(
            tiers.iter().all(|tier| tier.is_some()),
            "am pressure demotes, it must never drop a key: {tiers:?}",
        );

        let demoted: Vec<u32> = aged
            .iter()
            .zip(&tiers)
            .filter(|(_, tier)| **tier == Some(Tier::Slow))
            .map(|(key, _)| *key)
            .collect();

        assert!(!demoted.is_empty(), "am should have demoted at least one key");

        assert!(
            tiers.iter().any(|tier| *tier == Some(Tier::Fast)),
            "am should still have a fast segment — the budget sheds bytes, it does not empty",
        );

        // THE LRU TAIL, and only the tail: fast keys are a contiguous prefix
        // of `am` from the MRU end, and `aged` was promoted in ascending
        // order, so the demoted keys must be a contiguous PREFIX of `aged`
        // with no fast key sitting deeper than a slow one.
        let last_slow = tiers.iter().rposition(|tier| *tier == Some(Tier::Slow)).expect("checked above");
        let first_fast = tiers.iter().position(|tier| *tier == Some(Tier::Fast)).expect("checked above");

        assert!(
            last_slow < first_fast,
            "demotion must take am's LRU tail, in order: {:?}",
            aged.iter().zip(&tiers).collect::<Vec<_>>(),
        );

        // Real data movement: every demoted object still reads back
        // byte-for-byte out of PMEM. These gets re-promote, so they come last.
        for key in &demoted {
            assert!(cache.has(key), "a demotion must not drop the object");
            assert_eq!(cache.get(key).unwrap(), vec![*key as u8; 64]);
        }

        assert_eq!(
            cache.hybrid_stats().evictions,
            0,
            "am pressure demotes into PMEM; only a1_out overflow evicts",
        );
    }

    #[test]
    fn fast_tier_usage_stays_within_the_configured_budget() {
        ensure_pmem_allocator_warm();

        const FAST_TIER_BYTES: u64 = 2_000;

        let cache = PaperCache::<u32, TieredBuffer>::new(
            1_048_576,
            CacheTierSize::Bytes(FAST_TIER_BYTES),
            PaperPolicy::TwoQFullFastAdmissionHybrid(0.001, 0.5),
        ).expect("cache should construct");

        for key in 1..=40u32 {
            cache.set(key, &[key as u8; 64], None).expect("set should succeed");
        }

        for key in 1..=40u32 {
            let _ = cache.get(&key);
        }

        assert!(
            wait_until(MIGRATION_TIMEOUT, || cache.hybrid_stats().demotions > 0),
            "should have demoted under a 2000-byte fast tier",
        );

        std::thread::sleep(std::time::Duration::from_millis(300));

        let stats = cache.hybrid_stats();

        assert!(
            stats.fast_bytes_used <= FAST_TIER_BYTES,
            "fast_bytes_used {} exceeded the configured budget {}",
            stats.fast_bytes_used,
            FAST_TIER_BYTES,
        );
    }

    // ── TTL ───────────────────────────────────────────────────────────────

    #[test]
    fn ttl_survives_a_demotion() {
        ensure_pmem_allocator_warm();

        let cache = make_cache();

        cache.set(1u32, &[7u8; 64], Some(60)).expect("set with ttl should succeed");

        for key in 2..=40u32 {
            cache.set(key, &[key as u8; 64], None).expect("set should succeed");
        }

        assert!(
            wait_until(MIGRATION_TIMEOUT, || cache.tier_of(&1u32) == Some(Tier::Slow)),
            "the ttl'd key should have been demoted",
        );

        assert!(cache.has(&1u32));
        assert_eq!(cache.get(&1u32).unwrap(), vec![7u8; 64]);
    }

    #[test]
    fn ttl_expiry_still_fires_after_a_tier_move() {
        ensure_pmem_allocator_warm();

        let cache = make_cache();

        cache.set(1u32, &[7u8; 64], Some(1)).expect("set with ttl should succeed");

        for key in 2..=40u32 {
            cache.set(key, &[key as u8; 64], None).expect("set should succeed");
        }

        assert!(
            wait_until(std::time::Duration::from_secs(5), || !cache.has(&1u32)),
            "the ttl'd key should have expired regardless of which tier it ended in",
        );
    }

    // ── runtime reconfiguration ───────────────────────────────────────────

    #[test]
    fn set_fast_tier_size_takes_effect_at_runtime() {
        ensure_pmem_allocator_warm();

        // `make_cache`'s queue sizing (a1_in = K_IN × MAX_SIZE = 614 bytes = 9
        // residents at 64 migrating bytes each; a1_out = 4_096 bytes, roomy)
        // but with a deliberately generous 4_096-byte fast tier: am's
        // effective fast budget is 4_096 - 614 = 3_482, which holds all 15
        // promotions (15 × 64 = 960) with room to spare. `am` therefore
        // demotes NOTHING until the fast tier itself is resized, so every
        // demotion after `before` is attributable to `set_fast_tier_size`
        // alone.
        //
        // The original fixture's a1_in was 0.001 × 1_048_576 = 1_000 bytes,
        // which holds 11 objects — its 10 sets never aged a single key out, so
        // `am` was empty and shrinking the fast tier had nothing to demote.
        let cache = PaperCache::<u32, TieredBuffer>::new(
            MAX_SIZE,
            CacheTierSize::Bytes(4_096),
            PaperPolicy::TwoQFullFastAdmissionHybrid(K_IN, K_OUT),
        ).expect("cache should construct");

        // An a1_in hit is a no-op, so `get`ting a key twice right after
        // `set`ting it promotes nothing: `am` is reachable only by ageing into
        // a1_out and taking the hit there.
        let aged = set_and_age_out(&cache, 24, 15);
        promote_out_of_a1_out(&cache, &aged);

        assert!(
            wait_until(MIGRATION_TIMEOUT, || {
                aged.iter().all(|key| cache.tier_of(key) == Some(Tier::Fast))
            }),
            "a 3_482-byte am fast budget should hold all 15 promoted objects",
        );

        let before = cache.hybrid_stats().demotions;

        // 1_200 - 614 = 586 bytes left for `am`, against 960 bytes resident.
        cache.set_fast_tier_size(CacheTierSize::Bytes(1_200))
            .expect("resize should succeed");

        assert!(
            wait_until(MIGRATION_TIMEOUT, || {
                cache.hybrid_stats().demotions > before
                    && aged.iter().any(|key| cache.tier_of(key) == Some(Tier::Slow))
            }),
            "shrinking the fast tier should have demoted am's LRU tail into PMEM",
        );

        std::thread::sleep(std::time::Duration::from_millis(300));

        let stats = cache.hybrid_stats();

        // The new budget is actually respected — a1_in's 576 bytes plus what
        // is left of am's fast segment — rather than merely producing one
        // token migration.
        assert!(
            stats.fast_bytes_used <= 1_200,
            "fast_bytes_used {} exceeded the newly configured budget 1_200",
            stats.fast_bytes_used,
        );

        assert_eq!(
            stats.evictions,
            0,
            "a fast-tier shrink demotes into PMEM; it must not evict",
        );

        // Snapshot before reading, since a get on a slow `am` key re-promotes.
        let demoted: Vec<u32> = aged
            .iter()
            .copied()
            .filter(|key| cache.tier_of(key) == Some(Tier::Slow))
            .collect();

        assert!(!demoted.is_empty(), "the wait above already found one; pin it before reading back");

        for key in &demoted {
            assert!(cache.has(key), "a demotion must not drop the object");
            assert_eq!(cache.get(key).unwrap(), vec![*key as u8; 64]);
        }
    }

    /// `resize` rescales BOTH budgets — `a1_in`'s DRAM reservation and
    /// `a1_out`'s PMEM cap — and re-settles both immediately, which
    /// `TwoQHybridStack::resize` need not do.
    #[test]
    fn resize_rescales_both_budgets_and_re_settles() {
        ensure_pmem_allocator_warm();

        // The old fixture (max_size 2_000, k_in 0.5) gave a1_in a 1_000-byte
        // reservation — 11 of these 84-byte objects — while the whole 2_000-
        // byte cache could not even hold 12 of them, so no key could ever age
        // into a1_out and `am` stayed empty through both resizes. `make_cache`
        // is sized for exactly this: a1_in 614 bytes (9 objects at 64
        // migrating bytes each), am's effective fast budget
        // FAST_TIER - 614 = 410 bytes.
        let cache = make_cache();

        // An a1_in hit is a complete no-op, so `am` has to be populated the
        // long way round — age out into a1_out, then hit it there — before a
        // resize can be seen to squeeze it.
        let aged = set_and_age_out(&cache, 24, 15);
        promote_out_of_a1_out(&cache, &aged);

        // Keys that never aged out were never promoted either, so they are
        // exactly a1_in's 9 residents: 9 × 64 = 576 migrating bytes against
        // the 614-byte reservation.
        let a1_in_residents: Vec<u32> = (1..=24u32).filter(|key| !aged.contains(key)).collect();

        assert!(
            wait_until(MIGRATION_TIMEOUT, || {
                a1_in_residents.iter().all(|key| cache.tier_of(key) == Some(Tier::Fast))
                    && aged.iter().any(|key| cache.tier_of(key) == Some(Tier::Fast))
            }),
            "before the resize: a1_in is fast by construction and am has a fast segment",
        );

        std::thread::sleep(std::time::Duration::from_millis(200));

        // ── half one: `a1_in_capacity` is carved out of the SAME fast tier
        //    `am` uses, so growing max_size shrinks am's share and `resize`
        //    must re-settle the fast tier there and then.
        let fast_in_am: Vec<u32> = aged
            .iter()
            .copied()
            .filter(|key| cache.tier_of(key) == Some(Tier::Fast))
            .collect();

        assert!(!fast_in_am.is_empty(), "am needs a fast segment to squeeze");

        let before_growth = cache.hybrid_stats().demotions;

        // K_IN × 30_000 = 900 of the 1_024-byte fast tier, leaving `am` 124
        // where it had 410.
        cache.resize(30_000).expect("resize should succeed");

        assert!(
            wait_until(MIGRATION_TIMEOUT, || {
                cache.hybrid_stats().demotions > before_growth
                    && fast_in_am.iter().any(|key| cache.tier_of(key) == Some(Tier::Slow))
            }),
            "growing max_size grows a1_in's reservation and must demote from am immediately",
        );

        // ── half two: the `a1_in` invariant is re-established by `resize`
        //    itself, not lazily at the next insert — nothing is set, got or
        //    deleted between the call and the assertion.
        let before_shrink = cache.hybrid_stats().demotions;

        // K_IN × 10_240 = 307 bytes of a1_in against 576 bytes resident, so
        // a1_in's tail must drain into a1_out unprompted.
        cache.resize(10_240).expect("resize should succeed");

        assert!(
            wait_until(MIGRATION_TIMEOUT, || {
                a1_in_residents.iter().any(|key| cache.tier_of(key) == Some(Tier::Slow))
            }),
            "shrinking max_size must drain a1_in into a1_out with no intervening insert",
        );

        assert!(
            cache.hybrid_stats().demotions > before_shrink,
            "the a1_in drain is a demotion and must be counted as one",
        );

        std::thread::sleep(std::time::Duration::from_millis(300));

        // Both halves demote; neither evicts. a1_out's rescaled budget
        // (K_OUT × 10_240 = 2_048 bytes) still covers what was drained into
        // it, so `needs_capacity_eviction` stays quiet.
        for key in 1..=24u32 {
            assert!(cache.has(&key), "key {key} was re-settled out of DRAM, not evicted");
        }

        assert_eq!(
            cache.hybrid_stats().evictions,
            0,
            "resize re-settles by demoting; only an a1_out overrun evicts",
        );
    }

    // ── stats ─────────────────────────────────────────────────────────────

    #[test]
    fn hybrid_stats_reports_tier_movement() {
        ensure_pmem_allocator_warm();

        let cache = make_cache();

        // `a1_in` holds 9 of these (see the sizing block), so the 10th set is
        // the first to demote and 12 sets leave three keys in a1_out. Against
        // the old 819-byte reservation a1_in held twelve of them, so these
        // same 12 sets demoted NOTHING and this wait timed out.
        for key in 1..=12u32 {
            cache.set(key, &[key as u8; 64], None).expect("set should succeed");
        }

        assert!(
            wait_until(MIGRATION_TIMEOUT, || cache.hybrid_stats().demotions > 0),
            "should have produced a demotion to compare",
        );

        let stats = cache.hybrid_stats();

        assert!(stats.demotions > 0);
        assert_eq!(stats.fast_objects + stats.slow_objects, stats.total_objects());
    }

    // ── del / wipe ────────────────────────────────────────────────────────

    #[test]
    fn del_and_wipe_work_across_both_tiers() {
        ensure_pmem_allocator_warm();

        let cache = make_cache();

        for key in 1..=20u32 {
            cache.set(key, &[key as u8; 64], None).expect("set should succeed");
        }

        assert!(
            wait_until(MIGRATION_TIMEOUT, || {
                (1..=20u32).any(|key| cache.tier_of(&key) == Some(Tier::Slow))
                    && (1..=20u32).any(|key| cache.tier_of(&key) == Some(Tier::Fast))
            }),
            "should have a key in each tier to exercise",
        );

        let slow_key = (1..=20u32)
            .find(|key| cache.tier_of(key) == Some(Tier::Slow))
            .expect("checked above");

        let fast_key = (1..=20u32)
            .find(|key| cache.tier_of(key) == Some(Tier::Fast))
            .expect("checked above");

        cache.del(&slow_key).expect("del on a slow key should succeed");
        cache.del(&fast_key).expect("del on a fast key should succeed");

        assert!(!cache.has(&slow_key));
        assert!(!cache.has(&fast_key));

        cache.wipe().expect("wipe should succeed");

        for key in 1..=20u32 {
            assert!(!cache.has(&key), "key {key} should be gone after wipe");
        }

        cache.set(99u32, b"after wipe", None).expect("set should succeed");
        assert_eq!(cache.tier_of(&99u32), Some(Tier::Fast));
    }

    // ── edge cases ────────────────────────────────────────────────────────

    /// Two ratios, so BOTH have to be range-checked — the one thing
    /// `new_hybrid`'s `params_ok` could plausibly get half-right.
    #[test]
    fn invalid_construction_parameters_are_rejected() {
        assert!(matches!(
            PaperCache::<u32, TieredBuffer>::new(1_048_576, CacheTierSize::Bytes(0), PaperPolicy::TwoQFullFastAdmissionHybrid(0.1, 0.2)),
            Err(CacheError::InvalidFastTierSize),
        ));

        assert!(matches!(
            PaperCache::<u32, TieredBuffer>::new(1_024, CacheTierSize::Bytes(2_048), PaperPolicy::TwoQFullFastAdmissionHybrid(0.1, 0.2)),
            Err(CacheError::InvalidFastTierSize),
        ));

        assert!(matches!(
            PaperCache::<u32, TieredBuffer>::new(0, CacheTierSize::Bytes(1), PaperPolicy::TwoQFullFastAdmissionHybrid(0.1, 0.2)),
            Err(CacheError::ZeroCacheSize),
        ));

        // k_in out of range.
        assert!(matches!(
            PaperCache::<u32, TieredBuffer>::new(1_048_576, CacheTierSize::Bytes(1_024), PaperPolicy::TwoQFullFastAdmissionHybrid(1.5, 0.2)),
            Err(CacheError::InvalidPolicy),
        ));

        assert!(matches!(
            PaperCache::<u32, TieredBuffer>::new(1_048_576, CacheTierSize::Bytes(1_024), PaperPolicy::TwoQFullFastAdmissionHybrid(-0.1, 0.2)),
            Err(CacheError::InvalidPolicy),
        ));

        // k_out out of range -- the half a single-ratio check would miss.
        assert!(matches!(
            PaperCache::<u32, TieredBuffer>::new(1_048_576, CacheTierSize::Bytes(1_024), PaperPolicy::TwoQFullFastAdmissionHybrid(0.1, 1.5)),
            Err(CacheError::InvalidPolicy),
        ));

        assert!(matches!(
            PaperCache::<u32, TieredBuffer>::new(1_048_576, CacheTierSize::Bytes(1_024), PaperPolicy::TwoQFullFastAdmissionHybrid(0.1, -0.1)),
            Err(CacheError::InvalidPolicy),
        ));
    }

    /// `k_in * max_size >= fast_tier_size` leaves `am` no fast segment:
    /// legitimate, if degenerate, and it must not wedge the cache.
    #[test]
    fn an_a1_in_reservation_covering_the_whole_fast_tier_still_works() {
        ensure_pmem_allocator_warm();

        let cache = PaperCache::<u32, TieredBuffer>::new(
            10_240,
            CacheTierSize::Bytes(1_024),
            PaperPolicy::TwoQFullFastAdmissionHybrid(1.0, 0.5),
        ).expect("cache should construct");

        cache.set(1u32, &[1u8; 64], None).expect("set should succeed");

        // Admission is still fast: it goes into a1_in.
        assert_eq!(cache.tier_of(&1u32), Some(Tier::Fast));
        assert_eq!(cache.get(&1u32).unwrap(), vec![1u8; 64]);
    }

    #[test]
    fn re_setting_an_existing_key_keeps_it_readable_and_correctly_tiered() {
        ensure_pmem_allocator_warm();

        let cache = make_cache();

        cache.set(1u32, b"first", None).expect("set should succeed");
        assert_eq!(cache.tier_of(&1u32), Some(Tier::Fast));

        cache.set(1u32, b"second value", None).expect("re-set should succeed");

        std::thread::sleep(std::time::Duration::from_millis(200));

        assert_eq!(cache.tier_of(&1u32), Some(Tier::Fast));
        assert_eq!(cache.get(&1u32).unwrap(), b"second value");

        assert!(
            wait_until(MIGRATION_TIMEOUT, || {
                let stats = cache.hybrid_stats();
                stats.fast_objects + stats.slow_objects == 1
            }),
            "the re-set key should be counted exactly once",
        );
    }
}
