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
    //   a1_in reservation   = K_IN  * MAX_SIZE = 800 bytes (~8 test objects)
    //   effective am fast   = FAST_TIER - 800  = 800 bytes (~8 test objects)
    //   a1_out budget       = K_OUT * MAX_SIZE = 4_000 bytes (roomy, so the
    //                         demote-don't-evict path is what gets exercised)
    const MAX_SIZE: u64 = 20_000;
    const FAST_TIER: u64 = 1_600;
    const K_IN: f64 = 0.04;
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
            1_000_000,
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
            1_000_000,
            CacheTierSize::Bytes(500_000),
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
            1_000_000,
            CacheTierSize::Bytes(100_000),
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
            1_000_000,
            CacheTierSize::Bytes(100_000),
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
            1_000_000,
            CacheTierSize::Bytes(500_000),
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

        let cache = PaperCache::<u32, TieredBuffer>::new(
            1_000_000,
            CacheTierSize::Bytes(500_000),
            PaperPolicy::TwoQFullFastAdmissionHybrid(0.0002, 0.0002),
        ).expect("cache should construct");

        // Key 1 ages into a1_out, then proves itself and lands in `am`.
        cache.set(1u32, &[1u8; 64], None).expect("set should succeed");
        cache.set(2u32, &[2u8; 64], None).expect("set should succeed");

        assert!(
            wait_until(MIGRATION_TIMEOUT, || cache.tier_of(&1u32) == Some(Tier::Slow)),
            "key 1 should have aged into a1_out",
        );

        cache.get(&1u32).expect("get should hit");

        assert!(
            wait_until(MIGRATION_TIMEOUT, || cache.tier_of(&1u32) == Some(Tier::Fast)),
            "key 1 should have been promoted into am",
        );

        // Everything after this churns through a1_in into a1_out, which
        // overflows its tiny budget and is drained first.
        for key in 3..=40u32 {
            cache.set(key, &[key as u8; 64], None).expect("set should succeed");
        }

        assert!(
            wait_until(MIGRATION_TIMEOUT, || cache.hybrid_stats().evictions > 0),
            "should have evicted from a1_out",
        );

        assert!(
            cache.has(&1u32),
            "the proven main-queue key should outlive unproven a1_out keys",
        );
    }

    // ── am behaves like lru_hybrid_cache ──────────────────────────────────

    #[test]
    fn main_queue_pressure_demotes_the_lru_tail_with_real_data_movement() {
        ensure_pmem_allocator_warm();

        let cache = make_cache();

        // Each key is admitted, aged into a1_out by the next few admissions,
        // then re-accessed to promote it into `am`, where it competes for the
        // (reduced) fast budget.
        for key in 1..=12u32 {
            cache.set(key, &[key as u8; 64], None).expect("set should succeed");
        }

        for key in 1..=12u32 {
            let _ = cache.get(&key);
            let _ = cache.get(&key);
        }

        assert!(
            wait_until(MIGRATION_TIMEOUT, || cache.hybrid_stats().demotions > 0),
            "main-queue pressure should have demoted something",
        );

        let slow: Vec<u32> = (1..=12u32)
            .filter(|key| cache.tier_of(key) == Some(Tier::Slow))
            .collect();

        assert!(!slow.is_empty(), "at least one key should be slow");

        for key in &slow {
            assert_eq!(cache.get(key).unwrap(), vec![*key as u8; 64]);
        }
    }

    #[test]
    fn fast_tier_usage_stays_within_the_configured_budget() {
        ensure_pmem_allocator_warm();

        const FAST_TIER_BYTES: u64 = 2_000;

        let cache = PaperCache::<u32, TieredBuffer>::new(
            1_000_000,
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

        let cache = PaperCache::<u32, TieredBuffer>::new(
            1_000_000,
            CacheTierSize::Bytes(500_000),
            PaperPolicy::TwoQFullFastAdmissionHybrid(0.001, 0.5),
        ).expect("cache should construct");

        for key in 1..=10u32 {
            cache.set(key, &[key as u8; 64], None).expect("set should succeed");
        }

        for key in 1..=10u32 {
            let _ = cache.get(&key);
            let _ = cache.get(&key);
        }

        let before = cache.hybrid_stats().demotions;

        cache.set_fast_tier_size(CacheTierSize::Bytes(400))
            .expect("resize should succeed");

        assert!(
            wait_until(MIGRATION_TIMEOUT, || cache.hybrid_stats().demotions > before),
            "shrinking the fast tier should have demoted something",
        );
    }

    /// `resize` rescales BOTH budgets — `a1_in`'s DRAM reservation and
    /// `a1_out`'s PMEM cap — and re-settles both immediately, which
    /// `TwoQHybridStack::resize` need not do.
    #[test]
    fn resize_rescales_both_budgets_and_re_settles() {
        ensure_pmem_allocator_warm();

        let cache = PaperCache::<u32, TieredBuffer>::new(
            2_000,
            CacheTierSize::Bytes(1_500),
            PaperPolicy::TwoQFullFastAdmissionHybrid(0.5, 0.25),
        ).expect("cache should construct");

        for key in 1..=6u32 {
            cache.set(key, &[key as u8; 64], None).expect("set should succeed");
            let _ = cache.get(&key);
            let _ = cache.get(&key);
        }

        std::thread::sleep(std::time::Duration::from_millis(200));

        let before = cache.hybrid_stats().demotions;

        // Growing max_size grows a1_in's reservation (0.5 * max_size),
        // squeezing am's share of the same 1_500-byte fast tier.
        cache.resize(2_800).expect("resize should succeed");

        assert!(
            wait_until(MIGRATION_TIMEOUT, || cache.hybrid_stats().demotions > before),
            "growing max_size should have grown the a1_in reservation and demoted",
        );
    }

    // ── stats ─────────────────────────────────────────────────────────────

    #[test]
    fn hybrid_stats_reports_tier_movement() {
        ensure_pmem_allocator_warm();

        let cache = make_cache();

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
            PaperCache::<u32, TieredBuffer>::new(1_000_000, CacheTierSize::Bytes(0), PaperPolicy::TwoQFullFastAdmissionHybrid(0.1, 0.2)),
            Err(CacheError::InvalidFastTierSize),
        ));

        assert!(matches!(
            PaperCache::<u32, TieredBuffer>::new(1_000, CacheTierSize::Bytes(2_000), PaperPolicy::TwoQFullFastAdmissionHybrid(0.1, 0.2)),
            Err(CacheError::InvalidFastTierSize),
        ));

        assert!(matches!(
            PaperCache::<u32, TieredBuffer>::new(0, CacheTierSize::Bytes(1), PaperPolicy::TwoQFullFastAdmissionHybrid(0.1, 0.2)),
            Err(CacheError::ZeroCacheSize),
        ));

        // k_in out of range.
        assert!(matches!(
            PaperCache::<u32, TieredBuffer>::new(1_000_000, CacheTierSize::Bytes(1_000), PaperPolicy::TwoQFullFastAdmissionHybrid(1.5, 0.2)),
            Err(CacheError::InvalidPolicy),
        ));

        assert!(matches!(
            PaperCache::<u32, TieredBuffer>::new(1_000_000, CacheTierSize::Bytes(1_000), PaperPolicy::TwoQFullFastAdmissionHybrid(-0.1, 0.2)),
            Err(CacheError::InvalidPolicy),
        ));

        // k_out out of range -- the half a single-ratio check would miss.
        assert!(matches!(
            PaperCache::<u32, TieredBuffer>::new(1_000_000, CacheTierSize::Bytes(1_000), PaperPolicy::TwoQFullFastAdmissionHybrid(0.1, 1.5)),
            Err(CacheError::InvalidPolicy),
        ));

        assert!(matches!(
            PaperCache::<u32, TieredBuffer>::new(1_000_000, CacheTierSize::Bytes(1_000), PaperPolicy::TwoQFullFastAdmissionHybrid(0.1, -0.1)),
            Err(CacheError::InvalidPolicy),
        ));
    }

    /// `k_in * max_size >= fast_tier_size` leaves `am` no fast segment:
    /// legitimate, if degenerate, and it must not wedge the cache.
    #[test]
    fn an_a1_in_reservation_covering_the_whole_fast_tier_still_works() {
        ensure_pmem_allocator_warm();

        let cache = PaperCache::<u32, TieredBuffer>::new(
            10_000,
            CacheTierSize::Bytes(1_000),
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
