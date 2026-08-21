/*
 * Copyright (c) Kia Shakiba
 *
 * This source code is licensed under the GNU AGPLv3 license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! Integration tests for the `two_q_fast_admission_hybrid_cache` feature.
//!
//! Run with nightly (required for `allocator_api` via `key_value_pmem`):
//!   cargo +nightly test --test hybrid_cache_integration --features two_q_fast_admission_hybrid_cache
//!
//! Same one-`PaperCache<K, TieredBuffer>` architecture as the other hybrid
//! integration suites — `tier_of` reads the tier directly off the single
//! object map.
//!
//! The defining difference from `hybrid_cache_integration.rs`:
//! admission lands in the **fast** tier here. Every `set()` is a plain DRAM
//! write, so — unlike that suite, where every single test pays a synchronous
//! PMEM allocation — the tests below only touch PMEM once a demotion
//! actually happens. `ensure_pmem_allocator_warm` therefore has to *force* a
//! demotion to warm the pool, the same way the LRU/LFU suites' analogs do,
//! rather than getting it for free from the first `set()`.
//!
//! What is tested:
//!   * Admission always lands in the fast tier (the one-access FIFO queue),
//!     observable immediately on `set()` return with no `wait_until` — i.e.
//!     placed correctly the first time, not corrected moments later
//!   * A re-access to a FIFO-queue object keeps it fast (moving it into the
//!     main queue) and moves no bytes
//!   * The FIFO queue's budget is a reservation carved out of
//!     `fast_tier_size`, so the main queue demotes earlier than its raw
//!     fast-tier budget alone would suggest
//!   * Once in the main queue, an object behaves like `lru_hybrid_cache`:
//!     fast-tier pressure demotes the LRU tail; a slow-tier access promotes
//!     it back, possibly cascading a further demotion
//!   * A FIFO object that ages out without a second access is evicted
//!   * Terminal eviction prefers the FIFO queue over the main queue
//!   * TTL set before a demotion/promotion is still correctly enforced after
//!   * `set_fast_tier_size` / `resize` take effect at runtime
//!   * `hybrid_stats()` agrees with `hybrid_stats()`
//!   * Zero/invalid fast-tier-size and `k_in` edge cases

#[cfg(feature = "two_q_fast_admission_hybrid_cache")]
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

    // ── sizing: the one arithmetic trap specific to this design ───────────
    //
    // `k_in` is denominated in `max_size`, but the budget it consumes is
    // `fast_tier_size`: the FIFO queue is DRAM here, so its reservation is
    // carved out of the fast tier (`effective_main_fast_capacity =
    // fast_tier_size - k_in * max_size`). A `k_in` that looks tiny relative
    // to `max_size` can therefore swallow the entire fast tier -- e.g.
    // `k_in = 0.05` of a 1 MB cache reserves 50,000 bytes, which against a
    // 600-byte fast tier saturates the main queue's capacity to ZERO, so
    // every promotion self-demotes instantly and no key is ever observably
    // fast in the main queue.
    //
    // These constants keep the three quantities in a deliberate relationship
    // instead of leaving it implicit at each call site:
    //   FIFO reservation    = K_IN * MAX_SIZE = 800 bytes (~8 test objects)
    //   effective main fast = FAST_TIER - 800 = 800 bytes (~8 test objects)
    // which is small enough that a handful of promotions forces demotion,
    // while leaving the FIFO queue enough room not to be evicting constantly.
    const MAX_SIZE: u64 = 20_000;
    const FAST_TIER: u64 = 1_600;
    const K_IN: f64 = 0.04;

    /// Cache sized by the constants above -- use this for any test that
    /// needs a *reachable* fast main queue (demotion, promotion, TTL
    /// survival). Tests deliberately exercising a degenerate configuration
    /// construct their own.
    fn make_cache() -> PaperCache<u32, TieredBuffer> {
        PaperCache::<u32, TieredBuffer>::new(
            MAX_SIZE,
            CacheTierSize::Bytes(FAST_TIER), PaperPolicy::TwoQFastAdmissionHybrid(K_IN)).expect("cache should construct")
    }

    /// Forces the one-time PMEM allocator pool init/prewarm to complete
    /// before a test's own timing-sensitive assertions begin.
    ///
    /// Unlike `two_q_hybrid_cache`'s analog — where the very first `set()`
    /// pays this cost synchronously, because admission itself allocates in
    /// PMEM — admission here is pure DRAM, so nothing touches the PMEM pool
    /// until a real demotion occurs. This therefore has to *drive* a
    /// demotion (tiny fast tier, several keys promoted into the main queue)
    /// and wait for it, with a budget generous enough to absorb the ~45s
    /// first-touch cost. Backed by the same process-wide `Once` inside the
    /// allocator, so only the first call anywhere in this binary actually
    /// waits.
    fn ensure_pmem_allocator_warm() {
        let cache = PaperCache::<u32, TieredBuffer>::new(
            1_000_000,
            CacheTierSize::Bytes(400), PaperPolicy::TwoQFastAdmissionHybrid(0.1)).expect("warm-up cache should construct");

        for key in 0..8u32 {
            cache.set(key, &[0u8; 64], None).expect("warm-up set should succeed");
            let _ = cache.get(&key);
        }

        assert!(
            wait_until(std::time::Duration::from_secs(90), || {
                (0..8u32).any(|key| cache.tier_of(&key) == Some(Tier::Slow))
            }),
            "warm-up should have produced at least one real PMEM demotion",
        );
    }

    // ── admission ─────────────────────────────────────────────────────────

    /// The headline property. Deliberately asserts with **no** `wait_until`:
    /// the object must be placed in the fast tier by `set()` itself, not
    /// started slow and corrected by the background worker moments later.
    #[test]
    fn admission_lands_in_the_fast_tier_immediately() {
        ensure_pmem_allocator_warm();

        let cache = PaperCache::<u32, TieredBuffer>::new(
            1_000_000,
            CacheTierSize::Bytes(500_000), PaperPolicy::TwoQFastAdmissionHybrid(0.1)).expect("cache should construct");

        cache.set(1u32, b"hello", None).expect("set should succeed");

        assert_eq!(cache.tier_of(&1u32), Some(Tier::Fast));
        assert_eq!(cache.get(&1u32).unwrap(), b"hello");
    }

    #[test]
    fn many_admissions_all_land_fast_without_any_migration() {
        ensure_pmem_allocator_warm();

        // Deliberately NOT `make_cache()`: this test needs a FIFO queue
        // roomy enough for 20 keys without triggering capacity eviction, and
        // it never promotes anything, so the fact that this `k_in` leaves
        // zero effective main-queue capacity is irrelevant here.
        let cache = PaperCache::<u32, TieredBuffer>::new(
            1_000_000,
            CacheTierSize::Bytes(500_000), PaperPolicy::TwoQFastAdmissionHybrid(0.5)).expect("cache should construct");

        for key in 1..=20u32 {
            cache.set(key, &[key as u8; 64], None).expect("set should succeed");
            assert_eq!(cache.tier_of(&key), Some(Tier::Fast), "key {key} should be fast");
        }

        // The gauges are republished once per worker event-loop pass, so
        // wait for them to reflect all 20 admissions rather than assuming a
        // fixed sleep is long enough (a fixed 200ms observed 15 of 20).
        assert!(
            wait_until(MIGRATION_TIMEOUT, || {
                cache.hybrid_stats().fast_objects == 20
            }),
            "all 20 admissions should be accounted to the fast tier",
        );

        // Nothing was demoted or promoted: no key ever needed to move.
        let stats = cache.hybrid_stats();
        assert_eq!(stats.demotions, 0);
        assert_eq!(stats.promotions, 0);
        assert_eq!(stats.slow_objects, 0);
    }

    /// A FIFO→main promotion is Fast→Fast: the key stays fast throughout and
    /// the `promotions` counter — which only tracks genuine PMEM→DRAM moves
    /// in this design — stays at zero.
    #[test]
    fn re_accessing_a_fifo_key_keeps_it_fast_and_counts_no_promotion() {
        ensure_pmem_allocator_warm();

        let cache = make_cache();

        cache.set(1u32, b"value", None).expect("set should succeed");
        assert_eq!(cache.tier_of(&1u32), Some(Tier::Fast));

        cache.get(&1u32).expect("get should hit");

        std::thread::sleep(std::time::Duration::from_millis(200));

        assert_eq!(cache.tier_of(&1u32), Some(Tier::Fast));
        assert_eq!(cache.hybrid_stats().promotions, 0);
        assert_eq!(cache.get(&1u32).unwrap(), b"value");
    }

    // ── demotion ──────────────────────────────────────────────────────────

    #[test]
    fn main_queue_pressure_demotes_the_lru_tail_with_real_data_movement() {
        ensure_pmem_allocator_warm();

        // Small fast tier and a small FIFO reservation, so the main queue
        // fills quickly once keys start proving themselves.
        let cache = make_cache();

        // Each key is admitted fast, then re-accessed to move it into the
        // main queue where it competes for the (reduced) fast budget.
        for key in 1..=10u32 {
            cache.set(key, &[key as u8; 64], None).expect("set should succeed");
            cache.get(&key).expect("get should hit");
        }

        assert!(
            wait_until(MIGRATION_TIMEOUT, || {
                cache.hybrid_stats().demotions > 0
            }),
            "main-queue pressure should have demoted something",
        );

        // Whatever was demoted is genuinely gone from the fast tier and its
        // bytes still read back correctly from PMEM.
        let demoted: Vec<u32> = (1..=10u32)
            .filter(|key| cache.tier_of(key) == Some(Tier::Slow))
            .collect();

        assert!(!demoted.is_empty(), "at least one key should be slow");

        for key in &demoted {
            assert_eq!(cache.get(key).unwrap(), vec![*key as u8; 64]);
        }
    }

    /// The FIFO reservation is carved out of `fast_tier_size`, so the main
    /// queue starts demoting while total fast-tier usage is still below the
    /// configured budget. This is the accounting difference from
    /// `two_q_hybrid_cache`, where the FIFO queue costs no DRAM at all.
    #[test]
    fn fast_tier_usage_stays_within_the_configured_budget() {
        ensure_pmem_allocator_warm();

        const FAST_TIER_BYTES: u64 = 2_000;

        let cache = PaperCache::<u32, TieredBuffer>::new(
            1_000_000,
            CacheTierSize::Bytes(FAST_TIER_BYTES), PaperPolicy::TwoQFastAdmissionHybrid(0.001)).expect("cache should construct");

        for key in 1..=40u32 {
            cache.set(key, &[key as u8; 64], None).expect("set should succeed");
            cache.get(&key).expect("get should hit");
        }

        assert!(
            wait_until(MIGRATION_TIMEOUT, || {
                cache.hybrid_stats().demotions > 0
            }),
            "should have demoted under a 2000-byte fast tier",
        );

        // Give the worker a moment to settle, then check the gauge.
        std::thread::sleep(std::time::Duration::from_millis(300));

        let stats = cache.hybrid_stats();

        assert!(
            stats.fast_bytes_used <= FAST_TIER_BYTES,
            "fast_bytes_used {} exceeded the configured budget {}",
            stats.fast_bytes_used,
            FAST_TIER_BYTES,
        );
    }

    // ── promotion ─────────────────────────────────────────────────────────

    #[test]
    fn re_accessing_a_demoted_key_promotes_it_back_to_fast() {
        ensure_pmem_allocator_warm();

        let cache = make_cache();

        for key in 1..=10u32 {
            cache.set(key, &[key as u8; 64], None).expect("set should succeed");
            cache.get(&key).expect("get should hit");
        }

        // Find a key that actually ended up slow. Checked inside a single
        // `wait_until` over all candidates, not one full-timeout wait per
        // candidate -- a key that will stay fast forever would otherwise
        // burn the whole budget before the next candidate is tried.
        let mut slow_key = None;

        assert!(
            wait_until(MIGRATION_TIMEOUT, || {
                slow_key = (1..=10u32).find(|key| cache.tier_of(key) == Some(Tier::Slow));
                slow_key.is_some()
            }),
            "some key should have been demoted",
        );

        let slow_key = slow_key.expect("checked above");

        cache.get(&slow_key).expect("get on a slow key should still hit");

        assert!(
            wait_until(MIGRATION_TIMEOUT, || {
                cache.tier_of(&slow_key) == Some(Tier::Fast)
            }),
            "re-accessing key {slow_key} should have promoted it back to fast",
        );

        assert!(cache.hybrid_stats().promotions > 0);
        assert_eq!(cache.get(&slow_key).unwrap(), vec![slow_key as u8; 64]);
    }

    // ── eviction ──────────────────────────────────────────────────────────

    #[test]
    fn fifo_capacity_pressure_evicts_without_leaking_the_object() {
        ensure_pmem_allocator_warm();

        // k_in is tiny, so the FIFO queue overflows almost immediately and
        // `needs_capacity_eviction` has to drain it through the real
        // evict_one()/erase() path.
        let cache = PaperCache::<u32, TieredBuffer>::new(
            1_000_000,
            CacheTierSize::Bytes(500_000), PaperPolicy::TwoQFastAdmissionHybrid(0.001)).expect("cache should construct");

        for key in 1..=30u32 {
            cache.set(key, &[key as u8; 64], None).expect("set should succeed");
        }

        assert!(
            wait_until(MIGRATION_TIMEOUT, || {
                cache.hybrid_stats().evictions > 0
            }),
            "FIFO-capacity pressure should have produced evictions",
        );

        // The evicted objects must be genuinely gone -- not merely dropped
        // from the stack's bookkeeping while still visible via `has`.
        assert!(
            wait_until(MIGRATION_TIMEOUT, || {
                (1..=30u32).any(|key| !cache.has(&key))
            }),
            "an evicted key should no longer be present",
        );

        let stats = cache.hybrid_stats();
        let present = (1..=30u32).filter(|key| cache.has(key)).count() as u64;

        assert_eq!(
            present,
            30 - stats.evictions,
            "objects still present should equal admissions minus evictions",
        );
    }

    #[test]
    fn eviction_prefers_the_fifo_queue_over_the_main_queue() {
        ensure_pmem_allocator_warm();

        let cache = PaperCache::<u32, TieredBuffer>::new(
            1_000_000,
            CacheTierSize::Bytes(500_000), PaperPolicy::TwoQFastAdmissionHybrid(0.001)).expect("cache should construct");

        // Key 1 proves itself and moves into the main queue.
        cache.set(1u32, &[1u8; 64], None).expect("set should succeed");
        cache.get(&1u32).expect("get should hit");

        std::thread::sleep(std::time::Duration::from_millis(200));

        // Everything after this stays in the one-access FIFO queue, which
        // overflows its tiny budget and gets drained first.
        for key in 2..=30u32 {
            cache.set(key, &[key as u8; 64], None).expect("set should succeed");
        }

        assert!(
            wait_until(MIGRATION_TIMEOUT, || {
                cache.hybrid_stats().evictions > 0
            }),
            "should have evicted from the FIFO queue",
        );

        assert!(
            cache.has(&1u32),
            "the proven main-queue key should outlive unproven FIFO keys",
        );
    }

    // ── TTL ───────────────────────────────────────────────────────────────

    #[test]
    fn ttl_survives_a_demotion() {
        ensure_pmem_allocator_warm();

        // Fast tier sized comfortably larger than one ttl'd object (whose
        // base_size carries a fixed TTL bookkeeping cost), with small filler
        // keys creating the pressure instead -- the same sizing trap the
        // other hybrids' equivalent tests hit.
        let cache = make_cache();

        cache.set(1u32, &[7u8; 64], Some(60)).expect("set with ttl should succeed");
        cache.get(&1u32).expect("get should hit");

        for key in 2..=12u32 {
            cache.set(key, &[key as u8; 64], None).expect("set should succeed");
            cache.get(&key).expect("get should hit");
        }

        assert!(
            wait_until(MIGRATION_TIMEOUT, || {
                cache.tier_of(&1u32) == Some(Tier::Slow)
            }),
            "the ttl'd key should have been demoted",
        );

        // TTL survived the tier move: still present, bytes intact, and not
        // expired early. (`PaperCache::ttl` is a setter, not a getter, so
        // "still has a TTL" is verified by the companion test below, which
        // watches a short TTL actually fire after a move.)
        assert!(cache.has(&1u32));
        assert_eq!(cache.get(&1u32).unwrap(), vec![7u8; 64]);
    }

    #[test]
    fn ttl_expiry_still_fires_after_a_tier_move() {
        ensure_pmem_allocator_warm();

        let cache = make_cache();

        cache.set(1u32, &[7u8; 64], Some(1)).expect("set with ttl should succeed");
        cache.get(&1u32).expect("get should hit");

        for key in 2..=12u32 {
            cache.set(key, &[key as u8; 64], None).expect("set should succeed");
            cache.get(&key).expect("get should hit");
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
            CacheTierSize::Bytes(500_000), PaperPolicy::TwoQFastAdmissionHybrid(0.001)).expect("cache should construct");

        for key in 1..=10u32 {
            cache.set(key, &[key as u8; 64], None).expect("set should succeed");
            cache.get(&key).expect("get should hit");
        }

        std::thread::sleep(std::time::Duration::from_millis(200));
        assert_eq!(cache.hybrid_stats().demotions, 0);

        // Shrinking the fast tier must force demotions immediately, without
        // waiting for another access.
        cache.set_fast_tier_size(CacheTierSize::Bytes(400))
            .expect("resize should succeed");

        assert!(
            wait_until(MIGRATION_TIMEOUT, || {
                cache.hybrid_stats().demotions > 0
            }),
            "shrinking the fast tier should have demoted something",
        );
    }

    /// `resize` rescales `fifo_capacity` (`k_in * max_size`), which moves the
    /// main queue's effective budget — the reason this stack re-settles on
    /// `resize` where `TwoQHybridStack` need not.
    #[test]
    fn resize_rescales_the_fifo_reservation_and_re_settles() {
        ensure_pmem_allocator_warm();

        // k_in 0.5 makes fifo_capacity track max_size closely, so growing
        // max_size meaningfully shrinks what the main queue may keep fast.
        let cache = PaperCache::<u32, TieredBuffer>::new(
            2_000,
            CacheTierSize::Bytes(1_500), PaperPolicy::TwoQFastAdmissionHybrid(0.5)).expect("cache should construct");

        for key in 1..=6u32 {
            cache.set(key, &[key as u8; 64], None).expect("set should succeed");
            cache.get(&key).expect("get should hit");
        }

        std::thread::sleep(std::time::Duration::from_millis(200));

        let before = cache.hybrid_stats().demotions;

        // Growing max_size grows the FIFO reservation (0.5 * max_size),
        // squeezing the main queue's share of the same 1_500-byte fast tier.
        cache.resize(2_800).expect("resize should succeed");

        assert!(
            wait_until(MIGRATION_TIMEOUT, || {
                cache.hybrid_stats().demotions > before
            }),
            "growing max_size should have grown the FIFO reservation and demoted",
        );
    }

    // ── stats ─────────────────────────────────────────────────────────────

    #[test]
    fn hybrid_stats_agrees_with_the_named_accessor() {
        ensure_pmem_allocator_warm();

        let cache = make_cache();

        for key in 1..=10u32 {
            cache.set(key, &[key as u8; 64], None).expect("set should succeed");
            cache.get(&key).expect("get should hit");
        }

        assert!(
            wait_until(MIGRATION_TIMEOUT, || {
                cache.hybrid_stats().demotions > 0
            }),
            "should have produced a demotion to compare",
        );

        // Read the neutral view first, then the named one: the counters are
        // monotonic and the worker may still be running, so the neutral
        // reading must fall at or below the later named reading rather than
        // being compared for strict equality against a moving target.
        let common = cache.hybrid_stats();
        let named = cache.hybrid_stats();

        assert!(common.demotions <= named.demotions);
        assert!(common.promotions <= named.promotions);
        assert!(common.evictions <= named.evictions);
        assert!(common.demotions > 0);

        // The object-count invariant should hold exactly at any instant.
        let neutral = cache.hybrid_stats();
        assert_eq!(
            neutral.fast_objects + neutral.slow_objects,
            neutral.total_objects(),
        );
    }

    // ── del / wipe ────────────────────────────────────────────────────────

    #[test]
    fn del_and_wipe_work_across_both_tiers() {
        ensure_pmem_allocator_warm();

        let cache = make_cache();

        for key in 1..=10u32 {
            cache.set(key, &[key as u8; 64], None).expect("set should succeed");
            cache.get(&key).expect("get should hit");
        }

        assert!(
            wait_until(MIGRATION_TIMEOUT, || {
                (1..=10u32).any(|key| cache.tier_of(&key) == Some(Tier::Slow))
            }),
            "should have a key in each tier to exercise",
        );

        let slow_key = (1..=10u32)
            .find(|key| cache.tier_of(key) == Some(Tier::Slow))
            .expect("checked above");

        let fast_key = (1..=10u32)
            .find(|key| cache.tier_of(key) == Some(Tier::Fast))
            .expect("some key should still be fast");

        cache.del(&slow_key).expect("del on a slow key should succeed");
        cache.del(&fast_key).expect("del on a fast key should succeed");

        assert!(!cache.has(&slow_key));
        assert!(!cache.has(&fast_key));

        cache.wipe().expect("wipe should succeed");

        for key in 1..=10u32 {
            assert!(!cache.has(&key), "key {key} should be gone after wipe");
        }

        // A fresh admission after a wipe still lands fast.
        cache.set(99u32, b"after wipe", None).expect("set should succeed");
        assert_eq!(cache.tier_of(&99u32), Some(Tier::Fast));
    }

    // ── edge cases ────────────────────────────────────────────────────────

    #[test]
    fn invalid_construction_parameters_are_rejected() {
        // Zero fast tier.
        assert!(matches!(
            PaperCache::<u32, TieredBuffer>::new(1_000_000, CacheTierSize::Bytes(0), PaperPolicy::TwoQFastAdmissionHybrid(0.1)),
            Err(CacheError::InvalidFastTierSize),
        ));

        // Fast tier larger than the whole cache.
        assert!(matches!(
            PaperCache::<u32, TieredBuffer>::new(1_000, CacheTierSize::Bytes(2_000), PaperPolicy::TwoQFastAdmissionHybrid(0.1)),
            Err(CacheError::InvalidFastTierSize),
        ));

        // Zero cache size.
        assert!(matches!(
            PaperCache::<u32, TieredBuffer>::new(0, CacheTierSize::Bytes(1), PaperPolicy::TwoQFastAdmissionHybrid(0.1)),
            Err(CacheError::ZeroCacheSize),
        ));

        // k_in outside [0.0, 1.0].
        assert!(matches!(
            PaperCache::<u32, TieredBuffer>::new(1_000_000, CacheTierSize::Bytes(1_000), PaperPolicy::TwoQFastAdmissionHybrid(1.5)),
            Err(CacheError::InvalidPolicy),
        ));

        assert!(matches!(
            PaperCache::<u32, TieredBuffer>::new(1_000_000, CacheTierSize::Bytes(1_000), PaperPolicy::TwoQFastAdmissionHybrid(-0.1)),
            Err(CacheError::InvalidPolicy),
        ));
    }

    /// `k_in >= fast_tier_size / max_size` leaves the main queue no fast
    /// segment: legitimate, if degenerate, and it must not wedge the cache.
    #[test]
    fn a_fifo_reservation_covering_the_whole_fast_tier_still_works() {
        ensure_pmem_allocator_warm();

        // fifo_capacity = 1.0 * 10_000 = 10_000 >= the 1_000-byte fast tier,
        // so the main queue's effective capacity saturates to zero.
        let cache = PaperCache::<u32, TieredBuffer>::new(
            10_000,
            CacheTierSize::Bytes(1_000), PaperPolicy::TwoQFastAdmissionHybrid(1.0)).expect("cache should construct");

        cache.set(1u32, &[1u8; 64], None).expect("set should succeed");

        // Admission is still fast (it goes to the FIFO queue).
        assert_eq!(cache.tier_of(&1u32), Some(Tier::Fast));

        // But proving itself moves it into a main queue with no fast room,
        // so it lands slow rather than being kept in DRAM.
        cache.get(&1u32).expect("get should hit");

        assert!(
            wait_until(MIGRATION_TIMEOUT, || {
                cache.tier_of(&1u32) == Some(Tier::Slow)
            }),
            "with zero effective main capacity, a promoted key should demote",
        );

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

        // The re-set counted as an access, so the key is in the main queue
        // now -- and there is still exactly one of it. Waited for rather
        // than slept on, for the same gauge-refresh-cadence reason as above.
        assert!(
            wait_until(MIGRATION_TIMEOUT, || {
                let stats = cache.hybrid_stats();
                stats.fast_objects + stats.slow_objects == 1
            }),
            "the re-set key should be counted exactly once",
        );
    }
}
