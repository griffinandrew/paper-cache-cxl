/*
 * Copyright (c) Kia Shakiba
 *
 * This source code is licensed under the GNU AGPLv3 license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! T9: the PHYS_FAST identity. At quiescence -- the tier gauges agree with the
//! object map, and no migration is pending -- P, the bytes physically
//! allocated in the fast tier's value pool, equals the sum of the stack's
//! `fast_used` (`hybrid_stats().fast_bytes_used`), and once the cache is
//! dropped P is back where it started: every fast allocation was refunded.
//!
//! Run with:
//!   cargo +nightly test --release --features server,lru_compact_hybrid_cache \
//!     --test phys_fast_identity
//!
//! A binary of its own, holding ONE test that builds one cache at a time,
//! because P and the live-cache count are PROCESS-GLOBAL: a sibling test
//! building a cache on another thread would be counted in the same P. The
//! full-suite builds (`--tests`) run it in every unit build -- default,
//! merged_object_store and hashbrown_dram, each with and without
//! thin_header -- since it is gated only on `hybrid_cache_common`.
//!
//! Every hybrid design is compiled into every hybrid build (the design is the
//! runtime `PaperPolicy`), so each run covers the four orders both stores
//! implement: LRU, FIFO, CLOCK and LFU (`MergedOrder::from_policy`; the
//! DashMap-family stacks implement all of them).
//!
//! The workload makes the stack demote (the fast tier is a small fraction of
//! the data), promote (repeated gets on slow keys for LRU and LFU; CLOCK
//! recycles a referenced key at the eviction hand, so it promotes once the
//! cache is full and evicting), evict, overwrite (the superseded value is
//! refunded) and delete. Sets, overwrites and deletes are made ONE AT A TIME,
//! each followed by a wait for quiescence, and gets in bursts. That keeps the
//! workload clear of pre-existing placement bugs that are S3's to fix and
//! would fail the identity for reasons that are not P's. One is observed:
//! with the per-set wait removed, the DashMap LFU stack's stale admission
//! latch leaves every value of a burst of new keys in DRAM while the stack
//! counts most of them slow, with nothing queued to move them -- "stack
//! fast/slow objects 36/111 vs map 147 (147 physically fast)". Two more are
//! timing races a burst could hit and this workload's burst variant did not:
//! an overwrite racing a migration of its own key, and a delete and re-set
//! racing a stale queued demotion. A burst of gets touches none of them and
//! still exercises in-flight copies.
//!
//! Under `measured_accounting` + `segregated_value_arena` it also checks P
//! against the allocator: the change in `measured::allocated(NODE_FAST_VALUES)`
//! -- every value allocation in the segregated pool, at jemalloc's usable size
//! -- equals the change in P exactly, since the workload stores no zero-length
//! value (the one size where the two units differ; see `phys`'s module doc).
#![cfg(feature = "hybrid_cache_common")]

use std::{
    collections::BTreeMap,
    time::{Duration, Instant},
};

use paper_cache::{phys, CacheTierSize, PaperCache, PaperPolicy, Tier, TieredBuffer};

type Cache = PaperCache<u64, TieredBuffer>;

/// Small enough that the cache fills and evicts part-way through the first
/// phase, which is what runs the CLOCK hand.
const MAX_SIZE: u64 = 192 * 1024;
const FAST: u64 = 48 * 1024;
const KEYS: u64 = 160;

/// What the DashMap-family stacks add per FAST object on top of P under
/// `fused_value`: the u64 key and the 4-byte expiry, which `base_size` counts
/// although they are inside the item (no TTLs here, so no `get_ttl_overhead`).
/// Zero in every other build: there the stack's figure IS P's.
const STACK_EXTRA_PER_OBJECT: u64 =
    if cfg!(all(feature = "fused_value", not(feature = "merged_object_store"))) { 8 + 4 } else { 0 };

const QUIESCE_TIMEOUT: Duration = Duration::from_secs(20);

/// Value length for key `key`, generation `generation`: 300..2000 bytes, never
/// zero, spread across size classes so rounding matters.
fn len_of(key: u64, generation: u64) -> u32 {
    300 + ((key * 7_919 + generation * 104_729) % 1_700) as u32
}

fn value(key: u64, generation: u64) -> Vec<u8> {
    vec![(key ^ generation) as u8; len_of(key, generation) as usize]
}

/// The cache as the test's own model and the tag bits see it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Walk {
    /// Keys the map still holds (the rest were evicted or deleted).
    live: u64,
    /// Of those, the ones whose value's tag says fast.
    fast_live: u64,
    /// `value_charge` summed over the physically fast values: what P must be.
    fast_charge: u64,
    /// `value_charge` summed over every live value.
    total_charge: u64,
}

fn walk(cache: &Cache, lens: &BTreeMap<u64, u32>) -> Walk {
    let mut w = Walk { live: 0, fast_live: 0, fast_charge: 0, total_charge: 0 };

    for (key, len) in lens {
        let Some(tier) = cache.tier_of(key) else { continue };
        let charge = phys::value_charge::<u64>(*len);

        w.live += 1;
        w.total_charge += charge;

        if tier == Tier::Fast {
            w.fast_live += 1;
            w.fast_charge += charge;
        }
    }

    w
}

#[cfg(all(feature = "measured_accounting", feature = "segregated_value_arena"))]
fn measured_values() -> i64 {
    use paper_cache::numa_alloc::{measured, NODE_FAST_VALUES};

    measured::allocated(NODE_FAST_VALUES) as i64
}

/// Waits until the worker has taken everything the test did and every
/// migration it started has finished: the stack tracks exactly the keys the
/// map holds with exactly their bytes, its fast count is the number of values
/// physically in DRAM, nothing is queued, and all of that holds unchanged
/// across five polls. None of it reads P, so a broken counter cannot make the
/// wait pass or fail -- only the assertions after it.
fn quiesce(cache: &Cache, lens: &BTreeMap<u64, u32>, label: &str) -> (paper_cache::HybridStats, Walk) {
    let deadline = Instant::now() + QUIESCE_TIMEOUT;
    let mut last = None;
    let mut stable = 0;

    loop {
        let s = cache.hybrid_stats();
        let w = walk(cache, lens);
        let pending = phys::pending_migrations();

        let settled = s.fast_objects + s.slow_objects == w.live
            && s.fast_bytes_used + s.slow_bytes_used
                == w.total_charge + STACK_EXTRA_PER_OBJECT * w.live
            && s.fast_objects == w.fast_live
            && pending == (0, 0);

        let key = (
            s.fast_objects, s.slow_objects, s.fast_bytes_used, s.slow_bytes_used,
            s.promotions, s.demotions, s.evictions, w,
        );

        stable = if settled && last == Some(key) { stable + 1 } else { 0 };
        last = Some(key);

        if stable >= 5 {
            return (s, w);
        }

        assert!(
            Instant::now() < deadline,
            "{label}: never quiesced -- stack fast/slow objects {}/{} vs map {} ({} \
             physically fast), stack bytes {} vs model {}, pending {:?}",
            s.fast_objects, s.slow_objects, w.live, w.fast_live,
            s.fast_bytes_used + s.slow_bytes_used,
            w.total_charge + STACK_EXTRA_PER_OBJECT * w.live,
            pending,
        );

        std::thread::sleep(Duration::from_millis(2));
    }
}

/// The identity, checked at a quiescent point.
fn check(
    cache: &Cache,
    lens: &BTreeMap<u64, u32>,
    label: &str,
    p0: i64,
    #[allow(unused_variables)] m0: i64,
    hits: u64,
) {
    let (s, w) = quiesce(cache, lens, label);
    let p = phys::fast_bytes_signed() - p0;

    // The counter itself: exactly the live fast allocations the map holds,
    // nothing more (a missed refund) and nothing less (a missed charge).
    assert_eq!(
        p, w.fast_charge as i64,
        "{label}: P does not equal the physically fast values in the map \
         ({} fast of {} live)",
        w.fast_live, w.live,
    );

    // THE identity: the physical counter equals the stack's intent gauge.
    assert_eq!(
        p as u64 + STACK_EXTRA_PER_OBJECT * s.fast_objects,
        s.fast_bytes_used,
        "{label}: P != the stack's fast_used at quiescence ({} fast objects)",
        s.fast_objects,
    );

    assert_eq!(s.phys_fast_bytes, phys::fast_bytes(), "{label}: HybridStats exports P");
    assert_eq!(s.live_tiered_caches, 1, "{label}: exactly this cache is alive");
    assert_eq!(s.fast_hits + s.slow_hits, hits, "{label}: every hit is counted in one tier");
    assert!(s.effective_fast_capacity <= FAST, "{label}: eff = F - L * omega <= F");
    assert!(
        (phys::fast_bytes_signed() - phys::fast_bytes_approx()).abs()
            < (phys::SHARDS as i64) * phys::FOLD_BYTES,
        "{label}: approx is within SHARDS * FOLD_BYTES of exact",
    );

    #[cfg(all(feature = "measured_accounting", feature = "segregated_value_arena"))]
    assert_eq!(
        measured_values() - m0,
        p,
        "{label}: the segregated value pool's measured bytes moved by a different amount than P",
    );
}

/// One cache, one policy, start to drop. Returns the stats at the last check.
fn run(policy: PaperPolicy) -> paper_cache::HybridStats {
    let label = format!("{policy}");
    let p0 = phys::fast_bytes_signed();
    let live0 = phys::live_tiered_caches();

    #[cfg(all(feature = "measured_accounting", feature = "segregated_value_arena"))]
    let m0 = measured_values();
    #[cfg(not(all(feature = "measured_accounting", feature = "segregated_value_arena")))]
    let m0 = 0;

    assert_eq!(live0, 0, "{label}: no tiered cache alive before this one");

    let cache = Cache::new(MAX_SIZE, CacheTierSize::Bytes(FAST), policy).expect("hybrid cache");
    assert_eq!(phys::live_tiered_caches(), 1, "{label}: counted on construction");

    let mut lens = BTreeMap::new();
    let mut hits = 0u64;

    let set = |cache: &Cache, lens: &mut BTreeMap<u64, u32>, key: u64, generation: u64| {
        cache.set(key, &value(key, generation), None).expect("set");
        lens.insert(key, len_of(key, generation));
        quiesce(cache, lens, &format!("{label} set({key}, gen {generation})"));
    };

    let get_burst = |cache: &Cache, keys: &[u64], rounds: usize, hits: &mut u64| {
        for _ in 0..rounds {
            for key in keys {
                if cache.get(key).is_ok() {
                    *hits += 1;
                }
            }
        }
    };

    // A. New keys, one at a time: the fast tier fills and demotes, then the
    //    cache fills and evicts.
    for key in 0..KEYS {
        set(&cache, &mut lens, key, 0);
    }
    check(&cache, &lens, &format!("{label} A (sets)"), p0, m0, hits);

    // B. A burst of repeated gets on a third of the keys, most of them slow:
    //    LRU and LFU promote (and demote to make room), CLOCK sets reference
    //    bits, FIFO does nothing. Copies are in flight while it runs.
    let hot: Vec<u64> = (0..KEYS).step_by(3).collect();
    get_burst(&cache, &hot, 4, &mut hits);
    check(&cache, &lens, &format!("{label} B (gets)"), p0, m0, hits);

    // C. More new keys: every set now evicts, and CLOCK's hand recycles the
    //    keys B referenced -- a promotion for the slow ones.
    for key in KEYS..KEYS + 40 {
        set(&cache, &mut lens, key, 0);
    }
    check(&cache, &lens, &format!("{label} C (evicting sets)"), p0, m0, hits);

    // D. Overwrites at new sizes, one at a time: each old value is superseded
    //    and must be refunded whichever tier it was in.
    let live: Vec<u64> = lens.keys().copied().filter(|k| cache.tier_of(k).is_some()).collect();
    for key in live.iter().copied().step_by(4) {
        set(&cache, &mut lens, key, 1);
    }
    check(&cache, &lens, &format!("{label} D (overwrites)"), p0, m0, hits);

    // E. Deletes, one at a time.
    let live: Vec<u64> = lens.keys().copied().filter(|k| cache.tier_of(k).is_some()).collect();
    let deleted: Vec<u64> = live.iter().copied().skip(1).step_by(5).collect();
    for key in &deleted {
        cache.del(key).expect("del of a live key");
        quiesce(&cache, &lens, &format!("{label} del({key})"));
    }
    check(&cache, &lens, &format!("{label} E (dels)"), p0, m0, hits);

    // F. Re-sets of deleted keys, each after the delete has quiesced.
    for key in deleted.iter().copied().step_by(2) {
        set(&cache, &mut lens, key, 2);
    }

    // G. A last burst of gets over everything, hits and misses alike.
    let all: Vec<u64> = (0..KEYS + 40).collect();
    get_burst(&cache, &all, 2, &mut hits);
    check(&cache, &lens, &format!("{label} G (final gets)"), p0, m0, hits);

    // The allocator's side of the last check, printed so a log shows the
    // measured comparison really ran (`check` asserted the two equal).
    #[cfg(all(feature = "measured_accounting", feature = "segregated_value_arena"))]
    eprintln!(
        "T9 {label}: measured(NODE_FAST_VALUES) delta {} == P {} at the last check",
        measured_values() - m0,
        phys::fast_bytes_signed() - p0,
    );

    let stats = cache.hybrid_stats();

    // The peak is sampled at every worker pass, so it catches up with the
    // quiescent P within a pass; it is never below it.
    let p_now = phys::fast_bytes();
    let deadline = Instant::now() + Duration::from_secs(3);
    while cache.hybrid_stats().phys_fast_bytes_max < p_now {
        assert!(Instant::now() < deadline, "{label}: the peak never reached the quiescent P {p_now}");
        std::thread::sleep(Duration::from_millis(2));
    }

    eprintln!(
        "T9 {label}: promotions={} demotions={} evictions={} fast_hits={} slow_hits={} \
         P_peak>={} eff={} over_budget_byte_seconds={}",
        stats.promotions, stats.demotions, stats.evictions, stats.fast_hits, stats.slow_hits,
        cache.hybrid_stats().phys_fast_bytes_max, stats.effective_fast_capacity,
        stats.over_budget_byte_seconds,
    );

    drop(cache);

    assert_eq!(
        phys::fast_bytes_signed(),
        p0,
        "{label}: P did not return to its pre-cache value -- a fast allocation was never refunded",
    );
    assert_eq!(phys::live_tiered_caches(), live0, "{label}: uncounted on drop");

    #[cfg(all(feature = "measured_accounting", feature = "segregated_value_arena"))]
    assert_eq!(measured_values(), m0, "{label}: the value pool did not return to empty");

    stats
}

#[test]
fn phys_fast_equals_the_stacks_fast_used_at_quiescence_and_returns_to_zero() {
    let lru = run(PaperPolicy::LruCompactHybrid);
    let fifo = run(PaperPolicy::FifoCompactHybrid);
    let clock = run(PaperPolicy::ClockCompactHybrid);
    let lfu = run(PaperPolicy::LfuCompactHybrid);

    // The workload did what the identity is meant to survive: every design
    // demoted and evicted, and every design with a promotion rule promoted
    // (FIFO has none), so P was charged and refunded on consumer threads as
    // well as client ones.
    for (name, s) in [("LRU", lru), ("FIFO", fifo), ("CLOCK", clock), ("LFU", lfu)] {
        assert!(s.demotions > 0, "{name}: the workload never demoted");
        assert!(s.evictions > 0, "{name}: the workload never evicted");
        assert!(s.slow_hits > 0, "{name}: no hit was served from the slow tier");
        assert!(s.fast_hits > 0, "{name}: no hit was served from the fast tier");
    }

    for (name, s) in [("LRU", lru), ("CLOCK", clock), ("LFU", lfu)] {
        assert!(s.promotions > 0, "{name}: the workload never promoted");
    }
}
