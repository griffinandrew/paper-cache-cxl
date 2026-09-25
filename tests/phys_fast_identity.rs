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
//! A binary of its own because P and the live-cache counts are
//! PROCESS-GLOBAL: a test in another binary building a cache on another
//! thread would be counted in the same P. Inside this binary every test
//! takes [`one_cache_at_a_time`] for its whole body, so exactly one cache is
//! alive at a time here too. The full-suite builds (`--tests`) run it in
//! every unit build -- default, merged_object_store and hashbrown_dram, each
//! with and without thin_header -- since it is gated only on
//! `hybrid_cache_common`.
//!
//! Every hybrid design is compiled into every hybrid build (the design is the
//! runtime `PaperPolicy`), so:
//!
//!   * `phys_fast_equals_the_stacks_fast_used_at_quiescence_and_returns_to_zero`
//!     runs the four orders both stores implement -- LRU, FIFO, CLOCK and LFU
//!     (`MergedOrder::from_policy`) -- at the full workload, in every build;
//!   * one test per OTHER design -- the lazy-copy LRU, the size-split LRU
//!     (where P must also equal the small plus the large segment's fast
//!     bytes), LRU-LFU, the five 2Q and the thirteen S3-FIFO designs -- at a
//!     third of that workload, with every new key read twice and evicted keys
//!     set again (see `Workload::touch_and_readmit`), in the DashMap and
//!     hashbrown builds (the merged store refuses them at construction). One
//!     test each, so a design whose identity fails is `#[ignore]`d with its
//!     reason without hiding the rest. Two are: the faithful S3-FIFO designs
//!     with a SLOW small queue, whose stack strands a promotion (see their
//!     cases);
//!   * `a_flat_cache_with_fast_values_is_counted_apart_from_the_tiered_ones`
//!     checks the second live count.
//!
//! The workload makes the stack demote (the fast tier is a small fraction of
//! the data), promote (repeated gets on slow keys for the designs that
//! promote on a hit; CLOCK recycles a referenced key at the eviction hand, so
//! it promotes once the cache is full and evicting), evict, overwrite (the
//! superseded value is refunded) and delete. Sets, overwrites and deletes are
//! made ONE AT A TIME, each followed by a wait for quiescence, and gets in
//! bursts. That keeps the workload clear of pre-existing placement bugs that
//! are S3's to fix and would fail the identity for reasons that are not P's.
//! One is observed: with the per-set wait removed, the DashMap LFU stack's
//! stale admission latch leaves every value of a burst of new keys in DRAM
//! while the stack counts most of them slow, with nothing queued to move them
//! -- "stack fast/slow objects 36/111 vs map 147 (147 physically fast)". Two
//! more are timing races a burst could hit and this workload's burst variant
//! did not: an overwrite racing a migration of its own key, and a delete and
//! re-set racing a stale queued demotion. A burst of gets touches none of
//! them and still exercises in-flight copies.
//!
//! At every quiescent check, beside the identity: at least one value is
//! physically fast (an empty fast tier would make `P == 0 == fast_used`
//! vacuous); `effective_fast_capacity == F - L * omega`, the status' figure
//! against the test's own (the map's live keys times the reservation the
//! stack made for its first key); `effective_fast_capacity == F -
//! fast_metadata_bytes`, i.e. the stack reserves exactly that, in every
//! design but the seven that also reserve their ghost's DRAM
//! (`Design::reserves_ghost_dram`, whose excess is printed); and the peak,
//! forgotten first (`phys::reset_fast_bytes_max`), is sampled back up to the
//! quiescent P -- the peak is process-global, so without the reset one left
//! by an earlier cache or an earlier phase would satisfy the check unsampled.
//!
//! Under `measured_accounting` + `segregated_value_arena` it also checks P
//! against the allocator: the change in `measured::allocated(NODE_FAST_VALUES)`
//! -- every value allocation in the segregated pool, at jemalloc's usable size
//! -- equals the change in P exactly, since the workload stores no zero-length
//! value (the one size where the two units differ; see `phys`'s module doc).
#![cfg(feature = "hybrid_cache_common")]

use std::{
    collections::BTreeMap,
    sync::{Mutex, MutexGuard},
    time::{Duration, Instant},
};

use paper_cache::{
    phys, BufferDRAM, BufferPMEM, CacheTierSize, HybridStats, PaperCache, PaperPolicy, Tier,
    TieredBuffer,
};

type Cache = PaperCache<u64, TieredBuffer>;

/// What the DashMap-family stacks add per FAST object on top of P under
/// `fused_value`: the u64 key and the 4-byte expiry, which `base_size` counts
/// although they are inside the item (no TTLs here, so no `get_ttl_overhead`).
/// Zero in every other build: there the stack's figure IS P's.
const STACK_EXTRA_PER_OBJECT: u64 =
    if cfg!(all(feature = "fused_value", not(feature = "merged_object_store"))) { 8 + 4 } else { 0 };

const QUIESCE_TIMEOUT: Duration = Duration::from_secs(20);

/// One design's run: the cache's size and how many keys each phase sets.
#[derive(Debug, Clone, Copy)]
struct Workload {
    max_size: u64,
    /// F, the fast tier's whole budget (the size-split design splits it
    /// between its two segments).
    fast: u64,
    /// New keys in phase A.
    keys: u64,
    /// More new keys in phase C, every one of them evicting.
    evicting: u64,
    /// Each new key is read twice right after it is set, and after phase A
    /// every second key the cache has evicted is set again (a read-through
    /// re-admission). The designs that admit a new key to the SLOW tier --
    /// the plain 2Q and S3-FIFO families -- fill their fast tier only from
    /// keys that are re-read or re-admitted; without this they reach phase
    /// A's check with nothing fast, where the identity is vacuous. Off at
    /// FULL, which is S2's workload unchanged.
    touch_and_readmit: bool,
}

/// The four orders both stores implement, at S2's size: small enough that
/// the cache fills and evicts part-way through phase A, which is what runs
/// the CLOCK hand.
const FULL: Workload = Workload {
    max_size: 192 * 1024,
    fast: 48 * 1024,
    keys: 160,
    evicting: 40,
    touch_and_readmit: false,
};

/// Every other design, at a third of it, so the whole binary stays near 40 s
/// in the builds that run all of them: each quiescence wait costs ~6 ms, and
/// there are ~110 of them per design here against ~300 at FULL.
const SMALL: Workload = Workload {
    max_size: 64 * 1024,
    fast: 16 * 1024,
    keys: 48,
    evicting: 12,
    touch_and_readmit: true,
};

/// How a test builds its design's cache.
#[derive(Debug, Clone, Copy)]
enum Design {
    /// `PaperCache::new(max_size, fast, policy)`.
    Policy(PaperPolicy),
    /// The size-split design's own constructor: the fast budget split evenly
    /// between the small and the large segment, and a threshold that sends
    /// about half of the workload's values to each. Not built in the merged
    /// builds, whose store refuses the design.
    #[cfg_attr(feature = "merged_object_store", allow(dead_code))]
    Sized,
}

/// Values shorter than this are small for the size-split design: 300..1_150
/// against 1_150..2_000 (see `len_of`).
const SIZED_THRESHOLD: u64 = 1_150;

impl Design {
    fn label(self) -> String {
        match self {
            Design::Policy(policy) => format!("{policy}"),
            Design::Sized => format!("{}", PaperPolicy::LruSizedCompactHybrid),
        }
    }

    /// The lazy-copy LRU counts its fast OBJECTS by the policy's placement but
    /// its fast BYTES by where the bytes are (`LruLazyCopyCompactHybridStack::
    /// fast_bytes_used`'s doc: "The PHYSICAL number, deliberately"): an object
    /// it has demoted and not yet copied is counted slow and its bytes fast.
    /// Its fast count therefore cannot be matched against the physically fast
    /// values to detect quiescence; its bytes -- what the identity compares --
    /// can.
    fn fast_count_is_logical(self) -> bool {
        matches!(self, Design::Policy(PaperPolicy::LruLazyCopyCompactHybrid))
    }

    /// The designs whose `reserved_overhead` adds their ghost's DRAM to the
    /// per-object reservation (`self.ghost.dram_bytes()`, or ghost entries times
    /// `EXACT_GHOST_ENTRY_DRAM_OVERHEAD` in the faithful family), so their
    /// `fast_metadata_bytes` exceeds `L * omega` once anything has been
    /// evicted into the ghost, and `eff = F - L * omega` exceeds the budget
    /// their own settles leave for values by exactly that. The two faithful
    /// REPRIEVE designs share that code but never populate their ghost (their
    /// module doc: variants 3 and 4 carry none), so they are held to the
    /// equality like every other design.
    fn reserves_ghost_dram(self) -> bool {
        matches!(
            self,
            Design::Policy(
                PaperPolicy::S3FifoFaithfulCompactHybrid(..)
                    | PaperPolicy::S3FifoFaithfulFastAdmissionCompactHybrid(..)
                    | PaperPolicy::S3FifoGhostCompactHybrid(..)
                    | PaperPolicy::S3FifoGhostLazyDemotionCompactHybrid(..)
                    | PaperPolicy::S3FifoGhostLazyDemotionFastAdmissionCompactHybrid(..)
                    | PaperPolicy::S3FifoGhostLazyDemotionFastAdmissionMidpointCompactHybrid(..)
                    | PaperPolicy::TwoQGhostCompactHybrid(..)
            )
        )
    }

    fn build(self, w: Workload) -> Cache {
        match self {
            Design::Policy(policy) => Cache::new(w.max_size, CacheTierSize::Bytes(w.fast), policy),
            Design::Sized => Cache::new_sized_compact(
                w.max_size,
                CacheTierSize::Bytes(w.fast / 2),
                CacheTierSize::Bytes(w.fast - w.fast / 2),
                CacheTierSize::Bytes(SIZED_THRESHOLD),
            ),
        }
        .expect("hybrid cache")
    }
}

/// P and both live counts are PROCESS-GLOBAL, and libtest runs a binary's
/// tests on parallel threads, so every test here holds this for its whole
/// body: one cache alive at a time. Poisoning is stepped over: a failed test
/// dropped its cache while unwinding (its guard is declared first, so it is
/// released last), and its own failure is the one to read.
fn one_cache_at_a_time() -> MutexGuard<'static, ()> {
    static LOCK: Mutex<()> = Mutex::new(());

    LOCK.lock().unwrap_or_else(|poisoned| poisoned.into_inner())
}

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
/// physically in DRAM (not asked of the lazy-copy LRU, whose count is logical
/// by design -- `Design::fast_count_is_logical`), nothing is queued, and all
/// of that holds unchanged across five polls 1 ms apart (the worker passes
/// every 1 ms while sets are recent). None of it reads P, so a broken counter
/// cannot make the wait pass or fail -- only the assertions after it.
fn quiesce(
    cache: &Cache,
    lens: &BTreeMap<u64, u32>,
    design: Design,
    label: &str,
) -> (HybridStats, Walk) {
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
            && (s.fast_objects == w.fast_live || design.fast_count_is_logical())
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

        // P and the stack's fast bytes appear in the message only, for the
        // reader: the wait itself never reads P.
        assert!(
            Instant::now() < deadline,
            "{label}: never quiesced -- stack fast/slow objects {}/{} vs map {} ({} \
             physically fast), stack bytes {} vs model {}, pending {:?}; stack fast \
             bytes {} vs the physically fast values' {} (P {})",
            s.fast_objects, s.slow_objects, w.live, w.fast_live,
            s.fast_bytes_used + s.slow_bytes_used,
            w.total_charge + STACK_EXTRA_PER_OBJECT * w.live,
            pending,
            s.fast_bytes_used, w.fast_charge, s.phys_fast_bytes,
        );

        std::thread::sleep(Duration::from_millis(1));
    }
}

/// What one run's checks need besides the cache.
struct Run<'a> {
    label: &'a str,
    design: Design,
    fast: u64,
    /// The per-object reservation, as the stack made it for the first key
    /// (one object, nothing evicted, so no ghost).
    omega: u64,
    /// The most the stack's reservation exceeded `L * omega` at a check: its
    /// ghost's DRAM, in the designs that reserve it.
    ghost_max: std::cell::Cell<u64>,
    p0: i64,
    #[allow(dead_code)]
    m0: i64,
}

/// The identity, checked at a quiescent point.
fn check(cache: &Cache, lens: &BTreeMap<u64, u32>, run: &Run, phase: &str, hits: u64) {
    let label = format!("{} {phase}", run.label);
    let (s, w) = quiesce(cache, lens, run.design, &label);
    let p = phys::fast_bytes_signed() - run.p0;

    eprintln!(
        "T9 {label}: live={} fast_live={} P={p} fast_used={} fast/slow objects {}/{} \
         metadata={} eff={}",
        w.live, w.fast_live, s.fast_bytes_used, s.fast_objects, s.slow_objects,
        s.fast_metadata_bytes, s.effective_fast_capacity,
    );

    // Something is fast: with an empty fast tier `P == 0 == fast_used` would
    // hold whatever the counter did.
    assert!(w.fast_live > 0, "{label}: no value is physically fast ({} live)", w.live);

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

    // The size-split design's two fast segments, each gauged apart.
    if matches!(run.design, Design::Sized) {
        assert_eq!(
            p as u64 + STACK_EXTRA_PER_OBJECT * s.fast_objects,
            s.small_fast_bytes_used + s.large_fast_bytes_used,
            "{label}: P != the small plus the large segment's fast bytes ({} + {})",
            s.small_fast_bytes_used, s.large_fast_bytes_used,
        );
    }

    assert_eq!(s.phys_fast_bytes, phys::fast_bytes(), "{label}: HybridStats exports P");
    assert_eq!(s.live_tiered_caches, 1, "{label}: exactly this cache is alive");
    assert_eq!(s.live_flat_fast_caches, 0, "{label}: and no flat cache with fast values");
    assert_eq!(s.fast_hits + s.slow_hits, hits, "{label}: every hit is counted in one tier");

    // eff = F - L * omega: the status' figure (its object count times the
    // omega it recorded at construction) against the test's (the map's live
    // keys times the reservation the stack made for its first key).
    let per_object = w.live * run.omega;
    assert_eq!(
        s.effective_fast_capacity,
        run.fast.saturating_sub(per_object),
        "{label}: eff != F - L * omega ({} live, omega {})",
        w.live, run.omega,
    );

    // And the stack reserves exactly that -- `eff == F - fast_metadata_bytes`
    // -- except in the designs that also reserve their ghost's DRAM, where
    // the excess is recorded and reported (see `Design::reserves_ghost_dram`).
    assert!(
        s.fast_metadata_bytes >= per_object,
        "{label}: the stack reserves {} B, less than L * omega = {per_object}",
        s.fast_metadata_bytes,
    );
    let ghost = s.fast_metadata_bytes - per_object;
    if run.design.reserves_ghost_dram() {
        run.ghost_max.set(run.ghost_max.get().max(ghost));
    } else {
        assert_eq!(
            s.effective_fast_capacity,
            run.fast.saturating_sub(s.fast_metadata_bytes),
            "{label}: eff != F - the stack's reservation ({} B for {} objects)",
            s.fast_metadata_bytes, w.live,
        );
    }

    assert!(
        (phys::fast_bytes_signed() - phys::fast_bytes_approx()).abs()
            < (phys::SHARDS as i64) * phys::FOLD_BYTES,
        "{label}: approx is within SHARDS * FOLD_BYTES of exact",
    );

    // The peak: forgotten, then sampled again by this cache's worker, at
    // least up to the quiescent P (nothing is allocating, so no fold can).
    phys::reset_fast_bytes_max();
    let p_now = phys::fast_bytes();
    let deadline = Instant::now() + Duration::from_secs(3);
    while phys::fast_bytes_max() < p_now {
        assert!(
            Instant::now() < deadline,
            "{label}: no pass sampled P into the peak: {} after the reset, P {p_now}",
            phys::fast_bytes_max(),
        );
        std::thread::sleep(Duration::from_millis(1));
    }

    #[cfg(all(feature = "measured_accounting", feature = "segregated_value_arena"))]
    assert_eq!(
        measured_values() - run.m0,
        p,
        "{label}: the segregated value pool's measured bytes moved by a different amount than P",
    );
}

/// One cache, one design, start to drop. Returns the stats at the last check.
fn run(design: Design, w: Workload) -> HybridStats {
    let _one = one_cache_at_a_time();

    let label = design.label();
    let p0 = phys::fast_bytes_signed();

    #[cfg(all(feature = "measured_accounting", feature = "segregated_value_arena"))]
    let m0 = measured_values();
    #[cfg(not(all(feature = "measured_accounting", feature = "segregated_value_arena")))]
    let m0 = 0;

    assert_eq!(
        (phys::live_tiered_caches(), phys::live_flat_fast_caches()),
        (0, 0),
        "{label}: no other cache alive before this one",
    );

    let cache = design.build(w);
    assert_eq!(phys::live_tiered_caches(), 1, "{label}: counted on construction");

    let mut lens = BTreeMap::new();
    let mut hits = 0u64;

    let set = |cache: &Cache, lens: &mut BTreeMap<u64, u32>, key: u64, generation: u64| {
        cache.set(key, &value(key, generation), None).expect("set");
        lens.insert(key, len_of(key, generation));
        quiesce(cache, lens, design, &format!("{label} set({key}, gen {generation})")).0
    };

    // Two reads of a key just set, when the workload asks for them.
    let touch = |cache: &Cache, key: u64, hits: &mut u64| {
        if w.touch_and_readmit {
            for _ in 0..2 {
                if cache.get(&key).is_ok() {
                    *hits += 1;
                }
            }
        }
    };

    // The first key: one object, nothing evicted, so what the stack reserves
    // now is omega alone.
    let omega = set(&cache, &mut lens, 0, 0).fast_metadata_bytes;
    assert!(omega > 0, "{label}: no per-object reservation to check eff against");
    touch(&cache, 0, &mut hits);

    let run = Run {
        label: &label,
        design,
        fast: w.fast,
        omega,
        ghost_max: std::cell::Cell::new(0),
        p0,
        m0,
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
    //    cache fills and evicts. Then, when the workload asks, every second
    //    evicted key is set again.
    for key in 1..w.keys {
        set(&cache, &mut lens, key, 0);
        touch(&cache, key, &mut hits);
    }
    if w.touch_and_readmit {
        let evicted: Vec<u64> = (0..w.keys).filter(|k| cache.tier_of(k).is_none()).collect();
        for key in evicted.into_iter().step_by(2) {
            set(&cache, &mut lens, key, 0);
            touch(&cache, key, &mut hits);
        }
    }
    check(&cache, &lens, &run, "A (sets)", hits);

    // B. A burst of repeated gets on a third of the keys, most of them slow:
    //    the designs that promote on a hit promote (and demote to make room),
    //    CLOCK sets reference bits, FIFO does nothing. Copies are in flight
    //    while it runs.
    let hot: Vec<u64> = (0..w.keys).step_by(3).collect();
    get_burst(&cache, &hot, 4, &mut hits);
    check(&cache, &lens, &run, "B (gets)", hits);

    // C. More new keys: every set now evicts, and CLOCK's hand recycles the
    //    keys B referenced -- a promotion for the slow ones.
    for key in w.keys..w.keys + w.evicting {
        set(&cache, &mut lens, key, 0);
        touch(&cache, key, &mut hits);
    }
    check(&cache, &lens, &run, "C (evicting sets)", hits);

    // D. Overwrites at new sizes, one at a time: each old value is superseded
    //    and must be refunded whichever tier it was in.
    let live: Vec<u64> = lens.keys().copied().filter(|k| cache.tier_of(k).is_some()).collect();
    for key in live.iter().copied().step_by(4) {
        set(&cache, &mut lens, key, 1);
    }
    check(&cache, &lens, &run, "D (overwrites)", hits);

    // E. Deletes, one at a time.
    let live: Vec<u64> = lens.keys().copied().filter(|k| cache.tier_of(k).is_some()).collect();
    let deleted: Vec<u64> = live.iter().copied().skip(1).step_by(5).collect();
    for key in &deleted {
        cache.del(key).expect("del of a live key");
        quiesce(&cache, &lens, design, &format!("{label} del({key})"));
    }
    check(&cache, &lens, &run, "E (dels)", hits);

    // F. Re-sets of deleted keys, each after the delete has quiesced.
    for key in deleted.iter().copied().step_by(2) {
        set(&cache, &mut lens, key, 2);
    }

    // G. A last burst of gets over everything, hits and misses alike.
    let all: Vec<u64> = (0..w.keys + w.evicting).collect();
    get_burst(&cache, &all, 2, &mut hits);
    check(&cache, &lens, &run, "G (final gets)", hits);

    // The allocator's side of the last check, printed so a log shows the
    // measured comparison really ran (`check` asserted the two equal).
    #[cfg(all(feature = "measured_accounting", feature = "segregated_value_arena"))]
    eprintln!(
        "T9 {label}: measured(NODE_FAST_VALUES) delta {} == P {} at the last check",
        measured_values() - m0,
        phys::fast_bytes_signed() - p0,
    );

    let stats = cache.hybrid_stats();

    // This cache's own figures only: the peak is process-global (and was
    // reset at every check), so it is not printed as this design's.
    eprintln!(
        "T9 {label}: promotions={} demotions={} evictions={} fast_hits={} slow_hits={} \
         omega={omega} ghost_reservation_max={} over_budget_byte_seconds={}",
        stats.promotions, stats.demotions, stats.evictions, stats.fast_hits, stats.slow_hits,
        run.ghost_max.get(), stats.over_budget_byte_seconds,
    );

    drop(cache);

    assert_eq!(
        phys::fast_bytes_signed(),
        p0,
        "{label}: P did not return to its pre-cache value -- a fast allocation was never refunded",
    );
    assert_eq!(phys::live_tiered_caches(), 0, "{label}: uncounted on drop");

    #[cfg(all(feature = "measured_accounting", feature = "segregated_value_arena"))]
    assert_eq!(measured_values(), m0, "{label}: the value pool did not return to empty");

    stats
}

#[test]
fn phys_fast_equals_the_stacks_fast_used_at_quiescence_and_returns_to_zero() {
    let lru = run(Design::Policy(PaperPolicy::LruCompactHybrid), FULL);
    let fifo = run(Design::Policy(PaperPolicy::FifoCompactHybrid), FULL);
    let clock = run(Design::Policy(PaperPolicy::ClockCompactHybrid), FULL);
    let lfu = run(Design::Policy(PaperPolicy::LfuCompactHybrid), FULL);

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

/// The identity for one design beyond the four orders, at the SMALL
/// workload, and that the workload moved values across the tiers (a
/// migration charges or refunds P on a consumer thread) and evicted, and that
/// hits were served from both tiers.
#[cfg_attr(feature = "merged_object_store", allow(dead_code))]
fn other_design(design: Design) {
    let s = run(design, SMALL);
    let label = design.label();

    assert!(s.demotions + s.promotions > 0, "{label}: the workload never migrated anything");
    assert!(s.evictions > 0, "{label}: the workload never evicted");
    assert!(s.slow_hits > 0, "{label}: no hit was served from the slow tier");
    assert!(s.fast_hits > 0, "{label}: no hit was served from the fast tier");
}

/// One `#[test]` per design, cfg'd out of the merged builds, whose store
/// refuses every design but the four orders at construction.
macro_rules! other_designs {
    ($($(#[$attr:meta])* $name:ident => $design:expr;)*) => {$(
        $(#[$attr])*
        #[cfg(not(feature = "merged_object_store"))]
        #[test]
        fn $name() {
            other_design($design);
        }
    )*};
}

other_designs! {
    lru_lazy_copy => Design::Policy(PaperPolicy::LruLazyCopyCompactHybrid);
    lru_sized => Design::Sized;
    lru_lfu => Design::Policy(PaperPolicy::LruLfuCompactHybrid(2));
    two_q => Design::Policy(PaperPolicy::TwoQCompactHybrid(0.25));
    // k_in 0.1, not 0.25: at 0.25 of this cache the admission FIFO's
    // carve-out is the whole 16 KiB fast tier, and phase A ends with nothing
    // fast at all (P == fast_used == 0 -- vacuous, not wrong).
    two_q_fast_admission => Design::Policy(PaperPolicy::TwoQFastAdmissionCompactHybrid(0.1));
    two_q_fast_admission_reprieve =>
        Design::Policy(PaperPolicy::TwoQFastAdmissionReprieveCompactHybrid(0.1));
    two_q_full_fast_admission =>
        Design::Policy(PaperPolicy::TwoQFullFastAdmissionCompactHybrid(0.25, 0.5));
    two_q_ghost => Design::Policy(PaperPolicy::TwoQGhostCompactHybrid(0.5));
    s3_fifo => Design::Policy(PaperPolicy::S3FifoCompactHybrid(0.1));
    /// IGNORED -- the identity FAILS here, and the stack is at fault, not P.
    /// `S3FifoFaithfulCore::evict_small` (slow small queue) promotes a key
    /// seen twice to main's FAST front and, if main is then full, returns
    /// through `evict_main` BEFORE pushing that key's `(key, Fast)`
    /// migration: the stack counts it fast, its bytes stay in CXL, and
    /// nothing is queued to move them. At `set(51)` in phase C: stack fast
    /// objects 8 vs 7 physically fast, pending (0, 0), stack fast bytes
    /// 10,560 vs P 8,512. With the push moved above the early return (a
    /// diagnostic, not committed -- this step changes no behaviour) the whole
    /// case passes. S5's gate would read 8,512 where the settle reads 10,560.
    #[ignore = "stack strands a promotion: evict_small returns via evict_main before pushing its (key, Fast) migration; fast_used 10560 vs P 8512"]
    s3_fifo_faithful => Design::Policy(PaperPolicy::S3FifoFaithfulCompactHybrid(0.1));
    s3_fifo_faithful_fast_admission =>
        Design::Policy(PaperPolicy::S3FifoFaithfulFastAdmissionCompactHybrid(0.1));
    /// IGNORED -- the same `evict_small` early return as `s3_fifo_faithful`
    /// (the reprieve variant shares the core and its slow small queue): at
    /// `set(50)`, stack fast objects 8 vs 7 physically fast, pending (0, 0),
    /// stack fast bytes 10,048 vs P 9,280; passes with the push moved.
    #[ignore = "stack strands a promotion: evict_small returns via evict_main before pushing its (key, Fast) migration; fast_used 10048 vs P 9280"]
    s3_fifo_faithful_reprieve =>
        Design::Policy(PaperPolicy::S3FifoFaithfulReprieveCompactHybrid(0.1));
    s3_fifo_faithful_fast_admission_reprieve =>
        Design::Policy(PaperPolicy::S3FifoFaithfulFastAdmissionReprieveCompactHybrid(0.1));
    s3_fifo_ghost => Design::Policy(PaperPolicy::S3FifoGhostCompactHybrid(0.1));
    s3_fifo_ghost_lazy_demotion =>
        Design::Policy(PaperPolicy::S3FifoGhostLazyDemotionCompactHybrid(0.1));
    s3_fifo_ghost_lazy_demotion_fast_admission =>
        Design::Policy(PaperPolicy::S3FifoGhostLazyDemotionFastAdmissionCompactHybrid(0.1));
    s3_fifo_ghost_lazy_demotion_fast_admission_midpoint =>
        Design::Policy(PaperPolicy::S3FifoGhostLazyDemotionFastAdmissionMidpointCompactHybrid(0.1));
    s3_fifo_lazy_demotion_fast_admission_midpoint_reprieve =>
        Design::Policy(PaperPolicy::S3FifoLazyDemotionFastAdmissionMidpointReprieveCompactHybrid(0.1));
    s3_fifo_lazy_demotion_fast_admission_reprieve =>
        Design::Policy(PaperPolicy::S3FifoLazyDemotionFastAdmissionReprieveCompactHybrid(0.1));
    s3_fifo_lazy_demotion_reprieve =>
        Design::Policy(PaperPolicy::S3FifoLazyDemotionReprieveCompactHybrid(0.1));
    s3_fifo_lazy_demotion_fast_admission_split_slow_reprieve =>
        Design::Policy(PaperPolicy::S3FifoLazyDemotionFastAdmissionSplitSlowReprieveCompactHybrid(0.1));
}

/// The second live count. A flat cache whose values are fast charges P like a
/// tiered one and is counted in `live_flat_fast_caches`; a flat cache whose
/// values are slow charges nothing and is not; and a tiered cache built
/// beside them exports both counts, so a reader of its stats can see that P
/// is not its alone.
#[test]
fn a_flat_cache_with_fast_values_is_counted_apart_from_the_tiered_ones() {
    let _one = one_cache_at_a_time();

    let p0 = phys::fast_bytes_signed();
    let one_value = phys::value_charge::<u64>(1_000) as i64;

    assert_eq!((phys::live_tiered_caches(), phys::live_flat_fast_caches()), (0, 0));

    let flat = PaperCache::<u64, BufferDRAM>::new(1 << 20, &[PaperPolicy::Lru], PaperPolicy::Lru)
        .expect("a flat cache with fast values");
    assert_eq!(phys::live_flat_fast_caches(), 1, "counted on construction");
    assert_eq!(phys::live_tiered_caches(), 0, "a flat cache is not a tiered one");

    flat.set(1, &[7u8; 1_000], None).expect("set");
    assert_eq!(phys::fast_bytes_signed() - p0, one_value, "its value is in P");

    let pmem = PaperCache::<u64, BufferPMEM>::new(1 << 20, &[PaperPolicy::Lru], PaperPolicy::Lru)
        .expect("a flat cache with slow values");
    pmem.set(2, &[9u8; 1_000], None).expect("set");
    assert_eq!(phys::live_flat_fast_caches(), 1, "a flat cache with SLOW values is not counted");
    assert_eq!(phys::fast_bytes_signed() - p0, one_value, "and its value is not in P");

    let tiered = Design::Policy(PaperPolicy::LruCompactHybrid).build(SMALL);
    let s = tiered.hybrid_stats();
    assert_eq!(
        (s.live_tiered_caches, s.live_flat_fast_caches),
        (1, 1),
        "the tiered cache's stats say P is not its alone",
    );

    drop(tiered);
    drop(pmem);
    drop(flat);

    assert_eq!(
        (phys::live_tiered_caches(), phys::live_flat_fast_caches()),
        (0, 0),
        "uncounted on drop",
    );
    assert_eq!(phys::fast_bytes_signed(), p0, "the flat cache's value was refunded");
}
