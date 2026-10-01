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
//! every unit build -- default and merged_object_store, each
//! with and without thin_header -- since it is gated only on
//! `hybrid_cache_common`.
//!
//! Every hybrid design is compiled into every hybrid build (the design is the
//! runtime `PaperPolicy`), so:
//!
//!   * `phys_fast_equals_the_stacks_fast_used_at_quiescence_and_returns_to_zero`
//!     runs the four orders both stores implement -- LRU, FIFO, CLOCK and LFU
//!     (`MergedOrder::from_policy`) -- at the full workload, in every build;
//!   * one test per OTHER design -- the size-split LRU (where P must also
//!     equal the small plus the large segment's fast bytes), LRU-LFU, the
//!     four 2Q and the thirteen S3-FIFO designs -- at a
//!     third of that workload, with every new key read twice and evicted keys
//!     set again (see `Workload::touch_and_readmit`), in the DashMap builds
//!     (the merged store refuses them at construction). One
//!     test each, so a design whose identity fails can be `#[ignore]`d with
//!     its reason without hiding the rest. None is: the two faithful S3-FIFO
//!     designs with a SLOW small queue were, until S3 fixed the promotion
//!     their `evict_small` stranded;
//!   * `a_flat_cache_with_fast_values_is_counted_apart_from_the_tiered_ones`
//!     checks the second live count;
//!   * T7 (`an_lfu_burst_strands_no_value`, every build): LFU's sets made in
//!     BURSTS, the client far ahead of the worker, which is what outran the
//!     latch mirror before S3's reconcile.
//!
//! Every quiescent check also takes the PLACEMENT AUDIT (S3,
//! `PaperCache::placement_audit`) and requires it clean -- no value stranded,
//! lagging or untracked -- with its bytes equal to the model's. So every
//! design's case here is also the check that its `PolicyStack::placement_of`
//! agrees, key by key, with where each value's bytes are at quiescence.
//!
//! The workload makes the stack demote (the fast tier is a small fraction of
//! the data), promote (repeated gets on slow keys for the designs that
//! promote on a hit; CLOCK recycles a referenced key at the eviction hand, so
//! it promotes once the cache is full and evicting), evict, overwrite (the
//! superseded value is refunded) and delete. Sets, overwrites and deletes are
//! made ONE AT A TIME, each followed by a wait for quiescence, and gets in
//! bursts. That kept S2's workload clear of the placement bugs S3 fixed: with
//! the per-set wait removed, the DashMap LFU stack's stale admission latch left
//! every value of a burst of new keys in DRAM while the stack counted most of
//! them slow, with nothing queued to move them -- "stack fast/slow objects
//! 36/111 vs map 147 (147 physically fast)"; T7 is that burst, now clean. The
//! two timing races -- an overwrite racing a migration of its own key, and a
//! delete and re-set racing a stale queued migration -- are pinned
//! deterministically by `worker::policy::reconcile_tests` in the lib (the
//! second is now fixed at the re-set's `Set`, by the new-key rule); T7's
//! phase D makes deletes and re-sets in a burst.
//!
//! And because those phases are made one at a time with a fresh mirror and
//! nothing in flight -- the two reads of a new key wait for quiescence too --
//! NO set corrective may be queued in them (review m5): a
//! `reconcile_set_to_fast`/`_to_slow` delta over a phase would be a stack
//! placing a key without a push, or `admission_tier` building a value
//! somewhere the stack does not place it -- both of which the reconcile would
//! silently repair on every set, and the audit alone could not tell from a
//! correct design -- and a `reconcile_set_new_key` delta the new-key rule
//! fencing a key with nothing in flight. The get bursts may heal, and are
//! not held to it.
//!
//! At every quiescent check, beside the identity: no migration is in flight
//! in the cache's per-key buckets (`PaperCache::migrations_in_flight`: every
//! hand-off to the consumers was matched by a finish, whatever the entry did);
//! at least one value is physically fast (an empty fast tier would make `P == 0 == fast_used`
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
//! S5a: every check also reports M, the cache's own DRAM metadata as its
//! worker publishes it (`PaperCache::dram_metadata`: the map's structures,
//! the stack's, one header per live object), beside the model's `L * omega`
//! (`fast_metadata_bytes`), per object -- the two differ by design, since M
//! counts the structures at whatever load they are at -- and holds M to its
//! own parts: the export is the total, the header part is one header per
//! live key, `F - M` is `effective_fast_capacity_measured`. M against the
//! allocator is `tests/dram_metadata_identity.rs`'s. The quiescence wait
//! includes M, so a check reads it settled. T7 goes through the same check.
//!
//! Under `measured_accounting` + `segregated_value_arena` it also checks P
//! against the allocator: the change in `measured::allocated(NODE_FAST_VALUES)`
//! -- every value allocation in the segregated pool, at jemalloc's usable size
//! -- equals the change in P exactly, since the workload stores no zero-length
//! value (the one size where the two units differ; see `phys`'s module doc).
#![cfg(feature = "hybrid_cache_common")]

mod common;

use std::{
    collections::BTreeMap,
    sync::{Mutex, MutexGuard},
    time::{Duration, Instant},
};

use paper_cache::{
    phys, BufferDRAM, BufferPMEM, CacheTierSize, GateConfig, GateMode, GateState, HybridStats, MetadataModel,
    PaperCache, PaperPolicy, Tier, TieredBuffer,
};

type Cache = PaperCache<u64, TieredBuffer>;

const QUIESCE_TIMEOUT: Duration = Duration::from_secs(20);

/// The fast tier's drain target, as a fraction of its effective budget: the
/// crate's default (0.95, since E1b -- it was 0.98) unless this run overrides
/// it, read the way the crate reads it. Not exported by the crate, so the
/// default is restated here: the settle-target check below is only as tight
/// as this figure, and a default that moves again fails it rather than
/// leaving it loose.
fn drain_target_ratio() -> f64 {
    std::env::var("FAST_TIER_DRAIN_TARGET")
        .ok()
        .and_then(|v| v.parse::<f64>().ok())
        .filter(|v| *v > 0.0 && *v <= 1.0)
        .unwrap_or(0.95)
}

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

    /// The designs whose `reserved_overhead` adds their ghost's DRAM to the
    /// per-object reservation (`self.ghost.dram_bytes()`, or ghost entries times
    /// `EXACT_GHOST_ENTRY_DRAM_OVERHEAD` in the faithful family), so their
    /// `fast_metadata_bytes` exceeds `L * omega` once anything has been
    /// evicted into the ghost. Until S5, `eff = F - L * omega` exceeded the
    /// budget their own settles left for values by exactly that; since S5's
    /// ghost unification eff is `F - M_model`, the ghost included, in every
    /// design. The two faithful REPRIEVE designs share that code but never
    /// populate their ghost (their module doc: variants 3 and 4 carry none),
    /// so they reserve `L * omega` exactly, like every other design.
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

    /// The fast-admission pair of the faithful S3-FIFO family keeps its small
    /// queue in DRAM at `ratio * max_size`, unclamped to the tier (S5's
    /// design 0.6, question Q7): the one design whose DRAM its settles do not
    /// bound, and so the one held to the identity but not to the settle
    /// target.
    fn bounds_its_dram(self) -> bool {
        !matches!(
            self,
            Design::Policy(
                PaperPolicy::S3FifoFaithfulFastAdmissionCompactHybrid(..)
                    | PaperPolicy::S3FifoFaithfulFastAdmissionReprieveCompactHybrid(..)
            )
        )
    }

    /// Built under the PER-OBJECT metadata model (S5), which this binary's
    /// identities are written in: `eff = F - M_model`, M_model the stack's
    /// own reservation.
    fn build(self, w: Workload) -> Cache {
        self.build_with(w, GateConfig::default().mode)
    }

    /// The byte gate's state for this design's cache, alone in the process (S5
    /// B2): the designs whose settles do not bound their DRAM -- the faithful
    /// fast-admission pair -- run ungated.
    fn gate_state(self) -> GateState {
        match self {
            Design::Policy(PaperPolicy::S3FifoFaithfulFastAdmissionCompactHybrid(_))
            | Design::Policy(PaperPolicy::S3FifoFaithfulFastAdmissionReprieveCompactHybrid(_)) => GateState::Ungated,
            _ => GateState::Enabled,
        }
    }

    /// `build`, with the byte gate's mode (S5 B2): T9 runs with the default,
    /// `Block` -- one cache at a time, so the gate is on, and every design's
    /// rest is checked against it -- and T7 with `Off` (see `burst`).
    fn build_with(self, w: Workload, mode: GateMode) -> Cache {
        let mut gate = GateConfig::default();
        gate.metadata_model = MetadataModel::PerObject;
        gate.mode = mode;

        match self {
            Design::Policy(policy) => Cache::new_with_gate(w.max_size, CacheTierSize::Bytes(w.fast), policy, gate),
            Design::Sized => Cache::new_sized_compact_with_gate(
                w.max_size,
                CacheTierSize::Bytes(w.fast / 2),
                CacheTierSize::Bytes(w.fast - w.fast / 2),
                CacheTierSize::Bytes(SIZED_THRESHOLD),
                gate,
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
/// physically in DRAM, nothing is queued, and all of that holds unchanged
/// across five polls 1 ms apart (the worker passes every 1 ms while sets are
/// recent). None of it reads P, so a broken counter
/// cannot make the wait pass or fail -- only the assertions after it.
fn quiesce(
    cache: &Cache,
    lens: &BTreeMap<u64, u32>,
    label: &str,
) -> (HybridStats, Walk) {
    common::settle(
        QUIESCE_TIMEOUT,
        Duration::from_millis(1),
        || {
            let s = cache.hybrid_stats();
            let w = walk(cache, lens);
            let pending = phys::pending_migrations();

            let settled = s.fast_objects + s.slow_objects == w.live
                && s.fast_bytes_used + s.slow_bytes_used == w.total_charge
                && s.fast_objects == w.fast_live
                && pending == (0, 0);

            // M (S5a) too: its worker publishes it at the end of every pass, so a
            // wait that ended before that pass would read a stale one.
            let key = (
                s.fast_objects, s.slow_objects, s.fast_bytes_used, s.slow_bytes_used,
                s.promotions, s.demotions, s.evictions, w, cache.dram_metadata(),
            );

            (settled, key, (s, w))
        },
        |(s, w)| {
            // P and the stack's fast bytes appear in the message only, for the
            // reader: the wait itself never reads P.
            format!(
                "{label}: never quiesced -- stack fast/slow objects {}/{} vs map {} ({} \
                 physically fast), stack bytes {} vs model {}, pending {:?}; stack fast \
                 bytes {} vs the physically fast values' {} (P {})",
                s.fast_objects, s.slow_objects, w.live, w.fast_live,
                s.fast_bytes_used + s.slow_bytes_used,
                w.total_charge,
                phys::pending_migrations(),
                s.fast_bytes_used, w.fast_charge, s.phys_fast_bytes,
            )
        },
    )
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
    /// The byte gate's state every check must find (S5 B2): T9's cache runs
    /// gated, alone, except the designs the gate leaves ungated; T7's off.
    gate: GateState,
}

/// The identity, checked at a quiescent point.
fn check(cache: &Cache, lens: &BTreeMap<u64, u32>, run: &Run, phase: &str, hits: u64) {
    let label = format!("{} {phase}", run.label);
    let (s, w) = quiesce(cache, lens, &label);
    let p = phys::fast_bytes_signed() - run.p0;

    eprintln!(
        "T9 {label}: live={} fast_live={} P={p} fast_used={} fast/slow objects {}/{} \
         metadata={} eff={}",
        w.live, w.fast_live, s.fast_bytes_used, s.fast_objects, s.slow_objects,
        s.fast_metadata_bytes, s.effective_fast_capacity,
    );

    // S5a: M, measured, beside the model. Reported rather than held to the
    // model: M counts the structures at whatever load they are at (at this
    // size mostly the DashMap's 256 near-empty shards, or the merged store's
    // 32 first slab chunks), the model a per-object constant fitted at 2^k.
    let m = cache.dram_metadata();
    let per_object = |bytes: u64| bytes as f64 / w.live.max(1) as f64;
    eprintln!(
        "T9 {label}: M={} (map {} stack {} headers {} slow-node {}) = {:.1} B/object against \
         the model's {} = {:.1} B/object; eff measured {} vs modelled {}",
        m.total(), m.map, m.stack, m.headers, m.slow, per_object(m.total()),
        s.fast_metadata_bytes, per_object(s.fast_metadata_bytes),
        s.effective_fast_capacity_measured, s.effective_fast_capacity,
    );
    assert_eq!(s.dram_metadata_bytes, m.total(), "{label}: HybridStats exports M, its parts' sum");

    // S5 B2: the gate as it must be here -- running for T9's cache, alone (its
    // only gated coverage of most designs: the test review) -- and no set held
    // for a second.
    assert_eq!(s.gate_state, run.gate, "{label}: the byte gate's state");
    assert!(s.gate_wait_ns_max < 1_000_000_000, "{label}: a set waited {} ns", s.gate_wait_ns_max);
    assert_eq!(
        m.headers,
        w.live * paper_cache::value::dram_header_bytes::<u64>(),
        "{label}: M counts one DRAM value header per live object ({} live)",
        w.live,
    );
    assert!(m.map > 0 && m.stack > 0, "{label}: M has a map part and a stack part: {m:?}");
    assert_eq!(
        s.effective_fast_capacity_measured,
        run.fast.saturating_sub(m.total()),
        "{label}: eff measured != F - M",
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
        p as u64,
        s.fast_bytes_used,
        "{label}: P != the stack's fast_used at quiescence ({} fast objects)",
        s.fast_objects,
    );

    // The size-split design's two fast segments, each gauged apart.
    if matches!(run.design, Design::Sized) {
        assert_eq!(
            p as u64,
            s.small_fast_bytes_used + s.large_fast_bytes_used,
            "{label}: P != the small plus the large segment's fast bytes ({} + {})",
            s.small_fast_bytes_used, s.large_fast_bytes_used,
        );
    }

    assert_eq!(s.phys_fast_bytes, phys::fast_bytes(), "{label}: HybridStats exports P");
    assert_eq!(s.live_tiered_caches, 1, "{label}: exactly this cache is alive");
    assert_eq!(s.live_flat_fast_caches, 0, "{label}: and no flat cache with fast values");
    assert_eq!(s.fast_hits + s.slow_hits, hits, "{label}: every hit is counted in one tier");

    // The stack reserves L * omega -- the map's live keys times the
    // reservation it made for its first key -- plus, in the designs that keep
    // one, its ghost's DRAM, whose excess is recorded and reported (see
    // `Design::reserves_ghost_dram`).
    let per_object = w.live * run.omega;
    assert!(
        s.fast_metadata_bytes >= per_object,
        "{label}: the stack reserves {} B, less than L * omega = {per_object}",
        s.fast_metadata_bytes,
    );
    let ghost = s.fast_metadata_bytes - per_object;
    if run.design.reserves_ghost_dram() {
        run.ghost_max.set(run.ghost_max.get().max(ghost));
    } else {
        assert_eq!(ghost, 0, "{label}: a design without a ghost reserves more than L * omega");
    }

    // eff = F - M_model (S5), one figure in every design -- the ghost's DRAM
    // included, where until S5 the ghost designs' eff left it out: under the
    // per-object model this binary pins, M_model is the stack's reservation.
    assert_eq!(s.metadata_model, MetadataModel::PerObject, "{label}: the model");
    assert_eq!(s.dram_metadata_bytes_model, s.fast_metadata_bytes, "{label}: M_model is the stack's reservation");
    assert_eq!(
        s.effective_fast_capacity,
        run.fast.saturating_sub(s.fast_metadata_bytes),
        "{label}: eff != F - M_model ({} B for {} objects)",
        s.fast_metadata_bytes, w.live,
    );

    // T9+ (S5): at rest every design is at or under its settle target, `S =
    // 0.95 eff` (E1b; 0.98 before) -- the settles, the resettle each pass (LFU's latched
    // admissions never settled), the DRAM queues policed at their drain
    // targets -- so P + M_model <= F. The faithful fast-admission pair's small
    // queue is not bounded by the tier (`Design::bounds_its_dram`).
    if run.design.bounds_its_dram() {
        let target = (s.effective_fast_capacity as f64 * drain_target_ratio()) as u64;

        assert!(
            s.fast_bytes_used <= target,
            "{label}: {} fast bytes at rest, over the settle target {target} (eff {})",
            s.fast_bytes_used, s.effective_fast_capacity,
        );
        assert!(
            p as u64 + s.dram_metadata_bytes_model <= run.fast,
            "{label}: P {p} + M_model {} over F {}",
            s.dram_metadata_bytes_model, run.fast,
        );
    }

    assert!(
        (phys::fast_bytes_signed() - phys::fast_bytes_approx()).abs()
            < (phys::SHARDS as i64) * phys::FOLD_BYTES,
        "{label}: approx is within SHARDS * FOLD_BYTES of exact",
    );

    // The placement audit (S3): every live value's bytes are where the stack
    // places its key -- nothing stranded, lagging or untracked -- and the
    // audit's bytes, in the same unit, are the model's (so its fast bytes are
    // P). This is what makes every design's case the `placement_of`
    // agreement check.
    let a = cache.placement_audit().expect("a tiered cache answers the audit");
    assert!(
        a.is_clean(),
        "{label}: placement audit -- {} stranded ({} B), {} lagging ({} B), {} untracked \
         ({} B) of {} live",
        a.stranded, a.stranded_bytes, a.lagging, a.lagging_bytes, a.untracked,
        a.untracked_bytes, a.live,
    );
    assert_eq!(
        (a.live, a.fast, a.fast_bytes, a.fast_bytes + a.slow_bytes),
        (w.live, w.fast_live, w.fast_charge, w.total_charge),
        "{label}: the audit's (live, fast, fast bytes, bytes) against the model's",
    );

    // The new-key and heal rules' buckets: every entry handed to the
    // consumers finished -- applied, declined, gone or superseded alike.
    assert_eq!(
        cache.migrations_in_flight(),
        0,
        "{label}: migrations still counted in flight at quiescence -- a hand-off without its finish",
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

/// The PROCESS-GLOBAL set correctives: `(reconcile_set_to_fast,
/// reconcile_set_to_slow, reconcile_set_new_key)`. This binary has one cache
/// alive at a time, so a delta is that cache's.
fn set_correctives(cache: &Cache) -> (u64, u64, u64) {
    let s = cache.hybrid_stats();

    (s.reconcile_set_to_fast, s.reconcile_set_to_slow, s.reconcile_set_new_key)
}

/// No set corrective was queued since `before` (review m5): the phase was
/// made one at a time -- every set, and every read of a key just set,
/// followed by a wait for quiescence -- with a fresh mirror and nothing in
/// flight. So a corrective toward either tier would be a stack placing a key
/// without a push, or a value built where its stack does not place it, which
/// the reconcile would repair unseen; and a new-key one cannot happen at all:
/// nothing is in flight when a set's value is published, and nothing lands
/// between its mark and its handling -- not even another key's migration
/// sharing its bucket (the buckets are indexed by the key's hash, which the
/// cache's `RandomState` seeds anew each run, so such collisions are real).
/// Every build's correctives are the worker's: the merged store's client
/// queues nothing, and builds a new LFU key where the latch the worker
/// published says, as the DashMap stores' does.
fn assert_no_set_correctives(cache: &Cache, label: &str, phase: &str, before: (u64, u64, u64)) {
    let now = set_correctives(cache);

    assert_eq!(
        (now.0 - before.0, now.1 - before.1, now.2 - before.2),
        (0, 0, 0),
        "{label} {phase}: set correctives (to fast, to slow, new key) queued in a one-at-a-time phase",
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
        quiesce(cache, lens, &format!("{label} set({key}, gen {generation})")).0
    };

    // Two reads of a key just set, when the workload asks for them.
    //
    // Followed by a wait for quiescence, like every set: the next set is then
    // made with nothing in flight, which the m5 check below relies on (a read
    // promoting this key while the next key is set can land in the next key's
    // in-flight bucket -- the buckets collide -- and fence it).
    let touch = |cache: &Cache, lens: &BTreeMap<u64, u32>, key: u64, hits: &mut u64| {
        if w.touch_and_readmit {
            for _ in 0..2 {
                if cache.get(&key).is_ok() {
                    *hits += 1;
                }
            }

            quiesce(cache, lens, &format!("{label} touch({key})"));
        }
    };

    // The first key: one object, nothing evicted, so what the stack reserves
    // now is omega alone.
    let before_a = set_correctives(&cache);
    let omega = set(&cache, &mut lens, 0, 0).fast_metadata_bytes;
    assert!(omega > 0, "{label}: no per-object reservation to check eff against");
    touch(&cache, &lens, 0, &mut hits);

    let run = Run {
        label: &label,
        design,
        fast: w.fast,
        omega,
        ghost_max: std::cell::Cell::new(0),
        p0,
        m0,
        gate: design.gate_state(),
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
        touch(&cache, &lens, key, &mut hits);
    }
    if w.touch_and_readmit {
        let evicted: Vec<u64> = (0..w.keys).filter(|k| cache.tier_of(k).is_none()).collect();
        for key in evicted.into_iter().step_by(2) {
            set(&cache, &mut lens, key, 0);
            touch(&cache, &lens, key, &mut hits);
        }
    }
    check(&cache, &lens, &run, "A (sets)", hits);
    assert_no_set_correctives(&cache, &label, "A (sets)", before_a);

    // B. A burst of repeated gets on a third of the keys, most of them slow:
    //    the designs that promote on a hit promote (and demote to make room),
    //    CLOCK sets reference bits, FIFO does nothing. Copies are in flight
    //    while it runs.
    let hot: Vec<u64> = (0..w.keys).step_by(3).collect();
    get_burst(&cache, &hot, 4, &mut hits);
    check(&cache, &lens, &run, "B (gets)", hits);

    // C. More new keys: every set now evicts, and CLOCK's hand recycles the
    //    keys B referenced -- a promotion for the slow ones.
    let before = set_correctives(&cache);
    for key in w.keys..w.keys + w.evicting {
        set(&cache, &mut lens, key, 0);
        touch(&cache, &lens, key, &mut hits);
    }
    check(&cache, &lens, &run, "C (evicting sets)", hits);
    assert_no_set_correctives(&cache, &label, "C (evicting sets)", before);

    // D. Overwrites at new sizes, one at a time: each old value is superseded
    //    and must be refunded whichever tier it was in.
    let before = set_correctives(&cache);
    let live: Vec<u64> = lens.keys().copied().filter(|k| cache.tier_of(k).is_some()).collect();
    for key in live.iter().copied().step_by(4) {
        set(&cache, &mut lens, key, 1);
    }
    check(&cache, &lens, &run, "D (overwrites)", hits);
    assert_no_set_correctives(&cache, &label, "D (overwrites)", before);

    // E. Deletes, one at a time. F. Re-sets of deleted keys, each after the
    //    delete has quiesced.
    let before = set_correctives(&cache);
    let live: Vec<u64> = lens.keys().copied().filter(|k| cache.tier_of(k).is_some()).collect();
    let deleted: Vec<u64> = live.iter().copied().skip(1).step_by(5).collect();
    for key in &deleted {
        cache.del(key).expect("del of a live key");
        quiesce(&cache, &lens, &format!("{label} del({key})"));
    }
    check(&cache, &lens, &run, "E (dels)", hits);

    for key in deleted.iter().copied().step_by(2) {
        set(&cache, &mut lens, key, 2);
    }
    assert_no_set_correctives(&cache, &label, "E-F (dels, re-sets)", before);

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
    // Each order in a child process of its own. FIFO has no promotion rule.
    let orders = [
        ("LRU", PaperPolicy::LruCompactHybrid, true),
        ("FIFO", PaperPolicy::FifoCompactHybrid, false),
        ("CLOCK", PaperPolicy::ClockCompactHybrid, true),
        ("LFU", PaperPolicy::LfuCompactHybrid, true),
    ];

    common::each_alone(
        module_path!(),
        "phys_fast_equals_the_stacks_fast_used_at_quiescence_and_returns_to_zero",
        orders,
        |(name, policy, promotes)| {
            let s = run(Design::Policy(policy), FULL);

            // The workload did what the identity is meant to survive: every
            // design demoted and evicted, and every design with a promotion
            // rule promoted, so P was charged and refunded on consumer
            // threads as well as client ones.
            assert!(s.demotions > 0, "{name}: the workload never demoted");
            assert!(s.evictions > 0, "{name}: the workload never evicted");
            assert!(s.slow_hits > 0, "{name}: no hit was served from the slow tier");
            assert!(s.fast_hits > 0, "{name}: no hit was served from the fast tier");

            if promotes {
                assert!(s.promotions > 0, "{name}: the workload never promoted");
            }
        },
    );
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
            common::alone(module_path!(), stringify!($name), || other_design($design));
        }
    )*};
}

other_designs! {
    lru_sized => Design::Sized;
    lru_lfu => Design::Policy(PaperPolicy::LruLfuCompactHybrid(2));
    two_q => Design::Policy(PaperPolicy::TwoQCompactHybrid(0.25));
    // k_in 0.1, not 0.25: at 0.25 of this cache the admission FIFO's
    // carve-out is the whole 16 KiB fast tier, and phase A ends with nothing
    // fast at all (P == fast_used == 0 -- vacuous, not wrong).
    two_q_fast_admission_reprieve =>
        Design::Policy(PaperPolicy::TwoQFastAdmissionReprieveCompactHybrid(0.1));
    two_q_full_fast_admission =>
        Design::Policy(PaperPolicy::TwoQFullFastAdmissionCompactHybrid(0.25, 0.5));
    two_q_ghost => Design::Policy(PaperPolicy::TwoQGhostCompactHybrid(0.5));
    s3_fifo => Design::Policy(PaperPolicy::S3FifoCompactHybrid(0.1));
    /// Ignored until S3: `S3FifoFaithfulCore::evict_small` (slow small queue)
    /// promoted a key seen twice to main's FAST front and, if main was then
    /// full, returned through `evict_main` BEFORE pushing its `(key, Fast)`
    /// -- counted fast, bytes left in CXL, nothing queued (at `set(51)`:
    /// stack fast objects 8 vs 7 physically fast, fast_used 10,560 vs P
    /// 8,512). The push now comes first; the audit would report the key
    /// lagging.
    s3_fifo_faithful => Design::Policy(PaperPolicy::S3FifoFaithfulCompactHybrid(0.1));
    s3_fifo_faithful_fast_admission =>
        Design::Policy(PaperPolicy::S3FifoFaithfulFastAdmissionCompactHybrid(0.1));
    /// The same `evict_small` fix as `s3_fifo_faithful` (the reprieve variant
    /// shares the core and its slow small queue; it failed at `set(50)`,
    /// fast_used 10,048 vs P 9,280).
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

/// T7 (backpressure plan S3): LFU's sets made in BURSTS -- no wait between
/// them, the client far ahead of the worker -- the shape of S2's burst
/// diagnostic. The stack latches part-way through the first burst while the
/// client still reads the latch mirror open, so values the stack admits slow
/// are built in DRAM; before S3 they stayed there ("147 physically fast", the
/// stack counting 111 of them slow, nothing queued). The reconcile queues each
/// one's demotion from its `Set`'s built tier.
///
/// After each burst the audit is taken AT ONCE, with no quiescence wait: its
/// event queues behind every set of the burst, and the worker lands all it
/// decided for them before it walks, so it must find nothing stranded,
/// lagging or untracked. Then the full quiescent check -- the identity, and
/// the audit again. Phases: A, the burst of new keys that latches (the cache
/// fills and evicts part-way, as at FULL); B, a burst of gets over every key,
/// twice (promotions, and hits served from both tiers); C, a burst of more
/// new keys, every one evicting; D, gets on every third key, twice, and right
/// behind them each of those keys deleted and set again -- the client far
/// ahead of the worker, so the gets' promotions are decided for the OLD
/// values and land on the fresh ones, which a latched stack re-admits slow:
/// the new-key rule's case (review M1 (iii)), in a real cache. It is timing,
/// not a pin -- the lib's `reconcile_tests` pin each case. D is audited at
/// once only: in the split LFU it can leave nothing fast (the keys the gets
/// promoted are the ones deleted, and their re-sets are admitted slow), and
/// the quiescent check requires something fast; E, gets over every key,
/// twice, refills the tier, and the full check follows it. No overwrite
/// burst: an overwrite racing a migration of its own key is pinned there too.
///
/// That the burst OUTRAN the mirror is asserted, not assumed: phase A must
/// leave corrective demotions behind -- `reconcile_set_to_slow` moves, the
/// worker's reconcile queueing them, and the cache's
/// `reconcile_applied_to_slow` counts them landing. In every build: the latch
/// is published with the `Set` that shuts it, and the burst's later keys were
/// built before the worker reached that `Set`.
#[test]
fn an_lfu_burst_strands_no_value() {
    common::alone(module_path!(), "an_lfu_burst_strands_no_value", || {
        burst(Design::Policy(PaperPolicy::LfuCompactHybrid), FULL);
    });
}

/// T7's body: `run`'s bookkeeping, with each phase's sets made back to back.
fn burst(design: Design, w: Workload) {
    let _one = one_cache_at_a_time();

    let label = format!("T7 {}", design.label());
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

    // The byte gate off (S5 B2, design 0.9): T7 needs its burst to outrun the
    // LFU latch mirror, and a waiting gate would hold the burst to the rate
    // demotions free room at, which can erase that precondition.
    let cache = design.build_with(w, GateMode::Off);
    let mut lens = BTreeMap::new();
    let mut hits = 0u64;

    // The first key alone, for omega, as `run` takes it.
    cache.set(0, &value(0, 0), None).expect("set");
    lens.insert(0, len_of(0, 0));
    let omega = quiesce(&cache, &lens, &format!("{label} set(0)")).0.fast_metadata_bytes;

    let run = Run {
        label: &label,
        design,
        fast: w.fast,
        omega,
        ghost_max: std::cell::Cell::new(0),
        p0,
        m0,
        gate: GateState::Off,
    };

    let audit_now = |cache: &Cache, phase: &str| {
        let a = cache.placement_audit().expect("a tiered cache answers the audit");
        eprintln!("{label} {phase}: audited at once: {a:?}");
        assert!(
            a.is_clean(),
            "{label} {phase}: {} values stranded in DRAM ({} B), {} lagging in CXL ({} B), {} \
             untracked ({} B), of {} live, once the worker landed all it decided",
            a.stranded, a.stranded_bytes, a.lagging, a.lagging_bytes, a.untracked,
            a.untracked_bytes, a.live,
        );
    };

    // A. Every other new key, back to back.
    let before = cache.hybrid_stats();
    for key in 1..w.keys {
        cache.set(key, &value(key, 0), None).expect("set");
        lens.insert(key, len_of(key, 0));
    }
    audit_now(&cache, "A (a burst of new keys)");
    check(&cache, &lens, &run, "A (a burst of new keys)", hits);
    let after = cache.hybrid_stats();
    assert!(after.slow_objects > 0, "{label}: the burst never filled the fast tier");
    eprintln!(
        "{label} A: reconcile set->slow +{}, corrective demotions landed +{}",
        after.reconcile_set_to_slow - before.reconcile_set_to_slow,
        after.reconcile_applied_to_slow - before.reconcile_applied_to_slow,
    );
    assert!(
        after.reconcile_applied_to_slow > before.reconcile_applied_to_slow,
        "{label}: no corrective demotion landed -- the burst never outran the latch mirror",
    );
    assert!(
        after.reconcile_set_to_slow > before.reconcile_set_to_slow,
        "{label}: the reconcile queued no demotion -- the burst never outran the latch mirror",
    );

    // B. Gets over every key, twice, back to back.
    for _ in 0..2 {
        for key in 0..w.keys {
            if cache.get(&key).is_ok() {
                hits += 1;
            }
        }
    }
    audit_now(&cache, "B (a burst of gets)");
    check(&cache, &lens, &run, "B (a burst of gets)", hits);

    // C. More new keys, every one evicting, back to back.
    for key in w.keys..w.keys + w.evicting {
        cache.set(key, &value(key, 0), None).expect("set");
        lens.insert(key, len_of(key, 0));
    }
    audit_now(&cache, "C (a burst of evicting sets)");
    check(&cache, &lens, &run, "C (a burst of evicting sets)", hits);

    // D. Gets on every third live key, twice, then each of them deleted and
    //    set again at once, all back to back.
    let live: Vec<u64> = lens.keys().copied().filter(|k| cache.tier_of(k).is_some()).collect();
    let chosen: Vec<u64> = live.iter().copied().step_by(3).collect();
    for _ in 0..2 {
        for key in &chosen {
            if cache.get(key).is_ok() {
                hits += 1;
            }
        }
    }
    for &key in &chosen {
        cache.del(&key).expect("del of a live key");
        cache.set(key, &value(key, 1), None).expect("set");
        lens.insert(key, len_of(key, 1));
    }
    audit_now(&cache, "D (gets, then deletes and re-sets)");

    // E. Gets over every key, twice, back to back.
    for _ in 0..2 {
        for key in 0..w.keys + w.evicting {
            if cache.get(&key).is_ok() {
                hits += 1;
            }
        }
    }
    audit_now(&cache, "E (a burst of gets)");
    check(&cache, &lens, &run, "E (a burst of gets)", hits);

    let stats = cache.hybrid_stats();
    eprintln!(
        "{label}: evictions={} fast_hits={} slow_hits={} reconcile set->fast/set->slow/get->fast/new-key \
         {}/{}/{}/{} (process totals); correctives landed to fast/slow {}/{}",
        stats.evictions, stats.fast_hits, stats.slow_hits,
        stats.reconcile_set_to_fast, stats.reconcile_set_to_slow, stats.reconcile_get_to_fast,
        stats.reconcile_set_new_key, stats.reconcile_applied_to_fast, stats.reconcile_applied_to_slow,
    );
    assert!(stats.evictions > 0, "{label}: the bursts never evicted");

    drop(cache);

    assert_eq!(
        phys::fast_bytes_signed(),
        p0,
        "{label}: P did not return to its pre-cache value -- a fast allocation was never refunded",
    );
    assert_eq!(phys::live_tiered_caches(), 0, "{label}: uncounted on drop");
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

    let flat = PaperCache::<u64, BufferDRAM>::new(1 << 20, &[PaperPolicy::LruCompact], PaperPolicy::LruCompact)
        .expect("a flat cache with fast values");
    assert_eq!(phys::live_flat_fast_caches(), 1, "counted on construction");
    assert_eq!(phys::live_tiered_caches(), 0, "a flat cache is not a tiered one");

    flat.set(1, &[7u8; 1_000], None).expect("set");
    assert_eq!(phys::fast_bytes_signed() - p0, one_value, "its value is in P");

    let pmem = PaperCache::<u64, BufferPMEM>::new(1 << 20, &[PaperPolicy::LruCompact], PaperPolicy::LruCompact)
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
