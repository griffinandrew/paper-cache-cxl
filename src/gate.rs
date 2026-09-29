/*
 * Copyright (c) Kia Shakiba
 *
 * This source code is licensed under the GNU AGPLv3 license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! The admission path of a tiered cache's `set` (fast-tier backpressure plan
//! S5): what is decided from a value's LENGTH, before anything is allocated
//! (commit B1) -- and, for a value to be built in the fast tier, the byte gate
//! that holds it to the tier's budget, waiting until demotions free room
//! (commit B2) -- and the figures those decisions read.
//!
//! # One figure: eff
//!
//! Every consumer of "how much of the fast tier is left for values" reads ONE
//! published figure, `eff = F - M_model` (saturating), where F is the whole
//! fast-tier budget and `M_model` the cache's DRAM metadata under the
//! configured [`MetadataModel`]: the MEASURED bytes of the cache's own
//! structures (`crate::meta`, S5a -- the default, the user's decision), or the
//! per-object model the stacks reserved until now (`L * omega`, plus the
//! ghost's DRAM in the designs that keep one). The policy worker publishes it
//! once per pass (`PolicyWorker::publish_gate`) and pushes the same M into the
//! stack (`PolicyStack::set_dram_metadata`), so the stacks' settles, the
//! structural check below and the metadata cap cannot disagree by more than
//! one pass. `PAPER_DISABLE_SHARED_OVERHEAD=1` -- the mechanics tests at toy
//! scales -- forces the per-object model with `omega = 0`, which is exactly
//! the reservation those tests ran under before: none, beyond a ghost's.
//!
//! # What `set` decides before it allocates (`decide`)
//!
//!   0. the size checks, from the length (`OverheadManager::base_size_for`):
//!      an oversize value is refused without being built;
//!   1. the METADATA CAP: a NEW key whose metadata would not fit the fast
//!      tier gets `CacheError::MetadataOverflow` (the default), or -- opted in,
//!      [`MetadataOverflow::EvictToFit`] -- waits in FIFO order in the metadata
//!      lane while the policy worker evicts the policy's own victims for it
//!      (`WorkerEvent::MakeRoom`). Only under [`META_NEAR`], so a set far from
//!      the ceiling pays one relaxed load for this step and the structural one;
//!   2. `hybrid_policy::admission_tier`, unchanged;
//!   3. STRUCTURAL SLOW PLACEMENT: a value larger than an EMPTY fast tier
//!      (`v > eff`; for the size-split design, its size class's segment) is
//!      built slow whatever step 2 said, and its `Set` says
//!      `Placement::Structural`, so every stack places it slow and never
//!      promotes it (the user's decision: no DRAM write, no demotion copy, and
//!      when metadata fills the tier -- eff = 0 -- no tiering at all).
//!
//! # The key ceiling
//!
//! A table keeps its capacity, so M does not fall when a key is evicted, and a
//! cap on M alone could not be met by evicting. The cap is a ceiling on the
//! object count instead, `K_max` (`key_ceiling`): refilling up to the most
//! objects the cache has held (`L_hw`, whose structures were already paid for)
//! is always allowed, and beyond it each new object is estimated at the
//! per-object constant omega until M would reach `F - floor`. Under the
//! per-object model the ceiling is exactly the plan's `(L + 1) * omega > F`.
//! With omega 0 (`PAPER_DISABLE_SHARED_OVERHEAD=1`) there is no ceiling.
//!
//! # The byte gate (commit B2)
//!
//!   4. A value to be built FAST is held to `P + M_model <= F + slack`, i.e.
//!      `P <= B`: `P` the bytes physically in the fast tier's value pool
//!      (`crate::phys`, process-global), `B = eff + slack` the CLOSE level. The
//!      policy worker publishes, with eff, the settle target `S =
//!      drain_target(eff)` -- where every design rests after a pass (B1) --
//!      and the NEAR level `N`, 1% of eff below `B` by default ([`bands`]).
//!
//!      FAST PATH: the gate word's `NEAR` and `CLOSED` bits clear and the value
//!      no larger than `B - N`: admitted on the one relaxed load that already
//!      decided steps 1 and 3. `NEAR` is `approx + E >= N` (`approx` the folded
//!      part of P, `E` its error bound), re-evaluated at every fold of P
//!      (`GateShared::on_fold`) and by the worker each pass: a clear `NEAR`
//!      means `P < N`.
//!
//!      EXACT PATH: `P` (17 loads) and `R`, the bytes admitted sets hold
//!      reserved until their values are built: admitted iff `P + R + v <= B`,
//!      reserving `v` unless `P + v <= N`; a value larger than `B - S`
//!      (OVERSIZE) only on a settled tier, `P <= S` with nothing reserved.
//!
//!      Otherwise the set WAITS, in FIFO order, in the BYTES lane: `CLOSED`
//!      while the lane holds anyone, so a newcomer queues behind. It is woken by
//!      a migration consumer's landed demotion, the worker's pass, a released
//!      reservation, a wipe; at every wake it re-decides its tier and the
//!      structural check (a value that no longer fits even an empty tier leaves
//!      the lane as a structural set) and then the bytes.
//!
//! A NO-PROGRESS WATCHDOG (`GateConfig::stall_window`, 2 s): the lane's head
//! waits for as long as something is freed -- bytes refunded anywhere
//! (`phys::freed`, counted while a gate watches), a demotion landed, a waiter
//! admitted, the close level raised -- and the gate is STALLED only after a
//! whole window with none of these AND with the worker having completed passes
//! (each with its resettle) since the window began, so a late worker is not a
//! stall. A stalled set acts per [`OnStall`]: `CacheError::FastTierStalled` (the
//! default); built slow (`Divert`: `Placement::Diverted` -- placed by its policy,
//! healed on its first slow-served hit, never corrected toward fast at its
//! `Set`); or admitted over the budget (`AdmitOver`). While a stall is
//! unresolved every waiter and newcomer acts after a short probe with nothing
//! freed; the worker ends the stall at the first byte freed.
//!
//! The byte gate is DISABLED -- fast sets are admitted ungated, and
//! `HybridStats::gate_state` says why ([`GateState`]) -- under `GateMode::Off`;
//! while the cache is not P's only user (another tiered cache, or a flat cache
//! with fast values, is alive: `phys::sole_fast_user`, re-read at once when a
//! live count moves); for the designs whose settles do not bound their DRAM
//! (the lazy-copy LRU, plan P6; the faithful fast-admission pair, whose small
//! queue is not clamped to the tier, design Q7); for bands that would put the
//! settle target at or above the near level; and with no stack. A dead policy
//! worker fails a waiting set with `CacheError::Internal`.

use std::{
	collections::VecDeque,
	sync::{
		Arc,
		atomic::{AtomicBool, AtomicI64, AtomicU64, AtomicU8, AtomicUsize, Ordering},
	},
	thread::Thread,
	time::{Duration, Instant},
};

use crate::{
	CacheSize,
	HashedKey,
	Tier,
	error::CacheError,
	object::ObjectSize,
	policy::PaperPolicy,
	phys,
	status::AtomicStatus,
	worker::{Placement, drain_target},
};

/// Which figure stands for the cache's DRAM metadata in `eff = F - M_model`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum MetadataModel {
	/// The bytes the cache's own DRAM structures hold, counted from the
	/// structures (`crate::meta`, S5a) and published by the policy worker every
	/// pass: the object map's, the policy stack's and one value header per
	/// live object. The default.
	#[default]
	Measured,

	/// The per-object model the stacks reserved before S5: `omega` per
	/// tracked key (`get_hybrid_dram_shared_overhead`), plus the ghost's DRAM
	/// in the seven designs that keep one -- the stack's own
	/// `dram_reserved_bytes`. A fallback and a sanity check; the toy-scale
	/// tests pin it.
	PerObject,
}

/// What a NEW key whose metadata would not fit gets.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum MetadataOverflow {
	/// `CacheError::MetadataOverflow`, at once: nothing is allocated or sent.
	#[default]
	Error,

	/// The set waits, in FIFO order behind any other new key waiting, while
	/// the policy worker evicts the policy's own victims to make room
	/// (`WorkerEvent::MakeRoom`), and then proceeds. `MetadataOverflow` if the
	/// worker has nothing to evict, or makes no progress for `stall_window`.
	EvictToFit,
}

/// Whether the byte gate runs (S5, commit B2).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum GateMode {
	/// No byte gate: a fast set is admitted without looking at P. The metadata
	/// cap and structural placement still apply.
	Off,

	/// A fast set that would take P over the budget waits, FIFO, until
	/// demotions free room. The default.
	#[default]
	Block,
}

impl GateMode {
	/// The mode `GateConfig::default()` carries: `Block` -- except in this
	/// crate's own unit tests, where `Off`. P is process-global, and there the
	/// tests run in parallel and build values outside any cache, so a real
	/// cache that found itself P's only registered user would wait on other
	/// tests' bytes; the gate's tests opt in, each alone in a child process
	/// (the semantics review's fix).
	const DEFAULT: GateMode = if cfg!(test) { GateMode::Off } else { GateMode::Block };
}

/// What a set that waited `stall_window` with nothing freed does (S5, B2).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum OnStall {
	/// `CacheError::FastTierStalled`: a stuck state, not load, reported
	/// loudly. The default.
	#[default]
	Error,

	/// Built in the slow tier (`Placement::Diverted`) and placed by the
	/// design's policy -- typically fast, so the key lags in CXL until its
	/// first slow-served hit heals it. Opt-in.
	Divert,

	/// Admitted to the fast tier over the budget (its bytes reserved like any
	/// admission's). Opt-in.
	AdmitOver,
}

/// The byte gate's state: running, or why not (`HybridStats::gate_state`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[repr(u8)]
pub enum GateState {
	/// Running: fast sets are held to the budget.
	Enabled,

	/// `GateMode::Off`.
	#[default]
	Off,

	/// P is process-global and this cache is not its only user: another
	/// tiered cache, or a flat cache with fast values, is alive.
	NotSole,

	/// The design's settles do not bound its DRAM: the lazy-copy LRU (plan
	/// P6) and the faithful S3-FIFO fast-admission pair (design 0.6, Q7).
	Ungated,

	/// `drain_target::ratio() + near_frac >= 1`: the settle target would not
	/// be below the near level (a plain constructor under an environment-chosen
	/// drain target; `GateConfig::validate` refuses it elsewhere).
	Bands,

	/// No policy stack to settle anything (a flat policy switch's
	/// reconstruction).
	NoStack,
}

impl GateState {
	fn from_u8(state: u8) -> GateState {
		match state {
			0 => GateState::Enabled,
			2 => GateState::NotSole,
			3 => GateState::Ungated,
			4 => GateState::Bands,
			5 => GateState::NoStack,
			_ => GateState::Off,
		}
	}
}

/// A tiered cache's admission configuration. Built as `GateConfig::default()`
/// and adjusted field by field (the struct is `non_exhaustive`), then passed to
/// a `*_with_gate` constructor or `PaperCache::set_gate_config`.
#[derive(Clone, Copy, Debug, PartialEq)]
#[non_exhaustive]
pub struct GateConfig {
	/// `Measured` (default) or `PerObject`. Forced to `PerObject` by
	/// `PAPER_DISABLE_SHARED_OVERHEAD=1`, whatever this says.
	pub metadata_model: MetadataModel,

	/// Bytes of the fast tier kept for values: the metadata cap is
	/// `F - metadata_floor`. 0 by default -- metadata may fill the tier, and
	/// then every value is structural (provisional, design Q3).
	pub metadata_floor: CacheSize,

	/// `Error` (default) or `EvictToFit`.
	pub on_metadata_overflow: MetadataOverflow,

	/// How long a waiting set waits with NOTHING freed before it acts: the
	/// byte gate's no-progress watchdog (`on_stall`), and an `EvictToFit` set
	/// waiting for the policy worker to answer while the worker makes no
	/// progress at all (it counts every event it handles, so a set queued
	/// behind a long backlog keeps waiting while the backlog drains). 2 s
	/// (provisional, design Q1). `Duration::ZERO`: never wait -- a set that
	/// would wait acts at once.
	pub stall_window: Duration,

	/// How often a waiting set re-checks when nothing wakes it. 200 us.
	pub poll_interval: Duration,

	/// `Block` (the default) or `Off`: whether the byte gate runs.
	pub mode: GateMode,

	/// What a set does when the watchdog fires: `Error` (the default),
	/// `Divert` or `AdmitOver`.
	pub on_stall: OnStall,

	/// Bytes the fast tier may hold beyond its budget: the close level is
	/// `B = eff + slack`. 0 (design Q11).
	pub slack: CacheSize,

	/// The near band, as a fraction of eff: `N = B - near_frac x eff`, above
	/// which a fast set takes the exact path. 0.01 (design Q11). Must be in
	/// `[0, 1)`, and `drain_target::ratio() + near_frac < 1` under `Block`.
	pub near_frac: f64,

	/// Concurrent setters and a typical value size, which widen the near band
	/// to `concurrency_hint x value_hint` when that is wider: what a server
	/// with many connections sets (S9). 0.
	pub concurrency_hint: u32,
	pub value_hint: CacheSize,
}

impl Default for GateConfig {
	fn default() -> Self {
		GateConfig {
			metadata_model: MetadataModel::Measured,
			metadata_floor: 0,
			on_metadata_overflow: MetadataOverflow::Error,
			stall_window: Duration::from_secs(2),
			poll_interval: Duration::from_micros(200),
			mode: GateMode::DEFAULT,
			on_stall: OnStall::Error,
			slack: 0,
			near_frac: 0.01,
			concurrency_hint: 0,
			value_hint: 0,
		}
	}
}

impl GateConfig {
	/// `CacheError::InvalidGateConfig` for a configuration no set could run
	/// under: a zero poll interval (a waiting set would spin), a near band
	/// outside `[0, 1)`, or bands that cannot hold (`bands_hold`).
	pub fn validate(&self) -> Result<(), CacheError> {
		if self.poll_interval.is_zero()
			|| !self.near_frac.is_finite()
			|| !(0.0..1.0).contains(&self.near_frac)
			|| !self.bands_hold()
		{
			return Err(CacheError::InvalidGateConfig);
		}

		Ok(())
	}

	/// The construction check (design 3.9.2, the plan's "refuses a gate that
	/// is on when tau x eff >= N"), in its parameter form: under `Block`,
	/// `drain_target::ratio() + near_frac < 1`, which puts the settle target S
	/// below the near level N at every eff > 0 (with no concurrency hint; with
	/// one, `bands` clamps N above S). `Off` needs no bands.
	pub fn bands_hold(&self) -> bool {
		self.mode == GateMode::Off || drain_target::ratio() + self.near_frac < 1.0
	}
}

/// The gate word's flag: the object count is within `NEAR_KEYS` of the key
/// ceiling, so a set of a NEW key checks the ceiling. Set by the policy worker
/// at its publication and by a client whose insert brings the count near it;
/// cleared only by the worker.
pub(crate) const META_NEAR: u64 = 1;

/// The gate word's flag (B2): P may be at or above the near level -- `approx
/// + E >= N` -- so a fast set takes the exact path. Written by the fold hook
/// and the worker's publication; never set while the byte gate is disabled.
pub(crate) const NEAR: u64 = 1 << 1;

/// The gate word's flag (B2): the bytes lane holds a waiter, so a fast set
/// that is not its head queues behind it. Set by the lane's first waiter and
/// cleared by its last departure, both under the lane's lock.
pub(crate) const CLOSED: u64 = 1 << 2;

/// While a stall is unresolved, how long a waiter or a newcomer waits with
/// nothing freed before it acts (at most `stall_window`): a stuck state does
/// not cost every set a whole window, and a stall that has ended is found
/// out (the liveness review's probe).
const STALL_PROBE: Duration = Duration::from_millis(50);

/// How long a waiter behind the head parks between checks. It is woken when
/// it becomes the head, when the gate stalls or is disabled, on a wipe and
/// when the worker goes; this is only the safety net.
const NON_HEAD_PARK: Duration = Duration::from_millis(100);

/// The worker's gate passes that must complete after a no-progress window
/// began before the window can end in a stall: the one that may have been
/// running when it began, and one that began after it, with its resettle (the
/// liveness review: a late worker is not a stall).
const STALL_PASSES: u64 = 2;

/// Buckets of the wait histogram: `2^i` microseconds and up, the last open.
pub(crate) const WAIT_BUCKETS: usize = 16;

/// The gate word's low byte holds the flags; the rest is eff in 4 KiB pages,
/// rounded down.
const FLAG_BITS: u32 = 8;
const PAGE_SHIFT: u32 = 12;

/// How far below the key ceiling `META_NEAR` is set (provisional).
pub(crate) const NEAR_KEYS: u64 = 64;

/// How many victims one `MakeRoom` evicts at most.
pub(crate) const MAKE_ROOM_BATCH: u64 = 64;

/// The published model, as the status holds it.
const MODEL_MEASURED: u8 = 0;
const MODEL_PER_OBJECT: u8 = 1;

/// The part of a gate the fold hook reads (`phys::GATE_HOOK`): the gate word
/// and the near level. In an `Arc` of its own, so a fold running on any thread
/// never holds a cache's status alive.
pub(crate) struct GateShared {
	/// The word a set reads first: `META_NEAR`, `NEAR` and `CLOSED` in the low
	/// byte; above it, in 4 KiB pages rounded down, eff -- the minimum of the
	/// two class figures for the size-split design -- or, while the byte gate
	/// runs, the fast path's bound `B - N` when that is smaller. A set whose
	/// low byte is clear and whose value is at most the page figure is neither
	/// capped nor structural nor gated: one relaxed load.
	word: AtomicU64,

	/// N, as last published; `u64::MAX` while the byte gate is not enabled,
	/// which no `approx` reaches.
	band_n: AtomicU64,
}

impl GateShared {
	fn new() -> Self {
		GateShared {
			// Until the policy worker's first publication (at its construction,
			// before the cache is returned): nothing near, and room for anything.
			word: AtomicU64::new(u64::MAX << FLAG_BITS),
			band_n: AtomicU64::new(u64::MAX),
		}
	}

	/// `NEAR := approx + E >= N` (design 3.9.3), written only when it changes:
	/// the fold hook, with the `approx` its fold left, and the worker's pass. A
	/// CLEAR is checked again against `approx` read after it: a fold on another
	/// thread that raised `approx` after this thread's reading may have set the
	/// flag first, and this clear must not erase it (the liveness review).
	pub(crate) fn on_fold(&self, approx: i64) {
		let n = self.band_n.load(Ordering::Relaxed);
		let near = |approx: i64| n != u64::MAX && approx.saturating_add(phys::FOLD_ERROR) >= n.min(i64::MAX as u64) as i64;

		let now = near(approx);

		if now == (self.word.load(Ordering::Relaxed) & NEAR != 0) {
			return;
		}

		if now {
			self.word.fetch_or(NEAR, Ordering::AcqRel);
			return;
		}

		self.word.fetch_and(!NEAR, Ordering::AcqRel);

		if near(phys::fast_bytes_approx()) {
			self.word.fetch_or(NEAR, Ordering::AcqRel);
		}
	}
}

/// A counter incremented on a per-set path, sharded like P (by the thread's
/// arena slot) so the sets that increment it do not share one line.
struct Sharded([Padded<AtomicU64>; phys::SHARDS]);

#[repr(align(64))]
struct Padded<T>(T);

impl Sharded {
	fn new() -> Self {
		Sharded([const { Padded(AtomicU64::new(0)) }; phys::SHARDS])
	}

	fn incr(&self) {
		self.0[crate::numa_alloc::arena_slot() as usize % phys::SHARDS].0.fetch_add(1, Ordering::Relaxed);
	}

	fn sum(&self) -> u64 {
		self.0.iter().map(|shard| shard.0.load(Ordering::Relaxed)).fold(0, u64::wrapping_add)
	}

	fn reset(&self) {
		for shard in &self.0 {
			shard.0.store(0, Ordering::Relaxed);
		}
	}
}

/// A tiered cache's admission state, in its `AtomicStatus`: the configuration,
/// the figures the policy worker publishes for the client's decisions, the
/// metadata lane, the byte gate's state, levels, reservations and bytes lane,
/// the worker's idle and liveness bits, and the counters.
///
/// What a wipe resets (`reset_counters`): the event counters only. The
/// configuration, the lanes and their waiters, the reservations, a stall, the
/// published figures and the worker's bits are live state and survive it --
/// an admitted set's reservation is released by its own permit whatever
/// happened in between, so it can never be released twice or into a counter
/// that was reset under it (the liveness review).
pub(crate) struct Gate {
	config: parking_lot::RwLock<GateConfig>,

	/// `PAPER_DISABLE_SHARED_OVERHEAD=1` when the cache was built: the
	/// per-object model, whatever the configuration says.
	forced_per_object: AtomicBool,

	/// The word and the near level, shared with the fold hook.
	shared: Arc<GateShared>,

	/// eff, exact, and the size-split design's two class figures.
	eff: AtomicU64,
	eff_small: AtomicU64,
	eff_large: AtomicU64,

	/// `M_model` and the key ceiling, as last published.
	m_model: AtomicU64,
	k_max: AtomicU64,

	/// The model the last publication used (`MODEL_*`).
	model: AtomicU8,

	/// The policy worker is about to park on its long poll: the first set
	/// after that wakes it (`PolicyWorker::delay_event_loop`, and the fence
	/// pair in `PaperCache::commit`).
	pub(crate) worker_idle: AtomicBool,

	/// The policy worker's thread has exited, by return or by unwinding
	/// (`WorkerGoneGuard`): a waiter returns `CacheError::Internal`.
	worker_gone: AtomicBool,

	/// Events the policy worker has handled, stored every 64 events and at
	/// every pass end: a waiter's evidence that the worker is working through
	/// a backlog ahead of its `MakeRoom` rather than stuck.
	worker_progress: AtomicU64,

	/// The metadata lane: `EvictToFit` new keys, FIFO.
	pub(crate) meta_lane: Lane,

	/// The policy worker's last `MakeRoom` answer: `(request << 8) | evicted`.
	room_outcome: AtomicU64,

	/// The next `MakeRoom` request number: every request is answered under
	/// its own, so a head asking again reads the new answer, not the last.
	room_request: AtomicU64,

	metadata_overflows: AtomicU64,
	make_room_requests: AtomicU64,
	make_room_evictions: AtomicU64,
	make_room_failures: AtomicU64,
	structural_slow_sets: AtomicU64,
	structural_placements: AtomicU64,
	idle_kicks: AtomicU64,
	metadata_model_divergence: AtomicU64,

	/// B2: the byte gate's state (`GateState`), as the worker last evaluated
	/// it -- or `NotSole`, from a set that found a live count moved (`enabled`).
	state: AtomicU8,

	/// Whether this gate's `GateShared` is the fold hook (the worker's to
	/// write).
	hooked: AtomicBool,

	/// `phys::gate_epoch` when the live counts were last read for the state.
	epoch_seen: AtomicU64,

	/// The settle target S and the close level B, as last published (N is in
	/// `shared`); 0 while the gate is not enabled.
	band_s: AtomicU64,
	band_b: AtomicU64,

	/// Bytes admitted sets hold until their values are built (`Reservation`).
	reserved: AtomicU64,

	/// The bytes lane: fast sets waiting for room, FIFO.
	pub(crate) bytes_lane: Lane,

	/// Moved by a landed demotion while the bytes lane holds a waiter, and by
	/// every admission out of it: progress, for the watchdog.
	lane_progress: AtomicU64,

	/// The worker's gate passes (`worker_pass`): a no-progress window ends in
	/// a stall only after passes, not only after time.
	passes: AtomicU64,

	/// The watchdog fired and the worker has seen nothing freed since; with
	/// FREED, P and B when it fired.
	stalled: AtomicBool,
	stall_freed: AtomicU64,
	stall_p: AtomicI64,
	stall_b: AtomicU64,

	/// A near kick is allowed: re-armed at every worker pass, so at most one
	/// set per pass takes the kick's lock.
	near_kick_armed: AtomicBool,

	gate_disabled_sets: AtomicU64,
	gate_slow_paths: Sharded,
	gate_waits: AtomicU64,
	gate_wait_ns_total: AtomicU64,
	gate_wait_ns_max: AtomicU64,
	gate_wait_hist: [AtomicU64; WAIT_BUCKETS],
	gate_stalls: AtomicU64,
	gate_stall_errors: AtomicU64,
	divert_sets: AtomicU64,
	divert_bytes: AtomicU64,
	admit_over_sets: AtomicU64,
	admit_over_bytes: AtomicU64,
	oversize_admits: AtomicU64,
	near_kicks: AtomicU64,
	max_waiters: AtomicU64,

	/// Test builds: the policy worker panics at its next `MakeRoom`, for the
	/// dead-worker test.
	#[cfg(test)]
	pub(crate) test_panic_on_make_room: AtomicBool,

	/// Test builds: the policy worker panics at its next gate pass, for the
	/// byte gate's dead-worker test.
	#[cfg(test)]
	pub(crate) test_panic_on_pass: AtomicBool,

	/// Test builds: the bytes lane's admissions, each by the order its waiter
	/// joined the lane (`LaneSlot::id`), for the FIFO test.
	#[cfg(test)]
	pub(crate) test_admissions: parking_lot::Mutex<Vec<u64>>,

	/// Test builds: the idle spells the policy worker has begun -- each time
	/// it set its idle bit before a long park, parked or not -- for the
	/// one-kick-per-idle-spell test: a set clears the bit, and kicks, at most
	/// once per spell.
	#[cfg(test)]
	pub(crate) test_idle_spells: AtomicU64,
}

impl Default for Gate {
	fn default() -> Self {
		Gate {
			config: parking_lot::RwLock::new(GateConfig::default()),
			forced_per_object: AtomicBool::new(false),
			shared: Arc::new(GateShared::new()),
			eff: AtomicU64::new(u64::MAX),
			eff_small: AtomicU64::new(u64::MAX),
			eff_large: AtomicU64::new(u64::MAX),
			m_model: AtomicU64::new(0),
			k_max: AtomicU64::new(u64::MAX),
			model: AtomicU8::new(MODEL_MEASURED),
			worker_idle: AtomicBool::new(false),
			worker_gone: AtomicBool::new(false),
			worker_progress: AtomicU64::new(0),
			meta_lane: Lane::default(),
			room_outcome: AtomicU64::new(u64::MAX),
			room_request: AtomicU64::new(0),
			metadata_overflows: AtomicU64::new(0),
			make_room_requests: AtomicU64::new(0),
			make_room_evictions: AtomicU64::new(0),
			make_room_failures: AtomicU64::new(0),
			structural_slow_sets: AtomicU64::new(0),
			structural_placements: AtomicU64::new(0),
			idle_kicks: AtomicU64::new(0),
			metadata_model_divergence: AtomicU64::new(0),
			state: AtomicU8::new(GateState::Off as u8),
			hooked: AtomicBool::new(false),
			epoch_seen: AtomicU64::new(u64::MAX),
			band_s: AtomicU64::new(0),
			band_b: AtomicU64::new(0),
			reserved: AtomicU64::new(0),
			bytes_lane: Lane::default(),
			lane_progress: AtomicU64::new(0),
			passes: AtomicU64::new(0),
			stalled: AtomicBool::new(false),
			stall_freed: AtomicU64::new(0),
			stall_p: AtomicI64::new(0),
			stall_b: AtomicU64::new(0),
			near_kick_armed: AtomicBool::new(true),
			gate_disabled_sets: AtomicU64::new(0),
			gate_slow_paths: Sharded::new(),
			gate_waits: AtomicU64::new(0),
			gate_wait_ns_total: AtomicU64::new(0),
			gate_wait_ns_max: AtomicU64::new(0),
			gate_wait_hist: [const { AtomicU64::new(0) }; WAIT_BUCKETS],
			gate_stalls: AtomicU64::new(0),
			gate_stall_errors: AtomicU64::new(0),
			divert_sets: AtomicU64::new(0),
			divert_bytes: AtomicU64::new(0),
			admit_over_sets: AtomicU64::new(0),
			admit_over_bytes: AtomicU64::new(0),
			oversize_admits: AtomicU64::new(0),
			near_kicks: AtomicU64::new(0),
			max_waiters: AtomicU64::new(0),
			#[cfg(test)]
			test_panic_on_make_room: AtomicBool::new(false),
			#[cfg(test)]
			test_panic_on_pass: AtomicBool::new(false),
			#[cfg(test)]
			test_admissions: parking_lot::Mutex::new(Vec::new()),
			#[cfg(test)]
			test_idle_spells: AtomicU64::new(0),
		}
	}
}

/// What one publication of the policy worker says (`Gate::publish`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Published {
	pub(crate) model: MetadataModel,
	pub(crate) m_model: CacheSize,
	pub(crate) eff: CacheSize,
	/// The size-split design's class figures; `(eff, eff)` for every other.
	pub(crate) eff_small: CacheSize,
	pub(crate) eff_large: CacheSize,
	pub(crate) k_max: u64,
	/// B2: the byte gate's levels while it is enabled, `None` otherwise.
	pub(crate) bands: Option<Bands>,
}

/// The byte gate's levels for one eff (design 3.9.2; `bands`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct Bands {
	/// The settle target, `drain_target(eff)`: where every design rests after
	/// a pass.
	pub(crate) s: CacheSize,
	/// The near level: at or above it (`NEAR`), a fast set takes the exact path.
	pub(crate) n: CacheSize,
	/// The close level, `eff + slack`: `P <= B` is the budget.
	pub(crate) b: CacheSize,
}

/// The levels for `eff` under `config`: `S = drain_target(eff)`, `B = eff +
/// slack`, `N = B - max(near_frac x eff, concurrency_hint x value_hint)`
/// clamped into `[S + 1, B]` (`B` when `B <= S`, eff 0 among them).
pub(crate) fn bands(eff: CacheSize, config: &GateConfig) -> Bands {
	let s = drain_target::bytes(eff);
	let b = eff.saturating_add(config.slack);
	let near = ((config.near_frac * eff as f64) as CacheSize)
		.max((config.concurrency_hint as CacheSize).saturating_mul(config.value_hint));

	let n = match b > s {
		true => b.saturating_sub(near).clamp(s + 1, b),
		false => b,
	};

	Bands { s, n, b }
}

/// The gate's counters and published figures, for `HybridStats`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct GateStats {
	pub(crate) model: MetadataModel,
	pub(crate) m_model: u64,
	pub(crate) eff: u64,
	pub(crate) k_max: u64,
	pub(crate) metadata_overflows: u64,
	pub(crate) make_room_requests: u64,
	pub(crate) make_room_evictions: u64,
	pub(crate) make_room_failures: u64,
	pub(crate) structural_slow_sets: u64,
	pub(crate) structural_placements: u64,
	pub(crate) idle_kicks: u64,
	pub(crate) metadata_model_divergence: u64,
	pub(crate) state: GateState,
	pub(crate) gate_disabled_sets: u64,
	pub(crate) gate_slow_paths: u64,
	pub(crate) gate_waits: u64,
	pub(crate) gate_wait_ns_total: u64,
	pub(crate) gate_wait_ns_max: u64,
	pub(crate) gate_wait_hist: [u64; WAIT_BUCKETS],
	pub(crate) gate_stalls: u64,
	pub(crate) gate_stall_errors: u64,
	pub(crate) divert_sets: u64,
	pub(crate) divert_bytes: u64,
	pub(crate) admit_over_sets: u64,
	pub(crate) admit_over_bytes: u64,
	pub(crate) oversize_admits: u64,
	pub(crate) near_kicks: u64,
	pub(crate) max_waiters: u64,
	pub(crate) waiters: u64,
	pub(crate) reserved: u64,
	pub(crate) bands: Bands,
}

impl Gate {
	pub(crate) fn config(&self) -> GateConfig {
		*self.config.read()
	}

	pub(crate) fn set_config(&self, config: GateConfig) {
		*self.config.write() = config;
	}

	/// `PAPER_DISABLE_SHARED_OVERHEAD=1`: the per-object model whatever the
	/// configuration says. Recorded once, by the constructor.
	pub(crate) fn force_per_object(&self) {
		self.forced_per_object.store(true, Ordering::Relaxed);
	}

	/// The model the policy worker publishes under.
	pub(crate) fn model(&self) -> MetadataModel {
		match self.forced_per_object.load(Ordering::Relaxed) {
			true => MetadataModel::PerObject,
			false => self.config.read().metadata_model,
		}
	}

	/// eff as last published.
	pub(crate) fn eff(&self) -> CacheSize {
		self.eff.load(Ordering::Relaxed)
	}

	/// The eff a value of base size `base` is compared with for the structural
	/// check: its size class's under the size-split design (the class is
	/// chosen by base size against the threshold, as the stack classifies),
	/// the whole figure otherwise.
	pub(crate) fn eff_for(&self, status: &AtomicStatus, base: ObjectSize) -> CacheSize {
		match status.policy() {
			PaperPolicy::LruSizedCompactHybrid => {
				match (base as CacheSize) < status.hybrid_size_threshold() {
					true => self.eff_small.load(Ordering::Relaxed),
					false => self.eff_large.load(Ordering::Relaxed),
				}
			},

			_ => self.eff(),
		}
	}

	pub(crate) fn k_max(&self) -> u64 {
		self.k_max.load(Ordering::Relaxed)
	}

	pub(crate) fn m_model(&self) -> CacheSize {
		self.m_model.load(Ordering::Relaxed)
	}

	/// The policy worker's publication. The figures first, then the word, with
	/// `META_NEAR` decided from the object count `live` it read; a clear that
	/// raced a client's set is re-checked against the count once more, so a
	/// set that brought the count near the ceiling is never left unflagged.
	///
	/// B2: with the byte gate's levels (`bands`, `None` while it is disabled):
	/// `NEAR` from `approx` (`approx + E >= N`, a clear re-checked like
	/// META_NEAR's), and the word's page figure the smaller of eff and the fast
	/// path's bound `B - N`. An eff that GREW -- a resize, M falling -- wakes
	/// both lanes' heads, which may fit now (design 3.9.6).
	pub(crate) fn publish(&self, published: Published, live: impl Fn() -> u64, approx: impl Fn() -> i64) {
		let previous = self.eff.swap(published.eff, Ordering::Relaxed);
		self.eff_small.store(published.eff_small, Ordering::Relaxed);
		self.eff_large.store(published.eff_large, Ordering::Relaxed);
		self.m_model.store(published.m_model, Ordering::Relaxed);
		self.k_max.store(published.k_max, Ordering::Relaxed);
		self.model.store(
			match published.model {
				MetadataModel::Measured => MODEL_MEASURED,
				MetadataModel::PerObject => MODEL_PER_OBJECT,
			},
			Ordering::Relaxed,
		);

		match published.bands {
			Some(bands) => {
				self.band_s.store(bands.s, Ordering::Relaxed);
				self.band_b.store(bands.b, Ordering::Relaxed);
				self.shared.band_n.store(bands.n, Ordering::Relaxed);
			},

			None => {
				self.band_s.store(0, Ordering::Relaxed);
				self.band_b.store(0, Ordering::Relaxed);
				self.shared.band_n.store(u64::MAX, Ordering::Relaxed);
			},
		}

		let word_eff = published.eff.min(published.eff_small).min(published.eff_large);
		let fast_path = published.bands.map_or(u64::MAX, |bands| bands.b - bands.n);
		let pages = (word_eff.min(fast_path) >> PAGE_SHIFT).min(u64::MAX >> FLAG_BITS);
		let near = |count: u64| near_ceiling(count, published.k_max);
		let bytes_near = |approx: i64| {
			published.bands.is_some_and(|bands| approx.saturating_add(phys::FOLD_ERROR) >= bands.n.min(i64::MAX as u64) as i64)
		};

		let mut near_now = near(live());
		let mut bytes_near_now = bytes_near(approx());
		let word = &self.shared.word;
		let mut old = word.load(Ordering::Relaxed);

		loop {
			let flags = (old & ((1 << FLAG_BITS) - 1)) & !(META_NEAR | NEAR);
			let new = flags
				| if near_now { META_NEAR } else { 0 }
				| if bytes_near_now { NEAR } else { 0 }
				| (pages << FLAG_BITS);

			match word.compare_exchange_weak(old, new, Ordering::AcqRel, Ordering::Relaxed) {
				Ok(_) => break,
				Err(actual) => old = actual,
			}
		}

		// A client's `set_near` between the count read above and the store
		// was overwritten; its own increment is in the count by now.
		if !near_now {
			near_now = near(live());

			if near_now {
				word.fetch_or(META_NEAR, Ordering::AcqRel);
			}
		}

		// And a fold's NEAR the same way.
		if !bytes_near_now {
			bytes_near_now = bytes_near(approx());

			if bytes_near_now {
				word.fetch_or(NEAR, Ordering::AcqRel);
			}
		}

		// The heads re-check against the larger tier now, not at their next
		// poll. eff starts at u64::MAX, so a first publication never counts as
		// a growth.
		if published.eff > previous && self.waiting() {
			self.meta_lane.wake_head();
			self.bytes_lane.wake_head();
		}
	}

	/// A client whose new key brought the object count to `count`: flags the
	/// word when that is near the ceiling. One relaxed load when it is not.
	pub(crate) fn note_count(&self, count: u64) {
		if near_ceiling(count, self.k_max()) {
			self.shared.word.fetch_or(META_NEAR, Ordering::AcqRel);
		}
	}

	pub(crate) fn word(&self) -> u64 {
		self.shared.word.load(Ordering::Relaxed)
	}

	/// The policy worker's thread is gone.
	pub(crate) fn worker_gone(&self) -> bool {
		self.worker_gone.load(Ordering::Acquire)
	}

	/// Records the policy worker's exit (`WorkerGoneGuard`) and wakes every
	/// waiter -- the metadata lane's head and the whole bytes lane -- which
	/// then fails with `Internal`.
	pub(crate) fn mark_worker_gone(&self) {
		self.worker_gone.store(true, Ordering::Release);
		self.meta_lane.wake_head();
		self.bytes_lane.wake_all();
	}

	pub(crate) fn worker_progress(&self) -> u64 {
		self.worker_progress.load(Ordering::Relaxed)
	}

	pub(crate) fn set_worker_progress(&self, events: u64) {
		self.worker_progress.store(events, Ordering::Relaxed);
	}

	/// The policy worker's answer to `MakeRoom(request)`.
	pub(crate) fn answer_make_room(&self, request: u64, evicted: u64) {
		self.make_room_evictions.fetch_add(evicted, Ordering::Relaxed);
		self.room_outcome.store((request << 8) | evicted.min(255), Ordering::Release);
		self.meta_lane.wake_head();
	}

	/// A fresh `MakeRoom` request number.
	pub(crate) fn next_room_request(&self) -> u64 {
		self.room_request.fetch_add(1, Ordering::Relaxed) & (u64::MAX >> 8)
	}

	/// The answer to `request`, once the worker has given it.
	pub(crate) fn room_outcome(&self, request: u64) -> Option<u64> {
		let outcome = self.room_outcome.load(Ordering::Acquire);

		(outcome != u64::MAX && outcome >> 8 == request).then_some(outcome & 0xff)
	}

	pub(crate) fn count_overflow(&self) {
		self.metadata_overflows.fetch_add(1, Ordering::Relaxed);
	}

	pub(crate) fn count_make_room_request(&self) {
		self.make_room_requests.fetch_add(1, Ordering::Relaxed);
	}

	pub(crate) fn count_make_room_failure(&self) {
		self.make_room_failures.fetch_add(1, Ordering::Relaxed);
	}

	pub(crate) fn count_structural_set(&self) {
		self.structural_slow_sets.fetch_add(1, Ordering::Relaxed);
	}

	pub(crate) fn count_structural_placement(&self) {
		self.structural_placements.fetch_add(1, Ordering::Relaxed);
	}

	pub(crate) fn count_idle_kick(&self) {
		self.idle_kicks.fetch_add(1, Ordering::Relaxed);
	}

	pub(crate) fn count_divergence(&self) {
		self.metadata_model_divergence.fetch_add(1, Ordering::Relaxed);
	}

	/// A wipe resets the event counters, with the status' others. Everything
	/// else is live state and survives it.
	pub(crate) fn reset_counters(&self) {
		for counter in [
			&self.metadata_overflows,
			&self.make_room_requests,
			&self.make_room_evictions,
			&self.make_room_failures,
			&self.structural_slow_sets,
			&self.structural_placements,
			&self.idle_kicks,
			&self.metadata_model_divergence,
			&self.gate_disabled_sets,
			&self.gate_waits,
			&self.gate_wait_ns_total,
			&self.gate_wait_ns_max,
			&self.gate_stalls,
			&self.gate_stall_errors,
			&self.divert_sets,
			&self.divert_bytes,
			&self.admit_over_sets,
			&self.admit_over_bytes,
			&self.oversize_admits,
			&self.near_kicks,
			&self.max_waiters,
		] {
			counter.store(0, Ordering::Relaxed);
		}

		for bucket in &self.gate_wait_hist {
			bucket.store(0, Ordering::Relaxed);
		}

		self.gate_slow_paths.reset();
	}

	pub(crate) fn stats(&self) -> GateStats {
		GateStats {
			model: match self.model.load(Ordering::Relaxed) {
				MODEL_PER_OBJECT => MetadataModel::PerObject,
				_ => MetadataModel::Measured,
			},
			m_model: self.m_model(),
			eff: self.eff(),
			k_max: self.k_max(),
			metadata_overflows: self.metadata_overflows.load(Ordering::Relaxed),
			make_room_requests: self.make_room_requests.load(Ordering::Relaxed),
			make_room_evictions: self.make_room_evictions.load(Ordering::Relaxed),
			make_room_failures: self.make_room_failures.load(Ordering::Relaxed),
			structural_slow_sets: self.structural_slow_sets.load(Ordering::Relaxed),
			structural_placements: self.structural_placements.load(Ordering::Relaxed),
			idle_kicks: self.idle_kicks.load(Ordering::Relaxed),
			metadata_model_divergence: self.metadata_model_divergence.load(Ordering::Relaxed),
			state: self.state(),
			gate_disabled_sets: self.gate_disabled_sets.load(Ordering::Relaxed),
			gate_slow_paths: self.gate_slow_paths.sum(),
			gate_waits: self.gate_waits.load(Ordering::Relaxed),
			gate_wait_ns_total: self.gate_wait_ns_total.load(Ordering::Relaxed),
			gate_wait_ns_max: self.gate_wait_ns_max.load(Ordering::Relaxed),
			gate_wait_hist: std::array::from_fn(|i| self.gate_wait_hist[i].load(Ordering::Relaxed)),
			gate_stalls: self.gate_stalls.load(Ordering::Relaxed),
			gate_stall_errors: self.gate_stall_errors.load(Ordering::Relaxed),
			divert_sets: self.divert_sets.load(Ordering::Relaxed),
			divert_bytes: self.divert_bytes.load(Ordering::Relaxed),
			admit_over_sets: self.admit_over_sets.load(Ordering::Relaxed),
			admit_over_bytes: self.admit_over_bytes.load(Ordering::Relaxed),
			oversize_admits: self.oversize_admits.load(Ordering::Relaxed),
			near_kicks: self.near_kicks.load(Ordering::Relaxed),
			max_waiters: self.max_waiters.load(Ordering::Relaxed),
			waiters: self.bytes_lane.len() as u64,
			reserved: self.reserved.load(Ordering::Relaxed),
			bands: self.bands(),
		}
	}
}

/// Whether an object count of `count` is within `NEAR_KEYS` of the ceiling
/// `k_max`. No ceiling (`u64::MAX`, omega 0) is never near.
fn near_ceiling(count: u64, k_max: u64) -> bool {
	k_max != u64::MAX && count.saturating_add(NEAR_KEYS) >= k_max
}

/// The key ceiling `K_max` (see the module doc), from one publication's
/// figures:
///
///   * omega 0 -- `PAPER_DISABLE_SHARED_OVERHEAD=1`, or a status no tiered
///     constructor registered -- has no ceiling: `u64::MAX`;
///   * per-object: `floor((c_meta - ghost)+ / omega)`, `ghost` being what the
///     stack reserves beyond `stack_len * omega` -- the plan's `(L + 1) *
///     omega > F` exactly;
///   * measured: `max(l_hw, l_pub + floor((c_meta - m_model)+ / omega))` --
///     refills up to the high-water mark are free, growth beyond it is
///     estimated at omega per object, and a table step that lands M above
///     `c_meta` leaves the ceiling at `l_hw`: reuse only.
pub(crate) fn key_ceiling(
	model: MetadataModel,
	omega: CacheSize,
	c_meta: CacheSize,
	m_model: CacheSize,
	stack_len: u64,
	l_pub: u64,
	l_hw: u64,
) -> u64 {
	if omega == 0 {
		return u64::MAX;
	}

	match model {
		MetadataModel::PerObject => {
			let ghost = m_model.saturating_sub(stack_len.saturating_mul(omega));

			c_meta.saturating_sub(ghost) / omega
		},

		MetadataModel::Measured => {
			let room = c_meta.saturating_sub(m_model) / omega;

			l_hw.max(l_pub.saturating_add(room))
		},
	}
}

/// The size-split design's reservation, split between its two fast segments
/// in proportion to their capacities: `(small's, large's)`. THE split -- the
/// stack's settles and the policy worker's published class figures both call
/// it, so the client's structural check and the stack agree.
pub(crate) fn size_split_shares(reserved: CacheSize, small: CacheSize, large: CacheSize) -> (CacheSize, CacheSize) {
	let total = small + large;

	if total == 0 {
		return (0, 0);
	}

	let small_share = ((reserved as u128 * small as u128) / total as u128) as CacheSize;

	(small_share, reserved.saturating_sub(small_share))
}

/// What a set's length costs, computed before anything is allocated.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Sizes {
	/// `OverheadManager::base_size` of the object the set will build.
	pub(crate) base: ObjectSize,
	/// Its DRAM-resident part (`dram_resident_size`).
	pub(crate) resident: ObjectSize,
	/// The bytes that tier -- P's unit, and the stacks'
	/// (`resident_object_bytes`): what the structural check compares.
	pub(crate) value: CacheSize,
}

/// What `PaperCache::begin_set` decided for a set, for `commit` to carry out:
/// the tier to build in, the `Set`'s placement, and the inputs it was decided
/// from -- `commit` builds exactly the object `begin_set` checked -- and (B2)
/// the fast bytes the byte gate reserved for it, released once the value is
/// built (or the permit abandoned). Crate-only: the client path is `set`,
/// which pairs the two at once.
#[derive(Debug)]
pub(crate) struct SetPermit<'g> {
	pub(crate) hashed: HashedKey,
	pub(crate) tier: Tier,
	pub(crate) placement: Placement,
	pub(crate) len: usize,
	pub(crate) ttl: Option<u32>,
	pub(crate) sizes: Sizes,
	pub(crate) reservation: Reservation<'g>,
}

/// A set's admission verdict (`decide`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Verdict {
	/// Build the value in `tier`; its `Set` carries `placement`.
	Admit { tier: Tier, placement: Placement },

	/// A new key at the metadata ceiling under `EvictToFit`: wait in the
	/// metadata lane for the policy worker to make room.
	NeedsRoom,
}

/// Steps 1-3 of a set's admission (see the module doc), for a value whose
/// sizes are `sizes`: the metadata cap, `admission_tier`, the structural
/// check. `lane_head` is whether the caller is the metadata lane's head (a
/// newcomer queues behind a waiting new key even when room exists, so room
/// made for the head is not taken by a later arrival).
///
/// THE decision, for the real set path and for the reconcile harness's client
/// half alike (T14c), so the tests cover it and not a copy.
pub(crate) fn decide<K>(
	status: &AtomicStatus,
	objects: &crate::hybrid_policy::HybridObjectMap<K>,
	hashed: HashedKey,
	sizes: &Sizes,
	lane_head: bool,
) -> Result<Verdict, CacheError> {
	#[cfg(not(feature = "merged_object_store"))]
	use crate::object_store::ObjectStore;

	let gate = status.gate();
	let word = gate.word();

	// 1. The metadata cap: a new key, near the ceiling.
	if word & META_NEAR != 0 && objects.get_ref(&hashed).is_none() {
		let queued = !lane_head && gate.meta_lane.len() > 0;

		if queued || status.live_num_objects().saturating_add(1) > gate.k_max() {
			match gate.config().on_metadata_overflow {
				MetadataOverflow::Error => {
					gate.count_overflow();
					return Err(CacheError::MetadataOverflow);
				},

				MetadataOverflow::EvictToFit => return Ok(Verdict::NeedsRoom),
			}
		}
	}

	// 2-3. The design's tier, and the structural check.
	let (tier, placement) = place(status, objects, hashed, sizes, word);

	Ok(Verdict::Admit { tier, placement })
}

/// Steps 2-3 of a set's admission: the design's own tier (`admission_tier`),
/// then structural placement -- `word` the gate word the caller read. Re-run by
/// a set waiting in the bytes lane at every wake (B2): either can have moved
/// while it waited, and a value that no longer fits even an empty tier leaves
/// the lane as a structural set rather than waiting to be built fast (the
/// liveness review).
pub(crate) fn place<K>(
	status: &AtomicStatus,
	objects: &crate::hybrid_policy::HybridObjectMap<K>,
	hashed: HashedKey,
	sizes: &Sizes,
	word: u64,
) -> (Tier, Placement) {
	let gate = status.gate();

	// 2. The design's own admission rule.
	let mut tier = crate::hybrid_policy::admission_tier(status.policy(), hashed, status, objects);
	let mut placement = Placement::Normal;

	// 3. Structural: larger than an empty fast tier. The page figure first --
	// a value under it fits every class -- then the exact one.
	let pages = (word >> FLAG_BITS) << PAGE_SHIFT;

	if sizes.value > pages && sizes.value > gate.eff_for(status, sizes.base) {
		tier = Tier::Slow;
		placement = Placement::Structural;
		gate.count_structural_set();
	}

	(tier, placement)
}

/// A FIFO lane of waiting sets: an explicit queue of waiter slots, the head
/// being its front. Every exit -- admitted, failed, or unwinding -- goes
/// through `LaneGuard`'s drop, which removes its slot wherever it is and, if
/// it was the front, wakes the new front; so a waiter leaving out of turn can
/// never wedge the lane (the design review's blocker on a ticket lock).
#[derive(Default)]
pub(crate) struct Lane {
	queue: parking_lot::Mutex<VecDeque<Arc<LaneSlot>>>,

	/// `queue.len()`, readable without the lock.
	len: AtomicUsize,

	/// The front's thread, for the policy worker's wake-up: a lock of its own,
	/// which the worker takes alone (it never takes `queue`), and a client
	/// takes only under `queue`, so the two cannot deadlock.
	head: parking_lot::Mutex<Option<Thread>>,

	/// Test builds: the order waiters joined in, for the lane's tests.
	#[cfg(test)]
	next_id: AtomicU64,
}

pub(crate) struct LaneSlot {
	#[cfg(test)]
	id: u64,
	thread: Thread,
}

impl Lane {
	/// Waiters in the lane.
	pub(crate) fn len(&self) -> usize {
		self.len.load(Ordering::Acquire)
	}

	/// Joins the lane at its back.
	pub(crate) fn enqueue(&self) -> LaneGuard<'_> {
		self.join(None)
	}

	/// Joins the lane at its back, keeping `bit` of `word` set while the lane
	/// holds anyone -- set by the first waiter's join and cleared by the last
	/// one's departure, both under the lane's lock, so it is never stale: the
	/// bytes lane's `CLOSED` (B2) -- and a gate watching for freed bytes
	/// meanwhile (`phys::watch_freed`).
	pub(crate) fn enqueue_closing<'a>(&'a self, word: &'a AtomicU64, bit: u64) -> LaneGuard<'a> {
		self.join(Some((word, bit)))
	}

	fn join<'a>(&'a self, closing: Option<(&'a AtomicU64, u64)>) -> LaneGuard<'a> {
		let slot = Arc::new(LaneSlot {
			#[cfg(test)]
			id: self.next_id.fetch_add(1, Ordering::Relaxed),
			thread: std::thread::current(),
		});

		let mut queue = self.queue.lock();
		queue.push_back(slot.clone());
		self.len.store(queue.len(), Ordering::Release);

		if queue.len() == 1 {
			*self.head.lock() = Some(slot.thread.clone());

			if let Some((word, bit)) = closing {
				word.fetch_or(bit, Ordering::AcqRel);
				phys::watch_freed(true);
			}
		}

		LaneGuard { lane: self, slot, closing }
	}

	/// Wakes every waiter, each to re-check: the gate stalled or was disabled,
	/// a wipe, or the worker went. Takes the lane's lock -- the one place the
	/// policy worker does -- which no holder keeps across a wait.
	pub(crate) fn wake_all(&self) {
		for slot in self.queue.lock().iter() {
			slot.thread.unpark();
		}
	}

	/// Wakes the lane's head, if any. The policy worker's (and a departing
	/// head's) notification.
	pub(crate) fn wake_head(&self) {
		if let Some(head) = self.head.lock().as_ref() {
			head.unpark();
		}
	}
}

/// A place in a `Lane`, released on drop.
pub(crate) struct LaneGuard<'a> {
	lane: &'a Lane,
	slot: Arc<LaneSlot>,
	/// The word bit the lane keeps set while non-empty (`enqueue_closing`).
	closing: Option<(&'a AtomicU64, u64)>,
}

impl LaneGuard<'_> {
	#[cfg(test)]
	pub(crate) fn id(&self) -> u64 {
		self.slot.id
	}

	/// Whether this waiter is the lane's front.
	pub(crate) fn is_head(&self) -> bool {
		self.lane.queue.lock().front().is_some_and(|front| Arc::ptr_eq(front, &self.slot))
	}
}

impl Drop for LaneGuard<'_> {
	fn drop(&mut self) {
		let mut queue = self.lane.queue.lock();
		let was_front = queue.front().is_some_and(|front| Arc::ptr_eq(front, &self.slot));

		queue.retain(|slot| !Arc::ptr_eq(slot, &self.slot));
		self.lane.len.store(queue.len(), Ordering::Release);

		if queue.is_empty() {
			if let Some((word, bit)) = self.closing {
				word.fetch_and(!bit, Ordering::AcqRel);
				phys::watch_freed(false);
			}
		}

		if was_front {
			let next = queue.front().map(|front| front.thread.clone());

			if let Some(next) = &next {
				next.unpark();
			}

			*self.lane.head.lock() = next;
		}
	}
}

/// Parks the calling waiter for at most `timeout`: an unpark from a departing
/// head or from the policy worker ends it early, and a stale token only costs
/// an early return, after which the waiter re-checks.
pub(crate) fn park(timeout: Duration) {
	std::thread::park_timeout(timeout);
}

/// Waits, as the metadata lane's head, for the policy worker's answer to
/// `MakeRoom(request)` -- the request already sent -- and returns how many
/// victims it evicted.
///
/// `CacheError::Internal` if the worker is gone; `MetadataOverflow` if no
/// answer came while the worker made NO progress for `stall_window` (a worker
/// working through a backlog ahead of the request keeps the waiter waiting:
/// the backlog is part of this set's cost, as `MakeRoom` queues behind it).
pub(crate) fn await_room(gate: &Gate, request: u64, config: &GateConfig) -> Result<u64, CacheError> {
	let mut since = Instant::now();
	let mut progress = gate.worker_progress();

	loop {
		if let Some(evicted) = gate.room_outcome(request) {
			return Ok(evicted);
		}

		if gate.worker_gone() {
			return Err(CacheError::Internal);
		}

		let now = Instant::now();

		if gate.worker_progress() != progress {
			progress = gate.worker_progress();
			since = now;
		} else if now.saturating_duration_since(since) >= config.stall_window {
			gate.count_make_room_failure();
			return Err(CacheError::MetadataOverflow);
		}

		park(config.poll_interval.min(config.stall_window.max(Duration::from_micros(1))));
	}
}

/// Sets the gate's `worker_gone` when the policy worker's `run` ends, by
/// return or by unwinding, and wakes the metadata lane's head so it sees it.
pub(crate) struct WorkerGoneGuard {
	status: crate::StatusRef,
}

impl WorkerGoneGuard {
	pub(crate) fn new(status: &crate::StatusRef) -> Self {
		WorkerGoneGuard { status: status.clone() }
	}
}

impl Drop for WorkerGoneGuard {
	fn drop(&mut self) {
		let gate = self.status.gate();

		gate.mark_worker_gone();

		// The fold hook is the worker's to clear (B2): a gate whose worker is
		// gone is never re-evaluated.
		if gate.hooked.swap(false, Ordering::AcqRel) {
			phys::clear_gate_hook(&gate.shared);
		}
	}
}

impl Drop for Gate {
	/// A cache's gate going away undoes what it holds process-wide: the FREED
	/// watch of an unresolved stall -- otherwise every fast refund in the
	/// process would count its bytes for good -- and the fold hook, if its
	/// worker did not clear it (`WorkerGoneGuard` does, as the worker exits).
	/// The lanes are empty by then: a waiter borrows the cache.
	fn drop(&mut self) {
		if *self.stalled.get_mut() {
			phys::watch_freed(false);
		}

		if *self.hooked.get_mut() {
			phys::clear_gate_hook(&self.shared);
		}
	}
}

// ---------------------------------------------------------------------------
// The byte gate (S5, commit B2)

/// What the byte decision says for one attempt on given readings
/// (`byte_verdict`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ByteVerdict {
	/// Admitted with nothing reserved: `P + v <= N` (as the fast path).
	Admit,

	/// Admitted if `v` can be added to the reservation `R` it was decided
	/// against: `P + R + v <= B`.
	Reserve,

	/// Admitted OVERSIZE (`v > B - S`) on a settled tier -- `P <= S`, nothing
	/// reserved -- if the reservation is still 0.
	Oversize,

	/// Not admissible now.
	Wait,
}

/// The byte decision (design 3.9.5) on the readings `p` (P, exact), `r` (the
/// reservation) and the levels, for a value of `v` bytes that fits an empty
/// tier (`eff`): pure, so the cache and the tests decide by the same rule.
pub(crate) fn byte_verdict(v: CacheSize, p: CacheSize, r: CacheSize, bands: Bands, eff: CacheSize) -> ByteVerdict {
	let Bands { s, n, b } = bands;

	if v <= b.saturating_sub(s) {
		if p.saturating_add(r).saturating_add(v) > b {
			return ByteVerdict::Wait;
		}

		return match p.saturating_add(v) <= n {
			true => ByteVerdict::Admit,
			false => ByteVerdict::Reserve,
		};
	}

	// Oversize: only a value the tier could hold at all (a larger one is
	// structural, which step 3 decides), and only when admitting it keeps
	// `P <= S + v` -- a settled tier with nothing reserved.
	match v <= eff && p <= s && r == 0 {
		true => ByteVerdict::Oversize,
		false => ByteVerdict::Wait,
	}
}

/// What one attempt at the byte gate got (`Gate::admit_bytes`).
pub(crate) enum Bytes<'g> {
	/// Build the value fast; the reservation (if any) is released when the
	/// permit drops.
	Admit(Reservation<'g>),

	/// Wait: not admissible now.
	Wait,
}

/// Fast bytes the byte gate holds for an admitted set until its value is
/// built: released on drop -- after `commit`'s build charged P, or with a
/// permit abandoned before it -- saturating, and exactly once (a wipe never
/// resets the reservation).
#[must_use]
pub(crate) struct Reservation<'g> {
	gate: Option<&'g Gate>,
	bytes: CacheSize,
}

impl<'g> Reservation<'g> {
	/// Nothing held.
	pub(crate) fn none() -> Self {
		Reservation { gate: None, bytes: 0 }
	}

	/// `bytes`, already added to `gate`'s reservation.
	fn held(gate: &'g Gate, bytes: CacheSize) -> Self {
		Reservation { gate: Some(gate), bytes }
	}

	/// The bytes held.
	#[cfg(test)]
	pub(crate) fn bytes(&self) -> CacheSize {
		self.bytes
	}
}

impl Drop for Reservation<'_> {
	fn drop(&mut self) {
		if let Some(gate) = self.gate {
			gate.release(self.bytes);
		}
	}
}

impl std::fmt::Debug for Reservation<'_> {
	fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
		write!(f, "Reservation({} B)", self.bytes)
	}
}

/// The watchdog's progress readings (`progress_made`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct Marks {
	/// `phys::freed()`: bytes refunded while a gate watched.
	pub(crate) freed: u64,
	/// The bytes lane's progress count: landed demotions, admissions.
	pub(crate) progress: u64,
	/// The close level B.
	pub(crate) b: CacheSize,
}

/// Progress, for the watchdog (design 3.9.6): bytes were freed anywhere (a
/// demotion landing, an eviction, a delete, a reap, a reader dropping a copy),
/// a demotion landed or a waiter was admitted, or the close level rose (eff
/// grew). NOT "P fell": frees offset by promotion copies leave P flat, and
/// that is load, which waits, not a stall (T19c).
pub(crate) fn progress_made(then: Marks, now: Marks) -> bool {
	now.freed != then.freed || now.progress != then.progress || now.b > then.b
}

/// What the watchdog tells a waiting set (`Waiter::watch`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Watch {
	/// Park at most this long, then check again.
	Park(Duration),

	/// Stalled: act per `on_stall`.
	Stalled,
}

/// A fast set waiting in the bytes lane: its place -- released on drop,
/// wherever it is, which wakes the next head -- and its watchdog (design
/// 3.9.6-3.9.7, as the liveness review amended them). Its wait is recorded
/// when it drops, whatever the outcome.
pub(crate) struct Waiter<'g> {
	gate: &'g Gate,
	place: LaneGuard<'g>,
	started: Instant,

	/// The current no-progress window: when it began, the progress readings
	/// then, and the worker's gate passes then.
	since: Instant,
	marks: Marks,
	passes: u64,

	was_head: bool,
}

impl<'g> Waiter<'g> {
	/// Joins the bytes lane at its back (closing the gate if it is the first).
	pub(crate) fn enqueue(gate: &'g Gate) -> Self {
		let place = gate.bytes_lane.enqueue_closing(&gate.shared.word, CLOSED);

		gate.gate_waits.fetch_add(1, Ordering::Relaxed);
		gate.max_waiters.fetch_max(gate.bytes_lane.len() as u64, Ordering::Relaxed);

		let now = Instant::now();

		Waiter { gate, place, started: now, since: now, marks: gate.marks(), passes: gate.passes(), was_head: false }
	}

	/// Whether this waiter is the lane's head.
	pub(crate) fn is_head(&self) -> bool {
		self.place.is_head()
	}

	/// Admitted: the lane moved -- progress for whoever is behind.
	pub(crate) fn admitted(&self) {
		self.gate.lane_progress.fetch_add(1, Ordering::Relaxed);

		#[cfg(test)]
		self.gate.test_admissions.lock().push(self.place.slot.id);
	}

	/// The watchdog, once per wake. Progress (`progress_made`), or becoming
	/// the head, starts a new window. The HEAD stalls after a whole window
	/// with no progress and at least `STALL_PASSES` of the worker's passes
	/// since the window began; it then marks the gate stalled (the worker
	/// clears that at the first byte freed). While the gate is stalled any
	/// waiter -- a newcomer included -- acts after `STALL_PROBE` without
	/// progress. Otherwise: park until the next check.
	pub(crate) fn watch(&mut self, config: &GateConfig) -> Watch {
		let gate = self.gate;
		let now = Instant::now();
		let head = self.place.is_head();
		let marks = gate.marks();

		if progress_made(self.marks, marks) || (head && !self.was_head) {
			self.since = now;
			self.marks = marks;
			self.passes = gate.passes();
		}

		self.was_head = head;

		let stalled = gate.stalled();
		let window = match stalled {
			true => config.stall_window.min(STALL_PROBE),
			false => config.stall_window,
		};
		let quiet = now.saturating_duration_since(self.since);

		if (head || stalled) && quiet >= window && gate.passes() >= self.passes.saturating_add(STALL_PASSES) {
			gate.declare_stall(marks);
			return Watch::Stalled;
		}

		let park = match head {
			true => config.poll_interval,
			false => NON_HEAD_PARK,
		};

		match window.saturating_sub(quiet) {
			left if left.is_zero() => Watch::Park(park),
			left => Watch::Park(park.min(left)),
		}
	}
}

impl Drop for Waiter<'_> {
	fn drop(&mut self) {
		self.gate.record_wait(self.started.elapsed());
	}
}

impl Gate {
	/// The byte gate's state.
	pub(crate) fn state(&self) -> GateState {
		GateState::from_u8(self.state.load(Ordering::Acquire))
	}

	/// The levels as last published (all 0 while the gate is not enabled).
	pub(crate) fn bands(&self) -> Bands {
		let n = self.shared.band_n.load(Ordering::Relaxed);

		Bands {
			s: self.band_s.load(Ordering::Relaxed),
			n: if n == u64::MAX { 0 } else { n },
			b: self.band_b.load(Ordering::Relaxed),
		}
	}

	fn band_b(&self) -> CacheSize {
		self.band_b.load(Ordering::Relaxed)
	}

	/// The worker's gate passes so far.
	pub(crate) fn passes(&self) -> u64 {
		self.passes.load(Ordering::Acquire)
	}

	/// The watchdog fired and nothing has been freed since.
	pub(crate) fn stalled(&self) -> bool {
		self.stalled.load(Ordering::Acquire)
	}

	/// Bytes admitted sets hold reserved.
	pub(crate) fn reserved(&self) -> CacheSize {
		self.reserved.load(Ordering::Acquire)
	}

	fn marks(&self) -> Marks {
		Marks {
			freed: phys::freed(),
			progress: self.lane_progress.load(Ordering::Relaxed),
			b: self.band_b(),
		}
	}

	/// Whether the byte gate runs for a set now: its state, and -- when a live
	/// count moved since the worker last read them (`phys::gate_epoch`) -- the
	/// counts read again at once (design 3.9.8). A set that finds the cache is
	/// no longer P's only user disables the gate itself (`NotSole`) and
	/// releases the waiters; only the worker enables it again.
	fn enabled(&self) -> bool {
		if self.state() != GateState::Enabled {
			return false;
		}

		let epoch = phys::gate_epoch();

		if epoch != self.epoch_seen.load(Ordering::Acquire) {
			if !phys::sole_fast_user() {
				self.disable(GateState::NotSole);
				return false;
			}

			self.epoch_seen.store(epoch, Ordering::Release);
		}

		true
	}

	/// Disables the byte gate now, for `reason`: every waiter re-checks and is
	/// admitted ungated. From a set that saw the live counts move, and from
	/// `set_gate_config` turning the gate `Off`; the worker's pass does the
	/// rest (the fold hook).
	pub(crate) fn disable(&self, reason: GateState) {
		self.state.store(reason as u8, Ordering::Release);
		self.release_all();
	}

	/// The gate stops holding sets: `NEAR` cleared and the near level lifted
	/// (the fast path open), a stall ended, every waiter woken to re-check.
	fn release_all(&self) {
		self.shared.band_n.store(u64::MAX, Ordering::Relaxed);
		self.shared.word.fetch_and(!NEAR, Ordering::AcqRel);

		if self.stalled.swap(false, Ordering::AcqRel) {
			phys::watch_freed(false);
		}

		self.bytes_lane.wake_all();
	}

	/// Step 4 of a fast set's admission, one attempt with no waiting (design
	/// 3.9.5): `v` the value's bytes; `head` whether the caller is the bytes
	/// lane's head -- behind a waiter a newcomer waits, strict FIFO; `p` reads
	/// P when the exact path needs it; `kick` wakes the policy worker, at most
	/// once per pass, when P is at or above the near level.
	pub(crate) fn admit_bytes(&self, v: CacheSize, head: bool, p: impl FnOnce() -> i64, kick: impl FnOnce()) -> Bytes<'_> {
		let word = self.word();

		// The fast path: nothing near, nobody waiting, and a value within the
		// page figure (`B - N` while the gate runs, eff otherwise).
		if word & (NEAR | CLOSED) == 0 && v <= (word >> FLAG_BITS) << PAGE_SHIFT {
			return Bytes::Admit(Reservation::none());
		}

		// A zero-length value charges P nothing: nothing to hold it to.
		if v == 0 {
			return Bytes::Admit(Reservation::none());
		}

		if !self.enabled() {
			if self.state() != GateState::Off {
				self.gate_disabled_sets.fetch_add(1, Ordering::Relaxed);
			}

			return Bytes::Admit(Reservation::none());
		}

		self.gate_slow_paths.incr();

		if word & CLOSED != 0 && !head {
			return Bytes::Wait;
		}

		let p = p().max(0) as CacheSize;
		let bands = self.bands();

		if p >= bands.n && self.near_kick_armed.swap(false, Ordering::Relaxed) {
			self.near_kicks.fetch_add(1, Ordering::Relaxed);
			kick();
		}

		let eff = self.eff();
		let mut r = self.reserved.load(Ordering::Acquire);

		loop {
			match byte_verdict(v, p, r, bands, eff) {
				ByteVerdict::Admit => return Bytes::Admit(Reservation::none()),
				ByteVerdict::Wait => return Bytes::Wait,

				ByteVerdict::Reserve => match self.reserved.compare_exchange_weak(r, r + v, Ordering::AcqRel, Ordering::Acquire) {
					Ok(_) => return Bytes::Admit(Reservation::held(self, v)),
					Err(actual) => r = actual,
				},

				ByteVerdict::Oversize => match self.reserved.compare_exchange(0, v, Ordering::AcqRel, Ordering::Acquire) {
					Ok(_) => {
						self.oversize_admits.fetch_add(1, Ordering::Relaxed);
						return Bytes::Admit(Reservation::held(self, v));
					},

					Err(actual) => r = actual,
				},
			}
		}
	}

	/// What a stalled set does (design 3.9.7): `(tier, placement, reservation)`
	/// to build with, or the error.
	pub(crate) fn on_stall(&self, v: CacheSize, on_stall: OnStall) -> Result<(Tier, Placement, Reservation<'_>), CacheError> {
		match on_stall {
			OnStall::Error => {
				self.gate_stall_errors.fetch_add(1, Ordering::Relaxed);
				Err(CacheError::FastTierStalled)
			},

			OnStall::Divert => {
				self.divert_sets.fetch_add(1, Ordering::Relaxed);
				self.divert_bytes.fetch_add(v, Ordering::Relaxed);
				Ok((Tier::Slow, Placement::Diverted, Reservation::none()))
			},

			OnStall::AdmitOver => {
				self.admit_over_sets.fetch_add(1, Ordering::Relaxed);
				self.admit_over_bytes.fetch_add(v, Ordering::Relaxed);
				self.reserved.fetch_add(v, Ordering::AcqRel);
				Ok((Tier::Fast, Placement::Normal, Reservation::held(self, v)))
			},
		}
	}

	/// A permit's reservation, released: saturating (never below zero, which
	/// a counter reset under an outstanding permit would have caused -- the
	/// reservation is live state and a wipe leaves it alone), then the head
	/// woken if anyone waits.
	fn release(&self, bytes: CacheSize) {
		if bytes == 0 {
			return;
		}

		let mut r = self.reserved.load(Ordering::Acquire);

		loop {
			debug_assert!(r >= bytes, "releasing {bytes} B of a {r} B reservation");

			match self.reserved.compare_exchange_weak(r, r.saturating_sub(bytes), Ordering::AcqRel, Ordering::Acquire) {
				Ok(_) => break,
				Err(actual) => r = actual,
			}
		}

		if self.bytes_lane.len() > 0 {
			self.bytes_lane.wake_head();
		}
	}

	/// A migration consumer landed a demotion (design 3.9.6): while a set
	/// waits for fast bytes, progress, and the head woken. One relaxed load
	/// otherwise.
	pub(crate) fn note_demotion(&self) {
		if self.bytes_lane.len() > 0 {
			self.lane_progress.fetch_add(1, Ordering::Relaxed);
			self.bytes_lane.wake_head();
		}
	}

	/// A set waits in either lane: the worker polls SHORT and does not park
	/// on its idle poll.
	pub(crate) fn waiting(&self) -> bool {
		self.bytes_lane.len() > 0 || self.meta_lane.len() > 0
	}

	/// Every waiter, in both lanes, re-checks now: the worker's wipe.
	pub(crate) fn wake_waiters(&self) {
		self.meta_lane.wake_head();
		self.bytes_lane.wake_all();
	}

	/// The watchdog fired (the head's, or a waiter's while stalled): the
	/// readings at the stall, then the flag, a watch kept on freed bytes
	/// for as long as it holds (so the worker sees the stall end even after
	/// every waiter has gone), and every waiter woken to act.
	fn declare_stall(&self, marks: Marks) {
		if self.stalled.load(Ordering::Acquire) {
			return;
		}

		self.stall_freed.store(marks.freed, Ordering::Relaxed);
		self.stall_p.store(phys::fast_bytes_signed(), Ordering::Relaxed);
		self.stall_b.store(marks.b, Ordering::Relaxed);

		if !self.stalled.swap(true, Ordering::AcqRel) {
			phys::watch_freed(true);
			self.gate_stalls.fetch_add(1, Ordering::Relaxed);
			self.bytes_lane.wake_all();
		}
	}

	/// A waiter left the bytes lane after `waited`.
	fn record_wait(&self, waited: Duration) {
		let ns = waited.as_nanos().min(u64::MAX as u128) as u64;
		let us = (ns / 1_000).max(1);
		let bucket = (u64::BITS - 1 - us.leading_zeros()) as usize;

		self.gate_wait_ns_total.fetch_add(ns, Ordering::Relaxed);
		self.gate_wait_ns_max.fetch_max(ns, Ordering::Relaxed);
		self.gate_wait_hist[bucket.min(WAIT_BUCKETS - 1)].fetch_add(1, Ordering::Relaxed);
	}

	/// The policy worker's gate pass, at the end of every publication (design
	/// 3.9.9): the state `state` it evaluated -- the fold hook installed or
	/// cleared with it, the waiters released when it disables the gate; a stall
	/// ended when anything was freed since it fired, P fell below where it was,
	/// or the close level rose (the worker owns the stall, the liveness
	/// review); the bytes lane's head notified; the pass counted.
	pub(crate) fn worker_pass(&self, state: GateState, epoch: u64) {
		let enabled = state == GateState::Enabled;

		if enabled != self.hooked.load(Ordering::Relaxed) {
			match enabled {
				true => phys::install_gate_hook(&self.shared),
				false => phys::clear_gate_hook(&self.shared),
			}

			self.hooked.store(enabled, Ordering::Relaxed);
		}

		self.epoch_seen.store(epoch, Ordering::Release);

		if self.state.swap(state as u8, Ordering::AcqRel) == GateState::Enabled as u8 && !enabled {
			self.release_all();
		}

		if self.stalled() {
			let freed = phys::freed();

			if freed != self.stall_freed.load(Ordering::Relaxed)
				|| phys::fast_bytes_signed() < self.stall_p.load(Ordering::Relaxed)
				|| self.band_b() > self.stall_b.load(Ordering::Relaxed)
			{
				if self.stalled.swap(false, Ordering::AcqRel) {
					phys::watch_freed(false);
				}
			}
		}

		let waiting = self.bytes_lane.len();

		if waiting > 0 {
			self.max_waiters.fetch_max(waiting as u64, Ordering::Relaxed);

			#[cfg(test)]
			let notify = !test_hooks::no_pass_notify();
			#[cfg(not(test))]
			let notify = true;

			if notify {
				self.bytes_lane.wake_head();
			}
		}

		self.near_kick_armed.store(true, Ordering::Relaxed);
		self.passes.fetch_add(1, Ordering::AcqRel);
	}
}

/// Test hooks for the byte gate's tests, process-global (every gate test runs
/// alone in a child process): a pause and a pace for the migration consumers,
/// the worker held, its per-pass notify and its test flush off, M
/// overridden. Each is an RAII guard, released on drop and on unwinding, so a
/// failing test never leaves the process's workers paused (the liveness
/// review).
#[cfg(test)]
pub(crate) mod test_hooks {
	use std::{
		sync::atomic::{AtomicBool, AtomicU64, Ordering::Relaxed},
		time::Duration,
	};

	static CONSUMER_PAUSE: AtomicBool = AtomicBool::new(false);
	static CONSUMER_DELAY_US: AtomicU64 = AtomicU64::new(0);
	static NO_FLUSH: AtomicBool = AtomicBool::new(false);
	static NO_PASS_NOTIFY: AtomicBool = AtomicBool::new(false);
	static HOLD_WORKER: AtomicBool = AtomicBool::new(false);
	static M_OVERRIDE: AtomicU64 = AtomicU64::new(0);

	/// Sets `flag` until the guard drops, which restores what it was.
	pub(crate) struct Flag(&'static AtomicBool, bool);

	impl Drop for Flag {
		fn drop(&mut self) {
			self.0.store(self.1, Relaxed);
		}
	}

	fn flag(flag: &'static AtomicBool) -> Flag {
		Flag(flag, flag.swap(true, Relaxed))
	}

	/// Sets `value` until the guard drops, which restores what it was: the
	/// guards nest (a test that raises M over a held M).
	pub(crate) struct Value(&'static AtomicU64, u64);

	impl Drop for Value {
		fn drop(&mut self) {
			self.0.store(self.1, Relaxed);
		}
	}

	fn value(cell: &'static AtomicU64, v: u64) -> Value {
		Value(cell, cell.swap(v, Relaxed))
	}

	/// The migration consumers take no migration until the guard drops.
	pub(crate) fn pause_consumers() -> Flag {
		flag(&CONSUMER_PAUSE)
	}

	/// Each consumer landing waits `delay` first.
	pub(crate) fn pace_consumers(delay: Duration) -> Value {
		value(&CONSUMER_DELAY_US, delay.as_micros() as u64)
	}

	/// A real cache's worker does not flush the consumers after each batch
	/// (`test_flush`): with them paused it would spin in the flush, and its
	/// passes -- the gate's -- would stop.
	pub(crate) fn no_flush() -> Flag {
		flag(&NO_FLUSH)
	}

	/// The worker's pass notifies no waiter: only the other wake-ups do.
	pub(crate) fn no_pass_notify() -> bool {
		NO_PASS_NOTIFY.load(Relaxed)
	}

	pub(crate) fn suppress_pass_notify() -> Flag {
		flag(&NO_PASS_NOTIFY)
	}

	/// The workers stop at the top of their loop until the guard drops.
	pub(crate) fn hold_workers() -> Flag {
		flag(&HOLD_WORKER)
	}

	/// M_model is `bytes` at every publication.
	pub(crate) fn override_m(bytes: u64) -> Value {
		value(&M_OVERRIDE, bytes)
	}

	pub(crate) fn m_override() -> Option<u64> {
		Some(M_OVERRIDE.load(Relaxed)).filter(|&m| m != 0)
	}

	pub(crate) fn flush_off() -> bool {
		NO_FLUSH.load(Relaxed)
	}

	/// A consumer, before it lands a migration.
	pub(crate) fn consumer_wait() {
		while CONSUMER_PAUSE.load(Relaxed) {
			std::thread::sleep(Duration::from_millis(1));
		}

		match CONSUMER_DELAY_US.load(Relaxed) {
			0 => {},
			us => std::thread::sleep(Duration::from_micros(us)),
		}
	}

	/// A worker, at the top of its loop.
	pub(crate) fn worker_hold() {
		while HOLD_WORKER.load(Relaxed) {
			std::thread::sleep(Duration::from_millis(1));
		}
	}
}

#[cfg(test)]
mod tests {
	use super::*;

	/// The byte gate's levels (design 3.9.2): S the settle target, B = eff +
	/// slack, N one near band below B -- the wider of `near_frac x eff` and
	/// `concurrency_hint x value_hint` -- clamped into `[S + 1, B]`; at eff 0
	/// all three 0.
	#[test]
	fn the_bands_are_ordered_and_a_hint_widens_the_near_band() {
		let mut config = GateConfig::default();
		config.mode = GateMode::Block;

		let eff: CacheSize = 1_000_000;
		let near = (config.near_frac * eff as f64) as CacheSize;
		let s = drain_target::bytes(eff);

		assert_eq!(bands(eff, &config), Bands { s, n: eff - near, b: eff });
		assert!(s < eff - near, "S below N at the default band");

		config.slack = 5_000;
		assert_eq!(bands(eff, &config), Bands { s, n: eff + 5_000 - near, b: eff + 5_000 });

		// A server's hint (S9): wider than 1% of eff, and never down to S.
		config.slack = 0;
		config.concurrency_hint = 8;
		config.value_hint = 1_500;
		assert_eq!(bands(eff, &config).n, eff - 12_000);

		config.concurrency_hint = 64;
		config.value_hint = 1_000;
		assert_eq!(bands(eff, &config).n, s + 1, "clamped above the settle target");

		assert_eq!(bands(0, &GateConfig::default()), Bands { s: 0, n: 0, b: 0 });
	}

	/// The byte decision (design 3.9.5) on its readings: a value within the
	/// settled tier's headroom `B - S` is admitted while `P + R + v <= B`,
	/// reserving unless `P + v <= N`; an OVERSIZE one only on a settled tier
	/// with nothing reserved, and only if it could fit an empty tier at all.
	/// Red with oversize taken as normal (`oversizeasnormal`).
	#[test]
	fn the_byte_verdict_follows_the_bands() {
		use ByteVerdict::{Admit, Oversize, Reserve, Wait};

		let bands = Bands { s: 980, n: 990, b: 1_000 };
		let at = |v, p, r| byte_verdict(v, p, r, bands, 1_000);

		assert_eq!(at(10, 900, 0), Admit, "at or under N: nothing reserved");
		assert_eq!(at(10, 980, 0), Admit, "P + v = N");
		assert_eq!(at(10, 985, 0), Reserve, "over N, within B");
		assert_eq!(at(10, 985, 5), Reserve, "the reservation counted, to B exactly");
		assert_eq!(at(10, 985, 6), Wait, "P + R + v over B");
		assert_eq!(at(20, 980, 0), Reserve, "v = B - S is not oversize");
		assert_eq!(at(20, 981, 0), Wait);

		assert_eq!(at(21, 980, 0), Oversize, "oversize, on a settled tier");
		assert_eq!(at(21, 981, 0), Wait, "oversize waits until P <= S ...");
		assert_eq!(at(21, 0, 1), Wait, "... with nothing reserved");
		assert_eq!(at(1_000, 0, 0), Oversize, "the whole of an empty tier");
		assert_eq!(at(1_001, 0, 0), Wait, "larger than an empty tier: structural's, never the gate's");
	}

	/// The watchdog's progress (design 3.9.6, T19c): bytes freed, a landed
	/// demotion or an admission, or a higher close level -- NOT "P fell": frees
	/// offset by promotion copies leave P flat and are progress; a close level
	/// that FELL is not. Red with FREED ignored (`nofreed`).
	#[test]
	fn progress_is_something_freed_not_p_falling() {
		let then = Marks { freed: 10, progress: 5, b: 1_000 };

		assert!(!progress_made(then, then), "nothing moved");
		assert!(progress_made(then, Marks { freed: 11, ..then }), "bytes freed, whatever P did");
		assert!(progress_made(then, Marks { progress: 6, ..then }), "a demotion landed, or a waiter was admitted");
		assert!(progress_made(then, Marks { b: 1_001, ..then }), "the tier grew");
		assert!(!progress_made(then, Marks { b: 999, ..then }), "the tier shrank");
	}

	/// The word's page figure is the fast path's whole bound: while the byte
	/// gate runs it is at most `B - N`, below `B - S`, so an OVERSIZE value
	/// never takes the one-load path (the semantics review); eff otherwise.
	/// NEAR is `approx + E >= N`, from the worker's publication and from the
	/// fold hook. Red with the page figure eff while the gate runs
	/// (`fastoversize`).
	#[test]
	fn the_fast_path_never_admits_an_oversize_value_and_near_follows_approx() {
		let gate = Gate::default();
		let eff: CacheSize = 10_000_000;
		let bands = Bands { s: 9_800_000, n: 9_900_000, b: 10_000_000 };
		let published = |bands| Published {
			model: MetadataModel::PerObject,
			m_model: 0,
			eff,
			eff_small: eff,
			eff_large: eff,
			k_max: u64::MAX,
			bands,
		};
		let page_figure = |gate: &Gate| (gate.word() >> FLAG_BITS) << PAGE_SHIFT;

		gate.publish(published(Some(bands)), || 0, || 0);
		assert!(page_figure(&gate) <= bands.b - bands.n, "the fast path's bound {} over B - N", page_figure(&gate));
		assert!(page_figure(&gate) + (1 << PAGE_SHIFT) > bands.b - bands.n, "rounded down by less than a page");
		assert_eq!(gate.word() & NEAR, 0, "approx + E below N");

		let e = phys::FOLD_ERROR;
		gate.publish(published(Some(bands)), || 0, || bands.n as i64 - e);
		assert_eq!(gate.word() & NEAR, NEAR, "approx + E at N");

		gate.publish(published(None), || 0, || i64::MAX / 2);
		assert_eq!(gate.word() & NEAR, 0, "no bands: never NEAR");
		assert_eq!(page_figure(&gate), (eff >> PAGE_SHIFT) << PAGE_SHIFT, "the gate not running: eff");

		// The fold hook, against a near level no other test's P reaches.
		let shared = GateShared::new();
		shared.band_n.store(1 << 50, Ordering::Relaxed);
		shared.on_fold((1 << 50) - e);
		assert_eq!(shared.word.load(Ordering::Relaxed) & NEAR, NEAR, "a fold to N - E sets it");
		shared.on_fold(0);
		assert_eq!(shared.word.load(Ordering::Relaxed) & NEAR, 0, "a fold far below clears it");
	}

	/// `validate` (design 3.9.2): a zero poll, a near band outside `[0, 1)`,
	/// and -- under `Block` -- a near band that would not keep the settle
	/// target below the near level are refused; `Off` needs no bands. Red with
	/// the band check gone (`nocheck`).
	#[test]
	fn a_gate_whose_bands_cannot_hold_is_refused() {
		let tau = drain_target::ratio();
		let block = |near_frac: f64| {
			let mut config = GateConfig::default();
			config.mode = GateMode::Block;
			config.near_frac = near_frac;
			config
		};
		let refused = |config: GateConfig| matches!(config.validate(), Err(CacheError::InvalidGateConfig));

		assert!(block((1.0 - tau) / 2.0).validate().is_ok());
		assert!(refused(block(1.0 - tau + 0.001)), "tau + near_frac over 1");

		for near_frac in [-0.01, 1.0, f64::NAN, f64::INFINITY] {
			assert!(refused(block(near_frac)), "near_frac {near_frac}");
		}

		let mut off = block(1.0 - tau + 0.001);
		off.mode = GateMode::Off;
		assert!(off.validate().is_ok(), "Off needs no bands");

		let mut zero_poll = block((1.0 - tau) / 2.0);
		zero_poll.poll_interval = Duration::ZERO;
		assert!(refused(zero_poll));
	}

	#[test]
	fn a_gate_state_round_trips_through_its_byte() {
		for state in [
			GateState::Enabled,
			GateState::Off,
			GateState::NotSole,
			GateState::Ungated,
			GateState::Bands,
			GateState::NoStack,
		] {
			assert_eq!(GateState::from_u8(state as u8), state);
		}
	}

	/// The ceiling's arithmetic under both models (design 3.4.1) and with omega
	/// 0, which has none -- `PAPER_DISABLE_SHARED_OVERHEAD=1` sets omega to 0,
	/// and the measured formula would divide by it (the review's panic).
	#[test]
	fn the_key_ceiling_follows_the_model_and_omega_zero_has_none() {
		use MetadataModel::{Measured, PerObject};

		// Per-object: floor((C - ghost) / omega), the plan's (L + 1) * omega > F.
		assert_eq!(key_ceiling(PerObject, 64, 24_576, 10 * 64, 10, 10, 10), 384);
		assert_eq!(key_ceiling(PerObject, 64, 24_576, 10 * 64 + 640, 10, 10, 10), 374, "a ghost's bytes come off the cap");
		assert_eq!(key_ceiling(PerObject, 64, 100, 10 * 64 + 640, 10, 10, 10), 0, "saturating");

		// Measured: refills to the high-water mark free, growth at omega.
		assert_eq!(key_ceiling(Measured, 100, 10_000, 4_000, 0, 30, 50), 90);
		assert_eq!(key_ceiling(Measured, 100, 10_000, 9_000, 0, 30, 50), 50, "reuse up to L_hw");
		assert_eq!(key_ceiling(Measured, 100, 10_000, 20_000, 0, 30, 50), 50, "a step over C_meta: reuse only");
		assert_eq!(key_ceiling(Measured, 100, 10_000, 0, 0, 0, 0), 100);

		// omega 0: no ceiling, whatever the model -- and nothing divides by it.
		for model in [PerObject, Measured] {
			assert_eq!(key_ceiling(model, 0, 10_000, 20_000, 30, 30, 50), u64::MAX);
			assert_eq!(key_ceiling(model, 0, 0, 0, 0, 0, 0), u64::MAX);
		}
	}

	/// The split the size-split stack settles on is the one the worker
	/// publishes: the shares re-sum to the reservation.
	#[test]
	fn the_size_split_shares_re_sum_to_the_reservation() {
		assert_eq!(size_split_shares(1_000, 3_000, 1_000), (750, 250));
		assert_eq!(size_split_shares(1_001, 1, 2), (333, 668));
		assert_eq!(size_split_shares(1_000, 0, 0), (0, 0));

		for (r, s, l) in [(7, 3, 5), (123_457, 99, 1), (0, 5, 5), (u32::MAX as u64, 17, 19)] {
			let (a, b) = size_split_shares(r, s, l);
			assert_eq!(a + b, r, "{r} split {s}:{l}");
		}
	}

	/// The published word: eff in whole pages, and META_NEAR from the count
	/// against the ceiling -- set by a client past it, cleared by the worker
	/// only when the count, read again, is not near.
	#[test]
	fn the_word_carries_eff_in_pages_and_the_near_flag() {
		let gate = Gate::default();
		let publish = |eff: u64, k_max: u64, live: u64| {
			gate.publish(
				Published {
					model: MetadataModel::PerObject,
					m_model: 0,
					eff,
					eff_small: eff,
					eff_large: eff,
					k_max,
					bands: None,
				},
				|| live,
				|| 0,
			)
		};

		publish(3 * 4096 + 4095, 1_000, 10);
		assert_eq!(gate.word(), 3 << FLAG_BITS, "eff rounded down to 3 pages, not near");

		publish(3 * 4096, 1_000, 1_000 - NEAR_KEYS);
		assert_eq!(gate.word() & META_NEAR, META_NEAR, "within NEAR_KEYS of the ceiling");

		publish(3 * 4096, 1_000, 10);
		assert_eq!(gate.word() & META_NEAR, 0, "the worker clears it");

		gate.note_count(10);
		assert_eq!(gate.word() & META_NEAR, 0, "a count far from the ceiling sets nothing");

		gate.note_count(1_000 - NEAR_KEYS);
		assert_eq!(gate.word() & META_NEAR, META_NEAR, "a client's count near it sets the flag");

		// The worker's clear re-reads the count: a client counted in between
		// keeps its flag.
		let calls = std::cell::Cell::new(0);
		gate.publish(
			Published { model: MetadataModel::PerObject, m_model: 0, eff: 0, eff_small: 0, eff_large: 0, k_max: 1_000, bands: None },
			|| {
				calls.set(calls.get() + 1);
				if calls.get() == 1 { 10 } else { 1_000 }
			},
			|| 0,
		);
		assert_eq!(gate.word() & META_NEAR, META_NEAR, "re-checked after the clear");

		// No ceiling: never near.
		publish(0, u64::MAX, u64::MAX - 1);
		assert_eq!(gate.word() & META_NEAR, 0);
	}

	/// The lane is FIFO, and any waiter may leave out of turn -- the head, one
	/// behind it, or one unwinding -- without wedging it: the next waiter
	/// becomes the head, and a new one after that is admitted in turn (the
	/// design review's blocker: a ticket lock never passes a hole).
	#[test]
	fn the_lane_survives_departures_out_of_turn_and_unwinding() {
		let lane = Lane::default();

		let a = lane.enqueue();
		let b = lane.enqueue();
		let c = lane.enqueue();
		assert_eq!(lane.len(), 3);
		assert!(a.is_head() && !b.is_head() && !c.is_head());

		// A waiter behind the head leaves first (an error, a panic).
		drop(c);
		assert_eq!(lane.len(), 2);
		assert!(a.is_head() && !b.is_head());

		// The head leaves: the next is the head.
		drop(a);
		assert!(b.is_head());

		// One unwinds out of its wait: its guard still leaves the lane.
		let d_id = std::thread::scope(|scope| {
			scope
				.spawn(|| {
					let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
						let d = lane.enqueue();
						let id = d.id();
						assert!(!d.is_head());
						panic!("a waiter panics while it waits (id {id})");
					}));
					assert!(result.is_err());
					lane.len()
				})
				.join()
				.unwrap()
		});
		assert_eq!(d_id, 1, "the unwinding waiter left the lane");

		drop(b);
		assert_eq!(lane.len(), 0);

		// A newcomer is at once the head.
		let e = lane.enqueue();
		assert!(e.is_head());
	}

	/// A departing head wakes the next one: a waiter parked for a whole second
	/// is back well before that.
	#[test]
	fn a_departing_head_wakes_the_next() {
		let lane = Arc::new(Lane::default());
		let head = lane.enqueue();

		let (ready_tx, ready_rx) = crossbeam_channel::bounded(1);
		let waiter = {
			let lane = lane.clone();
			std::thread::spawn(move || {
				let me = lane.enqueue();
				ready_tx.send(()).unwrap();
				let start = Instant::now();

				while !me.is_head() {
					park(Duration::from_secs(1));
					assert!(start.elapsed() < Duration::from_secs(10), "never became the head");
				}

				start.elapsed()
			})
		};

		ready_rx.recv().unwrap();
		std::thread::sleep(Duration::from_millis(20));
		drop(head);

		let waited = waiter.join().unwrap();
		assert!(waited < Duration::from_millis(900), "woken by the departure, not the 1 s timeout: {waited:?}");
	}
}
