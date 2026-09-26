/*
 * Copyright (c) Kia Shakiba
 *
 * This source code is licensed under the GNU AGPLv3 license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! The admission path of a tiered cache's `set` (fast-tier backpressure plan
//! S5, commit B1): what is decided from a value's LENGTH, before anything is
//! allocated -- and the figures those decisions read.
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

use std::{
	collections::VecDeque,
	sync::{
		Arc,
		atomic::{AtomicBool, AtomicU64, AtomicU8, AtomicUsize, Ordering},
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
	status::AtomicStatus,
	worker::Placement,
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

/// A tiered cache's admission configuration. Built as `GateConfig::default()`
/// and adjusted field by field (the struct is `non_exhaustive`: S5's second
/// commit adds the byte gate's knobs), then passed to a `*_with_gate`
/// constructor or `PaperCache::set_gate_config`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
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

	/// How long an `EvictToFit` set waits for the policy worker to answer
	/// while the worker makes no progress at all (it counts every event it
	/// handles, so a set queued behind a long backlog keeps waiting while the
	/// backlog drains). 2 s (provisional, design Q1).
	pub stall_window: Duration,

	/// How often a waiting set re-checks when nothing wakes it. 200 us.
	pub poll_interval: Duration,
}

impl Default for GateConfig {
	fn default() -> Self {
		GateConfig {
			metadata_model: MetadataModel::Measured,
			metadata_floor: 0,
			on_metadata_overflow: MetadataOverflow::Error,
			stall_window: Duration::from_secs(2),
			poll_interval: Duration::from_micros(200),
		}
	}
}

impl GateConfig {
	/// `CacheError::InvalidGateConfig` for a configuration no set could run
	/// under: a zero poll interval (a waiting set would spin).
	pub fn validate(&self) -> Result<(), CacheError> {
		if self.poll_interval.is_zero() {
			return Err(CacheError::InvalidGateConfig);
		}

		Ok(())
	}
}

/// The gate word's flag: the object count is within `NEAR_KEYS` of the key
/// ceiling, so a set of a NEW key checks the ceiling. Set by the policy worker
/// at its publication and by a client whose insert brings the count near it;
/// cleared only by the worker.
pub(crate) const META_NEAR: u64 = 1;

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

/// A tiered cache's admission state, in its `AtomicStatus`: the configuration,
/// the figures the policy worker publishes for the client's decisions, the
/// metadata lane, the worker's idle and liveness bits, and the counters.
///
/// What a wipe resets (`reset_counters`): the event counters only. The
/// configuration, the lane and its waiters, the published figures and the
/// worker's bits are live state and survive it.
pub(crate) struct Gate {
	config: parking_lot::RwLock<GateConfig>,

	/// `PAPER_DISABLE_SHARED_OVERHEAD=1` when the cache was built: the
	/// per-object model, whatever the configuration says.
	forced_per_object: AtomicBool,

	/// The word a set reads first: `META_NEAR` in the low byte, eff in pages
	/// above it (the minimum of the two class figures for the size-split
	/// design). A set whose low byte is clear and whose value is at most the
	/// page figure is neither capped nor structural: one relaxed load.
	word: AtomicU64,

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

	/// Test builds: the policy worker panics at its next `MakeRoom`, for the
	/// dead-worker test.
	#[cfg(test)]
	pub(crate) test_panic_on_make_room: AtomicBool,

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
			// Until the policy worker's first publication (at its construction,
			// before the cache is returned): nothing near the ceiling, and room
			// for anything.
			word: AtomicU64::new(u64::MAX << FLAG_BITS),
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
			#[cfg(test)]
			test_panic_on_make_room: AtomicBool::new(false),
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
	pub(crate) fn publish(&self, published: Published, live: impl Fn() -> u64) {
		self.eff.store(published.eff, Ordering::Relaxed);
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

		let word_eff = published.eff.min(published.eff_small).min(published.eff_large);
		let pages = (word_eff >> PAGE_SHIFT).min(u64::MAX >> FLAG_BITS);
		let near = |count: u64| near_ceiling(count, published.k_max);

		let mut near_now = near(live());
		let mut old = self.word.load(Ordering::Relaxed);

		loop {
			let flags = (old & ((1 << FLAG_BITS) - 1)) & !META_NEAR;
			let new = flags | if near_now { META_NEAR } else { 0 } | (pages << FLAG_BITS);

			match self.word.compare_exchange_weak(old, new, Ordering::AcqRel, Ordering::Relaxed) {
				Ok(_) => break,
				Err(actual) => old = actual,
			}
		}

		// A client's `set_near` between the count read above and the store
		// was overwritten; its own increment is in the count by now.
		if !near_now {
			near_now = near(live());

			if near_now {
				self.word.fetch_or(META_NEAR, Ordering::AcqRel);
			}
		}
	}

	/// A client whose new key brought the object count to `count`: flags the
	/// word when that is near the ceiling. One relaxed load when it is not.
	pub(crate) fn note_count(&self, count: u64) {
		if near_ceiling(count, self.k_max()) {
			self.word.fetch_or(META_NEAR, Ordering::AcqRel);
		}
	}

	pub(crate) fn word(&self) -> u64 {
		self.word.load(Ordering::Relaxed)
	}

	/// The policy worker's thread is gone.
	pub(crate) fn worker_gone(&self) -> bool {
		self.worker_gone.load(Ordering::Acquire)
	}

	/// Records the policy worker's exit (`WorkerGoneGuard`) and wakes the
	/// metadata lane's head, which then fails with `Internal`.
	pub(crate) fn mark_worker_gone(&self) {
		self.worker_gone.store(true, Ordering::Release);
		self.meta_lane.wake_head();
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
		] {
			counter.store(0, Ordering::Relaxed);
		}
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
/// from -- `commit` builds exactly the object `begin_set` checked. Crate-only:
/// the client path is `set`, which pairs the two at once.
#[derive(Clone, Copy, Debug)]
pub(crate) struct SetPermit {
	pub(crate) hashed: HashedKey,
	pub(crate) tier: Tier,
	pub(crate) placement: Placement,
	pub(crate) len: usize,
	pub(crate) ttl: Option<u32>,
	pub(crate) sizes: Sizes,
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

	Ok(Verdict::Admit { tier, placement })
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
		}

		LaneGuard { lane: self, slot }
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
		self.status.gate().mark_worker_gone();
	}
}

#[cfg(test)]
mod tests {
	use super::*;

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
				},
				|| live,
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
			Published { model: MetadataModel::PerObject, m_model: 0, eff: 0, eff_small: 0, eff_large: 0, k_max: 1_000 },
			|| {
				calls.set(calls.get() + 1);
				if calls.get() == 1 { 10 } else { 1_000 }
			},
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
