/*
 * Copyright (c) Kia Shakiba
 *
 * This source code is licensed under the GNU AGPLv3 license found in the
 * LICENSE file in the root directory of this source tree.
 */

#[cfg(any(test, not(feature = "merged_object_store")))]
mod lfu_compact_stack;
#[cfg(any(test, not(feature = "merged_object_store")))]
mod lru_compact_stack;
#[cfg(any(test, not(feature = "merged_object_store")))]
mod arc_stack;
#[cfg(any(test, not(feature = "merged_object_store")))]
pub(crate) mod ghost_filter;

#[cfg(any(test, not(feature = "merged_object_store")))]
pub(crate) mod compact_queue_set;
#[cfg(any(test, not(feature = "merged_object_store")))]
pub mod arena_index;
#[cfg(any(test, not(feature = "merged_object_store")))]
pub mod arena_queue_set;
#[cfg(any(test, not(feature = "merged_object_store")))]
pub(crate) mod arena_frequency_chain;

/// The frequency chain `ArenaFrequencyChain` replaces, kept ONLY as the
/// reference implementation that pins it.
///
/// It has no production caller left: both LFU-ranked stacks moved to the arena,
/// which stores the key once instead of twice and measures 40 B/object against
/// this one's 72. It stays in the tree, test-gated, because
/// `arena_frequency_chain`'s two differential tests --
/// `the_frequency_face_agrees_with_the_compact_chain_it_replaces` and
/// `the_recency_face_agrees_with_the_compact_chain_it_replaces` -- drive the
/// two through the same random operation streams and require identical
/// observable state at every step. That is the evidence that the conversion
/// was a representation change and not an algorithm change, and it is worth
/// more in the tree than the module is worth out of it.
#[cfg(test)]
pub(crate) mod compact_frequency_chain;
#[cfg(test)]
mod measure_overhead;
#[cfg(test)]
mod golden;
#[cfg(all(test, feature = "hybrid_cache_common", not(feature = "eviction_stacks_pmem")))]
pub(crate) mod tier_golden;
#[cfg(test)]
mod tier_probes;
#[cfg(any(test, not(feature = "merged_object_store")))]
mod lru_lfu_compact_hybrid_stack;
#[cfg(any(test, not(feature = "merged_object_store")))]
mod tiered_stack;
#[cfg(any(test, not(feature = "merged_object_store")))]
mod lru_fifo_clock_hybrid_stacks;
#[cfg(any(test, not(feature = "merged_object_store")))]
mod lfu_compact_hybrid_stack;
#[cfg(any(test, not(feature = "merged_object_store")))]
mod two_q_hybrid_stacks;
#[cfg(any(test, not(feature = "merged_object_store")))]
mod two_q_fast_admission_reprieve_compact_hybrid_stack;
#[cfg(any(test, not(feature = "merged_object_store")))]
mod two_q_full_fast_admission_compact_hybrid_stack;
#[cfg(any(test, not(feature = "merged_object_store")))]
mod lru_sized_compact_hybrid_stack;
#[cfg(any(test, not(feature = "merged_object_store")))]
mod s3_fifo_hybrid_stacks;
#[cfg(any(test, not(feature = "merged_object_store")))]
mod s3_fifo_faithful_compact_hybrid_stack;
#[cfg(any(test, not(feature = "merged_object_store")))]
mod s3_fifo_ghost_lazy_demotion_fast_admission_compact_hybrid_stack;
#[cfg(any(test, not(feature = "merged_object_store")))]
mod s3_fifo_ghost_lazy_demotion_fast_admission_midpoint_compact_hybrid_stack;
#[cfg(any(test, not(feature = "merged_object_store")))]
mod s3_fifo_lazy_demotion_fast_admission_midpoint_reprieve_compact_hybrid_stack;
#[cfg(any(test, not(feature = "merged_object_store")))]
mod s3_fifo_lazy_demotion_fast_admission_reprieve_compact_hybrid_stack;
#[cfg(any(test, not(feature = "merged_object_store")))]
mod s3_fifo_lazy_demotion_reprieve_compact_hybrid_stack;
#[cfg(any(test, not(feature = "merged_object_store")))]
mod two_q_compact_stack;

/// `PolicyStack` over the merged object store -- the store IS the
/// eviction stack, so this forwards rather than owning anything.
#[cfg(feature = "merged_object_store")]
pub(crate) mod merged_stack;
#[cfg(any(test, not(feature = "merged_object_store")))]
mod s_three_fifo_compact_stack;
#[cfg(any(test, not(feature = "merged_object_store")))]
mod fifo_compact_stack;
#[cfg(any(test, not(feature = "merged_object_store")))]
mod clock_compact_stack;
#[cfg(any(test, not(feature = "merged_object_store")))]
mod sieve_compact_stack;
#[cfg(any(test, not(feature = "merged_object_store")))]
mod mru_compact_stack;
#[cfg(any(test, not(feature = "merged_object_store")))]
mod s3_fifo_lazy_demotion_fast_admission_split_slow_reprieve_compact_hybrid_stack;

use crate::{
	CacheSize,
	HashedKey,
	policy::PaperPolicy,
	object::ObjectSize,
};

// The split stacks `init_policy_stack` builds -- none in a merged build, whose
// stack is the object map itself (`merged_stack`), but for its tests.
#[cfg(any(test, not(feature = "merged_object_store")))]
use crate::{
	worker::policy::policy_stack::{
		lfu_compact_stack::LfuCompactStack,
		fifo_compact_stack::FifoCompactStack,
		clock_compact_stack::ClockCompactStack,
		sieve_compact_stack::SieveCompactStack,
		mru_compact_stack::MruCompactStack,
		two_q_compact_stack::TwoQCompactStack,
		s_three_fifo_compact_stack::SThreeFifoCompactStack,
		s3_fifo_faithful_compact_hybrid_stack::S3FifoFaithfulCompactHybridStack,
		s3_fifo_faithful_compact_hybrid_stack::S3FifoFaithfulFastAdmissionCompactHybridStack,
		s3_fifo_faithful_compact_hybrid_stack::S3FifoFaithfulReprieveCompactHybridStack,
		s3_fifo_faithful_compact_hybrid_stack::S3FifoFaithfulFastAdmissionReprieveCompactHybridStack,
		lru_compact_stack::LruCompactStack,
		arc_stack::ArcStack,
		lru_lfu_compact_hybrid_stack::LruLfuCompactHybridStack,
		lru_fifo_clock_hybrid_stacks::{ClockCompactHybridStack, FifoCompactHybridStack, LruCompactHybridStack},
		lfu_compact_hybrid_stack::LfuCompactHybridStack,
		two_q_hybrid_stacks::{TwoQCompactHybridStack, TwoQGhostCompactHybridStack},
		two_q_fast_admission_reprieve_compact_hybrid_stack::TwoQFastAdmissionReprieveCompactHybridStack,
		two_q_full_fast_admission_compact_hybrid_stack::TwoQFullFastAdmissionCompactHybridStack,
		lru_sized_compact_hybrid_stack::LruSizedCompactHybridStack,
		s3_fifo_hybrid_stacks::{S3FifoCompactHybridStack, S3FifoGhostCompactHybridStack, S3FifoGhostLazyDemotionCompactHybridStack},
		s3_fifo_ghost_lazy_demotion_fast_admission_compact_hybrid_stack::S3FifoGhostLazyDemotionFastAdmissionCompactHybridStack,
		s3_fifo_ghost_lazy_demotion_fast_admission_midpoint_compact_hybrid_stack::S3FifoGhostLazyDemotionFastAdmissionMidpointCompactHybridStack,
		s3_fifo_lazy_demotion_fast_admission_midpoint_reprieve_compact_hybrid_stack::S3FifoLazyDemotionFastAdmissionMidpointReprieveCompactHybridStack,
		s3_fifo_lazy_demotion_fast_admission_reprieve_compact_hybrid_stack::S3FifoLazyDemotionFastAdmissionReprieveCompactHybridStack,
		s3_fifo_lazy_demotion_reprieve_compact_hybrid_stack::S3FifoLazyDemotionReprieveCompactHybridStack,
		s3_fifo_lazy_demotion_fast_admission_split_slow_reprieve_compact_hybrid_stack::S3FifoLazyDemotionFastAdmissionSplitSlowReprieveCompactHybridStack,
	},
};

/// The level the fast tier is continuously held at, as a fraction of its
/// effective budget: 0.95 by default (E1b; it was 0.98 until then).
///
/// `settle_fast_tier` demotes whenever `fast_used` is above
/// `ratio * effective_capacity` and stops the moment it is back at it. ONE
/// threshold, not a band: there is no arm-here / drain-to-there gap, so a
/// settle moves only what the admission or promotion that triggered it
/// displaced, and the tier steady-states just under its budget rather than
/// sawtoothing between two marks.
///
/// The margin is burst headroom, and it is the whole reason the ratio is not
/// 1.0. `PaperCache::set()` writes a new object's bytes to DRAM synchronously
/// at the API layer, before the event reaches `PolicyWorker` at all; and a
/// demotion this function decides is not physically applied until a
/// `migration_queue` consumer runs it. Real DRAM therefore sits above what the
/// stack believes for as long as those two windows last. Held at exactly its
/// ceiling, a tier has nowhere for that overshoot to go but outside the
/// budget; held at 0.95, it lands inside.
///
/// Why 0.95, and not the 0.98 this was: the user asked for more room to
/// absorb bursts. The byte gate (S5) holds a settled tier at
/// `S = ratio * eff` and admits a fast set up to `B = eff + slack`, so what a
/// burst can land in before the gate closes is `B - S`: 5% of `eff` at 0.95
/// (with the default zero slack), 2% at 0.98. It is the depth of the margin
/// that changed, not its shape: one continuous level, one object at a time.
///
/// The band this replaced is the counter-example, and what was wrong with it
/// was not its 5% of depth. That 0.98/0.95 high/low pair
/// (`FAST_TIER_LOW_WATER_RATIO`; see `LRU_HYBRID_CACHE.md`) ARMED at 0.98 and
/// only then drained to 0.95 in one go, which emitted the drop as one burst
/// of demotions -- ~3% of the budget in a single `apply_tier_migrations`
/// batch -- on every pass. A single threshold at 0.95 pays the same 5% of the
/// tier once, as a standing margin, and never emits that burst. The single
/// threshold first kept the band's upper mark (0.98) as its level; this takes
/// the lower one, on purpose.
///
/// The gate's bands follow the ratio, and `GateConfig::validate` requires
/// `ratio + near_frac < 1` so the settle target stays below the near level:
/// at 0.95 the near band may be up to (just under) 5% of `eff`, against the
/// default 1%.
///
/// `FAST_TIER_DRAIN_TARGET=0.98` restores the previous default and `=1.0`
/// holds the tier at exactly its ceiling, which is the no-headroom behaviour.
/// The published results were measured at 0.98; S10 rebaselines them.
#[cfg(any(test, not(feature = "merged_object_store"), feature = "hybrid_cache_common"))]
pub mod drain_target {
	use std::sync::OnceLock;

	pub const DEFAULT_RATIO: f64 = 0.95;

	static RATIO: OnceLock<f64> = OnceLock::new();

	/// Read once through a `OnceLock`, so the env var is startup configuration
	/// rather than runtime adjustable. A value that fails to parse or falls
	/// outside `(0.0, 1.0]` is silently replaced by the default.
	pub fn ratio() -> f64 {
		*RATIO.get_or_init(|| {
			std::env::var("FAST_TIER_DRAIN_TARGET")
				.ok()
				.and_then(|v| v.parse::<f64>().ok())
				.filter(|v| *v > 0.0 && *v <= 1.0)
				.unwrap_or(DEFAULT_RATIO)
		})
	}

	/// The byte level a settle holds the tier at: both the point above which it
	/// demotes and the point it stops at.
	pub fn bytes(effective_capacity: u64) -> u64 {
		(effective_capacity as f64 * ratio()) as u64
	}
}

/// Which tier an object currently lives in, for policy stacks that track a
/// segmented (fast/slow) queue. Used by `LruCompactHybridStack`
/// (`PaperPolicy::LruCompactHybrid`, recency-segmented),
/// `LfuCompactHybridStack` (`PaperPolicy::LfuCompactHybrid`,
/// frequency-segmented), `TwoQCompactHybridStack`
/// (`PaperPolicy::TwoQCompactHybrid`, 2Q-segmented), and
/// `FifoCompactHybridStack` (`PaperPolicy::FifoCompactHybrid`,
/// insertion-order-segmented); every other stack's default
/// `drain_tier_migrations` never produces one.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Tier {
	Fast,
	Slow,
}

/// Who queued a tier migration: the policy stack, as a POLICY decision -- a
/// promotion or a demotion in the paper's sense -- or the reconcile, as a
/// CORRECTIVE that moves a value's bytes to where the stack already places
/// its key (the policy worker's correctives, `Observed` in
/// `worker/policy/mod.rs` -- the only ones, in every store). The tag travels with
/// the entry through `split_tier_migrations` and the migration queue, so a
/// completed corrective is counted as `RECONCILE_APPLIED_TO_*` and never as a
/// promotion or a demotion: it displaced nothing.
#[cfg(any(feature = "hybrid_cache_common", feature = "merged_object_store"))]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MigrationOrigin {
	Stack,
	Reconcile,
}

/// A migration with its origin: `(key, destination, origin)`. Sixteen bytes,
/// the size of the untagged `(key, destination)` pair it replaced on the
/// queue.
#[cfg(any(feature = "hybrid_cache_common", feature = "merged_object_store"))]
pub type TaggedMigration = (HashedKey, Tier, MigrationOrigin);

/// A drain entry, tagged or not: what `split_tier_migrations`, the batch
/// appliers and `MigrationQueue::push` read of it. An untagged `(key, tier)`
/// -- what every stack's `drain_tier_migrations` returns -- is the stack's own
/// decision (`MigrationOrigin::Stack`).
#[cfg(any(feature = "hybrid_cache_common", feature = "merged_object_store"))]
pub trait MigrationEntry: Copy + Send + Sync {
	fn key(&self) -> HashedKey;
	fn tier(&self) -> Tier;
	fn origin(&self) -> MigrationOrigin;

	fn tagged(&self) -> TaggedMigration {
		(self.key(), self.tier(), self.origin())
	}
}

#[cfg(any(feature = "hybrid_cache_common", feature = "merged_object_store"))]
impl MigrationEntry for (HashedKey, Tier) {
	fn key(&self) -> HashedKey {
		self.0
	}

	fn tier(&self) -> Tier {
		self.1
	}

	fn origin(&self) -> MigrationOrigin {
		MigrationOrigin::Stack
	}
}

#[cfg(any(feature = "hybrid_cache_common", feature = "merged_object_store"))]
impl MigrationEntry for TaggedMigration {
	fn key(&self) -> HashedKey {
		self.0
	}

	fn tier(&self) -> Tier {
		self.1
	}

	fn origin(&self) -> MigrationOrigin {
		self.2
	}
}

/// Narrows a DRAM-resident remainder so it fits an entry's spare padding byte.
///
/// The remainder is `key + expiry field (16) + Expiries entry (64 with a TTL)`,
/// so `u8` covers every key up to 175 bytes -- and the benchmark's keys are
/// pre-hashed `u64`s. Saturating is safe rather than merely convenient: any
/// excess is then treated as migrating, which is exactly the behaviour before
/// this accounting existed, so it degrades toward the old over-charge instead
/// of going wrong in a new way.
#[inline]
#[cfg(any(test, not(feature = "merged_object_store")))]
pub(crate) fn narrow_resident(resident: ObjectSize) -> u8 {
	resident.min(u8::MAX as ObjectSize) as u8
}

/// What a `Set` did to the object map, as the policy worker tells a stack in
/// `PolicyStack::insert_set`: the map insert replaced nothing (`Fresh`), or
/// it replaced a value -- and whether the base size changed (`resized`), the
/// DashMap FIFO and CLOCK stacks' criterion for settling after an overwrite
/// (they compare their stored size, the previous value's, with the new one).
/// Built from the event's `previous` (`PolicyWorker::handle_set`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SetEvent {
	Fresh,
	Replaced { resized: bool },
}

/// Where a set's value was built, and why, as its `Set` tells the stack (S5).
/// One byte, in the event's existing padding.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[repr(u8)]
pub enum Placement {
	/// Built where the design's admission rule said.
	#[default]
	Normal,

	/// Larger than an EMPTY fast tier (`v > eff`, eff = 0 when metadata fills
	/// the tier): built slow, and every stack places the key slow, keeps it in
	/// its place in the policy's order, and never promotes it while it stays
	/// that large (`gate::decide`, step 3).
	Structural,

	/// Built slow by the byte gate's opt-in `OnStall::Divert`: the fast tier
	/// freed nothing for the gate's window (S5, commit B2). Placed by the
	/// design's policy exactly as a `Normal` set -- a divert changes where the
	/// bytes are, not the policy -- so the key is typically LAGGING (in CXL,
	/// placed fast) until its first slow-served hit heals it; the worker's
	/// reconcile never corrects it toward fast at its `Set`, and drops a
	/// promotion the stack queued for it there (`Observed::Diverted`). Built
	/// only by the byte gate: never in a build without tiers.
	#[cfg_attr(not(feature = "hybrid_cache_common"), allow(dead_code))]
	Diverted,
}

/// The nearest key at or before `start`, toward the front of its queue, that
/// `is_stop` accepts. The boundary walk every design with a tier-boundary
/// cursor takes since S5: a structural key keeps its place in the order with
/// tier Slow, so the cursor -- the least-recently-used FAST key -- steps over
/// slow keys to the next fast one. Amortized O(1) per structural key: a
/// cursor only moves toward the front, and a key it stepped over stays behind
/// it until a hit brings it back to the front.
#[cfg(any(test, not(feature = "merged_object_store")))]
pub(crate) fn walk_to<P: Copy>(
	list: &arena_queue_set::ArenaQueueSet<P>,
	mut start: Option<HashedKey>,
	is_stop: impl Fn(&P) -> bool,
) -> Option<HashedKey> {
	while let Some(key) = start {
		match list.payload(key) {
			Some(payload) if is_stop(&payload) => return Some(key),
			_ => start = list.before(key),
		}
	}

	None
}

/// `walk_to` for a `NodePayload` list whose cursor names the least-recently-
/// used FAST key: the nearest key at or before `start` whose tier is Fast.
#[cfg(any(test, not(feature = "merged_object_store")))]
pub(crate) fn fast_at_or_before(
	list: &arena_queue_set::ArenaQueueSet<arena_queue_set::NodePayload>,
	start: Option<HashedKey>,
) -> Option<HashedKey> {
	walk_to(list, start, |payload| payload.tier == Some(Tier::Fast))
}

/// The cursor's step off `key`: the nearest FAST key before it.
#[cfg(any(test, not(feature = "merged_object_store")))]
pub(crate) fn prev_fast(
	list: &arena_queue_set::ArenaQueueSet<arena_queue_set::NodePayload>,
	key: HashedKey,
) -> Option<HashedKey> {
	fast_at_or_before(list, list.before(key))
}

/// The placement a stack applied (`PolicyStack::insert_placed`'s answer).
#[cfg(any(test, not(feature = "merged_object_store")))]
pub(crate) fn placed(structural: bool) -> Placement {
	match structural {
		true => Placement::Structural,
		false => Placement::Normal,
	}
}

/// How many second chances a CLOCK hand may grant in one `evict_one` before
/// it evicts whatever is at the tail: `2 * len + 8`. A liveness guard, not a
/// policy: sequentially a hand makes at most as many second chances as there
/// are set bits, since each one clears a bit and nothing inside `evict_one`
/// sets one -- and every stack's bits are set only on the policy worker, the
/// thread running the hand. Shared by `ClockCompactHybridStack` and the
/// merged store's `clock_victim`, so both stop at the same point.
pub(crate) fn clock_hand_budget(len: usize) -> usize {
	#[cfg(test)]
	if let Some(budget) = hand_budget_override::get() {
		return budget;
	}

	len.saturating_mul(2).saturating_add(8)
}

/// Test support: `clock_hand_budget`'s answer, overridden for the calling
/// thread while `with` runs -- how a test shows a hand that stops at its
/// budget, which no real sequence reaches.
#[cfg(test)]
pub(crate) mod hand_budget_override {
	use std::cell::Cell;

	thread_local! {
		static BUDGET: Cell<Option<usize>> = const { Cell::new(None) };
	}

	pub(crate) fn get() -> Option<usize> {
		BUDGET.with(Cell::get)
	}

	#[cfg(feature = "hybrid_cache_common")]
	pub(crate) fn with<R>(budget: usize, f: impl FnOnce() -> R) -> R {
		let previous = BUDGET.with(|cell| cell.replace(Some(budget)));
		let out = f();
		BUDGET.with(|cell| cell.set(previous));
		out
	}
}

pub trait PolicyStack
where
	Self: Send,
{
	#[cfg_attr(not(test), expect(dead_code, reason = "the stacks' tests only, since R1 removed the mini stacks"))]
	fn is_policy(&self, policy: &PaperPolicy) -> bool;
	fn len(&self) -> usize;

	#[cfg_attr(not(test), expect(dead_code, reason = "the stacks' tests only, since R1 removed the mini stacks"))]
	fn contains(&self, key: HashedKey) -> bool;
	fn insert(&mut self, key: HashedKey, size: ObjectSize);

	/// `insert`, plus the part of `size` that stays in DRAM whichever tier the
	/// object lands in (key, expiry field, and the `Expiries` entry when a TTL
	/// is set -- see `OverheadManager::dram_resident_size`).
	///
	/// Only the hybrid stacks care: they must keep `fast_used` / `slow_used` to
	/// bytes that actually migrate, since `Object::set_data` moves the value
	/// buffer alone. All-DRAM stacks have no tiers and ignore it.
	fn insert_resident(&mut self, key: HashedKey, size: ObjectSize, dram_resident: ObjectSize) {
		let _ = dram_resident;
		self.insert(key, size);
	}

	/// The policy worker's handling of a `Set`: `insert_resident`, told what
	/// the map insert did (`SetEvent`). Every stack that owns its own index
	/// already knows whether it tracks the key and its previous size, so the
	/// default ignores `event`. The merged store's handle is the one override:
	/// its index is the object map, which the client has already written, so
	/// the event is how it tells a first `Set` from an overwrite -- see
	/// `MergedStore::worker_set`.
	fn insert_set(&mut self, key: HashedKey, size: ObjectSize, dram_resident: ObjectSize, event: SetEvent) {
		let _ = event;
		self.insert_resident(key, size, dram_resident);
	}

	/// The policy worker's handling of a `Set` (S5): `insert_set`, told where
	/// the client placed the value (`Placement`). Returns the placement the
	/// stack APPLIED: `Structural` also when the stack's own check -- the value
	/// larger than its current eff -- made a `Normal` set structural (eff
	/// moved between the client's decision and this), which the worker counts.
	/// A flat stack has no tiers and ignores it.
	fn insert_placed(
		&mut self,
		key: HashedKey,
		size: ObjectSize,
		dram_resident: ObjectSize,
		event: SetEvent,
		placement: Placement,
	) -> Placement {
		self.insert_set(key, size, dram_resident, event);
		placement
	}

	/// S5: the metadata figure the stack reserves out of its fast capacity,
	/// pushed by the policy worker at its publication under the MEASURED model
	/// (`Some(M)`, replacing `len x omega` and a ghost's bytes); `None` under
	/// the per-object model restores today's reservation. The carve-out
	/// designs split it between their segments exactly as they split the
	/// per-object one. Flat stacks reserve nothing.
	fn set_dram_metadata(&mut self, _measured: Option<CacheSize>) {}

	/// S5: every settle the stack runs, against its current budget -- the
	/// policy worker's end-of-pass step (and `MakeRoom`'s), so a design whose
	/// new-key path never settles (LFU latched, the slow admission queues)
	/// still rests at or under its drain target once the reservation grew. A
	/// settle under its target returns at its first comparison. Flat stacks
	/// have none.
	fn resettle(&mut self) {}

	fn update(&mut self, _key: HashedKey) {}
	fn record_access(&mut self, key: HashedKey, hit: bool) {
		if hit {
			self.update(key);
		}
	}
	fn remove(&mut self, key: HashedKey);

	/// Whether `remove` removes only a DELETED entry and never a live key's,
	/// so `PolicyWorker::handle_expire` calls it whatever the object map holds
	/// by then. `false` for every stack but the merged store's handle, whose
	/// `remove` retires one DEAD slot of the key (`MergedStore::retire_dead`):
	/// the others keep one entry per key, which after a re-set belongs to the
	/// live value, so `handle_expire` guards on the map no longer holding it.
	fn remove_is_retire(&self) -> bool {
		false
	}

	fn resize(&mut self, _size: CacheSize) {}
	fn clear(&mut self);

	fn evict_one(&mut self) -> Option<HashedKey>;

	/// Runtime-adjusts the fast-tier byte budget. No-op for every stack
	/// except the hybrid stacks, which shrinking may trigger immediate
	/// demotions for (see `drain_tier_migrations`).
	fn resize_fast_tier(&mut self, _size: CacheSize) {}

	/// Drains and returns every (key, new tier) pair that crossed the
	/// fast/slow boundary since the last call. Only the hybrid stacks
	/// (`LruCompactHybridStack`, `LfuCompactHybridStack`,
	/// `TwoQCompactHybridStack`, ...) ever produce entries; every other stack
	/// keeps the default empty `Vec`. The caller
	/// (`PolicyWorker`) is responsible for physically migrating each
	/// returned key's object bytes to `new_tier`.
	///
	/// In emission order, and the order is meaningful: the caller drops every
	/// entry that a LATER entry of the same drain supersedes by naming the
	/// other tier (`split_tier_migrations`), so a stack may report a
	/// transition it then reverses within the same drain and only the
	/// reversal is applied.
	fn drain_tier_migrations(&mut self) -> Vec<(HashedKey, Tier)> {
		Vec::new()
	}

	/// `drain_tier_migrations`, each entry tagged with who queued it. Every
	/// entry a stack drains is its own policy decision
	/// (`MigrationOrigin::Stack`), so the default tags the drain and does
	/// nothing else -- in place: the tagged entry is the same size. The
	/// merged store's handle overrides it only because its log is kept tagged
	/// already (`MigrationLog`).
	#[cfg(any(feature = "hybrid_cache_common", feature = "merged_object_store"))]
	fn drain_tagged_migrations(&mut self) -> Vec<TaggedMigration> {
		self.drain_tier_migrations().into_iter().map(|entry| entry.tagged()).collect()
	}

	/// Where this stack places `key`'s BYTES once every migration it has
	/// queued so far is applied -- its PHYSICAL intent -- or `None` if it does
	/// not track the key. `None` for every all-DRAM stack, which has no tiers.
	///
	/// Read by the policy worker after the stack has handled a `Set` or a hit
	/// served from the slow tier, to queue a corrective migration toward it
	/// when the bytes are elsewhere (the reconcile and the heal: `Observed` in
	/// `worker/policy/mod.rs`), and by the placement audit, which reports every
	/// live value whose bytes are not where this says. So it must be what the
	/// design CONVERGES the bytes to, not a logical classification: a tier the
	/// design deliberately keeps the bytes out of would have the reconcile
	/// fight the design, copying on every set what the design chose not to
	/// copy.
	///
	/// For every design that is the tier the stack records for the key,
	/// because every design pushes the migration for a tier change in the same
	/// call that makes it (a promotion after the settle that may undo it,
	/// guarded on the key still being fast; a promotion out of a DRAM-resident
	/// queue pushes nothing because the bytes are already there). Per design:
	///
	/// | design | `placement_of` |
	/// |---|---|
	/// | LRU, FIFO, CLOCK, LFU, LRU-LFU | `tier_of` |
	/// | size-split LRU | `tier_of`: the tier of the key's queue |
	/// | 2Q, 2Q-ghost | `tier_of`: the admission FIFO slow, main by tier |
	/// | 2Q fast admission reprieve | `tier_of`: the admission FIFO fast, main by tier |
	/// | full 2Q (fast admission) | `tier_of`: `a1_in` fast, `a1_out` slow, `am` by tier |
	/// | S3-FIFO, ghost, ghost lazy demotion, lazy demotion reprieve | `tier_of`: one-access queue slow, main by tier |
	/// | the five S3-FIFO fast-admission designs | `tier_of`: one-access queue fast, main by tier (split slow: by segment) |
	/// | faithful S3-FIFO, the four variants | `tier_of`: the small queue fast or slow per variant, main by tier |
	/// | merged store (LRU, FIFO, CLOCK, LFU) | `MergedStore::tier_of`: the slot's tier |
	///
	/// The S3-FIFO "lazy demotion" and "reprieve" designs are no exception:
	/// there the laziness is the POLICY's, and whatever it decides is pushed
	/// at once. A key the settle reprieves (referenced since it was promoted)
	/// is not demoted at all, and keeps `Tier::Fast` and its bytes; a key
	/// spliced into main's slow segment instead of being evicted is pushed
	/// slow if it leaves a DRAM queue, and moves no bytes if it leaves a slow
	/// one.
	fn placement_of(&self, _key: HashedKey) -> Option<Tier> {
		None
	}

	/// DRAM reserved for shared per-object metadata across *both* tiers
	/// (`tracked objects x shared_overhead`), taken off `fast_capacity` before
	/// the drain target is applied.
	///
	/// Every tracked object is charged, whichever tier its value is in. A
	/// demotion moves the value bytes and nothing else, so a slow object's
	/// object-map row, eviction-stack node and value header are still in DRAM
	/// -- and `get_hybrid_dram_shared_overhead` already leaves out whichever of
	/// those a build places in PMEM, so the count must not discount them again
	/// by tier. `MergedStore` charges the same, `linked() x shared_overhead`:
	/// the keys its policy worker has linked, which is what a DashMap stack's
	/// `len()` counts.
	/// Charging fast objects only was tried and reverted: on cluster35 (DashMap
	/// LRU, 5 GiB fast tier) it reserved 313.6 MB against 896.7 MB of real
	/// metadata, and fast data plus metadata came to 5,851 MB in a 5,369 MB
	/// tier.
	///
	/// A reservation that meets or exceeds `fast_capacity` leaves every value
	/// budget derived from the fast tier at 0 -- every stack subtracts it
	/// saturating. That is the true DRAM state of a metadata-bound tier, not an
	/// accounting fault, but what follows differs by design. Most demote every
	/// value. The seven 2Q/S3-FIFO fast-admission designs also close DRAM
	/// admission: their admission queue is a carve-out of the tier clamped to
	/// it, and it pays a share of the reservation -- in proportion to its part
	/// of the tier in six of them, and in `two_q_full_fast_admission`, which
	/// charges main first, whatever the main queue cannot absorb. With no
	/// budget left it evicts each new key on arrival where its overflow is an
	/// eviction (the S3-FIFO ghost variants, into the ghost) and sends it to
	/// PMEM where its overflow is a demotion or reprieve (the reprieve
	/// variants at once, `two_q_full_fast_admission` on the next admission).
	/// The S3-FIFO ghost variants lose only a key's first arrival; its second
	/// goes from the ghost straight into main. The faithful S3-FIFO
	/// fast-admission variants are the exception: their DRAM small queue has no
	/// ceiling at all, so its values stay in DRAM on top of the reservation.
	///
	/// `fast_bytes_used` counts object bytes only, so the fast tier's true DRAM
	/// footprint is the two summed. `0` on all-DRAM stacks, which have no tiers
	/// and reserve nothing.
	fn dram_reserved_bytes(&self) -> CacheSize {
		0
	}

	/// S5a: the usable bytes (jemalloc size classes) allocated right now for
	/// this stack's OWN structures -- its slab chunks and their table, its
	/// keyless index, its free list, its LFU bucket maps, its ghost -- by the
	/// node they are on: DRAM, or the slow node under `eviction_stacks_pmem`,
	/// which puts every one of them there. Not the box the stack lives in (the
	/// policy worker adds that) and not the `migrations` vector a drain takes
	/// whole.
	///
	/// What the per-object model charges as the stack's share of `omega`, but
	/// counted from the structures, at whatever load they are at (see
	/// `crate::meta`). Each structure counts itself where it grows, so this is
	/// a handful of loads, cheap enough for the policy worker to compare after
	/// every event. `None` for a stack that does not meter itself -- the flat
	/// (all-DRAM) stacks, which only flat caches run, and a flat cache
	/// publishes no M. Every tiered design returns `Some`.
	fn structure_bytes(&self) -> Option<crate::meta::NodeBytes> {
		None
	}

	/// Current bytes accounted to the fast tier. `0` for every stack except
	/// the hybrid stacks.
	fn fast_bytes_used(&self) -> CacheSize {
		0
	}

	/// Current bytes accounted to the slow tier. `0` for every stack except
	/// the hybrid stacks.
	fn slow_bytes_used(&self) -> CacheSize {
		0
	}

	/// Current number of objects in the fast tier. `0` for every stack
	/// except the hybrid stacks.
	fn fast_object_count(&self) -> usize {
		0
	}

	/// Current number of objects in the slow tier. `0` for every stack
	/// except the hybrid stacks.
	fn slow_object_count(&self) -> usize {
		0
	}

	/// Returns `true` if this stack has an internal sub-structure over its
	/// own capacity budget and wants `apply_evictions` to keep calling
	/// `evict_one()` even though overall `status.used_size()` is still
	/// within `max_size`. Only the 2Q/S3-FIFO hybrid stacks override this
	/// (`TwoQCompactHybridStack` and its siblings -- their `fifo_queue` has
	/// its own `k_in`-derived byte budget, independent of
	/// — and often much tighter than — the overall cache capacity); every
	/// other stack keeps the default `false`. Unlike `drain_tier_migrations`,
	/// which the stack can safely apply to its own bookkeeping in-place, an
	/// eviction needs the caller to also remove the object from the shared
	/// object map and adjust `status`, which only `apply_evictions`'s
	/// `evict_one()` + `erase()` pairing does correctly — a stack must never
	/// silently drop a key from its own bookkeeping without going through
	/// that path, or the object map and the stack's view of the world
	/// desync permanently.
	fn needs_capacity_eviction(&self) -> bool {
		false
	}

	/// Drains and returns the number of genuine demotions (fast-tier objects
	/// moved to slow due to capacity pressure) recorded since the last call.
	/// Distinct from `drain_tier_migrations`'s `Tier::Slow` entries: for
	/// `LfuCompactHybridStack`, a `Tier::Slow` migration can *also* be a fresh
	/// admission routed directly to slow because the fast tier was already
	/// full — that still needs the same physical `Object::set_data`
	/// correction (the object was initially built as `Fast` by the API
	/// layer), but it isn't a demotion in the paper's sense (no existing
	/// fast-tier object was displaced). `LruCompactHybridStack` /
	/// `TwoQCompactHybridStack` never produce that ambiguity (their admission
	/// never lands fast
	/// unconditionally then needs correcting), so they keep the default `0`
	/// and their callers keep counting every `Tier::Slow` migration as a
	/// demotion directly.
	/// Whether `PolicyWorker` should count each applied `Tier::Slow`
	/// migration as a demotion. The LFU-style design returns `false`: its
	/// `Tier::Slow` entries are not always genuine demotions, and the true
	/// count comes from [`Self::drain_demotions`] instead.
	fn inline_demotion_accounting(&self) -> bool {
		true
	}

	fn drain_demotions(&mut self) -> u64 {
		0
	}

	/// Returns `true` if this stack has permanently closed brand-new-key
	/// admission to the fast tier (see `LfuCompactHybridStack`'s module doc
	/// for why a one-time latch is needed on top of a raw byte-capacity
	/// check).
	/// Every other stack keeps the default `false` — `LruCompactHybridStack`
	/// and `TwoQCompactHybridStack`'s admission rules don't have this
	/// ambiguity (LRU always lands fast; 2Q-hybrid always lands slow), so
	/// there's nothing
	/// to latch. `PolicyWorker` mirrors this onto `AtomicStatus` so the
	/// API-calling thread — which has no access to the stack itself, owned
	/// exclusively by the worker thread — can decide a brand-new key's
	/// physical tier placement (`TieredBuffer::new_fast` vs. `new_slow`) up
	/// front in `PaperCache::set()`, instead of always guessing fast and
	/// relying on an async correction.
	fn admission_latched(&self) -> bool {
		false
	}

	/// Runtime-adjusts the LARGE fast segment's byte budget. Only
	/// `LruSizedCompactHybridStack` overrides this -- the SMALL segment reuses
	/// `resize_fast_tier` above; every other stack keeps the default no-op.
	fn resize_large_fast_tier(&mut self, _size: CacheSize) {}

	/// Runtime-adjusts the small/large size-classification threshold. Only
	/// `LruSizedCompactHybridStack` overrides this; every other stack keeps the
	/// default no-op.
	fn resize_size_threshold(&mut self, _size: CacheSize) {}

	/// Current bytes accounted to the SMALL fast segment. `0` for every
	/// stack except `LruSizedCompactHybridStack`.
	fn small_fast_bytes_used(&self) -> CacheSize {
		0
	}

	/// Current bytes accounted to the LARGE fast segment. `0` for every
	/// stack except `LruSizedCompactHybridStack`.
	fn large_fast_bytes_used(&self) -> CacheSize {
		0
	}

	/// Current number of objects in the SMALL fast segment. `0` for every
	/// stack except `LruSizedCompactHybridStack`.
	fn small_fast_object_count(&self) -> usize {
		0
	}

	/// Current number of objects in the LARGE fast segment. `0` for every
	/// stack except `LruSizedCompactHybridStack`.
	fn large_fast_object_count(&self) -> usize {
		0
	}

	/// Current bytes accounted to the SMALL slow list. `0` for every stack
	/// except `LruSizedCompactHybridStack`.
	fn small_slow_bytes_used(&self) -> CacheSize {
		0
	}

	/// Current bytes accounted to the LARGE slow list. `0` for every stack
	/// except `LruSizedCompactHybridStack`.
	fn large_slow_bytes_used(&self) -> CacheSize {
		0
	}

	/// Current number of objects in the SMALL slow list. `0` for every
	/// stack except `LruSizedCompactHybridStack`.
	fn small_slow_object_count(&self) -> usize {
		0
	}

	/// Current number of objects in the LARGE slow list. `0` for every
	/// stack except `LruSizedCompactHybridStack`.
	fn large_slow_object_count(&self) -> usize {
		0
	}
}

#[cfg(any(test, not(feature = "merged_object_store")))]
pub fn init_policy_stack(policy: PaperPolicy, max_size: CacheSize) -> Box<dyn PolicyStack> {
	match policy {
		PaperPolicy::LfuCompact => Box::new(LfuCompactStack::default()),
		PaperPolicy::FifoCompact => Box::new(FifoCompactStack::default()),
		PaperPolicy::ClockCompact => Box::new(ClockCompactStack::default()),
		PaperPolicy::SieveCompact => Box::new(SieveCompactStack::default()),
		PaperPolicy::MruCompact => Box::new(MruCompactStack::default()),
		PaperPolicy::LruCompact => Box::new(LruCompactStack::default()),
		PaperPolicy::TwoQCompact(k_in, k_out) => Box::new(TwoQCompactStack::new(k_in, k_out, max_size)),
		PaperPolicy::Arc => Box::new(ArcStack::new(max_size)),
		PaperPolicy::SThreeFifoCompact(ratio) => Box::new(SThreeFifoCompactStack::new(ratio, max_size)),

		// Default fast-tier budget is 20% of the overall cache size, matching
		// the tiering manager's default `dram_threshold` ratio (see
		// `TieringManager::new` in lib.rs). Runtime-adjustable afterward via
		// `resize_fast_tier` / `PaperCache::set_fast_tier_size` (step 10).
		// `with_shared_overhead` reserves the DRAM cost of the shared object
		// hashtable + eviction stacks out of that budget so demotion bounds
		// total DRAM, not just fast-tier values. Every hybrid arm below is built
		// the same way.
		//
		// `promote_k` comes from the policy value itself rather than a default
		// here, since it is carried in the policy string.
		#[cfg(feature = "hybrid_cache_common")]
		PaperPolicy::LruLfuCompactHybrid(promote_k) => Box::new(
			LruLfuCompactHybridStack::new((max_size as f64 * 0.2) as CacheSize, promote_k)
				.with_shared_overhead(
					crate::object::overhead::get_hybrid_dram_shared_overhead(&policy) as CacheSize,
				),
		),

		// Same default fast-tier budget/override mechanism as above. The
		// reservation is not optional: without it the stack gets a larger
		// effective fast tier than every policy it is compared against, which is
		// exactly how the first run of this variant produced a flattering and
		// meaningless result.
		#[cfg(feature = "hybrid_cache_common")]
		PaperPolicy::LruCompactHybrid => Box::new(
			LruCompactHybridStack::new((max_size as f64 * 0.2) as CacheSize)
				.with_shared_overhead(
					crate::object::overhead::get_hybrid_dram_shared_overhead(&policy) as CacheSize,
				),
		),

		#[cfg(not(feature = "hybrid_cache_common"))]
		PaperPolicy::LruCompactHybrid =>
			Box::new(LruCompactHybridStack::new((max_size as f64 * 0.2) as CacheSize)),

		#[cfg(feature = "hybrid_cache_common")]
		PaperPolicy::LfuCompactHybrid => Box::new(
			LfuCompactHybridStack::new((max_size as f64 * 0.2) as CacheSize)
				.with_shared_overhead(
					crate::object::overhead::get_hybrid_dram_shared_overhead(&policy) as CacheSize,
				),
		),

		// k_in comes from the policy string itself (same as plain `TwoQ`);
		// the fast-tier budget still defaults to 20% of max_size, same
		// override mechanism as the other hybrids.
		#[cfg(feature = "hybrid_cache_common")]
		PaperPolicy::TwoQCompactHybrid(k_in) => Box::new(
			TwoQCompactHybridStack::new(k_in, max_size, (max_size as f64 * 0.2) as CacheSize)
				.with_shared_overhead(
					crate::object::overhead::get_hybrid_dram_shared_overhead(&policy) as CacheSize,
				),
		),

		#[cfg(not(feature = "hybrid_cache_common"))]
		PaperPolicy::TwoQCompactHybrid(k_in) => Box::new(
			TwoQCompactHybridStack::new(k_in, max_size, (max_size as f64 * 0.2) as CacheSize),
		),

		// Same construction shape as `TwoQCompactHybrid` above. Note the default
		// fast-tier budget matters more here: `fifo_capacity` (k_in *
		// max_size) is carved *out of* it rather than being an independent
		// PMEM budget, so at the 20% default a k_in above 0.2 would leave
		// the main queue no fast segment at all. That is a legitimate
		// configuration (see the stack's module doc), and callers override
		// the budget via `ResizeFastTier` immediately after construction
		// anyway, but it is worth knowing when picking k_in.
		#[cfg(feature = "hybrid_cache_common")]
		PaperPolicy::TwoQFastAdmissionReprieveCompactHybrid(k_in) => Box::new(
			TwoQFastAdmissionReprieveCompactHybridStack::new(k_in, max_size, (max_size as f64 * 0.2) as CacheSize).with_shared_overhead(
				crate::object::overhead::get_hybrid_dram_shared_overhead(&policy) as CacheSize,
			),
		),

		// The full three-queue 2Q -- the only design here whose queue
		// algorithm matches `PaperPolicy::TwoQCompact`'s (the other 2Q hybrids
		// are Simplified 2Q). Two parameters, not one: `k_out` sizes the live
		// `a1_out` overflow queue and is a real, read parameter here, unlike
		// in `TwoQCompactStack`. Same default fast-tier budget and the same
		// k_in-vs-fast-tier caveat as `TwoQFastAdmissionReprieveCompactHybrid`
		// above -- more acutely so, since `a1_in`'s reservation is carved out of the same
		// DRAM budget `am`'s fast segment draws on.
		#[cfg(feature = "hybrid_cache_common")]
		PaperPolicy::TwoQFullFastAdmissionCompactHybrid(k_in, k_out) => Box::new(
			TwoQFullFastAdmissionCompactHybridStack::new(k_in, k_out, max_size, (max_size as f64 * 0.2) as CacheSize).with_shared_overhead(
				crate::object::overhead::get_hybrid_dram_shared_overhead(&policy) as CacheSize,
			),
		),

		// Now carries the same `with_shared_overhead` reservation and the same
		// continuous drain-target settle as `LruCompactHybrid`/`LfuCompactHybrid`,
		// in the same two-arm with/without-feature shape this comment used to
		// ask for: a follow-up DRAM-usage measurement did show the same issue
		// (metadata is DRAM-resident but is not counted in `fast_used`, so
		// the fast tier overshot its budget).
		#[cfg(feature = "hybrid_cache_common")]
		PaperPolicy::FifoCompactHybrid => Box::new(
			FifoCompactHybridStack::new((max_size as f64 * 0.2) as CacheSize).with_shared_overhead(
				crate::object::overhead::get_hybrid_dram_shared_overhead(&policy) as CacheSize,
			),
		),

		// Same construction as `FifoCompactHybrid` above, which is the design
		// this is a second chance bolted onto -- see
		// `lru_fifo_clock_hybrid_stacks.rs`'s module doc.
		#[cfg(feature = "hybrid_cache_common")]
		PaperPolicy::ClockCompactHybrid => Box::new(
			ClockCompactHybridStack::new((max_size as f64 * 0.2) as CacheSize).with_shared_overhead(
				crate::object::overhead::get_hybrid_dram_shared_overhead(&policy) as CacheSize,
			),
		),

		// Default small/large fast-segment budgets: 10% of max_size each
		// (totaling the same 20% aggregate default the other four hybrids
		// use for their single fast tier), immediately overridden by the
		// real constructor-supplied values right after construction (see
		// `PaperCache::new_sized_hybrid`'s three broadcasts). The
		// 4096-byte (4 KiB) default size threshold is a fixed constant
		// rather than max_size-scaled -- there's no principled way to scale
		// a *classification* threshold with overall cache size the way a
		// capacity budget scales -- also immediately overridden by the real
		// constructor-supplied value.
		#[cfg(feature = "hybrid_cache_common")]
		PaperPolicy::LruSizedCompactHybrid => Box::new(
			LruSizedCompactHybridStack::new(
				(max_size as f64 * 0.1) as CacheSize,
				(max_size as f64 * 0.1) as CacheSize,
				4_096,
			).with_shared_overhead(
				crate::object::overhead::get_hybrid_dram_shared_overhead(&policy) as CacheSize,
			),
		),

		// `ratio` comes from the policy string itself (same as plain
		// `SThreeFifo`/`TwoQCompactHybrid`); the fast-tier budget still
		// defaults to 20% of max_size, same override mechanism as the
		// other hybrids -- immediately overridden by the caller's real
		// CacheTierSize via `new_hybrid`'s `ResizeFastTier` broadcast, same
		// as every other hybrid design. Carries the same
		// `with_shared_overhead` reservation as `TwoQCompactHybrid`, which now has
		// one too: admission being always-slow does not avoid the cost, since
		// the hashtable and eviction-stack entries are DRAM-resident for
		// slow-tier objects just as much as for fast-tier ones.
		#[cfg(feature = "hybrid_cache_common")]
		PaperPolicy::S3FifoCompactHybrid(ratio) => Box::new(
			S3FifoCompactHybridStack::new(ratio, max_size, (max_size as f64 * 0.2) as CacheSize)
				.with_shared_overhead(
					crate::object::overhead::get_hybrid_dram_shared_overhead(&policy) as CacheSize,
				),
		),

		// Faithful tier-segmented S3-FIFO: 0..=3 counter, lazy promotion, lazy
		// eviction. Same construction as `S3FifoCompactHybrid` above.
		#[cfg(feature = "hybrid_cache_common")]
		PaperPolicy::S3FifoFaithfulCompactHybrid(ratio) => Box::new(
			S3FifoFaithfulCompactHybridStack::new(ratio, max_size, (max_size as f64 * 0.2) as CacheSize)
				.with_shared_overhead(
					crate::object::overhead::get_hybrid_dram_shared_overhead(&policy) as CacheSize,
				),
		),

		// Faithful tier-segmented S3-FIFO: 0..=3 counter, lazy promotion, lazy
		// eviction. Same construction as `S3FifoCompactHybrid` above.
		#[cfg(feature = "hybrid_cache_common")]
		PaperPolicy::S3FifoFaithfulFastAdmissionCompactHybrid(ratio) => Box::new(
			S3FifoFaithfulFastAdmissionCompactHybridStack::new(ratio, max_size, (max_size as f64 * 0.2) as CacheSize)
				.with_shared_overhead(
					crate::object::overhead::get_hybrid_dram_shared_overhead(&policy) as CacheSize,
				),
		),

		// Faithful tier-segmented S3-FIFO: 0..=3 counter, lazy promotion, lazy
		// eviction. Same construction as `S3FifoCompactHybrid` above.
		#[cfg(feature = "hybrid_cache_common")]
		PaperPolicy::S3FifoFaithfulReprieveCompactHybrid(ratio) => Box::new(
			S3FifoFaithfulReprieveCompactHybridStack::new(ratio, max_size, (max_size as f64 * 0.2) as CacheSize)
				.with_shared_overhead(
					crate::object::overhead::get_hybrid_dram_shared_overhead(&policy) as CacheSize,
				),
		),

		// Faithful tier-segmented S3-FIFO: 0..=3 counter, lazy promotion, lazy
		// eviction. Same construction as `S3FifoCompactHybrid` above.
		#[cfg(feature = "hybrid_cache_common")]
		PaperPolicy::S3FifoFaithfulFastAdmissionReprieveCompactHybrid(ratio) => Box::new(
			S3FifoFaithfulFastAdmissionReprieveCompactHybridStack::new(ratio, max_size, (max_size as f64 * 0.2) as CacheSize)
				.with_shared_overhead(
					crate::object::overhead::get_hybrid_dram_shared_overhead(&policy) as CacheSize,
				),
		),

		// Same construction/default-fast-tier-budget shape as
		// TwoQCompactHybrid/S3FifoCompactHybrid above -- see
		// two_q_hybrid_stacks.rs's module doc for the ghost-queue
		// mechanics these add on top.
		#[cfg(feature = "hybrid_cache_common")]
		PaperPolicy::TwoQGhostCompactHybrid(k_in) => Box::new(
			TwoQGhostCompactHybridStack::new(k_in, max_size, (max_size as f64 * 0.2) as CacheSize).with_shared_overhead(
				crate::object::overhead::get_hybrid_dram_shared_overhead(&policy) as CacheSize,
			),
		),

		// Same shape again, with the ghost queue keyed on the S3-FIFO
		// one-access queue -- see s3_fifo_hybrid_stacks.rs.
		#[cfg(feature = "hybrid_cache_common")]
		PaperPolicy::S3FifoGhostCompactHybrid(ratio) => Box::new(
			S3FifoGhostCompactHybridStack::new(ratio, max_size, (max_size as f64 * 0.2) as CacheSize).with_shared_overhead(
				crate::object::overhead::get_hybrid_dram_shared_overhead(&policy) as CacheSize,
			),
		),

		// Same construction/default-fast-tier-budget shape as
		// S3FifoGhostCompactHybrid above -- see
		// s3_fifo_hybrid_stacks.rs's module doc for
		// the demotion-time reference-bit gate this adds on top.
		#[cfg(feature = "hybrid_cache_common")]
		PaperPolicy::S3FifoGhostLazyDemotionCompactHybrid(ratio) => Box::new(
			S3FifoGhostLazyDemotionCompactHybridStack::new(ratio, max_size, (max_size as f64 * 0.2) as CacheSize).with_shared_overhead(
				crate::object::overhead::get_hybrid_dram_shared_overhead(&policy) as CacheSize,
			),
		),

		// Same construction/default-fast-tier-budget shape as
		// S3FifoGhostLazyDemotionCompactHybrid above -- see
		// s3_fifo_ghost_lazy_demotion_fast_admission_compact_hybrid_stack.rs's
		// module doc for the shared-DRAM-budget accounting this adds (the
		// one-access queue now competes with the main queue's fast segment
		// for the same fast_capacity).
		#[cfg(feature = "hybrid_cache_common")]
		PaperPolicy::S3FifoGhostLazyDemotionFastAdmissionCompactHybrid(ratio) => Box::new(
			S3FifoGhostLazyDemotionFastAdmissionCompactHybridStack::new(ratio, max_size, (max_size as f64 * 0.2) as CacheSize).with_shared_overhead(
				crate::object::overhead::get_hybrid_dram_shared_overhead(&policy) as CacheSize,
			),
		),

		// Same construction/default-fast-tier-budget shape as
		// S3FifoGhostLazyDemotionFastAdmissionCompactHybrid above -- see
		// s3_fifo_ghost_lazy_demotion_fast_admission_midpoint_compact_hybrid_stack.rs's
		// module doc for the mid-slow-segment reference-bit checkpoint this
		// adds on top.
		#[cfg(feature = "hybrid_cache_common")]
		PaperPolicy::S3FifoGhostLazyDemotionFastAdmissionMidpointCompactHybrid(ratio) => Box::new(
			S3FifoGhostLazyDemotionFastAdmissionMidpointCompactHybridStack::new(ratio, max_size, (max_size as f64 * 0.2) as CacheSize).with_shared_overhead(
				crate::object::overhead::get_hybrid_dram_shared_overhead(&policy) as CacheSize,
			),
		),

		// Same construction/default-fast-tier-budget shape as
		// S3FifoGhostLazyDemotionFastAdmissionMidpointCompactHybrid above -- see
		// s3_fifo_lazy_demotion_fast_admission_midpoint_reprieve_compact_hybrid_stack.rs's
		// module doc: no ghost queue (removed entirely), and a one-access
		// key that ages out is spliced into the slow tier of the main
		// queue instead of being evicted.
		#[cfg(feature = "hybrid_cache_common")]
		PaperPolicy::S3FifoLazyDemotionFastAdmissionMidpointReprieveCompactHybrid(ratio) => Box::new(
			S3FifoLazyDemotionFastAdmissionMidpointReprieveCompactHybridStack::new(ratio, max_size, (max_size as f64 * 0.2) as CacheSize).with_shared_overhead(
				crate::object::overhead::get_hybrid_dram_shared_overhead(&policy) as CacheSize,
			),
		),

		// Same construction shape as the midpoint variant above, minus the
		// mid-slow checkpoint -- see
		// s3_fifo_lazy_demotion_fast_admission_reprieve_compact_hybrid_stack.rs's
		// module doc.
		#[cfg(feature = "hybrid_cache_common")]
		PaperPolicy::S3FifoLazyDemotionFastAdmissionReprieveCompactHybrid(ratio) => Box::new(
			S3FifoLazyDemotionFastAdmissionReprieveCompactHybridStack::new(ratio, max_size, (max_size as f64 * 0.2) as CacheSize).with_shared_overhead(
				crate::object::overhead::get_hybrid_dram_shared_overhead(&policy) as CacheSize,
			),
		),

		// Same construction shape as its fast-admission sibling above. The
		// one-access queue is slow-tier here, so its `one_access_capacity`
		// bounds PMEM rather than being carved out of the DRAM budget -- see
		// s3_fifo_lazy_demotion_reprieve_compact_hybrid_stack.rs's
		// `effective_main_fast_capacity`.
		#[cfg(feature = "hybrid_cache_common")]
		PaperPolicy::S3FifoLazyDemotionReprieveCompactHybrid(ratio) => Box::new(
			S3FifoLazyDemotionReprieveCompactHybridStack::new(ratio, max_size, (max_size as f64 * 0.2) as CacheSize).with_shared_overhead(
				crate::object::overhead::get_hybrid_dram_shared_overhead(&policy) as CacheSize,
			),
		),

		// Same construction/default-fast-tier-budget shape as its
		// predecessor above -- see
		// s3_fifo_lazy_demotion_fast_admission_split_slow_reprieve_compact_hybrid_stack.rs's
		// module doc: the slow tier is split into two physical FIFO
		// segments, and every object's reference bit is checked as it
		// crosses between them.
		#[cfg(feature = "hybrid_cache_common")]
		PaperPolicy::S3FifoLazyDemotionFastAdmissionSplitSlowReprieveCompactHybrid(ratio) => Box::new(
			S3FifoLazyDemotionFastAdmissionSplitSlowReprieveCompactHybridStack::new(ratio, max_size, (max_size as f64 * 0.2) as CacheSize).with_shared_overhead(
				crate::object::overhead::get_hybrid_dram_shared_overhead(&policy) as CacheSize,
			),
		),
		// Hybrid stacks are compiled in every build; without the hybrid
		// feature there is no shared-overhead reservation to wire in, so
		// the bare construction the per-feature fallbacks used is kept.
		#[cfg(not(feature = "hybrid_cache_common"))]
		PaperPolicy::LruLfuCompactHybrid(promote_k) => Box::new(
			LruLfuCompactHybridStack::new((max_size as f64 * 0.2) as CacheSize, promote_k),
		),

		#[cfg(not(feature = "hybrid_cache_common"))]
		PaperPolicy::LfuCompactHybrid =>
			Box::new(LfuCompactHybridStack::new((max_size as f64 * 0.2) as CacheSize)),
		#[cfg(not(feature = "hybrid_cache_common"))]
		#[cfg(not(feature = "hybrid_cache_common"))]
		PaperPolicy::TwoQFastAdmissionReprieveCompactHybrid(k_in) => Box::new(TwoQFastAdmissionReprieveCompactHybridStack::new(k_in, max_size, (max_size as f64 * 0.2) as CacheSize)),
		#[cfg(not(feature = "hybrid_cache_common"))]
		PaperPolicy::TwoQFullFastAdmissionCompactHybrid(k_in, k_out) => Box::new(TwoQFullFastAdmissionCompactHybridStack::new(k_in, k_out, max_size, (max_size as f64 * 0.2) as CacheSize)),
		#[cfg(not(feature = "hybrid_cache_common"))]
		PaperPolicy::FifoCompactHybrid => Box::new(FifoCompactHybridStack::new((max_size as f64 * 0.2) as CacheSize)),
		#[cfg(not(feature = "hybrid_cache_common"))]
		PaperPolicy::ClockCompactHybrid => Box::new(ClockCompactHybridStack::new((max_size as f64 * 0.2) as CacheSize)),
		#[cfg(not(feature = "hybrid_cache_common"))]
		PaperPolicy::LruSizedCompactHybrid => Box::new(LruSizedCompactHybridStack::new(
			(max_size as f64 * 0.1) as CacheSize,
			(max_size as f64 * 0.1) as CacheSize,
			4_096,
		)),
		#[cfg(not(feature = "hybrid_cache_common"))]
		PaperPolicy::S3FifoCompactHybrid(ratio) => Box::new(S3FifoCompactHybridStack::new(ratio, max_size, (max_size as f64 * 0.2) as CacheSize)),
		#[cfg(not(feature = "hybrid_cache_common"))]
		PaperPolicy::S3FifoFaithfulCompactHybrid(ratio) => Box::new(S3FifoFaithfulCompactHybridStack::new(ratio, max_size, (max_size as f64 * 0.2) as CacheSize)),
		#[cfg(not(feature = "hybrid_cache_common"))]
		PaperPolicy::S3FifoFaithfulFastAdmissionCompactHybrid(ratio) => Box::new(S3FifoFaithfulFastAdmissionCompactHybridStack::new(ratio, max_size, (max_size as f64 * 0.2) as CacheSize)),
		#[cfg(not(feature = "hybrid_cache_common"))]
		PaperPolicy::S3FifoFaithfulReprieveCompactHybrid(ratio) => Box::new(S3FifoFaithfulReprieveCompactHybridStack::new(ratio, max_size, (max_size as f64 * 0.2) as CacheSize)),
		#[cfg(not(feature = "hybrid_cache_common"))]
		PaperPolicy::S3FifoFaithfulFastAdmissionReprieveCompactHybrid(ratio) => Box::new(S3FifoFaithfulFastAdmissionReprieveCompactHybridStack::new(ratio, max_size, (max_size as f64 * 0.2) as CacheSize)),
		#[cfg(not(feature = "hybrid_cache_common"))]
		PaperPolicy::TwoQGhostCompactHybrid(k_in) => Box::new(TwoQGhostCompactHybridStack::new(k_in, max_size, (max_size as f64 * 0.2) as CacheSize)),
		#[cfg(not(feature = "hybrid_cache_common"))]
		PaperPolicy::S3FifoGhostCompactHybrid(ratio) => Box::new(S3FifoGhostCompactHybridStack::new(ratio, max_size, (max_size as f64 * 0.2) as CacheSize)),
		#[cfg(not(feature = "hybrid_cache_common"))]
		PaperPolicy::S3FifoGhostLazyDemotionCompactHybrid(ratio) => Box::new(S3FifoGhostLazyDemotionCompactHybridStack::new(ratio, max_size, (max_size as f64 * 0.2) as CacheSize)),
		#[cfg(not(feature = "hybrid_cache_common"))]
		PaperPolicy::S3FifoGhostLazyDemotionFastAdmissionCompactHybrid(ratio) => Box::new(S3FifoGhostLazyDemotionFastAdmissionCompactHybridStack::new(ratio, max_size, (max_size as f64 * 0.2) as CacheSize)),
		#[cfg(not(feature = "hybrid_cache_common"))]
		PaperPolicy::S3FifoGhostLazyDemotionFastAdmissionMidpointCompactHybrid(ratio) => Box::new(S3FifoGhostLazyDemotionFastAdmissionMidpointCompactHybridStack::new(ratio, max_size, (max_size as f64 * 0.2) as CacheSize)),
		#[cfg(not(feature = "hybrid_cache_common"))]
		PaperPolicy::S3FifoLazyDemotionFastAdmissionMidpointReprieveCompactHybrid(ratio) => Box::new(S3FifoLazyDemotionFastAdmissionMidpointReprieveCompactHybridStack::new(ratio, max_size, (max_size as f64 * 0.2) as CacheSize)),
		#[cfg(not(feature = "hybrid_cache_common"))]
		PaperPolicy::S3FifoLazyDemotionFastAdmissionReprieveCompactHybrid(ratio) => Box::new(S3FifoLazyDemotionFastAdmissionReprieveCompactHybridStack::new(ratio, max_size, (max_size as f64 * 0.2) as CacheSize)),
		#[cfg(not(feature = "hybrid_cache_common"))]
		PaperPolicy::S3FifoLazyDemotionReprieveCompactHybrid(ratio) => Box::new(S3FifoLazyDemotionReprieveCompactHybridStack::new(ratio, max_size, (max_size as f64 * 0.2) as CacheSize)),
		#[cfg(not(feature = "hybrid_cache_common"))]
		PaperPolicy::S3FifoLazyDemotionFastAdmissionSplitSlowReprieveCompactHybrid(ratio) => Box::new(S3FifoLazyDemotionFastAdmissionSplitSlowReprieveCompactHybridStack::new(ratio, max_size, (max_size as f64 * 0.2) as CacheSize)),

	}
}

#[cfg(test)]
mod init_policy_stack_tests {
	//! Runtime-dispatch tests for `init_policy_stack`.
	//!
	//! The "every design in every build, chosen at runtime" architecture rests
	//! entirely on the match above: all `POLICY_VARIANT_COUNT` policies
	//! (`HYBRID_DESIGN_COUNT` of them tiered) are compiled into every binary,
	//! and the design that actually runs is picked by the `PaperPolicy` value
	//! handed to `PaperCache::new`. Nothing else in the crate checks that the
	//! match returns the stack it was asked for. The arms are near-identical
	//! one-expression lines, several of which differ by a single word inside a
	//! 60-character type name
	//! (`S3FifoGhostLazyDemotionCompactHybridStack` vs.
	//! `S3FifoGhostLazyDemotionFastAdmissionCompactHybridStack`), so a
	//! copy-paste slip between two of them would silently run a different
	//! eviction design for
	//! the lifetime of the process and misattribute every number measured from
	//! it.
	//!
	//! One suite covers both cfg branches. The
	//! `feature = "hybrid_cache_common"` arms and the
	//! `not(feature = "hybrid_cache_common")` fallbacks construct the *same*
	//! stack types -- the feature only adds the `with_shared_overhead`
	//! reservation, which changes a fast-tier budget, not a stack's identity --
	//! and every one of the hybrid policies has an arm in both sets, so no
	//! variant is unreachable in either build and no gate is needed here. (A
	//! hybrid design added to only one of the two sets makes the match
	//! non-exhaustive in the other build, which is a compile error rather than
	//! something a test could observe.)
	//!
	//! Everything here is construction-only: no key is ever inserted, so
	//! nothing allocates through the `Hybrid` allocator and these tests need no
	//! warmed PMEM pool under `eviction_stacks_pmem`.

	use std::collections::HashSet;

	use super::*;

	/// Every arm derives its sub-budgets from `max_size`: `max_size * 0.2` for
	/// the hybrids' fast tier, `max_size * 0.1` per segment for
	/// `LruSizedCompactHybrid`, `k_in * max_size` for the 2Q family and
	/// `ratio * max_size` for the S3-FIFO family. 1 MB is the scale the hybrid
	/// integration suites build their caches at, and it keeps every one of
	/// those derived budgets comfortably non-zero -- below ~5 bytes the 20%
	/// fast-tier budget truncates to 0, which is a degenerate stack rather than
	/// a dispatch question.
	const TEST_MAX_SIZE: CacheSize = 1_000_000;

	/// Number of `PaperPolicy` variants, and therefore the number of rows the
	/// table below must have. Kept as a named constant so a mismatch reads as
	/// "a design is missing from the table", not as an off-by-one.
	const POLICY_VARIANT_COUNT: usize = 32;

	/// Number of variants for which `PaperPolicy::is_hybrid` must hold: the
	/// tiered designs this crate exists to compare.
	const HYBRID_DESIGN_COUNT: usize = 23;

	/// Every `PaperPolicy` variant, listed explicitly, in declaration order.
	///
	/// Column 0 is the value handed to `init_policy_stack`. Column 1 is a
	/// *different value of the same variant* -- a different `k_in`, `ratio` or
	/// `promote_k` -- used to pin down that `is_policy` discriminates on the
	/// variant and not on the payload. For the payload-free variants the two
	/// columns are necessarily the same value.
	///
	/// The list is written out rather than derived: `variant_name` below is an
	/// exhaustive match with no `_` arm, so adding a variant to `PaperPolicy`
	/// stops this file compiling until the new design is added here too.
	const POLICY_DISPATCH_TABLE: [(PaperPolicy, PaperPolicy); POLICY_VARIANT_COUNT] = [
		(PaperPolicy::LfuCompact, PaperPolicy::LfuCompact),
		(PaperPolicy::FifoCompact, PaperPolicy::FifoCompact),
		(PaperPolicy::ClockCompact, PaperPolicy::ClockCompact),
		(PaperPolicy::SieveCompact, PaperPolicy::SieveCompact),
		(PaperPolicy::MruCompact, PaperPolicy::MruCompact),
		(PaperPolicy::TwoQCompact(0.25, 0.5), PaperPolicy::TwoQCompact(0.25, 0.5)),
		(PaperPolicy::LruCompact, PaperPolicy::LruCompact),
		(PaperPolicy::Arc, PaperPolicy::Arc),
		(PaperPolicy::SThreeFifoCompact(0.1), PaperPolicy::SThreeFifoCompact(0.9)),
		(PaperPolicy::S3FifoFaithfulCompactHybrid(0.1), PaperPolicy::S3FifoFaithfulCompactHybrid(0.9)),
		(PaperPolicy::S3FifoFaithfulFastAdmissionCompactHybrid(0.1), PaperPolicy::S3FifoFaithfulFastAdmissionCompactHybrid(0.9)),
		(PaperPolicy::S3FifoFaithfulReprieveCompactHybrid(0.1), PaperPolicy::S3FifoFaithfulReprieveCompactHybrid(0.9)),
		(PaperPolicy::S3FifoFaithfulFastAdmissionReprieveCompactHybrid(0.1), PaperPolicy::S3FifoFaithfulFastAdmissionReprieveCompactHybrid(0.9)),
		(PaperPolicy::LruCompactHybrid, PaperPolicy::LruCompactHybrid),
		(PaperPolicy::LfuCompactHybrid, PaperPolicy::LfuCompactHybrid),
		(PaperPolicy::TwoQCompactHybrid(0.1), PaperPolicy::TwoQCompactHybrid(0.9)),
		(PaperPolicy::TwoQFastAdmissionReprieveCompactHybrid(0.1), PaperPolicy::TwoQFastAdmissionReprieveCompactHybrid(0.9)),
		(PaperPolicy::TwoQFullFastAdmissionCompactHybrid(0.25, 0.25), PaperPolicy::TwoQFullFastAdmissionCompactHybrid(0.5, 0.4)),
		(PaperPolicy::FifoCompactHybrid, PaperPolicy::FifoCompactHybrid),
		(PaperPolicy::ClockCompactHybrid, PaperPolicy::ClockCompactHybrid),
		(PaperPolicy::LruSizedCompactHybrid, PaperPolicy::LruSizedCompactHybrid),
		(PaperPolicy::LruLfuCompactHybrid(3), PaperPolicy::LruLfuCompactHybrid(7)),
		(PaperPolicy::S3FifoCompactHybrid(0.1), PaperPolicy::S3FifoCompactHybrid(0.9)),
		(PaperPolicy::TwoQGhostCompactHybrid(0.1), PaperPolicy::TwoQGhostCompactHybrid(0.9)),
		(PaperPolicy::S3FifoGhostCompactHybrid(0.1), PaperPolicy::S3FifoGhostCompactHybrid(0.9)),
		(PaperPolicy::S3FifoGhostLazyDemotionCompactHybrid(0.1), PaperPolicy::S3FifoGhostLazyDemotionCompactHybrid(0.9)),
		(PaperPolicy::S3FifoGhostLazyDemotionFastAdmissionCompactHybrid(0.1), PaperPolicy::S3FifoGhostLazyDemotionFastAdmissionCompactHybrid(0.9)),
		(PaperPolicy::S3FifoGhostLazyDemotionFastAdmissionMidpointCompactHybrid(0.1), PaperPolicy::S3FifoGhostLazyDemotionFastAdmissionMidpointCompactHybrid(0.9)),
		(PaperPolicy::S3FifoLazyDemotionFastAdmissionMidpointReprieveCompactHybrid(0.1), PaperPolicy::S3FifoLazyDemotionFastAdmissionMidpointReprieveCompactHybrid(0.9)),
		(PaperPolicy::S3FifoLazyDemotionFastAdmissionReprieveCompactHybrid(0.1), PaperPolicy::S3FifoLazyDemotionFastAdmissionReprieveCompactHybrid(0.9)),
		(PaperPolicy::S3FifoLazyDemotionReprieveCompactHybrid(0.1), PaperPolicy::S3FifoLazyDemotionReprieveCompactHybrid(0.9)),
		(PaperPolicy::S3FifoLazyDemotionFastAdmissionSplitSlowReprieveCompactHybrid(0.1), PaperPolicy::S3FifoLazyDemotionFastAdmissionSplitSlowReprieveCompactHybrid(0.9)),
	];

	/// The variant's name, ignoring any payload.
	///
	/// Deliberately exhaustive with no `_` arm: adding a variant to
	/// `PaperPolicy` breaks this match at compile time, and that is the signal
	/// to add the new design to `POLICY_DISPATCH_TABLE` (and to bump
	/// `POLICY_VARIANT_COUNT`) so it is dispatch-tested like every other one.
	fn variant_name(policy: &PaperPolicy) -> &'static str {
		match policy {
			PaperPolicy::LfuCompact => "LfuCompact",
			PaperPolicy::FifoCompact => "FifoCompact",
			PaperPolicy::ClockCompact => "ClockCompact",
			PaperPolicy::SieveCompact => "SieveCompact",
			PaperPolicy::MruCompact => "MruCompact",
			PaperPolicy::TwoQCompact(..) => "TwoQCompact",
			PaperPolicy::LruCompact => "LruCompact",
			PaperPolicy::Arc => "Arc",
			PaperPolicy::SThreeFifoCompact(_) => "SThreeFifoCompact",
			PaperPolicy::S3FifoFaithfulCompactHybrid(_) => "S3FifoFaithfulCompactHybrid",
			PaperPolicy::S3FifoFaithfulFastAdmissionCompactHybrid(_) => "S3FifoFaithfulFastAdmissionCompactHybrid",
			PaperPolicy::S3FifoFaithfulReprieveCompactHybrid(_) => "S3FifoFaithfulReprieveCompactHybrid",
			PaperPolicy::S3FifoFaithfulFastAdmissionReprieveCompactHybrid(_) => "S3FifoFaithfulFastAdmissionReprieveCompactHybrid",
			PaperPolicy::LruCompactHybrid => "LruCompactHybrid",
			PaperPolicy::LfuCompactHybrid => "LfuCompactHybrid",
			PaperPolicy::TwoQCompactHybrid(_) => "TwoQCompactHybrid",
			PaperPolicy::TwoQFastAdmissionReprieveCompactHybrid(_) => "TwoQFastAdmissionReprieveCompactHybrid",
			PaperPolicy::TwoQFullFastAdmissionCompactHybrid(..) => "TwoQFullFastAdmissionCompactHybrid",
			PaperPolicy::FifoCompactHybrid => "FifoCompactHybrid",
			PaperPolicy::ClockCompactHybrid => "ClockCompactHybrid",
			PaperPolicy::LruSizedCompactHybrid => "LruSizedCompactHybrid",
			PaperPolicy::LruLfuCompactHybrid(_) => "LruLfuCompactHybrid",
			PaperPolicy::S3FifoCompactHybrid(_) => "S3FifoCompactHybrid",
			PaperPolicy::TwoQGhostCompactHybrid(_) => "TwoQGhostCompactHybrid",
			PaperPolicy::S3FifoGhostCompactHybrid(_) => "S3FifoGhostCompactHybrid",
			PaperPolicy::S3FifoGhostLazyDemotionCompactHybrid(_) => "S3FifoGhostLazyDemotionCompactHybrid",
			PaperPolicy::S3FifoGhostLazyDemotionFastAdmissionCompactHybrid(_) => "S3FifoGhostLazyDemotionFastAdmissionCompactHybrid",
			PaperPolicy::S3FifoGhostLazyDemotionFastAdmissionMidpointCompactHybrid(_) => "S3FifoGhostLazyDemotionFastAdmissionMidpointCompactHybrid",
			PaperPolicy::S3FifoLazyDemotionFastAdmissionMidpointReprieveCompactHybrid(_) => "S3FifoLazyDemotionFastAdmissionMidpointReprieveCompactHybrid",
			PaperPolicy::S3FifoLazyDemotionFastAdmissionReprieveCompactHybrid(_) => "S3FifoLazyDemotionFastAdmissionReprieveCompactHybrid",
			PaperPolicy::S3FifoLazyDemotionReprieveCompactHybrid(_) => "S3FifoLazyDemotionReprieveCompactHybrid",
			PaperPolicy::S3FifoLazyDemotionFastAdmissionSplitSlowReprieveCompactHybrid(_) => "S3FifoLazyDemotionFastAdmissionSplitSlowReprieveCompactHybrid",
		}
	}

	/// The premise of the architecture: asking for a design gets you that
	/// design, for every one of them.
	#[test]
	fn every_policy_variant_dispatches_to_a_stack_that_claims_it() {
		for (policy, _) in POLICY_DISPATCH_TABLE {
			let stack = init_policy_stack(policy, TEST_MAX_SIZE);

			assert!(
				stack.is_policy(&policy),
				"`init_policy_stack` built a stack for `{policy}` that does not report itself as `{policy}`: that arm constructs some other design",
			);
		}
	}

	/// The other half of the premise, and the half a positive-only test cannot
	/// see: a stack that says yes to everything would pass the test above while
	/// making every runtime policy decision meaningless.
	///
	/// Foils are compared by *variant*, never by value, because `is_policy` is
	/// deliberately payload-blind (see
	/// `is_policy_matches_on_the_variant_not_its_payload`) -- asking a
	/// `TwoQCompactHybrid(0.1)` stack about `TwoQCompactHybrid(0.9)` is not a
	/// cross-check.
	#[test]
	fn no_stack_claims_a_policy_it_was_not_built_for() {
		for (policy, _) in POLICY_DISPATCH_TABLE {
			let stack = init_policy_stack(policy, TEST_MAX_SIZE);

			for (foil, _) in POLICY_DISPATCH_TABLE {
				if variant_name(&foil) == variant_name(&policy) {
					continue;
				}

				assert!(
					!stack.is_policy(&foil),
					"the stack built for `{policy}` also claims to be `{foil}`: `is_policy` is too loose to tell the two designs apart",
				);
			}
		}
	}

	/// `is_policy` discriminates on the variant, not on the payload. Every
	/// stack that carries a `k_in`/`ratio`/`promote_k` keeps it as its own
	/// field; the identity question it answers is "which design am I", not
	/// "which parameterisation am I".
	///
	/// This is the contract the cross-check above depends on, and it is what
	/// stops a `k_in` that round-tripped through the policy string from making
	/// a stack disown itself.
	#[test]
	fn is_policy_discriminates_on_the_payload_of_a_parameterised_policy() {
		// A parameterised policy names both a design AND its tuning, so a
		// stack built for one payload must not answer to another.
		// `TwoQCompactStack::is_policy` is the clearest statement of the rule --
		// `self.k_in == *k_in && self.k_out == *k_out`.
		//
		// `LruLfuCompactHybrid` is the exception in the crate: it matches its
		// own variant with `(_)` and ignores `promote_k`, so a change to that
		// knob alone does not read as a different policy. Pinned here rather
		// than papered over, so that if it is ever brought in line this test
		// fails and says so.
		for (policy, same_variant_other_payload) in POLICY_DISPATCH_TABLE {
			if policy == same_variant_other_payload {
				continue; // payload-free variant: nothing to discriminate
			}

			let stack = init_policy_stack(policy, TEST_MAX_SIZE);

			if matches!(policy, PaperPolicy::LruLfuCompactHybrid(_)) {
				assert!(
					stack.is_policy(&same_variant_other_payload),
					"`{policy}` is documented as the one payload-insensitive \
					 design, but its stack now rejects \
					 `{same_variant_other_payload}` -- if `is_policy` was \
					 deliberately tightened to compare `promote_k`, move it \
					 in with the others and delete this branch",
				);

				continue;
			}

			assert!(
				!stack.is_policy(&same_variant_other_payload),
				"the stack built for `{policy}` also claims to be \
				 `{same_variant_other_payload}`: `is_policy` ignores the \
				 payload",
			);
		}
	}

	/// Guards the three tests above from silently shrinking: they are only
	/// exhaustive if the table is.
	#[test]
	fn the_dispatch_table_covers_every_policy_variant_exactly_once() {
		let names = POLICY_DISPATCH_TABLE
			.into_iter()
			.map(|(policy, _)| variant_name(&policy))
			.collect::<HashSet<&'static str>>();

		assert_eq!(
			names.len(),
			POLICY_VARIANT_COUNT,
			"the dispatch table has {} rows but only {} distinct variants: a row is duplicated, so some design is not dispatch-tested at all",
			POLICY_DISPATCH_TABLE.len(),
			names.len(),
		);

		for (policy, same_variant_other_payload) in POLICY_DISPATCH_TABLE {
			assert_eq!(
				variant_name(&policy),
				variant_name(&same_variant_other_payload),
				"the second column of the `{policy}` row is a different variant (`{same_variant_other_payload}`), which would turn the payload check into an accidental cross-variant check",
			);
		}
	}

	/// `PaperPolicy::is_hybrid` is a hand-written `matches!` over the tiered
	/// variants with no exhaustiveness check of its own, and it is what
	/// decides whether
	/// a cache gets a fast tier at all. Anchor it to the one list that *is*
	/// compiler-checked against the enum.
	#[test]
	fn every_tiered_design_is_reported_as_hybrid() {
		let hybrids = POLICY_DISPATCH_TABLE
			.into_iter()
			.filter(|(policy, _)| policy.is_hybrid())
			.count();

		assert_eq!(
			hybrids, HYBRID_DESIGN_COUNT,
			"`is_hybrid` recognises {hybrids} of the {POLICY_VARIANT_COUNT} designs, expected {HYBRID_DESIGN_COUNT}",
		);

		for (policy, _) in POLICY_DISPATCH_TABLE {
			assert_eq!(
				policy.is_hybrid(),
				variant_name(&policy).ends_with("Hybrid"),
				"`{policy}`: `is_hybrid` disagrees with the variant's own name",
			);
		}
	}

	/// A freshly dispatched stack tracks nothing, for every design. The object
	/// map it will be paired with is empty at that moment, and `apply_evictions`
	/// trusts the stack's view: a stack that arrives holding keys of its own
	/// would have `evict_one` hand back a key the map has never heard of.
	#[test]
	fn every_freshly_built_stack_is_empty() {
		for (policy, _) in POLICY_DISPATCH_TABLE {
			let stack = init_policy_stack(policy, TEST_MAX_SIZE);
			let tracked = stack.len();

			assert_eq!(
				tracked, 0,
				"the stack built for `{policy}` reports {tracked} tracked keys before anything was inserted",
			);

			assert!(
				!stack.contains(1),
				"the stack built for `{policy}` claims to contain a key that was never inserted",
			);

			assert_eq!(
				stack.placement_of(1),
				None,
				"the stack built for `{policy}` places a key that was never inserted",
			);
		}
	}
}

/// The fast-tier reservation, pinned across every hybrid design at once.
///
/// Each design is built the way the worker builds it -- `init_policy_stack`,
/// so with its production `shared_overhead` -- and driven the way the worker
/// drives it: every admission is followed by two hits on the same key, and
/// every event by the evictions the worker would make -- for a sub-queue over
/// its own capacity, or for the cache over `max_size`. That pushes every
/// design's keys through its admission queue into main, eagerly or at
/// eviction, so each ends up with a slow tier whose metadata is several times
/// anything a ghost could account for. Whatever it holds there,
/// `dram_reserved_bytes` must cover EVERY tracked key, `len() x
/// shared_overhead`, plus -- for the designs with a ghost -- no more than the
/// ghost entries the evictions can have left. A rule that charges fast keys
/// alone falls short by the slow keys' metadata and cannot pass by accident.
///
/// Under `merged_object_store` the worker builds `MergedStackHandle` instead
/// and refuses most of these policies, so there this still drives the split
/// stacks, with the merged build's per-object figure: a test of the rule, not
/// a model of that build's worker.
///
/// Gated off `eviction_stacks_pmem`, whose stacks allocate through the
/// `Hybrid` allocator and would need a warmed PMEM pool.
#[cfg(all(test, feature = "hybrid_cache_common", not(feature = "eviction_stacks_pmem")))]
mod reservation_tests {
	use super::*;

	/// Each design's fast tier is 20% of this: 200_000 B.
	const MAX_SIZE: CacheSize = 1_000_000;

	/// 1_200_000 B of values: six times the fast tier, and enough over
	/// `MAX_SIZE` that the designs which promote only at eviction (the
	/// faithful S3-FIFO family) do.
	const N: HashedKey = 3_000;
	const SIZE: ObjectSize = 400;

	/// Per-entry ghost charges: the ghost filter's slot, and the faithful
	/// S3-FIFO's exact ghost queue entry.
	const FILTER: CacheSize = crate::object::overhead::GHOST_ENTRY_DRAM_OVERHEAD as CacheSize;
	const EXACT: CacheSize = crate::object::overhead::EXACT_GHOST_ENTRY_DRAM_OVERHEAD as CacheSize;

	/// Every hybrid design, with what it charges per ghost entry (0: no ghost).
	const DESIGNS: [(PaperPolicy, CacheSize); 23] = [
		(PaperPolicy::LruCompactHybrid, 0),
		(PaperPolicy::LfuCompactHybrid, 0),
		(PaperPolicy::LruLfuCompactHybrid(3), 0),
		(PaperPolicy::LruSizedCompactHybrid, 0),
		(PaperPolicy::FifoCompactHybrid, 0),
		(PaperPolicy::ClockCompactHybrid, 0),
		(PaperPolicy::TwoQCompactHybrid(0.1), 0),
		(PaperPolicy::TwoQFastAdmissionReprieveCompactHybrid(0.1), 0),
		(PaperPolicy::TwoQFullFastAdmissionCompactHybrid(0.1, 0.5), 0),
		(PaperPolicy::TwoQGhostCompactHybrid(0.1), FILTER),
		(PaperPolicy::S3FifoCompactHybrid(0.1), 0),
		(PaperPolicy::S3FifoFaithfulCompactHybrid(0.1), EXACT),
		(PaperPolicy::S3FifoFaithfulFastAdmissionCompactHybrid(0.1), EXACT),
		(PaperPolicy::S3FifoFaithfulReprieveCompactHybrid(0.1), EXACT),
		(PaperPolicy::S3FifoFaithfulFastAdmissionReprieveCompactHybrid(0.1), EXACT),
		(PaperPolicy::S3FifoGhostCompactHybrid(0.1), FILTER),
		(PaperPolicy::S3FifoGhostLazyDemotionCompactHybrid(0.1), FILTER),
		(PaperPolicy::S3FifoGhostLazyDemotionFastAdmissionCompactHybrid(0.1), FILTER),
		(PaperPolicy::S3FifoGhostLazyDemotionFastAdmissionMidpointCompactHybrid(0.1), FILTER),
		(PaperPolicy::S3FifoLazyDemotionFastAdmissionMidpointReprieveCompactHybrid(0.1), 0),
		(PaperPolicy::S3FifoLazyDemotionFastAdmissionReprieveCompactHybrid(0.1), 0),
		(PaperPolicy::S3FifoLazyDemotionReprieveCompactHybrid(0.1), 0),
		(PaperPolicy::S3FifoLazyDemotionFastAdmissionSplitSlowReprieveCompactHybrid(0.1), 0),
	];

	/// What the worker does after each event: evict while a sub-queue is over
	/// its own capacity or the cache is over `MAX_SIZE`, read here from the
	/// stack's own byte gauges. Returns how many keys went.
	fn evict_while_asked(stack: &mut dyn PolicyStack) -> CacheSize {
		let mut evicted = 0;

		while (stack.needs_capacity_eviction()
			|| stack.fast_bytes_used() + stack.slow_bytes_used() > MAX_SIZE)
			&& stack.evict_one().is_some()
		{
			evicted += 1;
		}

		evicted
	}

	#[test]
	fn every_hybrid_design_charges_every_tracked_key() {
		let mut failures = Vec::new();

		for (policy, ghost_entry) in DESIGNS {
			assert!(policy.is_hybrid(), "{policy} is not a hybrid design");

			let overhead =
				crate::object::overhead::get_hybrid_dram_shared_overhead(&policy) as CacheSize;

			assert!(
				overhead > 0,
				"{policy}: no reservation to test -- is PAPER_DISABLE_SHARED_OVERHEAD set?",
			);

			let mut stack = init_policy_stack(policy, MAX_SIZE);
			let mut evicted = 0;

			for key in 1..=N {
				stack.insert(key, SIZE);
				evicted += evict_while_asked(stack.as_mut());

				for _ in 0..2 {
					if stack.contains(key) {
						stack.update(key);
						evicted += evict_while_asked(stack.as_mut());
					}
				}
			}

			let tracked = stack.len() as CacheSize;
			let fast = stack.fast_object_count();
			let slow = stack.slow_object_count() as CacheSize;
			let reserved = stack.dram_reserved_bytes();
			let floor = tracked * overhead;
			let ghost_room = evicted * ghost_entry;

			if slow * overhead <= ghost_room {
				failures.push(format!(
					"{policy}: fixture too weak -- {slow} slow keys' metadata ({} B) does \
					 not exceed the {ghost_room} B a ghost could account for",
					slow * overhead,
				));
				continue;
			}

			if !(floor..=floor + ghost_room).contains(&reserved) {
				failures.push(format!(
					"{policy}: reserves {reserved} B, but {tracked} tracked keys x {overhead} B \
					 = {floor} B ({fast} fast, {slow} slow{})",
					match ghost_room {
						0 => String::new(),
						room => format!(", up to {room} B of ghost on top"),
					},
				));
			}
		}

		assert!(
			failures.is_empty(),
			"every tracked key's metadata is DRAM-resident whichever tier its value is \
			 in, so each must be charged:\n{}",
			failures.join("\n"),
		);
	}
}

/// S5a: every tiered design's `structure_bytes` is what the allocator holds
/// for its structures, EXACTLY, after every operation of a run that inserts,
/// hits, evicts and removes -- slab chunks and their table, the keyless
/// index, the free list, the LFU designs' bucket maps, the ghosts -- and puts
/// them on the node its build does: DRAM, or the slow node under
/// `eviction_stacks_pmem`. The stack's box, which the policy worker adds, is
/// the rest of what the allocator sees; the `migrations` a drain takes are
/// freed before each check.
#[cfg(all(test, feature = "hybrid_cache_common"))]
mod structure_bytes_tests {
	use super::*;

	/// Each design's fast tier is 20% of this.
	const MAX_SIZE: CacheSize = 1_000_000;

	/// 1_200_000 B of values at `SIZE`: the fast tier demotes, the cache
	/// evicts, and 5,000 keys take every slab past one chunk.
	const N: HashedKey = 5_000;
	const SIZE: ObjectSize = 240;

	const DESIGNS: [PaperPolicy; 23] = [
		PaperPolicy::LruCompactHybrid,
		PaperPolicy::LfuCompactHybrid,
		PaperPolicy::LruLfuCompactHybrid(3),
		PaperPolicy::LruSizedCompactHybrid,
		PaperPolicy::FifoCompactHybrid,
		PaperPolicy::ClockCompactHybrid,
		PaperPolicy::TwoQCompactHybrid(0.1),
		PaperPolicy::TwoQFastAdmissionReprieveCompactHybrid(0.1),
		PaperPolicy::TwoQFullFastAdmissionCompactHybrid(0.1, 0.5),
		PaperPolicy::TwoQGhostCompactHybrid(0.1),
		PaperPolicy::S3FifoCompactHybrid(0.1),
		PaperPolicy::S3FifoFaithfulCompactHybrid(0.1),
		PaperPolicy::S3FifoFaithfulFastAdmissionCompactHybrid(0.1),
		PaperPolicy::S3FifoFaithfulReprieveCompactHybrid(0.1),
		PaperPolicy::S3FifoFaithfulFastAdmissionReprieveCompactHybrid(0.1),
		PaperPolicy::S3FifoGhostCompactHybrid(0.1),
		PaperPolicy::S3FifoGhostLazyDemotionCompactHybrid(0.1),
		PaperPolicy::S3FifoGhostLazyDemotionFastAdmissionCompactHybrid(0.1),
		PaperPolicy::S3FifoGhostLazyDemotionFastAdmissionMidpointCompactHybrid(0.1),
		PaperPolicy::S3FifoLazyDemotionFastAdmissionMidpointReprieveCompactHybrid(0.1),
		PaperPolicy::S3FifoLazyDemotionFastAdmissionReprieveCompactHybrid(0.1),
		PaperPolicy::S3FifoLazyDemotionReprieveCompactHybrid(0.1),
		PaperPolicy::S3FifoLazyDemotionFastAdmissionSplitSlowReprieveCompactHybrid(0.1),
	];

	/// The allocator against the stack's own count, after a drain: `None`
	/// when they agree and every byte is on the build's node -- the slow one
	/// under `eviction_stacks_pmem`, DRAM otherwise (stated here, not asked of
	/// `NodeBytes::stack`, so a wrong split there fails this).
	fn mismatch(stack: &mut dyn PolicyStack, base: i64, boxed: u64) -> Option<(u64, crate::meta::NodeBytes)> {
		drop(stack.drain_tier_migrations());

		let live = (crate::meta::thread_live_bytes() - base) as u64;
		let bytes = stack.structure_bytes().expect("a tiered design meters itself");
		let off_node = match cfg!(feature = "eviction_stacks_pmem") {
			true => bytes.dram,
			false => bytes.slow,
		};

		(live != bytes.total() + boxed || off_node != 0).then_some((live, bytes))
	}

	#[test]
	fn every_tiered_design_counts_exactly_what_its_structures_allocated() {
		// Under `eviction_stacks_pmem` the stacks allocate on the slow node,
		// whose arenas are built on first use -- which allocates. Built here,
		// before any baseline.
		assert!(crate::numa_alloc::init_node(crate::numa_alloc::NODE_SLOW));

		let mut failures = Vec::new();

		for policy in DESIGNS {
			let base = crate::meta::thread_live_bytes();
			let mut stack = init_policy_stack(policy, MAX_SIZE);
			let boxed = crate::meta::box_bytes_of_val(&*stack);
			let mut checked = 0u64;
			let mut failure = None;

			let mut check = |stack: &mut dyn PolicyStack, what: HashedKey| {
				checked += 1;

				if failure.is_none() {
					failure = mismatch(stack, base, boxed).map(|found| (what, found));
				}
			};

			check(stack.as_mut(), 0);

			for key in 1..=N {
				stack.insert(key, SIZE);
				check(stack.as_mut(), key);

				while (stack.needs_capacity_eviction()
					|| stack.fast_bytes_used() + stack.slow_bytes_used() > MAX_SIZE)
					&& stack.evict_one().is_some()
				{
					check(stack.as_mut(), key);
				}

				// Hits: promotions, frequency bumps, reference bits.
				if key % 3 == 0 {
					for hit in [key, key / 2, key / 3] {
						if stack.contains(hit) {
							stack.update(hit);
							check(stack.as_mut(), hit);
						}
					}
				}

				// Removals: the free list.
				if key % 7 == 0 && stack.contains(key / 7) {
					stack.remove(key / 7);
					check(stack.as_mut(), key / 7);
				}
			}

			let (stack_bytes, placed) = (stack.structure_bytes().map(|b| b.total()), stack.structure_bytes());

			stack.clear();
			check(stack.as_mut(), 0);

			drop(check);

			// Before anything below allocates a message.
			drop(stack);
			let leaked = crate::meta::thread_live_bytes() - base;

			if let Some((key, (live, bytes))) = failure {
				failures.push(format!(
					"{policy}: at key {key}, the allocator holds {live} B (the {boxed} B box \
					 included) where the stack counts {bytes:?}",
				));
			}

			assert_eq!(leaked, 0, "{policy}: the stack did not free {leaked} B of what it allocated");
			assert!(checked > N, "{policy}: checked {checked} times");
			assert!(
				stack_bytes.is_some_and(|b| b > 128 * 1024),
				"{policy}: 5,000 keys should need more than one slab chunk: {placed:?}",
			);
		}

		assert!(failures.is_empty(), "{}", failures.join("\n"));
	}

	/// A flat stack does not meter itself: only a tiered cache publishes M.
	#[test]
	fn a_flat_stack_meters_nothing() {
		for policy in [PaperPolicy::LruCompact, PaperPolicy::LfuCompact, PaperPolicy::SThreeFifoCompact(0.1)] {
			assert_eq!(init_policy_stack(policy, MAX_SIZE).structure_bytes(), None, "{policy}");
		}
	}
}
