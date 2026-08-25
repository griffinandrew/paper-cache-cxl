/*
 * Copyright (c) Kia Shakiba
 *
 * This source code is licensed under the GNU AGPLv3 license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! `S3FifoLazyDemotionFastAdmissionReprieveHybridStack` —
//! `S3FifoGhostLazyDemotionFastAdmissionMidpointHybridStack` with two
//! behavioral changes, for
//! `PaperPolicy::S3FifoLazyDemotionFastAdmissionReprieveHybrid`:
//!
//! 1. **No ghost queue.** A one-access-queue key that ages out without a
//!    second access is no longer evicted at all -- see point 2 -- so there
//!    is no longer any event that ever populates a ghost list. Rather than
//!    keep a permanently-empty structure around, the ghost list and every
//!    piece of machinery that only existed to serve it
//!    (`admit_via_ghost_hit`, `is_ghost`, `trim_ghost`, the
//!    `ghost.contains()` admission check) are removed outright.
//! 2. **The one-access queue's tail is reprieved, not evicted.** Once
//!    `one_access_used` exceeds `one_access_capacity`, the tail key is
//!    moved directly into the slow tier of the main queue -- given a full
//!    life there, promotable via the ordinary `touch()`/midpoint/tail
//!    second-chance machinery -- instead of being permanently removed from
//!    the cache.
//!
//!    Critically, this relief runs *synchronously* from `insert()`/`resize()`
//!    (a new `settle_one_access()`, mirroring `settle_fast_tier()`'s
//!    relationship to the main queue's fast/slow boundary) -- **not** through
//!    `evict_one()`/`needs_capacity_eviction()`, even though that's the
//!    mechanism the predecessor variant's `evict_one_access_tail` used. The
//!    first draft of this stack routed it through `evict_one()` and hit a
//!    real bug: `PolicyWorker::apply_evictions`'s loop calls `evict_one()`
//!    whenever `needs_capacity_eviction()` is true and unconditionally
//!    erases whatever key it returns from the *entire cache* -- and if it
//!    returns `None`, `erase()`'s own fallback evicts a *random* object
//!    instead (see its doc comment: "the policy has run out of keys to
//!    evict... fall back to evicting a random object"). A reprieve is
//!    neither of those things: nothing should be permanently removed from
//!    the cache just because the one-access queue needed relief, and
//!    `over_max_size` might not even be true at that moment. Fixed by
//!    moving the relief to the same synchronous-settle pattern this whole
//!    hybrid family already uses for its OTHER internal capacity boundary
//!    (`settle_fast_tier`), which never touches `evict_one()` either --
//!    `evict_one()` in this stack is therefore purely about the main queue
//!    (the midpoint check plus the ordinary tail loop), and
//!    `needs_capacity_eviction()` stays at the trait's default `false`.
//!
//! Otherwise identical to
//! `S3FifoGhostLazyDemotionFastAdmissionMidpointHybridStack` (the fast-tier
//! one-access queue, the demotion-time reference-bit reprieve, the
//! mid-slow-segment checkpoint) -- see that stack's module doc, and the
//! stacks beneath it, for the full picture.
//!
//! ## Two physical main-queue lists, not one list plus a boundary cursor
//!
//! Every stack in this family up to and including the predecessor keeps the
//! main queue as a *single* `HashList` with a `main_boundary: Option<HashedKey>`
//! cursor marking the oldest still-fast key -- the fast tier is then the
//! contiguous prefix from the list's head up to and including that cursor,
//! and demotion is a pure relabel (flip `tier` to `Slow`, step the cursor one
//! position toward the front via `before()`) with nothing physically moving.
//! That trick is elegant *when demotion is the only thing that ever crosses
//! the boundary*, which was true for every predecessor.
//!
//! This variant breaks that premise: a reprieve has to *insert a brand-new
//! node* at the boundary, and neither `kwik::collections::HashList` nor
//! `PmemHashList` exposes an `insert_after`/`insert_before` primitive on the
//! key-addressed API -- only `push_front`, `push_back`, `move_front`, and
//! `move_back`, all of which operate on the list's absolute ends. The first
//! implementation of this stack worked around that by walking every
//! currently-fast key (`before()` from the boundary), `push_front`ing the new
//! key, then replaying all of them through `move_front` to restore their
//! order -- correct (its ordering was verified by a dedicated unit test), but
//! **O(number of currently-fast keys) per reprieve**. That is fine at
//! unit-test scale and catastrophic at real scale: benchmarked against a real
//! trace with a 6 GB fast tier, the worker thread burned ~18 minutes of CPU
//! without completing a single run, since reprieves fire continuously and the
//! fast tier holds tens of thousands of keys.
//!
//! Splitting the main queue into two physically separate lists removes the
//! problem outright rather than optimizing around it:
//!
//! * `main_fast` -- front = newest, back = oldest fast key (the demotion
//!   candidate, previously `main_boundary`).
//! * `main_slow` -- front = newest slow key (i.e. *exactly* the fast/slow
//!   boundary position), back = oldest slow key (the eviction candidate).
//!
//! The boundary is no longer a cursor into a shared list; it's just the front
//! of `main_slow`. So a reprieve is a plain `main_slow.push_front(key)` --
//! **O(1)**, landing precisely where the old code needed an O(n) walk to put
//! it, and with no risk of corrupting a boundary pointer (there isn't one).
//! Every other boundary-crossing operation gets simpler the same way:
//! demotion is `main_fast.pop_back()` + `main_slow.push_front()`, promotion is
//! `main_slow.remove()` + `main_fast.push_front()`, and eviction is
//! `main_slow.pop_back()` (falling back to `main_fast.pop_back()` only when
//! nothing has ever been demoted). All O(1).
//!
//! This also makes the midpoint cursor strictly cleaner: because `main_slow`
//! is homogeneous, a `before()` walk inside it can never wander into
//! fast-tagged territory, so the "only accept the neighbor if it's still
//! `Tier::Slow`" filter the predecessor needed at every cursor-redirect site
//! disappears entirely.
//!
//! This is not a novel shape for this crate -- `LfuHybridStack` already keeps
//! two independent frequency chains for the same reason, and
//! `LruSizedHybridStack`'s module doc records reaching the same conclusion
//! (four homogeneous lists turned out *simpler* than any cursor-based scheme
//! once more than one segment pair was involved).
//!
//! ## No mid-slow checkpoint
//!
//! Two predecessors added a reference-bit check partway through the slow
//! tier -- first an approximate sampled cursor
//! (`..._midpoint_reprieve_...`), then a real two-segment boundary
//! (`..._split_slow_reprieve_...`). Both were benchmarked against the same
//! three traces and **both left the hit rate bit-identical** to having no
//! mid-tier check at all, while the segment-boundary version measurably
//! cost GET latency and throughput.
//!
//! The reason is structural, and applies to any such checkpoint: terminal
//! eviction only ever removes the slow tier's *tail*, so an object whose
//! reference bit is set is already spared when it arrives there. An earlier
//! check can only change *when* a reaccessed object returns to DRAM, never
//! *whether* it survives -- and the extra Slow->Fast migrations it triggers
//! are pure added work on the `PolicyWorker` thread and the object map's
//! shard locks.
//!
//! This variant therefore drops the mid-tier check entirely and keeps only
//! the three reference-bit checks that do pay for themselves: at the
//! demotion boundary (`settle_fast_tier`), and at the eviction tail
//! (`evict_one`). The slow tier stays a single list.
//!
//! ## Shared DRAM-reservation overhead
//!
//! The object hashtable and this stack's own eviction bookkeeping (the three
//! `HashList`s plus the `entries` map) live in DRAM but are invisible to
//! `fast_used`/`one_access_used`, which only ever count object *values*. So
//! the fast tier's real DRAM footprint is its value bytes *plus* that
//! metadata, and demoting purely against value bytes lets total DRAM run past
//! `fast_capacity`. `shared_overhead` (see `crate::object::overhead::
//! get_hybrid_dram_shared_overhead`) is the approximate per-tracked-key cost
//! of that metadata; `reserved_overhead()` scales it by the number of tracked
//! keys, and the result is carved out of the fast-tier budget before any
//! demotion decision is made, so the budget bounds total DRAM rather than just
//! values.
//!
//! It is charged against **every tracked key**, not just the fast-tier ones: a
//! key's hashtable entry, its `entries` entry and its list node are all
//! DRAM-resident regardless of which tier its *data* sits in. This mirrors
//! `LruHybridStack`'s `stack.len()` and `LruSizedHybridStack`'s
//! `entries.len()`.
//!
//! This is a `fast_admission` variant, so it has **two** fast segments with
//! independent budgets competing for the same DRAM: the one-access queue
//! (`one_access_capacity`) and the main queue's fast portion (`fast_capacity`
//! minus that). The reservation is therefore split *proportionally* between
//! them (`reserved_shares`) rather than charged in full against each -- the
//! underlying metadata cost is real only once, and double-charging it would
//! waste usable DRAM budget for no reason. This follows `LruSizedHybridStack`,
//! which reached the same conclusion for its own two independently-capacitied
//! fast segments. The two shares always sum to exactly `reserved_overhead()`,
//! so the total effective fast budget is precisely `fast_capacity -
//! reserved_overhead()`. `main_slow` carries no capacity of its own, so it has
//! nothing to reserve against.
//!
//! The `watermarks` high/low pair is applied *on top of* the reduced effective
//! budget, never in place of it: a pass triggers at `high_bytes(capacity -
//! reserved)` and drains to `low_bytes(capacity - reserved)`.
//!
//! There is no ghost queue in this variant (see point 1 at the top), so unlike
//! its ghost-carrying ancestors there is no unbounded list of bare keys for
//! objects that are not in the cache -- nothing to charge on top of the
//! per-tracked-key term.
//!
//! One deliberate limitation: `insert()` of a brand-new key grows
//! `entries.len()` (and so the reservation) but only calls
//! `settle_one_access()`, never `settle_fast_tier()`. `main_fast` can
//! therefore sit briefly above its freshly-shrunk effective budget. This is
//! bounded and self-correcting rather than a leak: `fast_used` only ever
//! *grows* via `promote_from_one_access()` and `give_second_chance()`, both of
//! which call `settle_fast_tier()` immediately, so the next admission that
//! reaches the main queue re-settles it. Adding an unconditional
//! `settle_fast_tier()` to `insert()` would change demotion timing for the
//! overhead-free (`shared_overhead == 0`) case too, which this change
//! deliberately leaves byte-for-byte identical.

#[cfg(not(feature = "eviction_stacks_pmem"))]
use std::collections::HashMap;
#[cfg(feature = "eviction_stacks_pmem")]
use hashbrown::HashMap;

#[cfg(not(feature = "eviction_stacks_pmem"))]
use kwik::collections::HashList;
#[cfg(feature = "eviction_stacks_pmem")]
use super::pmem_collections::PmemHashList;

#[cfg(feature = "eviction_stacks_pmem")]
use crate::Hybrid;

use crate::{
	CacheSize,
	HashedKey,
	NoHasher,
	policy::PaperPolicy,
	object::ObjectSize,
	worker::policy::policy_stack::{PolicyStack, Tier, narrow_resident, watermarks},
};

/// Which live queue a key currently belongs to.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Queue {
	OneAccess,
	Main,
}

/// Combined per-key bookkeeping. `tier`/`accessed` are only meaningful
/// while `queue == Main`. `tier` is redundant with which of the two main
/// lists the key is physically in, but kept because `tier_of()` and the
/// `PolicyWorker` migration path both want it as a cheap map lookup rather
/// than a pair of `contains()` probes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct S3FifoEntry {queue: Queue,
	tier: Option<Tier>,
	/// Part of `size` that stays in DRAM in either tier; see `migrating`.
	dram_resident: u8,

	size: ObjectSize,
	accessed: bool,
}

impl S3FifoEntry {
/// The bytes that actually move between tiers when this object migrates.
	///
	/// `size` is `base_size`, which also counts the DRAM-resident remainder --
	/// the key and expiry field (inline in the object map) plus the `Expiries`
	/// entry when a TTL is set. `Object::set_data` replaces the value buffer
	/// alone, so none of that moves, and the key and expiry are already inside
	/// `shared_overhead`. Charging them to the tier counters double-counted
	/// every fast-tier object and made demotion appear to free DRAM it did not.
	#[inline]
	fn migrating(&self) -> CacheSize {
		(self.size as CacheSize).saturating_sub(self.dram_resident as CacheSize)
	}
}

/// `dram_resident` was meant to occupy padding the entry already had.
/// If this ever fails, the field is costing 4 more bytes on *every* tracked
/// object in both tiers, which defeats the point of storing it per entry.
const _: () = assert!(
	std::mem::size_of::<S3FifoEntry>() == 8,
	"S3FifoEntry grew past 8 bytes",
);


#[cfg(not(feature = "eviction_stacks_pmem"))]
type QueueList = HashList<HashedKey, NoHasher>;
#[cfg(feature = "eviction_stacks_pmem")]
type QueueList = PmemHashList<HashedKey, NoHasher>;

#[cfg(not(feature = "eviction_stacks_pmem"))]
type EntryMap = HashMap<HashedKey, S3FifoEntry, NoHasher>;
#[cfg(feature = "eviction_stacks_pmem")]
type EntryMap = HashMap<HashedKey, S3FifoEntry, NoHasher, Hybrid>;

pub struct S3FifoLazyDemotionFastAdmissionReprieveHybridStack {
	one_access_queue: QueueList,

	/// Main queue, fast portion. Front = newest, back = oldest (the
	/// demotion candidate).
	main_fast: QueueList,
	/// Main queue, slow portion. Front = newest slow key -- i.e. exactly
	/// the fast/slow boundary position -- back = oldest (the eviction
	/// candidate).
	main_slow: QueueList,

	entries: EntryMap,

	one_access_ratio: f64,
	one_access_capacity: CacheSize,
	one_access_used: CacheSize,

	fast_capacity: CacheSize,
	fast_used: CacheSize,
	slow_used: CacheSize,

	/// Approximate per-tracked-key DRAM cost of the shared structures (the
	/// object hashtable + this stack's own eviction bookkeeping), reserved
	/// proportionally out of the two fast segments' budgets -- see the module
	/// doc's "Shared DRAM-reservation overhead" section. `0` unless set via
	/// `with_shared_overhead`, so unit tests exercising the pure value-budget
	/// behaviour are unaffected.
	shared_overhead: CacheSize,

	migrations: Vec<(HashedKey, Tier)>,
}

impl S3FifoLazyDemotionFastAdmissionReprieveHybridStack {
	#[cfg(not(feature = "eviction_stacks_pmem"))]
	fn new_collections() -> (QueueList, QueueList, QueueList, EntryMap) {
		(HashList::default(), HashList::default(), HashList::default(), HashMap::default())
	}

	#[cfg(feature = "eviction_stacks_pmem")]
	fn new_collections() -> (QueueList, QueueList, QueueList, EntryMap) {
		(
			PmemHashList::with_hasher(NoHasher::default()),
			PmemHashList::with_hasher(NoHasher::default()),
			PmemHashList::with_hasher(NoHasher::default()),
			HashMap::with_hasher_in(NoHasher::default(), Hybrid),
		)
	}

	pub fn new(one_access_ratio: f64, max_size: CacheSize, fast_capacity: CacheSize) -> Self {
		let (one_access_queue, main_fast, main_slow, entries) = Self::new_collections();

		S3FifoLazyDemotionFastAdmissionReprieveHybridStack {
			one_access_queue,
			main_fast,
			main_slow,

			entries,

			one_access_ratio,
			one_access_capacity: (one_access_ratio * max_size as f64) as CacheSize,
			one_access_used: 0,

			fast_capacity,
			fast_used: 0,
			slow_used: 0,

			shared_overhead: 0,

			migrations: Vec::new(),
		}
	}

	/// Sets the approximate per-tracked-key shared-structure DRAM overhead
	/// (object hashtable + eviction stacks) reserved out of the fast-tier
	/// budget. See `crate::object::overhead::get_hybrid_dram_shared_overhead`.
	/// Builder-style so `init_policy_stack` can wire it in without disturbing
	/// `new`'s signature (unit tests keep the default `0`).
	pub fn with_shared_overhead(mut self, overhead: CacheSize) -> Self {
		self.shared_overhead = overhead;
		self
	}

	/// Total DRAM currently reserved for shared per-object metadata across
	/// *both* tiers (`tracked key count x shared_overhead`). Counted over
	/// `entries` -- i.e. every tracked key, whichever of the three lists it
	/// currently sits in and whichever tier its data is in -- because the
	/// hashtable entry, the `entries` entry and the list node are DRAM
	/// regardless. Split between the two fast segments by
	/// [`Self::reserved_shares`].
	fn reserved_overhead(&self) -> CacheSize {
		self.entries.len() as CacheSize * self.shared_overhead
	}

	/// The main queue's fast segment's *configured* budget: `fast_capacity`
	/// minus the one-access queue's reservation. This is a `fast_admission`
	/// variant, so the one-access queue is fast-tier and carves its budget out
	/// of the same DRAM allowance.
	fn main_fast_capacity(&self) -> CacheSize {
		self.fast_capacity.saturating_sub(self.one_access_capacity)
	}

	/// Splits [`Self::reserved_overhead`] proportionally between the two
	/// independently-capacitied fast segments, returning `(one_access_share,
	/// main_fast_share)`. The shares always sum to exactly the total, so the
	/// metadata is charged once overall rather than once per segment.
	/// `(0, 0)` when neither segment has any budget to proportion against.
	/// Follows `LruSizedHybridStack::reserved_shares`, including its "first
	/// segment takes the floor, second takes the remainder" rounding.
	fn reserved_shares(&self) -> (CacheSize, CacheSize) {
		let reserved = self.reserved_overhead();

		let one_access_capacity = self.one_access_capacity;
		let main_fast_capacity = self.main_fast_capacity();
		let total_capacity = one_access_capacity + main_fast_capacity;

		if total_capacity == 0 {
			return (0, 0);
		}

		let one_access_share = ((reserved as u128 * one_access_capacity as u128)
			/ total_capacity as u128) as CacheSize;
		let main_fast_share = reserved.saturating_sub(one_access_share);

		(one_access_share, main_fast_share)
	}

	/// The one-access queue's *effective* byte budget: its configured
	/// `one_access_capacity` minus its share of the shared-metadata
	/// reservation. Saturates to 0 when the reservation alone meets or exceeds
	/// the configured budget, in which case every admitted key is reprieved
	/// straight into `main_slow` -- a pure internal migration; nothing is ever
	/// evicted here.
	fn effective_one_access_capacity(&self) -> CacheSize {
		self.one_access_capacity.saturating_sub(self.reserved_shares().0)
	}

	/// The main queue's fast segment's *effective* byte budget:
	/// [`Self::main_fast_capacity`] minus its share of the shared-metadata
	/// reservation. This is the value `settle_fast_tier` takes the
	/// `watermarks` high/low pair against -- the reservation shrinks the
	/// budget, the watermarks then scale that reduced value.
	fn effective_main_fast_capacity(&self) -> CacheSize {
		self.main_fast_capacity().saturating_sub(self.reserved_shares().1)
	}

	pub fn tier_of(&self, key: HashedKey) -> Option<Tier> {
		let entry = self.entries.get(&key)?;

		match entry.queue {
			Queue::OneAccess => Some(Tier::Fast),
			Queue::Main => entry.tier,
		}
	}

	/// `new_resident` refreshes the entry's DRAM-resident remainder: a re-set
	/// can add or drop a TTL, which changes it by the `Expiries` entry's cost.
	/// Without this the entry keeps its old remainder and every later
	/// migration moves the wrong number of bytes.
	fn resize_key(&mut self, key: HashedKey, new_size: ObjectSize, new_resident: u8) {
		let Some(entry) = self.entries.get_mut(&key) else { return };

		let old_migrating = entry.migrating();
		entry.size = new_size;
		entry.dram_resident = new_resident;
		let delta = entry.migrating() as i64 - old_migrating as i64;

		match (entry.queue, entry.tier) {
			(Queue::OneAccess, _) => {
				self.one_access_used = (self.one_access_used as i64 + delta).max(0) as CacheSize;
			},

			(Queue::Main, Some(Tier::Fast)) => {
				self.fast_used = (self.fast_used as i64 + delta).max(0) as CacheSize;
			},

			(Queue::Main, Some(Tier::Slow)) => {
				self.slow_used = (self.slow_used as i64 + delta).max(0) as CacheSize;
			},

			(Queue::Main, None) => {},
		}
	}

	fn touch(&mut self, key: HashedKey) {
		match self.entries.get(&key).map(|entry| entry.queue) {
			Some(Queue::OneAccess) => self.promote_from_one_access(key),
			Some(Queue::Main) => self.mark_accessed(key),
			None => {},
		}
	}

	fn mark_accessed(&mut self, key: HashedKey) {
		if let Some(entry) = self.entries.get_mut(&key) {
			entry.accessed = true;
		}
	}

	fn promote_from_one_access(&mut self, key: HashedKey) {
		let Some(entry) = self.entries.get(&key) else { return };
		let size = entry.size;
		let dram_resident = entry.dram_resident;
		// Tier arithmetic moves only what migrates; `size` still rebuilds the entry.
		let size_bytes = entry.migrating();

		self.one_access_queue.remove(&key);
		self.one_access_used = self.one_access_used.saturating_sub(size_bytes);

		self.main_fast.push_front(key);
		self.entries.insert(key, S3FifoEntry { dram_resident, queue: Queue::Main,
			tier: Some(Tier::Fast),
			size,
			accessed: false,
		});
		self.fast_used += size_bytes;

		self.settle_fast_tier();
	}

	/// The eviction-time second chance, shared with the demotion-boundary
	/// reprieve: both mean "this key's reference bit is set, so spare it
	/// and move it to the front of the fast list".
	fn give_second_chance(&mut self, key: HashedKey) {
		let Some(entry) = self.entries.get(&key).copied() else { return };
		let size = entry.migrating();

		match entry.tier {
			// Already fast (only reachable from `evict_one`'s fast-tail
			// fallback, i.e. nothing has ever been demoted): just reorder
			// to the front, no tier change and no byte movement.
			Some(Tier::Fast) => {
				self.main_fast.move_front(&key);

				if let Some(entry) = self.entries.get_mut(&key) {
					entry.accessed = false;
				}
			},

			Some(Tier::Slow) => {
				self.main_slow.remove(&key);
				self.main_fast.push_front(key);

				if let Some(entry) = self.entries.get_mut(&key) {
					entry.tier = Some(Tier::Fast);
					entry.accessed = false;
				}

				self.slow_used = self.slow_used.saturating_sub(size);
				self.fast_used += size;

				},

			None => return,
		}

		self.settle_fast_tier();

		// Only record a migration if the key actually ended up Fast -- the
		// `settle_fast_tier` above can immediately demote it right back out
		// when the fast tier is at capacity, in which case that call has
		// already pushed the correct `Tier::Slow` migration itself.
		if self.entries.get(&key).and_then(|entry| entry.tier) == Some(Tier::Fast) {
			self.migrations.push((key, Tier::Fast));
		}
	}

	/// Demotes oldest-first from `main_fast` into `main_slow` back down to the
	/// shared *low* watermark -- but only once `fast_used` has crossed the
	/// shared *high* watermark in the first place -- giving any key whose
	/// reference bit is set a reprieve (moved to the front of `main_fast`, bit
	/// cleared) instead. Terminates even when every fast key's bit is set,
	/// since each reprieve clears one bit.
	///
	/// The ceiling is still `effective_main_fast_capacity()`: `fast_capacity`
	/// minus the one-access queue's `one_access_capacity` reservation -- this
	/// is a `fast_admission` variant, so its one-access queue is fast-tier and
	/// competes for the same DRAM budget -- and now also minus this segment's
	/// share of the shared per-object metadata reservation
	/// (`reserved_shares().1`; see the module doc's "Shared DRAM-reservation
	/// overhead" section), which makes the budget bound total DRAM rather than
	/// just object values. The `watermarks` helpers are applied *on top of*
	/// that effective value, never in place of it; only when a pass fires and
	/// how far it drains change, never the ceiling itself.
	///
	/// Previously this drained back to exactly the effective capacity, which
	/// pinned the tier at 100% utilisation and made essentially every
	/// promotion demote exactly one object (see the `watermarks` module doc).
	/// Setting both `FAST_TIER_HIGH_WATERMARK` and `FAST_TIER_LOW_WATERMARK`
	/// to `1.0` restores that behaviour byte-for-byte.
	///
	/// Per-demotion bookkeeping is deliberately untouched: each demoted object
	/// still retags its entry, still moves between `main_fast` and
	/// `main_slow`, still moves `fast_used`/`slow_used` by its own size, and
	/// still emits exactly one `Tier::Slow` migration. The reprieve branch is
	/// likewise unchanged.
	fn settle_fast_tier(&mut self) {
		let effective_capacity = self.effective_main_fast_capacity();

		if self.fast_used <= watermarks::high_bytes(effective_capacity) {
			return;
		}

		let drain_target = watermarks::low_bytes(effective_capacity);

		while self.fast_used > drain_target {
			let Some(candidate) = self.main_fast.back().copied() else { break };

			let accessed = self.entries.get(&candidate).map(|entry| entry.accessed).unwrap_or(false);

			if accessed {
				self.main_fast.move_front(&candidate);

				if let Some(entry) = self.entries.get_mut(&candidate) {
					entry.accessed = false;
				}

				continue;
			}

			let size = self.entries.get(&candidate).map(|entry| entry.migrating()).unwrap_or(0);

			self.main_fast.pop_back();
			self.main_slow.push_front(candidate);

			if let Some(entry) = self.entries.get_mut(&candidate) {
				entry.tier = Some(Tier::Slow);
			}

			self.fast_used = self.fast_used.saturating_sub(size);
			self.slow_used += size;

			self.migrations.push((candidate, Tier::Slow));
		}
	}

	/// Relieves one-access-queue pressure by moving its tail(s) to the front
	/// of `main_slow` -- which *is* the fast/slow boundary position, so this
	/// is a plain O(1) `push_front` (see the module doc's "Two physical
	/// main-queue lists" section for why this used to be an O(n) walk).
	/// Called synchronously from `insert()`/`resize()`, exactly mirroring
	/// `settle_fast_tier()`'s relationship to the fast/slow boundary. A pure
	/// internal migration: nothing is ever removed from the cache here, so
	/// this must never be routed through `evict_one()`/
	/// `needs_capacity_eviction()` -- see the module doc for the bug that
	/// caused.
	///
	/// The budget it settles against is `effective_one_access_capacity()` --
	/// the configured `one_access_capacity` minus this segment's share of the
	/// shared per-object metadata reservation (see the module doc). Hoisted
	/// out of the loop: `entries.len()` and both capacities are fixed for the
	/// duration of a settle (a reprieve moves a key between lists, it never
	/// adds or removes one), so re-reading it per iteration could not change
	/// the target. With `shared_overhead == 0` the share is 0 and this is
	/// byte-for-byte the previous `one_access_capacity` comparison.
	///
	/// No watermarks here: unlike `settle_fast_tier` this boundary was never
	/// given a high/low pair, and the reservation is not an excuse to
	/// introduce one.
	fn settle_one_access(&mut self) {
		let effective_capacity = self.effective_one_access_capacity();

		while self.one_access_used > effective_capacity {
			let Some(key) = self.one_access_queue.pop_back() else { break };
			let Some(entry) = self.entries.get(&key).copied() else { continue };
			let size = entry.migrating();

			self.one_access_used = self.one_access_used.saturating_sub(size);

			self.main_slow.push_front(key);

			if let Some(stored) = self.entries.get_mut(&key) {
				stored.queue = Queue::Main;
				stored.tier = Some(Tier::Slow);
				stored.accessed = false;
			}

			self.slow_used += size;

			self.migrations.push((key, Tier::Slow));
		}
	}
}

impl PolicyStack for S3FifoLazyDemotionFastAdmissionReprieveHybridStack {
	fn is_policy(&self, policy: &PaperPolicy) -> bool {
		matches!(policy, PaperPolicy::S3FifoLazyDemotionFastAdmissionReprieveHybrid(ratio) if *ratio == self.one_access_ratio)
	}

	fn len(&self) -> usize {
		self.entries.len()
	}

	fn contains(&self, key: HashedKey) -> bool {
		self.entries.contains_key(&key)
	}

	fn insert(&mut self, key: HashedKey, size: ObjectSize) {
		self.insert_resident(key, size, 0);
	}

	fn insert_resident(&mut self, key: HashedKey, size: ObjectSize, dram_resident: ObjectSize) {
		let dram_resident = narrow_resident(dram_resident);
		if self.entries.contains_key(&key) {
			self.resize_key(key, size, dram_resident);
			self.touch(key);
			return;
		}

		self.one_access_queue.push_front(key);
		self.entries.insert(key, S3FifoEntry { dram_resident, queue: Queue::OneAccess,
			tier: None,
			size,
			accessed: false,
		});
		self.one_access_used += (size as CacheSize).saturating_sub(dram_resident as CacheSize);

		self.settle_one_access();
	}

	fn update(&mut self, key: HashedKey) {
		if self.entries.contains_key(&key) {
			self.touch(key);
		}
	}

	fn remove(&mut self, key: HashedKey) {
		let Some(entry) = self.entries.remove(&key) else { return };
		let size = entry.migrating();

		match entry.queue {
			Queue::OneAccess => {
				self.one_access_queue.remove(&key);
				self.one_access_used = self.one_access_used.saturating_sub(size);
			},

			Queue::Main => match entry.tier {
				Some(Tier::Fast) => {
					self.main_fast.remove(&key);
					self.fast_used = self.fast_used.saturating_sub(size);
				},

				Some(Tier::Slow) => {
					self.main_slow.remove(&key);
					self.slow_used = self.slow_used.saturating_sub(size);

						},

				None => {},
			},
		}
	}

	fn resize(&mut self, max_size: CacheSize) {
		self.one_access_capacity = (self.one_access_ratio * max_size as f64) as CacheSize;
		self.settle_one_access();
		self.settle_fast_tier();
	}

	fn clear(&mut self) {
		self.one_access_queue.clear();
		self.main_fast.clear();
		self.main_slow.clear();
		self.entries.clear();

		self.one_access_used = 0;
		self.fast_used = 0;
		self.slow_used = 0;
		self.migrations.clear();
	}

	fn evict_one(&mut self) -> Option<HashedKey> {
		// The one-access queue never reaches here -- its own capacity
		// pressure is relieved synchronously by `settle_one_access()` (see
		// the module doc), the same way the main queue's fast/slow boundary
		// is settled by `settle_fast_tier()` rather than through eviction.
		// This is purely the main queue's ordinary tail loop.
		loop {
			// The slow tail is the real eviction candidate; fall back to
			// the fast tail only when nothing has ever been demoted.
			let (key, from_slow) = match self.main_slow.back().copied() {
				Some(key) => (key, true),
				None => (self.main_fast.back().copied()?, false),
			};

			let accessed = self.entries.get(&key).map(|entry| entry.accessed).unwrap_or(false);

			if accessed {
				self.give_second_chance(key);
				continue;
			}

			if from_slow {
				self.main_slow.pop_back();
			} else {
				self.main_fast.pop_back();
			}

			let removed = self.entries.remove(&key);
			let size = removed.map(|entry| entry.migrating()).unwrap_or(0);

			if from_slow {
				self.slow_used = self.slow_used.saturating_sub(size);
				} else {
				self.fast_used = self.fast_used.saturating_sub(size);
			}

			return Some(key);
		}
	}

	fn resize_fast_tier(&mut self, size: CacheSize) {
		self.fast_capacity = size;
		self.settle_fast_tier();
	}

	fn drain_tier_migrations(&mut self) -> Vec<(HashedKey, Tier)> {
		std::mem::take(&mut self.migrations)
	}

	fn fast_bytes_used(&self) -> CacheSize {
		self.fast_used + self.one_access_used
	}

	fn slow_bytes_used(&self) -> CacheSize {
		self.slow_used
	}

	fn fast_object_count(&self) -> usize {
		self.main_fast.len() + self.one_access_queue.len()
	}

	fn slow_object_count(&self) -> usize {
		self.main_slow.len()
	}
}

#[cfg(test)]
mod tests {
	use super::*;

	fn drain(stack: &mut S3FifoLazyDemotionFastAdmissionReprieveHybridStack) -> Vec<(HashedKey, Tier)> {
		stack.drain_tier_migrations()
	}

	/// `insert` + `update` -- the admit-into-the-one-access-queue-then-promote
	/// pairing every main-queue fast-tier test in this module already uses,
	/// since this stack never admits a fresh key straight into `main_fast`.
	fn promote(stack: &mut S3FifoLazyDemotionFastAdmissionReprieveHybridStack, key: HashedKey, size: ObjectSize) {
		stack.insert(key, size);
		stack.update(key);
	}

	/// Smallest *effective* main-fast budget (i.e. the value
	/// `effective_main_fast_capacity()` returns, `fast_capacity` minus the
	/// one-access reservation) whose *low* watermark still leaves room for
	/// `bytes`. Lets the fast-tier tests state their expectations in whole
	/// objects instead of hard-coded byte thresholds, so they hold at whatever
	/// `FAST_TIER_HIGH_WATERMARK`/`FAST_TIER_LOW_WATERMARK` pair is configured
	/// rather than only at the default ratios. The `while` loop absorbs
	/// the truncation in `watermarks::low_bytes`' `as u64` cast, which a bare
	/// `ceil()` on its own can land a byte short of.
	fn effective_capacity_holding(bytes: CacheSize) -> CacheSize {
		let mut capacity = (bytes as f64 / watermarks::low()).ceil() as CacheSize;

		while watermarks::low_bytes(capacity) < bytes {
			capacity += 1;
		}

		capacity
	}

	#[test]
	fn admission_always_lands_in_one_access_queue_fast() {
		let mut stack = S3FifoLazyDemotionFastAdmissionReprieveHybridStack::new(1.0, 1_000, 1_000);

		stack.insert(1, 10);
		stack.insert(2, 10);

		assert_eq!(stack.tier_of(1), Some(Tier::Fast));
		assert_eq!(stack.tier_of(2), Some(Tier::Fast));
		assert_eq!(drain(&mut stack), Vec::new());
	}

	#[test]
	fn a_key_aging_out_without_reaccess_is_moved_to_slow_instead_of_evicted() {
		// one_access_capacity = 0.01 * 1_000 = 10 -- fits exactly one 10-byte
		// key. Admitting a second pushes one_access_used to 20 > 10,
		// synchronously reprieving the oldest (key 1) from insert() itself.
		let mut stack = S3FifoLazyDemotionFastAdmissionReprieveHybridStack::new(0.01, 1_000, 1_000);

		stack.insert(1, 10);
		drain(&mut stack);
		assert_eq!(stack.tier_of(1), Some(Tier::Fast), "still in the one-access queue");

		stack.insert(2, 10);

		assert!(stack.contains(1), "the key must still be tracked, not gone");
		assert_eq!(stack.tier_of(1), Some(Tier::Slow), "aged-out key should land directly in the main queue's slow tier");
		assert_eq!(stack.tier_of(2), Some(Tier::Fast), "the newer key stays in the one-access queue");

		let migrations = drain(&mut stack);
		assert_eq!(migrations, vec![(1, Tier::Slow)], "a real Fast(DRAM)->Slow(PMEM) migration must still be recorded");
	}

	#[test]
	fn a_reprieved_key_can_still_be_promoted_by_a_later_access() {
		let mut stack = S3FifoLazyDemotionFastAdmissionReprieveHybridStack::new(0.01, 1_000, 1_000);

		// one_access_capacity = 0.01 * 1_000 = 10, fits exactly one key.
		// insert(2) pushes past it, reprieving key 1 (the oldest); insert(3)
		// pushes past it again, reprieving key 2 -- leaving key 3 sitting
		// safely in the one-access queue (untouched, under capacity) and
		// both 1 and 2 in main_slow, in that order: main_slow = [2, 1]
		// (2 at the front/freshest, 1 at the tail).
		stack.insert(1, 10);
		stack.insert(2, 10);
		stack.insert(3, 10);
		drain(&mut stack);

		assert_eq!(stack.tier_of(1), Some(Tier::Slow));
		assert_eq!(stack.tier_of(2), Some(Tier::Slow));
		assert_eq!(stack.tier_of(3), Some(Tier::Fast), "still sitting untouched in the one-access queue");

		// Re-access key 1 (the tail): sets the reference bit but must not
		// itself move or migrate it yet.
		stack.update(1);
		assert_eq!(stack.tier_of(1), Some(Tier::Slow));
		assert_eq!(drain(&mut stack), Vec::new());

		// The tail check finds key 1's bit set and gives it a second
		// chance instead of evicting it; eviction then proceeds to the
		// real (still genuinely cold) tail, key 2, in the same call.
		let evicted = stack.evict_one();

		assert_eq!(evicted, Some(2));
		assert_eq!(stack.tier_of(1), Some(Tier::Fast), "the reprieved key should have been promoted via the ordinary second chance");
	}

	#[test]
	fn reprieve_does_not_disturb_existing_fast_key_order() {
		// A comfortable one-access budget (ratio 1.0) during setup, so keys
		// 1-3 each safely survive their own insert()'s settle_one_access
		// before the very next line's update() promotes them via touch().
		// fast_capacity is set well above one_access_capacity (1_000) too --
		// effective_main_fast_capacity is fast_capacity minus
		// one_access_capacity, so leaving them equal would zero it out and
		// demote every promoted key immediately.
		let mut stack = S3FifoLazyDemotionFastAdmissionReprieveHybridStack::new(1.0, 1_000, 10_000);

		for key in 1..=3u64 {
			stack.insert(key, 10);
			stack.update(key);
		}
		drain(&mut stack);

		assert_eq!(stack.tier_of(1), Some(Tier::Fast));
		assert_eq!(stack.tier_of(2), Some(Tier::Fast));
		assert_eq!(stack.tier_of(3), Some(Tier::Fast));

		// Admit a fourth key that stays in the one-access queue (never
		// touched), then shrink the one-access budget to 0 -- forcing
		// settle_one_access to move it into main_slow synchronously, from
		// within this resize() call.
		stack.insert(4, 10);
		assert_eq!(stack.tier_of(4), Some(Tier::Fast), "still sitting untouched in the one-access queue");

		stack.resize(0);

		assert_eq!(stack.tier_of(4), Some(Tier::Slow));
		assert_eq!(drain(&mut stack), vec![(4, Tier::Slow)]);

		// The three original fast keys must all still be Fast and still in
		// their original oldest-first order -- shrink the fast budget to 0
		// and confirm every one demotes in that order, none skipped.
		stack.resize_fast_tier(0);
		let migrations = drain(&mut stack);

		assert_eq!(migrations, vec![(1, Tier::Slow), (2, Tier::Slow), (3, Tier::Slow)], "demotion order must be oldest-first, and no fast key may be silently skipped");
		assert_eq!(stack.tier_of(1), Some(Tier::Slow));
		assert_eq!(stack.tier_of(2), Some(Tier::Slow));
		assert_eq!(stack.tier_of(3), Some(Tier::Slow));
		assert_eq!(stack.fast_bytes_used(), 0);
	}

	#[test]
	fn reprieve_never_counts_toward_fast_bytes_used() {
		let mut stack = S3FifoLazyDemotionFastAdmissionReprieveHybridStack::new(0.0, 1_000, 1_000);

		stack.insert(1, 10);

		assert_eq!(stack.fast_bytes_used(), 0, "a reprieved key must never be counted as fast, even transiently");
		assert_eq!(stack.slow_bytes_used(), 10);
	}

	#[test]
	fn an_accessed_fast_boundary_key_is_reprieved_at_demotion_time_not_demoted() {
		// one_access_capacity = 1.0 * 1_000, so the effective main-fast budget
		// is whatever is added on top of it -- sized here so a triggered pass
		// drains to a low watermark that still holds one of these two 10-byte
		// objects, i.e. exactly one demotion. (Was a hard-coded 1_010, giving
		// an effective 10: correct back when a pass drained to the ceiling,
		// but under the watermarks a 10-byte effective budget triggers on the
		// very first promotion and drains to 7.)
		let mut stack = S3FifoLazyDemotionFastAdmissionReprieveHybridStack::new(1.0, 1_000, 1_000 + effective_capacity_holding(10));

		stack.insert(1, 10);
		stack.update(1);
		drain(&mut stack);

		stack.update(1);
		assert_eq!(drain(&mut stack), Vec::new());

		stack.insert(2, 10);
		stack.update(2);
		let migrations = drain(&mut stack);

		assert_eq!(stack.tier_of(1), Some(Tier::Fast));
		assert_eq!(stack.tier_of(2), Some(Tier::Slow));
		assert_eq!(migrations, vec![(2, Tier::Slow)]);
	}

	fn build_five_key_stack() -> S3FifoLazyDemotionFastAdmissionReprieveHybridStack {
		let mut stack = S3FifoLazyDemotionFastAdmissionReprieveHybridStack::new(1.0, 1_000, 1_020);

		for key in 1..=5u64 {
			stack.insert(key, 10);
			stack.update(key);
		}

		drain(&mut stack);
		stack
	}

	#[test]
	fn evict_one_gives_an_accessed_slow_key_a_second_chance() {
		// Same sizing rationale as
		// `an_accessed_fast_boundary_key_is_reprieved_at_demotion_time_not_demoted`
		// above: an effective main-fast budget holding exactly one 10-byte
		// object at the low watermark, so promoting the second key demotes the
		// first and nothing more.
		let mut stack = S3FifoLazyDemotionFastAdmissionReprieveHybridStack::new(1.0, 1_000, 1_000 + effective_capacity_holding(10));

		stack.insert(1, 10);
		stack.update(1);
		drain(&mut stack);

		stack.insert(2, 10);
		stack.update(2);
		drain(&mut stack);
		assert_eq!(stack.tier_of(1), Some(Tier::Slow));

		stack.update(1);
		assert_eq!(drain(&mut stack), Vec::new());

		let evicted = stack.evict_one();

		assert_eq!(evicted, Some(2));
		assert_eq!(stack.tier_of(1), Some(Tier::Fast));
		assert_eq!(stack.contains(2), false);
	}

	#[test]
	fn evict_one_falls_back_to_the_fast_tail_when_nothing_has_ever_been_demoted() {
		let mut stack = S3FifoLazyDemotionFastAdmissionReprieveHybridStack::new(1.0, 1_000, 10_000);

		for key in 1..=3u64 {
			stack.insert(key, 10);
			stack.update(key);
		}
		drain(&mut stack);

		assert_eq!(stack.slow_object_count(), 0, "nothing should have been demoted yet");

		// With main_slow empty, the oldest fast key is the only candidate.
		assert_eq!(stack.evict_one(), Some(1));
		assert!(!stack.contains(1));
		assert_eq!(stack.fast_bytes_used(), 20);
	}

	#[test]
	fn remove_and_clear_reset_bookkeeping() {
		let mut stack = S3FifoLazyDemotionFastAdmissionReprieveHybridStack::new(1.0, 1_000, 1_000);

		stack.insert(1, 10);
		stack.insert(2, 10);
		stack.update(1);
		drain(&mut stack);

		stack.remove(1);
		assert_eq!(stack.contains(1), false);

		stack.remove(2);
		assert_eq!(stack.contains(2), false);

		stack.insert(3, 10);
		stack.clear();

		assert_eq!(stack.len(), 0);
		assert_eq!(stack.tier_of(3), None);
		assert_eq!(stack.evict_one(), None);
	}

	#[test]
	fn fast_and_slow_gauges_include_one_access_queue_on_the_fast_side() {
		let mut stack = S3FifoLazyDemotionFastAdmissionReprieveHybridStack::new(1.0, 1_000, 1_000);

		stack.insert(1, 10);
		stack.insert(2, 10);
		drain(&mut stack);

		assert_eq!(stack.fast_bytes_used(), 20);
		assert_eq!(stack.slow_bytes_used(), 0);
		assert_eq!(stack.fast_object_count(), 2);
		assert_eq!(stack.slow_object_count(), 0);
	}

	// -- Shared high/low watermarks (`super::watermarks`) --------------------
	//
	// The ratios are process-wide `OnceLock`s read from the environment, so a
	// test cannot set them without racing every other test in the binary.
	// These compute their expectations from `watermarks::high_bytes()` /
	// `watermarks::low_bytes()` of the effective budget instead, and therefore
	// hold at any configured ratio rather than only at the default ratios.

	/// One-access ratio for the watermark tests. Small enough that
	/// `one_access_capacity` is a known round number, large enough that
	/// `insert()`'s `settle_one_access()` never reprieves a key out from under
	/// the `update()` that is about to promote it.
	const WM_ONE_ACCESS_RATIO: f64 = 0.1;
	const WM_MAX_SIZE: CacheSize = 100_000;

	/// The effective main-fast budget every watermark test works against --
	/// i.e. exactly what `effective_main_fast_capacity()` returns for
	/// `watermark_stack()`, since `fast_capacity` is built as
	/// `one_access_capacity + WM_EFFECTIVE`.
	const WM_EFFECTIVE: CacheSize = 1_000;

	fn watermark_stack() -> S3FifoLazyDemotionFastAdmissionReprieveHybridStack {
		let one_access_capacity = (WM_ONE_ACCESS_RATIO * WM_MAX_SIZE as f64) as CacheSize;

		let stack = S3FifoLazyDemotionFastAdmissionReprieveHybridStack::new(
			WM_ONE_ACCESS_RATIO,
			WM_MAX_SIZE,
			one_access_capacity + WM_EFFECTIVE,
		);

		assert_eq!(stack.effective_main_fast_capacity(), WM_EFFECTIVE);
		stack
	}

	/// (a) The trigger is a strict `>`, so usage sitting right *on* the high
	/// watermark -- the largest usage that is not over it -- must leave the
	/// fast tier completely alone.
	#[test]
	fn fast_usage_at_the_high_watermark_triggers_no_demotion() {
		let high = watermarks::high_bytes(WM_EFFECTIVE);

		let mut stack = watermark_stack();

		// Two objects summing to exactly the high watermark.
		promote(&mut stack, 1, (high - 1) as ObjectSize);
		promote(&mut stack, 2, 1);

		let migrations = drain(&mut stack);

		assert_eq!(stack.fast_bytes_used(), high);
		assert!(
			!migrations.iter().any(|(_, tier)| *tier == Tier::Slow),
			"usage at the high watermark must not trigger a demotion pass, got {migrations:?}",
		);
		assert_eq!(stack.tier_of(1), Some(Tier::Fast));
		assert_eq!(stack.tier_of(2), Some(Tier::Fast));
		assert_eq!(stack.slow_bytes_used(), 0);
		assert_eq!(stack.slow_object_count(), 0);
	}

	/// (b) One byte past the high watermark -- the smallest possible overshoot
	/// -- must fire a pass, and it must take `main_fast`'s oldest key rather
	/// than the key that just arrived.
	#[test]
	fn fast_usage_above_the_high_watermark_triggers_a_demotion_pass() {
		let high = watermarks::high_bytes(WM_EFFECTIVE);

		let mut stack = watermark_stack();

		promote(&mut stack, 1, high as ObjectSize);
		assert!(
			!drain(&mut stack).iter().any(|(_, tier)| *tier == Tier::Slow),
			"filling exactly to the high watermark must not demote anything yet",
		);

		promote(&mut stack, 2, 1);
		let migrations = drain(&mut stack);

		assert!(
			migrations.contains(&(1, Tier::Slow)),
			"usage past the high watermark must trigger a demotion pass, got {migrations:?}",
		);
		assert_eq!(stack.tier_of(1), Some(Tier::Slow));
		assert_eq!(stack.tier_of(2), Some(Tier::Fast));
		assert!(stack.fast_bytes_used() <= watermarks::low_bytes(WM_EFFECTIVE));
	}

	/// (c) A triggered pass keeps going down to the *low* watermark, not just
	/// back under the ceiling. With the defaults this drains 960 -> 750 across
	/// 21 demotions; the pre-watermark drain-to-ceiling loop would have
	/// stopped after a single one, at 950.
	#[test]
	fn a_triggered_pass_drains_all_the_way_to_the_low_watermark() {
		let size: ObjectSize = 10;
		let bytes = size as CacheSize;

		let high = watermarks::high_bytes(WM_EFFECTIVE);
		let low = watermarks::low_bytes(WM_EFFECTIVE);

		let mut stack = watermark_stack();

		// Exactly one object past the high watermark, so precisely one pass
		// fires -- with plenty of resident objects for it to chew through
		// before it reaches the low watermark.
		let count = high / bytes + 1;

		for key in 1..=count {
			promote(&mut stack, key, size);
		}

		let migrations = drain(&mut stack);
		let demoted = migrations.iter().filter(|(_, tier)| *tier == Tier::Slow).count() as CacheSize;

		// The pass halts at the first whole-object multiple at or below the
		// low watermark -- well under the effective ceiling, which is where
		// the old loop would have left it.
		let expected_used = low - low % bytes;

		assert_eq!(stack.fast_bytes_used(), expected_used);
		assert!(stack.fast_bytes_used() <= low);

		// ...and that is strictly tighter than the ceiling the old
		// drain-to-capacity loop stopped at -- except under the degenerate
		// `FAST_TIER_LOW_WATERMARK=1.0` setting, which deliberately restores
		// exactly that old behaviour and so has nothing to be tighter than.
		if low < WM_EFFECTIVE {
			assert!(stack.fast_bytes_used() < WM_EFFECTIVE, "a pass must drain past the ceiling, down to the low watermark");
		}

		assert_eq!(demoted, (count * bytes - expected_used) / bytes);
	}

	/// (d) Every byte counter and object count still agrees with the per-key
	/// tier tags once a full watermark drain has run -- the same per-demotion
	/// bookkeeping must have run once per demoted object, no more and no less.
	#[test]
	fn byte_and_object_counters_stay_consistent_across_a_watermark_drain() {
		let size: ObjectSize = 10;
		let bytes = size as CacheSize;

		let count = watermarks::high_bytes(WM_EFFECTIVE) / bytes + 1;

		let mut stack = watermark_stack();

		for key in 1..=count {
			promote(&mut stack, key, size);
		}

		drain(&mut stack);

		// `promote`'s `update()` empties the one-access queue every time, so
		// the fast gauges here are purely `main_fast`'s.
		assert_eq!(stack.one_access_used, 0);

		let fast_objects = stack.fast_object_count() as CacheSize;
		let slow_objects = stack.slow_object_count() as CacheSize;

		// Nothing was inserted, evicted or resized mid-pass, so every object
		// is still tracked, still `size` bytes, and still on exactly one side
		// of the fast/slow line.
		assert!(fast_objects > 0 && slow_objects > 0);
		assert_eq!(fast_objects + slow_objects, count);
		assert_eq!(stack.len() as CacheSize, count);

		assert_eq!(stack.fast_bytes_used(), fast_objects * bytes);
		assert_eq!(stack.slow_bytes_used(), slow_objects * bytes);
		assert_eq!(stack.fast_bytes_used() + stack.slow_bytes_used(), count * bytes);

		// And the aggregate counts agree with the per-key tier tags.
		let tagged_fast = (1..=count).filter(|key| stack.tier_of(*key) == Some(Tier::Fast)).count();
		let tagged_slow = (1..=count).filter(|key| stack.tier_of(*key) == Some(Tier::Slow)).count();

		assert_eq!(tagged_fast as CacheSize, fast_objects);
		assert_eq!(tagged_slow as CacheSize, slow_objects);
	}

	// -- Shared DRAM-reservation overhead ------------------------------------
	//
	// Every test *above* this line constructs the stack without
	// `with_shared_overhead`, so `shared_overhead` is 0, `reserved_overhead()`
	// is 0, `reserved_shares()` is `(0, 0)`, and both effective capacities are
	// byte-for-byte their pre-reservation values. That is why none of them
	// needed rescaling for this change -- not because their assertions were
	// weakened.

	/// Builds a stack whose two fast segments carry *equal* budgets --
	/// `one_access_capacity == fast_capacity - one_access_capacity == segment`
	/// -- so `reserved_shares()` splits the reservation exactly in half and
	/// each segment's effective budget is a hand-checkable `segment -
	/// reserved / 2`.
	fn equal_segment_stack(segment: CacheSize, overhead: CacheSize) -> S3FifoLazyDemotionFastAdmissionReprieveHybridStack {
		let stack = S3FifoLazyDemotionFastAdmissionReprieveHybridStack::new(0.5, 2 * segment, 2 * segment)
			.with_shared_overhead(overhead);

		assert_eq!(stack.one_access_capacity, segment);
		assert_eq!(stack.main_fast_capacity(), segment);

		stack
	}

	/// The reservation is charged *once* overall and divided between the two
	/// independently-capacitied fast segments in proportion to their budgets
	/// -- not charged in full against each.
	#[test]
	fn shared_overhead_splits_proportionally_between_the_two_fast_segments() {
		// one_access_capacity = 0.25 * 4_000 = 1_000, so main_fast_capacity =
		// 4_000 - 1_000 = 3_000. A 400-byte/key reservation over 2 tracked
		// keys is 800 bytes, split 1_000 : 3_000 -> 200 / 600.
		let mut stack = S3FifoLazyDemotionFastAdmissionReprieveHybridStack::new(0.25, 4_000, 4_000)
			.with_shared_overhead(400);

		assert_eq!(stack.one_access_capacity, 1_000);
		assert_eq!(stack.main_fast_capacity(), 3_000);

		// Both keys stay in the one-access queue (never `update`d); all that
		// matters here is that they are *tracked*, since the reservation
		// covers every tracked key regardless of queue or tier.
		stack.insert(1, 10);
		stack.insert(2, 10);

		assert_eq!(stack.len(), 2);
		assert_eq!(stack.reserved_overhead(), 800);
		assert_eq!(stack.reserved_shares(), (200, 600));

		// Charged once overall: the two shares sum to exactly the total.
		let (one_access_share, main_fast_share) = stack.reserved_shares();
		assert_eq!(one_access_share + main_fast_share, stack.reserved_overhead());

		// Charging the full 800 against each segment instead would have left
		// 200 and 2_200 -- 800 bytes of usable DRAM budget thrown away.
		assert_eq!(stack.effective_one_access_capacity(), 800);
		assert_eq!(stack.effective_main_fast_capacity(), 2_400);

		// ...so the whole fast tier's effective value budget is exactly
		// `fast_capacity - reserved_overhead()`.
		assert_eq!(
			stack.effective_one_access_capacity() + stack.effective_main_fast_capacity(),
			stack.fast_capacity - stack.reserved_overhead(),
		);
	}

	/// The reservation shrinks the effective capacity, and a stack carrying it
	/// demotes strictly earlier than an identical one without it. Modelled on
	/// `LruHybridStack::shared_overhead_reserves_dram_and_demotes_earlier`.
	#[test]
	fn shared_overhead_reserves_dram_and_demotes_earlier() {
		// The smallest effective main-fast budget whose *low* watermark still
		// holds one 10-byte object -- so a budget of exactly `holding` demotes
		// exactly one of two such objects and then stops. (Same device, and
		// the same `low >= high / 2` assumption, as
		// `an_accessed_fast_boundary_key_is_reprieved_at_demotion_time_not_demoted`
		// above.)
		let holding = effective_capacity_holding(10);

		// 100x that per segment, so with no reservation two 10-byte objects
		// sit nowhere near the high watermark.
		let segment = 100 * holding;

		// At 2 tracked keys the reservation is `2 * overhead` split evenly, so
		// each segment's share is exactly `overhead` -- leaving an effective
		// main-fast budget of `segment - overhead == holding`.
		let overhead = segment - holding;

		// Without the reservation, both objects stay fast.
		let mut plain = equal_segment_stack(segment, 0);

		promote(&mut plain, 1, 10);
		promote(&mut plain, 2, 10);

		assert_eq!(plain.effective_main_fast_capacity(), segment, "no overhead means no reservation");
		assert_eq!(drain(&mut plain), Vec::new());
		assert_eq!(plain.tier_of(1), Some(Tier::Fast));
		assert_eq!(plain.tier_of(2), Some(Tier::Fast));
		assert_eq!(plain.fast_bytes_used(), 20);

		// With it, the same two objects no longer fit: the reservation cuts
		// the effective budget from `segment` to `holding`, so promoting the
		// second pushes past its high watermark and the older one demotes.
		let mut stack = equal_segment_stack(segment, overhead);

		promote(&mut stack, 1, 10);
		assert_eq!(drain(&mut stack), Vec::new(), "one object still fits under a single key's reservation");

		promote(&mut stack, 2, 10);
		let migrations = drain(&mut stack);

		assert_eq!(stack.effective_main_fast_capacity(), holding);
		assert_eq!(migrations, vec![(1, Tier::Slow)]);
		assert_eq!(stack.tier_of(1), Some(Tier::Slow));
		assert_eq!(stack.tier_of(2), Some(Tier::Fast));
		assert_eq!(stack.fast_bytes_used(), 10);
		assert_eq!(stack.slow_bytes_used(), 10);

		// Demotion is the only response -- both keys are still tracked. The
		// DRAM reservation never evicts.
		assert_eq!(stack.len(), 2);
		assert!(!stack.needs_capacity_eviction());
	}

	/// A reservation big enough to swallow both segments' budgets saturates
	/// them to 0 and reprieves everything into the slow tier, without ever
	/// dropping an object.
	#[test]
	fn shared_overhead_exceeding_capacity_demotes_all_but_never_evicts() {
		// A single key's reservation (1_000) already exceeds the entire fast
		// budget (200), so both segments saturate to 0.
		let mut stack = equal_segment_stack(100, 1_000);

		stack.insert(1, 10);
		let migrations = drain(&mut stack);

		assert_eq!(stack.reserved_overhead(), 1_000);
		assert_eq!(stack.reserved_shares(), (500, 500));
		assert_eq!(stack.effective_one_access_capacity(), 0);
		assert_eq!(stack.effective_main_fast_capacity(), 0);

		assert_eq!(migrations, vec![(1, Tier::Slow)]);
		assert_eq!(stack.tier_of(1), Some(Tier::Slow));
		assert_eq!(stack.fast_bytes_used(), 0);
		assert_eq!(stack.slow_bytes_used(), 10);

		// Still tracked: `settle_one_access` is a pure internal migration and
		// `needs_capacity_eviction` stays at the trait default.
		assert_eq!(stack.len(), 1);
		assert!(!stack.needs_capacity_eviction());
	}

	/// Every byte counter and object count still agrees with the per-key tier
	/// tags after a pass that only fired because of the reservation -- the
	/// per-demotion bookkeeping ran once per demoted object, no more and no
	/// less.
	#[test]
	fn counters_stay_consistent_across_a_pass_triggered_by_the_reservation() {
		let size: ObjectSize = 10;
		let bytes = size as CacheSize;
		let count: CacheSize = 100;

		// Equal 1_000-byte segments with an 8-byte/key reservation: at `count`
		// tracked keys that is 800 bytes, 400 to each segment, so the
		// effective main-fast budget falls from 1_000 to 600 as the keys
		// arrive -- while the values themselves only ever total 1_000. Every
		// demotion here is the reservation's doing.
		let mut stack = equal_segment_stack(1_000, 8);

		for key in 1..=count {
			promote(&mut stack, key, size);
		}

		drain(&mut stack);

		assert_eq!(stack.reserved_overhead(), 8 * count);
		assert_eq!(stack.effective_one_access_capacity(), 1_000 - 4 * count);
		assert_eq!(stack.effective_main_fast_capacity(), 1_000 - 4 * count);

		// `promote`'s `update()` empties the one-access queue every time, and
		// its effective budget never fell below one object (600 at worst), so
		// nothing was reprieved out from under it.
		assert_eq!(stack.one_access_used, 0);

		let fast_objects = stack.fast_object_count() as CacheSize;
		let slow_objects = stack.slow_object_count() as CacheSize;

		assert!(slow_objects > 0, "the reservation must have triggered at least one demotion pass");
		assert!(fast_objects > 0, "a pass drains to the low watermark, never to empty");

		// Nothing was re-inserted, evicted or resized mid-pass, so every key
		// is still tracked, still `size` bytes, and still on exactly one side
		// of the fast/slow line.
		assert_eq!(fast_objects + slow_objects, count);
		assert_eq!(stack.len() as CacheSize, count);

		assert_eq!(stack.fast_bytes_used(), fast_objects * bytes);
		assert_eq!(stack.slow_bytes_used(), slow_objects * bytes);
		assert_eq!(stack.fast_bytes_used() + stack.slow_bytes_used(), count * bytes);

		// And the aggregate counts agree with the per-key tier tags.
		let tagged_fast = (1..=count).filter(|key| stack.tier_of(*key) == Some(Tier::Fast)).count();
		let tagged_slow = (1..=count).filter(|key| stack.tier_of(*key) == Some(Tier::Slow)).count();

		assert_eq!(tagged_fast as CacheSize, fast_objects);
		assert_eq!(tagged_slow as CacheSize, slow_objects);

		// The tier settled at or under the high watermark of its *effective*
		// budget, not of the raw 1_000.
		assert!(stack.fast_bytes_used() <= watermarks::high_bytes(stack.effective_main_fast_capacity()));
	}

	/// Composition order: the reservation comes off the capacity first, and
	/// the watermarks are taken against that reduced value -- so the drain
	/// target is `low_bytes(capacity - reserved)`, not `low_bytes(capacity)`.
	#[test]
	fn the_drain_target_is_the_low_watermark_of_the_reserved_budget() {
		const SEGMENT: CacheSize = 1_000;

		let size: ObjectSize = 10;
		let bytes = size as CacheSize;

		// 50 bytes/key over equal 1_000-byte segments: each segment's
		// effective budget is `1_000 - 25 * (tracked keys)`.
		let mut stack = equal_segment_stack(SEGMENT, 50);

		// Promote 10-byte objects until a pass actually fires. Looping to the
		// first pass (rather than promoting a fixed count) pins
		// `entries.len()` -- and so the effective budget -- to exactly what
		// the firing pass saw, at whatever high/low pair is configured.
		let mut key: HashedKey = 0;

		loop {
			key += 1;

			// The effective one-access budget is `1_000 - 25 * key`, so it
			// still holds a 10-byte object at key 39. A pass fires by key 29
			// even at `FAST_TIER_HIGH_WATERMARK=1.0`, the latest possible
			// setting; stop well short of the point where a *one-access*
			// reprieve could masquerade as a fast-tier demotion.
			assert!(key < 35, "a demotion pass should have fired long before the one-access budget ran out");

			promote(&mut stack, key, size);

			if drain(&mut stack).iter().any(|(_, tier)| *tier == Tier::Slow) {
				break;
			}
		}

		let effective = stack.effective_main_fast_capacity();

		// The reservation came off the raw capacity first...
		assert_eq!(effective, SEGMENT - 25 * key);
		assert_eq!(effective, SEGMENT - stack.reserved_shares().1);
		assert!(effective < SEGMENT);

		// ...and the watermark was then taken against that reduced value. All
		// objects are `size` bytes, so the pass halts at the largest whole
		// multiple of `size` at or below `low_bytes(effective)`.
		let target = watermarks::low_bytes(effective);

		assert!(target >= bytes, "sanity: the drain target must still hold at least one object");
		assert_eq!(stack.fast_bytes_used(), target - target % bytes);
		assert!(stack.fast_bytes_used() <= target);
		assert!(
			stack.fast_bytes_used() + bytes > target,
			"the pass must stop at the first whole object at or below the target, not overshoot",
		);

		// Had the reservation been ignored, the pass would have drained to
		// `low_bytes(SEGMENT)` instead -- strictly looser, leaving the tier
		// holding far more DRAM.
		assert!(target < watermarks::low_bytes(SEGMENT));
		assert!(stack.fast_bytes_used() < watermarks::low_bytes(SEGMENT));
	}
}
