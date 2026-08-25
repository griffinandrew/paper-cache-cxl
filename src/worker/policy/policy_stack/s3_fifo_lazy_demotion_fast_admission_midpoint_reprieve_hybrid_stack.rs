/*
 * Copyright (c) Kia Shakiba
 *
 * This source code is licensed under the GNU AGPLv3 license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! `S3FifoLazyDemotionFastAdmissionMidpointReprieveHybridStack` —
//! `S3FifoGhostLazyDemotionFastAdmissionMidpointHybridStack` with two
//! behavioral changes, for
//! `PaperPolicy::S3FifoLazyDemotionFastAdmissionMidpointReprieveHybrid`:
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
//! ## Locating "the middle"
//!
//! Unchanged in substance from the predecessor: `slow_midpoint` is a cursor
//! tracking (approximately) the middle of `main_slow`, maintained in O(1)
//! amortized time via a small drift counter rather than a rescan. See the
//! design notes below the `PolicyStack` impl for the arithmetic.
//!
//! ## Fast-tier watermarks
//!
//! `settle_fast_tier` no longer demotes the instant the main queue's fast
//! segment crosses its ceiling. It triggers once `fast_used` passes
//! `watermarks::high_bytes` of the effective budget and then drains down to
//! `watermarks::low_bytes` of that same budget, so demotions arrive as
//! occasional batches instead of one object per promotion. The budget the
//! watermarks are applied to is unchanged --
//! [`S3FifoLazyDemotionFastAdmissionMidpointReprieveHybridStack::effective_main_fast_capacity`],
//! i.e. `fast_capacity` minus the one-access queue's reservation -- the
//! watermarks sit on top of it, never in place of it. See
//! [`super::watermarks`] for the ratios, their env overrides, and how to
//! restore the original drain-to-the-ceiling behaviour exactly.
//!
//! `settle_one_access` is deliberately untouched *by the watermarks*: the
//! one-access queue's `one_access_capacity` is a queue-length rule of the
//! S3-FIFO design, not a tier-pressure threshold, and a reprieve out of it is
//! a Fast->Slow migration whose batching is governed by the fast/slow boundary
//! the watermarks already cover. (It does take its own share of the DRAM
//! metadata reservation described next -- that is a budget correction, not a
//! batching policy.)
//!
//! ## Shared-metadata DRAM reservation
//!
//! The fast tier is DRAM (NUMA node 0) and the slow tier is PMEM/CXL, but the
//! shared object hashtable and this stack's own bookkeeping (`entries`, plus
//! whichever of the three queue lists a key currently sits in) live in DRAM
//! for *every tracked key*, whichever tier that key's value bytes are in --
//! and none of it is counted in `fast_used`/`one_access_used`. Left alone, the
//! fast tier's real DRAM footprint therefore exceeds `fast_capacity` by
//! exactly that metadata.
//!
//! `shared_overhead` -- an approximate per-tracked-key byte cost, supplied by
//! `crate::object::overhead::get_hybrid_dram_shared_overhead` via
//! [`S3FifoLazyDemotionFastAdmissionMidpointReprieveHybridStack::with_shared_overhead`]
//! -- is reserved out of the fast-tier budget so demotion bounds *total* DRAM
//! rather than just fast-tier values. Because this is a `fast_admission`
//! variant, the fast tier is two segments with their own budgets (the
//! one-access queue's `one_access_capacity`, and whatever `fast_capacity`
//! leaves over for the main queue's fast portion), so the reservation is split
//! between them in proportion to those capacities
//! ([`S3FifoLazyDemotionFastAdmissionMidpointReprieveHybridStack::reserved_shares`],
//! the same scheme `LruSizedHybridStack` uses for its small/large fast
//! segments) -- never charged in full to each, which would double-reserve, and
//! never charged wholly to the main segment, which would leave the one-access
//! queue unbounded once the main segment saturated to 0. The two shares sum to
//! exactly the reservation, so
//!
//! ```text
//! one_access_used + fast_used + reserved_overhead() <= fast_capacity
//! ```
//!
//! holds at every settled point.
//!
//! Two consequences for *when* settling runs. The reservation scales with the
//! tracked key count, so admission is itself an event that tightens both
//! budgets: `insert` now settles the main fast segment as well as the
//! one-access queue. And `fast_capacity` is one of the two inputs to the
//! proportional split, so `resize_fast_tier` settles the one-access queue as
//! well as the main fast segment. Both extra calls are no-ops at the default
//! `shared_overhead` of `0` (nothing that raises `fast_used`/`one_access_used`
//! can leave either segment above its high watermark without settling), so
//! every test in this file that does not call `with_shared_overhead` sees the
//! pure value-budget behaviour unchanged.
//!
//! The reservation is charged against **all** tracked keys, not just fast-tier
//! ones (`entries.len()`), matching `LruHybridStack`'s `stack.len()` and
//! `LruSizedHybridStack`'s `entries.len()`: a slow-tier key still owns a
//! hashtable entry, an `entries` entry and a `main_slow` node, all in DRAM.
//! Gating on `eviction_stacks_pmem`/`global_hashtable_pmem` -- i.e. dropping
//! the terms whose structures have been relocated to PMEM -- happens once, in
//! `get_hybrid_dram_shared_overhead`, so this stack simply consumes whatever
//! number it is handed.

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

pub struct S3FifoLazyDemotionFastAdmissionMidpointReprieveHybridStack {
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

	/// Approximate per-tracked-key DRAM cost of the *shared* structures (the
	/// object hashtable + this stack's eviction-stack bookkeeping) that hold
	/// an entry for every tracked object regardless of which tier its value
	/// bytes sit in. Reserved out of the fast-tier budget -- split between the
	/// two fast segments by [`Self::reserved_shares`] -- so that budget bounds
	/// total DRAM (values *and* shared metadata), not just fast-tier values.
	/// `0` unless set via [`Self::with_shared_overhead`], so unit tests
	/// exercising the pure value-budget behaviour are unaffected.
	shared_overhead: CacheSize,

	/// Cursor tracking (approximately) the middle of `main_slow` -- see the
	/// design notes below the `PolicyStack` impl.
	slow_midpoint: Option<HashedKey>,
	/// Accumulates 0.5-position drift per qualifying event; reset (and the
	/// cursor nudged) every time it reaches 2.
	midpoint_drift: u8,

	migrations: Vec<(HashedKey, Tier)>,
}

impl S3FifoLazyDemotionFastAdmissionMidpointReprieveHybridStack {
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

		S3FifoLazyDemotionFastAdmissionMidpointReprieveHybridStack {
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

			slow_midpoint: None,
			midpoint_drift: 0,

			migrations: Vec::new(),
		}
	}

	/// Sets the approximate per-tracked-key shared-structure DRAM overhead
	/// (object hashtable + eviction stacks) reserved out of the fast-tier
	/// budget. See `crate::object::overhead::get_hybrid_dram_shared_overhead`,
	/// which is also where the `eviction_stacks_pmem`/`global_hashtable_pmem`
	/// gating lives -- this stack just spends the number it is given.
	/// Builder-style so `init_policy_stack` can wire it in without disturbing
	/// `new`'s signature (unit tests keep the default `0`).
	pub fn with_shared_overhead(mut self, overhead: CacheSize) -> Self {
		self.shared_overhead = overhead;
		self
	}

	/// Total DRAM currently reserved for shared per-object metadata. Counts
	/// *every* tracked key rather than just the fast-tier ones: the hashtable
	/// entry, the `entries` entry and the queue-list node all exist in DRAM
	/// whether the key's value bytes are in DRAM or PMEM (matching
	/// `LruHybridStack::reserved_overhead`'s `stack.len()` and
	/// `LruSizedHybridStack::reserved_shares`'s `entries.len()`).
	///
	/// Note this is constant across a demotion pass -- demoting a key does not
	/// untrack it -- so both settle loops can hoist it out of their condition.
	fn reserved_overhead(&self) -> CacheSize {
		self.entries.len() as CacheSize * self.shared_overhead
	}

	/// Splits [`Self::reserved_overhead`] proportionally between this stack's
	/// two independently-capacitied fast segments, returning
	/// `(one_access_share, main_share)`:
	///
	/// * the one-access queue, budgeted by `one_access_capacity`, and
	/// * the main queue's fast portion, budgeted by whatever `fast_capacity`
	///   leaves over once the one-access queue's reservation is taken out.
	///
	/// Both are fast-tier -- this is a `fast_admission` variant, and
	/// `fast_bytes_used()` sums them -- so the reservation must come out of
	/// the pair exactly once. Charging the full amount to each would
	/// double-reserve; charging it all to the main segment would leave the
	/// one-access queue unbounded once the main segment saturated to 0. Same
	/// scheme as `LruSizedHybridStack::reserved_shares`.
	///
	/// The remainder goes to the main share, so the two always sum to exactly
	/// `reserved_overhead()` and the
	/// `one_access + main + reserved <= fast_capacity` bound stays tight.
	/// `(0, 0)` when neither segment has any capacity to proportion against.
	fn reserved_shares(&self) -> (CacheSize, CacheSize) {
		let reserved = self.reserved_overhead();
		let main_capacity = self.fast_capacity.saturating_sub(self.one_access_capacity);
		let total_capacity = self.one_access_capacity + main_capacity;

		if total_capacity == 0 {
			return (0, 0);
		}

		let one_access_share =
			((reserved as u128 * self.one_access_capacity as u128) / total_capacity as u128) as CacheSize;
		let main_share = reserved.saturating_sub(one_access_share);

		(one_access_share, main_share)
	}

	/// The one-access queue's byte budget after its share of the shared
	/// per-object metadata reservation. Saturates to 0 rather than wrapping
	/// when the reservation alone meets or exceeds the segment's capacity.
	fn effective_one_access_capacity(&self) -> CacheSize {
		self.one_access_capacity.saturating_sub(self.reserved_shares().0)
	}

	/// The main queue's fast-segment byte budget: `fast_capacity` minus the
	/// one-access queue's `one_access_capacity` reservation, minus this
	/// segment's share of the shared per-object metadata reservation.
	fn effective_main_fast_capacity(&self) -> CacheSize {
		self.fast_capacity
			.saturating_sub(self.one_access_capacity)
			.saturating_sub(self.reserved_shares().1)
	}

	pub fn tier_of(&self, key: HashedKey) -> Option<Tier> {
		let entry = self.entries.get(&key)?;

		match entry.queue {
			Queue::OneAccess => Some(Tier::Fast),
			Queue::Main => entry.tier,
		}
	}

	/// Returns `true` if `key` is the current midpoint cursor target.
	/// Exposed for tests.
	pub fn is_midpoint(&self, key: HashedKey) -> bool {
		self.slow_midpoint == Some(key)
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

	/// Moves the midpoint cursor one step toward the front of `main_slow`.
	/// No-op if the cursor is empty or already at the front. Unlike the
	/// predecessor's equivalent, no tier check is needed: `main_slow` is
	/// homogeneous, so every neighbor within it is Slow by construction.
	fn nudge_midpoint_toward_front(&mut self) {
		let Some(current) = self.slow_midpoint else { return };

		if let Some(&candidate) = self.main_slow.before(&current) {
			self.slow_midpoint = Some(candidate);
		}
	}

	/// Call after any event that changes `main_slow`'s length by exactly one
	/// in either direction (a demotion, a reprieve, a slow-tier eviction, or
	/// a promotion/removal out of the slow segment) once the cursor is
	/// already initialized.
	fn bump_midpoint_drift(&mut self) {
		self.midpoint_drift += 1;

		if self.midpoint_drift >= 2 {
			self.midpoint_drift = 0;
			self.nudge_midpoint_toward_front();
		}
	}

	/// If `key` is currently the midpoint cursor's target, redirects it to
	/// the `before()` neighbor before `key` is unlinked from `main_slow`.
	/// Must be called while `key` is still linked -- `before()` needs that
	/// to resolve the neighbor.
	fn redirect_midpoint_before_removing(&mut self, key: HashedKey) {
		if self.slow_midpoint != Some(key) {
			return;
		}

		self.slow_midpoint = self.main_slow.before(&key).copied();
	}

	/// Checks the midpoint cursor's reference bit and, if set, gives it an
	/// early second chance. No-op if the slow segment is currently empty.
	/// Called once per `evict_one` pass over the main queue.
	fn check_slow_midpoint(&mut self) {
		let Some(candidate) = self.slow_midpoint else { return };
		let accessed = self.entries.get(&candidate).map(|entry| entry.accessed).unwrap_or(false);

		if accessed {
			self.give_second_chance(candidate);
		}
	}

	/// The eviction-time second chance -- also reused directly by
	/// `check_slow_midpoint` for the mid-segment check, since both are
	/// "this key's reference bit is set, so spare it and move it to the
	/// front of the fast list" with identical mechanics.
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
				self.redirect_midpoint_before_removing(key);

				self.main_slow.remove(&key);
				self.main_fast.push_front(key);

				if let Some(entry) = self.entries.get_mut(&key) {
					entry.tier = Some(Tier::Fast);
					entry.accessed = false;
				}

				self.slow_used = self.slow_used.saturating_sub(size);
				self.fast_used += size;

				self.bump_midpoint_drift();
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

	/// Demotes oldest-first from `main_fast` into `main_slow` -- *triggered*
	/// once `fast_used` crosses the shared HIGH watermark of the effective
	/// budget, and once triggered *drained* all the way down to the shared LOW
	/// watermark rather than merely back under the ceiling -- giving any key
	/// whose reference bit is set a reprieve (moved to the front of
	/// `main_fast`, bit cleared) instead. Terminates even when every fast
	/// key's bit is set, since each reprieve clears one bit.
	///
	/// The budget the watermarks are applied *to* is
	/// [`Self::effective_main_fast_capacity`]: `fast_capacity`, minus the
	/// one-access queue's `one_access_capacity` reservation -- this is a
	/// `fast_admission` variant, so its one-access queue is fast-tier and
	/// competes for the same DRAM budget -- minus this segment's share of the
	/// shared per-object metadata reservation ([`Self::reserved_shares`]),
	/// which is what makes the fast-tier budget bound total DRAM (values *and*
	/// the DRAM-resident hashtable/eviction-stack entries of every tracked
	/// key) instead of just fast-tier values.
	///
	/// The composition order matters and is fixed here: the overhead is
	/// subtracted *first*, and the watermarks are then applied to what remains
	/// -- so a triggered pass drains to `low_bytes(capacity - reserved)`, never
	/// to `low_bytes(capacity)`. The watermarks sit on top of the effective
	/// value and never replace it, so the "one-access queue + main fast
	/// segment + shared metadata <= `fast_capacity`" bound this stack relies on
	/// stays exactly as valid as before -- tighter now, never looser.
	///
	/// Draining below the ceiling instead of exactly to it is what turns the
	/// steady state from "every promotion demotes exactly one object" into
	/// occasional multi-object batches -- see [`watermarks`] for the full
	/// rationale, and for how to restore the old drain-to-the-ceiling
	/// behaviour byte-for-byte (`FAST_TIER_HIGH_WATERMARK=1.0`,
	/// `FAST_TIER_LOW_WATERMARK=1.0`).
	///
	/// Nothing else moves. The demotion-time reference-bit reprieve, the
	/// per-demotion bookkeeping (tier tag, the `main_fast` -> `main_slow`
	/// splice, `fast_used`/`slow_used`, the migration push) and the
	/// midpoint-cursor maintenance in the loop below are all unchanged, and
	/// each still runs exactly once per object the pass touches -- only the
	/// number of objects one pass touches changed.
	fn settle_fast_tier(&mut self) {
		let effective_capacity = self.effective_main_fast_capacity();

		// Trigger only once usage is past the high watermark...
		if self.fast_used <= watermarks::high_bytes(effective_capacity) {
			return;
		}

		// ...but once triggered, drain all the way down to the low one.
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

			// A real demotion always grows the slow segment by exactly one.
			if self.slow_midpoint.is_none() {
				self.slow_midpoint = Some(candidate);
			} else {
				self.bump_midpoint_drift();
			}
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
	/// The budget is [`Self::effective_one_access_capacity`]: the configured
	/// `one_access_capacity` minus this segment's share of the shared
	/// per-object metadata reservation. No watermarks here -- see the module
	/// doc for why this boundary is settled exactly rather than in batches --
	/// but the reservation applies just as it does to the main fast segment,
	/// since both segments are DRAM. Hoisted out of the loop condition because
	/// `reserved_overhead()` is constant across the loop: a reprieve moves a
	/// key between queues, it never untracks one.
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

			// Grows the slow segment by exactly one, same as a real demotion.
			if self.slow_midpoint.is_none() {
				self.slow_midpoint = Some(key);
			} else {
				self.bump_midpoint_drift();
			}
		}
	}
}

impl PolicyStack for S3FifoLazyDemotionFastAdmissionMidpointReprieveHybridStack {
	fn is_policy(&self, policy: &PaperPolicy) -> bool {
		matches!(policy, PaperPolicy::S3FifoLazyDemotionFastAdmissionMidpointReprieveHybrid(ratio) if *ratio == self.one_access_ratio)
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

		// Admitting a key grows `reserved_overhead()` by one key's worth,
		// which shrinks the main fast segment's effective budget too -- so
		// admission has to settle that segment as well, not just the queue the
		// key landed in. A no-op at the default `shared_overhead` of 0: every
		// path that raises `fast_used` already ends in `settle_fast_tier`, and
		// the trigger is a strict `>`, so usage sitting exactly on the high
		// watermark stays put.
		self.settle_fast_tier();
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
					// Redirect the cursor BEFORE unlinking -- `before()`
					// needs the key still linked.
					self.redirect_midpoint_before_removing(key);

					self.main_slow.remove(&key);
					self.slow_used = self.slow_used.saturating_sub(size);

					self.bump_midpoint_drift();
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
		self.slow_midpoint = None;
		self.midpoint_drift = 0;
		self.migrations.clear();
	}

	fn evict_one(&mut self) -> Option<HashedKey> {
		// The one-access queue never reaches here -- its own capacity
		// pressure is relieved synchronously by `settle_one_access()` (see
		// the module doc), the same way the main queue's fast/slow boundary
		// is settled by `settle_fast_tier()` rather than through eviction.
		// This is purely about the main queue: the midpoint check, then the
		// ordinary tail loop.
		self.check_slow_midpoint();

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
				// Redirect the cursor BEFORE unlinking.
				self.redirect_midpoint_before_removing(key);
				self.main_slow.pop_back();
			} else {
				self.main_fast.pop_back();
			}

			let removed = self.entries.remove(&key);
			let size = removed.map(|entry| entry.migrating()).unwrap_or(0);

			if from_slow {
				self.slow_used = self.slow_used.saturating_sub(size);
				self.bump_midpoint_drift();
			} else {
				self.fast_used = self.fast_used.saturating_sub(size);
			}

			return Some(key);
		}
	}

	fn resize_fast_tier(&mut self, size: CacheSize) {
		self.fast_capacity = size;

		// `fast_capacity` is one of the two inputs to the proportional split of
		// the metadata reservation, so changing it re-proportions the
		// one-access queue's share as well -- settle that segment first (a
		// reprieve out of it only adds slow-tier bytes, so it can never make
		// the fast/slow settle below harder). A no-op at the default
		// `shared_overhead` of 0, where the one-access budget does not depend
		// on `fast_capacity` at all.
		self.settle_one_access();
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

// ── Design notes: the "every 2 events, one step" drift derivation ─────────
//
// Model `main_slow` as a list of length N, positions 0 (the front, i.e. the
// fast/slow boundary) through N-1 (the tail). The true middle is at index
// N/2 (integer division). The cursor tracks a *specific object*, not an
// index -- its absolute index drifts as the list mutates around it.
//
// Front-insertion (a demotion or a reprieve, both `main_slow.push_front`):
// every existing object's index increases by 1, including the tracked one.
// The target index (N/2) increases by only 0.5 on average as N grows by 1.
// Net: the tracked object drifts +0.5 positions past the true middle per
// event.
//
// Tail-removal (an eviction) or arbitrary-position removal (a promotion):
// the tracked object's own index is unaffected (removal at/after it, or the
// tracked object being the one removed and immediately redirected to its
// front-ward neighbor, whose index is one less than the removed object's
// would have been), but N decreases by 1, so the target index (N/2)
// decreases by 0.5. Net: the tracked object again drifts +0.5 positions past
// the true middle per event -- same sign as growth.
//
// Since both kinds of qualifying event drift the tracked object the same
// direction by the same magnitude, a single counter suffices: accumulate 1
// per event, and every time it reaches 2 (i.e. every 2 events, matching
// 2 * 0.5 = 1 full position of accumulated drift), move the cursor one step
// toward the front via `before()` to cancel it out. Verified by hand against
// the small worked examples in this stack's unit tests below.

#[cfg(test)]
mod tests {
	use super::*;

	fn drain(stack: &mut S3FifoLazyDemotionFastAdmissionMidpointReprieveHybridStack) -> Vec<(HashedKey, Tier)> {
		stack.drain_tier_migrations()
	}

	/// `insert` + `update` -- the admit-into-the-one-access-queue-then-promote
	/// pairing every main-queue fast-tier test in this module already uses,
	/// since this stack never admits a fresh key straight into `main_fast`.
	fn promote(stack: &mut S3FifoLazyDemotionFastAdmissionMidpointReprieveHybridStack, key: HashedKey, size: ObjectSize) {
		stack.insert(key, size);
		stack.update(key);
	}

	/// Smallest *effective* main-fast budget (i.e. the value
	/// `effective_main_fast_capacity()` returns -- `fast_capacity` minus the
	/// one-access reservation) that holds a fast segment of exactly `bytes`
	/// across a settled pass: `bytes` sits at or below the LOW watermark, so a
	/// triggered pass stops there rather than draining past it, while one
	/// further `next`-byte object still pushes usage past the HIGH watermark
	/// and therefore does trigger one.
	///
	/// The hand-traced fixtures below were originally written against the
	/// drain-to-the-ceiling rule, where "capacity" and "the point a pass
	/// settles at" were the same number. They derive their capacity from this
	/// instead of hard-coding it, so their traces hold unchanged at any
	/// configured ratio pair -- including `1.0`/`1.0`, which reproduces the
	/// original literals (10 and 20) exactly. They cannot simply pin the
	/// ratios: the watermarks are process-global `OnceLock`s read once per
	/// process, so a test that set the env vars would race every other test in
	/// the binary.
	fn effective_capacity_holding(bytes: CacheSize, next: CacheSize) -> CacheSize {
		let mut capacity = (bytes as f64 / watermarks::low()).ceil() as CacheSize;

		// Guard against `low_bytes`'s `as u64` truncation landing a byte short
		// for some ratio/rounding combinations.
		while watermarks::low_bytes(capacity) < bytes {
			capacity += 1;
		}

		assert!(
			watermarks::high_bytes(capacity) < bytes + next,
			"watermark config leaves no room for this fixture",
		);

		capacity
	}

	#[test]
	fn admission_always_lands_in_one_access_queue_fast() {
		let mut stack = S3FifoLazyDemotionFastAdmissionMidpointReprieveHybridStack::new(1.0, 1_000, 1_000);

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
		let mut stack = S3FifoLazyDemotionFastAdmissionMidpointReprieveHybridStack::new(0.01, 1_000, 1_000);

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
		let mut stack = S3FifoLazyDemotionFastAdmissionMidpointReprieveHybridStack::new(0.01, 1_000, 1_000);

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
		let mut stack = S3FifoLazyDemotionFastAdmissionMidpointReprieveHybridStack::new(1.0, 1_000, 10_000);

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
		let mut stack = S3FifoLazyDemotionFastAdmissionMidpointReprieveHybridStack::new(0.0, 1_000, 1_000);

		stack.insert(1, 10);

		assert_eq!(stack.fast_bytes_used(), 0, "a reprieved key must never be counted as fast, even transiently");
		assert_eq!(stack.slow_bytes_used(), 10);
	}

	#[test]
	fn an_accessed_fast_boundary_key_is_reprieved_at_demotion_time_not_demoted() {
		// one_access_capacity = 1.0 * 1_000, so the effective main-fast budget
		// is whatever is added on top of it -- sized here to hold exactly one
		// of these two 10-byte objects once a pass settles, so the second
		// promotion triggers exactly one demotion. (Was a hard-coded 1_010,
		// i.e. an effective 10: correct back when a pass drained to the
		// ceiling, but under the watermarks a 10-byte effective budget
		// triggers on the very first promotion and drains below one object.)
		let mut stack = S3FifoLazyDemotionFastAdmissionMidpointReprieveHybridStack::new(1.0, 1_000, 1_000 + effective_capacity_holding(10, 10));

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

	// ── the mid-slow-segment checkpoint ────────────────────────────────────

	/// Five 10-byte keys admitted and promoted in order 1..=5 against an
	/// effective main-fast budget that holds exactly 2 of them once a pass
	/// settles (see `effective_capacity_holding`; was a hard-coded 1_020, i.e.
	/// an effective 20, back when a pass drained to the ceiling). Traced by
	/// hand: keys 1, 2, 3 demote oldest-first as keys 4 and 5 arrive, leaving
	/// main_slow = [3, 2, 1] and main_fast = [5, 4], and after exactly 3
	/// demotions the drift-correction cursor settles on the middle element,
	/// key 2. The per-demotion bookkeeping is per *object*, not per pass, so
	/// that end state is the same whether the three demotions arrive one per
	/// promotion or batched into a single drain.
	fn build_five_key_stack() -> S3FifoLazyDemotionFastAdmissionMidpointReprieveHybridStack {
		let mut stack = S3FifoLazyDemotionFastAdmissionMidpointReprieveHybridStack::new(1.0, 1_000, 1_000 + effective_capacity_holding(20, 10));

		for key in 1..=5u64 {
			stack.insert(key, 10);
			stack.update(key);
		}

		drain(&mut stack);
		stack
	}

	#[test]
	fn slow_midpoint_tracks_the_middle_of_the_slow_segment_as_it_grows() {
		let stack = build_five_key_stack();

		assert_eq!(stack.tier_of(1), Some(Tier::Slow));
		assert_eq!(stack.tier_of(2), Some(Tier::Slow));
		assert_eq!(stack.tier_of(3), Some(Tier::Slow));
		assert_eq!(stack.tier_of(4), Some(Tier::Fast));
		assert_eq!(stack.tier_of(5), Some(Tier::Fast));

		assert!(stack.is_midpoint(2), "expected key 2 (the middle of slow segment [3, 2, 1]) to be tracked");
	}

	#[test]
	fn a_reaccessed_midpoint_key_is_promoted_early_instead_of_waiting_for_the_tail() {
		let mut stack = build_five_key_stack();
		assert!(stack.is_midpoint(2));

		stack.update(2);
		assert_eq!(stack.tier_of(2), Some(Tier::Slow), "a mere access must not itself migrate or reorder");

		let evicted = stack.evict_one();

		assert_eq!(stack.tier_of(2), Some(Tier::Fast), "the reaccessed midpoint key should have been promoted early");
		assert_eq!(stack.tier_of(4), Some(Tier::Slow), "cascading demotion after the midpoint promotion");
		assert_eq!(evicted, Some(1), "the tail should still be evicted normally in the same call");
		assert!(!stack.contains(1));
	}

	#[test]
	fn an_unaccessed_midpoint_key_is_left_alone() {
		let mut stack = build_five_key_stack();
		assert!(stack.is_midpoint(2));

		let evicted = stack.evict_one();

		assert_eq!(stack.tier_of(2), Some(Tier::Slow), "an unaccessed midpoint key must not be promoted");
		assert_eq!(evicted, Some(1));
	}

	#[test]
	fn removing_the_midpoint_key_directly_redirects_the_cursor() {
		let mut stack = build_five_key_stack();
		assert!(stack.is_midpoint(2));

		stack.remove(2);

		assert!(!stack.is_midpoint(2));
		assert!(stack.is_midpoint(3), "cursor should redirect to the before()-neighbor in main_slow");
	}

	#[test]
	fn evict_one_gives_an_accessed_slow_key_a_second_chance() {
		// Same sizing rationale as
		// `an_accessed_fast_boundary_key_is_reprieved_at_demotion_time_not_demoted`
		// above: an effective main-fast budget holding exactly one 10-byte
		// object at the low watermark, so promoting the second key demotes the
		// first and nothing more.
		let mut stack = S3FifoLazyDemotionFastAdmissionMidpointReprieveHybridStack::new(1.0, 1_000, 1_000 + effective_capacity_holding(10, 10));

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
		let mut stack = S3FifoLazyDemotionFastAdmissionMidpointReprieveHybridStack::new(1.0, 1_000, 10_000);

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
		let mut stack = S3FifoLazyDemotionFastAdmissionMidpointReprieveHybridStack::new(1.0, 1_000, 1_000);

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
		assert!(!stack.is_midpoint(3));
	}

	#[test]
	fn fast_and_slow_gauges_include_one_access_queue_on_the_fast_side() {
		let mut stack = S3FifoLazyDemotionFastAdmissionMidpointReprieveHybridStack::new(1.0, 1_000, 1_000);

		stack.insert(1, 10);
		stack.insert(2, 10);
		drain(&mut stack);

		assert_eq!(stack.fast_bytes_used(), 20);
		assert_eq!(stack.slow_bytes_used(), 0);
		assert_eq!(stack.fast_object_count(), 2);
		assert_eq!(stack.slow_object_count(), 0);
	}

	// -- shared high/low watermarks (`super::watermarks`) ---------------------
	//
	// The ratios are process-global (`OnceLock`, seeded once from
	// `FAST_TIER_HIGH_WATERMARK` / `FAST_TIER_LOW_WATERMARK`), so these tests
	// cannot set the env vars for themselves without racing every other test
	// in the binary. They compute their expectations from
	// `watermarks::high_bytes()` / `watermarks::low_bytes()` of the *effective*
	// budget instead, and therefore hold at any configured ratio pair --
	// including the `1.0` / `1.0` setting that restores the original
	// drain-to-the-ceiling behaviour.

	/// One-access ratio for the watermark tests. Large enough that
	/// `insert()`'s `settle_one_access()` never reprieves a key out from under
	/// the `update()` that is about to promote it, and a round number so
	/// `one_access_capacity` is exact.
	const WM_ONE_ACCESS_RATIO: f64 = 0.1;
	const WM_MAX_SIZE: CacheSize = 100_000;
	const WM_ONE_ACCESS_CAPACITY: CacheSize = 10_000;

	/// The effective main-fast budget every watermark test works against --
	/// i.e. exactly what `effective_main_fast_capacity()` returns for
	/// `watermark_stack()`, since `fast_capacity` is built as
	/// `WM_ONE_ACCESS_CAPACITY + WM_EFFECTIVE`. Pins that the watermarks are
	/// applied to the effective budget, not to raw `fast_capacity`.
	const WM_EFFECTIVE: CacheSize = 1_000;

	fn watermark_stack() -> S3FifoLazyDemotionFastAdmissionMidpointReprieveHybridStack {
		let stack = S3FifoLazyDemotionFastAdmissionMidpointReprieveHybridStack::new(
			WM_ONE_ACCESS_RATIO,
			WM_MAX_SIZE,
			WM_ONE_ACCESS_CAPACITY + WM_EFFECTIVE,
		);

		assert_eq!(stack.one_access_capacity, WM_ONE_ACCESS_CAPACITY);
		assert_eq!(stack.effective_main_fast_capacity(), WM_EFFECTIVE);

		stack
	}

	/// (a) The trigger is a strict `>`, so usage sitting right *on* the high
	/// watermark -- the largest usage that is not over it -- must leave the
	/// fast tier completely alone. Under the old rule the whole band between
	/// here and the ceiling would have been demoting one object per promotion.
	#[test]
	fn fast_usage_at_the_high_watermark_triggers_no_demotion() {
		let high = watermarks::high_bytes(WM_EFFECTIVE);

		assert!(high > 1, "watermark config leaves no room for this test");

		let mut stack = watermark_stack();

		// Two objects summing to exactly the high watermark.
		promote(&mut stack, 1, (high - 1) as ObjectSize);
		promote(&mut stack, 2, 1);

		assert_eq!(stack.fast_used, high);
		assert!(drain(&mut stack).is_empty(), "demoted at the high watermark of {high}");

		assert_eq!(stack.tier_of(1), Some(Tier::Fast));
		assert_eq!(stack.tier_of(2), Some(Tier::Fast));
		assert_eq!(stack.slow_used, 0);
		assert_eq!(stack.slow_object_count(), 0);
		assert_eq!(stack.slow_midpoint, None);
	}

	/// (b) One byte past the high watermark -- the smallest possible overshoot
	/// -- must fire a pass, and it must take `main_fast`'s oldest key rather
	/// than the key that just arrived.
	#[test]
	fn fast_usage_above_the_high_watermark_triggers_a_pass() {
		let high = watermarks::high_bytes(WM_EFFECTIVE);
		let low = watermarks::low_bytes(WM_EFFECTIVE);

		assert!(low >= 1, "watermark config leaves no room for this test");

		let mut stack = watermark_stack();

		promote(&mut stack, 1, high as ObjectSize);
		assert_eq!(stack.fast_used, high);
		assert!(drain(&mut stack).is_empty(), "exactly on the high watermark must not trigger");

		// One more byte tips it over.
		promote(&mut stack, 2, 1);
		let migrations = drain(&mut stack);

		// Key 1 is `main_fast`'s tail and its reference bit is clear, so it is
		// the one that goes.
		assert_eq!(migrations, vec![(1, Tier::Slow)]);
		assert_eq!(stack.tier_of(1), Some(Tier::Slow));
		assert_eq!(stack.tier_of(2), Some(Tier::Fast));
		assert!(stack.fast_used <= low);

		// The first demotion into an empty slow segment seeds the midpoint
		// cursor, exactly as it did before.
		assert!(stack.is_midpoint(1));
	}

	/// (c) The point of the change: a triggered pass drains to the LOW
	/// watermark, not merely back under the ceiling it used to settle at.
	/// With the defaults this drains 960 -> 750 across 21 demotions; the
	/// pre-watermark loop would have stopped after a single one, at 950.
	#[test]
	fn a_triggered_pass_drains_to_the_low_watermark_not_the_ceiling() {
		let high = watermarks::high_bytes(WM_EFFECTIVE);
		let low = watermarks::low_bytes(WM_EFFECTIVE);
		let size: CacheSize = 10;

		// Largest whole number of 10-byte objects still at or below the high
		// watermark: no pass has fired yet.
		let filled = high / size;

		assert!(filled >= 1, "watermark config leaves no room for this test");

		let mut stack = watermark_stack();

		for key in 1..=filled {
			promote(&mut stack, key, size as ObjectSize);
		}

		assert_eq!(stack.fast_used, filled * size);
		assert!(drain(&mut stack).is_empty());

		// One more object tips it past the high watermark.
		promote(&mut stack, filled + 1, size as ObjectSize);
		let migrations = drain(&mut stack);

		assert!(!migrations.is_empty(), "crossing the high watermark must trigger a pass");

		// Drained past the ceiling, all the way down to the low watermark...
		assert!(stack.fast_used <= low);
		assert_eq!(stack.fast_used, low / size * size);

		// ...and stopped as soon as it got under it, rather than emptying the
		// segment: one fewer demotion would have left it above the target.
		assert!(stack.fast_used + size > low);

		// One `Tier::Slow` entry per demoted object, off `main_fast`'s tail in
		// promotion order -- a real batch, which is the whole behaviour being
		// bought here.
		let demoted = (filled + 1) - stack.main_fast.len() as CacheSize;

		assert_eq!(migrations.len() as CacheSize, demoted);
		assert_eq!(migrations, (1..=demoted).map(|key| (key, Tier::Slow)).collect::<Vec<_>>());

		if high >= low + size {
			assert!(stack.fast_used < WM_EFFECTIVE, "the pass must settle below the ceiling, not on it");
			assert!(migrations.len() > 1, "the pass must be a batch, not a single displacement");
		}
	}

	/// (d) Every byte and every object is still accounted for exactly once
	/// after a multi-object drain: the per-demotion bookkeeping did not
	/// change, only how many times it runs per pass.
	#[test]
	fn counters_stay_consistent_after_a_watermark_pass() {
		let size: CacheSize = 10;
		let promoted = watermarks::high_bytes(WM_EFFECTIVE) / size + 1;

		assert!(promoted >= 2, "watermark config leaves no room for this test");

		let mut stack = watermark_stack();

		for key in 1..=promoted {
			promote(&mut stack, key, size as ObjectSize);
		}

		// A brand-new key left sitting in the one-access queue, to pin that
		// the pass touched neither it nor the one-access accounting.
		stack.insert(promoted + 1, 40);

		let migrations = drain(&mut stack);
		let demoted = migrations.len() as CacheSize;

		assert!(demoted >= 1, "the pass must have run");
		assert!(migrations.iter().all(|(_, tier)| *tier == Tier::Slow));

		// Bytes: every promoted key is either main-fast or main-slow, and the
		// one-access key is neither.
		assert_eq!(stack.fast_used, (promoted - demoted) * size);
		assert_eq!(stack.slow_used, demoted * size);
		assert_eq!(stack.one_access_used, 40);
		assert_eq!(stack.fast_used + stack.slow_used, promoted * size);

		// Counts: the two physical main-queue lists partition every promoted
		// key, and nothing was dropped or double-counted.
		assert_eq!(stack.main_fast.len() as CacheSize, promoted - demoted);
		assert_eq!(stack.main_slow.len() as CacheSize, demoted);
		assert_eq!(stack.one_access_queue.len(), 1);
		assert_eq!(stack.len() as CacheSize, promoted + 1);

		// Reported gauges add the one-access queue back onto the fast side.
		assert_eq!(stack.fast_bytes_used(), (promoted - demoted) * size + 40);
		assert_eq!(stack.slow_bytes_used(), demoted * size);
		assert_eq!(stack.fast_object_count() as CacheSize, promoted - demoted + 1);
		assert_eq!(stack.slow_object_count() as CacheSize, demoted);

		// The per-key tier tags agree with those counters.
		let fast_tagged = (1..=promoted).filter(|key| stack.tier_of(*key) == Some(Tier::Fast)).count();
		let slow_tagged = (1..=promoted).filter(|key| stack.tier_of(*key) == Some(Tier::Slow)).count();

		assert_eq!(fast_tagged as CacheSize, promoted - demoted);
		assert_eq!(slow_tagged as CacheSize, demoted);
		assert_eq!(stack.tier_of(promoted + 1), Some(Tier::Fast));

		// The midpoint cursor was maintained once per demotion, so it still
		// points at a key that really is in the slow segment.
		let midpoint = stack.slow_midpoint.expect("a pass that demoted must have seeded the cursor");

		assert_eq!(stack.tier_of(midpoint), Some(Tier::Slow));
		assert!(stack.main_slow.contains(&midpoint));

		// And total DRAM is still within the configured fast tier.
		assert!(stack.fast_bytes_used() <= WM_ONE_ACCESS_CAPACITY + WM_EFFECTIVE);
	}

	// -- shared per-object metadata DRAM reservation --------------------------
	//
	// `shared_overhead` defaults to 0 and nothing above calls
	// `with_shared_overhead`, so every test up to this point sees an
	// unreserved budget (`reserved_overhead() == 0`, both shares 0, both
	// effective capacities exactly what they were before this feature) and is
	// unaffected by everything below -- including the two extra settle calls
	// `insert`/`resize_fast_tier` now make, which cannot fire while the
	// budgets are unchanged and the watermark trigger is a strict `>`.
	//
	// Like the watermark tests above, these compute their expectations from
	// `watermarks::high_bytes()` / `watermarks::low_bytes()` of the relevant
	// *effective* budget rather than hard-coding the default ratios, so they
	// hold at any configured pair -- including `1.0`/`1.0`.

	/// The reservation scales with the *tracked* key count (not the fast-tier
	/// key count) and is split between the two fast segments in proportion to
	/// their capacities, the two shares summing to exactly the whole.
	#[test]
	fn shared_overhead_splits_proportionally_between_the_two_fast_segments() {
		// Equal segments: one_access_capacity = 0.5 * 2_000 = 1_000, and a
		// fast_capacity of 2_000 leaves the main fast segment the other 1_000.
		let mut equal = S3FifoLazyDemotionFastAdmissionMidpointReprieveHybridStack::new(0.5, 2_000, 2_000)
			.with_shared_overhead(100);

		assert_eq!(equal.one_access_capacity, 1_000);

		// Nothing tracked yet -- nothing reserved.
		assert_eq!(equal.reserved_overhead(), 0);
		assert_eq!(equal.reserved_shares(), (0, 0));
		assert_eq!(equal.effective_one_access_capacity(), 1_000);
		assert_eq!(equal.effective_main_fast_capacity(), 1_000);

		// Four 1-byte keys: 4 x 100 = 400 reserved, split 200/200.
		for key in 1..=4u64 {
			equal.insert(key, 1);
		}

		assert_eq!(equal.len(), 4);
		assert_eq!(equal.reserved_overhead(), 400);
		assert_eq!(equal.reserved_shares(), (200, 200));
		assert_eq!(equal.effective_one_access_capacity(), 800);
		assert_eq!(equal.effective_main_fast_capacity(), 800);

		// Unequal segments: one_access_capacity = 0.25 * 4_000 = 1_000 out of
		// a 5_000 fast budget, so the main fast segment gets 4_000 and the
		// split is 1:4. Five 1-byte keys reserve 500:
		// floor(500 * 1_000 / 5_000) = 100 comes out of the one-access queue,
		// the remaining 400 out of the main fast segment.
		let mut unequal = S3FifoLazyDemotionFastAdmissionMidpointReprieveHybridStack::new(0.25, 4_000, 5_000)
			.with_shared_overhead(100);

		for key in 1..=5u64 {
			unequal.insert(key, 1);
		}

		assert_eq!(unequal.one_access_capacity, 1_000);
		assert_eq!(unequal.reserved_overhead(), 500);
		assert_eq!(unequal.reserved_shares(), (100, 400));
		assert_eq!(unequal.effective_one_access_capacity(), 900);
		assert_eq!(unequal.effective_main_fast_capacity(), 3_600);

		// The shares always sum to the whole reservation, so the two fast
		// value budgets plus the metadata are exactly `fast_capacity` -- never
		// more (double-reserving) and never less (leaking DRAM).
		let (one_access_share, main_share) = unequal.reserved_shares();

		assert_eq!(one_access_share + main_share, unequal.reserved_overhead());
		assert_eq!(
			unequal.effective_one_access_capacity() + unequal.effective_main_fast_capacity() + unequal.reserved_overhead(),
			5_000,
		);

		// Nothing to proportion against: both shares are 0 rather than a
		// division by zero.
		let mut zero = S3FifoLazyDemotionFastAdmissionMidpointReprieveHybridStack::new(0.0, 1_000, 0)
			.with_shared_overhead(100);

		zero.insert(1, 1);

		assert_eq!(zero.reserved_overhead(), 100);
		assert_eq!(zero.reserved_shares(), (0, 0));
		assert_eq!(zero.effective_one_access_capacity(), 0);
		assert_eq!(zero.effective_main_fast_capacity(), 0);
	}

	/// The reservation shrinks the effective fast budget, so a reserved stack
	/// demotes on byte-for-byte identical input that an otherwise identical
	/// unreserved stack absorbs without moving anything.
	#[test]
	fn shared_overhead_demotes_earlier_than_an_unreserved_stack() {
		// one_access_capacity = 1.0 * 10_000 = 10_000 -- far larger than
		// anything promoted below, so `settle_one_access` never fires and the
		// main fast segment is the only thing under pressure. A fast_capacity
		// of 12_000 leaves that segment a raw 2_000; total = 12_000.
		const MAIN: CacheSize = 2_000;
		const ONE_ACCESS: CacheSize = 10_000;
		const OVERHEAD: CacheSize = 3_000;

		// Main-segment share of the reservation at 1 and at 2 tracked keys:
		//   1 key : reserved 3_000 -> one-access floor(3_000 * 10_000/12_000)
		//           = 2_500, main 500   -> effective main 1_500
		//   2 keys: reserved 6_000 -> one-access 5_000, main 1_000
		//                               -> effective main 1_000
		const MAIN_SHARE_ONE: CacheSize = 500;
		const MAIN_SHARE_TWO: CacheSize = 1_000;

		let unreserved_high = watermarks::high_bytes(MAIN);
		let reserved_high = watermarks::high_bytes(MAIN - MAIN_SHARE_TWO);

		// Sized so the first object lands exactly ON the reserved stack's high
		// watermark at one tracked key (no trigger for either stack), and the
		// two together land exactly on the UNRESERVED stack's high watermark --
		// the largest usage that still leaves the unreserved stack alone.
		let first = watermarks::high_bytes(MAIN - MAIN_SHARE_ONE);
		let second = unreserved_high.saturating_sub(first);

		assert!(first >= 1, "watermark config leaves no room for this fixture");
		assert!(second >= 1, "watermark config leaves no room for this fixture");
		assert!(first > reserved_high, "watermark config leaves no room for this fixture");
		assert!(second <= reserved_high, "watermark config leaves no room for this fixture");

		// Unreserved: usage lands exactly on the high watermark of the full
		// 2_000 main budget, so nothing moves.
		let mut plain = S3FifoLazyDemotionFastAdmissionMidpointReprieveHybridStack::new(1.0, ONE_ACCESS, ONE_ACCESS + MAIN);

		assert_eq!(plain.effective_main_fast_capacity(), MAIN);

		promote(&mut plain, 1, first as ObjectSize);
		promote(&mut plain, 2, second as ObjectSize);

		assert_eq!(plain.fast_used, unreserved_high);
		assert!(drain(&mut plain).is_empty(), "the unreserved stack must not demote at its own high watermark");
		assert_eq!(plain.tier_of(1), Some(Tier::Fast));
		assert_eq!(plain.tier_of(2), Some(Tier::Fast));
		assert_eq!(plain.slow_object_count(), 0);

		// Reserved: identical input, but each tracked key now costs 3_000
		// bytes of DRAM metadata.
		let mut stack = S3FifoLazyDemotionFastAdmissionMidpointReprieveHybridStack::new(1.0, ONE_ACCESS, ONE_ACCESS + MAIN)
			.with_shared_overhead(OVERHEAD);

		promote(&mut stack, 1, first as ObjectSize);

		assert_eq!(stack.reserved_overhead(), OVERHEAD);
		assert_eq!(stack.reserved_shares(), (2_500, MAIN_SHARE_ONE));
		assert_eq!(stack.effective_main_fast_capacity(), MAIN - MAIN_SHARE_ONE);
		assert_eq!(stack.fast_used, first);
		assert!(drain(&mut stack).is_empty(), "usage exactly on the reserved high watermark must not trigger");
		assert_eq!(stack.tier_of(1), Some(Tier::Fast));

		// Admitting the second key doubles the reservation, which alone drops
		// the main fast budget from 1_500 to 1_000 -- and key 1's bytes are
		// already past the high watermark of that smaller budget.
		promote(&mut stack, 2, second as ObjectSize);

		assert_eq!(stack.reserved_overhead(), 2 * OVERHEAD);
		assert_eq!(stack.reserved_shares(), (5_000, MAIN_SHARE_TWO));
		assert_eq!(stack.effective_main_fast_capacity(), MAIN - MAIN_SHARE_TWO);

		let migrations = drain(&mut stack);

		assert_eq!(migrations, vec![(1, Tier::Slow)], "the reserved stack must demote on input the unreserved one absorbs");
		assert_eq!(stack.tier_of(1), Some(Tier::Slow));
		assert_eq!(stack.tier_of(2), Some(Tier::Fast));
		assert_eq!(stack.fast_used, second);
		assert_eq!(stack.slow_used, first);

		// Same tracked set, same bytes, strictly less of it left in DRAM.
		assert_eq!(stack.len(), plain.len());
		assert!(stack.fast_bytes_used() < plain.fast_bytes_used());

		// The bound the reservation exists to enforce.
		assert!(stack.fast_bytes_used() + stack.reserved_overhead() <= ONE_ACCESS + MAIN);
	}

	/// Fixture for the two composition tests below: 20 promoted 100-byte keys
	/// (2_000 fast bytes) at 100 bytes of overhead each (2_000 reserved),
	/// filled against a roomy 22_000 fast budget so the fill itself settles
	/// nothing, then shrunk to 4_000 -- where one_access_capacity (2_000) and
	/// the raw main capacity (2_000) are equal, so the 2_000 reservation
	/// splits exactly 1_000/1_000 and the main fast segment's *effective*
	/// budget is 1_000 against 2_000 bytes of resident values.
	const OV_SIZE: CacheSize = 100;
	const OV_KEYS: CacheSize = 20;
	const OV_OVERHEAD: CacheSize = 100;
	const OV_FAST_CAPACITY: CacheSize = 4_000;
	const OV_EFFECTIVE_MAIN: CacheSize = 1_000;

	fn overhead_stack() -> S3FifoLazyDemotionFastAdmissionMidpointReprieveHybridStack {
		let mut stack = S3FifoLazyDemotionFastAdmissionMidpointReprieveHybridStack::new(0.5, 4_000, 22_000)
			.with_shared_overhead(OV_OVERHEAD);

		assert_eq!(stack.one_access_capacity, 2_000);

		for key in 1..=OV_KEYS {
			promote(&mut stack, key, OV_SIZE as ObjectSize);
		}

		assert_eq!(stack.fast_used, OV_KEYS * OV_SIZE);
		assert_eq!(stack.reserved_overhead(), OV_KEYS * OV_OVERHEAD);
		assert!(drain(&mut stack).is_empty(), "watermark config leaves no room for this fixture");

		stack
	}

	/// The overhead composes *under* the watermarks, not over them: a
	/// triggered pass drains to `low_bytes(capacity - reserved)`, not to
	/// `low_bytes(capacity)`.
	#[test]
	fn overhead_composes_underneath_the_watermarks() {
		let mut stack = overhead_stack();

		stack.resize_fast_tier(OV_FAST_CAPACITY);

		assert_eq!(stack.reserved_shares(), (1_000, 1_000));
		assert_eq!(stack.effective_main_fast_capacity(), OV_EFFECTIVE_MAIN);

		let effective_low = watermarks::low_bytes(OV_EFFECTIVE_MAIN);
		let unreserved_low = watermarks::low_bytes(OV_FAST_CAPACITY - 2_000);

		assert!(
			watermarks::high_bytes(OV_EFFECTIVE_MAIN) < OV_KEYS * OV_SIZE,
			"watermark config leaves no room for this fixture",
		);
		assert!(effective_low < unreserved_low, "watermark config leaves no room for this fixture");

		let migrations = drain(&mut stack);

		assert!(!migrations.is_empty(), "crossing the effective high watermark must trigger a pass");

		// Drained down to the LOW watermark of the EFFECTIVE budget...
		assert!(stack.fast_used <= effective_low);

		// ...stopping as soon as it got under it (one fewer demotion would
		// have left it above), rather than emptying the segment...
		assert!(stack.fast_used + OV_SIZE > effective_low);

		// ...which is strictly tighter than `low_bytes` of the raw, unreserved
		// main capacity: had the overhead been applied over the watermarks
		// instead of under them, the pass would have settled at the larger
		// target and left the reservation unfunded.
		assert!(
			stack.fast_used < unreserved_low,
			"the drain target must be low_bytes(capacity - reserved), not low_bytes(capacity)",
		);
	}

	/// Every byte and every object is still accounted for exactly once after a
	/// pass triggered by the reservation.
	#[test]
	fn counters_stay_consistent_after_a_reserved_pass() {
		let mut stack = overhead_stack();

		// A brand-new key left sitting in the one-access queue, to pin that
		// the pass touched neither it nor the one-access accounting. Its own
		// admission raises the tracked count to 21, so the reservation is
		// 2_100 and the shares are 1_050/1_050 from here on.
		stack.insert(OV_KEYS + 1, 40);
		assert!(drain(&mut stack).is_empty(), "watermark config leaves no room for this fixture");

		stack.resize_fast_tier(OV_FAST_CAPACITY);

		assert_eq!(stack.reserved_overhead(), (OV_KEYS + 1) * OV_OVERHEAD);
		assert_eq!(stack.reserved_shares(), (1_050, 1_050));
		assert_eq!(stack.effective_main_fast_capacity(), 950);
		assert_eq!(stack.effective_one_access_capacity(), 950);

		let migrations = drain(&mut stack);
		let demoted = migrations.len() as CacheSize;

		assert!(demoted >= 1, "the reserved pass must have run");
		assert!(demoted < OV_KEYS, "the pass must not have emptied the segment");
		assert!(migrations.iter().all(|(_, tier)| *tier == Tier::Slow));

		// One entry per demoted object, off `main_fast`'s tail in promotion
		// order.
		assert_eq!(migrations, (1..=demoted).map(|key| (key, Tier::Slow)).collect::<Vec<_>>());

		// Bytes: every promoted key is either main-fast or main-slow, and the
		// one-access key is neither.
		assert_eq!(stack.fast_used, (OV_KEYS - demoted) * OV_SIZE);
		assert_eq!(stack.slow_used, demoted * OV_SIZE);
		assert_eq!(stack.fast_used + stack.slow_used, OV_KEYS * OV_SIZE);
		assert_eq!(stack.one_access_used, 40);

		// Counts: the three physical lists partition every tracked key.
		assert_eq!(stack.main_fast.len() as CacheSize, OV_KEYS - demoted);
		assert_eq!(stack.main_slow.len() as CacheSize, demoted);
		assert_eq!(stack.one_access_queue.len(), 1);
		assert_eq!(stack.len() as CacheSize, OV_KEYS + 1);
		assert_eq!(
			stack.one_access_queue.len() + stack.main_fast.len() + stack.main_slow.len(),
			stack.len(),
		);

		// Reported gauges add the one-access queue back onto the fast side.
		assert_eq!(stack.fast_bytes_used(), (OV_KEYS - demoted) * OV_SIZE + 40);
		assert_eq!(stack.slow_bytes_used(), demoted * OV_SIZE);
		assert_eq!(stack.fast_object_count() as CacheSize, OV_KEYS - demoted + 1);
		assert_eq!(stack.slow_object_count() as CacheSize, demoted);

		// The reservation is unchanged by a demotion -- a slow-tier key still
		// owns its DRAM hashtable/`entries`/list bookkeeping, which is exactly
		// why it is charged per *tracked* key rather than per fast key.
		assert_eq!(stack.reserved_overhead(), (OV_KEYS + 1) * OV_OVERHEAD);

		// The per-key tier tags agree with those counters.
		let fast_tagged = (1..=OV_KEYS).filter(|key| stack.tier_of(*key) == Some(Tier::Fast)).count();
		let slow_tagged = (1..=OV_KEYS).filter(|key| stack.tier_of(*key) == Some(Tier::Slow)).count();

		assert_eq!(fast_tagged as CacheSize, OV_KEYS - demoted);
		assert_eq!(slow_tagged as CacheSize, demoted);
		assert_eq!(stack.tier_of(OV_KEYS + 1), Some(Tier::Fast));

		// The midpoint cursor was maintained once per demotion, so it still
		// points at a key that really is in the slow segment.
		let midpoint = stack.slow_midpoint.expect("a pass that demoted must have seeded the cursor");

		assert_eq!(stack.tier_of(midpoint), Some(Tier::Slow));
		assert!(stack.main_slow.contains(&midpoint));

		// And the whole point: fast-tier values PLUS the DRAM metadata of
		// every tracked key now fit inside the configured fast tier, which is
		// false without the reservation (2_000 + 2_100 > 4_000).
		assert!(stack.fast_bytes_used() + stack.reserved_overhead() <= OV_FAST_CAPACITY);
	}

	/// A reservation larger than the whole fast budget saturates *both*
	/// segments to zero -- including the one-access queue, which is the point
	/// of splitting the reservation rather than charging it all to the main
	/// segment. Everything is demoted; nothing is evicted.
	#[test]
	fn shared_overhead_exceeding_the_fast_budget_demotes_all_but_never_evicts() {
		// one_access_capacity = 0.5 * 200 = 100, and a fast_capacity of 200
		// leaves the main fast segment the other 100. A single tracked key
		// already reserves 1_000 -- 500 from each segment -- so both effective
		// budgets saturate to 0.
		let mut stack = S3FifoLazyDemotionFastAdmissionMidpointReprieveHybridStack::new(0.5, 200, 200)
			.with_shared_overhead(1_000);

		stack.insert(1, 10);

		assert_eq!(stack.reserved_overhead(), 1_000);
		assert_eq!(stack.reserved_shares(), (500, 500));
		assert_eq!(stack.effective_one_access_capacity(), 0);
		assert_eq!(stack.effective_main_fast_capacity(), 0);

		let migrations = drain(&mut stack);

		assert_eq!(migrations, vec![(1, Tier::Slow)], "an over-reserved one-access queue must reprieve straight into the slow tier");
		assert_eq!(stack.tier_of(1), Some(Tier::Slow));
		assert_eq!(stack.fast_bytes_used(), 0);
		assert_eq!(stack.slow_bytes_used(), 10);

		// Demotion is the only response -- the key is still tracked, and the
		// DRAM budget never evicts (`needs_capacity_eviction` stays default).
		assert!(stack.contains(1));
		assert_eq!(stack.len(), 1);
		assert!(!stack.needs_capacity_eviction());
	}
}
