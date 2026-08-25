/*
 * Copyright (c) Kia Shakiba
 *
 * This source code is licensed under the GNU AGPLv3 license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! `S3FifoLazyDemotionFastAdmissionSplitSlowReprieveHybridStack` —
//! `S3FifoLazyDemotionFastAdmissionMidpointReprieveHybridStack` with the
//! approximate mid-slow-segment checkpoint replaced by a real structural
//! one, for
//! `PaperPolicy::S3FifoLazyDemotionFastAdmissionSplitSlowReprieveHybrid`.
//!
//! Everything else carries over unchanged: no ghost queue, a fast-tier
//! one-access queue whose tail is *reprieved into the slow tier* rather
//! than evicted (`settle_one_access`), demotion-time reference-bit
//! reprieve, and the two-physical-list main queue (see that stack's module
//! doc for why the single-list-plus-boundary-cursor shape was abandoned).
//!
//! ## Why the previous checkpoint was replaced
//!
//! The predecessor kept a `slow_midpoint` cursor -- one key, tracked at
//! approximately the middle of the slow segment via a drift counter -- and
//! checked *that single key's* reference bit once per `evict_one()` call.
//! Benchmarked against the real traces it was indistinguishable from not
//! being there at all (largest difference: 291 hits out of 2.34M accesses,
//! i.e. 0.01%).
//!
//! The reason is structural, and it is NOT a coverage problem. (An earlier
//! draft of this doc claimed the cursor sampled too few keys to matter;
//! that was wrong, and is corrected here. In steady state the cursor holds
//! a roughly fixed index while objects flow past it, so it lands on a new
//! object each cycle and sees most objects that cross the midpoint. Its
//! coverage was fine.)
//!
//! The actual reason: **an earlier checkpoint cannot save anything the tail
//! check wouldn't.** Terminal eviction only ever removes the slow tier's
//! *tail*, so any object whose reference bit is set is already spared when
//! it arrives there. A mid-tier check changes *when* a reaccessed object
//! returns to DRAM, never *whether* it survives.
//!
//! This variant tests the one remaining hypothesis that a checkpoint could
//! still pay off: that checking *every* crossing object -- via a real
//! structural boundary rather than a tracked cursor -- gets hot objects
//! back into DRAM earlier and more uniformly, a residency/latency effect
//! rather than a survival one.
//!
//! **It does not.** Measured against the same three traces, this variant
//! produced hit counts bit-identical to the cursor version on all three,
//! while costing 2.7-11.8% on GET p99 and 1.2-6.9% on GET throughput --
//! the extra Slow->Fast migrations are pure added work on the
//! `PolicyWorker` thread and the object map's shard locks. The lineage's
//! next variant (`..._reprieve_...`, no mid-tier checkpoint at all) drops
//! the mechanic entirely. This file is retained as the record of that
//! negative result.
//!
//! ## The two slow segments
//!
//! * `slow_head` -- front = newest slow object (i.e. exactly the fast/slow
//!   boundary), back = the crossing candidate.
//! * `slow_tail` -- front = objects that just crossed, back = oldest object
//!   overall, and the only terminal-eviction candidate.
//!
//! `slow_head` is held to at most `SLOW_HEAD_RATIO` of the slow tier's
//! bytes by `settle_slow_split()`, which is where the crossing check lives:
//!
//! ```text
//! one-access queue (DRAM) ─┐
//!                          ├─> main_fast (DRAM)
//!    promotions ───────────┘        │ demotion (bit clear)
//!                                   v
//!                              slow_head (PMEM)
//!                                   │ crossing check  ── bit set ──> back to main_fast front
//!                                   v bit clear
//!                              slow_tail (PMEM)
//!                                   │ eviction check  ── bit set ──> back to main_fast front
//!                                   v bit clear
//!                                 evicted
//! ```
//!
//! Both checks share one implementation (`give_second_chance`), since both
//! are "bit is set, so move this object to the front of the fast list",
//! which is also what the demotion-boundary reprieve already did.
//!
//! Note what this buys structurally over the predecessor's cursor: no
//! approximation, no drift counter, no cursor-redirect handling at four
//! separate call sites, and the check is O(1) per crossing rather than
//! O(1) per eviction-pass-sampling-one-key. The whole midpoint apparatus
//! is simply gone.
//!
//! ## Shared DRAM-reservation overhead
//!
//! The object hashtable and this stack's own eviction bookkeeping live in
//! DRAM but are not counted in any of the byte gauges below, so the fast
//! tier's real DRAM footprint would otherwise exceed its budget.
//! `shared_overhead` (see `crate::object::overhead::
//! get_hybrid_dram_shared_overhead`) is the approximate per-*tracked-key*
//! cost of those structures; `reserved_overhead()` scales it by
//! `entries.len()` and the result is carved out of the fast-tier budget, so
//! demotion bounds total DRAM rather than just fast-tier values.
//!
//! Two properties are worth spelling out, because both are easy to get
//! wrong:
//!
//! * **It is charged against every tracked key, not just the fast ones.**
//!   A key's `entries` slot and its `QueueList` node live in DRAM whether
//!   its *data* sits in DRAM or in PMEM -- only the value bytes move on a
//!   demotion. This matches `LruHybridStack`'s `stack.len()` and
//!   `LruSizedHybridStack`'s `entries.len()`.
//!
//! * **It is split proportionally between the two fast segments, not
//!   charged in full to each.** This stack has two independently-capacitied
//!   fast segments -- the one-access queue (`one_access_capacity`) and the
//!   main queue's fast portion (`main_fast_capacity()`, i.e.
//!   `fast_capacity` with the one-access carve-out removed). The underlying
//!   metadata cost is real only once, so `reserved_shares()` proportions it
//!   across the two budgets exactly as `LruSizedHybridStack::reserved_shares`
//!   does. The two slow segments carry no capacity of their own, so they
//!   have nothing to reserve against.
//!
//! There is deliberately no ghost-queue term: this lineage has no ghost
//! queue at all (see the opening of this doc), so `entries` is the complete
//! set of keys this stack holds metadata for, and a per-tracked-key constant
//! models it exactly.
//!
//! The watermarks are applied *on top of* the reserved value, never in
//! place of it -- `high_bytes(capacity - reserved)`, not
//! `high_bytes(capacity)`.

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

/// Fraction of the slow tier's bytes `slow_head` is allowed to hold before
/// `settle_slow_split` starts pushing its tail across into `slow_tail`.
/// 0.5 puts the boundary at the slow tier's midpoint, matching what the
/// predecessor's cursor was approximating -- the difference is that this
/// one is exact and every crossing object is checked, not a sample.
const SLOW_HEAD_RATIO: f64 = 0.5;

/// Which live list a key currently sits in. Doubles as the tier tag --
/// the predecessor carried a separate `Option<Tier>` field alongside a
/// coarser queue tag, which is redundant once the slow tier is two
/// physically distinct lists.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Queue {
	OneAccess,
	Fast,
	SlowHead,
	SlowTail,
}

impl Queue {
	fn tier(self) -> Tier {
		match self {
			Queue::OneAccess | Queue::Fast => Tier::Fast,
			Queue::SlowHead | Queue::SlowTail => Tier::Slow,
		}
	}

	fn is_slow(self) -> bool {
		matches!(self, Queue::SlowHead | Queue::SlowTail)
	}
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct S3FifoEntry {queue: Queue,
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

pub struct S3FifoLazyDemotionFastAdmissionSplitSlowReprieveHybridStack {
	one_access_queue: QueueList,

	/// Main queue, fast portion. Front = newest, back = demotion candidate.
	main_fast: QueueList,
	/// Slow tier, newer half. Front = the fast/slow boundary, back = the
	/// crossing candidate.
	slow_head: QueueList,
	/// Slow tier, older half. Back = oldest object overall, the only
	/// terminal-eviction candidate.
	slow_tail: QueueList,

	entries: EntryMap,

	one_access_ratio: f64,
	one_access_capacity: CacheSize,
	one_access_used: CacheSize,

	fast_capacity: CacheSize,
	fast_used: CacheSize,
	slow_head_used: CacheSize,
	slow_tail_used: CacheSize,

	/// Approximate per-*tracked-key* DRAM cost of the shared structures (the
	/// object hashtable + this stack's eviction bookkeeping), reserved
	/// proportionally between the two fast segments' capacities -- see the
	/// module doc. `0` unless set via `with_shared_overhead`, which is why
	/// every test below that doesn't opt in is unaffected by it.
	shared_overhead: CacheSize,

	migrations: Vec<(HashedKey, Tier)>,
}

impl S3FifoLazyDemotionFastAdmissionSplitSlowReprieveHybridStack {
	#[cfg(not(feature = "eviction_stacks_pmem"))]
	fn new_collections() -> (QueueList, QueueList, QueueList, QueueList, EntryMap) {
		(
			HashList::default(),
			HashList::default(),
			HashList::default(),
			HashList::default(),
			HashMap::default(),
		)
	}

	#[cfg(feature = "eviction_stacks_pmem")]
	fn new_collections() -> (QueueList, QueueList, QueueList, QueueList, EntryMap) {
		(
			PmemHashList::with_hasher(NoHasher::default()),
			PmemHashList::with_hasher(NoHasher::default()),
			PmemHashList::with_hasher(NoHasher::default()),
			PmemHashList::with_hasher(NoHasher::default()),
			HashMap::with_hasher_in(NoHasher::default(), Hybrid),
		)
	}

	pub fn new(one_access_ratio: f64, max_size: CacheSize, fast_capacity: CacheSize) -> Self {
		let (one_access_queue, main_fast, slow_head, slow_tail, entries) = Self::new_collections();

		S3FifoLazyDemotionFastAdmissionSplitSlowReprieveHybridStack {
			one_access_queue,
			main_fast,
			slow_head,
			slow_tail,

			entries,

			one_access_ratio,
			one_access_capacity: (one_access_ratio * max_size as f64) as CacheSize,
			one_access_used: 0,

			fast_capacity,
			fast_used: 0,
			slow_head_used: 0,
			slow_tail_used: 0,

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

	/// The CONFIGURED (pre-reservation) main-fast budget: `fast_capacity`
	/// with the one-access queue's carve-out removed. This is the
	/// proportioning basis for `reserved_shares`, and is deliberately *not*
	/// what `settle_fast_tier` settles against -- that is
	/// `effective_main_fast_capacity()`, which removes this segment's share
	/// of the reservation on top.
	fn main_fast_capacity(&self) -> CacheSize {
		self.fast_capacity.saturating_sub(self.one_access_capacity)
	}

	/// Total DRAM currently reserved for shared per-object metadata across
	/// *both* tiers (`tracked key count x shared_overhead`). A key occupies
	/// exactly one `QueueList` node plus exactly one `entries` slot no matter
	/// which of the four lists it is in, so a demotion or a crossing does not
	/// change this value -- which is what makes it loop-invariant inside
	/// `settle_fast_tier` and `settle_one_access`.
	fn reserved_overhead(&self) -> CacheSize {
		self.entries.len() as CacheSize * self.shared_overhead
	}

	/// Splits `reserved_overhead()` proportionally between the two
	/// independently-capacitied fast segments, as `(one_access, main_fast)`.
	/// Charging the full amount to each would double-count metadata that
	/// exists only once and would waste usable fast-tier budget for no
	/// reason. `(0, 0)` when neither segment has any capacity to proportion
	/// against. Widened to `u128` for the multiply so a large capacity times
	/// a large reservation cannot overflow `CacheSize` (`u64`) mid-expression.
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

	/// The one-access queue's byte budget once its share of the shared
	/// metadata reservation is carved out. Settled against by
	/// `settle_one_access`.
	fn effective_one_access_capacity(&self) -> CacheSize {
		self.one_access_capacity.saturating_sub(self.reserved_shares().0)
	}

	/// The main queue's fast-portion byte budget once its share of the shared
	/// metadata reservation is carved out. Settled against by
	/// `settle_fast_tier`, which applies the watermarks on top of this value.
	fn effective_main_fast_capacity(&self) -> CacheSize {
		self.main_fast_capacity().saturating_sub(self.reserved_shares().1)
	}

	pub fn tier_of(&self, key: HashedKey) -> Option<Tier> {
		self.entries.get(&key).map(|entry| entry.queue.tier())
	}

	/// Returns `true` if `key` currently sits in the older (`slow_tail`)
	/// slow segment -- i.e. it has already survived a crossing check.
	/// Exposed for tests.
	pub fn is_in_slow_tail(&self, key: HashedKey) -> bool {
		self.entries.get(&key).map(|entry| entry.queue) == Some(Queue::SlowTail)
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

		let counter = match entry.queue {
			Queue::OneAccess => &mut self.one_access_used,
			Queue::Fast => &mut self.fast_used,
			Queue::SlowHead => &mut self.slow_head_used,
			Queue::SlowTail => &mut self.slow_tail_used,
		};

		*counter = (*counter as i64 + delta).max(0) as CacheSize;
	}

	fn touch(&mut self, key: HashedKey) {
		match self.entries.get(&key).map(|entry| entry.queue) {
			Some(Queue::OneAccess) => self.promote_from_one_access(key),

			// Lazy: a hit on a main-queue key only sets the reference bit.
			// It is read at three points -- the demotion boundary
			// (`settle_fast_tier`), the slow-segment crossing
			// (`settle_slow_split`), and the eviction tail (`evict_one`).
			Some(_) => self.mark_accessed(key),

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
		self.entries.insert(key, S3FifoEntry { dram_resident, queue: Queue::Fast,
			size,
			accessed: false,
		});
		self.fast_used += size_bytes;

		self.settle_fast_tier();
	}

	/// Moves `key` to the front of the fast list and clears its reference
	/// bit. Shared by all three reference-bit check points (demotion
	/// boundary, slow-segment crossing, eviction tail), since all three
	/// mean the same thing: this object was reaccessed, so spare it.
	fn give_second_chance(&mut self, key: HashedKey) {
		let Some(entry) = self.entries.get(&key).copied() else { return };
		let size = entry.migrating();
		let was_slow = entry.queue.is_slow();

		match entry.queue {
			// Only reachable from `evict_one`'s fast-tail fallback (nothing
			// has ever been demoted): reorder within the fast list, no tier
			// change and no byte movement.
			Queue::Fast => {
				self.main_fast.move_front(&key);
			},

			Queue::SlowHead => {
				self.slow_head.remove(&key);
				self.slow_head_used = self.slow_head_used.saturating_sub(size);
				self.main_fast.push_front(key);
				self.fast_used += size;
			},

			Queue::SlowTail => {
				self.slow_tail.remove(&key);
				self.slow_tail_used = self.slow_tail_used.saturating_sub(size);
				self.main_fast.push_front(key);
				self.fast_used += size;
			},

			Queue::OneAccess => return,
		}

		if let Some(entry) = self.entries.get_mut(&key) {
			entry.queue = Queue::Fast;
			entry.accessed = false;
		}

		self.settle_fast_tier();

		// Only record a migration when the object genuinely crossed tiers
		// AND survived the settle above (which can demote it straight back
		// out, in which case that call already pushed the correct
		// `Tier::Slow` migration itself). A key that was already Fast needs
		// no migration at all -- unlike the predecessor, which pushed a
		// redundant Fast->Fast entry here and made `PolicyWorker` rebuild
		// an identical buffer for nothing. That waste is worth avoiding
		// specifically in this variant, where `give_second_chance` fires
		// far more often than before (every crossing, not one sampled key
		// per eviction).
		if was_slow && self.entries.get(&key).map(|entry| entry.queue) == Some(Queue::Fast) {
			self.migrations.push((key, Tier::Fast));
		}
	}

	/// Demotes oldest-first out of `main_fast` into `slow_head` under the
	/// shared fast-tier watermarks (`super::watermarks`), reprieving any key
	/// whose bit is set instead. Terminates even when every fast key's bit
	/// is set, since each reprieve clears one bit.
	///
	/// A pass triggers only once `fast_used` exceeds
	/// `high_bytes(effective_capacity)`, and once triggered it drains all the
	/// way down to `low_bytes(effective_capacity)` rather than stopping at
	/// the ceiling. This replaces the previous drain-to-exactly-the-ceiling
	/// rule, which pinned the fast tier at 100% utilisation and left almost
	/// every triggered pass demoting exactly one object -- migration batches
	/// of one, which maximise per-batch worker overhead and cannot be
	/// parallelised. See `super::watermarks`' doc for the full rationale and
	/// for the `FAST_TIER_HIGH_WATERMARK`/`FAST_TIER_LOW_WATERMARK`
	/// overrides (setting both to `1.0` restores the old drain-to-ceiling
	/// behaviour exactly).
	///
	/// `effective_main_fast_capacity()` is still this stack's fast-tier
	/// ceiling -- `fast_capacity` with the one-access queue's carve-out
	/// removed (that queue is fast-tier here and settles against its own
	/// budget), and now with this segment's proportional share of the shared
	/// per-object metadata reservation removed on top of that (see the module
	/// doc's "Shared DRAM-reservation overhead"; it saturates to 0 when the
	/// metadata alone meets or exceeds the segment's budget). The watermarks
	/// are applied on top of that effective value, never in place of it.
	///
	/// The effective value is loop-invariant, which is why it is read once up
	/// front: a demotion moves a key from `main_fast` into `slow_head` but
	/// leaves it tracked in `entries`, so `reserved_shares()` cannot shift
	/// underneath the loop.
	///
	/// Only the loop's entry condition and its stopping point changed. The
	/// per-demotion bookkeeping below (the reference-bit reprieve, the queue
	/// tag, `fast_used`, `slow_head_used`, and the migration emission) is
	/// untouched and still runs exactly once per demoted object.
	///
	/// Deliberately does NOT call `settle_slow_split` -- that method calls
	/// `give_second_chance`, which calls back into here, so the two must
	/// not be mutually recursive. `settle_slow_split` is instead driven
	/// from the public trait methods, and its own loop re-checks after any
	/// nested demotion this method performs.
	fn settle_fast_tier(&mut self) {
		let effective_capacity = self.effective_main_fast_capacity();

		// Trigger: nothing happens at all until usage crosses the high
		// watermark. Checked once, up front, rather than per iteration --
		// that is what lets a triggered pass keep draining *below* the high
		// watermark, down to the low one.
		if self.fast_used <= watermarks::high_bytes(effective_capacity) {
			return;
		}

		let low_water = watermarks::low_bytes(effective_capacity);

		while self.fast_used > low_water {
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
			self.slow_head.push_front(candidate);

			if let Some(entry) = self.entries.get_mut(&candidate) {
				entry.queue = Queue::SlowHead;
			}

			self.fast_used = self.fast_used.saturating_sub(size);
			self.slow_head_used += size;

			self.migrations.push((candidate, Tier::Slow));
		}
	}

	/// Holds `slow_head` to at most `SLOW_HEAD_RATIO` of the slow tier's
	/// bytes, and -- the point of this variant -- checks each object's
	/// reference bit at the moment it would cross into `slow_tail`. A set
	/// bit means the object was reaccessed since it was demoted, so it goes
	/// back to the front of the fast list instead of crossing.
	///
	/// Every crossing object is checked, which is the substantive
	/// difference from the predecessor's single-sampled-key cursor.
	///
	/// Termination: each iteration either moves an object across (strictly
	/// reducing `slow_head_used`) or promotes it out of `slow_head`
	/// entirely. A nested `settle_fast_tier` inside `give_second_chance`
	/// can push bytes back into `slow_head`, but only for keys whose bit is
	/// clear (that method reprieves the rest), and a clear-bit key at
	/// `slow_head`'s back always crosses on the following iteration.
	fn settle_slow_split(&mut self) {
		loop {
			let total = self.slow_head_used + self.slow_tail_used;

			if total == 0 || (self.slow_head_used as f64) <= total as f64 * SLOW_HEAD_RATIO {
				break;
			}

			let Some(candidate) = self.slow_head.back().copied() else { break };

			let accessed = self.entries.get(&candidate).map(|entry| entry.accessed).unwrap_or(false);

			if accessed {
				self.give_second_chance(candidate);
				continue;
			}

			let size = self.entries.get(&candidate).map(|entry| entry.migrating()).unwrap_or(0);

			self.slow_head.pop_back();
			self.slow_tail.push_front(candidate);

			if let Some(entry) = self.entries.get_mut(&candidate) {
				entry.queue = Queue::SlowTail;
			}

			self.slow_head_used = self.slow_head_used.saturating_sub(size);
			self.slow_tail_used += size;

			// No migration: both segments are the slow tier, so the bytes
			// do not move between DRAM and PMEM.
		}
	}

	/// Relieves one-access-queue pressure by moving its tail(s) to the front
	/// of `slow_head` -- the fast/slow boundary -- as an O(1) `push_front`.
	/// Called synchronously from `insert()`/`resize()`, never through
	/// `evict_one()`: nothing is removed from the cache here, and routing it
	/// through eviction would make `apply_evictions` erase a live object (or
	/// fall back to evicting a random one). See the predecessor's module doc
	/// for the full account of that bug.
	///
	/// Settles against `effective_one_access_capacity()` -- this segment's
	/// proportional share of the shared per-object metadata reservation
	/// carved out of `one_access_capacity`, since the one-access queue is a
	/// fast-tier (DRAM) segment with a budget of its own. Read once up front
	/// for the same reason as in `settle_fast_tier`: a reprieve into
	/// `slow_head` leaves the key tracked, so the reservation is
	/// loop-invariant here too.
	///
	/// No watermarks here, deliberately. This segment relieves pressure by
	/// reprieving into the slow tier synchronously from `insert`/`resize`,
	/// which is not a `PolicyWorker` migration batch, so there is no
	/// batch-of-one cost for a low-water drain to amortise away.
	fn settle_one_access(&mut self) {
		let effective_capacity = self.effective_one_access_capacity();

		while self.one_access_used > effective_capacity {
			let Some(key) = self.one_access_queue.pop_back() else { break };
			let Some(entry) = self.entries.get(&key).copied() else { continue };
			let size = entry.migrating();

			self.one_access_used = self.one_access_used.saturating_sub(size);

			self.slow_head.push_front(key);

			if let Some(stored) = self.entries.get_mut(&key) {
				stored.queue = Queue::SlowHead;
				stored.accessed = false;
			}

			self.slow_head_used += size;

			self.migrations.push((key, Tier::Slow));
		}
	}
}

impl PolicyStack for S3FifoLazyDemotionFastAdmissionSplitSlowReprieveHybridStack {
	fn is_policy(&self, policy: &PaperPolicy) -> bool {
		matches!(policy, PaperPolicy::S3FifoLazyDemotionFastAdmissionSplitSlowReprieveHybrid(ratio) if *ratio == self.one_access_ratio)
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
			self.settle_slow_split();
			return;
		}

		self.one_access_queue.push_front(key);
		self.entries.insert(key, S3FifoEntry { dram_resident, queue: Queue::OneAccess,
			size,
			accessed: false,
		});
		self.one_access_used += (size as CacheSize).saturating_sub(dram_resident as CacheSize);

		self.settle_one_access();
		self.settle_slow_split();
	}

	fn update(&mut self, key: HashedKey) {
		if self.entries.contains_key(&key) {
			self.touch(key);
			self.settle_slow_split();
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

			Queue::Fast => {
				self.main_fast.remove(&key);
				self.fast_used = self.fast_used.saturating_sub(size);
			},

			Queue::SlowHead => {
				self.slow_head.remove(&key);
				self.slow_head_used = self.slow_head_used.saturating_sub(size);
			},

			Queue::SlowTail => {
				self.slow_tail.remove(&key);
				self.slow_tail_used = self.slow_tail_used.saturating_sub(size);
			},
		}
	}

	fn resize(&mut self, max_size: CacheSize) {
		self.one_access_capacity = (self.one_access_ratio * max_size as f64) as CacheSize;
		self.settle_one_access();
		self.settle_fast_tier();
		self.settle_slow_split();
	}

	fn clear(&mut self) {
		self.one_access_queue.clear();
		self.main_fast.clear();
		self.slow_head.clear();
		self.slow_tail.clear();
		self.entries.clear();

		self.one_access_used = 0;
		self.fast_used = 0;
		self.slow_head_used = 0;
		self.slow_tail_used = 0;
		self.migrations.clear();
	}

	fn evict_one(&mut self) -> Option<HashedKey> {
		// Crossing checks fire here, keeping the split balanced before the
		// tail is evaluated. The one-access queue is never consulted --
		// its pressure is relieved synchronously by `settle_one_access`.
		self.settle_slow_split();

		loop {
			// The oldest slow object is the real candidate; fall back
			// through slow_head, then the fast tail, only when the older
			// lists are empty (i.e. little or nothing has been demoted).
			let (key, from) = if let Some(key) = self.slow_tail.back().copied() {
				(key, Queue::SlowTail)
			} else if let Some(key) = self.slow_head.back().copied() {
				(key, Queue::SlowHead)
			} else {
				(self.main_fast.back().copied()?, Queue::Fast)
			};

			let accessed = self.entries.get(&key).map(|entry| entry.accessed).unwrap_or(false);

			if accessed {
				self.give_second_chance(key);
				continue;
			}

			let size = self.entries.get(&key).map(|entry| entry.migrating()).unwrap_or(0);

			match from {
				Queue::SlowTail => {
					self.slow_tail.pop_back();
					self.slow_tail_used = self.slow_tail_used.saturating_sub(size);
				},

				Queue::SlowHead => {
					self.slow_head.pop_back();
					self.slow_head_used = self.slow_head_used.saturating_sub(size);
				},

				Queue::Fast => {
					self.main_fast.pop_back();
					self.fast_used = self.fast_used.saturating_sub(size);
				},

				Queue::OneAccess => break,
			}

			self.entries.remove(&key);

			return Some(key);
		}

		None
	}

	fn resize_fast_tier(&mut self, size: CacheSize) {
		self.fast_capacity = size;
		self.settle_fast_tier();
		self.settle_slow_split();
	}

	fn drain_tier_migrations(&mut self) -> Vec<(HashedKey, Tier)> {
		std::mem::take(&mut self.migrations)
	}

	fn fast_bytes_used(&self) -> CacheSize {
		self.fast_used + self.one_access_used
	}

	fn slow_bytes_used(&self) -> CacheSize {
		self.slow_head_used + self.slow_tail_used
	}

	fn fast_object_count(&self) -> usize {
		self.main_fast.len() + self.one_access_queue.len()
	}

	fn slow_object_count(&self) -> usize {
		self.slow_head.len() + self.slow_tail.len()
	}
}

#[cfg(test)]
mod tests {
	use super::*;

	fn drain(stack: &mut S3FifoLazyDemotionFastAdmissionSplitSlowReprieveHybridStack) -> Vec<(HashedKey, Tier)> {
		stack.drain_tier_migrations()
	}

	/// One-access capacity every watermark-sensitive fixture below is built
	/// with (ratio 1.0 x this `max_size`). Sized so `settle_one_access` never
	/// fires: a key sits in the one-access queue only until its `update()`
	/// promotes it into `main_fast`, which is the list the watermarks govern.
	const ONE_ACCESS_CAPACITY: CacheSize = 1_000;

	/// Effective main-fast budget the watermark tests size against -- i.e.
	/// what `effective_main_fast_capacity()` returns once the one-access
	/// carve-out above is subtracted from `fast_capacity`. Deliberately
	/// paired with the 1-byte objects `fill_fast` inserts: that makes the
	/// drain byte-exact, so a triggered pass lands on precisely `low_bytes()`
	/// and the expectations below hold at any configured watermark ratio
	/// rather than only at the default ratios. (The watermarks are
	/// process-global `OnceLock`s, so a test cannot pin them via env vars
	/// without racing every other test in the binary -- expectations are
	/// computed from `watermarks::` instead.)
	const CAPACITY: CacheSize = 1_000;

	/// A stack whose `effective_main_fast_capacity()` is exactly `CAPACITY`.
	fn watermark_stack() -> S3FifoLazyDemotionFastAdmissionSplitSlowReprieveHybridStack {
		S3FifoLazyDemotionFastAdmissionSplitSlowReprieveHybridStack::new(
			1.0, ONE_ACCESS_CAPACITY, ONE_ACCESS_CAPACITY + CAPACITY,
		)
	}

	/// Byte threshold at which a demotion pass triggers, for `CAPACITY`.
	fn high_bytes() -> CacheSize {
		watermarks::high_bytes(CAPACITY)
	}

	/// Byte target a triggered demotion pass drains down to, for `CAPACITY`.
	fn low_bytes() -> CacheSize {
		watermarks::low_bytes(CAPACITY)
	}

	/// Effective main-fast budget at which `resident` bytes sit at or below
	/// the low watermark while `resident + step` bytes sit above the high
	/// one -- so a pass fires once the `step` bytes land and stops with
	/// `resident` bytes still in `main_fast`. Derived from `watermarks::`
	/// rather than hard-coded so the fixture holds at any configured ratio;
	/// at the drain-to-ceiling pair (1.0/1.0) it returns exactly `resident`.
	///
	/// A two-object fixture like this one only exists when the marks are
	/// within a factor of two of each other (`low() > high() / 2`) -- below
	/// that, one object cannot sit under the low mark while two sit over the
	/// high one, whatever the budget. That is a property of the fixture's
	/// shape, not of `settle_fast_tier`; the watermark tests further down use
	/// 1-byte objects instead and hold at any pair.
	fn budget_holding(resident: CacheSize, step: CacheSize) -> CacheSize {
		(resident..=(resident + step) * 100)
			.find(|budget| {
				watermarks::low_bytes(*budget) >= resident
					&& watermarks::high_bytes(*budget) < resident + step
			})
			.expect("no budget fits this fixture; needs FAST_TIER_LOW_WATERMARK > FAST_TIER_HIGH_WATERMARK / 2")
	}

	/// Inserts `count` 1-byte keys numbered from `first` and re-accesses each
	/// one, promoting it out of the one-access queue into `main_fast` with a
	/// CLEAR reference bit (`promote_from_one_access` clears it), so nothing
	/// in the fill is eligible for a demotion-time reprieve. Returns the keys
	/// in promotion order, so index 0 is `main_fast`'s back -- the first
	/// demotion candidate.
	fn fill_fast(
		stack: &mut S3FifoLazyDemotionFastAdmissionSplitSlowReprieveHybridStack,
		first: HashedKey,
		count: CacheSize,
	) -> Vec<HashedKey> {
		(0..count)
			.map(|offset| {
				let key = first + offset;

				stack.insert(key, 1);
				stack.update(key);

				key
			})
			.collect()
	}

	#[test]
	fn admission_always_lands_in_one_access_queue_fast() {
		let mut stack = S3FifoLazyDemotionFastAdmissionSplitSlowReprieveHybridStack::new(1.0, 1_000, 1_000);

		stack.insert(1, 10);
		stack.insert(2, 10);

		assert_eq!(stack.tier_of(1), Some(Tier::Fast));
		assert_eq!(stack.tier_of(2), Some(Tier::Fast));
		assert_eq!(drain(&mut stack), Vec::new());
	}

	#[test]
	fn a_key_aging_out_without_reaccess_is_moved_to_slow_instead_of_evicted() {
		// one_access_capacity = 0.01 * 1_000 = 10 -- fits exactly one
		// 10-byte key, so admitting a second reprieves the first.
		let mut stack = S3FifoLazyDemotionFastAdmissionSplitSlowReprieveHybridStack::new(0.01, 1_000, 1_000);

		stack.insert(1, 10);
		drain(&mut stack);
		assert_eq!(stack.tier_of(1), Some(Tier::Fast), "still in the one-access queue");

		stack.insert(2, 10);

		assert!(stack.contains(1), "the key must still be tracked, not gone");
		assert_eq!(stack.tier_of(1), Some(Tier::Slow));
		assert_eq!(stack.tier_of(2), Some(Tier::Fast));
		assert_eq!(drain(&mut stack), vec![(1, Tier::Slow)]);
	}

	// ── the signature new mechanic: a real crossing checkpoint ─────────────

	/// Three keys in the slow tier, oldest-first. `one_access_ratio` of 0.0
	/// makes `settle_one_access` fire from within `insert()` itself, so each
	/// key lands in `slow_head` immediately with a CLEAR reference bit --
	/// note there is deliberately no `update()` here, unlike the equivalent
	/// helpers in this stack's predecessors: at this ratio a key never
	/// passes through the fast list, so an `update()` would only set the
	/// reference bit and defeat the point of the fixture.
	///
	/// `slow_head` is held to half the slow bytes, so 3 x 10 bytes settles
	/// at slow_head = [3] (10 bytes) / slow_tail = [2, 1] (20 bytes).
	fn build_three_slow_keys() -> S3FifoLazyDemotionFastAdmissionSplitSlowReprieveHybridStack {
		let mut stack = S3FifoLazyDemotionFastAdmissionSplitSlowReprieveHybridStack::new(0.0, 1_000, 0);

		for key in 1..=3u64 {
			stack.insert(key, 10);
		}

		drain(&mut stack);
		stack
	}

	#[test]
	fn the_split_pushes_the_older_slow_keys_into_the_tail_segment() {
		let stack = build_three_slow_keys();

		assert_eq!(stack.tier_of(1), Some(Tier::Slow));
		assert_eq!(stack.tier_of(2), Some(Tier::Slow));
		assert_eq!(stack.tier_of(3), Some(Tier::Slow));

		assert!(stack.is_in_slow_tail(1), "oldest should have crossed into the tail segment");
		assert!(stack.is_in_slow_tail(2), "second-oldest should have crossed too");
		assert!(!stack.is_in_slow_tail(3), "newest should still be in the head segment");
		assert_eq!(stack.slow_bytes_used(), 30);
	}

	#[test]
	fn a_reaccessed_key_is_promoted_at_the_crossing_instead_of_crossing() {
		// slow_head = [3], slow_tail = [2, 1]; key 3 is the next crossing
		// candidate.
		let mut stack = build_three_slow_keys();
		assert!(!stack.is_in_slow_tail(3));

		// Reaccess key 3 -- lazily, so nothing moves yet.
		stack.update(3);
		assert_eq!(stack.tier_of(3), Some(Tier::Slow), "a mere access must not itself migrate");
		drain(&mut stack);

		// Give the fast tier somewhere to promote into (it was 0 in the
		// fixture, which would have demoted the key straight back out).
		stack.resize_fast_tier(100);

		// Grow slow_head until a crossing is due: at 4 keys the split is
		// exactly balanced, at 5 it tips. Key 3 is the crossing candidate
		// and its bit is set, so it must be promoted rather than cross.
		stack.insert(4, 10);
		stack.insert(5, 10);

		assert_eq!(stack.tier_of(3), Some(Tier::Fast), "the reaccessed key should have been promoted at the crossing check");
		assert!(!stack.is_in_slow_tail(3), "and must not have crossed into the tail segment");
		assert!(drain(&mut stack).contains(&(3, Tier::Fast)), "a real Slow->Fast migration must be recorded");
	}

	#[test]
	fn an_unaccessed_key_crosses_normally() {
		let mut stack = build_three_slow_keys();

		// Key 3 is the head segment's only occupant and has a clear bit.
		// Growing the slow tier pushes it across rather than promoting it.
		assert!(!stack.is_in_slow_tail(3));

		// At 4 keys the split is exactly balanced; the 5th tips it and
		// makes key 3 the crossing candidate.
		stack.insert(4, 10);
		stack.insert(5, 10);
		drain(&mut stack);

		assert!(stack.is_in_slow_tail(3), "an unaccessed key must cross, not be promoted");
		assert_eq!(stack.tier_of(3), Some(Tier::Slow));
	}

	#[test]
	fn eviction_takes_the_slow_tail_and_still_honors_the_reference_bit() {
		let mut stack = build_three_slow_keys();

		// Key 1 is the oldest (slow_tail's back). Untouched -> evicted.
		assert_eq!(stack.evict_one(), Some(1));
		assert!(!stack.contains(1));

		// Now key 2 is the oldest; reaccess it so the tail check spares it.
		stack.update(2);
		stack.resize_fast_tier(100);

		let evicted = stack.evict_one();

		assert_eq!(stack.tier_of(2), Some(Tier::Fast), "an accessed tail key should be promoted, not evicted");
		assert_ne!(evicted, Some(2));
	}

	#[test]
	fn a_reprieved_key_can_still_be_promoted_by_a_later_access() {
		let mut stack = S3FifoLazyDemotionFastAdmissionSplitSlowReprieveHybridStack::new(0.01, 1_000, 1_000);

		stack.insert(1, 10);
		stack.insert(2, 10);
		stack.insert(3, 10);
		drain(&mut stack);

		assert_eq!(stack.tier_of(1), Some(Tier::Slow));
		assert_eq!(stack.tier_of(2), Some(Tier::Slow));

		stack.update(1);
		drain(&mut stack);

		let evicted = stack.evict_one();

		assert_eq!(stack.tier_of(1), Some(Tier::Fast), "a reprieved key stays promotable via the ordinary second chance");
		assert_ne!(evicted, Some(1));
	}

	#[test]
	fn an_accessed_fast_boundary_key_is_reprieved_at_demotion_time_not_demoted() {
		// `one_access_capacity` is 1_000 (ratio 1.0), so everything added on
		// top of it is the effective main-fast budget. Sized from the
		// watermarks so one 10-byte key rests at or below the low mark while
		// two cross the high one -- i.e. the pass fires and stops with
		// exactly one key resident. At the drain-to-ceiling config (1.0/1.0)
		// this is the literal 1_010 the fixture used before the watermarks.
		let mut stack = S3FifoLazyDemotionFastAdmissionSplitSlowReprieveHybridStack::new(
			1.0, 1_000, 1_000 + budget_holding(10, 10),
		);

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

	#[test]
	fn evict_one_falls_back_to_the_fast_tail_when_nothing_has_ever_been_demoted() {
		let mut stack = S3FifoLazyDemotionFastAdmissionSplitSlowReprieveHybridStack::new(1.0, 1_000, 10_000);

		for key in 1..=3u64 {
			stack.insert(key, 10);
			stack.update(key);
		}
		drain(&mut stack);

		assert_eq!(stack.slow_object_count(), 0, "nothing should have been demoted yet");

		assert_eq!(stack.evict_one(), Some(1));
		assert!(!stack.contains(1));
		assert_eq!(stack.fast_bytes_used(), 20);
	}

	#[test]
	fn remove_handles_every_list() {
		let mut stack = S3FifoLazyDemotionFastAdmissionSplitSlowReprieveHybridStack::new(0.02, 1_000, 10_000);

		// one_access_capacity = 20 (two 10-byte keys). Land one key in each
		// list: key 2 promoted to fast by its update(), key 1 pushed out of
		// the one-access queue into slow once keys 3 and 4 fill it, and
		// keys 3/4 left sitting in the one-access queue.
		stack.insert(1, 10);
		stack.insert(2, 10);
		stack.update(2);
		stack.insert(3, 10);
		stack.insert(4, 10);
		drain(&mut stack);

		assert_eq!(stack.tier_of(1), Some(Tier::Slow), "oldest one-access key should have been reprieved into slow");
		assert_eq!(stack.tier_of(2), Some(Tier::Fast));

		stack.remove(1);
		stack.remove(2);
		stack.remove(3);
		stack.remove(4);

		assert_eq!(stack.len(), 0);
		assert_eq!(stack.fast_bytes_used(), 0);
		assert_eq!(stack.slow_bytes_used(), 0);
	}

	#[test]
	fn clear_resets_all_four_lists() {
		let mut stack = build_three_slow_keys();

		stack.clear();

		assert_eq!(stack.len(), 0);
		assert_eq!(stack.tier_of(1), None);
		assert_eq!(stack.fast_bytes_used(), 0);
		assert_eq!(stack.slow_bytes_used(), 0);
		assert_eq!(stack.evict_one(), None);
	}

	#[test]
	fn fast_and_slow_gauges_include_one_access_queue_on_the_fast_side() {
		let mut stack = S3FifoLazyDemotionFastAdmissionSplitSlowReprieveHybridStack::new(1.0, 1_000, 1_000);

		stack.insert(1, 10);
		stack.insert(2, 10);
		drain(&mut stack);

		assert_eq!(stack.fast_bytes_used(), 20);
		assert_eq!(stack.slow_bytes_used(), 0);
		assert_eq!(stack.fast_object_count(), 2);
		assert_eq!(stack.slow_object_count(), 0);
	}

	// ---- fast-tier watermarks (`super::watermarks`) ----

	#[test]
	fn usage_at_the_high_watermark_triggers_no_demotion() {
		let mut stack = watermark_stack();

		// Fills to exactly `high_bytes()` -- the largest usage the trigger
		// (`fast_used > high_bytes`) still leaves alone.
		let keys = fill_fast(&mut stack, 1, high_bytes());

		assert_eq!(drain(&mut stack), Vec::new(), "no object should have crossed a tier boundary");
		assert_eq!(stack.fast_bytes_used(), high_bytes());
		assert_eq!(stack.slow_bytes_used(), 0);
		assert_eq!(stack.fast_object_count(), keys.len());
		assert_eq!(stack.slow_object_count(), 0);
	}

	#[test]
	fn usage_above_the_high_watermark_triggers_a_pass() {
		let mut stack = watermark_stack();

		fill_fast(&mut stack, 1, high_bytes());
		drain(&mut stack);

		assert_eq!(stack.slow_object_count(), 0, "nothing demotes below the high watermark");

		// A single byte over the high watermark is enough.
		fill_fast(&mut stack, high_bytes() + 1, 1);
		let migrations = drain(&mut stack);

		let demotions = migrations
			.iter()
			.filter(|(_, tier)| *tier == Tier::Slow)
			.count();

		assert!(demotions > 0);
		assert!(stack.slow_object_count() > 0);

		// ...and it fired while usage was still comfortably inside
		// `effective_main_fast_capacity()`, which is all the old
		// `fast_used > effective_capacity` rule ever waited for. Skipped when
		// the high watermark is configured back to 1.0, which deliberately
		// restores trigger-at-ceiling.
		if watermarks::high() < 1.0 {
			assert!(high_bytes() + 1 <= CAPACITY);
		}
	}

	#[test]
	fn a_triggered_pass_drains_to_the_low_watermark_not_the_ceiling() {
		let mut stack = watermark_stack();

		let keys = fill_fast(&mut stack, 1, high_bytes() + 1);
		let migrations = drain(&mut stack);

		let demoted = migrations
			.iter()
			.filter(|(_, tier)| *tier == Tier::Slow)
			.map(|(key, _)| *key)
			.collect::<Vec<_>>();

		// 1-byte objects make the drain byte-exact, so the pass stops on
		// precisely the object that brings usage to the low watermark -- not
		// one demotion earlier, not one later.
		assert_eq!(stack.fast_bytes_used(), low_bytes());
		assert_eq!(demoted.len() as CacheSize, high_bytes() + 1 - low_bytes());

		// The whole point of the low watermark: the pass keeps going well
		// past the ceiling the old rule stopped at. Skipped when the low
		// watermark is configured back to 1.0 (drain-to-ceiling).
		if watermarks::low() < 1.0 {
			assert!(stack.fast_bytes_used() < CAPACITY);
		}

		// Demotion order is oldest-first out of `main_fast`'s back, same as
		// before the watermarks.
		assert_eq!(demoted, keys[..demoted.len()].to_vec());
	}

	#[test]
	fn counters_stay_consistent_after_a_watermark_pass() {
		let mut stack = watermark_stack();

		let total = high_bytes() + 1;
		let keys = fill_fast(&mut stack, 1, total);
		drain(&mut stack);

		let demoted = total - low_bytes();

		// The pass moved bytes between the tier counters; it neither lost nor
		// double-counted any, and every key is still tracked exactly once.
		assert_eq!(stack.len(), total as usize);
		assert_eq!(stack.fast_bytes_used(), low_bytes());
		assert_eq!(stack.slow_bytes_used(), demoted);
		assert_eq!(stack.fast_bytes_used() + stack.slow_bytes_used(), total);

		// Objects are 1 byte apiece, so each byte counter doubles as a count.
		// `slow_object_count()` spans both slow segments, so this also covers
		// the objects `settle_slow_split` pushed on into `slow_tail`.
		assert_eq!(stack.fast_object_count() as CacheSize, low_bytes());
		assert_eq!(stack.slow_object_count() as CacheSize, demoted);
		assert_eq!(stack.fast_object_count() + stack.slow_object_count(), total as usize);

		// Per-key tier tags agree with the aggregate counters.
		let fast = keys.iter().filter(|key| stack.tier_of(**key) == Some(Tier::Fast)).count();
		let slow = keys.iter().filter(|key| stack.tier_of(**key) == Some(Tier::Slow)).count();

		assert_eq!(fast, stack.fast_object_count());
		assert_eq!(slow, stack.slow_object_count());

		// And the pass is idempotent: usage now rests under the high
		// watermark, so re-settling demotes nothing further.
		stack.resize_fast_tier(ONE_ACCESS_CAPACITY + CAPACITY);

		assert_eq!(drain(&mut stack), Vec::new());
		assert_eq!(stack.fast_bytes_used(), low_bytes());
		assert_eq!(stack.slow_bytes_used(), demoted);
	}

	// ---- shared DRAM-reservation overhead (`with_shared_overhead`) ----
	//
	// Every test ABOVE this line builds its stack without
	// `with_shared_overhead`, so `shared_overhead` is 0, `reserved_shares()`
	// is `(0, 0)`, and both effective capacities equal the configured ones.
	// That is why none of them needed rescaling for this change -- not
	// because the reservation was weakened anywhere.

	/// `watermark_stack()`'s segment shape (one-access 1_000 : main-fast
	/// 1_000 once resized), but with a per-tracked-key reservation and a
	/// deliberately oversized initial fast capacity, so the fill itself
	/// demotes nothing and a single later `resize_fast_tier` triggers exactly
	/// one pass at a known, FIXED tracked-key count. Fixing the count is what
	/// makes the reservation arithmetic exact: `reserved_overhead()` scales
	/// with `entries.len()`, which a mid-fill trigger would leave changing
	/// under the assertion.
	fn reserved_stack(overhead: CacheSize) -> S3FifoLazyDemotionFastAdmissionSplitSlowReprieveHybridStack {
		S3FifoLazyDemotionFastAdmissionSplitSlowReprieveHybridStack::new(
			1.0, ONE_ACCESS_CAPACITY, 1_000_000,
		)
		.with_shared_overhead(overhead)
	}

	/// Tracked-key count the two `reserved_stack` fixtures below fill to, and
	/// (since their objects are 1 byte apiece) their resident byte count.
	const RESERVED_KEYS: CacheSize = 500;

	/// Per-tracked-key reservation those fixtures use. 500 keys x 3 = 1_500
	/// bytes reserved, apportioned 1_000:1_000 across the two fast segments
	/// -> 750 each, leaving an effective main-fast budget of 250.
	const RESERVED_OVERHEAD: CacheSize = 3;

	/// The effective main-fast budget `reserved_stack(RESERVED_OVERHEAD)` has
	/// once resized to the standard fixture capacity. Chosen strictly below
	/// `RESERVED_KEYS` so a pass fires at ANY configured high watermark
	/// (`fast_used` 500 > `high_bytes(250)` <= 250), not only at the default.
	const RESERVED_EFFECTIVE: CacheSize = 250;

	#[test]
	fn shared_overhead_reserves_a_share_of_each_fast_segment() {
		// Equal segments: one_access_capacity = 0.5 x 1_000 = 500, and
		// main_fast_capacity() = fast_capacity - one_access_capacity = 500.
		let mut plain = S3FifoLazyDemotionFastAdmissionSplitSlowReprieveHybridStack::new(0.5, 1_000, 1_000);
		let mut split = S3FifoLazyDemotionFastAdmissionSplitSlowReprieveHybridStack::new(0.5, 1_000, 1_000)
			.with_shared_overhead(20);

		// Five 1-byte keys left sitting in the one-access queue (deliberately
		// no `update()`), so nothing settles anywhere and the tracked count is
		// exactly 5 in both stacks.
		for key in 1..=5u64 {
			plain.insert(key, 1);
			split.insert(key, 1);
		}

		assert_eq!(plain.len(), 5);
		assert_eq!(split.len(), 5);
		assert_eq!(plain.slow_object_count(), 0);
		assert_eq!(split.slow_object_count(), 0);

		// No reservation configured -> the effective budgets ARE the
		// configured ones. This is the property that leaves every test above
		// untouched by this change.
		assert_eq!(plain.reserved_overhead(), 0);
		assert_eq!(plain.reserved_shares(), (0, 0));
		assert_eq!(plain.effective_one_access_capacity(), 500);
		assert_eq!(plain.effective_main_fast_capacity(), 500);

		// 5 tracked keys x 20 bytes = 100 bytes of shared metadata, split
		// 500:500 -> 50 apiece.
		assert_eq!(split.reserved_overhead(), 100);
		assert_eq!(split.reserved_shares(), (50, 50));
		assert_eq!(split.effective_one_access_capacity(), 450);
		assert_eq!(split.effective_main_fast_capacity(), 450);

		// The shares SUM to the total: the metadata exists once, so it is
		// reserved once, not once per segment.
		let (one_access_share, main_fast_share) = split.reserved_shares();
		assert_eq!(one_access_share + main_fast_share, split.reserved_overhead());
	}

	#[test]
	fn the_reservation_splits_in_proportion_to_the_two_segment_budgets() {
		// Unequal segments: one_access_capacity = 0.25 x 1_000 = 250, and
		// main_fast_capacity() = 1_000 - 250 = 750.
		let mut stack = S3FifoLazyDemotionFastAdmissionSplitSlowReprieveHybridStack::new(0.25, 1_000, 1_000)
			.with_shared_overhead(20);

		for key in 1..=5u64 {
			stack.insert(key, 1);
		}

		// 100 bytes reserved, apportioned 250:750 -> 25 / 75.
		assert_eq!(stack.reserved_overhead(), 100);
		assert_eq!(stack.reserved_shares(), (25, 75));
		assert_eq!(stack.effective_one_access_capacity(), 225);
		assert_eq!(stack.effective_main_fast_capacity(), 675);

		// The bug this guards against: charging the full 100 to each segment
		// independently, which would have left 150 / 650 and thrown away 100
		// bytes of usable fast tier for metadata that exists only once.
		assert!(stack.effective_one_access_capacity() > 250 - 100);
		assert!(stack.effective_main_fast_capacity() > 750 - 100);
	}

	#[test]
	fn shared_overhead_reserves_dram_and_demotes_earlier() {
		// `high_bytes()` 1-byte keys is, by construction, the largest fill an
		// unreserved stack leaves completely alone -- see
		// `usage_at_the_high_watermark_triggers_no_demotion`. Both stacks get
		// that identical fixture and that identical fill; only the
		// reservation differs.
		let count = high_bytes();

		let mut plain = watermark_stack();
		fill_fast(&mut plain, 1, count);

		assert_eq!(drain(&mut plain), Vec::new(), "the unreserved stack rests exactly at its high watermark");
		assert_eq!(plain.slow_object_count(), 0);
		assert_eq!(plain.fast_bytes_used(), count);

		// 2 bytes per tracked key, apportioned 1_000:1_000, so the main-fast
		// share is ceil(n x 2 / 2) = n and the effective budget is
		// `CAPACITY - n` -- a budget that SHRINKS as the fill grows. At the
		// default 0.95 high mark the pass therefore first fires at n = 488
		// (488 > floor(0.95 x 512) = 486, while 487 > floor(0.95 x 513) = 487
		// is false), versus n = 951 with no reservation at all.
		let mut reserved = watermark_stack().with_shared_overhead(2);
		fill_fast(&mut reserved, 1, count);

		let migrations = drain(&mut reserved);

		assert!(
			migrations.iter().any(|(_, tier)| *tier == Tier::Slow),
			"the reservation must force demotions the unreserved stack never needed",
		);
		assert!(reserved.slow_object_count() > 0);
		assert!(reserved.fast_bytes_used() < plain.fast_bytes_used());

		// Both stacks still track every key: reserving DRAM demotes, it never
		// evicts.
		assert_eq!(plain.len(), count as usize);
		assert_eq!(reserved.len(), count as usize);
	}

	#[test]
	fn overhead_composes_with_the_watermarks_on_the_reduced_budget() {
		let mut stack = reserved_stack(RESERVED_OVERHEAD);

		fill_fast(&mut stack, 1, RESERVED_KEYS);
		drain(&mut stack);

		assert_eq!(stack.fast_bytes_used(), RESERVED_KEYS, "the oversized fixture must not demote during the fill");
		assert_eq!(stack.slow_object_count(), 0);
		assert_eq!(stack.len(), RESERVED_KEYS as usize);

		// Shrink to the standard fixture budget. The tracked count is pinned
		// at 500 across the whole pass, so the arithmetic is exact:
		//
		//   reserved            = 500 keys x 3 bytes  = 1_500
		//   segment budgets     = 1_000 : 1_000
		//   shares              =   750 :   750
		//   effective main-fast = 1_000 - 750         =   250
		stack.resize_fast_tier(ONE_ACCESS_CAPACITY + CAPACITY);

		assert_eq!(stack.reserved_overhead(), RESERVED_KEYS * RESERVED_OVERHEAD);
		assert_eq!(stack.reserved_shares(), (750, 750));
		assert_eq!(stack.effective_main_fast_capacity(), RESERVED_EFFECTIVE);

		// THE POINT: the pass drained to `low_bytes(250)` -- the low watermark
		// of the RESERVED budget -- and not to `low_bytes(CAPACITY)`. At the
		// default 0.75 low mark that is 187 bytes resident rather than 750.
		// 1-byte objects keep the drain byte-exact, so it lands precisely.
		assert_eq!(stack.fast_bytes_used(), watermarks::low_bytes(RESERVED_EFFECTIVE));

		// ...and it could not be the unreserved target even in principle: the
		// reserved budget is 250, so any watermark of it is at most 250, while
		// the unreserved budget is CAPACITY (1_000) on its own.
		assert!(stack.fast_bytes_used() <= RESERVED_EFFECTIVE);

		// Skipped at a low watermark configured under 0.25, where flooring
		// collapses the two targets together.
		if watermarks::low() > 0.25 {
			assert!(stack.fast_bytes_used() < watermarks::low_bytes(CAPACITY));
		}

		// The TRIGGER composed the same way, not just the drain target: 500
		// resident bytes sit comfortably under `high_bytes(CAPACITY)` (950 by
		// default), so without the reservation this pass would not have fired
		// at all. Skipped when the high mark is configured under 0.5, where
		// 500 bytes clears the unreserved trigger too.
		if watermarks::high() >= 0.5 {
			assert!(RESERVED_KEYS <= watermarks::high_bytes(CAPACITY));
		}
	}

	#[test]
	fn counters_stay_consistent_after_a_reserved_watermark_pass() {
		let mut stack = reserved_stack(RESERVED_OVERHEAD);

		let keys = fill_fast(&mut stack, 1, RESERVED_KEYS);
		drain(&mut stack);

		stack.resize_fast_tier(ONE_ACCESS_CAPACITY + CAPACITY);
		let migrations = drain(&mut stack);

		let resident = watermarks::low_bytes(RESERVED_EFFECTIVE);
		let demoted = RESERVED_KEYS - resident;

		// Every key is still tracked exactly once -- the reservation demotes,
		// it never evicts.
		assert_eq!(stack.len(), RESERVED_KEYS as usize);

		// Bytes moved between the tier counters; none were lost or
		// double-counted.
		assert_eq!(stack.fast_bytes_used(), resident);
		assert_eq!(stack.slow_bytes_used(), demoted);
		assert_eq!(stack.fast_bytes_used() + stack.slow_bytes_used(), RESERVED_KEYS);

		// Objects are 1 byte apiece, so each byte counter doubles as a count.
		// `slow_object_count()` spans both slow segments, so this also covers
		// whatever `settle_slow_split` pushed on into `slow_tail`.
		assert_eq!(stack.fast_object_count() as CacheSize, resident);
		assert_eq!(stack.slow_object_count() as CacheSize, demoted);
		assert_eq!(stack.fast_object_count() + stack.slow_object_count(), RESERVED_KEYS as usize);

		// Exactly one migration per demoted object, all in the demotion
		// direction, oldest-first out of `main_fast`'s back.
		let demotions = migrations
			.iter()
			.filter(|(_, tier)| *tier == Tier::Slow)
			.map(|(key, _)| *key)
			.collect::<Vec<_>>();

		assert_eq!(demotions.len() as CacheSize, demoted);
		assert_eq!(migrations.len() as CacheSize, demoted, "no spurious promotions");
		assert_eq!(demotions, keys[..demotions.len()].to_vec());

		// Per-key tier tags agree with the aggregate counters.
		let fast = keys.iter().filter(|key| stack.tier_of(**key) == Some(Tier::Fast)).count();
		let slow = keys.iter().filter(|key| stack.tier_of(**key) == Some(Tier::Slow)).count();

		assert_eq!(fast, stack.fast_object_count());
		assert_eq!(slow, stack.slow_object_count());

		// And the pass is idempotent against the REDUCED budget: usage now
		// rests at the low watermark of 250, which is at or below its high
		// watermark, so re-settling demotes nothing further.
		stack.resize_fast_tier(ONE_ACCESS_CAPACITY + CAPACITY);

		assert_eq!(drain(&mut stack), Vec::new());
		assert_eq!(stack.fast_bytes_used(), resident);
		assert_eq!(stack.slow_bytes_used(), demoted);
	}

	#[test]
	fn a_reservation_exceeding_the_fast_budget_reprieves_but_never_evicts() {
		// one_access_capacity = 1.0 x 100 = 100 and main_fast_capacity() = 0,
		// so the whole 100-byte fast budget belongs to the one-access queue
		// and takes the entire 1_000-byte reservation. Its effective budget
		// saturates to 0.
		let mut stack = S3FifoLazyDemotionFastAdmissionSplitSlowReprieveHybridStack::new(1.0, 100, 100)
			.with_shared_overhead(1_000);

		stack.insert(1, 10);

		assert_eq!(stack.reserved_overhead(), 1_000);
		assert_eq!(stack.reserved_shares(), (1_000, 0));
		assert_eq!(stack.effective_one_access_capacity(), 0);
		assert_eq!(stack.effective_main_fast_capacity(), 0);

		// The object is reprieved into the slow tier on admission...
		assert_eq!(drain(&mut stack), vec![(1, Tier::Slow)]);
		assert_eq!(stack.tier_of(1), Some(Tier::Slow));
		assert_eq!(stack.fast_bytes_used(), 0);

		// ...but is still tracked. The DRAM budget demotes; terminal eviction
		// stays governed solely by `max_size`, so `needs_capacity_eviction`
		// remains the trait's default `false`.
		assert_eq!(stack.len(), 1);
		assert!(stack.contains(1));
		assert!(!stack.needs_capacity_eviction());
	}
}
