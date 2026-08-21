/*
 * Copyright (c) Kia Shakiba
 *
 * This source code is licensed under the GNU AGPLv3 license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! `TwoQFastAdmissionReprieveHybridStack` —
//! `TwoQFastAdmissionHybridStack` with one behavioral change, for
//! `PaperPolicy::TwoQFastAdmissionReprieveHybrid`: **a one-access key that
//! ages out of the FIFO queue without a second access is reprieved into the
//! slow tier instead of being evicted outright.**
//!
//! Everything else is identical to `TwoQFastAdmissionHybridStack` — fast-tier
//! admission, the FIFO reservation carved out of `fast_capacity`, the main
//! queue's LRU fast/slow segmentation, no ghost queue. See that stack's
//! module doc for all of it.
//!
//! ## Where a reprieved key lands, and why it is the *bottom*
//!
//! `settle_fifo_queue` splices the aged-out key onto the **back of
//! `main_stack`** — the absolute LRU tail, i.e. the next terminal-eviction
//! candidate — tagged `Tier::Slow`.
//!
//! This is deliberately weaker than the equivalent in
//! `s3_fifo_lazy_demotion_fast_admission_reprieve_hybrid_stack`, which splices
//! to the *front* of its slow segment (`main_slow.push_front`), giving a
//! reprieved key a full traversal of the slow tier before it can be evicted.
//! Two reasons for the difference:
//!
//! 1. **Rank.** The main queue here is LRU-ordered, and its slow segment holds
//!    keys that were promoted to fast at least once and later demoted. A key
//!    aging out of the one-access queue has demonstrated *no* reuse at all, so
//!    ranking it above proven-but-cold objects would invert the ordering the
//!    main queue exists to maintain. The bottom is where an unproven object
//!    belongs.
//! 2. **Cost.** `push_back` is O(1) on the existing single `main_stack` list.
//!    The s3-fifo variant needed to insert at the fast/slow *boundary*, which
//!    `HashList`/`PmemHashList` cannot do (only `push_front`/`push_back`/
//!    `move_front`/`move_back` are exposed) — its first implementation walked
//!    every fast key per reprieve and burned ~18 minutes of worker CPU on a
//!    real trace without completing a run, which is what forced that stack's
//!    two-physical-lists restructure. Landing at the back needs no such
//!    restructure and cannot corrupt `main_boundary`.
//!
//! The tradeoff to be aware of when reading results: a reprieved key sitting
//! at the LRU tail may be evicted very soon under steady capacity pressure,
//! having cost a real DRAM→PMEM copy on the way there. Whether that copy buys
//! enough extra hits to pay for itself is exactly the question this variant
//! exists to answer — a null result here would be a real finding, not a bug.
//! If it *is* null, the front-of-slow placement above is the natural next
//! thing to try, and would need this stack's `main_stack` split in two the way
//! the s3-fifo variant's was.
//!
//! ## The boundary invariant survives a back-splice
//!
//! `main_boundary` marks the LRU-most `Tier::Fast` key, and the design relies
//! on fast keys forming a contiguous prefix from the list's head. Pushing a
//! `Tier::Slow` key onto the absolute back preserves that: everything fast is
//! still in front of everything slow, and `main_boundary` still points at the
//! same key it did before. No boundary update is needed, which is the other
//! reason this placement is O(1).
//!
//! ## The reprieve must NOT run through `evict_one()`
//!
//! `settle_fifo_queue` is called **synchronously from `insert()`/`resize()`**,
//! mirroring `settle_fast_tier`'s relationship to the fast/slow boundary — not
//! surfaced via `needs_capacity_eviction()`/`evict_one()` the way
//! `TwoQFastAdmissionHybridStack` surfaces the same pressure.
//!
//! That difference is load-bearing, and both halves of it are lessons this
//! crate already learned the hard way:
//!
//! * `PolicyWorker::apply_evictions` unconditionally *erases* whatever key
//!   `evict_one()` returns from the entire cache, and if it returns `None`,
//!   `erase()` falls back to evicting a **random** object. A reprieve is
//!   neither of those — nothing should leave the cache just because the FIFO
//!   queue needed relief, and `over_max_size` may not even be true at that
//!   moment. (`s3_fifo_lazy_demotion_fast_admission_reprieve`'s first draft
//!   routed a reprieve through `evict_one()` and hit exactly this.)
//! * The converse rule still holds: a `PolicyStack` may never *remove* a key
//!   on its own, because it cannot touch the object map or `AtomicStatus` —
//!   `TwoQHybridStack`'s first draft did, and permanently desynced the stack
//!   from the real object map (`has()` kept returning `true` for keys the
//!   stack had "forgotten"). That rule is not violated here precisely
//!   *because* a reprieve removes nothing: the key stays in `entries` and in
//!   the object map, and only moves between two of this stack's own lists.
//!
//! So `needs_capacity_eviction()` returns to the trait's default `false`, and
//! `evict_one()` becomes purely about the main queue — with one last-resort
//! FIFO fallback (see its doc) that exists only to avoid handing
//! `apply_evictions` a `None` it would answer with a random eviction.
//!
//! ## Shared DRAM-reservation overhead
//!
//! Both of this stack's own structures (`fifo_queue`/`main_stack` and
//! `entries`) and the shared object hashtable are DRAM-resident, and none of
//! their bytes are counted in `fifo_used`/`fast_used` — so the fast tier's
//! real DRAM footprint exceeds its budget by exactly that metadata.
//! `shared_overhead` (see `crate::object::overhead::
//! get_hybrid_dram_shared_overhead`) is the approximate per-tracked-key cost
//! of all of it. `reserved_overhead` multiplies it by the tracked-key count —
//! *every* tracked key, not just the fast-tier ones, because a slow-tier
//! object still owns a hashtable slot, a `main_stack` node and an `entries`
//! slot, all of them DRAM whichever tier its data sits in — and
//! `reserved_shares` divides that total between this stack's TWO
//! DRAM-resident segments, in proportion to their raw capacities:
//!
//! * the one-access queue's reservation, `fifo_capacity`, enforced by
//!   `settle_fifo_queue` via `effective_fifo_capacity`; and
//! * the main queue's fast segment, `fast_capacity - fifo_capacity`, enforced
//!   by `settle_fast_tier` via `effective_main_fast_capacity`.
//!
//! Dividing rather than charging the full amount against each is the point,
//! and is the same rule `LruSizedHybridStack::reserved_shares` follows for
//! its two fast segments: the metadata cost is real only once, and both
//! segments here are carved out of the same `fast_capacity`, so
//! double-charging would waste usable DRAM budget for nothing. The split
//! preserves `effective_fifo + effective_main + reserved <= fast_capacity`,
//! i.e. the fast-tier budget now bounds *total* DRAM (values + shared
//! metadata) rather than just fast-tier values.
//!
//! Charging the whole reservation to the main segment alone would bound total
//! DRAM just as well, but would distribute the cost backwards: the main fast
//! segment (proven, twice-accessed objects) would collapse toward nothing
//! while unproven one-access keys kept their full DRAM budget.
//!
//! The high/low watermarks are applied *on top of* each reserved value, never
//! in place of it — see `settle_fast_tier`.
//!
//! There is no ghost queue in this design (see the `TwoQFastAdmissionHybrid`
//! module doc), so there is no bare-key term to charge on top of the
//! per-tracked-key one: every key this stack holds a list node for is also in
//! `entries` and in the object map.

#[cfg(not(feature = "eviction_stacks_pmem"))]
use std::collections::HashMap;
#[cfg(feature = "eviction_stacks_pmem")]
use hashbrown::HashMap;

#[cfg(not(feature = "eviction_stacks_pmem"))]
use kwik::collections::HashList;
#[cfg(feature = "eviction_stacks_pmem")]
use super::pmem_collections::PmemHashList;

// Eviction-stack metadata is allocated through the same crate-wide `Hybrid`
// alias (`SlowObjects`, node-1 jemalloc arenas) that `BufferPMEM`/other PMEM
// features already use.
#[cfg(feature = "eviction_stacks_pmem")]
use crate::Hybrid;

use crate::{
	CacheSize,
	HashedKey,
	NoHasher,
	policy::PaperPolicy,
	object::ObjectSize,
	worker::policy::policy_stack::{PolicyStack, Tier, watermarks},
};

/// Which live queue a key currently belongs to.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Queue {
	Fifo,
	Main,
}

/// Combined per-key bookkeeping: which queue, which tier (only meaningful
/// while `queue == Main` — a `Fifo` key is always physically Fast in this
/// design, so it needs no stored tier), and the object's size.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct TwoQEntry {
	queue: Queue,
	tier: Option<Tier>,
	size: ObjectSize,
}

#[cfg(not(feature = "eviction_stacks_pmem"))]
type QueueList = HashList<HashedKey, NoHasher>;
#[cfg(feature = "eviction_stacks_pmem")]
type QueueList = PmemHashList<HashedKey, NoHasher>;

#[cfg(not(feature = "eviction_stacks_pmem"))]
type EntryMap = HashMap<HashedKey, TwoQEntry, NoHasher>;
#[cfg(feature = "eviction_stacks_pmem")]
type EntryMap = HashMap<HashedKey, TwoQEntry, NoHasher, Hybrid>;

pub struct TwoQFastAdmissionReprieveHybridStack {
	fifo_queue: QueueList,
	main_stack: QueueList,

	entries: EntryMap,

	k_in: f64,

	/// The FIFO queue's own byte budget. Unlike `TwoQHybridStack`, this is a
	/// reservation carved out of `fast_capacity` (both are DRAM now) — see
	/// `effective_main_fast_capacity`.
	fifo_capacity: CacheSize,
	fifo_used: CacheSize,

	/// Total fast-tier (DRAM) budget, covering BOTH the FIFO queue and the
	/// main queue's fast segment.
	fast_capacity: CacheSize,

	/// Approximate per-tracked-key DRAM cost of the shared structures (object
	/// hashtable + this stack's own eviction bookkeeping) that hold an entry
	/// for every tracked key of both tiers. Reserved out of `fast_capacity`
	/// — split between the two DRAM segments by `reserved_shares` — so the
	/// fast-tier budget bounds total DRAM, not just fast-tier values. `0`
	/// unless set via `with_shared_overhead`, so unit tests exercising the
	/// pure value-budget behavior are unaffected.
	shared_overhead: CacheSize,

	/// Bytes held by `main_stack` keys tagged `Tier::Fast`. Does NOT include
	/// `fifo_used`, even though both are physically DRAM — see
	/// `fast_bytes_used`, which sums them for reporting.
	fast_used: CacheSize,
	slow_used: CacheSize,

	/// Number of keys currently tagged `Tier::Fast` within `main_stack`.
	fast_count: usize,

	/// Number of keys currently in the `Main` queue (Fast or Slow).
	main_count: usize,

	/// The least-recently-used key currently tagged `Tier::Fast` within
	/// `main_stack` — i.e. the next demotion candidate. `None` iff no key in
	/// `main_stack` is currently Fast.
	main_boundary: Option<HashedKey>,

	/// (key, new tier) pairs recorded since the last `drain_tier_migrations`.
	migrations: Vec<(HashedKey, Tier)>,
}

impl TwoQFastAdmissionReprieveHybridStack {
	#[cfg(not(feature = "eviction_stacks_pmem"))]
	fn new_collections() -> (QueueList, QueueList, EntryMap) {
		(HashList::default(), HashList::default(), HashMap::default())
	}

	#[cfg(feature = "eviction_stacks_pmem")]
	fn new_collections() -> (QueueList, QueueList, EntryMap) {
		(
			PmemHashList::with_hasher(NoHasher::default()),
			PmemHashList::with_hasher(NoHasher::default()),
			HashMap::with_hasher_in(NoHasher::default(), Hybrid),
		)
	}

	pub fn new(k_in: f64, max_size: CacheSize, fast_capacity: CacheSize) -> Self {
		let (fifo_queue, main_stack, entries) = Self::new_collections();

		TwoQFastAdmissionReprieveHybridStack {
			fifo_queue,
			main_stack,

			entries,

			k_in,
			fifo_capacity: (k_in * max_size as f64) as CacheSize,
			fifo_used: 0,

			fast_capacity,
			shared_overhead: 0,
			fast_used: 0,
			slow_used: 0,
			fast_count: 0,
			main_count: 0,

			main_boundary: None,
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
	/// both tiers: `tracked key count × shared_overhead`.
	///
	/// Counts every key in `entries`, not just the fast-tier ones — a
	/// slow-tier object still owns a hashtable slot, a `main_stack` node and
	/// an `entries` slot, all DRAM regardless of where its *data* lives. This
	/// mirrors `LruHybridStack::reserved_overhead` (`stack.len()`) and
	/// `LruSizedHybridStack::reserved_shares` (`entries.len()`).
	///
	/// No ghost-queue term: this design has no ghost queue, so every key this
	/// stack holds a list node for is also a tracked `entries` key.
	fn reserved_overhead(&self) -> CacheSize {
		self.entries.len() as CacheSize * self.shared_overhead
	}

	/// Splits [`Self::reserved_overhead`] between the two DRAM-resident
	/// segments — `(one-access share, main-fast share)` — in proportion to
	/// their raw capacities, `fifo_capacity` and `fast_capacity -
	/// fifo_capacity`, which sum to `fast_capacity`.
	///
	/// Divided once rather than charged in full against each: both segments
	/// come out of the same `fast_capacity`, and the underlying metadata cost
	/// is real only once, so double-charging would waste usable DRAM budget.
	/// Same rule, same shape as `LruSizedHybridStack::reserved_shares`.
	///
	/// `(0, 0)` when `fast_capacity` is 0 (nothing to proportion against).
	/// The one-access term is capped at `fast_capacity` first, so the
	/// degenerate `fifo_capacity > fast_capacity` configuration (see
	/// [`Self::effective_main_fast_capacity`]) puts the whole reservation on
	/// the one-access side — the only segment with any budget left there.
	fn reserved_shares(&self) -> (CacheSize, CacheSize) {
		let reserved = self.reserved_overhead();

		if self.fast_capacity == 0 {
			return (0, 0);
		}

		let fifo_capacity = self.fifo_capacity.min(self.fast_capacity);

		let fifo_share = ((reserved as u128 * fifo_capacity as u128) / self.fast_capacity as u128) as CacheSize;
		let main_share = reserved.saturating_sub(fifo_share);

		(fifo_share, main_share)
	}

	/// How many bytes the one-access FIFO queue may hold once its share of
	/// the shared-metadata reservation is taken out. Enforced by
	/// [`Self::settle_fifo_queue`], which reprieves the tail into the slow
	/// tier until the queue fits.
	fn effective_fifo_capacity(&self) -> CacheSize {
		self.fifo_capacity.saturating_sub(self.reserved_shares().0)
	}

	/// How much of `fast_capacity` the main queue's fast segment may use,
	/// after BOTH the FIFO queue's reservation and this segment's share of
	/// the shared-metadata reservation are carved out.
	///
	/// Saturating rather than panicking on `fifo_capacity > fast_capacity`:
	/// that is a legitimate (if degenerate) configuration — see the module
	/// doc — and it means "the main queue gets no fast segment", not an
	/// error. The metadata reservation saturates the same way, so a
	/// reservation that meets or exceeds what is left simply drives the
	/// effective budget to 0 (every main key demotes; nothing is evicted).
	fn effective_main_fast_capacity(&self) -> CacheSize {
		self.fast_capacity
			.saturating_sub(self.fifo_capacity)
			.saturating_sub(self.reserved_shares().1)
	}

	/// Returns which tier the given (currently tracked) key is in, or `None`
	/// if the key isn't tracked. Exposed for tests/diagnostics.
	///
	/// The `Fifo` arm is the one line that differs from `TwoQHybridStack`'s
	/// equivalent: a one-access key is physically Fast here, not Slow.
	pub fn tier_of(&self, key: HashedKey) -> Option<Tier> {
		let entry = self.entries.get(&key)?;

		match entry.queue {
			Queue::Fifo => Some(Tier::Fast),
			Queue::Main => entry.tier,
		}
	}

	/// Records a size change for an already-tracked key without altering its
	/// queue/tier, adjusting whichever counter currently applies.
	///
	/// Callers must re-settle the fast tier afterwards when the key is
	/// `Fifo`-resident and grew: `fifo_used` is a DRAM reservation here, so
	/// growing it shrinks the main queue's effective budget. Both current
	/// callers do (`insert` explicitly, `update` via `touch`).
	fn resize_key(&mut self, key: HashedKey, new_size: ObjectSize) {
		let Some(entry) = self.entries.get_mut(&key) else { return };

		let old_size = entry.size;
		entry.size = new_size;
		let delta = new_size as i64 - old_size as i64;

		match (entry.queue, entry.tier) {
			(Queue::Fifo, _) => {
				self.fifo_used = (self.fifo_used as i64 + delta).max(0) as CacheSize;
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

	/// Treats an already-tracked key as accessed: a `Fifo` key promotes
	/// straight to `Main`+`Fast`; a `Main` key is handled by
	/// `touch_main_fast` (reorder if already Fast, promote if Slow).
	fn touch(&mut self, key: HashedKey) {
		match self.entries.get(&key).map(|entry| entry.queue) {
			Some(Queue::Fifo) => self.promote_from_fifo(key),
			Some(Queue::Main) => self.touch_main_fast(key),
			None => {},
		}
	}

	/// Moves a `fifo_queue`-resident key to the front of `main_stack`,
	/// tagging it `Tier::Fast`.
	///
	/// Emits **no** `(key, Tier::Fast)` migration, unlike
	/// `TwoQHybridStack::promote_from_fifo`: the key's bytes were already
	/// physically Fast (admission built them that way), so this is a
	/// bookkeeping move between two DRAM-resident structures, not a data
	/// move. See the module doc's "A migration this design no longer needs".
	///
	/// It can still *cause* migrations: the bytes move out of the FIFO
	/// reservation and into the main-fast budget, so `settle_fast_tier` below
	/// may have to demote other keys to make room — including, at a
	/// tight-enough budget, this very key straight back out again, which that
	/// call records correctly as a genuine `(key, Tier::Slow)`.
	fn promote_from_fifo(&mut self, key: HashedKey) {
		let Some(entry) = self.entries.get(&key) else { return };
		let size = entry.size;
		let size_bytes = size as CacheSize;

		self.fifo_queue.remove(&key);
		self.fifo_used = self.fifo_used.saturating_sub(size_bytes);

		self.main_stack.push_front(key);
		self.entries.insert(key, TwoQEntry { queue: Queue::Main, tier: Some(Tier::Fast), size });
		self.fast_used += size_bytes;
		self.fast_count += 1;
		self.main_count += 1;

		if self.main_boundary.is_none() {
			self.main_boundary = Some(key);
		}

		self.settle_fast_tier();
	}

	/// Moves an already-`Main`-tracked key to the front of `main_stack`,
	/// promoting it to `Tier::Fast` if it was `Slow`, then settles the fast
	/// tier. Unchanged from `TwoQHybridStack`: a slow→fast move here IS a
	/// real data movement (PMEM→DRAM), so it still emits a migration.
	fn touch_main_fast(&mut self, key: HashedKey) {
		let previous_tier = self.entries.get(&key).and_then(|entry| entry.tier);

		let already_at_front = self.main_stack.front() == Some(&key);
		let is_boundary = self.main_boundary == Some(key);

		let new_boundary_if_moved = if is_boundary && !already_at_front {
			self.main_stack.before(&key).copied()
		} else {
			None
		};

		self.main_stack.move_front(&key);

		if is_boundary && !already_at_front {
			self.main_boundary = new_boundary_if_moved;
		}

		let mut promoted = false;

		if previous_tier != Some(Tier::Fast) {
			if previous_tier == Some(Tier::Slow) {
				let size = self.entries.get(&key).map(|entry| entry.size).unwrap_or(0) as CacheSize;

				self.slow_used = self.slow_used.saturating_sub(size);
				self.fast_used += size;
				self.fast_count += 1;

				promoted = true;
			}

			if let Some(entry) = self.entries.get_mut(&key) {
				entry.tier = Some(Tier::Fast);
			}

			if self.main_boundary.is_none() {
				self.main_boundary = Some(key);
			}
		}

		self.settle_fast_tier();

		// Pushed *after* `settle_fast_tier` (which pushes any demotions this
		// promotion itself triggered), not before: `apply_tier_migrations`
		// applies demotions before promotions, and within each phase in push
		// order, so pushing the promotion first would risk its DRAM
		// allocation landing before the corresponding demotion's DRAM free.
		// Guarded on the key still being `Fast`: a tight budget can demote it
		// straight back out within the same `settle_fast_tier` call, in which
		// case that call already pushed the correct final `(key, Tier::Slow)`.
		if promoted && self.entries.get(&key).and_then(|entry| entry.tier) == Some(Tier::Fast) {
			self.migrations.push((key, Tier::Fast));
		}
	}

	/// Demotes the least-recently-used fast key(s) within `main_stack` once
	/// `fast_used` exceeds the HIGH watermark of
	/// [`Self::effective_main_fast_capacity`], then drains down to the LOW
	/// watermark rather than merely back under the ceiling.
	///
	/// Composition order: [`Self::effective_main_fast_capacity`] is
	/// `fast_capacity` minus the FIFO reservation minus this segment's share
	/// of the shared per-object DRAM metadata reservation, and the watermarks
	/// are applied to *that* value. So the drain target is
	/// `low_bytes(capacity - reservations)`, never `low_bytes(capacity)` —
	/// the reservation shrinks the budget the watermarks then scale, it does
	/// not replace them and they do not replace it.
	///
	/// The capacity the watermarks are applied *to* is the one substantive
	/// difference from `TwoQHybridStack::settle_fast_tier`, which applies
	/// them to raw `fast_capacity` — correct there, where the FIFO queue is
	/// PMEM and competes for nothing. The watermarks sit on top of that
	/// effective value; they never replace it.
	///
	/// Draining below the ceiling instead of exactly to it is what turns the
	/// steady state from "every promotion demotes exactly one object" into
	/// occasional multi-object batches — see [`watermarks`] for the full
	/// rationale, and for how to restore the old drain-to-ceiling behaviour
	/// exactly (`FAST_TIER_HIGH_WATERMARK=1.0`, `FAST_TIER_LOW_WATERMARK=1.0`).
	///
	/// Nothing else moves: the per-demotion bookkeeping in the loop below
	/// (tier tag, `fast_used`/`slow_used`, `fast_count`, the boundary walk,
	/// the migration push) is unchanged and still runs exactly once per
	/// demoted object. The reprieve path is untouched too, and cannot
	/// interact with this pass: `settle_fifo_queue` moves bytes from
	/// `fifo_used` to `slow_used` and never touches `fast_used`.
	fn settle_fast_tier(&mut self) {
		let effective = self.effective_main_fast_capacity();

		// Trigger only once usage is past the high watermark...
		if self.fast_used <= watermarks::high_bytes(effective) {
			return;
		}

		// ...but once triggered, drain all the way down to the low one.
		let drain_target = watermarks::low_bytes(effective);

		while self.fast_used > drain_target {
			let Some(demote_key) = self.main_boundary else { break };

			let size = self.entries.get(&demote_key).map(|entry| entry.size).unwrap_or(0) as CacheSize;
			let new_boundary = self.main_stack.before(&demote_key).copied();

			if let Some(entry) = self.entries.get_mut(&demote_key) {
				entry.tier = Some(Tier::Slow);
			}

			self.fast_used = self.fast_used.saturating_sub(size);
			self.fast_count = self.fast_count.saturating_sub(1);
			self.slow_used += size;
			self.main_boundary = new_boundary;

			self.migrations.push((demote_key, Tier::Slow));
		}
	}

	/// Relieves FIFO-queue pressure by **reprieving** its tail into the slow
	/// tier rather than evicting it: the aged-out key is spliced onto the
	/// back of `main_stack` (the LRU tail) tagged `Tier::Slow`, and stays in
	/// the cache.
	///
	/// This is the one behavioral difference from
	/// `TwoQFastAdmissionHybridStack`, where the same pressure is reported
	/// via `needs_capacity_eviction()` and drained by `apply_evictions`
	/// removing the key outright. See the module doc for why a reprieve must
	/// run synchronously here instead, and why the back (rather than the
	/// front of the slow segment) is the right landing spot.
	///
	/// Safe to do inside the stack precisely because nothing is removed: the
	/// key stays in `entries` and in the shared object map, moving only
	/// between two of this stack's own lists. The `(key, Tier::Slow)`
	/// migration it records is a real DRAM->PMEM copy that `PolicyWorker`
	/// applies, exactly as for an ordinary demotion.
	///
	/// The budget enforced is [`Self::effective_fifo_capacity`] — the
	/// configured `fifo_capacity` minus this segment's share of the shared
	/// per-object DRAM metadata reservation — not the raw `fifo_capacity`.
	/// The one-access queue is DRAM in this design, so it has to give up its
	/// proportional share of that reservation just as the main fast segment
	/// does; leaving it at the raw figure would let total DRAM overrun
	/// `fast_capacity` by the one-access share.
	///
	/// Hoisted out of the loop deliberately: a reprieve moves a key between
	/// two of this stack's own lists and removes nothing, so `entries.len()`
	/// (and hence the reservation) is invariant across the drain. Only
	/// `fifo_used` moves, which is what makes the loop terminate.
	fn settle_fifo_queue(&mut self) {
		let effective = self.effective_fifo_capacity();

		while self.fifo_used > effective {
			let Some(key) = self.fifo_queue.pop_back() else { break };
			let Some(entry) = self.entries.get(&key).copied() else { continue };
			let size = entry.size as CacheSize;

			self.fifo_used = self.fifo_used.saturating_sub(size);

			// Back of the list: still behind every fast key, so the
			// "fast keys are a contiguous prefix" invariant `main_boundary`
			// depends on is preserved and the boundary needs no update.
			self.main_stack.push_back(key);

			if let Some(stored) = self.entries.get_mut(&key) {
				stored.queue = Queue::Main;
				stored.tier = Some(Tier::Slow);
			}

			self.slow_used += size;
			self.main_count += 1;

			self.migrations.push((key, Tier::Slow));
		}
	}

	/// Pops and fully removes `fifo_queue`'s tail from this stack's own
	/// bookkeeping. Unlike `TwoQFastAdmissionHybridStack`, this is **not**
	/// the normal path for an aged-out one-access key -- that is
	/// `settle_fifo_queue` above, which reprieves rather than evicts. This
	/// exists only as `evict_one`'s last resort when the main queue is
	/// entirely empty; see there.
	fn evict_fifo_tail(&mut self) -> Option<HashedKey> {
		let key = self.fifo_queue.pop_back()?;
		let size = self.entries.remove(&key).map(|entry| entry.size).unwrap_or(0) as CacheSize;

		self.fifo_used = self.fifo_used.saturating_sub(size);

		Some(key)
	}
}

impl PolicyStack for TwoQFastAdmissionReprieveHybridStack {
	fn is_policy(&self, policy: &PaperPolicy) -> bool {
		matches!(policy, PaperPolicy::TwoQFastAdmissionReprieveHybrid(k_in) if *k_in == self.k_in)
	}

	fn len(&self) -> usize {
		self.entries.len()
	}

	fn contains(&self, key: HashedKey) -> bool {
		self.entries.contains_key(&key)
	}

	fn insert(&mut self, key: HashedKey, size: ObjectSize) {
		if self.entries.contains_key(&key) {
			// Existing key: track any size change, then treat as an access.
			// `touch` settles the fast tier on every path, so a FIFO-resident
			// key that grew is covered without an extra call here.
			self.resize_key(key, size);
			self.touch(key);
			return;
		}

		// Brand-new key: admitted into the FIFO queue, which is FAST here.
		// If this pushes fifo_used over fifo_capacity, `settle_fifo_queue`
		// below reprieves the queue's tail into the slow tier -- nothing is
		// evicted.
		self.fifo_queue.push_front(key);
		self.entries.insert(key, TwoQEntry { queue: Queue::Fifo, tier: None, size });
		self.fifo_used += size as CacheSize;

		// Relieve FIFO pressure immediately, by reprieving the tail into the
		// slow tier rather than reporting the pressure for `apply_evictions`
		// to remove -- the defining difference from
		// `TwoQFastAdmissionHybridStack`. See `settle_fifo_queue`.
		self.settle_fifo_queue();

		// Deliberately does NOT re-settle the *fast* tier. The reservation
		// carved out of `fast_capacity` is the fixed `fifo_capacity`, not the
		// live `fifo_used`, so the main queue's effective budget doesn't move
		// when the FIFO queue fills -- only `resize`/`resize_fast_tier` can
		// change it. (A reprieve moves bytes from `fifo_used` to `slow_used`,
		// leaving `fast_used` untouched, so it cannot create fast-tier
		// pressure either.)
	}

	fn update(&mut self, key: HashedKey) {
		if self.entries.contains_key(&key) {
			self.touch(key);
		}
	}

	fn remove(&mut self, key: HashedKey) {
		let Some(entry) = self.entries.remove(&key) else { return };
		let size = entry.size as CacheSize;

		match entry.queue {
			Queue::Fifo => {
				self.fifo_queue.remove(&key);
				self.fifo_used = self.fifo_used.saturating_sub(size);
			},

			Queue::Main => {
				let new_boundary_if_needed = if entry.tier == Some(Tier::Fast) && self.main_boundary == Some(key) {
					self.main_stack.before(&key).copied()
				} else {
					None
				};

				self.main_stack.remove(&key);
				self.main_count = self.main_count.saturating_sub(1);

				match entry.tier {
					Some(Tier::Fast) => {
						self.fast_used = self.fast_used.saturating_sub(size);
						self.fast_count = self.fast_count.saturating_sub(1);

						if self.main_boundary == Some(key) {
							self.main_boundary = new_boundary_if_needed;
						}
					},

					Some(Tier::Slow) => {
						self.slow_used = self.slow_used.saturating_sub(size);
					},

					None => {},
				}
			},
		}
	}

	fn resize(&mut self, max_size: CacheSize) {
		self.fifo_capacity = (self.k_in * max_size as f64) as CacheSize;

		// Re-settle, which `TwoQHybridStack::resize` has no need to do:
		// `fifo_capacity` is carved out of `fast_capacity` here, so growing
		// the cache grows the FIFO reservation and shrinks the main queue's
		// effective fast budget. Catch that now rather than at whatever
		// unrelated `insert`/`update` happens to come next.
		self.settle_fast_tier();

		// A shrink also lowers `fifo_capacity`, which may leave the FIFO
		// queue over budget; reprieve its tail down to fit rather than
		// reporting eviction pressure.
		self.settle_fifo_queue();
	}

	fn clear(&mut self) {
		self.fifo_queue.clear();
		self.main_stack.clear();
		self.entries.clear();

		self.fifo_used = 0;
		self.fast_used = 0;
		self.slow_used = 0;
		self.fast_count = 0;
		self.main_count = 0;
		self.main_boundary = None;
		self.migrations.clear();
	}

	/// Terminal eviction takes the main queue's LRU tail -- which, thanks to
	/// `settle_fifo_queue`, is where reprieved one-access keys land, so an
	/// object that never demonstrated reuse is still the first thing to go.
	///
	/// The FIFO queue is only touched as a last resort, when the main queue
	/// is completely empty. That fallback is not about eviction policy: it
	/// exists so this method never returns `None` while the stack still holds
	/// keys, because `apply_evictions` answers a `None` by evicting a
	/// **random** object (see `erase`'s doc). Reaching it requires every
	/// tracked key to still be in the one-access queue while overall
	/// `max_size` is already exceeded.
	fn evict_one(&mut self) -> Option<HashedKey> {
		if self.main_stack.len() == 0 {
			return self.evict_fifo_tail();
		}

		let key = self.main_stack.pop_back()?;
		let removed = self.entries.remove(&key);
		let size = removed.map(|entry| entry.size).unwrap_or(0) as CacheSize;
		let tier = removed.and_then(|entry| entry.tier);

		self.main_count = self.main_count.saturating_sub(1);

		match tier {
			Some(Tier::Fast) => {
				self.fast_used = self.fast_used.saturating_sub(size);
				self.fast_count = self.fast_count.saturating_sub(1);

				// The tail of main_stack can only be Fast-tagged if every
				// tracked Main key is still Fast (no demotion has ever
				// happened), in which case the boundary must have equaled
				// this key too. The new tail, if any, is then still Fast.
				if self.main_boundary == Some(key) {
					self.main_boundary = self.main_stack.back().copied();
				}
			},

			Some(Tier::Slow) => {
				self.slow_used = self.slow_used.saturating_sub(size);
			},

			None => {},
		}

		Some(key)
	}

	fn resize_fast_tier(&mut self, size: CacheSize) {
		self.fast_capacity = size;
		self.settle_fast_tier();

		// Also re-settle the one-access queue, which `TwoQFastAdmissionHybrid`
		// has no need to do: `reserved_shares` proportions the shared-metadata
		// reservation against `fast_capacity`, so changing it changes the
		// one-access queue's *effective* budget too (shrinking the fast tier
		// below `fifo_capacity` pushes the entire reservation onto that
		// queue). Catch that here rather than at whatever unrelated `insert`
		// happens to come next. A no-op while `shared_overhead` is 0, which is
		// why every pre-existing test is unaffected: `effective_fifo_capacity`
		// is then exactly `fifo_capacity`.
		self.settle_fifo_queue();
	}

	fn drain_tier_migrations(&mut self) -> Vec<(HashedKey, Tier)> {
		std::mem::take(&mut self.migrations)
	}

	/// Both DRAM-resident structures, summed: the FIFO queue plus the main
	/// queue's fast segment. The mirror image of `TwoQHybridStack`, where
	/// `fifo_used` counts toward the *slow* total instead.
	fn fast_bytes_used(&self) -> CacheSize {
		self.fifo_used + self.fast_used
	}

	fn slow_bytes_used(&self) -> CacheSize {
		self.slow_used
	}

	fn fast_object_count(&self) -> usize {
		self.fifo_queue.len() + self.fast_count
	}

	fn slow_object_count(&self) -> usize {
		self.main_count - self.fast_count
	}
}


#[cfg(test)]
mod tests {
	use super::*;

	fn drain(stack: &mut TwoQFastAdmissionReprieveHybridStack) -> Vec<(HashedKey, Tier)> {
		stack.drain_tier_migrations()
	}

	/// One `settle_fast_tier` pass over `count` equal-sized main-queue fast
	/// keys: how many are left tagged `Fast` afterwards.
	///
	/// The expectations below are derived from this rather than hard-coded so
	/// that they hold at any configured `FAST_TIER_HIGH_WATERMARK` /
	/// `FAST_TIER_LOW_WATERMARK` pair. They cannot simply pin the ratios: the
	/// watermarks are process-global `OnceLock`s read once per process, so a
	/// test that set them would race every other test in the binary.
	fn settled(effective: CacheSize, size: CacheSize, count: usize) -> usize {
		let high = watermarks::high_bytes(effective);
		let low = watermarks::low_bytes(effective);

		let mut fast = count;

		if fast as CacheSize * size > high {
			while fast > 0 && fast as CacheSize * size > low {
				fast -= 1;
			}
		}

		fast
	}

	/// Unchanged from `TwoQFastAdmissionHybridStack`: admission is still a
	/// plain DRAM write into the one-access queue.
	#[test]
	fn admission_still_lands_in_the_fifo_queue_in_the_fast_tier() {
		let mut stack = TwoQFastAdmissionReprieveHybridStack::new(0.5, 1_000, 1_000);

		stack.insert(1, 10);

		assert_eq!(stack.tier_of(1), Some(Tier::Fast));
		assert!(drain(&mut stack).is_empty());
	}

	/// The defining behavior: an aged-out one-access key survives, in slow.
	#[test]
	fn an_aged_out_one_access_key_is_reprieved_not_evicted() {
		// fifo_capacity = 0.1 * 1_000 = 100, so a third 50-byte key pushes
		// the queue over and the tail must be reprieved.
		let mut stack = TwoQFastAdmissionReprieveHybridStack::new(0.1, 1_000, 1_000);

		stack.insert(1, 50);
		stack.insert(2, 50);
		assert!(drain(&mut stack).is_empty());

		stack.insert(3, 50);

		// Key 1 was the FIFO tail. It is still tracked -- not removed --
		// and now lives in the slow tier.
		assert_eq!(stack.len(), 3);
		assert_eq!(stack.tier_of(1), Some(Tier::Slow));
		assert_eq!(stack.tier_of(2), Some(Tier::Fast));
		assert_eq!(stack.tier_of(3), Some(Tier::Fast));

		// And it produced a real DRAM->PMEM migration.
		assert_eq!(drain(&mut stack), vec![(1, Tier::Slow)]);
	}

	#[test]
	fn a_reprieve_moves_bytes_from_the_fast_gauges_to_the_slow_gauges() {
		let mut stack = TwoQFastAdmissionReprieveHybridStack::new(0.1, 1_000, 1_000);

		stack.insert(1, 50);
		stack.insert(2, 50);

		assert_eq!(stack.fast_bytes_used(), 100);
		assert_eq!(stack.slow_bytes_used(), 0);

		stack.insert(3, 50);

		assert_eq!(stack.fast_bytes_used(), 100);
		assert_eq!(stack.slow_bytes_used(), 50);
		assert_eq!(stack.fast_object_count(), 2);
		assert_eq!(stack.slow_object_count(), 1);
	}

	/// A reprieved key lands at the LRU tail, so it is the next thing
	/// evicted -- the placement decision this variant makes.
	#[test]
	fn a_reprieved_key_lands_at_the_bottom_and_is_evicted_first() {
		let mut stack = TwoQFastAdmissionReprieveHybridStack::new(0.1, 1_000, 1_000);

		// Key 1 proves itself and moves into the main queue as Fast.
		stack.insert(1, 50);
		stack.update(1);

		// Keys 2-4 stay unproven. fifo_capacity is 100, and the trigger is
		// strictly `fifo_used > fifo_capacity`, so three 50-byte keys are
		// needed to push it over -- key 1 left the queue when it was
		// promoted, so it no longer counts toward `fifo_used`.
		stack.insert(2, 50);
		stack.insert(3, 50);
		stack.insert(4, 50);

		drain(&mut stack);

		assert_eq!(stack.tier_of(2), Some(Tier::Slow));

		// The reprieved key goes before the proven one, despite being
		// added to the main queue later.
		assert_eq!(stack.evict_one(), Some(2));
		assert_eq!(stack.evict_one(), Some(1));
	}

	/// Pushing a Slow key onto the back must not disturb `main_boundary`,
	/// which the fast/slow prefix invariant depends on.
	#[test]
	fn a_reprieve_does_not_corrupt_the_fast_slow_boundary() {
		let mut stack = TwoQFastAdmissionReprieveHybridStack::new(0.1, 1_000, 1_000);

		// Two proven keys in the main queue, both Fast.
		stack.insert(1, 50);
		stack.update(1);
		stack.insert(2, 50);
		stack.update(2);
		drain(&mut stack);

		assert_eq!(stack.main_boundary, Some(1));

		// A reprieve appends a Slow key behind them.
		stack.insert(3, 50);
		stack.insert(4, 50);
		stack.insert(5, 50);
		drain(&mut stack);

		// Boundary is untouched, and both proven keys are still Fast.
		assert_eq!(stack.main_boundary, Some(1));
		assert_eq!(stack.tier_of(1), Some(Tier::Fast));
		assert_eq!(stack.tier_of(2), Some(Tier::Fast));

		// Fast-tier demotion still works correctly afterwards. Shrinking the
		// main queue's budget to one key's worth puts `fast_used` past the
		// high watermark; how far the resulting pass then drains depends on
		// the configured watermarks, so the expectation is derived rather
		// than hard-coded (see `settled`).
		stack.resize_fast_tier(50 + stack.fifo_capacity);

		let effective = stack.effective_main_fast_capacity();
		let left_fast = settled(effective, 50, 2);
		let demoted = 2 - left_fast;
		let migrations = drain(&mut stack);

		assert!(demoted >= 1, "shrinking to one key's worth must trigger a pass");

		// Demotions still come off the LRU end, in order, one entry each.
		assert_eq!(
			migrations,
			(1..=demoted as HashedKey).map(|key| (key, Tier::Slow)).collect::<Vec<_>>(),
		);
		assert_eq!(stack.main_boundary, if left_fast > 0 { Some(2) } else { None });
	}

	/// A reprieved key is a full main-queue citizen: re-accessing it
	/// promotes it back to fast like any other slow key.
	#[test]
	fn a_reprieved_key_can_still_be_promoted_by_a_later_access() {
		let mut stack = TwoQFastAdmissionReprieveHybridStack::new(0.1, 1_000, 1_000);

		stack.insert(1, 50);
		stack.insert(2, 50);
		stack.insert(3, 50);
		drain(&mut stack);

		assert_eq!(stack.tier_of(1), Some(Tier::Slow));

		stack.update(1);

		assert_eq!(stack.tier_of(1), Some(Tier::Fast));
		assert_eq!(drain(&mut stack), vec![(1, Tier::Fast)]);
	}

	/// The reprieve runs inside the stack, so the stack must never report
	/// eviction pressure -- see the module doc for why routing it through
	/// `evict_one()` was a real bug in the s3-fifo equivalent.
	#[test]
	fn capacity_pressure_is_never_reported_for_eviction() {
		let mut stack = TwoQFastAdmissionReprieveHybridStack::new(0.1, 1_000, 1_000);

		for key in 1..=10 {
			stack.insert(key, 50);
			assert!(
				!stack.needs_capacity_eviction(),
				"FIFO pressure must be relieved by reprieve, never reported as eviction",
			);
		}

		// Nothing was lost along the way.
		assert_eq!(stack.len(), 10);
	}

	/// Shrinking `max_size` shrinks `fifo_capacity`, which must reprieve
	/// rather than leave the queue over budget.
	#[test]
	fn shrinking_max_size_reprieves_down_to_the_new_budget() {
		let mut stack = TwoQFastAdmissionReprieveHybridStack::new(0.5, 1_000, 1_000);

		// fifo_capacity 500: four 100-byte keys fit comfortably.
		for key in 1..=4 {
			stack.insert(key, 100);
		}

		assert!(drain(&mut stack).is_empty());
		assert_eq!(stack.fifo_used, 400);

		// max_size 400 => fifo_capacity 200, so two keys must be reprieved.
		stack.resize(400);

		assert_eq!(stack.fifo_used, 200);
		assert_eq!(drain(&mut stack), vec![(1, Tier::Slow), (2, Tier::Slow)]);
		assert_eq!(stack.len(), 4);
	}

	/// The last-resort fallback: with nothing in the main queue, `evict_one`
	/// must still return a key rather than `None`, which `apply_evictions`
	/// would answer with a random eviction.
	#[test]
	fn eviction_falls_back_to_the_fifo_queue_when_the_main_queue_is_empty() {
		let mut stack = TwoQFastAdmissionReprieveHybridStack::new(1.0, 1_000, 1_000);

		// k_in 1.0 keeps the FIFO queue within budget, so nothing is ever
		// reprieved and the main queue stays empty.
		stack.insert(1, 50);
		stack.insert(2, 50);

		assert_eq!(stack.slow_object_count(), 0);

		assert_eq!(stack.evict_one(), Some(1));
		assert_eq!(stack.evict_one(), Some(2));
		assert_eq!(stack.evict_one(), None);
		assert_eq!(stack.len(), 0);
	}

	#[test]
	fn removing_a_reprieved_key_releases_its_slow_bytes() {
		let mut stack = TwoQFastAdmissionReprieveHybridStack::new(0.1, 1_000, 1_000);

		stack.insert(1, 50);
		stack.insert(2, 50);
		stack.insert(3, 50);
		drain(&mut stack);

		assert_eq!(stack.slow_bytes_used(), 50);

		stack.remove(1);

		assert_eq!(stack.slow_bytes_used(), 0);
		assert_eq!(stack.slow_object_count(), 0);
		assert_eq!(stack.len(), 2);
	}

	#[test]
	fn clear_resets_every_counter() {
		let mut stack = TwoQFastAdmissionReprieveHybridStack::new(0.1, 1_000, 1_000);

		for key in 1..=5 {
			stack.insert(key, 50);
		}

		stack.clear();

		assert_eq!(stack.len(), 0);
		assert_eq!(stack.fast_bytes_used(), 0);
		assert_eq!(stack.slow_bytes_used(), 0);
		assert_eq!(stack.fast_object_count(), 0);
		assert_eq!(stack.slow_object_count(), 0);
		assert!(drain(&mut stack).is_empty());
	}

	#[test]
	fn is_policy_matches_only_its_own_variant_and_k_in() {
		let stack = TwoQFastAdmissionReprieveHybridStack::new(0.1, 1_000, 300);

		assert!(stack.is_policy(&PaperPolicy::TwoQFastAdmissionReprieveHybrid(0.1)));
		assert!(!stack.is_policy(&PaperPolicy::TwoQFastAdmissionReprieveHybrid(0.2)));
		assert!(!stack.is_policy(&PaperPolicy::TwoQFastAdmissionHybrid(0.1)));
		assert!(!stack.is_policy(&PaperPolicy::TwoQHybrid(0.1)));
	}

	/// Watermark test rig: a 10_000-byte one-access reservation carved out of
	/// a 20_000-byte fast tier leaves the main queue a round 10_000, so
	/// `high_bytes` / `low_bytes` land on exact byte counts whatever the
	/// ratios are.
	///
	/// The reservation is deliberately as large as the main queue's budget,
	/// which `TwoQFastAdmissionHybridStack`'s equivalent rig has no need of: a
	/// key here reaches the main queue by passing *through* the one-access
	/// queue (`insert` then `update`), and `settle_fifo_queue` would reprieve
	/// it straight into the slow tier if it did not fit in that queue first.
	/// The largest key any test below promotes is the high watermark itself,
	/// so a reservation equal to the whole main-queue budget always holds it.
	fn watermark_stack() -> TwoQFastAdmissionReprieveHybridStack {
		let stack = TwoQFastAdmissionReprieveHybridStack::new(0.1, 100_000, 20_000);

		assert_eq!(stack.fifo_capacity, 10_000);
		assert_eq!(stack.effective_main_fast_capacity(), 10_000);

		stack
	}

	/// Sitting just under the high watermark triggers nothing at all -- the
	/// old rule only left the tier alone below the ceiling itself.
	#[test]
	fn usage_just_below_the_high_watermark_triggers_no_demotion() {
		let mut stack = watermark_stack();
		let effective = stack.effective_main_fast_capacity();
		let high = watermarks::high_bytes(effective);

		assert!(high > 1, "watermark config leaves no room for this test");

		// A single key sized one byte under the high watermark, promoted out
		// of the one-access queue and into the main queue.
		stack.insert(1, (high - 1) as ObjectSize);
		stack.update(1);

		assert_eq!(stack.fast_used, high - 1);
		assert_eq!(stack.tier_of(1), Some(Tier::Fast));
		assert!(drain(&mut stack).is_empty());
		assert_eq!(stack.slow_used, 0);
		assert_eq!(stack.slow_object_count(), 0);
	}

	/// Sitting exactly *on* the high watermark still triggers nothing (the
	/// test is `>`, not `>=`); one byte past it does.
	#[test]
	fn usage_above_the_high_watermark_triggers_a_pass() {
		let mut stack = watermark_stack();
		let effective = stack.effective_main_fast_capacity();
		let high = watermarks::high_bytes(effective);
		let low = watermarks::low_bytes(effective);

		assert!(high >= 1 && low >= 1, "watermark config leaves no room for this test");

		stack.insert(1, high as ObjectSize);
		stack.update(1);

		assert_eq!(stack.fast_used, high);
		assert!(drain(&mut stack).is_empty(), "exactly on the high watermark must not trigger");

		// One more byte tips it over.
		stack.insert(2, 1);
		stack.update(2);

		let migrations = drain(&mut stack);

		// Key 1 is the LRU-most fast key, so it is the one that goes. A
		// FIFO->main promotion emits nothing of its own here, so the pass's
		// demotions are the only entries in the batch.
		assert_eq!(migrations, vec![(1, Tier::Slow)]);
		assert_eq!(stack.tier_of(1), Some(Tier::Slow));
		assert_eq!(stack.tier_of(2), Some(Tier::Fast));
		assert!(stack.fast_used <= low);
	}

	/// The point of the change: a triggered pass drains to the LOW watermark,
	/// not merely back under the ceiling it used to settle at.
	#[test]
	fn a_triggered_pass_drains_to_the_low_watermark() {
		let mut stack = watermark_stack();
		let effective = stack.effective_main_fast_capacity();
		let high = watermarks::high_bytes(effective);
		let low = watermarks::low_bytes(effective);
		let size: CacheSize = 100;

		// Fill with the largest whole number of 100-byte keys that still sits
		// at or below the high watermark: no pass has run yet.
		let filled = high / size;

		assert!(filled >= 1, "watermark config leaves no room for this test");

		for key in 1..=filled {
			stack.insert(key, size as ObjectSize);
			stack.update(key);
		}

		assert_eq!(stack.fast_used, filled * size);
		assert!(drain(&mut stack).is_empty());

		// One more key tips it past the high watermark.
		stack.insert(filled + 1, size as ObjectSize);
		stack.update(filled + 1);

		let migrations = drain(&mut stack);

		assert!(!migrations.is_empty(), "crossing the high watermark must trigger a pass");

		// Drained past the ceiling, all the way down to the low watermark...
		assert!(stack.fast_used <= low);
		assert_eq!(stack.fast_used, low / size * size);
		assert!(stack.fast_used < effective || low >= effective);

		// ...and stopped as soon as it got under, rather than emptying the
		// segment: one fewer demotion would have left it above the target.
		assert!(stack.fast_used + size > low);

		// One `Tier::Slow` entry per demoted object, taken off the LRU end in
		// promotion order -- the single larger batch this change buys.
		let demoted = (filled + 1) - stack.fast_count as CacheSize;

		assert_eq!(migrations.len() as CacheSize, demoted);
		assert_eq!(migrations, (1..=demoted).map(|key| (key, Tier::Slow)).collect::<Vec<_>>());

		// Nothing was reprieved on the way: every key passed straight through
		// the one-access queue, so the whole batch came from the pass.
		assert_eq!(stack.fifo_used, 0);
		assert_eq!(stack.main_count as CacheSize, filled + 1);
	}

	/// Every byte and every object is still accounted for exactly once after a
	/// multi-object drain: the per-demotion bookkeeping did not change, only
	/// how many times it runs per pass.
	#[test]
	fn counters_stay_consistent_after_a_watermark_pass() {
		let mut stack = watermark_stack();
		let effective = stack.effective_main_fast_capacity();
		let size: CacheSize = 100;
		let promoted = watermarks::high_bytes(effective) / size + 1;

		assert!(promoted >= 2, "watermark config leaves no room for this test");

		for key in 1..=promoted {
			stack.insert(key, size as ObjectSize);
			stack.update(key);
		}

		// A brand-new key left sitting in the one-access queue, to pin that
		// the pass touched neither it nor the FIFO accounting.
		stack.insert(promoted + 1, 40);

		let migrations = drain(&mut stack);
		let demoted = migrations.len() as CacheSize;

		assert!(demoted >= 1, "the pass must have run");
		assert!(migrations.iter().all(|(_, tier)| *tier == Tier::Slow));

		// Bytes: every promoted key is either main-fast or main-slow, and the
		// one-access key is neither.
		assert_eq!(stack.fast_used, (promoted - demoted) * size);
		assert_eq!(stack.slow_used, demoted * size);
		assert_eq!(stack.fifo_used, 40);
		assert_eq!(stack.fast_used + stack.slow_used, promoted * size);

		// Counts: `fast_count` covers main-fast only, `main_count` every
		// promoted key, and the trait-level views add the one-access key back
		// onto the fast side.
		assert_eq!(stack.fast_count as CacheSize, promoted - demoted);
		assert_eq!(stack.main_count as CacheSize, promoted);
		assert_eq!(stack.fast_object_count() as CacheSize, promoted - demoted + 1);
		assert_eq!(stack.slow_object_count() as CacheSize, demoted);
		assert_eq!(stack.fast_bytes_used(), stack.fifo_used + stack.fast_used);
		assert_eq!(stack.slow_bytes_used(), stack.slow_used);

		// Nothing left the cache, the one-access key is still Fast, and the
		// boundary still tracks whether any fast key is left in the main
		// queue.
		assert_eq!(stack.len() as CacheSize, promoted + 1);
		assert_eq!(stack.tier_of(promoted + 1), Some(Tier::Fast));
		assert_eq!(stack.main_boundary.is_some(), stack.fast_count > 0);
	}

	// ---------------------------------------------------------------------
	// Shared per-object DRAM metadata reservation (`with_shared_overhead`).
	//
	// Every pre-existing test above constructs the stack WITHOUT
	// `with_shared_overhead`, so it sees `shared_overhead == 0`: the
	// reservation is 0, both shares are 0, `effective_fifo_capacity ==
	// fifo_capacity` and `effective_main_fast_capacity == fast_capacity -
	// fifo_capacity` exactly as before. Their capacities did not need
	// rescaling, and their assertions are untouched.
	// ---------------------------------------------------------------------

	/// The reservation is charged per *tracked* key (both tiers) and divided
	/// between the two DRAM segments rather than charged in full to each.
	#[test]
	fn shared_overhead_shrinks_both_effective_budgets_without_double_charging() {
		// A 20_000-byte fast tier whose one-access reservation is 10_000
		// (k_in 0.1 x max_size 100_000), so the two DRAM segments are equal
		// halves of the budget and every reservation splits exactly in two.
		let mut stack = TwoQFastAdmissionReprieveHybridStack::new(0.1, 100_000, 20_000)
			.with_shared_overhead(64);

		assert_eq!(stack.fifo_capacity, 10_000);

		// Nothing tracked yet, so nothing reserved.
		assert_eq!(stack.reserved_overhead(), 0);
		assert_eq!(stack.reserved_shares(), (0, 0));
		assert_eq!(stack.effective_fifo_capacity(), 10_000);
		assert_eq!(stack.effective_main_fast_capacity(), 10_000);

		// Ten tracked keys reserve 10 x 64 = 640 bytes, split 320 / 320.
		for key in 1..=10 {
			stack.insert(key, 10);
		}

		assert_eq!(stack.len(), 10);
		assert_eq!(stack.reserved_overhead(), 640);
		assert_eq!(stack.reserved_shares(), (320, 320));
		assert_eq!(stack.effective_fifo_capacity(), 10_000 - 320);
		assert_eq!(stack.effective_main_fast_capacity(), 10_000 - 320);

		// Charged once, not once per segment: the shares sum to the whole
		// reservation, and budget + reservation still fits `fast_capacity`.
		let (fifo_share, main_share) = stack.reserved_shares();

		assert_eq!(fifo_share + main_share, stack.reserved_overhead());
		assert_eq!(
			stack.effective_fifo_capacity()
				+ stack.effective_main_fast_capacity()
				+ stack.reserved_overhead(),
			stack.fast_capacity,
		);

		// The one-access reservation really is enforced against the effective
		// figure. Shrinking `max_size` to 1_000 drops `fifo_capacity` to 100,
		// which the 10 keys' 100 bytes would sit exactly on -- the 3-byte
		// one-access share (640 x 100 / 20_000) is the whole difference, and
		// it is enough to reprieve the tail into the slow tier.
		stack.resize(1_000);

		assert_eq!(stack.fifo_capacity, 100);
		assert_eq!(stack.reserved_shares().0, 3);
		assert_eq!(stack.effective_fifo_capacity(), 97);
		assert_eq!(stack.fifo_used, 90);
		assert_eq!(stack.slow_object_count(), 1);
		assert_eq!(drain(&mut stack), vec![(1, Tier::Slow)]);

		// A slow-tier key is still charged: nothing left the cache, so the
		// reservation is exactly where it was.
		assert_eq!(stack.len(), 10);
		assert_eq!(stack.reserved_overhead(), 640);

		// The identical shrink on an unreserved stack reprieves nothing.
		let mut plain = TwoQFastAdmissionReprieveHybridStack::new(0.1, 100_000, 20_000);

		for key in 1..=10 {
			plain.insert(key, 10);
		}

		plain.resize(1_000);

		assert_eq!(plain.effective_fifo_capacity(), 100);
		assert_eq!(plain.fifo_used, 100);
		assert_eq!(plain.slow_object_count(), 0);
		assert!(drain(&mut plain).is_empty());
	}

	/// The point of the reservation: a workload that fits comfortably in a
	/// stack without one demotes out of the same stack with one, because the
	/// reservation shrinks the budget the watermarks are then applied to.
	#[test]
	fn shared_overhead_reserves_dram_and_demotes_earlier() {
		const SIZE: CacheSize = 100;
		const OVERHEAD: CacheSize = 100;
		const RAW_MAIN: CacheSize = 10_000;

		// The largest whole number of `SIZE`-byte keys that still sits at or
		// below the high watermark of the raw main-fast segment. Derived from
		// `watermarks` rather than hard-coded, like every other watermark test
		// in this file, so it holds at any configured ratio pair.
		let count = watermarks::high_bytes(RAW_MAIN) / SIZE;

		assert!(count >= 2, "watermark config leaves no room for this test");

		// Without a reservation those keys sit at or below the high watermark
		// (the trigger is strict `>`), so nothing is ever demoted.
		let mut plain = TwoQFastAdmissionReprieveHybridStack::new(0.1, 100_000, 20_000);

		for key in 1..=count {
			plain.insert(key, SIZE as ObjectSize);
			plain.update(key);
		}

		assert_eq!(plain.effective_main_fast_capacity(), RAW_MAIN);
		assert_eq!(plain.fast_used, count * SIZE);
		assert!(drain(&mut plain).is_empty());
		assert_eq!(plain.slow_object_count(), 0);

		// The identical workload against a stack that reserves 100 bytes per
		// tracked key. Half of the total lands on this segment; the other half
		// is reserved against the one-access queue (which stays far inside its
		// own effective budget here -- each key is promoted straight out of it).
		let mut stack = TwoQFastAdmissionReprieveHybridStack::new(0.1, 100_000, 20_000)
			.with_shared_overhead(OVERHEAD);

		for key in 1..=count {
			stack.insert(key, SIZE as ObjectSize);
			stack.update(key);
		}

		let effective = stack.effective_main_fast_capacity();

		assert_eq!(stack.reserved_overhead(), count * OVERHEAD);
		assert_eq!(stack.reserved_shares(), (count * OVERHEAD / 2, count * OVERHEAD / 2));
		assert_eq!(effective, RAW_MAIN - count * OVERHEAD / 2);
		assert!(effective < plain.effective_main_fast_capacity());

		// The same bytes that sat exactly on the plain stack's high watermark
		// are past the reserved stack's, so a pass must have run.
		assert!(
			count * SIZE > watermarks::high_bytes(effective),
			"watermark config leaves no room for this test",
		);

		let migrations = drain(&mut stack);
		let demoted = migrations.len() as CacheSize;

		assert!(demoted >= 1, "the reservation must demote what the plain stack kept");

		// Demotions still come off the LRU end, in promotion order, one entry
		// each -- the reservation changes when a pass runs, not what it does.
		assert_eq!(migrations, (1..=demoted).map(|key| (key, Tier::Slow)).collect::<Vec<_>>());

		// Demotion, never eviction: every key is still tracked, every byte is
		// still on one side or the other, and nothing was reprieved on the way
		// (each key passed straight through the one-access queue).
		assert_eq!(stack.len() as CacheSize, count);
		assert!(!stack.needs_capacity_eviction());
		assert_eq!(stack.fifo_used, 0);
		assert_eq!(stack.fast_used, (count - demoted) * SIZE);
		assert_eq!(stack.fast_used + stack.slow_used, count * SIZE);
		assert_eq!(stack.slow_object_count() as CacheSize, demoted);
		assert!(stack.fast_used <= watermarks::high_bytes(effective));
	}

	/// A reservation larger than the whole fast tier drives BOTH effective
	/// budgets to 0. `high_bytes(0)` and `low_bytes(0)` are 0 at every
	/// watermark ratio, so this holds for any configuration.
	#[test]
	fn a_reservation_larger_than_the_fast_tier_demotes_everything_but_evicts_nothing() {
		// One tracked key reserves 30_000 against a 20_000-byte fast tier:
		// the one-access share is 30_000 x 10_000 / 20_000 = 15_000 and the
		// main share is the other 15_000, so both segments saturate to 0.
		let mut stack = TwoQFastAdmissionReprieveHybridStack::new(0.1, 100_000, 20_000)
			.with_shared_overhead(30_000);

		stack.insert(1, 10);

		assert_eq!(stack.reserved_overhead(), 30_000);
		assert_eq!(stack.reserved_shares(), (15_000, 15_000));
		assert_eq!(stack.effective_fifo_capacity(), 0);
		assert_eq!(stack.effective_main_fast_capacity(), 0);

		// Admission is reprieved straight out of the one-access queue.
		assert_eq!(stack.tier_of(1), Some(Tier::Slow));
		assert_eq!(drain(&mut stack), vec![(1, Tier::Slow)]);
		assert_eq!(stack.fast_bytes_used(), 0);
		assert_eq!(stack.slow_bytes_used(), 10);

		// Re-accessing it promotes it into the main queue and the settle pass
		// demotes it straight back out, so the promotion emits no `Fast`
		// migration of its own.
		stack.update(1);

		assert_eq!(drain(&mut stack), vec![(1, Tier::Slow)]);
		assert_eq!(stack.tier_of(1), Some(Tier::Slow));
		assert_eq!(stack.fast_bytes_used(), 0);

		// Demotion is the only response -- the DRAM budget never evicts.
		assert_eq!(stack.len(), 1);
		assert!(!stack.needs_capacity_eviction());
	}

	/// Rig for the two tests below: `count` equal-sized keys promoted into the
	/// main queue under a deliberately roomy fast budget, so the fill itself
	/// triggers nothing and the later `resize_fast_tier` produces exactly one
	/// pass that [`settled`] predicts exactly (a demotion removes nothing, so
	/// `entries.len()` -- and hence the reservation -- is constant across it).
	fn reserved_stack(
		overhead: CacheSize,
		count: CacheSize,
		size: ObjectSize,
	) -> TwoQFastAdmissionReprieveHybridStack {
		let mut stack = TwoQFastAdmissionReprieveHybridStack::new(0.1, 100_000, 10_000_000)
			.with_shared_overhead(overhead);

		for key in 1..=count {
			stack.insert(key, size);
			stack.update(key);
		}

		assert_eq!(stack.fifo_capacity, 10_000);
		assert_eq!(stack.fifo_used, 0);
		assert_eq!(stack.fast_used, count * size as CacheSize);
		assert!(
			stack.drain_tier_migrations().is_empty(),
			"the roomy setup budget must not itself trigger a pass",
		);

		stack
	}

	/// Composition: a triggered pass drains to `low_bytes(capacity -
	/// reservation)`, not `low_bytes(capacity)`.
	#[test]
	fn the_reservation_and_the_watermarks_compose_on_the_drain_target() {
		const SIZE: ObjectSize = 500;
		const COUNT: CacheSize = 20;

		let mut stack = reserved_stack(400, COUNT, SIZE);

		assert_eq!(stack.reserved_overhead(), 8_000);

		// Shrink the fast tier to 20_000. The one-access reservation is
		// 10_000 of it, so the two DRAM segments are equal halves and the
		// 8_000-byte metadata reservation splits 4_000 / 4_000: the main
		// segment's raw 10_000 becomes an effective 6_000.
		stack.resize_fast_tier(20_000);

		let raw_main = stack.fast_capacity - stack.fifo_capacity;
		let effective = stack.effective_main_fast_capacity();

		assert_eq!(stack.reserved_shares(), (4_000, 4_000));
		assert_eq!(raw_main, 10_000);
		assert_eq!(effective, 6_000);

		let target = watermarks::low_bytes(effective);
		let unreserved_target = watermarks::low_bytes(raw_main);

		assert!(
			target < unreserved_target,
			"watermark config leaves no room for this test",
		);

		let left_fast = settled(effective, SIZE as CacheSize, COUNT as usize);
		let demoted = COUNT - left_fast as CacheSize;
		let migrations = drain(&mut stack);

		// 20 x 500 = 10_000 bytes against an effective 6_000 is past the high
		// watermark at every ratio, so the pass always runs.
		assert!(demoted >= 1, "shrinking to 20_000 must trigger a pass");
		assert_eq!(migrations, (1..=demoted).map(|key| (key, Tier::Slow)).collect::<Vec<_>>());
		assert_eq!(stack.fast_used, left_fast as CacheSize * SIZE as CacheSize);

		// It drained to the LOW watermark of the RESERVED budget...
		assert!(stack.fast_used <= target);

		// ...and stopped as soon as it got under that target rather than
		// continuing, so the target it used was `target`, not
		// `unreserved_target` -- which it sailed straight past on the way
		// down. One fewer demotion would have left it above `target`.
		assert!(stack.fast_used + SIZE as CacheSize > target);
		assert!(stack.fast_used < unreserved_target);
	}

	/// Every byte and every object is still accounted for exactly once after a
	/// pass triggered by the reservation, including a one-access key that the
	/// pass must not touch.
	#[test]
	fn counters_stay_consistent_after_a_reserved_pass() {
		const SIZE: ObjectSize = 500;
		const COUNT: CacheSize = 20;

		let mut stack = reserved_stack(400, COUNT, SIZE);

		// A brand-new key left sitting in the one-access queue, to pin that
		// the pass touched neither it nor the FIFO accounting. It raises the
		// tracked count to 21, so the reservation becomes 8_400.
		stack.insert(COUNT + 1, 40);

		assert!(drain(&mut stack).is_empty());
		assert_eq!(stack.reserved_overhead(), 8_400);

		stack.resize_fast_tier(20_000);

		// 8_400 splits 4_200 / 4_200, so the main segment's effective budget
		// is 10_000 - 4_200 = 5_800 and the one-access queue's is 5_800 too
		// (far above its 40 bytes in flight, so nothing is reprieved).
		let effective = stack.effective_main_fast_capacity();

		assert_eq!(stack.reserved_shares(), (4_200, 4_200));
		assert_eq!(effective, 5_800);
		assert_eq!(stack.effective_fifo_capacity(), 5_800);

		let left_fast = settled(effective, SIZE as CacheSize, COUNT as usize);
		let demoted = COUNT - left_fast as CacheSize;
		let migrations = drain(&mut stack);

		assert!(demoted >= 1, "the pass must have run");
		assert_eq!(migrations.len() as CacheSize, demoted);
		assert!(migrations.iter().all(|(_, tier)| *tier == Tier::Slow));

		// Bytes: every promoted key is main-fast or main-slow, and the
		// one-access key is neither.
		assert_eq!(stack.fast_used, left_fast as CacheSize * SIZE as CacheSize);
		assert_eq!(stack.slow_used, demoted * SIZE as CacheSize);
		assert_eq!(stack.fifo_used, 40);
		assert_eq!(stack.fast_used + stack.slow_used, COUNT * SIZE as CacheSize);

		// Counts: `fast_count` covers main-fast only, `main_count` every
		// promoted key, and the trait-level views add the one-access key back
		// onto the fast side.
		assert_eq!(stack.fast_count as CacheSize, COUNT - demoted);
		assert_eq!(stack.main_count as CacheSize, COUNT);
		assert_eq!(stack.fast_object_count() as CacheSize, COUNT - demoted + 1);
		assert_eq!(stack.slow_object_count() as CacheSize, demoted);
		assert_eq!(stack.fast_bytes_used(), stack.fifo_used + stack.fast_used);
		assert_eq!(stack.slow_bytes_used(), stack.slow_used);

		// Nothing left the cache, the one-access key is still Fast, and the
		// boundary still tracks whether any fast key is left in the main
		// queue.
		assert_eq!(stack.len() as CacheSize, COUNT + 1);
		assert_eq!(stack.reserved_overhead(), (COUNT + 1) * 400);
		assert_eq!(stack.tier_of(COUNT + 1), Some(Tier::Fast));
		assert_eq!(stack.main_boundary.is_some(), stack.fast_count > 0);
		assert!(!stack.needs_capacity_eviction());
	}
}
