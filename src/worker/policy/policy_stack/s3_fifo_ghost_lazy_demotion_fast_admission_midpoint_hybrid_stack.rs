/*
 * Copyright (c) Kia Shakiba
 *
 * This source code is licensed under the GNU AGPLv3 license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! `S3FifoGhostLazyDemotionFastAdmissionMidpointHybridStack` —
//! `S3FifoGhostLazyDemotionFastAdmissionHybridStack` with one addition: a
//! checkpoint roughly halfway through the SLOW portion of the main queue
//! that gives a reaccessed object an early second chance, instead of
//! making it wait until it reaches the eviction tail. For
//! `PaperPolicy::S3FifoGhostLazyDemotionFastAdmissionMidpointHybrid`.
//!
//! Identical to `S3FifoGhostLazyDemotionFastAdmissionHybridStack` in every
//! other respect (the fast-tier one-access queue, the ghost queue
//! lifecycle, the demotion-time reference-bit reprieve, the "contiguous
//! front run" invariant) — see that stack's module doc, and the stacks
//! beneath it, for the full picture.
//!
//! ## The new mechanic
//!
//! The slow portion of the main queue was previously a passive holding
//! area: nothing looked at an object there until it either reached the
//! eviction tail (checked by `give_second_chance`) or was promoted via a
//! ghost hit. This variant adds one more checkpoint, positioned
//! approximately halfway between the fast/slow boundary and the tail: if
//! the object currently sitting there has its reference bit set (i.e. it
//! was reaccessed after being demoted), it's given the exact same
//! treatment as a tail-reached second chance -- moved to the front of the
//! fast segment via the existing `give_second_chance` -- instead of
//! having to survive all the way to the tail first. An object that's
//! genuinely cold (bit clear) at the midpoint is left alone; it keeps
//! aging normally and will still get its one real chance at the tail.
//!
//! The check runs once per `evict_one()` call, at the moment this stack
//! turns to the main queue for a real eviction -- either because the
//! one-access queue is empty, or because the main queue is full and the
//! one-access tail is therefore off limits (see the "Eviction order"
//! section) -- the same cadence `give_second_chance`'s own tail check
//! already runs at.
//!
//! ## Locating "the middle" without an O(n) scan
//!
//! The slow segment can hold hundreds of thousands of objects at the
//! scale this crate is benchmarked at, so recomputing its midpoint by
//! walking from the tail (or the boundary) on every check -- an O(slow
//! segment length) scan -- was rejected outright: called once per eviction
//! under steady-state pressure, that's O(n) per admission, i.e. O(n²) over
//! a cache's lifetime. `kwik::collections::HashList`'s `before()` (the
//! only directional-walk primitive `PmemHashList` also exposes, under
//! `eviction_stacks_pmem` -- `HashList::after()` exists but has no
//! `PmemHashList` counterpart, so a design relying on it would only work
//! for one of the two storage backends) makes a fresh full-segment walk
//! the only "exact" option available across both backends, which is
//! exactly the cost being avoided.
//!
//! Instead, `slow_midpoint: Option<HashedKey>` is a cursor maintained
//! incrementally, in O(1) amortized time, using only `before()`:
//!
//! * **Growth at the front** (a demotion always retags the object that was
//!   already sitting where the new `main_boundary` lands -- see
//!   `S3FifoHybridStack`'s module doc for why this never needs a real list
//!   insertion) and **shrinkage at the tail or from an arbitrary position**
//!   (a slow-tier eviction, or a promotion out of the slow segment via
//!   `give_second_chance`, including a promotion `check_slow_midpoint`
//!   itself triggers) both push the cursor's *tracked object* further from
//!   where the true middle currently is, at a rate of ~0.5 positions per
//!   event (worked out in full in the design notes below the trait impl).
//!   `bump_midpoint_drift()` accumulates this in a small counter and, once
//!   it reaches a full position's worth (every 2 qualifying events), moves
//!   the cursor one step toward the front via `nudge_midpoint_toward_front`
//!   -- the only direction ever needed, since both kinds of event drift the
//!   same way.
//! * **First demotion into an empty slow segment** initializes the cursor
//!   directly to the newly-demoted key (there's only one candidate).
//! * **The cursor's own target being removed or promoted** (explicit
//!   `remove()`, a slow-tier tail eviction, or a `give_second_chance`
//!   promotion) redirects it to the `before()` neighbor -- but only if
//!   that neighbor is *still Slow*; if it isn't (the cursor was one step
//!   from the boundary), the cursor is cleared instead of accidentally
//!   pointing into the fast segment. This redirect always runs *before*
//!   the key is physically unlinked, since `before()` needs it still
//!   linked to resolve its neighbor.
//!
//! This is a heuristic promotion trigger, not a correctness-critical exact
//! median -- "approximately halfway" is all the mechanic needs, and the
//! amortized correction keeps the cursor within a small, bounded distance
//! of the true middle without ever paying for a full rescan. See the
//! design notes after the `PolicyStack` impl for the arithmetic behind the
//! "every 2 events, one step" correction rate.
//!
//! ## Eviction order: the one-access tail goes first, but only while the
//! main queue has room
//!
//! `evict_one` drains the one-access queue's tail before it will touch the
//! main queue -- but that priority is CONDITIONAL, and now says so: it
//! applies only while the main queue is below its own byte budget. Once the
//! main queue is full, `evict_one` skips the one-access tail entirely and
//! goes straight to the main-queue eviction loop. This mirrors
//! `SThreeFifoStack::evict_one`, the crate's non-tiered S3-FIFO, exactly.
//! Previously the one-access tail was drained unconditionally -- an
//! unexamined divergence from the policy this stack is a tiering of, never
//! argued for anywhere, unlike `TwoQHybridStack`'s deliberately documented
//! eviction priority.
//!
//! The budget the gate reads is `main_capacity` -- `(1 - one_access_ratio) *
//! max_size`, the exact complement of `one_access_capacity`, computed beside
//! it in both `new` and `resize`. `is_main_full` compares it against
//! `fast_used + slow_used`, which is precisely the main queue's byte total
//! and nothing else: a one-access resident is tracked with `tier: None` and
//! moves only `one_access_used`, while `fast_used` / `slow_used` are touched
//! only by main-queue admission (`promote_from_one_access`,
//! `admit_via_ghost_hit`), demotion (`settle_fast_tier`), promotion back to
//! fast (`give_second_chance`), `remove`, and `evict_one`'s own main-queue
//! path.
//!
//! That sum is deliberately NOT `fast_bytes_used()`. This variant's
//! one-access queue is physically DRAM -- the whole point of it -- so that
//! trait method adds `one_access_used` back in to report total DRAM. Using
//! it here would charge the one-access queue against the main queue's budget
//! as well as its own, double-counting it and declaring the main queue full
//! far too early, which would quietly change eviction for every workload.
//! `main_capacity` is likewise unrelated to
//! `effective_main_fast_capacity()`, which bounds only the main queue's FAST
//! segment out of `fast_capacity`: a demotion shifts bytes from `fast_used`
//! to `slow_used` and leaves this gate's reading untouched, which is exactly
//! why a whole-queue budget is the right thing to compare against.
//!
//! The mid-segment check belongs to the main-queue path, not to the
//! one-access one, so it stays where it is: `check_slow_midpoint` still runs
//! once per `evict_one` call, immediately before the main-queue loop, and
//! now runs on both routes into that loop.
//!
//! Nothing else about eviction moves: the demotion-time reference-bit
//! reprieve, the eviction-time second chance (tail and midpoint alike), and
//! the eager promotion out of the one-access queue are all precisely as they
//! were.
//!
//! Degenerate case, inherited verbatim from the plain stack: at
//! `one_access_ratio == 1.0` the complement is `0`, so the main queue reads
//! as full from the outset and `evict_one` can never reach the one-access
//! tail -- a cache configured that way cannot evict at all.
//! `SThreeFifoStack` behaves identically at that ratio, so this is a
//! property of the rule being mirrored, not of this stack's accounting.
//!
//! ## Fast-tier watermarks
//!
//! `settle_fast_tier` no longer demotes the instant the main queue's fast
//! segment crosses its ceiling. It triggers once `fast_used` passes
//! `watermarks::high_bytes` of the effective budget and then drains down to
//! `watermarks::low_bytes` of that same budget, so demotions arrive as
//! occasional batches instead of one object per promotion. The budget the
//! watermarks are applied to is
//! [`S3FifoGhostLazyDemotionFastAdmissionMidpointHybridStack::effective_main_fast_capacity`]
//! -- the watermarks sit on top of it, never in place of it. See
//! [`super::watermarks`] for the ratios, their env overrides, and how to
//! restore the original drain-to-the-ceiling behaviour exactly.
//!
//! ## Shared DRAM-reservation overhead
//!
//! The object hashtable and this stack's own eviction-stack bookkeeping
//! (`one_access_queue` / `main_queue` / `ghost`, plus the `entries` index)
//! all live in DRAM, but none of their bytes are counted in `fast_used` or
//! `one_access_used`. Without a correction the fast tier's real DRAM
//! footprint therefore exceeds `fast_capacity` by exactly that metadata.
//!
//! `shared_overhead` (see `crate::object::overhead::
//! get_hybrid_dram_shared_overhead`, wired in via `with_shared_overhead`) is
//! the approximate per-*tracked-key* cost of those structures. It is charged
//! against every tracked key, not just fast-tier ones: a demotion moves the
//! object's *value* bytes to PMEM and leaves the key's list node, its
//! `entries` slot and its hashtable slot exactly where they were, in DRAM.
//! One tracked key costs exactly one list node, because a key is linked into
//! exactly one of `one_access_queue` / `main_queue` at a time -- promotion
//! (`promote_from_one_access`) removes from the former before pushing to the
//! latter, and the fast/slow split of the main queue is a tier *tag* plus
//! the `main_boundary` cursor, not a second list.
//!
//! Two things make the reservation's shape here differ from
//! `LruHybridStack`'s single subtraction:
//!
//! * **Two independently-capacitied fast segments.** The one-access queue
//!   carries its own budget (`one_access_capacity`, enforced by
//!   `needs_capacity_eviction`) and the main queue's fast segment gets
//!   whatever `fast_capacity` has left after it. The metadata cost is real
//!   only *once*, so charging it in full against both would waste usable
//!   fast-tier budget for no reason; `reserved_shares` splits it between
//!   them in proportion to their capacities and each segment subtracts only
//!   its own share -- the same treatment `LruSizedHybridStack` gives its
//!   small/large pair.
//! * **The ghost queue.** `ghost` holds BARE KEYS for objects that are no
//!   longer in the cache: no `entries` slot, no hashtable slot, no value
//!   bytes anywhere. Its DRAM scales with the ghost queue's own length, not
//!   with the tracked-key count, so a per-tracked-key constant cannot model
//!   it. It is reserved as a genuinely separate term instead --
//!   [`GHOST_ENTRY_DRAM_OVERHEAD`], the fixed per-node cost of the
//!   `HashList<HashedKey>` every ghost-keeping design uses, multiplied by the
//!   *actual* `ghost.len()`, which is exact rather than modelled.
//!   (`trim_ghost` bounds the queue at `main_count` entries, but only ever
//!   runs on a main-queue eviction, so the live length is the honest thing to
//!   read.)
//!
//! The ghost term is deliberately NOT caller-configurable: it is the same
//! constant for every stack that keeps a bare-key ghost queue, so
//! `reserved_overhead` reads it straight out of `crate::object::overhead`
//! rather than taking it from a builder a construction site could forget to
//! call (which would silently reserve nothing). It is still `cfg`-selected:
//! under `eviction_stacks_pmem` the ghost list is not in DRAM at all and the
//! constant is `0`.
//!
//! Only `shared_overhead` defaults to `0`, so unit tests that construct the
//! stack directly -- and never age a key out into the ghost queue -- still see
//! the pure value-budget behaviour, unchanged.

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
	object::{ObjectSize, overhead::GHOST_ENTRY_DRAM_OVERHEAD},
	worker::policy::policy_stack::{PolicyStack, Tier, narrow_resident, watermarks},
};

/// Which live queue a key currently belongs to.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Queue {
	OneAccess,
	Main,
}

/// Combined per-key bookkeeping. `tier`/`accessed` are only meaningful
/// while `queue == Main`.
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

pub struct S3FifoGhostLazyDemotionFastAdmissionMidpointHybridStack {
	one_access_queue: QueueList,
	main_queue: QueueList,
	ghost: QueueList,

	entries: EntryMap,

	one_access_ratio: f64,
	one_access_capacity: CacheSize,
	one_access_used: CacheSize,

	/// The MAIN queue's total byte budget, spanning both tiers --
	/// `(1 - one_access_ratio) * max_size`, the exact complement of
	/// `one_access_capacity` and recomputed beside it in `resize`. Read only by
	/// `is_main_full`, which gates `evict_one`'s one-access-tail priority; see
	/// the module doc's "Eviction order" section.
	///
	/// Not to be confused with the main queue's FAST-segment budget
	/// (`effective_main_fast_capacity()`): that is carved out of
	/// `fast_capacity` and governs demotion, this is carved out of
	/// `max_size` and governs eviction order.
	main_capacity: CacheSize,

	fast_capacity: CacheSize,
	fast_used: CacheSize,
	slow_used: CacheSize,

	/// Approximate per-*tracked-key* DRAM cost of the shared structures
	/// (object hashtable + this stack's eviction-stack bookkeeping) that
	/// hold an entry for every tracked object of both tiers. Reserved out of
	/// the fast-tier budget -- split between the one-access queue and the
	/// main queue's fast segment by [`S3FifoGhostLazyDemotionFastAdmissionMidpointHybridStack::reserved_shares`]
	/// -- so that budget bounds total DRAM (values + shared metadata) rather
	/// than just fast-tier values. `0` unless set via
	/// [`S3FifoGhostLazyDemotionFastAdmissionMidpointHybridStack::with_shared_overhead`],
	/// so unit tests exercising the pure value-budget behaviour are
	/// unaffected.
	shared_overhead: CacheSize,

	fast_count: usize,
	main_count: usize,

	main_boundary: Option<HashedKey>,

	/// Cursor tracking (approximately) the middle of the slow segment --
	/// see the module doc's "Locating 'the middle'" section.
	slow_midpoint: Option<HashedKey>,
	/// Accumulates 0.5-position drift per qualifying event; reset (and the
	/// cursor nudged) every time it reaches 2.
	midpoint_drift: u8,

	migrations: Vec<(HashedKey, Tier)>,
}

impl S3FifoGhostLazyDemotionFastAdmissionMidpointHybridStack {
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
		let (one_access_queue, main_queue, ghost, entries) = Self::new_collections();

		S3FifoGhostLazyDemotionFastAdmissionMidpointHybridStack {
			one_access_queue,
			main_queue,
			ghost,

			entries,

			one_access_ratio,
			one_access_capacity: (one_access_ratio * max_size as f64) as CacheSize,
			one_access_used: 0,
			main_capacity: ((1.0 - one_access_ratio) * max_size as f64) as CacheSize,

			fast_capacity,
			fast_used: 0,
			slow_used: 0,

			shared_overhead: 0,

			fast_count: 0,
			main_count: 0,

			main_boundary: None,
			slow_midpoint: None,
			midpoint_drift: 0,

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

	/// Total DRAM currently reserved for shared metadata:
	///
	/// * one `shared_overhead` per *tracked* key -- a tracked key occupies
	///   exactly one list node (`one_access_queue` XOR `main_queue`) plus one
	///   `entries` slot plus one hashtable slot, whichever tier its value
	///   bytes are in;
	/// * one [`GHOST_ENTRY_DRAM_OVERHEAD`] per *actual* ghost entry -- bare
	///   keys for objects that are not tracked at all, so they are charged by
	///   real `ghost.len()` rather than modelled per tracked key. That cost is
	///   the same shared constant for every ghost-keeping stack (they all keep
	///   the same `HashList<HashedKey>`), so it is read directly rather than
	///   injected by a builder; it is `0` under `eviction_stacks_pmem`, where
	///   the ghost list does not live in DRAM.
	///
	/// The two terms are independent: the ghost term is charged for whatever
	/// is in `ghost` regardless of `shared_overhead`, `0` included.
	///
	/// Split between the two fast segments by [`Self::reserved_shares`];
	/// nothing is charged twice.
	fn reserved_overhead(&self) -> CacheSize {
		self.entries.len() as CacheSize * self.shared_overhead
			+ self.ghost.len() as CacheSize * (GHOST_ENTRY_DRAM_OVERHEAD as CacheSize)
	}

	/// Splits [`Self::reserved_overhead`] between this stack's two
	/// independently-capacitied FAST segments -- the one-access queue
	/// (`one_access_capacity`) and the main queue's fast segment (whatever
	/// `fast_capacity` has left after it) -- in proportion to those
	/// capacities, returning `(one_access_share, main_share)`.
	///
	/// The underlying metadata cost is real only once, so charging it in
	/// full against each segment independently would waste usable fast-tier
	/// budget for no reason; `LruSizedHybridStack::reserved_shares` splits
	/// its own small/large pair in exactly this way. The remainder is given
	/// to the main segment so the two shares always re-add to the whole.
	/// `(0, 0)` when there is no capacity at all to proportion against.
	fn reserved_shares(&self) -> (CacheSize, CacheSize) {
		let reserved = self.reserved_overhead();

		let one_access_capacity = self.one_access_capacity;
		let main_capacity = self.fast_capacity.saturating_sub(self.one_access_capacity);
		let total_capacity = one_access_capacity + main_capacity;

		if total_capacity == 0 {
			return (0, 0);
		}

		let one_access_share =
			((reserved as u128 * one_access_capacity as u128) / total_capacity as u128) as CacheSize;
		let main_share = reserved.saturating_sub(one_access_share);

		(one_access_share, main_share)
	}

	/// The one-access queue's own byte budget after its share of the
	/// shared-metadata reservation is carved out -- the value
	/// `needs_capacity_eviction` compares `one_access_used` against.
	fn effective_one_access_capacity(&self) -> CacheSize {
		self.one_access_capacity.saturating_sub(self.reserved_shares().0)
	}

	/// The main queue's fast-segment byte budget: `fast_capacity`, minus the
	/// one-access queue's reservation (that queue is fast-tier here and
	/// competes for the same DRAM), minus this segment's share of the
	/// shared-metadata reservation. This is the value `settle_fast_tier`
	/// applies the watermarks to.
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

	/// Returns `true` if `key` currently has a ghost entry. Exposed for tests.
	pub fn is_ghost(&self, key: HashedKey) -> bool {
		self.ghost.contains(&key)
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

		self.main_queue.push_front(key);
		self.entries.insert(key, S3FifoEntry { dram_resident, queue: Queue::Main,
			tier: Some(Tier::Fast),
			size,
			accessed: false,
		});
		self.fast_used += size_bytes;
		self.fast_count += 1;
		self.main_count += 1;

		if self.main_boundary.is_none() {
			self.main_boundary = Some(key);
		}

		self.settle_fast_tier();
	}

	fn admit_via_ghost_hit(&mut self, key: HashedKey, size: ObjectSize, dram_resident: u8) {
		self.main_queue.push_front(key);
		self.entries.insert(key, S3FifoEntry { dram_resident, queue: Queue::Main,
			tier: Some(Tier::Fast),
			size,
			accessed: false,
		});
		self.fast_used += (size as CacheSize).saturating_sub(dram_resident as CacheSize);
		self.fast_count += 1;
		self.main_count += 1;

		if self.main_boundary.is_none() {
			self.main_boundary = Some(key);
		}

		self.settle_fast_tier();
	}

	/// Moves the midpoint cursor one step toward the front, if possible.
	/// No-op if the cursor is empty, or if the neighbor toward the front
	/// turns out to already be Fast (the cursor has reached the boundary)
	/// -- it just stays put until the next nudge, once growth or shrinkage
	/// resumes making room to move again.
	fn nudge_midpoint_toward_front(&mut self) {
		let Some(current) = self.slow_midpoint else { return };
		let Some(&candidate) = self.main_queue.before(&current) else { return };

		if self.entries.get(&candidate).and_then(|entry| entry.tier) == Some(Tier::Slow) {
			self.slow_midpoint = Some(candidate);
		}
	}

	/// Call after any event that changes the slow segment's size by
	/// exactly one in either direction (a demotion, a slow-tier eviction,
	/// or a promotion/removal out of the slow segment) once the cursor is
	/// already initialized. See the module doc's design notes for the
	/// "every 2 events, one step" derivation.
	fn bump_midpoint_drift(&mut self) {
		self.midpoint_drift += 1;

		if self.midpoint_drift >= 2 {
			self.midpoint_drift = 0;
			self.nudge_midpoint_toward_front();
		}
	}

	/// If `key` is currently the midpoint cursor's target, redirects it to
	/// the `before()` neighbor (only accepted if still Slow-tagged) before
	/// `key` is physically unlinked or moved. Must be called while `key`
	/// is still linked in `main_queue` -- `before()` needs that to resolve
	/// the neighbor.
	fn redirect_midpoint_before_removing(&mut self, key: HashedKey) {
		if self.slow_midpoint != Some(key) {
			return;
		}

		self.slow_midpoint = self.main_queue.before(&key)
			.copied()
			.filter(|candidate| self.entries.get(candidate).and_then(|entry| entry.tier) == Some(Tier::Slow));
	}

	/// Checks the midpoint cursor's reference bit and, if set, gives it an
	/// early second chance -- the whole point of this variant. No-op if
	/// the slow segment is currently empty. Called once per `evict_one`
	/// pass over the main queue.
	fn check_slow_midpoint(&mut self) {
		let Some(candidate) = self.slow_midpoint else { return };
		let accessed = self.entries.get(&candidate).map(|entry| entry.accessed).unwrap_or(false);

		if accessed {
			self.give_second_chance(candidate);
		}
	}

	/// The eviction-time second chance -- also reused directly by
	/// `check_slow_midpoint` for the new mid-segment check, since both are
	/// "promote this Slow key back to the front of Fast" with identical
	/// mechanics. A key reaching here with `tier == Some(Tier::Slow)`
	/// genuinely has PMEM-resident bytes, so the migration this produces
	/// is real, necessary work either way.
	fn give_second_chance(&mut self, key: HashedKey) {
		let Some(entry) = self.entries.get(&key).copied() else { return };
		let size = entry.migrating();
		let was_fast = entry.tier == Some(Tier::Fast);
		let was_boundary = was_fast && self.main_boundary == Some(key);

		if !was_fast {
			self.redirect_midpoint_before_removing(key);
		}

		let new_boundary_if_moved = if was_boundary {
			self.main_queue.before(&key).copied()
		} else {
			None
		};

		self.main_queue.move_front(&key);

		if was_boundary {
			self.main_boundary = new_boundary_if_moved;
		}

		if let Some(entry) = self.entries.get_mut(&key) {
			entry.tier = Some(Tier::Fast);
			entry.accessed = false;
		}

		if !was_fast {
			self.slow_used = self.slow_used.saturating_sub(size);
			self.fast_used += size;
			self.fast_count += 1;
			self.bump_midpoint_drift();
		}

		if self.main_boundary.is_none() {
			self.main_boundary = Some(key);
		}

		self.settle_fast_tier();

		if self.entries.get(&key).and_then(|entry| entry.tier) == Some(Tier::Fast) {
			self.migrations.push((key, Tier::Fast));
		}
	}

	/// Demotes out of the main queue's fast segment once `fast_used` exceeds
	/// the HIGH watermark of [`Self::effective_main_fast_capacity`], then
	/// drains down to the LOW watermark rather than merely back under the
	/// ceiling.
	///
	/// The budget the watermarks are applied *to* is
	/// [`Self::effective_main_fast_capacity`]: `fast_capacity`, minus the
	/// one-access queue's reservation (that queue is fast-tier here and
	/// competes for the same DRAM), minus this segment's share of the
	/// shared-metadata reservation ([`Self::reserved_shares`]). The
	/// watermarks sit on top of that effective value and never replace it,
	/// so the "one-access queue + main fast segment + shared metadata <=
	/// `fast_capacity`" bound this stack relies on stays exactly as valid as
	/// before — tighter now, never looser.
	///
	/// Composition order matters and is fixed: the overhead reservation comes
	/// out of the capacity *first*, and the watermarks are then taken of the
	/// remainder. A triggered pass therefore drains to
	/// `low_bytes(fast_capacity - one_access_capacity - main_share)`, never to
	/// `low_bytes(fast_capacity)`.
	///
	/// Reading the effective value once, before the loop, is exact rather
	/// than an approximation: `reserved_shares()` counts *tracked* keys and
	/// live ghost entries, and a demotion only retags a tracked key — it
	/// drops neither a tracked entry nor a ghost one — so the value cannot
	/// move while the loop below runs.
	///
	/// Draining below the ceiling instead of exactly to it is what turns the
	/// steady state from "every promotion demotes exactly one object" into
	/// occasional multi-object batches — see [`watermarks`] for the full
	/// rationale, and for how to restore the old drain-to-ceiling behaviour
	/// exactly (`FAST_TIER_HIGH_WATERMARK=1.0`, `FAST_TIER_LOW_WATERMARK=1.0`).
	///
	/// Nothing else moves. The demotion-time reference-bit reprieve, the
	/// per-demotion bookkeeping (tier tag, `fast_used`/`slow_used`,
	/// `fast_count`, the boundary walk, the migration push) and the
	/// midpoint-cursor maintenance in the loop below are all unchanged, and
	/// each still runs exactly once per object the pass touches — only the
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
			let Some(candidate) = self.main_boundary else { break };

			let accessed = self.entries.get(&candidate).map(|entry| entry.accessed).unwrap_or(false);

			if accessed {
				let new_boundary = self.main_queue.before(&candidate).copied();

				self.main_queue.move_front(&candidate);
				self.main_boundary = new_boundary;

				if let Some(entry) = self.entries.get_mut(&candidate) {
					entry.accessed = false;
				}

				continue;
			}

			let size = self.entries.get(&candidate).map(|entry| entry.migrating()).unwrap_or(0);
			let new_boundary = self.main_queue.before(&candidate).copied();

			if let Some(entry) = self.entries.get_mut(&candidate) {
				entry.tier = Some(Tier::Slow);
			}

			self.fast_used = self.fast_used.saturating_sub(size);
			self.fast_count = self.fast_count.saturating_sub(1);
			self.slow_used += size;
			self.main_boundary = new_boundary;

			self.migrations.push((candidate, Tier::Slow));

			// A real demotion always grows the slow segment by exactly one
			// -- see the module doc's design notes.
			if self.slow_midpoint.is_none() {
				self.slow_midpoint = Some(candidate);
			} else {
				self.bump_midpoint_drift();
			}
		}
	}

	fn evict_one_access_tail(&mut self) -> Option<HashedKey> {
		let key = self.one_access_queue.pop_back()?;
		let size = self.entries.remove(&key).map(|entry| entry.migrating()).unwrap_or(0);

		self.one_access_used = self.one_access_used.saturating_sub(size);
		self.ghost.push_front(key);

		Some(key)
	}

	/// Whether the main queue has reached its own byte budget -- the gate on
	/// `evict_one`'s one-access-tail priority, mirroring
	/// `SThreeFifoStack`'s `Stack::is_full` (`used >= max`, so a `0` budget
	/// reads as permanently full).
	///
	/// `fast_used + slow_used` IS the main queue's byte total: one-access
	/// residents carry `tier: None` and move `one_access_used` alone, so
	/// neither counter ever includes them. Deliberately not `fast_bytes_used()`,
	/// which folds `one_access_used` back in because this variant's one-access
	/// queue is DRAM too -- see the module doc's "Eviction order" section.
	fn is_main_full(&self) -> bool {
		self.fast_used + self.slow_used >= self.main_capacity
	}

	fn trim_ghost(&mut self) {
		while self.ghost.len() > self.main_count {
			self.ghost.pop_back();
		}
	}
}

impl PolicyStack for S3FifoGhostLazyDemotionFastAdmissionMidpointHybridStack {
	fn is_policy(&self, policy: &PaperPolicy) -> bool {
		matches!(policy, PaperPolicy::S3FifoGhostLazyDemotionFastAdmissionMidpointHybrid(ratio) if *ratio == self.one_access_ratio)
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

		if self.ghost.contains(&key) {
			self.admit_via_ghost_hit(key, size, dram_resident);
			return;
		}

		self.one_access_queue.push_front(key);
		self.entries.insert(key, S3FifoEntry { dram_resident, queue: Queue::OneAccess,
			tier: None,
			size,
			accessed: false,
		});
		self.one_access_used += (size as CacheSize).saturating_sub(dram_resident as CacheSize);
	}

	fn update(&mut self, key: HashedKey) {
		if self.entries.contains_key(&key) {
			self.touch(key);
		}
	}

	fn remove(&mut self, key: HashedKey) {
		self.ghost.remove(&key);

		let Some(entry) = self.entries.remove(&key) else { return };
		let size = entry.migrating();

		match entry.queue {
			Queue::OneAccess => {
				self.one_access_queue.remove(&key);
				self.one_access_used = self.one_access_used.saturating_sub(size);
			},

			Queue::Main => {
				let new_boundary_if_needed = if entry.tier == Some(Tier::Fast) && self.main_boundary == Some(key) {
					self.main_queue.before(&key).copied()
				} else {
					None
				};

				// Redirect the midpoint cursor BEFORE unlinking, if this
				// key is currently its target -- `before()` needs the key
				// still linked.
				let new_midpoint_if_needed = if entry.tier == Some(Tier::Slow) && self.slow_midpoint == Some(key) {
					self.main_queue.before(&key).copied()
						.filter(|candidate| self.entries.get(candidate).and_then(|e| e.tier) == Some(Tier::Slow))
				} else {
					None
				};

				self.main_queue.remove(&key);
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

						if self.slow_midpoint == Some(key) {
							self.slow_midpoint = new_midpoint_if_needed;
						}

						self.bump_midpoint_drift();
					},

					None => {},
				}
			},
		}
	}

	fn resize(&mut self, max_size: CacheSize) {
		self.one_access_capacity = (self.one_access_ratio * max_size as f64) as CacheSize;
		self.main_capacity = ((1.0 - self.one_access_ratio) * max_size as f64) as CacheSize;
		self.settle_fast_tier();
	}

	fn clear(&mut self) {
		self.one_access_queue.clear();
		self.main_queue.clear();
		self.ghost.clear();
		self.entries.clear();

		self.one_access_used = 0;
		self.fast_used = 0;
		self.slow_used = 0;
		self.fast_count = 0;
		self.main_count = 0;
		self.main_boundary = None;
		self.slow_midpoint = None;
		self.midpoint_drift = 0;
		self.migrations.clear();
	}

	fn evict_one(&mut self) -> Option<HashedKey> {
		// The one-access tail is only prioritized while the main queue still
		// has room, exactly as `SThreeFifoStack::evict_one` prioritizes its
		// small queue -- see the module doc's "Eviction order" section. With
		// the main queue full, fall straight through to the main-queue loop.
		if !self.is_main_full() {
			if let Some(key) = self.evict_one_access_tail() {
				return Some(key);
			}
		}

		// The new mid-segment check -- see the module doc. Runs once per
		// call, exactly when this stack is about to evaluate the main
		// queue for a real eviction.
		self.check_slow_midpoint();

		loop {
			let key = *self.main_queue.back()?;
			let accessed = self.entries.get(&key).map(|entry| entry.accessed).unwrap_or(false);

			if accessed {
				self.give_second_chance(key);
				continue;
			}

			// Redirect the midpoint cursor BEFORE unlinking, if this key
			// is currently its target.
			if self.slow_midpoint == Some(key) {
				let new_target = self.main_queue.before(&key)
					.copied()
					.filter(|candidate| self.entries.get(candidate).and_then(|e| e.tier) == Some(Tier::Slow));
				self.slow_midpoint = new_target;
			}

			self.main_queue.pop_back();
			let removed = self.entries.remove(&key);
			let size = removed.map(|entry| entry.migrating()).unwrap_or(0);
			let tier = removed.and_then(|entry| entry.tier);

			self.main_count = self.main_count.saturating_sub(1);

			match tier {
				Some(Tier::Fast) => {
					self.fast_used = self.fast_used.saturating_sub(size);
					self.fast_count = self.fast_count.saturating_sub(1);

					if self.main_boundary == Some(key) {
						self.main_boundary = self.main_queue.back().copied();
					}
				},

				Some(Tier::Slow) => {
					self.slow_used = self.slow_used.saturating_sub(size);
					self.bump_midpoint_drift();
				},

				None => {},
			}

			self.trim_ghost();

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
		self.fast_count + self.one_access_queue.len()
	}

	fn slow_object_count(&self) -> usize {
		self.main_count - self.fast_count
	}

	/// Compared against [`Self::effective_one_access_capacity`] rather than
	/// the raw `one_access_capacity`, so the one-access queue reserves its
	/// own share of the shared-metadata DRAM too. Applying only the main
	/// segment's share would leave the one-access share reserved nowhere,
	/// i.e. would under-reserve by exactly that amount.
	///
	/// This cannot spin: each one-access eviction removes a tracked entry
	/// (`-shared_overhead`) and adds at most one ghost entry
	/// (`+GHOST_ENTRY_DRAM_OVERHEAD`), and it always shrinks the queue by one,
	/// so `one_access_used` reaches `0` — at which point `0 > effective` is
	/// false for every effective value — after finitely many steps whatever
	/// the two constants are.
	fn needs_capacity_eviction(&self) -> bool {
		self.one_access_used > self.effective_one_access_capacity()
	}
}

// ── Design notes: the "every 2 events, one step" drift derivation ─────────
//
// Model the slow segment as a list of length N, positions 0 (nearest the
// fast/slow boundary) through N-1 (the tail). The true middle is at index
// N/2 (integer division). The cursor tracks a *specific object*, not an
// index -- its absolute index drifts as the segment mutates around it.
//
// Front-insertion (a demotion always lands its retagged object at index 0,
// since it was already sitting where the boundary lands -- nothing
// physically moves in the list): every existing object's index increases
// by 1, including the tracked one. The target index (N/2) increases by
// only 0.5 on average as N grows by 1. Net: the tracked object drifts +0.5
// positions past the true middle per event.
//
// Tail-removal (an eviction) or arbitrary-position removal (a promotion):
// the tracked object's own index is unaffected (removal at/after it, or
// the tracked object being the one removed and immediately redirected to
// its front-ward neighbor whose index is one less than the removed
// object's would have been), but N decreases by 1, so the target index
// (N/2) decreases by 0.5. Net: the tracked object again drifts +0.5
// positions past the true middle per event -- same sign as growth.
//
// Since both kinds of qualifying event drift the tracked object the same
// direction by the same magnitude, a single counter suffices: accumulate
// 1 per event, and every time it reaches 2 (i.e. every 2 events, matching
// 2 * 0.5 = 1 full position of accumulated drift), move the cursor one
// step toward the front via `before()` to cancel it out. Verified by hand
// against the small worked examples in this stack's unit tests below.

#[cfg(test)]
mod tests {
	use super::*;

	fn drain(stack: &mut S3FifoGhostLazyDemotionFastAdmissionMidpointHybridStack) -> Vec<(HashedKey, Tier)> {
		stack.drain_tier_migrations()
	}

	/// Smallest `fast_capacity` that holds a fast segment of exactly `bytes`
	/// without a demotion pass either triggering on it or -- once one does
	/// trigger -- draining it, i.e. `bytes` sits at or below the LOW
	/// watermark, while one further `next`-byte object still pushes usage
	/// past the HIGH one.
	///
	/// The hand-traced fixtures below were originally written against the
	/// drain-to-the-ceiling rule, where "capacity" and "the point a pass
	/// settles at" were the same number. They derive their capacity from this
	/// instead of hard-coding it, so their traces hold unchanged at any
	/// configured ratio pair -- including `1.0`/`1.0`, which reproduces the
	/// original literals (10 and 20) exactly. They cannot simply pin the
	/// ratios: the watermarks are process-global `OnceLock`s read once per
	/// process, so a test that set the env vars would race every other test
	/// in the binary.
	fn capacity_holding(bytes: CacheSize, next: CacheSize) -> CacheSize {
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
		let mut stack = S3FifoGhostLazyDemotionFastAdmissionMidpointHybridStack::new(1.0, 1_000, 1_000);

		stack.insert(1, 10);
		stack.insert(2, 10);

		assert_eq!(stack.tier_of(1), Some(Tier::Fast));
		assert_eq!(stack.tier_of(2), Some(Tier::Fast));
		assert_eq!(drain(&mut stack), Vec::new());
	}

	#[test]
	fn a_key_aging_out_without_reaccess_becomes_a_ghost_entry() {
		// `one_access_ratio` 0.5 rather than 1.0: `evict_one` only reaches the
		// one-access tail while the main queue is below `main_capacity`
		// (`(1 - ratio) * max_size` -- see the module doc's "Eviction order"
		// section), and at ratio 1.0 that budget is `0`, so the main queue reads
		// as full from the outset. The ghost-on-ageing-out behaviour under test
		// here is indifferent to the ratio.
		let mut stack = S3FifoGhostLazyDemotionFastAdmissionMidpointHybridStack::new(0.5, 1_000, 1_000);

		stack.insert(1, 10);
		drain(&mut stack);

		assert_eq!(stack.evict_one(), Some(1));
		assert!(stack.is_ghost(1));
	}

	#[test]
	fn ghost_hit_on_readmission_lands_in_fast_tier_without_a_migration() {
		let mut stack = S3FifoGhostLazyDemotionFastAdmissionMidpointHybridStack::new(0.0, 1_000, 1_000);

		stack.insert(1, 10);
		drain(&mut stack);
		stack.evict_one();
		assert!(stack.is_ghost(1));

		stack.insert(1, 10);

		assert_eq!(drain(&mut stack), Vec::new());
		assert_eq!(stack.tier_of(1), Some(Tier::Fast));
	}

	#[test]
	fn an_accessed_fast_boundary_key_is_reprieved_at_demotion_time_not_demoted() {
		let mut stack = S3FifoGhostLazyDemotionFastAdmissionMidpointHybridStack::new(0.0, 1_000, capacity_holding(10, 10));

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

	// ── the signature new mechanic: a checkpoint mid-slow-segment ──────────

	/// Builds a stack with 5 keys admitted and promoted in order 1..=5, a
	/// fast segment sized to hold exactly 2 objects across a settled pass
	/// (see `capacity_holding`), one_access_ratio=0.0. Traced by hand: keys
	/// 1, 2, 3 get demoted (oldest first) as keys 4 and 5 arrive, leaving the
	/// slow segment as [3, 2, 1] (front-to-boundary to tail) and the fast
	/// segment as [5, 4]. After exactly 3 demotions the drift-correction
	/// cursor settles on the middle element, key 2.
	fn build_five_key_stack() -> S3FifoGhostLazyDemotionFastAdmissionMidpointHybridStack {
		let mut stack = S3FifoGhostLazyDemotionFastAdmissionMidpointHybridStack::new(0.0, 1_000, capacity_holding(20, 10));

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
		assert!(!stack.is_midpoint(1));
		assert!(!stack.is_midpoint(3));
	}

	#[test]
	fn a_reaccessed_midpoint_key_is_promoted_early_instead_of_waiting_for_the_tail() {
		let mut stack = build_five_key_stack();
		assert!(stack.is_midpoint(2));

		// Reaccess key 2 (currently Slow, sitting at the midpoint) without
		// otherwise touching anything -- just sets its reference bit, same
		// lazy-bit convention as everywhere else in this design.
		stack.update(2);
		assert_eq!(stack.tier_of(2), Some(Tier::Slow), "a mere access must not itself migrate or reorder");

		// evict_one() must check the midpoint before it ever looks at the
		// tail. Traced by hand: key 2 is promoted (a real Slow->Fast
		// migration); that promotion pushes fast_used back over capacity,
		// cascading a real demotion of the current boundary (key 4, the
		// only unaccessed fast key); the call then proceeds to its own
		// normal tail eviction of key 1 (the tail, unaccessed).
		let evicted = stack.evict_one();

		assert_eq!(stack.tier_of(2), Some(Tier::Fast), "the reaccessed midpoint key should have been promoted early");
		assert_eq!(stack.tier_of(4), Some(Tier::Slow), "cascading demotion after the midpoint promotion");
		assert_eq!(evicted, Some(1), "the tail should still be evicted normally in the same call");
		assert!(!stack.contains(1));
		assert!(!stack.is_ghost(1), "main-queue tail evictions never populate the ghost queue");
	}

	#[test]
	fn an_unaccessed_midpoint_key_is_left_alone() {
		let mut stack = build_five_key_stack();
		assert!(stack.is_midpoint(2));

		// Key 2's bit is clear (never reaccessed) -- evict_one() must not
		// promote it. It should proceed straight to the normal tail
		// eviction of key 1 instead.
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
		assert!(stack.is_midpoint(3), "cursor should redirect to the before()-neighbor still in the slow segment");
	}

	#[test]
	fn evicting_the_only_slow_key_clears_the_midpoint_cursor() {
		let mut stack = S3FifoGhostLazyDemotionFastAdmissionMidpointHybridStack::new(0.0, 1_000, capacity_holding(10, 10));

		stack.insert(1, 10);
		stack.update(1);
		stack.insert(2, 10);
		stack.update(2); // demotes key 1 -- the only slow key, so it's also the midpoint
		drain(&mut stack);
		assert!(stack.is_midpoint(1));

		let evicted = stack.evict_one();

		assert_eq!(evicted, Some(1));
		assert!(!stack.is_midpoint(1));
	}

	#[test]
	fn evict_one_gives_an_accessed_slow_key_a_second_chance() {
		let mut stack = S3FifoGhostLazyDemotionFastAdmissionMidpointHybridStack::new(0.0, 1_000, capacity_holding(10, 10));

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
	fn remove_clears_ghost_entry_too() {
		// `one_access_ratio` 0.5 rather than 1.0, for the reason spelled out on
		// `a_key_aging_out_without_reaccess_becomes_a_ghost_entry` above.
		let mut stack = S3FifoGhostLazyDemotionFastAdmissionMidpointHybridStack::new(0.5, 1_000, 1_000);

		stack.insert(1, 10);
		drain(&mut stack);
		stack.evict_one();
		assert!(stack.is_ghost(1));

		stack.remove(1);
		assert!(!stack.is_ghost(1));
	}

	#[test]
	fn remove_and_clear_reset_bookkeeping() {
		let mut stack = S3FifoGhostLazyDemotionFastAdmissionMidpointHybridStack::new(1.0, 1_000, 1_000);

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
		let mut stack = S3FifoGhostLazyDemotionFastAdmissionMidpointHybridStack::new(1.0, 1_000, 1_000);

		stack.insert(1, 10);
		stack.insert(2, 10);
		drain(&mut stack);

		assert_eq!(stack.fast_bytes_used(), 20);
		assert_eq!(stack.slow_bytes_used(), 0);
		assert_eq!(stack.fast_object_count(), 2);
		assert_eq!(stack.slow_object_count(), 0);
	}

	// ── shared high/low watermarks (`super::watermarks`) ──────────────────
	//
	// The ratios are process-global (`OnceLock`, seeded once from
	// `FAST_TIER_HIGH_WATERMARK` / `FAST_TIER_LOW_WATERMARK`), so these tests
	// cannot set the env vars for themselves without racing every other test
	// in the binary. They compute their expectations from
	// `watermarks::high()` / `watermarks::low()` instead, and therefore hold
	// at any configured ratio pair -- including the `1.0` / `1.0` setting
	// that restores the original drain-to-the-ceiling behaviour.

	/// Watermark test rig: a one-access reservation of 1_000 bytes carved out
	/// of an 11_000-byte fast tier leaves the main queue a round 10_000, so
	/// `high_bytes` / `low_bytes` land on exact byte counts whatever the
	/// ratios are -- and pins that the watermarks are applied to the
	/// *effective* budget, not to raw `fast_capacity`.
	fn watermark_stack() -> S3FifoGhostLazyDemotionFastAdmissionMidpointHybridStack {
		let stack = S3FifoGhostLazyDemotionFastAdmissionMidpointHybridStack::new(0.1, 10_000, 11_000);

		assert_eq!(stack.effective_main_fast_capacity(), 10_000);

		stack
	}

	/// (a) Below the high watermark nothing happens at all -- under the old
	/// rule the whole band between the high watermark and the ceiling would
	/// have been sitting there demoting one object per promotion.
	#[test]
	fn usage_just_below_the_high_watermark_triggers_no_demotion() {
		let mut stack = watermark_stack();
		let effective = stack.effective_main_fast_capacity();
		let high = watermarks::high_bytes(effective);

		assert!(high > 1, "watermark config leaves no room for this test");

		// One key sized a byte under the high watermark, promoted out of the
		// one-access queue into the main queue's fast segment.
		stack.insert(1, (high - 1) as ObjectSize);
		stack.update(1);

		assert_eq!(stack.fast_used, high - 1);
		assert_eq!(stack.tier_of(1), Some(Tier::Fast));
		assert!(drain(&mut stack).is_empty(), "demoted below the high watermark of {high}");
		assert_eq!(stack.slow_used, 0);
		assert_eq!(stack.slow_object_count(), 0);
		assert_eq!(stack.slow_midpoint, None);
	}

	/// (b) Sitting exactly *on* the high watermark still triggers nothing
	/// (the trigger is `>`, not `>=`); one byte past it does.
	#[test]
	fn usage_above_the_high_watermark_triggers_a_pass() {
		let mut stack = watermark_stack();
		let effective = stack.effective_main_fast_capacity();
		let high = watermarks::high_bytes(effective);
		let low = watermarks::low_bytes(effective);

		assert!(low >= 1, "watermark config leaves no room for this test");

		stack.insert(1, high as ObjectSize);
		stack.update(1);

		assert_eq!(stack.fast_used, high);
		assert!(drain(&mut stack).is_empty(), "exactly on the high watermark must not trigger");

		// One more byte tips it over.
		stack.insert(2, 1);
		stack.update(2);

		let migrations = drain(&mut stack);

		// Key 1 is the boundary (the LRU end of the fast segment) and its
		// reference bit is clear, so it is the one that goes.
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
	#[test]
	fn a_triggered_pass_drains_to_the_low_watermark_not_the_ceiling() {
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

		// ...and stopped as soon as it got under it, rather than emptying the
		// segment: one fewer demotion would have left it above the target.
		assert!(stack.fast_used + size > low);

		// One `Tier::Slow` entry per demoted object, off the LRU end of the
		// fast segment in promotion order -- a real batch, which is the whole
		// behaviour being bought here. Draining back to the ceiling would
		// have demoted exactly one object and left the tier at `high`.
		let demoted = (filled + 1) - stack.fast_count as CacheSize;

		assert_eq!(migrations.len() as CacheSize, demoted);
		assert_eq!(migrations, (1..=demoted).map(|key| (key, Tier::Slow)).collect::<Vec<_>>());

		if high >= low + size {
			assert!(stack.fast_used < effective, "the pass must settle below the ceiling, not on it");
			assert!(migrations.len() > 1, "the pass must be a batch, not a single displacement");
		}
	}

	/// (d) Every byte and every object is still accounted for exactly once
	/// after a multi-object drain: the per-demotion bookkeeping did not
	/// change, only how many times it runs per pass.
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

		// Counts: `fast_count` covers the main queue's fast segment only,
		// `main_count` every promoted key.
		assert_eq!(stack.fast_count as CacheSize, promoted - demoted);
		assert_eq!(stack.main_count as CacheSize, promoted);
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

		// The boundary is the LRU-most still-fast key, or `None` if the pass
		// drained the fast segment completely.
		assert_eq!(
			stack.main_boundary,
			if stack.fast_count > 0 { Some(demoted + 1) } else { None },
		);

		// The midpoint cursor was maintained once per demotion, so it still
		// points at a key that really is in the slow segment.
		let midpoint = stack.slow_midpoint.expect("a pass that demoted must have seeded the cursor");

		assert_eq!(stack.tier_of(midpoint), Some(Tier::Slow));

		// And total DRAM is still within the configured fast tier.
		assert!(stack.fast_bytes_used() <= 11_000);
	}

	// ── shared DRAM-reservation overhead ──────────────────────────────────
	//
	// Every test above builds its stack with a plain `new(..)`, which leaves
	// `shared_overhead` at `0`, so `reserved_overhead()` reduces to the ghost
	// term alone -- and that term is `0` too for every one of them that
	// asserts on a capacity, a watermark or a migration: a main-queue tail
	// eviction never creates a ghost entry, and the three tests that DO age a
	// key out of the one-access queue assert only on ghost membership (plus,
	// in the readmission case, on a ten-byte fast segment three orders of
	// magnitude below any watermark, which one ghost node cannot move). So
	// `reserved_shares()` is `(0, 0)` for them and
	// `effective_main_fast_capacity`/`needs_capacity_eviction` reduce to
	// exactly the expressions they evaluated before this change -- which is
	// why none of them needed rescaling.

	/// The reservation shrinks the budget the main fast segment gets, by one
	/// `shared_overhead` per TRACKED key -- whichever queue the key is in and
	/// whichever tier its value bytes are in.
	#[test]
	fn shared_overhead_shrinks_the_effective_main_fast_capacity() {
		// `one_access_ratio` 0.0 -> `one_access_capacity` 0, so the
		// proportional split hands the whole reservation to the main segment
		// and the arithmetic below is exact.
		let mut stack = S3FifoGhostLazyDemotionFastAdmissionMidpointHybridStack::new(0.0, 10_000, 10_000)
			.with_shared_overhead(100);

		assert_eq!(stack.effective_main_fast_capacity(), 10_000);

		// A key parked in the one-access queue is charged...
		stack.insert(1, 10);

		assert_eq!(stack.len(), 1);
		assert_eq!(stack.reserved_overhead(), 100);
		assert_eq!(stack.effective_main_fast_capacity(), 9_900);

		// ...and so is one promoted into the main queue's fast segment. The
		// hashtable slot, the single list node and the `entries` slot are
		// DRAM either way, which is why the charge is per TRACKED key and not
		// per fast-tier key.
		stack.insert(2, 10);
		stack.update(2);

		assert_eq!(stack.len(), 2);
		assert_eq!(stack.tier_of(2), Some(Tier::Fast));
		assert_eq!(stack.reserved_overhead(), 200);
		assert_eq!(stack.effective_main_fast_capacity(), 9_800);

		// Dropping a key hands its share of the budget back.
		stack.remove(1);

		assert_eq!(stack.len(), 1);
		assert_eq!(stack.effective_main_fast_capacity(), 9_900);

		// `new` without the builder reserves nothing at all.
		let mut plain = S3FifoGhostLazyDemotionFastAdmissionMidpointHybridStack::new(0.0, 10_000, 10_000);

		plain.insert(1, 10);
		plain.insert(2, 10);

		assert_eq!(plain.reserved_overhead(), 0);
		assert_eq!(plain.effective_main_fast_capacity(), 10_000);
	}

	/// The model case: two stacks fed an identical sequence, differing only
	/// in whether they reserve. The reserved one demotes; the plain one does
	/// not.
	#[test]
	fn shared_overhead_reserves_dram_and_demotes_earlier() {
		const CAPACITY: CacheSize = 10_000;
		const PER_KEY: CacheSize = 1_000;

		// Two tracked keys -> a 2_000-byte reservation -> an effective main
		// budget of 8_000. Key 2 is sized to sit exactly ON the high
		// watermark of that reduced budget, so the pair (key 1 is one byte)
		// lands one byte past it -- while still sitting comfortably under the
		// high watermark of the RAW 10_000 capacity, so nothing but the
		// reservation can explain a demotion here.
		let trigger = watermarks::high_bytes(CAPACITY - 2 * PER_KEY);

		assert!(
			trigger >= 1 && trigger + 1 <= watermarks::high_bytes(CAPACITY),
			"watermark config leaves no room for this fixture",
		);

		// Without the reservation the pair fits: nothing is demoted.
		let mut plain = S3FifoGhostLazyDemotionFastAdmissionMidpointHybridStack::new(0.0, 10_000, CAPACITY);

		plain.insert(1, 1);
		plain.update(1);
		plain.insert(2, trigger as ObjectSize);
		plain.update(2);

		assert_eq!(plain.effective_main_fast_capacity(), CAPACITY);
		assert!(drain(&mut plain).is_empty(), "the unreserved budget is not crossed");
		assert_eq!(plain.tier_of(1), Some(Tier::Fast));
		assert_eq!(plain.tier_of(2), Some(Tier::Fast));
		assert_eq!(plain.fast_used, trigger + 1);

		// With it, the identical sequence demotes.
		let mut stack = S3FifoGhostLazyDemotionFastAdmissionMidpointHybridStack::new(0.0, 10_000, CAPACITY)
			.with_shared_overhead(PER_KEY);

		stack.insert(1, 1);
		stack.update(1);
		stack.insert(2, trigger as ObjectSize);
		stack.update(2);

		let migrations = drain(&mut stack);

		assert_eq!(stack.effective_main_fast_capacity(), CAPACITY - 2 * PER_KEY);
		assert!(!migrations.is_empty(), "the reservation must have triggered a pass");
		assert_eq!(migrations[0], (1, Tier::Slow), "the LRU end of the fast segment goes first");
		assert_eq!(stack.tier_of(1), Some(Tier::Slow));

		// It settled under the low watermark of the RESERVED budget, which
		// the plain stack is still sitting above.
		assert!(stack.fast_used <= watermarks::low_bytes(CAPACITY - 2 * PER_KEY));
		assert!(plain.fast_used > watermarks::low_bytes(CAPACITY - 2 * PER_KEY));

		// Values plus the whole reservation now fit inside the raw budget --
		// the point of the exercise.
		assert!(stack.fast_used + 2 * PER_KEY <= CAPACITY);
	}

	/// Composition order: the reservation comes out first and the watermarks
	/// are taken of the REMAINDER, so a triggered pass drains to
	/// `low_bytes(capacity - reserved)` and not to `low_bytes(capacity)`.
	#[test]
	fn a_triggered_pass_drains_to_the_low_watermark_of_the_reserved_budget() {
		const CAPACITY: CacheSize = 10_000;
		const PER_KEY: CacheSize = 200;
		const KEYS: CacheSize = 20;
		const SIZE: CacheSize = 500;

		let mut stack = S3FifoGhostLazyDemotionFastAdmissionMidpointHybridStack::new(0.0, 20_000, CAPACITY)
			.with_shared_overhead(PER_KEY);

		// Admit all 20 keys FIRST -- a plain `insert` parks them in the
		// one-access queue and never settles the main queue -- so the tracked
		// count, and with it the 20 x 200 = 4_000-byte reservation and the
		// 6_000-byte effective budget, is already final before the first
		// promotion. Every watermark below is therefore a fixed byte count
		// rather than a moving target.
		for key in 1..=KEYS {
			stack.insert(key, SIZE as ObjectSize);
		}

		let effective = stack.effective_main_fast_capacity();

		assert_eq!(stack.len() as CacheSize, KEYS);
		assert_eq!(effective, CAPACITY - KEYS * PER_KEY);
		assert!(drain(&mut stack).is_empty(), "admission alone migrates nothing");

		let high = watermarks::high_bytes(effective);
		let low = watermarks::low_bytes(effective);
		let filled = high / SIZE;

		assert!(
			filled >= 1
				&& filled + 1 <= KEYS
				&& (filled + 1) * SIZE <= watermarks::high_bytes(CAPACITY)
				&& low < watermarks::low_bytes(CAPACITY),
			"watermark config leaves no room for this fixture",
		);

		// Promote the largest whole number of keys that still sits at or
		// below the high watermark OF THE RESERVED BUDGET: no pass yet.
		for key in 1..=filled {
			stack.update(key);
		}

		assert_eq!(stack.fast_used, filled * SIZE);
		assert!(drain(&mut stack).is_empty(), "at or below the high watermark must not trigger");

		// One more promotion tips it past that watermark.
		stack.update(filled + 1);

		let migrations = drain(&mut stack);
		let demoted = migrations.len() as CacheSize;

		assert!(demoted >= 1, "crossing the high watermark must trigger a pass");

		// The pass drained to `low_bytes(effective)` and stopped the instant
		// it got under it -- one 500-byte object short of overshooting.
		assert!(stack.fast_used <= low);
		assert!(stack.fast_used + SIZE > low);
		assert_eq!(stack.fast_used, (filled + 1 - demoted) * SIZE);

		// And emphatically NOT to `low_bytes(CAPACITY)`: against the raw
		// capacity this fill is not even past the HIGH watermark, so a pass
		// composed the other way round would have demoted nothing at all.
		assert!(low < watermarks::low_bytes(CAPACITY));
		assert!(stack.fast_used < watermarks::low_bytes(CAPACITY));

		// Demotions come off the LRU end of the fast segment, in promotion
		// order.
		assert_eq!(migrations, (1..=demoted).map(|key| (key, Tier::Slow)).collect::<Vec<_>>());
	}

	/// Every byte and every object is still accounted for exactly once after
	/// a pass triggered against the reserved budget -- and the reservation
	/// itself is unmoved by it, since demotion retags rather than untracks.
	#[test]
	fn counters_stay_consistent_after_a_reserved_watermark_pass() {
		const CAPACITY: CacheSize = 10_000;
		const PER_KEY: CacheSize = 200;
		const KEYS: CacheSize = 20;
		const SIZE: CacheSize = 500;

		let mut stack = S3FifoGhostLazyDemotionFastAdmissionMidpointHybridStack::new(0.0, 20_000, CAPACITY)
			.with_shared_overhead(PER_KEY);

		for key in 1..=KEYS {
			stack.insert(key, SIZE as ObjectSize);
		}

		let effective = stack.effective_main_fast_capacity();

		assert_eq!(effective, CAPACITY - KEYS * PER_KEY);

		let promoted = watermarks::high_bytes(effective) / SIZE + 1;

		assert!(
			promoted >= 2 && promoted <= KEYS,
			"watermark config leaves no room for this fixture",
		);

		for key in 1..=promoted {
			stack.update(key);
		}

		let migrations = drain(&mut stack);
		let demoted = migrations.len() as CacheSize;

		assert!(demoted >= 1, "the pass must have run");
		assert!(migrations.iter().all(|(_, tier)| *tier == Tier::Slow));

		// Bytes: every promoted key is main-fast or main-slow; the rest are
		// still parked in the one-access queue, untouched by the pass.
		assert_eq!(stack.fast_used, (promoted - demoted) * SIZE);
		assert_eq!(stack.slow_used, demoted * SIZE);
		assert_eq!(stack.one_access_used, (KEYS - promoted) * SIZE);
		assert_eq!(stack.fast_used + stack.slow_used, promoted * SIZE);

		// Counts: `fast_count` covers the main queue's fast segment only,
		// `main_count` every promoted key, `len()` everything tracked.
		assert_eq!(stack.fast_count as CacheSize, promoted - demoted);
		assert_eq!(stack.main_count as CacheSize, promoted);
		assert_eq!(stack.len() as CacheSize, KEYS);

		// Reported gauges add the one-access queue back onto the fast side.
		assert_eq!(
			stack.fast_bytes_used(),
			(promoted - demoted) * SIZE + (KEYS - promoted) * SIZE,
		);
		assert_eq!(stack.slow_bytes_used(), demoted * SIZE);
		assert_eq!(stack.fast_object_count() as CacheSize, KEYS - demoted);
		assert_eq!(stack.slow_object_count() as CacheSize, demoted);

		// The reservation is exactly what it was before the pass: a demotion
		// retags a tracked key, it never drops one.
		assert_eq!(stack.reserved_overhead(), KEYS * PER_KEY);
		assert_eq!(stack.effective_main_fast_capacity(), effective);

		// The boundary is the LRU-most still-fast key, or `None` if the pass
		// drained the fast segment completely.
		assert_eq!(
			stack.main_boundary,
			if stack.fast_count > 0 { Some(demoted + 1) } else { None },
		);

		// The midpoint cursor was maintained once per demotion, so it still
		// points at a key that really is in the slow segment.
		let midpoint = stack.slow_midpoint.expect("a pass that demoted must have seeded the cursor");

		assert_eq!(stack.tier_of(midpoint), Some(Tier::Slow));

		// The main fast segment's values plus the WHOLE shared reservation
		// still fit inside `fast_capacity`.
		assert!(stack.fast_used + KEYS * PER_KEY <= CAPACITY);
	}

	/// The reservation is split between the two independently-capacitied fast
	/// segments in proportion to their capacities -- never charged in full to
	/// each.
	#[test]
	fn shared_overhead_splits_proportionally_between_the_two_fast_segments() {
		// `watermark_stack`'s rig: a 1_000-byte one-access reservation
		// (0.1 x max_size 10_000) carved out of an 11_000-byte fast tier
		// leaves the main queue a round 10_000. Eleven tracked keys at 100
		// bytes/key is a 1_100-byte reservation, split 1_000 : 10_000 by
		// capacity -> 100 to the one-access queue, 1_000 to the main segment.
		let mut stack = S3FifoGhostLazyDemotionFastAdmissionMidpointHybridStack::new(0.1, 10_000, 11_000)
			.with_shared_overhead(100);
		let mut plain = S3FifoGhostLazyDemotionFastAdmissionMidpointHybridStack::new(0.1, 10_000, 11_000);

		for key in 1..=11u64 {
			stack.insert(key, 82);
			plain.insert(key, 82);
		}

		assert_eq!(stack.len(), 11);
		assert_eq!(stack.reserved_overhead(), 1_100);
		assert_eq!(stack.reserved_shares(), (100, 1_000));
		assert_eq!(stack.effective_one_access_capacity(), 900);
		assert_eq!(stack.effective_main_fast_capacity(), 9_000);

		// Split, not double-charged: the two effective budgets plus the ONE
		// reservation add back up to the whole fast tier. Charging 1_100 to
		// each would have left 8_900 + 900 = 9_800 usable, throwing away
		// 1_100 bytes of real budget.
		assert_eq!(
			stack.effective_one_access_capacity() + stack.effective_main_fast_capacity() + 1_100,
			11_000,
		);

		// The one-access queue's own capacity trigger sees its share too:
		// 11 x 82 = 902 bytes is under the raw 1_000-byte budget but over the
		// 900-byte effective one.
		assert_eq!(stack.one_access_used, 902);
		assert!(stack.needs_capacity_eviction());

		// The same 902 bytes against an unreserved stack: no eviction.
		assert_eq!(plain.effective_one_access_capacity(), 1_000);
		assert_eq!(plain.effective_main_fast_capacity(), 10_000);
		assert!(!plain.needs_capacity_eviction());
	}

	/// The ghost queue's DRAM is a separate term, charged by real ghost
	/// length -- it cannot be folded into the per-tracked-key constant
	/// because a ghost entry exists precisely when the tracked entry does
	/// not.
	///
	/// Stated against [`GHOST_ENTRY_DRAM_OVERHEAD`] itself rather than against
	/// its current value, so it stays correct under `eviction_stacks_pmem`,
	/// where the ghost list is not DRAM-resident and the constant is `0`.
	#[test]
	fn ghost_queue_overhead_is_a_separate_term_charged_by_real_ghost_length() {
		const GHOST: CacheSize = GHOST_ENTRY_DRAM_OVERHEAD as CacheSize;

		// `one_access_ratio` 0.0 again, so the whole reservation lands on the
		// main segment and both terms are individually visible.
		let mut stack = S3FifoGhostLazyDemotionFastAdmissionMidpointHybridStack::new(0.0, 10_000, 10_000)
			.with_shared_overhead(100);

		stack.insert(1, 10);
		stack.insert(2, 10);

		assert_eq!(stack.len(), 2);
		assert_eq!(stack.reserved_overhead(), 200);
		assert_eq!(stack.effective_main_fast_capacity(), 9_800);

		// Ageing key 1 out of the one-access queue drops its tracked entry
		// (-100: gone from `entries`, gone from the hashtable) and creates a
		// bare ghost key (+GHOST_ENTRY_DRAM_OVERHEAD). The two terms move
		// independently, in opposite directions, on a single event.
		assert_eq!(stack.evict_one(), Some(1));
		assert!(stack.is_ghost(1));

		assert_eq!(stack.len(), 1);
		assert_eq!(stack.reserved_overhead(), 100 + GHOST);
		assert_eq!(stack.effective_main_fast_capacity(), 10_000 - (100 + GHOST));

		// Ageing key 2 out too: no tracked key is left, and the reservation is
		// now purely the ghost term, scaling with the queue's real length.
		assert_eq!(stack.evict_one(), Some(2));
		assert!(stack.is_ghost(2));

		assert_eq!(stack.len(), 0);
		assert_eq!(stack.reserved_overhead(), 2 * GHOST);
		assert_eq!(stack.effective_main_fast_capacity(), 10_000 - 2 * GHOST);
	}

	/// The regression the old builder-injected `ghost_overhead` allowed: it
	/// defaulted to `0`, so a stack whose construction site forgot the builder
	/// reserved nothing at all for its ghost queue. The ghost nodes occupy
	/// DRAM whether or not the per-tracked-key term happens to be configured,
	/// so the two are charged independently -- `shared_overhead == 0` must not
	/// zero the ghost term.
	#[test]
	fn the_ghost_term_is_charged_even_when_shared_overhead_is_zero() {
		const GHOST: CacheSize = GHOST_ENTRY_DRAM_OVERHEAD as CacheSize;

		// Plain `new(..)`: `shared_overhead` is `0`, exactly like a stack
		// built without `with_shared_overhead`.
		let mut stack = S3FifoGhostLazyDemotionFastAdmissionMidpointHybridStack::new(0.0, 10_000, 10_000);

		stack.insert(1, 10);
		stack.insert(2, 10);

		// Nothing is reserved for the two tracked keys...
		assert_eq!(stack.len(), 2);
		assert_eq!(stack.reserved_overhead(), 0);
		assert_eq!(stack.effective_main_fast_capacity(), 10_000);

		// ...but ageing one out into the ghost queue still is.
		assert_eq!(stack.evict_one(), Some(1));
		assert!(stack.is_ghost(1));

		assert_eq!(stack.len(), 1);
		assert_eq!(stack.reserved_overhead(), GHOST);
		assert_eq!(stack.effective_main_fast_capacity(), 10_000 - GHOST);

		// And it keeps scaling with the ghost queue, not with the tracked-key
		// count -- which is `0` by the time the second ghost exists.
		assert_eq!(stack.evict_one(), Some(2));
		assert!(stack.is_ghost(2));

		assert_eq!(stack.len(), 0);
		assert_eq!(stack.reserved_overhead(), 2 * GHOST);
		assert_eq!(stack.effective_main_fast_capacity(), 10_000 - 2 * GHOST);
	}

	/// A reservation that swallows the whole fast budget demotes everything
	/// but never evicts anything out of the main queue -- the DRAM budget is
	/// a demotion target, not a data-dropping ceiling.
	#[test]
	fn shared_overhead_exceeding_capacity_demotes_all_but_never_evicts() {
		let mut stack = S3FifoGhostLazyDemotionFastAdmissionMidpointHybridStack::new(0.0, 10_000, 50)
			.with_shared_overhead(100);

		stack.insert(1, 10);
		stack.update(1);

		let migrations = drain(&mut stack);

		assert_eq!(stack.effective_main_fast_capacity(), 0, "saturates rather than underflowing");
		assert_eq!(migrations, vec![(1, Tier::Slow)]);
		assert_eq!(stack.tier_of(1), Some(Tier::Slow));
		assert_eq!(stack.fast_used, 0);

		// Still tracked: demotion is the only response the budget has.
		assert_eq!(stack.len(), 1);
		assert!(stack.contains(1));
	}

	/// `evict_one`'s one-access-tail priority is CONDITIONAL on the main queue
	/// having room, exactly as `SThreeFifoStack::evict_one`'s small-queue
	/// priority is -- see the module doc's "Eviction order" section.
	///
	/// Both halves have the same shape: one key promoted into the main queue,
	/// one key left sitting in the one-access queue, one `evict_one` call. What
	/// separates them is the promoted key's size, which is what moves the main
	/// queue from "has room" to "full"; the one-access key's size only varies to
	/// keep half (a) sensitive to a gate that double-counts it.
	#[test]
	fn one_access_tail_is_evicted_first_only_while_the_main_queue_has_room() {
		// ratio 0.5 of max_size 100 -> one_access_capacity 50, main_capacity 50.
		// `fast_capacity` is far larger than either so no demotion pass can fire
		// and muddy the trace; the gate reads whole-queue bytes anyway, so a
		// demotion would not move it either way.

		// (a) Main queue below its budget: the one-access tail goes first. The
		// one-access resident is deliberately fat enough (45 of a 50-byte main
		// budget) that a gate reading `fast_bytes_used()` -- which folds
		// `one_access_used` in -- would wrongly see 10 + 45 >= 50 and call the
		// main queue full.
		let mut roomy = S3FifoGhostLazyDemotionFastAdmissionMidpointHybridStack::new(0.5, 100, 10_000);

		roomy.insert(1, 10);
		roomy.update(1); // promotes key 1 out of the one-access queue into main
		roomy.insert(2, 45); // key 2 stays in the one-access queue
		drain(&mut roomy);

		assert!(
			!roomy.is_main_full(),
			"10 bytes of MAIN queue against a 50-byte budget -- the 45 one-access bytes are not the main queue's",
		);

		assert_eq!(roomy.evict_one(), Some(2), "the one-access tail must go first");
		assert!(roomy.is_ghost(2), "an aged-out one-access key becomes a ghost entry");
		assert!(roomy.contains(1), "the main queue must be left alone");

		// (b) Same shape, but the promoted key alone fills main_capacity, so the
		// one-access tail is off limits and the main queue is evicted instead.
		let mut full = S3FifoGhostLazyDemotionFastAdmissionMidpointHybridStack::new(0.5, 100, 10_000);

		full.insert(1, 50);
		full.update(1);
		full.insert(2, 10);
		drain(&mut full);

		assert!(full.is_main_full(), "50 bytes of main queue against a 50-byte budget");

		assert_eq!(
			full.evict_one(),
			Some(1),
			"a full main queue must be evicted before the one-access tail",
		);
		assert!(full.contains(2), "the one-access resident must survive");
		assert!(!full.is_ghost(1), "a main-queue eviction does not create a ghost entry");
	}
}
