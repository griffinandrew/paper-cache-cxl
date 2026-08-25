/*
 * Copyright (c) Kia Shakiba
 *
 * This source code is licensed under the GNU AGPLv3 license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! `TwoQGhostHybridStack` — `TwoQHybridStack` plus a bare-key ghost queue,
//! for `PaperPolicy::TwoQGhostHybrid`.
//!
//! Identical to `TwoQHybridStack` in every other respect (see that stack's
//! module doc for the full admission/demotion/promotion/eviction rules) —
//! this file only adds what `TwoQHybridStack`'s own module doc flagged as
//! deliberately left out: a ghost queue remembering keys that aged out of
//! `fifo_queue` without a second access, so a later re-admission can be
//! trusted immediately instead of restarting from zero.
//!
//! Mirrors `s_three_fifo_stack.rs`'s existing `ghost: HashList<HashedKey>`
//! shape exactly — a bare key list, no object data, chosen over plain
//! `TwoQStack`'s heavier `a1_out` (which holds real live objects) per an
//! explicit user decision: a lightweight membership list, not a third place
//! actual bytes can live.
//!
//! ## Ghost lifecycle, matching `SThreeFifoStack`'s existing convention
//!
//! * **Added to** only by `evict_fifo_tail` (a `fifo_queue` object aging out
//!   without a second access) — never by a main-queue eviction.
//! * **Checked** by `insert`'s brand-new-key branch, before falling back to
//!   the normal `fifo_queue` admission.
//! * **Not removed immediately on a hit** — same lazy convention
//!   `SThreeFifoStack` already uses (see that file's
//!   `no_ghost_entry_routes_fresh_insertion_to_small_queue` test doc for the
//!   precedent). Only trimmed lazily, capped relative to `main_count`,
//!   during a genuine main-queue eviction (never during a `fifo_queue`
//!   eviction, which is what populates it) — and cleared outright by
//!   `remove`/`clear`.
//!
//! ## Where a ghost hit lands: fast tier, deliberately reversible
//!
//! A ghost hit is admitted directly into `main_stack` at `Tier::Fast` —
//! `admit_via_ghost_hit`, structurally identical to `promote_from_fifo`
//! minus the "remove from `fifo_queue`" step (the key was never there this
//! time). This was an explicit, acknowledged-as-arguable choice: the
//! alternative (land in the *slow* portion of `main_stack`, still having to
//! earn fast-tier promotion via a subsequent real access, the more
//! conservative reading) was flagged as possibly better and left as a
//! one-line change here (swap `Tier::Fast` for `Tier::Slow` and drop the
//! `settle_fast_tier`/fast-tier bookkeeping in `admit_via_ghost_hit`) if
//! real measurement says otherwise. Physically cheap either way: the
//! API-layer `set()` always builds a brand-new key as `TieredBuffer::
//! new_slow` regardless of ghost history (this stack has no equivalent of
//! `LfuHybridStack`'s admission-latch mirror onto `AtomicStatus` — a ghost
//! hit is corrected to `Fast` via the ordinary async migration path, the
//! same one every other promotion in this stack already uses), so a ghost
//! hit costs exactly one extra migration, not a synchronous PMEM-vs-DRAM
//! choice at the API layer.
//!
//! ## Shared-metadata DRAM reservation
//!
//! Like `LruHybridStack`, the fast-tier budget this stack settles against is
//! a *DRAM* budget, not merely a fast-tier-value budget: the shared object
//! hashtable and this stack's own bookkeeping (`fifo_queue`/`main_stack`/
//! `entries`) also live in DRAM and are invisible to `fast_used`.
//! `with_shared_overhead` wires in the per-tracked-object cost of those
//! structures (`crate::object::overhead::get_hybrid_dram_shared_overhead`,
//! which is also where the DRAM-vs-PMEM gating for them lives);
//! `reserved_overhead` charges it against *every* tracked key -- a
//! `fifo_queue` key included, since its `entries` row and its list node are
//! DRAM-resident even though its bytes are slow-tier -- and
//! `settle_fast_tier` subtracts the total from `fast_capacity` *before* the
//! watermarks are applied.
//!
//! There is exactly **one** fast segment here, so the reservation is charged
//! whole rather than split proportionally the way `LruSizedHybridStack` (two
//! independently-capacitied fast segments) has to. `fifo_capacity` is not a
//! second fast segment: it bounds `fifo_queue`, which is slow-tier
//! throughout -- `insert` tags a fresh key `TwoQEntry { dram_resident, queue: Queue::Fifo,
//! tier: None, .. }`, `tier_of` reports such a key as `Tier::Slow`, and
//! `slow_bytes_used` is `fifo_used + slow_used`.
//!
//! ### The ghost queue is a separate term
//!
//! `ghost` holds *bare keys* for objects that are no longer in the cache and
//! have no `entries` row at all, so its DRAM cost cannot be expressed as a
//! per-tracked-key constant -- it scales with `ghost.len()`, which
//! `trim_ghost` bounds by `main_count`, and only lazily (a run of
//! `fifo_queue` evictions, which is what *populates* `ghost`, never trims
//! it). It is therefore charged as its own term -- `ghost.len()` ×
//! [`crate::object::overhead::GHOST_ENTRY_DRAM_OVERHEAD`] -- rather than
//! folded into `shared_overhead`. A key admitted by `admit_via_ghost_hit` is
//! charged for both terms at once, which is accurate: under the lazy-trim
//! convention it really does occupy `entries` + `main_stack` *and* `ghost`
//! until the next `trim_ghost`.
//!
//! That per-ghost-entry cost is a fixed crate constant rather than something
//! the caller configures: every ghost-keeping hybrid stack stores the same
//! `HashList<HashedKey>` node, so there is nothing per-deployment to wire in.
//! The two terms are also *independent* -- `reserved_overhead` charges the
//! ghost term whether or not `with_shared_overhead` was ever called, because
//! ghost nodes occupy DRAM regardless of how (or whether) the per-tracked-key
//! term happens to be configured.
//!
//! ## `eviction_stacks_pmem`
//!
//! `ghost` follows the same DRAM/PMEM switch as `fifo_queue`/`main_stack`/
//! `entries` — see `TwoQHybridStack`'s module doc. That switch is why
//! [`crate::object::overhead::GHOST_ENTRY_DRAM_OVERHEAD`] is itself
//! `cfg`-selected -- on `eviction_stacks_pmem` alone, never on the hashtable
//! feature, since a ghost key has no hashtable slot to charge for: when the
//! eviction stacks are in PMEM, a ghost entry costs the fast tier nothing.

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
	Fifo,
	Main,
}

/// Combined per-key bookkeeping — see `TwoQHybridStack`'s "One combined
/// per-key map" module doc section.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct TwoQEntry {queue: Queue,
	tier: Option<Tier>,
	/// Part of `size` that stays in DRAM in either tier; see `migrating`.
	dram_resident: u8,

	size: ObjectSize,
}

impl TwoQEntry {
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
	std::mem::size_of::<TwoQEntry>() == 8,
	"TwoQEntry grew past 8 bytes",
);


#[cfg(not(feature = "eviction_stacks_pmem"))]
type QueueList = HashList<HashedKey, NoHasher>;
#[cfg(feature = "eviction_stacks_pmem")]
type QueueList = PmemHashList<HashedKey, NoHasher>;

#[cfg(not(feature = "eviction_stacks_pmem"))]
type EntryMap = HashMap<HashedKey, TwoQEntry, NoHasher>;
#[cfg(feature = "eviction_stacks_pmem")]
type EntryMap = HashMap<HashedKey, TwoQEntry, NoHasher, Hybrid>;

pub struct TwoQGhostHybridStack {
	fifo_queue: QueueList,
	main_stack: QueueList,
	ghost: QueueList,

	entries: EntryMap,

	k_in: f64,
	fifo_capacity: CacheSize,
	fifo_used: CacheSize,

	fast_capacity: CacheSize,
	fast_used: CacheSize,
	slow_used: CacheSize,

	/// Approximate per-tracked-object DRAM cost of the shared structures
	/// (object hashtable + this stack's eviction bookkeeping) that hold an
	/// entry for every tracked object of both tiers. Reserved out of
	/// `fast_capacity` in `settle_fast_tier` so the fast-tier budget bounds
	/// total DRAM (values + shared metadata), not just fast-tier values. `0`
	/// unless set via `with_shared_overhead` (so unit tests exercising the
	/// pure value-budget behaviour are unaffected). That default zeroes only
	/// *this* term: `reserved_overhead`'s ghost term comes from a shared crate
	/// constant and is charged either way.
	shared_overhead: CacheSize,

	/// Number of keys currently tagged `Tier::Fast` within `main_stack`.
	fast_count: usize,

	/// Number of keys currently in the `Main` queue (Fast or Slow). Also
	/// used as the ghost list's size cap reference (mirrors
	/// `SThreeFifoStack`'s `ghost.len() > main.stack.len()` bound).
	main_count: usize,

	/// The least-recently-used key currently tagged `Tier::Fast` within
	/// `main_stack` — the next demotion candidate. Mirrors
	/// `LruHybridStack::fast_boundary`.
	main_boundary: Option<HashedKey>,

	/// (key, new tier) pairs recorded since the last `drain_tier_migrations`.
	migrations: Vec<(HashedKey, Tier)>,
}

impl TwoQGhostHybridStack {
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

	pub fn new(k_in: f64, max_size: CacheSize, fast_capacity: CacheSize) -> Self {
		let (fifo_queue, main_stack, ghost, entries) = Self::new_collections();

		TwoQGhostHybridStack {
			fifo_queue,
			main_stack,
			ghost,

			entries,

			k_in,
			fifo_capacity: (k_in * max_size as f64) as CacheSize,
			fifo_used: 0,

			fast_capacity,
			fast_used: 0,
			slow_used: 0,
			shared_overhead: 0,
			fast_count: 0,
			main_count: 0,

			main_boundary: None,
			migrations: Vec::new(),
		}
	}

	/// Sets the approximate per-tracked-object shared-structure DRAM overhead
	/// (object hashtable + eviction stacks) reserved out of the fast-tier
	/// budget. See `crate::object::overhead::get_hybrid_dram_shared_overhead`,
	/// which is where the DRAM-vs-PMEM gating of those terms lives.
	/// Builder-style so `init_policy_stack` can wire it in without disturbing
	/// `new`'s signature (unit tests keep the default `0`).
	pub fn with_shared_overhead(mut self, overhead: CacheSize) -> Self {
		self.shared_overhead = overhead;
		self
	}

	/// Total DRAM currently reserved for shared metadata, subtracted from
	/// `fast_capacity` by `effective_fast_capacity`. Two terms:
	///
	/// * every *tracked* key's share of the object hashtable and of this
	///   stack's own bookkeeping (`entries.len() × shared_overhead`) --
	///   charged for `fifo_queue` keys too, since their `entries` row and
	///   their list node are DRAM-resident even though their bytes are
	///   slow-tier (this matches `LruHybridStack`'s `stack.len()` and
	///   `LruSizedHybridStack`'s `entries.len()`); and
	/// * every *ghost* key's bare-key list node (`ghost.len()` ×
	///   [`GHOST_ENTRY_DRAM_OVERHEAD`]) -- a term no per-tracked-key constant
	///   can express, because a ghost key has no `entries` row at all (and no
	///   object-hashtable slot either, which is why that shared constant is
	///   gated on `eviction_stacks_pmem` alone). See the module doc's "The
	///   ghost queue is a separate term".
	///
	/// A key sits in exactly *one* of `fifo_queue`/`main_stack` at a time
	/// (`promote_from_fifo` removes it from the former before pushing it onto
	/// the latter; a demotion only retags `TwoQEntry::tier` and leaves the key
	/// in `main_stack`), so the per-key term charges one list node, not two.
	///
	/// The two terms are independent, and deliberately so: the ghost term is
	/// charged from the shared crate constant on every call, including when
	/// `shared_overhead` is still at its `0` default (a stack built without
	/// `with_shared_overhead`). Ghost nodes occupy DRAM whether or not the
	/// per-tracked-key term happens to be configured, so coupling the two --
	/// as an `if self.shared_overhead == 0 { return 0; }` early return once
	/// did -- silently under-reserved for them.
	fn reserved_overhead(&self) -> CacheSize {
		self.entries.len() as CacheSize * self.shared_overhead
			+ self.ghost.len() as CacheSize * (GHOST_ENTRY_DRAM_OVERHEAD as CacheSize)
	}

	/// `fast_capacity` minus [`Self::reserved_overhead`] -- the effective
	/// value-byte budget the watermarks are applied to. Saturates to `0` when
	/// the shared metadata alone meets or exceeds the whole fast-tier budget.
	fn effective_fast_capacity(&self) -> CacheSize {
		self.fast_capacity.saturating_sub(self.reserved_overhead())
	}

	/// Returns which queue/tier the given (currently tracked) key is in, or
	/// `None` if the key isn't tracked. Exposed for tests/diagnostics.
	pub fn tier_of(&self, key: HashedKey) -> Option<Tier> {
		let entry = self.entries.get(&key)?;

		match entry.queue {
			Queue::Fifo => Some(Tier::Slow),
			Queue::Main => entry.tier,
		}
	}

	/// Returns `true` if `key` currently has a ghost entry. Exposed for tests.
	pub fn is_ghost(&self, key: HashedKey) -> bool {
		self.ghost.contains(&key)
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

	fn touch(&mut self, key: HashedKey) {
		match self.entries.get(&key).map(|entry| entry.queue) {
			Some(Queue::Fifo) => self.promote_from_fifo(key),
			Some(Queue::Main) => self.touch_main_fast(key),
			None => {},
		}
	}

	fn promote_from_fifo(&mut self, key: HashedKey) {
		let Some(entry) = self.entries.get(&key) else { return };
		let size = entry.size;
		let dram_resident = entry.dram_resident;
		// Tier arithmetic moves only what migrates; `size` still rebuilds the entry.
		let size_bytes = entry.migrating();

		self.fifo_queue.remove(&key);
		self.fifo_used = self.fifo_used.saturating_sub(size_bytes);

		self.main_stack.push_front(key);
		self.entries.insert(key, TwoQEntry { dram_resident, queue: Queue::Main, tier: Some(Tier::Fast), size });
		self.fast_used += size_bytes;
		self.fast_count += 1;
		self.main_count += 1;

		if self.main_boundary.is_none() {
			self.main_boundary = Some(key);
		}

		self.settle_fast_tier();

		if self.entries.get(&key).and_then(|entry| entry.tier) == Some(Tier::Fast) {
			self.migrations.push((key, Tier::Fast));
		}
	}

	/// Admits a brand-new key directly into `main_stack` at `Tier::Fast` —
	/// the ghost-hit path. Structurally identical to `promote_from_fifo`
	/// minus the "remove from `fifo_queue`" step, since the key was never
	/// there this time. See the module doc's "Where a ghost hit lands"
	/// section for why `Tier::Fast` specifically, and how to flip it.
	fn admit_via_ghost_hit(&mut self, key: HashedKey, size: ObjectSize, dram_resident: u8) {
		self.main_stack.push_front(key);
		self.entries.insert(key, TwoQEntry { dram_resident, queue: Queue::Main, tier: Some(Tier::Fast), size });
		self.fast_used += (size as CacheSize).saturating_sub(dram_resident as CacheSize);
		self.fast_count += 1;
		self.main_count += 1;

		if self.main_boundary.is_none() {
			self.main_boundary = Some(key);
		}

		self.settle_fast_tier();

		if self.entries.get(&key).and_then(|entry| entry.tier) == Some(Tier::Fast) {
			self.migrations.push((key, Tier::Fast));
		}
	}

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
				let size = self.entries.get(&key).map(|entry| entry.migrating()).unwrap_or(0);

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

		if promoted && self.entries.get(&key).and_then(|entry| entry.tier) == Some(Tier::Fast) {
			self.migrations.push((key, Tier::Fast));
		}
	}

	/// Demotes `main_stack`'s LRU-most fast-tier keys until the fast tier is
	/// back under the shared *low* watermark -- but only once usage has
	/// crossed the shared *high* watermark in the first place.
	///
	/// The ceiling this stack works against is `effective_fast_capacity()` --
	/// `fast_capacity` minus the DRAM reserved for shared per-object metadata
	/// (hashtable + eviction bookkeeping + ghost nodes) across both tiers,
	/// saturating to `0` when that metadata alone meets or exceeds
	/// `fast_capacity`. This is what makes the fast-tier budget bound total
	/// DRAM rather than just fast-tier values. There is only one fast segment
	/// to charge it to: unlike the `fast_admission` variants there is no
	/// one-access queue carved out of the same DRAM budget (`fifo_queue` here
	/// is slow-tier and bounded separately by `fifo_capacity`), so unlike
	/// `LruSizedHybridStack` there is no proportional split to do.
	///
	/// The `watermarks` helpers are applied *on top of* that effective value,
	/// never in place of it -- the reservation sets the ceiling; the
	/// watermarks decide only when a pass fires and how far it drains.
	///
	/// Previously this drained to exactly `fast_capacity`, which pinned the
	/// tier at 100% utilisation and made essentially every admission demote
	/// exactly one object (see the `watermarks` module doc). Setting both
	/// `FAST_TIER_HIGH_WATERMARK` and `FAST_TIER_LOW_WATERMARK` to `1.0`
	/// restores that behaviour byte-for-byte.
	///
	/// Per-demotion bookkeeping is deliberately untouched: each demoted
	/// object still retags its entry, still moves `fast_used`/`fast_count`/
	/// `slow_used` by its own size, still walks `main_boundary` one step
	/// toward the front, and still emits exactly one `Tier::Slow` migration.
	fn settle_fast_tier(&mut self) {
		// Reservation first, watermarks on top of what it leaves behind.
		let effective_capacity = self.effective_fast_capacity();

		if self.fast_used <= watermarks::high_bytes(effective_capacity) {
			return;
		}

		let drain_target = watermarks::low_bytes(effective_capacity);

		while self.fast_used > drain_target {
			let Some(demote_key) = self.main_boundary else { break };

			let size = self.entries.get(&demote_key).map(|entry| entry.migrating()).unwrap_or(0);
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

	/// Pops `fifo_queue`'s tail, removes it from this stack's own
	/// bookkeeping, and remembers it in `ghost` — the "aged out without a
	/// second access" case. Same "cannot self-evict from insert/resize"
	/// rationale as `TwoQHybridStack::evict_fifo_tail` — only called from
	/// `evict_one`.
	fn evict_fifo_tail(&mut self) -> Option<HashedKey> {
		let key = self.fifo_queue.pop_back()?;
		let size = self.entries.remove(&key).map(|entry| entry.migrating()).unwrap_or(0);

		self.fifo_used = self.fifo_used.saturating_sub(size);
		self.ghost.push_front(key);

		Some(key)
	}

	/// Trims `ghost` down to `main_count` entries, oldest first — called
	/// only from a genuine main-queue eviction (never from
	/// `evict_fifo_tail`, which is what populates `ghost` in the first
	/// place). Mirrors `SThreeFifoStack::evict_main`'s `while self.ghost.
	/// len() > self.main.stack.len()` cap exactly, using `main_count` as
	/// the size reference this stack already tracks.
	fn trim_ghost(&mut self) {
		while self.ghost.len() > self.main_count {
			self.ghost.pop_back();
		}
	}
}

impl PolicyStack for TwoQGhostHybridStack {
	fn is_policy(&self, policy: &PaperPolicy) -> bool {
		matches!(policy, PaperPolicy::TwoQGhostHybrid(k_in) if *k_in == self.k_in)
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

		self.fifo_queue.push_front(key);
		self.entries.insert(key, TwoQEntry { dram_resident, queue: Queue::Fifo, tier: None, size });
		self.fifo_used += (size as CacheSize).saturating_sub(dram_resident as CacheSize);
	}

	fn update(&mut self, key: HashedKey) {
		if self.entries.contains_key(&key) {
			self.touch(key);
		}
	}

	fn remove(&mut self, key: HashedKey) {
		// Unconditional and first: a key evicted from `fifo_queue` (via
		// `evict_fifo_tail`) has *already* been removed from `entries` by
		// the time it lives only in `ghost` -- gating this on
		// `entries.remove` succeeding (as the rest of this method's logic
		// legitimately does) would silently skip clearing a stale ghost
		// entry for exactly that case. Mirrors `SThreeFifoStack::remove`,
		// which also clears its ghost queue unconditionally.
		self.ghost.remove(&key);

		let Some(entry) = self.entries.remove(&key) else { return };
		let size = entry.migrating();

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
	}

	fn clear(&mut self) {
		self.fifo_queue.clear();
		self.main_stack.clear();
		self.ghost.clear();
		self.entries.clear();

		self.fifo_used = 0;
		self.fast_used = 0;
		self.slow_used = 0;
		self.fast_count = 0;
		self.main_count = 0;
		self.main_boundary = None;
		self.migrations.clear();
	}

	fn evict_one(&mut self) -> Option<HashedKey> {
		if let Some(key) = self.evict_fifo_tail() {
			return Some(key);
		}

		let key = self.main_stack.pop_back()?;
		let removed = self.entries.remove(&key);
		let size = removed.map(|entry| entry.migrating()).unwrap_or(0);
		let tier = removed.and_then(|entry| entry.tier);

		self.main_count = self.main_count.saturating_sub(1);

		match tier {
			Some(Tier::Fast) => {
				self.fast_used = self.fast_used.saturating_sub(size);
				self.fast_count = self.fast_count.saturating_sub(1);

				if self.main_boundary == Some(key) {
					self.main_boundary = self.main_stack.back().copied();
				}
			},

			Some(Tier::Slow) => {
				self.slow_used = self.slow_used.saturating_sub(size);
			},

			None => {},
		}

		self.trim_ghost();

		Some(key)
	}

	fn resize_fast_tier(&mut self, size: CacheSize) {
		self.fast_capacity = size;
		self.settle_fast_tier();
	}

	fn drain_tier_migrations(&mut self) -> Vec<(HashedKey, Tier)> {
		std::mem::take(&mut self.migrations)
	}

	fn fast_bytes_used(&self) -> CacheSize {
		self.fast_used
	}

	fn slow_bytes_used(&self) -> CacheSize {
		self.fifo_used + self.slow_used
	}

	fn fast_object_count(&self) -> usize {
		self.fast_count
	}

	fn slow_object_count(&self) -> usize {
		self.fifo_queue.len() + (self.main_count - self.fast_count)
	}

	fn needs_capacity_eviction(&self) -> bool {
		self.fifo_used > self.fifo_capacity
	}
}

#[cfg(test)]
mod tests {
	use super::*;

	/// The shared per-ghost-entry DRAM charge in `CacheSize` units, so the
	/// reservation tests below can state their expectations against the
	/// constant itself rather than a literal `44` -- keeping them correct
	/// under `eviction_stacks_pmem`, where it is `0`.
	const GHOST_COST: CacheSize = GHOST_ENTRY_DRAM_OVERHEAD as CacheSize;

	fn drain(stack: &mut TwoQGhostHybridStack) -> Vec<(HashedKey, Tier)> {
		stack.drain_tier_migrations()
	}

	/// `insert` + `update` -- the insert-into-`fifo_queue`-then-promote pairing
	/// every fast-tier test in this module already uses, since this stack never
	/// admits a fresh (non-ghost) key straight into `main_stack`'s fast tier.
	fn promote(stack: &mut TwoQGhostHybridStack, key: HashedKey, size: ObjectSize) {
		stack.insert(key, size);
		stack.update(key);
	}

	/// Smallest fast-tier capacity whose *low* watermark still leaves room for
	/// `bytes`. Lets the fast-tier tests state their expectations in whole
	/// objects instead of hard-coded byte thresholds, so they hold at whatever
	/// `FAST_TIER_HIGH_WATERMARK`/`FAST_TIER_LOW_WATERMARK` pair is configured
	/// rather than only at the default ratios. The `while` loop absorbs the
	/// truncation in `watermarks::low_bytes`' `as u64` cast, which a bare
	/// `ceil()` on its own can land a byte short of.
	fn capacity_holding(bytes: CacheSize) -> CacheSize {
		let mut capacity = (bytes as f64 / watermarks::low()).ceil() as CacheSize;

		while watermarks::low_bytes(capacity) < bytes {
			capacity += 1;
		}

		capacity
	}

	#[test]
	fn admission_always_lands_in_fifo_queue_slow() {
		let mut stack = TwoQGhostHybridStack::new(1.0, 1_000, 1_000);

		stack.insert(1, 10);
		stack.insert(2, 10);

		assert_eq!(stack.tier_of(1), Some(Tier::Slow));
		assert_eq!(stack.tier_of(2), Some(Tier::Slow));
		assert_eq!(drain(&mut stack), Vec::new());
	}

	#[test]
	fn reaccessing_a_fifo_key_promotes_it_to_fast() {
		let mut stack = TwoQGhostHybridStack::new(1.0, 1_000, 1_000);

		stack.insert(1, 10);
		drain(&mut stack);

		stack.update(1);
		let migrations = drain(&mut stack);

		assert_eq!(migrations, vec![(1, Tier::Fast)]);
		assert_eq!(stack.tier_of(1), Some(Tier::Fast));
	}

	#[test]
	fn a_key_aging_out_of_fifo_without_reaccess_becomes_a_ghost_entry() {
		let mut stack = TwoQGhostHybridStack::new(1.0, 1_000, 1_000);

		stack.insert(1, 10);
		drain(&mut stack);

		assert_eq!(stack.evict_one(), Some(1));
		assert_eq!(stack.contains(1), false);
		assert!(stack.is_ghost(1), "evicted fifo key should leave a ghost entry");
	}

	#[test]
	fn ghost_hit_on_readmission_lands_directly_in_fast_tier() {
		let mut stack = TwoQGhostHybridStack::new(1.0, 1_000, 1_000);

		stack.insert(1, 10);
		drain(&mut stack);
		stack.evict_one(); // key 1 ages out -> ghost
		assert!(stack.is_ghost(1));

		// Re-admission: ghost hit -> straight to Main/Fast, no fifo_queue stop.
		stack.insert(1, 10);
		let migrations = drain(&mut stack);

		assert_eq!(migrations, vec![(1, Tier::Fast)]);
		assert_eq!(stack.tier_of(1), Some(Tier::Fast));
	}

	#[test]
	fn fresh_key_with_no_ghost_history_still_lands_in_fifo_queue_slow() {
		let mut stack = TwoQGhostHybridStack::new(1.0, 1_000, 1_000);

		// No prior history for key 5 at all.
		stack.insert(5, 10);
		let migrations = drain(&mut stack);

		assert_eq!(migrations, Vec::new());
		assert_eq!(stack.tier_of(5), Some(Tier::Slow));
	}

	#[test]
	fn fifo_capacity_pressure_is_reported_not_self_evicted() {
		let mut stack = TwoQGhostHybridStack::new(1.0, 15, 1_000);

		stack.insert(1, 10);
		drain(&mut stack);
		assert_eq!(stack.needs_capacity_eviction(), false);

		stack.insert(2, 10);
		drain(&mut stack);
		assert_eq!(stack.needs_capacity_eviction(), true);

		assert_eq!(stack.evict_one(), Some(1));
		assert_eq!(stack.needs_capacity_eviction(), false);
	}

	#[test]
	fn fast_tier_pressure_within_main_queue_demotes_lru_tail() {
		// Sized so a triggered pass drains to a low watermark that still holds
		// two of these three 10-byte objects -- i.e. exactly one demotion, the
		// LRU-most fast key. (Was a hard-coded 25: correct back when a pass
		// drained to the ceiling, but under the watermarks a 25-byte ceiling
		// drains to 18 and takes key 2 down with key 1.)
		let mut stack = TwoQGhostHybridStack::new(1.0, 1_000, capacity_holding(20));

		stack.insert(1, 10);
		stack.insert(2, 10);
		stack.update(1);
		stack.update(2);
		drain(&mut stack);

		assert_eq!(stack.fast_bytes_used(), 20);

		stack.insert(3, 10);
		stack.update(3);
		let migrations = drain(&mut stack);

		assert!(migrations.iter().any(|(k, t)| *k == 1 && *t == Tier::Slow));
		assert_eq!(stack.tier_of(1), Some(Tier::Slow));
		assert_eq!(stack.tier_of(2), Some(Tier::Fast));
		assert_eq!(stack.tier_of(3), Some(Tier::Fast));
	}

	#[test]
	fn evict_one_prefers_fifo_queue_over_main_queue() {
		let mut stack = TwoQGhostHybridStack::new(1.0, 1_000, 1_000);

		stack.insert(1, 10);
		stack.insert(2, 10);
		stack.update(2);
		drain(&mut stack);

		assert_eq!(stack.evict_one(), Some(1));
		assert_eq!(stack.tier_of(2), Some(Tier::Fast));
	}

	#[test]
	fn remove_clears_ghost_entry_too() {
		let mut stack = TwoQGhostHybridStack::new(1.0, 1_000, 1_000);

		stack.insert(1, 10);
		drain(&mut stack);
		stack.evict_one();
		assert!(stack.is_ghost(1));

		stack.remove(1);
		assert!(!stack.is_ghost(1), "remove() should clear the ghost entry too");
	}

	#[test]
	fn remove_and_clear_reset_bookkeeping() {
		let mut stack = TwoQGhostHybridStack::new(1.0, 1_000, 1_000);

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

	/// (a) The trigger is a strict `>`, so usage sitting right *on* the high
	/// watermark -- the largest usage that is not over it -- must leave the
	/// tier completely alone.
	#[test]
	fn fast_usage_at_the_high_watermark_triggers_no_demotion() {
		let fast_capacity: CacheSize = 1_000;
		let high = watermarks::high_bytes(fast_capacity);

		let mut stack = TwoQGhostHybridStack::new(1.0, 100_000, fast_capacity);

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
	}

	/// (b) One byte past the high watermark -- the smallest possible overshoot
	/// -- must fire a pass, and it must take `main_stack`'s LRU-most fast key
	/// rather than the key that just arrived.
	#[test]
	fn fast_usage_above_the_high_watermark_triggers_a_demotion_pass() {
		let fast_capacity: CacheSize = 1_000;
		let high = watermarks::high_bytes(fast_capacity);

		let mut stack = TwoQGhostHybridStack::new(1.0, 100_000, fast_capacity);

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
		assert!(stack.fast_bytes_used() <= watermarks::low_bytes(fast_capacity));
	}

	/// (c) A triggered pass keeps going down to the *low* watermark, not just
	/// back under the ceiling. With the defaults this drains 960 -> 750 across
	/// 21 demotions; the pre-watermark drain-to-ceiling loop would have stopped
	/// after a single one, at 950.
	#[test]
	fn a_triggered_pass_drains_all_the_way_to_the_low_watermark() {
		let fast_capacity: CacheSize = 1_000;
		let size: ObjectSize = 10;
		let bytes = size as CacheSize;

		let high = watermarks::high_bytes(fast_capacity);
		let low = watermarks::low_bytes(fast_capacity);

		let mut stack = TwoQGhostHybridStack::new(1.0, 100_000, fast_capacity);

		// Exactly one object past the high watermark, so precisely one pass
		// fires -- with plenty of resident objects for it to chew through
		// before it reaches the low watermark.
		let count = high / bytes + 1;

		for key in 1..=count {
			promote(&mut stack, key, size);
		}

		let migrations = drain(&mut stack);
		let demoted = migrations.iter().filter(|(_, tier)| *tier == Tier::Slow).count() as CacheSize;

		// The pass halts at the first whole-object multiple at or below the low
		// watermark -- well under `fast_capacity`, which is where the old loop
		// would have left it.
		let expected_used = low - low % bytes;

		assert_eq!(stack.fast_bytes_used(), expected_used);
		assert!(stack.fast_bytes_used() <= low);
		assert_eq!(demoted, (count * bytes - expected_used) / bytes);
	}

	/// (d) Every byte counter and object count still agrees with the per-key
	/// tier tags once a full watermark drain has run -- the same per-demotion
	/// bookkeeping must have run once per demoted object, no more and no less.
	#[test]
	fn byte_and_object_counters_stay_consistent_across_a_watermark_drain() {
		let fast_capacity: CacheSize = 1_000;
		let size: ObjectSize = 10;
		let bytes = size as CacheSize;

		let count = watermarks::high_bytes(fast_capacity) / bytes + 1;

		let mut stack = TwoQGhostHybridStack::new(1.0, 100_000, fast_capacity);

		for key in 1..=count {
			promote(&mut stack, key, size);
		}

		drain(&mut stack);

		let fast_objects = stack.fast_object_count() as CacheSize;
		let slow_objects = stack.slow_object_count() as CacheSize;

		// Nothing was inserted, evicted or resized mid-pass, so every object is
		// still tracked, still `size` bytes, and still on exactly one side of
		// the fast/slow line.
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

	// ---------------------------------------------------------------------
	// Shared-metadata DRAM reservation.
	//
	// Every stack above is constructed *without* `with_shared_overhead`, so it
	// keeps the `0` default and its per-tracked-key term vanishes -- which is
	// why none of those tests needed rescaling. The ghost term does *not*
	// vanish with it (the two are independent -- see `reserved_overhead`), but
	// no test above puts a ghost backlog under fast-tier pressure: the ones
	// that evict into `ghost` hold a single node and assert on tiering and
	// membership, not on the budget. The tests below opt the per-key term in
	// explicitly, bar the two that exist precisely to pin the ghost term's
	// independence from it.
	//
	// Like the watermark tests, they derive their expectations from
	// `watermarks::high_bytes`/`low_bytes` of the *effective* (post-
	// reservation) budget rather than hard-coding the default ratios, so
	// they hold at whatever ratios are configured.
	// ---------------------------------------------------------------------

	/// The reservation covers every *tracked* key, whichever queue and
	/// whichever tier it is in: a `fifo_queue` key holds no fast-tier bytes at
	/// all, but its `entries` row and its list node are DRAM either way.
	#[test]
	fn every_tracked_key_is_charged_including_slow_fifo_queue_keys() {
		const OVERHEAD: CacheSize = 64;
		const CAPACITY: CacheSize = 10_000;

		let mut stack = TwoQGhostHybridStack::new(1.0, 1_000_000, CAPACITY)
			.with_shared_overhead(OVERHEAD);

		assert_eq!(stack.reserved_overhead(), 0);
		assert_eq!(stack.effective_fast_capacity(), CAPACITY);

		// Key 1 promoted into the fast tier; keys 2..=5 left sitting in the
		// slow `fifo_queue`.
		promote(&mut stack, 1, 10);

		for key in 2..=5 {
			stack.insert(key, 10);
		}

		drain(&mut stack);

		assert_eq!(stack.fast_bytes_used(), 10);
		assert_eq!(stack.slow_bytes_used(), 40);
		assert_eq!(stack.len(), 5);

		assert_eq!(stack.reserved_overhead(), 5 * OVERHEAD);
		assert_eq!(stack.effective_fast_capacity(), CAPACITY - 5 * OVERHEAD);

		// Untracking a key hands its reservation back.
		stack.remove(5);

		assert_eq!(stack.reserved_overhead(), 4 * OVERHEAD);
		assert_eq!(stack.effective_fast_capacity(), CAPACITY - 4 * OVERHEAD);
	}

	/// A ghost entry is DRAM held for a key that is *not* tracked -- no
	/// `entries` row, no `fifo_queue`/`main_stack` node -- so it is charged as
	/// its own term instead of being folded into the per-tracked-key constant.
	/// Stated against `GHOST_COST` (i.e. the shared
	/// `crate::object::overhead::GHOST_ENTRY_DRAM_OVERHEAD`) so it holds under
	/// `eviction_stacks_pmem` too, where that constant is `0`.
	#[test]
	fn ghost_entries_are_charged_as_their_own_term() {
		const OVERHEAD: CacheSize = 64;

		let mut stack = TwoQGhostHybridStack::new(1.0, 1_000_000, 10_000)
			.with_shared_overhead(OVERHEAD);

		stack.insert(1, 10);
		assert_eq!(stack.reserved_overhead(), OVERHEAD);

		// Aged out of `fifo_queue`: the tracked row is gone, the ghost node is
		// not -- and neither is the DRAM it occupies.
		assert_eq!(stack.evict_one(), Some(1));
		assert_eq!(stack.len(), 0);
		assert!(stack.is_ghost(1));
		assert_eq!(stack.reserved_overhead(), GHOST_COST);

		// A ghost hit genuinely occupies both structures at once under the lazy
		// `trim_ghost` convention, and is charged for both.
		stack.insert(1, 10);
		drain(&mut stack);

		assert_eq!(stack.tier_of(1), Some(Tier::Fast));
		assert!(stack.is_ghost(1));
		assert_eq!(stack.reserved_overhead(), OVERHEAD + GHOST_COST);

		// `remove` clears the ghost node, and its charge with it.
		stack.remove(1);
		assert_eq!(stack.reserved_overhead(), 0);
	}

	/// The two terms are independent. A stack built *without*
	/// `with_shared_overhead` -- per-tracked-key term flat `0` -- still
	/// reserves for its ghost nodes, because they occupy DRAM either way.
	///
	/// Regression test for the `if self.shared_overhead == 0 { return 0; }`
	/// early return `reserved_overhead` used to open with, which coupled the
	/// two and so reserved nothing for a ghost backlog whenever the per-key
	/// term happened to be unconfigured.
	#[test]
	fn ghost_entries_are_charged_with_no_shared_overhead_configured() {
		const CAPACITY: CacheSize = 10_000;

		let mut stack = TwoQGhostHybridStack::new(1.0, 1_000_000, CAPACITY);

		// Tracked keys alone reserve nothing here: the per-key term really is
		// `0`, so anything non-zero below is the ghost term and only that.
		stack.insert(1, 10);
		stack.insert(2, 10);

		assert_eq!(stack.len(), 2);
		assert_eq!(stack.reserved_overhead(), 0);
		assert_eq!(stack.effective_fast_capacity(), CAPACITY);

		// Aged out of `fifo_queue` oldest-first: no `entries` rows left at all,
		// and yet DRAM is held -- one bare-key ghost node apiece, accumulated
		// one eviction at a time.
		let mut expected: CacheSize = 0;

		assert_eq!(stack.evict_one(), Some(1));
		expected += GHOST_COST;

		assert!(stack.is_ghost(1));
		assert_eq!(stack.reserved_overhead(), expected);
		assert_eq!(stack.effective_fast_capacity(), CAPACITY - expected);

		assert_eq!(stack.evict_one(), Some(2));
		expected += GHOST_COST;

		assert!(stack.is_ghost(2));
		assert_eq!(stack.len(), 0);
		assert_eq!(stack.reserved_overhead(), expected);
		assert_eq!(stack.effective_fast_capacity(), CAPACITY - expected);
	}

	/// The same independence, behaviourally: with no per-key term configured at
	/// all, a ghost backlog on its own still shrinks the effective budget far
	/// enough to force a demotion. Identical to
	/// `a_ghost_backlog_alone_can_force_a_demotion` bar the missing
	/// `with_shared_overhead(1)`, which the old early return made the
	/// difference between a demotion and no reservation whatsoever.
	///
	/// Behavioural, so it is scoped to the DRAM-resident configuration: under
	/// `eviction_stacks_pmem` the ghost list costs the fast tier nothing and
	/// there is nothing here to observe.
	#[cfg(not(feature = "eviction_stacks_pmem"))]
	#[test]
	fn a_ghost_backlog_forces_a_demotion_with_no_shared_overhead_configured() {
		const CAPACITY: CacheSize = 500;

		let mut stack = TwoQGhostHybridStack::new(1.0, 1_000_000, CAPACITY);

		promote(&mut stack, 1, 100);

		assert_eq!(drain(&mut stack), vec![(1, Tier::Fast)]);
		assert_eq!(stack.tier_of(1), Some(Tier::Fast));
		assert_eq!(stack.reserved_overhead(), 0);

		// Ten keys admitted into `fifo_queue` and aged straight back out of it.
		for key in 2..=11 {
			stack.insert(key, 10);
			assert_eq!(stack.evict_one(), Some(key));
		}

		assert_eq!(stack.len(), 1);
		assert_eq!(stack.reserved_overhead(), 10 * GHOST_COST);

		// 500 - 440 = 60 bytes of effective budget left for 100 bytes of
		// fast-tier value. `high_bytes(e) <= e` for any configured ratio, so
		// this is over the trigger whatever the watermarks are set to.
		let effective = stack.effective_fast_capacity();

		assert!(
			stack.fast_bytes_used() > watermarks::high_bytes(effective),
			"an unconfigured per-key term must not zero out the ghost reservation",
		);

		// Growing `ghost` is not itself a fast-tier event, so nothing has
		// settled yet. The next fast-tier event demotes.
		stack.update(1);

		assert_eq!(drain(&mut stack), vec![(1, Tier::Slow)]);
		assert_eq!(stack.tier_of(1), Some(Tier::Slow));
		assert_eq!(stack.fast_bytes_used(), 0);
		assert_eq!(stack.slow_bytes_used(), 100);
	}

	/// A ghost backlog on its own -- with only a single tracked key, charged a
	/// single byte -- is enough DRAM to push the fast tier past its reserved
	/// budget. `trim_ghost` runs only on a *main-queue* eviction, so a run of
	/// `fifo_queue` evictions grows `ghost` unopposed.
	///
	/// Behavioural, so it is scoped to the DRAM-resident configuration: under
	/// `eviction_stacks_pmem` the ghost list costs the fast tier nothing and
	/// there is nothing here to observe.
	#[cfg(not(feature = "eviction_stacks_pmem"))]
	#[test]
	fn a_ghost_backlog_alone_can_force_a_demotion() {
		const CAPACITY: CacheSize = 500;

		let mut stack = TwoQGhostHybridStack::new(1.0, 1_000_000, CAPACITY)
			.with_shared_overhead(1);

		promote(&mut stack, 1, 100);

		assert_eq!(drain(&mut stack), vec![(1, Tier::Fast)]);
		assert_eq!(stack.tier_of(1), Some(Tier::Fast));

		// Ten keys admitted into `fifo_queue` and aged straight back out of it.
		for key in 2..=11 {
			stack.insert(key, 10);
			assert_eq!(stack.evict_one(), Some(key));
		}

		assert_eq!(stack.len(), 1);
		assert_eq!(stack.reserved_overhead(), 1 + 10 * GHOST_COST);

		// 500 - (1 + 440) = 59 bytes of effective budget left for 100 bytes of
		// fast-tier value. `high_bytes(e) <= e` for any configured ratio, so
		// this is over the trigger whatever the watermarks are set to.
		let effective = stack.effective_fast_capacity();

		assert!(
			stack.fast_bytes_used() > watermarks::high_bytes(effective),
			"a 10-entry ghost backlog should reserve the fast tier out from under key 1",
		);

		// Growing `ghost` is not itself a fast-tier event, so nothing has
		// settled yet. The next fast-tier event demotes.
		stack.update(1);

		assert_eq!(drain(&mut stack), vec![(1, Tier::Slow)]);
		assert_eq!(stack.tier_of(1), Some(Tier::Slow));
		assert_eq!(stack.fast_bytes_used(), 0);
		assert_eq!(stack.slow_bytes_used(), 100);
	}

	/// Same capacity, same two objects: the stack with a reservation demotes
	/// where the one without does not. Modelled on `LruHybridStack`'s
	/// `shared_overhead_reserves_dram_and_demotes_earlier`.
	#[test]
	fn shared_overhead_reserves_dram_and_demotes_earlier() {
		const SIZE: ObjectSize = 100;

		let bytes = SIZE as CacheSize;

		// Smallest capacity holding both objects under the *low* watermark, so
		// with nothing reserved no pass can fire at 200 bytes of usage.
		let capacity = capacity_holding(2 * bytes);

		let mut plain = TwoQGhostHybridStack::new(1.0, 1_000_000, capacity);

		promote(&mut plain, 1, SIZE);
		promote(&mut plain, 2, SIZE);

		let plain_migrations = drain(&mut plain);

		assert_eq!(plain.reserved_overhead(), 0);
		assert_eq!(plain.effective_fast_capacity(), capacity);
		assert_eq!(plain.fast_bytes_used(), 2 * bytes);
		assert_eq!(plain.tier_of(1), Some(Tier::Fast));
		assert_eq!(plain.tier_of(2), Some(Tier::Fast));
		assert!(!plain_migrations.iter().any(|(_, tier)| *tier == Tier::Slow));

		// Per-key reservation sized so that two tracked keys leave roughly one
		// object's worth of effective budget: strictly less than the two
		// objects need at any ratio, since `high_bytes(e) <= e`. One tracked
		// key still leaves ~(capacity + bytes)/2, which clears `bytes` for any
		// ratio pair with `low <= high`.
		let overhead = (capacity - bytes) / 2;

		let mut stack = TwoQGhostHybridStack::new(1.0, 1_000_000, capacity)
			.with_shared_overhead(overhead);

		promote(&mut stack, 1, SIZE);

		assert_eq!(stack.tier_of(1), Some(Tier::Fast));
		assert!(
			!drain(&mut stack).iter().any(|(_, tier)| *tier == Tier::Slow),
			"one tracked key's reservation should still leave room for its own value",
		);

		promote(&mut stack, 2, SIZE);
		let migrations = drain(&mut stack);

		let effective = stack.effective_fast_capacity();

		assert_eq!(stack.reserved_overhead(), 2 * overhead);
		assert_eq!(effective, capacity - 2 * overhead);
		assert!(effective < capacity, "the reservation must shrink the effective budget");

		// `main_stack`'s LRU-most fast key goes first, and the pass drains to
		// the low watermark of the *effective* budget.
		assert!(
			migrations.contains(&(1, Tier::Slow)),
			"the LRU-most fast key should demote first, got {migrations:?}",
		);

		assert_eq!(stack.tier_of(1), Some(Tier::Slow));
		assert!(stack.fast_bytes_used() <= watermarks::low_bytes(effective));

		// Same capacity, same two objects, strictly fewer value bytes left in
		// DRAM -- the reservation demoted earlier.
		assert!(stack.fast_bytes_used() < plain.fast_bytes_used());

		// Demotion is the only response: both keys are still tracked.
		assert_eq!(stack.len(), 2);
	}

	/// Composition order: the watermarks apply to what the reservation leaves
	/// behind, so a triggered pass drains to `low_bytes(fast_capacity -
	/// reserved)` -- strictly below `low_bytes(fast_capacity)`, which is where
	/// a watermark-only implementation would have stopped.
	#[test]
	fn the_drain_target_is_the_low_watermark_of_the_reserved_budget() {
		const OVERHEAD: CacheSize = 20;
		const SIZE: ObjectSize = 10;
		const CAPACITY: CacheSize = 10_000;

		let bytes = SIZE as CacheSize;

		let mut stack = TwoQGhostHybridStack::new(1.0, 10_000_000, CAPACITY)
			.with_shared_overhead(OVERHEAD);

		// Promote one object at a time until the growing reservation and the
		// growing usage between them trip the high watermark. Nothing is ever
		// evicted here, so `ghost` stays empty and the reservation is purely
		// `entries.len() * OVERHEAD`. This terminates: by 500 keys the
		// reservation alone is the whole capacity.
		let mut count: CacheSize = 0;

		let demoted = loop {
			count += 1;
			assert!(count < 1_000, "a demotion pass should have fired by now");

			promote(&mut stack, count, SIZE);

			let migrations = drain(&mut stack);
			let demoted = migrations.iter().filter(|(_, tier)| *tier == Tier::Slow).count() as CacheSize;

			if demoted > 0 {
				break demoted;
			}
		};

		assert_eq!(stack.reserved_overhead(), count * OVERHEAD);

		let effective = stack.effective_fast_capacity();
		let low = watermarks::low_bytes(effective);

		// Every object is the same size, so the pass halts at the first
		// whole-object multiple at or below the target.
		let expected_used = low - low % bytes;

		assert_eq!(stack.fast_bytes_used(), expected_used);
		assert_eq!(demoted, (count * bytes - expected_used) / bytes);

		// The load-bearing part. Had the reservation been ignored and the
		// watermarks applied to the raw capacity, the pass would have drained
		// to `low_bytes(CAPACITY)` -- far higher, and it would not have fired
		// this early at all.
		assert!(effective < CAPACITY);
		assert!(low < watermarks::low_bytes(CAPACITY));
		assert!(stack.fast_bytes_used() < watermarks::low_bytes(CAPACITY));
	}

	/// Every byte counter and object count still agrees with the per-key tier
	/// tags after a reservation-triggered pass: the reservation changes when a
	/// pass fires and where it stops, nothing about its per-demotion
	/// bookkeeping.
	#[test]
	fn counters_stay_consistent_across_a_reservation_triggered_pass() {
		const OVERHEAD: CacheSize = 20;
		const SIZE: ObjectSize = 10;
		const CAPACITY: CacheSize = 10_000;

		let bytes = SIZE as CacheSize;

		let mut stack = TwoQGhostHybridStack::new(1.0, 10_000_000, CAPACITY)
			.with_shared_overhead(OVERHEAD);

		let mut count: CacheSize = 0;

		loop {
			count += 1;
			assert!(count < 1_000, "a demotion pass should have fired by now");

			promote(&mut stack, count, SIZE);

			if drain(&mut stack).iter().any(|(_, tier)| *tier == Tier::Slow) {
				break;
			}
		}

		let fast_objects = stack.fast_object_count() as CacheSize;
		let slow_objects = stack.slow_object_count() as CacheSize;

		// Nothing was inserted, evicted or resized mid-pass, so every key is
		// still tracked, still `SIZE` bytes, and still on exactly one side of
		// the fast/slow line.
		assert!(slow_objects > 0, "the pass that fired must have demoted something");
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

		// A demotion moves bytes between tiers; it does not untrack a key, so
		// the reservation is exactly what it was before the pass.
		assert_eq!(stack.reserved_overhead(), count * OVERHEAD);
	}

	/// A reservation exceeding the whole fast budget saturates the effective
	/// budget to `0`: everything demotes, nothing is evicted.
	#[test]
	fn shared_overhead_exceeding_capacity_demotes_all_but_never_evicts() {
		let mut stack = TwoQGhostHybridStack::new(1.0, 1_000, 50).with_shared_overhead(100);

		promote(&mut stack, 1, 10);
		let migrations = drain(&mut stack);

		assert_eq!(stack.reserved_overhead(), 100);
		assert_eq!(stack.effective_fast_capacity(), 0);

		assert_eq!(migrations, vec![(1, Tier::Slow)]);
		assert_eq!(stack.tier_of(1), Some(Tier::Slow));
		assert_eq!(stack.fast_bytes_used(), 0);
		assert_eq!(stack.slow_bytes_used(), 10);

		// Demotion is the only response -- the key is still tracked, and the
		// DRAM budget never evicts (terminal eviction stays governed by
		// `fifo_capacity`/`max_size`).
		assert_eq!(stack.len(), 1);
		assert!(!stack.needs_capacity_eviction());
	}
}
