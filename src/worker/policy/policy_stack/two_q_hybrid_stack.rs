/*
 * Copyright (c) Kia Shakiba
 *
 * This source code is licensed under the GNU AGPLv3 license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! `TwoQHybridStack` — a segmented 2Q stack for `PaperPolicy::TwoQHybrid`.
//!
//! Two live queues, matching the paper text directly (unlike this crate's
//! plain `TwoQStack`, which has a heavier three-*live*-queue shape with a
//! real-object `a1_out` overflow queue): `fifo_queue`, a one-access FIFO
//! queue holding real objects that is always entirely in the slow tier, and
//! `main_stack`, a recency-ordered LRU queue segmented fast/slow exactly
//! like `LruHybridStack::stack`.
//!
//! Admission always lands in `fifo_queue` (byte-capped at `fifo_capacity =
//! k_in * max_size`, mirroring how plain `TwoQStack` sizes `a1_in`). A hit
//! on a `fifo_queue`-resident key promotes it straight to the top of
//! `main_stack` at `Tier::Fast`. A `fifo_queue` object that ages out
//! without a second access is evicted outright — no ghost/re-admission
//! memory is kept (see `CLAUDE.md`'s `two_q_hybrid_cache` section for why:
//! an exact-membership ghost check on every admission was flagged as an
//! unwelcome cost given every admission here already pays a synchronous
//! slow-tier/PMEM write; a probabilistic structure is the right tool to
//! revisit this and is left as future work).
//!
//! Note this stack never evicts on its own: `insert`/`resize` only update
//! `fifo_used`, and `needs_capacity_eviction` reports when it has exceeded
//! `fifo_capacity` — the caller (`PolicyWorker::apply_evictions`) is the
//! one that actually removes the object, via the same `evict_one()` +
//! `erase()` pairing it already uses for overall-`max_size` pressure (see
//! `evict_fifo_tail`'s doc comment for why: a `PolicyStack` has no
//! reference to the shared object map, so it cannot safely evict on its
//! own).
//! Once inside `main_stack`, an object behaves exactly like
//! `LruHybridStack`: a fast-tier hit just reorders; a slow-tier hit
//! promotes (and may cascade a demotion); fast-tier pressure demotes the
//! LRU tail down to the slow tier.
//!
//! `fifo_capacity` (sized by the policy-embedded `k_in`, fixed at
//! construction, rescaled on `resize()`) and `fast_capacity` (the main
//! queue's fast/slow split, via `fast_tier_size`/`set_fast_tier_size`,
//! freely adjustable at runtime) are two independent sizing knobs.
//!
//! Eviction priority: `fifo_queue`'s tail first, then `main_stack`'s slow
//! tail, falling back to `main_stack`'s fast tail only if nothing has ever
//! been demoted there yet (same fallback `LruHybridStack::evict_one` has).
//! This reconciles the paper's two eviction clauses into one rule:
//! sacrificing still-unproven FIFO objects before ever touching the proven
//! main queue reproduces both stated behaviors.
//!
//! ## One combined per-key map, not three
//!
//! Every tracked key needs a queue tag (`Fifo`/`Main`), a size, and — only
//! while in `Main` — a tier. An earlier version of this stack tracked these
//! in three separate maps (`queue`, `main_tiers`, `sizes`), mirroring how it
//! was originally built by extending `LruHybridStack`'s (`tiers`+`sizes`)
//! shape with a third map bolted on for the extra Fifo/Main dimension.
//! Checking every call site showed no operation ever wants just one of these
//! in isolation — `insert` touches queue+size together, `remove` touches all
//! three, etc. — so they're now one `entries: HashMap<HashedKey, TwoQEntry>`
//! (`TwoQEntry { dram_resident, queue, tier: Option<Tier>, size }`, `tier: None` iff
//! `queue == Fifo`). This eliminates two of the three hashtable-structural
//! overhead charges per tracked object (see `object/overhead.rs`'s
//! `TwoQHybrid` arm) and removes an entire class of possible desync bug
//! (a key present in one map but not another) by construction, since there
//! is now only one map for a key to be present or absent from. The one
//! `Some -> None` case, `main_tiers.len()` (used by `slow_object_count`), no
//! longer comes for free from the map itself, so a `main_count` counter
//! tracks it explicitly, mirroring the existing `fast_count` pattern.
//!
//! ## Shared-metadata DRAM reservation
//!
//! The object hashtable and this stack's own bookkeeping (`fifo_queue` /
//! `main_stack` list nodes plus the combined `entries` map) live in DRAM but
//! are not part of `fast_used`, so demoting purely against `fast_capacity`
//! would let the fast tier's *real* DRAM footprint exceed its budget.
//! `shared_overhead` (wired in by `init_policy_stack` from
//! `crate::object::overhead::get_hybrid_dram_shared_overhead`, `0` by default
//! so unit tests see the pure value-budget behaviour) is the approximate
//! per-tracked-key cost of that metadata; `reserved_overhead()` multiplies it
//! by `entries.len()` and `settle_fast_tier` subtracts the product from
//! `fast_capacity` before applying the watermarks — see that method's doc.
//!
//! The multiplier is `entries.len()`, i.e. **every** tracked key, not just
//! the fast ones: a `fifo_queue`-resident key's data sits in the slow tier,
//! but its hashtable entry, its `fifo_queue` list node and its `entries` slot
//! are all DRAM just the same. This mirrors `LruHybridStack`'s `stack.len()`
//! and `LruSizedHybridStack`'s `entries.len()`.
//!
//! The whole reservation is charged against `fast_capacity` alone — unlike
//! `LruSizedHybridStack`, this stack has only *one* fast segment to split
//! between. `fifo_capacity` is a **slow**-tier byte cap derived from the
//! overall `max_size` (`fifo_queue` is always entirely slow), so it has no
//! DRAM value budget to carve a share out of.
//!
//! ## `eviction_stacks_pmem`
//!
//! Both live queues (`fifo_queue`, `main_stack`) and the combined per-key
//! `entries` map are DRAM-backed by default. When `eviction_stacks_pmem` is
//! enabled, they are instead allocated in the slow tier (PMEM, via
//! `crate::Hybrid`) — the same switch `LruHybridStack`/`LfuHybridStack` make
//! under this flag. The `PmemHashList`/`hashbrown::HashMap` variants expose
//! the same method surface as the DRAM `HashList`/`std::collections::
//! HashMap` ones used below, so the stack logic itself is identical for both
//! backings; only the transient `migrations` scratch and the scalar
//! counters stay in DRAM.
//!
//! This stack needs no `cfg` of its own for that: under
//! `eviction_stacks_pmem` the eviction-stack term simply drops out of the
//! value `get_hybrid_dram_shared_overhead` hands to `with_shared_overhead`
//! (that function is the single place the PMEM gating lives), so
//! `shared_overhead` shrinks to just the hashtable term — or to `0` if the
//! hashtable moved to PMEM too.

#[cfg(not(feature = "eviction_stacks_pmem"))]
use std::collections::HashMap;
#[cfg(feature = "eviction_stacks_pmem")]
use hashbrown::HashMap;

#[cfg(not(feature = "eviction_stacks_pmem"))]
use kwik::collections::HashList;
#[cfg(feature = "eviction_stacks_pmem")]
use super::pmem_collections::PmemHashList;

// Eviction-stack metadata is allocated through the same crate-wide `Hybrid`
// alias (`numa_alloc::SlowObjects`, node-1-bound jemalloc arenas) that
// `BufferPMEM` and the other PMEM features use, so the stacks land on the
// same node as the slow-tier values they index.
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
	Fifo,
	Main,
}

/// Combined per-key bookkeeping: which queue, which tier (only meaningful
/// while `queue == Main`), and the object's size. See the module doc's "One
/// combined per-key map" section for why this replaced three separate maps.
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

pub struct TwoQHybridStack {
	fifo_queue: QueueList,
	main_stack: QueueList,

	entries: EntryMap,

	k_in: f64,
	fifo_capacity: CacheSize,
	fifo_used: CacheSize,

	fast_capacity: CacheSize,
	fast_used: CacheSize,
	slow_used: CacheSize,

	/// Approximate per-object DRAM cost of the shared structures (object
	/// hashtable + this stack's own lists and `entries` map) that hold an
	/// entry for every tracked object of both queues and both tiers.
	/// Reserved out of `fast_capacity` in `settle_fast_tier` so the
	/// fast-tier budget bounds total DRAM (values + shared metadata), not
	/// just fast-tier values. `0` unless set via `with_shared_overhead` (so
	/// unit tests exercising the pure value-budget behaviour are
	/// unaffected). Mirrors `LruHybridStack::shared_overhead`.
	shared_overhead: CacheSize,

	/// Number of keys currently tagged `Tier::Fast` within `main_stack`.
	/// Kept alongside `fast_used` so `fast_object_count`/`slow_object_count`
	/// don't need an O(n) scan over `entries` — mirrors
	/// `LruHybridStack::fast_count`.
	fast_count: usize,

	/// Number of keys currently in the `Main` queue (Fast or Slow). Kept
	/// explicitly since `entries.len()` now covers *both* queues; before the
	/// three-maps-to-one consolidation this was `main_tiers.len()`, free
	/// from that map's own length.
	main_count: usize,

	/// The least-recently-used key currently tagged `Tier::Fast` within
	/// `main_stack` — i.e. the next demotion candidate. `None` iff no key in
	/// `main_stack` is currently Fast. Mirrors `LruHybridStack::fast_boundary`.
	main_boundary: Option<HashedKey>,

	/// (key, new tier) pairs recorded since the last `drain_tier_migrations`.
	migrations: Vec<(HashedKey, Tier)>,
}

impl TwoQHybridStack {
	/// Constructs the (fifo list, main list, entry map) triple, DRAM- or
	/// PMEM-backed depending on `eviction_stacks_pmem`.
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

		TwoQHybridStack {
			fifo_queue,
			main_stack,

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

	/// Sets the approximate per-object shared-structure DRAM overhead (object
	/// hashtable + eviction stacks) reserved out of the fast-tier budget. See
	/// `crate::object::overhead::get_hybrid_dram_shared_overhead`, which is
	/// also where the `eviction_stacks_pmem`/`global_hashtable_pmem` gating
	/// lives — this stack just receives whatever total that leaves.
	/// Builder-style so `init_policy_stack` can wire it in without disturbing
	/// `new`'s signature (unit tests keep the default `0`).
	pub fn with_shared_overhead(mut self, overhead: CacheSize) -> Self {
		self.shared_overhead = overhead;
		self
	}

	/// The configured fast-tier byte budget, before the shared-metadata
	/// reservation. `settle_fast_tier` is what applies the reservation.
	pub fn fast_capacity(&self) -> CacheSize {
		self.fast_capacity
	}

	/// Total DRAM currently reserved for shared per-object metadata across
	/// both queues and both tiers (`tracked key count × shared_overhead`).
	/// Subtracted from `fast_capacity` to form the effective value-byte
	/// budget in `settle_fast_tier`.
	///
	/// `entries.len()` counts every tracked key — `fifo_queue`-resident ones
	/// included, even though their *data* is always slow-tier. Their
	/// hashtable entry, `fifo_queue` list node and `entries` slot are DRAM
	/// regardless, which is exactly what this reservation is protecting.
	/// Matches `LruHybridStack::reserved_overhead`'s `stack.len()`.
	fn reserved_overhead(&self) -> CacheSize {
		self.entries.len() as CacheSize * self.shared_overhead
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

	/// Records a size change for an already-tracked key without altering its
	/// queue/tier, adjusting whichever counter currently applies.
	fn resize_key(&mut self, key: HashedKey, new_size: ObjectSize) {
		let Some(entry) = self.entries.get_mut(&key) else { return };

		let old_migrating = entry.migrating();
		entry.size = new_size;
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
	/// tagging it `Tier::Fast`. A brand-new entry into `main_stack`, so no
	/// `before`/boundary bookkeeping is needed beyond setting `main_boundary`
	/// if this is the first Fast key.
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

		// Pushed *after* `settle_fast_tier` (which pushes any demotions this
		// promotion itself triggered), not before: `apply_tier_migrations`
		// applies a stack's migrations in push order, so pushing the
		// promotion first would apply its DRAM allocation before the
		// corresponding demotion's DRAM free -- a transient window with both
		// copies resident. Guarded on the key still being `Fast`: an
		// extremely tight budget can demote it straight back out within the
		// same `settle_fast_tier` call (self-eviction), in which case that
		// call already pushed the correct final `(key, Tier::Slow)` entry.
		if self.entries.get(&key).and_then(|entry| entry.tier) == Some(Tier::Fast) {
			self.migrations.push((key, Tier::Fast));
		}
	}

	/// Moves an already-`Main`-tracked key to the front of `main_stack`,
	/// promoting it to `Tier::Fast` if it was `Slow`, then settles the fast
	/// tier. Mirrors `LruHybridStack::touch_fast_key` exactly, scoped to
	/// `main_stack`.
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

		// See `promote_from_fifo`'s doc for why this is pushed after
		// `settle_fast_tier` and guarded on the key still being `Fast`.
		if promoted && self.entries.get(&key).and_then(|entry| entry.tier) == Some(Tier::Fast) {
			self.migrations.push((key, Tier::Fast));
		}
	}

	/// Demotes the least-recently-used fast key(s) within `main_stack` under
	/// the shared fast-tier watermarks (`super::watermarks`): a pass triggers
	/// only once `fast_used` exceeds `high_bytes(effective)`, and once
	/// triggered it drains all the way down to `low_bytes(effective)` rather
	/// than stopping at the ceiling.
	///
	/// This replaces the previous drain-to-exactly-`fast_capacity` rule, which
	/// pinned the fast tier at 100% utilisation and left almost every
	/// triggered pass demoting exactly one object -- migration batches of one,
	/// which maximise per-batch worker overhead and cannot be parallelised.
	/// See `super::watermarks`' doc for the full rationale and for the
	/// `FAST_TIER_HIGH_WATERMARK`/`FAST_TIER_LOW_WATERMARK` overrides (setting
	/// both to `1.0` restores the old drain-to-ceiling behaviour exactly).
	///
	/// `effective` here is `fast_capacity` minus `reserved_overhead()` — the
	/// DRAM the object hashtable and this stack's own bookkeeping already
	/// consume on behalf of every tracked key (see the module doc's
	/// "Shared-metadata DRAM reservation" section), saturating to `0` when
	/// that metadata alone meets or exceeds `fast_capacity`. That subtraction
	/// is what makes the fast-tier budget bound total DRAM rather than just
	/// fast-tier values, and it happens *first*: the watermarks are applied
	/// on top of the already-reduced budget, never in place of it, matching
	/// how `LruHybridStack::settle_fast_tier` composes the two.
	///
	/// `reserved_overhead()` is constant for the duration of one pass —
	/// demotion moves a key between tiers, it never changes `entries.len()` —
	/// so `effective` is safely computed once, before the loop.
	///
	/// Demotion is the only response here; this budget never evicts (terminal
	/// eviction stays governed by `max_size` via `evict_one`, and
	/// `fifo_capacity` pressure by `needs_capacity_eviction`).
	///
	/// Only the loop's entry condition and its stopping point changed. The
	/// per-demotion bookkeeping below (tier tag, `fast_used`, `fast_count`,
	/// `slow_used`, `main_boundary`, migration emission) is untouched and
	/// still runs exactly once per demoted object.
	fn settle_fast_tier(&mut self) {
		// The effective value budget: capacity minus the shared per-object
		// metadata reservation. Both watermarks below are taken against
		// *this*, not against the raw `fast_capacity`.
		let effective = self.fast_capacity.saturating_sub(self.reserved_overhead());

		// Trigger: nothing happens at all until usage crosses the high
		// watermark. Checked once, up front, rather than per iteration --
		// that is what lets a triggered pass keep draining *below* the high
		// watermark, down to the low one.
		if self.fast_used <= watermarks::high_bytes(effective) {
			return;
		}

		let low_water = watermarks::low_bytes(effective);

		while self.fast_used > low_water {
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

	/// Pops and fully removes `fifo_queue`'s tail from this stack's own
	/// bookkeeping (the "reached the top without re-access" key), if any.
	/// Used by `evict_one`'s FIFO-first priority.
	///
	/// Deliberately **not** called from `insert`/`resize` to self-evict
	/// under `k_in`-driven `fifo_capacity` pressure: a `PolicyStack` has no
	/// reference to the shared object map or `status`, so it can only ever
	/// update its own bookkeeping here — it cannot actually remove the
	/// object from the cache or adjust accounted size. Doing so anyway
	/// would silently desync this stack's view of the world from the real
	/// object map (the object would linger forever, untracked). Real
	/// removal always has to go through `PolicyWorker::apply_evictions`'s
	/// `evict_one()` + `erase()` pairing, which is why `fifo_capacity`
	/// pressure is instead surfaced via `needs_capacity_eviction` below —
	/// `apply_evictions` polls that and keeps calling `evict_one()`
	/// (through the correct removal path) until it's satisfied.
	fn evict_fifo_tail(&mut self) -> Option<HashedKey> {
		let key = self.fifo_queue.pop_back()?;
		let size = self.entries.remove(&key).map(|entry| entry.migrating()).unwrap_or(0);

		self.fifo_used = self.fifo_used.saturating_sub(size);

		Some(key)
	}
}

impl PolicyStack for TwoQHybridStack {
	fn is_policy(&self, policy: &PaperPolicy) -> bool {
		matches!(policy, PaperPolicy::TwoQHybrid(k_in) if *k_in == self.k_in)
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
			// Existing key: track any size change, then treat as an access.
			self.resize_key(key, size);
			self.touch(key);
			return;
		}

		// Brand-new key: always admitted into the FIFO queue, always slow.
		// If this pushes fifo_used over fifo_capacity, `needs_capacity_eviction`
		// will report it and `apply_evictions` will drain it via `evict_one`
		// (see that method's doc comment for why eviction can't happen here).
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
		// A shrink may push fifo_used over the new, smaller fifo_capacity;
		// `needs_capacity_eviction` reports it, `apply_evictions` drains it.
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

	fn drain(stack: &mut TwoQHybridStack) -> Vec<(HashedKey, Tier)> {
		stack.drain_tier_migrations()
	}

	/// Fast-tier capacity every watermark-sensitive test sizes against.
	/// Deliberately paired with 1-byte objects below: that makes the drain
	/// byte-exact, so a triggered pass lands on exactly `low_bytes()` and the
	/// expectations below hold at any configured watermark ratio rather than
	/// only at the default ratios. (The watermarks are process-global
	/// `OnceLock`s, so a test cannot pin them via env vars without racing
	/// every other test in the binary -- expectations are computed from
	/// `watermarks::` instead.)
	const CAPACITY: CacheSize = 1_000;

	/// Byte threshold at which a demotion pass triggers, for `CAPACITY`.
	fn high_bytes() -> CacheSize {
		watermarks::high_bytes(CAPACITY)
	}

	/// Byte target a triggered demotion pass drains down to, for `CAPACITY`.
	fn low_bytes() -> CacheSize {
		watermarks::low_bytes(CAPACITY)
	}

	/// Inserts `count` 1-byte keys numbered from `first` and re-accesses each
	/// one, promoting it out of `fifo_queue` into `main_stack`'s fast segment.
	/// Returns the keys in promotion order, so index 0 is the LRU-most of the
	/// batch -- i.e. the first demotion candidate.
	fn fill_fast(stack: &mut TwoQHybridStack, first: HashedKey, count: CacheSize) -> Vec<HashedKey> {
		(0..count)
			.map(|offset| {
				let key = first + offset;

				stack.insert(key, 1);
				stack.update(key);

				key
			})
			.collect()
	}

	/// Filters `keys` (given in promotion order) down to the ones currently
	/// tagged `Tier::Fast`, preserving that order -- which is exactly the
	/// order `settle_fast_tier` walks `main_boundary` in.
	fn fast_keys_lru_first(stack: &TwoQHybridStack, keys: &[HashedKey]) -> Vec<HashedKey> {
		keys.iter()
			.copied()
			.filter(|key| stack.tier_of(*key) == Some(Tier::Fast))
			.collect()
	}

	/// Promotes 1-byte keys (numbered from 1) into `main_stack`'s fast segment
	/// one at a time, stopping the moment the *first* demotion pass fires, and
	/// returns how many keys were promoted.
	///
	/// Two properties the overhead tests below rely on: at the point this
	/// returns, `fifo_queue` is empty and every one of the returned count of
	/// keys is in `main_stack` (so `entries.len()` equals the return value),
	/// and — because it stops at the first pass — every one of them was still
	/// `Tier::Fast` immediately before that pass ran.
	///
	/// Termination: with a non-zero `shared_overhead` the effective budget
	/// shrinks by that much per promotion and must reach 0 well within
	/// `CAPACITY` steps; with a zero one, the pass fires at exactly
	/// `high_bytes(CAPACITY) + 1`, which is `CAPACITY + 1` at the `1.0`
	/// watermark setting -- hence the inclusive `CAPACITY + 1` bound.
	fn promote_until_first_demotion(stack: &mut TwoQHybridStack) -> CacheSize {
		for promoted in 1..=(CAPACITY + 1) {
			fill_fast(stack, promoted, 1);

			if drain(stack).iter().any(|(_, tier)| *tier == Tier::Slow) {
				return promoted;
			}
		}

		panic!("no demotion pass fired within {} promotions", CAPACITY + 1);
	}

	#[test]
	fn admission_always_lands_in_fifo_queue_slow() {
		let mut stack = TwoQHybridStack::new(1.0, 1_000, 1_000);

		stack.insert(1, 10);
		stack.insert(2, 10);

		assert_eq!(stack.tier_of(1), Some(Tier::Slow));
		assert_eq!(stack.tier_of(2), Some(Tier::Slow));
		assert_eq!(stack.slow_bytes_used(), 20);
		assert_eq!(stack.fast_bytes_used(), 0);
		assert_eq!(drain(&mut stack), Vec::new());
	}

	#[test]
	fn reaccessing_a_fifo_key_promotes_it_to_fast() {
		let mut stack = TwoQHybridStack::new(1.0, 1_000, 1_000);

		stack.insert(1, 10);
		drain(&mut stack);

		stack.update(1);
		let migrations = drain(&mut stack);

		assert_eq!(migrations, vec![(1, Tier::Fast)]);
		assert_eq!(stack.tier_of(1), Some(Tier::Fast));
	}

	#[test]
	fn fifo_capacity_pressure_is_reported_not_self_evicted() {
		// k_in=1.0 against max_size=15 -> fifo_capacity fits exactly one
		// 10-byte object.
		let mut stack = TwoQHybridStack::new(1.0, 15, 1_000);

		stack.insert(1, 10);
		drain(&mut stack);
		assert_eq!(stack.contains(1), true);
		assert_eq!(stack.needs_capacity_eviction(), false);

		// New key exceeds fifo_capacity. The stack cannot evict on its own
		// (see `evict_fifo_tail`'s doc comment) -- both keys remain tracked,
		// and `needs_capacity_eviction` reports the pressure so the caller
		// (`apply_evictions`) drains it via the real `evict_one()` path.
		stack.insert(2, 10);
		let migrations = drain(&mut stack);

		assert_eq!(migrations, Vec::new());
		assert_eq!(stack.contains(1), true);
		assert_eq!(stack.contains(2), true);
		assert_eq!(stack.needs_capacity_eviction(), true);

		// Simulates what `apply_evictions` does when it observes
		// `needs_capacity_eviction() == true`: keep calling `evict_one()`
		// (the FIFO tail, key 1) until satisfied.
		assert_eq!(stack.evict_one(), Some(1));
		assert_eq!(stack.contains(1), false);
		assert_eq!(stack.needs_capacity_eviction(), false);
		assert_eq!(stack.tier_of(2), Some(Tier::Slow));
	}

	#[test]
	fn fast_tier_pressure_within_main_queue_demotes_lru_tail() {
		let mut stack = TwoQHybridStack::new(1.0, 1_000, CAPACITY);

		// Sitting exactly *on* the high watermark leaves everything alone --
		// the trigger is a strict `>`.
		let keys = fill_fast(&mut stack, 1, high_bytes());
		let settled = drain(&mut stack);

		assert_eq!(stack.fast_bytes_used(), high_bytes());
		assert!(settled.iter().all(|(_, tier)| *tier == Tier::Fast));

		// One more byte crosses it, and the pass starts at the LRU tail.
		let trigger = fill_fast(&mut stack, high_bytes() + 1, 1)[0];
		let migrations = drain(&mut stack);

		assert_eq!(migrations.first(), Some(&(keys[0], Tier::Slow)));
		assert_eq!(migrations.last(), Some(&(trigger, Tier::Fast)));
		assert_eq!(stack.tier_of(keys[0]), Some(Tier::Slow));
		assert_eq!(stack.tier_of(trigger), Some(Tier::Fast));
	}

	#[test]
	fn promotion_within_main_can_cascade_a_demotion() {
		let mut stack = TwoQHybridStack::new(1.0, 1_000, CAPACITY);

		// One pass' worth of pressure (which leaves usage at the low
		// watermark), then top the tier back up to the high watermark so the
		// single promotion below is what crosses it.
		let keys = fill_fast(&mut stack, 1, high_bytes() + 1);
		let topped = fill_fast(&mut stack, high_bytes() + 2, high_bytes() - low_bytes());
		drain(&mut stack);

		assert_eq!(stack.fast_bytes_used(), high_bytes());
		assert_eq!(stack.tier_of(keys[0]), Some(Tier::Slow));

		let tracked = [keys.as_slice(), topped.as_slice()].concat();
		let fast_lru_first = fast_keys_lru_first(&stack, &tracked);

		// Accessing the slow key promotes it back to fast, which pushes usage
		// past the high watermark and cascades demotions of the fast-tier LRU
		// tail down to the low watermark.
		stack.update(keys[0]);
		let migrations = drain(&mut stack);

		assert_eq!(stack.tier_of(keys[0]), Some(Tier::Fast));
		assert_eq!(stack.fast_bytes_used(), low_bytes());

		// Demotions are applied before the promotion that triggered them, so a
		// promotion never has its DRAM write applied before the corresponding
		// demotion's DRAM free (see `touch_main_fast`'s doc).
		let (demotions, promotion) = migrations.split_at(migrations.len() - 1);

		assert_eq!(promotion.to_vec(), vec![(keys[0], Tier::Fast)]);
		assert!(!demotions.is_empty());
		assert!(demotions.iter().all(|(_, tier)| *tier == Tier::Slow));
		assert_eq!(
			demotions.iter().map(|(key, _)| *key).collect::<Vec<_>>(),
			fast_lru_first[..demotions.len()].to_vec(),
		);
		assert_eq!(stack.tier_of(fast_lru_first[0]), Some(Tier::Slow));
	}

	#[test]
	fn evict_one_prefers_fifo_queue_over_main_queue() {
		let mut stack = TwoQHybridStack::new(1.0, 1_000, 1_000);

		stack.insert(1, 10); // fifo
		stack.insert(2, 10);
		stack.update(2); // promote 2 -> Main/Fast
		drain(&mut stack);

		assert_eq!(stack.evict_one(), Some(1));
		assert_eq!(stack.tier_of(2), Some(Tier::Fast));
	}

	#[test]
	fn evict_one_falls_back_to_main_slow_then_main_fast() {
		// `CAPACITY` rather than a hardcoded ceiling: 20 bytes has to stay
		// under the *high watermark* for "nothing demoted yet" to hold, which
		// a tight literal capacity no longer guarantees.
		let mut stack = TwoQHybridStack::new(1.0, 1_000, CAPACITY);

		stack.insert(1, 10);
		stack.insert(2, 10);
		stack.update(1);
		stack.update(2);
		drain(&mut stack);

		// fifo_queue empty; both keys are Main/Fast (nothing demoted yet).
		assert_eq!(stack.evict_one(), Some(1));
		assert_eq!(stack.tier_of(2), Some(Tier::Fast));
	}

	#[test]
	fn resize_rescales_fifo_capacity_and_reports_pressure() {
		let mut stack = TwoQHybridStack::new(0.5, 1_000, 1_000); // fifo_capacity = 500

		stack.insert(1, 100);
		stack.insert(2, 100);
		drain(&mut stack);
		assert_eq!(stack.slow_bytes_used(), 200);
		assert_eq!(stack.needs_capacity_eviction(), false);

		// Shrink overall max_size to 100 -> fifo_capacity = 50 -> both keys
		// now exceed it (200 > 50), reported via needs_capacity_eviction
		// rather than self-evicted (see `evict_fifo_tail`'s doc comment).
		stack.resize(100);

		assert_eq!(stack.contains(1), true);
		assert_eq!(stack.contains(2), true);
		assert_eq!(stack.needs_capacity_eviction(), true);

		assert_eq!(stack.evict_one(), Some(1));
		assert_eq!(stack.evict_one(), Some(2));
		assert_eq!(stack.needs_capacity_eviction(), false);
	}

	#[test]
	fn resize_fast_tier_shrink_triggers_demotions() {
		let mut stack = TwoQHybridStack::new(1.0, 1_000, CAPACITY);

		fill_fast(&mut stack, 1, high_bytes());
		drain(&mut stack);

		assert_eq!(stack.fast_bytes_used(), high_bytes());

		// Halving the budget puts usage over the *new* high watermark, so the
		// shrink drains all the way down to the new low watermark rather than
		// stopping at the new ceiling.
		let shrunk = CAPACITY / 2;

		stack.resize_fast_tier(shrunk);
		let migrations = drain(&mut stack);

		assert_eq!(stack.fast_bytes_used(), watermarks::low_bytes(shrunk));
		assert!(stack.fast_bytes_used() <= watermarks::high_bytes(shrunk));
		assert_eq!(migrations.len(), (high_bytes() - watermarks::low_bytes(shrunk)) as usize);
		assert!(migrations.iter().all(|(_, tier)| *tier == Tier::Slow));
	}

	#[test]
	fn remove_and_clear_reset_bookkeeping() {
		let mut stack = TwoQHybridStack::new(1.0, 1_000, 1_000);

		stack.insert(1, 10);
		stack.insert(2, 10);
		stack.update(1); // promote 1 -> Main/Fast
		drain(&mut stack);

		stack.remove(1);
		assert_eq!(stack.contains(1), false);
		assert_eq!(stack.fast_bytes_used(), 0);

		stack.remove(2);
		assert_eq!(stack.contains(2), false);
		assert_eq!(stack.slow_bytes_used(), 0);

		stack.insert(3, 10);
		stack.clear();

		assert_eq!(stack.len(), 0);
		assert_eq!(stack.fast_bytes_used(), 0);
		assert_eq!(stack.slow_bytes_used(), 0);
		assert_eq!(stack.tier_of(3), None);
		assert_eq!(stack.evict_one(), None);
	}

	// ---- fast-tier watermarks (`super::watermarks`) ----

	#[test]
	fn usage_at_the_high_watermark_triggers_no_demotion() {
		let mut stack = TwoQHybridStack::new(1.0, 1_000, CAPACITY);

		// Fills to exactly `high_bytes()` -- the largest usage the trigger
		// (`fast_used > high_bytes`) still leaves alone.
		let keys = fill_fast(&mut stack, 1, high_bytes());
		let migrations = drain(&mut stack);

		assert!(migrations.iter().all(|(_, tier)| *tier == Tier::Fast));
		assert_eq!(stack.fast_bytes_used(), high_bytes());
		assert_eq!(stack.slow_bytes_used(), 0);
		assert_eq!(stack.fast_object_count(), keys.len());
		assert_eq!(stack.slow_object_count(), 0);
	}

	#[test]
	fn usage_above_the_high_watermark_triggers_a_pass() {
		let mut stack = TwoQHybridStack::new(1.0, 1_000, CAPACITY);

		fill_fast(&mut stack, 1, high_bytes());
		drain(&mut stack);

		assert_eq!(stack.slow_object_count(), 0);

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
		// `fast_capacity`, which is all the old `fast_used > fast_capacity`
		// rule ever waited for. Skipped when the high watermark is configured
		// back to 1.0, which deliberately restores trigger-at-ceiling.
		if watermarks::high() < 1.0 {
			assert!(high_bytes() + 1 <= CAPACITY);
		}
	}

	#[test]
	fn a_triggered_pass_drains_to_the_low_watermark_not_the_ceiling() {
		let mut stack = TwoQHybridStack::new(1.0, 1_000, CAPACITY);

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

		// The whole point of the low watermark: the pass keeps going well past
		// the ceiling the old rule stopped at. Skipped when the low watermark
		// is configured back to 1.0 (drain-to-ceiling).
		if watermarks::low() < 1.0 {
			assert!(stack.fast_bytes_used() < CAPACITY);
		}

		// Demotion order is LRU-first, same as before the watermarks.
		assert_eq!(demoted, keys[..demoted.len()].to_vec());
	}

	#[test]
	fn counters_stay_consistent_after_a_watermark_pass() {
		let mut stack = TwoQHybridStack::new(1.0, 1_000, CAPACITY);

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
		assert_eq!(stack.fast_object_count() as CacheSize, low_bytes());
		assert_eq!(stack.slow_object_count() as CacheSize, demoted);
		assert_eq!(stack.fast_object_count() + stack.slow_object_count(), total as usize);

		// Per-key tier tags agree with the aggregate counters.
		let fast = keys.iter().filter(|key| stack.tier_of(**key) == Some(Tier::Fast)).count();
		let slow = keys.iter().filter(|key| stack.tier_of(**key) == Some(Tier::Slow)).count();

		assert_eq!(fast, stack.fast_object_count());
		assert_eq!(slow, stack.slow_object_count());

		// And the pass is idempotent: usage now sits under the high watermark,
		// so re-settling demotes nothing further.
		stack.resize_fast_tier(CAPACITY);

		assert_eq!(drain(&mut stack), Vec::new());
		assert_eq!(stack.fast_bytes_used(), low_bytes());
		assert_eq!(stack.slow_bytes_used(), demoted);
	}

	// ---------------------------------------------------------------------
	// Shared-metadata DRAM reservation (`with_shared_overhead`).
	//
	// Every test above constructs the stack through `new` alone, so it sees
	// `shared_overhead == 0`, `reserved_overhead() == 0` and therefore
	// `effective == fast_capacity` -- byte-for-byte the pre-reservation
	// behaviour. That is why none of them needed its capacity rescaled.
	// ---------------------------------------------------------------------

	#[test]
	fn two_q_entry_stays_eight_bytes() {
		// The `entries`-map half of this stack's proposed
		// `TWO_Q_HYBRID_EVICTION_STACK_DRAM_OVERHEAD` is
		// `hashbrown_entry_cost(size_of::<(HashedKey, TwoQEntry)>())`, i.e.
		// `cost(16) = 20` -- the same figure `LruEntry`/`SizedEntry` measured.
		// That holds only while `TwoQEntry` packs into 8 bytes: `size: u32`
		// (4) + `queue: Queue` (1) + `tier: Option<Tier>` (1, niche-packed
		// into `Tier`'s spare discriminants) = 6, padded to 8 by the u32's
		// alignment. A field that spilled past 8 would push the pair to 24
		// and the map charge to `cost(24) = 29`; this test makes that a
		// visible break rather than a silently stale constant.
		assert_eq!(std::mem::size_of::<TwoQEntry>(), 8);
		assert_eq!(std::mem::size_of::<(HashedKey, TwoQEntry)>(), 16);
	}

	#[test]
	fn shared_overhead_defaults_to_zero_and_changes_nothing() {
		let mut plain = TwoQHybridStack::new(1.0, 1_000, CAPACITY);
		let mut zeroed = TwoQHybridStack::new(1.0, 1_000, CAPACITY).with_shared_overhead(0);

		assert_eq!(plain.reserved_overhead(), 0);
		assert_eq!(zeroed.reserved_overhead(), 0);

		fill_fast(&mut plain, 1, high_bytes() + 1);
		fill_fast(&mut zeroed, 1, high_bytes() + 1);

		// Identical migration streams and identical end state: the builder is
		// strictly opt-in.
		assert_eq!(drain(&mut plain), drain(&mut zeroed));
		assert_eq!(plain.fast_bytes_used(), low_bytes());
		assert_eq!(zeroed.fast_bytes_used(), low_bytes());
		assert_eq!(plain.fast_capacity(), CAPACITY);
		assert_eq!(zeroed.fast_capacity(), CAPACITY);
	}

	#[test]
	fn shared_overhead_demotes_earlier_than_an_unreserved_stack() {
		// Sized so the reservation alone consumes the entire fast budget once
		// `KEYS` objects are tracked: `KEYS * OVERHEAD == CAPACITY`.
		const OVERHEAD: CacheSize = 100;
		const KEYS: CacheSize = CAPACITY / OVERHEAD;

		// Ten 1-byte objects are nothing against a 1_000-byte fast tier, so an
		// unreserved stack demotes none of them...
		let mut plain = TwoQHybridStack::new(1.0, 1_000, CAPACITY);
		let plain_keys = fill_fast(&mut plain, 1, KEYS);
		let plain_migrations = drain(&mut plain);

		assert!(plain_migrations.iter().all(|(_, tier)| *tier == Tier::Fast));
		assert_eq!(plain.fast_bytes_used(), KEYS);
		assert_eq!(plain.fast_object_count(), KEYS as usize);
		assert_eq!(plain.slow_object_count(), 0);

		// ...but the 10 x 100 bytes of shared DRAM metadata those same ten
		// keys occupy *is* the whole budget, so the reserved stack has already
		// pushed every one of them back down to the slow tier. Demoting
		// earlier is the entire point: the DRAM the hashtable and this stack's
		// own bookkeeping consume is no longer invisible to the fast tier.
		let mut reserved = TwoQHybridStack::new(1.0, 1_000, CAPACITY)
			.with_shared_overhead(OVERHEAD);

		let reserved_keys = fill_fast(&mut reserved, 1, KEYS);
		drain(&mut reserved);

		assert_eq!(reserved.reserved_overhead(), CAPACITY);
		assert_eq!(reserved.fast_bytes_used(), 0);
		assert_eq!(reserved.fast_object_count(), 0);
		assert_eq!(reserved.slow_object_count(), KEYS as usize);

		// Demotion, not eviction -- both stacks still track every key, and the
		// reservation never touches `fifo_capacity` pressure.
		assert_eq!(reserved_keys, plain_keys);
		assert_eq!(reserved.len(), plain.len());
		assert_eq!(reserved.len(), KEYS as usize);
		assert!(!reserved.needs_capacity_eviction());
	}

	#[test]
	fn shared_overhead_is_charged_for_fifo_resident_keys_too() {
		const OVERHEAD: CacheSize = 10;
		const PARKED: CacheSize = CAPACITY / OVERHEAD;

		let mut stack = TwoQHybridStack::new(1.0, 1_000, CAPACITY)
			.with_shared_overhead(OVERHEAD);

		// 100 one-access keys parked in `fifo_queue`: their *data* is entirely
		// slow-tier (`fast_used` stays 0), but their hashtable entries, their
		// `fifo_queue` list nodes and their `entries` slots are DRAM all the
		// same -- which is why `reserved_overhead` multiplies by
		// `entries.len()` and not by `fast_count`.
		for key in 1..=PARKED {
			stack.insert(key, 1);
		}

		assert_eq!(drain(&mut stack), Vec::new());
		assert_eq!(stack.fast_bytes_used(), 0);
		assert_eq!(stack.fast_object_count(), 0);
		assert_eq!(stack.slow_object_count(), PARKED as usize);

		// 100 x 10 == CAPACITY: the effective value budget is exactly 0 even
		// though not one byte of fast-tier data is in use.
		assert_eq!(stack.reserved_overhead(), CAPACITY);

		// So re-accessing one of them promotes it into `main_stack` and the
		// very same `settle_fast_tier` call demotes it straight back out. A
		// reservation counting only *fast-tier* keys would have reserved 0
		// here and left the key comfortably Fast.
		stack.update(1);
		let migrations = drain(&mut stack);

		assert_eq!(migrations, vec![(1, Tier::Slow)]);
		assert_eq!(stack.tier_of(1), Some(Tier::Slow));
		assert_eq!(stack.fast_bytes_used(), 0);
		assert_eq!(stack.fast_object_count(), 0);

		// Still tracked, still not evicted: this budget only ever demotes.
		assert_eq!(stack.len(), PARKED as usize);
		assert_eq!(stack.slow_bytes_used(), PARKED);
		assert!(!stack.needs_capacity_eviction());
	}

	#[test]
	fn overhead_is_reserved_before_the_watermarks_are_applied() {
		const OVERHEAD: CacheSize = 4;

		let mut stack = TwoQHybridStack::new(1.0, 1_000, CAPACITY)
			.with_shared_overhead(OVERHEAD);

		let promoted = promote_until_first_demotion(&mut stack);

		// `promote_until_first_demotion` leaves `fifo_queue` empty, so every
		// tracked key is a promoted one and the reservation is exactly
		// `promoted * OVERHEAD`. It is also constant across the pass:
		// demotion moves a key between tiers, it never changes `entries.len()`.
		assert_eq!(stack.len(), promoted as usize);
		assert_eq!(stack.reserved_overhead(), promoted * OVERHEAD);

		let effective = CAPACITY - promoted * OVERHEAD;

		// Trigger: measured against `high_bytes(capacity - reserved)`. The
		// strict `>` in `settle_fast_tier` plus the loop in the helper make
		// `promoted` the *first* usage that crosses it -- the step before did
		// not.
		assert!(promoted > watermarks::high_bytes(effective));
		assert!(promoted - 1 <= watermarks::high_bytes(CAPACITY - (promoted - 1) * OVERHEAD));

		// Drain target: `low_bytes(capacity - reserved)`, *not*
		// `low_bytes(capacity)`. 1-byte objects make the drain byte-exact, so
		// this is an equality rather than a bound -- the pass stops on
		// precisely the object that brings usage to the reduced target.
		assert_eq!(stack.fast_bytes_used(), watermarks::low_bytes(effective));
		assert!(stack.fast_bytes_used() <= low_bytes());
		assert_eq!(
			(promoted - stack.fast_bytes_used()) as usize,
			stack.slow_object_count(),
		);

		// The same stack with nothing reserved keeps the old, larger target
		// at the same `fast_capacity` -- and only starts demoting later. That
		// difference is the composition under test: the watermarks act on what
		// the reservation leaves behind, not on the raw capacity.
		let mut unreserved = TwoQHybridStack::new(1.0, 1_000, CAPACITY);
		let unreserved_promoted = promote_until_first_demotion(&mut unreserved);

		assert_eq!(unreserved_promoted, high_bytes() + 1);
		assert_eq!(unreserved.fast_bytes_used(), low_bytes());
		assert!(promoted <= unreserved_promoted);
	}

	#[test]
	fn counters_stay_consistent_across_an_overhead_triggered_pass() {
		const OVERHEAD: CacheSize = 4;

		let mut stack = TwoQHybridStack::new(1.0, 1_000, CAPACITY)
			.with_shared_overhead(OVERHEAD);

		let promoted = promote_until_first_demotion(&mut stack);

		let effective = CAPACITY - promoted * OVERHEAD;
		let expected_fast = watermarks::low_bytes(effective);
		let expected_slow = promoted - expected_fast;

		// The reservation moved where the pass stops; it neither lost, leaked
		// nor double-counted a byte or a key.
		assert_eq!(stack.len(), promoted as usize);
		assert_eq!(stack.fast_bytes_used(), expected_fast);
		assert_eq!(stack.slow_bytes_used(), expected_slow);
		assert_eq!(stack.fast_bytes_used() + stack.slow_bytes_used(), promoted);

		// Objects are 1 byte apiece, so each byte counter doubles as a count.
		assert_eq!(stack.fast_object_count() as CacheSize, expected_fast);
		assert_eq!(stack.slow_object_count() as CacheSize, expected_slow);
		assert_eq!(stack.fast_object_count() + stack.slow_object_count(), promoted as usize);

		// Per-key tier tags agree with the aggregate counters, and the demoted
		// ones are the LRU-most prefix, exactly as without a reservation.
		let keys = (1..=promoted).collect::<Vec<HashedKey>>();
		let fast = keys.iter().filter(|key| stack.tier_of(**key) == Some(Tier::Fast)).count();
		let slow = keys.iter().filter(|key| stack.tier_of(**key) == Some(Tier::Slow)).count();

		assert_eq!(fast, stack.fast_object_count());
		assert_eq!(slow, stack.slow_object_count());
		assert!(
			keys.iter()
				.take(expected_slow as usize)
				.all(|key| stack.tier_of(*key) == Some(Tier::Slow)),
		);

		// And the pass is idempotent at the *reserved* budget: re-settling
		// against the same capacity demotes nothing further, because
		// `low_bytes(effective) <= high_bytes(effective)`.
		stack.resize_fast_tier(CAPACITY);

		assert_eq!(drain(&mut stack), Vec::new());
		assert_eq!(stack.fast_bytes_used(), expected_fast);
		assert_eq!(stack.slow_bytes_used(), expected_slow);
		assert_eq!(stack.len(), promoted as usize);
	}
}
