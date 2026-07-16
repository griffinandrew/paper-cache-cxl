/*
 * Copyright (c) Kia Shakiba
 *
 * This source code is licensed under the GNU AGPLv3 license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! `LruHybridStack` — a single, segmented LRU queue for `PaperPolicy::LruHybrid`.
//!
//! One recency-ordered list backs both tiers. The fast tier is the maximal
//! prefix of the list (starting from the head/MRU end) whose cumulative byte
//! size fits within `fast_capacity`; everything else is the slow tier. Every
//! admission or access moves its key to the front and marks it `Tier::Fast`;
//! whenever that pushes `fast_used` over `fast_capacity`, the least recently
//! used fast key (tracked via `fast_boundary`, so no scan is needed) is
//! demoted. `evict_one` always pops the absolute LRU tail, which — once any
//! demotion has occurred — is always in the slow tier.
//!
//! ## Low-water headroom (reintroduced, smaller than before)
//!
//! `settle_fast_tier` still only *triggers* once `fast_used` genuinely
//! exceeds the effective budget (no early demotion), but once triggered,
//! drains down to `FAST_TIER_LOW_WATER_RATIO` of that budget rather than
//! back to exactly the ceiling. An earlier version of this stack had a
//! larger (90%) low-water floor, removed at the user's explicit request
//! ("keeping the 10% high water mark in the lru implementation hurts
//! performance so get rid of it") because idle headroom cost usable space
//! for no correctness benefit at the time. It was reintroduced, deliberately
//! smaller, for a *different* reason: `PaperCache::set()` writes a new
//! object's `TieredBuffer` to DRAM synchronously at the API layer, before
//! this stack (running on the background `PolicyWorker` thread) even sees
//! the corresponding event — so a burst of concurrent `set()` calls can
//! transiently push real DRAM usage above what the stack's own bookkeeping
//! shows, in the window between that physical write and the worker's next
//! pass. Draining slightly below the ceiling on each settle leaves that
//! burst some room to land in before the *next* settle needs to trigger
//! again. This headroom is not itself what bounds that window — see
//! `PolicyWorker::apply_tier_migrations`'s per-event (not per-batch)
//! scheduling for the change that actually shrinks it — it only reduces how
//! close to the edge the tier sits between settles, and is applied only to
//! `LruHybridStack` (not `LfuHybridStack`, which doesn't re-settle on every
//! admission the way this stack does).
//!
//! This stack only tracks *order and tier membership*; it does not move any
//! bytes itself. `PolicyWorker` drains `drain_tier_migrations` after each
//! `insert`/`update` call and performs the actual `TieredBuffer`
//! reallocation against the shared object map (see `Object::set_data`).

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
	worker::policy::policy_stack::{PolicyStack, Tier},
};

/// Fraction of the effective fast-tier budget `settle_fast_tier` drains down
/// to once triggered, rather than back to exactly the ceiling — see the
/// module doc's "Low-water headroom" section for why. Deliberately much
/// smaller than the 90% low-water floor an earlier version of this stack
/// used (and which was removed for hurting performance): this is a burst
/// safety margin, not a thrashing-reduction mechanism, so it only needs to
/// be big enough to give concurrent `set()` calls some landing room, not
/// large enough to meaningfully reduce demotion-pass frequency on its own.
const FAST_TIER_LOW_WATER_RATIO: f64 = 0.98;

// The recency list and per-key maps are DRAM-backed by default. When
// `eviction_stacks_pmem` is enabled, they are instead allocated in the slow
// tier (PMEM, via `crate::Hybrid`) — co-located with the slow-tier object
// bytes — exactly the way plain `LruStack` switches to PMEM collections under
// that flag. The method surface of the PMEM `PmemHashList`/`hashbrown::HashMap`
// variants matches the DRAM `HashList`/`std::collections::HashMap` ones used
// below, so the stack logic itself is identical for both backings. Only the
// transient `migrations` scratch and the scalar counters stay in DRAM.
#[cfg(not(feature = "eviction_stacks_pmem"))]
type RecencyList = HashList<HashedKey, NoHasher>;
#[cfg(feature = "eviction_stacks_pmem")]
type RecencyList = PmemHashList<HashedKey, NoHasher>;

#[cfg(not(feature = "eviction_stacks_pmem"))]
type SizeMap = HashMap<HashedKey, ObjectSize, NoHasher>;
#[cfg(feature = "eviction_stacks_pmem")]
type SizeMap = HashMap<HashedKey, ObjectSize, NoHasher, Hybrid>;

#[cfg(not(feature = "eviction_stacks_pmem"))]
type TierMap = HashMap<HashedKey, Tier, NoHasher>;
#[cfg(feature = "eviction_stacks_pmem")]
type TierMap = HashMap<HashedKey, Tier, NoHasher, Hybrid>;

pub struct LruHybridStack {
	stack: RecencyList,
	sizes: SizeMap,
	tiers: TierMap,

	fast_capacity: CacheSize,
	fast_used: CacheSize,
	slow_used: CacheSize,

	/// Approximate per-object DRAM cost of the shared structures (object
	/// hashtable + eviction stacks) that hold an entry for every object of
	/// both tiers. Reserved out of `fast_capacity` in `settle_fast_tier` so
	/// the fast-tier budget bounds total DRAM (values + shared metadata), not
	/// just fast-tier values. `0` unless set via `with_shared_overhead` (so
	/// unit tests exercising the pure value-budget behavior are unaffected).
	shared_overhead: CacheSize,

	/// Number of keys currently tagged `Tier::Fast`. Kept alongside
	/// `fast_used` (bytes) so `fast_object_count`/`slow_object_count` don't
	/// need an O(n) scan over `tiers`.
	fast_count: usize,

	/// The least-recently-used key currently tagged `Tier::Fast` — i.e. the
	/// next candidate for demotion. `None` iff no key is currently Fast.
	/// Because the fast tier is always a contiguous prefix of `stack`
	/// (starting from the head), this single key is enough to find the
	/// demotion candidate in O(1) instead of scanning the list.
	fast_boundary: Option<HashedKey>,

	/// (key, new tier) pairs recorded since the last `drain_tier_migrations`.
	migrations: Vec<(HashedKey, Tier)>,
}

impl LruHybridStack {
	/// Constructs the (recency list, size map, tier map) triple, DRAM- or
	/// PMEM-backed depending on `eviction_stacks_pmem`.
	#[cfg(not(feature = "eviction_stacks_pmem"))]
	fn new_collections() -> (RecencyList, SizeMap, TierMap) {
		(HashList::default(), HashMap::default(), HashMap::default())
	}

	#[cfg(feature = "eviction_stacks_pmem")]
	fn new_collections() -> (RecencyList, SizeMap, TierMap) {
		(
			PmemHashList::with_hasher(NoHasher::default()),
			HashMap::with_hasher_in(NoHasher::default(), Hybrid),
			HashMap::with_hasher_in(NoHasher::default(), Hybrid),
		)
	}

	pub fn new(fast_capacity: CacheSize) -> Self {
		let (stack, sizes, tiers) = Self::new_collections();

		LruHybridStack {
			stack,
			sizes,
			tiers,

			fast_capacity,
			fast_used: 0,
			slow_used: 0,
			shared_overhead: 0,
			fast_count: 0,

			fast_boundary: None,
			migrations: Vec::new(),
		}
	}

	/// Sets the approximate per-object shared-structure DRAM overhead (object
	/// hashtable + eviction stacks) reserved out of the fast-tier budget. See
	/// `crate::object::overhead::get_hybrid_dram_shared_overhead`. Builder-style
	/// so `init_policy_stack` can wire it in without disturbing `new`'s
	/// signature (unit tests keep the default `0`).
	pub fn with_shared_overhead(mut self, overhead: CacheSize) -> Self {
		self.shared_overhead = overhead;
		self
	}

	/// The configured fast-tier byte budget.
	pub fn fast_capacity(&self) -> CacheSize {
		self.fast_capacity
	}

	/// Total DRAM currently reserved for shared per-object metadata across
	/// both tiers (`tracked object count × shared_overhead`). Subtracted from
	/// `fast_capacity` to form the effective value-byte budget in
	/// `settle_fast_tier`.
	fn reserved_overhead(&self) -> CacheSize {
		self.stack.len() as CacheSize * self.shared_overhead
	}

	/// Returns the tier the given (currently tracked) key is in, or `None`
	/// if the key isn't tracked. Exposed for tests/diagnostics.
	pub fn tier_of(&self, key: HashedKey) -> Option<Tier> {
		self.tiers.get(&key).copied()
	}

	/// Records a size change for an already-tracked key without altering its
	/// tier, adjusting whichever tier's used-bytes counter currently applies.
	fn resize_key(&mut self, key: HashedKey, new_size: ObjectSize) {
		let old_size = self.sizes.insert(key, new_size).unwrap_or(0) as i64;
		let delta = new_size as i64 - old_size;

		match self.tiers.get(&key) {
			Some(Tier::Fast) => {
				self.fast_used = (self.fast_used as i64 + delta).max(0) as CacheSize;
			},

			Some(Tier::Slow) => {
				self.slow_used = (self.slow_used as i64 + delta).max(0) as CacheSize;
			},

			None => {},
		}
	}

	/// Moves an already-tracked key to the front of the recency list,
	/// promoting it to `Tier::Fast` if it was in the slow tier, then settles
	/// the fast tier (demoting as needed). Used by both `insert` (on an
	/// existing key — a `set()` always re-admits to the fast tier) and
	/// `update` (a fast-or-slow hit).
	fn touch_fast_key(&mut self, key: HashedKey) {
		let previous_tier = self.tiers.get(&key).copied();

		let already_at_front = self.stack.front() == Some(&key);
		let is_boundary = self.fast_boundary == Some(key);

		// If this key is the current fast-tier boundary and is about to
		// move away from its spot, the key immediately ahead of it (toward
		// the head) becomes the new boundary once it moves. Must be
		// captured *before* the move, since `before` reads current list
		// structure.
		let new_boundary_if_moved = if is_boundary && !already_at_front {
			self.stack.before(&key).copied()
		} else {
			None
		};

		self.stack.move_front(&key);

		if is_boundary && !already_at_front {
			self.fast_boundary = new_boundary_if_moved;
		}

		let mut promoted = false;

		if previous_tier != Some(Tier::Fast) {
			if previous_tier == Some(Tier::Slow) {
				let size = self.sizes.get(&key).copied().unwrap_or(0) as CacheSize;

				self.slow_used = self.slow_used.saturating_sub(size);
				self.fast_used += size;
				self.fast_count += 1;

				promoted = true;
			}

			self.tiers.insert(key, Tier::Fast);

			if self.fast_boundary.is_none() {
				self.fast_boundary = Some(key);
			}
		}

		self.settle_fast_tier();

		// Pushed *after* `settle_fast_tier` (which pushes any demotions this
		// promotion itself triggered) rather than before: `apply_tier_
		// migrations` applies a stack's migrations in order, physically
		// reallocating each one, so pushing the promotion first meant a
		// promotion that pushed `fast_used` over budget had its DRAM
		// allocation applied *before* the corresponding demotion's DRAM
		// free -- a real, reported transient window where both the
		// promoted object's new DRAM copy and a not-yet-demoted victim's
		// old DRAM copy were resident simultaneously. Guarded on the key
		// still being `Fast` afterward: an extremely tight budget can demote
		// this same key straight back out within the same `settle_fast_tier`
		// call (self-eviction, e.g. a fast tier that fits nothing) -- in
		// that case `settle_fast_tier` already pushed the correct final
		// `(key, Tier::Slow)` entry and no separate `Fast` entry should
		// follow it.
		if promoted && self.tiers.get(&key) == Some(&Tier::Fast) {
			self.migrations.push((key, Tier::Fast));
		}
	}

	/// Demotes the least-recently-used fast key(s), *triggered* once
	/// `fast_used` exceeds the *effective* value budget (`fast_capacity`
	/// minus the DRAM reserved for shared per-object metadata — hashtable +
	/// eviction stacks — across both tiers; this makes the fast-tier budget
	/// bound total DRAM, not just fast-tier values, saturating to 0 when the
	/// shared metadata alone meets/exceeds `fast_capacity`), but *drained*
	/// down to `FAST_TIER_LOW_WATER_RATIO` of that budget rather than back to
	/// the ceiling exactly — see the module doc for why this headroom was
	/// reintroduced. Demotion is the only response; the DRAM budget never
	/// evicts (terminal eviction stays governed solely by `max_size`).
	fn settle_fast_tier(&mut self) {
		let effective = self.fast_capacity.saturating_sub(self.reserved_overhead());

		if self.fast_used <= effective {
			return;
		}

		let drain_target = (effective as f64 * FAST_TIER_LOW_WATER_RATIO) as CacheSize;

		while self.fast_used > drain_target {
			let Some(demote_key) = self.fast_boundary else { break };

			let size = self.sizes.get(&demote_key).copied().unwrap_or(0) as CacheSize;
			let new_boundary = self.stack.before(&demote_key).copied();

			self.tiers.insert(demote_key, Tier::Slow);
			self.fast_used = self.fast_used.saturating_sub(size);
			self.fast_count = self.fast_count.saturating_sub(1);
			self.slow_used += size;
			self.fast_boundary = new_boundary;

			self.migrations.push((demote_key, Tier::Slow));
		}
	}
}

impl PolicyStack for LruHybridStack {
	fn is_policy(&self, policy: &PaperPolicy) -> bool {
		matches!(policy, PaperPolicy::LruHybrid)
	}

	fn len(&self) -> usize {
		self.stack.len()
	}

	fn contains(&self, key: HashedKey) -> bool {
		self.stack.contains(&key)
	}

	fn insert(&mut self, key: HashedKey, size: ObjectSize) {
		if self.stack.contains(&key) {
			// Existing key: track any size change, then treat as an access.
			// Per the admission policy, a `set()` always places the object
			// at the top of the fast tier, promoting it if it was slow.
			self.resize_key(key, size);
			self.touch_fast_key(key);
			return;
		}

		self.sizes.insert(key, size);
		self.stack.push_front(key);
		self.tiers.insert(key, Tier::Fast);
		self.fast_used += size as CacheSize;
		self.fast_count += 1;

		if self.fast_boundary.is_none() {
			self.fast_boundary = Some(key);
		}

		self.settle_fast_tier();
	}

	fn update(&mut self, key: HashedKey) {
		if self.tiers.contains_key(&key) {
			self.touch_fast_key(key);
		}
	}

	fn remove(&mut self, key: HashedKey) {
		let size = self.sizes.remove(&key).unwrap_or(0) as CacheSize;
		let tier = self.tiers.remove(&key);

		let new_boundary_if_needed = if tier == Some(Tier::Fast) && self.fast_boundary == Some(key) {
			self.stack.before(&key).copied()
		} else {
			None
		};

		self.stack.remove(&key);

		match tier {
			Some(Tier::Fast) => {
				self.fast_used = self.fast_used.saturating_sub(size);
				self.fast_count = self.fast_count.saturating_sub(1);

				if self.fast_boundary == Some(key) {
					self.fast_boundary = new_boundary_if_needed;
				}
			},

			Some(Tier::Slow) => {
				self.slow_used = self.slow_used.saturating_sub(size);
			},

			None => {},
		}
	}

	fn clear(&mut self) {
		self.stack.clear();
		self.sizes.clear();
		self.tiers.clear();

		self.fast_used = 0;
		self.slow_used = 0;
		self.fast_count = 0;
		self.fast_boundary = None;
		self.migrations.clear();
	}

	fn evict_one(&mut self) -> Option<HashedKey> {
		let key = self.stack.pop_back()?;
		let size = self.sizes.remove(&key).unwrap_or(0) as CacheSize;

		match self.tiers.remove(&key) {
			Some(Tier::Fast) => {
				self.fast_used = self.fast_used.saturating_sub(size);
				self.fast_count = self.fast_count.saturating_sub(1);

				// The tail of the whole list can only be Fast-tagged if
				// every tracked key is still Fast (no demotion has ever
				// happened), in which case the boundary must have equaled
				// this key too. The new tail, if any, is then still Fast.
				if self.fast_boundary == Some(key) {
					self.fast_boundary = self.stack.back().copied();
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
		self.slow_used
	}

	fn fast_object_count(&self) -> usize {
		self.fast_count
	}

	fn slow_object_count(&self) -> usize {
		self.tiers.len() - self.fast_count
	}
}

#[cfg(test)]
mod tests {
	use super::*;

	fn drain(stack: &mut LruHybridStack) -> Vec<(HashedKey, Tier)> {
		stack.drain_tier_migrations()
	}

	#[test]
	fn evict_one_terminates_when_both_keys_are_immediately_demoted() {
		// Reproduces the `apply_evictions` scenario where fast_capacity is
		// tiny relative to object sizes, so *both* inserted keys demote to
		// slow immediately (fast_boundary bounces to None each time).
		let mut stack = LruHybridStack::new(4);

		stack.insert(1, 19);
		stack.insert(2, 19);
		drain(&mut stack);

		assert_eq!(stack.tier_of(1), Some(Tier::Slow));
		assert_eq!(stack.tier_of(2), Some(Tier::Slow));
		assert_eq!(stack.len(), 2);

		assert_eq!(stack.evict_one(), Some(1));
		assert_eq!(stack.len(), 1);
		assert_eq!(stack.evict_one(), Some(2));
		assert_eq!(stack.len(), 0);
		assert_eq!(stack.evict_one(), None);
	}

	#[test]
	fn admission_always_lands_fast() {
		let mut stack = LruHybridStack::new(1_000);

		stack.insert(1, 10);
		stack.insert(2, 10);

		assert_eq!(stack.tier_of(1), Some(Tier::Fast));
		assert_eq!(stack.tier_of(2), Some(Tier::Fast));
		assert_eq!(stack.fast_bytes_used(), 20);
		assert_eq!(stack.slow_bytes_used(), 0);
	}

	#[test]
	fn fast_tier_pressure_demotes_lru_tail() {
		let mut stack = LruHybridStack::new(25);

		stack.insert(1, 10); // fast: [1]
		stack.insert(2, 10); // fast: [2, 1]
		drain(&mut stack);

		stack.insert(3, 10); // pushes fast_used to 30 > 25 -> demotes key 1 (LRU)
		let migrations = drain(&mut stack);

		assert_eq!(migrations, vec![(1, Tier::Slow)]);
		assert_eq!(stack.tier_of(1), Some(Tier::Slow));
		assert_eq!(stack.tier_of(2), Some(Tier::Fast));
		assert_eq!(stack.tier_of(3), Some(Tier::Fast));
		assert_eq!(stack.fast_bytes_used(), 20);
		assert_eq!(stack.slow_bytes_used(), 10);
	}

	#[test]
	fn accessing_a_slow_key_promotes_it_and_may_demote_another() {
		let mut stack = LruHybridStack::new(25);

		stack.insert(1, 10);
		stack.insert(2, 10);
		stack.insert(3, 10); // demotes 1 -> slow
		drain(&mut stack);

		assert_eq!(stack.tier_of(1), Some(Tier::Slow));

		// Accessing the slow key promotes it back to the fast tier, which
		// may itself demote the new fast-tier LRU tail (key 2).
		stack.update(1);
		let migrations = drain(&mut stack);

		assert_eq!(stack.tier_of(1), Some(Tier::Fast));
		// Demotion is applied before the promotion that triggered it, so a
		// promotion never has its DRAM write applied before the
		// corresponding demotion's DRAM free (see `touch_fast_key`'s doc).
		assert_eq!(migrations, vec![(2, Tier::Slow), (1, Tier::Fast)]);
		assert_eq!(stack.tier_of(2), Some(Tier::Slow));
		assert_eq!(stack.tier_of(3), Some(Tier::Fast));
	}

	#[test]
	fn promoted_key_is_absent_from_slow_afterward_and_demoted_key_from_fast() {
		let mut stack = LruHybridStack::new(25);

		stack.insert(1, 10);
		stack.insert(2, 10);
		stack.insert(3, 10); // demotes 1
		drain(&mut stack);

		stack.update(1); // promotes 1, likely demotes 2
		drain(&mut stack);

		// A key can never be tagged as being in both tiers simultaneously —
		// `tier_of` returns exactly one Tier per key by construction.
		assert_ne!(stack.tier_of(1), Some(Tier::Slow));
		assert_ne!(stack.tier_of(2), Some(Tier::Fast));
	}

	#[test]
	fn object_counts_track_tier_membership() {
		let mut stack = LruHybridStack::new(15);

		stack.insert(1, 10); // fast
		stack.insert(2, 10); // demotes 1 -> slow, fast holds 2
		drain(&mut stack);

		assert_eq!(stack.fast_object_count(), 1);
		assert_eq!(stack.slow_object_count(), 1);

		stack.remove(1);
		assert_eq!(stack.fast_object_count(), 1);
		assert_eq!(stack.slow_object_count(), 0);
	}

	#[test]
	fn evict_one_only_removes_from_slow_tier_once_demotions_have_happened() {
		let mut stack = LruHybridStack::new(15);

		stack.insert(1, 10); // fast
		stack.insert(2, 10); // demotes 1 -> slow, fast holds 2
		drain(&mut stack);

		assert_eq!(stack.tier_of(1), Some(Tier::Slow));
		assert_eq!(stack.tier_of(2), Some(Tier::Fast));

		// Tail of the whole list is the slow key (1), since it was demoted
		// and never re-accessed.
		assert_eq!(stack.evict_one(), Some(1));
		assert_eq!(stack.tier_of(2), Some(Tier::Fast));
		assert_eq!(stack.slow_bytes_used(), 0);
	}

	#[test]
	fn evict_one_falls_back_to_fast_tail_when_everything_still_fits() {
		let mut stack = LruHybridStack::new(1_000);

		stack.insert(1, 10);
		stack.insert(2, 10);
		drain(&mut stack);

		// Nothing has ever been demoted; the whole list is Fast.
		assert_eq!(stack.evict_one(), Some(1));
		assert_eq!(stack.tier_of(2), Some(Tier::Fast));
		assert_eq!(stack.fast_bytes_used(), 10);
	}

	#[test]
	fn zero_fast_capacity_demotes_immediately() {
		let mut stack = LruHybridStack::new(0);

		stack.insert(1, 10);
		let migrations = drain(&mut stack);

		assert_eq!(migrations, vec![(1, Tier::Slow)]);
		assert_eq!(stack.tier_of(1), Some(Tier::Slow));
		assert_eq!(stack.fast_bytes_used(), 0);
		assert_eq!(stack.slow_bytes_used(), 10);
	}

	#[test]
	fn resizing_an_existing_key_adjusts_the_correct_tier_counter() {
		let mut stack = LruHybridStack::new(1_000);

		stack.insert(1, 10);
		drain(&mut stack);
		assert_eq!(stack.fast_bytes_used(), 10);

		// re-`set()` with a larger value: still fast, counter adjusted.
		stack.insert(1, 30);
		assert_eq!(stack.tier_of(1), Some(Tier::Fast));
		assert_eq!(stack.fast_bytes_used(), 30);
	}

	#[test]
	fn shrinking_fast_tier_at_runtime_triggers_demotions() {
		let mut stack = LruHybridStack::new(1_000);

		stack.insert(1, 100);
		stack.insert(2, 100);
		drain(&mut stack);

		// effective=150, drain_target = (150 * 0.98) = 147 (truncated).
		// fast_used starts at 200; demoting the LRU tail (key 1, 100 bytes)
		// alone already lands at 100, comfortably under 147.
		stack.resize_fast_tier(150);
		let migrations = drain(&mut stack);

		assert_eq!(migrations, vec![(1, Tier::Slow)]);
		assert_eq!(stack.tier_of(2), Some(Tier::Fast));
		assert_eq!(stack.fast_bytes_used(), 100);
	}

	#[test]
	fn settle_drains_to_low_water_target_with_headroom() {
		let mut stack = LruHybridStack::new(1_000);

		for key in 1..=100 {
			stack.insert(key, 10); // fills to exactly 1_000 (== capacity, no trigger)
		}
		drain(&mut stack);
		assert_eq!(stack.fast_bytes_used(), 1_000);

		// 1_010 > 1_000 -> triggers; drains down to the low-water target
		// (1_000 * 0.98 = 980), demoting more than the single object that
		// would have been the bare minimum to get back under capacity --
		// this is the reintroduced headroom (see FAST_TIER_LOW_WATER_RATIO
		// and the module doc's "Low-water headroom" section).
		stack.insert(101, 10);
		let migrations = drain(&mut stack);

		assert_eq!(migrations, vec![(1, Tier::Slow), (2, Tier::Slow), (3, Tier::Slow)]);
		assert_eq!(stack.fast_bytes_used(), 980);
	}

	#[test]
	fn remove_updates_boundary_and_counters() {
		let mut stack = LruHybridStack::new(1_000);

		stack.insert(1, 10);
		stack.insert(2, 10);
		drain(&mut stack);

		stack.remove(2);
		assert_eq!(stack.contains(2), false);
		assert_eq!(stack.fast_bytes_used(), 10);

		// The remaining fast key (1) must still be demotable correctly.
		stack.resize_fast_tier(0);
		let migrations = drain(&mut stack);
		assert_eq!(migrations, vec![(1, Tier::Slow)]);
	}

	#[test]
	fn shared_overhead_reserves_dram_and_demotes_earlier() {
		// Without overhead, two 40-byte values fit in a 100-byte fast tier.
		let mut plain = LruHybridStack::new(100);
		plain.insert(1, 40);
		plain.insert(2, 40);
		drain(&mut plain);
		assert_eq!(plain.tier_of(1), Some(Tier::Fast));
		assert_eq!(plain.tier_of(2), Some(Tier::Fast));

		// With a 30-byte per-object shared reservation, the second insert
		// reserves 2 × 30 = 60, leaving an effective value budget of 40 --
		// tighter than the raw values (80) fit. Both objects end up demoted:
		// after the LRU tail (key 1) demotes, fast_used (40) is still above
		// the low-water drain target (40 * 0.98 = 39, truncated), so settle
		// continues and demotes key 2 as well, landing at 0. This is the
		// reintroduced low-water headroom (FAST_TIER_LOW_WATER_RATIO)
		// interacting with the DRAM-cap reservation at small scale -- with
		// only two same-sized objects there's no smaller demotion available
		// to land closer to the target without emptying the tier.
		let mut stack = LruHybridStack::new(100).with_shared_overhead(30);
		stack.insert(1, 40);
		stack.insert(2, 40);
		let migrations = drain(&mut stack);

		assert_eq!(migrations, vec![(1, Tier::Slow), (2, Tier::Slow)]);
		assert_eq!(stack.tier_of(1), Some(Tier::Slow));
		assert_eq!(stack.tier_of(2), Some(Tier::Slow));
		assert_eq!(stack.fast_bytes_used(), 0);
	}

	#[test]
	fn shared_overhead_exceeding_capacity_demotes_all_but_never_evicts() {
		// One object's shared reservation (100) already exceeds the whole
		// fast budget (50): the effective value budget saturates to 0, so the
		// object demotes to slow immediately on admission.
		let mut stack = LruHybridStack::new(50).with_shared_overhead(100);
		stack.insert(1, 10);
		let migrations = drain(&mut stack);

		assert_eq!(migrations, vec![(1, Tier::Slow)]);
		assert_eq!(stack.tier_of(1), Some(Tier::Slow));
		assert_eq!(stack.fast_bytes_used(), 0);

		// Demotion is the only response — the object is still tracked (the
		// DRAM budget never evicts; `needs_capacity_eviction` stays default).
		assert_eq!(stack.len(), 1);
		assert!(!stack.needs_capacity_eviction());
	}

	#[test]
	fn clear_resets_all_state() {
		let mut stack = LruHybridStack::new(15);

		stack.insert(1, 10);
		stack.insert(2, 10);
		drain(&mut stack);

		stack.clear();

		assert_eq!(stack.len(), 0);
		assert_eq!(stack.fast_bytes_used(), 0);
		assert_eq!(stack.slow_bytes_used(), 0);
		assert_eq!(stack.tier_of(1), None);
		assert_eq!(stack.evict_one(), None);
	}
}
