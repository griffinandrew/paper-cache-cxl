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
//! This stack only tracks *order and tier membership*; it does not move any
//! bytes itself. `PolicyWorker` drains `drain_tier_migrations` after each
//! `insert`/`update` call and performs the actual `TieredBuffer`
//! reallocation against the shared object map (see `Object::set_data`).

use std::collections::HashMap;

use kwik::collections::HashList;

use crate::{
	CacheSize,
	HashedKey,
	NoHasher,
	policy::PaperPolicy,
	object::ObjectSize,
	worker::policy::policy_stack::{PolicyStack, Tier},
};

// Unlike `LruStack`, this stack always keeps its recency list in DRAM
// (`kwik::collections::HashList`), even when `eviction_stacks_pmem` is also
// enabled: `PmemHashList` doesn't expose `before`/`back`/`move_front`, which
// the fast/slow boundary tracking below needs. `lru_hybrid_cache` doesn't
// depend on `eviction_stacks_pmem`, so this only matters if a caller enables
// both; in that combination, this stack's *metadata* (not object bytes)
// simply stays DRAM-resident.

pub struct LruHybridStack {
	stack: HashList<HashedKey, NoHasher>,
	sizes: HashMap<HashedKey, ObjectSize, NoHasher>,
	tiers: HashMap<HashedKey, Tier, NoHasher>,

	fast_capacity: CacheSize,
	fast_used: CacheSize,
	slow_used: CacheSize,

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
	pub fn new(fast_capacity: CacheSize) -> Self {
		LruHybridStack {
			stack: HashList::default(),
			sizes: HashMap::default(),
			tiers: HashMap::default(),

			fast_capacity,
			fast_used: 0,
			slow_used: 0,
			fast_count: 0,

			fast_boundary: None,
			migrations: Vec::new(),
		}
	}

	/// The configured fast-tier byte budget.
	pub fn fast_capacity(&self) -> CacheSize {
		self.fast_capacity
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

		if previous_tier != Some(Tier::Fast) {
			if previous_tier == Some(Tier::Slow) {
				let size = self.sizes.get(&key).copied().unwrap_or(0) as CacheSize;

				self.slow_used = self.slow_used.saturating_sub(size);
				self.fast_used += size;
				self.fast_count += 1;

				self.migrations.push((key, Tier::Fast));
			}

			self.tiers.insert(key, Tier::Fast);

			if self.fast_boundary.is_none() {
				self.fast_boundary = Some(key);
			}
		}

		self.settle_fast_tier();
	}

	/// Demotes the least-recently-used fast key(s) until `fast_used` fits
	/// within `fast_capacity`.
	fn settle_fast_tier(&mut self) {
		while self.fast_used > self.fast_capacity {
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
		assert_eq!(migrations, vec![(1, Tier::Fast), (2, Tier::Slow)]);
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

		stack.insert(1, 10);
		stack.insert(2, 10);
		drain(&mut stack);

		stack.resize_fast_tier(10);
		let migrations = drain(&mut stack);

		assert_eq!(migrations, vec![(1, Tier::Slow)]);
		assert_eq!(stack.tier_of(2), Some(Tier::Fast));
		assert_eq!(stack.fast_bytes_used(), 10);
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
