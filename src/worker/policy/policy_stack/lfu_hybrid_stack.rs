/*
 * Copyright (c) Kia Shakiba
 *
 * This source code is licensed under the GNU AGPLv3 license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! `LfuHybridStack` — a frequency-segmented LFU stack for `PaperPolicy::LfuHybrid`.
//!
//! Two independent frequency-bucket chains (`FrequencyChain`, an adapted
//! copy of `LfuStack`'s classic O(1) LFU structure — see that file) back the
//! fast and slow tiers respectively. A key lives in exactly one chain at a
//! time; `tiers` records which. Unlike `LruHybridStack` (one shared recency
//! list, fast/slow split by list position), LFU's fast/slow boundary is a
//! *frequency* threshold, not a position, so two chains — each queryable for
//! its own minimum frequency in O(1) — are the natural fit.
//!
//! Admission always lands in the fast chain at frequency 1, exactly like
//! `LruHybridStack`'s "always admit fast, let settle demote if needed"
//! design — deliberately *not* special-cased to route straight to the slow
//! tier once the fast tier is full. This still satisfies the paper's
//! admission rule as an emergent result: once the fast tier is full, a
//! freshly admitted key is tied for the fast tier's lowest frequency (1),
//! so `settle_fast_tier` immediately demotes *a* frequency-1 resident in
//! the same `insert` call — not necessarily the newcomer itself. Ties
//! within a frequency bucket break toward whichever key is
//! least-recently-touched (matching plain `LfuStack`'s existing
//! push-front-on-touch/pop-back-on-evict convention elsewhere in this
//! crate), so a newcomer pushed to the front of its bucket displaces
//! whatever frequency-1 key was already there instead of demoting itself.
//! Either way, the key that ends up slow is, by construction, tied for the
//! fast tier's lowest frequency — and it keeps `lib.rs`'s `set()` identical
//! to `lru_hybrid_cache`'s (always synchronously build `TieredBuffer::new_fast`,
//! no synchronous capacity check needed).
//!
//! A slow-tier access (`update`) bumps that key's frequency in the slow
//! chain; if the new count strictly exceeds the fast chain's current
//! minimum (or the fast chain is empty), the key is promoted — moved to the
//! fast chain, preserving its accumulated frequency via `insert_at` — which
//! may itself trigger `settle_fast_tier` to demote the (new) fast minimum.
//!
//! Unlike `LruHybridStack`, `settle_fast_tier` here drains exactly back to
//! `fast_capacity` (no low-water headroom): demotion pressure in this stack
//! is only triggered by a promotion or an explicit `resize_fast_tier`, not
//! by every admission, so the thrashing concern that motivated LRU-hybrid's
//! headroom largely doesn't apply.

use std::collections::HashMap;

use dlv_list::{VecList, Index};
use kwik::collections::HashList;

use crate::{
	CacheSize,
	HashedKey,
	NoHasher,
	policy::PaperPolicy,
	object::ObjectSize,
	worker::policy::policy_stack::{PolicyStack, Tier},
};

/// A classic O(1) LFU frequency-bucket chain: an ascending-by-count linked
/// list of `CountStack` buckets (each holding every key at that exact
/// frequency, itself recency-ordered so ties break LRU-within-frequency),
/// plus an index from key to its current bucket. Adapted from `LfuStack`
/// (`lfu_stack.rs`) with one addition needed here but not there: `insert_at`,
/// which places a key at an *arbitrary* existing count rather than always
/// starting at 1 or advancing by exactly 1 — needed when a key crosses from
/// the other chain carrying its already-accumulated frequency.
#[derive(Default)]
struct FrequencyChain {
	index_map: HashMap<HashedKey, Index<CountStack>, NoHasher>,
	count_stacks: VecList<CountStack>,
}

struct CountStack {
	count: u32,
	stack: HashList<HashedKey, NoHasher>,
}

impl CountStack {
	fn new(count: u32) -> Self {
		CountStack {
			count,
			stack: HashList::with_hasher(NoHasher::default()),
		}
	}

	fn is_empty(&self) -> bool {
		self.stack.is_empty()
	}

	fn push(&mut self, key: HashedKey) {
		self.stack.push_front(key);
	}

	fn pop(&mut self) -> HashedKey {
		self.stack.pop_back().unwrap()
	}

	fn remove(&mut self, key: HashedKey) {
		self.stack.remove(&key).unwrap();
	}
}

impl FrequencyChain {
	fn len(&self) -> usize {
		self.index_map.len()
	}

	/// The lowest frequency currently present in this chain, or `None` if
	/// the chain is empty. O(1) — just the head bucket's count.
	fn min_count(&self) -> Option<u32> {
		self.count_stacks.front().map(|count_stack| count_stack.count)
	}

	/// Inserts a brand-new key at frequency 1. Mirrors `LfuStack::insert`'s
	/// new-key branch. Returns the assigned count (always 1).
	fn insert_new(&mut self, key: HashedKey) -> u32 {
		if self.count_stacks.front().is_none_or(|count_stack| count_stack.count != 1) {
			self.count_stacks.push_front(CountStack::new(1));
		}

		let count_stack_index = self.count_stacks.front_index().unwrap();
		let count_stack = self.count_stacks.get_mut(count_stack_index).unwrap();

		count_stack.push(key);
		self.index_map.insert(key, count_stack_index);

		1
	}

	/// Moves an already-tracked key to the next-higher frequency bucket.
	/// Mirrors `LfuStack::update`. Returns the new count, or `0` if the key
	/// isn't tracked by this chain (callers are expected to only bump keys
	/// they know are present).
	fn bump(&mut self, key: HashedKey) -> u32 {
		let Some(count_stack_index) = self.index_map.get(&key).copied() else {
			return 0;
		};

		let prev_count_stack = self.count_stacks.get_mut(count_stack_index).unwrap();
		let prev_count = prev_count_stack.count;

		prev_count_stack.remove(key);
		let prev_is_empty = prev_count_stack.is_empty();

		if let Some(next_count_stack_index) = self.count_stacks.get_next_index(count_stack_index) {
			let next_count_stack = self.count_stacks.get_mut(next_count_stack_index).unwrap();

			if next_count_stack.count == prev_count + 1 {
				next_count_stack.push(key);
				self.index_map.insert(key, next_count_stack_index);

				if prev_is_empty {
					self.count_stacks.remove(count_stack_index);
				}

				return prev_count + 1;
			}
		}

		let mut new_count_stack = CountStack::new(prev_count + 1);
		new_count_stack.push(key);

		let new_count_stack_index = self.count_stacks.insert_after(count_stack_index, new_count_stack);
		self.index_map.insert(key, new_count_stack_index);

		if prev_is_empty {
			self.count_stacks.remove(count_stack_index);
		}

		prev_count + 1
	}

	/// Places `key` directly into the bucket for an arbitrary existing
	/// `count`, creating that bucket (in sorted position) if it doesn't
	/// already exist. Needed when a promoted/demoted key crosses chains
	/// carrying an accumulated frequency that may not be adjacent to
	/// anything already in this chain — unlike `bump`'s O(1) adjacent-bucket
	/// check, this requires a linear scan to find or create the correctly
	/// sorted bucket. Accepted as O(distinct frequencies in this chain);
	/// expected small in practice since the fast tier is DRAM-budget-limited.
	fn insert_at(&mut self, key: HashedKey, count: u32) {
		let mut cursor = self.count_stacks.front_index();

		while let Some(index) = cursor {
			let count_stack = self.count_stacks.get(index).unwrap();

			if count_stack.count == count {
				let count_stack = self.count_stacks.get_mut(index).unwrap();
				count_stack.push(key);
				self.index_map.insert(key, index);
				return;
			}

			if count_stack.count > count {
				let mut new_count_stack = CountStack::new(count);
				new_count_stack.push(key);

				let new_index = self.count_stacks.insert_before(index, new_count_stack);
				self.index_map.insert(key, new_index);
				return;
			}

			cursor = self.count_stacks.get_next_index(index);
		}

		let mut new_count_stack = CountStack::new(count);
		new_count_stack.push(key);

		let new_index = self.count_stacks.push_back(new_count_stack);
		self.index_map.insert(key, new_index);
	}

	fn remove(&mut self, key: HashedKey) {
		let Some(count_stack_index) = self.index_map.remove(&key) else {
			return;
		};

		let count_stack = self.count_stacks.get_mut(count_stack_index).unwrap();
		count_stack.remove(key);

		if count_stack.is_empty() {
			self.count_stacks.remove(count_stack_index);
		}
	}

	/// Removes and returns the lowest-frequency key (ties broken LRU-within-
	/// frequency) along with the count it held, or `None` if the chain is
	/// empty. Mirrors `LfuStack::evict_one`.
	fn pop_min(&mut self) -> Option<(HashedKey, u32)> {
		let count_stack_index = self.count_stacks.front_index()?;
		let count_stack = self.count_stacks.get_mut(count_stack_index)?;

		let key = count_stack.pop();
		let count = count_stack.count;

		self.index_map.remove(&key);

		if count_stack.is_empty() {
			self.count_stacks.remove(count_stack_index);
		}

		Some((key, count))
	}

	fn clear(&mut self) {
		self.index_map.clear();
		self.count_stacks.clear();
	}
}

pub struct LfuHybridStack {
	fast_chain: FrequencyChain,
	slow_chain: FrequencyChain,

	tiers: HashMap<HashedKey, Tier, NoHasher>,
	sizes: HashMap<HashedKey, ObjectSize, NoHasher>,

	fast_capacity: CacheSize,
	fast_used: CacheSize,
	slow_used: CacheSize,

	/// (key, new tier) pairs recorded since the last `drain_tier_migrations`.
	migrations: Vec<(HashedKey, Tier)>,
}

impl LfuHybridStack {
	pub fn new(fast_capacity: CacheSize) -> Self {
		LfuHybridStack {
			fast_chain: FrequencyChain::default(),
			slow_chain: FrequencyChain::default(),

			tiers: HashMap::default(),
			sizes: HashMap::default(),

			fast_capacity,
			fast_used: 0,
			slow_used: 0,

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

	/// Bumps an already-slow-tier key's frequency and, if the new count
	/// strictly exceeds the fast chain's current minimum (or the fast chain
	/// is empty), promotes it — moving it to the fast chain at its new,
	/// accumulated count. A tie does *not* promote (spec: "exceeds"), which
	/// avoids promote/demote ping-pong between equal-frequency neighbors.
	fn maybe_promote(&mut self, key: HashedKey) {
		let new_count = self.slow_chain.bump(key);
		let fast_min = self.fast_chain.min_count();

		let should_promote = match fast_min {
			None => true,
			Some(min) => new_count > min,
		};

		if !should_promote {
			return;
		}

		let size = self.sizes.get(&key).copied().unwrap_or(0) as CacheSize;

		self.slow_chain.remove(key);
		self.slow_used = self.slow_used.saturating_sub(size);

		self.fast_chain.insert_at(key, new_count);
		self.tiers.insert(key, Tier::Fast);
		self.fast_used += size;

		self.migrations.push((key, Tier::Fast));
	}

	/// Demotes the lowest-frequency fast key(s) until `fast_used` fits back
	/// within `fast_capacity`. Unlike `LruHybridStack`, drains to exactly
	/// `fast_capacity` — see the module doc for why no low-water floor is
	/// needed here.
	fn settle_fast_tier(&mut self) {
		while self.fast_used > self.fast_capacity {
			let Some((demote_key, count)) = self.fast_chain.pop_min() else { break };

			let size = self.sizes.get(&demote_key).copied().unwrap_or(0) as CacheSize;

			self.slow_chain.insert_at(demote_key, count);
			self.tiers.insert(demote_key, Tier::Slow);

			self.fast_used = self.fast_used.saturating_sub(size);
			self.slow_used += size;

			self.migrations.push((demote_key, Tier::Slow));
		}
	}
}

impl PolicyStack for LfuHybridStack {
	fn is_policy(&self, policy: &PaperPolicy) -> bool {
		matches!(policy, PaperPolicy::LfuHybrid)
	}

	fn len(&self) -> usize {
		self.tiers.len()
	}

	fn contains(&self, key: HashedKey) -> bool {
		self.tiers.contains_key(&key)
	}

	fn insert(&mut self, key: HashedKey, size: ObjectSize) {
		if self.tiers.contains_key(&key) {
			// Existing key: track any size change, then treat as an access
			// (matches `LfuStack::insert`'s existing-key delegation to
			// `update`).
			self.resize_key(key, size);

			match self.tiers.get(&key).copied() {
				Some(Tier::Fast) => {
					self.fast_chain.bump(key);
				},

				Some(Tier::Slow) => {
					self.maybe_promote(key);
				},

				None => {},
			}

			self.settle_fast_tier();
			return;
		}

		// Brand-new key: always admitted into the fast chain at frequency 1
		// — see the module doc for why this is not special-cased to route
		// straight to slow once the fast tier is full.
		self.sizes.insert(key, size);
		self.fast_chain.insert_new(key);
		self.tiers.insert(key, Tier::Fast);
		self.fast_used += size as CacheSize;

		self.settle_fast_tier();
	}

	fn update(&mut self, key: HashedKey) {
		match self.tiers.get(&key).copied() {
			Some(Tier::Fast) => {
				self.fast_chain.bump(key);
			},

			Some(Tier::Slow) => {
				self.maybe_promote(key);
				self.settle_fast_tier();
			},

			None => {},
		}
	}

	fn remove(&mut self, key: HashedKey) {
		let size = self.sizes.remove(&key).unwrap_or(0) as CacheSize;
		let tier = self.tiers.remove(&key);

		match tier {
			Some(Tier::Fast) => {
				self.fast_chain.remove(key);
				self.fast_used = self.fast_used.saturating_sub(size);
			},

			Some(Tier::Slow) => {
				self.slow_chain.remove(key);
				self.slow_used = self.slow_used.saturating_sub(size);
			},

			None => {},
		}
	}

	fn clear(&mut self) {
		self.fast_chain.clear();
		self.slow_chain.clear();
		self.tiers.clear();
		self.sizes.clear();

		self.fast_used = 0;
		self.slow_used = 0;
		self.migrations.clear();
	}

	fn evict_one(&mut self) -> Option<HashedKey> {
		if let Some((key, _count)) = self.slow_chain.pop_min() {
			let size = self.sizes.remove(&key).unwrap_or(0) as CacheSize;
			self.tiers.remove(&key);
			self.slow_used = self.slow_used.saturating_sub(size);
			return Some(key);
		}

		// Slow chain empty (e.g. `fast_capacity == max_size`, so nothing has
		// ever been demoted): fall back to the fast chain's minimum, mirrors
		// `LruHybridStack::evict_one`'s fallback for the same situation.
		let (key, _count) = self.fast_chain.pop_min()?;
		let size = self.sizes.remove(&key).unwrap_or(0) as CacheSize;
		self.tiers.remove(&key);
		self.fast_used = self.fast_used.saturating_sub(size);

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
		self.fast_chain.len()
	}

	fn slow_object_count(&self) -> usize {
		self.slow_chain.len()
	}
}

#[cfg(test)]
mod tests {
	use super::*;

	fn drain(stack: &mut LfuHybridStack) -> Vec<(HashedKey, Tier)> {
		stack.drain_tier_migrations()
	}

	#[test]
	fn admission_always_lands_fast() {
		let mut stack = LfuHybridStack::new(1_000);

		stack.insert(1, 10);
		stack.insert(2, 10);

		assert_eq!(stack.tier_of(1), Some(Tier::Fast));
		assert_eq!(stack.tier_of(2), Some(Tier::Fast));
		assert_eq!(stack.fast_bytes_used(), 20);
		assert_eq!(stack.slow_bytes_used(), 0);
	}

	#[test]
	fn admission_once_fast_is_full_is_demoted_immediately() {
		// Fast capacity fits exactly one 10-byte key. A freshly admitted
		// key is always frequency 1 — tied with any existing frequency-1
		// residents once fast is full. Ties within the same frequency
		// bucket break toward whichever key is least-recently-touched
		// within that bucket (matching plain `LfuStack`'s established
		// convention elsewhere in this crate — push_front on touch,
		// pop_back on eviction) — so it's key 1 (already resident, now the
		// LRU of the tied pair), not the newcomer, that demotes. Either
		// way the demoted key is, by construction, tied for the fast
		// tier's lowest frequency, matching the paper's admission rule.
		let mut stack = LfuHybridStack::new(10);

		stack.insert(1, 10);
		drain(&mut stack);
		assert_eq!(stack.tier_of(1), Some(Tier::Fast));

		stack.insert(2, 10);
		let migrations = drain(&mut stack);

		assert_eq!(migrations, vec![(1, Tier::Slow)]);
		assert_eq!(stack.tier_of(1), Some(Tier::Slow));
		assert_eq!(stack.tier_of(2), Some(Tier::Fast));
	}

	#[test]
	fn fast_tier_pressure_demotes_the_lowest_frequency_key() {
		let mut stack = LfuHybridStack::new(25);

		stack.insert(1, 10);
		stack.insert(2, 10);
		drain(&mut stack);

		// Access key 1 twice more so key 2 is the lowest-frequency fast
		// resident once a third key is admitted.
		stack.update(1);
		stack.update(1);
		drain(&mut stack);

		stack.insert(3, 10); // pushes fast_used to 30 > 25
		let migrations = drain(&mut stack);

		assert_eq!(migrations, vec![(2, Tier::Slow)]);
		assert_eq!(stack.tier_of(2), Some(Tier::Slow));
		assert_eq!(stack.tier_of(1), Some(Tier::Fast));
		assert_eq!(stack.tier_of(3), Some(Tier::Fast));
	}

	#[test]
	fn slow_key_promotes_once_frequency_strictly_exceeds_fast_minimum() {
		// Plenty of headroom (1_000) so promotion doesn't also cascade a
		// demotion — that combined behavior is covered separately by
		// `promotion_can_cascade_a_demotion`. This test isolates the
		// threshold check itself.
		let mut stack = LfuHybridStack::new(1_000);

		stack.insert(1, 10); // fast, count 1
		drain(&mut stack);

		// Manually place key 2 into the slow chain at count 1, bypassing
		// admission (which would land it fast, not slow) so the promotion
		// path can be exercised directly and deterministically.
		stack.sizes.insert(2, 10);
		stack.slow_chain.insert_new(2);
		stack.tiers.insert(2, Tier::Slow);
		stack.slow_used += 10;

		assert_eq!(stack.fast_chain.min_count(), Some(1));

		// One access brings key 2 to count 2, strictly exceeding the fast
		// minimum (1) -> promotes.
		stack.update(2);
		let migrations = drain(&mut stack);

		assert_eq!(migrations, vec![(2, Tier::Fast)]);
		assert_eq!(stack.tier_of(2), Some(Tier::Fast));
		assert_ne!(stack.tier_of(2), Some(Tier::Slow));
	}

	#[test]
	fn tie_with_fast_minimum_does_not_promote() {
		let mut stack = LfuHybridStack::new(1_000);

		stack.insert(1, 10); // fast, count 1
		stack.update(1); // fast, count 2 -> fast_min == 2

		// Manually place key 2 into the slow chain at count 1, so a single
		// real `update` bump (1 -> 2) lands it exactly on the fast
		// minimum (2), not strictly past it.
		stack.sizes.insert(2, 10);
		stack.slow_chain.insert_new(2);
		stack.tiers.insert(2, Tier::Slow);
		stack.slow_used += 10;

		assert_eq!(stack.fast_chain.min_count(), Some(2));

		stack.update(2); // bumps slow key 2 to count 2 -> ties fast_min
		let migrations = drain(&mut stack);

		assert_eq!(migrations, Vec::new(), "a tie must not promote");
		assert_eq!(stack.tier_of(2), Some(Tier::Slow));
	}

	#[test]
	fn promotion_can_cascade_a_demotion() {
		let mut stack = LfuHybridStack::new(20);

		stack.insert(1, 10);
		stack.insert(2, 10);
		stack.insert(3, 10); // demotes lowest-frequency key (1 or 2, tie -> LRU)
		drain(&mut stack);

		let (slow_key, other_fast_key) = if stack.tier_of(1) == Some(Tier::Slow) {
			(1, 2)
		} else {
			(2, 1)
		};

		assert_eq!(stack.tier_of(3), Some(Tier::Fast));

		// Bump the slow key past the fast minimum (count 1) so it promotes;
		// fast is already full (20), so the promotion must demote someone.
		stack.update(slow_key);
		let migrations = drain(&mut stack);

		assert!(migrations.iter().any(|(k, t)| *k == slow_key && *t == Tier::Fast));
		assert!(migrations.iter().any(|(_, t)| *t == Tier::Slow));

		assert_eq!(stack.tier_of(slow_key), Some(Tier::Fast));
		// Exactly one of {other_fast_key, 3} should now be slow.
		let now_slow = [other_fast_key, 3].into_iter()
			.filter(|k| stack.tier_of(*k) == Some(Tier::Slow))
			.count();
		assert_eq!(now_slow, 1);
	}

	#[test]
	fn evict_one_prefers_slow_falls_back_to_fast_when_slow_is_empty() {
		let mut stack = LfuHybridStack::new(1_000);

		stack.insert(1, 10);
		stack.insert(2, 10);
		drain(&mut stack);

		// Nothing has ever been demoted; slow chain is empty.
		assert_eq!(stack.evict_one(), Some(1));
		assert_eq!(stack.tier_of(2), Some(Tier::Fast));
	}

	#[test]
	fn evict_one_removes_from_slow_when_present() {
		let mut stack = LfuHybridStack::new(20);

		stack.insert(1, 10);
		stack.insert(2, 10);
		stack.insert(3, 10); // demotes one of {1, 2}
		drain(&mut stack);

		let slow_key = if stack.tier_of(1) == Some(Tier::Slow) { 1 } else { 2 };

		assert_eq!(stack.evict_one(), Some(slow_key));
		assert_eq!(stack.slow_bytes_used(), 0);
	}

	#[test]
	fn resize_fast_tier_shrink_triggers_demotions() {
		let mut stack = LfuHybridStack::new(1_000);

		stack.insert(1, 10);
		stack.insert(2, 10);
		drain(&mut stack);

		stack.resize_fast_tier(10);
		let migrations = drain(&mut stack);

		assert_eq!(migrations.len(), 1);
		assert_eq!(stack.fast_bytes_used(), 10);
	}

	#[test]
	fn resize_fast_tier_grow_creates_headroom_for_next_promotion() {
		let mut stack = LfuHybridStack::new(20);

		stack.insert(1, 10);
		stack.insert(2, 10);
		stack.insert(3, 10); // demotes one of {1, 2}
		drain(&mut stack);

		let slow_key = if stack.tier_of(1) == Some(Tier::Slow) { 1 } else { 2 };

		stack.resize_fast_tier(1_000); // plenty of headroom now
		drain(&mut stack);

		stack.update(slow_key); // bump to count 2, exceeds fast_min (1) -> promotes
		let migrations = drain(&mut stack);

		// With headroom, promotion should not need to demote anyone else.
		assert_eq!(migrations, vec![(slow_key, Tier::Fast)]);
		assert_eq!(stack.tier_of(slow_key), Some(Tier::Fast));
	}

	#[test]
	fn insert_at_preserves_an_arbitrary_accumulated_count() {
		let mut chain = FrequencyChain::default();

		chain.insert_new(1); // count 1
		chain.bump(1); // count 2
		chain.bump(1); // count 3

		let mut other = FrequencyChain::default();
		other.insert_new(2); // count 1

		// Move key 1 (count 3) into `other`, which has no bucket near 3.
		other.insert_at(1, 3);

		assert_eq!(other.min_count(), Some(1)); // key 2 still the minimum
		assert_eq!(other.len(), 2);

		// Popping the minimum should yield key 2 first (count 1), not key 1.
		assert_eq!(other.pop_min(), Some((2, 1)));
		assert_eq!(other.pop_min(), Some((1, 3)));
	}

	#[test]
	fn remove_and_clear_reset_bookkeeping() {
		let mut stack = LfuHybridStack::new(1_000);

		stack.insert(1, 10);
		stack.insert(2, 10);
		drain(&mut stack);

		stack.remove(1);
		assert_eq!(stack.contains(1), false);
		assert_eq!(stack.fast_bytes_used(), 10);

		stack.clear();
		assert_eq!(stack.len(), 0);
		assert_eq!(stack.fast_bytes_used(), 0);
		assert_eq!(stack.slow_bytes_used(), 0);
		assert_eq!(stack.tier_of(2), None);
		assert_eq!(stack.evict_one(), None);
	}
}
