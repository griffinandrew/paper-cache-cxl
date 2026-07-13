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

/// Which live queue a key currently belongs to.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Queue {
	Fifo,
	Main,
}

pub struct TwoQHybridStack {
	fifo_queue: HashList<HashedKey, NoHasher>,
	main_stack: HashList<HashedKey, NoHasher>,

	queue: HashMap<HashedKey, Queue, NoHasher>,
	main_tiers: HashMap<HashedKey, Tier, NoHasher>,
	sizes: HashMap<HashedKey, ObjectSize, NoHasher>,

	k_in: f64,
	fifo_capacity: CacheSize,
	fifo_used: CacheSize,

	fast_capacity: CacheSize,
	fast_used: CacheSize,
	slow_used: CacheSize,

	/// Number of keys currently tagged `Tier::Fast` within `main_stack`.
	/// Kept alongside `fast_used` so `fast_object_count`/`slow_object_count`
	/// don't need an O(n) scan over `main_tiers` — mirrors
	/// `LruHybridStack::fast_count`.
	fast_count: usize,

	/// The least-recently-used key currently tagged `Tier::Fast` within
	/// `main_stack` — i.e. the next demotion candidate. `None` iff no key in
	/// `main_stack` is currently Fast. Mirrors `LruHybridStack::fast_boundary`.
	main_boundary: Option<HashedKey>,

	/// (key, new tier) pairs recorded since the last `drain_tier_migrations`.
	migrations: Vec<(HashedKey, Tier)>,
}

impl TwoQHybridStack {
	pub fn new(k_in: f64, max_size: CacheSize, fast_capacity: CacheSize) -> Self {
		TwoQHybridStack {
			fifo_queue: HashList::default(),
			main_stack: HashList::default(),

			queue: HashMap::default(),
			main_tiers: HashMap::default(),
			sizes: HashMap::default(),

			k_in,
			fifo_capacity: (k_in * max_size as f64) as CacheSize,
			fifo_used: 0,

			fast_capacity,
			fast_used: 0,
			slow_used: 0,
			fast_count: 0,

			main_boundary: None,
			migrations: Vec::new(),
		}
	}

	/// Returns which queue/tier the given (currently tracked) key is in, or
	/// `None` if the key isn't tracked. Exposed for tests/diagnostics.
	pub fn tier_of(&self, key: HashedKey) -> Option<Tier> {
		match self.queue.get(&key)? {
			Queue::Fifo => Some(Tier::Slow),
			Queue::Main => self.main_tiers.get(&key).copied(),
		}
	}

	/// Records a size change for an already-tracked key without altering its
	/// queue/tier, adjusting whichever counter currently applies.
	fn resize_key(&mut self, key: HashedKey, new_size: ObjectSize) {
		let old_size = self.sizes.insert(key, new_size).unwrap_or(0) as i64;
		let delta = new_size as i64 - old_size;

		match self.queue.get(&key) {
			Some(Queue::Fifo) => {
				self.fifo_used = (self.fifo_used as i64 + delta).max(0) as CacheSize;
			},

			Some(Queue::Main) => {
				match self.main_tiers.get(&key) {
					Some(Tier::Fast) => {
						self.fast_used = (self.fast_used as i64 + delta).max(0) as CacheSize;
					},

					Some(Tier::Slow) => {
						self.slow_used = (self.slow_used as i64 + delta).max(0) as CacheSize;
					},

					None => {},
				}
			},

			None => {},
		}
	}

	/// Treats an already-tracked key as accessed: a `Fifo` key promotes
	/// straight to `Main`+`Fast`; a `Main` key is handled by
	/// `touch_main_fast` (reorder if already Fast, promote if Slow).
	fn touch(&mut self, key: HashedKey) {
		match self.queue.get(&key).copied() {
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
		let size = self.sizes.get(&key).copied().unwrap_or(0) as CacheSize;

		self.fifo_queue.remove(&key);
		self.fifo_used = self.fifo_used.saturating_sub(size);

		self.main_stack.push_front(key);
		self.queue.insert(key, Queue::Main);
		self.main_tiers.insert(key, Tier::Fast);
		self.fast_used += size;
		self.fast_count += 1;

		if self.main_boundary.is_none() {
			self.main_boundary = Some(key);
		}

		self.migrations.push((key, Tier::Fast));
		self.settle_fast_tier();
	}

	/// Moves an already-`Main`-tracked key to the front of `main_stack`,
	/// promoting it to `Tier::Fast` if it was `Slow`, then settles the fast
	/// tier. Mirrors `LruHybridStack::touch_fast_key` exactly, scoped to
	/// `main_stack`.
	fn touch_main_fast(&mut self, key: HashedKey) {
		let previous_tier = self.main_tiers.get(&key).copied();

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

		if previous_tier != Some(Tier::Fast) {
			if previous_tier == Some(Tier::Slow) {
				let size = self.sizes.get(&key).copied().unwrap_or(0) as CacheSize;

				self.slow_used = self.slow_used.saturating_sub(size);
				self.fast_used += size;
				self.fast_count += 1;

				self.migrations.push((key, Tier::Fast));
			}

			self.main_tiers.insert(key, Tier::Fast);

			if self.main_boundary.is_none() {
				self.main_boundary = Some(key);
			}
		}

		self.settle_fast_tier();
	}

	/// Demotes the least-recently-used fast key(s) within `main_stack` until
	/// `fast_used` fits back within `fast_capacity`. Unlike
	/// `LruHybridStack`, drains to exactly `fast_capacity` (no low-water
	/// floor): fast-tier pressure here is only ever triggered by a
	/// promotion or an explicit `resize_fast_tier`, never by every `set()`.
	fn settle_fast_tier(&mut self) {
		while self.fast_used > self.fast_capacity {
			let Some(demote_key) = self.main_boundary else { break };

			let size = self.sizes.get(&demote_key).copied().unwrap_or(0) as CacheSize;
			let new_boundary = self.main_stack.before(&demote_key).copied();

			self.main_tiers.insert(demote_key, Tier::Slow);
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
		let size = self.sizes.remove(&key).unwrap_or(0) as CacheSize;

		self.queue.remove(&key);
		self.fifo_used = self.fifo_used.saturating_sub(size);

		Some(key)
	}
}

impl PolicyStack for TwoQHybridStack {
	fn is_policy(&self, policy: &PaperPolicy) -> bool {
		matches!(policy, PaperPolicy::TwoQHybrid(k_in) if *k_in == self.k_in)
	}

	fn len(&self) -> usize {
		self.queue.len()
	}

	fn contains(&self, key: HashedKey) -> bool {
		self.queue.contains_key(&key)
	}

	fn insert(&mut self, key: HashedKey, size: ObjectSize) {
		if self.queue.contains_key(&key) {
			// Existing key: track any size change, then treat as an access.
			self.resize_key(key, size);
			self.touch(key);
			return;
		}

		// Brand-new key: always admitted into the FIFO queue, always slow.
		// If this pushes fifo_used over fifo_capacity, `needs_capacity_eviction`
		// will report it and `apply_evictions` will drain it via `evict_one`
		// (see that method's doc comment for why eviction can't happen here).
		self.sizes.insert(key, size);
		self.fifo_queue.push_front(key);
		self.queue.insert(key, Queue::Fifo);
		self.fifo_used += size as CacheSize;
	}

	fn update(&mut self, key: HashedKey) {
		if self.queue.contains_key(&key) {
			self.touch(key);
		}
	}

	fn remove(&mut self, key: HashedKey) {
		let Some(q) = self.queue.remove(&key) else { return };
		let size = self.sizes.remove(&key).unwrap_or(0) as CacheSize;

		match q {
			Queue::Fifo => {
				self.fifo_queue.remove(&key);
				self.fifo_used = self.fifo_used.saturating_sub(size);
			},

			Queue::Main => {
				let tier = self.main_tiers.remove(&key);

				let new_boundary_if_needed = if tier == Some(Tier::Fast) && self.main_boundary == Some(key) {
					self.main_stack.before(&key).copied()
				} else {
					None
				};

				self.main_stack.remove(&key);

				match tier {
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
		self.queue.clear();
		self.main_tiers.clear();
		self.sizes.clear();

		self.fifo_used = 0;
		self.fast_used = 0;
		self.slow_used = 0;
		self.fast_count = 0;
		self.main_boundary = None;
		self.migrations.clear();
	}

	fn evict_one(&mut self) -> Option<HashedKey> {
		if let Some(key) = self.evict_fifo_tail() {
			return Some(key);
		}

		let key = self.main_stack.pop_back()?;
		let size = self.sizes.remove(&key).unwrap_or(0) as CacheSize;
		self.queue.remove(&key);

		match self.main_tiers.remove(&key) {
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
		self.fifo_queue.len() + (self.main_tiers.len() - self.fast_count)
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
		let mut stack = TwoQHybridStack::new(1.0, 1_000, 25);

		stack.insert(1, 10);
		stack.insert(2, 10);
		stack.update(1); // promote 1 -> Main/Fast
		stack.update(2); // promote 2 -> Main/Fast
		drain(&mut stack);

		assert_eq!(stack.fast_bytes_used(), 20);

		stack.insert(3, 10);
		stack.update(3); // promote 3 -> pushes fast_used to 30 > 25 -> demotes key 1 (LRU)
		let migrations = drain(&mut stack);

		assert!(migrations.iter().any(|(k, t)| *k == 1 && *t == Tier::Slow));
		assert_eq!(stack.tier_of(1), Some(Tier::Slow));
		assert_eq!(stack.tier_of(2), Some(Tier::Fast));
		assert_eq!(stack.tier_of(3), Some(Tier::Fast));
	}

	#[test]
	fn promotion_within_main_can_cascade_a_demotion() {
		let mut stack = TwoQHybridStack::new(1.0, 1_000, 25);

		stack.insert(1, 10);
		stack.insert(2, 10);
		stack.insert(3, 10);
		stack.update(1);
		stack.update(2);
		stack.update(3); // demotes 1 (fast_used 30 > 25)
		drain(&mut stack);

		assert_eq!(stack.tier_of(1), Some(Tier::Slow));

		// Accessing the slow key promotes it back to fast, which may itself
		// demote the new fast-tier LRU tail (key 2).
		stack.update(1);
		let migrations = drain(&mut stack);

		assert_eq!(stack.tier_of(1), Some(Tier::Fast));
		assert_eq!(migrations, vec![(1, Tier::Fast), (2, Tier::Slow)]);
		assert_eq!(stack.tier_of(2), Some(Tier::Slow));
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
		let mut stack = TwoQHybridStack::new(1.0, 1_000, 25);

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
		let mut stack = TwoQHybridStack::new(1.0, 1_000, 1_000);

		stack.insert(1, 10);
		stack.insert(2, 10);
		stack.update(1);
		stack.update(2);
		drain(&mut stack);

		stack.resize_fast_tier(10);
		let migrations = drain(&mut stack);

		assert_eq!(migrations.len(), 1);
		assert_eq!(stack.fast_bytes_used(), 10);
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
}
