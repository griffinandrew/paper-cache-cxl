/*
 * Copyright (c) Kia Shakiba
 *
 * This source code is licensed under the GNU AGPLv3 license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::{
	cmp,
	borrow::Borrow,
	hash::{Hash, Hasher},
};

use kwik::collections::HashList;

use crate::{
	CacheSize,
	HashedKey,
	NoHasher,
	policy::PaperPolicy,
	object::ObjectSize,
	worker::policy::policy_stack::{AccessOutcome, PolicyStack},
};

pub struct SThreeFifoStack {
	ratio: f64,

	small: Stack,
	main: Stack,
	ghost: HashList<HashedKey, NoHasher>,
}

struct Stack {
	stack: HashList<Object, NoHasher>,

	used_size: CacheSize,
	max_size: Option<CacheSize>,
}

struct Object {
	key: HashedKey,
	size: ObjectSize,
	freq: u8,
}

impl PolicyStack for SThreeFifoStack {
	fn is_policy(&self, policy: &PaperPolicy) -> bool {
		let PaperPolicy::SThreeFifo(ratio) = policy else {
			return false;
		};

		self.ratio == *ratio
	}

	fn len(&self) -> usize {
		self.small.stack.len() + self.main.stack.len()
	}

	fn contains(&self, key: HashedKey) -> bool {
		self.any_stack_contains(key)
	}

	fn insert(&mut self, key: HashedKey, size: ObjectSize) {
		if self.any_stack_contains(key) {
			self.small.update(key, size);
			self.main.update(key, size);

			return self.update(key);
		}

		let object = Object::new(key, size);

		if self.ghost.contains(&key) {
			self.main.insert(object);
		} else {
			self.small.insert(object);
		}
	}

	fn record_access(&mut self, key: HashedKey, hit: bool) -> AccessOutcome {
		if hit {
			self.update(key);
			return AccessOutcome::None;
		}

		if self.any_stack_contains(key) {
			return AccessOutcome::None;
		}

		if self.ghost.contains(&key) {
			return AccessOutcome::GhostHit;
		}

		AccessOutcome::None
	}

	fn update(&mut self, key: HashedKey) {
		self.small.stack.update(&key, |object| object.incr_freq());
		self.main.stack.update(&key, |object| object.incr_freq());
	}

	fn remove(&mut self, key: HashedKey) {
		self.small.remove(key);
		self.main.remove(key);
		self.ghost.remove(&key);
	}

	fn resize(&mut self, max_size: CacheSize) {
		self.small.max_size = Some((self.ratio * max_size as f64) as u64);
		self.main.max_size = Some(((1.0 - self.ratio) * max_size as f64) as u64);
	}

	fn clear(&mut self) {
		self.small.clear();
		self.main.clear();
		self.ghost.clear();
	}

	fn evict_one(&mut self) -> Option<HashedKey> {
		if !self.main.is_full() {
			// prioritize evicting from the small stack when possible
			if let Some(key) = self.evict_small() {
				return Some(key);
			}
		}

		self.evict_main()
	}
}

impl SThreeFifoStack {
	pub fn new(ratio: f64, max_size: CacheSize) -> Self {
		let small = Stack::new(Some((ratio * max_size as f64) as u64));
		let main = Stack::new(Some(((1.0 - ratio) * max_size as f64) as u64));
		let ghost = HashList::with_hasher(NoHasher::default());

		SThreeFifoStack {
			ratio,

			small,
			main,
			ghost,
		}
	}

	fn any_stack_contains(&self, key: HashedKey) -> bool {
		self.small.stack.contains(&key) || self.main.stack.contains(&key)
	}

	fn evict_small(&mut self) -> Option<HashedKey> {
		loop {
			let object = self.small.pop()?;

			if object.freq > 1 {
				self.main.insert(object);

				if self.main.is_full() {
					return self.evict_main();
				}
			} else {
				self.ghost.push_front(object.key);
				return Some(object.key);
			}
		}
	}

	fn evict_main(&mut self) -> Option<HashedKey> {
		loop {
			let mut object = self.main.pop()?;

			if object.freq > 0 {
				object.freq -= 1;
				self.main.insert(object);
			} else {
				while self.ghost.len() > self.main.stack.len() {
					self.ghost.pop_back();
				}

				return Some(object.key);
			}
		}
	}
}

impl Stack {
	fn new(max_size: Option<CacheSize>) -> Self {
		Stack {
			stack: HashList::with_hasher(NoHasher::default()),

			used_size: 0,
			max_size,
		}
	}

	fn is_full(&self) -> bool {
		let Some(max_stack_size) = self.max_size else {
			return false;
		};

		self.used_size >= max_stack_size
	}

	fn insert(&mut self, object: Object) {
		self.used_size += object.size as CacheSize;
		self.stack.push_front(object);
	}

	fn update(&mut self, key: HashedKey, size: ObjectSize) {
		let Some(object) = self.stack.get(&key) else {
			return;
		};

		self.used_size -= object.size as CacheSize;
		self.used_size += size as CacheSize;

		self.stack.update(&key, |object| object.size = size);
	}

	fn remove(&mut self, key: HashedKey) -> Option<Object> {
		let object = self.stack.remove(&key)?;
		self.used_size -= object.size as CacheSize;

		Some(object)
	}

	fn pop(&mut self) -> Option<Object> {
		let object = self.stack.pop_back()?;
		self.used_size -= object.size as CacheSize;

		Some(object)
	}

	fn clear(&mut self) {
		self.stack.clear();
		self.used_size = 0;
	}
}

impl Object {
	fn new(key: HashedKey, size: ObjectSize) -> Self {
		Object {
			key,
			size,

			freq: 0,
		}
	}

	fn incr_freq(&mut self) {
		self.freq = cmp::min(self.freq + 1, 3);
	}
}

impl Borrow<HashedKey> for Object {
	fn borrow(&self) -> &HashedKey {
		&self.key
	}
}

impl Hash for Object {
	fn hash<H>(&self, state: &mut H)
	where
		H: Hasher,
	{
		self.key.hash(state)
	}
}

impl PartialEq for Object {
	fn eq(&self, other: &Self) -> bool {
		self.key == other.key
	}
}

impl Eq for Object {}

#[cfg(test)]
mod tests {
	#[test]
	fn eviction_order_is_correct() {
		use crate::worker::policy::policy_stack::{PolicyStack, SThreeFifoStack};

		let mut stack = SThreeFifoStack::new(0.5, 4);

		for access in [0, 1, 0, 2, 1, 3, 0, 4, 2, 5, 0] {
			stack.insert(access, 1);
		}

		for eviction in [1, 2, 3, 4, 5, 0] {
			assert_eq!(stack.evict_one(), Some(eviction));
		}

		assert_eq!(stack.evict_one(), None);
	}

	/// A key that passes through the ghost queue on its second visit is routed
	/// to S3-FIFO's *main* queue and therefore survives longer than fresh keys
	/// that are still in the *small* queue.
	///
	/// Scenario
	/// --------
	/// 1. Insert keys 0, 1, 2 into a capacity-4 stack (small.max=2, main.max=2).
	/// 2. Manually trigger one eviction: key 0 (tail of small, freq=0) is
	///    evicted and placed into the ghost queue.
	/// 3. Re-insert key 0.  Because it is in the ghost queue, it is routed to
	///    the *main* queue instead of the small queue.
	/// 4. Drain the stack.  Keys 1 and 2 (still in *small*) must be evicted
	///    before key 0 (in *main*), proving the ghost-queue admission control
	///    is working.
	#[test]
	fn ghost_queue_routes_reinserted_key_to_main() {
		use crate::worker::policy::policy_stack::{PolicyStack, SThreeFifoStack};

		// capacity=4, ratio=0.5 → small.max=2, main.max=2
		let mut stack = SThreeFifoStack::new(0.5, 4);

		// Insert three keys (all freq=0) – the stack temporarily exceeds its
		// per-queue capacity; eviction is driven by evict_one() below.
		stack.insert(0, 1); // small: [0]
		stack.insert(1, 1); // small: [1, 0]
		stack.insert(2, 1); // small: [2, 1, 0]

		// Evict one item from the *small* queue tail (key 0, freq=0).
		// Key 0 now lives in the ghost queue, not in any active queue.
		let evicted = stack.evict_one();
		assert_eq!(evicted, Some(0), "tail of small queue (key 0, freq=0) evicted first");
		assert!(!stack.contains(0), "key 0 must be absent after eviction");

		// Re-insert key 0: the ghost queue contains it → main queue admission.
		stack.insert(0, 1);
		assert!(stack.contains(0), "re-inserted key 0 must be in the stack");

		// Drain the remaining keys.
		//
		// Keys 1 and 2 are still in the *small* queue.  The PolicyWorker
		// prioritises small-queue evictions when main is not full, so keys 1
		// and 2 must be evicted before key 0 (which is in the *main* queue).
		assert_eq!(
			stack.evict_one(),
			Some(1),
			"small-queue tail (key 1) must be evicted before the main-queue key",
		);
		assert_eq!(
			stack.evict_one(),
			Some(2),
			"small-queue head (key 2) must be evicted before the main-queue key",
		);
		// Key 0 is in the main queue and outlasts both small-queue items.
		assert_eq!(
			stack.evict_one(),
			Some(0),
			"main-queue key 0 (ghost-hit admission) must be evicted last",
		);
		assert_eq!(stack.evict_one(), None, "stack must be empty after all evictions");
	}

	/// A key inserted for the **first time** (no ghost-queue entry) goes to the
	/// *small* queue, not the *main* queue.
	///
	/// This is the ghost-queue **miss** path — the complement of
	/// `ghost_queue_routes_reinserted_key_to_main` (ghost-queue hit path).
	///
	/// In a two-tier design built on this policy, this corresponds to a
	/// re-promotion from the far tier where the ghost entry has already been
	/// evicted (e.g., due to ghost queue overflow from large object counts).
	/// The re-inserted item must still land in the near tier, but via the
	/// small queue rather than the main queue.
	///
	/// Scenario
	/// --------
	/// 1. Build a capacity-4 stack (small.max=2, main.max=2).
	/// 2. Route key 0 into the *main* queue via ghost-queue hit (evict → re-insert).
	/// 3. Insert a **fresh** key 2 — it has never been in the stack and is not in
	///    the ghost queue → must be admitted to the *small* queue.
	/// 4. Drain the stack and assert: small-queue keys (1 and 2) are evicted before
	///    the main-queue key (0), confirming key 2 was in the small queue.
	#[test]
	fn no_ghost_entry_routes_fresh_insertion_to_small_queue() {
		use crate::worker::policy::policy_stack::{PolicyStack, SThreeFifoStack};

		// capacity=4, ratio=0.5 → small.max=2, main.max=2
		let mut stack = SThreeFifoStack::new(0.5, 4);

		// Step 1 — Establish a main-queue occupant via ghost-queue hit.
		stack.insert(0, 1); // small: [0]
		stack.insert(1, 1); // small: [1, 0]  (head=1, tail=0)

		// Evict tail of small (key 0, freq=0) → ghost: [0]
		let evicted = stack.evict_one();
		assert_eq!(evicted, Some(0), "tail of small queue (key 0, freq=0) evicted into ghost");
		assert!(!stack.contains(0), "key 0 must be absent from active queues");

		// Re-insert key 0: ghost hit → main queue.
		stack.insert(0, 1);
		// State: small: [1], main: [0], ghost: [0] (ghost still contains key 0 while
		// it is in main; re-insertion into main does NOT remove the ghost entry
		// immediately — the ghost entry is only removed by evict_main()).

		// Step 2 — Insert a fresh key 2 with NO ghost history.
		//           Ghost queue does not contain key 2 → small queue admission.
		stack.insert(2, 1);
		// State: small: [2, 1] (head=2, tail=1), main: [0]

		assert!(stack.contains(2), "freshly inserted key 2 must be in the stack");
		assert!(stack.contains(0), "main-queue key 0 must still be present");

		// Step 3 — Drain and verify eviction order.
		//
		// main is not full (1 item, max=2) → PolicyWorker drains small first.
		// small tail = key 1 → evicted first.
		assert_eq!(
			stack.evict_one(),
			Some(1),
			"small-queue tail (key 1) must be evicted before the fresh ghost-miss key",
		);
		// small tail = key 2 (the fresh ghost-miss insertion) → evicted second.
		assert_eq!(
			stack.evict_one(),
			Some(2),
			"fresh key 2 (ghost-miss → small queue) must be evicted before main-queue key",
		);
		// Only main-queue key 0 remains.
		assert_eq!(
			stack.evict_one(),
			Some(0),
			"main-queue key 0 (ghost-hit admission) must outlast both small-queue keys",
		);
		assert_eq!(stack.evict_one(), None, "stack must be empty after draining");
	}
}
