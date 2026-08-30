/*
 * Copyright (c) Kia Shakiba
 *
 * This source code is licensed under the GNU AGPLv3 license found in the
 * LICENSE file in the root directory of this source tree.
 */

// DRAM-backed LRU uses kwik's HashList (standard DRAM allocator).
// PMEM-backed LRU uses PmemHashList which routes allocations through
// `SlowObjects` so that the metadata lives in PMEM.
#[cfg(not(feature = "eviction_stacks_pmem"))]
use kwik::collections::HashList;

#[cfg(feature = "eviction_stacks_pmem")]
use super::pmem_collections::PmemHashList;

use crate::{
	HashedKey,
	NoHasher,
	policy::PaperPolicy,
	object::ObjectSize,
	worker::policy::policy_stack::PolicyStack,
};

// ── DRAM-backed LruStack (default) ────────────────────────────────────────────

#[cfg(not(feature = "eviction_stacks_pmem"))]
#[derive(Default)]
pub struct LruStack {
	stack: HashList<HashedKey, NoHasher>,
}

#[cfg(not(feature = "eviction_stacks_pmem"))]
impl PolicyStack for LruStack {
	fn is_policy(&self, policy: &PaperPolicy) -> bool {
		matches!(policy, PaperPolicy::Lru)
	}

	fn len(&self) -> usize {
		self.stack.len()
	}

	fn contains(&self, key: HashedKey) -> bool {
		self.stack.contains(&key)
	}

	fn insert(&mut self, key: HashedKey, _: ObjectSize) {
		if self.stack.contains(&key) {
			return self.update(key);
		}
		self.stack.push_front(key);
	}

	fn update(&mut self, key: HashedKey) {
		self.stack.move_front(&key);
	}

	fn remove(&mut self, key: HashedKey) {
		self.stack.remove(&key);
	}

	fn clear(&mut self) {
		self.stack.clear();
	}

	fn evict_one(&mut self) -> Option<HashedKey> {
		self.stack.pop_back()
	}
}

// ── PMEM-backed LruStack (eviction_stacks_pmem) ───────────────────────────────
//
// `PmemHashList` allocates its internal nodes through the PMEM eviction
// allocator so the LRU linked-list metadata lives in PMEM rather than DRAM.
// `PmemHashList::push_front` already handles the "re-insert at front" case
// (it removes the existing entry first), so both `insert` and `update` map
// directly to `push_front`.

#[cfg(feature = "eviction_stacks_pmem")]
pub struct LruStack {
	stack: PmemHashList<HashedKey, NoHasher>,
}

#[cfg(feature = "eviction_stacks_pmem")]
impl Default for LruStack {
	fn default() -> Self {
		LruStack {
			stack: PmemHashList::with_hasher(NoHasher::default()),
		}
	}
}

#[cfg(feature = "eviction_stacks_pmem")]
impl PolicyStack for LruStack {
	fn is_policy(&self, policy: &PaperPolicy) -> bool {
		matches!(policy, PaperPolicy::Lru)
	}

	fn len(&self) -> usize {
		self.stack.len()
	}

	fn contains(&self, key: HashedKey) -> bool {
		self.stack.contains(&key)
	}

	fn insert(&mut self, key: HashedKey, _: ObjectSize) {
		// push_front handles duplicate keys by removing and re-adding at front,
		// so a single call is correct for both "insert new" and "move to front".
		// This differs from the DRAM variant (which uses HashList::push_front +
		// move_front separately) because PmemHashList::push_front already
		// performs the combined remove-then-prepend operation atomically.
		self.stack.push_front(key);
	}

	fn update(&mut self, key: HashedKey) {
		// Move-to-front, and ONLY for a key already in the stack.
		//
		// The membership test is load-bearing and is the one place this
		// implementation cannot mirror `insert`. The DRAM variant's
		// `HashList::move_front` is a no-op on an absent key;
		// `PmemHashList::push_front` INSERTS one. Without the guard, a
		// `StackEvent::Get` for a key the worker has already evicted (the
		// client hit it, then eviction ran before the event was drained --
		// both happen on the policy worker, in that order) silently
		// resurrects it as a phantom: `len()` counts an entry with no object
		// behind it, and a later `evict_one` hands back a key the cache
		// cannot erase.
		//
		// Caught by `lru_compact_stack`'s fidelity test, which only runs this
		// path under `eviction_stacks_pmem`: `len` diverged 1 vs 0 on the
		// first `update` of an unseen key.
		if self.stack.contains(&key) {
			self.stack.push_front(key);
		}
	}

	fn remove(&mut self, key: HashedKey) {
		self.stack.remove(&key);
	}

	fn clear(&mut self) {
		self.stack.clear();
	}

	fn evict_one(&mut self) -> Option<HashedKey> {
		self.stack.pop_back()
	}
}

#[cfg(test)]
mod tests {
	#[test]
	fn eviction_order_is_correct() {
		use crate::worker::policy::policy_stack::{PolicyStack, LruStack};

		let mut stack = LruStack::default();

		for access in [0, 1, 1, 1, 0, 2, 3, 0, 2, 0] {
			stack.insert(access, 1);
		}

		for eviction in [1, 3, 2, 0] {
			assert_eq!(stack.evict_one(), Some(eviction));
		}

		assert_eq!(stack.evict_one(), None);
	}
}
