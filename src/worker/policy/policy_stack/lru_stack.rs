/*
 * Copyright (c) Kia Shakiba
 *
 * This source code is licensed under the GNU AGPLv3 license found in the
 * LICENSE file in the root directory of this source tree.
 */

// DRAM-backed LRU uses kwik's HashList (standard DRAM allocator).
// PMEM-backed LRU uses PmemHashList which routes allocations through the
// HybridObjects allocator so that the eviction metadata lives in PMEM.
#[cfg(not(feature = "eviction_stacks_pmem"))]
use kwik::collections::HashList;

#[cfg(feature = "eviction_stacks_pmem")]
use super::pmem_collections::PmemHashList;

use crate::{
    HashedKey, NoHasher, object::ObjectSize, policy::PaperPolicy,
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
// `PmemHashList` allocates its internal nodes through the `HybridObjects`
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
        // Move-to-front: push_front removes existing entry then prepends.
        // Same reasoning as insert() above.
        self.stack.push_front(key);
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
        use crate::worker::policy::policy_stack::{LruStack, PolicyStack};

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
