

/*
 * Copyright (c) Kia Shakiba
 *
 * This source code is licensed under the GNU AGPLv3 license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! Custom allocator-aware collections for PMEM-backed eviction stacks
//!
//! This module provides PMEM-aware alternatives to VecList and HashList that explicitly
//! use the PMEM eviction allocator instead of relying on the global allocator.
//! This is necessary because paper-cache is a library, and consuming binaries
//! may override the global allocator, which would break PMEM allocation.
//!
//! These implementations use a Vec-based architecture for simpler memory management
//! and reduced risk of segfaults compared to HashMap-based approaches.
//!
//! Allocator: `Hybrid` here is the same crate-wide `crate::Hybrid` alias
//! (`SlowObjects`, node-1 jemalloc arenas) that `BufferPMEM` and the other
//! PMEM features already use, so eviction-stack metadata lands on
//! the same node as the slow-tier values it indexes.

use hashbrown::HashMap;
use std::hash::{Hash, BuildHasher};
use crate::Hybrid;

// Use allocator-api2's Vec which works on stable Rust
use allocator_api2::vec::Vec;




// ============================================================================
// PmemVecList
// ============================================================================

/// Index into a PmemVecList - wraps a Vec index
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub struct PmemIndex(usize);

/// A doubly-linked list using Vec-based storage with PMEM allocator
/// 
/// Uses a Vec for node storage (simpler than HashMap) with a free list
/// to reuse deleted slots. This reduces allocator surface area and makes
/// debugging easier.
pub struct PmemVecList<T> {
    entries: Vec<Option<Node<T>>, Hybrid>,
    head: Option<usize>,
    free_list: Vec<usize, Hybrid>,
}

struct Node<T> {
    value: T,
    prev: Option<usize>,
    next: Option<usize>,
}

impl<T> PmemVecList<T> {
    /// Creates a new empty PmemVecList
    pub fn new() -> Self {
        PmemVecList {
            entries: Vec::new_in(Hybrid),
            head: None,
            free_list: Vec::new_in(Hybrid),
        }
    }

    /// Creates a new PmemVecList with specified capacity
    /// Use this to avoid reallocation crashes during high-throughput workloads
    pub fn with_capacity(capacity: usize) -> Self {
        PmemVecList {
            entries: Vec::with_capacity_in(capacity, Hybrid),
            head: None,
            free_list: Vec::with_capacity_in(capacity, Hybrid),
        }
    }
    
    /// Returns the index of the front element, if any
    pub fn front_index(&self) -> Option<PmemIndex> {
        self.head.map(PmemIndex)
    }
    
    /// Returns a reference to the front element, if any
    pub fn front(&self) -> Option<&T> {
        self.head.and_then(|idx| {
            self.entries.get(idx).and_then(|opt_node| {
                opt_node.as_ref().map(|node| &node.value)
            })
        })
    }
    
    /// Returns a reference to the element at the given index
    pub fn get(&self, index: PmemIndex) -> Option<&T> {
        self.entries.get(index.0).and_then(|opt_node| {
            opt_node.as_ref().map(|node| &node.value)
        })
    }

    /// Returns a mutable reference to the element at the given index
    pub fn get_mut(&mut self, index: PmemIndex) -> Option<&mut T> {
        self.entries.get_mut(index.0).and_then(|opt_node| {
            opt_node.as_mut().map(|node| &mut node.value)
        })
    }

    /// Returns the index of the next element after the given index
    pub fn get_next_index(&self, index: PmemIndex) -> Option<PmemIndex> {
        self.entries.get(index.0).and_then(|opt_node| {
            opt_node.as_ref().and_then(|node| node.next.map(PmemIndex))
        })
    }
    
    /// Pushes a value to the front of the list
    pub fn push_front(&mut self, value: T) -> PmemIndex {
        let node = Node {
            value,
            prev: None,
            next: self.head,
        };
        
        // Allocate index: reuse from free list or append
        let idx = if let Some(free_idx) = self.free_list.pop() {
            // Safety: Use ptr::write to overwrite the freed slot without running the drop
            // impl on its previous contents. When a slot is freed via remove(), we set it
            // to None. Reusing that slot via assignment would drop the None — which is
            // fine — but if PMEM returned uninitialized bytes, the memory at that location
            // might not be a valid Option<Node<T>> bitpattern, causing UB on drop.
            // ptr::write bypasses the drop path entirely, making the write safe.
            unsafe {
                std::ptr::write(&mut self.entries[free_idx], Some(node));
            }
            free_idx
        } else {
            let idx = self.entries.len();
            self.entries.push(Some(node));
            idx
        };
        
        // Update old head's prev pointer
        if let Some(old_head) = self.head {
            if let Some(Some(old_head_node)) = self.entries.get_mut(old_head) {
                old_head_node.prev = Some(idx);
            }
        }
        
        self.head = Some(idx);
        PmemIndex(idx)
    }
    
    /// Inserts a value after the given index
    pub fn insert_after(&mut self, index: PmemIndex, value: T) -> PmemIndex {
        let next_idx = self.entries.get(index.0)
            .and_then(|opt_node| opt_node.as_ref())
            .and_then(|node| node.next);
        
        let node = Node {
            value,
            prev: Some(index.0),
            next: next_idx,
        };
        
        // Allocate index: reuse from free list or append
        let idx = if let Some(free_idx) = self.free_list.pop() {
            // Safety: Use ptr::write to prevent drop of potentially invalid data when
            // reusing freed slots. See push_front for full explanation.
            unsafe {
                std::ptr::write(&mut self.entries[free_idx], Some(node));
            }
            free_idx
        } else {
            let idx = self.entries.len();
            self.entries.push(Some(node));
            idx
        };
        
        // Update previous node's next pointer
        if let Some(Some(prev_node)) = self.entries.get_mut(index.0) {
            prev_node.next = Some(idx);
        }
        
        // Update next node's prev pointer if it exists
        if let Some(next) = next_idx {
            if let Some(Some(next_node)) = self.entries.get_mut(next) {
                next_node.prev = Some(idx);
            }
        }

        PmemIndex(idx)
    }

    /// Inserts a value immediately before the given index. Implemented in terms
    /// of the existing `insert_after`/`push_front` primitives (no new pointer
    /// bookkeeping): if `index` has a predecessor, insert after it; otherwise
    /// `index` is the head, so this is a `push_front`.
    pub fn insert_before(&mut self, index: PmemIndex, value: T) -> PmemIndex {
        let prev_idx = self.entries.get(index.0)
            .and_then(|opt_node| opt_node.as_ref())
            .and_then(|node| node.prev);

        match prev_idx {
            Some(prev) => self.insert_after(PmemIndex(prev), value),
            None => self.push_front(value),
        }
    }

    /// Appends a value to the back of the list. The list tracks only `head`, so
    /// this walks `next` to the last node and reuses `insert_after`; the walk is
    /// O(len) but `PmemVecList` is only used for frequency-bucket chains (one
    /// node per distinct frequency), and only `insert_at`'s fallback hits this.
    pub fn push_back(&mut self, value: T) -> PmemIndex {
        let Some(mut idx) = self.head else {
            return self.push_front(value);
        };

        while let Some(next) = self.entries.get(idx)
            .and_then(|opt_node| opt_node.as_ref())
            .and_then(|node| node.next)
        {
            idx = next;
        }

        self.insert_after(PmemIndex(idx), value)
    }

    /// Removes the element at the given index
    pub fn remove(&mut self, index: PmemIndex) -> Option<T> {
        // take() replaces the value with None and returns the old value.
        // This is generally safe unless the memory was completely corrupted by realloc.
        let node = self.entries.get_mut(index.0)?.take()?;
        
        // Update prev node's next pointer
        if let Some(prev) = node.prev {
            if let Some(Some(prev_node)) = self.entries.get_mut(prev) {
                prev_node.next = node.next;
            }
        } else {
            self.head = node.next;
        }
        
        // Update next node's prev pointer
        if let Some(next) = node.next {
            if let Some(Some(next_node)) = self.entries.get_mut(next) {
                next_node.prev = node.prev;
            }
        }
        
        // Add index to free list for reuse
        self.free_list.push(index.0);
        
        Some(node.value)
    }
    
    /// Clears the list
    pub fn clear(&mut self) {
        self.entries.clear();
        self.head = None;
        self.free_list.clear();
    }
}

impl<T> Default for PmemVecList<T> {
    fn default() -> Self {
        Self::new()
    }
}



// ============================================================================
// PmemHashList
// ============================================================================

/// A hash-based doubly-linked list using Vec + HashMap with PMEM allocator
/// 
/// Uses Vec for node storage (simpler than two HashMaps) with HashMap for O(1) lookup.
/// Includes free list to reuse deleted slots.
pub struct PmemHashList<T, S> {
    entries: Vec<Option<Node<T>>, Hybrid>,
    lookup: HashMap<T, usize, S, Hybrid>,
    head: Option<usize>,
    tail: Option<usize>,
    free_list: Vec<usize, Hybrid>,
}

impl<T, S> PmemHashList<T, S>
where
    T: Hash + Eq + Clone,
    S: BuildHasher,
{
    /// Creates a new empty PmemHashList with the given hasher
    pub fn with_hasher(hasher: S) -> Self {
        PmemHashList {
            entries: Vec::new_in(Hybrid),
            lookup: HashMap::with_hasher_in(hasher, Hybrid),
            head: None,
            tail: None,
            free_list: Vec::new_in(Hybrid),
        }
    }

    /// Creates a new PmemHashList with capacity and hasher
    pub fn with_capacity_and_hasher(capacity: usize, hasher: S) -> Self {
        PmemHashList {
            entries: Vec::with_capacity_in(capacity, Hybrid),
            lookup: HashMap::with_capacity_and_hasher_in(capacity, hasher, Hybrid),
            head: None,
            tail: None,
            free_list: Vec::with_capacity_in(capacity, Hybrid),
        }
    }
    
    /// Returns true if the list is empty
    pub fn is_empty(&self) -> bool {
        self.lookup.is_empty()
    }
    
    /// Returns the number of elements in the list
    pub fn len(&self) -> usize {
        self.lookup.len()
    }
    
    /// Returns true if the list contains the given value
    pub fn contains(&self, value: &T) -> bool {
        self.lookup.contains_key(value)
    }

    /// Returns a reference to the value at the front (head) of the list.
    pub fn front(&self) -> Option<&T> {
        self.head.and_then(|idx| {
            self.entries.get(idx).and_then(|opt_node| {
                opt_node.as_ref().map(|node| &node.value)
            })
        })
    }

    /// Returns a reference to the value at the back (tail) of the list.
    pub fn back(&self) -> Option<&T> {
        self.tail.and_then(|idx| {
            self.entries.get(idx).and_then(|opt_node| {
                opt_node.as_ref().map(|node| &node.value)
            })
        })
    }

    /// Returns the value immediately before `value` — i.e. its neighbor toward
    /// the head — or `None` if `value` is absent or is itself the head. Matches
    /// `kwik::collections::HashList::before` as used by `LruHybridStack`.
    pub fn before(&self, value: &T) -> Option<&T> {
        let idx = *self.lookup.get(value)?;
        let prev = self.entries.get(idx)
            .and_then(|opt_node| opt_node.as_ref())
            .and_then(|node| node.prev)?;

        self.entries.get(prev).and_then(|opt_node| {
            opt_node.as_ref().map(|node| &node.value)
        })
    }

    /// Moves an existing value to the front of the list by SPLICING the node
    /// in place. A no-op if `value` is absent or already at the front.
    /// Matches `kwik::collections::HashList::move_front`'s `&T` signature.
    ///
    /// Deliberately NOT `remove` + `push_front` (which an earlier version
    /// was): that pattern costs two hash probes, a free-list round trip and a
    /// node rewrite per call, and this method runs once per GET hit on the
    /// stacks that use this type. With the list in far memory those are
    /// dependent CXL round trips, and the measured effect was the policy
    /// worker falling to 48.0k demotions/s against 91.5k in DRAM (-47.5%),
    /// which let the cache overrun its budget -- 133% of max_size on the LFU
    /// baseline, an oom-kill on the LRU baseline. The splice touches at most
    /// four nodes and neither the hash map nor the free list.
    pub fn move_front(&mut self, value: &T) {
        let Some(&idx) = self.lookup.get(value) else { return };
        if self.head == Some(idx) {
            return;
        }

        // Unlink from the current position.
        let (prev, next) = {
            let Some(Some(node)) = self.entries.get(idx) else { return };
            (node.prev, node.next)
        };
        if let Some(prev_idx) = prev {
            if let Some(Some(prev_node)) = self.entries.get_mut(prev_idx) {
                prev_node.next = next;
            }
        }
        if let Some(next_idx) = next {
            if let Some(Some(next_node)) = self.entries.get_mut(next_idx) {
                next_node.prev = prev;
            }
        } else {
            // The node was the tail; its predecessor is the tail now.
            self.tail = prev;
        }

        // Relink at the front.
        let old_head = self.head;
        if let Some(Some(node)) = self.entries.get_mut(idx) {
            node.prev = None;
            node.next = old_head;
        }
        if let Some(old_head_idx) = old_head {
            if let Some(Some(old_head_node)) = self.entries.get_mut(old_head_idx) {
                old_head_node.prev = Some(idx);
            }
        }
        self.head = Some(idx);
    }

    /// Pushes a value to the front of the list
    pub fn push_front(&mut self, value: T) {
        // If value already exists, remove it first
        if self.lookup.contains_key(&value) {
            self.remove(&value);
        }
        
        let node = Node {
            value: value.clone(),
            prev: None,
            next: self.head,
        };
        
        // Allocate index: reuse from free list or append
        let idx = if let Some(free_idx) = self.free_list.pop() {
            // Safety: Use ptr::write to prevent drop of potentially invalid data when
            // reusing freed slots. See PmemVecList::push_front for full explanation.
            unsafe {
                std::ptr::write(&mut self.entries[free_idx], Some(node));
            }
            free_idx
        } else {
            let idx = self.entries.len();
            self.entries.push(Some(node));
            idx
        };
        
        // Update old head's prev pointer
        if let Some(old_head_idx) = self.head {
            if let Some(Some(old_head_node)) = self.entries.get_mut(old_head_idx) {
                old_head_node.prev = Some(idx);
            }
        }
        
        self.head = Some(idx);
        
        // If list was empty, this is also the tail
        if self.tail.is_none() {
            self.tail = Some(idx);
        }
        
        self.lookup.insert(value, idx);
    }
    
    /// Pushes a value onto the back (tail) of the list.
    ///
    /// The mirror of `push_front`, added for
    /// `TwoQFastAdmissionReprieveHybridStack`, which splices a reprieved
    /// one-access key onto the LRU tail of its main queue -- the DRAM
    /// `kwik::collections::HashList` this type shadows already has one, so
    /// without it that stack could not compile under `eviction_stacks_pmem`.
    ///
    /// Same existing-value handling as `push_front` (remove first, so a key
    /// is never present twice) and the same `ptr::write` safety reasoning
    /// when reusing a freed slot -- see `PmemVecList::push_front` for the
    /// full explanation of why assignment would be unsound there.
    pub fn push_back(&mut self, value: T) {
        // If value already exists, remove it first
        if self.lookup.contains_key(&value) {
            self.remove(&value);
        }

        let node = Node {
            value: value.clone(),
            prev: self.tail,
            next: None,
        };

        // Allocate index: reuse from free list or append
        let idx = if let Some(free_idx) = self.free_list.pop() {
            unsafe {
                std::ptr::write(&mut self.entries[free_idx], Some(node));
            }
            free_idx
        } else {
            let idx = self.entries.len();
            self.entries.push(Some(node));
            idx
        };

        // Update old tail's next pointer
        if let Some(old_tail_idx) = self.tail {
            if let Some(Some(old_tail_node)) = self.entries.get_mut(old_tail_idx) {
                old_tail_node.next = Some(idx);
            }
        }

        self.tail = Some(idx);

        // If list was empty, this is also the head
        if self.head.is_none() {
            self.head = Some(idx);
        }

        self.lookup.insert(value, idx);
    }

    /// Pops a value from the back of the list
    pub fn pop_back(&mut self) -> Option<T> {
        let tail_idx = self.tail?;
        let node = self.entries.get_mut(tail_idx)?.take()?;
        
        self.lookup.remove(&node.value);
        self.tail = node.prev;
        
        // Update new tail's next pointer
        if let Some(new_tail_idx) = self.tail {
            if let Some(Some(new_tail_node)) = self.entries.get_mut(new_tail_idx) {
                new_tail_node.next = None;
            }
        } else {
            // List is now empty
            self.head = None;
        }
        
        // Add index to free list for reuse
        self.free_list.push(tail_idx);
        
        Some(node.value)
    }
    
    /// Removes a value from the list
    pub fn remove(&mut self, value: &T) -> Option<T> {
        let idx = self.lookup.remove(value)?;
        let node = self.entries.get_mut(idx)?.take()?;
        
        // Update prev node's next pointer
        if let Some(prev_idx) = node.prev {
            if let Some(Some(prev_node)) = self.entries.get_mut(prev_idx) {
                prev_node.next = node.next;
            }
        } else {
            self.head = node.next;
        }
        
        // Update next node's prev pointer
        if let Some(next_idx) = node.next {
            if let Some(Some(next_node)) = self.entries.get_mut(next_idx) {
                next_node.prev = node.prev;
            }
        } else {
            self.tail = node.prev;
        }
        
        // Add index to free list for reuse
        self.free_list.push(idx);
        
        Some(node.value)
    }
    
    /// Clears the list
    pub fn clear(&mut self) {
        self.lookup.clear();
        self.entries.clear();
        self.head = None;
        self.tail = None;
        self.free_list.clear();
    }
}

impl<T, S> Default for PmemHashList<T, S>
where
    T: Hash + Eq + Clone,
    S: BuildHasher + Default,
{
    fn default() -> Self {
        Self::with_hasher(S::default())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::hash::BuildHasherDefault;
    use nohash_hasher::NoHashHasher;

    type TestHasher = BuildHasherDefault<NoHashHasher<u64>>;

    #[test]
    fn pmem_vec_list_basic_operations() {
        let mut list = PmemVecList::<u64>::new();
        
        // Test empty list
        assert!(list.front().is_none());
        assert!(list.front_index().is_none());
        
        // Test push_front and front
        let idx1 = list.push_front(1);
        assert_eq!(list.front(), Some(&1));
        assert_eq!(list.front_index(), Some(idx1));
        
        let idx2 = list.push_front(2);
        assert_eq!(list.front(), Some(&2));
        assert_eq!(list.front_index(), Some(idx2));
        
        // Test get_mut
        assert_eq!(list.get_mut(idx1), Some(&mut 1));
        assert_eq!(list.get_mut(idx2), Some(&mut 2));
        
        // Test get_next_index
        assert_eq!(list.get_next_index(idx2), Some(idx1));
        assert_eq!(list.get_next_index(idx1), None);
    }

    #[test]
    fn pmem_vec_list_insert_after() {
        let mut list = PmemVecList::<u64>::new();
        
        let idx1 = list.push_front(1);
        let idx3 = list.insert_after(idx1, 3);
        let idx2 = list.insert_after(idx1, 2);
        
        // List should be: 1 -> 2 -> 3
        assert_eq!(list.front(), Some(&1));
        assert_eq!(list.get_next_index(idx1), Some(idx2));
        assert_eq!(list.get_next_index(idx2), Some(idx3));
        assert_eq!(list.get_next_index(idx3), None);
    }

    #[test]
    fn pmem_vec_list_remove_and_free_list() {
        let mut list = PmemVecList::<u64>::new();
        
        let idx1 = list.push_front(1);
        let idx2 = list.push_front(2);
        let idx3 = list.push_front(3);
        
        // Remove middle element
        assert_eq!(list.remove(idx2), Some(2));
        assert_eq!(list.front(), Some(&3));
        assert_eq!(list.get_next_index(idx3), Some(idx1));
        
        // Add new element - should reuse freed slot
        let idx4 = list.push_front(4);
        assert_eq!(list.front(), Some(&4));
        
        // Remove head
        assert_eq!(list.remove(idx4), Some(4));
        assert_eq!(list.front(), Some(&3));
        
        // Remove tail
        assert_eq!(list.remove(idx1), Some(1));
        assert_eq!(list.front(), Some(&3));
        assert_eq!(list.get_next_index(idx3), None);
    }

    #[test]
    fn pmem_vec_list_clear() {
        let mut list = PmemVecList::<u64>::new();
        
        list.push_front(1);
        list.push_front(2);
        list.push_front(3);
        
        list.clear();
        
        assert!(list.front().is_none());
        assert!(list.front_index().is_none());
    }

    #[test]
    fn pmem_hash_list_basic_operations() {
        let mut list = PmemHashList::<u64, TestHasher>::with_hasher(TestHasher::default());
        
        // Test empty list
        assert!(list.is_empty());
        assert_eq!(list.len(), 0);
        assert!(!list.contains(&1));
        
        // Test push_front
        list.push_front(1);
        assert!(!list.is_empty());
        assert_eq!(list.len(), 1);
        assert!(list.contains(&1));
        
        list.push_front(2);
        assert_eq!(list.len(), 2);
        assert!(list.contains(&2));
    }

    #[test]
    fn pmem_hash_list_pop_back() {
        let mut list = PmemHashList::<u64, TestHasher>::with_hasher(TestHasher::default());
        
        list.push_front(1);
        list.push_front(2);
        list.push_front(3);
        
        // List is: 3 -> 2 -> 1
        assert_eq!(list.pop_back(), Some(1));
        assert_eq!(list.len(), 2);
        assert!(!list.contains(&1));
        
        assert_eq!(list.pop_back(), Some(2));
        assert_eq!(list.len(), 1);
        
        assert_eq!(list.pop_back(), Some(3));
        assert_eq!(list.len(), 0);
        assert!(list.is_empty());
        
        assert_eq!(list.pop_back(), None);
    }

    #[test]
    fn pmem_hash_list_remove() {
        let mut list = PmemHashList::<u64, TestHasher>::with_hasher(TestHasher::default());
        
        list.push_front(1);
        list.push_front(2);
        list.push_front(3);
        
        // Remove middle element
        assert_eq!(list.remove(&2), Some(2));
        assert_eq!(list.len(), 2);
        assert!(!list.contains(&2));
        assert!(list.contains(&1));
        assert!(list.contains(&3));
        
        // Remove non-existent element
        assert_eq!(list.remove(&99), None);
        assert_eq!(list.len(), 2);
    }

    #[test]
    fn pmem_hash_list_push_existing() {
        let mut list = PmemHashList::<u64, TestHasher>::with_hasher(TestHasher::default());
        
        list.push_front(1);
        list.push_front(2);
        list.push_front(3);
        
        // Push existing element - should remove and re-add at front
        list.push_front(2);
        assert_eq!(list.len(), 3);
        
        // Pop order should be: 2 is at head, then 3, then 1
        assert_eq!(list.pop_back(), Some(1));
        assert_eq!(list.pop_back(), Some(3));
        assert_eq!(list.pop_back(), Some(2));
    }

    #[test]
    fn pmem_hash_list_free_list_reuse() {
        let mut list = PmemHashList::<u64, TestHasher>::with_hasher(TestHasher::default());
        
        // Add and remove multiple times to test free list
        for i in 0..10 {
            list.push_front(i);
        }
        
        assert_eq!(list.len(), 10);
        
        // Remove half
        for i in 0..5 {
            list.remove(&i);
        }
        
        assert_eq!(list.len(), 5);
        
        // Add more - should reuse freed slots
        for i in 100..105 {
            list.push_front(i);
        }
        
        assert_eq!(list.len(), 10);
    }

    #[test]
    fn pmem_hash_list_clear() {
        let mut list = PmemHashList::<u64, TestHasher>::with_hasher(TestHasher::default());
        
        list.push_front(1);
        list.push_front(2);
        list.push_front(3);
        
        list.clear();
        
        assert!(list.is_empty());
        assert_eq!(list.len(), 0);
        assert!(!list.contains(&1));
        assert!(!list.contains(&2));
        assert!(!list.contains(&3));
    }
}

#[cfg(test)]
mod structural_invariants {
    use super::*;
    use crate::NoHasher;

    /// Walks the list and checks it against `lookup`. Returns (forward, backward).
    fn walk(l: &PmemHashList<crate::HashedKey, NoHasher>) -> (usize, usize) {
        let mut n = 0usize;
        let mut cur = l.head;
        let mut seen = std::collections::HashSet::new();
        while let Some(i) = cur {
            assert!(seen.insert(i), "cycle in forward links at slot {i}");
            let node = l.entries[i].as_ref().expect("forward link into a freed slot");
            n += 1;
            cur = node.next;
        }
        let mut m = 0usize;
        let mut cur = l.tail;
        let mut seen2 = std::collections::HashSet::new();
        while let Some(i) = cur {
            assert!(seen2.insert(i), "cycle in backward links at slot {i}");
            let node = l.entries[i].as_ref().expect("backward link into a freed slot");
            m += 1;
            cur = node.prev;
        }
        (n, m)
    }

    /// Randomised push_front / move_front / remove / pop_back, checking after
    /// every operation that the links and the lookup map still agree.
    #[test]
    fn links_and_lookup_stay_consistent_under_churn() {
        let mut l: PmemHashList<crate::HashedKey, NoHasher> = PmemHashList::with_hasher(NoHasher::default());
        let mut x: u64 = 0x243F_6A88_85A3_08D3;
        for step in 0..50_000u64 {
            x ^= x << 13; x ^= x >> 7; x ^= x << 17;
            let key = (x >> 11) % 500;
            match step % 8 {
                0..=3 => l.push_front(key),
                4..=5 => l.move_front(&key),
                6 => { let _ = l.remove(&key); },
                _ => { let _ = l.pop_back(); },
            }
            let (fwd, bwd) = walk(&l);
            assert_eq!(
                fwd, l.len(),
                "step {step}: {fwd} nodes reachable from head but lookup holds {} \
                 -- keys in lookup that are not in the list can never be evicted",
                l.len(),
            );
            assert_eq!(bwd, l.len(), "step {step}: backward walk sees {bwd}, lookup {}", l.len());
            for k in l.lookup.keys() {
                assert!(l.contains(k), "step {step}: key {k} in lookup but not findable");
            }
        }
    }
}
