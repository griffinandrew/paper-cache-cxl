

/*
 * Copyright (c) Kia Shakiba
 *
 * This source code is licensed under the GNU AGPLv3 license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! Custom allocator-aware collections for PMEM-backed eviction stacks
//!
//! This module provides PMEM-aware alternatives to VecList and HashList that explicitly
//! use the HybridObjects allocator instead of relying on the global allocator.
//! This is necessary because paper-cache is a library, and consuming binaries
//! may override the global allocator, which would break PMEM allocation.
//!
//! These implementations use a Vec-based architecture for simpler memory management
//! and reduced risk of segfaults compared to HashMap-based approaches.

use hashbrown::HashMap;
use std::hash::{Hash, BuildHasher};
use crate::allocator::HybridObjects;

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
    entries: Vec<Option<Node<T>>, HybridObjects>,
    head: Option<usize>,
    free_list: Vec<usize, HybridObjects>,
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
            entries: Vec::new_in(HybridObjects),
            head: None,
            free_list: Vec::new_in(HybridObjects),
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
            self.entries[free_idx] = Some(node);
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
            self.entries[free_idx] = Some(node);
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
    
    /// Removes the element at the given index
    pub fn remove(&mut self, index: PmemIndex) -> Option<T> {
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
    entries: Vec<Option<Node<T>>, HybridObjects>,
    lookup: HashMap<T, usize, S, HybridObjects>,
    head: Option<usize>,
    tail: Option<usize>,
    free_list: Vec<usize, HybridObjects>,
}

impl<T, S> PmemHashList<T, S>
where
    T: Hash + Eq + Clone,
    S: BuildHasher,
{
    /// Creates a new empty PmemHashList with the given hasher
    pub fn with_hasher(hasher: S) -> Self {
        PmemHashList {
            entries: Vec::new_in(HybridObjects),
            lookup: HashMap::with_hasher_in(hasher, HybridObjects),
            head: None,
            tail: None,
            free_list: Vec::new_in(HybridObjects),
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
            self.entries[free_idx] = Some(node);
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