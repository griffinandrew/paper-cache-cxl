/* 

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

use hashbrown::HashMap;
use std::hash::{Hash, BuildHasher, BuildHasherDefault};
use nohash_hasher::NoHashHasher;
use crate::allocator::HybridObjects;

/// Type alias for the hasher used in PmemVecList's internal HashMap
type VecListHasher = BuildHasherDefault<NoHashHasher<usize>>;

/// Index into a PmemVecList - wraps the node ID
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub struct PmemIndex(usize);

/// A doubly-linked list implementation using HashMap with PMEM allocator
/// 
/// This replaces VecList<T> with an implementation that uses HybridObjects allocator
/// for all its internal storage, making it safe to use in library contexts where the
/// global allocator may be overridden.
pub struct PmemVecList<T> {
    nodes: HashMap<usize, Node<T>, VecListHasher, HybridObjects>,
    head: Option<usize>,
    next_id: usize,
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
            nodes: HashMap::with_hasher_in(VecListHasher::default(), HybridObjects),
            head: None,
            next_id: 0,
        }
    }
    
    /// Returns the index of the front element, if any
    pub fn front_index(&self) -> Option<PmemIndex> {
        self.head.map(PmemIndex)
    }
    
    /// Returns a reference to the front element, if any
    pub fn front(&self) -> Option<&T> {
        self.head.and_then(|id| self.nodes.get(&id).map(|node| &node.value))
    }
    
    /// Returns a mutable reference to the element at the given index
    pub fn get_mut(&mut self, index: PmemIndex) -> Option<&mut T> {
        self.nodes.get_mut(&index.0).map(|node| &mut node.value)
    }
    
    /// Returns the index of the next element after the given index
    pub fn get_next_index(&self, index: PmemIndex) -> Option<PmemIndex> {
        self.nodes.get(&index.0).and_then(|node| node.next.map(PmemIndex))
    }
    
    /// Pushes a value to the front of the list
    pub fn push_front(&mut self, value: T) -> PmemIndex {
        let id = self.next_id;
        self.next_id += 1;
        
        let node = Node {
            value,
            prev: None,
            next: self.head,
        };
        
        self.nodes.insert(id, node);
        
        if let Some(old_head) = self.head {
            if let Some(old_head_node) = self.nodes.get_mut(&old_head) {
                old_head_node.prev = Some(id);
            }
        }
        
        self.head = Some(id);
        
        PmemIndex(id)
    }
    
    /// Inserts a value after the given index
    pub fn insert_after(&mut self, index: PmemIndex, value: T) -> PmemIndex {
        let id = self.next_id;
        self.next_id += 1;
        
        let next_id = self.nodes.get(&index.0).and_then(|node| node.next);
        
        let node = Node {
            value,
            prev: Some(index.0),
            next: next_id,
        };
        
        self.nodes.insert(id, node);
        
        // Update the previous node
        if let Some(prev_node) = self.nodes.get_mut(&index.0) {
            prev_node.next = Some(id);
        }
        
        // Update the next node if it exists
        if let Some(next) = next_id {
            if let Some(next_node) = self.nodes.get_mut(&next) {
                next_node.prev = Some(id);
            }
        }
        
        PmemIndex(id)
    }
    
    /// Removes the element at the given index
    pub fn remove(&mut self, index: PmemIndex) -> Option<T> {
        let node = self.nodes.remove(&index.0)?;
        
        // Update prev node's next pointer
        if let Some(prev) = node.prev {
            if let Some(prev_node) = self.nodes.get_mut(&prev) {
                prev_node.next = node.next;
            }
        } else {
            self.head = node.next;
        }
        
        // Update next node's prev pointer
        if let Some(next) = node.next {
            if let Some(next_node) = self.nodes.get_mut(&next) {
                next_node.prev = node.prev;
            }
        }
        
        Some(node.value)
    }
    
    /// Clears the list
    pub fn clear(&mut self) {
        self.nodes.clear();
        self.head = None;
    }
}

impl<T> Default for PmemVecList<T> {
    fn default() -> Self {
        Self::new()
    }
}

/// A hash-based doubly-linked list implementation using PMEM allocator
/// 
/// This replaces HashList<T, S> with an implementation that uses HybridObjects allocator
/// for all its internal storage, making it safe to use in library contexts.
pub struct PmemHashList<T, S> {
    map: HashMap<T, ListNode<T>, S, HybridObjects>,
    head: Option<T>,
    tail: Option<T>,
}

struct ListNode<T> {
    prev: Option<T>,
    next: Option<T>,
}

impl<T, S> PmemHashList<T, S>
where
    T: Hash + Eq + Clone,
    S: BuildHasher,
{
    /// Creates a new empty PmemHashList with the given hasher
    pub fn with_hasher(hasher: S) -> Self {
        PmemHashList {
            map: HashMap::with_hasher_in(hasher, HybridObjects),
            head: None,
            tail: None,
        }
    }
    
    /// Returns true if the list is empty
    pub fn is_empty(&self) -> bool {
        self.map.is_empty()
    }
    
    /// Pushes a value to the front of the list
    pub fn push_front(&mut self, value: T) {
        if self.map.contains_key(&value) {
            // Value already exists, remove it first
            self.remove(&value);
        }
        
        let node = ListNode {
            prev: None,
            next: self.head.clone(),
        };
        
        if let Some(old_head) = &self.head {
            if let Some(old_head_node) = self.map.get_mut(old_head) {
                old_head_node.prev = Some(value.clone());
            }
        }
        
        self.head = Some(value.clone());
        
        if self.tail.is_none() {
            self.tail = Some(value.clone());
        }
        
        self.map.insert(value, node);
    }
    
    /// Pops a value from the back of the list
    pub fn pop_back(&mut self) -> Option<T> {
        let tail_value = self.tail.clone()?;
        self.remove(&tail_value).map(|_| tail_value)
    }
    
    /// Removes a value from the list
    pub fn remove(&mut self, value: &T) -> Option<T> {
        let node = self.map.remove(value)?;
        
        // Update prev node's next pointer
        if let Some(prev) = &node.prev {
            if let Some(prev_node) = self.map.get_mut(prev) {
                prev_node.next = node.next.clone();
            }
        } else {
            self.head = node.next.clone();
        }
        
        // Update next node's prev pointer
        if let Some(next) = &node.next {
            if let Some(next_node) = self.map.get_mut(next) {
                next_node.prev = node.prev.clone();
            }
        } else {
            self.tail = node.prev.clone();
        }
        
        Some(value.clone())
    }


    */


use hashbrown::HashMap;
use std::hash::{Hash, BuildHasher};
use crate::allocator::HybridObjects;

/// A hash-based doubly-linked list implementation using PMEM allocator
pub struct PmemHashList<T, S> {
    // Map from value to its node ID
    value_to_id: HashMap<T, usize, S, HybridObjects>,
    // Map from node ID to the actual node
    nodes: HashMap<usize, ListNode<T>, S, HybridObjects>,
    head: Option<usize>,
    tail: Option<usize>,
    next_id: usize,
}

struct ListNode<T> {
    value: T,
    prev: Option<usize>,
    next: Option<usize>,
}

impl<T, S> PmemHashList<T, S>
where
    T: Hash + Eq + Clone,
    S: BuildHasher + Clone,
{
    pub fn with_hasher(hasher: S) -> Self {
        PmemHashList {
            value_to_id: HashMap::with_hasher_in(hasher.clone(), HybridObjects),
            nodes: HashMap::with_hasher_in(hasher, HybridObjects),
            head: None,
            tail: None,
            next_id: 0,
        }
    }
    
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }
    
    pub fn len(&self) -> usize {
        self.nodes.len()
    }
    
    pub fn contains(&self, value: &T) -> bool {
        self.value_to_id.contains_key(value)
    }
    
    pub fn push_front(&mut self, value: T) {
        // Remove if already exists
        if self.value_to_id.contains_key(&value) {
            self.remove(&value);
        }
        
        let id = self.next_id;
        self.next_id += 1;
        
        let node = ListNode {
            value: value.clone(),
            prev: None,
            next: self.head,
        };
        
        // Update old head's prev pointer
        if let Some(old_head_id) = self.head {
            if let Some(old_head_node) = self.nodes.get_mut(&old_head_id) {
                old_head_node.prev = Some(id);
            }
        }
        
        self.head = Some(id);
        
        if self.tail.is_none() {
            self.tail = Some(id);
        }
        
        self.value_to_id.insert(value, id);
        self.nodes.insert(id, node);
    }
    
    pub fn pop_back(&mut self) -> Option<T> {
        let tail_id = self.tail?;
        let node = self.nodes.remove(&tail_id)?;
        self.value_to_id.remove(&node.value);
        
        // Update tail
        self.tail = node.prev;
        
        // Update new tail's next pointer
        if let Some(new_tail_id) = self.tail {
            if let Some(new_tail_node) = self.nodes.get_mut(&new_tail_id) {
                new_tail_node.next = None;
            }
        } else {
            // List is now empty
            self.head = None;
        }
        
        Some(node.value)
    }
    
    pub fn remove(&mut self, value: &T) -> Option<T> {
        let id = self.value_to_id.remove(value)?;
        let node = self.nodes.remove(&id)?;
        
        // Update prev node's next pointer
        if let Some(prev_id) = node.prev {
            if let Some(prev_node) = self.nodes.get_mut(&prev_id) {
                prev_node.next = node.next;
            }
        } else {
            self.head = node.next;
        }
        
        // Update next node's prev pointer
        if let Some(next_id) = node.next {
            if let Some(next_node) = self.nodes.get_mut(&next_id) {
                next_node.prev = node.prev;
            }
        } else {
            self.tail = node.prev;
        }
        
        Some(node.value)
    }
    
    pub fn move_front(&mut self, value: &T) {
        // Used by LRU - moves existing element to front
        if let Some(&id) = self.value_to_id.get(value) {
            // Only need to move if not already at head
            if self.head != Some(id) {
                // ⚠️ BUG FIX: Store prev/next BEFORE mutable borrows
                let (prev, next) = {
                    let node = self.nodes.get(&id).unwrap();
                    (node.prev, node.next)
                };
                
                // Update surrounding nodes
                if let Some(prev_id) = prev {
                    if let Some(prev_node) = self.nodes.get_mut(&prev_id) {
                        prev_node.next = next;
                    }
                }
                
                if let Some(next_id) = next {
                    if let Some(next_node) = self.nodes.get_mut(&next_id) {
                        next_node.prev = prev;
                    }
                } else {
                    self.tail = prev;
                }
                
                // Move to front
                if let Some(node) = self.nodes.get_mut(&id) {
                    node.prev = None;
                    node.next = self.head;
                }
                
                if let Some(old_head_id) = self.head {
                    if let Some(old_head_node) = self.nodes.get_mut(&old_head_id) {
                        old_head_node.prev = Some(id);
                    }
                }
                
                self.head = Some(id);
            }
        }
    }
    
    pub fn clear(&mut self) {
        self.value_to_id.clear();
        self.nodes.clear();
        self.head = None;
        self.tail = None;
    }
}

impl<T, S> Default for PmemHashList<T, S>
where
    T: Hash + Eq + Clone,
    S: BuildHasher + Clone + Default,
{
    fn default() -> Self {
        Self::with_hasher(S::default())
    }
}


