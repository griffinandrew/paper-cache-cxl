/*
 * Copyright (c) Kia Shakiba
 *
 * This source code is licensed under the GNU AGPLv3 license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::{
    borrow::Borrow,
    hash::{Hash, BuildHasher},
    ptr::NonNull,
};

#[cfg(any(feature = "alloc_with_hash", feature = "alloc_api_exp"))]
use hashbrown::HashMap;
#[cfg(feature = "alloc_with_hash")]
use crate::allocator::HybridObjects;

#[cfg(not(any(feature = "alloc_with_hash", feature = "alloc_api_exp")))]
use std::collections::HashMap;

/// A hash list that combines a HashMap with a doubly-linked list.
/// When PMem allocator features are enabled, uses hashbrown with the HybridObjects allocator.
/// Otherwise, falls back to std::collections::HashMap.
pub struct PmemHashList<T, S> {
    #[cfg(feature = "alloc_with_hash")]
    map: HashMap<u64, NonNull<Node<T>>, S, HybridObjects>,
    
    #[cfg(feature = "alloc_api_exp")]
    map: HashMap<u64, NonNull<Node<T>>, S>,
    
    #[cfg(not(any(feature = "alloc_with_hash", feature = "alloc_api_exp")))]
    map: HashMap<u64, NonNull<Node<T>>, S>,
    
    head: Option<NonNull<Node<T>>>,
    tail: Option<NonNull<Node<T>>>,
    len: usize,
}

struct Node<T> {
    item: T,
    prev: Option<NonNull<Node<T>>>,
    next: Option<NonNull<Node<T>>>,
}

impl<T, S> PmemHashList<T, S>
where
    T: Hash + Eq,
    S: BuildHasher,
{
    /// Creates a new empty PmemHashList with the given hasher
    pub fn with_hasher(hash_builder: S) -> Self {
        #[cfg(feature = "alloc_with_hash")]
        let map = HashMap::with_hasher_in(hash_builder, HybridObjects);
        
        #[cfg(not(feature = "alloc_with_hash"))]
        let map = HashMap::with_hasher(hash_builder);
        
        PmemHashList {
            map,
            head: None,
            tail: None,
            len: 0,
        }
    }

    /// Returns the number of elements in the list
    pub fn len(&self) -> usize {
        self.len
    }

    /// Returns true if the list is empty
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Returns true if the list contains the given key
    pub fn contains<Q>(&self, key: &Q) -> bool
    where
        T: Borrow<Q>,
        Q: Hash + Eq + ?Sized,
    {
        // Need to hash the key properly
        let hash = self.hash_key(key);
        self.map.contains_key(&hash)
    }

    /// Pushes an item to the front of the list
    pub fn push_front(&mut self, item: T) {
        let hash = self.hash_item(&item);
        
        #[cfg(feature = "alloc_with_hash")]
        let node_ptr = {
            let node = Box::new_in(Node {
                item,
                prev: None,
                next: self.head,
            }, HybridObjects);
            // Convert Box<T, A> to raw pointer manually to avoid type issues
            let ptr = &*node as *const Node<T> as *mut Node<T>;
            std::mem::forget(node);
            unsafe { NonNull::new_unchecked(ptr) }
        };
        
        #[cfg(not(feature = "alloc_with_hash"))]
        let node_ptr = {
            let node = Box::new(Node {
                item,
                prev: None,
                next: self.head,
            });
            unsafe { NonNull::new_unchecked(Box::into_raw(node)) }
        };
        
        if let Some(head) = self.head {
            unsafe { (*head.as_ptr()).prev = Some(node_ptr); }
        } else {
            self.tail = Some(node_ptr);
        }
        
        self.head = Some(node_ptr);
        self.map.insert(hash, node_ptr);
        self.len += 1;
    }

    /// Pops an item from the front of the list
    pub fn pop_front(&mut self) -> Option<T> {
        self.head.map(|head_ptr| {
            #[cfg(feature = "alloc_with_hash")]
            let head = unsafe { Box::from_raw_in(head_ptr.as_ptr(), HybridObjects) };
            
            #[cfg(not(feature = "alloc_with_hash"))]
            let head = unsafe { Box::from_raw(head_ptr.as_ptr()) };
            
            let hash = self.hash_item(&head.item);
            
            self.head = head.next;
            if let Some(new_head) = self.head {
                unsafe { (*new_head.as_ptr()).prev = None; }
            } else {
                self.tail = None;
            }
            
            self.map.remove(&hash);
            self.len -= 1;
            head.item
        })
    }

    /// Pops an item from the back of the list
    pub fn pop_back(&mut self) -> Option<T> {
        self.tail.map(|tail_ptr| {
            #[cfg(feature = "alloc_with_hash")]
            let tail = unsafe { Box::from_raw_in(tail_ptr.as_ptr(), HybridObjects) };
            
            #[cfg(not(feature = "alloc_with_hash"))]
            let tail = unsafe { Box::from_raw(tail_ptr.as_ptr()) };
            
            let hash = self.hash_item(&tail.item);
            
            self.tail = tail.prev;
            if let Some(new_tail) = self.tail {
                unsafe { (*new_tail.as_ptr()).next = None; }
            } else {
                self.head = None;
            }
            
            self.map.remove(&hash);
            self.len -= 1;
            tail.item
        })
    }

    /// Moves an existing item to the front of the list
    pub fn move_front<Q>(&mut self, key: &Q)
    where
        T: Borrow<Q>,
        Q: Hash + Eq + ?Sized,
    {
        let hash = self.hash_key(key);
        
        if let Some(&node_ptr) = self.map.get(&hash) {
            // If already at front, nothing to do
            if Some(node_ptr) == self.head {
                return;
            }
            
            unsafe {
                let node = node_ptr.as_ptr();
                
                // Remove from current position
                if let Some(prev) = (*node).prev {
                    (*prev.as_ptr()).next = (*node).next;
                }
                if let Some(next) = (*node).next {
                    (*next.as_ptr()).prev = (*node).prev;
                }
                if Some(node_ptr) == self.tail {
                    self.tail = (*node).prev;
                }
                
                // Move to front
                (*node).prev = None;
                (*node).next = self.head;
                
                if let Some(head) = self.head {
                    (*head.as_ptr()).prev = Some(node_ptr);
                }
                
                self.head = Some(node_ptr);
            }
        }
    }

    /// Removes an item from the list
    pub fn remove<Q>(&mut self, key: &Q) -> Option<T>
    where
        T: Borrow<Q>,
        Q: Hash + Eq + ?Sized,
    {
        let hash = self.hash_key(key);
        
        self.map.remove(&hash).map(|node_ptr| {
            #[cfg(feature = "alloc_with_hash")]
            let node = unsafe { Box::from_raw_in(node_ptr.as_ptr(), HybridObjects) };
            
            #[cfg(not(feature = "alloc_with_hash"))]
            let node = unsafe { Box::from_raw(node_ptr.as_ptr()) };
            
            unsafe {
                if let Some(prev) = node.prev {
                    (*prev.as_ptr()).next = node.next;
                } else {
                    self.head = node.next;
                }
                
                if let Some(next) = node.next {
                    (*next.as_ptr()).prev = node.prev;
                } else {
                    self.tail = node.prev;
                }
            }
            
            self.len -= 1;
            node.item
        })
    }

    /// Clears the list
    pub fn clear(&mut self) {
        while self.pop_front().is_some() {}
    }

    /// Gets a reference to an item in the list
    pub fn get<Q>(&self, key: &Q) -> Option<&T>
    where
        T: Borrow<Q>,
        Q: Hash + Eq + ?Sized,
    {
        let hash = self.hash_key(key);
        self.map.get(&hash).map(|&node_ptr| unsafe { &(*node_ptr.as_ptr()).item })
    }

    /// Updates an item in the list using a closure
    pub fn update<Q, F>(&mut self, key: &Q, f: F)
    where
        T: Borrow<Q>,
        Q: Hash + Eq + ?Sized,
        F: FnOnce(&mut T),
    {
        let hash = self.hash_key(key);
        if let Some(&node_ptr) = self.map.get(&hash) {
            unsafe {
                f(&mut (*node_ptr.as_ptr()).item);
            }
        }
    }

    /// Gets a reference to the item before the given key
    pub fn before<Q>(&self, key: &Q) -> Option<&T>
    where
        T: Borrow<Q>,
        Q: Hash + Eq + ?Sized,
    {
        let hash = self.hash_key(key);
        self.map.get(&hash).and_then(|&node_ptr| {
            unsafe { (*node_ptr.as_ptr()).prev.map(|prev| &(*prev.as_ptr()).item) }
        })
    }

    /// Gets a reference to the last item in the list
    pub fn back(&self) -> Option<&T> {
        self.tail.map(|tail_ptr| unsafe { &(*tail_ptr.as_ptr()).item })
    }

    /// Helper to hash a key
    fn hash_key<Q>(&self, key: &Q) -> u64
    where
        T: Borrow<Q>,
        Q: Hash + Eq + ?Sized,
    {
        use std::hash::Hasher;
        let mut hasher = self.map.hasher().build_hasher();
        key.hash(&mut hasher);
        hasher.finish()
    }

    /// Helper to hash an item
    fn hash_item(&self, item: &T) -> u64 {
        use std::hash::Hasher;
        let mut hasher = self.map.hasher().build_hasher();
        item.hash(&mut hasher);
        hasher.finish()
    }
}

impl<T, S> Default for PmemHashList<T, S>
where
    T: Hash + Eq,
    S: BuildHasher + Default,
{
    fn default() -> Self {
        Self::with_hasher(S::default())
    }
}

impl<T, S> Drop for PmemHashList<T, S> {
    fn drop(&mut self) {
        // Clear all nodes manually without needing Hash + Eq bounds
        #[cfg(feature = "alloc_with_hash")]
        {
            while let Some(head_ptr) = self.head {
                let head = unsafe { Box::from_raw_in(head_ptr.as_ptr(), HybridObjects) };
                self.head = head.next;
            }
        }
        
        #[cfg(not(feature = "alloc_with_hash"))]
        {
            while let Some(head_ptr) = self.head {
                let head = unsafe { Box::from_raw(head_ptr.as_ptr()) };
                self.head = head.next;
            }
        }
        
        self.tail = None;
        self.len = 0;
    }
}

// Safety: PmemHashList is Send if T is Send
unsafe impl<T: Send, S: Send> Send for PmemHashList<T, S> {}

// Safety: PmemHashList is Sync if T is Sync and S is Sync
unsafe impl<T: Sync, S: Sync> Sync for PmemHashList<T, S> {}
