/*
 * PMem-backed HashList implementation
 * 
 * This module provides a hash-backed linked list that uses PMem allocation
 * when the pmem_eviction_stacks feature is enabled. Falls back to kwik::collections::HashList otherwise.
 */

// When pmem_eviction_stacks feature is NOT enabled, just re-export the standard HashList
#[cfg(not(feature = "pmem_eviction_stacks"))]
pub use kwik::collections::HashList;

// When pmem_eviction_stacks feature IS enabled, provide a PMem-backed implementation
#[cfg(feature = "pmem_eviction_stacks")]
mod pmem_impl {
    use std::hash::{Hash, BuildHasher};
    use std::borrow::Borrow;
    use hashbrown::HashMap;
    use crate::allocator::HybridObjects as Hybrid;

    /// A hash-backed doubly-linked list with PMem storage via Hybrid allocator
    /// 
    /// This implementation uses hashbrown::HashMap with the Hybrid allocator
    /// to store data in PMem. Instead of using a Vec and rebuilding indices,
    /// we maintain a proper doubly-linked list structure with explicit prev/next pointers.
    /// This avoids O(n) rebuild_index() operations after every modification.
    /// 
    /// The list maintains proper links during all operations to prevent index corruption
    /// and uses bounds checking to prevent segfaults.
    pub struct HashList<K, S> {
        map: HashMap<K, NodeIndex, S, Hybrid>,  // Key -> node index
        nodes: Vec<Node<K>, Hybrid>,  // All nodes, allocated in PMem
        head: Option<NodeIndex>,  // Index of first node
        tail: Option<NodeIndex>,  // Index of last node
        free_list: Vec<NodeIndex, Hybrid>,  // Indices of deleted nodes for reuse
    }

    type NodeIndex = usize;

    #[derive(Clone)]
    struct Node<K> {
        key: K,
        prev: Option<NodeIndex>,
        next: Option<NodeIndex>,
        active: bool,  // false if node has been deleted (in free_list)
    }

    impl<K, S> HashList<K, S>
    where
        K: Hash + Eq + Clone,
        S: BuildHasher + Default,
    {
        /// Create a new HashList with the given hasher
        pub fn with_hasher(hasher: S) -> Self {
            println!("Creating PMem-backed HashList with Hybrid allocator (doubly-linked list)");
            HashList {
                map: HashMap::with_hasher_in(hasher, Hybrid),
                nodes: Vec::new_in(Hybrid),
                head: None,
                tail: None,
                free_list: Vec::new_in(Hybrid),
            }
        }

        /// Returns the number of active elements
        pub fn len(&self) -> usize {
            self.map.len()
        }

        /// Returns true if empty
        pub fn is_empty(&self) -> bool {
            self.map.is_empty()
        }

        /// Get a reference to the last (back) element
        pub fn back(&self) -> Option<&K> {
            let tail_idx = self.tail?;
            self.nodes.get(tail_idx).map(|node| &node.key)
        }

        /// Get a reference to the element before the given key
        pub fn before<Q>(&self, key: &Q) -> Option<&K>
        where
            K: Borrow<Q>,
            Q: Hash + Eq + ?Sized,
        {
            let node_idx = *self.map.get(key)?;
            let node = self.nodes.get(node_idx)?;
            
            // Safety: Check that node is active before accessing
            if !node.active {
                return None;
            }
            
            let next_idx = node.next?;
            self.nodes.get(next_idx).map(|n| &n.key)
        }

        /// Check if a key exists
        pub fn contains<Q>(&self, key: &Q) -> bool
        where
            K: Borrow<Q>,
            Q: Hash + Eq + ?Sized,
        {
            self.map.contains_key(key)
        }

        /// Get a reference to an element by key
        pub fn get<Q>(&self, key: &Q) -> Option<&K>
        where
            K: Borrow<Q>,
            Q: Hash + Eq + ?Sized,
        {
            let node_idx = *self.map.get(key)?;
            let node = self.nodes.get(node_idx)?;
            
            // Safety: Only return if node is active
            if node.active {
                Some(&node.key)
            } else {
                None
            }
        }

        /// Push an element to the front
        pub fn push_front(&mut self, key: K) {
            // Don't add duplicates
            if self.map.contains_key(&key) {
                return;
            }

            // Get or create a node index
            let node_idx = if let Some(idx) = self.free_list.pop() {
                // Reuse deleted node slot - update its fields
                let node = &mut self.nodes[idx];
                node.key = key.clone();
                node.prev = None;
                node.next = self.head;
                node.active = true;
                idx
            } else {
                // Allocate new node
                let idx = self.nodes.len();
                // Pre-allocate to avoid allocation failures mid-operation
                if self.nodes.capacity() <= idx {
                    self.nodes.reserve(8); // Reserve in small chunks
                }
                self.nodes.push(Node {
                    key: key.clone(),
                    prev: None,
                    next: self.head,
                    active: true,
                });
                idx
            };

            // Update the previous head's prev pointer
            if let Some(old_head_idx) = self.head {
                if let Some(old_head) = self.nodes.get_mut(old_head_idx) {
                    old_head.prev = Some(node_idx);
                }
            }

            // Update head pointer
            self.head = Some(node_idx);

            // If list was empty, also update tail
            if self.tail.is_none() {
                self.tail = Some(node_idx);
            }

            // Add to map
            self.map.insert(key, node_idx);
        }

        /// Remove and return the element from the front
        pub fn pop_front(&mut self) -> Option<K> {
            let head_idx = self.head?;
            let head_node = self.nodes.get(head_idx)?;
            
            if !head_node.active {
                return None;
            }

            let key = head_node.key.clone();
            let next = head_node.next;

            // Mark node as inactive and add to free list
            self.nodes[head_idx].active = false;
            self.free_list.push(head_idx);

            // Update head
            self.head = next;

            // Update new head's prev pointer
            if let Some(new_head_idx) = next {
                if let Some(new_head) = self.nodes.get_mut(new_head_idx) {
                    new_head.prev = None;
                }
            } else {
                // List is now empty
                self.tail = None;
            }

            // Remove from map
            self.map.remove(&key);

            Some(key)
        }

        /// Remove and return the element from the back
        pub fn pop_back(&mut self) -> Option<K> {
            let tail_idx = self.tail?;
            let tail_node = self.nodes.get(tail_idx)?;
            
            if !tail_node.active {
                return None;
            }

            let key = tail_node.key.clone();
            let prev = tail_node.prev;

            // Mark node as inactive and add to free list
            self.nodes[tail_idx].active = false;
            self.free_list.push(tail_idx);

            // Update tail
            self.tail = prev;

            // Update new tail's next pointer
            if let Some(new_tail_idx) = prev {
                if let Some(new_tail) = self.nodes.get_mut(new_tail_idx) {
                    new_tail.next = None;
                }
            } else {
                // List is now empty
                self.head = None;
            }

            // Remove from map
            self.map.remove(&key);

            Some(key)
        }

        /// Move an existing element to the front
        pub fn move_front<Q>(&mut self, key: &Q)
        where
            K: Borrow<Q>,
            Q: Hash + Eq + ?Sized,
        {
            let node_idx = match self.map.get(key) {
                Some(&idx) => idx,
                None => return,
            };

            // If already at front, nothing to do
            if Some(node_idx) == self.head {
                return;
            }

            // Safety check: ensure node is active
            if !self.nodes.get(node_idx).map_or(false, |n| n.active) {
                return;
            }

            // Unlink from current position
            let (prev, next) = {
                let node = &self.nodes[node_idx];
                (node.prev, node.next)
            };

            // Update prev node's next pointer
            if let Some(prev_idx) = prev {
                if let Some(prev_node) = self.nodes.get_mut(prev_idx) {
                    prev_node.next = next;
                }
            }

            // Update next node's prev pointer
            if let Some(next_idx) = next {
                if let Some(next_node) = self.nodes.get_mut(next_idx) {
                    next_node.prev = prev;
                }
            }

            // If this was the tail, update tail pointer
            if Some(node_idx) == self.tail {
                self.tail = prev;
            }

            // Move to front
            if let Some(old_head_idx) = self.head {
                if let Some(old_head) = self.nodes.get_mut(old_head_idx) {
                    old_head.prev = Some(node_idx);
                }
            }

            // Update the moved node
            let node = &mut self.nodes[node_idx];
            node.prev = None;
            node.next = self.head;

            // Update head
            self.head = Some(node_idx);
        }

        /// Remove a specific element
        pub fn remove<Q>(&mut self, key: &Q) -> Option<K>
        where
            K: Borrow<Q>,
            Q: Hash + Eq + ?Sized,
        {
            let node_idx = *self.map.get(key)?;
            
            // Safety check: ensure node is active
            if !self.nodes.get(node_idx).map_or(false, |n| n.active) {
                return None;
            }

            let node = &self.nodes[node_idx];
            let key_clone = node.key.clone();
            let prev = node.prev;
            let next = node.next;

            // Mark node as inactive
            self.nodes[node_idx].active = false;
            self.free_list.push(node_idx);

            // Update prev node's next pointer
            if let Some(prev_idx) = prev {
                if let Some(prev_node) = self.nodes.get_mut(prev_idx) {
                    prev_node.next = next;
                }
            } else {
                // Removing head
                self.head = next;
            }

            // Update next node's prev pointer
            if let Some(next_idx) = next {
                if let Some(next_node) = self.nodes.get_mut(next_idx) {
                    next_node.prev = prev;
                }
            } else {
                // Removing tail
                self.tail = prev;
            }

            // Remove from map
            self.map.remove(key);

            Some(key_clone)
        }

        /// Clear all elements
        pub fn clear(&mut self) {
            self.map.clear();
            self.nodes.clear();
            self.free_list.clear();
            self.head = None;
            self.tail = None;
        }

        /// Update an element using a closure
        pub fn update<Q, F>(&mut self, key: &Q, f: F)
        where
            K: Borrow<Q>,
            Q: Hash + Eq + ?Sized,
            F: FnOnce(&mut K),
        {
            if let Some(&node_idx) = self.map.get(key) {
                if let Some(node) = self.nodes.get_mut(node_idx) {
                    if node.active {
                        f(&mut node.key);
                    }
                }
            }
        }
    }

    // Implement Default for HashList when S implements Default
    impl<K, S> Default for HashList<K, S>
    where
        K: Hash + Eq + Clone,
        S: BuildHasher + Default,
    {
        fn default() -> Self {
            Self::with_hasher(S::default())
        }
    }
}

#[cfg(feature = "pmem_eviction_stacks")]
pub use pmem_impl::HashList;
