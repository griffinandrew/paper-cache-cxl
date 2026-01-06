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

    /// A hash-backed linked list with PMem storage via Hybrid allocator
    /// 
    /// This implementation uses hashbrown::HashMap with the Hybrid allocator
    /// to store data in PMem. The ordering is maintained using a Vec allocated
    /// in PMem as well (using Vec::new_in with the Hybrid allocator).
    /// 
    /// Note: This is less efficient than a true linked list for move operations
    /// (O(n) instead of O(1)), but it ensures PMem storage for eviction metadata.
    pub struct HashList<K, S> {
        map: HashMap<K, usize, S, Hybrid>,  // Key -> index in order vec
        order: Vec<K, Hybrid>,  // Ordered list of keys (front to back), allocated in PMem
    }

    impl<K, S> HashList<K, S>
    where
        K: Hash + Eq + Clone,
        S: BuildHasher + Default,
    {
        /// Create a new HashList with the given hasher
        pub fn with_hasher(hasher: S) -> Self {
            HashList {
                map: HashMap::with_hasher_in(hasher, Hybrid),
                order: Vec::new_in(Hybrid),
            }
        }

        /// Returns the number of elements
        pub fn len(&self) -> usize {
            self.order.len()
        }

        /// Returns true if empty
        pub fn is_empty(&self) -> bool {
            self.order.is_empty()
        }

        /// Get a reference to the last (back) element
        pub fn back(&self) -> Option<&K> {
            self.order.last()
        }

        /// Get a reference to the element before the given key
        pub fn before<Q>(&self, key: &Q) -> Option<&K>
        where
            K: Borrow<Q>,
            Q: Hash + Eq + ?Sized,
        {
            let index = *self.map.get(key)?;
            if index + 1 < self.order.len() {
                Some(&self.order[index + 1])
            } else {
                None
            }
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
            let index = *self.map.get(key)?;
            self.order.get(index)
        }

        /// Push an element to the front
        pub fn push_front(&mut self, key: K) {
            if self.map.contains_key(&key) {
                return; // Don't add duplicates
            }
            self.order.insert(0, key.clone());
            self.rebuild_index();
        }

        /// Remove and return the element from the front
        pub fn pop_front(&mut self) -> Option<K> {
            if self.order.is_empty() {
                return None;
            }
            let key = self.order.remove(0);
            self.map.remove(&key);
            self.rebuild_index();
            Some(key)
        }

        /// Remove and return the element from the back
        pub fn pop_back(&mut self) -> Option<K> {
            let key = self.order.pop()?;
            self.map.remove(&key);
            self.rebuild_index();
            Some(key)
        }

        /// Move an existing element to the front
        pub fn move_front<Q>(&mut self, key: &Q)
        where
            K: Borrow<Q>,
            Q: Hash + Eq + ?Sized,
        {
            if let Some(&index) = self.map.get(key) {
                if index == 0 {
                    return; // Already at front
                }
                let removed = self.order.remove(index);
                self.order.insert(0, removed);
                self.rebuild_index();
            }
        }

        /// Remove a specific element
        pub fn remove<Q>(&mut self, key: &Q) -> Option<K>
        where
            K: Borrow<Q>,
            Q: Hash + Eq + ?Sized,
        {
            let index = *self.map.get(key)?;
            self.map.remove(key);
            let removed = self.order.remove(index);
            self.rebuild_index();
            Some(removed)
        }

        /// Clear all elements
        pub fn clear(&mut self) {
            self.map.clear();
            self.order.clear();
        }

        /// Update an element using a closure
        pub fn update<Q, F>(&mut self, key: &Q, f: F)
        where
            K: Borrow<Q>,
            Q: Hash + Eq + ?Sized,
            F: FnOnce(&mut K),
        {
            if let Some(&index) = self.map.get(key) {
                if let Some(item) = self.order.get_mut(index) {
                    f(item);
                }
            }
        }

        // Rebuild the index map after order changes
        fn rebuild_index(&mut self) {
            self.map.clear();
            for (idx, key) in self.order.iter().enumerate() {
                self.map.insert(key.clone(), idx);
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
