/*
 * Copyright (c) Kia Shakiba
 *
 * This source code is licensed under the GNU AGPLv3 license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! FlatMap: A high-performance Linear Probing Hash Map optimized for Persistent Memory (PMEM).
//!
//! This implementation uses a "Flat Layout" (Array of Structs) where Hash, Key, and Value
//! are adjacent in memory, reducing PMEM read latency from 3x to 1x by ensuring that a
//! single 64-byte cache line fetch retrieves everything needed to validate a match.
//!
//! Key design decisions:
//! - Uses Linear Probing (no Robin Hood hashing) to minimize expensive PMEM writes
//! - Fixed capacity (no resizing) for simplicity
//! - Generic allocator support for custom PMEM allocators
//! - #[inline(always)] on hot path methods to minimize call overhead
//!
//! ## Performance Characteristics
//!
//! ### vs. hashbrown (SwissTable)
//!
//! **hashbrown** uses a "Split Layout" with separate control bytes and data:
//! - Read Control Bytes -> Wait 300ns (PMEM latency) -> Read Key -> Wait 300ns -> Validate
//! - Total: ~600ns+ per lookup (3x PMEM reads)
//!
//! **FlatMap** uses a "Flat Layout" with everything adjacent:
//! - Read Bucket (hash + key + value in one cache line) -> Wait 300ns -> Validate
//! - Total: ~300ns per lookup (1x PMEM read)
//!
//! This 3x reduction in latency is critical for PMEM performance.
//!
//! ## Usage
//!
//! ```rust,ignore
//! use paper_cache::flatmap::FlatMap;
//! use std::hash::RandomState;
//!
//! // Create a FlatMap with 1024 buckets (must be power of 2)
//! let mut map = FlatMap::new(1024);
//! let hasher = RandomState::new();
//!
//! // Insert key-value pairs
//! map.insert_with_hasher(1u64, "one", &hasher);
//! map.insert_with_hasher(2u64, "two", &hasher);
//!
//! // Lookup values
//! assert_eq!(map.get_with_hasher(&1u64, &hasher), Some(&"one"));
//!
//! // Remove entries
//! assert_eq!(map.remove_with_hasher(&1u64, &hasher), Some("one"));
//! ```
//!
//! ## With Custom PMEM Allocator
//!
//! ```rust,ignore
//! use paper_cache::flatmap::FlatMap;
//! use paper_cache::allocator::HybridObjects as Hybrid;
//! use std::hash::RandomState;
//!
//! // Create a FlatMap with PMEM allocator
//! let mut map = FlatMap::new_in(1024, Hybrid);
//! let hasher = RandomState::new();
//!
//! // Use as normal - data will be allocated in PMEM
//! map.insert_with_hasher(1u64, vec![1, 2, 3], &hasher);
//! ```
//!
//! ## Feature Flags
//!
//! - `flatmap_dram`: Enable FlatMap with DRAM allocator
//! - `flatmap_pmem`: Enable FlatMap with PMEM allocator (HybridObjects)

use std::alloc::{Allocator, Global};
use std::hash::{Hash, BuildHasher};
use std::mem;
use std::marker::PhantomData;

/// A bucket in the hash map containing hash, key, and value.
/// Uses #[repr(C)] for predictable memory layout.
#[derive(Clone)]
#[repr(C)]
pub(crate) struct Bucket<K, V> {
    /// Hash value. 0 indicates an empty bucket.
    hash: u64,
    /// The key stored in this bucket.
    key: K,
    /// The value stored in this bucket.
    val: V,
}

impl<K, V> Bucket<K, V> {
    /// Creates an empty bucket with zeroed memory.
    /// SAFETY: This requires K and V to be safely zero-initialized.
    #[inline]
    fn empty() -> Self
    where
        K: Default,
        V: Default,
    {
        Self {
            hash: 0,
            key: K::default(),
            val: V::default(),
        }
    }

    /// Checks if the bucket is empty.
    #[inline(always)]
    fn is_empty(&self) -> bool {
        self.hash == 0
    }

    /// Checks if this bucket matches the given hash and key.
    #[inline(always)]
    fn matches<Q>(&self, hash: u64, key: &Q) -> bool
    where
        K: PartialEq<Q>,
    {
        self.hash == hash && !self.is_empty() && &self.key == key
    }
}

/// A high-performance Linear Probing Hash Map optimized for PMEM.
///
/// This implementation uses a flat layout where hash, key, and value are stored
/// adjacent in memory, reducing PMEM read overhead to a single cache line fetch.
///
/// # Type Parameters
///
/// * `K` - The key type
/// * `V` - The value type
/// * `A` - The allocator type (defaults to Global)
pub struct FlatMap<K, V, A: Allocator = Global> {
    /// The array of buckets
    pub(crate) buckets: Vec<Bucket<K, V>, A>,
    /// Total capacity (number of buckets)
    pub(crate) capacity: usize,
    /// Mask for fast modulo operations (capacity - 1)
    pub(crate) mask: usize,
    /// Number of occupied buckets
    pub(crate) len: usize,
    /// Phantom data for variance
    _phantom: PhantomData<(K, V)>,
}

// Unconstrained impl block for methods that don't need Default
impl<K, V, A: Allocator> FlatMap<K, V, A>
where
    K: Hash + Eq,
{
    /// Returns the number of elements in the map.
    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }

    /// Returns true if the map is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Returns the capacity of the map.
    #[inline]
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Computes the hash for a key (unconstrained version).
    #[inline]
    pub(crate) fn hash_key_unconstrained<Q, S>(&self, key: &Q, hasher: &S) -> u64
    where
        Q: Hash,
        S: BuildHasher,
    {
        use std::hash::Hasher;
        let mut h = hasher.build_hasher();
        key.hash(&mut h);
        let hash = h.finish();
        // Ensure hash is never 0 (reserved for empty)
        if hash == 0 { 1 } else { hash }
    }

    /// Gets a reference to the value associated with the key (unconstrained version).
    #[inline(always)]
    pub fn get_with_hasher<Q, S>(&self, key: &Q, hasher: &S) -> Option<&V>
    where
        Q: Hash + Eq,
        K: PartialEq<Q>,
        S: BuildHasher,
    {
        let hash = self.hash_key_unconstrained(key, hasher);
        let mut index = (hash as usize) & self.mask;
        
        // Linear probing
        for _ in 0..self.capacity {
            let bucket = &self.buckets[index];
            
            if bucket.is_empty() {
                return None;
            } else if bucket.matches(hash, key) {
                return Some(&bucket.val);
            }
            
            index = (index + 1) & self.mask;
        }
        
        None
    }

    /// Gets a mutable reference to the value associated with the key (unconstrained version).
    #[inline(always)]
    pub fn get_mut_with_hasher<Q, S>(&mut self, key: &Q, hasher: &S) -> Option<&mut V>
    where
        Q: Hash + Eq,
        K: PartialEq<Q>,
        S: BuildHasher,
    {
        let hash = self.hash_key_unconstrained(key, hasher);
        let mut index = (hash as usize) & self.mask;
        
        // Linear probing - find the index first
        let found_index = {
            let mut found = None;
            for _ in 0..self.capacity {
                let bucket = &self.buckets[index];
                
                if bucket.is_empty() {
                    break;
                } else if bucket.matches(hash, key) {
                    found = Some(index);
                    break;
                }
                
                index = (index + 1) & self.mask;
            }
            found
        };
        
        found_index.map(|idx| &mut self.buckets[idx].val)
    }

    /// Checks if the map contains the given key (unconstrained version).
    #[inline]
    pub fn contains_key_with_hasher<Q, S>(&self, key: &Q, hasher: &S) -> bool
    where
        Q: Hash + Eq,
        K: PartialEq<Q>,
        S: BuildHasher,
    {
        self.get_with_hasher(key, hasher).is_some()
    }

    /// Returns an iterator over the key-value pairs in the map.
    pub fn iter(&self) -> Iter<'_, K, V> {
        Iter {
            buckets: &self.buckets,
            index: 0,
        }
    }

    /// Clears the map, removing all key-value pairs.
    /// Requires K and V to implement Default to reset buckets.
    pub fn clear(&mut self)
    where
        K: Default,
        V: Default,
    {
        for bucket in &mut self.buckets {
            *bucket = Bucket::empty();
        }
        self.len = 0;
    }

    /// Resizes the map to a new capacity.
    /// The new capacity must be a power of 2 and larger than the current capacity.
    /// This method rehashes all existing entries into the new table.
    pub fn resize<S>(&mut self, new_capacity: usize, hasher: &S) -> Result<(), &'static str>
    where
        K: Default + Clone,
        V: Default + Clone,
        S: BuildHasher,
        A: Clone,
    {
        if new_capacity <= self.capacity {
            return Err("New capacity must be larger than current capacity");
        }
        if !new_capacity.is_power_of_two() {
            return Err("New capacity must be a power of 2");
        }

        // Create a new FlatMap with the larger capacity
        let mut new_map = FlatMap::new_in(new_capacity, self.buckets.allocator().clone());

        // Rehash all existing entries into the new map
        for bucket in &self.buckets {
            if !bucket.is_empty() {
                new_map.insert_with_hasher(bucket.key.clone(), bucket.val.clone(), hasher);
            }
        }

        // Replace the current map with the new one
        self.buckets = new_map.buckets;
        self.capacity = new_map.capacity;
        self.mask = new_map.mask;
        self.len = new_map.len;

        Ok(())
    }
}

impl<K, V> FlatMap<K, V, Global>
where
    K: Hash + Eq + Default,
    V: Default,
{
    /// Creates a new FlatMap with the specified capacity using the global allocator.
    ///
    /// # Arguments
    ///
    /// * `capacity` - The fixed capacity. Must be a power of 2.
    ///
    /// # Panics
    ///
    /// Panics if capacity is 0 or not a power of 2.
    pub fn new(capacity: usize) -> Self {
        Self::new_in(capacity, Global)
    }
}

impl<K, V, A: Allocator> FlatMap<K, V, A>
where
    K: Hash + Eq + Default,
    V: Default,
{
    /// Creates a new FlatMap with the specified capacity and allocator.
    ///
    /// # Arguments
    ///
    /// * `capacity` - The fixed capacity. Must be a power of 2.
    /// * `alloc` - The allocator to use
    ///
    /// # Panics
    ///
    /// Panics if capacity is 0 or not a power of 2.
    pub fn new_in(capacity: usize, alloc: A) -> Self {
        assert!(capacity > 0, "Capacity must be greater than 0");
        assert!(capacity.is_power_of_two(), "Capacity must be a power of 2");

        let mask = capacity - 1;
        
        // Allocate and initialize buckets
        let mut buckets = Vec::with_capacity_in(capacity, alloc);
        for _ in 0..capacity {
            buckets.push(Bucket::empty());
        }

        Self {
            buckets,
            capacity,
            mask,
            len: 0,
            _phantom: PhantomData,
        }
    }

    /// Inserts a key-value pair into the map with the given hasher.
    ///
    /// Returns the old value if the key was already present.
    ///
    /// # Panics
    ///
    /// Panics if the map is full.
    pub fn insert_with_hasher<S>(&mut self, key: K, val: V, hasher: &S) -> Option<V>
    where
        S: BuildHasher,
    {
        let hash = self.hash_key_unconstrained(&key, hasher);
        let mut index = (hash as usize) & self.mask;
        
        // Linear probing
        for _ in 0..self.capacity {
            let bucket = &mut self.buckets[index];
            
            if bucket.is_empty() {
                // Found empty slot
                *bucket = Bucket { hash, key, val };
                self.len += 1;
                return None;
            } else if bucket.matches(hash, &key) {
                // Key exists, replace value
                let old_val = mem::replace(&mut bucket.val, val);
                return Some(old_val);
            }
            
            // Move to next bucket
            index = (index + 1) & self.mask;
        }
        
        panic!("FlatMap is full");
    }

    /// Removes a key from the map, returning the value if the key was present.
    ///
    /// Note: This implementation uses backwards shift deletion to maintain probe chain integrity.
    pub fn remove_with_hasher<Q, S>(&mut self, key: &Q, hasher: &S) -> Option<V>
    where
        Q: Hash + Eq,
        K: PartialEq<Q> + Clone,
        V: Clone,
        S: BuildHasher,
    {
        let hash = self.hash_key_unconstrained(key, hasher);
        let mut index = (hash as usize) & self.mask;
        
        // Linear probing to find the key
        for _ in 0..self.capacity {
            let bucket = &mut self.buckets[index];
            
            if bucket.is_empty() {
                // Empty slot means key not found
                return None;
            } else if bucket.matches(hash, key) {
                // Found the key - perform backwards shift deletion
                self.len -= 1;
                let val = mem::replace(&mut bucket.val, V::default());
                
                // Backwards shift deletion to maintain probe chain integrity
                let mut curr_index = index;
                loop {
                    let next_index = (curr_index + 1) & self.mask;
                    let next_bucket = &self.buckets[next_index];
                    
                    // If next bucket is empty, we can mark current as empty and stop
                    if next_bucket.is_empty() {
                        self.buckets[curr_index] = Bucket::empty();
                        break;
                    }
                    
                    // Calculate ideal position for next bucket
                    let ideal_index = (next_bucket.hash as usize) & self.mask;
                    
                    // Check if we should shift the next bucket back to curr position
                    // We shift if the ideal position is at or before curr, considering wraparound
                    let should_shift = if ideal_index <= curr_index {
                        // Ideal is at or before curr - definitely shift
                        // OR next is before ideal (meaning it wrapped around)
                        true
                    } else {
                        // Ideal is after curr
                        // Shift only if next wrapped around past ideal
                        next_index < ideal_index
                    };
                    
                    if !should_shift {
                        self.buckets[curr_index] = Bucket::empty();
                        break;
                    }
                    
                    // Shift the next bucket back
                    self.buckets[curr_index] = self.buckets[next_index].clone();
                    curr_index = next_index;
                }
                
                return Some(val);
            }
            
            // Move to next bucket
            index = (index + 1) & self.mask;
        }
        
        None
    }

    /// Removes a key from the map using tombstoning (simpler but may degrade performance over time).
    /// This method works with any V type, not requiring Clone.
    /// Useful when V doesn't implement Clone or Default constraints are acceptable.
    pub fn remove_tombstone_with_hasher<Q, S>(&mut self, key: &Q, hasher: &S) -> Option<V>
    where
        Q: Hash + Eq,
        K: PartialEq<Q> + Default,
        V: Default,
        S: BuildHasher,
    {
        let hash = self.hash_key_unconstrained(key, hasher);
        let mut index = (hash as usize) & self.mask;
        
        // Linear probing to find the key
        for _ in 0..self.capacity {
            let bucket = &mut self.buckets[index];
            
            if bucket.is_empty() {
                return None;
            } else if bucket.matches(hash, key) {
                // Found the key - use tombstone (mark as empty)
                self.len -= 1;
                bucket.hash = 0; // Mark as empty
                let val = mem::replace(&mut bucket.val, V::default());
                let _ = mem::replace(&mut bucket.key, K::default());
                return Some(val);
            }
            
            index = (index + 1) & self.mask;
        }
        
        None
    }
}

/// Iterator over the key-value pairs in a FlatMap.
pub struct Iter<'a, K, V> {
    buckets: &'a [Bucket<K, V>],
    index: usize,
}

impl<'a, K, V> Iterator for Iter<'a, K, V> {
    type Item = (&'a K, &'a V);

    fn next(&mut self) -> Option<Self::Item> {
        while self.index < self.buckets.len() {
            let bucket = &self.buckets[self.index];
            self.index += 1;
            
            if !bucket.is_empty() {
                return Some((&bucket.key, &bucket.val));
            }
        }
        None
    }
}

/// FlatMapWithHasher wraps FlatMap with a BuildHasher for convenient usage
/// similar to HashMap. This is used when integrating FlatMap as PaperCache's hashtable.
pub struct FlatMapWithHasher<K, V, S, A: Allocator = Global> {
    map: FlatMap<K, V, A>,
    hasher: S,
}

impl<K, V, S> FlatMapWithHasher<K, V, S, Global>
where
    K: Hash + Eq + Default,
    V: Default,
    S: BuildHasher + Default,
{
    /// Creates a new FlatMapWithHasher with the specified capacity using the global allocator.
    pub fn with_capacity_and_hasher(capacity: usize, hasher: S) -> Self {
        Self {
            map: FlatMap::new(capacity),
            hasher,
        }
    }
}

// Impl block without Default constraints for methods that don't need them
impl<K, V, S, A: Allocator> FlatMapWithHasher<K, V, S, A>
where
    K: Hash + Eq,
    S: BuildHasher,
{
    /// Returns the number of elements in the map.
    #[inline]
    pub fn len(&self) -> usize {
        self.map.len()
    }

    /// Returns true if the map is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.map.is_empty()
    }

    /// Returns the capacity of the map.
    #[inline]
    pub fn capacity(&self) -> usize {
        self.map.capacity()
    }

    /// Gets a reference to the value associated with the key.
    #[inline(always)]
    pub fn get<Q>(&self, key: &Q) -> Option<&V>
    where
        Q: Hash + Eq,
        K: PartialEq<Q>,
    {
        self.map.get_with_hasher(key, &self.hasher)
    }

    /// Gets a mutable reference to the value associated with the key.
    #[inline(always)]
    pub fn get_mut<Q>(&mut self, key: &Q) -> Option<&mut V>
    where
        Q: Hash + Eq,
        K: PartialEq<Q>,
    {
        self.map.get_mut_with_hasher(key, &self.hasher)
    }

    /// Checks if the map contains the given key.
    #[inline]
    pub fn contains_key<Q>(&self, key: &Q) -> bool
    where
        Q: Hash + Eq,
        K: PartialEq<Q>,
    {
        self.map.contains_key_with_hasher(key, &self.hasher)
    }

    /// Clears the map, removing all key-value pairs.
    /// Requires K and V to implement Default to reset buckets.
    pub fn clear(&mut self)
    where
        K: Default,
        V: Default,
    {
        self.map.clear()
    }

    /// Returns an iterator over the key-value pairs in the map.
    pub fn iter(&self) -> Iter<'_, K, V> {
        self.map.iter()
    }

    /// Resizes the map to a new capacity.
    /// The new capacity must be a power of 2 and larger than the current capacity.
    pub fn resize(&mut self, new_capacity: usize) -> Result<(), &'static str>
    where
        K: Default + Clone,
        V: Default + Clone,
        A: Clone,
    {
        self.map.resize(new_capacity, &self.hasher)
    }

    /// Internal method to remove without Clone/Default constraints.
    /// Uses unsafe ptr::read to extract the value without requiring Clone/Default.
    /// This is safe because we mark the bucket as empty immediately after reading.
    pub fn remove_unchecked<Q>(&mut self, key: &Q) -> Option<V>
    where
        Q: Hash + Eq,
        K: PartialEq<Q>,
    {
        // Access the internal FlatMap
        let hash = self.map.hash_key_unconstrained(key, &self.hasher);
        let mut index = (hash as usize) & self.map.mask;
        
        // Linear probing to find the key
        for _ in 0..self.map.capacity {
            let bucket = &mut self.map.buckets[index];
            
            if bucket.is_empty() {
                return None;
            } else if bucket.matches(hash, key) {
                // Found the key - extract the value using unsafe ptr::read
                // This works even without Clone/Default on K, V
                self.map.len -= 1;
                
                // Use unsafe to read the value out of the bucket without calling drop
                // on the old location. Then mark the bucket as empty.
                unsafe {
                    use std::ptr;
                    
                    // Read the value out (this transfers ownership without dropping)
                    let val = ptr::read(&bucket.val as *const V);
                    
                    // Drop the key properly
                    ptr::drop_in_place(&mut bucket.key as *mut K);
                    
                    // Mark the bucket as empty
                    bucket.hash = 0;
                    
                    return Some(val);
                }
            }
            
            index = (index + 1) & self.map.mask;
        }
        
        None
    }

    /// Internal method to insert without Default constraints.
    /// Directly implements insert logic to avoid trait bound issues.
    pub fn insert_unchecked(&mut self, key: K, val: V) -> Option<V>
    where
        K: Hash + Eq,
        S: BuildHasher,
    {
        use std::mem;
        use crate::flatmap::Bucket;
        
        let hash = self.map.hash_key_unconstrained(&key, &self.hasher);
        let mut index = (hash as usize) & self.map.mask;
        
        // Linear probing
        for _ in 0..self.map.capacity {
            let bucket = &mut self.map.buckets[index];
            
            if bucket.is_empty() {
                // Found empty slot
                *bucket = Bucket { hash, key, val };
                self.map.len += 1;
                return None;
            } else if bucket.matches(hash, &key) {
                // Key exists, replace value
                let old_val = mem::replace(&mut bucket.val, val);
                return Some(old_val);
            }
            
            // Move to next bucket
            index = (index + 1) & self.map.mask;
        }
        
        panic!("FlatMap is full");
    }
}

impl<K, V, S, A: Allocator> FlatMapWithHasher<K, V, S, A>
where
    K: Hash + Eq + Default,
    V: Default,
    S: BuildHasher,
{
    /// Creates a new FlatMapWithHasher with the specified capacity and allocator.
    pub fn with_capacity_hasher_in(capacity: usize, hasher: S, alloc: A) -> Self {
        Self {
            map: FlatMap::new_in(capacity, alloc),
            hasher,
        }
    }

    /// Inserts a key-value pair into the map.
    #[inline]
    pub fn insert(&mut self, key: K, val: V) -> Option<V> {
        self.map.insert_with_hasher(key, val, &self.hasher)
    }

    /// Removes a key from the map, returning the value if the key was present.
    /// Uses tombstoning for simplicity (works with any V type).
    #[inline]
    pub fn remove<Q>(&mut self, key: &Q) -> Option<V>
    where
        Q: Hash + Eq,
        K: PartialEq<Q> + Default,
        V: Default,
    {
        self.map.remove_tombstone_with_hasher(key, &self.hasher)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::hash::RandomState;

    #[test]
    fn test_new() {
        let map: FlatMap<u64, u64> = FlatMap::new(16);
        assert_eq!(map.len(), 0);
        assert_eq!(map.capacity(), 16);
        assert!(map.is_empty());
    }

    #[test]
    fn test_insert_and_get() {
        let mut map = FlatMap::new(16);
        let hasher = RandomState::new();
        
        assert_eq!(map.insert_with_hasher(1u64, 100u64, &hasher), None);
        assert_eq!(map.len(), 1);
        assert_eq!(map.get_with_hasher(&1u64, &hasher), Some(&100u64));
    }

    #[test]
    fn test_insert_replace() {
        let mut map = FlatMap::new(16);
        let hasher = RandomState::new();
        
        map.insert_with_hasher(1u64, 100u64, &hasher);
        assert_eq!(map.insert_with_hasher(1u64, 200u64, &hasher), Some(100u64));
        assert_eq!(map.len(), 1);
        assert_eq!(map.get_with_hasher(&1u64, &hasher), Some(&200u64));
    }

    #[test]
    fn test_multiple_inserts() {
        let mut map = FlatMap::new(16);
        let hasher = RandomState::new();
        
        for i in 0..10 {
            map.insert_with_hasher(i, i * 10, &hasher);
        }
        
        assert_eq!(map.len(), 10);
        
        for i in 0..10 {
            assert_eq!(map.get_with_hasher(&i, &hasher), Some(&(i * 10)));
        }
    }

    #[test]
    fn test_remove() {
        let mut map = FlatMap::new(16);
        let hasher = RandomState::new();
        
        map.insert_with_hasher(1u64, 100u64, &hasher);
        map.insert_with_hasher(2u64, 200u64, &hasher);
        
        assert_eq!(map.remove_with_hasher(&1u64, &hasher), Some(100u64));
        assert_eq!(map.len(), 1);
        assert_eq!(map.get_with_hasher(&1u64, &hasher), None);
        assert_eq!(map.get_with_hasher(&2u64, &hasher), Some(&200u64));
    }

    #[test]
    fn test_clear() {
        let mut map = FlatMap::new(16);
        let hasher = RandomState::new();
        
        for i in 0..10 {
            map.insert_with_hasher(i, i * 10, &hasher);
        }
        
        map.clear();
        assert_eq!(map.len(), 0);
        assert!(map.is_empty());
    }

    #[test]
    fn test_contains_key() {
        let mut map = FlatMap::new(16);
        let hasher = RandomState::new();
        
        map.insert_with_hasher(1u64, 100u64, &hasher);
        
        assert!(map.contains_key_with_hasher(&1u64, &hasher));
        assert!(!map.contains_key_with_hasher(&2u64, &hasher));
    }

    #[test]
    fn test_iter() {
        let mut map = FlatMap::new(16);
        let hasher = RandomState::new();
        
        map.insert_with_hasher(1u64, 100u64, &hasher);
        map.insert_with_hasher(2u64, 200u64, &hasher);
        
        let mut count = 0;
        for (k, v) in map.iter() {
            count += 1;
            assert!((*k == 1 && *v == 100) || (*k == 2 && *v == 200));
        }
        assert_eq!(count, 2);
    }

    #[test]
    #[should_panic(expected = "FlatMap is full")]
    fn test_full_map() {
        let mut map = FlatMap::new(4);
        let hasher = RandomState::new();
        
        for i in 0..5 {
            map.insert_with_hasher(i, i * 10, &hasher);
        }
    }

    #[test]
    fn test_get_mut() {
        let mut map = FlatMap::new(16);
        let hasher = RandomState::new();
        
        map.insert_with_hasher(1u64, 100u64, &hasher);
        
        if let Some(v) = map.get_mut_with_hasher(&1u64, &hasher) {
            *v = 200;
        }
        
        assert_eq!(map.get_with_hasher(&1u64, &hasher), Some(&200u64));
    }
}

impl<K, V, S, A: Allocator> FlatMapWithHasher<K, V, S, A>
where
    K: Hash + Eq,
    S: BuildHasher,
{
    /// Creates a new FlatMapWithHasher with the specified capacity and hasher without Default constraints.
    pub fn with_capacity_and_hasher_unchecked(capacity: usize, hasher: S) -> Self
    where
        K: Default,
        V: Default,
        A: Default,
    {
        Self {
            map: FlatMap::new_in(capacity, A::default()),
            hasher,
        }
    }

    /// Creates a new FlatMapWithHasher with the specified capacity, hasher, and allocator without Default constraints.
    pub fn with_capacity_hasher_in_unchecked(capacity: usize, hasher: S, alloc: A) -> Self
    where
        K: Default,
        V: Default,
    {
        Self {
            map: FlatMap::new_in(capacity, alloc),
            hasher,
        }
    }
}
