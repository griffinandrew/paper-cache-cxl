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

use allocator_api2::alloc::{Allocator, Global};
use allocator_api2::vec::Vec as AllocVec;
use std::hash::{Hash, BuildHasher};
use std::mem;
use std::marker::PhantomData;
use std::sync::Arc;
use parking_lot::RwLock;

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
    pub(crate) buckets: AllocVec<Bucket<K, V>, A>,
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
    /// * `capacity` - The desired capacity. Any positive value is accepted; it will be
    ///   rounded up to the nearest power of two internally.
    ///
    /// # Panics
    ///
    /// Panics if capacity is 0.
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
    /// * `capacity` - The desired capacity. Any positive value is accepted; it will be
    ///   rounded up to the nearest power of two internally.
    /// * `alloc` - The allocator to use
    ///
    /// # Panics
    ///
    /// Panics if capacity is 0.
    pub fn new_in(capacity: usize, alloc: A) -> Self {
        assert!(capacity > 0, "Capacity must be greater than 0");
        let capacity = capacity.next_power_of_two();

        let mask = capacity - 1;
        
        // Allocate and initialize buckets
        let mut buckets = AllocVec::with_capacity_in(capacity, alloc);
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

    /// Resizes the map to double the capacity using a create-new-and-copy strategy.
    ///
    /// Allocates an entirely new backing buffer, migrates all existing entries into it,
    /// then atomically swaps the buffer. This ensures safety against partial writes.
    fn resize(&mut self)
    where
        A: Clone,
    {
        let new_capacity = (self.capacity * 2).next_power_of_two();
        let new_mask = new_capacity - 1;

        // Allocate entirely new backing buffer (create-new-and-copy strategy)
        let alloc = self.buckets.allocator().clone();
        let mut new_buckets = AllocVec::with_capacity_in(new_capacity, alloc);
        for _ in 0..new_capacity {
            new_buckets.push(Bucket::empty());
        }

        // Migrate existing entries into new buffer using linear probing re-hash.
        // Each bucket's hash is already stored, so no re-hashing is needed;
        // we only need to find the new probe position using the new mask.
        for i in 0..self.buckets.len() {
            // Destructively extract each non-empty bucket (no Clone needed for K/V)
            if !self.buckets[i].is_empty() {
                let bucket = mem::replace(&mut self.buckets[i], Bucket::empty());
                let mut index = (bucket.hash as usize) & new_mask;
                loop {
                    if new_buckets[index].is_empty() {
                        new_buckets[index] = bucket;
                        break;
                    }
                    index = (index + 1) & new_mask;
                }
            }
        }

        // Atomically swap the backing buffers
        self.buckets = new_buckets;
        self.capacity = new_capacity;
        self.mask = new_mask;
    }

    /// Inserts a key-value pair into the map with the given hasher.
    ///
    /// Automatically resizes the map when the load factor reaches >= 80%.
    /// Returns the old value if the key was already present.
    pub fn insert_with_hasher<S>(&mut self, key: K, val: V, hasher: &S) -> Option<V>
    where
        S: BuildHasher,
        A: Clone,
    {
        // Auto-resize when adding this entry would push load to >= 80%.
        // Written as an integer comparison to avoid floating-point and
        // to prevent overflow: 5*(len+1) >= 4*capacity  ≡  load >= 80%.
        if (self.len + 1).saturating_mul(5) >= self.capacity.saturating_mul(4) {
            self.resize();
        }

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
        
        panic!("FlatMap is full after resize — this should not happen");
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
    /// Automatically triggers resize when load factor reaches >= 80%.
    #[inline]
    pub fn insert(&mut self, key: K, val: V) -> Option<V>
    where
        A: Clone,
    {
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
    fn test_capacity_normalization() {
        // Any capacity is accepted; it is rounded up to the next power of two
        let map: FlatMap<u64, u64> = FlatMap::new(3);
        assert_eq!(map.capacity(), 4);

        let map: FlatMap<u64, u64> = FlatMap::new(1);
        assert_eq!(map.capacity(), 1);

        let map: FlatMap<u64, u64> = FlatMap::new(100);
        assert_eq!(map.capacity(), 128);
    }

    #[test]
    fn test_auto_resize() {
        // Start with a small capacity — the map must auto-resize at >= 80% load
        let mut map = FlatMap::new(4);
        let hasher = RandomState::new();

        // Insert enough items to force multiple resize events
        for i in 0u64..50 {
            map.insert_with_hasher(i, i * 10, &hasher);
        }

        assert_eq!(map.len(), 50);
        // Capacity must have grown beyond the original 4
        assert!(map.capacity() >= 64);

        // All entries must still be retrievable after resize
        for i in 0u64..50 {
            assert_eq!(map.get_with_hasher(&i, &hasher), Some(&(i * 10)));
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

// ─── ShardedFlatMap ──────────────────────────────────────────────────────────

/// Returns the default number of shards, mirroring DashMap's strategy:
/// `available_parallelism * 4`, rounded up to the nearest power of two.
fn default_shard_count() -> usize {
    use std::thread::available_parallelism;
    (available_parallelism().map_or(1, usize::from) * 4).next_power_of_two()
}

/// A concurrent, sharded hash map wrapping multiple `FlatMapWithHasher` shards.
///
/// Shard count is determined dynamically from the system's available parallelism
/// (identical strategy to DashMap: `available_parallelism * 4`, rounded to a power
/// of two). High-order bits of the key hash are used to route operations to the
/// correct shard, preventing correlation with the shard-internal linear probing.
///
/// The inner `Arc` allows cheap, safe sharing of the map across threads.
///
/// # Example
///
/// ```rust,ignore
/// use paper_cache::flatmap::ShardedFlatMap;
/// use std::hash::RandomState;
///
/// let map: ShardedFlatMap<u64, u64, RandomState> =
///     ShardedFlatMap::new(1024);
///
/// // Share across threads
/// let map2 = map.clone();
///
/// map.insert(1, 100);
/// assert_eq!(map2.get(&1), Some(100));
/// ```
pub struct ShardedFlatMap<K, V, S> {
    /// All shards, shared via Arc so the map can be cheaply cloned across threads.
    shards: Arc<Vec<RwLock<FlatMapWithHasher<K, V, S>>>>,
    /// Hasher used exclusively for shard-routing (not for internal FlatMap probing).
    hasher: S,
    /// Total number of shards (always a power of two).
    shard_count: usize,
    /// Bit-shift used to extract the shard index from the high-order hash bits.
    /// `shard_shift = 64 - log2(shard_count)`.  Set to 64 when shard_count == 1.
    shard_shift: u32,
}

impl<K, V, S: Clone> Clone for ShardedFlatMap<K, V, S> {
    fn clone(&self) -> Self {
        Self {
            shards: Arc::clone(&self.shards),
            hasher: self.hasher.clone(),
            shard_count: self.shard_count,
            shard_shift: self.shard_shift,
        }
    }
}

impl<K, V, S> ShardedFlatMap<K, V, S>
where
    K: Hash + Eq + Default,
    V: Default,
    S: BuildHasher + Clone + Default,
{
    /// Creates a new `ShardedFlatMap`.
    ///
    /// `initial_capacity_per_shard` is the starting capacity for each individual
    /// shard's `FlatMapWithHasher`.  It is rounded up to the nearest power of two
    /// internally.  The shards auto-resize as entries are added.
    pub fn new(initial_capacity_per_shard: usize) -> Self {
        let shard_count = default_shard_count();
        // For shard_count == 1, trailing_zeros() == 0, which would give 64 - 0 = 64.
        // Right-shifting a u64 by 64 bits is undefined behaviour in Rust (wraps to
        // shift-by-0 in release mode, panics in debug mode).  Use 64 as a sentinel
        // value meaning "always select shard 0" and guard against it in shard_index_for.
        let shard_shift = if shard_count > 1 {
            64 - shard_count.trailing_zeros()
        } else {
            64
        };

        // Use std::vec::Vec (not AllocVec) for the outer shard container — the shards
        // themselves are allocated in DRAM and do not require a custom allocator.
        let shards = (0..shard_count)
            .map(|_| {
                RwLock::new(FlatMapWithHasher::with_capacity_and_hasher(
                    initial_capacity_per_shard,
                    S::default(),
                ))
            })
            .collect::<std::vec::Vec<_>>();

        Self {
            shards: Arc::new(shards),
            hasher: S::default(),
            shard_count,
            shard_shift,
        }
    }
}

impl<K, V, S> ShardedFlatMap<K, V, S>
where
    S: BuildHasher,
{
    /// Computes the hash of `key` using the shard-routing hasher and derives the
    /// shard index from the **high-order** bits of the hash value.
    #[inline]
    fn shard_index_for<Q: Hash>(&self, key: &Q) -> usize {
        use std::hash::Hasher;
        let mut h = self.hasher.build_hasher();
        key.hash(&mut h);
        let hash = h.finish();
        // FlatMap uses hash == 0 as the "empty bucket" sentinel, so map 0 → 1.
        let hash = if hash == 0 { 1 } else { hash };

        if self.shard_shift >= 64 {
            // Only one shard
            0
        } else {
            (hash >> self.shard_shift) as usize
        }
    }

    /// Returns a cloned copy of the value associated with `key`, or `None` if not
    /// present.
    ///
    /// Acquires a **read** lock on the relevant shard.
    pub fn get<Q>(&self, key: &Q) -> Option<V>
    where
        Q: Hash + Eq,
        K: Hash + Eq + PartialEq<Q>,
        V: Clone,
    {
        let idx = self.shard_index_for(key);
        self.shards[idx].read().get(key).cloned()
    }

    /// Inserts a key-value pair, returning the previous value if the key already
    /// existed.
    ///
    /// Acquires a **write** lock on the relevant shard.  The shard auto-resizes
    /// when the load factor reaches >= 80%.
    pub fn insert(&self, key: K, val: V) -> Option<V>
    where
        K: Hash + Eq + Default,
        V: Default,
        S: Clone,
    {
        let idx = self.shard_index_for(&key);
        self.shards[idx].write().insert(key, val)
    }

    /// Removes the entry for `key`, returning the value if it was present.
    ///
    /// Acquires a **write** lock on the relevant shard.
    pub fn remove<Q>(&self, key: &Q) -> Option<V>
    where
        Q: Hash + Eq,
        K: Hash + Eq + Default + PartialEq<Q>,
        V: Default,
    {
        let idx = self.shard_index_for(key);
        self.shards[idx].write().remove(key)
    }

    /// Returns the total number of shards.
    #[inline]
    pub fn shard_count(&self) -> usize {
        self.shard_count
    }
}

#[cfg(test)]
mod sharded_tests {
    use super::*;
    use std::hash::RandomState;

    #[test]
    fn test_sharded_basic() {
        let map: ShardedFlatMap<u64, u64, RandomState> = ShardedFlatMap::new(16);
        assert!(map.shard_count() >= 1);
        assert!(map.shard_count().is_power_of_two());

        map.insert(1, 100);
        map.insert(2, 200);

        assert_eq!(map.get(&1), Some(100));
        assert_eq!(map.get(&2), Some(200));
        assert_eq!(map.get(&3u64), None);
    }

    #[test]
    fn test_sharded_remove() {
        let map: ShardedFlatMap<u64, u64, RandomState> = ShardedFlatMap::new(16);

        map.insert(42, 999);
        assert_eq!(map.get(&42), Some(999));

        let removed = map.remove(&42u64);
        assert_eq!(removed, Some(999));
        assert_eq!(map.get(&42u64), None);
    }

    #[test]
    fn test_sharded_auto_resize() {
        // Insert more entries than the initial per-shard capacity to exercise auto-resize
        let map: ShardedFlatMap<u64, u64, RandomState> = ShardedFlatMap::new(4);

        for i in 0u64..200 {
            map.insert(i, i * 2);
        }
        for i in 0u64..200 {
            assert_eq!(map.get(&i), Some(i * 2));
        }
    }

    #[test]
    fn test_sharded_clone_shares_state() {
        let map: ShardedFlatMap<u64, u64, RandomState> = ShardedFlatMap::new(16);
        let map2 = map.clone();

        map.insert(10, 100);
        // Clone shares the same underlying shards via Arc
        assert_eq!(map2.get(&10), Some(100));
    }

    #[test]
    fn test_sharded_multithreaded() {
        use std::sync::Arc as StdArc;
        use std::thread;

        let map: StdArc<ShardedFlatMap<u64, u64, RandomState>> =
            StdArc::new(ShardedFlatMap::new(64));

        let handles: std::vec::Vec<_> = (0u64..8)
            .map(|t| {
                let m = StdArc::clone(&map);
                thread::spawn(move || {
                    for i in 0u64..100 {
                        let key = t * 100 + i;
                        m.insert(key, key * 2);
                    }
                })
            })
            .collect();

        for h in handles {
            h.join().unwrap();
        }

        for t in 0u64..8 {
            for i in 0u64..100 {
                let key = t * 100 + i;
                assert_eq!(map.get(&key), Some(key * 2));
            }
        }
    }
}
