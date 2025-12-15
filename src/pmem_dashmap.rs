/*
 * PMEM-backed DashMap wrapper
 * 
 * This module provides a DashMap variant that allocates its internal hash table
 * structures (buckets, nodes, keys, values) exclusively in PMEM (persistent memory).
 * 
 * The purpose is to isolate hash table placement effects on cache performance.
 * By forcing hash tables to PMEM, we can measure the performance impact of
 * hash table memory placement separately from cached data placement.
 */

use std::sync::atomic::{AtomicBool, Ordering};
use dashmap::DashMap;
use std::hash::{Hash, BuildHasher};

/// Thread-local flag to force PMEM allocation for DashMap internals.
/// When true, the global allocator (HybridObjects) will always use PMEM,
/// regardless of DRAM limits.
static FORCE_PMEM_ALLOCATION: AtomicBool = AtomicBool::new(false);

/// Checks if PMEM allocation should be forced (used by allocator)
#[inline(always)]
pub fn should_force_pmem() -> bool {
    FORCE_PMEM_ALLOCATION.load(Ordering::Acquire)
}

/// Creates a DashMap with hash table structures allocated in PMEM.
/// 
/// This function temporarily sets a flag that forces the global allocator
/// to use PMEM for all allocations made during DashMap construction.
/// 
/// # Why PMEM for Hash Tables?
/// 
/// Hash tables contain:
/// - Bucket arrays (the main hash table storage)
/// - Node structures (key-value pairs with metadata)
/// - Internal metadata and pointers
/// 
/// By placing these in PMEM instead of DRAM, we can:
/// 1. Isolate the performance impact of hash table placement
/// 2. Measure cache behavior with hot path (hash lookups) in far memory
/// 3. Keep data values in their original memory tier (DRAM or PMEM as configured)
///
/// # Arguments
///
/// * `hasher` - The hash builder to use for the DashMap
///
/// # Returns
///
/// A DashMap with internal hash table structures allocated in PMEM
pub fn create_pmem_dashmap<K, V, S>(hasher: S) -> DashMap<K, V, S>
where
    K: Eq + Hash,
    S: BuildHasher + Clone,
{
    // Set flag to force PMEM allocation
    FORCE_PMEM_ALLOCATION.store(true, Ordering::Release);
    
    // Create DashMap - all internal allocations will go to PMEM
    let map = DashMap::with_hasher(hasher);
    
    // Reset flag to normal allocation mode
    FORCE_PMEM_ALLOCATION.store(false, Ordering::Release);
    
    map
}
