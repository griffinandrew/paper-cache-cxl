/*
 * Performance Counters for Memory Access Tracking
 * 
 * This module provides thread-safe atomic counters to track memory access patterns
 * for hashmap structures with both DRAM and PMEM configurations.
 */

use std::sync::atomic::{AtomicU64, Ordering};

/// Performance counters for hashmap memory accesses
#[derive(Debug, Default)]
pub struct HashMapCounters {
    // Read operations
    pub reads: AtomicU64,           // Total read accesses (get, contains_key, iteration)
    pub lookups: AtomicU64,         // Lookup operations (get, contains_key)
    pub iterations: AtomicU64,      // Iteration accesses
    
    // Write operations
    pub writes: AtomicU64,          // Total write accesses (insert, remove, clear)
    pub insertions: AtomicU64,      // Insert operations
    pub deletions: AtomicU64,       // Remove operations
    pub clears: AtomicU64,          // Clear operations
}

impl HashMapCounters {
    pub fn new() -> Self {
        Self::default()
    }

    // Read operation tracking
    #[inline]
    pub fn incr_read(&self) {
        self.reads.fetch_add(1, Ordering::Relaxed);
    }

    #[inline]
    pub fn incr_lookup(&self) {
        self.reads.fetch_add(1, Ordering::Relaxed);
        self.lookups.fetch_add(1, Ordering::Relaxed);
    }

    #[inline]
    pub fn incr_iteration(&self) {
        self.reads.fetch_add(1, Ordering::Relaxed);
        self.iterations.fetch_add(1, Ordering::Relaxed);
    }

    // Write operation tracking
    #[inline]
    pub fn incr_write(&self) {
        self.writes.fetch_add(1, Ordering::Relaxed);
    }

    #[inline]
    pub fn incr_insertion(&self) {
        self.writes.fetch_add(1, Ordering::Relaxed);
        self.insertions.fetch_add(1, Ordering::Relaxed);
    }

    #[inline]
    pub fn incr_deletion(&self) {
        self.writes.fetch_add(1, Ordering::Relaxed);
        self.deletions.fetch_add(1, Ordering::Relaxed);
    }

    #[inline]
    pub fn incr_clear(&self) {
        self.writes.fetch_add(1, Ordering::Relaxed);
        self.clears.fetch_add(1, Ordering::Relaxed);
    }

    // Getters
    pub fn get_reads(&self) -> u64 {
        self.reads.load(Ordering::Relaxed)
    }

    pub fn get_lookups(&self) -> u64 {
        self.lookups.load(Ordering::Relaxed)
    }

    pub fn get_iterations(&self) -> u64 {
        self.iterations.load(Ordering::Relaxed)
    }

    pub fn get_writes(&self) -> u64 {
        self.writes.load(Ordering::Relaxed)
    }

    pub fn get_insertions(&self) -> u64 {
        self.insertions.load(Ordering::Relaxed)
    }

    pub fn get_deletions(&self) -> u64 {
        self.deletions.load(Ordering::Relaxed)
    }

    pub fn get_clears(&self) -> u64 {
        self.clears.load(Ordering::Relaxed)
    }

    pub fn get_total_accesses(&self) -> u64 {
        self.get_reads() + self.get_writes()
    }

    /// Reset all counters
    pub fn reset(&self) {
        self.reads.store(0, Ordering::Relaxed);
        self.lookups.store(0, Ordering::Relaxed);
        self.iterations.store(0, Ordering::Relaxed);
        self.writes.store(0, Ordering::Relaxed);
        self.insertions.store(0, Ordering::Relaxed);
        self.deletions.store(0, Ordering::Relaxed);
        self.clears.store(0, Ordering::Relaxed);
    }
}

/// Global performance counters for the cache
#[derive(Debug, Default)]
pub struct GlobalPerfCounters {
    // Global hashtable counters
    #[cfg(feature = "hashbrown_dram")]
    pub global_hashbrown_dram: HashMapCounters,
    
    #[cfg(feature = "global_hashtable_pmem")]
    pub global_hashbrown_pmem: HashMapCounters,
    
    #[cfg(feature = "global_flatmap_dram")]
    pub global_flatmap_dram: HashMapCounters,
    
    #[cfg(feature = "global_flatmap_pmem")]
    pub global_flatmap_pmem: HashMapCounters,
    
    // Tiering manager hashtable counters
    #[cfg(all(any(feature = "key_value_pmem", feature = "alloc_api_exp"), feature = "enable_tiering_manager", not(feature = "tiering_hashtable_pmem")))]
    pub tiering_hashtable_dram: HashMapCounters,
    
    #[cfg(all(any(feature = "key_value_pmem", feature = "alloc_api_exp"), feature = "enable_tiering_manager", feature = "tiering_hashtable_pmem"))]
    pub tiering_hashtable_pmem: HashMapCounters,
    
    // Total memory accesses (optional - for calculating percentages)
    pub total_memory_accesses: AtomicU64,
}

impl GlobalPerfCounters {
    pub fn new() -> Self {
        Self::default()
    }

    #[inline]
    pub fn incr_total_memory_access(&self) {
        self.total_memory_accesses.fetch_add(1, Ordering::Relaxed);
    }

    pub fn get_total_memory_accesses(&self) -> u64 {
        self.total_memory_accesses.load(Ordering::Relaxed)
    }
}

/// Statistics snapshot for reporting
#[derive(Debug, Clone)]
pub struct HashMapStats {
    pub reads: u64,
    pub lookups: u64,
    pub iterations: u64,
    pub writes: u64,
    pub insertions: u64,
    pub deletions: u64,
    pub clears: u64,
    pub total_accesses: u64,
}

impl From<&HashMapCounters> for HashMapStats {
    fn from(counters: &HashMapCounters) -> Self {
        HashMapStats {
            reads: counters.get_reads(),
            lookups: counters.get_lookups(),
            iterations: counters.get_iterations(),
            writes: counters.get_writes(),
            insertions: counters.get_insertions(),
            deletions: counters.get_deletions(),
            clears: counters.get_clears(),
            total_accesses: counters.get_total_accesses(),
        }
    }
}

impl std::fmt::Display for HashMapStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "HashMap Performance Statistics:")?;
        writeln!(f, "  Total Accesses: {}", self.total_accesses)?;
        writeln!(f, "  Reads: {} ({:.1}%)", self.reads, 
                 100.0 * self.reads as f64 / self.total_accesses.max(1) as f64)?;
        writeln!(f, "    - Lookups: {}", self.lookups)?;
        writeln!(f, "    - Iterations: {}", self.iterations)?;
        writeln!(f, "  Writes: {} ({:.1}%)", self.writes,
                 100.0 * self.writes as f64 / self.total_accesses.max(1) as f64)?;
        writeln!(f, "    - Insertions: {}", self.insertions)?;
        writeln!(f, "    - Deletions: {}", self.deletions)?;
        writeln!(f, "    - Clears: {}", self.clears)?;
        Ok(())
    }
}

/// Global instance of performance counters
use std::sync::OnceLock;

static PERF_COUNTERS: OnceLock<GlobalPerfCounters> = OnceLock::new();

/// Get the global performance counters instance
pub fn get_global_counters() -> &'static GlobalPerfCounters {
    PERF_COUNTERS.get_or_init(GlobalPerfCounters::new)
}

/// Get hashmap statistics based on active features
/// Returns statistics for the active global hashtable configuration
pub fn get_hashmap_stats() -> Option<HashMapStats> {
    let counters = get_global_counters();
    
    #[cfg(feature = "hashbrown_dram")]
    {
        return Some(HashMapStats::from(&counters.global_hashbrown_dram));
    }
    
    #[cfg(all(feature = "global_hashtable_pmem", not(feature = "hashbrown_dram")))]
    {
        return Some(HashMapStats::from(&counters.global_hashbrown_pmem));
    }
    
    #[cfg(all(feature = "global_flatmap_dram", not(feature = "hashbrown_dram"), not(feature = "global_hashtable_pmem")))]
    {
        return Some(HashMapStats::from(&counters.global_flatmap_dram));
    }
    
    #[cfg(all(feature = "global_flatmap_pmem", not(feature = "hashbrown_dram"), not(feature = "global_hashtable_pmem"), not(feature = "global_flatmap_dram")))]
    {
        return Some(HashMapStats::from(&counters.global_flatmap_pmem));
    }
    
    #[cfg(not(any(feature = "hashbrown_dram", feature = "global_hashtable_pmem", feature = "global_flatmap_dram", feature = "global_flatmap_pmem")))]
    {
        None
    }
}

/// Get tiering hashtable statistics if tiering is enabled
pub fn get_tiering_hashtable_stats() -> Option<HashMapStats> {
    #[cfg(all(any(feature = "key_value_pmem", feature = "alloc_api_exp"), feature = "enable_tiering_manager", not(feature = "tiering_hashtable_pmem")))]
    {
        let counters = get_global_counters();
        return Some(HashMapStats::from(&counters.tiering_hashtable_dram));
    }
    
    #[cfg(all(any(feature = "key_value_pmem", feature = "alloc_api_exp"), feature = "enable_tiering_manager", feature = "tiering_hashtable_pmem"))]
    {
        let counters = get_global_counters();
        return Some(HashMapStats::from(&counters.tiering_hashtable_pmem));
    }
    
    #[cfg(not(all(any(feature = "key_value_pmem", feature = "alloc_api_exp"), feature = "enable_tiering_manager")))]
    {
        None
    }
}

/// Print performance statistics summary
pub fn print_perf_stats() {
    println!("\n=== PaperCache Performance Statistics ===\n");
    
    if let Some(stats) = get_hashmap_stats() {
        #[cfg(feature = "hashbrown_dram")]
        println!("Global HashMap (hashbrown in DRAM):");
        
        #[cfg(all(feature = "global_hashtable_pmem", not(feature = "hashbrown_dram")))]
        println!("Global HashMap (hashbrown in PMEM):");
        
        #[cfg(all(feature = "global_flatmap_dram", not(feature = "hashbrown_dram"), not(feature = "global_hashtable_pmem")))]
        println!("Global HashMap (FlatMap in DRAM):");
        
        #[cfg(all(feature = "global_flatmap_pmem", not(feature = "hashbrown_dram"), not(feature = "global_hashtable_pmem"), not(feature = "global_flatmap_dram")))]
        println!("Global HashMap (FlatMap in PMEM):");
        
        print!("{}", stats);
    } else {
        println!("Global HashMap: Using DashMap (no performance counters)");
    }
    
    println!();
    
    if let Some(stats) = get_tiering_hashtable_stats() {
        #[cfg(all(any(feature = "key_value_pmem", feature = "alloc_api_exp"), feature = "enable_tiering_manager", not(feature = "tiering_hashtable_pmem")))]
        println!("Tiering Manager HashMap (in DRAM):");
        
        #[cfg(all(any(feature = "key_value_pmem", feature = "alloc_api_exp"), feature = "enable_tiering_manager", feature = "tiering_hashtable_pmem"))]
        println!("Tiering Manager HashMap (in PMEM):");
        
        print!("{}", stats);
        println!();
    }
    
    let total_mem = get_global_counters().get_total_memory_accesses();
    if total_mem > 0 {
        println!("Total Memory Accesses: {}", total_mem);
        
        if let Some(stats) = get_hashmap_stats() {
            let percentage = 100.0 * stats.total_accesses as f64 / total_mem as f64;
            println!("HashMap accesses as % of total: {:.2}%", percentage);
        }
    }
    
    println!("\n==========================================\n");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hashmap_counters() {
        let counters = HashMapCounters::new();
        
        assert_eq!(counters.get_reads(), 0);
        assert_eq!(counters.get_writes(), 0);
        
        counters.incr_lookup();
        assert_eq!(counters.get_reads(), 1);
        assert_eq!(counters.get_lookups(), 1);
        
        counters.incr_insertion();
        assert_eq!(counters.get_writes(), 1);
        assert_eq!(counters.get_insertions(), 1);
        
        assert_eq!(counters.get_total_accesses(), 2);
        
        counters.reset();
        assert_eq!(counters.get_total_accesses(), 0);
    }

    #[test]
    fn test_stats_conversion() {
        let counters = HashMapCounters::new();
        counters.incr_lookup();
        counters.incr_insertion();
        
        let stats = HashMapStats::from(&counters);
        assert_eq!(stats.reads, 1);
        assert_eq!(stats.writes, 1);
        assert_eq!(stats.total_accesses, 2);
    }
}
