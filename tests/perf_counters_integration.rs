#[cfg(feature = "hashbrown_dram")]
#[cfg(test)]
mod test_perf_counters_hashbrown_dram {
    use paper_cache::{PaperCache, PaperPolicy};
    use paper_cache::perf_counters::{get_global_counters, get_hashmap_stats};

    #[test]
    fn test_counters_track_operations() {
        let counters = get_global_counters();
        
        // Record initial values
        let initial_insertions = counters.global_hashbrown_dram.get_insertions();
        let initial_reads = counters.global_hashbrown_dram.get_reads();
        let initial_deletions = counters.global_hashbrown_dram.get_deletions();
        let initial_writes = counters.global_hashbrown_dram.get_writes();

        // Create cache
        let cache = PaperCache::<u64, Box<[u8]>>::new(
            10_000_000,
            &[PaperPolicy::Lru],
            PaperPolicy::Lru,
        ).unwrap();

        // Insert 10 items (10 writes)
        for i in 0..10 {
            cache.set(i, b"test", None).unwrap();
        }
        // Should be at least 10 insertions more than initial
        assert!(counters.global_hashbrown_dram.get_insertions() - initial_insertions >= 10);
        assert!(counters.global_hashbrown_dram.get_writes() - initial_writes >= 10);

        // Read 5 items (5 reads)
        for i in 0..5 {
            let _ = cache.get(&i);
        }
        let reads_after_get = counters.global_hashbrown_dram.get_reads() - initial_reads;
        assert!(reads_after_get >= 5);

        // Check 3 items with has() (3 more reads)
        for i in 5..8 {
            let _ = cache.has(&i);
        }
        let reads_after_has = counters.global_hashbrown_dram.get_reads() - initial_reads;
        assert!(reads_after_has >= reads_after_get + 3);

        // Delete 2 items (2 more writes)
        for i in 0..2 {
            let _ = cache.del(&i);
        }
        assert!(counters.global_hashbrown_dram.get_deletions() - initial_deletions >= 2);
        assert!(counters.global_hashbrown_dram.get_writes() - initial_writes >= 12);

        // Check the stats match our expectations
        let stats = get_hashmap_stats().unwrap();
        assert!(stats.total_accesses >= 20); // At least 20 from this test
        assert!(stats.insertions >= 10);
        assert!(stats.deletions >= 2);
    }

    #[test]
    fn test_peek_increments_counter() {
        let counters = get_global_counters();
        counters.global_hashbrown_dram.reset();

        let cache = PaperCache::<u64, Box<[u8]>>::new(
            10_000_000,
            &[PaperPolicy::Lru],
            PaperPolicy::Lru,
        ).unwrap();

        cache.set(1, b"test", None).unwrap();
        let initial_reads = counters.global_hashbrown_dram.get_reads();
        
        let _ = cache.peek(&1);
        
        // peek should increment read counter
        assert_eq!(counters.global_hashbrown_dram.get_reads(), initial_reads + 1);
    }
}

#[cfg(all(feature = "global_hashtable_pmem", not(feature = "hashbrown_dram")))]
#[cfg(test)]
mod test_perf_counters_hashbrown_pmem {
    use paper_cache::{PaperCache, PaperPolicy};
    use paper_cache::perf_counters::{get_global_counters, get_hashmap_stats};

    #[test]
    fn test_counters_track_operations_pmem() {
        // Reset counters
        let counters = get_global_counters();
        counters.global_hashbrown_pmem.reset();

        // Create cache
        let cache = PaperCache::<u64, Box<[u8]>>::new(
            10_000_000,
            &[PaperPolicy::Lru],
            PaperPolicy::Lru,
        ).unwrap();

        // Initially should be 0
        assert_eq!(counters.global_hashbrown_pmem.get_total_accesses(), 0);

        // Insert 5 items (5 writes)
        for i in 0..5 {
            cache.set(i, b"test", None).unwrap();
        }
        assert_eq!(counters.global_hashbrown_pmem.get_insertions(), 5);
        assert_eq!(counters.global_hashbrown_pmem.get_writes(), 5);

        // Read 3 items (3 reads)
        for i in 0..3 {
            let _ = cache.get(&i);
        }
        assert_eq!(counters.global_hashbrown_pmem.get_lookups(), 3);
        assert_eq!(counters.global_hashbrown_pmem.get_reads(), 3);

        // Total should be reads + writes
        assert_eq!(counters.global_hashbrown_pmem.get_total_accesses(), 8);

        // Test stats retrieval
        let stats = get_hashmap_stats().unwrap();
        assert_eq!(stats.total_accesses, 8);
        assert_eq!(stats.reads, 3);
        assert_eq!(stats.writes, 5);
    }
}
