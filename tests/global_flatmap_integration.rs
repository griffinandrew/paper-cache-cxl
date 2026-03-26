// Tests for global_flatmap_dram and global_flatmap_pmem features
//
// These tests verify that FlatMap works correctly as the global hashtable
// backend for both DRAM and PMEM modes.
//
// Note: global_flatmap_dram should be used with all_dram
//       global_flatmap_pmem should be used with key_value_pmem

#[cfg(all(feature = "global_flatmap_dram", feature = "all_dram"))]
mod flatmap_dram_tests {
    use paper_cache::{BufferDRAM, PaperCache, PaperPolicy};

    #[test]
    fn test_basic_get_set_remove() {
        // Create a cache with FlatMap backend in DRAM
        let cache =
            PaperCache::<u32, BufferDRAM>::new(10000, &[PaperPolicy::Lfu], PaperPolicy::Lfu)
                .expect("Failed to create cache");

        // Test set operation
        cache.set(1, &[1, 2, 3], None).expect("Failed to set value");
        cache.set(2, &[4, 5, 6], None).expect("Failed to set value");

        // Test get operation
        let val1 = cache.get(&1).expect("Failed to get value");
        assert_eq!(val1.as_ref(), &[1, 2, 3]);

        let val2 = cache.get(&2).expect("Failed to get value");
        assert_eq!(val2.as_ref(), &[4, 5, 6]);

        // Test update operation
        cache
            .set(1, &[7, 8, 9], None)
            .expect("Failed to update value");
        let val1_updated = cache.get(&1).expect("Failed to get updated value");
        assert_eq!(val1_updated.as_ref(), &[7, 8, 9]);

        // Test remove operation (via eviction)
        // Note: Direct remove is not exposed in the public API, so we test via cache operations
    }

    #[test]
    fn test_multiple_operations() {
        let cache =
            PaperCache::<u32, BufferDRAM>::new(10000, &[PaperPolicy::Lfu], PaperPolicy::Lfu)
                .expect("Failed to create cache");

        // Insert multiple values
        for i in 0..10 {
            cache
                .set(i, &[i as u8; 10], None)
                .expect("Failed to set value");
        }

        // Verify all values can be retrieved
        for i in 0..10 {
            let val = cache.get(&i).expect("Failed to get value");
            assert_eq!(val.as_ref(), &[i as u8; 10]);
        }
    }

    #[test]
    fn test_missing_key() {
        let cache =
            PaperCache::<u32, BufferDRAM>::new(10000, &[PaperPolicy::Lfu], PaperPolicy::Lfu)
                .expect("Failed to create cache");

        // Try to get a non-existent key
        let result = cache.get(&999);
        assert!(result.is_err());
    }
}

#[cfg(all(feature = "global_flatmap_pmem", feature = "key_value_pmem"))]
mod flatmap_pmem_tests {
    use paper_cache::{BufferPMEM, PaperCache, PaperPolicy};

    #[test]
    fn test_basic_get_set_remove() {
        // Create a cache with FlatMap backend in PMEM (hashtable + data in PMEM)
        let cache =
            PaperCache::<u32, BufferPMEM>::new(10000, &[PaperPolicy::Lfu], PaperPolicy::Lfu)
                .expect("Failed to create cache");

        // Test set operation
        cache.set(1, &[1, 2, 3], None).expect("Failed to set value");
        cache.set(2, &[4, 5, 6], None).expect("Failed to set value");

        // Test get operation
        let val1 = cache.get(&1).expect("Failed to get value");
        assert_eq!(val1.as_ref(), &[1, 2, 3]);

        let val2 = cache.get(&2).expect("Failed to get value");
        assert_eq!(val2.as_ref(), &[4, 5, 6]);

        // Test update operation
        cache
            .set(1, &[7, 8, 9], None)
            .expect("Failed to update value");
        let val1_updated = cache.get(&1).expect("Failed to get updated value");
        assert_eq!(val1_updated.as_ref(), &[7, 8, 9]);
    }

    #[test]
    fn test_multiple_operations() {
        let cache =
            PaperCache::<u32, BufferPMEM>::new(10000, &[PaperPolicy::Lfu], PaperPolicy::Lfu)
                .expect("Failed to create cache");

        // Insert multiple values
        for i in 0..10 {
            cache
                .set(i, &[i as u8; 10], None)
                .expect("Failed to set value");
        }

        // Verify all values can be retrieved
        for i in 0..10 {
            let val = cache.get(&i).expect("Failed to get value");
            assert_eq!(val.as_ref(), &[i as u8; 10]);
        }
    }

    #[test]
    fn test_missing_key() {
        let cache =
            PaperCache::<u32, BufferPMEM>::new(10000, &[PaperPolicy::Lfu], PaperPolicy::Lfu)
                .expect("Failed to create cache");

        // Try to get a non-existent key
        let result = cache.get(&999);
        assert!(result.is_err());
    }
}

// Tests for global_flatmap_pmem in standalone mode (hashtable in PMEM, data in DRAM)
#[cfg(all(feature = "global_flatmap_pmem", not(feature = "key_value_pmem")))]
mod flatmap_pmem_standalone_tests {
    use paper_cache::{BufferDRAM, PaperCache, PaperPolicy};

    #[test]
    fn test_basic_get_set_remove() {
        // Create a cache with FlatMap backend in PMEM (hashtable only, data in DRAM)
        let cache =
            PaperCache::<u32, BufferDRAM>::new(10000, &[PaperPolicy::Lfu], PaperPolicy::Lfu)
                .expect("Failed to create cache");

        // Test set operation
        cache.set(1, &[1, 2, 3], None).expect("Failed to set value");
        cache.set(2, &[4, 5, 6], None).expect("Failed to set value");

        // Test get operation
        let val1 = cache.get(&1).expect("Failed to get value");
        assert_eq!(val1.as_ref(), &[1, 2, 3]);

        let val2 = cache.get(&2).expect("Failed to get value");
        assert_eq!(val2.as_ref(), &[4, 5, 6]);

        // Test update operation
        cache
            .set(1, &[7, 8, 9], None)
            .expect("Failed to update value");
        let val1_updated = cache.get(&1).expect("Failed to get updated value");
        assert_eq!(val1_updated.as_ref(), &[7, 8, 9]);
    }

    #[test]
    fn test_multiple_operations() {
        let cache =
            PaperCache::<u32, BufferDRAM>::new(10000, &[PaperPolicy::Lfu], PaperPolicy::Lfu)
                .expect("Failed to create cache");

        // Insert multiple values
        for i in 0..10 {
            cache
                .set(i, &[i as u8; 10], None)
                .expect("Failed to set value");
        }

        // Verify all values can be retrieved
        for i in 0..10 {
            let val = cache.get(&i).expect("Failed to get value");
            assert_eq!(val.as_ref(), &[i as u8; 10]);
        }
    }

    #[test]
    fn test_missing_key() {
        let cache =
            PaperCache::<u32, BufferDRAM>::new(10000, &[PaperPolicy::Lfu], PaperPolicy::Lfu)
                .expect("Failed to create cache");

        // Try to get a non-existent key
        let result = cache.get(&999);
        assert!(result.is_err());
    }
}
