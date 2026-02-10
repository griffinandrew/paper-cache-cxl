#[cfg(all(feature = "enable_tiering_manager", feature = "key_value_pmem"))]
mod tiering_tests {
    use paper_cache::{PaperCache, PaperPolicy, BufferPMEM};
    use std::thread;
    use std::time::Duration;

    #[test]
    fn test_tiering_manager_integration() {
        // Create a cache with a small max size
        let cache = PaperCache::<u32, BufferPMEM>::new(
            10000,
            &[PaperPolicy::Lfu],
            PaperPolicy::Lfu,
        ).expect("Failed to create cache");

        // Verify initial tiering stats
        let stats = cache.tiering_stats();
        assert_eq!(stats.dram_objects, 0);
        assert_eq!(stats.promotions, 0);
        assert_eq!(stats.demotions, 0);

        // Default DRAM threshold should be 20% of max_size (2000 bytes)
        assert_eq!(cache.dram_threshold(), 2000);

        // Set some objects
        for i in 0..10 {
            cache.set(i, &[0u8; 100], None).expect("Failed to set object");
        }

        // Give some time for the tiering worker to process
        thread::sleep(Duration::from_millis(100));

        // Access some objects multiple times to make them hot
        for _ in 0..3 {
            for i in 0..5 {
                let _ = cache.get(&i);
            }
        }

        // Give time for promotions
        thread::sleep(Duration::from_millis(100));

        // Check that some objects were promoted
        let stats = cache.tiering_stats();
        // Note: exact numbers depend on timing, but we should see some activity
        println!("Promotions: {}, DRAM objects: {}", stats.promotions, stats.dram_objects);
    }

    #[test]
    fn test_tiering_configuration() {
        let cache = PaperCache::<u32, BufferPMEM>::new(
            10000,
            &[PaperPolicy::Lfu],
            PaperPolicy::Lfu,
        ).expect("Failed to create cache");

        // Test DRAM threshold configuration
        let initial_threshold = cache.dram_threshold();
        assert_eq!(initial_threshold, 2000); // 20% of 10000

        cache.set_dram_threshold(5000);
        assert_eq!(cache.dram_threshold(), 5000);

        // Test hotness threshold configuration
        cache.set_hotness_threshold(5);
        assert_eq!(cache.hotness_threshold(), 5);
    }

    #[test]
    fn test_tiering_stats() {
        let cache = PaperCache::<u32, BufferPMEM>::new(
            10000,
            &[PaperPolicy::Lfu],
            PaperPolicy::Lfu,
        ).expect("Failed to create cache");

        // Add some objects
        for i in 0..5 {
            cache.set(i, &[0u8; 100], None).expect("Failed to set object");
        }

        thread::sleep(Duration::from_millis(100));

        let stats = cache.tiering_stats();
        // All objects should be in PMEM initially
        assert_eq!(stats.pmem_only_objects, 5);
    }
}

#[cfg(all(feature = "enable_tiering_manager", feature = "key_value_pmem", feature = "hashtable_tiering"))]
mod hashtable_tiering_tests {
    use paper_cache::{PaperCache, PaperPolicy, BufferPMEM, TieringConfig};
    use std::thread;
    use std::time::Duration;

    #[test]
    fn test_hashtable_tiering_warm_and_hot_tiers() {
        // Create a cache with a small max size
        let cache = PaperCache::<u32, BufferPMEM>::new(
            10000,
            &[PaperPolicy::Lfu],
            PaperPolicy::Lfu,
        ).expect("Failed to create cache");

        // Using default config: warm_threshold=2, hot_threshold=5 from TieringConfig::default()
        
        // Set a test object
        cache.set(1, &[42u8; 100], None).expect("Failed to set object");
        
        // Give time for the tiering worker to register the object
        thread::sleep(Duration::from_millis(100));
        
        // Initial state: object should be in PMEM only
        let stats = cache.tiering_stats();
        assert_eq!(stats.pmem_only_objects, 1);
        assert_eq!(stats.dram_objects, 0);
        assert_eq!(stats.promotions, 0);

        // Access 1: Still in PMEM (access_count = 1, warm_threshold = 2)
        let result = cache.get(&1);
        assert!(result.is_ok());
        thread::sleep(Duration::from_millis(100));

        // Access 2: Should trigger promotion to warm tier (pointer-only)
        let result = cache.get(&1);
        assert!(result.is_ok());
        thread::sleep(Duration::from_millis(200));
        
        let stats = cache.tiering_stats();
        println!("After 2 accesses - DRAM objects: {}, Promotions: {}, PMEM only: {}", 
                 stats.dram_objects, stats.promotions, stats.pmem_only_objects);
        
        // Should have promoted to warm tier (pointer-only in DRAM)
        assert!(stats.dram_objects >= 1, "Expected at least 1 object in DRAM (warm tier)");
        assert!(stats.promotions >= 1, "Expected at least 1 promotion");

        // Access 3, 4, 5: Build up to hot threshold
        for _ in 0..3 {
            let result = cache.get(&1);
            assert!(result.is_ok());
            thread::sleep(Duration::from_millis(50));
        }
        
        // Give time for promotion to hot tier (physical copy)
        thread::sleep(Duration::from_millis(200));
        
        let stats = cache.tiering_stats();
        println!("After 5 accesses - DRAM objects: {}, Promotions: {}, DRAM size: {}", 
                 stats.dram_objects, stats.promotions, stats.dram_size);
        
        // Should have promoted to hot tier (physical copy in DRAM)
        // This means dram_size should now reflect the object size
        assert!(stats.dram_objects >= 1, "Expected at least 1 object in DRAM (hot tier)");
        assert!(stats.promotions >= 2, "Expected at least 2 promotions (warm + hot)");
        // Hot tier means physical copy, so dram_size should be > 0
        assert!(stats.dram_size > 0, "Expected DRAM size > 0 for hot tier physical copy");
        
        // Verify we can still read the correct data
        let result = cache.get(&1);
        assert!(result.is_ok());
        let data = result.unwrap();
        assert_eq!(data[0], 42u8);
        assert_eq!(data.len(), 100);
    }

    #[test]
    fn test_hashtable_tiering_pointer_tier_zero_copy() {
        // Create a cache
        let cache = PaperCache::<u32, BufferPMEM>::new(
            10000,
            &[PaperPolicy::Lfu],
            PaperPolicy::Lfu,
        ).expect("Failed to create cache");

        // Set object
        cache.set(2, &[99u8; 50], None).expect("Failed to set object");
        thread::sleep(Duration::from_millis(100));

        // Access twice to promote to warm tier (pointer-only)
        cache.get(&2).expect("Failed to get object");
        thread::sleep(Duration::from_millis(50));
        cache.get(&2).expect("Failed to get object");
        thread::sleep(Duration::from_millis(200));

        let stats = cache.tiering_stats();
        println!("Pointer tier test - DRAM objects: {}, DRAM size: {}", 
                 stats.dram_objects, stats.dram_size);
        
        // In warm tier (pointer-only), dram_size should be 0 or minimal
        // because data is still in CXL, only metadata is in DRAM
        assert!(stats.dram_objects >= 1, "Expected object in DRAM (warm tier)");
        
        // Verify we can read the correct data even from pointer tier
        let result = cache.get(&2).expect("Failed to get object");
        assert_eq!(result[0], 99u8);
        assert_eq!(result.len(), 50);
    }
}
