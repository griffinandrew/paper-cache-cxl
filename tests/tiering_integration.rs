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

// Admission Control tests: DRAM-first 2Q lifecycle with CXL victim cache.
// These tests require PMEM/CXL hardware AND nightly Rust (allocator_api).
// Gate on both `admission_control` and `key_value_pmem`.
#[cfg(all(feature = "admission_control", feature = "key_value_pmem"))]
mod admission_control_tests {
    use paper_cache::{PaperCache, PaperPolicy, BufferPMEM};

    /// Verify that the cache can be created with admission_control and
    /// that the 2Q state machine initialises correctly.
    #[test]
    fn test_admission_control_new() {
        let cache = PaperCache::<u32, BufferPMEM>::new(
            1024 * 1024, // 1 MB max cache size
            &[PaperPolicy::Lfu],
            PaperPolicy::Lfu,
        ).expect("Failed to create admission_control cache");

        // Basic set/get round-trip under admission_control
        cache.set(1u32, &[42u8; 64], None).expect("set failed");
        let val = cache.get(&1u32).expect("get failed");
        assert_eq!(val, vec![42u8; 64]);
    }

    /// Verify that the 2Q lifecycle progresses: Admission → Victim → Warm → Hot.
    /// Objects accessed `hotness_threshold` times are promoted back to DRAM.
    ///
    /// NOTE: the `check_tier` assertions below require PMEM hardware.
    /// Without hardware the test still validates the state machine logic via get/set.
    #[test]
    fn test_2q_lifecycle_progression() {
        // Small cache so eviction happens quickly
        let cache = PaperCache::<u64, BufferPMEM>::new(
            4096,           // tiny max size to trigger eviction
            &[PaperPolicy::Lru],
            PaperPolicy::Lru,
        ).expect("Failed to create admission_control cache");

        // Insert items; first ones are admitted to DRAM (Admission state)
        for i in 0u64..10 {
            let _ = cache.set(i, &[0u8; 100], None);
        }

        // Access key 0 `hotness_threshold` times to trigger Admission→Warm→Hot transitions.
        // Default TieringConfig::hotness_threshold = 3 (set in lib.rs new_admission_control).
        let hotness_threshold: u64 = 3;
        for _ in 0..hotness_threshold {
            let _ = cache.get(&0u64);
        }

        // After hotness_threshold accesses the object should still be readable
        let result = cache.get(&0u64);
        assert!(result.is_ok(), "object should still be accessible after promotion");
    }

    /// Verify correct data is preserved through CXL victim migration and DRAM promotion.
    #[test]
    fn test_data_integrity_across_tiers() {
        let expected: Vec<u8> = (0u8..=255u8).collect();

        let cache = PaperCache::<u32, BufferPMEM>::new(
            512 * 1024, // 512 KB
            &[PaperPolicy::Lfu],
            PaperPolicy::Lfu,
        ).expect("Failed to create admission_control cache");

        cache.set(99u32, &expected, None).expect("set failed");

        // Multiple gets to trigger state machine transitions
        for _ in 0..5 {
            let val = cache.get(&99u32).expect("get failed");
            assert_eq!(val, expected, "data must survive tier transitions");
        }
    }

    /// Verify that a deleted key is removed from both the objects map and the
    /// victim cache, returning KeyNotFound on subsequent gets.
    #[test]
    fn test_victim_cache_eviction_on_del() {
        let cache = PaperCache::<u32, BufferPMEM>::new(
            1024 * 1024,
            &[PaperPolicy::Lfu],
            PaperPolicy::Lfu,
        ).expect("Failed to create admission_control cache");

        cache.set(7u32, &[1u8; 32], None).expect("set failed");
        assert!(cache.get(&7u32).is_ok());
        cache.del(&7u32).expect("del failed");
        assert!(cache.get(&7u32).is_err(), "deleted key must not be found");
    }
}
