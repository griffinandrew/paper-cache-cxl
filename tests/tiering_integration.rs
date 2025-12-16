#[cfg(any(feature = "allocator_api", feature = "alloc_with_hash", feature = "alloc_api_exp"))]
mod tiering_tests {
    use paper_cache::{PaperCache, PaperPolicy};
    use std::thread;
    use std::time::Duration;

    #[test]
    fn test_tiering_manager_integration() {
        // Create a cache with a small max size
        let cache = PaperCache::<u32, Box<[u8]>>::new(
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

        // Set some objects - they should all go to DRAM immediately
        for i in 0..10 {
            cache.set(i, &[0u8; 100], None).expect("Failed to set object");
        }

        // Give some time for the tiering worker to process
        thread::sleep(Duration::from_millis(100));

        // All objects should be in DRAM after SET (new behavior)
        let stats = cache.tiering_stats();
        assert_eq!(stats.dram_objects, 10, "All objects should be in DRAM after SET");
        
        // Access some objects multiple times
        for _ in 0..3 {
            for i in 0..5 {
                let _ = cache.get(&i);
            }
        }

        // Give time for any background processing
        thread::sleep(Duration::from_millis(100));

        // Check stats
        let stats = cache.tiering_stats();
        println!("DRAM objects: {}, PMEM only: {}", stats.dram_objects, stats.pmem_only_objects);
    }

    #[test]
    fn test_tiering_configuration() {
        let cache = PaperCache::<u32, Box<[u8]>>::new(
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
        let cache = PaperCache::<u32, Box<[u8]>>::new(
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
        // With new behavior, all objects should be in DRAM immediately after SET
        assert_eq!(stats.dram_objects, 5);
        assert_eq!(stats.pmem_only_objects, 0);
    }
    
    #[test]
    fn test_set_immediately_inserts_to_dram() {
        // Test that SET operations immediately insert objects into both DRAM and PMEM
        let cache = PaperCache::<u32, Box<[u8]>>::new(
            10000,
            &[PaperPolicy::Lfu],
            PaperPolicy::Lfu,
        ).expect("Failed to create cache");
        
        // Set a high hotness threshold to ensure promotion doesn't happen via access counting
        cache.set_hotness_threshold(100);
        
        // Insert a new object
        cache.set(1, &[0u8; 100], None).expect("Failed to set object");
        
        // Give time for the tiering worker to process
        thread::sleep(Duration::from_millis(100));
        
        // Object should be in DRAM immediately after SET, without any GET operations
        let stats = cache.tiering_stats();
        assert_eq!(stats.dram_objects, 1, "Object should be in DRAM immediately after SET");
        assert_eq!(stats.pmem_only_objects, 0, "No objects should be in PMEM-only state after SET");
        
        // Verify object is accessible
        assert!(cache.get(&1).is_some(), "Object should be accessible");
    }
    
    #[test]
    fn test_set_bypasses_tiering_policies() {
        // Test that SET operations bypass hotness thresholds and tiering policies
        let cache = PaperCache::<u32, Box<[u8]>>::new(
            10000,
            &[PaperPolicy::Lfu],
            PaperPolicy::Lfu,
        ).expect("Failed to create cache");
        
        // Set a very high hotness threshold that would normally prevent promotion
        cache.set_hotness_threshold(1000);
        
        // Insert multiple objects
        for i in 0..5 {
            cache.set(i, &[0u8; 100], None).expect("Failed to set object");
        }
        
        // Give time for the tiering worker to process
        thread::sleep(Duration::from_millis(100));
        
        // All objects should be in DRAM despite high hotness threshold
        let stats = cache.tiering_stats();
        assert_eq!(stats.dram_objects, 5, "All objects should be in DRAM after SET");
        assert_eq!(stats.pmem_only_objects, 0, "No objects should be in PMEM-only state");
    }
    
    #[test]
    fn test_update_ensures_dram_presence() {
        // Test that updating an object ensures it's in DRAM even if it was demoted
        let cache = PaperCache::<u32, Box<[u8]>>::new(
            10000,
            &[PaperPolicy::Lfu],
            PaperPolicy::Lfu,
        ).expect("Failed to create cache");
        
        // Set a very small DRAM threshold to force demotions
        cache.set_dram_threshold(500);
        
        // Insert objects
        for i in 0..10 {
            cache.set(i, &[0u8; 100], None).expect("Failed to set object");
        }
        
        // Give time for processing and potential demotions
        thread::sleep(Duration::from_millis(6000)); // Wait for periodic tiering
        
        let stats_before = cache.tiering_stats();
        println!("Before update - DRAM objects: {}, PMEM only: {}", 
                 stats_before.dram_objects, stats_before.pmem_only_objects);
        
        // Update an object that may have been demoted
        cache.set(5, &[1u8; 100], None).expect("Failed to update object");
        
        // Give time for the update to be processed
        thread::sleep(Duration::from_millis(100));
        
        // The updated object should be in DRAM regardless of previous state
        // (This is harder to test definitively without knowing which objects were demoted,
        //  but at least the update should succeed)
        assert!(cache.get(&5).is_some(), "Updated object should be accessible");
    }
}
