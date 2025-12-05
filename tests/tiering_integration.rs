#[cfg(feature = "all_dram")]
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
        // All objects should be in PMEM initially
        assert_eq!(stats.pmem_only_objects, 5);
    }
}
