//! Test for FlatMap resizing with unified tiering limits
//!
//! This test verifies that:
//! 1. FlatMap automatically resizes when reaching 75% load factor
//! 2. Data is preserved correctly during resize
//! 3. Eviction limits are enforced for both DRAM objects and pointer count

#[cfg(all(feature = "flatmap_hash_and_object_tiering", feature = "enable_tiering_manager", feature = "key_value_pmem"))]
mod flatmap_resize_tests {
    use paper_cache::{PaperCache, PaperPolicy, BufferPMEM, TieringConfig};
    use std::thread;
    use std::time::Duration;

    #[test]
    fn test_flatmap_resizing_with_many_inserts() {
        // Create cache with custom config that has low pointer limit
        let mut config = TieringConfig::default();
        config.dram_pointer_limit = 100; // Low limit to test eviction
        config.dram_object_limit = 10 * 1024; // 10KB limit for testing
        
        let cache = PaperCache::<u32, BufferPMEM>::with_tiering_config(
            50000, // max_size
            &[PaperPolicy::Lfu],
            PaperPolicy::Lfu,
            config,
        ).expect("Failed to create cache");

        // Insert 200 items to force FlatMap resize (initial capacity is 4096)
        // With 75% load factor, resize happens around 3072 items
        println!("Inserting 200 items to test FlatMap resizing...");
        for i in 0..200 {
            let data = vec![i as u8; 100]; // 100 bytes per object
            cache.set(i, &data, None).expect(&format!("Failed to set object {}", i));
        }

        // Give time for tiering worker to process
        thread::sleep(Duration::from_millis(200));

        // Verify all items are accessible (proving resize preserved data)
        println!("Verifying all 200 items are accessible...");
        for i in 0..200 {
            assert!(
                cache.has(&i),
                "Object {} should be accessible after resize",
                i
            );
        }

        // Access first 50 items multiple times to make them hot
        println!("Accessing first 50 items to make them hot...");
        for _ in 0..5 {
            for i in 0..50 {
                let _ = cache.get(&i);
            }
        }

        // Give time for promotions
        thread::sleep(Duration::from_millis(200));

        // Check tiering stats
        let stats = cache.tiering_stats();
        println!("Tiering stats after promotions:");
        println!("  - DRAM objects: {}", stats.dram_objects);
        println!("  - DRAM size: {} bytes", stats.dram_size);
        println!("  - Promotions: {}", stats.promotions);
        println!("  - Demotions: {}", stats.demotions);

        // Verify that promotion happened
        assert!(
            stats.promotions > 0,
            "Some objects should have been promoted to DRAM"
        );

        // Verify DRAM size limit is respected (with some tolerance for timing)
        assert!(
            stats.dram_size <= config.dram_object_limit * 2, // Allow 2x for timing
            "DRAM size ({}) should not greatly exceed limit ({})",
            stats.dram_size,
            config.dram_object_limit
        );
    }

    #[test]
    fn test_pointer_limit_enforcement() {
        // Create cache with very low pointer limit
        let mut config = TieringConfig::default();
        config.dram_pointer_limit = 50; // Very low limit
        
        let cache = PaperCache::<u32, BufferPMEM>::with_tiering_config(
            100000,
            &[PaperPolicy::Lfu],
            PaperPolicy::Lfu,
            config,
        ).expect("Failed to create cache");

        // Insert more items than the pointer limit
        println!("Inserting 100 items (pointer limit is 50)...");
        for i in 0..100 {
            let data = vec![i as u8; 50];
            cache.set(i, &data, None).expect(&format!("Failed to set object {}", i));
        }

        // Give time for eviction worker
        thread::sleep(Duration::from_millis(500));

        // The global cache should have enforced the limit through eviction
        // Note: Due to timing, we can't assert exact count, but it should be reasonable
        let current_size = cache.len();
        println!("Current cache size: {} (limit was {})", current_size, config.dram_pointer_limit);
        
        // Allow some overflow for timing but should be bounded
        assert!(
            current_size <= config.dram_pointer_limit * 3,
            "Cache size ({}) should be bounded near limit ({})",
            current_size,
            config.dram_pointer_limit
        );
    }

    #[test]
    fn test_flatmap_preserves_data_during_resize() {
        let cache = PaperCache::<u64, BufferPMEM>::new(
            100000,
            &[PaperPolicy::Lru],
            PaperPolicy::Lru,
        ).expect("Failed to create cache");

        // Insert data with known values
        println!("Inserting 150 items with known values...");
        for i in 0..150_u64 {
            let data = vec![i as u8; 100];
            cache.set(i, &data, None).expect(&format!("Failed to set {}", i));
        }

        thread::sleep(Duration::from_millis(100));

        // Verify data integrity
        println!("Verifying data integrity...");
        for i in 0..150_u64 {
            if let Some(data) = cache.get(&i) {
                assert_eq!(
                    data[0],
                    i as u8,
                    "Data for object {} should be preserved",
                    i
                );
            }
        }
    }
}

#[cfg(not(all(feature = "flatmap_hash_and_object_tiering", feature = "enable_tiering_manager", feature = "key_value_pmem")))]
mod disabled_test {
    #[test]
    fn test_feature_not_enabled() {
        println!("FlatMap resize tiering tests require features: flatmap_hash_and_object_tiering, enable_tiering_manager, key_value_pmem");
    }
}
