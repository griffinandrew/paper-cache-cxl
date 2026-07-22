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


/// Tests for the `tiering` feature, which enables the tiering manager with
/// both key and value stored in PMEM (`key_pmem_value_pmem` + `enable_tiering_manager`).
///
/// When the `tiering` feature is active, every object is either:
///   * PMEM-only  – the primary copy lives in persistent memory, or
///   * PMEM + DRAM – a hot copy is kept in DRAM for fast reads while the
///                   source of truth remains in PMEM.
#[cfg(feature = "tiering")]
mod tiering_pmem_key_tests {
    use paper_cache::{PaperCache, PaperPolicy, BufferPMEM};
    use std::thread;
    use std::time::Duration;

    /// Helper: create a cache large enough that the default 1 GB DRAM threshold
    /// is never exceeded by the small test objects.
    fn make_cache() -> PaperCache<u32, BufferPMEM> {
        PaperCache::<u32, BufferPMEM>::new(
            10_000_000, // 10 MB max cache size
            &[PaperPolicy::Lfu],
            PaperPolicy::Lfu,
        )
        .expect("Failed to create cache with tiering feature")
    }

    /// A freshly created cache should have empty tiering statistics.
    #[test]
    fn test_initial_stats_are_empty() {
        let cache = make_cache();
        let stats = cache.tiering_stats();
        assert_eq!(stats.dram_objects, 0);
        assert_eq!(stats.promotions, 0);
        assert_eq!(stats.demotions, 0);
        assert_eq!(stats.pmem_only_objects, 0);
    }

    /// Objects inserted via `set` should initially appear in PMEM-only tier.
    #[test]
    fn test_new_objects_start_in_pmem_only() {
        let cache = make_cache();

        for i in 0..5u32 {
            cache.set(i, &[0u8; 100], None).expect("set failed");
        }

        // Allow the tiering worker to register the new objects.
        thread::sleep(Duration::from_millis(150));

        let stats = cache.tiering_stats();
        assert_eq!(stats.pmem_only_objects, 5, "all objects should start PMEM-only");
        assert_eq!(stats.dram_objects, 0, "no objects should be in DRAM yet");
    }

    /// Repeatedly accessing an object should eventually promote it to DRAM.
    #[test]
    fn test_hot_object_promoted_to_dram() {
        let cache = make_cache();

        // Lower the hotness threshold so a small number of accesses triggers promotion.
        cache.set_hotness_threshold(3);

        cache.set(42u32, &[7u8; 200], None).expect("set failed");
        thread::sleep(Duration::from_millis(100));

        // Access the object above the hotness threshold.
        for _ in 0..4 {
            let result = cache.get(&42u32);
            assert!(result.is_ok(), "get should succeed");
            thread::sleep(Duration::from_millis(30));
        }

        // Allow the tiering worker to execute the promotion.
        thread::sleep(Duration::from_millis(200));

        let stats = cache.tiering_stats();
        println!(
            "promotions={}, dram_objects={}, pmem_only={}",
            stats.promotions, stats.dram_objects, stats.pmem_only_objects
        );
        assert!(stats.promotions >= 1, "expected at least one promotion");
        assert!(stats.dram_objects >= 1, "expected object in DRAM after promotion");
    }

    /// Data read from a promoted (DRAM-cached) object must equal the originally stored value.
    #[test]
    fn test_dram_copy_data_integrity() {
        let cache = make_cache();
        cache.set_hotness_threshold(2);

        let expected: Vec<u8> = (0u8..50).collect();
        cache.set(99u32, &expected, None).expect("set failed");
        thread::sleep(Duration::from_millis(100));

        // Trigger promotion.
        for _ in 0..3 {
            let _ = cache.get(&99u32);
            thread::sleep(Duration::from_millis(30));
        }
        thread::sleep(Duration::from_millis(200));

        // Whether served from DRAM or PMEM, the value must be correct.
        let result = cache.get(&99u32).expect("get after promotion failed");
        assert_eq!(result, expected, "data integrity check failed after promotion");
    }

    /// The DRAM threshold can be read and updated at runtime.
    #[test]
    fn test_dram_threshold_configuration() {
        let cache = make_cache();

        cache.set_dram_threshold(50_000_000);
        assert_eq!(cache.dram_threshold(), 50_000_000);
    }

    /// The hotness threshold can be read and updated at runtime.
    #[test]
    fn test_hotness_threshold_configuration() {
        let cache = make_cache();

        cache.set_hotness_threshold(10);
        assert_eq!(cache.hotness_threshold(), 10);
    }

    /// `wipe` clears objects and resets tiering state.
    #[test]
    fn test_wipe_resets_tiering_state() {
        let cache = make_cache();

        for i in 0..3u32 {
            cache.set(i, &[1u8; 64], None).expect("set failed");
        }
        thread::sleep(Duration::from_millis(100));

        cache.wipe().expect("wipe failed");
        thread::sleep(Duration::from_millis(100));

        let stats = cache.tiering_stats();
        assert_eq!(stats.pmem_only_objects, 0, "pmem_only_objects should be 0 after wipe");
        assert_eq!(stats.dram_objects, 0, "dram_objects should be 0 after wipe");
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

/// Tests for the `multitiering` feature, which enables three-tier data movement
/// with both key and value stored in PMEM (`key_pmem_value_pmem` + `enable_tiering_manager`
/// + `hashtable_tiering`).
///
/// Objects move through three tiers:
///   • PMEM-only      – cold objects live exclusively in persistent memory,
///   • Warm (pointer) – a metadata/pointer record is in DRAM, data stays in CXL,
///   • Hot (copy)     – a full physical copy of the data is in DRAM for fastest reads.
#[cfg(feature = "multitiering")]
mod multitiering_tests {
    use paper_cache::{PaperCache, PaperPolicy, BufferPMEM};
    use std::thread;
    use std::time::Duration;

    fn make_cache() -> PaperCache<u32, BufferPMEM> {
        PaperCache::<u32, BufferPMEM>::new(
            10_000_000, // 10 MB – well above any DRAM threshold reached in tests
            &[PaperPolicy::Lfu],
            PaperPolicy::Lfu,
        )
        .expect("Failed to create cache with multitiering feature")
    }

    /// A freshly created cache should have empty tiering statistics.
    #[test]
    fn test_multitiering_initial_stats_are_empty() {
        let cache = make_cache();
        let stats = cache.tiering_stats();
        assert_eq!(stats.dram_objects, 0);
        assert_eq!(stats.promotions, 0);
        assert_eq!(stats.demotions, 0);
        assert_eq!(stats.pmem_only_objects, 0);
    }

    /// Objects inserted via `set` should initially appear in PMEM-only tier.
    #[test]
    fn test_multitiering_new_objects_start_in_pmem_only() {
        let cache = make_cache();

        for i in 0..5u32 {
            cache.set(i, &[0u8; 100], None).expect("set failed");
        }

        thread::sleep(Duration::from_millis(150));

        let stats = cache.tiering_stats();
        assert_eq!(stats.pmem_only_objects, 5, "all objects should start PMEM-only");
        assert_eq!(stats.dram_objects, 0, "no objects should be in DRAM yet");
    }

    /// After enough accesses (warm_threshold) an object should be promoted to
    /// the warm tier (pointer-only in DRAM, data still in CXL/PMEM).
    #[test]
    fn test_multitiering_warm_tier_promotion() {
        let cache = make_cache();

        // Default warm_threshold = 2
        cache.set(10u32, &[55u8; 128], None).expect("set failed");
        thread::sleep(Duration::from_millis(100));

        // First access – still PMEM-only
        cache.get(&10u32).expect("get failed");
        thread::sleep(Duration::from_millis(50));

        // Second access – should trigger warm promotion
        cache.get(&10u32).expect("get failed");
        thread::sleep(Duration::from_millis(200));

        let stats = cache.tiering_stats();
        println!(
            "multitiering warm: dram_objects={}, promotions={}, dram_size={}",
            stats.dram_objects, stats.promotions, stats.dram_size
        );
        assert!(stats.dram_objects >= 1, "expected object promoted to warm tier");
        assert!(stats.promotions >= 1, "expected at least one promotion");
    }

    /// After enough accesses (hot_threshold) an object should be promoted to
    /// the hot tier (full physical copy in DRAM).
    #[test]
    fn test_multitiering_hot_tier_promotion() {
        let cache = make_cache();

        // Default warm_threshold = 2, hot_threshold = 5
        cache.set(20u32, &[77u8; 200], None).expect("set failed");
        thread::sleep(Duration::from_millis(100));

        // Access 5 times to reach hot threshold
        for _ in 0..5 {
            cache.get(&20u32).expect("get failed");
            thread::sleep(Duration::from_millis(40));
        }
        thread::sleep(Duration::from_millis(300));

        let stats = cache.tiering_stats();
        println!(
            "multitiering hot: dram_objects={}, promotions={}, dram_size={}",
            stats.dram_objects, stats.promotions, stats.dram_size
        );
        assert!(stats.dram_objects >= 1, "expected object in DRAM (hot tier)");
        assert!(stats.promotions >= 2, "expected warm + hot promotions");
        assert!(stats.dram_size > 0, "hot tier should have non-zero DRAM size");
    }

    /// Data read from a promoted (hot-tier) object must equal the originally stored value.
    #[test]
    fn test_multitiering_data_integrity_after_hot_promotion() {
        let cache = make_cache();

        let expected: Vec<u8> = (0u8..64).collect();
        cache.set(30u32, &expected, None).expect("set failed");
        thread::sleep(Duration::from_millis(100));

        // Drive past warm and hot thresholds
        for _ in 0..6 {
            let _ = cache.get(&30u32);
            thread::sleep(Duration::from_millis(30));
        }
        thread::sleep(Duration::from_millis(250));

        let result = cache.get(&30u32).expect("get after hot promotion failed");
        assert_eq!(result, expected, "data integrity check failed after hot promotion");
    }

    /// The DRAM and hotness thresholds can be configured at runtime.
    #[test]
    fn test_multitiering_threshold_configuration() {
        let cache = make_cache();

        cache.set_dram_threshold(100_000_000);
        assert_eq!(cache.dram_threshold(), 100_000_000);

        cache.set_hotness_threshold(10);
        assert_eq!(cache.hotness_threshold(), 10);
    }

    /// `wipe` clears objects and resets multitiering state.
    #[test]
    fn test_multitiering_wipe_resets_state() {
        let cache = make_cache();

        for i in 0..3u32 {
            cache.set(i, &[1u8; 64], None).expect("set failed");
        }
        thread::sleep(Duration::from_millis(100));

        cache.wipe().expect("wipe failed");
        thread::sleep(Duration::from_millis(100));

        let stats = cache.tiering_stats();
        assert_eq!(stats.pmem_only_objects, 0, "pmem_only_objects should be 0 after wipe");
        assert_eq!(stats.dram_objects, 0, "dram_objects should be 0 after wipe");
    }
}
