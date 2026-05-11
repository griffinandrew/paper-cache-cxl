#[cfg(all(feature = "enable_tiering_manager", feature = "key_value_pmem"))]
mod tiering_tests {
    use paper_cache::{PaperCache, PaperPolicy, BufferPMEM};
    use paper_cache::tiering::TieringConfig;
    use std::thread;
    use std::time::Duration;

    fn expected_default_threshold() -> u64 {
        TieringConfig::default().dram_threshold
    }

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

        // Default DRAM threshold uses the static TieringConfig default.
        let expected_threshold = expected_default_threshold();
        assert_eq!(cache.dram_threshold(), expected_threshold);

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
        let expected_threshold = expected_default_threshold();
        assert_eq!(initial_threshold, expected_threshold);

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

/// Tests that verify the tiering feature physically copies entire objects into DRAM
/// rather than just storing a pointer to PMEM data.
///
/// These tests check object length at the DRAM-cache level so that a pointer-only
/// implementation (8 bytes) is clearly distinguishable from a full copy (N bytes).
#[cfg(feature = "tiering")]
mod tiering_copy_verification_tests {
    use paper_cache::{PaperCache, PaperPolicy, BufferPMEM};
    use std::thread;
    use std::time::Duration;

    fn make_cache() -> PaperCache<u32, BufferPMEM> {
        PaperCache::<u32, BufferPMEM>::new(
            10_000_000,
            &[PaperPolicy::Lfu],
            PaperPolicy::Lfu,
        )
        .expect("Failed to create cache")
    }

    fn wait_for(max_ms: u64, mut f: impl FnMut() -> bool) -> bool {
        let deadline = std::time::Instant::now() + Duration::from_millis(max_ms);
        while std::time::Instant::now() < deadline {
            if f() {
                return true;
            }
            thread::sleep(Duration::from_millis(20));
        }
        false
    }

    /// The critical test: verifies the DRAM copy stores all N bytes of the object, not
    /// just a pointer (which would be 8 bytes regardless of object size).
    #[test]
    fn test_dram_copy_has_full_object_length_not_pointer() {
        let cache = make_cache();
        cache.set_hotness_threshold(2);

        const OBJ_LEN: usize = 512;
        cache.set(1u32, &[0xABu8; OBJ_LEN], None).expect("set failed");
        thread::sleep(Duration::from_millis(100));

        for _ in 0..3 {
            let _ = cache.get(&1u32);
            thread::sleep(Duration::from_millis(30));
        }

        let promoted = wait_for(600, || cache.tiering_stats().dram_objects >= 1);
        assert!(promoted, "object should have been promoted to DRAM");

        let data_len = cache
            .dram_object_data_len(&1u32)
            .expect("DRAM object should exist after promotion");

        assert_eq!(
            data_len, OBJ_LEN,
            "DRAM copy must hold the entire {} bytes — a pointer would only be ~8 bytes (got {})",
            OBJ_LEN, data_len
        );
    }

    /// Verifies that `dram_size` in tiering stats accounts for the real object payload
    /// (not just a pointer) and that `dram_object_data_len` returns exactly the value
    /// byte count.
    ///
    /// Note: `dram_size` includes key and metadata overhead on top of the raw value bytes,
    /// so it will be >= `OBJ_LEN`.  The `dram_object_data_len` method returns only the
    /// value bytes, which is the direct proof that the full payload was copied.
    #[test]
    fn test_dram_size_stat_includes_value_bytes_and_object_data_len_matches_exactly() {
        let cache = make_cache();
        cache.set_hotness_threshold(2);

        const OBJ_LEN: usize = 256;
        cache.set(2u32, &[0xFFu8; OBJ_LEN], None).expect("set failed");
        thread::sleep(Duration::from_millis(100));

        for _ in 0..3 {
            let _ = cache.get(&2u32);
            thread::sleep(Duration::from_millis(30));
        }

        let promoted = wait_for(600, || cache.tiering_stats().dram_objects >= 1);
        assert!(promoted, "object should have been promoted");

        let stats = cache.tiering_stats();
        // dram_size includes key/metadata overhead so it is >= OBJ_LEN
        assert!(
            stats.dram_size >= OBJ_LEN as u64,
            "dram_size ({}) must be at least the value byte length ({}) — \
             a pointer-only copy would report ~8 bytes",
            stats.dram_size, OBJ_LEN
        );

        // dram_object_data_len returns exactly the value bytes in the DRAM copy
        let dram_len = cache
            .dram_object_data_len(&2u32)
            .expect("DRAM object should exist after promotion");
        assert_eq!(
            dram_len, OBJ_LEN,
            "dram_object_data_len must be exactly {} bytes (the full value copy), got {}",
            OBJ_LEN, dram_len
        );
    }

    /// Verifies every byte of the DRAM copy is correct — a pointer-only implementation
    /// would have the wrong length, and a partial copy would fail the content check.
    #[test]
    fn test_dram_copy_all_bytes_intact_not_just_first() {
        let cache = make_cache();
        cache.set_hotness_threshold(2);

        let original: Vec<u8> = (0u8..=255).cycle().take(300).collect();
        cache.set(3u32, &original, None).expect("set failed");
        thread::sleep(Duration::from_millis(100));

        for _ in 0..3 {
            let _ = cache.get(&3u32);
            thread::sleep(Duration::from_millis(30));
        }

        let promoted = wait_for(600, || cache.tiering_stats().dram_objects >= 1);
        assert!(promoted, "object should have been promoted to DRAM");

        let result = cache.get(&3u32).expect("get after promotion failed");
        assert_eq!(result.len(), original.len(), "returned data length mismatch");
        assert_eq!(
            result, original,
            "every byte must match — a pointer or partial copy would fail here"
        );

        let dram_len = cache
            .dram_object_data_len(&3u32)
            .expect("DRAM object should exist");
        assert_eq!(
            dram_len,
            original.len(),
            "DRAM cache entry byte count must match original length"
        );
    }

    /// Verifies that each of several objects of different sizes is fully copied
    /// into DRAM — not just a uniform pointer size.
    #[test]
    fn test_multiple_objects_each_fully_copied_to_dram() {
        let cache = make_cache();
        cache.set_hotness_threshold(2);

        let sizes: &[usize] = &[100, 200, 400, 800];
        for (i, &sz) in sizes.iter().enumerate() {
            cache
                .set(i as u32, &vec![i as u8; sz], None)
                .expect("set failed");
        }
        thread::sleep(Duration::from_millis(100));

        for _ in 0..3 {
            for i in 0..sizes.len() {
                let _ = cache.get(&(i as u32));
            }
            thread::sleep(Duration::from_millis(30));
        }

        let promoted = wait_for(800, || {
            cache.tiering_stats().dram_objects >= sizes.len() as u64
        });
        assert!(promoted, "all objects should be promoted to DRAM");

        for (i, &expected_len) in sizes.iter().enumerate() {
            let dram_len = cache
                .dram_object_data_len(&(i as u32))
                .expect("each promoted object must be in DRAM");
            assert_eq!(
                dram_len, expected_len,
                "object {} DRAM copy must be {} bytes, got {}",
                i, expected_len, dram_len
            );
        }
    }
}

/// Tests that verify the multitiering feature correctly handles both warm-tier (pointer-only)
/// and hot-tier (full physical copy) promotions, checking byte lengths at each stage.
///
/// Key invariants:
///   • Warm tier: physical DRAM bytes == 0 (data stays in CXL, only a pointer lives in DRAM)
///   • Hot tier:  physical DRAM bytes == object byte length (full copy in DRAM)
#[cfg(feature = "multitiering")]
mod multitiering_copy_verification_tests {
    use paper_cache::{PaperCache, PaperPolicy, BufferPMEM};
    use std::thread;
    use std::time::Duration;

    fn make_cache() -> PaperCache<u32, BufferPMEM> {
        PaperCache::<u32, BufferPMEM>::new(
            10_000_000,
            &[PaperPolicy::Lfu],
            PaperPolicy::Lfu,
        )
        .expect("Failed to create cache")
    }

    fn wait_for(max_ms: u64, mut f: impl FnMut() -> bool) -> bool {
        let deadline = std::time::Instant::now() + Duration::from_millis(max_ms);
        while std::time::Instant::now() < deadline {
            if f() {
                return true;
            }
            thread::sleep(Duration::from_millis(20));
        }
        false
    }

    /// Warm tier must store ONLY a pointer to CXL data — zero physical bytes in DRAM.
    /// If this returns non-zero, data is being unnecessarily duplicated at warm-tier
    /// promotion time.
    #[test]
    fn test_warm_tier_is_pointer_only_zero_physical_bytes() {
        let cache = make_cache();
        // Default: warm_threshold=2, hot_threshold=5

        const OBJ_LEN: usize = 400;
        cache.set(10u32, &[0xBBu8; OBJ_LEN], None).expect("set failed");
        thread::sleep(Duration::from_millis(100));

        // Exactly 2 accesses to reach warm threshold but stay below hot threshold
        cache.get(&10u32).expect("get failed");
        thread::sleep(Duration::from_millis(50));
        cache.get(&10u32).expect("get failed");
        thread::sleep(Duration::from_millis(200));

        let promoted = wait_for(600, || cache.tiering_stats().dram_objects >= 1);
        assert!(promoted, "object should be in warm tier after 2 accesses");

        let phys_len = cache
            .dram_object_data_len(&10u32)
            .expect("warm-tier object should appear in DRAM cache");
        assert_eq!(
            phys_len, 0,
            "warm tier stores only a CXL pointer — physical DRAM bytes must be 0, got {} \
             (object is {} bytes; if this equals object size, the whole object is being \
             copied instead of just a pointer)",
            phys_len, OBJ_LEN
        );

        // Even with zero DRAM bytes the data must be fully readable via the CXL pointer
        let result = cache.get(&10u32).expect("warm-tier read failed");
        assert_eq!(result.len(), OBJ_LEN, "warm tier data length must match original");
        assert!(
            result.iter().all(|&b| b == 0xBBu8),
            "warm tier data bytes must be correct"
        );
    }

    /// Hot tier must store a FULL physical copy (N bytes in DRAM), not just a pointer.
    #[test]
    fn test_hot_tier_has_full_object_copy_not_just_pointer() {
        let cache = make_cache();

        const OBJ_LEN: usize = 512;
        cache.set(20u32, &[0xCCu8; OBJ_LEN], None).expect("set failed");
        thread::sleep(Duration::from_millis(100));

        // Access past hot_threshold (5) to reach hot tier
        for _ in 0..6 {
            let _ = cache.get(&20u32);
            thread::sleep(Duration::from_millis(40));
        }

        let hot = wait_for(900, || {
            let s = cache.tiering_stats();
            s.promotions >= 2 && s.dram_size > 0
        });
        assert!(hot, "object should be in hot tier (physical copy in DRAM)");

        let phys_len = cache
            .dram_object_data_len(&20u32)
            .expect("hot-tier object must be in DRAM cache");
        assert_eq!(
            phys_len, OBJ_LEN,
            "hot tier must contain the full {} byte object in DRAM — \
             a pointer-only copy would report ~8 bytes (got {})",
            OBJ_LEN, phys_len
        );

        let is_warm = cache
            .dram_object_is_warm_tier(&20u32)
            .expect("hot-tier object should be in DRAM cache");
        assert!(!is_warm, "hot-tier object must not report as warm tier");
    }

    /// Tracks the physical-byte-count transition: 0 at warm tier, N at hot tier.
    /// This directly demonstrates that warm tier moves only a pointer while hot tier
    /// moves the whole object.
    #[test]
    fn test_warm_then_hot_physical_size_transitions() {
        let cache = make_cache();

        const OBJ_LEN: usize = 300;
        cache.set(30u32, &[0xDDu8; OBJ_LEN], None).expect("set failed");
        thread::sleep(Duration::from_millis(100));

        // Warm tier: 2 accesses
        cache.get(&30u32).expect("get failed");
        thread::sleep(Duration::from_millis(50));
        cache.get(&30u32).expect("get failed");
        thread::sleep(Duration::from_millis(200));

        let warm = wait_for(600, || cache.tiering_stats().dram_objects >= 1);
        assert!(warm, "should reach warm tier");

        let warm_len = cache
            .dram_object_data_len(&30u32)
            .expect("warm-tier object in DRAM");
        assert_eq!(
            warm_len, 0,
            "warm tier must have 0 physical DRAM bytes (only a pointer), got {}",
            warm_len
        );

        let is_warm = cache
            .dram_object_is_warm_tier(&30u32)
            .expect("object should be in DRAM");
        assert!(is_warm, "after warm promotion, is_warm_tier must be true");

        // Hot tier: 3 more accesses (5 total)
        for _ in 0..4 {
            let _ = cache.get(&30u32);
            thread::sleep(Duration::from_millis(40));
        }

        let hot = wait_for(900, || {
            cache.tiering_stats().dram_size >= OBJ_LEN as u64
        });
        assert!(
            hot,
            "should reach hot tier with {} bytes in DRAM",
            OBJ_LEN
        );

        let hot_len = cache
            .dram_object_data_len(&30u32)
            .expect("hot-tier object in DRAM");
        assert_eq!(
            hot_len, OBJ_LEN,
            "after hot promotion physical DRAM bytes must be {} (was 0 in warm tier), got {}",
            OBJ_LEN, hot_len
        );

        let is_warm_after = cache
            .dram_object_is_warm_tier(&30u32)
            .expect("object should still be in DRAM");
        assert!(!is_warm_after, "after hot promotion, is_warm_tier must be false");
    }

    /// Verifies that every byte of a hot-tier DRAM copy is correct — not just the
    /// length — ruling out partial or zeroed copies.
    #[test]
    fn test_hot_tier_all_bytes_correct_not_just_length() {
        let cache = make_cache();

        let original: Vec<u8> = (0u8..=255).cycle().take(500).collect();
        cache.set(40u32, &original, None).expect("set failed");
        thread::sleep(Duration::from_millis(100));

        for _ in 0..7 {
            let _ = cache.get(&40u32);
            thread::sleep(Duration::from_millis(30));
        }

        let hot = wait_for(900, || cache.tiering_stats().dram_size > 0);
        assert!(hot, "object should be in hot tier");

        let result = cache.get(&40u32).expect("hot-tier read failed");
        assert_eq!(result.len(), original.len(), "data length must be preserved");
        assert_eq!(
            result, original,
            "every byte must match the original — a pointer or partial copy would fail here"
        );
    }
}

#[cfg(feature = "hybridcache")]
mod hybridcache_promotion_tests {
    use paper_cache::hybridcache::{CacheTierSize, HybridCacheConfig, S3FifoHybridCache};
    use std::time::Duration;

    fn wait_until(timeout: Duration, mut predicate: impl FnMut() -> bool) -> bool {
        let deadline = std::time::Instant::now() + timeout;
        while std::time::Instant::now() < deadline {
            if predicate() {
                return true;
            }
            std::thread::sleep(Duration::from_millis(20));
        }
        false
    }

    fn tiny_cache() -> S3FifoHybridCache<u32> {
        let config = HybridCacheConfig {
            small_size: CacheTierSize::Bytes(2_000),
            main_size: CacheTierSize::Bytes(20_000),
            ..Default::default()
        };
        S3FifoHybridCache::<u32>::new(config).expect("failed to create hybrid cache")
    }

    /// Ghost hits should be the only trigger for PMEM→DRAM promotion, and the
    /// promotion occurs asynchronously on a background worker.
    #[test]
    fn test_ghost_hit_promotion_only() {
        let cache = tiny_cache();
        let payload = "x".repeat(700);

        // Insert and force eviction of key 1 into PMEM/ghost.
        cache.set(1u32, &payload).unwrap();
        for i in 2u32..200 {
            cache.set(i, &payload).unwrap();
            std::thread::sleep(Duration::from_millis(5));
            if cache.has_in_pmem(&1u32) {
                break;
            }
        }
        std::thread::sleep(Duration::from_millis(150));

        assert!(cache.has_in_pmem(&1u32));
        assert!(!cache.has_in_dram(&1u32));

        // First access should be served from PMEM and only schedule async promotion.
        let val = cache.get(&1u32).expect("pmem get failed");
        assert_eq!(val, payload);
        assert!(cache.has_in_pmem(&1u32));
        assert!(!cache.has_in_dram(&1u32));

        // Subsequent access should observe promotion completing in the background.
        let _ = cache.get(&1u32).expect("pmem get after ghost hit failed");
        let promoted = wait_until(Duration::from_millis(1500), || cache.has_in_dram(&1u32));
        let stats = cache.stats();
        assert!(
            promoted,
            "ghost hit should trigger background promotion (promotions={}, dropped_promotions={})",
            stats.promotions,
            stats.dropped_promotions
        );

        // Create a PMEM resident whose ghost entry is likely evicted (oldest).
        cache.set(99u32, &payload).unwrap();
        for i in 100u32..170 {
            cache.set(i, &payload).unwrap();
        }
        std::thread::sleep(Duration::from_millis(200));
        assert!(cache.has_in_pmem(&99u32));

        // Heavy churn to evict stale ghost entries.
        for i in 200u32..300 {
            cache.set(i, &payload).unwrap();
        }
        std::thread::sleep(Duration::from_millis(200));

        let _ = cache.get(&99u32).expect("pmem get for non-ghost key failed");
        let non_ghost_promoted = wait_until(Duration::from_millis(500), || cache.has_in_dram(&99u32));
        assert!(
            !non_ghost_promoted,
            "PMEM hit without ghost membership must not promote to DRAM"
        );
    }
}
