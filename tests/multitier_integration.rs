#[cfg(feature = "multitiering")]
mod multitier_tests {
    use paper_cache::{
        MultitieringManager,
        MultitieringConfig,
        MultitieringTier as Tier,
    };
    use std::time::Duration;

    fn make_manager(warm_cap: u64, hot_cap: u64) -> MultitieringManager<u64, Vec<u8>> {
        MultitieringManager::new(MultitieringConfig {
            warm_capacity_bytes: warm_cap,
            hot_capacity_bytes: hot_cap,
            warm_threshold: 1,
            hot_threshold: 2,
            evaluation_interval: Duration::from_secs(5),
            warm_high_water_mark: 0.9,
            warm_low_water_mark: 0.7,
            hot_high_water_mark: 0.9,
            hot_low_water_mark: 0.7,
        })
    }

    #[test]
    fn test_multitier_register_and_tier() {
        let mgr = make_manager(1000, 1000);
        mgr.register_object(1, 100);
        assert_eq!(mgr.tier_of(&1), Some(Tier::PmemOnly));
    }

    #[test]
    fn test_multitier_promote_to_warm() {
        let mgr = make_manager(1000, 1000);
        mgr.register_object(1, 100);
        assert!(mgr.promote_to_warm(1));
        let stats = mgr.stats();
        assert_eq!(stats.warm_size, 100);
        assert_eq!(stats.warm_objects, 1);
        assert_eq!(stats.cold_objects, 0);
    }

    #[test]
    fn test_multitier_promote_to_hot() {
        let mgr = make_manager(1000, 1000);
        mgr.register_object(1, 100);
        mgr.promote_to_warm(1);
        // Use the metadata-only promotion helper so that this test does not need a real
        // Object<K, V> value.
        assert!(mgr.promote_to_hot_no_data(1));
        let stats = mgr.stats();
        assert_eq!(stats.hot_size, 100);
        assert_eq!(stats.hot_objects, 1);
        assert_eq!(stats.warm_size, 0);
        assert_eq!(stats.warm_objects, 0);
    }

    #[test]
    fn test_multitier_remove_object() {
        let mgr = make_manager(1000, 1000);
        mgr.register_object(1, 100);
        mgr.promote_to_warm(1);
        mgr.remove_object(1);
        assert_eq!(mgr.tier_of(&1), None);
        let stats = mgr.stats();
        assert_eq!(stats.warm_objects, 0);
        assert_eq!(stats.warm_size, 0);
    }

    #[test]
    fn test_multitier_clear() {
        let mgr = make_manager(1000, 1000);
        for i in 1_u64..=5 {
            mgr.register_object(i, 100);
            mgr.promote_to_warm(i);
        }
        mgr.clear();
        let stats = mgr.stats();
        assert_eq!(stats.warm_size, 0);
        assert_eq!(stats.hot_size, 0);
        assert_eq!(stats.cold_objects, 0);
        assert_eq!(stats.warm_objects, 0);
        assert_eq!(stats.hot_objects, 0);
    }

    #[test]
    fn test_multitiering_byte_threshold_evictions() {
        // warm_capacity = 250 bytes, hot_capacity = 150 bytes
        // high_water_mark: 1.0  → trigger demotion when usage EXCEEDS capacity
        // low_water_mark:  0.8  → stop when usage drops to 80% of capacity
        //   warm low_water = 250 * 0.8 = 200  → demote 1 object (300→200)
        //   hot  low_water = 150 * 0.8 = 120  → demote 1 object (200→100)
        let mgr: MultitieringManager<u64, Vec<u8>> = MultitieringManager::new(MultitieringConfig {
            warm_capacity_bytes: 250,
            hot_capacity_bytes: 150,
            warm_threshold: 1,
            hot_threshold: 2,
            evaluation_interval: Duration::from_secs(5),
            warm_high_water_mark: 1.0,
            warm_low_water_mark: 0.8,
            hot_high_water_mark: 1.0,
            hot_low_water_mark: 0.8,
        });

        // Phase 1: promote 3 objects (100 bytes each) to warm.
        // After 3 × 100 = 300 bytes the high_water (250 bytes) is exceeded.
        for key in [1_u64, 2, 3] {
            mgr.register_object(key, 100);
            assert!(mgr.promote_to_warm(key));
            std::thread::sleep(std::time::Duration::from_millis(5));
        }

        // Run a manual demotion sweep (simulates the periodic worker tick).
        for (key, _) in mgr.get_keys_to_demote() {
            mgr.demote_to_cold(key);
        }

        // Oldest object (1) must be demoted back to Cold. Objects 2 & 3 remain Warm.
        assert_eq!(mgr.tier_of(&1), Some(Tier::PmemOnly));
        assert_eq!(mgr.tier_of(&2), Some(Tier::DramPtrToPmem));
        assert_eq!(mgr.tier_of(&3), Some(Tier::DramPtrToPmem));

        let stats = mgr.stats();
        assert_eq!(stats.warm_size, 200);
        assert_eq!(stats.warm_objects, 2);
        assert_eq!(stats.cold_objects, 1);
        assert_eq!(stats.demotions, 1);

        // Phase 2: promote objects 2 and 3 to hot tier (limit = 150 bytes).
        assert!(mgr.promote_to_hot_no_data(2));
        std::thread::sleep(std::time::Duration::from_millis(5));
        assert!(mgr.promote_to_hot_no_data(3));

        // Run another demotion sweep for the hot tier.
        // hot_size = 200 > high_water (150). low_water = 120. Demote object 2 (oldest):
        // 200 - 100 = 100 ≤ 120 → stop. Only object 2 is demoted.
        for (key, _) in mgr.get_keys_to_demote() {
            mgr.demote_to_cold(key);
        }

        // Oldest hot object (2) must be demoted directly to Cold — NOT to Warm.
        assert_eq!(mgr.tier_of(&2), Some(Tier::PmemOnly));
        assert_eq!(mgr.tier_of(&3), Some(Tier::DramAndPmem));

        let stats = mgr.stats();
        assert_eq!(stats.hot_size, 100);
        assert_eq!(stats.hot_objects, 1);
        assert_eq!(stats.warm_size, 0);
        assert_eq!(stats.warm_objects, 0);
        assert_eq!(stats.demotions, 2);

        // Phase 3: verify counters after further insertions
        mgr.register_object(4, 50);
        mgr.promote_to_warm(4);
        let stats = mgr.stats();
        assert_eq!(stats.warm_size, 50);
        assert_eq!(stats.warm_objects, 1);
        assert_eq!(stats.hot_size, 100);
        assert_eq!(stats.hot_objects, 1);
    }

    #[test]
    fn test_watermark_based_demotion() {
        // hot_capacity = 300, high_water = 0.9 (270), low_water = 0.7 (210)
        let mgr: MultitieringManager<u64, Vec<u8>> = MultitieringManager::new(MultitieringConfig {
            warm_capacity_bytes: 10_000,
            hot_capacity_bytes: 300,
            warm_threshold: 1,
            hot_threshold: 2,
            evaluation_interval: Duration::from_secs(5),
            warm_high_water_mark: 0.9,
            warm_low_water_mark: 0.7,
            hot_high_water_mark: 0.9,
            hot_low_water_mark: 0.7,
        });

        // Promote 3 objects (each 100 bytes) into the hot tier using promote_to_hot_no_data
        // so there's no hot-tier LRU eviction here (total = 300 bytes < capacity 300)
        for key in [1_u64, 2, 3] {
            mgr.register_object(key, 100);
            mgr.promote_to_warm(key);
            mgr.promote_to_hot_no_data(key);
            std::thread::sleep(std::time::Duration::from_millis(5));
        }

        // hot_size = 300, high_water = 270 → should demote
        let to_demote = mgr.get_keys_to_demote();
        // We must demote until hot_size <= 210 (low_water)
        // Each object = 100 bytes, so we need to demote at least 1 object (300 - 100 = 200 ≤ 210)
        assert!(!to_demote.is_empty(), "should have objects to demote");

        for (key, _) in &to_demote {
            mgr.demote_to_cold(*key);
        }

        let stats = mgr.stats();
        let low_water = (300_f64 * 0.7) as u64;
        assert!(stats.hot_size <= low_water, "hot_size {} should be <= low_water {}", stats.hot_size, low_water);
    }

    #[test]
    fn test_record_access_and_get_keys_to_promote() {
        let mgr: MultitieringManager<u64, Vec<u8>> = MultitieringManager::new(MultitieringConfig {
            warm_capacity_bytes: 10_000,
            hot_capacity_bytes: 10_000,
            warm_threshold: 2,
            hot_threshold: 4,
            evaluation_interval: Duration::from_secs(5),
            warm_high_water_mark: 0.9,
            warm_low_water_mark: 0.7,
            hot_high_water_mark: 0.9,
            hot_low_water_mark: 0.7,
        });

        mgr.register_object(1, 100);

        // First access — below warm_threshold of 2
        mgr.record_access(1);
        assert!(mgr.get_keys_to_promote().is_empty());

        // Second access — meets warm_threshold
        mgr.record_access(1);
        let to_promote = mgr.get_keys_to_promote();
        assert_eq!(to_promote.len(), 1);
        assert_eq!(to_promote[0], (1, Tier::DramPtrToPmem));

        mgr.promote_to_warm(1);

        // Two more accesses to meet hot_threshold (access_count resets to 0 on promotion)
        mgr.record_access(1);
        mgr.record_access(1);
        mgr.record_access(1);
        mgr.record_access(1);
        let to_promote = mgr.get_keys_to_promote();
        assert_eq!(to_promote.len(), 1);
        assert_eq!(to_promote[0], (1, Tier::DramAndPmem));
    }
}

