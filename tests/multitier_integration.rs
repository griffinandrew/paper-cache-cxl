#[cfg(feature = "multitiering")]
mod multitier_tests {
    use paper_cache::{
        MultitieringManager,
        MultitieringConfig,
        MultitieringTier as Tier,
    };

    fn make_manager(warm_cap: u64, hot_cap: u64) -> MultitieringManager<u64, Vec<u8>> {
        MultitieringManager::new(MultitieringConfig {
            warm_capacity_bytes: warm_cap,
            hot_capacity_bytes: hot_cap,
            warm_threshold: 1,
            hot_threshold: 2,
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
        assert!(mgr.promote_to_hot(1));
        let stats = mgr.stats();
        assert_eq!(stats.hot_size, 100);
        assert_eq!(stats.hot_objects, 1);
        assert_eq!(stats.warm_size, 0);
        assert_eq!(stats.warm_objects, 0);
    }

    #[test]
    fn test_multitiering_byte_threshold_evictions() {
        let mgr: MultitieringManager<u64, Vec<u8>> = MultitieringManager::new(MultitieringConfig {
            warm_capacity_bytes: 250,
            hot_capacity_bytes: 150,
            warm_threshold: 1,
            hot_threshold: 2,
        });

        // Phase 1: fill the Warm tier (250-byte limit, 100-byte objects)
        for key in [1_u64, 2, 3] {
            mgr.register_object(key, 100);
            assert!(mgr.promote_to_warm(key));
        }

        // object 1 (oldest) should be evicted back to Cold
        assert_eq!(mgr.tier_of(&1), Some(Tier::PmemOnly));
        assert_eq!(mgr.tier_of(&2), Some(Tier::DramPtrToPmem));
        assert_eq!(mgr.tier_of(&3), Some(Tier::DramPtrToPmem));

        let stats = mgr.stats();
        assert_eq!(stats.warm_size, 200);
        assert_eq!(stats.warm_objects, 2);
        assert_eq!(stats.cold_objects, 1);
        assert_eq!(stats.demotions, 1);

        // Phase 2: fill the Hot tier (150-byte limit)
        assert!(mgr.promote_to_hot(2));
        assert!(mgr.promote_to_hot(3));

        // object 2 (oldest hot) must go directly to Cold, not Warm
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
}
