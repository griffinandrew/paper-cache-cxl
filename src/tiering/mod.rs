//! Tiering Manager Module
//! 
//! This module provides functionality to manage objects between DRAM and PMEM tiers
//! with **actual physical data copying**.
//! 
//! # Overview
//! 
//! The tiering system implements a two-tier caching strategy with physical data copies:
//! - **Far Tier (PMEM)**: All objects are stored here (source of truth)
//! - **Near Tier (DRAM)**: Hot objects are **physically copied** here for faster access
//! 
//! # Architecture
//! 
//! The tiering manager integrates with the existing worker manager workflow:
//! - Receives `Get` and `Set` events from the worker manager
//! - Tracks object access patterns to determine hotness
//! - **Copies hot object data** to a separate DRAM cache
//! - Get operations check DRAM cache first, then fall back to PMEM
//! - Periodically evaluates objects for promotion or demotion
//! - Coordinates with the LFU eviction stack for runtime access metrics
//! 
//! # Data Copy Model
//! 
//! Unlike simple metadata tracking, this implementation maintains two physical copies:
//! - **PMEM Cache**: Main object storage (always contains all objects)
//! - **DRAM Cache**: Separate DashMap storing copies of hot objects
//! 
//! When an object is promoted:
//! 1. The object data is cloned from PMEM
//! 2. The copy is stored in the DRAM cache
//! 3. Get operations read from DRAM (fast path)
//! 
//! When an object is demoted:
//! 1. The DRAM copy is removed
//! 2. The PMEM copy remains (source of truth)
//! 3. Get operations read from PMEM (slower path)
//! 
//! # Configuration
//! 
//! The tiering manager supports runtime configuration:
//! - **DRAM Threshold**: Maximum size of DRAM tier (default: 20% of cache size)
//! - **Hotness Threshold**: Minimum accesses before promotion (default: 2)
//! - **High Water Mark**: Percentage of threshold to trigger demotion (default: 90%)
//! - **Low Water Mark**: Target percentage after demotion (default: 70%)
//! 
//! # Data Consistency
//! 
//! - **Lazy Promotion**: Objects are copied to DRAM in the background
//! - **Strong Consistency**: Updates write to PMEM and update DRAM copy if it exists
//! - **Deletes**: Remove from both PMEM and DRAM caches
//! - **PMEM as Source of Truth**: PMEM always contains all objects
//! 
//! # Example
//! 
//! ```ignore
//! use paper_cache::{PaperCache, PaperPolicy, TieringStats};
//! 
//! let cache = PaperCache::<u32, Box<[u8]>>::new(
//!     10_000_000,
//!     &[PaperPolicy::Lfu],
//!     PaperPolicy::Lfu,
//! ).unwrap();
//! 
//! // Configure tiering
//! cache.set_dram_threshold(2_000_000);  // 2 MB DRAM tier
//! cache.set_hotness_threshold(3);        // Promote after 3 accesses
//! 
//! // Use the cache normally
//! cache.set(1, &vec![0u8; 1000], None).unwrap();
//! cache.get(&1).unwrap();  // First access - from PMEM
//! cache.get(&1).unwrap();  // Second access - from PMEM
//! cache.get(&1).unwrap();  // Third access - triggers promotion, data copied to DRAM
//! 
//! // After a short delay for background promotion:
//! std::thread::sleep(std::time::Duration::from_millis(100));
//! cache.get(&1).unwrap();  // Now served from DRAM cache (faster!)
//! 
//! // Check tiering stats
//! let stats = cache.tiering_stats();
//! println!("Objects in DRAM: {}", stats.dram_objects);
//! println!("Promotions: {}", stats.promotions);
//! ```

pub mod manager;
pub mod object;

#[cfg(feature = "key_value_pmem")]
pub use manager::TieringManager;
pub use manager::TieringConfig;
#[cfg(feature = "key_value_pmem")]
pub use manager::TieringStats;
#[cfg(feature = "key_value_pmem")]
pub use object::TieringObject;

#[cfg(feature = "multitiering")]
pub use manager::MultiTieringManager;
#[cfg(feature = "multitiering")]
pub use manager::MultiTieringStats;
#[cfg(feature = "multitiering")]
pub use manager::TierState;

#[cfg(all(test, feature = "multitiering"))]
mod multitiering_tests {
    use super::*;

    fn make_manager(warm_cap: usize, hot_cap_bytes: u64) -> MultiTieringManager {
        MultiTieringManager::new(TieringConfig {
            warm_capacity_items: warm_cap,
            hot_capacity_bytes: hot_cap_bytes,
            ..TieringConfig::default()
        })
    }

    /// An object inserted at rest must start in the Cold tier.
    #[test]
    fn test_initial_state_is_cold() {
        let mgr = make_manager(10, 10_000);
        mgr.register_object(1, 100);
        assert_eq!(mgr.get_tier_state(&1), Some(TierState::Cold));
    }

    /// First access Cold -> Warm; second access Warm -> Hot.
    #[test]
    fn test_cold_to_warm_to_hot_lifecycle() {
        let mgr = make_manager(10, 10_000);
        mgr.register_object(1, 100);

        // First access: Cold -> Warm
        let state = mgr.record_access(1);
        assert_eq!(state, Some(TierState::Warm));
        assert_eq!(mgr.get_tier_state(&1), Some(TierState::Warm));

        // Second access: Warm -> Hot
        let state = mgr.record_access(1);
        assert_eq!(state, Some(TierState::Hot));
        assert_eq!(mgr.get_tier_state(&1), Some(TierState::Hot));

        let stats = mgr.stats();
        assert_eq!(stats.promotions_to_warm, 1);
        assert_eq!(stats.promotions_to_hot, 1);
        assert_eq!(stats.hot_objects, 1);
        assert_eq!(stats.warm_objects, 0);
        assert_eq!(stats.cold_objects, 0);
    }

    /// Accessing a Hot object must leave it in Hot (no regression).
    #[test]
    fn test_hot_object_stays_hot() {
        let mgr = make_manager(10, 10_000);
        mgr.register_object(1, 100);
        mgr.record_access(1); // -> Warm
        mgr.record_access(1); // -> Hot
        let state = mgr.record_access(1); // should remain Hot
        assert_eq!(state, Some(TierState::Hot));
        assert_eq!(mgr.get_tier_state(&1), Some(TierState::Hot));
    }

    /// When inserting past `hot_capacity_bytes`, the *oldest* Hot object must be
    /// demoted **directly** to Cold (bypassing Warm).
    #[test]
    fn test_hot_overflow_demotes_oldest_to_cold_directly() {
        // hot_capacity_bytes = 200; each object is 100 bytes -> 2 objects fit.
        let mgr = make_manager(100, 200);

        // Register and promote three objects to Hot
        for key in [1u64, 2, 3] {
            mgr.register_object(key, 100);
            mgr.record_access(key); // Cold -> Warm
            mgr.record_access(key); // Warm -> Hot
        }
        // Objects were pushed in order 1, 2, 3 (front=newest).
        // Hot queue (front->back): [3, 2, 1].
        // After adding key=3 (hot_bytes would be 300 > 200):
        //   evict key=1 (oldest, at back) -> Cold; hot_bytes = 200.
        assert_eq!(mgr.get_tier_state(&1), Some(TierState::Cold),
            "oldest Hot object must be demoted directly to Cold");
        assert_eq!(mgr.get_tier_state(&2), Some(TierState::Hot));
        assert_eq!(mgr.get_tier_state(&3), Some(TierState::Hot));

        let stats = mgr.stats();
        assert_eq!(stats.hot_objects, 2);
        assert_eq!(stats.cold_objects, 1);
        assert_eq!(stats.demotions_to_cold, 1);
    }

    /// When the Warm pool overflows `warm_capacity_items`, the *oldest* Warm entry
    /// must be demoted to Cold.
    #[test]
    fn test_warm_overflow_demotes_oldest_to_cold() {
        // warm_capacity_items = 2 -> only two metadata entries can live in Warm.
        let mgr = make_manager(2, 1_000_000);

        mgr.register_object(1, 100);
        mgr.register_object(2, 100);
        mgr.register_object(3, 100);

        // Promote all three to Warm
        mgr.record_access(1); // warm_queue: [1]
        mgr.record_access(2); // warm_queue: [2, 1]
        mgr.record_access(3); // warm_queue: [3, 2, 1] -> overflow -> evict 1 (oldest)

        assert_eq!(mgr.get_tier_state(&1), Some(TierState::Cold),
            "oldest Warm object must be demoted to Cold on overflow");
        assert_eq!(mgr.get_tier_state(&2), Some(TierState::Warm));
        assert_eq!(mgr.get_tier_state(&3), Some(TierState::Warm));

        let stats = mgr.stats();
        assert_eq!(stats.warm_objects, 2);
        assert_eq!(stats.cold_objects, 1);
        assert_eq!(stats.demotions_to_cold, 1);
    }
}
