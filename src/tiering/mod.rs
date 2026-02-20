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
//! ```no_run
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

#[cfg(any(feature = "key_value_pmem", feature = "alloc_api_exp"))]
pub mod manager;
#[cfg(any(feature = "key_value_pmem", feature = "alloc_api_exp", feature = "multitiering"))]
pub mod object;

#[cfg(feature = "multitiering")]
pub mod multitier_manager;

#[cfg(any(feature = "key_value_pmem", feature = "alloc_api_exp"))]
pub use manager::TieringManager;
#[cfg(any(feature = "key_value_pmem", feature = "alloc_api_exp"))]
pub use manager::TieringConfig;
#[cfg(any(feature = "key_value_pmem", feature = "alloc_api_exp"))]
pub use manager::TieringStats;
#[cfg(any(feature = "key_value_pmem", feature = "alloc_api_exp", feature = "multitiering"))]
pub use object::TieringObject;

#[cfg(feature = "multitiering")]
pub use multitier_manager::MultitieringManager;
#[cfg(feature = "multitiering")]
pub use multitier_manager::MultitieringConfig;
#[cfg(feature = "multitiering")]
pub use multitier_manager::MultitieringStats;
