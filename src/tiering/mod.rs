//! Tiering Manager Module
//! 
//! This module provides functionality to manage objects between DRAM and PMEM tiers
//! with **actual physical data copying**.
//! 
//! # Overview
//! 
//! The tiering system implements a two-tier caching strategy with physical data copies:
//! - **Far Tier (PMEM)**: All objects are stored here (source of truth)
//! - **Near Tier (DRAM)**: All objects are **physically copied** here on SET operations
//! 
//! # Architecture
//! 
//! The tiering manager integrates with the existing worker manager workflow:
//! - Receives `Get` and `Set` events from the worker manager
//! - **SET operations immediately insert objects to both DRAM and PMEM**, bypassing all policies
//! - Get operations check DRAM cache first, then fall back to PMEM
//! - Periodically evaluates objects for demotion based on LRU policy
//! - Coordinates with eviction policies for runtime management
//! 
//! # Data Copy Model
//! 
//! This implementation maintains two physical copies of all objects:
//! - **PMEM Cache**: Main object storage (always contains all objects)
//! - **DRAM Cache**: Separate DashMap storing copies of objects (all objects after SET)
//! 
//! When an object is SET (new or updated):
//! 1. The object is stored in PMEM (main cache)
//! 2. The object data is immediately cloned to DRAM cache
//! 3. This bypasses hotness thresholds and tiering policies
//! 
//! When an object is demoted (due to DRAM pressure):
//! 1. The DRAM copy is removed
//! 2. The PMEM copy remains (source of truth)
//! 3. Get operations read from PMEM (slower path)
//! 
//! # Configuration
//! 
//! The tiering manager supports runtime configuration:
//! - **DRAM Threshold**: Maximum size of DRAM tier (default: 20% of cache size)
//! - **Hotness Threshold**: Not used for SET operations; only for legacy promotion logic
//! - **High Water Mark**: Percentage of threshold to trigger demotion (default: 90%)
//! - **Low Water Mark**: Target percentage after demotion (default: 70%)
//! 
//! # Data Consistency
//! 
//! - **Immediate Insertion**: Objects are placed in DRAM immediately on SET
//! - **Strong Consistency**: Updates write to PMEM and update DRAM copy unconditionally
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
//! 
//! // Use the cache normally
//! cache.set(1, &vec![0u8; 1000], None).unwrap();
//! // Object is now in BOTH DRAM and PMEM immediately!
//! 
//! cache.get(&1).unwrap();  // Served from DRAM cache (faster!)
//! 
//! // Check tiering stats
//! let stats = cache.tiering_stats();
//! println!("Objects in DRAM: {}", stats.dram_objects);
//! ```

pub mod manager;

pub use manager::TieringManager;
pub use manager::TieringConfig;
pub use manager::TieringStats;
