//! Tiering Manager Module
//! 
//! This module provides functionality to manage objects between DRAM and PMEM tiers.
//! Objects in DRAM are copies of those in PMEM, with PMEM serving as the source of truth.
//! 
//! # Overview
//! 
//! The tiering system implements a two-tier caching strategy:
//! - **Far Tier (PMEM)**: All objects are stored here by default
//! - **Near Tier (DRAM)**: Hot objects are promoted here for faster access
//! 
//! # Architecture
//! 
//! The tiering manager integrates with the existing worker manager workflow:
//! - Receives `Get` and `Set` events from the worker manager
//! - Tracks object access patterns to determine hotness
//! - Periodically evaluates objects for promotion or demotion
//! - Coordinates with the LFU eviction stack for runtime access metrics
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
//! - **Lazy Promotion**: Objects are promoted in the background
//! - **Strong Consistency**: Updates and deletes are applied to both tiers immediately
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
//! cache.get(&1).unwrap();  // First access
//! cache.get(&1).unwrap();  // Second access
//! cache.get(&1).unwrap();  // Third access - object will be promoted
//! 
//! // Check tiering stats
//! let stats = cache.tiering_stats();
//! println!("Objects in DRAM: {}", stats.dram_objects);
//! println!("Promotions: {}", stats.promotions);
//! ```

pub mod manager;

pub use manager::TieringManager;
pub use manager::TieringConfig;
pub use manager::TieringStats;
