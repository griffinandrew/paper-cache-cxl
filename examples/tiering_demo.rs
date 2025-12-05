/*
 * Example demonstrating the tiering manager functionality
 * 
 * This example shows how to:
 * 1. Configure a tiering manager with a custom threshold
 * 2. Register objects in PMEM
 * 3. Promote hot objects to DRAM
 * 4. Monitor DRAM usage and automatically demote cold objects
 * 5. Track tiering statistics
 */

use paper_cache::{TieringManager, TieringConfig};

fn main() {
    println!("=== Tiering Manager Example ===\n");
    
    // Configure tiering with a 1MB DRAM threshold
    let config = TieringConfig {
        dram_threshold: 1_048_576,  // 1 MB
        high_water_mark: 0.9,        // Start demoting at 90% (900 KB)
        low_water_mark: 0.7,         // Demote until 70% (700 KB)
    };
    
    let manager = TieringManager::new(config);
    
    println!("Initial configuration:");
    println!("  DRAM threshold: {} bytes (1 MB)", manager.dram_threshold());
    println!("  High water mark: 90%");
    println!("  Low water mark: 70%\n");
    
    // Simulate adding objects to cache
    println!("Step 1: Registering 20 objects in PMEM (each 50 KB)...");
    for i in 0..20 {
        manager.register_object(i, 51_200); // 50 KB each
    }
    
    let stats = manager.stats();
    println!("  PMEM-only objects: {}", stats.pmem_only_objects);
    println!("  DRAM objects: {}", stats.dram_objects);
    println!("  DRAM usage: {} bytes\n", stats.dram_size);
    
    // Simulate accessing objects (triggering promotions)
    println!("Step 2: Accessing objects to trigger promotions...");
    for i in 0..15 {
        // Access each object twice to trigger promotion
        manager.record_access(i);
        if manager.record_access(i) {
            manager.promote_to_dram(i);
        }
    }
    
    let stats = manager.stats();
    println!("  PMEM-only objects: {}", stats.pmem_only_objects);
    println!("  DRAM objects: {}", stats.dram_objects);
    println!("  DRAM usage: {} bytes ({:.2}% of threshold)", 
             stats.dram_size, 
             (stats.dram_size as f64 / manager.dram_threshold() as f64) * 100.0);
    println!("  Promotions: {}\n", stats.promotions);
    
    // Try to promote more objects (will exceed threshold)
    println!("Step 3: Attempting to promote more objects...");
    for i in 15..20 {
        manager.record_access(i);
        if manager.record_access(i) {
            if manager.promote_to_dram(i) {
                println!("  ✓ Promoted object {}", i);
            } else {
                println!("  ✗ Failed to promote object {} (threshold would be exceeded)", i);
            }
        }
    }
    
    let stats = manager.stats();
    println!("\n  Current DRAM usage: {} bytes ({:.2}% of threshold)\n", 
             stats.dram_size,
             (stats.dram_size as f64 / manager.dram_threshold() as f64) * 100.0);
    
    // Check if demotion is needed
    println!("Step 4: Checking if demotion is needed...");
    let keys_to_demote = manager.get_keys_to_demote();
    
    if keys_to_demote.is_empty() {
        println!("  No demotion needed (below high water mark)\n");
    } else {
        println!("  DRAM usage exceeded high water mark!");
        println!("  Objects to demote: {} (to reach low water mark)", keys_to_demote.len());
        
        // Perform demotions
        for key in &keys_to_demote {
            manager.demote_from_dram(*key);
        }
        
        let stats = manager.stats();
        println!("  DRAM objects after demotion: {}", stats.dram_objects);
        println!("  DRAM usage after demotion: {} bytes ({:.2}% of threshold)", 
                 stats.dram_size,
                 (stats.dram_size as f64 / manager.dram_threshold() as f64) * 100.0);
        println!("  Total demotions: {}\n", stats.demotions);
    }
    
    // Final statistics
    println!("=== Final Statistics ===");
    let stats = manager.stats();
    println!("  DRAM objects: {}", stats.dram_objects);
    println!("  PMEM-only objects: {}", stats.pmem_only_objects);
    println!("  Total objects: {}", stats.dram_objects + stats.pmem_only_objects);
    println!("  DRAM usage: {} bytes ({:.2} MB)", 
             stats.dram_size, 
             stats.dram_size as f64 / 1_048_576.0);
    println!("  Total promotions: {}", stats.promotions);
    println!("  Total demotions: {}", stats.demotions);
    
    println!("\n=== Example Complete ===");
}
