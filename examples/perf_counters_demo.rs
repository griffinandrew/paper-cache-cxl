/// Performance Counters Demo
/// 
/// This example demonstrates how to use the performance counters
/// to track hashmap memory accesses.
/// 
/// To run with hashbrown_dram:
/// ```
/// cargo run --example perf_counters_demo --no-default-features --features hashbrown_dram
/// ```

use paper_cache::{PaperCache, PaperPolicy};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Performance Counters Demo ===\n");
    
    // Create a cache instance
    // Note: For hashbrown_dram feature, the value type is Box<[u8]>
    let cache = PaperCache::<u64, Box<[u8]>>::new(
        10_000_000,  // 10 MB cache
        &[PaperPolicy::Lru],
        PaperPolicy::Lru,
    )?;
    
    println!("Cache created with 10MB capacity\n");
    
    // Perform some operations
    println!("Performing cache operations...\n");
    
    // Insert some values
    for i in 0..100 {
        let key = i;
        let value = format!("value_{}", i).into_bytes();
        cache.set(key, &value, None)?;
    }
    println!("Inserted 100 items");
    
    // Read some values
    for i in 0..50 {
        let _ = cache.get(&i);
    }
    println!("Read 50 items");
    
    // Check some values
    for i in 50..75 {
        let _ = cache.has(&i);
    }
    println!("Checked 25 items with has()");
    
    // Delete some values
    for i in 0..10 {
        let _ = cache.del(&i);
    }
    println!("Deleted 10 items");
    
    println!("\n");
    
    // Print performance statistics
    paper_cache::perf_counters::print_perf_stats();
    
    // Access statistics programmatically
    if let Some(stats) = paper_cache::perf_counters::get_hashmap_stats() {
        println!("Programmatic access to stats:");
        println!("  Total accesses: {}", stats.total_accesses);
        println!("  Reads: {}", stats.reads);
        println!("  Writes: {}", stats.writes);
        println!("  Lookups: {}", stats.lookups);
        println!("  Insertions: {}", stats.insertions);
        println!("  Deletions: {}", stats.deletions);
    }
    
    Ok(())
}
