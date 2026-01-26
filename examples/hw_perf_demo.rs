/// Hardware Performance Counters Demo
/// 
/// This example demonstrates how to use hardware performance counters
/// to track actual CPU-level memory accesses during hashmap operations.
/// 
/// To run with hashbrown in DRAM:
/// ```
/// cargo run --example hw_perf_demo --no-default-features --features hashbrown_dram
/// ```
/// 
/// Note: Requires access to Linux perf_event (may need sudo or perf_event_paranoid settings)

use paper_cache::{PaperCache, PaperPolicy};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Hardware Performance Counters Demo ===\n");
    
    // Check if hardware counters are available
    println!("Checking hardware performance counter availability...");
    let test_counter = paper_cache::hw_perf_counters::PerfCounterGroup::new();
    if !test_counter.is_available() {
        println!("\nWARNING: Hardware performance counters not available!");
        println!("This may be due to:");
        println!("  - Running in a container or VM");
        println!("  - Insufficient permissions (try: sudo sysctl kernel.perf_event_paranoid=-1)");
        println!("  - Hardware doesn't support performance counters");
        println!("\nContinuing with software counters only...\n");
    } else {
        println!("✓ Hardware performance counters available!\n");
    }
    
    // Create a cache instance
    let cache = PaperCache::<u64, Box<[u8]>>::new(
        10_000_000,  // 10 MB cache
        &[PaperPolicy::Lru],
        PaperPolicy::Lru,
    )?;
    
    println!("Cache created with 10MB capacity\n");
    println!("Performing cache operations with hardware monitoring...\n");
    
    // Perform SET operations with hardware measurement
    println!("Inserting 100 items...");
    for i in 0..100 {
        let key = i;
        let value = format!("value_{}", i).into_bytes();
        
        // Measure the set operation
        let (result, measurement) = paper_cache::measure_operation(|| {
            cache.set(key, &value, None)
        });
        
        // Record the measurement if available
        if let Some(hw_measure) = measurement {
            #[cfg(feature = "hashbrown_dram")]
            paper_cache::get_hw_counters().global_hashbrown_dram.record_set(hw_measure);
        }
        
        result?;
    }
    println!("✓ Inserted 100 items\n");
    
    // Perform GET operations with hardware measurement
    println!("Reading 50 items...");
    for i in 0..50 {
        let (result, measurement) = paper_cache::measure_operation(|| {
            cache.get(&i)
        });
        
        if let Some(hw_measure) = measurement {
            #[cfg(feature = "hashbrown_dram")]
            paper_cache::get_hw_counters().global_hashbrown_dram.record_get(hw_measure);
        }
        
        let _ = result;
    }
    println!("✓ Read 50 items\n");
    
    // Perform HAS operations with hardware measurement
    println!("Checking 25 items with has()...");
    for i in 50..75 {
        let (result, measurement) = paper_cache::measure_operation(|| {
            cache.has(&i)
        });
        
        if let Some(hw_measure) = measurement {
            #[cfg(feature = "hashbrown_dram")]
            paper_cache::get_hw_counters().global_hashbrown_dram.record_has(hw_measure);
        }
        
        let _ = result;
    }
    println!("✓ Checked 25 items\n");
    
    // Perform DELETE operations with hardware measurement
    println!("Deleting 10 items...");
    for i in 0..10 {
        let (result, measurement) = paper_cache::measure_operation(|| {
            cache.del(&i)
        });
        
        if let Some(hw_measure) = measurement {
            #[cfg(feature = "hashbrown_dram")]
            paper_cache::get_hw_counters().global_hashbrown_dram.record_del(hw_measure);
        }
        
        let _ = result;
    }
    println!("✓ Deleted 10 items\n");
    
    println!("\n");
    
    // Print hardware performance statistics
    paper_cache::print_hw_perf_stats();
    
    // Also print software counters for comparison
    paper_cache::perf_counters::print_perf_stats();
    
    // Access statistics programmatically
    if let Some(stats) = paper_cache::get_hw_hashmap_stats() {
        println!("\nDetailed Analysis:");
        println!("  Total operations measured: {}", stats.total_operations());
        println!("  Total CPU cycles: {}", stats.total_cycles());
        println!("  Total cache references: {}", stats.total_cache_refs());
        println!("  Total cache misses: {}", stats.total_cache_misses());
        
        if stats.total_cache_refs() > 0 {
            let miss_rate = 100.0 * stats.total_cache_misses() as f64 / stats.total_cache_refs() as f64;
            println!("  Overall cache miss rate: {:.2}%", miss_rate);
        }
        
        println!("\nPer-operation breakdown:");
        if stats.get.count > 0 {
            println!("  GET: {} cycles/op, {:.2}% cache miss rate", 
                     stats.get.avg_cycles, stats.get.cache_miss_rate());
        }
        if stats.set.count > 0 {
            println!("  SET: {} cycles/op, {:.2}% cache miss rate", 
                     stats.set.avg_cycles, stats.set.cache_miss_rate());
        }
        if stats.has.count > 0 {
            println!("  HAS: {} cycles/op, {:.2}% cache miss rate", 
                     stats.has.avg_cycles, stats.has.cache_miss_rate());
        }
        if stats.del.count > 0 {
            println!("  DEL: {} cycles/op, {:.2}% cache miss rate", 
                     stats.del.avg_cycles, stats.del.cache_miss_rate());
        }
    }
    
    Ok(())
}
