/// Hardware Performance Counters Demo
/// 
/// This example demonstrates how to use hardware performance counters
/// to track actual CPU-level memory accesses during cache operations.
/// 
/// To run:
/// ```
/// cargo run --example hw_perf_demo
/// ```
/// 
/// Note: Requires access to Linux perf_event (may need sudo or perf_event_paranoid settings)

use paper_cache::{PaperCache, PaperPolicy, measure_operation, get_hw_counters, print_hw_perf_stats};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Hardware Performance Counters Demo ===\n");
    
    // Create a cache instance
    let cache = PaperCache::<String, Vec<u8>>::new(
        10_000_000,  // 10 MB cache
        &[PaperPolicy::Lru],
        PaperPolicy::Lru,
    )?;
    
    println!("Cache created with 10MB capacity\n");
    
    // Create a test key and value
    let test_key = format!("key_{:0>30}", 100);
    let test_value = vec![0u8; 10240]; // 10KB value
    
    println!("Single GET operation for key {}:", test_key);
    
    // First insert the key
    cache.set(test_key.clone(), test_value.clone(), None)?;
    
    // Measure a GET operation
    let (result, hw_measurement) = measure_operation(|| cache.get(&test_key));
    
    match result {
        Ok(_) => println!("GET success: key={} len={}", test_key, test_value.len()),
        Err(e) => println!("GET failed: {:?}", e),
    }
    
    // Print individual measurement
    if let Some(measurement) = hw_measurement {
        println!("Hardware performance stats for the GET operation:\n");
        
        // Record the measurement
        get_hw_counters().global_hashbrown_dram.record_get(measurement.clone());
        
        // Print the raw measurement for debugging
        println!("{:#?}", measurement);
        
        // Check which counters are available
        println!("Available counters:");
        println!("  Cycles: {}", if measurement.cycles > 0 { "✓" } else { "✗" });
        println!("  Instructions: {}", if measurement.instructions > 0 { "✓" } else { "✗" });
        println!("  Branch tracking: {}", if measurement.branch_instructions > 0 { "✗" } else { "✗" });
        println!("  L1 D-cache: {}", if measurement.l1_dcache_loads > 0 { "✓" } else { "✗" });
        println!("  LLC: {}", if measurement.llc_loads > 0 { "✓" } else { "✗" });
        println!("  TLB: {}", if measurement.dtlb_loads > 0 { "✓" } else { "✗" });
        
        if measurement.cycles > 0 {
            println!("\nCycles: {}, Instructions: {}, IPC: {:.2}", 
                     measurement.cycles, measurement.instructions, measurement.ipc());
            println!("Cache misses: {}, Miss rate: {:.2}%", 
                     measurement.cache_misses, measurement.cache_miss_rate());
        } else {
            println!("\n⚠ WARNING: All counter values are zero!");
            println!("This may indicate:");
            println!("  - Insufficient permissions (try: sudo sysctl kernel.perf_event_paranoid=-1)");
            println!("  - Running in a container or VM without PMU access");
            println!("  - Hardware doesn't support performance counters");
        }
    } else {
        println!("Hardware performance counters not available");
    }
    
    println!("\n");
    
    // Print aggregated statistics
    print_hw_perf_stats();
    
    Ok(())
}
