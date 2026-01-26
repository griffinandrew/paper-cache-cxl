# Hardware Performance Counters Guide

## Overview

This implementation uses Linux `perf_event` to track actual CPU-level memory accesses during hashmap operations. This provides **real hardware metrics** rather than just software operation counts.

## What Gets Measured

### Hardware Events

1. **CPU Cycles** - Total CPU cycles consumed by the operation
2. **Instructions** - Number of CPU instructions executed
3. **Cache References** - Total cache lookups (all levels)
4. **Cache Misses** - Cache misses across all cache levels (L1, L2, L3)

### Derived Metrics

- **IPC (Instructions Per Cycle)** - Instruction throughput
- **Cache Miss Rate** - Percentage of cache lookups that miss
- **Average Metrics** - Per-operation averages for cycles, cache misses, etc.

## Requirements

### Operating System
- **Linux kernel** with `perf_event` support (kernel 2.6.31+)
- **x86_64, ARM, or other architecture** with Performance Monitoring Unit (PMU)

### Permissions

Hardware performance counters require special permissions. By default, Linux restricts access. You have several options:

#### Option 1: Temporarily Allow Access (Recommended for Development)
```bash
# Allow all users to access performance counters
sudo sysctl kernel.perf_event_paranoid=-1

# Or allow only for current boot
echo -1 | sudo tee /proc/sys/kernel/perf_event_paranoid
```

#### Option 2: Run with Sudo
```bash
sudo cargo run --example hw_perf_demo --no-default-features --features hashbrown_dram
```

#### Option 3: Add CAP_PERFMON Capability (Linux 5.8+)
```bash
sudo setcap cap_perfmon=eip target/debug/examples/hw_perf_demo
./target/debug/examples/hw_perf_demo
```

#### Option 4: Add to `perf_users` Group (if configured)
```bash
sudo usermod -a -G perf_users $USER
```

### Check Current Settings
```bash
# Check perf_event_paranoid level
cat /proc/sys/kernel/perf_event_paranoid

# Levels:
#  -1: Allow all events for all users
#   0: Allow access to CPU events for all users
#   1: Allow access to kernel profiling for privileged users
#   2: Allow only CPU events for privileged users (default on most systems)
```

### Environments Where Counters May Not Be Available

❌ **Docker containers** (without `--privileged` or `--cap-add=SYS_ADMIN`)
❌ **Some VMs** (depends on hypervisor passthrough support)
❌ **WSL1** (Windows Subsystem for Linux v1)
❌ **CI/CD environments** (GitHub Actions, GitLab CI, etc.)
✅ **WSL2** (with proper kernel)
✅ **Native Linux** (with permissions)
✅ **Some VMs** (with PMU virtualization enabled)

## Usage

### Basic Usage

```rust
use paper_cache::{PaperCache, PaperPolicy, measure_operation};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cache = PaperCache::<u64, Box<[u8]>>::new(
        10_000_000,
        &[PaperPolicy::Lru],
        PaperPolicy::Lru,
    )?;

    // Measure a single GET operation
    let key = 42;
    let (result, hw_measurement) = measure_operation(|| cache.get(&key));

    if let Some(measurement) = hw_measurement {
        println!("GET operation consumed:");
        println!("  {} CPU cycles", measurement.cycles);
        println!("  {} instructions (IPC: {:.2})", 
                 measurement.instructions, measurement.ipc());
        println!("  {} cache references", measurement.cache_references);
        println!("  {} cache misses ({:.2}% miss rate)", 
                 measurement.cache_misses, measurement.cache_miss_rate());
    } else {
        println!("Hardware performance counters not available");
    }

    Ok(())
}
```

### Measuring Multiple Operations

```rust
use paper_cache::{get_hw_counters, measure_operation};

// Measure multiple GET operations
for i in 0..100 {
    let (result, hw_measurement) = measure_operation(|| cache.get(&i));
    
    if let Some(measurement) = hw_measurement {
        // Record the measurement
        #[cfg(feature = "hashbrown_dram")]
        get_hw_counters().global_hashbrown_dram.record_get(measurement);
    }
}

// Get aggregated statistics
if let Some(stats) = paper_cache::get_hw_hashmap_stats() {
    println!("GET operations: {} calls", stats.get.count);
    println!("Average cycles per GET: {}", stats.get.avg_cycles);
    println!("Average cache misses: {}", stats.get.avg_cache_misses);
    println!("Overall cache miss rate: {:.2}%", stats.get.cache_miss_rate());
}
```

### Comparing DRAM vs PMEM

```rust
// Build with hashbrown_dram
cargo build --no-default-features --features hashbrown_dram
// Run and collect metrics...

// Build with global_hashtable_pmem
cargo +nightly build --no-default-features --features global_hashtable_pmem
// Run and collect metrics...

// Compare:
// - Average cycles: DRAM should be lower
// - Cache miss rate: PMEM may be higher due to different access patterns
// - Total memory accesses: Shows actual hardware memory traffic
```

### Error Handling

```rust
use paper_cache::hw_perf_counters::PerfCounterGroup;

// Check if counters are available before measuring
let test_counter = PerfCounterGroup::new();
if !test_counter.is_available() {
    eprintln!("WARNING: Hardware performance counters not available");
    eprintln!("Possible reasons:");
    eprintln!("  - Insufficient permissions (try: sudo or adjust perf_event_paranoid)");
    eprintln!("  - Running in container/VM");
    eprintln!("  - Hardware doesn't support PMU");
    
    // Continue with software counters only
} else {
    println!("Hardware performance counters available!");
}
```

## Interpreting Results

### CPU Cycles

- **Lower is better** - Fewer cycles means faster operation
- **Typical values**:
  - GET on hot cache: 50-200 cycles
  - GET with cache miss: 200-1000+ cycles
  - SET operation: 100-500 cycles

### Cache Miss Rate

- **Lower is better** - More data in cache
- **Typical values**:
  - Hot data (DRAM): 1-5% miss rate
  - Cold data: 20-50% miss rate
  - PMEM access: Variable, depends on layout

### IPC (Instructions Per Cycle)

- **Higher is better** - More work done per cycle
- **Typical values**:
  - Modern CPUs: 1.5-2.5 IPC
  - Memory-bound code: 0.5-1.0 IPC
  - Compute-bound code: 2.0-4.0 IPC

### What to Look For

#### Performance Issues
- **High cache miss rate** (>20%) - Consider:
  - Data structure layout optimization
  - Prefetching strategies
  - Reduce data structure size

- **Low IPC** (<1.0) - Indicates:
  - Memory-bound operations
  - Cache misses causing stalls
  - Dependencies limiting parallelism

#### DRAM vs PMEM Comparison
- **Cycles**: PMEM operations should show higher cycle counts
- **Cache misses**: Different access patterns may affect miss rates
- **Memory traffic**: PMEM shows actual difference in memory subsystem

## Advanced Features

### Custom Event Groups

The implementation can be extended to measure additional events:

```rust
// Future: Add support for custom events
// - MEM_LOAD_RETIRED.L3_MISS (Intel)
// - MEM_INST_RETIRED.ALL_STORES (Intel)
// - Branch prediction events
// - TLB misses
```

### Architecture-Specific Events

Different CPUs support different events:
- **Intel**: Precise events, load/store tracking
- **AMD**: Similar but different event codes
- **ARM**: Architecture-specific PMU events

## Troubleshooting

### "Permission denied" or "Operation not permitted"
```bash
# Check current setting
cat /proc/sys/kernel/perf_event_paranoid

# Set to -1 to allow
sudo sysctl kernel.perf_event_paranoid=-1
```

### "No hardware performance counters available"
- Check if running in container: `systemd-detect-virt`
- Check CPU support: `perf list` (install `linux-tools-common`)
- Try: `perf stat ls` to verify perf works

### Counters Always Return Zero
- May be in virtualized environment
- Some hypervisors don't expose PMU to guests
- Check VM settings for PMU passthrough

### High Variance in Measurements
- Normal for small operations (<100 cycles)
- Use multiple measurements and average
- Disable CPU frequency scaling: `cpupower frequency-set -g performance`

## Performance Overhead

Hardware performance counters have **minimal overhead**:
- ~10-50 nanoseconds per measurement
- No impact on measured code execution
- Negligible memory usage

The overhead comes from:
1. System call to enable/disable counters
2. Reading counter values
3. Recording measurements

For production use, measure operations in batches rather than every single operation.

## Examples

### Example 1: Benchmark GET Performance
```bash
cargo run --example hw_perf_demo --no-default-features --features hashbrown_dram
```

### Example 2: Compare DRAM vs PMEM
```bash
# DRAM version
cargo run --example hw_perf_demo --no-default-features --features hashbrown_dram > dram_results.txt

# PMEM version (requires nightly + PMEM hardware)
cargo +nightly run --example hw_perf_demo --no-default-features --features global_hashtable_pmem > pmem_results.txt

# Compare results
diff dram_results.txt pmem_results.txt
```

## References

- [Linux perf_event documentation](https://man7.org/linux/man-pages/man2/perf_event_open.2.html)
- [perf-event Rust crate](https://docs.rs/perf-event/)
- [Intel Performance Counter Monitor](https://software.intel.com/content/www/us/en/develop/articles/intel-performance-counter-monitor.html)
- [Brendan Gregg's perf examples](http://www.brendangregg.com/perf.html)
