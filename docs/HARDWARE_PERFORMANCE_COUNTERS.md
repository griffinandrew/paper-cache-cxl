# Hardware Performance Counters

This module provides hardware performance monitoring using Linux `perf_event` to track actual CPU-level memory accesses with focus on LLC (Last Level Cache) specific events.

## Features

- **Essential Hardware Counters** (6 counters to avoid multiplexing):
  - CPU_CYCLES - Total CPU cycles consumed
  - INSTRUCTIONS - Number of CPU instructions executed
  - CACHE_REFERENCES - Total cache lookups (all levels)
  - CACHE_MISSES - Cache misses across all levels
  - **LLC_LOADS** - Last Level Cache read operations (using `WhichCache::LL`)
  - **LLC_LOAD_MISSES** - Last Level Cache read misses (using `WhichCache::LL`)

- **Derived Metrics**:
  - IPC (Instructions Per Cycle)
  - Cache Miss Rate
  - LLC Miss Rate (specific to Last Level Cache)

## Requirements

### Operating System
- Linux kernel with `perf_event` support (kernel 2.6.31+)
- x86_64, ARM, or other architecture with Performance Monitoring Unit (PMU)

### Permissions

Hardware performance counters require special permissions. You have several options:

#### Option 1: Temporarily Allow Access (Recommended for Development)
```bash
# Allow all users to access performance counters
sudo sysctl kernel.perf_event_paranoid=-1

# Or allow only for current boot
echo -1 | sudo tee /proc/sys/kernel/perf_event_paranoid
```

#### Option 2: Run with Sudo
```bash
sudo cargo run --example hw_perf_demo
```

#### Option 3: Add CAP_PERFMON Capability (Linux 5.8+)
```bash
sudo setcap cap_perfmon=eip target/debug/examples/hw_perf_demo
./target/debug/examples/hw_perf_demo
```

### Check Current Settings
```bash
# Check perf_event_paranoid level
cat /proc/sys/kernel/perf_event_paranoid

# Levels:
# -1: Allow all events for all users
# 0: Allow access to CPU events for all users
# 1: Allow access to kernel profiling for privileged users
# 2: Allow only CPU events for privileged users (default on most systems)
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
use paper_cache::{PaperCache, PaperPolicy, measure_operation, get_hw_counters};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cache = PaperCache::<String, Vec<u8>>::new(
        10_000_000,
        &[PaperPolicy::Lru],
        PaperPolicy::Lru,
    )?;

    // Measure a single GET operation
    let key = "test_key".to_string();
    let value = vec![0u8; 1024];
    
    cache.set(key.clone(), value, None)?;
    
    let (result, hw_measurement) = measure_operation(|| cache.get(&key));

    if let Some(measurement) = hw_measurement {
        println!("GET operation consumed:");
        println!(" {} CPU cycles", measurement.cycles);
        println!(" {} instructions (IPC: {:.2})",
            measurement.instructions, measurement.ipc());
        println!(" {} cache misses ({:.2}% miss rate)",
            measurement.cache_misses, measurement.cache_miss_rate());
        println!(" {} LLC loads", measurement.llc_loads);
        println!(" {} LLC load misses ({:.2}% LLC miss rate)",
            measurement.llc_load_misses, measurement.llc_miss_rate());
        
        // Record the measurement
        get_hw_counters().global_hashbrown_dram.record_get(measurement);
    } else {
        println!("Hardware performance counters not available");
    }

    Ok(())
}
```

### Running the Example

```bash
# Build and run the example
cargo run --example hw_perf_demo

# If you get permission errors:
sudo sysctl kernel.perf_event_paranoid=-1
cargo run --example hw_perf_demo
```

### Measuring Multiple Operations

```rust
use paper_cache::{get_hw_counters, measure_operation, print_hw_perf_stats};

// Measure multiple GET operations
for i in 0..100 {
    let (result, hw_measurement) = measure_operation(|| cache.get(&i));

    if let Some(measurement) = hw_measurement {
        // Record the measurement
        get_hw_counters().global_hashbrown_dram.record_get(measurement);
    }
}

// Print aggregated statistics
print_hw_perf_stats();
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

### LLC Miss Rate
- **Last Level Cache specific** - Using `WhichCache::LL`
- **Lower is better** - Better LLC utilization
- More precise than general cache miss rate for memory subsystem analysis

### IPC (Instructions Per Cycle)
- **Higher is better** - More work done per cycle
- **Typical values**:
  - Modern CPUs: 1.5-2.5 IPC
  - Memory-bound code: 0.5-1.0 IPC
  - Compute-bound code: 2.0-4.0 IPC

## Troubleshooting

### "Permission denied" or "Operation not permitted"
```bash
# Check current setting
cat /proc/sys/kernel/perf_event_paranoid

# Set to -1 to allow
sudo sysctl kernel.perf_event_paranoid=-1
```

### "No such file or directory"
- Running in virtualized environment without PMU access
- Check if running in container: `systemd-detect-virt`
- Some hypervisors don't expose PMU to guests

### Counters Always Return Zero
Even with successful creation, counters may return zero if:
1. **Not properly enabled** - Make sure `enable()` is called before operation
2. **Not properly disabled** - Make sure `disable()` is called before reading
3. **Reset not called** - Call `reset()` before starting measurement
4. **Operation too fast** - The measured operation might be too quick to register
5. **Virtualized environment** - Some VMs don't properly emulate PMU

### High Variance in Measurements
- Normal for small operations (<100 cycles)
- Use multiple measurements and average
- Disable CPU frequency scaling: `cpupower frequency-set -g performance`

## Implementation Details

### Why Only 6 Counters?

Modern CPUs typically support 6-8 hardware performance counters simultaneously. Using more counters than available hardware units causes "multiplexing" where the kernel time-shares counters, leading to inaccurate results. This implementation uses 6 essential counters to stay within hardware limits.

### Why LLC-Specific Events?

The `WhichCache::LL` parameter ensures we're measuring Last Level Cache (typically L3) specifically, rather than combined metrics across all cache levels. This provides:
- More precise memory subsystem analysis
- Better correlation with actual DRAM/CXL memory access patterns
- Clearer distinction between cache hierarchy levels

### Thread-Local Counters

Performance counters are thread-local (stored in `thread_local!`) because:
- `perf_event` counters are per-thread by default
- Avoids synchronization overhead
- Allows parallel measurement of different operations

## References

- [Linux perf_event documentation](https://man7.org/linux/man-pages/man2/perf_event_open.2.html)
- [perf-event Rust crate](https://docs.rs/perf-event/)
- [WhichCache enum documentation](https://docs.rs/perf-event/latest/perf_event/events/enum.WhichCache.html)
- [Intel Performance Counter Monitor](https://software.intel.com/content/www/us/en/develop/articles/intel-performance-counter-monitor.html)
