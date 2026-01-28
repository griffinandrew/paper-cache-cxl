# Hardware Performance Counters Guide

## Overview

This implementation uses Linux `perf_event` to track comprehensive microarchitectural and memory hierarchy statistics during hashmap operations. This provides **real hardware metrics** for deep performance analysis and debugging.

**Note**: Hardware performance counters are gated behind the `hw_perf_counters` feature flag. You must enable this feature to use them:

```bash
cargo build --features hw_perf_counters
```

## Counter Optimization for Accuracy

**Hardware Limitation**: Most CPUs support only ~8 simultaneous performance counters. When more counters are requested, the kernel multiplexes them (time-slices), reducing accuracy and adding overhead.

**Our Approach**: This implementation is optimized to use only **6 essential counters** focused on memory access patterns for hashtables:

### 6 Essential Counters (Active)

1. **CPU_CYCLES** - Overall execution time baseline
2. **INSTRUCTIONS** - For calculating IPC (Instructions Per Cycle)
3. **CACHE_REFERENCES (LLC)** - Last-Level Cache references only (using WhichCache::LL)
4. **CACHE_MISSES (LLC)** - Last-Level Cache misses only (using WhichCache::LL)
5. **LLC_LOADS** - Last-level cache load accesses (same as item 3, kept for backwards compatibility)
6. **LLC_LOAD_MISSES** - LLC load misses (same as item 4, kept for backwards compatibility)

**Note**: CACHE_REFERENCES and CACHE_MISSES have been updated to specifically track LLC (Last-Level Cache) only, using `WhichCache::LL` from the perf-event crate. This ensures precise tracking of LLC behavior rather than aggregate cache statistics across all levels. Items 3-4 and 5-6 measure the same events but are kept as separate counters for backwards compatibility with code that uses both field sets.

These 6 counters provide critical insights into:
- **CPU efficiency** (IPC from cycles/instructions)
- **LLC cache behavior** (precise LLC references/misses using WhichCache::LL)
- **Memory hierarchy performance** (LLC loads and load misses)

### Counters Intentionally Disabled (24 total)

To avoid multiplexing overhead, the following are **not** created:
- Reference CPU cycles
- Branch prediction metrics
- Pipeline stall counters (not supported on all platforms)
- L1 cache metrics (all L1 D-cache and I-cache counters disabled)
- LLC store operations (keeping only loads)
- TLB metrics (all dTLB and iTLB counters disabled)
- Software events (page faults, context switches)

> **Note**: The struct fields for all counters remain in the code for backward compatibility, but only the 6 essential counters are actually created and measured.

## What Gets Measured

The following sections describe all the metrics that **could** be measured. In practice, only the 6 essential counters listed above are active.

### Timing & Duration
- **Wall-clock duration** - Nanosecond precision timing for each operation
- **Timestamp** - When the measurement was taken

### CPU Execution Metrics
- **CPU Cycles** - Total cycles consumed (frequency-dependent)
- **Reference CPU Cycles** - Cycles unaffected by frequency scaling
- **Instructions Retired** - Completed instructions
- **IPC (Instructions Per Cycle)** - Instruction throughput efficiency
- **CPI (Cycles Per Instruction)** - Inverse of IPC

### Branch Prediction
- **Branch Instructions** - Total branch operations
- **Branch Mispredictions** - Failed predictions
- **Branch Miss Rate** - Percentage of mispredicted branches

### Pipeline Stalls
- **Frontend Stalls** - Cycles stalled on instruction fetch/decode
- **Backend Stalls** - Cycles stalled on execution/memory
- **Stall Percentages** - Frontend and backend stalls as % of total cycles

### Memory Hierarchy - Cache Performance

#### Generic Cache Metrics
- **Cache References** - Total cache accesses (all levels)
- **Cache Misses** - Total cache misses (all levels)
- **Overall Cache Miss Rate** - Aggregate miss percentage

#### L1 Data Cache (L1 D-cache)
- **Loads** - Load operations accessing L1 D-cache
- **Load Misses** - L1 D-cache load misses
- **Stores** - Store operations accessing L1 D-cache  
- **Store Misses** - L1 D-cache store misses
- **Load/Store Miss Rates** - Separate miss percentages

#### L1 Instruction Cache (L1 I-cache)
- **Loads** - Instruction fetch operations
- **Load Misses** - L1 I-cache misses
- **Miss Rate** - Instruction cache miss percentage

#### Last-Level Cache (LLC)
- **Loads** - LLC load accesses
- **Load Misses** - LLC load misses
- **Stores** - LLC store accesses
- **Store Misses** - LLC store misses
- **Overall LLC Miss Rate** - Combined load/store miss percentage

### TLB (Translation Lookaside Buffer) Performance

#### Data TLB (dTLB)
- **Load Accesses** - Virtual-to-physical translations for loads
- **Load Misses** - dTLB load misses (page table walk required)
- **Store Accesses** - Virtual-to-physical translations for stores
- **Store Misses** - dTLB store misses
- **Overall dTLB Miss Rate** - Combined miss percentage

#### Instruction TLB (iTLB)
- **Accesses** - Instruction address translations
- **Misses** - iTLB misses requiring page table walk
- **Miss Rate** - Instruction TLB miss percentage

### Software Events & System-Level Metrics
- **Page Faults** - Total page faults (minor + major)
- **Minor Page Faults** - Resolved without disk I/O
- **Major Page Faults** - Required disk I/O to resolve
- **Context Switches** - Task scheduler context switches
- **CPU Migrations** - Thread migrated to different CPU core

### Derived Analytics
- **Memory Access Patterns** - Load vs store breakdowns
- **Cache Hierarchy Behavior** - Multi-level miss correlation
- **TLB Efficiency** - Translation overhead analysis
- **Pipeline Efficiency** - Stall attribution and IPC analysis

## Requirements

### Feature Flag
- **`hw_perf_counters`** - Must be enabled in Cargo.toml or via command line

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
sudo cargo run --example hw_perf_demo --no-default-features --features "hashbrown_dram,hw_perf_counters"
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

## Debugging Counter Issues

If performance counters are returning all zeros or `None`, use the debug mode to diagnose the issue:

### Common Issue: Counters Return Zero Despite Being Created

**Symptom**: Debug output shows counters are created successfully (✓) but all values are 0.

**Root Cause**: When counters are part of a `Group`, they must be read using `group.read()` to get a `Counts` object, then individual counter values are retrieved by indexing into the `Counts` object. Calling `.read()` directly on individual counters does not work for grouped counters.

**Fix Applied**: The implementation now correctly uses `group.read()` and retrieves values with `counts.get(counter)`. This follows the proper perf-event API usage as demonstrated in the crate's examples.

**Example of Correct Usage**:
```rust
// Enable and run the operation
group.enable()?;
// ... do work ...
group.disable()?;

// Read all counters from the group (CORRECT)
let counts = group.read()?;

// Get individual counter values (CORRECT)
if let Some(ref cycles_counter) = cycles {
    let cycles_value = counts.get(cycles_counter).copied().unwrap_or(0);
}
```

**Example of Incorrect Usage** (will return 0):
```rust
// WRONG: Reading individual counter directly
if let Some(mut cycles_counter) = cycles {
    let cycles_value = cycles_counter.read()?; // This doesn't work for grouped counters!
}
```

### Debug Mode

Set the `PAPER_CACHE_DEBUG_PERF` environment variable to enable debug output:

#### Basic Debug Info
```bash
PAPER_CACHE_DEBUG_PERF=1 cargo run --example hw_perf_demo --features hw_perf_counters
```

Output shows (optimized to 6 essential counters):
```
[DEBUG] Successfully created performance counter group
[DEBUG] Performance counter creation summary (limited to 6 essential counters):
[DEBUG] 
[DEBUG] === ESSENTIAL COUNTERS (attempting to create) ===
[DEBUG]   ✓ CPU_CYCLES
[DEBUG]   ✓ INSTRUCTIONS
[DEBUG]   ✓ CACHE_REFERENCES (LLC)
[DEBUG]   ✓ CACHE_MISSES (LLC)
[DEBUG]   ✓ LLC_LOADS
[DEBUG]   ✓ LLC_LOAD_MISSES
[DEBUG] 
[DEBUG] Counter creation complete: 6 enabled, 24 disabled (to avoid multiplexing)
[DEBUG] Note: 24 counters are intentionally disabled to stay within hardware limits (~6-8 counters)
```

#### Verbose Debug Info (Detailed Errors)
```bash
PAPER_CACHE_DEBUG_PERF=verbose cargo run --example hw_perf_demo --features hw_perf_counters
```

Output shows detailed error for each of the 6 essential counters (not all 30):
```
[DEBUG] Successfully created performance counter group
[DEBUG]   Counter CPU_CYCLES failed: No such file or directory (os error 2)
[DEBUG]   Counter INSTRUCTIONS failed: No such file or directory (os error 2)
[DEBUG]   Counter CACHE_REFERENCES (LLC) failed: No such file or directory (os error 2)
... (6 counters total, not 30)
```

### Common Error Messages

#### "Permission denied (os error 13)"
**Cause**: Insufficient permissions to access perf_event subsystem.  
**Solution**: 
- Run with `sudo sysctl kernel.perf_event_paranoid=-1`
- Or run your program with `sudo`
- Or add CAP_PERFMON capability

#### "No such file or directory (os error 2)"
**Cause**: Hardware performance counters not available (common in VMs/containers).  
**Solution**:
- Run on bare metal hardware for full counter support
- Some VMs support PMU passthrough - check hypervisor settings
- Note: Even when unavailable, only the 6 essential counters are attempted (not all 30)

#### "Failed to create performance counter group"
**Cause**: The perf_event subsystem couldn't create a group.  
**Solution**: Check if you're in a restricted environment (container, VM) and verify perf_event_paranoid settings.

### Why Only 6 Counters?

Hardware performance monitoring units (PMUs) typically support only 4-8 simultaneous counters. When you request more:
- **Kernel multiplexes** counters (time-slices them)
- **Measurement overhead** increases
- **Accuracy decreases** due to time-slicing
- **Results become less reliable**

By limiting to 6 essential memory-focused counters, we ensure:
- ✓ **No multiplexing** - all counters run simultaneously
- ✓ **Maximum accuracy** - no time-slicing artifacts  
- ✓ **Consistent measurements** - no scheduling effects
- ✓ **Focus on memory** - counters chosen for hashtable analysis

### Debug Output Control

- **No env var**: Clean output, no debug messages (production mode)
- **PAPER_CACHE_DEBUG_PERF=1**: Summary of counter availability
- **PAPER_CACHE_DEBUG_PERF=verbose**: Detailed per-counter failure reasons

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

### Execution Metrics

#### CPU Cycles
- **Lower is better** - Fewer cycles means faster operation
- **Typical values**:
  - GET on hot cache: 50-200 cycles
  - GET with cache miss: 200-1000+ cycles
  - SET operation: 100-500 cycles
- **Analysis**: Compare across configurations to identify performance bottlenecks

#### IPC (Instructions Per Cycle)
- **Higher is better** - More instructions executed per cycle
- **Typical values**:
  - Modern CPUs: 1.5-2.5 IPC
  - Memory-bound code: 0.5-1.0 IPC
  - Compute-bound code: 2.0-4.0 IPC
- **Analysis**: Low IPC (<1.0) indicates memory stalls or data dependencies

#### Branch Prediction
- **Lower miss rate is better** - Better prediction accuracy
- **Typical values**:
  - Well-predicted branches: <5% miss rate
  - Random/unpredictable: 20-50% miss rate
- **Analysis**: High miss rates indicate unpredictable control flow

#### Pipeline Stalls
- **Lower percentage is better** - Less time waiting
- **Frontend stalls**: Instruction fetch/decode bottleneck
  - Check: I-cache misses, instruction bandwidth
- **Backend stalls**: Execution/memory bottleneck
  - Check: D-cache misses, TLB misses, data dependencies
- **Analysis**: Helps identify whether bottleneck is in instruction supply or data/execution

### Cache Hierarchy

#### L1 Data Cache
- **Lower miss rate is better** - More hits in fastest cache
- **Typical values**:
  - Sequential access: 1-5% miss rate
  - Random access: 10-30% miss rate
  - Large working set: 30-60% miss rate
- **Analysis**: 
  - High load misses: Poor data locality, consider prefetching
  - High store misses: Write-heavy workload, may benefit from write combining

#### L1 Instruction Cache
- **Lower miss rate is better** - Better instruction locality
- **Typical values**:
  - Tight loops: <1% miss rate
  - Large code: 5-15% miss rate
- **Analysis**: High miss rates indicate code size issues or poor branch prediction

#### LLC (Last-Level Cache)
- **Lower miss rate is critical** - LLC misses go to DRAM/PMEM
- **Typical values**:
  - Hot working set: <5% miss rate
  - Cold/large data: 20-80% miss rate
- **Analysis**: LLC miss = memory access (200-400 cycles penalty)
- **DRAM vs PMEM**: PMEM shows much higher latency on LLC misses

### TLB Performance

#### dTLB (Data TLB)
- **Lower miss rate is better** - Fewer page table walks
- **Typical values**:
  - Sequential access: <1% miss rate
  - Random access: 5-20% miss rate
  - Large pages: Much lower miss rate
- **Analysis**: 
  - High miss rates indicate scattered memory access
  - Consider using huge pages (2MB/1GB) to reduce misses

#### iTLB (Instruction TLB)
- **Lower miss rate is better** - Efficient instruction fetching
- **Typical values**: Usually <1% for typical code
- **Analysis**: High miss rates indicate large code footprint across many pages

### System Events

#### Page Faults
- **Lower is better** - Indicates stable memory resident set
- **Minor faults**: Acceptable, just mapping new pages
- **Major faults**: Bad, requires disk I/O (very slow)
- **Analysis**: 
  - Many minor faults: Application warming up
  - Major faults: Memory pressure, swapping occurring

#### Context Switches
- **Lower is better** - Less overhead from task switching
- **Analysis**: High counts indicate CPU contention or I/O waiting

#### CPU Migrations
- **Lower is better** - Thread stays on same CPU (better cache locality)
- **Analysis**: Many migrations indicate CPU scheduler thrashing or poor affinity

### Performance Analysis Workflows

#### Identifying Memory Bottlenecks
1. Check **IPC**: Low (<1.0) suggests memory-bound
2. Check **Backend Stalls**: High % confirms memory bottleneck
3. Check **LLC Miss Rate**: High rate = frequent DRAM/PMEM access
4. Check **dTLB Miss Rate**: High rate adds translation overhead
5. **Action**: Improve data locality, use prefetching, optimize layout

#### Identifying Instruction Bottlenecks
1. Check **Frontend Stalls**: High % suggests instruction supply issue
2. Check **I-cache Miss Rate**: High rate confirms instruction fetch problem
3. Check **iTLB Miss Rate**: High rate adds translation overhead
4. **Action**: Reduce code size, improve branch prediction, optimize loops

#### DRAM vs PMEM Comparison
1. **Cycles**: PMEM should show 2-4x higher cycles for LLC misses
2. **LLC Miss Rate**: May differ due to different access patterns
3. **dTLB Performance**: PMEM may show different TLB behavior
4. **Duration**: Wall-clock time reflects real latency differences
5. **Analysis**: Use to quantify PMEM performance impact

### What to Look For

#### High Performance (Good)
- ✅ IPC > 1.5
- ✅ L1 D-cache miss rate < 10%
- ✅ LLC miss rate < 10%
- ✅ TLB miss rates < 1%
- ✅ Frontend/backend stalls < 20% each
- ✅ No major page faults

#### Performance Issues (Bad)
- ❌ **High cache miss rate (>20%)**: Consider data structure optimization, prefetching, reduce size
- ❌ **Low IPC (<1.0)**: Memory-bound, cache misses causing stalls, data dependencies
- ❌ **High TLB miss rate (>5%)**: Scattered memory access, consider huge pages
- ❌ **High backend stalls (>50%)**: Memory/execution bottleneck, check cache and TLB
- ❌ **High frontend stalls (>30%)**: Instruction supply bottleneck, check I-cache
- ❌ **Major page faults**: Memory pressure, system swapping
- ❌ **Many context switches**: CPU contention
- ❌ **Many CPU migrations**: Poor thread affinity

## Metrics Collection Overhead

The hardware performance counters have **minimal overhead**:
- ~10-50 nanoseconds per measurement
- No impact on measured code execution (counters run in parallel)
- Negligible memory usage (small counter structures)
- Thread-local counters avoid synchronization overhead

**Best Practices**:
- For production: Measure in batches, not every operation
- For development: Measure all operations for detailed analysis
- For benchmarking: Use with consistent environment (fixed CPU frequency, isolated cores)

## Metric Units and Definitions

| Metric | Unit | Definition | Collection Method |
|--------|------|------------|-------------------|
| Duration | nanoseconds | Wall-clock time | `std::time::Instant` |
| Cycles | count | CPU clock cycles | `PERF_COUNT_HW_CPU_CYCLES` |
| Instructions | count | Retired instructions | `PERF_COUNT_HW_INSTRUCTIONS` |
| Cache References | count | All cache accesses | `PERF_COUNT_HW_CACHE_REFERENCES` |
| Cache Misses | count | All cache misses | `PERF_COUNT_HW_CACHE_MISSES` |
| Branch Instructions | count | Branch operations | `PERF_COUNT_HW_BRANCH_INSTRUCTIONS` |
| Branch Misses | count | Mispredicted branches | `PERF_COUNT_HW_BRANCH_MISSES` |
| L1 D-cache Loads | count | L1 data load accesses | `PERF_COUNT_HW_CACHE_L1D_READ_ACCESS` |
| L1 D-cache Stores | count | L1 data store accesses | `PERF_COUNT_HW_CACHE_L1D_WRITE_ACCESS` |
| LLC Loads/Stores | count | Last-level cache accesses | `PERF_COUNT_HW_CACHE_LL_READ/WRITE` |
| dTLB Loads/Stores | count | Data TLB translations | `PERF_COUNT_HW_CACHE_DTLB_READ/WRITE` |
| iTLB Loads | count | Instruction TLB translations | `PERF_COUNT_HW_CACHE_ITLB_READ` |
| Page Faults | count | All page faults | `PERF_COUNT_SW_PAGE_FAULTS` |
| Context Switches | count | Task switches | `PERF_COUNT_SW_CONTEXT_SWITCHES` |
| CPU Migrations | count | Thread migrated to new CPU | `PERF_COUNT_SW_CPU_MIGRATIONS` |

## Output Example

With the enhanced metrics, you'll see output like:

```
=== Hardware Performance Counter Statistics ===

Global HashMap (hashbrown in DRAM):
Hardware Performance Statistics (HashMap):
  Total Operations: 1000
  Total Cycles: 45230000
  Total Cache References: 850000
  Total Cache Misses: 85000 (10.00% miss rate)

GET Operations (100 calls):
  ┌─ Execution Metrics:
  │  Duration: 1.23 µs avg
  │  Cycles: 2500 avg, 250000 total
  │  Instructions: 4800 avg (IPC: 1.92)
  │  Branches: 180 avg, 5 mispredictions (2.78% miss rate)
  │  Stalls: Frontend 8.5%, Backend 15.2%
  ├─ Cache Hierarchy:
  │  Overall: 125 refs, 12 misses (9.60% miss rate)
  │  L1 D-cache:
  │    Loads: 85 avg, 8 misses (9.41% miss rate)
  │    Stores: 25 avg, 2 misses (8.00% miss rate)
  │  L1 I-cache: 200 loads, 2 misses (1.00% miss rate)
  │  LLC:
  │    Loads: 15 avg, 2 misses
  │    Stores: 5 avg, 0 misses
  │    Overall: 10.00% miss rate
  ├─ TLB Performance:
  │  dTLB: 110 accesses, 1 misses (0.91% miss rate)
  │  iTLB: 200 accesses, 0 misses (0.00% miss rate)
  └─ System Events:
     Page Faults: 0 total (0 minor, 0 major)
     Context Switches: 0 avg
     CPU Migrations: 0 avg

...
```

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
cargo run --example hw_perf_demo --no-default-features --features "hashbrown_dram,hw_perf_counters"
```

### Example 2: Compare DRAM vs PMEM
```bash
# DRAM version
cargo run --example hw_perf_demo --no-default-features --features "hashbrown_dram,hw_perf_counters" > dram_results.txt

# PMEM version (requires nightly + PMEM hardware)
cargo +nightly run --example hw_perf_demo --no-default-features --features "global_hashtable_pmem,hw_perf_counters" > pmem_results.txt

# Compare results
diff dram_results.txt pmem_results.txt
```

### Example 3: Both Software and Hardware Counters
```bash
# Run with both counter types enabled
cargo run --example hw_perf_demo --no-default-features --features "hashbrown_dram,perf_counters,hw_perf_counters"
```

## References

- [Linux perf_event documentation](https://man7.org/linux/man-pages/man2/perf_event_open.2.html)
- [perf-event Rust crate](https://docs.rs/perf-event/)
- [Intel Performance Counter Monitor](https://software.intel.com/content/www/us/en/develop/articles/intel-performance-counter-monitor.html)
- [Brendan Gregg's perf examples](http://www.brendangregg.com/perf.html)
