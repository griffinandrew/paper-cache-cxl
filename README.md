# paper-cache

PaperCache is an in-memory cache which supports the dynamic switching between any eviction policy at runtime.

Note: this crate should not be used directly; please use the paper-server crate instead.

This branch makes it such that both the key and value reside in the pmem tier.

Here we will add a custom basic hashmap because the architecture of the swissmap used by both dashmap and hashbrown hashmap might be hurting perf in a pmem scenario

This branch will add the needed instrumentation to count how many memory accesses are performed to the hashmap

add support for tiering the hash table acorss memory levels

stage copy of current working branch.... 

## Performance Counters

PaperCache includes **two types** of performance counters to track memory access patterns for hashmap structures with both DRAM and PMEM configurations.

**Both counter types are optional and gated behind feature flags:**
- `perf_counters` - Enable software performance counters
- `hw_perf_counters` - Enable hardware performance counters (requires Linux)

### 1. Software Performance Counters (Feature: `perf_counters`)

Track high-level hashmap operations:
- **Atomic counters** for thread-safe tracking of hashmap operations
- **Read tracking**: `get`, `has`, `peek` operations (lookups)
- **Write tracking**: `insert`, `remove` operations (insertions, deletions)
- **Operation breakdown**: Separate counters for different operation types
- **Feature-aware**: Automatically tracks the correct hashmap based on enabled features

### 2. Hardware Performance Counters (Feature: `hw_perf_counters`)

Track comprehensive microarchitectural and memory hierarchy statistics using Linux `perf_event`:

**Execution Metrics:**
- **CPU cycles** and **instructions** retired per operation
- **IPC (Instructions Per Cycle)** and **CPI** metrics
- **Branch prediction**: Instructions, mispredictions, and miss rates
- **Pipeline stalls**: Frontend (fetch/decode) and backend (execution/memory) stalls

**Memory Hierarchy:**
- **L1 D-cache**: Load/store accesses and misses with separate miss rates
- **L1 I-cache**: Instruction fetch accesses and misses
- **LLC (Last-Level Cache)**: Load/store accesses and misses
- **Overall cache**: References, misses, and aggregate miss rates

**TLB Performance:**
- **dTLB**: Data translation misses for loads and stores
- **iTLB**: Instruction translation misses
- **Miss rates**: Separate for each TLB type

**System Events:**
- **Page faults**: Total, minor (no I/O), and major (disk I/O)
- **Context switches**: Task scheduler switches
- **CPU migrations**: Thread moved to different core

**Timing:**
- **Duration tracking**: Nanosecond precision wall-clock timing
- **Timestamps**: When each measurement was taken

**Per-operation statistics**: All metrics available as averages and totals for GET/SET/DEL/HAS operations

### Supported Configurations

Performance counters are available for:
- `hashbrown_dram`: hashbrown HashMap in DRAM
- `global_hashtable_pmem`: hashbrown HashMap in PMEM
- Future: `global_flatmap_dram`, `global_flatmap_pmem`

### Usage

#### Software Counters

```rust
use paper_cache::{PaperCache, PaperPolicy};

// Create a cache
let cache = PaperCache::<u64, Box<[u8]>>::new(
    10_000_000,
    &[PaperPolicy::Lru],
    PaperPolicy::Lru,
)?;

// Perform operations
cache.set(1, b"value", None)?;
cache.get(&1)?;
cache.has(&1);
cache.del(&1)?;

// Print statistics
paper_cache::perf_counters::print_perf_stats();

// Or access programmatically
if let Some(stats) = paper_cache::perf_counters::get_hashmap_stats() {
    println!("Total accesses: {}", stats.total_accesses);
    println!("Reads: {}", stats.reads);
    println!("Writes: {}", stats.writes);
}
```

#### Hardware Counters

```rust
use paper_cache::{PaperCache, PaperPolicy, measure_operation};

let cache = PaperCache::<u64, Box<[u8]>>::new(10_000_000, &[PaperPolicy::Lru], PaperPolicy::Lru)?;

// Measure a GET operation with hardware counters
let (result, hw_measurement) = measure_operation(|| cache.get(&key));

if let Some(measurement) = hw_measurement {
    println!("Cycles: {}, Cache misses: {}", 
             measurement.cycles, measurement.cache_misses);
    println!("Cache miss rate: {:.2}%", measurement.cache_miss_rate());
}

// Print aggregated hardware statistics
paper_cache::print_hw_perf_stats();
```

### Running the Examples

```bash
# Software counters demo (requires perf_counters feature)
cargo run --example perf_counters_demo --no-default-features --features "hashbrown_dram,perf_counters"

# Hardware counters demo (requires hw_perf_counters feature and Linux perf_event access)
cargo run --example hw_perf_demo --no-default-features --features "hashbrown_dram,hw_perf_counters"

# Both counters together
cargo run --example hw_perf_demo --no-default-features --features "hashbrown_dram,perf_counters,hw_perf_counters"

# With hashbrown in PMEM (requires nightly + PMEM hardware)
cargo +nightly run --example hw_perf_demo --no-default-features --features "global_hashtable_pmem,hw_perf_counters"
```

**Note**: Hardware performance counters require Linux `perf_event` access. If running in a container or without sufficient permissions, you may need to:
```bash
# Allow non-root access to performance counters
sudo sysctl kernel.perf_event_paranoid=-1

# Or run with sudo
sudo cargo run --example hw_perf_demo --no-default-features --features "hashbrown_dram,hw_perf_counters"
```

### Output Example

#### Software Counters
```
=== PaperCache Performance Statistics ===

Global HashMap (hashbrown in DRAM):
HashMap Performance Statistics:
  Total Accesses: 185
  Reads: 75 (40.5%)
    - Lookups: 75
    - Iterations: 0
  Writes: 110 (59.5%)
    - Insertions: 100
    - Deletions: 10
    - Clears: 0
```

#### Hardware Counters
```
=== Hardware Performance Counter Statistics ===

Global HashMap (hashbrown in DRAM):
Hardware Performance Statistics (HashMap):
  Total Operations: 100
  Total Cycles: 250000
  Total Cache References: 12500
  Total Cache Misses: 1200 (9.60% miss rate)

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
```

## Tiering Manager

The tiering manager provides a two-tier caching architecture with **actual data copies**:
- **Far Tier (PMEM)**: All objects are stored in persistent memory by default (source of truth)
- **Near Tier (DRAM)**: Hot objects are **physically copied** to DRAM for faster access

### Features

- **Automatic Promotion with Data Copying**: Objects accessed frequently are automatically **copied** to DRAM
- **Two-Tier Reads**: Get operations check DRAM cache first, then fall back to PMEM
- **Configurable Thresholds**: 
  - DRAM capacity (default: 20% of cache size)
  - Hotness threshold (default: 2 accesses before promotion)
- **Runtime Controls**: Adjust tiering parameters at runtime without restart
- **Statistics Tracking**: Monitor promotions, demotions, and tier distribution
- **Strong Consistency**: Updates and deletions are applied to both tiers immediately

### API

```rust
use paper_cache::{PaperCache, PaperPolicy};

let cache = PaperCache::<u32, Box<[u8]>>::new(
    10_000_000,  // 10 MB cache
    &[PaperPolicy::Lfu],
    PaperPolicy::Lfu,
).unwrap();

// Get tiering statistics
let stats = cache.tiering_stats();
println!("DRAM objects: {}", stats.dram_objects);
println!("Promotions: {}", stats.promotions);

// Configure DRAM tier size (in bytes)
cache.set_dram_threshold(5_000_000);  // 5 MB for hot objects

// Configure hotness threshold
cache.set_hotness_threshold(3);  // Promote after 3 accesses
```

### How It Works

1. **Object Storage**: All objects are initially stored in PMEM (far tier)
2. **Access Tracking**: Each `get()` operation increments the object's access count
3. **Promotion with Data Copy**: When an object's access count reaches the hotness threshold, its data is **physically copied** to a DRAM cache
4. **Fast Reads**: Subsequent `get()` operations check the DRAM cache first for hot objects, providing faster access
5. **Eviction**: When DRAM reaches capacity, the least recently used objects are demoted (DRAM copies removed, PMEM copies retained)
6. **Consistency**: All updates write to PMEM and update DRAM copies if they exist; deletions remove from both tiers

### Background Workers

The tiering manager runs as a worker thread alongside the policy and TTL workers, processing events from the cache and making promotion/demotion decisions periodically (every 5 seconds).

### Data Copy Model

Unlike simple metadata tracking, this implementation maintains **two physical copies** of hot objects:
- **PMEM copy**: Always exists, serves as source of truth
- **DRAM copy**: Created on promotion, provides fast access, removed on demotion

This ensures hot objects benefit from DRAM speed while maintaining data durability in PMEM.

## Feature Flags for Memory Tier Configuration

PaperCache provides fine-grained control over memory placement through feature flags:

### Tiering Manager Control

- **`enable_tiering_manager`**: Enables the tiering manager functionality
  - When enabled: Automatic promotion/demotion between DRAM and PMEM tiers
  - When disabled: No automatic tiering, but global hashtable can still be placed in PMEM

### Hashtable Memory Placement

Independent control over where hashtables are stored (requires one of the allocator features):

- **`tiering_hashtable_pmem`**: Places the tiering manager's internal hashtable in persistent memory
  - Requires: `enable_tiering_manager` + one of (`key_value_pmem`, `alloc_api_exp`)
  - When disabled: Tiering hashtable uses DRAM

- **`global_hashtable_pmem`**: Places the main cache hashtable in persistent memory
  - Requires: One of (`key_value_pmem`, `alloc_api_exp`)
  - Works independently of tiering manager
  - When disabled: Global hashtable uses DRAM

### Usage Examples

#### All combinations with tiering enabled:

```toml
# Both hashtables in PMEM (maximum persistence)
[dependencies]
paper-cache = { features = ["enable_tiering_manager", "tiering_hashtable_pmem", "global_hashtable_pmem", "key_value_pmem"] }

# Only global hashtable in PMEM (cache data persistent, tiering metadata in DRAM)
[dependencies]
paper-cache = { features = ["enable_tiering_manager", "global_hashtable_pmem", "key_value_pmem"] }

# Only tiering hashtable in PMEM (tiering metadata persistent, cache data in DRAM)
[dependencies]
paper-cache = { features = ["enable_tiering_manager", "tiering_hashtable_pmem", "key_value_pmem"] }

# Neither in PMEM (all in DRAM, baseline performance)
[dependencies]
paper-cache = { features = ["enable_tiering_manager", "key_value_pmem"] }
```

#### Without tiering manager:

```toml
# Global hashtable in PMEM without automatic tiering
[dependencies]
paper-cache = { features = ["global_hashtable_pmem", "key_value_pmem"] }
```

### Performance Considerations

- **PMEM hashtables**: Slower access but persistent across restarts
- **DRAM hashtables**: Faster access but volatile
- **Tiering manager off + global in PMEM**: Simplest persistent cache without automatic promotion/demotion overhead

### Required Base Features

These feature flags work in conjunction with:
- `key_value_pmem`: Place key and value data in PMEM
- `alloc_api_exp`: Experimental allocator API for testing

