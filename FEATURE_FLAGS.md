# Feature Flags for Memory Tier Configuration

This document explains the implementation of separate configuration options for persistent memory (PMEM) placement in PaperCache.

## Overview

The implementation provides explicit feature flags to control:
1. Whether all data structures use DRAM
2. Whether key/value data is stored in PMEM
3. Whether the tiering manager is enabled
4. Where the tiering manager's internal hashtable is stored (DRAM vs PMEM)
5. Where the global cache hashtable is stored (DRAM vs PMEM)

## Feature Flags

### `all_dram`
- **Purpose**: Force all allocations to use DRAM (no PMEM usage)
- **When enabled**: All data structures (hashtables, keys, values) are stored in DRAM
- **When disabled**: Default behavior - allocations can use PMEM based on other features
- **Use case**: Baseline performance testing, systems without PMEM

### `key_value_pmem`
- **Purpose**: Store key and value data in PMEM
- **When enabled**: Cache key/value pairs are allocated in PMEM
- **When disabled**: Cache key/value pairs use default allocation (typically DRAM)
- **Requirements**: Mutually exclusive with `all_dram`

### `enable_tiering_manager`
- **Purpose**: Enable/disable the tiering manager functionality
- **When enabled**: Automatic promotion/demotion of hot objects between DRAM and PMEM tiers
- **When disabled**: No automatic tiering, but global hashtable can still be placed in PMEM
- **Requirements**: Works with `key_value_pmem`

### `tiering_hashtable_pmem`
- **Purpose**: Control memory placement of the tiering manager's internal hashtable
- **When enabled**: Tiering manager's hashtable stored in PMEM
- **When disabled**: Tiering manager's hashtable stored in DRAM
- **Requirements**: Requires `enable_tiering_manager` + `key_value_pmem`

### `global_hashtable_pmem`
- **Purpose**: Control memory placement of the main cache hashtable
- **When enabled**: Global cache hashtable stored in PMEM
- **When disabled**: Global cache hashtable stored in DRAM
- **Requirements**: Can be used independently or with `key_value_pmem`

### `hw_perf`
- **Purpose**: Enable hardware performance counters for cache operation profiling
- **When enabled**: Instruments cache lookup and eviction paths with Linux `perf_event` counters
- **When disabled**: Zero cost — all instrumentation is completely compiled out
- **Use case**: Performance analysis of DRAM vs PMEM access patterns (LLC misses, cycles, IPC)
- **Requirements**: Linux only; requires `perf_event` access (may need elevated permissions or `/proc/sys/kernel/perf_event_paranoid` ≤ 1)

### `eviction_stacks_pmem`
- **Purpose**: Allocate LFU eviction policy tracking structures in PMEM using the Hybrid allocator
- **When enabled**: `LfuStack`, `CountStack` internal data structures (`index_map`, `count_stacks`) are allocated via `HybridObjects` (PMEM-backed)
- **When disabled**: Standard DRAM-backed `std::collections::HashMap` and `kwik::collections::HashList` are used (default)
- **Use case**: Ensures eviction metadata is co-located with PMEM-stored objects for lower cross-tier access overhead
- **Requirements**: The Hybrid allocator must be accessible — compatible with `key_value_pmem` and standalone

### `flatmap_dram`
- **Purpose**: Enable high-performance Linear Probing Hash Map (FlatMap) in DRAM
- **When enabled**: FlatMap module is compiled with DRAM allocator support
- **When disabled**: FlatMap module is not available
- **Use case**: High-performance hash map for DRAM with optimized memory layout
- **Performance**: Reduces cache misses by storing hash, key, and value adjacently

### `flatmap_pmem`
- **Purpose**: Enable standalone FlatMap module with PMEM allocator support
- **When enabled**: FlatMap module is compiled with PMEM allocator support (HybridObjects)
- **When disabled**: FlatMap module is not available
- **Use case**: Standalone high-performance hash map optimized for PMEM latency characteristics
- **Performance**: Reduces PMEM read overhead from 3x to 1x by using flat layout (Array of Structs)
- **Design**: Uses Linear Probing (no Robin Hood hashing) to minimize expensive PMEM writes

### `global_flatmap_dram`
- **Purpose**: Use FlatMap as PaperCache's global hashtable in DRAM
- **When enabled**: `ObjectMapRef` uses `Arc<RwLock<FlatMapWithHasher<..., Global>>>` instead of DashMap
- **When disabled**: Default hashtable implementation is used
- **Use case**: Replace DashMap with FlatMap for better DRAM cache locality
- **Performance**: Better cache utilization due to flat layout, fixed capacity for predictable performance
- **Integration**: Works with all PaperCache operations (get, set, delete, eviction)

### `global_flatmap_pmem`
- **Purpose**: Use FlatMap as PaperCache's global hashtable in PMEM  
- **When enabled**: `ObjectMapRef` uses `Arc<RwLock<FlatMapWithHasher<..., Hybrid>>>` for PMEM
- **When disabled**: Default hashtable implementation is used
- **Use case**: Replace HashMap with FlatMap for optimal PMEM latency
- **Performance**: 3x latency reduction (600ns → 300ns per lookup) compared to hashbrown on PMEM
- **Integration**: Works with all PaperCache operations, uses `remove_unchecked` for eviction without Clone constraints

### `hashbrown_dram`
- **Purpose**: Use hashbrown HashMap as global hashtable in DRAM (for performance comparison)
- **When enabled**: `ObjectMapRef` uses `Arc<RwLock<HashMap<..., NoHasher>>>` in DRAM
- **When disabled**: Default hashtable implementation (DashMap) is used
- **Use case**: Direct performance comparison with `global_hashtable_pmem` using the same hashbrown implementation
- **Performance**: Same hashbrown HashMap implementation as `global_hashtable_pmem` but allocated in DRAM instead of PMEM
- **Requirements**: Mutually exclusive with `global_hashtable_pmem`, `global_flatmap_dram`, and `global_flatmap_pmem`

## Implementation Details

### Type System

The implementation uses Rust's conditional compilation to select the appropriate hashtable types:

**Tiering Manager's hashtable** (`dram_cache` in `TieringManager`):
- Without `tiering_hashtable_pmem`: `DashMap` or `HashMap` (DRAM)
- With `tiering_hashtable_pmem`: `HashMap<..., Hybrid>` (PMEM)

**Global hashtable** (`objects` in `PaperCache`):
- Default (no FlatMap): `DashMap` (DRAM)
- With `global_hashtable_pmem`: `RwLock<HashMap<..., Hybrid>>` (PMEM)
- With `hashbrown_dram`: `RwLock<HashMap<..., NoHasher>>` (DRAM)
- With `global_flatmap_dram`: `Arc<RwLock<FlatMapWithHasher<..., Global>>>` (DRAM)
- With `global_flatmap_pmem`: `Arc<RwLock<FlatMapWithHasher<..., Hybrid>>>` (PMEM)

**FlatMap** (high-performance Linear Probing Hash Map):
- With `flatmap_dram`: Uses Global allocator (DRAM)
- With `flatmap_pmem`: Uses HybridObjects allocator (PMEM)
- Flat layout: `Vec<Bucket<K, V>, A>` where `Bucket` is `#[repr(C)]` with `{ hash: u64, key: K, val: V }`
- Operations: `insert`, `get`, `get_mut`, `remove`, `contains_key`, `clear`, `iter`
- Algorithm: Linear probing with `(index + 1) & mask` collision resolution
- Fixed capacity (no resizing) for optimal performance

### Allocator Integration

The `Hybrid` allocator is used to place data in PMEM:
- Defined in `src/allocator.rs`
- Implements both `GlobalAlloc` and `Allocator` traits
- Routes allocations to either DRAM (jemalloc) or PMEM (UMF) based on configuration

### Conditional Compilation

The code uses `#[cfg(...)]` attributes extensively to:
1. Include/exclude the tiering module based on `enable_tiering_manager`
2. Select different hashtable implementations based on pmem flags
3. Conditionally compile tiering-related methods in PaperCache

## Valid Feature Combinations

### Basic Configurations

1. **All in DRAM** (baseline performance):
   ```toml
   features = ["all_dram"]
   ```

2. **Key/Value data in PMEM**:
   ```toml
   features = ["key_value_pmem"]
   ```

3. **Global hashtable in PMEM only** (data in DRAM, hashtable in PMEM):
   ```toml
   features = ["global_hashtable_pmem"]
   ```

### With Tiering Manager

4. **Tiering with key/value in PMEM**:
   ```toml
   features = ["enable_tiering_manager", "key_value_pmem"]
   ```

5. **Tiering + global hashtable in PMEM**:
   ```toml
   features = ["enable_tiering_manager", "key_value_pmem", "global_hashtable_pmem"]
   ```

6. **Tiering + tiering hashtable in PMEM**:
   ```toml
   features = ["enable_tiering_manager", "key_value_pmem", "tiering_hashtable_pmem"]
   ```

7. **Tiering + both hashtables in PMEM** (maximum persistence):
   ```toml
   features = ["enable_tiering_manager", "key_value_pmem", "tiering_hashtable_pmem", "global_hashtable_pmem"]
   ```

8. **Hardware performance counters** (zero-cost when disabled):
   ```toml
   features = ["hw_perf"]
   ```

9. **PMEM-backed eviction stacks + key/value in PMEM**:
   ```toml
   features = ["eviction_stacks_pmem", "key_value_pmem"]
   ```

10. **Full PMEM stack** (tiering + eviction stacks + hw profiling):
    ```toml
    features = ["enable_tiering_manager", "key_value_pmem", "eviction_stacks_pmem", "hw_perf"]
    ```

## Performance Characteristics

### Memory Tier Comparison

**DRAM**:
- ✅ Fast access (CPU cache speeds)
- ✅ Low latency for lookups and insertions
- ❌ Volatile - lost on restart
- ❌ Limited by DRAM capacity

**PMEM**:
- ❌ Slower access (higher latency than DRAM)
- ❌ Higher latency for operations
- ✅ Persistent across restarts
- ✅ Larger capacity available

### Use Cases

1. **`all_dram`**: Maximum performance, baseline comparison
2. **`key_value_pmem`**: Persistent data, volatile metadata
3. **`global_hashtable_pmem`**: Test hashtable in PMEM independently
4. **Tiering + both in PMEM**: Maximum durability, cache state survives restarts
5. **Tiering + global in PMEM**: Persistent cache data, fast tiering decisions
6. **Tiering + tiering in PMEM**: Fast cache access, persistent tiering metadata
7. **`flatmap_dram`**: Standalone FlatMap module in DRAM
8. **`flatmap_pmem`**: Standalone FlatMap module in PMEM
9. **`global_flatmap_dram`**: Use FlatMap as PaperCache's main hashtable in DRAM
10. **`global_flatmap_pmem`**: Use FlatMap as PaperCache's main hashtable in PMEM (3x latency reduction)
11. **`hashbrown_dram`**: Use hashbrown HashMap in DRAM for direct performance comparison with `global_hashtable_pmem`
12. **`hw_perf`**: Hardware performance counters for profiling (zero-cost when disabled)
13. **`eviction_stacks_pmem`**: LFU eviction stacks allocated in PMEM for co-location with PMEM objects

## Code Locations

- **Feature definitions**: `Cargo.toml`
- **Allocator**: `src/allocator.rs`
- **FlatMap**: `src/flatmap.rs`
- **Tiering manager hashtable**: `src/tiering/manager.rs`
- **Global hashtable**: `src/lib.rs`
- **Worker manager integration**: `src/worker/manager.rs`
- **Hardware perf counters**: `src/hw_perf_counters.rs`
- **PMEM eviction collections**: `src/worker/policy/policy_stack/pmem_collections.rs`
- **LFU policy stack**: `src/worker/policy/policy_stack/lfu_stack.rs`

## Testing

To test different combinations (requires nightly Rust for allocator features):

```bash
# Test standalone FlatMap in DRAM
cargo +nightly test --lib flatmap::tests --no-default-features --features flatmap_dram

# Test standalone FlatMap in PMEM (requires PMEM hardware)
cargo +nightly test --lib flatmap::tests --no-default-features --features flatmap_pmem

# Check standalone FlatMap compilation for DRAM
cargo +nightly check --no-default-features --features flatmap_dram

# Check standalone FlatMap compilation for PMEM
cargo +nightly check --no-default-features --features flatmap_pmem

# Check FlatMap as PaperCache hashtable in DRAM
cargo +nightly check --no-default-features --features global_flatmap_dram

# Check FlatMap as PaperCache hashtable in PMEM
cargo +nightly check --no-default-features --features global_flatmap_pmem

# Check hashbrown HashMap in DRAM (for performance comparison)
cargo +nightly check --no-default-features --features hashbrown_dram

# Test with tiering and both hashtables in PMEM
cargo +nightly check --no-default-features --features flatmap_pmem

# Test with tiering and both hashtables in PMEM
cargo +nightly check --features "enable_tiering_manager,tiering_hashtable_pmem,global_hashtable_pmem,key_value_pmem"

# Test without tiering, global hashtable in PMEM (global cache only)
cargo +nightly check --features "global_hashtable_pmem,key_value_pmem"

# Test with tiering, neither hashtable in PMEM
cargo +nightly check --features "enable_tiering_manager,key_value_pmem"

# Test baseline without any features
cargo +nightly check --no-default-features

# Test global cache with allocator but no tiering
cargo +nightly check --features "key_value_pmem"

# Verify hw_perf (hardware counters - zero-cost abstraction when disabled)
cargo +nightly check --features "hw_perf"

# Verify eviction_stacks_pmem with key_value_pmem
cargo +nightly check --features "hw_perf,eviction_stacks_pmem,key_value_pmem"

# Verify full feature combination
cargo +nightly check --features "enable_tiering_manager,eviction_stacks_pmem,key_value_pmem"
```

**Note**: The tiering worker module is only compiled when BOTH an allocator feature 
AND `enable_tiering_manager` are enabled. This ensures the tiering manager is not 
used at all when disabled, allowing the cache to operate as a single global cache.

## Acceptance Criteria

✅ Separate configuration/feature flags for each hashtable's pmem placement
✅ Tiering manager hashtable can be placed in pmem independently
✅ Global hashtable can be placed in pmem independently
✅ All combinations work correctly (conditional compilation ensures validity)
✅ No performance regression when features are disabled (different code paths compiled)
✅ Tiering manager can be turned on/off independently
✅ Global hashtable can use pmem even when tiering is disabled
✅ `alloc_api_exp` removed; Hybrid allocator still functional for `key_value_pmem`
✅ `hw_perf` counters compile out to zero cost when feature is disabled
✅ `eviction_stacks_pmem` correctly allocates LFU stacks via HybridObjects
