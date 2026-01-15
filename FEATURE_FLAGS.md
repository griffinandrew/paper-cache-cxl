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
- **Requirements**: Works with `key_value_pmem` or `alloc_api_exp`

### `tiering_hashtable_pmem`
- **Purpose**: Control memory placement of the tiering manager's internal hashtable
- **When enabled**: Tiering manager's hashtable stored in PMEM
- **When disabled**: Tiering manager's hashtable stored in DRAM
- **Requirements**: Requires `enable_tiering_manager` + (`key_value_pmem` or `alloc_api_exp`)

### `global_hashtable_pmem`
- **Purpose**: Control memory placement of the main cache hashtable
- **When enabled**: Global cache hashtable stored in PMEM
- **When disabled**: Global cache hashtable stored in DRAM
- **Requirements**: Can be used independently or with `key_value_pmem`/`alloc_api_exp`

### `alloc_api_exp` (Experimental)
- **Purpose**: Experimental allocator API for testing
- **When enabled**: Uses experimental allocator implementation with same hashtable type as `key_value_pmem`
- **Use case**: Testing and development

## Implementation Details

### Type System

The implementation uses Rust's conditional compilation to select the appropriate hashtable types:

**Tiering Manager's hashtable** (`dram_cache` in `TieringManager`):
- Without `tiering_hashtable_pmem`: `DashMap` or `HashMap` (DRAM)
- With `tiering_hashtable_pmem`: `HashMap<..., Hybrid>` (PMEM)

**Global hashtable** (`objects` in `PaperCache`):
- Without `global_hashtable_pmem`: `DashMap` (DRAM)
- With `global_hashtable_pmem`: `RwLock<HashMap<..., Hybrid>>` (PMEM)

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

## Code Locations

- **Feature definitions**: `Cargo.toml`
- **Allocator**: `src/allocator.rs`
- **Tiering manager hashtable**: `src/tiering/manager.rs`
- **Global hashtable**: `src/lib.rs`
- **Worker manager integration**: `src/worker/manager.rs`

## Testing

To test different combinations (requires nightly Rust for allocator features):

```bash
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
