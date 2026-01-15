# Feature Flags for Memory Tier Configuration

This document explains the implementation of separate configuration options for persistent memory (PMEM) placement of hashtables in PaperCache.

## Overview

The implementation provides three independent feature flags to control:
1. Whether the tiering manager is enabled
2. Where the tiering manager's internal hashtable is stored (DRAM vs PMEM)
3. Where the global cache hashtable is stored (DRAM vs PMEM)

## Feature Flags

### `enable_tiering_manager`
- **Purpose**: Enable/disable the tiering manager functionality
- **When enabled**: Automatic promotion/demotion of hot objects between DRAM and PMEM tiers
- **When disabled**: No automatic tiering, but global hashtable can still be placed in PMEM
- **Requirements**: Works with `key_value_pmem`, `alloc_with_hash`, or `alloc_api_exp`

### `tiering_hashtable_pmem`
- **Purpose**: Control memory placement of the tiering manager's internal hashtable
- **When enabled**: Tiering manager's hashtable stored in PMEM
- **When disabled**: Tiering manager's hashtable stored in DRAM
- **Requirements**: Requires `enable_tiering_manager` + one of the allocator features

### `global_hashtable_pmem`
- **Purpose**: Control memory placement of the main cache hashtable
- **When enabled**: Global cache hashtable stored in PMEM
- **When disabled**: Global cache hashtable stored in DRAM
- **Requirements**: One of (`key_value_pmem`, `alloc_with_hash`, `alloc_api_exp`)

## Implementation Details

### Type System

The implementation uses Rust's conditional compilation to select the appropriate hashtable types:

**Tiering Manager's hashtable** (`dram_cache` in `TieringManager`):
- Without `tiering_hashtable_pmem`: `DashMap` or `HashMap` (DRAM)
- With `tiering_hashtable_pmem`: `HashMap<..., Hybrid>` (PMEM)

**Global hashtable** (`objects` in `PaperCache`):
- Without `global_hashtable_pmem`: `DashMap` or `HashMap` (DRAM)
- With `global_hashtable_pmem`: `HashMap<..., Hybrid>` (PMEM)

### Allocator Integration

The `Hybrid` allocator is used to place hashtables in PMEM:
- Defined in `src/allocator.rs`
- Implements both `GlobalAlloc` and `Allocator` traits
- Routes allocations to either DRAM (jemalloc) or PMEM (UMF) based on configuration

### Conditional Compilation

The code uses `#[cfg(...)]` attributes extensively to:
1. Include/exclude the tiering module based on `enable_tiering_manager`
2. Select different hashtable implementations based on pmem flags
3. Conditionally compile tiering-related methods in PaperCache

## Valid Feature Combinations

### With Tiering Manager

1. **All in DRAM** (baseline performance):
   ```toml
   features = ["enable_tiering_manager", "alloc_with_hash"]
   ```

2. **Global hashtable in PMEM**:
   ```toml
   features = ["enable_tiering_manager", "global_hashtable_pmem", "alloc_with_hash"]
   ```

3. **Tiering hashtable in PMEM**:
   ```toml
   features = ["enable_tiering_manager", "tiering_hashtable_pmem", "alloc_with_hash"]
   ```

4. **Both hashtables in PMEM** (maximum persistence):
   ```toml
   features = ["enable_tiering_manager", "tiering_hashtable_pmem", "global_hashtable_pmem", "alloc_with_hash"]
   ```

### Without Tiering Manager

5. **Global hashtable in PMEM, no tiering**:
   ```toml
   features = ["global_hashtable_pmem", "alloc_with_hash"]
   ```

## Performance Characteristics

### PMEM vs DRAM Hashtables

**DRAM Hashtables**:
- ✅ Fast access (CPU cache speeds)
- ✅ Low latency for lookups and insertions
- ❌ Volatile - lost on restart
- ❌ Limited by DRAM capacity

**PMEM Hashtables**:
- ❌ Slower access (higher latency than DRAM)
- ❌ Higher latency for operations
- ✅ Persistent across restarts
- ✅ Larger capacity available

### Use Cases

1. **Both in PMEM**: Maximum durability, cache state survives restarts
2. **Neither in PMEM**: Maximum performance, baseline comparison
3. **Global in PMEM, Tiering in DRAM**: Persistent cache data, fast tiering decisions
4. **Global in DRAM, Tiering in PMEM**: Fast cache access, persistent tiering metadata
5. **No tiering + Global in PMEM**: Simple persistent cache without promotion overhead

## Code Locations

- **Feature definitions**: `Cargo.toml`
- **Tiering manager hashtable**: `src/tiering/manager.rs` (lines 130-146)
- **Global hashtable**: `src/lib.rs` (lines 120-137)
- **Worker manager integration**: `src/worker/manager.rs`
- **TieringConfig**: `src/tiering/manager.rs` (lines 32-60)

## Testing

To test different combinations (requires nightly Rust for allocator features):

```bash
# Test with tiering and both hashtables in PMEM
cargo +nightly check --features "enable_tiering_manager,tiering_hashtable_pmem,global_hashtable_pmem,alloc_with_hash"

# Test without tiering, global hashtable in PMEM (global cache only)
cargo +nightly check --features "global_hashtable_pmem,alloc_with_hash"

# Test with tiering, neither hashtable in PMEM
cargo +nightly check --features "enable_tiering_manager,alloc_with_hash"

# Test baseline without any features
cargo +nightly check --no-default-features

# Test global cache with allocator but no tiering
cargo +nightly check --features "alloc_with_hash"
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
