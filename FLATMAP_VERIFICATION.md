# FlatMap Resizing & Unified Tiering Limits - Verification Guide

## Overview

This document describes how to verify the implementation of resizable FlatMap with unified tiering limits.

## Features Implemented

### 1. Automatic FlatMap Resizing
- FlatMap now resizes automatically when load factor exceeds 75%
- Capacity doubles each time resize is triggered
- All data is preserved during resize through rehashing

### 2. Dual Limit Configuration
- `dram_object_limit`: Hard limit for DRAM tiering cache (bytes)
- `dram_pointer_limit`: Soft limit for global cache (pointer count)
- Both configurable via `TieringConfig`

### 3. New Feature Flag
- `flatmap_hash_and_object_tiering`: Enables FlatMap for both Global and Tiering caches

## How to Build and Test

### Prerequisites
- Rust nightly toolchain
- UMF allocator (optional, for PMEM features)

### Build Commands

#### 1. Test FlatMap Resizing (Standalone)
```bash
cargo +nightly test --lib flatmap::tests --no-default-features --features flatmap_dram
```

**Expected Result:** All 12 FlatMap tests should pass, including:
- `test_resize` - Verifies automatic resizing with 10 items in capacity-4 map
- `test_resize_preserves_data` - Verifies data integrity through resize
- `test_resize_instead_of_panic` - Verifies resize happens instead of panic

#### 2. Check Compilation with New Feature
```bash
cargo +nightly check --no-default-features --features "flatmap_hash_and_object_tiering,enable_tiering_manager,key_value_pmem"
```

**Current Status:** ⚠️ DOES NOT COMPILE - 26 errors remain
**Reason:** Access patterns need updating to use RwLock (see "Remaining Work" below)

#### 3. Run Integration Tests (Once Compilation Fixed)
```bash
cargo +nightly test --test flatmap_resize_tiering --features "flatmap_hash_and_object_tiering,enable_tiering_manager,key_value_pmem"
```

**Tests Included:**
- `test_flatmap_resizing_with_many_inserts` - 200 inserts to force resize, verifies data preservation
- `test_pointer_limit_enforcement` - Verifies global cache pointer limit works
- `test_flatmap_preserves_data_during_resize` - Verifies data integrity

## What to Verify

### 1. FlatMap Resizing Works Correctly
**Test:** Insert more items than initial capacity
**Verification:**
- No panics occur
- All inserted items remain accessible
- Capacity increases (check with `.capacity()` method)

**Example:**
```rust
let mut map = FlatMap::new(4); // Initial capacity 4
for i in 0..10 {
    map.insert_with_hasher(i, i * 10, &hasher); // Triggers resize at item 4
}
assert_eq!(map.len(), 10);
assert!(map.capacity() >= 8); // Should have resized
```

### 2. Dual Limits Are Enforced
**Test:** Create cache with low limits, insert many items
**Verification:**
- DRAM size stays near `dram_object_limit`
- Global cache size stays near `dram_pointer_limit`
- Eviction occurs to maintain limits

**Example:**
```rust
let mut config = TieringConfig::default();
config.dram_object_limit = 10 * 1024; // 10KB
config.dram_pointer_limit = 50;

let cache = PaperCache::with_tiering_config(..., config);
// Insert 100 items
// Verify cache.len() <= ~50 and stats.dram_size <= ~10KB
```

### 3. Performance Characteristics
**What to Check:**
- Initial allocation is small (4096 capacity)
- Resizing overhead is reasonable (amortized O(1) inserts)
- Memory usage grows appropriately with data

## Remaining Work

### Code Paths Needing Updates (26 locations)

The flatmap is wrapped in `Arc<RwLock<FlatMap>>` for thread safety. All access patterns need to use:
- `.read().unwrap()` for read operations
- `.write().unwrap()` for write operations

**Affected Files:**
1. `src/tiering/manager.rs` - dram_cache access
2. `src/lib.rs` - objects access  
3. `src/worker/tiering.rs` - both caches

**Pattern Examples:**

**Before (DashMap):**
```rust
self.dram_cache.get(key)
self.dram_cache.insert(key, value)
```

**After (FlatMap with RwLock):**
```rust
self.dram_cache.read().unwrap().get(key).map(|v| Arc::new(v.clone()))
self.dram_cache.write().unwrap().insert(key, value)
```

### Methods Still Needing Implementation

1. **TieringManager methods:**
   - Various cache access methods need RwLock patterns
   - `update_dram_copy`, `promote_to_dram`, `demote_from_dram`

2. **PaperCache methods:**
   - All `objects` field accesses in `get`, `set`, `remove`, etc.
   - Worker thread accesses to objects

3. **Constructor methods:**
   - `PaperCache::with_tiering_config` doesn't exist yet (needed by tests)

### Eviction Logic

Phase 5 (not started) requires:
1. Check `stats.dram_size` against `config.dram_object_limit`
2. Check `global_cache.len()` against `config.dram_pointer_limit`
3. Trigger eviction when limits exceeded
4. Implement in eviction worker or get_keys_to_demote method

## Success Criteria

✅ **Resizing Works:**
- All standalone FlatMap tests pass
- Data preserved through multiple resizes
- No panics on full capacity

⚠️ **Integration Compiles:**
- Feature flag compilation succeeds
- No type errors or access pattern issues

⚠️ **Limits Enforced:**
- DRAM object bytes stay within limit
- Pointer count stays within limit
- Eviction triggers appropriately

⚠️ **Tests Pass:**
- Integration tests in `tests/flatmap_resize_tiering.rs` pass
- Existing tiering tests still pass

## Known Limitations

1. **Clone Requirement:** FlatMap with RwLock requires Clone on K, V (needed for Arc wrapping)
2. **Performance:** Extra clone overhead when retrieving from FlatMap vs DashMap's Ref
3. **Partial Implementation:** Core resizing works, but integration incomplete

## Next Steps for Completion

1. Update all 26 access pattern locations to use RwLock
2. Add `with_tiering_config` constructor to PaperCache
3. Implement limit enforcement in eviction logic
4. Test thoroughly with the provided integration tests
5. Run full test suite to ensure no regressions
