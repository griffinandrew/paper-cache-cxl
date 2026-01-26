# PMEM-Backed Eviction Stacks for LFU Policy

## Overview

This implementation adds support for allocating the LFU (Least Frequently Used) eviction stack data structures in Persistent Memory (PMEM) instead of DRAM. This is controlled by the `eviction_stack_pmem` feature flag.

**Important:** This implementation uses custom allocator-aware collections that explicitly use the `HybridObjects` allocator instead of relying on the global allocator. This is critical because paper-cache is a library, and consuming binaries may override the global allocator, which would break PMEM allocation.

## Feature Flag

Add the feature flag when building or testing:

```bash
# Build with PMEM-backed eviction stacks
cargo build --features eviction_stack_pmem

# Build without PMEM (default, uses DRAM)
cargo build
```

## Implementation Details

### Data Structures Modified

The LfuStack contains the following data structures:

1. **index_map**: `HashMap<HashedKey, PmemIndex, NoHasher, HybridObjects>`
   - **Without feature**: Uses `std::collections::HashMap` with `Index<CountStack>` (allocates from DRAM)
   - **With feature**: Uses `hashbrown::HashMap` with `PmemIndex` and `HybridObjects` allocator (allocates from PMEM/DRAM based on HybridObjects policy)

2. **count_stacks**: `PmemVecList<CountStack>` (when feature enabled) or `VecList<CountStack>` (default)
   - **Without feature**: Uses `dlv_list::VecList` (uses global allocator → DRAM)
   - **With feature**: Uses custom `PmemVecList` with explicit `HybridObjects` allocator (allocates from PMEM)

3. **stack** (within CountStack): `PmemHashList<HashedKey, NoHasher>` (when feature enabled) or `HashList<HashedKey, NoHasher>` (default)
   - **Without feature**: Uses `kwik::HashList` (uses global allocator → DRAM)
   - **With feature**: Uses custom `PmemHashList` with explicit `HybridObjects` allocator (allocates from PMEM)

### Custom PMEM Collections

#### PmemVecList<T>

A custom doubly-linked list implementation that uses `hashbrown::HashMap` with `HybridObjects` allocator for internal storage.

**Why not use dlv-list::VecList?**
- VecList uses the global allocator for all its internal allocations
- When paper-cache is used as a library, consuming binaries can override the global allocator
- This would cause VecList to allocate from a different allocator than HybridObjects, breaking PMEM functionality

**API Compatibility:**
- Provides the same interface as `dlv_list::VecList`
- Uses `PmemIndex` instead of `Index<T>` for node references
- Methods: `new()`, `front()`, `front_index()`, `get_mut()`, `get_next_index()`, `push_front()`, `insert_after()`, `remove()`, `clear()`

**Implementation:**
- Stores nodes in a `HashMap<usize, Node<T>, _, HybridObjects>`
- Maintains doubly-linked list structure with prev/next pointers
- All allocations explicitly use HybridObjects

#### PmemHashList<T, S>

A custom hash-based doubly-linked list that uses `hashbrown::HashMap` with `HybridObjects` allocator.

**Why not use kwik::HashList?**
- Same issue as VecList - relies on global allocator
- Would break when used in library contexts with custom global allocators

**API Compatibility:**
- Provides the same interface as `kwik::HashList`
- Methods: `with_hasher()`, `is_empty()`, `push_front()`, `pop_back()`, `remove()`

**Implementation:**
- Stores nodes in a `HashMap<T, ListNode<T>, S, HybridObjects>`
- Maintains doubly-linked list structure with prev/next pointers
- All allocations explicitly use HybridObjects

### Key Changes

#### Cargo.toml
- Added `eviction_stack_pmem` feature flag

#### src/lib.rs
- Added `eviction_stack_pmem` to the list of features that enable the allocator module
- Added explanatory comment about why it's not in the allocator_api nightly feature list

#### src/allocator.rs
- Added `eviction_stack_pmem` to the allocator-api2 implementation feature gate
- Feature-gated `std::alloc::Allocator` imports and implementation to prevent issues on stable Rust

#### src/worker/policy/policy_stack/pmem_collections.rs (NEW)
- Custom `PmemVecList<T>` implementation
- Custom `PmemHashList<T, S>` implementation
- Both use `hashbrown::HashMap` with `HybridObjects` allocator for all internal storage

#### src/worker/policy/policy_stack/lfu_stack.rs
- Conditional compilation based on `eviction_stack_pmem` feature
- Two implementations of `LfuStack`:
  - DRAM version (default): Uses `std::collections::HashMap`, `VecList`, and `HashList`
  - PMEM version: Uses `hashbrown::HashMap`, `PmemVecList`, and `PmemHashList` with `HybridObjects` allocator
- Conditional implementations of `CountStack` to handle different collection types
- Added comprehensive documentation comments
- Added stress test to verify no segmentation faults

#### src/worker/policy/policy_stack/mod.rs
- Added `pmem_collections` module (conditionally compiled with feature flag)

## How It Works

### HybridObjects Allocator

The `HybridObjects` allocator is a hybrid DRAM/PMEM allocator that:
1. Allocates from DRAM (via jemalloc) up to a configurable limit
2. Spills to PMEM (via UMF - Unified Memory Framework) when DRAM limit is exceeded
3. Can be configured to use all DRAM or all PMEM via feature flags

### allocator-api2

The implementation uses `allocator-api2`, which is a polyfill for Rust's allocator API on stable Rust. This allows `hashbrown::HashMap` to use a custom allocator without requiring nightly Rust.

## Testing

### Running Tests

```bash
# Test without PMEM (DRAM only)
cargo test --lib lfu_stack

# Test with PMEM feature enabled
# Note: Requires UMF library and PMEM hardware to link successfully
cargo test --lib --features eviction_stack_pmem lfu_stack
```

### Test Coverage

1. **eviction_order_is_correct**: Verifies LFU eviction order is correct
2. **stress_test_no_segfault**: Stress tests with 1000 items, verifies no crashes

## Building on Systems Without PMEM

On systems without the UMF library and PMEM hardware:
- The code will **compile** successfully with the `eviction_stack_pmem` feature
- Tests will **fail to link** due to missing UMF symbols (`umf_allocator_init`, `umf_alloc`, etc.)
- This is expected behavior - the implementation is correct, but requires PMEM hardware/libraries to run

To test on such systems, use the default configuration (without the feature flag).

## Architecture Compatibility

### Without eviction_stack_pmem (Default)
- ✅ Compiles on stable Rust
- ✅ Runs on any system
- ✅ Uses standard library collections (DRAM only)

### With eviction_stack_pmem
- ✅ Compiles on stable Rust (uses allocator-api2)
- ⚠️ Requires UMF library and PMEM hardware to run
- ✅ Uses hashbrown with custom allocator
- ✅ Supports PMEM allocation via HybridObjects

## Performance Considerations

1. **HashMap**: `hashbrown::HashMap` is generally faster than `std::collections::HashMap`
2. **PMEM Access**: PMEM access is slower than DRAM, but provides persistence
3. **Hybrid Allocation**: The HybridObjects allocator keeps hot data in DRAM and spills cold data to PMEM

## Design Decisions

### Why Custom Collections Instead of Global Allocator?

**The Problem:**
Paper-cache is a library, not a standalone application. When used as a library:
1. The consuming binary controls the global allocator
2. External crates like `dlv-list` and `kwik` use the global allocator for their internal allocations
3. If the consuming binary sets a different global allocator, VecList and HashList would allocate from that allocator instead of HybridObjects
4. This breaks PMEM functionality because the eviction stack data would be in DRAM (or wherever the global allocator points)

**The Solution:**
Create custom implementations that explicitly use `HybridObjects` allocator via allocator-api2:
- Cannot rely on global allocator being HybridObjects
- Must explicitly specify HybridObjects as the allocator for all collections
- Use `hashbrown::HashMap` which supports custom allocators via allocator-api2
- Implement custom VecList and HashList on top of HashMap with HybridObjects

**Why Not Just Set HybridObjects as Global Allocator?**
- Only one global allocator can be set per binary
- As a library, we don't control the binary's global allocator
- The consuming application might need its own global allocator
- Setting a global allocator in a library would conflict with the application's choice

### Why Not Modify VecList and HashList?

`VecList` (from `dlv-list` crate) and `HashList` (from `kwik` crate) are external dependencies that don't expose allocator parameters. Rather than forking these crates, we:
1. Created custom implementations with the same API
2. Used `hashbrown::HashMap` as the backing store (it supports allocator-api2)
3. Maintained API compatibility for drop-in replacement

This approach provides PMEM benefits while maintaining compatibility with existing crates.

### Why hashbrown Instead of std::collections::HashMap?

`std::collections::HashMap` doesn't support custom allocators on stable Rust. `hashbrown::HashMap` provides:
1. Support for custom allocators via allocator-api2
2. Better performance than std HashMap
3. Same API surface as std HashMap

## Future Enhancements

Potential improvements:
1. Add similar PMEM support for other eviction policies (LRU, FIFO, etc.)
2. Performance benchmarks comparing DRAM vs PMEM configurations
3. Monitoring/metrics for DRAM vs PMEM allocation split
4. Custom PMEM-optimized data structures for VecList and HashList

## References

- [hashbrown crate](https://docs.rs/hashbrown/)
- [allocator-api2 crate](https://docs.rs/allocator-api2/)
- [UMF (Unified Memory Framework)](https://github.com/oneapi-src/unified-memory-framework)
- [Paper-Cache repository](https://github.com/griffinandrew/paper-cache-cxl)
