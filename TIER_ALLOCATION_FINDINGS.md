# Key-Value Tier Allocation - Implementation Findings

## Part 1: Verification

### Current Implementation Analysis

#### Before Changes:
- **Values**: Stored in pmem tier using `BufferPMEM = Box<[u8], Hybrid>`
  - Reference: `src/lib.rs:95`
  - Values are allocated using the Hybrid allocator which places data in pmem
  
- **Keys**: Stored in DRAM tier as plain type `K`
  - Reference: `src/object/mod.rs:22` (old implementation)
  - Keys were NOT wrapped in any pmem allocator
  - Keys resided in regular heap memory (DRAM)

#### Code References:
1. **Object Structure** (`src/object/mod.rs:21-26`):
   ```rust
   pub struct Object<K, V> {
       key: K,              // DRAM allocation
       data: Arc<V>,        // V can be BufferPMEM (pmem) or regular type
       expiry: ExpireTime,
   }
   ```

2. **Value Allocation** (`src/lib.rs:1536`):
   ```rust
   let val_buf: BufferPMEM = value.to_vec_in(Hybrid).into_boxed_slice();
   ```
   This explicitly uses the Hybrid allocator to place values in pmem.

3. **Key Allocation** (before fix):
   Keys were passed directly to `Object::new()` without any special allocation,
   meaning they used the default DRAM allocator.

### Memory Tier Verification

The codebase uses UMF (Unified Memory Framework) allocator wrapper to distinguish memory tiers:

- **Tier 0**: DRAM (jemalloc)
- **Tier 1**: PMEM (UMF devdax provider)

Reference: `umf_allocator/umf_allocator_wrapper.c:113-126`
```c
int check_tier(void *ptr) {
    umf_memory_pool_handle_t curr_pool;
    if (umfPoolByPtr(ptr, &curr_pool) == UMF_RESULT_SUCCESS) {
        if (curr_pool == pool) {
            return 1; //pmem
        }
    }
    else {
        return 0; //dram
    }
    return -1; //not from any UMF pool
}
```

## Part 2: Implementation

### Changes Made

#### 1. Object Structure Modification (`src/object/mod.rs`)

**Added conditional compilation** to use different key storage based on feature flags:

```rust
#[cfg(feature = "allocator_api")]
pub struct Object<K, V> {
    key: Box<K, Hybrid>,  // Now allocates in pmem tier
    data: Arc<V>,
    expiry: ExpireTime,
}

#[cfg(not(feature = "allocator_api"))]
pub struct Object<K, V> {
    key: K,               // Remains in DRAM for non-pmem builds
    data: Arc<V>,
    expiry: ExpireTime,
}
```

#### 2. Object Construction

**allocator_api feature** (`src/object/mod.rs:49`):
```rust
Object {
    key: Box::new_in(key, Hybrid),  // Allocates key in pmem
    data: Arc::new(data),
    expiry,
}
```

This uses Rust's `allocator_api` feature to allocate the key using the Hybrid allocator,
which places it in the pmem tier.

#### 3. Key Comparison

Updated `key_matches` to dereference the Box:
```rust
#[cfg(feature = "allocator_api")]
pub fn key_matches(&self, key: &K) -> bool {
    self.key.as_ref().eq(key)  // Dereferences Box<K, Hybrid>
}
```

#### 4. Helper Methods for Testing

Added to allocator_api impl block (`src/lib.rs:1870-1896`):
```rust
#[cfg(test)]
pub fn get_key_ptr(&self, key: &K) -> Result<*const K, CacheError>

#[cfg(test)]
pub fn get_value_ptr(&self, key: &K) -> Result<*const u8, CacheError>
```

Also added public helper function (`src/lib.rs:113-122`):
```rust
#[cfg(feature = "allocator_api")]
pub fn check_memory_tier<T>(ptr: *const T) -> i32
```

### Backwards Compatibility

- **Non-allocator_api builds**: Keys remain in DRAM (no change in behavior)
- **allocator_api builds**: Both keys and values now in pmem tier
- All existing functionality preserved
- API surface unchanged

## Part 3: Testing

### Test Suite Added (`src/lib.rs:2447-2538`)

#### 1. `test_key_stored_in_pmem_tier`
Verifies that a single key is allocated in the pmem tier (tier = 1)

#### 2. `test_value_stored_in_pmem_tier`
Verifies that a single value is allocated in the pmem tier (tier = 1)

#### 3. `test_multiple_keys_in_pmem_tier`
Verifies that multiple keys (0-9) are all allocated in the pmem tier

#### 4. `test_key_and_value_tiers_distinguishable`
Verifies that:
- Both key and value are in pmem tier
- They have different memory addresses (are distinct allocations)

### Testing Strategy

Tests use the `check_memory_tier()` function which calls the C function `check_tier()` from
the UMF allocator wrapper. This function returns:
- `0` for DRAM tier
- `1` for PMEM tier  
- `-1` for unknown/error

## Summary

### Findings
- ✅ Confirmed: Only values were in pmem tier before changes
- ✅ Confirmed: Keys were in DRAM tier before changes
- ✅ Implemented: Keys now allocated in pmem tier (allocator_api feature)
- ✅ Implemented: Comprehensive test suite to verify tier allocation

### Code Changes
- Modified: `src/object/mod.rs` (Object struct and impl)
- Modified: `src/lib.rs` (helper functions and tests)

### Deliverables
1. ✅ Confirmation of current key/value tier behavior (documented above)
2. ✅ Code changes to move keys to pmem tier
3. ✅ New test suite validating key placement in pmem tier

## Known Limitations

1. **Build System**: The `build.rs` file contains hard-coded paths that need to be updated
   for the project to build in different environments
   
2. **Serialization**: The current implementation requires `K` to be a type that can be
   stored directly in a `Box`. Complex types requiring serialization would need additional
   work.

3. **Testing**: Tests can only be run when:
   - UMF (Unified Memory Framework) is installed
   - PMEM device is available (`/dev/dax0.0`)
   - allocator_api feature is enabled
   - Nightly Rust compiler is used (for allocator_api feature)
