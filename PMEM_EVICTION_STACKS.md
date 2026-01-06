# PMem Eviction Stacks Feature

## Overview

The `pmem_eviction_stacks` feature flag enables storing cache eviction policy metadata in Persistent Memory (PMem) instead of DRAM. This is independent from where the actual cached objects are stored, allowing flexible memory tier configuration.

## Feature Independence

Previously, the eviction stack storage was coupled with the object table storage via shared feature flags (`allocator_api`, `alloc_with_hash`, `alloc_api_exp`). Now, they can be configured independently:

| Configuration | Object Table | Eviction Stacks | Use Case |
|--------------|--------------|-----------------|----------|
| Default | DRAM (DashMap) | DRAM (kwik HashList) | Standard in-memory cache |
| `alloc_with_hash` | PMem (HashMap + Hybrid) | DRAM (kwik HashList) | Persistent objects, fast eviction metadata |
| `pmem_eviction_stacks` + `alloc_with_hash` | PMem (HashMap + Hybrid) | PMem (PmemHashList + Hybrid) | Full PMem storage |
| `pmem_eviction_stacks` + `allocator_api` | DRAM (DashMap) | PMem (PmemHashList + Hybrid) | DRAM objects, persistent eviction metadata |

## Building

### With Nightly Rust (Required for PMem)

```bash
# PMem eviction stacks + PMem object table
cargo +nightly build --features alloc_with_hash,pmem_eviction_stacks

# PMem eviction stacks only
cargo +nightly build --features allocator_api,pmem_eviction_stacks
```

### With Stable Rust (DRAM only)

```bash
# Standard DRAM cache
cargo build

# PMem object table, DRAM eviction stacks
cargo build --features alloc_with_hash
```

## Requirements

### For PMem Features

1. **Nightly Rust**: Required for `allocator_api` feature
2. **PMem Device**: `/dev/dax0.0` or similar DAX device
3. **UMF Library**: Unified Memory Framework C library
4. **Build tools**: C compiler, bindgen dependencies

### Check PMem Device

```bash
# List available PMem namespaces
sudo ndctl list --namespaces

# Verify DAX device
ls -l /dev/dax0.0

# Check permissions
stat /dev/dax0.0
```

## Architecture

### PmemHashList Implementation

When `pmem_eviction_stacks` is enabled, the eviction policy stacks use `PmemHashList` instead of `kwik::collections::HashList`:

```rust
// Without feature (DRAM)
use kwik::collections::HashList;

// With pmem_eviction_stacks (PMem)
use crate::worker::policy::pmem_hashlist::HashList;
```

`PmemHashList` uses:
- `hashbrown::HashMap` with `HybridObjects` allocator for key lookup
- `Vec` with `HybridObjects` allocator for ordered storage
- Both allocated in PMem via the Hybrid allocator

### Affected Policy Stacks

All eviction policies use the feature-gated HashList:
- LRU (Least Recently Used)
- FIFO (First In First Out)
- LFU (Least Frequently Used)
- MRU (Most Recently Used)
- ARC (Adaptive Replacement Cache)
- CLOCK
- SIEVE
- 2Q
- S3-FIFO

## Performance Considerations

### Trade-offs

**PMem Eviction Stacks:**
- ✓ Reduced DRAM pressure
- ✓ Potential for persistence (if needed in future)
- ✗ Slower than DRAM for metadata operations
- ✗ `rebuild_index()` is O(n) instead of O(1) for true linked list

**DRAM Eviction Stacks:**
- ✓ Faster metadata operations
- ✓ True O(1) list operations with kwik::collections::HashList
- ✗ Uses valuable DRAM space
- ✗ No persistence

### Performance Tips

1. **For read-heavy workloads**: PMem eviction stacks have minimal impact
2. **For write-heavy workloads**: Consider DRAM eviction stacks
3. **Large caches**: PMem eviction stacks save significant DRAM
4. **Small caches**: DRAM eviction stacks may be faster

## Safety Improvements

The implementation includes several safety features to prevent segfaults:

### 1. Null Pointer Checks
```rust
unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
    if ptr.is_null() {
        return; // Don't try to free null pointers
    }
    // ... rest of deallocation
}
```

### 2. Pre-allocation
```rust
fn rebuild_index(&mut self) {
    self.map.clear();
    self.map.reserve(self.order.len()); // Avoid mid-operation allocation
    // ... rebuild
}
```

### 3. Bounds Checking
```rust
pub fn before<Q>(&self, key: &Q) -> Option<&K> {
    let index = *self.map.get(key)?;
    // Use bounds-checked get() instead of direct indexing
    self.order.get(index + 1)
}
```

## Testing

### Unit Tests

Built-in policy stack tests work with both configurations:

```bash
# Test with DRAM eviction stacks
cargo test --features alloc_with_hash

# Test with PMem eviction stacks (requires PMem device)
cargo +nightly test --features alloc_with_hash,pmem_eviction_stacks
```

### Integration Tests

```bash
# Run PMem eviction stack integration tests
cargo +nightly test --test pmem_eviction_stacks --features alloc_with_hash,pmem_eviction_stacks
```

Note: Tests requiring actual PMem operations will fail without a PMem device.

## Troubleshooting

### Build Errors

**Error: `undefined symbol: umf_alloc`**
- Cause: UMF library not linked
- Fix: Ensure `build.rs` compiles the C wrapper and links correctly

**Error: `feature(allocator_api) is unstable`**
- Cause: Using stable Rust
- Fix: Use `cargo +nightly` for PMem features

### Runtime Issues

**Segfault on eviction operations**
- See `PMEM_EVICTION_STACKS_DEBUGGING.md` for detailed debugging guide
- See `SEGFAULT_FIXES.md` for specific fix recommendations
- Check PMem device availability and permissions

**Warning: `UMF wrapper.h not found; skipping bindgen`**
- This is expected in environments without PMem
- The feature will fail to link if actually used
- For development: use features without `pmem_eviction_stacks`

## Example Usage

### Basic Cache with PMem Eviction Stacks

```rust
use paper_cache::{PaperCache, PaperPolicy, BufferPMEM};

fn main() {
    // Create cache with PMem eviction stacks
    let cache = PaperCache::<u32, BufferPMEM>::new(
        10_000_000, // 10 MB
        &[PaperPolicy::Lru],
        PaperPolicy::Lru,
    ).expect("Failed to create cache");

    // Use cache normally
    for i in 0..1000 {
        cache.set(i, &vec![0u8; 1000], None).unwrap();
    }

    // Eviction metadata is in PMem, transparent to user
    for i in 0..500 {
        let _ = cache.get(&i);
    }
}
```

### Feature Detection

```rust
#[cfg(feature = "pmem_eviction_stacks")]
fn with_pmem_eviction() {
    println!("Using PMem for eviction stacks");
}

#[cfg(not(feature = "pmem_eviction_stacks"))]
fn without_pmem_eviction() {
    println!("Using DRAM for eviction stacks");
}
```

## Migration Guide

### From Old Behavior (Coupled)

**Before:**
```toml
# In Cargo.toml
features = ["alloc_with_hash"]
```
This put BOTH objects AND eviction stacks in PMem.

**After (Same Behavior):**
```toml
# In Cargo.toml
features = ["alloc_with_hash", "pmem_eviction_stacks"]
```

**After (New Option - Objects in PMem, Eviction in DRAM):**
```toml
# In Cargo.toml
features = ["alloc_with_hash"]
# Omit pmem_eviction_stacks for DRAM eviction metadata
```

### From All-DRAM

No changes needed! Default behavior remains all-DRAM.

## Contributing

When modifying eviction policy implementations:

1. Use feature-gated imports:
   ```rust
   #[cfg(feature = "pmem_eviction_stacks")]
   use crate::worker::policy::pmem_hashlist::HashList;
   
   #[cfg(not(feature = "pmem_eviction_stacks"))]
   use kwik::collections::HashList;
   ```

2. Test with both configurations:
   ```bash
   cargo test --features alloc_with_hash
   cargo +nightly test --features alloc_with_hash,pmem_eviction_stacks
   ```

3. Document any PMem-specific behavior

## References

- [PMEM_EVICTION_STACKS_DEBUGGING.md](./PMEM_EVICTION_STACKS_DEBUGGING.md) - Debugging guide
- [SEGFAULT_FIXES.md](./SEGFAULT_FIXES.md) - Segfault prevention details
- [Rust Allocator API](https://doc.rust-lang.org/std/alloc/trait.Allocator.html)
- [hashbrown documentation](https://docs.rs/hashbrown/)
