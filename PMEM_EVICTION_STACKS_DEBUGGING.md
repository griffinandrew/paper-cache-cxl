# PMem Eviction Stacks - Debugging Guide

## Overview

The `pmem_eviction_stacks` feature enables storing eviction policy metadata in persistent memory (PMem) instead of DRAM. This document helps debug segfaults and issues.

## Common Issues and Solutions

### 1. Segfault on Basic Operations

**Symptoms:**
- Segfault when inserting items into eviction stacks
- Segfault during eviction/removal operations
- Crashes in `rebuild_index()`

**Potential Causes:**

#### A. PMem Device Not Properly Initialized
The allocator expects `/dev/dax0.0` to be available and properly configured.

**Check:**
```bash
ls -l /dev/dax0.0
ndctl list --namespaces
```

**Fix:**
- Ensure PMEM device is mounted and accessible
- Verify permissions on `/dev/dax0.0`
- Check that `DAX_SIZE` in `src/allocator.rs` matches your device

#### B. Hybrid Allocator Issues
The `HybridObjects` allocator may not be correctly switching between DRAM and PMEM.

**Debug:**
In `src/allocator.rs`, add debug output:
```rust
unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
    eprintln!("ALLOC: size={}, align={}", layout.size(), layout.align());
    // ... existing code
}
```

#### C. Clone() Operations on PMem Keys
The `rebuild_index()` function clones keys frequently. If keys contain pointers or non-trivial data, this could cause issues.

**Fix:**
Ensure `HashedKey` (u64) doesn't contain invalid data.

### 2. Memory Corruption

**Symptoms:**
- Random crashes
- Data corruption in eviction stacks
- Inconsistent behavior

**Potential Causes:**

#### A. Vec Allocation Issues
The `order` Vec is allocated in PMem using `Vec::new_in(Hybrid)`. If the allocator returns invalid pointers, this will crash.

**Debug:**
Add checks in `pmem_hashlist.rs`:
```rust
pub fn push_front(&mut self, key: K) {
    if self.map.contains_key(&key) {
        return;
    }
    eprintln!("Before insert: order.len()={}, order.capacity()={}", 
              self.order.len(), self.order.capacity());
    self.order.insert(0, key.clone());
    eprintln!("After insert: order.len()={}", self.order.len());
    self.rebuild_index();
}
```

#### B. HashMap with Custom Allocator
hashbrown's HashMap with the Hybrid allocator may have issues if the allocator doesn't properly implement the allocator_api2 traits.

**Check:**
Verify that `HybridObjects` correctly implements:
- `Allocator` trait
- Proper alignment handling
- Null pointer checks

### 3. Feature Flag Mismatches

**Symptoms:**
- Linker errors
- Undefined symbols
- Type mismatches

**Solution:**
Ensure consistent feature flags:

**For PMem eviction stacks:**
```bash
cargo build --features alloc_with_hash,pmem_eviction_stacks
```

**For DRAM eviction stacks (default):**
```bash
cargo build --features alloc_with_hash
```

### 4. UMF Allocator Not Available

**Symptoms:**
- Linker errors: `undefined symbol: umf_alloc`
- Warning: "UMF wrapper.h not found; skipping bindgen"

**Cause:**
The UMF (Unified Memory Framework) C library is not built or linked.

**Fix:**
1. Ensure `umf_allocator/umf_allocator_wrapper.c` is being compiled
2. Check `build.rs` for proper compilation
3. Verify that `wrapper.h` exists and is accessible

## Testing Strategy

### Without PMem Device (Development)

Use DRAM-based eviction stacks:
```bash
cargo test --features alloc_with_hash
```

### With PMem Device

1. **Check device first:**
```bash
sudo ndctl list --namespaces
ls -l /dev/dax0.0
```

2. **Run with PMem eviction stacks:**
```bash
sudo cargo +nightly test --features alloc_with_hash,pmem_eviction_stacks
```

3. **Monitor for crashes:**
```bash
dmesg | grep -i "segfault\|fault"
```

### Stress Testing

```rust
// Add to tests/pmem_eviction_stacks.rs
#[test]
fn stress_test_pmem_allocations() {
    let cache = PaperCache::<u32, BufferPMEM>::new(
        100_000,
        &[PaperPolicy::Lru],
        PaperPolicy::Lru,
    ).unwrap();

    // Rapid insertions
    for i in 0..10_000 {
        cache.set(i, &[0u8; 50], None).unwrap();
    }

    // Mixed operations
    for i in 0..5_000 {
        cache.get(&i).ok();
        cache.set(i + 10_000, &[0u8; 50], None).unwrap();
    }

    std::thread::sleep(Duration::from_secs(2));
    println!("Stress test completed");
}
```

## Monitoring PMem Usage

### Check Allocations

Add to `src/allocator.rs`:
```rust
static PMEM_ALLOCS: AtomicUsize = AtomicUsize::new(0);
static PMEM_DEALLOCS: AtomicUsize = AtomicUsize::new(0);

// In umf_alloc path:
PMEM_ALLOCS.fetch_add(1, Ordering::Relaxed);

// In umf_dealloc path:
PMEM_DEALLOCS.fetch_add(1, Ordering::Relaxed);

// Print periodically
if PMEM_ALLOCS.load(Ordering::Relaxed) % 1000 == 0 {
    eprintln!("PMem allocs: {}, deallocs: {}", 
              PMEM_ALLOCS.load(Ordering::Relaxed),
              PMEM_DEALLOCS.load(Ordering::Relaxed));
}
```

## Known Limitations

1. **Requires nightly Rust**: The `allocator_api` feature is unstable
2. **Requires PMem hardware**: Tests fail without actual PMEM device
3. **Performance overhead**: `rebuild_index()` is O(n), unlike true linked list O(1)
4. **No persistence**: Data is lost on crash (by design for cache metadata)

## Debugging Checklist

- [ ] PMem device exists and is accessible (`/dev/dax0.0`)
- [ ] Correct feature flags are used (`alloc_with_hash,pmem_eviction_stacks`)
- [ ] UMF allocator library is compiled and linked
- [ ] Running with nightly Rust compiler
- [ ] Added debug output to track allocations
- [ ] Checked dmesg for kernel-level faults
- [ ] Verified that simpler operations work before complex ones
- [ ] Tested with smaller cache sizes first

## Getting Help

If segfaults persist:

1. Create a minimal reproducible example
2. Run with RUST_BACKTRACE=full
3. Use valgrind (if PMem allows):
   ```bash
   valgrind --leak-check=full ./target/debug/your_test
   ```
4. Check if issue reproduces with DRAM-only configuration
