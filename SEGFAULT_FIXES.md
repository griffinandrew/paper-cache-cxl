# Potential PMem Eviction Stack Segfault Fixes

## Issues Found

### 1. Null Pointer Checks in rebuild_index()

The `rebuild_index()` function doesn't check if allocations succeed when growing the HashMap.

**File:** `src/worker/policy/pmem_hashlist.rs`

**Current Code (line 172-177):**
```rust
fn rebuild_index(&mut self) {
    self.map.clear();
    for (idx, key) in self.order.iter().enumerate() {
        self.map.insert(key.clone(), idx);
    }
}
```

**Suggested Fix:**
Add capacity reservation to avoid reallocation during iteration:
```rust
fn rebuild_index(&mut self) {
    self.map.clear();
    // Pre-allocate to avoid reallocations during insert
    self.map.reserve(self.order.len());
    for (idx, key) in self.order.iter().enumerate() {
        self.map.insert(key.clone(), idx);
    }
}
```

### 2. Vec Growth Issues

The `order` Vec can grow during operations, and if PMem allocation fails, it could cause corruption.

**File:** `src/worker/policy/pmem_hashlist.rs`

**Add to all mutation methods:**
```rust
pub fn push_front(&mut self, key: K) {
    if self.map.contains_key(&key) {
        return;
    }
    // Reserve capacity before insert to ensure allocation happens first
    if self.order.len() + 1 > self.order.capacity() {
        self.order.reserve(1);
    }
    self.order.insert(0, key.clone());
    self.rebuild_index();
}
```

### 3. Allocator Safety

**File:** `src/allocator.rs`

The allocator has unsafe code that could be causing issues. Specific problems:

#### A. Race condition on DRAM_LIMIT_OBJECTS (lines 38, 59, 115, 122)

**Problem:** Multiple threads reading mutable static without synchronization

**Fix:** Add proper unsafe blocks:
```rust
unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
    let dram_limit = unsafe { DRAM_LIMIT_OBJECTS };
    
    if dram_limit == 0 {
        unsafe {
            INIT.call_once(|| {
                // ... initialization
            });
        }
        let ptr = unsafe { allocator_bindings::umf_alloc(layout.size(), layout.align()) as *mut u8 };
        // ... rest
    }
    // ...
}
```

#### B. check_tier() called on potentially invalid pointers (line 130)

**Problem:** If ptr is from a different allocator or corrupted, `check_tier` could segfault

**Fix:** Add null check:
```rust
unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
    if ptr.is_null() {
        return; // Don't try to free null pointers
    }
    
    let dram_limit = unsafe { DRAM_LIMIT_OBJECTS };
    
    if dram_limit == 0 {
        unsafe { allocator_bindings::umf_dealloc(ptr as *mut std::ffi::c_void); }
        return;
    }
    // ...
}
```

### 4. HashMap Clone Operations

**File:** `src/worker/policy/pmem_hashlist.rs`

The code clones keys frequently. If the key type has issues with cloning, this could cause problems.

**Verification:**
Check that `HashedKey` (u64) is safe to clone - this should be fine, but if custom types are used later, add:

```rust
impl<K, S> HashList<K, S>
where
    K: Hash + Eq + Clone + Copy,  // Add Copy bound for safety
    S: BuildHasher + Default,
{
    // ...
}
```

## Testing Recommendations

### 1. Add Allocation Failure Simulation

```rust
#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_allocation_stress() {
        let mut list = HashList::<u64, RandomState>::default();
        
        // Rapid insertions and removals
        for i in 0..10000 {
            list.push_front(i);
        }
        
        for i in 0..5000 {
            list.pop_back();
        }
        
        // Rebuild operations
        for i in 10000..15000 {
            list.push_front(i);
        }
        
        assert!(list.len() > 0);
    }
}
```

### 2. Check for Memory Leaks

Add to allocator.rs:
```rust
#[cfg(debug_assertions)]
pub fn print_allocation_stats() {
    unsafe {
        println!("Total allocations: {}", NUM_ALLOCS);
        println!("Total deallocations: {}", NUM_DEALLOCS);
        println!("Leaked: {}", NUM_ALLOCS - NUM_DEALLOCS);
    }
}
```

### 3. Add Bounds Checking

In pmem_hashlist.rs methods that access indices:
```rust
pub fn before<Q>(&self, key: &Q) -> Option<&K>
where
    K: Borrow<Q>,
    Q: Hash + Eq + ?Sized,
{
    let index = *self.map.get(key)?;
    // Add bounds check
    if index + 1 >= self.order.len() {
        return None;
    }
    // Additional safety: verify the index is in bounds
    self.order.get(index + 1)  // Use get() instead of direct index
}
```

## Priority Fixes

1. **HIGH:** Add null pointer checks in allocator dealloc
2. **HIGH:** Add proper unsafe blocks around DRAM_LIMIT_OBJECTS access
3. **MEDIUM:** Add capacity reservation in rebuild_index()
4. **MEDIUM:** Add bounds checking in index access methods
5. **LOW:** Add comprehensive error logging

## Verification Steps

After applying fixes:

1. Run with RUST_BACKTRACE=full
2. Use AddressSanitizer: `RUSTFLAGS="-Z sanitizer=address" cargo +nightly build`
3. Run under valgrind if possible
4. Add extensive logging at allocation points
5. Test with varying cache sizes (small, medium, large)
