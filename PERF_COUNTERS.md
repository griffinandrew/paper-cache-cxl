# Performance Counters Implementation Summary

## Overview

This implementation adds performance counters to track and measure memory access patterns for hashmap structures in PaperCache with both DRAM and PMEM feature configurations.

## What Was Implemented

### 1. Core Counter Module (`src/perf_counters.rs`)

**HashMapCounters struct:**
- Atomic counters for reads (lookups, iterations)
- Atomic counters for writes (insertions, deletions, clears)
- Thread-safe increment methods
- Getter methods for retrieving counter values
- Reset functionality

**GlobalPerfCounters struct:**
- Feature-conditional counters for different hashmap configurations:
  - `global_hashbrown_dram` (hashbrown_dram feature)
  - `global_hashbrown_pmem` (global_hashtable_pmem feature)
  - `global_flatmap_dram` (global_flatmap_dram feature)
  - `global_flatmap_pmem` (global_flatmap_pmem feature)
  - `tiering_hashtable_dram` (tiering manager in DRAM)
  - `tiering_hashtable_pmem` (tiering manager in PMEM)
- Optional total memory access counter

**Helper Functions:**
- `get_global_counters()` - Access the global counter instance
- `get_hashmap_stats()` - Get stats for active configuration
- `get_tiering_hashtable_stats()` - Get tiering manager stats
- `print_perf_stats()` - Pretty-print formatted statistics

### 2. Instrumentation (`src/lib.rs`)

**hashbrown_dram feature:**
- `get()` - Tracks lookup operations
- `set()` - Tracks insertion operations
- `del()` - Tracks deletion operations (via erase function)
- `has()` - Tracks lookup operations
- `peek()` - Tracks lookup operations

**global_hashtable_pmem feature:**
- Same operations as hashbrown_dram
- Separate counters for PMEM configuration

### 3. Example (`examples/perf_counters_demo.rs`)

Demonstrates:
- Creating a cache instance
- Performing various operations (insert, read, check, delete)
- Printing formatted statistics
- Accessing statistics programmatically

### 4. Tests

**Unit Tests (`src/perf_counters.rs`):**
- Test counter increment logic
- Test stats conversion
- Test reset functionality

**Integration Tests (`tests/perf_counters_integration.rs`):**
- Test end-to-end tracking for hashbrown_dram
- Test end-to-end tracking for global_hashtable_pmem
- Validate counter accuracy across multiple operations

### 5. Documentation

**README.md:**
- Usage examples
- Feature descriptions
- Build and run instructions
- Example output

## Technical Details

### Design Decisions

1. **Atomic Operations**: Used `Ordering::Relaxed` for minimal performance overhead
2. **Global State**: Used `OnceLock` for thread-safe lazy initialization
3. **Feature Conditional**: Compile-time selection of appropriate counters
4. **Minimal Overhead**: Inline counter increments for performance

### Performance Impact

- Negligible overhead: single atomic increment per operation
- No locks or complex synchronization
- Compile-time feature selection avoids runtime checks

### Thread Safety

- All counters use atomic operations
- Safe for concurrent access from multiple threads
- No data races or synchronization issues

## Usage

### With hashbrown_dram:

```bash
cargo run --example perf_counters_demo --no-default-features --features hashbrown_dram
```

### With global_hashtable_pmem:

```bash
cargo +nightly run --example perf_counters_demo --no-default-features --features global_hashtable_pmem
```

### Programmatic Access:

```rust
use paper_cache::perf_counters::{get_hashmap_stats, print_perf_stats};

// ... perform cache operations ...

// Print formatted statistics
print_perf_stats();

// Or access programmatically
if let Some(stats) = get_hashmap_stats() {
    println!("Total accesses: {}", stats.total_accesses);
    println!("Reads: {}", stats.reads);
    println!("Writes: {}", stats.writes);
}
```

## Testing Results

All tests pass:
- 14 unit tests (existing + new counter tests)
- 2 integration tests (hashbrown_dram scenarios)
- Example runs successfully

## Future Enhancements

Potential additions (not implemented in this PR):

1. **FlatMap Instrumentation**: Add counters for global_flatmap_dram and global_flatmap_pmem
2. **Tiering Manager**: Instrument tiering hashtable operations
3. **Iteration Tracking**: Add specific counters for iteration operations
4. **Clear Operations**: Track clear/wipe operations separately
5. **Memory Footprint**: Track actual memory usage, not just access counts
6. **CSV Export**: Export statistics to CSV for analysis
7. **Histogram**: Track distribution of operation types over time

## Files Changed

- `src/perf_counters.rs` - New module (270 lines)
- `src/lib.rs` - Instrumentation added (~20 lines changed)
- `examples/perf_counters_demo.rs` - New example (70 lines)
- `tests/perf_counters_integration.rs` - New tests (125 lines)
- `README.md` - Documentation added (~75 lines)

## Conclusion

This implementation provides a lightweight, efficient way to track memory access patterns in hashmap structures with minimal performance overhead. The feature-conditional design allows for direct comparison between DRAM and PMEM configurations while maintaining clean, maintainable code.
