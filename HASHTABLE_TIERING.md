# Hashtable Tiering Feature

## Overview

The `hashtable_tiering` feature implements a three-tier object caching strategy that optimizes memory usage and access latency by using DRAM for metadata and pointers while keeping data in CXL/PMEM for warm objects.

## Architecture

### Three-Tier Model

1. **Far Tier (PMEM Only)**: Cold objects reside entirely in CXL/PMEM
   - No DRAM overhead
   - Full latency of CXL access

2. **Warm Tier (DRAM Pointer to PMEM)**: Moderately accessed objects
   - Metadata and pointer in DRAM
   - Actual data remains in CXL/PMEM
   - **Zero-copy** - no data movement, only pointer storage
   - Avoids double hashtable lookup

3. **Hot Tier (DRAM Copy)**: Frequently accessed objects
   - Full physical copy of data in DRAM
   - Fastest access latency
   - Higher DRAM usage

### Promotion Flow

```
PmemOnly → (warm_threshold accesses) → DramPtrToPmem → (hot_threshold accesses) → DramAndPmem
```

## Configuration

### Default Thresholds

- `warm_threshold`: 2 accesses - promotes to pointer tier
- `hot_threshold`: 5 accesses - promotes to full copy tier

### Configuring via TieringConfig

```rust
use paper_cache::TieringConfig;

let mut config = TieringConfig::default();
config.warm_threshold = 3;  // Customize warm tier threshold
config.hot_threshold = 7;   // Customize hot tier threshold
```

## Implementation Details

### Data Storage (TieringData enum)

```rust
pub enum TieringData<V> {
    PhysicalCopy(Arc<Box<[u8]>>),  // Hot tier: DRAM copy
    CxlReference(Arc<V>),           // Warm tier: pointer to CXL
}
```

### Key Components

1. **TieringObject**: Modified to hold either physical copy or CXL reference
2. **Tier enum**: Extended with `DramPtrToPmem` variant
3. **record_access()**: Handles two-stage promotion logic
4. **get()**: Optimized to avoid double lookup for warm tier objects

### Safety Guarantees

- **Arc<V>** ensures CXL memory is not deallocated while DRAM pointers are active
- Zero-copy semantics for warm tier transitions
- Feature-gated to maintain backward compatibility

## Usage

### Building with the Feature

```bash
cargo build --features "key_value_pmem,enable_tiering_manager,hashtable_tiering"
```

### Example Code

```rust
use paper_cache::{PaperCache, PaperPolicy, BufferPMEM};

// Create cache with tiering enabled
let cache = PaperCache::<u32, BufferPMEM>::new(
    10_000_000,
    &[PaperPolicy::Lfu],
    PaperPolicy::Lfu,
).expect("Failed to create cache");

// Set an object
cache.set(1, &[42u8; 100], None).expect("Failed to set");

// Access 1: Object in Far Tier (PMEM only)
cache.get(&1);

// Access 2: Promotes to Warm Tier (pointer in DRAM, data in CXL)
cache.get(&1);

// Accesses 3-5: Build up to hot threshold
for _ in 0..3 {
    cache.get(&1);
}

// Now in Hot Tier (full copy in DRAM)

// Check tiering statistics
let stats = cache.tiering_stats();
println!("DRAM objects: {}", stats.dram_objects);
println!("DRAM size: {}", stats.dram_size);
println!("Promotions: {}", stats.promotions);
```

## Performance Benefits

1. **Reduced DRAM Usage**: Warm objects don't consume DRAM for data storage
2. **Faster Lookups**: Warm tier avoids double hashtable lookup
3. **Adaptive Tiering**: Objects naturally migrate based on access patterns
4. **Zero-Copy Promotion**: Warm tier promotion is instantaneous

## Testing

Integration tests are provided in `tests/tiering_integration.rs`:

```bash
cargo test --features "key_value_pmem,enable_tiering_manager,hashtable_tiering" test_hashtable_tiering
```

Tests verify:
- Warm tier promotion at warm_threshold
- Hot tier promotion at hot_threshold  
- Zero-copy semantics for warm tier
- Data correctness across all tiers

## Requirements

- **Rust Version**: Nightly (requires `allocator_api` feature)
- **Features**: `key_value_pmem`, `enable_tiering_manager`, `hashtable_tiering`
- **External Dependencies**: UMF allocator library (for actual PMEM allocation)

## Backward Compatibility

All changes are properly feature-gated with `#[cfg(feature = "hashtable_tiering")]`. The codebase builds and functions correctly both with and without this feature enabled.
