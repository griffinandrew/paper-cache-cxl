# Tiering Manager

A prototype tiering manager for moving objects between DRAM and PMEM based on configurable threshold values.

## Overview

The tiering manager provides intelligent object placement between fast DRAM and slower PMEM storage tiers. It maintains PMEM as the source of truth while caching hot objects in DRAM for improved performance.

### Key Features

- **Threshold-based Management**: Configurable DRAM usage threshold with high/low water marks
- **Copy-on-Promote**: Objects in DRAM are copies of those in PMEM (PMEM is source of truth)
- **Automatic Demotion**: LRU-based demotion when DRAM usage exceeds threshold
- **Access Tracking**: Records object accesses to guide promotion decisions
- **Statistics**: Real-time tracking of promotions, demotions, and tier distribution

## Architecture

### Design Principles

1. **PMEM as Source of Truth**: All objects exist in PMEM. DRAM contains hot copies only.
2. **Existing Eviction Stacks**: Uses the cache's existing eviction stacks for consistency
3. **Threshold-Based Policy**: DRAM capacity managed via configurable threshold
4. **LRU Demotion**: Least Recently Used objects are demoted first when threshold is exceeded

### Configuration

```rust
use paper_cache::TieringConfig;

let config = TieringConfig {
    dram_threshold: 1_073_741_824,  // 1 GB DRAM limit
    high_water_mark: 0.9,            // Start demoting at 90%
    low_water_mark: 0.7,             // Demote until 70%
};
```

- **dram_threshold**: Maximum DRAM usage in bytes
- **high_water_mark**: Percentage (0.0-1.0) of threshold at which to start demotion
- **low_water_mark**: Percentage to demote down to

## Usage

### Basic Example

```rust
use paper_cache::{TieringManager, TieringConfig};

// Create manager with default config (1GB threshold)
let manager = TieringManager::with_defaults();

// Register an object in PMEM
manager.register_object(key, size_in_bytes);

// Track accesses (promotes after threshold hits)
if manager.record_access(key) {
    manager.promote_to_dram(key);
}

// Check and perform automatic demotion
let keys_to_demote = manager.get_keys_to_demote();
for key in keys_to_demote {
    manager.demote_from_dram(key);
}

// Get statistics
let stats = manager.stats();
println!("DRAM objects: {}", stats.dram_objects);
println!("DRAM usage: {} bytes", stats.dram_size);
```

### Integration with Cache

The tiering manager is designed to work alongside the existing cache implementation:

1. **On Set**: Register object in PMEM tier
2. **On Get**: Record access; promote if access threshold met
3. **Before Eviction**: Check for demotion needs; demote cold objects
4. **On Delete**: Remove object from tracking

## API Reference

### TieringManager

#### Methods

- `new(config: TieringConfig) -> Self` - Create with custom config
- `with_defaults() -> Self` - Create with default 1GB threshold
- `register_object(key: HashedKey, size: ObjectSize)` - Register new object in PMEM
- `record_access(key: HashedKey) -> bool` - Record access, returns true if should promote
- `promote_to_dram(key: HashedKey) -> bool` - Promote object to DRAM (copy)
- `demote_from_dram(key: HashedKey) -> bool` - Demote object from DRAM
- `get_keys_to_demote() -> Vec<HashedKey>` - Get LRU keys to demote
- `remove_object(key: HashedKey)` - Remove object from tracking
- `is_in_dram(key: &HashedKey) -> bool` - Check if object is in DRAM
- `stats() -> TieringStats` - Get current statistics
- `set_dram_threshold(threshold: u64)` - Update DRAM threshold
- `clear()` - Clear all tracking (for cache wipe)

### TieringStats

Statistics structure:

```rust
pub struct TieringStats {
    pub dram_objects: u64,      // Count of objects in DRAM
    pub dram_size: u64,         // Total DRAM usage in bytes
    pub promotions: u64,        // Total promotions to DRAM
    pub demotions: u64,         // Total demotions from DRAM
    pub pmem_only_objects: u64, // Count of PMEM-only objects
}
```

## Examples

Run the included example:

```bash
cargo run --example tiering_demo
```

This demonstrates:
- Registering objects in PMEM
- Access-based promotion to DRAM
- Threshold enforcement
- Automatic demotion when capacity exceeded
- Statistics tracking

## Testing

Run the tiering manager tests:

```bash
# Run all tiering tests
cargo test tiering

# Run integration tests
cargo test --test tiering_integration
```

Test coverage includes:
- Basic promotion/demotion operations
- Threshold enforcement
- Automatic demotion logic
- Access-based promotion
- Statistics tracking
- Edge cases (remove, clear, etc.)

## Performance Considerations

1. **DRAM Access**: Objects in DRAM tier have faster access (copy returned)
2. **PMEM Fallback**: PMEM-only objects require PMEM access
3. **Promotion Overhead**: Copying objects to DRAM has a one-time cost
4. **Demotion**: Simply removes DRAM copy, PMEM remains untouched
5. **Tracking**: HashMap-based tracking with RwLock for thread safety

## Future Enhancements

Potential improvements for the tiering manager:

1. **Multiple Policies**: Support LFU, MRU, or custom demotion policies
2. **Async Operations**: Background promotion/demotion threads
3. **Hot/Cold Classification**: More sophisticated heat detection
4. **Statistics Export**: Metrics for monitoring and analysis
5. **Dynamic Thresholds**: Adaptive threshold based on workload
6. **Batched Operations**: Batch promote/demote for efficiency

## License

This source code is licensed under the GNU AGPLv3 license found in the LICENSE file in the root directory of this source tree.
