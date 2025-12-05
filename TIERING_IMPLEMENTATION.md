# Tiering Manager Implementation

## Overview

This implementation adds a tiering manager that uses eviction stacks to determine which objects should be placed in DRAM vs PMEM tiers.

## Key Components

### 1. TieringManager (`src/worker/tiering_manager.rs`)

The `TieringManager` is responsible for tracking which objects should reside in the DRAM tier based on access patterns derived from the eviction stacks.

**Key Features:**
- Tracks a set of "hot" objects that should be in DRAM
- Provides a `should_prefer_dram()` flag to guide allocation decisions
- Manages promotion and demotion of objects between tiers
- Updates capacity dynamically when cache is resized

**Design Principle:**
- DRAM acts as a cache layer for hot objects
- All objects exist in PMEM (base storage)
- Frequently accessed objects (those at the front of eviction stacks) are also cached in DRAM
- DRAM capacity is configurable as a ratio of total cache size (default: 20%)

### 2. Integration with PolicyWorker (`src/worker/policy/mod.rs`)

The `PolicyWorker` now includes tiering management alongside eviction policy management:

**On object access (GET):**
1. Update eviction stack (existing behavior)
2. Call `manage_dram_tier()` to potentially promote the object to DRAM

**On object insertion (SET):**
1. Insert into eviction stack (existing behavior)
2. Call `manage_dram_tier()` to potentially promote the object to DRAM

**On object deletion (DEL):**
1. Remove from eviction stack (existing behavior)
2. Remove from DRAM tier tracking

**On cache resize:**
1. Resize eviction stacks (existing behavior)
2. Update DRAM tier capacity proportionally

**On cache wipe:**
1. Clear eviction stacks (existing behavior)
2. Clear DRAM tier tracking

### 3. Tiering Logic (`manage_dram_tier()`)

The tiering logic works as follows:

1. When an object is accessed, attempt to promote it to DRAM
2. If DRAM has space, promotion succeeds immediately
3. If DRAM is full:
   - Use the eviction stack to find cold objects currently in DRAM
   - Demote a cold object from DRAM (it remains in PMEM)
   - Promote the newly accessed object to DRAM

This ensures that DRAM contains the most recently accessed objects according to the current eviction policy.

## Configuration

- `DRAM_TIER_RATIO`: Constant in `mod.rs` that determines what fraction of objects can be in DRAM (default: 0.2 = 20%)

## Benefits

1. **Policy-Aware Tiering**: Uses the same eviction stack that determines cache evictions to also determine DRAM placement
2. **Consistent Behavior**: Hot objects (front of eviction stack) are promoted to DRAM; cold objects (back of eviction stack) are demoted
3. **Automatic Adaptation**: Works with any eviction policy (LRU, LFU, FIFO, etc.) and adapts when policies change
4. **Dynamic Sizing**: DRAM capacity adjusts automatically when cache is resized

## Usage

The tiering manager operates automatically and transparently:
- No API changes required
- Tiering decisions are made based on access patterns
- The `should_prefer_dram()` flag can be used by allocators to guide memory placement

## Future Enhancements

The current implementation provides metadata about which objects should be in DRAM. Future work could:
1. Integrate with the allocator to actually control memory placement
2. Implement actual data copying between tiers
3. Add metrics/logging for tier hit rates
4. Support multiple tier levels (e.g., DRAM, PMEM, SSD)
