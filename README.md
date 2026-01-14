# paper-cache

PaperCache is an in-memory cache which supports the dynamic switching between any eviction policy at runtime.

Note: this crate should not be used directly; please use the paper-server crate instead.

This branch makes it such that both the key and value reside in the pmem tier.
origtinal feature flags for reference... 

## Tiering Manager

The tiering manager provides a two-tier caching architecture with **actual data copies**:
- **Far Tier (PMEM)**: All objects are stored in persistent memory by default (source of truth)
- **Near Tier (DRAM)**: Hot objects are **physically copied** to DRAM for faster access

### Features

- **Automatic Promotion with Data Copying**: Objects accessed frequently are automatically **copied** to DRAM
- **Two-Tier Reads**: Get operations check DRAM cache first, then fall back to PMEM
- **Configurable Thresholds**: 
  - DRAM capacity (default: 20% of cache size)
  - Hotness threshold (default: 2 accesses before promotion)
- **Runtime Controls**: Adjust tiering parameters at runtime without restart
- **Statistics Tracking**: Monitor promotions, demotions, and tier distribution
- **Strong Consistency**: Updates and deletions are applied to both tiers immediately

### API

```rust
use paper_cache::{PaperCache, PaperPolicy};

let cache = PaperCache::<u32, Box<[u8]>>::new(
    10_000_000,  // 10 MB cache
    &[PaperPolicy::Lfu],
    PaperPolicy::Lfu,
).unwrap();

// Get tiering statistics
let stats = cache.tiering_stats();
println!("DRAM objects: {}", stats.dram_objects);
println!("Promotions: {}", stats.promotions);

// Configure DRAM tier size (in bytes)
cache.set_dram_threshold(5_000_000);  // 5 MB for hot objects

// Configure hotness threshold
cache.set_hotness_threshold(3);  // Promote after 3 accesses
```

### How It Works

1. **Object Storage**: All objects are initially stored in PMEM (far tier)
2. **Access Tracking**: Each `get()` operation increments the object's access count
3. **Promotion with Data Copy**: When an object's access count reaches the hotness threshold, its data is **physically copied** to a DRAM cache
4. **Fast Reads**: Subsequent `get()` operations check the DRAM cache first for hot objects, providing faster access
5. **Eviction**: When DRAM reaches capacity, the least recently used objects are demoted (DRAM copies removed, PMEM copies retained)
6. **Consistency**: All updates write to PMEM and update DRAM copies if they exist; deletions remove from both tiers

### Background Workers

The tiering manager runs as a worker thread alongside the policy and TTL workers, processing events from the cache and making promotion/demotion decisions periodically (every 5 seconds).

### Data Copy Model

Unlike simple metadata tracking, this implementation maintains **two physical copies** of hot objects:
- **PMEM copy**: Always exists, serves as source of truth
- **DRAM copy**: Created on promotion, provides fast access, removed on demotion

This ensures hot objects benefit from DRAM speed while maintaining data durability in PMEM.

