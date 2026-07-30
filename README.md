# paper-cache
claude edit

PaperCache is an in-memory cache which supports the dynamic switching between any eviction policy at runtime.

Note: this crate should not be used directly; please use the paper-server crate instead.

This branch makes it such that both the key and value reside in the pmem tier.

Here we will add a custom basic hashmap because the architecture of the swissmap used by both dashmap and hashbrown hashmap might be hurting perf in a pmem scenario

trying with an admision policy structure....

adding new tiering manager... 

this is the working original branch.... 

add comprehensive functionality from all other repos

stage copy before trying more advnaced tiering configs...

this should add key_pmem_value_pmem to the enable tiering feature....




this branch will be for adding s3fifo tiered cache...


when the feature global hashtbale is enabled... the key and tll should also live in dram...

stage new branch for new agent task

new branch for removanble of FFI calls

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

## Feature Flags for Memory Tier Configuration

PaperCache provides fine-grained control over memory placement through feature flags:

### Tiering Manager Control

- **`enable_tiering_manager`**: Enables the tiering manager functionality
  - When enabled: Automatic promotion/demotion between DRAM and PMEM tiers
  - When disabled: No automatic tiering, but global hashtable can still be placed in PMEM

### Hashtable Memory Placement

Independent control over where hashtables are stored (requires one of the allocator features):

- **`tiering_hashtable_pmem`**: Places the tiering manager's internal hashtable in persistent memory
  - Requires: `enable_tiering_manager` + `key_value_pmem`
  - When disabled: Tiering hashtable uses DRAM

- **`global_hashtable_pmem`**: Places the main cache hashtable in persistent memory
  - Requires: `key_value_pmem`
  - Works independently of tiering manager
  - When disabled: Global hashtable uses DRAM

### Usage Examples

#### All combinations with tiering enabled:

```toml
# Both hashtables in PMEM (maximum persistence)
[dependencies]
paper-cache = { features = ["enable_tiering_manager", "tiering_hashtable_pmem", "global_hashtable_pmem", "key_value_pmem"] }

# Only global hashtable in PMEM (cache data persistent, tiering metadata in DRAM)
[dependencies]
paper-cache = { features = ["enable_tiering_manager", "global_hashtable_pmem", "key_value_pmem"] }

# Only tiering hashtable in PMEM (tiering metadata persistent, cache data in DRAM)
[dependencies]
paper-cache = { features = ["enable_tiering_manager", "tiering_hashtable_pmem", "key_value_pmem"] }

# Neither in PMEM (all in DRAM, baseline performance)
[dependencies]
paper-cache = { features = ["enable_tiering_manager", "key_value_pmem"] }
```

#### Without tiering manager:

```toml
# Global hashtable in PMEM without automatic tiering
[dependencies]
paper-cache = { features = ["global_hashtable_pmem", "key_value_pmem"] }
```

### Performance Considerations

- **PMEM hashtables**: Slower access but persistent across restarts
- **DRAM hashtables**: Faster access but volatile
- **Tiering manager off + global in PMEM**: Simplest persistent cache without automatic promotion/demotion overhead

### Required Base Features

These feature flags work in conjunction with:
- `key_value_pmem`: Place key and value data in PMEM

