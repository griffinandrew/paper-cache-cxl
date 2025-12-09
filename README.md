# paper-cache

PaperCache is an in-memory cache which supports the dynamic switching between any eviction policy at runtime.

Note: this crate should not be used directly; please use the paper-server crate instead.

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

### External Configuration File

You can configure the tiering manager parameters using an external configuration file. Both JSON and TOML formats are supported.

#### Using Configuration Files

```rust
use paper_cache::{TieringConfig, TieringManager};

// Load from JSON file
let config = TieringConfig::from_json_file("tiering_config.json")
    .expect("Failed to load config");

// Or load from TOML file
let config = TieringConfig::from_toml_file("tiering_config.toml")
    .expect("Failed to load config");

// Or auto-detect format by file extension
let config = TieringConfig::from_file("tiering_config.json")
    .expect("Failed to load config");

// Create tiering manager with loaded config
let manager = TieringManager::new(config);
```

#### Example JSON Configuration

```json
{
  "dram_threshold": 1073741824,
  "high_water_mark": 0.95,
  "low_water_mark": 0.7,
  "hotness_threshold": 2
}
```

#### Example TOML Configuration

```toml
# DRAM threshold in bytes (1 GB default)
dram_threshold = 1073741824

# High water mark (95% of threshold)
high_water_mark = 0.95

# Low water mark (70% of threshold)
low_water_mark = 0.7

# Hotness threshold (promote after 2 accesses)
hotness_threshold = 2
```

#### Configuration Parameters

- **dram_threshold**: Maximum size of DRAM tier in bytes (default: 1 GB)
- **high_water_mark**: Percentage of threshold to trigger demotion (default: 0.95)
- **low_water_mark**: Target percentage after demotion (default: 0.7)
- **hotness_threshold**: Minimum accesses before promotion (default: 2)

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

