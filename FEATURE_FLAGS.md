# Feature Flags for Memory Tier Configuration

This document explains the implementation of separate configuration options for persistent memory (PMEM) placement in PaperCache.

## Overview

The implementation provides explicit feature flags to control:
1. Whether all data structures use DRAM
2. Whether key/value data is stored in PMEM
3. Whether the tiering manager is enabled
4. Where the tiering manager's internal hashtable is stored (DRAM vs PMEM)
5. Where the global cache hashtable is stored (DRAM vs PMEM)

## Feature Flags

### `all_dram`
- **Purpose**: Force all allocations to use DRAM (no PMEM usage)
- **When enabled**: All data structures (hashtables, keys, values) are stored in DRAM
- **When disabled**: Default behavior - allocations can use PMEM based on other features
- **Use case**: Baseline performance testing, systems without PMEM

### `key_value_pmem`
- **Purpose**: Store key and value data in PMEM
- **When enabled**: Cache key/value pairs are allocated in PMEM
- **When disabled**: Cache key/value pairs use default allocation (typically DRAM)
- **Requirements**: Mutually exclusive with `all_dram`

### `pmem_region_alloc`
- **Purpose**: Replace UMF per-allocation PMEM calls with a pre-mapped bump allocator region
- **When enabled**: Region allocator implementation (`RegionHybrid`) is compiled and available
- **Use case**: Any PMEM structure (hashtable, eviction stacks, key/value objects) that benefits from low alloc/free overhead over fine-grained reclamation
- **Requirements**: No implicit feature deps; compatible with any PMEM feature (`global_hashtable_pmem`, `eviction_stacks_pmem`, `key_value_pmem`, etc.)

### `region_hybrid_allocator`
- **Purpose**: Select `RegionHybrid` as the crate-wide custom PMEM allocator path
- **When enabled**: `Hybrid` resolves to `RegionHybrid` for allocator-aware PMEM structures
- **When disabled**: `Hybrid` resolves to `HybridObjects` (UMF-backed allocator path), unless `pmem_region_alloc` is enabled
- **Use case**: Force region-backed allocator selection for all `Hybrid`-based PMEM call-sites without changing other PMEM feature combinations; works standalone with `global_hashtable_pmem`, `eviction_stacks_pmem`, or any combination
- **Requirements**: No implicit feature deps; compatible with any PMEM feature combination

### `enable_tiering_manager`
- **Purpose**: Enable/disable the tiering manager functionality
- **When enabled**: Automatic promotion/demotion of hot objects between DRAM and PMEM tiers
- **When disabled**: No automatic tiering, but global hashtable can still be placed in PMEM
- **Requirements**: Works with `key_value_pmem`

### `tiering_hashtable_pmem`
- **Purpose**: Control memory placement of the tiering manager's internal hashtable
- **When enabled**: Tiering manager's hashtable stored in PMEM
- **When disabled**: Tiering manager's hashtable stored in DRAM
- **Requirements**: Requires `enable_tiering_manager` + `key_value_pmem`

### `global_hashtable_pmem`
- **Purpose**: Control memory placement of the main cache hashtable
- **When enabled**: Global cache hashtable stored in PMEM
- **When disabled**: Global cache hashtable stored in DRAM
- **Requirements**: Can be used independently or with `key_value_pmem`

### `hw_perf`
- **Purpose**: Enable hardware performance counters for cache operation profiling
- **When enabled**: Instruments cache lookup and eviction paths with Linux `perf_event` counters
- **When disabled**: Zero cost — all instrumentation is completely compiled out
- **Use case**: Performance analysis of DRAM vs PMEM access patterns (LLC misses, cycles, IPC)
- **Requirements**: Linux only; requires `perf_event` access (may need elevated permissions or `/proc/sys/kernel/perf_event_paranoid` ≤ 1)

### `eviction_stacks_pmem`
- **Purpose**: Allocate eviction policy tracking structures in PMEM using feature-selected PMEM allocators
- **When enabled**: `LfuStack` (`index_map`, `count_stacks`) and `LruStack` (`stack`) internal data structures are PMEM-backed
- **When disabled**: Standard DRAM-backed `std::collections::HashMap` and `kwik::collections::HashList` are used (default)
- **Use case**: Ensures eviction metadata is co-located with PMEM-stored objects for lower cross-tier access overhead
- **Requirements**: Uses `HybridObjects` by default; with `pmem_region_alloc` or `region_hybrid_allocator`, these structures use `RegionHybrid`

### `flatmap_dram`
- **Purpose**: Enable high-performance Linear Probing Hash Map (FlatMap) in DRAM
- **When enabled**: FlatMap module is compiled with DRAM allocator support
- **When disabled**: FlatMap module is not available
- **Use case**: High-performance hash map for DRAM with optimized memory layout
- **Performance**: Reduces cache misses by storing hash, key, and value adjacently

### `flatmap_pmem`
- **Purpose**: Enable standalone FlatMap module with PMEM allocator support
- **When enabled**: FlatMap module is compiled with PMEM allocator support (HybridObjects)
- **When disabled**: FlatMap module is not available
- **Use case**: Standalone high-performance hash map optimized for PMEM latency characteristics
- **Performance**: Reduces PMEM read overhead from 3x to 1x by using flat layout (Array of Structs)
- **Design**: Uses Linear Probing (no Robin Hood hashing) to minimize expensive PMEM writes

### `global_flatmap_dram`
- **Purpose**: Use FlatMap as PaperCache's global hashtable in DRAM
- **When enabled**: `ObjectMapRef` uses `Arc<RwLock<FlatMapWithHasher<..., Global>>>` instead of DashMap
- **When disabled**: Default hashtable implementation is used
- **Use case**: Replace DashMap with FlatMap for better DRAM cache locality
- **Performance**: Better cache utilization due to flat layout, fixed capacity for predictable performance
- **Integration**: Works with all PaperCache operations (get, set, delete, eviction)

### `global_flatmap_pmem`
- **Purpose**: Use FlatMap as PaperCache's global hashtable in PMEM  
- **When enabled**: `ObjectMapRef` uses `Arc<RwLock<FlatMapWithHasher<..., Hybrid>>>` for PMEM
- **When disabled**: Default hashtable implementation is used
- **Use case**: Replace HashMap with FlatMap for optimal PMEM latency
- **Performance**: 3x latency reduction (600ns → 300ns per lookup) compared to hashbrown on PMEM
- **Integration**: Works with all PaperCache operations, uses `remove_unchecked` for eviction without Clone constraints

### `hashbrown_dram`
- **Purpose**: Use hashbrown HashMap as global hashtable in DRAM (for performance comparison)
- **When enabled**: `ObjectMapRef` uses `Arc<RwLock<HashMap<..., NoHasher>>>` in DRAM
- **When disabled**: Default hashtable implementation (DashMap) is used
- **Use case**: Direct performance comparison with `global_hashtable_pmem` using the same hashbrown implementation
- **Performance**: Same hashbrown HashMap implementation as `global_hashtable_pmem` but allocated in DRAM instead of PMEM
- **Requirements**: Mutually exclusive with `global_hashtable_pmem`, `global_flatmap_dram`, and `global_flatmap_pmem`

### `hybridcache`
- **Purpose**: Two-tier cache built by composing *two independent* `PaperCache` instances — a small DRAM
  tier running S3-FIFO and a far PMEM tier running LRU — rather than one unified instance
- **When enabled**: Adds `S3FifoHybridCache<K>`, `HybridCacheConfig`, `HybridCacheStats`, `CacheTierSize`
- **When disabled**: None of the above are compiled
- **Behavior**: Admission always goes to the small DRAM tier. Demotion (small-tier eviction) writes bytes
  to the far PMEM tier asynchronously over a bounded channel. Promotion is driven by the small tier's
  S3-FIFO *ghost queue*: a ghost hit schedules a background re-insertion into the small tier. Uses
  **copy-on-read** — the far-tier (PMEM) copy is never deleted on promotion, so a key can legitimately
  exist in both tiers at once. Contrast with `lru_hybrid_cache` below, which is one unified `PaperCache`
  instance and does real (non-copying) data movement instead
- **Requirements**: `["all_dram", "key_pmem_value_pmem"]` — needs both a DRAM-typed tier and a
  PMEM-typed tier simultaneously, since it's two separate `PaperCache<K, V>` instances with two
  different `V` types (`BufferDRAM` and `BufferPMEM`)

### `lru_hybrid_cache`
- **Purpose**: Single-instance, segmented-LRU hybrid cache — implements the paper design where the LRU
  eviction queue is segmented across a fast (DRAM) tier and a slow (PMEM) tier as two zones of *one*
  logical queue, rather than composing two independent `PaperCache` instances (contrast with
  `hybridcache` above)
- **When enabled**: Adds `PaperPolicy::LruHybrid` and a new `PaperCache<K, TieredBuffer, S>` impl block
  (`new(max_size, fast_tier_size)`, `get`/`set`/`del`/`has`/`peek`/`ttl`/`size`/`wipe`/`resize`, plus
  `set_fast_tier_size`/`fast_tier_size`, `lru_hybrid_stats`, and a `tier_of` diagnostic accessor). Also
  exports `TieredBuffer`, `LruHybridStats`, and `Tier` from the crate root, and shares both the
  `CacheTierSize` unit type (with `hybridcache`) and the `TieredBuffer` value type (with
  `lfu_hybrid_cache`) — see `src/size.rs`/`src/tiered_buffer.rs`, each gated
  `any(hybridcache, lru_hybrid_cache, lfu_hybrid_cache)` or `any(lru_hybrid_cache, lfu_hybrid_cache)`
  respectively
- **When disabled**: None of the above types/methods are compiled; `PaperPolicy::LruHybrid` doesn't exist
- **Behavior**: Every `set()` admits (or re-admits) the object at the top of the fast tier. Whenever
  fast-tier usage exceeds the configured fast-tier byte budget, the least-recently-used fast-tier object
  is demoted to the slow tier. Accessing (`get()`) a slow-tier object promotes it back to the top of the
  fast tier — possibly cascading a further demotion if the fast tier is now over budget. Once the
  cache's overall `max_size` is exceeded, the least-recently-used *slow-tier* object is evicted (counted
  in `lru_hybrid_stats().evictions`). Every promotion/demotion is **actual data movement**
  (`Object::set_data` swaps a `TieredBuffer::Fast(Box<[u8]>)` for a `TieredBuffer::Slow(Box<[u8], Hybrid>)`,
  or vice versa) — a live object's bytes exist in exactly one tier's allocation at a time, never copied
  into both. TTL survives every tier move unmodified, since a migration only ever replaces
  `Object::data`, never `key` or `expiry`
- **Requirements**: `["key_value_pmem"]` only — *not* `all_dram` + `key_pmem_value_pmem` like
  `hybridcache`. A plain `Box<[u8]>` (the fast-tier representation) already allocates through the
  crate's global DRAM allocator (`DRAMObjects`) regardless of feature flags, and this feature only needs
  to migrate *value* bytes between tiers, so the smaller `key_value_pmem` dependency (which makes
  `BufferPMEM`/`Hybrid` available without forcing the *key* into PMEM) is sufficient and keeps keys
  DRAM-resident. **Mutually exclusive with `lfu_hybrid_cache`** (both define their own inherent-method
  impl block on the identical `PaperCache<K, TieredBuffer, S>` type; `lib.rs` has a `compile_error!`
  guard rejecting both enabled together)
- **Use case**: A single unified cache whose "two tiers" are a property of where each object's bytes
  currently live (not two separate caches/hashtables); useful for comparing against `hybridcache`'s
  two-`PaperCache`-instance, copy-on-read design for the same fast-DRAM/slow-PMEM workload shape

### `lfu_hybrid_cache`
- **Purpose**: Single-instance, segmented-LFU hybrid cache — same one-`PaperCache<K, TieredBuffer>`
  architecture as `lru_hybrid_cache` above, but the fast/slow boundary is *frequency*-ordered rather
  than recency-ordered: the most-frequently-accessed objects belong in the fast tier
- **When enabled**: Adds `PaperPolicy::LfuHybrid` and a new `PaperCache<K, TieredBuffer, S>` impl block
  — identical method surface to `lru_hybrid_cache`'s (`new`, `get`/`set`/`del`/`has`/`peek`/`ttl`/`size`/
  `wipe`/`resize`, `set_fast_tier_size`/`fast_tier_size`, `tier_of`), plus `lfu_hybrid_stats`. Also
  exports `LfuHybridStats` from the crate root; shares `TieredBuffer`/`Tier`/`CacheTierSize` with
  `lru_hybrid_cache` rather than duplicating them
- **When disabled**: None of the above types/methods are compiled; `PaperPolicy::LfuHybrid` doesn't exist
- **Behavior**: While the fast tier has spare capacity, new objects are admitted there; internally,
  admission always lands in the fast chain first (mirroring `lru_hybrid_cache`'s "admit fast, let
  settle demote if needed" design) rather than being special-cased to route straight to slow — this
  still satisfies the paper's admission rule as an emergent result, since a freshly admitted object
  (frequency 1) is always tied for the fast tier's lowest frequency once the tier is full. Demotion:
  whenever fast-tier usage exceeds the configured byte budget, the lowest-frequency fast-tier object is
  moved to the slow tier (ties within the same frequency break toward whichever key is
  least-recently-touched, matching the plain `LfuStack` policy's existing convention). Promotion:
  accessing (`get()`) a slow-tier object bumps its frequency; once that frequency *strictly* exceeds the
  minimum frequency among fast-tier residents, it's promoted back to the fast tier — possibly cascading
  a further demotion. A tie does not promote. Eviction: once the cache's overall `max_size` is exceeded,
  the lowest-frequency *slow-tier* object is evicted (falling back to the fast tier's own minimum if the
  slow tier happens to be empty), counted in `lfu_hybrid_stats().evictions`. Every promotion/demotion is
  actual data movement, same as `lru_hybrid_cache` — a live object's bytes exist in exactly one tier's
  allocation at a time. TTL survives every tier move unmodified for the same reason (`Object::set_data`
  only ever replaces `data`, never `key` or `expiry`)
- **Requirements**: `["key_value_pmem"]` only, same reasoning as `lru_hybrid_cache`. **Mutually exclusive
  with `lru_hybrid_cache`** (see above)
- **Use case**: Same "single unified cache" shape as `lru_hybrid_cache`, for workloads where recency
  alone is a poor eviction signal and access-frequency skew should determine what stays in the fast tier
- **A subtlety worth knowing**: `fast_capacity` (the policy stack's internal fast/slow byte budget) and
  `max_size` (the overall eviction budget) are tracked in different units — `fast_capacity` only counts
  raw `base_size` (key + value + expiry-slot bytes), while `max_size`'s accounting additionally adds a
  fixed per-object policy overhead (`get_policy_overhead`, `object/overhead.rs`) on top. Setting
  `fast_tier_size == max_size` at construction does *not* guarantee nothing ever demotes: enough small
  objects can accumulate in raw bytes to exceed `fast_capacity` well before their overhead-inclusive
  total exceeds `max_size`. This is a general property of the accounting design (applies to
  `lru_hybrid_cache` too), not a bug — see `tests/lfu_hybrid_cache_integration.rs`'s
  `terminal_eviction_falls_back_to_fast_tier_when_slow_tier_is_empty` test for how to reliably construct
  a "slow tier stays empty" scenario (pace admissions so eviction keeps up, rather than relying on the
  capacity numbers alone)

## Implementation Details

### Type System

The implementation uses Rust's conditional compilation to select the appropriate hashtable types:

**Tiering Manager's hashtable** (`dram_cache` in `TieringManager`):
- Without `tiering_hashtable_pmem`: `DashMap` or `HashMap` (DRAM)
- With `tiering_hashtable_pmem`: `HashMap<..., Hybrid>` (PMEM)

**Global hashtable** (`objects` in `PaperCache`):
- Default (no FlatMap): `DashMap` (DRAM)
- With `global_hashtable_pmem`: `RwLock<HashMap<..., Hybrid>>` (PMEM)
- With `hashbrown_dram`: `RwLock<HashMap<..., NoHasher>>` (DRAM)
- With `global_flatmap_dram`: `Arc<RwLock<FlatMapWithHasher<..., Global>>>` (DRAM)
- With `global_flatmap_pmem`: `Arc<RwLock<FlatMapWithHasher<..., Hybrid>>>` (PMEM)

**FlatMap** (high-performance Linear Probing Hash Map):
- With `flatmap_dram`: Uses Global allocator (DRAM)
- With `flatmap_pmem`: Uses HybridObjects allocator (PMEM)
- Flat layout: `Vec<Bucket<K, V>, A>` where `Bucket` is `#[repr(C)]` with `{ hash: u64, key: K, val: V }`
- Operations: `insert`, `get`, `get_mut`, `remove`, `contains_key`, `clear`, `iter`
- Algorithm: Linear probing with `(index + 1) & mask` collision resolution
- Fixed capacity (no resizing) for optimal performance

### Allocator Integration

The `Hybrid` allocator is used to place data in PMEM:
- Defined in `src/allocator.rs`
- Implements both `GlobalAlloc` and `Allocator` traits
- Default PMEM path (`HybridObjects`) routes allocations through UMF
- Region PMEM path (`RegionHybrid`, selected via `pmem_region_alloc` or `region_hybrid_allocator`) uses one large `mmap` region, optional NUMA `mbind`, lock-free bump allocation, no-op deallocate, and bulk reclaim via `reclaim_all`/`reset_epoch`

### Conditional Compilation

The code uses `#[cfg(...)]` attributes extensively to:
1. Include/exclude the tiering module based on `enable_tiering_manager`
2. Select different hashtable implementations based on pmem flags
3. Conditionally compile tiering-related methods in PaperCache

## Valid Feature Combinations

### Basic Configurations

1. **All in DRAM** (baseline performance):
   ```toml
   features = ["all_dram"]
   ```

2. **Key/Value data in PMEM**:
   ```toml
   features = ["key_value_pmem"]
   ```

3. **Global hashtable in PMEM only** (data in DRAM, hashtable in PMEM):
   ```toml
   features = ["global_hashtable_pmem"]
   ```

### With Tiering Manager

4. **Tiering with key/value in PMEM**:
   ```toml
   features = ["enable_tiering_manager", "key_value_pmem"]
   ```

5. **Tiering + global hashtable in PMEM**:
   ```toml
   features = ["enable_tiering_manager", "key_value_pmem", "global_hashtable_pmem"]
   ```

6. **Tiering + tiering hashtable in PMEM**:
   ```toml
   features = ["enable_tiering_manager", "key_value_pmem", "tiering_hashtable_pmem"]
   ```

7. **Tiering + both hashtables in PMEM** (maximum persistence):
   ```toml
   features = ["enable_tiering_manager", "key_value_pmem", "tiering_hashtable_pmem", "global_hashtable_pmem"]
   ```

8. **Hardware performance counters** (zero-cost when disabled):
   ```toml
   features = ["hw_perf"]
   ```

9. **PMEM-backed eviction stacks + key/value in PMEM**:
   ```toml
   features = ["eviction_stacks_pmem", "key_value_pmem"]
   ```

10. **Full PMEM stack** (tiering + eviction stacks + hw profiling):
    ```toml
    features = ["enable_tiering_manager", "key_value_pmem", "eviction_stacks_pmem", "hw_perf"]
    ```

## Performance Characteristics

### Memory Tier Comparison

**DRAM**:
- ✅ Fast access (CPU cache speeds)
- ✅ Low latency for lookups and insertions
- ❌ Volatile - lost on restart
- ❌ Limited by DRAM capacity

**PMEM**:
- ❌ Slower access (higher latency than DRAM)
- ❌ Higher latency for operations
- ✅ Persistent across restarts
- ✅ Larger capacity available

### Use Cases

1. **`all_dram`**: Maximum performance, baseline comparison
2. **`key_value_pmem`**: Persistent data, volatile metadata
3. **`global_hashtable_pmem`**: Test hashtable in PMEM independently
4. **Tiering + both in PMEM**: Maximum durability, cache state survives restarts
5. **Tiering + global in PMEM**: Persistent cache data, fast tiering decisions
6. **Tiering + tiering in PMEM**: Fast cache access, persistent tiering metadata
7. **`flatmap_dram`**: Standalone FlatMap module in DRAM
8. **`flatmap_pmem`**: Standalone FlatMap module in PMEM
9. **`global_flatmap_dram`**: Use FlatMap as PaperCache's main hashtable in DRAM
10. **`global_flatmap_pmem`**: Use FlatMap as PaperCache's main hashtable in PMEM (3x latency reduction)
11. **`hashbrown_dram`**: Use hashbrown HashMap in DRAM for direct performance comparison with `global_hashtable_pmem`
12. **`hw_perf`**: Hardware performance counters for profiling (zero-cost when disabled)
13. **`eviction_stacks_pmem`**: LFU eviction stacks allocated in PMEM for co-location with PMEM objects

## Code Locations

- **Feature definitions**: `Cargo.toml`
- **Allocator**: `src/allocator.rs`
- **FlatMap**: `src/flatmap.rs`
- **Tiering manager hashtable**: `src/tiering/manager.rs`
- **Global hashtable**: `src/lib.rs`
- **Worker manager integration**: `src/worker/manager.rs`
- **Hardware perf counters**: `src/hw_perf_counters.rs`
- **PMEM eviction collections**: `src/worker/policy/policy_stack/pmem_collections.rs`
- **LFU policy stack**: `src/worker/policy/policy_stack/lfu_stack.rs`
- **hybridcache (two-instance hybrid cache)**: `src/hybridcache/mod.rs`
- **lru_hybrid_cache (single-instance hybrid cache)**: `src/lru_hybrid_cache/` (`stats.rs` for
  `LruHybridStats`), `src/worker/policy/policy_stack/lru_hybrid_stack.rs`
  (`LruHybridStack`), `src/policy.rs` (`PaperPolicy::LruHybrid`), `src/status.rs` (counters/gauges +
  fast-tier capacity), `PaperCache<K, TieredBuffer, S>` impl block in `src/lib.rs`. Shared tier-size
  unit type: `src/size.rs` (`CacheTierSize`). Shared value type: `src/tiered_buffer.rs`
  (`TieredBuffer`). See `CLAUDE.md` and `LRU_HYBRID_CACHE.md` for the full design writeup.
- **lfu_hybrid_cache (single-instance, frequency-segmented hybrid cache)**: `src/lfu_hybrid_cache/`
  (`stats.rs` for `LfuHybridStats`), `src/worker/policy/policy_stack/lfu_hybrid_stack.rs`
  (`LfuHybridStack` + its internal `FrequencyChain` helper), `src/policy.rs` (`PaperPolicy::LfuHybrid`),
  `src/status.rs` (`lfu_hybrid_*` counters/gauges), `PaperCache<K, TieredBuffer, S>` impl block in
  `src/lib.rs`. Reuses `src/tiered_buffer.rs`/`src/size.rs` from `lru_hybrid_cache` rather than
  duplicating them (the two features are mutually exclusive).

## Testing

To test different combinations (requires nightly Rust for allocator features):

```bash
# Test standalone FlatMap in DRAM
cargo +nightly test --lib flatmap::tests --no-default-features --features flatmap_dram

# Test standalone FlatMap in PMEM (requires PMEM hardware)
cargo +nightly test --lib flatmap::tests --no-default-features --features flatmap_pmem

# Check standalone FlatMap compilation for DRAM
cargo +nightly check --no-default-features --features flatmap_dram

# Check standalone FlatMap compilation for PMEM
cargo +nightly check --no-default-features --features flatmap_pmem

# Check FlatMap as PaperCache hashtable in DRAM
cargo +nightly check --no-default-features --features global_flatmap_dram

# Check FlatMap as PaperCache hashtable in PMEM
cargo +nightly check --no-default-features --features global_flatmap_pmem

# Check hashbrown HashMap in DRAM (for performance comparison)
cargo +nightly check --no-default-features --features hashbrown_dram

# Test with tiering and both hashtables in PMEM
cargo +nightly check --no-default-features --features flatmap_pmem

# Test with tiering and both hashtables in PMEM
cargo +nightly check --features "enable_tiering_manager,tiering_hashtable_pmem,global_hashtable_pmem,key_value_pmem"

# Test without tiering, global hashtable in PMEM (global cache only)
cargo +nightly check --features "global_hashtable_pmem,key_value_pmem"

# Test with tiering, neither hashtable in PMEM
cargo +nightly check --features "enable_tiering_manager,key_value_pmem"

# Test baseline without any features
cargo +nightly check --no-default-features

# Test global cache with allocator but no tiering
cargo +nightly check --features "key_value_pmem"

# Verify hw_perf (hardware counters - zero-cost abstraction when disabled)
cargo +nightly check --features "hw_perf"

# Verify eviction_stacks_pmem with key_value_pmem
cargo +nightly check --features "hw_perf,eviction_stacks_pmem,key_value_pmem"

# Verify full feature combination
cargo +nightly check --features "enable_tiering_manager,eviction_stacks_pmem,key_value_pmem"

# Check hybridcache (two-instance hybrid cache)
cargo +nightly check --features hybridcache

# Run hybridcache's PMEM integration tests (requires real PMEM/UMF hardware)
cargo +nightly test --test hybridcache_integration --features hybridcache

# Check lru_hybrid_cache (single-instance hybrid cache)
cargo +nightly check --features lru_hybrid_cache

# Run lru_hybrid_cache's unit + inline tests
cargo +nightly test --lib --features lru_hybrid_cache

# Run lru_hybrid_cache's PMEM integration tests (requires real PMEM/UMF hardware)
cargo +nightly test --test lru_hybrid_cache_integration --features lru_hybrid_cache

# Check lfu_hybrid_cache (single-instance, frequency-segmented hybrid cache)
cargo +nightly check --features lfu_hybrid_cache

# Run lfu_hybrid_cache's unit + inline tests
cargo +nightly test --lib --features lfu_hybrid_cache

# Run lfu_hybrid_cache's PMEM integration tests (requires real PMEM/UMF hardware)
cargo +nightly test --test lfu_hybrid_cache_integration --features lfu_hybrid_cache

# Confirm lru_hybrid_cache and lfu_hybrid_cache are mutually exclusive (expected to fail to compile)
cargo +nightly check --features lru_hybrid_cache,lfu_hybrid_cache
```

**Note**: The tiering worker module is only compiled when BOTH an allocator feature 
AND `enable_tiering_manager` are enabled. This ensures the tiering manager is not 
used at all when disabled, allowing the cache to operate as a single global cache.

## Acceptance Criteria

✅ Separate configuration/feature flags for each hashtable's pmem placement
✅ Tiering manager hashtable can be placed in pmem independently
✅ Global hashtable can be placed in pmem independently
✅ All combinations work correctly (conditional compilation ensures validity)
✅ No performance regression when features are disabled (different code paths compiled)
✅ Tiering manager can be turned on/off independently
✅ Global hashtable can use pmem even when tiering is disabled
✅ `alloc_api_exp` removed; Hybrid allocator still functional for `key_value_pmem`
✅ `hw_perf` counters compile out to zero cost when feature is disabled
✅ `eviction_stacks_pmem` correctly allocates LFU/LRU stacks via `Hybrid` (UMF or `RegionHybrid` when `pmem_region_alloc` / `region_hybrid_allocator` are enabled)
✅ `hybridcache` composes two `PaperCache` instances (DRAM small tier, PMEM far tier) with real PMEM writes/reads via `Hybrid`
✅ `lru_hybrid_cache` implements a single unified `PaperCache<K, TieredBuffer>` with a segmented-LRU
  fast/slow boundary; promotion/demotion verified as real data movement (never present in both tiers)
  end to end on real PMEM hardware (`tests/lru_hybrid_cache_integration.rs`, 14/14 passing)
✅ `lfu_hybrid_cache` implements the same single-unified-instance architecture with a segmented-LFU
  (frequency-ordered, not recency-ordered) fast/slow boundary; promotion/demotion verified as real data
  movement end to end on real PMEM hardware (`tests/lfu_hybrid_cache_integration.rs`, 17/17 passing).
  Mutually exclusive with `lru_hybrid_cache` at compile time (verified via `compile_error!`)
