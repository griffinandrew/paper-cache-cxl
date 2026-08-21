# Feature Flags for Memory Tier Configuration

This document explains the implementation of separate configuration options for persistent memory (PMEM) placement in PaperCache.

> **Scope note.** The `*_hybrid_cache` sections below cover only the first six designs, in the
> detail they were written up with at the time. There are now **18**, and this file has not
> kept pace. For the complete set — what each one does, and how they relate — see
> `HYBRID_CACHES.md`; `Cargo.toml` carries a comment on every feature. The placement flags in
> this file (`all_dram`, `key_value_pmem`, the hashtable flags, `eviction_stacks_pmem`) are
> current.

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

### `eviction_stacks_pmem`
- **Purpose**: Allocate eviction policy tracking structures in PMEM using feature-selected PMEM allocators
- **When enabled**: `LfuStack` (`index_map`, `count_stacks`) and `LruStack` (`stack`) internal data structures are PMEM-backed
- **When disabled**: Standard DRAM-backed `std::collections::HashMap` and `kwik::collections::HashList` are used (default)
- **Use case**: Ensures eviction metadata is co-located with PMEM-stored objects for lower cross-tier access overhead
- **Requirements**: Uses `Hybrid` (`numa_alloc::SlowObjects`, node-1-bound jemalloc arenas)

### `hashbrown_dram`
- **Purpose**: Use hashbrown HashMap as global hashtable in DRAM (for performance comparison)
- **When enabled**: `ObjectMapRef` uses `Arc<RwLock<HashMap<..., NoHasher>>>` in DRAM
- **When disabled**: Default hashtable implementation (DashMap) is used
- **Use case**: Direct performance comparison with `global_hashtable_pmem` using the same hashbrown implementation
- **Performance**: Same hashbrown HashMap implementation as `global_hashtable_pmem` but allocated in DRAM instead of PMEM
- **Requirements**: Mutually exclusive with `global_hashtable_pmem`

### `lru_hybrid_cache`
- **Purpose**: Single-instance, segmented-LRU hybrid cache — implements the paper design where the LRU
  eviction queue is segmented across a fast (DRAM) tier and a slow (PMEM) tier as two zones of *one*
  logical queue, rather than composing two independent `PaperCache` instances
- **When enabled**: Adds `PaperPolicy::LruHybrid` and a new `PaperCache<K, TieredBuffer, S>` impl block
  (`new(max_size, fast_tier_size)`, `get`/`set`/`del`/`has`/`peek`/`ttl`/`size`/`wipe`/`resize`, plus
  `set_fast_tier_size`/`fast_tier_size`, `lru_hybrid_stats`, and a `tier_of` diagnostic accessor). Also
  exports `TieredBuffer`, `LruHybridStats`, and `Tier` from the crate root, and shares both the
  `CacheTierSize` unit type and the `TieredBuffer` value type with `lfu_hybrid_cache`/
  `two_q_hybrid_cache`/`fifo_hybrid_cache` — see `src/size.rs`/`src/tiered_buffer.rs`, each gated
  `any(lru_hybrid_cache, lfu_hybrid_cache, two_q_hybrid_cache, fifo_hybrid_cache)`
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
- **Requirements**: `["key_value_pmem"]` only. A plain `Box<[u8]>` (the fast-tier representation)
  already allocates through the crate's global DRAM allocator (`numa_alloc::FastAlloc`, node-0-bound jemalloc arenas) regardless of feature
  flags, and this feature only needs to migrate *value* bytes between tiers, so the smaller
  `key_value_pmem` dependency (which makes `BufferPMEM`/`Hybrid` available without forcing the *key*
  into PMEM) is sufficient and keeps keys DRAM-resident. **Mutually exclusive with `lfu_hybrid_cache`**
  (both define their own inherent-method impl block on the identical `PaperCache<K, TieredBuffer, S>`
  type; `lib.rs` has a `compile_error!` guard rejecting both enabled together)
- **Use case**: A single unified cache whose "two tiers" are a property of where each object's bytes
  currently live, with real (non-copying) data movement between them, for the same fast-DRAM/
  slow-PMEM workload shape

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
- **Behavior**: Admission does an explicit capacity check before touching the fast chain: while
  `fast_used + size <= fast_capacity`, a new object is admitted to the fast tier; once the fast tier
  is full, every subsequent new object is admitted directly to the slow tier instead — matching the
  paper's admission rule literally ("every new object is admitted into the slow tier") rather than
  relying on frequency tie-breaking to decide who ends up slow. (An earlier implementation admitted
  every new key to the fast chain unconditionally and let tie-breaking demote whoever lost, which
  could demote an *older* resident instead of the newcomer — a real deviation from the paper's spec,
  fixed by making the capacity check explicit at admission time.) Demotion: whenever fast-tier usage
  exceeds the configured byte budget as a result of a *promotion* (see below) or a `resize_fast_tier`
  call, the lowest-frequency fast-tier object is moved to the slow tier (ties within the same
  frequency break toward whichever key is least-recently-touched, matching the plain `LfuStack`
  policy's existing convention) — plain admission never triggers this, since it no longer touches
  the fast chain once that chain is full. Promotion:
  accessing (`get()`) a slow-tier object bumps its frequency; once that frequency *strictly* exceeds the
  minimum frequency among fast-tier residents, it's promoted back to the fast tier — possibly cascading
  a further demotion. A tie does not promote. Eviction: once the cache's overall `max_size` is exceeded,
  the lowest-frequency *slow-tier* object is evicted (falling back to the fast tier's own minimum if the
  slow tier happens to be empty), counted in `lfu_hybrid_stats().evictions`. Every promotion/demotion is
  actual data movement, same as `lru_hybrid_cache` — a live object's bytes exist in exactly one tier's
  allocation at a time. TTL survives every tier move unmodified for the same reason (`Object::set_data`
  only ever replaces `data`, never `key` or `expiry`)
- **Requirements**: `["key_value_pmem"]` only, same reasoning as `lru_hybrid_cache`. **Mutually exclusive
  with `lru_hybrid_cache`/`two_q_hybrid_cache`** (see above / below)
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

### `two_q_hybrid_cache`
- **Purpose**: Single-instance, segmented-2Q hybrid cache — same one-`PaperCache<K, TieredBuffer>`
  architecture as `lru_hybrid_cache`/`lfu_hybrid_cache` above, but new objects are never admitted
  directly to the fast tier at all: every `set()` places the object in a one-access FIFO queue that
  lives entirely in the slow tier, and only a re-access promotes it into a main LRU queue segmented
  fast/slow (which then behaves exactly like `lru_hybrid_cache`)
- **When enabled**: Adds `PaperPolicy::TwoQHybrid(f64)` (carries `k_in`, the FIFO queue's own byte
  budget as a fraction of `max_size` — unlike `LruHybrid`/`LfuHybrid`, which take no embedded params)
  and a new `PaperCache<K, TieredBuffer, S>` impl block. Method surface matches the other two hybrids
  except `new`/`with_hasher` take an extra `k_in: f64` parameter (`new(max_size, fast_tier_size,
  k_in)`). Also exports `TwoQHybridStats` from the crate root; shares `TieredBuffer`/`Tier`/
  `CacheTierSize` with the other two hybrids rather than duplicating them
- **When disabled**: None of the above types/methods are compiled; `PaperPolicy::TwoQHybrid` doesn't exist
- **Behavior**: Admission: every new object → the FIFO queue, always slow tier — this is a real,
  synchronous PMEM write on every single `set()` call (`TieredBuffer::new_slow` built directly at the
  API layer), not an async/eventual placement like the other two hybrids' fast-first admission.
  Promotion: a re-access to a FIFO-queue object moves it straight to the top of the main queue's fast
  tier; once inside the main queue, a slow-tier access promotes it back to fast the same way
  `lru_hybrid_cache` does, possibly cascading a further demotion. Demotion: the main queue's fast-tier
  LRU tail moves to its slow-tier portion under fast-tier pressure — never touches the FIFO queue.
  Eviction: prefers the FIFO queue's tail (an object that aged out without a second access) before ever
  touching the main queue's slow tail — this single priority rule reconciles both of the paper's stated
  eviction triggers ("a FIFO object ages out" and "capacity is exhausted") into one `evict_one()`
  implementation. No ghost/re-admission memory is kept for objects that age out of the FIFO queue
  (unlike classic 2Q's `A1out` or this crate's own `SThreeFifoStack::ghost`) — an exact-membership check
  on every admission was judged an unwelcome added cost given every admission already pays a synchronous
  PMEM write; a probabilistic structure (e.g. a counting Bloom filter) is the right tool to revisit this
  and is left as future work
- **Requirements**: `["key_value_pmem"]` only, same reasoning as the other two hybrids. **Mutually
  exclusive with `lru_hybrid_cache`/`lfu_hybrid_cache`**
- **Use case**: Same "single unified cache" shape as the other two hybrids, for workloads with a large
  fraction of one-time/scan-like accesses that should be filtered out before ever touching DRAM — the
  literal cost of every write landing in PMEM first is the point, not a side effect
- **A design/correctness note worth knowing**: a `PolicyStack` has no reference to the shared object map
  or `AtomicStatus`, so it can never safely evict an object on its own — only `PolicyWorker::apply_evictions`'s
  `evict_one()` + `erase()` pairing can (this already correctly happens for overall-`max_size` pressure on
  every hybrid stack). `TwoQHybridStack`'s FIFO queue has its *own* independent capacity budget
  (`fifo_capacity = k_in * max_size`) that can be exceeded well before overall `max_size` is — to trigger
  a real eviction for that case too, the `PolicyStack` trait has a `needs_capacity_eviction()` method
  (default `false`, overridden by `TwoQHybridStack` to report `fifo_used > fifo_capacity`) that
  `apply_evictions`'s loop condition also checks, so FIFO-capacity-driven pressure drains through the
  same, correct removal path as global eviction. An earlier draft called a stack-only eviction routine
  directly from `insert()`/`resize()`; that dropped the key from the stack's own bookkeeping without
  ever removing it from the real object map, permanently desyncing the two (`has()` kept returning
  `true` for an object the stack had already "forgotten") — caught by
  `tests/two_q_hybrid_cache_integration.rs`'s FIFO-eviction tests failing outright, not merely flaking

### `two_q_fast_admission_hybrid_cache`
- **Purpose**: `two_q_hybrid_cache` with the one-access FIFO queue relocated to the **fast (DRAM)
  tier**, so `set()` is a plain DRAM write rather than a synchronous PMEM allocation on the
  calling thread. The same trade `s3_fifo_ghost_lazy_demotion_fast_admission_hybrid_cache` makes for
  the s3-fifo family. The logical 2Q structure is unchanged — only the physical placement of the
  one-access queue's bytes differs
- **When enabled**: Adds `PaperPolicy::TwoQFastAdmissionHybrid(f64)` (string form
  `"2q-fast-admission-hybrid-{k_in}"`) and a new `PaperCache<K, TieredBuffer, S>` impl block with the
  same `new(max_size, fast_tier_size, k_in)` signature as `two_q_hybrid_cache`. Exports
  `TwoQFastAdmissionHybridStats`; shares `TieredBuffer`/`Tier`/`CacheTierSize` with every other hybrid
- **When disabled**: None of the above types/methods are compiled
- **Behavior**: Admission: every new object → the one-access FIFO queue, in the **fast** tier
  (`admission_tier` is unconditionally `Tier::Fast`, needing no object-map probe at all — one fewer
  `DashMap` lookup per `set()` than `two_q_hybrid_cache`, on top of the avoided PMEM allocation).
  Promotion: a re-accessed FIFO object moves into the main queue's fast portion — a bookkeeping move
  that emits **no migration**, since the bytes are already in DRAM (so `promotions` counts only
  genuine PMEM→DRAM moves in this design, never FIFO→main). Demotion, eviction priority, and the
  no-ghost-queue decision are all identical to `two_q_hybrid_cache`
- **The one accounting difference that matters**: `fifo_capacity = k_in * max_size` is now a DRAM
  reservation **carved out of `fast_tier_size`**, not an independent PMEM budget:
  `effective_main_fast_capacity = fast_tier_size − k_in * max_size`. Since `k_in` is denominated in
  `max_size` while the budget it consumes is `fast_tier_size` (typically a small fraction of
  `max_size`), a `k_in` that is unremarkable under `two_q_hybrid_cache` can swallow the whole fast
  tier here — at a 24 GB cache with a 4 GB fast tier, `k_in = 0.1` reserves 2.4 GB (60%) for objects
  with no demonstrated reuse. If the reservation meets or exceeds `fast_tier_size`, the main queue
  gets zero fast capacity and every promotion self-demotes immediately: legitimate, but rarely
  intended. **Sweep `k_in` down here in a way that is unnecessary for `two_q_hybrid_cache`.** The
  reservation is the *fixed* `fifo_capacity`, not live `fifo_used`, so the main queue's budget stays
  stable as the FIFO queue fills and drains (and admission therefore never demotes anyone by itself);
  `resize()` re-settles, which `TwoQHybridStack::resize` need not
- **Requirements**: `["key_value_pmem"]`, same as every other hybrid. **Mutually exclusive with every
  other hybrid-cache feature**
- **Use case**: Workloads where SET latency matters and there is DRAM headroom to spend on unproven
  objects — the inverse of `two_q_hybrid_cache`'s tradeoff, which spends SET latency to keep DRAM
  exclusively for proven-hot objects
- **Measured** (800K accesses of `standard_web.bin`, `-c 1`, 2 GB cache / 1 GB fast tier, `k_in`
  0.1, same binary otherwise): SET mean **7.11 µs → 3.30 µs (2.15x)**, SET p99 **25.48 µs → 9.36 µs
  (2.72x)**, miss ratio essentially unchanged (0.3308 → 0.3211) — as expected, since the logical
  queue structure is identical. GET mean also improved (4.09 → 3.34 µs), but that is **specific to
  this configuration**, not a general property: at this scale the whole retained working set fit
  inside the effective fast budget, so nothing was ever demoted (slow tier empty, 0 demotions) while
  `two_q_hybrid_cache` had 200 MB in PMEM by construction. The cost side shows in the same numbers:
  624 MB of DRAM used versus 374 MB for the same workload

### `two_q_fast_admission_reprieve_hybrid_cache`
- **Purpose**: `two_q_fast_admission_hybrid_cache` with one change — a one-access object that ages
  out of the FIFO queue without a second access is **reprieved into the slow tier** (spliced onto
  the bottom of the main queue) rather than evicted outright
- **When enabled**: Adds `PaperPolicy::TwoQFastAdmissionReprieveHybrid(f64)` (string form
  `"2q-fast-admission-reprieve-hybrid-{k_in}"`), a `PaperCache<K, TieredBuffer, S>` impl block with
  the same `new(max_size, fast_tier_size, k_in)` signature, and `TwoQFastAdmissionReprieveHybridStats`
- **Behavior**: Admission, promotion, main-queue demotion and the fast-tier accounting are all
  identical to `two_q_fast_admission_hybrid_cache`. The difference is `settle_fifo_queue`, which runs
  **synchronously from `insert`/`resize`** (never through `evict_one`) and moves the FIFO tail to the
  back of the main queue tagged `Tier::Slow`. `needs_capacity_eviction()` therefore returns to the
  trait default `false`, and `evict_one` becomes purely about the main queue's LRU tail — with a
  last-resort FIFO fallback that exists only so `apply_evictions` is never handed a `None` (which it
  answers by evicting a *random* object)
- **Placement**: the bottom of the main queue, not the top of its slow segment. An object with no
  demonstrated reuse should rank below proven-but-cold ones, and `push_back` is O(1) on the existing
  single list — the s3-fifo equivalent needed to insert at the fast/slow *boundary*, which forced
  that stack's two-physical-lists restructure after an O(n)-per-reprieve first attempt burned ~18
  minutes of worker CPU on a real trace
- **Counter semantics**: a reprieve is counted in `demotions` (it is a real DRAM→PMEM copy), **not**
  `evictions`. So `evictions` here means only "removed from the cache", which is a narrower thing
  than the same field in the non-reprieve variant
- **Requirements**: `["key_value_pmem"]`. **Mutually exclusive with every other hybrid-cache feature**
- **Measured** (800K accesses of `standard_web.bin`, `-c 1`, 2 GB cache / 1 GB fast tier, `k_in` 0.1,
  same binary otherwise, all three 2Q designs run back to back):

  | | `2q-hybrid` | `2q-fast-admission` | `2q-fast-admission-reprieve` |
  |---|---|---|---|
  | miss ratio | 0.3308 | 0.3211 | **0.2673** |
  | SET mean | 7.11 µs | **3.30 µs** | 4.35 µs |
  | SET p99 | 25.48 µs | **9.36 µs** | 23.49 µs |
  | GET mean | 4.09 µs | **3.34 µs** | 4.59 µs |
  | objects retained | 34,666 | 37,668 | **120,416** |
  | evictions | 229,983 | 219,232 | 93,410 |
  | demotions | 0 | 0 | 190,066 |
  | DRAM / PMEM | 374 / 200 MB | 624 / 0 MB | 906 / 1,083 MB |

  **The headline is the miss ratio, and the reason is worth understanding before reading the rest.**
  The non-reprieve variant filled only 598 MB of its 2 GB cache yet still evicted 219,232 objects:
  with nothing ever demoted, objects could only leave via FIFO-capacity eviction, so the cache
  self-limited at roughly `fifo_capacity + main_fast` and threw away objects it had room to keep.
  Reprieving instead routes those objects into the (uncapped) slow tier, so the cache actually uses
  its configured size — 3.2x more objects retained and a 16.8% relative reduction in misses.
  The costs are real and visible in the same row: SET mean is 32% higher and SET p99 2.5x higher
  than the non-reprieve variant despite admission still being a pure DRAM write, because 190,066
  reprieve copies on the `PolicyWorker` thread contend with the API threads for the allocator (the
  same contention documented at length in `CLAUDE.md`); and GET is slower because 65,637 objects now
  live in PMEM rather than none
- **Use case**: when the working set exceeds what the fast tier plus FIFO reservation can hold and
  hit rate matters more than tail SET latency. If the non-reprieve variant is leaving a large part
  of `max_size` unused (compare `used_size` against `max_size` in the summary CSV), that is the
  signal this variant is worth trying

### `lru_sized_hybrid_cache`
- **Purpose**: Single-instance, segmented-LRU hybrid cache with a *size-split* fast AND slow tier — same
  LRU admission/promotion/demotion/eviction semantics as `lru_hybrid_cache`, but the fast (DRAM) tier's
  and the slow (PMEM) tier's bookkeeping are each further split into two independently-tracked segments
  ("small"/"large") by a runtime-configurable byte threshold, so a handful of large objects can't
  dominate/starve many small ones (or vice versa) in either tier's recency order purely because of size
- **When enabled**: Adds `PaperPolicy::LruSizedHybrid` and a new `PaperCache<K, TieredBuffer, S>` impl
  block — `new(max_size, small_fast_tier_size, large_fast_tier_size, size_threshold)`,
  `get`/`set`/`del`/`has`/`peek`/`ttl`/`size`/`wipe`/`resize`/`tier_of` (shared with the other hybrids),
  plus `set_fast_tier_size`/`fast_tier_size` (reused from the shared block to mean the SMALL segment's
  capacity specifically — this design is the only hybrid with a second, independent fast segment with no
  shared-block equivalent), `set_large_fast_tier_size`/`large_fast_tier_size`,
  `set_size_threshold`/`size_threshold`, and `lru_sized_hybrid_stats`. Also exports
  `LruSizedHybridStats` from the crate root; shares `TieredBuffer`/`Tier`/`CacheTierSize` with the other
  four hybrids rather than duplicating them — still exactly one physical DRAM allocator path and one
  physical PMEM allocator path (`Tier` stays 2 variants, `TieredBuffer` is unchanged); the size split is
  purely which of four internal recency lists a key's bookkeeping is tracked in, invisible at the
  `TieredBuffer`/physical-allocation level
- **When disabled**: None of the above types/methods are compiled; `PaperPolicy::LruSizedHybrid` doesn't exist
- **Behavior**: Admission/re-admission-on-overwrite/promotion all route through one rule: classify by the
  object's current size against `size_threshold` (`size < size_threshold` → small, else large) and land
  in the matching FAST segment — mirrors `lru_hybrid_cache`'s existing "any touch always promotes to
  fast" rule, just adding "which of the two fast segments" on top of it, so a reclassifying overwrite
  (a `set()` whose new size crosses the threshold) moves between the two fast segments directly with
  **no migration emitted** (both segments are physically `TieredBuffer::Fast`). Demotion: each fast
  segment's own pressure (its raw byte usage exceeding its own configured capacity, minus a proportional
  share of the shared-metadata DRAM reservation) demotes only *that* segment's LRU tail, into the
  *matching* slow list — segment-local, never crossing. Eviction: prefers whichever of the two slow
  lists is non-empty; if both are non-empty, whichever currently holds more objects (a cheap proxy for
  "probably has the older tail," avoiding real cross-list timestamps). Only if *both* slow lists are
  empty (nothing has ever been demoted) does eviction fall back to whichever fast segment is furthest
  over its own budget by ratio — a direct port of `lru_hybrid_cache`'s own documented last-resort
  fallback for the equivalent single-fast-tier case, not a new behavior invented for this design. Every
  promotion/demotion that does cross the fast/slow boundary is actual data movement, same as
  `lru_hybrid_cache` — a live object's bytes exist in exactly one tier's allocation at a time. TTL
  survives every tier/segment move unmodified for the same reason (`Object::set_data` only ever replaces
  `data`, never `key` or `expiry`)
- **Requirements**: `["key_value_pmem"]` only, same reasoning as the other four hybrids. **Mutually
  exclusive with `lru_hybrid_cache`/`lfu_hybrid_cache`/`two_q_hybrid_cache`/`fifo_hybrid_cache`**
- **Use case**: Same "single unified cache" shape as the other hybrids, for workloads with a real mix of
  small and large object sizes where a single shared recency budget would otherwise let size (rather
  than actual access pattern) dictate which objects survive in DRAM/hot PMEM
- **A design note worth knowing**: only the two fast segments carry independent, configurable capacities
  — the two slow lists carry no capacity of their own and stay governed purely by the overall `max_size`
  terminal-eviction trigger, exactly like `lru_hybrid_cache`'s single slow tier today. The slow-tier
  split is entirely about eviction-order fairness (which recency list a demoted object's eviction
  candidacy is tracked in), not a new capacity dimension — confirmed as the intended scope during design
  (a physically-separate-PMEM-arenas-per-size-class alternative was considered and explicitly rejected,
  given this project's own prior history of multi-arena PMEM/DRAM allocator experiments proving costly —
  see `CLAUDE.md`)

## Implementation Details

### Type System

The implementation uses Rust's conditional compilation to select the appropriate hashtable types:

**Tiering Manager's hashtable** (`dram_cache` in `TieringManager`):
- Without `tiering_hashtable_pmem`: `DashMap` or `HashMap` (DRAM)
- With `tiering_hashtable_pmem`: `HashMap<..., Hybrid>` (PMEM)

**Global hashtable** (`objects` in `PaperCache`):
- Default: `DashMap` (DRAM)
- With `global_hashtable_pmem`: `RwLock<HashMap<..., Hybrid>>` (PMEM)
- With `hashbrown_dram`: `RwLock<HashMap<..., NoHasher>>` (DRAM)

### Allocator Integration

The `Hybrid` allocator is used to place data in PMEM:
- Defined in `src/allocator.rs`
- Implements both `GlobalAlloc` and `Allocator` traits
- Resolves to `numa_alloc::SlowObjects`, which routes allocations through node-1-bound jemalloc arenas

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

8. **PMEM-backed eviction stacks + key/value in PMEM**:
   ```toml
   features = ["eviction_stacks_pmem", "key_value_pmem"]
   ```

9. **Full PMEM stack** (tiering + eviction stacks):
    ```toml
    features = ["enable_tiering_manager", "key_value_pmem", "eviction_stacks_pmem"]
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
7. **`hashbrown_dram`**: Use hashbrown HashMap in DRAM for direct performance comparison with `global_hashtable_pmem`
8. **`eviction_stacks_pmem`**: LFU eviction stacks allocated in PMEM for co-location with PMEM objects

## Code Locations

- **Feature definitions**: `Cargo.toml`
- **Allocator**: `src/allocator.rs`
- **Tiering manager hashtable**: `src/tiering/manager.rs`
- **Global hashtable**: `src/lib.rs`
- **Worker manager integration**: `src/worker/manager.rs`
- **PMEM eviction collections**: `src/worker/policy/policy_stack/pmem_collections.rs`
- **LFU policy stack**: `src/worker/policy/policy_stack/lfu_stack.rs`
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
  duplicating them (all three hybrid-cache features are mutually exclusive).
- **two_q_hybrid_cache (single-instance, 2Q-segmented hybrid cache)**: `src/two_q_hybrid_cache/`
  (`stats.rs` for `TwoQHybridStats`), `src/worker/policy/policy_stack/two_q_hybrid_stack.rs`
  (`TwoQHybridStack`), `src/policy.rs` (`PaperPolicy::TwoQHybrid(f64)`), `src/status.rs`
  (`two_q_hybrid_*` counters/gauges), `PaperCache<K, TieredBuffer, S>` impl block in `src/lib.rs`.
  Also the source of the `PolicyStack::needs_capacity_eviction` trait method (default `false`) and
  `PolicyWorker::apply_evictions`'s loop-condition change in `src/worker/policy/mod.rs` — both are
  generic additions the other two hybrids don't need. Reuses `src/tiered_buffer.rs`/`src/size.rs`
  rather than duplicating them.

## Testing

To test different combinations (requires nightly Rust for allocator features):

```bash
# Check hashbrown HashMap in DRAM (for performance comparison)
cargo +nightly check --no-default-features --features hashbrown_dram

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

# Verify eviction_stacks_pmem with key_value_pmem
cargo +nightly check --features "eviction_stacks_pmem,key_value_pmem"

# Verify full feature combination
cargo +nightly check --features "enable_tiering_manager,eviction_stacks_pmem,key_value_pmem"

# Check lru_hybrid_cache (single-instance hybrid cache)
cargo +nightly check --features lru_hybrid_cache

# Run lru_hybrid_cache's unit + inline tests
cargo +nightly test --lib --features lru_hybrid_cache

# Run lru_hybrid_cache's PMEM integration tests (requires a second NUMA node)
cargo +nightly test --test lru_hybrid_cache_integration --features lru_hybrid_cache

# Check lfu_hybrid_cache (single-instance, frequency-segmented hybrid cache)
cargo +nightly check --features lfu_hybrid_cache

# Run lfu_hybrid_cache's unit + inline tests
cargo +nightly test --lib --features lfu_hybrid_cache

# Run lfu_hybrid_cache's PMEM integration tests (requires a second NUMA node)
cargo +nightly test --test lfu_hybrid_cache_integration --features lfu_hybrid_cache

# Check two_q_hybrid_cache (single-instance, 2Q-segmented hybrid cache)
cargo +nightly check --features two_q_hybrid_cache

# Run two_q_hybrid_cache's unit + inline tests
cargo +nightly test --lib --features two_q_hybrid_cache

# Run two_q_hybrid_cache's PMEM integration tests (requires a second NUMA node)
cargo +nightly test --test two_q_hybrid_cache_integration --features two_q_hybrid_cache

# Confirm every pairwise combination of the three hybrid-cache features is
# mutually exclusive (each expected to fail to compile)
cargo +nightly check --features lru_hybrid_cache,lfu_hybrid_cache
cargo +nightly check --features lru_hybrid_cache,two_q_hybrid_cache
cargo +nightly check --features lfu_hybrid_cache,two_q_hybrid_cache
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
✅ `eviction_stacks_pmem` correctly allocates LFU/LRU stacks via `Hybrid` (node-1-bound jemalloc arenas)
✅ `lru_hybrid_cache` implements a single unified `PaperCache<K, TieredBuffer>` with a segmented-LRU
  fast/slow boundary; promotion/demotion verified as real data movement (never present in both tiers)
  end to end on real PMEM hardware (`tests/lru_hybrid_cache_integration.rs`, 14/14 passing)
✅ `lfu_hybrid_cache` implements the same single-unified-instance architecture with a segmented-LFU
  (frequency-ordered, not recency-ordered) fast/slow boundary; promotion/demotion verified as real data
  movement end to end on real PMEM hardware (`tests/lfu_hybrid_cache_integration.rs`, 17/17 passing).
  Mutually exclusive with `lru_hybrid_cache` at compile time (verified via `compile_error!`)
✅ `two_q_hybrid_cache` implements the same single-unified-instance architecture with a 2Q-segmented
  boundary — admission always to a slow-tier FIFO queue, promotion to the main queue's fast tier only on
  re-access; terminal eviction correctly prioritizes the FIFO queue over the main queue, and FIFO-capacity
  pressure (`k_in`) correctly triggers real evictions through the same `evict_one()`/`erase()` path as
  global `max_size` pressure (`PolicyStack::needs_capacity_eviction`) — verified end to end on real PMEM
  hardware (`tests/two_q_hybrid_cache_integration.rs`, 18/18 passing, run twice to confirm not flaky).
  Mutually exclusive with `lru_hybrid_cache`/`lfu_hybrid_cache` at compile time (verified via
  `compile_error!`)
