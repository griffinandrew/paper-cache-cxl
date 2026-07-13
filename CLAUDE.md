# CLAUDE.md

Guidance for Claude Code (and other agents) working in this repository.

## What this crate is

`paper-cache` (crate name in `Cargo.toml`) is an in-memory Rust cache library ("PaperCache") that
supports runtime-switchable eviction policies and, on this branch, DRAM/PMEM (CXL) memory
tiering. It is consumed by a separate `paper-server` crate (not in this repo) and should not
normally be used directly by application code.

The defining theme of this fork/branch is **experimenting with where cache data structures and
object bytes physically live** — DRAM vs. persistent/CXL memory (PMEM) — via a large matrix of
Cargo feature flags, plus several higher-level "hybrid cache" designs that explicitly manage a
fast (DRAM) tier and a slow (PMEM) tier as two cooperating caches.

Read `FEATURE_FLAGS.md` and `README.md` before making changes; they document the feature-flag
matrix and its history. This file focuses on code structure and the in-flight `lru_hybrid_cache`
feature.

## Layout

```
src/
  lib.rs                      PaperCache<K, V, S> — the core cache type. Repeated per storage
                               combination (BufferDRAM vs BufferPMEM, with/without eviction
                               callback, etc.) behind #[cfg(feature = ...)] impl blocks. This file
                               is large (~5500 lines) because most of the "same API, different
                               backing storage" variants are written out longhand rather than
                               generically — grep for `impl<K, S> PaperCache<K, Buffer` to find
                               each variant.
  policy.rs                   PaperPolicy enum (Lfu, Fifo, Clock, Sieve, Lru, Mru, TwoQ, Arc,
                               SThreeFifo) + its Display/FromStr for the string config format
                               used by paper-server (e.g. "2q-0.2-0.2", "s3-fifo-0.1").
  error.rs                    CacheError.
  status.rs                   AtomicStatus — size accounting, hit/miss/set/del counters, current
                               policy, max size. Shared, lock-light state read/written by both the
                               PaperCache API and the background workers.
  object/                     Object<K, V> (the stored entry) + overhead accounting
                               (object/overhead.rs computes per-object bookkeeping size).
  allocator.rs                Custom allocators: HybridObjects / RegionHybrid / DevDaxBump /
                               DRAMObjects — the concrete types the crate-wide `Hybrid` type alias
                               resolves to, selected by feature flags in lib.rs.
  umf_bindings.rs,
  umf_allocator_bindings.rs    FFI bindings to UMF (Unified Memory Framework), used for real PMEM
                               allocation. `build.rs` generates a stub (malloc/free-backed) when
                               UMF isn't available so the crate still builds on non-PMEM machines.

  worker/                     Background-thread machinery. PaperCache::new() spawns a
                               WorkerManager thread that owns all mutation of eviction state so
                               the hot get()/set() path stays lock-cheap.
    manager.rs                 WorkerManager — fans WorkerEvent out to sub-workers
                                (PolicyWorker, TtlWorker, TieringWorker).
    mod.rs                      WorkerEvent enum: Get(key, hit), Promote(key), Set(key, size,
                                 expiry, old_info), Del(key, expiry), Ttl, Wipe, Resize, Policy.
                                 This is the *only* channel between PaperCache's API calls and
                                 all background eviction/tiering logic.
    policy/
      mod.rs                    PolicyWorker — drives the active PolicyStack, runs evictions
                                 when status.used_size() exceeds max_size, and (under the
                                 `hybridcache` feature) can carry an eviction_callback that fires
                                 per evicted object, plus a promotion_tx sender used when a
                                 policy stack reports AccessOutcome::GhostHit.
      policy_stack/              One file per eviction policy, all implementing the PolicyStack
                                 trait (insert/update/remove/evict_one/resize/clear + record_access
                                 returning AccessOutcome::{None, GhostHit}). init_policy_stack()
                                 in mod.rs maps PaperPolicy -> Box<dyn PolicyStack>.
                                   lru_stack.rs      plain LRU (HashList, or PmemHashList under
                                                     `eviction_stacks_pmem`) — the policy this new
                                                     feature's two tiers will each use internally.
                                   two_q_stack.rs,
                                   s_three_fifo_stack.rs   examples of policies with a *ghost*
                                                     queue; only these currently emit GhostHit.
      mini_stack/                Lightweight per-policy stacks used for PaperCache's "auto" mode
                                 (estimating what other policies would have done, to support
                                 switching policy live).
      trace/                    Optional access-trace recording/replay used by mini-stacks.
    ttl/                        TtlWorker — background expiry sweep.
    tiering.rs                  TieringWorker — bridges WorkerEvent to TieringManager
                                 (see below); only compiled under the tiering-manager features.

  tiering/                     The **existing** DRAM/PMEM tiering mechanism (`enable_tiering_manager`
                                / `tiering` / `multitiering` / `sets_dram` features). This is a
                                *hotness-threshold, copy-based* design: PMEM is always the source
                                of truth, and objects accessed >= a configurable threshold get a
                                physical copy placed in a separate DRAM side-cache
                                (manager.rs: promote_to_dram_with_object / demote_from_dram).
                                Under `hashtable_tiering` it adds a third, zero-copy "warm" state
                                (object/TieringObject holds a CXL reference instead of a physical
                                copy). NOTE: this module intentionally keeps copies in both tiers
                                simultaneously — it is the opposite data-movement model from the
                                new lru_hybrid_cache feature described below. Don't reuse its
                                copy-on-promote pattern for the new feature.

  hybridcache/                 Higher-level two-tier caches built by composing *two independent
                                PaperCache instances* (one BufferDRAM, one BufferPMEM) rather than
                                by adding tiering logic inside a single PaperCache. This is the
                                module the new lru_hybrid_cache feature belongs in.
    mod.rs                      S3FifoHybridCache<K> (feature = "hybridcache"): small DRAM tier
                                 runs S3-FIFO, far PMEM tier runs LRU. Admission always goes to
                                 the small tier. Demotion: PolicyWorker's eviction_callback (see
                                 worker/policy/mod.rs) fires on small-tier eviction and
                                 asynchronously writes the bytes to the far tier over a bounded
                                 channel + dedicated worker thread. Promotion: driven by the small
                                 tier's S3-FIFO *ghost queue* — a ghost hit sends
                                 WorkerEvent::Promote, a background worker looks the key up in a
                                 demoted_lookup: DashMap<HashedKey, Arc<K>> and re-inserts far-tier
                                 bytes into the small tier. Uses copy-on-read: the far-tier (PMEM)
                                 copy is *never deleted* on promotion, so a key can legitimately
                                 exist in both tiers at once. An in_flight_demotions: DashSet<K>
                                 plus yield-and-retry in get() covers the DRAM->PMEM migration
                                 window.

tests/
  hybridcache_integration.rs   Integration tests for S3FifoHybridCache; run with
                               `cargo +nightly test --test hybridcache_integration --features hybridcache`.
                               Good template for the new feature's test file — mirrors tier
                               routing, eviction propagation, promotion, stats counters, and
                               in-flight-migration edge cases.
  tiering_integration.rs        Integration tests for the copy-based tiering manager.
  global_flatmap_integration.rs,
  pmem_region_alloc_integration.rs   Tests for other storage-backend feature combinations.
```

## Feature-flag model (read `FEATURE_FLAGS.md` for full detail)

Nearly everything in this crate is gated by Cargo features because the whole point of the branch
is comparing storage-placement strategies. Key points relevant to new work:

- `all_dram` — force every allocation to DRAM.
- `key_value_pmem` / `key_pmem_value_pmem` — put value (or key+value) bytes in PMEM via the
  `Hybrid` allocator alias (resolves to `HybridObjects`, `RegionHybrid`, or `DevDaxBump` depending
  on which of `pmem_region_alloc` / `region_hybrid_allocator` / `devdax_bump` is also set).
- `enable_tiering_manager`, `tiering`, `multitiering`, `sets_dram`, `hashtable_tiering` — the
  existing copy-based tiering manager (`src/tiering/`), not the same as `hybridcache`.
- `hybridcache = ["all_dram", "key_pmem_value_pmem"]` — the S3-FIFO/LRU two-tier cache. This is
  the closest precedent for the new feature: it needs a DRAM-typed tier (`BufferDRAM`, via
  `all_dram`) and a PMEM-typed tier (`BufferPMEM`, via `key_pmem_value_pmem`) simultaneously,
  which is exactly what `lru_hybrid_cache` will also need.
- Combining features generally means AND-ing requirements; several `compile_error!` guards in
  `lib.rs` reject known-invalid combinations (e.g. `global_flatmap_dram` + `global_flatmap_pmem`).

`BufferDRAM = Box<[u8]>` and `BufferPMEM = Box<[u8], Hybrid>` (see `lib.rs`) are the two value
types the hybrid caches wrap `PaperCache<K, _>` around.

## Feature: `lru_hybrid_cache` (steps 1–10 implemented; see status below)

### Source (paper description this implements)

> The LRU eviction queue is segmented across two tiers. New objects are admitted into the fast
> tier at the top of the queue. As objects age without being accessed they drift down the queue
> and are eventually demoted to the slow tier. Accessing a slow-tier object promotes it back to
> the top of the fast tier. When cache capacity is exhausted, the least recently accessed object
> is evicted from the slow tier.
>
> - **Admission**: every new object → top of the fast tier.
> - **Demotion**: LRU-tail fast-tier object → slow tier, whenever fast-tier space is needed.
> - **Promotion**: slow-tier object accessed → moved to top of fast tier.
> - **Eviction**: LRU-tail slow-tier object removed when slow-tier capacity is exhausted.

**Requirements from the feature owner (confirmed):**

1. **One unified cache instance**, not two `PaperCache`s stitched together with channels/DashSets
   like `S3FifoHybridCache`. There is a single logical LRU queue, a single object table, a single
   background worker — "fast" and "slow" are a property of *where an object's bytes currently
   live*, not two separate caches.
2. **Actual data movement.** A live object's bytes exist in exactly one tier's allocation at a
   time — never copied into both, unlike `S3FifoHybridCache` (copy-on-read) and the `tiering/`
   module (`promote_to_dram_with_object` keeps PMEM as a permanent second copy).
3. **TTL survives a tier move.**
4. **Tier size is configurable, in MB by default but flexible** (reuse the existing
   `CacheTierSize` bytes/Mb/Gb enum rather than inventing a new unit type).
5. **Terminal (slow-tier) evictions are counted in stats.**

This is a meaningfully different shape than `S3FifoHybridCache`/the tiering manager, both of
which get their "two tiers" by literally having two separate `PaperCache` instances or two
separate storage maps. Here there is **one** `PaperCache<K, V>`, one `objects` map, one
`AtomicStatus`, one `PolicyWorker` — the tier a given object is in is encoded per-object inside
the single value type stored in that one map.

### Design

**New value type — a tagged union, not two caches:**

```rust
pub enum TieredBuffer {
    Fast(Box<[u8]>),           // DRAM — plain Box, goes through the crate's global allocator
    Slow(Box<[u8], Hybrid>),   // PMEM — Hybrid-allocated, same as BufferPMEM today
}
```

implementing `AsRef<[u8]>` (match both arms), `TypeSize` (`self.as_ref().len()`, matching how
`BufferPMEM`'s `TypeSize` impl already just does `self.len()` in `lib.rs`), and `Clone`. This
becomes the `V` in a single new `impl<K, S> PaperCache<K, TieredBuffer, S> { ... }` block in
`lib.rs`, written by adapting (not reusing) the existing `BufferDRAM`/`BufferPMEM` blocks —
`new`, `get`, `set`, `del`, `has`, `peek`, `ttl`, `size`, `wipe`, `resize`, `policy` all follow the
same shape as those blocks; none of that is new logic, it's the mechanical bulk of the change.

Feature dependency is lighter than `hybridcache`'s: **`lru_hybrid_cache = ["key_value_pmem"]`**,
not `all_dram` + `key_pmem_value_pmem`. Reasoning: a plain `Box<[u8]>` already allocates through
the crate's `#[global_allocator]` (`DRAMObjects`, set unconditionally near the top of `lib.rs`)
regardless of feature flags, so the `Fast` arm needs nothing special. `key_pmem_value_pmem` would
additionally force the *key* into PMEM crate-wide (`Object`'s `_key_pmem` branch) — this feature
only needs to migrate *value* bytes between tiers, so plain `key_value_pmem` (which makes
`BufferPMEM`/`Hybrid` available without moving keys) is the correct, smaller dependency; keys stay
DRAM-resident regardless of which tier an object's value is in.

**TTL survives automatically, "for free," precisely because this is one unified instance:**
`Object<K, V>` already stores `key`, `data: Arc<V>`, `expiry` together. A tier migration only
needs to replace `data` in place — add one small method to `src/object/mod.rs`:

```rust
pub fn set_data(&mut self, data: V) {
    self.data = Arc::new(data);
}
```

and call it via `self.objects.get_mut(&hashed_key)` inside the migration logic. `key` and
`expiry` are untouched, so TTL (and the key) survive by construction — no manual re-application
needed (this is actually *simpler* than it would have been in the two-instance design, where TTL
had to be read on one instance and re-applied on the other).

**No `AtomicStatus`/size-accounting delta is needed on migration.** Checked
`object/overhead.rs::OverheadManager::base_size`/`total_size`: both depend only on the *logical*
byte length (`V::get_size()`) and the active `PaperPolicy`'s fixed per-object overhead — neither
depends on which allocator backs the bytes. So swapping `Fast(..)` ↔ `Slow(..)` never changes an
object's accounted size; `status.used_size()` needs no adjustment when a migration happens, only
when an object is actually admitted/deleted/evicted (unchanged from today).

**Concurrency is simpler than the two-instance design.** All tier migrations happen inside the
single `PolicyWorker` background thread (see below) — there's no second thread doing PMEM writes
that the public API thread has to coordinate with via an `in_flight_demotions`/`in_flight_promotions`
`DashSet` like `S3FifoHybridCache` needs. A reader calling `get()` concurrently with a migration
just sees the object map's normal per-shard locking (`DashMap::get`/`get_mut`) hand it either the
pre- or post-migration `Arc<TieredBuffer>` — never a torn or duplicated state. This removes an
entire category of bookkeeping the wrapper design needed.

**New policy: `PaperPolicy::LruHybrid`** (`policy.rs`), string form `"lru-hybrid"`. Deliberately
carries *no* embedded fast-tier-size parameter (unlike `TwoQ(f64, f64)`/`SThreeFifo(f64)`) because
the size must be runtime-configurable (requirement 4), not fixed at policy-string-parse time.

**New policy stack: `worker/policy/policy_stack/lru_hybrid_stack.rs` — `LruHybridStack`.** Same
recency-ordered `HashList<HashedKey>` as `LruStack` (reuse that structure directly), plus:
  - `fast_capacity: CacheSize` (bytes; settable at runtime — see below) and `fast_used: CacheSize`.
  - a `HashMap<HashedKey, Tier>` recording which tier each currently-tracked key is logically in
    (needed because the fast/slow boundary is a *byte* threshold, not a fixed slot count, so it
    can't be derived purely from list position without a scan).
  - On `insert`/`update` (admission or access): move the key to the front of the list as `LruStack`
    already does, mark it `Tier::Fast`, add its size to `fast_used`, then — only once `fast_used`
    has actually exceeded `fast_capacity` (the trigger; no early demotion) — walk backward from the
    tier boundary evicting-into-slow (flipping `Tier::Fast` → `Tier::Slow` in the map, subtracting
    from `fast_used`) until `fast_used <= fast_low_water` (a fixed 90%-of-`fast_capacity` floor,
    `FAST_TIER_LOW_WATER_RATIO`), not merely `<= fast_capacity`. This headroom is what stops nearly
    every subsequent `set()` from re-triggering a demotion once the fast tier is near capacity —
    added after the initial implementation, once the user asked whether the fast tier should keep
    headroom. Unlike `tiering::manager`'s `TieringConfig::{high,low}_water_mark` (checked only
    periodically, so its *ceiling* also needs headroom to avoid overshoot between polls), this
    stack's drain runs synchronously on every insert/update, so only the drain *target* needed a
    lower floor — the trigger stays the user-configured `fast_capacity` exactly. Collect every key
    that flipped tier this call.
  - `evict_one()` pops the absolute LRU tail of the list — by construction this is always a
    `Tier::Slow` key once any demotion has ever happened (matches "Eviction policy: evict from the
    slow tier"); remove it from the tier map too.
  - New trait surface needed on `PolicyStack` (`worker/policy/policy_stack/mod.rs`), added as
    default-no-op methods so no other policy stack needs changes:
    ```rust
    fn resize_fast_tier(&mut self, _size: CacheSize) {}
    fn drain_tier_migrations(&mut self) -> Vec<(HashedKey, Tier)> { Vec::new() }
    ```
    (`Tier` = a small `enum { Fast, Slow }`, indicating the *direction just moved*.)

**`PolicyWorker` changes (`worker/policy/mod.rs`), gated `#[cfg(feature = "lru_hybrid_cache")]`:**
after `handle_get`/`handle_set` call into the stack as today, additionally call
`stack.drain_tier_migrations()` and, for each `(key, new_tier)` pair, look the object up in
`self.objects` (already held by `PolicyWorker`) and physically reallocate its bytes into the
target tier's `TieredBuffer` variant via the new `Object::set_data`, incrementing
`LruHybridStats::promotions` or `::demotions` accordingly. `apply_evictions()`'s existing loop
(`while status.used_size() > max_size { stack.evict_one() ... }`) needs no change — it already
calls the callback/erase path generically; just record `LruHybridStats::evictions` there too
(gated the same way `hybridcache`'s eviction callback is optionally wired in today).

**Runtime-configurable fast-tier size** (requirement 4), mirroring the existing
`set_dram_threshold`/`set_hotness_threshold` precedent in `tiering/manager.rs` + `lib.rs`:
  - `WorkerEvent::ResizeFastTier(CacheSize)` (new variant in `worker/mod.rs`).
  - `PaperCache::set_fast_tier_size(&self, size: CacheTierSize) -> Result<(), CacheError>` /
    `fast_tier_size(&self) -> CacheSize`, broadcasting/reading through the same channel `resize()`
    already uses. Validate `0 < size <= max_size` — add `CacheError::InvalidFastTierSize` to
    `error.rs`.
  - `CacheTierSize` (bytes/Mb/Gb) currently lives in `src/hybridcache/mod.rs` gated by
    `feature = "hybridcache"`. Move it to a small shared module (e.g. `src/size.rs`) gated
    `any(feature = "hybridcache", feature = "lru_hybrid_cache")` and re-export from both, so this
    feature gets MB-configurable sizing without depending on `hybridcache` or duplicating the type.

**Stats struct**, mirroring `TieringStats`/`HybridCacheStats`:
```rust
pub struct LruHybridStats {
    pub promotions: u64,
    pub demotions: u64,
    pub evictions: u64,      // requirement 5 — terminal removals from the slow tier
    pub fast_bytes_used: u64,
    pub slow_bytes_used: u64,
    pub fast_objects: u64,
    pub slow_objects: u64,
}
```
exposed via `PaperCache::lru_hybrid_stats()`, same as `tiering_stats()` today. **Design correction
made during implementation:** the counters do *not* live in a separate `AtomicLruHybridStats`
struct owned by `PolicyWorker`. `PaperCache`'s struct definition (`lib.rs`) is shared across every
value type (`BufferDRAM`, `BufferPMEM`, `TieredBuffer`, ...) and its literal is duplicated across
~10 constructors throughout the file (same as the existing `tiering_manager` field) — adding a new
field there would force every other constructor to change too, just to plumb a value only
`TieredBuffer` uses. Instead, the seven counters/gauges live directly on `AtomicStatus`
(`status.rs`), which is *already* the one shared `Arc` both `PaperCache` and `PolicyWorker` hold
(`status: StatusRef`) — `AtomicStatus::new()` is the single construction site, so no other
constructor needs to change. `PolicyWorker::new_with_tier_migration` therefore takes no separate
stats parameter; it calls `self.status.record_lru_hybrid_promotion()` / `::demotion()` /
`::eviction()` and `self.status.set_lru_hybrid_gauges(...)` directly.

### Implementation status: complete (steps 1–13)

All 13 steps are implemented and verified, including full PMEM tier-crossing coverage on real
hardware (this sandbox does have a working `Hybrid`/UMF allocator + a memory-only NUMA node — see
"Allocator bug this uncovered" below). Step 11 (Cargo feature) landed early, as part of step 4, to
make the crate compile-checkable along the way — see note there.

1. ✅ `src/error.rs`: `CacheError::InvalidFastTierSize`.
2. ✅ `CacheTierSize` moved to `src/size.rs` (shared, gated `any(hybridcache, lru_hybrid_cache)`),
   re-exported from both `hybridcache` and the crate root for source compatibility.
3. ✅ `src/object/mod.rs`: `Object::set_data`.
4. ✅ `src/lru_hybrid_cache/buffer.rs`: `TieredBuffer` + `AsRef<[u8]>`/`TypeSize`/`Clone`, plus
   `new_fast`/`new_slow`/`is_fast`/`is_slow` helpers. (Module is `src/lru_hybrid_cache/`, not
   `src/lru_hybrid/` as originally sketched.) `Cargo.toml`:
   `lru_hybrid_cache = ["key_value_pmem"]` landed here (step 11), earlier than planned, purely so
   the new module could be compile-checked incrementally.
5. ✅ `src/policy.rs`: `PaperPolicy::LruHybrid` + `Display`/`FromStr` ("lru-hybrid"). Adding the
   variant forced (unavoidably) two other exhaustive-match sites to be updated in the same step:
   `object/overhead.rs::get_policy_overhead` (new per-object overhead estimate) and
   `worker/policy/policy_stack/mod.rs::init_policy_stack` (registered — see step 8).
6. ✅ `src/worker/mod.rs`: `WorkerEvent::ResizeFastTier(CacheSize)`.
7. ✅ `src/worker/policy/policy_stack/mod.rs`: `Tier` enum + default-no-op `resize_fast_tier` /
   `drain_tier_migrations` trait methods, *plus* four more default-`0` gauge methods
   (`fast_bytes_used`, `slow_bytes_used`, `fast_object_count`, `slow_object_count`) needed so
   `PolicyWorker` can read a stack's current tier gauges without downcasting the `Box<dyn
   PolicyStack>` trait object.
8. ✅ `src/worker/policy/policy_stack/lru_hybrid_stack.rs`: `LruHybridStack`. One correction vs.
   the original sketch: it does **not** reuse `PmemHashList` under `eviction_stacks_pmem` the way
   `LruStack` does — `PmemHashList` doesn't expose `before`/`back`/`move_front`, which the
   fast/slow boundary tracking needs, so this stack always keeps its recency list in DRAM
   (`kwik::collections::HashList`) regardless of that feature. `lru_hybrid_cache` doesn't depend
   on `eviction_stacks_pmem` anyway, so this only matters if both are enabled together.
9. ✅ `src/worker/policy/mod.rs` + `src/worker/manager.rs`: `PolicyWorker::new_with_tier_migration`
   / `WorkerManager::new_with_tier_migration`, `apply_tier_migrations` (drains the stack, calls
   `Object::set_data`, records stats — see the `AtomicStatus` correction above),
   `handle_resize_fast_tier`. One privacy wrinkle: `worker::manager` is a *sibling* of
   `worker::policy`, not a descendant, so it can't see the private `policy_stack` submodule
   directly — fixed with `pub use policy_stack::Tier;` inside `worker/policy/mod.rs` (fully `pub`,
   not `pub(crate)`, since `Tier` also needs to flow out to `PaperCache::tier_of`'s public return
   type — see step 10).
10. ✅ `src/lib.rs`: new `impl<K, S> PaperCache<K, TieredBuffer, S>` block (`new`, `with_hasher`,
    `get`, `set`, `del`, `has`, `peek`, `ttl`, `size`, `wipe`, `resize`, `set_fast_tier_size`,
    `fast_tier_size`, `lru_hybrid_stats`, `tier_of`); no `policy()` method (there's only one
    policy, unlike the multi-policy caches this was adapted from). Re-exports: `TieredBuffer`,
    `LruHybridStats`, `Tier` (the latter re-exported via `worker::Tier` → `crate::Tier`, flattening
    through the private `worker`/`worker::policy` module chain).

11. ✅ `Cargo.toml`: `lru_hybrid_cache = ["key_value_pmem"]` (landed at step 4, see above).
12. ✅ New `tests/lru_hybrid_cache_integration.rs` (14 tests, modeled on
    `tests/hybridcache_integration.rs`), exercising the real `TieredBuffer` PMEM path end to end:
    admission always fast; fast-tier pressure demotes with real data movement (`tier_of` confirms
    gone from fast); a slow-tier hit promotes (`tier_of` confirms gone from slow); a cascading
    demotion on promotion; TTL survives both a demotion and a promotion; terminal eviction only
    ever removes the slow-tier tail and is counted; `set_fast_tier_size` takes effect at runtime;
    zero/invalid/tiny fast-tier-size edge cases; `del`/`wipe` across tiers.
13. ✅ Confirmed passing twice in a row (not flaky) with:
    `cargo +nightly test --test lru_hybrid_cache_integration --features lru_hybrid_cache`.

### A real bug this caught: migrations must be byte-length-preserving

While testing step 9's `PolicyWorker` wiring with a throwaway `migrate` closure that tagged buffers
by *appending* a marker byte, `apply_evictions` hung forever. Root cause: `status.base_used_size`
is computed once at insert time via `overhead_manager.base_size(&object)` and subtracted again at
erase time via the *same* recomputation — appending a byte made the erase-time size larger than
the insert-time size, so the unsigned `base_used_size` counter underflowed/wrapped to near
`u64::MAX`, and `apply_evictions`'s `while status.used_size() > max_size` loop then never
terminated (busy-looping on an empty object map). This confirms the invariant already assumed
above ("no `AtomicStatus` delta is needed on migration") is load-bearing, not just an optimization:
**a `migrate` closure must never change a value's byte length.** The real `TieredBuffer::new_fast`/
`new_slow` already satisfy this (straight byte copies); the test closure was fixed to overwrite a
byte in place instead of appending one.

### Allocator bug this uncovered (fixed, outside `lru_hybrid_cache` proper)

Getting step 12 running on real PMEM surfaced a genuine, pre-existing bug in `src/allocator.rs`
that also silently affected `hybridcache`'s PMEM tier: `HybridObjects` (PMEM, NUMA node 1) and
`DRAMObjects` (the crate's `#[global_allocator]`, NUMA node 0) shared a single `static INIT: Once`.
Since `DRAMObjects::alloc` fires on the very first heap allocation of the *entire process*, it
always won the race to consume that `Once`, so `HybridObjects`'s own `init_and_prewarm` (which
creates the UMF pool for NUMA node 1) **never ran** — every `umf_alloc(1, ...)` call returned NULL
forever, aborting with "memory allocation of N bytes failed". `RegionHybrid`/`DevDaxBump` already
correctly use their own dedicated `Once`s (`REGION_INIT`/`DEVDAX_INIT`); fixed `HybridObjects` /
`DRAMObjects` the same way (`HybridObjects` keeps `INIT`, added a separate `DRAM_INIT` for
`DRAMObjects`). This sandbox does have a real UMF library and a memory-only NUMA node 1 (124GB,
0 CPUs — `numactl --hardware`), so with this fix `TieredBuffer::new_slow` (and `hybridcache`'s
`BufferPMEM`) now genuinely allocates through it — confirmed by re-running
`tests/hybridcache_integration.rs`, which went from "aborts immediately on every PMEM test" to
35/37 passing (the pool's one-time init/prewarm takes ~45s on first touch — see below).

Separately (and out of scope for this feature): 2 of `hybridcache_integration.rs`'s 37 tests now
*run* but fail on timing (`test_demotion_counter`, `test_pmem_items_after_eviction` — both expect
an async demotion count > 0 shortly after an overfill and get 0). Not investigated further since
they're pre-existing `hybridcache` tests, not part of this feature; worth a look if anyone touches
that demotion channel's timing assumptions.

### A design trap this surfaced: `wait_until`-style tests need to account for one-time PMEM warm-up

The real UMF pool's first-ever init + prewarm (touching ~4.7M pages on the memory-only NUMA node)
takes on the order of **45 seconds** in this sandbox, paid once per test *process* (gated by the
same `Once` above). A test with its own tight wall-clock assertion — e.g. checking TTL expiry with
a 1-second TTL — can lose a race against that one-time cost if its thread happens to be the one
that triggers it. `tests/lru_hybrid_cache_integration.rs` handles this with an
`ensure_pmem_allocator_warm()` helper called at the top of every PMEM-touching test: it forces a
real demotion (with a 90s budget) before the test's own timing-sensitive logic starts. Because
it's backed by the same process-wide `Once`, only the very first call anywhere in the binary
actually waits; every other call returns almost immediately. Separately, the TTL tests originally
used a fast-tier capacity sized only for `None`-ttl objects; `overhead_manager.base_size` adds a
fixed TTL bookkeeping cost (`get_ttl_overhead()`) that made a *single* ttl'd object alone exceed
that capacity, causing it to self-demote immediately after every promotion (before ever being
observed as `Fast`) and then expire mid-test. Fixed by sizing the TTL tests' fast tier comfortably
larger than one ttl'd object and using several small filler keys to create demotion pressure
instead of a second same-sized key.

### Remaining work

- Byte-budgeted (not slot-counted) fast/slow boundary means a single admission/promotion can, in
  principle, push more than one key past the boundary in one step if object sizes vary a lot. The
  implementation already returns a `Vec` of migrations per call rather than assuming exactly one,
  so this is handled — flagging only because it's worth a dedicated test case (mixed small/huge
  object sizes) rather than assuming the common "one in, one out" case is the only path.
- The 2 pre-existing `hybridcache_integration.rs` timing failures noted above.

## Feature: `lfu_hybrid_cache` (implemented — mirrors `lru_hybrid_cache`)

### Source (paper description this implements)

> Functionally, the LFU frequency ordering is segmented across the two tiers... Since the LFU
> eviction policy assumes the most frequently accessed objects are the most likely to be
> reaccessed again, it makes sense to place these objects in the fast tier. However, while the
> fast tier has not yet reached its capacity, new objects are admitted into the fast tier. Once
> fast tier capacity is reached, every new object is admitted into the slow tier, as it is, by
> definition, the least frequently accessed object. At some point, an object in the slow tier
> will have an access count higher than the object with the lowest frequency count in the fast
> tier. When that happens, the object should be promoted to the fast tier, which may cause
> objects in the fast tier to be demoted. When slow tier capacity is exhausted, the least
> frequently accessed object is evicted from the slow tier.
>
> - **Admission**: while fast tier capacity is unreached, objects go to the fast tier; once
>   reached, every new object goes to the slow tier.
> - **Demotion**: the least frequently accessed fast-tier object moves to the slow tier when
>   fast-tier space is needed.
> - **Promotion**: a slow-tier object moves to the fast tier once its access frequency exceeds
>   the minimum frequency among fast-tier residents.
> - **Eviction**: the least frequently accessed slow-tier object is removed when cache capacity
>   is exhausted.

Same overall shape as `lru_hybrid_cache` (see above): **one** `PaperCache<K, TieredBuffer>`, not
two composed instances — "fast" vs. "slow" is which allocator a given object's bytes currently
live in, not two separate caches. Requirements 1–5 from the `lru_hybrid_cache` section (single
instance, actual data movement, TTL survives a tier move, configurable tier size, terminal
evictions counted) all carry over unchanged; only the tier-membership *rule* differs
(frequency-ordered rather than recency-ordered).

### Design decisions (differences from `lru_hybrid_cache`, confirmed during planning)

1. **Shared `TieredBuffer`, mutually exclusive features.** `lru_hybrid_cache` and
   `lfu_hybrid_cache` both define an inherent-method `impl<K, S> PaperCache<K, TieredBuffer, S>`
   block. Two such blocks for the identical concrete type can't coexist, so `TieredBuffer` was
   relocated out of `lru_hybrid_cache/buffer.rs` into a neutral `src/tiered_buffer.rs`, gated
   `any(lru_hybrid_cache, lfu_hybrid_cache)` and re-exported from both feature modules (source
   compatible — `paper_cache::TieredBuffer` still resolves the same way). `lib.rs` has a
   `compile_error!` guard rejecting `all(lru_hybrid_cache, lfu_hybrid_cache)`.
2. **No low-water headroom for demotion.** `LruHybridStack` drains to a 90%-of-capacity floor
   because *every* `set()` re-admits to fast and could re-trigger demotion. `LfuHybridStack`'s
   demotion is only triggered by a promotion or an explicit `resize_fast_tier` — inherently
   rarer — so `settle_fast_tier` drains exactly back to `fast_capacity`, no separate constant.
3. **Admission still always lands fast first**, exactly like `LruHybridStack` — deliberately
   *not* special-cased to route new objects straight to the slow tier once the fast tier is
   full. This keeps `lib.rs`'s `set()` for `TieredBuffer` under `lfu_hybrid_cache` identical to
   the `lru_hybrid_cache` version (always synchronously build `TieredBuffer::new_fast`, no
   synchronous capacity check). It still satisfies the paper's admission rule as an *emergent*
   result: once the fast tier is full, a freshly admitted object (frequency 1) is always tied
   for the fast tier's lowest frequency, so `settle_fast_tier` demotes *someone* tied at that
   frequency in the same `insert` call — see the next point for who, specifically.
4. **Ties within a frequency bucket break toward the least-recently-touched key**, matching the
   existing `LfuStack` (plain LFU) convention already in this codebase
   (`worker/policy/policy_stack/lfu_stack.rs`: `CountStack::push` = `push_front` on touch,
   `pop` = `pop_back` on eviction). This means a freshly admitted key does **not** necessarily
   demote itself once fast is full — if an older, untouched key shares its frequency, that older
   key demotes instead (it's the "more LRU" of the tied pair). Either way, the demoted key is,
   by construction, tied for the fast tier's lowest frequency. Caught this the hard way: an
   initial unit test asserted the newcomer always demotes and failed — see below.

### Implementation

**New policy: `PaperPolicy::LfuHybrid`** (`policy.rs`), string form `"lfu-hybrid"`, a bare
literal like `LruHybrid` (no embedded params).

**New policy stack: `LfuHybridStack`** (`worker/policy/policy_stack/lfu_hybrid_stack.rs`). Two
independent frequency-bucket chains (`FrequencyChain`, a private helper adapted from `LfuStack`'s
classic O(1) LFU structure — ascending-by-count `dlv_list::VecList<CountStack>` +
`HashMap<HashedKey, Index<CountStack>>`), one for each tier — chosen over a single shared
structure (as `LruHybridStack` uses for recency) because LFU's fast/slow boundary is a
*frequency* threshold, and each chain needs its own O(1)-queryable minimum. `FrequencyChain` adds
one operation `LfuStack` doesn't need: `insert_at(key, count)`, which places a key directly into
an *arbitrary* existing count's bucket (creating it in sorted position if absent) — needed when a
promoted/demoted key crosses chains carrying its already-accumulated frequency, rather than
always starting fresh at count 1. Unlike `bump`'s O(1) adjacent-bucket check, `insert_at` requires
a linear scan to find/create the right bucket — an accepted O(distinct frequencies in the target
chain) cost, expected small since the fast tier is DRAM-budget-limited.

`LfuHybridStack` fields: `fast_chain`/`slow_chain: FrequencyChain`, `tiers: HashMap<HashedKey,
Tier>`, `sizes: HashMap<HashedKey, ObjectSize>`, `fast_capacity`/`fast_used`/`slow_used`, and
`migrations: Vec<(HashedKey, Tier)>` — same shape as `LruHybridStack`'s fields minus the
recency-list/boundary bookkeeping, plus the second chain. `insert`/`update`/`remove`/`clear`/
`evict_one`/`resize_fast_tier`/`drain_tier_migrations`/the four gauge methods all implement the
same `PolicyStack` trait `LruHybridStack` already extended (no new trait methods needed —
`resize_fast_tier`, `drain_tier_migrations`, and the four gauge methods were already added as
default-no-op methods for `LruHybridStack` and are reused as-is). `evict_one` prefers the slow
chain, falling back to the fast chain's minimum if slow is empty (mirrors `LruHybridStack`'s
fallback for "nothing has ever been demoted").

**Worker plumbing (`worker/policy/mod.rs`, `worker/manager.rs`, `worker/mod.rs`) is almost
entirely reused, not duplicated per-feature** — `PolicyWorker::new_with_tier_migration`,
`handle_resize_fast_tier`, `WorkerEvent::ResizeFastTier`, and the `Tier` re-export chain are all
generic over `Tier`/`PolicyStack`, so their `#[cfg(feature = "lru_hybrid_cache")]` gates were
simply widened to `any(feature = "lru_hybrid_cache", feature = "lfu_hybrid_cache")`. Two spots
are genuinely feature-specific and got a parallel sibling instead of a shared implementation
(since they call out to differently-named `AtomicStatus` methods/types per feature — `lru_hybrid_stats`
vs. `lfu_hybrid_stats` — and the two features are mutually exclusive, so only one sibling ever
compiles): `apply_tier_migrations` (records to `lru_hybrid_*` vs. `lfu_hybrid_*` counters/gauges)
and the terminal-eviction counter increment inside `apply_evictions`.

**`AtomicStatus`** (`status.rs`): `fast_tier_capacity` is a single shared field/method pair
(gated `any(lru_hybrid_cache, lfu_hybrid_cache)` — safe to share since the two features are
mutually exclusive), but the promotions/demotions/evictions counters and gauges are a separate,
independently-named `lfu_hybrid_*` set (not merged with `lru_hybrid_*`), so each feature stays
self-contained/removable, matching how `tiering_manager` and `hybridcache` already coexist as
separate concepts in this file.

**Overhead estimate** (`object/overhead.rs::get_policy_overhead`): additive, following the
established pattern — plain `Lfu`'s existing overhead (84) plus a 24-byte `tiers` HashMap entry
(+1 byte `Tier` tag) plus a 24-byte `sizes` HashMap entry (+4 bytes for the object size, matching
the "+4" charge already used for `TwoQ`/`Arc`/`SThreeFifo`) = 137 total.

**New module `src/lfu_hybrid_cache/`** (`mod.rs`, `stats.rs` for `LfuHybridStats`) mirrors
`lru_hybrid_cache/`'s shape but re-exports `TieredBuffer` from the shared `tiered_buffer` module
rather than owning it.

**`lib.rs`**: a new `#[cfg(feature = "lfu_hybrid_cache")] impl<K, S> PaperCache<K, TieredBuffer,
S>` block, adapted mechanically (not generically) from the `lru_hybrid_cache` one — same method
list, only the seeded `PaperPolicy` and the stats method/type differ.

**Cargo feature**: `lfu_hybrid_cache = ["key_value_pmem"]`, same dependency reasoning as
`lru_hybrid_cache`.

### A test-writing lesson this caught: don't assume the newcomer is always the one that demotes

An early unit test (`admission_once_fast_is_full_is_demoted_immediately`) asserted that when a
fast tier fits exactly one key and a second key is admitted (tying its frequency at 1), the
*newcomer* would be the one demoted — reasoning loosely from the paper's "every new object is,
by definition, the least frequently accessed" framing. Running the test against the real
implementation showed the opposite: the *older* key demoted. This is correct, not a bug — see
design decision 4 above: ties within a frequency bucket break toward whichever key is
least-recently-touched, and a newcomer is pushed to the *front* of its bucket (most-recently
touched), so it's actually the pre-existing, untouched resident that's LRU-within-the-tie and
gets evicted first. This matches `LfuStack`'s (the plain LFU policy already in this crate)
existing, already-tested tie-breaking convention — it just wasn't the direction the test author
initially expected. Fixed by correcting the test's expected outcome and its doc comment, not the
implementation. A second, related pitfall: the module-level doc comment originally claimed
`settle_fast_tier` "demotes [the freshly admitted key] right back out in the same `insert` call"
— also corrected, since that's only sometimes true (see the tie-breaking behavior above).

### A test-design lesson: `fast_capacity == max_size` does not guarantee "nothing ever demotes"

An integration test wanted a scenario where the slow tier stays empty, so `evict_one`'s
fast-tier fallback path gets exercised. The first attempt set `fast_tier_size == max_size`,
reasoning that admission failures would prevent fast_used from ever exceeding fast_capacity. This
is wrong on two counts: (1) `set()` only rejects a *single* object whose own `base_size` alone
exceeds `max_size` — it never blocks based on cumulative usage, relying entirely on the
asynchronous eviction loop to trim afterward; and (2) `fast_capacity` (tracked in raw `base_size`
bytes only) and `max_size` (tracked in `base_size` *plus* a fixed per-object policy overhead, 137
bytes here) are different accounting units, so several small objects can accumulate past
`fast_capacity` in raw bytes well before their overhead-inclusive total ever threatens
`max_size`. A tight burst of `set()` calls made this materialize as spurious demotions; even after
pacing the inserts, a *first* demotion still triggered a real, cold `TieredBuffer::new_slow` PMEM
allocation in a test that had never called `ensure_pmem_allocator_warm()`, silently stalling the
worker thread for the one-time ~45s warm-up and making eviction (a completely independent
mechanism from demotion) look like it had stopped working entirely. Fixed by making the test wait
for `status().used_size() <= max_size` after each individual `set()` (keeping admission and
eviction in lockstep so raw bytes never have a chance to spike past `fast_capacity`) rather than
relying on the capacity numbers or a fixed sleep alone. This subtlety applies to `lru_hybrid_cache`
too, not just this feature — noted in `FEATURE_FLAGS.md`.

### Implementation status: complete

All of `policy.rs`, `object/overhead.rs`, `worker/policy/policy_stack/lfu_hybrid_stack.rs`,
`worker/policy/policy_stack/mod.rs`, `status.rs`, `src/lfu_hybrid_cache/`, `src/tiered_buffer.rs`
(relocated, shared), `lib.rs`, and `Cargo.toml` are done. 55 unit/inline tests pass under
`--features lfu_hybrid_cache` (including the relocated `tiered_buffer` tests and a
`worker::policy::lfu_hybrid_tests` module mirroring `lru_hybrid_tests`'s synthetic-buffer wiring
tests). `tests/lfu_hybrid_cache_integration.rs` (17 tests, real PMEM/UMF allocator, modeled on
`tests/lru_hybrid_cache_integration.rs`) passes twice in a row (not flaky):
`cargo +nightly test --test lfu_hybrid_cache_integration --features lfu_hybrid_cache`. Confirmed
`lru_hybrid_cache`'s own test suites (unit + its 14 integration tests) are unaffected by the
`TieredBuffer` relocation and are still 100% passing.

### Remaining work

- No dedicated test yet for a multi-key single-step promotion/demotion cascade with mixed
  small/huge object sizes (same caveat noted for `lru_hybrid_cache` above — the implementation
  already returns a `Vec` of migrations per call, so this is handled, just not yet exercised by a
  test with deliberately varied sizes).
