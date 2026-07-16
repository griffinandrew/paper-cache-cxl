# `lru_hybrid_cache`: a single-instance, segmented-LRU hybrid cache

This document explains what `lru_hybrid_cache` is and why it's built the way it is — the design
rationale and the road taken to get there. For the current, authoritative description of exactly
*how* this and its two siblings (`lfu_hybrid_cache`, `two_q_hybrid_cache`) work today — including
the shared migration machinery and the DRAM-budget reservation added after this document was
written — see `HYBRID_CACHES.md`, which supersedes this file wherever the two disagree. For the
day-to-day feature-flag reference see `FEATURE_FLAGS.md`; for the implementation history (what was
tried, what broke, what was fixed) see `CLAUDE.md`.

## What problem this solves

The feature implements this eviction-policy design (a segmented LRU across a fast and a slow
memory tier):

> The LRU eviction queue is segmented across two tiers. New objects are admitted into the fast
> tier at the top of the queue. As objects age without being accessed they drift down the queue
> and are eventually demoted to the slow tier. When a slow tier object is accessed, it is promoted
> back to the top of the fast tier. When cache capacity is exhausted, the least recently accessed
> object is evicted from the slow tier.

Concretely, in this crate: the fast tier is DRAM, the slow tier is persistent/CXL memory (PMEM),
and "the queue" is an LRU recency order shared by every object the cache holds — there is no
separate eviction policy per tier. An object's tier is not a separate cache membership; it's a
property of *where that object's bytes are currently allocated*.

Two hard requirements shaped every design decision below:

1. **One unified cache instance.** Not two `PaperCache`s wired together with channels — a single
   logical LRU queue, a single object table, a single background worker.
2. **Actual data movement.** A live object's bytes exist in exactly one tier's allocation at a
   time. Promotion/demotion physically reallocate the object; nothing is ever copied into both
   tiers simultaneously.

This is a deliberate departure from the two other tiering mechanisms already in this crate:

- `hybridcache` (`S3FifoHybridCache`) gets its "two tiers" by composing **two independent**
  `PaperCache` instances (one `BufferDRAM`, one `BufferPMEM`), glued together with channels and
  `DashSet`s for in-flight tracking. Its promotion uses **copy-on-read**: the PMEM copy is never
  deleted, so a key can legitimately live in both tiers at once.
- The `tiering/` module (`enable_tiering_manager` / `tiering` / `multitiering`) is
  hotness-threshold, copy-based: PMEM is always the source of truth, and "hot" objects get an
  additional physical copy placed in a DRAM side-cache. Again, two copies can coexist.

`lru_hybrid_cache` copies neither pattern. Below is how it actually achieves "one instance, real
data movement" instead.

## The core idea: tier is a value, not a cache

`PaperCache<K, V, S>` is generic over the value type `V`. Every other storage variant in this
crate (`BufferDRAM = Box<[u8]>`, `BufferPMEM = Box<[u8], Hybrid>`) is a fixed, single allocation
strategy for the whole cache. `lru_hybrid_cache` instead introduces one new value type that is
itself a **tagged union of the two allocation strategies**:

```rust
// src/lru_hybrid_cache/buffer.rs
pub enum TieredBuffer {
    Fast(Box<[u8]>),           // DRAM — the crate's global allocator
    Slow(Box<[u8], Hybrid>),   // PMEM — the same Hybrid/UMF allocator BufferPMEM uses
}
```

`PaperCache<K, TieredBuffer, S>` is then just **one more ordinary instantiation** of the existing
generic cache type — one object map (`DashMap<HashedKey, Object<K, TieredBuffer>>`), one
`AtomicStatus`, one background `WorkerManager`/`PolicyWorker`, exactly like every other `V`. There
is no second cache, no second hashtable, no wrapper struct. "Fast" and "slow" are simply which
variant of `TieredBuffer` a given `Object`'s `data` field currently holds. Migrating a key between
tiers means replacing that one enum value in place — nothing else about the cache changes.

`TieredBuffer` implements what the generic `PaperCache` machinery needs:

- `AsRef<[u8]>` — matches both arms, so `get()`/`peek()` can read bytes uniformly regardless of
  tier.
- `TypeSize` (`get_size() == self.as_ref().len()`) — matches how `BufferPMEM`'s own `TypeSize` impl
  already just returns `self.len()`; size accounting doesn't care which allocator backs the bytes.
- `Clone`, plus `new_fast(bytes)` / `new_slow(bytes)` constructors and `is_fast()` / `is_slow()`
  predicates.

## Where the "one logical queue, two tiers" idea actually lives

Given the tier is per-object, *something* still has to decide, on every access, which objects
count as "fast" and which as "slow," and to do it without knowing anything about `TieredBuffer`
specifically. That's the job of a new `PolicyStack` implementation.

### `PaperPolicy::LruHybrid` and `LruHybridStack`

`policy.rs` gets a new variant, `PaperPolicy::LruHybrid` (string form `"lru-hybrid"`). Unlike
`TwoQ(f64, f64)` or `SThreeFifo(f64)`, it carries no embedded parameter — the fast-tier size must
be adjustable at runtime (see below), not fixed when the policy is chosen.

`worker/policy/policy_stack/lru_hybrid_stack.rs` implements `LruHybridStack`, the `PolicyStack` for
this policy. It tracks *order and tier membership only* — it never touches actual object bytes;
that's the `PolicyWorker`'s job (next section). Its state:

```rust
pub struct LruHybridStack {
    stack: HashList<HashedKey, NoHasher>,      // one recency-ordered list, same structure LruStack uses
    sizes: HashMap<HashedKey, ObjectSize, NoHasher>,
    tiers: HashMap<HashedKey, Tier, NoHasher>, // which tier each tracked key is logically in

    fast_capacity: CacheSize,                  // the runtime-configurable fast-tier byte budget
    fast_used: CacheSize,                      // current bytes accounted to the fast tier
    slow_used: CacheSize,

    fast_boundary: Option<HashedKey>,          // see below
    migrations: Vec<(HashedKey, Tier)>,        // pending (key, new tier) pairs, drained each pass
}
```

**Key structural fact this design leans on:** because every admission/promotion moves a key to the
*front* of the list and every demotion removes from the *tail end of the fast segment*, the set of
`Tier::Fast` keys is always a **contiguous prefix** of the recency list, starting from the head.
There is never a Slow key ahead of (more recently used than) a Fast key. That means the "boundary"
between the two tiers is always a single point in the list — the last Fast key, adjacent to
whatever the first Slow key is (or to the tail, if there are no Slow keys yet).

`LruHybridStack` exploits this to avoid ever scanning the list: it keeps a single
`fast_boundary: Option<HashedKey>` pointing at *the current least-recently-used Fast key* — the
next demotion candidate — and maintains it incrementally using `HashList::before`/`front` (an O(1)
"what's immediately ahead of this key" lookup), rather than walking the list to find it.

### The four operations

- **Admission** (`insert`, called from `set()`): a brand-new key is pushed to the front, tagged
  `Tier::Fast`, and its size added to `fast_used`. An existing key being re-`set()` is treated the
  same way — even if it was previously `Slow`, a `set()` always re-admits to the top of the fast
  tier (this is also what makes an update-in-place safely converge: there is never a stale copy
  left behind in the other tier, because the old tier's accounting is subtracted before the new
  tier's is added).
- **Demotion** (`settle_fast_tier`, called after every `insert`/`update`): triggered only once
  `fast_used` actually exceeds `fast_capacity` (an early-return guard skips the rest of the
  function otherwise — no early demotion below the user-configured budget). Once triggered, the
  drain loop repeatedly takes the key at `fast_boundary`, flips its tier to `Slow`, moves
  `fast_boundary` to whatever key was `before` it in the list (which, by the contiguous-prefix
  invariant, must still be Fast), and records `(key, Tier::Slow)` in `migrations` — draining exactly
  back down to `fast_capacity`, not below it (see "No headroom" below). Because this is a `while`
  loop rather than a single check, one oversized admission or promotion can demote more than one
  key in a single call — the implementation never assumes "one in, one out."
- **Promotion** (`update`, called from a `get()` hit): if the accessed key is currently `Slow`,
  move it to the front, flip it to `Fast`, add its size to `fast_used`, record `(key, Tier::Fast)`
  in `migrations`, then immediately run the same `settle_fast_tier` demotion check — a promotion
  can cascade into demoting whatever is now the fast tier's new least-recently-used key.
- **Eviction** (`evict_one`, called by the existing generic `apply_evictions` loop whenever overall
  `status.used_size() > max_size`): pops the absolute tail of the recency list. Once any demotion
  has ever happened, that tail is guaranteed to be a `Slow` key (nothing Fast can be behind a Slow
  key, by the same invariant). If the whole working set still fits in the fast tier (no demotion
  has happened yet), this degrades gracefully to evicting the Fast tail instead of panicking or
  refusing — a defensive fallback, not the expected steady state.

`drain_tier_migrations()` hands the accumulated `migrations: Vec<(HashedKey, Tier)>` to the caller
and clears it; `resize_fast_tier(new_capacity)` updates `fast_capacity` and immediately re-runs
`settle_fast_tier` (so shrinking the budget at runtime demotes eagerly, not lazily on the next
access).

Four new default (no-op) methods were added to the `PolicyStack` trait itself so no other policy
had to change: `resize_fast_tier`, `drain_tier_migrations`, and four gauge readers
(`fast_bytes_used`, `slow_bytes_used`, `fast_object_count`, `slow_object_count`) that
`LruHybridStack` overrides and every other stack ignores.

### Low-water headroom: removed, then reintroduced smaller, for a different reason

An earlier version of this implementation drained demotions down to a 90%-of-capacity floor
(`fast_low_water`) instead of exactly `fast_capacity`, reasoning that draining to the exact ceiling
would leave the fast tier hovering right at the boundary — so almost every subsequent `set()` would
push `fast_used` back over `fast_capacity` and re-trigger a demotion pass. That headroom was removed
at the user's explicit request: keeping idle capacity in reserve costs usable fast-tier space for a
marginal reduction in demotion-pass frequency, and the user judged that trade not worth it. For a
while, `settle_fast_tier` drained exactly back down to `fast_capacity` and no lower.

**This was later revisited for an unrelated reason and a small floor was reintroduced.**
`PaperCache::set()` writes a new object's `TieredBuffer` to DRAM *synchronously*, at the API layer,
before the corresponding event even reaches `PolicyWorker` — so a burst of concurrent `set()` calls
can transiently push real DRAM usage above what the stack's own bookkeeping shows, in the window
between that physical write and the worker processing it. `settle_fast_tier` now drains to 98% of
the effective budget (`FAST_TIER_LOW_WATER_RATIO`), not back to exactly the ceiling — a small margin
that leaves a concurrent burst some room to land in before the next settle needs to trigger again.
This is much smaller than the original 90% floor and exists for a different reason (burst safety,
not thrashing reduction); it's paired with (not a substitute for) `apply_tier_migrations` running
per-event rather than per-batch, which is what actually shrinks the demote-decision-to-physical-
DRAM-free window. See `HYBRID_CACHES.md`'s "Low-water headroom" section for the current, precise
description, and `CLAUDE.md` for the full back-and-forth.

## Turning "this key changed tier" into an actual byte move

`LruHybridStack` only ever manipulates `HashedKey`s and a `Tier` enum — it has no access to the
object map and no idea what `TieredBuffer` is. The physical move happens one layer up, in
`PolicyWorker` (`worker/policy/mod.rs`), which already owns the shared object map.

```rust
// One extra field on PolicyWorker<K, V>, only populated for lru_hybrid_cache:
tier_migration_fn: Option<Box<dyn Fn(&V, Tier) -> V + Send + Sync>>,
```

After processing *each* `Get`/`Set`/etc event (not once per batch, so a demotion decision made
mid-batch gets physically executed as soon as possible under concurrent load), `PolicyWorker::run()`
calls `apply_tier_migrations()`, which drains the stack's pending migrations
and applies them in two parallel phases — every demotion, fully complete, before any promotion
begins — via `rayon`. See `HYBRID_CACHES.md`'s "Turning 'this key changed tier' into an actual byte
move" section for the current, exact mechanism (this document previously showed a simple sequential
`for` loop here, which was accurate at the time but has since been replaced for throughput reasons
under demotion-heavy real workloads).

`migrate` is supplied at construction time (`PaperCache::new`, see below) as:

```rust
Box::new(|buffer: &TieredBuffer, tier| match tier {
    Tier::Fast => TieredBuffer::new_fast(buffer.as_ref()),
    Tier::Slow => TieredBuffer::new_slow(buffer.as_ref()),
})
```

`Object::set_data` (`object/mod.rs`, one new method) is what makes this an in-place replacement
rather than a delete-then-reinsert:

```rust
pub fn set_data(&mut self, data: V) {
    self.data = Arc::new(data);
}
```

It touches only `data`. `key` and `expiry` are untouched — which is exactly how TTL survives a
tier move for free: there's nothing to "carry over," because the same `Object` never stopped
existing. This is a case where the "one unified instance" requirement turned out to make the
*harder*-sounding requirement (TTL survival) trivial: in a two-instance design, TTL would have to
be read off one instance and manually reapplied on the other during every migration.

**Byte-length invariant.** `migrate` must never change the value's byte length. `AtomicStatus`'s
`base_used_size` is computed once at insert time (`overhead_manager.base_size(&object)`, which
depends only on `key_size + value_len + expiry_slot_size`, plus a fixed TTL bookkeeping constant if
`expiry.is_some()`) and subtracted again at erase time via the identical recomputation. If a
migration silently changed the byte length, insert-time and erase-time size would disagree, and
since `base_used_size` is an unsigned counter, the mismatch would eventually underflow/wrap to a
huge value — which was in fact discovered during development (see `CLAUDE.md`) when a throwaway
test `migrate` closure appended a marker byte instead of overwriting one in place; the mismatch
made `apply_evictions`'s `while used_size() > max_size` loop never terminate. `TieredBuffer::
new_fast`/`new_slow` are straight byte-for-byte copies, so this is satisfied by construction in the
real implementation.

**No status delta is needed for a migration itself.** `overhead_manager.base_size`/`total_size`
depend only on the *logical* byte length and the active policy's fixed per-object overhead —
neither depends on which allocator backs the bytes. So swapping `Fast` ↔ `Slow` never changes an
object's accounted size; `status.used_size()` is only ever touched by real admission/deletion/
eviction, not by a migration.

**Concurrency.** All tier migrations happen on the single `PolicyWorker` background thread — there
is no second thread doing PMEM writes for the public API to coordinate with, unlike `hybridcache`'s
`in_flight_demotions`/`in_flight_promotions` `DashSet`s. A `get()` on the public API thread racing a
migration on the worker thread just sees `DashMap`'s normal per-shard locking hand it either the
pre- or post-migration `Arc<TieredBuffer>` — never a torn state, and never a state where the key
is visible in neither or both tiers, because there is only one map entry for that key, ever.

## Stats: why they live on `AtomicStatus`, not a new field on `PaperCache`

`LruHybridStats` (`src/lru_hybrid_cache/stats.rs`) is a plain snapshot struct:

```rust
pub struct LruHybridStats {
    pub promotions: u64,
    pub demotions: u64,
    pub evictions: u64,       // terminal removals from the slow tier
    pub fast_bytes_used: u64,
    pub slow_bytes_used: u64,
    pub fast_objects: u64,
    pub slow_objects: u64,
}
```

The natural-looking design — a dedicated `Arc<AtomicLruHybridStats>` field on `PaperCache`, mirroring
`hybridcache`'s `S3FifoHybridCache::stats: Arc<AtomicHybridStats>` — turned out to be wrong for this
feature specifically, because `PaperCache<K, V, S>`'s struct definition is **one shared definition**
used by every value type in the crate, and its literal is duplicated across roughly ten
constructors throughout `lib.rs` (the same reason the existing `tiering_manager` field is
`#[cfg(...)]`-gated rather than added unconditionally). Adding a new field there would force every
other constructor — none of which know or care about `TieredBuffer` — to also learn how to
initialize it.

Instead, the seven counters/gauges are fields directly on `AtomicStatus` (`status.rs`), gated
`#[cfg(feature = "lru_hybrid_cache")]`. `AtomicStatus` is already the one structure both
`PaperCache` and `PolicyWorker` hold an `Arc` to (`status: StatusRef`), and it has exactly one
construction site (`AtomicStatus::new`), so this needed no changes anywhere else. `PolicyWorker`
writes directly via `self.status.record_lru_hybrid_promotion()` / `::demotion()` / `::eviction()`
and `self.status.set_lru_hybrid_gauges(...)`; `PaperCache::lru_hybrid_stats()` reads them back with
`self.status.lru_hybrid_stats()`. `evictions` is only incremented from `apply_evictions()` when the
active policy is actually `PaperPolicy::LruHybrid`, so building with the feature enabled doesn't
pollute unrelated caches' bookkeeping.

## Runtime-configurable fast-tier size

Requirement: tier size must be adjustable at runtime, in bytes/MB/GB (reusing the existing
`CacheTierSize` enum, moved from `hybridcache`-only to a shared `src/size.rs` gated `any(hybridcache,
lru_hybrid_cache)` so neither feature depends on the other for it).

This mirrors the existing `resize()`/`WorkerEvent::Resize` precedent exactly:

- `AtomicStatus` gets `fast_tier_capacity: AtomicCacheSize` plus `fast_tier_capacity()` /
  `set_fast_tier_capacity()`.
- `WorkerEvent::ResizeFastTier(CacheSize)` is a new event variant (a no-op for every policy stack
  except `LruHybridStack`, via the default trait method).
- `PaperCache::set_fast_tier_size(&self, size: CacheTierSize) -> Result<(), CacheError>` validates
  `0 < bytes <= max_size` (returning the new `CacheError::InvalidFastTierSize` otherwise), writes
  `status.set_fast_tier_capacity(bytes)` synchronously, then broadcasts `ResizeFastTier` so
  `PolicyWorker` calls `stack.resize_fast_tier(bytes)` — which, as noted above, eagerly re-runs
  `settle_fast_tier`, so shrinking the budget can trigger immediate demotions rather than waiting
  for the next access.
- `PaperCache::fast_tier_size(&self) -> CacheSize` just reads `status.fast_tier_capacity()` back —
  no round trip to the worker thread needed, since `status` is the shared source of truth for the
  *current* value (the worker only needs to be told when it changes).

## Public API (`impl<K, S> PaperCache<K, TieredBuffer, S>`, in `lib.rs`)

Adapted from the existing `BufferDRAM`/`BufferPMEM` impl blocks — mechanically the same shape, no
new logic beyond what's described above:

| Method | Notes |
|---|---|
| `new(max_size, fast_tier_size: CacheTierSize)` / `with_hasher(...)` | Fixed single policy (`PaperPolicy::LruHybrid`); constructs via `WorkerManager::new_with_tier_migration`, then broadcasts the caller's requested fast-tier size to override `init_policy_stack`'s 20%-of-`max_size` default |
| `get(&self, key: &K) -> Result<Vec<u8>, CacheError>` | A hit on a `Slow`-tier key triggers promotion via the worker (not synchronously — see below) |
| `set(&self, key: K, value: &[u8], ttl: Option<u32>) -> Result<(), CacheError>` | Always admits via `TieredBuffer::new_fast` |
| `del`, `has`, `peek`, `ttl`, `size`, `wipe`, `resize` | Same shape as every other `PaperCache<K, V, S>` variant |
| `set_fast_tier_size(&self, size: CacheTierSize) -> Result<(), CacheError>` / `fast_tier_size(&self) -> CacheSize` | See above |
| `lru_hybrid_stats(&self) -> LruHybridStats` | Snapshot, see above |
| `tier_of(&self, key: &K) -> Option<Tier>` | Test/diagnostic accessor: reads the tier directly off the single object map (`object.data().is_fast()`), returning `None` if the key is absent or expired. There is no `has_in_dram`/`has_in_pmem` pair to reuse here (that pattern is specific to `hybridcache`'s two-instance design) since there's only one map to look in |

There is deliberately **no `policy()` method** on this impl block — every other multi-policy
`PaperCache` variant exposes `policy()` to switch between several configured policies at runtime,
but this cache is only ever configured with the one, fixed `PaperPolicy::LruHybrid`.

Note that `get()`'s promotion is *not* synchronous with the call that returns the value: `get()`
reads whatever `TieredBuffer` variant is currently in the map (fast or slow, either way readable
via `AsRef<[u8]>`) and returns immediately; the promotion (the actual byte move to the fast tier)
happens asynchronously on the `PolicyWorker` thread shortly after, once it processes the
corresponding `WorkerEvent::Get(key, hit=true)`. This is why integration tests that need to observe
a promotion poll (`tier_of(key) == Some(Tier::Fast)`) rather than asserting immediately after
`get()` returns.

## Module map

```
src/
  policy.rs                              PaperPolicy::LruHybrid ("lru-hybrid")
  object/
    mod.rs                               Object::set_data
    overhead.rs                          per-object overhead estimate for PaperPolicy::LruHybrid
  status.rs                              AtomicStatus: fast_tier_capacity + the 7 lru_hybrid_* counters/gauges
  size.rs                                CacheTierSize (shared with hybridcache)
  error.rs                               CacheError::InvalidFastTierSize
  worker/
    mod.rs                               WorkerEvent::ResizeFastTier
    manager.rs                           WorkerManager::new_with_tier_migration
    policy/
      mod.rs                             PolicyWorker::new_with_tier_migration, apply_tier_migrations,
                                          handle_resize_fast_tier; re-exports Tier
      policy_stack/
        mod.rs                           Tier enum; PolicyStack::{resize_fast_tier, drain_tier_migrations,
                                          fast_bytes_used, slow_bytes_used, fast_object_count, slow_object_count}
        lru_hybrid_stack.rs              LruHybridStack (the segmented-LRU algorithm itself)
  lru_hybrid_cache/
    mod.rs                               module docs, re-exports
    buffer.rs                            TieredBuffer
    stats.rs                             LruHybridStats (plain snapshot struct)
  lib.rs                                 impl<K, S> PaperCache<K, TieredBuffer, S>; crate-root re-exports
                                          (TieredBuffer, LruHybridStats, Tier)

tests/
  lru_hybrid_cache_integration.rs         end-to-end tests against the real PMEM allocator (see below)
```

## How this differs from `hybridcache` and `tiering/`, concretely

| | `tiering/` (`enable_tiering_manager`) | `hybridcache` (`S3FifoHybridCache`) | `lru_hybrid_cache` |
|---|---|---|---|
| Cache instances | 1 (`PaperCache`) + a side `TieringManager` hashtable | 2 (`PaperCache<K,BufferDRAM>` + `PaperCache<K,BufferPMEM>`) | 1 (`PaperCache<K,TieredBuffer>`) |
| Promotion trigger | access count ≥ configurable threshold | S3-FIFO ghost-queue hit | any access to a Slow-tier key |
| Data on promotion/demotion | copied (PMEM stays source of truth) | copied (copy-on-read; PMEM copy never deleted) | moved (`Object::set_data`, one copy ever) |
| Can a key be in both tiers? | yes, always (PMEM + DRAM cache) | yes (until re-evicted) | never |
| Eviction tier | N/A (nothing is ever fully evicted by the tiering manager itself) | either tier's own independent LRU/S3-FIFO | always the slow tier's LRU tail |
| Cross-thread coordination | shared `Arc<TieringManager>` | `DashSet`s for in-flight demotion/promotion windows | none needed — single worker thread owns all migrations |

## Testing

- **Unit tests** (`worker/policy/policy_stack/lru_hybrid_stack.rs`): the algorithm in isolation —
  admission, demotion under pressure, promotion (including cascading demotion), both `evict_one`
  branches, zero-capacity, runtime resize, remove, clear, object counts. No threads, no allocator.
- **`worker/policy/mod.rs::lru_hybrid_tests`**: the `PolicyWorker` wiring end to end, using a plain
  `Box<[u8]>` value type (not `TieredBuffer`) and a marker-tagging `migrate` closure, so these run
  without the real PMEM allocator — proves `apply_tier_migrations`/`Object::set_data`/stats
  recording all work together correctly.
- **`lib.rs::test_lru_hybrid_cache`**: the real public `PaperCache<K, TieredBuffer>` API, kept to
  the fast-tier-only path (`fast_tier_size == max_size`) so it never needs the PMEM allocator.
- **`tests/lru_hybrid_cache_integration.rs`** (14 tests): the full path through the real
  `Hybrid`/UMF PMEM allocator — this is the suite that actually proves the "real data movement, one
  copy ever" requirement, by demoting a key and confirming with `tier_of` that it is *gone* from
  the fast tier (not just present in the slow tier), and symmetrically for promotion. Also covers
  TTL surviving both directions of migration, terminal eviction accounting, runtime
  `set_fast_tier_size`, and edge cases (zero/invalid/tiny capacities).

Run everything:

```bash
cargo +nightly test --lib --features lru_hybrid_cache
cargo +nightly test --test lru_hybrid_cache_integration --features lru_hybrid_cache
```

The integration tests need a machine where the `Hybrid`/UMF allocator can actually create a PMEM
pool (a real PMEM/CXL DIMM, or — as in the sandbox this was developed and verified in — any
NUMA node UMF can bind an OS memory provider to). The very first PMEM allocation in a test process
triggers a one-time pool init + prewarm that can take on the order of a minute; the test file's
`ensure_pmem_allocator_warm()` helper pays that cost up front so it doesn't race against any single
test's own timing assertions.
