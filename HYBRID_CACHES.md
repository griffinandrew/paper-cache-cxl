# The hybrid caches: how `lru_hybrid_cache`, `lfu_hybrid_cache`, and `two_q_hybrid_cache` work

This is the reference for how all three single-instance DRAM/PMEM hybrid caches are built and,
specifically, exactly how objects move between the fast (DRAM) and slow (PMEM) tier. For the
day-to-day feature-flag reference see `FEATURE_FLAGS.md`; for the chronological implementation
history (what was tried, what broke, what was fixed, and why) see `CLAUDE.md`. `LRU_HYBRID_CACHE.md`
is a deeper design-rationale writeup of `lru_hybrid_cache` specifically and is still worth reading
for the "why," but this document is the current, authoritative description of *how* — for all
three policies — and supersedes anything in that file that disagrees with it.

## The shared idea: tier is a value, not a cache

All three features implement the same architectural pattern, just with a different eviction
algorithm deciding *when* an object crosses tiers. Each is **one** `PaperCache<K, TieredBuffer, S>`
— one object map, one `AtomicStatus`, one background `PolicyWorker` — not two `PaperCache`s wired
together. "Fast" and "slow" are not two caches an object gets copied between; they're a property of
*where a single object's bytes are currently allocated*, encoded in the value type itself:

```rust
// src/tiered_buffer.rs — shared by lru_hybrid_cache and lfu_hybrid_cache
// (two_q_hybrid_cache uses the same type, just admits into Slow first)
pub enum TieredBuffer {
    Fast(Box<[u8]>),           // DRAM — the crate's global allocator
    Slow(Box<[u8], Hybrid>),   // PMEM — the same Hybrid/UMF allocator BufferPMEM uses
}
```

A promotion or demotion **physically reallocates** the object's bytes into the other variant —
never a copy left behind in both. This is a deliberate departure from the other two tiering
mechanisms already in this crate:

| | `tiering/` (`enable_tiering_manager`) | `hybridcache` (`S3FifoHybridCache`) | These three hybrid caches |
|---|---|---|---|
| Cache instances | 1 `PaperCache` + a side `TieringManager` hashtable | 2 (`PaperCache<K,BufferDRAM>` + `PaperCache<K,BufferPMEM>`) | 1 (`PaperCache<K,TieredBuffer>`) |
| Data on promotion/demotion | copied (PMEM stays source of truth) | copied (copy-on-read; PMEM copy never deleted) | **moved** (`Object::set_data`, one copy ever) |
| Can a key be in both tiers? | yes, always | yes (until re-evicted) | **never** |
| Cross-thread coordination | shared `Arc<TieringManager>` | `DashSet`s for in-flight demotion/promotion windows | none needed — one worker thread owns all migrations |

`PaperCache<K, V, S>` is generic over the value type `V`; `TieredBuffer` is just one more
instantiation of that, alongside `BufferDRAM = Box<[u8]>` and `BufferPMEM = Box<[u8], Hybrid>`.

Each feature adds one new `PaperPolicy` variant and one new `PolicyStack` implementation that
decides tier membership; everything else (the object map, the background worker, the migration
machinery below) is shared code, not duplicated per feature.

| Feature | Policy | Policy stack | Fixed policy string |
|---|---|---|---|
| `lru_hybrid_cache` | `PaperPolicy::LruHybrid` | `LruHybridStack` | `"lru-hybrid"` |
| `lfu_hybrid_cache` | `PaperPolicy::LfuHybrid` | `LfuHybridStack` | `"lfu-hybrid"` |
| `two_q_hybrid_cache` | `PaperPolicy::TwoQHybrid(k_in)` | `TwoQHybridStack` | `"2q-hybrid-{k_in}"` |

The three features are mutually exclusive with each other (`lib.rs` has `compile_error!` guards for
every pairwise combination) since each defines the same inherent-method `impl<K, S>
PaperCache<K, TieredBuffer, S>` block for the same concrete type.

## How a `PolicyStack` decides — without ever touching a byte

A `PolicyStack` (`worker/policy/policy_stack/mod.rs`) only ever manipulates `HashedKey`s and a
small `Tier` enum:

```rust
pub enum Tier { Fast, Slow }
```

It has **no access to the object map** and no idea what `TieredBuffer` is — it just tracks *which
tier each key is logically in* using whatever ordering its algorithm calls for (a recency list for
LRU, two frequency-bucket chains for LFU, two live queues for 2Q — see the per-policy sections
below). Every stack method that can change a key's tier appends a `(HashedKey, Tier)` pair to an
internal `migrations: Vec<...>` buffer; `drain_tier_migrations()` hands that buffer to the caller
and clears it. This split — *decide* in the stack, *move bytes* one layer up — is what lets all
three policies share one physical-migration path instead of each reimplementing it.

Five trait methods (all default no-ops so no other, non-hybrid `PolicyStack` needs to change) carry
this:

```rust
fn resize_fast_tier(&mut self, _size: CacheSize) {}
fn drain_tier_migrations(&mut self) -> Vec<(HashedKey, Tier)> { Vec::new() }
fn fast_bytes_used(&self) -> CacheSize { 0 }
fn slow_bytes_used(&self) -> CacheSize { 0 }
fn fast_object_count(&self) -> usize { 0 }
fn slow_object_count(&self) -> usize { 0 }
```

Two more exist for reasons specific to one policy each (see their sections below):
`drain_demotions() -> u64` (LFU only — a `Tier::Slow` migration there isn't always a genuine
demotion) and `needs_capacity_eviction() -> bool` (2Q only — its FIFO queue can need eviction
independently of the overall `max_size` budget).

## Turning "this key changed tier" into an actual byte move

`PolicyWorker` (`worker/policy/mod.rs`) owns the shared object map, so it's the one that physically
applies migrations. It's built with one extra field, populated only via
`PolicyWorker::new_with_tier_migration` (the only constructor all three hybrid caches use):

```rust
tier_migration_fn: Option<Box<dyn Fn(&V, Tier) -> V + Send + Sync>>,
```

supplied at `PaperCache::new` as:

```rust
Box::new(|buffer: &TieredBuffer, tier| match tier {
    Tier::Fast => TieredBuffer::new_fast(buffer.as_ref()),
    Tier::Slow => TieredBuffer::new_slow(buffer.as_ref()),
})
```

After every processed `WorkerEvent` (`Get`/`Set`/`Del`/`Wipe`/`Resize`/`ResizeFastTier`/`Policy`),
`PolicyWorker::run()`'s event loop calls `apply_tier_migrations()` — per event, not once per batch,
specifically to keep the window between a demotion decision and its physical DRAM-freeing copy as
short as possible under concurrent load.

### The physical copy is parallelized, with a strict demotion-before-promotion barrier

`apply_tier_migrations` drains the stack's pending migrations, splits them into two groups, and
applies them in two sequential phases:

```rust
let (demotions, promotions): (Vec<_>, Vec<_>) = migrations
    .into_iter()
    .partition(|(_, tier)| *tier == Tier::Slow);

demotions.into_par_iter().for_each(|entry| {
    apply_physical(entry);
    status.record_..._demotion();
});

// only starts once every demotion above has returned
promotions.into_par_iter().for_each(|entry| {
    apply_physical(entry);
    status.record_..._promotion();
});
```

Each phase runs its entries concurrently via `rayon` (`into_par_iter().for_each`) for real
multi-core throughput on the actual byte copy (`TieredBuffer::new_slow`'s PMEM allocation is the
expensive part — profiling showed it as ~26% of total process CPU time under a demotion-heavy real
workload). The two phases are **not** interleaved: `for_each` only returns once every entry in that
call has completed, so this is a hard, batch-wide guarantee that every demotion has already freed
its fast-tier DRAM before any promotion in the same batch starts allocating new fast-tier DRAM —
never just a pairwise/push-order approximation of that ordering.

`apply_physical` for each entry, regardless of policy:

```rust
let apply_physical = |(key, tier): (HashedKey, Tier)| {
    if let Some(mut object) = objects.get().get_mut(&key) {
        let new_data = migrate(&object.data(), tier);
        object.set_data(new_data);
    }
};
```

`objects` here is `&self.objects` wrapped in a small `AssertSync<T>` newtype (unconditional `unsafe
impl Send + Sync`, with a `.get() -> &T` accessor rather than a public `.0` field — Rust's
disjoint-closure-capture analysis would otherwise capture the field projection `objects.0` as just
the inner `&ObjectMapRef<K, V>`, bypassing the wrapper's `unsafe impl` entirely; a method call
forces capturing the whole wrapper). This lets the closures be shared across `rayon` worker threads
without threading `K: Send + Sync, V: Send + Sync` bounds through the whole generic
worker/policy-stack call chain — mirroring the crate's existing `unsafe impl<K, V> Send for
PolicyWorker<K, V>` and `PaperCache<K, V, S>: Send + Sync` precedent (both unconditional, for the
same reason). Safety rests on the same argument already given for those: all access to the object
map goes through `DashMap`'s own per-shard locking, so no unsynchronized mutable access is ever
actually exposed.

`LfuHybridStack`'s sibling applies every migration the same way but counts demotions differently
(see its section below) via `stack.drain_demotions()` instead of counting `Tier::Slow` entries
directly, since not every `Tier::Slow` migration there is a genuine demotion.

### `Object::set_data`: why this is a move, not a delete-then-reinsert

```rust
// src/object/mod.rs
pub fn set_data(&mut self, data: V) {
    self.data = Arc::new(data);
}
```

It touches only `data`; `key` and `expiry` are untouched. That's the entire reason TTL survives a
tier move "for free" — there's nothing to carry over, because the same `Object` never stopped
existing. In a two-instance design (like `hybridcache`) TTL would have to be read off one instance
and manually reapplied on the other during every migration; here it's structurally impossible for
it to diverge.

**Byte-length invariant.** `migrate` must never change the value's byte length.
`AtomicStatus::base_used_size` is computed once at insert time and subtracted again at erase time
via the identical recomputation, which depends only on the byte length. A migration that silently
changed length would desync those two computations — since `base_used_size` is an unsigned counter,
that mismatch would eventually underflow to a huge value and hang `apply_evictions`'s `while
used_size() > max_size` loop forever. (This was found in exactly this form during development, when
a throwaway test `migrate` closure appended a marker byte instead of overwriting one in place —
see `CLAUDE.md`.) `TieredBuffer::new_fast`/`new_slow` are straight byte-for-byte copies, so real
migrations satisfy this by construction.

**No `used_size()` delta from a migration itself.** `overhead_manager.base_size`/`total_size`
depend only on the object's logical byte length and the active policy's fixed per-object overhead —
neither depends on which allocator backs the bytes. Swapping `Fast` ↔ `Slow` never changes an
object's accounted size; `status.used_size()` is only touched by real admission/deletion/eviction.

**Concurrency.** All tier decisions and all physical migrations happen inside `PolicyWorker` (on its
own background thread, now parallelized internally via `rayon` for the physical-copy phase — see
above); there is no second thread doing PMEM writes that the public API has to coordinate with, and
no in-flight-tracking `DashSet`s like `hybridcache` needs. A `get()`/`set()` on the API-calling
thread racing a migration just sees `DashMap`'s normal per-shard locking hand it either the pre- or
post-migration `Arc<TieredBuffer>` for that key — never torn, never visible in neither or both
tiers, because there is only ever one map entry for that key.

## Bounding total DRAM, not just fast-tier values (`lru_hybrid_cache` / `lfu_hybrid_cache` only)

`fast_tier_size` naively only bounds the *value* bytes of fast-tier objects. But two more things
also live in DRAM for every tracked object of *either* tier: the shared object hashtable (one entry
per object of both tiers) and, unless `eviction_stacks_pmem` relocates them, the eviction stack's
own per-key bookkeeping (the recency list / frequency chains + their `tiers`/`sizes` maps). Both
`LruHybridStack` and `LfuHybridStack` reserve an approximate per-object cost for these out of the
fast-tier budget:

```rust
fn reserved_overhead(&self) -> CacheSize {
    self.stack.len() as CacheSize * self.shared_overhead   // "tracked_count × per-object cost"
}

fn settle_fast_tier(&mut self) {
    let effective = self.fast_capacity.saturating_sub(self.reserved_overhead());
    // demote against `effective`, not raw `fast_capacity`
}
```

`shared_overhead` defaults to `0` (so unit tests constructing a stack directly via `new(...)` keep
pure value-budget behavior) and is set via `with_shared_overhead(...)` only by
`init_policy_stack`, which computes the real value from
`object::overhead::get_hybrid_dram_shared_overhead(&policy)`:

- `HASHTABLE_ENTRY_OVERHEAD = 11` bytes, included unless `global_hashtable_pmem`/
  `global_flatmap_pmem` moves the object map to PMEM.
- `LRU_HYBRID_EVICTION_STACK_DRAM_OVERHEAD = 84` / `LFU_HYBRID_EVICTION_STACK_DRAM_OVERHEAD = 113`
  bytes, included unless `eviction_stacks_pmem` moves the eviction stacks to PMEM.

This is **demotion-only** — it never triggers eviction. If the shared metadata alone meets or
exceeds `fast_capacity`, the effective value budget saturates to `0` and every fast value drains to
slow, but terminal eviction remains governed solely by `status.used_size() > max_size`, popping the
slow tail exactly as it always does. `two_q_hybrid_cache` does not currently have this reservation
(left out of scope when it was added).

## Runtime-configurable fast-tier size

All three policies support adjusting the fast-tier budget after construction, mirroring the
existing `resize()`/`WorkerEvent::Resize` precedent:

- `CacheTierSize` (`src/size.rs`, bytes/Mb/Gb, shared with `hybridcache`) is the input type.
- `WorkerEvent::ResizeFastTier(CacheSize)` → `PolicyWorker::handle_resize_fast_tier` →
  `stack.resize_fast_tier(new_capacity)`, which re-runs the stack's own settle/demote logic
  immediately (so *shrinking* the budget can trigger demotions right away, not lazily on the next
  access).
- `PaperCache::set_fast_tier_size(&self, size: CacheTierSize) -> Result<(), CacheError>` validates
  `0 < bytes <= max_size` (`CacheError::InvalidFastTierSize` otherwise), writes
  `status.set_fast_tier_capacity(bytes)` synchronously, then broadcasts the resize event.
- `PaperCache::fast_tier_size(&self) -> CacheSize` reads `status.fast_tier_capacity()` directly —
  no round trip to the worker thread, since `status` is the shared, always-current source of truth.

`two_q_hybrid_cache` has a second, independent sizing knob on top of this: `k_in` (fixed at
construction, `PaperPolicy::TwoQHybrid(k_in)`), which sizes the one-access FIFO queue as
`fifo_capacity = k_in * max_size`. `fast_tier_size` governs `main_stack`'s fast/slow split; `k_in`
governs how much of the cache the FIFO queue gets. They're unrelated and both freely adjustable
(`k_in` rescales on `resize()`; `fast_tier_size` is adjustable at any time via
`set_fast_tier_size`).

## Stats

Each policy has its own independently-named set of counters/gauges living directly on
`AtomicStatus` (`status.rs`), not a separate field on `PaperCache` — `PaperCache`'s struct
definition is shared across every value type in the crate and duplicated across roughly ten
constructors, so a new field there would force every unrelated constructor to also learn how to
initialize it. `AtomicStatus` is already the one structure both `PaperCache` and `PolicyWorker`
hold an `Arc` to, with one construction site, so this needed no changes elsewhere.

```rust
pub struct LruHybridStats {   // LfuHybridStats / TwoQHybridStats: same shape
    pub promotions: u64,
    pub demotions: u64,
    pub evictions: u64,       // terminal removals from the slow tier
    pub fast_bytes_used: u64,
    pub slow_bytes_used: u64,
    pub fast_objects: u64,
    pub slow_objects: u64,
}
```

read via `PaperCache::lru_hybrid_stats()` / `lfu_hybrid_stats()` / `two_q_hybrid_stats()`. The
`fast_*`/`slow_*` gauges are refreshed unconditionally on every `apply_tier_migrations` call (not
gated on that call having produced a migration) — an earlier version gated the refresh on
`!migrations.is_empty()`, which let the gauges go stale and never catch up to the stack's true
state whenever a batch happened to process without triggering a further migration (see
`CLAUDE.md`'s DRAM-usage investigation for how this was originally caught).

---

## `lru_hybrid_cache`: segmented by recency

> New objects are admitted into the fast tier. As objects age without being accessed they drift
> down the queue and are eventually demoted to the slow tier. Accessing a slow-tier object promotes
> it back to the top of the fast tier. When capacity is exhausted, the least recently accessed
> object is evicted from the slow tier.

One recency-ordered list (`LruHybridStack::stack`) backs both tiers. The fast tier is always the
**contiguous prefix** of that list, starting from the head (most-recently-used) end: every
admission/access moves its key to the front, and every demotion only ever removes from the current
fast tier's tail — so a Slow key can never be more recently used than a Fast key. This invariant
lets the stack track the single boundary key (`fast_boundary`, the current least-recently-used Fast
key) incrementally instead of scanning the list on every demotion.

| | Rule |
|---|---|
| **Admission** | Every `set()` — new key or overwrite of an existing key — moves it to the front of the list and tags it `Tier::Fast`. Overwriting a currently-`Slow` key is itself a promotion. |
| **Promotion** | A `get()` hit on a `Slow` key moves it to the front and tags it `Tier::Fast`, then immediately re-runs the demotion check below (a promotion can cascade a demotion of whatever is now the new LRU tail of the fast tier). |
| **Demotion** | *Triggered* only once `fast_used` genuinely exceeds the effective budget (`fast_capacity` minus the DRAM reserved for shared metadata, see above) — never early. Once triggered, *drains* down to 98% of that budget (`FAST_TIER_LOW_WATER_RATIO`), not back to exactly the ceiling — see "Low-water headroom" below. A `while` loop, so one oversized admission/promotion can demote more than one key in a single pass. |
| **Eviction** | Pops the absolute LRU tail of the list. Once any demotion has ever happened, that tail is guaranteed `Slow` (nothing Fast can be behind a Slow key). If the whole working set still fits in the fast tier, this gracefully falls back to evicting the Fast tail instead of refusing. |

### Low-water headroom: why demotion doesn't drain to exactly the ceiling

`PaperCache::set()` writes a new object's `TieredBuffer` to DRAM **synchronously**, at the API
layer, before the corresponding event even reaches `PolicyWorker`. That means a burst of concurrent
`set()` calls can transiently push real DRAM usage above what the stack's own bookkeeping shows, in
the window between that physical write and the worker processing it. `settle_fast_tier` draining to
a target slightly below the ceiling (`FAST_TIER_LOW_WATER_RATIO = 0.98`) leaves that burst some
room to land in before the *next* settle needs to trigger again — a small safety margin layered on
top of (not a substitute for) `apply_tier_migrations` running per-event rather than per-batch, which
is what actually shrinks the demote-decision-to-physical-DRAM-free window.

This ratio has moved twice in this crate's history: an original 90%-of-capacity floor was removed
entirely at explicit request ("keeping the 10% high water mark ... hurts performance"), then a much
smaller 98% floor was reintroduced later for the burst-safety reason above — see `CLAUDE.md` for
the full back-and-forth. It applies to `LruHybridStack` only: `LfuHybridStack` doesn't re-settle on
every admission the way this stack does (see below), so the same burst-vs-thrashing tradeoff
doesn't apply the same way there, and `TwoQHybridStack` similarly has no floor (its fast-tier
pressure is only ever triggered by a promotion or an explicit resize, never by every `set()`).

---

## `lfu_hybrid_cache`: segmented by access frequency, admission gated on capacity

> While the fast tier has not yet reached capacity, new objects are admitted there. Once fast-tier
> capacity is reached, every new object is admitted into the slow tier. When a slow-tier object's
> access count exceeds the minimum frequency among fast-tier residents, it is promoted (which may
> cause a demotion). When slow-tier capacity is exhausted, the least frequently accessed object is
> evicted from the slow tier.

Two independent frequency-bucket chains (`fast_chain`/`slow_chain`, an O(1) classic-LFU structure —
ascending-by-count buckets + a `HashMap<HashedKey, Index>`), rather than one shared structure: LFU's
fast/slow boundary is a *frequency* threshold, and each chain needs its own O(1)-queryable minimum,
which a single shared structure (as LRU uses for recency) can't give directly.

| | Rule |
|---|---|
| **Admission** | An **explicit, unconditional capacity check** — a brand-new key is admitted to the fast chain only while `fast_used + size` fits the effective budget; once the fast tier is full, every subsequent new key goes straight to the slow chain. This is checked at `set()` time on the API-calling thread too (see "Admission latch" below), not decided later by the stack. |
| **Promotion** | A `get()`/re-`set()` hit on a `Slow` key promotes it only if its new access count is **strictly greater than** the fast chain's current minimum frequency (a tie does *not* promote). Promotion carries the key's already-accumulated frequency into the fast chain (`insert_at`, not a fresh count of 1), then re-runs the demotion check (can cascade). |
| **Demotion** | Repeatedly pops the fast chain's lowest-frequency key (ties break toward the least-recently-touched, matching plain `LfuStack`'s existing convention) until `fast_used` fits the effective budget again. No low-water floor — demotion here is only ever triggered by a promotion or an explicit resize, not by every admission. |
| **Eviction** | Pops the slow chain's minimum-frequency key. Falls back to the fast chain's minimum if the slow chain is empty (nothing has ever been demoted yet), mirroring LRU's fallback. |

### The admission latch: why a raw byte check alone isn't enough

Demotion here is per-*object*, not per-*byte*: demoting one low-frequency fast object to cover a
small promotion overage can free far more bytes than the overage itself, leaving slack. A raw
`fast_used + size <= budget` check alone would let a brand-new, frequency-1 key sneak into that
slack — bypassing "prove yourself via promotion" — even when every current fast resident already
has a higher frequency, which doesn't honor frequency order.

`fast_tier_latched: bool` closes this: the first time fast-tier capacity is genuinely reached (a
failed admission, or any demotion firing inside `settle_fast_tier`), it latches shut permanently —
every subsequent brand-new key goes straight to slow regardless of later byte slack, reachable only
via promotion. It resets on `clear()` and on `resize_fast_tier` **growing** the budget (a deliberate
capacity increase should be immediately usable, not gated behind promotions); a shrink leaves it
as-is.

The latch is mirrored onto `AtomicStatus` (`lfu_hybrid_admission_latched`) so `PaperCache::set()`,
running on the API-calling thread with no direct access to the worker-owned stack, can build a
brand-new key's `TieredBuffer` directly as `new_slow` once latched — matching what the stack would
decide anyway, avoiding a synchronous DRAM write immediately followed by an async PMEM correction
for the common steady-state case. This is a real, accepted latency tradeoff: that specific `set()`
call now allocates via the PMEM/UMF allocator synchronously on the calling thread, in exchange for
eliminating the write-then-correct round trip. An *existing* key is never affected by this check —
re-setting one is an access, which only the stack can decide whether to promote.

### Why demotions are counted via `drain_demotions()`, not per `Tier::Slow` entry

Unlike the other two hybrids, a `Tier::Slow` migration here isn't always a genuine demotion — it's
also how a fresh admission that the latch (or the capacity check) routed directly to slow gets
physically corrected from the API layer's default `TieredBuffer::new_fast` build. Counting every
`Tier::Slow` entry as a demotion would inflate `lfu_hybrid_stats().demotions` for admissions that
displaced nothing. `PolicyStack::drain_demotions() -> u64` (default `0`; only `LfuHybridStack`
overrides it) is backed by a `pending_demotions` counter incremented *only* inside
`settle_fast_tier` (never on admission) — `apply_tier_migrations`'s LFU sibling still physically
applies every migration in both directions, but counts demotions once per pass via this method
afterward instead of inferring them from `Tier::Slow` entries.

---

## `two_q_hybrid_cache`: an unproven FIFO queue feeding a segmented main LRU queue

> Every new object is placed in a one-access FIFO queue, entirely in the slow tier. A re-access
> promotes it to the top of the fast tier's main LRU queue (which behaves like `lru_hybrid_cache`
> from that point on). An object that reaches the top of the FIFO queue without a second access is
> evicted. When capacity is exhausted, the least recently accessed object at the bottom of the slow
> portion of the main queue is evicted.

Two live queues, matching the paper's two-queue shape (unlike this crate's plain, heavier `TwoQ`
policy, which has a real-object overflow queue too): `fifo_queue` (real objects, always slow) and
`main_stack` (recency-ordered, segmented fast/slow, structurally identical to
`LruHybridStack::stack`). No ghost/re-admission memory is kept — see "No ghost queue" below.

| | Rule |
|---|---|
| **Admission** | Every `set()` — new or existing key — is admitted or re-admitted into `fifo_queue`, entirely slow. `PaperCache::set()` always synchronously builds `TieredBuffer::new_slow` for this policy: the physical tier the API layer chooses and the tier the stack assigns agree by construction, so admission never itself produces a migration. |
| **Promotion** | A re-access (`touch`) dispatches on which queue the key is in: a `fifo_queue` hit (`promote_from_fifo`) moves it straight to the front of `main_stack` at `Tier::Fast`; a `main_stack` hit re-orders if already Fast, or promotes-and-settles if Slow (`touch_main_fast`, identical logic to `LruHybridStack::touch_fast_key`). Either can cascade a demotion. |
| **Demotion** | Only within `main_stack`, once its fast portion exceeds `fast_capacity` — identical mechanics to `lru_hybrid_cache`'s demotion, but with **no low-water floor** (fast-tier pressure here is only ever triggered by a promotion or an explicit resize, never by every `set()`, since admission never touches the fast tier directly). |
| **Eviction** | `fifo_queue`'s tail first (still-unproven, one-access objects are sacrificed before ever touching the proven main queue), then `main_stack`'s slow tail, falling back to `main_stack`'s fast tail only if nothing has ever been demoted there yet. This single priority order reconciles the paper's two separately-stated eviction clauses. |

### Why `fifo_queue` pressure can't self-evict, and the `needs_capacity_eviction` trait method

An early implementation had `insert`/`resize` pop `fifo_queue`'s tail directly whenever
`fifo_used > fifo_capacity`. This compiled and passed unit tests (which only exercise the bare
stack), but broke every integration test: a `PolicyStack` has no reference to the shared object map
or `AtomicStatus`, so popping a key from the stack's own bookkeeping without going through the real
removal path desyncs the stack from the object map permanently — the object leaks forever, still
present and `has()`-visible, but uncounted and unreachable via the stack.

The fix generalizes an existing rule: only `PolicyWorker::apply_evictions`'s `evict_one()` +
`erase()` pairing is allowed to actually remove an object (the same rule `LruHybridStack`/
`LfuHybridStack`'s demotions already respect — they only ever swap `Object::data` in place, never
remove). `PolicyStack::needs_capacity_eviction() -> bool` (default `false`; `TwoQHybridStack`
overrides it as `fifo_used > fifo_capacity`) lets `insert`/`resize` report pressure without acting
on it; `apply_evictions`'s loop condition became `while used_size() > max_size ||
policy_stack.needs_capacity_eviction()` (guarded by `stack.len() > 0` too, defensively), so
`fifo_capacity` pressure now drains through the exact same generic path global `max_size` pressure
already used. This is a pure additive default for `LruHybridStack`/`LfuHybridStack`.

### No ghost queue

An early draft added a classic-2Q-style ghost queue (bare evicted keys, checked on every admission
so a "reformed" object could skip straight back to the main queue). Rejected: an exact-membership
check on *every* `set()` — which already pays a synchronous slow-tier/PMEM write for this policy —
was flagged as an unwelcome added cost. A FIFO object that ages out without a second access is
simply evicted outright, no trace kept. A probabilistic structure (a counting Bloom filter or
similar) is the natural next step if re-admission-after-eviction turns out to matter for real
workloads, without paying an exact-membership check on every write — left as future work.

---

## Comparison at a glance

| | `lru_hybrid_cache` | `lfu_hybrid_cache` | `two_q_hybrid_cache` |
|---|---|---|---|
| Admission lands in | Fast, always | Fast while there's room, then Slow (latched) | Slow (the FIFO queue), always |
| `set()` builds | `TieredBuffer::new_fast` always | `new_slow` if new + latched, else `new_fast` | `new_slow` always |
| Promotion trigger | Any access to a Slow key | Access count strictly exceeds fast tier's minimum frequency | A second access to a FIFO-queue key, or an access to a Slow main-queue key |
| Demotion low-water floor | 98% of budget | None | None |
| DRAM-budget reservation (`shared_overhead`) | Yes | Yes | No |
| Extra sizing knob | — | — | `k_in` (FIFO queue's own byte budget) |
| Ghost/ ​re-admission memory | N/A | N/A | None (considered and rejected) |
| Terminal eviction source | Slow tail (falls back to Fast tail) | Slow chain min (falls back to Fast chain min) | FIFO tail, then main-queue slow tail, then main-queue fast tail |

## Testing

Each feature has the same three layers, at the same relative depth:

```bash
# Unit tests: the algorithm in isolation, no threads, no allocator
cargo +nightly test --lib --features lru_hybrid_cache lru_hybrid
cargo +nightly test --lib --features lfu_hybrid_cache lfu_hybrid
cargo +nightly test --lib --features two_q_hybrid_cache two_q_hybrid

# PolicyWorker wiring, using a plain Box<[u8]> value type (not TieredBuffer) and a
# marker-tagging migrate closure, so these run without the real PMEM allocator
cargo +nightly test --lib --features lru_hybrid_cache worker::policy::lru_hybrid_tests
cargo +nightly test --lib --features lfu_hybrid_cache worker::policy::lfu_hybrid_tests
cargo +nightly test --lib --features two_q_hybrid_cache worker::policy::two_q_hybrid_tests

# Integration: the real public API through the real Hybrid/UMF PMEM allocator —
# the suite that actually proves "real data movement, one copy ever" via `tier_of`
cargo +nightly test --test lru_hybrid_cache_integration --features lru_hybrid_cache
cargo +nightly test --test lfu_hybrid_cache_integration --features lfu_hybrid_cache
cargo +nightly test --test two_q_hybrid_cache_integration --features two_q_hybrid_cache
```

The integration tests need a machine where the `Hybrid`/UMF allocator can actually create a PMEM
pool. The first PMEM allocation in a test process triggers a one-time pool init + prewarm that can
take on the order of a minute; each integration test file's `ensure_pmem_allocator_warm()` helper
pays that cost up front (gated by a process-wide `Once`, so only the very first call anywhere in the
binary actually waits) so it doesn't race a test's own timing assertions — see `CLAUDE.md` for the
specific TTL-test pitfall this was written to avoid.

## Known limitations / not yet done

- **Real DRAM usage does not track `fast_tier_size` under sustained load**, for `lru_hybrid_cache`
  and `lfu_hybrid_cache` (`two_q_hybrid_cache` hasn't been separately profiled for this). The
  *logical* accounting above (`fast_bytes_used`, demotion timing, the DRAM-budget reservation) is
  correct and settles precisely — the gap is that the underlying TBB allocator retains freed memory
  rather than returning it to the OS for this workload's fragmentation pattern, confirmed
  unfixable via any exposed UMF/TBB API. See `CLAUDE.md`'s "Investigation: real DRAM usage vs.
  `fast_tier_size`" section for the full investigation, including the one approach that does fix it
  (bypassing pooling via `mmap`/`MADV_DONTNEED`) and the real throughput cost that came with it.
- `two_q_hybrid_cache` has no ghost/re-admission memory (see its section above) and no DRAM-budget
  reservation.
- None of the three have a dedicated test yet for a multi-key single-step promotion/demotion
  cascade with deliberately mixed small/huge object sizes — the byte-budgeted (not slot-counted)
  boundary already handles this correctly (each stack returns a `Vec` of migrations per call, not
  an assumed single pair), it just isn't yet exercised by a test built specifically for it.
- Re-verifying the parallel migration-copy change against the real `paper-benchmark-cxl` benchmark
  under real concurrent load (not just this crate's own tests) to directly confirm the wall-clock
  throughput win — done once (see `CLAUDE.md`), worth repeating if this area changes again.
