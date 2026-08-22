# The hybrid caches

This crate hosts 18 two-tier cache designs, to compare how eviction disciplines use a small DRAM
tier in front of a large CXL/PMEM tier. Every hybrid build compiles all 18; which one a given
cache runs is chosen at construction time by the `PaperPolicy` value passed to the constructor,
and is fixed for that cache's lifetime. Two caches in one process may run different designs.

This document covers the machinery all 18 share, then each design individually. For the
feature-flag matrix see `FEATURE_FLAGS.md`; for one design end to end in maximum detail see
`LRU_HYBRID_CACHE.md`.

---

# Part 1: What every design shares

## Tier is a property of the value, not a separate cache

There is one `PaperCache<K, TieredBuffer>`. `TieredBuffer` is a tagged union recording where
this object's bytes currently are:

```rust
pub enum TieredBuffer {
    Fast(Box<[u8]>),          // node-0 arenas, via the global allocator
    Slow(Box<[u8], Hybrid>),  // node-1 arenas, via numa_alloc::SlowObjects
}
```

A live object's bytes exist in **exactly one** tier. Promotion and demotion replace the
`TieredBuffer` in place (`Object::set_data`), so a migration is a byte *move*. Contrast
`src/tiering/`, the legacy manager, which deliberately keeps a copy in both tiers at once.

All 18 share one implementation. There are exactly two inherent
`impl<K, S> PaperCache<K, TieredBuffer, S>` blocks — the shared engine, and a second carrying the
size-split design's three-scalar constructor — both gated only on `hybrid_cache_common`. The
per-design behaviour that survives is dispatched at runtime: `hybrid_policy::admission_tier`
matches on the policy to pick a placement, and `init_policy_stack` builds the matching
`PolicyStack`.

The 18 `*_hybrid_cache` features are consequently **not** mutually exclusive; any subset may be
enabled. Each now gates only a name-compatibility shim — a `pub mod <design>_hybrid_cache`
re-exporting `TieredBuffer` plus a `<Design>HybridStats` type alias — its integration-test file,
and one per-object DRAM-overhead accounting term. `lib.rs`'s single `compile_error!` rejects
`hashbrown_dram` with `global_hashtable_pmem`, unrelated to the designs.

> Earlier revisions gave each design its own impl block, forcing mutual exclusion and 153
> pairwise guards. Both are gone. Many module doc comments in `src/` — including those on the 18
> shim modules — still describe that older world.

## How a stack decides, without ever touching a byte

A `PolicyStack` tracks **order and tier membership only**. It never allocates, copies, or frees
object bytes, and it holds no reference to the object map — it cannot, since it runs inside
`PolicyWorker` while API threads are concurrently reading the map.

When a stack decides a key belongs in a different tier it pushes `(HashedKey, Tier)` onto an
internal `migrations` vector. `PolicyWorker` drains that with `drain_tier_migrations()` after
each event and performs the real work.

This is why a stack can be unit-tested with no allocator, no map, and no threads: its entire
observable output is the sequence of `(key, tier)` pairs it emits.

## Turning "this key changed tier" into an actual byte move

`PolicyWorker` owns the shared object map, so it applies migrations. It carries:

```rust
tier_migration_fn: Option<Arc<dyn Fn(&V, Tier) -> Option<V> + Send + Sync>>,
```

populated only by `PolicyWorker::new_with_tier_migration`, which every hybrid design uses. The
closure returns `Option<V>`: `None` means **declined** — the value is already in the requested
tier, so there is nothing to move. That case is normal, not an error; see "Counters vs physical
copies" below.

### The work is done by a standing consumer pool, not by the worker

`apply_tier_migrations` partitions the drained migrations into demotions and promotions, then
hands each entry to `apply_physical`. With the migration queue enabled — the default — that
function does one thing:

```rust
if let Some(queue) = migration_queue {
    queue.push((key, tier));
    return;
}
```

The worker returns immediately and gets back to mutating the stack. The **`migration_queue`
consumer threads** (`mig-N`, 2 by default) perform the allocate-copy-swap. Setting
`MIGRATION_QUEUE_THREADS=0` disables the queue, and the worker runs the identical body inline
instead.

The queue keeps **one channel per consumer, indexed by key hash**. A single shared channel with
N consumers would let two migrations for the *same* key be applied out of order: a demote and a
following promote could be picked up concurrently, and whichever won the shard lock last would
be the one whose `Arc::ptr_eq` check fails and gets discarded. Each swap is individually
correct — the value is never corrupted — but the survivor could be the *older* decision,
stranding the object in the wrong tier. Sharding by key hash makes per-key order structural.

### Ordering: demotions before promotions, at enqueue time

Demotions are enqueued fully before promotions, so the fast tier gives back space before
anything tries to move into it. Note what this is **not**: with the queue enabled it is an
ordering of *enqueues*, not a physical barrier. Consumers run concurrently across shards, so a
promotion of key B may physically complete before a demotion of key A. Per-key ordering is
guaranteed; cross-key ordering is not.

(An earlier revision did enforce a hard batch-wide barrier via two sequential
`rayon::into_par_iter().for_each` phases. That is gone — see "The abandoned fan-out" below.)

### The copy runs with no map guard held

Building the destination buffer is a real allocation plus a full byte copy — a PMEM write on
demotion, a PMEM read on promotion — and at this crate's object sizes (~16 KB average on the
benchmark traces) that is microseconds. Holding the shard's write guard across it stalls every
concurrent `get()` hashing to the same shard, surfacing as GET tail latency.

So the consumer snapshots the source bytes with no guard held (`Object::data()` is an `Arc`
refcount bump, and the `Arc` keeps the bytes alive independently of the map), does the copy, and
re-acquires the shard only to swap the pointer:

```rust
if let Some(mut object) = objects.get_mut_ref(&key) {
    if Arc::ptr_eq(&object.data(), &old_data) {
        object.set_data(new_data);
    }
}
```

The `ptr_eq` guard is load-bearing. `PaperCache::set()` runs on an API thread and can replace
the entry while the copy is in flight; writing `new_data` over a *replacement* would resurrect
the bytes of the value it replaced. If the object changed or was evicted, the migration is
stale — it is dropped, and the stack's next event re-derives the correct tier. The same `Arc`
strong reference makes this immune to ABA: the allocation cannot be reused while it is held.

### The abandoned fan-out

`parallel_migration` fanned a single *batch* across a rayon pool. It is still on the call path
as a dispatcher, but `DEFAULT_THRESHOLD = 0` makes it degenerate to a sequential `for_each` on
the worker — which, with the queue enabled, is just a loop of channel pushes.

It was measured and abandoned rather than tuned: 99.4% of demotion volume arrives as
single-object batches (37M calls of exactly 1 object against 9 calls of ≥16K), so there is
nothing to fan out. `migration_queue` replaced it by decoupling from batch boundaries
entirely — the work is genuinely fine-grained, not genuinely serial.

### Counters vs physical copies

The stats count **tier decisions**, not byte copies. They diverge in three legitimate ways:

- **Counters lead the copies.** Migrations are applied asynchronously, so a mid-run snapshot
  reports decisions not yet performed. They converge as the queue drains. There is no public
  flush — `MigrationQueue` is crate-internal, and the only call to its `flush` is
  `#[cfg(test)]`-gated, so test assertions on buffer contents stay deterministic while still
  exercising the real queue path.
- **Declined migrations are normal.** `PaperCache::set()` picks a placement of its own via
  the free function `hybrid_policy::admission_tier(policy, ...)` -- a runtime `match` carrying
  each design's rule -- so an object can already be where the stack is about to say it
  belongs. `TwoQHybridStack` hits this by design: `admission_tier` returns `Fast` for a re-set
  (correct — the key is now most-recently-used), so the value is already in DRAM by the time
  `touch_main_fast` emits its promotion.
- **A persistent gap is a defect signal.** `LfuHybridStack` once emitted a `Tier::Slow`
  migration on every latched admission whose bytes `admission_tier` had *already* placed in
  PMEM — 445,465,067 migrations against ~448M sets on cluster12, ~99% of its reported demotions.
  Any `lfu_hybrid_cache` demotion figure from before that fix is inflated and not comparable.

The four tier gauges (`fast_objects`/`slow_objects`/`fast_bytes_used`/`slow_bytes_used`) are
republished by `refresh_tier_gauges` once per event-loop pass, so they are up to one pass stale.
That refresh is deliberately unconditional: gating it on "a migration just happened" is what
once let the gauges go stale indefinitely.

## Shared machinery

### High/low watermarks

`settle_fast_tier` triggers once `fast_used` exceeds `watermarks::high_bytes` of the effective
budget, then drains in one pass down to `watermarks::low_bytes` rather than back to exactly the
ceiling. Defaults are **`DEFAULT_HIGH = 0.98` / `DEFAULT_LOW = 0.95`**, overridable at runtime
via `FAST_TIER_HIGH_WATERMARK` / `FAST_TIER_LOW_WATERMARK`. Setting both to `1.0` restores the
original drain-to-the-ceiling behaviour exactly.

The pair exists because draining to exactly the ceiling pinned the tier at 100% utilisation and
made almost every pass a single-object migration batch. It trades a slice of resident fast
capacity for larger, less frequent batches, and it is tuned in one place for all 18 stacks.

> `watermarks::DEFAULT_HIGH` / `DEFAULT_LOW` are authoritative. Both are read once through a
> `OnceLock` on first use, so the env vars are startup configuration rather than runtime
> adjustable, and a value that fails to parse or falls outside `(0.0, 1.0]` is silently
> replaced by the default. `low()` is clamped to at most `high()` so a misconfigured pair
> cannot invert and turn every pass into a no-op.

### The shared-DRAM-overhead reservation

The object hashtable and each stack's own bookkeeping live in DRAM but are not part of
`fast_used`, so demoting purely against `fast_capacity` lets real DRAM exceed its budget — which
is exactly what was observed in practice.

`get_hybrid_dram_shared_overhead` gives a per-tracked-key byte figure; `reserved_overhead()`
multiplies it by the tracked key count; `settle_fast_tier` subtracts that from `fast_capacity`
**before** applying the watermarks. The composition order is load-bearing: the reservation comes
out of capacity first, and the watermarks are ratios of what is left.

The multiplier is *every* tracked key, not just fast ones — a slow-tier object still has a
hashtable entry, a list node and an `entries` slot in DRAM.

### `eviction_stacks_pmem`

Moves each stack's lists and per-key map into the slow tier via `crate::Hybrid`. The
`PmemHashList`/`hashbrown` variants expose the same method surface as the DRAM ones, so stack
logic is identical either way. Each stack still cfgs its own imports and type aliases, but none
needs a cfg for the *accounting*: under this flag the eviction-stack term simply drops out of the
value `get_hybrid_dram_shared_overhead` returns.

### One combined per-key map

Every stack keeps a single `entries: HashMap<HashedKey, XEntry>` rather than parallel
`tiers`/`sizes`/`queue` maps. Nearly every operation touches those fields together, and merging
them removes one hashtable-structural-overhead charge per object per map eliminated, plus an
entire class of desync bug (a key present in one map but not another) by construction.

`LruEntry { tier, size }` is 8 bytes (`u8` + `u32`), pairing with the 8-byte `HashedKey` to
exactly 16 — the figure `object/overhead.rs`'s DRAM constants are derived from. This is why
`LruLfuHybridStack`'s frequency counter is a `u16`: a `u32` would push the entry to 12 bytes and
add 8 bytes to *every* object in *both* tiers.

### Boundary cursor vs homogeneous lists

Two structural idioms recur:

- **One list plus a cursor.** `LruHybridStack` keeps a single recency list with a
  `fast_boundary` cursor marking where the fast prefix ends. Cheap, but every operation that
  can move the boundary has to repair it.
- **Homogeneous per-tier lists.** `LruSizedHybridStack` (four lists) and `LruLfuHybridStack`
  (a recency list plus a frequency chain) give each tier its own structure, so each list's own
  tail is directly its own candidate and no cursor exists. Both arrived at this independently,
  and in both cases it came out *simpler* than the cursor.

---

# Part 2: The 18 designs

```
BASE                    2Q FAMILY                      S3-FIFO FAMILY
lru                     two_q                          s3_fifo
lfu                     ├── +fast admission            ├── +ghost
fifo                    │   └── +reprieve              │   └── +lazy demotion
lru_sized               └── +ghost                     │       ├── +fast admission
lru_lfu                                                │       │   └── +midpoint
                                                       │       │       ├── -ghost +reprieve
                                                       │       │       │   ├── -midpoint
                                                       │       │       │   └── +split slow
                                                       │       └── (slow admission) +reprieve
```

## Base designs

### `lru_hybrid_cache`

`LruHybridStack` · `PaperPolicy::LruHybrid` — selected at runtime by passing this policy to `new()`

One recency-ordered list backs both tiers. The fast tier is the maximal prefix from the MRU end
whose cumulative size fits `fast_capacity`; everything behind is slow.

- **Admission** — front of the list, `Tier::Fast`, unconditionally.
- **Promotion** — every access *and every overwrite* moves the key to the front and tags it
  `Fast`. A `set()` on an existing key is treated exactly like a hit.
- **Demotion** — when `fast_used` crosses the high watermark, the LRU-most fast key (found via
  `fast_boundary`, no scan) is demoted, repeating down to the low watermark.
- **Eviction** — the absolute LRU tail, which after any demotion is always slow.

This stack is where sub-ceiling headroom was first needed, for a specific reason:
`PaperCache::set()` writes a new object's `TieredBuffer` to DRAM synchronously at the API layer,
before the stack running on `PolicyWorker` sees the event. A burst of concurrent `set()` calls
can transiently push real DRAM above what the stack's bookkeeping shows, and draining below the
ceiling leaves that burst somewhere to land. The pressure is sharpest here because this is the
stack that re-settles on every admission.

The headroom is no longer stack-local, though: the old `FAST_TIER_LOW_WATER_RATIO` is now dead
code, and every design gets the same behaviour from the shared watermark pair above.

### `lfu_hybrid_cache`

`LfuHybridStack` · `PaperPolicy::LfuHybrid` — selected at runtime by passing this policy to `new()`

Two independent `FrequencyChain`s (the classic O(1) LFU bucket structure) back the two tiers.
LFU's boundary is a *frequency* threshold, not a list position, so two chains — each queryable
for its own minimum in O(1) — are the natural fit.

- **Admission** — fast at frequency 1 while the fast tier has room; once full, straight to slow.
  This is the paper rule read literally: "every new object is admitted into the slow tier" means
  *the new object*, not whichever key loses a tie-break.
- **Promotion** — a slow access bumps that key's frequency; if it strictly exceeds the fast
  chain's current minimum (or the fast chain is empty) the key moves to the fast chain,
  **preserving its accumulated frequency** via `insert_at`.
- **Demotion** — `settle_fast_tier` on the watermark, demoting the fast minimum.

**The admission latch.** A raw byte check is not enough to keep admission honouring frequency
order, because demotion granularity is per-object, not per-byte: demoting a 90-byte object to
cover a 5-byte overage leaves 85 bytes of slack, and a brand-new frequency-1 key admitted purely
because that slack exists bypasses the "prove yourself via promotion" path entirely — even when
every fast resident already has frequency ≥ 2. So `fast_tier_latched` permanently closes
brand-new-key fast admission the first time capacity is genuinely reached. It resets on
`clear()` and on `resize_fast_tier` *growing* the budget.

### `fifo_hybrid_cache`

`FifoHybridStack` · `PaperPolicy::FifoHybrid` — selected at runtime by passing this policy to `new()`

One insertion-ordered list. **No promotion policy at all** — this is the defining difference
from every sibling.

- **Admission** — front (bottom of the fast tier).
- **Promotion** — none. `update()` is deliberately left as the trait's default no-op body,
  matching this crate's plain `FifoStack`. A hit on a slow key must never migrate it back.
  *Do not add an override here* — a refactor pass assuming this stack "forgot" to override
  `update()` like its siblings would be wrong.
- **Overwrite** — never repositions the key and never changes its tier; only the size accounting
  for whichever tier it already occupies is corrected.
- **Demotion** — the oldest fast key. **Eviction** — the absolute tail.

### `lru_sized_hybrid_cache`

`LruSizedHybridStack` · `PaperPolicy::LruSizedHybrid` — the one design with its own
constructor: `new_sized(max_size, small_fast_tier_size, large_fast_tier_size, size_threshold)`,
which takes no policy argument. Passing `LruSizedHybrid` to `new()` returns `InvalidPolicy`.

`lru_hybrid_cache`'s semantics with each tier's bookkeeping split into two size-routed segments
by a runtime-configurable byte threshold. Four homogeneous lists (`small_fast`, `large_fast`,
`small_slow`, `large_slow`), no cursor — two independent fast sources each feeding their own slow
destination is a shape a cursor does not generalise to.

- **Classification** uses the same `ObjectSize` (key + value + expiry slot) every other stack
  budgets against, not a raw `value.len()`. Threading a second size channel through
  `PolicyStack::insert` for all 27 stacks that implement it would buy only a small near-constant
  offset.
- **Admission, promotion and reclassification** all funnel through one `touch_fast` method,
  landing wherever `classify` says the key's *current* size belongs. A fast→fast segment move
  emits **no** migration — both segments are physically `TieredBuffer::Fast`.
- **Eviction** prefers whichever slow list holds more objects (a cheap proxy for "probably has
  the older tail"); only if both are empty does it fall back to whichever fast segment is
  furthest over budget by ratio.
- Only the two fast segments have capacities. The slow lists are governed by the overall
  `max_size` trigger, exactly like the single slow tier in `lru_hybrid_cache`.
- The shared overhead is split *proportionally* between the two fast segments, not charged in
  full against each — the per-object metadata cost is real only once.

`set_large_fast_tier_size()`/`large_fast_tier_size()` and `set_size_threshold()`/
`size_threshold()` live on every hybrid cache -- they sit on the shared impl block -- but take
effect only when the cache is running this design.

### `lru_lfu_hybrid_cache`

`LruLfuHybridStack` · `PaperPolicy::LruLfuHybrid(promote_k)` — selected at runtime by passing
this policy to `new()`; `promote_k` is a `u16` and is rejected if zero.

The first design whose two tiers rank by *different* metrics: a recency list for the fast tier, a
frequency chain for the slow tier. In one line: **frequency is the admission control into DRAM;
recency is the retention policy within DRAM.**

The rationale is that the tiers have different jobs. The fast tier is small and holds the active
working set, where recency is the right short-term signal. The slow tier is far larger and holds
a long cold tail, where LRU is close to meaningless — its tail position is dominated by scans and
one-hit-wonders. LFU is scan-resistant and actually identifies which cold objects deserve
promotion.

- **Admission** — fast tier, recency head, frequency 1. Admitting to slow instead would make
  every `set()` a synchronous PMEM allocation, measured at 2.15x SET mean / 2.72x SET p99 in the
  `two_q` vs `two_q_fast_admission` comparison, on traces that are 80–94% SETs.
- **Demotion** — the fast LRU tail, **carrying its accumulated frequency** into the slow chain.
- **Promotion** — a slow object whose frequency *reaches* `promote_k` moves to the fast recency
  head, and its counter **resets**.
- **Eviction** — the slow chain's minimum-frequency key (ties broken least-recently-touched).

**`promote_k` is an absolute frequency, not accesses-since-demotion.** This is easy to get wrong
(this file's own tests did): a key admitted and never accessed demotes carrying frequency 1, so
`promote_k == 2` behaves exactly like `promote_k == 1`. The first threshold that filters anything
is **3**. Conversely a genuinely hot key arrives in the slow tier at or near the cap and promotes
on its next access regardless — deliberate, and the point of carrying frequency at all: proven
popularity needs one confirmation, a one-hit-wonder has to earn the whole climb.

**Why the fast tier counts a frequency it does not rank by:** without it, an object hit 50 times
in DRAM would land in the slow chain indistinguishable from a one-hit-wonder demoted in the same
pass — precisely the objects most likely to be referenced again.

**Why the counter is capped, and capped small.** `FREQUENCY_CAP` pays twice. *Memory*: a `u16`
keeps `LruLfuEntry` at 8 bytes so the pair stays 16; a `u32` would cost +8 bytes on every object
in both tiers. *Demotion cost*: `FrequencyChain::insert_at` is a linear scan over count buckets
and every demotion performs one, so the cap directly bounds it. Pair the cap with
reset-on-promotion, without which a repeatedly-promoted object accumulates an unassailable count
and becomes effectively un-evictable.

**A `set()` is an access, not an automatic promotion** — a deliberate divergence from
`lru_hybrid_cache`. On an 80–94% SET trace, "any set promotes" would push nearly everything into
DRAM without demonstrating reuse, and the slow tier's frequency ordering would never filter
anything. Consequently `admission_tier` must look up an existing key's current tier, so an
overwrite of a slow key is written straight to PMEM rather than to DRAM and corrected afterward.

## 2Q family

A one-access FIFO queue feeding a segmented main LRU queue. All four carry `k_in` in their policy
payload -- `PaperPolicy::TwoQHybrid(k_in)` and siblings, validated into `0.0..=1.0` -- where
`k_in * max_size` is the FIFO queue's byte budget.

### `two_q_hybrid_cache`

`TwoQHybridStack` · `PaperPolicy::TwoQHybrid(f64)`

Two live queues, matching the paper text directly — unlike this crate's plain `TwoQStack`, which
carries a heavier three-live-queue shape with a real-object `a1_out` overflow queue.

- `fifo_queue` — one-access, holds real objects, **always entirely in the slow tier**.
- `main_stack` — recency-ordered, segmented fast/slow exactly like `LruHybridStack::stack`.

- **Admission** — a brand-new key lands in `fifo_queue`, so a first `set()` is a synchronous
  PMEM write. A re-`set()` of an already-tracked key is built in DRAM instead: `admission_tier`
  returns `Tier::Fast` once the key is in the object map, because `touch()` always ends with the
  key in the fast tier.
- **Promotion** — a hit on a `fifo_queue` key moves it straight to the top of `main_stack` at
  `Tier::Fast`. Once inside `main_stack` an object behaves exactly like `lru_hybrid_cache`.
- **Ageing out** — a `fifo_queue` object reaching the tail without a second access is evicted
  outright. No ghost queue: an exact-membership check on every admission was judged an unwelcome
  cost given admission already pays a synchronous PMEM write. A probabilistic structure is the
  right tool to revisit this, and is left as future work.
- **Eviction priority** — `fifo_queue` tail, then `main_stack`'s slow tail, falling back to its
  fast tail only if nothing has ever been demoted. This reconciles the paper's two eviction
  clauses into one rule: sacrifice unproven FIFO objects before touching the proven main queue.

`fifo_capacity` and `fast_capacity` are two independent knobs here — one governs a slow/PMEM
queue, the other the main queue's DRAM portion.

**This stack never evicts on its own.** `needs_capacity_eviction` reports when `fifo_used`
exceeds `fifo_capacity` and the caller (`PolicyWorker::apply_evictions`) does the removal, since
a `PolicyStack` has no reference to the object map.

`entries` holds `TwoQEntry { queue, tier: Option<Tier>, size }`, with `tier: None` iff the key is
in the FIFO queue.

### `two_q_fast_admission_hybrid_cache`

`TwoQFastAdmissionHybridStack` · `PaperPolicy::TwoQFastAdmissionHybrid(f64)`

`two_q_hybrid_cache` with the one-access queue in the **fast** tier, so admission is a cheap DRAM
write instead of a synchronous PMEM allocation. Only the physical placement changes; the logical
queue structure is untouched, and a key still has to prove itself with a second access to reach
the recency-durable part of the cache. Only its bytes are in DRAM *while on probation*.

**The accounting had to change, not just the label.** With the FIFO queue also in DRAM, both
budgets draw on the same physical pool, and leaving them independent would let real DRAM grow to
`fast_capacity + fifo_capacity`. Fixed by treating `fifo_capacity` as a reservation carved out
first — `effective_main_fast_capacity =
fast_capacity.saturating_sub(fifo_capacity).saturating_sub(reserved_overhead())`, so the FIFO
carve-out and the shared per-object DRAM reservation both come out before the watermarks apply. The net result is
`fast_used (main) + fifo_used <= fast_capacity` by construction.

### `two_q_fast_admission_reprieve_hybrid_cache`

`TwoQFastAdmissionReprieveHybridStack` · `PaperPolicy::TwoQFastAdmissionReprieveHybrid(f64)`

As above, but a one-access key ageing out **without** a second access is reprieved into the slow
tier rather than evicted. `settle_fifo_queue` splices it onto the **back** of `main_stack` — the
absolute LRU tail, i.e. the next eviction candidate — tagged `Tier::Slow`.

Deliberately weaker than the s3-fifo equivalent, which splices to the *front* of its slow
segment. Two reasons:

1. **Rank.** The main queue is LRU-ordered and its slow segment holds keys that were promoted at
   least once and later demoted. A key ageing out of the one-access queue has demonstrated *no*
   reuse, so ranking it above proven-but-cold objects would invert the ordering.
2. **Cost.** `push_back` is O(1) on the existing list. The s3-fifo variant needed to insert at
   the fast/slow *boundary*, which `HashList` cannot do — its first implementation walked every
   fast key per reprieve and burned ~18 minutes of worker CPU on a real trace without completing,
   which is what forced that stack's two-physical-list restructure.

The tradeoff when reading results: a reprieved key at the LRU tail may be evicted very soon under
pressure, having cost a real DRAM→PMEM copy on the way there. Whether that buys enough extra hits
is exactly what this variant exists to measure — a null result is a finding, not a bug.

### `two_q_ghost_hybrid_cache`

`TwoQGhostHybridStack` · `PaperPolicy::TwoQGhostHybrid(f64)`

`two_q_hybrid_cache` plus a bare-key ghost queue, adding what that stack deliberately left out.
Mirrors `s_three_fifo_stack.rs`'s `ghost: HashList<HashedKey>` shape — a lightweight membership
list, not a third place bytes can live (explicitly chosen over plain `TwoQStack`'s heavier
`a1_out`, which holds real objects).

Ghost lifecycle, matching `SThreeFifoStack`'s convention:

- **Added to** only when a `fifo_queue` object ages out without a second access — never by a
  main-queue eviction.
- **Checked** by `insert`'s brand-new-key branch.
- **Not removed on a hit** — trimmed lazily, capped relative to `main_count`, only during a
  genuine main-queue eviction; cleared outright by `remove`/`clear`.

A ghost hit is admitted directly into `main_stack` at `Tier::Fast`. This was an explicit,
acknowledged-as-arguable choice — the conservative alternative (land in the slow portion and earn
promotion normally) is a one-line change if measurement says otherwise. It is cheap either way:
`set()` always builds a brand-new key as `new_slow` regardless of ghost history, so a ghost hit
costs one extra migration rather than a synchronous PMEM-vs-DRAM decision at the API layer.

## S3-FIFO family

All nine carry `one_access_ratio` in their policy payload -- `PaperPolicy::S3FifoHybrid(ratio)`
and siblings, validated into `0.0..=1.0`.

### `s3_fifo_hybrid_cache`

`S3FifoHybridStack` · `PaperPolicy::S3FifoHybrid(f64)`

Structurally close to `two_q_hybrid_cache` — a one-access queue in the slow tier feeding a main
queue segmented fast/slow — but the mechanism deciding who stays is different.

- `one_access_queue` behaves like 2Q's: a re-access promotes **eagerly** to the front of
  `main_queue` at `Tier::Fast`. Anything still there at the tail has by construction never been
  re-accessed, so eviction from it is unconditional — no bit to check.
- `main_queue` is **pure FIFO, never reordered on access**. Every `Main` key carries an
  `accessed: bool` reference bit (classic CLOCK/second-chance), set on every touch regardless of
  which portion the key occupies. The bit is consulted **lazily**, only when a key reaches the
  tail and is about to be evicted: set → `give_second_chance` (reinserted at the front, retagged
  `Fast`, bit cleared); clear → evicted for real.
- **Demotion is unconditional** — whichever key anchors `main_boundary` ages down. The reference
  bit plays no part here, only at eviction time. This is S3-FIFO's "quick demotion, lazy
  promotion".

The asymmetry is a direct reading of the paper: "if accessed again *before reaching the top* of
the one-access FIFO queue, they are promoted" (eager) versus "if they *have been re-accessed
during this period*, they are reinserted", evaluated only "after objects have traversed through
both portions... and are about to be evicted" (lazy).

**The contiguous front run invariant.** `main_queue` is never reordered except by insertion at
the front and by demotion, which moves nothing in the list at all — it only re-tags the key
`main_boundary` points at and walks the cursor one step. So "the front `main_boundary`-worth of
keys are exactly the `Tier::Fast` ones" always holds. A key given a second chance re-enters at
the front, deliberately scrambling true insertion age in exchange for matching the paper's
wording — exactly how a real CLOCK sweep works.

### `s3_fifo_ghost_hybrid_cache`

`S3FifoGhostHybridStack` · `PaperPolicy::S3FifoGhostHybrid(f64)`

Adds a bare-key ghost queue with the same lifecycle as `two_q_ghost_hybrid_cache`'s, bringing the
hybrid design in line with this crate's plain `SThreeFifoStack`, which already has a ghost queue
of exactly this shape.

### `s3_fifo_ghost_lazy_demotion_hybrid_cache`

`S3FifoGhostLazyDemotionHybridStack` · `PaperPolicy::S3FifoGhostLazyDemotionHybrid(f64)`

One change: **demotion is now reference-bit gated too.** Before demoting the key anchoring
`main_boundary`, its `accessed` bit is checked.

- **Bit set** — the key was touched since promotion, so it gets a fresh start here: moved to the
  front of the fast portion, bit cleared, tier and accounting untouched. This is a *reprieve, not
  a promotion* — it was already `Fast` and stays `Fast`, so **no migration is produced**. The
  sweep continues to the next-oldest fast key.
- **Bit clear** — demoted for real.

S3-FIFO's tagline becomes "lazy demotion, lazy promotion": the bit now gates both tier
transitions. The eviction-time second chance is unchanged and still matters — the two mechanisms
protect different things (an unfairly *demoted* fast key here, an unfairly *evicted* slow key
there) and compose naturally. Termination is guaranteed because each reprieve clears the bit and
moves the key to the front, so it cannot be re-examined until every other fast key has had a turn.

### `s3_fifo_ghost_lazy_demotion_fast_admission_hybrid_cache`

`S3FifoGhostLazyDemotionFastAdmissionHybridStack`

Moves the one-access queue to the **fast** tier, so admission is a cheap DRAM write. Same change,
same motivation, and same shared-DRAM-budget accounting as
`two_q_fast_admission_hybrid_cache`: `one_access_capacity` becomes a reservation carved out of
`fast_capacity` rather than an independent budget.

### `s3_fifo_ghost_lazy_demotion_fast_admission_midpoint_hybrid_cache`

`S3FifoGhostLazyDemotionFastAdmissionMidpointHybridStack`

Adds a checkpoint roughly halfway through the **slow** portion of the main queue. The slow
portion was previously a passive holding area — nothing looked at an object there until it
reached the eviction tail; a reaccess in the meantime only set its reference bit for that tail
check to find. Now, if the object at the midpoint has
its reference bit set, it gets the same treatment as a tail-reached second chance instead of
having to survive all the way to the tail. A genuinely cold object at the midpoint is left alone
and still gets its one real chance at the tail.

The check runs once per `evict_one()` call, after the one-access queue is confirmed empty — the
same cadence the tail check already runs at.

Locating "the middle" uses an incrementally-maintained cursor with a drift counter, not a walk:
the slow segment holds hundreds of thousands of objects at benchmark scale, and an O(n) scan once
per eviction would be O(n²) over a cache's lifetime.

### `s3_fifo_lazy_demotion_fast_admission_midpoint_reprieve_hybrid_cache`

`S3FifoLazyDemotionFastAdmissionMidpointReprieveHybridStack`

Two changes from the midpoint variant:

1. **No ghost queue.** Since a one-access key ageing out is no longer evicted (see 2), nothing
   ever populates a ghost list. Rather than keep a permanently-empty structure, the ghost list and
   all machinery serving it were removed outright.
2. **The one-access tail is reprieved, not evicted** — moved into the slow tier of the main queue
   and given a full life there, promotable via the ordinary touch/midpoint/tail machinery.

**The reprieve runs synchronously from `insert()`/`resize()`** via a new `settle_one_access()`,
mirroring `settle_fast_tier()` — deliberately *not* through `evict_one()`. The first draft routed
it through `evict_one()` and hit a real bug: `apply_evictions` unconditionally erases whatever key
`evict_one()` returns from the *entire cache*, and if it returns `None`, `erase()` falls back to
evicting a **random** object. A reprieve is neither of those, and `over_max_size` might not even
be true at that moment. So `evict_one()` here is purely about the main queue, and
`needs_capacity_eviction()` stays at the trait's default `false`.

### `s3_fifo_lazy_demotion_fast_admission_reprieve_hybrid_cache`

`S3FifoLazyDemotionFastAdmissionReprieveHybridStack`

The reprieve design with **no mid-slow checkpoint at all**. Keeps only the checks that pay for
themselves: the demotion boundary and the eviction tail. See "Recorded negative results" below
for why both checkpoint attempts were dropped.

### `s3_fifo_lazy_demotion_reprieve_hybrid_cache`

`S3FifoLazyDemotionReprieveHybridStack`

Fills the family's one empty design cell — a **slow**-tier one-access queue whose aged-out keys
are **reprieved**:

| variant | one-access tier | ages out without reaccess |
|---|---|---|
| `s3_fifo` (+ghost, +lazy demotion) | slow | evicted |
| `...fast_admission...` (+midpoint) | fast | evicted |
| `...fast_admission_reprieve...` (+split slow) | fast | reprieved |
| **this** | **slow** | **reprieved** |

**The splice costs nothing.** In the fast-admission reprieve variants the one-access queue is
DRAM and the main queue's slow segment is PMEM, so every aged-out object costs a real
`TieredBuffer::new_slow` copy. Here both structures are in PMEM: `settle_one_access` moves the key
between lists and emits **no migration at all**. The reprieve is strictly cheaper than the
eviction it replaces.

The cost is on the other side of the ledger — the paper-literal admission rule it keeps means a
brand-new key's `set()` is a synchronous PMEM write, which is exactly what the fast-admission
branch exists to avoid. (As in `two_q_hybrid_cache`, a re-`set()` is not: `admission_tier` keeps
an existing key in whichever tier it already occupies, so a re-`set()` of a fast-resident key is
built in DRAM. Here that is load-bearing rather than an optimisation — this stack records no tier
transition for a `set()` on a tracked key, so building in the wrong tier would strand the object
physically in one tier while the stack accounts it to the other, and nothing reconciles that.)

**Promotion is a real move again**, the mirror image: the fast-admission variants push no
migration on promotion from the one-access queue because the bytes were already in DRAM. Here a
one-access key really is in PMEM, so promotion is a genuine PMEM→DRAM move and must emit the
migration — guarded, since `settle_fast_tier` may demote it straight back out in the same call.

### `s3_fifo_lazy_demotion_fast_admission_split_slow_reprieve_hybrid_cache`

`S3FifoLazyDemotionFastAdmissionSplitSlowReprieveHybridStack`

Replaces the approximate midpoint cursor with a **real structural boundary**: the slow tier is
split into two physical FIFO segments, and every object's reference bit is checked at the moment
it would cross from the newer segment into the older one — full coverage rather than one sampled
key per eviction pass.

## Recorded negative results

Three variants exist as measured dead ends and are kept deliberately, so the lineage is not
re-explored by accident.

**The midpoint cursor did nothing.** Benchmarked against the real traces it was
indistinguishable from not being there: largest difference 291 hits out of 2.34M accesses, i.e.
0.01%.

**And not for the reason first assumed.** An earlier draft of that stack's own doc claimed the
cursor sampled too few keys to matter. *That was wrong.* In steady state the cursor holds a
roughly fixed index while objects flow past it, so it lands on a new object each cycle and sees
most objects crossing the midpoint. Its coverage was fine.

**The actual reason is structural: an earlier checkpoint cannot save anything the tail check
wouldn't.** Terminal eviction only ever removes the slow tier's *tail*, so any object whose
reference bit is set is already spared when it arrives there. A mid-tier check changes *when* a
reaccessed object returns to DRAM, never *whether* it survives.

**The full structural boundary did nothing either, and cost.** The split-slow variant tested the
one remaining hypothesis — that checking *every* crossing object gets hot objects back to DRAM
earlier and more uniformly, a residency effect rather than a survival one. It does not: hit
counts were bit-identical to the cursor version on all three traces, while costing **2.7–11.8% on
GET p99** and **1.2–6.9% on GET throughput**. The extra Slow→Fast migrations are pure added work
on the `PolicyWorker` thread and the object map's shard locks.

**The batch fan-out did nothing.** See "The abandoned fan-out" above: 99.4% of demotion volume
arrives as single-object batches.

## Comparison at a glance

| Design | Fast tier order | Slow tier order | Admission | Promotion | Knob |
|---|---|---|---|---|---|
| `lru` | recency | recency (same list) | fast | any access or set | — |
| `lfu` | frequency | frequency | fast until latched | freq > fast min | — |
| `fifo` | insertion | insertion (same list) | fast | **never** | — |
| `lru_sized` | recency ×2 | recency ×2 | fast | any access or set | `size_threshold` |
| `lru_lfu` | recency | **frequency** | fast | freq reaches `promote_k` | `promote_k` |
| `two_q` | recency | recency | slow (FIFO queue) | 2nd access, eager | `k_in` |
| `two_q_fast_admission` | recency | recency | **fast** (FIFO queue) | 2nd access, eager | `k_in` |
| `s3_fifo` | insertion + ref bit | insertion + ref bit | slow (1-access queue) | eager from queue, lazy in main | `one_access_ratio` |
| `s3_fifo_*_fast_admission_*` | insertion + ref bit | insertion + ref bit | **fast** (1-access queue) | eager from queue, lazy in main | `one_access_ratio` |
