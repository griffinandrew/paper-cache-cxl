# paper-cache (DRAM/CXL tiering fork)

PaperCache is an in-memory cache that supports switching between eviction policies at
runtime. This fork adds **two-tier memory placement**: every object's bytes live either in
DRAM (the *fast* tier, NUMA node 0) or in PMEM/CXL (the *slow* tier, NUMA node 1), and the
cache moves them between the two as the access pattern changes.

The research question the fork exists to answer is *which eviction discipline makes the best
use of a small DRAM tier in front of a large CXL tier*. It is answered by running the same
cache 18 different ways — one `PaperPolicy` variant per design — and measuring them against
identical traces. Every hybrid build compiles all 18; the design is chosen at runtime, by the
`PaperPolicy` value handed to the constructor, and is then fixed for that cache's lifetime.

> This crate is a library and is not meant to be used directly by application code; the
> intended consumer is the separate `paper-server` crate. The benchmark harness is
> `paper-benchmark-cxl`.

## Requirements

- **Nightly Rust.** The tiered value type is `Box<[u8], Hybrid>`, which needs
  `allocator_api` and `clone_from_ref`. Every build command below uses `cargo +nightly`.
- **A two-node NUMA machine.** `numa_alloc::NODE_FAST = 0` and `NODE_SLOW = 1` are compiled
  in. The crate still builds and runs on a single-node box, but the "slow tier" will not be
  physically distinct from the fast one, so latency numbers are meaningless.
- **Linux.** `mbind(2)` and `/proc/self/numa_maps` are used directly.

## Quick start

```bash
cargo +nightly build --release --features lru_hybrid_cache
```

Enabling any one hybrid feature is all you need to get the hybrid API: `lru_hybrid_cache` pulls
in `key_value_pmem` and `hybrid_cache_common`, and `hybrid_cache_common` pulls in
`numa_jemalloc`. Naming those explicitly is harmless but redundant. The feature does **not**
select the design — that is a runtime argument, and any hybrid build hosts all 18.

```rust
use paper_cache::{PaperCache, CacheTierSize, TieredBuffer, Tier, PaperPolicy};

// 24 GB total cache, of which 4 GB is the DRAM fast tier, running segmented LRU.
let cache = PaperCache::<u64, TieredBuffer>::new(
    24_000_000_000,
    CacheTierSize::Gib(4),
    PaperPolicy::LruHybrid,
)?;

cache.set(1u64, b"hello world", None)?;

let value: Vec<u8> = cache.get(&1u64)?;
assert_eq!(cache.tier_of(&1u64), Some(Tier::Fast));

// Design-neutral counters: works whichever policy the cache was built with.
let stats = cache.hybrid_stats();
println!("promotions={} demotions={} evictions={}",
    stats.promotions, stats.demotions, stats.evictions);

// The fast/slow boundary can be moved at runtime.
cache.set_fast_tier_size(CacheTierSize::Gib(2))?;
```

## How it works

### Tier is a property of the value, not a separate cache

There is exactly **one** `PaperCache<K, TieredBuffer>`. `TieredBuffer` is a tagged union
recording where this object's bytes currently are:

```rust
pub enum TieredBuffer {
    Fast(Box<[u8]>),          // node-0 arenas, via the global allocator
    Slow(Box<[u8], Hybrid>),  // node-1 arenas, via numa_alloc::SlowObjects
}
```

A live object's bytes exist in **exactly one** tier at a time. Promotion and demotion replace
the `TieredBuffer` in place (`Object::set_data`), so a migration is a byte *move*, not a copy
into a second map. This is the opposite of the legacy `tiering/` module (see
[Legacy](#legacy-the-copy-based-tiering-manager)), which deliberately keeps a copy in both
tiers.

All 18 designs share **one** implementation. There are exactly two inherent
`impl<K, S> PaperCache<K, TieredBuffer, S>` blocks — the shared engine, and a second holding the
size-split design's three-scalar constructor — and both are gated only on `hybrid_cache_common`.
The per-design behaviour that remains is dispatched at runtime: one `match` over the cache's
`PaperPolicy` selects the admission rule, and `init_policy_stack` builds the corresponding
`PolicyStack`.

The hybrid features are therefore **not mutually exclusive** — enable any subset. `lib.rs`
carries a single `compile_error!`, rejecting `hashbrown_dram` together with
`global_hashtable_pmem`; it has nothing to do with the designs. (Earlier revisions gave each
design its own impl block, which forced mutual exclusion and 153 pairwise guards. Both are gone.)

### Who decides, and who moves the bytes

The API thread never touches eviction state. `get`/`set`/`del` update the object map and
push a `WorkerEvent` onto a channel; everything else happens on background workers.

```
API thread                PolicyWorker                    migration consumers
----------                ------------                    -------------------
set(k, v) ──WorkerEvent──> policy stack decides tiers
                           apply_tier_migrations()
                             demotions first, then       ──(k, tier)──> allocate
                             promotions                                 copy bytes
                                                                        swap pointer
```

- **`PolicyWorker`** owns the active policy stack and is the only thing that mutates it. It
  decides which keys should change tier and runs terminal evictions when `used_size()` exceeds
  `max_size`.
- **Ordering.** Demotions are applied before promotions. On the inline path
  (`MIGRATION_QUEUE_THREADS=0`) that is a physical barrier: the fast tier has given back space
  before anything moves into it. With the queue enabled it is an ordering of *enqueues* — per-key
  order is guaranteed by the hash sharding, but a promotion for one key can be physically applied
  before an unrelated key's demotion has run.
- **Watermarks.** Demotion triggers at `FAST_TIER_HIGH_WATERMARK` of the effective fast-tier
  budget and then drains in one pass down to `FAST_TIER_LOW_WATERMARK` (0.98 / 0.95), rather
  than trimming back to exactly the ceiling. Draining to the ceiling pinned the tier at 100%
  utilisation and made almost every pass a single-object migration batch.
- **`migration_queue`** (`worker/policy/mod.rs`) is a standing pool of consumer threads that
  perform the allocate-copy-swap off the worker. It has **one channel per consumer, indexed by
  key hash**, so two migrations for the same key can never be applied out of order. On by
  default with 2 consumers; `MIGRATION_QUEUE_THREADS=0` disables it and applies every
  migration inline on the worker.

An earlier approach, `parallel_migration`, fanned a single *batch* across a rayon pool. It is
compiled in but **disabled by default** (`PARALLEL_MIGRATION_THRESHOLD=0`) because it was
measured not to help: 99.4% of demotion volume arrives as single-object batches, so there is
nothing to fan out. `migration_queue` replaced it by decoupling from batch boundaries
entirely.

### Where memory physically goes

`src/numa_alloc.rs` gives each NUMA node its own jemalloc arenas whose extents are `mmap`'d
and then `mbind(MPOL_BIND | MPOL_F_STATIC_NODES)`'d **before** jemalloc hands them out, so
placement is decided by kernel policy at first fault rather than by whichever CPU happens to
touch the page first. The allocation hook fails closed: a failed bind `munmap`s and returns
null rather than silently yielding unbound memory, and anything that cannot reach a bound
arena is counted in `unbound_fallbacks` instead of passing unnoticed.

- `NumaAlloc<NODE_FAST>` is the crate's `#[global_allocator]` — so the fast tier and ordinary
  Rust allocation are the same thing.
- `numa_alloc::SlowObjects` (aliased crate-wide as `Hybrid`) backs `TieredBuffer::Slow`.

**This cannot cover the whole process.** jemalloc is built with `JEMALLOC_PREFIX=_rjem_` and
does not interpose `malloc`, so glibc's heap, bindgen'd C libraries and pthread stacks are
outside its reach. Pair with `numactl --membind=0` when the whole process must be bound.

Verify placement against `/proc/self/numa_maps` rather than the allocator's own counters —
the counters record what was *requested*, the kernel reports where pages actually are.

### Migration counters vs physical copies

The hybrid stats (`hybrid_stats()` and the `MIGSTATS` instrumentation) count **tier decisions made by the policy stack**, not physical byte copies.
The two are normally identical, but they are not the same quantity, and the distinction
matters when reading the numbers.

A migration is emitted whenever a stack changes an object's tier tag. The worker then asks the
migrate closure to move the bytes — and the closure returns `None`, skipping the copy, when the
value is **already** in the requested tier. That happens because the API thread chooses a
placement of its own: `PaperCache::set()` calls the free function
`hybrid_policy::admission_tier(policy, ...)` — one runtime `match` carrying each design's
admission rule — and builds the value with `TieredBuffer::new_fast` or `new_slow` accordingly, so an object can
already be where the stack is about to say it should be.

Consequences when interpreting stats:

- **Counters lead the copies.** With the migration queue enabled (the default) the copies are
  applied asynchronously by the consumer pool, so a mid-run snapshot reports decisions that
  have been made but not yet physically performed. They converge once the queue drains. There
  is no public flush — `MigrationQueue` is crate-internal (`mod worker` is private), and the only
  call to its `flush` is `#[cfg(test)]`-gated so test assertions on buffer contents stay
  deterministic. Set `MIGRATION_QUEUE_THREADS=0` if you need
  a snapshot with no in-flight window at all.
- **The four tier gauges are polled, not live.** `fast_objects`/`slow_objects`/
  `fast_bytes_used`/`slow_bytes_used` are republished by `PolicyWorker::refresh_tier_gauges`
  once per event-loop pass, so they are up to one polling interval stale. The three counters
  (`promotions`/`demotions`/`evictions`) are monotonic totals since creation or last `wipe()`.
- **A persistent gap is a defect signal, not noise.** If the decision count exceeds the copies
  performed, some stack is emitting migrations for objects already in the target tier — wasted
  work, and silent. `LfuHybridStack` did exactly this on every latched admission (445,465,067
  migrations against ~448M sets on cluster12, roughly 99% of its reported demotions) because
  `admission_tier` already returned `Tier::Slow` and built the bytes in PMEM while the stack
  emitted a `Tier::Slow` migration anyway. Any `demotions` figure for `lfu_hybrid_cache`
  collected before that fix is inflated by that factor and is not comparable with other designs
  or with later runs.
- **Declined migrations are legitimate, not errors.** `TwoQHybridStack` reaches this case by
  design under a lookaside workload: `admission_tier` returns `Fast` for a re-set (correct — the
  key is now most-recently-used), so the value is already in DRAM by the time
  `touch_main_fast` emits its promotion. The decision is real and is counted; the copy is
  correctly skipped.

## Choosing a design

Seventeen of the 18 are built with the one shared constructor,
`new(max_size, fast_tier_size, policy)`, where `policy` is the design's `PaperPolicy` variant and
carries that design's tuning knob in its payload — `PaperPolicy::TwoQHybrid(k_in)`,
`PaperPolicy::LruLfuHybrid(promote_k)`, and so on. Four variants take no payload.

The eighteenth, the size-split design, has its own constructor `new_sized(...)`: it needs three
sizing scalars rather than one, and it takes no policy argument (it hardcodes
`PaperPolicy::LruSizedHybrid`). Passing `LruSizedHybrid` to `new()` returns
`CacheError::InvalidPolicy`.

`PaperPolicy` also round-trips through `FromStr`/`Display` (`"lru-hybrid"`, `"2q-hybrid-0.2"`,
`"s3-fifo-ghost-lazy-demotion-fast-admission-hybrid-0.1"`, ...) and deserializes via serde, so a
design and its parameter can come from a config file or a command line with no rebuild.

### Base designs

| Feature (re-export/test gate) | Policy value | Fast/slow boundary |
|---|---|---|
| `lru_hybrid_cache` | `PaperPolicy::LruHybrid` | One LRU queue, segmented by byte budget |
| `lfu_hybrid_cache` | `PaperPolicy::LfuHybrid` | Frequency-ordered, admission gated on capacity |
| `fifo_hybrid_cache` | `PaperPolicy::FifoHybrid` | Insertion order; no promotion at all |
| `lru_sized_hybrid_cache` | `PaperPolicy::LruSizedHybrid` — via `new_sized(max_size, small_fast_tier_size, large_fast_tier_size, size_threshold)` | LRU, with each tier's bookkeeping split small/large by object size |
| `lru_lfu_hybrid_cache` | `PaperPolicy::LruLfuHybrid(promote_k)` | LRU fast tier, LFU slow tier — promotion is a fixed access-count threshold |

### 2Q family — a one-access FIFO queue feeding a segmented main queue

All four carry `k_in` in their policy payload — e.g. `PaperPolicy::TwoQHybrid(k_in)` — where
`k_in * max_size` is the FIFO queue's byte budget. `k_in` must lie in `0.0..=1.0`.

| Feature | Builds on | Change |
|---|---|---|
| `two_q_hybrid_cache` | -- | Baseline: FIFO queue in the **slow** tier, so every `set()` is a real PMEM write |
| `two_q_fast_admission_hybrid_cache` | baseline | FIFO queue moved to the **fast** tier; its budget is carved out of `fast_tier_size` |
| `two_q_fast_admission_reprieve_hybrid_cache` | fast admission | A key aging out of the FIFO queue is spliced into the slow tier instead of evicted |
| `two_q_ghost_hybrid_cache` | baseline | A bare-key ghost queue, so re-admission skips the FIFO queue |

### S3-FIFO family — lazy, reference-bit-gated promotion

All nine carry `one_access_ratio` in their policy payload — e.g.
`PaperPolicy::S3FifoHybrid(one_access_ratio)` — validated into `0.0..1.0` for the six designs that size a main queue at `(1 - one_access_ratio) * max_size` (the plain `s3-fifo` stack and the five non-reprieve hybrids), where a ratio of 1 would leave that queue zero bytes and stall eviction; `0.0..=1.0` for the four reprieve designs, which derive no budget from `1 - ratio` and so cannot be starved by it. Rows are in the order
the designs were built, each described against the one above it — note that the later ones
*remove* as much as they add.

| Feature | Change from the row above |
|---|---|
| `s3_fifo_hybrid_cache` | Baseline: CLOCK-style lazy promotion; one-access queue in the slow tier |
| `s3_fifo_ghost_hybrid_cache` | Bare-key ghost queue |
| `s3_fifo_ghost_lazy_demotion_hybrid_cache` | Demotion is reference-bit gated too, not just eviction |
| `s3_fifo_ghost_lazy_demotion_fast_admission_hybrid_cache` | One-access queue moves to the fast tier (DRAM admission) |
| `s3_fifo_ghost_lazy_demotion_fast_admission_midpoint_hybrid_cache` | A sampled checkpoint halfway through the slow segment |
| `s3_fifo_lazy_demotion_fast_admission_midpoint_reprieve_hybrid_cache` | Drops the ghost queue; aged-out keys are reprieved into the slow tier |
| `s3_fifo_lazy_demotion_fast_admission_reprieve_hybrid_cache` | Drops the midpoint checkpoint — measured bit-identical to no check |
| `s3_fifo_lazy_demotion_reprieve_hybrid_cache` | Returns admission to the **slow** tier, keeping reprieve, so the splice moves no bytes at all |
| `s3_fifo_lazy_demotion_fast_admission_split_slow_reprieve_hybrid_cache` | *(branches from the midpoint-reprieve row, not the one above)* Replaces the sampled midpoint cursor with a real two-segment slow tier, so every crossing object's bit is checked |

The two midpoint variants are kept as recorded negative results: an approximate sampled cursor
and a real segment boundary both measured bit-identical hit rates to having no check at all,
because terminal eviction only ever removes the slow tier's tail, where the reference bit is
already honoured.

## API surface

Shared by every hybrid design (`impl<K, S> PaperCache<K, TieredBuffer, S>`):

| Method | Notes |
|---|---|
| `get(&key) -> Result<Vec<u8>>` | May trigger a promotion decision |
| `set(key, &[u8], ttl: Option<u32>)` | Placement chosen by `hybrid_policy::admission_tier` for the active policy |
| `del(&key)`, `has(&key)`, `size(&key)` | |
| `peek(&key) -> Result<Arc<TieredBuffer>>` | No access recorded, so no promotion |
| `ttl(&key, Option<u32>)` | |
| `tier_of(&key) -> Option<Tier>` | Where the bytes are right now |
| `hybrid_stats() -> HybridStats` | The only stats accessor: 3 counters + 4 tier gauges + 8 size-split gauges (the latter zero unless running `LruSizedHybrid`) |
| `fast_tier_size()`, `set_fast_tier_size(CacheTierSize)` | Boundary is movable at runtime |
| `large_fast_tier_size()`, `set_large_fast_tier_size()`, `size_threshold()`, `set_size_threshold()` | Present on every hybrid cache; take effect only under `LruSizedHybrid` |
| `resize(max_size)`, `wipe()`, `status()`, `version()` | |

`CacheTierSize` is `Bytes`/`Mb`/`Gb`, decimal (1 MB = 1,000,000 bytes).

## Environment variables

| Variable | Default | Effect |
|---|---|---|
| `MIGRATION_QUEUE_THREADS` | `2` | Migration consumer count. `0` disables the queue and applies migrations inline on the worker. |
| `PARALLEL_MIGRATION_THRESHOLD` | `0` (off) | Batch size at or above which batch fan-out engages. Off because it was measured not to pay; see `parallel_migration`. |
| `PARALLEL_MIGRATION_THREADS` | `4` | Pool size if the above is enabled. |
| `FAST_TIER_HIGH_WATERMARK` | `0.98` | Fast-tier fraction at which demotion triggers. |
| `FAST_TIER_LOW_WATERMARK` | `0.95` | Fraction a triggered demotion pass drains down to. Set both to `1.0` to restore drain-to-the-ceiling. |
| `NUMA_ARENAS_PER_NODE` | `8` | jemalloc arenas per node (clamped to 32). Swept on cluster12: a single arena costs 5% of SET latency at one client and 27% at sixteen, while 8→32 buys 1–2%, inside the run-to-run spread. |
| `PAPER_NUMA_SLOW_TCACHE` | off | Per-thread cache for slow-tier allocations. Correct but measured not worth enabling. |
| `DRAM_OVERHEAD_RESIDENT_FACTOR` | `1.12` | Recalibrates the per-object DRAM overhead reservation. Recalibrate when the workload or allocator changes. |
| `PAPER_CACHE_EVICTION_STACK_CAPACITY` | — | Pre-sizes the eviction stack's backing collections. |

## Testing

One integration-test file per design, each gated on its own feature:

```bash
cargo +nightly test --release --test lru_hybrid_cache_integration --features lru_hybrid_cache
```

Some reproductions are `#[ignore]`d because they take minutes and allocate tens of GB:

```bash
cargo +nightly test --release --test lru_hybrid_cache_integration --features lru_hybrid_cache \
  repro_real_dram_usage_at_scale -- --ignored --nocapture
```

Test builds drain the migration queue synchronously after each batch, so assertions on buffer
contents are deterministic while still exercising the real queue path.

## Benchmarking

There is no longer a `scripts/` directory in this repo. The old
`run_hybrid_benchmark_matrix.sh` rebuilt `paper-benchmark-cxl` once per design, rewriting the
`features=[...]` line in its `Cargo.toml` between runs — a premise the unification removed. A
single build now hosts all 18 designs, so a sweep is a loop over `PaperPolicy` values (or over
their string forms, via `FromStr`) with no rebuild between cells.

`paper_cache::jemalloc_stats()` samples allocated/active/resident/mapped/retained at peak,
which `stats_print:true` cannot do — that runs from an atexit handler, long after the cache
has been dropped.

## Further reading

| Document | Covers |
|---|---|
| `FEATURE_FLAGS.md` | Every feature flag, including the placement flags not described here |
| `HYBRID_CACHES.md` | How the stacks decide, and how a decision becomes a byte move |
| `LRU_HYBRID_CACHE.md` | One design end to end, in the most detail |
| `CLAUDE.md` | Code structure, plus a log of past investigations and their outcomes |

Design rationale generally lives in module doc comments rather than in these files — the policy
stacks in `src/worker/policy/policy_stack/` each carry their algorithm's derivation at the top.

## Legacy: the copy-based tiering manager

`src/tiering/` implements an older, unrelated design, still reachable behind `tiering` /
`multitiering` (or `enable_tiering_manager` together with `key_value_pmem` -- on its own that
feature implies nothing and the module is not compiled at all). It is a *hotness-threshold, copy-based*
scheme: PMEM is the permanent source of truth, and an object accessed at least
`hotness_threshold` times gets a **second, physical copy** placed in a DRAM side-cache, kept
consistent on write and dropped on demotion. Under `hashtable_tiering` it adds a third,
zero-copy "warm" state holding a CXL reference instead of a copy.

This is the opposite data-movement model from the hybrid designs above, which keep exactly one
copy and move it. The two are not interchangeable, and the hybrid designs do not use any of
this module. It is documented here only so the feature flags are not mysterious.

## License

AGPL-3.0. See `LICENSE`.
