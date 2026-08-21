# paper-cache (DRAM/CXL tiering fork)

PaperCache is an in-memory cache that supports switching between eviction policies at
runtime. This fork adds **two-tier memory placement**: every object's bytes live either in
DRAM (the *fast* tier, NUMA node 0) or in PMEM/CXL (the *slow* tier, NUMA node 1), and the
cache moves them between the two as the access pattern changes.

The research question the fork exists to answer is *which eviction discipline makes the best
use of a small DRAM tier in front of a large CXL tier*. It is answered by building the same
cache 18 different ways — one Cargo feature per design — and measuring them against identical
traces. Only one design is compiled into any given build.

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

Selecting a hybrid design is all you need: `lru_hybrid_cache` pulls in `key_value_pmem` and
`hybrid_cache_common`, and `hybrid_cache_common` pulls in `numa_jemalloc`. Naming those
explicitly is harmless but redundant.

```rust
use paper_cache::{PaperCache, CacheTierSize, TieredBuffer, Tier};

// 24 GB total cache, of which 4 GB is the DRAM fast tier.
let cache = PaperCache::<u64, TieredBuffer>::new(
    24_000_000_000,
    CacheTierSize::Gb(4),
)?;

cache.set(1u64, b"hello world", None)?;

let value: Vec<u8> = cache.get(&1u64)?;
assert_eq!(cache.tier_of(&1u64), Some(Tier::Fast));

// Feature-neutral counters: works whichever design was compiled in.
let stats = cache.hybrid_stats();
println!("promotions={} demotions={} evictions={}",
    stats.promotions, stats.demotions, stats.evictions);

// The fast/slow boundary can be moved at runtime.
cache.set_fast_tier_size(CacheTierSize::Gb(2))?;
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

Because each design defines its own inherent `impl<K, S> PaperCache<K, TieredBuffer, S>`
block, and two such blocks cannot coexist for one concrete type, **the hybrid features are
mutually exclusive**. `lib.rs` names all 153 conflicting pairs -- every pair of the 18 designs
-- in its own `compile_error!` guard, so enabling two surfaces as a sentence naming both rather
than as a duplicate-definition error deep in a generic impl.

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
- **Ordering.** Migrations are applied demotions-before-promotions, so the fast tier has
  already given back space before anything tries to move into it.
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

The hybrid stats (`hybrid_stats()`, the per-design `*_hybrid_stats()`, and the `MIGSTATS`
instrumentation) count **tier decisions made by the policy stack**, not physical byte copies.
The two are normally identical, but they are not the same quantity, and the distinction
matters when reading the numbers.

A migration is emitted whenever a stack changes an object's tier tag. The worker then asks the
migrate closure to move the bytes — and the closure returns `None`, skipping the copy, when the
value is **already** in the requested tier. That happens because the API thread chooses a
placement of its own: `PaperCache::set()` consults each design's `HybridPolicy::admission_tier`
and builds the value with `TieredBuffer::new_fast` or `new_slow` accordingly, so an object can
already be where the stack is about to say it should be.

Consequences when interpreting stats:

- **Counters lead the copies.** With the migration queue enabled (the default) the copies are
  applied asynchronously by the consumer pool, so a mid-run snapshot reports decisions that
  have been made but not yet physically performed. They converge once the queue drains. There
  is no public flush — `MigrationQueue::flush` exists but is `#[cfg(test)]`-only, so that tests
  can assert on buffer contents deterministically. Set `MIGRATION_QUEUE_THREADS=0` if you need
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

All 18 take `max_size` (total bytes across both tiers) followed by the fast tier's capacity --
a single `fast_tier_size` for every design except `lru_sized_hybrid_cache`, which splits it in
two. Any further argument is that design's own tuning knob.

### Base designs

| Feature | Constructor | Fast/slow boundary |
|---|---|---|
| `lru_hybrid_cache` | `new(max_size, fast_tier_size)` | One LRU queue, segmented by byte budget |
| `lfu_hybrid_cache` | `new(max_size, fast_tier_size)` | Frequency-ordered, admission gated on capacity |
| `fifo_hybrid_cache` | `new(max_size, fast_tier_size)` | Insertion order; no promotion at all |
| `lru_sized_hybrid_cache` | `new(max_size, small_fast, large_fast, size_threshold)` | LRU, with each tier's bookkeeping split small/large by object size |
| `lru_lfu_hybrid_cache` | `new(max_size, fast_tier_size, promote_k)` | LRU fast tier, LFU slow tier — promotion is a fixed access-count threshold |

### 2Q family — a one-access FIFO queue feeding a segmented main queue

All take `new(max_size, fast_tier_size, k_in)`, where `k_in * max_size` is the FIFO queue's
byte budget.

| Feature | Builds on | Change |
|---|---|---|
| `two_q_hybrid_cache` | -- | Baseline: FIFO queue in the **slow** tier, so every `set()` is a real PMEM write |
| `two_q_fast_admission_hybrid_cache` | baseline | FIFO queue moved to the **fast** tier; its budget is carved out of `fast_tier_size` |
| `two_q_fast_admission_reprieve_hybrid_cache` | fast admission | A key aging out of the FIFO queue is spliced into the slow tier instead of evicted |
| `two_q_ghost_hybrid_cache` | baseline | A bare-key ghost queue, so re-admission skips the FIFO queue |

### S3-FIFO family — lazy, reference-bit-gated promotion

All take `new(max_size, fast_tier_size, one_access_ratio)`. Rows are in the order the designs
were built, each described against the one above it — note that the later ones *remove* as
much as they add.

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
| `s3_fifo_lazy_demotion_fast_admission_split_slow_reprieve_hybrid_cache` | Replaces the sampled midpoint with a real two-segment slow tier |

The two midpoint variants are kept as recorded negative results: an approximate sampled cursor
and a real segment boundary both measured bit-identical hit rates to having no check at all,
because terminal eviction only ever removes the slow tier's tail, where the reference bit is
already honoured.

## API surface

Shared by every hybrid design (`impl<K, S> PaperCache<K, TieredBuffer, S>`):

| Method | Notes |
|---|---|
| `get(&key) -> Result<Vec<u8>>` | May trigger a promotion decision |
| `set(key, &[u8], ttl: Option<u32>)` | Placement chosen by the design's `admission_tier` |
| `del(&key)`, `has(&key)`, `size(&key)` | |
| `peek(&key) -> Result<Arc<TieredBuffer>>` | No access recorded, so no promotion |
| `ttl(&key, Option<u32>)` | |
| `tier_of(&key) -> Option<Tier>` | Where the bytes are right now |
| `hybrid_stats() -> HybridStats` | Feature-neutral; 3 counters + 4 gauges |
| `<design>_hybrid_stats()` | Design-specific struct, with any extra gauges |
| `fast_tier_size()`, `set_fast_tier_size(CacheTierSize)` | Boundary is movable at runtime |
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
| `DRAM_OVERHEAD_RESIDENT_FACTOR` | per-config constant | Recalibrates the per-object DRAM overhead reservation. Recalibrate when the workload or allocator changes. |
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

`scripts/run_hybrid_benchmark_matrix.sh` builds `paper-benchmark-cxl` once per (paper-cache
feature, benchmark feature) pair and runs it against every trace in `$TRACES_DIR`, capturing
GET/SET latency per run. It rewrites the `features=[...]` line in the benchmark's `Cargo.toml`
in place, so it assumes that crate is path-pointed at a local checkout of this one. Paths at
the top of the script are absolute and need editing for another host.

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

`src/tiering/` implements an older, unrelated design, still reachable behind
`enable_tiering_manager` / `tiering` / `multitiering`. It is a *hotness-threshold, copy-based*
scheme: PMEM is the permanent source of truth, and an object accessed at least
`hotness_threshold` times gets a **second, physical copy** placed in a DRAM side-cache, kept
consistent on write and dropped on demotion. Under `hashtable_tiering` it adds a third,
zero-copy "warm" state holding a CXL reference instead of a copy.

This is the opposite data-movement model from the hybrid designs above, which keep exactly one
copy and move it. The two are not interchangeable, and the hybrid designs do not use any of
this module. It is documented here only so the feature flags are not mysterious.

## License

AGPL-3.0. See `LICENSE`.
