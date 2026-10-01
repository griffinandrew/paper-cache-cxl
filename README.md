# paper-cache (DRAM/CXL tiering fork)

PaperCache is an in-memory cache with a choice of eviction policies, one per cache and fixed when
it is built. This fork adds **two-tier memory placement**: every object's bytes live either in
DRAM (the *fast* tier, NUMA node 0) or in PMEM/CXL (the *slow* tier, NUMA node 1), and the
cache moves them between the two as the access pattern changes.

The research question the fork exists to answer is *which eviction discipline makes the best
use of a small DRAM tier in front of a large CXL tier*. It is answered by running the same
cache 23 different ways — one `PaperPolicy` variant per design — and measuring them against
identical traces. Every hybrid build compiles all 23; the design is chosen at runtime, by the
`PaperPolicy` value handed to the constructor, and is then fixed for that cache's lifetime.

> This crate is a library and is not meant to be used directly by application code; the
> intended consumer is the separate `paper-server` crate. The benchmark harness is
> `paper-benchmark-cxl`.

## Requirements

- **Nightly Rust.** The tiered value type is `Box<[u8], Hybrid>`, which needs
  `allocator_api` (and `btreemap_alloc`). Every build command below uses `cargo +nightly`.
- **A two-node NUMA machine.** `numa_alloc::NODE_FAST = 0` and `NODE_SLOW = 1` are compiled
  in. The crate still builds and runs on a single-node box, but the "slow tier" will not be
  physically distinct from the fast one, so latency numbers are meaningless.
- **Linux.** `mbind(2)` and `/proc/self/numa_maps` are used directly.

## Quick start

```bash
cargo +nightly build --release --features lru_compact_hybrid_cache
```

Enabling any one hybrid feature is all you need to get the hybrid API: `lru_compact_hybrid_cache` pulls
in `key_value_pmem` and `hybrid_cache_common`, and `hybrid_cache_common` pulls in
`numa_jemalloc`. Naming those explicitly is harmless but redundant. The feature does **not**
select the design — that is a runtime argument, and any hybrid build hosts all 23.

```rust
use paper_cache::{PaperCache, CacheTierSize, TieredBuffer, Tier, PaperPolicy};

// 24 GB total cache, of which 4 GB is the DRAM fast tier, running segmented LRU.
let cache = PaperCache::<u64, TieredBuffer>::new(
    24_000_000_000,
    CacheTierSize::Gib(4),
    PaperPolicy::LruCompactHybrid,
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

All 23 designs share **one** implementation. There are exactly two inherent
`impl<K, S> PaperCache<K, TieredBuffer, S>` blocks — the shared engine, and a second holding the
size-split design's three-scalar constructor — and both are gated only on `hybrid_cache_common`.
The per-design behaviour that remains is dispatched at runtime: one `match` over the cache's
`PaperPolicy` selects the admission rule, and `init_policy_stack` builds the corresponding
`PolicyStack`.

The hybrid features are therefore **not mutually exclusive** — enable any subset. (Earlier
revisions gave each design its own impl block, which forced mutual exclusion and 153 pairwise
guards. Both are gone.)

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
  before an unrelated key's demotion has run. The split drops every entry that a later entry for
  the other tier supersedes (`split_tier_migrations`), so it can never reverse a key's own
  intents: a key promoted and demoted again in one drain used to come out demote-first and stay
  in DRAM while the stack counted it slow.
- **Drain target.** `settle_fast_tier` holds the fast tier at `FAST_TIER_DRAIN_TARGET` (0.95) of
  its effective budget: it demotes whenever usage is above that level and stops the moment it is
  back at it. One threshold, not a band — a settle moves only what the event that triggered it
  displaced. The 5% margin is headroom for the DRAM a burst of `set()`s puts down before the
  policy worker sees it, and for demotions still queued behind `migration_queue`; it is also the
  room under the byte gate's close level (`B - S`). It was 0.98 (2%) until the default moved to
  give bursts more room; set `FAST_TIER_DRAIN_TARGET=0.98` to reproduce earlier results.
- **Eviction watermark.** Capacity eviction is held at `EVICTION_HIGH_WATERMARK` (0.98) of
  `max_size`, cache-wide: evictions start once `used_size` passes 98% of the cap and drain to the
  same level, one object at a time (the fast tier's shape: one threshold, not a band). It is read
  against the current `max_size` on every pass, so `resize` moves it, and it is the same for
  tiered and flat caches and for both object stores. It is separate from the drain target above
  (which holds the fast tier, not the cache) and from the byte gate (which bounds the fast tier's
  bytes). Before this default the cache settled exactly at `max_size`; set
  `EVICTION_HIGH_WATERMARK=1.0` to reproduce that (published results predate the change).
  A `set` whose accounted size (the base size plus the per-object overhead, as `used_size`
  counts it) is over that level is refused with `ExceedingValueSize`: the cache could not
  hold it, and accepting it would evict everything else and then the value itself. With
  `EVICTION_HIGH_WATERMARK=1.0` the refusal is the old one, the base size against `max_size`.
- **`migration_queue`** (`worker/policy/mod.rs`) is a standing pool of consumer threads that
  perform the allocate-copy-swap off the worker. It has **one channel per consumer, indexed by
  key hash**, so two migrations for the same key can never be applied out of order. On by
  default with 2 consumers; `MIGRATION_QUEUE_THREADS=0` disables it and applies every
  migration inline on the worker.

An earlier approach, `parallel_migration`, fanned a single *batch* across a rayon pool. It was
measured not to help -- 99.4% of demotion volume arrives as single-object batches, so there is
nothing to fan out -- and has been removed (R1). `migration_queue` replaced it by decoupling
from batch boundaries entirely.

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

The hybrid stats (`hybrid_stats()` and the `MIGSTATS` instrumentation) count **tier decisions made by the policy stack**, not physical byte copies. Both are per cache: each cache starts at zero, and the `MIGSTATS` stderr lines are its own (the migration queue's depth, backlog and dispositions, the batch-size histograms and the reconcile counters are in `HybridStats` too).
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
  work, and silent. `LfuCompactHybridStack` did exactly this on every latched admission (445,465,067
  migrations against ~448M sets on cluster12, roughly 99% of its reported demotions) because
  `admission_tier` already returned `Tier::Slow` and built the bytes in PMEM while the stack
  emitted a `Tier::Slow` migration anyway. Any `demotions` figure for `lfu_compact_hybrid_cache`
  collected before that fix is inflated by that factor and is not comparable with other designs
  or with later runs.
- **Declined migrations are legitimate, not errors.** `TwoQCompactHybridStack` reaches this case by
  design under a lookaside workload: `admission_tier` returns `Fast` for a re-set (correct — the
  key is now most-recently-used), so the value is already in DRAM by the time
  `touch_main_fast` emits its promotion. The decision is real and is counted; the copy is
  correctly skipped.

## Choosing a design

Twenty-two of the 23 are built with the one shared constructor,
`new(max_size, fast_tier_size, policy)`, where `policy` is the design's `PaperPolicy` variant and
carries that design's tuning knob in its payload — `PaperPolicy::TwoQCompactHybrid(k_in)`,
`PaperPolicy::LruLfuCompactHybrid(promote_k)`, and so on. Four variants take no payload.

The size-split design has its own constructor, `new_sized_compact(...)`: it needs three
sizing scalars rather than one, and it takes no policy argument (it hardcodes
`PaperPolicy::LruSizedCompactHybrid`). Passing `LruSizedCompactHybrid` to `new()` returns
`CacheError::InvalidPolicy`.

`PaperPolicy` also round-trips through `FromStr`/`Display` (`"lru-compact-hybrid"`, `"2q-compact-hybrid-0.2"`,
`"s3-fifo-ghost-lazy-demotion-fast-admission-compact-hybrid-0.1"`, ...) and deserializes via serde, so a
design and its parameter can come from a config file or a command line with no rebuild.

### Base designs

| Feature (re-export/test gate) | Policy value | Fast/slow boundary |
|---|---|---|
| `lru_compact_hybrid_cache` | `PaperPolicy::LruCompactHybrid` | One LRU queue, segmented by byte budget |
| `lfu_compact_hybrid_cache` | `PaperPolicy::LfuCompactHybrid` | Frequency-ordered, admission gated on capacity |
| `fifo_compact_hybrid_cache` | `PaperPolicy::FifoCompactHybrid` | Insertion order; no promotion at all |
| `lru_sized_compact_hybrid_cache` | `PaperPolicy::LruSizedCompactHybrid` — via `new_sized_compact(max_size, small_fast_tier_size, large_fast_tier_size, size_threshold)` | LRU, with each tier's bookkeeping split small/large by object size |
| `lru_lfu_compact_hybrid_cache` | `PaperPolicy::LruLfuCompactHybrid(promote_k)` | LRU fast tier, LFU slow tier — promotion is a fixed access-count threshold |

### 2Q family — a one-access FIFO queue feeding a segmented main queue

All three carry `k_in` in their policy payload — e.g. `PaperPolicy::TwoQCompactHybrid(k_in)` — where
`k_in * max_size` is the FIFO queue's byte budget. `k_in` must lie in `0.0..=1.0`.

| Feature | Builds on | Change |
|---|---|---|
| `two_q_compact_hybrid_cache` | -- | Baseline: FIFO queue in the **slow** tier, so every `set()` is a real PMEM write |
| `two_q_fast_admission_reprieve_compact_hybrid_cache` | baseline | FIFO queue moved to the **fast** tier (its budget is carved out of `fast_tier_size`); a key aging out of it is spliced into the slow tier instead of evicted |
| `two_q_ghost_compact_hybrid_cache` | baseline | A bare-key ghost queue, so re-admission skips the FIFO queue |

### S3-FIFO family — lazy, reference-bit-gated promotion

All nine carry `one_access_ratio` in their policy payload — e.g.
`PaperPolicy::S3FifoCompactHybrid(one_access_ratio)` — validated into `0.0..1.0` for the six designs that size a main queue at `(1 - one_access_ratio) * max_size` (the plain `s3-fifo` stack and the five non-reprieve hybrids), where a ratio of 1 would leave that queue zero bytes and stall eviction; `0.0..=1.0` for the four reprieve designs, which derive no budget from `1 - ratio` and so cannot be starved by it. Rows are in the order
the designs were built, each described against the one above it — note that the later ones
*remove* as much as they add.

| Feature | Change from the row above |
|---|---|
| `s3_fifo_compact_hybrid_cache` | Baseline: CLOCK-style lazy promotion; one-access queue in the slow tier |
| `s3_fifo_ghost_compact_hybrid_cache` | Bare-key ghost queue |
| `s3_fifo_ghost_lazy_demotion_compact_hybrid_cache` | Demotion is reference-bit gated too, not just eviction |
| `s3_fifo_ghost_lazy_demotion_fast_admission_compact_hybrid_cache` | One-access queue moves to the fast tier (DRAM admission) |
| `s3_fifo_ghost_lazy_demotion_fast_admission_midpoint_compact_hybrid_cache` | A sampled checkpoint halfway through the slow segment |
| `s3_fifo_lazy_demotion_fast_admission_midpoint_reprieve_compact_hybrid_cache` | Drops the ghost queue; aged-out keys are reprieved into the slow tier |
| `s3_fifo_lazy_demotion_fast_admission_reprieve_compact_hybrid_cache` | Drops the midpoint checkpoint — measured bit-identical to no check |
| `s3_fifo_lazy_demotion_reprieve_compact_hybrid_cache` | Returns admission to the **slow** tier, keeping reprieve, so the splice moves no bytes at all |
| `s3_fifo_lazy_demotion_fast_admission_split_slow_reprieve_compact_hybrid_cache` | *(branches from the midpoint-reprieve row, not the one above)* Replaces the sampled midpoint cursor with a real two-segment slow tier, so every crossing object's bit is checked |

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
| `reserve_set(key, len, ttl, deadline: Instant) -> Result<SetPermit>` | Admits a set BEFORE its value is read (a server's SET): the size checks, the metadata cap, the tier, the byte gate -- waiting for room at most until `deadline` (`FastTierStalled`; `MetadataOverflow` in the metadata lane). `permit.fill()` allocates the value in its tier, uninitialized, and returns a `PendingSet` to write in place (`read_exact_from(&mut reader)`, `write`, or `unfilled` + `advance`); `commit()` publishes it exactly as `set` does; `set_ttl` gives a TTL read after the value. Dropping a permit or a pending set abandons the set: reservation released, P refunded once, nothing inserted. `set` is these steps |
| `register_setter() -> SetterGuard`, `live_setters()` | Counts a live setter (an accepted connection) into the byte gate's near band until the guard drops |
| `del(&key)`, `has(&key)`, `size(&key)` | |
| `peek(&key) -> Result<Arc<TieredBuffer>>` | No access recorded, so no promotion |
| `ttl(&key, Option<u32>)` | |
| `tier_of(&key) -> Option<Tier>` | Where the bytes are right now |
| `hybrid_stats() -> HybridStats` | The only stats accessor: 3 counters + 4 tier gauges + 8 size-split gauges (the latter zero unless running `LruSizedCompactHybrid`) |
| `fast_tier_size()`, `set_fast_tier_size(CacheTierSize)` | Boundary is movable at runtime |
| `large_fast_tier_size()`, `set_large_fast_tier_size()`, `size_threshold()`, `set_size_threshold()` | Present on every hybrid cache; take effect only under `LruSizedCompactHybrid` |
| `resize(max_size)`, `wipe()`, `status()`, `version()` | |

`CacheTierSize` is `Bytes`/`Mb`/`Gb`, decimal (1 MB = 1,000,000 bytes).

## Environment variables

| Variable | Default | Effect |
|---|---|---|
| `MIGRATION_QUEUE_THREADS` | `2` | Migration consumer count. `0` disables the queue and applies migrations inline on the worker. |
| `FAST_TIER_DRAIN_TARGET` | `0.95` | Fraction of the effective fast-tier budget the tier is continuously held at (`0.98` before the default moved). `1.0` leaves no burst headroom. |
| `EVICTION_HIGH_WATERMARK` | `0.98` | Fraction of `max_size` above which capacity eviction starts, in `(0, 1]`. `1.0` evicts only past `max_size`, the behaviour before the 0.98 default. It is also the level above which a `set` is refused (`ExceedingValueSize`), by the object's accounted size; at `1.0`, by its base size against `max_size`, as before. |
| `EVICTION_LOW_WATERMARK` | the high mark | Fraction of `max_size` an armed eviction pass drains to, clamped to at most the high mark. Unset, it follows the high mark: one threshold. Set below it, the pass is a band and evicts in bursts. |
| `PAPER_GATE_MODE` | `block` | The fast-tier byte gate: `block` holds a fast set to the tier's budget and waits for demotions to free room; `off` admits fast sets ungated (the metadata cap and structural slow placement still apply). |
| `PAPER_GATE_STALL_WINDOW_MS` | `2000` | How long a waiting set waits with nothing freed before it acts (`PAPER_GATE_ON_STALL`), in ms. `0` never waits: a set that would wait acts at once. |
| `PAPER_GATE_ON_STALL` | `error` | What a set does when the watchdog fires: `error` (`FastTierStalled`), `divert` (built in the slow tier, healed on its first slow hit) or `admit_over` (admitted over the budget). |
| `PAPER_GATE_ON_METADATA_OVERFLOW` | `error` | What a new key whose metadata would not fit the fast tier gets: `error` (`MetadataOverflow`) or `evict_to_fit` (wait while the worker evicts the policy's victims for it). |
| `PAPER_GATE_METADATA_MODEL` | `measured` | The figure taken off the fast tier's budget for the cache's DRAM metadata: `measured` (counted from the structures) or `per_object` (the stacks' per-object reservation). |
| `PAPER_GATE_NEAR_FRAC` | `0.01` | The near band, a fraction of the effective budget in `[0, 1)`, below the close level above which a fast set takes the exact path. With `block`, `FAST_TIER_DRAIN_TARGET + PAPER_GATE_NEAR_FRAC` must stay under 1. |
| `PAPER_GATE_POLL_INTERVAL_US` | `200` | How often a waiting set re-checks when nothing wakes it, in microseconds (at least 1). |
| `PAPER_GATE_METADATA_FLOOR_BYTES` | `0` | Bytes of the fast tier kept for values: the metadata cap is the tier's budget less this. |
| `PAPER_GATE_SLACK_BYTES` | `0` | Bytes the fast tier may hold beyond its budget: the close level is the budget plus this. |
| `PAPER_GATE_CONCURRENCY_HINT`, `PAPER_GATE_VALUE_HINT_BYTES` | `0`, `0` | Concurrent setters and a typical value size: they widen the near band to their product when that is wider. The setters a server registers (`register_setter`) count with the hint, so it need not know its connections up front; with no value size they widen nothing. |
| `NUMA_ARENAS_PER_NODE` | `8` | jemalloc arenas per node (clamped to 32). Swept on cluster12: a single arena costs 5% of SET latency at one client and 27% at sixteen, while 8→32 buys 1–2%, inside the run-to-run spread. |
| `PAPER_CACHE_EVICTION_STACK_CAPACITY` | — | Pre-sizes the eviction stack's backing collections. |

The `PAPER_GATE_*` variables are the tiered cache's `GateConfig` from the environment, read once
per process. They configure a cache built **without** a `GateConfig` (the plain constructors:
`PaperCache::new`, `with_hasher`, `new_sized_compact`, ...): the defaults, with these applied on top. A configuration passed in code --
`new_with_gate`, `with_hasher_and_gate`, `set_gate_config` -- is used exactly as given and the
environment is not consulted for it (a struct cannot tell a field set on purpose from one left at its
default, so the granularity is the whole configuration). Words are case-insensitive (`admit_over`,
`Admit-Over`); an empty variable is unset; a value that does not parse, or that would make the
configuration one `GateConfig::validate` refuses, is ignored (the default stays) with one note on
stderr. `PaperCache::gate_config()` reads back what the cache runs. `PAPER_DISABLE_SHARED_OVERHEAD=1`
still forces the per-object metadata model over all of it.

## Testing

One integration-test file per design, each gated on its own feature:

```bash
cargo +nightly test --release --test lru_compact_hybrid_cache_integration --features lru_compact_hybrid_cache
```

Some reproductions are `#[ignore]`d because they take minutes and allocate tens of GB:

```bash
cargo +nightly test --release --test lru_compact_hybrid_cache_integration --features lru_compact_hybrid_cache \
  repro_real_dram_usage_at_scale -- --ignored --nocapture
```

Test builds drain the migration queue synchronously after each batch, so assertions on buffer
contents are deterministic while still exercising the real queue path.

## Benchmarking

`scripts/` holds one tool, `probe_server.py`, a protocol probe for `paper_server`. The old
`run_hybrid_benchmark_matrix.sh` rebuilt `paper-benchmark-cxl` once per design, rewriting the
`features=[...]` line in its `Cargo.toml` between runs — a premise the unification removed (and
the feature names it built no longer exist; it was deleted in R1). A single build now hosts all
23 designs, so a sweep is a loop over `PaperPolicy` values (or over their string forms, via
`FromStr`) with no rebuild between cells.

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

## License

AGPL-3.0. See `LICENSE`.
