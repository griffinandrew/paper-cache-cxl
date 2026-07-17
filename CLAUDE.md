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

## Investigation: real DRAM usage vs. `fast_tier_size` — two confirmed bugs, one open hypothesis

Reported by the user: with `lru_hybrid_cache`, `fast_tier_size = 1 GB`, real DRAM usage (measured
via `/proc/<pid>/numa_maps`, node0) ran to several GB, and their benchmark harness had no easy way
to print `lru_hybrid_stats()` for cross-checking. Rather than continue guessing from code review,
reproduced the reported workload (~1M objects, ~16 KB average, moderate/single-threaded/steady
insertion rate) directly inside this crate as a real, `#[ignore]`d integration test —
`tests/lru_hybrid_cache_integration.rs::repro_real_dram_usage_at_scale` — measuring real
`/proc/self/numa_maps` from the test process itself (`read_own_numa_usage_mb()`), rather than
relying on the cache's own self-reported stats to explain a discrepancy in those same stats. Run
with `cargo +nightly test --release --test lru_hybrid_cache_integration --features
lru_hybrid_cache repro_real_dram_usage_at_scale -- --ignored --nocapture` (several minutes; real
PMEM migrations for ~90%+ of a million objects).

**Bug 1 (confirmed, fixed): allocator prewarm ignores its own size parameter and hardcodes 18 GiB,
twice, per node, unconditionally.** `HybridObjects::init_and_prewarm`, `DRAMObjects::
init_and_prewarm`, and `ValueDRAM::init_and_prewarm` (`src/allocator.rs`) each accept a
`prewarm_bytes: usize` parameter — and every call site passes a different, clearly-intentional
value (32 GiB, 30 GiB, 35 GiB respectively) — but the function *bodies* never read that parameter,
instead hardcoding `18 * 1024 * 1024 * 1024` twice (once at 2 MB chunk granularity, once at 4 KB)
regardless of what's passed in. Confirmed directly: the reproduction's startup log showed `touched
9216 chunks x 2097152 bytes = 18432 MiB` and `touched 4718592 chunks x 4096 bytes = 18432 MiB` on
*both* NUMA nodes, before a single cache object was inserted — a large, fixed, config-independent
memory cost baked into the very first heap allocation of the process. This alone accounted for
roughly 18–24 GB per node in the reproduction (node0 dropped 37.3 GB → 12.9 GB after the fix).
Fixed per the user's explicit direction: commented out the prewarm body in all three
`init_and_prewarm` impls (`umf_allocator_init` itself — actual pool setup — is untouched; only the
prewarm/pre-fault step is disabled). `DAXPMEM::init_and_prewarm` has no prewarm call to begin with
(unaffected); the dead, fully-commented-out second `HybridObjects`/`UnifiedAllocator` impl block
further down the file (already inert, would be a duplicate-definition compile error if live) was
left alone.

**Bug 2 (confirmed, fixed): `lru_hybrid_stats`/`lfu_hybrid_stats`/`two_q_hybrid_stats`'
`fast_objects`/`slow_objects`/`fast_bytes_used`/`slow_bytes_used` gauges could go stale and never
catch up to the stack's true state.** All three `PolicyWorker::apply_tier_migrations` siblings
(`src/worker/policy/mod.rs`) refreshed these gauges only *after* an `if migrations.is_empty() {
return; }` early exit — meaning any Set/Get event that didn't itself trigger a fast/slow migration
left the gauges exactly as they were, even though the stack's own internal state had genuinely
advanced (a new key admitted with room to spare, an access that didn't cross the fast/slow
boundary, etc.). Caught because the reproduction's `wait_until(fast_objects + slow_objects ==
num_objects)` completion check *never* became true even after 10 minutes with the worker
provably idle (confirmed via `gdb -p <pid> -batch -ex "thread apply all bt"` on the stalled
process: the real `PolicyWorker<u64, TieredBuffer>` thread was parked in a normal sleep inside its
own polling loop, channel empty — not deadlocked, not crashed, just never having refreshed its
last few gauge updates). Fixed by restructuring all three `apply_tier_migrations` methods so the
gauge-refresh block runs unconditionally on every call, independent of whether that call's
`migrations` was empty — cheap now that this method already runs once per event (see
`lru_hybrid_cache`'s "burst-write headroom" post-implementation-fix section above) rather than
once per batch. The physical-migration loop and promotion/demotion counters stay correctly gated
on `!migrations.is_empty()` (nothing to physically move or count otherwise); only the gauge
refresh moved outside that gate. Re-running the reproduction after this fix showed genuine,
verifiable full settlement (`fast_objects + slow_objects == num_objects` exactly) within ~25
seconds for 1M objects — the "stall" was purely a stats-reporting artifact, not a real backlog or
hang. (A secondary, previously-unexplained SIGSEGV-on-process-exit that had appeared in every
earlier reproduction run also stopped reproducing once tests reached genuine full settlement
before the `PaperCache` — and its worker threads — dropped; plausibly a robustness gap in dropping
a cache in the middle of an in-flight migration backlog, not investigated further since it stopped
occurring once settlement was reliable.)

**Confirmed root cause (not fixed — third-party allocator behavior, not a bug in this crate):
real DRAM usage tracks total cumulative admission volume, not the fast-tier budget, because the
underlying TBB scalable-pool allocator never releases fragmented freed blocks back to the OS.**
Even with both bugs above fixed and settlement genuinely verified (`fast_objects + slow_objects ==
num_objects` exactly, not just a plausible-looking gauge snapshot), real DRAM (node0) still ran
~13–14x over the configured fast-tier budget at 1M objects (13.2 GB real vs. 953.7 MB configured
`CacheTierSize::Gb(1)` — note `Gb` is decimal SI, 10^9 bytes, ~7% smaller than a binary GiB; the
stack's own `fast_bytes_used` correctly stayed under budget throughout, ~848 MB).

Confirmed via a **peak-vs-settled, multi-scale comparison** added to the same reproduction test
(`repro_real_dram_usage_at_scale` now reads `REPRO_OBJECT_COUNT` from the environment, sampling
`numa_maps` both immediately after the insert burst — before the worker has caught up — and again
after genuine full settlement):

| objects | demotions | peak node0 | settled node0 | settled/peak |
|---|---|---|---|---|
| 50,000 | 0 (all fit in budget) | 1,106.5 MB | 1,120.3 MB | 1.012 |
| 200,000 | 140,972 | 3,959.0 MB | 3,983.3 MB | 1.006 |
| 1,000,000 | 945,823 | 13,098.4 MB | 13,246.3 MB | 1.011 |

Two things this table proves directly: (1) settled/peak ≈ 1.0 at every scale — DRAM never shrinks
back down once the worker genuinely catches up, it just stays wherever the peak left it; (2) at
50K objects, where *nothing* was ever demoted (the whole workload fit inside the fast-tier budget),
real DRAM tracked the configured budget closely (1.17x — a small, explicable margin, not a mystery
multiplier). The "peak" itself is real: every `set()` synchronously builds `TieredBuffer::new_fast`
in DRAM before the worker has decided a tier, so cumulative *admissions* (not the final tier
occupancy) drive how much memory is ever touched.

Traced to the exact line: `umf_allocator/umf_allocator_wrapper.c`'s `umf_allocator_init` explicitly
calls `umfScalablePoolParamsSetKeepAllMemory(scalable_params, 1)`, documented in its own comment as
forcing "the TBB backend to retain freed blocks rather than triggering purging calls down into the
OS memory provider." This looked like the obvious lever — but **flipping it to `0` and rebuilding
made no measurable difference** (200K-object test: 3994.9/4018.5 MB, ratio 1.006 — statistically
identical to `KeepAllMemory=1`'s 3959.0/3983.3 MB). Traced one level deeper: `umfScalablePoolOps()`
wraps Intel TBB's scalable allocator (`libtbbmalloc`), a slab/superblock allocator (2 MB granularity
here, via `umfScalablePoolParamsSetGranularity`). `KeepAllMemory` only gates whether the *pool* is
permitted to call the *provider's* free at all — and the provider's own `os_free()` (in UMF's
vendored `provider_os_memory.c`) does correctly call `munmap()` when invoked. But TBB's internal
superblock-retention heuristics — which decide whether a given freed region is ever fully empty and
thus eligible to hand back — are opaque from the UMF wrapper level and evidently never consider
these superblocks reclaimable under this allocation/free pattern (many small, staggered-lifetime
16 KB objects spread across superblocks), independent of the flag. Reverted `KeepAllMemory` to its
original `1` (documented in `umf_allocator_wrapper.c`) since `0` provided no benefit and only adds
provider round-trip overhead for no gain.

**Also tested and ruled out: `PolicyWorker`'s per-event (vs. batched) migration timing.** The user
suspected the earlier per-event `apply_tier_migrations` change (see the "burst-write headroom"
post-implementation-fix section above) might be *causing* the fragmentation, by interleaving
free/allocate/free/allocate in tight lockstep (a demoted object's DRAM buffer freed right as the
next admission allocates, potentially keeping many superblocks perpetually non-empty) versus
batched frees letting a whole cluster of older superblocks go empty together while new allocations
land elsewhere. Tested directly: reverted the call site to once-per-batch (outside the inner
`for event in events` loop, matching its pre-session position), rebuilt, and re-ran the 200K-object
comparison. Result: 4030.4/4039.1 MB (ratio 1.002) — statistically identical to per-event's
3959.0/3983.3 MB (ratio 1.006), well within normal run-to-run noise. Reverted back to per-event
(comment-only diff in `worker/policy/mod.rs`) since batching bought no memory benefit and would
have reintroduced the DRAM-write-vs-PMEM-migration latency window that change was written to close.
The allocator-level retention behavior is independent of migration granularity — consistent with
the earlier finding that it's TBB's own internal superblock heuristics, not anything this crate's
worker loop controls.

### Fix: switched the pool backend from TBB scalable to UMF's jemalloc pool — confirmed, large win

Per the user's request, tested swapping `umf_allocator_init`'s pool (`umf_allocator/
umf_allocator_wrapper.c`, shared by every allocator that routes through it — `HybridObjects`/node 1
*and* `DRAMObjects`/node 0, so this affects every PMEM-backed feature, not just the hybrid caches)
from `umfScalablePoolOps()` (TBB, `KeepAllMemory=1`) to `umfJemallocPoolOps()`, using jemalloc's
*default* decay-based release behavior as-is (no attempt to replicate `KeepAllMemory`'s
retain-everything semantics — the whole point was to let a decay-based reclaimer actually reclaim).
`libumf.so`/`libumf.a` already bundles jemalloc statically (confirmed via `strings`/`nm` — no new
link dependency needed) and exports the pool ops via the already-included `pool_jemalloc.h`; the
DAXPMEM path elsewhere in the same file had a *dead, broken* reference attempt at this pattern
(referenced a `new_pool` that was never assigned) — not used as a template, rewritten from scratch
mirroring the already-correct scalable-pool code's structure (local `new_pool`, proper lock +
atomic-publish sequencing).

Re-ran the same peak/settled/decayed multi-scale comparison (the `repro_real_dram_usage_at_scale`
test gained a third measurement, taken 30s after the "settled" one, since jemalloc's default
`dirty_decay`/`muzzy_decay` release freed pages only after an idle period — ~10s by default — not
immediately on free, so a measurement taken only 5s after settling could still catch pages
mid-decay):

| objects | demotions | peak node0 | settled node0 (+5s) | decayed node0 (+35s) | vs. TBB settled |
|---|---|---|---|---|---|
| 50,000 | 0 | 1,013.4 MB | 1,025.5 MB | 1,025.5 MB (1.08x budget) | 1,120.3 MB (1.17x) — slightly better |
| 200,000 | 140,972 | 3,533.8 MB | 1,382.2 MB | 1,358.4 MB (1.42x budget) | 3,983.3 MB (4.18x) — **~2.9x better** |
| 1,000,000 | 945,823 | 11,402.0 MB | 2,032.4 MB | 2,032.4 MB (2.13x budget) | 13,246.3 MB (13.89x) — **~6.5x better** |

At 1M objects, decayed/peak ratio dropped to **0.178** (vs. TBB's ~1.0 at every scale) — strong,
direct confirmation that jemalloc is genuinely releasing freed pages back to the OS as the fast
tier settles, not just retaining the historical peak. Real DRAM now tracks `fast_tier_size` within
roughly 1.1–2.1x across three orders of magnitude, rather than diverging without bound as object
count grew (13.89x at 1M under the old pool, and climbing — see the table in the section above).
The residual multiple (worse at larger scale) is consistent with jemalloc's own arena/decay
overhead plus the shared-metadata terms already accounted for in the DRAM-cap feature itself
(`object/overhead.rs`) — not fully investigated further, but far smaller in magnitude than the
retention behavior this replaces.

Verified no regressions: full `lru_hybrid_cache`/`lfu_hybrid_cache`/`two_q_hybrid_cache` lib +
integration suites pass unchanged, and `hybridcache_integration.rs` (which also routes through
`HybridObjects`) stays at its documented pre-existing baseline (35/37 — the same two timing-
sensitive tests, `test_demotion_counter`/`test_pmem_items_after_eviction`, that were already
failing before this session for unrelated reasons; not new).

Left in place, not reverted, since it's a clear, measured improvement with no observed downside —
unlike the `KeepAllMemory` flip and the migration-batching experiment, both tested and reverted
earlier in this same investigation for producing no benefit.

### Three further jemalloc-tuning experiments — all tested, all made things worse, all reverted

After the pool-backend swap above, the user asked to push the residual ~1.4x–2.1x multiple even
closer to the configured budget ("the allocated budget should be as closely matched at possibly
1gb should be at max 1gb"). Three statistical/tuning knobs were tried, each via rebuild + the
200K-object peak/settled comparison against the jemalloc-pool baseline (3959.0/3983.3 MB,
ratio ~1.006/1.42x budget) — **all three made real DRAM usage worse, not better**, and were
reverted:

1. **`umfJemallocPoolParamsSetNumArenas(jemalloc_params, 2)`** (down from the default,
   `num_cpus * 4`): 1.91x budget vs. the default's 1.42x. Hypothesis was fewer arenas → less
   per-arena retained overhead; actual effect was the opposite — fewer arenas means more threads
   contend for the same arena, and the resulting cross-thread allocation-pattern fragmentation
   cost more than the per-arena footprint saved. Reverted (comment-only diff documenting the
   result in `umf_allocator_wrapper.c`, no `SetNumArenas` call — left at the default).
2. **`MALLOC_CONF=dirty_decay_ms:0,muzzy_decay_ms:0`** (immediate page release instead of the
   ~10s default decay): worse. Tested via shell env var only, never committed to source (nothing
   to revert in-tree).
3. **`MALLOC_CONF=background_thread:true`**: worse. Same as above — env-var-only test, no source
   change.

Conclusion: jemalloc's plain defaults (as already landed in the pool-backend swap) are the best
configuration found via statistical/tuning-based approaches. Further improvement requires a
structural mechanism, not a tuning knob — see the next section.

### Tried and abandoned: a structural DRAM hard cap on the global `DRAMObjects` allocator

A structural hard cap was implemented (a custom jemalloc arena, bump-allocated within a fixed
`mmap`+`mbind`-pinned region, returning `NULL` once a configured `PAPER_CACHE_DRAM_CAP_BYTES` was
exhausted, opt-in and scoped to node 0/`DRAMObjects` only) after the earlier jemalloc-pool-swap
investigation (see below) plateaued at ~1.4–2.1x `fast_tier_size`, per the user's ask for something
closer to a true ceiling. It required resolving the system jemalloc's `mallctl`/`mallocx`/`dallocx`
at runtime — first via build-time linking (found to interpose over the *entire process's* `malloc`/
`free` via ELF symbol resolution, since most non-prefixed jemalloc builds export unprefixed libc
symbol names — this alone caused real, reported allocation failures with the cap completely
*unset*, root-caused via `nm -D`), then via `dlopen(RTLD_LOCAL)`, which avoided the interposition
but surfaced a second, unrelated pre-existing bug: `allocator.rs` used `println!` (which lazily
allocates its own stdout buffer on first use) inside the alloc-failure diagnostic of all four
`GlobalAlloc` impls, so a first-ever allocation failure recursed into initializing that same
`OnceLock` from within itself and deadlocked forever (confirmed via `gdb` on a real hung process
matching a user report of the benchmark hanging for many hours; fixed independently by switching to
`eprintln!`, which is unbuffered and allocation-free — this fix is retained even though the feature
that surfaced it was later removed, since it protects the same four allocators' failure paths in
general).

Ultimately abandoned and fully removed at the user's explicit request, for two independent reasons
confirmed by direct testing rather than assumption:
- On this sandbox, `dlopen("libjemalloc.so.2", ...)` fails unconditionally with "cannot allocate
  memory in static TLS block" — a real glibc limit on the small TLS surplus reserved for post-
  startup loading of libraries using the initial-exec TLS model (which jemalloc's thread-local
  caches use). Tested whether loading as early as physically possible (a C `__attribute__((constructor))`
  running before `main()`) could dodge this — it made no difference, failing identically. This is a
  structural constraint of this specific glibc/jemalloc-build combination, not something fixable by
  changing *when* the dlopen happens.
- Separately (see below), UMF's own jemalloc pool — the mechanism this hard cap's design was meant
  to route *around* — turned out to be independently unreliable under real load anyway, making the
  whole DRAM-cap effort moot regardless of the dlopen/TLS issue.

All `dram_cap_*` code, the `PAPER_CACHE_DRAM_CAP_BYTES` environment variable, and the associated
`build.rs` wiring have been removed entirely. `umf_allocator_init`/`umf_alloc`/`umf_dealloc`/
`check_tier` are back to their pre-this-feature shape (a plain per-node `pools[]` array, no dram-cap
branch). If this direction is revisited, the two blockers above (process-wide malloc interposition
from build-time linking; this sandbox's dlopen/static-TLS failure) still apply and would need
addressing first.

### Reverted, twice: `umfJemallocPoolOps()` (UMF's jemalloc pool) is unreliable under real concurrent load

The earlier TBB→jemalloc pool swap (below) was a measured DRAM-retention win against this crate's
own tests (single/few-threaded, bounded object counts) — but crashes reproducibly the first time it
was exercised under real load: `paper-benchmark-cxl`'s actual `read-through` client, multiple
concurrent rayon worker threads, a 2.3M-access real trace. Reported by the user as the benchmark
process dying partway through a run; reproduced directly under `gdb -batch -ex run -ex "thread apply
all bt full"`, confirmed via `dmesg` it wasn't an OOM kill (100+ GB free at the time).

The full backtrace traced entirely through UMF's own internals, no crate code involved: jemalloc's
completely routine internal extent-splitting (normal as arenas grow/shrink under concurrent
allocation pressure) invokes **`arena_extent_split`**, UMF's own extent-hooks callback installed on
`umfJemallocPoolOps()`'s pool so UMF can track which bytes belong to which provider — which crashes
inside UMF's own critnib (compressed radix trie) memory-tracker implementation
(`umfMemoryTrackerAddAtLevel` → `critnib_insert` → `add_metadata_and_align`, all in
`libumf.so.1.0.3`, UMF version 1.0.3). This is a bug in UMF's own prebuilt library, not in this
crate's code — no source access to `libumf.so` to patch it directly, and UMF's jemalloc pool params
expose no option that avoids the crashing path (the only exposed knob,
`umfJemallocPoolParamsSetNumArenas`, doesn't touch extent-splitting at all, and was separately
already tested and found to make memory usage worse, not better — see above).

Reverted to `umfScalablePoolOps()` (TBB, `KeepAllMemory=1`) — the previously-proven-stable
configuration — reintroducing the already-documented DRAM-retention tradeoff (real usage tracks peak
cumulative admission, not settled occupancy) as a known, accepted cost. Verified end-to-end against
the actual benchmark (not just this crate's own tests): the exact command that crashed completed
cleanly to 100% with full GET/SET stats, no new `dmesg` segfaults.

**Re-tested a second time, per explicit request, rather than relying on memory of the above
result** (after the DRAM-hard-cap feature above was abandoned, to check whether "just use UMF
jemalloc" — without the hard-cap complexity — might behave differently): failed again, a *different*
way. No SIGSEGV this time; instead a corrupted-looking allocation-failure abort partway through the
same benchmark (`DRAMObjects: UMF alloc failed for N bytes` messages with visibly torn/interleaved
byte counts mid-string, e.g. one message's tail merged into another's — a signature of concurrent
heap corruption, not a clean error path), with 160 GB of system memory still free at the time
(confirmed via `free -h`), ruling out genuine exhaustion as the cause. Two separate test runs, two
different failure modes, zero successful completions — consistent with an underlying UMF concurrency
bug that manifests differently run to run (as races typically do), not a one-off fluke. Reverted to
TBB a second time; this is now the settled, tested-working configuration, confirmed via a third
successful end-to-end benchmark run after the DRAM-hard-cap code was fully removed (previous section).

**Bottom line for future work**: `umfJemallocPoolOps()` should be treated as unsafe under real
concurrent load on this UMF version (1.0.3) until proven otherwise by UMF fixing this upstream —
don't re-enable it without re-running the actual benchmark (not just this crate's own test suite,
which is too low-concurrency/low-scale to reproduce either failure mode) to confirm.

### Confirmed, definitively: TBB's own retention behavior is not reachable via any exposed API

Per explicit request ("see if u can get rid of the memory retention issue in tbb"), tested the two
remaining levers directly (both outside this crate, via a standalone C reproduction linking the
same UMF scalable-pool code path and `libtbbmalloc.so.2`) rather than continuing to guess:

- **UMF's `KeepAllMemory` flag** (already tested earlier in this investigation, see above) —
  reconfirmed: `0` vs `1` makes no difference to real RSS for this allocation pattern.
- **TBB's own documented release API**, `scalable_allocation_command(TBBMALLOC_CLEAN_ALL_BUFFERS,
  NULL)` (declared in `/usr/include/tbb/scalable_allocator.h`, exported by `libtbbmalloc.so.2` —
  confirmed safe to link directly against, unlike jemalloc: `nm -D` shows it exports only
  `scalable_*`-prefixed symbols, no unprefixed `malloc`/`free`/`calloc`/`realloc`, so no risk of the
  process-wide malloc-interposition problem that ruled out linking jemalloc directly earlier in
  this investigation). Confirmed via `nm`/`strings` that UMF's scalable pool `dlopen`s this exact
  same system library at runtime, so a direct call from this process operates on the identical
  allocator instance/state UMF's pool uses internally — no separate/inert copy.

Reproduced the crate's real fragmentation pattern directly (300,000 × 16 KB allocations through a
real `umfScalablePoolOps()` pool at the crate's actual 2 MB granularity, then freeing ~90% at
scattered/staggered positions rather than a clean contiguous run, matching how demotion leaves
survivors scattered rather than clustered): RSS went from a 3 MB baseline to 6,519 MB after
admission, stayed at exactly 6,519 MB after freeing 90%, and **stayed at exactly 6,519 MB** after
calling `scalable_allocation_command(TBBMALLOC_CLEAN_ALL_BUFFERS, ...)` — which returned
`TBBMALLOC_NO_EFFECT` (not an error; TBB's own honest answer that it found nothing releasable).
Retested with `KeepAllMemory=0` too: identical result, `NO_EFFECT` again.

**Root cause, now confirmed at the TBB API level directly rather than inferred**: at 2 MB
granularity and ~16 KB objects, each chunk holds ~128 objects; with only 10% survival scattered
essentially at random, nearly every chunk still contains at least one live object, so none are
"empty" by TBB's own accounting and none are eligible for release under *any* exposed API — this
isn't a missing flag, it's TBB's slab/superblock design colliding with this workload's actual
survivor-scattering pattern. No further UMF- or TBB-level knob exists to try.

**The only approach that structurally can't have this problem — bypassing pooling entirely
(`mmap`/`munmap`, or a persistent arena + `MADV_DONTNEED` on free) — works, but at a real,
measured throughput cost.** Validated both directly, same reproduction: `mmap`+touch per object
took ~7.5 µs/op, `MADV_DONTNEED`+re-touch (against a pre-existing arena, avoiding repeated `mmap`
syscalls) took ~3.5–6 µs/op — versus TBB's effectively-free in-pool reuse. Both gave RSS that
tracked survivors almost exactly (472–474 MB after freeing 90% of 4,691–4,693 MB admitted, matching
the true 10% survival rate). The cost is fundamentally a page-fault cost, not syscall overhead —
even a long-lived arena pays a fresh fault every time a `MADV_DONTNEED`'d slot is re-touched on
reuse, since the physical page is genuinely gone — so it can't be tuned away by avoiding `mmap`
calls specifically; giving memory back to the OS accurately and reusing it near-instantly are in
direct tension. At this crate's measured real throughput (~150K sets/sec, ~5.5 µs avg SET latency
end-to-end in the real benchmark), several extra µs per fast-tier admission/demotion would likely
become the dominant cost and could substantially cut throughput — not confirmed via a full
benchmark run since the option was not pursued (see below), but the per-op numbers alone make this
a real risk, not a rounding error.

**Decision (explicit, after being presented with this tradeoff): do not implement it.** Real DRAM
usage continues to track cumulative peak admission volume rather than `fast_tier_size`, as already
documented above — this is the accepted, known cost of staying on the TBB backend, which remains
the only pool backend proven stable under real concurrent load (see the jemalloc-pool section
above). If this is revisited, the two validated-but-rejected levers (raw `mmap`/`munmap` per
object; a persistent arena + `MADV_DONTNEED`) are the correct starting point — no further time
should be spent on UMF/TBB flags or purge APIs, both now confirmed dead ends at the API level, not
just empirically absent from measurements.

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
    from `fast_used`) until `fast_used <= fast_capacity` exactly — no low-water floor below the
    ceiling. Collect every key that flipped tier this call. **Design history:** an earlier version
    of this stack drained down to a fixed 90%-of-`fast_capacity` floor (`fast_low_water`,
    `FAST_TIER_LOW_WATER_RATIO`) instead of exactly `fast_capacity`, on the reasoning that draining
    to the exact ceiling would leave the fast tier hovering right at the boundary so almost every
    subsequent `set()` would re-trigger a demotion pass. The user explicitly asked for that headroom
    to be removed (*"keeping the 10% high water mark in the lru implementation hurts performance so
    get rid of it"*): reserving idle fast-tier capacity to reduce demotion-pass frequency was judged
    not worth the usable-space cost. `settle_fast_tier` now drains to exactly `fast_capacity`; the
    fast tier can legitimately sit at 100% utilization.
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

### Post-implementation fix: burst-write headroom, and why headroom alone doesn't bound it

The user asked whether `LruHybridStack` should keep some headroom below `fast_capacity` so
concurrent `set()` calls don't immediately push it over the threshold, given the background
`PolicyWorker` thread processes tier decisions asynchronously. Investigating the actual mechanism
first: `PaperCache::set()` writes a new object's `TieredBuffer` to DRAM **synchronously**, at the
API layer, before the corresponding `WorkerEvent::Set` is even broadcast — the worker only updates
`LruHybridStack`'s bookkeeping (and decides demotions) once it later processes that event. Tracing
through `PolicyWorker::run` showed the bookkeeping decision itself is already eager (`settle_fast_tier`
runs synchronously inside `stack.insert()`, not deferred) — the actual gap is between that decision
and `apply_tier_migrations` physically moving demoted bytes to PMEM, which used to run once *per
polling-loop batch* (after draining and processing every currently-queued event), not per event.
Headroom in `settle_fast_tier`'s drain target doesn't touch that gap at all: a burst of concurrent
`set()` calls still write straight to DRAM regardless of what threshold the stack targets internally.
Two changes were made together, addressing the two different halves of the problem:

1. **`PolicyWorker::apply_tier_migrations` is now called per-event, inside the event loop's `for`
   block, instead of once after the whole batch drains** (`worker/policy/mod.rs`). This is the
   change that actually shrinks the DRAM-write-vs-PMEM-migration window: for a large batch of
   queued events, earlier ones' migrations now execute immediately rather than waiting for the rest
   of the batch to be logically processed first. Safe to call more often — it's a cheap early
   return when there's nothing to migrate (`drain_tier_migrations` returns empty) — and applies
   uniformly to `lru_hybrid_cache`/`lfu_hybrid_cache`/`two_q_hybrid_cache` alike, since they share
   this loop.
2. **`LruHybridStack::settle_fast_tier` regained a low-water floor** — `FAST_TIER_LOW_WATER_RATIO
   = 0.98` (2%), reintroduced deliberately smaller than the 90% floor removed earlier in this same
   file's history (*"keeping the 10% high water mark in the lru implementation hurts performance so
   get rid of it"*). The trigger condition is unchanged (only fires once genuinely over the
   effective budget — no early demotion), but the drain target is now `effective *
   FAST_TIER_LOW_WATER_RATIO` rather than `effective` exactly, leaving concurrent bursts some room
   to land in before the next settle needs to trigger again. Scoped to `LruHybridStack` only, per
   the user's explicit "at least for lru" — `LfuHybridStack` doesn't re-settle on every admission
   the way this stack does, so the same thrashing-vs-burst-room tradeoff doesn't apply the same way
   there. Framed honestly in the code as a *smaller* safety margin layered on top of fix #1, not a
   bound on transient overshoot by itself.

At the razor-thin fast-tier capacities several existing unit/worker tests intentionally use ("fits
exactly one object"), a 2% shave of an already-tiny effective budget can land the drain target
*below* what a single already-resident object needs, cascading an extra demotion that isn't the
scenario under test (observed as, e.g., a stack test that meant to assert one demotion instead
demoting both same-sized objects to zero, and a worker-level cascade test flipping from 2 to 3
demotions). Fixed two ways depending on the test's intent: where the point was specifically to
demonstrate the reintroduced headroom, the expected outcome was updated to match the real (fully-
evacuating, at that tiny scale) result; where the tight capacity was incidental to a *different*
scenario under test, the capacity was scaled up via a small `low_water_safe(target) = ceil(target /
0.98)` helper so the object(s) meant to survive comfortably clear the drain target regardless of
the shave.

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
2. **No low-water headroom for demotion.** `settle_fast_tier` drains exactly back to
   `fast_capacity`, no separate low-water constant — `LfuHybridStack`'s demotion is only
   triggered by a promotion or an explicit `resize_fast_tier`, not by every `set()`. (At the time
   this decision was made, `LruHybridStack` still drained to a 90%-of-capacity floor because
   *every* `set()` there re-admits to fast and could re-trigger demotion — so this was framed as
   a difference between the two stacks. That floor was later removed from `LruHybridStack` too,
   per the user's explicit request that it hurt performance — see that section's "Design
   history" note — so both stacks now drain to exactly `fast_capacity` for the same reason:
   `LruHybridStack`'s trigger point stayed exactly `fast_capacity` from the start, only its drain
   *target* changed.)
3. **Admission does an explicit fast-tier capacity check before touching the fast chain** — a
   new key is admitted to the fast chain only while `fast_used + size <= fast_capacity`; once the
   fast tier is full, every subsequent new key is routed directly to the slow chain instead,
   matching the paper's admission rule literally ("every new object is admitted into the slow
   tier"). **Design correction made after initial implementation:** the original design (recorded
   here during planning) had admission always land fast first — deliberately *not*
   special-cased to route straight to slow — reasoning that this still satisfied the paper's
   admission rule as an *emergent* result, since a freshly admitted object (frequency 1) is
   always tied for the fast tier's lowest frequency once the tier is full, so `settle_fast_tier`
   would demote *someone* tied at that frequency in the same `insert` call. In practice this let
   tie-breaking (point 4 below) decide *who* gets demoted, which could pick an *older, existing*
   resident instead of the newcomer — a real deviation from the paper's literal spec, reported by
   the user (*"sets should only enter the fast tier until capacity is met then they should be
   inserted into the slow tier"*) and fixed by making the capacity check explicit and unconditional
   at admission time, with no reliance on tie-breaking for the admission decision at all. This also
   meant `lib.rs`'s `set()` for `TieredBuffer` under `lfu_hybrid_cache` could no longer stay
   textually identical to `lru_hybrid_cache`'s (which still always synchronously builds
   `TieredBuffer::new_fast` — no capacity check at the API layer, since `LruHybridStack`'s
   admission genuinely always lands fast); `PolicyWorker::apply_tier_migrations`'s LFU sibling
   physically corrects a fresh admission-to-slow's `TieredBuffer` from `Fast` to `Slow` after the
   fact instead (see the stats-double-counting fix below).
4. **Ties within a frequency bucket break toward the least-recently-touched key**, matching the
   existing `LfuStack` (plain LFU) convention already in this codebase
   (`worker/policy/policy_stack/lfu_stack.rs`: `CountStack::push` = `push_front` on touch,
   `pop` = `pop_back` on eviction). This still governs *demotion* tie-breaking (whenever
   `settle_fast_tier` runs, e.g. after a promotion), but — per the admission fix above — no
   longer has any bearing on which key lands where at *admission* time; a freshly admitted key
   either fits in fast or is routed directly to slow, deterministically, regardless of any
   existing resident's frequency or recency.

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

*(Historical — describes the original "admit fast, let tie-breaking demote someone" design,
which was later found to deviate from the paper's admission rule and replaced with the explicit
capacity check in decision 3 above. Kept for context on how the bug was first surfaced; the
tie-breaking convention it describes still governs genuine demotions, just no longer admission.)*

An early unit test (`admission_once_fast_is_full_is_demoted_immediately`) asserted that when a
fast tier fits exactly one key and a second key is admitted (tying its frequency at 1), the
*newcomer* would be the one demoted — reasoning loosely from the paper's "every new object is,
by definition, the least frequently accessed" framing. Running the test against the real
implementation showed the opposite: the *older* key demoted. At the time this was treated as
correct-not-a-bug — ties within a frequency bucket break toward whichever key is
least-recently-touched, and a newcomer is pushed to the *front* of its bucket (most-recently
touched), so it's actually the pre-existing, untouched resident that's LRU-within-the-tie and
gets evicted first, matching `LfuStack`'s (the plain LFU policy already in this crate) existing,
already-tested tie-breaking convention. That reasoning about the *tie-break mechanics* was and
still is accurate — what turned out to be wrong was relying on it for *admission* at all: the
paper's spec says the newcomer specifically goes to slow once fast is full, not "whichever key
loses a tie-break," and demoting an older resident instead is an observably different outcome
from the paper's rule. The user caught this in practice and reported it; the fix (decision 3
above) makes admission a deterministic capacity check with no tie-breaking involved, and the
integration test this originally validated (`admission_once_fast_is_full_is_demoted_immediately`)
was rewritten as `admission_once_fast_is_full_goes_directly_to_slow` to assert the newcomer lands
in slow while the existing resident stays put. A second, related pitfall from this same era: the
module-level doc comment originally claimed `settle_fast_tier` "demotes [the freshly admitted
key] right back out in the same `insert` call" — also no longer applicable, since admission never
touches `settle_fast_tier` at all now.

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

### Post-implementation fix: deterministic admission, and why demotions needed a separate counter

Reported by the user after the feature above had already shipped: *"i see errors in the lfu
implementation.... sets should only enter the fast tier until capacity is met then they should be
inserted into the slow tier"* (with the paper description re-pasted for emphasis). This is the bug
described in decision 3 above — `insert()`'s new-key branch now does an explicit
`if self.fast_used + size as CacheSize <= self.fast_capacity` check before touching `fast_chain`,
routing directly to `slow_chain` if the fast tier is already full, with a `Tier::Slow` entry
pushed to `migrations` either way (needed so `PolicyWorker` physically corrects the API layer's
default `TieredBuffer::new_fast` construction — see `lib.rs`'s `set()`, which still always builds
`Fast` first and relies on the stack's migration to fix it up, same as before).

This surfaced a second, previously-latent bug: a fresh admission-to-slow now produces a
`Tier::Slow` migration for a reason that has nothing to do with demoting an existing resident, but
`PolicyWorker`'s LFU-specific `apply_tier_migrations` sibling was counting *every* `Tier::Slow`
migration as a genuine demotion (`match tier { Fast => promotion, Slow => demotion }`), inflating
`lfu_hybrid_stats().demotions` for admission events that displaced nothing. Fixed with a new
`PolicyStack::drain_demotions() -> u64` trait method (default `0`, see
`worker/policy/policy_stack/mod.rs`'s doc comment on it for the full Fast/Slow-migration-vs-
demotion distinction), backed by a `pending_demotions: u64` field on `LfuHybridStack` incremented
*only* inside `settle_fast_tier` (never on admission), reset in `clear()`. `AtomicStatus` gained a
batch method, `record_lfu_hybrid_demotions(count: u64)`, and the LFU `apply_tier_migrations`
sibling now still physically applies every migration (both directions — a `Tier::Fast` migration
is always a genuine promotion, counted per-entry as before) but counts demotions once per pass via
`stack.drain_demotions()` afterward instead of inferring them from `Tier::Slow` entries.
`LruHybridStack`/`TwoQHybridStack` don't have this ambiguity (their admission never lands fast
unconditionally then needs correcting) and keep the trait method's default `0`.

Fixing this required rewriting every test whose assertions had implicitly relied on the old
tie-break-decides-admission behavior — see `tests/lfu_hybrid_cache_integration.rs` and the
`lfu_hybrid_tests` module in `worker/policy/mod.rs`. One test-writing pitfall specific to the
integration-test rewrite: `wait_until`-per-candidate loops (e.g. "find whichever of these five
keys ended up in the slow tier") must check *all* candidates inside a single `wait_until`
predicate, not give each candidate its own full-timeout `wait_until` call in sequence — a
candidate that's going to stay `Fast` forever burns its entire timeout budget before the loop
moves to the next candidate, and a short-TTL test can lose the race against its own key's
expiry as a result (`ttl_survives_a_demotion` hit exactly this: the TTL'd key had already expired
by the time a several-times-10-second sequential search finally found a slow filler to promote).

Companion fix in the same session, unrelated to LFU: `LruHybridStack`'s 90%-of-capacity low-water
floor (see the "No headroom" note in `lru_hybrid_cache`'s section above) was removed at the user's
explicit request (*"keeping the 10% high water mark in the lru implementation hurts performance so
get rid of it"*) — `settle_fast_tier` there now also drains to exactly `fast_capacity`.

### Post-implementation fix: fast-tier size as a total-DRAM budget, and two follow-on corrections

In a later session, `lru_hybrid_cache`/`lfu_hybrid_cache`'s fast-tier budget (`CacheTierSize`,
e.g. `CacheTierSize::Gb(4)`) was extended to bound *total DRAM*, not just fast-tier object values:
both stacks gained a `shared_overhead: CacheSize` field (`0` unless set via `with_shared_overhead`,
wired in by `init_policy_stack` via `object::overhead::get_hybrid_dram_shared_overhead(&policy)`)
representing the approximate per-object DRAM cost of the shared object hashtable + eviction stacks
(both hold an entry for every object of *both* tiers). `settle_fast_tier` demotes against an
*effective* budget of `fast_capacity − (tracked_count × shared_overhead)` rather than raw
`fast_capacity`. Per the user's explicit direction, this is **demotion-only** — it never triggers
eviction; terminal eviction remains governed solely by `used_size() > max_size`, popping the slow
tail exactly as before. `eviction_stacks_pmem` moving the eviction stacks to PMEM (see the section
below) drops that term from the reservation; a hashtable-PMEM feature would similarly drop the
hashtable term. `two_q_hybrid_cache` was left out of scope.

Two follow-on bugs were then reported and fixed in the same area:

**1. LFU admission wasn't honoring frequency order once the fast tier had ever been demoted from.**
`LfuHybridStack::settle_fast_tier`'s demotion granularity is per-*object*, not per-*byte*: demoting
one lowest-frequency fast object to cover a small promotion overage can free far more bytes than
the overage itself (e.g. demoting a 90-byte object to cover a 5-byte overage leaves 85 bytes of
slack). The byte-capacity admission check alone let a brand-new, frequency-1 key sneak into that
slack — bypassing the "prove yourself via promotion from slow" path — even when every current fast
resident already had frequency ≥ 2. Reproduced directly against the stack (two 50-byte keys fill a
100-byte fast tier; a promotion demotes one via LRU-tied eviction, leaving headroom; once every
remaining fast resident is bumped past frequency 1, a brand-new frequency-1 key still lands in
fast purely on leftover bytes). Two fixes were considered — gating admission on whether the fast
chain's *current minimum frequency* is still 1 (rejected by the user: "it still is not really
honoring lfu order" — it still lets a newcomer ride in alongside any surviving untouched
frequency-1 resident) vs. a one-time latch (chosen) — `fast_tier_latched: bool` permanently closes
brand-new-key admission to fast the first time capacity is genuinely reached (a failed admission,
or any `settle_fast_tier` demotion). Once latched, every subsequent brand-new key goes straight to
slow regardless of later byte slack, only reachable via promotion. Resets on `clear()`; also resets
on `resize_fast_tier` *growing* the budget (a deliberate capacity increase should be immediately
usable, not gated behind promotions) but not on shrinking. See `lfu_hybrid_stack.rs`'s module doc
for the full derivation and the rejected frequency-gate alternative, kept as a documented option
for a future revision.

**2. The DRAM-reservation constants were overestimating by a meaningful margin.** Investigation
(measuring real `std::mem::size_of` values for the concrete types involved, combined with
`hashbrown`'s ~7/8 max-load-factor amortization formula, `ceil((raw_pair_size + 1) * 8 / 7)` —
see `object/overhead.rs`'s derivation block) found two concrete sources of over-counting: (a)
`HASHTABLE_ENTRY_OVERHEAD` was an arbitrary `24`, versus a derived `11` (the map's own 8-byte
`HashedKey`, distinct from any key the `Object` stores internally, plus its amortized hashbrown
overhead); (b) `get_hybrid_dram_shared_overhead`'s eviction-stack term reused `get_policy_overhead`
verbatim, which turned out to measure `size_of::<HashList<..>>()` — the *container's* one-time
fixed struct size (a 32-byte `HashMap` header + 2 pointers) — as if it were a per-entry cost, on
top of a separately redundant "+8 for the key" charge (the key is already stored once, inside the
list's heap node). Replaced with dedicated, derived-from-measurement constants
(`LRU_HYBRID_EVICTION_STACK_DRAM_OVERHEAD = 84`, `LFU_HYBRID_EVICTION_STACK_DRAM_OVERHEAD = 113`)
computed independently of `get_policy_overhead` (which is left untouched — it's shared,
`used_size`-oriented, out of scope here, and reusing it was the mistake). Net effect: total
reserved per-object overhead dropped from 105→95 bytes (LRU) and 161→124 bytes (LFU). Two
integration tests (`dram_cap_reserves_shared_metadata_*`) that relied on the reservation alone
forcing demotion/slow-routing within 40 tiny objects had their object counts bumped to 300 for
comfortable margin under the smaller, now-correct numbers (especially under `eviction_stacks_pmem`,
where only the ~11-byte hashtable term applies, not the larger eviction-stack term).

### Post-implementation fix: the admission latch only fixed bookkeeping, not physical placement

Reported by the user after the latch above had already shipped: the admission fix only changes
*`LfuHybridStack`'s logical* decision about which tier a brand-new key belongs in — it doesn't
touch `PaperCache::set()` (the `TieredBuffer` impl block, `lib.rs`), which still unconditionally
built every brand-new key as `TieredBuffer::new_fast` regardless of what the stack would go on to
decide. Once latched, this meant *every* admission still paid a synchronous DRAM write followed by
an async PMEM correction (physically applied by the now-per-event `apply_tier_migrations`) — the
latch closed the "who gets to stay fast" question but never shortened that per-admission round
trip once closed.

Fixing this required exposing the latch to the API-calling thread, which has no direct access to
the worker-owned `policy_stack`. Added `PolicyStack::admission_latched() -> bool` (default `false`;
only `LfuHybridStack` overrides it — `LruHybridStack`/`TwoQHybridStack` have no such ambiguity,
since their admission rules are unconditional in one direction each), mirrored onto a new
`AtomicStatus::lfu_hybrid_admission_latched: AtomicBool`, written by
`PolicyWorker::apply_tier_migrations`'s LFU sibling *unconditionally at the top* (before the
empty-migrations early return, so the mirror stays fresh even on iterations whose event alone
didn't happen to produce a migration) and reset synchronously by `AtomicStatus::clear()` (called
from `wipe()` on the API thread) rather than waiting for the worker to catch up. `PaperCache::set()`
now checks `!self.objects.contains_key(&hashed_key) && self.status.lfu_hybrid_admission_latched()`
— gated on the key being genuinely new, via one extra cheap `DashMap::contains_key` — before
choosing `TieredBuffer::new_slow` over the default `new_fast`; an **existing** key is deliberately
never affected by this check, regardless of its current tier, since re-setting one is an access
(may or may not promote it) that only the stack can decide.

**Known, accepted tradeoff, confirmed with the user before implementing:** building
`TieredBuffer::new_slow` directly means that specific `set()` call now allocates via the PMEM/UMF
allocator *synchronously, on the calling thread* — a real latency cost the previous always-DRAM
`set()` didn't have, previously deferred entirely to the background worker. Chosen anyway (the
alternative — leaving `set()` unconditionally fast and relying solely on the faster
`apply_tier_migrations` — was explicitly offered and declined) because once warmed up this
eliminates the write-then-correct round trip entirely for the common steady-state case (every
`set()` after the fast tier first fills).

Verified end-to-end (not just via the stack's own bookkeeping) with a new integration test,
`set_places_a_brand_new_key_directly_in_slow_once_admission_is_latched`, that reads `tier_of()`
immediately after `set()` returns with **no** `wait_until` — proving the object was placed
correctly the first time rather than starting `Fast` and being corrected moments later — plus an
assertion that re-setting an existing fast key stays fast (the `is_new` guard).

### Remaining work

- No dedicated test yet for a multi-key single-step promotion/demotion cascade with mixed
  small/huge object sizes (same caveat noted for `lru_hybrid_cache` above — the implementation
  already returns a `Vec` of migrations per call, so this is handled, just not yet exercised by a
  test with deliberately varied sizes).

## Feature: `two_q_hybrid_cache` (implemented — mirrors `lru_hybrid_cache`/`lfu_hybrid_cache`)

### Source (paper description this implements)

> The originally proposed 2Q algorithm maintains two queues: a small one-access FIFO queue for
> newly admitted objects, and a main LRU queue for objects with demonstrated reuse. This is very
> similar to S3-FIFO, aiming to filter out one-access objects with a separate small queue, but
> replaces the main FIFO queue with an LRU queue. In a hybrid design, the one-access FIFO queue
> resides entirely in the slow tier, as these objects have not yet demonstrated reuse and are
> considered cold. The main LRU queue is segmented across both tiers. A re-access to an object in
> the one-access FIFO queue promotes it immediately to the top of the main LRU queue in the fast
> tier; an object that reaches the top of the one-access FIFO queue without a second access is
> evicted. Once in the main LRU queue, the object behaves as described in the LRU-hybrid section.
>
> - **Admission**: every new object is placed in the one-access FIFO queue in the slow tier.
> - **Demotion**: the least recently accessed object at the bottom of the fast tier portion of
>   the main LRU queue is moved to the top of the slow tier portion when fast tier space is
>   needed.
> - **Promotion**: a re-accessed one-access FIFO queue object is moved to the top of the fast
>   tier portion of the main LRU queue; a re-accessed object in the slow tier portion of the main
>   LRU queue is moved to the top of the fast tier portion.
> - **Eviction**: the least recently accessed object at the bottom of the slow tier portion of
>   the main LRU queue is removed when capacity is exhausted, or when a one-access object that
>   reaches the top of the one-access FIFO queue without re-access is removed.

Same overall shape as `lru_hybrid_cache`/`lfu_hybrid_cache`: **one** `PaperCache<K, TieredBuffer>`,
not two composed instances. Requirements 1–5 from the `lru_hybrid_cache` section (single instance,
actual data movement, TTL survives a tier move, configurable tier size, terminal evictions
counted) all carry over unchanged. The defining difference from the other two hybrids: admission
never lands in the fast tier at all — every `set()` is a real, synchronous slow-tier/PMEM write.

### Design decisions (confirmed during planning, in order)

1. **Two live structures, not three.** This crate's existing plain `PaperPolicy::TwoQ(k_in,
   k_out)` (`two_q_stack.rs`) has a heavier shape — a size-capped FIFO-in queue, a size-capped
   "overflow" queue that also holds real live objects (not just ghost keys), and an uncapped main
   queue. The hybrid version instead follows the pasted paper text literally: `fifo_queue` (real
   objects, always slow) and `main_stack` (segmented fast/slow, structurally identical to
   `LruHybridStack::stack`) are the only two live structures.
2. **No ghost queue.** An early draft added a classic-2Q-style ghost queue (bare evicted keys,
   checked on every admission so a "reformed" object could skip straight back into the main
   queue). Rejected: an exact-membership check on *every* `set()` — which already pays a
   synchronous slow-tier/PMEM write — was flagged as an unwelcome added cost. A cheaper
   *probabilistic* structure (e.g. a counting Bloom filter) would be the right tool to revisit
   this, left as future work (see below). A FIFO object that ages out without a second access is
   simply evicted outright — fully removed, no trace kept.
3. **`k_in` is a real, settable parameter — `PaperPolicy::TwoQHybrid(f64)`** — not a bare
   param-free variant like `LruHybrid`/`LfuHybrid`. `fifo_capacity = k_in * max_size` is the
   FIFO queue's own byte budget, mirroring plain `TwoQ`'s `k_in` exactly (no `k_out`, since
   there's no ghost/overflow queue to size). This means `PaperCache::<K, TieredBuffer>::new`
   under this feature takes an extra parameter other hybrids don't: `new(max_size,
   fast_tier_size, k_in)`.
4. **The main queue's fast/slow split is a separate mechanism from `k_in`.** `fast_tier_size`/
   `set_fast_tier_size` — the exact same runtime-configurable mechanism `lru_hybrid_cache`/
   `lfu_hybrid_cache` already use — governs how much of `main_stack` is fast vs. slow,
   independent of `k_in`. `TwoQHybridStack` ends up with two independent sizing knobs: `k_in`
   (fixed at construction, proportional to `max_size`, rescaled on `resize()`) and
   `fast_tier_size` (settable at construction *and* freely adjustable afterward).
5. **`set()` always synchronously builds `TieredBuffer::new_slow`** for a brand-new key — this is
   the literal, intended cost of "every new object is placed in the slow tier," confirmed
   explicitly rather than treated as an accidental side effect: only proven-hot (re-accessed)
   objects ever reach fast DRAM. Since the physical tier the API layer chooses (slow) and the
   tier the stack assigns a brand-new key (also always slow) agree by construction, admission
   never needs to produce a migration — unlike a promotion, which does.
6. **Eviction priority: FIFO tail first, then the main queue's slow tail** (falling back to the
   main queue's fast tail only if nothing has ever been demoted there yet — same fallback
   `LruHybridStack`/`LfuHybridStack` already have). This is the one reading required to reconcile
   the paper's two eviction clauses into a single `evict_one()` rule: preferring to sacrifice
   still-unproven FIFO objects before ever touching the proven main queue reproduces both stated
   behaviors as a single priority order.
7. **No low-water headroom for `settle_fast_tier`**, matching `LfuHybridStack`'s reasoning, not
   `LruHybridStack`'s: fast-tier pressure here is only ever triggered by a promotion or an
   explicit `resize_fast_tier`, never by every `set()` (admission never touches the fast tier
   directly).

### Implementation

**New policy: `PaperPolicy::TwoQHybrid(f64)`** (`policy.rs`), string form `"2q-hybrid-{k_in}"`.
**Important ordering fix**: `FromStr`'s existing `value if value.starts_with("2q-") =>
parse_two_q(value)?` guard would incorrectly swallow every `"2q-hybrid-..."` string (since it
also starts with `"2q-"`) if left in its original position — the new, more specific
`value.starts_with("2q-hybrid-")` guard has to be checked *first*. A collision test
(`two_q_hybrid_does_not_collide_with_parameterized_2q`) locks this in.

**New policy stack: `TwoQHybridStack`** (`worker/policy/policy_stack/two_q_hybrid_stack.rs`).
Fields: `fifo_queue`/`main_stack: HashList<HashedKey>`, `queue: HashMap<HashedKey, Queue>` (which
live structure a key is in — `Fifo` or `Main`), `main_tiers: HashMap<HashedKey, Tier>` (only
populated for `Main` keys), `sizes`, `k_in`, `fifo_capacity`/`fifo_used`,
`fast_capacity`/`fast_used`/`slow_used`/`fast_count`, `main_boundary` (mirrors
`LruHybridStack::fast_boundary`, scoped to `main_stack`), `migrations`. `insert` on a brand-new
key always pushes into `fifo_queue`. `update`/re-`insert` on an existing key dispatches on
`queue`: a `Fifo` hit promotes straight to `Main`+`Fast` (mirrors admission's `push_front`, plus
tier/size bookkeeping and a migration); a `Main` hit reuses `touch_main_fast`, copied nearly
verbatim from `LruHybridStack::touch_fast_key` (reorder-only if already Fast, promote-and-settle
if Slow). `settle_fast_tier` is a straight copy of `LruHybridStack`'s minus the low-water floor.
`evict_one` tries the FIFO tail first, then falls back to the same `main_stack`-tail logic
`LruHybridStack::evict_one` already has.

**A real correctness bug this caught: a `PolicyStack` cannot evict on its own.** The first
implementation gave `TwoQHybridStack` a private `settle_fifo_queue` method — modeled loosely on
`LruHybridStack::settle_fast_tier` — called directly from `insert`/`resize`, which popped
`fifo_queue`'s tail and dropped it from the stack's own `queue`/`sizes`/`fifo_used` bookkeeping
whenever `fifo_used` exceeded `fifo_capacity`. This compiled and every *unit* test (which only
exercises the bare stack, with no surrounding object map) passed. Every *integration* test
involving eviction failed outright, not flakily: `cache.has(&key)` kept returning `true` for keys
the stack itself had already "evicted." The bug: `PolicyStack` implementations have no reference
to the shared object map or `AtomicStatus` — the only place that's allowed to physically remove
an object and adjust accounted size is `PolicyWorker::apply_evictions`'s existing `evict_one()` +
`erase()` pairing (already correct for every other stack, including `LruHybridStack`/
`LfuHybridStack`'s demotions, which only ever swap `Object::data` in place, never remove the
object). `settle_fifo_queue` broke that invariant by fully removing a key from the stack's *own*
bookkeeping without going through `erase()` at all — permanently desyncing the stack's view of
the world from the real object map, with no way to reconcile afterward (the object just leaked
forever, uncounted and unreachable via the stack, but still fully present and `has()`-visible).

Fixed by adding a new `PolicyStack` trait method, `fn needs_capacity_eviction(&self) -> bool {
false }` (default no-op, matching the style of the other hybrid-only trait extensions), which
`TwoQHybridStack` overrides as `self.fifo_used > self.fifo_capacity`. `insert`/`resize` no longer
evict anything themselves — they only update `fifo_used` and let this method report the pressure.
`PolicyWorker::apply_evictions`'s loop condition became `while status.used_size() > max_size ||
policy_stack.needs_capacity_eviction()` (guarded by `stack.len() > 0` too, defensively, so a
stack that ever reports pressure with nothing left to evict can't spin forever) — meaning
`fifo_capacity` pressure now drains through the *exact same* `evict_one()`/`erase()` path as
global `max_size` pressure, which was already correct. This generalization required touching
`worker/policy/mod.rs`'s `apply_evictions` (shared code), but is a pure additive default for
`LruHybridStack`/`LfuHybridStack` — confirmed both features' full test suites still pass
unaffected afterward.

**Worker plumbing, `AtomicStatus`, and the `lib.rs` impl block are adapted mechanically from
`lfu_hybrid_cache`'s**, same pattern as before: the shared `any(feature = "lru_hybrid_cache",
feature = "lfu_hybrid_cache", feature = "two_q_hybrid_cache")` gates widened a third time, a
third `apply_tier_migrations`/eviction-recording sibling added (mutually exclusive, so only one
ever compiles), `two_q_hybrid_*` counters/gauges on `AtomicStatus` (independently named, not
merged with the other two hybrids' sets), and a new `#[cfg(feature = "two_q_hybrid_cache")] impl
PaperCache<K, TieredBuffer, S>` block in `lib.rs`. The one real code difference from
`lfu_hybrid_cache`'s block: `set()` builds `TieredBuffer::new_slow` instead of `new_fast`, and
`new`/`with_hasher` take the extra `k_in` parameter (validated `(0.0..=1.0)`, `CacheError::
InvalidPolicy` otherwise). The mutual-exclusion `compile_error!` guard became three pairwise
guards (`lru`+`lfu`, `lru`+`two_q`, `lfu`+`two_q`) rather than a single N-way check, matching this
file's existing pairwise-guard style for other feature conflicts.

**A test-writing lesson this caught (same one `lru_hybrid_cache`/`lfu_hybrid_cache` already
hit): TTL tests need a fast tier sized comfortably larger than one ttl'd object.** The first
`ttl_survives_a_demotion` draft used a fast tier sized only for `None`-ttl objects and a single
second key to force demotion pressure. A ttl'd object's `base_size` (via `get_ttl_overhead`) is
large enough on its own to exceed that capacity, so promoting it immediately re-triggered
`settle_fast_tier`, demoting the very key just promoted — both migrations land in the same
`drain_tier_migrations` batch, so the test's `wait_until(... == Some(Tier::Fast))` poll had no
window to ever observe the intermediate Fast state and just timed out. Fixed the same way the
other two hybrids' equivalent tests were: a fast tier sized comfortably larger than one ttl'd
object, with several small filler keys creating demotion pressure instead of a single
same-sized key.

### Implementation status: complete

All of `policy.rs`, `object/overhead.rs`, `worker/policy/policy_stack/two_q_hybrid_stack.rs`,
`worker/policy/policy_stack/mod.rs` (new `needs_capacity_eviction` trait method),
`worker/policy/mod.rs` (`apply_evictions`'s widened loop condition, third `apply_tier_migrations`
sibling), `status.rs`, `src/two_q_hybrid_cache/`, `lib.rs`, and `Cargo.toml` are done. 69
unit/inline tests pass under `--features two_q_hybrid_cache` (including a
`worker::policy::two_q_hybrid_tests` module mirroring the other two hybrids' synthetic-buffer
wiring tests). `tests/two_q_hybrid_cache_integration.rs` (18 tests, real PMEM/UMF allocator,
modeled on the other two hybrids' integration files — every test here pays the PMEM warm-up cost,
unlike the other two, since admission itself is always a slow-tier write) passes twice in a row
(not flaky): `cargo +nightly test --test two_q_hybrid_cache_integration --features
two_q_hybrid_cache`. Confirmed `lru_hybrid_cache`'s and `lfu_hybrid_cache`'s own test suites (unit
+ integration) are unaffected by the shared `apply_evictions` change.

### Remaining work

- No ghost/re-admission memory (see design decision 2) — a probabilistic structure (counting
  Bloom filter or similar) is the natural next step if re-admission-after-eviction turns out to
  matter for real workloads, without paying an exact-membership check on every slow-tier write.
- No dedicated test yet for a multi-key single-step promotion/demotion cascade with mixed
  small/huge object sizes (same caveat noted for the other two hybrids above).

## Performance: parallelizing `apply_tier_migrations`'s physical migration copies

Per profiling earlier in this investigation (real `perf` capture against the actual
`paper-benchmark-cxl` benchmark, not just this crate's own tests) `TieredBuffer::new_slow`
(the demotion path's PMEM byte copy, run inside `PolicyWorker::apply_tier_migrations`) consumed
~26% of total process CPU time under a demotion-heavy workload — the single `PolicyWorker`
background thread doing this work sequentially, one migration at a time, was a genuine bottleneck
(spawning additional `PolicyWorker` threads doesn't help — each `WorkerManager` constructor
registers exactly one). Per the user's explicit request ("look at parallelizing the migration copy
for demotions or promotions ... but make sure to always respect the fast tier dram threshold so
demotions should occur before promotions"), all three hybrids' `apply_tier_migrations` siblings
(`worker/policy/mod.rs`) now apply a batch's physical migrations in two sequential phases instead
of one interleaved loop: `migrations` is partitioned into `demotions` (`Tier::Slow`) and
`promotions` (`Tier::Fast`); `demotions.into_par_iter().for_each(...)` (via `rayon`, already a
crate dependency — `mini_stack/manager.rs` already used it) runs every demotion in the batch
concurrently and **fully returns** before `promotions.into_par_iter().for_each(...)` starts. This
is a strict, batch-wide barrier — stronger than the earlier per-push-order fix (commit `620f376`,
"Apply demotions before the promotion that triggers them, not after"), which only ordered
migrations pairwise as pushed; this guarantees *every* demotion in a call has physically freed its
fast-tier bytes before *any* promotion in that same call begins allocating new ones, regardless of
batch size or push order. `LfuHybridStack`'s sibling keeps its existing `drain_demotions()`-based
counting (a `Tier::Slow` entry there isn't always a genuine demotion — see that method's doc
comment) unchanged, just moved the physical copy onto the parallel demotion phase.

**A real, non-obvious compile problem this hit, and why the fix isn't `K: Send + Sync` bounds
threaded through the call chain:** `rayon`'s `for_each` requires the closure (and its captures) to
be `Send + Sync`, which requires `ObjectMapRef<K, V>: Sync`. Naively adding `K: Send + Sync, V:
Send + Sync` to `apply_tier_migrations`'s signature doesn't work in isolation — `run()` (the only
caller, on `impl Worker for PolicyWorker<K, V>`) doesn't have those bounds either, and adding them
*there* cascades outward through every `WorkerManager` constructor (used by every policy, not just
the three hybrids), since `PolicyWorker<K, V>: Send` is otherwise satisfied for free via an
existing, pre-this-session `unsafe impl<K, V> Send for PolicyWorker<K, V> where K: TypeSize, V:
TypeSize {}` near the bottom of `worker/policy/mod.rs` — this crate's established pattern
(mirrored at the top level by `lib.rs`'s unconditional `unsafe impl<K, V, S> Send`/`Sync for
PaperCache<K, V, S>`) is to assert thread-safety unconditionally at a few key boundary points
rather than thread `Send`/`Sync` bounds through ~5000 lines of generic worker/policy-stack code,
trusting that every concrete `K`/`V` this crate is ever built with (integer-typed keys,
`TieredBuffer`) is genuinely thread-safe in practice. Threading explicit bounds through the call
chain would have been a much larger, inconsistent-with-existing-style change, and risked breaking
other, non-hybrid `K`/`V` combinations that currently rely on the same unconditional-`unsafe impl`
escape hatch. Instead, added a second, narrowly-scoped wrapper following the *same* established
pattern: `AssertSync<T>(T)` with unconditional `unsafe impl<T> Send`/`Sync`, used only to wrap
`&self.objects` for the duration of the parallel closures inside `apply_tier_migrations`. Safety
argument is the same one already given for `PaperCache: Sync` (`lib.rs`): all access goes through
`DashMap`'s own per-shard locking (`get_mut`), so no unsynchronized mutable access is actually
exposed — only the compile-time bound was missing.

A second, genuinely subtle compile issue surfaced *after* adding `AssertSync`: wrapping
`&self.objects` and then calling `objects.0.get_mut(&key)` (direct tuple-field access) still
failed with the same `Send`/`Sync` errors, because Rust's disjoint-closure-capture analysis (RFC
2229) captures a direct field projection (`wrapper.0.foo()`) as just the *inner* field's type,
bypassing the wrapper's `unsafe impl` entirely — the closure captured `&Arc<DashMap<...>>`
directly, never actually capturing the `AssertSync` value itself. Fixed by replacing the public
`.0` field access with a private `.get(&self) -> &T` accessor method: a method call forces the
closure to capture the whole `AssertSync<T>` receiver (method resolution isn't as transparent to
the disjoint-capture analysis as a syntactic field projection is), which does pick up the
unconditional `Send`/`Sync` impls as intended.

Verified: `cargo +nightly build --features {lru_hybrid_cache,lfu_hybrid_cache,two_q_hybrid_cache}`
each compile clean; unit tests for all three hybrids pass (25/25 lru, 27/27 lfu, 21/21 two_q,
filtered to each's own module); all three real-PMEM integration suites pass twice in a row (not
flaky): 15/15 (+1 ignored bench) lru, 19/19 lfu, 18/18 two_q — matching this file's previously
documented baselines exactly, confirming no regression from the physical-migration restructuring.
Also confirmed representative non-hybrid feature builds (`hybridcache`, `all_dram`,
`key_value_pmem`) still compile clean, since the only unconditional (non-`#[cfg]`-gated) change is
the addition of the `rayon` import and the new `AssertSync` type, both inert unless a hybrid
feature is active.

Not yet done (at the time this section was first written): re-verifying against the real
`paper-benchmark-cxl` benchmark to directly confirm the parallelization reduces `PolicyWorker`'s
wall-clock migration latency / CPU burden under real concurrent load — see the dedicated section
below, which resolves this.

## Combining each hybrid stack's separate per-key maps into one

Per the user's question "why not just one queue: for the tier mapping, queue mapping and size
mapping" — `LruHybridStack`/`LfuHybridStack` (`sizes`+`tiers`) and `TwoQHybridStack`
(`queue`+`main_tiers`+`sizes`) each collapsed their separate per-key maps into a single
`entries: HashMap<HashedKey, Entry>` per stack (`LruEntry`/`LfuEntry { tier, size }`,
`TwoQEntry { queue, tier: Option<Tier>, size }`). Motivated by checking every call site in all
three stacks and finding no operation ever wanted just one of the separate maps in isolation —
`insert` touches queue+size together, `remove` touches all of them, etc. — so the split was pure
historical accident (`TwoQHybridStack` was built by extending `LruHybridStack`'s `tiers`+`sizes`
shape with a third map bolted on, not a from-scratch design).

Real, measured (not `size_of`-estimated) structural sizes confirmed via a standalone Rust check:
`Tier` and `Option<Tier>` are both niche-optimized to 1 byte, so all three combined entry structs
(`LruEntry`, `LfuEntry`, `TwoQEntry`) are 8 bytes, and `(HashedKey, Entry)` pairs to exactly 16
bytes for all three — the same pair size either of the two separate maps LRU/LFU used to cost
individually. This let `object/overhead.rs`'s constants be updated precisely rather than re-guessed:
`get_policy_overhead`'s `TwoQHybrid` arm dropped from 134 to 86 bytes/object (three maps to one is
the biggest single win), `LruHybrid` 81→85, `LfuHybrid` 137→113; the more precise
`get_hybrid_dram_shared_overhead` constants (used for the fast-tier DRAM-budget reservation, LRU/LFU
only) dropped 84→64 (LRU) and 113→93 (LFU).

Verified: all 6 build combos (3 policies × with/without `eviction_stacks_pmem`) compile clean; unit
tests 25/25 (LRU), 27/27 (LFU), 21/21 (2Q) both ways; real-PMEM integration suites 15/15+1 ignored,
19/19, 18/18, each run twice, matching documented baselines exactly. One non-reproducible SIGSEGV
was observed during unit testing (1 out of ~26 repeated runs of 2Q's `--lib` suite specifically) —
crash signature (two threads at an identical instruction pointer, small offset from null) points at
allocator-level concurrency, not application logic, and is architecturally expected to concentrate
on 2Q specifically since 2Q's admission always writes to PMEM synchronously (unlike LRU/LFU, whose
unit-level tests mostly avoid touching the real allocator) — treated as a rare, pre-existing
environmental flake given the diff contains no unsafe code or new allocator-touching logic, and the
dedicated integration suite (which exists specifically to exercise real concurrent PMEM load safely,
via its `ensure_pmem_allocator_warm()` warm-up) passed cleanly twice.

### Real-DRAM impact, measured via a controlled before/after benchmark A/B

Per explicit request to verify this against the real benchmark (not just assume the byte-count
math translates to real savings): built `paper-benchmark-cxl` twice against `paper-cache-cxl` — once
pinned to commit `4844716` (immediately before this consolidation) and once at current HEAD
`6e8be5b` (after) — identical binary otherwise, identical workload. Used a purpose-built synthetic
trace (2,000,000 distinct keys, 200-byte values, matching the crate's real 25-byte trace record
format) rather than the standard traces, since metadata is only a meaningful fraction of footprint
for small objects — the real traces (`final_traces/*.bin`) average ~16 KB objects, where metadata
is under 1% of the total and this change would be statistically invisible. `fast_tier_size` forced
down to 50 MB (vs. ~400 MB of raw value data) via a temporary benchmark-side edit to force real
demotion churn; single client; `/proc/<pid>/numa_maps` node0 sampled every second to settlement.

| | before (`4844716`) | after (`6e8be5b`) |
|---|---|---|
| Settled real DRAM (node0) | 1,717.3 MB | 1,588.5 MB |
| SET throughput | 1,220,487/sec | 1,251,657/sec |

**128.8 MB less real DRAM for the identical 2M-object workload — a 7.5% reduction, ~67.5
bytes/object.** Larger than the ~20–24 bytes/object predicted from the `hashbrown_entry_cost`
arithmetic alone (see the derivation block in `object/overhead.rs`) — consistent with eliminating
an *entire separate* `HashMap` (its own `RawTable` allocation, with its own capacity-doubling
overprovisioning) mattering more in practice than the "amortized per-entry" formula captures.
Throughput was unaffected (marginally better, consistent with fewer hashtable lookups per
operation). No crashes in either run (`dmesg` checked against wall-clock timestamps for both).

**Overage ratio against the configured 50 MB fast tier: 34.3x (before) → 31.8x (after).** A real,
measured improvement, but a modest dent, not a fix — the dominant driver of that 30x+ overage
remains the already-documented TBB allocator retention behavior (see "Investigation: real DRAM
usage vs. `fast_tier_size`" above), which this change does not touch at all (it only reduced the
fixed *metadata* cost per tracked object, not the *value-byte* retention behavior that dominates
the overage at this scale). Metadata was always a small fraction of the total next to 2M × 200B of
retained value bytes, so a large move in the 30x number was never on the table here.

## Performance follow-up: the migration-copy parallelization is a net loss at the standard test concurrency

Per explicit skepticism ("i dont think it did if it didnt get rid of it [the DRAM overage]") —
first, a scope clarification worth stating precisely: the parallelization (commit `d3050f5`) was
never aimed at the DRAM-retention/overage problem documented above. It targeted a completely
different, already-separately-confirmed finding — `perf` profiling showing `TieredBuffer::new_slow`
(the demotion path's PMEM byte copy) consuming ~26% of total process CPU time under a
demotion-heavy real workload. Whether it moved that CPU number is an independent question from
whether real DRAM tracks `fast_tier_size`, and this section answers the CPU/latency question
directly rather than assuming the earlier reasoning ("profiling showed 26% CPU, so parallelizing
the hot spot should help") actually held up under real measurement.

**It didn't hold up — not at the concurrency this investigation has been testing at.** Built
`paper-benchmark-cxl` twice more, pinned to `18cf802` (immediately before the parallelization,
sequential migration copy) and `d3050f5` (immediately after, parallel two-phase copy) — isolating
just this one change, since `4844716`→`6e8be5b` (the map consolidation) touched the same method
later and would have confounded the comparison. Real trace (`standard_web.bin`, 14M accesses), same
demotion-heavy config (20GB cache / 4GB fast tier) as the DRAM investigation above, `/usr/bin/time
-v` for real process CPU time (not just wall clock), `dmesg` cross-checked against wall-clock
timestamps for both runs (clean, no crashes):

| | sequential (`18cf802`) | parallel (`d3050f5`) | delta |
|---|---|---|---|
| **-c 1** (single client — the config this whole investigation has used) | | | |
| total CPU (user+sys) | 334.42s | 489.37s | **+46.3%** |
| wall clock | 181.04s | 194.18s | **+7.3%** |
| **-c 8** (multiple concurrent clients — the scenario the original `perf` profiling used) | | | |
| total CPU (user+sys) | 472.82s | 495.10s | +4.7% |
| wall clock | 176.73s | 150.74s | **-14.7%** |

At `-c 1`, parallelizing made things **worse on both axes** — 46% more total CPU burned and 7%
slower wall clock. This is consistent with the crate's own per-event migration scheduling (see
"Applied per-event rather than once after the whole batch drains" in `apply_tier_migrations`'s
comment): each `WorkerEvent` typically produces a tiny migrations batch (often 0–2 entries), and
spinning up `rayon`'s work-stealing scheduler for a batch that small pays real coordination
overhead without enough parallel work to amortize it — the higher CPU% (252% vs. 184%) confirms
more cores were engaged, but that engagement was net-wasted, not productive.

At `-c 8`, the picture flips: parallel wins meaningfully on wall clock (-14.7%) for a modest extra
CPU cost (+4.7%). With more concurrent client threads, more events queue up between `PolicyWorker`
poll iterations, producing genuinely larger per-event migration batches (more demotions and
promotions accumulate before each `apply_tier_migrations` call), which is exactly the shape `rayon`
needs to pay off — closer to the original `perf` profiling's own test conditions ("multiple
concurrent rayon worker threads, a real trace").

**Conclusion: the parallelization is conditionally beneficial, not unconditionally beneficial, and
this investigation's own established single-client test methodology is exactly the case where it
loses.** It was landed and verified against this crate's low-concurrency unit/integration tests
only (see the original section above — "not just this crate's own (lower-scale, lower-concurrency)
test suite" was flagged as a gap at the time, now resolved with this real answer). A batch-size
guard (skip `rayon` and apply sequentially below some small threshold, e.g. 2–3 entries) would very
likely recover the `-c 1` regression while keeping the `-c 8` win, but hasn't been implemented —
this section reports the measurement, not a fix.

### Follow-up: the batch-size guard doesn't have a clean answer — the bottleneck is the allocator, not rayon

Implemented the guard flagged as "hasn't been implemented" above (`PARALLEL_MIGRATION_THRESHOLD`,
`apply_migrations_batch` in `worker/policy/mod.rs`, commit `e8dbcc8`), but the small-threshold
premise turned out to be wrong. Direct instrumentation against the real benchmark (temporary atomic
histogram counters over `apply_tier_migrations`' batch sizes, removed again once the data was
collected) showed real per-event migration batches are **bimodal**, not uniformly small: ~50/50
split between 0 and 1 entries, plus a rare (~0.016% of calls) but volume-dominant tail up to 4,899
entries in a single batch — that tail alone accounted for 61% of all migrated bytes in one sample.
Thresholds of 4 and 1,000 both made no measurable difference against unconditional parallelization,
because real batches land either far below or far above either value — there's no "medium" batch
size for a threshold in that range to actually intercept.

Only a threshold *above* the observed max (`10_000`) genuinely serializes every real batch, and
that's what recovered sequential-equivalent CPU/wall-clock numbers at `-c 1`. This directly
confirms the original regression was never about `rayon` scheduling overhead on small batches — it's
that parallelizing the *large* batches (concurrent `TieredBuffer::new_slow` calls, i.e. concurrent
PMEM allocations) contends on the underlying TBB/UMF allocator, which doesn't scale under concurrent
access to begin with (consistent with everything already documented above about this allocator).
The same contention cost is why `-c 8`'s `-14.7%` wall-clock win also evaporates under the
`10_000` threshold — serializing there lands back at sequential-equivalent numbers too, so there is
**no threshold value that helps `-c 1` without also losing `-c 8`'s benefit**: the two scenarios
disagree about whether concurrent allocator access is net-helpful, and a single global constant
can't resolve that per-workload.

Landed `PARALLEL_MIGRATION_THRESHOLD = 10_000` anyway — in practice "always sequential for this
allocator" — rather than deleting the `rayon` path outright, since the mechanism costs nothing when
unused and stays available if a future allocator swap (see the jemalloc-pool retest below, which
did not pan out) ever has different concurrent-access characteristics where parallel migration
copies genuinely pay off. Verified no regressions: all three hybrids' unit tests (25/25 lru, 27/27
lfu, 21/21 two_q) and real-PMEM integration suites (15/15+1 ignored, 19/19, 18/18, each run twice)
pass unchanged — the threshold only changes *how* a batch is applied, never the result.

## Third retest: `umfJemallocPoolOps()` still crashes under real concurrent load — and a methodological near-miss

Per explicit request ("use jemalloc for the fast tier to better align with memory constraints"),
and after being shown the two previously-documented `umfJemallocPoolOps()` crash writeups above and
explicitly choosing "retry anyway," swapped `umf_allocator_wrapper.c`'s active `umf_allocator_init`
(shared by both `HybridObjects`/node 1 and `DRAMObjects`/node 0) from `umfScalablePoolOps()` (TBB)
back to `umfJemallocPoolOps()` a third time, to get a direct, current answer rather than relying on
memory of the prior two failures.

**A genuine methodological mistake, caught before it produced a false conclusion.** The first two
benchmark runs against this change (`-c 8`, `standard_web.bin`, 20 GiB cache) both completed
cleanly — full GET/SET stats, exit 0, no new `dmesg` entries — which would have contradicted both
prior documented crashes. Before trusting that surprising result, checked whether the binary
actually *exercised* the new code: `nm -D` on the tested binary showed only `umfScalablePoolOps`
symbols, no `umfJemallocPoolOps` at all. Root cause: `paper-benchmark-cxl`'s `Cargo.toml` had been
repointed from a git branch to `path = "/home/griff/work/paper-cache-cxl"` (the established pattern
for testing uncommitted local changes), but `cargo build`'s fingerprint for the *new* path-based
source resolved to a build-script `OUT_DIR` whose `umf_allocator_wrapper.o` still dated from
*before* the C-file edit — i.e., `cargo build --release` reported "Finished... in 3.60s" and never
actually re-invoked the C compiler, silently reusing a stale cached object from an earlier session.
Confirmed by comparing file mtimes (`.o` at 08:02, source edit at 08:04) and by `cargo clean -p
paper-cache --release && touch umf_allocator_wrapper.c && cargo build -v`, which forced a genuine
recompile and produced a binary whose `nm -D` correctly showed `umfJemallocPoolOps` and no TBB
symbols. **The first two "successful" runs were invalid — they tested the old TBB binary by
accident, not jemalloc — and must be disregarded**, not reported as a real result.

Re-ran against the verified binary (freshly rebuilt, symbol-checked before running): **crashed on
the very first attempt**, 15.7 seconds in (`Command terminated by signal 11`), `dmesg` confirming
`paper-benchmark[195167]: segfault at 30 ... in libumf.so.1.0.3` at the matching wall-clock
timestamp. This is a third confirmed failure, consistent with (though not identical in signature
to) the two previously documented ones — all three point at the same underlying conclusion: UMF's
jemalloc pool (version 1.0.3) is not safe under this crate's real concurrent access pattern.

Reverted `umf_allocator_wrapper.c` back to `umfScalablePoolOps()` (`git checkout --`, since the
change was uncommitted) and confirmed the crate still builds clean afterward. `paper-benchmark-cxl`'s
`Cargo.toml` was restored to its prior `branch = "libcache8-claude"` state.

**Bottom line, now confirmed a third time**: do not re-enable `umfJemallocPoolOps()` without a fix
from upstream UMF. TBB (`umfScalablePoolOps()`, `KeepAllMemory=1`) remains the only pool backend
proven stable under real concurrent load in this environment — this investigation's real-DRAM-vs-
`fast_tier_size` gap (documented at length above) stays an accepted, known cost rather than
something fixable by an allocator swap.

**Process lesson for future sessions**: when repointing a downstream crate's dependency to a local
`path` after editing C source compiled via a `build.rs` `cc::Build`, don't trust a suspiciously fast
`cargo build` completion as proof the change was picked up — verify the *linked binary* actually
references the expected symbols (`nm -D <binary> | grep <expected_symbol>`) before treating a
benchmark run's outcome as meaningful, especially when the result contradicts prior findings. A
successful run that silently tested old code is worse than an honest failure, because nothing about
its output looks wrong on its own.
