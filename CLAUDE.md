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

  worker/                     Background-thread machinery. PaperCache::new() spawns the workers
                               that own all mutation of eviction state so the hot get()/set()
                               path stays lock-cheap.
    manager.rs                 WorkerFanout — routes each WorkerEvent to the sub-workers
                                (PolicyWorker, TtlWorker, TieringWorker) that actually consume
                                it, per worker/mod.rs's `Events` subscription masks. NOT a
                                thread: the fan-out runs inline on the calling thread. It was a
                                thread until the GET-path work below; see "Performance: the
                                background event pipeline on the GET path".
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
      trace/                    Optional access-trace recording/replay, the input
                                 reconstruct_policy_stack() replays to rebuild a *different*
                                 policy's stack after a live policy switch. Only spawned when
                                 more than one policy is configured (see trace_is_useful) —
                                 with a single policy, which is every hybrid cache, no switch is
                                 reachable and the trace would be written and never read.
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

## Removed the migration-copy parallelization entirely — inline sequential in the policy worker

Per explicit request ("get rid of using background threads for the eviction and promotion... just
have the policy worker do it like it was done initially... doing it with background appears to be
hurting performance"), the `rayon`-based parallel migration copies (commit `d3050f5`, later gated
behind `PARALLEL_MIGRATION_THRESHOLD = 10_000` in `e8dbcc8`) were removed outright. All three
hybrids' `apply_tier_migrations` siblings (`worker/policy/mod.rs`) now apply demotions and
promotions with a plain sequential `into_iter().for_each(...)` directly on the `PolicyWorker`
thread — the original pre-`d3050f5` shape. Deleted along with them: the `apply_migrations_batch`
dispatch helper, the `PARALLEL_MIGRATION_THRESHOLD` const, the `AssertSync<T>` cross-thread wrapper
(only needed to share `&self.objects` across `rayon` workers), and the now-unused `use
rayon::prelude::*` import (`rayon` stays a crate dependency — `mini_stack/manager.rs` still uses it).

This folds into an actual code removal the conclusion the two sections above had already reached
from measurement: at `-c 1` (the standard single-client config) unconditional parallelization was
**+46.3% CPU / +7.3% wall clock**, and no `PARALLEL_MIGRATION_THRESHOLD` value recovered `-c 1`
without also losing the narrow `-c 8` wall-clock win — because the bottleneck is TBB/UMF allocator
contention under concurrent access, which parallelizing PMEM-allocating migration copies can only
worsen. The measurement tables in those sections are kept as the historical record of *why* this
was removed.

**Retained deliberately: the demotion-before-promotion ordering.** Each sibling still `partition`s
the batch into demotions (`Tier::Slow`) and promotions (`Tier::Fast`) and runs every demotion
before any promotion — this is a correctness guarantee (a promotion must not allocate fast-tier
DRAM ahead of the demotion freeing room for it), an earlier explicit user requirement, and now just
a sequential two-pass loop rather than a `rayon` barrier. Everything else is unchanged: the
per-event call cadence (`run()`, gated on a non-empty drain), the physical `Object::set_data`
reallocation, all stats/gauge recording, LFU's `drain_demotions()`-based demotion count and
`admission_latched` mirror. `apply_evictions` was already sequential and is untouched.

Verified: all three hybrids' unit tests (25/25 lru, 27/27 lfu, 21/21 two_q) and real-PMEM
integration suites (15/15 +1 ignored lru, 19/19 lfu, 18/18 two_q, each run twice) pass unchanged —
the removal only changes *how* a batch is applied (sequential vs. the effectively-already-sequential
threshold path), never the result.

## Generics unification: collapsed the storage-matrix and hybrid-cache `impl` block duplication

Per explicit request ("generate a plan to use generics for the all_dram ... as well as all the
hybridcache designs"), `lib.rs` had grown to 7,564 lines specifically *because* the "same API,
different backing storage" design documented at the top of this file was written out longhand —
once per Cargo-feature storage combination — rather than behind generics, on the original reasoning
that comparing storage strategies shouldn't cost a shared abstraction's overhead. That tradeoff had
started costing real maintenance weight (any `set()` bugfix needed hand-copying into 6-12 places),
so this was revisited on a branch (`generics-unification`, off `jemalloc-extent-hooks-cxl`), in
three parts, after two scoping questions were confirmed with the user: **"do it broadly... but
forget all the flatmap stuff.. in fact all that flatmap stuff can be removed"** (broad unification,
plus remove FlatMap entirely rather than migrate it), and, for the four hybrid-cache designs,
**"Keep compile-time exclusivity, just dedupe source"** (no runtime policy selection — still exactly
one Cargo feature compiles per build, same mutual-exclusion model as everything else in this crate —
purely a source-level dedup).

**Part 1 — removed the FlatMap storage backend entirely**
(`flatmap_dram`/`flatmap_pmem`/`global_flatmap_dram`/`global_flatmap_pmem`, `src/flatmap.rs`,
`tests/global_flatmap_integration.rs`, and every FlatMap-gated `impl PaperCache<K, BufferDRAM/
BufferPMEM, S>` block, cfg branch, and doc section) — this was also a scope-reducer for Part 2,
since without FlatMap the object-map storage axis has exactly two shapes left (`DashMap`; `RwLock<
HashMap<..., A>>` generic over an allocator `A`) instead of five. Caught one genuine pre-existing bug
while verifying (unrelated to FlatMap): `GlobalHwPerfCounters` (`hw_perf_counters.rs`) referenced a
`dashmap_counters` field it never actually declared, so any `hw_perf` build without also enabling
`hashbrown_dram`/`global_hashtable_pmem` (e.g. `all_dram,hw_perf`) failed to compile even before this
session — confirmed via `git show HEAD` — fixed in passing by adding the missing field.

**Part 2 — collapsed the five remaining non-hybrid `impl PaperCache<K, V, S>` blocks (all_dram,
key_value_pmem, global_hashtable_pmem, hashbrown_dram, and key_value_pmem+global_hashtable_pmem
together) into two**, one per object-map shape, via two new traits: `ObjectStore<K, V>`
(`src/object_store.rs` — `get_ref`/`get_mut_ref`/`insert`/`clear`/`len`, using return-position `impl
Trait` in traits so DashMap's own `Ref`/`RefMut` guards satisfy it for free, and a small
`RwLockObjectRef`/`RwLockObjectMut` wrapper re-indexes the RwLock shape's guard on `Deref` to match)
and `ValueBuffer` (`src/value_buffer.rs` — one `from_bytes(&[u8]) -> Self`, replacing four different
existing spellings of "copy these bytes into a fresh buffer" across the five blocks: `Box::
clone_from_ref`, `Box::clone_from_ref_in`, `value.into()`, `value.to_vec_in(Hybrid).into_boxed_slice()`).
`key_value_pmem`'s `enable_tiering_manager`/`sets_dram` tiering-manager logic (including the
`sets_dram` closure's PMEM allocation, now via `V::from_bytes` instead of a hardcoded `BufferPMEM`
literal) stayed as internal `#[cfg]` branches inside the shared bodies rather than being genericized
away, since `TieringManager<K, V>` is already generic and this only matters when `key_value_pmem`
pins `V` to `BufferPMEM` at the real call site — confirmed this compiles correctly for both `V`s
without further changes. `new_with_eviction_callback` (`hybridcache`'s always-`BufferDRAM` small
tier) stayed its own small non-generic block rather than being forced generic. One real bug the merge
surfaced: `Arc<V>::as_ref()` resolves to `Arc`'s own blanket `AsRef<V>` (giving `&V`) *before* ever
reaching `V`'s `AsRef<[u8]>`, unlike the old concrete `Box<[u8]>`/`Box<[u8], Hybrid>` code where
`&Box<[u8]>` auto-derefs through to `[u8]` for the following `.to_vec()` — a bare generic `V` has no
such deref chain, so this needed an explicit `AsRef::<[u8]>::as_ref(&*arc_val).to_vec()` at the three
call sites it affected. `lib.rs`: 6,620 → 4,854 lines.

**Part 3 — collapsed the four hybrid-cache designs' (`lru_hybrid_cache`/`lfu_hybrid_cache`/
`two_q_hybrid_cache`/`fifo_hybrid_cache`) `impl PaperCache<K, TieredBuffer, S>` blocks into one**,
via a new `HybridPolicy` trait (`src/hybrid_policy.rs`) plus one small marker struct per design
(`LruHybridPolicy` etc., one in each design's own `mod.rs`), selected per build through a `#[cfg]
type ActiveHybridPolicy = ...;` alias (mirroring the crate's existing `ObjectMapRef`/`Hybrid`/
`BufferDRAM` pattern). Confirmed via direct diff that the four ~380-line blocks differed only in:
which `PaperPolicy` variant gets seeded; the `Stats` type and its `{name}_hybrid_stats()` accessor's
name; one admission-rule branch inside `set()` (lru always fast; lfu conditionally slow via
`is_new && lfu_hybrid_admission_latched()`; two_q always slow; fifo looks up an *existing* key's
current tier directly — the one design whose rule isn't purely a function of "is this key new"); and
`two_q_hybrid_cache`'s extra `k_in: f64` constructor parameter (`HybridPolicy::ExtraConfig`, `()` for
the other three). One shared generic block now carries every method except `new`/`with_hasher`/the
stats accessor; four tiny per-feature blocks (kept in `lib.rs` itself rather than distributed into
each hybrid module, simpler since inherent impls don't need to be co-located with their type) supply
just those three, preserving each feature's distinct external name/signature (`paper-server`/
`paper-benchmark-cxl` compatibility) unchanged. `lib.rs`: 4,872 → 3,914 lines.

**Net result**: `lib.rs` 7,564 → 3,914 lines (48% reduction) across all three parts, zero change to
the public API surface. Verified end-to-end: every storage-combo/hybrid-feature build listed in the
"Layout"/feature-flag sections above still compiles and passes its full unit suite unchanged (83/81/
84/81 non-hybrid; 91/91/91/92/92 hybrid, counts differ from the pre-refactor per-module baselines
only because `--lib` now runs the whole crate's test binary rather than a module-filtered subset);
all four hybrid integration suites pass twice in a row unchanged (15/15+2 ignored lru, 19/19 lfu,
18/18 two_q, 14/14 fifo); the `hybridcache`/`tiering`/`eviction_stacks_pmem`/`jemalloc_cxl_slow_tier`
untouched-feature regression builds and the `lru_hybrid_cache`+`lfu_hybrid_cache` mutual-exclusion
`compile_error!` all still behave identically. `paper-benchmark-cxl` (external consumer, `path`-
pointed at this checkout) builds clean in release mode against both its currently-configured feature
set (`lru_hybrid_cache,jemalloc_cxl_slow_tier`) and `all_dram` — confirmed the public API surface it
depends on is unaffected — but a full trace-driven run wasn't performed in this session (no `.bin`
trace file was available in this sandbox to drive one); this is a lower-risk gap than it would be for
a logic change, since nothing in `worker`/`allocator`/policy-stack code was touched — only the
`PaperCache` API-layer inherent-method bodies were restructured, calling the exact same underlying
functions as before.

Two small, pre-existing (not introduced by this refactor, confirmed via side-by-side diff against
the pre-Part-2 commit) issues were found and left alone as out of scope: a bare `key_value_pmem,
sets_dram` build (without `enable_tiering_manager`) fails with a `WorkerManager::new` argument-count
mismatch in `worker/manager.rs`; `key_value_pmem,enable_tiering_manager`/`multitiering,key_value_pmem`
builds fine but their `--lib` test run hits an unrelated compile error inside `tiering/manager.rs`'s
own test module (a `TieringManager::with_defaults()` type-inference failure and a `TieringConfig`
missing-field literal). Also pre-existing: the doc-comment examples on the merged non-hybrid blocks
(`PaperCache::<u32, u32>::new(...)`) were already failing as doctests before this refactor, since
`u32` never satisfied the concrete `BufferDRAM`/`BufferPMEM` bound the original non-generic blocks
required either.

## Cleanup pass: removed rdtsc/phase profiling, `sets_dram`, `hw_perf`, and the old non-TBB/non-jemalloc allocators

Per explicit request, four unrelated pieces of legacy/experimental machinery were removed outright
from the `generics-unification` branch, each verified independently (build + `--lib` test for
`all_dram`/`key_value_pmem`/`global_hashtable_pmem`/`hashbrown_dram`, plus a full untouched-feature
regression build sweep and all four hybrid integration suites, after all four removals):

- **rdtsc/phase-analysis instrumentation**: deleted `src/rdtsc_probes.rs` entirely (the `PhaseStats`
  histogram type, all 13 `PHASE_*` statics, `rdtsc()`, `calibrate_tsc_hz`/`calibrate_probe_overhead`,
  `report_get`/`report_set`) and the one place it was still wired up post-Part-2 — the RwLock-shape
  `get()`'s `not(key_value_pmem)` branch (added during the generics-unification Part 2 merge,
  documented there as "ad hoc... predates this merge") — collapsing that method back to the single
  plain body every other shape already had.
- **`sets_dram`**: removed the Cargo feature and every line gated on it. This was woven through five
  files: `lib.rs` (the `set()` early-return branch that bypassed the shared object map entirely,
  delegating straight to `tiering_manager.set_dram`, and the constructor's `Arc::new_cyclic` backfill
  closure), `tiering/manager.rs` (the `PmemBackfillJob` struct, the `pmem_tx`/`_pmem_consumer`/
  `pending_jobs`/`sync_persist_fn` fields, `Tier::DramOnly` and every match arm referencing it across
  8 call sites, `register_object`'s sets_dram-only arm, `spawn_pmem_consumer`, both `new_with_backfill`
  arms, `set_dram`, `mark_persisted`, and the `is_dram_only`/`force_sync_persist` impl block --
  confirmed via a live/dead block-comment nesting scan, using Python to count `/*`/`*/` depth, that
  none of this touched the file's own large pre-existing dead-code comment blocks), `worker/manager.rs`
  (the sets_dram-specific 5-arg `WorkerManager::new` variant -- this is also where the pre-existing,
  already-documented `key_value_pmem,sets_dram`-without-`enable_tiering_manager` arg-count bug lived;
  moot now, the whole variant is gone), `worker/policy/mod.rs` (`new_with_tiering`, the
  `tiering_manager` field, and the priority-demotion check in `apply_evictions`), and
  `worker/tiering.rs` (a second, entirely duplicate `TieringWorker` impl block gated on `sets_dram`).
- **`hw_perf`**: removed the Cargo feature, the `perf-event` dependency, `src/hw_perf_counters.rs`
  (`get_hw_counters`/`get_hw_hashmap_stats`/`print_hw_perf_stats`/`measure_operation`/
  `HwHashMapStats`/`HwPerfMeasurement`), and its one `#[cfg(test)]` module in `lib.rs`.
- **Old non-TBB/non-jemalloc allocators**: removed `pmem_region_alloc`, `region_hybrid_allocator`,
  and `devdax_bump` (Cargo features + `RegionHybrid`/`DevDaxBump`/`DaxPtr` from `src/allocator.rs`),
  and `DAXPMEM` (never Cargo-feature-gated as reachable at all -- confirmed dead, its only reference
  was a commented-out `//use crate::allocator::DAXPMEM as Hybrid;` in `lib.rs` -- so this one wasn't
  "old" in the sense of a still-selectable backend, just genuinely unused code). Used the same
  block-comment-depth scan to confirm large stretches of `RegionHybrid`/`DevDaxBump` were *already*
  dead/commented-out duplicates before touching anything (this crate's established "second copy for
  reference" pattern, same as the `HybridObjects`/`UnifiedAllocator` dead block already documented
  above -- left untouched, unrelated to this cleanup). `lib.rs`'s `Hybrid` type-alias cascade (5
  cfg arms picking between `DevDaxBump`/`RegionHybrid`/`HybridObjects`) collapsed to one unconditional
  `pub(crate) use crate::allocator::HybridObjects as Hybrid;` -- every PMEM feature now routes through
  the same TBB-backed UMF pool (`HybridObjects`) or, for eviction-stack metadata specifically, the
  separate jemalloc_cxl extent-hooks allocator (`EvictionStackAllocator`/`SlowTierJemallocAllocator`,
  untouched by this cleanup). Also deleted `tests/pmem_region_alloc_integration.rs` (tested a feature
  that no longer exists) and a byte-identical-both-arms `#[cfg(any(pmem_region_alloc,
  region_hybrid_allocator))]` branch in `lfu_stack.rs`'s eviction-stack default capacity (both arms
  already returned the same `50_000_000` -- collapsed to one unconditional value regardless of the
  feature removal).

Verified: all four non-hybrid storage combos and all four hybrid-cache features build clean and
pass their full `--lib` suites unchanged (same counts as every prior checkpoint in this document);
untouched-feature regression builds (`eviction_stacks_pmem`, `hybridcache`, `tiering`,
`multitiering`, `jemalloc_cxl_slow_tier`) all still build clean; all four hybrid integration suites
(15/15+2 ignored lru, 19/19 lfu, 18/18 two_q, 14/14 fifo) pass unchanged; `hybridcache_integration.rs`
shows the same already-documented non-deterministic timing flakiness (not a regression). Confirmed
(via `git stash`, rebuilding the pre-cleanup commit) that a bare no-features `cargo build` already
failed identically before this cleanup too -- this crate has never supported building with zero
features selected; not something introduced here, and out of scope to fix.

## Removed the original S3-FIFO hybridcache module entirely

Per explicit follow-up request, `hybridcache` (`S3FifoHybridCache<K>`, `HybridCacheConfig`,
`HybridCacheStats`) -- the small-DRAM/far-PMEM two-*instance* design that predates and motivated
`lru_hybrid_cache`/`lfu_hybrid_cache`/`two_q_hybrid_cache`/`fifo_hybrid_cache`'s single-instance
designs -- was removed outright: `src/hybridcache/` (whole module), the `hybridcache` and
`far_tier_pmem_evst_hash` (which only ever existed to extend `hybridcache`) Cargo features,
`tests/hybridcache_integration.rs`, and every piece of wiring that existed solely to support it:
`PaperCache::new_with_eviction_callback` (the `BufferDRAM`-only constructor, `lib.rs`),
`WorkerManager::new_with_eviction_callback` (`worker/manager.rs`), `PolicyWorker::
new_with_eviction_callback` plus the `eviction_callback` field and the `apply_evictions` callback
invocation (`worker/policy/mod.rs`), and a dead `#[cfg(feature = "hybridcache")] mod
hybridcache_promotion_tests` left over in `tests/tiering_integration.rs` (referenced a module that
no longer exists, would never have compiled once the feature was gone). Cleaned up the doc-comment
cross-references describing the other four hybrid designs by contrast with `hybridcache` (`size.rs`,
`tiered_buffer.rs`, `lru_hybrid_cache`/`lfu_hybrid_cache`/`fifo_hybrid_cache`'s module docs, `status.rs`,
`s_three_fifo_stack.rs`, both remaining hybrid integration test files' doc comments) and the
corresponding sections of `FEATURE_FLAGS.md` and `Cargo.toml`'s comments -- left `HYBRID_CACHES.md`/
`LRU_HYBRID_CACHE.md` (the two dedicated design docs, which discuss `hybridcache` extensively as
historical design context for decisions already made and shipped) untouched, matching this file's
own convention of not rewriting historical narrative after the fact.

Verified: all four non-hybrid storage combos and all four hybrid-cache features build clean;
`cargo build --features hybridcache` now correctly reports "the package 'paper-cache' does not
contain this feature" (feature genuinely gone, not just broken); untouched-feature regression
builds (`eviction_stacks_pmem`, `tiering`, `multitiering`, `jemalloc_cxl_slow_tier`) still build
clean; all four hybrid-cache features' `--lib` suites and real-PMEM integration suites pass at
their established baselines unchanged (91/91/92/92 unit; 15/15+2 ignored lru, 19/19 lfu, 18/18
two_q, 14/14 fifo integration); `tiering_integration.rs`'s remaining 10 tests (the
`hybridcache_promotion_tests` module removed, `tiering_tests`/`tiering_pmem_key_tests` untouched)
pass unchanged.

## Removed the `tier_allocator` crate: it still used two allocator instances under the hood

A prior session added a standalone `umf_tier_allocator` crate (`Add tier_allocator:
runtime-parameterized NUMA-tier allocation via UMF`, then `Unify tier_allocator into one
mechanism: #[global_allocator] + explicit alloc_on`) and wired it into all four hybrid caches'
`TieredBuffer` (`Integrate tier_allocator into TieredBuffer for all four hybrid caches`), replacing
`Fast`'s reliance on `DRAMObjects` (node 0) and `Slow`'s `Box<[u8], Hybrid>` (`HybridObjects`, node
1) with `tier_allocator::NumaAllocator`/`tier_allocator::alloc_on` against a shared per-node
registry. It also grew two optional alternate UMF pool backends (`umf_jemalloc_pool`,
`umf_disjoint_pool`), both already documented elsewhere in this file as confirmed unsafe under
real concurrent load.

Per explicit user request, checked whether this crate delivered anything over what already
existed, given the suspicion that it "still uses two allocator instances under the hood." It did:
`umf_tier_allocator/src/registry.rs`'s `REGISTRY` is a per-NUMA-node array — `pool_for_node(0)` and
`pool_for_node(1)` each lazily construct and cache their own independent `TierAllocator` (i.e. UMF
pool). For the two-tier hybrid caches (fast tier pinned to node 0, slow tier to node 1), this means
exactly two live pool instances, structurally identical in shape to the `DRAMObjects` (node 0) +
`HybridObjects` (node 1) pair the crate already had before `tier_allocator` was added — same
default backend (`umfScalablePoolOps`, TBB), same one-pool-per-node property, just reimplemented in
a second crate with runtime NUMA-node parameterization neither tier actually needed (both nodes are
hardcoded constants at every real call site, 0 and 1). The crate's own doc comments (`tiered_buffer.rs`'s
old module doc, `lib.rs`'s global-allocator comment) had framed "both access patterns resolve to
the exact same per-node pool" as an improvement over "two independent allocator instances" — but
that framing compared the two *access patterns for a single node* (implicit `GlobalAlloc` vs.
explicit `alloc_on`), not the two *tiers*, which were never sharing a pool and structurally
couldn't (different NUMA nodes require different `mbind` targets). So the premise was confirmed:
the crate added an entire second UMF/TBB integration without changing the number of pool instances
the four hybrid caches actually run with, or any other externally-observable property.

Removed entirely: `umf_tier_allocator/` (whole crate directory), the `tier_allocator` path
dependency and its `dep:tier_allocator` requirement on all four hybrid-cache features, and the
`umf_jemalloc_pool`/`umf_disjoint_pool` Cargo features (both were solely alternate pool backends
*within* `tier_allocator`, meaningless without it). `TieredBuffer::Slow` (`src/tiered_buffer.rs`)
reverted to `Box<[u8], Hybrid>` (`Hybrid` = `HybridObjects`, the same alias every other PMEM feature
already uses); `TieredBuffer::Fast`'s `Clone` impl simplified to a plain `buffer.clone()` for the
`Hybrid` case, since `Box<[u8], Hybrid>: Clone` already holds for free via this crate's existing
`clone_from_ref` nightly feature gate (`Hybrid`/`HybridObjects` is `Clone + Copy`) — no manual
duplicate-via-explicit-allocator dance needed, unlike `TierBuffer`, which required one.
`lib.rs`'s `#[global_allocator]` selection collapsed back to unconditionally `DRAMObjects` whenever
`jemalloc_cxl_slow_tier` isn't active (previously branched on whether a hybrid-cache feature was
enabled, to pick between `DRAMObjects` and `tier_allocator::NumaAllocator`). The
`jemalloc_cxl_slow_tier` alternate backend (a genuinely different, unrelated mechanism -- one
jemalloc instance for both tiers, no UMF at all) is untouched and still available.

Verified: all four hybrid-cache features (`lru_hybrid_cache`/`lfu_hybrid_cache`/
`two_q_hybrid_cache`/`fifo_hybrid_cache`, individually and combined with `jemalloc_cxl_slow_tier`)
and four untouched-feature regression builds (`key_value_pmem`, `all_dram`, `eviction_stacks_pmem`,
`hashbrown_dram`) all build clean; `Cargo.lock` no longer references `tier_allocator`.
`lru_hybrid_cache`'s `--lib` suite (91/91) and real-PMEM integration suite (15/15 +2 ignored) both
pass at their previously-documented baselines, now genuinely exercising `Hybrid`/`HybridObjects`
(this sandbox's real UMF pool + memory-only NUMA node) rather than `tier_allocator`'s reimplementation
of the same thing.

## Removed the deprecated `original` Cargo feature

A standalone `original = []` feature, marked `# Legacy features (deprecated, will be removed)` in
`Cargo.toml`, gated a pre-generics-unification `impl<K, V, S> PaperCache<K, V, S>` block (generic
over any `V: TypeSize + Clone`, `lib.rs`) plus one gated import and one `#[cfg(all(test, feature =
"original"))]` test module. Checked whether it was safe to delete rather than assuming: not pulled
in by `all_dram`, any hybrid-cache feature, or anything else (nothing else's feature list named
it); combined with any real storage feature (e.g. `all_dram`) it failed to compile outright with 16
duplicate-definition errors (`E0592`/`E0034`) against the generics-unification work already done to
`PaperCache`'s other `impl` blocks; built alone it also failed (`E0433`, since this crate has never
supported a bare/no-storage-feature build). No external consumer either -- `paper-benchmark-cxl` has
its own unrelated `original = []` feature in its own `Cargo.toml` but nothing in its `src/` gates on
`feature = "original"`. Removed the feature, the impl block, the gated import, and the test module.
Verified: `all_dram`/`key_value_pmem`/all four hybrid-cache features still build and pass their
`--lib` suites at documented baselines (83/91/91/92/92) unchanged.

## Migrated `eviction_stacks_pmem` off jemalloc_cxl onto `Hybrid`/`HybridObjects`; removed jemalloc_cxl and `jemalloc_cxl_slow_tier` entirely

Per a user question ("is eviction_stacks_pmem now using TBB with NUMA node 1 instead of the
jemalloc extent hooks") that turned out to have the wrong premise -- checked directly rather than
assuming either way: `eviction_stacks_pmem` still depended on `jemalloc_cxl` (`Cargo.toml`'s
`eviction_stacks_pmem = ["dep:jemalloc_cxl"]`) and its metadata allocator, `EvictionStackAllocator`
(`src/allocator.rs`), was still built entirely on `jemalloc_cxl::{CxlAllocator, CxlArena,
TcacheMode}` -- nothing had switched it to TBB. Flagged the distinction the user's follow-up request
("delete the jemalloc extent hooks directory, it's not stable") glossed over: the *documented*
instability (three confirmed crashes, see the `umfJemallocPoolOps()` sections above) is specifically
about UMF's own built-in jemalloc pool backend -- a different mechanism from this repo's own
`jemalloc_cxl` crate (a custom-extent-hooks NUMA/CXL arena allocator). `jemalloc_cxl_slow_tier`
(the other feature depending on the same crate) had its own, different, already-fixed SIGSEGV
history, and its only remaining documented issue is a concurrency ceiling (a clean allocation-
failure abort at high client counts, not a crash) -- which is why it stayed opt-in, not evidence it
"can't be used." `eviction_stacks_pmem`'s own use of the crate had no documented instability at all.
Surfaced this before deleting anything, since removing the crate outright would have broken
`eviction_stacks_pmem` (its only allocator, no fallback) as a side effect, not just
`jemalloc_cxl_slow_tier`. Given three options (cancel; remove both features + the crate; migrate
`eviction_stacks_pmem` to TBB then delete), the user chose the migration.

`HybridObjects` (`src/allocator.rs`, UMF/TBB, NUMA node 1 -- the same allocator that already backs
`BufferPMEM`/every other PMEM feature) already implements `allocator_api2::alloc::Allocator` gated
under `any(feature = "global_hashtable_pmem", feature = "tiering_hashtable_pmem", feature =
"eviction_stacks_pmem")` -- the exact trait `EvictionStackAllocator` implemented for
`PmemHashList`/`PmemVecList`/the per-stack `EntryMap`s (`worker/policy/policy_stack/
pmem_collections.rs` and five call sites: `lru_hybrid_stack.rs`, `lfu_hybrid_stack.rs`,
`two_q_hybrid_stack.rs`, `fifo_hybrid_stack.rs`, `lfu_stack.rs`) to consume. So no new allocator
code was needed: each of those six files' `use crate::allocator::EvictionStackAllocator as Hybrid;`
became `use crate::Hybrid;` (the crate-level alias, already `HybridObjects`) -- eviction-stack
metadata now allocates through the exact same UMF/TBB pool `BufferPMEM` uses, not a second,
independent one. (Coincidentally, `lru_stack.rs`'s existing comment already claimed `PmemHashList`
"routes allocations through `HybridObjects`" -- that was aspirational/wrong before this change,
since `pmem_collections.rs` actually used `EvictionStackAllocator`/jemalloc_cxl at the time; it's
accurate now.)

With `eviction_stacks_pmem` no longer needing `jemalloc_cxl` at all, and `jemalloc_cxl_slow_tier`
(the crate's only other consumer) carrying the same never-fully-proven-safe-under-load status as
the UMF jemalloc pool investigations above, removed `jemalloc_cxl` entirely rather than leaving it
half-used: the `eviction_stack_allocator`, `slow_tier_jemalloc_allocator`, `dram_multi_arena`
(`DramMultiArenaObjects`, the `jemalloc_cxl_slow_tier` `#[global_allocator]`), and the
`numa_arena_pool` helper they all shared (dead once its only two consumers were gone) --
`src/allocator.rs`'s lines 691-1423, its entire back half -- were deleted outright. `tiered_buffer.rs`'s
`jemalloc_cxl_slow_tier` branch of `TieredBuffer::Slow`/`new_slow`/`Clone` was removed, leaving
`Slow(Box<[u8], Hybrid>)` as the only shape (matching every other PMEM feature). `lib.rs`'s
`#[global_allocator]` selection collapsed back to unconditionally `DRAMObjects` (the
`jemalloc_cxl_slow_tier`-gated `DramMultiArenaObjects` arm removed), and `jemalloc_cxl_slow_tier`
was dropped from the `#![cfg_attr(...)]` nightly-feature gate and the `mod allocator` gate (which
still includes `eviction_stacks_pmem`, still needed for `Hybrid`'s `allocator_api2::alloc::Allocator`
impl). `Cargo.toml`: removed the `jemalloc_cxl` path dependency, removed the `jemalloc_cxl_slow_tier`
feature entirely, and changed `eviction_stacks_pmem = ["dep:jemalloc_cxl"]` to `eviction_stacks_pmem
= []`. Deleted the `jemalloc_cxl/` crate directory itself (confirmed fully committed, no
uncommitted changes, before deleting).

Verified: `eviction_stacks_pmem` alone and combined with each of the four hybrid-cache features
builds clean and passes its `--lib` suite (91/101/101/102/102 -- the +10 over each hybrid's
non-`eviction_stacks_pmem` baseline is `eviction_stacks_pmem`'s own additional PMEM-stack test
coverage, unaffected by the allocator swap); all four hybrid-cache real-PMEM integration suites
combined with `eviction_stacks_pmem` pass at their documented baselines unchanged (15/15+2 ignored
lru, 19/19 lfu, 18/18 two_q, 14/14 fifo) -- genuinely exercising `HybridObjects` under real PMEM
load via `PmemHashList`/`PmemVecList`, not just the bare-stack unit tests. `cargo build --features
jemalloc_cxl_slow_tier` now correctly reports "the package 'paper-cache' does not contain this
feature." `all_dram`/`key_value_pmem`/`global_hashtable_pmem`/`hashbrown_dram`/`tiering`/
`multitiering` regression builds all still build clean. `Cargo.lock` no longer references
`jemalloc_cxl`.

## Feature: `lru_sized_hybrid_cache` (implemented — mirrors `lru_hybrid_cache`, with a size-split fast AND slow tier)

### Source (design brief from the user)

Requested as "very similar in structure to `lru_hybrid_cache`... should use the same logic for
promotion, demotion, and admission" but with the fast tier "separate[d]... into two different
configurable fast tier sizes" routed by "a configurable sizing threshold." Confirmed through a
plan-mode back-and-forth (recapped here since the reasoning shaped several non-obvious design
choices):

1. **Still exactly one physical DRAM tier and one physical PMEM tier** — the size split is purely a
   bookkeeping concern, not a new allocator/arena. Confirmed explicitly: "There is still 1 dram
   tier and 1 pmem tier... however each object size class has their own data movement."
2. **The slow (PMEM) tier also splits by size, bookkeeping-only** — the user's own follow-up
   ("i think the pmem arena would also be local to the size here") was clarified down to: two
   independent slow *recency lists*, still sharing the one physical `Hybrid`/`HybridObjects` PMEM
   pool — explicitly **not** two separate physical PMEM arenas, given this project's own
   extensively-documented history (see the `jemalloc_cxl`/UMF-jemalloc-pool sections above) of
   multi-arena allocator experiments proving costly or unstable.
3. **Neither slow list gets an independent capacity** — confirmed the split is purely for eviction
   fairness, matching the user's original scope ("only asked for... configurable fast tier sizes").
4. **Terminal eviction, three rungs**: prefer the slow list with more objects (recency proxy avoiding
   real cross-list timestamps); if both slow lists are non-empty, whichever has more objects wins; if
   *both* are empty, fall back to whichever fast segment is furthest over its own budget by ratio.
5. **The last-resort fast-tier-eviction fallback is deliberately preserved, not eliminated.** Before
   designing this feature, `lru_hybrid_cache`'s own `evict_one()` was checked directly and confirmed
   to already have this same fallback (its combined fast+slow list's absolute tail can be a
   fast-tagged key if nothing has ever been demoted, and `apply_evictions` still erases it) — reported
   to the user, who explicitly asked the new design to replicate it rather than "fix" it away via
   admission-time rejection.

### Design

**Four independent, homogeneous recency lists (`small_fast`/`large_fast`/`small_slow`/`large_slow`)
plus one combined `entries: HashMap<HashedKey, SizedEntry>`** — a deliberate departure from
`LruHybridStack`'s single-combined-list-plus-`fast_boundary`-cursor trick, which only works for
exactly one fast segment sharing one list with exactly one slow segment. With two independent fast
sources each feeding their own independent slow destination, four separate homogeneous lists turned
out *simpler* than any cursor-based scheme: each list's own tail is directly its own
demotion/eviction candidate, no cursor needed anywhere (see `worker/policy/policy_stack/
lru_sized_hybrid_stack.rs`'s module doc for the full derivation).

**Classification compares against `ObjectSize` (`base_size`), not raw `value.len()`** — a deliberate
deviation from the literal request, confirmed as the right tradeoff rather than silently decided:
`PolicyStack::insert`'s only size parameter is the same `base_size` every other stack already
budgets against; threading a second raw-length parameter through would mean changing that trait
method's signature for all nine other stacks for a benefit that's only ever a small, near-constant
offset near the threshold boundary.

**Admission, promotion, and reclassification are all one `touch_fast` code path**, mirroring
`LruHybridStack::touch_fast_key`'s existing "any touch always promotes to fast, whichever tier it
was in" rule — this design just adds "which of the two fast segments" on top of that unchanged rule.
A fast↔fast reclassification (an overwrite whose new size crosses the threshold) moves between the
two fast lists directly and **emits no `(key, Tier)` migration** — both segments are physically
`TieredBuffer::Fast`, so `PolicyWorker::apply_tier_migrations`'s existing binary
demotion/promotion-partition pipeline needed **zero changes** to support this feature; only the
stack's own internal bookkeeping needed the four-way split, entirely invisible to `Tier`/
`TieredBuffer`/the `migrate` closure.

**Shared DRAM-reservation overhead is split proportionally between the two fast segments'
capacities** (`LruSizedHybridStack::reserved_shares`, using a `u128` intermediate to avoid overflow),
not charged in full against each independently, which would double-count the same physical metadata
cost. The two slow lists have no capacity to reserve against.

**API surface**: `PaperCache::<K, TieredBuffer>::new(max_size, small_fast_tier_size,
large_fast_tier_size, size_threshold)`. The shared generic block's existing
`set_fast_tier_size`/`fast_tier_size` are reused to mean the SMALL segment specifically (documented
clearly on this feature's own impl block, since every other hybrid design means "the whole fast
tier" by those same method names) — new bespoke `set_large_fast_tier_size`/`large_fast_tier_size`
and `set_size_threshold`/`size_threshold` cover the rest. Doesn't reuse the shared `new_hybrid`
helper (needs three sizing scalars routed to three different places — two `AtomicStatus` fields plus
three `WorkerEvent` broadcasts — rather than `new_hybrid`'s one `CacheTierSize`/one broadcast);
duplicates that ~65-line setup in a bespoke `new_sized_hybrid` instead, judged less invasive than
widening `new_hybrid`'s signature for every other hybrid design's benefit. `new_hybrid` itself picked
up a narrower inner `#[cfg]` (the original four features only) so an `lru_sized_hybrid_cache`-only
build doesn't compile an unused method.

**`PolicyStack` trait gained 10 new default (no-op) methods**, purely additive:
`resize_large_fast_tier`, `resize_size_threshold`, and eight granular
`small_fast_bytes_used`/`large_fast_bytes_used`/`small_fast_object_count`/`large_fast_object_count`/
`small_slow_bytes_used`/`large_slow_bytes_used`/`small_slow_object_count`/`large_slow_object_count`
accessors — every other stack keeps compiling unaffected via the defaults.

**Stats (`LruSizedHybridStats`, 15 fields)** keep the existing 7-field shape (`promotions`/
`demotions`/`evictions`/combined `fast_bytes_used`/`slow_bytes_used`/`fast_objects`/`slow_objects`,
for drop-in consistency with the other four hybrids) plus 8 new per-segment granular fields, since
the slow tier genuinely has two independently-tracked lists now too — symmetry judged more useful
than parsimony here.

### A real bug this caught while writing the integration test: hand-derived overhead math was wrong twice

The `terminal_eviction_falls_back_to_the_more_over_budget_fast_segment_when_slow_is_empty`
integration test needed a scenario where overall `max_size` is exceeded while both slow lists stay
genuinely empty — only possible if a fast segment's own effective budget (capacity minus its
proportional share of the shared-metadata DRAM reservation) never gets exceeded by real usage, while
the overhead-inclusive `used_size()` does exceed `max_size`. Two attempts at hand-deriving the
needed numbers from the overhead constants in `object/overhead.rs` (75 bytes/object reservation, 85
bytes/object policy overhead, an independently-measured 49-byte `base_size` for a representative
test object) were both measurably wrong: the first (`small_capacity == large_capacity == max_size ==
500`) under-shot the reservation's real bite and 7 of the 10 test objects demoted for real, not the
intended 0; the second guess for `max_size` (1200) overshot the real measured `used_size()` (1180)
by just 20. Root cause of the discrepancy: the 49-byte probe measurement used a throwaway
`Object<u32, Box<[u8]>>` rather than the real `Object<u32, TieredBuffer>` the feature actually
stores, and the two buffer types' overhead isn't quite identical. Fixed by measuring the real
`status().used_size()` directly (via a temporary `eprintln!` diagnostic, removed once the true
number was known) rather than continuing to guess from constants — a lesson worth generalizing:
prefer a direct runtime measurement over hand-deriving from multiple independent overhead constants
when precision actually matters for a test's correctness, not just its intent.

### Implementation status: complete

`policy.rs`, `status.rs`, `object/overhead.rs`, `worker/mod.rs`, `worker/manager.rs`,
`worker/policy/mod.rs`, `worker/policy/policy_stack/mod.rs`,
`worker/policy/policy_stack/lru_sized_hybrid_stack.rs` (new), `src/lru_sized_hybrid_cache/` (new),
`lib.rs`, and `Cargo.toml` are done. 115 unit/inline tests pass under `--features
lru_sized_hybrid_cache` (17 in the new stack file, 6 in a `worker::policy::lru_sized_hybrid_tests`
module mirroring `lru_hybrid_tests`'s synthetic-buffer wiring pattern, 4 in a
`test_lru_sized_hybrid_cache` lib.rs module mirroring the other hybrids' fast-tier-only happy-path
suite, plus the existing suite's other tests unaffected). `tests/lru_sized_hybrid_cache_integration.rs`
(20 tests, real PMEM/UMF allocator, modeled on `tests/lru_hybrid_cache_integration.rs`) passes twice
in a row (not flaky): `cargo +nightly test --test lru_sized_hybrid_cache_integration --features
lru_sized_hybrid_cache`.

### Remaining work

- No dedicated stress/scale test yet (the `#[ignore]`d `repro_real_dram_usage_at_scale`/
  `concurrent_set_from_multiple_threads_still_demotes` tests `lru_hybrid_cache_integration.rs` has
  for its own DRAM-usage/concurrency investigations weren't ported — this feature's functional test
  suite is complete, but no large-N/concurrent-access real-DRAM measurement has been done for the
  four-list design specifically).

## Restored the jemalloc_cxl multi-arena extent-hooks allocator, on request — available again, not rewired

Per explicit request ("bring back the multi-arena jemalloc extent hooks that recently got
removed... it should be doable to use this as allocator, tho just return it and don't worry about
that for now"), restored the `jemalloc_cxl/` crate and everything in `src/allocator.rs` that the
earlier `original`-feature-removal session (see "Removed the deprecated `original` feature; migrate
`eviction_stacks_pmem` off jemalloc_cxl onto Hybrid/HybridObjects" above) deleted:
`numa_arena_pool`, `EvictionStackAllocator`, `SlowTierJemallocAllocator`, and
`DramMultiArenaObjects`, plus `jemalloc_cxl_slow_tier`'s `Cargo.toml` feature, `TieredBuffer`'s
jemalloc_cxl-backed `Slow` variant (`src/tiered_buffer.rs`), and `lib.rs`'s
`#[global_allocator]`/`cfg_attr`/`mod allocator` wiring for it. Restored via `git checkout
<pre-removal-commit> -- jemalloc_cxl/ src/allocator.rs src/tiered_buffer.rs` (both files were
untouched by any work done since that removal, so a clean full-file checkout was safe) plus manual
re-application of the `Cargo.toml`/`lib.rs` hunks that would otherwise have collided with this
session's unrelated `lru_sized_hybrid_cache` additions to those same two files.

**One deliberate change from the original shape, not a full byte-for-byte revert**:
`EvictionStackAllocator`'s gate moved from `#[cfg(feature = "eviction_stacks_pmem")]` to
`#[cfg(feature = "jemalloc_cxl_slow_tier")]` (both the module and its `pub use`), and
`numa_arena_pool`'s gate dropped `eviction_stacks_pmem` from its `any(...)` down to
`jemalloc_cxl_slow_tier` alone. Explicit tradeoff, not an oversight: `eviction_stacks_pmem` was
migrated to `crate::Hybrid`/`HybridObjects` earlier this session specifically so it would stop
depending on `jemalloc_cxl` — restoring `EvictionStackAllocator` under its *original* gate would
have silently reintroduced that dependency (`use jemalloc_cxl::...` inside a module compiled
whenever `eviction_stacks_pmem` is enabled, forcing `eviction_stacks_pmem = ["dep:jemalloc_cxl"]`
again) for code nothing actually calls anymore — undoing a change already made, verified, and
documented in this same file. Re-gating under `jemalloc_cxl_slow_tier` instead keeps
`eviction_stacks_pmem` exactly as it is today (confirmed via `cargo tree --features
eviction_stacks_pmem`: still no `jemalloc_cxl` in the dependency graph) while still making
`EvictionStackAllocator`'s code fully available again, compiling and ready to be wired back into
its six original call sites (`worker/policy/policy_stack/{lru,lfu,two_q,fifo}_hybrid_stack.rs`,
`lfu_stack.rs`, `pmem_collections.rs`) later if ever wanted — which is the "return it, don't worry
about wiring it in for now" scope this was explicitly asked for. None of those six files were
touched; they still import `crate::Hybrid`.

Verified: `cargo +nightly build --features jemalloc_cxl_slow_tier` builds clean; `cargo +nightly
build --features eviction_stacks_pmem` still builds clean with `jemalloc_cxl` absent from `cargo
tree`'s output; the full unit-test regression sweep (`lru_sized_hybrid_cache`,
`lru_sized_hybrid_cache,eviction_stacks_pmem`, all four existing hybrid-cache features,
`lru_hybrid_cache,jemalloc_cxl_slow_tier`) passes at the exact same counts as before this
restoration (115/125/112/112/113/113/114) — this was purely additive, nothing it touched changed
behavior for any build that doesn't explicitly opt into `jemalloc_cxl_slow_tier`.

**Bottom line, unchanged from before removal**: `jemalloc_cxl_slow_tier` is available again but
still not the default, and still not proven safe under real concurrent load — three separate
retests earlier in this project's history all failed (see the "Reverted, twice" and "Third retest"
`umfJemallocPoolOps`/`jemalloc_cxl_slow_tier` sections above). Passing this crate's own low-
concurrency unit/integration suites was never in question; don't treat that as evidence of fitness
for real concurrent load, and don't re-run `paper-benchmark-cxl` against it without being prepared
for it to fail the same way again.

## Performance: the background event pipeline on the GET path

Per explicit request ("see what efficiencies can be made to the background eviction queue manager
to get the best performance out of get requests"). Unlike most of the performance work above,
which chased where *memory* goes, this one is about what a single `get()` costs the rest of the
process — the background machinery every read feeds, not the read's own object-map lookup and byte
copy.

Traced the full per-read path first rather than guessing at hot spots. A cache read used to cost:
(1) a `try_send` from the API thread into `WorkerManager`'s channel; (2) `WorkerManager` — a
dedicated thread **spinning** on `try_recv` — popping it, cloning it, and pushing a copy into
*every* sub-worker's channel including `TtlWorker`, whose `run` loop has no `Get` arm at all and
discards it; (3) `PolicyWorker` sending a *second* channel message, per read, to `TraceWorker`;
(4) `TraceWorker` copying a 13-byte record per hit into an on-disk temp file; (5)
`apply_tier_migrations` republishing four gauges into `AtomicStatus`, per event, via four virtual
calls and four atomic stores. Steps 3–4 exist for exactly one consumer,
`reconstruct_policy_stack`, which replays the trace to rebuild a *different* policy's stack after
a live policy switch — and every hybrid cache is constructed with a single fixed policy and
exposes no way to switch, so that trace was being written and never read.

### Methodology: measure the ceiling before optimizing toward it

The first measurement attempt was nearly useless and worth recording as a trap. A throughput
harness (temporary `tests/zz_get_bench.rs`, deleted after use: 20,000 x 4 KiB objects, 512 MB
cache, 64 MB fast tier so demotion is real, 2M reads over a deterministic per-thread xorshift key
stream, `BENCH_THREADS` selecting the client count) showed the changes landing at *zero* measurable
difference. Correct, and uninformative: GET throughput in that harness is bound by the API thread's
own hash + `DashMap` read + 4 KiB copy, so shaving worker-thread work is invisible. Two things
made it informative:

1. **A ceiling probe.** Temporarily gating the `WorkerEvent::Get` broadcast out of `get()` entirely
   (behind a `OnceLock`-cached env check — note `std::env::var_os` per call would itself have
   poisoned the measurement) answers "what does the background pipeline cost a read *at all*",
   which bounds every possible optimization. Answer: **677K -> 834K gets/sec single-threaded
   (+23%)**, 3.25M -> 3.63M at 8 threads. Large enough to be worth real work — and, crucially,
   large enough that a change measuring "no difference" is a change that missed.
2. **Sweeping the client-thread count** (1 / 4 / 8 on this 8-core box). Every conclusion below
   flips sign somewhere in that sweep. A single-thread-only or 8-thread-only measurement would
   have produced a confidently wrong answer either way.

A second probe isolated *why* the pipeline was expensive: replacing `WorkerManager`'s
`std::hint::spin_loop()` with a short sleep recovered ~4.5% on its own. The spinning thread was
hammering the head/tail cache lines of the very channel the API threads were producing into.

### Landed

1. **`WorkerManager` (a thread) became `WorkerFanout` (an inline call).** The whole type existed to
   pop one event and push copies of it to N sub-workers. Doing that on the calling thread removes
   one channel round trip per operation, both horns of the receive-side dilemma (blocking `recv()`
   parked on a futex — the ~30%-of-cycles problem that motivated the spin in the first place; the
   spin traded that for producer-side cache-line contention), and an entire permanently-occupied
   core. `PaperCache::worker_manager: Arc<WorkerSender>` became `workers: Arc<WorkerFanout>`; the
   five construction sites lost their `unbounded()` + `thread::spawn(move || worker_manager.run())`
   pair; `broadcast` became `self.workers.send(event)`. `WorkerFanout::send` hands the event itself
   to the *last* subscriber and clones only for the ones before it, so the single-subscriber case
   (every `Get`, after change 2) is a plain move — exactly the cost the old single unconditional
   send into the manager's channel had.
2. **Per-worker event subscription masks** (`EventMask` / `Events` in `worker/mod.rs`, one bit per
   `WorkerEvent` variant plus one subscription constant per sub-worker; `WorkerEvent::mask_bit()`
   maps a variant to its bit). `TtlWorker` no longer receives a copy of every read. `PolicyWorker`
   is also deliberately *not* subscribed to `Promote` (delivered to the tiering worker directly via
   its own `promotion_tx`, never routed back through the fan-out) or `Ttl` (no arm; an expiry
   change doesn't reorder or resize anything a policy stack tracks). **Maintenance hazard worth
   knowing**: a worker's mask and its `run` match arms are now a single edit — adding an arm
   without adding the bit silently drops that event.
3. **The trace subsystem is off unless a policy switch is actually reachable** (`trace_is_useful`:
   `status.policies().len() > 1`). `PolicyWorker::trace_worker` became `Option<Sender<StackEvent>>`;
   when `None`, no `TraceWorker` thread is spawned, the per-event `StackEvent` isn't even derived
   (not just not sent), `apply_evictions` skips pushing eviction records into `buffered_events`, and
   `flush_buffered_events` returns immediately. `PaperPolicy::Auto` doesn't change the condition:
   auto-switching picks from the same `policies` list, so a one-entry list can only ever
   "switch" to the policy already running. `handle_policy` grew a defensive early return for the
   trace-disabled case — bailing *before* it clears `policy_stack`, since a reconstruction
   that can never deliver would otherwise leave the worker with no stack, permanently.
4. **`refresh_tier_gauges` split out of all 14 `apply_tier_migrations` siblings**, called once per
   event-loop pass instead of once per event. These are pure gauges — a snapshot of state the
   stack already owns — so republishing after each batch reports the same values as republishing
   after each event; only the write frequency changes, and that frequency was putting four
   virtual calls and four atomic stores into `AtomicStatus` on the path of every read. Still
   unconditional (not gated on a migration having happened) — that gate is what let them go stale
   indefinitely, see "Bug 2" above; this moves *when* the refresh runs, never *whether*.
   `LfuHybridStack`'s `admission_latched` mirror moved with the gauges: the latch is one-way, so a
   mirror lagging by one pass only means a handful of sets around the transition build `Fast` and
   get corrected — the pre-latch behaviour, briefly, not an incorrect placement.
5. **`PolicyWorker::run` reuses its event buffer** instead of `try_iter().collect::<Vec<_>>()` per
   pass, so steady-state polling allocates nothing.

### Measured (8-core box, 20k x 4 KiB working set, 2M reads, median of 5 runs)

| client threads | before | after | delta |
|---|---|---|---|
| 1 | 684,195 | 685,608 | +0.2% |
| 4 | 2,482,037 | 2,456,834 | -1.0% |
| 8 | 3,184,438 | **4,223,833** | **+32.6%** |

The shape is the point, and it's the reason the thread-count sweep mattered. The win arrives
exactly at saturation, because that is when a dedicated fan-out thread stops being free work on a
spare core and becomes a core taken away from the request path. Below saturation the spare core
absorbed the fan-out, so doing it inline is a wash to marginally negative — the 4-thread column is
within this harness's run-to-run spread. Note the 8-thread result (4.22M) also clears the earlier
ceiling probe's 3.63M: that probe measured the pipeline's cost *with the manager thread still
running*, so freeing its core is worth more than the entire per-read pipeline it was serving.

### Tried, measured, reverted: draining the backlog instead of sleeping

`delay_event_loop` sleeps `SHORT_POLLING_DURATION` (1 ms) after every pass, including when the pass
just processed a full batch and more work is already queued. That looks obviously wrong — the
backlog visibly grows, and every decision the loop makes (demotions, evictions, the ordering a
subsequent GET is served against) runs that much further behind reality — so it was changed to
return immediately whenever the pass had events.

Measured: **-24% at 8 threads** (4.20M -> 3.17M gets/sec). `thread::yield_now()` in place of the
sleep measured the same as the spin (3.15M), because with every client thread runnable a yield
returns almost immediately. Reverted; the sleep is now documented as load-bearing with those
numbers attached, so it doesn't get "fixed" again. The reasoning it was reverted on: this thread
shares cores with the request path, and `try_iter()` drains everything accumulated during the
sleep, so sleeping costs staleness bounded by the polling interval but *not* throughput — the
queue does not grow without bound, it stabilizes at whatever arrived during one poll interval.
Skipping the sleep converts the loop into a spin competing with the clients it exists to serve.

This is the same lesson as the migration-parallelization sections above, arrived at independently:
on a core-constrained box, making a background worker more eager is a transfer from the request
path, not a free improvement.

### Verified

All 23 feature-flag build combos compile (all 14 hybrid variants, `all_dram`, `key_value_pmem`,
`global_hashtable_pmem`, `hashbrown_dram`, `eviction_stacks_pmem`, `tiering`, `multitiering`,
`key_value_pmem,enable_tiering_manager` — the 3-sub-worker fan-out path — and
`lru_hybrid_cache,eviction_stacks_pmem`). Unit suites pass unchanged (240 lru, 240 lfu, 241 two_q,
241 fifo, 243 lru_sized, 232 all_dram, 230 key_value_pmem, 233 global_hashtable_pmem, 230
hashbrown_dram). All five real-PMEM hybrid integration suites pass twice consecutively at their
documented baselines (15/15 +2 ignored lru, 19/19 lfu, 18/18 two_q, 14/14 fifo, 20/20 lru_sized),
as does `tiering_integration` (3/3).

The 33 worker unit tests that drive `apply_tier_migrations()` directly and then assert on gauges
now also call `refresh_tier_gauges()`, mirroring what `run()` does — the split is internal, so the
tests had to follow it rather than the behaviour having changed.

One honest caveat: a single `two_q_hybrid_cache_integration` failure appeared once mid-session and
did not reproduce in 8 consecutive re-runs afterward; the test name wasn't captured before it was
lost. Consistent with that suite's already-documented timing sensitivity, but unproven.

### Noticed, deliberately not done (outside the worker pipeline)

- **`AtomicStatus` has no cache-line separation between hot-read and hot-write groups.** The
  per-read `total_hits`/`total_gets` counters live in the same struct as `base_used_size`/
  `num_objects` and every hybrid's worker-written gauges, with `repr(Rust)` layout, so nothing
  prevents an API-thread counter from sharing a 64-byte line with a worker-written gauge. Change 4
  removes most of the *frequency* of those worker writes without addressing the layout. Fixing it
  properly needs `#[repr(C)]` plus explicit padding (no `crossbeam-utils` dependency today, so no
  `CachePadded`), which is a `status.rs` change, not a worker one.
- **`incr_hits` does two contended read-modify-writes per read** (`total_gets` and `total_hits`) on
  lines shared by every client thread. Sharded or per-thread counters would fix it, but that
  changes `Status`'s semantics and is well outside "the background eviction queue manager".

## Reporting: hybrid tier stats in the benchmark, and a summary CSV per run

The benchmark recorded **none** of the hybrid designs' tier statistics. Nothing called any
`*_hybrid_stats()` accessor; the only trace of intent was a commented-out
`//CacheBackend::report_stats_lru(backend)` at the end of `main.rs`. `--output-csv` writes only a
100-row latency *distribution* (`Stats::save_latency_percentiles`), truncating on open, with
nowhere sensible to put a scalar like "total demotions" and one file per run to stitch together
afterwards.

### The obstacle, and where the `#[cfg]` cascade belongs

Each of the 15 hybrid designs exposes a differently-*named* accessor returning a differently-named
type (`lru_hybrid_stats() -> LruHybridStats`, `s3_fifo_ghost_hybrid_stats() ->
S3FifoGhostHybridStats`, ...). Any consumer generic over "whichever design this binary was built
against" therefore needed its own 15-arm cascade naming all 15 methods *and* all 15 types,
duplicated per call site, needing a new arm every time a design is added here.

Checked all 15 `*_hybrid_cache/stats.rs` structs directly: they share the same seven fields
(`promotions`/`demotions`/`evictions`/`fast_bytes_used`/`slow_bytes_used`/`fast_objects`/
`slow_objects`) *identically*; only `LruSizedHybridStats` carries extras (8 per-size-segment
gauges). So the cascade was collapsed to one place inside this crate:

- **`src/hybrid_stats.rs`** — `HybridStats`, the neutral seven-field snapshot, plus
  `total_objects()`/`total_bytes_used()`.
- **`AtomicStatus::hybrid_stats()`** (`status.rs`) — 15 one-line `#[cfg]`'d bodies, each delegating
  to that design's *existing* named accessor through a `common_hybrid_stats!` macro. Reading
  *through* the named accessor (rather than the underlying atomics) is deliberate: the two can
  never disagree about a counter because there is only one place either loads from. A design added
  without an arm here is a **compile error**, which is the intended failure mode — never a
  silently-zeroed column in someone's results CSV.
- **`PaperCache::hybrid_stats()`** (`lib.rs`) — one method on the shared generic hybrid impl block.

The per-design named accessors are untouched: `paper-server` and existing callers keep the names
they use, and designs with extra fields keep them.

### Benchmark side

- **`CacheBackend::cache_report()`** (`cache_backend.rs`) — one trait method, one
  `#[cfg(any(..))]`, no per-design arms. Returns policy, max/used size, object count, RSS/HWM, miss
  ratio, fast-tier budget, and the seven tier stats. Returns benchmark-local structs (`CacheReport`/
  `HybridStatsSnapshot`) rather than `paper_cache::HybridStats`, which only exists under a hybrid
  feature — naming it in the trait would push the same cascade back out into `main.rs`/`stats.rs`.
  **The policy string comes from the cache itself** (`Status::policy()`), so each row labels its own
  design instead of a hardcoded 15-name table in the harness.
- **`src/summary.rs`** — a `*** CACHE stats ***` / `*** TIER stats ***` stdout block (tier block
  skipped entirely on non-hybrid builds, where seven zeros would read as "nothing moved" rather
  than "not applicable"), plus `--summary-csv`: **one appended row per run**, 27 columns, header
  written only when the file is new or empty. Point every run of a sweep at the same file and it
  accumulates directly comparable rows — the shape the aggregate counts are actually wanted in.
  Hand-rolled rather than via `kwik`'s `CsvWriter`, which opens with `File::create` (truncating,
  the opposite of an accumulating sweep file) and whose `WriteRow` signature differs between the
  benchmark's `hot_fix` feature states.
- `Stats::get_summary()`/`set_summary()` (`stats.rs`) — the scalar count/mean/p99/bytes columns,
  the same numbers `print_get_stats`/`print_set_stats` render as distributions.

### Verified against real traces, all 15 designs — including a falsification test

800K accesses of `standard_web.bin`, 2 GB cache, `-c 1`, one run per design (rebuilt against each
paper-cache hybrid feature in turn). Three independent cross-checks, all exact:

1. **`fast_objects + slow_objects == num_objects`, exactly, for all 15.** These come from
   completely separate paths — `num_objects` is incremented on the API thread in `set()`, the tier
   gauges from the policy stack's own bookkeeping in `PolicyWorker`. Exact agreement at ~120K
   objects across 15 different stack implementations means neither side drifts.
2. **`used_size − (fast_bytes_used + slow_bytes_used)` is an exact integer multiple of
   `num_objects`**, and the per-object value matches that design's `get_policy_overhead` constant
   to the byte: 85 (LruHybrid/FifoHybrid/LruSized), 86 (TwoQHybrid/2Q-ghost), 87 (s3-fifo family),
   113 (LfuHybrid). Zero remainder — a third accounting path (`OverheadManager::base_size`)
   agreeing exactly.
3. **`fast_bytes_used <= fast_tier_size`** for every design.

Several designs legitimately report a **0** in one counter, and the shapes are all explained by
their own documented semantics rather than a reporting gap: `fifo-hybrid` has 0 promotions (FIFO
does not reorder on access — no promotion mechanism exists); the two non-reprieve `fast_admission`
variants have 0 promotions because their one-access queue is already Fast, so the
one-access→main promotion is Fast→Fast and emits no migration (exactly the optimization documented
in `s3_fifo_ghost_lazy_demotion_fast_admission_hybrid_stack.rs`'s module doc); and six designs
reported 0 demotions with the fast tier sitting well under budget.

That last group was the one that could plausibly have been a stuck counter, so it was tested
directly rather than argued: **the same sweep was re-run at a 750 MB fast tier instead of 1 GB.**
The three ghost designs whose fast tier went from 91% to 100% of budget flipped from 0 demotions to
**10,931 / 9,592 / 9,736**; the designs still reporting 0 (`2q-hybrid` at 49.7%, `s3-fifo-hybrid` at
49.9%, `s3-fifo-lazy-demotion-reprieve` at 93.4%) are exactly the ones still under budget. Every
design's demotion count moved monotonically with fast-tier pressure. The zeros were genuine
headroom, not a broken gauge.

Also verified: a non-hybrid (`all_dram`) build still writes a valid row (`policy=lru`, empty
`fast_tier_size`, zeroed tier columns) and skips the tier block; every row in both CSVs has exactly
27 fields matching the 27 headers; unit suites pass unchanged (253 lru, 253 lfu, 254 two_q, 254
fifo, 256 lru_sized, 246 s3_fifo, 246 s3_fifo_lazy_demotion_reprieve); six real-PMEM integration
suites pass at their documented baselines (15/15 +2 ignored lru, 19/19 lfu, 18/18 two_q, 14/14
fifo, 20/20 lru_sized, 20/20 s3_fifo); and `all_dram`/`key_value_pmem`/`global_hashtable_pmem`/
`hashbrown_dram`/`eviction_stacks_pmem`/`tiering` still build clean (the crate-side change is purely
additive — a new module and one new method).

### A real pre-existing bug this surfaced: `FAST_TIER_GB` was integer-only in 13 of 15 backends

Found while setting up the verification sweep, not by code review: `--features hybrid_2q` died with
`InvalidFastTierSize` at `FAST_TIER_GB=0.25`. Root cause in `cache_backend.rs`: only `hybrid` and
`hybrid_lfu` parsed the env var as `f64`; the other 13 parsed it as **`u64`**, so
`"0.25".parse::<u64>()` failed, `.ok()` swallowed the error, and `.unwrap_or(4)` silently
substituted the 4 GB default. **Any fast-tier sweep that passed a fractional GB value was measuring
4 GB for 13 of the 15 designs** — silently, with no error, producing plausible-looking results at
the wrong configuration. (It only surfaced as a hard failure here because 4 GB exceeded the 2 GB
`--cache-max-size` the verification used; at production cache sizes it would just quietly
mismeasure.)

Fixed at all 13 sites, mirroring the two already-correct ones: parse `f64`, convert via
`CacheTierSize::Mb((gb * 1000.0).round() as u64)` instead of `CacheTierSize::Gb(gb)`.
`hybrid_lru_sized`'s even split needed its own variant (`(gb * 1000.0 / 2.0).round() as u64`).
**Behavior-preserving for whole-GB values**: `CacheTierSize` is decimal in both units (`Mb` = 10^6,
`Gb` = 10^9), so `Mb(gb * 1000)` is byte-identical to the old `Gb(gb)` — existing whole-GB sweep
results stay directly comparable. Verified by re-running all 15 designs at `FAST_TIER_GB=0.75`
against a 2 GB cache (a value unparseable as `u64`, with the old 4 GB fallback exceeding
`max_size`, so any unconverted design would fail outright rather than mismeasure): all 15 built,
ran, and reported exactly `fast_tier_size=750000000`, with all three invariants above still exact.

### One API asymmetry worth knowing: `fast_tier_size()` means something different for `lru_sized`

The first sweep showed `lru-sized-hybrid` at **194.7%** of its fast-tier budget — not a leak, and
not an accounting bug. `lru_sized_hybrid_cache` is the one design where `fast_tier_size()` means the
**SMALL segment only**, with the LARGE segment carried separately as `large_fast_tier_size()` (see
that feature's section above); the benchmark's constructor splits `FAST_TIER_GB` evenly between
them. So `fast_bytes_used` (both segments combined) was being divided by half the real budget.
Fixed in `cache_report()` by summing the two segments for that design specifically — re-measured at
98.0%, in line with every other design. Worth remembering for any other consumer of
`fast_tier_size()`: it is the whole DRAM budget for 14 of the 15 designs and half of it for the
fifteenth.

## Feature: `two_q_fast_admission_hybrid_cache` (implemented)

`two_q_hybrid_cache` with the one-access FIFO queue in the **fast** tier instead of the slow tier, so
`set()` is a plain DRAM write rather than a synchronous PMEM/UMF allocation on the calling thread.
Requested directly ("using 2q... but having the single access queue in the fast tier so set
performance is good"). This is the same delta
`s3_fifo_ghost_lazy_demotion_fast_admission_hybrid_cache` already applies to the s3-fifo family, so
most of the design risk was already retired — that stack's module doc was the working template.

### What actually differs from `TwoQHybridStack`

The logical 2Q structure is untouched (two live queues, no ghost queue, main-queue LRU segmentation,
FIFO-tail-first eviction, one combined per-key `entries` map). Only physical placement changes:

1. **`admission_tier` is unconditionally `Tier::Fast`**, for brand-new and existing keys alike. This
   removes the `DashMap` probe `TwoQHybridPolicy::admission_tier` needs to distinguish the two cases
   — one fewer lookup per `set()`, on top of the avoided PMEM allocation.
2. **`effective_main_fast_capacity() = fast_capacity.saturating_sub(fifo_capacity)`**, checked by
   `settle_fast_tier` in place of raw `fast_capacity`. Both structures are DRAM now, so leaving the
   budgets independent would let real DRAM reach `fast_capacity + fifo_capacity`.
3. **Gauges flip**: `fast_bytes_used = fifo_used + fast_used`, `slow_bytes_used = slow_used`,
   `fast_object_count = fifo_queue.len() + fast_count`, `slow_object_count = main_count - fast_count`.
4. **`promote_from_fifo` emits no migration.** In `TwoQHybridStack` that `(key, Tier::Fast)` push was
   load-bearing — the API layer had built the bytes Slow, so the migration was the only thing that
   ever moved them to DRAM. Here it would copy a value onto itself. A FIFO promotion can still
   *cause* migrations (the bytes move from the FIFO reservation into the main-fast budget, which can
   demote someone else), producing a batch with `Tier::Slow` entries and no matching `Tier::Fast` —
   a shape `apply_tier_migrations` already handles, so no worker-side change was needed.
5. **`resize()` re-settles**, which `TwoQHybridStack::resize` need not: `fifo_capacity` derives from
   `max_size`, so growing the cache grows the reservation and shrinks the main queue's effective
   budget.

### A design decision a failing test forced me to get right

The first implementation also called `settle_fast_tier()` from `insert`, on the reasoning that
"admission now consumes DRAM, so it must be able to demote." A unit test written from that same
reasoning (`a_brand_new_admission_alone_can_force_a_demotion`) failed — correctly. The reservation is
the **fixed `fifo_capacity`, not live `fifo_used`**, so admission moves no capacity and that call was
dead code.

The alternative (charging live `fifo_used`) would bound total DRAM more tightly, but would make the
main queue's budget breathe with FIFO occupancy and churn migrations as the queue fills and drains.
Kept the fixed reservation — matching the s3-fifo precedent — removed the dead call, and replaced the
test with two that pin the decision down in both directions
(`admission_does_not_move_the_main_queues_budget`,
`the_fifo_reservation_is_held_even_while_the_fifo_queue_is_empty`), documenting that they are the
tests that should fail if this is ever revisited. Two doc comments written from the same wrong
premise were corrected at the same time.

### The sizing trap this design introduces, and how the tests hit it

`k_in` is denominated in `max_size`, but the budget it consumes is `fast_tier_size`. Four of the
first seventeen integration tests failed on this: at `max_size = 1_000_000`, `k_in = 0.05` reserves
50,000 bytes against a 600-byte test fast tier, saturating the main queue's capacity to **zero** — so
every promotion self-demoted instantly and no key was ever observably fast in the main queue. Not an
implementation bug; the configurations were degenerate. Fixed with documented module-level constants
(`MAX_SIZE`/`FAST_TIER`/`K_IN`) and a `make_cache()` helper keeping the three quantities in a stated
relationship, rather than re-deriving the arithmetic at each call site. The one test that
deliberately keeps a degenerate-looking config (`many_admissions_all_land_fast_without_any_migration`,
which needs a FIFO queue roomy enough for 20 keys and never promotes) carries a comment saying so.

The same trap applies at real scale, which is why it is called out in `FEATURE_FLAGS.md` and the
benchmark backend: at a 24 GB cache with a 4 GB fast tier, `k_in = 0.1` reserves 2.4 GB — 60% of the
DRAM budget — for objects with no demonstrated reuse. Per explicit instruction, `k_in` stays
denominated in `max_size` (consistent with `two_q_hybrid_cache` and the s3-fifo family); tune the
value rather than the denominator. The benchmark backend reads `TWO_Q_K_IN` (default 0.1) so a k_in
sweep needs no rebuild.

Two other tests failed on **gauge-refresh cadence**, not correctness: a fixed 200 ms sleep observed
15 of 20 admissions, because `refresh_tier_gauges` runs once per worker event-loop pass. Switched to
`wait_until`, the convention the rest of this suite already uses.

### Files touched

`Cargo.toml`, `policy.rs` (variant + `Display`/`FromStr` + 3 tests, with the new
`"2q-fast-admission-hybrid-"` guard placed **before** `"2q-ghost-hybrid-"`/`"2q-hybrid-"`/`"2q-"` —
the same prefix-ordering trap `two_q_hybrid` hit, locked in by
`two_q_fast_admission_hybrid_does_not_collide_with_other_2q_forms`), `object/overhead.rs` (86
bytes/object, structurally identical to `TwoQHybrid`), `status.rs` (counters/gauges + an arm in
`hybrid_stats()`), `worker/mod.rs`, `worker/manager.rs`, `worker/policy/mod.rs` (16th
`apply_tier_migrations`/`refresh_tier_gauges` sibling, eviction counter, 6-test worker module),
`worker/policy/policy_stack/{mod.rs,two_q_fast_admission_hybrid_stack.rs}`,
`src/two_q_fast_admission_hybrid_cache/{mod.rs,stats.rs}`, `lib.rs` (module, `ActiveHybridPolicy`
alias, 15 pairwise `compile_error!` guards, impl block), and
`tests/two_q_fast_admission_hybrid_cache_integration.rs`. Benchmark side:
`hybrid_2q_fast_admission` feature + backend in `paper-benchmark-cxl`.

The 29 canonical `any(...)` hybrid cfg lists were updated by matching the exact
`feature = "two_q_hybrid_cache", feature = "fifo_hybrid_cache"` substring — verified beforehand to
occur exactly 29 times and never in a single-feature gate, so single-design `#[cfg]`s could not be
caught by accident.

### Verified

272/272 unit tests (including 17 new stack tests and 6 worker-level wiring tests);
`tests/two_q_fast_admission_hybrid_cache_integration.rs` 17/17, run three times consecutively (not
flaky); builds clean alone and with `eviction_stacks_pmem`; the mutual-exclusion `compile_error!`
fires as intended. No regressions: 12 other feature builds pass their full `--lib` suites (counts up
by exactly the 3 new `policy.rs` tests), and `two_q_hybrid_cache` (18/18) and `lru_hybrid_cache`
(15/15 +2 ignored) real-PMEM integration suites are unchanged.

**End-to-end against the real benchmark** (800K accesses of `standard_web.bin`, `-c 1`, 2 GB cache /
1 GB fast tier, `k_in` 0.1, same binary otherwise, both designs run back to back):

| | `2q-hybrid-0.1` | `2q-fast-admission-hybrid-0.1` |
|---|---|---|
| SET mean | 7.11 µs | **3.30 µs** (2.15x) |
| SET p99 | 25.48 µs | **9.36 µs** (2.72x) |
| GET mean | 4.09 µs | 3.34 µs |
| miss ratio | 0.3308 | 0.3211 |
| DRAM (fast_bytes_used) | 374 MB | 624 MB |
| PMEM (slow_bytes_used) | 200 MB | 0 MB |
| promotions / demotions | 22,565 / 0 | 0 / 0 |

The SET win is the designed effect and matches the ~2–3x the s3-fifo fast-admission variants
measured. The miss ratio being unchanged is the expected confirmation that only placement moved.
**The GET improvement should not be generalized**: at this scale the entire retained working set fit
inside the effective fast budget, so nothing was ever demoted (slow tier empty) while
`two_q_hybrid_cache` had 200 MB in PMEM by construction — that is a difference in tier placement, not
a like-for-like latency comparison, and the same rows show the cost (624 MB of DRAM versus 374 MB for
the identical workload). A configuration whose working set exceeds the effective fast budget would
exercise the demotion path and likely show GET neutral or slightly worse. Zero promotions is correct
by construction, not a missing counter: FIFO→main is Fast→Fast and emits no migration, and nothing
was demoted, so no PMEM→DRAM move ever occurred.

### Remaining work

- No large-scale/high-concurrency run yet (the comparison above is `-c 1` on an 800K-access slice).
  A configuration that genuinely exceeds the effective fast budget is the interesting one — it would
  exercise demotion and give a real GET comparison, which this run does not.
- `k_in` has not been swept. Given the reservation now comes out of DRAM, the value inherited from
  `two_q_hybrid_cache` (0.1) is unlikely to be the right one here; `TWO_Q_K_IN` exists to sweep it
  without a rebuild.

## Feature: `two_q_fast_admission_reprieve_hybrid_cache` (implemented)

`two_q_fast_admission_hybrid_cache` with one change, requested as a separate variant: a one-access
object that ages out of the FIFO queue without a second access is **reprieved into the slow tier**
rather than evicted. The placement was the user's own suggestion — *"adding an element to the bottom
of the lru queue if its not reaccessed again"* — and it turned out to be both the right ranking and
the cheap implementation; see below.

### The two traps the s3-fifo precedent had already documented

Reading `s3_fifo_lazy_demotion_fast_admission_reprieve_hybrid_stack.rs`'s module doc *before*
writing this saved repeating both of its bugs:

1. **A reprieve must not run through `evict_one()`.** `apply_evictions` unconditionally *erases*
   whatever key `evict_one()` returns, and answers a `None` by evicting a **random** object. So
   relief runs synchronously from `insert`/`resize` via `settle_fifo_queue()`, mirroring
   `settle_fast_tier`; `needs_capacity_eviction()` returns to the trait default `false`.
   This does **not** contradict `two_q_hybrid_cache`'s opposite lesson ("a `PolicyStack` cannot evict
   on its own", which desynced the stack from the object map) — the two rules are about different
   operations. *Eviction* must go through `evict_one`/`erase` because the stack cannot touch the
   object map; a *reprieve* removes nothing, so the key stays in `entries` and in the object map and
   only moves between two of the stack's own lists. Both rules are now stated together in the
   stack's module doc so the next person doesn't have to reconcile them from two places.
2. **Inserting at the fast/slow boundary is O(n).** `HashList`/`PmemHashList` expose only
   `push_front`/`push_back`/`move_front`/`move_back`, so the s3-fifo variant's first attempt walked
   every fast key per reprieve and burned ~18 minutes of worker CPU on a real trace without
   finishing a run — which is what forced its two-physical-lists restructure. **This variant sidesteps
   that entirely by landing at the back**, which is `push_back` — O(1) on the existing single
   `main_stack`, no restructure needed.

### Why the bottom is also the right answer, not just the cheap one

The main queue is LRU-ordered and its slow segment holds keys that were promoted at least once and
later demoted. A key aging out of the one-access queue has demonstrated *no* reuse, so ranking it
above proven-but-cold objects would invert the ordering the main queue exists to maintain. The
s3-fifo variant splices to the *front* of its slow segment (a full traversal before eviction); that
is defensible there because its main queue is FIFO-ish, and it is documented here as the natural
alternative if this placement proves too weak.

The boundary invariant survives a back-splice for free: `main_boundary` marks the LRU-most `Fast`
key and relies on fast keys forming a contiguous prefix. Appending a `Slow` key behind everything
preserves that and needs no boundary update — the second reason this placement is O(1). A dedicated
test (`a_reprieve_does_not_corrupt_the_fast_slow_boundary`) pins it down, including that a
subsequent fast-tier demotion still picks the right key.

### Measured: the hit-rate win is real, and its cause is not what it first looks like

800K accesses of `standard_web.bin`, `-c 1`, 2 GB cache / 1 GB fast tier, `k_in` 0.1, all three 2Q
designs run back to back with the same binary:

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

The 16.8% relative miss-ratio reduction is not mainly "second chances pay off". It is that **the
non-reprieve variant was leaving most of its configured cache unused**: it filled only 598 MB of
2 GB yet still evicted 219,232 objects, because with nothing ever demoted the only exit was
FIFO-capacity eviction, so the cache self-limited at roughly `fifo_capacity + main_fast`. Reprieving
routes those objects into the uncapped slow tier instead, so the cache actually uses `max_size` —
3.2x more objects retained. That is worth knowing because it means the comparison is partly between
two different *effective* cache sizes, and it also flags that the non-reprieve variant's `k_in` is
badly tuned at this configuration.

The costs are real and show in the same row. SET mean is 32% higher and SET p99 **2.5x** higher than
the non-reprieve variant *despite admission still being a pure DRAM write* — 190,066 reprieve copies
on the `PolicyWorker` thread contend with the API threads for the allocator, the same contention
documented throughout this file. GET is slower because 65,637 objects now live in PMEM instead of
none. So this variant trades tail SET latency and GET latency for hit rate.

### Two bugs of my own, both caught by the regression sweep

Worth recording because both came from scripted code duplication, not from design:

1. **A cloned `status.rs` region swept up neighbouring designs.** Duplicating the
   `two_q_fast_admission_*` recorder block by slicing from its first method to the `hybrid_stats`
   doc comment also copied every design defined in between (`fifo_hybrid_*`, `s3_fifo_hybrid_*`,
   `two_q_ghost_hybrid_*`, ...) verbatim — 63 methods — which only surfaced as `E0592 duplicate
   definitions` when building `fifo_hybrid_cache`, a feature the new variant does not touch. Fixed
   by deleting the duplicated tail, with a scripted assertion that every method being removed
   already existed earlier in the file so nothing unique could be lost.
2. **`PmemHashList` had no `push_back`.** The DRAM `HashList` this type shadows does, so the stack
   compiled fine by default and only failed under `eviction_stacks_pmem`. Added it, mirroring
   `push_front` exactly (same remove-if-present handling, same `ptr::write` reasoning for reused
   freed slots).

Both are arguments for running the *whole* feature-flag sweep rather than only the features a change
appears to touch.

### Verified

289/289 unit tests (14 new stack tests, 2 new worker-level tests), 299/299 with
`eviction_stacks_pmem`; `tests/two_q_fast_admission_reprieve_hybrid_cache_integration.rs` 19/19,
run twice; the mutual-exclusion `compile_error!` fires. No regressions across 13 other feature
suites, and `two_q_fast_admission_hybrid_cache` (17/17), `two_q_hybrid_cache` (18/18) integration
suites unchanged. `tiering`/`multitiering` `--lib` still fails on the **pre-existing**
`TieringManager::with_defaults()` inference error already documented above — confirmed identical on
the stashed pre-session tree (11 errors either way); both still *build* clean.

### Remaining work

- `k_in` unswept for both fast-admission variants. The measurement above strongly suggests 0.1 is
  wrong for the non-reprieve one (it caps the effective cache size); `TWO_Q_K_IN` sweeps it without
  a rebuild.
- No high-concurrency run. The SET p99 regression is allocator contention from the reprieve copies,
  so `-c 8` may look materially different from this `-c 1` measurement — in either direction.
- The front-of-slow-segment placement (what the s3-fifo variant does) is untested here, and would
  need `main_stack` split in two the way that stack's was.

## Bug fix: a TTL reap left the policy stack (and the hybrid tier gauges) stale

Found while checking whether TTLs are reaped actively, in the course of characterizing the
`cluster12`/`cluster31` Twitter traces (both carry a TTL on ~80–94% of records). They are reaped
actively — `TtlWorker` (`worker/ttl/mod.rs`) keeps a `BTreeMap<Instant, HashedKey>` (`Expiries`)
and each pass drains everything due via `pop_expired(now)` → `erase(...)`, sleeping 1 ms when the
nearest expiry is within 2 s and 1000 ms otherwise. Reads additionally check `Object::is_expired()`
on every path (`get`/`has`/`peek`/`ttl`), so an expired object is never *served* regardless.

**The bug: reaping told nobody.** `erase` only touches the object map and `AtomicStatus` —
`TtlWorker` had no sender at all, so it could not broadcast anything. The reaped key therefore
stayed in the policy stack's recency/frequency structures **and** kept counting its bytes toward
the hybrid stacks' `fast_used`/`slow_used`. `status.used_size()` stayed correct (erase decrements
it), so global eviction pressure was right, but `settle_fast_tier` compares `fast_used` against the
fast-tier budget — so it was demoting live objects to PMEM to make room for bytes that no longer
existed. The stack only self-corrected when a phantom key eventually reached the eviction tail
(`evict_one()` pops it, `erase` answers `KeyNotFound`, `apply_evictions` `continue`s), which on a
TTL-dominated workload — where objects leave by expiring rather than by eviction — could be a very
long time, or never. Measured directly: 8 objects × 1 KB reaped to `num_objects() == 0`, with
`lru_hybrid_stats().fast_bytes_used` still reporting **8,864 bytes across 8 objects**, indefinitely.

**Fix**: a new `WorkerEvent::Expire(HashedKey)`, sent point-to-point from `TtlWorker` to
`PolicyWorker` after each reap (`TtlWorker::notify_expired`), handled by
`PolicyWorker::handle_expire`, which delegates to the existing `handle_del` (policy stack + mini
stacks) — after one guard `handle_del` does not need.

Three decisions worth recording:

1. **A dedicated variant rather than reusing `Del`.** `Del`'s contract is "an API call just
   successfully erased this"; a reap is asynchronous with the policy worker, and the receiver has
   to behave differently because of it (point 2). Reusing `Del` would have meant changing
   `handle_del` for every existing caller to fix a problem only the reap path has.
2. **`handle_expire` re-reads the object map before touching the stack.** `TtlWorker` erases and
   *then* sends; nothing stops a `set()` on that key from landing in between, in which case the map
   entry has already been replaced by a live one and removing it from the stack would desync in the
   opposite, worse direction — an object present in the map but absent from the stack can never be
   chosen for eviction, and its bytes go unaccounted for in the tier gauges for as long as it
   lives. Re-reading *at handling time* (not send time) is what makes this safe: the re-set's
   `Set` and this `Expire` land in the same single-consumer channel, so in either arrival order the
   lookup agrees with the stack state the worker is about to produce.
3. **Sent point-to-point, not through `WorkerFanout`.** `TtlWorker` has no handle on the fanout
   (the fanout is built *from* the sub-workers it constructs, so handing it back would need an
   `Arc::new_cyclic` dance), and a fanned-out `Expire` would be delivered back to `TtlWorker`
   itself. `PolicyWorker` already takes the tiering worker's sender straight into its constructor
   (`promotion_tx`) — same pattern. `Events::EXPIRE` and its `POLICY_WORKER` mask bit are still
   added, so the file's stated "a worker's mask and its `run` arms are a single edit" rule holds
   and a future fanout-routed `Expire` would reach the right worker.

The notification is **unconditional**, not gated on `erase`'s result: `erase` answers
`KeyNotFound` both for a key that was already gone and for one it just successfully removed that
turned out to be expired — which is every reap, by construction (its trailing
`match !object.is_expired()`) — so the result cannot distinguish the two cases. Notifying in the
already-gone case is harmless: `handle_expire` re-checks the map, and `PolicyStack::remove` on an
untracked key is a no-op for every stack.

**A build-config trap this hit**: `handle_expire`'s map lookup needs `ObjectStore::get_ref`, but
`object_store` is itself gated on a *storage* feature (`all_dram`/`key_value_pmem`/
`global_hashtable_pmem`/`hashbrown_dram`), while the import in `worker/policy/mod.rs` had been
gated on the *hybrid* features that were its only prior users. Making it unconditional broke a bare
`eviction_stacks_pmem` build (no storage feature selected → no `object_store` module at all), which
had built clean before. Fixed with `PolicyWorker::object_exists`, gated the same way `object_store`
is, plus a second body using `DashMap::contains_key` directly for the no-storage-feature case.
Confirmed via `git stash` that the failure was this session's regression and not pre-existing —
worth repeating for any change here, since this crate's feature matrix makes "which cfg gates the
thing I need" a real question rather than a formality.

**Not changed (pre-existing, still true)**: `TieringWorker` (`enable_tiering_manager`) also
subscribes to `Del` and also goes stale on a TTL reap; it is not notified, exactly as before. It
would be a one-line mask/arm addition, deliberately left out to keep this change off a legacy path
whose own `--lib` tests already don't compile (documented above). Also unchanged: `Expire` is not
mapped in `StackEvent::maybe_from_worker_event`, so TTL reaps stay absent from the policy-switch
trace exactly as they were — mapping it to `StackEvent::Del` would be more accurate for
reconstruction but would record a removal `handle_expire` may decline to make, and the trace is
disabled entirely for every hybrid cache anyway (single policy, see `trace_is_useful`).

Separately worth knowing, found while reading this code and *not* fixed: `Expiries` is a
`BTreeMap<Instant, HashedKey>` — **one key per instant** — and `insert` is a plain
`BTreeMap::insert`, so two objects whose expiry lands on the exact same `Instant` silently evict
each other from the reaper index; the loser is then only cleaned up lazily on read or by eviction.
`get_expiry_from_ttl` is `Instant::now() + Duration::from_secs(ttl)`, so this needs a
nanosecond-exact collision — unlikely single-threaded, more plausible at `-c 8`. (`remove` already
guards with a key-match check before deleting, so the aliasing was at least anticipated.)

### Verified

Regression test `ttl_reap_clears_the_fast_tier_gauges` (plus
`a_key_re_set_after_expiring_is_still_tracked_by_the_stack`, covering decision 2's guard) in
`tests/lru_hybrid_cache_integration.rs`. **Negative-controlled**: with `notify_expired` commented
out, the first test fails on exactly the gauge assertion (`fast_objects=8 fast_bytes_used=8864`
after everything was reaped) and passes with it restored — the second passes either way, since it
guards against a regression this change could introduce rather than the original bug.

27 feature-flag build combos OK (all 17 hybrid variants, `all_dram`, `key_value_pmem`,
`global_hashtable_pmem`, `hashbrown_dram`, `eviction_stacks_pmem`, `tiering`, `multitiering`,
`key_value_pmem,enable_tiering_manager`, and two `*,eviction_stacks_pmem` pairs). Unit suites pass
unchanged (288 lru, 288 lfu, 289 two_q, 289 fifo, 291 lru_sized, 281 s3_fifo, 280 all_dram, 278
key_value_pmem, 281 global_hashtable_pmem, 278 hashbrown_dram). Real-PMEM integration suites pass
at their documented baselines (17/17 +2 ignored lru — 15 plus the 2 new tests — 19/19 lfu, 18/18
two_q, 14/14 fifo, 20/20 lru_sized, 20/20 s3_fifo), and `tiering_integration` 3/3. The lru suite
was run 5 consecutive times after the test-flakiness fix below, all green.

**Test-writing trap this re-taught**: the first version of the regression test used a 1 s TTL and
did not call `ensure_pmem_allocator_warm()`, on the reasoning that a test which never demotes never
touches PMEM and so has no warm-up to wait for. That reasoning is wrong in a way the module doc
already warned about: a *concurrently running* test in the same binary triggers the process-wide
warm-up and stalls everything, so all 8 keys expired and were reaped before the admission gauges
were ever observed and the pre-expiry baseline assertion failed. It passed on the first run and
failed on the second. Fixed by calling `ensure_pmem_allocator_warm()` (cheap after the first call
anywhere in the process) and raising the TTL to 5 s — the same pattern the other TTL tests here
already use.

## Feature: `lru_lfu_hybrid_cache` (implemented) — the first design whose two tiers rank by different metrics

Requested directly: "use lru for the fast tier and lfu for the slow tier ... how u would admit and
migrate objects across tiers."

Every prior hybrid here orders both tiers by the *same* metric, and that is load-bearing:
`LruHybridStack` is literally one recency list with a `fast_boundary` cursor marking the cut, and
`LfuHybridStack` runs two chains that both rank by frequency. Splitting the metrics breaks one rule
outright — **`LfuHybridStack`'s promotion rule is not portable**. It promotes when a slow object's
frequency exceeds the *fast tier's minimum frequency*; a recency-ordered fast tier has no
O(1)-queryable minimum frequency, and making it queryable means maintaining a frequency chain over
the fast tier *in addition to* its recency list, roughly doubling that tier's structural cost to
answer a question a constant answers for free. So promotion became a fixed threshold, `promote_k`.

**Motivation** (not "mix two policies for variety"): the tiers have different jobs. Recency is right
for the small hot tier and costs one splice per access. LRU over a very large slow tier is close to
meaningless — its tail is dominated by scans and one-hit-wonders — whereas LFU is scan-resistant and
actually identifies which cold objects deserve promotion and which deserve eviction. In one line:
**frequency is the admission control *into* DRAM; recency is the retention policy *within* DRAM.**

### Rules

- **Admission**: new object → fast tier, recency head, frequency 1. Admitting to slow instead would
  make every `set()` a synchronous PMEM allocation, which `two_q_hybrid` vs `two_q_fast_admission`
  measured at 2.15x SET mean / 2.72x SET p99 — and the traces this is aimed at are 80–94% SETs.
- **Demotion**: fast tier's LRU tail → slow chain via `FrequencyChain::insert_at`, **carrying its
  accumulated frequency**.
- **Promotion**: a slow object whose frequency reaches `promote_k` → fast tier's recency head,
  counter reset.
- **Eviction**: slow chain's minimum-frequency key (ties LRU-within-frequency, matching `LfuStack`),
  falling back to the fast tier's LRU tail when nothing has ever been demoted — the same last-resort
  fallback every hybrid stack here keeps.

### Three decisions worth recording

**1. The fast tier counts a frequency it does not rank by.** If demoted objects entered the slow
chain at frequency 1, everything the fast tier learned would be discarded — an object hit 50 times
in DRAM would land indistinguishable from a one-hit-wonder demoted in the same pass, and those are
exactly the objects most likely to be referenced again. So the counter is carried metadata, handed
to `insert_at` on demotion. Unit test `a_demoted_object_outranks_a_one_hit_wonder_in_the_slow_tier`
pins the payoff; the integration suite checks it end-to-end.

**2. The counter is a `u16`, and that is not a micro-optimization — it pays twice.** Verified
against real measured type sizes rather than assumed: `LruEntry { tier, size }` is `u8 + u32` = 8
bytes, pairing with the 8-byte `HashedKey` to exactly 16, which is the figure `object/overhead.rs`'s
DRAM-reservation constants are derived from. A `u32` counter gives `4+4+1 = 9` → padded to 12 → a
**24-byte pair, +8 bytes on every object in both tiers**. A `u16` gives `4+2+1 = 7` → padded back to
8, so the pair stays 16 and this design costs no more per-object DRAM than `LruHybridStack`. Locked
in by a unit test (`entry_packs_to_eight_bytes`) that asserts both sizes, so the overhead constant
can't silently drift. The same cap independently bounds `FrequencyChain::insert_at`'s **linear scan**
over count buckets — every demotion performs one into the *large* slow chain, and capping counts at
`1..=16` caps the chain at 16 buckets. Capping is what keeps demotion cheap, not just what keeps the
entry small.

**3. A `set()` is an access, not an automatic promotion** — a deliberate divergence from
`LruHybridStack`, where an overwrite always re-admits to the fast tier. The gate would otherwise be
porous on exactly the workloads this targets: on an 80–94% SET trace, "any set promotes" means
nearly everything reaches DRAM without demonstrating reuse and the slow tier's ordering never
filters anything. Consequence for the API layer: `admission_tier` must look up an *existing* key's
current tier (the `fifo_hybrid_cache` precedent) rather than answering purely from "is this key
new", so an overwrite of a slow key is written straight to PMEM instead of DRAM-then-corrected.

### Structure — simpler than `LruHybridStack`, not harder

```
fast_stack:  RecencyList      // fast tier only, recency-ordered
slow_chain:  FrequencyChain   // slow tier only, frequency-ordered
entries:     EntryMap         // { size: u32, freq: u16, tier: u8 } = 8 bytes
```

`fast_boundary` is **gone**. With the slow tier no longer part of the same list there is no cut to
track — the fast list's own tail *is* the demotion candidate in O(1) — and all the boundary-repair
logic `touch_fast_key`/`remove`/`evict_one` need in `LruHybridStack` simply does not exist here.
Same conclusion `LruSizedHybridStack` reached from the other direction: homogeneous per-tier
structures beat cursor tricks. `FrequencyChain` was lifted from `lfu_hybrid_stack.rs` minus
`min_count`/`bump` (only one tier needs a chain), plus `move_to` for reordering after a bump.

No new `PolicyStack` trait methods were needed, and no `drain_demotions()`-style disambiguation
either (unlike the LFU sibling): admission always lands fast and emits no migration, so every
`Tier::Slow` migration is a genuine demotion and every `Tier::Fast` one a genuine promotion.

### The bug my own tests caught: `promote_k` is an ABSOLUTE frequency, not "accesses since demotion"

The integration suite failed on `a_single_slow_access_does_not_promote` at `promote_k = 2`, and the
test was wrong in a way that exposed a real usability trap rather than a coding slip. Because the
counter carries across demotion, a key admitted and never accessed demotes at frequency **1** — so a
single slow access reaches 2. **`promote_k = 2` therefore behaves exactly like `promote_k = 1`, i.e.
exactly like `lru_hybrid_cache`, making the entire feature invisible at what looked like the obvious
default.** The first threshold that filters anything is **3**.

Kept the behavior rather than "fixing" it, for a reason worth stating: a key that was genuinely hot
before demoting arrives near the cap and returns on its very next access at any `promote_k`, and
that is the *desired* property — an object with demonstrated popularity needs one confirmation it is
still in use, while a one-hit-wonder must earn the whole climb. Sharpening this to true
"accesses since demotion" would need a second per-key counter (or the demotion-time frequency stored
alongside), which pushes `LruLfuEntry` past 8 bytes and forfeits decision 2 entirely. So the
semantics are now documented precisely in three places (stack module doc, the `promote_k` field, and
`PaperCache::new`), the integration default moved to 3, and a unit test
(`a_slow_access_below_the_absolute_threshold_does_not_promote`) asserts *both* halves — that k=2
promotes on one access, and that k=3 does not.

A second, more ordinary test bug: two unit tests sized the fast tier to exactly what should survive
and got an extra cascaded demotion, because `settle_fast_tier` triggers at the ceiling but drains to
`FAST_TIER_LOW_WATER_RATIO` (98%) of it. This is the same trap `LruHybridStack`'s tests already
document; fixed with the same `low_water_safe()` helper rather than by adjusting expectations to
whatever the code happened to do.

### Which trace can actually evaluate this

- **`cluster31` cannot.** Every one of its 82.9M GET keys is unique — zero GET reuse, 100%
  compulsory miss floor — so no promotion ever fires and the slow tier's frequency ordering never
  differentiates anything. Any measurement there is noise.
- **`cluster12` can.** 29.2% of GETs hit a previously-GET key, and its reachable working set
  (526 GiB) is two orders of magnitude larger than any plausible fast tier — exactly the regime
  where *which* slow object gets promoted and evicted dominates hit ratio, and where LRU on the slow
  tier is weakest.
- Watch on cluster12: sizes are bimodal (p50 6 B, p99 15 KB), and pure LFU eviction is size-blind,
  so it will retain many tiny popular objects over a few large ones. Hit ratio and byte-hit-ratio
  will diverge much more than on `standard_web.bin`. If byte-hit-ratio matters, frequency/size
  ranking (GDSF-style) is the natural follow-on.

### Verified

300/300 unit tests (17 new stack tests), 310/310 with `eviction_stacks_pmem`;
`tests/lru_lfu_hybrid_cache_integration.rs` 13/13, run twice consecutively; all 27 feature-flag
build combos OK; the mutual-exclusion `compile_error!` fires. No regressions — 12 other features'
`--lib` suites and 5 other real-PMEM integration suites pass at their documented baselines (17/17 +2
ignored lru, 19/19 lfu, 18/18 two_q, 14/14 fifo, 20/20 lru_sized), each run twice.

### Remaining work

- **`promote_k` is unswept.** The measurement above says 3 is the floor for it to mean anything, but
  nothing establishes the right value. Unlike `TWO_Q_K_IN` there is no benchmark env var for it yet;
  adding one (`LRU_LFU_PROMOTE_K`) is the prerequisite for a sweep without rebuilds.
- **No benchmark backend yet** — `paper-benchmark-cxl` has no `hybrid_lru_lfu` feature/backend, so
  this has not been run against a real trace at all. The crate side is complete and tested; the
  end-to-end comparison against `lru_hybrid_cache` on `cluster12` is the actual open question.
- **No aging beyond cap + reset-on-promotion.** Over a 7-day trace with shifting popularity, capped
  counts may still ossify. Real decay is O(n) over the slow chain — deliberately out of scope, but
  the thing to suspect if hit ratio degrades over a run's back half.

### Benchmark backend + synthetic-trace sweep across all 18 hybrid designs

`paper-benchmark-cxl` gained a `hybrid_lru_lfu` feature and `PaperCacheBackend`, following the same
shape as every other hybrid backend: `FAST_TIER_GB` (parsed as `f64` — see the already-documented
bug where 13 backends parsed it as `u64` and silently fell back to 4 GB on any fractional value),
plus a new `LRU_LFU_PROMOTE_K` env var so the threshold can be swept without a rebuild (same
convention as `TWO_Q_K_IN`). It defaults to **3**, not 1 or 2, for the reason documented above:
below 3 the design degenerates to `lru_hybrid_cache` and measuring it would be measuring nothing.

Then all **18** hybrid designs were run against all **3** synthetic traces
(`final_traces/{standard_web,low_alpha_cold,uniform_baseline}.bin`) — 54 runs, 0 failures, driver at
`work/synthetic_sweep.sh`, results in `work/synthetic_sweep/{summary.csv,raw/,progress.txt}`.

**Sizing is the part that makes the sweep meaningful, and it is not the default.** All three traces
are 100% GET, ~1M distinct keys of ~16.5 KB, WSS **15.6–16.7 GiB**. The benchmark's default
`--cache-max-size` is 24 GiB — *larger than the whole working set*, so nothing would ever evict and
all 18 designs would report identical, meaningless numbers. The sweep uses **8 GB cache (~48% of
WSS)** so eviction is real, and **2 GB fast tier (25% of the cache)** so demotion/promotion are real.

Three independent invariants held on all 54 runs: `fast_objects + slow_objects == num_objects`
exactly, `fast_bytes_used <= fast_tier_size`, and the `used_size` overhead residual matching each
design's `get_policy_overhead` constant.

#### Result: the new design does not win on these traces, and the reason is not a bug

| trace | `lru-hybrid` | `lfu-hybrid` | `lru-lfu-hybrid-3` | vs lru | vs lfu |
|---|---|---|---|---|---|
| standard_web | 0.1237 | **0.1159** | 0.1228 | **-0.7%** | +6.0% |
| low_alpha_cold | **0.3034** | 0.3300 | 0.3465 | **+14.2%** | +5.0% |
| uniform_baseline | **0.5274** | 0.5878 | 0.5588 | **+5.9%** | -4.9% |

Best overall per trace: `lfu-hybrid` (standard_web, 0.1159), `lru-sized-hybrid` (low_alpha_cold,
0.2868), `s3-fifo-lazy-demotion-fast-admission-midpoint-reprieve` (uniform_baseline, 0.5253).
`lru-lfu-hybrid-3` placed 8th / 9th / 7th of 18.

**The mechanism demonstrably works — it just doesn't pay here.** Promotions dropped ~40% against
`lru-hybrid` on every trace (1.03M vs 1.72M on standard_web; 217K vs 415K on uniform_baseline),
which is exactly what a frequency gate on promotion is supposed to do, and demotions fell with them.
So the gate is filtering real traffic; the filtered-out promotions simply were not the ones that
mattered on these workloads.

That is consistent with what these traces *are*. `uniform_baseline` is uniform-random, so frequency
carries **no** signal at all and an LFU-ordered slow tier is pure overhead against LRU — the design
losing there is the expected outcome, not a surprise. `low_alpha_cold` is low-skew for the same
reason. Only `standard_web` has enough popularity structure for frequency to mean anything, and
there it is a wash (-0.7%, within run-to-run noise).

It also is not the regime this was designed for. The argument for LFU in the slow tier is that LRU
becomes meaningless when the slow tier is *far* larger than the fast tier and dominated by a long
cold tail. Here the slow tier is 6 GB against a 2 GB fast tier — **3x**. On `cluster12` the
read-through-reachable working set is 526 GiB against a few GB of DRAM — **two orders of magnitude**
— which is the case actually worth testing, and which these traces cannot stand in for.

#### A configuration artifact worth knowing before reading any of these tables

Three designs — `2q-hybrid-0.1`, `s3-fifo-hybrid-0.1`, `2q-fast-admission-hybrid-0.1` — **never
filled the cache**, and their much worse miss ratios are mostly that, not policy quality:

| trace | design | used_size | miss |
|---|---|---|---|
| standard_web | `2q-hybrid-0.1` / `s3-fifo-hybrid-0.1` | 4.56 GB of 8 | 0.1816 |
| uniform_baseline | `2q-hybrid-0.1` / `s3-fifo-hybrid-0.1` | 2.43 GB of 8 | 0.9157 |
| uniform_baseline | `2q-fast-admission-hybrid-0.1` | 2.55 GB of 8 | 0.9073 |

This is the already-documented `k_in`/one-access-queue self-limiting effect (see
`two_q_fast_admission_reprieve_hybrid_cache`'s section: with nothing ever reprieved, the only exit
from the FIFO queue is eviction, so the cache stabilizes at roughly `fifo_capacity + main_fast`
rather than `max_size`). At `k_in = 0.1` against an 8 GB cache it is severe — on
`uniform_baseline` those designs used **30%** of the configured cache and evicted ~2M objects while
demoting 0. The reprieve variants of the same designs, which route aged-out one-access objects into
the slow tier instead of evicting them, fill the cache normally and score near the top. Any
comparison that reads these three as "2Q and S3-FIFO are bad policies" is reading a `k_in`
misconfiguration.

#### Remaining work

- **`LRU_LFU_PROMOTE_K` is still unswept** — every run above used 3. Whether a higher threshold
  helps on a trace with real popularity structure is untested.
- **`k_in` is unswept and is currently distorting three designs**, per above. A `k_in` sweep is
  worth more than another policy variant right now.
- **The traces that would actually test this design have not been run.** `cluster12` (526 GiB
  reachable WSS, 29.2% GET reuse) is the discriminating case; `cluster31` cannot evaluate it at all
  (zero GET reuse, 100% compulsory miss floor).
