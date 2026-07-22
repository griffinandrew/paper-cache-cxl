//! `tier_allocator`: place byte buffers on a specific NUMA memory tier at
//! runtime, backed by Intel's Unified Memory Framework (UMF).
//!
//! # Design
//!
//! [`TierAllocator`] is a runtime-constructed handle bound to a NUMA node
//! (e.g. node 0 = local DRAM, node 1 = CXL-attached memory), in contrast to
//! the sibling `paper-cache-cxl` crate's existing `HybridObjects`/
//! `DRAMObjects`/`ValueDRAM`/`DAXPMEM` allocator types, which hardcode their
//! target NUMA node as a compile-time constant selected via Cargo feature
//! flags. A single process can hold multiple live `TierAllocator` instances
//! side by side and pick tier placement per-allocation at runtime, with no
//! rebuild required.
//!
//! ## Why not `std::alloc::Allocator` + `Vec::with_capacity_in`?
//!
//! An earlier design used Rust's unstable `Allocator` trait so ordinary
//! `Vec`/`Box` could opt into a tier via `Vec::with_capacity_in(cap, &tier)`.
//! That hits a real wall for a shared-pool use case (e.g. an object store
//! where many entries need to share one long-lived pool per tier): making
//! the allocator handle an owned, `'static`, shareable value would require
//! something like `Arc<TierAllocator>` to itself implement `Allocator` --
//! but Rust's orphan/coherence rules forbid implementing a foreign trait
//! (`Allocator`) for a foreign generic type (`Arc<T>`) wrapping a local
//! type. A bare `&TierAllocator` reference sidesteps that, but forces a
//! lifetime onto every buffer. `Vec::with_capacity_in` also requires every
//! downstream crate touching the resulting `Vec`/`Box` to independently
//! enable `#![feature(allocator_api)]` and build on nightly.
//!
//! [`TierBuffer`] avoids all of that: it's a plain, hand-rolled owned type
//! that holds the raw UMF pool handle directly (a `Copy`-able opaque
//! pointer, not an `Arc`, not a borrow) and frees itself in its own `Drop`.
//! No orphan-rule problem, no nightly feature required anywhere -- this
//! crate and its callers build on stable Rust.
//!
//! ## `TierAllocator` has no teardown
//!
//! `TierAllocator` deliberately does not implement `Drop`. Once created, its
//! underlying UMF pool lives for the rest of the process. This matches the
//! sibling `paper-cache-cxl` crate's own `HybridObjects`/`DRAMObjects`
//! precedent (documented there as intentional: background worker threads
//! may still be alloc/dealloc'ing after `main()` returns, so early teardown
//! caused real UMF fatal asserts). It's also exactly what makes
//! [`TierBuffer`] safe to copy a pool handle out of a `TierAllocator` and
//! outlive the specific value that created it -- the pool itself never goes
//! away. Intended usage: construct one `TierAllocator` per tier once (e.g.
//! in a `std::sync::OnceLock` or a long-lived `static`), then call
//! [`TierAllocator::alloc`] on it as many times as needed.
//!
//! # Backend stability warning
//!
//! The default pool backend is Intel TBB (`umfScalablePoolOps`), the only
//! backend proven stable under real concurrent load against this UMF build
//! in the sibling `paper-cache-cxl` crate's own testing (see that crate's
//! `CLAUDE.md`). The optional `jemalloc_pool` feature exposes
//! `umfJemallocPoolOps` for experimentation ONLY -- it has crashed **four**
//! separate times under real concurrent multi-threaded load on this exact
//! UMF version (1.0.3): twice with a SIGSEGV inside UMF's own critnib
//! memory-tracker during jemalloc's internal extent-splitting, once with a
//! corrupted/torn allocation-failure message under concurrent heap
//! pressure, and once (this crate's own `registry.rs` wired to use
//! `new_numa_jemalloc` uniformly for both `NumaAllocator`'s
//! `#[global_allocator]` role and explicit `alloc_on` -- i.e. jemalloc used
//! exactly the way TBB is used by default) with a SIGSEGV inside jemalloc's
//! *own* internal extent-coalescing code (`ph_remove` /
//! `je_edata_heap_remove` / `extent_coalesce`, a null-pointer dereference in
//! jemalloc's pairing-heap free-extent tracking, confirmed via `gdb -batch
//! -ex run -ex "thread apply all bt full"` against the real
//! `paper-benchmark-cxl` benchmark, `-c 8`, a 14M-access real trace,
//! `lru_hybrid_cache` + `umf_jemalloc_pool`). All four were root-caused to
//! UMF's own prebuilt library and/or jemalloc's own internals as wired by
//! UMF, not caller code, and are not fixable from this wrapper -- the newest
//! crash is in a genuinely different function than the first three, but the
//! same overall subsystem (jemalloc extent/free-list management under real
//! concurrent allocation pressure, as integrated by UMF). **Do not enable
//! `jemalloc_pool` in production expecting it to be safe.**
//!
//! A third backend, initially promising but now also confirmed unsafe: the
//! optional `disjoint_pool` feature exposes `umfDisjointPoolOps` -- UMF's
//! *own* pool implementation, not a third-party allocator. Architecturally,
//! it buckets allocations by size class and tracks each bucket's slabs
//! individually with an explicit `capacity`; there is no cross-size-class
//! extent coalescing at all, so it has no analog to the eset/pairing-heap
//! code that crashed `jemalloc_pool`. A controlled 300k-object, 90%-freed
//! reproduction showed real settled memory landing within ~1.23x of the
//! true theoretical minimum (`slab_min_size` tuned near the real average
//! object size), versus TBB's ~1.0x-of-*peak* under the identical test --
//! and a standalone 24M-allocation/8-thread stress test (direct
//! `TierAllocator::alloc` calls, not installed as `#[global_allocator]`)
//! passed cleanly with correct checksums.
//!
//! **That standalone result did not hold up once wired into the real
//! `paper-cache-cxl` integration.** A direct, minimal reproduction (`cargo
//! +nightly test --test lru_hybrid_cache_integration --features
//! lru_hybrid_cache,umf_disjoint_pool
//! concurrent_set_from_multiple_threads_still_demotes -- --ignored
//! --nocapture`, in that sibling crate): N threads concurrently calling
//! `set()`, well within available memory on both nodes. TBB passes cleanly
//! at every thread count tried; the disjoint pool passes at 2 and 4
//! threads but reliably fails at 6+ with spurious allocation failures on
//! **both** the fast tier's global-allocator pool (node 0) and the slow
//! tier's independent explicit-`alloc_on` pool (node 1) simultaneously --
//! two separate pool instances failing together rules out ordinary
//! per-node memory exhaustion and points at a genuine concurrency bug
//! inside `umfDisjointPoolOps` itself. The standalone stress test evidently
//! never exercised the real pattern of this pool serving as the *entire*
//! process's global allocator under the full, varied allocation traffic a
//! real multi-threaded Rust program generates (thread stacks, hashmap
//! bucket growth, channel internals, etc., not just a narrow fixed set of
//! explicitly-sized test buffers). **Do not enable `disjoint_pool` in
//! production** until this is root-caused or fixed upstream in UMF --
//! TBB remains the only backend proven stable under real concurrent load.
//!
//! # Dual access: `#[global_allocator]` vs explicit `alloc_on`
//!
//! [`NumaAllocator`] and the free functions in this module ([`alloc_on`],
//! [`allocator_for`]) are two *access patterns* over one shared mechanism,
//! not two allocators. Both resolve to the same lazily-initialized,
//! per-NUMA-node pool registry (internal `registry` module): `NumaAllocator`
//! implements `GlobalAlloc` so it can be installed as a crate's
//! `#[global_allocator]`, giving every ordinary `Box`/`Vec` allocation
//! implicit access to whichever node it's bound to; `alloc_on(node, len)`
//! is for reaching any *other* node explicitly. Since every call site knows
//! its target node statically -- the global allocator's node is fixed at
//! construction, and every `alloc_on` call names its node directly --
//! `dealloc` never needs to guess which pool a pointer came from, unlike a
//! thread-local "current node" design would.
//!
//! This is deliberately not the thread-local-scoped design considered
//! earlier: reading an ambient "current node" at allocation time would
//! leave `dealloc` unable to trust that same thread-local at free time (an
//! allocation may outlive the thread/scope that created it), forcing a
//! cross-pool pointer-to-pool lookup on *every* free via UMF's `umfFree`
//! auto-detection -- a real cost, and more exposure to the same UMF
//! memory-tracking subsystem already implicated in the sibling
//! `paper-cache-cxl` crate's documented jemalloc-pool crash history. Every
//! access pattern here keeps its target node statically known instead, so
//! `dealloc` is always a direct, single-pool free.
//!
//! One concrete consequence: a caller that installs `NumaAllocator::new(0)`
//! as `#[global_allocator]` gets an ordinary `Box<[u8]>` on node 0 "for
//! free" (it's just a normal heap allocation once installed), while still
//! being able to call `alloc_on(1, len)` to place bytes on node 1
//! explicitly via a [`TierBuffer`] -- both routes share the exact same
//! per-node pool as any other caller asking for that node, so there's
//! never a redundant second pool for a node that's already in use.
//!
//! # Future integration
//!
//! `paper-cache-cxl`'s `lru_hybrid_cache`/`lfu_hybrid_cache`/
//! `two_q_hybrid_cache`/`fifo_hybrid_cache` features are the first
//! consumers of this dual-access design: `NumaAllocator::new(0)` becomes
//! their `#[global_allocator]` (replacing that crate's own `DRAMObjects`
//! for those four features only), and `TieredBuffer::Fast` collapses to a
//! plain `Box<[u8]>` (an ordinary allocation, now implicitly on this same
//! node-0 pool), while `TieredBuffer::Slow` uses `alloc_on(1, len)`
//! explicitly, same as it already did via `TierAllocator` directly.

mod error;
mod ffi;
mod numa_allocator;
mod registry;
mod tier_allocator;
mod tier_buffer;

pub use error::TierAllocError;
pub use numa_allocator::NumaAllocator;
pub use registry::{alloc_on, alloc_on_aligned, allocator_for};
pub use tier_allocator::TierAllocator;
pub use tier_buffer::TierBuffer;
