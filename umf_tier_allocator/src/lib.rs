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
//! `umfJemallocPoolOps` for experimentation ONLY -- it has crashed three
//! separate times under real concurrent multi-threaded load on this exact
//! UMF version (1.0.3): twice with a SIGSEGV inside UMF's own critnib
//! memory-tracker during jemalloc's internal extent-splitting, once with a
//! corrupted/torn allocation-failure message under concurrent heap
//! pressure. All three were root-caused to UMF's own prebuilt library, not
//! caller code, and are not fixable from this wrapper. **Do not enable
//! `jemalloc_pool` in production expecting it to be safe.**
//!
//! # Future integration
//!
//! A natural follow-up (not part of this crate) would let `paper-cache-cxl`
//! adopt `TierAllocator`/`TierBuffer` alongside or instead of its
//! compile-time `HybridObjects`/`DRAMObjects`/`ValueDRAM`/`DAXPMEM` markers
//! -- e.g. `TieredBuffer::Slow(Box<[u8], Hybrid>)` becoming
//! `TieredBuffer::Slow(TierBuffer)`, a straightforward variant swap. Out of
//! scope here.

mod error;
mod ffi;
mod tier_allocator;
mod tier_buffer;

pub use error::TierAllocError;
pub use tier_allocator::TierAllocator;
pub use tier_buffer::TierBuffer;
