# tier_allocator

Runtime-parameterized NUMA-tier memory allocation for Rust, backed by
Intel's Unified Memory Framework (UMF).

## What this is

`TierAllocator` is a handle bound to a specific NUMA node (e.g. node 0 =
local DRAM, node 1 = CXL-attached memory), constructed at runtime via
`TierAllocator::new_numa(node)`. A single process can hold multiple live
instances side by side and pick tier placement per-allocation dynamically —
in contrast to the sibling `paper-cache-cxl` crate's existing allocator
types (`HybridObjects`, `DRAMObjects`, etc.), which hardcode their target
NUMA node as a compile-time constant selected via Cargo feature flags.

`TierAllocator::alloc(len)` returns a `TierBuffer` — a small, owned,
`Box<[u8]>`-like handle (`Deref`/`DerefMut` to `[u8]`) that frees itself on
`Drop`. This crate deliberately does **not** implement Rust's unstable
`std::alloc::Allocator` trait / `Vec::with_capacity_in` — see the
crate-level doc comment in `src/lib.rs` for why that hits a real orphan-rule
wall for the actual shared-pool use case this crate targets. The result:
**this crate builds on stable Rust**, no `#![feature(...)]` required
anywhere, by this crate or its callers.

## `TierAllocator` has no teardown

Once created, a `TierAllocator`'s underlying UMF pool lives for the rest of
the process — there is no `Drop` impl. This matches `paper-cache-cxl`'s own
`HybridObjects`/`DRAMObjects` precedent (background worker threads may still
be alloc/dealloc'ing after `main()` returns, so early teardown there caused
real UMF fatal asserts) and is exactly what lets a `TierBuffer` safely
outlive the specific `TierAllocator` value that allocated it. Construct one
`TierAllocator` per tier once (e.g. in a `std::sync::OnceLock`) and reuse it.

## ⚠️ Backend stability warning

The default pool backend is Intel TBB (`umfScalablePoolOps`) — the only
backend proven stable under real concurrent load against this UMF build
(1.0.3) in the sibling `paper-cache-cxl` crate's own testing. The optional
`jemalloc_pool` feature exposes `umfJemallocPoolOps` for experimentation
ONLY: it has crashed **three separate times** under real concurrent
multi-threaded load on this exact UMF version (twice a SIGSEGV inside UMF's
own critnib memory-tracker during jemalloc's internal extent-splitting, once
a corrupted/torn allocation-failure message under concurrent heap pressure).
All three were root-caused to UMF's own prebuilt library, not caller code.
**Do not enable `jemalloc_pool` in production expecting it to be safe.**

## Building

Links against an externally-installed UMF (no vendoring). Paths default to
this machine's install but are overridable:

```bash
export UMF_INCLUDE_DIR=/path/to/umf/include   # default: /home/griff/umf-install/include
export UMF_LIB_DIR=/path/to/umf/lib64         # default: /home/griff/umf-install/lib64
```

`build.rs` also runs `bindgen`, which needs `libclang` at build time. This
development machine has no system-wide `libclang` install and no root
access to add one via `dnf`; the working fix was installing the PyPI
`libclang` package (ships a prebuilt `libclang.so`, no root needed) and
pointing bindgen at it:

```bash
pip install --user libclang
export LIBCLANG_PATH=$(python3 -c "import clang, os; print(os.path.join(os.path.dirname(clang.__file__), 'native'))")

cargo build
cargo test
cargo test --features jemalloc_pool   # opt-in path; see stability warning above
```

## Future integration

A natural follow-up (not part of this crate) would let `paper-cache-cxl`
adopt `TierAllocator`/`TierBuffer` instead of its compile-time allocator
markers — e.g. `TieredBuffer::Slow(Box<[u8], Hybrid>)` becoming
`TieredBuffer::Slow(TierBuffer)`, a straightforward variant swap.
