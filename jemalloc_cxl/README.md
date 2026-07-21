# jemalloc_cxl

A prototype: **one** jemalloc instance, multiple arenas -- one of which is
`mmap`'d and NUMA-bound to a CXL/far-memory node via custom extent hooks,
reachable through a nightly `Allocator` handle.

```rust
#![feature(allocator_api)]
use jemalloc_cxl::{create_cxl_arena, CxlAllocator, CxlArenaConfig, NumaPolicy};

let arena = create_cxl_arena(CxlArenaConfig::new(1, NumaPolicy::Preferred))?;
let alloc = CxlAllocator::new(arena);

let mut v: Vec<u64, CxlAllocator> = Vec::new_in(alloc);
v.push(42);
```

Run the demo: `cargo +nightly run --release --example cxl_vec -- 1 bind`
(creates an arena on NUMA node 1, allocates a 512 MiB `Vec`, faults every
page in, and reports where it actually landed via `/proc/self/numa_maps`).

## This is one jemalloc instance, not two allocators

[`Jemalloc`](https://docs.rs/tikv-jemallocator) (from `tikv-jemallocator`) is
installed as `#[global_allocator]` in `src/lib.rs`. Every ordinary Rust
allocation -- any `Vec<T>`, `Box<T>`, `String` that doesn't explicitly name
a different allocator -- continues to go through that same instance exactly
as if this crate didn't exist.

[`create_cxl_arena`] doesn't start a second allocator. It asks that *same*
instance, via `mallctl("arenas.create")`, for a new independently-managed
**arena** -- jemalloc has always supported multiple arenas internally (it
creates several automatically, even without this crate, to reduce
cross-thread contention). This crate's only addition is attaching a custom
`extent_hooks_t` (see `src/extent.rs`) to one specific arena, so that
arena's memory is `mmap`'d and `mbind`'d onto a chosen NUMA node instead of
jemalloc's ordinary "wherever the OS chooses" mmap behavior.

[`CxlAllocator`] is a small `Copy` **routing handle** over that arena, not
an allocator implementation of its own -- `allocate`/`deallocate` are thin
wrappers around `mallocx`/`sdallocx` with an `MALLOCX_ARENA(..)` flag baked
in. It is best understood as "a `Vec`/`Box` that asks jemalloc for arena N's
memory," never as a competing allocator.

## tcache can obscure arena accounting

jemalloc gives each thread a **tcache** (thread-local allocation cache) by
default. Freed memory often goes back into the freeing thread's tcache
rather than immediately back to its arena -- so if you're watching
`stats.arenas.<i>.*` counters (or just eyeballing `/proc/self/numa_maps`)
expecting a `deallocate()` call to shrink an arena's footprint right away,
you may not see it happen promptly. [`TcacheMode::None`]
(`MALLOCX_TCACHE_NONE`) bypasses the tcache entirely, at some throughput
cost -- **use it for benchmarks or any experiment where you need arena
accounting to reflect reality immediately**, and leave [`TcacheMode::Automatic`]
for ordinary steady-state use.

## `MPOL_BIND` vs `MPOL_PREFERRED`

- [`NumaPolicy::BindStrict`] (`MPOL_BIND`): allocations/page-faults on the
  bound node only. If that node is under memory pressure, a page fault in
  this range can genuinely fail (OOM) rather than silently landing
  somewhere else -- appropriate when you need a hard placement guarantee
  and can tolerate/handle failure under pressure.
- [`NumaPolicy::Preferred`] (`MPOL_PREFERRED`): prefer the given node, but
  the kernel is free to fall back to another node rather than failing.
  Safer under memory pressure, but **does not guarantee placement** -- don't
  assume every byte landed on the requested node just because you asked for
  `Preferred`; check `/proc/self/numa_maps` (or `move_pages(2)`) if you need
  to confirm.

Both are implemented via the same `mbind(2)` syscall, always with
`MPOL_F_STATIC_NODES` OR'd into the *mode* argument (not passed as `mbind`'s
separate trailing `flags` argument -- those are two different flag
namespaces; see the doc comment on `extent::mbind` for how this was
confirmed empirically against this environment's kernel after an initial,
silently-wrong attempt at the other argument). Without
`MPOL_F_STATIC_NODES`, the node id in the mask would be interpreted
relative to the calling thread's cpuset/cgroup-allowed node list rather
than as an absolute physical node number -- surprising under any restrictive
cgroup.

**`mbind` sets a policy governing future page faults -- it is not itself an
immediate physical-placement operation.** A freshly `mmap`'d, never-touched
region has a NUMA policy attached but no physical pages yet; pages land on
the configured node only as each is first faulted in. `examples/cxl_vec.rs`
demonstrates this directly: it touches every element of a large `Vec`
*before* checking `/proc/self/numa_maps`, specifically so the reported
`N<node>=` resident-page counts reflect real, placed memory rather than
untouched virtual address space.

## Extent hooks are advanced jemalloc API and must remain valid forever

jemalloc retains the `extent_hooks_t*` pointer passed to `arenas.create` for
as long as that arena exists. This crate never destroys the arenas it
creates (matching jemalloc's own arena 0, which is also never torn down),
so the hooks must be valid for the rest of the process. A common pattern
for this is `Box::leak`-ing a fresh hook struct per arena; this crate uses
an even simpler variant: **one `'static` `ExtentHooks` value
(`extent::CXL_HOOKS`), shared by every CXL arena**, since the hooks contain
no per-arena state themselves -- they look up the calling `arena_ind` in a
small registry (`extent::register_arena_numa`) on every call instead. Either
approach is sound for the same underlying reason; this crate just never
needs more than one hook-struct instance to exist. See `src/arena.rs`'s doc
comment on `create_cxl_arena` for the fuller version of this note.

Two of the nine hooks are documented opt-outs rather than full
implementations, and the choice matters if you extend this crate:

- **`decommit`** always returns `true` (opt out). `commit` always reports
  "already committed" (this crate's memory is always a plain, fully-backed
  anonymous mapping, never a reserved-but-uncommitted region), so honoring a
  decommit request would also require handling a subsequent re-commit of
  that same range -- opting out keeps the contract simple. Memory is still
  reclaimable via `purge_forced` (`MADV_DONTNEED`).
- **`split`/`merge`** always return `true` (opt out). Splitting/merging a
  single `mmap` region into independently-freeable pieces is possible in
  principle but adds real bookkeeping this prototype doesn't need; jemalloc
  falls back to managing each extent as one indivisible unit.

## Don't mix allocation/deallocation APIs

Every allocation made via `mallocx(.., flags)` **must** be freed via
`dallocx`/`sdallocx` with **the same `MALLOCX_ARENA`/`MALLOCX_ALIGN`/tcache
flags** (or at least flags that resolve to the same arena/tcache) -- never
via plain `free()`, and never via a `CxlAllocator` value with different
`arena`/`tcache` fields than the one that allocated it. `CxlAllocator`'s
`Allocator` impl reconstructs the exact same flags in `deallocate` from its
own fields specifically to guarantee this, but if you ever reach for the raw
`ffi` functions directly, this invariant is now your responsibility.

## Nightly is required

Stable Rust's `Vec`/`Box` cannot be parameterized by a custom allocator --
`Vec<T, A>`/`Box<T, A>` and the `std::alloc::Allocator` trait are all gated
behind `#![feature(allocator_api)]`. There is currently no stable
workaround for ergonomic per-container custom-allocator routing; this
entire crate (and anything using [`CxlAllocator`]) requires
`cargo +nightly`.

## Layout

| File | Purpose |
|---|---|
| `native/jemalloc_shim.c` | Computes jemalloc's `MALLOCX_*` flag macros (not linkable symbols) as callable functions, against the exact `jemalloc.h` this crate is built against. |
| `build.rs` | Locates that header via `tikv-jemalloc-sys`'s build-script metadata (`DEP_JEMALLOC_ROOT`) and compiles the shim. |
| `src/ffi.rs` | Raw `extern "C"` declarations for `mallctl`/`mallocx`/`dallocx`/`sdallocx`/`nallocx` (against the real, `_rjem_`-prefixed symbols) and the shim functions. |
| `src/extent.rs` | The custom `extent_hooks_t`: `mmap`+`mbind` on alloc, `munmap` on dalloc/destroy, `madvise` on purge, documented opt-outs for decommit/split/merge. |
| `src/arena.rs` | `create_cxl_arena`, wrapping `mallctl("arenas.create")`. |
| `src/allocator.rs` | `CxlAllocator`, the nightly `Allocator` routing handle. |
| `src/thread_arena.rs` | `ThreadArenaGuard`, scoped whole-thread arena routing via `"thread.arena"`. |
| `examples/cxl_vec.rs` | End-to-end demo with real NUMA placement verification. |

## Platform support

Linux only. NUMA placement here is implemented via the Linux-specific
`mbind(2)` syscall and `MADV_FREE`/`MADV_DONTNEED`; neither exists on other
platforms. Building on a non-Linux target fails at compile time with a
clear `compile_error!`, rather than a confusing link error or a silently
wrong runtime behavior.

## Testing

```sh
cargo +nightly test                                  # unit tests (real jemalloc, real arenas)
cargo +nightly run --release --example cxl_vec -- 1 bind       # real NUMA placement demo, node 1, strict
cargo +nightly run --release --example cxl_vec -- 0 preferred  # node 0, soft preference
```

Tests that specifically need a second NUMA node (rather than just "any
working jemalloc instance") are marked `#[ignore]` and skip gracefully at
runtime if `/sys/devices/system/node/node1` doesn't exist, so the suite
still passes meaningfully on a single-node machine -- see
`tests/numa_integration.rs`.
