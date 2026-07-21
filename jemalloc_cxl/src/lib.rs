//! One jemalloc instance, multiple arenas: a NUMA/CXL-tier arena reachable
//! through a nightly `Allocator` handle.
//!
//! # The one-instance model
//!
//! There is exactly one jemalloc runtime linked into a process using this
//! crate. [`Jemalloc`] (re-exported from `tikv-jemallocator`) is installed
//! as `#[global_allocator]`, so ordinary Rust code -- every `Vec<T>`,
//! `Box<T>`, `String`, etc. that doesn't explicitly name a different
//! allocator -- continues to allocate through that same instance exactly
//! as it would without this crate. [`arena::create_cxl_arena`] doesn't
//! start a second allocator; it asks that *same* instance for a new,
//! independently-managed arena (via `mallctl("arenas.create")`) with
//! custom extent hooks ([`extent`]) that `mmap`+`mbind` its memory onto a
//! chosen NUMA node. [`allocator::CxlAllocator`] is a small `Copy` handle
//! that routes a specific container's `mallocx`/`sdallocx` calls at that
//! arena; [`thread_arena::ThreadArenaGuard`] does the same for a whole
//! thread's *implicit* allocations, scoped to a guard's lifetime.
//!
//! # Module map
//!
//! - [`ffi`] -- raw `extern "C"` declarations: `mallctl`/`mallocx`/
//!   `dallocx`/`sdallocx`/`nallocx` (against jemalloc's real, `_rjem_`-
//!   prefixed symbols) plus the small C shim (`native/jemalloc_shim.c`)
//!   that computes `MALLOCX_ARENA`/`MALLOCX_ALIGN`/`MALLOCX_TCACHE(_NONE)`
//!   -- these are preprocessor macros in jemalloc.h, not linkable symbols,
//!   so a tiny C shim computes them rather than reimplementing jemalloc's
//!   own bit-twiddling (some of it involving `ffs()`) in Rust.
//! - [`extent`] -- the custom `extent_hooks_t` this crate attaches to every
//!   CXL arena: `mmap` + `mbind` on alloc, `munmap` on dalloc/destroy,
//!   `madvise` on purge, opt-outs for decommit/split/merge.
//! - [`arena`] -- [`arena::create_cxl_arena`], wrapping
//!   `mallctl("arenas.create")`.
//! - [`allocator`] -- [`allocator::CxlAllocator`], the nightly `Allocator`
//!   handle.
//! - [`thread_arena`] -- [`thread_arena::ThreadArenaGuard`], scoped
//!   whole-thread arena routing via `"thread.arena"`.
//!
//! See `README.md` for the full pitfalls list (tcache accounting, `mbind`'s
//! actual guarantees, extent-hook lifetime, why this needs nightly).

#![feature(allocator_api)]

#[cfg(not(target_os = "linux"))]
compile_error!(
    "jemalloc_cxl only supports Linux: NUMA placement here is implemented via the \
     Linux-specific mbind(2) syscall and MADV_FREE/MADV_DONTNEED, neither of which \
     exist on other platforms. See README.md."
);

mod ffi;

pub mod allocator;
pub mod arena;
pub mod extent;
pub mod thread_arena;

pub use allocator::{CxlAllocator, TcacheMode};
pub use arena::{create_cxl_arena, ArenaError, CxlArena, CxlArenaConfig};
pub use extent::NumaPolicy;
pub use thread_arena::{ThreadArenaError, ThreadArenaGuard};

/// The one jemalloc instance this crate (and every ordinary, non-CXL
/// allocation made while it's linked in) uses. Re-exported so downstream
/// crates don't need their own direct `tikv-jemallocator` dependency just
/// to see the type used here.
pub use tikv_jemallocator::Jemalloc;

#[global_allocator]
static GLOBAL: Jemalloc = Jemalloc;
