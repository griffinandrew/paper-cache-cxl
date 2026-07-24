//! Creating a new jemalloc arena bound to a specific NUMA node.
//!
//! This is still the same one jemalloc instance as everything else in this
//! crate (see the crate-level docs) -- `arenas.create` just asks that one
//! instance for a new, independently-managed arena, and hands it a set of
//! extent hooks (see [`crate::extent`]) that mmap/`mbind` its memory onto a
//! chosen NUMA node instead of jemalloc's ordinary "wherever the OS
//! chooses" mmap behavior.

use std::ffi::c_void;
use std::mem::size_of;
use std::os::raw::c_uint;
use std::sync::Once;

use crate::extent::{self, ArenaNumaConfig, NumaPolicy, CXL_HOOKS, ExtentHooks};
use crate::ffi;

/// Enables jemalloc's background thread (a process-wide toggle, not
/// per-arena -- `mallctl("background_thread", ...)` affects every arena in
/// this one jemalloc instance) the first time any CXL arena is created.
///
/// **Why this matters, confirmed by direct measurement**: with
/// `background_thread` off (jemalloc's own default), its dirty/muzzy decay
/// purge is *tick-driven*, not timer-driven -- a decay check only runs as a
/// side effect of enough subsequent `alloc`/`dealloc` calls on that arena.
/// A workload that does a burst of allocation/deallocation and then goes
/// quiet (a very ordinary pattern -- e.g. a batch of cache admissions,
/// followed by an idle period) can leave large amounts of genuinely
/// freeable memory permanently unreclaimed, because nothing ever pokes the
/// ticker again to notice the decay window has passed. Reproduced directly:
/// 300,000 x 16 KiB allocations then freeing 90% (scattered) left real RSS
/// flat at ~5.9 GB even 35+ seconds later and even after an extra manual
/// alloc/dealloc "poke" -- zero `purge_lazy`/`purge_forced` hook calls the
/// entire time. An explicit `arena.<i>.purge` call *did* reclaim it (down
/// to ~645 MB, matching the ~469 MB expected survivor footprint plus normal
/// overhead), proving the memory was always genuinely freeable -- jemalloc
/// just never checked. Enabling `background_thread` (a dedicated thread
/// that sweeps arenas independent of allocation activity) reproduced that
/// same ~645 MB outcome automatically, within 15 seconds, with no explicit
/// purge call needed.
///
/// This requires the split/merge support the extent hooks gained alongside
/// this change (see `extent.rs`) -- without split/merge, an arena's freed
/// extents can never coalesce back into the larger contiguous free regions
/// jemalloc's own decay/purge logic acts on, so there would be far less for
/// `background_thread` to actually reclaim.
fn ensure_background_thread_enabled() {
    static INIT: Once = Once::new();
    INIT.call_once(|| {
        let mut want: bool = true;
        // SAFETY: `newp` points at `want`, a valid `bool` for the duration
        // of this call, with `newlen` matching its size exactly; `oldp`/
        // `oldlenp` are null, meaning "caller does not want the previous
        // value back" (we don't need it -- this only ever runs once, and
        // we're setting an intentional new value regardless of what it was
        // before).
        let rc = unsafe {
            ffi::mallctl(
                c"background_thread".as_ptr(),
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                (&raw mut want).cast::<c_void>(),
                size_of::<bool>(),
            )
        };
        if rc != 0 {
            eprintln!(
                "jemalloc_cxl: failed to enable background_thread (mallctl error {rc}) -- CXL \
                 arenas will still function, but freed memory may not be reclaimed automatically \
                 without an explicit arena.<i>.purge call"
            );
        }
    });
}

/// Configuration for a new CXL/NUMA-tier arena.
#[derive(Debug, Clone, Copy)]
pub struct CxlArenaConfig {
    /// The NUMA node every extent allocated in this arena will be bound to.
    pub numa_node: u32,
    /// Strict (`MPOL_BIND`) or soft (`MPOL_PREFERRED`) placement.
    pub policy: NumaPolicy,
}

impl CxlArenaConfig {
    #[must_use]
    pub fn new(numa_node: u32, policy: NumaPolicy) -> Self {
        CxlArenaConfig { numa_node, policy }
    }
}

/// A created jemalloc arena, routed to a specific NUMA node.
///
/// This is a plain arena index into the one jemalloc instance -- not a
/// separate allocator, not a separate heap implementation. Use it with
/// [`crate::allocator::CxlAllocator`] to route `Vec`/`Box` allocations into
/// it, or with [`crate::thread_arena::ThreadArenaGuard`] to route a whole
/// thread's *implicit* allocations into it for a scope.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CxlArena {
    index: c_uint,
}

impl CxlArena {
    /// The raw jemalloc arena index (as `MALLOCX_ARENA(index)` would encode
    /// it, or as `"thread.arena"` expects it).
    #[must_use]
    pub fn index(&self) -> u32 {
        self.index
    }
}

#[derive(Debug, thiserror::Error)]
pub enum ArenaError {
    #[error("mallctl(\"arenas.create\") failed with jemalloc error code {0}")]
    MallctlFailed(i32),
}

/// Creates a new jemalloc arena whose memory is `mmap`'d and NUMA-bound
/// per `config`, via `mallctl("arenas.create", ...)` with this crate's
/// shared [`CXL_HOOKS`] attached.
///
/// # Hook storage lifetime
///
/// jemalloc retains the `extent_hooks_t*` pointer passed here for as long
/// as the arena exists, and this crate never destroys arenas it creates
/// (process-lifetime, matching jemalloc's own arena-0 which is never torn
/// down either) -- so the hooks must be valid for the rest of the process.
/// A common pattern for this is `Box::leak`-ing a freshly allocated hook
/// struct per arena; this crate uses an even simpler variant of the same
/// idea: **one `'static` [`ExtentHooks`] value, shared by every CXL arena**,
/// since the hooks themselves contain no per-arena state -- they look up
/// the calling `arena_ind` in [`extent`]'s registry on every call instead.
/// Either approach is sound for the same reason (the pointer jemalloc holds
/// is valid forever); this crate just never needs more than one such
/// pointer to exist.
pub fn create_cxl_arena(config: CxlArenaConfig) -> Result<CxlArena, ArenaError> {
    ensure_background_thread_enabled();

    let mut arena_ind: c_uint = 0;
    let mut arena_ind_size = size_of::<c_uint>();

    // `arenas.create`'s newp, if provided, must point to an
    // `extent_hooks_t*` (a pointer to the hooks struct, not the struct
    // itself) -- so we pass the address of this local pointer variable.
    let mut hooks_ptr: *const ExtentHooks = &CXL_HOOKS;

    // SAFETY: `oldp`/`oldlenp` point at valid, appropriately-sized local
    // variables (`arena_ind`/`arena_ind_size`) for jemalloc to write the
    // new arena index into; `newp` points at `hooks_ptr`, a valid
    // `*const ExtentHooks` for the duration of this call, with `newlen`
    // matching a pointer's size exactly, as `arenas.create` requires.
    let rc = unsafe {
        ffi::mallctl(
            c"arenas.create".as_ptr(),
            (&raw mut arena_ind).cast::<c_void>(),
            &raw mut arena_ind_size,
            (&raw mut hooks_ptr).cast::<c_void>(),
            size_of::<*const ExtentHooks>(),
        )
    };

    if rc != 0 {
        return Err(ArenaError::MallctlFailed(rc));
    }

    extent::register_arena_numa(
        arena_ind,
        ArenaNumaConfig {
            node: config.numa_node,
            policy: config.policy,
        },
    );

    Ok(CxlArena { index: arena_ind })
}

#[cfg(test)]
mod tests {
    use super::*;

    // Requires a real jemalloc instance (always true when this crate links
    // -- tikv-jemallocator is a hard dependency, not optional) but not any
    // particular NUMA topology: node 0 always exists.
    #[test]
    fn create_cxl_arena_on_node_0_succeeds() {
        let arena = create_cxl_arena(CxlArenaConfig::new(0, NumaPolicy::Preferred))
            .expect("arena creation on node 0 should succeed");
        assert!(arena.index() > 0, "arena 0 is jemalloc's own default arena");
    }

    #[test]
    fn distinct_calls_produce_distinct_arenas() {
        let a = create_cxl_arena(CxlArenaConfig::new(0, NumaPolicy::Preferred)).unwrap();
        let b = create_cxl_arena(CxlArenaConfig::new(0, NumaPolicy::Preferred)).unwrap();
        assert_ne!(a.index(), b.index());
    }
}
