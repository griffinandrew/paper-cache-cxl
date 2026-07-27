//! Raw FFI declarations for the one jemalloc instance this crate uses.
//!
//! `tikv-jemalloc-sys` (a direct dependency, also the source of the
//! `#[global_allocator]` in [`crate::Jemalloc`]) builds jemalloc with a
//! `_rjem_` symbol prefix (confirmed via that crate's own build-script
//! metadata: `JEMALLOC_PREFIX: _rjem_`), so the real, linked symbol names
//! are `_rjem_mallctl`, `_rjem_mallocx`, etc., not the bare names jemalloc's
//! own headers document. Declaring them here with `#[link_name]` lets the
//! rest of this crate call them under their ordinary names while linking
//! against the exact same static library `tikv-jemallocator` installs as
//! `#[global_allocator]` -- there is only one jemalloc instance in the
//! final binary, and every FFI call in this crate (mallctl for arena
//! creation/thread-arena switching, mallocx/dallocx/sdallocx for
//! [`crate::allocator::CxlAllocator`]) goes through it.

// dallocx/nallocx/mallocx_zero are part of the required FFI surface (see
// this crate's spec) but not currently called by any higher-level module
// here (deallocate() uses sdallocx, which folds size in rather than
// looking it up via a separate call; mallocx_zero has no current caller
// since CxlAllocator doesn't yet expose a zeroing allocation path) --
// kept available for direct/advanced use and future callers rather than
// removed.
#![allow(dead_code)]

use libc::{c_char, c_int, c_void, size_t};

unsafe extern "C" {
    #[link_name = "_rjem_mallctl"]
    pub fn mallctl(
        name: *const c_char,
        oldp: *mut c_void,
        oldlenp: *mut size_t,
        newp: *mut c_void,
        newlen: size_t,
    ) -> c_int;

    #[link_name = "_rjem_mallocx"]
    pub fn mallocx(size: size_t, flags: c_int) -> *mut c_void;

    #[link_name = "_rjem_dallocx"]
    pub fn dallocx(ptr: *mut c_void, flags: c_int);

    #[link_name = "_rjem_sdallocx"]
    pub fn sdallocx(ptr: *mut c_void, size: size_t, flags: c_int);

    #[link_name = "_rjem_nallocx"]
    pub fn nallocx(size: size_t, flags: c_int) -> size_t;
}

// The small C shim in native/jemalloc_shim.c -- computes jemalloc's
// MALLOCX_* flag macros (which can't be linked against directly; they're
// preprocessor macros, some involving `ffs()`) against the exact jemalloc.h
// this crate is built against. See build.rs for how that header is located.
unsafe extern "C" {
    fn jemalloc_cxl_shim_mallocx_arena(arena_ind: i32) -> i32;
    fn jemalloc_cxl_shim_mallocx_align(alignment: size_t) -> i32;
    fn jemalloc_cxl_shim_mallocx_tcache_none() -> i32;
    fn jemalloc_cxl_shim_mallocx_tcache(tcache_ind: i32) -> i32;
    fn jemalloc_cxl_shim_mallocx_zero() -> i32;
}

/// `MALLOCX_ARENA(arena_ind)` -- route this allocation to a specific arena.
#[must_use]
pub fn mallocx_arena(arena_ind: u32) -> c_int {
    // SAFETY: pure function of its integer argument, no side effects.
    unsafe { jemalloc_cxl_shim_mallocx_arena(arena_ind as i32) }
}

/// `MALLOCX_ALIGN(alignment)` -- `alignment` must be a power of two.
#[must_use]
pub fn mallocx_align(alignment: usize) -> c_int {
    // SAFETY: pure function of its integer argument, no side effects.
    unsafe { jemalloc_cxl_shim_mallocx_align(alignment) }
}

/// `MALLOCX_TCACHE_NONE` -- bypass the calling thread's tcache entirely.
#[must_use]
pub fn mallocx_tcache_none() -> c_int {
    // SAFETY: pure function, no arguments, no side effects.
    unsafe { jemalloc_cxl_shim_mallocx_tcache_none() }
}

/// `MALLOCX_TCACHE(tcache_ind)` -- route through a specific explicit tcache.
#[must_use]
pub fn mallocx_tcache(tcache_ind: i32) -> c_int {
    // SAFETY: pure function of its integer argument, no side effects.
    unsafe { jemalloc_cxl_shim_mallocx_tcache(tcache_ind) }
}

/// `MALLOCX_ZERO` -- zero-fill the allocation.
#[must_use]
pub fn mallocx_zero() -> c_int {
    // SAFETY: pure function, no arguments, no side effects.
    unsafe { jemalloc_cxl_shim_mallocx_zero() }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Expected values below are jemalloc.h's own macro definitions
    // (verified directly against this build's generated header -- see
    // build.rs / the crate-level docs), computed independently in Rust
    // here so a regression in the C shim (a typo, a wrong header pulled
    // in, ...) shows up as a test failure rather than a silent runtime
    // misroute.

    #[test]
    fn mallocx_arena_matches_macro_definition() {
        // #define MALLOCX_ARENA(a) ((((int)(a))+1) << 20)
        assert_eq!(mallocx_arena(0), 1 << 20);
        assert_eq!(mallocx_arena(5), 6 << 20);
        assert_eq!(mallocx_arena(33), 34 << 20);
    }

    #[test]
    fn mallocx_align_matches_macro_definition_for_page_sized_alignments() {
        // #define MALLOCX_ALIGN(a) ((int)(ffs((int)(a))-1)) on this
        // platform (LG_SIZEOF_PTR == 8) for a < INT_MAX -- ffs returns the
        // 1-based index of the least-significant set bit, so for a power
        // of two 2^n this is exactly n.
        assert_eq!(mallocx_align(1), 0);
        assert_eq!(mallocx_align(2), 1);
        assert_eq!(mallocx_align(4096), 12);
        assert_eq!(mallocx_align(1 << 21), 21);
    }

    #[test]
    fn mallocx_tcache_none_matches_macro_definition() {
        // #define MALLOCX_TCACHE(tc) ((int)(((tc)+2) << 8))
        // #define MALLOCX_TCACHE_NONE MALLOCX_TCACHE(-1)
        assert_eq!(mallocx_tcache_none(), ((-1i32 + 2) << 8));
        assert_eq!(mallocx_tcache_none(), mallocx_tcache(-1));
    }

    #[test]
    fn mallocx_tcache_matches_macro_definition() {
        assert_eq!(mallocx_tcache(0), 2 << 8);
        assert_eq!(mallocx_tcache(3), 5 << 8);
    }

    #[test]
    fn mallocx_zero_matches_macro_definition() {
        assert_eq!(mallocx_zero(), 0x40);
    }

    #[test]
    fn flags_from_different_macros_never_collide() {
        // A realistic combined-flags value (arena 33, 2 MiB alignment, no
        // tcache) should decode back losslessly -- i.e. the bit ranges
        // MALLOCX_ARENA/MALLOCX_ALIGN/MALLOCX_TCACHE occupy don't overlap
        // for realistic inputs, which is what makes OR-ing them together
        // in CxlAllocator::mallocx_flags meaningful.
        let combined = mallocx_arena(33) | mallocx_align(1 << 21) | mallocx_tcache_none();
        assert_eq!(combined & 0x1F, 21); // low 5 bits: alignment (2^21)
        assert_eq!((combined >> 8) & 0xFFF, 1); // tcache field: TCACHE(-1) = 1
        assert_eq!(combined >> 20, 34); // arena field: 33 + 1
    }
}
