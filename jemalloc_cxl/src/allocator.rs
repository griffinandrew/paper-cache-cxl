//! [`CxlAllocator`]: a nightly `Allocator` handle that routes individual
//! containers' allocations into a specific jemalloc arena via `mallocx`.
//!
//! This is a *routing handle*, not a second allocator implementation --
//! every byte it hands out still comes from the one jemalloc instance this
//! crate installs as `#[global_allocator]` ([`crate::Jemalloc`]); the only
//! difference from an ordinary `Vec<T>`/`Box<T>` is which arena's memory
//! (and therefore which extent hooks, and therefore which NUMA node) the
//! bytes are drawn from.

use std::alloc::{AllocError, Allocator, Layout};
use std::ffi::c_void;
use std::os::raw::c_int;
use std::ptr::NonNull;

use crate::arena::CxlArena;
use crate::ffi;

/// How a [`CxlAllocator`] should interact with jemalloc's per-thread
/// allocation cache (tcache).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TcacheMode {
    /// No explicit tcache flag -- jemalloc uses the calling thread's
    /// automatic tcache as it would for any ordinary allocation. Fastest
    /// for steady-state use, but see the README's note on tcache obscuring
    /// arena-level accounting (freed memory can sit in a tcache rather than
    /// being visible as returned to the arena).
    Automatic,
    /// `MALLOCX_TCACHE_NONE` -- bypass the tcache entirely. Slower, but
    /// gives accurate, immediate arena-level accounting; recommended for
    /// benchmarks and strict-accounting experiments (see README).
    None,
    /// `MALLOCX_TCACHE(ind)` -- route through a specific, explicitly
    /// created tcache (via `"tcache.create"`), for advanced use cases not
    /// otherwise covered by this crate.
    Explicit(i32),
}

impl TcacheMode {
    fn flag(self) -> c_int {
        match self {
            TcacheMode::Automatic => 0,
            TcacheMode::None => ffi::mallocx_tcache_none(),
            TcacheMode::Explicit(ind) => ffi::mallocx_tcache(ind),
        }
    }
}

/// A handle routing allocations into one jemalloc arena (a [`CxlArena`],
/// typically NUMA-bound to a CXL/far-memory node -- see
/// [`crate::arena::create_cxl_arena`]).
///
/// `Copy`, zero-sized-beyond-two-integers, and cheap to pass by value --
/// intended to be created once per arena and reused, e.g. as the `A` type
/// parameter of many `Vec<T, CxlAllocator>`/`Box<T, CxlAllocator>` values.
#[derive(Debug, Clone, Copy)]
pub struct CxlAllocator {
    arena: u32,
    tcache: TcacheMode,
}

impl CxlAllocator {
    /// A `CxlAllocator` for `arena`, using jemalloc's automatic per-thread
    /// tcache (see [`TcacheMode::Automatic`]).
    #[must_use]
    pub fn new(arena: CxlArena) -> Self {
        CxlAllocator {
            arena: arena.index(),
            tcache: TcacheMode::Automatic,
        }
    }

    /// A `CxlAllocator` for `arena` with an explicit tcache policy.
    #[must_use]
    pub fn with_tcache(arena: CxlArena, tcache: TcacheMode) -> Self {
        CxlAllocator {
            arena: arena.index(),
            tcache,
        }
    }

    /// The jemalloc arena index this allocator routes into.
    #[must_use]
    pub fn arena_index(&self) -> u32 {
        self.arena
    }

    fn mallocx_flags(&self, align: usize) -> c_int {
        let mut flags = ffi::mallocx_arena(self.arena);
        // jemalloc's own size classes already guarantee natural alignment
        // up to a point (its smallest size classes are 8/16-byte aligned);
        // MALLOCX_ALIGN is only needed -- and only well-defined -- for
        // alignments jemalloc wouldn't otherwise guarantee, so this only
        // adds the flag when a request actually exceeds pointer-width
        // alignment.
        if align > size_of::<usize>() {
            flags |= ffi::mallocx_align(align);
        }
        flags |= self.tcache.flag();
        flags
    }
}

// SAFETY: `Allocator`'s safety contract requires that a given allocator
// value always behaves consistently (same arena, same tcache policy) across
// calls, and that memory allocated through it is only ever deallocated
// through an allocator that compares equal to the one it was allocated
// with. `CxlAllocator` upholds both: its fields are plain data, never
// mutated after construction, and `mallocx`/`sdallocx` route purely on the
// `arena`/`tcache` flags encoded from those fields -- two `CxlAllocator`
// values with the same fields are always interchangeable.
unsafe impl Allocator for CxlAllocator {
    fn allocate(&self, layout: Layout) -> Result<NonNull<[u8]>, AllocError> {
        if layout.size() == 0 {
            // Nothing to allocate -- jemalloc is never called. Per the
            // `Allocator` contract, return a well-aligned, non-null,
            // never-to-be-dereferenced pointer built directly from the
            // requested alignment (a power of two, hence always a valid,
            // non-null pointer value on its own). `deallocate` below
            // recognizes a zero-size layout and correspondingly does
            // nothing, so allocate/deallocate stay paired.
            let dangling = NonNull::new(layout.align() as *mut u8).ok_or(AllocError)?;
            return Ok(NonNull::slice_from_raw_parts(dangling, 0));
        }

        let flags = self.mallocx_flags(layout.align());

        // SAFETY: `flags` encodes a valid arena index (this allocator's
        // `arena` field, always populated by `create_cxl_arena` before a
        // `CxlAllocator` referencing it can exist) and, when present, a
        // valid power-of-two alignment via MALLOCX_ALIGN -- exactly what
        // `mallocx` requires.
        let raw = unsafe { ffi::mallocx(layout.size(), flags) };

        let ptr = NonNull::new(raw as *mut u8).ok_or(AllocError)?;
        Ok(NonNull::slice_from_raw_parts(ptr, layout.size()))
    }

    unsafe fn deallocate(&self, ptr: NonNull<u8>, layout: Layout) {
        if layout.size() == 0 {
            // Paired with the zero-size case in `allocate` above: nothing
            // was actually allocated, so there is nothing to free.
            return;
        }

        let flags = self.mallocx_flags(layout.align());

        // SAFETY: caller guarantees (per `Allocator::deallocate`'s
        // contract) that `ptr`/`layout` describe a still-live allocation
        // previously returned by this same allocator's `allocate` -- so
        // `flags` here reconstructs the exact same arena/alignment/tcache
        // encoding `sdallocx` needs to free it correctly. `sdallocx` (the
        // "sized" variant) is used over plain `dallocx` because `layout`
        // already gives us the size for free, letting jemalloc skip its
        // own internal size lookup.
        unsafe {
            ffi::sdallocx(ptr.as_ptr() as *mut c_void, layout.size(), flags);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::arena::CxlArenaConfig;
    use crate::extent::NumaPolicy;

    fn test_allocator() -> CxlAllocator {
        let arena = crate::arena::create_cxl_arena(CxlArenaConfig::new(0, NumaPolicy::Preferred))
            .expect("arena creation on node 0 should succeed");
        CxlAllocator::new(arena)
    }

    #[test]
    fn vec_allocates_and_deallocates_through_the_cxl_arena() {
        let alloc = test_allocator();
        let mut v: Vec<u64, CxlAllocator> = Vec::new_in(alloc);
        for i in 0..1024u64 {
            v.push(i);
        }
        assert_eq!(v.len(), 1024);
        assert_eq!(v[1023], 1023);
        assert_eq!(v.iter().sum::<u64>(), (0..1024u64).sum());
        // dropped here -- exercises deallocate() via Vec's own Drop
    }

    #[test]
    fn boxed_slice_allocates_through_the_cxl_arena() {
        let alloc = test_allocator();
        let boxed = Box::new_zeroed_slice_in(4096, alloc);
        let mut boxed: Box<[u8], CxlAllocator> = unsafe { boxed.assume_init() };
        boxed.fill(0x42);
        assert!(boxed.iter().all(|&b| b == 0x42));
    }

    #[test]
    fn zero_sized_allocation_round_trips() {
        let alloc = test_allocator();
        let layout = Layout::from_size_align(0, 8).unwrap();
        let ptr = alloc.allocate(layout).expect("zero-size allocate should succeed");
        assert_eq!(ptr.len(), 0);
        // SAFETY: ptr/layout are exactly what allocate() just returned.
        unsafe {
            alloc.deallocate(ptr.cast(), layout);
        }
    }

    #[test]
    fn tcache_none_allocation_round_trips() {
        let arena = crate::arena::create_cxl_arena(CxlArenaConfig::new(0, NumaPolicy::Preferred))
            .expect("arena creation on node 0 should succeed");
        let alloc = CxlAllocator::with_tcache(arena, TcacheMode::None);
        let mut v: Vec<u8, CxlAllocator> = Vec::with_capacity_in(4096, alloc);
        v.extend(std::iter::repeat(7u8).take(4096));
        assert!(v.iter().all(|&b| b == 7));
    }
}
