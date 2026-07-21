use std::alloc::{GlobalAlloc, Layout};
use std::ptr;

use crate::ffi;
use crate::registry;

/// A const-constructible [`GlobalAlloc`] implementation bound to a single
/// NUMA node, backed by the same shared per-node registry that
/// [`crate::alloc_on`]/[`crate::allocator_for`] use.
///
/// This is the "implicit" access pattern: install one `NumaAllocator` as a
/// crate's `#[global_allocator]` and every ordinary `Box`/`Vec`/etc.
/// allocation on that node's default node routes through the shared
/// registry automatically. The "explicit" pattern ([`crate::alloc_on`]) is
/// for any *other* node -- both patterns resolve to the exact same
/// per-node pool, so there is exactly one real UMF pool per node, not two
/// independent allocator instances.
///
/// Holds only a plain `i32` node id (no pool handle, no `Arc`) -- the
/// actual pool is looked up from the shared registry on every `alloc`/
/// `dealloc` call. This is what makes the type `const fn`-constructible
/// (required for a `#[global_allocator]` static) despite UMF pool
/// construction itself being fallible and requiring real work.
#[derive(Debug)]
pub struct NumaAllocator {
    node: i32,
}

impl NumaAllocator {
    /// Constructs a `NumaAllocator` bound to `node`. Does not itself touch
    /// UMF -- the underlying pool is lazily constructed (via the shared
    /// registry) on first `alloc` call.
    #[must_use]
    pub const fn new(node: i32) -> Self {
        NumaAllocator { node }
    }

    /// The NUMA node this allocator is bound to.
    #[must_use]
    pub fn node(&self) -> i32 {
        self.node
    }
}

// No `unsafe impl Send`/`Sync` needed: the sole field is a plain `i32`,
// automatically `Send + Sync`, unlike `TierAllocator` (which holds an
// opaque FFI pool handle and needs an explicit impl).

unsafe impl GlobalAlloc for NumaAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let allocator = match registry::pool_for_node(self.node) {
            Ok(a) => a,
            Err(e) => {
                eprintln!("NumaAllocator: pool init failed for node {}: {e}", self.node);
                return ptr::null_mut();
            }
        };

        // GlobalAlloc::alloc's contract guarantees layout.size() > 0
        // always -- no zero-size special case needed here (unlike
        // TierAllocator::alloc_aligned, whose own 0-byte handling exists
        // for its own separate, explicit-API contract).
        let raw = unsafe {
            ffi::umfPoolAlignedMalloc(allocator.pool_handle(), layout.size(), layout.align())
        };

        if raw.is_null() {
            eprintln!(
                "NumaAllocator: UMF alloc failed for {} bytes on node {}",
                layout.size(),
                self.node
            );
        }

        raw as *mut u8
    }

    unsafe fn dealloc(&self, ptr: *mut u8, _layout: Layout) {
        // A pointer only ever reaches here if alloc() already succeeded for
        // this exact node, so this lookup is expected to always hit the
        // registry's cached Ok value. Never panic in dealloc: on the
        // (unreachable in practice) Err path, log and leak rather than
        // dereference an unknown handle -- matching the "eprintln!, never
        // panic in a free path" convention already used by TierBuffer's own
        // Drop impl.
        match registry::pool_for_node(self.node) {
            Ok(allocator) => {
                let rc = unsafe { ffi::umfPoolFree(allocator.pool_handle(), ptr as *mut _) };
                if rc != ffi::umf_result_t_UMF_RESULT_SUCCESS {
                    eprintln!("NumaAllocator: umfPoolFree failed on node {}: {:?}", self.node, rc);
                }
            }
            Err(e) => {
                eprintln!(
                    "NumaAllocator: dealloc for node {} whose pool was never initialized ({e}) -- leaking",
                    self.node
                );
            }
        }
    }
}
