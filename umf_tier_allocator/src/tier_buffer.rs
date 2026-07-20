use std::ops::{Deref, DerefMut};
use std::ptr::NonNull;

use crate::error::TierAllocError;
use crate::ffi;
use crate::tier_allocator::TierAllocator;

/// An owned byte buffer allocated from a [`TierAllocator`]'s UMF pool.
///
/// Holds a raw copy of its parent [`TierAllocator`]'s pool handle directly
/// (not an `Arc`, not a borrow) so it can free itself independently in its
/// own `Drop` impl. This is safe because `TierAllocator` deliberately has no
/// teardown (see that type's doc comment) -- the pool handle stays valid
/// for the life of the process regardless of what happens to the specific
/// `TierAllocator` value that created this buffer.
///
/// Behaves like `Box<[u8]>` via `Deref`/`DerefMut`/`AsRef<[u8]>`: indexing,
/// `.len()`, iteration, etc. all work as expected. Not `Clone` (owns unique
/// bytes, matching `Box<[u8]>` semantics) -- use [`TierBuffer::duplicate`]
/// for an explicit deep copy.
pub struct TierBuffer {
    ptr: NonNull<u8>,
    len: usize,
    // `None` for a zero-length buffer (never allocated via UMF, nothing to
    // free) -- see `TierBuffer::empty`.
    pool: Option<ffi::umf_memory_pool_handle_t>,
}

// SAFETY: bytes are uniquely owned by this value (no concurrent access to
// the same buffer without external synchronization, same as `Box<[u8]>`),
// and `umfPoolFree` is safe to call concurrently against the same pool from
// any thread (proven in the sibling `paper-cache-cxl` crate's own
// real-workload testing) -- same justification class as `unsafe impl
// Send/Sync for PaperCache<K, V, S>` in that crate's `src/lib.rs`.
unsafe impl Send for TierBuffer {}
unsafe impl Sync for TierBuffer {}

impl TierBuffer {
    /// Constructs an empty (zero-length) `TierBuffer` that never touches
    /// UMF at all -- there is nothing to allocate or later free.
    pub(crate) fn empty() -> Self {
        TierBuffer {
            ptr: NonNull::dangling(),
            len: 0,
            pool: None,
        }
    }

    /// Wraps an already-allocated UMF pointer. `ptr` must have been
    /// returned by `umfPoolAlignedMalloc(pool, len, _)` and not yet freed.
    pub(crate) fn from_raw(ptr: NonNull<u8>, len: usize, pool: ffi::umf_memory_pool_handle_t) -> Self {
        TierBuffer {
            ptr,
            len,
            pool: Some(pool),
        }
    }

    /// Returns the number of bytes in this buffer.
    #[must_use]
    pub fn len(&self) -> usize {
        self.len
    }

    /// Returns `true` if this buffer has zero length.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Allocates a new, independent `TierBuffer` from `allocator` with the
    /// same contents as `self`. Mirrors the copy-based tier-migration
    /// pattern the parent `paper-cache-cxl` crate's `TieredBuffer::new_fast`/
    /// `new_slow` already use, since an eventual integration would need
    /// exactly this operation.
    pub fn duplicate(&self, allocator: &TierAllocator) -> Result<TierBuffer, TierAllocError> {
        let mut new_buf = allocator.alloc(self.len)?;
        new_buf.copy_from_slice(self);
        Ok(new_buf)
    }
}

impl Deref for TierBuffer {
    type Target = [u8];

    fn deref(&self) -> &[u8] {
        // SAFETY: `ptr` is valid for `len` bytes for the lifetime of this
        // value (either dangling with len == 0, the `empty()` case, which
        // `from_raw_parts` documents as sound regardless of pointer
        // validity when len is 0; or a live UMF allocation of exactly
        // `len` bytes that this value uniquely owns).
        unsafe { std::slice::from_raw_parts(self.ptr.as_ptr(), self.len) }
    }
}

impl DerefMut for TierBuffer {
    fn deref_mut(&mut self) -> &mut [u8] {
        // SAFETY: see `Deref::deref` -- this value uniquely owns the bytes.
        unsafe { std::slice::from_raw_parts_mut(self.ptr.as_ptr(), self.len) }
    }
}

impl AsRef<[u8]> for TierBuffer {
    fn as_ref(&self) -> &[u8] {
        self
    }
}

impl Drop for TierBuffer {
    fn drop(&mut self) {
        let Some(pool) = self.pool else {
            return; // empty buffer: never allocated via UMF, nothing to free
        };

        let rc = unsafe { ffi::umfPoolFree(pool, self.ptr.as_ptr() as *mut _) };
        if rc != ffi::umf_result_t_UMF_RESULT_SUCCESS {
            // eprintln!, not println!: a println! here would lazily
            // initialize stdout's own buffer allocation on first use,
            // which -- per the sibling paper-cache-cxl crate's documented
            // "println! first-allocation deadlock" bug -- can recursively
            // re-enter allocator-adjacent init paths. Never panic in Drop.
            eprintln!("TierBuffer: umfPoolFree failed for {} bytes: {:?}", self.len, rc);
        }
    }
}
