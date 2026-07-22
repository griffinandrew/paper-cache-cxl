use std::ptr;

use crate::error::TierAllocError;
use crate::ffi;
use crate::tier_buffer::TierBuffer;

/// Bytes per allocation-granularity chunk the pool requests from its
/// underlying memory provider. Copied from the parent `paper-cache-cxl`
/// crate's own proven-working TBB-pool tuning
/// (`umf_allocator/umf_allocator_wrapper.c`) as a safe v0.1 default.
///
/// NOTE(tuning): this is cache-workload-specific tuning (many small,
/// similarly-sized objects with staggered lifetimes) baked into a nominally
/// generic library. A different workload (e.g. one large long-lived buffer)
/// might want a different value. A builder-style override on `new_numa` is
/// a natural v0.2 follow-up rather than something to solve now.
const POOL_GRANULARITY_BYTES: usize = 2 * 1024 * 1024;

fn check(rc: ffi::umf_result_t, err: impl FnOnce(ffi::umf_result_t) -> TierAllocError) -> Result<(), TierAllocError> {
    if rc == ffi::umf_result_t_UMF_RESULT_SUCCESS {
        Ok(())
    } else {
        Err(err(rc))
    }
}

/// A handle to a UMF memory pool bound to a single NUMA node.
///
/// This is a plain, `Copy`-able value -- just a pool handle and a node id,
/// no ownership semantics to enforce -- because **it deliberately has no
/// `Drop` impl**. See the crate-level doc comment for the full "pools live
/// for the process" rationale: this matches the parent `paper-cache-cxl`
/// crate's own `HybridObjects`/`DRAMObjects` precedent (never torn down,
/// since background worker threads may still be alloc/dealloc'ing after
/// `main()` returns), and is what lets a [`TierBuffer`] allocated from a
/// `TierAllocator` safely outlive the specific `TierAllocator` value that
/// created it -- the underlying pool itself never goes away.
///
/// Intended usage: construct one `TierAllocator` per tier once (e.g. stored
/// in a `std::sync::OnceLock` or a long-lived `static`), and call
/// [`TierAllocator::alloc`] on it as many times as needed for the life of
/// the process.
#[derive(Clone, Copy, Debug)]
pub struct TierAllocator {
    pool: ffi::umf_memory_pool_handle_t,
    node: i32,
}

// SAFETY: `pool` is a UMF memory-pool handle. UMF's scalable-pool backend
// (Intel TBB) is documented, and proven in the sibling `paper-cache-cxl`
// crate's own real-workload testing, to be safe for concurrent alloc/free
// from multiple threads against the same pool handle -- mirroring the
// identical class of argument used for `unsafe impl Send/Sync for
// PaperCache<K, V, S>` in that crate (src/lib.rs), which is also backed by
// internally-synchronized third-party state rather than any
// synchronization this type performs itself.
unsafe impl Send for TierAllocator {}
unsafe impl Sync for TierAllocator {}

impl TierAllocator {
    /// Constructs a `TierAllocator` bound to NUMA node `node`, backed by
    /// UMF's scalable (Intel TBB) pool -- the only pool backend proven
    /// stable under real concurrent load in this environment (see the
    /// crate-level doc comment's jemalloc-pool warning).
    ///
    /// Note: this succeeding does **not** confirm `node` actually exists on
    /// the running machine. UMF's OS memory provider only stores the node
    /// list/mode at construction time; the underlying `mbind()`/`mmap()`
    /// call that would reject a nonexistent node is deferred to the first
    /// [`TierAllocator::alloc`] call, which surfaces the failure there
    /// instead (confirmed empirically: `new_numa(9999)` succeeds on this
    /// development machine, but the first `alloc()` against it fails).
    pub fn new_numa(node: i32) -> Result<Self, TierAllocError> {
        unsafe {
            // 1. OS memory provider params: bind to exactly this node.
            let mut provider_params: ffi::umf_os_memory_provider_params_handle_t = ptr::null_mut();
            check(
                ffi::umfOsMemoryProviderParamsCreate(&mut provider_params),
                TierAllocError::ProviderParamsCreate,
            )?;

            let mut numa_list: [u32; 1] = [node as u32];
            let set_list_rc = ffi::umfOsMemoryProviderParamsSetNumaList(
                provider_params,
                numa_list.as_mut_ptr(),
                1,
            );
            if let Err(e) = check(set_list_rc, TierAllocError::ProviderParamsSetNumaList) {
                ffi::umfOsMemoryProviderParamsDestroy(provider_params);
                return Err(e);
            }

            // Required alongside SetNumaList: UMF_NUMA_MODE_DEFAULT (the
            // implicit default otherwise) requires the nodemask to be
            // NULL/empty per UMF's own header doc comment, so the numa_list
            // set above would otherwise be silently ignored/invalid.
            let set_mode_rc = ffi::umfOsMemoryProviderParamsSetNumaMode(
                provider_params,
                ffi::umf_numa_mode_t_UMF_NUMA_MODE_BIND,
            );
            if let Err(e) = check(set_mode_rc, TierAllocError::ProviderParamsSetNumaMode) {
                ffi::umfOsMemoryProviderParamsDestroy(provider_params);
                return Err(e);
            }

            // 2. Provider.
            let mut provider: ffi::umf_memory_provider_handle_t = ptr::null_mut();
            let create_provider_rc = ffi::umfMemoryProviderCreate(
                ffi::umfOsMemoryProviderOps(),
                provider_params as *const _,
                &mut provider,
            );
            // Params are a transient object -- always destroyed once the
            // provider is created, regardless of outcome.
            ffi::umfOsMemoryProviderParamsDestroy(provider_params);
            check(create_provider_rc, TierAllocError::ProviderCreate)?;

            // 3. Scalable (TBB) pool params.
            let mut pool_params: ffi::umf_scalable_pool_params_handle_t = ptr::null_mut();
            if let Err(e) = check(
                ffi::umfScalablePoolParamsCreate(&mut pool_params),
                TierAllocError::PoolParamsCreate,
            ) {
                // Construction failed before any pool exists to own the
                // provider -- this is the one place in this crate a UMF
                // teardown function is ever called (see the crate-level
                // "no teardown" doc comment): the provider was never
                // successfully handed off to a pool, so it would otherwise
                // leak.
                ffi::umfMemoryProviderDestroy(provider);
                return Err(e);
            }
            if let Err(e) = check(
                ffi::umfScalablePoolParamsSetKeepAllMemory(pool_params, true),
                TierAllocError::PoolParamsSetKeepAllMemory,
            ) {
                ffi::umfScalablePoolParamsDestroy(pool_params);
                ffi::umfMemoryProviderDestroy(provider);
                return Err(e);
            }
            if let Err(e) = check(
                ffi::umfScalablePoolParamsSetGranularity(pool_params, POOL_GRANULARITY_BYTES),
                TierAllocError::PoolParamsSetGranularity,
            ) {
                ffi::umfScalablePoolParamsDestroy(pool_params);
                ffi::umfMemoryProviderDestroy(provider);
                return Err(e);
            }

            // 4. Pool. UMF_POOL_CREATE_FLAG_OWN_PROVIDER is passed for
            // documentation of intent even though umfPoolDestroy is never
            // actually called in this design (see "no teardown" above) --
            // in case a future version does add explicit teardown.
            let mut pool: ffi::umf_memory_pool_handle_t = ptr::null_mut();
            let create_pool_rc = ffi::umfPoolCreate(
                ffi::umfScalablePoolOps(),
                provider,
                pool_params as *const _,
                ffi::umf_pool_create_flag_t_UMF_POOL_CREATE_FLAG_OWN_PROVIDER,
                &mut pool,
            );
            // Transient params object -- always destroyed once the pool has
            // copied whatever it needs at creation time, regardless of
            // outcome.
            ffi::umfScalablePoolParamsDestroy(pool_params);
            if let Err(e) = check(create_pool_rc, TierAllocError::PoolCreate) {
                ffi::umfMemoryProviderDestroy(provider);
                return Err(e);
            }

            Ok(TierAllocator { pool, node })
        }
    }

    /// The NUMA node this allocator is bound to.
    #[must_use]
    pub fn node(&self) -> i32 {
        self.node
    }

    /// The raw UMF pool handle, for use by `NumaAllocator`'s `GlobalAlloc`
    /// impl, which frees raw pointers handed to it by `Box`/`Vec`'s own drop
    /// glue rather than a `TierBuffer` -- so it needs direct pool access,
    /// not the `TierBuffer`-wrapping `alloc`/`alloc_aligned` methods.
    pub(crate) fn pool_handle(&self) -> ffi::umf_memory_pool_handle_t {
        self.pool
    }

    /// Constructs a `TierAllocator` bound to NUMA node `node`, backed by
    /// UMF's jemalloc pool instead of the default scalable (TBB) pool.
    ///
    /// # ⚠️ Stability warning
    ///
    /// UMF's jemalloc pool (`umfJemallocPoolOps`) has crashed four separate
    /// times under real concurrent multi-threaded load on this exact UMF
    /// version (1.0.3), in the sibling `paper-cache-cxl` crate's own
    /// testing: twice a SIGSEGV inside UMF's own critnib memory-tracker
    /// during jemalloc's internal extent-splitting, once a corrupted/torn
    /// allocation-failure message under concurrent heap pressure, and once
    /// (with this same constructor wired into `registry.rs` uniformly for
    /// both tiers, mirroring TBB's default usage) a SIGSEGV inside
    /// jemalloc's own extent-coalescing code (`ph_remove` --
    /// `include/jemalloc/internal/ph.h`, a null-pointer deref in its
    /// pairing-heap free-extent tracking). All four were root-caused to bugs
    /// inside UMF's own prebuilt library and/or jemalloc's internals as
    /// wired by UMF, not caller code, and are not fixable from this wrapper.
    /// **Do not use this constructor in production expecting it to be
    /// safe** -- it
    /// exists only for experimentation/future re-testing against a UMF
    /// version that has actually fixed the underlying bug.
    #[cfg(feature = "jemalloc_pool")]
    pub fn new_numa_jemalloc(node: i32) -> Result<Self, TierAllocError> {
        unsafe {
            let mut provider_params: ffi::umf_os_memory_provider_params_handle_t = ptr::null_mut();
            check(
                ffi::umfOsMemoryProviderParamsCreate(&mut provider_params),
                TierAllocError::ProviderParamsCreate,
            )?;

            let mut numa_list: [u32; 1] = [node as u32];
            if let Err(e) = check(
                ffi::umfOsMemoryProviderParamsSetNumaList(provider_params, numa_list.as_mut_ptr(), 1),
                TierAllocError::ProviderParamsSetNumaList,
            ) {
                ffi::umfOsMemoryProviderParamsDestroy(provider_params);
                return Err(e);
            }
            if let Err(e) = check(
                ffi::umfOsMemoryProviderParamsSetNumaMode(provider_params, ffi::umf_numa_mode_t_UMF_NUMA_MODE_BIND),
                TierAllocError::ProviderParamsSetNumaMode,
            ) {
                ffi::umfOsMemoryProviderParamsDestroy(provider_params);
                return Err(e);
            }

            let mut provider: ffi::umf_memory_provider_handle_t = ptr::null_mut();
            let create_provider_rc = ffi::umfMemoryProviderCreate(
                ffi::umfOsMemoryProviderOps(),
                provider_params as *const _,
                &mut provider,
            );
            ffi::umfOsMemoryProviderParamsDestroy(provider_params);
            check(create_provider_rc, TierAllocError::ProviderCreate)?;

            // Jemalloc pool params: left entirely at UMF's defaults (e.g.
            // `umfJemallocPoolParamsSetNumArenas` is never called) -- the
            // sibling paper-cache-cxl crate's own testing found overriding
            // the arena count made memory usage worse, not better.
            let mut pool_params: ffi::umf_jemalloc_pool_params_handle_t = ptr::null_mut();
            if let Err(e) = check(
                ffi::umfJemallocPoolParamsCreate(&mut pool_params),
                TierAllocError::JemallocPoolParamsCreate,
            ) {
                ffi::umfMemoryProviderDestroy(provider);
                return Err(e);
            }

            let mut pool: ffi::umf_memory_pool_handle_t = ptr::null_mut();
            let create_pool_rc = ffi::umfPoolCreate(
                ffi::umfJemallocPoolOps(),
                provider,
                pool_params as *const _,
                ffi::umf_pool_create_flag_t_UMF_POOL_CREATE_FLAG_OWN_PROVIDER,
                &mut pool,
            );
            ffi::umfJemallocPoolParamsDestroy(pool_params);
            if let Err(e) = check(create_pool_rc, TierAllocError::PoolCreate) {
                ffi::umfMemoryProviderDestroy(provider);
                return Err(e);
            }

            Ok(TierAllocator { pool, node })
        }
    }

    /// Allocates a byte-aligned [`TierBuffer`] of `len` bytes from this
    /// tier.
    pub fn alloc(&self, len: usize) -> Result<TierBuffer, TierAllocError> {
        self.alloc_aligned(len, 1)
    }

    /// Allocates a [`TierBuffer`] of `len` bytes, aligned to `align` bytes,
    /// from this tier.
    pub fn alloc_aligned(&self, len: usize, align: usize) -> Result<TierBuffer, TierAllocError> {
        if len == 0 {
            // UMF's headers don't specify 0-byte umfPoolAlignedMalloc
            // behavior -- simplest to just not call it.
            return Ok(TierBuffer::empty());
        }

        let raw = unsafe { ffi::umfPoolAlignedMalloc(self.pool, len, align) };

        let Some(ptr) = std::ptr::NonNull::new(raw as *mut u8) else {
            return Err(TierAllocError::AllocFailed { requested_bytes: len });
        };

        Ok(TierBuffer::from_raw(ptr, len, self.pool))
    }
}
