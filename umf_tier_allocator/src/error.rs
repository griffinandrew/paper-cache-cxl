use crate::ffi::umf_result_t;

/// Error produced while constructing a [`crate::TierAllocator`] or
/// allocating a [`crate::TierBuffer`] from one.
///
/// Wraps the raw `umf_result_t` returned by the failing UMF call so
/// callers/logs retain enough detail to debug against UMF's own error-code
/// reference, without this crate needing to re-derive a full mapping of
/// every UMF error variant into its own type up front.
#[derive(Debug, thiserror::Error)]
pub enum TierAllocError {
    #[error("umfOsMemoryProviderParamsCreate failed: {0:?}")]
    ProviderParamsCreate(umf_result_t),

    #[error("umfOsMemoryProviderParamsSetNumaList failed: {0:?}")]
    ProviderParamsSetNumaList(umf_result_t),

    #[error("umfOsMemoryProviderParamsSetNumaMode failed: {0:?}")]
    ProviderParamsSetNumaMode(umf_result_t),

    #[error("umfMemoryProviderCreate failed: {0:?}")]
    ProviderCreate(umf_result_t),

    #[error("umfScalablePoolParamsCreate failed: {0:?}")]
    PoolParamsCreate(umf_result_t),

    #[error("umfScalablePoolParamsSetKeepAllMemory failed: {0:?}")]
    PoolParamsSetKeepAllMemory(umf_result_t),

    #[error("umfScalablePoolParamsSetGranularity failed: {0:?}")]
    PoolParamsSetGranularity(umf_result_t),

    #[error("umfPoolCreate failed: {0:?}")]
    PoolCreate(umf_result_t),

    /// `umfPoolAlignedMalloc` returned null (e.g. out of memory on the
    /// bound NUMA node).
    #[error("allocation of {requested_bytes} bytes failed (pool returned null)")]
    AllocFailed { requested_bytes: usize },

    /// Jemalloc-pool-specific construction errors (only reachable via
    /// `TierAllocator::new_numa_jemalloc`, behind the `jemalloc_pool`
    /// feature).
    #[cfg(feature = "jemalloc_pool")]
    #[error("umfJemallocPoolParamsCreate failed: {0:?}")]
    JemallocPoolParamsCreate(umf_result_t),
}
