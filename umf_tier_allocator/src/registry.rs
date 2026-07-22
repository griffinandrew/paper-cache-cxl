use std::sync::OnceLock;

use crate::error::TierAllocError;
use crate::tier_allocator::TierAllocator;
use crate::tier_buffer::TierBuffer;

#[cfg(all(feature = "jemalloc_pool", feature = "disjoint_pool"))]
compile_error!("jemalloc_pool and disjoint_pool are mutually exclusive pool backends -- pick one");

/// Disjoint-pool tuning, only relevant under `disjoint_pool`. `SLAB_MIN_SIZE`
/// is set close to this crate's real average object size (~16 KB, per the
/// sibling `paper-cache-cxl` crate's own documented trace analysis) rather
/// than UMF's own 64 KB default -- measured directly (a controlled
/// 300k-object, 90%-freed, randomly-scattered repro) to make the difference
/// between real settled memory landing at ~0.35x peak (64 KB default) vs.
/// ~0.13x peak (16 KB, within ~1.23x of the true theoretical minimum).
/// `MAX_POOLABLE_SIZE`/`CAPACITY` are left at UMF's own defaults (2 MB / 4).
#[cfg(feature = "disjoint_pool")]
const DISJOINT_SLAB_MIN_SIZE: usize = 16 * 1024;
#[cfg(feature = "disjoint_pool")]
const DISJOINT_MAX_POOLABLE_SIZE: usize = 2 * 1024 * 1024;
#[cfg(feature = "disjoint_pool")]
const DISJOINT_CAPACITY: usize = 4;

/// Upper bound on NUMA node ids this registry will track a pool for.
/// Matches the sibling `paper-cache-cxl` crate's own C wrapper
/// (`umf_allocator/umf_allocator_wrapper.c`) convention, even though only
/// nodes 0/1 are used anywhere in this workspace today.
pub(crate) const MAX_NODES: usize = 8;

/// The one shared, per-node pool table. Both access patterns --
/// [`NumaAllocator`](crate::NumaAllocator)'s implicit `GlobalAlloc` dispatch
/// and this module's explicit `alloc_on`/`allocator_for` functions -- read
/// from this same table, so there is exactly one real UMF pool per node for
/// the life of the process no matter how many call sites ask for it.
///
/// Caches `Result<TierAllocator, TierAllocError>`, not just `TierAllocator`,
/// because `OnceLock::get_or_init` requires an infallible closure while
/// `TierAllocator::new_numa` returns a `Result` -- a prior construction
/// failure is cached and replayed on subsequent calls rather than retried
/// (requires `TierAllocError: Copy`, added alongside this module).
///
/// `[const { OnceLock::new() }; MAX_NODES]`: inline-const array-repeat
/// syntax, stable since Rust 1.79. Needed because `OnceLock<T>` isn't
/// `Copy`, so a plain `[OnceLock::new(); MAX_NODES]` repeat expression
/// doesn't type-check.
static REGISTRY: [OnceLock<Result<TierAllocator, TierAllocError>>; MAX_NODES] =
    [const { OnceLock::new() }; MAX_NODES];

/// Returns the shared `TierAllocator` for `node`, lazily constructing it on
/// first call and caching the result (success or failure) for every
/// subsequent call against the same node.
pub(crate) fn pool_for_node(node: i32) -> Result<&'static TierAllocator, TierAllocError> {
    let Ok(idx) = usize::try_from(node) else {
        return Err(TierAllocError::InvalidNode { node });
    };

    let Some(slot) = REGISTRY.get(idx) else {
        return Err(TierAllocError::InvalidNode { node });
    };

    match slot.get_or_init(|| {
        #[cfg(feature = "disjoint_pool")]
        {
            TierAllocator::new_numa_disjoint(
                node,
                DISJOINT_SLAB_MIN_SIZE,
                DISJOINT_MAX_POOLABLE_SIZE,
                DISJOINT_CAPACITY,
            )
        }
        #[cfg(all(feature = "jemalloc_pool", not(feature = "disjoint_pool")))]
        {
            TierAllocator::new_numa_jemalloc(node)
        }
        #[cfg(not(any(feature = "jemalloc_pool", feature = "disjoint_pool")))]
        {
            TierAllocator::new_numa(node)
        }
    }) {
        Ok(allocator) => Ok(allocator),
        Err(e) => Err(*e),
    }
}

/// Returns the shared `TierAllocator` bound to `node`, for callers that want
/// the allocator handle itself (e.g. to call [`TierAllocator::alloc`]
/// directly, or [`crate::TierBuffer::duplicate`]).
pub fn allocator_for(node: i32) -> Result<&'static TierAllocator, TierAllocError> {
    pool_for_node(node)
}

/// Allocates a `len`-byte [`TierBuffer`] on `node`'s shared pool.
pub fn alloc_on(node: i32, len: usize) -> Result<TierBuffer, TierAllocError> {
    allocator_for(node)?.alloc(len)
}

/// Allocates a `len`-byte [`TierBuffer`], aligned to `align` bytes, on
/// `node`'s shared pool.
pub fn alloc_on_aligned(node: i32, len: usize, align: usize) -> Result<TierBuffer, TierAllocError> {
    allocator_for(node)?.alloc_aligned(len, align)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pool_for_node_returns_same_pool_across_calls() {
        let first = pool_for_node(0).expect("node 0 should be constructible");
        let second = pool_for_node(0).expect("node 0 should be constructible");
        assert!(std::ptr::eq(first, second), "expected the same cached TierAllocator instance");
    }

    #[test]
    fn pool_for_node_rejects_out_of_range_node() {
        assert!(matches!(pool_for_node(-1), Err(TierAllocError::InvalidNode { node: -1 })));
        assert!(matches!(
            pool_for_node(MAX_NODES as i32),
            Err(TierAllocError::InvalidNode { node }) if node == MAX_NODES as i32
        ));
    }

    #[test]
    fn alloc_on_and_allocator_for_agree_for_the_same_node() {
        let buffer = alloc_on(0, 16).expect("alloc_on(0, 16) should succeed");
        assert_eq!(buffer.len(), 16);

        let allocator = allocator_for(0).expect("allocator_for(0) should succeed");
        assert_eq!(allocator.node(), 0);
    }
}
