/*
 * umf_stub.c
 *
 * Stub implementation of the UMF allocator API.
 *
 * This file is compiled when the real UMF library (wrapper.h) is NOT available
 * — for example in CI or on developer machines without PMEM hardware.
 *
 * All allocations are backed by standard malloc/free so that the Rust code
 * compiles and links without UMF and all integration tests pass.  The stub
 * deliberately makes check_tier() return 1 ("PMEM") so that the allocator
 * routing logic in HybridObjects::dealloc() correctly calls umf_dealloc()
 * rather than jemalloc's dealloc — keeping the control flow symmetric.
 *
 * On a real PMEM machine the actual umf_allocator_wrapper.c is compiled
 * instead and UMF routes allocations to CXL/persistent memory via the OS
 * NUMA provider.
 */

#include <stddef.h>
#include <stdlib.h>

/* No-op: stub allocator needs no initialisation.
 * The `numa_node` parameter mirrors the real UMF API signature so that the
 * same Rust `extern "C"` declaration works for both the real UMF wrapper and
 * this stub — the stub simply ignores it. */
int umf_allocator_init(int numa_node) {
    (void)numa_node;
    return 0;
}

/* Allocate using standard aligned_alloc / malloc.
 * aligned_alloc requires size to be a multiple of alignment; we fall back to
 * malloc for small alignments that are already guaranteed by malloc. */
void *umf_alloc(size_t size, size_t align) {
    if (size == 0) return NULL;
    if (align > sizeof(void *)) {
        /* Round size up to alignment multiple as required by aligned_alloc. */
        size_t aligned_size = (size + align - 1) & ~(align - 1);
        return aligned_alloc(align, aligned_size);
    }
    return malloc(size);
}

/* Free memory allocated by umf_alloc. */
void umf_dealloc(void *ptr) {
    free(ptr);
}

/* No-op teardown. */
void umf_allocator_finalize(void) {}

/* Return a newly malloc'd block — only used for diagnostics in real UMF. */
void *return_pmem_base(size_t dax_size) {
    (void)dax_size;
    return NULL;
}

/* Return the size of PMEM pool — 0 in stub. */
size_t return_pmem_size(void) {
    return 0;
}

/* Every pointer allocated by the stub is treated as "tier 1" (PMEM) so that
 * HybridObjects::dealloc() calls umf_dealloc() for everything it allocated
 * through HybridObjects, keeping alloc/dealloc symmetric. */
int check_tier(void *ptr) {
    (void)ptr;
    return 1; /* 1 = PMEM tier */
}
