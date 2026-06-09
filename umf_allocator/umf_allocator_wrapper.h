#ifndef UMF_ALLOCATOR_H
#define UMF_ALLOCATOR_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Maximum NUMA node id supported. Bump if your machine has more nodes. */
#define UMF_ALLOCATOR_MAX_NODES 8

/*
 * Initialize a UMF scalable pool bound to the given NUMA node.
 *
 * Each NUMA node gets its own independent pool. Calling this multiple times
 * with the same numa_node is idempotent (subsequent calls return 0 without
 * reinitializing). Different numa_node values create independent pools that
 * can coexist.
 *
 * Returns:
 *    0 on success (including the already-initialized case)
 *   -1 if numa_node is out of range [0, UMF_ALLOCATOR_MAX_NODES)
 *  1..6 on various UMF init failures (see source for specifics)
 */
int umf_allocator_init(int numa_node);

/*
 * Allocate `size` bytes from the pool on `numa_node`. If `align` is greater
 * than sizeof(void*), the allocation is aligned to `align` bytes; otherwise
 * a normal malloc-style allocation is performed.
 *
 * Returns NULL if size == 0, numa_node is out of range, the pool for that
 * node has not been initialized, or the underlying UMF allocation failed.
 *
 * Thread-safe. The scalable pool handles its own per-thread caching; no
 * global locking is performed on the alloc fast path.
 */
void *umf_alloc(int numa_node, size_t size, size_t align);

/*
 * Free a pointer previously returned by umf_alloc on the same numa_node.
 * Passing NULL is a no-op. Passing an out-of-range numa_node is a no-op.
 * Passing a pointer to a different node's pool is undefined behavior — use
 * check_tier() first if you need to find the owning node.
 */
void umf_dealloc(int numa_node, void *ptr);

/*
 * Return the NUMA node id that owns `ptr`, or -1 if `ptr` is not managed by
 * any of our UMF pools.
 *
 * NOTE: return semantics differ from the previous single-pool version, which
 * returned 1 = pmem / 0 = not-ours. Callers that treated the return as a
 * bool must be updated.
 */
int check_tier(void *ptr);

/*
 * Prewarm the pool for `numa_node` by allocating `bytes` worth of memory in
 * `chunk`-sized pieces, touching every page to force the page faults to
 * happen now (and to bind the pages to the target node), then freeing the
 * memory back into the pool for reuse.
 *
 * Call AFTER umf_allocator_init(numa_node), BEFORE the measured workload.
 *
 * If `chunk` is 0, defaults to 4096 bytes. Typical use: 2 MiB chunks to
 * match the pool granularity.
 *
 * Returns:
 *   0 on success
 *   1 if numa_node is out of range or the pool is not initialized
 *   2 if the bookkeeping array could not be allocated on the host
 *   3 if the pool ran out of memory partway through (partial work undone)
 */
int umf_allocator_prewarm(int numa_node, size_t bytes, size_t chunk);



//dax calls 

void umf_allocator_finalize_dax(void);


int umf_allocator_init_dax(const char *dax_path, size_t dax_size);

void *umf_alloc_dax(size_t size, size_t align);

void umf_dealloc_dax(void *ptr);

int check_tier_dax(void *ptr);




#ifdef __cplusplus
}
#endif

#endif /* UMF_ALLOCATOR_H */