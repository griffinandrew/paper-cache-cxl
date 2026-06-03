/* ===========================================================================
 * Prefaulted memory provider for UMF.
 *
 * Replaces umfOsMemoryProvider as the backing store under the scalable pool.
 * Instead of mmap-on-demand (which faults pages lazily during the workload —
 * the ~30k cyc/page first-touch PMEM cost), this provider mmaps ONE big region
 * up front, binds it to the target NUMA node, and faults EVERY page before the
 * workload starts. Then every umfMemoryProviderAlloc just bumps a pointer into
 * the already-faulted region.
 *
 * The scalable pool (TBB) sits on top exactly as before and handles
 * alloc/free/reuse — so transient allocations recycle normally and never
 * exhaust the region, while the prefault property holds for ALL allocations
 * (values and transient) because the underlying pages are all faulted.
 *
 * This keeps DRAMObjects / HybridObjects and the per-node pool scheme
 * unchanged; only the provider feeding the pool changes.
 *
 * Build: needs <numaif.h> for mbind (link -lnuma not required; mbind is in
 * libc). Compile alongside your existing umf_allocator.c.
 * ===========================================================================*/

#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <unistd.h>
#include <numaif.h>   /* mbind, MPOL_BIND */

#include <umf/memory_provider.h>
#include <umf/base.h>

/* ---- provider instance state ---- */
typedef struct prefault_provider_t {
    void  *base;        /* start of the prefaulted region            */
    size_t size;        /* total bytes                               */
    atomic_size_t off;  /* bump offset                               */
    int    numa_node;   /* node the region is bound to               */
} prefault_provider_t;

/* params passed at provider creation */
typedef struct prefault_params_t {
    size_t size;
    int    numa_node;
} prefault_params_t;

/* ---- helpers ---- */
static int prefault_region(void *base, size_t size, int numa_node) {
    /* bind strictly to the target node */
    unsigned long nodemask = 1UL << numa_node;
    if (mbind(base, size, MPOL_BIND, &nodemask, 64,
              MPOL_MF_MOVE | MPOL_MF_STRICT) != 0) {
        perror("prefault_region: mbind");
        /* continue: on DRAM this may still be fine, on PMEM it matters */
    }

    long pg = sysconf(_SC_PAGESIZE);
    if (pg <= 0) pg = 4096;

    /* write one byte per page to force the write-fault now */
    volatile char *p = (volatile char *)base;
    for (size_t o = 0; o < size; o += (size_t)pg) {
        p[o] = 0;
    }
    return 0;
}

/* ---- UMF provider ops ---- */

static umf_result_t prefault_initialize(void *params, void **provider) {
    prefault_params_t *pp = (prefault_params_t *)params;
    if (!pp || pp->size == 0) return UMF_RESULT_ERROR_INVALID_ARGUMENT;

    prefault_provider_t *st = calloc(1, sizeof(*st));
    if (!st) return UMF_RESULT_ERROR_OUT_OF_HOST_MEMORY;

    /* round to 2 MiB */
    size_t two_mb = 2 * 1024 * 1024ULL;
    size_t size = (pp->size + two_mb - 1) & ~(two_mb - 1);

    void *base = mmap(NULL, size, PROT_READ | PROT_WRITE,
                      MAP_PRIVATE | MAP_ANONYMOUS | MAP_POPULATE, -1, 0);
    if (base == MAP_FAILED) {
        free(st);
        return UMF_RESULT_ERROR_OUT_OF_HOST_MEMORY;
    }

    prefault_region(base, size, pp->numa_node);

    st->base = base;
    st->size = size;
    atomic_store(&st->off, 0);
    st->numa_node = pp->numa_node;

    fprintf(stderr,
            "prefault_provider: %zu MiB mmapped + faulted on node %d\n",
            size >> 20, pp->numa_node);

    *provider = st;
    return UMF_RESULT_SUCCESS;
}

static void prefault_finalize(void *provider) {
    prefault_provider_t *st = (prefault_provider_t *)provider;
    if (!st) return;
    munmap(st->base, st->size);
    free(st);
}

static umf_result_t prefault_alloc(void *provider, size_t size,
                                   size_t alignment, void **ptr) {
    prefault_provider_t *st = (prefault_provider_t *)provider;
    if (alignment == 0) alignment = 64;

    /* lock-free bump with alignment */
    for (;;) {
        size_t cur = atomic_load(&st->off);
        size_t aligned = (cur + alignment - 1) & ~(alignment - 1);
        size_t end = aligned + size;
        if (end > st->size) {
            /* region exhausted: pool will see this as provider OOM. Size the
             * region bigger if this fires. */
            return UMF_RESULT_ERROR_OUT_OF_HOST_MEMORY;
        }
        if (atomic_compare_exchange_weak(&st->off, &cur, end)) {
            *ptr = (char *)st->base + aligned;
            return UMF_RESULT_SUCCESS;
        }
        /* contended, retry */
    }
}

/* The scalable pool retains/reuses freed blocks itself; the provider never
 * needs to take memory back. free is a no-op (bump-forward provider). Pages
 * stay mapped + faulted for the life of the provider. */
static umf_result_t prefault_free(void *provider, void *ptr, size_t size) {
    (void)provider; (void)ptr; (void)size;
    return UMF_RESULT_SUCCESS;
}

static const char *prefault_get_name(void *provider) {
    (void)provider;
    return "prefault_provider";
}

/* Required-but-trivial ops. Fill the rest of the ops struct with stubs that
 * report not-supported; the scalable pool does not need them for basic use. */
static umf_result_t prefault_get_last_native_error(void *provider,
                                                   const char **msg,
                                                   int32_t *code) {
    (void)provider;
    if (msg)  *msg  = "prefault_provider: no native error detail";
    if (code) *code = 0;
    return UMF_RESULT_SUCCESS;
}

static umf_result_t prefault_get_recommended_page_size(void *provider,
                                                       size_t size,
                                                       size_t *page_size) {
    (void)provider; (void)size;
    long pg = sysconf(_SC_PAGESIZE);
    *page_size = (pg > 0) ? (size_t)pg : 4096;
    return UMF_RESULT_SUCCESS;
}

static umf_result_t prefault_get_min_page_size(void *provider, void *ptr,
                                               size_t *page_size) {
    (void)provider; (void)ptr;
    long pg = sysconf(_SC_PAGESIZE);
    *page_size = (pg > 0) ? (size_t)pg : 4096;
    return UMF_RESULT_SUCCESS;
}

/* Build the ops table. Field set depends on your UMF version's
 * umf_memory_provider_ops_t — adjust names if your headers differ. */
static umf_memory_provider_ops_t PREFAULT_OPS = {
    .version            = UMF_VERSION_CURRENT,
    .initialize         = prefault_initialize,
    .finalize           = prefault_finalize,
    .alloc              = prefault_alloc,
    .free               = prefault_free,
    .get_last_native_error      = prefault_get_last_native_error,
    .get_recommended_page_size  = prefault_get_recommended_page_size,
    .get_min_page_size          = prefault_get_min_page_size,
    .get_name           = prefault_get_name,
    /* .ext / .ipc left zeroed (not used) */
};

const umf_memory_provider_ops_t *umfPrefaultProviderOps(void) {
    return &PREFAULT_OPS;
}