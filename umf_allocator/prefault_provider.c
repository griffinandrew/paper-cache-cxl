/* ===========================================================================
 * Prefaulted memory provider for UMF.
 *
 * Replaces umfOsMemoryProvider as the backing store under the scalable pool.
 * Maps ONE big region up front, binds it to the target NUMA node, and
 * (optionally) faults EVERY page before the workload starts. Then every
 * umfMemoryProviderAlloc just bumps a pointer into the region.
 *
 * Two corrections vs. the previous version:
 *
 *   1. ORDERING. The old code used mmap(MAP_POPULATE), which faults every
 *      page at mmap time under the *default* (local) NUMA policy — i.e. on
 *      the init thread's node, typically DRAM node 0 — and only afterwards
 *      called mbind. MPOL_MF_MOVE made that "work" by migrating, but a
 *      partial/failed migration under MPOL_MF_STRICT was swallowed by a
 *      perror-and-continue, so the region could silently end up on DRAM.
 *      Now: mmap WITHOUT populate, mbind the empty VMA, THEN fault under the
 *      bound policy. No MOVE needed (nothing is faulted yet). mbind failure
 *      is fatal instead of ignored.
 *
 *   2. TOGGLE. Prefault is now conditional (params->do_prefault). With it
 *      off, the region is mapped + bound but NOT faulted, so first-touch
 *      faults land during the workload — exactly the variable you want to
 *      flip to attribute the memcpy-phase cost. Everything else (provider,
 *      bump, bind) is byte-identical between the two runs.
 *
 * Build: needs <numaif.h> for mbind (in libc). MADV_POPULATE_WRITE needs
 * Linux 5.14+ and _GNU_SOURCE. Compile alongside your existing
 * umf_allocator.c.
 * ===========================================================================*/

#define _GNU_SOURCE
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <sys/mman.h>
#include <unistd.h>
#include <numaif.h>   /* mbind, MPOL_BIND */

#include <umf/memory_provider.h>
#include <umf/base.h>

#ifndef MADV_POPULATE_WRITE
#define MADV_POPULATE_WRITE 23   /* Linux 5.14+; define for older headers */
#endif

/* ---- provider instance state ---- */
typedef struct prefault_provider_t {
    void  *base;        /* start of the region                       */
    size_t size;        /* total bytes                               */
    atomic_size_t off;  /* bump offset                               */
    int    numa_node;   /* node the region is bound to               */
    int    prefaulted;  /* whether pages were faulted at init        */
} prefault_provider_t;

/* params passed at provider creation */
typedef struct prefault_params_t {
    size_t size;
    int    numa_node;
    int    do_prefault;   /* 1 = fault whole region at init, 0 = lazy */
} prefault_params_t;

/* ---- UMF provider ops ---- */

static umf_result_t prefault_initialize(void *params, void **provider) {
    prefault_params_t *pp = (prefault_params_t *)params;
    if (!pp || pp->size == 0) return UMF_RESULT_ERROR_INVALID_ARGUMENT;
    if (pp->numa_node < 0 || pp->numa_node >= 64) {
        /* 1UL << node is UB / inexpressible in a single-long mask for >= 64 */
        fprintf(stderr, "prefault_provider: numa_node %d unsupported "
                        "(single-long nodemask covers 0..63)\n", pp->numa_node);
        return UMF_RESULT_ERROR_INVALID_ARGUMENT;
    }

    prefault_provider_t *st = calloc(1, sizeof(*st));
    if (!st) return UMF_RESULT_ERROR_OUT_OF_HOST_MEMORY;

    /* round up to 2 MiB */
    size_t two_mb = 2 * 1024 * 1024ULL;
    size_t size = (pp->size + two_mb - 1) & ~(two_mb - 1);

    /* NO MAP_POPULATE: we must set NUMA policy before any page faults in. */
    void *base = mmap(NULL, size, PROT_READ | PROT_WRITE,
                      MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (base == MAP_FAILED) {
        perror("prefault_provider: mmap");
        free(st);
        return UMF_RESULT_ERROR_OUT_OF_HOST_MEMORY;
    }

    /* Bind the still-unfaulted region strictly to the target node. Nothing is
     * faulted yet, so no MPOL_MF_MOVE is required — pages will land on the
     * bound node at first touch. A failed bind is FATAL: continuing would mean
     * silently measuring DRAM-backed pages labelled as the target node. */
    unsigned long nodemask = 1UL << pp->numa_node;
    if (mbind(base, size, MPOL_BIND, &nodemask, 64, MPOL_MF_STRICT) != 0) {
        perror("prefault_provider: mbind");
        munmap(base, size);
        free(st);
        return UMF_RESULT_ERROR_OUT_OF_HOST_MEMORY;
    }

    /* Conditionally fault every page now, under the bound policy. With
     * do_prefault == 0 the pages stay unmapped and first-touch faults occur
     * during the workload — the variable under test. */
    if (pp->do_prefault) {
        if (madvise(base, size, MADV_POPULATE_WRITE) != 0) {
            /* Fall back to an explicit per-page write touch if the kernel
             * lacks MADV_POPULATE_WRITE. Still faults under the bound policy. */
            if (errno == EINVAL) {
                long pg = sysconf(_SC_PAGESIZE);
                if (pg <= 0) pg = 4096;
                volatile char *p = (volatile char *)base;
                for (size_t o = 0; o < size; o += (size_t)pg) p[o] = 0;
            } else {
                perror("prefault_provider: madvise(MADV_POPULATE_WRITE)");
                munmap(base, size);
                free(st);
                return UMF_RESULT_ERROR_OUT_OF_HOST_MEMORY;
            }
        }
    }

    st->base       = base;
    st->size       = size;
    atomic_store(&st->off, 0);
    st->numa_node  = pp->numa_node;
    st->prefaulted = pp->do_prefault;

    fprintf(stderr,
            "prefault_provider: %zu MiB mmapped + bound on node %d, "
            "prefault=%s\n",
            size >> 20, pp->numa_node, pp->do_prefault ? "on" : "off");

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
        size_t cur     = atomic_load(&st->off);
        size_t aligned = (cur + alignment - 1) & ~(alignment - 1);
        if (aligned < cur) return UMF_RESULT_ERROR_OUT_OF_HOST_MEMORY; /* align overflow */
        size_t end = aligned + size;
        if (end < aligned) return UMF_RESULT_ERROR_OUT_OF_HOST_MEMORY; /* size overflow */
        if (end > st->size) {
            /* region exhausted: pool sees provider OOM. Size PREFAULT_BYTES
             * bigger if this fires. */
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
 * takes memory back. free is a no-op (bump-forward provider). Pages stay
 * mapped (and faulted, if prefaulted) for the life of the provider. */
static umf_result_t prefault_free(void *provider, void *ptr, size_t size) {
    (void)provider; (void)ptr; (void)size;
    return UMF_RESULT_SUCCESS;
}

static const char *prefault_get_name(void *provider) {
    (void)provider;
    return "prefault_provider";
}

/* Required-but-trivial ops. */
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
    .version                   = UMF_VERSION_CURRENT,
    .initialize                = prefault_initialize,
    .finalize                  = prefault_finalize,
    .alloc                     = prefault_alloc,
    .free                      = prefault_free,
    .get_last_native_error     = prefault_get_last_native_error,
    .get_recommended_page_size = prefault_get_recommended_page_size,
    .get_min_page_size         = prefault_get_min_page_size,
    .get_name                  = prefault_get_name,
    /* .ext / .ipc left zeroed (not used) */
};

const umf_memory_provider_ops_t *umfPrefaultProviderOps(void) {
    return &PREFAULT_OPS;
}