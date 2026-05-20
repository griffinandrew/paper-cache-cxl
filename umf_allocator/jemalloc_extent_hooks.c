#define _GNU_SOURCE
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdbool.h>
#include <string.h>
#include <errno.h>
#include <limits.h>
#include <stdatomic.h>
#include <pthread.h>
#include <unistd.h>
#include <sys/mman.h>
#include <numaif.h>
#include <numa.h>
#include <jemalloc/jemalloc.h>

#ifdef JEMALLOC_USE_PREFIX
#define mallctl  je_mallctl
#define mallocx  je_mallocx
#define dallocx  je_dallocx
#endif

/* ------------------------------------------------------------------ */
/* Design                                                              */
/* ------------------------------------------------------------------ */
/*
 * The cache uses jemalloc for object management (size classes, tcache,
 * reuse). These extent hooks only control WHERE jemalloc gets its raw
 * memory from.
 *
 * Key change vs. the mmap-per-extent version: at init we reserve ONE big
 * contiguous region (mmap, MAP_NORESERVE — reservation only, NOT faulted)
 * and mbind it to the target node once. extent_alloc then just carves
 * bump-pointer slices out of that region. No per-extent mmap, no
 * per-extent mbind, no per-extent syscalls.
 *
 * Faults still happen LAZILY on first touch, on the hot path. This is NOT
 * prefaulting. The only thing that changes is that every fault now lands
 * inside one large contiguous VMA (like stock jemalloc's arena) instead of
 * thousands of scattered small mmap'd VMAs — which is the fault-pattern
 * difference that made the old hooks slow without prefault.
 *
 * extent_dalloc is a no-op (bump region, bulk reclaim at teardown). The
 * cache workload has no eviction, so nothing is freed mid-run anyway.
 */

#ifndef MADV_NOHUGEPAGE
#define MADV_NOHUGEPAGE 15
#endif

/* Default reserved region size. Override with PAPER_CACHE_REGION_BYTES. */
#define DEFAULT_REGION_BYTES (40ULL * 1024 * 1024 * 1024)

/* ------------------------------------------------------------------ */
/* State                                                               */
/* ------------------------------------------------------------------ */

static _Atomic unsigned arena_ind   = UINT_MAX;
static _Atomic unsigned tcache_ind  = UINT_MAX; 
static int              target_node = 0;

static void  *region_base = NULL;
static size_t region_cap  = 0;
static _Atomic size_t region_off = 0;   /* bump cursor */

static pthread_mutex_t lifecycle_lock = PTHREAD_MUTEX_INITIALIZER;

/* ------------------------------------------------------------------ */
/* Region setup                                                        */
/* ------------------------------------------------------------------ */

/* mbind the whole region to target_node. Policy only — sets where pages
 * WILL come from when faulted. Does not fault anything. */
static void bind_region(void *addr, size_t size) {
    if (target_node < 0) return;

    struct bitmask *mask = numa_bitmask_alloc(numa_num_possible_nodes());
    if (!mask) return;
    numa_bitmask_setbit(mask, target_node);
    long rc = mbind(addr, size, MPOL_BIND, mask->maskp, mask->size + 1, 0);
    if (rc != 0) {
        fprintf(stderr,
                "bind_region: mbind(node=%d, %zu bytes) FAILED: %s\n",
                target_node, size, strerror(errno));
    }
    numa_bitmask_free(mask);
}

/* Reserve the contiguous region. MAP_NORESERVE: address space only, no
 * commit accounting, NO faulting. */
static int region_reserve(void) {
    size_t bytes = DEFAULT_REGION_BYTES;
    const char *env = getenv("PAPER_CACHE_REGION_BYTES");
    if (env) {
        unsigned long long v = strtoull(env, NULL, 10);
        if (v > 0) bytes = (size_t)v;
    }

    void *p = mmap(NULL, bytes, PROT_READ | PROT_WRITE,
                   MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
    if (p == MAP_FAILED) {
        fprintf(stderr, "region_reserve: mmap(%zu) failed: %s\n",
                bytes, strerror(errno));
        return 1;
    }

    /* Bind the whole region to the target node ONCE. */
    bind_region(p, bytes);

    region_base = p;
    region_cap  = bytes;
    atomic_store_explicit(&region_off, 0, memory_order_release);

    fprintf(stderr, "region_reserve: %zu bytes reserved at %p, node %d\n",
            bytes, p, target_node);
    return 0;
}

/* ------------------------------------------------------------------ */
/* Extent hooks                                                         */
/* ------------------------------------------------------------------ */

/* Carve an aligned slice out of the reserved region. No mmap, no syscall,
 * no fault — the fault happens later, lazily, when jemalloc/the cache
 * first writes to the returned memory. */
static void *extent_alloc(extent_hooks_t *hooks, void *new_addr, size_t size,
                          size_t alignment, bool *zero, bool *commit,
                          unsigned arena) {
    (void)hooks; (void)new_addr; (void)arena;

    if (!region_base || region_cap == 0) return NULL;
    size_t align = alignment ? alignment : 1;

    for (;;) {
        size_t cur     = atomic_load_explicit(&region_off,
                                              memory_order_relaxed);
        size_t aligned = (cur + align - 1) & ~(align - 1);
        if (aligned + size < aligned) return NULL;     /* overflow */
        size_t end = aligned + size;
        if (end > region_cap) {
            fprintf(stderr,
                    "extent_alloc: region exhausted (need %zu, cap %zu)\n",
                    end, region_cap);
            return NULL;
        }
        if (atomic_compare_exchange_weak_explicit(
                &region_off, &cur, end,
                memory_order_seq_cst, memory_order_relaxed)) {
            void *p = (char *)region_base + aligned;
            /* Pages from a fresh MAP_ANONYMOUS region are zero on first
             * fault, so we can promise jemalloc zeroed + committed. */
            *zero   = true;
            *commit = true;
            return p;
        }
        /* CAS lost a race (rare; single-threaded benchmark) — retry. */
    }
}

/* No-op: bump region, bulk reclaim at teardown. The workload has no
 * eviction so this is essentially never called mid-run anyway. Returning
 * true tells jemalloc the extent was NOT released, so it retains and
 * reuses it. */
static bool extent_dalloc(extent_hooks_t *hooks, void *addr, size_t size,
                          bool committed, unsigned arena) {
    (void)hooks; (void)addr; (void)size; (void)committed; (void)arena;
    return true;
}

static void extent_destroy(extent_hooks_t *hooks, void *addr, size_t size,
                           bool committed, unsigned arena) {
    (void)hooks; (void)addr; (void)size; (void)committed; (void)arena;
    /* Region is freed as a whole in umf_allocator_finalize. */
}

static bool extent_commit(extent_hooks_t *h, void *a, size_t s,
                          size_t o, size_t l, unsigned ar) {
    (void)h;(void)a;(void)s;(void)o;(void)l;(void)ar;
    return false;   /* already committed */
}

static bool extent_decommit(extent_hooks_t *h, void *a, size_t s,
                            size_t o, size_t l, unsigned ar) {
    (void)h;(void)a;(void)s;(void)o;(void)l;(void)ar;
    return true;    /* refuse — keep pages */
}

/* Refuse purges: keep pages resident, never MADV_DONTNEED them away. */
static bool extent_purge_lazy(extent_hooks_t *h, void *a, size_t s,
                              size_t o, size_t l, unsigned ar) {
    (void)h;(void)a;(void)s;(void)o;(void)l;(void)ar;
    return true;
}

static bool extent_purge_forced(extent_hooks_t *h, void *a, size_t s,
                                size_t o, size_t l, unsigned ar) {
    (void)h;(void)a;(void)s;(void)o;(void)l;(void)ar;
    return true;
}

/* Region is one contiguous mapping — splits and merges are free. */
static bool extent_split(extent_hooks_t *h, void *a, size_t s,
                         size_t sa, size_t sb, bool c, unsigned ar) {
    (void)h;(void)a;(void)s;(void)sa;(void)sb;(void)c;(void)ar;
    return false;   /* false == success */
}

static bool extent_merge(extent_hooks_t *h, void *a, size_t sa,
                         void *b, size_t sb, bool c, unsigned ar) {
    (void)h;(void)a;(void)sa;(void)b;(void)sb;(void)c;(void)ar;
    return false;   /* false == success */
}

static extent_hooks_t numa_hooks = {
    .alloc        = extent_alloc,
    .dalloc       = extent_dalloc,
    .destroy      = extent_destroy,
    .commit       = extent_commit,
    .decommit     = extent_decommit,
    .purge_lazy   = extent_purge_lazy,
    .purge_forced = extent_purge_forced,
    .split        = extent_split,
    .merge        = extent_merge,
};

/* ------------------------------------------------------------------ */
/* Decay control                                                        */
/* ------------------------------------------------------------------ */

static void disable_arena_decay(unsigned ind) {
    char path[64];
    ssize_t neg1 = -1;
    snprintf(path, sizeof(path), "arena.%u.dirty_decay_ms", ind);
    if (mallctl(path, NULL, NULL, &neg1, sizeof(neg1)) != 0)
        fprintf(stderr, "warning: dirty_decay disable failed (arena %u)\n", ind);
    snprintf(path, sizeof(path), "arena.%u.muzzy_decay_ms", ind);
    if (mallctl(path, NULL, NULL, &neg1, sizeof(neg1)) != 0)
        fprintf(stderr, "warning: muzzy_decay disable failed (arena %u)\n", ind);
}

/* ------------------------------------------------------------------ */
/* Public API                                                           */
/* ------------------------------------------------------------------ */

int umf_allocator_init(int numa_node) {
    pthread_mutex_lock(&lifecycle_lock);

    if (atomic_load_explicit(&arena_ind, memory_order_acquire) != UINT_MAX) {
        pthread_mutex_unlock(&lifecycle_lock);
        return 0;
    }

    if (numa_available() < 0) {
        fprintf(stderr, "libnuma not available\n");
        pthread_mutex_unlock(&lifecycle_lock);
        return 1;
    }
    target_node = numa_node;

    /* Reserve the one big contiguous region up front (no faulting). */
    if (region_reserve() != 0) {
        pthread_mutex_unlock(&lifecycle_lock);
        return 4;
    }

    unsigned ind;
    size_t ind_sz = sizeof(ind);
    if (mallctl("arenas.create", &ind, &ind_sz, NULL, 0) != 0) {
        fprintf(stderr, "arenas.create failed\n");
        pthread_mutex_unlock(&lifecycle_lock);
        return 2;
    }

    char path[64];
    snprintf(path, sizeof(path), "arena.%u.extent_hooks", ind);
    extent_hooks_t *hooks_ptr = &numa_hooks;
    if (mallctl(path, NULL, NULL, &hooks_ptr, sizeof(hooks_ptr)) != 0) {
        fprintf(stderr, "setting extent_hooks on arena %u failed\n", ind);
        snprintf(path, sizeof(path), "arena.%u.destroy", ind);
        mallctl(path, NULL, NULL, NULL, 0);
        pthread_mutex_unlock(&lifecycle_lock);
        return 3;
    }

    disable_arena_decay(ind);

    {
        unsigned tc;
        size_t tc_sz = sizeof(tc);
        if (mallctl("tcache.create", &tc, &tc_sz, NULL, 0) != 0) {
            fprintf(stderr, "tcache.create failed — falling back to no tcache flag\n");
            /* leave tcache_ind = UINT_MAX; umf_alloc will skip the flag */
        } else {
            atomic_store_explicit(&tcache_ind, tc, memory_order_release);
            fprintf(stderr, "tcache.create: tcache_ind=%u\n", tc);
        }
    }





    atomic_store_explicit(&arena_ind, ind, memory_order_release);
    pthread_mutex_unlock(&lifecycle_lock);
    return 0;
}

/*void umf_allocator_finalize(void) {
    pthread_mutex_lock(&lifecycle_lock);
    unsigned ind = atomic_exchange_explicit(&arena_ind, UINT_MAX,
                                            memory_order_acq_rel);
    if (ind != UINT_MAX) {
        char path[64];
        snprintf(path, sizeof(path), "arena.%u.destroy", ind);
        mallctl(path, NULL, NULL, NULL, 0);
    }
    if (region_base) {
        munmap(region_base, region_cap);
        region_base = NULL;
        region_cap  = 0;
    }
    pthread_mutex_unlock(&lifecycle_lock);
}*/


void umf_allocator_finalize(void) {
    pthread_mutex_lock(&lifecycle_lock);
    unsigned ind = atomic_exchange_explicit(&arena_ind, UINT_MAX,
                                            memory_order_acq_rel);
    if (ind != UINT_MAX) {
        char path[64];
        snprintf(path, sizeof(path), "arena.%u.destroy", ind);
        mallctl(path, NULL, NULL, NULL, 0);
    }
    if (region_base) {
        munmap(region_base, region_cap);
        region_base = NULL;
        region_cap  = 0;
    }
    pthread_mutex_unlock(&lifecycle_lock);
}


void *umf_alloc(size_t size, size_t align) {
    if (size == 0) return NULL;
    unsigned ind = atomic_load_explicit(&arena_ind, memory_order_acquire);
    if (ind == UINT_MAX) return NULL;

    /* tcache left ENABLED — it absorbs most allocations on the fast path.
     * MALLOCX_ARENA routes backing memory through our hooks / region. */
    //int flags = MALLOCX_ARENA(ind);
    //int flags = MALLOCX_ARENA(ind) | MALLOCX_TCACHE(0);
    //if (align && align > sizeof(void *)) flags |= MALLOCX_ALIGN(align);
    //return mallocx(size, flags);
    int flags = MALLOCX_ARENA(ind);
    unsigned tc = atomic_load_explicit(&tcache_ind, memory_order_acquire);
    if (tc != UINT_MAX) flags |= MALLOCX_TCACHE(tc);
    if (align && align > sizeof(void *)) flags |= MALLOCX_ALIGN(align);
    return mallocx(size, flags);
}

void umf_dealloc(void *ptr) {
    if (!ptr) return;
    unsigned ind = atomic_load_explicit(&arena_ind, memory_order_acquire);
    if (ind == UINT_MAX) return;
    //int flags = MALLOCX_ARENA(ind) | MALLOCX_TCACHE(0);
    //dallocx(ptr, flags);

    int flags = MALLOCX_ARENA(ind);
    unsigned tc = atomic_load_explicit(&tcache_ind, memory_order_acquire);
    if (tc != UINT_MAX) flags |= MALLOCX_TCACHE(tc);
    dallocx(ptr, flags);
}

int check_tier(void *ptr) {
    (void)ptr;
    return 1;
}




