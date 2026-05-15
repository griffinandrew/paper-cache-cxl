// jemalloc_extent_hooks.c — drop-in replacement for the UMF wrapper.
// Same public API: umf_allocator_init / _finalize / umf_alloc / umf_dealloc
// / check_tier. No UMF, no tracker, direct jemalloc arena with mbind.

#define _GNU_SOURCE
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdbool.h>
#include <string.h>
#include <limits.h>
#include <stdatomic.h>
#include <pthread.h>
#include <unistd.h>
#include <sys/mman.h>
#include <numaif.h>
#include <numa.h>
#include <jemalloc/jemalloc.h>

/* If your system's jemalloc symbols are prefixed (installed alongside glibc
 * malloc rather than replacing it), uncomment. Check with:
 *   nm -D /usr/lib64/libjemalloc.so.2 | grep -E 'mallctl|mallocx' | head
 *
 * #define mallctl  je_mallctl
 * #define mallocx  je_mallocx
 * #define dallocx  je_dallocx
 */

/* ------------------------------------------------------------------ */
/* State                                                               */
/* ------------------------------------------------------------------ */

/* Hot-path arena index. Sentinel UINT_MAX = uninitialized. */
static _Atomic unsigned arena_ind = UINT_MAX;

/* NUMA node to bind extents to. Written once in init, read-only thereafter. */
static int target_numa_node = -1;

static pthread_mutex_t lifecycle_lock = PTHREAD_MUTEX_INITIALIZER;

/* ------------------------------------------------------------------ */
/* Extent hooks                                                        */
/* ------------------------------------------------------------------ */

static void bind_to_node(void *addr, size_t size) {
    if (target_numa_node < 0) return;

    struct bitmask *mask = numa_bitmask_alloc(numa_num_possible_nodes());
    if (!mask) return;
    numa_bitmask_setbit(mask, target_numa_node);
    mbind(addr, size, MPOL_BIND, mask->maskp, mask->size + 1, 0);
    numa_bitmask_free(mask);
}

static void *extent_alloc(extent_hooks_t *hooks, void *new_addr, size_t size,
                          size_t alignment, bool *zero, bool *commit,
                          unsigned arena) {
    (void)hooks; (void)arena;
    long pagesize = sysconf(_SC_PAGESIZE);

    if (alignment <= (size_t)pagesize) {
        void *p = mmap(new_addr, size, PROT_READ | PROT_WRITE,
                       MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        if (p == MAP_FAILED) return NULL;
        bind_to_node(p, size);
        *zero = true; *commit = true;
        return p;
    }

    size_t over = size + alignment;
    void *p = mmap(new_addr, over, PROT_READ | PROT_WRITE,
                   MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (p == MAP_FAILED) return NULL;

    uintptr_t aligned = ((uintptr_t)p + alignment - 1) & ~(alignment - 1);
    size_t head = aligned - (uintptr_t)p;
    size_t tail = over - head - size;
    if (head) munmap(p, head);
    if (tail) munmap((void *)(aligned + size), tail);

    bind_to_node((void *)aligned, size);
    *zero = true; *commit = true;
    return (void *)aligned;
}

static bool extent_dalloc(extent_hooks_t *hooks, void *addr, size_t size,
                          bool committed, unsigned arena) {
    (void)hooks; (void)committed; (void)arena;
    return munmap(addr, size) != 0;
}

static void extent_destroy(extent_hooks_t *hooks, void *addr, size_t size,
                           bool committed, unsigned arena) {
    (void)hooks; (void)committed; (void)arena;
    munmap(addr, size);
}

static bool extent_commit(extent_hooks_t *h, void *a, size_t s,
                          size_t o, size_t l, unsigned ar) {
    (void)h;(void)a;(void)s;(void)o;(void)l;(void)ar;
    return false;  /* already committed */
}

static bool extent_decommit(extent_hooks_t *h, void *a, size_t s,
                            size_t o, size_t l, unsigned ar) {
    (void)h;(void)a;(void)s;(void)o;(void)l;(void)ar;
    return true;   /* decommit unsupported, keep pages */
}

static bool extent_purge_lazy(extent_hooks_t *h, void *addr, size_t s,
                              size_t offset, size_t length, unsigned ar) {
    (void)h;(void)s;(void)ar;
    return madvise((char *)addr + offset, length, MADV_FREE) != 0;
}

static bool extent_purge_forced(extent_hooks_t *h, void *addr, size_t s,
                                size_t offset, size_t length, unsigned ar) {
    (void)h;(void)s;(void)ar;
    return madvise((char *)addr + offset, length, MADV_DONTNEED) != 0;
}

static bool extent_split(extent_hooks_t *h, void *a, size_t s,
                         size_t sa, size_t sb, bool c, unsigned ar) {
    (void)h;(void)a;(void)s;(void)sa;(void)sb;(void)c;(void)ar;
    return false;
}

static bool extent_merge(extent_hooks_t *h, void *a, size_t sa,
                         void *b, size_t sb, bool c, unsigned ar) {
    (void)h;(void)a;(void)sa;(void)b;(void)sb;(void)c;(void)ar;
    return false;
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
/* Public API — same shape as the UMF wrapper                          */
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
    target_numa_node = numa_node;

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

    atomic_store_explicit(&arena_ind, ind, memory_order_release);
    pthread_mutex_unlock(&lifecycle_lock);

    /* Do NOT register finalize via atexit; same reasoning as before. */
    return 0;
}

void umf_allocator_finalize(void) {
    pthread_mutex_lock(&lifecycle_lock);
    unsigned ind = atomic_exchange_explicit(&arena_ind, UINT_MAX,
                                            memory_order_acq_rel);
    if (ind != UINT_MAX) {
        char path[64];
        snprintf(path, sizeof(path), "arena.%u.destroy", ind);
        mallctl(path, NULL, NULL, NULL, 0);
    }
    pthread_mutex_unlock(&lifecycle_lock);
}

void *umf_alloc(size_t size, size_t align) {
    if (size == 0) return NULL;
    unsigned ind = atomic_load_explicit(&arena_ind, memory_order_acquire);
    if (ind == UINT_MAX) return NULL;

    int flags = MALLOCX_ARENA(ind) | MALLOCX_TCACHE_NONE;
    if (align && align > sizeof(void *)) {
        flags |= MALLOCX_ALIGN(align);
    }
    return mallocx(size, flags);
}

void umf_dealloc(void *ptr) {
    if (!ptr) return;
    unsigned ind = atomic_load_explicit(&arena_ind, memory_order_acquire);
    if (ind == UINT_MAX) return;
    dallocx(ptr, MALLOCX_ARENA(ind) | MALLOCX_TCACHE_NONE);
}

int check_tier(void *ptr) {
    /* Stub. Tag tier in object metadata instead of querying the allocator. */
    (void)ptr;
    return 1;  /* matches the UMF version's "pmem" return for our pool */
}


