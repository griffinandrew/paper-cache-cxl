// jemalloc_extent_hooks.c — drop-in replacement for the UMF wrapper.
// Same public API: umf_allocator_init / _finalize / umf_alloc / umf_dealloc
// / check_tier. No UMF, no tracker, direct jemalloc arena with mbind.



/* working but doent fault in all pages
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

*/
/* If your system's jemalloc symbols are prefixed (installed alongside glibc
 * malloc rather than replacing it), uncomment. Check with:
 *   nm -D /usr/lib64/libjemalloc.so.2 | grep -E 'mallctl|mallocx' | head
 *
 */

 /*
#ifdef JEMALLOC_USE_PREFIX
#define mallctl  je_mallctl
#define mallocx  je_mallocx
#define dallocx  je_dallocx
#endif
*/

/* ------------------------------------------------------------------ */
/* State                                                               */
/* ------------------------------------------------------------------ */

/*
//Hot-path arena index. Sentinel UINT_MAX = uninitialized. 
static _Atomic unsigned arena_ind = UINT_MAX;

// NUMA node to bind extents to. Written once in init, read-only thereafter.
static int target_numa_node = -1;

static pthread_mutex_t lifecycle_lock = PTHREAD_MUTEX_INITIALIZER;
*/
/* ------------------------------------------------------------------ */
/* Extent hooks                                                        */
/* ------------------------------------------------------------------ */

/*
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
    return false; 
}

static bool extent_decommit(extent_hooks_t *h, void *a, size_t s,
                            size_t o, size_t l, unsigned ar) {
    (void)h;(void)a;(void)s;(void)o;(void)l;(void)ar;
    return true;  
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

//------------------------------------------------------------------ 
// Public API — same shape as the UMF wrapper                         
// ------------------------------------------------------------------ 

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

    // Do NOT register finalize via atexit; same reasoning as before.
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
    // Stub. Tag tier in object metadata instead of querying the allocator.
    (void)ptr;
    return 1;  // matches the UMF version's "pmem" return for our pool
}

*/

//end old working jemalloc extent hooks implementation, which is now unused but kept for reference (it wasnt prefaulting in....



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
 */
 
#ifdef JEMALLOC_USE_PREFIX
#define mallctl  je_mallctl
#define mallocx  je_mallocx
#define dallocx  je_dallocx
#endif
 
/* MADV_NOHUGEPAGE may be missing from very old <sys/mman.h>; define it so the
 * build doesn't fail. Value is stable in the Linux ABI. */
#ifndef MADV_NOHUGEPAGE
#define MADV_NOHUGEPAGE 15
#endif
 
/* ------------------------------------------------------------------ */
/* State                                                               */
/* ------------------------------------------------------------------ */
 
/* Hot-path arena index. Sentinel UINT_MAX = uninitialized. */
static _Atomic unsigned arena_ind = UINT_MAX;
 
/* NUMA node to bind extents to. Written once in init, read-only thereafter. */
static int target_numa_node = -1;
 
static pthread_mutex_t lifecycle_lock = PTHREAD_MUTEX_INITIALIZER;
 
/* When true, extent_dalloc retains memory (no munmap) so prewarmed,
 * prefaulted extents stay resident in jemalloc's pool for reuse.
 * Set before prewarm; left true for the lifetime of the benchmark. */
static _Atomic bool retain_extents = false;
 
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
 
/* Opt a freshly-mmap'd range out of transparent huge pages, so it is backed
 * only by standard (4 KiB) pages. MADV_NOHUGEPAGE overrides a system-wide
 * THP mode of "always" and prevents khugepaged from collapsing the range.
 * Must run BEFORE the range is faulted in. */
static void disable_thp(void *addr, size_t size) {
    madvise(addr, size, MADV_NOHUGEPAGE);
}
 
/* Touch every page so the kernel faults + zeroes physical pages now,
 * on the node selected by the preceding bind_to_node(). Moves first-touch
 * fault cost out of the eventual write to the object.
 * volatile store so the compiler cannot elide it. */
static void prefault_extent(void *addr, size_t size) {
    long pagesize = sysconf(_SC_PAGESIZE);
    if (pagesize <= 0) pagesize = 4096;
 
    volatile char *p = (volatile char *)addr;
    for (size_t off = 0; off < size; off += (size_t)pagesize) {
        p[off] = 0;
    }
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
        disable_thp(p, size);              /* standard pages only */
        prefault_extent(p, size);          /* after bind + THP opt-out */
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
    disable_thp((void *)aligned, size);       /* standard pages only */
    prefault_extent((void *)aligned, size);   /* after bind + trim + opt-out */
    *zero = true; *commit = true;
    return (void *)aligned;
}
 
static bool extent_dalloc(extent_hooks_t *hooks, void *addr, size_t size,
                          bool committed, unsigned arena) {
    (void)hooks; (void)committed; (void)arena;
    /* When retention is on, refuse the dalloc: returning true tells jemalloc
     * the hook did NOT free the extent, so jemalloc keeps it in its retained
     * pool — still mapped, still faulted, still bound to the PMEM node —
     * and will hand it back out on a future extent_alloc request instead of
     * calling mmap again. This is what makes prewarming stick. */
    if (atomic_load_explicit(&retain_extents, memory_order_acquire)) {
        return true;   /* not deallocated; jemalloc retains it */
    }
    return munmap(addr, size) != 0;
}
 
static void extent_destroy(extent_hooks_t *hooks, void *addr, size_t size,
                           bool committed, unsigned arena) {
    (void)hooks; (void)committed; (void)arena;
    /* destroy is called on arena teardown — always really unmap here,
     * regardless of retention, so we don't leak the region. */
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
 
/* Purge hooks: when retention is on, refuse to purge. Returning true means
 * "purge failed / not done", so jemalloc keeps the pages dirty and resident
 * rather than MADV_FREE / MADV_DONTNEED-ing them away. This stops the decay
 * machinery from quietly dropping prewarmed pages. */
static bool extent_purge_lazy(extent_hooks_t *h, void *addr, size_t s,
                              size_t offset, size_t length, unsigned ar) {
    (void)h;(void)s;(void)ar;
    if (atomic_load_explicit(&retain_extents, memory_order_acquire)) {
        return true;   /* not purged */
    }
    return madvise((char *)addr + offset, length, MADV_FREE) != 0;
}
 
static bool extent_purge_forced(extent_hooks_t *h, void *addr, size_t s,
                                size_t offset, size_t length, unsigned ar) {
    (void)h;(void)s;(void)ar;
    if (atomic_load_explicit(&retain_extents, memory_order_acquire)) {
        return true;   /* not purged */
    }
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
/* Decay control                                                       */
/* ------------------------------------------------------------------ */
 
/* Disable dirty/muzzy decay on the arena so jemalloc never tries to return
 * idle pages to the OS on a timer. Combined with the purge hooks refusing to
 * purge, this keeps prewarmed extents resident. Returns 0 on success. */
static int disable_arena_decay(unsigned ind) {
    char path[64];
    ssize_t neg1 = -1;   /* jemalloc interprets -1 as "never decay" */
 
    snprintf(path, sizeof(path), "arena.%u.dirty_decay_ms", ind);
    if (mallctl(path, NULL, NULL, &neg1, sizeof(neg1)) != 0) {
        fprintf(stderr, "warning: could not disable dirty_decay on arena %u\n", ind);
        return 1;
    }
 
    snprintf(path, sizeof(path), "arena.%u.muzzy_decay_ms", ind);
    if (mallctl(path, NULL, NULL, &neg1, sizeof(neg1)) != 0) {
        fprintf(stderr, "warning: could not disable muzzy_decay on arena %u\n", ind);
        return 1;
    }
    return 0;
}
 
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
 
    /* Stop the decay timer from purging idle pages. Non-fatal if it fails;
     * the purge hooks still refuse purges once retain_extents is set. */
    disable_arena_decay(ind);
 
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
        /* Allow real unmapping during teardown. */
        atomic_store_explicit(&retain_extents, false, memory_order_release);
        char path[64];
        snprintf(path, sizeof(path), "arena.%u.destroy", ind);
        mallctl(path, NULL, NULL, NULL, 0);
    }
    pthread_mutex_unlock(&lifecycle_lock);
}
 
/* Prewarm: force the arena to acquire, bind, opt-out-of-THP, and prefault
 * `bytes` worth of extents up front, then free them back into the arena's
 * retained pool so the benchmark's SETs reuse already-faulted memory.
 *
 * Must be called AFTER umf_allocator_init and BEFORE the measured workload.
 *
 * `chunk` is the size of each individual allocation used to grow the arena.
 * Using many medium chunks rather than one giant allocation makes the
 * resulting retained extents closer in size to what real SETs will request,
 * so jemalloc can actually reuse them for object-sized allocations.
 *
 * Returns 0 on success. */
int umf_allocator_prewarm(size_t bytes, size_t chunk) {
    unsigned ind = atomic_load_explicit(&arena_ind, memory_order_acquire);
    if (ind == UINT_MAX) return 1;
    if (bytes == 0) return 0;
    if (chunk == 0) chunk = 2 * 1024 * 1024;   /* 2 MiB default */
 
    /* Turn retention ON before we allocate, so when we free the prewarm
     * chunks below, extent_dalloc retains them instead of munmap-ing. */
    atomic_store_explicit(&retain_extents, true, memory_order_release);
 
    int flags = MALLOCX_ARENA(ind) | MALLOCX_TCACHE_NONE;
 
    size_t n = (bytes + chunk - 1) / chunk;
    void **ptrs = calloc(n, sizeof(void *));
    if (!ptrs) return 2;
 
    size_t got = 0;
    for (size_t i = 0; i < n; i++) {
        void *p = mallocx(chunk, flags);
        if (!p) {
            /* Ran out — free what we have and report partial. */
            for (size_t j = 0; j < got; j++) {
                dallocx(ptrs[j], flags);
            }
            free(ptrs);
            return 3;
        }
        /* extent_alloc already prefaulted the underlying extent, but a
         * mallocx may be satisfied from an extent larger than `chunk`;
         * touching the returned region guarantees this chunk specifically
         * is resident even if extent bookkeeping split it. */
        prefault_extent(p, chunk);
        ptrs[got++] = p;
    }
 
    /* Free everything back. With retain_extents set and decay disabled,
     * the pages stay mapped, faulted, and bound — parked in jemalloc's
     * pool ready to satisfy the benchmark's allocations. */
    for (size_t i = 0; i < got; i++) {
        dallocx(ptrs[i], flags);
    }
    free(ptrs);
    return 0;
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
 



