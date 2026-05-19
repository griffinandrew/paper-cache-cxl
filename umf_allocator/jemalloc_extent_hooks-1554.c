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
/* Design note                                                         */
/* ------------------------------------------------------------------ */
/*
 * This allocator does NOT prefault. Instead it makes the extent hooks
 * behave like native jemalloc: memory is acquired from the OS rarely,
 * faulted lazily on first touch, and then RECYCLED internally forever.
 *
 * The three things that make this work, vs. the original hook file:
 *   1. extent_dalloc RETAINS the extent (no munmap) — jemalloc keeps it
 *      mapped + faulted and hands it back out on the next extent_alloc.
 *   2. The purge hooks REFUSE to purge — pages are never MADV_DONTNEED'd
 *      out from under a retained extent.
 *   3. Arena dirty/muzzy decay is DISABLED — jemalloc never tries to
 *      return idle pages to the OS on a timer.
 *
 * Net effect: each page faults exactly once across the whole run, and the
 * cost amortizes away — the same reason numactl + stock jemalloc has good
 * steady-state SET latency. A short warmup phase before measurement
 * absorbs those one-time first-touch faults; no prefault is needed.
 *
 * Caveat: with retention + no purge + no decay, the arena only ever GROWS.
 * It never returns memory to the OS for the process lifetime. This is a
 * benchmark build — do not use it in long-running or memory-constrained
 * processes.
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
    long rc = mbind(addr, size, MPOL_BIND,
                    mask->maskp, mask->size + 1, 0);
    if (rc != 0) {
        /* Silent mbind failure means pages land on the default (DRAM) node
         * and the benchmark measures the wrong tier — so make it loud. */
        fprintf(stderr,
                "bind_to_node: mbind(node=%d, %zu bytes) FAILED: %s\n",
                target_numa_node, size, strerror(errno));
    }
    numa_bitmask_free(mask);
}

/* Opt a freshly-mmap'd range out of transparent huge pages so it is backed
 * only by standard (4 KiB) pages. Overrides a system THP mode of "always". */
static void disable_thp(void *addr, size_t size) {
    madvise(addr, size, MADV_NOHUGEPAGE);
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
        disable_thp(p, size);
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
    disable_thp((void *)aligned, size);
    *zero = true; *commit = true;
    return (void *)aligned;
}

/* RETAIN, do not munmap. Returning true tells jemalloc the hook did NOT
 * free the extent, so jemalloc keeps it in its pool — still mapped, still
 * faulted, still NUMA-bound — and reuses it for a future extent_alloc.
 * This is the single most important change: it converts a re-fault-on-every-
 * free storm into fault-once-then-recycle, matching native jemalloc. */
static bool extent_dalloc(extent_hooks_t *hooks, void *addr, size_t size,
                          bool committed, unsigned arena) {
    (void)hooks; (void)addr; (void)size; (void)committed; (void)arena;
    return true;   /* not deallocated — jemalloc retains and reuses it */
}

static void extent_destroy(extent_hooks_t *hooks, void *addr, size_t size,
                           bool committed, unsigned arena) {
    (void)hooks; (void)committed; (void)arena;
    /* destroy is called on arena teardown — really unmap here so the
     * region is not leaked when the benchmark exits. */
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

/* REFUSE to purge. Returning true means "purge not done", so jemalloc keeps
 * the pages dirty and resident rather than MADV_FREE / MADV_DONTNEED-ing
 * them. Without this, pages get dropped out of retained extents and
 * re-fault on next touch — defeating the retention above. */
static bool extent_purge_lazy(extent_hooks_t *h, void *addr, size_t s,
                              size_t offset, size_t length, unsigned ar) {
    (void)h;(void)addr;(void)s;(void)offset;(void)length;(void)ar;
    return true;   /* not purged — keep pages resident */
}

static bool extent_purge_forced(extent_hooks_t *h, void *addr, size_t s,
                                size_t offset, size_t length, unsigned ar) {
    (void)h;(void)addr;(void)s;(void)offset;(void)length;(void)ar;
    return true;   /* not purged — keep pages resident */
}

static bool extent_split(extent_hooks_t *h, void *a, size_t s,
                         size_t sa, size_t sb, bool c, unsigned ar) {
    (void)h;(void)a;(void)s;(void)sa;(void)sb;(void)c;(void)ar;
    /* Returning false here means "split succeeded", which lets jemalloc
     * subdivide a large retained extent to satisfy smaller requests —
     * important for reuse efficiency. The address space is already one
     * contiguous mapping, so no syscall is needed to split. */
    return false;
}

static bool extent_merge(extent_hooks_t *h, void *a, size_t sa,
                         void *b, size_t sb, bool c, unsigned ar) {
    (void)h;(void)a;(void)sa;(void)b;(void)sb;(void)c;(void)ar;
    /* Returning false means "merge succeeded", letting jemalloc coalesce
     * adjacent retained extents back into a larger one. */
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
 * idle pages to the OS on a timer. Combined with the purge hooks refusing
 * to purge, this keeps faulted pages resident for the process lifetime. */
static void disable_arena_decay(unsigned ind) {
    char path[64];
    ssize_t neg1 = -1;   /* jemalloc interprets -1 as "never decay" */

    snprintf(path, sizeof(path), "arena.%u.dirty_decay_ms", ind);
    if (mallctl(path, NULL, NULL, &neg1, sizeof(neg1)) != 0) {
        fprintf(stderr,
                "warning: could not disable dirty_decay on arena %u\n", ind);
    }

    snprintf(path, sizeof(path), "arena.%u.muzzy_decay_ms", ind);
    if (mallctl(path, NULL, NULL, &neg1, sizeof(neg1)) != 0) {
        fprintf(stderr,
                "warning: could not disable muzzy_decay on arena %u\n", ind);
    }
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

    /* Disable decay so retained extents are never purged on a timer. */
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
        char path[64];
        /* arena.<ind>.destroy invokes extent_destroy on every extent,
         * which really munmaps — so retained memory is released here. */
        snprintf(path, sizeof(path), "arena.%u.destroy", ind);
        mallctl(path, NULL, NULL, NULL, 0);
    }
    pthread_mutex_unlock(&lifecycle_lock);
}

void *umf_alloc(size_t size, size_t align) {
    if (size == 0) return NULL;
    unsigned ind = atomic_load_explicit(&arena_ind, memory_order_acquire);
    if (ind == UINT_MAX) return NULL;

    /* Note: tcache is intentionally LEFT ENABLED (no MALLOCX_TCACHE_NONE).
     * The thread cache absorbs most allocations lock-free, which is a large
     * part of why native jemalloc has low per-op latency. The arena binding
     * via MALLOCX_ARENA still routes the backing memory through our hooks. */
    int flags = MALLOCX_ARENA(ind);
    if (align && align > sizeof(void *)) {
        flags |= MALLOCX_ALIGN(align);
    }
    return mallocx(size, flags);
}

void umf_dealloc(void *ptr) {
    if (!ptr) return;
    unsigned ind = atomic_load_explicit(&arena_ind, memory_order_acquire);
    if (ind == UINT_MAX) return;
    /* Match umf_alloc: tcache enabled. */
    dallocx(ptr, MALLOCX_ARENA(ind));
}

int check_tier(void *ptr) {
    /* Stub. Tag tier in object metadata instead of querying the allocator. */
    (void)ptr;
    return 1;  /* matches the UMF version's "pmem" return for our pool */
}



