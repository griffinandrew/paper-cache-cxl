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
 * One jemalloc instance, N tiers. Each tier owns:
 *   - one big contiguous region (mmap MAP_NORESERVE, mbind'd to its node
 *     ONCE, never prefaulted), carved by a bump pointer;
 *   - one jemalloc arena whose extent hooks pull from that region;
 *   - one jemalloc tcache, so cached objects never cross tiers.
 *
 * Faulting is still LAZY on first touch (not prefault). Every fault lands
 * inside that tier's single large VMA, which is the contiguous fault
 * pattern you wanted.
 *
 * Per-tier context: extent_hooks_t is the FIRST member of tier_t, so the
 * `hooks` pointer jemalloc passes each hook IS the tier_t. That's the only
 * way to drive multiple arenas/regions from one allocator without globals.
 *
 * Tier routing:
 *   umf_alloc(tier, ...)  -> that tier's (arena, tcache)
 *   umf_dealloc(ptr)      -> range-lookup the owning tier, free into ITS
 *                            tcache (NOT the caller's) so tcache stays pure
 *   check_tier(ptr)       -> tier id by range lookup, or -1
 */

#ifndef MADV_NOHUGEPAGE
#define MADV_NOHUGEPAGE 15
#endif

#define MAX_TIERS 4
#define DEFAULT_REGION_BYTES (25ULL * 1024 * 1024 * 1024)

/* ------------------------------------------------------------------ */
/* Per-tier state                                                      */
/* ------------------------------------------------------------------ */

typedef struct {
    extent_hooks_t   hooks;        /* MUST be first member (cast target) */
    int              in_use;
    int              target_node;
    void            *region_base;
    size_t           region_cap;
    _Atomic size_t   region_off;   /* bump cursor */
    _Atomic unsigned arena_ind;
    _Atomic unsigned tcache_ind;
} tier_t;

static tier_t          tiers[MAX_TIERS];
static pthread_mutex_t lifecycle_lock = PTHREAD_MUTEX_INITIALIZER;

/* ------------------------------------------------------------------ */
/* Extent hooks — recover tier via cast on the hooks pointer            */
/* ------------------------------------------------------------------ */

static void *extent_alloc(extent_hooks_t *hooks, void *new_addr, size_t size,
                          size_t alignment, bool *zero, bool *commit,
                          unsigned arena) {
    (void)new_addr; (void)arena;
    tier_t *t = (tier_t *)hooks;          /* hooks == &tier->hooks == tier */
    if (!t->region_base || t->region_cap == 0) return NULL;

    size_t align = alignment ? alignment : 1;
    for (;;) {
        size_t cur     = atomic_load_explicit(&t->region_off,
                                               memory_order_relaxed);
        size_t aligned = (cur + align - 1) & ~(align - 1);
        if (aligned < cur) return NULL;               /* alignment overflow */
        if (aligned + size < aligned) return NULL;    /* size overflow */
        size_t end = aligned + size;
        if (end > t->region_cap) {
            fprintf(stderr,
                    "extent_alloc[node %d]: region exhausted (need %zu, cap %zu)\n",
                    t->target_node, end, t->region_cap);
            return NULL;
        }
        if (atomic_compare_exchange_weak_explicit(
                &t->region_off, &cur, end,
                memory_order_seq_cst, memory_order_relaxed)) {
            *zero   = true;   /* fresh anon pages zero on first fault */
            *commit = true;
            return (char *)t->region_base + aligned;
        }
        /* CAS lost a race — retry. */
    }
}

static bool extent_dalloc(extent_hooks_t *h, void *a, size_t s,
                          bool c, unsigned ar) {
    (void)h;(void)a;(void)s;(void)c;(void)ar;
    return true;    /* bump region: retain, bulk reclaim at teardown */
}

static void extent_destroy(extent_hooks_t *h, void *a, size_t s,
                           bool c, unsigned ar) {
    (void)h;(void)a;(void)s;(void)c;(void)ar;   /* region freed at finalize */
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

static bool extent_purge_lazy(extent_hooks_t *h, void *a, size_t s,
                              size_t o, size_t l, unsigned ar) {
    (void)h;(void)a;(void)s;(void)o;(void)l;(void)ar;
    return true;    /* refuse — never MADV_DONTNEED our pages */
}

static bool extent_purge_forced(extent_hooks_t *h, void *a, size_t s,
                                size_t o, size_t l, unsigned ar) {
    (void)h;(void)a;(void)s;(void)o;(void)l;(void)ar;
    return true;
}

static bool extent_split(extent_hooks_t *h, void *a, size_t s,
                         size_t sa, size_t sb, bool c, unsigned ar) {
    (void)h;(void)a;(void)s;(void)sa;(void)sb;(void)c;(void)ar;
    return false;   /* one contiguous VMA — split is free */
}

static bool extent_merge(extent_hooks_t *h, void *a, size_t sa,
                         void *b, size_t sb, bool c, unsigned ar) {
    (void)h;(void)a;(void)sa;(void)b;(void)sb;(void)c;(void)ar;
    return false;
}

/* Fill a tier's embedded hooks struct (same fns for every tier; the
 * per-tier data lives in the surrounding tier_t). */
static void tier_init_hooks(tier_t *t) {
    t->hooks.alloc        = extent_alloc;
    t->hooks.dalloc       = extent_dalloc;
    t->hooks.destroy      = extent_destroy;
    t->hooks.commit       = extent_commit;
    t->hooks.decommit     = extent_decommit;
    t->hooks.purge_lazy   = extent_purge_lazy;
    t->hooks.purge_forced = extent_purge_forced;
    t->hooks.split        = extent_split;
    t->hooks.merge        = extent_merge;
}

/* ------------------------------------------------------------------ */
/* Region setup                                                        */
/* ------------------------------------------------------------------ */

static void bind_region(tier_t *t, void *addr, size_t size) {
    if (t->target_node < 0) return;
    struct bitmask *mask = numa_bitmask_alloc(numa_num_possible_nodes());
    if (!mask) return;
    numa_bitmask_setbit(mask, t->target_node);
    /* Policy only (flags 0): nothing is faulted yet, so no MOVE needed;
     * faults land on target_node afterward. */
    if (mbind(addr, size, MPOL_BIND, mask->maskp, mask->size + 1, 0) != 0) {
        fprintf(stderr, "bind_region[node %d]: mbind(%zu) FAILED: %s\n",
                t->target_node, size, strerror(errno));
    }
    numa_bitmask_free(mask);
}

static int region_reserve(tier_t *t, size_t bytes) {
    if (bytes == 0) bytes = DEFAULT_REGION_BYTES;
    void *p = mmap(NULL, bytes, PROT_READ | PROT_WRITE,
                   MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
    if (p == MAP_FAILED) {
        fprintf(stderr, "region_reserve[node %d]: mmap(%zu) failed: %s\n",
                t->target_node, bytes, strerror(errno));
        return 1;
    }
    bind_region(t, p, bytes);
    t->region_base = p;
    t->region_cap  = bytes;
    atomic_store_explicit(&t->region_off, 0, memory_order_release);
    fprintf(stderr, "region_reserve: tier node %d, %zu bytes at %p\n",
            t->target_node, bytes, p);
    return 0;
}

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
/* Lookup                                                               */
/* ------------------------------------------------------------------ */

static int tier_of_ptr(void *ptr) {
    uintptr_t p = (uintptr_t)ptr;
    for (int i = 0; i < MAX_TIERS; i++) {
        if (!tiers[i].in_use) continue;
        uintptr_t base = (uintptr_t)tiers[i].region_base;
        if (p >= base && p < base + tiers[i].region_cap) return i;
    }
    return -1;
}

/* ------------------------------------------------------------------ */
/* Public API                                                           */
/* ------------------------------------------------------------------ */

/* Initialize one tier. Call once per tier (e.g. tier 0 -> DRAM node,
 * tier 1 -> PMEM-as-system-ram node). bytes == 0 uses the default. */
int umf_tier_init(int tier_id, int numa_node, size_t bytes) {
    if (tier_id < 0 || tier_id >= MAX_TIERS) return 5;
    pthread_mutex_lock(&lifecycle_lock);

    tier_t *t = &tiers[tier_id];
    if (t->in_use) {                       /* idempotent */
        pthread_mutex_unlock(&lifecycle_lock);
        return 0;
    }

    if (numa_available() < 0) {
        fprintf(stderr, "libnuma not available\n");
        pthread_mutex_unlock(&lifecycle_lock);
        return 1;
    }

    atomic_store_explicit(&t->arena_ind,  UINT_MAX, memory_order_relaxed);
    atomic_store_explicit(&t->tcache_ind, UINT_MAX, memory_order_relaxed);
    t->target_node = numa_node;
    tier_init_hooks(t);

    if (region_reserve(t, bytes) != 0) {
        pthread_mutex_unlock(&lifecycle_lock);
        return 4;
    }

    unsigned ind;
    size_t ind_sz = sizeof(ind);
    if (mallctl("arenas.create", &ind, &ind_sz, NULL, 0) != 0) {
        fprintf(stderr, "arenas.create failed\n");
        munmap(t->region_base, t->region_cap);
        t->region_base = NULL; t->region_cap = 0;
        pthread_mutex_unlock(&lifecycle_lock);
        return 2;
    }

    char path[64];
    snprintf(path, sizeof(path), "arena.%u.extent_hooks", ind);
    extent_hooks_t *hooks_ptr = &t->hooks;
    if (mallctl(path, NULL, NULL, &hooks_ptr, sizeof(hooks_ptr)) != 0) {
        fprintf(stderr, "setting extent_hooks on arena %u failed\n", ind);
        snprintf(path, sizeof(path), "arena.%u.destroy", ind);
        mallctl(path, NULL, NULL, NULL, 0);
        munmap(t->region_base, t->region_cap);
        t->region_base = NULL; t->region_cap = 0;
        pthread_mutex_unlock(&lifecycle_lock);
        return 3;
    }

    disable_arena_decay(ind);

    /* Per-tier tcache: keeps cached objects from ever crossing tiers. */
    unsigned tc;
    size_t tc_sz = sizeof(tc);
    if (mallctl("tcache.create", &tc, &tc_sz, NULL, 0) != 0) {
        fprintf(stderr, "tcache.create failed (tier %d) — running without tcache\n",
                tier_id);
    } else {
        atomic_store_explicit(&t->tcache_ind, tc, memory_order_release);
        fprintf(stderr, "tcache.create: tier %d tcache_ind=%u\n", tier_id, tc);
    }

    atomic_store_explicit(&t->arena_ind, ind, memory_order_release);
    t->in_use = 1;
    pthread_mutex_unlock(&lifecycle_lock);
    return 0;
}

/* Back-compat shim: old single-node entry point initializes tier 0. */
int umf_allocator_init(int numa_node) {
    return umf_tier_init(0, numa_node, 0);
}

void umf_allocator_finalize(void) {
    pthread_mutex_lock(&lifecycle_lock);
    for (int i = 0; i < MAX_TIERS; i++) {
        tier_t *t = &tiers[i];
        if (!t->in_use) continue;

        unsigned tc = atomic_exchange_explicit(&t->tcache_ind, UINT_MAX,
                                               memory_order_acq_rel);
        if (tc != UINT_MAX) {
            mallctl("tcache.flush",   NULL, NULL, &tc, sizeof(tc));
            mallctl("tcache.destroy", NULL, NULL, &tc, sizeof(tc));
        }

        unsigned ind = atomic_exchange_explicit(&t->arena_ind, UINT_MAX,
                                               memory_order_acq_rel);
        if (ind != UINT_MAX) {
            char path[64];
            snprintf(path, sizeof(path), "arena.%u.destroy", ind);
            mallctl(path, NULL, NULL, NULL, 0);
        }

        if (t->region_base) {
            munmap(t->region_base, t->region_cap);
            t->region_base = NULL;
            t->region_cap  = 0;
        }
        t->in_use = 0;
    }
    pthread_mutex_unlock(&lifecycle_lock);
}

/* Route a new allocation to a specific tier. */
void *umf_alloc(int tier, size_t size, size_t align) {
    if (size == 0 || tier < 0 || tier >= MAX_TIERS) return NULL;
    tier_t *t = &tiers[tier];
    if (!t->in_use) return NULL;

    unsigned ind = atomic_load_explicit(&t->arena_ind, memory_order_acquire);
    if (ind == UINT_MAX) return NULL;

    int flags = MALLOCX_ARENA(ind);
    unsigned tc = atomic_load_explicit(&t->tcache_ind, memory_order_acquire);
    if (tc != UINT_MAX) flags |= MALLOCX_TCACHE(tc);
    if (align && align > sizeof(void *)) flags |= MALLOCX_ALIGN(align);
    return mallocx(size, flags);
}

/* Free, routing back into the OWNING tier's tcache (found by range), not
 * the caller's — this is what keeps each tcache single-tier. */
void umf_dealloc(void *ptr) {
    if (!ptr) return;
    int ti = tier_of_ptr(ptr);
    if (ti < 0) return;                  /* not from any of our regions */
    tier_t *t = &tiers[ti];

    unsigned ind = atomic_load_explicit(&t->arena_ind, memory_order_acquire);
    if (ind == UINT_MAX) return;
    int flags = MALLOCX_ARENA(ind);
    unsigned tc = atomic_load_explicit(&t->tcache_ind, memory_order_acquire);
    if (tc != UINT_MAX) flags |= MALLOCX_TCACHE(tc);
    dallocx(ptr, flags);
}

/* Now returns the real tier id (0-based) or -1 if the pointer isn't ours. */
int check_tier(void *ptr) {
    return tier_of_ptr(ptr);
}





