


#include <pthread.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>
#include <string.h>
#include <errno.h>
#include <limits.h>
#include <fcntl.h>
#include <sys/mman.h>

#include <umf/memory_pool.h>
#include <umf/memory_provider.h>
#include <umf/providers/provider_os_memory.h>
#include <umf/pools/pool_scalable.h>
#include <umf/pools/pool_jemalloc.h>

// DRAM hard cap (node 0 only) uses the SYSTEM jemalloc's extent-hooks API
// (distinct from the jemalloc statically bundled inside libumf.so that
// backs umfJemallocPoolOps() below for node 1/PMEM) -- see the "DRAM hard
// cap" block further down for why the two coexist safely in one process.
// Still include the header for its type/macro declarations (extent_hooks_t,
// MALLOCX_* macros) -- but NOT for its extern mallctl/mallocx/dallocx
// declarations, which are deliberately never called directly (see the
// dlopen/dlsym note below).
#include <jemalloc/jemalloc.h>
#include <numaif.h>
#include <numa.h>
#include <dlfcn.h>

// mallctl/mallocx/dallocx are resolved via dlopen(RTLD_LOCAL)+dlsym at
// runtime, lazily, only when PAPER_CACHE_DRAM_CAP_BYTES is actually set --
// deliberately NOT linked at build time (no cargo:rustc-link-lib / rustc-
// link-arg for libjemalloc.so.2). A first version linked it directly, which
// compiled and worked when the cap was exercised, but broke allocation
// process-wide even with the cap *unset*: the system libjemalloc.so.2 (like
// most jemalloc builds without --with-jemalloc-prefix) exports unprefixed
// `malloc`/`free`/`calloc`/`realloc` symbols (confirmed via `nm -D`), so
// merely linking it into the binary makes it a global malloc replacement
// for the *entire* process via standard ELF symbol interposition -- every
// allocation anywhere (UMF's own internals, libnuma, any other C
// dependency, not just this file's own code) silently gets rerouted to
// jemalloc's regular default arenas, and any place still holding a
// glibc-malloc'd pointer risks being freed through the interposed
// allocator instead (or vice versa), a mismatched-allocator condition that
// is undefined behavior and can corrupt heap metadata over many alloc/free
// cycles. Reported by the user as allocation failures (both `DRAMObjects`
// and `HybridObjects`) on a >1M-object, unrelated (cap-disabled) run.
// `dlopen(..., RTLD_LOCAL)` avoids this: `RTLD_LOCAL` (the default, but
// named explicitly here) keeps the loaded library's symbols out of the
// process's global symbol table, so they are reachable only via this
// file's own `dlsym` handles -- no interposition -- and skipping the
// dlopen entirely when the env var is unset means libjemalloc.so.2 is
// never even loaded into the process for users who don't opt into this
// feature at all.
typedef int   (*jemalloc_mallctl_fn)(const char *, void *, size_t *, void *, size_t);
typedef void *(*jemalloc_mallocx_fn)(size_t, int);
typedef void  (*jemalloc_dallocx_fn)(void *, int);

static void *dram_cap_jemalloc_handle = NULL;
static jemalloc_mallctl_fn dram_cap_mallctl = NULL;
static jemalloc_mallocx_fn dram_cap_mallocx = NULL;
static jemalloc_dallocx_fn dram_cap_dallocx = NULL;

// Tries both the unprefixed and `je_`-prefixed symbol names since different
// distributions' jemalloc packages differ (this sandbox's is unprefixed,
// confirmed via `nm -D`; kept generic rather than hardcoding that).
static bool dram_cap_load_jemalloc(void) {
    if (dram_cap_mallctl && dram_cap_mallocx && dram_cap_dallocx) {
        return true;
    }

    static const char *candidates[] = {
        "libjemalloc.so.2",
        "/usr/lib64/libjemalloc.so.2",
        "/usr/lib/x86_64-linux-gnu/libjemalloc.so.2",
    };

    for (size_t i = 0; i < sizeof(candidates) / sizeof(candidates[0]); i++) {
        dram_cap_jemalloc_handle = dlopen(candidates[i], RTLD_NOW | RTLD_LOCAL);
        if (dram_cap_jemalloc_handle) break;
    }

    if (!dram_cap_jemalloc_handle) {
        fprintf(stderr, "dram_cap: dlopen(libjemalloc.so.2) failed: %s\n", dlerror());
        return false;
    }

    dram_cap_mallctl = (jemalloc_mallctl_fn)dlsym(dram_cap_jemalloc_handle, "mallctl");
    dram_cap_mallocx = (jemalloc_mallocx_fn)dlsym(dram_cap_jemalloc_handle, "mallocx");
    dram_cap_dallocx = (jemalloc_dallocx_fn)dlsym(dram_cap_jemalloc_handle, "dallocx");

    if (!dram_cap_mallctl || !dram_cap_mallocx || !dram_cap_dallocx) {
        dram_cap_mallctl = (jemalloc_mallctl_fn)dlsym(dram_cap_jemalloc_handle, "je_mallctl");
        dram_cap_mallocx = (jemalloc_mallocx_fn)dlsym(dram_cap_jemalloc_handle, "je_mallocx");
        dram_cap_dallocx = (jemalloc_dallocx_fn)dlsym(dram_cap_jemalloc_handle, "je_dallocx");
    }

    if (!dram_cap_mallctl || !dram_cap_mallocx || !dram_cap_dallocx) {
        fprintf(stderr, "dram_cap: dlsym failed to resolve mallctl/mallocx/dallocx "
                         "(tried both unprefixed and je_-prefixed names)\n");
        return false;
    }

    return true;
}

#define MAX_NODES 8

// Per-node state. Index by NUMA node id.
static atomic_uintptr_t pools[MAX_NODES];
static umf_memory_provider_handle_t providers[MAX_NODES];
static umf_os_memory_provider_params_handle_t os_params_arr[MAX_NODES];
static pthread_mutex_t lifecycle_lock = PTHREAD_MUTEX_INITIALIZER;

/* ------------------------------------------------------------------ */
/* DRAM hard cap (node 0 / DRAMObjects, the crate's #[global_allocator], */
/* only) -- a custom jemalloc arena whose extent hooks bump-allocate    */
/* from a fixed-size, MAP_NORESERVE-reserved, MPOL_BIND-pinned region   */
/* and return NULL once the cap is exhausted: a genuine structural      */
/* ceiling, not a statistical tendency like the pool-level tuning       */
/* (KeepAllMemory, arena count, decay knobs -- see CLAUDE.md's DRAM-    */
/* usage investigation) tried and ruled out before this.                */
/*                                                                      */
/* Ported from the previously-unused umf_allocator/jemalloc_extent_     */
/* hooks.c (not compiled into this build), simplified from its N-tier   */
/* design to a single tier since only node 0 ever uses this path;       */
/* node 1 (HybridObjects/PMEM) keeps using the umfJemallocPoolOps()     */
/* pool above unchanged. Dealloc/decommit/purge hooks all refuse to     */
/* release -- bump/retain at the extent level, while individual         */
/* allocations within the region still get jemalloc's normal size-class */
/* reuse via its own arena machinery.                                   */
/*                                                                      */
/* Opt-in via the PAPER_CACHE_DRAM_CAP_BYTES environment variable, read */
/* once at node 0's lazy init (DRAMObjects::INIT.call_once fires on the */
/* process's first-ever heap allocation -- before any PaperCache::new() */
/* call, so the cap size cannot be threaded through as a constructor    */
/* parameter the way fast_tier_size is). If unset, node 0 falls back to */
/* the existing UMF jemalloc pool path below, unchanged.                */
/* ------------------------------------------------------------------ */

typedef struct {
    extent_hooks_t   hooks;        /* MUST be first member (cast target) */
    void            *region_base;
    size_t           region_cap;
    _Atomic size_t   region_off;   /* bump cursor */
    _Atomic unsigned arena_ind;
} dram_cap_tier_t;

static dram_cap_tier_t dram_cap_tier;
static atomic_bool     dram_cap_active = ATOMIC_VAR_INIT(false);

static void *dram_cap_extent_alloc(extent_hooks_t *hooks, void *new_addr, size_t size,
                                    size_t alignment, bool *zero, bool *commit,
                                    unsigned arena) {
    (void)new_addr; (void)arena;
    dram_cap_tier_t *t = (dram_cap_tier_t *)hooks;  /* hooks == &t->hooks == t */
    if (!t->region_base || t->region_cap == 0) return NULL;

    size_t align = alignment ? alignment : 1;
    for (;;) {
        size_t cur     = atomic_load_explicit(&t->region_off, memory_order_relaxed);
        size_t aligned = (cur + align - 1) & ~(align - 1);
        if (aligned < cur) return NULL;               /* alignment overflow */
        if (aligned + size < aligned) return NULL;     /* size overflow */
        size_t end = aligned + size;
        if (end > t->region_cap) {
            fprintf(stderr,
                    "dram_cap: DRAM hard cap exhausted (need %zu, cap %zu)\n",
                    end, t->region_cap);
            return NULL;
        }
        if (atomic_compare_exchange_weak_explicit(
                &t->region_off, &cur, end,
                memory_order_seq_cst, memory_order_relaxed)) {
            *zero   = true;   /* fresh anon pages zero on first fault */
            *commit = true;
            return (char *)t->region_base + aligned;
        }
        /* CAS lost a race -- retry. */
    }
}

static bool dram_cap_extent_dalloc(extent_hooks_t *h, void *a, size_t s,
                                   bool c, unsigned ar) {
    (void)h;(void)a;(void)s;(void)c;(void)ar;
    return true;    /* bump region: retain, bulk reclaim at process exit */
}

static void dram_cap_extent_destroy(extent_hooks_t *h, void *a, size_t s,
                                    bool c, unsigned ar) {
    (void)h;(void)a;(void)s;(void)c;(void)ar;
}

static bool dram_cap_extent_commit(extent_hooks_t *h, void *a, size_t s,
                                   size_t o, size_t l, unsigned ar) {
    (void)h;(void)a;(void)s;(void)o;(void)l;(void)ar;
    return false;   /* already committed */
}

static bool dram_cap_extent_decommit(extent_hooks_t *h, void *a, size_t s,
                                     size_t o, size_t l, unsigned ar) {
    (void)h;(void)a;(void)s;(void)o;(void)l;(void)ar;
    return true;    /* refuse -- keep pages */
}

static bool dram_cap_extent_purge_lazy(extent_hooks_t *h, void *a, size_t s,
                                       size_t o, size_t l, unsigned ar) {
    (void)h;(void)a;(void)s;(void)o;(void)l;(void)ar;
    return true;    /* refuse -- never MADV_DONTNEED our pages */
}

static bool dram_cap_extent_purge_forced(extent_hooks_t *h, void *a, size_t s,
                                         size_t o, size_t l, unsigned ar) {
    (void)h;(void)a;(void)s;(void)o;(void)l;(void)ar;
    return true;
}

static bool dram_cap_extent_split(extent_hooks_t *h, void *a, size_t s,
                                  size_t sa, size_t sb, bool c, unsigned ar) {
    (void)h;(void)a;(void)s;(void)sa;(void)sb;(void)c;(void)ar;
    return false;   /* one contiguous VMA -- split is free */
}

static bool dram_cap_extent_merge(extent_hooks_t *h, void *a, size_t sa,
                                  void *b, size_t sb, bool c, unsigned ar) {
    (void)h;(void)a;(void)sa;(void)b;(void)sb;(void)c;(void)ar;
    return false;
}

static void dram_cap_init_hooks(dram_cap_tier_t *t) {
    t->hooks.alloc        = dram_cap_extent_alloc;
    t->hooks.dalloc       = dram_cap_extent_dalloc;
    t->hooks.destroy      = dram_cap_extent_destroy;
    t->hooks.commit       = dram_cap_extent_commit;
    t->hooks.decommit     = dram_cap_extent_decommit;
    t->hooks.purge_lazy   = dram_cap_extent_purge_lazy;
    t->hooks.purge_forced = dram_cap_extent_purge_forced;
    t->hooks.split        = dram_cap_extent_split;
    t->hooks.merge        = dram_cap_extent_merge;
}

static void dram_cap_bind_region(void *addr, size_t size) {
    struct bitmask *mask = numa_bitmask_alloc(numa_num_possible_nodes());
    if (!mask) return;
    numa_bitmask_setbit(mask, 0); /* DRAM NUMA node -- node 0 by definition here */
    /* Policy only (flags 0): nothing is faulted yet, so no MOVE needed;
     * faults land on node 0 afterward. */
    if (mbind(addr, size, MPOL_BIND, mask->maskp, mask->size + 1, 0) != 0) {
        fprintf(stderr, "dram_cap: mbind(%zu) failed: %s\n", size, strerror(errno));
    }
    numa_bitmask_free(mask);
}

/* Reads PAPER_CACHE_DRAM_CAP_BYTES and, if set to a nonzero value, tries to
 * reserve the region and install the capped arena. Returns:
 *   0 -- cap successfully activated
 *   1 -- no cap configured, OR a cap was requested but could not be set up
 *        for any reason; caller should fall back to the existing UMF
 *        jemalloc pool path for node 0
 *
 * Every failure path below falls back (returns 1) rather than a distinct
 * "real error" code. An earlier version returned a distinct nonzero code
 * per failure, which `umf_allocator_init` treated as fatal -- skipping
 * `umf_pool_init` entirely and leaving `pools[0]` permanently NULL. That
 * silently broke *all* node-0 (DRAM) allocation for the rest of the
 * process, including the very first one -- which, combined with a then-
 * separate, pre-existing bug (`allocator.rs` using `println!`, which lazily
 * allocates its own stdout buffer on first use, inside the alloc-failure
 * diagnostic of `DRAMObjects::alloc` itself), produced a genuine self-
 * deadlock: the first-ever allocation failed, tried to print via `println!`,
 * which needed to allocate to set up stdout's buffer, which failed and
 * tried to print again, recursing into the same uninitialized `OnceLock`
 * from the same thread forever (confirmed via `gdb` on a hung process that
 * a user reported running for many hours). Falling back here instead means
 * an unusable cap (for whatever reason) degrades to "the feature quietly
 * doesn't activate," not "the allocator stops working."
 */
static int dram_cap_init_from_env(void) {
    const char *env = getenv("PAPER_CACHE_DRAM_CAP_BYTES");
    if (!env || env[0] == '\0') return 1;

    unsigned long long requested = strtoull(env, NULL, 10);
    if (requested == 0) return 1;
    size_t bytes = (size_t)requested;

    // Only reached (and libjemalloc.so.2 only ever dlopen'd) once the env
    // var is confirmed set -- see the dlopen/dlsym note above this function.
    // Confirmed to genuinely fail on at least one real system: this
    // sandbox's libjemalloc.so.2 cannot be dlopen'd after process start at
    // all ("cannot allocate memory in static TLS block" -- a real glibc
    // limit on the small TLS surplus reserved for post-startup dlopen of
    // libraries using the initial-exec TLS model, which jemalloc's own
    // thread-local caches use). Falling back (not aborting) here is the
    // correct behavior for that case; see this function's remediation
    // suggestions in the printed message below for how to actually get the
    // cap working on such a system.
    if (!dram_cap_load_jemalloc()) {
        fprintf(stderr,
                "dram_cap: could not load libjemalloc.so.2 for the DRAM hard "
                "cap (PAPER_CACHE_DRAM_CAP_BYTES=%s) -- falling back to "
                "normal (uncapped) allocation for node 0. If this is a "
                "\"cannot allocate memory in static TLS block\" error, your "
                "system's libjemalloc.so.2 cannot be dlopen'd after process "
                "start; try either `LD_PRELOAD=libjemalloc.so.2` (loads it "
                "at startup instead, avoiding the static-TLS limit -- note "
                "this also makes jemalloc your process-wide malloc) or "
                "`GLIBC_TUNABLES=glibc.rtld.optional_static_tls=<bytes>` "
                "(increases the reserved dlopen-time TLS surplus, glibc "
                ">= 2.35) before launching.\n",
                env);
        return 1;
    }

    dram_cap_init_hooks(&dram_cap_tier);

    void *p = mmap(NULL, bytes, PROT_READ | PROT_WRITE,
                   MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
    if (p == MAP_FAILED) {
        fprintf(stderr, "dram_cap: mmap(%zu) failed: %s -- falling back to "
                         "normal (uncapped) allocation for node 0\n",
                bytes, strerror(errno));
        return 1;
    }
    dram_cap_bind_region(p, bytes);
    dram_cap_tier.region_base = p;
    dram_cap_tier.region_cap  = bytes;
    atomic_store_explicit(&dram_cap_tier.region_off, 0, memory_order_release);

    unsigned ind;
    size_t ind_sz = sizeof(ind);
    if (dram_cap_mallctl("arenas.create", &ind, &ind_sz, NULL, 0) != 0) {
        fprintf(stderr, "dram_cap: arenas.create failed -- falling back to "
                         "normal (uncapped) allocation for node 0\n");
        munmap(p, bytes);
        dram_cap_tier.region_base = NULL;
        dram_cap_tier.region_cap  = 0;
        return 1;
    }

    char path[64];
    snprintf(path, sizeof(path), "arena.%u.extent_hooks", ind);
    extent_hooks_t *hooks_ptr = &dram_cap_tier.hooks;
    if (dram_cap_mallctl(path, NULL, NULL, &hooks_ptr, sizeof(hooks_ptr)) != 0) {
        fprintf(stderr, "dram_cap: setting extent_hooks on arena %u failed -- "
                         "falling back to normal (uncapped) allocation for "
                         "node 0\n", ind);
        snprintf(path, sizeof(path), "arena.%u.destroy", ind);
        dram_cap_mallctl(path, NULL, NULL, NULL, 0);
        munmap(p, bytes);
        dram_cap_tier.region_base = NULL;
        dram_cap_tier.region_cap  = 0;
        return 1;
    }

    /* No dedicated tcache: DRAMObjects is the crate's #[global_allocator],
     * called concurrently from every OS thread in the process. A tcache
     * (thread cache) is not safe to share across threads -- its bin
     * freelists are plain linked lists with no internal locking, only
     * correct when each tcache is used by a single thread at a time. A
     * first version of this code created ONE tcache per tier via
     * `tcache.create` and reused its index (`MALLOCX_TCACHE(tc)`) from every
     * calling thread; under concurrent load this corrupted jemalloc's own
     * internal extent-heap bookkeeping and crashed inside
     * je_tcache_bin_flush_small -> je_extent_heap_remove (confirmed via gdb
     * on a real SIGSEGV during the 200K-object repro test). Every alloc/
     * dealloc below instead goes straight to the arena's own bins via
     * MALLOCX_TCACHE_NONE, which jemalloc already protects with a proper
     * per-arena-bin lock (slower than a tcache hit, but correct). */
    atomic_store_explicit(&dram_cap_tier.arena_ind, ind, memory_order_release);
    atomic_store_explicit(&dram_cap_active, true, memory_order_release);

    fprintf(stderr,
            "dram_cap: DRAM hard cap active -- %zu bytes reserved at %p "
            "(PAPER_CACHE_DRAM_CAP_BYTES=%s)\n",
            bytes, p, env);
    return 0;
}

static bool dram_cap_is_active(void) {
    return atomic_load_explicit(&dram_cap_active, memory_order_acquire);
}

static bool dram_cap_owns(void *ptr) {
    if (!dram_cap_tier.region_base) return false;
    uintptr_t p    = (uintptr_t)ptr;
    uintptr_t base = (uintptr_t)dram_cap_tier.region_base;
    return p >= base && p < base + dram_cap_tier.region_cap;
}

static void *dram_cap_alloc(size_t size, size_t align) {
    if (size == 0) return NULL;
    unsigned ind = atomic_load_explicit(&dram_cap_tier.arena_ind, memory_order_acquire);
    if (ind == UINT_MAX) return NULL;

    int flags = MALLOCX_ARENA(ind) | MALLOCX_TCACHE_NONE;
    if (align && align > sizeof(void *)) flags |= MALLOCX_ALIGN(align);
    return dram_cap_mallocx(size, flags);
}

static void dram_cap_dealloc(void *ptr) {
    if (!ptr) return;
    unsigned ind = atomic_load_explicit(&dram_cap_tier.arena_ind, memory_order_acquire);
    if (ind == UINT_MAX) return;
    dram_cap_dallocx(ptr, MALLOCX_ARENA(ind) | MALLOCX_TCACHE_NONE);
}


// numa_node = NUMA node id (check with numactl -H)

static int umf_pool_init(int numa_node);

int umf_allocator_init(int numa_node) {
    // dram_cap_init_from_env() only ever returns 0 (cap activated) or 1
    // (no cap configured, or one was requested but couldn't be set up --
    // see its doc comment for why every failure there falls back rather
    // than returning a distinct "real error" code). Either way, rc == 1
    // means: fall through to the existing UMF jemalloc pool path below,
    // unchanged.
    if (numa_node == 0 && dram_cap_init_from_env() == 0) {
        return 0;
    }
    return umf_pool_init(numa_node);
}

static int umf_pool_init(int numa_node) {
    //setenv("UMF_CONF", "umf.provider.os.params.mmap_flags=0x8000", 0);
    umf_memory_pool_handle_t new_pool = NULL;
    umf_scalable_pool_params_handle_t scalable_params = NULL;
    umf_result_t res;

    if (numa_node < 0 || numa_node >= MAX_NODES) {
        fprintf(stderr, "umf_allocator_init: numa_node %d out of range [0,%d)\n",
                numa_node, MAX_NODES);
        return -1;
    }

    pthread_mutex_lock(&lifecycle_lock);

    if (atomic_load_explicit(&pools[numa_node], memory_order_acquire) != NULL) {
        pthread_mutex_unlock(&lifecycle_lock);
        return 0;
    }

    // Create OS provider params
    res = umfOsMemoryProviderParamsCreate(&os_params_arr[numa_node]);
    if (res != UMF_RESULT_SUCCESS) {
        fprintf(stderr, "Failed to create OS params (node %d): %d\n", numa_node, res);
        pthread_mutex_unlock(&lifecycle_lock);
        return 1;
    }

    // Set NUMA node list
    unsigned numa_list[] = { (unsigned)numa_node };

    res = umfOsMemoryProviderParamsSetNumaList(os_params_arr[numa_node], numa_list, 1);
    if (res != UMF_RESULT_SUCCESS) {
        fprintf(stderr, "Failed to set NUMA list (node %d): %d\n", numa_node, res);
        pthread_mutex_unlock(&lifecycle_lock);
        return 2;
    }

    // Bind strictly to that NUMA node
    res = umfOsMemoryProviderParamsSetNumaMode(os_params_arr[numa_node], UMF_NUMA_MODE_BIND);
    if (res != UMF_RESULT_SUCCESS) {
        fprintf(stderr, "Failed to set NUMA mode (node %d): %d\n", numa_node, res);
        pthread_mutex_unlock(&lifecycle_lock);
        return 3;
    }

    // Create provider
    res = umfMemoryProviderCreate(
            umfOsMemoryProviderOps(),
            os_params_arr[numa_node],
            &providers[numa_node]);
    if (res != UMF_RESULT_SUCCESS) {
        fprintf(stderr, "Failed to create OS provider (node %d): %d\n", numa_node, res);
        pthread_mutex_unlock(&lifecycle_lock);
        return 4;
    }

    // -------------------------------------------------------------------------
    // REVERTED back to the TBB scalable pool (from UMF's jemalloc-backed
    // pool, which this code briefly switched to -- see CLAUDE.md's "real
    // DRAM usage vs. fast_tier_size" investigation for that swap's
    // measured DRAM-retention win). The jemalloc pool was found, via a real
    // benchmark run under paper-benchmark-cxl (multiple concurrent rayon
    // worker threads, real trace-driven load), to SEGFAULT reproducibly
    // inside UMF's own internals: confirmed via gdb -- jemalloc's extent
    // splitting (inside umfJemallocPoolOps(), needed as arenas grow/shrink
    // under concurrent allocation pressure) invokes UMF's custom
    // arena_extent_split extent hook, which crashes inside UMF's own
    // critnib-based memory tracker (umfMemoryTrackerAddAtLevel ->
    // critnib_insert -> add_metadata_and_align, all in libumf.so.1.0.3,
    // UMF version 1.0.3). This is a bug inside UMF's own prebuilt library,
    // not in this crate's code -- not something fixable here. A crash is a
    // strictly worse outcome than elevated-but-bounded DRAM usage, so this
    // reverts to the previously-proven-stable TBB pool (with
    // KeepAllMemory=1, as before the jemalloc-pool experiment) until either
    // UMF fixes this upstream or a jemalloc-pool configuration is found
    // that avoids triggering the crashing code path under real concurrent
    // load. The DRAM hard cap (dram_cap_* functions above, opt-in via
    // PAPER_CACHE_DRAM_CAP_BYTES) does NOT go through this pool at all for
    // node 0 -- it is unaffected by this revert.
    // -------------------------------------------------------------------------
    res = umfScalablePoolParamsCreate(&scalable_params);
    if (res != UMF_RESULT_SUCCESS) {
        fprintf(stderr, "Failed to create scalable pool params (node %d): %d\n", numa_node, res);
        pthread_mutex_unlock(&lifecycle_lock);
        return 5;
    }

    // Keep retaining freed blocks so transient allocs recycle in-pool and the
    // provider's bump offset advances slowly (only net-new live memory grows it).
    umfScalablePoolParamsSetKeepAllMemory(scalable_params, 1);

    size_t huge_chunk_size = 2 * 1024 * 1024ULL;
    umfScalablePoolParamsSetGranularity(scalable_params, huge_chunk_size);

    // Create pool into a local first; only publish once fully constructed.
    res = umfPoolCreate(
            umfScalablePoolOps(),
            providers[numa_node],
            scalable_params,
            0,
            &new_pool);

    umfScalablePoolParamsDestroy(scalable_params);

    if (res != UMF_RESULT_SUCCESS) {
        fprintf(stderr, "Failed to create pool (node %d): %d\n", numa_node, res);
        pthread_mutex_unlock(&lifecycle_lock);
        return 6;
    }

    //Release-store publishes the fully-initialized pool. Any thread that
    //later observes a non-NULL pool via acquire-load is guaranteed to see
    //a consistent pool object. 
    atomic_store_explicit(&pools[numa_node], (uintptr_t)new_pool, memory_order_release);

    pthread_mutex_unlock(&lifecycle_lock);

     //Do NOT register umf_allocator_finalize via atexit.
     //Under large cache loads, background PolicyWorker and reinsertion threads
     //continue to alloc/dealloc PMEM buffers via HybridObjects after main()
     //returns.  An atexit handler would destroy the UMF pool while those
     //threads are still active, causing [FATAL UMF] assertion failures and
     //"memory allocation of N bytes failed" panics in Rust.
     //The OS reclaims all virtual memory (including UMF-managed PMEM pages)
     //when the process exits, so explicit pool teardown is unnecessary.
     
    return 0;
}





/*

extern const umf_memory_provider_ops_t *umfPrefaultProviderOps(void);
typedef struct prefault_params_t { size_t size; int numa_node; } prefault_params_t;
 
// size the prefaulted region per node; tune to your largest trace
#define PREFAULT_BYTES (45ULL * 1024 * 1024 * 1024) 
 
int umf_allocator_init(int numa_node) {
    umf_memory_pool_handle_t new_pool = NULL;
    umf_scalable_pool_params_handle_t scalable_params = NULL;
    umf_result_t res;
 
    if (numa_node < 0 || numa_node >= MAX_NODES) {
        fprintf(stderr, "umf_allocator_init: numa_node %d out of range\n", numa_node);
        return -1;
    }
 
    pthread_mutex_lock(&lifecycle_lock);
 
    if (atomic_load_explicit(&pools[numa_node], memory_order_acquire) != NULL) {
        pthread_mutex_unlock(&lifecycle_lock);
        return 0;
    }
 
    //---- PROVIDER: prefault provider instead of OS provider ---- 
    prefault_params_t pf_params = {
        .size      = PREFAULT_BYTES,
        .numa_node = numa_node,
    };
 
    res = umfMemoryProviderCreate(
            umfPrefaultProviderOps(),
            &pf_params,
            &providers[numa_node]);
    if (res != UMF_RESULT_SUCCESS) {
        fprintf(stderr, "Failed to create prefault provider (node %d): %d\n",
                numa_node, res);
        pthread_mutex_unlock(&lifecycle_lock);
        return 4;
    }
    //NOTE: os_params_arr umfOsMemoryProviderParams* no longer used on this
    // path — the prefault provider takes its config via pf_params above. 
 
    //---- POOL: unchanged scalable pool on top ----
    res = umfScalablePoolParamsCreate(&scalable_params);
    if (res != UMF_RESULT_SUCCESS) {
        fprintf(stderr, "Failed to create scalable pool params (node %d): %d\n",
                numa_node, res);
        pthread_mutex_unlock(&lifecycle_lock);
        return 5;
    }
 
    // keep retaining freed blocks so transient allocs recycle in-pool and the
    // provider's bump offset advances slowly (only net-new live memory grows it)
    umfScalablePoolParamsSetKeepAllMemory(scalable_params, 1);
 
    size_t huge_chunk_size = 2 * 1024 * 1024ULL;
    umfScalablePoolParamsSetGranularity(scalable_params, huge_chunk_size);
 
    res = umfPoolCreate(
            umfScalablePoolOps(),
            providers[numa_node],
            scalable_params,
            0,
            &new_pool);
 
    umfScalablePoolParamsDestroy(scalable_params);
 
    if (res != UMF_RESULT_SUCCESS) {
        fprintf(stderr, "Failed to create pool (node %d): %d\n", numa_node, res);
        pthread_mutex_unlock(&lifecycle_lock);
        return 6;
    }
 
    atomic_store_explicit(&pools[numa_node], (uintptr_t)new_pool, memory_order_release);
    pthread_mutex_unlock(&lifecycle_lock);
    return 0;
}





*/
    


void *umf_alloc(int numa_node, size_t size, size_t align) {
    if (size == 0) return NULL;
    if (numa_node < 0 || numa_node >= MAX_NODES) return NULL;

    if (numa_node == 0 && dram_cap_is_active()) {
        return dram_cap_alloc(size, align);
    }

    /* Lock-free fast path. The scalable pool (TBB) handles its own thread
     * safety via per-thread caches; wrapping every alloc in a global mutex
     * would serialize the entire process and destroy scalability. */
    umf_memory_pool_handle_t p = (umf_memory_pool_handle_t)
        atomic_load_explicit(&pools[numa_node], memory_order_acquire);
    if (!p) return NULL;

    if (align && align > sizeof(void*)) {
        return umfPoolAlignedMalloc(p, size, align);
    }
    return umfPoolMalloc(p, size);
}


void umf_dealloc(int numa_node, void *ptr) {
    if (!ptr) return;
    if (numa_node < 0 || numa_node >= MAX_NODES) return;

    if (numa_node == 0 && dram_cap_is_active()) {
        dram_cap_dealloc(ptr);
        return;
    }

    umf_memory_pool_handle_t p = (umf_memory_pool_handle_t)
        atomic_load_explicit(&pools[numa_node], memory_order_acquire);
    if (!p) return;  /* pool already destroyed; OS will reclaim on exit */

    umfPoolFree(p, ptr);
}


/* Returns the NUMA node id that owns this pointer, or -1 if the pointer
 * is not managed by any of our UMF pools (or, when active, the DRAM hard
 * cap's own region).
 *
 * NOTE: return semantics changed. The old version returned 1=pmem / 0=dram.
 * Callers that used the return as a bool must be updated. Use the node id
 * directly, or compare against a known value (e.g. `check_tier(p) == 1`).
 */
int check_tier(void *ptr) {
    if (dram_cap_is_active() && dram_cap_owns(ptr)) {
        return 0;
    }

    umf_memory_pool_handle_t curr_pool;
    if (umfPoolByPtr(ptr, &curr_pool) != UMF_RESULT_SUCCESS) {
        return -1;
    }
    for (int i = 0; i < MAX_NODES; i++) {
        umf_memory_pool_handle_t our_pool = (umf_memory_pool_handle_t)
            atomic_load_explicit(&pools[i], memory_order_acquire);
        if (our_pool != NULL && our_pool == curr_pool) {
            return i;
        }
    }
    return -1;
}


/* Prewarm the UMF pool for `numa_node` by allocating `bytes` worth of memory
 * through it in `chunk`-sized pieces, touching every page so the OS provider
 * faults + zeroes them on the target NUMA node now, then freeing back into
 * the pool.
 *
 * Whether the prewarm "sticks" depends on the pool retaining freed memory
 * rather than handing it back to the OS provider. The scalable pool (TBB)
 * retains aggressively by default — freed blocks stay in the pool's per-
 * thread caches and superblock free-lists.
 *
 * Call AFTER umf_allocator_init, BEFORE the measured workload.
 * Returns 0 on success. */


/*int umf_allocator_prewarm(int numa_node, size_t bytes, size_t chunk) {
    if (bytes == 0) return 0;
    if (chunk == 0) chunk = 4096;
    if (numa_node < 0 || numa_node >= MAX_NODES) {
        fprintf(stderr, "umf_allocator_prewarm: numa_node %d out of range\n", numa_node);
        return 1;
    }

    umf_memory_pool_handle_t p = (umf_memory_pool_handle_t)
        atomic_load_explicit(&pools[numa_node], memory_order_acquire);
    if (!p) {
        fprintf(stderr, "umf_allocator_prewarm: pool not initialized (node %d)\n", numa_node);
        return 1;
    }

    long pg = sysconf(_SC_PAGESIZE);
    if (pg <= 0) pg = 4096;

    size_t n = (bytes + chunk - 1) / chunk;
    void **ptrs = calloc(n, sizeof(void *));
    if (!ptrs) {
        fprintf(stderr, "umf_allocator_prewarm: out of host memory\n");
        return 2;
    }

    size_t got = 0;
    for (size_t i = 0; i < n; i++) {
        void *blk = umfPoolMalloc(p, chunk);
        if (!blk) {
            fprintf(stderr,
                    "umf_allocator_prewarm (node %d): pool exhausted at %zu/%zu chunks "
                    "(%zu bytes requested)\n",
                    numa_node, got, n, chunk);
            for (size_t j = 0; j < got; j++) umfPoolFree(p, ptrs[j]);
            free(ptrs);
            return 3;
        }

        // Touch one byte per page to force the fault. volatile so the
        // store is not elided by the compiler.
        volatile char *vp = (volatile char *)blk;
        for (size_t off = 0; off < chunk; off += (size_t)pg) {
            vp[off] = 0;
        }
        ptrs[got++] = blk;
    }

    fprintf(stderr,
            "umf_allocator_prewarm (node %d): touched %zu chunks x %zu bytes = %zu MiB\n",
            numa_node, got, chunk, (got * chunk) >> 20);

    //Free everything back into the pool. The scalable pool retains these
    //blocks for fast reuse; the OS provider does not unmap, so the pages
    // stay mapped, faulted, and bound to the target NUMA node. 
    for (size_t i = 0; i < got; i++) umfPoolFree(p, ptrs[i]);
    free(ptrs);
    return 0;
}

*/





int umf_allocator_prewarm(int numa_node, size_t bytes, size_t chunk) {
    if (bytes == 0) return 0;
    if (chunk == 0) chunk = 2 * 1024 * 1024;
    if (numa_node < 0 || numa_node >= MAX_NODES) {
        fprintf(stderr, "umf_allocator_prewarm: numa_node %d out of range\n", numa_node);
        return 1;
    }

    umf_memory_pool_handle_t p = (umf_memory_pool_handle_t)
        atomic_load_explicit(&pools[numa_node], memory_order_acquire);
    if (!p) {
        fprintf(stderr, "umf_allocator_prewarm: pool not initialized (node %d)\n", numa_node);
        return 1;
    }

    long pg = sysconf(_SC_PAGESIZE);
    if (pg <= 0) pg = 4096;

    size_t n = (bytes + chunk - 1) / chunk;
    void **ptrs = calloc(n, sizeof(void *));
    if (!ptrs) {
        fprintf(stderr, "umf_allocator_prewarm: out of host memory\n");
        return 2;
    }

    size_t got = 0;
    for (size_t i = 0; i < n; i++) {
        void *blk = umfPoolMalloc(p, chunk);
        if (!blk) {
            fprintf(stderr,
                    "umf_allocator_prewarm (node %d): pool exhausted at %zu/%zu chunks "
                    "(%zu bytes requested)\n",
                    numa_node, got, n, chunk);
            for (size_t j = 0; j < got; j++) umfPoolFree(p, ptrs[j]);
            free(ptrs);
            return 3;
        }

        /* Touch one byte per page to force the fault. volatile so the
         * store is not elided by the compiler. */
        volatile char *vp = (volatile char *)blk;
        for (size_t off = 0; off < chunk; off += (size_t)pg) {
            vp[off] = 0;
        }
        ptrs[got++] = blk;
    }

    fprintf(stderr,
            "umf_allocator_prewarm (node %d): touched %zu chunks x %zu bytes = %zu MiB\n",
            numa_node, got, chunk, (got * chunk) >> 20);

    /* Free everything back into the pool. The scalable pool retains these
     * blocks for fast reuse; the OS provider does not unmap, so the pages
     * stay mapped, faulted, and bound to the target NUMA node. */
    for (size_t i = 0; i < got; i++) umfPoolFree(p, ptrs[i]);
    free(ptrs);
    return 0;
}









#include <umf/providers/provider_devdax_memory.h>
//#include <umf/pools/pool_jemalloc.h>



//static umf_memory_provider_handle_t dax_provider = NULL;
static umf_devdax_memory_provider_params_handle_t dax_params = NULL;

/*
void umf_allocator_finalize_dax(void) {
    pools[5] = NULL; //pool 5 is our DAX pool
    if (pools[5]) {
        umfPoolDestroy(pools[5]);
        pools[5] = NULL;
    }
    if (providers[5]) {
        umfMemoryProviderDestroy(providers[5]);
        providers[5] = NULL;
    }
    if (dax_params) {
        umfDevDaxMemoryProviderParamsDestroy(dax_params);
        dax_params = NULL;
    }
}
    */


    /*
int umf_allocator_init_dax(const char *dax_path, size_t dax_size) {
    umf_memory_pool_handle_t new_pool = NULL;
    umf_jemalloc_pool_params_handle_t jemalloc_params = NULL;

    umf_result_t res;

    pthread_mutex_lock(&lifecycle_lock);

    if (atomic_load_explicit(&pools[5], memory_order_acquire) != NULL) {
        pthread_mutex_unlock(&lifecycle_lock);
        return 0;
    }

    res = umfDevDaxMemoryProviderParamsCreate(dax_path, dax_size, &dax_params);
    if (res != UMF_RESULT_SUCCESS) {
        fprintf(stderr, "Failed to create DAX params: %d\n", res);
        return 1;
    }

    res = umfMemoryProviderCreate(umfDevDaxMemoryProviderOps(), dax_params, &providers[5]);
    if (res != UMF_RESULT_SUCCESS) {
        fprintf(stderr, "Failed to create DAX provider: %d\n", res);
        return 2;
    }

    res = umfJemallocPoolParamsCreate(&jemalloc_params);
    if (res != UMF_RESULT_SUCCESS) {
        fprintf(stderr, "Failed to create jemalloc pool params: %d\n", res);
        return 3;
    }

    res = umfPoolCreate(umfJemallocPoolOps(), providers[5], jemalloc_params, 0, &pools[5]);
    umfJemallocPoolParamsDestroy(jemalloc_params);

    if (res != UMF_RESULT_SUCCESS) {
        fprintf(stderr, "Failed to create memory pool: %d\n", res);
        return 4;
    }

    atomic_store_explicit(&pools[5], (uintptr_t)new_pool, memory_order_release);


    // Zero all memory in the pool
    // in case of persistence.. dont think it matters for devdax tho.. 
    //size_t pool_size = dax_size;
    //void *base = umfPoolMalloc(pools[5], pool_size);
    //if (base) {
        //memset(base, 0, pool_size);
       // umfPoolFree(pools[5], base);
    //}
    //printf("base pointer init %p\n", base);

   // if (res != UMF_RESULT_SUCCESS) {
        //fprintf(stderr, "Failed to create pool (node %d): %d\n", 5, res);
        //pthread_mutex_unlock(&lifecycle_lock);
        //return 6;
    //}

    //Release-store publishes the fully-initialized pool. Any thread that
    //later observes a non-NULL pool via acquire-load is guaranteed to see
    //a consistent pool object. 

    pthread_mutex_unlock(&lifecycle_lock);

    //atexit(umf_allocator_finalize_dax);
    return 0;
}
*/

/*
void *return_pmem_base_dax(size_t dax_size) {
    void *base = umfPoolMalloc(pool, dax_size);
    
    //if (base) {
    //    memset(base, 0, dax_size);
    //    umfPoolFree(pool, base);
    //}
    //printf("base pointer pmem %p\n", base);
    return base; //this should be the base address of the mapped PMEM region
}*/



int umf_allocator_init_dax(const char *dax_path, size_t dax_size) {
    umf_memory_pool_handle_t new_pool = NULL;
    umf_scalable_pool_params_handle_t scalable_params = NULL;
    umf_result_t res;

    if (dax_path == NULL || dax_size == 0) {
        fprintf(stderr, "umf_allocator_init_dax: bad args (path=%p size=%zu)\n",
                (void *)dax_path, dax_size);
        return -1;
    }

    pthread_mutex_lock(&lifecycle_lock);

    if (atomic_load_explicit(&pools[5], memory_order_acquire) != NULL) {
        pthread_mutex_unlock(&lifecycle_lock);
        return 0;
    }

    res = umfDevDaxMemoryProviderParamsCreate(dax_path, dax_size, &dax_params);
    if (res != UMF_RESULT_SUCCESS) {
        fprintf(stderr, "Failed to create DAX params: %d\n", res);
        pthread_mutex_unlock(&lifecycle_lock);
        return 1;
    }

    res = umfMemoryProviderCreate(umfDevDaxMemoryProviderOps(), dax_params, &providers[5]);
    if (res != UMF_RESULT_SUCCESS) {
        fprintf(stderr, "Failed to create DAX provider: %d\n", res);
        umfDevDaxMemoryProviderParamsDestroy(dax_params);
        dax_params = NULL;
        pthread_mutex_unlock(&lifecycle_lock);
        return 2;
    }


    int fd = open(dax_path, O_RDWR);
    if (fd < 0) {
        perror("prefault: open dax");
    } else {
        void *m = mmap(NULL, dax_size, PROT_READ | PROT_WRITE,
                       MAP_SHARED | MAP_POPULATE, fd, 0);
        if (m == MAP_FAILED) {
            perror("prefault: mmap");
        } else {
            memset(m, 0, dax_size);   // force the write fault on every page
            munmap(m, dax_size);
        }
        close(fd);
    }


    //res = umfJemallocPoolParamsCreate(&jemalloc_params);
    res = umfScalablePoolParamsCreate(&scalable_params);
    if (res != UMF_RESULT_SUCCESS) {
        fprintf(stderr, "Failed to create jemalloc pool params: %d\n", res);
        umfMemoryProviderDestroy(providers[5]);
        providers[5] = NULL;
        umfDevDaxMemoryProviderParamsDestroy(dax_params);
        dax_params = NULL;
        pthread_mutex_unlock(&lifecycle_lock);
        return 3;
    }

   // res = umfPoolCreate(umfJemallocPoolOps(), providers[5], jemalloc_params, 0, &new_pool);
    //umfJemallocPoolParamsDestroy(jemalloc_params);

    res = umfPoolCreate(umfScalablePoolOps(), providers[5], scalable_params, 0, &new_pool);
    umfScalablePoolParamsDestroy(scalable_params);

    if (res != UMF_RESULT_SUCCESS) {
        fprintf(stderr, "Failed to create memory pool: %d\n", res);
        umfMemoryProviderDestroy(providers[5]);
        providers[5] = NULL;
        umfDevDaxMemoryProviderParamsDestroy(dax_params);
        dax_params = NULL;
        pthread_mutex_unlock(&lifecycle_lock);
        return 4;
    }

    atomic_store_explicit(&pools[5], (uintptr_t)new_pool, memory_order_release);

    pthread_mutex_unlock(&lifecycle_lock);
    return 0;
}

void *umf_alloc_dax(size_t size, size_t align) {

    umf_memory_pool_handle_t p = (umf_memory_pool_handle_t)
        atomic_load_explicit(&pools[5], memory_order_acquire);
    if (!p) return NULL;
    //pthread_mutex_lock(&pool_lock);
    //void *ptr = umfPoolMalloc(pool, size); //might want to use the aligned version

    //respect alignment.... although jemalloc should do this for us...........
    void *ptr = umfPoolAlignedMalloc(p, size, align);

    //pthread_mutex_unlock(&pool_lock);
    return ptr;
}

void umf_dealloc_dax(void *ptr) {
    //pthread_mutex_lock(&pool_lock);
    umf_memory_pool_handle_t p = (umf_memory_pool_handle_t)
        atomic_load_explicit(&pools[5], memory_order_acquire);
    if (!p) return;  /* pool already destroyed; OS will reclaim on exit */
    umfPoolFree(p, ptr);
    //pthread_mutex_unlock(&pool_lock);
}

int check_tier_dax(void *ptr) {
    umf_memory_pool_handle_t curr_pool;
    if (umfPoolByPtr(ptr, &curr_pool) == UMF_RESULT_SUCCESS) {

        if (curr_pool == pools[5]) {
            return 1; //pmem
        }
    }
    else {
        return 0; //dram
    }
    //tjhis is unreachabke thoo
    return -1; //not from any UMF pool
}

