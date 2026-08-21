/* RTLD_DEFAULT (used by umf_clean_all_buffers) is a GNU extension. */
#define _GNU_SOURCE



#include <pthread.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <dlfcn.h>
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

#define MAX_NODES 8

// Per-node state. Index by NUMA node id.
static atomic_uintptr_t pools[MAX_NODES];
static umf_memory_provider_handle_t providers[MAX_NODES];
static umf_os_memory_provider_params_handle_t os_params_arr[MAX_NODES];
static pthread_mutex_t lifecycle_lock = PTHREAD_MUTEX_INITIALIZER;

// numa_node = NUMA node id (check with numactl -H)

int umf_allocator_init(int numa_node) {
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
    // umfJemallocPoolOps() (UMF's jemalloc-backed pool) is a KNOWN, REPEATEDLY
    // CONFIRMED source of instability under this crate's real concurrent
    // load (paper-benchmark-cxl, multiple concurrent rayon worker threads,
    // a real trace) -- tested twice, failed both times, two different ways:
    // (1) a reproducible SIGSEGV inside UMF's own internals, confirmed via
    // gdb -- jemalloc's routine extent-splitting invokes UMF's own
    // arena_extent_split extent hook, which crashes inside UMF's own
    // critnib-based memory tracker (umfMemoryTrackerAddAtLevel ->
    // critnib_insert -> add_metadata_and_align, libumf.so.1.0.3); (2) on a
    // second, independent test run, a corrupted-looking allocation-failure
    // abort (garbled/interleaved diagnostic strings) with 160 GB of system
    // memory still free, ruling out genuine exhaustion -- consistent with
    // heap corruption from the same underlying concurrency bug, manifesting
    // differently that run. Both are bugs inside UMF's own prebuilt
    // library, not in this crate's code, and not fixable here -- UMF's
    // jemalloc pool params expose no option that avoids the crashing path
    // (only umfJemallocPoolParamsSetNumArenas, which doesn't touch it).
    // Reverted to the TBB scalable pool (KeepAllMemory=1) -- the only
    // configuration that has actually completed this benchmark successfully
    // end-to-end, more than once. See CLAUDE.md's "real DRAM usage vs.
    // fast_tier_size" investigation for the full incident writeup and data,
    // including the DRAM-retention tradeoff this reintroduces.
    // -------------------------------------------------------------------------
    res = umfScalablePoolParamsCreate(&scalable_params);
    if (res != UMF_RESULT_SUCCESS) {
        fprintf(stderr, "Failed to create scalable pool params (node %d): %d\n", numa_node, res);
        pthread_mutex_unlock(&lifecycle_lock);
        return 5;
    }

    // Keep retaining freed blocks so transient allocs recycle in-pool and the
    // provider's bump offset advances slowly (only net-new live memory grows it).
    // Whether the pool retains freed blocks instead of returning them to the
    // provider. Overridable via UMF_KEEP_ALL_MEMORY (0/1) so it can be A/B'd
    // without a rebuild -- measurement showed TBB's resident footprint sits
    // ~29% above the bytes its own size classes reserve, and this is the only
    // UMF-level knob that plausibly governs that retention.
    {
        int keep_all = 0;
        const char *keep_env = getenv("UMF_KEEP_ALL_MEMORY");
        if (keep_env != NULL && keep_env[0] == '1') {
            keep_all = 1;
        }
        umfScalablePoolParamsSetKeepAllMemory(scalable_params, keep_all);
    }

    // Pool granularity: the unit the scalable pool requests from the provider.
    // Overridable via UMF_POOL_GRANULARITY (bytes) so it can be swept without
    // a rebuild. A large granularity means a single live object can pin a
    // whole chunk, which is one candidate explanation for resident memory
    // running ~1.7x the cache's accounted live bytes. Values below 4 KiB (one
    // page) are ignored.
    size_t huge_chunk_size = 2 * 1024 * 1024ULL;
    {
        const char *granularity_env = getenv("UMF_POOL_GRANULARITY");
        if (granularity_env != NULL) {
            unsigned long long parsed = strtoull(granularity_env, NULL, 10);
            if (parsed >= 4096ULL) {
                huge_chunk_size = (size_t)parsed;
            }
        }
    }
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
    // Whether the pool retains freed blocks instead of returning them to the
    // provider. Overridable via UMF_KEEP_ALL_MEMORY (0/1) so it can be A/B'd
    // without a rebuild -- measurement showed TBB's resident footprint sits
    // ~29% above the bytes its own size classes reserve, and this is the only
    // UMF-level knob that plausibly governs that retention.
    {
        int keep_all = 0;
        const char *keep_env = getenv("UMF_KEEP_ALL_MEMORY");
        if (keep_env != NULL && keep_env[0] == '1') {
            keep_all = 1;
        }
        umfScalablePoolParamsSetKeepAllMemory(scalable_params, keep_all);
    }
 
    // Pool granularity: the unit the scalable pool requests from the provider.
    // Overridable via UMF_POOL_GRANULARITY (bytes) so it can be swept without
    // a rebuild. A large granularity means a single live object can pin a
    // whole chunk, which is one candidate explanation for resident memory
    // running ~1.7x the cache's accounted live bytes. Values below 4 KiB (one
    // page) are ignored.
    size_t huge_chunk_size = 2 * 1024 * 1024ULL;
    {
        const char *granularity_env = getenv("UMF_POOL_GRANULARITY");
        if (granularity_env != NULL) {
            unsigned long long parsed = strtoull(granularity_env, NULL, 10);
            if (parsed >= 4096ULL) {
                huge_chunk_size = (size_t)parsed;
            }
        }
    }
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
    


/* Diagnostic accounting: TBB, unlike jemalloc, exposes no stats at all, so
 * the only way to decompose its resident footprint is to measure the two
 * quantities ourselves. `live_usable` is what the pool actually reserves per
 * live allocation (request rounded up to its size class); the *requested*
 * total is tracked on the Rust side, which knows `layout.size()` on both the
 * alloc and dealloc paths. usable/requested is then internal fragmentation and
 * resident/usable is everything else. Costs one msize query and one atomic per
 * alloc and per free -- diagnostic builds only. */
static _Atomic size_t live_usable[MAX_NODES];

size_t umf_live_usable(int numa_node) {
    if (numa_node < 0 || numa_node >= MAX_NODES) return 0;
    return atomic_load_explicit(&live_usable[numa_node], memory_order_relaxed);
}

/* Off unless UMF_TRACK_USABLE=1. The msize query runs on every alloc and every
 * free and costs on the order of a hundred nanoseconds per operation, which is
 * material against ~1.5us cache ops -- so it must never be on for a run whose
 * latency numbers are being recorded. Resolved once; the getenv race is benign
 * because every racing thread computes the same value. */
static atomic_int track_usable_enabled = -1;

static int usable_tracking_on(void) {
    int enabled = atomic_load_explicit(&track_usable_enabled, memory_order_relaxed);

    if (enabled < 0) {
        const char *env = getenv("UMF_TRACK_USABLE");
        enabled = (env != NULL && env[0] == '1') ? 1 : 0;
        atomic_store_explicit(&track_usable_enabled, enabled, memory_order_relaxed);
    }

    return enabled;
}

static void track_usable(umf_memory_pool_handle_t p, void *ptr, int numa_node, int add) {
    size_t usable = 0;
    if (ptr == NULL) return;
    if (!usable_tracking_on()) return;
    if (umfPoolMallocUsableSize(p, ptr, &usable) != UMF_RESULT_SUCCESS) return;
    if (add) {
        atomic_fetch_add_explicit(&live_usable[numa_node], usable, memory_order_relaxed);
    } else {
        atomic_fetch_sub_explicit(&live_usable[numa_node], usable, memory_order_relaxed);
    }
}

/* TBB's own purge, reached directly rather than through UMF.
 *
 * `umfScalablePoolParamsSetKeepAllMemory(0)` governs only whether the pool
 * returns memory to the provider. TBB keeps per-thread block free-lists and a
 * large-object cache ABOVE the pool, which that flag never reaches -- and this
 * workload allocates on the API thread while freeing on the worker and
 * migration consumers, so freed blocks accumulate in the freeing thread's
 * cache and are never handed back to the allocating one. This is the only call
 * that empties them.
 *
 * Returns TBBMALLOC_OK (0) if anything was released, TBBMALLOC_NO_EFFECT (1)
 * if there was nothing to release. Declared locally rather than including
 * <tbb/scalable_allocator.h> so the wrapper keeps building without TBB headers
 * present; the symbol is exported by libtbbmalloc, which UMF already links. */
#define TBB_CMD_CLEAN_ALL_BUFFERS 0

int umf_clean_all_buffers(void) {
    /* Resolved at run time rather than linked: the binary links -lumf, not
     * -ltbbmalloc -- UMF loads TBB itself -- so the symbol is present in the
     * process but not on the link line. RTLD_DEFAULT searches everything
     * already loaded. Returns -1 if TBB is not the backing pool. */
    static int (*cmd)(int, void *) = NULL;
    static atomic_int resolved = 0;

    if (!atomic_load_explicit(&resolved, memory_order_acquire)) {
        /* RTLD_DEFAULT does NOT work here: libumf dlopens libtbbmalloc
         * without RTLD_GLOBAL, so TBB's symbols are loaded but not in the
         * global namespace. RTLD_NOLOAD returns a handle to the already-mapped
         * library without loading a second copy -- which matters, since a
         * fresh copy would have its own empty caches and report nothing. */
        void *tbb = dlopen("libtbbmalloc.so.2", RTLD_LAZY | RTLD_NOLOAD);

        if (tbb != NULL) {
            cmd = (int (*)(int, void *))dlsym(tbb, "scalable_allocation_command");
        }

        atomic_store_explicit(&resolved, 1, memory_order_release);
    }

    if (!cmd) return -1;
    return cmd(TBB_CMD_CLEAN_ALL_BUFFERS, 0);
}

void *umf_alloc(int numa_node, size_t size, size_t align) {
    if (size == 0) return NULL;
    if (numa_node < 0 || numa_node >= MAX_NODES) return NULL;

    /* Lock-free fast path. The pool handles its own thread safety via
     * per-thread caches; wrapping every alloc in a global mutex would
     * serialize the entire process and destroy scalability. */
    umf_memory_pool_handle_t p = (umf_memory_pool_handle_t)
        atomic_load_explicit(&pools[numa_node], memory_order_acquire);
    if (!p) return NULL;

    void *ptr;
    if (align && align > sizeof(void*)) {
        ptr = umfPoolAlignedMalloc(p, size, align);
    } else {
        ptr = umfPoolMalloc(p, size);
    }
    track_usable(p, ptr, numa_node, 1);
    return ptr;
}


void umf_dealloc(int numa_node, void *ptr) {
    if (!ptr) return;
    if (numa_node < 0 || numa_node >= MAX_NODES) return;

    umf_memory_pool_handle_t p = (umf_memory_pool_handle_t)
        atomic_load_explicit(&pools[numa_node], memory_order_acquire);
    if (!p) return;  /* pool already destroyed; OS will reclaim on exit */

    track_usable(p, ptr, numa_node, 0);
    umfPoolFree(p, ptr);
}


/* Returns the NUMA node id that owns this pointer, or -1 if the pointer
 * is not managed by any of our UMF pools.
 *
 * NOTE: return semantics changed. The old version returned 1=pmem / 0=dram.
 * Callers that used the return as a bool must be updated. Use the node id
 * directly, or compare against a known value (e.g. `check_tier(p) == 1`).
 */
int check_tier(void *ptr) {
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

