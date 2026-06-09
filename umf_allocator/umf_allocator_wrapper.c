


#include <pthread.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <stddef.h>
#include <string.h>

#include <umf/memory_pool.h>
#include <umf/memory_provider.h>
#include <umf/providers/provider_os_memory.h>
#include <umf/pools/pool_scalable.h>

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
    // Create scalable pool params and tell it to KEEP all memory.
    //This forces the TBB backend to retain freed blocks rather than triggering
    //purging calls down into the OS memory provider.
    // -------------------------------------------------------------------------
    res = umfScalablePoolParamsCreate(&scalable_params);
    if (res != UMF_RESULT_SUCCESS) {
        fprintf(stderr, "Failed to create scalable pool params (node %d): %d\n", numa_node, res);
        pthread_mutex_unlock(&lifecycle_lock);
        return 5;
    }

    // 1 means TRUE -> retain all memory blocks inside the pool user-space cache
    umfScalablePoolParamsSetKeepAllMemory(scalable_params, 1);

    // 2 MiB == default superblock size for the scalable pool. Matching the
    // granularity to the superblock size reduces fragmentation and improves
    // performance for typical workloads.
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

    umf_memory_pool_handle_t p = (umf_memory_pool_handle_t)
        atomic_load_explicit(&pools[numa_node], memory_order_acquire);
    if (!p) return;  /* pool already destroyed; OS will reclaim on exit */

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
#include <umf/pools/pool_jemalloc.h>



static umf_memory_pool_handle_t pool = NULL;
//static umf_memory_provider_handle_t dax_provider = NULL;
static umf_devdax_memory_provider_params_handle_t dax_params = NULL;


void umf_allocator_finalize_dax(void) {
    if (pool) {
        umfPoolDestroy(pool);
        pool = NULL;
    }
    if (providers[5]) {
        umfMemoryProviderDestroy(providers[5]);
        providers[5] = NULL
    }
    if (dax_params) {
        umfDevDaxMemoryProviderParamsDestroy(dax_params);
        dax_params = NULL;
    }
}

int umf_allocator_init_dax(const char *dax_path, size_t dax_size) {
    umf_jemalloc_pool_params_handle_t jemalloc_params = NULL;
    umf_result_t res;

    res = umfDevDaxMemoryProviderParamsCreate(dax_path, dax_size, &dax_params);
    if (res != UMF_RESULT_SUCCESS) {
        fprintf(stderr, "Failed to create DAX params: %d\n", res);
        return 1;
    }

    res = umfMemoryProviderCreate(umfDevDaxMemoryProviderOps(), dax_params, &dax_provider);
    if (res != UMF_RESULT_SUCCESS) {
        fprintf(stderr, "Failed to create DAX provider: %d\n", res);
        return 2;
    }

    res = umfJemallocPoolParamsCreate(&jemalloc_params);
    if (res != UMF_RESULT_SUCCESS) {
        fprintf(stderr, "Failed to create jemalloc pool params: %d\n", res);
        return 3;
    }

    res = umfPoolCreate(umfJemallocPoolOps(), providers[5], jemalloc_params, 0, &pool);
    umfJemallocPoolParamsDestroy(jemalloc_params);

    if (res != UMF_RESULT_SUCCESS) {
        fprintf(stderr, "Failed to create memory pool: %d\n", res);
        return 4;
    }

    // Zero all memory in the pool
    // in case of persistence.. dont think it matters for devdax tho.. 
    size_t pool_size = dax_size;
    void *base = umfPoolMalloc(pool, pool_size);
    if (base) {
        memset(base, 0, pool_size);
        umfPoolFree(pool, base);
    }
    //printf("base pointer init %p\n", base);

    atexit(umf_allocator_finalize_dax);
    return 0;
}


void *return_pmem_base_dax(size_t dax_size) {
    void *base = umfPoolMalloc(pool, dax_size);
    
    //if (base) {
    //    memset(base, 0, dax_size);
    //    umfPoolFree(pool, base);
    //}
    //printf("base pointer pmem %p\n", base);
    return base; //this should be the base address of the mapped PMEM region
}

void *umf_alloc_dax(size_t size, size_t align) {
    //pthread_mutex_lock(&pool_lock);
    if (!pool || size == 0) {
        //pthread_mutex_unlock(&pool_lock);
        fprintf(stderr, "Invalid allocation request: pool is NULL or size is 0, size=%zu\n", size);
        return NULL;
    }
    //void *ptr = umfPoolMalloc(pool, size); //might want to use the aligned version

    //respect alignment.... although jemalloc should do this for us...........
    void *ptr = umfPoolAlignedMalloc(pool, size, align);

    //pthread_mutex_unlock(&pool_lock);
    return ptr;
}

void umf_dealloc_dax(void *ptr) {
    //pthread_mutex_lock(&pool_lock);
    if (!pool || !ptr) {
        //pthread_mutex_unlock(&pool_lock);
        return;
    }
    umfPoolFree(pool, ptr);
    //pthread_mutex_unlock(&pool_lock);
}

int check_tier_dax(void *ptr) {
    umf_memory_pool_handle_t curr_pool;
    if (umfPoolByPtr(ptr, &curr_pool) == UMF_RESULT_SUCCESS) {

        if (curr_pool == pool) {
            return 1; //pmem
        }
    }
    else {
        return 0; //dram
    }
    //tjhis is unreachabke thoo
    return -1; //not from any UMF pool
}

