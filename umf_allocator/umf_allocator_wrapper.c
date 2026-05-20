/*

#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <umf/providers/provider_devdax_memory.h>
#include <umf/pools/pool_jemalloc.h>
#include <umf/memory_pool.h>
#include <umf/memory_provider.h>
#include <pthread.h>
#include <string.h>

static umf_memory_pool_handle_t pool = NULL;
static umf_memory_provider_handle_t dax_provider = NULL;
static umf_devdax_memory_provider_params_handle_t dax_params = NULL;


void umf_allocator_finalize(void) {
    if (pool) {
        umfPoolDestroy(pool);
        pool = NULL;
    }
    if (dax_provider) {
        umfMemoryProviderDestroy(dax_provider);
        dax_provider = NULL;
    }
    if (dax_params) {
        umfDevDaxMemoryProviderParamsDestroy(dax_params);
        dax_params = NULL;
    }
}

int umf_allocator_init(const char *dax_path, size_t dax_size) {
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

    res = umfPoolCreate(umfJemallocPoolOps(), dax_provider, jemalloc_params, 0, &pool);
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

    atexit(umf_allocator_finalize);
    return 0;
}


void *return_pmem_base(size_t dax_size) {
    void *base = umfPoolMalloc(pool, dax_size);
    
    //if (base) {
    //    memset(base, 0, dax_size);
    //    umfPoolFree(pool, base);
    //}
    //printf("base pointer pmem %p\n", base);
    return base; //this should be the base address of the mapped PMEM region
}

void *umf_alloc(size_t size, size_t align) {
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

void umf_dealloc(void *ptr) {
    //pthread_mutex_lock(&pool_lock);
    if (!pool || !ptr) {
        //pthread_mutex_unlock(&pool_lock);
        return;
    }
    umfPoolFree(pool, ptr);
    //pthread_mutex_unlock(&pool_lock);
}

int check_tier(void *ptr) {
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

*/


//working NUMA verison for only using pmem


/*
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>

#include <umf/providers/provider_os_memory.h>
#include <umf/pools/pool_jemalloc.h>
#include <umf/memory_pool.h>
#include <umf/memory_provider.h>

static umf_memory_pool_handle_t pool = NULL;
static umf_memory_provider_handle_t provider = NULL;
static umf_os_memory_provider_params_handle_t os_params = NULL;

static pthread_mutex_t pool_lock = PTHREAD_MUTEX_INITIALIZER;



void umf_allocator_finalize(void) {
    pthread_mutex_lock(&pool_lock);

    if (pool) {
        umfPoolDestroy(pool);
        pool = NULL;
    }

    if (provider) {
        umfMemoryProviderDestroy(provider);
        provider = NULL;
    }

    if (os_params) {
        umfOsMemoryProviderParamsDestroy(os_params);
        os_params = NULL;
    }

    pthread_mutex_unlock(&pool_lock);
}



// numa_node = NUMA node id (check with numactl -H) 
int umf_allocator_init(int numa_node) {
    umf_jemalloc_pool_params_handle_t jemalloc_params = NULL;
    umf_result_t res;

    pthread_mutex_lock(&pool_lock);

    if (pool != NULL) {
        pthread_mutex_unlock(&pool_lock);
        return 0;
    }

    // Create OS provider params 
    res = umfOsMemoryProviderParamsCreate(&os_params);
    if (res != UMF_RESULT_SUCCESS) {
        fprintf(stderr, "Failed to create OS params: %d\n", res);
        pthread_mutex_unlock(&pool_lock);
        return 1;
    }

    // Set NUMA node list 
    unsigned numa_list[] = { (unsigned)numa_node };

    res = umfOsMemoryProviderParamsSetNumaList(
            os_params,
            numa_list,
            1);

    if (res != UMF_RESULT_SUCCESS) {
        fprintf(stderr, "Failed to set NUMA list: %d\n", res);
        pthread_mutex_unlock(&pool_lock);
        return 2;
    }

    // Bind strictly to that NUMA node 
    res = umfOsMemoryProviderParamsSetNumaMode(
            os_params,
            //UMF_NUMA_MODE_BIND);  // UMF_NUMA_MODE_DEFAULT this is what you want for numa auto tiering... 
            UMF_NUMA_MODE_BIND); // this allows the OS to manage the NUMA allocation, which should enable auto-tiering if configured properly in the OS

    if (res != UMF_RESULT_SUCCESS) {
        fprintf(stderr, "Failed to set NUMA mode: %d\n", res);
        pthread_mutex_unlock(&pool_lock);
        return 3;
    }

    // Create provider 
    res = umfMemoryProviderCreate(
            umfOsMemoryProviderOps(),
            os_params,
            &provider);

    if (res != UMF_RESULT_SUCCESS) {
        fprintf(stderr, "Failed to create OS provider: %d\n", res);
        pthread_mutex_unlock(&pool_lock);
        return 4;
    }

    // Create jemalloc pool params 
    res = umfJemallocPoolParamsCreate(&jemalloc_params);
    if (res != UMF_RESULT_SUCCESS) {
        fprintf(stderr, "Failed to create jemalloc params: %d\n", res);
        pthread_mutex_unlock(&pool_lock);
        return 5;
    }

    // Create pool 
    res = umfPoolCreate(
            umfJemallocPoolOps(),
            provider,
            jemalloc_params,
            0,
            &pool);

    umfJemallocPoolParamsDestroy(jemalloc_params);

    if (res != UMF_RESULT_SUCCESS) {
        fprintf(stderr, "Failed to create pool: %d\n", res);
        pthread_mutex_unlock(&pool_lock);
        return 6;
    }

    pthread_mutex_unlock(&pool_lock);

    // Do NOT register umf_allocator_finalize via atexit.
     *
     * Under large cache loads, background PolicyWorker and reinsertion threads
     * continue to alloc/dealloc PMEM buffers via HybridObjects after main()
     * returns.  An atexit handler would destroy the UMF pool while those
     * threads are still active, causing [FATAL UMF] assertion failures
     * (umfPoolFree / umfPoolMalloc receiving a NULL pool handle) and
     * "memory allocation of N bytes failed" panics in Rust.
     *
     * The OS reclaims all virtual memory (including UMF-managed PMEM pages)
     * when the process exits, so explicit pool teardown is unnecessary.
     * Call umf_allocator_finalize() explicitly before exit if deterministic
     * cleanup is required in a controlled shutdown path.
     //
    return 0;
}



void *umf_alloc(size_t size, size_t align) {
    if (size == 0)
        return NULL;

    pthread_mutex_lock(&pool_lock);

    // Guard against explicit umf_allocator_finalize() calls from a controlled
     * shutdown path: if the pool has already been destroyed, return NULL rather
     * than passing a NULL pool handle to umfPoolMalloc (which would trigger a
     * [FATAL UMF] assertion).
     //
    if (!pool) {
        pthread_mutex_unlock(&pool_lock);
        return NULL;
    }

    void *ptr;
    if (align && align > sizeof(void*)) {
        ptr = umfPoolAlignedMalloc(pool, size, align);
    } else {
        ptr = umfPoolMalloc(pool, size);
    }

    pthread_mutex_unlock(&pool_lock);
    return ptr;
}


void umf_dealloc(void *ptr) {
    if (!ptr)
        return;

    pthread_mutex_lock(&pool_lock);

      // Guard against explicit umf_allocator_finalize() calls from a controlled
     * shutdown path: if the pool has already been destroyed, skip the free
     * rather than passing a NULL pool handle to umfPoolFree (which would
     * trigger a [FATAL UMF] assertion).  The memory was already released by
     * umfPoolDestroy when finalize was called. 
    if (pool) {
        umfPoolFree(pool, ptr);
    }

    pthread_mutex_unlock(&pool_lock);
}


int check_tier(void *ptr) {
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


//end solo numa node allocation for pmem


*/




// working NUMA version for only using pmem

#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdatomic.h>
#include <pthread.h>

#include <umf/providers/provider_os_memory.h>
//#include <umf/pools/pool_jemalloc.h>
#include <umf/pools/pool_scalable.h>
#include <umf/memory_pool.h>
#include <umf/memory_provider.h>

/* Hot-path handle: read lock-free from umf_alloc / umf_dealloc / check_tier.
 * Written only under lifecycle_lock during init / finalize, using release
 * semantics so the pool is fully constructed before any thread observes it. */
static _Atomic(umf_memory_pool_handle_t) pool = NULL;

/* Lifecycle-only state: touched solely during init / finalize. */
static umf_memory_provider_handle_t provider = NULL;
static umf_os_memory_provider_params_handle_t os_params = NULL;

/* Serializes init / finalize against each other. Does NOT guard the hot path;
 * jemalloc is already thread-safe and has its own per-arena locks. */
static pthread_mutex_t lifecycle_lock = PTHREAD_MUTEX_INITIALIZER;


void umf_allocator_finalize(void) {
    pthread_mutex_lock(&lifecycle_lock);

    /* Publish NULL with release semantics. Any allocator threads still running
     * at this point are a programmer error: finalize must only be called from
     * a controlled shutdown path where no allocations are in flight. */
    umf_memory_pool_handle_t p = atomic_exchange_explicit(
        &pool, NULL, memory_order_acq_rel);

    if (p) {
        umfPoolDestroy(p);
    }

    if (provider) {
        umfMemoryProviderDestroy(provider);
        provider = NULL;
    }

    if (os_params) {
        umfOsMemoryProviderParamsDestroy(os_params);
        os_params = NULL;
    }

    pthread_mutex_unlock(&lifecycle_lock);
}

/*
// numa_node = NUMA node id (check with numactl -H)
int umf_allocator_init(int numa_node) {
    umf_jemalloc_pool_params_handle_t jemalloc_params = NULL;
    umf_memory_pool_handle_t new_pool = NULL;
    umf_result_t res;

    pthread_mutex_lock(&lifecycle_lock);

    if (atomic_load_explicit(&pool, memory_order_acquire) != NULL) {
        pthread_mutex_unlock(&lifecycle_lock);
        return 0;
    }

    // Create OS provider params
    res = umfOsMemoryProviderParamsCreate(&os_params);
    if (res != UMF_RESULT_SUCCESS) {
        fprintf(stderr, "Failed to create OS params: %d\n", res);
        pthread_mutex_unlock(&lifecycle_lock);
        return 1;
    }

    // Set NUMA node list
    unsigned numa_list[] = { (unsigned)numa_node };

    res = umfOsMemoryProviderParamsSetNumaList(os_params, numa_list, 1);
    if (res != UMF_RESULT_SUCCESS) {
        fprintf(stderr, "Failed to set NUMA list: %d\n", res);
        pthread_mutex_unlock(&lifecycle_lock);
        return 2;
    }

    // Bind strictly to that NUMA node
    res = umfOsMemoryProviderParamsSetNumaMode(os_params, UMF_NUMA_MODE_BIND);
    if (res != UMF_RESULT_SUCCESS) {
        fprintf(stderr, "Failed to set NUMA mode: %d\n", res);
        pthread_mutex_unlock(&lifecycle_lock);
        return 3;
    }

    // Create provider
    res = umfMemoryProviderCreate(
            umfOsMemoryProviderOps(),
            os_params,
            &provider);
    if (res != UMF_RESULT_SUCCESS) {
        fprintf(stderr, "Failed to create OS provider: %d\n", res);
        pthread_mutex_unlock(&lifecycle_lock);
        return 4;
    }

    // Create jemalloc pool params
    res = umfJemallocPoolParamsCreate(&jemalloc_params);
    if (res != UMF_RESULT_SUCCESS) {
        fprintf(stderr, "Failed to create jemalloc params: %d\n", res);
        pthread_mutex_unlock(&lifecycle_lock);
        return 5;
    }

    // Create pool into a local first; only publish once fully constructed
    res = umfPoolCreate(
            umfJemallocPoolOps(),
            provider,
            jemalloc_params,
            0,
            &new_pool);

    umfJemallocPoolParamsDestroy(jemalloc_params);

    if (res != UMF_RESULT_SUCCESS) {
        fprintf(stderr, "Failed to create pool: %d\n", res);
        pthread_mutex_unlock(&lifecycle_lock);
        return 6;
    }

    /* Release-store publishes the fully-initialized pool. Any thread that
     * later observes a non-NULL pool via acquire-load is guaranteed to see
     * a consistent pool object. */
    //atomic_store_explicit(&pool, new_pool, memory_order_release);

    //pthread_mutex_unlock(&lifecycle_lock);

    /* Do NOT register umf_allocator_finalize via atexit.
     *
     * Under large cache loads, background PolicyWorker and reinsertion threads
     * continue to alloc/dealloc PMEM buffers via HybridObjects after main()
     * returns.  An atexit handler would destroy the UMF pool while those
     * threads are still active, causing [FATAL UMF] assertion failures and
     * "memory allocation of N bytes failed" panics in Rust.
     *
     * The OS reclaims all virtual memory (including UMF-managed PMEM pages)
     * when the process exits, so explicit pool teardown is unnecessary.
     * Call umf_allocator_finalize() explicitly before exit if deterministic
     * cleanup is required in a controlled shutdown path where no allocator
     * threads are still running.
     */
    //return 0;
//}




// numa_node = NUMA node id (check with numactl -H)
int umf_allocator_init(int numa_node) {
    umf_memory_pool_handle_t new_pool = NULL;
    umf_result_t res;

    pthread_mutex_lock(&lifecycle_lock);

    if (atomic_load_explicit(&pool, memory_order_acquire) != NULL) {
        pthread_mutex_unlock(&lifecycle_lock);
        return 0;
    }

    // Create OS provider params
    res = umfOsMemoryProviderParamsCreate(&os_params);
    if (res != UMF_RESULT_SUCCESS) {
        fprintf(stderr, "Failed to create OS params: %d\n", res);
        pthread_mutex_unlock(&lifecycle_lock);
        return 1;
    }

    // Set NUMA node list
    unsigned numa_list[] = { (unsigned)numa_node };

    res = umfOsMemoryProviderParamsSetNumaList(os_params, numa_list, 1);
    if (res != UMF_RESULT_SUCCESS) {
        fprintf(stderr, "Failed to set NUMA list: %d\n", res);
        pthread_mutex_unlock(&lifecycle_lock);
        return 2;
    }

    // Bind strictly to that NUMA node
    res = umfOsMemoryProviderParamsSetNumaMode(os_params, UMF_NUMA_MODE_BIND);
    if (res != UMF_RESULT_SUCCESS) {
        fprintf(stderr, "Failed to set NUMA mode: %d\n", res);
        pthread_mutex_unlock(&lifecycle_lock);
        return 3;
    }

    // Create provider
    res = umfMemoryProviderCreate(
            umfOsMemoryProviderOps(),
            os_params,
            &provider);
    if (res != UMF_RESULT_SUCCESS) {
        fprintf(stderr, "Failed to create OS provider: %d\n", res);
        pthread_mutex_unlock(&lifecycle_lock);
        return 4;
    }

    // Create pool into a local first; only publish once fully constructed.
    // Scalable pool takes no params, so pass NULL.
    res = umfPoolCreate(
            umfScalablePoolOps(),
            provider,
            NULL,
            0,
            &new_pool);

    if (res != UMF_RESULT_SUCCESS) {
        fprintf(stderr, "Failed to create pool: %d\n", res);
        pthread_mutex_unlock(&lifecycle_lock);
        return 6;
    }

    /* Release-store publishes the fully-initialized pool. Any thread that
     * later observes a non-NULL pool via acquire-load is guaranteed to see
     * a consistent pool object. */
    atomic_store_explicit(&pool, new_pool, memory_order_release);

    pthread_mutex_unlock(&lifecycle_lock);

    /* Do NOT register umf_allocator_finalize via atexit.
     *
     * Under large cache loads, background PolicyWorker and reinsertion threads
     * continue to alloc/dealloc PMEM buffers via HybridObjects after main()
     * returns.  An atexit handler would destroy the UMF pool while those
     * threads are still active, causing [FATAL UMF] assertion failures and
     * "memory allocation of N bytes failed" panics in Rust.
     *
     * The OS reclaims all virtual memory (including UMF-managed PMEM pages)
     * when the process exits, so explicit pool teardown is unnecessary.
     * Call umf_allocator_finalize() explicitly before exit if deterministic
     * cleanup is required in a controlled shutdown path where no allocator
     * threads are still running.
     */
    return 0;
}



void *umf_alloc(size_t size, size_t align) {
    if (size == 0)
        return NULL;

    /* Lock-free fast path. jemalloc handles its own thread safety via
     * per-arena locks and thread caches; wrapping every alloc in a global
     * mutex would serialize the entire process and destroy scalability. */
    umf_memory_pool_handle_t p = atomic_load_explicit(&pool, memory_order_acquire);
    if (!p)
        return NULL;

    if (align && align > sizeof(void*)) {
        return umfPoolAlignedMalloc(p, size, align);
    }
    return umfPoolMalloc(p, size);
}


void umf_dealloc(void *ptr) {
    if (!ptr)
        return;

    umf_memory_pool_handle_t p = atomic_load_explicit(&pool, memory_order_acquire);
    if (!p)
        return;  /* pool already destroyed; OS will reclaim on exit */

    umfPoolFree(p, ptr);
}


int check_tier(void *ptr) {
    umf_memory_pool_handle_t curr_pool;
    umf_memory_pool_handle_t our_pool =
        atomic_load_explicit(&pool, memory_order_acquire);

    if (umfPoolByPtr(ptr, &curr_pool) == UMF_RESULT_SUCCESS) {
        if (curr_pool == our_pool) {
            return 1; // pmem
        }
        return 0;     // some other UMF pool (treat as not-ours)
    }
    return 0;         // not from any UMF pool → dram
}

// end solo numa node allocation for pmem


#include <unistd.h>

/* Prewarm the UMF pool by allocating `bytes` worth of memory through it in
 * `chunk`-sized pieces, touching every page so the OS provider faults +
 * zeroes them on the target NUMA node now, then freeing back into the pool.
 *
 * Whether the prewarm "sticks" depends on the pool retaining freed memory
 * rather than handing it back to the OS provider. The scalable pool (TBB)
 * retains aggressively by default — freed blocks stay in the pool's per-
 * thread caches and superblock free-lists. No decay timer the way jemalloc
 * has, so there's no special config needed for retention.
 *
 * Call AFTER umf_allocator_init, BEFORE the measured workload.
 * Returns 0 on success. */
int umf_allocator_prewarm(size_t bytes, size_t chunk) {
    if (bytes == 0) return 0;
    if (chunk == 0) chunk = 2 * 1024 * 1024;  /* 2 MiB default */

    umf_memory_pool_handle_t p =
        atomic_load_explicit(&pool, memory_order_acquire);
    if (!p) {
        fprintf(stderr, "umf_allocator_prewarm: pool not initialized\n");
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
                    "umf_allocator_prewarm: pool exhausted at %zu/%zu chunks "
                    "(%zu bytes requested)\n",
                    got, n, chunk);
            /* Free what we got; report partial. */
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
            "umf_allocator_prewarm: touched %zu chunks x %zu bytes = %zu MiB\n",
            got, chunk, (got * chunk) >> 20);

    /* Free everything back into the pool. The scalable pool retains these
     * blocks for fast reuse; the OS provider does not unmap, so the pages
     * stay mapped, faulted, and bound to the target NUMA node. */
    //for (size_t i = 0; i < got; i++) umfPoolFree(p, ptrs[i]);
    //free(ptrs);
    return 0;
}











































/* doesnt work... tried to have bioht numa nodes in same provider.... 
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>

#include <umf/providers/provider_os_memory.h>
#include <umf/pools/pool_jemalloc.h>
#include <umf/memory_pool.h>
#include <umf/memory_provider.h>

static umf_memory_pool_handle_t pool = NULL;
static umf_memory_provider_handle_t provider = NULL;
static umf_os_memory_provider_params_handle_t os_params = NULL;

static pthread_mutex_t pool_lock = PTHREAD_MUTEX_INITIALIZER;


void umf_allocator_finalize(void) {
    pthread_mutex_lock(&pool_lock);

    if (pool) {
        umfPoolDestroy(pool);
        pool = NULL;
    }

    if (provider) {
        umfMemoryProviderDestroy(provider);
        provider = NULL;
    }

    if (os_params) {
        umfOsMemoryProviderParamsDestroy(os_params);
        os_params = NULL;
    }

    pthread_mutex_unlock(&pool_lock);
}


// Initialize UMF allocator with a list of NUMA nodes.
// numa_nodes: array of NUMA node IDs
// node_count: number of nodes in the array
//

int umf_allocator_init(const unsigned *numa_nodes, size_t node_count) {
    umf_jemalloc_pool_params_handle_t jemalloc_params = NULL;
    umf_result_t res;

    if (!numa_nodes || node_count == 0) {
        fprintf(stderr, "Invalid NUMA node list\n");
        return 1;
    }

    pthread_mutex_lock(&pool_lock);

    if (pool != NULL) {
        pthread_mutex_unlock(&pool_lock);
        return 0; // already initialized
    }

    // Create OS provider params 
    res = umfOsMemoryProviderParamsCreate(&os_params);
    if (res != UMF_RESULT_SUCCESS) {
        fprintf(stderr, "Failed to create OS params: %d\n", res);
        pthread_mutex_unlock(&pool_lock);
        return 2;
    }

    // Set NUMA node list 
    res = umfOsMemoryProviderParamsSetNumaList(os_params, numa_nodes, node_count);
    if (res != UMF_RESULT_SUCCESS) {
        fprintf(stderr, "Failed to set NUMA list: %d\n", res);
        pthread_mutex_unlock(&pool_lock);
        return 3;
    }

    // Set default NUMA mode (OS decides allocation, enables auto-tiering) 
    res = umfOsMemoryProviderParamsSetNumaMode(os_params, UMF_NUMA_MODE_DEFAULT);
    if (res != UMF_RESULT_SUCCESS) {
        fprintf(stderr, "Failed to set NUMA mode: %d\n", res);
        pthread_mutex_unlock(&pool_lock);
        return 4;
    }

    // Create OS provider 
    res = umfMemoryProviderCreate(umfOsMemoryProviderOps(), os_params, &provider);
    if (res != UMF_RESULT_SUCCESS) {
        fprintf(stderr, "Failed to create OS provider: %d\n", res);
        pthread_mutex_unlock(&pool_lock);
        return 5;
    }

    // Create jemalloc pool params 
    res = umfJemallocPoolParamsCreate(&jemalloc_params);
    if (res != UMF_RESULT_SUCCESS) {
        fprintf(stderr, "Failed to create jemalloc params: %d\n", res);
        pthread_mutex_unlock(&pool_lock);
        return 6;
    }

    // Create UMF pool 
    res = umfPoolCreate(umfJemallocPoolOps(), provider, jemalloc_params, 0, &pool);
    umfJemallocPoolParamsDestroy(jemalloc_params);

    if (res != UMF_RESULT_SUCCESS) {
        fprintf(stderr, "Failed to create pool: %d\n", res);
        pthread_mutex_unlock(&pool_lock);
        return 7;
    }

    pthread_mutex_unlock(&pool_lock);

    // Register cleanup at exit
    atexit(umf_allocator_finalize);
    return 0;
}


void *umf_alloc(size_t size, size_t align) {
    if (!pool || size == 0)
        return NULL;

    pthread_mutex_lock(&pool_lock);

    void *ptr;
    if (align && align > sizeof(void *)) {
        ptr = umfPoolAlignedMalloc(pool, size, align);
    } else {
        ptr = umfPoolMalloc(pool, size);
    }

    pthread_mutex_unlock(&pool_lock);
    return ptr;
}

// Deallocate memory back to the UMF pool
void umf_dealloc(void *ptr) {
    if (!pool || !ptr)
        return;

    pthread_mutex_lock(&pool_lock);
    umfPoolFree(pool, ptr);
    pthread_mutex_unlock(&pool_lock);
}

*/
