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

    /* Do NOT register umf_allocator_finalize via atexit.
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
     */
    return 0;
}



void *umf_alloc(size_t size, size_t align) {
    if (size == 0)
        return NULL;

    pthread_mutex_lock(&pool_lock);

    /* Guard against explicit umf_allocator_finalize() calls from a controlled
     * shutdown path: if the pool has already been destroyed, return NULL rather
     * than passing a NULL pool handle to umfPoolMalloc (which would trigger a
     * [FATAL UMF] assertion). */
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

    /* Guard against explicit umf_allocator_finalize() calls from a controlled
     * shutdown path: if the pool has already been destroyed, skip the free
     * rather than passing a NULL pool handle to umfPoolFree (which would
     * trigger a [FATAL UMF] assertion).  The memory was already released by
     * umfPoolDestroy when finalize was called. */
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