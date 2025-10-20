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


/*
int check_tier(void *ptr) {
    umf_memory_pool_handle_t pool;
    if (umfPoolByPtr(ptr, &pool) == UMF_RESULT_SUCCESS) {
        if (pool == NULL) {
            printf("Pointer %p is not from any UMF pool\n", ptr);
        } else if (pool == pool) {
            printf("Pointer %p is from PMEM pool\n", ptr);
        } else {
            printf("Pointer %p is from another UMF pool\n", ptr);
        }
        return 0;
    } else {
        printf("Failed to determine pool for pointer %p\n", ptr);
    }
    return 1;
}
*/


/*


#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <umf/memory_pool.h>
#include <umf/pools/pool_jemalloc.h>
#include <umf/providers/provider_devdax_memory.h>
#include <umf/memory_provider.h>

#include <sys/mman.h>
#include <fcntl.h>
#include <pthread.h>
#include <string.h>

static umf_memory_pool_handle_t pool = NULL;
static void *pmem_base = NULL;
static size_t pmem_size = 0;
static umf_devdax_memory_provider_params_handle_t dax_params = NULL;

int umf_allocator_init(const char *dax_path, size_t size, void *fixed_addr) {
    pmem_size = size;

    // mmap DAX device at fixed address
    int fd = open(dax_path, O_RDWR | O_SYNC);
    if (fd < 0) {
        perror("open dax");
        return 1;
    }

    pmem_base = mmap(fixed_addr, size, PROT_READ | PROT_WRITE,
                     MAP_SHARED | MAP_FIXED, fd, 0);
    if (pmem_base == MAP_FAILED) {
        perror("mmap");
        return 2;
    }
    //close(fd);

    // create custom memory provider
    umf_memory_provider_handle_t provider;
    if (umfMemoryProviderCreate(pmem_base, dax_params, &provider) != UMF_RESULT_SUCCESS) {
        fprintf(stderr, "Failed to create custom provider\n");
        return 3;
    }

    // create jemalloc pool on top of it
    if (umfPoolCreate(umfJemallocPoolOps(), provider, NULL, 0, &pool) != UMF_RESULT_SUCCESS) {
        fprintf(stderr, "Failed to create PMEM pool\n");
        return 4;
    }

    return 0;
}

void *umf_alloc(size_t size) {
    if (!pool || size == 0) return NULL;
    return umfPoolMalloc(pool, size);
}

void umf_dealloc(void *ptr) {
    if (!pool || !ptr) return;
    umfPoolFree(pool, ptr);
}


// return base and size
void *return_pmem_base() { return pmem_base; }
size_t return_pmem_size() { return pmem_size; }


*/
