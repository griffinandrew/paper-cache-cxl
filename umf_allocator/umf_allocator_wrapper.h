/*

int umf_allocator_init(const char *dax_path, size_t dax_size);
void *umf_alloc(size_t size, size_t align);
void umf_dealloc(void *ptr);
void umf_allocator_finalize(void);
void *return_pmem_base(size_t dax_size);
int check_tier(void *ptr); 

*/

/*
 * umf_allocator_wrapper.c
 *
 * This file provides a wrapper around the UMF allocator to manage memory pools and providers.
 * It initializes a UMF memory pool using the jemalloc provider, and provides functions for
 * allocating and deallocating memory, as well as finalizing the allocator.
 *
 * Note: This implementation assumes that the UMF library is properly installed and configured on the system.
 * The allocator is designed to be thread-safe using a mutex lock to protect access to the memory pool.
 * 
 *
 * 
*/


int umf_allocator_init(int numa_node);
void *umf_alloc(size_t size, size_t align);
void umf_dealloc(void *ptr);
void umf_allocator_finalize(void);
void *return_pmem_base(size_t dax_size);
int check_tier(void *ptr); 