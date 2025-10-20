int umf_allocator_init(const char *dax_path, size_t dax_size);
void *umf_alloc(size_t size, size_t align);
void umf_dealloc(void *ptr);
void umf_allocator_finalize(void);
void *return_pmem_base(size_t dax_size);
int check_tier(void *ptr);
