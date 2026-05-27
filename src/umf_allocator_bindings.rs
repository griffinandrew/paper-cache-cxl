

//solo numa and daxdev config


/* 
unsafe extern "C" {

    //this one is for devdax 
    //pub fn umf_allocator_init(dax_path: *const libc::c_char, dax_size: usize) -> libc::c_int;

    //this one is for numa... 
    pub fn umf_allocator_init(numa_node: libc::c_int) -> libc::c_int;
    pub fn umf_alloc(size: usize, align: usize) -> *mut libc::c_void;
    pub fn umf_dealloc(ptr: *mut libc::c_void);
    pub fn umf_allocator_finalize();
    pub fn return_pmem_base(dax_size: usize) -> *mut libc::c_void;
    pub fn return_pmem_size() -> usize;
    pub fn check_tier(ptr: *mut libc::c_void) -> libc::c_int;
    pub fn umf_allocator_prewarm(bytes: usize, chunk: usize) -> i32;
} 

*/







unsafe extern "C" {
    pub fn umf_allocator_init(numa_node: c_int) -> c_int;
    pub fn umf_alloc(numa_node: c_int, size: usize, align: usize) -> *mut c_void;
    pub fn umf_dealloc(numa_node: c_int, ptr: *mut c_void);
    pub fn umf_allocator_prewarm(numa_node: c_int, bytes: usize, chunk: usize) -> c_int;
    pub fn check_tier(ptr: *mut c_void) -> c_int;  // now returns node id, was bool-ish
}



/*
unsafe extern "C" {
    /// Initialize UMF allocator with a list of NUMA nodes
    /// `numa_nodes` -> pointer to array of `unsigned` NUMA IDs
    /// `node_count` -> number of nodes in the array
    pub fn umf_allocator_init(numa_nodes: *const libc::c_uint, node_count: usize) -> libc::c_int;
    pub fn umf_alloc(size: usize, align: usize) -> *mut libc::c_void;
    pub fn umf_dealloc(ptr: *mut libc::c_void);
    pub fn umf_allocator_finalize();

    // Optional DevDAX/PMEM functions if you still have them
    pub fn return_pmem_base(dax_size: usize) -> *mut libc::c_void;
    pub fn return_pmem_size() -> usize;
    pub fn check_tier(ptr: *mut libc::c_void) -> libc::c_int;

}

*/