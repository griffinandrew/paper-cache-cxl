// Nothing to build.
//
// This compiled UMF's C wrapper and linked libumf/libnuma for the TBB-backed
// allocator. That allocator is gone: both tiers now go through NUMA-bound
// jemalloc arenas (`src/numa_alloc.rs`), which links jemalloc via
// `tikv-jemalloc-sys` and issues `mbind` as a raw syscall, so there is no C
// to compile and no extra link flags to emit.
fn main() {}
