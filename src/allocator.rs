// UMF-backed allocator for the PMEM / CXL far tier.
//
// `HybridObjects` is used exclusively as the typed allocator for `BufferPMEM`
// (`Box<[u8], HybridObjects>`).  It always delegates to the UMF C functions
// (`umf_alloc` / `umf_dealloc`):
//
//  - On a real PMEM machine the build script links the real UMF shared library
//    and `umf_allocator_wrapper.c`.  Allocations land on CXL/persistent memory
//    via the UMF OS-NUMA provider on NUMA node 1 (the PMEM DIMM).
//
//  - In CI or on developer machines (no UMF available) the build script
//    compiles `umf_stub.c` which provides the same C symbols backed by
//    standard `malloc`/`free`.  The far tier is functionally correct even
//    without real PMEM hardware, making all integration tests pass on any
//    machine.
//
// The DRAM small tier uses `BufferDRAM = Box<[u8]>` which allocates through
// the Rust global allocator (jemalloc when the `all_dram` feature is set).
// `HybridObjects` is intentionally NOT the global allocator so that DRAM-tier
// allocations are never accidentally routed to UMF.

use core::alloc::{GlobalAlloc, Layout};
use std::sync::{Once, atomic::{AtomicUsize, Ordering}};
use std::ptr;
use std::alloc::{Allocator, AllocError};
use std::ptr::NonNull;

mod allocator_bindings {
    include!("umf_allocator_bindings.rs"); // UMF extern "C" declarations
}

/// Allocator that routes every allocation through UMF.
///
/// On real PMEM hardware UMF maps memory from the CXL/NUMA-1 PMEM node.
/// In CI/testing environments the stub `umf_stub.c` backs UMF with standard
/// `malloc`/`free`, so the allocator is fully functional without hardware.
#[derive(Clone, Copy)]
pub struct HybridObjects;

// NOTE: `HybridObjects` (PMEM, NUMA node 1) and `DRAMObjects` (the crate's
// `#[global_allocator]`, NUMA node 0) used to share this single `Once`. Since
// `DRAMObjects::alloc` fires on the very first heap allocation of the whole
// process, it always won the race to run its closure, so `HybridObjects`'s
// `init_and_prewarm` never ran and `pools[1]` (the PMEM pool) was never
// created — every `umf_alloc(1, ...)` call returned NULL forever, aborting
// with "memory allocation of N bytes failed". Gave each allocator its own
// `Once` instead.
static INIT: Once = Once::new();
static PRINT_THRESHOLD: usize = 10000;
static mut NUM_ALLOCS: usize = 0;
static mut NUM_DEALLOCS: usize = 0;
static ALL_MEM_ALLOCATED: AtomicUsize = AtomicUsize::new(0);

//static mut NUM_CALLS_PMEM: usize = 0;



impl HybridObjects {
    /// Initialize the UMF pool and prewarm a working-set-sized region.
    /// Call from main() before the benchmark loop.
    pub fn init_and_prewarm(numa_node: i32, prewarm_bytes: usize) {
        //INIT.call_once(|| {
        unsafe { allocator_bindings::umf_allocator_init(numa_node); }
        //#[cfg(debug_assertions)]
        //println!("HybridObjects: UMF pool initialised on NUMA node {}", numa_node);
        //});

        // Prewarm disabled: this unconditionally touched a hardcoded 18 GiB
        // (ignoring the `prewarm_bytes` argument entirely -- every call site
        // passes a different intended size, e.g. 32 GiB here, confirming this
        // was dead/stale rather than intentional), regardless of the actual
        // configured cache size. Real memory usage should track
        // `max_size`/`fast_tier_size`, not a large fixed prewarm.
        /*
        let bytes = 18 * 1024 * 1024 * 1024;
        let chunk = 2 * 1024 * 1024usize;
        let rc = unsafe { allocator_bindings::umf_allocator_prewarm(numa_node, bytes, chunk) };
        if rc != 0 {
            eprintln!("UMF prewarm returned {}", rc);
        }

        let chunk_2 = 1024 * 4usize;
        let bytes_2 = 18 * 1024 * 1024 * 1024;
        let rc = unsafe { allocator_bindings::umf_allocator_prewarm(numa_node, bytes_2, chunk_2) };
        if rc != 0 {
            eprintln!("UMF prewarm returned {}", rc);
        }
        */
    }
    const NODE: i32 = 1;
}



unsafe impl GlobalAlloc for HybridObjects {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        // Initialise the UMF pool on first allocation.  On real hardware this
        // maps a jemalloc pool over the PMEM NUMA node.  In the stub this is
        // a no-op.
        //INIT.call_once(|| {
        //    let numa_node = 1; // PMEM NUMA node (ignored by stub)
        //    allocator_bindings::umf_allocator_init(numa_node);
        //    #[cfg(debug_assertions)]
        //    println!("HybridObjects: UMF pool initialised on NUMA node {}", numa_node);
        //});

        INIT.call_once( || { 
            HybridObjects::init_and_prewarm(1, 32 * 1024 * 1024 * 1024 )}
        );


        let ptr = allocator_bindings::umf_alloc(Self::NODE,layout.size(), layout.align()) as *mut u8;
        if ptr.is_null() {
            eprintln!("HybridObjects: UMF alloc failed for {} bytes", layout.size());
            return ptr::null_mut();
        }
        //println!("HybridObjects: UMF alloc succeeded for {} bytes at {:p} with node {}", layout.size(), ptr, Self::NODE);

        //unsafe {
        //    NUM_CALLS_PMEM += 1;
        //}
        //if NUM_CALLS_PMEM % PRINT_THRESHOLD == 0 {
        //    println!("HybridObjects: UMF alloc called {} times", NUM_CALLS_PMEM);
        //}
        #[cfg(debug_assertions)]
        {
            ALL_MEM_ALLOCATED.fetch_add(layout.size(), Ordering::SeqCst);
            unsafe {
                if PRINT_THRESHOLD < NUM_ALLOCS {
                    println!("HybridObjects alloc: {} bytes (total {} bytes)",
                        layout.size(), ALL_MEM_ALLOCATED.load(Ordering::SeqCst));
                    NUM_ALLOCS = 0;
                }
                NUM_ALLOCS += 1;
            }
        }

        ptr
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        allocator_bindings::umf_dealloc(Self::NODE, ptr as *mut std::ffi::c_void);

        #[cfg(debug_assertions)]
        {
            ALL_MEM_ALLOCATED.fetch_sub(layout.size(), Ordering::SeqCst);
            unsafe {
                if PRINT_THRESHOLD < NUM_DEALLOCS {
                    println!("HybridObjects dealloc: {} bytes (total {} bytes)",
                        layout.size(), ALL_MEM_ALLOCATED.load(Ordering::SeqCst));
                    NUM_DEALLOCS = 0;
                }
                NUM_DEALLOCS += 1;
            }
        }
    }
}

// Allocator trait — used by Vec<u8, HybridObjects> and Box<[u8], HybridObjects>
// (i.e. BufferPMEM) via the nightly allocator_api feature.
unsafe impl Allocator for HybridObjects {
    fn allocate(&self, layout: Layout) -> Result<NonNull<[u8]>, AllocError> {
        unsafe {
            HybridObjects::alloc(self, layout)
                .as_mut()
                .map(|ptr| NonNull::slice_from_raw_parts(
                    NonNull::new_unchecked(ptr),
                    layout.size(),
                ))
                .ok_or(AllocError)
        }
    }

    unsafe fn deallocate(&self, ptr: NonNull<u8>, layout: Layout) {
        HybridObjects::dealloc(self, ptr.as_ptr(), layout);
    }
}

// allocator_api2 support (for hashbrown and dlv-list under eviction_stacks_pmem)
#[cfg(any(feature = "global_hashtable_pmem", feature = "tiering_hashtable_pmem", feature = "eviction_stacks_pmem"))]
unsafe impl allocator_api2::alloc::Allocator for HybridObjects {
    fn allocate(&self, layout: Layout) -> Result<NonNull<[u8]>, allocator_api2::alloc::AllocError> {
        let ptr = unsafe { self.alloc(layout) };
        if ptr.is_null() {
            Err(allocator_api2::alloc::AllocError)
        } else {
            let slice = unsafe { std::slice::from_raw_parts_mut(ptr, layout.size()) };
            Ok(NonNull::from(slice))
        }
    }

    unsafe fn deallocate(&self, ptr: NonNull<u8>, layout: Layout) {
        unsafe { self.dealloc(ptr.as_ptr(), layout) }
    }
}






//------------ umf for numa 0


#[derive(Clone, Copy)]
pub struct DRAMObjects;

// Own dedicated `Once` — see the note on `HybridObjects`'s `INIT` above for
// why this must not be shared with it.
static DRAM_INIT: Once = Once::new();

//static mut NUM_CALLS_DRAM: usize = 0;



impl DRAMObjects {
    /// Initialize the UMF pool and prewarm a working-set-sized region.
    /// Call from main() before the benchmark loop.
    pub fn init_and_prewarm(numa_node: i32, prewarm_bytes: usize) {
        //INIT.call_once(|| {
        unsafe { allocator_bindings::umf_allocator_init(numa_node); }
        //#[cfg(debug_assertions)]
        //println!("DRAMObjects: UMF pool initialised on NUMA node {}", numa_node);
        //});

        // Prewarm disabled: this unconditionally touched a hardcoded 18 GiB
        // (ignoring the `prewarm_bytes` argument entirely -- the call site
        // passes 30 GiB, confirming this was dead/stale rather than
        // intentional), regardless of the actual configured cache size. Real
        // memory usage should track `max_size`/`fast_tier_size`, not a large
        // fixed prewarm.
        /*
        let bytes = 18 * 1024 * 1024 * 1024;
        let chunk = 2 * 1024 * 1024usize;
        let rc = unsafe { allocator_bindings::umf_allocator_prewarm(numa_node, bytes, chunk) };
        if rc != 0 {
            eprintln!("UMF prewarm returned {}", rc);
        }

        let chunk_2 = 1024 * 4usize;
        let bytes_2 = 18 * 1024 * 1024 * 1024;
        let rc = unsafe { allocator_bindings::umf_allocator_prewarm(numa_node, bytes_2, chunk_2) };
        if rc != 0 {
            eprintln!("UMF prewarm returned {}", rc);
        }
        */
    }
    const NODE_DRAM: i32 = 0;
}



unsafe impl GlobalAlloc for DRAMObjects {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        // Initialise the UMF pool on first allocation.  On real hardware this
        // maps a jemalloc pool over the PMEM NUMA node.  In the stub this is
        // a no-op.
        //INIT.call_once(|| {
        //    let numa_node = 0; // DRAM NUMA node (ignored by stub)
        //    allocator_bindings::umf_allocator_init(numa_node);
        //    #[cfg(debug_assertions)]
        //    println!("DRAMObjects: UMF pool initialised on NUMA node {}", numa_node);
        //});

        DRAM_INIT.call_once( || {
            DRAMObjects::init_and_prewarm(Self::NODE_DRAM, 30 * 1024 * 1024 * 1024);

            //println!("DRAMObjects: Initialising and prewarming UMF pool on NUMA node {} with {} bytes",
            //    Self::NODE_DRAM, 35 * 1024 * 1024 * 1024);
        });

        let ptr = allocator_bindings::umf_alloc(Self::NODE_DRAM,layout.size(), layout.align()) as *mut u8;
        if ptr.is_null() {
            eprintln!("DRAMObjects: UMF alloc failed for {} bytes", layout.size());
            return ptr::null_mut();
        }

        //println!("DRAMObjects: UMF alloc succeeded for {} bytes at {:p} with node {}", layout.size(), ptr, Self::NODE_DRAM);

        //unsafe {
        //    NUM_CALLS_DRAM += 1;
        //}

        //if  NUM_CALLS_DRAM % PRINT_THRESHOLD == 0 {
        //    println!("DRAMObjects: UMF alloc called {} times", NUM_CALLS_DRAM);
        //}


        #[cfg(debug_assertions)]
        {
            ALL_MEM_ALLOCATED.fetch_add(layout.size(), Ordering::SeqCst);
            unsafe {
                if PRINT_THRESHOLD < NUM_ALLOCS {
                    println!("DRAMObjects alloc: {} bytes (total {} bytes)",
                        layout.size(), ALL_MEM_ALLOCATED.load(Ordering::SeqCst));
                    NUM_ALLOCS = 0;
                }
                NUM_ALLOCS += 1;
            }
        }

        ptr
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        allocator_bindings::umf_dealloc(Self::NODE_DRAM, ptr as *mut std::ffi::c_void);

        #[cfg(debug_assertions)]
        {
            ALL_MEM_ALLOCATED.fetch_sub(layout.size(), Ordering::SeqCst);
            unsafe {
                if PRINT_THRESHOLD < NUM_DEALLOCS {
                    println!("DRAMObjects dealloc: {} bytes (total {} bytes)",
                        layout.size(), ALL_MEM_ALLOCATED.load(Ordering::SeqCst));
                    NUM_DEALLOCS = 0;
                }
                NUM_DEALLOCS += 1;
            }
        }
    }
}

// Allocator trait — used by Vec<u8, HybridObjects> and Box<[u8], HybridObjects>
// (i.e. BufferPMEM) via the nightly allocator_api feature.
unsafe impl Allocator for DRAMObjects {
    fn allocate(&self, layout: Layout) -> Result<NonNull<[u8]>, AllocError> {
        unsafe {
            DRAMObjects::alloc(self, layout)
                .as_mut()
                .map(|ptr| NonNull::slice_from_raw_parts(
                    NonNull::new_unchecked(ptr),
                    layout.size(),
                ))
                .ok_or(AllocError)
        }
    }

    unsafe fn deallocate(&self, ptr: NonNull<u8>, layout: Layout) {
        DRAMObjects::dealloc(self, ptr.as_ptr(), layout);
    }
}

// allocator_api2 support (for hashbrown and dlv-list under eviction_stacks_pmem)
#[cfg(any(feature = "global_hashtable_pmem", feature = "tiering_hashtable_pmem", feature = "eviction_stacks_pmem"))]
unsafe impl allocator_api2::alloc::Allocator for DRAMObjects {
    fn allocate(&self, layout: Layout) -> Result<NonNull<[u8]>, allocator_api2::alloc::AllocError> {
        let ptr = unsafe { self.alloc(layout) };
        if ptr.is_null() {
            Err(allocator_api2::alloc::AllocError)
        } else {
            let slice = unsafe { std::slice::from_raw_parts_mut(ptr, layout.size()) };
            Ok(NonNull::from(slice))
        }
    }

    unsafe fn deallocate(&self, ptr: NonNull<u8>, layout: Layout) {
        unsafe { self.dealloc(ptr.as_ptr(), layout) }
    }
}









//hardcode dram for box value..
#[derive(Clone, Copy)]
pub struct ValueDRAM;

//static mut NUM_CALLS_DRAM: usize = 0;



impl ValueDRAM {
    /// Initialize the UMF pool and prewarm a working-set-sized region.
    /// Call from main() before the benchmark loop.
    pub fn init_and_prewarm(numa_node: i32, prewarm_bytes: usize) {
        //INIT.call_once(|| {
        unsafe { allocator_bindings::umf_allocator_init(numa_node); }
        //#[cfg(debug_assertions)]
        //println!("DRAMObjects: UMF pool initialised on NUMA node {}", numa_node);
        //});

        // Prewarm disabled: this unconditionally touched a hardcoded 18 GiB
        // (ignoring the `prewarm_bytes` argument entirely), regardless of the
        // actual configured cache size. Already dead in practice (every call
        // site invoking `ValueDRAM::init_and_prewarm` is itself commented
        // out), but neutered here too for consistency should it be
        // reactivated. Real memory usage should track `max_size`/
        // `fast_tier_size`, not a large fixed prewarm.
        /*
        let bytes = 18 * 1024 * 1024 * 1024;
        let chunk = 2 * 1024 * 1024usize;
        let rc = unsafe { allocator_bindings::umf_allocator_prewarm(numa_node, bytes, chunk) };
        if rc != 0 {
            eprintln!("UMF prewarm returned {}", rc);
        }

        let chunk_2 = 1024 * 4usize;
        let bytes_2 = 18 * 1024 * 1024 * 1024;
        let rc = unsafe { allocator_bindings::umf_allocator_prewarm(numa_node, bytes_2, chunk_2) };
        if rc != 0 {
            eprintln!("UMF prewarm returned {}", rc);
        }
        */
    }
    const VALUE_DRAM_NODE: i32 = 2;
}



unsafe impl GlobalAlloc for ValueDRAM {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        // Initialise the UMF pool on first allocation.  On real hardware this
        // maps a jemalloc pool over the PMEM NUMA node.  In the stub this is
        // a no-op.
        //INIT.call_once(|| {
        //    let numa_node = 0; // DRAM NUMA node (ignored by stub)
        //    allocator_bindings::umf_allocator_init(numa_node);
        //    #[cfg(debug_assertions)]
        //    println!("DRAMObjects: UMF pool initialised on NUMA node {}", numa_node);
        //});

        //INIT.call_once( || { 
            //ValueDRAM::init_and_prewarm(Self::VALUE_DRAM_NODE, 8 * 1024 * 1024 * 1024);

            //println!("ValueDRAM: Initialising and prewarming UMF pool on NUMA node {} with {} bytes",
            //    Self::VALUE_DRAM_NODE, 8 * 1024 * 1024 * 1024);
        //});

        let ptr = allocator_bindings::umf_alloc(Self::VALUE_DRAM_NODE,layout.size(), layout.align()) as *mut u8;
        if ptr.is_null() {
            eprintln!("ValueDRAM: UMF alloc failed for {} bytes", layout.size());
            return ptr::null_mut();
        }

        //println!("ValueDRAM: UMF alloc succeeded for {} bytes at {:p} with node {}", layout.size(), ptr, Self::VALUE_DRAM_NODE);

        //unsafe {
        //    NUM_CALLS_DRAM += 1;
        //}

        //if  NUM_CALLS_DRAM % PRINT_THRESHOLD == 0 {
        //    println!("ValueDRAM: UMF alloc called {} times", NUM_CALLS_DRAM);
        //}


        #[cfg(debug_assertions)]
        {
            ALL_MEM_ALLOCATED.fetch_add(layout.size(), Ordering::SeqCst);
            unsafe {
                if PRINT_THRESHOLD < NUM_ALLOCS {
                    println!("ValueDRAM alloc: {} bytes (total {} bytes)",
                        layout.size(), ALL_MEM_ALLOCATED.load(Ordering::SeqCst));
                    NUM_ALLOCS = 0;
                }
                NUM_ALLOCS += 1;
            }
        }

        ptr
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        allocator_bindings::umf_dealloc(Self::VALUE_DRAM_NODE, ptr as *mut std::ffi::c_void);

        #[cfg(debug_assertions)]
        {
            ALL_MEM_ALLOCATED.fetch_sub(layout.size(), Ordering::SeqCst);
            unsafe {
                if PRINT_THRESHOLD < NUM_DEALLOCS {
                    println!("ValueDRAM dealloc: {} bytes (total {} bytes)",
                        layout.size(), ALL_MEM_ALLOCATED.load(Ordering::SeqCst));
                    NUM_DEALLOCS = 0;
                }
                NUM_DEALLOCS += 1;
            }
        }
    }
}

// Allocator trait — used by Vec<u8, HybridObjects> and Box<[u8], HybridObjects>
// (i.e. BufferPMEM) via the nightly allocator_api feature.
unsafe impl Allocator for ValueDRAM {
    fn allocate(&self, layout: Layout) -> Result<NonNull<[u8]>, AllocError> {
        unsafe {
            ValueDRAM::alloc(self, layout)
                .as_mut()
                .map(|ptr| NonNull::slice_from_raw_parts(
                    NonNull::new_unchecked(ptr),
                    layout.size(),
                ))
                .ok_or(AllocError)
        }
    }

    unsafe fn deallocate(&self, ptr: NonNull<u8>, layout: Layout) {
        ValueDRAM::dealloc(self, ptr.as_ptr(), layout);
    }
}

// allocator_api2 support (for hashbrown and dlv-list under eviction_stacks_pmem)
#[cfg(any(feature = "global_hashtable_pmem", feature = "tiering_hashtable_pmem", feature = "eviction_stacks_pmem"))]
unsafe impl allocator_api2::alloc::Allocator for ValueDRAM {
    fn allocate(&self, layout: Layout) -> Result<NonNull<[u8]>, allocator_api2::alloc::AllocError> {
        let ptr = unsafe { self.alloc(layout) };
        if ptr.is_null() {
            Err(allocator_api2::alloc::AllocError)
        } else {
            let slice = unsafe { std::slice::from_raw_parts_mut(ptr, layout.size()) };
            Ok(NonNull::from(slice))
        }
    }

    unsafe fn deallocate(&self, ptr: NonNull<u8>, layout: Layout) {
        unsafe { self.dealloc(ptr.as_ptr(), layout) }
    }
}
















































/* 

unsafe impl GlobalAlloc for UnifiedAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        // Initialise the UMF pool on first allocation.  On real hardware this
        // maps a jemalloc pool over the PMEM NUMA node.  In the stub this is
        // a no-op.
        //INIT.call_once(|| {
        //    let numa_node = 0; // DRAM NUMA node (ignored by stub)
        //    allocator_bindings::umf_allocator_init(numa_node);
        //    #[cfg(debug_assertions)]
        //    println!("DRAMObjects: UMF pool initialised on NUMA node {}", numa_node);
        //});

        INIT.call_once( || { 
            //UnifiedAllocator::init_and_prewarm(Self::NODE_DRAM, 35 * 1024 * 1024 * 1024);

            UnifiedAllocator::init_and_prewarm(1, 35 * 1024 * 1024 * 1024);
            UnifiedAllocator::init_and_prewarm(0, 35 * 1024 * 1024 * 1024);
    


            //println!("DRAMObjects: Initialising and prewarming UMF pool on NUMA node {} with {} bytes",
            //    Self::NODE_DRAM, 35 * 1024 * 1024 * 1024);
        });

        let ptr = allocator_bindings::umf_alloc(Self::NODE_DRAM,layout.size(), layout.align()) as *mut u8;
        if ptr.is_null() {
            println!("DRAMObjects: UMF alloc failed for {} bytes", layout.size());
            return ptr::null_mut();
        }

        //println!("DRAMObjects: UMF alloc succeeded for {} bytes at {:p} with node {}", layout.size(), ptr, Self::NODE_DRAM);

        //unsafe {
        //    NUM_CALLS_DRAM += 1;
        //}

        //if  NUM_CALLS_DRAM % PRINT_THRESHOLD == 0 {
        //    println!("DRAMObjects: UMF alloc called {} times", NUM_CALLS_DRAM);
        //}


        #[cfg(debug_assertions)]
        {
            ALL_MEM_ALLOCATED.fetch_add(layout.size(), Ordering::SeqCst);
            unsafe {
                if PRINT_THRESHOLD < NUM_ALLOCS {
                    println!("DRAMObjects alloc: {} bytes (total {} bytes)",
                        layout.size(), ALL_MEM_ALLOCATED.load(Ordering::SeqCst));
                    NUM_ALLOCS = 0;
                }
                NUM_ALLOCS += 1;
            }
        }

        ptr
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        allocator_bindings::umf_dealloc(Self::NODE_DRAM, ptr as *mut std::ffi::c_void);

        #[cfg(debug_assertions)]
        {
            ALL_MEM_ALLOCATED.fetch_sub(layout.size(), Ordering::SeqCst);
            unsafe {
                if PRINT_THRESHOLD < NUM_DEALLOCS {
                    println!("DRAMObjects dealloc: {} bytes (total {} bytes)",
                        layout.size(), ALL_MEM_ALLOCATED.load(Ordering::SeqCst));
                    NUM_DEALLOCS = 0;
                }
                NUM_DEALLOCS += 1;
            }
        }
    }
}

// Allocator trait — used by Vec<u8, HybridObjects> and Box<[u8], HybridObjects>
// (i.e. BufferPMEM) via the nightly allocator_api feature.
unsafe impl Allocator for DRAMObjects {
    fn allocate(&self, layout: Layout) -> Result<NonNull<[u8]>, AllocError> {
        unsafe {
            DRAMObjects::alloc(self, layout)
                .as_mut()
                .map(|ptr| NonNull::slice_from_raw_parts(
                    NonNull::new_unchecked(ptr),
                    layout.size(),
                ))
                .ok_or(AllocError)
        }
    }

    unsafe fn deallocate(&self, ptr: NonNull<u8>, layout: Layout) {
        DRAMObjects::dealloc(self, ptr.as_ptr(), layout);
    }
}

// allocator_api2 support (for hashbrown and dlv-list under eviction_stacks_pmem)
#[cfg(any(feature = "global_hashtable_pmem", feature = "tiering_hashtable_pmem", feature = "eviction_stacks_pmem"))]
unsafe impl allocator_api2::alloc::Allocator for DRAMObjects {
    fn allocate(&self, layout: Layout) -> Result<NonNull<[u8]>, allocator_api2::alloc::AllocError> {
        let ptr = unsafe { self.alloc(layout) };
        if ptr.is_null() {
            Err(allocator_api2::alloc::AllocError)
        } else {
            let slice = unsafe { std::slice::from_raw_parts_mut(ptr, layout.size()) };
            Ok(NonNull::from(slice))
        }
    }

    unsafe fn deallocate(&self, ptr: NonNull<u8>, layout: Layout) {
        unsafe { self.dealloc(ptr.as_ptr(), layout) }
    }
}





*/

// ---------------------------------------------------------------------
// EvictionStackAllocator: eviction-stack metadata PMEM allocation via
// jemalloc_cxl's custom-extent-hooks NUMA/CXL arena.
// ---------------------------------------------------------------------
//
// Deliberately independent of `Hybrid`/`HybridObjects` (the UMF-based
// allocator backing `BufferPMEM` and the other PMEM features above) --
// eviction-stack metadata (`PmemHashList`/`PmemVecList`/the per-stack
// `EntryMap`s in `worker/policy/policy_stack/`) is a different, much
// smaller allocation workload (per-key list/map nodes, not full object
// byte buffers), so it gets its own dedicated jemalloc arena rather than
// sharing UMF's. See `jemalloc_cxl/README.md` for the underlying mechanism:
// one jemalloc instance, a custom `extent_hooks_t` doing `mmap`+`mbind`,
// and a nightly `Allocator` handle over one arena.
#[cfg(feature = "eviction_stacks_pmem")]
mod eviction_stack_allocator {
    use std::sync::OnceLock;

    use jemalloc_cxl::{create_cxl_arena, CxlAllocator, CxlArena, CxlArenaConfig, NumaPolicy};

    /// NUMA node eviction-stack metadata is bound to, matching this crate's
    /// own `HybridObjects::NODE`/PMEM-node convention (node 1).
    const EVICTION_STACK_NODE: u32 = 1;

    /// Lazily creates (once, process-lifetime -- matching every other
    /// pool/arena in this file) the CXL arena backing all eviction-stack
    /// metadata. `BindStrict` (`MPOL_BIND`) matches `HybridObjects`'s own
    /// existing UMF configuration (`UMF_NUMA_MODE_BIND`), for closer
    /// behavioral parity between the two PMEM paths.
    fn arena() -> CxlArena {
        static ARENA: OnceLock<CxlArena> = OnceLock::new();
        *ARENA.get_or_init(|| {
            create_cxl_arena(CxlArenaConfig::new(EVICTION_STACK_NODE, NumaPolicy::BindStrict))
                .expect("eviction-stack CXL arena creation should succeed")
        })
    }

    /// Zero-sized allocator handle for eviction-stack metadata, routed
    /// through a dedicated jemalloc arena via `jemalloc_cxl`. A plain,
    /// `Copy`, no-state value -- matching `HybridObjects`/`DRAMObjects`'s
    /// own zero-sized-marker-type shape -- with the real arena/allocator
    /// handle looked up lazily via `arena()` on every call, the same
    /// lazy-init pattern this file already uses for its UMF pools.
    #[derive(Debug, Clone, Copy, Default)]
    pub struct EvictionStackAllocator;

    // `allocator_api2::alloc::Allocator`, not the nightly `std::alloc::
    // Allocator` -- `PmemHashList`/`PmemVecList`/`hashbrown::HashMap`'s
    // `EntryMap`s (the only consumers of this type) are all built on the
    // stable-Rust-compatible `allocator_api2` crate, matching
    // `HybridObjects`/`DRAMObjects`'s own `allocator_api2` impls above.
    unsafe impl allocator_api2::alloc::Allocator for EvictionStackAllocator {
        fn allocate(
            &self,
            layout: std::alloc::Layout,
        ) -> Result<std::ptr::NonNull<[u8]>, allocator_api2::alloc::AllocError> {
            // Bridges to CxlAllocator's own nightly `Allocator::allocate`
            // (jemalloc_cxl's only public allocation entry point) -- no
            // unsafe work of our own, just adapting one allocator-trait
            // shape to another; both mirror the same allocate contract.
            <CxlAllocator as std::alloc::Allocator>::allocate(&CxlAllocator::new(arena()), layout)
                .map_err(|_| allocator_api2::alloc::AllocError)
        }

        unsafe fn deallocate(&self, ptr: std::ptr::NonNull<u8>, layout: std::alloc::Layout) {
            // SAFETY: caller upholds `allocator_api2::alloc::Allocator::
            // deallocate`'s contract (`ptr`/`layout` describe a still-live
            // allocation previously returned by this same allocator's
            // `allocate`). `CxlAllocator::deallocate`'s contract is
            // identical, and `CxlAllocator::new(arena())` reconstructs the
            // exact same arena/tcache encoding on every call (`arena()`
            // itself is cached behind a `OnceLock`, and `CxlAllocator::new`
            // always uses `TcacheMode::Automatic`) -- so this is the same
            // allocator in every sense `deallocate`'s contract cares about.
            unsafe {
                <CxlAllocator as std::alloc::Allocator>::deallocate(
                    &CxlAllocator::new(arena()),
                    ptr,
                    layout,
                )
            }
        }
    }
}

#[cfg(feature = "eviction_stacks_pmem")]
pub use eviction_stack_allocator::EvictionStackAllocator;

// ---------------------------------------------------------------------
// SlowTierJemallocAllocator: TieredBuffer::Slow's value bytes, via
// jemalloc_cxl's custom-extent-hooks NUMA/CXL arena, as an alternative
// backend to Hybrid/HybridObjects (the default -- see tiered_buffer.rs).
// ---------------------------------------------------------------------
//
// Unlike EvictionStackAllocator (which bridges to the stable-Rust-compatible
// `allocator_api2::alloc::Allocator` for PmemHashList/hashbrown), this backs
// `Box<[u8], SlowTierJemallocAllocator>` directly, so it implements the
// nightly `std::alloc::Allocator` trait CxlAllocator already implements --
// no bridging layer needed, just a direct delegation.
#[cfg(feature = "jemalloc_cxl_slow_tier")]
mod slow_tier_jemalloc_allocator {
    use std::alloc::{AllocError, Allocator, Layout};
    use std::ptr::NonNull;
    use std::sync::OnceLock;

    use jemalloc_cxl::{create_cxl_arena, CxlAllocator, CxlArena, CxlArenaConfig, NumaPolicy, TcacheMode};

    /// NUMA node backing the slow tier, matching `HybridObjects`'s own PMEM
    /// node (this file, above).
    const SLOW_TIER_NODE: u32 = 1;

    fn arena() -> CxlArena {
        static ARENA: OnceLock<CxlArena> = OnceLock::new();
        *ARENA.get_or_init(|| {
            create_cxl_arena(CxlArenaConfig::new(SLOW_TIER_NODE, NumaPolicy::BindStrict))
                .expect("slow-tier CXL arena creation should succeed")
        })
    }

    /// Zero-sized allocator handle for `TieredBuffer::Slow`'s value bytes,
    /// routed through a dedicated jemalloc arena via `jemalloc_cxl`,
    /// independent of `Hybrid`/`HybridObjects`'s own UMF/TBB pool for the
    /// same node. A plain, `Copy`, no-state value, matching this file's other
    /// allocator-marker types; the real arena/allocator handle is looked
    /// up lazily via `arena()` on every call.
    #[derive(Debug, Clone, Copy, Default)]
    pub struct SlowTierJemallocAllocator;

    unsafe impl Allocator for SlowTierJemallocAllocator {
        fn allocate(&self, layout: Layout) -> Result<NonNull<[u8]>, AllocError> {
            // TcacheMode::None (MALLOCX_TCACHE_NONE): TcacheMode::Automatic
            // let jemalloc satisfy MALLOCX_ARENA(33) requests from the
            // calling thread's own tcache, which is bound to whichever
            // arena that thread was auto-assigned to (not this arena) --
            // confirmed directly: instrumenting both this call site and the
            // extent-hooks alloc callback showed 140,000+ successful
            // "arena 33" allocations (totaling ~2.3 GB) against only ~8 MB
            // of extents ever actually mapped/mbind'd for arena 33, and a
            // corresponding real /proc/self/numa_maps read showed 0 MB on
            // the target NUMA node despite `lru_hybrid_stats().slow_bytes_used`
            // reporting multiple GB -- i.e. the data was silently served
            // from the wrong (unbound) arena's cached memory, not genuinely
            // NUMA-placed. MALLOCX_TCACHE_NONE forces every call to bypass
            // the tcache and draw directly from the named arena's own
            // extents, which is what actually makes the CXL/NUMA placement
            // real rather than nominal.
            CxlAllocator::with_tcache(arena(), TcacheMode::None).allocate(layout)
        }

        unsafe fn deallocate(&self, ptr: NonNull<u8>, layout: Layout) {
            // SAFETY: caller upholds Allocator::deallocate's contract
            // (ptr/layout describe a still-live allocation previously
            // returned by this same allocator's `allocate`). Must match
            // `allocate`'s TcacheMode::None exactly (see the README's "Don't
            // mix allocation/deallocation APIs" section) -- deallocating
            // with mismatched tcache flags is undefined per jemalloc's own
            // contract.
            unsafe { CxlAllocator::with_tcache(arena(), TcacheMode::None).deallocate(ptr, layout) }
        }
    }
}

#[cfg(feature = "jemalloc_cxl_slow_tier")]
pub use slow_tier_jemalloc_allocator::SlowTierJemallocAllocator;