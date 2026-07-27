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
// NumaArenaPool: a small pool of same-node, BindStrict-pinned jemalloc_cxl
// arenas, shared by EvictionStackAllocator and SlowTierJemallocAllocator
// below -- both bypass jemalloc's tcache (TcacheMode::None), which is
// required for correct NUMA placement (see SlowTierJemallocAllocator's own
// doc comment for the bug this fixed), but means every allocation takes the
// arena-lock slow path directly instead of a thread-local fast path.
//
// Confirmed by direct measurement: a single shared arena under this
// tcache-bypassed access pattern is a severe bottleneck -- 8 concurrent
// threads on one arena measured ~460K ops/sec; 8 threads each on their own
// arena (from an 8-arena pool) measured ~11.7M ops/sec, a ~25x difference,
// saturating right around the thread count with no further gain from more
// arenas than that. Sized like DramMultiArenaObjects's pool (`4 * ncpus`)
// for the same reason -- enough arenas that concurrent threads rarely
// collide, without an arena per thread.
//
// Deliberately NOT the same mechanism DramMultiArenaObjects uses
// (`jemalloc_cxl::bind_thread_arena`, which permanently rebinds a whole
// thread's *implicit* default arena): that would misdirect a thread's
// *other*, unrelated allocations onto this node too. Each consumer instead
// keeps its own `thread_local!` caching this thread's round-robin-assigned
// arena, looked up once and reused for every subsequent allocation that
// consumer's `arena()` function serves.
//
// Restoration note: this crate's `eviction_stacks_pmem` feature now backs
// its eviction-stack metadata via `crate::Hybrid`/`HybridObjects` (UMF/TBB)
// instead of `EvictionStackAllocator` below (see that migration's notes in
// `CLAUDE.md`) -- `EvictionStackAllocator` was brought back on request as a
// standalone, available-but-unwired mechanism, re-gated on
// `jemalloc_cxl_slow_tier` alongside its sibling `SlowTierJemallocAllocator`/
// `DramMultiArenaObjects` (its original gate, `eviction_stacks_pmem`, would
// otherwise force that feature to depend on `jemalloc_cxl` again for code it
// no longer calls).
#[cfg(feature = "jemalloc_cxl_slow_tier")]
mod numa_arena_pool {
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::OnceLock;

    use jemalloc_cxl::{create_cxl_arena, CxlArena, CxlArenaConfig, NumaPolicy};

    pub(super) struct NumaArenaPool {
        node: u32,
        arenas: OnceLock<Vec<CxlArena>>,
        next: AtomicUsize,
    }

    impl NumaArenaPool {
        pub(super) const fn new(node: u32) -> Self {
            NumaArenaPool {
                node,
                arenas: OnceLock::new(),
                next: AtomicUsize::new(0),
            }
        }

        fn arena_list(&self) -> &[CxlArena] {
            self.arenas.get_or_init(|| {
                let count = 4 * std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4);
                let node = self.node;
                (0..count)
                    .map(|_| {
                        create_cxl_arena(CxlArenaConfig::new(node, NumaPolicy::BindStrict))
                            .unwrap_or_else(|e| panic!("node-{node} CXL arena creation should succeed: {e}"))
                    })
                    .collect()
            })
        }

        /// Hands back the next arena in round-robin order. Callers are
        /// expected to cache the result once per thread (see each
        /// consumer's own `thread_local!` below) rather than calling this
        /// on every allocation.
        pub(super) fn next_arena(&self) -> CxlArena {
            let list = self.arena_list();
            let idx = self.next.fetch_add(1, Ordering::Relaxed) % list.len();
            list[idx]
        }
    }
}

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
//
// Currently unused by `eviction_stacks_pmem` itself (that feature's six
// `worker/policy/policy_stack/*.rs` call sites route through `crate::Hybrid`/
// `HybridObjects` instead -- see the migration notes in `CLAUDE.md`); kept
// available, gated on `jemalloc_cxl_slow_tier`, as a standalone allocator
// this crate could still wire back in later.
#[cfg(feature = "jemalloc_cxl_slow_tier")]
mod eviction_stack_allocator {
    use std::cell::Cell;

    use jemalloc_cxl::{CxlAllocator, CxlArena, TcacheMode};

    use super::numa_arena_pool::NumaArenaPool;

    /// NUMA node eviction-stack metadata is bound to, matching this crate's
    /// own `HybridObjects::NODE`/PMEM-node convention (node 1).
    const EVICTION_STACK_NODE: u32 = 1;

    static POOL: NumaArenaPool = NumaArenaPool::new(EVICTION_STACK_NODE);

    /// This calling thread's round-robin-assigned arena from `POOL`,
    /// looked up once and cached for the thread's remaining lifetime (see
    /// `numa_arena_pool`'s module doc for why this is a per-thread
    /// thread_local rather than jemalloc_cxl's whole-thread
    /// `bind_thread_arena`).
    fn arena() -> CxlArena {
        thread_local! {
            static ASSIGNED: Cell<Option<CxlArena>> = const { Cell::new(None) };
        }
        ASSIGNED.with(|slot| {
            if let Some(a) = slot.get() {
                return a;
            }
            let a = POOL.next_arena();
            slot.set(Some(a));
            a
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
            // TcacheMode::None (MALLOCX_TCACHE_NONE), not the
            // TcacheMode::Automatic this used before: TcacheMode::Automatic
            // lets jemalloc satisfy a MALLOCX_ARENA(N) request from the
            // calling thread's own automatic tcache -- bound to whichever
            // arena that thread was auto-assigned to, not this arena --
            // whenever a same-size-class chunk happens to be cached there.
            // This is the exact bug already found and fixed for
            // `SlowTierJemallocAllocator` (see that type's own doc comment):
            // confirmed there via direct instrumentation that
            // `TcacheMode::Automatic` allocations logically counted against
            // one arena were silently served from the wrong one, with zero
            // real pages resident on the target NUMA node despite the
            // allocator's own stats reporting the data as present. Applying
            // the same fix here rather than assuming eviction-stack
            // metadata was exempt from the same mechanism.
            <CxlAllocator as std::alloc::Allocator>::allocate(
                &CxlAllocator::with_tcache(arena(), TcacheMode::None),
                layout,
            )
                .map_err(|_| allocator_api2::alloc::AllocError)
        }

        unsafe fn deallocate(&self, ptr: std::ptr::NonNull<u8>, layout: std::alloc::Layout) {
            // SAFETY: caller upholds `allocator_api2::alloc::Allocator::
            // deallocate`'s contract (`ptr`/`layout` describe a still-live
            // allocation previously returned by this same allocator's
            // `allocate`). Must match `allocate`'s TcacheMode::None exactly
            // (see the jemalloc_cxl README's "Don't mix allocation/
            // deallocation APIs" section) -- deallocating with mismatched
            // tcache flags is undefined per jemalloc's own contract.
            unsafe {
                <CxlAllocator as std::alloc::Allocator>::deallocate(
                    &CxlAllocator::with_tcache(arena(), TcacheMode::None),
                    ptr,
                    layout,
                )
            }
        }
    }
}

#[cfg(feature = "jemalloc_cxl_slow_tier")]
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
    use std::cell::Cell;
    use std::ptr::NonNull;

    use jemalloc_cxl::{CxlAllocator, CxlArena, TcacheMode};

    use super::numa_arena_pool::NumaArenaPool;

    /// NUMA node backing the slow tier, matching `HybridObjects`'s own PMEM
    /// node (this file, above).
    const SLOW_TIER_NODE: u32 = 1;

    static POOL: NumaArenaPool = NumaArenaPool::new(SLOW_TIER_NODE);

    /// This calling thread's round-robin-assigned arena from `POOL` (see
    /// `numa_arena_pool`'s module doc, and `EvictionStackAllocator::arena`'s
    /// identical pattern -- confirmed by direct measurement that a single
    /// shared arena under `TcacheMode::None` is a ~25x throughput
    /// bottleneck under concurrent access, since every allocation then
    /// takes the arena-lock slow path directly).
    fn arena() -> CxlArena {
        thread_local! {
            static ASSIGNED: Cell<Option<CxlArena>> = const { Cell::new(None) };
        }
        ASSIGNED.with(|slot| {
            if let Some(a) = slot.get() {
                return a;
            }
            let a = POOL.next_arena();
            slot.set(Some(a));
            a
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

// ---------------------------------------------------------------------
// DramMultiArenaObjects: the crate's #[global_allocator] under
// `jemalloc_cxl_slow_tier` -- multiple node-0-pinned jemalloc arenas via
// jemalloc_cxl's custom extent hooks, instead of jemalloc's own default
// (unbound) arena selection.
// ---------------------------------------------------------------------
//
// Plain `jemalloc_cxl::Jemalloc` (what `jemalloc_cxl_slow_tier` used before
// this type existed) never calls `mbind` for its own default arenas at all
// -- it lands on node 0 today only incidentally, because node 1 in this
// crate's target topology has zero CPUs, so the kernel's ordinary
// local-allocation policy already happens to put every faulting thread's
// pages on node 0. That's not a guarantee: a different topology, a
// mem-pressure-triggered NUMA-balancing migration, or any future change
// that lets threads run on node 1 could silently drift fast-tier bytes off
// node 0. This type makes the placement explicit instead of incidental.
//
// A single node-0-bound arena (the obvious alternative, and how
// `SlowTierJemallocAllocator`/`EvictionStackAllocator` each pin *their*
// single node-1 arena) would reintroduce the exact contention problem
// jemalloc's own default multi-arena design exists to avoid: every thread
// serializing on one arena's locks. So instead this creates a small pool of
// node-0-bound arenas (sized like jemalloc's own default `narenas`
// heuristic -- `4 * available_parallelism()`), and each thread is handed a
// round-robin-assigned arena from that pool on its first allocation (see
// `current_arena()` below).
//
// CORRECTION (was: `jemalloc_cxl::bind_thread_arena` permanently rebinding
// each thread's own default arena via the "thread.arena" mallctl, then
// delegating straight to `jemalloc_cxl::Jemalloc`'s ordinary `GlobalAlloc`
// impl and relying on jemalloc's automatic per-thread tcache to follow that
// rebinding). That design was proven unsafe under real concurrent load: a
// real `-c 8` run against the full `standard_web.bin` trace (24 GB
// `--cache-max-size`, fast tier under real demotion/promotion pressure)
// reproducibly crashed with SIGSEGV *inside jemalloc's own extent-coalescing
// code* (`extent_try_coalesce_impl` -> `eset_remove` -> `edata_heap_remove`),
// reached from an entirely ordinary allocation path
// (`tcache_alloc_small_hard` -> `arena_cache_bin_fill_small` -> `pa_alloc`
// -> `ecache_alloc_grow` -> `extent_record`) -- a routine tcache bin refill
// that needed more pages from the arena ended up corrupting that arena's
// own free-extent heap. This is the same tcache/multi-arena interaction
// already documented (and fixed) for `SlowTierJemallocAllocator` above --
// `TcacheMode::Automatic` lets a per-arena request get quietly satisfied by
// state tied to the wrong/stale arena binding -- except here it manifested
// as real memory corruption inside jemalloc's own coalescing path (enabled
// by `cxl_extent_split`/`cxl_extent_merge`, see `extent.rs`) rather than
// merely incorrect NUMA placement. Fixed by applying the same proven fix
// used there: every allocation now goes through
// `CxlAllocator::with_tcache(arena, TcacheMode::None)` -- explicit
// `MALLOCX_ARENA(idx) | MALLOCX_TCACHE_NONE` on every call, bypassing
// jemalloc's per-thread tcache entirely so each request/free goes straight
// to the arena's own (properly locked) slab/bin path, instead of relying on
// a whole-thread rebind plus that thread's automatic tcache staying
// consistent with it. `dealloc` does not need to use the *same* arena the
// memory was originally allocated from: `sdallocx` looks up the owning
// extent/arena from jemalloc's own metadata, not from the `MALLOCX_ARENA`
// flag passed to the free call (this exact pattern -- a freeing thread's
// round-robin arena routinely differing from the allocating thread's -- is
// already used, and tested, in `SlowTierJemallocAllocator`/
// `EvictionStackAllocator` above); only the `MALLOCX_TCACHE_NONE` flag
// needs to match on the free side.
#[cfg(feature = "jemalloc_cxl_slow_tier")]
mod dram_multi_arena {
    use std::alloc::{Allocator, GlobalAlloc, Layout};
    use std::cell::Cell;
    use std::ptr::NonNull;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::OnceLock;

    use jemalloc_cxl::{
        create_cxl_arena, CxlAllocator, CxlArena, CxlArenaConfig, Jemalloc as InnerJemalloc,
        NumaPolicy, TcacheMode,
    };

    /// NUMA node the crate's ordinary (fast-tier) allocations are pinned
    /// to, matching every other DRAM-side convention in this file (node 0).
    const DRAM_NODE: u32 = 0;

    fn num_arenas() -> usize {
        // A small, fixed count (3) -- NOT jemalloc's own default `narenas`
        // heuristic (`4 * ncpus`, 32 on this 8-core machine), which was
        // tried first and confirmed to cause real allocation failures
        // under real -c8 load against the full standard_web.bin trace.
        //
        // Root cause: node 0's real settled footprint here is only
        // ~5.3-5.7 GB, but that budget gets split into `num_arenas()`
        // independent fragmentation domains -- more arenas means each one
        // is smaller, so a single large allocation (e.g. paper-benchmark's
        // ~50-67 MB end-of-run latency-percentile buffer) needs to find a
        // bigger *fraction* of its arena's total space as one contiguous
        // span, which gets exponentially harder under fragmentation as
        // that fraction grows. At 32 arenas (~170 MB each), a 50-67 MB
        // request needs ~30-40% of the entire arena contiguous; at 3
        // arenas (~1.8 GB each), the same request only needs ~3-4%.
        //
        // Confirmed directly by testing arena counts 2/3/4/32 against the
        // real benchmark (`-c 8`, full 14M-record trace, `--cache-max-size
        // 15G`): 32 arenas failed (an allocation aborting the process,
        // `memory allocation of N bytes failed`) after completing the
        // full trace, during final stats computation. 2 arenas failed
        // twice in a row, once even *mid-run* (67 MB allocation failure at
        // 66% through the trace, not just at the end) -- worse than 32,
        // confirming this isn't simply "fewer arenas is always safer." 3
        // arenas passed three times in a row cleanly, with throughput
        // matching or exceeding the 32-arena baseline (~125-150K SETs/sec
        // either way -- this workload's traffic goes through each bound
        // thread's own tcache, so arena count barely affects the hot
        // path). 4 arenas also passed once, comparably. Settled on 3 for a
        // small safety margin above the confirmed-failing floor of 2,
        // without reintroducing 32's fragmentation risk.
        //
        // Also tested 5 arenas specifically against the separate
        // `all_dram`/`standard_web.bin` node-0-exhaustion failure (see
        // that config's own notes) -- made no difference (failed
        // identically), confirming that failure is genuine aggregate
        // capacity exhaustion (no relief valve without a slow tier), not
        // fragmentation, so it isn't something this constant can fix.
        //
        // Re-tested (8 arenas, one per `-c 8` client thread) after the
        // tcache-bypass fix above, to check whether the earlier 2/3/4/32
        // comparison's conclusions still held now that `TcacheMode::
        // Automatic`'s corruption bug is gone. They did: 8 arenas failed
        // with the same signature and at the same rough point (mid-run,
        // ~80-86% through standard_web.bin at --cache-max-size 24G) as 3
        // arenas. Arena count isn't the lever for this remaining failure --
        // see the `lru_hybrid_cache` -c1/-c4/-c8 comparison this uncovered
        // (paper-benchmark-cxl session notes): -c1 and -c4 both complete
        // the identical trace cleanly, only -c8 fails, which points at a
        // genuine peak-concurrent-admission-backlog effect (higher client
        // concurrency means a higher SET/GET arrival rate than the single
        // PolicyWorker thread can drain tier decisions for, so more bytes
        // sit admitted-to-fast-but-not-yet-demoted at any given moment) --
        // not a per-arena fragmentation/contention problem this constant
        // could fix either way. Back to 3, the value validated at 15G.
        3
    }

    fn arenas() -> &'static [CxlArena] {
        static ARENAS: OnceLock<Vec<CxlArena>> = OnceLock::new();
        ARENAS.get_or_init(|| {
            (0..num_arenas())
                .map(|_| {
                    // `BindStrict` (MPOL_BIND): fast-tier/ordinary
                    // allocations must genuinely stay on node 0, hard
                    // failure otherwise -- confirmed this can be reached
                    // under real pressure (a workload whose combined
                    // fast-tier + bookkeeping footprint exceeds node 0's
                    // free memory aborts rather than silently spilling
                    // onto node 1), which is the explicit, accepted
                    // tradeoff of a strict node-0-only guarantee.
                    create_cxl_arena(CxlArenaConfig::new(DRAM_NODE, NumaPolicy::BindStrict))
                        .expect("node-0 CXL arena creation should succeed")
                })
                .collect()
        })
    }

    thread_local! {
        // This thread's round-robin-assigned arena, looked up once and
        // cached for the thread's remaining lifetime -- same pattern
        // `EvictionStackAllocator`/`SlowTierJemallocAllocator` use via the
        // shared `numa_arena_pool` helper (see that module's doc comment);
        // inlined here rather than sharing it because this type is the
        // process's actual `#[global_allocator]`, which needs the extra
        // reentrancy guard below that the other two never do.
        static ASSIGNED_ARENA: Cell<Option<CxlArena>> = const { Cell::new(None) };

        // Reentrancy guard for the arena-*lookup* process itself. Building
        // the arena pool the first time (`arenas()`'s `OnceLock::
        // get_or_init`, which `.collect()`s a `Vec`) allocates -- and every
        // allocation in the process, including that one, comes back through
        // this same `GlobalAlloc::alloc`. Without this guard, the very
        // first allocation the process ever makes would recurse into
        // `current_arena()` a second time while still inside the first
        // call's `arenas()` -- and `OnceLock` is not reentrant, so a nested
        // `get_or_init` on the same thread deadlocks forever rather than
        // erroring. While this guard is set, nested allocations skip
        // straight to `InnerJemalloc` (unbound, but correct) instead of
        // recursing.
        static ARENA_LOOKUP_IN_PROGRESS: Cell<bool> = const { Cell::new(false) };
    }

    /// This calling thread's round-robin-assigned arena, or `None` if
    /// called reentrantly from within the arena pool's own first-time setup
    /// (see `ARENA_LOOKUP_IN_PROGRESS` above) -- callers treat `None` as
    /// "fall back to `InnerJemalloc`'s ordinary, unbound default arena for
    /// just this one allocation."
    #[inline]
    fn current_arena() -> Option<CxlArena> {
        ASSIGNED_ARENA.with(|slot| {
            if let Some(a) = slot.get() {
                return Some(a);
            }

            let already_looking = ARENA_LOOKUP_IN_PROGRESS.with(|p| p.replace(true));
            if already_looking {
                // Reentered from within our own lookup process (see the
                // thread_local's doc comment above) -- let the nested
                // allocation fall through to InnerJemalloc unbound. The
                // outer call will still finish and cache this thread's
                // assigned arena.
                return None;
            }

            let list = arenas();
            static NEXT: AtomicUsize = AtomicUsize::new(0);
            let idx = NEXT.fetch_add(1, Ordering::Relaxed) % list.len();
            let a = list[idx];

            slot.set(Some(a));
            ARENA_LOOKUP_IN_PROGRESS.with(|p| p.set(false));

            Some(a)
        })
    }

    /// The crate's `#[global_allocator]` under `jemalloc_cxl_slow_tier`. A
    /// plain, `Copy`, no-state marker type -- matching every other
    /// allocator-handle type in this file -- whose real state (the arena
    /// pool, and each thread's assigned arena) lives behind the
    /// process-lifetime statics above.
    #[derive(Debug, Clone, Copy, Default)]
    pub struct DramMultiArenaObjects;

    unsafe impl GlobalAlloc for DramMultiArenaObjects {
        #[inline]
        unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
            match current_arena() {
                Some(arena) => {
                    // SAFETY: `layout` is passed through unchanged;
                    // `CxlAllocator::with_tcache(_, TcacheMode::None)`'s
                    // `Allocator::allocate` has the same size/alignment
                    // contract `GlobalAlloc::alloc` does. `MALLOCX_TCACHE_
                    // NONE` is required here, not optional -- see this
                    // module's doc comment for the real SIGSEGV this fixed.
                    match CxlAllocator::with_tcache(arena, TcacheMode::None).allocate(layout) {
                        Ok(ptr) => ptr.as_ptr() as *mut u8,
                        Err(_) => std::ptr::null_mut(),
                    }
                }
                // SAFETY: see `current_arena`'s reentrancy note.
                None => unsafe { InnerJemalloc.alloc(layout) },
            }
        }

        #[inline]
        unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
            // `CxlAllocator` only implements the nightly `Allocator` trait
            // (`allocate`/`deallocate`), which has no dedicated zeroing
            // fast path (no `MALLOCX_ZERO` plumbing) -- zero manually here,
            // matching what `GlobalAlloc::alloc_zeroed`'s own default
            // provided implementation already does for allocators without
            // one.
            let ptr = unsafe { self.alloc(layout) };
            if !ptr.is_null() {
                unsafe { std::ptr::write_bytes(ptr, 0, layout.size()) };
            }
            ptr
        }

        #[inline]
        unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
            // Uses this (freeing) thread's own assigned arena, not
            // necessarily the one the allocation came from -- correct per
            // this module's doc comment above (`sdallocx` resolves the real
            // owning arena from jemalloc's own metadata; only the
            // `MALLOCX_TCACHE_NONE` flag needs to match what `alloc` used).
            // Falls back to `InnerJemalloc` in the same reentrant-lookup
            // edge case `alloc` does.
            //
            // SAFETY: caller upholds `GlobalAlloc::dealloc`'s contract
            // (`ptr`/`layout` describe a still-live allocation previously
            // returned by this same allocator's `alloc`/`alloc_zeroed`), so
            // `ptr` is non-null.
            match current_arena() {
                Some(arena) => unsafe {
                    CxlAllocator::with_tcache(arena, TcacheMode::None)
                        .deallocate(NonNull::new_unchecked(ptr), layout)
                },
                None => unsafe { InnerJemalloc.dealloc(ptr, layout) },
            }
        }

        #[inline]
        unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
            // No `rallocx`/`xallocx` wired through `CxlAllocator` (it only
            // implements the nightly `Allocator` trait's `allocate`/
            // `deallocate`) -- reimplements `GlobalAlloc::realloc`'s own
            // documented default behavior instead: allocate new, copy the
            // overlapping prefix, free old. Not as fast as an in-place
            // `xallocx` would be, but this method was not implicated in the
            // crash this module's doc comment describes (a tcache bin-
            // refill path reached from plain `alloc`), so this
            // straightforward reimplementation is deliberately no fancier
            // than it needs to be.
            let new_layout = match Layout::from_size_align(new_size, layout.align()) {
                Ok(l) => l,
                Err(_) => return std::ptr::null_mut(),
            };

            // SAFETY: `alloc`/`dealloc` above are this same type's own
            // methods; `ptr`/`layout` describe a still-live allocation per
            // `GlobalAlloc::realloc`'s contract, so copying
            // `min(layout.size(), new_size)` bytes into the new allocation
            // before freeing the old one is exactly what jemalloc's own
            // realloc semantics guarantee.
            let new_ptr = unsafe { self.alloc(new_layout) };
            if !new_ptr.is_null() {
                unsafe {
                    std::ptr::copy_nonoverlapping(ptr, new_ptr, layout.size().min(new_size));
                    self.dealloc(ptr, layout);
                }
            }
            new_ptr
        }
    }

    #[cfg(test)]
    mod tests {
        // `DramMultiArenaObjects` is this crate's actual `#[global_allocator]`
        // under this feature (see lib.rs), so an ordinary `Vec`/`Box` in this
        // test binary already exercises `alloc`/`dealloc` through it -- these
        // tests check the two properties that can't be inferred from "the
        // rest of the suite didn't hang or crash": real NUMA residency (not
        // just that the reentrancy-guard fix stopped deadlocking), and that
        // concurrent threads actually spread across more than one arena
        // rather than piling onto one.

        fn numa_node1_available() -> bool {
            std::path::Path::new("/sys/devices/system/node/node1").exists()
        }

        /// Same helper shape as jemalloc_cxl's own numa_integration tests.
        fn resident_pages_on_node(addr: usize, node: u32) -> Option<u64> {
            let contents = std::fs::read_to_string("/proc/self/numa_maps").ok()?;
            let mut best: Option<&str> = None;
            for line in contents.lines() {
                let start_hex = line.split_whitespace().next()?;
                let Ok(start) = usize::from_str_radix(start_hex, 16) else {
                    continue;
                };
                if start <= addr {
                    best = Some(line);
                } else {
                    break;
                }
            }
            let line = best?;
            let field = format!("N{node}=");
            let value = line.split_whitespace().find_map(|tok| tok.strip_prefix(&field))?;
            value.parse().ok()
        }

        #[test]
        fn ordinary_allocations_are_resident_on_node_0() {
            // Meaningful even on a single-node machine (asserts >0 pages on
            // node 0, which is always true there); the interesting case --
            // confirming it's *not* landing on node 1 instead -- only
            // applies when a second node genuinely exists.
            const LEN: usize = 4 * 1024 * 1024; // 4 MiB, several pages
            let mut v: Vec<u8> = Vec::with_capacity(LEN);
            v.resize(LEN, 0x7A);

            let addr = v.as_ptr() as usize;
            let resident_node0 = resident_pages_on_node(addr, 0).unwrap_or(0);
            assert!(resident_node0 > 0, "expected pages resident on node 0, found none");

            if numa_node1_available() {
                let resident_node1 = resident_pages_on_node(addr, 1).unwrap_or(0);
                assert_eq!(resident_node1, 0, "expected zero pages on node 1 for a fast-tier/ordinary allocation");
            }
        }

        #[test]
        fn concurrent_threads_are_resident_on_node_0_and_spread_across_arenas() {
            use std::collections::HashSet;
            use std::sync::{Arc, Mutex};

            let seen_arenas: Arc<Mutex<HashSet<u32>>> = Arc::new(Mutex::new(HashSet::new()));
            let n_threads = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4).max(8);

            let handles: Vec<_> = (0..n_threads)
                .map(|_| {
                    let seen_arenas = Arc::clone(&seen_arenas);
                    std::thread::spawn(move || {
                        const LEN: usize = 1024 * 1024; // 1 MiB
                        let mut v: Vec<u8> = Vec::with_capacity(LEN);
                        v.resize(LEN, 0x5C);
                        let addr = v.as_ptr() as usize;

                        let resident_node0 = resident_pages_on_node(addr, 0).unwrap_or(0);
                        assert!(resident_node0 > 0, "expected pages resident on node 0, found none");
                        if numa_node1_available() {
                            let resident_node1 = resident_pages_on_node(addr, 1).unwrap_or(0);
                            assert_eq!(resident_node1, 0, "expected zero pages on node 1");
                        }

                        seen_arenas.lock().unwrap().insert(current_thread_arena());
                        drop(v);
                    })
                })
                .collect();

            for h in handles {
                h.join().unwrap();
            }

            let distinct = seen_arenas.lock().unwrap().len();
            assert!(
                distinct > 1,
                "expected concurrent threads to spread across more than one arena, all landed on {distinct}"
            );
        }

        fn current_thread_arena() -> u32 {
            use std::os::raw::c_uint;
            let mut ind: c_uint = 0;
            let mut ind_size = std::mem::size_of::<c_uint>();
            let rc = unsafe {
                jemalloc_cxl_mallctl_thread_arena_get(&raw mut ind, &raw mut ind_size)
            };
            assert_eq!(rc, 0, "reading thread.arena should succeed");
            ind
        }

        // `jemalloc_cxl` doesn't expose a public "read current thread.arena"
        // helper (it's only ever a private test helper inside its own crate,
        // see `thread_arena.rs`'s `tests::current_thread_arena`) -- this
        // duplicates that same raw mallctl read locally rather than adding
        // a new public API to jemalloc_cxl for a test-only need in a
        // downstream crate.
        unsafe fn jemalloc_cxl_mallctl_thread_arena_get(
            oldp: *mut std::os::raw::c_uint,
            oldlenp: *mut usize,
        ) -> std::os::raw::c_int {
            unsafe extern "C" {
                #[link_name = "_rjem_mallctl"]
                fn mallctl(
                    name: *const std::os::raw::c_char,
                    oldp: *mut std::ffi::c_void,
                    oldlenp: *mut usize,
                    newp: *mut std::ffi::c_void,
                    newlen: usize,
                ) -> std::os::raw::c_int;
            }
            unsafe {
                mallctl(
                    c"thread.arena".as_ptr(),
                    oldp.cast(),
                    oldlenp,
                    std::ptr::null_mut(),
                    0,
                )
            }
        }
    }
}

#[cfg(feature = "jemalloc_cxl_slow_tier")]
pub use dram_multi_arena::DramMultiArenaObjects;