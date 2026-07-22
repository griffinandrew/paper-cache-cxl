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
#[cfg(any(feature = "pmem_region_alloc", feature = "region_hybrid_allocator", feature = "devdax_bump"))]
use std::sync::atomic::AtomicU64;
use std::ptr;
#[cfg(feature = "devdax_bump")]
use std::ffi::CString;
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
// `Once`, matching the pattern already used correctly by `RegionHybrid`
// (`REGION_INIT`) and `DevDaxBump` (`DEVDAX_INIT`) below.
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











//hardcode dram for box value..
#[derive(Clone, Copy)]
pub struct DAXPMEM;

//static mut NUM_CALLS_DRAM: usize = 0;



impl DAXPMEM {
    /// Initialize the UMF pool and prewarm a working-set-sized region.
    /// Call from main() before the benchmark loop.
    pub fn init_and_prewarm() {
        //INIT.call_once(|| {
        unsafe { allocator_bindings::umf_allocator_init_dax(Self::DEFAULT_DEVDAX_PATH.as_ptr(), Self::DAX_SIZE_BYTES); }
        //#[cfg(debug_assertions)]
        //println!("DRAMObjects: UMF pool initialised on NUMA node {}", numa_node);
        //});
    }

    //const DEFAULT_DEVDAX_PATH: &str = "/dev/dax0.0";
    const DEFAULT_DEVDAX_PATH: &'static std::ffi::CStr = c"/dev/dax0.0";
    const DAX_SIZE_BYTES: usize = 131061514240; // 100 GiB
}



unsafe impl GlobalAlloc for DAXPMEM {
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

        let ptr = allocator_bindings::umf_alloc_dax(layout.size(), layout.align()) as *mut u8;
        if ptr.is_null() {
            eprintln!("DAXPMEM: UMF alloc failed for {} bytes", layout.size());
            return ptr::null_mut();
        }

        //println!("DAXPMEM: UMF alloc succeeded for {} bytes at {:p} with node {}", layout.size(), ptr, Self::VALUE_DRAM_NODE);

        //unsafe {
        //    NUM_CALLS_DRAM += 1;
        //}

        //if  NUM_CALLS_DRAM % PRINT_THRESHOLD == 0 {
        //    println!("DAXPMEM: UMF alloc called {} times", NUM_CALLS_DRAM);
        //}


        #[cfg(debug_assertions)]
        {
            ALL_MEM_ALLOCATED.fetch_add(layout.size(), Ordering::SeqCst);
            unsafe {
                if PRINT_THRESHOLD < NUM_ALLOCS {
                    println!("DAXPMEM alloc: {} bytes (total {} bytes)",
                        layout.size(), ALL_MEM_ALLOCATED.load(Ordering::SeqCst));
                    NUM_ALLOCS = 0;
                }
                NUM_ALLOCS += 1;
            }
        }

        ptr
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        allocator_bindings::umf_dealloc_dax(ptr as *mut std::ffi::c_void);

        #[cfg(debug_assertions)]
        {
            ALL_MEM_ALLOCATED.fetch_sub(layout.size(), Ordering::SeqCst);
            unsafe {
                if PRINT_THRESHOLD < NUM_DEALLOCS {
                    println!("DAXPMEM dealloc: {} bytes (total {} bytes)",
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
unsafe impl Allocator for DAXPMEM {
    fn allocate(&self, layout: Layout) -> Result<NonNull<[u8]>, AllocError> {
        unsafe {
            DAXPMEM::alloc(self, layout)
                .as_mut()
                .map(|ptr| NonNull::slice_from_raw_parts(
                    NonNull::new_unchecked(ptr),
                    layout.size(),
                ))
                .ok_or(AllocError)
        }
    }

    unsafe fn deallocate(&self, ptr: NonNull<u8>, layout: Layout) {
        DAXPMEM::dealloc(self, ptr.as_ptr(), layout);
    }
}

// allocator_api2 support (for hashbrown and dlv-list under eviction_stacks_pmem)
#[cfg(any(feature = "global_hashtable_pmem", feature = "tiering_hashtable_pmem", feature = "eviction_stacks_pmem"))]
unsafe impl allocator_api2::alloc::Allocator for DAXPMEM {
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
/// PMEM region allocator:
/// - reserves one large mmap region
/// - attempts NUMA binding to PMEM node
/// - allocates with lock-free bump pointer
/// - deallocate is a no-op (bulk reclaim via `reclaim_all`)
#[cfg(any(feature = "pmem_region_alloc", feature = "region_hybrid_allocator"))]
#[derive(Clone, Copy)]
pub struct RegionHybrid;

#[cfg(any(feature = "pmem_region_alloc", feature = "region_hybrid_allocator"))]
const DEFAULT_PMEM_REGION_BYTES: usize = 120 * 1024 * 1024 * 1024; // 48 GiB virtual region

#[cfg(any(feature = "pmem_region_alloc", feature = "region_hybrid_allocator"))]
const DEFAULT_PMEM_NUMA_NODE: usize = 1;

#[cfg(any(feature = "pmem_region_alloc", feature = "region_hybrid_allocator"))]
static REGION_INIT: Once = Once::new();
#[cfg(any(feature = "pmem_region_alloc", feature = "region_hybrid_allocator"))]
static REGION_BASE_ADDR: AtomicUsize = AtomicUsize::new(0);
#[cfg(any(feature = "pmem_region_alloc", feature = "region_hybrid_allocator"))]
static REGION_SIZE_BYTES: AtomicUsize = AtomicUsize::new(0);
#[cfg(any(feature = "pmem_region_alloc", feature = "region_hybrid_allocator"))]
static REGION_OFFSET: AtomicUsize = AtomicUsize::new(0);
#[cfg(any(feature = "pmem_region_alloc", feature = "region_hybrid_allocator"))]
static REGION_GENERATION: AtomicU64 = AtomicU64::new(0);

#[cfg(any(feature = "pmem_region_alloc", feature = "region_hybrid_allocator"))]
impl RegionHybrid {
        /// Eagerly initialize the PMEM region (mmap + mbind + prefault).
    /// Call this once at process startup, before any latency-sensitive
    /// path, so the prefault cost is not charged to the first SET.
    pub fn init() {
        Self::init_if_needed();
    }
    
    /*#[inline]
    fn init_if_needed() {
        REGION_INIT.call_once(|| {
            let bytes = std::env::var("PAPER_CACHE_PMEM_REGION_SIZE")
                .ok()
                .and_then(|v| v.parse::<usize>().ok())
                .filter(|v| *v > 0)
                .unwrap_or(DEFAULT_PMEM_REGION_BYTES);

            let mapped = unsafe {
                libc::mmap(
                    ptr::null_mut(),
                    bytes,
                    libc::PROT_READ | libc::PROT_WRITE,
                    libc::MAP_PRIVATE | libc::MAP_ANONYMOUS,
                    -1,
                    0,
                )
            };

            assert_ne!(
                mapped,
                libc::MAP_FAILED,
                "RegionHybrid: mmap failed to reserve PMEM region"
            );

            let numa_node = std::env::var("PAPER_CACHE_PMEM_NUMA_NODE")
                .ok()
                .and_then(|v| v.parse::<usize>().ok())
                .unwrap_or(DEFAULT_PMEM_NUMA_NODE);

            #[cfg(target_os = "linux")]
            unsafe {
                Self::bind_region_to_numa(mapped, bytes, numa_node);
            }

            REGION_BASE_ADDR.store(mapped as usize, Ordering::SeqCst);
            REGION_SIZE_BYTES.store(bytes, Ordering::SeqCst);
            REGION_OFFSET.store(0, Ordering::SeqCst);

            #[cfg(debug_assertions)]
            println!(
                "RegionHybrid: mmap region={} bytes numa_node={} base={:p}",
                bytes, numa_node, mapped
            );
        });
    }
  

    //const MAP_HUGE_2MB: libc::c_int = 21 << 26; // Flags for explicitly selecting 2MB size

    #[inline]
    fn init_if_needed() {
        REGION_INIT.call_once(|| {
            let bytes = std::env::var("PAPER_CACHE_PMEM_REGION_SIZE")
                .ok()
                .and_then(|v| v.parse::<usize>().ok())
                .filter(|v| *v > 0)
                .unwrap_or(DEFAULT_PMEM_REGION_BYTES);

            const MAP_HUGE_2MB: libc::c_int = 21 << 26;

            let mapped = unsafe {
                libc::mmap(
                    ptr::null_mut(),
                    bytes,
                    libc::PROT_READ | libc::PROT_WRITE,
                    //libc::MAP_PRIVATE | libc::MAP_ANONYMOUS,
                    libc::MAP_PRIVATE | libc::MAP_ANONYMOUS | libc::MAP_HUGETLB | MAP_HUGE_2MB,
                    -1,
                    0,
                )
            };

            assert_ne!(
                mapped,
                libc::MAP_FAILED,
                "RegionHybrid: mmap failed to reserve PMEM region"
            );

            let numa_node = std::env::var("PAPER_CACHE_PMEM_NUMA_NODE")
                .ok()
                .and_then(|v| v.parse::<usize>().ok())
                .unwrap_or(DEFAULT_PMEM_NUMA_NODE);

            #[cfg(target_os = "linux")]
            unsafe {
                // Policy first, so faulted pages land on the PMEM node...
                Self::bind_region_to_numa(mapped, bytes, numa_node);
                // ...then fault every page in here, off the SET hot path.
                //Self::prefault_region(mapped, bytes);
            }

            REGION_BASE_ADDR.store(mapped as usize, Ordering::SeqCst);
            REGION_SIZE_BYTES.store(bytes, Ordering::SeqCst);
            REGION_OFFSET.store(0, Ordering::SeqCst);

            #[cfg(debug_assertions)]
            println!(
                "RegionHybrid: mmap region={} bytes numa_node={} base={:p} (prefaulted: touch-loop)",
                bytes, numa_node, mapped
            );
        });
    }

    */

   #[inline]
    fn init_if_needed() {
        REGION_INIT.call_once(|| {
            let bytes = std::env::var("PAPER_CACHE_PMEM_REGION_SIZE")
                .ok()
                .and_then(|v| v.parse::<usize>().ok())
                .filter(|v| *v > 0)
                .unwrap_or(DEFAULT_PMEM_REGION_BYTES);

            let numa_node = std::env::var("PAPER_CACHE_PMEM_NUMA_NODE")
                .ok()
                .and_then(|v| v.parse::<usize>().ok())
                .unwrap_or(DEFAULT_PMEM_NUMA_NODE);

            const MAP_HUGE_2MB: libc::c_int = 21 << 26;
            const MPOL_BIND: libc::c_int = 2;
            const MPOL_DEFAULT: libc::c_int = 0;

            let mapped = unsafe {
                // 1. Construct a thread mask pointing to Node 1
                let mut nodemask: libc::c_ulong = 0;
                let bits = (std::mem::size_of::<libc::c_ulong>() * 8) as usize;
                if numa_node < bits {
                    nodemask |= 1u64.wrapping_shl(numa_node as u32) as libc::c_ulong;
                }
                let maxnode = bits as libc::c_ulong;

                // 2. Temporarily switch thread policy to target Node 1's pools
                libc::syscall(
                    libc::SYS_set_mempolicy as libc::c_long,
                    MPOL_BIND,
                    &nodemask as *const libc::c_ulong,
                    maxnode,
                );

                // 3. Request the huge page layout mapping
                let ptr = libc::mmap(
                    ptr::null_mut(),
                    bytes,
                    libc::PROT_READ | libc::PROT_WRITE,
                    libc::MAP_PRIVATE | libc::MAP_ANONYMOUS | libc::MAP_HUGETLB | MAP_HUGE_2MB,
                    -1,
                    0,
                );

                // 4. Restore the default thread allocation policy for the rest of your app
                libc::syscall(
                    libc::SYS_set_mempolicy as libc::c_long,
                    MPOL_DEFAULT,
                    ptr::null::<libc::c_ulong>(), // Fixed compiler type inference
                    0,
                );

                ptr
            };

            assert_ne!(
                mapped,
                libc::MAP_FAILED,
                "RegionHybrid: mmap failed to reserve PMEM region"
            );

            #[cfg(target_os = "linux")]
            unsafe {
                Self::bind_region_to_numa(mapped, bytes, numa_node);
            }

            REGION_BASE_ADDR.store(mapped as usize, Ordering::SeqCst);
            REGION_SIZE_BYTES.store(bytes, Ordering::SeqCst);
            REGION_OFFSET.store(0, Ordering::SeqCst);

            #[cfg(debug_assertions)]
            println!(
                "RegionHybrid: mmap region={} bytes numa_node={} base={:p}",
                bytes, numa_node, mapped
            );
        });
    }

    /// Touch every page so the kernel allocates + zeroes physical pages now,
    /// on the PMEM node selected by the preceding `mbind`.
    #[cfg(target_os = "linux")]
    unsafe fn prefault_region(addr: *mut libc::c_void, size: usize) {
        // Optional: keep for a TLB-fair comparison vs a THP-backed numactl baseline.
        let _ = unsafe { libc::madvise(addr, size, libc::MADV_HUGEPAGE) };

        //let page_size = {
        //    let v = unsafe { libc::sysconf(libc::_SC_PAGESIZE) };
        //    if v > 0 { v as usize } else { 4096 }
        //};
        let page_size = 2 * 1024 * 1024;

        let base = addr as *mut u8;
        let mut offset = 0usize;
        while offset < size {
            // volatile so the store isn't elided — the fault must actually happen.
            unsafe { ptr::write_volatile(base.add(offset), 0u8); }
            offset += page_size;
        }
    }

    #[inline]
    fn alloc_bump(layout: Layout) -> *mut u8 {
        Self::init_if_needed();

        let base = REGION_BASE_ADDR.load(Ordering::SeqCst);
        let cap = REGION_SIZE_BYTES.load(Ordering::SeqCst);
        if base == 0 || cap == 0 {
            return ptr::null_mut();
        }

        let align = layout.align().max(1);
        let size = layout.size().max(1);

        loop {
            let curr = REGION_OFFSET.load(Ordering::Relaxed);
            let aligned = (curr + (align - 1)) & !(align - 1);
            let end = match aligned.checked_add(size) {
                Some(v) => v,
                None => return ptr::null_mut(),
            };

            if end > cap {
                return ptr::null_mut();
            }

            if REGION_OFFSET
                .compare_exchange_weak(curr, end, Ordering::SeqCst, Ordering::Relaxed)
                .is_ok()
            {
                return (base as *mut u8).wrapping_add(aligned);
            }
        }
    }

    pub fn reclaim_all() {
        Self::init_if_needed();
        REGION_OFFSET.store(0, Ordering::SeqCst);
        REGION_GENERATION.fetch_add(1, Ordering::SeqCst);
    }

    pub fn reset_epoch() {
        Self::reclaim_all();
    }

    pub fn generation() -> u64 {
        REGION_GENERATION.load(Ordering::SeqCst)
    }

    #[cfg(target_os = "linux")]
    unsafe fn bind_region_to_numa(addr: *mut libc::c_void, size: usize, node: usize) {
        const MPOL_BIND: libc::c_int = 2;
        let mut nodemask: libc::c_ulong = 0;
        let bits = (std::mem::size_of::<libc::c_ulong>() * 8) as usize;
        if node < bits {
            nodemask |= 1u64.wrapping_shl(node as u32) as libc::c_ulong;
        }
        let maxnode = bits as libc::c_ulong;

        #[allow(clippy::cast_possible_wrap)]
        let rc = unsafe {
            libc::syscall(
                libc::SYS_mbind as libc::c_long,
                addr,
                size,
                MPOL_BIND,
                &nodemask as *const libc::c_ulong,
                maxnode,
                0usize,
            )
        };

        #[cfg(debug_assertions)]
        if rc != 0 {
            eprintln!(
                "RegionHybrid: mbind(node={}) failed: {}",
                node,
                std::io::Error::last_os_error()
            );
        }
    }
}

#[cfg(any(feature = "pmem_region_alloc", feature = "region_hybrid_allocator"))]
unsafe impl GlobalAlloc for RegionHybrid {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        Self::alloc_bump(layout)
    }

    unsafe fn dealloc(&self, _ptr: *mut u8, _layout: Layout) {
        // Intentionally a no-op: this allocator uses bulk reclamation.
    }
}

#[cfg(any(feature = "pmem_region_alloc", feature = "region_hybrid_allocator"))]
unsafe impl Allocator for RegionHybrid {
    fn allocate(&self, layout: Layout) -> Result<NonNull<[u8]>, AllocError> {
        let ptr = Self::alloc_bump(layout);
        if ptr.is_null() {
            return Err(AllocError);
        }

        let slice = unsafe { std::slice::from_raw_parts_mut(ptr, layout.size().max(1)) };
        Ok(NonNull::from(slice))
    }

    unsafe fn deallocate(&self, _ptr: NonNull<u8>, _layout: Layout) {
        // Intentionally a no-op: this allocator uses bulk reclamation.
    }
}

// allocator_api2 support (for hashbrown and dlv-list under PMEM allocator modes)
#[cfg(all(
    any(feature = "pmem_region_alloc", feature = "region_hybrid_allocator"),
    any(
        feature = "global_hashtable_pmem",
        feature = "tiering_hashtable_pmem",
        feature = "eviction_stacks_pmem"
    )
))]
unsafe impl allocator_api2::alloc::Allocator for RegionHybrid {
    fn allocate(&self, layout: Layout) -> Result<NonNull<[u8]>, allocator_api2::alloc::AllocError> {
        let ptr = Self::alloc_bump(layout);
        if ptr.is_null() {
            return Err(allocator_api2::alloc::AllocError);
        }
        let slice = unsafe { std::slice::from_raw_parts_mut(ptr, layout.size().max(1)) };
        Ok(NonNull::from(slice))
    }

    unsafe fn deallocate(&self, _ptr: NonNull<u8>, _layout: Layout) {
        // Intentionally a no-op: this allocator uses bulk reclamation.
    }
}
*/







//use std::alloc::{AllocError, Allocator, GlobalAlloc, Layout};
///use std::ptr::{self, NonNull};
//use std::sync::Once;
//use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};

/// PMEM region allocator:
/// - reserves one large mmap region
/// - attempts NUMA binding to PMEM node
/// - allocates with lock-free bump pointer
/// - deallocate is a no-op (bulk reclaim via `reclaim_all`)
#[cfg(any(feature = "pmem_region_alloc", feature = "region_hybrid_allocator"))]
#[derive(Clone, Copy)]
pub struct RegionHybrid;

#[cfg(any(feature = "pmem_region_alloc", feature = "region_hybrid_allocator"))]
const DEFAULT_PMEM_REGION_BYTES: usize = 100 * 1024 * 1024 * 1024; // 120 GiB virtual region

#[cfg(any(feature = "pmem_region_alloc", feature = "region_hybrid_allocator"))]
const DEFAULT_PMEM_NUMA_NODE: usize = 1;

#[cfg(any(feature = "pmem_region_alloc", feature = "region_hybrid_allocator"))]
static REGION_INIT: Once = Once::new();
#[cfg(any(feature = "pmem_region_alloc", feature = "region_hybrid_allocator"))]
static REGION_BASE_ADDR: AtomicUsize = AtomicUsize::new(0);
#[cfg(any(feature = "pmem_region_alloc", feature = "region_hybrid_allocator"))]
static REGION_SIZE_BYTES: AtomicUsize = AtomicUsize::new(0);
#[cfg(any(feature = "pmem_region_alloc", feature = "region_hybrid_allocator"))]
static REGION_OFFSET: AtomicUsize = AtomicUsize::new(0);
#[cfg(any(feature = "pmem_region_alloc", feature = "region_hybrid_allocator"))]
static REGION_GENERATION: AtomicU64 = AtomicU64::new(0);

#[cfg(any(feature = "pmem_region_alloc", feature = "region_hybrid_allocator"))]
impl RegionHybrid {
    /// Eagerly initialize the PMEM region (mmap + mbind).
    /// Call this once at process startup, before any latency-sensitive
    /// path, so the kernel binds the target range.
    pub fn init() {
        Self::init_if_needed();
    }
    
    #[inline]
    fn init_if_needed() {
        REGION_INIT.call_once(|| {
            let bytes = std::env::var("PAPER_CACHE_PMEM_REGION_SIZE")
                .ok()
                .and_then(|v| v.parse::<usize>().ok())
                .filter(|v| *v > 0)
                .unwrap_or(DEFAULT_PMEM_REGION_BYTES);

            let numa_node = std::env::var("PAPER_CACHE_PMEM_NUMA_NODE")
                .ok()
                .and_then(|v| v.parse::<usize>().ok())
                .unwrap_or(DEFAULT_PMEM_NUMA_NODE);

            const MAP_HUGE_2MB: libc::c_int = 21 << 26;

            let mapped = unsafe {
                libc::mmap(
                    ptr::null_mut(),
                    bytes,
                    libc::PROT_READ | libc::PROT_WRITE,
                    //libc::MAP_PRIVATE | libc::MAP_ANONYMOUS | libc::MAP_HUGETLB | MAP_HUGE_2MB,
                    libc::MAP_PRIVATE | libc::MAP_ANONYMOUS,
                    -1,
                    0,
                )
            };

            assert_ne!(
                mapped,
                libc::MAP_FAILED,
                "RegionHybrid: mmap failed to reserve PMEM region"
            );

            #[cfg(target_os = "linux")]
            unsafe {
                // Apply strict binding onto the VMA range so future page faults 
                // reliably target your secondary tier node without needing full prefault loops.
                Self::bind_region_to_numa(mapped, bytes, numa_node);
                Self::prefault_region(mapped, bytes);
            }

            REGION_BASE_ADDR.store(mapped as usize, Ordering::SeqCst);
            REGION_SIZE_BYTES.store(bytes, Ordering::SeqCst);
            REGION_OFFSET.store(0, Ordering::SeqCst);

            #[cfg(debug_assertions)]
            println!(
                "RegionHybrid: mmap region={} bytes numa_node={} base={:p}",
                bytes, numa_node, mapped
            );
        });
    }

    /// Touch every page so the kernel allocates + zeroes physical pages now,
    /// on the PMEM node selected by the preceding `mbind`.
    #[cfg(target_os = "linux")]
    pub unsafe fn prefault_region(addr: *mut libc::c_void, size: usize) {
        let _ = unsafe { libc::madvise(addr, size, libc::MADV_HUGEPAGE) };
        let page_size = 2 * 1024 * 1024; // 2MB boundaries matching MAP_HUGE_2MB
        let base = addr as *mut u8;
        let mut offset = 0usize;
        while offset < size {
            unsafe { ptr::write_volatile(base.add(offset), 0u8); }
            offset += page_size;
        }
    }

    #[inline]
    fn alloc_bump(layout: Layout) -> *mut u8 {
        Self::init_if_needed();

        let base = REGION_BASE_ADDR.load(Ordering::Acquire);
        let cap = REGION_SIZE_BYTES.load(Ordering::Acquire);
        if base == 0 || cap == 0 {
            return ptr::null_mut();
        }

        let align = layout.align().max(1);
        let size = layout.size().max(1);

        loop {
            let curr = REGION_OFFSET.load(Ordering::Relaxed);
            let aligned = (curr + (align - 1)) & !(align - 1);
            let end = match aligned.checked_add(size) {
                Some(v) => v,
                None => return ptr::null_mut(),
            };

            if end > cap {
                return ptr::null_mut();
            }

            if REGION_OFFSET
                .compare_exchange_weak(curr, end, Ordering::Release, Ordering::Relaxed)
                .is_ok()
            {
                return (base as *mut u8).wrapping_add(aligned);
            }
        }
    }

    pub fn reclaim_all() {
        Self::init_if_needed();
        REGION_OFFSET.store(0, Ordering::Release);
        REGION_GENERATION.fetch_add(1, Ordering::SeqCst);
    }

    pub fn reset_epoch() {
        Self::reclaim_all();
    }

    pub fn generation() -> u64 {
        REGION_GENERATION.load(Ordering::Acquire)
    }

    #[cfg(target_os = "linux")]
    unsafe fn bind_region_to_numa(addr: *mut libc::c_void, size: usize, node: usize) {
        const MPOL_BIND: libc::c_int = 2;
        const MPOL_MF_STRICT: libc::c_int = 1;
        
        let mut nodemask: libc::c_ulong = 0;
        let bits = (std::mem::size_of::<libc::c_ulong>() * 8) as usize;
        if node < bits {
            nodemask |= 1u64.wrapping_shl(node as u32) as libc::c_ulong;
        }
        let maxnode = bits as libc::c_ulong;

        #[allow(clippy::cast_possible_wrap)]
        let rc = unsafe {
            libc::syscall(
                libc::SYS_mbind as libc::c_long,
                addr,
                size,
                MPOL_BIND,
                &nodemask as *const libc::c_ulong,
                maxnode,
                MPOL_MF_STRICT, // Added strict constraint validation to guarantee bindings hook onto the VMA
            )
        };

        if rc != 0 {
            let err = std::io::Error::last_os_error();
            panic!(
                "RegionHybrid: mbind(node={}) failed: {}",
                node, err
            );
        }
    }
}

#[cfg(any(feature = "pmem_region_alloc", feature = "region_hybrid_allocator"))]
unsafe impl GlobalAlloc for RegionHybrid {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        Self::alloc_bump(layout)
    }

    unsafe fn dealloc(&self, _ptr: *mut u8, _layout: Layout) {
        // Intentionally a no-op: this allocator uses bulk reclamation.
    }
}

#[cfg(any(feature = "pmem_region_alloc", feature = "region_hybrid_allocator"))]
unsafe impl Allocator for RegionHybrid {
    fn allocate(&self, layout: Layout) -> Result<NonNull<[u8]>, AllocError> {
        let ptr = Self::alloc_bump(layout);
        if ptr.is_null() {
            return Err(AllocError);
        }

        let slice = unsafe { std::slice::from_raw_parts_mut(ptr, layout.size().max(1)) };
        Ok(NonNull::from(slice))
    }

    unsafe fn deallocate(&self, _ptr: NonNull<u8>, _layout: Layout) {
        // Intentionally a no-op: this allocator uses bulk reclamation.
    }
}

// allocator_api2 support (for hashbrown and dlv-list under PMEM allocator modes)
#[cfg(all(
    any(feature = "pmem_region_alloc", feature = "region_hybrid_allocator"),
    any(
        feature = "global_hashtable_pmem",
        feature = "tiering_hashtable_pmem",
        feature = "eviction_stacks_pmem"
    )
))]
unsafe impl allocator_api2::alloc::Allocator for RegionHybrid {
    fn allocate(&self, layout: Layout) -> Result<NonNull<[u8]>, allocator_api2::alloc::AllocError> {
        let ptr = Self::alloc_bump(layout);
        if ptr.is_null() {
            return Err(allocator_api2::alloc::AllocError);
        }
        let slice = unsafe { std::slice::from_raw_parts_mut(ptr, layout.size().max(1)) };
        Ok(NonNull::from(slice))
    }

    unsafe fn deallocate(&self, _ptr: NonNull<u8>, _layout: Layout) {
        // Intentionally a no-op: this allocator uses bulk reclamation.
    }
}



#[cfg(feature = "devdax_bump")]
const DEFAULT_DEVDAX_PATH: &str = "/dev/dax0.0";

#[cfg(feature = "devdax_bump")]
const DEFAULT_DEVDAX_SIZE: usize = 110 * 1024 * 1024 * 1024; // 110 GiB

#[cfg(feature = "devdax_bump")]
#[derive(Clone, Copy)]
pub struct DevDaxBump;

#[cfg(feature = "devdax_bump")]
static DEVDAX_INIT:       Once        = Once::new();
#[cfg(feature = "devdax_bump")]
static DEVDAX_BASE_ADDR:  AtomicUsize = AtomicUsize::new(0);
#[cfg(feature = "devdax_bump")]
static DEVDAX_SIZE_BYTES: AtomicUsize = AtomicUsize::new(0);
#[cfg(feature = "devdax_bump")]
static DEVDAX_OFFSET:     AtomicUsize = AtomicUsize::new(0);
#[cfg(feature = "devdax_bump")]
static DEVDAX_GENERATION: AtomicU64   = AtomicU64::new(0);

#[cfg(feature = "devdax_bump")]
impl DevDaxBump {
    pub fn init() {
        Self::init_if_needed();
    }

    #[inline]
    fn init_if_needed() {
        DEVDAX_INIT.call_once(|| {
            let path = std::env::var("PAPER_CACHE_DEVDAX_PATH")
                .unwrap_or_else(|_| DEFAULT_DEVDAX_PATH.to_string());

            let size = std::env::var("PAPER_CACHE_DEVDAX_SIZE")
                .ok()
                .and_then(|v| v.parse::<usize>().ok())
                .filter(|v| *v > 0)
                .unwrap_or(DEFAULT_DEVDAX_SIZE);

            let cpath = CString::new(path.clone())
                .expect("PAPER_CACHE_DEVDAX_PATH contains NUL");

            let fd = unsafe { libc::open(cpath.as_ptr(), libc::O_RDWR) };
            assert!(
                fd >= 0,
                "DevDaxBump: open({}) failed: {}",
                path,
                std::io::Error::last_os_error()
            );

            let mapped = unsafe {
                libc::mmap(
                    ptr::null_mut(),
                    size,
                    libc::PROT_READ | libc::PROT_WRITE,
                    libc::MAP_SHARED,
                    fd,
                    0,
                )
            };

            unsafe { libc::close(fd); }

            assert_ne!(
                mapped,
                libc::MAP_FAILED,
                "DevDaxBump: mmap({}, {} bytes) failed: {}",
                path,
                size,
                std::io::Error::last_os_error()
            );

            DEVDAX_BASE_ADDR.store(mapped as usize, Ordering::SeqCst);
            DEVDAX_SIZE_BYTES.store(size, Ordering::SeqCst);
            DEVDAX_OFFSET.store(0, Ordering::SeqCst);

            eprintln!(
                "DevDaxBump: mapped {} bytes from {} at {:p}",
                size, path, mapped
            );
        });
    }

    #[inline]
    fn alloc_bump(layout: Layout) -> *mut u8 {
        Self::init_if_needed();

        let base = DEVDAX_BASE_ADDR.load(Ordering::SeqCst);
        let cap  = DEVDAX_SIZE_BYTES.load(Ordering::SeqCst);
        if base == 0 || cap == 0 {
            return ptr::null_mut();
        }

        let align = layout.align().max(1);
        let size  = layout.size().max(1);

        loop {
            let curr    = DEVDAX_OFFSET.load(Ordering::Relaxed);
            let aligned = (curr + (align - 1)) & !(align - 1);
            let end = match aligned.checked_add(size) {
                Some(v) => v,
                None    => return ptr::null_mut(),
            };

            if end > cap {
                eprintln!(
                    "DevDaxBump: region exhausted (need {}, cap {})",
                    end, cap
                );
                return ptr::null_mut();
            }

            if DEVDAX_OFFSET
                .compare_exchange_weak(curr, end, Ordering::SeqCst, Ordering::Relaxed)
                .is_ok()
            {
                return (base as *mut u8).wrapping_add(aligned);
            }
        }
    }

    pub fn reclaim_all() {
        Self::init_if_needed();
        DEVDAX_OFFSET.store(0, Ordering::SeqCst);
        DEVDAX_GENERATION.fetch_add(1, Ordering::SeqCst);
    }

    pub fn reset_epoch() {
        Self::reclaim_all();
    }

    pub fn generation() -> u64 {
        DEVDAX_GENERATION.load(Ordering::SeqCst)
    }
}

#[cfg(feature = "devdax_bump")]
unsafe impl GlobalAlloc for DevDaxBump {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        Self::alloc_bump(layout)
    }

    unsafe fn dealloc(&self, _ptr: *mut u8, _layout: Layout) {}
}

#[cfg(feature = "devdax_bump")]
unsafe impl Allocator for DevDaxBump {
    fn allocate(&self, layout: Layout) -> Result<NonNull<[u8]>, AllocError> {
        let ptr = Self::alloc_bump(layout);
        if ptr.is_null() {
            return Err(AllocError);
        }
        let slice = unsafe { std::slice::from_raw_parts_mut(ptr, layout.size().max(1)) };
        Ok(NonNull::from(slice))
    }

    unsafe fn deallocate(&self, _ptr: NonNull<u8>, _layout: Layout) {}
}

#[cfg(all(
    feature = "devdax_bump",
    any(
        feature = "global_hashtable_pmem",
        feature = "tiering_hashtable_pmem",
        feature = "eviction_stacks_pmem"
    )
))]
unsafe impl allocator_api2::alloc::Allocator for DevDaxBump {
    fn allocate(&self, layout: Layout) -> Result<NonNull<[u8]>, allocator_api2::alloc::AllocError> {
        let ptr = Self::alloc_bump(layout);
        if ptr.is_null() {
            return Err(allocator_api2::alloc::AllocError);
        }
        let slice = unsafe { std::slice::from_raw_parts_mut(ptr, layout.size().max(1)) };
        Ok(NonNull::from(slice))
    }

    unsafe fn deallocate(&self, _ptr: NonNull<u8>, _layout: Layout) {}
}




 // above is dev dax bump with lazy faults...




 /* 

#[cfg(feature = "devdax_bump")]
const DEFAULT_DEVDAX_PATH: &str = "/dev/dax0.0";

#[cfg(feature = "devdax_bump")]
const DEFAULT_DEVDAX_SIZE: usize = 110 * 1024 * 1024 * 1024; // 64 GiB

#[cfg(feature = "devdax_bump")]
#[derive(Clone, Copy)]
pub struct DevDaxBump;

#[cfg(feature = "devdax_bump")]
static DEVDAX_INIT:       Once        = Once::new();
#[cfg(feature = "devdax_bump")]
static DEVDAX_BASE_ADDR:  AtomicUsize = AtomicUsize::new(0);
#[cfg(feature = "devdax_bump")]
static DEVDAX_SIZE_BYTES: AtomicUsize = AtomicUsize::new(0);
#[cfg(feature = "devdax_bump")]
static DEVDAX_OFFSET:     AtomicUsize = AtomicUsize::new(0);
#[cfg(feature = "devdax_bump")]
static DEVDAX_GENERATION: AtomicU64   = AtomicU64::new(0);

/// A generation-tracked smart pointer protecting against use-after-free
/// bugs caused by `reclaim_all` epochs resetting beneath live data.
#[cfg(feature = "devdax_bump")]
#[derive(Debug)]
pub struct DaxPtr<T> {
    ptr: *mut T,
    generation: u64,
}

#[cfg(feature = "devdax_bump")]
impl<T> Clone for DaxPtr<T> {
    #[inline]
    fn clone(&self) -> Self { *self }
}

#[cfg(feature = "devdax_bump")]
impl<T> Copy for DaxPtr<T> {}

#[cfg(feature = "devdax_bump")]
impl<T> DaxPtr<T> {
    /// Safely dereference the pointer, validating that the allocation epoch
    /// matches the current live memory manager generation state.
    #[inline]
    pub fn as_ref(&self) -> &T {
        let current_gen = DevDaxBump::generation();
        assert_eq!(
            self.generation, current_gen,
            "Use-After-Free: Checked out an allocation from generation {}, but current live generation is {}",
            self.generation, current_gen
        );
        unsafe { &*self.ptr }
    }

    #[inline]
    pub fn as_mut(&mut self) -> &mut T {
        let current_gen = DevDaxBump::generation();
        assert_eq!(
            self.generation, current_gen,
            "Use-After-Free: Checked out a mutable allocation from generation {}, but current live generation is {}",
            self.generation, current_gen
        );
        unsafe { &mut *self.ptr }
    }

    #[inline]
    pub fn raw_ptr(&self) -> *mut T {
        self.ptr
    }
}

#[cfg(feature = "devdax_bump")]
impl DevDaxBump {
    pub fn init() {
        Self::init_if_needed();
    }

    #[inline]
    fn init_if_needed() {
        DEVDAX_INIT.call_once(|| {
            let path = std::env::var("PAPER_CACHE_DEVDAX_PATH")
                .unwrap_or_else(|_| DEFAULT_DEVDAX_PATH.to_string());

            let size = std::env::var("PAPER_CACHE_DEVDAX_SIZE")
                .ok()
                .and_then(|v| v.parse::<usize>().ok())
                .filter(|v| *v > 0)
                .unwrap_or(DEFAULT_DEVDAX_SIZE);

            let cpath = CString::new(path.clone())
                .expect("PAPER_CACHE_DEVDAX_PATH contains NUL");

            let fd = unsafe { libc::open(cpath.as_ptr(), libc::O_RDWR) };
            assert!(
                fd >= 0,
                "DevDaxBump: open({}) failed: {}",
                path,
                std::io::Error::last_os_error()
            );

            // Added MAP_POPULATE to eagerly back memory space with physical pages.
            // Bypasses runtime page faults during cache processing execution loops.
            let mapped = unsafe {
                libc::mmap(
                    ptr::null_mut(),
                    size,
                    libc::PROT_READ | libc::PROT_WRITE,
                    libc::MAP_SHARED | libc::MAP_POPULATE,
                    fd,
                    0,
                )
            };

            unsafe { libc::close(fd); }

            assert_ne!(
                mapped,
                libc::MAP_FAILED,
                "DevDaxBump: mmap({}, {} bytes) failed: {}",
                path,
                size,
                std::io::Error::last_os_error()
            );

            DEVDAX_BASE_ADDR.store(mapped as usize, Ordering::SeqCst);
            DEVDAX_SIZE_BYTES.store(size, Ordering::SeqCst);
            DEVDAX_OFFSET.store(0, Ordering::SeqCst);

            eprintln!(
                "DevDaxBump: mapped {} bytes from {} at {:p} with MAP_POPULATE",
                size, path, mapped
            );
        });
    }

    #[inline]
    fn alloc_bump(layout: Layout) -> *mut u8 {
        Self::init_if_needed();

        let base = DEVDAX_BASE_ADDR.load(Ordering::SeqCst);
        let cap  = DEVDAX_SIZE_BYTES.load(Ordering::SeqCst);
        if base == 0 || cap == 0 {
            return ptr::null_mut();
        }

        let align = layout.align().max(1);
        let size  = layout.size().max(1);

        loop {
            let curr    = DEVDAX_OFFSET.load(Ordering::Relaxed);
            
            // Align up the starting boundary for this allocation target
            let aligned = (curr + (align - 1)) & !(align - 1);
            
            let end_raw = match aligned.checked_add(size) {
                Some(v) => v,
                None    => return ptr::null_mut(),
            };

            // Fix: Guarantee that the trailing offset boundary saved to the global
            // state is properly aligned to the cache boundary layout as well.
            // This prevents consecutive threads from dealing with alignment trash loop cycles.
            let end = (end_raw + (align - 1)) & !(align - 1);

            if end > cap {
                eprintln!(
                    "DevDaxBump: region exhausted (need {}, cap {})",
                    end, cap
                );
                return ptr::null_mut();
            }

            if DEVDAX_OFFSET
                .compare_exchange_weak(curr, end, Ordering::SeqCst, Ordering::Relaxed)
                .is_ok()
            {
                return (base as *mut u8).wrapping_add(aligned);
            }
        }
    }

    /// Allocates an element wrapped inside a generation guard tracking system.
    pub fn alloc_tracked<T>(value: T) -> Result<DaxPtr<T>, AllocError> {
        let layout = Layout::new::<T>();
        let raw = Self::alloc_bump(layout) as *mut T;
        if raw.is_null() {
            return Err(AllocError);
        }
        unsafe {
            ptr::write(raw, value);
        }
        Ok(DaxPtr {
            ptr: raw,
            generation: Self::generation(),
        })
    }

    pub fn reclaim_all() {
        Self::init_if_needed();
        DEVDAX_OFFSET.store(0, Ordering::SeqCst);
        DEVDAX_GENERATION.fetch_add(1, Ordering::SeqCst);
    }

    pub fn reset_epoch() {
        Self::reclaim_all();
    }

    #[inline]
    pub fn generation() -> u64 {
        DEVDAX_GENERATION.load(Ordering::SeqCst)
    }
}

#[cfg(feature = "devdax_bump")]
unsafe impl GlobalAlloc for DevDaxBump {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        Self::alloc_bump(layout)
    }

    unsafe fn dealloc(&self, _ptr: *mut u8, _layout: Layout) {}
}

#[cfg(feature = "devdax_bump")]
unsafe impl Allocator for DevDaxBump {
    fn allocate(&self, layout: Layout) -> Result<NonNull<[u8]>, AllocError> {
        let ptr = Self::alloc_bump(layout);
        if ptr.is_null() {
            return Err(AllocError);
        }
        let slice = unsafe { std::slice::from_raw_parts_mut(ptr, layout.size().max(1)) };
        Ok(NonNull::from(slice))
    }

    unsafe fn deallocate(&self, _ptr: NonNull<u8>, _layout: Layout) {}
}

#[cfg(all(
    feature = "devdax_bump",
    any(
        feature = "global_hashtable_pmem",
        feature = "tiering_hashtable_pmem",
        feature = "eviction_stacks_pmem"
    )
))]
unsafe impl allocator_api2::alloc::Allocator for DevDaxBump {
    fn allocate(&self, layout: Layout) -> Result<NonNull<[u8]>, allocator_api2::alloc::AllocError> {
        let ptr = Self::alloc_bump(layout);
        if ptr.is_null() {
            return Err(allocator_api2::alloc::AllocError);
        }
        let slice = unsafe { std::slice::from_raw_parts_mut(ptr, layout.size().max(1)) };
        Ok(NonNull::from(slice))
    }

    unsafe fn deallocate(&self, _ptr: NonNull<u8>, _layout: Layout) {}
}










impl HybridObjects {
    /// Initialize the UMF pool and prewarm a working-set-sized region.
    /// Call from main() before the benchmark loop.
    pub fn init_and_prewarm(numa_node: i32, prewarm_bytes: usize) {
        //INIT.call_once(|| {
        unsafe { allocator_bindings::umf_allocator_init(numa_node); }
        //#[cfg(debug_assertions)]
        //println!("HybridObjects: UMF pool initialised on NUMA node {}", numa_node);
        //});
        //let chunk = 2 * 1024 * 1024usize;
        //let rc = unsafe { allocator_bindings::umf_allocator_prewarm(numa_node, prewarm_bytes, chunk) };
        //if rc != 0 {
        //    eprintln!("UMF prewarm returned {}", rc);
        //}
    }
    //const NODE: i32 = 1;
}


*/


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
// backend to tier_allocator (the default -- see tiered_buffer.rs).
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

    /// NUMA node backing the slow tier, matching `tiered_buffer.rs`'s own
    /// `SLOW_TIER_NODE` constant (kept separate/duplicated rather than
    /// shared, since this module must compile standalone from
    /// `tiered_buffer.rs`'s own cfg-gated half of that constant).
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
    /// independent of `tier_allocator`'s own UMF/TBB pool for the same
    /// node. A plain, `Copy`, no-state value, matching this file's other
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