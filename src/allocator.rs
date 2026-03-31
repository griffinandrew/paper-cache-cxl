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
use std::ptr::NonNull;

use allocator_api2::alloc::{AllocError, Allocator};

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

static INIT: Once = Once::new();
static PRINT_THRESHOLD: usize = 10000;
static mut NUM_ALLOCS: usize = 0;
static mut NUM_DEALLOCS: usize = 0;
static ALL_MEM_ALLOCATED: AtomicUsize = AtomicUsize::new(0);

unsafe impl GlobalAlloc for HybridObjects {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        // Initialise the UMF pool on first allocation.  On real hardware this
        // maps a jemalloc pool over the PMEM NUMA node.  In the stub this is
        // a no-op.
        INIT.call_once(|| {
            let numa_node = 1; // PMEM NUMA node (ignored by stub)
            unsafe { allocator_bindings::umf_allocator_init(numa_node) };
            #[cfg(debug_assertions)]
            println!("HybridObjects: UMF pool initialised on NUMA node {}", numa_node);
        });

        let ptr = unsafe { allocator_bindings::umf_alloc(layout.size(), layout.align()) } as *mut u8;
        if ptr.is_null() {
            println!("HybridObjects: UMF alloc failed for {} bytes", layout.size());
            return ptr::null_mut();
        }

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
        unsafe { allocator_bindings::umf_dealloc(ptr as *mut std::ffi::c_void) };

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

// allocator_api2 support (for all PMEM-backed allocations)
unsafe impl Allocator for HybridObjects {
    fn allocate(&self, layout: Layout) -> Result<NonNull<[u8]>, AllocError> {
        let ptr = unsafe { self.alloc(layout) };
        if ptr.is_null() {
            Err(AllocError)
        } else {
            let slice = unsafe { std::slice::from_raw_parts_mut(ptr, layout.size()) };
            Ok(NonNull::from(slice))
        }
    }

    unsafe fn deallocate(&self, ptr: NonNull<u8>, layout: Layout) {
        unsafe { self.dealloc(ptr.as_ptr(), layout) }
    }
}
