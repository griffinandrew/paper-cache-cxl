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
#[cfg(feature = "pmem_region_alloc")]
use std::sync::atomic::AtomicU64;
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
            allocator_bindings::umf_allocator_init(numa_node);
            #[cfg(debug_assertions)]
            println!("HybridObjects: UMF pool initialised on NUMA node {}", numa_node);
        });

        let ptr = allocator_bindings::umf_alloc(layout.size(), layout.align()) as *mut u8;
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
        allocator_bindings::umf_dealloc(ptr as *mut std::ffi::c_void);

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

/// PMEM region allocator:
/// - reserves one large mmap region
/// - attempts NUMA binding to PMEM node
/// - allocates with lock-free bump pointer
/// - deallocate is a no-op (bulk reclaim via `reclaim_all`)
#[cfg(feature = "pmem_region_alloc")]
#[derive(Clone, Copy)]
pub struct RegionHybrid;

#[cfg(feature = "pmem_region_alloc")]
const DEFAULT_PMEM_REGION_BYTES: usize = 32 * 1024 * 1024 * 1024; // 8 GiB virtual region

#[cfg(feature = "pmem_region_alloc")]
const DEFAULT_PMEM_NUMA_NODE: usize = 1;

#[cfg(feature = "pmem_region_alloc")]
static REGION_INIT: Once = Once::new();
#[cfg(feature = "pmem_region_alloc")]
static REGION_BASE_ADDR: AtomicUsize = AtomicUsize::new(0);
#[cfg(feature = "pmem_region_alloc")]
static REGION_SIZE_BYTES: AtomicUsize = AtomicUsize::new(0);
#[cfg(feature = "pmem_region_alloc")]
static REGION_OFFSET: AtomicUsize = AtomicUsize::new(0);
#[cfg(feature = "pmem_region_alloc")]
static REGION_GENERATION: AtomicU64 = AtomicU64::new(0);

#[cfg(feature = "pmem_region_alloc")]
impl RegionHybrid {
    #[inline]
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

#[cfg(feature = "pmem_region_alloc")]
unsafe impl GlobalAlloc for RegionHybrid {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        Self::alloc_bump(layout)
    }

    unsafe fn dealloc(&self, _ptr: *mut u8, _layout: Layout) {
        // Intentionally a no-op: this allocator uses bulk reclamation.
    }
}

#[cfg(feature = "pmem_region_alloc")]
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
    feature = "pmem_region_alloc",
    any(
        feature = "global_hashtable_pmem",
        feature = "tiering_hashtable_pmem",
        feature = "eviction_stacks_pmem",
        feature = "global_flatmap_pmem"
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
