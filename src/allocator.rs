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
        #[cfg(debug_assertions)]
        println!("HybridObjects: UMF pool initialised on NUMA node {}", numa_node);
        //});
        let chunk = 2 * 1024 * 1024usize;
        let rc = unsafe { allocator_bindings::umf_allocator_prewarm(numa_node, prewarm_bytes, chunk) };
        if rc != 0 {
            eprintln!("UMF prewarm returned {}", rc);
        }
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

        //INIT.call_once( || { 
        //    HybridObjects::init_and_prewarm(1, 50 * 1024 * 1024 * 1024 )}
        //);


        let ptr = allocator_bindings::umf_alloc(Self::NODE,layout.size(), layout.align()) as *mut u8;
        if ptr.is_null() {
            println!("HybridObjects: UMF alloc failed for {} bytes", layout.size());
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

//static mut NUM_CALLS_DRAM: usize = 0;



impl DRAMObjects {
    /// Initialize the UMF pool and prewarm a working-set-sized region.
    /// Call from main() before the benchmark loop.
    pub fn init_and_prewarm(numa_node: i32, prewarm_bytes: usize) {
        //INIT.call_once(|| {
        unsafe { allocator_bindings::umf_allocator_init(numa_node); }
        #[cfg(debug_assertions)]
        println!("DRAMObjects: UMF pool initialised on NUMA node {}", numa_node);
        //});
        let chunk = 2 * 1024 * 1024usize;
        let rc = unsafe { allocator_bindings::umf_allocator_prewarm(numa_node, prewarm_bytes, chunk) };
        if rc != 0 {
            eprintln!("UMF prewarm returned {}", rc);
        }
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

        INIT.call_once( || { 
            DRAMObjects::init_and_prewarm(Self::NODE_DRAM, 35 * 1024 * 1024 * 1024);

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
        }


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
        feature = "eviction_stacks_pmem",
        feature = "global_flatmap_pmem"
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
        feature = "eviction_stacks_pmem",
        feature = "global_flatmap_pmem"
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


*/