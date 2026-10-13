//! NUMA-bound jemalloc arenas.
//!
//! Gives each NUMA node its own set of jemalloc arenas whose extents are
//! `mmap`ed and then `mbind`ed to that node before jemalloc ever hands them
//! out, so placement is decided by kernel policy at first fault rather than by
//! which CPU happens to touch the page.
//!
//! # What this does and does not guarantee
//!
//! It guarantees placement for memory obtained **through these arenas**. It
//! cannot guarantee anything about the rest of the process: jemalloc here is
//! built with `JEMALLOC_PREFIX=_rjem_` and therefore does not interpose
//! `malloc`, so glibc's heap, every bindgen'd C library and
//! all pthread stacks are outside its reach. Covering those needs a *task*
//! mempolicy -- `numactl --membind=0`, or `set_mempolicy` before any thread is
//! spawned. Nothing here checks for one; [`assert_numa_environment`] and
//! [`numa_environment_warnings`] report the kernel settings that bear on
//! placement.
//!
//! # Design notes
//!
//! Two properties drive the structure, both learned from a previous
//! implementation that got them wrong:
//!
//! 1. **Per-arena config travels in the hooks pointer, not a side table.**
//!    `arenas.create` calls the `alloc` hook for the arena's own metadata
//!    *before it returns*, so a registry keyed on `arena_ind` and populated
//!    afterwards necessarily misses that first call. The previous version
//!    silently fell back to unbound memory there, leaving every arena's
//!    metadata on whatever node the kernel picked. Embedding `extent_hooks_t`
//!    as field 0 of [`NumaHooks`] means the config is recoverable by casting
//!    the pointer jemalloc already passes to every hook, and is therefore
//!    available on that very first call.
//!
//! 2. **Hooks never allocate, never lock, and never panic.** They run under
//!    jemalloc's `grow_mtx`/`ecache` mutexes, and jemalloc explicitly
//!    disclaims reentrancy from them. The previous version held a global
//!    `Mutex<HashMap>` across `mbind` *and* an `eprintln!` whose `io::Error`
//!    formatting allocates -- which, with that allocator installed globally,
//!    re-enters the hook and deadlocks on a non-reentrant mutex, reachable
//!    exactly under node-0 pressure. Everything here is raw syscalls and
//!    atomics.
//!
//! Hook return convention, which is easy to invert: `false` means **success**
//! for every `bool`-returning hook. `alloc` signals failure with null, and
//! `destroy` cannot fail.
//!
//! # Thread caches and the alignment flag
//!
//! Which jemalloc tcache a call names decides placement as surely as the
//! arena does: a tcache hit pops whatever block its bin holds, from whichever
//! arena -- and so node -- it came. The fast tier uses the thread's default
//! tcache and the segregated value pool an explicit one of its own
//! (`value_tcache_flag`). The slow tier uses none (`MALLOCX_TCACHE_NONE`)
//! unless `SLOW_TCACHE=1` gives each thread an explicit slow-only cache
//! (`slow_tcache_flag`, which states the invariant that keeps placement).
//!
//! The alignment flag decides whether a tcache is used at all. With any
//! `MALLOCX_LG_ALIGN`, `mallocx` takes jemalloc's aligned path, which serves a
//! size above `SC_SMALL_MAXCLASS` (14 KiB) straight from the arena
//! (`arena_palloc` -> `large_malloc`) and never looks at the tcache, while
//! `sdallocx` still frees the block INTO it (`arena_dalloc_large` ->
//! `tcache_dalloc_large`). Every value is allocated `align = 8`, so the flag is
//! passed only where jemalloc would not give the alignment anyway
//! ([`align_flags`]); `ALLOC_NATURAL_ALIGN=0` puts it back on every
//! `align > 1` call, the flags this module passed before.

use std::ffi::c_char;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicBool, AtomicU8, AtomicU64, Ordering};

use libc::{c_int, c_uint, c_void, size_t};

// Linkage only: pulls libjemalloc onto the link line so the `_rjem_*`
// symbols declared below resolve.
use tikv_jemalloc_sys as _;

// ---------------------------------------------------------------------------
// jemalloc FFI
// ---------------------------------------------------------------------------

unsafe extern "C" {
	#[link_name = "_rjem_mallctl"]
	fn mallctl(
		name: *const c_char,
		oldp: *mut c_void,
		oldlenp: *mut size_t,
		newp: *mut c_void,
		newlen: size_t,
	) -> c_int;

	#[link_name = "_rjem_mallocx"]
	fn mallocx(size: size_t, flags: c_int) -> *mut c_void;

	#[link_name = "_rjem_sdallocx"]
	fn sdallocx(ptr: *mut c_void, size: size_t, flags: c_int);

	#[link_name = "_rjem_rallocx"]
	fn rallocx(ptr: *mut c_void, size: size_t, flags: c_int) -> *mut c_void;
}

/// `MALLOCX_ARENA(a)` from `jemalloc_macros.h`: `(((int)(a))+1) << 20`.
const fn mallocx_arena(arena: c_uint) -> c_int {
	((arena as c_int) + 1) << 20
}

/// `MALLOCX_TCACHE(tc)`: `(((tc)+2) << 8)`. Note the shift is 8, not 12.
const fn mallocx_tcache(tcache: c_uint) -> c_int {
	((tcache as c_int) + 2) << 8
}

/// `MALLOCX_LG_ALIGN(la)`: the alignment is passed as its base-2 logarithm.
const fn mallocx_lg_align(lg_align: u32) -> c_int {
	lg_align as c_int
}

/// `MALLOCX_ZERO`.
const MALLOCX_ZERO: c_int = 0x40;

// ---------------------------------------------------------------------------
// extent hooks
// ---------------------------------------------------------------------------

/// Mirrors `struct extent_hooks_s` from jemalloc 5.3. Field order is ABI, not
/// style -- it must match the header exactly.
#[repr(C)]
pub struct ExtentHooks {
	alloc: Option<
		unsafe extern "C" fn(
			*mut ExtentHooks,
			*mut c_void,
			size_t,
			size_t,
			*mut bool,
			*mut bool,
			c_uint,
		) -> *mut c_void,
	>,
	dalloc: Option<unsafe extern "C" fn(*mut ExtentHooks, *mut c_void, size_t, bool, c_uint) -> bool>,
	destroy: Option<unsafe extern "C" fn(*mut ExtentHooks, *mut c_void, size_t, bool, c_uint)>,
	commit: Option<
		unsafe extern "C" fn(*mut ExtentHooks, *mut c_void, size_t, size_t, size_t, c_uint) -> bool,
	>,
	decommit: Option<
		unsafe extern "C" fn(*mut ExtentHooks, *mut c_void, size_t, size_t, size_t, c_uint) -> bool,
	>,
	purge_lazy: Option<
		unsafe extern "C" fn(*mut ExtentHooks, *mut c_void, size_t, size_t, size_t, c_uint) -> bool,
	>,
	purge_forced: Option<
		unsafe extern "C" fn(*mut ExtentHooks, *mut c_void, size_t, size_t, size_t, c_uint) -> bool,
	>,
	split: Option<
		unsafe extern "C" fn(*mut ExtentHooks, *mut c_void, size_t, size_t, size_t, bool, c_uint) -> bool,
	>,
	merge: Option<
		unsafe extern "C" fn(
			*mut ExtentHooks,
			*mut c_void,
			size_t,
			*mut c_void,
			size_t,
			bool,
			c_uint,
		) -> bool,
	>,
}

/// Number of 64-bit words in the nodemask handed to `mbind`. 1024 bits is what
/// the kernel accepts as `maxnode` and costs nothing to over-provision.
const NODEMASK_WORDS: usize = 16;
const NODEMASK_BITS: c_ulong_alias = (NODEMASK_WORDS * 64) as c_ulong_alias;

#[allow(non_camel_case_types)]
type c_ulong_alias = libc::c_ulong;

/// Per-arena state. `hooks` **must** stay field 0: every hook recovers this
/// struct by casting the `extent_hooks` pointer jemalloc passes it, which is
/// only valid because the two share an address.
#[repr(C)]
pub struct NumaHooks {
	hooks: ExtentHooks,

	/// `MPOL_BIND | MPOL_F_STATIC_NODES`, precomputed so the hook does no work.
	///
	/// `MPOL_F_STATIC_NODES` belongs in the *mode* argument, not the trailing
	/// `flags` argument -- passing it as flags compiles and then fails with
	/// `EINVAL` on every call. Without it node ids are interpreted relative to
	/// the current cpuset rather than as physical node numbers.
	mode: c_int,
	nodemask: [c_ulong_alias; NODEMASK_WORDS],
	node: u32,

	// Observability. Lock-free by necessity: read from anywhere, written from
	// inside the hooks.
	mapped_bytes: AtomicU64,
	unmapped_bytes: AtomicU64,
	mbind_ok: AtomicU64,
	mbind_failed: AtomicU64,
	alloc_declined: AtomicU64,
}

// The hooks are shared across every thread using the arena; all mutable state
// is atomic and the rest is immutable after construction.
unsafe impl Sync for NumaHooks {}
unsafe impl Send for NumaHooks {}

impl NumaHooks {
	fn new(node: u32) -> Self {
		let mut nodemask = [0 as c_ulong_alias; NODEMASK_WORDS];
		nodemask[(node as usize) / 64] = 1 << ((node as usize) % 64);

		NumaHooks {
			hooks: ExtentHooks {
				alloc: Some(numa_extent_alloc),
				dalloc: Some(numa_extent_dalloc),
				destroy: Some(numa_extent_destroy),
				commit: Some(numa_extent_commit),
				decommit: Some(numa_extent_decommit),
				purge_lazy: Some(numa_extent_purge_lazy),
				purge_forced: Some(numa_extent_purge_forced),
				split: Some(numa_extent_split),
				merge: Some(numa_extent_merge),
			},
			mode: libc::MPOL_BIND | libc::MPOL_F_STATIC_NODES,
			nodemask,
			node,
			mapped_bytes: AtomicU64::new(0),
			unmapped_bytes: AtomicU64::new(0),
			mbind_ok: AtomicU64::new(0),
			mbind_failed: AtomicU64::new(0),
			alloc_declined: AtomicU64::new(0),
		}
	}

	/// Recovers the config from the pointer jemalloc hands every hook.
	///
	/// # Safety
	/// `hooks` must point at a `NumaHooks` -- true for every arena created by
	/// [`create_node_arena`], because `hooks` is field 0.
	#[inline(always)]
	unsafe fn from_ptr<'a>(hooks: *mut ExtentHooks) -> &'a NumaHooks {
		unsafe { &*(hooks as *const NumaHooks) }
	}

	/// Binds `[addr, addr+len)` to this node.
	///
	/// Called on a fresh anonymous mapping whose pages have not been faulted,
	/// so the policy is installed on the VMA before any page exists and every
	/// subsequent fault is placed by it. Raw syscall because `libc` exposes no
	/// `mbind` wrapper.
	#[inline]
	unsafe fn bind(&self, addr: *mut c_void, len: size_t) -> bool {
		let ret = unsafe {
			libc::syscall(
				libc::SYS_mbind,
				addr,
				len as c_ulong_alias,
				self.mode,
				self.nodemask.as_ptr(),
				NODEMASK_BITS,
				0 as c_uint,
			)
		};

		if ret == 0 {
			self.mbind_ok.fetch_add(1, Ordering::Relaxed);
			true
		} else {
			self.mbind_failed.fetch_add(1, Ordering::Relaxed);
			false
		}
	}
}

/// `mmap` `size` bytes aligned to `alignment`, or null.
///
/// Over-maps and trims when `alignment` exceeds a page, which is what jemalloc
/// asks for on its 2 MiB-aligned extents.
unsafe fn map_aligned(size: size_t, alignment: size_t) -> *mut c_void {
	const PROT: c_int = libc::PROT_READ | libc::PROT_WRITE;
	const FLAGS: c_int = libc::MAP_PRIVATE | libc::MAP_ANONYMOUS;

	let page = unsafe { libc::sysconf(libc::_SC_PAGESIZE) } as size_t;

	if alignment <= page {
		let p = unsafe { libc::mmap(std::ptr::null_mut(), size, PROT, FLAGS, -1, 0) };
		return if p == libc::MAP_FAILED { std::ptr::null_mut() } else { p };
	}

	// Over-map so an aligned window is guaranteed to exist inside it, then
	// return the slack to the kernel.
	let Some(span) = size.checked_add(alignment) else {
		return std::ptr::null_mut();
	};

	let base = unsafe { libc::mmap(std::ptr::null_mut(), span, PROT, FLAGS, -1, 0) };
	if base == libc::MAP_FAILED {
		return std::ptr::null_mut();
	}

	let addr = base as usize;
	let aligned = (addr + alignment - 1) & !(alignment - 1);
	let lead = aligned - addr;
	let trail = span - lead - size;

	if lead > 0 {
		unsafe { libc::munmap(base, lead) };
	}
	if trail > 0 {
		unsafe { libc::munmap((aligned + size) as *mut c_void, trail) };
	}

	aligned as *mut c_void
}

unsafe extern "C" fn numa_extent_alloc(
	hooks: *mut ExtentHooks,
	new_addr: *mut c_void,
	size: size_t,
	alignment: size_t,
	zero: *mut bool,
	commit: *mut bool,
	_arena_ind: c_uint,
) -> *mut c_void {
	let cfg = unsafe { NumaHooks::from_ptr(hooks) };

	// Declining a specific address is legal -- jemalloc's contract permits
	// returning NULL -- and honouring it via MAP_FIXED_NOREPLACE was tested
	// and changed nothing, so the simple path stays.
	if !new_addr.is_null() {
		cfg.alloc_declined.fetch_add(1, Ordering::Relaxed);
		return std::ptr::null_mut();
	}

	let addr = unsafe { map_aligned(size, alignment) };
	if addr.is_null() {
		return std::ptr::null_mut();
	}

	// Fail closed. Returning unbound memory here is what turns "guaranteed"
	// into "usually" -- the caller asked for node-local memory and silently
	// getting something else is worse than an allocation failure, which
	// jemalloc will surface.
	if !unsafe { cfg.bind(addr, size) } {
		unsafe { libc::munmap(addr, size) };
		return std::ptr::null_mut();
	}

	cfg.mapped_bytes.fetch_add(size as u64, Ordering::Relaxed);

	// Fresh anonymous pages read as zero, and PROT_READ|PROT_WRITE means the
	// range is committed. Reporting both truthfully lets jemalloc skip work.
	unsafe {
		*zero = true;
		*commit = true;
	}

	addr
}

unsafe extern "C" fn numa_extent_dalloc(
	hooks: *mut ExtentHooks,
	addr: *mut c_void,
	size: size_t,
	_committed: bool,
	_arena_ind: c_uint,
) -> bool {
	let cfg = unsafe { NumaHooks::from_ptr(hooks) };

	// Under `retain`, refuse -- exactly what jemalloc's own hook does:
	//
	//     bool extent_dalloc_mmap(void *addr, size_t size) {
	//         if (!opt_retain) { pages_unmap(addr, size); }
	//         return opt_retain;
	//     }
	//
	// `retain` means jemalloc keeps the address space and reuses it. Unmapping
	// here while jemalloc still believes it owns the range makes the two
	// disagree about what is mapped, and its retained-extent tree eventually
	// walks into memory that is gone -- a SIGSEGV inside
	// `extent_try_coalesce`. It surfaced only with several arenas and a
	// concurrent migration worker (more retained extents, more coalescing),
	// which is why it looked like an arena-count problem and was not.
	//
	// Costs nothing resident: retained extents are decommitted, and peak RSS
	// measured identical either way (12.50 GB vs 12.46 GB over 20M accesses).
	if retain_enabled() {
		return true;
	}

	let rc = unsafe { libc::munmap(addr, size) };

	if rc == 0 {
		cfg.unmapped_bytes.fetch_add(size as u64, Ordering::Relaxed);
	}

	// `false` == success.
	rc != 0
}

/// jemalloc's `opt.retain`, read from jemalloc rather than assumed.
///
/// It defaults on for 64-bit Linux but is configurable, and the `dalloc`
/// contract differs between the two settings, so it is queried.
fn retain_enabled() -> bool {
	static ON: OnceLock<bool> = OnceLock::new();
	*ON.get_or_init(|| {
		let mut value: bool = false;
		let mut sz = size_of::<bool>();
		let rc = unsafe {
			mallctl(
				c"opt.retain".as_ptr(),
				(&raw mut value).cast(),
				&raw mut sz,
				std::ptr::null_mut(),
				0,
			)
		};
		rc == 0 && value
	})
}

unsafe extern "C" fn numa_extent_destroy(
	hooks: *mut ExtentHooks,
	addr: *mut c_void,
	size: size_t,
	_committed: bool,
	_arena_ind: c_uint,
) {
	let cfg = unsafe { NumaHooks::from_ptr(hooks) };
	if unsafe { libc::munmap(addr, size) } == 0 {
		cfg.unmapped_bytes.fetch_add(size as u64, Ordering::Relaxed);
	}
}

unsafe extern "C" fn numa_extent_commit(
	_hooks: *mut ExtentHooks,
	_addr: *mut c_void,
	_size: size_t,
	_offset: size_t,
	_length: size_t,
	_arena_ind: c_uint,
) -> bool {
	// Mapped PROT_READ|PROT_WRITE up front, so always already committed.
	false
}

unsafe extern "C" fn numa_extent_decommit(
	_hooks: *mut ExtentHooks,
	_addr: *mut c_void,
	_size: size_t,
	_offset: size_t,
	_length: size_t,
	_arena_ind: c_uint,
) -> bool {
	// Refuse. Decommitting via munmap or PROT_NONE would drop the VMA and its
	// mbind policy; MADV_DONTNEED would keep both, but measured identically
	// (SET 1446 vs 1454, same peak RSS) because the purge hooks already
	// release the pages. Refusing keeps the policy attached with no cost.
	true
}

unsafe extern "C" fn numa_extent_purge_lazy(
	_hooks: *mut ExtentHooks,
	addr: *mut c_void,
	_size: size_t,
	offset: size_t,
	length: size_t,
	_arena_ind: c_uint,
) -> bool {
	// MADV_FREE keeps the VMA -- and therefore the binding -- intact.
	let rc = unsafe { libc::madvise((addr as usize + offset) as *mut c_void, length, libc::MADV_FREE) };
	rc != 0
}

unsafe extern "C" fn numa_extent_purge_forced(
	_hooks: *mut ExtentHooks,
	addr: *mut c_void,
	_size: size_t,
	offset: size_t,
	length: size_t,
	_arena_ind: c_uint,
) -> bool {
	// MADV_DONTNEED likewise preserves the VMA; pages fault back in under the
	// same policy.
	let rc =
		unsafe { libc::madvise((addr as usize + offset) as *mut c_void, length, libc::MADV_DONTNEED) };
	rc != 0
}

unsafe extern "C" fn numa_extent_split(
	_hooks: *mut ExtentHooks,
	_addr: *mut c_void,
	_size: size_t,
	_size_a: size_t,
	_size_b: size_t,
	_committed: bool,
	_arena_ind: c_uint,
) -> bool {
	// Splitting an anonymous mapping needs no syscall and both halves keep the
	// VMA's policy. Accepting lets jemalloc reuse extents instead of returning
	// them, which is most of the difference between bounded and unbounded RSS.
	false
}

unsafe extern "C" fn numa_extent_merge(
	_hooks: *mut ExtentHooks,
	_addr_a: *mut c_void,
	_size_a: size_t,
	_addr_b: *mut c_void,
	_size_b: size_t,
	_committed: bool,
	_arena_ind: c_uint,
) -> bool {
	// Both extents belong to the same arena and share a policy. Each is an
	// independent `mmap`, so a merge spans two mappings -- legal on Linux,
	// where adjacent anonymous VMAs with the same policy coalesce. Refusing
	// merges was tested against the extent-tree corruption and made no
	// difference, so the permissive answer stands.
	false
}

// ---------------------------------------------------------------------------
// arena pool
// ---------------------------------------------------------------------------

/// Arenas per node.
///
/// **Swept, not guessed.** cluster12, 20M accesses, 15 GB cache / 5 GB fast
/// tier, one run per cell -- SET latency in ns:
///
/// ```text
///   arenas:      1      2      4      8     16     32
///   1 client  1533   1505   1465   1454   1465   1459
///   4        3187   3026   2962   2880   2937   2854
///   8        5089   4769   4613   4551   4575   4485
///   16       8328   7036   6491   6213   6226   6109
/// ```
///
/// Two things decide the value. The cost of a single arena *grows with
/// concurrency* -- 5% at one client, 12% at eight, 27% at sixteen -- because
/// every thread then contends for the same arena on extent growth and slab
/// refill, and the slow tier has no thread cache in front of it to absorb
/// that. And the gains are essentially spent by 8: 8 -> 32 buys 1-2%, inside
/// the run-to-run spread.
///
/// Memory does not push back. Peak RSS and node-0 residency were flat across
/// the sweep (spans 1-13%, no trend), so the retention argument that
/// previously kept this number small is not supported by measurement.
///
/// Caveats: single runs per cell, so trust the monotonic trend rather than any
/// one number; and the 16-client row is oversubscribed on this 8-CPU host, so
/// it mixes scheduler queueing into the figure. The sweep predates
/// `SLOW_TCACHE` (it ran with the slow tier uncached). Override with
/// `NUMA_ARENAS_PER_NODE`.
const DEFAULT_ARENAS_PER_NODE: usize = 8;

/// Upper bound on the per-node arena array. Sizing the array statically keeps
/// the hot path free of indirection; anything above this is clamped.
const MAX_ARENAS_PER_NODE: usize = 32;

fn arenas_per_node() -> usize {
	static COUNT: OnceLock<usize> = OnceLock::new();

	*COUNT.get_or_init(|| {
		std::env::var("NUMA_ARENAS_PER_NODE")
			.ok()
			.and_then(|value| value.parse::<usize>().ok())
			.filter(|count| *count > 0)
			.unwrap_or(DEFAULT_ARENAS_PER_NODE)
			.min(MAX_ARENAS_PER_NODE)
	})
}

/// Node hosting DRAM (the fast tier), and node hosting PMEM/CXL (the slow tier).
///
/// Machine-specific: on mlsys-03 the CXL expander is the CPU-less node 2.
pub const NODE_FAST: u32 = 0;
pub const NODE_SLOW: u32 = 2;

/// Logical index for the segregated value pool. Its arenas are `mbind`ed to
/// the *physical* fast node (`NODE_FAST`) -- this is a DRAM pool, not a third
/// NUMA node -- but it keeps its own arena set and its own explicit tcache so
/// long-lived value buffers never share allocator structures with the transient
/// per-GET `to_vec()` destinations. See the `segregated_value_arena` feature.
///
/// Must never equal a physical node id: `physical_node` remaps this index to
/// DRAM, so a collision would silently put that node's pool on node 0. No
/// node id reaches the nodemask's width.
pub const NODE_FAST_VALUES: u32 = NODEMASK_BITS as u32;

/// The physical NUMA node backing a logical pool index.
#[inline]
fn physical_node(node: u32) -> u32 {
	if node == NODE_FAST_VALUES { NODE_FAST } else { node }
}

struct NodeArenas {
	indices: [c_uint; MAX_ARENAS_PER_NODE],
	hooks: [Option<&'static NumaHooks>; MAX_ARENAS_PER_NODE],
	count: usize,
}

/// Arenas are built per node, on first use of that node.
///
/// Separate `OnceLock`s rather than one pool so a DRAM-only configuration --
/// `FastAlloc` installed globally and `SlowAlloc` never constructed -- never
/// creates node-1 arenas at all. Each arena costs a `mallctl`, a leaked hooks
/// struct, and its own base extent, so unused ones are not free.
static FAST_ARENAS: OnceLock<Option<NodeArenas>> = OnceLock::new();
static SLOW_ARENAS: OnceLock<Option<NodeArenas>> = OnceLock::new();
static VALUE_ARENAS: OnceLock<Option<NodeArenas>> = OnceLock::new();

/// Allocations that could not be routed to a bound arena.
///
/// Must be zero for the placement guarantee to hold. Counted rather than
/// hidden precisely because the previous implementation's equivalent path
/// returned unbound memory silently: an unbound allocation is
/// indistinguishable from a bound one afterwards, so the only chance to notice
/// is at the moment it happens.
static UNBOUND_FALLBACKS: AtomicU64 = AtomicU64::new(0);

/// Enables jemalloc's background purge thread. Process-global, so it is done
/// once for the whole process rather than per node.
static BACKGROUND_THREAD: OnceLock<()> = OnceLock::new();

std::thread_local! {
	/// Which arena slot this thread uses. `Cell<u32>` with a const initialiser
	/// so the thread-local never allocates -- one needing lazy init would
	/// re-enter the allocator on its own first use.
	static SLOT: std::cell::Cell<u32> = const { std::cell::Cell::new(u32::MAX) };

	/// Set while this thread is inside pool construction, so the allocations
	/// `arenas.create` performs do not recurse into it.
	static IN_INIT: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };

	/// This thread's explicit tcache for value-buffer allocations
	/// (`value_tcache_flag`).
	///
	/// Per *thread*, not per arena slot: arenas are lock-protected and safe to
	/// share, tcaches are not, so two threads mapped to the same slot must
	/// still hold different tcaches.
	static VALUE_TCACHE: std::cell::Cell<u32> = const { std::cell::Cell::new(u32::MAX) };

	/// Guards the allocation `tcache.create` itself performs.
	static IN_TCACHE_INIT: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };

	/// This thread's explicit tcache for SLOW-tier allocations and frees
	/// (`slow_tcache_flag`, `SLOW_TCACHE=1`). Its own cache: never
	/// `VALUE_TCACHE` and never the default one, since a cache shared with
	/// another pool would hand that pool's blocks to slow-tier requests.
	/// `u32::MAX` = not created yet; `SLOW_TCACHE_FAILED` = `tcache.create`
	/// failed on this thread, so it runs uncached rather than retrying on
	/// every call.
	static SLOW_TCACHE: std::cell::Cell<u32> = const { std::cell::Cell::new(u32::MAX) };
}

#[cfg(test)]
std::thread_local! {
	/// Allocations this thread has made through the pools (`thread_allocs`).
	/// A `Cell<u64>` with a const initialiser, as the slots above, so that
	/// counting never allocates.
	static THREAD_ALLOCS: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };

	/// Makes `node_arenas` report every pool as unbuilt on this thread, so a
	/// test can reach the unbound fallback after the pools exist. See
	/// `tests::ForceUnbound`.
	static FORCE_UNBOUND: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// How many allocations the calling thread has made through the pools (the
/// global allocator and the explicit slow-tier one), ever: a test that wants
/// to know whether a code path allocates takes the difference around it. Only
/// this thread's, so the cache's workers do not count; and zero-size requests
/// are not allocations.
#[cfg(all(test, feature = "hybrid_cache_common", not(feature = "stock_jemalloc")))]
pub(crate) fn thread_allocs() -> u64 {
	THREAD_ALLOCS.with(std::cell::Cell::get)
}

/// Number of explicit value-pool tcaches created (`value_tcache_flag`), for
/// observability. The slow tier's are counted apart, below.
static TCACHES_CREATED: AtomicU64 = AtomicU64::new(0);

/// Slow-tier tcaches created, and `tcache.create` calls for one that failed
/// (that thread then runs uncached); see [`tcache_report`].
static SLOW_TCACHES_CREATED: AtomicU64 = AtomicU64::new(0);
static SLOW_TCACHE_CREATE_FAILED: AtomicU64 = AtomicU64::new(0);

/// Latched once any slow-pool allocation takes the unbound fallback: from
/// then on the slow tier allocates and frees with `MALLOCX_TCACHE_NONE`, so
/// that unbound block can never enter a slow tcache. See `slow_tcache_flag`.
static SLOW_TCACHE_POISONED: AtomicBool = AtomicBool::new(false);

/// Creates one arena bound to `node`.
fn create_node_arena(node: u32) -> Option<(c_uint, &'static NumaHooks)> {
	// Leaked deliberately: jemalloc requires the hooks struct outlive the
	// arena, and arenas live until process exit.
	let hooks: &'static mut NumaHooks = Box::leak(Box::new(NumaHooks::new(node)));
	let mut hooks_ptr: *mut ExtentHooks = &mut hooks.hooks;

	let mut arena_ind: c_uint = 0;
	let mut sz = size_of::<c_uint>();

	// `newlen` must be exactly `sizeof(extent_hooks_t*)`; ctl's WRITE macro
	// rejects anything else with EINVAL.
	let rc = unsafe {
		mallctl(
			c"arenas.create".as_ptr(),
			(&raw mut arena_ind).cast(),
			&raw mut sz,
			(&raw mut hooks_ptr).cast(),
			size_of::<*mut ExtentHooks>(),
		)
	};

	if rc != 0 {
		return None;
	}

	Some((arena_ind, hooks))
}

fn enable_background_thread() {
	BACKGROUND_THREAD.get_or_init(|| {
		// Decay is otherwise tick-driven, so a burst-then-quiet workload never
		// purges and RSS stays at its high-water mark indefinitely.
		let mut enable = true;
		unsafe {
			mallctl(
				c"background_thread".as_ptr(),
				std::ptr::null_mut(),
				std::ptr::null_mut(),
				(&raw mut enable).cast(),
				size_of::<bool>(),
			);
		}
	});
}

fn build_node_arenas(node: u32) -> Option<NodeArenas> {
	enable_background_thread();

	let count = arenas_per_node();
	let mut indices = [0 as c_uint; MAX_ARENAS_PER_NODE];
	let mut hooks: [Option<&'static NumaHooks>; MAX_ARENAS_PER_NODE] = [None; MAX_ARENAS_PER_NODE];

	for slot in 0..count {
		let (ind, h) = create_node_arena(physical_node(node))?;
		indices[slot] = ind;
		hooks[slot] = Some(h);
	}

	Some(NodeArenas { indices, hooks, count })
}

/// Arenas for `node`, built on first use.
#[inline]
fn node_arenas(node: u32) -> Option<&'static NodeArenas> {
	#[cfg(test)]
	if FORCE_UNBOUND.with(|f| f.get()) {
		return None;
	}

	let cell = if node == NODE_FAST {
		&FAST_ARENAS
	} else if node == NODE_FAST_VALUES {
		&VALUE_ARENAS
	} else {
		&SLOW_ARENAS
	};

	if let Some(built) = cell.get() {
		return built.as_ref();
	}

	// Construction allocates (mallctl, Box::leak). Route those through plain
	// jemalloc rather than recursing into a half-built pool.
	if IN_INIT.with(|f| f.get()) {
		return None;
	}

	IN_INIT.with(|f| f.set(true));
	let built = cell.get_or_init(|| build_node_arenas(node));
	IN_INIT.with(|f| f.set(false));

	built.as_ref()
}

/// Eagerly builds the arenas for `node`.
///
/// Optional -- the first allocation builds them -- but calling it from `main`
/// keeps construction off the measured path. A DRAM-only configuration should
/// call this for [`NODE_FAST`] only.
pub fn init_node(node: u32) -> bool {
	node_arenas(node).is_some()
}

/// Builds arenas for both nodes. Use [`init_node`] instead when only one tier
/// is in play.
pub fn init() -> bool {
	init_node(NODE_FAST) && init_node(NODE_SLOW)
}

#[inline]
fn slot_for_thread(count: usize) -> usize {
	static NEXT: AtomicU64 = AtomicU64::new(0);

	let current = SLOT.with(|s| s.get());
	if current != u32::MAX {
		return (current as usize) % count;
	}

	let assigned = NEXT.fetch_add(1, Ordering::Relaxed) as usize;
	SLOT.with(|s| s.set((assigned % MAX_ARENAS_PER_NODE) as u32));
	assigned % count
}

/// This thread's arena slot as `slot_for_thread` assigned it, or `u32::MAX` on
/// a thread that has not allocated through a bound arena yet.
///
/// For per-thread-sharded counters outside this module (`crate::phys`), which
/// key on the slot the allocator already assigns -- as `measured` does --
/// rather than adding a second thread-local. The shard choice affects
/// contention, never a sum.
#[cfg(feature = "hybrid_cache_common")]
#[inline]
pub(crate) fn arena_slot() -> u32 {
	SLOT.with(|s| s.get())
}

/// `mallocx` flags for a `size`-byte allocation aligned to `align` on `node`.
///
/// The fast tier uses the default per-thread cache and the value pool its
/// own explicit one. The slow tier uses no cache (`MALLOCX_TCACHE_NONE`)
/// unless `SLOW_TCACHE=1` gives it an explicit cache of its own
/// (`slow_tcache_flag`). What is load-bearing is that the slow tier never
/// SHARES a cache: a tcache hands back whatever block it holds without
/// consulting the arena, so one cache serving both nodes would recycle a
/// node-0 block into a node-2 allocation. Nothing crashes -- placement just
/// decays, and only under cross-thread free patterns, which is exactly this
/// workload.
///
/// The alignment bits are [`align_flags`]'s, the same function the free,
/// `realloc` and `measured::usable` use.
#[inline]
fn flags_for(node: u32, align: usize, size: usize) -> Option<c_int> {
	let arenas = node_arenas(node)?;
	let mut flags = mallocx_arena(arenas.indices[slot_for_thread(arenas.count)]);

	if node == NODE_FAST_VALUES {
		flags |= value_tcache_flag();
	} else if node != NODE_FAST {
		flags |= slow_tcache_flag();
	}

	flags |= align_flags(align, size);

	Some(flags)
}

/// `MALLOCX_TCACHE_NONE` is `MALLOCX_TCACHE(-1)`.
const fn mallocx_tcache_none() -> c_int {
	((-1i32) + 2) << 8
}

// ---------------------------------------------------------------------------
// the alignment flag
// ---------------------------------------------------------------------------

/// The largest alignment jemalloc gives a request of at least that many
/// bytes without being asked: its quantum, `1 << LG_QUANTUM`.
///
/// From the jemalloc 5.3.0 source this crate links (tikv-jemalloc-sys 0.6.1):
/// `quantum.h` sets `LG_QUANTUM 4` on x86-64 (its `build.rs` passes
/// `--with-lg-quantum` only for `JEMALLOC_SYS_WITH_LG_QUANTUM`, which nothing
/// here sets). `sc.h`: "all objects of size (1 << LG_QUANTUM) or larger are
/// at least (1 << LG_QUANTUM) aligned"; below that are the tiny classes,
/// powers of two from `1 << SC_LG_TINY_MIN` = 8. `sz_sa2u`, which sizes an
/// aligned request, takes `sz_s2u(ALIGNMENT_CEILING(size, alignment))` for a
/// small class and plain `sz_s2u(size)` for a large one (large extents are
/// `large_malloc`'s `CACHELINE`-aligned at least). So when `align <= 16 &&
/// align <= size` the class `sz_s2u(size)` is already a multiple of `align`,
/// rounding the size up to `align` first lands in that same class, and the
/// flag changes neither the size class nor the usable size -- only the path
/// `mallocx` takes. This is tikv-jemallocator's rule too (`layout_to_flags`:
/// no flag when `align <= ALIGNOF_MAX_ALIGN_T`, 16 on x86-64, `&& align <=
/// size`). `natural_alignment_keeps_every_size_class` reads the quantum and
/// the smallest class back from the linked jemalloc and checks the classes
/// against `nallocx`, rather than trusting this comment.
const NATURAL_ALIGN_MAX: usize = 16;

/// `ALLOC_NATURAL_ALIGN`: `0` passes `MALLOCX_LG_ALIGN` on every `align > 1`
/// call, exactly the flags this module passed before; unset or anything else
/// drops it where jemalloc gives the alignment anyway ([`NATURAL_ALIGN_MAX`]).
/// For A/B runs only.
///
/// Read once, with `libc::getenv` and not `std::env`: the global allocator
/// asks on its very first allocation, and `std::env::var` allocates, which
/// would re-enter here. That first allocation is made before `main` (or the
/// test harness) has spawned a thread, so no `setenv` can race the read.
/// Racing first reads all store the same answer.
#[inline]
fn natural_align_enabled() -> bool {
	match NATURAL_ALIGN_STATE.load(Ordering::Relaxed) {
		1 => false,
		2 => true,
		_ => read_natural_align(),
	}
}

/// [`natural_align_enabled`]'s latch: 0 = not read yet, 1 = off, 2 = on.
static NATURAL_ALIGN_STATE: AtomicU8 = AtomicU8::new(0);

#[cold]
fn read_natural_align() -> bool {
	let off = unsafe {
		let raw = libc::getenv(c"ALLOC_NATURAL_ALIGN".as_ptr());
		!raw.is_null() && *raw as u8 == b'0' && *raw.add(1) == 0
	};
	NATURAL_ALIGN_STATE.store(if off { 1 } else { 2 }, Ordering::Relaxed);
	!off
}

/// The `MALLOCX_LG_ALIGN` bits for a `size`-byte request aligned to `align`,
/// with natural alignment on or off: none for `align <= 1`, none when
/// `natural` and jemalloc's size class already guarantees `align`
/// ([`NATURAL_ALIGN_MAX`]), else `MALLOCX_LG_ALIGN(log2 align)`.
#[inline]
const fn align_flags_with(natural: bool, align: usize, size: usize) -> c_int {
	if align <= 1 || (natural && align <= NATURAL_ALIGN_MAX && align <= size) {
		0
	} else {
		mallocx_lg_align(align.trailing_zeros())
	}
}

/// The `MALLOCX_LG_ALIGN` bits every `mallocx`, `rallocx`, `sdallocx` and
/// `nallocx` this crate makes for a `size`-byte request aligned to `align`
/// carries, both tiers and every pool alike, so a block is always allocated,
/// resized, freed and counted under one size class. See the module docs for
/// why the flag is left off where it can be.
#[inline]
pub(crate) fn align_flags(align: usize, size: usize) -> c_int {
	align_flags_with(natural_align_enabled(), align, size)
}

// ---------------------------------------------------------------------------
// allocators
// ---------------------------------------------------------------------------

/// Allocator bound to a specific NUMA node.
///
/// `NODE_FAST` is usable as `#[global_allocator]`; `NODE_SLOW` is meant for
/// explicit placement (`Box::new_in`, `Vec::new_in`).
#[derive(Clone, Copy, Default, Debug)]
pub struct NumaAlloc<const NODE: u32>;

pub type FastAlloc = NumaAlloc<NODE_FAST>;
pub type SlowAlloc = NumaAlloc<NODE_SLOW>;

/// The slow-tier allocator as a plain unit struct.
///
/// `SlowAlloc` is a type alias to a generic struct, and an alias lives only in
/// the type namespace -- so `Box<[u8], SlowAlloc>` resolves but
/// `Box::new_in(x, SlowAlloc)` does not. The crate-wide `Hybrid` name is used
/// in both positions (it replaced an ordinary unit struct),
/// so the replacement has to occupy both namespaces too. Delegates
/// everything to `NumaAlloc<NODE_SLOW>`.
#[derive(Clone, Copy, Default)]
pub struct SlowObjects;

unsafe impl std::alloc::GlobalAlloc for SlowObjects {
	unsafe fn alloc(&self, layout: std::alloc::Layout) -> *mut u8 {
		unsafe { std::alloc::GlobalAlloc::alloc(&NumaAlloc::<NODE_SLOW>, layout) }
	}

	unsafe fn alloc_zeroed(&self, layout: std::alloc::Layout) -> *mut u8 {
		unsafe { std::alloc::GlobalAlloc::alloc_zeroed(&NumaAlloc::<NODE_SLOW>, layout) }
	}

	unsafe fn realloc(
		&self,
		ptr: *mut u8,
		layout: std::alloc::Layout,
		new_size: usize,
	) -> *mut u8 {
		unsafe { std::alloc::GlobalAlloc::realloc(&NumaAlloc::<NODE_SLOW>, ptr, layout, new_size) }
	}

	unsafe fn dealloc(&self, ptr: *mut u8, layout: std::alloc::Layout) {
		unsafe { std::alloc::GlobalAlloc::dealloc(&NumaAlloc::<NODE_SLOW>, ptr, layout) }
	}
}

unsafe impl std::alloc::Allocator for SlowObjects {
	fn allocate(
		&self,
		layout: std::alloc::Layout,
	) -> Result<std::ptr::NonNull<[u8]>, std::alloc::AllocError> {
		std::alloc::Allocator::allocate(&NumaAlloc::<NODE_SLOW>, layout)
	}

	fn allocate_zeroed(
		&self,
		layout: std::alloc::Layout,
	) -> Result<std::ptr::NonNull<[u8]>, std::alloc::AllocError> {
		std::alloc::Allocator::allocate_zeroed(&NumaAlloc::<NODE_SLOW>, layout)
	}

	unsafe fn deallocate(&self, ptr: std::ptr::NonNull<u8>, layout: std::alloc::Layout) {
		unsafe { std::alloc::Allocator::deallocate(&NumaAlloc::<NODE_SLOW>, ptr, layout) }
	}
}

unsafe impl allocator_api2::alloc::Allocator for SlowObjects {
	fn allocate(
		&self,
		layout: std::alloc::Layout,
	) -> Result<std::ptr::NonNull<[u8]>, allocator_api2::alloc::AllocError> {
		allocator_api2::alloc::Allocator::allocate(&NumaAlloc::<NODE_SLOW>, layout)
	}

	fn allocate_zeroed(
		&self,
		layout: std::alloc::Layout,
	) -> Result<std::ptr::NonNull<[u8]>, allocator_api2::alloc::AllocError> {
		allocator_api2::alloc::Allocator::allocate_zeroed(&NumaAlloc::<NODE_SLOW>, layout)
	}

	unsafe fn deallocate(&self, ptr: std::ptr::NonNull<u8>, layout: std::alloc::Layout) {
		unsafe { allocator_api2::alloc::Allocator::deallocate(&NumaAlloc::<NODE_SLOW>, ptr, layout) }
	}
}

// SAFETY: a stateless handle to the process-wide slow arenas, so any copy can free what
// another allocated, and cloning or dropping one can neither unwind nor free anything.
unsafe impl std::alloc::AllocatorClone for SlowObjects {}


/// Live allocated bytes per pool, counted the way Redis's `zmalloc` counts --
/// per allocation, not sampled.
///
/// # Why not `mallctl`
///
/// `stats.arenas.<i>.small.allocated` is `curregs * size_class`, and `curregs`
/// moves when a TCACHE BATCH is filled or flushed, not when the application
/// allocates. The fast tier here runs on the default tcache and the slow tier
/// on `MALLOCX_TCACHE_NONE` (unless `SLOW_TCACHE=1`), so sampling is exact
/// for the slow node only while it runs uncached, and inexact for node 0 --
/// the tier being bounded -- always. The error scales with thread count and
/// free burstiness, so it cannot be calibrated away. Refreshing `epoch` also
/// takes a process-global lock and walks every arena's bins.
///
/// # What it counts
///
/// Size-class-rounded usable bytes, via `nallocx`. Because that is a pure
/// function of `(size, align)`, the add and the subtract cancel exactly with
/// no per-pointer side table -- `dealloc` recomputes it from the `Layout` it
/// already holds.
///
/// # Scope
///
/// RUST-allocated bytes through these allocators, not process RSS and not
/// resident DRAM. jemalloc is built `JEMALLOC_PREFIX=_rjem_` and does not
/// interpose malloc, so glibc's heap and pthread stacks are outside it. The
/// gap to residency is jemalloc's retained dirty pages and fragmentation.
#[cfg(feature = "measured_accounting")]
pub mod measured {
	use std::sync::atomic::{AtomicI64, Ordering};

	use super::{align_flags, NODE_FAST, NODE_FAST_VALUES, NODE_SLOW, SLOT};

	/// One per pool: fast (node 0), slow (`NODE_SLOW`), fast values (node 0 too --
	/// a separate ARENA set, not a separate node; see `physical_node`).
	const POOLS: usize = 3;

	/// Sharded so concurrent connections do not serialise on one line.
	const SHARDS: usize = 16;

	/// Cache-line isolated: without this, sixteen shards share two lines and
	/// the sharding buys nothing.
	#[repr(align(64))]
	struct Padded(AtomicI64);

	/// SIGNED, and only the SUM is meaningful. Allocate-on-A / free-on-B is
	/// ordinary here (a migration consumer frees what a worker allocated), so
	/// an individual shard legitimately goes negative.
	static POOL_BYTES: [[Padded; SHARDS]; POOLS] = [
		[const { Padded(AtomicI64::new(0)) }; SHARDS],
		[const { Padded(AtomicI64::new(0)) }; SHARDS],
		[const { Padded(AtomicI64::new(0)) }; SHARDS],
	];

	/// What jemalloc will actually hand back for this request.
	///
	/// Takes `(size, align)` rather than a `Layout` so `realloc` can ask about
	/// its new size without constructing one that might not be valid.
	///
	/// NOTE the align bits. `object::overhead::resident_value_bytes` calls
	/// `nallocx(n, 0)`, which is correct only for `align <= 8`; copying that
	/// here would mis-round every over-aligned allocation. They are
	/// `align_flags`', the bits the allocation itself carried; where natural
	/// alignment leaves them off the class is the flagged one anyway
	/// (`natural_alignment_keeps_every_size_class`).
	#[inline]
	pub fn usable(size: usize, align: usize) -> usize {
		if size == 0 {
			return 0;
		}

		unsafe { tikv_jemalloc_sys::nallocx(size, align_flags(align, size)) }
	}

	/// Reuses the arena slot the allocator already assigns per thread rather
	/// than adding a second thread-local. An unassigned thread (one that only
	/// ever frees) reads `u32::MAX` and lands in the last shard, which is
	/// fine: the shard choice affects contention, never the sum.
	#[inline]
	fn shard() -> usize {
		(SLOT.with(|s| s.get()) as usize) % SHARDS
	}

	#[inline]
	const fn pool_of(node: u32) -> usize {
		match node {
			NODE_FAST => 0,
			NODE_SLOW => 1,
			_ => 2,
		}
	}

	#[inline]
	pub fn charge<const NODE: u32>(delta: i64) {
		POOL_BYTES[pool_of(NODE)][shard()].0.fetch_add(delta, Ordering::Relaxed);
	}

	/// Live bytes in one pool. Clamped at zero: a torn read across shards can
	/// momentarily show a negative sum, which is a reporting artifact rather
	/// than a real state.
	pub fn allocated(node: u32) -> u64 {
		let sum: i64 = POOL_BYTES[pool_of(node)]
			.iter()
			.map(|p| p.0.load(Ordering::Relaxed))
			.sum();

		sum.max(0) as u64
	}

	/// Live DRAM bytes: node 0 plus the segregated value pool, which is a
	/// different arena set on the SAME physical node.
	pub fn dram_allocated() -> u64 {
		allocated(NODE_FAST) + allocated(NODE_FAST_VALUES)
	}

	/// Live slow-tier bytes.
	pub fn slow_allocated() -> u64 {
		allocated(NODE_SLOW)
	}

	/// The RAW signed sum, unclamped.
	///
	/// `allocated` clamps at zero because a torn read across shards can
	/// momentarily show a negative total, which is a reporting artifact. That
	/// clamp also hides a PERSISTENT negative drift -- the exact failure an
	/// uninstrumented `realloc` produces -- so the drift tests read this
	/// instead. Verified by removing the realloc charge: against `allocated`
	/// the test stayed green, against this one it fails.
	#[cfg(test)]
	pub fn raw_sum(node: u32) -> i64 {
		POOL_BYTES[pool_of(node)]
			.iter()
			.map(|p| p.0.load(Ordering::Relaxed))
			.sum()
	}
}

impl<const NODE: u32> NumaAlloc<NODE> {
	#[inline]
	unsafe fn raw_alloc(&self, layout: std::alloc::Layout, zeroed: bool) -> *mut u8 {
		if layout.size() == 0 {
			return layout.align() as *mut u8;
		}

		// Test builds only: one more allocation on this thread (`thread_allocs`).
		#[cfg(test)]
		THREAD_ALLOCS.with(|count| count.set(count.get() + 1));

		let ptr = match flags_for(NODE, layout.align(), layout.size()) {
			Some(mut flags) => {
				if zeroed {
					flags |= MALLOCX_ZERO;
				}
				unsafe { mallocx(layout.size(), flags) as *mut u8 }
			},

			// Pool not up yet (process startup, or reentry from inside its own
			// construction). Counted, never silent -- see UNBOUND_FALLBACKS.
			None => {
				UNBOUND_FALLBACKS.fetch_add(1, Ordering::Relaxed);
				let mut flags = if zeroed { MALLOCX_ZERO } else { 0 };
				// An unbound block now exists in the slow pool, and it must
				// never be freed into a slow tcache: latch before the pointer
				// is returned (see `slow_tcache_flag`). And uncached: with no
				// tcache flag this would pop whatever the thread's DEFAULT
				// tcache -- the fast tier's -- holds, a node-0 block.
				if NODE != NODE_FAST && NODE != NODE_FAST_VALUES {
					poison_slow_tcache();
					flags |= mallocx_tcache_none();
				}
				// `dealloc` frees with `align_flags`, so the fallback must
				// allocate with them too, or the block is freed under another
				// size class.
				flags |= align_flags(layout.align(), layout.size());
				unsafe { mallocx(layout.size(), flags) as *mut u8 }
			},
		};

		// Both arms, and only on success: a failed allocation owns nothing.
		// Every allocating method in all three trait impls delegates here, and
		// `SlowObjects`/`FastValues` are one-line forwards, so this one site
		// covers the whole add side.
		#[cfg(feature = "measured_accounting")]
		if !ptr.is_null() {
			measured::charge::<NODE>(measured::usable(layout.size(), layout.align()) as i64);
		}

		ptr
	}
}

unsafe impl<const NODE: u32> std::alloc::GlobalAlloc for NumaAlloc<NODE> {
	#[inline]
	unsafe fn alloc(&self, layout: std::alloc::Layout) -> *mut u8 {
		unsafe { self.raw_alloc(layout, false) }
	}

	#[inline]
	unsafe fn alloc_zeroed(&self, layout: std::alloc::Layout) -> *mut u8 {
		unsafe { self.raw_alloc(layout, true) }
	}

	#[inline]
	unsafe fn dealloc(&self, ptr: *mut u8, layout: std::alloc::Layout) {
		if layout.size() == 0 {
			return;
		}

		// The arena is recovered from the extent's own metadata, so the arena
		// flag is not needed on free and would be wrong for a block allocated
		// by a thread in a different slot. The tcache flag *does* matter, and
		// must mirror the alloc path exactly: it decides which cache the block
		// lands in. A slow-tier block freed into the default cache would put a
		// node-2 block where a node-0 allocation could take it, and a value
		// block freed there would reintroduce the very mixing the value pool
		// exists to prevent. The slow tier frees into its own tcache under
		// `SLOW_TCACHE=1` (`MALLOCX_TCACHE_NONE` otherwise) -- the same flag
		// its alloc and realloc use; this and `rallocx`'s free of the old block
		// are the only ways into that cache, and both only ever see blocks the
		// slow pool handed out.
		let tcache = if NODE == NODE_FAST {
			0
		} else if NODE == NODE_FAST_VALUES {
			value_tcache_flag()
		} else {
			slow_tcache_flag()
		};

		let flags = tcache | align_flags(layout.align(), layout.size());

		// Mirror of the add in `raw_alloc`: `usable` is a pure function of
		// (size, align), so this cancels the birth charge exactly without any
		// per-pointer bookkeeping. Both `Allocator::deallocate` impls forward
		// here rather than duplicating the sdallocx logic, so this one site
		// covers the whole subtract side.
		#[cfg(feature = "measured_accounting")]
		measured::charge::<NODE>(-(measured::usable(layout.size(), layout.align()) as i64));

		unsafe { sdallocx(ptr as *mut c_void, layout.size(), flags) }
	}

	#[inline]
	unsafe fn realloc(
		&self,
		ptr: *mut u8,
		layout: std::alloc::Layout,
		new_size: usize,
	) -> *mut u8 {
		if layout.size() == 0 || new_size == 0 {
			// Degenerate cases: fall back to the generic alloc/copy/free path
			// rather than special-casing rallocx.
			let new_layout = match std::alloc::Layout::from_size_align(new_size, layout.align()) {
				Ok(l) => l,
				Err(_) => return std::ptr::null_mut(),
			};

			let fresh = unsafe { self.raw_alloc(new_layout, false) };
			if !fresh.is_null() && layout.size() > 0 {
				unsafe {
					std::ptr::copy_nonoverlapping(ptr, fresh, layout.size().min(new_size));
					self.dealloc(ptr, layout);
				}
			}
			return fresh;
		}

		// THE THIRD POINT, and the one whose absence is silent. `rallocx` frees
		// the old block and returns a new one without going through either
		// `raw_alloc` or `dealloc`, so a counter that instruments only those
		// two drifts NEGATIVE without bound -- a Vec growing 32 -> 64 is
		// charged nallocx(32) at birth and credited nallocx(64) at death.
		//
		// Charged INSIDE these two arms rather than at function entry: the
		// degenerate branch above already routes through `raw_alloc` and
		// `self.dealloc`, so an entry hook would double count it.
		// The alignment bits are the NEW block's, `align_flags(align,
		// new_size)`: that is the block `dealloc` will free (rallocx frees the
		// old one itself, by its looked-up size).
		let new_ptr = match flags_for(NODE, layout.align(), new_size) {
			Some(flags) => unsafe { rallocx(ptr as *mut c_void, new_size, flags) as *mut u8 },
			None => {
				UNBOUND_FALLBACKS.fetch_add(1, Ordering::Relaxed);
				// Same reason as the fallback in `raw_alloc`.
				let mut flags = align_flags(layout.align(), new_size);
				// As in `raw_alloc`: the new block is unbound, and must not come
				// from the default tcache. Here it matters twice: `rallocx` that
				// moves the block frees the OLD one -- a slow-arena, node-2
				// block -- into the tcache the call names, and the default one
				// would hand it to the fast tier's next allocation of its class.
				if NODE != NODE_FAST && NODE != NODE_FAST_VALUES {
					poison_slow_tcache();
					flags |= mallocx_tcache_none();
				}
				unsafe { rallocx(ptr as *mut c_void, new_size, flags) as *mut u8 }
			},
		};

		// Only on success: a failed `rallocx` leaves the OLD block live and
		// owned, so its charge must stand.
		#[cfg(feature = "measured_accounting")]
		if !new_ptr.is_null() {
			let before = measured::usable(layout.size(), layout.align()) as i64;
			let after = measured::usable(new_size, layout.align()) as i64;
			measured::charge::<NODE>(after - before);
		}

		new_ptr
	}
}

unsafe impl<const NODE: u32> std::alloc::Allocator for NumaAlloc<NODE> {
	fn allocate(
		&self,
		layout: std::alloc::Layout,
	) -> Result<std::ptr::NonNull<[u8]>, std::alloc::AllocError> {
		let ptr = unsafe { self.raw_alloc(layout, false) };
		std::ptr::NonNull::new(ptr)
			.map(|p| std::ptr::NonNull::slice_from_raw_parts(p, layout.size()))
			.ok_or(std::alloc::AllocError)
	}

	fn allocate_zeroed(
		&self,
		layout: std::alloc::Layout,
	) -> Result<std::ptr::NonNull<[u8]>, std::alloc::AllocError> {
		let ptr = unsafe { self.raw_alloc(layout, true) };
		std::ptr::NonNull::new(ptr)
			.map(|p| std::ptr::NonNull::slice_from_raw_parts(p, layout.size()))
			.ok_or(std::alloc::AllocError)
	}

	unsafe fn deallocate(&self, ptr: std::ptr::NonNull<u8>, layout: std::alloc::Layout) {
		unsafe { std::alloc::GlobalAlloc::dealloc(self, ptr.as_ptr(), layout) }
	}
}

/// The same allocator through `allocator_api2`'s trait.
///
/// The PMEM collections (`hashbrown` maps and vectors parameterised by an
/// allocator) take `allocator_api2::alloc::Allocator`, not the unstable std
/// one, so serving them requires both -- the previous allocator implemented
/// this trait too.
unsafe impl<const NODE: u32> allocator_api2::alloc::Allocator for NumaAlloc<NODE> {
	fn allocate(
		&self,
		layout: std::alloc::Layout,
	) -> Result<std::ptr::NonNull<[u8]>, allocator_api2::alloc::AllocError> {
		let ptr = unsafe { self.raw_alloc(layout, false) };
		std::ptr::NonNull::new(ptr)
			.map(|p| std::ptr::NonNull::slice_from_raw_parts(p, layout.size()))
			.ok_or(allocator_api2::alloc::AllocError)
	}

	fn allocate_zeroed(
		&self,
		layout: std::alloc::Layout,
	) -> Result<std::ptr::NonNull<[u8]>, allocator_api2::alloc::AllocError> {
		let ptr = unsafe { self.raw_alloc(layout, true) };
		std::ptr::NonNull::new(ptr)
			.map(|p| std::ptr::NonNull::slice_from_raw_parts(p, layout.size()))
			.ok_or(allocator_api2::alloc::AllocError)
	}

	unsafe fn deallocate(&self, ptr: std::ptr::NonNull<u8>, layout: std::alloc::Layout) {
		unsafe { std::alloc::GlobalAlloc::dealloc(self, ptr.as_ptr(), layout) }
	}
}

// ---------------------------------------------------------------------------
// environment checks and observability
// ---------------------------------------------------------------------------

/// Binds the **calling thread** to `node` for everything it subsequently
/// faults -- its own stack growth included -- without touching any other
/// thread's policy.
///
/// `set_mempolicy` is per-thread in Linux, not per-process, so this is a
/// scalpel rather than the `numactl --membind` hammer: verified with a
/// three-thread probe where a self-bound thread's stack landed on node 1
/// while the main thread's policy and stack stayed on node 0.
///
/// Two limits worth knowing. It governs *faults*, so pages this thread has
/// already touched do not move (the first few stack pages fault during
/// pthread startup, before any of this runs). And it applies to allocations
/// that reach the kernel through this thread -- memory served from a
/// jemalloc arena was already bound by that arena's extent hooks, and is
/// unaffected.
///
/// That second point is what makes binding safe for the migration workers: an
/// explicit VMA policy from `mbind` overrides a thread's `set_mempolicy`
/// default, so a consumer bound to node 0 still writes node-1 memory when it
/// migrates. Verified: slow-tier bytes were identical to seven significant
/// figures across unbound, bound-to-0 and bound-to-1 runs, at 2.58M
/// demotions each.
///
/// Installs the policy unconditionally for `node` and returns whether the
/// kernel accepted it. The `PAPER_BIND_WORKERS` gating lives in
/// `worker_bind_node`, not here.
pub fn bind_current_thread(node: u32) -> bool {
	let mut nodemask = [0 as c_ulong_alias; NODEMASK_WORDS];
	nodemask[(node as usize) / 64] |= 1 << ((node as usize) % 64);

	let rc = unsafe {
		libc::syscall(
			libc::SYS_set_mempolicy,
			(libc::MPOL_BIND | libc::MPOL_F_STATIC_NODES) as c_int,
			nodemask.as_ptr(),
			NODEMASK_BITS,
		)
	};

	if rc == 0 {
		THREADS_BOUND.fetch_add(1, Ordering::Relaxed);
		true
	} else {
		false
	}
}

/// Which node the cache's own threads bind themselves to.
///
/// Node 0 by default. Measured at n=6 per configuration on cluster7 and
/// cluster23 (20M accesses, 15 GB / 1 GB, configurations interleaved within
/// each rep, exact permutation test): binding to node 0 costs nothing at any
/// percentile -- every effect is under 1% with sign counts at chance, at one
/// and at eight clients.
///
/// Binding to node 1 is a different story: client GET p90/p95/p99 rise by
/// 3.4-8.7%, all six reps separating, p=0.002 at the design floor, at both
/// client counts. Node 1 is CPU-less here, so a thread bound to it takes every
/// allocation and stack fault across the interconnect. The cost appears only
/// in the tail -- p50 does not move at all -- so a median-and-mean reading of
/// the very same runs looks like a null.
///
/// An earlier n=3 measurement recorded a ~7% SET *improvement* at 8 clients,
/// equal for either node. That does not survive. At n=3v3 the exact
/// permutation p-floor is 0.10, so the reported p=0.10 was the smallest value
/// the test could emit and carried no evidence; the effect did not reproduce
/// at n=6, and reversed sign between traces.
///
/// Node 0 is the default because that is where these threads' own scratch and
/// stack growth belong; migrations are unaffected either way (see below).
///
/// `PAPER_BIND_WORKERS=1` targets `NODE_SLOW`; any other value disables binding.
///
/// Note this is `MPOL_BIND`, a hard constraint: if node 0 fills, an unpolicied
/// allocation on a bound thread is OOM-killed rather than spilling
/// (`CONSTRAINT_MEMORY_POLICY`). The exposure is small -- stacks and glibc
/// only, since arena memory carries its own VMA policy -- but it is not zero.
pub fn worker_bind_node() -> Option<u32> {
	static NODE: OnceLock<Option<u32>> = OnceLock::new();
	*NODE.get_or_init(|| unsafe {
		let raw = libc::getenv(c"PAPER_BIND_WORKERS".as_ptr());
		if raw.is_null() {
			return Some(NODE_FAST);
		}
		match *raw as u8 {
			b'0' => Some(NODE_FAST),
			b'1' => Some(NODE_SLOW),
			_ => None,
		}
	})
}

/// Convenience for thread entry points: bind if configured, else do nothing.
pub fn bind_worker_thread_if_configured() {
	if let Some(node) = worker_bind_node() {
		bind_current_thread(node);
	}
}

/// Threads that installed a policy, for observability.
pub static THREADS_BOUND: AtomicU64 = AtomicU64::new(0);

/// Kernel settings that can silently move pages off the node they were bound to.
///
/// The failure is not observable from inside the process, and the knob is
/// writable at runtime, so it is checked rather than assumed. `mbind` binds
/// the VMA, but page *migration* is a separate mechanism that operates on
/// already-placed pages.
///
/// Only settings that can move pages bound through these arenas are errors.
/// `kernel.numa_balancing` used to be one; it cannot, and is now reported by
/// [`numa_environment_warnings`] instead.
pub fn assert_numa_environment() -> Result<(), String> {
	let mut problems = Vec::new();

	// A CPU-less node is exactly what the kernel treats as a demotion target,
	// which makes this the relevant knob for a PMEM/CXL slow tier.
	if sysfs_flag_is_on("/sys/kernel/mm/numa/demotion_enabled") {
		problems.push(
			"numa demotion_enabled=true -- cold pages on node 0 may be demoted to the \
			 CPU-less node under pressure"
				.to_string(),
		);
	}

	if problems.is_empty() { Ok(()) } else { Err(problems.join("; ")) }
}

/// Kernel settings worth reporting that do NOT move pages bound through these
/// arenas.
///
/// `kernel.numa_balancing` (any mode): the balancing scanner skips every VMA
/// whose mempolicy lacks `MPOL_F_MOF`, and an `mbind(MPOL_BIND)` without
/// `MPOL_F_NUMA_BALANCING` -- which is how every extent here is bound
/// (`the_arenas_do_not_opt_into_numa_balancing`) -- never sets it. So the
/// arenas' pages stay where they were bound. Measured on
/// mlsys-03 (kernel 6.8, `numa_balancing=1`): a 128 MiB region bound to node 2
/// with this module's flags kept all 32,768 pages there through 75 s of access
/// from node-0 CPUs, while an unbound control region migrated from node 1 to
/// node 0 in full; and across the unpinned cluster35 matrix runs node 1 never
/// held more than ~74 MB of a ~16 GB process.
///
/// What balancing can still do is migrate memory outside these arenas (glibc's
/// heap, pthread stacks, `unbound_fallbacks`) and move threads between nodes,
/// which shows up as timing rather than placement. Pin the process
/// (`numactl --cpunodebind=0`) when that matters.
pub fn numa_environment_warnings() -> Vec<String> {
	let mut warnings = Vec::new();

	let balancing = read_sysfs("/proc/sys/kernel/numa_balancing")
		.and_then(|value| value.parse::<i32>().ok());

	if let Some(mode) = balancing && mode != 0 {
		warnings.push(format!(
			"kernel.numa_balancing={mode} -- does not move pages bound through these \
			 arenas, but can migrate unbound memory and move threads between nodes"
		));
	}

	warnings
}

/// A sysfs or procfs value, trimmed; `None` when the file is absent or unreadable.
fn read_sysfs(path: &str) -> Option<String> {
	Some(std::fs::read_to_string(path).ok()?.trim().to_owned())
}

/// Whether a boolean kernel knob is on.
///
/// Boolean knobs are not spelled consistently: procfs ones read `0`/`1`, but
/// `/sys/kernel/mm/numa/demotion_enabled` reads `true`/`false`. The check this
/// replaced parsed an integer, so on the kernels that spell it `true` it could
/// never fire.
fn sysfs_flag_is_on(path: &str) -> bool {
	read_sysfs(path).is_some_and(|value| flag_value_is_on(&value))
}

fn flag_value_is_on(value: &str) -> bool {
	matches!(value, "1" | "true" | "y" | "Y" | "yes")
}

/// Per-node counters, plus the number of allocations that missed the bound
/// arenas entirely.
pub fn stats() -> String {
	fn sum(arenas: Option<&NodeArenas>) -> (u64, u64, u64, u64, u64) {
		let Some(arenas) = arenas else { return (0, 0, 0, 0, 0) };
		let mut mapped = 0;
		let mut unmapped = 0;
		let mut ok = 0;
		let mut failed = 0;
		let mut declined = 0;

		for hooks in arenas.hooks.iter().flatten() {
			mapped += hooks.mapped_bytes.load(Ordering::Relaxed);
			unmapped += hooks.unmapped_bytes.load(Ordering::Relaxed);
			ok += hooks.mbind_ok.load(Ordering::Relaxed);
			failed += hooks.mbind_failed.load(Ordering::Relaxed);
			declined += hooks.alloc_declined.load(Ordering::Relaxed);
		}

		(mapped, unmapped, ok, failed, declined)
	}

	let (fm, fu, fok, ffail, fdec) = sum(FAST_ARENAS.get().and_then(|a| a.as_ref()));
	let (sm, su, sok, sfail, sdec) = sum(SLOW_ARENAS.get().and_then(|a| a.as_ref()));

	format!(
		"NUMAALLOC fast[mapped={fm} unmapped={fu} live={} mbind_ok={fok} mbind_failed={ffail} declined={fdec}] \
slow[mapped={sm} unmapped={su} live={} mbind_ok={sok} mbind_failed={sfail} declined={sdec}] \
unbound_fallbacks={} tcaches={}",
		fm.saturating_sub(fu),
		sm.saturating_sub(su),
		UNBOUND_FALLBACKS.load(Ordering::Relaxed),
		TCACHES_CREATED.load(Ordering::Relaxed),
	)
}

/// Ground-truth placement, read from the kernel rather than inferred.
///
/// Walks `/proc/self/numa_maps` and totals the resident pages this process has
/// on the fast and slow nodes, returned as `(fast, slow)`. Unlike the counters
/// above -- which record what was *asked for* -- this reports where pages
/// actually are, so the two disagreeing is itself the signal.
pub fn resident_pages_per_node() -> Result<(u64, u64), std::io::Error> {
	let maps = std::fs::read_to_string("/proc/self/numa_maps")?;
	let fast_prefix = format!("N{NODE_FAST}=");
	let slow_prefix = format!("N{NODE_SLOW}=");
	let mut fast = 0u64;
	let mut slow = 0u64;

	for line in maps.lines() {
		for field in line.split_whitespace() {
			if let Some(count) = field.strip_prefix(fast_prefix.as_str()) {
				fast += count.parse::<u64>().unwrap_or(0);
			} else if let Some(count) = field.strip_prefix(slow_prefix.as_str()) {
				slow += count.parse::<u64>().unwrap_or(0);
			}
		}
	}

	Ok((fast, slow))
}

// ---------------------------------------------------------------------------
// tests
// ---------------------------------------------------------------------------

#[cfg(test)]
pub(crate) mod tests {

	/// `realloc` is the third instrumentation point, and the only one whose
	/// absence is silent.
	///
	/// `rallocx` frees the old block and returns a new one without passing
	/// through `raw_alloc` or `dealloc`, so a counter that hooks only those two
	/// is charged `nallocx(old)` at birth and credited `nallocx(new)` at death.
	/// Every grow leaks the difference NEGATIVE, unboundedly, on every `Vec`
	/// and `String` growth in the process.
	///
	/// Asserted as a BOUND rather than an equality, and over many cycles: the
	/// counter is process-global and every other test in the binary allocates
	/// against it, so an exact before/after would be the same flake
	/// `dropping_the_last_handle_frees_the_value` had to be rewritten to avoid.
	/// Uninstrumented, this leaks ~4 KiB per cycle and blows the bound by three
	/// orders of magnitude; instrumented, the residual is other threads' noise.
	///
	/// IGNORED BY DEFAULT, for the same reason as
	/// `fast_tier_allocations_land_on_node_0` and the placement checks: the
	/// counter is process-global and every other test in this binary allocates
	/// against it, so under the default parallel runner the delta is dominated
	/// by foreign traffic -- measured at 1.3 MB and 0.9 MB against signatures
	/// of 4.0 MB and ~3 KB. `global_counter_lock` does not help; it serialises
	/// test BODIES, not the threads they spawn. Run with the process to itself:
	///
	/// ```text
	/// cargo +nightly test --lib --features <cache>,measured_accounting \
	///     -- --ignored --test-threads=1 numa_alloc::tests
	/// ```
	#[test]
	#[ignore = "process-global counter; needs --test-threads=1"]
	#[cfg(feature = "measured_accounting")]
	fn realloc_does_not_drift_the_counter() {
		use std::alloc::{GlobalAlloc, Layout};

		use super::measured;

		assert!(super::init(), "the node-0 and node-1 arena pools must build");

		const CYCLES: usize = 1_000;
		const SMALL: usize = 64;
		const LARGE: usize = 4096;
		const ALIGN: usize = 8;

		// If `realloc` is uncounted each cycle leaks
		// `usable(SMALL) - usable(LARGE)`, so the drift is CYCLES times that.
		let leak_per_cycle =
			measured::usable(LARGE, ALIGN) as i64 - measured::usable(SMALL, ALIGN) as i64;
		assert!(leak_per_cycle > 0, "the test sizes must span a size class");

		let allocator = NumaAlloc::<NODE_FAST>;
		let small = Layout::from_size_align(SMALL, ALIGN).expect("small layout");
		let large = Layout::from_size_align(LARGE, ALIGN).expect("large layout");

		// RAW, not `allocated`: the clamp at zero hides exactly the negative
		// drift this test exists to find.
		let before = measured::raw_sum(NODE_FAST);

		for _ in 0..CYCLES {
			unsafe {
				let ptr = GlobalAlloc::alloc(&allocator, small);
				assert!(!ptr.is_null(), "allocation failed");

				let grown = GlobalAlloc::realloc(&allocator, ptr, small, LARGE);
				assert!(!grown.is_null(), "realloc failed");

				GlobalAlloc::dealloc(&allocator, grown, large);
			}
		}

		let after = measured::raw_sum(NODE_FAST);
		let drift = (after - before).abs();
		let would_leak = leak_per_cycle * CYCLES as i64;

		assert!(
			drift < would_leak / 8,
			"the counter drifted {drift} B over {CYCLES} grow/free cycles; an \
			 uninstrumented realloc would drift about {would_leak} B, so this \
			 looks like realloc is not carrying its delta",
		);
	}

	/// The add and the subtract must cancel EXACTLY for a plain alloc/free,
	/// which is what lets the counter work with no per-pointer side table.
	/// Same bound-not-equality reasoning as above.
	///
	/// IGNORED BY DEFAULT, for the same reason as
	/// `fast_tier_allocations_land_on_node_0` and the placement checks: the
	/// counter is process-global and every other test in this binary allocates
	/// against it, so under the default parallel runner the delta is dominated
	/// by foreign traffic -- measured at 1.3 MB and 0.9 MB against signatures
	/// of 4.0 MB and ~3 KB. `global_counter_lock` does not help; it serialises
	/// test BODIES, not the threads they spawn. Run with the process to itself:
	///
	/// ```text
	/// cargo +nightly test --lib --features <cache>,measured_accounting \
	///     -- --ignored --test-threads=1 numa_alloc::tests
	/// ```
	#[test]
	#[ignore = "process-global counter; needs --test-threads=1"]
	#[cfg(feature = "measured_accounting")]
	fn alloc_and_free_cancel() {
		use std::alloc::{GlobalAlloc, Layout};

		use super::measured;

		assert!(super::init(), "the node-0 and node-1 arena pools must build");

		const CYCLES: usize = 1_000;
		// Over-aligned on purpose: `usable` must apply the `mallocx_lg_align`
		// bits, and `resident_value_bytes`'s `nallocx(n, 0)` -- correct only
		// for align <= 8 -- must not have been copied into it.
		let layout = Layout::from_size_align(3000, 64).expect("layout");

		let before = measured::raw_sum(NODE_FAST);

		for _ in 0..CYCLES {
			unsafe {
				let ptr = GlobalAlloc::alloc(&NumaAlloc::<NODE_FAST>, layout);
				assert!(!ptr.is_null(), "allocation failed");
				GlobalAlloc::dealloc(&NumaAlloc::<NODE_FAST>, ptr, layout);
			}
		}

		let drift = (measured::raw_sum(NODE_FAST) - before).abs();
		let one_object = measured::usable(3000, 64) as i64;

		assert!(
			drift < one_object * 8,
			"alloc/free left {drift} B behind over {CYCLES} cycles at \
			 {one_object} B each -- the two sides are not cancelling",
		);
	}
	use super::*;
	use std::alloc::{Allocator, Layout};

	/// Pages must actually land on the requested node.
	///
	/// Reads `/proc/self/numa_maps` rather than trusting the hooks' own
	/// counters: those record what was asked for, the kernel reports where
	/// pages are, and the whole point of this module is that those two can
	/// disagree silently.
	///
	/// Touches every page, because `mmap` alone places nothing -- binding
	/// governs the *fault*, so an untouched mapping has no pages on any node.
	fn placement_check(node: u32, alloc: &dyn Fn(Layout) -> *mut u8) {
		const BYTES: usize = 256 * 1024 * 1024;
		let page = unsafe { libc::sysconf(libc::_SC_PAGESIZE) } as usize;

		assert!(init(), "arena pool must build");

		let (before_fast, before_slow) = resident_pages_per_node().expect("numa_maps readable");

		let layout = Layout::from_size_align(BYTES, page).unwrap();
		let ptr = alloc(layout);
		assert!(!ptr.is_null(), "allocation failed");

		// Fault every page in.
		unsafe {
			for offset in (0..BYTES).step_by(page) {
				std::ptr::write_volatile(ptr.add(offset), 1u8);
			}
		}

		let (after_fast, after_slow) = resident_pages_per_node().expect("numa_maps readable");
		let grew_fast = after_fast.saturating_sub(before_fast);
		let grew_slow = after_slow.saturating_sub(before_slow);
		let expected = (BYTES / page) as u64;

		let (target, other) =
			if node == NODE_FAST { (grew_fast, grew_slow) } else { (grew_slow, grew_fast) };

		println!(
			"PLACEMENT node={node} requested={BYTES}B expected_pages={expected} \
fast_grew={grew_fast} slow_grew={grew_slow} -> on_target={target} elsewhere={other}"
		);

		// Allow slack for unrelated activity, but the split must be decisive:
		// nearly all of it on the target node, and the other node essentially
		// untouched by this allocation.
		assert!(
			target >= expected * 9 / 10,
			"node {node}: expected ~{expected} pages, saw {target} (other node grew {other})"
		);
		assert!(
			other < expected / 10,
			"node {node}: other node grew {other} pages, binding leaked"
		);

		unsafe { std::alloc::GlobalAlloc::dealloc(&NumaAlloc::<0>, ptr, layout) };
	}

	/// Ignored by default: `placement_check` measures process-wide growth in
	/// `/proc/self/numa_maps`, so any test allocating concurrently lands in
	/// the same delta and the "other node barely grew" assertion fails. Under
	/// the default runner the whole suite allocates in parallel, so this can
	/// only give a truthful answer with the process to itself:
	///
	/// ```text
	/// cargo +nightly test --release --features <policy>,numa_jemalloc --lib \
	///     numa_alloc::tests -- --test-threads=1
	/// ```
	#[test]
	#[ignore = "measures process-wide numa_maps; needs --test-threads=1 (see doc comment)"]
	fn fast_tier_allocations_land_on_node_0() {
		placement_check(NODE_FAST, &|layout| {
			FastAlloc::default()
				.allocate(layout)
				.expect("fast allocation")
				.as_ptr()
				.cast()
		});
	}

	/// Ignored by default: `placement_check` measures process-wide growth in
	/// `/proc/self/numa_maps`, so any test allocating concurrently lands in
	/// the same delta and the "other node barely grew" assertion fails. Under
	/// the default runner the whole suite allocates in parallel, so this can
	/// only give a truthful answer with the process to itself:
	///
	/// ```text
	/// cargo +nightly test --release --features <policy>,numa_jemalloc --lib \
	///     numa_alloc::tests -- --test-threads=1
	/// ```
	#[test]
	#[ignore = "measures process-wide numa_maps; needs --test-threads=1 (see doc comment)"]
	fn slow_tier_allocations_land_on_the_slow_node() {
		placement_check(NODE_SLOW, &|layout| {
			SlowAlloc::default()
				.allocate(layout)
				.expect("slow allocation")
				.as_ptr()
				.cast()
		});
	}

	/// The counters must show every extent bound and nothing falling through.
	///
	/// `unbound_fallbacks` is the honest measure of the guarantee: it counts
	/// allocations that could not reach a bound arena. Non-zero means the
	/// guarantee held only partially, which is exactly the failure the
	/// previous implementation hid by returning unbound memory silently.
	#[test]
	fn no_extent_is_left_unbound() {
		assert!(init());

		let page = unsafe { libc::sysconf(libc::_SC_PAGESIZE) } as usize;
		let layout = Layout::from_size_align(64 * 1024 * 1024, page).unwrap();

		let fast = FastAlloc::default().allocate(layout).expect("fast");
		let slow = SlowAlloc::default().allocate(layout).expect("slow");

		let report = stats();
		assert!(report.contains("mbind_failed=0"), "an mbind failed: {report}");

		unsafe {
			FastAlloc::default().deallocate(fast.cast(), layout);
			SlowAlloc::default().deallocate(slow.cast(), layout);
		}
	}

	/// Makes `node_arenas` report every pool as unbuilt on this thread until
	/// dropped.
	///
	/// The unbound fallback is otherwise unreachable from a test: the global
	/// allocator builds the fast pool before any test body runs, and `IN_INIT`
	/// is only consulted while a pool is still unbuilt.
	struct ForceUnbound;

	impl ForceUnbound {
		fn new() -> Self {
			FORCE_UNBOUND.with(|f| f.set(true));
			ForceUnbound
		}
	}

	impl Drop for ForceUnbound {
		fn drop(&mut self) {
			FORCE_UNBOUND.with(|f| f.set(false));
		}
	}

	fn unbound_fallback_alignment_check<const NODE: u32>() {
		use std::alloc::GlobalAlloc;

		const N: usize = 256;
		const ALIGN: usize = 64;
		// Unaligned, 1 B is the 8 B class, where only one region in eight sits
		// on a 64 B boundary, and 100 B is the 112 B class. Aligned to 64 they
		// are the 64 B and 128 B classes, so a batch cannot pass by luck and the
		// grow has to move the block.
		const SIZE: usize = 1;
		const GROWN: usize = 100;

		let allocator = NumaAlloc::<NODE>;
		let small = Layout::from_size_align(SIZE, ALIGN).unwrap();
		let grown = Layout::from_size_align(GROWN, ALIGN).unwrap();
		let lg_align = mallocx_lg_align(ALIGN.trailing_zeros());

		// The class `dealloc` tells `sdallocx` a block is in, against the class
		// jemalloc actually carved it from.
		let class = |size| unsafe { tikv_jemalloc_sys::nallocx(size, lg_align) };
		let actual = |ptr: *mut u8| unsafe { tikv_jemalloc_sys::sallocx(ptr.cast_const().cast(), 0) };

		let mut blocks = Vec::with_capacity(N);
		let fallbacks_before = UNBOUND_FALLBACKS.load(Ordering::Relaxed);

		{
			let _unbound = ForceUnbound::new();

			for i in 0..N {
				let zeroed = i % 2 == 1;
				let ptr = unsafe {
					if zeroed {
						GlobalAlloc::alloc_zeroed(&allocator, small)
					} else {
						GlobalAlloc::alloc(&allocator, small)
					}
				};
				assert!(!ptr.is_null(), "node {NODE}: fallback allocation failed");
				assert_eq!(ptr as usize % ALIGN, 0, "node {NODE}: fallback alloc returned {ptr:p}");
				assert_eq!(
					actual(ptr),
					class(SIZE),
					"node {NODE}: fallback alloc's size class is not the one dealloc frees it as",
				);
				if zeroed {
					assert_eq!(unsafe { *ptr }, 0, "node {NODE}: fallback alloc_zeroed was not zeroed");
				}
				blocks.push(ptr);
			}

			for ptr in &mut blocks {
				let moved = unsafe { GlobalAlloc::realloc(&allocator, *ptr, small, GROWN) };
				assert!(!moved.is_null(), "node {NODE}: fallback realloc failed");
				assert_eq!(moved as usize % ALIGN, 0, "node {NODE}: fallback realloc returned {moved:p}");
				assert_eq!(
					actual(moved),
					class(GROWN),
					"node {NODE}: fallback realloc's size class is not the one dealloc frees it as",
				);
				*ptr = moved;
			}
		}

		// `>=`: the counter is process-global. Anything less means the loops
		// above ran on the bound path and tested nothing.
		let fallbacks = UNBOUND_FALLBACKS.load(Ordering::Relaxed) - fallbacks_before;
		assert!(
			fallbacks >= 2 * N as u64,
			"node {NODE}: {fallbacks} fallbacks counted for {N} allocs and {N} reallocs",
		);

		for ptr in blocks {
			unsafe { GlobalAlloc::dealloc(&allocator, ptr, grown) };
		}

		// A block freed under the wrong size class is filed in that class's
		// cache bin and handed straight back out from there. Refill the class
		// through the bound path and check nothing misfiled comes back.
		let mut refill = Vec::with_capacity(N);
		for _ in 0..N {
			let ptr = unsafe { GlobalAlloc::alloc(&allocator, grown) };
			assert!(!ptr.is_null(), "node {NODE}: bound allocation failed");
			assert_eq!(ptr as usize % ALIGN, 0, "node {NODE}: refill returned {ptr:p}");
			assert_eq!(actual(ptr), class(GROWN), "node {NODE}: refill got a misfiled block");
			refill.push(ptr);
		}
		for ptr in refill {
			unsafe { GlobalAlloc::dealloc(&allocator, ptr, grown) };
		}
	}

	/// The unbound fallback must honour `layout.align()` like the bound path.
	///
	/// `dealloc` always passes `mallocx_lg_align` to `sdallocx`, so a fallback
	/// that drops it returns misaligned memory AND frees it under the size
	/// class of the aligned request, which for a small size is a different
	/// class from the one the block came from. Nothing fails at the free; the
	/// block is filed in the wrong bin.
	#[test]
	fn unbound_fallback_honours_alignment() {
		assert!(init(), "the node-0 and node-1 arena pools must build");

		unbound_fallback_alignment_check::<NODE_FAST>();
		unbound_fallback_alignment_check::<NODE_SLOW>();
	}

	/// The slow pool's unbound fallback must bypass the thread's DEFAULT
	/// tcache -- the fast tier's cache -- in both directions. With no tcache
	/// flag there, a fallback `mallocx` pops whatever that cache's bin holds,
	/// a fast-arena (node-0) block, into the slow tier, and a fallback
	/// `rallocx` that moves its block frees the old one -- a slow-arena
	/// (node-2) block -- into it, where the fast tier's next allocation of
	/// that class takes it. Each step is checked right after it, since a
	/// tcache bin is LIFO: fast blocks of both classes are freed onto the
	/// cache, the fallbacks run, then the fast tier asks for the old block's
	/// class. Classes: two small bins, and 20 KB (the 20 KiB class), above
	/// 14 KiB and below the default `tcache_max` -- with natural alignment off
	/// it is not served FROM the cache, but is still freed into it. Every
	/// grow or shrink changes class, so `rallocx` must move the block. On a
	/// thread of its own, so nothing else touches that thread's default cache
	/// between the steps. Holds under any `SLOW_TCACHE` and
	/// `ALLOC_NATURAL_ALIGN`. With the fallback passing no tcache flag, every
	/// one of the three counts was non-zero.
	#[test]
	fn unbound_slow_fallback_bypasses_the_default_tcache() {
		assert!(init(), "the node-0 and node-2 arena pools must build");

		std::thread::spawn(|| {
			use std::alloc::GlobalAlloc;

			const PAIRS: [(usize, usize); 3] = [(64, 3000), (3000, 20_000), (20_000, 64)];
			const ROUNDS: usize = 32 * PAIRS.len();

			let fast = NumaAlloc::<NODE_FAST>;
			let slow = NumaAlloc::<NODE_SLOW>;
			let fast_arenas = pool_arenas(NODE_FAST);
			let slow_arenas = pool_arenas(NODE_SLOW);
			let arena = |p: *mut u8| arena_of(p).expect("arenas.lookup");
			let fallbacks_before = UNBOUND_FALLBACKS.load(Ordering::Relaxed);
			// Fallback allocs and reallocs served a fast block; fast
			// allocations served a slow one; reallocs that moved.
			let (mut took_alloc, mut took_realloc, mut gave_fast, mut moved) = (0usize, 0usize, 0usize, 0usize);

			for round in 0..ROUNDS {
				let (size, grown) = PAIRS[round % PAIRS.len()];
				let small = Layout::from_size_align(size, 8).unwrap();
				let big = Layout::from_size_align(grown, 8).unwrap();
				unsafe {
					// A slow-arena block, from the bound path.
					let s = slow.alloc(small);
					// A fast block of each class, on top of the default cache's bins.
					let fs = fast.alloc(small);
					let fb = fast.alloc(big);
					assert!(!s.is_null() && !fs.is_null() && !fb.is_null(), "round {round}: bound allocation failed");
					fast.dealloc(fs, small);
					fast.dealloc(fb, big);

					let (a, m) = {
						let _unbound = ForceUnbound::new();
						(slow.alloc(small), slow.realloc(s, small, grown))
					};
					assert!(!a.is_null() && !m.is_null(), "round {round}: fallback allocation failed");
					// The fast tier's next block of `s`'s class.
					let f = fast.alloc(small);
					assert!(!f.is_null(), "round {round}: fast allocation failed");

					if a == fs || fast_arenas.contains(&arena(a)) {
						took_alloc += 1;
					}
					if m == fb || fast_arenas.contains(&arena(m)) {
						took_realloc += 1;
					}
					if f == s || slow_arenas.contains(&arena(f)) {
						gave_fast += 1;
					}
					if m != s {
						moved += 1;
					}

					slow.dealloc(a, small);
					slow.dealloc(m, big);
					fast.dealloc(f, small);
				}
			}

			let fallbacks = UNBOUND_FALLBACKS.load(Ordering::Relaxed) - fallbacks_before;
			println!(
				"FALLBACKTCACHE rounds={ROUNDS} took_alloc={took_alloc} took_realloc={took_realloc} gave_fast={gave_fast} \
moved={moved} fallbacks={fallbacks} natural_align={} slow_tcache={}",
				natural_align_enabled(),
				slow_tcache_enabled()
			);
			// `>=`: the counter is process-global. Less means the loop ran on
			// the bound path and tested nothing.
			assert!(fallbacks >= 2 * ROUNDS as u64, "{fallbacks} fallbacks counted for {ROUNDS} allocs and {ROUNDS} reallocs");
			assert_eq!(moved, ROUNDS, "every realloc changes class, so every one must move its block");
			assert_eq!(took_alloc, 0, "slow fallback allocations took {took_alloc} of {ROUNDS} blocks from the default tcache");
			assert_eq!(took_realloc, 0, "slow fallback reallocs took {took_realloc} of {ROUNDS} blocks from the default tcache");
			assert_eq!(gave_fast, 0, "the fast tier got {gave_fast} of {ROUNDS} slow blocks a fallback realloc freed into the default tcache");
		})
		.join()
		.expect("the fallback thread");
	}

	/// Cross-thread free must not move a block between nodes.
	///
	/// This is the pattern the cache actually runs -- allocate on one thread,
	/// free on another -- and the one that makes tcache dangerous: a cache
	/// serving both nodes would hand a freed node-0 block back for a node-1
	/// request. Nothing would crash; placement would just decay under load.
	#[test]
	fn cross_thread_free_preserves_placement() {
		assert!(init());

		const ROUNDS: usize = 2_000;
		let page = unsafe { libc::sysconf(libc::_SC_PAGESIZE) } as usize;
		let layout = Layout::from_size_align(page * 4, page).unwrap();

		let (tx, rx) = std::sync::mpsc::channel::<(usize, u32)>();

		let freer = std::thread::spawn(move || {
			while let Ok((addr, node)) = rx.recv() {
				let ptr = std::ptr::NonNull::new(addr as *mut u8).unwrap();
				unsafe {
					match node {
						NODE_FAST => FastAlloc::default().deallocate(ptr, layout),
						_ => SlowAlloc::default().deallocate(ptr, layout),
					}
				}
			}
		});

		for round in 0..ROUNDS {
			let node = if round % 2 == 0 { NODE_FAST } else { NODE_SLOW };
			let ptr = match node {
				NODE_FAST => FastAlloc::default().allocate(layout).unwrap(),
				_ => SlowAlloc::default().allocate(layout).unwrap(),
			};
			tx.send((ptr.as_ptr() as *mut u8 as usize, node)).unwrap();
		}

		drop(tx);
		freer.join().expect("freer thread");

		let report = stats();
		assert!(report.contains("mbind_failed=0"), "{report}");
	}

	/// A plain-old-data mallctl read, or `None` if jemalloc refuses it.
	fn ctl<T: Default + Copy>(name: &str) -> Option<T> {
		let name = std::ffi::CString::new(name).unwrap();
		let mut v = T::default();
		let mut len = size_of::<T>();
		let rc = unsafe { mallctl(name.as_ptr(), (&raw mut v).cast(), &raw mut len, std::ptr::null_mut(), 0) };
		(rc == 0).then_some(v)
	}

	/// Index of the large size class `class` (`arenas.lextent.<j>.size`).
	fn lextent_index(class: usize) -> usize {
		let n: c_uint = ctl("arenas.nlextents").expect("arenas.nlextents");
		(0..n as usize)
			.find(|j| ctl::<size_t>(&format!("arenas.lextent.{j}.size")) == Some(class))
			.unwrap_or_else(|| panic!("no large class of {class} B"))
	}

	/// `stats.arenas.<i>.lextents.<j>.nmalloc` summed over one pool's arenas,
	/// for the large class `class`: arena-level large allocations only, so a
	/// tcache hit does not count and a miss does. Other tests in the process
	/// can only add to it.
	fn pool_lextent_nmalloc(node: u32, class: usize) -> u64 {
		unsafe {
			let mut e: u64 = 1;
			let mut l = size_of::<u64>();
			mallctl(c"epoch".as_ptr(), (&raw mut e).cast(), &raw mut l, (&raw mut e).cast(), size_of::<u64>());
		}
		let j = lextent_index(class);
		pool_arenas(node)
			.into_iter()
			.map(|a| {
				ctl::<u64>(&format!("stats.arenas.{a}.lextents.{j}.nmalloc"))
					.unwrap_or_else(|| panic!("stats unavailable for arena {a}"))
			})
			.sum()
	}

	/// Natural alignment (`align_flags` with `ALLOC_NATURAL_ALIGN` on) must
	/// change only the path `mallocx` takes, never a size class: what the
	/// cache charges (`measured::usable`, `meta::usable`, and the
	/// `nallocx(n, 0)` of `object::overhead`) and what `sdallocx` is told
	/// must be what the block really is, whichever way the flag went.
	///
	/// 1. The threshold's premise, read from the jemalloc this binary links:
	///    the quantum is `NATURAL_ALIGN_MAX` and the smallest class is 8 B.
	/// 2. `nallocx` with and without the flag, for every size up to 64 KiB --
	///    every small class, and the large ones past four times the default
	///    `tcache_max` -- and both edges of every large class up to 64 GiB, at
	///    alignments 2 to 16. That covers every value layout the crate makes:
	///    `value`'s `(len.max(1), 8)` and `value_thin`'s items (header plus
	///    bytes, aligned to 8 or to a key's 16).
	/// 3. Real blocks of the value alignment (8) and of 16 from every pool,
	///    fresh and recycled through a cache: aligned -- so a value pointer's
	///    three tag bits are clear -- and in exactly the flagged class
	///    (`sallocx`).
	///
	/// Holds under either `ALLOC_NATURAL_ALIGN`; the suites run it at both.
	#[test]
	fn natural_alignment_keeps_every_size_class() {
		assert!(init(), "the node-0 and node-2 arena pools must build");
		assert!(init_node(NODE_FAST_VALUES), "the value pool must build");

		assert_eq!(ctl::<size_t>("arenas.quantum"), Some(NATURAL_ALIGN_MAX), "the quantum is the threshold");
		assert_eq!(ctl::<size_t>("arenas.bin.0.size"), Some(8), "the smallest class must be 8 B (8-aligned)");

		let nallocx = |size: usize, flags: c_int| unsafe { tikv_jemalloc_sys::nallocx(size, flags) };

		let mut sizes: Vec<usize> = (1..=65_536).collect();
		let nlextents: c_uint = ctl("arenas.nlextents").expect("arenas.nlextents");
		for j in 0..nlextents {
			let class: size_t = ctl(&format!("arenas.lextent.{j}.size")).expect("a large class");
			if class > 1 << 36 {
				break;
			}
			sizes.extend([class - 1, class, class + 1]);
		}

		let mut unflagged = 0usize;
		for align in [2usize, 4, 8, 16] {
			for &size in &sizes {
				let natural = align_flags_with(true, align, size);
				let flagged = align_flags_with(false, align, size);
				assert_ne!(flagged, 0, "align {align} must be flagged without natural alignment");
				if natural == 0 {
					unflagged += 1;
				}

				let class = nallocx(size, flagged);
				assert_eq!(nallocx(size, natural), class, "{size} B aligned to {align}: the class moved");
				assert_eq!(class % align, 0, "{size} B aligned to {align}: class {class}");
				assert_eq!(crate::meta::usable(size, align), class as u64, "{size} B aligned to {align}: meta::usable");
				#[cfg(feature = "measured_accounting")]
				assert_eq!(measured::usable(size, align), class, "{size} B aligned to {align}: measured::usable");
			}
		}
		// The rule must actually drop the flag, or (2) proved nothing: at each
		// of the four alignments, for every size from the alignment up.
		assert!(unflagged > 4 * 60_000, "natural alignment dropped the flag for only {unflagged} requests");
		assert_eq!(align_flags_with(true, 8, 7), mallocx_lg_align(3), "align > size keeps the flag");
		assert_eq!(align_flags_with(true, 32, 4096), mallocx_lg_align(5), "align > 16 keeps the flag");
		assert_eq!(align_flags_with(true, 1, 1), 0);
		assert_eq!(align_flags_with(false, 1, 1), 0, "align 1 never had a flag");

		natural_alignment_blocks::<NODE_FAST>();
		natural_alignment_blocks::<NODE_SLOW>();
		natural_alignment_blocks::<NODE_FAST_VALUES>();
	}

	fn natural_alignment_blocks<const NODE: u32>() {
		use std::alloc::GlobalAlloc;

		const SIZES: [usize; 23] = [
			1, 7, 8, 9, 15, 16, 17, 24, 40, 100, 4096, 4097, 14_336, 14_337, 16_384, 16_385, 20_000, 28_672,
			32_768, 32_769, 65_536, 100_000, 1 << 20,
		];

		let allocator = NumaAlloc::<NODE>;
		for align in [VALUE_ALIGN_FOR_TESTS, 16] {
			for size in SIZES {
				let layout = Layout::from_size_align(size, align).unwrap();
				let class = unsafe { tikv_jemalloc_sys::nallocx(size, align_flags_with(false, align, size)) };
				let check = |ptr: *mut u8, when: &str| {
					assert!(!ptr.is_null(), "pool {NODE}: {size} B aligned to {align} ({when}) failed");
					assert_eq!(ptr as usize % align, 0, "pool {NODE}: {size} B aligned to {align} ({when}) at {ptr:p}");
					assert_eq!(ptr as usize & 0b111, 0, "pool {NODE}: {ptr:p} ({when}) has a tag bit set");
					let real = unsafe { tikv_jemalloc_sys::sallocx(ptr.cast_const().cast(), 0) };
					assert_eq!(real, class, "pool {NODE}: {size} B aligned to {align} ({when}) is in class {real}, not {class}");
					unsafe {
						ptr.write_volatile(1);
						ptr.add(size - 1).write_volatile(1);
					}
				};

				// Fresh, then -- after the frees -- recycled through whatever
				// cache the pool's frees went into.
				let blocks: Vec<*mut u8> = (0..4)
					.map(|_| {
						let ptr = unsafe { GlobalAlloc::alloc(&allocator, layout) };
						check(ptr, "fresh");
						ptr
					})
					.collect();
				for ptr in blocks {
					unsafe { GlobalAlloc::dealloc(&allocator, ptr, layout) };
				}
				for _ in 0..4 {
					let ptr = unsafe { GlobalAlloc::alloc(&allocator, layout) };
					check(ptr, "recycled");
					unsafe { GlobalAlloc::dealloc(&allocator, ptr, layout) };
				}
			}
		}
	}

	/// `value::VALUE_ALIGN`, which is private to that module.
	const VALUE_ALIGN_FOR_TESTS: usize = 8;

	/// What natural alignment is FOR, shown where nothing else interferes: on
	/// the value pool's own explicit tcache, a 28 KB (28 KiB class) block --
	/// above `SC_SMALL_MAXCLASS`, below the default `tcache_max` -- allocated
	/// and freed in a loop reaches the arena about once with natural
	/// alignment, and every time without it, because the flagged request takes
	/// the aligned path past the cache. Asserted for whichever
	/// `ALLOC_NATURAL_ALIGN` the process has; the suites run both.
	#[test]
	fn natural_alignment_lets_a_large_class_use_the_tcache() {
		assert!(init_node(NODE_FAST_VALUES), "the value pool must build");

		const LOOP: u64 = 2_000;
		const SIZE: usize = 28_000;
		let class = unsafe { tikv_jemalloc_sys::nallocx(SIZE, 0) };
		let tcache_max: size_t = ctl("arenas.tcache_max").expect("arenas.tcache_max");
		if class > tcache_max {
			println!("NATURALTCACHE skipped: class {class} above tcache_max {tcache_max}");
			return;
		}

		let layout = Layout::from_size_align(SIZE, VALUE_ALIGN_FOR_TESTS).unwrap();
		let before = pool_lextent_nmalloc(NODE_FAST_VALUES, class);
		for _ in 0..LOOP {
			let block = std::alloc::Allocator::allocate(&FastValues, layout).expect("value block");
			unsafe { std::alloc::Allocator::deallocate(&FastValues, block.cast(), layout) };
		}
		let arena_mallocs = pool_lextent_nmalloc(NODE_FAST_VALUES, class) - before;
		let cached = VALUE_TCACHE.with(|t| t.get()) != u32::MAX;

		println!(
			"NATURALTCACHE natural_align={} value_tcache={cached} loop={LOOP} class={class} arena_large_mallocs={arena_mallocs}",
			natural_align_enabled()
		);
		if !natural_align_enabled() {
			assert!(arena_mallocs >= LOOP, "{arena_mallocs} arena mallocs for {LOOP} flagged allocations");
		} else if cached {
			assert!(arena_mallocs < LOOP / 10, "the value tcache is not serving {class} B: {arena_mallocs} arena mallocs for {LOOP}");
		} else {
			println!("NATURALTCACHE tcache.create failed on this thread; caching assertion skipped");
		}
	}

	/// The slow tier's explicit tcache (`SLOW_TCACHE=1`) must keep every slow
	/// block on the slow node, in a slow arena, under the pattern that makes a
	/// tcache dangerous -- allocate on one thread, free on another, with DRAM
	/// blocks of the SAME size classes churning through the same threads'
	/// other caches -- and it must actually cache (the positive control), or
	/// the placement half proves nothing.
	///
	/// Runs under either setting; with `SLOW_TCACHE` off it checks placement
	/// and that the slow tier stays uncached. Classes above 14 KiB are served
	/// from the cache only with natural alignment, so with
	/// `ALLOC_NATURAL_ALIGN=0` the caching half expects hits for the small
	/// classes only. If another test in this process latched
	/// `SLOW_TCACHE_POISONED` (the unbound-fallback test does), or this
	/// thread's `tcache.create` failed, the caching half is skipped with a
	/// note; run it alone to be sure it ran:
	///
	/// ```text
	/// SLOW_TCACHE=1 cargo test --release --lib --features <cache> -- --exact \
	///     numa_alloc::tests::slow_tcache_cross_thread_placement --nocapture
	/// ```
	#[test]
	fn slow_tcache_cross_thread_placement() {
		assert!(init());

		// Small (bin) classes, and large ones up to the default tcache_max.
		const SIZES: [usize; 5] = [64, 3000, 14_336, 16_384, 20_000];
		const ROUNDS: usize = 3_000;
		const SMALL_MAX: usize = 14_336;
		let poisoned_at_start = SLOW_TCACHE_POISONED.load(Ordering::Relaxed);
		let slow_arenas = pool_arenas(NODE_SLOW);
		let check = |ptr: *mut u8, arenas: &[c_uint], who: &str| {
			unsafe { std::ptr::write_volatile(ptr, 1u8) };
			let node = node_of(ptr);
			assert_eq!(node, NODE_SLOW as i32, "{who}: slow block {ptr:p} on node {node}");
			let arena = arena_of(ptr).expect("arenas.lookup");
			assert!(arenas.contains(&arena), "{who}: slow block {ptr:p} from arena {arena}, not {arenas:?}");
		};

		let (tx, rx) = std::sync::mpsc::channel::<(usize, usize, bool)>();
		let (back_tx, back_rx) = std::sync::mpsc::channel::<(usize, usize, bool)>();

		// B frees what A sent (slow blocks into B's slow cache, DRAM blocks
		// into B's default cache), then allocates the same sizes from both
		// tiers and returns them for A to free.
		let b_arenas = slow_arenas.clone();
		let b = std::thread::spawn(move || {
			let mut hits = [0usize; SIZES.len()];
			while let Ok((addr, size, slow)) = rx.recv() {
				let layout = Layout::from_size_align(size, 8).unwrap();
				let ptr = std::ptr::NonNull::new(addr as *mut u8).unwrap();
				unsafe {
					if slow { SlowAlloc::default().deallocate(ptr, layout) } else { FastAlloc::default().deallocate(ptr, layout) }
				}
				let p = if slow { SlowAlloc::default().allocate(layout) } else { FastAlloc::default().allocate(layout) }
					.unwrap()
					.as_ptr() as *mut u8;
				if slow {
					if p as usize == addr {
						hits[SIZES.iter().position(|&x| x == size).unwrap()] += 1;
					}
					check(p, &b_arenas, "B");
				}
				back_tx.send((p as usize, size, slow)).unwrap();
			}
			(hits, SLOW_TCACHE.with(|t| t.get()))
		});

		for round in 0..ROUNDS {
			let size = SIZES[round % SIZES.len()];
			let layout = Layout::from_size_align(size, 8).unwrap();
			for slow in [true, false] {
				let p = if slow { SlowAlloc::default().allocate(layout) } else { FastAlloc::default().allocate(layout) }
					.unwrap()
					.as_ptr() as *mut u8;
				if slow {
					check(p, &slow_arenas, "A");
				}
				tx.send((p as usize, size, slow)).unwrap();
			}
			// A frees B's blocks: B-allocated slow blocks into A's slow cache.
			while let Ok((addr, size, slow)) = back_rx.try_recv() {
				let layout = Layout::from_size_align(size, 8).unwrap();
				let ptr = std::ptr::NonNull::new(addr as *mut u8).unwrap();
				unsafe {
					if slow { SlowAlloc::default().deallocate(ptr, layout) } else { FastAlloc::default().deallocate(ptr, layout) }
				}
			}
		}
		drop(tx);
		let (b_hits, b_tcache) = b.join().expect("B");
		while let Ok((addr, size, slow)) = back_rx.recv() {
			let layout = Layout::from_size_align(size, 8).unwrap();
			let ptr = std::ptr::NonNull::new(addr as *mut u8).unwrap();
			unsafe {
				if slow { SlowAlloc::default().deallocate(ptr, layout) } else { FastAlloc::default().deallocate(ptr, layout) }
			}
		}

		// Positive control: an alloc/free loop of one large class on this
		// thread reaches the arena about once with a working cache, every
		// time without one.
		const LOOP: u64 = 2_000;
		let class = unsafe { tikv_jemalloc_sys::nallocx(20_000, 0) };
		let layout = Layout::from_size_align(20_000, 8).unwrap();
		let before = pool_lextent_nmalloc(NODE_SLOW, class);
		for _ in 0..LOOP {
			let p = SlowAlloc::default().allocate(layout).unwrap();
			check(p.as_ptr() as *mut u8, &slow_arenas, "loop");
			unsafe { SlowAlloc::default().deallocate(p.cast(), layout) };
		}
		let arena_mallocs = pool_lextent_nmalloc(NODE_SLOW, class) - before;
		let poisoned_at_end = SLOW_TCACHE_POISONED.load(Ordering::Relaxed);
		let a_tcache = SLOW_TCACHE.with(|t| t.get());
		let large_cached = natural_align_enabled();

		println!(
			"SLOWTCACHE enabled={} natural_align={large_cached} poisoned={poisoned_at_start}->{poisoned_at_end} \
a_tcache={a_tcache} b_tcache={b_tcache} sizes={SIZES:?} b_same_addr_hits={b_hits:?} of {} each, loop={LOOP} \
arena_large_mallocs={arena_mallocs} {}",
			slow_tcache_enabled(),
			ROUNDS / SIZES.len(),
			tcache_report()
		);

		let created = |id: u32| id < SLOW_TCACHE_FAILED;
		if !slow_tcache_enabled() {
			// Other tests may add to the count, never subtract.
			assert!(arena_mallocs >= LOOP, "slow tier cached with SLOW_TCACHE off: {arena_mallocs} arena mallocs for {LOOP}");
		} else if poisoned_at_start || poisoned_at_end {
			println!("SLOWTCACHE poisoned by another test in this process; caching assertion skipped");
		} else if !created(a_tcache) || !created(b_tcache) {
			println!("SLOWTCACHE tcache.create failed on a test thread (ids exhausted?); caching assertion skipped");
		} else {
			if large_cached {
				assert!(arena_mallocs < LOOP / 10, "slow tcache not caching: {arena_mallocs} arena mallocs for {LOOP} allocs");
			} else {
				assert!(arena_mallocs >= LOOP, "{arena_mallocs} arena mallocs for {LOOP} flagged allocations");
			}
			// B freed A's block and asked for the same size: a LIFO cache hands
			// that very block back, for every class it serves.
			for (size, hits) in SIZES.iter().zip(b_hits) {
				if *size <= SMALL_MAX || large_cached {
					assert!(hits * 10 >= (ROUNDS / SIZES.len()) * 9, "size {size}: only {hits} cache hits on B");
				}
			}
		}

		let report = stats();
		assert!(report.contains("mbind_failed=0"), "{report}");
	}

	/// A DRAM-only configuration must not create node-1 arenas.
	///
	/// The all-DRAM case installs `FastAlloc` globally and never constructs
	/// `SlowAlloc`; each unused arena would still cost a `mallctl`, a leaked
	/// hooks struct and its own base extent.
	///
	/// Ignored by default because it asserts on process-global state that the
	/// other tests in this module deliberately populate -- any test touching
	/// `SlowAlloc` builds the node-1 arenas for the whole process, so this is
	/// only meaningful in a fresh one:
	///
	/// ```text
	/// cargo test --release --features <policy>,numa_jemalloc --lib \
	///     numa_alloc::tests::dram_only_never_builds_slow_arenas -- --ignored --exact
	/// ```
	#[test]
	#[ignore = "asserts process-global lazy state; must run in a fresh process (see doc comment)"]
	fn dram_only_never_builds_slow_arenas() {
		assert!(init_node(NODE_FAST), "fast arenas must build");

		assert!(
			SLOW_ARENAS.get().is_none(),
			"node-1 arenas were built despite only the fast tier being used"
		);

		let page = unsafe { libc::sysconf(libc::_SC_PAGESIZE) } as usize;
		let layout = Layout::from_size_align(page * 16, page).unwrap();
		let ptr = FastAlloc::default().allocate(layout).expect("fast allocation");
		unsafe { FastAlloc::default().deallocate(ptr.cast(), layout) };

		assert!(
			SLOW_ARENAS.get().is_none(),
			"a fast-tier allocation built node-1 arenas"
		);
	}

	#[test]
	fn environment_permits_a_placement_guarantee() {
		// Not an assertion about our code -- it reports whether the kernel is
		// configured such that a binding can hold at all. Settings that cannot
		// move bound pages are printed rather than failed; see
		// `numa_environment_warnings`.
		for warning in numa_environment_warnings() {
			eprintln!("numa environment warning: {warning}");
		}

		if let Err(problems) = assert_numa_environment() {
			panic!("environment defeats NUMA binding: {problems}");
		}
	}

	/// `demotion_enabled` reads `true`/`false` while procfs knobs read `0`/`1`;
	/// an integer-only parse made the demotion check unable to fire.
	#[test]
	fn boolean_knobs_are_read_in_either_spelling() {
		for on in ["1", "true", "y", "Y", "yes"] {
			assert!(flag_value_is_on(on), "{on:?} should read as on");
		}

		for off in ["0", "false", "n", "N", "no", ""] {
			assert!(!flag_value_is_on(off), "{off:?} should read as off");
		}
	}

	/// `kernel.numa_balancing` is a warning, not a failure, only because the
	/// arenas' binding does not opt into it: an `mbind` mode carrying
	/// `MPOL_F_NUMA_BALANCING` sets `MPOL_F_MOF` on the VMA's policy, the
	/// balancing scanner then visits it, and balancing could move bound pages
	/// again (see `numa_environment_warnings`). Pins that premise to the mode
	/// every extent is actually bound with.
	#[test]
	fn the_arenas_do_not_opt_into_numa_balancing() {
		for node in [NODE_FAST, NODE_SLOW] {
			let mode = NumaHooks::new(node).mode;

			assert_eq!(
				mode & libc::MPOL_F_NUMA_BALANCING,
				0,
				"node {node}'s arenas bind with MPOL_F_NUMA_BALANCING: kernel.numa_balancing \
				 can move their pages, so assert_numa_environment must fail on it again"
			);
		}
	}

	/// Query which NUMA node a specific address's page is on.
	///
	/// `numa_maps` is per-mapping and far too coarse for tcache-sized blocks,
	/// which live inside a shared extent; `get_mempolicy` answers for one
	/// address.
	/// `pub(crate)`: `value`'s placement test asks the same question of value
	/// buffers, and one `get_mempolicy` call site is better than two.
	pub(crate) fn node_of(ptr: *const u8) -> i32 {
		const MPOL_F_NODE: c_int = 1 << 0;
		const MPOL_F_ADDR: c_int = 1 << 1;
		let mut node: c_int = -1;
		let rc = unsafe {
			libc::syscall(
				libc::SYS_get_mempolicy,
				&raw mut node,
				std::ptr::null_mut::<c_ulong_alias>(),
				0usize,
				ptr as usize,
				(MPOL_F_NODE | MPOL_F_ADDR) as c_ulong_alias,
			)
		};
		if rc != 0 { -1 } else { node }
	}

	/// Which jemalloc arena owns `ptr`, via the `arenas.lookup` mallctl.
	///
	/// [`node_of`] answers where memory PHYSICALLY landed, and that cannot tell
	/// a `NODE_FAST_VALUES` block from a `NODE_FAST` one: both pools are
	/// `mbind`ed to physical node 0 (see [`physical_node`]), so both answer 0.
	/// This answers which POOL served the block instead, which is exactly the
	/// distinction `segregated_value_arena` exists to make -- and the one
	/// `value`'s routing test needs, because under that feature the fast
	/// allocator is [`FastValues`] and NOT the global allocator.
	///
	/// `pub(crate)` for the same reason as [`node_of`]: one FFI call site.
	///
	/// Returns `None` if the mallctl fails, which is how a jemalloc build
	/// without `arenas.lookup` reports itself -- a caller must skip rather than
	/// assert on `None`.
	pub(crate) fn arena_of(ptr: *const u8) -> Option<c_uint> {
		let mut ind: c_uint = 0;
		let mut len = std::mem::size_of::<c_uint>() as size_t;

		// `arenas.lookup` takes the pointer to look up as `newp`, which is why
		// it needs a mutable local to point at rather than `ptr` itself.
		let mut arg = ptr;

		let rc = unsafe {
			mallctl(
				c"arenas.lookup".as_ptr(),
				(&raw mut ind).cast::<c_void>(),
				&raw mut len,
				(&raw mut arg).cast::<c_void>(),
				std::mem::size_of::<*const u8>() as size_t,
			)
		};

		if rc == 0 { Some(ind) } else { None }
	}

	/// The arena indices making up one logical pool, empty if it was never
	/// built. Pair with [`arena_of`] to assert an allocation came from the pool
	/// its tier names.
	pub(crate) fn pool_arenas(node: u32) -> Vec<c_uint> {
		match node_arenas(node) {
			Some(arenas) => arenas.indices[..arenas.count].to_vec(),
			None => Vec::new(),
		}
	}

	/// `arenas.lookup` must work at all, or every test leaning on it is
	/// silently vacuous.
	#[test]
	fn arena_lookup_answers_for_each_pool() {
		assert!(init(), "the node-0 and node-1 arena pools must build");
		assert!(init_node(NODE_FAST_VALUES), "the value pool must build");

		let layout = Layout::from_size_align(4096, 8).unwrap();

		for node in [NODE_FAST, NODE_SLOW, NODE_FAST_VALUES] {
			let indices = pool_arenas(node);
			assert!(!indices.is_empty(), "pool {node} reported no arenas");

			let block = match node {
				NODE_FAST => FastAlloc::default().allocate(layout),
				NODE_SLOW => SlowAlloc::default().allocate(layout),
				_ => std::alloc::Allocator::allocate(&FastValues, layout),
			}
			.expect("pool allocation");

			let ind = arena_of(block.as_ptr().cast::<u8>())
				.expect("arenas.lookup must be available in this jemalloc build");

			assert!(
				indices.contains(&ind),
				"pool {node} served arena {ind}, which is not one of its own {indices:?}",
			);

			unsafe {
				match node {
					NODE_FAST => FastAlloc::default().deallocate(block.cast(), layout),
					NODE_SLOW => SlowAlloc::default().deallocate(block.cast(), layout),
					_ => std::alloc::Allocator::deallocate(&FastValues, block.cast(), layout),
				}
			}
		}

		// The three pools must not share arenas, or `arena_of` cannot
		// distinguish them and the assertion above is trivially true.
		let fast = pool_arenas(NODE_FAST);
		let slow = pool_arenas(NODE_SLOW);
		let values = pool_arenas(NODE_FAST_VALUES);

		for (a, an, b, bn) in [
			(&fast, "fast", &slow, "slow"),
			(&fast, "fast", &values, "values"),
			(&slow, "slow", &values, "values"),
		] {
			assert!(
				!a.iter().any(|i| b.contains(i)),
				"the {an} pool {a:?} and the {bn} pool {b:?} share an arena",
			);
		}
	}

}


/// Plain jemalloc: no arena binding, no NUMA extent hooks, no tcache flags --
/// exactly what `raw_alloc`'s unbound fallback path already does for the DRAM
/// pools (the slow pool's adds `MALLOCX_TCACHE_NONE`), alignment bits included (`align_flags`, which with natural alignment on is also
/// tikv-jemallocator's own rule). Exists to answer whether the custom
/// allocator is itself the per-operation cost.
#[cfg(feature = "stock_jemalloc")]
pub struct StockAlloc;

#[cfg(feature = "stock_jemalloc")]
impl StockAlloc {
	#[inline]
	fn align_flags(layout: std::alloc::Layout) -> c_int {
		align_flags(layout.align(), layout.size())
	}
}

#[cfg(feature = "stock_jemalloc")]
unsafe impl std::alloc::GlobalAlloc for StockAlloc {
	#[inline]
	unsafe fn alloc(&self, layout: std::alloc::Layout) -> *mut u8 {
		if layout.size() == 0 {
			return layout.align() as *mut u8;
		}
		unsafe { mallocx(layout.size(), Self::align_flags(layout)) as *mut u8 }
	}

	#[inline]
	unsafe fn alloc_zeroed(&self, layout: std::alloc::Layout) -> *mut u8 {
		if layout.size() == 0 {
			return layout.align() as *mut u8;
		}
		let flags = Self::align_flags(layout) | MALLOCX_ZERO;
		unsafe { mallocx(layout.size(), flags) as *mut u8 }
	}

	#[inline]
	unsafe fn dealloc(&self, ptr: *mut u8, layout: std::alloc::Layout) {
		if layout.size() == 0 {
			return;
		}
		unsafe { sdallocx(ptr as *mut c_void, layout.size(), Self::align_flags(layout)) }
	}
}


// ---------------------------------------------------------------------------
// slow-tier tcache
// ---------------------------------------------------------------------------

/// Whether the slow tier runs on its own per-thread tcache when
/// `SLOW_TCACHE` is unset: no, for now. The prototype's single runs
/// (CLAUDE.md, "Natural alignment for both tiers") gained 10% throughput on
/// cluster53 and lost 35% on cluster50, a loss not yet repeated or explained;
/// the default stays off until it is.
const SLOW_TCACHE_DEFAULT: bool = false;

/// `SLOW_TCACHE`: `1` gives the slow tier a per-thread explicit jemalloc
/// tcache (`slow_tcache_flag`), `0` runs it uncached (`MALLOCX_TCACHE_NONE` on
/// every slow alloc, realloc and free, as before); unset or anything else is
/// [`SLOW_TCACHE_DEFAULT`]. Read once. `std::env` is safe here, unlike in
/// [`natural_align_enabled`]: it allocates through the GLOBAL allocator
/// (node 0), never the slow pool, so the read cannot re-enter itself, and it
/// takes std's environment lock against the tests' `set_var`.
#[inline]
fn slow_tcache_enabled() -> bool {
	static ON: OnceLock<bool> = OnceLock::new();
	*ON.get_or_init(|| match std::env::var_os("SLOW_TCACHE") {
		Some(v) if v == "1" => true,
		Some(v) if v == "0" => false,
		_ => SLOW_TCACHE_DEFAULT,
	})
}

/// Sentinel in `SLOW_TCACHE`: creation failed on this thread, run uncached.
const SLOW_TCACHE_FAILED: u32 = u32::MAX - 1;

/// Called when a slow-pool allocation took the unbound fallback. From then on
/// the slow tier allocates and frees with `MALLOCX_TCACHE_NONE`, so that
/// unbound block can never enter a slow tcache. Relaxed is enough: the store
/// is sequenced before the pointer is returned, and any thread that frees the
/// block received it through some synchronising hand-off after that, so the
/// store happens-before the freeing thread's load.
#[cold]
fn poison_slow_tcache() {
	SLOW_TCACHE_POISONED.store(true, Ordering::Relaxed);
}

/// The tcache bits for every slow-tier `mallocx`, `rallocx` and `sdallocx`:
/// `MALLOCX_TCACHE_NONE` unless `SLOW_TCACHE=1` (and nothing has poisoned
/// it), else this thread's own slow tcache, created on first use.
///
/// # Why this keeps the placement guarantee
///
/// A jemalloc tcache does not consult the arena on a hit: `tcache_alloc_*`
/// pops whatever its bin holds, whichever arena -- and so node -- the block
/// came from. That is why the slow tier ran uncached: the DEFAULT tcache is
/// shared by every allocation the thread makes, so a node-0 block freed into
/// it would be handed to a node-2 request. The guarantee therefore rests on
/// one invariant: **every block in a slow tcache came from a slow-pool
/// (node-2-bound) arena.** It holds because of who can put blocks in and
/// where fills come from:
///
/// 1. *Frees.* Only `NumaAlloc::<NODE_SLOW>::dealloc`, and `rallocx`'s own
///    free of the old block on the slow `realloc` path, pass this flag. Both
///    only ever see blocks the slow pool allocated -- `SlowObjects` and
///    `Hybrid` free only what they allocated (the allocator contract) -- and
///    those came from `mallocx`/`rallocx` with `MALLOCX_ARENA(<slow arena>)`,
///    or from this cache. The only slow-pool block NOT from a slow arena is an
///    unbound fallback (`flags_for` returned `None`); allocating one latches
///    `SLOW_TCACHE_POISONED` before the pointer is returned, and a poisoned
///    process frees every slow block with `MALLOCX_TCACHE_NONE`, so a fallback
///    block can never enter the cache. The fallback itself allocates and
///    reallocates with `MALLOCX_TCACHE_NONE` too, never the DEFAULT tcache:
///    with no tcache flag it would take a fast-tier block out of that cache,
///    and a moving `rallocx` would free the old slow block into it, for the
///    fast tier to hand out (`unbound_slow_fallback_bypasses_the_default_tcache`).
///    A fallback block is not recognisable at its free, so the free side rests
///    on the latch alone.
/// 2. *Fills.* A miss is served from the arena in the call's
///    `MALLOCX_ARENA` -- always a slow one, because this flag is only ever
///    added next to `flags_for(NODE_SLOW)`'s arena bits.
/// 3. *Other pools.* No other call site uses this tcache id: the fast tier
///    uses the default tcache, the value pool `VALUE_TCACHE`, and the id lives
///    in a thread-local no other flag is built from.
///
/// Cross-thread frees are fine: a migration thread freeing a CXL copy that a
/// client thread allocated puts it in the MIGRATION thread's slow cache, still
/// a slow-arena block in a slow-only cache. Which slow arena (slot) it came
/// from does not matter -- every slow arena is bound to the same node -- and
/// a flush returns each block to its own arena, looked up from its extent
/// (`tcache_bin_flush_impl`), not to the tcache's associated one.
/// `slow_tcache_cross_thread_placement` checks node and arena for every
/// block of that pattern.
///
/// # Costs and caveats
///
/// - Only classes up to `opt.tcache_max` (32 KiB by default in 5.3) are
///   cached; larger ones go to the arena as before. `_RJEM_MALLOC_CONF=
///   tcache_max:<bytes>` raises it, process-wide (the default and value
///   tcaches too).
/// - Classes above 14 KiB are only served FROM the cache with natural
///   alignment on: under `ALLOC_NATURAL_ALIGN=0` their allocations take the
///   aligned path past it (see the module docs) while their frees still enter
///   it, so it caches only the small classes and churns the large ones.
/// - Memory parked in the cache is slow-node memory the cache's accounting
///   (`nallocx` per live allocation) does not see. Bounded per thread by the
///   bin limits (5.3: up to 200 slots per small class, 20 per large class up
///   to `tcache_max`). An explicit tcache gets no incremental GC -- jemalloc
///   runs that (`tcache_event`) on a thread's own default tcache only -- so
///   its bins are trimmed only when one fills and flushes.
/// - jemalloc's per-arena stats for the slow arenas stop being exact (blocks
///   in a tcache count as allocated). `measured` is unaffected.
/// - THREAD EXIT: an explicit tcache is not destroyed when its thread exits
///   (`tcache_cleanup` destroys only the thread's own default tcache, and
///   nothing here calls `tcache.destroy`). Its cached blocks stay stranded,
///   bounded as above, and its id stays taken; jemalloc allows
///   `MALLOCX_TCACHE_MAX` (4093) explicit tcaches, past which `tcache.create`
///   fails and that thread runs uncached. Fine for the benchmark, whose
///   client, policy-worker and migration threads live for the process; a
///   server with a thread per connection would want a `thread_local!` guard
///   whose `Drop` calls `tcache.destroy`. `VALUE_TCACHE` has always had the
///   same property.
#[inline]
fn slow_tcache_flag() -> c_int {
	if !slow_tcache_enabled() || SLOW_TCACHE_POISONED.load(Ordering::Relaxed) {
		return mallocx_tcache_none();
	}

	let existing = SLOW_TCACHE.with(|t| t.get());
	if existing == SLOW_TCACHE_FAILED {
		return mallocx_tcache_none();
	}
	if existing != u32::MAX {
		return mallocx_tcache(existing);
	}

	// Same reentrancy guard as `value_tcache_flag`.
	if IN_TCACHE_INIT.with(|f| f.get()) {
		return mallocx_tcache_none();
	}

	IN_TCACHE_INIT.with(|f| f.set(true));

	let mut id: c_uint = 0;
	let mut sz = size_of::<c_uint>();
	let rc = unsafe {
		mallctl(
			c"tcache.create".as_ptr(),
			(&raw mut id).cast(),
			&raw mut sz,
			std::ptr::null_mut(),
			0,
		)
	};

	IN_TCACHE_INIT.with(|f| f.set(false));

	// Creation failed: uncached on this thread from now on -- never the
	// default cache, which would defeat the whole point.
	if rc != 0 || id >= SLOW_TCACHE_FAILED {
		SLOW_TCACHE.with(|t| t.set(SLOW_TCACHE_FAILED));
		SLOW_TCACHE_CREATE_FAILED.fetch_add(1, Ordering::Relaxed);
		return mallocx_tcache_none();
	}

	SLOW_TCACHE.with(|t| t.set(id));
	SLOW_TCACHES_CREATED.fetch_add(1, Ordering::Relaxed);

	mallocx_tcache(id)
}

/// One line for a run log: the effective tcache limits (to see that a
/// `_RJEM_MALLOC_CONF=tcache_max:...` took effect), the two allocator knobs
/// as read, and the slow tcaches' state. `paper_cache::jemalloc_stats`
/// appends it as its second line.
pub fn tcache_report() -> String {
	fn read<T: Default + Copy>(name: &std::ffi::CStr) -> Option<T> {
		let mut v = T::default();
		let mut len = size_of::<T>();
		let rc = unsafe { mallctl(name.as_ptr(), (&raw mut v).cast(), &raw mut len, std::ptr::null_mut(), 0) };
		(rc == 0).then_some(v)
	}
	let show = |v: Option<i64>| v.map_or("-1".to_string(), |v| v.to_string());

	format!(
		"TCACHE opt.tcache={} opt.tcache_max={} arenas.tcache_max={} natural_align={} slow_tcache={} \
slow_tcache_poisoned={} slow_tcaches_created={} slow_tcache_create_failed={} value_tcaches_created={}",
		show(read::<bool>(c"opt.tcache").map(i64::from)),
		show(read::<size_t>(c"opt.tcache_max").map(|v| v as i64)),
		show(read::<size_t>(c"arenas.tcache_max").map(|v| v as i64)),
		natural_align_enabled() as u8,
		slow_tcache_enabled() as u8,
		SLOW_TCACHE_POISONED.load(Ordering::Relaxed) as u8,
		SLOW_TCACHES_CREATED.load(Ordering::Relaxed),
		SLOW_TCACHE_CREATE_FAILED.load(Ordering::Relaxed),
		TCACHES_CREATED.load(Ordering::Relaxed),
	)
}

// ---------------------------------------------------------------------------
// segregated value pool
// ---------------------------------------------------------------------------

/// This thread's explicit tcache for value-buffer allocations.
///
/// Unlike the slow tier -- which runs uncached, or on a slow-only cache of its
/// own (`slow_tcache_flag`), because a shared cache would recycle a node-0
/// block into a node-2 allocation -- the
/// value pool is on the same physical node as everything else, so there is no
/// placement hazard and caching is pure upside. It gets its OWN tcache rather
/// than the default one: an explicit cache keeps value traffic from evicting
/// the `to_vec()` destinations out of the default cache, which is the entire
/// point of the pool. Measured: an explicit tcache held GET p50 at 356 ns
/// (vs 351 with no cache at all) while improving SET p50 from 1055 to 915 ns,
/// so caching costs nothing on the read path and pays on the write path.
///
/// Falls back to `MALLOCX_TCACHE_NONE` if `tcache.create` fails -- correct but
/// uncached, and never the default cache, which would defeat the segregation.
#[inline]
fn value_tcache_flag() -> c_int {
	let existing = VALUE_TCACHE.with(|t| t.get());
	if existing != u32::MAX {
		return mallocx_tcache(existing);
	}

	// `tcache.create` allocates, which re-enters this function; the guard
	// makes that inner allocation uncached rather than recursive.
	if IN_TCACHE_INIT.with(|f| f.get()) {
		return mallocx_tcache_none();
	}

	IN_TCACHE_INIT.with(|f| f.set(true));

	let mut id: c_uint = 0;
	let mut sz = size_of::<c_uint>();
	let rc = unsafe {
		mallctl(
			c"tcache.create".as_ptr(),
			(&raw mut id).cast(),
			&raw mut sz,
			std::ptr::null_mut(),
			0,
		)
	};

	IN_TCACHE_INIT.with(|f| f.set(false));

	if rc != 0 {
		return mallocx_tcache_none();
	}

	VALUE_TCACHE.with(|t| t.set(id));
	TCACHES_CREATED.fetch_add(1, Ordering::Relaxed);

	mallocx_tcache(id)
}

/// The segregated DRAM value pool as a plain unit struct.
///
/// Same two-namespace reason as [`SlowObjects`]: `Box<[u8], FastValues>` needs
/// the type and building one from bytes needs the value.
/// Delegates everything to `NumaAlloc<NODE_FAST_VALUES>`.
#[derive(Clone, Copy, Default)]
pub struct FastValues;

unsafe impl std::alloc::GlobalAlloc for FastValues {
	unsafe fn alloc(&self, layout: std::alloc::Layout) -> *mut u8 {
		unsafe { std::alloc::GlobalAlloc::alloc(&NumaAlloc::<NODE_FAST_VALUES>, layout) }
	}

	unsafe fn alloc_zeroed(&self, layout: std::alloc::Layout) -> *mut u8 {
		unsafe { std::alloc::GlobalAlloc::alloc_zeroed(&NumaAlloc::<NODE_FAST_VALUES>, layout) }
	}

	unsafe fn realloc(
		&self,
		ptr: *mut u8,
		layout: std::alloc::Layout,
		new_size: usize,
	) -> *mut u8 {
		unsafe {
			std::alloc::GlobalAlloc::realloc(&NumaAlloc::<NODE_FAST_VALUES>, ptr, layout, new_size)
		}
	}

	unsafe fn dealloc(&self, ptr: *mut u8, layout: std::alloc::Layout) {
		unsafe { std::alloc::GlobalAlloc::dealloc(&NumaAlloc::<NODE_FAST_VALUES>, ptr, layout) }
	}
}

unsafe impl std::alloc::Allocator for FastValues {
	fn allocate(
		&self,
		layout: std::alloc::Layout,
	) -> Result<std::ptr::NonNull<[u8]>, std::alloc::AllocError> {
		std::alloc::Allocator::allocate(&NumaAlloc::<NODE_FAST_VALUES>, layout)
	}

	fn allocate_zeroed(
		&self,
		layout: std::alloc::Layout,
	) -> Result<std::ptr::NonNull<[u8]>, std::alloc::AllocError> {
		std::alloc::Allocator::allocate_zeroed(&NumaAlloc::<NODE_FAST_VALUES>, layout)
	}

	unsafe fn deallocate(&self, ptr: std::ptr::NonNull<u8>, layout: std::alloc::Layout) {
		unsafe { std::alloc::Allocator::deallocate(&NumaAlloc::<NODE_FAST_VALUES>, ptr, layout) }
	}
}

unsafe impl allocator_api2::alloc::Allocator for FastValues {
	fn allocate(
		&self,
		layout: std::alloc::Layout,
	) -> Result<std::ptr::NonNull<[u8]>, allocator_api2::alloc::AllocError> {
		allocator_api2::alloc::Allocator::allocate(&NumaAlloc::<NODE_FAST_VALUES>, layout)
	}

	fn allocate_zeroed(
		&self,
		layout: std::alloc::Layout,
	) -> Result<std::ptr::NonNull<[u8]>, allocator_api2::alloc::AllocError> {
		allocator_api2::alloc::Allocator::allocate_zeroed(&NumaAlloc::<NODE_FAST_VALUES>, layout)
	}

	unsafe fn deallocate(&self, ptr: std::ptr::NonNull<u8>, layout: std::alloc::Layout) {
		unsafe {
			allocator_api2::alloc::Allocator::deallocate(&NumaAlloc::<NODE_FAST_VALUES>, ptr, layout)
		}
	}
}
