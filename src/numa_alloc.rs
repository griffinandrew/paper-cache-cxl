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
//! spawned. [`assert_numa_environment`] reports whether one is in effect.
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

use std::ffi::c_char;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicU64, Ordering};

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
	addr: *mut c_void,
	_size: size_t,
	offset: size_t,
	length: size_t,
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
/// it mixes scheduler queueing into the figure. Override with
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
pub const NODE_FAST: u32 = 0;
pub const NODE_SLOW: u32 = 1;

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

	/// This thread's explicit tcache for slow-tier allocations.
	///
	/// The slow tier cannot share the default tcache with the fast tier: a
	/// cache bin is indexed by size class alone and `tcache_alloc_small`
	/// returns whatever block it holds without consulting the arena, so one
	/// cache serving both nodes would hand a node-0 block to a node-1
	/// request. Contrast TBB, which gets this isolation structurally by
	/// having two independent pools rather than one allocator with several
	/// arenas.
	///
	/// Per *thread*, not per arena slot: arenas are lock-protected and safe to
	/// share, tcaches are not, so two threads mapped to the same slot must
	/// still hold different tcaches.
	static SLOW_TCACHE: std::cell::Cell<u32> = const { std::cell::Cell::new(u32::MAX) };

	/// Guards the allocation `tcache.create` itself performs.
	static IN_TCACHE_INIT: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// Number of explicit tcaches created, for observability.
static TCACHES_CREATED: AtomicU64 = AtomicU64::new(0);

/// This thread's slow-tier tcache flag, creating the tcache on first use.
///
/// Returns `MALLOCX_TCACHE_NONE` if one cannot be created, which is correct
/// but uncached -- never the default tcache, which would reintroduce the
/// cross-node mixing this exists to prevent.
/// Whether slow-tier caching is on. Off by default; `PAPER_NUMA_SLOW_TCACHE=1`
/// enables it.
///
/// Measured on cluster12 (15 GB cap / 5 GB fast tier), caching against
/// `MALLOCX_TCACHE_NONE`, exact two-sided permutation tests:
///
/// ```text
///            1 client (n=3/2)        8 clients (n=4/5)
///   SET      -0.91%  not resolved    -1.00%  p=0.603
///   GET      -0.36%  not resolved    -0.86%  p=0.294
///   RSS      +0.13%  not resolved    +5.49%  p=0.056
/// ```
///
/// Re-measured later on 8 arenas per node with the `dalloc`-under-retain fix
/// in place (cluster12, 4M accesses, 15 GB / 1 GB, n=3 per arm):
///
/// ```text
///            1 client              8 clients
///   GET      +0.41%  p=0.30        -2.31%  p=0.20
///   SET      +0.18%  p=0.50        -3.45%  p=0.50
///   RSS      +0.00%  p=1.00        +0.35%  p=1.00
/// ```
///
/// Directionally better than the first measurement -- the RSS cost is gone
/// and the concurrent latencies favour it -- but nothing clears significance
/// (n=3v3 floors p at 0.10).
///
/// It also does not compose with worker binding, which is on by default: at
/// 8 clients binding alone moves SET -7.00%, while binding plus this knob
/// moves it only -2.52%. Whatever binding gains, caching gives most of it
/// back. Left off for that reason; `PAPER_NUMA_SLOW_TCACHE=1` enables it.
///
/// Correctness is settled independently of the performance question, see
/// `tcache_hits_preserve_node_placement` and
/// `concurrent_tcache_creation_is_safe`.
///
/// Read with `getenv` rather than `std::env::var` because this is reached from
/// inside allocation and must not itself allocate.
fn slow_tcache_enabled() -> bool {
	static ENABLED: OnceLock<bool> = OnceLock::new();
	*ENABLED.get_or_init(|| unsafe {
		let raw = libc::getenv(c"PAPER_NUMA_SLOW_TCACHE".as_ptr());
		!raw.is_null() && *raw == b'1' as libc::c_char
	})
}

#[inline]
fn slow_tcache_flag() -> c_int {
	if !slow_tcache_enabled() {
		return mallocx_tcache_none();
	}

	let existing = SLOW_TCACHE.with(|t| t.get());
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

	SLOW_TCACHE.with(|t| t.set(id));
	TCACHES_CREATED.fetch_add(1, Ordering::Relaxed);

	mallocx_tcache(id)
}

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
		let (ind, h) = create_node_arena(node)?;
		indices[slot] = ind;
		hooks[slot] = Some(h);
	}

	Some(NodeArenas { indices, hooks, count })
}

/// Arenas for `node`, built on first use.
#[inline]
fn node_arenas(node: u32) -> Option<&'static NodeArenas> {
	let cell = if node == NODE_FAST { &FAST_ARENAS } else { &SLOW_ARENAS };

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

/// `mallocx` flags for an allocation on `node`.
///
/// The fast tier uses the default per-thread cache; the slow tier disables
/// caching entirely (`MALLOCX_TCACHE_NONE`). That asymmetry is load-bearing: a
/// tcache hands back whatever block it holds without consulting the arena, so
/// one cache serving both nodes would recycle a node-0 block into a node-1
/// allocation. Nothing crashes -- placement just decays, and only under
/// cross-thread free patterns, which is exactly this workload. Keeping the
/// slow tier out of the cache makes that impossible by construction.
#[inline]
fn flags_for(node: u32, align: usize) -> Option<c_int> {
	let arenas = node_arenas(node)?;
	let mut flags = mallocx_arena(arenas.indices[slot_for_thread(arenas.count)]);

	if node != NODE_FAST {
		flags |= slow_tcache_flag();
	}

	if align > 1 {
		flags |= mallocx_lg_align(align.trailing_zeros());
	}

	Some(flags)
}

/// `MALLOCX_TCACHE_NONE` is `MALLOCX_TCACHE(-1)`.
const fn mallocx_tcache_none() -> c_int {
	((-1i32) + 2) << 8
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

/// The node-1 allocator as a plain unit struct.
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


impl<const NODE: u32> NumaAlloc<NODE> {
	#[inline]
	unsafe fn raw_alloc(&self, layout: std::alloc::Layout, zeroed: bool) -> *mut u8 {
		if layout.size() == 0 {
			return layout.align() as *mut u8;
		}

		match flags_for(NODE, layout.align()) {
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
				let flags = if zeroed { MALLOCX_ZERO } else { 0 };
				unsafe { mallocx(layout.size(), flags) as *mut u8 }
			},
		}
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
		// by a thread in a different slot. The tcache flag *does* matter: it
		// decides which cache the block lands in, so the slow tier must bypass
		// the cache here for the same reason it does on alloc.
		// Must match the alloc path: the tcache flag decides which cache the
		// block lands in, so freeing a slow-tier block into the default cache
		// would put a node-1 block where a node-0 allocation could take it.
		let mut flags = if NODE == NODE_FAST { 0 } else { slow_tcache_flag() };

		if layout.align() > 1 {
			flags |= mallocx_lg_align(layout.align().trailing_zeros());
		}

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

		match flags_for(NODE, layout.align()) {
			Some(flags) => unsafe { rallocx(ptr as *mut c_void, new_size, flags) as *mut u8 },
			None => {
				UNBOUND_FALLBACKS.fetch_add(1, Ordering::Relaxed);
				unsafe { rallocx(ptr as *mut c_void, new_size, 0) as *mut u8 }
			},
		}
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
/// `PAPER_BIND_WORKERS=1` targets node 1; any other value disables binding.
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
/// Neither failure is observable from inside the process, and both knobs are
/// writable at runtime, so they are checked rather than assumed. `mbind` binds
/// the VMA, but page *migration* is a separate mechanism that operates on
/// already-placed pages.
pub fn assert_numa_environment() -> Result<(), String> {
	fn read_flag(path: &str) -> Option<i32> {
		std::fs::read_to_string(path).ok()?.trim().parse().ok()
	}

	let mut problems = Vec::new();

	if read_flag("/proc/sys/kernel/numa_balancing") == Some(1) {
		problems.push(
			"kernel.numa_balancing=1 -- automatic balancing migrates pages between nodes, \
			 defeating mbind placement"
				.to_string(),
		);
	}

	// A CPU-less node is exactly what the kernel treats as a demotion target,
	// which makes this the relevant knob for a PMEM/CXL slow tier.
	if read_flag("/sys/kernel/mm/numa/demotion_enabled") == Some(1) {
		problems.push(
			"numa demotion_enabled=1 -- cold pages on node 0 may be demoted to the \
			 CPU-less node under pressure"
				.to_string(),
		);
	}

	if problems.is_empty() { Ok(()) } else { Err(problems.join("; ")) }
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
unbound_fallbacks={} slow_tcache={} slow_tcaches={}",
		fm.saturating_sub(fu),
		sm.saturating_sub(su),
		UNBOUND_FALLBACKS.load(Ordering::Relaxed),
		if slow_tcache_enabled() { "on" } else { "off" },
		TCACHES_CREATED.load(Ordering::Relaxed),
	)
}

/// Ground-truth placement, read from the kernel rather than inferred.
///
/// Walks `/proc/self/numa_maps` and totals the resident pages this process has
/// on each node. Unlike the counters above -- which record what was *asked
/// for* -- this reports where pages actually are, so the two disagreeing is
/// itself the signal.
pub fn resident_pages_per_node() -> Result<(u64, u64), std::io::Error> {
	let maps = std::fs::read_to_string("/proc/self/numa_maps")?;
	let mut node0 = 0u64;
	let mut node1 = 0u64;

	for line in maps.lines() {
		for field in line.split_whitespace() {
			if let Some(count) = field.strip_prefix("N0=") {
				node0 += count.parse::<u64>().unwrap_or(0);
			} else if let Some(count) = field.strip_prefix("N1=") {
				node1 += count.parse::<u64>().unwrap_or(0);
			}
		}
	}

	Ok((node0, node1))
}

// ---------------------------------------------------------------------------
// tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
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

		let (before0, before1) = resident_pages_per_node().expect("numa_maps readable");

		let layout = Layout::from_size_align(BYTES, page).unwrap();
		let ptr = alloc(layout);
		assert!(!ptr.is_null(), "allocation failed");

		// Fault every page in.
		unsafe {
			for offset in (0..BYTES).step_by(page) {
				std::ptr::write_volatile(ptr.add(offset), 1u8);
			}
		}

		let (after0, after1) = resident_pages_per_node().expect("numa_maps readable");
		let grew0 = after0.saturating_sub(before0);
		let grew1 = after1.saturating_sub(before1);
		let expected = (BYTES / page) as u64;

		let (target, other) = if node == NODE_FAST { (grew0, grew1) } else { (grew1, grew0) };

		println!(
			"PLACEMENT node={node} requested={BYTES}B expected_pages={expected} \
node0_grew={grew0} node1_grew={grew1} -> on_target={target} elsewhere={other}"
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
	fn slow_tier_allocations_land_on_node_1() {
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
		// configured such that a binding can hold at all.
		if let Err(problems) = assert_numa_environment() {
			panic!("environment defeats NUMA binding: {problems}");
		}
	}

	/// Query which NUMA node a specific address's page is on.
	///
	/// `numa_maps` is per-mapping and far too coarse for tcache-sized blocks,
	/// which live inside a shared extent; `get_mempolicy` answers for one
	/// address.
	fn node_of(ptr: *const u8) -> i32 {
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

	/// Blocks served *from* a tcache must still be on the right node.
	///
	/// This is the case the 256 MB `placement_check` tests cannot reach:
	/// jemalloc's tcache only caches up to `opt.tcache_max` (32 KiB default),
	/// so every existing placement test bypasses the cache entirely and would
	/// pass even with the tcache flags wrong.
	///
	/// The hazard is specific: a cache bin is indexed by size class alone, and
	/// on a hit `tcache_alloc_small` returns the cached block without ever
	/// consulting `MALLOCX_ARENA`. So the arena flag is decorative on the hit
	/// path, and correctness rests entirely on a cache never holding two
	/// nodes' blocks. Interleaving same-size fast and slow allocations on one
	/// thread, then freeing both, is the shortest path to violating that if
	/// the flags are wrong.
	#[test]
	#[ignore = "needs PAPER_NUMA_SLOW_TCACHE=1; slow-tier tcache is off by default, so this returns without asserting anything"]
	fn tcache_hits_preserve_node_placement() {
		// Off by default, so this must opt in or it would assert on the
		// uncached path and quietly stop testing what it names.
		if !slow_tcache_enabled() {
			eprintln!("skipping: set PAPER_NUMA_SLOW_TCACHE=1 to test the cached path");
			return;
		}

		assert!(init(), "arena pool must build");

		// Comfortably under tcache_max so these are cached, not extent-backed.
		const BLOCK: usize = 4096;
		const N: usize = 512;
		const ROUNDS: usize = 6;

		let layout = Layout::from_size_align(BLOCK, 64).unwrap();
		let mut checked = 0usize;

		// Pool construction itself allocates before the arenas exist, so a
		// non-zero baseline is expected and is not what this test is about.
		// Only the delta across the churn below can be attributed to tcaches.
		let unbound_before = UNBOUND_FALLBACKS.load(Ordering::Relaxed);

		for round in 0..ROUNDS {
			let mut fast = Vec::with_capacity(N);
			let mut slow = Vec::with_capacity(N);

			// Interleave so both tiers churn the same size class together.
			for _ in 0..N {
				let f = FastAlloc::default().allocate(layout).expect("fast");
				let s = SlowAlloc::default().allocate(layout).expect("slow");
				unsafe {
					std::ptr::write_volatile(f.as_ptr().cast::<u8>(), 1u8);
					std::ptr::write_volatile(s.as_ptr().cast::<u8>(), 1u8);
				}
				fast.push(f);
				slow.push(s);
			}

			// Rounds after the first are served from the tcache the frees below
			// populated -- that is the path under test.
			if round > 0 {
				for p in &fast {
					let n = node_of(p.as_ptr().cast());
					assert_eq!(
						n, NODE_FAST as i32,
						"round {round}: fast block on node {n}; a tcache served \
						 a node-1 block to a fast-tier request"
					);
					checked += 1;
				}
				for p in &slow {
					let n = node_of(p.as_ptr().cast());
					assert_eq!(
						n, NODE_SLOW as i32,
						"round {round}: slow block on node {n}; a tcache served \
						 a node-0 block to a slow-tier request"
					);
					checked += 1;
				}
			}

			unsafe {
				for p in fast { FastAlloc::default().deallocate(p.cast(), layout); }
				for p in slow { SlowAlloc::default().deallocate(p.cast(), layout); }
			}
		}

		let report = stats();
		println!("TCACHE PLACEMENT checked={checked} blocks -- {report}");
		assert!(checked > 0, "no cached blocks were verified");
		assert!(
			report.contains("mbind_failed=0"),
			"an mbind failed: {report}"
		);
		let leaked = UNBOUND_FALLBACKS.load(Ordering::Relaxed) - unbound_before;
		println!("TCACHE unbound_delta={leaked} (baseline {unbound_before})");
		assert_eq!(
			leaked, 0,
			"{leaked} allocations fell through to unbound memory during tcache \
			 churn (baseline was {unbound_before}): {report}"
		);
	}

	/// Many threads reaching tcache creation at once must not deadlock.
	///
	/// `tcache.create` allocates, so it re-enters the allocator on the very
	/// path that is trying to set it up. The previous implementation
	/// deadlocked on exactly this shape of reentrancy, so the guard is tested
	/// under contention rather than on one thread.
	#[test]
	#[ignore = "needs PAPER_NUMA_SLOW_TCACHE=1; slow-tier tcache is off by default, so this returns without asserting anything"]
	fn concurrent_tcache_creation_is_safe() {
		// Off by default, so this must opt in or it would assert on the
		// uncached path and quietly stop testing what it names.
		if !slow_tcache_enabled() {
			eprintln!("skipping: set PAPER_NUMA_SLOW_TCACHE=1 to test the cached path");
			return;
		}

		assert!(init(), "arena pool must build");

		const THREADS: usize = 32;
		const PER_THREAD: usize = 400;
		const BLOCK: usize = 2048;

		let before = TCACHES_CREATED.load(Ordering::Relaxed);
		let barrier = std::sync::Arc::new(std::sync::Barrier::new(THREADS));
		let bad = std::sync::Arc::new(AtomicU64::new(0));

		let handles: Vec<_> = (0..THREADS)
			.map(|_| {
				let barrier = barrier.clone();
				let bad = bad.clone();
				std::thread::spawn(move || {
					let layout = Layout::from_size_align(BLOCK, 64).unwrap();
					// Release every thread into its first slow allocation
					// simultaneously -- that is the creation path.
					barrier.wait();
					for _ in 0..PER_THREAD {
						let f = FastAlloc::default().allocate(layout).expect("fast");
						let s = SlowAlloc::default().allocate(layout).expect("slow");
						unsafe {
							std::ptr::write_volatile(f.as_ptr().cast::<u8>(), 1u8);
							std::ptr::write_volatile(s.as_ptr().cast::<u8>(), 1u8);
						}
						if node_of(f.as_ptr().cast()) != NODE_FAST as i32
							|| node_of(s.as_ptr().cast()) != NODE_SLOW as i32
						{
							bad.fetch_add(1, Ordering::Relaxed);
						}
						unsafe {
							FastAlloc::default().deallocate(f.cast(), layout);
							SlowAlloc::default().deallocate(s.cast(), layout);
						}
					}
				})
			})
			.collect();

		for h in handles {
			h.join().expect("no thread may panic or hang");
		}

		let created = TCACHES_CREATED.load(Ordering::Relaxed) - before;
		let misplaced = bad.load(Ordering::Relaxed);
		println!(
			"CONCURRENT threads={THREADS} tcaches_created={created} misplaced={misplaced} -- {}",
			stats()
		);
		assert_eq!(misplaced, 0, "{misplaced} blocks landed on the wrong node");
		// `>=`, not `==`: the counter is process-global, so if another test
		// runs concurrently with the tcache enabled it also creates tcaches.
		// What this test needs is that each of its own threads got one.
		assert!(
			created >= THREADS as u64,
			"expected at least one tcache per thread ({THREADS}), got {created}"
		);
	}

	/// Quantify what departing threads leave behind.
	///
	/// Explicit tcaches are process-lived, not thread-lived: jemalloc does not
	/// destroy one when the thread that created it exits, so its cached blocks
	/// stay retained. Long-lived workers make this bounded, but it is a real
	/// cost under thread churn and worth a number rather than a caveat.
	#[test]
	#[ignore = "needs PAPER_NUMA_SLOW_TCACHE=1; slow-tier tcache is off by default, so this returns without asserting anything"]
	fn departed_threads_leak_their_tcaches() {
		// Off by default, so this must opt in -- with caching disabled no
		// tcache is created and there is nothing to leak.
		if !slow_tcache_enabled() {
			eprintln!("skipping: set PAPER_NUMA_SLOW_TCACHE=1 to measure tcache retention");
			return;
		}

		assert!(init(), "arena pool must build");

		const CHURN: usize = 200;
		let before = TCACHES_CREATED.load(Ordering::Relaxed);
		let slow_before = SLOW_ARENAS
			.get()
			.and_then(|a| a.as_ref())
			.map(|a| a.hooks.iter().flatten().map(|h| {
				h.mapped_bytes.load(Ordering::Relaxed) - h.unmapped_bytes.load(Ordering::Relaxed)
			}).sum::<u64>())
			.unwrap_or(0);

		for _ in 0..CHURN {
			std::thread::spawn(|| {
				let layout = Layout::from_size_align(4096, 64).unwrap();
				let p = SlowAlloc::default().allocate(layout).expect("slow");
				unsafe {
					std::ptr::write_volatile(p.as_ptr().cast::<u8>(), 1u8);
					SlowAlloc::default().deallocate(p.cast(), layout);
				}
			})
			.join()
			.expect("thread");
		}

		let created = TCACHES_CREATED.load(Ordering::Relaxed) - before;
		let slow_after = SLOW_ARENAS
			.get()
			.and_then(|a| a.as_ref())
			.map(|a| a.hooks.iter().flatten().map(|h| {
				h.mapped_bytes.load(Ordering::Relaxed) - h.unmapped_bytes.load(Ordering::Relaxed)
			}).sum::<u64>())
			.unwrap_or(0);

		println!(
			"CHURN threads={CHURN} tcaches_created={created} \
			 slow_live_delta={} bytes ({:.2} KiB/thread)",
			slow_after as i64 - slow_before as i64,
			(slow_after as i64 - slow_before as i64) as f64 / CHURN as f64 / 1024.0
		);

		// Documents the behaviour rather than asserting it away: one tcache
		// per departed thread, never reclaimed.
		// `>=` for the same reason as `concurrent_tcache_creation_is_safe`:
		// the counter is process-global.
		assert!(
			created >= CHURN as u64,
			"expected at least one tcache per churned thread ({CHURN}), got {created}"
		);
	}

}
