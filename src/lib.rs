/*
 * Copyright (c) Kia Shakiba
 *
 * This source code is licensed under the GNU AGPLv3 license found in the
 * LICENSE file in the root directory of this source tree.
 * correct
 */

#![cfg_attr(any(feature = "hashbrown_dram", feature = "all_dram", feature = "key_value_pmem", feature = "global_hashtable_pmem", feature = "tiering_hashtable_pmem", feature = "eviction_stacks_pmem"), feature(allocator_api), feature(clone_from_ref), feature(btreemap_alloc))]


// Validate that hashbrown_dram is not enabled with other global hashtable features
#[cfg(all(feature = "hashbrown_dram", feature = "global_hashtable_pmem"))]
compile_error!("Cannot enable both 'hashbrown_dram' and 'global_hashtable_pmem' features simultaneously. Please choose only one global hashtable mode.");


/// Node-0-bound jemalloc arenas as the process allocator.
///
/// Covers everything Rust allocates. It does NOT cover glibc's heap, the C
/// libraries reached through bindgen, or pthread stacks --
/// jemalloc is built `JEMALLOC_PREFIX=_rjem_` and so does not interpose
/// `malloc`. Pair with `numactl --membind=0` when the whole process must be
/// bound.
#[cfg(not(feature = "stock_jemalloc"))]
#[global_allocator]
static GLOBAL: numa_alloc::FastAlloc = numa_alloc::NumaAlloc;

#[cfg(feature = "stock_jemalloc")]
#[global_allocator]
static GLOBAL_STOCK: numa_alloc::StockAlloc = numa_alloc::StockAlloc;

pub mod numa_alloc;

use std::arch::x86_64::{_mm_clflush, _mm_sfence};

// `Hybrid` is the crate-wide PMEM allocator alias: every PMEM feature routes
// through node-1-bound jemalloc arenas (`numa_alloc::SlowAlloc`).
//
// This was UMF's TBB-backed pool. Both place memory on NUMA node 1 -- the
// "PMEM" features were never using persistent-memory hardware, only far
// memory -- so the swap is an equivalent placement, not an approximation.
// jemalloc measured 16% lower SET latency and 17% lower peak RSS on
// cluster12, and TBB retained ~1.75x the memory in use without returning it.
#[cfg(any(
    feature = "key_value_pmem",
    feature = "key_pmem_value_pmem",
    feature = "global_hashtable_pmem",
    feature = "tiering_hashtable_pmem",
    feature = "eviction_stacks_pmem",
    feature = "segregated_value_arena",
))]
pub(crate) use crate::numa_alloc::SlowObjects as Hybrid;

#[cfg(feature = "key_value_pmem")]
impl typesize::TypeSize for BufferPMEM {
    fn get_size(&self) -> usize {
        self.len()
    }
}

// `typesize` blankets `Box<[T]>` only for the default allocator, so the
// segregated-pool boxed slice needs its own impl, exactly as `BufferPMEM` does.
#[cfg(feature = "segregated_value_arena")]
impl typesize::TypeSize for BufferDRAM {
    fn get_size(&self) -> usize {
        self.len()
    }
}

mod error;
mod worker;
mod object;
mod policy;
mod status;

// Shared object-map storage-backend abstraction and value-buffer
// abstraction (see each module's doc comment) -- used by the generic
// `impl<K, V, S> PaperCache<K, V, S>` blocks below to replace what used to
// be one impl block per (object-map shape, value-buffer type) combination.
#[cfg(any(feature = "all_dram", feature = "key_value_pmem", feature = "global_hashtable_pmem", feature = "hashbrown_dram"))]
mod object_store;
#[cfg(any(feature = "all_dram", feature = "key_value_pmem", feature = "global_hashtable_pmem", feature = "hashbrown_dram"))]
mod value_buffer;

#[cfg(any(feature = "all_dram", feature = "key_value_pmem", feature = "global_hashtable_pmem", feature = "hashbrown_dram"))]
use crate::object_store::ObjectStore;
#[cfg(any(feature = "all_dram", feature = "key_value_pmem", feature = "global_hashtable_pmem", feature = "hashbrown_dram"))]
use crate::value_buffer::ValueBuffer;

// Shared tier-size unit type (bytes/Mb/Gb), used by `lru_hybrid_cache`,
// `lfu_hybrid_cache`, `two_q_hybrid_cache`, and `fifo_hybrid_cache` so none
// of them has to depend on any of the others for it.
#[cfg(feature = "hybrid_cache_common")]
mod size;

#[cfg(feature = "hybrid_cache_common")]
pub use crate::size::CacheTierSize;

// Shared value type for every hybrid design. Each design's module also
// re-exports it for source compatibility, so `paper_cache::TieredBuffer` and
// `paper_cache::<design>_hybrid_cache::TieredBuffer` both work.
#[cfg(feature = "hybrid_cache_common")]
mod tiered_buffer;

#[cfg(feature = "hybrid_cache_common")]
pub use crate::tiered_buffer::TieredBuffer;

// Design-neutral view of whichever hybrid design a cache is running. The only
// stats accessor: the per-design `<design>_hybrid_stats()` methods are gone and
// the `<Design>HybridStats` names are aliases of this one struct. See
// `hybrid_stats.rs`'s module doc.
#[cfg(feature = "hybrid_cache_common")]
mod hybrid_stats;

#[cfg(feature = "hybrid_cache_common")]
pub use crate::hybrid_stats::HybridStats;

#[cfg(all(feature = "key_value_pmem", feature = "enable_tiering_manager"))]
pub mod tiering;

// Single-instance, segmented-LRU hybrid cache: one PaperCache<K, TieredBuffer>.
#[cfg(feature = "lru_hybrid_cache")]
pub mod lru_hybrid_cache;

#[cfg(feature = "lru_hybrid_cache")]
pub use crate::lru_hybrid_cache::LruHybridStats;

// Single-instance, segmented-LFU hybrid cache. Same architecture as
// `lru_hybrid_cache` (one PaperCache<K, TieredBuffer>, not two composed
// instances) but the fast/slow boundary is frequency-ordered rather than
// recency-ordered — see the `lfu_hybrid_cache` module docs.
#[cfg(feature = "lfu_hybrid_cache")]
pub mod lfu_hybrid_cache;

#[cfg(feature = "lfu_hybrid_cache")]
pub use crate::lfu_hybrid_cache::LfuHybridStats;

// Single-instance, segmented-2Q hybrid cache. Same one-PaperCache<K,
// TieredBuffer> architecture as `lru_hybrid_cache`/`lfu_hybrid_cache`, but
// admission always lands in a one-access FIFO queue in the slow tier —
// see the `two_q_hybrid_cache` module docs.
#[cfg(feature = "two_q_hybrid_cache")]
pub mod two_q_hybrid_cache;

#[cfg(feature = "two_q_hybrid_cache")]
pub use crate::two_q_hybrid_cache::TwoQHybridStats;

// Two-tier segmented-2Q hybrid cache with the one-access FIFO queue in the
// FAST tier — same design as `two_q_hybrid_cache` except admission is a DRAM
// write rather than a synchronous PMEM allocation; see that module's docs.
#[cfg(feature = "two_q_fast_admission_hybrid_cache")]
pub mod two_q_fast_admission_hybrid_cache;

#[cfg(feature = "two_q_fast_admission_hybrid_cache")]
pub use crate::two_q_fast_admission_hybrid_cache::TwoQFastAdmissionHybridStats;

// As above, but a one-access object that ages out without a second access is
// reprieved into the slow tier instead of evicted; see that module's docs.
#[cfg(feature = "two_q_fast_admission_reprieve_hybrid_cache")]
pub mod two_q_fast_admission_reprieve_hybrid_cache;

#[cfg(feature = "two_q_fast_admission_reprieve_hybrid_cache")]
pub use crate::two_q_fast_admission_reprieve_hybrid_cache::TwoQFastAdmissionReprieveHybridStats;

// The FULL three-queue 2Q with fast-tier admission -- the only design here
// whose queue algorithm matches `PaperPolicy::TwoQ`'s. `a1_out` holds real
// resident objects in the slow tier rather than ghosts, and `k_out` is a
// live parameter; see that module's docs.
#[cfg(feature = "two_q_full_fast_admission_hybrid_cache")]
pub mod two_q_full_fast_admission_hybrid_cache;

#[cfg(feature = "two_q_full_fast_admission_hybrid_cache")]
pub use crate::two_q_full_fast_admission_hybrid_cache::TwoQFullFastAdmissionHybridStats;

// Single-instance, segmented-FIFO hybrid cache. Same one-PaperCache<K,
// TieredBuffer> architecture as the other three, but with no promotion
// policy at all — an object's position and tier are fixed for life once
// admitted — see the `fifo_hybrid_cache` module docs.
#[cfg(feature = "fifo_hybrid_cache")]
pub mod fifo_hybrid_cache;

#[cfg(feature = "fifo_hybrid_cache")]
pub use crate::fifo_hybrid_cache::FifoHybridStats;

// Single-instance, segmented-LRU hybrid cache with a size-split fast AND
// slow tier. Same one-PaperCache<K, TieredBuffer> architecture and LRU
// admission/promotion/demotion/eviction semantics as `lru_hybrid_cache`, but
// both tiers' bookkeeping are each further split into two size-routed
// segments — see the `lru_sized_hybrid_cache` module docs.
#[cfg(feature = "lru_sized_hybrid_cache")]
pub mod lru_sized_hybrid_cache;

#[cfg(feature = "lru_sized_hybrid_cache")]
pub use crate::lru_sized_hybrid_cache::LruSizedHybridStats;

// Single-instance hybrid cache with a DIFFERENT eviction discipline per tier:
// recency (LRU) in the fast tier, frequency (LFU) in the slow tier. The first
// design here whose two tiers do not rank by the same metric -- see the
// `lru_lfu_hybrid_cache` module docs.
#[cfg(feature = "lru_lfu_hybrid_cache")]
pub mod lru_lfu_hybrid_cache;

#[cfg(feature = "lru_lfu_hybrid_cache")]
pub use crate::lru_lfu_hybrid_cache::LruLfuHybridStats;

// Single-instance, segmented-S3-FIFO hybrid cache. Same one-PaperCache<K,
// TieredBuffer> architecture as the other hybrids. Structurally closest to
// `two_q_hybrid_cache` (a one-access FIFO queue always in the slow tier
// feeding a segmented main queue), but the main queue's promotion is the
// classic S3-FIFO/CLOCK lazy, reference-bit-gated mechanism rather than
// `two_q_hybrid_cache`'s eager, reorder-on-every-touch LRU one — see the
// `s3_fifo_hybrid_cache` module docs.
#[cfg(feature = "s3_fifo_hybrid_cache")]
pub mod s3_fifo_hybrid_cache;

#[cfg(feature = "s3_fifo_hybrid_cache")]
pub use crate::s3_fifo_hybrid_cache::S3FifoHybridStats;

// Single-instance, segmented-2Q hybrid cache with a ghost queue. Same
// architecture as `two_q_hybrid_cache` plus a bare-key ghost queue -- see
// the `two_q_ghost_hybrid_cache` module docs.
#[cfg(feature = "two_q_ghost_hybrid_cache")]
pub mod two_q_ghost_hybrid_cache;

#[cfg(feature = "two_q_ghost_hybrid_cache")]
pub use crate::two_q_ghost_hybrid_cache::TwoQGhostHybridStats;

// Single-instance, segmented-S3-FIFO hybrid cache with a ghost queue. Same
// architecture as `s3_fifo_hybrid_cache` plus a bare-key ghost queue -- see
// the `s3_fifo_ghost_hybrid_cache` module docs.
#[cfg(feature = "s3_fifo_ghost_hybrid_cache")]
pub mod s3_fifo_ghost_hybrid_cache;

#[cfg(feature = "s3_fifo_ghost_hybrid_cache")]
pub use crate::s3_fifo_ghost_hybrid_cache::S3FifoGhostHybridStats;

// `s3_fifo_ghost_hybrid_cache` plus one more change: demotion is now
// reference-bit gated too, not just eviction -- see the
// `s3_fifo_ghost_lazy_demotion_hybrid_cache` module docs.
#[cfg(feature = "s3_fifo_ghost_lazy_demotion_hybrid_cache")]
pub mod s3_fifo_ghost_lazy_demotion_hybrid_cache;

#[cfg(feature = "s3_fifo_ghost_lazy_demotion_hybrid_cache")]
pub use crate::s3_fifo_ghost_lazy_demotion_hybrid_cache::S3FifoGhostLazyDemotionHybridStats;

// `s3_fifo_ghost_lazy_demotion_hybrid_cache` plus one more change: the
// one-access queue now lives in the FAST tier instead of the slow tier --
// see the `s3_fifo_ghost_lazy_demotion_fast_admission_hybrid_cache` module
// docs.
#[cfg(feature = "s3_fifo_ghost_lazy_demotion_fast_admission_hybrid_cache")]
pub mod s3_fifo_ghost_lazy_demotion_fast_admission_hybrid_cache;

#[cfg(feature = "s3_fifo_ghost_lazy_demotion_fast_admission_hybrid_cache")]
pub use crate::s3_fifo_ghost_lazy_demotion_fast_admission_hybrid_cache::S3FifoGhostLazyDemotionFastAdmissionHybridStats;

// `s3_fifo_ghost_lazy_demotion_fast_admission_hybrid_cache` plus one more
// addition: a checkpoint roughly halfway through the SLOW portion of the
// main queue -- see the
// `s3_fifo_ghost_lazy_demotion_fast_admission_midpoint_hybrid_cache` module
// docs.
#[cfg(feature = "s3_fifo_ghost_lazy_demotion_fast_admission_midpoint_hybrid_cache")]
pub mod s3_fifo_ghost_lazy_demotion_fast_admission_midpoint_hybrid_cache;

#[cfg(feature = "s3_fifo_ghost_lazy_demotion_fast_admission_midpoint_hybrid_cache")]
pub use crate::s3_fifo_ghost_lazy_demotion_fast_admission_midpoint_hybrid_cache::S3FifoGhostLazyDemotionFastAdmissionMidpointHybridStats;

// `s3_fifo_ghost_lazy_demotion_fast_admission_midpoint_hybrid_cache` minus
// the ghost queue (removed entirely) -- a one-access-queue key that ages
// out is spliced into the slow tier of the main queue instead of being
// evicted. See the
// `s3_fifo_lazy_demotion_fast_admission_midpoint_reprieve_hybrid_cache`
// module docs.
#[cfg(feature = "s3_fifo_lazy_demotion_fast_admission_midpoint_reprieve_hybrid_cache")]
pub mod s3_fifo_lazy_demotion_fast_admission_midpoint_reprieve_hybrid_cache;

#[cfg(feature = "s3_fifo_lazy_demotion_fast_admission_midpoint_reprieve_hybrid_cache")]
pub use crate::s3_fifo_lazy_demotion_fast_admission_midpoint_reprieve_hybrid_cache::S3FifoLazyDemotionFastAdmissionMidpointReprieveHybridStats;

// The reprieve design with NO mid-slow-tier checkpoint -- see the
// `s3_fifo_lazy_demotion_fast_admission_reprieve_hybrid_cache` module docs for
// why both earlier checkpoint designs were dropped.
#[cfg(feature = "s3_fifo_lazy_demotion_fast_admission_reprieve_hybrid_cache")]
pub mod s3_fifo_lazy_demotion_fast_admission_reprieve_hybrid_cache;

#[cfg(feature = "s3_fifo_lazy_demotion_reprieve_hybrid_cache")]
pub mod s3_fifo_lazy_demotion_reprieve_hybrid_cache;

#[cfg(feature = "s3_fifo_lazy_demotion_fast_admission_reprieve_hybrid_cache")]
pub use crate::s3_fifo_lazy_demotion_fast_admission_reprieve_hybrid_cache::S3FifoLazyDemotionFastAdmissionReprieveHybridStats;

#[cfg(feature = "s3_fifo_lazy_demotion_reprieve_hybrid_cache")]
pub use crate::s3_fifo_lazy_demotion_reprieve_hybrid_cache::S3FifoLazyDemotionReprieveHybridStats;

// `s3_fifo_lazy_demotion_fast_admission_midpoint_reprieve_hybrid_cache` with the
// approximate mid-slow-segment cursor replaced by a real two-segment slow
// tier, checking every object's reference bit as it crosses -- see the
// `s3_fifo_lazy_demotion_fast_admission_split_slow_reprieve_hybrid_cache` module docs.
#[cfg(feature = "s3_fifo_lazy_demotion_fast_admission_split_slow_reprieve_hybrid_cache")]
pub mod s3_fifo_lazy_demotion_fast_admission_split_slow_reprieve_hybrid_cache;

#[cfg(feature = "s3_fifo_lazy_demotion_fast_admission_split_slow_reprieve_hybrid_cache")]
pub use crate::s3_fifo_lazy_demotion_fast_admission_split_slow_reprieve_hybrid_cache::S3FifoLazyDemotionFastAdmissionSplitSlowReprieveHybridStats;

// Re-exported so `PaperCache::tier_of`'s return type is nameable by callers
// without reaching into the private `worker` module tree directly.
#[cfg(feature = "hybrid_cache_common")]
pub use crate::worker::Tier;

// The one thing that still differs per design on the `set()` path: which tier
// a value is built in. A runtime `match` over the cache's `PaperPolicy` -- see
// `hybrid_policy.rs`'s module doc.
#[cfg(feature = "hybrid_cache_common")]
mod hybrid_policy;

use std::{
	sync::{
		Arc,
		atomic::AtomicU64,
	},
	hash::{
		Hash,
		RandomState,
		BuildHasher,
		BuildHasherDefault,
	},
	ptr
};

#[cfg(any(
	feature = "global_hashtable_pmem",
	feature = "hashbrown_dram",
))]
use std::sync::RwLock;

#[cfg(not(any(feature = "global_hashtable_pmem", feature = "hashbrown_dram")))]
use dashmap::{
	DashMap,
	mapref::entry::Entry,
};

#[cfg(any(feature = "global_hashtable_pmem", feature = "hashbrown_dram"))]
use hashbrown::HashMap;

#[cfg(any(feature = "global_hashtable_pmem", feature = "hashbrown_dram"))]
use hashbrown::hash_map::Entry;

use typesize::TypeSize;
use nohash_hasher::NoHashHasher;
use log::{info, error};

/// INSTRUMENTATION: times the eviction loop fell back to evicting a random
/// object because the policy stack had no candidate. That path drops the
/// object from the map WITHOUT removing it from the stack.
pub static ERASE_FALLBACK: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);



use kwik::{
	fmt,
	math::set::Multiset,
};

use crate::{
	status::{AtomicStatus, Status},
	object::{
		Object,
		ObjectSize,
		overhead::OverheadManager,
	},
	worker::{
		WorkerEvent,
		WorkerFanout,
		WorkerHandles,
	},
};

pub use crate::{
	error::CacheError,
	policy::PaperPolicy,
};

#[cfg(all(feature = "key_value_pmem", feature = "enable_tiering_manager"))]
pub use crate::tiering::{TieringManager, TieringConfig, TieringStats};

pub type CacheSize = u64;
pub type AtomicCacheSize = AtomicU64;

pub type HashedKey = u64;
pub type NoHasher = BuildHasherDefault<NoHashHasher<HashedKey>>;

#[cfg(feature = "key_value_pmem")]
pub type BufferPMEM = Box<[u8], Hybrid>;


// Both tiers are allocated by `numa_alloc` (src/numa_alloc.rs): node-0-bound
// jemalloc arenas back the fast tier and the process allocator, node-1-bound
// arenas back `Hybrid`/`TieredBuffer::Slow`. This replaced a jemalloc pool per
// node, which held ~1.75x the memory in use and would not return it; see the
// numa_alloc module doc for the placement guarantee and its failure modes.

/// Samples jemalloc's internal accounting, for diagnosing where resident
/// memory goes relative to what the cache thinks it holds.
///
/// Returns `None` unless the process actually links jemalloc
/// (`numa_jemalloc`) and it was built with stats support.
///
/// The three ratios this exposes decompose resident memory into its causes,
/// which `used_size` alone cannot distinguish:
///
/// - `allocated` -- bytes the application asked for and still holds.
/// - `active` / `allocated` -- **external fragmentation**: pages in slabs that
///   hold at least one live allocation. A slab cannot be released while any
///   object in it is live, so under high churn this rises even though live
///   bytes are flat.
/// - `resident` / `active` -- pages the allocator has not yet returned to the
///   OS (dirty/muzzy, subject to decay).
/// - `retained` -- address space mapped but madvised away; costs no RSS.
///
/// `stats_print:true` via `_RJEM_MALLOC_CONF` cannot answer this: it runs from
/// jemalloc's atexit handler, by which point the cache has been dropped and
/// `allocated` has fallen to a few MB. This can be called at peak.
#[cfg(feature = "numa_jemalloc")]
pub fn jemalloc_stats() -> Option<String> {
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

	// jemalloc's stats are cached; advancing `epoch` refreshes them.
	unsafe {
		let mut epoch: u64 = 1;
		let mut epoch_len = std::mem::size_of::<u64>();

		mallctl(
			c"epoch".as_ptr(),
			(&raw mut epoch).cast(),
			&raw mut epoch_len,
			(&raw mut epoch).cast(),
			std::mem::size_of::<u64>(),
		);
	}

	fn read(name: &std::ffi::CStr) -> Option<usize> {
		let mut value: usize = 0;
		let mut len = std::mem::size_of::<usize>();

		let rc = unsafe {
			mallctl(
				name.as_ptr(),
				(&raw mut value).cast(),
				&raw mut len,
				std::ptr::null_mut(),
				0,
			)
		};

		(rc == 0).then_some(value)
	}

	let allocated = read(c"stats.allocated")?;
	let active = read(c"stats.active")?;
	let resident = read(c"stats.resident")?;
	let mapped = read(c"stats.mapped")?;
	let retained = read(c"stats.retained")?;

	let ratio = |numerator: usize, denominator: usize| {
		if denominator == 0 { 0.0 } else { numerator as f64 / denominator as f64 }
	};

	Some(format!(
		"JEMALLOC allocated={allocated} active={active} resident={resident} \
mapped={mapped} retained={retained} \
active/allocated={:.4} resident/active={:.4} resident/allocated={:.4}",
		ratio(active, allocated),
		ratio(resident, active),
		ratio(resident, allocated),
	))
}

/// Not linking jemalloc: nothing to sample.
#[cfg(not(feature = "numa_jemalloc"))]
pub fn jemalloc_stats() -> Option<String> {
	None
}

#[cfg(not(feature = "all_dram"))]
use std::alloc::{Layout, Allocator}; // Essential imports


//#[cfg(feature = "all_dram")]
#[cfg(not(feature = "segregated_value_arena"))]
pub type BufferDRAM = Box<[u8]>;

/// With `segregated_value_arena`, DRAM value buffers carry their own allocator
/// in the type -- exactly as `BufferPMEM = Box<[u8], Hybrid>` does -- so that
/// `dealloc` routes back to the same pool no matter which thread frees them.
/// That matters here: values are allocated on the client thread and freed on
/// the policy worker during eviction.
#[cfg(feature = "segregated_value_arena")]
pub type BufferDRAM = Box<[u8], numa_alloc::FastValues>;


/// Initial capacity (in entries) for the hashbrown-backed object map used
/// by `hashbrown_dram`, `global_hashtable_pmem`, and any hybrid-cache
/// feature combined with `hashbrown_dram` (see `new_hybrid_object_map`).
/// Sized to hold every object across this project's real benchmark traces
/// (`/home/griff/final_traces/*.bin`, distinct GET-driven keys measured at
/// ~1.06M-1.09M per trace) without ever growing/rehashing mid-benchmark.
#[cfg(any(feature = "global_hashtable_pmem", feature = "hashbrown_dram"))]
const HASHBROWN_INITIAL_CAPACITY: usize = 1_500_000;

#[cfg(all(not(feature = "global_hashtable_pmem"), not(feature = "hashbrown_dram")))]
pub type ObjectMapRef<K, V> = Arc<DashMap<HashedKey, Object<K, V>, NoHasher>>;

#[cfg(feature = "global_hashtable_pmem")]
pub type ObjectMapRef<K, V> = Arc<RwLock<HashMap<HashedKey, Object<K, V>, BuildHasherDefault<NoHashHasher<HashedKey>>, Hybrid>>>;

// Hashbrown HashMap in DRAM (for performance comparison with global_hashtable_pmem)
#[cfg(feature = "hashbrown_dram")]
pub type ObjectMapRef<K, V> = Arc<RwLock<HashMap<HashedKey, Object<K, V>, BuildHasherDefault<NoHashHasher<HashedKey>>>>>;


pub type StatusRef = Arc<AtomicStatus>;
pub type OverheadManagerRef = Arc<OverheadManager>;


pub struct PaperCache<K, V, S = RandomState> {
	objects: ObjectMapRef<K, V>,
	status: StatusRef,

	/// Routes each `WorkerEvent` to the background workers that consume it,
	/// inline on the calling thread -- see `WorkerFanout` for why this is not
	/// a thread of its own.
	workers: Arc<WorkerFanout>,
	/// Join handles for every background thread spawned on this cache's
	/// behalf (`PolicyWorker`/`TtlWorker`/`TieringWorker` -- `PolicyWorker`'s
	/// own `TraceWorker` child, when it has one, is joined internally by
	/// `PolicyWorker` itself, see its `Shutdown` handling, so it never
	/// appears here). `Drop` sends `WorkerEvent::Shutdown` through `workers`
	/// and then joins these, so that by the time a `PaperCache` has finished
	/// dropping, none of its background threads are still running --
	/// closing the real race this fixes: before this existed, no worker
	/// thread was ever joined at all, so a `PaperCache` being dropped (or a
	/// process exiting without explicitly dropping one) could leave a
	/// `PolicyWorker` thread genuinely still executing, mid-allocation,
	/// concurrently with the global allocator's own process-exit teardown
	/// -- confirmed directly via a real SIGSEGV inside a jemalloc pool's
	/// allocations, racing that pool's own teardown.
	worker_handles: WorkerHandles,
	overhead_manager: OverheadManagerRef,

	#[cfg(all(feature = "key_value_pmem", feature = "enable_tiering_manager"))]
	tiering_manager: Arc<TieringManager<K, V>>,

	hasher: S,
}

impl<K, V, S> Drop for PaperCache<K, V, S> {
	fn drop(&mut self) {
		// Best-effort: if a worker thread already exited on its own for some
		// other reason, its channel is already disconnected; `send` returning
		// `Err` here just means there's nothing left to signal, not a bug.
		if GI_N.load(std::sync::atomic::Ordering::Relaxed) > 0 {
			use std::sync::atomic::Ordering::Relaxed;
			let n = GI_N.load(Relaxed).max(1);
			eprintln!(
				"GIPROF n={} hash={} lookup={} copy={} bcast={} (avg ns/hit; Instant overhead ~20-40ns/step, equal across configs)",
				n,
				GI_HASH.load(Relaxed) / n,
				GI_LOOKUP.load(Relaxed) / n,
				GI_COPY.load(Relaxed) / n,
				GI_BCAST.load(Relaxed) / n,
			);
			if let Ok(v) = GI_SAMPLES.lock() {
				let pct = |xs: &mut Vec<u64>, f: f64| -> u64 {
					if xs.is_empty() {
						return 0;
					}
					xs.sort_unstable();
					xs[(((xs.len() - 1) as f64) * f).round() as usize]
				};
				// (label, payload cap, tier filter: 2 = any, 1 = fast only, 0 = slow only)
				for (label, keep, tier) in [
					("ALL", usize::MAX, 2u64),
					("SMALL<=256B", 256, 2),
					("SMALL-FAST", 256, 1),
					("SMALL-SLOW", 256, 0),
				] {
					let sel: Vec<&[u64; 6]> = v
						.iter()
						.filter(|s| (s[0] as usize) <= keep && (tier == 2 || s[5] == tier))
						.collect();
					if sel.is_empty() {
						continue;
					}
					let mut cols: [Vec<u64>; 5] = Default::default();
					for s in &sel {
						for k in 0..4 {
							cols[k].push(s[k + 1]);
						}
						// Per-op total: the quantity whose median the benchmark reports.
						cols[4].push(s[1] + s[2] + s[3] + s[4]);
					}
					let mut line = format!("GIPCT {} n={}", label, sel.len());
					for (name, col) in ["hash", "lookup", "copy", "bcast", "TOTAL"].iter().zip(cols.iter_mut()) {
						line += &format!(
							" {}[p25={} p50={} p75={} p90={}]",
							name,
							pct(col, 0.25),
							pct(col, 0.50),
							pct(col, 0.75),
							pct(col, 0.90),
						);
					}
					eprintln!("{}", line);
				}
				// Joint stall structure on the median population. Equal step
				// MARGINALS with unequal TOTAL medians means the difference is in
				// co-occurrence: scattered stalls push the median op over a stall;
				// concentrated stalls leave the median op clean.
				let small: Vec<&[u64; 6]> = v.iter().filter(|s| (s[0] as usize) <= 256).collect();
				if !small.is_empty() {
					let n = small.len() as f64;
					let p = |c: usize| c as f64 / n;
					let ls = small.iter().filter(|s| s[2] > 250).count();
					let cs = small.iter().filter(|s| s[3] > 150).count();
					let bs = small.iter().filter(|s| s[4] > 150).count();
					let lc = small.iter().filter(|s| s[2] > 250 && s[3] > 150).count();
					let any = small.iter().filter(|s| s[2] > 250 || s[3] > 150 || s[4] > 150).count();
					eprintln!(
						"GICORR SMALL n={} P(lookup>250)={:.3} P(copy>150)={:.3} P(bcast>150)={:.3} P(lookup&copy)={:.3} P(any)={:.3}",
						small.len(), p(ls), p(cs), p(bs), p(lc), p(any),
					);
				}
			}
		}

		let _ = self.workers.send(WorkerEvent::Shutdown);

		for handle in self.worker_handles.drain(..) {
			// A worker thread's own `Err`/panic is already logged from
			// inside `run()` (or by the default panic hook); nothing
			// further to do with the join result here beyond waiting for
			// it, which is this loop's entire purpose.
			let _ = handle.join();
		}
	}
}




//////////////////////////////////////////////////////////
/// 
/// 

// ---------------------------------------------------------------------
// Shape A: DashMap-backed object map. Covers `all_dram` (V = BufferDRAM)
// and `key_value_pmem` without `global_hashtable_pmem` (V = BufferPMEM) --
// see `ObjectMapRef`'s DashMap arm above. One generic-over-`V: ValueBuffer`
// block replaces what used to be two nearly-identical impl blocks (one per
// concrete V); the value-buffer axis and the tiering-manager machinery
// below (V-agnostic: `TieringManager<K, V>` is itself generic) are the only
// things that used to force separate blocks.
//
// Excludes `hashbrown_dram` (in addition to `global_hashtable_pmem`) to
// stay disjoint from Shape B below, mirroring `ObjectMapRef`'s own DashMap
// arm gate exactly -- without this, `hashbrown_dram` combined with
// `all_dram`/`key_value_pmem` would compile both this block and Shape B
// for the same `V: ValueBuffer`, a duplicate-inherent-impl error.
// ---------------------------------------------------------------------
#[cfg(any(all(feature = "all_dram", not(feature = "hashbrown_dram")), all(feature = "key_value_pmem", not(any(feature = "global_hashtable_pmem", feature = "hashbrown_dram")))))]
impl<K, V, S> PaperCache<K, V, S>
where
	K: 'static + Eq + Hash + TypeSize + Clone + Send + Sync,
	V: ValueBuffer,
	S: Default + Clone + BuildHasher,
{
	/// Creates an empty `PaperCache` with maximum size `max_size` and
	/// eviction policy `policy`. If the maximum size is zero, a
	/// [`CacheError`] will be returned.
	///
	/// # Examples
	///
	/// ```
	/// use paper_cache::{BufferDRAM, PaperCache, PaperPolicy};
	///
	/// let cache = PaperCache::<u32, BufferDRAM>::new(
	///     1000,
	///     &[PaperPolicy::Lfu],
	///     PaperPolicy::Lfu,
	/// );
	///
	/// assert!(cache.is_ok());
	///
	/// // Supplying a maximum size of zero will return a `CacheError`.
	/// let cache = PaperCache::<u32, BufferDRAM>::new(
	///     0,
	///     &[PaperPolicy::Lfu],
	///     PaperPolicy::Lfu,
	/// );
	///
	/// assert!(cache.is_err());
	///
	/// // Supplying duplicate policies will return a `CacheError`.
	/// let cache = PaperCache::<u32, BufferDRAM>::new(
	///     1000,
	///     &[PaperPolicy::Lfu, PaperPolicy::Lru, PaperPolicy::Lfu],
	///     PaperPolicy::Lfu,
	/// );
	///
	/// assert!(cache.is_err());
	///
	/// // Supplying a non-configured policy will return a `CacheError`.
	/// let cache = PaperCache::<u32, BufferDRAM>::new(
	///     1000,
	///     &[PaperPolicy::Lfu],
	///     PaperPolicy::Lru,
	/// );
	///
	/// assert!(cache.is_err());
	/// ```
	pub fn new(
		max_size: CacheSize,
		policies: &[PaperPolicy],
		policy: PaperPolicy,
	) -> Result<Self, CacheError> {
		Self::with_hasher(
			max_size,
			policies,
			policy,
			Default::default(),
		)
	}

	/// Creates an empty `PaperCache` with the supplied hasher.
	///
	/// # Examples
	///
	/// ```
	/// use std::hash::RandomState;
	/// use paper_cache::{BufferDRAM, PaperCache, PaperPolicy};
	///
	/// let cache = PaperCache::<u32, BufferDRAM>::with_hasher(
	///     1000,
	///     &[PaperPolicy::Lfu],
	///     PaperPolicy::Lfu,
	///     RandomState::default(),
	/// );
	///
	/// assert!(cache.is_ok());
	/// ```
	pub fn with_hasher(
		max_size: CacheSize,
		policies: &[PaperPolicy],
		policy: PaperPolicy,
		hasher: S,
	) -> Result<Self, CacheError> {
		if max_size == 0 {
			return Err(CacheError::ZeroCacheSize);
		}

		if policies.is_empty() {
			return Err(CacheError::EmptyPolicies);
		}

		if policies.contains(&PaperPolicy::Auto) {
			return Err(CacheError::ConfiguredAutoPolicy);
		}

		if policies.iter().is_multiset() {
			return Err(CacheError::DuplicatePolicies);
		}

		if !policy.is_auto() && !policies.contains(&policy) {
			return Err(CacheError::UnconfiguredPolicy);
		}

		// Every CONFIGURED policy is checked, not just the active one:
		// `PaperPolicy::Auto` can promote any of them later, and the runtime
		// `policy` setter only accepts policies already on this list -- so
		// validating the list here is what makes that setter safe by
		// construction.
		if policies
			.iter()
			.any(|configured| s_three_fifo_starves_main(*configured, max_size))
			|| s_three_fifo_starves_main(policy, max_size)
		{
			return Err(CacheError::InvalidPolicy);
		}

		let objects = Arc::new(DashMap::with_hasher(NoHasher::default()));
		let status = Arc::new(AtomicStatus::new(max_size, policies, policy)?);
		let overhead_manager = Arc::new(OverheadManager::new(&status));

		#[cfg(all(feature = "key_value_pmem", feature = "enable_tiering_manager"))]
		let tiering_manager = {
			// Create tiering manager with default DRAM threshold at 20% of max_size
			let mut tiering_config = tiering::TieringConfig::default();
			tiering_config.dram_threshold = (max_size as f64 * 0.2) as u64;
			Arc::new(TieringManager::new(tiering_config))
		};

		#[cfg(all(feature = "key_value_pmem", feature = "enable_tiering_manager"))]
		let (worker_fanout, worker_handles) = WorkerFanout::new(
			&objects,
			&status,
			&overhead_manager,
			&tiering_manager,
		)?;

		#[cfg(not(all(feature = "key_value_pmem", feature = "enable_tiering_manager")))]
		let (worker_fanout, worker_handles) = WorkerFanout::new(
			&objects,
			&status,
			&overhead_manager,
		)?;

		let cache = PaperCache {
			objects,
			status,

			workers: Arc::new(worker_fanout),
			worker_handles,
			overhead_manager,

			#[cfg(all(feature = "key_value_pmem", feature = "enable_tiering_manager"))]
			tiering_manager,

			hasher,
		};

		Ok(cache)
	}

	/// Returns the current cache version.
	///
	/// # Examples
	/// ```
	/// use paper_cache::{BufferDRAM, PaperCache, PaperPolicy};
	///
	/// let mut cache = PaperCache::<u32, BufferDRAM>::new(
	///     1000,
	///     &[PaperPolicy::Lfu],
	///     PaperPolicy::Lfu
	/// ).unwrap();
	///
	/// assert_eq!(cache.version(), env!("CARGO_PKG_VERSION"));
	/// ```
	#[must_use]
	pub fn version(&self) -> String {
		env!("CARGO_PKG_VERSION").to_owned()
	}

	/// Returns the current statistics.
	///
	/// # Examples
	/// ```
	/// use paper_cache::{BufferDRAM, PaperCache, PaperPolicy};
	///
	/// let mut cache = PaperCache::<u32, BufferDRAM>::new(
	///     1000,
	///     &[PaperPolicy::Lfu],
	///     PaperPolicy::Lfu,
	/// ).unwrap();
	///
	/// cache.set(0, &[0], None);
	///
	/// let status = cache.status().unwrap();
	/// assert!(status.used_size() > 0);
	/// ```
	pub fn status(&self) -> Result<Status, CacheError> {
		self.status.try_to_status()
	}

	/// Gets the value associated with the supplied key.
	/// If the key was not found in the cache, returns a [`CacheError`].
	///
	/// # Examples
	/// ```
	/// use paper_cache::{BufferDRAM, PaperCache, PaperPolicy};
	///
	/// let mut cache = PaperCache::<u32, BufferDRAM>::new(
	///     1000,
	///     &[PaperPolicy::Lfu],
	///     PaperPolicy::Lfu,
	/// ).unwrap();
	///
	/// cache.set(0, &[0], None);
	///
	/// // Getting a key which exists in the cache will return the associated value.
	/// assert!(cache.get(&0).is_ok());
	/// // Getting a key which does not exist in the cache will return a CacheError.
	/// assert!(cache.get(&1).is_err());
	/// ```
	pub fn get(&self, key: &K) -> Result<Vec<u8>, CacheError> {
		let hashed_key = self.hash_key(key);

		// Check the DRAM tier first (only ever wired up together with the
		// `key_value_pmem` copy-based tiering manager -- see `src/tiering/`).
		#[cfg(all(feature = "key_value_pmem", feature = "enable_tiering_manager", not(feature = "hashtable_tiering")))]
		if let Some(dram_object_ref) = self.tiering_manager.get_from_dram(&hashed_key) {
			if !dram_object_ref.is_expired() && dram_object_ref.key_matches(key) {
				self.status.incr_hits();
				self.broadcast(WorkerEvent::Get(hashed_key, true))?;
				let arc_val = dram_object_ref.data();
				return Ok(arc_val.as_ref().to_vec());
			}
		}

		#[cfg(all(feature = "key_value_pmem", feature = "enable_tiering_manager", feature = "hashtable_tiering"))]
		if let Some(dram_object_ref) = self.tiering_manager.get_from_dram(&hashed_key) {
			if !dram_object_ref.is_expired() && dram_object_ref.key_matches(key) {
				self.status.incr_hits();
				self.broadcast(WorkerEvent::Get(hashed_key, true))?;
				// Use data_as_bytes to handle both PhysicalCopy and CxlReference
				return Ok(dram_object_ref.data_as_bytes());
			}
		}

		// Guard released before the copy -- see the `TieredBuffer` `get()`
		// below for the full rationale. `Object::data()` is only an `Arc`
		// refcount bump, and the `Arc` keeps the bytes alive on its own, so
		// the shard lock is not needed for the (potentially multi-KB,
		// potentially PMEM-backed) copy itself.
		let maybe_data = match self.objects.get_ref(&hashed_key) {
			Some(object) if object.key_matches(key) && !object.is_expired() => Some(object.data()),
			_ => None,
		};

		let result = match maybe_data {
			Some(arc_val) => {
				self.status.incr_hits();
				Ok(AsRef::<[u8]>::as_ref(&*arc_val).to_vec())
			},

			None => {
				self.status.incr_misses();
				Err(CacheError::KeyNotFound)
			},
		};

		self.broadcast(WorkerEvent::Get(hashed_key, result.is_ok()))?;

		result
	}

	/// Diagnostic twin of [`Self::get`] that copies a hit into a caller-owned
	/// buffer instead of allocating a fresh `Vec` per call.
	///
	/// `get()` fuses two independent costs: locating and reading the value --
	/// which is what a tiering design changes -- and allocating the buffer to
	/// return it in, which is what the allocator configuration changes. Measured
	/// on Twitter cluster13 (2026-08-28) the second term dominated the first at
	/// the median, because the median value is 123 B while the mean is 4.9 KB.
	/// Comparing two cache designs through `get()` therefore compares their
	/// allocator behaviour as much as their cache behaviour; this method exists
	/// to measure them apart. See the `segregated_value_arena` feature.
	#[cfg(not(feature = "enable_tiering_manager"))]
	pub fn get_into(&self, key: &K, out: &mut Vec<u8>) -> Result<(), CacheError> {
		// Sampled step profiler -- see the GI_* statics at the bottom of this
		// file. One call in 64; hits only, matching what GET latency measures.
		let prof = gi_prof_enabled()
			&& GI_TICK.with(|c| {
				let t = c.get();
				c.set(t.wrapping_add(1));
				t & 63 == 0
			});
		let t0 = if prof { Some(std::time::Instant::now()) } else { None };

		let hashed_key = self.hash_key(key);
		let t1 = if prof { Some(std::time::Instant::now()) } else { None };

		let maybe_data = match self.objects.get_ref(&hashed_key) {
			Some(object) if object.key_matches(key) && !object.is_expired() => Some(object.data()),
			_ => None,
		};
		let t2 = if prof { Some(std::time::Instant::now()) } else { None };

		// Which tier served this hit (1 = fast/DRAM; the all-DRAM shape never
		// reassigns it). Read only by the sampled profiler below.
		#[allow(unused_mut)]
		let mut gi_fast: u64 = 1;

		let result = match maybe_data {
			Some(arc_val) => {
				self.status.incr_hits();
				out.clear();
				out.extend_from_slice(AsRef::<[u8]>::as_ref(&*arc_val));
				Ok(())
			},

			None => {
				self.status.incr_misses();
				Err(CacheError::KeyNotFound)
			},
		};
		let t3 = if prof { Some(std::time::Instant::now()) } else { None };

		self.broadcast(WorkerEvent::Get(hashed_key, result.is_ok()))?;

		if let (Some(t0), Some(t1), Some(t2), Some(t3), true) = (t0, t1, t2, t3, result.is_ok()) {
			let t4 = std::time::Instant::now();
			use std::sync::atomic::Ordering::Relaxed;
			let (h, l, c, b) = (
				(t1 - t0).as_nanos() as u64,
				(t2 - t1).as_nanos() as u64,
				(t3 - t2).as_nanos() as u64,
				(t4 - t3).as_nanos() as u64,
			);
			GI_N.fetch_add(1, Relaxed);
			GI_HASH.fetch_add(h, Relaxed);
			GI_LOOKUP.fetch_add(l, Relaxed);
			GI_COPY.fetch_add(c, Relaxed);
			GI_BCAST.fetch_add(b, Relaxed);
			// Off the timed steps (after t4); the lock is uncontended at 1-in-64.
			if let Ok(mut v) = GI_SAMPLES.lock() {
				if v.capacity() == 0 {
					v.reserve_exact(1 << 20);
				}
				if v.len() < (1 << 20) {
					v.push([out.len() as u64, h, l, c, b, gi_fast]);
				}
			}
		}

		result
	}

	/// Sets the supplied key and value in the cache.
	/// Returns a [`CacheError`] if the value size is zero or larger than
	/// the cache's maximum size.
	///
	/// If the key already exists in the cache, the associated value is updated
	/// to the supplied value.
	///
	/// # Examples
	/// ```
	/// use paper_cache::{BufferDRAM, PaperCache, PaperPolicy};
	///
	/// let mut cache = PaperCache::<u32, BufferDRAM>::new(
	///     1000,
	///     &[PaperPolicy::Lfu],
	///     PaperPolicy::Lfu,
	/// ).unwrap();
	///
	/// assert!(cache.set(0, &[0], None).is_ok());
	/// ```
	pub fn set(&self, key: K, value: &[u8], ttl: Option<u32>) -> Result<(), CacheError> {
		let hashed_key = self.hash_key(&key);

		let val_buf: V = V::from_bytes(value);
		let object = Object::new(key, val_buf, ttl);
		let base_size = self.overhead_manager.base_size(&object);
		let dram_resident = self.overhead_manager.dram_resident_size(&object);
		let expiry = object.expiry();

		if base_size == 0 {
			return Err(CacheError::ZeroValueSize);
		}

		if self.status.exceeds_max_size(base_size) {
			return Err(CacheError::ExceedingValueSize);
		}

		self.status.incr_sets();

		let old_object_info = self.objects
			.insert(hashed_key, object)
			.map(|old_object| {
				let base_size = self.overhead_manager.base_size(&old_object);
				let expiry = old_object.expiry();

				(base_size, expiry)
			});

		let base_size_delta = if let Some((old_object_size, _)) = old_object_info {
			base_size as i64 - old_object_size as i64
		} else {
			// the object is new, so increase the number of objects count
			self.status.incr_num_objects();
			base_size as i64
		};

		self.status.update_base_used_size(base_size_delta);
		self.broadcast(WorkerEvent::Set(
			hashed_key,
			base_size,
			dram_resident,
			expiry,
			old_object_info,
		))?;

		Ok(())
	}

	/// Deletes the object associated with the supplied key in the cache.
	/// Returns a [`CacheError`] if the key was not found in the cache.
	///
	/// # Examples
	/// ```
	/// use paper_cache::{BufferDRAM, PaperCache, PaperPolicy};
	///
	/// let mut cache = PaperCache::<u32, BufferDRAM>::new(
	///     1000,
	///     &[PaperPolicy::Lfu],
	///     PaperPolicy::Lfu,
	/// ).unwrap();
	///
	/// cache.set(0, &[0], None);
	/// assert!(cache.del(&0).is_ok());
	///
	/// // Deleting a key which does not exist in the cache will return a CacheError.
	/// assert!(cache.del(&1).is_err());
	/// ```
	pub fn del(&self, key: &K) -> Result<(), CacheError> {
		let hashed_key = self.hash_key(key);

		let (removed_hashed_key, object) = erase(
			&self.objects,
			&self.status,
			&self.overhead_manager,
			Some(EraseKey::Original(key, hashed_key)),
		)?;

		self.status.incr_dels();
		self.broadcast(WorkerEvent::Del(removed_hashed_key, object.expiry()))?;

		Ok(())
	}

	/// Checks if an object with the supplied key exists in the cache without
	/// altering any of the cache's internal queues.
	///
	/// # Examples
	/// ```
	/// use paper_cache::{BufferDRAM, PaperCache, PaperPolicy};
	///
	/// let mut cache = PaperCache::<u32, BufferDRAM>::new(
	///     1000,
	///     &[PaperPolicy::Lfu],
	///     PaperPolicy::Lfu,
	/// ).unwrap();
	///
	/// cache.set(0, &[0], None);
	///
	/// assert!(cache.has(&0));
	/// assert!(!cache.has(&1));
	/// ```
	pub fn has(&self, key: &K) -> bool {
		let hashed_key = self.hash_key(key);

		self.objects
			.get_ref(&hashed_key)
			.is_some_and(|object| object.key_matches(key) && !object.is_expired())
	}

	/// Gets (peeks) the value associated with the supplied key without altering
	/// any of the cache's internal queues.
	/// If the key was not found in the cache, returns a [`CacheError`].
	///
	/// # Examples
	/// ```
	/// use paper_cache::{BufferDRAM, PaperCache, PaperPolicy};
	///
	/// let mut cache = PaperCache::<u32, BufferDRAM>::new(
	///     1000,
	///     &[PaperPolicy::Lfu],
	///     PaperPolicy::Lfu,
	/// ).unwrap();
	///
	/// cache.set(0, &[0], None);
	/// cache.set(1, &[0], None);
	///
	/// // Peeking a key which exists in the cache will return the associated value.
	/// assert!(cache.peek(&0).is_ok());
	/// // Peeking a key which does not exist in the cache will return a CacheError.
	/// assert!(cache.peek(&2).is_err());
	///
	/// cache.set(2, &[0], None);
	///
	/// // Peeking a key will not alter the eviction order of the objects.
	/// assert!(cache.peek(&1).is_ok());
	/// assert!(cache.peek(&2).is_ok());
	/// ```
	pub fn peek(&self, key: &K) -> Result<Arc<V>, CacheError> {
		let hashed_key = self.hash_key(key);

		match self.objects.get_ref(&hashed_key) {
			Some(object) if object.key_matches(key) && !object.is_expired() =>
				Ok(object.data()),

			_ => Err(CacheError::KeyNotFound),
		}
	}

	/// Sets the TTL associated with the supplied key.
	/// If the key was not found in the cache, returns a [`CacheError`].
	///
	/// # Examples
	/// ```
	/// use paper_cache::{BufferDRAM, PaperCache, PaperPolicy};
	///
	/// let mut cache = PaperCache::<u32, BufferDRAM>::new(
	///     1000,
	///     &[PaperPolicy::Lfu],
	///     PaperPolicy::Lfu,
	/// ).unwrap();
	///
	/// cache.set(0, &[0], None); // value will not expire
	/// cache.ttl(&0, Some(5)); // value will expire in 5 seconds
	/// ```
	pub fn ttl(&self, key: &K, ttl: Option<u32>) -> Result<(), CacheError> {
		let hashed_key = self.hash_key(key);

		let mut object = match self.objects.get_mut_ref(&hashed_key) {
			Some(object) if object.key_matches(key) && !object.is_expired() => object,
			_ => return Err(CacheError::KeyNotFound),
		};

		let old_expiry = object.expiry();
		let old_base_size = self.overhead_manager.base_size(&object);

		object.expires(ttl);

		let new_expiry = object.expiry();
		let new_base_size = self.overhead_manager.base_size(&object);

		self.status.update_base_used_size(new_base_size as i64 - old_base_size as i64);
		self.broadcast(WorkerEvent::Ttl(hashed_key, old_expiry, new_expiry))?;

		Ok(())
	}

	/// Gets the size of the value associated with the supplied key in bytes.
	/// If the key was not found in the cache, returns a [`CacheError`].
	///
	/// # Examples
	/// ```
	/// use paper_cache::{BufferDRAM, PaperCache, PaperPolicy};
	///
	/// let mut cache = PaperCache::<u32, BufferDRAM>::new(
	///     1000,
	///     &[PaperPolicy::Lfu],
	///     PaperPolicy::Lfu,
	/// ).unwrap();
	///
	/// cache.set(0, &[0], None);
	///
	/// // Sizing a key which exists in the cache will return the size of the associated value.
	/// assert!(cache.size(&0).is_ok());
	/// // Sizing a key which does not exist in the cache will return a CacheError.
	/// assert!(cache.size(&1).is_err());
	/// ```
	pub fn size(&self, key: &K) -> Result<ObjectSize, CacheError> {
		let hashed_key = self.hash_key(key);

		match self.objects.get_ref(&hashed_key) {
			Some(object) if object.key_matches(key) && !object.is_expired() =>
				Ok(self.overhead_manager.total_size(&object)),

			_ => Err(CacheError::KeyNotFound),
		}
	}

	/// Deletes all objects in the cache and sets the cache's used size to zero.
	/// Returns a [`CacheError`] if the objects could not be wiped.
	///
	/// # Examples
	/// ```
	/// use paper_cache::{BufferDRAM, PaperCache, PaperPolicy};
	///
	/// let mut cache = PaperCache::<u32, BufferDRAM>::new(
	///     1000,
	///     &[PaperPolicy::Lfu],
	///     PaperPolicy::Lfu,
	/// ).unwrap();
	///
	/// cache.wipe();
	/// ```
	pub fn wipe(&self) -> Result<(), CacheError> {
		info!("Wiping cache");

		self.objects.clear();
		self.status.clear();

		self.broadcast(WorkerEvent::Wipe)?;

		Ok(())
	}

	/// Resizes the cache to the supplied maximum size.
	/// If the supplied size is zero, returns a [`CacheError`].
	///
	/// # Examples
	/// ```
	/// use paper_cache::{BufferDRAM, PaperCache, PaperPolicy};
	///
	/// let mut cache = PaperCache::<u32, BufferDRAM>::new(
	///     1000,
	///     &[PaperPolicy::Lfu],
	///     PaperPolicy::Lfu,
	/// ).unwrap();
	///
	/// assert!(cache.resize(1).is_ok());
	///
	/// // Resizing to a size of zero will return a CacheError.
	/// assert!(cache.resize(0).is_err());
	/// ```
	pub fn resize(&self, max_size: CacheSize) -> Result<(), CacheError> {
		if max_size == 0 {
			return Err(CacheError::ZeroCacheSize);
		}

		// `Stack::resize` recomputes the main budget against the NEW size, so
		// a resize can starve a queue that was fine at construction.
		if self
			.status
			.policies()
			.iter()
			.any(|configured| s_three_fifo_starves_main(*configured, max_size))
		{
			return Err(CacheError::InvalidPolicy);
		}

		let current_max_size = self.status.max_size();

		if max_size == current_max_size {
			return Ok(());
		}

		info!(
			"Resizing cache from {} to {}",
			fmt::memory(current_max_size, Some(2)),
			fmt::memory(max_size, Some(2)),
		);

		self.status.set_max_size(max_size);
		self.broadcast(WorkerEvent::Resize(max_size))?;

		Ok(())
	}

	/// Sets the eviction policy of the cache to the supplied policy.
	///
	/// # Examples
	/// ```
	/// use paper_cache::{BufferDRAM, PaperCache, PaperPolicy};
	///
	/// let mut cache = PaperCache::<u32, BufferDRAM>::new(
	///     1000,
	///     &[PaperPolicy::Lfu],
	///     PaperPolicy::Lfu,
	/// ).unwrap();
	///
	/// assert!(cache.policy(PaperPolicy::Lfu).is_ok());
	/// assert!(cache.policy(PaperPolicy::Lru).is_err());
	/// ```
	pub fn policy(&self, policy: PaperPolicy) -> Result<(), CacheError> {
		if !policy.is_auto() && !self.status.policies().contains(&policy) {
			return Err(CacheError::UnconfiguredPolicy);
		}

		self.status.set_policy(policy)?;
		self.broadcast(WorkerEvent::Policy(policy))?;

		Ok(())
	}

	#[cfg(all(feature = "key_value_pmem", feature = "enable_tiering_manager"))]
	/// Gets tiering statistics including objects in DRAM, promotions, and demotions.
	pub fn tiering_stats(&self) -> tiering::TieringStats {
		self.tiering_manager.stats()
	}

	#[cfg(all(feature = "key_value_pmem", feature = "enable_tiering_manager"))]
	/// Sets the DRAM tier threshold in bytes.
	pub fn set_dram_threshold(&self, threshold: u64) {
		self.tiering_manager.set_dram_threshold(threshold);
	}

	#[cfg(all(feature = "key_value_pmem", feature = "enable_tiering_manager"))]
	/// Gets the current DRAM tier threshold in bytes.
	pub fn dram_threshold(&self) -> u64 {
		self.tiering_manager.dram_threshold()
	}

	#[cfg(all(feature = "key_value_pmem", feature = "enable_tiering_manager"))]
	/// Sets the hotness threshold for promotion to DRAM.
	pub fn set_hotness_threshold(&self, threshold: u64) {
		self.tiering_manager.set_hotness_threshold(threshold);
	}

	#[cfg(all(feature = "key_value_pmem", feature = "enable_tiering_manager"))]
	/// Gets the current hotness threshold.
	pub fn hotness_threshold(&self) -> u64 {
		self.tiering_manager.hotness_threshold()
	}

	fn broadcast(&self, event: WorkerEvent) -> Result<(), CacheError> {
		self.workers.send(event)
	}

	fn hash_key(&self, key: &K) -> HashedKey {
		self.hasher.hash_one(key)
	}
}

// ---------------------------------------------------------------------
// Shape B: `RwLock<HashMap<..., A>>`-backed object map, generic over the
// allocator `A`. Covers `global_hashtable_pmem` alone (V = BufferDRAM,
// A = Hybrid), `hashbrown_dram` (V = BufferDRAM, A = default/Global), and
// `key_value_pmem` + `global_hashtable_pmem` together (V = BufferPMEM,
// A = Hybrid) -- see `ObjectMapRef`'s two RwLock arms above. One
// generic-over-`V: ValueBuffer` block replaces what used to be three
// nearly-identical impl blocks.
//
// Unlike Shape A, this shape's tiering-manager support is limited to the
// plain `enable_tiering_manager` DRAM-side-cache check in `get()` -- there
// is no `tiering_stats`/`set_dram_threshold`/etc. accessors here, matching
// this shape's pre-merge behavior exactly (only Shape A ever had those).
// ---------------------------------------------------------------------
#[cfg(any(feature = "global_hashtable_pmem", feature = "hashbrown_dram"))]
impl<K, V, S> PaperCache<K, V, S>
where
	// `Send + Sync` because `WorkerFanout::new` hands the object map to worker
	// threads. Every other `PaperCache` impl carries these; this shape was
	// merged without them, so the two features that select it never built.
	K: 'static + Eq + Hash + TypeSize + Clone + Send + Sync,
	V: ValueBuffer,
	S: Default + Clone + BuildHasher,
{
	/// Creates an empty `PaperCache` with maximum size `max_size` and
	/// eviction policy `policy`. If the maximum size is zero, a
	/// [`CacheError`] will be returned.
	pub fn new(
		max_size: CacheSize,
		policies: &[PaperPolicy],
		policy: PaperPolicy,
	) -> Result<Self, CacheError> {
		Self::with_hasher(
			max_size,
			policies,
			policy,
			Default::default(),
		)
	}

	/// Creates an empty `PaperCache` with the supplied hasher.
	pub fn with_hasher(
		max_size: CacheSize,
		policies: &[PaperPolicy],
		policy: PaperPolicy,
		hasher: S,
	) -> Result<Self, CacheError> {
		if max_size == 0 {
			return Err(CacheError::ZeroCacheSize);
		}

		if policies.is_empty() {
			return Err(CacheError::EmptyPolicies);
		}

		if policies.contains(&PaperPolicy::Auto) {
			return Err(CacheError::ConfiguredAutoPolicy);
		}

		if policies.iter().is_multiset() {
			return Err(CacheError::DuplicatePolicies);
		}

		if !policy.is_auto() && !policies.contains(&policy) {
			return Err(CacheError::UnconfiguredPolicy);
		}

		// Every CONFIGURED policy is checked, not just the active one:
		// `PaperPolicy::Auto` can promote any of them later, and the runtime
		// `policy` setter only accepts policies already on this list -- so
		// validating the list here is what makes that setter safe by
		// construction.
		if policies
			.iter()
			.any(|configured| s_three_fifo_starves_main(*configured, max_size))
			|| s_three_fifo_starves_main(policy, max_size)
		{
			return Err(CacheError::InvalidPolicy);
		}

		// Global hashtable in PMEM (Hybrid allocator) when
		// `global_hashtable_pmem` is on; otherwise a plain-DRAM hashbrown
		// table (`hashbrown_dram`'s default allocator).
		#[cfg(feature = "global_hashtable_pmem")]
		let objects = Arc::new(RwLock::new(HashMap::with_capacity_and_hasher_in(
			HASHBROWN_INITIAL_CAPACITY,
			NoHasher::default(),
			Hybrid,
		)));

		#[cfg(not(feature = "global_hashtable_pmem"))]
		let objects = Arc::new(RwLock::new(HashMap::with_capacity_and_hasher(
			HASHBROWN_INITIAL_CAPACITY,
			NoHasher::default(),
		)));

		let status = Arc::new(AtomicStatus::new(max_size, policies, policy)?);
		let overhead_manager = Arc::new(OverheadManager::new(&status));

		#[cfg(all(feature = "key_value_pmem", feature = "enable_tiering_manager"))]
		let tiering_manager = {
			// Create tiering manager with default DRAM threshold at 20% of max_size
			let mut tiering_config = tiering::TieringConfig::default();
			tiering_config.dram_threshold = (max_size as f64 * 0.2) as u64;
			Arc::new(TieringManager::new(tiering_config))
		};

		#[cfg(all(feature = "key_value_pmem", feature = "enable_tiering_manager"))]
		let (worker_fanout, worker_handles) = WorkerFanout::new(
			&objects,
			&status,
			&overhead_manager,
			&tiering_manager,
		)?;

		#[cfg(not(all(feature = "key_value_pmem", feature = "enable_tiering_manager")))]
		let (worker_fanout, worker_handles) = WorkerFanout::new(
			&objects,
			&status,
			&overhead_manager,
		)?;

		let cache = PaperCache {
			objects,
			status,
			workers: Arc::new(worker_fanout),
			worker_handles,
			overhead_manager,

			#[cfg(all(feature = "key_value_pmem", feature = "enable_tiering_manager"))]
			tiering_manager,

			hasher,
		};

		Ok(cache)
	}

	#[must_use]
	pub fn version(&self) -> String {
		env!("CARGO_PKG_VERSION").to_owned()
	}

	pub fn status(&self) -> Result<Status, CacheError> {
		self.status.try_to_status()
	}

	/// Gets the value associated with the supplied key.
	/// If the key was not found in the cache, returns a [`CacheError`].
	pub fn get(&self, key: &K) -> Result<Vec<u8>, CacheError> {
		let hashed_key = self.hash_key(key);

		#[cfg(feature = "enable_tiering_manager")]
		if let Some(dram_object_ref) = self.tiering_manager.get_from_dram(&hashed_key) {
			if !dram_object_ref.is_expired() && dram_object_ref.key_matches(key) {
				self.status.incr_hits();
				self.broadcast(WorkerEvent::Get(hashed_key, true))?;
				let arc_val = dram_object_ref.data();
				return Ok(arc_val.as_ref().to_vec());
			}
		}

		// Guard released before the copy -- see the `TieredBuffer` `get()`
		// below for the full rationale. `Object::data()` is only an `Arc`
		// refcount bump, and the `Arc` keeps the bytes alive on its own, so
		// the shard lock is not needed for the (potentially multi-KB,
		// potentially PMEM-backed) copy itself.
		let maybe_data = match self.objects.get_ref(&hashed_key) {
			Some(object) if object.key_matches(key) && !object.is_expired() => Some(object.data()),
			_ => None,
		};

		let result = match maybe_data {
			Some(arc_val) => {
				self.status.incr_hits();
				Ok(AsRef::<[u8]>::as_ref(&*arc_val).to_vec())
			},

			None => {
				self.status.incr_misses();
				Err(CacheError::KeyNotFound)
			},
		};

		self.broadcast(WorkerEvent::Get(hashed_key, result.is_ok()))?;

		result
	}

	/// Sets the supplied key and value in the cache.
	/// Returns a [`CacheError`] if the value size is zero or larger than
	/// the cache's maximum size.
	pub fn set(&self, key: K, value: &[u8], ttl: Option<u32>) -> Result<(), CacheError> {
		let hashed_key = self.hash_key(&key);

		let val_buf: V = V::from_bytes(value);
		let object = Object::new(key, val_buf, ttl);

		let base_size = self.overhead_manager.base_size(&object);
		let dram_resident = self.overhead_manager.dram_resident_size(&object);
		let expiry = object.expiry();

		if base_size == 0 {
			return Err(CacheError::ZeroValueSize);
		}

		if self.status.exceeds_max_size(base_size) {
			return Err(CacheError::ExceedingValueSize);
		}

		self.status.incr_sets();

		let old_object_info = self.objects
			.insert(hashed_key, object)
			.map(|old_object| {
				let base_size = self.overhead_manager.base_size(&old_object);
				let expiry = old_object.expiry();
				(base_size, expiry)
			});

		let base_size_delta = if let Some((old_object_size, _)) = old_object_info {
			base_size as i64 - old_object_size as i64
		} else {
			self.status.incr_num_objects();
			base_size as i64
		};

		self.status.update_base_used_size(base_size_delta);
		self.broadcast(WorkerEvent::Set(
			hashed_key,
			base_size,
			dram_resident,
			expiry,
			old_object_info,
		))?;

		Ok(())
	}

	pub fn del(&self, key: &K) -> Result<(), CacheError> {
		let hashed_key = self.hash_key(key);

		let (removed_hashed_key, object) = erase(
			&self.objects,
			&self.status,
			&self.overhead_manager,
			Some(EraseKey::Original(key, hashed_key)),
		)?;

		self.status.incr_dels();
		self.broadcast(WorkerEvent::Del(removed_hashed_key, object.expiry()))?;

		Ok(())
	}

	pub fn has(&self, key: &K) -> bool {
		let hashed_key = self.hash_key(key);

		self.objects
			.get_ref(&hashed_key)
			.is_some_and(|object| object.key_matches(key) && !object.is_expired())
	}

	pub fn peek(&self, key: &K) -> Result<Arc<V>, CacheError> {
		let hashed_key = self.hash_key(key);

		match self.objects.get_ref(&hashed_key) {
			Some(object) if object.key_matches(key) && !object.is_expired() =>
				Ok(object.data()),
			_ => Err(CacheError::KeyNotFound),
		}
	}

	pub fn ttl(&self, key: &K, ttl: Option<u32>) -> Result<(), CacheError> {
		let hashed_key = self.hash_key(key);

		let mut object = match self.objects.get_mut_ref(&hashed_key) {
			Some(object) if object.key_matches(key) && !object.is_expired() => object,
			_ => return Err(CacheError::KeyNotFound),
		};

		let old_expiry = object.expiry();
		let old_base_size = self.overhead_manager.base_size(&object);

		object.expires(ttl);

		let new_expiry = object.expiry();
		let new_base_size = self.overhead_manager.base_size(&object);

		self.status.update_base_used_size(new_base_size as i64 - old_base_size as i64);
		self.broadcast(WorkerEvent::Ttl(hashed_key, old_expiry, new_expiry))?;

		Ok(())
	}

	pub fn size(&self, key: &K) -> Result<ObjectSize, CacheError> {
		let hashed_key = self.hash_key(key);

		match self.objects.get_ref(&hashed_key) {
			Some(object) if object.key_matches(key) && !object.is_expired() =>
				Ok(self.overhead_manager.total_size(&object)),
			_ => Err(CacheError::KeyNotFound),
		}
	}

	pub fn wipe(&self) -> Result<(), CacheError> {
		info!("Wiping cache");

		self.objects.clear();
		self.status.clear();

		self.broadcast(WorkerEvent::Wipe)?;

		Ok(())
	}

	pub fn resize(&self, max_size: CacheSize) -> Result<(), CacheError> {
		if max_size == 0 {
			return Err(CacheError::ZeroCacheSize);
		}

		// `Stack::resize` recomputes the main budget against the NEW size, so
		// a resize can starve a queue that was fine at construction.
		if self
			.status
			.policies()
			.iter()
			.any(|configured| s_three_fifo_starves_main(*configured, max_size))
		{
			return Err(CacheError::InvalidPolicy);
		}

		let current_max_size = self.status.max_size();

		if max_size == current_max_size {
			return Ok(());
		}

		info!(
			"Resizing cache from {} to {}",
			fmt::memory(current_max_size, Some(2)),
			fmt::memory(max_size, Some(2)),
		);

		self.status.set_max_size(max_size);
		self.broadcast(WorkerEvent::Resize(max_size))?;

		Ok(())
	}

	pub fn policy(&self, policy: PaperPolicy) -> Result<(), CacheError> {
		if !policy.is_auto() && !self.status.policies().contains(&policy) {
			return Err(CacheError::UnconfiguredPolicy);
		}

		self.status.set_policy(policy)?;
		self.broadcast(WorkerEvent::Policy(policy))?;

		Ok(())
	}

	fn broadcast(&self, event: WorkerEvent) -> Result<(), CacheError> {
		self.workers.send(event)
	}

	fn hash_key(&self, key: &K) -> HashedKey {
		self.hasher.hash_one(key)
	}
}



pub enum EraseKey<'a, K> {
	Original(&'a K, HashedKey),
	Hashed(HashedKey),
}


#[cfg(any(feature = "global_hashtable_pmem", feature = "hashbrown_dram"))]
pub fn erase<K, V>(
	objects: &ObjectMapRef<K, V>,
	status: &StatusRef,
	overhead_manager: &OverheadManagerRef,
	maybe_key: Option<EraseKey<K>>,
) -> Result<(HashedKey, Object<K, V>), CacheError>
where
	K: Eq + TypeSize,
	V: TypeSize,
{
	let hashed_key = match maybe_key {
		Some(EraseKey::Original(_, hashed_key)) => hashed_key,
		Some(EraseKey::Hashed(hashed_key)) => hashed_key,

		None => {
			// INSTRUMENTATION: this path removes an object from the MAP without
			// informing the eviction STACK, which is exactly the shape of the
			// observed map>stack divergence. Counted so the hypothesis is
			// testable rather than plausible.
			crate::ERASE_FALLBACK.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
			// the policy has run out of keys to evict (either it's a mini stack or
			// something went wrong during policy reconstruction) so we fall back
			// to evicting a random object

			//let Some(object) = objects.iter().next() else {
			//let Some(object) = objects.read().unwrap().iter().next() else {
			let mut objects_guard = objects.write().unwrap();
			let Some(object) = objects_guard.iter().next() else {
				error!("Object store is empty with non-zero used size");
				return Err(CacheError::Internal);
			};

			//object.key().to_owned()
			object.0.to_owned()
		},
	};

	// don't remove the object right away because if we have the original key,
	// we need to do a validation check that it matches the object's key in
	// case of a hash collision
	//let Entry::Occupied(entry) = objects.entry(hashed_key) else {
	let mut objects_lock = objects.write().unwrap();
	let Entry::Occupied(entry) = objects_lock.entry(hashed_key) else {
		return Err(CacheError::KeyNotFound);
	};

	//if let Some(EraseKey::Original(key, _)) = maybe_key && !entry.get().key_matches(key) {
	if let Some(EraseKey::Original(key, _)) = maybe_key && !entry.get().key_matches(key) {
		return Err(CacheError::KeyNotFound);
	};

	let object = entry.remove();
	let base_size = overhead_manager.base_size(&object) as i64;

	status.update_base_used_size(-base_size);
	status.decr_num_objects();

	match !object.is_expired() {
		true => Ok((hashed_key, object)),
		false => Err(CacheError::KeyNotFound),
	}
}











#[cfg(not(any(feature = "global_hashtable_pmem", feature = "hashbrown_dram")))]
pub fn erase<K, V>(
	objects: &ObjectMapRef<K, V>,
	status: &StatusRef,
	overhead_manager: &OverheadManagerRef,
	maybe_key: Option<EraseKey<K>>,
) -> Result<(HashedKey, Object<K, V>), CacheError>
where
	K: Eq + TypeSize,
	V: TypeSize,
{
	let hashed_key = match maybe_key {
		Some(EraseKey::Original(_, hashed_key)) => hashed_key,
		Some(EraseKey::Hashed(hashed_key)) => hashed_key,

		None => {
			// INSTRUMENTATION: this path removes an object from the MAP without
			// informing the eviction STACK, which is exactly the shape of the
			// observed map>stack divergence. Counted so the hypothesis is
			// testable rather than plausible.
			crate::ERASE_FALLBACK.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
			// the policy has run out of keys to evict (either it's a mini stack or
			// something went wrong during policy reconstruction) so we fall back
			// to evicting a random object

			let Some(object) = objects.iter().next() else {
				error!("Object store is empty with non-zero used size");
				return Err(CacheError::Internal);
			};

			object.key().to_owned()
		},
	};

	// don't remove the object right away because if we have the original key,
	// we need to do a validation check that it matches the object's key in
	// case of a hash collision
	let Entry::Occupied(entry) = objects.entry(hashed_key) else {
		return Err(CacheError::KeyNotFound);
	};

	if let Some(EraseKey::Original(key, _)) = maybe_key && !entry.get().key_matches(key) {
		return Err(CacheError::KeyNotFound);
	};

	let object = entry.remove();
	let base_size = overhead_manager.base_size(&object) as i64;

	status.update_base_used_size(-base_size);
	status.decr_num_objects();

	match !object.is_expired() {
		true => Ok((hashed_key, object)),
		false => Err(CacheError::KeyNotFound),
	}
}

unsafe impl<K, V, S> Send for PaperCache<K, V, S> {}
// SAFETY: `PaperCache` uses a `DashMap` (internally sharded, `Sync`)
// for the object store and `Arc`-wrapped atomics / `crossbeam_channel`
// senders for all shared state.  No unsynchronised mutable access
// is exposed, so sharing a `&PaperCache` across threads is safe.
unsafe impl<K, V, S> Sync for PaperCache<K, V, S> {}

/// Builds the object map every hybrid-cache design (`lru_hybrid_cache`/
/// `lfu_hybrid_cache`/`two_q_hybrid_cache`/`fifo_hybrid_cache`/
/// `lru_sized_hybrid_cache`) stores its objects in -- mirrors Shape B's
/// `with_hasher` (see above) rather than hardcoding `DashMap`, so
/// `hashbrown_dram` gets the same plain-DRAM `hashbrown::HashMap` object
/// table it already gives the non-hybrid storage combos, instead of always
/// silently using `DashMap` regardless of that feature. The return type
/// (`ObjectMapRef<K, V>`) is picked by the same cfg that already selects it
/// crate-wide -- this just has to build a matching value.
#[cfg(feature = "hybrid_cache_common")]
fn new_hybrid_object_map<K, V>() -> ObjectMapRef<K, V> {
	#[cfg(feature = "hashbrown_dram")]
	{
		Arc::new(RwLock::new(HashMap::with_capacity_and_hasher(
			HASHBROWN_INITIAL_CAPACITY,
			NoHasher::default(),
		)))
	}

	#[cfg(not(feature = "hashbrown_dram"))]
	{
		Arc::new(DashMap::with_hasher(NoHasher::default()))
	}
}

/// True when `policy` sizes its main queue from `1 - ratio` and that budget
/// truncates to zero at `max_size` -- the configuration that spins
/// `apply_evictions`, since `Stack::is_full` is `used >= max` and so an empty
/// zero-capacity queue reports itself full.
///
/// Covers the non-tiered `SThreeFifo` design only. The hybrid designs go
/// through `s3_fifo_queue_budgets`, which additionally has to tell the stacks
/// that size a main queue apart from the reprieve stacks that do not.
fn s_three_fifo_starves_main(policy: PaperPolicy, max_size: CacheSize) -> bool {
	let PaperPolicy::SThreeFifo(ratio) = policy else {
		return false;
	};

	((1.0 - ratio) * max_size as f64) as CacheSize == 0
}

/// For an s3-fifo design: its one-access ratio, and whether it also sizes a
/// main queue at `(1 - ratio) * max_size`. `None` for anything else.
///
/// Exists so `new_hybrid` can reject a config whose computed queue budget
/// rounds to zero. The second field is not cosmetic: the four reprieve stacks
/// size `one_access_capacity` and nothing else -- they derive no budget from
/// `1 - ratio` and never gate eviction on main fullness -- so a main budget
/// truncating to zero means nothing to them, and checking it would refuse a
/// config that works. The 2Q designs are absent for the same reason, one step
/// further: they derive no budget from `1 - k_in` at all.
#[cfg(feature = "hybrid_cache_common")]
/// Parameter ranges the per-design constructors used to enforce. Extracted
/// from `new_hybrid` so it can be CALLED: the `_ => true` arm below fails
/// OPEN, so a design missing from it is silently ACCEPTED with parameters
/// its baseline rejects, and as a `let` inside the constructor that could
/// not be asserted on. `compact_parity` now checks every baseline/compact
/// pair agrees here.
fn params_ok(policy: PaperPolicy) -> bool {
	match policy {
		PaperPolicy::LruLfuHybrid(promote_k)
		| PaperPolicy::LruLfuCompactHybrid(promote_k) => promote_k != 0,

		// The one two-ratio design: BOTH must be in range, so it cannot
		// join the single-ratio group below.
		PaperPolicy::TwoQFullFastAdmissionHybrid(k_in, k_out)
		| PaperPolicy::TwoQFullFastAdmissionCompactHybrid(k_in, k_out) => {
			(0.0..=1.0).contains(&k_in) && (0.0..=1.0).contains(&k_out)
		},

		// The 2Q family keeps an INCLUSIVE upper bound. No 2Q stack
		// derives a budget from `1 - k_in`: `fifo_capacity` is
		// `k_in * max_size` and the main queue is bounded by the
		// cache's overall `max_size`, so `k_in == 1.0` gives the FIFO
		// queue the whole cache -- extreme, but every queue still has
		// capacity and eviction drains the FIFO tail unconditionally.
		PaperPolicy::TwoQHybrid(r)
		| PaperPolicy::TwoQCompactHybrid(r)
		| PaperPolicy::TwoQFastAdmissionCompactHybrid(r)
		| PaperPolicy::TwoQFastAdmissionHybrid(r)
		| PaperPolicy::TwoQFastAdmissionReprieveCompactHybrid(r)
		| PaperPolicy::TwoQFastAdmissionReprieveHybrid(r)
		| PaperPolicy::TwoQGhostCompactHybrid(r)
		| PaperPolicy::TwoQGhostHybrid(r) => (0.0..=1.0).contains(&r),

		// The s3-fifo family EXCLUDES 1.0. These stacks size the main
		// queue at `(1 - ratio) * max_size`, mirroring
		// `SThreeFifoStack`, so a ratio of exactly 1 leaves it zero
		// bytes. `Stack::is_full` is `used >= max`, so an *empty* main
		// queue then reports itself full: `evict_one` skips the
		// one-access queue and `evict_main` pops nothing, returning
		// `None` while the cache is still over budget, and
		// `apply_evictions` spins on it. Rejecting the endpoint makes
		// that unreachable rather than guarding it after the fact.
		// (`SThreeFifoStack` has the same degeneracy at 1.0; its own
		// parser is tightened to match.)
		PaperPolicy::S3FifoHybrid(r)
		| PaperPolicy::S3FifoFaithfulCompactHybrid(r)
		| PaperPolicy::S3FifoFaithfulFastAdmissionCompactHybrid(r)
		| PaperPolicy::S3FifoFaithfulReprieveCompactHybrid(r)
		| PaperPolicy::S3FifoFaithfulFastAdmissionReprieveCompactHybrid(r)
		| PaperPolicy::S3FifoCompactHybrid(r)
		| PaperPolicy::S3FifoGhostCompactHybrid(r)
		| PaperPolicy::S3FifoGhostHybrid(r)
		| PaperPolicy::S3FifoGhostLazyDemotionCompactHybrid(r)
		| PaperPolicy::S3FifoGhostLazyDemotionFastAdmissionCompactHybrid(r)
		| PaperPolicy::S3FifoGhostLazyDemotionHybrid(r)
		| PaperPolicy::S3FifoGhostLazyDemotionFastAdmissionHybrid(r)
		| PaperPolicy::S3FifoGhostLazyDemotionFastAdmissionMidpointHybrid(r)
		| PaperPolicy::S3FifoGhostLazyDemotionFastAdmissionMidpointCompactHybrid(r) => {
			(0.0..1.0).contains(&r)
		},

		// The four reprieve designs keep the INCLUSIVE bound, for the
		// same reason the 2Q family does: they derive no budget from
		// `1 - ratio`. Their `evict_one` is purely the main queue's tail
		// loop -- the one-access queue never reaches it, being drained
		// synchronously by `settle_one_access()` against its own
		// capacity -- so the `!main.is_full()` dispatch gate that
		// `main_capacity` exists to serve is absent here, and no queue
		// can report itself full at zero capacity. Their real budgets
		// (`one_access_capacity` and `fast_capacity`) partition the
		// DRAM/PMEM axis instead, which `1 - ratio` says nothing about.
		PaperPolicy::S3FifoLazyDemotionFastAdmissionMidpointReprieveHybrid(r)
		| PaperPolicy::S3FifoLazyDemotionFastAdmissionMidpointReprieveCompactHybrid(r)
		| PaperPolicy::S3FifoLazyDemotionFastAdmissionReprieveHybrid(r)
		| PaperPolicy::S3FifoLazyDemotionFastAdmissionReprieveCompactHybrid(r)
		| PaperPolicy::S3FifoLazyDemotionReprieveHybrid(r)
		| PaperPolicy::S3FifoLazyDemotionReprieveCompactHybrid(r)
		| PaperPolicy::S3FifoLazyDemotionFastAdmissionSplitSlowReprieveHybrid(r)
		| PaperPolicy::S3FifoLazyDemotionFastAdmissionSplitSlowReprieveCompactHybrid(r) => {
			(0.0..=1.0).contains(&r)
		},

		_ => true,
	}
}

fn s3_fifo_queue_budgets(policy: PaperPolicy) -> Option<(f64, bool)> {
	match policy {
		// These five size main at `(1 - ratio) * max_size`, mirroring
		// `SThreeFifoStack`, and gate eviction on its fullness.
		PaperPolicy::S3FifoHybrid(r)
		| PaperPolicy::S3FifoCompactHybrid(r)
		| PaperPolicy::S3FifoGhostCompactHybrid(r)
		| PaperPolicy::S3FifoGhostHybrid(r)
		| PaperPolicy::S3FifoGhostLazyDemotionCompactHybrid(r)
		| PaperPolicy::S3FifoGhostLazyDemotionFastAdmissionCompactHybrid(r)
		| PaperPolicy::S3FifoGhostLazyDemotionHybrid(r)
		| PaperPolicy::S3FifoGhostLazyDemotionFastAdmissionHybrid(r)
		| PaperPolicy::S3FifoGhostLazyDemotionFastAdmissionMidpointHybrid(r)
		| PaperPolicy::S3FifoFaithfulCompactHybrid(r)
		| PaperPolicy::S3FifoFaithfulFastAdmissionCompactHybrid(r)
		| PaperPolicy::S3FifoFaithfulReprieveCompactHybrid(r)
		| PaperPolicy::S3FifoFaithfulFastAdmissionReprieveCompactHybrid(r)
		| PaperPolicy::S3FifoGhostLazyDemotionFastAdmissionMidpointCompactHybrid(r) => Some((r, true)),

		// The reprieve stacks: one-access budget only.
		PaperPolicy::S3FifoLazyDemotionFastAdmissionMidpointReprieveHybrid(r)
		| PaperPolicy::S3FifoLazyDemotionFastAdmissionMidpointReprieveCompactHybrid(r)
		| PaperPolicy::S3FifoLazyDemotionFastAdmissionReprieveHybrid(r)
		| PaperPolicy::S3FifoLazyDemotionFastAdmissionReprieveCompactHybrid(r)
		| PaperPolicy::S3FifoLazyDemotionReprieveHybrid(r)
		| PaperPolicy::S3FifoLazyDemotionReprieveCompactHybrid(r)
		| PaperPolicy::S3FifoLazyDemotionFastAdmissionSplitSlowReprieveHybrid(r)
		| PaperPolicy::S3FifoLazyDemotionFastAdmissionSplitSlowReprieveCompactHybrid(r) => Some((r, false)),

		_ => None,
	}
}

/// The engine every hybrid design runs on: `new`/`with_hasher`, the cache
/// operations, and the single `hybrid_stats()` accessor. The design is not
/// chosen here -- it arrives as the `PaperPolicy` argument to `new`, is stored
/// in `AtomicStatus`, and is consulted at runtime for the two things that
/// still vary: which stack `init_policy_stack` builds, and which arm
/// `hybrid_policy::admission_tier` takes inside `set()`.
///
/// Only one other impl block on this type exists, below: the size-split
/// design's `new_sized`/`with_hasher_sized`, which take three sizing scalars
/// instead of one and so cannot share this block's constructor.
#[cfg(feature = "hybrid_cache_common")]
impl<K, S> PaperCache<K, TieredBuffer, S>
where
	K: 'static + Eq + Hash + TypeSize + Clone + Send + Sync,
	S: Default + Clone + BuildHasher,
{
	/// Creates an empty tiered cache running the given hybrid `policy`, with
	/// overall byte budget `max_size` and initial fast-tier budget
	/// `fast_tier_size` (adjustable afterward via
	/// [`Self::set_fast_tier_size`]). Policy parameters (`k_in`, ghost
	/// ratios, `promote_k`, ...) travel inside the [`PaperPolicy`] value.
	///
	/// The size-split design has its own constructor, [`Self::new_sized`],
	/// because it takes three sizing scalars rather than one.
	///
	/// # Errors
	///
	/// [`CacheError::InvalidPolicy`] if `policy` is not a hybrid design (or
	/// is one of the two size-split designs, which `new_sized`/
	/// `new_sized_compact` serve), if its parameters are
	/// out of range, or -- for the s3-fifo designs that size a main queue at
	/// `(1 - ratio) * max_size` -- if that budget truncates to zero at this
	/// `max_size`, which would leave the eviction loop unable to free
	/// anything. Note the ratio bound alone cannot catch the last case, since
	/// it depends on a `max_size` the policy never sees; a zero-length
	/// ONE-ACCESS queue is legal, being exactly what `ratio == 0.0` asks for.
	/// [`CacheError::ZeroCacheSize`]/[`CacheError::InvalidFastTierSize`] as
	/// for every other constructor.
	pub fn new(
		max_size: CacheSize,
		fast_tier_size: CacheTierSize,
		policy: PaperPolicy,
	) -> Result<Self, CacheError> {
		Self::with_hasher(max_size, fast_tier_size, policy, Default::default())
	}

	/// Creates an empty tiered cache with the supplied hasher. See [`Self::new`].
	pub fn with_hasher(
		max_size: CacheSize,
		fast_tier_size: CacheTierSize,
		policy: PaperPolicy,
		hasher: S,
	) -> Result<Self, CacheError> {
		Self::new_hybrid(max_size, fast_tier_size, policy, hasher)
	}

	// `lru_sized_hybrid_cache` doesn't call this: it needs three sizing
	// scalars (two independent fast-segment capacities + a threshold)
	// threaded to three different places rather than this method's single
	// `CacheTierSize`, so it has its own bespoke `new_sized_hybrid` instead
	// (see the size-split impl block below).
	#[cfg(feature = "hybrid_cache_common")]
	fn new_hybrid(
		max_size: CacheSize,
		fast_tier_size: CacheTierSize,
		policy: PaperPolicy,
		hasher: S,
	) -> Result<Self, CacheError> {
		if max_size == 0 {
			return Err(CacheError::ZeroCacheSize);
		}

		// The size-split design needs three sizing scalars and has its own
		// constructor (`new_sized`); everything non-hybrid is simply not a
		// tiered design.
		if !policy.is_hybrid()
			|| matches!(policy, PaperPolicy::LruSizedHybrid | PaperPolicy::LruSizedCompactHybrid)
		{
			return Err(CacheError::InvalidPolicy);
		}

		// Parameter ranges the per-design constructors used to enforce:
		// every ratio-shaped parameter lives in [0, 1], and a promotion
		// threshold of zero degenerates to plain LRU and is rejected rather
		// than silently meaning something else.
		if !params_ok(policy) {
			return Err(CacheError::InvalidPolicy);
		}

		let fast_capacity = fast_tier_size.to_bytes();

		if fast_capacity == 0 || fast_capacity > max_size {
			return Err(CacheError::InvalidFastTierSize);
		}

		// The main budget is a truncating cast, so a ratio well inside (0, 1)
		// still rounds it to zero when `max_size` is small enough: 1 - 0.9995
		// of 1_000 is 0.5, which truncates to 0. That is the same
		// zero-capacity main queue the endpoint exclusion above prevents,
		// reached by a different route, and no bound on the ratio alone can
		// catch it -- whether a ratio is too extreme depends on `max_size`,
		// which the policy parser never sees. This is the only place both are
		// known.
		//
		// Only the MAIN budget is checked. A one-access budget of zero is not
		// a failure: it is what `ratio == 0.0` asks for, it is documented and
		// tested as legal, and it degrades cleanly -- every insert goes
		// straight to main, which holds the entire budget. Rejecting it here
		// would have made this contradict the parser.
		//
		// Deliberately after the `max_size == 0` and fast-tier checks, so a
		// zero-sized cache still reports `ZeroCacheSize`/`InvalidFastTierSize`
		// rather than being re-diagnosed as a bad ratio.
		if let Some((ratio, sizes_main)) = s3_fifo_queue_budgets(policy) {
			if sizes_main && ((1.0 - ratio) * max_size as f64) as CacheSize == 0 {
				return Err(CacheError::InvalidPolicy);
			}
		}

		let policies = [policy];

		let objects = new_hybrid_object_map();
		let status = Arc::new(AtomicStatus::new(max_size, &policies, policy)?);
		let overhead_manager = Arc::new(OverheadManager::new(&status));

		// Requirement: fast-tier size is runtime-configurable (not baked
		// into the policy string, unlike e.g. `TwoQ`/`SThreeFifo`), so the
		// requested capacity is recorded on the shared status immediately;
		// `init_policy_stack`'s 20%-of-max_size default (see
		// `policy_stack/mod.rs`) is overridden below via `ResizeFastTier`.
		status.set_fast_tier_capacity(fast_capacity);

		// Reallocates a value into the target tier's representation. Must
		// preserve byte length exactly: both `status.base_used_size` and
		// the active stack's own per-key size bookkeeping assume a
		// migration never changes an object's accounted size.
		// Returns `None` when the value is already in the requested tier, in
		// which case the worker skips the swap entirely.
		//
		// This is the only place the check can live: the worker is generic
		// over `V` and cannot ask an arbitrary value which tier it occupies,
		// which is why a stack emitting a migration for an already-correctly
		// -placed object used to cost a full allocate-and-memcpy that produced
		// a byte-identical object at a new address. `LfuHybridStack` did
		// exactly that on every latched admission (445,465,067 migrations
		// against ~448M sets on cluster12 before it was fixed at source), and
		// `TwoQHybridStack` still reaches this case legitimately under a
		// lookaside workload: `admission_tier` returns `Fast` for a re-set --
		// correct, since the key is now MRU -- so `set()` has already built
		// the bytes in DRAM by the time `touch_main_fast` emits its
		// `(key, Tier::Fast)` promotion.
		let migrate: Box<dyn Fn(&TieredBuffer, Tier) -> Option<TieredBuffer> + Send + Sync> =
			Box::new(|buffer, tier| match (tier, buffer.is_fast()) {
				(Tier::Fast, true) | (Tier::Slow, false) => None,
				(Tier::Fast, false) => Some(TieredBuffer::new_fast(buffer.as_ref())),
				(Tier::Slow, true) => Some(TieredBuffer::new_slow(buffer.as_ref())),
			});

		let (worker_fanout, worker_handles) = WorkerFanout::new_with_tier_migration(
			&objects,
			&status,
			&overhead_manager,
			migrate,
		)?;

		let cache = PaperCache {
			objects,
			status,
			workers: Arc::new(worker_fanout),
			worker_handles,
			overhead_manager,
			hasher,
		};

		cache.broadcast(WorkerEvent::ResizeFastTier(fast_capacity))?;

		Ok(cache)
	}

	/// Returns the current cache version.
	#[must_use]
	pub fn version(&self) -> String {
		env!("CARGO_PKG_VERSION").to_owned()
	}

	/// Returns the current statistics.
	pub fn status(&self) -> Result<Status, CacheError> {
		self.status.try_to_status()
	}

	/// Gets the value associated with the supplied key.
	/// If the key was not found in the cache, returns a [`CacheError`].
	///
	/// The object-map guard is deliberately released *before* the value
	/// bytes are copied out. `Object::data()` is only an `Arc` refcount
	/// bump, and the `Arc` keeps the buffer alive on its own, so the shard
	/// lock is only needed for the lookup/validation -- not for the copy,
	/// which at this crate's real object sizes (~16 KB average on the
	/// benchmark traces) is by far the expensive part, and is a PMEM read
	/// whenever the object is slow-tier resident.
	///
	/// This matters because `PolicyWorker::apply_tier_migrations` takes a
	/// *write* guard on the same shards to physically move bytes between
	/// tiers. Holding a read guard across a multi-microsecond copy stalls
	/// those writers (and, transitively, readers queued behind them), which
	/// shows up as GET tail latency rather than as a uniform slowdown.
	/// Dropping the guard first shrinks this critical section to a hash
	/// lookup, a key compare, an expiry check and a refcount bump.
	///
	/// Releasing early means a concurrent migration or eviction can retire
	/// the object while the copy is in flight. That is correct, not a race:
	/// the `Arc` guarantees the bytes stay valid, and the caller gets a
	/// snapshot that was live at the moment of the lookup -- the same
	/// guarantee it had before, since the value could equally have changed
	/// the instant after the guard was dropped.
	pub fn get(&self, key: &K) -> Result<Vec<u8>, CacheError> {
		let hashed_key = self.hash_key(key);

		let maybe_data = match self.objects.get_ref(&hashed_key) {
			Some(object) if object.key_matches(key) && !object.is_expired() => Some(object.data()),
			_ => None,
		};

		let result = match maybe_data {
			Some(arc_val) => {
				self.status.incr_hits();
				let bytes: &[u8] = arc_val.as_ref().as_ref();
				Ok(bytes.to_vec())
			},

			None => {
				self.status.incr_misses();
				Err(CacheError::KeyNotFound)
			},
		};

		self.broadcast(WorkerEvent::Get(hashed_key, result.is_ok()))?;

		result
	}

	/// Diagnostic twin of [`Self::get`] that copies a hit into a caller-owned
	/// buffer instead of allocating a fresh `Vec` per call.
	///
	/// `get()` fuses two independent costs: locating and reading the value --
	/// which is what a tiering design changes -- and allocating the buffer to
	/// return it in, which is what the allocator configuration changes. Measured
	/// on Twitter cluster13 (2026-08-28) the second term dominated the first at
	/// the median, because the median value is 123 B while the mean is 4.9 KB.
	/// Comparing two cache designs through `get()` therefore compares their
	/// allocator behaviour as much as their cache behaviour; this method exists
	/// to measure them apart. See the `segregated_value_arena` feature.
	#[cfg(not(feature = "enable_tiering_manager"))]
	pub fn get_into(&self, key: &K, out: &mut Vec<u8>) -> Result<(), CacheError> {
		// Sampled step profiler -- see the GI_* statics at the bottom of this
		// file. One call in 64; hits only, matching what GET latency measures.
		let prof = gi_prof_enabled()
			&& GI_TICK.with(|c| {
				let t = c.get();
				c.set(t.wrapping_add(1));
				t & 63 == 0
			});
		let t0 = if prof { Some(std::time::Instant::now()) } else { None };

		let hashed_key = self.hash_key(key);
		let t1 = if prof { Some(std::time::Instant::now()) } else { None };

		let maybe_data = match self.objects.get_ref(&hashed_key) {
			Some(object) if object.key_matches(key) && !object.is_expired() => Some(object.data()),
			_ => None,
		};
		let t2 = if prof { Some(std::time::Instant::now()) } else { None };

		// Which tier served this hit (1 = fast/DRAM; the all-DRAM shape never
		// reassigns it). Read only by the sampled profiler below.
		#[allow(unused_mut)]
		let mut gi_fast: u64 = 1;

		let result = match maybe_data {
			Some(arc_val) => {
				self.status.incr_hits();
				out.clear();
				gi_fast = if arc_val.is_fast() { 1 } else { 0 };
				out.extend_from_slice(arc_val.as_ref().as_ref());
				Ok(())
			},

			None => {
				self.status.incr_misses();
				Err(CacheError::KeyNotFound)
			},
		};
		let t3 = if prof { Some(std::time::Instant::now()) } else { None };

		self.broadcast(WorkerEvent::Get(hashed_key, result.is_ok()))?;

		if let (Some(t0), Some(t1), Some(t2), Some(t3), true) = (t0, t1, t2, t3, result.is_ok()) {
			let t4 = std::time::Instant::now();
			use std::sync::atomic::Ordering::Relaxed;
			let (h, l, c, b) = (
				(t1 - t0).as_nanos() as u64,
				(t2 - t1).as_nanos() as u64,
				(t3 - t2).as_nanos() as u64,
				(t4 - t3).as_nanos() as u64,
			);
			GI_N.fetch_add(1, Relaxed);
			GI_HASH.fetch_add(h, Relaxed);
			GI_LOOKUP.fetch_add(l, Relaxed);
			GI_COPY.fetch_add(c, Relaxed);
			GI_BCAST.fetch_add(b, Relaxed);
			// Off the timed steps (after t4); the lock is uncontended at 1-in-64.
			if let Ok(mut v) = GI_SAMPLES.lock() {
				if v.capacity() == 0 {
					v.reserve_exact(1 << 20);
				}
				if v.len() < (1 << 20) {
					v.push([out.len() as u64, h, l, c, b, gi_fast]);
				}
			}
		}

		result
	}

	/// Sets the supplied key and value in the cache. Which tier the value's
	/// bytes are built in is decided by `hybrid_policy::admission_tier`, whose
	/// match arm for the cache's policy carries that design's admission rule.
	/// Returns a
	/// [`CacheError`] if the value size is zero or larger than the cache's
	/// maximum size.
	pub fn set(&self, key: K, value: &[u8], ttl: Option<u32>) -> Result<(), CacheError> {
		let hashed_key = self.hash_key(&key);

		let tier = crate::hybrid_policy::admission_tier(
			self.status.policy(),
			hashed_key,
			&self.status,
			&self.objects,
		);
		let val_buf = match tier {
			Tier::Fast => TieredBuffer::new_fast(value),
			Tier::Slow => TieredBuffer::new_slow(value),
		};

		let object = Object::new(key, val_buf, ttl);
		let base_size = self.overhead_manager.base_size(&object);
		let dram_resident = self.overhead_manager.dram_resident_size(&object);
		let expiry = object.expiry();

		if base_size == 0 {
			return Err(CacheError::ZeroValueSize);
		}

		if self.status.exceeds_max_size(base_size) {
			return Err(CacheError::ExceedingValueSize);
		}

		self.status.incr_sets();

		let old_object_info = self.objects
			.insert(hashed_key, object)
			.map(|old_object| {
				let base_size = self.overhead_manager.base_size(&old_object);
				let expiry = old_object.expiry();

				(base_size, expiry)
			});

		let base_size_delta = if let Some((old_object_size, _)) = old_object_info {
			base_size as i64 - old_object_size as i64
		} else {
			self.status.incr_num_objects();
			base_size as i64
		};

		self.status.update_base_used_size(base_size_delta);
		self.broadcast(WorkerEvent::Set(
			hashed_key,
			base_size,
			dram_resident,
			expiry,
			old_object_info,
		))?;

		Ok(())
	}

	/// Deletes the object associated with the supplied key in the cache.
	/// Returns a [`CacheError`] if the key was not found in the cache.
	pub fn del(&self, key: &K) -> Result<(), CacheError> {
		let hashed_key = self.hash_key(key);

		let (removed_hashed_key, object) = erase(
			&self.objects,
			&self.status,
			&self.overhead_manager,
			Some(EraseKey::Original(key, hashed_key)),
		)?;

		self.status.incr_dels();
		self.broadcast(WorkerEvent::Del(removed_hashed_key, object.expiry()))?;

		Ok(())
	}

	/// Checks if an object with the supplied key exists in the cache without
	/// altering any of the cache's internal queues.
	pub fn has(&self, key: &K) -> bool {
		let hashed_key = self.hash_key(key);

		self.objects
			.get_ref(&hashed_key)
			.is_some_and(|object| object.key_matches(key) && !object.is_expired())
	}

	/// Gets (peeks) the value associated with the supplied key without
	/// altering any of the cache's internal queues (including tier — a peek
	/// never triggers a promotion). If the key was not found in the cache,
	/// returns a [`CacheError`].
	pub fn peek(&self, key: &K) -> Result<Arc<TieredBuffer>, CacheError> {
		let hashed_key = self.hash_key(key);

		match self.objects.get_ref(&hashed_key) {
			Some(object) if object.key_matches(key) && !object.is_expired() =>
				Ok(object.data()),

			_ => Err(CacheError::KeyNotFound),
		}
	}

	/// Sets the TTL associated with the supplied key.
	/// If the key was not found in the cache, returns a [`CacheError`].
	pub fn ttl(&self, key: &K, ttl: Option<u32>) -> Result<(), CacheError> {
		let hashed_key = self.hash_key(key);

		let mut object = match self.objects.get_mut_ref(&hashed_key) {
			Some(object) if object.key_matches(key) && !object.is_expired() => object,
			_ => return Err(CacheError::KeyNotFound),
		};

		let old_expiry = object.expiry();
		let old_base_size = self.overhead_manager.base_size(&object);

		object.expires(ttl);

		let new_expiry = object.expiry();
		let new_base_size = self.overhead_manager.base_size(&object);

		self.status.update_base_used_size(new_base_size as i64 - old_base_size as i64);
		self.broadcast(WorkerEvent::Ttl(hashed_key, old_expiry, new_expiry))?;

		Ok(())
	}

	/// Gets the size of the value associated with the supplied key in bytes.
	/// If the key was not found in the cache, returns a [`CacheError`].
	pub fn size(&self, key: &K) -> Result<ObjectSize, CacheError> {
		let hashed_key = self.hash_key(key);

		match self.objects.get_ref(&hashed_key) {
			Some(object) if object.key_matches(key) && !object.is_expired() =>
				Ok(self.overhead_manager.total_size(&object)),

			_ => Err(CacheError::KeyNotFound),
		}
	}

	/// Deletes all objects in the cache and sets the cache's used size to zero.
	pub fn wipe(&self) -> Result<(), CacheError> {
		info!("Wiping cache");

		self.objects.clear();
		self.status.clear();

		self.broadcast(WorkerEvent::Wipe)?;

		Ok(())
	}

	/// Resizes the cache's overall maximum size.
	/// If the supplied size is zero, returns a [`CacheError`].
	///
	/// Note this is the *overall* cache capacity, independent of the
	/// fast-tier budget — see [`Self::set_fast_tier_size`]. (`two_q_hybrid_cache`
	/// additionally rescales its FIFO queue's byte budget proportionally,
	/// inside `TwoQHybridStack::resize` -- not this method.)
	///
	/// # Errors
	///
	/// [`CacheError::ZeroCacheSize`] if `max_size` is zero, and
	/// [`CacheError::InvalidPolicy`] if the active s3-fifo design's queue
	/// budgets would not survive the new size -- the same condition `new`
	/// rejects, reported the same way.
	pub fn resize(&self, max_size: CacheSize) -> Result<(), CacheError> {
		if max_size == 0 {
			return Err(CacheError::ZeroCacheSize);
		}

		// `Stack::resize` recomputes both s3-fifo budgets against the NEW
		// max_size, so a resize can reintroduce exactly the zero-capacity
		// main queue the constructor refuses -- a ratio of 0.9995 is fine at
		// max_size 1_000_000 (main = 500 B) and degenerate at 1_000
		// (main = 0 B), which spins the eviction loop. The size is legal and
		// the policy is legal; it is the pair that is not, so this has to be
		// checked here as well as in `new`.
		if let Some((ratio, sizes_main)) = s3_fifo_queue_budgets(self.status.policy()) {
			if sizes_main && ((1.0 - ratio) * max_size as f64) as CacheSize == 0 {
				return Err(CacheError::InvalidPolicy);
			}
		}

		let current_max_size = self.status.max_size();

		if max_size == current_max_size {
			return Ok(());
		}

		info!(
			"Resizing cache from {} to {}",
			fmt::memory(current_max_size, Some(2)),
			fmt::memory(max_size, Some(2)),
		);

		self.status.set_max_size(max_size);
		self.broadcast(WorkerEvent::Resize(max_size))?;

		Ok(())
	}

	/// Runtime-adjusts the fast-tier byte budget. Shrinking it may trigger
	/// immediate demotions (see the active policy stack's `settle_fast_tier`).
	///
	/// # Errors
	///
	/// Returns [`CacheError::InvalidFastTierSize`] if `size` resolves to
	/// zero bytes or exceeds the cache's overall `max_size`.
	pub fn set_fast_tier_size(&self, size: CacheTierSize) -> Result<(), CacheError> {
		let bytes = size.to_bytes();

		if bytes == 0 || bytes > self.status.max_size() {
			return Err(CacheError::InvalidFastTierSize);
		}

		self.status.set_fast_tier_capacity(bytes);
		self.broadcast(WorkerEvent::ResizeFastTier(bytes))?;

		Ok(())
	}

	/// Returns the current fast-tier byte budget.
	#[must_use]
	pub fn fast_tier_size(&self) -> CacheSize {
		self.status.fast_tier_capacity()
	}

	/// Returns the active hybrid design's tier-movement counters and live
	/// tier gauges, in a design-neutral shape.
	///
	/// The only stats accessor there is: the per-design
	/// `<design>_hybrid_stats()` methods were removed with the runtime-policy
	/// unification, and the 19 `<Design>HybridStats` names are aliases of the
	/// one `HybridStats` struct. The 8 size-split gauges read zero unless the
	/// cache is running `LruSizedHybrid`. Read from
	/// the design and want its extras.
	#[must_use]
	pub fn hybrid_stats(&self) -> HybridStats {
		self.status.hybrid_stats()
	}

	/// Returns which tier `key` currently lives in, or `None` if the key
	/// isn't present (or has expired). Useful for tests/diagnostics — unlike
	/// an external two-cache composition's `has_in_dram`/`has_in_pmem` pair, there's only
	/// one object map here, so tier is a property read off the object itself.
	#[must_use]
	pub fn tier_of(&self, key: &K) -> Option<Tier> {
		let hashed_key = self.hash_key(key);

		self.objects.get_ref(&hashed_key).and_then(|object| {
			if !object.key_matches(key) || object.is_expired() {
				return None;
			}

			Some(if object.data().is_fast() { Tier::Fast } else { Tier::Slow })
		})
	}

	fn broadcast(&self, event: WorkerEvent) -> Result<(), CacheError> {
		self.workers.send(event)
	}

	fn hash_key(&self, key: &K) -> HashedKey {
		self.hasher.hash_one(key)
	}
}

/// Single-instance, segmented-LRU hybrid cache with a size-split fast AND
/// slow tier: same `PaperCache<K, TieredBuffer>` architecture and LRU
/// admission/promotion/demotion/eviction semantics as `lru_hybrid_cache`,
/// but each tier's bookkeeping is split into two independently-tracked
/// segments ("small"/"large") by object size. See the
/// `lru_sized_hybrid_cache` module docs.
///
/// Sizing knobs: [`Self::set_fast_tier_size`]/[`Self::fast_tier_size`]
/// (defined on the shared generic block above) resize/read the SMALL fast
/// segment specifically for this design -- unlike every other hybrid, where
/// they mean the whole fast tier -- because this design has a second,
/// independent fast segment with no shared-block equivalent.
/// [`Self::set_large_fast_tier_size`]/[`Self::large_fast_tier_size`] and
/// [`Self::set_size_threshold`]/[`Self::size_threshold`] are this design's
/// own bespoke accessors, defined here.
#[cfg(feature = "hybrid_cache_common")]
impl<K, S> PaperCache<K, TieredBuffer, S>
where
	K: 'static + Eq + Hash + TypeSize + Clone + Send + Sync,
	S: Default + Clone + BuildHasher,
{
	/// Creates an empty `PaperCache` running `PaperPolicy::LruSizedHybrid`,
	/// with the given overall `max_size` and initial small/large fast-segment
	/// byte budgets and size-classification threshold (each independently
	/// adjustable afterward via [`Self::set_fast_tier_size`]/
	/// [`Self::set_large_fast_tier_size`]/[`Self::set_size_threshold`]). An
	/// object whose size is strictly below `size_threshold` routes to the
	/// small segment on admission, promotion, or a reclassifying overwrite;
	/// at or above routes to the large segment.
	///
	/// # Errors
	///
	/// Returns [`CacheError::ZeroCacheSize`] if `max_size` is zero, or
	/// [`CacheError::InvalidFastTierSize`] if either `small_fast_tier_size`
	/// or `large_fast_tier_size` resolves to zero bytes or exceeds
	/// `max_size` (checked independently -- there is no requirement that
	/// their sum stay under `max_size`). `size_threshold` is never rejected.
	pub fn new_sized(
		max_size: CacheSize,
		small_fast_tier_size: CacheTierSize,
		large_fast_tier_size: CacheTierSize,
		size_threshold: CacheTierSize,
	) -> Result<Self, CacheError> {
		Self::with_hasher_sized(max_size, small_fast_tier_size, large_fast_tier_size, size_threshold, Default::default())
	}

	/// Creates an empty size-split cache with the supplied hasher. See [`Self::new_sized`].
	pub fn with_hasher_sized(
		max_size: CacheSize,
		small_fast_tier_size: CacheTierSize,
		large_fast_tier_size: CacheTierSize,
		size_threshold: CacheTierSize,
		hasher: S,
	) -> Result<Self, CacheError> {
		Self::new_sized_hybrid(max_size, small_fast_tier_size, large_fast_tier_size, size_threshold, PaperPolicy::LruSizedHybrid, hasher)
	}

	/// Creates an empty `PaperCache` running
	/// `PaperPolicy::LruSizedCompactHybrid` -- the slab-backed compaction of
	/// `PaperPolicy::LruSizedHybrid`, behaviourally identical to it. Same
	/// arguments, same errors, same runtime accessors as [`Self::new_sized`].
	pub fn new_sized_compact(
		max_size: CacheSize,
		small_fast_tier_size: CacheTierSize,
		large_fast_tier_size: CacheTierSize,
		size_threshold: CacheTierSize,
	) -> Result<Self, CacheError> {
		Self::with_hasher_sized_compact(max_size, small_fast_tier_size, large_fast_tier_size, size_threshold, Default::default())
	}

	/// Creates an empty compact size-split cache with the supplied hasher. See
	/// [`Self::new_sized_compact`].
	pub fn with_hasher_sized_compact(
		max_size: CacheSize,
		small_fast_tier_size: CacheTierSize,
		large_fast_tier_size: CacheTierSize,
		size_threshold: CacheTierSize,
		hasher: S,
	) -> Result<Self, CacheError> {
		Self::new_sized_hybrid(max_size, small_fast_tier_size, large_fast_tier_size, size_threshold, PaperPolicy::LruSizedCompactHybrid, hasher)
	}

	/// Duplicates `new_hybrid`'s common setup rather than reusing it: this
	/// design needs three sizing scalars (two fast-segment capacities + a
	/// threshold) threaded to three different places -- two `AtomicStatus`
	/// fields plus three `WorkerEvent` broadcasts -- rather than
	/// `new_hybrid`'s single `CacheTierSize`/one broadcast, so widening
	/// `new_hybrid`'s signature for every other hybrid design's benefit was
	/// judged more invasive than this small duplication.
	fn new_sized_hybrid(
		max_size: CacheSize,
		small_fast_tier_size: CacheTierSize,
		large_fast_tier_size: CacheTierSize,
		size_threshold: CacheTierSize,
		policy: PaperPolicy,
		hasher: S,
	) -> Result<Self, CacheError> {
		if max_size == 0 {
			return Err(CacheError::ZeroCacheSize);
		}

		let small_capacity = small_fast_tier_size.to_bytes();
		let large_capacity = large_fast_tier_size.to_bytes();
		let threshold = size_threshold.to_bytes();

		if small_capacity == 0 || small_capacity > max_size {
			return Err(CacheError::InvalidFastTierSize);
		}

		if large_capacity == 0 || large_capacity > max_size {
			return Err(CacheError::InvalidFastTierSize);
		}

		let policies = [policy];

		let objects = new_hybrid_object_map();
		let status = Arc::new(AtomicStatus::new(max_size, &policies, policy)?);
		let overhead_manager = Arc::new(OverheadManager::new(&status));

		status.set_fast_tier_capacity(small_capacity);
		status.set_hybrid_large_fast_capacity(large_capacity);
		status.set_hybrid_size_threshold(threshold);

		// Same byte-length-preserving contract `new_hybrid`'s `migrate`
		// closure documents.
		// Returns `None` when the value is already in the requested tier, in
		// which case the worker skips the swap entirely.
		//
		// This is the only place the check can live: the worker is generic
		// over `V` and cannot ask an arbitrary value which tier it occupies,
		// which is why a stack emitting a migration for an already-correctly
		// -placed object used to cost a full allocate-and-memcpy that produced
		// a byte-identical object at a new address. `LfuHybridStack` did
		// exactly that on every latched admission (445,465,067 migrations
		// against ~448M sets on cluster12 before it was fixed at source), and
		// `TwoQHybridStack` still reaches this case legitimately under a
		// lookaside workload: `admission_tier` returns `Fast` for a re-set --
		// correct, since the key is now MRU -- so `set()` has already built
		// the bytes in DRAM by the time `touch_main_fast` emits its
		// `(key, Tier::Fast)` promotion.
		let migrate: Box<dyn Fn(&TieredBuffer, Tier) -> Option<TieredBuffer> + Send + Sync> =
			Box::new(|buffer, tier| match (tier, buffer.is_fast()) {
				(Tier::Fast, true) | (Tier::Slow, false) => None,
				(Tier::Fast, false) => Some(TieredBuffer::new_fast(buffer.as_ref())),
				(Tier::Slow, true) => Some(TieredBuffer::new_slow(buffer.as_ref())),
			});

		let (worker_fanout, worker_handles) = WorkerFanout::new_with_tier_migration(
			&objects,
			&status,
			&overhead_manager,
			migrate,
		)?;

		let cache = PaperCache {
			objects,
			status,
			workers: Arc::new(worker_fanout),
			worker_handles,
			overhead_manager,
			hasher,
		};

		cache.broadcast(WorkerEvent::ResizeFastTier(small_capacity))?;
		cache.broadcast(WorkerEvent::ResizeLargeFastTier(large_capacity))?;
		cache.broadcast(WorkerEvent::ResizeSizeThreshold(threshold))?;

		Ok(cache)
	}

	/// Runtime-adjusts the LARGE fast segment's byte budget. The SMALL
	/// segment is adjusted via the shared [`Self::set_fast_tier_size`]
	/// instead (see this impl block's own doc for why).
	pub fn set_large_fast_tier_size(&self, size: CacheTierSize) -> Result<(), CacheError> {
		let bytes = size.to_bytes();

		if bytes == 0 || bytes > self.status.max_size() {
			return Err(CacheError::InvalidFastTierSize);
		}

		self.status.set_hybrid_large_fast_capacity(bytes);
		self.broadcast(WorkerEvent::ResizeLargeFastTier(bytes))?;

		Ok(())
	}

	/// Returns the LARGE fast segment's current byte budget.
	#[must_use]
	pub fn large_fast_tier_size(&self) -> CacheSize {
		self.status.hybrid_large_fast_capacity()
	}

	/// Runtime-adjusts the small/large size-classification threshold. Only
	/// affects future admissions, overwrites, and slow-to-fast promotions --
	/// already-tracked keys are not retroactively rescanned/reclassified.
	pub fn set_size_threshold(&self, threshold: CacheTierSize) -> Result<(), CacheError> {
		let bytes = threshold.to_bytes();

		self.status.set_hybrid_size_threshold(bytes);
		self.broadcast(WorkerEvent::ResizeSizeThreshold(bytes))?;

		Ok(())
	}

	/// Returns the current size-classification threshold, in bytes.
	#[must_use]
	pub fn size_threshold(&self) -> CacheSize {
		self.status.hybrid_size_threshold()
	}
}

// Tests for global_hashtable_pmem alone (without key_value_pmem)
#[cfg(all(feature = "global_hashtable_pmem", not(feature = "key_value_pmem")))]
#[cfg(all(test, feature = "global_hashtable_pmem"))]
mod test_global_hashtable_pmem_alone {
    use crate::{BufferDRAM, PaperCache, PaperPolicy};
    use std::hash::RandomState;

    #[test]
    fn test_basic_operations() {
        // Create cache with global hashtable in PMEM, values in DRAM
        let cache: PaperCache<u32, BufferDRAM, RandomState> = PaperCache::new(
            1000000,
            &[PaperPolicy::Lfu],
            PaperPolicy::Lfu,
        ).expect("Failed to create cache");

        // Test set operation
        let value = vec![1, 2, 3, 4, 5];
        assert!(cache.set(1, &value, None).is_ok());

        // Test get operation
        let retrieved = cache.get(&1).expect("Failed to get value");
        assert_eq!(retrieved, value);

        // Test has operation
        assert!(cache.has(&1));
        assert!(!cache.has(&999));

        // Test del operation
        assert!(cache.del(&1).is_ok());
        assert!(!cache.has(&1));
    }

    #[test]
    fn test_multiple_keys() {
        let cache: PaperCache<u32, BufferDRAM, RandomState> = PaperCache::new(
            10000000,
            &[PaperPolicy::Lru],
            PaperPolicy::Lru,
        ).expect("Failed to create cache");

        // Insert multiple key-value pairs
        for i in 0..100 {
            let value = vec![i as u8; 10];
            assert!(cache.set(i, &value, None).is_ok());
        }

        // Verify all keys exist
        for i in 0..100 {
            assert!(cache.has(&i));
            let retrieved = cache.get(&i).expect("Failed to get value");
            assert_eq!(retrieved, vec![i as u8; 10]);
        }
    }

    #[test]
    fn test_wipe() {
        let cache: PaperCache<String, BufferDRAM, RandomState> = PaperCache::new(
            1000000,
            &[PaperPolicy::Lfu],
            PaperPolicy::Lfu,
        ).expect("Failed to create cache");

        let key1 = "key1".to_string();
        let key2 = "key2".to_string();
        
        cache.set(key1.clone(), b"value1", None).unwrap();
        cache.set(key2.clone(), b"value2", None).unwrap();
        
        assert!(cache.has(&key1));
        assert!(cache.has(&key2));

        cache.wipe().expect("Failed to wipe cache");

        assert!(!cache.has(&key1));
        assert!(!cache.has(&key2));
    }
}

/// Unit tests verifying structural compilation and initialization with new feature flags.
/// These tests prove that eviction_stacks_pmem allocations integrate correctly with
/// the cache initialization path.
///
/// Gate on `all_dram` to get a DashMap-backed PaperCache<K, BufferDRAM> that is
/// available without any PMEM hardware. The `eviction_stacks_pmem` feature is
/// tested separately via its own test module in lfu_stack.rs.
#[cfg(all(test, feature = "all_dram"))]
mod test_new_features {
    use crate::{BufferDRAM, PaperCache, PaperPolicy};
    use std::hash::RandomState;

    /// Verify that the cache initializes and operates correctly with the LFU policy.
    /// The LfuStack used for eviction is backed by DRAM or PMEM depending on the
    /// `eviction_stacks_pmem` feature flag — both paths must initialize correctly.
    #[test]
    fn test_cache_init_with_lfu_eviction() {
        let cache: PaperCache<u32, BufferDRAM, RandomState> = PaperCache::new(
            1_000_000,
            &[PaperPolicy::Lfu],
            PaperPolicy::Lfu,
        ).expect("Cache with LFU policy must initialize successfully");

        let value: Vec<u8> = vec![10, 20, 30];
        cache.set(1u32, &value, None).expect("set must succeed");
        assert!(cache.has(&1u32), "inserted key must be present");

        let retrieved = cache.get(&1u32).expect("get must return value");
        assert_eq!(retrieved, value, "retrieved value must match inserted value");

        cache.del(&1u32).expect("del must succeed");
        assert!(!cache.has(&1u32), "deleted key must not be present");
    }

    /// Verify multiple policies work at initialization.
    #[test]
    fn test_cache_init_with_multiple_policies() {
        let cache: PaperCache<u32, BufferDRAM, RandomState> = PaperCache::new(
            1_000_000,
            &[PaperPolicy::Lfu, PaperPolicy::Lru],
            PaperPolicy::Lfu,
        ).expect("Cache with multiple policies must initialize successfully");

        cache.set(42u32, b"hello", None).expect("set must succeed");
        assert!(cache.has(&42u32));
    }
}

/// Exercises the real public `PaperCache<K, TieredBuffer>` API end to end.
///
/// Deliberately stays on the fast-tier-only path (fast_tier_size == max_size,
/// tiny values) so no object ever demotes to the slow tier: `TieredBuffer::
/// new_slow` allocates through the `Hybrid` slow-tier allocator, which
/// requires real PMEM/DAX
/// hardware and aborts ("memory allocation ... failed") in a plain dev
/// sandbox. A full integration test covering demotion/promotion/eviction
/// belongs in `tests/hybrid_cache_integration.rs` (not yet written —
/// see `CLAUDE.md`'s `lru_hybrid_cache` plan, step 12) and should be run on
/// PMEM-capable hardware.
#[cfg(all(test, feature = "lru_hybrid_cache"))]
mod test_lru_hybrid_cache {
	use crate::PaperPolicy;
    use crate::{PaperCache, TieredBuffer, CacheTierSize, Tier, CacheError};

    #[test]
    fn basic_construction_and_fast_tier_only_roundtrip() {
        let cache = PaperCache::<u32, TieredBuffer>::new(1_000_000, CacheTierSize::Bytes(1_000_000), PaperPolicy::LruHybrid).expect("cache should construct");

        cache.set(1u32, b"hello world", None).expect("set should succeed");
        assert!(cache.has(&1u32));
        assert_eq!(cache.get(&1u32).unwrap(), b"hello world");
        assert_eq!(cache.tier_of(&1u32), Some(Tier::Fast));

        let stats = cache.hybrid_stats();
        assert_eq!(stats.demotions, 0);
        assert_eq!(stats.promotions, 0);
        assert_eq!(stats.evictions, 0);

        assert_eq!(cache.fast_tier_size(), 1_000_000);
        cache.set_fast_tier_size(CacheTierSize::Bytes(500_000)).expect("resize should succeed");
        assert_eq!(cache.fast_tier_size(), 500_000);

        cache.del(&1u32).expect("del should succeed");
        assert!(!cache.has(&1u32));
        assert_eq!(cache.tier_of(&1u32), None);
    }

    #[test]
    fn invalid_fast_tier_size_is_rejected() {
        assert!(matches!(
            PaperCache::<u32, TieredBuffer>::new(1000, CacheTierSize::Bytes(2000), PaperPolicy::LruHybrid),
            Err(CacheError::InvalidFastTierSize),
        ));

        assert!(matches!(
            PaperCache::<u32, TieredBuffer>::new(1000, CacheTierSize::Bytes(0), PaperPolicy::LruHybrid),
            Err(CacheError::InvalidFastTierSize),
        ));

        let cache = PaperCache::<u32, TieredBuffer>::new(1000, CacheTierSize::Bytes(500), PaperPolicy::LruHybrid)
            .expect("cache should construct");

        assert!(matches!(
            cache.set_fast_tier_size(CacheTierSize::Bytes(2000)),
            Err(CacheError::InvalidFastTierSize),
        ));
    }

    #[test]
    fn ttl_is_preserved_across_a_set() {
        let cache = PaperCache::<u32, TieredBuffer>::new(
            1_000_000,
            CacheTierSize::Bytes(1_000_000), PaperPolicy::LruHybrid).expect("cache should construct");

        cache.set(1u32, b"value", Some(60)).expect("set should succeed");
        assert!(cache.ttl(&1u32, Some(120)).is_ok());
        assert_eq!(cache.get(&1u32).unwrap(), b"value");
    }
}

/// Exercises the real public `PaperCache<K, TieredBuffer>` API end to end
/// for `lfu_hybrid_cache`. See `test_lru_hybrid_cache`'s doc comment for why
/// this deliberately stays on the fast-tier-only path (no PMEM allocation) —
/// the full tier-crossing coverage lives in
/// `tests/hybrid_cache_integration.rs`.
#[cfg(all(test, feature = "lfu_hybrid_cache"))]
mod test_lfu_hybrid_cache {
	use crate::PaperPolicy;
    use crate::{PaperCache, TieredBuffer, CacheTierSize, Tier, CacheError};

    #[test]
    fn basic_construction_and_fast_tier_only_roundtrip() {
        let cache = PaperCache::<u32, TieredBuffer>::new(1_000_000, CacheTierSize::Bytes(1_000_000), PaperPolicy::LfuHybrid).expect("cache should construct");

        cache.set(1u32, b"hello world", None).expect("set should succeed");
        assert!(cache.has(&1u32));
        assert_eq!(cache.get(&1u32).unwrap(), b"hello world");
        assert_eq!(cache.tier_of(&1u32), Some(Tier::Fast));

        let stats = cache.hybrid_stats();
        assert_eq!(stats.demotions, 0);
        assert_eq!(stats.promotions, 0);
        assert_eq!(stats.evictions, 0);

        assert_eq!(cache.fast_tier_size(), 1_000_000);
        cache.set_fast_tier_size(CacheTierSize::Bytes(500_000)).expect("resize should succeed");
        assert_eq!(cache.fast_tier_size(), 500_000);

        cache.del(&1u32).expect("del should succeed");
        assert!(!cache.has(&1u32));
        assert_eq!(cache.tier_of(&1u32), None);
    }

    #[test]
    fn invalid_fast_tier_size_is_rejected() {
        assert!(matches!(
            PaperCache::<u32, TieredBuffer>::new(1000, CacheTierSize::Bytes(2000), PaperPolicy::LfuHybrid),
            Err(CacheError::InvalidFastTierSize),
        ));

        assert!(matches!(
            PaperCache::<u32, TieredBuffer>::new(1000, CacheTierSize::Bytes(0), PaperPolicy::LfuHybrid),
            Err(CacheError::InvalidFastTierSize),
        ));

        let cache = PaperCache::<u32, TieredBuffer>::new(1000, CacheTierSize::Bytes(500), PaperPolicy::LfuHybrid)
            .expect("cache should construct");

        assert!(matches!(
            cache.set_fast_tier_size(CacheTierSize::Bytes(2000)),
            Err(CacheError::InvalidFastTierSize),
        ));
    }

    #[test]
    fn ttl_is_preserved_across_a_set() {
        let cache = PaperCache::<u32, TieredBuffer>::new(
            1_000_000,
            CacheTierSize::Bytes(1_000_000), PaperPolicy::LfuHybrid).expect("cache should construct");

        cache.set(1u32, b"value", Some(60)).expect("set should succeed");
        assert!(cache.ttl(&1u32, Some(120)).is_ok());
        assert_eq!(cache.get(&1u32).unwrap(), b"value");
    }
}

/// Exercises the real public `PaperCache<K, TieredBuffer>` API end to end
/// for `two_q_hybrid_cache`. Unlike `test_lru_hybrid_cache`/
/// `test_lfu_hybrid_cache`, this module cannot avoid the real `Hybrid`
/// PMEM allocator: `set()` always admits via `TieredBuffer::new_slow`
/// regardless of `fast_tier_size`, so even a single `set()` call here pays
/// the one-time PMEM pool warm-up cost (see `tests/hybrid_cache_integration.rs`'s
/// module doc for details). The full tier-crossing coverage lives there.
#[cfg(all(test, feature = "two_q_hybrid_cache"))]
mod test_two_q_hybrid_cache {
	use crate::PaperPolicy;
    use crate::{PaperCache, TieredBuffer, CacheTierSize, Tier, CacheError};

    #[test]
    fn basic_construction_and_slow_tier_admission_roundtrip() {
        let cache = PaperCache::<u32, TieredBuffer>::new(
            1_000_000,
            CacheTierSize::Bytes(1_000_000), PaperPolicy::TwoQHybrid(0.5)).expect("cache should construct");

        cache.set(1u32, b"hello world", None).expect("set should succeed");
        assert!(cache.has(&1u32));
        assert_eq!(cache.get(&1u32).unwrap(), b"hello world");
        assert_eq!(cache.tier_of(&1u32), Some(Tier::Slow));

        let stats = cache.hybrid_stats();
        assert_eq!(stats.demotions, 0);
        assert_eq!(stats.evictions, 0);

        assert_eq!(cache.fast_tier_size(), 1_000_000);
        cache.set_fast_tier_size(CacheTierSize::Bytes(500_000)).expect("resize should succeed");
        assert_eq!(cache.fast_tier_size(), 500_000);

        cache.del(&1u32).expect("del should succeed");
        assert!(!cache.has(&1u32));
        assert_eq!(cache.tier_of(&1u32), None);
    }

    #[test]
    fn invalid_fast_tier_size_is_rejected() {
        assert!(matches!(
            PaperCache::<u32, TieredBuffer>::new(1000, CacheTierSize::Bytes(2000), PaperPolicy::TwoQHybrid(0.5)),
            Err(CacheError::InvalidFastTierSize),
        ));

        assert!(matches!(
            PaperCache::<u32, TieredBuffer>::new(1000, CacheTierSize::Bytes(0), PaperPolicy::TwoQHybrid(0.5)),
            Err(CacheError::InvalidFastTierSize),
        ));

        let cache = PaperCache::<u32, TieredBuffer>::new(1000, CacheTierSize::Bytes(500), PaperPolicy::TwoQHybrid(0.5))
            .expect("cache should construct");

        assert!(matches!(
            cache.set_fast_tier_size(CacheTierSize::Bytes(2000)),
            Err(CacheError::InvalidFastTierSize),
        ));
    }

    #[test]
    fn invalid_k_in_is_rejected() {
        assert!(matches!(
            PaperCache::<u32, TieredBuffer>::new(1000, CacheTierSize::Bytes(500), PaperPolicy::TwoQHybrid(1.5)),
            Err(CacheError::InvalidPolicy),
        ));

        assert!(matches!(
            PaperCache::<u32, TieredBuffer>::new(1000, CacheTierSize::Bytes(500), PaperPolicy::TwoQHybrid(-0.1)),
            Err(CacheError::InvalidPolicy),
        ));
    }

    #[test]
    fn ttl_is_preserved_across_a_set() {
        let cache = PaperCache::<u32, TieredBuffer>::new(
            1_000_000,
            CacheTierSize::Bytes(1_000_000), PaperPolicy::TwoQHybrid(0.5)).expect("cache should construct");

        cache.set(1u32, b"value", Some(60)).expect("set should succeed");
        assert!(cache.ttl(&1u32, Some(120)).is_ok());
        assert_eq!(cache.get(&1u32).unwrap(), b"value");
    }
}

/// Exercises the real public `PaperCache<K, TieredBuffer>` API end to end
/// for `fifo_hybrid_cache`. See `test_lru_hybrid_cache`'s doc comment for why
/// this deliberately stays on the fast-tier-only path (no PMEM allocation) —
/// the full tier-crossing coverage (including Correction 2's slow-tier
/// overwrite path) lives in `tests/hybrid_cache_integration.rs`.
#[cfg(all(test, feature = "fifo_hybrid_cache"))]
mod test_fifo_hybrid_cache {
	use crate::PaperPolicy;
    use crate::{PaperCache, TieredBuffer, CacheTierSize, Tier, CacheError};

    #[test]
    fn basic_construction_and_fast_tier_only_roundtrip() {
        let cache = PaperCache::<u32, TieredBuffer>::new(1_000_000, CacheTierSize::Bytes(1_000_000), PaperPolicy::FifoHybrid).expect("cache should construct");

        cache.set(1u32, b"hello world", None).expect("set should succeed");
        assert!(cache.has(&1u32));
        assert_eq!(cache.get(&1u32).unwrap(), b"hello world");
        assert_eq!(cache.tier_of(&1u32), Some(Tier::Fast));

        let stats = cache.hybrid_stats();
        assert_eq!(stats.demotions, 0);
        assert_eq!(stats.promotions, 0);
        assert_eq!(stats.evictions, 0);

        assert_eq!(cache.fast_tier_size(), 1_000_000);
        cache.set_fast_tier_size(CacheTierSize::Bytes(500_000)).expect("resize should succeed");
        assert_eq!(cache.fast_tier_size(), 500_000);

        cache.del(&1u32).expect("del should succeed");
        assert!(!cache.has(&1u32));
        assert_eq!(cache.tier_of(&1u32), None);
    }

    #[test]
    fn invalid_fast_tier_size_is_rejected() {
        assert!(matches!(
            PaperCache::<u32, TieredBuffer>::new(1000, CacheTierSize::Bytes(2000), PaperPolicy::FifoHybrid),
            Err(CacheError::InvalidFastTierSize),
        ));

        assert!(matches!(
            PaperCache::<u32, TieredBuffer>::new(1000, CacheTierSize::Bytes(0), PaperPolicy::FifoHybrid),
            Err(CacheError::InvalidFastTierSize),
        ));

        let cache = PaperCache::<u32, TieredBuffer>::new(1000, CacheTierSize::Bytes(500), PaperPolicy::FifoHybrid)
            .expect("cache should construct");

        assert!(matches!(
            cache.set_fast_tier_size(CacheTierSize::Bytes(2000)),
            Err(CacheError::InvalidFastTierSize),
        ));
    }

    #[test]
    fn ttl_is_preserved_across_a_set() {
        let cache = PaperCache::<u32, TieredBuffer>::new(
            1_000_000,
            CacheTierSize::Bytes(1_000_000), PaperPolicy::FifoHybrid).expect("cache should construct");

        cache.set(1u32, b"value", Some(60)).expect("set should succeed");
        assert!(cache.ttl(&1u32, Some(120)).is_ok());
        assert_eq!(cache.get(&1u32).unwrap(), b"value");
    }

    #[test]
    fn overwrite_of_a_still_fast_key_stays_fast_and_keeps_working() {
        // Sanity check for Correction 2's tier-aware `set()`: the common
        // (both-Fast) case must not regress — overwriting a key that's
        // still in the fast tier should stay fast and just work.
        let cache = PaperCache::<u32, TieredBuffer>::new(
            1_000_000,
            CacheTierSize::Bytes(1_000_000), PaperPolicy::FifoHybrid).expect("cache should construct");

        cache.set(1u32, b"hello", None).expect("set should succeed");
        assert_eq!(cache.tier_of(&1u32), Some(Tier::Fast));

        cache.set(1u32, b"hello world", None).expect("overwrite should succeed");
        assert_eq!(cache.tier_of(&1u32), Some(Tier::Fast));
        assert_eq!(cache.get(&1u32).unwrap(), b"hello world");
    }
}

/// Exercises the real public `PaperCache<K, TieredBuffer>` API for
/// `lru_sized_hybrid_cache` end to end. Deliberately stays on the
/// fast-tier-only path (both fast-segment capacities == max_size, tiny
/// values) so no object ever demotes -- see `test_lru_hybrid_cache`'s
/// identical rationale. A full integration test covering demotion/
/// promotion/eviction across both segments and both tiers belongs in
/// `tests/hybrid_cache_integration.rs` and should be run on
/// PMEM-capable hardware.
#[cfg(all(test, feature = "lru_sized_hybrid_cache"))]
mod test_lru_sized_hybrid_cache {
    use crate::{PaperCache, TieredBuffer, CacheTierSize, Tier, CacheError};

    #[test]
    fn basic_construction_and_fast_tier_only_roundtrip() {
        let cache = PaperCache::<u32, TieredBuffer>::new_sized(
            1_000_000,
            CacheTierSize::Bytes(1_000_000), // small segment == whole cache
            CacheTierSize::Bytes(1_000_000), // large segment == whole cache
            CacheTierSize::Bytes(1_000_000), // threshold huge -> everything classifies small
        ).expect("cache should construct");

        cache.set(1u32, b"hello world", None).expect("set should succeed");
        assert!(cache.has(&1u32));
        assert_eq!(cache.get(&1u32).unwrap(), b"hello world");
        assert_eq!(cache.tier_of(&1u32), Some(Tier::Fast));

        let stats = cache.hybrid_stats();
        assert_eq!(stats.demotions, 0);
        assert_eq!(stats.promotions, 0);
        assert_eq!(stats.evictions, 0);

        assert_eq!(cache.fast_tier_size(), 1_000_000);
        cache.set_fast_tier_size(CacheTierSize::Bytes(500_000)).expect("resize should succeed");
        assert_eq!(cache.fast_tier_size(), 500_000);

        assert_eq!(cache.large_fast_tier_size(), 1_000_000);
        cache.set_large_fast_tier_size(CacheTierSize::Bytes(500_000)).expect("resize should succeed");
        assert_eq!(cache.large_fast_tier_size(), 500_000);

        assert_eq!(cache.size_threshold(), 1_000_000);
        cache.set_size_threshold(CacheTierSize::Bytes(4_096)).expect("threshold change should succeed");
        assert_eq!(cache.size_threshold(), 4_096);

        cache.del(&1u32).expect("del should succeed");
        assert!(!cache.has(&1u32));
        assert_eq!(cache.tier_of(&1u32), None);
    }

    #[test]
    fn invalid_fast_tier_size_is_rejected() {
        assert!(matches!(
            PaperCache::<u32, TieredBuffer>::new_sized(
                1000, CacheTierSize::Bytes(2000), CacheTierSize::Bytes(500), CacheTierSize::Bytes(100),
            ),
            Err(CacheError::InvalidFastTierSize),
        ));

        assert!(matches!(
            PaperCache::<u32, TieredBuffer>::new_sized(
                1000, CacheTierSize::Bytes(0), CacheTierSize::Bytes(500), CacheTierSize::Bytes(100),
            ),
            Err(CacheError::InvalidFastTierSize),
        ));

        assert!(matches!(
            PaperCache::<u32, TieredBuffer>::new_sized(
                1000, CacheTierSize::Bytes(500), CacheTierSize::Bytes(2000), CacheTierSize::Bytes(100),
            ),
            Err(CacheError::InvalidFastTierSize),
        ));

        let cache = PaperCache::<u32, TieredBuffer>::new_sized(
            1000, CacheTierSize::Bytes(500), CacheTierSize::Bytes(500), CacheTierSize::Bytes(100),
        ).expect("cache should construct");

        assert!(matches!(
            cache.set_fast_tier_size(CacheTierSize::Bytes(2000)),
            Err(CacheError::InvalidFastTierSize),
        ));

        assert!(matches!(
            cache.set_large_fast_tier_size(CacheTierSize::Bytes(2000)),
            Err(CacheError::InvalidFastTierSize),
        ));
    }

    #[test]
    fn ttl_is_preserved_across_a_set() {
        let cache = PaperCache::<u32, TieredBuffer>::new_sized(
            1_000_000,
            CacheTierSize::Bytes(1_000_000),
            CacheTierSize::Bytes(1_000_000),
            CacheTierSize::Bytes(1_000_000),
        ).expect("cache should construct");

        cache.set(1u32, b"value", Some(60)).expect("set should succeed");
        assert!(cache.ttl(&1u32, Some(120)).is_ok());
        assert_eq!(cache.get(&1u32).unwrap(), b"value");
    }

    #[test]
    fn overwrite_of_a_still_fast_key_stays_fast_and_keeps_working() {
        let cache = PaperCache::<u32, TieredBuffer>::new_sized(
            1_000_000,
            CacheTierSize::Bytes(1_000_000),
            CacheTierSize::Bytes(1_000_000),
            CacheTierSize::Bytes(1_000_000),
        ).expect("cache should construct");

        cache.set(1u32, b"hello", None).expect("set should succeed");
        assert_eq!(cache.tier_of(&1u32), Some(Tier::Fast));

        cache.set(1u32, b"hello world", None).expect("overwrite should succeed");
        assert_eq!(cache.tier_of(&1u32), Some(Tier::Fast));
        assert_eq!(cache.get(&1u32).unwrap(), b"hello world");
    }
}

/// Exercises the real public `PaperCache<K, TieredBuffer>` API for
/// `lru_sized_compact_hybrid_cache` end to end, mirroring
/// `test_lru_sized_hybrid_cache` one for one: the compact stack is a
/// compaction of the size-split one, so the public surface has to be
/// indistinguishable. Deliberately stays on the fast-tier-only path (both
/// fast-segment capacities == max_size, tiny values) so no object ever
/// demotes -- same rationale as the module it mirrors.
#[cfg(all(test, feature = "lru_sized_compact_hybrid_cache"))]
mod test_lru_sized_compact_hybrid_cache {
    use crate::{PaperCache, PaperPolicy, TieredBuffer, CacheTierSize, Tier, CacheError};

    #[test]
    fn basic_construction_and_fast_tier_only_roundtrip() {
        let cache = PaperCache::<u32, TieredBuffer>::new_sized_compact(
            1_000_000,
            CacheTierSize::Bytes(1_000_000), // small segment == whole cache
            CacheTierSize::Bytes(1_000_000), // large segment == whole cache
            CacheTierSize::Bytes(1_000_000), // threshold huge -> everything classifies small
        ).expect("cache should construct");

        cache.set(1u32, b"hello world", None).expect("set should succeed");
        assert!(cache.has(&1u32));
        assert_eq!(cache.get(&1u32).unwrap(), b"hello world");
        assert_eq!(cache.tier_of(&1u32), Some(Tier::Fast));

        let stats = cache.hybrid_stats();
        assert_eq!(stats.demotions, 0);
        assert_eq!(stats.promotions, 0);
        assert_eq!(stats.evictions, 0);

        assert_eq!(cache.fast_tier_size(), 1_000_000);
        cache.set_fast_tier_size(CacheTierSize::Bytes(500_000)).expect("resize should succeed");
        assert_eq!(cache.fast_tier_size(), 500_000);

        assert_eq!(cache.large_fast_tier_size(), 1_000_000);
        cache.set_large_fast_tier_size(CacheTierSize::Bytes(500_000)).expect("resize should succeed");
        assert_eq!(cache.large_fast_tier_size(), 500_000);

        assert_eq!(cache.size_threshold(), 1_000_000);
        cache.set_size_threshold(CacheTierSize::Bytes(4_096)).expect("threshold change should succeed");
        assert_eq!(cache.size_threshold(), 4_096);

        cache.del(&1u32).expect("del should succeed");
        assert!(!cache.has(&1u32));
        assert_eq!(cache.tier_of(&1u32), None);
    }

    /// Both size-split designs need three sizing scalars, so the generic
    /// hybrid constructor must refuse them rather than quietly building one
    /// from a single `CacheTierSize` (and a default second segment it was
    /// never told about). The compact variant was added to that rejection
    /// alongside the baseline; this pins both.
    #[test]
    fn the_generic_hybrid_constructor_rejects_both_size_split_designs() {
        assert!(matches!(
            PaperCache::<u32, TieredBuffer>::new(
                1_000_000, CacheTierSize::Bytes(1_000_000), PaperPolicy::LruSizedCompactHybrid,
            ),
            Err(CacheError::InvalidPolicy),
        ));

        assert!(matches!(
            PaperCache::<u32, TieredBuffer>::new(
                1_000_000, CacheTierSize::Bytes(1_000_000), PaperPolicy::LruSizedHybrid,
            ),
            Err(CacheError::InvalidPolicy),
        ));
    }

    #[test]
    fn invalid_fast_tier_size_is_rejected() {
        assert!(matches!(
            PaperCache::<u32, TieredBuffer>::new_sized_compact(
                1000, CacheTierSize::Bytes(2000), CacheTierSize::Bytes(500), CacheTierSize::Bytes(100),
            ),
            Err(CacheError::InvalidFastTierSize),
        ));

        assert!(matches!(
            PaperCache::<u32, TieredBuffer>::new_sized_compact(
                1000, CacheTierSize::Bytes(0), CacheTierSize::Bytes(500), CacheTierSize::Bytes(100),
            ),
            Err(CacheError::InvalidFastTierSize),
        ));

        assert!(matches!(
            PaperCache::<u32, TieredBuffer>::new_sized_compact(
                1000, CacheTierSize::Bytes(500), CacheTierSize::Bytes(2000), CacheTierSize::Bytes(100),
            ),
            Err(CacheError::InvalidFastTierSize),
        ));

        let cache = PaperCache::<u32, TieredBuffer>::new_sized_compact(
            1000, CacheTierSize::Bytes(500), CacheTierSize::Bytes(500), CacheTierSize::Bytes(100),
        ).expect("cache should construct");

        assert!(matches!(
            cache.set_fast_tier_size(CacheTierSize::Bytes(2000)),
            Err(CacheError::InvalidFastTierSize),
        ));

        assert!(matches!(
            cache.set_large_fast_tier_size(CacheTierSize::Bytes(2000)),
            Err(CacheError::InvalidFastTierSize),
        ));
    }

    #[test]
    fn ttl_is_preserved_across_a_set() {
        let cache = PaperCache::<u32, TieredBuffer>::new_sized_compact(
            1_000_000,
            CacheTierSize::Bytes(1_000_000),
            CacheTierSize::Bytes(1_000_000),
            CacheTierSize::Bytes(1_000_000),
        ).expect("cache should construct");

        cache.set(1u32, b"value", Some(60)).expect("set should succeed");
        assert!(cache.ttl(&1u32, Some(120)).is_ok());
        assert_eq!(cache.get(&1u32).unwrap(), b"value");
    }

    #[test]
    fn overwrite_of_a_still_fast_key_stays_fast_and_keeps_working() {
        let cache = PaperCache::<u32, TieredBuffer>::new_sized_compact(
            1_000_000,
            CacheTierSize::Bytes(1_000_000),
            CacheTierSize::Bytes(1_000_000),
            CacheTierSize::Bytes(1_000_000),
        ).expect("cache should construct");

        cache.set(1u32, b"hello", None).expect("set should succeed");
        assert_eq!(cache.tier_of(&1u32), Some(Tier::Fast));

        cache.set(1u32, b"hello world", None).expect("overwrite should succeed");
        assert_eq!(cache.tier_of(&1u32), Some(Tier::Fast));
        assert_eq!(cache.get(&1u32).unwrap(), b"hello world");
    }
}

#[cfg(test)]
mod s_three_fifo_budget_tests {
	use super::*;

	/// The non-tiered design sizes main at `(1 - ratio) * max_size`, so an
	/// extreme ratio starves it. This is the predicate the non-hybrid
	/// constructor and `resize` gate on; the livelock it prevents is an
	/// eviction loop that can never bring the cache under budget.
	#[test]
	fn a_main_budget_that_truncates_to_zero_is_detected() {
		// (1 - 1.0) * 1_000 == 0 -- the endpoint.
		assert!(s_three_fifo_starves_main(PaperPolicy::SThreeFifo(1.0), 1_000));

		// (1 - 0.9995) * 1_000 == 0.5, truncated to 0 -- inside the open
		// range, and reachable only because `max_size` is small.
		assert!(s_three_fifo_starves_main(PaperPolicy::SThreeFifo(0.9995), 1_000));

		// The same ratio is fine once main gets a byte: 500 here.
		assert!(!s_three_fifo_starves_main(PaperPolicy::SThreeFifo(0.9995), 1_000_000));
	}

	/// A zero-length ONE-ACCESS queue is legal and must not be caught here:
	/// it is what `ratio == 0.0` asks for, and it degrades cleanly because
	/// main then holds the entire budget.
	#[test]
	fn a_zero_length_one_access_queue_is_not_flagged() {
		assert!(!s_three_fifo_starves_main(PaperPolicy::SThreeFifo(0.0), 1_000));
		assert!(!s_three_fifo_starves_main(PaperPolicy::SThreeFifo(0.0005), 1_000));
	}

	/// Only the s3-fifo design derives a budget from `1 - ratio`. 2Q sizes
	/// its FIFO queue at `k_in * max_size` and bounds its main queue by the
	/// cache's overall `max_size`, so `k_in == 1.0` starves nothing there and
	/// must not be rejected.
	#[test]
	fn other_policies_are_never_flagged() {
		assert!(!s_three_fifo_starves_main(PaperPolicy::TwoQ(1.0, 0.0), 1_000));
		assert!(!s_three_fifo_starves_main(PaperPolicy::Lru, 1_000));
		assert!(!s_three_fifo_starves_main(PaperPolicy::Lfu, 1_000));
	}
}

/// Every compact stack must be behaviourally indistinguishable from the
/// baseline it compacts. Four crate-level predicates decide how a policy is
/// treated, and **all four fail OPEN** -- a design missing from them is not a
/// compile error, it is a plausible default:
///
/// | predicate | missing arm yields |
/// |---|---|
/// | `params_ok` | `true` -- ACCEPTS parameters the baseline rejects |
/// | `s3_fifo_queue_budgets` | `None` -- skips the queue-starvation check |
/// | `PaperPolicy::is_hybrid` | `false` -- no fast tier at all |
/// | `Display`/`FromStr` | a policy that cannot be named or parsed |
///
/// Every conversion in this series missed at least one, and none of them
/// failed to build. This module pins each compact twin to its baseline, so
/// the next conversion that forgets a site fails here instead of silently
/// running a different algorithm than the one it is being compared against.
#[cfg(all(test, feature = "hybrid_cache_common"))]
mod compact_parity {
	use crate::{PaperPolicy, params_ok, s3_fifo_queue_budgets};

	/// Deliberately includes both endpoints and beyond: the 2Q family and the
	/// reprieve designs take an INCLUSIVE upper bound while the s3-fifo family
	/// EXCLUDES 1.0 (those size a main queue at `(1 - ratio) * max_size`, so
	/// 1.0 leaves it zero bytes and the eviction loop spins). A twin placed in
	/// the wrong group agrees everywhere except at exactly 1.0.
	const RATIOS: [f64; 10] =
		[-1.0, -0.001, 0.0, 0.001, 0.5, 0.999, 1.0, 1.001, 2.0, f64::NAN];

	/// `promote_k == 0` degenerates to plain LRU and is rejected; the cap is
	/// 16, so values far above it must still round-trip.
	const THRESHOLDS: [u16; 6] = [0, 1, 2, 3, 16, u16::MAX];

	fn pairs() -> Vec<(PaperPolicy, PaperPolicy)> {
		let mut v = Vec::new();

		macro_rules! ratio_pair {
			($b:ident, $c:ident) => {
				for r in RATIOS {
					v.push((PaperPolicy::$b(r), PaperPolicy::$c(r)));
				}
			};
		}
		ratio_pair!(TwoQHybrid, TwoQCompactHybrid);
		ratio_pair!(TwoQFastAdmissionHybrid, TwoQFastAdmissionCompactHybrid);
		ratio_pair!(TwoQFastAdmissionReprieveHybrid, TwoQFastAdmissionReprieveCompactHybrid);
		ratio_pair!(TwoQGhostHybrid, TwoQGhostCompactHybrid);
		ratio_pair!(S3FifoHybrid, S3FifoCompactHybrid);
		ratio_pair!(S3FifoGhostHybrid, S3FifoGhostCompactHybrid);
		ratio_pair!(S3FifoGhostLazyDemotionHybrid, S3FifoGhostLazyDemotionCompactHybrid);
		ratio_pair!(S3FifoGhostLazyDemotionFastAdmissionHybrid, S3FifoGhostLazyDemotionFastAdmissionCompactHybrid);
		ratio_pair!(S3FifoGhostLazyDemotionFastAdmissionMidpointHybrid, S3FifoGhostLazyDemotionFastAdmissionMidpointCompactHybrid);
		ratio_pair!(S3FifoLazyDemotionReprieveHybrid, S3FifoLazyDemotionReprieveCompactHybrid);
		ratio_pair!(S3FifoLazyDemotionFastAdmissionReprieveHybrid, S3FifoLazyDemotionFastAdmissionReprieveCompactHybrid);
		ratio_pair!(S3FifoLazyDemotionFastAdmissionMidpointReprieveHybrid, S3FifoLazyDemotionFastAdmissionMidpointReprieveCompactHybrid);
		ratio_pair!(S3FifoLazyDemotionFastAdmissionSplitSlowReprieveHybrid, S3FifoLazyDemotionFastAdmissionSplitSlowReprieveCompactHybrid);

		// The one two-ratio design: BOTH parameters must be in range, so it
		// cannot join the single-ratio group.
		for a in RATIOS {
			for b in RATIOS {
				v.push((
					PaperPolicy::TwoQFullFastAdmissionHybrid(a, b),
					PaperPolicy::TwoQFullFastAdmissionCompactHybrid(a, b),
				));
			}
		}

		for k in THRESHOLDS {
			v.push((PaperPolicy::LruLfuHybrid(k), PaperPolicy::LruLfuCompactHybrid(k)));
		}

		v.push((PaperPolicy::LruHybrid, PaperPolicy::LruCompactHybrid));
		v.push((PaperPolicy::LfuHybrid, PaperPolicy::LfuCompactHybrid));
		v.push((PaperPolicy::FifoHybrid, PaperPolicy::FifoCompactHybrid));
		v.push((PaperPolicy::LruSizedHybrid, PaperPolicy::LruSizedCompactHybrid));

		v
	}

	/// The table is only exhaustive if it names every design. Anchor it to the
	/// compiler-checked variant list rather than to a number typed by hand.
	#[test]
	fn the_pair_table_covers_every_hybrid_design() {
		use std::collections::HashSet;
		use std::mem::discriminant;

		let mut designs = HashSet::new();
		for (base, compact) in pairs() {
			designs.insert(discriminant(&base));
			designs.insert(discriminant(&compact));
		}

		assert_eq!(
			designs.len(),
			38,
			"the pair table names {} enum variants, expected 38 (19 baselines + \
			 19 compact twins). A design missing here is a design whose twin is \
			 never checked against it at all.",
			designs.len(),
		);
	}

	/// `params_ok` ends in `_ => true`. A twin missing from it, or placed in a
	/// group with the wrong bound, ACCEPTS a configuration its baseline
	/// rejects -- and the two then run at different capacities while being
	/// reported as the same experiment.
	#[test]
	fn every_compact_twin_agrees_with_its_baseline_on_params_ok() {
		for (base, compact) in pairs() {
			assert_eq!(
				params_ok(base),
				params_ok(compact),
				"`{base}` and `{compact}` disagree on parameter validity: \
				 baseline={}, compact={}. Either the twin is missing from \
				 `params_ok` (and fell to `_ => true`), or it was put in a \
				 group whose upper bound differs from its baseline's.",
				params_ok(base),
				params_ok(compact),
			);
		}
	}

	/// `s3_fifo_queue_budgets` returns `None` for an unlisted policy, which
	/// skips the zero-capacity queue check entirely rather than failing.
	#[test]
	fn every_compact_twin_agrees_with_its_baseline_on_queue_budgets() {
		for (base, compact) in pairs() {
			let (b, c) = (s3_fifo_queue_budgets(base), s3_fifo_queue_budgets(compact));
			match (b, c) {
				(None, None) => {},
				(Some((br, bmain)), Some((cr, cmain))) => assert!(
					(br == cr || (br.is_nan() && cr.is_nan())) && bmain == cmain,
					"`{base}` and `{compact}` disagree on queue budgets: {b:?} vs {c:?}",
				),
				_ => panic!(
					"`{base}` and `{compact}` disagree on queue budgets: {b:?} vs {c:?} -- \
					 an unlisted policy yields `None`, silently skipping the \
					 queue-starvation check the baseline gets",
				),
			}
		}
	}

	/// `is_hybrid` is a hand-written `matches!` and cannot fire a compile
	/// error. A twin missing from it gets no fast tier at all.
	#[test]
	fn every_compact_twin_agrees_with_its_baseline_on_is_hybrid() {
		for (base, compact) in pairs() {
			assert!(base.is_hybrid(), "`{base}` is not reported as hybrid");
			assert!(
				compact.is_hybrid(),
				"`{compact}` is not reported as hybrid but its baseline `{base}` is: \
				 missing from the `is_hybrid` `matches!`, so it would run with no fast tier",
			);
		}
	}

	/// A twin whose `FromStr` prefix guard is ordered after a baseline it is a
	/// superstring of parses as the WRONG policy rather than failing.
	#[test]
	fn every_compact_twin_round_trips_through_display_and_from_str() {
		for (base, compact) in pairs() {
			let (bt, ct) = (format!("{base}"), format!("{compact}"));
			let (bp, cp) = (bt.parse::<PaperPolicy>(), ct.parse::<PaperPolicy>());

			// Parity first: the twin must accept exactly what its baseline
			// accepts. A NaN ratio renders but is not a parseable policy, and
			// that has to be true of BOTH or the pair is not interchangeable.
			assert_eq!(
				bp.is_ok(),
				cp.is_ok(),
				"`{bt}` parses={} but `{ct}` parses={} -- the twin disagrees with \
				 its baseline about what is a valid policy string",
				bp.is_ok(),
				cp.is_ok(),
			);

			// Where the baseline round-trips, the twin must too -- and must not
			// be captured by another design\'s prefix guard on the way back.
			if let (Ok(bv), Ok(cv)) = (bp, cp) {
				assert_eq!(
					format!("{bv}"), bt,
					"`{bt}` round-trips to `{bv}` -- a prefix guard matched the wrong design",
				);
				assert_eq!(
					format!("{cv}"), ct,
					"`{ct}` round-trips to `{cv}` -- a prefix guard matched the wrong design. \
					 Compact policy strings are SUPERSTRINGS of their baselines, so an \
					 arm ordered after the baseline\'s captures them.",
				);
			}
		}
	}
}


// ---------------------------------------------------------------------------
// Sampled step profiler for `get_into` (diagnostic; GETINTO_PROFILE=1).
//
// Exists to ATTRIBUTE a measured per-GET latency contrast to a specific step
// of the read path -- hash, map lookup + Arc clone, copy, event send -- after
// a 199 ns p50 difference between the all-DRAM and hybrid builds survived the
// removal of every allocation from the measured region (2026-08-28).
// ---------------------------------------------------------------------------

static GI_N: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static GI_HASH: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static GI_LOOKUP: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static GI_COPY: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static GI_BCAST: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

thread_local! {
	static GI_TICK: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
}

/// Sampled per-hit tuples: [payload_len, hash_ns, lookup_ns, copy_ns, bcast_ns, served_from_fast].
/// Bounded at 2^20 entries; reported as per-step percentiles at Drop.
static GI_SAMPLES: std::sync::Mutex<Vec<[u64; 6]>> = std::sync::Mutex::new(Vec::new());

fn gi_prof_enabled() -> bool {
	static FLAG: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
	*FLAG.get_or_init(|| std::env::var("GETINTO_PROFILE").map(|v| v == "1").unwrap_or(false))
}
