/*
 * Copyright (c) Kia Shakiba
 *
 * This source code is licensed under the GNU AGPLv3 license found in the
 * LICENSE file in the root directory of this source tree.
 * correct
 */

#![cfg_attr(any(feature = "hashbrown_dram", feature = "all_dram", feature = "key_value_pmem", feature = "global_hashtable_pmem", feature = "tiering_hashtable_pmem", feature = "eviction_stacks_pmem", feature = "jemalloc_cxl_slow_tier"), feature(allocator_api), feature(clone_from_ref))]


// `lru_hybrid_cache`, `lfu_hybrid_cache`, `two_q_hybrid_cache`, and
// `fifo_hybrid_cache` each define their own inherent methods (`new`, `get`,
// `set`, ...) on `PaperCache<K, TieredBuffer, S>`. Two such impl blocks for
// the identical concrete type cannot coexist, so only one of these
// hybrid-cache flavors may be enabled at a time.
#[cfg(all(feature = "lru_hybrid_cache", feature = "lfu_hybrid_cache"))]
compile_error!("Cannot enable both 'lru_hybrid_cache' and 'lfu_hybrid_cache' features simultaneously. Both define their own PaperCache<K, TieredBuffer, S> impl block; choose only one hybrid-cache flavor.");

#[cfg(all(feature = "lru_hybrid_cache", feature = "two_q_hybrid_cache"))]
compile_error!("Cannot enable both 'lru_hybrid_cache' and 'two_q_hybrid_cache' features simultaneously. Both define their own PaperCache<K, TieredBuffer, S> impl block; choose only one hybrid-cache flavor.");

#[cfg(all(feature = "lfu_hybrid_cache", feature = "two_q_hybrid_cache"))]
compile_error!("Cannot enable both 'lfu_hybrid_cache' and 'two_q_hybrid_cache' features simultaneously. Both define their own PaperCache<K, TieredBuffer, S> impl block; choose only one hybrid-cache flavor.");

#[cfg(all(feature = "fifo_hybrid_cache", feature = "lru_hybrid_cache"))]
compile_error!("Cannot enable both 'fifo_hybrid_cache' and 'lru_hybrid_cache' features simultaneously. Both define their own PaperCache<K, TieredBuffer, S> impl block; choose only one hybrid-cache flavor.");

#[cfg(all(feature = "fifo_hybrid_cache", feature = "lfu_hybrid_cache"))]
compile_error!("Cannot enable both 'fifo_hybrid_cache' and 'lfu_hybrid_cache' features simultaneously. Both define their own PaperCache<K, TieredBuffer, S> impl block; choose only one hybrid-cache flavor.");

#[cfg(all(feature = "fifo_hybrid_cache", feature = "two_q_hybrid_cache"))]
compile_error!("Cannot enable both 'fifo_hybrid_cache' and 'two_q_hybrid_cache' features simultaneously. Both define their own PaperCache<K, TieredBuffer, S> impl block; choose only one hybrid-cache flavor.");

#[cfg(all(feature = "lru_sized_hybrid_cache", feature = "lru_hybrid_cache"))]
compile_error!("Cannot enable both 'lru_sized_hybrid_cache' and 'lru_hybrid_cache' features simultaneously. Both define their own PaperCache<K, TieredBuffer, S> impl block; choose only one hybrid-cache flavor.");

#[cfg(all(feature = "lru_sized_hybrid_cache", feature = "lfu_hybrid_cache"))]
compile_error!("Cannot enable both 'lru_sized_hybrid_cache' and 'lfu_hybrid_cache' features simultaneously. Both define their own PaperCache<K, TieredBuffer, S> impl block; choose only one hybrid-cache flavor.");

#[cfg(all(feature = "lru_sized_hybrid_cache", feature = "two_q_hybrid_cache"))]
compile_error!("Cannot enable both 'lru_sized_hybrid_cache' and 'two_q_hybrid_cache' features simultaneously. Both define their own PaperCache<K, TieredBuffer, S> impl block; choose only one hybrid-cache flavor.");

#[cfg(all(feature = "lru_sized_hybrid_cache", feature = "fifo_hybrid_cache"))]
compile_error!("Cannot enable both 'lru_sized_hybrid_cache' and 'fifo_hybrid_cache' features simultaneously. Both define their own PaperCache<K, TieredBuffer, S> impl block; choose only one hybrid-cache flavor.");

// Validate that hashbrown_dram is not enabled with other global hashtable features
#[cfg(all(feature = "hashbrown_dram", feature = "global_hashtable_pmem"))]
compile_error!("Cannot enable both 'hashbrown_dram' and 'global_hashtable_pmem' features simultaneously. Please choose only one global hashtable mode.");

#[cfg(all(feature = "jemalloc_cxl_slow_tier", feature = "tikv_jemalloc_global"))]
compile_error!("Cannot enable both 'jemalloc_cxl_slow_tier' and 'tikv_jemalloc_global' simultaneously -- both install a #[global_allocator], and only one static GLOBAL can be declared.");

#[cfg(feature = "tikv_jemalloc_global")]
use tikv_jemallocator::Jemalloc;


#[cfg(any(feature = "hashbrown_dram", feature = "key_value_pmem", feature = "global_hashtable_pmem", feature = "tiering_hashtable_pmem", feature = "eviction_stacks_pmem", feature = "all_dram", feature = "jemalloc_cxl_slow_tier"))]
pub mod allocator;

use std::arch::x86_64::{_mm_clflush, _mm_sfence};

// `Hybrid` is the crate-wide PMEM allocator alias: every PMEM feature routes
// through UMF's TBB-backed pool (`HybridObjects`, see `allocator.rs`).
#[cfg(any(
    feature = "key_value_pmem",
    feature = "key_pmem_value_pmem",
    feature = "global_hashtable_pmem",
    feature = "tiering_hashtable_pmem",
    feature = "eviction_stacks_pmem",
))]
pub(crate) use crate::allocator::HybridObjects as Hybrid;

// UMF bindings are always needed when any PMEM feature is active.
// The build script guarantees that the UMF C symbols are always present:
// either the real UMF library (when wrapper.h exists) or the stub
// implementation (umf_stub.c, using malloc/free) when UMF is unavailable.
#[cfg(any(feature = "key_value_pmem", feature = "global_hashtable_pmem", feature = "tiering_hashtable_pmem", feature = "eviction_stacks_pmem"))]
mod allocator_bindings {
    include!("umf_allocator_bindings.rs"); // UMF extern "C" declarations
}

#[cfg(feature = "key_value_pmem")]
impl typesize::TypeSize for BufferPMEM {
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
#[cfg(any(feature = "lru_hybrid_cache", feature = "lfu_hybrid_cache", feature = "two_q_hybrid_cache", feature = "fifo_hybrid_cache", feature = "lru_sized_hybrid_cache"))]
mod size;

#[cfg(any(feature = "lru_hybrid_cache", feature = "lfu_hybrid_cache", feature = "two_q_hybrid_cache", feature = "fifo_hybrid_cache", feature = "lru_sized_hybrid_cache"))]
pub use crate::size::CacheTierSize;

// Shared value type for the segmented hybrid-cache features. `lru_hybrid_cache`,
// `lfu_hybrid_cache`, `two_q_hybrid_cache`, and `fifo_hybrid_cache` are
// mutually exclusive (see the `compile_error!` guards above) and all
// re-export it from their own module for source compatibility
// (`paper_cache::TieredBuffer` works either way).
#[cfg(any(feature = "lru_hybrid_cache", feature = "lfu_hybrid_cache", feature = "two_q_hybrid_cache", feature = "fifo_hybrid_cache", feature = "lru_sized_hybrid_cache"))]
mod tiered_buffer;

#[cfg(any(feature = "lru_hybrid_cache", feature = "lfu_hybrid_cache", feature = "two_q_hybrid_cache", feature = "fifo_hybrid_cache", feature = "lru_sized_hybrid_cache"))]
pub use crate::tiered_buffer::TieredBuffer;

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

// Re-exported so `PaperCache::tier_of`'s return type is nameable by callers
// without reaching into the private `worker` module tree directly.
#[cfg(any(feature = "lru_hybrid_cache", feature = "lfu_hybrid_cache", feature = "two_q_hybrid_cache", feature = "fifo_hybrid_cache", feature = "lru_sized_hybrid_cache"))]
pub use crate::worker::Tier;

// Trait abstracting the behavior that differs between the four hybrid-cache
// designs, plus the compile-time selection of exactly one concrete
// implementation -- see `hybrid_policy.rs`'s module doc for why this stays
// a compile-time (not runtime) dispatch.
#[cfg(any(feature = "lru_hybrid_cache", feature = "lfu_hybrid_cache", feature = "two_q_hybrid_cache", feature = "fifo_hybrid_cache", feature = "lru_sized_hybrid_cache"))]
mod hybrid_policy;
#[cfg(any(feature = "lru_hybrid_cache", feature = "lfu_hybrid_cache", feature = "two_q_hybrid_cache", feature = "fifo_hybrid_cache", feature = "lru_sized_hybrid_cache"))]
use crate::hybrid_policy::HybridPolicy;

#[cfg(feature = "lru_hybrid_cache")]
type ActiveHybridPolicy = crate::lru_hybrid_cache::LruHybridPolicy;
#[cfg(feature = "lfu_hybrid_cache")]
type ActiveHybridPolicy = crate::lfu_hybrid_cache::LfuHybridPolicy;
#[cfg(feature = "two_q_hybrid_cache")]
type ActiveHybridPolicy = crate::two_q_hybrid_cache::TwoQHybridPolicy;
#[cfg(feature = "fifo_hybrid_cache")]
type ActiveHybridPolicy = crate::fifo_hybrid_cache::FifoHybridPolicy;
#[cfg(feature = "lru_sized_hybrid_cache")]
type ActiveHybridPolicy = crate::lru_sized_hybrid_cache::LruSizedHybridPolicy;

use std::{
	thread,
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
use crossbeam_channel::unbounded;
use log::{info, error};


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
		Worker,
		WorkerSender,
		WorkerEvent,
		WorkerManager,
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


//#[cfg(feature = "all_dram")]
//#[global_allocator]
//static GLOBAL: allocator::HybridObjects = allocator::HybridObjects;

//pub mod allocator;

// DRAMObjects (NUMA node 0, src/allocator.rs) is the crate's global
// allocator for every feature, including the four hybrid-cache features
// (lru/lfu/two_q/fifo_hybrid_cache) -- an ordinary heap allocation on this
// global allocator IS the fast tier for those features (see
// tiered_buffer.rs's TieredBuffer::Fast = Box<[u8]>); the slow tier uses
// `Hybrid`/`HybridObjects` (NUMA node 1) explicitly via TieredBuffer::Slow.
// A prior session tried routing both tiers through a separate,
// runtime-parameterized `tier_allocator` crate instead -- removed again
// (see tiered_buffer.rs's module doc) since its default backend still
// constructed one independent UMF/TBB pool per NUMA node, the same shape
// DRAMObjects+HybridObjects already had; it added a second implementation
// of the same mechanism without changing that mechanism's properties.
//
// A later, separate `jemalloc_cxl_slow_tier` feature swaps this for
// `allocator::DramMultiArenaObjects` (a pool of node-0-pinned jemalloc
// arenas via jemalloc_cxl's custom extent hooks) paired with
// `SlowTierJemallocAllocator` for the slow tier -- one jemalloc instance for
// both tiers, UMF/Hybrid unused. This was removed once already (never
// proven stable under real concurrent load in three separate retests -- see
// the UMF-jemalloc-pool and jemalloc_cxl_slow_tier retest history in
// `CLAUDE.md`) and has been brought back on request as a standalone,
// available-but-not-the-default mechanism: `DRAMObjects`/`HybridObjects`
// (TBB/UMF) remains the default allocator pairing for every feature, and
// `jemalloc_cxl_slow_tier` stays opt-in only -- do not re-enable it as the
// default, and do not trust it under real concurrent load without
// re-running the actual benchmark (not just this crate's own test suite)
// to confirm, per that retest history.
#[cfg(not(any(feature = "jemalloc_cxl_slow_tier", feature = "tikv_jemalloc_global")))]
#[global_allocator]
static GLOBAL: allocator::DRAMObjects = allocator::DRAMObjects;

#[cfg(feature = "jemalloc_cxl_slow_tier")]
#[global_allocator]
static GLOBAL: allocator::DramMultiArenaObjects = allocator::DramMultiArenaObjects;

// Swaps ONLY the fast-tier/general-purpose global allocator (DRAMObjects,
// NUMA node 0, UMF/TBB) for the plain, off-the-shelf tikv-jemallocator
// binding -- unlike `jemalloc_cxl_slow_tier` (a custom NUMA-extent-hooks
// jemalloc arena) or the UMF jemalloc *pool* backend (umfJemallocPoolOps,
// separately retested three times and found unsafe under real concurrent
// load -- see CLAUDE.md), this is plain jemalloc via its own well-tested
// Rust bindings, with no UMF or custom-extent-hooks machinery involved at
// all. No NUMA pinning is applied explicitly; since every CPU on this
// machine's topology is on node 0 (node 1 is memory-only), the kernel's
// default local-allocation policy still lands these pages on node 0 in
// practice. The slow tier is untouched -- `Hybrid`/`HybridObjects`
// (UMF/TBB, NUMA node 1) still backs `TieredBuffer::Slow` regardless of
// which allocator this static resolves to.
#[cfg(feature = "tikv_jemalloc_global")]
#[global_allocator]
static GLOBAL: Jemalloc = Jemalloc;

#[cfg(not(feature = "all_dram"))]
use std::alloc::{Layout, Allocator}; // Essential imports


//#[cfg(feature = "all_dram")]
pub type BufferDRAM = Box<[u8]>;


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

	worker_manager: Arc<WorkerSender>,
	/// Join handles for every background thread spawned on this cache's
	/// behalf (`WorkerManager` itself, plus its `PolicyWorker`/`TtlWorker`/
	/// `TieringWorker` sub-workers -- `PolicyWorker`'s own `TraceWorker`
	/// child is joined internally by `PolicyWorker` itself, see its
	/// `Shutdown` handling, so it never appears here). `Drop` sends
	/// `WorkerEvent::Shutdown` through `worker_manager` and then joins
	/// these, so that by the time a `PaperCache` value has finished
	/// dropping, none of its background threads are still running --
	/// closing the real race this fixes: before this existed, no worker
	/// thread was ever joined at all, so a `PaperCache` being dropped (or a
	/// process exiting without explicitly dropping one) could leave a
	/// `PolicyWorker` thread genuinely still executing, mid-allocation,
	/// concurrently with the global allocator's own process-exit teardown
	/// -- confirmed directly via a real SIGSEGV inside a UMF/TBB pool's
	/// `tbb_malloc`, racing that pool's own `umfTearDown` destructor.
	worker_handles: WorkerHandles,
	overhead_manager: OverheadManagerRef,

	#[cfg(all(feature = "key_value_pmem", feature = "enable_tiering_manager"))]
	tiering_manager: Arc<TieringManager<K, V>>,

	hasher: S,
}

impl<K, V, S> Drop for PaperCache<K, V, S> {
	fn drop(&mut self) {
		// Best-effort: if every other clone of `worker_manager` (there are
		// none today -- this field is never cloned out of `PaperCache`
		// itself -- but the type is `Arc` defensively) is already gone, or
		// `WorkerManager`'s thread already exited on its own for some other
		// reason, the channel may already be disconnected; `send` returning
		// `Err` here just means there's nothing left to signal, not a bug.
		let _ = self.worker_manager.send(WorkerEvent::Shutdown);

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
	K: 'static + Eq + Hash + TypeSize + std::fmt::Debug + Clone + Send + Sync,
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
	/// use paper_cache::{PaperCache, PaperPolicy};
	///
	/// let cache = PaperCache::<u32, u32>::new(
	///     1000,
	///     &[PaperPolicy::Lfu],
	///     PaperPolicy::Lfu,
	/// );
	///
	/// assert!(cache.is_ok());
	///
	/// // Supplying a maximum size of zero will return a `CacheError`.
	/// let cache = PaperCache::<u32, u32>::new(
	///     0,
	///     &[PaperPolicy::Lfu],
	///     PaperPolicy::Lfu,
	/// );
	///
	/// assert!(cache.is_err());
	///
	/// // Supplying duplicate policies will return a `CacheError`.
	/// let cache = PaperCache::<u32, u32>::new(
	///     1000,
	///     &[PaperPolicy::Lfu, PaperPolicy::Lru, PaperPolicy::Lfu],
	///     PaperPolicy::Lfu,
	/// );
	///
	/// assert!(cache.is_err());
	///
	/// // Supplying a non-configured policy will return a `CacheError`.
	/// let cache = PaperCache::<u32, u32>::new(
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
	/// use paper_cache::{PaperCache, PaperPolicy};
	///
	/// let cache = PaperCache::<u32, u32>::with_hasher(
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

		let (worker_sender, worker_listener) = unbounded();

		#[cfg(all(feature = "key_value_pmem", feature = "enable_tiering_manager"))]
		let (mut worker_manager, mut worker_handles) = WorkerManager::new(
			worker_listener,
			&objects,
			&status,
			&overhead_manager,
			&tiering_manager,
		)?;

		#[cfg(not(all(feature = "key_value_pmem", feature = "enable_tiering_manager")))]
		let (mut worker_manager, mut worker_handles) = WorkerManager::new(
			worker_listener,
			&objects,
			&status,
			&overhead_manager,
		)?;

		worker_handles.push(thread::spawn(move || worker_manager.run()));

		let cache = PaperCache {
			objects,
			status,

			worker_manager: Arc::new(worker_sender),
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
	/// use paper_cache::{PaperCache, PaperPolicy};
	///
	/// let mut cache = PaperCache::<u32, u32>::new(
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
	/// use paper_cache::{PaperCache, PaperPolicy};
	///
	/// let mut cache = PaperCache::<u32, u32>::new(
	///     1000,
	///     &[PaperPolicy::Lfu],
	///     PaperPolicy::Lfu,
	/// ).unwrap();
	///
	/// cache.set(0, 0, None);
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
	/// use paper_cache::{PaperCache, PaperPolicy};
	///
	/// let mut cache = PaperCache::<u32, u32>::new(
	///     1000,
	///     &[PaperPolicy::Lfu],
	///     PaperPolicy::Lfu,
	/// ).unwrap();
	///
	/// cache.set(0, 0, None);
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

		let result = match self.objects.get_ref(&hashed_key) {
			Some(object) if object.key_matches(key) && !object.is_expired() => {
				self.status.incr_hits();
				let arc_val = object.data();
				Ok(AsRef::<[u8]>::as_ref(&*arc_val).to_vec())
			},

			_ => {
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
	///
	/// If the key already exists in the cache, the associated value is updated
	/// to the supplied value.
	///
	/// # Examples
	/// ```
	/// use paper_cache::{PaperCache, PaperPolicy};
	///
	/// let mut cache = PaperCache::<u32, u32>::new(
	///     1000,
	///     &[PaperPolicy::Lfu],
	///     PaperPolicy::Lfu,
	/// ).unwrap();
	///
	/// assert!(cache.set(0, 0, None).is_ok());
	/// ```
	pub fn set(&self, key: K, value: &[u8], ttl: Option<u32>) -> Result<(), CacheError> {
		let hashed_key = self.hash_key(&key);

		let val_buf: V = V::from_bytes(value);
		let object = Object::new(key, val_buf, ttl);
		let base_size = self.overhead_manager.base_size(&object);
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
		self.broadcast(WorkerEvent::Set(hashed_key, base_size, expiry, old_object_info))?;

		Ok(())
	}

	/// Deletes the object associated with the supplied key in the cache.
	/// Returns a [`CacheError`] if the key was not found in the cache.
	///
	/// # Examples
	/// ```
	/// use paper_cache::{PaperCache, PaperPolicy};
	///
	/// let mut cache = PaperCache::<u32, u32>::new(
	///     1000,
	///     &[PaperPolicy::Lfu],
	///     PaperPolicy::Lfu,
	/// ).unwrap();
	///
	/// cache.set(0, 0, None);
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
	/// use paper_cache::{PaperCache, PaperPolicy};
	///
	/// let mut cache = PaperCache::<u32, u32>::new(
	///     1000,
	///     &[PaperPolicy::Lfu],
	///     PaperPolicy::Lfu,
	/// ).unwrap();
	///
	/// cache.set(0, 0, None);
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
	/// use paper_cache::{PaperCache, PaperPolicy};
	///
	/// let mut cache = PaperCache::<u32, u32>::new(
	///     1000,
	///     &[PaperPolicy::Lfu],
	///     PaperPolicy::Lfu,
	/// ).unwrap();
	///
	/// cache.set(0, 0, None);
	/// cache.set(1, 0, None);
	///
	/// // Peeking a key which exists in the cache will return the associated value.
	/// assert!(cache.peek(&0).is_ok());
	/// // Peeking a key which does not exist in the cache will return a CacheError.
	/// assert!(cache.peek(&2).is_err());
	///
	/// cache.set(2, 0, None);
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
	/// use paper_cache::{PaperCache, PaperPolicy};
	///
	/// let mut cache = PaperCache::<u32, u32>::new(
	///     1000,
	///     &[PaperPolicy::Lfu],
	///     PaperPolicy::Lfu,
	/// ).unwrap();
	///
	/// cache.set(0, 0, None); // value will not expire
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
	/// use paper_cache::{PaperCache, PaperPolicy};
	///
	/// let mut cache = PaperCache::<u32, u32>::new(
	///     1000,
	///     &[PaperPolicy::Lfu],
	///     PaperPolicy::Lfu,
	/// ).unwrap();
	///
	/// cache.set(0, 0, None);
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
	/// use paper_cache::{PaperCache, PaperPolicy};
	///
	/// let mut cache = PaperCache::<u32, u32>::new(
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
	/// use paper_cache::{PaperCache, PaperPolicy};
	///
	/// let mut cache = PaperCache::<u32, u32>::new(
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
	/// use paper_cache::{PaperCache, PaperPolicy};
	///
	/// let mut cache = PaperCache::<u32, u32>::new(
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
		if let Err(err) = self.worker_manager.try_send(event) {
			error!("Could not communicate with workers: {err:?}");
			return Err(CacheError::Internal);
		}

		Ok(())
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
	K: 'static + Eq + Hash + TypeSize + std::fmt::Debug + Clone,
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

		// Global hashtable in PMEM (Hybrid allocator) when
		// `global_hashtable_pmem` is on; otherwise a plain-DRAM hashbrown
		// table (`hashbrown_dram`'s default allocator).
		#[cfg(feature = "global_hashtable_pmem")]
		let objects = Arc::new(RwLock::new(HashMap::with_hasher_in(
			NoHasher::default(),
			Hybrid,
		)));

		#[cfg(not(feature = "global_hashtable_pmem"))]
		let objects = Arc::new(RwLock::new(HashMap::with_hasher(
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

		let (worker_sender, worker_listener) = unbounded();

		#[cfg(all(feature = "key_value_pmem", feature = "enable_tiering_manager"))]
		let (mut worker_manager, mut worker_handles) = WorkerManager::new(
			worker_listener,
			&objects,
			&status,
			&overhead_manager,
			&tiering_manager,
		)?;

		#[cfg(not(all(feature = "key_value_pmem", feature = "enable_tiering_manager")))]
		let (mut worker_manager, mut worker_handles) = WorkerManager::new(
			worker_listener,
			&objects,
			&status,
			&overhead_manager,
		)?;

		worker_handles.push(thread::spawn(move || worker_manager.run()));

		let cache = PaperCache {
			objects,
			status,
			worker_manager: Arc::new(worker_sender),
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

		let result = match self.objects.get_ref(&hashed_key) {
			Some(object) if object.key_matches(key) && !object.is_expired() => {
				self.status.incr_hits();
				let arc_val = object.data();
				Ok(AsRef::<[u8]>::as_ref(&*arc_val).to_vec())
			},

			_ => {
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
		self.broadcast(WorkerEvent::Set(hashed_key, base_size, expiry, old_object_info))?;

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
		if let Err(err) = self.worker_manager.try_send(event) {
			error!("Could not communicate with workers: {err:?}");
			return Err(CacheError::Internal);
		}

		Ok(())
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
#[cfg(any(feature = "lru_hybrid_cache", feature = "lfu_hybrid_cache", feature = "two_q_hybrid_cache", feature = "fifo_hybrid_cache", feature = "lru_sized_hybrid_cache"))]
fn new_hybrid_object_map<K, V>() -> ObjectMapRef<K, V> {
	#[cfg(feature = "hashbrown_dram")]
	{
		Arc::new(RwLock::new(HashMap::with_hasher(NoHasher::default())))
	}

	#[cfg(not(feature = "hashbrown_dram"))]
	{
		Arc::new(DashMap::with_hasher(NoHasher::default()))
	}
}

/// Shared engine behind every hybrid-cache feature's own inherent methods
/// (`new`/`with_hasher`/`get`/`set`/.../the `{name}_hybrid_stats()`
/// accessor). Confirmed via direct diff that the four original per-feature
/// impl blocks this replaces differed only in: which `PaperPolicy` variant
/// gets seeded; the `Stats` type and its accessor method's name; one
/// admission-rule branch inside `set()`; and `two_q_hybrid_cache`'s extra
/// `k_in: f64` constructor parameter -- all captured by `ActiveHybridPolicy`
/// (see `hybrid_policy.rs`). The four small per-feature impl blocks below
/// this one exist only to preserve each feature's distinct public
/// constructor/stats-accessor names and signatures for source
/// compatibility with existing callers (`paper-server`/`paper-benchmark-cxl`).
#[cfg(any(feature = "lru_hybrid_cache", feature = "lfu_hybrid_cache", feature = "two_q_hybrid_cache", feature = "fifo_hybrid_cache", feature = "lru_sized_hybrid_cache"))]
impl<K, S> PaperCache<K, TieredBuffer, S>
where
	K: 'static + Eq + Hash + TypeSize + std::fmt::Debug + Clone + Send + Sync,
	S: Default + Clone + BuildHasher,
{
	// `lru_sized_hybrid_cache` doesn't call this: it needs three sizing
	// scalars (two independent fast-segment capacities + a threshold)
	// threaded to three different places rather than this method's single
	// `CacheTierSize`, so it has its own bespoke `new_sized_hybrid` instead
	// (see that feature's own small impl block below). Gated narrower than
	// the outer block so an `lru_sized_hybrid_cache`-only build doesn't
	// compile (and warn about) an unused method.
	#[cfg(any(feature = "lru_hybrid_cache", feature = "lfu_hybrid_cache", feature = "two_q_hybrid_cache", feature = "fifo_hybrid_cache"))]
	fn new_hybrid(
		max_size: CacheSize,
		fast_tier_size: CacheTierSize,
		extra: <ActiveHybridPolicy as HybridPolicy>::ExtraConfig,
		hasher: S,
	) -> Result<Self, CacheError> {
		if max_size == 0 {
			return Err(CacheError::ZeroCacheSize);
		}

		let fast_capacity = fast_tier_size.to_bytes();

		if fast_capacity == 0 || fast_capacity > max_size {
			return Err(CacheError::InvalidFastTierSize);
		}

		let policy = ActiveHybridPolicy::seed_policy(extra);
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
		let migrate: Box<dyn Fn(&TieredBuffer, Tier) -> TieredBuffer + Send + Sync> =
			Box::new(|buffer, tier| match tier {
				Tier::Fast => TieredBuffer::new_fast(buffer.as_ref()),
				Tier::Slow => TieredBuffer::new_slow(buffer.as_ref()),
			});

		let (worker_sender, worker_listener) = unbounded();

		let (mut worker_manager, mut worker_handles) = WorkerManager::new_with_tier_migration(
			worker_listener,
			&objects,
			&status,
			&overhead_manager,
			migrate,
		)?;

		worker_handles.push(thread::spawn(move || worker_manager.run()));

		let cache = PaperCache {
			objects,
			status,
			worker_manager: Arc::new(worker_sender),
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
	pub fn get(&self, key: &K) -> Result<Vec<u8>, CacheError> {
		let hashed_key = self.hash_key(key);

		let result = match self.objects.get_ref(&hashed_key) {
			Some(object) if object.key_matches(key) && !object.is_expired() => {
				self.status.incr_hits();
				let arc_val = object.data();
				let bytes: &[u8] = arc_val.as_ref().as_ref();
				Ok(bytes.to_vec())
			},

			_ => {
				self.status.incr_misses();
				Err(CacheError::KeyNotFound)
			},
		};

		self.broadcast(WorkerEvent::Get(hashed_key, result.is_ok()))?;

		result
	}

	/// Sets the supplied key and value in the cache. Which tier the value's
	/// bytes are built in is decided by `ActiveHybridPolicy::admission_tier`
	/// -- see each hybrid-cache feature's marker-type doc comment (in its
	/// own `mod.rs`) for its specific admission rule. Returns a
	/// [`CacheError`] if the value size is zero or larger than the cache's
	/// maximum size.
	pub fn set(&self, key: K, value: &[u8], ttl: Option<u32>) -> Result<(), CacheError> {
		let hashed_key = self.hash_key(&key);

		let tier = ActiveHybridPolicy::admission_tier(hashed_key, &self.status, &self.objects);
		let val_buf = match tier {
			Tier::Fast => TieredBuffer::new_fast(value),
			Tier::Slow => TieredBuffer::new_slow(value),
		};

		let object = Object::new(key, val_buf, ttl);
		let base_size = self.overhead_manager.base_size(&object);
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
		self.broadcast(WorkerEvent::Set(hashed_key, base_size, expiry, old_object_info))?;

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
	pub fn resize(&self, max_size: CacheSize) -> Result<(), CacheError> {
		if max_size == 0 {
			return Err(CacheError::ZeroCacheSize);
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
		if let Err(err) = self.worker_manager.try_send(event) {
			error!("Could not communicate with workers: {err:?}");
			return Err(CacheError::Internal);
		}

		Ok(())
	}

	fn hash_key(&self, key: &K) -> HashedKey {
		self.hasher.hash_one(key)
	}
}

/// Single-instance, segmented-LRU hybrid cache: one `PaperCache<K,
/// TieredBuffer>` running `PaperPolicy::LruHybrid`, in contrast with
/// composing two independent `PaperCache` instances. See the `lru_hybrid_cache` module docs for the
/// full design. This impl block only carries `new`/`with_hasher` (this
/// design's constructor has no extra parameters beyond `max_size`/
/// `fast_tier_size`) and the `lru_hybrid_stats()` accessor -- every other
/// method is shared, in the generic block above (see
/// `crate::hybrid_policy::HybridPolicy`).
#[cfg(feature = "lru_hybrid_cache")]
impl<K, S> PaperCache<K, TieredBuffer, S>
where
	K: 'static + Eq + Hash + TypeSize + std::fmt::Debug + Clone + Send + Sync,
	S: Default + Clone + BuildHasher,
{
	/// Creates an empty `PaperCache` running `PaperPolicy::LruHybrid`, with
	/// the given overall `max_size` and initial fast-tier byte budget
	/// `fast_tier_size` (adjustable afterward via
	/// [`Self::set_fast_tier_size`]).
	///
	/// # Errors
	///
	/// Returns [`CacheError::ZeroCacheSize`] if `max_size` is zero, or
	/// [`CacheError::InvalidFastTierSize`] if `fast_tier_size` resolves to
	/// zero bytes or exceeds `max_size`.
	///
	/// # Examples
	///
	/// ```ignore
	/// use paper_cache::{PaperCache, TieredBuffer, CacheTierSize};
	///
	/// let cache = PaperCache::<u32, TieredBuffer>::new(
	///     10_000_000,
	///     CacheTierSize::Mb(2),
	/// ).unwrap();
	///
	/// cache.set(1u32, b"hello world", None).unwrap();
	/// assert_eq!(cache.get(&1u32).unwrap(), b"hello world");
	/// ```
	pub fn new(max_size: CacheSize, fast_tier_size: CacheTierSize) -> Result<Self, CacheError> {
		Self::with_hasher(max_size, fast_tier_size, Default::default())
	}

	/// Creates an empty `PaperCache` with the supplied hasher. See [`Self::new`].
	pub fn with_hasher(
		max_size: CacheSize,
		fast_tier_size: CacheTierSize,
		hasher: S,
	) -> Result<Self, CacheError> {
		Self::new_hybrid(max_size, fast_tier_size, (), hasher)
	}

	/// Returns a point-in-time snapshot of `lru_hybrid_cache` statistics.
	#[must_use]
	pub fn lru_hybrid_stats(&self) -> LruHybridStats {
		ActiveHybridPolicy::stats_from_status(&self.status)
	}
}

/// Single-instance, segmented-LFU hybrid cache: one `PaperCache<K,
/// TieredBuffer>` running `PaperPolicy::LfuHybrid`, in contrast with
/// composing two independent `PaperCache` instances. See the `lfu_hybrid_cache` module docs for the
/// full design; only `new`/`with_hasher`/`lfu_hybrid_stats()` live here --
/// everything else is shared (see the generic block above).
///
/// Admission always lands in the fast tier; once fast-tier pressure demotes
/// a key, it is (by construction) always the fast tier's lowest-frequency
/// resident, satisfying the paper's "new objects are, by definition, the
/// least frequently accessed" admission rule as an emergent result. A
/// slow-tier access promotes a key back to the fast tier once its frequency
/// strictly exceeds the fast tier's minimum, which may itself cascade a
/// demotion. Terminal evictions (once overall `max_size` is exceeded) only
/// ever remove the slow-tier minimum-frequency resident. Every migration
/// physically reallocates the object's bytes (see [`TieredBuffer`] and
/// `Object::set_data`) — a key is never present in both tiers at once.
///
/// Mutually exclusive with `lru_hybrid_cache` (see `lib.rs`'s
/// `compile_error!` guard) since both would otherwise define the same
/// inherent-method impl block on the identical `PaperCache<K, TieredBuffer,
/// S>` type.
#[cfg(feature = "lfu_hybrid_cache")]
impl<K, S> PaperCache<K, TieredBuffer, S>
where
	K: 'static + Eq + Hash + TypeSize + std::fmt::Debug + Clone + Send + Sync,
	S: Default + Clone + BuildHasher,
{
	/// Creates an empty `PaperCache` running `PaperPolicy::LfuHybrid`, with
	/// the given overall `max_size` and initial fast-tier byte budget
	/// `fast_tier_size` (adjustable afterward via
	/// [`Self::set_fast_tier_size`]).
	///
	/// # Errors
	///
	/// Returns [`CacheError::ZeroCacheSize`] if `max_size` is zero, or
	/// [`CacheError::InvalidFastTierSize`] if `fast_tier_size` resolves to
	/// zero bytes or exceeds `max_size`.
	pub fn new(max_size: CacheSize, fast_tier_size: CacheTierSize) -> Result<Self, CacheError> {
		Self::with_hasher(max_size, fast_tier_size, Default::default())
	}

	/// Creates an empty `PaperCache` with the supplied hasher. See [`Self::new`].
	pub fn with_hasher(
		max_size: CacheSize,
		fast_tier_size: CacheTierSize,
		hasher: S,
	) -> Result<Self, CacheError> {
		Self::new_hybrid(max_size, fast_tier_size, (), hasher)
	}

	/// Returns a point-in-time snapshot of `lfu_hybrid_cache` statistics.
	#[must_use]
	pub fn lfu_hybrid_stats(&self) -> LfuHybridStats {
		ActiveHybridPolicy::stats_from_status(&self.status)
	}
}

/// Single-instance, segmented-2Q hybrid cache: one `PaperCache<K,
/// TieredBuffer>` running `PaperPolicy::TwoQHybrid`, in contrast with
/// composing two independent `PaperCache` instances. See the `two_q_hybrid_cache` module docs for the
/// full design; this impl block only carries `new`/`with_hasher` (the one
/// design with an extra `k_in` constructor parameter) and
/// `two_q_hybrid_stats()` -- everything else is shared (see the generic
/// block above).
///
/// Admission always lands in a one-access FIFO queue entirely in the slow
/// tier. A re-access to a FIFO-queue object promotes it straight to the top
/// of the main queue's fast tier. Once in the main queue, an object behaves
/// exactly like `lru_hybrid_cache`: fast-tier pressure demotes the LRU
/// tail; a slow-tier access promotes it back, possibly cascading a further
/// demotion. Terminal evictions prefer the FIFO queue's tail (an object
/// that aged out without a second access) before ever touching the main
/// queue's slow tail. Every migration physically reallocates the object's
/// bytes (see [`TieredBuffer`] and `Object::set_data`) — a key is never
/// present in both tiers at once.
///
/// Mutually exclusive with `lru_hybrid_cache`/`lfu_hybrid_cache` (see
/// `lib.rs`'s `compile_error!` guards) since all three would otherwise
/// define the same inherent-method impl block on the identical
/// `PaperCache<K, TieredBuffer, S>` type.
#[cfg(feature = "two_q_hybrid_cache")]
impl<K, S> PaperCache<K, TieredBuffer, S>
where
	K: 'static + Eq + Hash + TypeSize + std::fmt::Debug + Clone + Send + Sync,
	S: Default + Clone + BuildHasher,
{
	/// Creates an empty `PaperCache` running `PaperPolicy::TwoQHybrid`, with
	/// the given overall `max_size`, initial fast-tier byte budget
	/// `fast_tier_size` (adjustable afterward via
	/// [`Self::set_fast_tier_size`]), and `k_in` — the FIFO queue's byte
	/// budget as a fraction of `max_size` (fixed for the lifetime of the
	/// cache, rescaled proportionally on [`Self::resize`], same as plain
	/// `PaperPolicy::TwoQ`'s `k_in`).
	///
	/// # Errors
	///
	/// Returns [`CacheError::ZeroCacheSize`] if `max_size` is zero,
	/// [`CacheError::InvalidFastTierSize`] if `fast_tier_size` resolves to
	/// zero bytes or exceeds `max_size`, or [`CacheError::InvalidPolicy`] if
	/// `k_in` is outside `[0.0, 1.0]`.
	///
	/// # Examples
	///
	/// ```ignore
	/// use paper_cache::{PaperCache, TieredBuffer, CacheTierSize};
	///
	/// let cache = PaperCache::<u32, TieredBuffer>::new(
	///     10_000_000,
	///     CacheTierSize::Mb(2),
	///     0.2,
	/// ).unwrap();
	///
	/// cache.set(1u32, b"hello world", None).unwrap();
	/// assert_eq!(cache.get(&1u32).unwrap(), b"hello world");
	/// ```
	pub fn new(max_size: CacheSize, fast_tier_size: CacheTierSize, k_in: f64) -> Result<Self, CacheError> {
		Self::with_hasher(max_size, fast_tier_size, k_in, Default::default())
	}

	/// Creates an empty `PaperCache` with the supplied hasher. See [`Self::new`].
	pub fn with_hasher(
		max_size: CacheSize,
		fast_tier_size: CacheTierSize,
		k_in: f64,
		hasher: S,
	) -> Result<Self, CacheError> {
		if !(0.0..=1.0).contains(&k_in) {
			return Err(CacheError::InvalidPolicy);
		}

		Self::new_hybrid(max_size, fast_tier_size, k_in, hasher)
	}

	/// Returns a point-in-time snapshot of `two_q_hybrid_cache` statistics.
	#[must_use]
	pub fn two_q_hybrid_stats(&self) -> TwoQHybridStats {
		ActiveHybridPolicy::stats_from_status(&self.status)
	}
}

/// Single-instance, segmented-FIFO hybrid cache: one `PaperCache<K,
/// TieredBuffer>` running `PaperPolicy::FifoHybrid`, in contrast with
/// composing two independent `PaperCache` instances. See the `fifo_hybrid_cache` module docs for the
/// full design; only `new`/`with_hasher`/`fifo_hybrid_stats()` live here --
/// everything else is shared (see the generic block above).
///
/// Mutually exclusive with `lru_hybrid_cache`/`lfu_hybrid_cache`/
/// `two_q_hybrid_cache` (see `lib.rs`'s `compile_error!` guards) since all
/// four would otherwise define the same inherent-method impl block on the
/// identical `PaperCache<K, TieredBuffer, S>` type.
#[cfg(feature = "fifo_hybrid_cache")]
impl<K, S> PaperCache<K, TieredBuffer, S>
where
	K: 'static + Eq + Hash + TypeSize + std::fmt::Debug + Clone + Send + Sync,
	S: Default + Clone + BuildHasher,
{
	/// Creates an empty `PaperCache` running `PaperPolicy::FifoHybrid`, with
	/// the given overall `max_size` and initial fast-tier byte budget
	/// `fast_tier_size` (adjustable afterward via
	/// [`Self::set_fast_tier_size`]).
	///
	/// # Errors
	///
	/// Returns [`CacheError::ZeroCacheSize`] if `max_size` is zero, or
	/// [`CacheError::InvalidFastTierSize`] if `fast_tier_size` resolves to
	/// zero bytes or exceeds `max_size`.
	pub fn new(max_size: CacheSize, fast_tier_size: CacheTierSize) -> Result<Self, CacheError> {
		Self::with_hasher(max_size, fast_tier_size, Default::default())
	}

	/// Creates an empty `PaperCache` with the supplied hasher. See [`Self::new`].
	pub fn with_hasher(
		max_size: CacheSize,
		fast_tier_size: CacheTierSize,
		hasher: S,
	) -> Result<Self, CacheError> {
		Self::new_hybrid(max_size, fast_tier_size, (), hasher)
	}

	/// Returns a point-in-time snapshot of `fifo_hybrid_cache` statistics.
	#[must_use]
	pub fn fifo_hybrid_stats(&self) -> FifoHybridStats {
		ActiveHybridPolicy::stats_from_status(&self.status)
	}
}

/// Single-instance, segmented-LRU hybrid cache with a size-split fast AND
/// slow tier: same `PaperCache<K, TieredBuffer>` architecture and LRU
/// admission/promotion/demotion/eviction semantics as `lru_hybrid_cache`,
/// but each tier's bookkeeping is split into two independently-tracked
/// segments ("small"/"large") by object size. See the
/// `lru_sized_hybrid_cache` module docs.
///
/// Mutually exclusive with `lru_hybrid_cache`/`lfu_hybrid_cache`/
/// `two_q_hybrid_cache`/`fifo_hybrid_cache` (see `lib.rs`'s `compile_error!`
/// guards).
///
/// Sizing knobs: [`Self::set_fast_tier_size`]/[`Self::fast_tier_size`]
/// (defined on the shared generic block above) resize/read the SMALL fast
/// segment specifically for this design -- unlike every other hybrid, where
/// they mean the whole fast tier -- because this design has a second,
/// independent fast segment with no shared-block equivalent.
/// [`Self::set_large_fast_tier_size`]/[`Self::large_fast_tier_size`] and
/// [`Self::set_size_threshold`]/[`Self::size_threshold`] are this design's
/// own bespoke accessors, defined here.
#[cfg(feature = "lru_sized_hybrid_cache")]
impl<K, S> PaperCache<K, TieredBuffer, S>
where
	K: 'static + Eq + Hash + TypeSize + std::fmt::Debug + Clone + Send + Sync,
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
	pub fn new(
		max_size: CacheSize,
		small_fast_tier_size: CacheTierSize,
		large_fast_tier_size: CacheTierSize,
		size_threshold: CacheTierSize,
	) -> Result<Self, CacheError> {
		Self::with_hasher(max_size, small_fast_tier_size, large_fast_tier_size, size_threshold, Default::default())
	}

	/// Creates an empty `PaperCache` with the supplied hasher. See [`Self::new`].
	pub fn with_hasher(
		max_size: CacheSize,
		small_fast_tier_size: CacheTierSize,
		large_fast_tier_size: CacheTierSize,
		size_threshold: CacheTierSize,
		hasher: S,
	) -> Result<Self, CacheError> {
		Self::new_sized_hybrid(max_size, small_fast_tier_size, large_fast_tier_size, size_threshold, hasher)
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

		let policy = PaperPolicy::LruSizedHybrid;
		let policies = [policy];

		let objects = new_hybrid_object_map();
		let status = Arc::new(AtomicStatus::new(max_size, &policies, policy)?);
		let overhead_manager = Arc::new(OverheadManager::new(&status));

		status.set_fast_tier_capacity(small_capacity);
		status.set_lru_sized_hybrid_large_fast_capacity(large_capacity);
		status.set_lru_sized_hybrid_size_threshold(threshold);

		// Same byte-length-preserving contract `new_hybrid`'s `migrate`
		// closure documents.
		let migrate: Box<dyn Fn(&TieredBuffer, Tier) -> TieredBuffer + Send + Sync> =
			Box::new(|buffer, tier| match tier {
				Tier::Fast => TieredBuffer::new_fast(buffer.as_ref()),
				Tier::Slow => TieredBuffer::new_slow(buffer.as_ref()),
			});

		let (worker_sender, worker_listener) = unbounded();

		let (mut worker_manager, mut worker_handles) = WorkerManager::new_with_tier_migration(
			worker_listener,
			&objects,
			&status,
			&overhead_manager,
			migrate,
		)?;

		worker_handles.push(thread::spawn(move || worker_manager.run()));

		let cache = PaperCache {
			objects,
			status,
			worker_manager: Arc::new(worker_sender),
			worker_handles,
			overhead_manager,
			hasher,
		};

		cache.broadcast(WorkerEvent::ResizeFastTier(small_capacity))?;
		cache.broadcast(WorkerEvent::ResizeLargeFastTier(large_capacity))?;
		cache.broadcast(WorkerEvent::ResizeSizeThreshold(threshold))?;

		Ok(cache)
	}

	/// Returns a point-in-time snapshot of `lru_sized_hybrid_cache` statistics.
	#[must_use]
	pub fn lru_sized_hybrid_stats(&self) -> LruSizedHybridStats {
		ActiveHybridPolicy::stats_from_status(&self.status)
	}

	/// Runtime-adjusts the LARGE fast segment's byte budget. The SMALL
	/// segment is adjusted via the shared [`Self::set_fast_tier_size`]
	/// instead (see this impl block's own doc for why).
	pub fn set_large_fast_tier_size(&self, size: CacheTierSize) -> Result<(), CacheError> {
		let bytes = size.to_bytes();

		if bytes == 0 || bytes > self.status.max_size() {
			return Err(CacheError::InvalidFastTierSize);
		}

		self.status.set_lru_sized_hybrid_large_fast_capacity(bytes);
		self.broadcast(WorkerEvent::ResizeLargeFastTier(bytes))?;

		Ok(())
	}

	/// Returns the LARGE fast segment's current byte budget.
	#[must_use]
	pub fn large_fast_tier_size(&self) -> CacheSize {
		self.status.lru_sized_hybrid_large_fast_capacity()
	}

	/// Runtime-adjusts the small/large size-classification threshold. Only
	/// affects future admissions, overwrites, and slow-to-fast promotions --
	/// already-tracked keys are not retroactively rescanned/reclassified.
	pub fn set_size_threshold(&self, threshold: CacheTierSize) -> Result<(), CacheError> {
		let bytes = threshold.to_bytes();

		self.status.set_lru_sized_hybrid_size_threshold(bytes);
		self.broadcast(WorkerEvent::ResizeSizeThreshold(bytes))?;

		Ok(())
	}

	/// Returns the current size-classification threshold, in bytes.
	#[must_use]
	pub fn size_threshold(&self) -> CacheSize {
		self.status.lru_sized_hybrid_size_threshold()
	}
}

// Tests for global_hashtable_pmem alone (without key_value_pmem)
#[cfg(all(feature = "global_hashtable_pmem", not(feature = "key_value_pmem")))]
#[cfg(all(test, feature = "global_hashtable_pmem"))]
mod test_global_hashtable_pmem_alone {
    use crate::{PaperCache, PaperPolicy};
    use std::hash::RandomState;

    #[test]
    fn test_basic_operations() {
        // Create cache with global hashtable in PMEM, values in DRAM
        let cache: PaperCache<u32, Box<[u8]>, RandomState> = PaperCache::new(
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
        let cache: PaperCache<u32, Box<[u8]>, RandomState> = PaperCache::new(
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
        let cache: PaperCache<String, Box<[u8]>, RandomState> = PaperCache::new(
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
    use crate::{PaperCache, PaperPolicy};
    use std::hash::RandomState;

    /// Verify that the cache initializes and operates correctly with the LFU policy.
    /// The LfuStack used for eviction is backed by DRAM or PMEM depending on the
    /// `eviction_stacks_pmem` feature flag — both paths must initialize correctly.
    #[test]
    fn test_cache_init_with_lfu_eviction() {
        let cache: PaperCache<u32, Box<[u8]>, RandomState> = PaperCache::new(
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
        let cache: PaperCache<u32, Box<[u8]>, RandomState> = PaperCache::new(
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
/// new_slow` allocates through the `Hybrid`/UMF PMEM allocator, which
/// requires real PMEM/DAX
/// hardware and aborts ("memory allocation ... failed") in a plain dev
/// sandbox. A full integration test covering demotion/promotion/eviction
/// belongs in `tests/lru_hybrid_cache_integration.rs` (not yet written —
/// see `CLAUDE.md`'s `lru_hybrid_cache` plan, step 12) and should be run on
/// PMEM-capable hardware.
#[cfg(all(test, feature = "lru_hybrid_cache"))]
mod test_lru_hybrid_cache {
    use crate::{PaperCache, TieredBuffer, CacheTierSize, Tier, CacheError};

    #[test]
    fn basic_construction_and_fast_tier_only_roundtrip() {
        let cache = PaperCache::<u32, TieredBuffer>::new(
            1_000_000,
            CacheTierSize::Bytes(1_000_000), // fast tier == whole cache: no demotion possible
        ).expect("cache should construct");

        cache.set(1u32, b"hello world", None).expect("set should succeed");
        assert!(cache.has(&1u32));
        assert_eq!(cache.get(&1u32).unwrap(), b"hello world");
        assert_eq!(cache.tier_of(&1u32), Some(Tier::Fast));

        let stats = cache.lru_hybrid_stats();
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
            PaperCache::<u32, TieredBuffer>::new(1000, CacheTierSize::Bytes(2000)),
            Err(CacheError::InvalidFastTierSize),
        ));

        assert!(matches!(
            PaperCache::<u32, TieredBuffer>::new(1000, CacheTierSize::Bytes(0)),
            Err(CacheError::InvalidFastTierSize),
        ));

        let cache = PaperCache::<u32, TieredBuffer>::new(1000, CacheTierSize::Bytes(500))
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
            CacheTierSize::Bytes(1_000_000),
        ).expect("cache should construct");

        cache.set(1u32, b"value", Some(60)).expect("set should succeed");
        assert!(cache.ttl(&1u32, Some(120)).is_ok());
        assert_eq!(cache.get(&1u32).unwrap(), b"value");
    }
}

/// Exercises the real public `PaperCache<K, TieredBuffer>` API end to end
/// for `lfu_hybrid_cache`. See `test_lru_hybrid_cache`'s doc comment for why
/// this deliberately stays on the fast-tier-only path (no PMEM allocation) —
/// the full tier-crossing coverage lives in
/// `tests/lfu_hybrid_cache_integration.rs`.
#[cfg(all(test, feature = "lfu_hybrid_cache"))]
mod test_lfu_hybrid_cache {
    use crate::{PaperCache, TieredBuffer, CacheTierSize, Tier, CacheError};

    #[test]
    fn basic_construction_and_fast_tier_only_roundtrip() {
        let cache = PaperCache::<u32, TieredBuffer>::new(
            1_000_000,
            CacheTierSize::Bytes(1_000_000), // fast tier == whole cache: no demotion possible
        ).expect("cache should construct");

        cache.set(1u32, b"hello world", None).expect("set should succeed");
        assert!(cache.has(&1u32));
        assert_eq!(cache.get(&1u32).unwrap(), b"hello world");
        assert_eq!(cache.tier_of(&1u32), Some(Tier::Fast));

        let stats = cache.lfu_hybrid_stats();
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
            PaperCache::<u32, TieredBuffer>::new(1000, CacheTierSize::Bytes(2000)),
            Err(CacheError::InvalidFastTierSize),
        ));

        assert!(matches!(
            PaperCache::<u32, TieredBuffer>::new(1000, CacheTierSize::Bytes(0)),
            Err(CacheError::InvalidFastTierSize),
        ));

        let cache = PaperCache::<u32, TieredBuffer>::new(1000, CacheTierSize::Bytes(500))
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
            CacheTierSize::Bytes(1_000_000),
        ).expect("cache should construct");

        cache.set(1u32, b"value", Some(60)).expect("set should succeed");
        assert!(cache.ttl(&1u32, Some(120)).is_ok());
        assert_eq!(cache.get(&1u32).unwrap(), b"value");
    }
}

/// Exercises the real public `PaperCache<K, TieredBuffer>` API end to end
/// for `two_q_hybrid_cache`. Unlike `test_lru_hybrid_cache`/
/// `test_lfu_hybrid_cache`, this module cannot avoid the real `Hybrid`/UMF
/// PMEM allocator: `set()` always admits via `TieredBuffer::new_slow`
/// regardless of `fast_tier_size`, so even a single `set()` call here pays
/// the one-time PMEM pool warm-up cost (see `tests/two_q_hybrid_cache_integration.rs`'s
/// module doc for details). The full tier-crossing coverage lives there.
#[cfg(all(test, feature = "two_q_hybrid_cache"))]
mod test_two_q_hybrid_cache {
    use crate::{PaperCache, TieredBuffer, CacheTierSize, Tier, CacheError};

    #[test]
    fn basic_construction_and_slow_tier_admission_roundtrip() {
        let cache = PaperCache::<u32, TieredBuffer>::new(
            1_000_000,
            CacheTierSize::Bytes(1_000_000),
            0.5,
        ).expect("cache should construct");

        cache.set(1u32, b"hello world", None).expect("set should succeed");
        assert!(cache.has(&1u32));
        assert_eq!(cache.get(&1u32).unwrap(), b"hello world");
        assert_eq!(cache.tier_of(&1u32), Some(Tier::Slow));

        let stats = cache.two_q_hybrid_stats();
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
            PaperCache::<u32, TieredBuffer>::new(1000, CacheTierSize::Bytes(2000), 0.5),
            Err(CacheError::InvalidFastTierSize),
        ));

        assert!(matches!(
            PaperCache::<u32, TieredBuffer>::new(1000, CacheTierSize::Bytes(0), 0.5),
            Err(CacheError::InvalidFastTierSize),
        ));

        let cache = PaperCache::<u32, TieredBuffer>::new(1000, CacheTierSize::Bytes(500), 0.5)
            .expect("cache should construct");

        assert!(matches!(
            cache.set_fast_tier_size(CacheTierSize::Bytes(2000)),
            Err(CacheError::InvalidFastTierSize),
        ));
    }

    #[test]
    fn invalid_k_in_is_rejected() {
        assert!(matches!(
            PaperCache::<u32, TieredBuffer>::new(1000, CacheTierSize::Bytes(500), 1.5),
            Err(CacheError::InvalidPolicy),
        ));

        assert!(matches!(
            PaperCache::<u32, TieredBuffer>::new(1000, CacheTierSize::Bytes(500), -0.1),
            Err(CacheError::InvalidPolicy),
        ));
    }

    #[test]
    fn ttl_is_preserved_across_a_set() {
        let cache = PaperCache::<u32, TieredBuffer>::new(
            1_000_000,
            CacheTierSize::Bytes(1_000_000),
            0.5,
        ).expect("cache should construct");

        cache.set(1u32, b"value", Some(60)).expect("set should succeed");
        assert!(cache.ttl(&1u32, Some(120)).is_ok());
        assert_eq!(cache.get(&1u32).unwrap(), b"value");
    }
}

/// Exercises the real public `PaperCache<K, TieredBuffer>` API end to end
/// for `fifo_hybrid_cache`. See `test_lru_hybrid_cache`'s doc comment for why
/// this deliberately stays on the fast-tier-only path (no PMEM allocation) —
/// the full tier-crossing coverage (including Correction 2's slow-tier
/// overwrite path) lives in `tests/fifo_hybrid_cache_integration.rs`.
#[cfg(all(test, feature = "fifo_hybrid_cache"))]
mod test_fifo_hybrid_cache {
    use crate::{PaperCache, TieredBuffer, CacheTierSize, Tier, CacheError};

    #[test]
    fn basic_construction_and_fast_tier_only_roundtrip() {
        let cache = PaperCache::<u32, TieredBuffer>::new(
            1_000_000,
            CacheTierSize::Bytes(1_000_000), // fast tier == whole cache: no demotion possible
        ).expect("cache should construct");

        cache.set(1u32, b"hello world", None).expect("set should succeed");
        assert!(cache.has(&1u32));
        assert_eq!(cache.get(&1u32).unwrap(), b"hello world");
        assert_eq!(cache.tier_of(&1u32), Some(Tier::Fast));

        let stats = cache.fifo_hybrid_stats();
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
            PaperCache::<u32, TieredBuffer>::new(1000, CacheTierSize::Bytes(2000)),
            Err(CacheError::InvalidFastTierSize),
        ));

        assert!(matches!(
            PaperCache::<u32, TieredBuffer>::new(1000, CacheTierSize::Bytes(0)),
            Err(CacheError::InvalidFastTierSize),
        ));

        let cache = PaperCache::<u32, TieredBuffer>::new(1000, CacheTierSize::Bytes(500))
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
            CacheTierSize::Bytes(1_000_000),
        ).expect("cache should construct");

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
/// `lru_sized_hybrid_cache` end to end. Deliberately stays on the
/// fast-tier-only path (both fast-segment capacities == max_size, tiny
/// values) so no object ever demotes -- see `test_lru_hybrid_cache`'s
/// identical rationale. A full integration test covering demotion/
/// promotion/eviction across both segments and both tiers belongs in
/// `tests/lru_sized_hybrid_cache_integration.rs` and should be run on
/// PMEM-capable hardware.
#[cfg(all(test, feature = "lru_sized_hybrid_cache"))]
mod test_lru_sized_hybrid_cache {
    use crate::{PaperCache, TieredBuffer, CacheTierSize, Tier, CacheError};

    #[test]
    fn basic_construction_and_fast_tier_only_roundtrip() {
        let cache = PaperCache::<u32, TieredBuffer>::new(
            1_000_000,
            CacheTierSize::Bytes(1_000_000), // small segment == whole cache
            CacheTierSize::Bytes(1_000_000), // large segment == whole cache
            CacheTierSize::Bytes(1_000_000), // threshold huge -> everything classifies small
        ).expect("cache should construct");

        cache.set(1u32, b"hello world", None).expect("set should succeed");
        assert!(cache.has(&1u32));
        assert_eq!(cache.get(&1u32).unwrap(), b"hello world");
        assert_eq!(cache.tier_of(&1u32), Some(Tier::Fast));

        let stats = cache.lru_sized_hybrid_stats();
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
            PaperCache::<u32, TieredBuffer>::new(
                1000, CacheTierSize::Bytes(2000), CacheTierSize::Bytes(500), CacheTierSize::Bytes(100),
            ),
            Err(CacheError::InvalidFastTierSize),
        ));

        assert!(matches!(
            PaperCache::<u32, TieredBuffer>::new(
                1000, CacheTierSize::Bytes(0), CacheTierSize::Bytes(500), CacheTierSize::Bytes(100),
            ),
            Err(CacheError::InvalidFastTierSize),
        ));

        assert!(matches!(
            PaperCache::<u32, TieredBuffer>::new(
                1000, CacheTierSize::Bytes(500), CacheTierSize::Bytes(2000), CacheTierSize::Bytes(100),
            ),
            Err(CacheError::InvalidFastTierSize),
        ));

        let cache = PaperCache::<u32, TieredBuffer>::new(
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
        let cache = PaperCache::<u32, TieredBuffer>::new(
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
        let cache = PaperCache::<u32, TieredBuffer>::new(
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
