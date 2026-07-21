/*
 * Copyright (c) Kia Shakiba
 *
 * This source code is licensed under the GNU AGPLv3 license found in the
 * LICENSE file in the root directory of this source tree.
 * correct
 */

#![cfg_attr(any(feature = "hashbrown_dram", feature = "all_dram", feature = "key_value_pmem", feature = "global_hashtable_pmem", feature = "tiering_hashtable_pmem", feature = "flatmap_dram", feature = "flatmap_pmem", feature = "global_flatmap_dram", feature = "global_flatmap_pmem", feature = "eviction_stacks_pmem", feature = "pmem_region_alloc", feature = "region_hybrid_allocator", feature = "devdax_bump"), feature(allocator_api), feature(clone_from_ref))]

//#![cfg_attr(any(feature = "key_value_pmem", feature = "global_hashtable_pmem", feature = "tiering_hashtable_pmem", feature = "flatmap_dram", feature = "flatmap_pmem", feature = "global_flatmap_dram", feature = "global_flatmap_pmem", feature = "eviction_stacks_pmem", feature = "pmem_region_alloc", feature = "region_hybrid_allocator", feature = "devdax_bump"), feature(allocator_api))]

// Validate that both global_flatmap_dram and global_flatmap_pmem are not enabled together
#[cfg(all(feature = "global_flatmap_dram", feature = "global_flatmap_pmem"))]
compile_error!("Cannot enable both 'global_flatmap_dram' and 'global_flatmap_pmem' features simultaneously. Please choose only one FlatMap mode for the global hashtable.");

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

// Validate that hashbrown_dram is not enabled with other global hashtable features
#[cfg(all(feature = "hashbrown_dram", feature = "global_hashtable_pmem"))]
compile_error!("Cannot enable both 'hashbrown_dram' and 'global_hashtable_pmem' features simultaneously. Please choose only one global hashtable mode.");

#[cfg(all(feature = "hashbrown_dram", feature = "global_flatmap_dram"))]
compile_error!("Cannot enable both 'hashbrown_dram' and 'global_flatmap_dram' features simultaneously. Please choose only one global hashtable mode.");

#[cfg(all(feature = "hashbrown_dram", feature = "global_flatmap_pmem"))]
compile_error!("Cannot enable both 'hashbrown_dram' and 'global_flatmap_pmem' features simultaneously. Please choose only one global hashtable mode.");

// When all_dram is enabled, use jemalloc as the global allocator
//#[cfg(feature = "all_dram")]
//use tikv_jemallocator::Jemalloc;

//#[cfg(feature = "all_dram")]
//#[global_allocator]
//static GLOBAL: Jemalloc = Jemalloc;


#[cfg(any(feature = "hashbrown_dram", feature = "key_value_pmem", feature = "global_hashtable_pmem", feature = "tiering_hashtable_pmem", feature = "flatmap_pmem", feature = "global_flatmap_pmem", feature = "eviction_stacks_pmem", feature = "pmem_region_alloc", feature = "region_hybrid_allocator", feature = "devdax_bump", feature = "all_dram"))]
pub mod allocator;


/*
#[cfg(all(
    any(feature = "key_value_pmem", feature = "global_hashtable_pmem", feature = "tiering_hashtable_pmem", feature = "flatmap_pmem", feature = "global_flatmap_pmem", feature = "eviction_stacks_pmem", feature = "pmem_region_alloc", feature = "region_hybrid_allocator"),
    any(feature = "pmem_region_alloc", feature = "region_hybrid_allocator")
))]
use crate::allocator::RegionHybrid as Hybrid;

#[cfg(all(
    any(feature = "key_value_pmem", feature = "global_hashtable_pmem", feature = "tiering_hashtable_pmem", feature = "flatmap_pmem", feature = "global_flatmap_pmem", feature = "eviction_stacks_pmem", feature = "pmem_region_alloc", feature = "region_hybrid_allocator"),
    not(any(feature = "pmem_region_alloc", feature = "region_hybrid_allocator"))
))]
use crate::allocator::HybridObjects as Hybrid;
*/



use std::arch::x86_64::{_mm_clflush, _mm_sfence};


#[cfg(all(
    any(
        feature = "key_value_pmem",
        feature = "global_hashtable_pmem",
        feature = "tiering_hashtable_pmem",
        feature = "flatmap_pmem",
        feature = "global_flatmap_pmem",
        feature = "eviction_stacks_pmem",
        feature = "pmem_region_alloc",
        feature = "region_hybrid_allocator",
        feature = "devdax_bump",
    ),
    feature = "devdax_bump"
))]
use crate::allocator::DevDaxBump as Hybrid;

#[cfg(all(
    any(
        feature = "key_value_pmem",
        feature = "global_hashtable_pmem",
        feature = "tiering_hashtable_pmem",
        feature = "flatmap_pmem",
        feature = "global_flatmap_pmem",
        feature = "eviction_stacks_pmem",
        feature = "pmem_region_alloc",
        feature = "region_hybrid_allocator",
        feature = "devdax_bump",
    ),
    not(feature = "devdax_bump"),
    any(feature = "pmem_region_alloc", feature = "region_hybrid_allocator")
))]
use crate::allocator::RegionHybrid as Hybrid;

#[cfg(all(
    any(
        feature = "key_value_pmem",
		feature = "key_pmem_value_pmem",
        feature = "global_hashtable_pmem",
        feature = "tiering_hashtable_pmem",
        feature = "flatmap_pmem",
        feature = "global_flatmap_pmem",
        feature = "eviction_stacks_pmem",
        feature = "pmem_region_alloc",
        feature = "region_hybrid_allocator",
        feature = "devdax_bump",
    ),
    not(any(
        feature = "devdax_bump",
        feature = "pmem_region_alloc",
        feature = "region_hybrid_allocator",
    ))
))]
use crate::allocator::HybridObjects as Hybrid;
//use crate::allocator::DAXPMEM as Hybrid;



// UMF bindings are always needed when any PMEM feature is active.
// The build script guarantees that the UMF C symbols are always present:
// either the real UMF library (when wrapper.h exists) or the stub
// implementation (umf_stub.c, using malloc/free) when UMF is unavailable.
#[cfg(any(feature = "key_value_pmem", feature = "global_hashtable_pmem", feature = "tiering_hashtable_pmem", feature = "flatmap_pmem", feature = "global_flatmap_pmem", feature = "eviction_stacks_pmem", feature = "pmem_region_alloc", feature = "region_hybrid_allocator"))]
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

// Shared tier-size unit type (bytes/Mb/Gb), used by `hybridcache`,
// `lru_hybrid_cache`, `lfu_hybrid_cache`, `two_q_hybrid_cache`, and
// `fifo_hybrid_cache` so none of them has to depend on any of the others for
// it.
#[cfg(any(feature = "hybridcache", feature = "lru_hybrid_cache", feature = "lfu_hybrid_cache", feature = "two_q_hybrid_cache", feature = "fifo_hybrid_cache"))]
mod size;

#[cfg(any(feature = "hybridcache", feature = "lru_hybrid_cache", feature = "lfu_hybrid_cache", feature = "two_q_hybrid_cache", feature = "fifo_hybrid_cache"))]
pub use crate::size::CacheTierSize;

// Shared value type for the segmented hybrid-cache features. `lru_hybrid_cache`,
// `lfu_hybrid_cache`, `two_q_hybrid_cache`, and `fifo_hybrid_cache` are
// mutually exclusive (see the `compile_error!` guards above) and all
// re-export it from their own module for source compatibility
// (`paper_cache::TieredBuffer` works either way).
#[cfg(any(feature = "lru_hybrid_cache", feature = "lfu_hybrid_cache", feature = "two_q_hybrid_cache", feature = "fifo_hybrid_cache"))]
mod tiered_buffer;

#[cfg(any(feature = "lru_hybrid_cache", feature = "lfu_hybrid_cache", feature = "two_q_hybrid_cache", feature = "fifo_hybrid_cache"))]
pub use crate::tiered_buffer::TieredBuffer;

#[cfg(any(all(feature = "key_value_pmem", feature = "enable_tiering_manager"), all(feature = "key_value_pmem", feature = "sets_dram")))]
pub mod tiering;

// FlatMap module - high-performance Linear Probing Hash Map for PMEM/DRAM
#[cfg(any(feature = "flatmap_dram", feature = "flatmap_pmem", feature = "global_flatmap_dram", feature = "global_flatmap_pmem"))]
pub mod flatmap;

#[cfg(feature = "hw_perf")]
pub mod hw_perf_counters;

#[cfg(feature = "hw_perf")]
pub use crate::hw_perf_counters::{get_hw_counters, get_hw_hashmap_stats, print_hw_perf_stats, measure_operation, HwHashMapStats, HwPerfMeasurement};

// Two-tier DRAM-first cache with S3-FIFO-inspired promotion logic.
#[cfg(feature = "hybridcache")]
pub mod hybridcache;

#[cfg(feature = "hybridcache")]
pub use crate::hybridcache::{S3FifoHybridCache, HybridCacheConfig, HybridCacheStats};

// Single-instance, segmented-LRU hybrid cache. Contrast with `hybridcache`:
// one PaperCache<K, TieredBuffer> rather than two composed instances.
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

// Re-exported so `PaperCache::tier_of`'s return type is nameable by callers
// without reaching into the private `worker` module tree directly.
#[cfg(any(feature = "lru_hybrid_cache", feature = "lfu_hybrid_cache", feature = "two_q_hybrid_cache", feature = "fifo_hybrid_cache"))]
pub use crate::worker::Tier;

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
	feature = "global_flatmap_dram",
	feature = "global_flatmap_pmem",
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

#[cfg(feature = "original")]
use std::ops::Deref;



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
	},
};

pub use crate::{
	error::CacheError,
	policy::PaperPolicy,
};

#[cfg(any(all(feature = "key_value_pmem", feature = "enable_tiering_manager"), all(feature = "key_value_pmem", feature = "sets_dram")))]
pub use crate::tiering::{TieringManager, TieringConfig, TieringStats};

pub type CacheSize = u64;
pub type AtomicCacheSize = AtomicU64;

pub type HashedKey = u64;
pub type NoHasher = BuildHasherDefault<NoHashHasher<HashedKey>>;

#[cfg(feature = "key_value_pmem")]
pub type BufferPMEM = Box<[u8], Hybrid>;


//#[cfg(feature = "pmem_region_alloc")]

//pub mod allocator;

//#[cfg(feature = "all_dram")]
//#[global_allocator]
//static GLOBAL: allocator::HybridObjects = allocator::HybridObjects;

//pub mod allocator;

// The four hybrid-cache features (lru/lfu/two_q/fifo_hybrid_cache) install
// tier_allocator's NumaAllocator (bound to NUMA node 0) as the global
// allocator instead of DRAMObjects. This is what lets TieredBuffer::Fast
// collapse to a plain Box<[u8]> (see tiered_buffer.rs) -- an ordinary heap
// allocation on this global allocator IS the fast tier for those features,
// sharing one real UMF pool per node with the slow tier's explicit
// tier_allocator::alloc_on(SLOW_TIER_NODE, ..) calls rather than each tier
// having its own independent, redundant pool. Gated on exactly the same
// predicate as `mod tiered_buffer;`/`pub use crate::tiered_buffer::
// TieredBuffer;` above, so there is no reachable build where TieredBuffer
// exists without this also being the active global allocator.
#[cfg(any(feature = "lru_hybrid_cache", feature = "lfu_hybrid_cache", feature = "two_q_hybrid_cache", feature = "fifo_hybrid_cache"))]
#[global_allocator]
static GLOBAL: tier_allocator::NumaAllocator = tier_allocator::NumaAllocator::new(0);

#[cfg(not(any(feature = "lru_hybrid_cache", feature = "lfu_hybrid_cache", feature = "two_q_hybrid_cache", feature = "fifo_hybrid_cache")))]
#[global_allocator]
static GLOBAL: allocator::DRAMObjects = allocator::DRAMObjects;



//static GLOBAL: allocator::RegionHybrid = allocator::RegionHybrid;


#[cfg(not(feature = "all_dram"))]
use std::alloc::{Layout, Allocator}; // Essential imports


//#[cfg(feature = "all_dram")]
pub type BufferDRAM = Box<[u8]>;


#[cfg(all(not(feature = "global_hashtable_pmem"), not(feature = "global_flatmap_dram"), not(feature = "global_flatmap_pmem"), not(feature = "hashbrown_dram")))]
pub type ObjectMapRef<K, V> = Arc<DashMap<HashedKey, Object<K, V>, NoHasher>>;

#[cfg(all(feature = "global_hashtable_pmem", not(feature = "global_flatmap_pmem")))]
pub type ObjectMapRef<K, V> = Arc<RwLock<HashMap<HashedKey, Object<K, V>, BuildHasherDefault<NoHashHasher<HashedKey>>, Hybrid>>>;

// FlatMap in DRAM (alternative to DashMap)
#[cfg(feature = "global_flatmap_dram")]
pub type ObjectMapRef<K, V> = Arc<RwLock<crate::flatmap::FlatMapWithHasher<HashedKey, Object<K, V>, NoHasher>>>;

// FlatMap in PMEM (alternative to HashMap with Hybrid allocator)
#[cfg(feature = "global_flatmap_pmem")]
pub type ObjectMapRef<K, V> = Arc<RwLock<crate::flatmap::FlatMapWithHasher<HashedKey, Object<K, V>, NoHasher, Hybrid>>>;

// Hashbrown HashMap in DRAM (for performance comparison with global_hashtable_pmem)
#[cfg(feature = "hashbrown_dram")]
pub type ObjectMapRef<K, V> = Arc<RwLock<HashMap<HashedKey, Object<K, V>, BuildHasherDefault<NoHashHasher<HashedKey>>>>>;


pub type StatusRef = Arc<AtomicStatus>;
pub type OverheadManagerRef = Arc<OverheadManager>;


pub struct PaperCache<K, V, S = RandomState> {
	objects: ObjectMapRef<K, V>,
	status: StatusRef,

	worker_manager: Arc<WorkerSender>,
	overhead_manager: OverheadManagerRef,
	
	#[cfg(all(feature = "key_value_pmem", any(feature = "enable_tiering_manager", feature = "sets_dram")))]
	tiering_manager: Arc<TieringManager<K, V>>,

	hasher: S,
}



pub mod rdtsc_probes;

use crate::rdtsc_probes::{
    rdtsc, PHASE_PRE_ALLOC, PHASE_ALLOC, PHASE_MEMCPY, PHASE_POST, PHASE_INSERT, PHASE_GET_HASH, PHASE_GET_LOCK, PHASE_GET_LOOKUP, PHASE_GET_VALIDATE, PHASE_GET_COPY, PHASE_GET_BROADCAST, PHASE_PROBE, PHASE_SET_BROADCAST,
};

pub use rdtsc_probes::{calibrate_tsc_hz, report_set, report_get, calibrate_probe_overhead};


//#[cfg(feature = "devdax_bump")]
//pub mod devdax_bump;

//#[cfg(feature = "devdax_bump")]
//pub use devdax_bump::DevDaxBump;


#[cfg(feature = "original")]
impl<K, V, S> PaperCache<K, V, S>
where
	K: 'static + Eq + Hash + TypeSize + std::fmt::Debug, //note added Debug for logging might impact perf thoooo
	V: 'static + TypeSize + Clone,
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

		#[cfg(all(not(feature = "global_hashtable_pmem"), not(feature = "global_flatmap_dram"), not(feature = "global_flatmap_pmem"), not(feature = "hashbrown_dram")))]
		let objects = Arc::new(DashMap::with_hasher(NoHasher::default()));
		
		#[cfg(all(feature = "global_hashtable_pmem", not(feature = "global_flatmap_pmem")))]
		let objects = Arc::new(RwLock::new(HashMap::with_hasher_in(NoHasher::default(), Hybrid)));
		
		// Hashbrown HashMap in DRAM (for performance comparison)
		#[cfg(feature = "hashbrown_dram")]
		let objects = Arc::new(RwLock::new(HashMap::with_hasher(NoHasher::default())));
		
		// FlatMap in DRAM: Use a capacity based on max_size with 2x overhead for low load factor
		// Capacity must be power of 2, so we find the next power of 2 >= (max_size * 2)
		#[cfg(feature = "global_flatmap_dram")]
		let objects = {
			// Estimate number of objects based on average object size (conservative estimate: 1KB per object)
			let estimated_objects = (max_size / 1024).max(1024) as usize;
			// Double for low load factor and find next power of 2
			let capacity = (estimated_objects * 2).next_power_of_two();
			Arc::new(RwLock::new(crate::flatmap::FlatMapWithHasher::with_capacity_and_hasher_unchecked(capacity, NoHasher::default())))
		};
		
		// FlatMap in PMEM: Use a capacity based on max_size with 2x overhead for low load factor
		#[cfg(feature = "global_flatmap_pmem")]
		let objects = {
			// Estimate number of objects based on average object size (conservative estimate: 1KB per object)
			let estimated_objects = (max_size / 1024).max(1024) as usize;
			// Double for low load factor and find next power of 2
			let capacity = (estimated_objects * 2).next_power_of_two();
			Arc::new(RwLock::new(crate::flatmap::FlatMapWithHasher::with_capacity_hasher_in(capacity, NoHasher::default(), Hybrid)))
		};
		
		let status = Arc::new(AtomicStatus::new(max_size, policies, policy)?);
		let overhead_manager = Arc::new(OverheadManager::new(&status));

		#[cfg(all(feature = "key_value_pmem", feature = "enable_tiering_manager"))]
		let tiering_manager = {
			// Create tiering manager with default DRAM threshold at 20% of max_size
			let mut tiering_config = tiering::TieringConfig::default();
			tiering_config.dram_threshold = (max_size as f64 * 0.2) as u64;
			println!("Created tiering manager with DRAM threshold: {}", tiering_config.dram_threshold);
			Arc::new(TieringManager::new(tiering_config))
		};

		let (worker_sender, worker_listener) = unbounded();

		#[cfg(all(feature = "key_value_pmem", feature = "enable_tiering_manager"))]
		let mut worker_manager = WorkerManager::new(
			worker_listener,
			&objects,
			&status,
			&overhead_manager,
			&tiering_manager,
		)?;

		#[cfg(not(all(feature = "key_value_pmem", feature = "enable_tiering_manager")))]
		let mut worker_manager = WorkerManager::new(
			worker_listener,
			&objects,
			&status,
			&overhead_manager,
		)?;

		thread::spawn(move || worker_manager.run());

		let cache = PaperCache {
			objects,
			status,

			worker_manager: Arc::new(worker_sender),
			overhead_manager,
			
			#[cfg(all(feature = "key_value_pmem", any(feature = "enable_tiering_manager", feature = "sets_dram")))]
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
	/// 
	
	pub fn get(&self, key: &K) -> Result<V, CacheError>
	where
		V: Deref<Target = [u8]> + Clone, // Clone so we can return an owned V cloned from the Arc
	{
		let hashed_key = self.hash_key(key);

		let result = match self.objects.get(&hashed_key) {
			Some(object) if object.key_matches(key) && !object.is_expired() => {
				self.status.incr_hits();
				// object.data() returns an Arc<V, Hybrid> — clone the inner V and return it
				let arc_val = object.data();
				//println!("CACHE: get for key {:?}: {:?}", key, arc_val.as_ref().clone());
				Ok(arc_val.as_ref().clone())
			},

			_ => {
				self.status.incr_misses();
				Err(CacheError::KeyNotFound)
			},
		};

		self.broadcast(WorkerEvent::Get(hashed_key, result.is_ok()))?;

		// Optional: inspect the underlying bytes/tier of the returned value for debugging
		//println!("CACHE: get result for key {:?}: {:?} ", key, result);
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

	pub fn set(&self, key: K, value: V, ttl: Option<u32>) -> Result<(), CacheError> {
		let hashed_key = self.hash_key(&key);

		let object = Object::new(key, value, ttl);
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
			.get(&hashed_key)
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

		match self.objects.get(&hashed_key) {
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

		let mut object = match self.objects.get_mut(&hashed_key) {
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

		match self.objects.get(&hashed_key) {
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




//////////////////////////////////////////////////////////
/// 
/// 

#[cfg(all(feature = "all_dram", not(feature = "global_flatmap_dram")))]
impl<K, S> PaperCache<K, BufferDRAM, S>
where
	K: 'static + Eq + Hash + TypeSize + std::fmt::Debug, //note added Debug for logging might impact perf thoooo
	//V: 'static + TypeSize,
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

		#[cfg(all(feature = "key_value_pmem", feature = "sets_dram", not(feature = "enable_tiering_manager")))]
		let tiering_manager = {
			let tiering_config = tiering::TieringConfig::default();
			let persist_cb = move |_: crate::tiering::manager::PmemBackfillJob<K>| {};
			Arc::new(TieringManager::new_with_backfill(tiering_config, persist_cb))
		};

		let (worker_sender, worker_listener) = unbounded();

		#[cfg(all(feature = "key_value_pmem", feature = "enable_tiering_manager"))]
		let mut worker_manager = WorkerManager::new(
			worker_listener,
			&objects,
			&status,
			&overhead_manager,
			&tiering_manager,
		)?;

		#[cfg(not(all(feature = "key_value_pmem", feature = "enable_tiering_manager")))]
		let mut worker_manager = WorkerManager::new(
			worker_listener,
			&objects,
			&status,
			&overhead_manager,
		)?;

		thread::spawn(move || worker_manager.run());

		let cache = PaperCache {
			objects,
			status,

			worker_manager: Arc::new(worker_sender),
			overhead_manager,
			
			#[cfg(all(feature = "key_value_pmem", any(feature = "enable_tiering_manager", feature = "sets_dram")))]
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
	/// 
	
	pub fn get(&self, key: &K) -> Result<Vec<u8>, CacheError>
	{
		let hashed_key = self.hash_key(key);

		#[cfg(debug_assertions)] println!("GET for key in all dram");

		// all_dram implementation - no tiering, all data in DRAM
		let result = match self.objects.get(&hashed_key) {
			Some(object) if object.key_matches(key) && !object.is_expired() => {
				self.status.incr_hits();
				// object.data() returns Arc<Box<[u8]>>
				// We need to clone the actual byte slice into a Vec
				let arc_val = object.data();
				Ok(arc_val.as_ref().to_vec())
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
		
		//let t0 = rdtsc();
		let hashed_key = self.hash_key(&key);
		//let t1 = rdtsc();
		//PHASE_PRE_ALLOC.record(t1 - t0);

		//allocate it as a regular buffer... 
		//let val_buf: Box<[u8]> = value.to_vec().into_boxed_slice();

		//let key_buff = Box::new(key);

		//let t2_start = rdtsc();
		//let mut val_buf: Vec<u8> = Vec::with_capacity(value.len());

		//let t2_end = rdtsc();
		//PHASE_ALLOC.record(t2_end - t2_start);

		// === phase 3: memcpy value into PMEM ===
		//let t3_start = rdtsc();
		//val_buf.extend_from_slice(value);
		//let val_buf: BufferDRAM = val_buf.into_boxed_slice();
		let val_buf: BufferDRAM = Box::clone_from_ref(value);

		/* 
		unsafe {
			let ptr = val_buf.as_ptr();
			let len = val_buf.len();

			if len > 0 {
				let cache_line_size = 64usize;
				let start = ptr as usize;
				let end = start + len;

				let mut addr = start & !(cache_line_size - 1);

				while addr < end {
					_mm_clflush(addr as *const u8);
					addr += cache_line_size;
				}

				_mm_sfence();
			}
		}
		*/
		//let t3_end = rdtsc();
		//PHASE_MEMCPY.record(t3_end - t3_start);

		//let t4_start = rdtsc();
		let object = Object::new(key, val_buf, ttl);
		//let t4_end = rdtsc();
		//PHASE_POST.record(t4_end - t4_start);
		let base_size = self.overhead_manager.base_size(&object);
		let expiry = object.expiry();

		if base_size == 0 {
			return Err(CacheError::ZeroValueSize);
		}

		if self.status.exceeds_max_size(base_size) {
			return Err(CacheError::ExceedingValueSize);
		}

		self.status.incr_sets();

		//let t5_start = rdtsc();
		let old_object_info = self.objects
			.insert(hashed_key, object)
			.map(|old_object| {
				let base_size = self.overhead_manager.base_size(&old_object);
				let expiry = old_object.expiry();

				(base_size, expiry)
			});
		//let t5_end = rdtsc();
		//PHASE_INSERT.record(t5_end - t5_start);

		let base_size_delta = if let Some((old_object_size, _)) = old_object_info {
			base_size as i64 - old_object_size as i64
		} else {
			// the object is new, so increase the number of objects count
			self.status.incr_num_objects();
			base_size as i64
		};

		self.status.update_base_used_size(base_size_delta);
		//let t6_start = rdtsc();
		self.broadcast(WorkerEvent::Set(hashed_key, base_size, expiry, old_object_info))?;
		//let t6_end = rdtsc();
		//PHASE_SET_BROADCAST.record(t6_end - t6_start);

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
			.get(&hashed_key)
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
	pub fn peek(&self, key: &K) -> Result<Arc<BufferDRAM>, CacheError> {
		let hashed_key = self.hash_key(key);

		match self.objects.get(&hashed_key) {
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

		let mut object = match self.objects.get_mut(&hashed_key) {
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

		match self.objects.get(&hashed_key) {
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

	/// Creates an empty `PaperCache` that fires `eviction_callback` each time
	/// the policy worker evicts an item.
	///
	/// The callback receives the item's hashed key, an `Arc` to its value, and
	/// a reference to its original key.  It is invoked from the policy worker
	/// background thread so it must be `Send + Sync`.
	///
	/// Used by [`crate::hybridcache::S3FifoHybridCache`] to write evicted items
	/// to the far-memory tier.
	#[cfg(feature = "hybridcache")]
	pub fn new_with_eviction_callback(
		max_size: CacheSize,
		policies: &[PaperPolicy],
		policy: PaperPolicy,
		eviction_callback: Box<dyn for<'a> Fn(HashedKey, Arc<BufferDRAM>, &'a K) + Send + Sync>,
		promotion_tx: Option<crate::worker::WorkerSender>,
	) -> Result<Self, CacheError>
	where
		S: Default,
		K: Clone,
	{
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

		let (worker_sender, worker_listener) = unbounded();

		let mut worker_manager = WorkerManager::new_with_eviction_callback(
			worker_listener,
			&objects,
			&status,
			&overhead_manager,
			eviction_callback,
			promotion_tx,
		)?;

		thread::spawn(move || worker_manager.run());

		let cache = PaperCache {
			objects,
			status,
			worker_manager: Arc::new(worker_sender),
			overhead_manager,
			hasher: Default::default(),
		};

		Ok(cache)
	}
}






#[cfg(all(feature = "key_value_pmem", not(feature = "global_hashtable_pmem"), not(feature = "global_flatmap_pmem")))]
impl<K, S> PaperCache<K, BufferPMEM, S>
where
    K: 'static + Eq + Hash + TypeSize + std::fmt::Debug + Clone + Send + Sync,
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

		#[cfg(all(feature = "key_value_pmem", feature = "enable_tiering_manager", not(feature = "sets_dram")))]
		let tiering_manager = {
			// Create tiering manager with default DRAM threshold at 20% of max_size
			let tiering_config = tiering::TieringConfig::default();
			Arc::new(TieringManager::new(tiering_config))
		};

		#[cfg(all(feature = "key_value_pmem", feature = "sets_dram"))]
		let tiering_manager = {
			let tiering_config = tiering::TieringConfig::default();

			let objects_bg = objects.clone();
			let status_bg = status.clone();
			let overhead_bg = overhead_manager.clone();

			// Arc::new_cyclic lets the persist closure capture a Weak<TieringManager>
			// so it can call mark_persisted() after the background PMEM write succeeds.
			// The Weak is safe to upgrade once jobs are dequeued, which is always after
			// the Arc is fully initialised and returned from with_hasher.
			Arc::new_cyclic(|weak_tm: &std::sync::Weak<TieringManager<K, BufferPMEM>>| {
				let weak_tm = weak_tm.clone();

				let batch_persist_cb = move |batch: Vec<crate::tiering::manager::PmemBackfillJob<K>>| {
					// Upgrade the weak reference once per batch to avoid repeated
					// atomic operations and to fail fast if the manager was dropped.
					let Some(tm) = weak_tm.upgrade() else { return };

					// ── Phase 1: Pre-allocate PMEM for each job ─────────────────────
					// Perform all allocations before touching the objects map so that
					// the DashMap shards are locked for the shortest possible window.
					// Also performs a TOCTOU check: skip any key whose tier has already
					// changed (e.g. a concurrent delete or a newer set replaced it).
					let mut pmem_objects = Vec::with_capacity(batch.len());
					for job in batch {
						// TOCTOU: if the key is no longer DramOnly, a concurrent
						// operation already updated or removed it – skip to avoid
						// overwriting a newer value.
						if !tm.is_dram_only(job.hashed_key) {
							continue;
						}

						// Allocate value bytes in PMEM via the Hybrid allocator.
						let mut pmem_vec = Vec::<u8, Hybrid>::with_capacity_in(job.value.len(), Hybrid);
						pmem_vec.extend_from_slice(&job.value);
						let val_buf: BufferPMEM = pmem_vec.into_boxed_slice();

						let object = crate::object::Object::new(job.key, val_buf, job.ttl);
						let base_size = overhead_bg.base_size(&object);

						if base_size == 0 || status_bg.exceeds_max_size(base_size) {
							// PMEM write cannot proceed; remove the stuck DramOnly entry.
							tm.remove_object(job.hashed_key);
							continue;
						}

						pmem_objects.push((job.hashed_key, object, base_size));
					}

					// ── Phase 2: Minimal locking – batch inserts into the objects map ──
					let mut batch_delta: i64 = 0;
					let mut batch_count: u64 = 0;

					for (hashed_key, object, base_size) in pmem_objects {
						let old_size = objects_bg
							.insert(hashed_key, object)
							.map(|old| overhead_bg.base_size(&old));

						match old_size {
							Some(old) => batch_delta += base_size as i64 - old as i64,
							None => {
								batch_count += 1;
								batch_delta += base_size as i64;
							}
						}

						// Transition tier from DramOnly -> DramAndPmem immediately
						// after each insert so the window where the object is in the
						// objects map but still DramOnly is as short as possible.
						tm.mark_persisted(hashed_key);
					}

					// ── Phase 3: Deferred atomics – one update per batch ─────────────
					// Apply the accumulated size delta and new-object count in a single
					// pair of atomic operations to minimise cache-line bouncing.
					status_bg.update_base_used_size(batch_delta);
					status_bg.add_num_objects(batch_count);
				};

				TieringManager::new_with_backfill(tiering_config, batch_persist_cb)
			})
		};

		let (worker_sender, worker_listener) = unbounded();

		#[cfg(all(feature = "enable_tiering_manager", not(feature = "sets_dram")))]
		let mut worker_manager = WorkerManager::new(
			worker_listener,
			&objects,
			&status,
			&overhead_manager,
			&tiering_manager,
		)?;

		#[cfg(feature = "sets_dram")]
		let mut worker_manager = WorkerManager::new(
			worker_listener,
			&objects,
			&status,
			&overhead_manager,
			&tiering_manager,
		)?;

		#[cfg(all(not(feature = "enable_tiering_manager"), not(feature = "sets_dram")))]
		let mut worker_manager = WorkerManager::new(
			worker_listener,
			&objects,
			&status,
			&overhead_manager,
		)?;

		thread::spawn(move || worker_manager.run());

		let cache = PaperCache {
			objects,
			status,

			worker_manager: Arc::new(worker_sender),
			overhead_manager,
			
			#[cfg(all(feature = "key_value_pmem", any(feature = "enable_tiering_manager", feature = "sets_dram")))]
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
	/// 
	
/*
	pub fn get(&self, key: &K) -> Result<Vec<u8>, CacheError>
	{
		let hashed_key = self.hash_key(key);

		// Check DRAM tier first
		#[cfg(feature = "enable_tiering_manager")]
		if let Some(dram_object_ref) = self.tiering_manager.get_from_dram(&hashed_key) {
			if !dram_object_ref.is_expired() && dram_object_ref.key_matches(key) {
				self.status.incr_hits();
				self.broadcast(WorkerEvent::Get(hashed_key, true))?;
				let arc_val = dram_object_ref.data();
				//println!("CACHE: get for key {:?} from DRAM tier", key);
				//println!("CACHE: get for key {:?}: {:?}", key, arc_val.as_ref().clone());
				//println!("CACHE: get for key {:?} value size: {}", key, arc_val.as_ref().len());
				return Ok(arc_val.as_ref().to_vec());
			}
		}

		let result = match self.objects.get(&hashed_key) {
			Some(object) if object.key_matches(key) && !object.is_expired() => {
				self.status.incr_hits();
				// object.data() returns an Arc<V, Hybrid> — convert to Vec<u8>
				let arc_val = object.data();
				//println!("CACHE: get for key {:?}: {:?}", key, arc_val.as_ref().clone());
				Ok(arc_val.as_ref().to_vec())
			},

			_ => {
				self.status.incr_misses();
				Err(CacheError::KeyNotFound)
			},
		};

		self.broadcast(WorkerEvent::Get(hashed_key, result.is_ok()))?;

		// Optional: inspect the underlying bytes/tier of the returned value for debugging
		//println!("CACHE: get result for key {:?}: {:?} ", key, result);
		result
	}

	*/


	pub fn get(&self, key: &K) -> Result<Vec<u8>, CacheError>
	{
		let hashed_key = self.hash_key(key);

		// Check DRAM tier first
		#[cfg(all(feature = "enable_tiering_manager", not(feature = "hashtable_tiering")))]
		if let Some(dram_object_ref) = self.tiering_manager.get_from_dram(&hashed_key) {
			if !dram_object_ref.is_expired() && dram_object_ref.key_matches(key) {
				self.status.incr_hits();
				self.broadcast(WorkerEvent::Get(hashed_key, true))?;
				let arc_val = dram_object_ref.data();
				//println!("CACHE: get for key {:?} from DRAM tier", key);
				//println!("CACHE: get for key {:?}: {:?}", key, arc_val.as_ref().clone());
				//println!("CACHE: get for key {:?} value size: {}", key, arc_val.as_ref().len());
				return Ok(arc_val.as_ref().to_vec());
			}
		}

		#[cfg(all(feature = "enable_tiering_manager", feature = "hashtable_tiering"))]
		if let Some(dram_object_ref) = self.tiering_manager.get_from_dram(&hashed_key) {
			if !dram_object_ref.is_expired() && dram_object_ref.key_matches(key) {
				self.status.incr_hits();
				self.broadcast(WorkerEvent::Get(hashed_key, true))?;
				// Use data_as_bytes method to handle both PhysicalCopy and CxlReference
				//this could be incorrect.....
				return Ok(dram_object_ref.data_as_bytes());
			}
		}

		let result = match self.objects.get(&hashed_key) {
			Some(object) if object.key_matches(key) && !object.is_expired() => {
				self.status.incr_hits();
				// object.data() returns an Arc<V, Hybrid> — convert to Vec<u8>
				let arc_val = object.data();
				//println!("CACHE: get for key {:?}: {:?}", key, arc_val.as_ref().clone());
				Ok(arc_val.as_ref().to_vec())
			},

			_ => {
				self.status.incr_misses();
				Err(CacheError::KeyNotFound)
			},
		};

		self.broadcast(WorkerEvent::Get(hashed_key, result.is_ok()))?;

		// Optional: inspect the underlying bytes/tier of the returned value for debugging
		//println!("CACHE: get result for key {:?}: {:?} ", key, result);
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
	
	// not V but &[u8]?? 


	pub fn set(&self, key: K, value: &[u8], ttl: Option<u32>) -> Result<(), CacheError> 
	where
    	//V: AsRef<[u8]> + TypeSize,
		K: 'static + Eq + Hash + TypeSize + std::fmt::Debug,
	{

		//let t0 = rdtsc();
		let hashed_key = self.hash_key(&key);
		//let t1 = rdtsc();
		//PHASE_PRE_ALLOC.record(t1 - t0);


		//println!("CACHE: set called for key {:?} with value size {}", key, value.len());

		//allocate the object in the cache itself.... lets say pmem buffer
		
		//println!("CACHE: set called for key {:?} with value size {}", key, value.len());
		//let mut buf1: Vec<u8, Hybrid> = Vec::with_capacity_in(value.len(), Hybrid);
		//buf1.extend_from_slice(&value);

		//let buf: BufferPMEM = buf1.into_boxed_slice();


		#[cfg(feature = "sets_dram")]
		{
			self.tiering_manager.set_dram(hashed_key, key.clone(), value, ttl);
			return Ok(());
		}


		#[cfg(not(feature = "sets_dram"))]
		{

			/* 
			let layout = Layout::from_size_align(value.len(), 1).unwrap();

			let memory_block = Hybrid.allocate(layout)
				.map_err(|_| CacheError::Internal)?;

			// 1. Get the raw pointer to the start of the memory
			let memory_ptr = memory_block.as_ptr() as *mut u8;

			unsafe {
				ptr::copy_nonoverlapping(value.as_ptr(), memory_ptr, value.len());

				use std::arch::x86_64::{_mm_clflush, _mm_sfence};

				let cache_line_size = 64usize;
				let start = memory_ptr as usize;
				let end = start + value.len();
				// Round start down to a cache-line boundary so we flush the line
				// containing the first byte even if the allocation isn't aligned.
				let mut addr = start & !(cache_line_size - 1);
				while addr < end {
					_mm_clflush(addr as *const u8);
					addr += cache_line_size;
				}
				_mm_sfence();
			}

			let val_buf: BufferPMEM = unsafe {
				Box::from_raw_in(
					ptr::slice_from_raw_parts_mut(memory_ptr, value.len()),
					Hybrid
				)
			};
			

			 */

			//let val_buf: BufferPMEM = value.to_vec_in(Hybrid).into_boxed_slice();


			//pub fn clone_from_ref_in(src: &T, alloc: A) -> Box<T, A>

			// === phase 2: allocation of value buffer ===
			//let t2_start = rdtsc();
			//let mut val_buf: Vec<u8, Hybrid> = Vec::with_capacity_in(value.len(), Hybrid);
			let val_buf: BufferPMEM = Box::clone_from_ref_in(value, Hybrid);
			//let mut uninit_buf = Box::new_uninit_slice_in(value.len(), Hybrid);
			//let t2_end = rdtsc();
			//PHASE_ALLOC.record(t2_end - t2_start);

			// === phase 3: memcpy value into PMEM ===
			//let t3_start = rdtsc();
			//val_buf.extend_from_slice(value);
			//let val_buf: BufferPMEM = val_buf.into_boxed_slice();

			/*
			unsafe {
				let ptr = val_buf.as_ptr();
				let len = val_buf.len();

				if len > 0 {
					let cache_line_size = 64usize;
					let start = ptr as usize;
					let end = start + len;

					let mut addr = start & !(cache_line_size - 1);

					while addr < end {
						_mm_clflush(addr as *const u8);
						addr += cache_line_size;
					}

					_mm_sfence();
				}
			}
			*/

				

		
			//unsafe {
				// Copy directly into the uninitialized raw pointer
			//	ptr::copy_nonoverlapping(
			//		value.as_ptr(), 
			//		uninit_buf.as_mut_ptr() as *mut u8, 
			//		value.len()
			//	);
			//}
			//let val_buf: BufferPMEM = unsafe { uninit_buf.assume_init() };

			//let val_buf: BufferPMEM = { uninit_buf.assume_init() };
			//let t3_end = rdtsc();
			//PHASE_MEMCPY.record(t3_end - t3_start);

			//let key_buf: BufferPMEM = 

			//let mut buf1: Vec<u8, Hybrid> = Vec::with_capacity_in(key.len(), Hybrid); 
			//buf1.extend_from_slice(&key);
			//let key_buf: BufferPMEM = buf1.into_boxed_slice();

			//let key_buf: BufferPMEM = key.to_vec_in(Hybrid).into_boxed_slice();

			//the key should also be in pmem... this is stale or wrong... mut have changed it back??
			//let t4_start = rdtsc();
			let object = Object::new(key, val_buf, ttl);
			//let t4_end = rdtsc();
			//PHASE_POST.record(t4_end - t4_start);

			let base_size = self.overhead_manager.base_size(&object);
			let expiry = object.expiry();

			if base_size == 0 {
				return Err(CacheError::ZeroValueSize);
			}

			if self.status.exceeds_max_size(base_size) {
				return Err(CacheError::ExceedingValueSize);
			}

			self.status.incr_sets();

			//let t5_start = rdtsc();
			let old_object_info = self.objects
				.insert(hashed_key, object)
				.map(|old_object| {
					let base_size = self.overhead_manager.base_size(&old_object);
					let expiry = old_object.expiry();

					(base_size, expiry)
				});
			//let t5_end = rdtsc();
			//PHASE_INSERT.record(t5_end - t5_start);

			let base_size_delta = if let Some((old_object_size, _)) = old_object_info {
				base_size as i64 - old_object_size as i64
			} else {
				// the object is new, so increase the number of objects count
				self.status.incr_num_objects();
				base_size as i64
			};

			self.status.update_base_used_size(base_size_delta);
			//let t6_start = rdtsc();
			self.broadcast(WorkerEvent::Set(hashed_key, base_size, expiry, old_object_info))?;
			//let t6_end = rdtsc();
			//PHASE_SET_BROADCAST.record(t6_end - t6_start);
			Ok(())
		}

		//self.broadcast(WorkerEvent::Set(hashed_key, base_size, expiry, old_object_info))?;

		//Ok(())
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
			.get(&hashed_key)
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
	

	pub fn peek(&self, key: &K) -> Result<Arc<BufferPMEM>, CacheError> {
		let hashed_key = self.hash_key(key);

		match self.objects.get(&hashed_key) {
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

		let mut object = match self.objects.get_mut(&hashed_key) {
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

		match self.objects.get(&hashed_key) {
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


// Implementation for global_hashtable_pmem alone (without key_value_pmem)
// This case uses BufferDRAM for values (data in DRAM) but hashtable in PMEM
#[cfg(all(feature = "global_hashtable_pmem", not(feature = "key_value_pmem"), not(feature = "global_flatmap_pmem")))]
impl<K, S> PaperCache<K, BufferDRAM, S>
where
    K: 'static + Eq + Hash + TypeSize + std::fmt::Debug + Clone,
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

		// Global hashtable in PMEM using Hybrid allocator
		let objects = Arc::new(RwLock::new(HashMap::with_hasher_in(
			NoHasher::default(),
			Hybrid,
		)));

		let status = Arc::new(AtomicStatus::new(max_size, policies, policy)?);
		let overhead_manager = Arc::new(OverheadManager::new(&status));

		let (worker_sender, worker_listener) = unbounded();

		let mut worker_manager = WorkerManager::new(
			worker_listener,
			&objects,
			&status,
			&overhead_manager,
		)?;

		thread::spawn(move || worker_manager.run());

		let cache = PaperCache {
			objects,
			status,
			worker_manager: Arc::new(worker_sender),
			overhead_manager,
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


/* 
	// none instrumented get
	pub fn get(&self, key: &K) -> Result<Vec<u8>, CacheError> {
		let hashed_key = self.hash_key(key);

		let result = match self.objects.read().unwrap().get(&hashed_key) {
			Some(object) if object.key_matches(key) && !object.is_expired() => {
				self.status.incr_hits();
				let arc_val = object.data();
				Ok(arc_val.as_ref().to_vec())
			},
			_ => {
				self.status.incr_misses();
				Err(CacheError::KeyNotFound)
			},
		};

		self.broadcast(WorkerEvent::Get(hashed_key, result.is_ok()))?;
		result
	}

*/




	pub fn get(&self, key: &K) -> Result<Vec<u8>, CacheError> {
		let t0 = rdtsc();
		let hashed_key = self.hash_key(key);
		let t1 = rdtsc();
		PHASE_GET_HASH.record(t1 - t0);

		let guard = self.objects.read().unwrap();
		let t2 = rdtsc();
		PHASE_GET_LOCK.record(t2 - t1);

		let lookup = guard.get(&hashed_key);
		let t3 = rdtsc();
		PHASE_GET_LOOKUP.record(t3 - t2);

		let result = match lookup {
			Some(object) => {
				let matched = object.key_matches(key) && !object.is_expired();
				let t4 = rdtsc();
				PHASE_GET_VALIDATE.record(t4 - t3);

				if matched {
					self.status.incr_hits();
					let arc_val = object.data();
					let v = arc_val.as_ref().to_vec();
					let t5 = rdtsc();
					PHASE_GET_COPY.record(t5 - t4);
					Ok(v)
				} else {
					self.status.incr_misses();
					Err(CacheError::KeyNotFound)
				}
			}
			None => {
				self.status.incr_misses();
				Err(CacheError::KeyNotFound)
			}
		};

		drop(guard); // release read lock before broadcast — matches original lock scope

		let t6 = rdtsc();
		let br = self.broadcast(WorkerEvent::Get(hashed_key, result.is_ok()));
		let t7 = rdtsc();
		PHASE_GET_BROADCAST.record(t7 - t6);
		br?;

		result
	}


	pub fn set(&self, key: K, value: &[u8], ttl: Option<u32>) -> Result<(), CacheError> {
		let hashed_key = self.hash_key(&key);

		// Values stored in DRAM (BufferDRAM = Box<[u8]>)
		let val_buf: BufferDRAM = value.into();
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
			.write().unwrap().insert(hashed_key, object)
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
			.read().unwrap().get(&hashed_key)
			.is_some_and(|object| object.key_matches(key) && !object.is_expired())
	}

	pub fn peek(&self, key: &K) -> Result<Arc<BufferDRAM>, CacheError> {
		let hashed_key = self.hash_key(key);

		match self.objects.read().unwrap().get(&hashed_key) {
			Some(object) if object.key_matches(key) && !object.is_expired() =>
				Ok(object.data()),
			_ => Err(CacheError::KeyNotFound),
		}
	}

	pub fn ttl(&self, key: &K, ttl: Option<u32>) -> Result<(), CacheError> {
		let hashed_key = self.hash_key(key);

		let mut objects_guard = self.objects.write().unwrap();
		let object = match objects_guard.get_mut(&hashed_key) {
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

		match self.objects.read().unwrap().get(&hashed_key) {
			Some(object) if object.key_matches(key) && !object.is_expired() =>
				Ok(self.overhead_manager.total_size(&object)),
			_ => Err(CacheError::KeyNotFound),
		}
	}

	pub fn wipe(&self) -> Result<(), CacheError> {
		info!("Wiping cache");

		self.objects.write().unwrap().clear();
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


// Implementation for hashbrown_dram feature (hashbrown HashMap in DRAM)
// This allows direct performance comparison with global_hashtable_pmem
#[cfg(all(feature = "hashbrown_dram", not(feature = "key_value_pmem"), not(feature = "global_hashtable_pmem")))]
impl<K, S> PaperCache<K, BufferDRAM, S>
where
    K: 'static + Eq + Hash + TypeSize + std::fmt::Debug + Clone,
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

		// Hashbrown HashMap in DRAM (no Hybrid allocator)
		let objects = Arc::new(RwLock::new(HashMap::with_hasher(
			NoHasher::default(),
		)));

		let status = Arc::new(AtomicStatus::new(max_size, policies, policy)?);
		let overhead_manager = Arc::new(OverheadManager::new(&status));

		let (worker_sender, worker_listener) = unbounded();

		let mut worker_manager = WorkerManager::new(
			worker_listener,
			&objects,
			&status,
			&overhead_manager,
		)?;

		thread::spawn(move || worker_manager.run());

		let cache = PaperCache {
			objects,
			status,
			worker_manager: Arc::new(worker_sender),
			overhead_manager,
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

/* 
	// none instrumented get
	pub fn get(&self, key: &K) -> Result<Vec<u8>, CacheError> {
		let hashed_key = self.hash_key(key);

		let result = match self.objects.read().unwrap().get(&hashed_key) {
			Some(object) if object.key_matches(key) && !object.is_expired() => {
				self.status.incr_hits();
				let arc_val = object.data();
				Ok(arc_val.as_ref().to_vec())
			},
			_ => {
				self.status.incr_misses();
				Err(CacheError::KeyNotFound)
			},
		};

		self.broadcast(WorkerEvent::Get(hashed_key, result.is_ok()))?;
		result
	}

	*/
	

	pub fn get(&self, key: &K) -> Result<Vec<u8>, CacheError> {
		let t0 = rdtsc();
		let hashed_key = self.hash_key(key);
		let t1 = rdtsc();
		PHASE_GET_HASH.record(t1 - t0);

		let guard = self.objects.read().unwrap();
		let t2 = rdtsc();
		PHASE_GET_LOCK.record(t2 - t1);

		let lookup = guard.get(&hashed_key);
		let t3 = rdtsc();
		PHASE_GET_LOOKUP.record(t3 - t2);

		let result = match lookup {
			Some(object) => {
				let matched = object.key_matches(key) && !object.is_expired();
				let t4 = rdtsc();
				PHASE_GET_VALIDATE.record(t4 - t3);

				if matched {
					self.status.incr_hits();
					let arc_val = object.data();
					let v = arc_val.as_ref().to_vec();
					let t5 = rdtsc();
					PHASE_GET_COPY.record(t5 - t4);
					Ok(v)
				} else {
					self.status.incr_misses();
					Err(CacheError::KeyNotFound)
				}
			}
			None => {
				self.status.incr_misses();
				Err(CacheError::KeyNotFound)
			}
		};

		drop(guard); // release read lock before broadcast — matches original lock scope

		let t6 = rdtsc();
		let br = self.broadcast(WorkerEvent::Get(hashed_key, result.is_ok()));
		let t7 = rdtsc();
		PHASE_GET_BROADCAST.record(t7 - t6);
		br?;

		result
	}

	pub fn set(&self, key: K, value: &[u8], ttl: Option<u32>) -> Result<(), CacheError> {
		let hashed_key = self.hash_key(&key);

		// Values stored in DRAM (BufferDRAM = Box<[u8]>)
		let val_buf: BufferDRAM = value.into();
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
			.write().unwrap().insert(hashed_key, object)
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
			.read().unwrap().get(&hashed_key)
			.is_some_and(|object| object.key_matches(key) && !object.is_expired())
	}

	pub fn peek(&self, key: &K) -> Result<Arc<BufferDRAM>, CacheError> {
		let hashed_key = self.hash_key(key);

		match self.objects.read().unwrap().get(&hashed_key) {
			Some(object) if object.key_matches(key) && !object.is_expired() =>
				Ok(object.data()),
			_ => Err(CacheError::KeyNotFound),
		}
	}

	pub fn ttl(&self, key: &K, ttl: Option<u32>) -> Result<(), CacheError> {
		let hashed_key = self.hash_key(key);

		let mut objects_guard = self.objects.write().unwrap();
		let object = match objects_guard.get_mut(&hashed_key) {
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

		match self.objects.read().unwrap().get(&hashed_key) {
			Some(object) if object.key_matches(key) && !object.is_expired() =>
				Ok(self.overhead_manager.total_size(&object)),
			_ => Err(CacheError::KeyNotFound),
		}
	}

	pub fn wipe(&self) -> Result<(), CacheError> {
		info!("Wiping cache");

		self.objects.write().unwrap().clear();
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


#[cfg(all(feature = "key_value_pmem", feature = "global_hashtable_pmem", not(feature = "global_flatmap_pmem")))]
impl<K, S> PaperCache<K, BufferPMEM, S>
where
    K: 'static + Eq + Hash + TypeSize + std::fmt::Debug + Clone, //note added Debug for logging might impact perf thoooo
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

		//let objects = Arc::new(DashMap::with_hasher(NoHasher::default()));

		#[cfg(not(feature = "global_hashtable_pmem"))]
		let objects = Arc::new(RwLock::new(HashMap::with_hasher(
			NoHasher::default(),
		)));

		#[cfg(feature = "global_hashtable_pmem")]
		let objects = Arc::new(RwLock::new(HashMap::with_hasher_in(
			NoHasher::default(),
			Hybrid,
		)));

		let status = Arc::new(AtomicStatus::new(max_size, policies, policy)?);
		let overhead_manager = Arc::new(OverheadManager::new(&status));

		#[cfg(feature = "enable_tiering_manager")]
		let tiering_manager = {
			// Create tiering manager with default DRAM threshold at 20% of max_size
			let mut tiering_config = tiering::TieringConfig::default();
			//tiering_config.dram_threshold = (max_size as f64 * 0.2) as u64;
			Arc::new(TieringManager::new(tiering_config))
		};

		let (worker_sender, worker_listener) = unbounded();

		#[cfg(feature = "enable_tiering_manager")]
		let mut worker_manager = WorkerManager::new(
			worker_listener,
			&objects,
			&status,
			&overhead_manager,
			&tiering_manager,
		)?;

		#[cfg(not(feature = "enable_tiering_manager"))]
		let mut worker_manager = WorkerManager::new(
			worker_listener,
			&objects,
			&status,
			&overhead_manager,
		)?;

		thread::spawn(move || worker_manager.run());

		let cache = PaperCache {
			objects,
			status,

			worker_manager: Arc::new(worker_sender),
			overhead_manager,
			
			#[cfg(feature = "enable_tiering_manager")]
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
	/// 
	

	pub fn get(&self, key: &K) -> Result<Vec<u8>, CacheError>
	{
		let hashed_key = self.hash_key(key);

		// Check DRAM tier first
		#[cfg(feature = "enable_tiering_manager")]
		if let Some(dram_object_ref) = self.tiering_manager.get_from_dram(&hashed_key) {
			if !dram_object_ref.is_expired() && dram_object_ref.key_matches(key) {
				self.status.incr_hits();
				self.broadcast(WorkerEvent::Get(hashed_key, true))?;
				let arc_val = dram_object_ref.data();
				//println!("CACHE: get for key {:?} from DRAM tier", key);
				//println!("CACHE: get for key {:?}: {:?}", key, arc_val.as_ref().clone());
				//println!("CACHE: get for key {:?} value size: {}", key, arc_val.as_ref().len());
				return Ok(arc_val.as_ref().to_vec());
			}

		}

		let result = match self.objects.read().unwrap().get(&hashed_key) {
			Some(object) if object.key_matches(key) && !object.is_expired() => {
				self.status.incr_hits();
				// object.data() returns an Arc<V, Hybrid> — convert to Vec<u8>
				let arc_val = object.data();
				//println!("CACHE: get for key {:?}: {:?}", key, arc_val.as_ref().clone());
				Ok(arc_val.as_ref().to_vec())
			},

			_ => {
				self.status.incr_misses();
				Err(CacheError::KeyNotFound)
			},
		};

		self.broadcast(WorkerEvent::Get(hashed_key, result.is_ok()))?;

		// Optional: inspect the underlying bytes/tier of the returned value for debugging
		//println!("CACHE: get result for key {:?}: {:?} ", key, result);
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
	
	// not V but &[u8]?? 

	pub fn set(&self, key: K, value: &[u8], ttl: Option<u32>) -> Result<(), CacheError> 
	where
    	//V: AsRef<[u8]> + TypeSize,
		K: 'static + Eq + Hash + TypeSize + std::fmt::Debug,
	{

		let hashed_key = self.hash_key(&key);

		//println!("CACHE: set called for key {:?} with value size {}", key, value.len());

		//allocate the object in the cache itself.... lets say pmem buffer
		
		//println!("CACHE: set called for key {:?} with value size {}", key, value.len());
		//let mut buf1: Vec<u8, Hybrid> = Vec::with_capacity_in(value.len(), Hybrid);
		//buf1.extend_from_slice(&value);

		//let buf: BufferPMEM = buf1.into_boxed_slice();

		let val_buf: BufferPMEM = value.to_vec_in(Hybrid).into_boxed_slice();

		//let key_buf: BufferPMEM = 

		//let mut buf1: Vec<u8, Hybrid> = Vec::with_capacity_in(key.len(), Hybrid); 
		//buf1.extend_from_slice(&key);
		//let key_buf: BufferPMEM = buf1.into_boxed_slice();

		//let key_buf: BufferPMEM = key.to_vec_in(Hybrid).into_boxed_slice();

		let object = Object::new(key, val_buf, ttl);

		//should =turn this into pmem buffer .... 


		//let object = Object::new(key, value, ttl);
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
			//.insert(hashed_key, object)
			.write().unwrap().insert(hashed_key, object)
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
			//.get(&hashed_key)
			.read().unwrap().get(&hashed_key)
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
	

	pub fn peek(&self, key: &K) -> Result<Arc<BufferPMEM>, CacheError> {
		let hashed_key = self.hash_key(key);

		//match self.objects.get(&hashed_key) {
		match self.objects.read().unwrap().get(&hashed_key) {
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

		//let mut object = match self.objects.get_mut(&hashed_key) {
		//let mut object = match self.objects.write().unwrap().get_mut(&hashed_key) {
		let mut objects_guard = self.objects.write().unwrap();
		let object = match objects_guard.get_mut(&hashed_key) {
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

		//match self.objects.get(&hashed_key) {
		match self.objects.read().unwrap().get(&hashed_key) {
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

		//self.objects.clear();
		self.objects.write().unwrap().clear();
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

#[cfg(all(feature = "global_flatmap_pmem", not(feature = "key_value_pmem"), not(feature = "global_hashtable_pmem")))]
impl<K, S> PaperCache<K, BufferDRAM, S>
where
    K: 'static + Eq + Hash + TypeSize + std::fmt::Debug + Clone + Default,
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

		// Global flatmap in PMEM using Hybrid allocator
		//hardcoded estimated objects for PMEM global flatmap for now...
		let estimated_objects = 1_500_000 as usize;
		let capacity = (estimated_objects * 2).next_power_of_two();
		let objects = Arc::new(RwLock::new(crate::flatmap::FlatMapWithHasher::with_capacity_hasher_in_unchecked(
			capacity,
			NoHasher::default(),
			Hybrid,
		)));

		let status = Arc::new(AtomicStatus::new(max_size, policies, policy)?);
		let overhead_manager = Arc::new(OverheadManager::new(&status));

		let (worker_sender, worker_listener) = unbounded();

		let mut worker_manager = WorkerManager::new(
			worker_listener,
			&objects,
			&status,
			&overhead_manager,
		)?;

		thread::spawn(move || worker_manager.run());

		let cache = PaperCache {
			objects,
			status,
			worker_manager: Arc::new(worker_sender),
			overhead_manager,
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

	pub fn get(&self, key: &K) -> Result<Vec<u8>, CacheError> {
		let hashed_key = self.hash_key(key);

		let result = match self.objects.read().unwrap().get(&hashed_key) {
			Some(object) if object.key_matches(key) && !object.is_expired() => {
				self.status.incr_hits();
				let arc_val = object.data();
				Ok(arc_val.as_ref().to_vec())
			},
			_ => {
				self.status.incr_misses();
				Err(CacheError::KeyNotFound)
			},
		};

		self.broadcast(WorkerEvent::Get(hashed_key, result.is_ok()))?;
		result
	}

	pub fn set(&self, key: K, value: &[u8], ttl: Option<u32>) -> Result<(), CacheError> {
		let hashed_key = self.hash_key(&key);

		// Values stored in DRAM (BufferDRAM = Box<[u8]>)
		let val_buf: BufferDRAM = value.into();
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
			.write().unwrap().insert(hashed_key, object)
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
			.read().unwrap().get(&hashed_key)
			.is_some_and(|object| object.key_matches(key) && !object.is_expired())
	}

	pub fn peek(&self, key: &K) -> Result<Arc<BufferDRAM>, CacheError> {
		let hashed_key = self.hash_key(key);

		match self.objects.read().unwrap().get(&hashed_key) {
			Some(object) if object.key_matches(key) && !object.is_expired() =>
				Ok(object.data()),
			_ => Err(CacheError::KeyNotFound),
		}
	}

	pub fn ttl(&self, key: &K, ttl: Option<u32>) -> Result<(), CacheError> {
		let hashed_key = self.hash_key(key);

		let mut objects_guard = self.objects.write().unwrap();
		let object = match objects_guard.get_mut(&hashed_key) {
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

		match self.objects.read().unwrap().get(&hashed_key) {
			Some(object) if object.key_matches(key) && !object.is_expired() =>
				Ok(self.overhead_manager.total_size(&object)),
			_ => Err(CacheError::KeyNotFound),
		}
	}

	pub fn wipe(&self) -> Result<(), CacheError> {
		info!("Wiping cache");

		self.objects.write().unwrap().clear();
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


#[cfg(all(feature = "global_flatmap_dram", feature = "all_dram"))]
impl<K, S> PaperCache<K, BufferDRAM, S>
where
	K: 'static + Eq + Hash + TypeSize + std::fmt::Debug + Clone + Default,
	S: Default + Clone + BuildHasher,
{
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

		let estimated_objects = 1_500_000 as usize;
		let capacity = (estimated_objects * 2).next_power_of_two();
		let objects = Arc::new(RwLock::new(crate::flatmap::FlatMapWithHasher::with_capacity_and_hasher_unchecked(
			capacity,
			NoHasher::default(),
		)));

		let status = Arc::new(AtomicStatus::new(max_size, policies, policy)?);
		let overhead_manager = Arc::new(OverheadManager::new(&status));

		let (worker_sender, worker_listener) = unbounded();

		let mut worker_manager = WorkerManager::new(
			worker_listener,
			&objects,
			&status,
			&overhead_manager,
		)?;

		thread::spawn(move || worker_manager.run());

		let cache = PaperCache {
			objects,
			status,
			worker_manager: Arc::new(worker_sender),
			overhead_manager,
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

	pub fn get(&self, key: &K) -> Result<Vec<u8>, CacheError> {
		let hashed_key = self.hash_key(key);

		let result = match self.objects.read().unwrap().get(&hashed_key) {
			Some(object) if object.key_matches(key) && !object.is_expired() => {
				self.status.incr_hits();
				self.broadcast(WorkerEvent::Get(hashed_key, true))?;
				let arc_val = object.data();
				Ok(arc_val.as_ref().to_vec())
			},
			_ => {
				self.status.incr_misses();
				Err(CacheError::KeyNotFound)
			},
		};

		result
	}

	pub fn set(&self, key: K, value: &[u8], ttl: Option<u32>) -> Result<(), CacheError> {
		let hashed_key = self.hash_key(&key);

		let val_buf: Box<[u8]> = value.to_vec().into_boxed_slice();
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
			.write().unwrap().insert_unchecked(hashed_key, object)
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
			.read().unwrap().get(&hashed_key)
			.is_some_and(|object| object.key_matches(key) && !object.is_expired())
	}

	pub fn peek(&self, key: &K) -> Result<Arc<BufferDRAM>, CacheError> {
		let hashed_key = self.hash_key(key);

		match self.objects.read().unwrap().get(&hashed_key) {
			Some(object) if object.key_matches(key) && !object.is_expired() =>
				Ok(object.data()),
			_ => Err(CacheError::KeyNotFound),
		}
	}

	pub fn ttl(&self, key: &K, ttl: Option<u32>) -> Result<(), CacheError> {
		let hashed_key = self.hash_key(key);

		let mut objects_guard = self.objects.write().unwrap();
		let object = match objects_guard.get_mut(&hashed_key) {
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

		match self.objects.read().unwrap().get(&hashed_key) {
			Some(object) if object.key_matches(key) && !object.is_expired() =>
				Ok(self.overhead_manager.total_size(&object)),
			_ => Err(CacheError::KeyNotFound),
		}
	}

	pub fn wipe(&self) -> Result<(), CacheError> {
		info!("Wiping cache");

		let mut objects_guard = self.objects.write().unwrap();
		let keys: Vec<_> = objects_guard.iter().map(|(k, _)| *k).collect();
		for key in keys {
			objects_guard.remove_unchecked(&key);
		}
		drop(objects_guard);
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


#[cfg(all(feature = "global_flatmap_pmem", feature = "key_value_pmem"))]
impl<K, S> PaperCache<K, BufferPMEM, S>
where
	K: 'static + Eq + Hash + TypeSize + std::fmt::Debug + Clone + Default,
	S: Default + Clone + BuildHasher,
{
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
		
		//hardcoded estimated objects for PMEM global flatmap for now...
		let estimated_objects = 1_500_000 as usize;
		let capacity = (estimated_objects * 2).next_power_of_two();
		let objects = Arc::new(RwLock::new(crate::flatmap::FlatMapWithHasher::with_capacity_hasher_in_unchecked(
			capacity,
			NoHasher::default(),
			Hybrid,
		)));

		let status = Arc::new(AtomicStatus::new(max_size, policies, policy)?);
		let overhead_manager = Arc::new(OverheadManager::new(&status));

		let (worker_sender, worker_listener) = unbounded();

		let mut worker_manager = WorkerManager::new(
			worker_listener,
			&objects,
			&status,
			&overhead_manager,
		)?;

		thread::spawn(move || worker_manager.run());

		let cache = PaperCache {
			objects,
			status,
			worker_manager: Arc::new(worker_sender),
			overhead_manager,
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

	pub fn get(&self, key: &K) -> Result<Vec<u8>, CacheError> {
		let hashed_key = self.hash_key(key);

		let result = match self.objects.read().unwrap().get(&hashed_key) {
			Some(object) if object.key_matches(key) && !object.is_expired() => {
				self.status.incr_hits();
				let arc_val = object.data();
				Ok(arc_val.as_ref().to_vec())
			},
			_ => {
				self.status.incr_misses();
				Err(CacheError::KeyNotFound)
			},
		};

		result
	}

	pub fn set(&self, key: K, value: &[u8], ttl: Option<u32>) -> Result<(), CacheError> {
		let hashed_key = self.hash_key(&key);

		let val_buf: BufferPMEM = value.to_vec_in(Hybrid).into_boxed_slice();
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
			.write().unwrap().insert(hashed_key, object)
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
			.read().unwrap().get(&hashed_key)
			.is_some_and(|object| object.key_matches(key) && !object.is_expired())
	}

	pub fn peek(&self, key: &K) -> Result<Arc<BufferPMEM>, CacheError> {
		let hashed_key = self.hash_key(key);

		match self.objects.read().unwrap().get(&hashed_key) {
			Some(object) if object.key_matches(key) && !object.is_expired() =>
				Ok(object.data()),
			_ => Err(CacheError::KeyNotFound),
		}
	}

	pub fn ttl(&self, key: &K, ttl: Option<u32>) -> Result<(), CacheError> {
		let hashed_key = self.hash_key(key);

		let mut objects_guard = self.objects.write().unwrap();
		let object = match objects_guard.get_mut(&hashed_key) {
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

		match self.objects.read().unwrap().get(&hashed_key) {
			Some(object) if object.key_matches(key) && !object.is_expired() =>
				Ok(self.overhead_manager.total_size(&object)),
			_ => Err(CacheError::KeyNotFound),
		}
	}

	pub fn wipe(&self) -> Result<(), CacheError> {
		info!("Wiping cache");

		let mut objects_guard = self.objects.write().unwrap();
		let keys: Vec<_> = objects_guard.iter().map(|(k, _)| *k).collect();
		for key in keys {
			objects_guard.remove_unchecked(&key);
		}
		drop(objects_guard);
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


#[cfg(all(any(feature = "global_hashtable_pmem", feature = "hashbrown_dram"), not(feature = "global_flatmap_dram"), not(feature = "global_flatmap_pmem")))]
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


// FlatMap erase function - uses get/remove instead of entry API
#[cfg(any(feature = "global_flatmap_dram", feature = "global_flatmap_pmem"))]
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

			let objects_guard = objects.read().unwrap();
			let Some((_key, _object)) = objects_guard.iter().next() else {
				error!("Object store is empty with non-zero used size");
				return Err(CacheError::Internal);
			};

			*_key
		},
	};

	// Check if key exists and validate if we have the original key
	let mut objects_lock = objects.write().unwrap();
	
	// Get the object to check if it exists and matches the key
	let object_ref = objects_lock.get(&hashed_key).ok_or(CacheError::KeyNotFound)?;
	
	// If we have the original key, validate it matches
	if let Some(EraseKey::Original(key, _)) = &maybe_key {
		if !object_ref.key_matches(key) {
			return Err(CacheError::KeyNotFound);
		}
	}
	
	// Remove the object using unchecked remove (doesn't require Clone/Default)
	let object = objects_lock.remove_unchecked(&hashed_key).ok_or(CacheError::KeyNotFound)?;
	let base_size = overhead_manager.base_size(&object) as i64;

	status.update_base_used_size(-base_size);
	status.decr_num_objects();

	match !object.is_expired() {
		true => Ok((hashed_key, object)),
		false => Err(CacheError::KeyNotFound),
	}
}













#[cfg(all(not(any(feature = "global_hashtable_pmem", feature = "global_flatmap_dram", feature = "global_flatmap_pmem", feature = "hashbrown_dram"))))]
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

/// Single-instance, segmented-LRU hybrid cache: one `PaperCache<K,
/// TieredBuffer>` running `PaperPolicy::LruHybrid`, in contrast with
/// `hybridcache`'s [`crate::hybridcache::S3FifoHybridCache`] (two composed
/// `PaperCache` instances). See the `lru_hybrid_cache` module docs and
/// `CLAUDE.md`'s `lru_hybrid_cache` plan for the full design.
///
/// Admission always lands in the fast tier; fast-tier pressure demotes the
/// LRU tail to the slow tier; a slow-tier access promotes it back to the
/// fast tier; terminal evictions (once overall `max_size` is exceeded) only
/// ever remove the slow-tier LRU tail. Every migration physically
/// reallocates the object's bytes (see [`TieredBuffer`] and
/// `Object::set_data`) — a key is never present in both tiers at once.
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
		if max_size == 0 {
			return Err(CacheError::ZeroCacheSize);
		}

		let fast_capacity = fast_tier_size.to_bytes();

		if fast_capacity == 0 || fast_capacity > max_size {
			return Err(CacheError::InvalidFastTierSize);
		}

		let policies = [PaperPolicy::LruHybrid];

		let objects = Arc::new(DashMap::with_hasher(NoHasher::default()));
		let status = Arc::new(AtomicStatus::new(max_size, &policies, PaperPolicy::LruHybrid)?);
		let overhead_manager = Arc::new(OverheadManager::new(&status));

		// Requirement: fast-tier size is runtime-configurable (not baked
		// into the policy string, unlike e.g. `TwoQ`/`SThreeFifo`), so the
		// requested capacity is recorded on the shared status immediately;
		// `init_policy_stack`'s 20%-of-max_size default (see
		// `policy_stack/mod.rs`) is overridden below via `ResizeFastTier`.
		status.set_fast_tier_capacity(fast_capacity);

		// Reallocates a value into the target tier's representation. Must
		// preserve byte length exactly: both `status.base_used_size` and
		// `LruHybridStack`'s own per-key size bookkeeping assume a
		// migration never changes an object's accounted size.
		let migrate: Box<dyn Fn(&TieredBuffer, Tier) -> TieredBuffer + Send + Sync> =
			Box::new(|buffer, tier| match tier {
				Tier::Fast => TieredBuffer::new_fast(buffer.as_ref()),
				Tier::Slow => TieredBuffer::new_slow(buffer.as_ref()),
			});

		let (worker_sender, worker_listener) = unbounded();

		let mut worker_manager = WorkerManager::new_with_tier_migration(
			worker_listener,
			&objects,
			&status,
			&overhead_manager,
			migrate,
		)?;

		thread::spawn(move || worker_manager.run());

		let cache = PaperCache {
			objects,
			status,
			worker_manager: Arc::new(worker_sender),
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

		let result = match self.objects.get(&hashed_key) {
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

	/// Sets the supplied key and value in the cache. Admission always lands
	/// in the fast tier — including when overwriting a key currently in the
	/// slow tier, which this promotes back to fast (see `LruHybridStack::insert`).
	/// Returns a [`CacheError`] if the value size is zero or larger than the
	/// cache's maximum size.
	pub fn set(&self, key: K, value: &[u8], ttl: Option<u32>) -> Result<(), CacheError> {
		let hashed_key = self.hash_key(&key);

		let val_buf = TieredBuffer::new_fast(value);
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
			.get(&hashed_key)
			.is_some_and(|object| object.key_matches(key) && !object.is_expired())
	}

	/// Gets (peeks) the value associated with the supplied key without
	/// altering any of the cache's internal queues (including tier — a peek
	/// never triggers a promotion). If the key was not found in the cache,
	/// returns a [`CacheError`].
	pub fn peek(&self, key: &K) -> Result<Arc<TieredBuffer>, CacheError> {
		let hashed_key = self.hash_key(key);

		match self.objects.get(&hashed_key) {
			Some(object) if object.key_matches(key) && !object.is_expired() =>
				Ok(object.data()),

			_ => Err(CacheError::KeyNotFound),
		}
	}

	/// Sets the TTL associated with the supplied key.
	/// If the key was not found in the cache, returns a [`CacheError`].
	pub fn ttl(&self, key: &K, ttl: Option<u32>) -> Result<(), CacheError> {
		let hashed_key = self.hash_key(key);

		let mut object = match self.objects.get_mut(&hashed_key) {
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

		match self.objects.get(&hashed_key) {
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
	/// fast-tier budget — see [`Self::set_fast_tier_size`].
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
	/// immediate demotions (see `LruHybridStack::settle_fast_tier`).
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

	/// Returns a point-in-time snapshot of `lru_hybrid_cache` statistics.
	#[must_use]
	pub fn lru_hybrid_stats(&self) -> LruHybridStats {
		self.status.lru_hybrid_stats()
	}

	/// Returns which tier `key` currently lives in, or `None` if the key
	/// isn't present (or has expired). Useful for tests/diagnostics — unlike
	/// `S3FifoHybridCache`'s `has_in_dram`/`has_in_pmem` pair, there's only
	/// one object map here, so tier is a property read off the object itself.
	#[must_use]
	pub fn tier_of(&self, key: &K) -> Option<Tier> {
		let hashed_key = self.hash_key(key);

		self.objects.get(&hashed_key).and_then(|object| {
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

/// Single-instance, segmented-LFU hybrid cache: one `PaperCache<K,
/// TieredBuffer>` running `PaperPolicy::LfuHybrid`, in contrast with
/// `hybridcache`'s [`crate::hybridcache::S3FifoHybridCache`] (two composed
/// `PaperCache` instances). See the `lfu_hybrid_cache` module docs for the
/// full design; this impl block mirrors `lru_hybrid_cache`'s mechanically
/// (same shape: `new`, `get`, `set`, ...) — the only behavioral difference
/// lives in `LfuHybridStack`, not here.
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
/// `compile_error!` guard) since both define this same inherent-method
/// impl block on the identical `PaperCache<K, TieredBuffer, S>` type.
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
		if max_size == 0 {
			return Err(CacheError::ZeroCacheSize);
		}

		let fast_capacity = fast_tier_size.to_bytes();

		if fast_capacity == 0 || fast_capacity > max_size {
			return Err(CacheError::InvalidFastTierSize);
		}

		let policies = [PaperPolicy::LfuHybrid];

		let objects = Arc::new(DashMap::with_hasher(NoHasher::default()));
		let status = Arc::new(AtomicStatus::new(max_size, &policies, PaperPolicy::LfuHybrid)?);
		let overhead_manager = Arc::new(OverheadManager::new(&status));

		// Requirement: fast-tier size is runtime-configurable (not baked
		// into the policy string), so the requested capacity is recorded on
		// the shared status immediately; `init_policy_stack`'s
		// 20%-of-max_size default (see `policy_stack/mod.rs`) is overridden
		// below via `ResizeFastTier`.
		status.set_fast_tier_capacity(fast_capacity);

		// Reallocates a value into the target tier's representation. Must
		// preserve byte length exactly: both `status.base_used_size` and
		// `LfuHybridStack`'s own per-key size bookkeeping assume a
		// migration never changes an object's accounted size.
		let migrate: Box<dyn Fn(&TieredBuffer, Tier) -> TieredBuffer + Send + Sync> =
			Box::new(|buffer, tier| match tier {
				Tier::Fast => TieredBuffer::new_fast(buffer.as_ref()),
				Tier::Slow => TieredBuffer::new_slow(buffer.as_ref()),
			});

		let (worker_sender, worker_listener) = unbounded();

		let mut worker_manager = WorkerManager::new_with_tier_migration(
			worker_listener,
			&objects,
			&status,
			&overhead_manager,
			migrate,
		)?;

		thread::spawn(move || worker_manager.run());

		let cache = PaperCache {
			objects,
			status,
			worker_manager: Arc::new(worker_sender),
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

		let result = match self.objects.get(&hashed_key) {
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

	/// Sets the supplied key and value in the cache. Admission always lands
	/// in the fast tier — including when overwriting a key currently in the
	/// slow tier, which this treats as an access (see `LfuHybridStack::insert`).
	/// Returns a [`CacheError`] if the value size is zero or larger than the
	/// cache's maximum size.
	pub fn set(&self, key: K, value: &[u8], ttl: Option<u32>) -> Result<(), CacheError> {
		let hashed_key = self.hash_key(&key);

		// A brand-new key built once the fast tier has genuinely reached
		// capacity (`LfuHybridStack`'s one-time admission latch, mirrored
		// onto `AtomicStatus` since this thread has no direct access to the
		// worker-owned stack) goes straight into `TieredBuffer::new_slow` --
		// this is what the stack would decide anyway, so building it fast
		// first would only cost a synchronous DRAM write immediately
		// followed by an async PMEM correction. An *existing* key is never
		// affected by this check regardless of its current tier: re-setting
		// one is an access (see this method's doc comment above), which may
		// or may not promote it, and only the stack can decide that.
		let is_new = !self.objects.contains_key(&hashed_key);

		let val_buf = if is_new && self.status.lfu_hybrid_admission_latched() {
			TieredBuffer::new_slow(value)
		} else {
			TieredBuffer::new_fast(value)
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
			.get(&hashed_key)
			.is_some_and(|object| object.key_matches(key) && !object.is_expired())
	}

	/// Gets (peeks) the value associated with the supplied key without
	/// altering any of the cache's internal queues (including tier — a peek
	/// never triggers a promotion). If the key was not found in the cache,
	/// returns a [`CacheError`].
	pub fn peek(&self, key: &K) -> Result<Arc<TieredBuffer>, CacheError> {
		let hashed_key = self.hash_key(key);

		match self.objects.get(&hashed_key) {
			Some(object) if object.key_matches(key) && !object.is_expired() =>
				Ok(object.data()),

			_ => Err(CacheError::KeyNotFound),
		}
	}

	/// Sets the TTL associated with the supplied key.
	/// If the key was not found in the cache, returns a [`CacheError`].
	pub fn ttl(&self, key: &K, ttl: Option<u32>) -> Result<(), CacheError> {
		let hashed_key = self.hash_key(key);

		let mut object = match self.objects.get_mut(&hashed_key) {
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

		match self.objects.get(&hashed_key) {
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
	/// fast-tier budget — see [`Self::set_fast_tier_size`].
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
	/// immediate demotions (see `LfuHybridStack::settle_fast_tier`).
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

	/// Returns a point-in-time snapshot of `lfu_hybrid_cache` statistics.
	#[must_use]
	pub fn lfu_hybrid_stats(&self) -> LfuHybridStats {
		self.status.lfu_hybrid_stats()
	}

	/// Returns which tier `key` currently lives in, or `None` if the key
	/// isn't present (or has expired).
	#[must_use]
	pub fn tier_of(&self, key: &K) -> Option<Tier> {
		let hashed_key = self.hash_key(key);

		self.objects.get(&hashed_key).and_then(|object| {
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

/// Single-instance, segmented-2Q hybrid cache: one `PaperCache<K,
/// TieredBuffer>` running `PaperPolicy::TwoQHybrid`, in contrast with
/// `hybridcache`'s [`crate::hybridcache::S3FifoHybridCache`] (two composed
/// `PaperCache` instances). See the `two_q_hybrid_cache` module docs for the
/// full design; this impl block mirrors `lru_hybrid_cache`'s/
/// `lfu_hybrid_cache`'s mechanically (same shape: `new`, `get`, `set`, ...)
/// with two differences: `new`/`with_hasher` take an extra `k_in` parameter
/// (this policy's FIFO-queue byte budget is embedded like plain
/// `PaperPolicy::TwoQ`'s, not purely runtime-configurable), and `set()`
/// admits into the slow tier rather than the fast tier.
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
/// `lib.rs`'s `compile_error!` guards) since all three define this same
/// inherent-method impl block on the identical `PaperCache<K, TieredBuffer,
/// S>` type.
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
		if max_size == 0 {
			return Err(CacheError::ZeroCacheSize);
		}

		if !(0.0..=1.0).contains(&k_in) {
			return Err(CacheError::InvalidPolicy);
		}

		let fast_capacity = fast_tier_size.to_bytes();

		if fast_capacity == 0 || fast_capacity > max_size {
			return Err(CacheError::InvalidFastTierSize);
		}

		let policies = [PaperPolicy::TwoQHybrid(k_in)];

		let objects = Arc::new(DashMap::with_hasher(NoHasher::default()));
		let status = Arc::new(AtomicStatus::new(max_size, &policies, PaperPolicy::TwoQHybrid(k_in))?);
		let overhead_manager = Arc::new(OverheadManager::new(&status));

		// Requirement: the main queue's fast-tier size is runtime-configurable
		// (not baked into the policy string, unlike `k_in`), so the requested
		// capacity is recorded on the shared status immediately;
		// `init_policy_stack`'s 20%-of-max_size default (see
		// `policy_stack/mod.rs`) is overridden below via `ResizeFastTier`.
		status.set_fast_tier_capacity(fast_capacity);

		// Reallocates a value into the target tier's representation. Must
		// preserve byte length exactly: both `status.base_used_size` and
		// `TwoQHybridStack`'s own per-key size bookkeeping assume a
		// migration never changes an object's accounted size.
		let migrate: Box<dyn Fn(&TieredBuffer, Tier) -> TieredBuffer + Send + Sync> =
			Box::new(|buffer, tier| match tier {
				Tier::Fast => TieredBuffer::new_fast(buffer.as_ref()),
				Tier::Slow => TieredBuffer::new_slow(buffer.as_ref()),
			});

		let (worker_sender, worker_listener) = unbounded();

		let mut worker_manager = WorkerManager::new_with_tier_migration(
			worker_listener,
			&objects,
			&status,
			&overhead_manager,
			migrate,
		)?;

		thread::spawn(move || worker_manager.run());

		let cache = PaperCache {
			objects,
			status,
			worker_manager: Arc::new(worker_sender),
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

		let result = match self.objects.get(&hashed_key) {
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

	/// Sets the supplied key and value in the cache. Admission always lands
	/// in the slow tier — every new object starts in the one-access FIFO
	/// queue (see `TwoQHybridStack::insert`); only a re-access promotes it
	/// to the fast tier. Returns a [`CacheError`] if the value size is zero
	/// or larger than the cache's maximum size.
	pub fn set(&self, key: K, value: &[u8], ttl: Option<u32>) -> Result<(), CacheError> {
		let hashed_key = self.hash_key(&key);

		let val_buf = TieredBuffer::new_slow(value);
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
			.get(&hashed_key)
			.is_some_and(|object| object.key_matches(key) && !object.is_expired())
	}

	/// Gets (peeks) the value associated with the supplied key without
	/// altering any of the cache's internal queues (including tier — a peek
	/// never triggers a promotion). If the key was not found in the cache,
	/// returns a [`CacheError`].
	pub fn peek(&self, key: &K) -> Result<Arc<TieredBuffer>, CacheError> {
		let hashed_key = self.hash_key(key);

		match self.objects.get(&hashed_key) {
			Some(object) if object.key_matches(key) && !object.is_expired() =>
				Ok(object.data()),

			_ => Err(CacheError::KeyNotFound),
		}
	}

	/// Sets the TTL associated with the supplied key.
	/// If the key was not found in the cache, returns a [`CacheError`].
	pub fn ttl(&self, key: &K, ttl: Option<u32>) -> Result<(), CacheError> {
		let hashed_key = self.hash_key(key);

		let mut object = match self.objects.get_mut(&hashed_key) {
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

		match self.objects.get(&hashed_key) {
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

	/// Resizes the cache's overall maximum size. If the supplied size is
	/// zero, returns a [`CacheError`]. Because `k_in` is a fraction of
	/// `max_size`, this also proportionally rescales the FIFO queue's byte
	/// budget (`TwoQHybridStack::resize`) — which may trigger immediate FIFO
	/// evictions on a shrink.
	///
	/// Note this is the *overall* cache capacity, independent of the main
	/// queue's fast-tier budget — see [`Self::set_fast_tier_size`].
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

	/// Runtime-adjusts the main queue's fast-tier byte budget. Shrinking it
	/// may trigger immediate demotions (see `TwoQHybridStack::settle_fast_tier`).
	/// Independent of `k_in` (the FIFO queue's own budget), which is fixed
	/// for the lifetime of the cache.
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

	/// Returns the current main-queue fast-tier byte budget.
	#[must_use]
	pub fn fast_tier_size(&self) -> CacheSize {
		self.status.fast_tier_capacity()
	}

	/// Returns a point-in-time snapshot of `two_q_hybrid_cache` statistics.
	#[must_use]
	pub fn two_q_hybrid_stats(&self) -> TwoQHybridStats {
		self.status.two_q_hybrid_stats()
	}

	/// Returns which tier `key` currently lives in, or `None` if the key
	/// isn't present (or has expired).
	#[must_use]
	pub fn tier_of(&self, key: &K) -> Option<Tier> {
		let hashed_key = self.hash_key(key);

		self.objects.get(&hashed_key).and_then(|object| {
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

/// Single-instance, segmented-FIFO hybrid cache: one `PaperCache<K,
/// TieredBuffer>` running `PaperPolicy::FifoHybrid`, in contrast with
/// `hybridcache`'s [`crate::hybridcache::S3FifoHybridCache`] (two composed
/// `PaperCache` instances). See the `fifo_hybrid_cache` module docs for the
/// full design; this impl block mirrors `lru_hybrid_cache`'s mechanically
/// (same shape: `new`, `get`, `set`, ...) with one deliberate deviation in
/// `set()` (see its doc comment below) — the rest of the behavioral
/// difference lives in `FifoHybridStack`, not here.
///
/// Admission always lands in the fast tier; fast-tier pressure demotes the
/// oldest fast-tier object to the slow tier; there is no promotion policy at
/// all — objects are never reordered by subsequent access, so a `get()` hit
/// never migrates anything. Terminal evictions (once overall `max_size` is
/// exceeded) only ever remove the slow-tier oldest object. Every migration
/// physically reallocates the object's bytes (see [`TieredBuffer`] and
/// `Object::set_data`) — a key is never present in both tiers at once.
///
/// Mutually exclusive with `lru_hybrid_cache`/`lfu_hybrid_cache`/
/// `two_q_hybrid_cache` (see `lib.rs`'s `compile_error!` guards) since all
/// four define this same inherent-method impl block on the identical
/// `PaperCache<K, TieredBuffer, S>` type.
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
		if max_size == 0 {
			return Err(CacheError::ZeroCacheSize);
		}

		let fast_capacity = fast_tier_size.to_bytes();

		if fast_capacity == 0 || fast_capacity > max_size {
			return Err(CacheError::InvalidFastTierSize);
		}

		let policies = [PaperPolicy::FifoHybrid];

		let objects = Arc::new(DashMap::with_hasher(NoHasher::default()));
		let status = Arc::new(AtomicStatus::new(max_size, &policies, PaperPolicy::FifoHybrid)?);
		let overhead_manager = Arc::new(OverheadManager::new(&status));

		// Requirement: fast-tier size is runtime-configurable (not baked
		// into the policy string), so the requested capacity is recorded on
		// the shared status immediately; `init_policy_stack`'s
		// 20%-of-max_size default (see `policy_stack/mod.rs`) is overridden
		// below via `ResizeFastTier`.
		status.set_fast_tier_capacity(fast_capacity);

		// Reallocates a value into the target tier's representation. Must
		// preserve byte length exactly: both `status.base_used_size` and
		// `FifoHybridStack`'s own per-key size bookkeeping assume a
		// migration never changes an object's accounted size.
		let migrate: Box<dyn Fn(&TieredBuffer, Tier) -> TieredBuffer + Send + Sync> =
			Box::new(|buffer, tier| match tier {
				Tier::Fast => TieredBuffer::new_fast(buffer.as_ref()),
				Tier::Slow => TieredBuffer::new_slow(buffer.as_ref()),
			});

		let (worker_sender, worker_listener) = unbounded();

		let mut worker_manager = WorkerManager::new_with_tier_migration(
			worker_listener,
			&objects,
			&status,
			&overhead_manager,
			migrate,
		)?;

		thread::spawn(move || worker_manager.run());

		let cache = PaperCache {
			objects,
			status,
			worker_manager: Arc::new(worker_sender),
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

		let result = match self.objects.get(&hashed_key) {
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

	/// Sets the supplied key and value in the cache. A genuinely new key is
	/// always admitted at the bottom of the fast tier. Overwriting an
	/// existing key never changes its tier or position — FIFO has no
	/// promotion/reordering policy at all (see `FifoHybridStack`'s module
	/// doc) — the value is written into whichever tier's representation the
	/// key already occupies, looked up here since the API-calling thread has
	/// no access to the worker-owned policy stack. This is the one method in
	/// this impl block that deviates from a verbatim `lru_hybrid_cache`
	/// copy: `LruHybridStack::insert` always re-admits an existing key to
	/// fast (promoting it), so `lru_hybrid_cache`'s `set()` can always build
	/// `TieredBuffer::new_fast` unconditionally — but `FifoHybridStack` never
	/// changes an existing key's tier, so unconditionally building
	/// `new_fast` here would desync a slow-tier key's physical
	/// representation from the stack's own bookkeeping permanently.
	/// Returns a [`CacheError`] if the value size is zero or larger than the
	/// cache's maximum size.
	pub fn set(&self, key: K, value: &[u8], ttl: Option<u32>) -> Result<(), CacheError> {
		let hashed_key = self.hash_key(&key);

		let existing_tier = self.objects.get(&hashed_key)
			.map(|object| if object.data().is_fast() { Tier::Fast } else { Tier::Slow });

		let val_buf = match existing_tier {
			Some(Tier::Slow) => TieredBuffer::new_slow(value),
			Some(Tier::Fast) | None => TieredBuffer::new_fast(value),
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
			.get(&hashed_key)
			.is_some_and(|object| object.key_matches(key) && !object.is_expired())
	}

	/// Gets (peeks) the value associated with the supplied key without
	/// altering any of the cache's internal queues. If the key was not found
	/// in the cache, returns a [`CacheError`].
	pub fn peek(&self, key: &K) -> Result<Arc<TieredBuffer>, CacheError> {
		let hashed_key = self.hash_key(key);

		match self.objects.get(&hashed_key) {
			Some(object) if object.key_matches(key) && !object.is_expired() =>
				Ok(object.data()),

			_ => Err(CacheError::KeyNotFound),
		}
	}

	/// Sets the TTL associated with the supplied key.
	/// If the key was not found in the cache, returns a [`CacheError`].
	pub fn ttl(&self, key: &K, ttl: Option<u32>) -> Result<(), CacheError> {
		let hashed_key = self.hash_key(key);

		let mut object = match self.objects.get_mut(&hashed_key) {
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

		match self.objects.get(&hashed_key) {
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
	/// fast-tier budget — see [`Self::set_fast_tier_size`].
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
	/// immediate demotions (see `FifoHybridStack::settle_fast_tier`).
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

	/// Returns a point-in-time snapshot of `fifo_hybrid_cache` statistics.
	#[must_use]
	pub fn fifo_hybrid_stats(&self) -> FifoHybridStats {
		self.status.fifo_hybrid_stats()
	}

	/// Returns which tier `key` currently lives in, or `None` if the key
	/// isn't present (or has expired).
	#[must_use]
	pub fn tier_of(&self, key: &K) -> Option<Tier> {
		let hashed_key = self.hash_key(key);

		self.objects.get(&hashed_key).and_then(|object| {
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

#[cfg(all(test, feature = "original"))]
mod tests {
	use crate::{PaperCache, PaperPolicy, CacheError};

	const TEST_CACHE_MAX_SIZE: u64 = 1000;

	#[test]
	fn it_returns_correct_version() {
		let cache = init_test_cache();
		assert_eq!(cache.version(), env!("CARGO_PKG_VERSION"));
	}

	#[test]
	fn it_returns_status() {
		let cache = init_test_cache();
		let status = cache.status().unwrap();

		assert_eq!(status.max_size(), TEST_CACHE_MAX_SIZE);
	}

	#[test]
	fn it_gets_an_existing_object() {
		let cache = init_test_cache();

		assert!(cache.set(0, 1, None).is_ok());
		assert_eq!(cache.get(&0).as_deref(), Ok(&1));
	}

	#[test]
	fn it_does_not_get_a_non_existing_object() {
		let cache = init_test_cache();

		assert!(cache.set(0, 1, None).is_ok());
		assert_eq!(cache.get(&1), Err(CacheError::KeyNotFound));
	}

	#[test]
	fn it_calculates_miss_ratio_correctly() {
		let cache = init_test_cache();

		assert!(cache.set(0, 1, None).is_ok());
		assert!(cache.get(&0).is_ok());
		assert!(cache.get(&0).is_ok());
		assert!(cache.get(&0).is_ok());
		assert!(cache.get(&1).is_err());

		let status = cache.status().unwrap();
		assert_eq!(status.miss_ratio(), 0.25);
	}

	#[test]
	fn it_sets_with_no_ttl() {
		let cache = init_test_cache();

		assert!(cache.set(0, 1, None).is_ok());
		assert!(cache.get(&0).is_ok());
	}

	#[test]
	fn it_sets_with_ttl() {
		use std::{
			thread,
			time::Duration,
		};

		let cache = init_test_cache();
		assert!(cache.set(0, 1, Some(1)).is_ok());

		assert!(cache.get(&0).is_ok());
		thread::sleep(Duration::from_secs(2));
		assert!(cache.get(&0).is_err());
	}

	#[test]
	fn it_dels_an_existing_object() {
		let cache = init_test_cache();
		assert!(cache.set(0, 1, Some(1)).is_ok());

		assert!(cache.get(&0).is_ok());
		assert!(cache.del(&0).is_ok());
		assert!(cache.get(&0).is_err());
	}

	#[test]
	fn it_does_not_del_a_non_existing_object() {
		let cache = init_test_cache();
		assert_eq!(cache.del(&0), Err(CacheError::KeyNotFound));
	}

	#[test]
	fn it_does_not_allow_empty_policies() {
		let try_cache = PaperCache::<u32, u32>::new(
			TEST_CACHE_MAX_SIZE,
			&[],
			PaperPolicy::Lfu,
		);

		assert!(try_cache.is_err_and(|err| err == CacheError::EmptyPolicies));

		let try_cache = PaperCache::<u32, u32>::new(
			TEST_CACHE_MAX_SIZE,
			&[],
			PaperPolicy::Auto,
		);

		assert!(try_cache.is_err_and(|err| err == CacheError::EmptyPolicies));
	}

	#[test]
	fn it_does_not_allow_auto_policy_in_configured_policies() {
		let try_cache = PaperCache::<u32, u32>::new(
			TEST_CACHE_MAX_SIZE,
			&[PaperPolicy::Auto],
			PaperPolicy::Auto,
		);

		assert!(try_cache.is_err_and(|err| err == CacheError::ConfiguredAutoPolicy));

		let try_cache = PaperCache::<u32, u32>::new(
			TEST_CACHE_MAX_SIZE,
			&[PaperPolicy::Auto, PaperPolicy::Lru],
			PaperPolicy::Auto,
		);

		assert!(try_cache.is_err_and(|err| err == CacheError::ConfiguredAutoPolicy));
	}

	#[test]
	fn it_does_not_allow_duplicate_policies() {
		let try_cache = PaperCache::<u32, u32>::new(
			TEST_CACHE_MAX_SIZE,
			&[PaperPolicy::Lfu, PaperPolicy::Lru, PaperPolicy::Lfu],
			PaperPolicy::Lfu,
		);

		assert!(try_cache.is_err_and(|err| err == CacheError::DuplicatePolicies));

		let try_cache = PaperCache::<u32, u32>::new(
			TEST_CACHE_MAX_SIZE,
			&[PaperPolicy::Lfu, PaperPolicy::Lru],
			PaperPolicy::Lfu,
		);

		assert!(try_cache.is_ok());
	}

	#[test]
	fn it_has_an_existing_object() {
		let cache = init_test_cache();

		assert!(cache.set(0, 1, Some(1)).is_ok());
		assert!(cache.has(&0));
	}

	#[test]
	fn it_does_not_have_a_non_existing_object() {
		let cache = init_test_cache();
		assert!(!cache.has(&1));
	}

	#[test]
	fn it_peeks_an_existing_object() {
		let cache = init_test_cache();

		assert!(cache.set(0, 1, None).is_ok());
		assert_eq!(cache.peek(&0).as_deref(), Ok(&1));
	}

	#[test]
	fn it_does_not_peek_a_non_existing_object() {
		let cache = init_test_cache();

		assert!(cache.set(0, 1, None).is_ok());
		assert_eq!(cache.peek(&1), Err(CacheError::KeyNotFound));
	}

	#[test]
	fn it_sets_an_existing_objects_ttl() {
		use std::{
			thread,
			time::Duration,
		};

		let cache = init_test_cache();

		assert!(cache.set(0, 1, None).is_ok());
		assert!(cache.get(&0).is_ok());

		assert!(cache.ttl(&0, Some(1)).is_ok());

		thread::sleep(Duration::from_secs(2));
		assert_eq!(cache.get(&0), Err(CacheError::KeyNotFound));
	}

	#[test]
	fn it_does_not_set_a_non_existing_objects_ttl() {
		let cache = init_test_cache();
		assert_eq!(cache.ttl(&0, Some(1)), Err(CacheError::KeyNotFound));
	}

	#[test]
	fn it_resets_an_objects_ttl() {
		use std::{
			thread,
			time::Duration,
		};

		let cache = init_test_cache();

		assert!(cache.set(0, 1, Some(1)).is_ok());
		assert!(cache.get(&0).is_ok());

		assert!(cache.ttl(&0, Some(5)).is_ok());

		thread::sleep(Duration::from_secs(2));
		assert!(cache.get(&0).is_ok());
	}

	#[test]
	fn it_gets_an_objects_size() {
		use std::mem;

		use crate::object::{
			ExpireTime,
			overhead::get_policy_overhead,
		};

		let cache = init_test_cache();

		let expected = 4 + 4
			+ mem::size_of::<ExpireTime>() as u32
			+ get_policy_overhead(&PaperPolicy::Lfu);

		assert!(cache.set(0, 1, None).is_ok());
		assert_eq!(cache.size(&0), Ok(expected));
	}

	#[test]
	fn it_gets_an_expiring_objects_size() {
		use std::mem;

		use crate::object::{
			ExpireTime,
			overhead::{get_policy_overhead, get_ttl_overhead},
		};

		let cache = init_test_cache();

		let expected = 4 + 4
			+ mem::size_of::<ExpireTime>() as u32
			+ get_policy_overhead(&PaperPolicy::Lfu)
			+ get_ttl_overhead();

		assert!(cache.set(0, 1, Some(1)).is_ok());
		assert_eq!(cache.size(&0), Ok(expected));
	}

	#[test]
	fn it_gets_an_objects_size_after_policy_switch() {
		use std::mem;

		use crate::object::{
			ExpireTime,
			overhead::get_policy_overhead,
		};

		let cache = PaperCache::<u32, u32>::new(
			TEST_CACHE_MAX_SIZE,
			&[PaperPolicy::Lru, PaperPolicy::Lfu],
			PaperPolicy::Lfu,
		).expect("Could not initialize test cache");

		let base_expected = 4 + 4 + mem::size_of::<ExpireTime>() as u32;
		let lfu_expected = base_expected + get_policy_overhead(&PaperPolicy::Lfu);
		let lru_expected = base_expected + get_policy_overhead(&PaperPolicy::Lru);

		assert!(cache.set(0, 1, None).is_ok());
		assert_eq!(cache.size(&0), Ok(lfu_expected));

		assert!(cache.policy(PaperPolicy::Lru).is_ok());
		assert_eq!(cache.size(&0), Ok(lru_expected));
	}

	#[test]
	fn it_gets_an_objects_size_after_set_ttl() {
		use std::mem;

		use crate::object::{
			ExpireTime,
			overhead::{get_policy_overhead, get_ttl_overhead},
		};

		let cache = init_test_cache();

		let pre_expected = 4 + 4
			+ mem::size_of::<ExpireTime>() as u32
			+ get_policy_overhead(&PaperPolicy::Lfu);

		let post_expected = pre_expected + get_ttl_overhead();

		assert!(cache.set(0, 1, None).is_ok());
		assert_eq!(cache.size(&0), Ok(pre_expected));

		assert!(cache.ttl(&0, Some(1)).is_ok());
		assert_eq!(cache.size(&0), Ok(post_expected));
	}

	#[test]
	fn it_gets_an_objects_size_after_unset_ttl() {
		use std::mem;

		use crate::object::{
			ExpireTime,
			overhead::{get_policy_overhead, get_ttl_overhead},
		};

		let cache = init_test_cache();

		let pre_expected = 4 + 4
			+ mem::size_of::<ExpireTime>() as u32
			+ get_policy_overhead(&PaperPolicy::Lfu)
			+ get_ttl_overhead();

		let post_expected = pre_expected - get_ttl_overhead();

		assert!(cache.set(0, 1, Some(1)).is_ok());
		assert_eq!(cache.size(&0), Ok(pre_expected));

		assert!(cache.ttl(&0, None).is_ok());
		assert_eq!(cache.size(&0), Ok(post_expected));
	}

	#[test]
	fn status_shows_correct_used_size() {
		use std::mem;

		use crate::object::{
			ExpireTime,
			overhead::{get_policy_overhead, get_ttl_overhead},
		};

		let cache = init_test_cache();

		let expected = (4 + 4) * 2
			+ mem::size_of::<ExpireTime>() as u32 * 2
			+ get_policy_overhead(&PaperPolicy::Lfu) * 2
			+ get_ttl_overhead();

		assert!(cache.set(0, 1, None).is_ok());
		assert!(cache.set(1, 1, Some(1)).is_ok());

		let status = cache.status().unwrap();
		assert_eq!(status.used_size(), expected as u64);
	}

	#[test]
	fn status_shows_correct_used_size_after_policy_switch() {
		use std::mem;

		use crate::object::{
			ExpireTime,
			overhead::get_policy_overhead,
		};

		let cache = PaperCache::<u32, u32>::new(
			TEST_CACHE_MAX_SIZE,
			&[PaperPolicy::Lru, PaperPolicy::Lfu],
			PaperPolicy::Lfu,
		).expect("Could not initialize test cache");

		let base_expected = 4 + 4 + mem::size_of::<ExpireTime>() as u32;
		let lfu_expected = base_expected + get_policy_overhead(&PaperPolicy::Lfu);
		let lru_expected = base_expected + get_policy_overhead(&PaperPolicy::Lru);

		assert!(cache.set(0, 1, None).is_ok());
		let status = cache.status().unwrap();
		assert_eq!(status.used_size(), lfu_expected as u64);

		assert!(cache.policy(PaperPolicy::Lru).is_ok());
		let status = cache.status().unwrap();
		assert_eq!(status.used_size(), lru_expected as u64);
	}

	#[test]
	fn status_shows_correct_used_size_after_set_ttl() {
		use std::mem;

		use crate::object::{
			ExpireTime,
			overhead::{get_policy_overhead, get_ttl_overhead},
		};

		let cache = init_test_cache();

		let pre_expected = 4 + 4
			+ mem::size_of::<ExpireTime>() as u32
			+ get_policy_overhead(&PaperPolicy::Lfu);

		let post_expected = pre_expected + get_ttl_overhead();

		assert!(cache.set(0, 1, None).is_ok());
		let status = cache.status().unwrap();
		assert_eq!(status.used_size(), pre_expected as u64);

		assert!(cache.ttl(&0, Some(1)).is_ok());
		let status = cache.status().unwrap();
		assert_eq!(status.used_size(), post_expected as u64);
	}

	#[test]
	fn status_shows_correct_used_size_after_unset_ttl() {
		use std::mem;

		use crate::object::{
			ExpireTime,
			overhead::{get_policy_overhead, get_ttl_overhead},
		};

		let cache = init_test_cache();

		let pre_expected = 4 + 4
			+ mem::size_of::<ExpireTime>() as u32
			+ get_policy_overhead(&PaperPolicy::Lfu)
			+ get_ttl_overhead();

		let post_expected = pre_expected - get_ttl_overhead();

		assert!(cache.set(0, 1, Some(1)).is_ok());
		let status = cache.status().unwrap();
		assert_eq!(status.used_size(), pre_expected as u64);

		assert!(cache.ttl(&0, None).is_ok());
		let status = cache.status().unwrap();
		assert_eq!(status.used_size(), post_expected as u64);
	}

	fn init_test_cache() -> PaperCache<u32, u32> {
		PaperCache::<u32, u32>::new(
			TEST_CACHE_MAX_SIZE,
			&[PaperPolicy::Lfu],
			PaperPolicy::Lfu,
		).expect("Could not initialize test cache")
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
/// These tests prove that hw_perf instrumentation and eviction_stacks_pmem allocations
/// integrate correctly with the cache initialization path.
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

/// Verify that hw_perf counters can be measured when the feature is enabled.
/// When the feature is disabled, this entire module is compiled out — zero cost.
#[cfg(all(test, feature = "hw_perf"))]
mod test_hw_perf {
    /// Verify measure_operation is callable and returns correct results.
    /// In CI environments without perf_event access, measurement may return None — that's fine.
    #[test]
    fn test_hw_perf_measure_operation() {
        use crate::measure_operation;

        let (result, _measurement) = measure_operation(|| {
            let mut sum: u64 = 0;
            for i in 0u64..1000 {
                sum = sum.wrapping_add(i);
            }
            sum
        });

        assert_eq!(result, 499500u64, "operation result must be correct regardless of perf counter availability");
    }
}

/// Exercises the real public `PaperCache<K, TieredBuffer>` API end to end.
///
/// Deliberately stays on the fast-tier-only path (fast_tier_size == max_size,
/// tiny values) so no object ever demotes to the slow tier: `TieredBuffer::
/// new_slow` allocates through the `Hybrid`/UMF PMEM allocator, which
/// (like `hybridcache`'s PMEM-tier integration tests) requires real PMEM/DAX
/// hardware and aborts ("memory allocation ... failed") in a plain dev
/// sandbox. A full integration test covering demotion/promotion/eviction
/// belongs in `tests/lru_hybrid_cache_integration.rs` (not yet written —
/// see `CLAUDE.md`'s `lru_hybrid_cache` plan, step 12) and should be run on
/// PMEM-capable hardware, same as `tests/hybridcache_integration.rs`.
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
