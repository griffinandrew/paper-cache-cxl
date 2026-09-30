/*
 * Copyright (c) Kia Shakiba
 *
 * This source code is licensed under the GNU AGPLv3 license found in the
 * LICENSE file in the root directory of this source tree.
 * correct
 */

#![cfg_attr(any(feature = "hashbrown_dram", feature = "all_dram", feature = "key_value_pmem", feature = "global_hashtable_pmem", feature = "eviction_stacks_pmem", feature = "merged_object_store", feature = "hybrid_cache_common"), feature(allocator_api), feature(clone_from_ref), feature(btreemap_alloc))]


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

// `Hybrid` is the crate-wide PMEM allocator alias: every PMEM feature routes
// through `NODE_SLOW`-bound jemalloc arenas (`numa_alloc::SlowAlloc`).
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
    feature = "eviction_stacks_pmem",
    feature = "segregated_value_arena",
))]
pub(crate) use crate::numa_alloc::SlowObjects as Hybrid;


mod error;
mod worker;

/// The v5 value representation: an entire cached value in one eight-byte
/// tagged pointer, with no refcount and no per-value allocation header. The
/// ONLY module in this crate holding unsafe value code -- see `value`.
///
/// Compiled in every configuration on purpose. A partial feature list is how
/// this tree loses test coverage silently, and the one module that owns every
/// `alloc`, `dealloc` and pointer reinterpretation for values is the last one
/// that should be skippable.
/// The cached value. Two layouts, chosen at compile time; the public surface
/// is identical, so nothing outside this module changes with the choice.
///
/// * default -- a DRAM-resident refcounted header pointing at bytes that live
///   in their own allocation, on whichever tier the policy put them.
/// * `fused_value` -- one allocation holding the count, the metadata, the key
///   AND the bytes, tiering as a unit.
/// * `thin_header` -- a 16-byte DRAM header holding only the count and a
///   tagged pointer, in front of one tiered item holding the length, the
///   expiry, the key and the bytes.
///
/// The default is not an accident: see `value.rs`'s "Why the bytes are a
/// SECOND allocation" for the measurements that chose it.
#[cfg(not(any(feature = "fused_value", feature = "thin_header")))]
#[path = "value.rs"]
pub mod value;

// Under `stock_jemalloc` the global allocator is not `NumaAlloc`, so the fast
// pool would read ZERO while the slow one still worked -- a green build with
// the measurement silently disabled on exactly the tier being bounded.
#[cfg(all(feature = "measured_accounting", feature = "stock_jemalloc"))]
compile_error!(
	"measured_accounting requires the NUMA allocator; under stock_jemalloc the \
	 fast-pool counter would silently read zero"
);

#[cfg(all(feature = "fused_value", not(feature = "thin_header")))]
#[path = "value_fused.rs"]
pub mod value;

#[cfg(feature = "thin_header")]
#[path = "value_thin.rs"]
pub mod value;

// Alternative layouts for the same type, and each moves the accounting
// differently (`object::overhead`), so a silent precedence between them would
// measure one while the build said the other.
#[cfg(all(feature = "thin_header", feature = "fused_value"))]
compile_error!("thin_header and fused_value are alternative value layouts; enable at most one");

// `key_pmem_value_pmem` boxes every key into its own persistent-memory
// allocation. Under `thin_header` the key's placement IS the layout -- it lives
// in the item and tiers with it -- so the two would disagree about where every
// key is.
#[cfg(all(feature = "thin_header", feature = "key_pmem_value_pmem"))]
compile_error!(
	"thin_header stores the key inside the tiered item; it cannot be combined \
	 with key_pmem_value_pmem"
);

/// `paper_cache::TieredValue`, alongside `paper_cache::TieredBuffer`.
pub use crate::value::TieredValue;

/// The two non-hybrid cache SHAPES. Since v5 both store a `TieredValue`; the
/// marker only says which tier `set()` allocates in -- see `value::ValueShape`.
pub use crate::value::{BufferDRAM, BufferPMEM};

/// A value plus the epoch pin that keeps it alive, which is how every read
/// path touches value bytes with the shard guard already released.

/// The concurrency gate for epoch-based value reclamation: eight readers
/// copying while a flapper migrates and an overwriter frees underneath them.
/// Single-threaded tests cannot see the window this design closes.
///
/// `#[ignore]`d and gated on one hybrid feature, so it neither changes any
/// suite's count nor runs alongside tests that would pollute the process-wide
/// jemalloc statistics it reads.
#[cfg(all(test, feature = "lru_compact_hybrid_cache"))]
mod value_stress;

mod object;
mod policy;
mod status;

/// M, the bytes the cache's own DRAM metadata structures hold -- the object
/// map's, the policy stack's and the value headers' -- counted from the
/// structures (S5a). Public for its unit helpers (`usable`, `vec_bytes`) and
/// for `DramMetadata`; see the module doc.
pub mod meta;

#[cfg(feature = "hybrid_cache_common")]
pub use crate::meta::DramMetadata;

/// S5: the admission path of a tiered cache's set -- the size checks, the
/// metadata cap, structural slow placement (B1), the byte gate that holds a
/// fast set to the tier's budget (B2) -- and the figures it reads.
#[cfg(feature = "hybrid_cache_common")]
pub mod gate;

#[cfg(feature = "hybrid_cache_common")]
pub use crate::gate::{GateConfig, GateMode, GateState, MetadataModel, MetadataOverflow, OnStall};

// Shared object-map storage-backend abstraction and value-buffer
// abstraction (see each module's doc comment) -- used by the generic
// `impl<K, V, S> PaperCache<K, V, S>` blocks below to replace what used to
// be one impl block per (object-map shape, value-buffer type) combination.
#[cfg(any(feature = "all_dram", feature = "key_value_pmem", feature = "global_hashtable_pmem", feature = "hashbrown_dram"))]
mod object_store;

/// Object map, recency order and tier placement in ONE structure.
///
/// Gated, which it was not originally. Compiled unconditionally, a syntax error
/// in this module broke EVERY configuration -- including builds that do not use
/// it at all. That is not hypothetical: a refactor of this file took out an
/// unrelated `lru_compact_hybrid_cache` build mid-flight. Nothing outside the
/// feature references it, so there was never a reason for it to be reachable
/// from a build that has the feature off.
#[cfg(feature = "merged_object_store")]
pub mod merged_store;

#[cfg(any(feature = "all_dram", feature = "key_value_pmem", feature = "global_hashtable_pmem", feature = "hashbrown_dram"))]
use crate::object_store::ObjectStore;
#[cfg(any(feature = "all_dram", feature = "key_value_pmem", feature = "global_hashtable_pmem", feature = "hashbrown_dram"))]
use crate::value::ValueShape;

// Shared tier-size unit type (bytes/Mb/Gb), used by every hybrid design so
// none of them has to depend on any of the others for it.
#[cfg(feature = "hybrid_cache_common")]
mod size;

#[cfg(feature = "hybrid_cache_common")]
pub use crate::size::CacheTierSize;

// Shared value type for every hybrid design: `paper_cache::TieredBuffer`.
// The per-design `<design>_hybrid_cache` shim modules that used to re-export
// it are gone, along with the designs they named.
#[cfg(feature = "hybrid_cache_common")]
mod tiered_buffer;

#[cfg(feature = "hybrid_cache_common")]
pub use crate::tiered_buffer::TieredBuffer;

// Design-neutral view of whichever hybrid design a cache is running. The only
// stats accessor: the per-design `<design>_hybrid_stats()` methods and the
// `<Design>HybridStats` aliases are gone. See `hybrid_stats.rs`'s module doc.
#[cfg(feature = "hybrid_cache_common")]
mod hybrid_stats;

#[cfg(feature = "hybrid_cache_common")]
pub use crate::hybrid_stats::HybridStats;

// PHYS_FAST -- the bytes physically allocated in the fast tier's value pool,
// charged and refunded by the value constructors and destructors -- and the
// count of live tiered caches it is meaningful under. Public so a harness or
// a test can read it directly; see `phys.rs`'s module doc.
#[cfg(feature = "hybrid_cache_common")]
pub mod phys;

// Re-exported so `PaperCache::tier_of`'s return type is nameable by callers
// without reaching into the private `worker` module tree directly.
//
// Unconditional, where it used to be gated on `hybrid_cache_common`:
// `value::TieredValue::tier()` is `pub`, is compiled in every configuration,
// and returns this type, so it must be publicly nameable in every
// configuration too.
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

pub type CacheSize = u64;
pub type AtomicCacheSize = AtomicU64;


/// Serialises every test that reads a process-global counter as a delta.
///
/// `VALUE_FREES`, `PENDING_DEMOTE` and the allocator-routing counters are all
/// process-global, and cargo runs tests in parallel by default, so a test that
/// samples one, does an operation, and asserts the difference is racing every
/// other test that touches the same counter. Both `VALUE_FREES` and
/// `PENDING_DEMOTE` were observed failing that way in a six-run sweep -- two
/// flakes in six, on tests that are individually correct.
///
/// One lock rather than one per module, because the races are BETWEEN modules:
/// a value dropped by a `value` test moves the counter an `object` test is
/// asserting on. Poisoning is ignored -- a panic in one such test must not
/// convert every other one into a failure that hides it.
#[cfg(test)]
pub(crate) fn global_counter_lock() -> std::sync::MutexGuard<'static, ()> {
	static LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

	LOCK.lock().unwrap_or_else(|poisoned| poisoned.into_inner())
}

pub type HashedKey = u64;
pub type NoHasher = BuildHasherDefault<NoHashHasher<HashedKey>>;



// Both tiers are allocated by `numa_alloc` (src/numa_alloc.rs): node-0-bound
// jemalloc arenas back the fast tier and the process allocator, `NODE_SLOW`-bound
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





/// Initial capacity (in entries) for the hashbrown-backed object map used
/// by `hashbrown_dram`, `global_hashtable_pmem`, and any hybrid-cache
/// feature combined with `hashbrown_dram` (see `new_hybrid_object_map`).
/// Sized to hold every object across this project's real benchmark traces
/// (`/home/griff/final_traces/*.bin`, distinct GET-driven keys measured at
/// ~1.06M-1.09M per trace) without ever growing/rehashing mid-benchmark.
#[cfg(any(feature = "global_hashtable_pmem", feature = "hashbrown_dram"))]
const HASHBROWN_INITIAL_CAPACITY: usize = 1_500_000;

#[cfg(all(not(feature = "global_hashtable_pmem"), not(feature = "hashbrown_dram"), not(feature = "merged_object_store")))]
pub type ObjectMapRef<K, V> = Arc<DashMap<HashedKey, Object<K, V>, NoHasher>>;

/// Object map and LRU eviction order in ONE structure -- see `merged_store`.
/// Measured 208 B/object against the split design's 280.
#[cfg(feature = "merged_object_store")]
pub type ObjectMapRef<K, V> = Arc<crate::merged_store::MergedStore<K, V>>;

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
	/// behalf (`PolicyWorker`/`TtlWorker` -- `PolicyWorker`'s
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

		// The object map goes next, in the compiler-generated field drops
		// after this body returns, and every object it holds defers its
		// value's free. Advancing the epoch here first retires whatever the
		// workers left behind, so that the last flush -- from a thread with
		// no pins outstanding, all workers joined -- has the best chance of
		// actually running the deferrals rather than parking them in a bag
		// that nothing will ever drain again.
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
// concrete V).
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
	V: ValueShape,
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

		#[cfg(not(feature = "merged_object_store"))]
		let objects = Arc::new(DashMap::with_hasher(NoHasher::default()));

		// The merged store is both the object map and the eviction order; the
		// worker builds its `PolicyStack` from this same `Arc`.
		#[cfg(feature = "merged_object_store")]
		let objects = Arc::new(crate::merged_store::MergedStore::new());
		let status = Arc::new(AtomicStatus::new(max_size, policies, policy)?);
		let overhead_manager = Arc::new(OverheadManager::new(&status));

		// A flat cache whose values are FAST builds them through the same
		// `TieredValue::new_in` as a tiered one, so PHYS_FAST counts them:
		// counted in `phys::live_flat_fast_caches` until the status is freed,
		// since P describes one tiered cache only while that reads 0.
		#[cfg(feature = "hybrid_cache_common")]
		if matches!(V::TIER, crate::Tier::Fast) {
			status.register_flat_fast_cache();
		}

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

	/// A flat cache has no tiers to audit: `None`. See the tiered cache's
	/// `placement_audit`.
	#[cfg(feature = "hybrid_cache_common")]
	#[must_use]
	pub fn placement_audit(&self) -> Option<crate::phys::PlacementAudit> {
		None
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

		// Take the value handle under the shard guard, release the guard, and
		// only then copy. The handle owns a strong reference, so a writer that
		// unpublishes this value while the copy is in flight decrements a count
		// that is not yet zero and frees nothing.
		let snapshot = match self.objects.get_ref(&hashed_key) {
			Some(object) if object.key_matches(key) && !object.is_expired() =>
				Some(object.snapshot()),
			_ => None,
		};

		// The tier the hit is served from -- the snapshot's tag -- or `None`
		// on a miss: the policy worker's heal needs it (`WorkerEvent::Get`).
		let served = snapshot.as_ref().map(|value| value.tier());

		let result = match snapshot {
			Some(value) => {
				self.status.incr_hits();
				Ok(value.bytes().to_vec())
			},

			None => {
				self.status.incr_misses();
				Err(CacheError::KeyNotFound)
			},
		};


		self.broadcast(WorkerEvent::Get(hashed_key, served))?;

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

		let snapshot = match self.objects.get_ref(&hashed_key) {
			Some(object) if object.key_matches(key) && !object.is_expired() =>
				Some(object.snapshot()),
			_ => None,
		};
		let t2 = if prof { Some(std::time::Instant::now()) } else { None };

		// Which tier served this hit (1 = fast/DRAM; the all-DRAM shape never
		// reassigns it). Read only by the sampled profiler below.
		#[allow(unused_mut)]
		let mut gi_fast: u64 = 1;

		// The tier the hit is served from -- the snapshot's tag -- or `None`
		// on a miss: the policy worker's heal needs it (`WorkerEvent::Get`).
		let served = snapshot.as_ref().map(|value| value.tier());

		let result = match snapshot {
			Some(value) => {
				self.status.incr_hits();
				out.clear();
				out.extend_from_slice(value.bytes());
				Ok(())
			},

			None => {
				self.status.incr_misses();
				Err(CacheError::KeyNotFound)
			},
		};
		let t3 = if prof { Some(std::time::Instant::now()) } else { None };

		self.broadcast(WorkerEvent::Get(hashed_key, served))?;

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

		// The size checks before anything is allocated (S5), with the
		// predicates they always had: the length's base size
		// (`OverheadManager::base_size_for`, which `base_size` of the object
		// built below equals) -- a value too large is refused unbuilt.
		match self.overhead_manager.base_size_for(&key, value.len(), ttl) {
			None => return Err(CacheError::ExceedingValueSize),
			Some(0) => return Err(CacheError::ZeroValueSize),
			Some(base) if self.status.exceeds_max_size(base) => return Err(CacheError::ExceedingValueSize),
			Some(_) => {},
		}

		// The one thing the shape still decides: which allocator the value
		// comes from. `BufferDRAM` names the fast tier, `BufferPMEM` the slow
		// one -- see `value::ValueShape`.
		let object = Object::new_in(key, value, V::TIER, ttl);
		let base_size = self.overhead_manager.base_size(&object);
		let dram_resident = self.overhead_manager.dram_resident_size(&object);
		let expiry = object.expiry();
		// Where the bytes were allocated, for the worker's reconcile
		// (`WorkerEvent::Set`).
		let built = object.value().tier();

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
			built,
			// A flat cache queues no migration: nothing to mark.
			0,
			// Nor places anything by tier.
			crate::worker::Placement::Normal,
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

		// No epoch pin, deliberately -- unlike `get`/`get_into`/`peek`. This
		// reads the key, the expiry, the length and the tag bit, never the
		// value's bytes, and the shard guard it holds while doing so keeps the
		// object -- and through its handle everything those live in -- alive.
		// A pin would protect nothing that is read here. (Under `thin_header`
		// the first three are in the tiered item: one remote cache line for a
		// slow object.)
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
	/// # API change (v5)
	///
	/// This returned a `Shared<V>` -- a refcounted handle onto the value --
	/// until the refcount was removed. It now returns an owned `Vec<u8>`, the
	/// same thing [`Self::get`] returns.
	///
	/// It cannot return a borrow. A value is now a bare pointer whose lifetime
	/// is managed by epoch reclamation, so the only two honest return types
	/// are a copy or a guard object holding the pin open -- and a guard held by
	/// a caller that then blocks would pin the epoch and stall reclamation for
	/// every thread, which is the one failure mode this design has to avoid.
	/// A copy has the same semantics the `Shared` did anyway: a snapshot that
	/// was live at the moment of the lookup.
	pub fn peek(&self, key: &K) -> Result<Vec<u8>, CacheError> {
		let hashed_key = self.hash_key(key);
		let snapshot = match self.objects.get_ref(&hashed_key) {
			Some(object) if object.key_matches(key) && !object.is_expired() =>
				Some(object.snapshot()),

			_ => None,
		};

		let result = match snapshot {
			Some(value) => Ok(value.bytes().to_vec()),
			None => Err(CacheError::KeyNotFound),
		};


		result
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

		// No epoch pin, deliberately -- unlike `get`/`get_into`/`peek`. This
		// reads the key, the expiry, the length and the tag bit, never the
		// value's bytes, and the shard guard it holds while doing so keeps the
		// object -- and through its handle everything those live in -- alive.
		// A pin would protect nothing that is read here. (Under `thin_header`
		// the first three are in the tiered item: one remote cache line for a
		// slow object.)
		match self.objects.get_ref(&hashed_key) {
			Some(object) if object.key_matches(key) && !object.is_expired() =>
				Ok(self.overhead_manager.total_size(&object)),

			_ => Err(CacheError::KeyNotFound),
		}
	}

	/// Deletes all objects in the cache and sets the cache's used size to zero.
	/// Returns a [`CacheError`] if the objects could not be wiped.
	///
	/// The policy worker does it, and this returns when it has (see the tiered
	/// cache's `wipe`).
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

		// The policy worker wipes -- the object map, its stack, the status
		// counters and the tier gauges -- and answers when it is done
		// (`PolicyWorker::handle_wipe`); this thread waits for the answer. It
		// used to clear the map and the status here and leave the stack to
		// the worker, and a `Set` the worker handled in between left a live
		// key its stack no longer tracked. The kick wakes a worker parked on
		// its idle poll (up to 1 s); the wait still includes the events queued
		// ahead of the `Wipe`. The values `clear_counted` drops retire into the
		// worker's epoch bag, which its pass flushes.
		let (ack, done) = crossbeam_channel::bounded(1);
		let sent = self.broadcast(WorkerEvent::Wipe(Some(ack)));

		self.status.kick_policy_worker();

		match done.recv() {
			// Wiped. A failed delivery to another subscriber (a dead TTL
			// worker) is still reported, as it always was.
			Ok(()) => sent,

			// Every sender is gone without an answer: the policy worker is dead
			// (its channel dropped, with the event in it) and the other
			// subscribers have handled or dropped their copies -- the TTL
			// worker within its poll, 1 s at most. Wipe here so the cache is
			// empty all the same, and say it failed.
			Err(_) => {
				let cleared = self.objects.clear_counted(|object| self.overhead_manager.base_size(object));
				self.status.clear(cleared);

				Err(CacheError::Internal)
			},
		}
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
// ---------------------------------------------------------------------
#[cfg(any(feature = "global_hashtable_pmem", feature = "hashbrown_dram"))]
impl<K, V, S> PaperCache<K, V, S>
where
	// `Send + Sync` because `WorkerFanout::new` hands the object map to worker
	// threads. Every other `PaperCache` impl carries these; this shape was
	// merged without them, so the two features that select it never built.
	K: 'static + Eq + Hash + TypeSize + Clone + Send + Sync,
	V: ValueShape,
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

		// A flat cache whose values are FAST builds them through the same
		// `TieredValue::new_in` as a tiered one, so PHYS_FAST counts them:
		// counted in `phys::live_flat_fast_caches` until the status is freed,
		// since P describes one tiered cache only while that reads 0.
		#[cfg(feature = "hybrid_cache_common")]
		if matches!(V::TIER, crate::Tier::Fast) {
			status.register_flat_fast_cache();
		}

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

	/// A flat cache has no tiers to audit: `None`. See the tiered cache's
	/// `placement_audit`.
	#[cfg(feature = "hybrid_cache_common")]
	#[must_use]
	pub fn placement_audit(&self) -> Option<crate::phys::PlacementAudit> {
		None
	}

	/// Gets the value associated with the supplied key.
	/// If the key was not found in the cache, returns a [`CacheError`].
	pub fn get(&self, key: &K) -> Result<Vec<u8>, CacheError> {
		let hashed_key = self.hash_key(key);

		// Take the value handle under the shard guard, release the guard, and
		// only then copy. The handle owns a strong reference, so a writer that
		// unpublishes this value while the copy is in flight decrements a count
		// that is not yet zero and frees nothing.
		let snapshot = match self.objects.get_ref(&hashed_key) {
			Some(object) if object.key_matches(key) && !object.is_expired() =>
				Some(object.snapshot()),
			_ => None,
		};

		// The tier the hit is served from -- the snapshot's tag -- or `None`
		// on a miss: the policy worker's heal needs it (`WorkerEvent::Get`).
		let served = snapshot.as_ref().map(|value| value.tier());

		let result = match snapshot {
			Some(value) => {
				self.status.incr_hits();
				Ok(value.bytes().to_vec())
			},

			None => {
				self.status.incr_misses();
				Err(CacheError::KeyNotFound)
			},
		};


		self.broadcast(WorkerEvent::Get(hashed_key, served))?;

		result
	}

	/// Sets the supplied key and value in the cache.
	/// Returns a [`CacheError`] if the value size is zero or larger than
	/// the cache's maximum size.
	pub fn set(&self, key: K, value: &[u8], ttl: Option<u32>) -> Result<(), CacheError> {
		let hashed_key = self.hash_key(&key);

		// The size checks before anything is allocated (S5), with the
		// predicates they always had: the length's base size
		// (`OverheadManager::base_size_for`, which `base_size` of the object
		// built below equals) -- a value too large is refused unbuilt.
		match self.overhead_manager.base_size_for(&key, value.len(), ttl) {
			None => return Err(CacheError::ExceedingValueSize),
			Some(0) => return Err(CacheError::ZeroValueSize),
			Some(base) if self.status.exceeds_max_size(base) => return Err(CacheError::ExceedingValueSize),
			Some(_) => {},
		}

		// The one thing the shape still decides: which allocator the value
		// comes from. `BufferDRAM` names the fast tier, `BufferPMEM` the slow
		// one -- see `value::ValueShape`.
		let object = Object::new_in(key, value, V::TIER, ttl);

		let base_size = self.overhead_manager.base_size(&object);
		let dram_resident = self.overhead_manager.dram_resident_size(&object);
		let expiry = object.expiry();
		// Where the bytes were allocated, for the worker's reconcile
		// (`WorkerEvent::Set`).
		let built = object.value().tier();

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
			built,
			// A flat cache queues no migration: nothing to mark.
			0,
			// Nor places anything by tier.
			crate::worker::Placement::Normal,
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

		// No epoch pin, deliberately -- unlike `get`/`get_into`/`peek`. This
		// reads the key, the expiry, the length and the tag bit, never the
		// value's bytes, and the shard guard it holds while doing so keeps the
		// object -- and through its handle everything those live in -- alive.
		// A pin would protect nothing that is read here. (Under `thin_header`
		// the first three are in the tiered item: one remote cache line for a
		// slow object.)
		self.objects
			.get_ref(&hashed_key)
			.is_some_and(|object| object.key_matches(key) && !object.is_expired())
	}

	/// # API change (v5)
	///
	/// This returned a `Shared<V>` -- a refcounted handle onto the value --
	/// until the refcount was removed. It now returns an owned `Vec<u8>`, the
	/// same thing [`Self::get`] returns.
	///
	/// It cannot return a borrow. A value is now a bare pointer whose lifetime
	/// is managed by epoch reclamation, so the only two honest return types
	/// are a copy or a guard object holding the pin open -- and a guard held by
	/// a caller that then blocks would pin the epoch and stall reclamation for
	/// every thread, which is the one failure mode this design has to avoid.
	/// A copy has the same semantics the `Shared` did anyway: a snapshot that
	/// was live at the moment of the lookup.
	pub fn peek(&self, key: &K) -> Result<Vec<u8>, CacheError> {
		let hashed_key = self.hash_key(key);
		let snapshot = match self.objects.get_ref(&hashed_key) {
			Some(object) if object.key_matches(key) && !object.is_expired() =>
				Some(object.snapshot()),

			_ => None,
		};

		let result = match snapshot {
			Some(value) => Ok(value.bytes().to_vec()),
			None => Err(CacheError::KeyNotFound),
		};


		result
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

		// No epoch pin, deliberately -- unlike `get`/`get_into`/`peek`. This
		// reads the key, the expiry, the length and the tag bit, never the
		// value's bytes, and the shard guard it holds while doing so keeps the
		// object -- and through its handle everything those live in -- alive.
		// A pin would protect nothing that is read here. (Under `thin_header`
		// the first three are in the tiered item: one remote cache line for a
		// slow object.)
		match self.objects.get_ref(&hashed_key) {
			Some(object) if object.key_matches(key) && !object.is_expired() =>
				Ok(self.overhead_manager.total_size(&object)),
			_ => Err(CacheError::KeyNotFound),
		}
	}

	pub fn wipe(&self) -> Result<(), CacheError> {
		info!("Wiping cache");

		// The policy worker wipes -- the object map, its stack, the status
		// counters and the tier gauges -- and answers when it is done
		// (`PolicyWorker::handle_wipe`); this thread waits for the answer. It
		// used to clear the map and the status here and leave the stack to
		// the worker, and a `Set` the worker handled in between left a live
		// key its stack no longer tracked. The kick wakes a worker parked on
		// its idle poll (up to 1 s); the wait still includes the events queued
		// ahead of the `Wipe`. The values `clear_counted` drops retire into the
		// worker's epoch bag, which its pass flushes.
		let (ack, done) = crossbeam_channel::bounded(1);
		let sent = self.broadcast(WorkerEvent::Wipe(Some(ack)));

		self.status.kick_policy_worker();

		match done.recv() {
			// Wiped. A failed delivery to another subscriber (a dead TTL
			// worker) is still reported, as it always was.
			Ok(()) => sent,

			// Every sender is gone without an answer: the policy worker is dead
			// (its channel dropped, with the event in it) and the other
			// subscribers have handled or dropped their copies -- the TTL
			// worker within its poll, 1 s at most. Wipe here so the cache is
			// empty all the same, and say it failed.
			Err(_) => {
				let cleared = self.objects.clear_counted(|object| self.overhead_manager.base_size(object));
				self.status.clear(cleared);

				Err(CacheError::Internal)
			},
		}
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
	/// Remove the object stored under this key, after checking that the
	/// object at the hash really is this key's -- a hash collision is not.
	Original(&'a K, HashedKey),

	/// Remove whatever object is stored at this hash, live or not. Capacity
	/// eviction's path: the stack chose the victim and it must go.
	Hashed(HashedKey),

	/// Remove the object at this hash only if it has expired; otherwise leave
	/// it in place and answer `KeyNotFound`. The TTL reaper's path.
	///
	/// A due index entry can be stale. `set` writes the object map and only
	/// then sends the event that retires the old entry, and `ttl` extends the
	/// object's expiry in place before its event moves the entry. In between,
	/// the index still holds the old deadline while the map already holds a
	/// live object -- a re-set with no TTL or a later one, or the same object
	/// with its TTL extended. `Hashed` would delete it. The expiry is tested
	/// under the same lock as the removal, so nothing can replace the object
	/// between the check and the erase.
	Expired(HashedKey),
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
{
	let hashed_key = match maybe_key {
		Some(EraseKey::Original(_, hashed_key)) => hashed_key,
		Some(EraseKey::Hashed(hashed_key)) => hashed_key,
		Some(EraseKey::Expired(hashed_key)) => hashed_key,

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

	// A reap must not take an object that is live again -- see
	// `EraseKey::Expired`. Tested on the occupied entry, under the lock the
	// removal below also holds.
	if matches!(maybe_key, Some(EraseKey::Expired(_))) && !entry.get().is_expired() {
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











// merged_erase_marker
/// `erase` for the merged store.
///
/// The arms map onto who is removing, because in this store a removal is
/// policy work only on the policy worker:
///
///   * `Original` (a client's `del`) and `Expired` (the TTL reaper) are the
///     CLIENT's `MergedStore::take_if`: a value the worker has not linked is
///     freed, a linked one goes DEAD -- its object gone, its slot left on the
///     list for the worker to retire at the `Del`/`Expire` that follows, as a
///     DashMap stack keeps a deleted key until its `Del` -- and no policy state
///     moves;
///   * `Hashed` (the eviction `apply_evictions` pairs with a nomination) is the
///     WORKER's `MergedStore::take_evict`, which unlinks and uncharges in the
///     same operation that removes from the index, so the map-greater-than-
///     stack divergence `ERASE_FALLBACK` counts has no way to occur; it
///     refuses a value the worker has not linked, which is then
///     `KeyNotFound`;
///   * the no-key fallback takes the LRU TAIL rather than an arbitrary object
///     -- `oldest_linked_key`, read-only -- since this store IS the eviction
///     order. Reachable only from `apply_mini_evictions`, which a merged build
///     never runs.
#[cfg(feature = "merged_object_store")]
pub fn erase<K, V>(
	objects: &ObjectMapRef<K, V>,
	status: &StatusRef,
	overhead_manager: &OverheadManagerRef,
	maybe_key: Option<EraseKey<K>>,
) -> Result<(HashedKey, Object<K, V>), CacheError>
where
	K: Eq + TypeSize,
{
	let hashed_key = match maybe_key {
		Some(EraseKey::Original(_, hashed_key)) => hashed_key,
		Some(EraseKey::Hashed(hashed_key)) => hashed_key,
		Some(EraseKey::Expired(hashed_key)) => hashed_key,

		None => {
			crate::ERASE_FALLBACK.fetch_add(1, std::sync::atomic::Ordering::Relaxed);

			let Some(key) = objects.oldest_linked_key() else {
				error!("Object store is empty with non-zero used size");
				return Err(CacheError::Internal);
			};

			key
		},
	};

	// Validate and remove under ONE shard write guard, so a hash collision
	// does not evict the wrong object and a reap does not take an object that
	// is live again (`EraseKey::Expired`). Checking through `get_ref` and then
	// calling `take` would release the lock in between, and a `set` landing
	// there would have its new object removed.
	let taken = match maybe_key {
		Some(EraseKey::Original(key, _)) =>
			objects.take_if(&hashed_key, |object| object.key_matches(key)),

		Some(EraseKey::Expired(_)) =>
			objects.take_if(&hashed_key, |object| object.is_expired()),

		Some(EraseKey::Hashed(_)) | None => objects.take_evict(&hashed_key),
	};

	let Some(object) = taken else {
		return Err(CacheError::KeyNotFound);
	};

	let base_size = overhead_manager.base_size(&object) as i64;

	status.update_base_used_size(-base_size);
	status.decr_num_objects();

	match !object.is_expired() {
		true => Ok((hashed_key, object)),
		false => Err(CacheError::KeyNotFound),
	}
}

#[cfg(not(any(feature = "global_hashtable_pmem", feature = "hashbrown_dram", feature = "merged_object_store")))]
pub fn erase<K, V>(
	objects: &ObjectMapRef<K, V>,
	status: &StatusRef,
	overhead_manager: &OverheadManagerRef,
	maybe_key: Option<EraseKey<K>>,
) -> Result<(HashedKey, Object<K, V>), CacheError>
where
	K: Eq + TypeSize,
{
	let hashed_key = match maybe_key {
		Some(EraseKey::Original(_, hashed_key)) => hashed_key,
		Some(EraseKey::Hashed(hashed_key)) => hashed_key,
		Some(EraseKey::Expired(hashed_key)) => hashed_key,

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

	// A reap must not take an object that is live again -- see
	// `EraseKey::Expired`. Tested on the occupied entry, under the lock the
	// removal below also holds.
	if matches!(maybe_key, Some(EraseKey::Expired(_))) && !entry.get().is_expired() {
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

/// Builds the object map every hybrid-cache design stores its objects in --
/// mirrors Shape B's
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

	#[cfg(all(not(feature = "hashbrown_dram"), not(feature = "merged_object_store")))]
	{
		Arc::new(DashMap::with_hasher(NoHasher::default()))
	}

	#[cfg(feature = "merged_object_store")]
	{
		Arc::new(crate::merged_store::MergedStore::new())
	}
}

/// Installs a tiered cache's admission configuration on its status, before its
/// policy worker is built (S5): validated; pinned to the per-object model on a
/// thread a lib test pinned (`test_overheads`); and FORCED to it under
/// `PAPER_DISABLE_SHARED_OVERHEAD=1`, the mechanics tests' switch, which also
/// makes omega 0 -- so those caches run exactly the reservation they ran
/// before S5: none, beyond a ghost's.
///
/// `gate` is `None` from a plain constructor, which takes the default
/// configuration. That one is not refused when its byte-gate levels cannot
/// hold under an environment-chosen drain target (`FAST_TIER_DRAIN_TARGET` of
/// 0.99 or more, against the default 1% near band): the cache starts with the
/// byte gate disabled (`GateState::Bands`) and says so once on stderr (design
/// Q10). An explicit configuration is refused (`InvalidGateConfig`).
#[cfg(feature = "hybrid_cache_common")]
fn install_gate(status: &AtomicStatus, gate: Option<GateConfig>) -> Result<(), CacheError> {
	let explicit = gate.is_some();

	#[allow(unused_mut)]
	let mut gate = gate.unwrap_or_default();

	match gate.validate() {
		Ok(()) => {},

		Err(_) if !explicit && !gate.bands_hold() => eprintln!(
			"paper-cache: the fast-tier byte gate is disabled (GateState::Bands): the drain target {} plus \
			 the gate's near band {} is at least 1, so the settle target would not be below the near level",
			crate::worker::drain_target::ratio(),
			gate.near_frac,
		),

		Err(error) => return Err(error),
	}

	#[cfg(test)]
	if crate::object::overhead::test_overheads::per_object_pinned() {
		gate.metadata_model = crate::gate::MetadataModel::PerObject;
	}

	status.gate().set_config(gate);

	if std::env::var_os("PAPER_DISABLE_SHARED_OVERHEAD").is_some_and(|v| v == "1") {
		status.gate().force_per_object();
	}

	Ok(())
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
/// not be asserted on.
fn params_ok(policy: PaperPolicy) -> bool {
	match policy {
		PaperPolicy::LruLfuCompactHybrid(promote_k) => promote_k != 0,

		// The one two-ratio design: BOTH must be in range, so it cannot
		// join the single-ratio group below.
		PaperPolicy::TwoQFullFastAdmissionCompactHybrid(k_in, k_out) => {
			(0.0..=1.0).contains(&k_in) && (0.0..=1.0).contains(&k_out)
		},

		// The 2Q family keeps an INCLUSIVE upper bound. No 2Q stack
		// derives a budget from `1 - k_in`: `fifo_capacity` is
		// `k_in * max_size` and the main queue is bounded by the
		// cache's overall `max_size`, so `k_in == 1.0` gives the FIFO
		// queue the whole cache -- extreme, but every queue still has
		// capacity and eviction drains the FIFO tail unconditionally.
		PaperPolicy::TwoQCompactHybrid(r)
		| PaperPolicy::TwoQFastAdmissionCompactHybrid(r)
		| PaperPolicy::TwoQFastAdmissionReprieveCompactHybrid(r)
		| PaperPolicy::TwoQGhostCompactHybrid(r) => (0.0..=1.0).contains(&r),

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
		PaperPolicy::S3FifoFaithfulCompactHybrid(r)
		| PaperPolicy::S3FifoFaithfulFastAdmissionCompactHybrid(r)
		| PaperPolicy::S3FifoFaithfulReprieveCompactHybrid(r)
		| PaperPolicy::S3FifoFaithfulFastAdmissionReprieveCompactHybrid(r)
		| PaperPolicy::S3FifoCompactHybrid(r)
		| PaperPolicy::S3FifoGhostCompactHybrid(r)
		| PaperPolicy::S3FifoGhostLazyDemotionCompactHybrid(r)
		| PaperPolicy::S3FifoGhostLazyDemotionFastAdmissionCompactHybrid(r)
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
		PaperPolicy::S3FifoLazyDemotionFastAdmissionMidpointReprieveCompactHybrid(r)
		| PaperPolicy::S3FifoLazyDemotionFastAdmissionReprieveCompactHybrid(r)
		| PaperPolicy::S3FifoLazyDemotionReprieveCompactHybrid(r)
		| PaperPolicy::S3FifoLazyDemotionFastAdmissionSplitSlowReprieveCompactHybrid(r) => {
			(0.0..=1.0).contains(&r)
		},

		_ => true,
	}
}

fn s3_fifo_queue_budgets(policy: PaperPolicy) -> Option<(f64, bool)> {
	match policy {
		// These nine size main at `(1 - ratio) * max_size`, mirroring
		// `SThreeFifoStack`, and gate eviction on its fullness.
		PaperPolicy::S3FifoCompactHybrid(r)
		| PaperPolicy::S3FifoGhostCompactHybrid(r)
		| PaperPolicy::S3FifoGhostLazyDemotionCompactHybrid(r)
		| PaperPolicy::S3FifoGhostLazyDemotionFastAdmissionCompactHybrid(r)
		| PaperPolicy::S3FifoFaithfulCompactHybrid(r)
		| PaperPolicy::S3FifoFaithfulFastAdmissionCompactHybrid(r)
		| PaperPolicy::S3FifoFaithfulReprieveCompactHybrid(r)
		| PaperPolicy::S3FifoFaithfulFastAdmissionReprieveCompactHybrid(r)
		| PaperPolicy::S3FifoGhostLazyDemotionFastAdmissionMidpointCompactHybrid(r) => Some((r, true)),

		// The reprieve stacks: one-access budget only.
		PaperPolicy::S3FifoLazyDemotionFastAdmissionMidpointReprieveCompactHybrid(r)
		| PaperPolicy::S3FifoLazyDemotionFastAdmissionReprieveCompactHybrid(r)
		| PaperPolicy::S3FifoLazyDemotionReprieveCompactHybrid(r)
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
/// design's `new_sized_compact`/`with_hasher_sized_compact`, which take three sizing scalars
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
	/// The size-split design has its own constructor,
	/// [`Self::new_sized_compact`], because it takes three sizing scalars
	/// rather than one.
	///
	/// # Errors
	///
	/// [`CacheError::InvalidPolicy`] if `policy` is not a hybrid design (or
	/// is the size-split design, which `new_sized_compact` serves), if its
	/// parameters are
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
		Self::new_hybrid(max_size, fast_tier_size, policy, hasher, None)
	}

	/// [`Self::new`], with an admission configuration (S5): the metadata
	/// model, the metadata floor, what a new key whose metadata would not fit
	/// gets, and the waits. See [`GateConfig`].
	///
	/// # Errors
	///
	/// As [`Self::new`], and [`CacheError::InvalidGateConfig`] for a
	/// configuration `GateConfig::validate` refuses.
	pub fn new_with_gate(
		max_size: CacheSize,
		fast_tier_size: CacheTierSize,
		policy: PaperPolicy,
		gate: GateConfig,
	) -> Result<Self, CacheError> {
		Self::new_hybrid(max_size, fast_tier_size, policy, Default::default(), Some(gate))
	}

	/// [`Self::new_with_gate`] with the supplied hasher.
	pub fn with_hasher_and_gate(
		max_size: CacheSize,
		fast_tier_size: CacheTierSize,
		policy: PaperPolicy,
		hasher: S,
		gate: GateConfig,
	) -> Result<Self, CacheError> {
		Self::new_hybrid(max_size, fast_tier_size, policy, hasher, Some(gate))
	}

	// The size-split design doesn't call this: it needs three sizing
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
		gate: Option<GateConfig>,
	) -> Result<Self, CacheError> {
		if max_size == 0 {
			return Err(CacheError::ZeroCacheSize);
		}

		// The size-split design needs three sizing scalars and has its own
		// constructor (`new_sized_compact`); everything non-hybrid is simply
		// not a tiered design.
		if !policy.is_hybrid()
			|| matches!(policy, PaperPolicy::LruSizedCompactHybrid)
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

		// A TIERED cache: counted in `phys::live_tiered_caches` until its
		// status is freed, and its per-object reservation recorded for
		// `effective_fast_capacity` -- the figure `init_policy_stack` hands
		// the stack, from the same function.
		status.register_tiered_cache(
			crate::object::overhead::get_hybrid_dram_shared_overhead(&policy) as CacheSize,
		);

		// S5: the admission configuration, on the status before the policy
		// worker exists -- its construction publishes the first figures.
		install_gate(&status, gate)?;

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
		// a byte-identical object at a new address. `LfuCompactHybridStack`
		// did exactly that on every latched admission (445,465,067 migrations
		// against ~448M sets on cluster12 before it was fixed at source), and
		// `TwoQCompactHybridStack` still reaches this case legitimately under a
		// lookaside workload: `admission_tier` returns `Fast` for a re-set --
		// correct, since the key is now MRU -- so `set()` has already built
		// the bytes in DRAM by the time `touch_main_fast` emits its
		// `(key, Tier::Fast)` promotion.
		let (worker_fanout, worker_handles) = WorkerFanout::new_with_tier_migration(
			&objects,
			&status,
			&overhead_manager,
		)?;

		// Annotated, and load-bearing. Nothing else in this function mentions
		// `V` before the `broadcast` call below, and an
		// unresolved `V` makes that call ambiguous between this block and the
		// flat `impl<K, V, S> ... where V: ValueShape` one (E0034: both are
		// inherent candidates, and the where-clause can only rule one out once
		// `V` is known).
		let cache: Self = PaperCache {
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

		let snapshot = match self.objects.get_ref(&hashed_key) {
			Some(object) if object.key_matches(key) && !object.is_expired() =>
				Some(object.snapshot()),
			_ => None,
		};

		// The tier the hit is served from -- the snapshot's tag -- or `None`
		// on a miss: the policy worker's heal needs it (`WorkerEvent::Get`).
		let served = snapshot.as_ref().map(|value| value.tier());

		let result = match snapshot {
			Some(value) => {
				self.status.incr_hits();
				// The tier the copy below reads from: the snapshot's tag.
				self.status.incr_served_hit(value.tier());
				Ok(value.bytes().to_vec())
			},

			None => {
				self.status.incr_misses();
				Err(CacheError::KeyNotFound)
			},
		};


		self.broadcast(WorkerEvent::Get(hashed_key, served))?;

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

		let snapshot = match self.objects.get_ref(&hashed_key) {
			Some(object) if object.key_matches(key) && !object.is_expired() =>
				Some(object.snapshot()),
			_ => None,
		};
		let t2 = if prof { Some(std::time::Instant::now()) } else { None };

		// Which tier served this hit. Read only by the sampled profiler below.
		#[allow(unused_mut)]
		let mut gi_fast: u64 = 1;

		// The tier the hit is served from -- the snapshot's tag -- or `None`
		// on a miss: the policy worker's heal needs it (`WorkerEvent::Get`).
		let served = snapshot.as_ref().map(|value| value.tier());

		let result = match snapshot {
			Some(value) => {
				self.status.incr_hits();
				self.status.incr_served_hit(value.tier());
				out.clear();
				gi_fast = if value.is_fast() { 1 } else { 0 };
				out.extend_from_slice(value.bytes());
				Ok(())
			},

			None => {
				self.status.incr_misses();
				Err(CacheError::KeyNotFound)
			},
		};
		let t3 = if prof { Some(std::time::Instant::now()) } else { None };

		self.broadcast(WorkerEvent::Get(hashed_key, served))?;

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
	///
	/// Decided from the value's LENGTH before anything is allocated
	/// (`begin_set`, S5): the size checks; for a NEW key near the metadata
	/// ceiling, the metadata cap; the tier, by `hybrid_policy::admission_tier`,
	/// whose match arm for the cache's policy carries that design's admission
	/// rule; structural slow placement -- a value larger than an empty fast
	/// tier is built in the slow tier and placed there; and, for a value to be
	/// built in the fast tier, the byte gate, which holds it to the tier's
	/// budget and WAITS, FIFO, while the tier is over it, until demotions free
	/// room (`GateConfig::mode`). Then the value is built and published
	/// (`commit`).
	///
	/// # Errors
	///
	/// [`CacheError::ExceedingValueSize`] for a value larger than the cache's
	/// maximum size (refused before it is built); [`CacheError::ZeroValueSize`]
	/// as before (a zero base size, which no object has -- an empty value is
	/// stored); [`CacheError::MetadataOverflow`] for a new key whose metadata
	/// would not fit (see [`GateConfig::on_metadata_overflow`]);
	/// [`CacheError::FastTierStalled`] when the set waited for room in the fast
	/// tier and nothing was freed for the gate's `stall_window` once the policy
	/// worker had caught up, or the worker hung (see
	/// [`GateConfig::on_stall`]); [`CacheError::Internal`] if the policy worker
	/// is gone while this set waits, or a worker could not be told.
	pub fn set(&self, key: K, value: &[u8], ttl: Option<u32>) -> Result<(), CacheError> {
		let permit = self.begin_set(&key, value.len(), ttl)?;

		self.commit(permit, key, value)
	}

	/// The admission half of `set` (S5): everything decided from `key`, the
	/// value's length and its ttl, before anything is allocated -- the size
	/// checks, the metadata cap (waiting in the metadata lane under
	/// `EvictToFit`), `admission_tier`, the structural check
	/// (`gate::decide`) -- and, for a value to be built fast, the byte gate
	/// (`admit_bytes`, B2), which may wait in the bytes lane. No lock is held
	/// across a wait, and nothing has been allocated or sent by then.
	pub(crate) fn begin_set(&self, key: &K, len: usize, ttl: Option<u32>) -> Result<crate::gate::SetPermit<'_>, CacheError> {
		use crate::gate::{self, Verdict};

		let hashed = self.hash_key(key);

		// 0. The size checks, with today's predicates.
		let Some(base) = self.overhead_manager.base_size_for(key, len, ttl) else {
			return Err(CacheError::ExceedingValueSize);
		};

		if base == 0 {
			return Err(CacheError::ZeroValueSize);
		}

		if self.status.exceeds_max_size(base) {
			return Err(CacheError::ExceedingValueSize);
		}

		let sizes = gate::Sizes {
			base,
			resident: self.overhead_manager.dram_resident_size_for(key, ttl),
			value: crate::phys::value_charge::<K>(len as ObjectSize),
		};

		let gate = self.status.gate();
		let mut lane: Option<gate::LaneGuard<'_>> = None;
		let mut last_evicted: Option<u64> = None;

		// 1-3. The metadata cap -- a new key at the ceiling waits in the
		// metadata lane under `EvictToFit` -- the design's tier and the
		// structural check.
		let (tier, placement) = loop {
			let head = lane.as_ref().is_some_and(|place| place.is_head());

			match gate::decide(&self.status, &self.objects, hashed, &sizes, head)? {
				Verdict::Admit { tier, placement } => break (tier, placement),

				// `EvictToFit`: a new key at the ceiling waits in FIFO order.
				Verdict::NeedsRoom => {
					if gate.worker_gone() {
						return Err(CacheError::Internal);
					}

					let config = gate.config();

					// `stall_window` 0 never waits: a MakeRoom would evict the
					// policy's victim for a set that then fails without waiting
					// for it (the correctness review). It fails at once and
					// evicts nothing, as `Error` does.
					if config.stall_window.is_zero() {
						gate.count_overflow();
						return Err(CacheError::MetadataOverflow);
					}

					match &lane {
						None => lane = Some(gate.meta_lane.enqueue()),

						// Behind another new key: woken when the head leaves.
						Some(place) if !place.is_head() => gate::park(std::time::Duration::from_millis(100)),

						// The head: ask the worker to evict for it, and wait. A
						// request that evicted nothing, with the ceiling still
						// shut, is the end: there is nothing to make room from.
						Some(_) => {
							if last_evicted == Some(0) {
								gate.count_make_room_failure();
								return Err(CacheError::MetadataOverflow);
							}

							let request = gate.next_room_request();

							gate.count_make_room_request();
							self.broadcast(WorkerEvent::MakeRoom(request))?;
							self.status.kick_policy_worker();

							last_evicted = Some(gate::await_room(gate, request, &config)?);
						},
					}
				},
			}
		};

		// The metadata lane's place, if any, is released here -- waking the
		// next -- before the byte gate: a set never waits in one lane holding a
		// place in the other.
		drop(lane);

		// 4. The byte gate (B2), for a value to be built fast.
		let (tier, placement, reservation) = match tier {
			Tier::Fast => self.admit_bytes(hashed, &sizes)?,
			Tier::Slow => (tier, placement, gate::Reservation::none()),
		};

		Ok(gate::SetPermit { hashed, tier, placement, len, ttl, sizes, reservation })
	}

	/// Step 4 of `begin_set` (S5, commit B2): the byte gate, for a value to be
	/// built FAST (see `crate::gate`'s module doc). One attempt -- the fast
	/// path, one relaxed load, or the exact path -- and, when the tier is over
	/// its budget, a WAIT in FIFO order in the bytes lane, until demotions free
	/// room: woken by a consumer's landed demotion, the worker's pass, a
	/// released reservation or a wipe, and at every wake the tier and the
	/// structural check decided again (a value that no longer fits even an
	/// empty tier leaves as a structural set, built slow). The no-progress
	/// watchdog ends a wait with nothing freed per `GateConfig::on_stall`;
	/// `stall_window` 0 acts at once, without waiting. `Internal` if the policy
	/// worker is gone. Returns the tier and placement to build with and the
	/// bytes held for the value until it is built.
	fn admit_bytes(
		&self,
		hashed: HashedKey,
		sizes: &crate::gate::Sizes,
	) -> Result<(Tier, crate::worker::Placement, crate::gate::Reservation<'_>), CacheError> {
		use crate::gate::{self, Bytes, Watch};
		use crate::worker::Placement;

		let gate = self.status.gate();
		let v = sizes.value;
		let p = crate::phys::fast_bytes_signed;
		let kick = || self.status.kick_policy_worker();

		if let Bytes::Admit(reservation) = gate.admit_bytes(v, false, p, kick) {
			return Ok((Tier::Fast, Placement::Normal, reservation));
		}

		let config = gate.config();

		// `stall_window` 0: never wait -- a set that would wait acts at once.
		if config.stall_window.is_zero() {
			return gate.on_stall(v, config.on_stall);
		}

		let mut waiter = gate::Waiter::enqueue(gate);
		self.status.kick_policy_worker();

		loop {
			if gate.worker_gone() {
				return Err(CacheError::Internal);
			}

			// 2-3 again: the tier and the structural check can have moved while
			// this set waited.
			let (tier, placement) = gate::place(&self.status, &self.objects, hashed, sizes, gate.word());

			if tier == Tier::Slow {
				return Ok((tier, placement, gate::Reservation::none()));
			}

			// No kick from inside the wait: the worker polls SHORT while any
			// set waits and every pass it ends wakes the head, so a kick here
			// only started its next pass at once -- the head and the worker
			// woke each other flat out, ~165,000 passes a second, for the whole
			// wait (the critic of commit C's review). The first attempt's near
			// kick and the kick at enqueue stay.
			if let Bytes::Admit(reservation) = gate.admit_bytes(v, waiter.is_head(), p, || {}) {
				waiter.admitted();
				return Ok((Tier::Fast, Placement::Normal, reservation));
			}

			let config = gate.config();

			match waiter.watch(&config) {
				Watch::Park(timeout) => gate::park(timeout),
				Watch::Stalled => return gate.on_stall(v, config.on_stall),
			}
		}
	}

	/// The build-and-publish half of `set` (S5): builds exactly the object
	/// `begin_set` decided on -- in its tier -- inserts it, and sends its
	/// `Set`, carrying the placement. Then the set-path kick: a policy worker
	/// parked on its long idle poll is woken by the first set after it.
	pub(crate) fn commit(&self, permit: crate::gate::SetPermit<'_>, key: K, value: &[u8]) -> Result<(), CacheError> {
		let crate::gate::SetPermit { hashed: hashed_key, tier, placement, len, ttl, sizes, reservation } = permit;

		debug_assert_eq!(value.len(), len, "commit builds the value begin_set checked");
		debug_assert_eq!(self.hash_key(&key), hashed_key, "commit builds the key begin_set checked");

		let object = Object::new_in(key, value, tier, ttl);

		// B2: the value is built -- charged to P if fast -- so the bytes the byte
		// gate held for it go back (waking the lane's head if anyone waits).
		drop(reservation);
		let base_size = sizes.base;
		let dram_resident = sizes.resident;
		let expiry = object.expiry();
		// Where the bytes were allocated, for the worker's reconcile
		// (`WorkerEvent::Set`).
		let built = object.value().tier();

		debug_assert_eq!(self.overhead_manager.base_size(&object), base_size, "base_size_for is base_size");
		debug_assert_eq!(self.overhead_manager.dram_resident_size(&object), dram_resident);

		self.status.incr_sets();

		// The new-key rule's mark (`WorkerEvent::Set`): the landed count of
		// this key's migration bucket, read BEFORE the insert publishes the
		// value, so that a migration landing on the value is ordered after
		// this read. One load.
		let mark = self.status.migration_in_flight().mark(hashed_key);

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
			// A new object: near the key ceiling, the next new key's set
			// checks it (`gate::META_NEAR`).
			let before = self.status.incr_num_objects();
			self.status.gate().note_count(before.saturating_add(1));

			base_size as i64
		};

		self.status.update_base_used_size(base_size_delta);
		self.broadcast(WorkerEvent::Set(
			hashed_key,
			base_size,
			dram_resident,
			expiry,
			old_object_info,
			built,
			mark,
			placement,
		))?;

		self.kick_idle_worker();

		Ok(())
	}

	/// The set-path kick (S5): after a `Set` is in the policy worker's
	/// channel, wakes the worker if it is parked on its long idle poll. A
	/// Dekker pair with `PolicyWorker::delay_event_loop`: the worker writes
	/// its idle bit, fences, and parks only if its channel is empty; this
	/// thread wrote the channel, fences, and reads the bit -- so at least one
	/// sees the other: the worker does not park, or it is kicked. The `swap`
	/// makes it one kick per idle spell. Per set: a fence and a relaxed load;
	/// the kick (a lock and an unpark) only on the first set after an idle
	/// spell. It restores the real-time bound S4 removed for the merged store
	/// and closes S1's up-to-1 s window in both stores.
	fn kick_idle_worker(&self) {
		std::sync::atomic::fence(std::sync::atomic::Ordering::SeqCst);

		let gate = self.status.gate();

		if gate.worker_idle.load(std::sync::atomic::Ordering::Relaxed)
			&& gate.worker_idle.swap(false, std::sync::atomic::Ordering::AcqRel)
		{
			self.status.kick_policy_worker();
			gate.count_idle_kick();
		}
	}

	/// The cache's admission configuration (S5). See [`GateConfig`].
	#[must_use]
	pub fn gate_config(&self) -> GateConfig {
		self.status.gate().config()
	}

	/// Replaces the cache's admission configuration (S5): validated, stored at
	/// once (a set reads the overflow mode, `on_stall` and the waits at once),
	/// and the policy worker kicked; the metadata model and floor, and the byte
	/// gate's mode and levels, take effect at its next pass, which republishes
	/// eff, the key ceiling and the levels -- except that turning the byte gate
	/// `Off` disables it at once, releasing every waiter. The model is forced to
	/// `PerObject` under `PAPER_DISABLE_SHARED_OVERHEAD=1` whatever this says.
	///
	/// # Errors
	///
	/// [`CacheError::InvalidGateConfig`] for a configuration
	/// `GateConfig::validate` refuses.
	pub fn set_gate_config(&self, gate: GateConfig) -> Result<(), CacheError> {
		gate.validate()?;

		self.status.gate().set_config(gate);

		if gate.mode == crate::gate::GateMode::Off {
			self.status.gate().disable(crate::gate::GateState::Off);
		}

		self.status.kick_policy_worker();

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

		// No epoch pin, deliberately -- unlike `get`/`get_into`/`peek`. This
		// reads the key, the expiry, the length and the tag bit, never the
		// value's bytes, and the shard guard it holds while doing so keeps the
		// object -- and through its handle everything those live in -- alive.
		// A pin would protect nothing that is read here. (Under `thin_header`
		// the first three are in the tiered item: one remote cache line for a
		// slow object.)
		self.objects
			.get_ref(&hashed_key)
			.is_some_and(|object| object.key_matches(key) && !object.is_expired())
	}

	/// Gets (peeks) the value associated with the supplied key without
	/// altering any of the cache's internal queues (including tier — a peek
	/// never triggers a promotion). If the key was not found in the cache,
	/// returns a [`CacheError`].
	/// # API change (v5)
	///
	/// This returned a `Shared<V>` -- a refcounted handle onto the value --
	/// until the refcount was removed. It now returns an owned `Vec<u8>`, the
	/// same thing [`Self::get`] returns.
	///
	/// It cannot return a borrow. A value is now a bare pointer whose lifetime
	/// is managed by epoch reclamation, so the only two honest return types
	/// are a copy or a guard object holding the pin open -- and a guard held by
	/// a caller that then blocks would pin the epoch and stall reclamation for
	/// every thread, which is the one failure mode this design has to avoid.
	/// A copy has the same semantics the `Shared` did anyway: a snapshot that
	/// was live at the moment of the lookup.
	pub fn peek(&self, key: &K) -> Result<Vec<u8>, CacheError> {
		let hashed_key = self.hash_key(key);
		let snapshot = match self.objects.get_ref(&hashed_key) {
			Some(object) if object.key_matches(key) && !object.is_expired() =>
				Some(object.snapshot()),

			_ => None,
		};

		let result = match snapshot {
			Some(value) => Ok(value.bytes().to_vec()),
			None => Err(CacheError::KeyNotFound),
		};


		result
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

		// No epoch pin, deliberately -- unlike `get`/`get_into`/`peek`. This
		// reads the key, the expiry, the length and the tag bit, never the
		// value's bytes, and the shard guard it holds while doing so keeps the
		// object -- and through its handle everything those live in -- alive.
		// A pin would protect nothing that is read here. (Under `thin_header`
		// the first three are in the tiered item: one remote cache line for a
		// slow object.)
		match self.objects.get_ref(&hashed_key) {
			Some(object) if object.key_matches(key) && !object.is_expired() =>
				Ok(self.overhead_manager.total_size(&object)),

			_ => Err(CacheError::KeyNotFound),
		}
	}

	/// Deletes all objects in the cache and sets the cache's used size to zero.
	///
	/// The policy worker does it -- the object map, its stack, the status
	/// counters and the tier gauges -- and this returns when it has: at once
	/// after it the cache reads empty, and a set racing it cannot leave a key
	/// in the map that the stack does not track. It waits for the events queued
	/// ahead of the wipe; a worker idle on its long poll is kicked. `Err` if the
	/// policy worker is gone (the cache is then wiped here), or if another
	/// worker could not be told.
	pub fn wipe(&self) -> Result<(), CacheError> {
		info!("Wiping cache");

		// The policy worker wipes -- the object map, its stack, the status
		// counters and the tier gauges -- and answers when it is done
		// (`PolicyWorker::handle_wipe`); this thread waits for the answer. It
		// used to clear the map and the status here and leave the stack to
		// the worker, and a `Set` the worker handled in between left a live
		// key its stack no longer tracked. The kick wakes a worker parked on
		// its idle poll (up to 1 s); the wait still includes the events queued
		// ahead of the `Wipe`. The values `clear_counted` drops retire into the
		// worker's epoch bag, which its pass flushes.
		let (ack, done) = crossbeam_channel::bounded(1);
		let sent = self.broadcast(WorkerEvent::Wipe(Some(ack)));

		self.status.kick_policy_worker();

		match done.recv() {
			// Wiped. A failed delivery to another subscriber (a dead TTL
			// worker) is still reported, as it always was.
			Ok(()) => sent,

			// Every sender is gone without an answer: the policy worker is dead
			// (its channel dropped, with the event in it) and the other
			// subscribers have handled or dropped their copies -- the TTL
			// worker within its poll, 1 s at most. Wipe here so the cache is
			// empty all the same, and say it failed.
			Err(_) => {
				let cleared = self.objects.clear_counted(|object| self.overhead_manager.base_size(object));
				self.status.clear(cleared);

				Err(CacheError::Internal)
			},
		}
	}

	/// Resizes the cache's overall maximum size.
	/// If the supplied size is zero, returns a [`CacheError`].
	///
	/// Note this is the *overall* cache capacity, independent of the
	/// fast-tier budget — see [`Self::set_fast_tier_size`]. (The 2Q designs
	/// additionally rescale their FIFO queue's byte budget proportionally,
	/// inside `TwoQCompactHybridStack::resize` -- not this method.)
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

		// S5: eff moved with F; the worker republishes it at its next pass.
		self.status.kick_policy_worker();

		Ok(())
	}

	/// Returns the current fast-tier byte budget.
	#[must_use]
	pub fn fast_tier_size(&self) -> CacheSize {
		self.status.fast_tier_capacity()
	}

	/// eff (S5): the fast tier's budget for VALUE bytes, `F - M_model`,
	/// saturating, as the policy worker last published it -- the figure the
	/// settles, the structural check and the metadata cap read. See
	/// `AtomicStatus::effective_fast_capacity`.
	#[must_use]
	pub fn effective_fast_capacity(&self) -> CacheSize {
		self.status.effective_fast_capacity()
	}

	/// M, the bytes this cache's own DRAM metadata structures hold (S5a), in
	/// jemalloc's usable-size unit: the object map's, the policy stack's and
	/// one value header per live object, as the policy worker last published
	/// it. See `crate::meta`.
	#[must_use]
	pub fn dram_metadata_bytes(&self) -> u64 {
		self.status.dram_metadata_bytes()
	}

	/// M's parts, and the structures on the slow node reported apart. See
	/// [`DramMetadata`].
	#[must_use]
	pub fn dram_metadata(&self) -> DramMetadata {
		self.status.dram_metadata()
	}

	/// `F - M`, saturating: the fast tier's budget for value bytes with the
	/// MEASURED metadata taken off, beside `effective_fast_capacity`'s
	/// modelled `F - L * omega`. Reporting only at this step (S5a); S5
	/// switches the budget's consumers onto it.
	#[must_use]
	pub fn effective_fast_capacity_measured(&self) -> CacheSize {
		self.status.effective_fast_capacity_measured()
	}

	/// Returns the active hybrid design's tier-movement counters and live
	/// tier gauges, in a design-neutral shape.
	///
	/// The only stats accessor there is: the per-design
	/// `<design>_hybrid_stats()` methods and the `<Design>HybridStats` aliases
	/// were removed with the runtime-policy unification, leaving this one
	/// `HybridStats` struct. The 8 size-split gauges read zero unless the
	/// cache is running `LruSizedCompactHybrid`.
	#[must_use]
	pub fn hybrid_stats(&self) -> HybridStats {
		self.status.hybrid_stats()
	}

	/// DIAGNOSTIC: every live value's bytes against where the policy stack
	/// places its key -- how many values, and how many bytes (the stacks'
	/// unit), are stranded (in DRAM, placed slow), lagging (in CXL, placed
	/// fast) or untracked. See [`phys::PlacementAudit`].
	///
	/// It BLOCKS THE POLICY WORKER for the whole run, and this thread waits for
	/// it: the worker first lands every migration it has decided -- its pending
	/// drain, then a flush that waits for the migration consumers to finish
	/// their backlog -- and then walks the entire object map, one
	/// `placement_of` lookup per value, before it takes another event. It is
	/// handled where its event falls, possibly mid-batch: the eviction pass
	/// that ends each batch has not run, so a cache over its size still holds
	/// -- and the audit counts -- the values that pass will evict. The walk
	/// holds each map shard's read lock while it reads that shard, so a writer
	/// to it waits. In the hashbrown build it holds the map's ONE
	/// `std::sync::RwLock` read guard for the whole walk, and that lock
	/// prefers writers: once a `set` is waiting for it, every `get` waits too,
	/// so the whole cache stalls for the walk. For end-of-run checks and
	/// tests, not for a hot path.
	///
	/// EXACT ONLY AT CLIENT QUIESCENCE. While clients run, a value set, moved
	/// or deleted during the walk is read before or after the change, and a
	/// value whose `Set` the worker has not taken yet is reported untracked
	/// (or, over a tracked key, against its old placement).
	///
	/// `None` if the policy worker is gone.
	pub fn placement_audit(&self) -> Option<phys::PlacementAudit> {
		let (reply, answer) = crossbeam_channel::bounded(1);

		self.broadcast(WorkerEvent::Audit(reply)).ok()?;

		// A worker parked on its idle poll would otherwise take up to 1 s to
		// see the request.
		self.status.kick_policy_worker();

		answer.recv().ok()
	}

	/// DIAGNOSTIC: migrations handed to this cache's consumers and not
	/// finished yet, summed over the per-key buckets the reconcile's new-key
	/// and heal rules read (`migration_queue::InFlight`). The hand-offs and
	/// the finishes balance, so this is 0 whenever the queue is idle -- at
	/// quiescence, and after an audit's flush -- and always 0 with
	/// `MIGRATION_QUEUE_THREADS=0`. The exception is a migration consumer
	/// thread that has died: the entries still in its channel are never
	/// finished, and stay counted here for the cache's life -- the same class
	/// as the queue's `processed` count, which then never catches up, so a
	/// flush (an audit's included) would not return. One load per bucket
	/// (16,384).
	pub fn migrations_in_flight(&self) -> u64 {
		self.status.migration_in_flight().total_pending()
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

			// The tag bit, read off the word inside the object under the shard
			// guard. No pin: nothing here follows the pointer.
			Some(object.value().tier())
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
/// admission/promotion/demotion/eviction semantics as the plain segmented-LRU
/// design, but each tier's bookkeeping is split into two independently-tracked
/// segments ("small"/"large") by object size. See
/// `lru_sized_compact_hybrid_stack.rs`'s module doc.
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
	/// Creates an empty `PaperCache` running
	/// `PaperPolicy::LruSizedCompactHybrid`, with the given overall `max_size`
	/// and initial small/large fast-segment byte budgets and
	/// size-classification threshold (each independently adjustable afterward
	/// via [`Self::set_fast_tier_size`]/[`Self::set_large_fast_tier_size`]/
	/// [`Self::set_size_threshold`]). An object whose size is strictly below
	/// `size_threshold` routes to the small segment on admission, promotion,
	/// or a reclassifying overwrite; at or above routes to the large segment.
	///
	/// # Errors
	///
	/// Returns [`CacheError::ZeroCacheSize`] if `max_size` is zero, or
	/// [`CacheError::InvalidFastTierSize`] if either `small_fast_tier_size`
	/// or `large_fast_tier_size` resolves to zero bytes or exceeds
	/// `max_size` (checked independently -- there is no requirement that
	/// their sum stay under `max_size`). `size_threshold` is never rejected.
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
		Self::new_sized_hybrid(max_size, small_fast_tier_size, large_fast_tier_size, size_threshold, PaperPolicy::LruSizedCompactHybrid, hasher, None)
	}

	/// [`Self::new_sized_compact`], with an admission configuration (S5).
	/// See [`Self::new_with_gate`].
	pub fn new_sized_compact_with_gate(
		max_size: CacheSize,
		small_fast_tier_size: CacheTierSize,
		large_fast_tier_size: CacheTierSize,
		size_threshold: CacheTierSize,
		gate: GateConfig,
	) -> Result<Self, CacheError> {
		Self::new_sized_hybrid(max_size, small_fast_tier_size, large_fast_tier_size, size_threshold, PaperPolicy::LruSizedCompactHybrid, Default::default(), Some(gate))
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
		gate: Option<GateConfig>,
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

		// As in `new_hybrid`: a tiered cache, and its per-object reservation.
		status.register_tiered_cache(
			crate::object::overhead::get_hybrid_dram_shared_overhead(&policy) as CacheSize,
		);

		// As in `new_hybrid`: the admission configuration, before the worker.
		install_gate(&status, gate)?;

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
		// a byte-identical object at a new address. `LfuCompactHybridStack`
		// did exactly that on every latched admission (445,465,067 migrations
		// against ~448M sets on cluster12 before it was fixed at source), and
		// `TwoQCompactHybridStack` still reaches this case legitimately under a
		// lookaside workload: `admission_tier` returns `Fast` for a re-set --
		// correct, since the key is now MRU -- so `set()` has already built
		// the bytes in DRAM by the time `touch_main_fast` emits its
		// `(key, Tier::Fast)` promotion.
		let (worker_fanout, worker_handles) = WorkerFanout::new_with_tier_migration(
			&objects,
			&status,
			&overhead_manager,
		)?;

		// Annotated, and load-bearing. Nothing else in this function mentions
		// `V` before the `broadcast` call below, and an
		// unresolved `V` makes that call ambiguous between this block and the
		// flat `impl<K, V, S> ... where V: ValueShape` one (E0034: both are
		// inherent candidates, and the where-clause can only rule one out once
		// `V` is known).
		let cache: Self = PaperCache {
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

		// S5: as `set_fast_tier_size`.
		self.status.kick_policy_worker();

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

		// S5: a class's structural figure is chosen by the threshold.
		self.status.kick_policy_worker();

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

/// Exercises the real public `PaperCache<K, TieredBuffer>` API for
/// `lru_sized_compact_hybrid_cache` end to end. Deliberately stays on the
/// fast-tier-only path (both fast-segment capacities == max_size, tiny values)
/// so no object ever demotes: `TieredBuffer::new_slow` allocates through the
/// `Hybrid` slow-tier allocator, which needs real far-memory hardware. Full
/// tier-crossing coverage lives in this design's integration test.
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

    /// The size-split design needs three sizing scalars, so the generic
    /// hybrid constructor must refuse it rather than quietly building one
    /// from a single `CacheTierSize` (and a default second segment it was
    /// never told about). This pins that rejection.
    #[test]
    fn the_generic_hybrid_constructor_rejects_the_size_split_design() {
        assert!(matches!(
            PaperCache::<u32, TieredBuffer>::new(
                1_000_000, CacheTierSize::Bytes(1_000_000), PaperPolicy::LruSizedCompactHybrid,
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
