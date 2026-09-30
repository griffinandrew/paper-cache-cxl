/*
 * Copyright (c) Kia Shakiba
 *
 * This source code is licensed under the GNU AGPLv3 license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::{
	process,
	sync::{
		Arc,
		atomic::{Ordering, AtomicU64},
	},
};

#[cfg(feature = "hybrid_cache_common")]
use std::sync::atomic::AtomicBool;

use num_traits::AsPrimitive;
use log::error;

use kwik::{
	time,
	sys::mem,
};

use crate::{
	CacheSize,
	AtomicCacheSize,
	error::CacheError,
	policy::PaperPolicy,
	object::overhead::get_policy_overhead,
};

#[cfg(feature = "hybrid_cache_common")]
use crate::hybrid_stats::HybridStats;

/// What a clear of the object map removed, in the two figures the status keeps
/// per object: the objects, and their base bytes (`OverheadManager::
/// base_size`). Counted by the clear itself, under the lock that removes each
/// object (`ObjectStore::clear_counted`, `MergedStore::clear_counted`), for
/// `AtomicStatus::clear` to take off.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Cleared {
	pub objects: u64,
	pub base_bytes: CacheSize,
}

/// A figure the status keeps as a sum of signed changes, read as the signed
/// number it is and clamped at zero. See `AtomicStatus::clear`: a change can
/// land ahead of the one it offsets, so for that moment the sum is below zero
/// -- a wrapped `u64` -- and it reads as nothing rather than as 2^64 less a
/// little.
fn signed_or_zero(value: u64) -> u64 {
	(value as i64).max(0) as u64
}

/// Loads the active design's tier counters and gauges into [`HybridStats`].
///
/// One accessor serves every design: the per-design `<design>_hybrid_stats()`
/// methods were removed by the runtime-policy unification, and the
/// `<Design>HybridStats` names are aliases of the one struct. Of its
/// fields, the 8 size-split gauges are populated only under
/// `PaperPolicy::LruSizedCompactHybrid` and read zero elsewhere.
#[derive(Debug)]
pub struct Status {
	pid: u32,

	max_size: CacheSize,
	used_size: CacheSize,
	num_objects: u64,

	rss: u64,
	hwm: u64,

	total_hits: u64,
	total_gets: u64,
	total_sets: u64,
	total_dels: u64,

	policies: Arc<[PaperPolicy]>,
	policy: PaperPolicy,

	start_time: u64,
}

pub struct AtomicStatus {
	max_size: AtomicCacheSize,
	base_used_size: AtomicCacheSize,
	num_objects: AtomicU64,

	total_hits: AtomicU64,
	total_gets: AtomicU64,
	total_sets: AtomicU64,
	total_dels: AtomicU64,

	policies: Arc<[PaperPolicy]>,
	policy: PaperPolicy,

	start_time: AtomicU64,

	/// Runtime-configurable fast-tier byte budget, shared by every hybrid
	/// design (`PaperPolicy::LruCompactHybrid`,
	/// `PaperPolicy::LfuCompactHybrid`, `PaperPolicy::TwoQCompactHybrid`,
	/// `PaperPolicy::FifoCompactHybrid` and the rest — one field serving
	/// whichever design the cache was constructed with). Written by
	/// `PaperCache::set_fast_tier_size`, read back by both
	/// `PaperCache::fast_tier_size` and `PolicyWorker` (via the
	/// `WorkerEvent::ResizeFastTier` broadcast, not by reading this field
	/// directly — mirrors how `max_size` and `resize()`/`Resize` work).
	///
	/// For `PaperPolicy::LruSizedCompactHybrid`
	/// (`lru_sized_compact_hybrid_cache`) specifically, this field means the
	/// SMALL fast segment's capacity — that design has a second, independent
	/// fast segment ("large") with its own dedicated
	/// `hybrid_large_fast_capacity` field below, since a single shared field
	/// can't represent two independent budgets.
	#[cfg(feature = "hybrid_cache_common")]
	fast_tier_capacity: AtomicCacheSize,

	/// The hybrid tier counters/gauges, updated by `PolicyWorker` as it
	/// processes tier migrations and evictions; read via `hybrid_stats`.
	/// Lives here (rather than as a field on `PaperCache` itself) so that
	/// adding this feature doesn't require touching every other value type's
	/// constructor throughout lib.rs — `AtomicStatus::new` is the only
	/// construction site.
	#[cfg(feature = "hybrid_cache_common")]
	hybrid_promotions: AtomicU64,
	#[cfg(feature = "hybrid_cache_common")]
	hybrid_demotions: AtomicU64,
	#[cfg(feature = "hybrid_cache_common")]
	hybrid_evictions: AtomicU64,

	/// Completed CORRECTIVES (`MigrationOrigin::Reconcile`), by destination:
	/// written by the migration consumers and the inline path, never counted
	/// in `hybrid_promotions`/`hybrid_demotions`. See
	/// `HybridStats::reconcile_applied_to_fast`.
	#[cfg(feature = "hybrid_cache_common")]
	hybrid_reconcile_applied_to_fast: AtomicU64,
	#[cfg(feature = "hybrid_cache_common")]
	hybrid_reconcile_applied_to_slow: AtomicU64,

	/// The migration pipeline's per-key-bucket in-flight and landed counts
	/// (`migration_queue::InFlight`): charged and finished by the policy
	/// worker and the migration consumers, and read by `PaperCache::set` for
	/// the new-key rule's mark. Here because the status is the one structure
	/// the client, the worker and the consumers all hold. Built on first use
	/// (128 KiB), so a flat cache never pays for it; never reset (a wipe
	/// leaves entries in flight).
	#[cfg(feature = "hybrid_cache_common")]
	migration_in_flight: std::sync::OnceLock<Arc<crate::worker::InFlight>>,

	#[cfg(feature = "hybrid_cache_common")]
	hybrid_fast_bytes_used: AtomicCacheSize,
	#[cfg(feature = "hybrid_cache_common")]
	hybrid_slow_bytes_used: AtomicCacheSize,
	hybrid_fast_metadata_bytes: AtomicCacheSize,
	#[cfg(feature = "hybrid_cache_common")]
	hybrid_fast_objects: AtomicU64,
	#[cfg(feature = "hybrid_cache_common")]
	hybrid_slow_objects: AtomicU64,



	/// Mirrors the stack's `admission_latched()` -- `LfuCompactHybridStack`'s
	/// latch, or the merged store's under its `Lfu` order (see that trait
	/// method's doc). Written only by `PolicyWorker::publish_admission_latch`,
	/// right after every stack call that can move the latch, and read by
	/// `PaperCache::set()` -- running on the API-calling thread, which has no
	/// direct access to the worker-owned policy stack -- so a brand-new key is
	/// built as `TieredBuffer::new_slow` directly once the fast tier has
	/// genuinely filled. It trails the stack by the worker's event backlog;
	/// a key built fast in that window gets its corrective from the reconcile
	/// of its own `Set`.
	#[cfg(feature = "hybrid_cache_common")]
	hybrid_admission_latched: AtomicBool,





	#[cfg(feature = "hybrid_cache_common")]
	hybrid_small_fast_bytes_used: AtomicCacheSize,
	#[cfg(feature = "hybrid_cache_common")]
	hybrid_large_fast_bytes_used: AtomicCacheSize,
	#[cfg(feature = "hybrid_cache_common")]
	hybrid_small_fast_objects: AtomicU64,
	#[cfg(feature = "hybrid_cache_common")]
	hybrid_large_fast_objects: AtomicU64,
	#[cfg(feature = "hybrid_cache_common")]
	hybrid_small_slow_bytes_used: AtomicCacheSize,
	#[cfg(feature = "hybrid_cache_common")]
	hybrid_large_slow_bytes_used: AtomicCacheSize,
	#[cfg(feature = "hybrid_cache_common")]
	hybrid_small_slow_objects: AtomicU64,
	#[cfg(feature = "hybrid_cache_common")]
	hybrid_large_slow_objects: AtomicU64,

	/// The LARGE fast segment's own capacity (the SMALL segment reuses the
	/// shared `fast_tier_capacity` field above — see that field's doc for
	/// why). Written by `PaperCache::set_large_fast_tier_size`, read back by
	/// `PaperCache::large_fast_tier_size` and `PolicyWorker` (via
	/// `WorkerEvent::ResizeLargeFastTier`, not by reading this field
	/// directly — same indirection `fast_tier_capacity` uses).
	#[cfg(feature = "hybrid_cache_common")]
	hybrid_large_fast_capacity: AtomicCacheSize,

	/// The small/large size-classification threshold, in bytes. Written by
	/// `PaperCache::set_size_threshold`, read back by
	/// `PaperCache::size_threshold` and `PolicyWorker` (via
	/// `WorkerEvent::ResizeSizeThreshold`).
	#[cfg(feature = "hybrid_cache_common")]
	hybrid_size_threshold: AtomicCacheSize,

	/// The per-object DRAM reservation `omega`: what
	/// `get_hybrid_dram_shared_overhead` returns for this cache's policy, the
	/// figure its stack reserves per live object (0 under
	/// `PAPER_DISABLE_SHARED_OVERHEAD=1`). Written once, by
	/// `register_tiered_cache`; read by `effective_fast_capacity`.
	#[cfg(feature = "hybrid_cache_common")]
	hybrid_shared_overhead: AtomicCacheSize,

	/// Hits by the tier that served them: the tier of the value a hit copied,
	/// read off the snapshot's tag at the lookup. Incremented beside
	/// `total_hits` in the hybrid `get`/`get_into` and unsharded like it;
	/// `peek` counts neither, as it counts no hit.
	#[cfg(feature = "hybrid_cache_common")]
	hybrid_fast_hits: AtomicU64,
	#[cfg(feature = "hybrid_cache_common")]
	hybrid_slow_hits: AtomicU64,

	/// The policy worker's over-budget integral in whole byte-seconds (see
	/// `phys::over_budget_increment`), published once per pass.
	#[cfg(feature = "hybrid_cache_common")]
	hybrid_over_budget_byte_seconds: AtomicU64,

	/// M, the bytes the cache's own DRAM metadata structures hold (S5a):
	/// published by the policy worker (`PolicyWorker::publish_metadata`) at
	/// construction, every pass, after a wipe and after any event that grew
	/// its stack, and read with ONE load (`dram_metadata_bytes`). Its parts and
	/// the slow-node structures beside it, for the reports; each is a load of
	/// its own, so a reader of all four can see two publications mixed, and
	/// only the total is a single reading.
	#[cfg(feature = "hybrid_cache_common")]
	dram_metadata: AtomicU64,
	#[cfg(feature = "hybrid_cache_common")]
	dram_metadata_map: AtomicU64,
	#[cfg(feature = "hybrid_cache_common")]
	dram_metadata_stack: AtomicU64,
	#[cfg(feature = "hybrid_cache_common")]
	dram_metadata_headers: AtomicU64,
	#[cfg(feature = "hybrid_cache_common")]
	slow_metadata: AtomicU64,

	/// This cache's place in `phys::live_tiered_caches`. Installed by
	/// `register_tiered_cache`, released when the status is freed -- after
	/// the cache has joined the workers that share it.
	#[cfg(feature = "hybrid_cache_common")]
	tiered_registration: std::sync::OnceLock<crate::phys::LiveRegistration>,

	/// This cache's place in `phys::live_flat_fast_caches`: installed by
	/// `register_flat_fast_cache` when a FLAT cache's values are fast,
	/// released when the status is freed, like `tiered_registration`.
	#[cfg(feature = "hybrid_cache_common")]
	flat_fast_registration: std::sync::OnceLock<crate::phys::LiveRegistration>,

	/// The policy worker's thread, for `kick_policy_worker`. `None` until
	/// `PolicyWorker::run` publishes it on entry.
	///
	/// On the status because the status is the one per-cache object every
	/// API-side path already holds, and it is built before the worker thread
	/// exists. One worker thread serves a cache for its whole life -- a wipe
	/// clears the stack and respawns nothing -- so this is written once in
	/// practice. It is still a replaceable slot rather than a
	/// `OnceLock`, so a status ever run by a second worker kicks the live one
	/// and not one that has exited.
	///
	/// A lock, not an atomic: `Thread` is a handle, the only writer runs once
	/// per worker, and the kick is on no per-request path.
	policy_worker: parking_lot::Mutex<Option<std::thread::Thread>>,

	/// Test-only: passes the policy worker has completed, so a test can see a
	/// kick land without depending on what the pass did.
	#[cfg(test)]
	policy_worker_passes: AtomicU64,

	/// S5: the admission state -- the configuration, the figures the policy
	/// worker publishes for a set's decisions (eff, the key ceiling), the
	/// metadata lane, the worker's idle and liveness bits, and the counters.
	/// See `crate::gate`.
	#[cfg(feature = "hybrid_cache_common")]
	gate: crate::gate::Gate,
}

/// This struct holds the basic statistical information about `PaperCache`.
impl Status {
	/// Returns the cache's PID.
	#[must_use]
	pub fn pid(&self) -> u32 {
		self.pid
	}

	/// Returns the cache's maximum size.
	#[must_use]
	pub fn max_size(&self) -> CacheSize {
		self.max_size
	}

	/// Returns the cache's used size.
	#[must_use]
	pub fn used_size(&self) -> CacheSize {
		self.used_size
	}

	/// Returns the number of objects in the cache.
	#[must_use]
	pub fn num_objects(&self) -> u64 {
		self.num_objects
	}

	/// Returns the cache's resident set size.
	#[must_use]
	pub fn rss(&self) -> u64 {
		self.rss
	}

	/// Returns the cache's resident set size high water mark.
	#[must_use]
	pub fn hwm(&self) -> u64 {
		self.hwm
	}

	/// Returns the cache's total number of gets.
	#[must_use]
	pub fn total_gets(&self) -> u64 {
		self.total_gets
	}

	/// Returns the cache's total number of sets.
	#[must_use]
	pub fn total_sets(&self) -> u64 {
		self.total_sets
	}

	/// Returns the cache's total number of dels.
	#[must_use]
	pub fn total_dels(&self) -> u64 {
		self.total_dels
	}

	/// Returns the cache's current miss ratio.
	#[must_use]
	pub fn miss_ratio(&self) -> f64 {
		if self.total_gets == 0 {
			return 1.0;
		}

		1.0 - self.total_hits as f64 / self.total_gets as f64
	}

	/// Returns the cache's configured eviction policies.
	#[must_use]
	pub fn policies(&self) -> &[PaperPolicy] {
		&self.policies
	}

	/// Returns the cache's current eviction policy.
	#[must_use]
	pub fn policy(&self) -> PaperPolicy {
		self.policy
	}

	/// Returns the cache's current uptime.
	#[must_use]
	pub fn uptime(&self) -> u64 {
		time::timestamp() - self.start_time
	}
}

/// This struct holds the basic statistical information about `PaperCache`
/// and allows for atomic updates of its fields.
impl AtomicStatus {
	pub fn new(
		max_size: CacheSize,
		policies: &[PaperPolicy],
		policy: PaperPolicy,
	) -> Result<Self, CacheError> {
		let policies: Arc<[PaperPolicy]> = policies.into();

		// The running policy, fixed for the cache's life: one of the configured
		// ones, as it always had to be.
		if !policies.contains(&policy) {
			error!("The policy is not among the configured ones");
			return Err(CacheError::Internal);
		}

		let status = AtomicStatus {
			max_size: AtomicCacheSize::new(max_size),
			base_used_size: AtomicCacheSize::default(),
			num_objects: AtomicU64::default(),

			total_hits: AtomicU64::default(),
			total_gets: AtomicU64::default(),
			total_sets: AtomicU64::default(),
			total_dels: AtomicU64::default(),

			policies,
			policy,

			start_time: AtomicU64::new(time::timestamp()),

			#[cfg(feature = "hybrid_cache_common")]
			fast_tier_capacity: AtomicCacheSize::default(),
			#[cfg(feature = "hybrid_cache_common")]
			hybrid_promotions: AtomicU64::default(),
			#[cfg(feature = "hybrid_cache_common")]
			hybrid_demotions: AtomicU64::default(),
			#[cfg(feature = "hybrid_cache_common")]
			hybrid_evictions: AtomicU64::default(),
			#[cfg(feature = "hybrid_cache_common")]
			hybrid_reconcile_applied_to_fast: AtomicU64::default(),
			#[cfg(feature = "hybrid_cache_common")]
			hybrid_reconcile_applied_to_slow: AtomicU64::default(),
			#[cfg(feature = "hybrid_cache_common")]
			migration_in_flight: std::sync::OnceLock::new(),
			#[cfg(feature = "hybrid_cache_common")]
			hybrid_fast_bytes_used: AtomicCacheSize::default(),
			#[cfg(feature = "hybrid_cache_common")]
			hybrid_slow_bytes_used: AtomicCacheSize::default(),
			hybrid_fast_metadata_bytes: AtomicCacheSize::default(),
			#[cfg(feature = "hybrid_cache_common")]
			hybrid_fast_objects: AtomicU64::default(),
			#[cfg(feature = "hybrid_cache_common")]
			hybrid_slow_objects: AtomicU64::default(),
			#[cfg(feature = "hybrid_cache_common")]
			hybrid_admission_latched: AtomicBool::default(),
			#[cfg(feature = "hybrid_cache_common")]
			hybrid_small_fast_bytes_used: AtomicCacheSize::default(),
			#[cfg(feature = "hybrid_cache_common")]
			hybrid_large_fast_bytes_used: AtomicCacheSize::default(),
			#[cfg(feature = "hybrid_cache_common")]
			hybrid_small_slow_bytes_used: AtomicCacheSize::default(),
			#[cfg(feature = "hybrid_cache_common")]
			hybrid_large_slow_bytes_used: AtomicCacheSize::default(),
			#[cfg(feature = "hybrid_cache_common")]
			hybrid_small_fast_objects: AtomicU64::default(),
			#[cfg(feature = "hybrid_cache_common")]
			hybrid_large_fast_objects: AtomicU64::default(),
			#[cfg(feature = "hybrid_cache_common")]
			hybrid_small_slow_objects: AtomicU64::default(),
			#[cfg(feature = "hybrid_cache_common")]
			hybrid_large_slow_objects: AtomicU64::default(),
			#[cfg(feature = "hybrid_cache_common")]
			hybrid_large_fast_capacity: AtomicCacheSize::default(),
			#[cfg(feature = "hybrid_cache_common")]
			hybrid_size_threshold: AtomicCacheSize::default(),
			#[cfg(feature = "hybrid_cache_common")]
			hybrid_shared_overhead: AtomicCacheSize::default(),
			#[cfg(feature = "hybrid_cache_common")]
			hybrid_fast_hits: AtomicU64::default(),
			#[cfg(feature = "hybrid_cache_common")]
			hybrid_slow_hits: AtomicU64::default(),
			#[cfg(feature = "hybrid_cache_common")]
			hybrid_over_budget_byte_seconds: AtomicU64::default(),
			#[cfg(feature = "hybrid_cache_common")]
			dram_metadata: AtomicU64::default(),
			#[cfg(feature = "hybrid_cache_common")]
			dram_metadata_map: AtomicU64::default(),
			#[cfg(feature = "hybrid_cache_common")]
			dram_metadata_stack: AtomicU64::default(),
			#[cfg(feature = "hybrid_cache_common")]
			dram_metadata_headers: AtomicU64::default(),
			#[cfg(feature = "hybrid_cache_common")]
			slow_metadata: AtomicU64::default(),
			#[cfg(feature = "hybrid_cache_common")]
			tiered_registration: std::sync::OnceLock::new(),
			#[cfg(feature = "hybrid_cache_common")]
			flat_fast_registration: std::sync::OnceLock::new(),

			policy_worker: parking_lot::Mutex::new(None),

			#[cfg(test)]
			policy_worker_passes: AtomicU64::default(),

			#[cfg(feature = "hybrid_cache_common")]
			gate: crate::gate::Gate::default(),
		};

		Ok(status)
	}

	/// Records the thread `kick_policy_worker` wakes. Called by
	/// `PolicyWorker::run` on entry; a later call replaces the handle.
	pub(crate) fn set_policy_worker_thread(&self, thread: std::thread::Thread) {
		*self.policy_worker.lock() = Some(thread);
	}

	/// Wakes the policy worker if it is parked between passes, so it runs its
	/// next pass now rather than when its poll runs out -- up to 1 s
	/// (`LONG_POLLING_DURATION`) once it has stopped seeing sets.
	///
	/// An `unpark`, so it can neither be lost nor do harm: a worker that is
	/// mid-pass keeps the token and its next park returns at once, costing
	/// one extra pass; a worker not yet started has no handle, so the kick is
	/// a no-op; one that has exited ignores it.
	///
	/// Called where a client WAITS for the worker: `PaperCache::wipe`, which
	/// the worker performs and answers, and `placement_audit` -- without it
	/// an idle cache's wipe took up to `LONG_POLLING_DURATION`. The fast-tier
	/// gate will kick it as admissions approach the budget. There is no kick
	/// on the set path yet, and `polling_delay`'s idle fix is not a substitute
	/// for one. That fix makes the pass that FIRST sees a burst choose the
	/// short poll, so the rest of the burst is taken within milliseconds; but a
	/// worker already parked on the long poll when the burst begins sleeps out
	/// up to `LONG_POLLING_DURATION` before that pass runs -- as does a new
	/// cache's worker, whose first pass normally sees no set and parks long.
	/// The set-path kick closes that window (S5, `PaperCache::commit`'s
	/// `kick_idle_worker` with the gate's `worker_idle` bit): a merged store's
	/// values published while the worker is parked were unlinked -- uncharged
	/// and unevictable -- until it woke, as a DashMap stack's are untracked,
	/// and the first set after an idle spell now wakes it.
	pub(crate) fn kick_policy_worker(&self) {
		if let Some(worker) = self.policy_worker.lock().as_ref() {
			worker.unpark();
		}
	}

	/// Test-only: see `policy_worker_passes`.
	#[cfg(test)]
	pub(crate) fn record_policy_worker_pass(&self) {
		self.policy_worker_passes.fetch_add(1, Ordering::Release);
	}

	/// Test-only: passes the policy worker has completed so far.
	#[cfg(test)]
	pub(crate) fn policy_worker_passes(&self) -> u64 {
		self.policy_worker_passes.load(Ordering::Acquire)
	}

	#[must_use]
	pub fn max_size(&self) -> CacheSize {
		self.max_size.load(Ordering::Relaxed)
	}

	/// The base bytes of every object plus the policy's per-object overhead,
	/// each figure clamped at zero (`signed_or_zero`: an insert's own update
	/// can trail a clear that already took its object off).
	#[must_use]
	pub fn used_size(&self, policy: &PaperPolicy) -> CacheSize {
		let base_used_size = signed_or_zero(self.base_used_size.load(Ordering::Acquire));
		let num_objects = signed_or_zero(self.num_objects.load(Ordering::Acquire));
		let policy_overhead = get_policy_overhead(policy);

		base_used_size + num_objects * policy_overhead as CacheSize
	}

	#[must_use]
	pub fn policies(&self) -> &[PaperPolicy] {
		&self.policies
	}

	#[must_use]
	pub fn policy(&self) -> PaperPolicy {
		self.policy
	}

	pub fn incr_hits(&self) {
		self.total_gets.fetch_add(1, Ordering::Relaxed);
		self.total_hits.fetch_add(1, Ordering::Relaxed);
	}

	pub fn incr_misses(&self) {
		self.total_gets.fetch_add(1, Ordering::Relaxed);
	}

	pub fn incr_sets(&self) {
		self.total_sets.fetch_add(1, Ordering::Relaxed);
	}

	pub fn incr_dels(&self) {
		self.total_dels.fetch_add(1, Ordering::Relaxed);
	}

	pub fn set_max_size(&self, max_size: u64) {
		self.max_size.store(max_size, Ordering::Relaxed);
	}

	pub fn update_base_used_size(&self, delta: impl AsPrimitive<i64>) {
		let delta = delta.as_();

		if delta > 0 {
			self.base_used_size.fetch_add(delta.unsigned_abs(), Ordering::AcqRel);
		} else if delta < 0 {
			self.base_used_size.fetch_sub(delta.unsigned_abs(), Ordering::AcqRel);
		}
	}

	/// INSTRUMENTATION: live object count on the atomic status, for the
	/// map-vs-stack divergence sampler in the policy worker. Clamped at zero,
	/// as `used_size` reads it.
	pub fn live_num_objects(&self) -> u64 {
		signed_or_zero(self.num_objects.load(Ordering::Acquire))
	}

	/// Counts one new object and returns the count BEFORE it, clamped at zero
	/// as `live_num_objects` reads it -- what a set's metadata-cap flag is
	/// decided from (`gate::Gate::note_count`), at no extra load.
	pub fn incr_num_objects(&self) -> u64 {
		signed_or_zero(self.num_objects.fetch_add(1, Ordering::AcqRel))
	}

	pub fn add_num_objects(&self, count: u64) {
		if count > 0 {
			self.num_objects.fetch_add(count, Ordering::AcqRel);
		}
	}

	pub fn decr_num_objects(&self) {
		self.num_objects.fetch_sub(1, Ordering::AcqRel);
	}

	#[must_use]
	pub fn exceeds_max_size(&self, size: impl AsPrimitive<u64>) -> bool {
		size.as_() > self.max_size.load(Ordering::Relaxed)
	}

	/// Current fast-tier byte budget (every hybrid design's whole fast tier,
	/// or `PaperPolicy::LruSizedCompactHybrid`'s SMALL segment — see the
	/// field's doc on the struct).
	#[cfg(feature = "hybrid_cache_common")]
	#[must_use]
	pub fn fast_tier_capacity(&self) -> CacheSize {
		self.fast_tier_capacity.load(Ordering::Relaxed)
	}

	/// Sets the fast-tier byte budget. Callers are responsible for also
	/// broadcasting `WorkerEvent::ResizeFastTier` so the active stack's own
	/// internal capacity is updated (mirrors `set_max_size` + `Resize`).
	#[cfg(feature = "hybrid_cache_common")]
	pub fn set_fast_tier_capacity(&self, size: CacheSize) {
		self.fast_tier_capacity.store(size, Ordering::Relaxed);
	}

	/// Marks this status as a TIERED cache's: records `omega`, the per-object
	/// reservation its stack is built with, and counts the cache in
	/// `phys::live_tiered_caches` until the status is freed. Called once, by
	/// the two hybrid constructors; a second call changes nothing.
	#[cfg(feature = "hybrid_cache_common")]
	pub fn register_tiered_cache(&self, shared_overhead: CacheSize) {
		self.tiered_registration.get_or_init(|| {
			self.hybrid_shared_overhead.store(shared_overhead, Ordering::Relaxed);
			crate::phys::LiveRegistration::tiered_cache()
		});
	}

	/// Marks this status as a FLAT cache's whose values are fast (`V::TIER ==
	/// Tier::Fast`): counts it in `phys::live_flat_fast_caches` until the
	/// status is freed, since its values are charged to PHYS_FAST like a
	/// tiered cache's. Called once, by the two flat constructors; a second
	/// call changes nothing.
	#[cfg(feature = "hybrid_cache_common")]
	pub fn register_flat_fast_cache(&self) {
		self.flat_fast_registration.get_or_init(crate::phys::LiveRegistration::flat_fast_cache);
	}

	/// `omega`, the per-object DRAM reservation this cache's stack makes.
	/// Zero on a status no hybrid constructor registered.
	#[cfg(feature = "hybrid_cache_common")]
	#[must_use]
	pub fn hybrid_shared_overhead(&self) -> CacheSize {
		self.hybrid_shared_overhead.load(Ordering::Relaxed)
	}

	/// `F`, the fast tier's whole DRAM budget: `fast_tier_capacity`, plus the
	/// LARGE segment's for the size-split design, whose `fast_tier_capacity`
	/// is its small segment only. Every other design leaves the large
	/// capacity at 0, so there this is exactly `fast_tier_capacity`.
	#[cfg(feature = "hybrid_cache_common")]
	#[must_use]
	pub fn whole_fast_tier_capacity(&self) -> CacheSize {
		self.fast_tier_capacity().saturating_add(self.hybrid_large_fast_capacity())
	}

	/// eff, THE figure (S5): the fast tier's budget for VALUE bytes, `F -
	/// M_model`, saturating, as the policy worker last published it
	/// (`PolicyWorker::publish_gate`) -- what the stacks' settles, the
	/// structural check and the metadata cap all read. `M_model` is the
	/// cache's measured DRAM metadata (`dram_metadata_bytes`) or, under the
	/// per-object model, the stack's own reservation (`L * omega`, plus a
	/// ghost's DRAM), per `gate::MetadataModel`. One load.
	#[cfg(feature = "hybrid_cache_common")]
	#[must_use]
	pub fn effective_fast_capacity(&self) -> CacheSize {
		self.gate.eff()
	}

	/// The per-object model's figure from the status alone, `F - L * omega`,
	/// saturating: what `effective_fast_capacity` was until S5 (S2's), with
	/// `L` the object map's count. Kept as the model's sanity reading -- the
	/// policy worker compares the measured M with `L * omega` -- and for its
	/// unit test; nothing decides on it.
	#[cfg(feature = "hybrid_cache_common")]
	#[must_use]
	pub fn per_object_effective_fast_capacity(&self) -> CacheSize {
		let reserved = self.live_num_objects().saturating_mul(self.hybrid_shared_overhead());

		self.whole_fast_tier_capacity().saturating_sub(reserved)
	}

	/// The admission state (S5, `crate::gate`).
	#[cfg(feature = "hybrid_cache_common")]
	pub(crate) fn gate(&self) -> &crate::gate::Gate {
		&self.gate
	}

	/// Test builds: pins a status no constructor configured -- a hand-driven
	/// harness's -- to the per-object metadata model its tests were written
	/// against (S5). A cache built through a constructor gets its
	/// `GateConfig` from it instead.
	#[cfg(all(test, feature = "hybrid_cache_common"))]
	pub(crate) fn pin_per_object(&self) {
		let mut config = self.gate.config();
		config.metadata_model = crate::gate::MetadataModel::PerObject;
		self.gate.set_config(config);
	}

	/// M, the bytes this cache's own DRAM metadata structures hold, in
	/// jemalloc's usable-size unit: the object map's (its tables, arrays and
	/// `Arc`), the policy stack's (its slab chunks and their table, index,
	/// free list, bucket maps, ghost, and its box) and one value header per
	/// live object (S5a; `crate::meta`). One load of what the policy worker
	/// last published; 0 on a status no tiered cache's worker publishes to.
	///
	/// Structures on the slow node are not in it (`dram_metadata().slow`).
	/// It lags the map by up to one worker pass: a client's insert grows the
	/// map before the worker handles its `Set`.
	#[cfg(feature = "hybrid_cache_common")]
	#[must_use]
	pub fn dram_metadata_bytes(&self) -> u64 {
		self.dram_metadata.load(Ordering::Relaxed)
	}

	/// M's parts and the slow-node structures beside them, as last published.
	/// Four loads (see the field's doc): for the reports, not for a budget.
	#[cfg(feature = "hybrid_cache_common")]
	#[must_use]
	pub fn dram_metadata(&self) -> crate::meta::DramMetadata {
		crate::meta::DramMetadata {
			map: self.dram_metadata_map.load(Ordering::Relaxed),
			stack: self.dram_metadata_stack.load(Ordering::Relaxed),
			headers: self.dram_metadata_headers.load(Ordering::Relaxed),
			slow: self.slow_metadata.load(Ordering::Relaxed),
		}
	}

	/// Publishes M (the policy worker's). The parts first, then the total.
	#[cfg(feature = "hybrid_cache_common")]
	pub(crate) fn set_dram_metadata(&self, metadata: crate::meta::DramMetadata) {
		self.dram_metadata_map.store(metadata.map, Ordering::Relaxed);
		self.dram_metadata_stack.store(metadata.stack, Ordering::Relaxed);
		self.dram_metadata_headers.store(metadata.headers, Ordering::Relaxed);
		self.slow_metadata.store(metadata.slow, Ordering::Relaxed);
		self.dram_metadata.store(metadata.total(), Ordering::Relaxed);
	}

	/// `F - M`, saturating at zero: the whole fast-tier budget
	/// (`whole_fast_tier_capacity`) less the MEASURED metadata, where
	/// `effective_fast_capacity` takes off the modelled `L * omega`. Beside
	/// it, not instead of it: S5a changes no consumer, S5 moves them onto
	/// this. Reporting only.
	#[cfg(feature = "hybrid_cache_common")]
	#[must_use]
	pub fn effective_fast_capacity_measured(&self) -> CacheSize {
		self.whole_fast_tier_capacity().saturating_sub(self.dram_metadata_bytes())
	}

	/// Counts a hit served from `tier`, the tier of the value the hit copies.
	/// Called beside `incr_hits`, never instead of it.
	#[cfg(feature = "hybrid_cache_common")]
	pub fn incr_served_hit(&self, tier: crate::Tier) {
		let counter = match tier {
			crate::Tier::Fast => &self.hybrid_fast_hits,
			crate::Tier::Slow => &self.hybrid_slow_hits,
		};

		counter.fetch_add(1, Ordering::Relaxed);
	}

	/// Publishes the policy worker's over-budget integral.
	#[cfg(feature = "hybrid_cache_common")]
	pub(crate) fn set_hybrid_over_budget_byte_seconds(&self, byte_seconds: u64) {
		self.hybrid_over_budget_byte_seconds.store(byte_seconds, Ordering::Relaxed);
	}

	#[cfg(feature = "hybrid_cache_common")]
	pub fn record_hybrid_promotion(&self) {
		self.hybrid_promotions.fetch_add(1, Ordering::Relaxed);
	}

	#[cfg(feature = "hybrid_cache_common")]
	pub fn record_hybrid_demotion(&self) {
		self.hybrid_demotions.fetch_add(1, Ordering::Relaxed);
	}

	#[cfg(feature = "hybrid_cache_common")]
	pub fn record_hybrid_eviction(&self) {
		self.hybrid_evictions.fetch_add(1, Ordering::Relaxed);
	}

	/// Overwrites the live tier gauges (bytes/objects currently in each
	/// tier). Called by `PolicyWorker` each time it drains tier migrations.
	#[cfg(feature = "hybrid_cache_common")]
	pub fn set_hybrid_gauges(
		&self,
		fast_bytes_used: CacheSize,
		slow_bytes_used: CacheSize,
		fast_objects: u64,
		slow_objects: u64,
		fast_metadata_bytes: CacheSize,
	) {
		self.hybrid_fast_metadata_bytes.store(fast_metadata_bytes, Ordering::Relaxed);
		self.hybrid_fast_bytes_used.store(fast_bytes_used, Ordering::Relaxed);
		self.hybrid_slow_bytes_used.store(slow_bytes_used, Ordering::Relaxed);
		self.hybrid_fast_objects.store(fast_objects, Ordering::Relaxed);
		self.hybrid_slow_objects.store(slow_objects, Ordering::Relaxed);
	}

	#[cfg(feature = "hybrid_cache_common")]
	/// The size-split (`lru_sized`) design's four-segment gauges. Writes only
	/// the granular small/large fields; the two-tier totals stay owned by
	/// [`Self::set_hybrid_gauges`], which every hybrid design calls.
	pub fn set_hybrid_sized_gauges(
		&self,
		small_fast_bytes_used: CacheSize,
		large_fast_bytes_used: CacheSize,
		small_slow_bytes_used: CacheSize,
		large_slow_bytes_used: CacheSize,
		small_fast_objects: u64,
		large_fast_objects: u64,
		small_slow_objects: u64,
		large_slow_objects: u64,
	) {
		self.hybrid_small_fast_bytes_used.store(small_fast_bytes_used, Ordering::Relaxed);
		self.hybrid_large_fast_bytes_used.store(large_fast_bytes_used, Ordering::Relaxed);
		self.hybrid_small_slow_bytes_used.store(small_slow_bytes_used, Ordering::Relaxed);
		self.hybrid_large_slow_bytes_used.store(large_slow_bytes_used, Ordering::Relaxed);
		self.hybrid_small_fast_objects.store(small_fast_objects, Ordering::Relaxed);
		self.hybrid_large_fast_objects.store(large_fast_objects, Ordering::Relaxed);
		self.hybrid_small_slow_objects.store(small_slow_objects, Ordering::Relaxed);
		self.hybrid_large_slow_objects.store(large_slow_objects, Ordering::Relaxed);
	}

	#[cfg(feature = "hybrid_cache_common")]
	/// Point-in-time snapshot of the hybrid tier counters and gauges,
	/// whichever hybrid design is running.
	#[must_use]
	pub fn hybrid_stats(&self) -> HybridStats {
		let (
			reconcile_set_to_fast,
			reconcile_set_to_slow,
			reconcile_get_to_fast,
			reconcile_set_new_key,
			reconcile_get_heal_skipped,
		) = crate::worker::reconciled();

		let gate = self.gate.stats();

		HybridStats {
			promotions: self.hybrid_promotions.load(Ordering::Relaxed),
			demotions: self.hybrid_demotions.load(Ordering::Relaxed),
			evictions: self.hybrid_evictions.load(Ordering::Relaxed),
			fast_bytes_used: self.hybrid_fast_bytes_used.load(Ordering::Relaxed),
			slow_bytes_used: self.hybrid_slow_bytes_used.load(Ordering::Relaxed),
			fast_metadata_bytes: self.hybrid_fast_metadata_bytes.load(Ordering::Relaxed),
			fast_objects: self.hybrid_fast_objects.load(Ordering::Relaxed),
			slow_objects: self.hybrid_slow_objects.load(Ordering::Relaxed),
			small_fast_bytes_used: self.hybrid_small_fast_bytes_used.load(Ordering::Relaxed),
			large_fast_bytes_used: self.hybrid_large_fast_bytes_used.load(Ordering::Relaxed),
			small_slow_bytes_used: self.hybrid_small_slow_bytes_used.load(Ordering::Relaxed),
			large_slow_bytes_used: self.hybrid_large_slow_bytes_used.load(Ordering::Relaxed),
			small_fast_objects: self.hybrid_small_fast_objects.load(Ordering::Relaxed),
			large_fast_objects: self.hybrid_large_fast_objects.load(Ordering::Relaxed),
			small_slow_objects: self.hybrid_small_slow_objects.load(Ordering::Relaxed),
			large_slow_objects: self.hybrid_large_slow_objects.load(Ordering::Relaxed),
			phys_fast_bytes: crate::phys::fast_bytes(),
			phys_fast_bytes_max: crate::phys::fast_bytes_max(),
			effective_fast_capacity: gate.eff,
			over_budget_byte_seconds: self.hybrid_over_budget_byte_seconds.load(Ordering::Relaxed),
			fast_hits: self.hybrid_fast_hits.load(Ordering::Relaxed),
			slow_hits: self.hybrid_slow_hits.load(Ordering::Relaxed),
			live_tiered_caches: crate::phys::live_tiered_caches(),
			live_flat_fast_caches: crate::phys::live_flat_fast_caches(),
			dram_metadata_bytes: self.dram_metadata_bytes(),
			slow_metadata_bytes: self.slow_metadata.load(Ordering::Relaxed),
			effective_fast_capacity_measured: self.effective_fast_capacity_measured(),
			reconcile_set_to_fast,
			reconcile_set_to_slow,
			reconcile_get_to_fast,
			reconcile_set_new_key,
			reconcile_get_heal_skipped,
			reconcile_applied_to_fast: self.hybrid_reconcile_applied_to_fast.load(Ordering::Relaxed),
			reconcile_applied_to_slow: self.hybrid_reconcile_applied_to_slow.load(Ordering::Relaxed),
			metadata_model: gate.model,
			dram_metadata_bytes_model: gate.m_model,
			metadata_key_ceiling: gate.k_max,
			metadata_bound: gate.eff == 0,
			metadata_overflows: gate.metadata_overflows,
			make_room_requests: gate.make_room_requests,
			make_room_evictions: gate.make_room_evictions,
			make_room_failures: gate.make_room_failures,
			structural_slow_sets: gate.structural_slow_sets,
			structural_placements: gate.structural_placements,
			idle_kicks: gate.idle_kicks,
			metadata_model_divergence: gate.metadata_model_divergence,
			gate_state: gate.state,
			gate_disabled_sets: gate.gate_disabled_sets,
			gate_slow_paths: gate.gate_slow_paths,
			gate_waits: gate.gate_waits,
			gate_wait_ns_total: gate.gate_wait_ns_total,
			gate_wait_ns_max: gate.gate_wait_ns_max,
			gate_wait_hist: gate.gate_wait_hist,
			gate_stalls: gate.gate_stalls,
			gate_stall_errors: gate.gate_stall_errors,
			divert_sets: gate.divert_sets,
			divert_bytes: gate.divert_bytes,
			admit_over_sets: gate.admit_over_sets,
			admit_over_bytes: gate.admit_over_bytes,
			oversize_admits: gate.oversize_admits,
			near_kicks: gate.near_kicks,
			max_waiters: gate.max_waiters,
			waiters: gate.waiters,
			reserved_bytes: gate.reserved,
			band_s: gate.bands.s,
			band_n: gate.bands.n,
			band_b: gate.bands.b,
		}
	}

	/// Records completed correctives (`MigrationOrigin::Reconcile`), by
	/// destination, for this cache and in the process-global MIGSTATS totals
	/// -- batched like `record_hybrid_promotions`, and never counted as
	/// promotions or demotions.
	#[cfg(feature = "hybrid_cache_common")]
	pub(crate) fn record_reconcile_applied(&self, to_fast: u64, to_slow: u64) {
		if to_fast != 0 {
			self.hybrid_reconcile_applied_to_fast.fetch_add(to_fast, Ordering::Relaxed);
		}

		if to_slow != 0 {
			self.hybrid_reconcile_applied_to_slow.fetch_add(to_slow, Ordering::Relaxed);
		}

		crate::worker::reconcile_applied(to_fast, to_slow);
	}

	/// The migration pipeline's per-key-bucket counts
	/// (`migration_queue::InFlight`), built on first use.
	#[cfg(feature = "hybrid_cache_common")]
	pub(crate) fn migration_in_flight(&self) -> &Arc<crate::worker::InFlight> {
		self.migration_in_flight.get_or_init(|| Arc::new(crate::worker::InFlight::new()))
	}









	/// Records `count` demotions at once — used when draining
	/// `LfuCompactHybridStack::drain_demotions`, which reports genuine
	/// `settle_fast_tier` demotions in a single batch per
	/// `apply_tier_migrations` pass, distinct from admission-to-slow
	/// corrections (see that method's doc comment for why the two aren't
	/// the same thing).
	#[cfg(feature = "hybrid_cache_common")]
	pub fn record_hybrid_demotions(&self, count: u64) {
		if count > 0 {
			self.hybrid_demotions.fetch_add(count, Ordering::Relaxed);
		}
	}

	/// Records `count` promotions at once — the batched form of
	/// [`Self::record_hybrid_promotion`]. Used by the migration consumers,
	/// which tally completed promotions in a plain local `u64` and flush the
	/// total with a single atomic when their channel drains, rather than
	/// paying one atomic per migration.
	#[cfg(feature = "hybrid_cache_common")]
	pub fn record_hybrid_promotions(&self, count: u64) {
		if count > 0 {
			self.hybrid_promotions.fetch_add(count, Ordering::Relaxed);
		}
	}



	/// Publishes the stack's `admission_latched()`. Only
	/// `PolicyWorker::publish_admission_latch` calls it -- see the field's doc
	/// on the struct for why this crosses threads via an atomic rather than a
	/// direct call into the stack.
	#[cfg(feature = "hybrid_cache_common")]
	pub fn set_hybrid_admission_latched(&self, latched: bool) {
		self.hybrid_admission_latched.store(latched, Ordering::Relaxed);
	}

	/// The stack's `admission_latched()` as last published: stale by at most
	/// the events the worker has not handled yet -- see the field's doc.
	#[cfg(feature = "hybrid_cache_common")]
	#[must_use]
	pub fn hybrid_admission_latched(&self) -> bool {
		self.hybrid_admission_latched.load(Ordering::Relaxed)
	}












	// `record_hybrid_promotion` intentionally does not exist:
	// `FifoCompactHybridStack` never emits a `Tier::Fast` migration (no
	// promotion policy at all — see that stack's module doc), so
	// `hybrid_promotions` is only ever read (always 0), never written.




























































	/// Current LARGE fast segment's byte budget (the SMALL segment reuses
	/// `fast_tier_capacity` above).
	#[cfg(feature = "hybrid_cache_common")]
	#[must_use]
	pub fn hybrid_large_fast_capacity(&self) -> CacheSize {
		self.hybrid_large_fast_capacity.load(Ordering::Relaxed)
	}

	/// Sets the LARGE fast segment's byte budget. Callers are responsible
	/// for also broadcasting `WorkerEvent::ResizeLargeFastTier`.
	#[cfg(feature = "hybrid_cache_common")]
	pub fn set_hybrid_large_fast_capacity(&self, size: CacheSize) {
		self.hybrid_large_fast_capacity.store(size, Ordering::Relaxed);
	}

	/// Current small/large size-classification threshold, in bytes.
	#[cfg(feature = "hybrid_cache_common")]
	#[must_use]
	pub fn hybrid_size_threshold(&self) -> CacheSize {
		self.hybrid_size_threshold.load(Ordering::Relaxed)
	}

	/// Sets the small/large size-classification threshold. Callers are
	/// responsible for also broadcasting `WorkerEvent::ResizeSizeThreshold`.
	#[cfg(feature = "hybrid_cache_common")]
	pub fn set_hybrid_size_threshold(&self, size: CacheSize) {
		self.hybrid_size_threshold.store(size, Ordering::Relaxed);
	}
























	/// Empties the status after a wipe: takes what the object map's clear
	/// removed (`cleared`) off the object count and the base size, and
	/// resets every counter.
	///
	/// SUBTRACTED, not stored as 0. A client's insert racing the clear lands
	/// either in a shard the clear has not reached yet -- removed with it,
	/// and in `cleared` -- or in one it has already emptied -- live, and not
	/// in `cleared` -- and the insert's own `incr_num_objects` and
	/// `update_base_used_size` land whenever they land; so once the clients
	/// are quiet the two figures are exactly the map's, in every order. Stored
	/// as 0 they were not: an insert into an already-cleared shard whose
	/// update came before the store was live and uncounted -- its later
	/// removal's `fetch_sub` wrapped `base_used_size`, and `used_size` was
	/// garbage for the rest of the cache's life -- and one the clear removed
	/// whose update came after the store stayed counted with nothing behind
	/// it. Until an insert the clear removed has made its own update, the two
	/// figures are below zero (wrapped), which `used_size` and
	/// `live_num_objects` read as zero -- as a `set` and a `del` of one key
	/// racing could already leave them.
	pub fn clear(&self, cleared: Cleared) {
		self.base_used_size.fetch_sub(cleared.base_bytes, Ordering::AcqRel);
		self.num_objects.fetch_sub(cleared.objects, Ordering::AcqRel);

		self.total_hits.store(0, Ordering::Relaxed);
		self.total_gets.store(0, Ordering::Relaxed);
		self.total_sets.store(0, Ordering::Relaxed);
		self.total_dels.store(0, Ordering::Relaxed);

		// The served-tier split of `total_hits`, reset with it.
		#[cfg(feature = "hybrid_cache_common")]
		self.hybrid_fast_hits.store(0, Ordering::Relaxed);
		#[cfg(feature = "hybrid_cache_common")]
		self.hybrid_slow_hits.store(0, Ordering::Relaxed);

		// `HybridStats` documents the three tier-movement counters as totals
		// since creation or the last `wipe()`, so a wipe resets them along
		// with the request counters above.
		#[cfg(feature = "hybrid_cache_common")]
		self.hybrid_promotions.store(0, Ordering::Relaxed);
		#[cfg(feature = "hybrid_cache_common")]
		self.hybrid_demotions.store(0, Ordering::Relaxed);
		#[cfg(feature = "hybrid_cache_common")]
		self.hybrid_evictions.store(0, Ordering::Relaxed);
		#[cfg(feature = "hybrid_cache_common")]
		self.hybrid_reconcile_applied_to_fast.store(0, Ordering::Relaxed);
		#[cfg(feature = "hybrid_cache_common")]
		self.hybrid_reconcile_applied_to_slow.store(0, Ordering::Relaxed);

		// An empty stack is unlatched. The policy worker clears the status in
		// its `Wipe` handling, after the stack, and republishes the stack's
		// latch right after; `wipe()` returns only then.
		#[cfg(feature = "hybrid_cache_common")]
		self.hybrid_admission_latched.store(false, Ordering::Relaxed);

		// The admission counters, with the others. The gate's live state -- its
		// configuration, both lanes and their waiters, the reservations, a
		// stall, the published figures and levels, the worker's bits -- is not a
		// counter and survives (S5): an outstanding permit's release must find
		// the reservation it added to.
		#[cfg(feature = "hybrid_cache_common")]
		self.gate.reset_counters();
	}

	pub fn try_to_status(&self) -> Result<Status, CacheError> {
		let policy = self.policy();

		let Ok(rss) = mem::rss(None) else {
			error!("Could not get RSS");
			return Err(CacheError::Internal);
		};

		let Ok(hwm) = mem::hwm(None) else {
			error!("Could not get HWM");
			return Err(CacheError::Internal);
		};

		let status = Status {
			pid: process::id(),

			max_size: self.max_size(),
			used_size: self.used_size(&policy),
			num_objects: self.live_num_objects(),

			rss,
			hwm,

			total_hits: self.total_hits.load(Ordering::Relaxed),
			total_gets: self.total_gets.load(Ordering::Relaxed),
			total_sets: self.total_sets.load(Ordering::Relaxed),
			total_dels: self.total_dels.load(Ordering::Relaxed),

			policies: self.policies.clone(),
			policy: self.policy,

			start_time: self.start_time.load(Ordering::Relaxed),
		};

		Ok(status)
	}
}

#[cfg(test)]
mod tests {
	use std::sync::atomic::Ordering;

	use crate::{
		CacheSize,
		PaperPolicy,
		object::overhead::get_policy_overhead,
		status::{AtomicStatus, Cleared},
	};

	#[test]
	fn it_clears_atomic_status() {
		let status = AtomicStatus::new(
			1000,
			&[PaperPolicy::LfuCompact],
			PaperPolicy::LfuCompact,
		).expect("Could not initialize atomic status");

		status.update_base_used_size(1);
		status.incr_num_objects();
		status.incr_hits();
		status.incr_sets();
		status.incr_dels();

		assert_eq!(status.base_used_size.load(Ordering::Acquire), 1);
		assert_eq!(status.num_objects.load(Ordering::Acquire), 1);
		assert_eq!(status.total_gets.load(Ordering::Relaxed), 1);
		assert_eq!(status.total_hits.load(Ordering::Relaxed), 1);
		assert_eq!(status.total_sets.load(Ordering::Relaxed), 1);
		assert_eq!(status.total_dels.load(Ordering::Relaxed), 1);

		status.clear(Cleared { objects: 1, base_bytes: 1 });

		assert_eq!(status.base_used_size.load(Ordering::Acquire), 0);
		assert_eq!(status.num_objects.load(Ordering::Acquire), 0);
		assert_eq!(status.total_gets.load(Ordering::Relaxed), 0);
		assert_eq!(status.total_hits.load(Ordering::Relaxed), 0);
		assert_eq!(status.total_sets.load(Ordering::Relaxed), 0);
		assert_eq!(status.total_dels.load(Ordering::Relaxed), 0);
	}

	/// A wipe's clear takes off what the map's clear removed, so a client's
	/// insert racing it ends counted exactly, whichever side of the clear it
	/// landed on; and while the insert's own update is still to come the two
	/// figures are below zero, which the readers take as zero. Red with the
	/// figures stored as 0 (`wipestorezero`: the removed insert's update then
	/// leaves one object counted) and with the clamp off (`noclamp`: the
	/// readers see 2^64 less a little).
	#[test]
	fn a_clear_takes_off_what_the_map_removed_and_a_racing_insert_ends_exact() {
		let policy = PaperPolicy::LfuCompact;
		let status = AtomicStatus::new(1_000_000, &[policy], policy).expect("a status");
		let overhead = get_policy_overhead(&policy) as CacheSize;

		// Two objects set and counted; a third inserted into the map, its own
		// update not made yet.
		for _ in 0..2 {
			status.incr_num_objects();
			status.update_base_used_size(100);
		}

		assert_eq!(status.used_size(&policy), 200 + 2 * overhead);

		// The map's clear removed all three.
		status.clear(Cleared { objects: 3, base_bytes: 300 });

		assert_eq!(status.live_num_objects(), 0, "below zero, read as zero");
		assert_eq!(status.used_size(&policy), 0, "below zero, read as zero");

		// The third insert's update lands: nothing left counted.
		status.incr_num_objects();
		status.update_base_used_size(100);

		assert_eq!((status.live_num_objects(), status.used_size(&policy)), (0, 0), "the removed insert is not counted");

		// An insert into a shard the clear had already emptied: live, counted,
		// and its removal brings both figures back to zero.
		status.incr_num_objects();
		status.update_base_used_size(100);

		assert_eq!((status.live_num_objects(), status.used_size(&policy)), (1, 100 + overhead));

		status.update_base_used_size(-100);
		status.decr_num_objects();

		assert_eq!((status.live_num_objects(), status.used_size(&policy)), (0, 0), "nothing wrapped");
	}

	#[cfg(feature = "hybrid_cache_common")]
	#[test]
	fn it_clears_hybrid_counters() {
		let status = AtomicStatus::new(
			1000,
			&[PaperPolicy::LfuCompact],
			PaperPolicy::LfuCompact,
		).expect("Could not initialize atomic status");

		status.record_hybrid_promotion();
		status.record_hybrid_demotion();
		status.record_hybrid_eviction();

		let stats = status.hybrid_stats();
		assert_ne!(stats.promotions, 0);
		assert_ne!(stats.demotions, 0);
		assert_ne!(stats.evictions, 0);

		status.clear(Cleared::default());

		let stats = status.hybrid_stats();
		assert_eq!(stats.promotions, 0);
		assert_eq!(stats.demotions, 0);
		assert_eq!(stats.evictions, 0);
	}

	#[cfg(feature = "hybrid_cache_common")]
	#[test]
	fn served_hits_are_split_by_tier_and_cleared_with_the_hits() {
		use crate::Tier;

		let status = AtomicStatus::new(
			1000,
			&[PaperPolicy::LruCompactHybrid],
			PaperPolicy::LruCompactHybrid,
		).expect("Could not initialize atomic status");

		status.incr_served_hit(Tier::Fast);
		status.incr_served_hit(Tier::Fast);
		status.incr_served_hit(Tier::Slow);

		let stats = status.hybrid_stats();
		assert_eq!((stats.fast_hits, stats.slow_hits), (2, 1));

		status.clear(Cleared::default());

		let stats = status.hybrid_stats();
		assert_eq!((stats.fast_hits, stats.slow_hits), (0, 0));
	}

	#[cfg(feature = "hybrid_cache_common")]
	#[test]
	fn effective_fast_capacity_takes_the_reservation_off_the_whole_budget() {
		let status = AtomicStatus::new(
			1_000_000,
			&[PaperPolicy::LruCompactHybrid],
			PaperPolicy::LruCompactHybrid,
		).expect("Could not initialize atomic status");

		status.set_fast_tier_capacity(10_000);
		assert_eq!(status.per_object_effective_fast_capacity(), 10_000, "nothing live, nothing reserved");

		status.register_tiered_cache(100);
		status.register_tiered_cache(7);
		assert_eq!(status.hybrid_shared_overhead(), 100, "the first registration stands");

		status.add_num_objects(30);
		assert_eq!(status.per_object_effective_fast_capacity(), 10_000 - 30 * 100);

		// The size-split design's large segment is part of F.
		status.set_hybrid_large_fast_capacity(500);
		assert_eq!(status.per_object_effective_fast_capacity(), 10_500 - 3_000);

		// Saturating: a reservation past the budget leaves nothing, not a wrap.
		status.add_num_objects(100);
		assert_eq!(status.per_object_effective_fast_capacity(), 0);

		// S5: the exported figure is the one the policy worker publishes, and no
		// worker has published to this status.
		assert_eq!(status.hybrid_stats().effective_fast_capacity, status.effective_fast_capacity());
	}
}
