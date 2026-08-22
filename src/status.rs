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
		atomic::{Ordering, AtomicBool, AtomicU64, AtomicUsize},
	},
};

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

/// Loads the active design's tier counters and gauges into [`HybridStats`].
///
/// One accessor serves every design: the per-design `<design>_hybrid_stats()`
/// methods were removed by the runtime-policy unification, and the 18
/// `<Design>HybridStats` names are aliases of the one struct. Of its 15
/// fields, the 8 size-split gauges are populated only under
/// `PaperPolicy::LruSizedHybrid` and read zero elsewhere.
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
	is_auto_policy: bool,

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
	policy_index: AtomicUsize,
	is_auto_policy: AtomicBool,

	start_time: AtomicU64,

	/// Runtime-configurable fast-tier byte budget for `PaperPolicy::LruHybrid`
	/// / `PaperPolicy::LfuHybrid` / `PaperPolicy::TwoQHybrid` /
	/// `PaperPolicy::FifoHybrid` (`lru_hybrid_cache` / `lfu_hybrid_cache` /
	/// `two_q_hybrid_cache` / `fifo_hybrid_cache` — one field serving whichever
	/// design the cache was constructed with). Written by
	/// `PaperCache::set_fast_tier_size`, read back by both
	/// `PaperCache::fast_tier_size` and `PolicyWorker` (via the
	/// `WorkerEvent::ResizeFastTier` broadcast, not by reading this field
	/// directly — mirrors how `max_size` and `resize()`/`Resize` work).
	///
	/// For `PaperPolicy::LruSizedHybrid` (`lru_sized_hybrid_cache`)
	/// specifically, this field means the SMALL fast segment's capacity —
	/// that design has a second, independent fast segment ("large") with its
	/// own dedicated `hybrid_large_fast_capacity` field below,
	/// since a single shared field can't represent two independent budgets.
	#[cfg(feature = "hybrid_cache_common")]
	fast_tier_capacity: AtomicCacheSize,

	/// `lru_hybrid_cache` counters/gauges, updated by `PolicyWorker` as it
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
	#[cfg(feature = "hybrid_cache_common")]
	hybrid_fast_bytes_used: AtomicCacheSize,
	#[cfg(feature = "hybrid_cache_common")]
	hybrid_slow_bytes_used: AtomicCacheSize,
	#[cfg(feature = "hybrid_cache_common")]
	hybrid_fast_objects: AtomicU64,
	#[cfg(feature = "hybrid_cache_common")]
	hybrid_slow_objects: AtomicU64,



	/// Mirrors `LfuHybridStack::admission_latched()` (see that trait method's
	/// doc). Written by `PolicyWorker` every time it runs
	/// `apply_tier_migrations`, read by `PaperCache::set()` — running on the
	/// API-calling thread, which has no direct access to the worker-owned
	/// policy stack — so a brand-new key can be built as
	/// `TieredBuffer::new_slow` directly once the fast tier has genuinely
	/// filled, instead of always guessing `new_fast` and relying on an async
	/// correction.
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

	/// Returns `true` if the cache is configured to automatically
	/// switch eviction policies.
	#[must_use]
	pub fn is_auto_policy(&self) -> bool {
		self.is_auto_policy
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
		mut policy: PaperPolicy,
	) -> Result<Self, CacheError> {
		let policies: Arc<[PaperPolicy]> = policies.into();
		let is_auto_policy = policy.is_auto();

		if is_auto_policy {
			policy = PaperPolicy::Lfu;
		}

		let policy_index = get_policy_index(&policies, policy)?;

		let status = AtomicStatus {
			max_size: AtomicCacheSize::new(max_size),
			base_used_size: AtomicCacheSize::default(),
			num_objects: AtomicU64::default(),

			total_hits: AtomicU64::default(),
			total_gets: AtomicU64::default(),
			total_sets: AtomicU64::default(),
			total_dels: AtomicU64::default(),

			policies,
			policy_index: AtomicUsize::new(policy_index),
			is_auto_policy: AtomicBool::new(is_auto_policy),

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
			hybrid_fast_bytes_used: AtomicCacheSize::default(),
			#[cfg(feature = "hybrid_cache_common")]
			hybrid_slow_bytes_used: AtomicCacheSize::default(),
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

		};

		Ok(status)
	}

	#[must_use]
	pub fn max_size(&self) -> CacheSize {
		self.max_size.load(Ordering::Relaxed)
	}

	#[must_use]
	pub fn used_size(&self, policy: &PaperPolicy) -> CacheSize {
		let base_used_size = self.base_used_size.load(Ordering::Acquire);
		let num_objects = self.num_objects.load(Ordering::Acquire);
		let policy_overhead = get_policy_overhead(policy);

		base_used_size + num_objects * policy_overhead as CacheSize
	}

	#[must_use]
	pub fn policies(&self) -> &[PaperPolicy] {
		&self.policies
	}

	#[must_use]
	pub fn policy(&self) -> PaperPolicy {
		let policy_index = self.policy_index.load(Ordering::Relaxed);
		self.policies[policy_index]
	}

	#[must_use]
	pub fn is_auto_policy(&self) -> bool {
		self.is_auto_policy.load(Ordering::Relaxed)
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

	pub fn incr_num_objects(&self) {
		self.num_objects.fetch_add(1, Ordering::AcqRel);
	}

	pub fn add_num_objects(&self, count: u64) {
		if count > 0 {
			self.num_objects.fetch_add(count, Ordering::AcqRel);
		}
	}

	pub fn decr_num_objects(&self) {
		self.num_objects.fetch_sub(1, Ordering::AcqRel);
	}

	pub fn set_policy(&self, policy: PaperPolicy) -> Result<(), CacheError> {
		if policy.is_auto() {
			self.is_auto_policy.store(true, Ordering::Relaxed);
			return Ok(());
		}

		let index = get_policy_index(&self.policies, policy)?;

		self.policy_index.store(index, Ordering::Relaxed);
		self.is_auto_policy.store(false, Ordering::Relaxed);

		Ok(())
	}

	pub fn set_auto_policy(&self, policy: PaperPolicy) -> Result<(), CacheError> {
		if policy.is_auto() {
			error!("Attempting to set recursive auto policy");
			return Err(CacheError::Internal);
		}

		let index = get_policy_index(&self.policies, policy)?;
		self.policy_index.store(index, Ordering::Relaxed);

		Ok(())
	}

	#[must_use]
	pub fn exceeds_max_size(&self, size: impl AsPrimitive<u64>) -> bool {
		size.as_() > self.max_size.load(Ordering::Relaxed)
	}

	/// Current fast-tier byte budget (`PaperPolicy::LruHybrid` /
	/// `PaperPolicy::LfuHybrid` / `PaperPolicy::TwoQHybrid` /
	/// `PaperPolicy::FifoHybrid` / `PaperPolicy::LruSizedHybrid`'s SMALL
	/// segment — see the field's doc on the struct).
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
	) {
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
		HybridStats {
			promotions: self.hybrid_promotions.load(Ordering::Relaxed),
			demotions: self.hybrid_demotions.load(Ordering::Relaxed),
			evictions: self.hybrid_evictions.load(Ordering::Relaxed),
			fast_bytes_used: self.hybrid_fast_bytes_used.load(Ordering::Relaxed),
			slow_bytes_used: self.hybrid_slow_bytes_used.load(Ordering::Relaxed),
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
		}
	}









	/// Records `count` demotions at once — used when draining
	/// `LfuHybridStack::drain_demotions`, which reports genuine
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



	/// Mirrors `LfuHybridStack::admission_latched()`'s current value. Written
	/// by `PolicyWorker::apply_tier_migrations` every time it runs, read by
	/// `PaperCache::set()` on the API-calling thread — see the field's doc
	/// on the struct for why this needs to cross threads via an atomic
	/// rather than a direct call into the stack.
	#[cfg(feature = "hybrid_cache_common")]
	pub fn set_hybrid_admission_latched(&self, latched: bool) {
		self.hybrid_admission_latched.store(latched, Ordering::Relaxed);
	}

	/// Current best-known value of `LfuHybridStack::admission_latched()`.
	/// May be up to one worker event-loop iteration stale relative to the
	/// stack's true internal state — see `set_hybrid_admission_latched`.
	#[cfg(feature = "hybrid_cache_common")]
	#[must_use]
	pub fn hybrid_admission_latched(&self) -> bool {
		self.hybrid_admission_latched.load(Ordering::Relaxed)
	}












	// `record_hybrid_promotion` intentionally does not exist:
	// `FifoHybridStack` never emits a `Tier::Fast` migration (no promotion
	// policy at all — see that stack's module doc), so
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
























	pub fn clear(&self) {
		self.base_used_size.store(0, Ordering::Release);
		self.num_objects.store(0, Ordering::Release);

		self.total_hits.store(0, Ordering::Relaxed);
		self.total_gets.store(0, Ordering::Relaxed);
		self.total_sets.store(0, Ordering::Relaxed);
		self.total_dels.store(0, Ordering::Relaxed);

		// Reset synchronously here (called from `wipe()` on the API-calling
		// thread) rather than waiting for `PolicyWorker` to process the
		// corresponding `WorkerEvent::Wipe` and resync via
		// `apply_tier_migrations` — closes the window where `set()` could
		// otherwise read a stale `true` and build a brand-new key as
		// `TieredBuffer::new_slow` right after a wipe, before the stack
		// itself has caught up to also being empty (and thus unlatched).
		#[cfg(feature = "hybrid_cache_common")]
		self.hybrid_admission_latched.store(false, Ordering::Relaxed);
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
			num_objects: self.num_objects.load(Ordering::Acquire),

			rss,
			hwm,

			total_hits: self.total_hits.load(Ordering::Relaxed),
			total_gets: self.total_gets.load(Ordering::Relaxed),
			total_sets: self.total_sets.load(Ordering::Relaxed),
			total_dels: self.total_dels.load(Ordering::Relaxed),

			policies: self.policies.clone(),
			policy: self.policies[self.policy_index.load(Ordering::Relaxed)],
			is_auto_policy: self.is_auto_policy.load(Ordering::Relaxed),

			start_time: self.start_time.load(Ordering::Relaxed),
		};

		Ok(status)
	}
}

fn get_policy_index(
	policies: &[PaperPolicy],
	policy: PaperPolicy,
) -> Result<usize, CacheError> {
	let maybe_index = policies
		.iter()
		.position(|configured_policy| configured_policy.eq(&policy));

	match maybe_index {
		Some(index) => Ok(index),

		None => {
			error!("Could not find policy index");
			Err(CacheError::Internal)
		},
	}
}

#[cfg(test)]
mod tests {
	use std::sync::atomic::Ordering;

	use crate::{
		PaperPolicy,
		status::AtomicStatus,
	};

	#[test]
	fn it_clears_atomic_status() {
		let status = AtomicStatus::new(
			1000,
			&[PaperPolicy::Lfu],
			PaperPolicy::Lfu,
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

		status.clear();

		assert_eq!(status.base_used_size.load(Ordering::Acquire), 0);
		assert_eq!(status.num_objects.load(Ordering::Acquire), 0);
		assert_eq!(status.total_gets.load(Ordering::Relaxed), 0);
		assert_eq!(status.total_hits.load(Ordering::Relaxed), 0);
		assert_eq!(status.total_sets.load(Ordering::Relaxed), 0);
		assert_eq!(status.total_dels.load(Ordering::Relaxed), 0);
	}
}
