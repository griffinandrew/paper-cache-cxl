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
	/// `two_q_hybrid_cache` / `fifo_hybrid_cache` — mutually exclusive
	/// features, see `lib.rs`'s `compile_error!` guards, so this single field
	/// serves whichever one is active). Written by
	/// `PaperCache::set_fast_tier_size`, read back by both
	/// `PaperCache::fast_tier_size` and `PolicyWorker` (via the
	/// `WorkerEvent::ResizeFastTier` broadcast, not by reading this field
	/// directly — mirrors how `max_size` and `resize()`/`Resize` work).
	///
	/// For `PaperPolicy::LruSizedHybrid` (`lru_sized_hybrid_cache`)
	/// specifically, this field means the SMALL fast segment's capacity —
	/// that design has a second, independent fast segment ("large") with its
	/// own dedicated `lru_sized_hybrid_large_fast_capacity` field below,
	/// since a single shared field can't represent two independent budgets.
	#[cfg(any(feature = "lru_hybrid_cache", feature = "lfu_hybrid_cache", feature = "two_q_hybrid_cache", feature = "fifo_hybrid_cache", feature = "lru_sized_hybrid_cache", feature = "s3_fifo_hybrid_cache", feature = "two_q_ghost_hybrid_cache", feature = "s3_fifo_ghost_hybrid_cache", feature = "s3_fifo_ghost_lazy_demotion_hybrid_cache", feature = "s3_fifo_ghost_lazy_demotion_fast_admission_hybrid_cache", feature = "s3_fifo_ghost_lazy_demotion_fast_admission_midpoint_hybrid_cache"))]
	fast_tier_capacity: AtomicCacheSize,

	/// `lru_hybrid_cache` counters/gauges, updated by `PolicyWorker` as it
	/// processes tier migrations and evictions; read via `lru_hybrid_stats`.
	/// Lives here (rather than as a field on `PaperCache` itself) so that
	/// adding this feature doesn't require touching every other value type's
	/// constructor throughout lib.rs — `AtomicStatus::new` is the only
	/// construction site.
	#[cfg(feature = "lru_hybrid_cache")]
	lru_hybrid_promotions: AtomicU64,
	#[cfg(feature = "lru_hybrid_cache")]
	lru_hybrid_demotions: AtomicU64,
	#[cfg(feature = "lru_hybrid_cache")]
	lru_hybrid_evictions: AtomicU64,
	#[cfg(feature = "lru_hybrid_cache")]
	lru_hybrid_fast_bytes_used: AtomicCacheSize,
	#[cfg(feature = "lru_hybrid_cache")]
	lru_hybrid_slow_bytes_used: AtomicCacheSize,
	#[cfg(feature = "lru_hybrid_cache")]
	lru_hybrid_fast_objects: AtomicU64,
	#[cfg(feature = "lru_hybrid_cache")]
	lru_hybrid_slow_objects: AtomicU64,

	/// `lfu_hybrid_cache` counters/gauges — same rationale as the
	/// `lru_hybrid_*` fields above (kept as a separate, independently named
	/// set rather than merged with them, so each feature stays
	/// self-contained/removable, matching how `tiering_manager` already
	/// coexists as a separate concept in this file).
	#[cfg(feature = "lfu_hybrid_cache")]
	lfu_hybrid_promotions: AtomicU64,
	#[cfg(feature = "lfu_hybrid_cache")]
	lfu_hybrid_demotions: AtomicU64,
	#[cfg(feature = "lfu_hybrid_cache")]
	lfu_hybrid_evictions: AtomicU64,
	#[cfg(feature = "lfu_hybrid_cache")]
	lfu_hybrid_fast_bytes_used: AtomicCacheSize,
	#[cfg(feature = "lfu_hybrid_cache")]
	lfu_hybrid_slow_bytes_used: AtomicCacheSize,
	#[cfg(feature = "lfu_hybrid_cache")]
	lfu_hybrid_fast_objects: AtomicU64,
	#[cfg(feature = "lfu_hybrid_cache")]
	lfu_hybrid_slow_objects: AtomicU64,

	/// Mirrors `LfuHybridStack::admission_latched()` (see that trait method's
	/// doc). Written by `PolicyWorker` every time it runs
	/// `apply_tier_migrations`, read by `PaperCache::set()` — running on the
	/// API-calling thread, which has no direct access to the worker-owned
	/// policy stack — so a brand-new key can be built as
	/// `TieredBuffer::new_slow` directly once the fast tier has genuinely
	/// filled, instead of always guessing `new_fast` and relying on an async
	/// correction.
	#[cfg(feature = "lfu_hybrid_cache")]
	lfu_hybrid_admission_latched: AtomicBool,

	/// `two_q_hybrid_cache` counters/gauges — same rationale as the
	/// `lru_hybrid_*`/`lfu_hybrid_*` fields above.
	#[cfg(feature = "two_q_hybrid_cache")]
	two_q_hybrid_promotions: AtomicU64,
	#[cfg(feature = "two_q_hybrid_cache")]
	two_q_hybrid_demotions: AtomicU64,
	#[cfg(feature = "two_q_hybrid_cache")]
	two_q_hybrid_evictions: AtomicU64,
	#[cfg(feature = "two_q_hybrid_cache")]
	two_q_hybrid_fast_bytes_used: AtomicCacheSize,
	#[cfg(feature = "two_q_hybrid_cache")]
	two_q_hybrid_slow_bytes_used: AtomicCacheSize,
	#[cfg(feature = "two_q_hybrid_cache")]
	two_q_hybrid_fast_objects: AtomicU64,
	#[cfg(feature = "two_q_hybrid_cache")]
	two_q_hybrid_slow_objects: AtomicU64,

	/// `fifo_hybrid_cache` counters/gauges — same rationale as the
	/// `lru_hybrid_*`/`lfu_hybrid_*`/`two_q_hybrid_*` fields above.
	/// `fifo_hybrid_promotions` stays permanently `0` — FIFO has no
	/// promotion policy at all — kept for API-shape symmetry with the other
	/// three hybrids' stats (see `FifoHybridStats::promotions`'s doc).
	#[cfg(feature = "fifo_hybrid_cache")]
	fifo_hybrid_promotions: AtomicU64,
	#[cfg(feature = "fifo_hybrid_cache")]
	fifo_hybrid_demotions: AtomicU64,
	#[cfg(feature = "fifo_hybrid_cache")]
	fifo_hybrid_evictions: AtomicU64,
	#[cfg(feature = "fifo_hybrid_cache")]
	fifo_hybrid_fast_bytes_used: AtomicCacheSize,
	#[cfg(feature = "fifo_hybrid_cache")]
	fifo_hybrid_slow_bytes_used: AtomicCacheSize,
	#[cfg(feature = "fifo_hybrid_cache")]
	fifo_hybrid_fast_objects: AtomicU64,
	#[cfg(feature = "fifo_hybrid_cache")]
	fifo_hybrid_slow_objects: AtomicU64,

	/// `lru_sized_hybrid_cache` counters/gauges — same rationale as the
	/// `lru_hybrid_*`/`lfu_hybrid_*`/`two_q_hybrid_*`/`fifo_hybrid_*` fields
	/// above. The `_fast_bytes_used`/`_slow_bytes_used`/`_fast_objects`/
	/// `_slow_objects` gauges are the small+large COMBINED totals (kept for
	/// drop-in consistency with the other four hybrids' stats shape); the
	/// `_small_*`/`_large_*` gauges below give the finer per-segment
	/// breakdown this design's two independent fast segments and two
	/// independent slow lists actually need.
	#[cfg(feature = "lru_sized_hybrid_cache")]
	lru_sized_hybrid_promotions: AtomicU64,
	#[cfg(feature = "lru_sized_hybrid_cache")]
	lru_sized_hybrid_demotions: AtomicU64,
	#[cfg(feature = "lru_sized_hybrid_cache")]
	lru_sized_hybrid_evictions: AtomicU64,
	#[cfg(feature = "lru_sized_hybrid_cache")]
	lru_sized_hybrid_fast_bytes_used: AtomicCacheSize,
	#[cfg(feature = "lru_sized_hybrid_cache")]
	lru_sized_hybrid_slow_bytes_used: AtomicCacheSize,
	#[cfg(feature = "lru_sized_hybrid_cache")]
	lru_sized_hybrid_fast_objects: AtomicU64,
	#[cfg(feature = "lru_sized_hybrid_cache")]
	lru_sized_hybrid_slow_objects: AtomicU64,
	#[cfg(feature = "lru_sized_hybrid_cache")]
	lru_sized_hybrid_small_fast_bytes_used: AtomicCacheSize,
	#[cfg(feature = "lru_sized_hybrid_cache")]
	lru_sized_hybrid_large_fast_bytes_used: AtomicCacheSize,
	#[cfg(feature = "lru_sized_hybrid_cache")]
	lru_sized_hybrid_small_fast_objects: AtomicU64,
	#[cfg(feature = "lru_sized_hybrid_cache")]
	lru_sized_hybrid_large_fast_objects: AtomicU64,
	#[cfg(feature = "lru_sized_hybrid_cache")]
	lru_sized_hybrid_small_slow_bytes_used: AtomicCacheSize,
	#[cfg(feature = "lru_sized_hybrid_cache")]
	lru_sized_hybrid_large_slow_bytes_used: AtomicCacheSize,
	#[cfg(feature = "lru_sized_hybrid_cache")]
	lru_sized_hybrid_small_slow_objects: AtomicU64,
	#[cfg(feature = "lru_sized_hybrid_cache")]
	lru_sized_hybrid_large_slow_objects: AtomicU64,

	/// The LARGE fast segment's own capacity (the SMALL segment reuses the
	/// shared `fast_tier_capacity` field above — see that field's doc for
	/// why). Written by `PaperCache::set_large_fast_tier_size`, read back by
	/// `PaperCache::large_fast_tier_size` and `PolicyWorker` (via
	/// `WorkerEvent::ResizeLargeFastTier`, not by reading this field
	/// directly — same indirection `fast_tier_capacity` uses).
	#[cfg(feature = "lru_sized_hybrid_cache")]
	lru_sized_hybrid_large_fast_capacity: AtomicCacheSize,

	/// The small/large size-classification threshold, in bytes. Written by
	/// `PaperCache::set_size_threshold`, read back by
	/// `PaperCache::size_threshold` and `PolicyWorker` (via
	/// `WorkerEvent::ResizeSizeThreshold`).
	#[cfg(feature = "lru_sized_hybrid_cache")]
	lru_sized_hybrid_size_threshold: AtomicCacheSize,

	/// `s3_fifo_hybrid_cache` counters/gauges — same rationale as the
	/// `lru_hybrid_*`/`lfu_hybrid_*`/`two_q_hybrid_*`/`fifo_hybrid_*`/
	/// `lru_sized_hybrid_*` fields above.
	#[cfg(feature = "s3_fifo_hybrid_cache")]
	s3_fifo_hybrid_promotions: AtomicU64,
	#[cfg(feature = "s3_fifo_hybrid_cache")]
	s3_fifo_hybrid_demotions: AtomicU64,
	#[cfg(feature = "s3_fifo_hybrid_cache")]
	s3_fifo_hybrid_evictions: AtomicU64,
	#[cfg(feature = "s3_fifo_hybrid_cache")]
	s3_fifo_hybrid_fast_bytes_used: AtomicCacheSize,
	#[cfg(feature = "s3_fifo_hybrid_cache")]
	s3_fifo_hybrid_slow_bytes_used: AtomicCacheSize,
	#[cfg(feature = "s3_fifo_hybrid_cache")]
	s3_fifo_hybrid_fast_objects: AtomicU64,
	#[cfg(feature = "s3_fifo_hybrid_cache")]
	s3_fifo_hybrid_slow_objects: AtomicU64,

	/// `two_q_ghost_hybrid_cache` counters/gauges — same rationale as the
	/// fields above. The ghost queue itself has no separate gauge (see
	/// `object/overhead.rs`'s ghost-hybrid arms for why its memory isn't
	/// charged/tracked per-object at all).
	#[cfg(feature = "two_q_ghost_hybrid_cache")]
	two_q_ghost_hybrid_promotions: AtomicU64,
	#[cfg(feature = "two_q_ghost_hybrid_cache")]
	two_q_ghost_hybrid_demotions: AtomicU64,
	#[cfg(feature = "two_q_ghost_hybrid_cache")]
	two_q_ghost_hybrid_evictions: AtomicU64,
	#[cfg(feature = "two_q_ghost_hybrid_cache")]
	two_q_ghost_hybrid_fast_bytes_used: AtomicCacheSize,
	#[cfg(feature = "two_q_ghost_hybrid_cache")]
	two_q_ghost_hybrid_slow_bytes_used: AtomicCacheSize,
	#[cfg(feature = "two_q_ghost_hybrid_cache")]
	two_q_ghost_hybrid_fast_objects: AtomicU64,
	#[cfg(feature = "two_q_ghost_hybrid_cache")]
	two_q_ghost_hybrid_slow_objects: AtomicU64,

	/// `s3_fifo_ghost_hybrid_cache` counters/gauges — same rationale.
	#[cfg(feature = "s3_fifo_ghost_hybrid_cache")]
	s3_fifo_ghost_hybrid_promotions: AtomicU64,
	#[cfg(feature = "s3_fifo_ghost_hybrid_cache")]
	s3_fifo_ghost_hybrid_demotions: AtomicU64,
	#[cfg(feature = "s3_fifo_ghost_hybrid_cache")]
	s3_fifo_ghost_hybrid_evictions: AtomicU64,
	#[cfg(feature = "s3_fifo_ghost_hybrid_cache")]
	s3_fifo_ghost_hybrid_fast_bytes_used: AtomicCacheSize,
	#[cfg(feature = "s3_fifo_ghost_hybrid_cache")]
	s3_fifo_ghost_hybrid_slow_bytes_used: AtomicCacheSize,
	#[cfg(feature = "s3_fifo_ghost_hybrid_cache")]
	s3_fifo_ghost_hybrid_fast_objects: AtomicU64,
	#[cfg(feature = "s3_fifo_ghost_hybrid_cache")]
	s3_fifo_ghost_hybrid_slow_objects: AtomicU64,

	/// `s3_fifo_ghost_lazy_demotion_hybrid_cache` counters/gauges — same
	/// rationale.
	#[cfg(feature = "s3_fifo_ghost_lazy_demotion_hybrid_cache")]
	s3_fifo_ghost_lazy_demotion_hybrid_promotions: AtomicU64,
	#[cfg(feature = "s3_fifo_ghost_lazy_demotion_hybrid_cache")]
	s3_fifo_ghost_lazy_demotion_hybrid_demotions: AtomicU64,
	#[cfg(feature = "s3_fifo_ghost_lazy_demotion_hybrid_cache")]
	s3_fifo_ghost_lazy_demotion_hybrid_evictions: AtomicU64,
	#[cfg(feature = "s3_fifo_ghost_lazy_demotion_hybrid_cache")]
	s3_fifo_ghost_lazy_demotion_hybrid_fast_bytes_used: AtomicCacheSize,
	#[cfg(feature = "s3_fifo_ghost_lazy_demotion_hybrid_cache")]
	s3_fifo_ghost_lazy_demotion_hybrid_slow_bytes_used: AtomicCacheSize,
	#[cfg(feature = "s3_fifo_ghost_lazy_demotion_hybrid_cache")]
	s3_fifo_ghost_lazy_demotion_hybrid_fast_objects: AtomicU64,
	#[cfg(feature = "s3_fifo_ghost_lazy_demotion_hybrid_cache")]
	s3_fifo_ghost_lazy_demotion_hybrid_slow_objects: AtomicU64,

	/// `s3_fifo_ghost_lazy_demotion_fast_admission_hybrid_cache`
	/// counters/gauges — same rationale.
	#[cfg(feature = "s3_fifo_ghost_lazy_demotion_fast_admission_hybrid_cache")]
	s3_fifo_ghost_lazy_demotion_fast_admission_hybrid_promotions: AtomicU64,
	#[cfg(feature = "s3_fifo_ghost_lazy_demotion_fast_admission_hybrid_cache")]
	s3_fifo_ghost_lazy_demotion_fast_admission_hybrid_demotions: AtomicU64,
	#[cfg(feature = "s3_fifo_ghost_lazy_demotion_fast_admission_hybrid_cache")]
	s3_fifo_ghost_lazy_demotion_fast_admission_hybrid_evictions: AtomicU64,
	#[cfg(feature = "s3_fifo_ghost_lazy_demotion_fast_admission_hybrid_cache")]
	s3_fifo_ghost_lazy_demotion_fast_admission_hybrid_fast_bytes_used: AtomicCacheSize,
	#[cfg(feature = "s3_fifo_ghost_lazy_demotion_fast_admission_hybrid_cache")]
	s3_fifo_ghost_lazy_demotion_fast_admission_hybrid_slow_bytes_used: AtomicCacheSize,
	#[cfg(feature = "s3_fifo_ghost_lazy_demotion_fast_admission_hybrid_cache")]
	s3_fifo_ghost_lazy_demotion_fast_admission_hybrid_fast_objects: AtomicU64,
	#[cfg(feature = "s3_fifo_ghost_lazy_demotion_fast_admission_hybrid_cache")]
	s3_fifo_ghost_lazy_demotion_fast_admission_hybrid_slow_objects: AtomicU64,

	/// `s3_fifo_ghost_lazy_demotion_fast_admission_midpoint_hybrid_cache`
	/// counters/gauges — same rationale.
	#[cfg(feature = "s3_fifo_ghost_lazy_demotion_fast_admission_midpoint_hybrid_cache")]
	s3_fifo_ghost_lazy_demotion_fast_admission_midpoint_hybrid_promotions: AtomicU64,
	#[cfg(feature = "s3_fifo_ghost_lazy_demotion_fast_admission_midpoint_hybrid_cache")]
	s3_fifo_ghost_lazy_demotion_fast_admission_midpoint_hybrid_demotions: AtomicU64,
	#[cfg(feature = "s3_fifo_ghost_lazy_demotion_fast_admission_midpoint_hybrid_cache")]
	s3_fifo_ghost_lazy_demotion_fast_admission_midpoint_hybrid_evictions: AtomicU64,
	#[cfg(feature = "s3_fifo_ghost_lazy_demotion_fast_admission_midpoint_hybrid_cache")]
	s3_fifo_ghost_lazy_demotion_fast_admission_midpoint_hybrid_fast_bytes_used: AtomicCacheSize,
	#[cfg(feature = "s3_fifo_ghost_lazy_demotion_fast_admission_midpoint_hybrid_cache")]
	s3_fifo_ghost_lazy_demotion_fast_admission_midpoint_hybrid_slow_bytes_used: AtomicCacheSize,
	#[cfg(feature = "s3_fifo_ghost_lazy_demotion_fast_admission_midpoint_hybrid_cache")]
	s3_fifo_ghost_lazy_demotion_fast_admission_midpoint_hybrid_fast_objects: AtomicU64,
	#[cfg(feature = "s3_fifo_ghost_lazy_demotion_fast_admission_midpoint_hybrid_cache")]
	s3_fifo_ghost_lazy_demotion_fast_admission_midpoint_hybrid_slow_objects: AtomicU64,
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

			#[cfg(any(feature = "lru_hybrid_cache", feature = "lfu_hybrid_cache", feature = "two_q_hybrid_cache", feature = "fifo_hybrid_cache", feature = "lru_sized_hybrid_cache", feature = "s3_fifo_hybrid_cache", feature = "two_q_ghost_hybrid_cache", feature = "s3_fifo_ghost_hybrid_cache", feature = "s3_fifo_ghost_lazy_demotion_hybrid_cache", feature = "s3_fifo_ghost_lazy_demotion_fast_admission_hybrid_cache", feature = "s3_fifo_ghost_lazy_demotion_fast_admission_midpoint_hybrid_cache"))]
			fast_tier_capacity: AtomicCacheSize::default(),
			#[cfg(feature = "lru_hybrid_cache")]
			lru_hybrid_promotions: AtomicU64::default(),
			#[cfg(feature = "lru_hybrid_cache")]
			lru_hybrid_demotions: AtomicU64::default(),
			#[cfg(feature = "lru_hybrid_cache")]
			lru_hybrid_evictions: AtomicU64::default(),
			#[cfg(feature = "lru_hybrid_cache")]
			lru_hybrid_fast_bytes_used: AtomicCacheSize::default(),
			#[cfg(feature = "lru_hybrid_cache")]
			lru_hybrid_slow_bytes_used: AtomicCacheSize::default(),
			#[cfg(feature = "lru_hybrid_cache")]
			lru_hybrid_fast_objects: AtomicU64::default(),
			#[cfg(feature = "lru_hybrid_cache")]
			lru_hybrid_slow_objects: AtomicU64::default(),

			#[cfg(feature = "lfu_hybrid_cache")]
			lfu_hybrid_promotions: AtomicU64::default(),
			#[cfg(feature = "lfu_hybrid_cache")]
			lfu_hybrid_demotions: AtomicU64::default(),
			#[cfg(feature = "lfu_hybrid_cache")]
			lfu_hybrid_evictions: AtomicU64::default(),
			#[cfg(feature = "lfu_hybrid_cache")]
			lfu_hybrid_fast_bytes_used: AtomicCacheSize::default(),
			#[cfg(feature = "lfu_hybrid_cache")]
			lfu_hybrid_slow_bytes_used: AtomicCacheSize::default(),
			#[cfg(feature = "lfu_hybrid_cache")]
			lfu_hybrid_fast_objects: AtomicU64::default(),
			#[cfg(feature = "lfu_hybrid_cache")]
			lfu_hybrid_slow_objects: AtomicU64::default(),
			#[cfg(feature = "lfu_hybrid_cache")]
			lfu_hybrid_admission_latched: AtomicBool::new(false),

			#[cfg(feature = "two_q_hybrid_cache")]
			two_q_hybrid_promotions: AtomicU64::default(),
			#[cfg(feature = "two_q_hybrid_cache")]
			two_q_hybrid_demotions: AtomicU64::default(),
			#[cfg(feature = "two_q_hybrid_cache")]
			two_q_hybrid_evictions: AtomicU64::default(),
			#[cfg(feature = "two_q_hybrid_cache")]
			two_q_hybrid_fast_bytes_used: AtomicCacheSize::default(),
			#[cfg(feature = "two_q_hybrid_cache")]
			two_q_hybrid_slow_bytes_used: AtomicCacheSize::default(),
			#[cfg(feature = "two_q_hybrid_cache")]
			two_q_hybrid_fast_objects: AtomicU64::default(),
			#[cfg(feature = "two_q_hybrid_cache")]
			two_q_hybrid_slow_objects: AtomicU64::default(),

			#[cfg(feature = "fifo_hybrid_cache")]
			fifo_hybrid_promotions: AtomicU64::default(),
			#[cfg(feature = "fifo_hybrid_cache")]
			fifo_hybrid_demotions: AtomicU64::default(),
			#[cfg(feature = "fifo_hybrid_cache")]
			fifo_hybrid_evictions: AtomicU64::default(),
			#[cfg(feature = "fifo_hybrid_cache")]
			fifo_hybrid_fast_bytes_used: AtomicCacheSize::default(),
			#[cfg(feature = "fifo_hybrid_cache")]
			fifo_hybrid_slow_bytes_used: AtomicCacheSize::default(),
			#[cfg(feature = "fifo_hybrid_cache")]
			fifo_hybrid_fast_objects: AtomicU64::default(),
			#[cfg(feature = "fifo_hybrid_cache")]
			fifo_hybrid_slow_objects: AtomicU64::default(),

			#[cfg(feature = "lru_sized_hybrid_cache")]
			lru_sized_hybrid_promotions: AtomicU64::default(),
			#[cfg(feature = "lru_sized_hybrid_cache")]
			lru_sized_hybrid_demotions: AtomicU64::default(),
			#[cfg(feature = "lru_sized_hybrid_cache")]
			lru_sized_hybrid_evictions: AtomicU64::default(),
			#[cfg(feature = "lru_sized_hybrid_cache")]
			lru_sized_hybrid_fast_bytes_used: AtomicCacheSize::default(),
			#[cfg(feature = "lru_sized_hybrid_cache")]
			lru_sized_hybrid_slow_bytes_used: AtomicCacheSize::default(),
			#[cfg(feature = "lru_sized_hybrid_cache")]
			lru_sized_hybrid_fast_objects: AtomicU64::default(),
			#[cfg(feature = "lru_sized_hybrid_cache")]
			lru_sized_hybrid_slow_objects: AtomicU64::default(),
			#[cfg(feature = "lru_sized_hybrid_cache")]
			lru_sized_hybrid_small_fast_bytes_used: AtomicCacheSize::default(),
			#[cfg(feature = "lru_sized_hybrid_cache")]
			lru_sized_hybrid_large_fast_bytes_used: AtomicCacheSize::default(),
			#[cfg(feature = "lru_sized_hybrid_cache")]
			lru_sized_hybrid_small_fast_objects: AtomicU64::default(),
			#[cfg(feature = "lru_sized_hybrid_cache")]
			lru_sized_hybrid_large_fast_objects: AtomicU64::default(),
			#[cfg(feature = "lru_sized_hybrid_cache")]
			lru_sized_hybrid_small_slow_bytes_used: AtomicCacheSize::default(),
			#[cfg(feature = "lru_sized_hybrid_cache")]
			lru_sized_hybrid_large_slow_bytes_used: AtomicCacheSize::default(),
			#[cfg(feature = "lru_sized_hybrid_cache")]
			lru_sized_hybrid_small_slow_objects: AtomicU64::default(),
			#[cfg(feature = "lru_sized_hybrid_cache")]
			lru_sized_hybrid_large_slow_objects: AtomicU64::default(),
			#[cfg(feature = "lru_sized_hybrid_cache")]
			lru_sized_hybrid_large_fast_capacity: AtomicCacheSize::default(),
			#[cfg(feature = "lru_sized_hybrid_cache")]
			lru_sized_hybrid_size_threshold: AtomicCacheSize::default(),

			#[cfg(feature = "s3_fifo_hybrid_cache")]
			s3_fifo_hybrid_promotions: AtomicU64::default(),
			#[cfg(feature = "s3_fifo_hybrid_cache")]
			s3_fifo_hybrid_demotions: AtomicU64::default(),
			#[cfg(feature = "s3_fifo_hybrid_cache")]
			s3_fifo_hybrid_evictions: AtomicU64::default(),
			#[cfg(feature = "s3_fifo_hybrid_cache")]
			s3_fifo_hybrid_fast_bytes_used: AtomicCacheSize::default(),
			#[cfg(feature = "s3_fifo_hybrid_cache")]
			s3_fifo_hybrid_slow_bytes_used: AtomicCacheSize::default(),
			#[cfg(feature = "s3_fifo_hybrid_cache")]
			s3_fifo_hybrid_fast_objects: AtomicU64::default(),
			#[cfg(feature = "s3_fifo_hybrid_cache")]
			s3_fifo_hybrid_slow_objects: AtomicU64::default(),

			#[cfg(feature = "two_q_ghost_hybrid_cache")]
			two_q_ghost_hybrid_promotions: AtomicU64::default(),
			#[cfg(feature = "two_q_ghost_hybrid_cache")]
			two_q_ghost_hybrid_demotions: AtomicU64::default(),
			#[cfg(feature = "two_q_ghost_hybrid_cache")]
			two_q_ghost_hybrid_evictions: AtomicU64::default(),
			#[cfg(feature = "two_q_ghost_hybrid_cache")]
			two_q_ghost_hybrid_fast_bytes_used: AtomicCacheSize::default(),
			#[cfg(feature = "two_q_ghost_hybrid_cache")]
			two_q_ghost_hybrid_slow_bytes_used: AtomicCacheSize::default(),
			#[cfg(feature = "two_q_ghost_hybrid_cache")]
			two_q_ghost_hybrid_fast_objects: AtomicU64::default(),
			#[cfg(feature = "two_q_ghost_hybrid_cache")]
			two_q_ghost_hybrid_slow_objects: AtomicU64::default(),

			#[cfg(feature = "s3_fifo_ghost_hybrid_cache")]
			s3_fifo_ghost_hybrid_promotions: AtomicU64::default(),
			#[cfg(feature = "s3_fifo_ghost_hybrid_cache")]
			s3_fifo_ghost_hybrid_demotions: AtomicU64::default(),
			#[cfg(feature = "s3_fifo_ghost_hybrid_cache")]
			s3_fifo_ghost_hybrid_evictions: AtomicU64::default(),
			#[cfg(feature = "s3_fifo_ghost_hybrid_cache")]
			s3_fifo_ghost_hybrid_fast_bytes_used: AtomicCacheSize::default(),
			#[cfg(feature = "s3_fifo_ghost_hybrid_cache")]
			s3_fifo_ghost_hybrid_slow_bytes_used: AtomicCacheSize::default(),
			#[cfg(feature = "s3_fifo_ghost_hybrid_cache")]
			s3_fifo_ghost_hybrid_fast_objects: AtomicU64::default(),
			#[cfg(feature = "s3_fifo_ghost_hybrid_cache")]
			s3_fifo_ghost_hybrid_slow_objects: AtomicU64::default(),

			#[cfg(feature = "s3_fifo_ghost_lazy_demotion_hybrid_cache")]
			s3_fifo_ghost_lazy_demotion_hybrid_promotions: AtomicU64::default(),
			#[cfg(feature = "s3_fifo_ghost_lazy_demotion_hybrid_cache")]
			s3_fifo_ghost_lazy_demotion_hybrid_demotions: AtomicU64::default(),
			#[cfg(feature = "s3_fifo_ghost_lazy_demotion_hybrid_cache")]
			s3_fifo_ghost_lazy_demotion_hybrid_evictions: AtomicU64::default(),
			#[cfg(feature = "s3_fifo_ghost_lazy_demotion_hybrid_cache")]
			s3_fifo_ghost_lazy_demotion_hybrid_fast_bytes_used: AtomicCacheSize::default(),
			#[cfg(feature = "s3_fifo_ghost_lazy_demotion_hybrid_cache")]
			s3_fifo_ghost_lazy_demotion_hybrid_slow_bytes_used: AtomicCacheSize::default(),
			#[cfg(feature = "s3_fifo_ghost_lazy_demotion_hybrid_cache")]
			s3_fifo_ghost_lazy_demotion_hybrid_fast_objects: AtomicU64::default(),
			#[cfg(feature = "s3_fifo_ghost_lazy_demotion_hybrid_cache")]
			s3_fifo_ghost_lazy_demotion_hybrid_slow_objects: AtomicU64::default(),

			#[cfg(feature = "s3_fifo_ghost_lazy_demotion_fast_admission_hybrid_cache")]
			s3_fifo_ghost_lazy_demotion_fast_admission_hybrid_promotions: AtomicU64::default(),
			#[cfg(feature = "s3_fifo_ghost_lazy_demotion_fast_admission_hybrid_cache")]
			s3_fifo_ghost_lazy_demotion_fast_admission_hybrid_demotions: AtomicU64::default(),
			#[cfg(feature = "s3_fifo_ghost_lazy_demotion_fast_admission_hybrid_cache")]
			s3_fifo_ghost_lazy_demotion_fast_admission_hybrid_evictions: AtomicU64::default(),
			#[cfg(feature = "s3_fifo_ghost_lazy_demotion_fast_admission_hybrid_cache")]
			s3_fifo_ghost_lazy_demotion_fast_admission_hybrid_fast_bytes_used: AtomicCacheSize::default(),
			#[cfg(feature = "s3_fifo_ghost_lazy_demotion_fast_admission_hybrid_cache")]
			s3_fifo_ghost_lazy_demotion_fast_admission_hybrid_slow_bytes_used: AtomicCacheSize::default(),
			#[cfg(feature = "s3_fifo_ghost_lazy_demotion_fast_admission_hybrid_cache")]
			s3_fifo_ghost_lazy_demotion_fast_admission_hybrid_fast_objects: AtomicU64::default(),
			#[cfg(feature = "s3_fifo_ghost_lazy_demotion_fast_admission_hybrid_cache")]
			s3_fifo_ghost_lazy_demotion_fast_admission_hybrid_slow_objects: AtomicU64::default(),

			#[cfg(feature = "s3_fifo_ghost_lazy_demotion_fast_admission_midpoint_hybrid_cache")]
			s3_fifo_ghost_lazy_demotion_fast_admission_midpoint_hybrid_promotions: AtomicU64::default(),
			#[cfg(feature = "s3_fifo_ghost_lazy_demotion_fast_admission_midpoint_hybrid_cache")]
			s3_fifo_ghost_lazy_demotion_fast_admission_midpoint_hybrid_demotions: AtomicU64::default(),
			#[cfg(feature = "s3_fifo_ghost_lazy_demotion_fast_admission_midpoint_hybrid_cache")]
			s3_fifo_ghost_lazy_demotion_fast_admission_midpoint_hybrid_evictions: AtomicU64::default(),
			#[cfg(feature = "s3_fifo_ghost_lazy_demotion_fast_admission_midpoint_hybrid_cache")]
			s3_fifo_ghost_lazy_demotion_fast_admission_midpoint_hybrid_fast_bytes_used: AtomicCacheSize::default(),
			#[cfg(feature = "s3_fifo_ghost_lazy_demotion_fast_admission_midpoint_hybrid_cache")]
			s3_fifo_ghost_lazy_demotion_fast_admission_midpoint_hybrid_slow_bytes_used: AtomicCacheSize::default(),
			#[cfg(feature = "s3_fifo_ghost_lazy_demotion_fast_admission_midpoint_hybrid_cache")]
			s3_fifo_ghost_lazy_demotion_fast_admission_midpoint_hybrid_fast_objects: AtomicU64::default(),
			#[cfg(feature = "s3_fifo_ghost_lazy_demotion_fast_admission_midpoint_hybrid_cache")]
			s3_fifo_ghost_lazy_demotion_fast_admission_midpoint_hybrid_slow_objects: AtomicU64::default(),
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
	#[cfg(any(feature = "lru_hybrid_cache", feature = "lfu_hybrid_cache", feature = "two_q_hybrid_cache", feature = "fifo_hybrid_cache", feature = "lru_sized_hybrid_cache", feature = "s3_fifo_hybrid_cache", feature = "two_q_ghost_hybrid_cache", feature = "s3_fifo_ghost_hybrid_cache", feature = "s3_fifo_ghost_lazy_demotion_hybrid_cache", feature = "s3_fifo_ghost_lazy_demotion_fast_admission_hybrid_cache", feature = "s3_fifo_ghost_lazy_demotion_fast_admission_midpoint_hybrid_cache"))]
	#[must_use]
	pub fn fast_tier_capacity(&self) -> CacheSize {
		self.fast_tier_capacity.load(Ordering::Relaxed)
	}

	/// Sets the fast-tier byte budget. Callers are responsible for also
	/// broadcasting `WorkerEvent::ResizeFastTier` so the active stack's own
	/// internal capacity is updated (mirrors `set_max_size` + `Resize`).
	#[cfg(any(feature = "lru_hybrid_cache", feature = "lfu_hybrid_cache", feature = "two_q_hybrid_cache", feature = "fifo_hybrid_cache", feature = "lru_sized_hybrid_cache", feature = "s3_fifo_hybrid_cache", feature = "two_q_ghost_hybrid_cache", feature = "s3_fifo_ghost_hybrid_cache", feature = "s3_fifo_ghost_lazy_demotion_hybrid_cache", feature = "s3_fifo_ghost_lazy_demotion_fast_admission_hybrid_cache", feature = "s3_fifo_ghost_lazy_demotion_fast_admission_midpoint_hybrid_cache"))]
	pub fn set_fast_tier_capacity(&self, size: CacheSize) {
		self.fast_tier_capacity.store(size, Ordering::Relaxed);
	}

	#[cfg(feature = "lru_hybrid_cache")]
	pub fn record_lru_hybrid_promotion(&self) {
		self.lru_hybrid_promotions.fetch_add(1, Ordering::Relaxed);
	}

	#[cfg(feature = "lru_hybrid_cache")]
	pub fn record_lru_hybrid_demotion(&self) {
		self.lru_hybrid_demotions.fetch_add(1, Ordering::Relaxed);
	}

	#[cfg(feature = "lru_hybrid_cache")]
	pub fn record_lru_hybrid_eviction(&self) {
		self.lru_hybrid_evictions.fetch_add(1, Ordering::Relaxed);
	}

	/// Overwrites the live tier gauges (bytes/objects currently in each
	/// tier). Called by `PolicyWorker` each time it drains tier migrations.
	#[cfg(feature = "lru_hybrid_cache")]
	pub fn set_lru_hybrid_gauges(
		&self,
		fast_bytes_used: CacheSize,
		slow_bytes_used: CacheSize,
		fast_objects: u64,
		slow_objects: u64,
	) {
		self.lru_hybrid_fast_bytes_used.store(fast_bytes_used, Ordering::Relaxed);
		self.lru_hybrid_slow_bytes_used.store(slow_bytes_used, Ordering::Relaxed);
		self.lru_hybrid_fast_objects.store(fast_objects, Ordering::Relaxed);
		self.lru_hybrid_slow_objects.store(slow_objects, Ordering::Relaxed);
	}

	/// Returns a point-in-time snapshot of `lru_hybrid_cache` statistics.
	#[cfg(feature = "lru_hybrid_cache")]
	#[must_use]
	pub fn lru_hybrid_stats(&self) -> crate::lru_hybrid_cache::LruHybridStats {
		crate::lru_hybrid_cache::LruHybridStats {
			promotions: self.lru_hybrid_promotions.load(Ordering::Relaxed),
			demotions: self.lru_hybrid_demotions.load(Ordering::Relaxed),
			evictions: self.lru_hybrid_evictions.load(Ordering::Relaxed),
			fast_bytes_used: self.lru_hybrid_fast_bytes_used.load(Ordering::Relaxed),
			slow_bytes_used: self.lru_hybrid_slow_bytes_used.load(Ordering::Relaxed),
			fast_objects: self.lru_hybrid_fast_objects.load(Ordering::Relaxed),
			slow_objects: self.lru_hybrid_slow_objects.load(Ordering::Relaxed),
		}
	}

	#[cfg(feature = "lfu_hybrid_cache")]
	pub fn record_lfu_hybrid_promotion(&self) {
		self.lfu_hybrid_promotions.fetch_add(1, Ordering::Relaxed);
	}

	#[cfg(feature = "lfu_hybrid_cache")]
	pub fn record_lfu_hybrid_demotion(&self) {
		self.lfu_hybrid_demotions.fetch_add(1, Ordering::Relaxed);
	}

	/// Records `count` demotions at once — used when draining
	/// `LfuHybridStack::drain_demotions`, which reports genuine
	/// `settle_fast_tier` demotions in a single batch per
	/// `apply_tier_migrations` pass, distinct from admission-to-slow
	/// corrections (see that method's doc comment for why the two aren't
	/// the same thing).
	#[cfg(feature = "lfu_hybrid_cache")]
	pub fn record_lfu_hybrid_demotions(&self, count: u64) {
		if count > 0 {
			self.lfu_hybrid_demotions.fetch_add(count, Ordering::Relaxed);
		}
	}

	#[cfg(feature = "lfu_hybrid_cache")]
	pub fn record_lfu_hybrid_eviction(&self) {
		self.lfu_hybrid_evictions.fetch_add(1, Ordering::Relaxed);
	}

	/// Overwrites the live tier gauges (bytes/objects currently in each
	/// tier). Called by `PolicyWorker` each time it drains tier migrations.
	#[cfg(feature = "lfu_hybrid_cache")]
	pub fn set_lfu_hybrid_gauges(
		&self,
		fast_bytes_used: CacheSize,
		slow_bytes_used: CacheSize,
		fast_objects: u64,
		slow_objects: u64,
	) {
		self.lfu_hybrid_fast_bytes_used.store(fast_bytes_used, Ordering::Relaxed);
		self.lfu_hybrid_slow_bytes_used.store(slow_bytes_used, Ordering::Relaxed);
		self.lfu_hybrid_fast_objects.store(fast_objects, Ordering::Relaxed);
		self.lfu_hybrid_slow_objects.store(slow_objects, Ordering::Relaxed);
	}

	/// Mirrors `LfuHybridStack::admission_latched()`'s current value. Written
	/// by `PolicyWorker::apply_tier_migrations` every time it runs, read by
	/// `PaperCache::set()` on the API-calling thread — see the field's doc
	/// on the struct for why this needs to cross threads via an atomic
	/// rather than a direct call into the stack.
	#[cfg(feature = "lfu_hybrid_cache")]
	pub fn set_lfu_hybrid_admission_latched(&self, latched: bool) {
		self.lfu_hybrid_admission_latched.store(latched, Ordering::Relaxed);
	}

	/// Current best-known value of `LfuHybridStack::admission_latched()`.
	/// May be up to one worker event-loop iteration stale relative to the
	/// stack's true internal state — see `set_lfu_hybrid_admission_latched`.
	#[cfg(feature = "lfu_hybrid_cache")]
	#[must_use]
	pub fn lfu_hybrid_admission_latched(&self) -> bool {
		self.lfu_hybrid_admission_latched.load(Ordering::Relaxed)
	}

	/// Returns a point-in-time snapshot of `lfu_hybrid_cache` statistics.
	#[cfg(feature = "lfu_hybrid_cache")]
	#[must_use]
	pub fn lfu_hybrid_stats(&self) -> crate::lfu_hybrid_cache::LfuHybridStats {
		crate::lfu_hybrid_cache::LfuHybridStats {
			promotions: self.lfu_hybrid_promotions.load(Ordering::Relaxed),
			demotions: self.lfu_hybrid_demotions.load(Ordering::Relaxed),
			evictions: self.lfu_hybrid_evictions.load(Ordering::Relaxed),
			fast_bytes_used: self.lfu_hybrid_fast_bytes_used.load(Ordering::Relaxed),
			slow_bytes_used: self.lfu_hybrid_slow_bytes_used.load(Ordering::Relaxed),
			fast_objects: self.lfu_hybrid_fast_objects.load(Ordering::Relaxed),
			slow_objects: self.lfu_hybrid_slow_objects.load(Ordering::Relaxed),
		}
	}

	#[cfg(feature = "two_q_hybrid_cache")]
	pub fn record_two_q_hybrid_promotion(&self) {
		self.two_q_hybrid_promotions.fetch_add(1, Ordering::Relaxed);
	}

	#[cfg(feature = "two_q_hybrid_cache")]
	pub fn record_two_q_hybrid_demotion(&self) {
		self.two_q_hybrid_demotions.fetch_add(1, Ordering::Relaxed);
	}

	#[cfg(feature = "two_q_hybrid_cache")]
	pub fn record_two_q_hybrid_eviction(&self) {
		self.two_q_hybrid_evictions.fetch_add(1, Ordering::Relaxed);
	}

	/// Overwrites the live tier gauges (bytes/objects currently in each
	/// tier). Called by `PolicyWorker` each time it drains tier migrations.
	#[cfg(feature = "two_q_hybrid_cache")]
	pub fn set_two_q_hybrid_gauges(
		&self,
		fast_bytes_used: CacheSize,
		slow_bytes_used: CacheSize,
		fast_objects: u64,
		slow_objects: u64,
	) {
		self.two_q_hybrid_fast_bytes_used.store(fast_bytes_used, Ordering::Relaxed);
		self.two_q_hybrid_slow_bytes_used.store(slow_bytes_used, Ordering::Relaxed);
		self.two_q_hybrid_fast_objects.store(fast_objects, Ordering::Relaxed);
		self.two_q_hybrid_slow_objects.store(slow_objects, Ordering::Relaxed);
	}

	/// Returns a point-in-time snapshot of `two_q_hybrid_cache` statistics.
	#[cfg(feature = "two_q_hybrid_cache")]
	#[must_use]
	pub fn two_q_hybrid_stats(&self) -> crate::two_q_hybrid_cache::TwoQHybridStats {
		crate::two_q_hybrid_cache::TwoQHybridStats {
			promotions: self.two_q_hybrid_promotions.load(Ordering::Relaxed),
			demotions: self.two_q_hybrid_demotions.load(Ordering::Relaxed),
			evictions: self.two_q_hybrid_evictions.load(Ordering::Relaxed),
			fast_bytes_used: self.two_q_hybrid_fast_bytes_used.load(Ordering::Relaxed),
			slow_bytes_used: self.two_q_hybrid_slow_bytes_used.load(Ordering::Relaxed),
			fast_objects: self.two_q_hybrid_fast_objects.load(Ordering::Relaxed),
			slow_objects: self.two_q_hybrid_slow_objects.load(Ordering::Relaxed),
		}
	}

	// `record_fifo_hybrid_promotion` intentionally does not exist:
	// `FifoHybridStack` never emits a `Tier::Fast` migration (no promotion
	// policy at all — see that stack's module doc), so
	// `fifo_hybrid_promotions` is only ever read (always 0), never written.

	#[cfg(feature = "fifo_hybrid_cache")]
	pub fn record_fifo_hybrid_demotion(&self) {
		self.fifo_hybrid_demotions.fetch_add(1, Ordering::Relaxed);
	}

	#[cfg(feature = "fifo_hybrid_cache")]
	pub fn record_fifo_hybrid_eviction(&self) {
		self.fifo_hybrid_evictions.fetch_add(1, Ordering::Relaxed);
	}

	/// Overwrites the live tier gauges (bytes/objects currently in each
	/// tier). Called by `PolicyWorker` each time it drains tier migrations.
	#[cfg(feature = "fifo_hybrid_cache")]
	pub fn set_fifo_hybrid_gauges(
		&self,
		fast_bytes_used: CacheSize,
		slow_bytes_used: CacheSize,
		fast_objects: u64,
		slow_objects: u64,
	) {
		self.fifo_hybrid_fast_bytes_used.store(fast_bytes_used, Ordering::Relaxed);
		self.fifo_hybrid_slow_bytes_used.store(slow_bytes_used, Ordering::Relaxed);
		self.fifo_hybrid_fast_objects.store(fast_objects, Ordering::Relaxed);
		self.fifo_hybrid_slow_objects.store(slow_objects, Ordering::Relaxed);
	}

	/// Returns a point-in-time snapshot of `fifo_hybrid_cache` statistics.
	#[cfg(feature = "fifo_hybrid_cache")]
	#[must_use]
	pub fn fifo_hybrid_stats(&self) -> crate::fifo_hybrid_cache::FifoHybridStats {
		crate::fifo_hybrid_cache::FifoHybridStats {
			promotions: self.fifo_hybrid_promotions.load(Ordering::Relaxed),
			demotions: self.fifo_hybrid_demotions.load(Ordering::Relaxed),
			evictions: self.fifo_hybrid_evictions.load(Ordering::Relaxed),
			fast_bytes_used: self.fifo_hybrid_fast_bytes_used.load(Ordering::Relaxed),
			slow_bytes_used: self.fifo_hybrid_slow_bytes_used.load(Ordering::Relaxed),
			fast_objects: self.fifo_hybrid_fast_objects.load(Ordering::Relaxed),
			slow_objects: self.fifo_hybrid_slow_objects.load(Ordering::Relaxed),
		}
	}

	#[cfg(feature = "s3_fifo_hybrid_cache")]
	pub fn record_s3_fifo_hybrid_promotion(&self) {
		self.s3_fifo_hybrid_promotions.fetch_add(1, Ordering::Relaxed);
	}

	#[cfg(feature = "s3_fifo_hybrid_cache")]
	pub fn record_s3_fifo_hybrid_demotion(&self) {
		self.s3_fifo_hybrid_demotions.fetch_add(1, Ordering::Relaxed);
	}

	#[cfg(feature = "s3_fifo_hybrid_cache")]
	pub fn record_s3_fifo_hybrid_eviction(&self) {
		self.s3_fifo_hybrid_evictions.fetch_add(1, Ordering::Relaxed);
	}

	/// Overwrites the live tier gauges (bytes/objects currently in each
	/// tier). Called by `PolicyWorker` each time it drains tier migrations.
	#[cfg(feature = "s3_fifo_hybrid_cache")]
	pub fn set_s3_fifo_hybrid_gauges(
		&self,
		fast_bytes_used: CacheSize,
		slow_bytes_used: CacheSize,
		fast_objects: u64,
		slow_objects: u64,
	) {
		self.s3_fifo_hybrid_fast_bytes_used.store(fast_bytes_used, Ordering::Relaxed);
		self.s3_fifo_hybrid_slow_bytes_used.store(slow_bytes_used, Ordering::Relaxed);
		self.s3_fifo_hybrid_fast_objects.store(fast_objects, Ordering::Relaxed);
		self.s3_fifo_hybrid_slow_objects.store(slow_objects, Ordering::Relaxed);
	}

	/// Returns a point-in-time snapshot of `s3_fifo_hybrid_cache` statistics.
	#[cfg(feature = "s3_fifo_hybrid_cache")]
	#[must_use]
	pub fn s3_fifo_hybrid_stats(&self) -> crate::s3_fifo_hybrid_cache::S3FifoHybridStats {
		crate::s3_fifo_hybrid_cache::S3FifoHybridStats {
			promotions: self.s3_fifo_hybrid_promotions.load(Ordering::Relaxed),
			demotions: self.s3_fifo_hybrid_demotions.load(Ordering::Relaxed),
			evictions: self.s3_fifo_hybrid_evictions.load(Ordering::Relaxed),
			fast_bytes_used: self.s3_fifo_hybrid_fast_bytes_used.load(Ordering::Relaxed),
			slow_bytes_used: self.s3_fifo_hybrid_slow_bytes_used.load(Ordering::Relaxed),
			fast_objects: self.s3_fifo_hybrid_fast_objects.load(Ordering::Relaxed),
			slow_objects: self.s3_fifo_hybrid_slow_objects.load(Ordering::Relaxed),
		}
	}

	#[cfg(feature = "two_q_ghost_hybrid_cache")]
	pub fn record_two_q_ghost_hybrid_promotion(&self) {
		self.two_q_ghost_hybrid_promotions.fetch_add(1, Ordering::Relaxed);
	}

	#[cfg(feature = "two_q_ghost_hybrid_cache")]
	pub fn record_two_q_ghost_hybrid_demotion(&self) {
		self.two_q_ghost_hybrid_demotions.fetch_add(1, Ordering::Relaxed);
	}

	#[cfg(feature = "two_q_ghost_hybrid_cache")]
	pub fn record_two_q_ghost_hybrid_eviction(&self) {
		self.two_q_ghost_hybrid_evictions.fetch_add(1, Ordering::Relaxed);
	}

	/// Overwrites the live tier gauges (bytes/objects currently in each
	/// tier). Called by `PolicyWorker` each time it drains tier migrations.
	#[cfg(feature = "two_q_ghost_hybrid_cache")]
	pub fn set_two_q_ghost_hybrid_gauges(
		&self,
		fast_bytes_used: CacheSize,
		slow_bytes_used: CacheSize,
		fast_objects: u64,
		slow_objects: u64,
	) {
		self.two_q_ghost_hybrid_fast_bytes_used.store(fast_bytes_used, Ordering::Relaxed);
		self.two_q_ghost_hybrid_slow_bytes_used.store(slow_bytes_used, Ordering::Relaxed);
		self.two_q_ghost_hybrid_fast_objects.store(fast_objects, Ordering::Relaxed);
		self.two_q_ghost_hybrid_slow_objects.store(slow_objects, Ordering::Relaxed);
	}

	/// Returns a point-in-time snapshot of `two_q_ghost_hybrid_cache` statistics.
	#[cfg(feature = "two_q_ghost_hybrid_cache")]
	#[must_use]
	pub fn two_q_ghost_hybrid_stats(&self) -> crate::two_q_ghost_hybrid_cache::TwoQGhostHybridStats {
		crate::two_q_ghost_hybrid_cache::TwoQGhostHybridStats {
			promotions: self.two_q_ghost_hybrid_promotions.load(Ordering::Relaxed),
			demotions: self.two_q_ghost_hybrid_demotions.load(Ordering::Relaxed),
			evictions: self.two_q_ghost_hybrid_evictions.load(Ordering::Relaxed),
			fast_bytes_used: self.two_q_ghost_hybrid_fast_bytes_used.load(Ordering::Relaxed),
			slow_bytes_used: self.two_q_ghost_hybrid_slow_bytes_used.load(Ordering::Relaxed),
			fast_objects: self.two_q_ghost_hybrid_fast_objects.load(Ordering::Relaxed),
			slow_objects: self.two_q_ghost_hybrid_slow_objects.load(Ordering::Relaxed),
		}
	}

	#[cfg(feature = "s3_fifo_ghost_hybrid_cache")]
	pub fn record_s3_fifo_ghost_hybrid_promotion(&self) {
		self.s3_fifo_ghost_hybrid_promotions.fetch_add(1, Ordering::Relaxed);
	}

	#[cfg(feature = "s3_fifo_ghost_hybrid_cache")]
	pub fn record_s3_fifo_ghost_hybrid_demotion(&self) {
		self.s3_fifo_ghost_hybrid_demotions.fetch_add(1, Ordering::Relaxed);
	}

	#[cfg(feature = "s3_fifo_ghost_hybrid_cache")]
	pub fn record_s3_fifo_ghost_hybrid_eviction(&self) {
		self.s3_fifo_ghost_hybrid_evictions.fetch_add(1, Ordering::Relaxed);
	}

	/// Overwrites the live tier gauges (bytes/objects currently in each
	/// tier). Called by `PolicyWorker` each time it drains tier migrations.
	#[cfg(feature = "s3_fifo_ghost_hybrid_cache")]
	pub fn set_s3_fifo_ghost_hybrid_gauges(
		&self,
		fast_bytes_used: CacheSize,
		slow_bytes_used: CacheSize,
		fast_objects: u64,
		slow_objects: u64,
	) {
		self.s3_fifo_ghost_hybrid_fast_bytes_used.store(fast_bytes_used, Ordering::Relaxed);
		self.s3_fifo_ghost_hybrid_slow_bytes_used.store(slow_bytes_used, Ordering::Relaxed);
		self.s3_fifo_ghost_hybrid_fast_objects.store(fast_objects, Ordering::Relaxed);
		self.s3_fifo_ghost_hybrid_slow_objects.store(slow_objects, Ordering::Relaxed);
	}

	/// Returns a point-in-time snapshot of `s3_fifo_ghost_hybrid_cache` statistics.
	#[cfg(feature = "s3_fifo_ghost_hybrid_cache")]
	#[must_use]
	pub fn s3_fifo_ghost_hybrid_stats(&self) -> crate::s3_fifo_ghost_hybrid_cache::S3FifoGhostHybridStats {
		crate::s3_fifo_ghost_hybrid_cache::S3FifoGhostHybridStats {
			promotions: self.s3_fifo_ghost_hybrid_promotions.load(Ordering::Relaxed),
			demotions: self.s3_fifo_ghost_hybrid_demotions.load(Ordering::Relaxed),
			evictions: self.s3_fifo_ghost_hybrid_evictions.load(Ordering::Relaxed),
			fast_bytes_used: self.s3_fifo_ghost_hybrid_fast_bytes_used.load(Ordering::Relaxed),
			slow_bytes_used: self.s3_fifo_ghost_hybrid_slow_bytes_used.load(Ordering::Relaxed),
			fast_objects: self.s3_fifo_ghost_hybrid_fast_objects.load(Ordering::Relaxed),
			slow_objects: self.s3_fifo_ghost_hybrid_slow_objects.load(Ordering::Relaxed),
		}
	}

	#[cfg(feature = "s3_fifo_ghost_lazy_demotion_hybrid_cache")]
	pub fn record_s3_fifo_ghost_lazy_demotion_hybrid_promotion(&self) {
		self.s3_fifo_ghost_lazy_demotion_hybrid_promotions.fetch_add(1, Ordering::Relaxed);
	}

	#[cfg(feature = "s3_fifo_ghost_lazy_demotion_hybrid_cache")]
	pub fn record_s3_fifo_ghost_lazy_demotion_hybrid_demotion(&self) {
		self.s3_fifo_ghost_lazy_demotion_hybrid_demotions.fetch_add(1, Ordering::Relaxed);
	}

	#[cfg(feature = "s3_fifo_ghost_lazy_demotion_hybrid_cache")]
	pub fn record_s3_fifo_ghost_lazy_demotion_hybrid_eviction(&self) {
		self.s3_fifo_ghost_lazy_demotion_hybrid_evictions.fetch_add(1, Ordering::Relaxed);
	}

	/// Overwrites the live tier gauges (bytes/objects currently in each
	/// tier). Called by `PolicyWorker` each time it drains tier migrations.
	#[cfg(feature = "s3_fifo_ghost_lazy_demotion_hybrid_cache")]
	pub fn set_s3_fifo_ghost_lazy_demotion_hybrid_gauges(
		&self,
		fast_bytes_used: CacheSize,
		slow_bytes_used: CacheSize,
		fast_objects: u64,
		slow_objects: u64,
	) {
		self.s3_fifo_ghost_lazy_demotion_hybrid_fast_bytes_used.store(fast_bytes_used, Ordering::Relaxed);
		self.s3_fifo_ghost_lazy_demotion_hybrid_slow_bytes_used.store(slow_bytes_used, Ordering::Relaxed);
		self.s3_fifo_ghost_lazy_demotion_hybrid_fast_objects.store(fast_objects, Ordering::Relaxed);
		self.s3_fifo_ghost_lazy_demotion_hybrid_slow_objects.store(slow_objects, Ordering::Relaxed);
	}

	/// Returns a point-in-time snapshot of
	/// `s3_fifo_ghost_lazy_demotion_hybrid_cache` statistics.
	#[cfg(feature = "s3_fifo_ghost_lazy_demotion_hybrid_cache")]
	#[must_use]
	pub fn s3_fifo_ghost_lazy_demotion_hybrid_stats(&self) -> crate::s3_fifo_ghost_lazy_demotion_hybrid_cache::S3FifoGhostLazyDemotionHybridStats {
		crate::s3_fifo_ghost_lazy_demotion_hybrid_cache::S3FifoGhostLazyDemotionHybridStats {
			promotions: self.s3_fifo_ghost_lazy_demotion_hybrid_promotions.load(Ordering::Relaxed),
			demotions: self.s3_fifo_ghost_lazy_demotion_hybrid_demotions.load(Ordering::Relaxed),
			evictions: self.s3_fifo_ghost_lazy_demotion_hybrid_evictions.load(Ordering::Relaxed),
			fast_bytes_used: self.s3_fifo_ghost_lazy_demotion_hybrid_fast_bytes_used.load(Ordering::Relaxed),
			slow_bytes_used: self.s3_fifo_ghost_lazy_demotion_hybrid_slow_bytes_used.load(Ordering::Relaxed),
			fast_objects: self.s3_fifo_ghost_lazy_demotion_hybrid_fast_objects.load(Ordering::Relaxed),
			slow_objects: self.s3_fifo_ghost_lazy_demotion_hybrid_slow_objects.load(Ordering::Relaxed),
		}
	}

	#[cfg(feature = "s3_fifo_ghost_lazy_demotion_fast_admission_hybrid_cache")]
	pub fn record_s3_fifo_ghost_lazy_demotion_fast_admission_hybrid_promotion(&self) {
		self.s3_fifo_ghost_lazy_demotion_fast_admission_hybrid_promotions.fetch_add(1, Ordering::Relaxed);
	}

	#[cfg(feature = "s3_fifo_ghost_lazy_demotion_fast_admission_hybrid_cache")]
	pub fn record_s3_fifo_ghost_lazy_demotion_fast_admission_hybrid_demotion(&self) {
		self.s3_fifo_ghost_lazy_demotion_fast_admission_hybrid_demotions.fetch_add(1, Ordering::Relaxed);
	}

	#[cfg(feature = "s3_fifo_ghost_lazy_demotion_fast_admission_hybrid_cache")]
	pub fn record_s3_fifo_ghost_lazy_demotion_fast_admission_hybrid_eviction(&self) {
		self.s3_fifo_ghost_lazy_demotion_fast_admission_hybrid_evictions.fetch_add(1, Ordering::Relaxed);
	}

	/// Overwrites the live tier gauges (bytes/objects currently in each
	/// tier). Called by `PolicyWorker` each time it drains tier migrations.
	#[cfg(feature = "s3_fifo_ghost_lazy_demotion_fast_admission_hybrid_cache")]
	pub fn set_s3_fifo_ghost_lazy_demotion_fast_admission_hybrid_gauges(
		&self,
		fast_bytes_used: CacheSize,
		slow_bytes_used: CacheSize,
		fast_objects: u64,
		slow_objects: u64,
	) {
		self.s3_fifo_ghost_lazy_demotion_fast_admission_hybrid_fast_bytes_used.store(fast_bytes_used, Ordering::Relaxed);
		self.s3_fifo_ghost_lazy_demotion_fast_admission_hybrid_slow_bytes_used.store(slow_bytes_used, Ordering::Relaxed);
		self.s3_fifo_ghost_lazy_demotion_fast_admission_hybrid_fast_objects.store(fast_objects, Ordering::Relaxed);
		self.s3_fifo_ghost_lazy_demotion_fast_admission_hybrid_slow_objects.store(slow_objects, Ordering::Relaxed);
	}

	/// Returns a point-in-time snapshot of
	/// `s3_fifo_ghost_lazy_demotion_fast_admission_hybrid_cache` statistics.
	#[cfg(feature = "s3_fifo_ghost_lazy_demotion_fast_admission_hybrid_cache")]
	#[must_use]
	pub fn s3_fifo_ghost_lazy_demotion_fast_admission_hybrid_stats(&self) -> crate::s3_fifo_ghost_lazy_demotion_fast_admission_hybrid_cache::S3FifoGhostLazyDemotionFastAdmissionHybridStats {
		crate::s3_fifo_ghost_lazy_demotion_fast_admission_hybrid_cache::S3FifoGhostLazyDemotionFastAdmissionHybridStats {
			promotions: self.s3_fifo_ghost_lazy_demotion_fast_admission_hybrid_promotions.load(Ordering::Relaxed),
			demotions: self.s3_fifo_ghost_lazy_demotion_fast_admission_hybrid_demotions.load(Ordering::Relaxed),
			evictions: self.s3_fifo_ghost_lazy_demotion_fast_admission_hybrid_evictions.load(Ordering::Relaxed),
			fast_bytes_used: self.s3_fifo_ghost_lazy_demotion_fast_admission_hybrid_fast_bytes_used.load(Ordering::Relaxed),
			slow_bytes_used: self.s3_fifo_ghost_lazy_demotion_fast_admission_hybrid_slow_bytes_used.load(Ordering::Relaxed),
			fast_objects: self.s3_fifo_ghost_lazy_demotion_fast_admission_hybrid_fast_objects.load(Ordering::Relaxed),
			slow_objects: self.s3_fifo_ghost_lazy_demotion_fast_admission_hybrid_slow_objects.load(Ordering::Relaxed),
		}
	}

	#[cfg(feature = "s3_fifo_ghost_lazy_demotion_fast_admission_midpoint_hybrid_cache")]
	pub fn record_s3_fifo_ghost_lazy_demotion_fast_admission_midpoint_hybrid_promotion(&self) {
		self.s3_fifo_ghost_lazy_demotion_fast_admission_midpoint_hybrid_promotions.fetch_add(1, Ordering::Relaxed);
	}

	#[cfg(feature = "s3_fifo_ghost_lazy_demotion_fast_admission_midpoint_hybrid_cache")]
	pub fn record_s3_fifo_ghost_lazy_demotion_fast_admission_midpoint_hybrid_demotion(&self) {
		self.s3_fifo_ghost_lazy_demotion_fast_admission_midpoint_hybrid_demotions.fetch_add(1, Ordering::Relaxed);
	}

	#[cfg(feature = "s3_fifo_ghost_lazy_demotion_fast_admission_midpoint_hybrid_cache")]
	pub fn record_s3_fifo_ghost_lazy_demotion_fast_admission_midpoint_hybrid_eviction(&self) {
		self.s3_fifo_ghost_lazy_demotion_fast_admission_midpoint_hybrid_evictions.fetch_add(1, Ordering::Relaxed);
	}

	/// Overwrites the live tier gauges (bytes/objects currently in each
	/// tier). Called by `PolicyWorker` each time it drains tier migrations.
	#[cfg(feature = "s3_fifo_ghost_lazy_demotion_fast_admission_midpoint_hybrid_cache")]
	pub fn set_s3_fifo_ghost_lazy_demotion_fast_admission_midpoint_hybrid_gauges(
		&self,
		fast_bytes_used: CacheSize,
		slow_bytes_used: CacheSize,
		fast_objects: u64,
		slow_objects: u64,
	) {
		self.s3_fifo_ghost_lazy_demotion_fast_admission_midpoint_hybrid_fast_bytes_used.store(fast_bytes_used, Ordering::Relaxed);
		self.s3_fifo_ghost_lazy_demotion_fast_admission_midpoint_hybrid_slow_bytes_used.store(slow_bytes_used, Ordering::Relaxed);
		self.s3_fifo_ghost_lazy_demotion_fast_admission_midpoint_hybrid_fast_objects.store(fast_objects, Ordering::Relaxed);
		self.s3_fifo_ghost_lazy_demotion_fast_admission_midpoint_hybrid_slow_objects.store(slow_objects, Ordering::Relaxed);
	}

	/// Returns a point-in-time snapshot of
	/// `s3_fifo_ghost_lazy_demotion_fast_admission_midpoint_hybrid_cache`
	/// statistics.
	#[cfg(feature = "s3_fifo_ghost_lazy_demotion_fast_admission_midpoint_hybrid_cache")]
	#[must_use]
	pub fn s3_fifo_ghost_lazy_demotion_fast_admission_midpoint_hybrid_stats(&self) -> crate::s3_fifo_ghost_lazy_demotion_fast_admission_midpoint_hybrid_cache::S3FifoGhostLazyDemotionFastAdmissionMidpointHybridStats {
		crate::s3_fifo_ghost_lazy_demotion_fast_admission_midpoint_hybrid_cache::S3FifoGhostLazyDemotionFastAdmissionMidpointHybridStats {
			promotions: self.s3_fifo_ghost_lazy_demotion_fast_admission_midpoint_hybrid_promotions.load(Ordering::Relaxed),
			demotions: self.s3_fifo_ghost_lazy_demotion_fast_admission_midpoint_hybrid_demotions.load(Ordering::Relaxed),
			evictions: self.s3_fifo_ghost_lazy_demotion_fast_admission_midpoint_hybrid_evictions.load(Ordering::Relaxed),
			fast_bytes_used: self.s3_fifo_ghost_lazy_demotion_fast_admission_midpoint_hybrid_fast_bytes_used.load(Ordering::Relaxed),
			slow_bytes_used: self.s3_fifo_ghost_lazy_demotion_fast_admission_midpoint_hybrid_slow_bytes_used.load(Ordering::Relaxed),
			fast_objects: self.s3_fifo_ghost_lazy_demotion_fast_admission_midpoint_hybrid_fast_objects.load(Ordering::Relaxed),
			slow_objects: self.s3_fifo_ghost_lazy_demotion_fast_admission_midpoint_hybrid_slow_objects.load(Ordering::Relaxed),
		}
	}

	#[cfg(feature = "lru_sized_hybrid_cache")]
	pub fn record_lru_sized_hybrid_promotion(&self) {
		self.lru_sized_hybrid_promotions.fetch_add(1, Ordering::Relaxed);
	}

	#[cfg(feature = "lru_sized_hybrid_cache")]
	pub fn record_lru_sized_hybrid_demotion(&self) {
		self.lru_sized_hybrid_demotions.fetch_add(1, Ordering::Relaxed);
	}

	#[cfg(feature = "lru_sized_hybrid_cache")]
	pub fn record_lru_sized_hybrid_eviction(&self) {
		self.lru_sized_hybrid_evictions.fetch_add(1, Ordering::Relaxed);
	}

	/// Overwrites every live tier gauge (bytes/objects currently in each of
	/// the four segments, plus the small+large combined fast/slow totals
	/// derived from them). Called by `PolicyWorker` each time it drains tier
	/// migrations.
	#[cfg(feature = "lru_sized_hybrid_cache")]
	pub fn set_lru_sized_hybrid_gauges(
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
		self.lru_sized_hybrid_small_fast_bytes_used.store(small_fast_bytes_used, Ordering::Relaxed);
		self.lru_sized_hybrid_large_fast_bytes_used.store(large_fast_bytes_used, Ordering::Relaxed);
		self.lru_sized_hybrid_small_slow_bytes_used.store(small_slow_bytes_used, Ordering::Relaxed);
		self.lru_sized_hybrid_large_slow_bytes_used.store(large_slow_bytes_used, Ordering::Relaxed);
		self.lru_sized_hybrid_small_fast_objects.store(small_fast_objects, Ordering::Relaxed);
		self.lru_sized_hybrid_large_fast_objects.store(large_fast_objects, Ordering::Relaxed);
		self.lru_sized_hybrid_small_slow_objects.store(small_slow_objects, Ordering::Relaxed);
		self.lru_sized_hybrid_large_slow_objects.store(large_slow_objects, Ordering::Relaxed);

		self.lru_sized_hybrid_fast_bytes_used.store(small_fast_bytes_used + large_fast_bytes_used, Ordering::Relaxed);
		self.lru_sized_hybrid_slow_bytes_used.store(small_slow_bytes_used + large_slow_bytes_used, Ordering::Relaxed);
		self.lru_sized_hybrid_fast_objects.store(small_fast_objects + large_fast_objects, Ordering::Relaxed);
		self.lru_sized_hybrid_slow_objects.store(small_slow_objects + large_slow_objects, Ordering::Relaxed);
	}

	/// Returns a point-in-time snapshot of `lru_sized_hybrid_cache` statistics.
	#[cfg(feature = "lru_sized_hybrid_cache")]
	#[must_use]
	pub fn lru_sized_hybrid_stats(&self) -> crate::lru_sized_hybrid_cache::LruSizedHybridStats {
		crate::lru_sized_hybrid_cache::LruSizedHybridStats {
			promotions: self.lru_sized_hybrid_promotions.load(Ordering::Relaxed),
			demotions: self.lru_sized_hybrid_demotions.load(Ordering::Relaxed),
			evictions: self.lru_sized_hybrid_evictions.load(Ordering::Relaxed),
			fast_bytes_used: self.lru_sized_hybrid_fast_bytes_used.load(Ordering::Relaxed),
			slow_bytes_used: self.lru_sized_hybrid_slow_bytes_used.load(Ordering::Relaxed),
			fast_objects: self.lru_sized_hybrid_fast_objects.load(Ordering::Relaxed),
			slow_objects: self.lru_sized_hybrid_slow_objects.load(Ordering::Relaxed),
			small_fast_bytes_used: self.lru_sized_hybrid_small_fast_bytes_used.load(Ordering::Relaxed),
			large_fast_bytes_used: self.lru_sized_hybrid_large_fast_bytes_used.load(Ordering::Relaxed),
			small_fast_objects: self.lru_sized_hybrid_small_fast_objects.load(Ordering::Relaxed),
			large_fast_objects: self.lru_sized_hybrid_large_fast_objects.load(Ordering::Relaxed),
			small_slow_bytes_used: self.lru_sized_hybrid_small_slow_bytes_used.load(Ordering::Relaxed),
			large_slow_bytes_used: self.lru_sized_hybrid_large_slow_bytes_used.load(Ordering::Relaxed),
			small_slow_objects: self.lru_sized_hybrid_small_slow_objects.load(Ordering::Relaxed),
			large_slow_objects: self.lru_sized_hybrid_large_slow_objects.load(Ordering::Relaxed),
		}
	}

	/// Current LARGE fast segment's byte budget (the SMALL segment reuses
	/// `fast_tier_capacity` above).
	#[cfg(feature = "lru_sized_hybrid_cache")]
	#[must_use]
	pub fn lru_sized_hybrid_large_fast_capacity(&self) -> CacheSize {
		self.lru_sized_hybrid_large_fast_capacity.load(Ordering::Relaxed)
	}

	/// Sets the LARGE fast segment's byte budget. Callers are responsible
	/// for also broadcasting `WorkerEvent::ResizeLargeFastTier`.
	#[cfg(feature = "lru_sized_hybrid_cache")]
	pub fn set_lru_sized_hybrid_large_fast_capacity(&self, size: CacheSize) {
		self.lru_sized_hybrid_large_fast_capacity.store(size, Ordering::Relaxed);
	}

	/// Current small/large size-classification threshold, in bytes.
	#[cfg(feature = "lru_sized_hybrid_cache")]
	#[must_use]
	pub fn lru_sized_hybrid_size_threshold(&self) -> CacheSize {
		self.lru_sized_hybrid_size_threshold.load(Ordering::Relaxed)
	}

	/// Sets the small/large size-classification threshold. Callers are
	/// responsible for also broadcasting `WorkerEvent::ResizeSizeThreshold`.
	#[cfg(feature = "lru_sized_hybrid_cache")]
	pub fn set_lru_sized_hybrid_size_threshold(&self, size: CacheSize) {
		self.lru_sized_hybrid_size_threshold.store(size, Ordering::Relaxed);
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
		#[cfg(feature = "lfu_hybrid_cache")]
		self.lfu_hybrid_admission_latched.store(false, Ordering::Relaxed);
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
