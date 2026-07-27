/*
 * Copyright (c) Kia Shakiba
 *
 * This source code is licensed under the GNU AGPLv3 license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! Statistics for `lru_sized_hybrid_cache`.
//!
//! Same rationale as `lru_hybrid_cache::stats`: the counters backing this
//! snapshot live directly on `AtomicStatus` (`status.rs`) rather than in a
//! separate atomics struct here, since that's already the one shared,
//! per-cache structure both `PaperCache` and `PolicyWorker` hold.

/// A point-in-time snapshot of `lru_sized_hybrid_cache` statistics.
///
/// Assembled by `AtomicStatus::lru_sized_hybrid_stats` and returned from
/// `PaperCache::lru_sized_hybrid_stats`.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct LruSizedHybridStats {
	/// Objects moved from either slow list to the matching fast segment
	/// (accessed while slow).
	pub promotions: u64,
	/// Objects moved from either fast segment to its matching slow list
	/// (that segment's own fast-tier pressure).
	pub demotions: u64,
	/// Objects permanently removed from either slow list (cache capacity
	/// exhausted).
	pub evictions: u64,
	/// Current bytes accounted to the fast tier -- both segments combined
	/// (live gauge).
	pub fast_bytes_used: u64,
	/// Current bytes accounted to the slow tier -- both lists combined
	/// (live gauge).
	pub slow_bytes_used: u64,
	/// Current number of objects in the fast tier -- both segments combined
	/// (live gauge).
	pub fast_objects: u64,
	/// Current number of objects in the slow tier -- both lists combined
	/// (live gauge).
	pub slow_objects: u64,
	/// Current bytes accounted to the SMALL fast segment specifically (live
	/// gauge).
	pub small_fast_bytes_used: u64,
	/// Current bytes accounted to the LARGE fast segment specifically (live
	/// gauge).
	pub large_fast_bytes_used: u64,
	/// Current number of objects in the SMALL fast segment specifically
	/// (live gauge).
	pub small_fast_objects: u64,
	/// Current number of objects in the LARGE fast segment specifically
	/// (live gauge).
	pub large_fast_objects: u64,
	/// Current bytes accounted to the SMALL slow list specifically (live
	/// gauge).
	pub small_slow_bytes_used: u64,
	/// Current bytes accounted to the LARGE slow list specifically (live
	/// gauge).
	pub large_slow_bytes_used: u64,
	/// Current number of objects in the SMALL slow list specifically (live
	/// gauge).
	pub small_slow_objects: u64,
	/// Current number of objects in the LARGE slow list specifically (live
	/// gauge).
	pub large_slow_objects: u64,
}
