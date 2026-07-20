/*
 * Copyright (c) Kia Shakiba
 *
 * This source code is licensed under the GNU AGPLv3 license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! Statistics for `fifo_hybrid_cache`.
//!
//! The counters backing this snapshot live directly on `AtomicStatus`
//! (`status.rs`) rather than in a separate atomics struct here — same
//! rationale as `lru_hybrid_cache::stats`: `AtomicStatus` is already the one
//! shared, per-cache structure both `PaperCache` and `PolicyWorker` hold, so
//! no new field is needed on `PaperCache` itself.

/// A point-in-time snapshot of `fifo_hybrid_cache` statistics.
///
/// Assembled by `AtomicStatus::fifo_hybrid_stats` and returned from
/// `PaperCache::fifo_hybrid_stats`.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct FifoHybridStats {
	/// Always `0` — FIFO has no promotion policy at all (objects are never
	/// reordered after insertion, so nothing is ever moved from the slow
	/// tier back to the fast tier). Kept for API-shape symmetry with
	/// `LruHybridStats`/`LfuHybridStats`/`TwoQHybridStats`.
	pub promotions: u64,
	/// Objects moved from the fast tier to the slow tier (fast tier pressure).
	pub demotions: u64,
	/// Objects permanently removed from the slow tier (cache capacity exhausted).
	pub evictions: u64,
	/// Current bytes accounted to the fast tier (live gauge).
	pub fast_bytes_used: u64,
	/// Current bytes accounted to the slow tier (live gauge).
	pub slow_bytes_used: u64,
	/// Current number of objects in the fast tier (live gauge).
	pub fast_objects: u64,
	/// Current number of objects in the slow tier (live gauge).
	pub slow_objects: u64,
}
