/*
 * Copyright (c) Kia Shakiba
 *
 * This source code is licensed under the GNU AGPLv3 license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! Statistics for `s3_fifo_ghost_hybrid_cache`.
//!
//! Same rationale as `s3_fifo_hybrid_cache::S3FifoHybridStats`: the
//! counters backing this snapshot live directly on `AtomicStatus`
//! (`status.rs`) rather than in a separate atomics struct here.

/// A point-in-time snapshot of `s3_fifo_ghost_hybrid_cache` statistics.
///
/// Assembled by `AtomicStatus::s3_fifo_ghost_hybrid_stats` and returned
/// from `PaperCache::s3_fifo_ghost_hybrid_stats`. Same shape as
/// `s3_fifo_hybrid_cache::S3FifoHybridStats` — a ghost-queue hit still
/// shows up as an ordinary `promotions` increment.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct S3FifoGhostHybridStats {
	/// Objects moved from the one-access queue (eager, on re-access or a
	/// ghost-queue hit) or given a second chance out of the main queue's
	/// tail (lazy, at eviction time) into the main queue's fast portion.
	pub promotions: u64,
	/// Objects moved from the main queue's fast portion to its slow portion
	/// (fast tier pressure) — unconditional aging, independent of the
	/// reference bit.
	pub demotions: u64,
	/// Objects permanently removed — either the one-access queue's tail
	/// aging out without a second access, or the main queue's tail once its
	/// reference bit is found clear during an eviction sweep.
	pub evictions: u64,
	/// Current bytes accounted to the fast tier (live gauge).
	pub fast_bytes_used: u64,
	/// Current bytes accounted to the slow tier — one-access queue and main
	/// queue's slow portion combined (live gauge).
	pub slow_bytes_used: u64,
	/// Current number of objects in the fast tier (live gauge).
	pub fast_objects: u64,
	/// Current number of objects in the slow tier (live gauge).
	pub slow_objects: u64,
}
