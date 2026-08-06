/*
 * Copyright (c) Kia Shakiba
 *
 * This source code is licensed under the GNU AGPLv3 license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! Statistics for `s3_fifo_ghost_lazy_demotion_hybrid_cache`.
//!
//! Same rationale as `s3_fifo_ghost_hybrid_cache::S3FifoGhostHybridStats`:
//! the counters backing this snapshot live directly on `AtomicStatus`
//! (`status.rs`) rather than in a separate atomics struct here.

/// A point-in-time snapshot of `s3_fifo_ghost_lazy_demotion_hybrid_cache`
/// statistics.
///
/// Assembled by `AtomicStatus::s3_fifo_ghost_lazy_demotion_hybrid_stats` and
/// returned from `PaperCache::s3_fifo_ghost_lazy_demotion_hybrid_stats`.
/// Same shape as `s3_fifo_ghost_hybrid_cache::S3FifoGhostHybridStats` — a
/// demotion-time reprieve (see the stack's module doc) is not itself
/// counted anywhere; only a genuine tier change increments `demotions` or
/// `promotions`.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct S3FifoGhostLazyDemotionHybridStats {
	/// Objects moved from the one-access queue (eager, on re-access or a
	/// ghost-queue hit), given a second chance out of the main queue's tail
	/// (lazy, at eviction time), or reprieved back to the front by the
	/// demotion-time reference-bit check into the main queue's fast
	/// portion.
	pub promotions: u64,
	/// Objects moved from the main queue's fast portion to its slow portion
	/// (fast tier pressure) — now reference-bit gated: only a key found
	/// with its bit clear at the demotion boundary is actually demoted.
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
