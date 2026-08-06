/*
 * Copyright (c) Kia Shakiba
 *
 * This source code is licensed under the GNU AGPLv3 license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! Statistics for
//! `s3_fifo_ghost_lazy_demotion_fast_admission_midpoint_hybrid_cache`.
//!
//! Same rationale as
//! `s3_fifo_ghost_lazy_demotion_fast_admission_hybrid_cache::S3FifoGhostLazyDemotionFastAdmissionHybridStats`:
//! the counters backing this snapshot live directly on `AtomicStatus`
//! (`status.rs`) rather than in a separate atomics struct here.

/// A point-in-time snapshot of
/// `s3_fifo_ghost_lazy_demotion_fast_admission_midpoint_hybrid_cache`
/// statistics.
///
/// Same shape as
/// `s3_fifo_ghost_lazy_demotion_fast_admission_hybrid_cache::S3FifoGhostLazyDemotionFastAdmissionHybridStats`
/// -- `promotions` now covers BOTH the eviction-time tail second chance and
/// the new mid-segment checkpoint promotion (both go through the same
/// `give_second_chance`, and both move genuinely-Slow bytes back to Fast,
/// so they're not distinguished in this counter).
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct S3FifoGhostLazyDemotionFastAdmissionMidpointHybridStats {
	/// Objects given a second chance -- either at the main queue's slow
	/// tail during an eviction sweep, or earlier, at the mid-slow-segment
	/// checkpoint this variant adds. Both are real Slow→Fast physical
	/// migrations, counted uniformly.
	pub promotions: u64,
	/// Objects moved from the main queue's fast portion to its slow portion
	/// (fast tier pressure, checked against the shared-budget effective
	/// capacity) — reference-bit gated: only a key found with its bit clear
	/// at the demotion boundary is actually demoted.
	pub demotions: u64,
	/// Objects permanently removed — either the one-access queue's tail
	/// aging out without a second access, or the main queue's tail once its
	/// reference bit is found clear during an eviction sweep.
	pub evictions: u64,
	/// Current bytes accounted to the fast tier (live gauge) — main queue's
	/// fast portion PLUS the one-access queue.
	pub fast_bytes_used: u64,
	/// Current bytes accounted to the slow tier (live gauge) — main queue's
	/// slow portion only; the one-access queue no longer touches slow.
	pub slow_bytes_used: u64,
	/// Current number of objects in the fast tier (live gauge) — main
	/// queue's fast portion PLUS the one-access queue.
	pub fast_objects: u64,
	/// Current number of objects in the slow tier (live gauge) — main
	/// queue's slow portion only.
	pub slow_objects: u64,
}
