/*
 * Copyright (c) Kia Shakiba
 *
 * This source code is licensed under the GNU AGPLv3 license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! Statistics for
//! `s3_fifo_lazy_demotion_fast_admission_reprieve_hybrid_cache`.
//!
//! Same rationale as
//! `s3_fifo_ghost_lazy_demotion_fast_admission_midpoint_hybrid_cache::S3FifoGhostLazyDemotionFastAdmissionMidpointHybridStats`:
//! the counters backing this snapshot live directly on `AtomicStatus`
//! (`status.rs`) rather than in a separate atomics struct here.

/// A point-in-time snapshot of
/// `s3_fifo_lazy_demotion_fast_admission_reprieve_hybrid_cache`
/// statistics.
///
/// Same shape as
/// `s3_fifo_ghost_lazy_demotion_fast_admission_midpoint_hybrid_cache::S3FifoGhostLazyDemotionFastAdmissionMidpointHybridStats`,
/// minus the ghost queue this variant removes -- `evictions` now only ever
/// counts real main-queue tail removals, since a one-access-queue key that
/// ages out is spliced into the slow tier instead of being evicted (see the
/// module doc).
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct S3FifoLazyDemotionFastAdmissionReprieveHybridStats {
	/// Objects given a second chance -- either at the main queue's slow
	/// tail during an eviction sweep, or earlier, at the mid-slow-segment
	/// checkpoint. Also counts a one-access-queue key's initial splice into
	/// the slow tier (a real Fast(DRAM)→Slow(PMEM) migration, recorded as a
	/// demotion, not a promotion -- see `demotions` below) as distinct from
	/// its later re-promotion back to fast, which IS counted here.
	pub promotions: u64,
	/// Objects moved from the main queue's fast portion to its slow portion
	/// -- either genuine fast/slow-boundary demotion pressure (reference-bit
	/// gated, same as the predecessor variant), or a one-access-queue key
	/// being spliced into the slow tier once its own queue's capacity is
	/// exceeded (see `settle_one_access` in the policy stack).
	pub demotions: u64,
	/// Objects permanently removed -- only ever the main queue's tail, once
	/// its reference bit is found clear during an eviction sweep. Unlike
	/// the predecessor variant, a one-access-queue key aging out is never
	/// counted here, since it's reprieved into the slow tier instead of
	/// being evicted.
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
