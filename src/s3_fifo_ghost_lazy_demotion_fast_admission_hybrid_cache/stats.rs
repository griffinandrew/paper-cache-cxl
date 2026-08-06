/*
 * Copyright (c) Kia Shakiba
 *
 * This source code is licensed under the GNU AGPLv3 license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! Statistics for `s3_fifo_ghost_lazy_demotion_fast_admission_hybrid_cache`.
//!
//! Same rationale as
//! `s3_fifo_ghost_lazy_demotion_hybrid_cache::S3FifoGhostLazyDemotionHybridStats`:
//! the counters backing this snapshot live directly on `AtomicStatus`
//! (`status.rs`) rather than in a separate atomics struct here.

/// A point-in-time snapshot of
/// `s3_fifo_ghost_lazy_demotion_fast_admission_hybrid_cache` statistics.
///
/// Assembled by
/// `AtomicStatus::s3_fifo_ghost_lazy_demotion_fast_admission_hybrid_stats`
/// and returned from
/// `PaperCache::s3_fifo_ghost_lazy_demotion_fast_admission_hybrid_stats`.
/// `fast_bytes_used`/`fast_objects` now include the one-access queue (it's
/// DRAM-resident in this variant); `slow_bytes_used`/`slow_objects` cover
/// only the main queue's slow portion. `promotions` no longer increments
/// for a one-access → main promotion or a ghost-hit re-admission that stays
/// fast (their bytes were already Fast, so no physical migration -- and
/// hence no counted promotion -- happens); it still increments for a real
/// eviction-time second chance, which does move genuinely-Slow bytes back
/// to Fast.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct S3FifoGhostLazyDemotionFastAdmissionHybridStats {
	/// Objects given a second chance out of the main queue's slow tail at
	/// eviction time (a real Slow→Fast physical migration) — the only
	/// promotion this variant's migrations still record. See the struct
	/// doc for why one-access promotions and ghost hits no longer count
	/// here.
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
