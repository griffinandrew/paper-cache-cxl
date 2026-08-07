/*
 * Copyright (c) Kia Shakiba
 *
 * This source code is licensed under the GNU AGPLv3 license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! Single-instance, segmented-S3-FIFO hybrid cache with a demotion-time
//! reference-bit gate, a fast-tier-resident one-access queue, a
//! mid-slow-segment reference-bit checkpoint, and no ghost queue.
//!
//! Identical architecture and admission/eviction rules to
//! `s3_fifo_ghost_lazy_demotion_fast_admission_midpoint_hybrid_cache` — see
//! that module's docs for the full design — with two changes: there is no
//! ghost queue (removed entirely, since nothing ever populates it here),
//! and a one-access-queue key that ages out without a second access is
//! spliced directly into the slow tier of the main queue instead of being
//! evicted. See
//! `worker::policy::policy_stack::s3_fifo_lazy_demotion_fast_admission_midpoint_reprieve_hybrid_stack`'s
//! module doc for the full mechanics, including the O(number of currently-
//! fast keys) splice technique used to place the reprieved key exactly
//! adjacent to the fast/slow boundary without ever tagging it Fast.

mod stats;

pub use crate::tiered_buffer::TieredBuffer;
pub use stats::S3FifoLazyDemotionFastAdmissionMidpointReprieveHybridStats;

/// Marker type selecting
/// `s3_fifo_lazy_demotion_fast_admission_midpoint_reprieve_hybrid_cache`'s
/// behavior for the shared generic `impl<K, S> PaperCache<K, TieredBuffer,
/// S>` block in `lib.rs` (see `crate::hybrid_policy::HybridPolicy`).
/// Admission is unconditional to the *fast* tier, same reasoning as
/// `S3FifoGhostLazyDemotionFastAdmissionMidpointHybridPolicy` -- the
/// reprieve mechanic only changes what happens to a one-access-queue key
/// that ages out, not admission itself. `ExtraConfig = f64` carries the
/// one-access queue's byte-budget ratio, same as that policy.
pub struct S3FifoLazyDemotionFastAdmissionMidpointReprieveHybridPolicy;

impl crate::hybrid_policy::HybridPolicy for S3FifoLazyDemotionFastAdmissionMidpointReprieveHybridPolicy {
	type Stats = S3FifoLazyDemotionFastAdmissionMidpointReprieveHybridStats;
	type ExtraConfig = f64;

	fn seed_policy(ratio: f64) -> crate::PaperPolicy {
		crate::PaperPolicy::S3FifoLazyDemotionFastAdmissionMidpointReprieveHybrid(ratio)
	}

	fn stats_from_status(status: &crate::status::AtomicStatus) -> S3FifoLazyDemotionFastAdmissionMidpointReprieveHybridStats {
		status.s3_fifo_lazy_demotion_fast_admission_midpoint_reprieve_hybrid_stats()
	}

	fn admission_tier<K>(
		_hashed_key: crate::HashedKey,
		_status: &crate::status::AtomicStatus,
		_objects: &crate::hybrid_policy::HybridObjectMap<K>,
	) -> crate::Tier {
		crate::Tier::Fast
	}
}
