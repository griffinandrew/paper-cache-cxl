/*
 * Copyright (c) Kia Shakiba
 *
 * This source code is licensed under the GNU AGPLv3 license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! Single-instance, segmented-S3-FIFO hybrid cache with a ghost queue, a
//! demotion-time reference-bit gate, AND a fast-tier-resident one-access
//! queue.
//!
//! Identical architecture and eviction/promotion rules to
//! `s3_fifo_ghost_lazy_demotion_hybrid_cache` — see that module's docs for
//! the full design (the ghost queue's lifecycle, the "contiguous front run"
//! invariant, the demotion-time reprieve) — plus one change: the one-access
//! queue now lives in the FAST tier instead of the slow tier, so admission
//! is a cheap DRAM write instead of a synchronous PMEM write. See
//! `worker::policy::policy_stack::s3_fifo_ghost_lazy_demotion_fast_admission_hybrid_stack`'s
//! module doc for the full mechanics, including how the one-access queue's
//! byte budget is now reserved out of the fast-tier size rather than being
//! an independent PMEM-side budget.

mod stats;

pub use crate::tiered_buffer::TieredBuffer;
pub use stats::S3FifoGhostLazyDemotionFastAdmissionHybridStats;

/// Marker type selecting
/// `s3_fifo_ghost_lazy_demotion_fast_admission_hybrid_cache`'s behavior for
/// the shared generic `impl<K, S> PaperCache<K, TieredBuffer, S>` block in
/// `lib.rs` (see `crate::hybrid_policy::HybridPolicy`). Admission is
/// unconditional to the *fast* tier now -- both a brand-new one-access
/// admission and a ghost-hit re-admission need Fast bytes, so there's no
/// case that needs Slow at admission time in this variant, unlike
/// `S3FifoGhostLazyDemotionHybridPolicy`. `ExtraConfig = f64` carries the
/// one-access queue's byte-budget ratio, same as that policy.
pub struct S3FifoGhostLazyDemotionFastAdmissionHybridPolicy;

impl crate::hybrid_policy::HybridPolicy for S3FifoGhostLazyDemotionFastAdmissionHybridPolicy {
	type Stats = S3FifoGhostLazyDemotionFastAdmissionHybridStats;
	type ExtraConfig = f64;

	fn seed_policy(ratio: f64) -> crate::PaperPolicy {
		crate::PaperPolicy::S3FifoGhostLazyDemotionFastAdmissionHybrid(ratio)
	}

	fn stats_from_status(status: &crate::status::AtomicStatus) -> S3FifoGhostLazyDemotionFastAdmissionHybridStats {
		status.s3_fifo_ghost_lazy_demotion_fast_admission_hybrid_stats()
	}

	fn admission_tier<K>(
		_hashed_key: crate::HashedKey,
		_status: &crate::status::AtomicStatus,
		_objects: &crate::hybrid_policy::HybridObjectMap<K>,
	) -> crate::Tier {
		crate::Tier::Fast
	}
}
