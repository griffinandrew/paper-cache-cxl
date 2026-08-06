/*
 * Copyright (c) Kia Shakiba
 *
 * This source code is licensed under the GNU AGPLv3 license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! Single-instance, segmented-S3-FIFO hybrid cache with a ghost queue AND a
//! demotion-time reference-bit gate.
//!
//! Identical architecture and admission/promotion/eviction rules to
//! `s3_fifo_ghost_hybrid_cache` — see that module's docs for the full design
//! (including the bare-key ghost queue's lifecycle and the "contiguous
//! front run" invariant) — plus one change: demotion (moving a fast
//! main-queue key to the slow tier under fast-tier pressure) is now
//! reference-bit gated too, not just eviction. See
//! `worker::policy::policy_stack::s3_fifo_ghost_lazy_demotion_hybrid_stack`'s
//! module doc for the full mechanics ("lazy demotion, lazy promotion").

mod stats;

pub use crate::tiered_buffer::TieredBuffer;
pub use stats::S3FifoGhostLazyDemotionHybridStats;

/// Marker type selecting `s3_fifo_ghost_lazy_demotion_hybrid_cache`'s
/// behavior for the shared generic `impl<K, S> PaperCache<K, TieredBuffer,
/// S>` block in `lib.rs` (see `crate::hybrid_policy::HybridPolicy`).
/// Admission is unconditional to the *slow* tier regardless of ghost
/// history — same reasoning as `S3FifoGhostHybridPolicy`. `ExtraConfig =
/// f64` carries the one-access queue's byte-budget ratio, same as
/// `S3FifoGhostHybridPolicy`.
pub struct S3FifoGhostLazyDemotionHybridPolicy;

impl crate::hybrid_policy::HybridPolicy for S3FifoGhostLazyDemotionHybridPolicy {
	type Stats = S3FifoGhostLazyDemotionHybridStats;
	type ExtraConfig = f64;

	fn seed_policy(ratio: f64) -> crate::PaperPolicy {
		crate::PaperPolicy::S3FifoGhostLazyDemotionHybrid(ratio)
	}

	fn stats_from_status(status: &crate::status::AtomicStatus) -> S3FifoGhostLazyDemotionHybridStats {
		status.s3_fifo_ghost_lazy_demotion_hybrid_stats()
	}

	fn admission_tier<K>(
		_hashed_key: crate::HashedKey,
		_status: &crate::status::AtomicStatus,
		_objects: &crate::hybrid_policy::HybridObjectMap<K>,
	) -> crate::Tier {
		crate::Tier::Slow
	}
}
