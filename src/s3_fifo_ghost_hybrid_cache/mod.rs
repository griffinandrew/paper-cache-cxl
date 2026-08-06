/*
 * Copyright (c) Kia Shakiba
 *
 * This source code is licensed under the GNU AGPLv3 license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! Single-instance, segmented-S3-FIFO hybrid cache with a ghost queue.
//!
//! Identical architecture and admission/demotion/promotion/eviction rules
//! to `s3_fifo_hybrid_cache` — see that module's docs for the full design
//! (including the "contiguous front run" invariant and the eager
//! one-access-queue-promotion vs. lazy main-queue-reference-bit asymmetry)
//! — plus a bare-key ghost queue remembering objects that aged out of the
//! one-access queue without a second access, so a later re-admission is
//! trusted immediately (lands directly in the main queue's fast tier)
//! instead of restarting from the one-access queue. See
//! `worker::policy::policy_stack::s3_fifo_ghost_hybrid_stack`'s module doc
//! for the full ghost-queue mechanics and lifecycle.

mod stats;

pub use crate::tiered_buffer::TieredBuffer;
pub use stats::S3FifoGhostHybridStats;

/// Marker type selecting `s3_fifo_ghost_hybrid_cache`'s behavior for the
/// shared generic `impl<K, S> PaperCache<K, TieredBuffer, S>` block in
/// `lib.rs` (see `crate::hybrid_policy::HybridPolicy`). Admission is
/// unconditional to the *slow* tier regardless of ghost history — same
/// reasoning as `TwoQGhostHybridPolicy`. `ExtraConfig = f64` carries the
/// one-access queue's byte-budget ratio, same as `S3FifoHybridPolicy`.
pub struct S3FifoGhostHybridPolicy;

impl crate::hybrid_policy::HybridPolicy for S3FifoGhostHybridPolicy {
	type Stats = S3FifoGhostHybridStats;
	type ExtraConfig = f64;

	fn seed_policy(ratio: f64) -> crate::PaperPolicy {
		crate::PaperPolicy::S3FifoGhostHybrid(ratio)
	}

	fn stats_from_status(status: &crate::status::AtomicStatus) -> S3FifoGhostHybridStats {
		status.s3_fifo_ghost_hybrid_stats()
	}

	fn admission_tier<K>(
		_hashed_key: crate::HashedKey,
		_status: &crate::status::AtomicStatus,
		_objects: &crate::hybrid_policy::HybridObjectMap<K>,
	) -> crate::Tier {
		crate::Tier::Slow
	}
}
