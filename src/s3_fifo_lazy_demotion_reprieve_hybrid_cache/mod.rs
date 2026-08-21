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
//! `worker::policy::policy_stack::s3_fifo_lazy_demotion_reprieve_hybrid_stack`'s
//! module doc for the full mechanics, including the O(number of currently-
//! fast keys) splice technique used to place the reprieved key exactly
//! adjacent to the fast/slow boundary without ever tagging it Fast.

mod stats;

pub use crate::tiered_buffer::TieredBuffer;
pub use stats::S3FifoLazyDemotionReprieveHybridStats;

/// Marker type selecting
/// `s3_fifo_lazy_demotion_reprieve_hybrid_cache`'s
/// behavior for the shared generic `impl<K, S> PaperCache<K, TieredBuffer,
/// S>` block in `lib.rs` (see `crate::hybrid_policy::HybridPolicy`).
/// Admission is unconditional to the *fast* tier, same reasoning as
/// `S3FifoGhostLazyDemotionFastAdmissionMidpointHybridPolicy` -- the
/// reprieve mechanic only changes what happens to a one-access-queue key
/// that ages out, not admission itself. `ExtraConfig = f64` carries the
/// one-access queue's byte-budget ratio, same as that policy.
pub struct S3FifoLazyDemotionReprieveHybridPolicy;

impl crate::hybrid_policy::HybridPolicy for S3FifoLazyDemotionReprieveHybridPolicy {
	type Stats = S3FifoLazyDemotionReprieveHybridStats;
	type ExtraConfig = f64;

	fn seed_policy(ratio: f64) -> crate::PaperPolicy {
		crate::PaperPolicy::S3FifoLazyDemotionReprieveHybrid(ratio)
	}

	fn stats_from_status(status: &crate::status::AtomicStatus) -> S3FifoLazyDemotionReprieveHybridStats {
		status.hybrid_stats()
	}

	fn admission_tier<K>(
		hashed_key: crate::HashedKey,
		_status: &crate::status::AtomicStatus,
		objects: &crate::hybrid_policy::HybridObjectMap<K>,
	) -> crate::Tier {
		use crate::object_store::ObjectStore;

		// A brand-new key is admitted to the SLOW tier -- this design's
		// one-access queue lives in PMEM (that is the whole point of the
		// variant), so the API layer must build the value as `Slow` to match
		// where the stack is about to put it. This is the inverse of the
		// fast-admission variants, whose one-access queue is DRAM.
		//
		// An *existing* key keeps whatever tier it is already in, which is not
		// merely a nicety: this design's stack records no tier transition for a
		// `set()` on a tracked key -- `insert()` routes to `touch()`, which only
		// marks the reference bit -- so rebuilding the value in the wrong tier
		// would leave the object physically in one tier while the stack accounts
		// it to the other. Nothing ever reconciles that, so the tier gauges
		// would drift and real DRAM usage could pass the fast-tier budget.
		match objects.get_ref(&hashed_key) {
			Some(object) if object.data().is_slow() => crate::Tier::Slow,
			// Existing and currently fast -- promoted at some point, keep it there.
			Some(_) => crate::Tier::Fast,
			// Brand new -> the one-access queue -> PMEM.
			None => crate::Tier::Slow,
		}
	}
}
