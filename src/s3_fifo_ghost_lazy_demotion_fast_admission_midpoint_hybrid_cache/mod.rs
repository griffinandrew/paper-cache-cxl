/*
 * Copyright (c) Kia Shakiba
 *
 * This source code is licensed under the GNU AGPLv3 license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! Single-instance, segmented-S3-FIFO hybrid cache with a ghost queue, a
//! demotion-time reference-bit gate, a fast-tier-resident one-access
//! queue, AND a mid-slow-segment reference-bit checkpoint.
//!
//! Identical architecture and admission/eviction rules to
//! `s3_fifo_ghost_lazy_demotion_fast_admission_hybrid_cache` — see that
//! module's docs for the full design — plus one addition: a checkpoint
//! roughly halfway through the SLOW portion of the main queue gives a
//! reaccessed object an early second chance, instead of making it wait
//! until it reaches the eviction tail. See
//! `worker::policy::policy_stack::s3_fifo_ghost_lazy_demotion_fast_admission_midpoint_hybrid_stack`'s
//! module doc for the full mechanics, including the O(1)-amortized cursor
//! that locates "the middle" without an O(n) scan.

mod stats;

pub use crate::tiered_buffer::TieredBuffer;
pub use stats::S3FifoGhostLazyDemotionFastAdmissionMidpointHybridStats;

/// Marker type selecting
/// `s3_fifo_ghost_lazy_demotion_fast_admission_midpoint_hybrid_cache`'s
/// behavior for the shared generic `impl<K, S> PaperCache<K, TieredBuffer,
/// S>` block in `lib.rs` (see `crate::hybrid_policy::HybridPolicy`).
/// Admission is unconditional to the *fast* tier, same reasoning as
/// `S3FifoGhostLazyDemotionFastAdmissionHybridPolicy` -- the new
/// mid-segment checkpoint only changes what happens to objects already
/// resident in the slow portion of the main queue, not admission itself.
/// `ExtraConfig = f64` carries the one-access queue's byte-budget ratio,
/// same as that policy.
pub struct S3FifoGhostLazyDemotionFastAdmissionMidpointHybridPolicy;

impl crate::hybrid_policy::HybridPolicy for S3FifoGhostLazyDemotionFastAdmissionMidpointHybridPolicy {
	type Stats = S3FifoGhostLazyDemotionFastAdmissionMidpointHybridStats;
	type ExtraConfig = f64;

	fn seed_policy(ratio: f64) -> crate::PaperPolicy {
		crate::PaperPolicy::S3FifoGhostLazyDemotionFastAdmissionMidpointHybrid(ratio)
	}

	fn stats_from_status(status: &crate::status::AtomicStatus) -> S3FifoGhostLazyDemotionFastAdmissionMidpointHybridStats {
		status.s3_fifo_ghost_lazy_demotion_fast_admission_midpoint_hybrid_stats()
	}

	fn admission_tier<K>(
		hashed_key: crate::HashedKey,
		_status: &crate::status::AtomicStatus,
		objects: &crate::hybrid_policy::HybridObjectMap<K>,
	) -> crate::Tier {
		use crate::object_store::ObjectStore;

		// A brand-new key follows this design's own admission rule (below).
		// An *existing* key keeps whatever tier it currently occupies.
		//
		// That second clause is load-bearing. This stack records no tier
		// transition for a `set()` on a key already in its main queue --
		// `insert()` routes to `touch()`, which only marks the reference bit
		// -- so choosing a tier here that disagrees with where the object
		// already lives leaves physical placement and the stack's accounting
		// permanently out of step, with no event that ever reconciles them.
		// Real DRAM then drifts away from `fast_tier_size` while
		// `fast_bytes_used` reports the stack's (wrong) view.
		//
		// Where the stack *does* move the key, it emits a migration and the
		// worker corrects the placement, so preserving the current tier is
		// safe in every case.
		match objects.get_ref(&hashed_key) {
			Some(object) => match object.data().is_fast() {
				true => crate::Tier::Fast,
				false => crate::Tier::Slow,
			},

			// Brand-new key: this design admits to the fast tier.
			None => crate::Tier::Fast,
		}
	}
}
