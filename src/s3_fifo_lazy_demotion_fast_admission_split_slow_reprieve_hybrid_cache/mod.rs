/*
 * Copyright (c) Kia Shakiba
 *
 * This source code is licensed under the GNU AGPLv3 license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! Single-instance, segmented-S3-FIFO hybrid cache with a demotion-time
//! reference-bit gate, a fast-tier-resident one-access queue, a two-segment
//! slow tier with a reference-bit check at the crossing, and no ghost
//! queue.
//!
//! Identical architecture and admission/eviction rules to
//! `s3_fifo_lazy_demotion_fast_admission_midpoint_reprieve_hybrid_cache` — see
//! that module's docs for the full design — with one change: the
//! approximate mid-slow-segment cursor is replaced by splitting the slow
//! tier into two physical FIFO segments, with every object's reference bit
//! checked as it crosses between them. See
//! `worker::policy::policy_stack::s3_fifo_lazy_demotion_fast_admission_split_slow_reprieve_hybrid_stack`'s
//! module doc for the full mechanics, including why the predecessor's
//! sampled-cursor checkpoint was replaced.

mod stats;

pub use crate::tiered_buffer::TieredBuffer;
pub use stats::S3FifoLazyDemotionFastAdmissionSplitSlowReprieveHybridStats;

/// Marker type selecting
/// `s3_fifo_lazy_demotion_fast_admission_split_slow_reprieve_hybrid_cache`'s
/// behavior for the shared generic `impl<K, S> PaperCache<K, TieredBuffer,
/// S>` block in `lib.rs` (see `crate::hybrid_policy::HybridPolicy`).
/// Admission is unconditional to the *fast* tier, same reasoning as every
/// other stack in this lineage -- neither the reprieve nor the slow-tier
/// split changes admission itself. `ExtraConfig = f64` carries the
/// one-access queue's byte-budget ratio, same as that policy.
pub struct S3FifoLazyDemotionFastAdmissionSplitSlowReprieveHybridPolicy;

impl crate::hybrid_policy::HybridPolicy for S3FifoLazyDemotionFastAdmissionSplitSlowReprieveHybridPolicy {
	type Stats = S3FifoLazyDemotionFastAdmissionSplitSlowReprieveHybridStats;
	type ExtraConfig = f64;

	fn seed_policy(ratio: f64) -> crate::PaperPolicy {
		crate::PaperPolicy::S3FifoLazyDemotionFastAdmissionSplitSlowReprieveHybrid(ratio)
	}

	fn stats_from_status(status: &crate::status::AtomicStatus) -> S3FifoLazyDemotionFastAdmissionSplitSlowReprieveHybridStats {
		status.s3_fifo_lazy_demotion_fast_admission_split_slow_reprieve_hybrid_stats()
	}

	fn admission_tier<K>(
		hashed_key: crate::HashedKey,
		_status: &crate::status::AtomicStatus,
		objects: &crate::hybrid_policy::HybridObjectMap<K>,
	) -> crate::Tier {
		use crate::object_store::ObjectStore;

		// A brand-new key is admitted to the fast tier (the one-access queue).
		//
		// An *existing* key keeps whatever tier it is already in, which is not
		// merely a nicety: this design's stack records no tier transition for a
		// `set()` on a tracked key -- `insert()` routes to `touch()`, which only
		// marks the reference bit -- so unconditionally rebuilding the value as
		// `Fast` leaves the object physically in DRAM while the stack still
		// accounts it to the slow tier. Nothing ever reconciles that, so every
		// re-set of a demoted key silently pushes real DRAM usage past the
		// fast-tier budget, and the `fast_bytes_used` gauge under-reports it.
		//
		// The cost is a synchronous PMEM write on a `set()` to a slow-resident
		// key -- the same tradeoff `lfu_hybrid_cache` already accepts for its
		// latched admissions -- in exchange for physical placement matching the
		// stack's accounting.
		match objects.get_ref(&hashed_key) {
			Some(object) if object.data().is_slow() => crate::Tier::Slow,
			_ => crate::Tier::Fast,
		}
	}
}
