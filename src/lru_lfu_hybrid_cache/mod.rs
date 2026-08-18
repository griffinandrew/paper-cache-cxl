/*
 * Copyright (c) Kia Shakiba
 *
 * This source code is licensed under the GNU AGPLv3 license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! Single-instance hybrid cache with a different eviction discipline per
//! tier: **recency (LRU) in the fast tier, frequency (LFU) in the slow
//! tier**.
//!
//! Like the other hybrids here this is **one** `PaperCache<K, TieredBuffer>`,
//! not two composed instances, and a live object's bytes exist in exactly one
//! tier's allocation at a time (see [`crate::tiered_buffer::TieredBuffer`]
//! and `Object::set_data`). What is new is that the two tiers no longer rank
//! by the same metric, which is what makes this design distinct rather than a
//! reparameterization of `lru_hybrid_cache`:
//!
//! - **Admission**: new object → fast tier, recency head, frequency 1.
//! - **Demotion**: fast tier's LRU tail → slow tier, carrying its
//!   accumulated frequency.
//! - **Promotion**: a slow-tier object reaching `promote_k` accesses → fast
//!   tier's recency head, counter reset.
//! - **Eviction**: the slow tier's minimum-frequency object.
//!
//! In one line: frequency is the admission control *into* DRAM; recency is
//! the retention policy *within* DRAM.
//!
//! The full derivation — why promotion is a fixed threshold rather than the
//! cross-tier frequency comparison `lfu_hybrid_cache` uses, why the fast tier
//! counts a frequency it does not rank by, why that counter is capped, and
//! why an overwrite goes through the same gate a read does — lives in
//! `worker::policy::policy_stack::lru_lfu_hybrid_stack`'s module doc
//! (`PaperPolicy::LruLfuHybrid`).

mod stats;

pub use crate::tiered_buffer::TieredBuffer;
pub use stats::LruLfuHybridStats;

/// Marker type selecting `lru_lfu_hybrid_cache`'s behavior for the shared
/// generic `impl<K, S> PaperCache<K, TieredBuffer, S>` block in `lib.rs`
/// (see `crate::hybrid_policy::HybridPolicy`).
pub struct LruLfuHybridPolicy;

impl crate::hybrid_policy::HybridPolicy for LruLfuHybridPolicy {
	type Stats = LruLfuHybridStats;
	/// `promote_k`: accesses a slow-tier object must accumulate to earn the
	/// fast tier. Carried into the seeded `PaperPolicy` value.
	type ExtraConfig = u16;

	fn seed_policy(promote_k: u16) -> crate::PaperPolicy {
		crate::PaperPolicy::LruLfuHybrid(promote_k)
	}

	fn stats_from_status(status: &crate::status::AtomicStatus) -> LruLfuHybridStats {
		status.lru_lfu_hybrid_stats()
	}

	/// A brand-new key is admitted to the fast tier; an **existing** key is
	/// written back to whichever tier it currently occupies.
	///
	/// The second half is what makes the frequency gate real. An overwrite
	/// is an access, not an automatic promotion (see the stack's module
	/// doc), so a `set()` on a slow-tier key must not be built in DRAM and
	/// then corrected back to PMEM — it is written straight to PMEM, and if
	/// the stack decides that access crossed `promote_k` it emits a
	/// `Tier::Fast` migration that moves it. Same shape as
	/// `fifo_hybrid_cache`'s, which needs an existing key's tier for its own
	/// reasons.
	fn admission_tier<K>(
		hashed_key: crate::HashedKey,
		_status: &crate::status::AtomicStatus,
		objects: &crate::hybrid_policy::HybridObjectMap<K>,
	) -> crate::Tier {
		use crate::object_store::ObjectStore;

		let existing_tier = objects.get_ref(&hashed_key)
			.map(|object| if object.data().is_fast() { crate::Tier::Fast } else { crate::Tier::Slow });

		match existing_tier {
			Some(crate::Tier::Slow) => crate::Tier::Slow,
			Some(crate::Tier::Fast) | None => crate::Tier::Fast,
		}
	}
}
