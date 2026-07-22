/*
 * Copyright (c) Kia Shakiba
 *
 * This source code is licensed under the GNU AGPLv3 license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! Single-instance, segmented-LFU hybrid cache.
//!
//! Same overall architecture as `lru_hybrid_cache` — **one**
//! `PaperCache<K, TieredBuffer>`, not two composed instances — but the
//! fast/slow boundary is frequency-ordered rather than recency-ordered:
//!
//! * Admission: while the fast tier has capacity, new objects are admitted
//!   into the fast tier. Once the fast tier is full, every new object is,
//!   by definition, the least frequently accessed object, so it lands in
//!   the slow tier — see `LfuHybridStack`'s doc comment for how this is
//!   achieved as an emergent result of "always admit fast, let settle
//!   demote if needed" rather than a special-cased admission check.
//! * Demotion: the least frequently accessed fast-tier object moves to the
//!   slow tier when fast-tier space is needed.
//! * Promotion: a slow-tier object moves to the fast tier once its access
//!   frequency strictly exceeds the minimum frequency among fast-tier
//!   residents — which may itself demote the (new) fast-tier minimum.
//! * Eviction: the least frequently accessed slow-tier object is removed
//!   when overall cache capacity is exhausted.
//!
//! A live object's bytes exist in exactly one tier's allocation at a time —
//! see [`crate::tiered_buffer::TieredBuffer`] and `Object::set_data`, which
//! together make promotion/demotion an in-place data move rather than a
//! copy. `TieredBuffer` itself lives in the crate-root `tiered_buffer`
//! module, shared with `lru_hybrid_cache` (the two features are mutually
//! exclusive — see `lib.rs`'s `compile_error!` guard — since both define
//! their own inherent-method `PaperCache<K, TieredBuffer, S>` impl block).
//!
//! The policy stack lives at
//! `worker::policy::policy_stack::lfu_hybrid_stack::LfuHybridStack`
//! (`PaperPolicy::LfuHybrid`) and `PolicyWorker` performs the actual tier
//! migrations it reports, recording counters directly on `AtomicStatus`
//! (see `stats` module docs for why).

mod stats;

pub use crate::tiered_buffer::TieredBuffer;
pub use stats::LfuHybridStats;

/// Marker type selecting `lfu_hybrid_cache`'s behavior for the shared
/// generic `impl<K, S> PaperCache<K, TieredBuffer, S>` block in `lib.rs`
/// (see `crate::hybrid_policy::HybridPolicy`). A brand-new key built once
/// the fast tier has genuinely reached capacity (`LfuHybridStack`'s
/// one-time admission latch, mirrored onto `AtomicStatus` since this
/// thread has no direct access to the worker-owned stack) goes straight
/// to the slow tier -- this is what the stack would decide anyway, so
/// building it fast first would only cost a synchronous DRAM write
/// immediately followed by an async correction. An *existing* key is
/// never affected by this check regardless of its current tier:
/// re-setting one is an access, which may or may not promote it, and only
/// the stack can decide that.
pub struct LfuHybridPolicy;

impl crate::hybrid_policy::HybridPolicy for LfuHybridPolicy {
	type Stats = LfuHybridStats;
	type ExtraConfig = ();

	fn seed_policy(_extra: ()) -> crate::PaperPolicy {
		crate::PaperPolicy::LfuHybrid
	}

	fn stats_from_status(status: &crate::status::AtomicStatus) -> LfuHybridStats {
		status.lfu_hybrid_stats()
	}

	fn admission_tier<K>(
		hashed_key: crate::HashedKey,
		status: &crate::status::AtomicStatus,
		objects: &std::sync::Arc<crate::hybrid_policy::HybridObjectMap<K>>,
	) -> crate::Tier {
		let is_new = !objects.contains_key(&hashed_key);

		if is_new && status.lfu_hybrid_admission_latched() {
			crate::Tier::Slow
		} else {
			crate::Tier::Fast
		}
	}
}
