/*
 * Copyright (c) Kia Shakiba
 *
 * This source code is licensed under the GNU AGPLv3 license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! Single-instance, segmented-LRU hybrid cache with a size-split fast AND
//! slow tier.
//!
//! Same one-`PaperCache<K, TieredBuffer>` architecture and LRU
//! admission/promotion/demotion/eviction semantics as `lru_hybrid_cache` --
//! still exactly one physical DRAM allocator path and one physical PMEM
//! allocator path, `Tier`/`TieredBuffer` unchanged -- but the fast (DRAM)
//! tier's *and* the slow (PMEM) tier's bookkeeping are each split into two
//! independently-tracked segments ("small"/"large") by a runtime-
//! configurable byte threshold on each object's size. Only the two fast
//! segments carry independent, configurable capacities; the two slow lists
//! carry no capacity of their own and stay governed by the overall
//! `max_size` terminal-eviction trigger, exactly like `lru_hybrid_cache`'s
//! single slow tier today -- the slow-tier split is purely about which
//! recency list an object's eviction candidacy is tracked in, for fairness,
//! not a new capacity dimension.
//!
//! A live object's bytes exist in exactly one tier's allocation at a time,
//! same as `lru_hybrid_cache` -- see [`crate::tiered_buffer::TieredBuffer`]
//! and `Object::set_data`. Moving between the two segments *within* a tier
//! (a reclassifying overwrite, or promotion routing) never touches
//! `TieredBuffer` at all: both fast segments are physically
//! `TieredBuffer::Fast`, both slow lists are physically `TieredBuffer::Slow`
//! -- only a genuine fast/slow crossing allocates.
//!
//! The policy stack lives at `worker::policy::policy_stack::
//! lru_sized_hybrid_stack::LruSizedHybridStack` (`PaperPolicy::
//! LruSizedHybrid`) -- see that module's doc for the full algorithm,
//! including why it uses four independent homogeneous recency lists rather
//! than `LruHybridStack`'s single-list-with-boundary-cursor trick, and the
//! eviction-priority/fallback rules. `PolicyWorker` performs the actual tier
//! migrations it reports, recording counters directly on `AtomicStatus` (see
//! `stats` module docs for why, same rationale as `lru_hybrid_cache`).

mod stats;

pub use crate::tiered_buffer::TieredBuffer;
pub use stats::LruSizedHybridStats;

/// Marker type selecting `lru_sized_hybrid_cache`'s behavior for the shared
/// generic `impl<K, S> PaperCache<K, TieredBuffer, S>` block in `lib.rs`
/// (see `crate::hybrid_policy::HybridPolicy`). Admission is unconditional:
/// every `set()` builds `TieredBuffer::new_fast`, matching
/// `LruSizedHybridStack::insert` always re-admitting to one of the two fast
/// segments (including on overwrite, which it treats as a promotion/
/// reclassification). Which of the two fast segments an admission lands in
/// is decided entirely by `LruSizedHybridStack` on the worker thread, using
/// the `size: ObjectSize` it's already given -- invisible at this layer,
/// since both segments share the identical `TieredBuffer::Fast` physical
/// representation.
pub struct LruSizedHybridPolicy;

impl crate::hybrid_policy::HybridPolicy for LruSizedHybridPolicy {
	type Stats = LruSizedHybridStats;
	type ExtraConfig = ();

	fn seed_policy(_extra: ()) -> crate::PaperPolicy {
		crate::PaperPolicy::LruSizedHybrid
	}

	fn stats_from_status(status: &crate::status::AtomicStatus) -> LruSizedHybridStats {
		status.lru_sized_hybrid_stats()
	}

	fn admission_tier<K>(
		_hashed_key: crate::HashedKey,
		_status: &crate::status::AtomicStatus,
		_objects: &std::sync::Arc<crate::hybrid_policy::HybridObjectMap<K>>,
	) -> crate::Tier {
		crate::Tier::Fast
	}
}
