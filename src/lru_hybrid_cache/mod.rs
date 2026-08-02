/*
 * Copyright (c) Kia Shakiba
 *
 * This source code is licensed under the GNU AGPLv3 license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! Single-instance, segmented-LRU hybrid cache.
//!
//! Unlike a design that composes two independent `PaperCache` instances,
//! `lru_hybrid_cache` is **one** `PaperCache<K, TieredBuffer>`. The fast
//! (DRAM) and slow (PMEM) tiers are a single logical
//! LRU queue segmented by a byte-budgeted boundary: every new object is
//! admitted at the top of the fast tier; as objects age past the boundary
//! they are demoted (physically moved) to the slow tier; accessing a
//! slow-tier object promotes (physically moves) it back to the top of the
//! fast tier; when overall cache capacity is exhausted, the least recently
//! accessed slow-tier object is evicted.
//!
//! A live object's bytes exist in exactly one tier's allocation at a time —
//! see [`crate::tiered_buffer::TieredBuffer`] and `Object::set_data`, which
//! together make promotion/demotion an in-place data move rather than a
//! copy. `TieredBuffer` itself lives in the crate-root `tiered_buffer`
//! module (shared with `lfu_hybrid_cache`, which needs the identical type —
//! see that module's doc comment for why).
//!
//! The policy stack lives at
//! `worker::policy::policy_stack::lru_hybrid_stack::LruHybridStack`
//! (`PaperPolicy::LruHybrid`) and `PolicyWorker` performs the actual tier
//! migrations it reports, recording counters directly on `AtomicStatus`
//! (see `stats` module docs for why).

mod stats;

pub use crate::tiered_buffer::TieredBuffer;
pub use stats::LruHybridStats;

/// Marker type selecting `lru_hybrid_cache`'s behavior for the shared
/// generic `impl<K, S> PaperCache<K, TieredBuffer, S>` block in `lib.rs`
/// (see `crate::hybrid_policy::HybridPolicy`). Admission is unconditional:
/// every `set()` builds `TieredBuffer::new_fast`, matching
/// `LruHybridStack::insert` always re-admitting to the fast tier
/// (including on overwrite, which it treats as a promotion).
pub struct LruHybridPolicy;

impl crate::hybrid_policy::HybridPolicy for LruHybridPolicy {
	type Stats = LruHybridStats;
	type ExtraConfig = ();

	fn seed_policy(_extra: ()) -> crate::PaperPolicy {
		crate::PaperPolicy::LruHybrid
	}

	fn stats_from_status(status: &crate::status::AtomicStatus) -> LruHybridStats {
		status.lru_hybrid_stats()
	}

	fn admission_tier<K>(
		_hashed_key: crate::HashedKey,
		_status: &crate::status::AtomicStatus,
		_objects: &crate::hybrid_policy::HybridObjectMap<K>,
	) -> crate::Tier {
		crate::Tier::Fast
	}
}
