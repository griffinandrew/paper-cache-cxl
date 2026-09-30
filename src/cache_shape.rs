/*
 * Copyright (c) Kia Shakiba
 *
 * This source code is licensed under the GNU AGPLv3 license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! What differs between the flat cache and the tiered one on the READ side.
//!
//! `PaperCache`'s `get`, `get_into`, `del`, `has`, `peek`, `ttl`, `size`,
//! `wipe` and `resize` are one impl block, generic over the value shape `V`
//! (`BufferDRAM` and `BufferPMEM`, the flat shapes, and `TieredBuffer`, the
//! tiered one -- they stay distinct types because the flat and the tiered
//! constructors must stay disjoint, see `value::ValueShape`). Two places in
//! those bodies are not the same for the two, and each is a hook here:
//!
//! * a hit served from a tier is counted by the tiered cache, by tier
//!   (`hit_served`); a flat cache has one tier and counts hits only;
//! * a resize is refused when it would leave a queue with no budget, and which
//!   policies carry such a budget differs (`resize_refused`).
//!
//! The trait is public only because it bounds a public impl; it is in a
//! private module and cannot be named outside the crate.

use crate::{CacheSize, Tier, status::AtomicStatus};

/// A value shape the shared read side runs over. See the module doc.
pub trait CacheShape: 'static + Send + Sync {
	/// A hit was served from `tier`.
	fn hit_served(status: &AtomicStatus, tier: Tier);

	/// Whether a resize to `max_size` must be refused as `InvalidPolicy`: the
	/// stack recomputes its queue budgets against the NEW size, so a resize can
	/// starve a queue that was fine at construction, and a zero-capacity main
	/// queue spins the eviction loop.
	fn resize_refused(status: &AtomicStatus, max_size: CacheSize) -> bool;
}

/// The flat shapes: one tier, and the non-tiered s3-fifo design's main budget
/// is the one that can truncate to zero -- every configured policy is checked,
/// as the list is validated whole at construction.
#[cfg(any(feature = "all_dram", feature = "key_value_pmem"))]
impl<V: crate::value::ValueShape> CacheShape for V {
	fn hit_served(_status: &AtomicStatus, _tier: Tier) {}

	fn resize_refused(status: &AtomicStatus, max_size: CacheSize) -> bool {
		status
			.policies()
			.iter()
			.any(|configured| crate::s_three_fifo_starves_main(*configured, max_size))
	}
}

/// The tiered shape: hits are counted by the tier they were served from, and
/// the active policy's s3-fifo queue budgets must survive the new size --
/// the same condition the tiered constructor rejects.
#[cfg(feature = "hybrid_cache_common")]
impl CacheShape for crate::TieredBuffer {
	fn hit_served(status: &AtomicStatus, tier: Tier) {
		status.incr_served_hit(tier);
	}

	fn resize_refused(status: &AtomicStatus, max_size: CacheSize) -> bool {
		crate::s3_fifo_queue_budgets(status.policy())
			.is_some_and(|(ratio, sizes_main)| sizes_main && ((1.0 - ratio) * max_size as f64) as CacheSize == 0)
	}
}
