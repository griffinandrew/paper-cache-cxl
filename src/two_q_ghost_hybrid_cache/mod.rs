/*
 * Copyright (c) Kia Shakiba
 *
 * This source code is licensed under the GNU AGPLv3 license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! Single-instance, segmented-2Q hybrid cache with a ghost queue.
//!
//! Identical architecture and admission/demotion/promotion/eviction rules
//! to `two_q_hybrid_cache` — see that module's docs for the full design —
//! plus a bare-key ghost queue remembering objects that aged out of the
//! one-access FIFO queue without a second access, so a later re-admission
//! is trusted immediately (lands directly in the main queue's fast tier)
//! instead of restarting from the FIFO queue. See
//! `worker::policy::policy_stack::two_q_ghost_hybrid_stack`'s module doc
//! for the full ghost-queue mechanics, lifecycle, and the "where a ghost
//! hit lands" design note.

mod stats;

pub use crate::tiered_buffer::TieredBuffer;
pub use stats::TwoQGhostHybridStats;

/// Marker type selecting `two_q_ghost_hybrid_cache`'s behavior for the
/// shared generic `impl<K, S> PaperCache<K, TieredBuffer, S>` block in
/// `lib.rs` (see `crate::hybrid_policy::HybridPolicy`). Admission is
/// unconditional to the *slow* tier regardless of ghost history — a ghost
/// hit is corrected to `Fast` by the worker's ordinary async migration path
/// (see the stack's module doc), not by the API layer guessing differently
/// up front. `ExtraConfig = f64` carries `k_in`, same as
/// `TwoQHybridPolicy`.
pub struct TwoQGhostHybridPolicy;

impl crate::hybrid_policy::HybridPolicy for TwoQGhostHybridPolicy {
	type Stats = TwoQGhostHybridStats;
	type ExtraConfig = f64;

	fn seed_policy(k_in: f64) -> crate::PaperPolicy {
		crate::PaperPolicy::TwoQGhostHybrid(k_in)
	}

	fn stats_from_status(status: &crate::status::AtomicStatus) -> TwoQGhostHybridStats {
		status.two_q_ghost_hybrid_stats()
	}

	fn admission_tier<K>(
		_hashed_key: crate::HashedKey,
		_status: &crate::status::AtomicStatus,
		_objects: &crate::hybrid_policy::HybridObjectMap<K>,
	) -> crate::Tier {
		crate::Tier::Slow
	}
}
