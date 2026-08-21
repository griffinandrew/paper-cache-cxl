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
		status.hybrid_stats()
	}

	fn admission_tier<K>(
		hashed_key: crate::HashedKey,
		_status: &crate::status::AtomicStatus,
		objects: &crate::hybrid_policy::HybridObjectMap<K>,
	) -> crate::Tier {
		use crate::object_store::ObjectStore;

		// Brand-new key: admitted to the one-access FIFO queue, always slow.
		//
		// An *existing* key is built Fast, because this stack's `touch()`
		// always ends with the key in the fast tier -- a `Fifo` hit promotes
		// it to the main queue's fast portion, and a `Main` hit reorders it
		// there (promoting first if it was slow). Building Slow here, as this
		// previously did unconditionally, left every re-set of an
		// already-fast key physically in PMEM while the stack accounted it to
		// the fast tier, and nothing ever reconciled the two.
		match objects.get_ref(&hashed_key) {
			Some(_) => crate::Tier::Fast,
			None => crate::Tier::Slow,
		}
	}
}
