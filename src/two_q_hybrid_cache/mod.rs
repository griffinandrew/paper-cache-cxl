/*
 * Copyright (c) Kia Shakiba
 *
 * This source code is licensed under the GNU AGPLv3 license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! Single-instance, segmented-2Q hybrid cache.
//!
//! Same overall architecture as `lru_hybrid_cache`/`lfu_hybrid_cache` — one
//! `PaperCache<K, TieredBuffer>`, not two composed instances — but the
//! object flow follows the 2Q algorithm:
//!
//! * Admission: every new object is placed in a one-access FIFO queue that
//!   lives entirely in the slow tier.
//! * Demotion: the LRU tail of the main queue's fast-tier portion moves to
//!   the top of its slow-tier portion when fast-tier space is needed.
//! * Promotion: a re-accessed FIFO-queue object moves straight to the top
//!   of the main queue's fast-tier portion; a re-accessed main-queue
//!   slow-tier object moves to the top of the fast-tier portion.
//! * Eviction: the FIFO queue's tail is sacrificed first (an object that
//!   ages out without a second access), falling back to the main queue's
//!   slow tail once the FIFO queue is empty.
//!
//! Unlike classic 2Q (and this crate's own plain `PaperPolicy::TwoQ`), no
//! ghost queue is kept for objects that age out of the FIFO queue — an
//! exact-membership check on every admission (which already pays a
//! synchronous slow-tier/PMEM write here) was judged an unwelcome added
//! cost; see `CLAUDE.md`'s `two_q_hybrid_cache` section for the reasoning
//! and the probabilistic-structure alternative left as future work.
//!
//! A live object's bytes exist in exactly one tier's allocation at a time —
//! see [`crate::tiered_buffer::TieredBuffer`] and `Object::set_data`, which
//! together make promotion/demotion an in-place data move rather than a
//! copy. `TieredBuffer` itself lives in the crate-root `tiered_buffer`
//! module, shared with `lru_hybrid_cache`/`lfu_hybrid_cache` (all three
//! hybrid-cache features are mutually exclusive — see `lib.rs`'s
//! `compile_error!` guards — since each defines its own inherent-method
//! `PaperCache<K, TieredBuffer, S>` impl block).
//!
//! The policy stack lives at
//! `worker::policy::policy_stack::two_q_hybrid_stack::TwoQHybridStack`
//! (`PaperPolicy::TwoQHybrid`) and `PolicyWorker` performs the actual tier
//! migrations it reports, recording counters directly on `AtomicStatus`
//! (see `stats` module docs for why).

mod stats;

pub use crate::tiered_buffer::TieredBuffer;
pub use stats::TwoQHybridStats;

/// Marker type selecting `two_q_hybrid_cache`'s behavior for the shared
/// generic `impl<K, S> PaperCache<K, TieredBuffer, S>` block in `lib.rs`
/// (see `crate::hybrid_policy::HybridPolicy`). Admission is unconditional
/// to the *slow* tier: every new object starts in the one-access FIFO
/// queue (`TwoQHybridStack::insert`); only a re-access promotes it to the
/// fast tier. `ExtraConfig = f64` carries `k_in`, the FIFO queue's byte
/// budget as a fraction of `max_size` -- the one hybrid design with a
/// constructor parameter beyond `max_size`/`fast_tier_size`.
pub struct TwoQHybridPolicy;

impl crate::hybrid_policy::HybridPolicy for TwoQHybridPolicy {
	type Stats = TwoQHybridStats;
	type ExtraConfig = f64;

	fn seed_policy(k_in: f64) -> crate::PaperPolicy {
		crate::PaperPolicy::TwoQHybrid(k_in)
	}

	fn stats_from_status(status: &crate::status::AtomicStatus) -> TwoQHybridStats {
		status.two_q_hybrid_stats()
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
