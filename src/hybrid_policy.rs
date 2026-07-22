/*
 * Copyright (c) Kia Shakiba
 *
 * This source code is licensed under the GNU AGPLv3 license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! Abstracts over the four mutually-exclusive `TieredBuffer`-based
//! hybrid-cache designs (`lru_hybrid_cache`, `lfu_hybrid_cache`,
//! `two_q_hybrid_cache`, `fifo_hybrid_cache`) behind one trait, so `lib.rs`
//! needs only one generic `impl<K, S> PaperCache<K, TieredBuffer, S>` block
//! instead of four nearly-identical ones.
//!
//! This stays a *compile-time* dispatch, not a runtime one: exactly one of
//! the four features is ever enabled at once (see `lib.rs`'s
//! `compile_error!` guards), and `lib.rs` selects a single concrete
//! `ActiveHybridPolicy` type alias per build (mirroring the existing
//! `ObjectMapRef`/`Hybrid`/`BufferDRAM` pattern). The four impl blocks this
//! replaces were confirmed (via direct diff) to differ only in: which
//! `PaperPolicy` variant gets seeded; the `Stats` type and its accessor
//! method's name; one admission-rule branch inside `set()`; and
//! `two_q_hybrid_cache`'s extra `k_in: f64` constructor parameter.

use std::sync::Arc;

use dashmap::DashMap;

use crate::{HashedKey, NoHasher, Tier};
use crate::status::AtomicStatus;
use crate::object::Object;
use crate::policy::PaperPolicy;
use crate::tiered_buffer::TieredBuffer;

/// The object map type every hybrid-cache design uses -- always a plain
/// DRAM-resident `DashMap` (none of the four features interact with
/// `global_hashtable_pmem`/`hashbrown_dram`), so `admission_tier` can name
/// it concretely rather than going through the `ObjectStore` abstraction
/// built for the non-hybrid storage matrix.
pub type HybridObjectMap<K> = DashMap<HashedKey, Object<K, TieredBuffer>, NoHasher>;

/// The behavior that varies between the four hybrid-cache designs; see
/// each design's own module doc comment for the full paper-derived
/// admission/demotion/promotion/eviction rules. Implemented once by a
/// small marker struct per feature (`LruHybridPolicy`, `LfuHybridPolicy`,
/// `TwoQHybridPolicy`, `FifoHybridPolicy`), selected at compile time via
/// `lib.rs`'s `ActiveHybridPolicy` type alias.
pub trait HybridPolicy {
	/// The stats snapshot type returned by this policy's
	/// `{name}_hybrid_stats()` accessor (e.g. `LruHybridStats`).
	type Stats;

	/// Extra constructor input beyond `max_size`/`fast_tier_size`: `()` for
	/// every design except `two_q_hybrid_cache`, which needs `k_in: f64`
	/// (the FIFO queue's byte budget, fixed at construction).
	type ExtraConfig: Copy;

	/// Builds the `PaperPolicy` value seeded into `AtomicStatus::new` at
	/// construction time.
	fn seed_policy(extra: Self::ExtraConfig) -> PaperPolicy;

	/// Reads this policy's stats snapshot off the shared status.
	fn stats_from_status(status: &AtomicStatus) -> Self::Stats;

	/// Decides which tier `set()` should build a value's bytes in.
	/// Implementations that need to know whether `hashed_key` is a
	/// brand-new key, or what tier it currently occupies, look that up
	/// from `objects` themselves -- the two designs that need this
	/// (`lfu_hybrid_cache`, `fifo_hybrid_cache`) each look up exactly what
	/// they need; the two that don't (`lru_hybrid_cache`,
	/// `two_q_hybrid_cache`, both unconditional) ignore the arguments
	/// entirely.
	fn admission_tier<K>(
		hashed_key: HashedKey,
		status: &AtomicStatus,
		objects: &Arc<HybridObjectMap<K>>,
	) -> Tier;
}
