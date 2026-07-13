/*
 * Copyright (c) Kia Shakiba
 *
 * This source code is licensed under the GNU AGPLv3 license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! Statistics for `two_q_hybrid_cache`.
//!
//! The counters backing this snapshot live directly on `AtomicStatus`
//! (`status.rs`) rather than in a separate atomics struct here — same
//! rationale as `lru_hybrid_cache::LruHybridStats`/`lfu_hybrid_cache::LfuHybridStats`:
//! `AtomicStatus` is already the one shared, per-cache structure both
//! `PaperCache` and `PolicyWorker` hold (`status: StatusRef`), so no new
//! field is needed on `PaperCache` itself, which matters because
//! `PaperCache`'s struct definition in `lib.rs` is shared across every
//! value type and adding a field there would ripple into every other
//! constructor in the file, not just this feature's.

/// A point-in-time snapshot of `two_q_hybrid_cache` statistics.
///
/// Assembled by `AtomicStatus::two_q_hybrid_stats` and returned from
/// `PaperCache::two_q_hybrid_stats`.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct TwoQHybridStats {
	/// Objects moved from the FIFO queue or the main queue's slow portion
	/// into the main queue's fast portion.
	pub promotions: u64,
	/// Objects moved from the main queue's fast portion to its slow portion
	/// (fast tier pressure).
	pub demotions: u64,
	/// Objects permanently removed — either the FIFO queue's tail aging out
	/// without a second access, or the main queue's slow tail once overall
	/// cache capacity is exhausted.
	pub evictions: u64,
	/// Current bytes accounted to the fast tier (live gauge).
	pub fast_bytes_used: u64,
	/// Current bytes accounted to the slow tier — FIFO queue and main
	/// queue's slow portion combined (live gauge).
	pub slow_bytes_used: u64,
	/// Current number of objects in the fast tier (live gauge).
	pub fast_objects: u64,
	/// Current number of objects in the slow tier (live gauge).
	pub slow_objects: u64,
}
