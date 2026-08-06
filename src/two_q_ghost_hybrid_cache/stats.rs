/*
 * Copyright (c) Kia Shakiba
 *
 * This source code is licensed under the GNU AGPLv3 license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! Statistics for `two_q_ghost_hybrid_cache`.
//!
//! Same rationale as `two_q_hybrid_cache::TwoQHybridStats`: the counters
//! backing this snapshot live directly on `AtomicStatus` (`status.rs`)
//! rather than in a separate atomics struct here.

/// A point-in-time snapshot of `two_q_ghost_hybrid_cache` statistics.
///
/// Assembled by `AtomicStatus::two_q_ghost_hybrid_stats` and returned from
/// `PaperCache::two_q_ghost_hybrid_stats`. Same shape as
/// `two_q_hybrid_cache::TwoQHybridStats` — a ghost-queue hit still shows up
/// as an ordinary `promotions` increment (it's a real `Tier::Fast` landing,
/// just via a different admission path); there's no separate ghost-hit
/// counter (see `TwoQGhostHybridStack`'s module doc for why the ghost
/// queue's own memory isn't tracked/gauged at all).
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct TwoQGhostHybridStats {
	/// Objects moved from the FIFO queue, the main queue's slow portion, or
	/// admitted directly via a ghost-queue hit, into the main queue's fast
	/// portion.
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
