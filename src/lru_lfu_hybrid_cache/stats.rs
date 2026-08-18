/*
 * Copyright (c) Kia Shakiba
 *
 * This source code is licensed under the GNU AGPLv3 license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! Statistics for `lru_lfu_hybrid_cache`.
//!
//! The counters backing this snapshot live directly on `AtomicStatus`
//! (`status.rs`) rather than in a separate atomics struct here — see
//! `lru_hybrid_cache::stats`'s module doc for the reasoning, which applies
//! unchanged.

/// A point-in-time snapshot of `lru_lfu_hybrid_cache` statistics.
///
/// Assembled by `AtomicStatus::lru_lfu_hybrid_stats` and returned from
/// `PaperCache::lru_lfu_hybrid_stats`.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct LruLfuHybridStats {
	/// Objects moved from the slow tier to the fast tier. Every promotion
	/// here is a threshold crossing — a slow-tier object accumulated
	/// `promote_k` accesses — so this doubles as the count of objects that
	/// earned DRAM by demonstrating reuse.
	pub promotions: u64,
	/// Objects moved from the fast tier to the slow tier (fast-tier
	/// pressure), each carrying its accumulated frequency across.
	pub demotions: u64,
	/// Objects permanently removed, normally the slow tier's
	/// minimum-frequency key (cache capacity exhausted).
	pub evictions: u64,
	/// Current bytes accounted to the fast tier (live gauge).
	pub fast_bytes_used: u64,
	/// Current bytes accounted to the slow tier (live gauge).
	pub slow_bytes_used: u64,
	/// Current number of objects in the fast tier (live gauge).
	pub fast_objects: u64,
	/// Current number of objects in the slow tier (live gauge).
	pub slow_objects: u64,
}
