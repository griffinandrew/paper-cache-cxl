/*
 * Copyright (c) Kia Shakiba
 *
 * This source code is licensed under the GNU AGPLv3 license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! Statistics for `lru_hybrid_cache`.
//!
//! The counters backing this snapshot live directly on `AtomicStatus`
//! (`status.rs`) rather than in a separate atomics struct here: `AtomicStatus`
//! is already the one shared, per-cache structure both `PaperCache` and
//! `PolicyWorker` hold (`status: StatusRef`), so no new field is needed on
//! `PaperCache` itself — which matters because `PaperCache`'s struct
//! definition in `lib.rs` is shared across every value type (`BufferDRAM`,
//! `BufferPMEM`, `TieredBuffer`, ...), and adding a field there would ripple
//! into every other constructor in the file, not just this feature's.

/// A point-in-time snapshot of `lru_hybrid_cache` statistics.
///
/// Assembled by `AtomicStatus::lru_hybrid_stats` and returned from
/// `PaperCache::lru_hybrid_stats`.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct LruHybridStats {
	/// Objects moved from the slow tier to the fast tier (accessed while slow).
	pub promotions: u64,
	/// Objects moved from the fast tier to the slow tier (fast tier pressure).
	pub demotions: u64,
	/// Objects permanently removed from the slow tier (cache capacity exhausted).
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
