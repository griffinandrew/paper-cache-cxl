/*
 * Copyright (c) Kia Shakiba
 *
 * This source code is licensed under the GNU AGPLv3 license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! A single, feature-neutral stats snapshot shared by every hybrid-cache
//! design, plus `PaperCache::hybrid_stats()` / `AtomicStatus::hybrid_stats()`
//! to read it.
//!
//! Every one of the hybrid designs already exposes its own accessor
//! (`lru_hybrid_stats()`, `s3_fifo_ghost_hybrid_stats()`, ...) returning its
//! own named struct (`LruHybridStats`, `S3FifoGhostHybridStats`, ...). Those
//! stay exactly as they are — `paper-server` and existing callers keep the
//! names they already use, and designs that track *extra* fields keep them
//! (`LruSizedHybridStats` carries 8 per-size-segment gauges on top of the
//! shared 7).
//!
//! The problem this solves is for a *consumer* that doesn't care which design
//! it was built against: `paper-benchmark-cxl` builds one binary per hybrid
//! feature and wants to report demotions/promotions/evictions from whichever
//! one is active. Without this, every such consumer needs its own
//! 15-arm `#[cfg]` cascade naming all 15 accessors and all 15 struct types,
//! duplicated at every call site, and needs a new arm added every time a
//! design is added here. With it, the cascade lives once, in
//! `AtomicStatus::hybrid_stats` (`status.rs`), next to the fields it reads.
//!
//! The seven fields below are exactly the set every design tracks — verified
//! against all 15 `*_hybrid_cache/stats.rs` structs, which share these seven
//! names identically. Design-specific extras are deliberately *not* here;
//! read them from the design's own accessor when you know which design you
//! have.

/// Feature-neutral snapshot of the active hybrid cache's tier-movement
/// counters and live tier gauges.
///
/// The three counters are monotonic totals since the cache was created (or
/// since the last `wipe()`); the four gauges are point-in-time readings of
/// the active policy stack's own bookkeeping, republished by
/// `PolicyWorker::refresh_tier_gauges` once per event-loop pass and so up to
/// one polling interval stale.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct HybridStats {
	/// Total slow→fast tier migrations (objects physically moved into DRAM).
	pub promotions: u64,

	/// Total fast→slow tier migrations (objects physically moved into PMEM).
	pub demotions: u64,

	/// Total terminal evictions — objects removed from the cache entirely,
	/// as opposed to moved between tiers.
	pub evictions: u64,

	/// Bytes currently accounted to the fast (DRAM) tier.
	pub fast_bytes_used: u64,

	/// Bytes currently accounted to the slow (PMEM) tier.
	pub slow_bytes_used: u64,

	/// Objects currently in the fast (DRAM) tier.
	pub fast_objects: u64,

	/// Objects currently in the slow (PMEM) tier.
	pub slow_objects: u64,
}

impl HybridStats {
	/// Total objects tracked across both tiers.
	pub fn total_objects(&self) -> u64 {
		self.fast_objects + self.slow_objects
	}

	/// Total bytes accounted across both tiers.
	pub fn total_bytes_used(&self) -> u64 {
		self.fast_bytes_used + self.slow_bytes_used
	}
}
