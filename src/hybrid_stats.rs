/*
 * Copyright (c) Kia Shakiba
 *
 * This source code is licensed under the GNU AGPLv3 license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! The single stats snapshot every hybrid design reports through, plus
//! `PaperCache::hybrid_stats()` / `AtomicStatus::hybrid_stats()` to read it.
//!
//! There is one accessor, not one per design. The 18 `<Design>HybridStats`
//! names each design's module re-exports are type *aliases* of the struct
//! below, kept so existing callers (`paper-server`, `paper-benchmark-cxl`)
//! compile unchanged; they are the same type and carry no extra fields.
//!
//! That matters for a consumer that does not care which design it is talking
//! to. Before the runtime-policy unification, reporting
//! demotions/promotions/evictions meant a `#[cfg]` cascade naming every
//! accessor and every struct type, repeated at each call site and needing a
//! new arm per design added. Now the cascade lives once, in
//! `AtomicStatus::hybrid_stats` (`status.rs`), next to the fields it reads.
//!
//! The 15 fields below are 3 monotonic counters, 4 two-tier gauges, and 8
//! size-split gauges that only `LruSizedHybrid` ever populates -- they read
//! zero under every other design.

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

	/// Four-segment (small/large x fast/slow) gauges. Populated only by the
	/// size-split `lru_sized` design; zero for every other policy.
	pub small_fast_bytes_used: u64,
	pub large_fast_bytes_used: u64,
	pub small_slow_bytes_used: u64,
	pub large_slow_bytes_used: u64,
	pub small_fast_objects: u64,
	pub large_fast_objects: u64,
	pub small_slow_objects: u64,
	pub large_slow_objects: u64,
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
