/*
 * Copyright (c) Kia Shakiba
 *
 * This source code is licensed under the GNU AGPLv3 license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! The single stats snapshot every hybrid design reports through, plus
//! `PaperCache::hybrid_stats()` / `AtomicStatus::hybrid_stats()` to read it.
//!
//! There is one accessor, not one per design. Every design reports through
//! the single struct below; the per-design `<Design>HybridStats` aliases that
//! each design's own module used to re-export are gone.
//!
//! That matters for a consumer that does not care which design it is talking
//! to. Before the runtime-policy unification, reporting
//! demotions/promotions/evictions meant a `#[cfg]` cascade naming every
//! accessor and every struct type, repeated at each call site and needing a
//! new arm per design added. Now the cascade lives once, in
//! `AtomicStatus::hybrid_stats` (`status.rs`), next to the fields it reads.
//!
//! The fields below are 3 monotonic tier-movement counters, 5 two-tier gauges
//! (fast/slow bytes and objects, and the metadata reservation), 8 size-split
//! gauges that only `LruSizedCompactHybrid` ever populates -- they read zero
//! under every other design -- and, since the fast-tier backpressure plan's
//! S2, 7 readings of the PHYSICAL fast tier and its budget: PHYS_FAST and its
//! peak, `effective_fast_capacity`, the over-budget integral, hits by serving
//! tier, and the live tiered-cache count. Three of those are PROCESS-GLOBAL
//! (`phys_fast_bytes`, `phys_fast_bytes_max`, `live_tiered_caches`); see each
//! field.

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

	/// DRAM reserved for shared per-object metadata -- the object hashtable,
	/// eviction stacks and `Arc` headers -- across *both* tiers, since that
	/// metadata is DRAM-resident whichever tier an object's value is in.
	///
	/// `fast_bytes_used` counts object bytes only. The fast tier's real DRAM
	/// footprint is the two added together, and it is this term that decides
	/// how many objects fit: at 196 B/object a 4 GiB tier saturates on
	/// metadata alone at ~21.9 M objects, whatever their size.
	pub fast_metadata_bytes: u64,

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

	/// PHYS_FAST (`paper_cache::phys`): bytes PHYSICALLY allocated in the fast
	/// tier's value pool right now, in the stacks' own per-object unit --
	/// where `fast_bytes_used` is the stacks' INTENT. Includes unsettled sets,
	/// pending demotions, in-flight migration copies, superseded copies and
	/// readers' snapshots. PROCESS-GLOBAL: every fast value of every cache in
	/// the process, so it describes this cache only while
	/// `live_tiered_caches == 1`.
	pub phys_fast_bytes: u64,

	/// Peak of `phys_fast_bytes` seen at a shard fold or a policy-worker pass:
	/// a LOWER BOUND on the true peak. Process-global and never reset.
	pub phys_fast_bytes_max: u64,

	/// `F - L * omega`: the fast tier's budget for value bytes once the
	/// per-object reservation is taken off. This cache's; reporting only.
	pub effective_fast_capacity: u64,

	/// The integral over time of `max(0, phys_fast_bytes + L * omega - F)`,
	/// in whole byte-seconds, accumulated by this cache's policy worker once
	/// per pass since the cache was built (a `wipe()` does not reset it).
	pub over_budget_byte_seconds: u64,

	/// Hits served from the fast / the slow tier: the tier of the value a
	/// `get`/`get_into` hit copied. Together they are the status' hit count,
	/// and `wipe()` resets them with it.
	pub fast_hits: u64,
	pub slow_hits: u64,

	/// Tiered caches alive in this process (`phys::live_tiered_caches`).
	pub live_tiered_caches: u64,
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
