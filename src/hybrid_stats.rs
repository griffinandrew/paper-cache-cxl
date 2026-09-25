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
//! S2, 8 readings of the PHYSICAL fast tier and its budget: PHYS_FAST and its
//! peak, `effective_fast_capacity`, the over-budget integral, hits by serving
//! tier, and the live tiered-cache and flat-fast-cache counts. Four of those
//! are PROCESS-GLOBAL (`phys_fast_bytes`, `phys_fast_bytes_max`,
//! `live_tiered_caches`, `live_flat_fast_caches`); see each field. Since S3,
//! 4 more process-global totals -- the corrective migrations the policy
//! worker's reconcile queued, by reason (`reconcile_set_*`,
//! `reconcile_get_to_fast`) -- a fifth, the slow-served hits whose heal a
//! busy in-flight bucket skipped (`reconcile_get_heal_skipped`), and 2
//! counters of this cache's own: the correctives that LANDED
//! (`reconcile_applied_*`), which are not promotions or demotions. Since S5a,
//! 3 readings of the cache's MEASURED DRAM metadata: M
//! (`dram_metadata_bytes`), its structures on the slow node beside it, and
//! `F - M` -- beside the modelled `fast_metadata_bytes` and
//! `effective_fast_capacity`, which they do not replace.

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
	/// Total slow→fast tier migrations (objects physically moved into DRAM)
	/// the policy stack DECIDED. A corrective -- a move to where the stack
	/// already placed the key -- is not one: see `reconcile_applied_to_fast`.
	pub promotions: u64,

	/// Total fast→slow tier migrations (objects physically moved into PMEM)
	/// the policy stack DECIDED, as far as the design counts them (the
	/// LFU-style design counts its settle's demotions when it decides them,
	/// not its admission-to-slow moves). A corrective is not one: see
	/// `reconcile_applied_to_slow`.
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
	/// readers' snapshots. PROCESS-GLOBAL: every fast value in the process, a
	/// flat fast cache's included, so it describes this cache only while
	/// `live_tiered_caches == 1` and `live_flat_fast_caches == 0`.
	pub phys_fast_bytes: u64,

	/// Peak of `phys_fast_bytes` seen at a shard fold or a policy-worker pass:
	/// a lower bound on the true peak up to one sample's torn read -- a burst
	/// between two samples is missed, and a sample, not being a snapshot, can
	/// over-state by the bytes charged and refunded while it reads (see
	/// `phys`'s `Counter::max`). Process-global, and never reset outside tests
	/// (`phys::reset_fast_bytes_max`).
	pub phys_fast_bytes_max: u64,

	/// `F - L * omega`: the fast tier's budget for value bytes once the
	/// per-object reservation is taken off. This cache's; reporting only.
	pub effective_fast_capacity: u64,

	/// The integral over time of `max(0, phys_fast_bytes + L * omega - F)`,
	/// in whole byte-seconds, accumulated by this cache's policy worker once
	/// per pass since the cache was built (a `wipe()` does not reset it). A
	/// right Riemann sum over the worker's poll intervals: it over-states an
	/// excursion's head and can drop its tail, each by up to one interval (1
	/// ms while sets are recent, 1 s otherwise) -- see
	/// `phys::over_budget_increment`. Its integrand is not `phys_fast_bytes -
	/// effective_fast_capacity` once `L * omega > F`, where the latter
	/// saturates at 0.
	pub over_budget_byte_seconds: u64,

	/// Hits served from the fast / the slow tier: the tier of the value a
	/// `get`/`get_into` hit copied. Together they are the status' hit count,
	/// and `wipe()` resets them with it.
	pub fast_hits: u64,
	pub slow_hits: u64,

	/// Tiered caches alive in this process (`phys::live_tiered_caches`).
	pub live_tiered_caches: u64,

	/// Flat caches whose values are fast (`PaperCache<K, BufferDRAM>`) alive
	/// in this process (`phys::live_flat_fast_caches`). Their values are in
	/// `phys_fast_bytes` too.
	pub live_flat_fast_caches: u64,
	/// M (S5a): the bytes this cache's own DRAM metadata structures hold, in
	/// jemalloc's usable-size unit -- the object map's (its tables, arrays and
	/// `Arc`), the policy stack's (slab chunks and their table, index, free
	/// list, bucket maps, ghost, and its box) and one value header per live
	/// object -- counted from the structures, where `fast_metadata_bytes` is
	/// the modelled reservation (`L * omega`, plus the ghost reservation of
	/// the seven designs that make one). The policy worker's publication
	/// (`AtomicStatus::dram_metadata_bytes`), up to one pass behind the map.
	/// This cache's own; reporting only.
	pub dram_metadata_bytes: u64,
	/// This cache's structures on the SLOW node, which M leaves out: the
	/// eviction stacks under `eviction_stacks_pmem`, the object table under
	/// `global_hashtable_pmem`. 0 in every other build.
	pub slow_metadata_bytes: u64,
	/// `F - M`, saturating: `effective_fast_capacity` with the measured
	/// metadata in place of the modelled. Reporting only.
	pub effective_fast_capacity_measured: u64,

	/// Corrective migrations the policy worker's reconcile QUEUED: a `set`
	/// whose value was built in the slow tier for a key the stack places
	/// fast, the reverse, and a hit served from the slow tier on a key the
	/// stack places fast (the heal). Intents: one that finds the bytes
	/// already moved is declined by its consumer. PROCESS-GLOBAL totals, the
	/// MIGSTATS line's `reconcile_*` fields.
	pub reconcile_set_to_fast: u64,
	pub reconcile_set_to_slow: u64,
	pub reconcile_get_to_fast: u64,

	/// Correctives the NEW-KEY RULE alone queued: a key (re-)admitted as new
	/// while a migration of its bucket was in flight, or had landed after its
	/// value was published, whose value was built where the stack places it
	/// -- queued only to land LAST, behind any stale entry for the key. An
	/// intent, PROCESS-GLOBAL like the three above; the MIGSTATS line's
	/// `reconcile_set_new_key`. Under a BACKLOG the rule fires for most fresh
	/// sets: with D entries in flight a bucket is busy with probability about
	/// 1 - e^(-D/16384) (63% at D = 16k, 95% at 50k), so this then counts
	/// mostly correctives that decline -- each handed to the consumers, and
	/// in the `PENDING_*` gauges, like any entry.
	pub reconcile_set_new_key: u64,

	/// Hits served from the slow tier whose HEAL was skipped because
	/// something of the key's in-flight bucket was busy: the worker read no
	/// placement and queued nothing. An UPPER BOUND on the heals skipped -- a
	/// hit on a key placed slow, or on a key whose own promotion is what is
	/// in flight, needed none. PROCESS-GLOBAL like the four above; the
	/// MIGSTATS reconcile line's `reconcile_get_heal_skipped`, its last field
	/// but `t_ms`. Under the backlog above it is most slow-served hits, and
	/// heals are effectively off until the backlog drains.
	pub reconcile_get_heal_skipped: u64,

	/// Reconcile-origin migrations -- the worker's correctives, in every
	/// store -- that LANDED, i.e. moved a value's
	/// bytes, by destination. THIS cache's totals since its creation or its
	/// last `wipe()`, like `promotions` and `demotions`, and never counted in
	/// them: a corrective moves bytes to where the stack ALREADY placed the
	/// key and displaces nothing, so it is not a promotion or a demotion in
	/// the paper's sense. So every completed move into DRAM is `promotions +
	/// reconcile_applied_to_fast`, and every completed move out of it is
	/// `demotions + reconcile_applied_to_slow` -- except under the LFU-style
	/// design, whose `demotions` count its settle's decisions rather than
	/// completed moves (see `demotions`). One that found the bytes already
	/// moved (declined), the key gone, or its value replaced mid-copy
	/// (superseded) is not counted. The MIGSTATS line's
	/// `reconcile_applied_*` are the process-global totals.
	pub reconcile_applied_to_fast: u64,
	pub reconcile_applied_to_slow: u64,
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
