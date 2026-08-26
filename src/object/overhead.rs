/*
 * Copyright (c) Kia Shakiba
 *
 * This source code is licensed under the GNU AGPLv3 license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::{
	mem,
	time::Instant,
};

use typesize::TypeSize;

use crate::{
	StatusRef,
	policy::PaperPolicy,
	object::{Object, ObjectSize},
};

pub struct OverheadManager {
	status: StatusRef,
}

impl OverheadManager {
	pub fn new(status: &StatusRef) -> Self {
		OverheadManager {
			status: status.clone(),
		}
	}

	/// Returns the size of the object including non-policy-related overheads.
	/// The part of `base_size` that stays in DRAM whichever tier the object is in.
	///
	/// `Object::set_data` replaces only the value buffer, so a migration moves the
	/// value and nothing else. The key and the expiry live inline in the object map
	/// -- which is DRAM -- and a set TTL additionally owns an entry in `Expiries`,
	/// also DRAM. None of that moves, so none of it belongs in `fast_used` /
	/// `slow_used`.
	///
	/// The key and expiry are moreover already inside `shared_overhead`: the
	/// empirically fitted `OBJECT_MAP_ENTRY_OVERHEAD` was derived from the 40-byte
	/// `(HashedKey, Object)` pair, and `Object` is `{key, Arc ptr, expiry}`. Before
	/// this existed they were charged twice -- once here, once there.
	///
	/// Equals `base_size(object) - value bytes`, computed without touching the
	/// value so the set path does not pay an `Arc` clone.
	pub fn dram_resident_size<K, V>(&self, object: &Object<K, V>) -> ObjectSize
	where
		K: TypeSize,
		V: TypeSize,
	{
		let mut resident =
			object.key_size() + mem::size_of::<crate::object::ExpireTime>() as ObjectSize;

		if object.expiry().is_some() {
			resident += get_ttl_overhead();
		}

		resident
	}

	pub fn base_size<K, V>(&self, object: &Object<K, V>) -> ObjectSize
	where
		K: TypeSize,
		V: TypeSize,
	{
		// The value is counted as the bytes jemalloc actually commits, asked of
		// the allocator rather than estimated -- see `resident_value_bytes`.
		// Before this it was counted as the bytes *requested*, so a tier sized
		// to its accounted bytes overran.
		//
		// The key and expiry are deliberately NOT scaled here: they are inside
		// `shared_overhead`, which applies its own factor to them already, and
		// scaling them twice is exactly the double-charge this module has been
		// untangling elsewhere.
		let value = resident_value_bytes(object.data_size());
		let mut total_size = object.key_size()
			+ value
			+ mem::size_of::<crate::object::ExpireTime>() as ObjectSize;

		if object.expiry().is_some() {
			total_size += get_ttl_overhead();
		}

		total_size
	}

	/// Returns the size of the object including base and policy-related overheads.
	pub fn total_size<K, V>(&self, object: &Object<K, V>) -> ObjectSize
	where
		K: TypeSize,
		V: TypeSize,
	{
		let policy = self.status.policy();
		self.base_size(object) + get_policy_overhead(&policy)
	}
}

/// Returns the per-object policy overhead.
pub fn get_policy_overhead(policy: &PaperPolicy) -> ObjectSize {
	// the overheads are just rough estimates of the number of bytes per object

	match policy {
		PaperPolicy::Auto => 0,

		// 24 bytes for the HashMap entry 48 bytes for the HashList entry,
		// 8 bytes for the HashedKey, 4 bytes for the count
		PaperPolicy::Lfu => 24 + 48 + 8 + 4,

		// 48 bytes for the HashList entry, 8 bytes for the HashedKey
		PaperPolicy::Fifo => 48 + 8,

		// 48 bytes for the HashList entry, 8 bytes for the HashedKey,
		// 1 byte for the visited flag
		PaperPolicy::Clock => 48 + 8 + 1,

		// 48 bytes for the HashList entry, 8 bytes for the HashedKey,
		// 1 byte for the visited flag
		PaperPolicy::Sieve => 48 + 8 + 1,

		// 48 bytes for the HashList entry, 8 bytes for the HashedKey
		PaperPolicy::Lru => 48 + 8,

		// 48 bytes for the HashList entry, 8 bytes for the HashedKey
		PaperPolicy::Mru => 48 + 8,

		// 48 bytes for the HashList entry, 8 bytes for the HashedKey,
		// 4 bytes for the object size
		PaperPolicy::TwoQ(_, _) => 48 + 8 + 4,

		// 48 bytes for the HashList entry, 8 bytes for the HashedKey,
		// 4 bytes for the object size
		PaperPolicy::Arc => 48 + 8 + 4,

		// 48 bytes for the HashList entry, 8 bytes for the HashedKey,
		// 4 bytes for the object size, 1 byte for the frequency count
		PaperPolicy::SThreeFifo(_) => 48 + 8 + 4 + 1,

		// 48 bytes for the HashList entry, 8 bytes for the HashedKey,
		// 24 bytes for the single combined per-key `entries` HashMap entry
		// (tier + size, one map — see `LruHybridStack`'s module doc for why
		// this collapsed from two separate maps), 1 byte for the Tier tag,
		// 4 bytes for the object size
		PaperPolicy::LruHybrid => 48 + 8 + 24 + 1 + 4,

		// Structurally identical to LruHybrid, and deliberately so: the fast
		// tier is the same 48-byte HashList entry + 8-byte HashedKey, and a
		// slow-tier key occupies one CountStack HashList entry instead --
		// never both, since a key is in exactly one tier's structure at a
		// time. The single combined `entries` map charge (24) is unchanged,
		// and the added frequency counter is a u16 that packs into the
		// existing padding of `LruLfuEntry { size: u32, freq: u16, tier: u8 }`
		// -- 7 bytes padded to 8, exactly what `LruEntry { tier, size }`
		// already measured. See `lru_lfu_hybrid_stack.rs`'s "Why the counter
		// is capped" section and its `entry_packs_to_eight_bytes` test, which
		// is what keeps this arm honest.
		PaperPolicy::LruLfuHybrid(_) => 48 + 8 + 24 + 1 + 4,

		// Base LFU overhead (24 HashMap entry + 48 bucket-list entry + 8
		// HashedKey + 4 count = 84) plus what LfuHybridStack needs beyond
		// plain LfuStack: a single combined per-key `entries` HashMap entry
		// (tier + size, one map — see `LfuHybridStack`'s module doc) — 24
		// bytes for the entry, 1 byte for the Tier tag, 4 bytes for the
		// object size (matching the "+4" charge already used for
		// TwoQ/Arc/SThreeFifo)
		PaperPolicy::LfuHybrid => (24 + 48 + 8 + 4) + (24 + 1 + 4),

		// One 32-byte slab slot plus a 16-byte index entry. No `entries` map
		// and no per-key list node: the slot the index returns already carries
		// tier, size and frequency. Measured 47.4 B/key against this 48.
		PaperPolicy::LruCompactHybrid => 24 + 16,
		PaperPolicy::LfuCompactHybrid => 32 + 16,

		// Worst-case charge for a key resident in main_stack as Fast:
		// 48-byte HashList entry + 8-byte HashedKey + a single combined
		// per-key `entries` HashMap entry (queue + tier + size, one map —
		// see `TwoQHybridStack`'s module doc for why this collapsed from
		// three separate maps) — 24 bytes for the entry, 1 byte for the
		// Queue tag, 1 byte for the Option<Tier> tag (only meaningful for
		// keys currently in Main), 4 bytes for the object size
		PaperPolicy::TwoQCompactHybrid(_) => 16 + 24,
		PaperPolicy::TwoQHybrid(_) => (48 + 8) + (24 + 1 + 1 + 4),

		// Structurally identical to TwoQHybrid: `TwoQFastAdmissionHybridStack`
		// is the same two-list/one-combined-entry-map shape, differing only in
		// which physical tier the one-access FIFO queue's bytes live in (fast
		// rather than slow) — a placement decision that costs no extra
		// per-key metadata.
		PaperPolicy::TwoQFastAdmissionHybrid(_) => (48 + 8) + (24 + 1 + 1 + 4),

		// Structurally identical again: the reprieve variant changes where an
		// aged-out one-access key goes, not what is tracked per key.
		PaperPolicy::TwoQFastAdmissionReprieveHybrid(_) => (48 + 8) + (24 + 1 + 1 + 4),

		// Structurally identical again, despite the third queue: a key is
		// resident in exactly one of `a1_in`/`a1_out`/`am` at any moment, so
		// it still costs one HashList entry plus one combined `entries` row
		// (queue tag + Option<Tier> tag + size). No reference bit, and no
		// ghost list -- `a1_out` holds the real objects.
		PaperPolicy::TwoQFullFastAdmissionHybrid(_, _) => (48 + 8) + (24 + 1 + 1 + 4),

		// Structurally identical to LruHybrid: 48 bytes for the HashList
		// entry, 8 bytes for the HashedKey, 24 bytes for the single
		// combined per-key `entries` HashMap entry (tier + size, one map —
		// see `FifoHybridStack`'s module doc), 1 byte for the Tier tag,
		// 4 bytes for the object size.
		PaperPolicy::FifoHybrid => 48 + 8 + 24 + 1 + 4,

		// Structurally identical to LruHybrid despite having 4 recency
		// lists instead of 1: a key is only ever resident in exactly ONE of
		// {small_fast, large_fast, small_slow, large_slow} at a time, so
		// only one 48-byte HashList entry is ever charged, and the
		// 4-variant `SizeQueue` tag still fits in the same 1 byte `Tier`'s
		// 2-variant tag did. 48 bytes for the one HashList entry the key
		// currently occupies, 8 bytes for the HashedKey, 24 bytes for the
		// single combined `entries: HashMap<HashedKey, SizedEntry>` entry
		// (SizedEntry { queue: SizeQueue, size: ObjectSize }), 1 byte for
		// the SizeQueue tag, 4 bytes for the object size.
		PaperPolicy::LruSizedHybrid => 48 + 8 + 24 + 1 + 4,

		// Structurally identical to TwoQHybrid's charge (same shape: a
		// one-access queue + a segmented main FIFO queue, one combined
		// per-key `entries` HashMap entry — see `S3FifoHybridStack`'s
		// module doc): 48 bytes for the HashList entry, 8 bytes for the
		// HashedKey, 24 bytes for the combined entry, 1 byte for the Queue
		// tag, 1 byte for the Option<Tier> tag (only meaningful for keys
		// currently in Main), 4 bytes for the object size, plus 1 more byte
		// than TwoQHybrid for the `accessed: bool` reference bit (only
		// meaningful for keys currently in Main — see that field's doc).
		PaperPolicy::S3FifoCompactHybrid(_) => 16 + 24,
		PaperPolicy::S3FifoHybrid(_) => (48 + 8) + (24 + 1 + 1 + 4 + 1),

		// Ghost-hybrid variants: identical per-*tracked*-object charge to
		// their non-ghost counterparts. The ghost list's own memory isn't
		// charged here at all, matching this crate's existing precedent for
		// `SThreeFifo`'s plain (non-hybrid) ghost queue above -- a ghost
		// entry only ever exists for a key that has already been evicted
		// (no longer counted in `num_objects`, which is what this whole
		// function's result gets multiplied by), so it isn't a *tracked*
		// object's overhead to add to in the first place.
		PaperPolicy::TwoQGhostHybrid(_) => (48 + 8) + (24 + 1 + 1 + 4),
		PaperPolicy::S3FifoGhostHybrid(_) => (48 + 8) + (24 + 1 + 1 + 4 + 1),

		// Identical entry shape to S3FifoGhostHybrid (same S3FifoEntry
		// fields: queue, tier, size, accessed) -- the reference-bit gate
		// this variant adds only changes when the bit is read, not
		// anything about the per-entry bookkeeping shape.
		PaperPolicy::S3FifoGhostLazyDemotionHybrid(_) => (48 + 8) + (24 + 1 + 1 + 4 + 1),


		// Identical entry shape to S3FifoGhostLazyDemotionHybrid (same
		// S3FifoEntry fields) -- moving the one-access queue into the fast
		// tier is a placement/accounting change, not a bookkeeping-shape
		// change.
		PaperPolicy::S3FifoGhostLazyDemotionFastAdmissionHybrid(_) => (48 + 8) + (24 + 1 + 1 + 4 + 1),


		// Identical entry shape to S3FifoGhostLazyDemotionFastAdmissionHybrid
		// (same S3FifoEntry fields) -- the midpoint cursor is a
		// stack-level field (like main_boundary), not a per-object one, so
		// it doesn't change this per-tracked-object charge.
		PaperPolicy::S3FifoGhostLazyDemotionFastAdmissionMidpointHybrid(_) => (48 + 8) + (24 + 1 + 1 + 4 + 1),

		// Same S3FifoEntry shape as S3FifoGhostLazyDemotionFastAdmissionMidpointHybrid,
		// minus the ghost list -- this variant removes it entirely (a
		// one-access key that ages out is spliced into the slow tier of
		// the main queue instead of being evicted, so there's no longer
		// any event that ever populates a ghost entry). No per-tracked-
		// object charge changes either way (the ghost list was never
		// charged per-object to begin with -- see the comment on
		// TwoQGhostHybrid/S3FifoGhostHybrid above), so the number is
		// identical; only the removed list's fixed struct-level cost
		// (irrelevant here, this function is purely per-object) is gone.
		PaperPolicy::S3FifoLazyDemotionFastAdmissionMidpointReprieveHybrid(_) => (48 + 8) + (24 + 1 + 1 + 4 + 1),

		// Same per-object charge as the midpoint variant -- dropping the
		// mid-slow checkpoint removes stack-level fields (a cursor and a
		// drift counter), not per-object ones.
		PaperPolicy::S3FifoLazyDemotionFastAdmissionReprieveHybrid(_) => (48 + 8) + (24 + 1 + 1 + 4 + 1),

		// Identical per-object bookkeeping to the fast-admission reprieve
		// variant above: same `S3FifoEntry { queue, tier, size, accessed }`,
		// same two-list main queue. Moving the one-access queue to the slow
		// tier changes which allocator backs an object's bytes, not what the
		// stack records per key.
		PaperPolicy::S3FifoLazyDemotionReprieveHybrid(_) => (48 + 8) + (24 + 1 + 1 + 4 + 1),

		// Same per-object charge as the predecessor. The slow tier being
		// two physical lists instead of one doesn't change what a tracked
		// object costs -- it's still one list node plus one combined
		// entry -- and this variant actually drops the separate
		// `Option<Tier>` field (the queue tag now carries the tier), so
		// if anything this is a slight over-estimate rather than under.
		PaperPolicy::S3FifoLazyDemotionFastAdmissionSplitSlowReprieveHybrid(_) => (48 + 8) + (24 + 1 + 1 + 4 + 1),
	}
}

pub fn get_ttl_overhead() -> ObjectSize {
	// the size of an Option<Instant> plus 48 bytes for the BTreeMap entry
	mem::size_of::<Option<Instant>>() as ObjectSize + 48
}

// ── Measured building blocks for the hybrid-cache DRAM reservation below ───
//
// Unlike `get_policy_overhead`'s eyeballed round numbers (explicitly
// documented above as "just rough estimates"), the constants in this section
// were derived from `std::mem::size_of` measurements of the *actual*
// concrete types involved (`HashedKey = u64`, `ObjectSize = u32`, the 1-byte
// `Tier` tag, `dlv_list::Index<T>` — 16 bytes regardless of `T`, since it's
// just `{ generation: u64, index: NonMaxUsize }` — and `kwik::HashList`'s
// heap-allocated `Entry<T>` node — `size_of::<T>() + 16` for its two
// intrusive-list pointers), combined with `hashbrown`'s documented ~7/8
// maximum load factor to get a real amortized per-entry cost rather than a
// flat guess. This matters here specifically because `get_policy_overhead`'s
// "48 bytes for the HashList entry" turned out — on inspection — to be
// `size_of::<HashList<..>>()` (the *container's* fixed struct size: a 32-byte
// HashMap header + 2 pointers), not a per-entry cost at all; reusing it
// verbatim (as an earlier version of this function did) inherited that
// mismatch on top of a redundant separate "+8 for the key" charge (the key is
// already stored once, inside the list's heap node).
//
// `hashbrown_entry_cost(raw_pair_size)` — the per-entry cost (amortized over
// load factor, control byte included) of a `hashbrown`-based map (backs
// `std::collections::HashMap`, `dashmap`, and `hashbrown::HashMap` alike)
// storing a pair of `raw_pair_size` bytes — is, at the worst-case point right
// before a table resize (capacity C satisfies `entries <= (7/8) * C`, so
// `C ≈ (8/7) * entries`, each bucket costing `raw_pair_size + 1` bytes: the
// pair plus one control byte): `ceil((8/7) * (raw_pair_size + 1))`. Applied
// below with pair sizes taken from real `size_of::<(HashedKey, V)>()`
// measurements (which include Rust's own alignment padding, e.g. a 9-byte
// `(u64, u8)` logical pair actually occupies 16 bytes).
//
//   HashedKey entry alone (map's own key, e.g. for the shared object
//     hashtable):                     cost(8)  = ceil(9*8/7)   = ceil(10.29) = 11
//   HashMap<HashedKey, Tier>:         cost(16) = ceil(17*8/7)  = ceil(19.43) = 20
//   HashMap<HashedKey, ObjectSize>:   cost(16) = ceil(17*8/7)  = ceil(19.43) = 20
//   HashMap<HashedKey, Index<_>>:     cost(24) = ceil(25*8/7)  = ceil(28.57) = 29
//   kwik HashList<HashedKey> entry:   24 (heap Entry<HashedKey> node:
//                                     8 data + 8 prev + 8 next) + cost(16)
//                                     (internal map slot: two 8-byte
//                                     pointers) = 24 + 20 = 44

/// Approximate per-entry structural overhead of the shared object hashtable
/// (`DashMap`), *beyond* the stored `Object` itself (which `base_size`
/// already accounts for, including any key `K` the `Object` stores
/// internally — see `object/mod.rs`). The map's own key is a *separate*
/// `HashedKey` (the hash of `K`, not `K` itself), so this charges that
/// 8-byte key's own storage plus its amortized hashbrown overhead: `cost(8)
/// = 11` (see the derivation above).
///
/// This deliberately does **not** add load-factor slack proportional to the
/// stored `Object<K, V>`'s own size: this function isn't generic over `K`/
/// `V`, so that size is unknowable here. This makes the hashtable term an
/// under-estimate for large objects (their slack is real DRAM cost this
/// reservation doesn't see), not a safety margin — acceptable given the
/// fast-tier budget is a demotion target, not a hard data-dropping ceiling
/// (see the `lru_hybrid_cache`/`lfu_hybrid_cache` design notes). TODO: if an
/// exact DRAM ceiling is ever needed, thread a `size_of::<Object<K,V>>()`
/// hint through from the generic `PaperCache::new` call site instead.
#[cfg(feature = "hybrid_cache_common")]
#[allow(dead_code)] // superseded by OBJECT_MAP_ENTRY_OVERHEAD; kept for the derivation notes above
pub const HASHTABLE_ENTRY_OVERHEAD: ObjectSize = 11;

/// Dedicated per-object DRAM cost of `LruHybridStack`'s eviction-stack
/// bookkeeping (see the derivation block above): the shared recency list's
/// per-key entry (44) + the combined `entries: HashMap<HashedKey, LruEntry>`
/// entry (20 — `LruEntry { tier, size }` is one `hashbrown`-measured 8-byte
/// value, `cost(16)` for the `(HashedKey, LruEntry)` pair, same as either of
/// the two separate maps this replaced individually cost — see
/// `LruHybridStack`'s module doc for why `tiers`/`sizes` collapsed into one
/// map. That collapse is what dropped this constant from 84 to 64: one of
/// the two 20-byte map-entry charges is simply gone, not re-derived smaller).
///
/// Computed independently of [`get_policy_overhead`]'s `LruHybrid` arm
/// rather than reusing it: that arm is tuned for `used_size`'s DRAM+PMEM
/// budget (where reuse was previously convenient) but, on inspection,
/// double-charges the key (see the derivation block above) — an error that
/// roughly canceled out there, but isn't a reliable basis to build on for a
/// *different* budget with its own correctness requirements.
///
/// MEASURED, not derived: the least-squares slope of jemalloc
/// `stats.allocated` against object count, one point per process at
/// 2^20..2^23 objects, R^2 = 1.0000. See `policy_stack::measure_overhead`.
///
/// ALLOCATED, not resident -- size-class-rounded usable bytes, the quantity
/// `malloc_usable_size` returns and therefore the same quantity Redis reports
/// as `used_memory`. An earlier revision measured RSS instead, which counts
/// retained-but-freed pages that belong in a fragmentation ratio rather than in
/// a per-object cost, and which disagreed with itself by 20% depending on where
/// the sample points fell.
///
/// The field-by-field derivation that used to sit here understated this stack
/// by roughly a third: it counted struct fields and not size-class rounding,
/// index-map load factor, or the growth slack of every doubling structure.
/// Being a measured allocation figure it is NOT multiplied by
/// `resident_factor()` -- see the split in `get_hybrid_dram_shared_overhead`.
#[cfg(feature = "hybrid_cache_common")]
const LRU_HYBRID_EVICTION_STACK_DRAM_OVERHEAD: ObjectSize = 112;

/// Dedicated per-object DRAM cost of `LfuHybridStack`'s eviction-stack
/// bookkeeping: this key's entry in its current chain's internal
/// `HashList<HashedKey>` (44, same derivation as the LRU recency list) + its
/// `index_map: HashMap<HashedKey, Index<CountStack>>` entry (29) + the
/// combined `entries: HashMap<HashedKey, LfuEntry>` entry (20 — same
/// `cost(16)` measurement as LRU's, since `LfuEntry { tier, size }` is also
/// an 8-byte value; see `LfuHybridStack`'s module doc for why `tiers`/
/// `sizes` collapsed into one map, which is what dropped this constant from
/// 113 to 93).
///
/// Does **not** additionally charge for a brand-new `CountStack`/`VecList`
/// bucket node (which would apply if this key were the *only* one at its
/// frequency) — realistic access-frequency distributions are heavily skewed
/// (Zipfian), so most keys share a bucket with others at the same count,
/// making that marginal cost amortize toward zero in aggregate; charging it
/// per-key would model the rare worst case (one key per frequency) as the
/// typical one.
///
/// MEASURED, not derived: the least-squares slope of jemalloc
/// `stats.allocated` against object count, one point per process at
/// 2^20..2^23 objects, R^2 = 1.0000. See `policy_stack::measure_overhead`.
///
/// ALLOCATED, not resident -- size-class-rounded usable bytes, the quantity
/// `malloc_usable_size` returns and therefore the same quantity Redis reports
/// as `used_memory`. An earlier revision measured RSS instead, which counts
/// retained-but-freed pages that belong in a fragmentation ratio rather than in
/// a per-object cost, and which disagreed with itself by 20% depending on where
/// the sample points fell.
///
/// The field-by-field derivation that used to sit here understated this stack
/// by roughly a third: it counted struct fields and not size-class rounding,
/// index-map load factor, or the growth slack of every doubling structure.
/// Being a measured allocation figure it is NOT multiplied by
/// `resident_factor()` -- see the split in `get_hybrid_dram_shared_overhead`.
#[cfg(feature = "hybrid_cache_common")]
const LFU_HYBRID_EVICTION_STACK_DRAM_OVERHEAD: ObjectSize = 168;

/// Per-object DRAM cost of `LruCompactHybridStack`'s eviction stack.
///
/// MEASURED: jemalloc `stats.allocated`, one point per process at 2^20..2^23
/// objects, R^2 = 1.0000. See `policy_stack::measure_overhead`.
///
/// 64 B against `LruHybridStack`'s 112 -- a 42.9% reduction, and below even the
/// 72 B that NON-tiered all-DRAM LRU costs, because a 24-byte slab entry beats
/// a `kwik::HashList` node that jemalloc rounds to the 32-byte size class. The
/// saving is the elimination of the second index: a `HashList` carries its own
/// key->node map AND a separate `entries` map holds the 8-byte payload, one row
/// each per object.
///
/// Measured with `PAPER_DISABLE_SHARED_OVERHEAD=1` so the stack's own
/// pre-reservation stays out of the figure; with it active the slab doubles out
/// of the reservation partway through the range and the fit drops to R^2 0.84.
#[cfg(feature = "hybrid_cache_common")]
const LRU_COMPACT_HYBRID_EVICTION_STACK_DRAM_OVERHEAD: ObjectSize = 64;

/// Per-object DRAM cost of `LfuCompactHybridStack`'s eviction-stack
/// bookkeeping.
///
/// One slab slot (32 -- key 8, prev/next 4 each, freq 4, size 4, tier and
/// resident 1 each, padded) plus one `HashMap<HashedKey, u32>` index entry
/// (16 with hashbrown's slack). There is no third structure: the slot the
/// index returns already carries tier and size, so the `entries` map that
/// `LFU_HYBRID_EVICTION_STACK_DRAM_OVERHEAD`'s trailing `20` pays for does
/// not exist here.
///
/// Unlike every other constant in this block, this one is **measured**:
/// 47.4 B/key as an RSS delta over two million keys, against this model's
/// 48. `FrequencyChain` measured 95.9 against its model of 93.
///
/// MEASURED, not derived: the least-squares slope of jemalloc
/// `stats.allocated` against object count, one point per process at
/// 2^20..2^23 objects, R^2 = 1.0000. See `policy_stack::measure_overhead`.
///
/// ALLOCATED, not resident -- size-class-rounded usable bytes, the quantity
/// `malloc_usable_size` returns and therefore the same quantity Redis reports
/// as `used_memory`. An earlier revision measured RSS instead, which counts
/// retained-but-freed pages that belong in a fragmentation ratio rather than in
/// a per-object cost, and which disagreed with itself by 20% depending on where
/// the sample points fell.
///
/// The field-by-field derivation that used to sit here understated this stack
/// by roughly a third: it counted struct fields and not size-class rounding,
/// index-map load factor, or the growth slack of every doubling structure.
/// Being a measured allocation figure it is NOT multiplied by
/// `resident_factor()` -- see the split in `get_hybrid_dram_shared_overhead`.
#[cfg(feature = "hybrid_cache_common")]
const LFU_COMPACT_HYBRID_EVICTION_STACK_DRAM_OVERHEAD: ObjectSize = 72;

/// Dedicated per-object DRAM cost of `LruSizedHybridStack`'s eviction-stack
/// bookkeeping. Identical derivation and identical value to
/// `LRU_HYBRID_EVICTION_STACK_DRAM_OVERHEAD` despite tracking 4 recency
/// lists instead of 1: a key occupies exactly one list at a time (44 — the
/// list entry it currently sits in) plus its one combined `entries:
/// HashMap<HashedKey, SizedEntry>` entry (20 — `SizedEntry { queue, size }`
/// is, like `LruEntry`, an 8-byte value, `cost(16)` for the pair). Returned
/// as a single total here; `LruSizedHybridStack` is responsible for
/// splitting it proportionally between its two independently-capacitied
/// fast segments (the two slow lists have no capacity to reserve against).
///
/// MEASURED, not derived: the least-squares slope of jemalloc
/// `stats.allocated` against object count, one point per process at
/// 2^20..2^23 objects, R^2 = 1.0000. See `policy_stack::measure_overhead`.
///
/// ALLOCATED, not resident -- size-class-rounded usable bytes, the quantity
/// `malloc_usable_size` returns and therefore the same quantity Redis reports
/// as `used_memory`. An earlier revision measured RSS instead, which counts
/// retained-but-freed pages that belong in a fragmentation ratio rather than in
/// a per-object cost, and which disagreed with itself by 20% depending on where
/// the sample points fell.
///
/// The field-by-field derivation that used to sit here understated this stack
/// by roughly a third: it counted struct fields and not size-class rounding,
/// index-map load factor, or the growth slack of every doubling structure.
/// Being a measured allocation figure it is NOT multiplied by
/// `resident_factor()` -- see the split in `get_hybrid_dram_shared_overhead`.
#[cfg(feature = "hybrid_cache_common")]
const LRU_SIZED_HYBRID_EVICTION_STACK_DRAM_OVERHEAD: ObjectSize = 112;

/// Dedicated per-object DRAM cost of `LruLfuHybridStack`'s eviction-stack
/// bookkeeping. A key is resident in exactly one tier's structure at a time,
/// so this is a worst-case charge over the two:
///
/// - **Fast tier**: the recency list's per-key entry (44) + the combined
///   `entries` map entry (20) = 64, identical to
///   `LRU_HYBRID_EVICTION_STACK_DRAM_OVERHEAD` (the added frequency counter
///   is free — it packs into `LruLfuEntry`'s existing padding).
/// - **Slow tier**: its `CountStack` list entry (44) + the chain's
///   `index_map` entry (29) + the combined `entries` entry (20) = 93,
///   matching `LFU_HYBRID_EVICTION_STACK_DRAM_OVERHEAD`'s derivation.
///
/// The slow-tier figure is the larger, but charging it would over-reserve:
/// this constant is subtracted from the *fast-tier* budget, and the DRAM it
/// is reserving against is dominated by fast-tier residents. Under
/// `eviction_stacks_pmem` the whole term drops out anyway (both structures
/// move to PMEM). Charged at the fast-tier figure, which is what the
/// reservation is actually protecting.
///
/// MEASURED, not derived: the least-squares slope of jemalloc
/// `stats.allocated` against object count, one point per process at
/// 2^20..2^23 objects, R^2 = 1.0000. See `policy_stack::measure_overhead`.
///
/// ALLOCATED, not resident -- size-class-rounded usable bytes, the quantity
/// `malloc_usable_size` returns and therefore the same quantity Redis reports
/// as `used_memory`. An earlier revision measured RSS instead, which counts
/// retained-but-freed pages that belong in a fragmentation ratio rather than in
/// a per-object cost, and which disagreed with itself by 20% depending on where
/// the sample points fell.
///
/// The field-by-field derivation that used to sit here understated this stack
/// by roughly a third: it counted struct fields and not size-class rounding,
/// index-map load factor, or the growth slack of every doubling structure.
/// Being a measured allocation figure it is NOT multiplied by
/// `resident_factor()` -- see the split in `get_hybrid_dram_shared_overhead`.
#[cfg(feature = "hybrid_cache_common")]
const LRU_LFU_HYBRID_EVICTION_STACK_DRAM_OVERHEAD: ObjectSize = 112;

/// Per-*ghost-entry* DRAM cost shared by every hybrid design that keeps a
/// bare-key ghost queue (`TwoQGhostHybrid`, and the four `S3Fifo*Ghost*`
/// variants).
///
/// One `HashList<HashedKey>` node: 24-byte heap `Entry<HashedKey>` (8 data +
/// 8 prev + 8 next) plus the list's internal key->node map slot,
/// `cost(16) = 20`. Same 44 every other list-entry term in this module uses.
///
/// Deliberately *not* a per-tracked-object charge, and so deliberately not
/// part of [`get_hybrid_dram_shared_overhead`]'s return value: a ghost entry
/// exists precisely for a key that is *no longer in the cache*, so there is
/// no tracked-object count to multiply it by and — critically — it has **no
/// object-hashtable slot**, which is why it must never carry
/// [`HASHTABLE_ENTRY_OVERHEAD`]. The owning stacks multiply it by
/// `ghost.len()` inside their own `reserved_overhead`.
///
/// **8 bytes**, not 44: the ghost is a `GhostFilter` — a 4-byte fingerprint
/// plus a 4-byte insertion timestamp, per S3-FIFO's own description of G as
/// "part of the indexing structure". It was a `HashList<HashedKey>`, whose
/// heap `Entry { key, prev, next }` (24) plus index slot (20) cost 44 bytes to
/// hold an 8-byte key, and whose capacity bound was unreachable from the path
/// that populated it: on a no-reuse trace the ghost grew without limit, to
/// 1.94 GB — 45% of a 4 GiB fast tier — on Twitter cluster38.
///
/// Gated on `eviction_stacks_pmem` **only** (never `global_hashtable_pmem`,
/// per the no-hashtable-slot point above): when that feature moves the
/// eviction stacks — ghost list included — to PMEM, the ghost costs the
/// fast/DRAM tier nothing and the term drops to 0.
///
/// Unlike the per-policy constants below this is *not* gated on
/// `hybrid_cache_common`: the policy-stack modules are declared
/// unconditionally (see `worker::policy::policy_stack`), so they compile —
/// and reference this — under every feature combination, including none.
#[cfg(not(feature = "eviction_stacks_pmem"))]
pub const GHOST_ENTRY_DRAM_OVERHEAD: ObjectSize = 8;

/// PMEM-resident ghost list: costs the fast/DRAM tier nothing. See the
/// `not(eviction_stacks_pmem)` arm above for the derivation and rationale.
#[cfg(feature = "eviction_stacks_pmem")]
pub const GHOST_ENTRY_DRAM_OVERHEAD: ObjectSize = 0;

/// Per-object DRAM cost of `FifoHybridStack`'s eviction-stack bookkeeping.
///
/// 44 + 20, the same two-term shape (and, as it happens, the same total) as
/// [`LRU_HYBRID_EVICTION_STACK_DRAM_OVERHEAD`]: one `HashList<HashedKey>`
/// node for the single queue the key occupies at any one time (24-byte heap
/// `Entry` + `cost(16) = 20` internal index slot = 44), plus one slot in the
/// combined `entries` map (20 — the entry struct packs to 8 bytes, so
/// `cost(16)` for the `(HashedKey, Entry)` pair).
///
/// The single-node term is correct rather than an undercount: this design's
/// queues are disjoint and a key is removed from one before being pushed to
/// the other, so it is never resident in two at once. The key itself is
/// stored once, inside the heap node, and is not charged again (the
/// double-charge [`get_policy_overhead`] makes and this module's derivation
/// block flags).
///
/// MEASURED, not derived: the least-squares slope of jemalloc
/// `stats.allocated` against object count, one point per process at
/// 2^20..2^23 objects, R^2 = 1.0000. See `policy_stack::measure_overhead`.
///
/// ALLOCATED, not resident -- size-class-rounded usable bytes, the quantity
/// `malloc_usable_size` returns and therefore the same quantity Redis reports
/// as `used_memory`. An earlier revision measured RSS instead, which counts
/// retained-but-freed pages that belong in a fragmentation ratio rather than in
/// a per-object cost, and which disagreed with itself by 20% depending on where
/// the sample points fell.
///
/// The field-by-field derivation that used to sit here understated this stack
/// by roughly a third: it counted struct fields and not size-class rounding,
/// index-map load factor, or the growth slack of every doubling structure.
/// Being a measured allocation figure it is NOT multiplied by
/// `resident_factor()` -- see the split in `get_hybrid_dram_shared_overhead`.
#[cfg(feature = "hybrid_cache_common")]
const FIFO_HYBRID_EVICTION_STACK_DRAM_OVERHEAD: ObjectSize = 112;

/// Per-object DRAM cost of `TwoQCompactHybridStack`'s eviction stack.
///
/// MEASURED: jemalloc `stats.allocated`, one point per process at 2^20..2^23
/// objects, R^2 = 1.0000. See `policy_stack::measure_overhead`.
///
/// 72 B against `TwoQHybridStack`'s 112 -- a 35.7% reduction. `TwoQHybridStack`
/// keeps THREE indexes for a population where every key is in exactly one of
/// its two queues: a FIFO `HashList` and an LRU `HashList`, each owning its own
/// key-to-node map, plus the separate `entries` map. This keeps one.
///
/// 8 B above `LruCompactHybridStack`'s 64, and that gap is the layout choice
/// rather than the policy: this stack carries the payload in the index value
/// (layout B) where the LRU list carries it in the slab slot (layout A). The
/// standalone comparison measured layout B at +8.01 B/object, so the two
/// results agree to within a byte. B is right here because `mark_accessed` and
/// the queue-dispatch read in `touch` are hot AND touch no queue order.
#[cfg(feature = "hybrid_cache_common")]
const TWO_Q_COMPACT_HYBRID_EVICTION_STACK_DRAM_OVERHEAD: ObjectSize = 72;

/// Per-object DRAM cost of `TwoQHybridStack`'s eviction-stack bookkeeping.
///
/// 44 + 20, the same two-term shape (and, as it happens, the same total) as
/// [`LRU_HYBRID_EVICTION_STACK_DRAM_OVERHEAD`]: one `HashList<HashedKey>`
/// node for the single queue the key occupies at any one time (24-byte heap
/// `Entry` + `cost(16) = 20` internal index slot = 44), plus one slot in the
/// combined `entries` map (20 — the entry struct packs to 8 bytes, so
/// `cost(16)` for the `(HashedKey, Entry)` pair).
///
/// The single-node term is correct rather than an undercount: this design's
/// queues are disjoint and a key is removed from one before being pushed to
/// the other, so it is never resident in two at once. The key itself is
/// stored once, inside the heap node, and is not charged again (the
/// double-charge [`get_policy_overhead`] makes and this module's derivation
/// block flags).
///
/// MEASURED, not derived: the least-squares slope of jemalloc
/// `stats.allocated` against object count, one point per process at
/// 2^20..2^23 objects, R^2 = 1.0000. See `policy_stack::measure_overhead`.
///
/// ALLOCATED, not resident -- size-class-rounded usable bytes, the quantity
/// `malloc_usable_size` returns and therefore the same quantity Redis reports
/// as `used_memory`. An earlier revision measured RSS instead, which counts
/// retained-but-freed pages that belong in a fragmentation ratio rather than in
/// a per-object cost, and which disagreed with itself by 20% depending on where
/// the sample points fell.
///
/// The field-by-field derivation that used to sit here understated this stack
/// by roughly a third: it counted struct fields and not size-class rounding,
/// index-map load factor, or the growth slack of every doubling structure.
/// Being a measured allocation figure it is NOT multiplied by
/// `resident_factor()` -- see the split in `get_hybrid_dram_shared_overhead`.
#[cfg(feature = "hybrid_cache_common")]
const TWO_Q_HYBRID_EVICTION_STACK_DRAM_OVERHEAD: ObjectSize = 112;

/// Per-object DRAM cost of `TwoQFastAdmissionHybridStack`'s eviction-stack bookkeeping.
///
/// 44 + 20, the same two-term shape (and, as it happens, the same total) as
/// [`LRU_HYBRID_EVICTION_STACK_DRAM_OVERHEAD`]: one `HashList<HashedKey>`
/// node for the single queue the key occupies at any one time (24-byte heap
/// `Entry` + `cost(16) = 20` internal index slot = 44), plus one slot in the
/// combined `entries` map (20 — the entry struct packs to 8 bytes, so
/// `cost(16)` for the `(HashedKey, Entry)` pair).
///
/// The single-node term is correct rather than an undercount: this design's
/// queues are disjoint and a key is removed from one before being pushed to
/// the other, so it is never resident in two at once. The key itself is
/// stored once, inside the heap node, and is not charged again (the
/// double-charge [`get_policy_overhead`] makes and this module's derivation
/// block flags).
///
/// MEASURED, not derived: the least-squares slope of jemalloc
/// `stats.allocated` against object count, one point per process at
/// 2^20..2^23 objects, R^2 = 1.0000. See `policy_stack::measure_overhead`.
///
/// ALLOCATED, not resident -- size-class-rounded usable bytes, the quantity
/// `malloc_usable_size` returns and therefore the same quantity Redis reports
/// as `used_memory`. An earlier revision measured RSS instead, which counts
/// retained-but-freed pages that belong in a fragmentation ratio rather than in
/// a per-object cost, and which disagreed with itself by 20% depending on where
/// the sample points fell.
///
/// The field-by-field derivation that used to sit here understated this stack
/// by roughly a third: it counted struct fields and not size-class rounding,
/// index-map load factor, or the growth slack of every doubling structure.
/// Being a measured allocation figure it is NOT multiplied by
/// `resident_factor()` -- see the split in `get_hybrid_dram_shared_overhead`.
#[cfg(feature = "hybrid_cache_common")]
const TWO_Q_FAST_ADMISSION_HYBRID_EVICTION_STACK_DRAM_OVERHEAD: ObjectSize = 112;

/// Per-object DRAM cost of `TwoQFastAdmissionReprieveHybridStack`'s eviction-stack bookkeeping.
///
/// 44 + 20, the same two-term shape (and, as it happens, the same total) as
/// [`LRU_HYBRID_EVICTION_STACK_DRAM_OVERHEAD`]: one `HashList<HashedKey>`
/// node for the single queue the key occupies at any one time (24-byte heap
/// `Entry` + `cost(16) = 20` internal index slot = 44), plus one slot in the
/// combined `entries` map (20 — the entry struct packs to 8 bytes, so
/// `cost(16)` for the `(HashedKey, Entry)` pair).
///
/// The single-node term is correct rather than an undercount: this design's
/// queues are disjoint and a key is removed from one before being pushed to
/// the other, so it is never resident in two at once. The key itself is
/// stored once, inside the heap node, and is not charged again (the
/// double-charge [`get_policy_overhead`] makes and this module's derivation
/// block flags).
///
/// MEASURED, not derived: the least-squares slope of jemalloc
/// `stats.allocated` against object count, one point per process at
/// 2^20..2^23 objects, R^2 = 1.0000. See `policy_stack::measure_overhead`.
///
/// ALLOCATED, not resident -- size-class-rounded usable bytes, the quantity
/// `malloc_usable_size` returns and therefore the same quantity Redis reports
/// as `used_memory`. An earlier revision measured RSS instead, which counts
/// retained-but-freed pages that belong in a fragmentation ratio rather than in
/// a per-object cost, and which disagreed with itself by 20% depending on where
/// the sample points fell.
///
/// The field-by-field derivation that used to sit here understated this stack
/// by roughly a third: it counted struct fields and not size-class rounding,
/// index-map load factor, or the growth slack of every doubling structure.
/// Being a measured allocation figure it is NOT multiplied by
/// `resident_factor()` -- see the split in `get_hybrid_dram_shared_overhead`.
#[cfg(feature = "hybrid_cache_common")]
const TWO_Q_FAST_ADMISSION_REPRIEVE_HYBRID_EVICTION_STACK_DRAM_OVERHEAD: ObjectSize = 112;

/// Per-object DRAM cost of `TwoQFullFastAdmissionHybridStack`'s eviction-stack bookkeeping.
///
/// 44 + 20, unchanged from its two-queue siblings despite the third queue:
/// one `HashList<HashedKey>` node for the single queue the key occupies at
/// any one time (24-byte heap `Entry` + `cost(16) = 20` internal index slot
/// = 44), plus one slot in the combined `entries` map (20 — the entry
/// struct packs to 8 bytes, so `cost(16)` for the `(HashedKey, Entry)`
/// pair).
///
/// The single-node term stays correct: `a1_in`, `a1_out` and `am` are
/// disjoint, and a key is removed from one before being pushed to the next,
/// so it is never resident in two at once. There is no ghost list to charge
/// for either — `a1_out` holds real resident objects, which are already
/// counted as tracked keys.
///
/// MEASURED, not derived: the least-squares slope of jemalloc
/// `stats.allocated` against object count, one point per process at
/// 2^20..2^23 objects, R^2 = 1.0000. See `policy_stack::measure_overhead`.
///
/// ALLOCATED, not resident -- size-class-rounded usable bytes, the quantity
/// `malloc_usable_size` returns and therefore the same quantity Redis reports
/// as `used_memory`. An earlier revision measured RSS instead, which counts
/// retained-but-freed pages that belong in a fragmentation ratio rather than in
/// a per-object cost, and which disagreed with itself by 20% depending on where
/// the sample points fell.
///
/// The field-by-field derivation that used to sit here understated this stack
/// by roughly a third: it counted struct fields and not size-class rounding,
/// index-map load factor, or the growth slack of every doubling structure.
/// Being a measured allocation figure it is NOT multiplied by
/// `resident_factor()` -- see the split in `get_hybrid_dram_shared_overhead`.
#[cfg(feature = "hybrid_cache_common")]
const TWO_Q_FULL_FAST_ADMISSION_HYBRID_EVICTION_STACK_DRAM_OVERHEAD: ObjectSize = 112;

/// Per-object DRAM cost of `TwoQGhostHybridStack`'s eviction-stack bookkeeping.
///
/// 44 + 20, the same two-term shape (and, as it happens, the same total) as
/// [`LRU_HYBRID_EVICTION_STACK_DRAM_OVERHEAD`]: one `HashList<HashedKey>`
/// node for the single queue the key occupies at any one time (24-byte heap
/// `Entry` + `cost(16) = 20` internal index slot = 44), plus one slot in the
/// combined `entries` map (20 — the entry struct packs to 8 bytes, so
/// `cost(16)` for the `(HashedKey, Entry)` pair).
///
/// The single-node term is correct rather than an undercount: this design's
/// queues are disjoint and a key is removed from one before being pushed to
/// the other, so it is never resident in two at once. The key itself is
/// stored once, inside the heap node, and is not charged again (the
/// double-charge [`get_policy_overhead`] makes and this module's derivation
/// block flags).
///
/// Excludes the ghost queue, which is charged separately via
/// [`GHOST_ENTRY_DRAM_OVERHEAD`] against `ghost.len()`. That is now an
/// 8-byte fingerprint + timestamp bounded by its insertion window, not a
/// 44-byte `HashList` node bounded only by a `trim_ghost` the populating
/// path never called -- which is how a ghost reached 1.94 GB, 45% of a
/// 4 GiB fast tier, on Twitter cluster38. It stays a separate term because
/// ghost entries outlive the tracked keys they came from.
///
/// MEASURED, not derived: the least-squares slope of jemalloc
/// `stats.allocated` against object count, one point per process at
/// 2^20..2^23 objects, R^2 = 1.0000. See `policy_stack::measure_overhead`.
///
/// ALLOCATED, not resident -- size-class-rounded usable bytes, the quantity
/// `malloc_usable_size` returns and therefore the same quantity Redis reports
/// as `used_memory`. An earlier revision measured RSS instead, which counts
/// retained-but-freed pages that belong in a fragmentation ratio rather than in
/// a per-object cost, and which disagreed with itself by 20% depending on where
/// the sample points fell.
///
/// The field-by-field derivation that used to sit here understated this stack
/// by roughly a third: it counted struct fields and not size-class rounding,
/// index-map load factor, or the growth slack of every doubling structure.
/// Being a measured allocation figure it is NOT multiplied by
/// `resident_factor()` -- see the split in `get_hybrid_dram_shared_overhead`.
#[cfg(feature = "hybrid_cache_common")]
const TWO_Q_GHOST_HYBRID_EVICTION_STACK_DRAM_OVERHEAD: ObjectSize = 112;

/// Per-object DRAM cost of `S3FifoCompactHybridStack`'s eviction stack.
///
/// MEASURED: jemalloc `stats.allocated`, one point per process at 2^20..2^23
/// objects, R^2 = 1.0000. See `policy_stack::measure_overhead`.
///
/// 72 B against `S3FifoHybridStack`'s 112 -- a 35.7% reduction -- and identical
/// to the measured `TwoQCompactHybridStack`, which is the expected result:
/// the two share the primitive and both payloads are 8 bytes. Predicted before
/// the run and confirmed by it.
#[cfg(feature = "hybrid_cache_common")]
const S3_FIFO_COMPACT_HYBRID_EVICTION_STACK_DRAM_OVERHEAD: ObjectSize = 72;

/// Per-object DRAM cost of `S3FifoHybridStack`'s eviction-stack bookkeeping.
///
/// 44 + 20, the same two-term shape (and, as it happens, the same total) as
/// [`LRU_HYBRID_EVICTION_STACK_DRAM_OVERHEAD`]: one `HashList<HashedKey>`
/// node for the single queue the key occupies at any one time (24-byte heap
/// `Entry` + `cost(16) = 20` internal index slot = 44), plus one slot in the
/// combined `entries` map (20 — the entry struct packs to 8 bytes, so
/// `cost(16)` for the `(HashedKey, Entry)` pair).
///
/// The single-node term is correct rather than an undercount: this design's
/// queues are disjoint and a key is removed from one before being pushed to
/// the other, so it is never resident in two at once. The key itself is
/// stored once, inside the heap node, and is not charged again (the
/// double-charge [`get_policy_overhead`] makes and this module's derivation
/// block flags).
///
/// MEASURED, not derived: the least-squares slope of jemalloc
/// `stats.allocated` against object count, one point per process at
/// 2^20..2^23 objects, R^2 = 1.0000. See `policy_stack::measure_overhead`.
///
/// ALLOCATED, not resident -- size-class-rounded usable bytes, the quantity
/// `malloc_usable_size` returns and therefore the same quantity Redis reports
/// as `used_memory`. An earlier revision measured RSS instead, which counts
/// retained-but-freed pages that belong in a fragmentation ratio rather than in
/// a per-object cost, and which disagreed with itself by 20% depending on where
/// the sample points fell.
///
/// The field-by-field derivation that used to sit here understated this stack
/// by roughly a third: it counted struct fields and not size-class rounding,
/// index-map load factor, or the growth slack of every doubling structure.
/// Being a measured allocation figure it is NOT multiplied by
/// `resident_factor()` -- see the split in `get_hybrid_dram_shared_overhead`.
#[cfg(feature = "hybrid_cache_common")]
const S3_FIFO_HYBRID_EVICTION_STACK_DRAM_OVERHEAD: ObjectSize = 112;

/// Per-object DRAM cost of `S3FifoGhostHybridStack`'s eviction-stack bookkeeping.
///
/// 44 + 20, the same two-term shape (and, as it happens, the same total) as
/// [`LRU_HYBRID_EVICTION_STACK_DRAM_OVERHEAD`]: one `HashList<HashedKey>`
/// node for the single queue the key occupies at any one time (24-byte heap
/// `Entry` + `cost(16) = 20` internal index slot = 44), plus one slot in the
/// combined `entries` map (20 — the entry struct packs to 8 bytes, so
/// `cost(16)` for the `(HashedKey, Entry)` pair).
///
/// The single-node term is correct rather than an undercount: this design's
/// queues are disjoint and a key is removed from one before being pushed to
/// the other, so it is never resident in two at once. The key itself is
/// stored once, inside the heap node, and is not charged again (the
/// double-charge [`get_policy_overhead`] makes and this module's derivation
/// block flags).
///
/// Excludes the ghost queue, which is charged separately via
/// [`GHOST_ENTRY_DRAM_OVERHEAD`] against `ghost.len()`. That is now an
/// 8-byte fingerprint + timestamp bounded by its insertion window, not a
/// 44-byte `HashList` node bounded only by a `trim_ghost` the populating
/// path never called -- which is how a ghost reached 1.94 GB, 45% of a
/// 4 GiB fast tier, on Twitter cluster38. It stays a separate term because
/// ghost entries outlive the tracked keys they came from.
///
/// MEASURED, not derived: the least-squares slope of jemalloc
/// `stats.allocated` against object count, one point per process at
/// 2^20..2^23 objects, R^2 = 1.0000. See `policy_stack::measure_overhead`.
///
/// ALLOCATED, not resident -- size-class-rounded usable bytes, the quantity
/// `malloc_usable_size` returns and therefore the same quantity Redis reports
/// as `used_memory`. An earlier revision measured RSS instead, which counts
/// retained-but-freed pages that belong in a fragmentation ratio rather than in
/// a per-object cost, and which disagreed with itself by 20% depending on where
/// the sample points fell.
///
/// The field-by-field derivation that used to sit here understated this stack
/// by roughly a third: it counted struct fields and not size-class rounding,
/// index-map load factor, or the growth slack of every doubling structure.
/// Being a measured allocation figure it is NOT multiplied by
/// `resident_factor()` -- see the split in `get_hybrid_dram_shared_overhead`.
#[cfg(feature = "hybrid_cache_common")]
const S3_FIFO_GHOST_HYBRID_EVICTION_STACK_DRAM_OVERHEAD: ObjectSize = 112;

/// Per-object DRAM cost of `S3FifoGhostLazyDemotionHybridStack`'s eviction-stack bookkeeping.
///
/// 44 + 20, the same two-term shape (and, as it happens, the same total) as
/// [`LRU_HYBRID_EVICTION_STACK_DRAM_OVERHEAD`]: one `HashList<HashedKey>`
/// node for the single queue the key occupies at any one time (24-byte heap
/// `Entry` + `cost(16) = 20` internal index slot = 44), plus one slot in the
/// combined `entries` map (20 — the entry struct packs to 8 bytes, so
/// `cost(16)` for the `(HashedKey, Entry)` pair).
///
/// The single-node term is correct rather than an undercount: this design's
/// queues are disjoint and a key is removed from one before being pushed to
/// the other, so it is never resident in two at once. The key itself is
/// stored once, inside the heap node, and is not charged again (the
/// double-charge [`get_policy_overhead`] makes and this module's derivation
/// block flags).
///
/// Excludes the ghost queue, which is charged separately via
/// [`GHOST_ENTRY_DRAM_OVERHEAD`] against `ghost.len()`. That is now an
/// 8-byte fingerprint + timestamp bounded by its insertion window, not a
/// 44-byte `HashList` node bounded only by a `trim_ghost` the populating
/// path never called -- which is how a ghost reached 1.94 GB, 45% of a
/// 4 GiB fast tier, on Twitter cluster38. It stays a separate term because
/// ghost entries outlive the tracked keys they came from.
///
/// MEASURED, not derived: the least-squares slope of jemalloc
/// `stats.allocated` against object count, one point per process at
/// 2^20..2^23 objects, R^2 = 1.0000. See `policy_stack::measure_overhead`.
///
/// ALLOCATED, not resident -- size-class-rounded usable bytes, the quantity
/// `malloc_usable_size` returns and therefore the same quantity Redis reports
/// as `used_memory`. An earlier revision measured RSS instead, which counts
/// retained-but-freed pages that belong in a fragmentation ratio rather than in
/// a per-object cost, and which disagreed with itself by 20% depending on where
/// the sample points fell.
///
/// The field-by-field derivation that used to sit here understated this stack
/// by roughly a third: it counted struct fields and not size-class rounding,
/// index-map load factor, or the growth slack of every doubling structure.
/// Being a measured allocation figure it is NOT multiplied by
/// `resident_factor()` -- see the split in `get_hybrid_dram_shared_overhead`.
#[cfg(feature = "hybrid_cache_common")]
const S3_FIFO_GHOST_LAZY_DEMOTION_HYBRID_EVICTION_STACK_DRAM_OVERHEAD: ObjectSize = 112;

/// Per-object DRAM cost of `S3FifoGhostLazyDemotionFastAdmissionHybridStack`'s eviction-stack bookkeeping.
///
/// 44 + 20, the same two-term shape (and, as it happens, the same total) as
/// [`LRU_HYBRID_EVICTION_STACK_DRAM_OVERHEAD`]: one `HashList<HashedKey>`
/// node for the single queue the key occupies at any one time (24-byte heap
/// `Entry` + `cost(16) = 20` internal index slot = 44), plus one slot in the
/// combined `entries` map (20 — the entry struct packs to 8 bytes, so
/// `cost(16)` for the `(HashedKey, Entry)` pair).
///
/// The single-node term is correct rather than an undercount: this design's
/// queues are disjoint and a key is removed from one before being pushed to
/// the other, so it is never resident in two at once. The key itself is
/// stored once, inside the heap node, and is not charged again (the
/// double-charge [`get_policy_overhead`] makes and this module's derivation
/// block flags).
///
/// Excludes the ghost queue, which is charged separately via
/// [`GHOST_ENTRY_DRAM_OVERHEAD`] against `ghost.len()`. That is now an
/// 8-byte fingerprint + timestamp bounded by its insertion window, not a
/// 44-byte `HashList` node bounded only by a `trim_ghost` the populating
/// path never called -- which is how a ghost reached 1.94 GB, 45% of a
/// 4 GiB fast tier, on Twitter cluster38. It stays a separate term because
/// ghost entries outlive the tracked keys they came from.
///
/// MEASURED, not derived: the least-squares slope of jemalloc
/// `stats.allocated` against object count, one point per process at
/// 2^20..2^23 objects, R^2 = 1.0000. See `policy_stack::measure_overhead`.
///
/// ALLOCATED, not resident -- size-class-rounded usable bytes, the quantity
/// `malloc_usable_size` returns and therefore the same quantity Redis reports
/// as `used_memory`. An earlier revision measured RSS instead, which counts
/// retained-but-freed pages that belong in a fragmentation ratio rather than in
/// a per-object cost, and which disagreed with itself by 20% depending on where
/// the sample points fell.
///
/// The field-by-field derivation that used to sit here understated this stack
/// by roughly a third: it counted struct fields and not size-class rounding,
/// index-map load factor, or the growth slack of every doubling structure.
/// Being a measured allocation figure it is NOT multiplied by
/// `resident_factor()` -- see the split in `get_hybrid_dram_shared_overhead`.
#[cfg(feature = "hybrid_cache_common")]
const S3_FIFO_GHOST_LAZY_DEMOTION_FAST_ADMISSION_HYBRID_EVICTION_STACK_DRAM_OVERHEAD: ObjectSize = 112;

/// Per-object DRAM cost of `S3FifoGhostLazyDemotionFastAdmissionMidpointHybridStack`'s eviction-stack bookkeeping.
///
/// 44 + 20, the same two-term shape (and, as it happens, the same total) as
/// [`LRU_HYBRID_EVICTION_STACK_DRAM_OVERHEAD`]: one `HashList<HashedKey>`
/// node for the single queue the key occupies at any one time (24-byte heap
/// `Entry` + `cost(16) = 20` internal index slot = 44), plus one slot in the
/// combined `entries` map (20 — the entry struct packs to 8 bytes, so
/// `cost(16)` for the `(HashedKey, Entry)` pair).
///
/// The single-node term is correct rather than an undercount: this design's
/// queues are disjoint and a key is removed from one before being pushed to
/// the other, so it is never resident in two at once. The key itself is
/// stored once, inside the heap node, and is not charged again (the
/// double-charge [`get_policy_overhead`] makes and this module's derivation
/// block flags).
///
/// Excludes the ghost queue, which is charged separately via
/// [`GHOST_ENTRY_DRAM_OVERHEAD`] against `ghost.len()`. That is now an
/// 8-byte fingerprint + timestamp bounded by its insertion window, not a
/// 44-byte `HashList` node bounded only by a `trim_ghost` the populating
/// path never called -- which is how a ghost reached 1.94 GB, 45% of a
/// 4 GiB fast tier, on Twitter cluster38. It stays a separate term because
/// ghost entries outlive the tracked keys they came from.
///
/// MEASURED, not derived: the least-squares slope of jemalloc
/// `stats.allocated` against object count, one point per process at
/// 2^20..2^23 objects, R^2 = 1.0000. See `policy_stack::measure_overhead`.
///
/// ALLOCATED, not resident -- size-class-rounded usable bytes, the quantity
/// `malloc_usable_size` returns and therefore the same quantity Redis reports
/// as `used_memory`. An earlier revision measured RSS instead, which counts
/// retained-but-freed pages that belong in a fragmentation ratio rather than in
/// a per-object cost, and which disagreed with itself by 20% depending on where
/// the sample points fell.
///
/// The field-by-field derivation that used to sit here understated this stack
/// by roughly a third: it counted struct fields and not size-class rounding,
/// index-map load factor, or the growth slack of every doubling structure.
/// Being a measured allocation figure it is NOT multiplied by
/// `resident_factor()` -- see the split in `get_hybrid_dram_shared_overhead`.
#[cfg(feature = "hybrid_cache_common")]
const S3_FIFO_GHOST_LAZY_DEMOTION_FAST_ADMISSION_MIDPOINT_HYBRID_EVICTION_STACK_DRAM_OVERHEAD: ObjectSize = 112;

/// Per-object DRAM cost of `S3FifoLazyDemotionReprieveHybridStack`'s eviction-stack bookkeeping.
///
/// 44 + 20, the same two-term shape (and, as it happens, the same total) as
/// [`LRU_HYBRID_EVICTION_STACK_DRAM_OVERHEAD`]: one `HashList<HashedKey>`
/// node for the single queue the key occupies at any one time (24-byte heap
/// `Entry` + `cost(16) = 20` internal index slot = 44), plus one slot in the
/// combined `entries` map (20 — the entry struct packs to 8 bytes, so
/// `cost(16)` for the `(HashedKey, Entry)` pair).
///
/// The single-node term is correct rather than an undercount: this design's
/// queues are disjoint and a key is removed from one before being pushed to
/// the other, so it is never resident in two at once. The key itself is
/// stored once, inside the heap node, and is not charged again (the
/// double-charge [`get_policy_overhead`] makes and this module's derivation
/// block flags).
///
/// MEASURED, not derived: the least-squares slope of jemalloc
/// `stats.allocated` against object count, one point per process at
/// 2^20..2^23 objects, R^2 = 1.0000. See `policy_stack::measure_overhead`.
///
/// ALLOCATED, not resident -- size-class-rounded usable bytes, the quantity
/// `malloc_usable_size` returns and therefore the same quantity Redis reports
/// as `used_memory`. An earlier revision measured RSS instead, which counts
/// retained-but-freed pages that belong in a fragmentation ratio rather than in
/// a per-object cost, and which disagreed with itself by 20% depending on where
/// the sample points fell.
///
/// The field-by-field derivation that used to sit here understated this stack
/// by roughly a third: it counted struct fields and not size-class rounding,
/// index-map load factor, or the growth slack of every doubling structure.
/// Being a measured allocation figure it is NOT multiplied by
/// `resident_factor()` -- see the split in `get_hybrid_dram_shared_overhead`.
#[cfg(feature = "hybrid_cache_common")]
const S3_FIFO_LAZY_DEMOTION_REPRIEVE_HYBRID_EVICTION_STACK_DRAM_OVERHEAD: ObjectSize = 112;

/// Per-object DRAM cost of `S3FifoLazyDemotionFastAdmissionReprieveHybridStack`'s eviction-stack bookkeeping.
///
/// 44 + 20, the same two-term shape (and, as it happens, the same total) as
/// [`LRU_HYBRID_EVICTION_STACK_DRAM_OVERHEAD`]: one `HashList<HashedKey>`
/// node for the single queue the key occupies at any one time (24-byte heap
/// `Entry` + `cost(16) = 20` internal index slot = 44), plus one slot in the
/// combined `entries` map (20 — the entry struct packs to 8 bytes, so
/// `cost(16)` for the `(HashedKey, Entry)` pair).
///
/// The single-node term is correct rather than an undercount: this design's
/// queues are disjoint and a key is removed from one before being pushed to
/// the other, so it is never resident in two at once. The key itself is
/// stored once, inside the heap node, and is not charged again (the
/// double-charge [`get_policy_overhead`] makes and this module's derivation
/// block flags).
///
/// MEASURED, not derived: the least-squares slope of jemalloc
/// `stats.allocated` against object count, one point per process at
/// 2^20..2^23 objects, R^2 = 1.0000. See `policy_stack::measure_overhead`.
///
/// ALLOCATED, not resident -- size-class-rounded usable bytes, the quantity
/// `malloc_usable_size` returns and therefore the same quantity Redis reports
/// as `used_memory`. An earlier revision measured RSS instead, which counts
/// retained-but-freed pages that belong in a fragmentation ratio rather than in
/// a per-object cost, and which disagreed with itself by 20% depending on where
/// the sample points fell.
///
/// The field-by-field derivation that used to sit here understated this stack
/// by roughly a third: it counted struct fields and not size-class rounding,
/// index-map load factor, or the growth slack of every doubling structure.
/// Being a measured allocation figure it is NOT multiplied by
/// `resident_factor()` -- see the split in `get_hybrid_dram_shared_overhead`.
#[cfg(feature = "hybrid_cache_common")]
const S3_FIFO_LAZY_DEMOTION_FAST_ADMISSION_REPRIEVE_HYBRID_EVICTION_STACK_DRAM_OVERHEAD: ObjectSize = 112;

/// Per-object DRAM cost of `S3FifoLazyDemotionFastAdmissionMidpointReprieveHybridStack`'s eviction-stack bookkeeping.
///
/// 44 + 20, the same two-term shape (and, as it happens, the same total) as
/// [`LRU_HYBRID_EVICTION_STACK_DRAM_OVERHEAD`]: one `HashList<HashedKey>`
/// node for the single queue the key occupies at any one time (24-byte heap
/// `Entry` + `cost(16) = 20` internal index slot = 44), plus one slot in the
/// combined `entries` map (20 — the entry struct packs to 8 bytes, so
/// `cost(16)` for the `(HashedKey, Entry)` pair).
///
/// The single-node term is correct rather than an undercount: this design's
/// queues are disjoint and a key is removed from one before being pushed to
/// the other, so it is never resident in two at once. The key itself is
/// stored once, inside the heap node, and is not charged again (the
/// double-charge [`get_policy_overhead`] makes and this module's derivation
/// block flags).
///
/// MEASURED, not derived: the least-squares slope of jemalloc
/// `stats.allocated` against object count, one point per process at
/// 2^20..2^23 objects, R^2 = 1.0000. See `policy_stack::measure_overhead`.
///
/// ALLOCATED, not resident -- size-class-rounded usable bytes, the quantity
/// `malloc_usable_size` returns and therefore the same quantity Redis reports
/// as `used_memory`. An earlier revision measured RSS instead, which counts
/// retained-but-freed pages that belong in a fragmentation ratio rather than in
/// a per-object cost, and which disagreed with itself by 20% depending on where
/// the sample points fell.
///
/// The field-by-field derivation that used to sit here understated this stack
/// by roughly a third: it counted struct fields and not size-class rounding,
/// index-map load factor, or the growth slack of every doubling structure.
/// Being a measured allocation figure it is NOT multiplied by
/// `resident_factor()` -- see the split in `get_hybrid_dram_shared_overhead`.
#[cfg(feature = "hybrid_cache_common")]
const S3_FIFO_LAZY_DEMOTION_FAST_ADMISSION_MIDPOINT_REPRIEVE_HYBRID_EVICTION_STACK_DRAM_OVERHEAD: ObjectSize = 112;

/// Per-object DRAM cost of `S3FifoLazyDemotionFastAdmissionSplitSlowReprieveHybridStack`'s eviction-stack bookkeeping.
///
/// 44 + 20, the same two-term shape (and, as it happens, the same total) as
/// [`LRU_HYBRID_EVICTION_STACK_DRAM_OVERHEAD`]: one `HashList<HashedKey>`
/// node for the single queue the key occupies at any one time (24-byte heap
/// `Entry` + `cost(16) = 20` internal index slot = 44), plus one slot in the
/// combined `entries` map (20 — the entry struct packs to 8 bytes, so
/// `cost(16)` for the `(HashedKey, Entry)` pair).
///
/// The single-node term is correct rather than an undercount: this design's
/// queues are disjoint and a key is removed from one before being pushed to
/// the other, so it is never resident in two at once. The key itself is
/// stored once, inside the heap node, and is not charged again (the
/// double-charge [`get_policy_overhead`] makes and this module's derivation
/// block flags).
///
/// MEASURED, not derived: the least-squares slope of jemalloc
/// `stats.allocated` against object count, one point per process at
/// 2^20..2^23 objects, R^2 = 1.0000. See `policy_stack::measure_overhead`.
///
/// ALLOCATED, not resident -- size-class-rounded usable bytes, the quantity
/// `malloc_usable_size` returns and therefore the same quantity Redis reports
/// as `used_memory`. An earlier revision measured RSS instead, which counts
/// retained-but-freed pages that belong in a fragmentation ratio rather than in
/// a per-object cost, and which disagreed with itself by 20% depending on where
/// the sample points fell.
///
/// The field-by-field derivation that used to sit here understated this stack
/// by roughly a third: it counted struct fields and not size-class rounding,
/// index-map load factor, or the growth slack of every doubling structure.
/// Being a measured allocation figure it is NOT multiplied by
/// `resident_factor()` -- see the split in `get_hybrid_dram_shared_overhead`.
#[cfg(feature = "hybrid_cache_common")]
const S3_FIFO_LAZY_DEMOTION_FAST_ADMISSION_SPLIT_SLOW_REPRIEVE_HYBRID_EVICTION_STACK_DRAM_OVERHEAD: ObjectSize = 112;


/// Approximate per-object DRAM cost of the *shared* structures (the object
/// hashtable + the eviction stacks) that hold an entry for every object of both
/// tiers. Used by the LRU/LFU/LRU-sized hybrid stacks to reserve room in the
/// fast-tier (DRAM) budget so demotion bounds total DRAM, not just fast-tier
/// values.
///
/// Unlike [`get_policy_overhead`] — which `used_size` charges unconditionally
/// because the eviction-stack bytes count toward the overall DRAM+PMEM budget
/// regardless of which tier they physically live in — this counts only the
/// terms that are actually DRAM-resident: the eviction-stack term is dropped
/// when `eviction_stacks_pmem` moves those stacks to PMEM, and the hashtable
/// entry is dropped when a hashtable-PMEM feature moves the object map to PMEM.
/// Per-object DRAM cost of the value's `Arc` header: the `Arc` inner
/// allocation (two 8-byte refcounts) plus the `TieredBuffer` enum it wraps
/// (16-byte fat pointer + discriminant, padded), landing in the 48-byte size
/// class. Fixed-size, so independent of the trace's value distribution.
/// Always DRAM-resident: the `Arc` itself is allocated by the global
/// allocator even when the buffer it points at lives in PMEM.
#[cfg(feature = "hybrid_cache_common")]
const ARC_VALUE_HEADER_OVERHEAD: ObjectSize = 48;

/// Per-object DRAM cost of the object map (`DashMap<HashedKey, Object>`):
/// the `(u64, Object{key, Arc ptr, expiry})` pair plus hashbrown's control
/// byte and load-factor slack.
///
/// Measured by fitting node-0 live requested bytes against object count over
/// three steady-state runs (cluster12, fifo, 2/6/15 GB caches, all at their
/// cap), then subtracting the two analytically-known terms:
///
///   metadata = 175.4 B/object x objects + 1.016 GB fixed
///   175.4 - 64 (eviction stack) - 48 (`Arc` header) = 63
///
/// The 63 agrees with an independent analytic estimate of ~72 (a 40-byte
/// pair, hashbrown's 7/8 load factor and its power-of-two table slack), which
/// is the main reason to trust it.
///
/// The 1.016 GB intercept is deliberately NOT reserved: it is benchmark-side,
/// not cache metadata. `Access::from_chunk` synthesises a value buffer per
/// trace record (`[0u8].repeat(value_size)`) and the reader prefetches
/// `PREFETCH_RECORDS` (256K) of them, so ~440 MB of node 0 belongs to the
/// harness, plus its channels and client state. An earlier estimate of 165 B
/// came from dividing total metadata by object count at a single cache size,
/// which silently amortised that fixed cost into the per-object term -- at
/// 1.2M objects the same arithmetic yields 989 B/object, which is what made
/// the contamination obvious.
#[cfg(feature = "hybrid_cache_common")]
const OBJECT_MAP_ENTRY_OVERHEAD: ObjectSize = 63;

/// Requested-to-resident multiplier for the DRAM metadata reserved above.
///
/// The terms above count bytes the cache *requests*; the fast-tier budget is
/// meant to bound bytes actually *resident*, and an allocator holds more than
/// was asked for -- size-class rounding plus whatever it retains rather than
/// returning to the OS. Measured at peak on cluster12:
///
/// | allocator                    | rounding | retention | total |
/// |------------------------------|----------|-----------|-------|
/// | jemalloc (default)            | 1.064    | 1.29      | ~1.37 |
/// | jemalloc (`numa_jemalloc`) | 1.061 | 1.017-1.056 | 1.08-1.12 |
///
/// Rounding is near-identical between them; the entire difference is
/// retention, because TBB's per-thread and large-object caches have no purge
/// discipline while jemalloc decays dirty pages back.
///
/// # This is an allocator property, not a workload constant
///
/// It is a *ratio*, so unlike the per-object terms it is not inflated by the
/// harness's own allocations -- those bytes pay the same multiplier. What it
/// does depend on is churn, which depends on the fast-tier size, which this
/// number helps determine: measured values on TBB ranged 1.29-2.75 across
/// configurations for exactly that reason. Treat the constants as starting
/// points for the shipped configurations and recalibrate with
/// `DRAM_OVERHEAD_RESIDENT_FACTOR` when the workload or allocator changes;
/// `jemalloc_stats()` reports the inputs.
///
/// A second-order caveat: the ratio is measured process-wide, so a workload
/// whose non-cache allocations have a very different size profile from the
/// cache's metadata will skew it slightly.
/// Measured 1.08-1.12 resident/allocated on the NUMA-bound jemalloc arenas
/// that back every build. The jemalloc pairing this replaced needed 1.37, and
/// carrying TBB's number into a jemalloc build over-reserved by ~22%,
/// shrinking the effective fast tier for no reason.
#[cfg(feature = "hybrid_cache_common")]
const DEFAULT_RESIDENT_FACTOR: f64 = 1.12;

/// Bytes jemalloc will actually commit for a value of this requested size.
///
/// Asked of the allocator rather than estimated. This is what Redis does --
/// `used_memory` is the sum of `malloc_usable_size` per allocation, never a
/// scaled request -- and `nallocx` answers the same question without
/// allocating, so it costs a size-class lookup.
///
/// A flat factor was tried first and was wrong in both directions. Measured
/// against jemalloc's real classes: 8 -> 8 (1.000x), 24 -> 32 (1.333x),
/// 194 -> 224 (1.155x), 1024 -> 1024 (1.000x), 4096 -> 4096 (1.000x). The
/// ratio is 1.0 for anything landing on a class and up to 1.33 just above one;
/// no constant models that.
///
/// It also mis-attributed process-level waste. A live run showed the slow tier
/// accounting 8.24 GiB against 9.90 GiB resident on node1 -- 20% -- so 1.20 was
/// adopted. But size-class rounding over these corpora's object-size mix is
/// only **1.081x**; the rest is jemalloc's retained dirty pages and arena
/// fragmentation, which are properties of the process, not of an object.
/// Charging them per object over-reserved every small value by ~11%.
///
/// Process-level waste belongs in a reported ratio, the way Redis reports
/// `mem_fragmentation_ratio`, not inside a per-object budget. See
/// `AtomicStatus::fragmentation_ratio`.
#[cfg(feature = "numa_jemalloc")]
fn resident_value_bytes(requested: ObjectSize) -> ObjectSize {
	// SAFETY: `nallocx` is a pure size-class computation. It allocates
	// nothing, dereferences nothing, and cannot fail for a non-zero size.
	match requested {
		0 => 0,
		n => unsafe { tikv_jemalloc_sys::nallocx(n as usize, 0) as ObjectSize },
	}
}

/// Without jemalloc there is no allocator to ask, so the request stands.
/// Estimating here would reintroduce exactly the error described above.
#[cfg(not(feature = "numa_jemalloc"))]
fn resident_value_bytes(requested: ObjectSize) -> ObjectSize {
	requested
}

#[cfg(feature = "hybrid_cache_common")]
fn resident_factor() -> f64 {
	use std::sync::OnceLock;
	static FACTOR: OnceLock<f64> = OnceLock::new();

	*FACTOR.get_or_init(|| {
		std::env::var("DRAM_OVERHEAD_RESIDENT_FACTOR")
			.ok()
			.and_then(|value| value.parse::<f64>().ok())
			.filter(|factor| (1.0..=4.0).contains(factor))
			.unwrap_or(DEFAULT_RESIDENT_FACTOR)
	})
}

#[cfg(feature = "hybrid_cache_common")]
pub fn get_hybrid_dram_shared_overhead(policy: &PaperPolicy) -> ObjectSize {
	// Test support: the tier-mechanics integration tests choreograph
	// promotions and demotions with fast-tier budgets of tens of bytes,
	// where the ~75 B/object metadata reservation below exceeds the whole
	// budget and no promotion can ever fit. Those tests predate the
	// reservation and test policy mechanics, not DRAM accounting, so
	// `PAPER_DISABLE_SHARED_OVERHEAD=1` restores their value-only
	// semantics, and those binaries set it from their own
	// `ensure_pmem_allocator_warm()`.
	//
	// The reservation itself is therefore covered by separate test binaries
	// -- tests/{lru,lfu,lru_sized}_hybrid_cache_shared_overhead.rs -- which
	// never set the variable, so every cache they build gets the production
	// default. They are separate PROCESSES on purpose: this is read at every
	// cache construction, so a test flipping the variable back would race
	// every sibling test constructing a cache on another thread.
	if std::env::var_os("PAPER_DISABLE_SHARED_OVERHEAD").is_some_and(|v| v == "1") {
		return 0;
	}

	#[allow(unused_mut)]
	let mut overhead: ObjectSize = 0;

	// Eviction stacks live in DRAM unless `eviction_stacks_pmem` relocates them.
	//
	// Selected by a runtime `match`, not by `cfg`. These are integers: nothing
	// about a policy's constant requires its stack module to be compiled, and
	// gating them meant a build without that feature silently contributed 0 --
	// no error, no warning, no failing test. That is not hypothetical: a binary
	// built with only `lru_hybrid_cache` charged every other policy
	// Arc(48) + map(63) = 111 -> 124 B/object instead of its real 196 or 228,
	// so each non-LRU policy was handed a larger effective fast tier than it
	// should have had, for a whole sweep, before anyone noticed.
	//
	// The match is exhaustive deliberately: adding a policy without giving it an
	// overhead term is now a compile error rather than a silent zero.
	// Measured resident, kept separate from the derived terms below.
	#[allow(unused_mut)]
	let mut stack_resident: ObjectSize = 0;

	#[cfg(not(feature = "eviction_stacks_pmem"))]
	{
		stack_resident = match policy {
			PaperPolicy::LruHybrid => LRU_HYBRID_EVICTION_STACK_DRAM_OVERHEAD,
			PaperPolicy::LfuHybrid => LFU_HYBRID_EVICTION_STACK_DRAM_OVERHEAD,
			PaperPolicy::LruCompactHybrid => LRU_COMPACT_HYBRID_EVICTION_STACK_DRAM_OVERHEAD,
			PaperPolicy::LfuCompactHybrid => LFU_COMPACT_HYBRID_EVICTION_STACK_DRAM_OVERHEAD,
			PaperPolicy::LruSizedHybrid => LRU_SIZED_HYBRID_EVICTION_STACK_DRAM_OVERHEAD,
			PaperPolicy::LruLfuHybrid(..) => LRU_LFU_HYBRID_EVICTION_STACK_DRAM_OVERHEAD,
			PaperPolicy::FifoHybrid => FIFO_HYBRID_EVICTION_STACK_DRAM_OVERHEAD,
			PaperPolicy::TwoQCompactHybrid(..) => TWO_Q_COMPACT_HYBRID_EVICTION_STACK_DRAM_OVERHEAD,
			PaperPolicy::TwoQHybrid(..) => TWO_Q_HYBRID_EVICTION_STACK_DRAM_OVERHEAD,
			PaperPolicy::TwoQFastAdmissionHybrid(..) => TWO_Q_FAST_ADMISSION_HYBRID_EVICTION_STACK_DRAM_OVERHEAD,
			PaperPolicy::TwoQFastAdmissionReprieveHybrid(..) => TWO_Q_FAST_ADMISSION_REPRIEVE_HYBRID_EVICTION_STACK_DRAM_OVERHEAD,
			PaperPolicy::TwoQFullFastAdmissionHybrid(..) => TWO_Q_FULL_FAST_ADMISSION_HYBRID_EVICTION_STACK_DRAM_OVERHEAD,
			PaperPolicy::TwoQGhostHybrid(..) => TWO_Q_GHOST_HYBRID_EVICTION_STACK_DRAM_OVERHEAD,
			PaperPolicy::S3FifoCompactHybrid(..) => S3_FIFO_COMPACT_HYBRID_EVICTION_STACK_DRAM_OVERHEAD,
			PaperPolicy::S3FifoHybrid(..) => S3_FIFO_HYBRID_EVICTION_STACK_DRAM_OVERHEAD,
			PaperPolicy::S3FifoGhostHybrid(..) => S3_FIFO_GHOST_HYBRID_EVICTION_STACK_DRAM_OVERHEAD,
			PaperPolicy::S3FifoGhostLazyDemotionHybrid(..) => S3_FIFO_GHOST_LAZY_DEMOTION_HYBRID_EVICTION_STACK_DRAM_OVERHEAD,
			PaperPolicy::S3FifoGhostLazyDemotionFastAdmissionHybrid(..) => S3_FIFO_GHOST_LAZY_DEMOTION_FAST_ADMISSION_HYBRID_EVICTION_STACK_DRAM_OVERHEAD,
			PaperPolicy::S3FifoGhostLazyDemotionFastAdmissionMidpointHybrid(..) => S3_FIFO_GHOST_LAZY_DEMOTION_FAST_ADMISSION_MIDPOINT_HYBRID_EVICTION_STACK_DRAM_OVERHEAD,
			PaperPolicy::S3FifoLazyDemotionReprieveHybrid(..) => S3_FIFO_LAZY_DEMOTION_REPRIEVE_HYBRID_EVICTION_STACK_DRAM_OVERHEAD,
			PaperPolicy::S3FifoLazyDemotionFastAdmissionReprieveHybrid(..) => S3_FIFO_LAZY_DEMOTION_FAST_ADMISSION_REPRIEVE_HYBRID_EVICTION_STACK_DRAM_OVERHEAD,
			PaperPolicy::S3FifoLazyDemotionFastAdmissionMidpointReprieveHybrid(..) => S3_FIFO_LAZY_DEMOTION_FAST_ADMISSION_MIDPOINT_REPRIEVE_HYBRID_EVICTION_STACK_DRAM_OVERHEAD,
			PaperPolicy::S3FifoLazyDemotionFastAdmissionSplitSlowReprieveHybrid(..) => S3_FIFO_LAZY_DEMOTION_FAST_ADMISSION_SPLIT_SLOW_REPRIEVE_HYBRID_EVICTION_STACK_DRAM_OVERHEAD,

			// All-DRAM policies have no tiers and reserve no fast-tier metadata.
			PaperPolicy::Auto
			| PaperPolicy::Lfu
			| PaperPolicy::Fifo
			| PaperPolicy::Clock
			| PaperPolicy::Sieve
			| PaperPolicy::Lru
			| PaperPolicy::Mru
			| PaperPolicy::TwoQ(..)
			| PaperPolicy::Arc
			| PaperPolicy::SThreeFifo(..) => 0,
		};
		overhead += stack_resident;
	}

	// The value's Arc header is DRAM-resident regardless of which tier the
	// buffer itself occupies, and regardless of any hashtable-PMEM feature.
	overhead += ARC_VALUE_HEADER_OVERHEAD;

	// The object map lives in DRAM unless a hashtable-PMEM feature
	// relocates it (`global_hashtable_pmem`).
	#[cfg(not(feature = "global_hashtable_pmem"))]
	{
		overhead += OBJECT_MAP_ENTRY_OVERHEAD;
	}

	// Requested -> resident: what this reservation is protecting is real
	// DRAM, and the allocator holds more than was requested (size-class
	// rounding plus retained freed pages; see `DEFAULT_RESIDENT_FACTOR`).
	//
	// This applies to the Arc header and map entry only. The eviction-stack
	// term is subtracted out first and added back afterwards, because it is
	// MEASURED RESIDENT -- it comes from RSS (see `measure_overhead`), so it
	// already contains the size-class rounding and retained pages this factor
	// exists to model. Multiplying it again would charge that rounding twice.
	let derived = overhead - stack_resident;
	stack_resident + (derived as f64 * resident_factor()) as ObjectSize
}

#[cfg(all(test, feature = "hybrid_cache_common"))]
mod shared_overhead_is_feature_independent {
	use super::*;

	/// A policy's DRAM term must not depend on which *other* stack modules were
	/// compiled.
	///
	/// This is a regression test for a silent, whole-sweep measurement error.
	/// The terms used to be `cfg`-gated per policy, so a binary built with only
	/// `lru_hybrid_cache` -- which is how the benchmark was configured --
	/// charged every non-LRU policy `Arc(48) + map(63) = 111 -> 124` B/object
	/// instead of its real 196 or 228. Each of those policies was therefore
	/// handed a larger effective fast tier than it should have had, and nothing
	/// failed: no error, no warning, no test.
	///
	/// This test runs under whatever feature set the build has, so it fails if
	/// anyone reintroduces the gating.
	#[test]
	fn every_hybrid_policy_keeps_its_own_term() {
		if std::env::var_os("PAPER_DISABLE_SHARED_OVERHEAD").is_some() {
			return; // the escape hatch zeroes everything by design
		}

		let lru = get_hybrid_dram_shared_overhead(&PaperPolicy::LruHybrid);
		let lfu = get_hybrid_dram_shared_overhead(&PaperPolicy::LfuHybrid);
		let fifo = get_hybrid_dram_shared_overhead(&PaperPolicy::FifoHybrid);
		let s3 = get_hybrid_dram_shared_overhead(&PaperPolicy::S3FifoHybrid(0.1));

		// The value with NO eviction-stack term -- what a gated-out policy used
		// to collapse to. No hybrid policy may equal it.
		let no_stack_term =
			((ARC_VALUE_HEADER_OVERHEAD + OBJECT_MAP_ENTRY_OVERHEAD) as f64
				* resident_factor()) as ObjectSize;

		for (name, got) in [("lru", lru), ("lfu", lfu), ("fifo", fifo), ("s3-fifo", s3)] {
			assert_ne!(
				got, no_stack_term,
				"{name} lost its eviction-stack term and collapsed to {no_stack_term} \
				 B/object -- the per-policy cfg gating is back",
			);
		}

		// LFU carries a frequency structure the others do not (44+29+20 vs 44+20).
		assert!(
			lfu > lru,
			"lfu ({lfu}) must exceed lru ({lru}): it has the extra frequency term",
		);
		assert_eq!(fifo, lru, "fifo and lru have the same 44+20 stack shape");
		assert_eq!(s3, lru, "s3-fifo and lru have the same 44+20 stack shape");
	}
}

#[cfg(all(test, feature = "hybrid_cache_common"))]
mod value_resident_factor_applies {
	use std::sync::Arc;

	use super::*;
	use crate::{policy::PaperPolicy, status::AtomicStatus};

	/// `base_size` must report what the allocator holds, not what was asked
	/// for. Before this, only metadata carried a resident factor; a live run
	/// showed the slow tier accounting 8.24 GiB while node1 held 9.90 GiB.
	///
	/// Every existing test derives its expectations from `base_size` itself,
	/// so all of them stayed green through this change -- good design on their
	/// part, but it means not one of them would have caught the omission.
	#[test]
	fn the_value_is_counted_at_its_allocated_size() {
		let status: crate::StatusRef = Arc::new(
			AtomicStatus::new(1_000_000, &[PaperPolicy::LruHybrid], PaperPolicy::LruHybrid)
				.expect("status"),
		);
		let manager = OverheadManager::new(&status);
		let object = Object::new(0u32, vec![0u8; 1000].into_boxed_slice(), None);

		let got = manager.base_size(&object);
		let key = object.key_size();
		let expiry = mem::size_of::<crate::object::ExpireTime>() as ObjectSize;
		// `data_size` for a `Box<[u8]>` counts the 16-byte fat pointer as well
		// as the 1000 payload bytes. On the hybrid path V is `TieredBuffer`,
		// whose `get_size` is the length exactly.
		let raw = object.data_size();
		let scaled = resident_value_bytes(raw);

		assert_eq!(
			got, key + scaled + expiry,
			"the value is counted at jemalloc's committed size; the key and \
			 expiry are left alone, since shared_overhead covers them",
		);

		assert!(
			got > key + raw + expiry,
			"a {raw}-byte value must account for more than {raw} bytes: got {got}, \
			 unscaled would be {}",
			key + raw + expiry,
		);
	}
}

/// What jemalloc would actually commit for a given request, versus the flat
/// factor this module estimates with.
///
/// Redis answers this question by calling `malloc_usable_size` per allocation;
/// memcached sidesteps it by budgeting slab pages, so its rounding waste is
/// explicit. This crate estimates instead, which is why the value side was out
/// by 20% until it was measured. `nallocx` gives the exact size class for a
/// request without allocating -- the same information Redis uses.
///
///   cargo +nightly test --release --features lru_hybrid_cache --lib \
///       what_jemalloc -- --ignored --nocapture
#[cfg(all(test, feature = "numa_jemalloc"))]
mod what_jemalloc_actually_rounds_to {
	#[test]
	#[ignore]
	fn what_jemalloc_rounds_to() {
		use tikv_jemalloc_sys::nallocx;

		println!("NX  requested   jemalloc    ratio");
		let mut worst: f64 = 0.0;
		for s in [8usize, 13, 24, 29, 64, 100, 130, 194, 225, 500, 1000,
		          1024, 1497, 1500, 4096, 5607, 8392, 60103] {
			let a = unsafe { nallocx(s, 0) };
			let r = a as f64 / s as f64;
			if r > worst { worst = r; }
			println!("NX  {:>9} {:>10}   {:.3}x", s, a, r);
		}
		println!("NX  worst single-size ratio: {:.3}x", worst);

		// Weighted by the object-size mix actually measured in these corpora:
		// Meta kvcache medians are 13-132 B, Twitter clusters 77-8392 B.
		let mix: [(usize, f64); 7] = [
			(13, 25.0), (29, 20.0), (77, 15.0), (132, 15.0),
			(194, 10.0), (1497, 10.0), (5607, 5.0),
		];
		let (mut req, mut act) = (0.0, 0.0);
		for (s, w) in mix {
			req += s as f64 * w;
			act += unsafe { nallocx(s, 0) } as f64 * w;
		}
		println!("NX  trace-weighted mix: {:.3}x  (the flat estimate in use: 1.20)",
			act / req);
	}
}
