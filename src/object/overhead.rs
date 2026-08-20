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
	pub fn base_size<K, V>(&self, object: &Object<K, V>) -> ObjectSize
	where
		K: TypeSize,
		V: TypeSize,
	{
		let mut total_size = object.total_size();

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

		// Worst-case charge for a key resident in main_stack as Fast:
		// 48-byte HashList entry + 8-byte HashedKey + a single combined
		// per-key `entries` HashMap entry (queue + tier + size, one map —
		// see `TwoQHybridStack`'s module doc for why this collapsed from
		// three separate maps) — 24 bytes for the entry, 1 byte for the
		// Queue tag, 1 byte for the Option<Tier> tag (only meaningful for
		// keys currently in Main), 4 bytes for the object size
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
#[cfg(feature = "lru_hybrid_cache")]
const LRU_HYBRID_EVICTION_STACK_DRAM_OVERHEAD: ObjectSize = 44 + 20;

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
#[cfg(feature = "lfu_hybrid_cache")]
const LFU_HYBRID_EVICTION_STACK_DRAM_OVERHEAD: ObjectSize = 44 + 29 + 20;

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
#[cfg(feature = "lru_sized_hybrid_cache")]
const LRU_SIZED_HYBRID_EVICTION_STACK_DRAM_OVERHEAD: ObjectSize = 44 + 20;

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
#[cfg(feature = "lru_lfu_hybrid_cache")]
const LRU_LFU_HYBRID_EVICTION_STACK_DRAM_OVERHEAD: ObjectSize = 44 + 20;

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
pub const GHOST_ENTRY_DRAM_OVERHEAD: ObjectSize = 44;

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
#[cfg(feature = "fifo_hybrid_cache")]
const FIFO_HYBRID_EVICTION_STACK_DRAM_OVERHEAD: ObjectSize = 44 + 20;

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
#[cfg(feature = "two_q_hybrid_cache")]
const TWO_Q_HYBRID_EVICTION_STACK_DRAM_OVERHEAD: ObjectSize = 44 + 20;

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
#[cfg(feature = "two_q_fast_admission_hybrid_cache")]
const TWO_Q_FAST_ADMISSION_HYBRID_EVICTION_STACK_DRAM_OVERHEAD: ObjectSize = 44 + 20;

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
#[cfg(feature = "two_q_fast_admission_reprieve_hybrid_cache")]
const TWO_Q_FAST_ADMISSION_REPRIEVE_HYBRID_EVICTION_STACK_DRAM_OVERHEAD: ObjectSize = 44 + 20;

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
/// Excludes the bare-key ghost queue, which is charged separately via
/// [`GHOST_ENTRY_DRAM_OVERHEAD`] against `ghost.len()`: its length is
/// bounded only lazily (`trim_ghost` runs solely on a genuine main-queue
/// eviction, while one-access-queue evictions are what grow it), so it is
/// not expressible as a per-tracked-key term.
#[cfg(feature = "two_q_ghost_hybrid_cache")]
const TWO_Q_GHOST_HYBRID_EVICTION_STACK_DRAM_OVERHEAD: ObjectSize = 44 + 20;

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
#[cfg(feature = "s3_fifo_hybrid_cache")]
const S3_FIFO_HYBRID_EVICTION_STACK_DRAM_OVERHEAD: ObjectSize = 44 + 20;

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
/// Excludes the bare-key ghost queue, which is charged separately via
/// [`GHOST_ENTRY_DRAM_OVERHEAD`] against `ghost.len()`: its length is
/// bounded only lazily (`trim_ghost` runs solely on a genuine main-queue
/// eviction, while one-access-queue evictions are what grow it), so it is
/// not expressible as a per-tracked-key term.
#[cfg(feature = "s3_fifo_ghost_hybrid_cache")]
const S3_FIFO_GHOST_HYBRID_EVICTION_STACK_DRAM_OVERHEAD: ObjectSize = 44 + 20;

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
/// Excludes the bare-key ghost queue, which is charged separately via
/// [`GHOST_ENTRY_DRAM_OVERHEAD`] against `ghost.len()`: its length is
/// bounded only lazily (`trim_ghost` runs solely on a genuine main-queue
/// eviction, while one-access-queue evictions are what grow it), so it is
/// not expressible as a per-tracked-key term.
#[cfg(feature = "s3_fifo_ghost_lazy_demotion_hybrid_cache")]
const S3_FIFO_GHOST_LAZY_DEMOTION_HYBRID_EVICTION_STACK_DRAM_OVERHEAD: ObjectSize = 44 + 20;

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
/// Excludes the bare-key ghost queue, which is charged separately via
/// [`GHOST_ENTRY_DRAM_OVERHEAD`] against `ghost.len()`: its length is
/// bounded only lazily (`trim_ghost` runs solely on a genuine main-queue
/// eviction, while one-access-queue evictions are what grow it), so it is
/// not expressible as a per-tracked-key term.
#[cfg(feature = "s3_fifo_ghost_lazy_demotion_fast_admission_hybrid_cache")]
const S3_FIFO_GHOST_LAZY_DEMOTION_FAST_ADMISSION_HYBRID_EVICTION_STACK_DRAM_OVERHEAD: ObjectSize = 44 + 20;

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
/// Excludes the bare-key ghost queue, which is charged separately via
/// [`GHOST_ENTRY_DRAM_OVERHEAD`] against `ghost.len()`: its length is
/// bounded only lazily (`trim_ghost` runs solely on a genuine main-queue
/// eviction, while one-access-queue evictions are what grow it), so it is
/// not expressible as a per-tracked-key term.
#[cfg(feature = "s3_fifo_ghost_lazy_demotion_fast_admission_midpoint_hybrid_cache")]
const S3_FIFO_GHOST_LAZY_DEMOTION_FAST_ADMISSION_MIDPOINT_HYBRID_EVICTION_STACK_DRAM_OVERHEAD: ObjectSize = 44 + 20;

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
#[cfg(feature = "s3_fifo_lazy_demotion_reprieve_hybrid_cache")]
const S3_FIFO_LAZY_DEMOTION_REPRIEVE_HYBRID_EVICTION_STACK_DRAM_OVERHEAD: ObjectSize = 44 + 20;

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
#[cfg(feature = "s3_fifo_lazy_demotion_fast_admission_reprieve_hybrid_cache")]
const S3_FIFO_LAZY_DEMOTION_FAST_ADMISSION_REPRIEVE_HYBRID_EVICTION_STACK_DRAM_OVERHEAD: ObjectSize = 44 + 20;

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
#[cfg(feature = "s3_fifo_lazy_demotion_fast_admission_midpoint_reprieve_hybrid_cache")]
const S3_FIFO_LAZY_DEMOTION_FAST_ADMISSION_MIDPOINT_REPRIEVE_HYBRID_EVICTION_STACK_DRAM_OVERHEAD: ObjectSize = 44 + 20;

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
#[cfg(feature = "s3_fifo_lazy_demotion_fast_admission_split_slow_reprieve_hybrid_cache")]
const S3_FIFO_LAZY_DEMOTION_FAST_ADMISSION_SPLIT_SLOW_REPRIEVE_HYBRID_EVICTION_STACK_DRAM_OVERHEAD: ObjectSize = 44 + 20;


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
/// bytes and load-factor slack.
///
/// CALIBRATED, not purely analytic: measured on cluster12 at 9.46M objects
/// (u32 keys) by instrumenting the node-0 allocator -- live requested bytes
/// minus the cache's own fast-tier value bytes came to ~277 B/object, of
/// which the analytic policy + Arc terms account for ~112, leaving ~165 for
/// the object map. Two known sources of variation: hashbrown's slack moves
/// with object count relative to the table's power-of-two capacity (a
/// sawtooth of up to tens of bytes), and a different key type changes
/// `Object`'s size. Replaces the old 11-byte estimate, which was a 4-5x
/// under-reservation in practice.
#[cfg(feature = "hybrid_cache_common")]
const OBJECT_MAP_ENTRY_OVERHEAD: ObjectSize = 165;

/// Requested-to-resident multiplier for DRAM metadata, as (numerator,
/// denominator).
///
/// The reservation above counts *requested* bytes, but the fast-tier budget
/// is meant to bound *resident* DRAM, and the allocator holds more than was
/// requested: size-class rounding (~6% on both allocators, measured), plus
/// whatever freed memory the allocator retains rather than returning.
/// Measured at peak on the same cluster12 run:
///
/// - UMF/TBB (default): usable/requested 1.0643, resident/usable 1.29 --
///   TBB's per-thread and large-object caches hold freed pages with no purge
///   discipline -- so ~1.37 overall.
/// - `tikv_jemalloc_global`: active/allocated 1.0612, resident/active 1.0533
///   (decay returns dirty pages) -- ~1.12 overall.
///
/// The TBB retention component is churn-dependent (this trace cycles ~34M
/// evictions over ~9M live objects), so it is the least general of these
/// calibrations; override for other workloads via
/// `DRAM_OVERHEAD_RESIDENT_FACTOR` (a float, e.g. `1.2`), recalibrating from
/// the `umf_dram_stats()` / `jemalloc_stats()` probes.
#[cfg(all(feature = "hybrid_cache_common", not(feature = "tikv_jemalloc_global")))]
const DEFAULT_RESIDENT_FACTOR: f64 = 1.37;
#[cfg(all(feature = "hybrid_cache_common", feature = "tikv_jemalloc_global"))]
const DEFAULT_RESIDENT_FACTOR: f64 = 1.12;

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
	#[allow(unused_mut)]
	let mut overhead: ObjectSize = 0;

	// Eviction stacks live in DRAM unless `eviction_stacks_pmem` relocates them.
	#[cfg(not(feature = "eviction_stacks_pmem"))]
	{
		#[cfg(feature = "lru_hybrid_cache")]
		if matches!(policy, PaperPolicy::LruHybrid) {
			overhead += LRU_HYBRID_EVICTION_STACK_DRAM_OVERHEAD;
		}

		#[cfg(feature = "lfu_hybrid_cache")]
		if matches!(policy, PaperPolicy::LfuHybrid) {
			overhead += LFU_HYBRID_EVICTION_STACK_DRAM_OVERHEAD;
		}

		#[cfg(feature = "lru_sized_hybrid_cache")]
		if matches!(policy, PaperPolicy::LruSizedHybrid) {
			overhead += LRU_SIZED_HYBRID_EVICTION_STACK_DRAM_OVERHEAD;
		}

		#[cfg(feature = "lru_lfu_hybrid_cache")]
		if matches!(policy, PaperPolicy::LruLfuHybrid(_)) {
			overhead += LRU_LFU_HYBRID_EVICTION_STACK_DRAM_OVERHEAD;
		}

		#[cfg(feature = "fifo_hybrid_cache")]
		if matches!(policy, PaperPolicy::FifoHybrid) {
			overhead += FIFO_HYBRID_EVICTION_STACK_DRAM_OVERHEAD;
		}

		#[cfg(feature = "two_q_hybrid_cache")]
		if matches!(policy, PaperPolicy::TwoQHybrid(_)) {
			overhead += TWO_Q_HYBRID_EVICTION_STACK_DRAM_OVERHEAD;
		}

		#[cfg(feature = "two_q_fast_admission_hybrid_cache")]
		if matches!(policy, PaperPolicy::TwoQFastAdmissionHybrid(_)) {
			overhead += TWO_Q_FAST_ADMISSION_HYBRID_EVICTION_STACK_DRAM_OVERHEAD;
		}

		#[cfg(feature = "two_q_fast_admission_reprieve_hybrid_cache")]
		if matches!(policy, PaperPolicy::TwoQFastAdmissionReprieveHybrid(_)) {
			overhead += TWO_Q_FAST_ADMISSION_REPRIEVE_HYBRID_EVICTION_STACK_DRAM_OVERHEAD;
		}

		#[cfg(feature = "two_q_ghost_hybrid_cache")]
		if matches!(policy, PaperPolicy::TwoQGhostHybrid(_)) {
			overhead += TWO_Q_GHOST_HYBRID_EVICTION_STACK_DRAM_OVERHEAD;
		}

		#[cfg(feature = "s3_fifo_hybrid_cache")]
		if matches!(policy, PaperPolicy::S3FifoHybrid(_)) {
			overhead += S3_FIFO_HYBRID_EVICTION_STACK_DRAM_OVERHEAD;
		}

		#[cfg(feature = "s3_fifo_ghost_hybrid_cache")]
		if matches!(policy, PaperPolicy::S3FifoGhostHybrid(_)) {
			overhead += S3_FIFO_GHOST_HYBRID_EVICTION_STACK_DRAM_OVERHEAD;
		}

		#[cfg(feature = "s3_fifo_ghost_lazy_demotion_hybrid_cache")]
		if matches!(policy, PaperPolicy::S3FifoGhostLazyDemotionHybrid(_)) {
			overhead += S3_FIFO_GHOST_LAZY_DEMOTION_HYBRID_EVICTION_STACK_DRAM_OVERHEAD;
		}

		#[cfg(feature = "s3_fifo_ghost_lazy_demotion_fast_admission_hybrid_cache")]
		if matches!(policy, PaperPolicy::S3FifoGhostLazyDemotionFastAdmissionHybrid(_)) {
			overhead += S3_FIFO_GHOST_LAZY_DEMOTION_FAST_ADMISSION_HYBRID_EVICTION_STACK_DRAM_OVERHEAD;
		}

		#[cfg(feature = "s3_fifo_ghost_lazy_demotion_fast_admission_midpoint_hybrid_cache")]
		if matches!(policy, PaperPolicy::S3FifoGhostLazyDemotionFastAdmissionMidpointHybrid(_)) {
			overhead += S3_FIFO_GHOST_LAZY_DEMOTION_FAST_ADMISSION_MIDPOINT_HYBRID_EVICTION_STACK_DRAM_OVERHEAD;
		}

		#[cfg(feature = "s3_fifo_lazy_demotion_reprieve_hybrid_cache")]
		if matches!(policy, PaperPolicy::S3FifoLazyDemotionReprieveHybrid(_)) {
			overhead += S3_FIFO_LAZY_DEMOTION_REPRIEVE_HYBRID_EVICTION_STACK_DRAM_OVERHEAD;
		}

		#[cfg(feature = "s3_fifo_lazy_demotion_fast_admission_reprieve_hybrid_cache")]
		if matches!(policy, PaperPolicy::S3FifoLazyDemotionFastAdmissionReprieveHybrid(_)) {
			overhead += S3_FIFO_LAZY_DEMOTION_FAST_ADMISSION_REPRIEVE_HYBRID_EVICTION_STACK_DRAM_OVERHEAD;
		}

		#[cfg(feature = "s3_fifo_lazy_demotion_fast_admission_midpoint_reprieve_hybrid_cache")]
		if matches!(policy, PaperPolicy::S3FifoLazyDemotionFastAdmissionMidpointReprieveHybrid(_)) {
			overhead += S3_FIFO_LAZY_DEMOTION_FAST_ADMISSION_MIDPOINT_REPRIEVE_HYBRID_EVICTION_STACK_DRAM_OVERHEAD;
		}

		#[cfg(feature = "s3_fifo_lazy_demotion_fast_admission_split_slow_reprieve_hybrid_cache")]
		if matches!(policy, PaperPolicy::S3FifoLazyDemotionFastAdmissionSplitSlowReprieveHybrid(_)) {
			overhead += S3_FIFO_LAZY_DEMOTION_FAST_ADMISSION_SPLIT_SLOW_REPRIEVE_HYBRID_EVICTION_STACK_DRAM_OVERHEAD;
		}
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
	(overhead as f64 * resident_factor()) as ObjectSize
}
