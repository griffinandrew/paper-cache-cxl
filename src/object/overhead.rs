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
#[cfg(any(feature = "lru_hybrid_cache", feature = "lfu_hybrid_cache"))]
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

/// Approximate per-object DRAM cost of the *shared* structures (the object
/// hashtable + the eviction stacks) that hold an entry for every object of both
/// tiers. Used by the LRU/LFU hybrid stacks to reserve room in the fast-tier
/// (DRAM) budget so demotion bounds total DRAM, not just fast-tier values.
///
/// Unlike [`get_policy_overhead`] — which `used_size` charges unconditionally
/// because the eviction-stack bytes count toward the overall DRAM+PMEM budget
/// regardless of which tier they physically live in — this counts only the
/// terms that are actually DRAM-resident: the eviction-stack term is dropped
/// when `eviction_stacks_pmem` moves those stacks to PMEM, and the hashtable
/// entry is dropped when a hashtable-PMEM feature moves the object map to PMEM.
#[cfg(any(feature = "lru_hybrid_cache", feature = "lfu_hybrid_cache"))]
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
	}

	// The object hashtable lives in DRAM unless a hashtable-PMEM feature
	// relocates it (`global_hashtable_pmem` / `global_flatmap_pmem`).
	#[cfg(not(any(feature = "global_hashtable_pmem", feature = "global_flatmap_pmem")))]
	{
		overhead += HASHTABLE_ENTRY_OVERHEAD;
	}

	overhead
}
