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
		// 24 bytes for the per-key tier-tracking HashMap entry, 1 byte for
		// the Tier tag
		PaperPolicy::LruHybrid => 48 + 8 + 24 + 1,

		// Base LFU overhead (24 HashMap entry + 48 bucket-list entry + 8
		// HashedKey + 4 count = 84) plus what LfuHybridStack needs beyond
		// plain LfuStack: a 24-byte per-key tier-tracking HashMap entry
		// (+ 1 byte Tier tag), and a 24-byte per-key sizes HashMap entry
		// (+ 4 bytes for the object size, matching the "+4" charge already
		// used for TwoQ/Arc/SThreeFifo)
		PaperPolicy::LfuHybrid => (24 + 48 + 8 + 4) + (24 + 1) + (24 + 4),

		// Worst-case charge for a key resident in main_stack as Fast:
		// 48-byte HashList entry + 8-byte HashedKey + a 24-byte `queue`
		// HashMap entry (+1 byte Queue tag, Fifo vs Main) + a 24-byte
		// `main_tiers` HashMap entry (+1 byte Tier tag, only populated
		// for keys currently in Main) + a 24-byte `sizes` HashMap entry
		// (+4 bytes for the object size)
		PaperPolicy::TwoQHybrid(_) => (48 + 8) + (24 + 1) + (24 + 1) + (24 + 4),
	}
}

pub fn get_ttl_overhead() -> ObjectSize {
	// the size of an Option<Instant> plus 48 bytes for the BTreeMap entry
	mem::size_of::<Option<Instant>>() as ObjectSize + 48
}

/// Approximate per-entry structural overhead of the shared object hashtable
/// (`DashMap`), *beyond* the stored `Object` itself (which `base_size` already
/// accounts for). Covers the duplicated 8-byte `HashedKey`, the hashbrown
/// control byte, and load-factor slack.
///
/// NOTE: this is a rough placeholder in the same spirit as the
/// `get_policy_overhead` estimates. In reality hashbrown's load-factor slack
/// scales with the full slot size (key + `Object`), not a fixed constant, and
/// capacity grows in powers of two — so this under-counts for large objects.
/// TODO: replace with a real measured allocated-size query if/when the hybrid
/// caches need an exact DRAM ceiling rather than a demotion target.
#[cfg(any(feature = "lru_hybrid_cache", feature = "lfu_hybrid_cache"))]
pub const HASHTABLE_ENTRY_OVERHEAD: ObjectSize = 24;

/// Approximate per-object DRAM cost of the *shared* structures (the object
/// hashtable + the eviction stacks) that hold an entry for every object of both
/// tiers. Used by the LRU/LFU hybrid stacks to reserve room in the fast-tier
/// (DRAM) budget so demotion bounds total DRAM, not just fast-tier values.
///
/// Unlike [`get_policy_overhead`] — which `used_size` charges unconditionally
/// because the eviction-stack bytes count toward the overall DRAM+PMEM budget
/// regardless of which tier they physically live in — this counts only the
/// terms that are actually DRAM-resident: the eviction-stack node is dropped
/// when `eviction_stacks_pmem` moves those stacks to PMEM, and the hashtable
/// entry is dropped when a hashtable-PMEM feature moves the object map to PMEM.
/// Approximate; see [`HASHTABLE_ENTRY_OVERHEAD`].
#[cfg(any(feature = "lru_hybrid_cache", feature = "lfu_hybrid_cache"))]
pub fn get_hybrid_dram_shared_overhead(policy: &PaperPolicy) -> ObjectSize {
	#[allow(unused_mut)]
	let mut overhead: ObjectSize = 0;

	// Eviction stacks live in DRAM unless `eviction_stacks_pmem` relocates them.
	#[cfg(not(feature = "eviction_stacks_pmem"))]
	{
		overhead += get_policy_overhead(policy);
	}

	// The object hashtable lives in DRAM unless a hashtable-PMEM feature
	// relocates it (`global_hashtable_pmem` / `global_flatmap_pmem`).
	#[cfg(not(any(feature = "global_hashtable_pmem", feature = "global_flatmap_pmem")))]
	{
		overhead += HASHTABLE_ENTRY_OVERHEAD;
	}

	let _ = policy;
	overhead
}
