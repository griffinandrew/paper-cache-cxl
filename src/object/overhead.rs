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
	}
}

pub fn get_ttl_overhead() -> ObjectSize {
	// the size of an Option<Instant> plus 48 bytes for the BTreeMap entry
	mem::size_of::<Option<Instant>>() as ObjectSize + 48
}
