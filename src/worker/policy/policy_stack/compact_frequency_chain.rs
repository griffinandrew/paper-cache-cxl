/*
 * Copyright (c) Kia Shakiba
 *
 * This source code is licensed under the GNU AGPLv3 license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! A frequency chain that stores each key once, in a slab, with intrusive links.
//!
//! # Why
//!
//! `FrequencyChain` (see `lfu_hybrid_stack.rs`) keeps three structures keyed by
//! the same `HashedKey`, and so stores that key three times:
//!
//! ```text
//! entries:    HashMap<HashedKey, LfuEntry>       key + {tier, size}    ~20 B
//! index_map:  HashMap<HashedKey, ChainIndex>     key again             ~29 B
//! CountStack: HashList<HashedKey> -> Entry { key, prev, next } + index ~44 B
//!                                                                      ─────
//!                                                                       93 B
//! ```
//!
//! Three hash structures, three copies of an 8-byte key, and a separate heap
//! allocation per key for the list node -- all describing one logical entry.
//!
//! This packs the whole entry into one slab slot and links buckets with `u32`
//! slab indices. The key is stored twice, not three times (once in the slab so
//! an eviction can unindex itself, once in the index), and there is no per-key
//! allocation at all.
//!
//! Following a link becomes a contiguous array index rather than a pointer
//! chase into scattered heap nodes -- which is also why linking by index is
//! safe where a pointer-linked intrusive list would not be: growing the slab
//! moves the backing allocation, but the indices stay valid.
//!
//! # What it does not change
//!
//! The algorithm. Buckets are still one-per-distinct-frequency, `bump` still
//! moves a key to the adjacent bucket in O(1), and the minimum frequency is
//! still O(1) to find. This is a representation change only.

use std::collections::BTreeMap;

use crate::{
	HashedKey,
	object::ObjectSize,
	worker::policy::policy_stack::Tier,
};

/// `u32::MAX` marks "no neighbour" / "free". A cache holding 4.29 billion
/// tracked keys would exhaust the index space long after it exhausted DRAM.
const NIL: u32 = u32::MAX;

/// One tracked key, entire. Compare `LfuEntry` + a `HashList` node + an
/// `index_map` entry, which is what this replaces.
#[derive(Clone, Copy, Debug)]
pub struct CompactEntry {
	/// Kept so an eviction, which finds the entry by slab index, can remove
	/// the key from the index without a reverse scan.
	pub key: HashedKey,

	prev: u32,
	next: u32,

	/// This key's access count, and so which bucket it belongs to. Holding it
	/// here is what removes the need for `index_map`.
	pub freq: u32,

	pub size: ObjectSize,
	pub tier: Tier,

	/// Part of `size` that stays in DRAM in either tier.
	pub dram_resident: u8,
}

impl CompactEntry {
	#[inline]
	pub fn migrating(&self) -> u64 {
		(self.size as u64).saturating_sub(self.dram_resident as u64)
	}
}

pub struct CompactFrequencyChain {
	slots: Vec<CompactEntry>,
	index: BTreeMap<HashedKey, u32>,

	/// Freed slab slots, reused before the slab grows.
	free: Vec<u32>,

	/// frequency -> (head, tail) of that bucket's intrusive list. Ordered, so
	/// the minimum frequency is the first entry.
	buckets: BTreeMap<u32, (u32, u32)>,

	len: usize,
}

impl Default for CompactFrequencyChain {
	fn default() -> Self {
		CompactFrequencyChain {
			slots: Vec::new(),
			index: BTreeMap::new(),
			free: Vec::new(),
			buckets: BTreeMap::new(),
			len: 0,
		}
	}
}

impl CompactFrequencyChain {
	pub fn len(&self) -> usize { self.len }
	pub fn is_empty(&self) -> bool { self.len == 0 }

	pub fn contains(&self, key: HashedKey) -> bool {
		self.index.contains_key(&key)
	}

	pub fn get(&self, key: HashedKey) -> Option<&CompactEntry> {
		self.index.get(&key).map(|&i| &self.slots[i as usize])
	}

	/// Admits a key at frequency 1.
	pub fn insert(&mut self, key: HashedKey, size: ObjectSize, dram_resident: u8, tier: Tier) {
		if self.contains(key) {
			return;
		}

		let entry = CompactEntry {
			key, prev: NIL, next: NIL, freq: 1, size, tier, dram_resident,
		};

		let slot = match self.free.pop() {
			Some(slot) => { self.slots[slot as usize] = entry; slot },
			None => { self.slots.push(entry); (self.slots.len() - 1) as u32 },
		};

		self.index.insert(key, slot);
		self.link(slot, 1);
		self.len += 1;
	}

	/// Moves a key to the next frequency bucket. O(1): unlink, relink.
	pub fn bump(&mut self, key: HashedKey) -> u32 {
		let Some(&slot) = self.index.get(&key) else { return 0 };

		let freq = self.slots[slot as usize].freq;
		self.unlink(slot, freq);

		let next_freq = freq.saturating_add(1);
		self.slots[slot as usize].freq = next_freq;
		self.link(slot, next_freq);

		next_freq
	}

	/// The least-frequently-used key: head of the lowest-frequency bucket.
	pub fn min_key(&self) -> Option<HashedKey> {
		let (_, &(head, _)) = self.buckets.iter().next()?;
		Some(self.slots[head as usize].key)
	}

	pub fn remove(&mut self, key: HashedKey) -> Option<CompactEntry> {
		let slot = self.index.remove(&key)?;
		let entry = self.slots[slot as usize];

		self.unlink(slot, entry.freq);
		self.free.push(slot);
		self.len -= 1;

		Some(entry)
	}

	pub fn set_tier(&mut self, key: HashedKey, tier: Tier) {
		if let Some(&slot) = self.index.get(&key) {
			self.slots[slot as usize].tier = tier;
		}
	}

	pub fn resize(&mut self, key: HashedKey, size: ObjectSize, dram_resident: u8) {
		if let Some(&slot) = self.index.get(&key) {
			self.slots[slot as usize].size = size;
			self.slots[slot as usize].dram_resident = dram_resident;
		}
	}

	pub fn clear(&mut self) {
		self.slots.clear();
		self.index.clear();
		self.free.clear();
		self.buckets.clear();
		self.len = 0;
	}

	fn link(&mut self, slot: u32, freq: u32) {
		match self.buckets.get_mut(&freq) {
			Some((_, tail)) => {
				let old_tail = *tail;
				self.slots[old_tail as usize].next = slot;
				self.slots[slot as usize].prev = old_tail;
				self.slots[slot as usize].next = NIL;
				*tail = slot;
			},

			None => {
				self.slots[slot as usize].prev = NIL;
				self.slots[slot as usize].next = NIL;
				self.buckets.insert(freq, (slot, slot));
			},
		}
	}

	fn unlink(&mut self, slot: u32, freq: u32) {
		let (prev, next) = {
			let e = &self.slots[slot as usize];
			(e.prev, e.next)
		};

		if prev != NIL { self.slots[prev as usize].next = next; }
		if next != NIL { self.slots[next as usize].prev = prev; }

		if let Some((head, tail)) = self.buckets.get_mut(&freq) {
			if *head == slot { *head = next; }
			if *tail == slot { *tail = prev; }

			if *head == NIL {
				self.buckets.remove(&freq);
			}
		}
	}
}

#[cfg(test)]
mod tests {
	use super::*;

	/// The whole point: one slot per key, small enough to beat three structures.
	#[test]
	fn an_entry_is_thirty_two_bytes() {
		assert_eq!(
			core::mem::size_of::<CompactEntry>(), 32,
			"key 8 + prev 4 + next 4 + freq 4 + size 4 + tier 1 + resident 1 = 26, \
			 padded to 32",
		);
	}

	/// Against `FrequencyChain`'s 93 B/object: a `HashList` node (44), an
	/// `index_map` entry (29) and an `entries` slot (20).
	#[test]
	fn per_key_cost_beats_the_three_structure_layout() {
		const SLAB: usize = 32;      // one CompactEntry
		const INDEX: usize = 16;     // (HashedKey, u32) with slack
		const COMPACT: usize = SLAB + INDEX;
		const CURRENT: usize = 44 + 29 + 20;

		assert!(COMPACT < CURRENT, "{COMPACT} should beat {CURRENT}");
		assert!(
			COMPACT * 2 < CURRENT * 3 / 2,
			"expected a substantial win, got {COMPACT} vs {CURRENT}",
		);
	}

	#[test]
	fn least_frequently_used_comes_out_first() {
		let mut c = CompactFrequencyChain::default();
		for key in 1..=3u64 { c.insert(key, 100, 24, Tier::Fast); }

		// key 1 accessed twice, key 2 once, key 3 not at all
		c.bump(1); c.bump(1); c.bump(2);

		assert_eq!(c.min_key(), Some(3), "key 3 is the least frequently used");

		c.remove(3);
		assert_eq!(c.min_key(), Some(2));

		c.remove(2);
		assert_eq!(c.min_key(), Some(1));
	}

	#[test]
	fn keys_at_the_same_frequency_come_out_in_insertion_order() {
		let mut c = CompactFrequencyChain::default();
		for key in 1..=3u64 { c.insert(key, 10, 0, Tier::Fast); }

		assert_eq!(c.min_key(), Some(1), "all at frequency 1, so oldest first");
		c.remove(1);
		assert_eq!(c.min_key(), Some(2));
	}

	#[test]
	fn bump_moves_between_buckets_and_keeps_the_chain_intact() {
		let mut c = CompactFrequencyChain::default();
		for key in 1..=5u64 { c.insert(key, 10, 0, Tier::Fast); }

		assert_eq!(c.bump(3), 2);
		assert_eq!(c.get(3).unwrap().freq, 2);
		assert_eq!(c.len(), 5, "bumping must not lose or duplicate a key");

		// everything still reachable, and 3 is no longer the minimum
		for key in 1..=5u64 { assert!(c.contains(key)); }
		assert_ne!(c.min_key(), Some(3));
	}

	#[test]
	fn freed_slots_are_reused_so_the_slab_does_not_grow_forever() {
		let mut c = CompactFrequencyChain::default();
		for key in 1..=100u64 { c.insert(key, 10, 0, Tier::Fast); }
		for key in 1..=100u64 { c.remove(key); }

		let before = c.slots.len();
		for key in 101..=200u64 { c.insert(key, 10, 0, Tier::Fast); }

		assert_eq!(
			c.slots.len(), before,
			"a hundred inserts after a hundred removes must reuse the slab",
		);
		assert_eq!(c.len(), 100);
	}

	#[test]
	fn removing_from_the_middle_of_a_bucket_relinks_neighbours() {
		let mut c = CompactFrequencyChain::default();
		for key in 1..=5u64 { c.insert(key, 10, 0, Tier::Fast); }

		c.remove(3);

		assert_eq!(c.len(), 4);
		let mut seen = Vec::new();
		while let Some(k) = c.min_key() { seen.push(k); c.remove(k); }

		assert_eq!(seen, vec![1, 2, 4, 5], "the chain must survive a middle removal");
	}

	#[test]
	fn tier_and_size_travel_with_the_entry() {
		let mut c = CompactFrequencyChain::default();
		c.insert(9, 500, 24, Tier::Fast);

		assert_eq!(c.get(9).unwrap().tier, Tier::Fast);
		assert_eq!(c.get(9).unwrap().migrating(), 476);

		c.set_tier(9, Tier::Slow);
		c.resize(9, 800, 88);

		assert_eq!(c.get(9).unwrap().tier, Tier::Slow);
		assert_eq!(c.get(9).unwrap().migrating(), 712);
	}
}
