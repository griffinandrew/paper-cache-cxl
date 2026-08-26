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

use std::collections::{BTreeMap, HashMap};

use crate::{
	HashedKey,
	NoHasher,
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
	/// `NoHasher`: `HashedKey` is already a hash, so a lookup is mask, probe,
	/// compare -- no hashing. `buckets` stays a `BTreeMap` because the minimum
	/// frequency must be ordered; this one only needs point lookups.
	index: HashMap<HashedKey, u32, NoHasher>,

	/// Freed slab slots, reused before the slab grows.
	free: Vec<u32>,

	/// frequency -> (head, tail) of that bucket's intrusive list, one map per
	/// tier. Ordered, so a tier's minimum frequency is its first entry.
	///
	/// Two bucket sets over *one* slab is what lets this replace both
	/// `FrequencyChain`s **and** the `entries` map they were paired with: a key
	/// is located in a single probe, and the slot that probe returns already
	/// carries its tier, size and frequency. `LfuHybridStack` needs three
	/// lookups across three structures for the same information.
	fast_buckets: BTreeMap<u32, (u32, u32)>,
	slow_buckets: BTreeMap<u32, (u32, u32)>,

	fast_len: usize,
	slow_len: usize,
}

impl Default for CompactFrequencyChain {
	fn default() -> Self {
		CompactFrequencyChain {
			slots: Vec::new(),
			index: HashMap::default(),
			free: Vec::new(),
			fast_buckets: BTreeMap::new(),
			slow_buckets: BTreeMap::new(),
			fast_len: 0,
			slow_len: 0,
		}
	}
}

impl CompactFrequencyChain {
	/// Pre-sizes the slab and index for `objects` entries.
	///
	/// The slab is a `Vec`, so growth is never in place: every doubling
	/// reallocates and COPIES every entry. At eval-trace scale that is one
	/// multi-hundred-millisecond stall on the policy worker -- measured at
	/// 827 ms -- and it would never have surfaced as a regression, because the
	/// policy stack runs behind an unbounded channel on its own thread and the
	/// client latency columns structurally cannot observe it.
	///
	/// Reserving costs no resident memory: the pages are not touched until
	/// entries occupy them.
	pub fn reserve(&mut self, objects: usize) {
		self.slots.reserve(objects);
		self.index.reserve(objects);
	}

	pub fn len(&self) -> usize { self.fast_len + self.slow_len }
	pub fn is_empty(&self) -> bool { self.len() == 0 }
	pub fn fast_len(&self) -> usize { self.fast_len }
	pub fn slow_len(&self) -> usize { self.slow_len }

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
		self.link(slot, 1, tier);

		match tier {
			Tier::Fast => self.fast_len += 1,
			Tier::Slow => self.slow_len += 1,
		}
	}

	/// Moves a key to the next frequency bucket. O(1): unlink, relink.
	pub fn bump(&mut self, key: HashedKey) -> u32 {
		let Some(&slot) = self.index.get(&key) else { return 0 };

		let (freq, tier) = {
			let e = &self.slots[slot as usize];
			(e.freq, e.tier)
		};

		self.unlink(slot, freq, tier);

		let next_freq = freq.saturating_add(1);
		self.slots[slot as usize].freq = next_freq;
		self.link(slot, next_freq, tier);

		next_freq
	}

	/// The least-frequently-used key in a tier: head of its lowest-frequency
	/// bucket. O(log D) in the number of distinct frequencies present -- the
	/// original's `VecList::front` is O(1), which is headroom this design has
	/// not taken yet.
	pub fn min_key(&self, tier: Tier) -> Option<HashedKey> {
		let buckets = match tier {
			Tier::Fast => &self.fast_buckets,
			Tier::Slow => &self.slow_buckets,
		};

		let (_, &(head, _)) = buckets.iter().next()?;
		Some(self.slots[head as usize].key)
	}

	/// The lowest frequency present in a tier, or `None` if it is empty.
	///
	/// The promotion rule compares a slow key's new count against this: a slow
	/// key overtakes the fast tier only by *strictly* exceeding its minimum.
	pub fn min_count(&self, tier: Tier) -> Option<u32> {
		let buckets = match tier {
			Tier::Fast => &self.fast_buckets,
			Tier::Slow => &self.slow_buckets,
		};

		buckets.keys().next().copied()
	}

	/// The least-frequently-used key in a tier together with its count.
	pub fn min_with_count(&self, tier: Tier) -> Option<(HashedKey, u32)> {
		let buckets = match tier {
			Tier::Fast => &self.fast_buckets,
			Tier::Slow => &self.slow_buckets,
		};

		let (&freq, &(head, _)) = buckets.iter().next()?;
		Some((self.slots[head as usize].key, freq))
	}

	pub fn remove(&mut self, key: HashedKey) -> Option<CompactEntry> {
		let slot = self.index.remove(&key)?;
		let entry = self.slots[slot as usize];

		self.unlink(slot, entry.freq, entry.tier);
		self.free.push(slot);

		match entry.tier {
			Tier::Fast => self.fast_len -= 1,
			Tier::Slow => self.slow_len -= 1,
		}

		Some(entry)
	}

	/// Moves a key between tiers, preserving its frequency. Relinks it from one
	/// bucket set into the other -- the key never moves in the slab, so its
	/// index entry and every link to it stay valid.
	pub fn set_tier(&mut self, key: HashedKey, tier: Tier) {
		let Some(&slot) = self.index.get(&key) else { return };

		let (freq, old_tier) = {
			let e = &self.slots[slot as usize];
			(e.freq, e.tier)
		};

		if old_tier == tier {
			return;
		}

		self.unlink(slot, freq, old_tier);
		self.slots[slot as usize].tier = tier;
		self.link(slot, freq, tier);

		match tier {
			Tier::Fast => { self.fast_len += 1; self.slow_len -= 1; },
			Tier::Slow => { self.slow_len += 1; self.fast_len -= 1; },
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
		self.fast_buckets.clear();
		self.slow_buckets.clear();
		self.fast_len = 0;
		self.slow_len = 0;
	}

	fn link(&mut self, slot: u32, freq: u32, tier: Tier) {
		let buckets = match tier {
			Tier::Fast => &mut self.fast_buckets,
			Tier::Slow => &mut self.slow_buckets,
		};

		match buckets.get_mut(&freq) {
			Some((_, tail)) => {
				let old_tail = *tail;
				*tail = slot;
				self.slots[old_tail as usize].next = slot;
				self.slots[slot as usize].prev = old_tail;
				self.slots[slot as usize].next = NIL;
			},

			None => {
				buckets.insert(freq, (slot, slot));
				self.slots[slot as usize].prev = NIL;
				self.slots[slot as usize].next = NIL;
			},
		}
	}

	fn unlink(&mut self, slot: u32, freq: u32, tier: Tier) {
		let (prev, next) = {
			let e = &self.slots[slot as usize];
			(e.prev, e.next)
		};

		if prev != NIL { self.slots[prev as usize].next = next; }
		if next != NIL { self.slots[next as usize].prev = prev; }

		let buckets = match tier {
			Tier::Fast => &mut self.fast_buckets,
			Tier::Slow => &mut self.slow_buckets,
		};

		if let Some((head, tail)) = buckets.get_mut(&freq) {
			if *head == slot { *head = next; }
			if *tail == slot { *tail = prev; }

			if *head == NIL {
				buckets.remove(&freq);
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

		assert_eq!(c.min_key(Tier::Fast), Some(3), "key 3 is the least frequently used");

		c.remove(3);
		assert_eq!(c.min_key(Tier::Fast), Some(2));

		c.remove(2);
		assert_eq!(c.min_key(Tier::Fast), Some(1));
	}

	#[test]
	fn keys_at_the_same_frequency_come_out_in_insertion_order() {
		let mut c = CompactFrequencyChain::default();
		for key in 1..=3u64 { c.insert(key, 10, 0, Tier::Fast); }

		assert_eq!(c.min_key(Tier::Fast), Some(1), "all at frequency 1, so oldest first");
		c.remove(1);
		assert_eq!(c.min_key(Tier::Fast), Some(2));
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
		assert_ne!(c.min_key(Tier::Fast), Some(3));
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
		while let Some(k) = c.min_key(Tier::Fast) { seen.push(k); c.remove(k); }

		assert_eq!(seen, vec![1, 2, 4, 5], "the chain must survive a middle removal");
	}

	/// Two bucket sets over one slab: a tier move relinks the key without
	/// touching the slab or the index, so every other link stays valid.

	/// A tier move is the *only* thing that happens on promotion or demotion.
	///
	/// `FrequencyChain` has to `remove` from one chain and `insert_at` into the
	/// other, carrying the count across by hand. Here the entry never moves in
	/// the slab, so its frequency, size and links are all preserved by
	/// construction -- there is no count to carry and nothing to get wrong.
	#[test]
	fn promotion_is_a_tier_move_and_nothing_else() {
		let mut c = CompactFrequencyChain::default();
		c.insert(1, 100, 24, Tier::Fast);
		c.insert(2, 200, 24, Tier::Slow);
		for _ in 0..5 { c.bump(2); }

		assert_eq!(c.min_count(Tier::Fast), Some(1));
		assert_eq!(c.min_count(Tier::Slow), Some(6));

		// key 2 strictly exceeds the fast minimum, so it promotes
		c.set_tier(2, Tier::Fast);

		assert_eq!(c.get(2).unwrap().freq, 6, "count survives the move");
		assert_eq!(c.get(2).unwrap().size, 200, "size survives the move");
		assert_eq!(c.min_count(Tier::Slow), None);
		assert_eq!(c.min_with_count(Tier::Fast), Some((1, 1)),
			"key 1 is still the fast minimum at count 1");
	}

	#[test]
	fn moving_a_key_between_tiers_preserves_its_frequency_and_position() {
		let mut c = CompactFrequencyChain::default();
		for key in 1..=3u64 { c.insert(key, 100, 24, Tier::Fast); }
		c.bump(2); c.bump(2);

		assert_eq!(c.fast_len(), 3);
		assert_eq!(c.slow_len(), 0);
		assert_eq!(c.min_key(Tier::Fast), Some(1));

		c.set_tier(2, Tier::Slow);

		assert_eq!(c.fast_len(), 2);
		assert_eq!(c.slow_len(), 1);
		assert_eq!(c.get(2).unwrap().freq, 3, "frequency must survive the move");
		assert_eq!(c.get(2).unwrap().tier, Tier::Slow);
		assert_eq!(c.min_key(Tier::Slow), Some(2));
		assert_eq!(c.min_key(Tier::Fast), Some(1), "the fast chain is intact");

		// and back again
		c.set_tier(2, Tier::Fast);
		assert_eq!(c.fast_len(), 3);
		assert_eq!(c.slow_len(), 0);
		assert_eq!(c.min_key(Tier::Slow), None);
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
