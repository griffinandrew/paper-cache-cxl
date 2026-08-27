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

#[cfg(not(feature = "eviction_stacks_pmem"))]
use std::collections::HashMap;
#[cfg(feature = "eviction_stacks_pmem")]
use hashbrown::HashMap;
use std::collections::BTreeMap;

use crate::{
	HashedKey,
	NoHasher,
	object::ObjectSize,
	worker::policy::policy_stack::Tier,
};

/// `u32::MAX` marks "no neighbour" / "free". A cache holding 4.29 billion
/// tracked keys would exhaust the index space long after it exhausted DRAM.
const NIL: u32 = u32::MAX;

/// Per-key data, carried in the INDEX VALUE rather than in the slab slot.
///
/// 12 bytes. This is layout B, and for LFU it is a pure win rather than the
/// trade it is elsewhere. Measured across the hash load cycle:
///
/// ```text
/// n           load     slot     index    delta
/// 4,194,304   50.0%   72.11     72.14    +0.03
/// 6,291,456   75.0%   69.41     58.76   -10.65
/// 7,340,032   87.5%   59.49     50.37    -9.13
/// ```
///
/// It is smaller here and larger for the 8-byte payloads because a 12-byte
/// payload removed from a slot with a `u64` key also recovers alignment
/// padding: the slot goes 32 -> 16, not 32 -> 20. That 16-byte saving outweighs
/// the 8 bytes the wider bucket costs, which the smaller payloads do not manage.
///
/// It is also faster on both paths -- 41.0 ns against 77.3 for a metadata read,
/// 344 against 393 for a list operation -- because the denser slab touches
/// fewer cache lines.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CompactEntry {
	pub freq: u32,

	pub size: ObjectSize,
	pub tier: Tier,

	pub dram_resident: u8,
}

const _: () = assert!(
	std::mem::size_of::<CompactEntry>() == 12,
	"CompactEntry grew past 12 bytes",
);

impl CompactEntry {
	/// Bytes that actually move between tiers.
	pub fn migrating(&self) -> u64 {
		(self.size as u64).saturating_sub(self.dram_resident as u64)
	}
}

/// One slab node: links only. 16 bytes, exactly, with no padding to spare.
#[derive(Clone, Copy, Debug)]
struct CompactSlot {
	key: HashedKey,
	prev: u32,
	next: u32,
}

const _: () = assert!(
	std::mem::size_of::<CompactSlot>() == 16,
	"CompactSlot grew past 16 bytes",
);

// Under `eviction_stacks_pmem` every structure here is allocated through the
// crate-wide `Hybrid` allocator, which binds to the far (CXL/PMEM) NUMA node,
// exactly as the stacks this one replaces do. The logic is identical either
// way; only the backing memory moves.
//
// Not optional: `get_hybrid_dram_shared_overhead` drops the eviction-stack term
// to ZERO under this feature, on the premise the stack is not in DRAM. A slab
// stack without these gates would sit in DRAM and be charged nothing -- silently
// wrong rather than merely unoptimised, and it would invalidate any experiment
// measuring what far-memory index placement costs.
//
// A slab is also a better shape for far memory than what it replaces: a
// `HashList` puts every node in its own small allocation, so relocating it
// scatters thousands of separate objects across the far node, where the slab is
// ONE contiguous region.
#[cfg(not(feature = "eviction_stacks_pmem"))]
type SlotVec = Vec<CompactSlot>;
#[cfg(feature = "eviction_stacks_pmem")]
type SlotVec = Vec<CompactSlot, crate::Hybrid>;

#[cfg(not(feature = "eviction_stacks_pmem"))]
type FreeVec = Vec<u32>;
#[cfg(feature = "eviction_stacks_pmem")]
type FreeVec = Vec<u32, crate::Hybrid>;

#[cfg(not(feature = "eviction_stacks_pmem"))]
type SlotIndex = HashMap<HashedKey, (u32, CompactEntry), NoHasher>;
#[cfg(feature = "eviction_stacks_pmem")]
type SlotIndex = HashMap<HashedKey, (u32, CompactEntry), NoHasher, crate::Hybrid>;

/// One entry per DISTINCT frequency rather than per object, so this stays small.
#[cfg(not(feature = "eviction_stacks_pmem"))]
type BucketMap = BTreeMap<u32, (u32, u32)>;
#[cfg(feature = "eviction_stacks_pmem")]
type BucketMap = BTreeMap<u32, (u32, u32), crate::Hybrid>;

#[cfg(not(feature = "eviction_stacks_pmem"))]
fn new_collections() -> (SlotVec, SlotIndex, FreeVec, BucketMap, BucketMap) {
	(Vec::new(), HashMap::default(), Vec::new(), BTreeMap::new(), BTreeMap::new())
}

#[cfg(feature = "eviction_stacks_pmem")]
fn new_collections() -> (SlotVec, SlotIndex, FreeVec, BucketMap, BucketMap) {
	(
		Vec::new_in(crate::Hybrid),
		HashMap::with_hasher_in(NoHasher::default(), crate::Hybrid),
		Vec::new_in(crate::Hybrid),
		BTreeMap::new_in(crate::Hybrid),
		BTreeMap::new_in(crate::Hybrid),
	)
}


pub struct CompactFrequencyChain {
	slots: SlotVec,
	/// `NoHasher`: `HashedKey` is already a hash, so a lookup is mask, probe,
	/// compare -- no hashing. `buckets` stays a `BTreeMap` because the minimum
	/// frequency must be ordered; this one only needs point lookups.
	index: SlotIndex,

	/// Freed slab slots, reused before the slab grows.
	free: FreeVec,

	/// frequency -> (head, tail) of that bucket's intrusive list, one map per
	/// tier. Ordered, so a tier's minimum frequency is its first entry.
	///
	/// Two bucket sets over *one* slab is what lets this replace both
	/// `FrequencyChain`s **and** the `entries` map they were paired with: a key
	/// is located in a single probe, and the slot that probe returns already
	/// carries its tier, size and frequency. `LfuHybridStack` needs three
	/// lookups across three structures for the same information.
	fast_buckets: BucketMap,
	slow_buckets: BucketMap,

	fast_len: usize,
	slow_len: usize,

	/// Head and tail of the DISTINGUISHED RECENCY LIST: a third intrusive list
	/// over the SAME slab, ordered by recency rather than by frequency.
	///
	/// `LruLfuCompactHybridStack` needs a recency-ordered fast tier beside a
	/// frequency-ordered slow one. Those two populations are disjoint -- a key
	/// is in the fast tier or the slow tier, never both -- so one `prev`/`next`
	/// pair per slot serves either, and `fast_len`/`slow_len` keep counting
	/// tier membership exactly as they do for LFU. That is what lets one slab
	/// and one index carry a policy whose tiers rank by different metrics.
	///
	/// The frequency-bucket stacks (`LfuCompactHybridStack`) never call a
	/// `recency_*` method, so for them these stay `NIL` for the structure's
	/// whole life and every other method behaves exactly as it did before.
	recency_head: u32,
	recency_tail: u32,
}

impl Default for CompactFrequencyChain {
	fn default() -> Self {
		let (slots, index, free, fast_buckets, slow_buckets) = new_collections();
		CompactFrequencyChain {
			slots,
			index,
			free,
			fast_buckets,
			slow_buckets,
			fast_len: 0,
			slow_len: 0,
			recency_head: NIL,
			recency_tail: NIL,
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

	/// One probe: the payload is already in the bucket.
	pub fn get(&self, key: HashedKey) -> Option<CompactEntry> {
		self.index.get(&key).map(|&(_, e)| e)
	}

	/// Admits a key at frequency 1.
	pub fn insert(&mut self, key: HashedKey, size: ObjectSize, dram_resident: u8, tier: Tier) {
		if self.contains(key) {
			return;
		}

		let slot = self.alloc_slot(key);

		self.index.insert(key, (slot, CompactEntry { freq: 1, size, tier, dram_resident }));
		self.link(slot, 1, tier);

		match tier {
			Tier::Fast => self.fast_len += 1,
			Tier::Slow => self.slow_len += 1,
		}
	}

	/// Moves a key to the next frequency bucket. O(1): unlink, relink.
	pub fn bump(&mut self, key: HashedKey) -> u32 {
		let Some(&(slot, entry)) = self.index.get(&key) else { return 0 };
		let (freq, tier) = (entry.freq, entry.tier);

		self.unlink(slot, freq, tier);

		let next_freq = freq.saturating_add(1);
		if let Some((_, e)) = self.index.get_mut(&key) {
			e.freq = next_freq;
		}
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
		let (slot, entry) = self.index.remove(&key)?;

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
		let Some(&(slot, entry)) = self.index.get(&key) else { return };
		let (freq, old_tier) = (entry.freq, entry.tier);

		if old_tier == tier {
			return;
		}

		self.unlink(slot, freq, old_tier);
		if let Some((_, e)) = self.index.get_mut(&key) {
			e.tier = tier;
		}
		self.link(slot, freq, tier);

		match tier {
			Tier::Fast => { self.fast_len += 1; self.slow_len -= 1; },
			Tier::Slow => { self.slow_len += 1; self.fast_len -= 1; },
		}
	}

	pub fn resize(&mut self, key: HashedKey, size: ObjectSize, dram_resident: u8) {
		if let Some((_, e)) = self.index.get_mut(&key) {
			e.size = size;
			e.dram_resident = dram_resident;
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
		self.recency_head = NIL;
		self.recency_tail = NIL;
	}

	/// Takes a free slab slot, or grows the slab by one. Extracted from
	/// `insert` unchanged so the recency admissions below allocate the same
	/// way -- one slab, one free list, whichever list the key joins.
	fn alloc_slot(&mut self, key: HashedKey) -> u32 {
		let node = CompactSlot { key, prev: NIL, next: NIL };

		match self.free.pop() {
			Some(slot) => { self.slots[slot as usize] = node; slot },
			None => { self.slots.push(node); (self.slots.len() - 1) as u32 },
		}
	}

	// ── the distinguished recency list ────────────────────────────────────
	//
	// Everything below is additive: it maintains `recency_head`/`recency_tail`
	// over the same slots the frequency buckets use, and touches
	// `fast_buckets` never. `LfuCompactHybridStack` calls none of it.

	fn recency_link_front(&mut self, slot: u32) {
		let old = self.recency_head;

		{
			let s = &mut self.slots[slot as usize];
			s.prev = NIL;
			s.next = old;
		}

		match old {
			NIL => self.recency_tail = slot,
			o => self.slots[o as usize].prev = slot,
		}

		self.recency_head = slot;
	}

	fn recency_unlink(&mut self, slot: u32) {
		let (prev, next) = {
			let s = &self.slots[slot as usize];
			(s.prev, s.next)
		};

		match prev {
			NIL => self.recency_head = next,
			p => self.slots[p as usize].next = next,
		}

		match next {
			NIL => self.recency_tail = prev,
			n => self.slots[n as usize].prev = prev,
		}
	}

	/// Admits a NEW key at the recency head, in the fast tier, at `freq`.
	///
	/// The frequency is carried metadata here, not a ranking key: nothing in
	/// the recency list is ordered by it. It exists so a later demotion can
	/// enter the slow tier at the count the key actually earned.
	pub fn recency_push_front(
		&mut self,
		key: HashedKey,
		size: ObjectSize,
		dram_resident: u8,
		freq: u32,
	) {
		if self.contains(key) {
			return;
		}

		let slot = self.alloc_slot(key);

		self.index.insert(key, (slot, CompactEntry { freq, size, tier: Tier::Fast, dram_resident }));
		self.recency_link_front(slot);

		self.fast_len += 1;
	}

	/// Moves an existing recency-list key to the head. O(1).
	pub fn recency_move_front(&mut self, key: HashedKey) {
		let Some(&(slot, _)) = self.index.get(&key) else { return };

		if self.recency_head == slot {
			return;
		}

		self.recency_unlink(slot);
		self.recency_link_front(slot);
	}

	/// The LRU end of the recency list: the demotion (and last-resort
	/// eviction) candidate.
	pub fn recency_back(&self) -> Option<HashedKey> {
		(self.recency_tail != NIL).then(|| self.slots[self.recency_tail as usize].key)
	}

	/// Removes a recency-list key outright, freeing its slot.
	pub fn recency_remove(&mut self, key: HashedKey) -> Option<CompactEntry> {
		let (slot, entry) = self.index.remove(&key)?;

		self.recency_unlink(slot);
		self.free.push(slot);
		self.fast_len -= 1;

		Some(entry)
	}

	/// Moves the recency tail into the slow tier, into the bucket for the
	/// frequency it already carries. Returns the demoted key and its entry as
	/// it now stands.
	///
	/// This is the whole of a demotion. `FrequencyChain` needs a `pop_back`
	/// from one structure and an `insert_at` into another with the count
	/// passed across by hand; here the entry never moves in the slab, so the
	/// count is carried by construction and there is nothing to drop.
	pub fn demote_recency_back(&mut self) -> Option<(HashedKey, CompactEntry)> {
		if self.recency_tail == NIL {
			return None;
		}

		let slot = self.recency_tail;
		let key = self.slots[slot as usize].key;

		let Some((_, e)) = self.index.get_mut(&key) else { return None };
		e.tier = Tier::Slow;
		let entry = *e;

		self.recency_unlink(slot);
		self.link(slot, entry.freq, Tier::Slow);

		self.fast_len -= 1;
		self.slow_len += 1;

		Some((key, entry))
	}

	/// Moves a slow-tier key to the recency head, setting its frequency to
	/// `freq`. The whole of a promotion; `None` if the key is untracked or is
	/// not in the slow tier.
	pub fn promote_to_recency_front(&mut self, key: HashedKey, freq: u32) -> Option<CompactEntry> {
		let &(slot, entry) = self.index.get(&key)?;

		if entry.tier != Tier::Slow {
			return None;
		}

		self.unlink(slot, entry.freq, Tier::Slow);

		let (_, e) = self.index.get_mut(&key)?;
		e.tier = Tier::Fast;
		e.freq = freq;
		let entry = *e;

		self.recency_link_front(slot);

		self.slow_len -= 1;
		self.fast_len += 1;

		Some(entry)
	}

	/// Sets a key's frequency and touches no list.
	///
	/// For a recency-list key only: its counter is carried metadata that ranks
	/// nothing, so there is no bucket to move it between. Calling this on a
	/// bucketed key would leave the buckets keyed on a stale frequency.
	pub fn set_freq(&mut self, key: HashedKey, freq: u32) {
		if let Some((_, e)) = self.index.get_mut(&key) {
			e.freq = freq;
		}
	}

	/// Sets a SLOW key's frequency and relinks it into that bucket.
	///
	/// Relinks even when `freq` is unchanged, which moves the key to the
	/// newest position within its bucket. That is deliberate and matches
	/// `FrequencyChain::move_to`'s unconditional remove-then-insert: at the
	/// frequency cap a further access cannot raise the count, but it still
	/// refreshes the key's standing against its equally-frequent peers.
	pub fn slow_relink_at(&mut self, key: HashedKey, freq: u32) {
		let Some(&(slot, entry)) = self.index.get(&key) else { return };

		if entry.tier != Tier::Slow {
			return;
		}

		self.unlink(slot, entry.freq, Tier::Slow);

		if let Some((_, e)) = self.index.get_mut(&key) {
			e.freq = freq;
		}

		self.link(slot, freq, Tier::Slow);
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

	/// Pins both halves of the split. Growth in either costs bytes on EVERY
	/// tracked object across both tiers, which is the point of the structure.
	///
	/// The payload was in the slot until it was measured: at 32 bytes per slot
	/// plus a bare u32 index, versus 16 plus a 12-byte payload in the bucket,
	/// the second is 9 to 11 B/object smaller at realistic hash loads AND
	/// faster on both access paths. The slot dropping 32 -> 16 rather than
	/// 32 -> 20 is why -- removing the payload also recovers alignment padding.
	#[test]
	fn the_split_layout_is_sixteen_plus_twelve() {
		assert_eq!(
			std::mem::size_of::<CompactSlot>(),
			16,
			"key 8 + prev 4 + next 4 = 16, with no padding to spare",
		);
		assert_eq!(
			std::mem::size_of::<CompactEntry>(),
			12,
			"freq 4 + size 4 + tier 1 + resident 1 = 10, padded to 12",
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

	// ── the distinguished recency list ────────────────────────────────────

	#[test]
	fn the_recency_list_orders_by_recency_not_frequency() {
		let mut c = CompactFrequencyChain::default();
		for key in 1..=3u64 { c.recency_push_front(key, 10, 0, 1); }

		// 3 was pushed last, so 1 is the LRU tail regardless of counts.
		assert_eq!(c.recency_back(), Some(1));
		assert_eq!(c.fast_len(), 3);
		assert_eq!(c.slow_len(), 0);

		c.set_freq(1, 9);
		assert_eq!(c.recency_back(), Some(1), "frequency must not reorder the recency list");

		c.recency_move_front(1);
		assert_eq!(c.recency_back(), Some(2), "a touch moves the key off the tail");
	}

	#[test]
	fn moving_the_recency_head_to_the_front_is_a_no_op() {
		let mut c = CompactFrequencyChain::default();
		for key in 1..=3u64 { c.recency_push_front(key, 10, 0, 1); }

		c.recency_move_front(3);

		assert_eq!(c.recency_back(), Some(1));
		assert_eq!(c.fast_len(), 3);
	}

	#[test]
	fn demotion_carries_the_count_into_the_slow_buckets() {
		let mut c = CompactFrequencyChain::default();
		c.recency_push_front(1, 100, 24, 1);
		c.recency_push_front(2, 200, 24, 1);
		c.set_freq(2, 7);
		c.recency_move_front(2);

		// 1 is the tail; demoting it must land it at ITS count, not 2's.
		let (key, entry) = c.demote_recency_back().unwrap();

		assert_eq!(key, 1);
		assert_eq!(entry.tier, Tier::Slow);
		assert_eq!(entry.freq, 1);
		assert_eq!(entry.size, 100, "size survives the move");
		assert_eq!(c.min_with_count(Tier::Slow), Some((1, 1)));
		assert_eq!(c.fast_len(), 1);
		assert_eq!(c.slow_len(), 1);
		assert_eq!(c.recency_back(), Some(2), "the recency list closed over the gap");

		// and a hot key demotes into a HIGHER bucket than a cold one
		let (key, entry) = c.demote_recency_back().unwrap();
		assert_eq!(key, 2);
		assert_eq!(entry.freq, 7);
		assert_eq!(c.min_with_count(Tier::Slow), Some((1, 1)), "the cold key still ranks lowest");
		assert_eq!(c.recency_back(), None);
		assert_eq!(c.fast_len(), 0);
		assert_eq!(c.slow_len(), 2);
	}

	#[test]
	fn promotion_leaves_the_slow_buckets_and_enters_the_recency_head() {
		let mut c = CompactFrequencyChain::default();
		c.recency_push_front(1, 10, 0, 1);
		c.recency_push_front(2, 10, 0, 1);
		c.demote_recency_back().unwrap(); // 1 -> slow

		let entry = c.promote_to_recency_front(1, 1).unwrap();

		assert_eq!(entry.tier, Tier::Fast);
		assert_eq!(entry.freq, 1, "the counter resets on the way in");
		assert_eq!(c.min_key(Tier::Slow), None, "the slow bucket is gone");
		assert_eq!(c.recency_back(), Some(2), "1 entered at the head, so 2 is now the tail");
		assert_eq!(c.fast_len(), 2);
		assert_eq!(c.slow_len(), 0);

		// promoting something that is not slow is a no-op
		assert_eq!(c.promote_to_recency_front(1, 1), None);
		assert_eq!(c.promote_to_recency_front(999, 1), None);
	}

	#[test]
	fn relinking_a_slow_key_at_an_unchanged_count_still_refreshes_it() {
		let mut c = CompactFrequencyChain::default();
		for key in 1..=3u64 { c.recency_push_front(key, 10, 0, 4); }
		for _ in 0..3 { c.demote_recency_back().unwrap(); }

		// all three sit in bucket 4, oldest-demoted first
		assert_eq!(c.min_with_count(Tier::Slow), Some((1, 4)));

		c.slow_relink_at(1, 4);

		assert_eq!(
			c.min_key(Tier::Slow), Some(2),
			"an unchanged relink must still move the key to the back of its bucket",
		);

		c.slow_relink_at(2, 9);
		assert_eq!(c.get(2).unwrap().freq, 9);
		assert_eq!(c.min_key(Tier::Slow), Some(3), "2 left the minimum bucket");
	}

	#[test]
	fn the_recency_list_and_the_slow_buckets_share_one_slab_and_one_index() {
		let mut c = CompactFrequencyChain::default();
		for key in 1..=100u64 { c.recency_push_front(key, 10, 0, 1); }
		for _ in 0..50 { c.demote_recency_back().unwrap(); }

		assert_eq!(c.len(), 100, "one index, so one count");
		assert_eq!(c.fast_len() + c.slow_len(), 100);
		assert_eq!(c.slots.len(), 100, "one slab, one slot per key");

		// every key is reachable through the single index
		for key in 1..=100u64 { assert!(c.contains(key)); }

		// removing through the right door frees the slot for reuse
		let before = c.slots.len();
		for key in 1..=50u64 { c.remove(key); }             // slow half
		for key in 51..=100u64 { c.recency_remove(key); }   // fast half

		assert_eq!(c.len(), 0);
		assert_eq!(c.fast_len(), 0);
		assert_eq!(c.slow_len(), 0);
		assert_eq!(c.recency_back(), None);

		for key in 201..=300u64 { c.recency_push_front(key, 10, 0, 1); }
		assert_eq!(c.slots.len(), before, "the freed slots must be reused");
	}

	#[test]
	fn removing_from_the_middle_of_the_recency_list_relinks_neighbours() {
		let mut c = CompactFrequencyChain::default();
		for key in 1..=5u64 { c.recency_push_front(key, 10, 0, 1); }

		c.recency_remove(3);

		let mut seen = Vec::new();
		while let Some(k) = c.recency_back() { seen.push(k); c.recency_remove(k); }

		assert_eq!(seen, vec![1, 2, 4, 5], "the recency list must survive a middle removal");
	}

	#[test]
	fn clear_resets_the_recency_list_too() {
		let mut c = CompactFrequencyChain::default();
		for key in 1..=5u64 { c.recency_push_front(key, 10, 0, 1); }
		c.demote_recency_back().unwrap();

		c.clear();

		assert_eq!(c.len(), 0);
		assert_eq!(c.recency_back(), None);

		// and it is usable again afterwards
		c.recency_push_front(9, 10, 0, 1);
		assert_eq!(c.recency_back(), Some(9));
		assert_eq!(c.fast_len(), 1);
	}

	/// The recency list must be invisible to the frequency-bucket stacks: a
	/// chain driven only through `insert`/`bump`/`set_tier`/`remove` -- which
	/// is exactly what `LfuCompactHybridStack` does -- must leave it empty.
	#[test]
	fn the_frequency_only_path_never_touches_the_recency_list() {
		let mut c = CompactFrequencyChain::default();
		for key in 1..=10u64 { c.insert(key, 10, 0, Tier::Fast); }
		for key in 1..=5u64 { c.bump(key); }
		c.set_tier(3, Tier::Slow);
		c.remove(7);

		assert_eq!(c.recency_back(), None, "no recency link may exist on the LFU path");
		assert_eq!(c.recency_head, NIL);
		assert_eq!(c.recency_tail, NIL);
		assert_eq!(c.fast_len() + c.slow_len(), 9);
	}
}
