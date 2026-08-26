/*
 * Copyright (c) Kia Shakiba
 *
 * This source code is licensed under the GNU AGPLv3 license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! Slab-backed recency list: the LRU/FIFO counterpart of
//! [`CompactFrequencyChain`](super::compact_frequency_chain).
//!
//! `LruHybridStack` keeps a `kwik::collections::HashList` plus a separate
//! `entries` map. Both are keyed by the same `HashedKey` and both hold exactly
//! one row per object, because the `HashList` carries its own index
//! (`map: HashMap<DataRef<T>, NonNull<Entry<T>>>`) and cannot store a payload.
//! The payload is 8 bytes -- `tier`, `dram_resident`, `size` -- so an entire
//! hash map exists to hold what fits in the list node's padding. Measured, that
//! second map is **40 B/object**: all-DRAM LRU (one list, no `entries`) costs
//! 72 B/object and hybrid LRU costs 112.
//!
//! This holds one slab and one index. Entries are linked by `u32` slot indices
//! rather than 8-byte pointers, and the whole slab is one allocation instead of
//! one `malloc` per node.
//!
//! The payload lives in the SLOT rather than in the index map's value, which is
//! the smaller of the two layouts but costs an extra indirection on reads that
//! want metadata without touching list order. That trade is right *here* and
//! would be wrong for the 2Q and S3-FIFO families: LRU's hot path (`update` ->
//! `touch_fast_key`) always reorders the list, so it pays the indirection
//! anyway, whereas `mark_accessed` in the S3-FIFO family is metadata-only by
//! design and would become measurably slower. Those stacks should carry the
//! payload in the index value instead.

use std::collections::HashMap;

use crate::{
	object::ObjectSize,
	worker::policy::policy_stack::{CacheSize, HashedKey, Tier},
	NoHasher,
};

/// Sentinel for "no slot". `u32::MAX` rather than `Option<u32>` so a slot stays
/// 16 bytes: an `Option<u32>` would take 8 with padding and add 8 per entry
/// across both tiers.
pub const NIL: u32 = u32::MAX;

/// One tracked object. 24 bytes.
///
/// NOT free: `key` + `prev` + `next` already pack to exactly 16 with no
/// padding, so the six payload bytes cost a further 8 after alignment. The win
/// is still large -- 24 B here against a 32 B-size-class `HashList` node plus a
/// whole second hash-map row per object -- but it comes from eliminating the
/// second index, not from spare padding. An earlier version of this comment
/// claimed the payload rode free; `slot_is_twenty_four_bytes` exists because
/// it did not.
#[derive(Clone, Copy, Debug)]
pub struct RecencySlot {
	pub key: HashedKey,
	pub prev: u32,
	pub next: u32,
	pub size: ObjectSize,
	pub tier: Tier,
	/// The part of `size` that stays in DRAM whichever tier this entry is in:
	/// the key and expiry field inline in the object map, plus the `Expiries`
	/// entry when a TTL is set. `Object::set_data` replaces only the value
	/// buffer, so none of it migrates.
	pub dram_resident: u8,
}

impl RecencySlot {
	/// Bytes that actually move between tiers.
	pub fn migrating(&self) -> CacheSize {
		(self.size as CacheSize).saturating_sub(self.dram_resident as CacheSize)
	}
}

/// Doubly-linked recency order over a slab, MRU at `head`.
pub struct CompactRecencyList {
	slots: Vec<RecencySlot>,
	index: HashMap<HashedKey, u32, NoHasher>,
	free: Vec<u32>,
	head: u32,
	tail: u32,
	len: usize,
}

impl Default for CompactRecencyList {
	fn default() -> Self {
		CompactRecencyList {
			slots: Vec::new(),
			index: HashMap::default(),
			free: Vec::new(),
			head: NIL,
			tail: NIL,
			len: 0,
		}
	}
}

impl CompactRecencyList {
	/// Pre-sizes the slab and index for `objects` entries.
	///
	/// Growth is never in place: every `Vec` doubling reallocates and COPIES
	/// every entry, which at eval-trace scale is a single multi-hundred-
	/// millisecond stall on the policy worker. It would not show up as a
	/// regression either, because the policy stack sits behind an unbounded
	/// channel on its own thread and the client latency percentiles cannot
	/// observe it. Reserving costs no resident memory -- untouched pages are
	/// not resident -- so an over-estimate costs address space, not DRAM.
	pub fn reserve(&mut self, objects: usize) {
		self.slots.reserve(objects);
		self.index.reserve(objects);
	}

	pub fn len(&self) -> usize {
		self.len
	}

	pub fn is_empty(&self) -> bool {
		self.len == 0
	}

	pub fn contains(&self, key: HashedKey) -> bool {
		self.index.contains_key(&key)
	}

	pub fn slot_of(&self, key: HashedKey) -> Option<u32> {
		self.index.get(&key).copied()
	}

	pub fn get(&self, key: HashedKey) -> Option<&RecencySlot> {
		self.index.get(&key).map(|&i| &self.slots[i as usize])
	}

	pub fn get_mut(&mut self, key: HashedKey) -> Option<&mut RecencySlot> {
		match self.index.get(&key) {
			Some(&i) => Some(&mut self.slots[i as usize]),
			None => None,
		}
	}

	/// MRU key.
	pub fn front(&self) -> Option<HashedKey> {
		(self.head != NIL).then(|| self.slots[self.head as usize].key)
	}

	/// LRU key.
	pub fn back(&self) -> Option<HashedKey> {
		(self.tail != NIL).then(|| self.slots[self.tail as usize].key)
	}

	/// The key one step toward the MRU end, i.e. `HashList::before`.
	pub fn before(&self, key: HashedKey) -> Option<HashedKey> {
		let i = *self.index.get(&key)?;
		let p = self.slots[i as usize].prev;
		(p != NIL).then(|| self.slots[p as usize].key)
	}

	/// The key one step toward the LRU end.
	pub fn after(&self, key: HashedKey) -> Option<HashedKey> {
		let i = *self.index.get(&key)?;
		let n = self.slots[i as usize].next;
		(n != NIL).then(|| self.slots[n as usize].key)
	}

	fn unlink(&mut self, i: u32) {
		let (prev, next) = {
			let s = &self.slots[i as usize];
			(s.prev, s.next)
		};

		match prev {
			NIL => self.head = next,
			p => self.slots[p as usize].next = next,
		}

		match next {
			NIL => self.tail = prev,
			n => self.slots[n as usize].prev = prev,
		}
	}

	fn link_front(&mut self, i: u32) {
		let old = self.head;
		{
			let s = &mut self.slots[i as usize];
			s.prev = NIL;
			s.next = old;
		}

		match old {
			NIL => self.tail = i,
			o => self.slots[o as usize].prev = i,
		}

		self.head = i;
	}

	/// Inserts at the MRU end. Existing keys are moved rather than duplicated.
	pub fn insert_front(
		&mut self,
		key: HashedKey,
		size: ObjectSize,
		dram_resident: u8,
		tier: Tier,
	) {
		if let Some(&i) = self.index.get(&key) {
			{
				let s = &mut self.slots[i as usize];
				s.size = size;
				s.dram_resident = dram_resident;
				s.tier = tier;
			}
			self.move_front(key);
			return;
		}

		let slot = RecencySlot { key, prev: NIL, next: NIL, size, tier, dram_resident };
		let i = match self.free.pop() {
			Some(i) => {
				self.slots[i as usize] = slot;
				i
			},
			None => {
				let i = self.slots.len() as u32;
				assert!(i != NIL, "CompactRecencyList exceeded u32::MAX - 1 slots");
				self.slots.push(slot);
				i
			},
		};

		self.index.insert(key, i);
		self.link_front(i);
		self.len += 1;
	}

	/// Moves an existing key to the MRU end. No-op if absent or already there.
	pub fn move_front(&mut self, key: HashedKey) {
		let Some(&i) = self.index.get(&key) else { return };
		if self.head == i {
			return;
		}

		self.unlink(i);
		self.link_front(i);
	}

	/// Removes a key, returning its slot contents.
	pub fn remove(&mut self, key: HashedKey) -> Option<RecencySlot> {
		let i = self.index.remove(&key)?;
		self.unlink(i);
		self.free.push(i);
		self.len -= 1;
		Some(self.slots[i as usize])
	}

	pub fn clear(&mut self) {
		self.slots.clear();
		self.index.clear();
		self.free.clear();
		self.head = NIL;
		self.tail = NIL;
		self.len = 0;
	}
}

#[cfg(test)]
mod tests {
	use super::*;

	fn keys(list: &CompactRecencyList) -> Vec<HashedKey> {
		let mut out = Vec::new();
		let mut i = list.head;
		while i != NIL {
			out.push(list.slots[i as usize].key);
			i = list.slots[i as usize].next;
		}
		out
	}

	fn keys_reverse(list: &CompactRecencyList) -> Vec<HashedKey> {
		let mut out = Vec::new();
		let mut i = list.tail;
		while i != NIL {
			out.push(list.slots[i as usize].key);
			i = list.slots[i as usize].prev;
		}
		out.reverse();
		out
	}

	fn build(ks: &[HashedKey]) -> CompactRecencyList {
		let mut l = CompactRecencyList::default();
		for &k in ks {
			l.insert_front(k, 100, 0, Tier::Fast);
		}
		l
	}

	/// Pins the slot size. Anything added here costs bytes on EVERY tracked
	/// object across both tiers, which is the entire point of the structure, so
	/// growth must be a deliberate decision rather than a side effect.
	#[test]
	fn slot_is_twenty_four_bytes() {
		assert_eq!(core::mem::size_of::<RecencySlot>(), 24);
	}

	/// Every test below walks forward; this is the one that would catch a
	/// `prev` chain that disagrees with the `next` chain.
	#[test]
	fn forward_and_backward_orders_agree() {
		let mut l = build(&[1, 2, 3, 4, 5]);
		l.move_front(3);
		l.remove(1);
		l.insert_front(9, 100, 0, Tier::Fast);
		assert_eq!(keys(&l), keys_reverse(&l));
	}

	#[test]
	fn insert_front_orders_most_recent_first() {
		let l = build(&[1, 2, 3]);
		assert_eq!(keys(&l), vec![3, 2, 1]);
		assert_eq!(l.front(), Some(3));
		assert_eq!(l.back(), Some(1));
		assert_eq!(l.len(), 3);
	}

	#[test]
	fn move_front_promotes_without_duplicating() {
		let mut l = build(&[1, 2, 3]);
		l.move_front(1);
		assert_eq!(keys(&l), vec![1, 3, 2]);
		assert_eq!(l.len(), 3);
	}

	#[test]
	fn move_front_on_head_is_a_noop() {
		let mut l = build(&[1, 2, 3]);
		l.move_front(3);
		assert_eq!(keys(&l), vec![3, 2, 1]);
	}

	#[test]
	fn move_front_on_absent_key_is_a_noop() {
		let mut l = build(&[1, 2]);
		l.move_front(99);
		assert_eq!(keys(&l), vec![2, 1]);
	}

	/// `before` is what `settle_fast_tier` walks the tier boundary with, so it
	/// has to mean the same thing as `HashList::before`: one step toward MRU.
	#[test]
	fn before_and_after_step_in_the_expected_directions() {
		let l = build(&[1, 2, 3]);
		assert_eq!(l.before(2), Some(3));
		assert_eq!(l.after(2), Some(1));
		assert_eq!(l.before(3), None);
		assert_eq!(l.after(1), None);
	}

	#[test]
	fn remove_returns_the_slot_and_relinks_neighbours() {
		let mut l = build(&[1, 2, 3]);
		let slot = l.remove(2).expect("present");
		assert_eq!(slot.key, 2);
		assert_eq!(keys(&l), vec![3, 1]);
		assert_eq!(l.before(1), Some(3));
		assert_eq!(l.len(), 2);
		assert!(l.remove(2).is_none());
	}

	#[test]
	fn removing_head_and_tail_maintains_both_ends() {
		let mut l = build(&[1, 2, 3]);
		l.remove(3);
		assert_eq!(l.front(), Some(2));
		l.remove(1);
		assert_eq!(l.back(), Some(2));
		assert_eq!(keys(&l), vec![2]);
		l.remove(2);
		assert!(l.is_empty());
		assert_eq!(l.front(), None);
		assert_eq!(l.back(), None);
	}

	/// Freed slots must be reused, or the slab grows without bound under the
	/// insert/evict churn a cache runs at steady state.
	#[test]
	fn freed_slots_are_recycled() {
		let mut l = build(&[1, 2, 3]);
		let before = l.slots.len();
		l.remove(2);
		l.insert_front(4, 100, 0, Tier::Fast);
		assert_eq!(l.slots.len(), before, "slab grew instead of reusing the free slot");
		assert_eq!(keys(&l), vec![4, 3, 1]);
	}

	#[test]
	fn reinserting_an_existing_key_moves_it_and_updates_payload() {
		let mut l = build(&[1, 2, 3]);
		l.insert_front(1, 512, 7, Tier::Slow);
		assert_eq!(keys(&l), vec![1, 3, 2]);
		assert_eq!(l.len(), 3);
		let s = l.get(1).expect("present");
		assert_eq!((s.size, s.dram_resident, s.tier), (512, 7, Tier::Slow));
	}

	#[test]
	fn migrating_excludes_the_dram_resident_remainder() {
		let mut l = CompactRecencyList::default();
		l.insert_front(1, 1000, 24, Tier::Fast);
		assert_eq!(l.get(1).unwrap().migrating(), 976);
	}

	#[test]
	fn clear_empties_every_structure() {
		let mut l = build(&[1, 2, 3]);
		l.clear();
		assert!(l.is_empty());
		assert_eq!(l.front(), None);
		assert_eq!(l.back(), None);
		assert!(!l.contains(1));
		l.insert_front(1, 100, 0, Tier::Fast);
		assert_eq!(keys(&l), vec![1]);
	}

	/// Churn far past the initial capacity, asserting the two chains stay
	/// consistent and the slab does not grow beyond the live set.
	#[test]
	fn survives_sustained_churn() {
		let mut l = CompactRecencyList::default();
		for i in 0..64u64 {
			l.insert_front(i, 100, 0, Tier::Fast);
		}
		for round in 0..1_000u64 {
			let victim = l.back().expect("non-empty");
			l.remove(victim);
			l.insert_front(1_000 + round, 100, 0, Tier::Fast);
			l.move_front(l.back().expect("non-empty"));
		}
		assert_eq!(l.len(), 64);
		assert_eq!(keys(&l), keys_reverse(&l));
		assert_eq!(l.slots.len(), 64, "slab grew under steady-state churn");
	}
}
