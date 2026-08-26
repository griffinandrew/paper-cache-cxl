/*
 * Copyright (c) Kia Shakiba
 *
 * This source code is licensed under the GNU AGPLv3 license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! Slab-backed set of intrusive queues: the 2Q and S3-FIFO counterpart of
//! [`CompactRecencyList`](super::compact_recency_list).
//!
//! Every stack in those two families keeps one `kwik::HashList` PER QUEUE --
//! each owning its own key-to-node index -- plus a separate `entries` map
//! holding the combined per-key payload. So a 2Q stack carries three indexes
//! for a population that is in exactly one queue at a time.
//!
//! That "exactly one queue at a time" is what makes a single slab possible: one
//! `Vec` of nodes, one index, and a queue tag inside the payload. Moving
//! between queues becomes an unlink and a relink -- a handful of `u32` writes
//! -- instead of a remove from one hash-indexed list and an insert into another.
//!
//! # Layout B, deliberately, and unlike the LRU list
//!
//! The payload lives in the INDEX MAP'S VALUE here, not in the slab slot:
//!
//! ```text
//! index: HashedKey -> (slot: u32, payload: P)
//! slab:  [ key | prev | next ]                    links only
//! ```
//!
//! so a metadata read is one probe with the payload already in the bucket,
//! rather than a probe followed by a dereference into the slab. Measured, that
//! is 59.9 ns against 97.4 ns -- 1.63x -- and it costs 0 to 8 B/object more,
//! because those 8 bytes move from the densely packed slab into hash buckets
//! that are only 50-87.5% occupied.
//!
//! `CompactRecencyList` makes the opposite choice, and both are right.
//! `mark_accessed` here is the hottest per-get operation in the S3-FIFO family
//! AND touches no queue order, so it pays that dereference on every single get
//! for nothing. LRU has no such path: every `update` reorders the list, so it
//! would be buying speed it cannot use and paying bytes for it.

#[cfg(not(feature = "eviction_stacks_pmem"))]
use std::collections::HashMap;
#[cfg(feature = "eviction_stacks_pmem")]
use hashbrown::HashMap;

use crate::{
	worker::policy::policy_stack::HashedKey,
	NoHasher,
};

/// Sentinel for "no slot". `u32::MAX` rather than `Option<u32>` so a slot stays
/// 16 bytes.
pub const NIL: u32 = u32::MAX;

/// Queues a single stack may hold. 2Q uses 2 (a1_in, am) or 3 with a live
/// a1_out, S3-FIFO uses 2, `LruSizedHybridStack` uses 4.
pub const MAX_QUEUES: usize = 4;

/// One node. 16 bytes: links only -- the payload is in the index.
#[derive(Clone, Copy, Debug)]
pub struct QueueSlot {
	pub key: HashedKey,
	pub prev: u32,
	pub next: u32,
}

// Under `eviction_stacks_pmem` the whole structure is allocated through the
// crate-wide `Hybrid` allocator (the far CXL/PMEM node). Not optional:
// `get_hybrid_dram_shared_overhead` drops the eviction-stack term to zero under
// that feature on the premise the stack is not in DRAM.
#[cfg(not(feature = "eviction_stacks_pmem"))]
type SlotVec = Vec<QueueSlot>;
#[cfg(feature = "eviction_stacks_pmem")]
type SlotVec = Vec<QueueSlot, crate::Hybrid>;

#[cfg(not(feature = "eviction_stacks_pmem"))]
type FreeVec = Vec<u32>;
#[cfg(feature = "eviction_stacks_pmem")]
type FreeVec = Vec<u32, crate::Hybrid>;

#[cfg(not(feature = "eviction_stacks_pmem"))]
type SlotIndex<P> = HashMap<HashedKey, (u32, P), NoHasher>;
#[cfg(feature = "eviction_stacks_pmem")]
type SlotIndex<P> = HashMap<HashedKey, (u32, P), NoHasher, crate::Hybrid>;

#[cfg(not(feature = "eviction_stacks_pmem"))]
fn new_collections<P>() -> (SlotVec, SlotIndex<P>, FreeVec) {
	(Vec::new(), HashMap::default(), Vec::new())
}

#[cfg(feature = "eviction_stacks_pmem")]
fn new_collections<P>() -> (SlotVec, SlotIndex<P>, FreeVec) {
	(
		Vec::new_in(crate::Hybrid),
		HashMap::with_hasher_in(NoHasher::default(), crate::Hybrid),
		Vec::new_in(crate::Hybrid),
	)
}

/// `MAX_QUEUES` intrusive doubly-linked queues over one slab, with the per-key
/// payload carried in the index.
///
/// `P` is the stack's own combined entry -- `queue`, `tier`, `dram_resident`,
/// `size`, and `accessed` for the S3-FIFO family. Which queue a key is in is
/// the caller's business, recorded inside `P`; this structure only maintains
/// the orders.
pub struct CompactQueueSet<P: Copy> {
	slots: SlotVec,
	index: SlotIndex<P>,
	free: FreeVec,

	heads: [u32; MAX_QUEUES],
	tails: [u32; MAX_QUEUES],
	lens: [usize; MAX_QUEUES],
}

impl<P: Copy> Default for CompactQueueSet<P> {
	fn default() -> Self {
		let (slots, index, free) = new_collections();
		CompactQueueSet {
			slots,
			index,
			free,
			heads: [NIL; MAX_QUEUES],
			tails: [NIL; MAX_QUEUES],
			lens: [0; MAX_QUEUES],
		}
	}
}

impl<P: Copy> CompactQueueSet<P> {
	/// Pre-sizes the slab and index. Growth is never in place: every `Vec`
	/// doubling copies every entry, which at eval-trace scale is one
	/// multi-hundred-millisecond stall on the policy worker that the client
	/// latency percentiles structurally cannot observe. Reserving costs no
	/// resident memory, since untouched pages are not resident.
	pub fn reserve(&mut self, objects: usize) {
		self.slots.reserve(objects);
		self.index.reserve(objects);
	}

	pub fn len(&self) -> usize {
		self.lens.iter().sum()
	}

	pub fn is_empty(&self) -> bool {
		self.len() == 0
	}

	pub fn queue_len(&self, q: usize) -> usize {
		self.lens[q]
	}

	pub fn contains(&self, key: HashedKey) -> bool {
		self.index.contains_key(&key)
	}

	/// The payload, in ONE probe. This is the path layout B exists for.
	pub fn payload(&self, key: HashedKey) -> Option<P> {
		self.index.get(&key).map(|&(_, p)| p)
	}

	pub fn payload_mut(&mut self, key: HashedKey) -> Option<&mut P> {
		self.index.get_mut(&key).map(|(_, p)| p)
	}

	pub fn front(&self, q: usize) -> Option<HashedKey> {
		let i = self.heads[q];
		(i != NIL).then(|| self.slots[i as usize].key)
	}

	pub fn back(&self, q: usize) -> Option<HashedKey> {
		let i = self.tails[q];
		(i != NIL).then(|| self.slots[i as usize].key)
	}

	/// One step toward the front of whichever queue `key` is in.
	pub fn before(&self, key: HashedKey) -> Option<HashedKey> {
		let &(i, _) = self.index.get(&key)?;
		let p = self.slots[i as usize].prev;
		(p != NIL).then(|| self.slots[p as usize].key)
	}

	/// One step toward the back.
	pub fn after(&self, key: HashedKey) -> Option<HashedKey> {
		let &(i, _) = self.index.get(&key)?;
		let n = self.slots[i as usize].next;
		(n != NIL).then(|| self.slots[n as usize].key)
	}

	fn unlink(&mut self, q: usize, i: u32) {
		let (prev, next) = {
			let s = &self.slots[i as usize];
			(s.prev, s.next)
		};
		match prev {
			NIL => self.heads[q] = next,
			p => self.slots[p as usize].next = next,
		}
		match next {
			NIL => self.tails[q] = prev,
			n => self.slots[n as usize].prev = prev,
		}
		self.lens[q] -= 1;
	}

	fn link_front(&mut self, q: usize, i: u32) {
		let old = self.heads[q];
		{
			let s = &mut self.slots[i as usize];
			s.prev = NIL;
			s.next = old;
		}
		match old {
			NIL => self.tails[q] = i,
			o => self.slots[o as usize].prev = i,
		}
		self.heads[q] = i;
		self.lens[q] += 1;
	}

	fn link_back(&mut self, q: usize, i: u32) {
		let old = self.tails[q];
		{
			let s = &mut self.slots[i as usize];
			s.prev = old;
			s.next = NIL;
		}
		match old {
			NIL => self.heads[q] = i,
			o => self.slots[o as usize].next = i,
		}
		self.tails[q] = i;
		self.lens[q] += 1;
	}

	fn alloc_slot(&mut self, key: HashedKey) -> u32 {
		let slot = QueueSlot { key, prev: NIL, next: NIL };
		match self.free.pop() {
			Some(i) => {
				self.slots[i as usize] = slot;
				i
			},
			None => {
				let i = self.slots.len() as u32;
				assert!(i != NIL, "CompactQueueSet exceeded u32::MAX - 1 slots");
				self.slots.push(slot);
				i
			},
		}
	}

	/// Inserts a NEW key at the back of `q` (FIFO admission).
	pub fn push_back(&mut self, q: usize, key: HashedKey, payload: P) {
		debug_assert!(!self.index.contains_key(&key), "push_back on an existing key");
		let i = self.alloc_slot(key);
		self.index.insert(key, (i, payload));
		self.link_back(q, i);
	}

	/// Inserts a NEW key at the front of `q` (MRU admission).
	pub fn push_front(&mut self, q: usize, key: HashedKey, payload: P) {
		debug_assert!(!self.index.contains_key(&key), "push_front on an existing key");
		let i = self.alloc_slot(key);
		self.index.insert(key, (i, payload));
		self.link_front(q, i);
	}

	/// Moves an existing key to the front of `q`, which it must already be in.
	pub fn move_front(&mut self, q: usize, key: HashedKey) {
		let Some(&(i, _)) = self.index.get(&key) else { return };
		if self.heads[q] == i {
			return;
		}
		self.unlink(q, i);
		self.link_front(q, i);
	}

	/// Moves an existing key from queue `from` to the back of queue `to`. The
	/// caller updates the queue tag inside the payload; this maintains order.
	pub fn move_to_back_of(&mut self, from: usize, to: usize, key: HashedKey) {
		let Some(&(i, _)) = self.index.get(&key) else { return };
		self.unlink(from, i);
		self.link_back(to, i);
	}

	/// As `move_to_back_of`, entering at the front instead.
	pub fn move_to_front_of(&mut self, from: usize, to: usize, key: HashedKey) {
		let Some(&(i, _)) = self.index.get(&key) else { return };
		self.unlink(from, i);
		self.link_front(to, i);
	}

	/// Removes a key from queue `q`, returning its payload.
	pub fn remove(&mut self, q: usize, key: HashedKey) -> Option<P> {
		let (i, payload) = self.index.remove(&key)?;
		self.unlink(q, i);
		self.free.push(i);
		Some(payload)
	}

	/// Removes the front of `q`.
	pub fn pop_front(&mut self, q: usize) -> Option<(HashedKey, P)> {
		let key = self.front(q)?;
		let payload = self.remove(q, key)?;
		Some((key, payload))
	}

	/// Removes the back of `q`.
	pub fn pop_back(&mut self, q: usize) -> Option<(HashedKey, P)> {
		let key = self.back(q)?;
		let payload = self.remove(q, key)?;
		Some((key, payload))
	}

	pub fn clear(&mut self) {
		self.slots.clear();
		self.index.clear();
		self.free.clear();
		self.heads = [NIL; MAX_QUEUES];
		self.tails = [NIL; MAX_QUEUES];
		self.lens = [0; MAX_QUEUES];
	}
}

#[cfg(test)]
mod tests {
	use super::*;

	/// Stand-in for a real stack payload: 2Q and S3-FIFO entries are both
	/// 8 bytes (queue, tier, dram_resident, size, plus accessed for S3-FIFO).
	#[derive(Clone, Copy, Debug, PartialEq, Eq)]
	struct P {
		queue: u8,
		size: u32,
		accessed: bool,
	}

	fn p(queue: u8) -> P {
		P { queue, size: 1024, accessed: false }
	}

	fn keys(set: &CompactQueueSet<P>, q: usize) -> Vec<HashedKey> {
		let mut out = Vec::new();
		let mut i = set.heads[q];
		while i != NIL {
			out.push(set.slots[i as usize].key);
			i = set.slots[i as usize].next;
		}
		out
	}

	fn keys_reverse(set: &CompactQueueSet<P>, q: usize) -> Vec<HashedKey> {
		let mut out = Vec::new();
		let mut i = set.tails[q];
		while i != NIL {
			out.push(set.slots[i as usize].key);
			i = set.slots[i as usize].prev;
		}
		out.reverse();
		out
	}

	/// Links only: the payload is in the index, so the slot must not grow.
	#[test]
	fn slot_is_sixteen_bytes() {
		assert_eq!(core::mem::size_of::<QueueSlot>(), 16);
	}

	#[test]
	fn push_back_is_fifo_and_push_front_is_lru() {
		let mut s: CompactQueueSet<P> = Default::default();
		for k in 1..=3 {
			s.push_back(0, k, p(0));
		}
		assert_eq!(keys(&s, 0), vec![1, 2, 3]);
		let mut t: CompactQueueSet<P> = Default::default();
		for k in 1..=3 {
			t.push_front(0, k, p(0));
		}
		assert_eq!(keys(&t, 0), vec![3, 2, 1]);
	}

	/// Every other test walks forward; this is the one that catches a `prev`
	/// chain disagreeing with the `next` chain.
	#[test]
	fn forward_and_backward_orders_agree_after_churn() {
		let mut s: CompactQueueSet<P> = Default::default();
		for k in 1..=6 {
			s.push_back(0, k, p(0));
		}
		s.move_front(0, 4);
		s.move_to_back_of(0, 1, 2);
		s.remove(0, 5);
		s.push_back(0, 9, p(0));
		for q in 0..2 {
			assert_eq!(keys(&s, q), keys_reverse(&s, q), "queue {q} chains disagree");
		}
	}

	/// The point of the structure: queues are independent orders over ONE slab.
	#[test]
	fn queues_are_independent() {
		let mut s: CompactQueueSet<P> = Default::default();
		s.push_back(0, 1, p(0));
		s.push_back(1, 2, p(1));
		s.push_back(0, 3, p(0));
		s.push_back(2, 4, p(2));
		assert_eq!(keys(&s, 0), vec![1, 3]);
		assert_eq!(keys(&s, 1), vec![2]);
		assert_eq!(keys(&s, 2), vec![4]);
		assert_eq!(s.len(), 4);
		assert_eq!((s.queue_len(0), s.queue_len(1), s.queue_len(2)), (2, 1, 1));
	}

	/// A queue move must not reallocate, duplicate, or lose the payload -- it
	/// is an unlink and a relink of the same slot.
	#[test]
	fn moving_between_queues_preserves_slot_and_payload() {
		let mut s: CompactQueueSet<P> = Default::default();
		s.push_back(0, 1, p(0));
		s.push_back(0, 2, p(0));
		let slot_before = s.index[&1].0;
		let slab_before = s.slots.len();

		s.move_to_back_of(0, 1, 1);
		if let Some(pl) = s.payload_mut(1) {
			pl.queue = 1;
		}

		assert_eq!(s.index[&1].0, slot_before, "slot changed on a queue move");
		assert_eq!(s.slots.len(), slab_before, "slab grew on a queue move");
		assert_eq!(keys(&s, 0), vec![2]);
		assert_eq!(keys(&s, 1), vec![1]);
		assert_eq!(s.payload(1).unwrap().queue, 1);
		assert_eq!(s.len(), 2);
	}

	#[test]
	fn move_front_within_a_queue() {
		let mut s: CompactQueueSet<P> = Default::default();
		for k in 1..=3 {
			s.push_back(0, k, p(0));
		}
		s.move_front(0, 3);
		assert_eq!(keys(&s, 0), vec![3, 1, 2]);
		s.move_front(0, 3);
		assert_eq!(keys(&s, 0), vec![3, 1, 2], "move_front on the head must be a no-op");
	}

	#[test]
	fn before_and_after_step_in_the_expected_directions() {
		let mut s: CompactQueueSet<P> = Default::default();
		for k in 1..=3 {
			s.push_back(0, k, p(0));
		}
		assert_eq!(s.before(2), Some(1));
		assert_eq!(s.after(2), Some(3));
		assert_eq!(s.before(1), None);
		assert_eq!(s.after(3), None);
	}

	#[test]
	fn pop_front_and_back_return_payloads_and_maintain_ends() {
		let mut s: CompactQueueSet<P> = Default::default();
		for k in 1..=3 {
			s.push_back(0, k, p(0));
		}
		assert_eq!(s.pop_front(0).map(|(k, _)| k), Some(1));
		assert_eq!(s.pop_back(0).map(|(k, _)| k), Some(3));
		assert_eq!(keys(&s, 0), vec![2]);
		assert_eq!(s.pop_front(0).map(|(k, _)| k), Some(2));
		assert!(s.pop_front(0).is_none());
		assert!(s.is_empty());
	}

	/// Freed slots must be recycled or the slab grows without bound under the
	/// insert/evict churn a cache runs at steady state.
	#[test]
	fn freed_slots_are_recycled() {
		let mut s: CompactQueueSet<P> = Default::default();
		for k in 1..=3 {
			s.push_back(0, k, p(0));
		}
		let before = s.slots.len();
		s.remove(0, 2);
		s.push_back(0, 4, p(0));
		assert_eq!(s.slots.len(), before, "slab grew instead of reusing a free slot");
		assert_eq!(keys(&s, 0), vec![1, 3, 4]);
	}

	#[test]
	fn removing_head_and_tail_maintains_both_ends() {
		let mut s: CompactQueueSet<P> = Default::default();
		for k in 1..=3 {
			s.push_back(0, k, p(0));
		}
		s.remove(0, 1);
		assert_eq!(s.front(0), Some(2));
		s.remove(0, 3);
		assert_eq!(s.back(0), Some(2));
		s.remove(0, 2);
		assert_eq!(s.front(0), None);
		assert_eq!(s.back(0), None);
		assert_eq!(s.queue_len(0), 0);
	}

	#[test]
	fn clear_empties_every_queue() {
		let mut s: CompactQueueSet<P> = Default::default();
		s.push_back(0, 1, p(0));
		s.push_back(1, 2, p(1));
		s.clear();
		assert!(s.is_empty());
		assert_eq!(s.front(0), None);
		assert_eq!(s.front(1), None);
		assert!(!s.contains(1));
	}

	/// Sustained churn across queues, asserting the chains stay consistent and
	/// the slab stays bounded by the live set.
	#[test]
	fn survives_sustained_cross_queue_churn() {
		let mut s: CompactQueueSet<P> = Default::default();
		for k in 0..64u64 {
			s.push_back(0, k, p(0));
		}
		for round in 0..1_000u64 {
			if let Some(k) = s.front(0) {
				s.move_to_back_of(0, 1, k);
			}
			if let Some(k) = s.front(1) {
				s.remove(1, k);
			}
			s.push_back(0, 1_000 + round, p(0));
		}
		for q in 0..2 {
			assert_eq!(keys(&s, q), keys_reverse(&s, q), "queue {q} chains disagree");
		}
		assert_eq!(s.len(), s.queue_len(0) + s.queue_len(1));
		assert!(s.slots.len() <= 66, "slab grew to {} under churn", s.slots.len());
	}
}
