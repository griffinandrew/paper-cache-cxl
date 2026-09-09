/*
 * Copyright (c) Kia Shakiba
 *
 * This source code is licensed under the GNU AGPLv3 license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! [`CompactQueueSet`] with the key stored ONCE instead of twice.
//!
//! # The redundancy
//!
//! `CompactQueueSet` holds every tracked key in two places at the same time:
//!
//! ```text
//! slots: Vec<QueueSlot>                          QueueSlot { key, prev, next }
//! index: HashMap<HashedKey, (u32, P), NoHasher>  key -> (slot, payload)
//! ```
//!
//! The index carries the key so it can hash it and compare it on a probe; the
//! slot carries the key so eviction can name the victim it just unlinked. Same
//! eight bytes, twice, inside ONE structure. Every earlier count of duplicated
//! keying in this crate looked BETWEEN structures -- the object map against the
//! eviction stack -- and so never counted this one.
//!
//! Measured (jemalloc `stats.allocated`, one process per point, powers of two):
//!
//! ```text
//! object map row (DashMap)   80.00 B/object
//! CompactQueueSet            72.00 B/object   <- what this attacks
//! ```
//!
//! 56 of that 72 is the index: the slot is 16 bytes, so everything above that
//! is index. Its entry is `(u64 key, (u32 slot, P payload))` = 20 bytes padded
//! to 24, and hashbrown at a power-of-two population sits near 50% fill.
//!
//! # What changes
//!
//! **The payload moves out of the index and into the slot.** Every hybrid
//! payload in this crate is exactly 8 bytes and statically asserted so in all
//! 22 policy stacks, which makes [`ArenaSlot`] a uniform 24 bytes for hybrid
//! policies and 16 for the ZST-payload flat ones.
//!
//! **The index becomes a bare open-addressed table of `u32` slot indices, with
//! no keys in it at all.** A lookup mixes the key, indexes the table, reads a
//! `u32`, and verifies `slots[i].key == key`. The verification key is the one
//! the slot already had to carry, which is exactly why the index does not need
//! a copy of its own. At a power-of-two population with 2x slack that is
//! `4 B * 2 = 8 B/object` against the 56 above.
//!
//! ```text
//! CompactQueueSet   slot 16 (key,prev,next)   + index 56    = 72
//! ArenaQueueSet        slot 24 (key,prev,next,P) + index    8   = 32
//! ```
//!
//! MEASURED, and not the arithmetic above: jemalloc `stats.allocated`, one
//! process per point, sampled at powers of two, 8-byte payload.
//!
//! ```text
//! n        CompactQueueSet   ArenaQueueSet
//! 2^20          72.58           32.50
//! 2^21          72.29           32.25
//! 2^22          72.14           32.12
//! 2^23          72.07           32.06
//! marginal      72.00           32.00
//! ```
//!
//! The same 40 B/object shows up through `init_policy_stack` on all three
//! wired stacks -- `lru-compact-hybrid`, `2q-compact-hybrid-0.25` and
//! `s3-fifo-faithful-compact-hybrid-0.1` all move 72.00 -> 32.00.
//!
//! 32 is the POWER-OF-TWO figure, which is the phase this crate's measurement
//! methodology fixes on and the phase `CompactQueueSet`'s 72 is quoted at. It
//! is a FLOOR, and a lower one than the 72 it is compared against.
//!
//! An earlier revision of this paragraph put v4's range at "32-40 across the
//! cycle", counting the index's oscillation and forgetting the slab's. BOTH
//! structures double, so both terms swing: at `n = 2^k` the slab is exactly
//! full (24 B/object) and the index exactly half full (8), and just after a
//! doubling the slab is half empty (48) and the index quarter full (16). The
//! real range is 32-64, and 32 is its bottom.
//!
//! MEASURED at the population a real cluster26 run reaches, by
//! `tests::measure_four_structures_at_one_population`, on the same objects and
//! in the same process as `CompactQueueSet` and `SlotArena`:
//!
//! ```text
//!   n                     4,194,304   4,816,172   4,888,044   8,388,608
//!   phase                     2^22       real        real        2^23
//!   slab capacity          4,194,304   8,388,608   8,388,608   8,388,608
//!   index load                0.500      0.287       0.291       0.500
//!   ---
//!   CompactQueueSet            72.00      76.64       75.51       72.00
//!   ArenaQueueSet                 32.00      55.74       54.92       32.00
//!   SlotArena                  24.00      41.80       41.19       24.00
//! ```
//!
//! So the 40 B/object v4 takes off `CompactQueueSet` at a power of two is
//! 20.90 at the population a run actually reaches -- close to half. That is
//! not a harness artifact: the shipped cluster26 runs at 5 GiB, holding the
//! same 4,816,172 objects under both builds, report jemalloc `stats.allocated`
//! of 7,123,219,256 for v4 against 7,223,887,456 for the split baseline, and
//! that difference is 20.90 B/object.
//!
//! v4 loses the most of the three because it is the only design whose BOTH
//! terms are slack at once: 24 x 1.742 = 41.80 of slab and 4 x 16,777,216 /
//! 4,816,172 = 13.94 of index. `SlotArena` is slab only, and
//! `CompactQueueSet`'s 16-byte slot carries less slack per object precisely
//! because its slot is smaller -- the thing v4 made bigger.
//!
//! Everything else is deliberately preserved: `MAX_QUEUES` orders over one
//! slab, the `heads`/`tails`/`lens` arrays, the free list, and a PUBLIC API
//! that is still addressed by key and method-for-method identical to
//! `CompactQueueSet`, so a stack switches by changing a type name.
//!
//! # What it costs
//!
//! `CompactQueueSet`'s layout B exists so a metadata read is ONE probe with the
//! payload already in the hash bucket. v4 gives that up: the payload read is a
//! probe into the index followed by a dereference into the slab, which is
//! layout A's shape. What is not the same as layout A is the size of the thing
//! being probed -- the index is 8 B/object here against 56 there, so the
//! first touch is into a table seven times likelier to be cache-resident. Which way that nets out is a measurement, not an argument;
//! it is not made here.
//!
//! A one-byte payload is the one shape that loses slab bytes: `key`, `prev` and
//! `next` already fill 16 bytes at 8-byte alignment, so a `bool` payload rounds
//! the slot to 24 rather than staying at 16. That costs 8 B/object in the slab
//! and still saves far more than it costs in the index.
//!
//! # Where the index went
//!
//! Into [`arena_index`], and the two sections that used to close this doc went
//! with it -- why a removal is a backward shift rather than a tombstone, and
//! what the Fibonacci mix does and does not buy on a real trace.
//!
//! It moved because there are now TWO structures over this node. This one holds
//! `MAX_QUEUES` queues in a fixed `[u32; 4]` of heads and tails.
//! [`ArenaFrequencyChain`] holds one ordered bucket per DISTINCT FREQUENCY,
//! because LFU eviction has to find the minimum and a four-queue tag cannot
//! express that. The orders differ; the index does not, and one copy of
//! backward-shift deletion is enough.
//!
//! [`arena_index`]: super::arena_index
//! [`ArenaFrequencyChain`]: super::arena_frequency_chain::ArenaFrequencyChain

use crate::{ObjectSize, worker::policy::policy_stack::{HashedKey, Tier}};

pub use super::arena_index::{ArenaSlot, NIL};
use super::arena_index::{KeylessIndex, SlotVec, U32Vec, new_slot_vec, new_u32_vec};

// ---------------------------------------------------------------------------
// the one node payload
// ---------------------------------------------------------------------------

/// What every policy records about a tracked key, in one shape.
///
/// Policies differ in which fields they read, not in which they have: a pure
/// recency stack never touches `freq` or `ts`, an LFU-style one never touches
/// `queue`, and a flat stack never touches `tier`. Carrying the union costs
/// eight bytes over the per-policy payloads it replaces and removes the reason
/// each policy needed its own queue set.
///
/// Sixteen bytes, which with the slot's key and links is a 32-byte node:
///
/// ```text
///   key   8  |  prev 4  |  next 4  |  payload 16   =  32
/// ```
///
/// Against `CompactQueueSet`'s 72 B/object -- a 16-byte slot plus a 56-byte
/// index -- the saving is the index, which this structure does not have: the
/// bucket array holds bare slot numbers and a probe is verified against the
/// slot's own key.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct NodePayload {
	/// Bytes of the whole object: its header, its key and its value.
	pub size: ObjectSize,

	/// Frequency, for the policies that rank or admit by it.
	///
	/// Also carries the S3-FIFO family's REFERENCE BIT, as `freq != 0`: a
	/// reference bit is a one-bit frequency counter, and no stack in the tree
	/// wants both. `u32` because `CompactFrequencyChain` needs that width to
	/// bucket by exact frequency; the S3-FIFO variants saturate at 3 and
	/// simply do not use the range.
	pub freq: u32,

	/// A COARSE AGING EPOCH, and deliberately not a recency order.
	///
	/// Recency is `prev`/`next`; this is for policies that age or decay on a
	/// window. `u32` is safe for that because wrapping is harmless below 2^31,
	/// the same argument `GhostFilter::inserted_at` rests on. It must never be
	/// used as an ordering key: `MergedStore::Slot::last_access` was a `u32`
	/// wrapping difference and had to be widened to `u64` for a silent
	/// corruption reachable on a 306M-record replay.
	///
	/// No stack reads it today. It is here because the node is meant to be the
	/// one shape every policy shares, and an aging policy that had to add a
	/// field would defeat that.
	pub ts: u32,

	/// Which queue this key is in, for the multi-queue policies. Zero for the
	/// single-queue ones, which never read it.
	pub queue: u8,

	/// Which tier the POLICY believes this object is in. Distinct from the
	/// tier bit the value carries, which says which allocator frees its bytes:
	/// this one is a decision, that one is a physical fact.
	///
	/// `Option` because the 2Q and S3-FIFO families leave it `None` while a key
	/// sits in a queue that is entirely slow-tier and so has no tier of its
	/// own to record. The niche makes that free.
	pub tier: Option<Tier>,

	/// Where the bytes PHYSICALLY are, when that can differ from `tier`.
	///
	/// Only the lazy-copy design uses it: it promotes logically and defers the
	/// byte copy, so for a window the two disagree. Everyone else leaves it
	/// equal to `tier` and never reads it.
	pub phys: Option<Tier>,

	/// The part of `size` that stays in DRAM whichever tier the object is in.
	pub dram_resident: u8,
}

/// Pinned. The node is the per-object cost of every eviction stack, so growth
/// here is paid on every tracked key in every policy.
const _: () = assert!(
	core::mem::size_of::<NodePayload>() == 16,
	"NodePayload grew past 16 bytes -- the node is 32 only while this is 16",
);

const _: () = assert!(
	core::mem::size_of::<ArenaSlot<NodePayload>>() == 32,
	"the arena node grew past 32 bytes -- re-measure before changing this",
);

impl NodePayload {
	/// The bytes that would move if this object migrated tier.
	#[inline]
	pub fn migrating(&self) -> crate::CacheSize {
		(self.size as crate::CacheSize)
			.saturating_sub(self.dram_resident as crate::CacheSize)
	}
}


/// Queues a single stack may hold, unchanged from `CompactQueueSet`: 2Q uses 2
/// (a1_in, am) or 3 with a live a1_out, S3-FIFO uses 2 or 3, and
/// `LruSizedHybridStack` uses 4.
pub const MAX_QUEUES: usize = 4;

/// `MAX_QUEUES` intrusive doubly-linked queues over one slab, indexed by a
/// keyless open-addressed table of slot numbers.
///
/// A drop-in replacement for `CompactQueueSet`: same methods, same signatures,
/// same semantics, same key-addressed surface. `P` is the stack's own combined
/// entry -- `queue`, `tier`, `dram_resident`, `size`, `freq` and so on. Which
/// queue a key is in remains the caller's business, recorded inside `P`; this
/// structure only maintains the orders.
pub struct ArenaQueueSet<P: Copy> {
	slots: SlotVec<P>,
	free: U32Vec,

	/// Open-addressed, linear-probed table of slot numbers. `NIL` is empty.
	/// Holds NO keys: a probe verifies against `slots[i].key`. Shared with
	/// [`ArenaFrequencyChain`](super::arena_frequency_chain::ArenaFrequencyChain),
	/// which needs the same index under different orders.
	index: KeylessIndex,

	heads: [u32; MAX_QUEUES],
	tails: [u32; MAX_QUEUES],
	lens: [usize; MAX_QUEUES],
}

impl<P: Copy> Default for ArenaQueueSet<P> {
	fn default() -> Self {
		ArenaQueueSet {
			slots: new_slot_vec(),
			free: new_u32_vec(),
			index: KeylessIndex::default(),
			heads: [NIL; MAX_QUEUES],
			tails: [NIL; MAX_QUEUES],
			lens: [0; MAX_QUEUES],
		}
	}
}

// ---------------------------------------------------------------------------
// The keyless index, which lives in `arena_index` because the frequency chain
// needs the same one. These three forward to it; nothing else here knows it
// exists.
// ---------------------------------------------------------------------------

impl<P: Copy> ArenaQueueSet<P> {
	/// Slot holding `key`, or `NIL`.
	#[inline]
	fn index_get(&self, key: HashedKey) -> u32 {
		self.index.get(&self.slots, key)
	}

	/// Places an ALREADY-ALLOCATED slot in the index. `slots[slot].key` must
	/// already be written, since that is the only copy of the key there is.
	fn index_insert(&mut self, slot: u32) {
		self.index.insert(&self.slots, slot);
	}

	/// Removes `key` from the index, returning its slot.
	fn index_remove(&mut self, key: HashedKey) -> Option<u32> {
		self.index.remove(&self.slots, key)
	}
}

// ---------------------------------------------------------------------------
// The public surface, method-for-method identical to `CompactQueueSet`.
// ---------------------------------------------------------------------------

impl<P: Copy> ArenaQueueSet<P> {
	/// Slab slots currently allocated. Exposed so a test can assert that
	/// construction does NOT allocate from the cache budget: these stacks grow
	/// dynamically, and an eager reservation sized from capacity reserves
	/// far more than the eval workload can hold while still not preventing
	/// doubling on the real traces.
	pub fn slab_capacity(&self) -> usize {
		self.slots.capacity()
	}

	/// Buckets in the index. Exposed for the probe-length measurement, which
	/// has to know the table size to reason about a run.
	pub fn index_capacity(&self) -> usize {
		self.index.capacity()
	}

	/// Pre-sizes the slab and the index. Growth is never in place: every `Vec`
	/// doubling copies every entry, which at eval-trace scale is one
	/// multi-hundred-millisecond stall on the policy worker that the client
	/// latency percentiles structurally cannot observe.
	pub fn reserve(&mut self, objects: usize) {
		self.slots.reserve(objects);
		self.index.reserve(&self.slots, objects);
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
		self.index_get(key) != NIL
	}

	/// The payload: one index probe, then one slab dereference.
	pub fn payload(&self, key: HashedKey) -> Option<P> {
		match self.index_get(key) {
			NIL => None,
			i => Some(self.slots[i as usize].payload),
		}
	}

	pub fn payload_mut(&mut self, key: HashedKey) -> Option<&mut P> {
		match self.index_get(key) {
			NIL => None,
			i => Some(&mut self.slots[i as usize].payload),
		}
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
		let i = self.index_get(key);

		if i == NIL {
			return None;
		}

		let p = self.slots[i as usize].prev;
		(p != NIL).then(|| self.slots[p as usize].key)
	}

	/// One step toward the back.
	pub fn after(&self, key: HashedKey) -> Option<HashedKey> {
		let i = self.index_get(key);

		if i == NIL {
			return None;
		}

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

	fn alloc_slot(&mut self, key: HashedKey, payload: P) -> u32 {
		let slot = ArenaSlot { key, prev: NIL, next: NIL, payload };

		match self.free.pop() {
			Some(i) => {
				self.slots[i as usize] = slot;
				i
			},
			None => {
				let i = self.slots.len() as u32;
				assert!(i != NIL, "ArenaQueueSet exceeded u32::MAX - 1 slots");
				self.slots.push(slot);
				i
			},
		}
	}

	/// Inserts a NEW key at the back of `q` (FIFO admission).
	pub fn push_back(&mut self, q: usize, key: HashedKey, payload: P) {
		debug_assert!(!self.contains(key), "push_back on an existing key");
		let i = self.alloc_slot(key, payload);
		self.index_insert(i);
		self.link_back(q, i);
	}

	/// Inserts a NEW key at the front of `q` (MRU admission).
	pub fn push_front(&mut self, q: usize, key: HashedKey, payload: P) {
		debug_assert!(!self.contains(key), "push_front on an existing key");
		let i = self.alloc_slot(key, payload);
		self.index_insert(i);
		self.link_front(q, i);
	}

	/// Moves an existing key to the front of `q`, which it must already be in.
	pub fn move_front(&mut self, q: usize, key: HashedKey) {
		let i = self.index_get(key);

		if i == NIL || self.heads[q] == i {
			return;
		}

		self.unlink(q, i);
		self.link_front(q, i);
	}

	/// Moves an existing key from queue `from` to the back of queue `to`. The
	/// caller updates the queue tag inside the payload; this maintains order.
	pub fn move_to_back_of(&mut self, from: usize, to: usize, key: HashedKey) {
		let i = self.index_get(key);

		if i == NIL {
			return;
		}

		self.unlink(from, i);
		self.link_back(to, i);
	}

	/// As `move_to_back_of`, entering at the front instead.
	pub fn move_to_front_of(&mut self, from: usize, to: usize, key: HashedKey) {
		let i = self.index_get(key);

		if i == NIL {
			return;
		}

		self.unlink(from, i);
		self.link_front(to, i);
	}

	/// Removes a key from queue `q`, returning its payload.
	pub fn remove(&mut self, q: usize, key: HashedKey) -> Option<P> {
		let i = self.index_remove(key)?;
		self.unlink(q, i);
		let payload = self.slots[i as usize].payload;
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
		self.free.clear();

		// The table keeps its capacity, exactly as `HashMap::clear` does, so a
		// cleared-and-refilled stack does not pay the doubling ladder twice.
		self.index.clear();

		self.heads = [NIL; MAX_QUEUES];
		self.tails = [NIL; MAX_QUEUES];
		self.lens = [0; MAX_QUEUES];
	}
}

#[cfg(test)]
mod tests {
	use super::*;
	use crate::worker::policy::policy_stack::arena_index::GOLDEN;
	use crate::worker::policy::policy_stack::compact_queue_set::CompactQueueSet;

	/// Stand-in for a real stack payload: LRU, 2Q and S3-FIFO entries are all
	/// exactly 8 bytes.
	#[derive(Clone, Copy, Debug, PartialEq, Eq)]
	struct P {
		queue: u8,
		size: u32,
		accessed: bool,
	}

	fn p(queue: u8) -> P {
		P { queue, size: 1024, accessed: false }
	}

	fn keys(set: &ArenaQueueSet<P>, q: usize) -> Vec<HashedKey> {
		let mut out = Vec::new();
		let mut i = set.heads[q];
		while i != NIL {
			out.push(set.slots[i as usize].key);
			i = set.slots[i as usize].next;
		}
		out
	}

	fn keys_reverse(set: &ArenaQueueSet<P>, q: usize) -> Vec<HashedKey> {
		let mut out = Vec::new();
		let mut i = set.tails[q];
		while i != NIL {
			out.push(set.slots[i as usize].key);
			i = set.slots[i as usize].prev;
		}
		out.reverse();
		out
	}

	// -------------------------------------------------------------------
	// Layout
	// -------------------------------------------------------------------

	/// The whole premise: the payload rides in the slot, so a slot is the
	/// links plus the key plus eight bytes and nothing else.
	#[test]
	fn an_eight_byte_payload_makes_a_twenty_four_byte_slot() {
		assert_eq!(core::mem::size_of::<ArenaSlot<P>>(), 24);
	}

	/// A ZST payload must not grow the slot: the flat stacks index
	/// `CompactQueueSet<()>` and moving them onto v4 must not cost them slab
	/// bytes to save them index bytes.
	#[test]
	fn a_zero_sized_payload_leaves_the_slot_at_sixteen_bytes() {
		assert_eq!(core::mem::size_of::<ArenaSlot<()>>(), 16);
	}

	/// The one shape that loses slab bytes, asserted so the module doc's claim
	/// about it is checked rather than believed. `key`, `prev` and `next`
	/// already fill 16 bytes at 8-byte alignment, so a one-byte payload -- what
	/// the CLOCK and SIEVE flat stacks carry -- rounds the slot to 24 where
	/// `CompactQueueSet` kept it at 16. It buys that 8 back many times over in
	/// the index, but it is a cost and not a saving and should read as one.
	#[test]
	fn a_one_byte_payload_still_rounds_the_slot_to_twenty_four() {
		assert_eq!(core::mem::size_of::<ArenaSlot<bool>>(), 24);
	}

	/// The index holds slot numbers and nothing else, so its whole cost is four
	/// bytes a bucket at half load. At a power-of-two population that is
	/// exactly the 8 B/object the design is built around, and a wider bucket or
	/// a looser load factor would silently move it.
	#[test]
	fn the_index_costs_eight_bytes_per_object_at_a_power_of_two_population() {
		for exponent in 8..14u32 {
			let n = 1u64 << exponent;
			let mut s: ArenaQueueSet<P> = Default::default();
			for k in 0..n {
				s.push_back(0, k.wrapping_mul(GOLDEN), p(0));
			}
			let bytes = s.index_capacity() * core::mem::size_of::<u32>();
			assert_eq!(
				bytes as u64 / n,
				8,
				"{n} keys sat in a {}-bucket table",
				s.index_capacity(),
			);
		}
	}

	// -------------------------------------------------------------------
	// Behaviour, mirroring `compact_queue_set`'s own suite
	// -------------------------------------------------------------------

	#[test]
	fn push_back_is_fifo_and_push_front_is_lru() {
		let mut s: ArenaQueueSet<P> = Default::default();
		for k in 1..=3 {
			s.push_back(0, k, p(0));
		}
		assert_eq!(keys(&s, 0), vec![1, 2, 3]);
		let mut t: ArenaQueueSet<P> = Default::default();
		for k in 1..=3 {
			t.push_front(0, k, p(0));
		}
		assert_eq!(keys(&t, 0), vec![3, 2, 1]);
	}

	/// Every other test walks forward; this is the one that catches a `prev`
	/// chain disagreeing with the `next` chain.
	#[test]
	fn forward_and_backward_orders_agree_after_churn() {
		let mut s: ArenaQueueSet<P> = Default::default();
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
		let mut s: ArenaQueueSet<P> = Default::default();
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
		let mut s: ArenaQueueSet<P> = Default::default();
		s.push_back(0, 1, p(0));
		s.push_back(0, 2, p(0));
		let slot_before = s.index_get(1);
		let slab_before = s.slots.len();

		s.move_to_back_of(0, 1, 1);
		if let Some(pl) = s.payload_mut(1) {
			pl.queue = 1;
		}

		assert_eq!(s.index_get(1), slot_before, "slot changed on a queue move");
		assert_eq!(s.slots.len(), slab_before, "slab grew on a queue move");
		assert_eq!(keys(&s, 0), vec![2]);
		assert_eq!(keys(&s, 1), vec![1]);
		assert_eq!(s.payload(1).unwrap().queue, 1);
		assert_eq!(s.len(), 2);
	}

	#[test]
	fn move_front_within_a_queue() {
		let mut s: ArenaQueueSet<P> = Default::default();
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
		let mut s: ArenaQueueSet<P> = Default::default();
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
		let mut s: ArenaQueueSet<P> = Default::default();
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
		let mut s: ArenaQueueSet<P> = Default::default();
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
		let mut s: ArenaQueueSet<P> = Default::default();
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
		let mut s: ArenaQueueSet<P> = Default::default();
		s.push_back(0, 1, p(0));
		s.push_back(1, 2, p(1));
		s.clear();
		assert!(s.is_empty());
		assert_eq!(s.front(0), None);
		assert_eq!(s.front(1), None);
		assert!(!s.contains(1));
	}

	/// A cleared set must be usable again: `clear` leaves the table sized but
	/// blank, and a stale slot number left in a bucket would resurrect a key
	/// that is no longer there.
	#[test]
	fn a_cleared_set_can_be_refilled_without_resurrecting_anything() {
		let mut s: ArenaQueueSet<P> = Default::default();
		for k in 0..500u64 {
			s.push_back(0, k, p(0));
		}
		s.clear();
		for k in 500..1_000u64 {
			s.push_back(0, k, p(1));
		}
		for k in 0..500u64 {
			assert!(!s.contains(k), "key {k} survived a clear");
		}
		for k in 500..1_000u64 {
			assert_eq!(s.payload(k).map(|pl| pl.queue), Some(1), "key {k} lost after refill");
		}
		assert_eq!(s.len(), 500);
	}

	/// Sustained churn across queues, asserting the chains stay consistent and
	/// the slab stays bounded by the live set.
	#[test]
	fn survives_sustained_cross_queue_churn() {
		let mut s: ArenaQueueSet<P> = Default::default();
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

	// -------------------------------------------------------------------
	// Deletion, which is where open addressing goes wrong
	// -------------------------------------------------------------------

	/// Multiplicative inverse of `GOLDEN` mod 2^64, by Newton's method. Lets a
	/// test name a bucket and get keys that land in it, which is the only way
	/// to build a probe run on purpose rather than by luck.
	fn golden_inverse() -> u64 {
		let mut x = GOLDEN;
		for _ in 0..6 {
			x = x.wrapping_mul(2u64.wrapping_sub(GOLDEN.wrapping_mul(x)));
		}
		debug_assert_eq!(GOLDEN.wrapping_mul(x), 1);
		x
	}

	/// `n` distinct keys whose home bucket is all the same `bucket`, in a table
	/// of `capacity` buckets.
	fn colliding_keys(capacity: usize, bucket: usize, n: u64) -> Vec<HashedKey> {
		let shift = 64 - capacity.trailing_zeros();
		let inverse = golden_inverse();
		(0..n)
			.map(|i| inverse.wrapping_mul(((bucket as u64) << shift) | i))
			.collect()
	}

	/// The property backward-shift deletion exists to preserve. Ten keys share
	/// one home bucket, so they occupy one ten-long probe run; removing one
	/// from the middle must leave every other one findable, which blanking the
	/// bucket would not.
	#[test]
	fn removing_from_the_middle_of_a_probe_run_leaves_the_rest_findable() {
		let mut s: ArenaQueueSet<P> = Default::default();
		s.reserve(1_000);
		let capacity = s.index_capacity();
		assert_eq!(capacity, 2_048, "reserve must fix the table size for this test");

		let run = colliding_keys(capacity, 300, 10);
		for (i, &k) in run.iter().enumerate() {
			s.push_back(0, k, p(i as u8));
		}
		for &k in &run {
			assert_eq!(s.index.home(k), 300, "test key did not land in the intended bucket");
		}

		s.remove(0, run[4]);

		assert!(!s.contains(run[4]), "the removed key is still findable");
		for (i, &k) in run.iter().enumerate() {
			if i == 4 {
				continue;
			}
			assert_eq!(
				s.payload(k).map(|pl| pl.queue),
				Some(i as u8),
				"key {i} of the run was stranded by the removal",
			);
		}
	}

	/// The same property under the harder pattern: an entire probe run removed
	/// one at a time, in an order chosen to move the hole backward and forward.
	#[test]
	fn a_probe_run_can_be_dismantled_in_any_order() {
		for order in [
			vec![0usize, 1, 2, 3, 4, 5, 6, 7],
			vec![7, 6, 5, 4, 3, 2, 1, 0],
			vec![3, 0, 6, 1, 7, 2, 5, 4],
			vec![4, 4, 4, 4, 4, 4, 4, 4],
		] {
			let mut s: ArenaQueueSet<P> = Default::default();
			s.reserve(1_000);
			let run = colliding_keys(s.index_capacity(), 77, 8);
			for &k in &run {
				s.push_back(0, k, p(0));
			}

			let mut alive: Vec<HashedKey> = run.clone();

			for &pick in &order {
				if alive.is_empty() {
					break;
				}
				let victim = alive.remove(pick.min(alive.len() - 1));
				s.remove(0, victim);

				assert!(!s.contains(victim), "removed key still findable");
				for &k in &alive {
					assert!(s.contains(k), "a survivor was stranded by a removal");
				}
			}
		}
	}

	/// The run crossing the end of the table. Backward shift is stated
	/// cyclically and a non-cyclic comparison passes every test whose run does
	/// not wrap -- so one has to.
	#[test]
	fn a_probe_run_that_wraps_the_table_deletes_correctly() {
		let mut s: ArenaQueueSet<P> = Default::default();
		s.reserve(1_000);
		let capacity = s.index_capacity();

		// Home the run on the LAST bucket, so it wraps to 0 immediately.
		let run = colliding_keys(capacity, capacity - 1, 12);
		for (i, &k) in run.iter().enumerate() {
			s.push_back(0, k, p(i as u8));
		}

		for cut in [0usize, 5, 11] {
			let mut t: ArenaQueueSet<P> = Default::default();
			t.reserve(1_000);
			for (i, &k) in run.iter().enumerate() {
				t.push_back(0, k, p(i as u8));
			}

			t.remove(0, run[cut]);

			assert!(!t.contains(run[cut]));
			for (i, &k) in run.iter().enumerate() {
				if i == cut {
					continue;
				}
				assert_eq!(
					t.payload(k).map(|pl| pl.queue),
					Some(i as u8),
					"key {i} stranded after cutting {cut} from a wrapping run",
				);
			}
		}
	}

	/// Steady-state cache churn against an all-colliding key set: one removal
	/// per admission, forever, which is the pattern that makes tombstones
	/// degrade and the one backward shift has to survive.
	#[test]
	fn an_all_colliding_key_set_survives_sustained_insert_and_remove_churn() {
		let mut s: ArenaQueueSet<P> = Default::default();
		s.reserve(2_000);
		let pool = colliding_keys(s.index_capacity(), 1_000, 4_000);

		let mut alive: Vec<HashedKey> = Vec::new();
		for &k in pool.iter().take(500) {
			s.push_back(0, k, p(0));
			alive.push(k);
		}

		let mut x: u64 = 0x243F_6A88_85A3_08D3;
		for round in 500..pool.len() {
			x ^= x << 13;
			x ^= x >> 7;
			x ^= x << 17;

			let victim_at = (x % alive.len() as u64) as usize;
			let victim = alive.swap_remove(victim_at);
			s.remove(0, victim);
			assert!(!s.contains(victim), "removed key still findable at round {round}");

			s.push_back(0, pool[round], p(0));
			alive.push(pool[round]);

			// Periodically rather than every round: the run is 500 buckets
			// long, so a full sweep is quadratic and says nothing a sample of
			// sweeps does not.
			if round % 97 == 0 {
				for &k in &alive {
					assert!(s.contains(k), "a live key was stranded at round {round}");
				}
			}
		}

		assert_eq!(s.len(), alive.len());
		for &k in &alive {
			assert!(s.contains(k), "a live key was stranded by the churn");
		}
	}

	/// Growth must not lose anything: the table is rebuilt from the slab's
	/// keys, and a rehash that read the wrong slot would silently drop keys.
	#[test]
	fn every_key_survives_the_tables_growth_ladder() {
		let mut s: ArenaQueueSet<P> = Default::default();
		for k in 0..20_000u64 {
			let key = k.wrapping_mul(0x9E37_79B9_7F4A_7C15);
			s.push_back(0, key, p(0));
			assert_eq!(s.len(), (k + 1) as usize);
		}
		for k in 0..20_000u64 {
			let key = k.wrapping_mul(0x9E37_79B9_7F4A_7C15);
			assert!(s.contains(key), "key {k} lost across the growth ladder");
		}
		assert!(
			s.index_capacity() >= 40_000,
			"the table must stay at most half full",
		);
	}

	// -------------------------------------------------------------------
	// Differential, against the structure this replaces
	// -------------------------------------------------------------------

	/// Walks a queue through the PUBLIC surface only, so it can be applied to
	/// either structure and the two answers compared.
	macro_rules! walk {
		($set:expr, $q:expr) => {{
			let mut out = Vec::new();
			let mut cur = $set.front($q);
			while let Some(k) = cur {
				out.push(k);
				cur = $set.after(k);
			}
			out
		}};
	}

	/// v4 must be indistinguishable from `CompactQueueSet` through the API, not
	/// merely correct on its own terms. This drives both with one randomized op
	/// stream over two queues and compares every observable after every op --
	/// which is what catches a divergence that a self-consistent structure
	/// would never notice.
	#[test]
	fn is_indistinguishable_from_the_compact_queue_set_under_random_ops() {
		let mut a: CompactQueueSet<P> = Default::default();
		let mut b: ArenaQueueSet<P> = Default::default();

		let mut x: u64 = 0x853C_49E6_748F_EA9B;
		let mut next: u64 = 1;

		for step in 0..200_000u64 {
			x ^= x << 13;
			x ^= x >> 7;
			x ^= x << 17;

			// A small key space against a growing slab: collisions in the
			// queues, reuse of freed slots, and repeated removal of the same
			// keys are all what a cache actually does.
			let key = (x % 512).wrapping_mul(0x9E37_79B9_7F4A_7C15);

			// Which queue the key is in, if any. Every op below is guarded on
			// it: `CompactQueueSet` and v4 alike take the queue from the
			// caller, so naming the wrong one corrupts a length in both and
			// tests nothing.
			let queue = a.payload(key).map(|pl| pl.queue as usize);

			match x % 10 {
				0 | 1 | 2 => {
					if queue.is_none() {
						a.push_front(0, key, p(0));
						b.push_front(0, key, p(0));
					}
				},
				3 => {
					if queue.is_none() {
						a.push_back(1, key, p(1));
						b.push_back(1, key, p(1));
					}
				},
				4 | 5 => {
					if let Some(q) = queue {
						a.move_front(q, key);
						b.move_front(q, key);
					}
				},
				6 => {
					if queue == Some(0) {
						a.move_to_back_of(0, 1, key);
						b.move_to_back_of(0, 1, key);
						if let Some(pl) = a.payload_mut(key) {
							pl.queue = 1;
						}
						if let Some(pl) = b.payload_mut(key) {
							pl.queue = 1;
						}
					}
				},
				7 => {
					if let Some(q) = queue {
						assert_eq!(a.remove(q, key), b.remove(q, key), "remove diverged");
					}
				},
				8 => {
					assert_eq!(a.pop_back(0), b.pop_back(0), "pop_back diverged");
					assert_eq!(a.pop_front(1), b.pop_front(1), "pop_front diverged");
				},
				_ => {
					// A brand-new key, so the slab keeps growing and the index
					// keeps rehashing rather than settling into one table size.
					let fresh = next.wrapping_mul(0xD6E8_FEB8_6659_FD93);
					next += 1;
					if !a.contains(fresh) {
						a.push_front(0, fresh, p(0));
						b.push_front(0, fresh, p(0));
					}
				},
			}

			// Cheap invariants every step; the full walk periodically, since it
			// is O(n) and the stream is long.
			assert_eq!(a.len(), b.len(), "len diverged at step {step}");
			assert_eq!(a.contains(key), b.contains(key), "contains diverged at step {step}");
			assert_eq!(a.payload(key), b.payload(key), "payload diverged at step {step}");
			assert_eq!(a.front(0), b.front(0), "front diverged at step {step}");
			assert_eq!(a.back(0), b.back(0), "back diverged at step {step}");
			assert_eq!(a.before(key), b.before(key), "before diverged at step {step}");
			assert_eq!(a.after(key), b.after(key), "after diverged at step {step}");

			if step % 1_000 == 0 {
				for q in 0..2 {
					assert_eq!(walk!(a, q), walk!(b, q), "queue {q} order diverged at step {step}");
					assert_eq!(a.queue_len(q), b.queue_len(q), "queue {q} len diverged");
				}
			}
		}

		for q in 0..2 {
			assert_eq!(walk!(a, q), walk!(b, q), "queue {q} order diverged at the end");
		}
	}

	// -------------------------------------------------------------------
	// Measurements
	// -------------------------------------------------------------------

	/// Buckets examined by a successful lookup of `key`. One means the key was
	/// found in its own home bucket.
	fn probe_length(set: &ArenaQueueSet<P>, key: HashedKey) -> Option<usize> {
		let capacity = set.index.capacity();

		if capacity == 0 {
			return None;
		}

		let mask = capacity - 1;
		let mut b = set.index.home(key);

		for probes in 1usize.. {
			let slot = set.index.bucket(b);

			if slot == NIL {
				return None;
			}

			if set.slots[slot as usize].key == key {
				return Some(probes);
			}

			b = (b + 1) & mask;
		}

		None
	}

	/// The same table with the mix REMOVED: bucket is `key & mask`, which is
	/// what "the key is already a hash, index it directly" actually means, and
	/// what the `NoHasher`'d index this replaces would have done if hashbrown
	/// did not mix internally to derive its control byte.
	///
	/// Same capacity, same keys, same insertion order, same linear probing. The
	/// mix is the only difference between the two, so the gap between the two
	/// distributions is what the mix buys -- which may be nothing, and saying
	/// so requires measuring it rather than assuming it.
	struct UnmixedTable<'a> {
		keys: &'a [HashedKey],
		buckets: Vec<u32>,
	}

	impl<'a> UnmixedTable<'a> {
		fn build(keys: &'a [HashedKey], capacity: usize) -> Self {
			let mask = capacity - 1;
			let mut buckets = vec![NIL; capacity];

			for (i, &key) in keys.iter().enumerate() {
				let mut b = (key as usize) & mask;
				while buckets[b] != NIL {
					b = (b + 1) & mask;
				}
				buckets[b] = i as u32;
			}

			UnmixedTable { keys, buckets }
		}

		fn probe_length(&self, key: HashedKey) -> usize {
			let mask = self.buckets.len() - 1;
			let mut b = (key as usize) & mask;

			for probes in 1usize.. {
				let at = self.buckets[b];
				assert!(at != NIL, "key absent from the unmixed table");
				if self.keys[at as usize] == key {
					return probes;
				}
				b = (b + 1) & mask;
			}

			unreachable!()
		}
	}

	/// Probe lengths as a histogram rather than a list: they are small integers
	/// and there are tens of millions of them.
	#[derive(Default)]
	struct Probes {
		counts: Vec<u64>,
	}

	impl Probes {
		fn record(&mut self, length: usize) {
			if self.counts.len() <= length {
				self.counts.resize(length + 1, 0);
			}
			self.counts[length] += 1;
		}

		fn report(&self, label: &str) {
			let total: u64 = self.counts.iter().sum();

			if total == 0 {
				println!("PROBE {label} no-samples");
				return;
			}

			let weighted: u64 = self
				.counts
				.iter()
				.enumerate()
				.map(|(length, count)| length as u64 * count)
				.sum();
			let mean = weighted as f64 / total as f64;

			let quantile = |q: f64| -> usize {
				let want = (total as f64 * q).ceil() as u64;
				let mut seen = 0u64;
				for (length, count) in self.counts.iter().enumerate() {
					seen += count;
					if seen >= want {
						return length;
					}
				}
				self.counts.len() - 1
			};

			println!(
				"PROBE {label} n={total} mean={mean:.4} p50={} p90={} p99={} p999={} \
				 p9999={} max={}",
				quantile(0.50),
				quantile(0.90),
				quantile(0.99),
				quantile(0.999),
				quantile(0.9999),
				self.counts.len() - 1,
			);
		}
	}

	/// Probe-length distribution on REAL trace keys.
	///
	/// Not on `0..n`: a dense integer range is the one input on which any mixer
	/// looks perfect, and it is not what the cache sees. This reads keys out of
	/// a Twitter cluster trace in the benchmark's own 25-byte record format
	/// (`u64 timestamp, u8 command, u64 key, u32 value_size, u32 ttl`, all
	/// little-endian) and crosses two choices:
	///
	/// - **hashed** vs **raw**. `hashed` is `RandomState::hash_one(key)`, the
	///   `HashedKey` the shipped cache actually puts in this structure; `raw`
	///   is the trace's own `u64` used directly, which is the adversarial input
	///   for a mixer and what would arrive under an identity hasher.
	/// - **mixed** vs **unmixed**: this structure's Fibonacci multiply against
	///   a bare `key & mask` on the same keys in the same table.
	///
	/// and reports each distribution both per distinct key and weighted by
	/// access, since a hot key is probed once per access and that is what the
	/// cache actually pays.
	///
	/// `V4_TRACE_DISTINCT` caps the resident set. Without it the table lands
	/// wherever the doubling ladder leaves it; set it to a power of two to
	/// measure the WORST phase, a table exactly half full.
	///
	/// Ignored by default; it wants a multi-GB trace file. Run with:
	/// `V4_TRACE=/path/cluster26.bin cargo test --release --lib -- --ignored \
	///  --nocapture measure_probe_lengths`
	#[test]
	#[ignore]
	fn measure_probe_lengths_on_a_real_trace() {
		use std::io::Read;

		let Ok(path) = std::env::var("V4_TRACE") else {
			println!("PROBE skipped -- set V4_TRACE to a trace file");
			return;
		};

		let records: usize = std::env::var("V4_TRACE_RECORDS")
			.map(|v| v.parse().expect("V4_TRACE_RECORDS"))
			.unwrap_or(20_000_000);
		let cap: usize = std::env::var("V4_TRACE_DISTINCT")
			.map(|v| v.parse().expect("V4_TRACE_DISTINCT"))
			.unwrap_or(usize::MAX);

		const RECORD: usize = 25;
		const KEY_AT: usize = 9;

		let mut file = std::fs::File::open(&path).expect("trace file");
		let mut buffer = vec![0u8; RECORD * 65_536];
		let mut raw_keys: Vec<u64> = Vec::with_capacity(records.min(1 << 26));

		while raw_keys.len() < records {
			let mut filled = 0;
			while filled < buffer.len() {
				match file.read(&mut buffer[filled..]) {
					Ok(0) => break,
					Ok(n) => filled += n,
					Err(e) => panic!("reading {path}: {e}"),
				}
			}

			if filled < RECORD {
				break;
			}

			for chunk in buffer[..filled].chunks_exact(RECORD) {
				let mut key = [0u8; 8];
				key.copy_from_slice(&chunk[KEY_AT..KEY_AT + 8]);
				raw_keys.push(u64::from_le_bytes(key));
			}

			if filled < buffer.len() {
				break;
			}
		}

		assert!(!raw_keys.is_empty(), "no records read from {path}");

		for label in ["hashed", "raw"] {
			use std::hash::{BuildHasher, RandomState};

			// One fixed `RandomState` for the whole run, exactly as one cache
			// instance has one hasher for its lifetime.
			let state = RandomState::new();
			let derive = |k: u64| -> HashedKey {
				if label == "hashed" { state.hash_one(k) } else { k }
			};

			let mut set: ArenaQueueSet<P> = Default::default();
			let mut distinct: Vec<HashedKey> = Vec::new();

			for &k in &raw_keys {
				if distinct.len() >= cap {
					break;
				}
				let key = derive(k);
				if !set.contains(key) {
					set.push_back(0, key, p(0));
					distinct.push(key);
				}
			}

			let capacity = set.index_capacity();
			let unmixed = UnmixedTable::build(&distinct, capacity);

			let mut mixed_keys = Probes::default();
			let mut unmixed_keys = Probes::default();
			for &key in &distinct {
				mixed_keys.record(probe_length(&set, key).expect("inserted"));
				unmixed_keys.record(unmixed.probe_length(key));
			}

			// Access-weighted: what the cache actually pays, since a hot key is
			// probed on every one of its accesses. Accesses to keys the cap
			// excluded are skipped -- they are not in the table.
			let mut mixed_hits = Probes::default();
			let mut unmixed_hits = Probes::default();
			let mut accesses = 0u64;
			for &k in &raw_keys {
				let key = derive(k);
				let Some(length) = probe_length(&set, key) else { continue };
				mixed_hits.record(length);
				unmixed_hits.record(unmixed.probe_length(key));
				accesses += 1;
			}

			println!(
				"PROBE {label} records={} accesses={accesses} distinct={} table={capacity} \
				 load={:.4} index_bytes_per_object={:.2}",
				raw_keys.len(),
				distinct.len(),
				distinct.len() as f64 / capacity as f64,
				(capacity * core::mem::size_of::<u32>()) as f64 / distinct.len() as f64,
			);
			mixed_keys.report(&format!("{label}/mixed/per-key"));
			mixed_hits.report(&format!("{label}/mixed/access-weighted"));
			unmixed_keys.report(&format!("{label}/unmixed/per-key"));
			unmixed_hits.report(&format!("{label}/unmixed/access-weighted"));
		}
	}

	/// Bytes this structure allocates per tracked object, one point per
	/// process, against `CompactQueueSet` measured the same way.
	///
	/// Same rules as `measure_overhead`: jemalloc `stats.allocated` rather than
	/// RSS, ONE point per process (a doubling abandons its old buffer, so two
	/// in-process points are separated by whatever slack lies between them),
	/// and the caller samples at POWERS OF TWO so every point sits at the same
	/// phase of the doubling cycle.
	///
	/// Run with:
	/// `V4_MEASURE_N=8388608 V4_MEASURE_SET=v4 cargo test --release --lib -- \
	///  --ignored --nocapture measure_bytes_per_object`
	/// The figure that actually matters: the arena carrying the shared node,
	/// not the eight-byte stand-in the test above uses.
	///
	/// `ARENA_MEASURE_N=$((1<<22))`, one process per point, release.
	#[test]
	#[ignore]
	fn measure_node_bytes_per_object() {
		let Ok(n) = std::env::var("ARENA_MEASURE_N") else {
			println!("ARENABYTES skipped -- set ARENA_MEASURE_N");
			return;
		};
		let n: usize = n.parse().expect("ARENA_MEASURE_N");

		let base = crate::worker::policy::policy_stack::measure_overhead::allocated_bytes();

		let mut set: ArenaQueueSet<NodePayload> = ArenaQueueSet::default();
		for i in 0..n as u64 {
			set.push_front(
				0,
				i.wrapping_mul(0x9E37_79B9_7F4A_7C15),
				NodePayload {
					size: 64,
					freq: 0,
					ts: 0,
					queue: 0,
					tier: Some(Tier::Fast),
					phys: Some(Tier::Fast),
					dram_resident: 0,
				},
			);
		}

		let after = crate::worker::policy::policy_stack::measure_overhead::allocated_bytes();
		core::hint::black_box(&set);

		let used = after.saturating_sub(base);
		println!("ARENABYTES {} {} {:.4}", n, used, used as f64 / n as f64);
	}

	#[test]
	#[ignore]
	fn measure_bytes_per_object() {
		let Ok(n) = std::env::var("V4_MEASURE_N") else {
			println!("V4BYTES skipped -- set V4_MEASURE_N");
			return;
		};
		let n: u64 = n.parse().expect("V4_MEASURE_N");
		let which = std::env::var("V4_MEASURE_SET").unwrap_or_else(|_| "v4".into());

		let base = crate::worker::policy::policy_stack::measure_overhead::allocated_bytes();
		let used = match which.as_str() {
			"v4" => {
				let mut s: ArenaQueueSet<P> = Default::default();
				for i in 0..n {
					s.push_back(0, i.wrapping_mul(GOLDEN), p(0));
				}
				let after =
					crate::worker::policy::policy_stack::measure_overhead::allocated_bytes();
				core::hint::black_box(&s);
				after.saturating_sub(base)
			},
			"compact" => {
				let mut s: CompactQueueSet<P> = Default::default();
				for i in 0..n {
					s.push_back(0, i.wrapping_mul(GOLDEN), p(0));
				}
				let after =
					crate::worker::policy::policy_stack::measure_overhead::allocated_bytes();
				core::hint::black_box(&s);
				after.saturating_sub(base)
			},
			other => panic!("V4_MEASURE_SET must be v4 or compact, got {other}"),
		};

		println!(
			"V4BYTES {which} {n} {used} {:.4}",
			used as f64 / n as f64,
		);
	}

	/// All FOUR structures at ONE population, in ONE process, on the SAME
	/// objects -- and at a population a real run actually reaches rather than at
	/// a power of two.
	///
	/// # Why this exists
	///
	/// Every `*_EVICTION_STACK_DRAM_OVERHEAD` constant in `object::overhead` is
	/// measured at a POWER OF TWO. That is deliberate and it is what makes the
	/// constants comparable with each other -- but it is also the densest phase
	/// of every structure involved, so every one of them is a FLOOR:
	///
	/// - a `Vec`-backed slab at `n = 2^k` grew into a capacity it exactly fills;
	///   away from that point the slot term is `size x capacity/len`;
	/// - v4's index is a power-of-two open-addressed table, so at `n = 2^k` with
	///   2x slack it sits at exactly half load and 4 B/bucket is 8 B/object;
	///   just after a doubling the same table costs 16;
	/// - hashbrown resizes at 7/8, so a `DashMap` shard at `n = 2^k` is barely
	///   past a doubling and at `n = 7/8 x 2^k` it is at its densest.
	///
	/// The three constants therefore describe three structures all photographed
	/// at their best angle. This measures the same three plus the object map at
	/// ONE arbitrary population -- pass the object count a benchmark summary
	/// actually reports -- so the question "does the ORDERING survive off a power
	/// of two, and by how much" has an answer rather than an assumption.
	///
	/// Same shape and the same arms as `merged_stack_v3::tests::
	/// prints_the_measured_metadata_per_object`, with `ArenaQueueSet` added as a
	/// fourth, so the numbers are directly comparable to the ones that test
	/// prints at 2^20..2^22. Same method as everywhere else in this tree:
	/// jemalloc `stats.allocated` (size-class-rounded usable bytes, never RSS),
	/// the value and its refcount header measured separately and differenced
	/// out, and each arm built and dropped in turn -- `stats.allocated` is a LIVE
	/// figure, so a freed arm leaves nothing behind for the next one.
	///
	/// Ignored, and must run single-threaded: the statistic is process-wide, so
	/// any other test allocating concurrently lands in the delta.
	///
	///   MEASURE_POP_N=4816172 cargo +nightly test --release --lib \
	///     --features merged_object_store_v3 -- --ignored --nocapture \
	///     --test-threads=1 measure_four_structures_at_one_population
	#[cfg(all(feature = "merged_object_store_v3", feature = "hybrid_cache_common"))]
	#[test]
	#[ignore]
	fn measure_four_structures_at_one_population() {
		use crate::{
			merged_store_v3::SlottedStore,
			object::Object,
			worker::policy::policy_stack::{
				measure_overhead::allocated_bytes,
				slot_arena::SlotArena,
			},
		};

		let n: usize = std::env::var("MEASURE_POP_N")
			.map(|v| v.parse().expect("MEASURE_POP_N"))
			.unwrap_or(1 << 20);

		let vsize: usize = std::env::var("MEASURE_POP_VALUE")
			.map(|v| v.parse().expect("MEASURE_POP_VALUE"))
			.unwrap_or(64);

		type Obj = Object<u64, crate::TieredBuffer>;

		let value = vec![0u8; vsize];
		let key_of = |i: usize| (i as u64).wrapping_mul(GOLDEN);
		let payload = p(0);
		let per = |delta: u64| delta as f64 / n as f64;

		// -- the value and its refcount header, which every arm pays and which
		//    therefore comes off all of them before they are compared.
		let base = allocated_bytes();
		let mut held: Vec<Obj> = Vec::with_capacity(n);
		for i in 0..n {
			held.push(Obj::new(key_of(i), crate::TieredBuffer::new_fast(&value), None));
		}
		let values = allocated_bytes().saturating_sub(base);
		core::hint::black_box(&held);
		drop(held);

		let value_heap = per(values) - core::mem::size_of::<Obj>() as f64;

		// -- the object map alone, the same DashMap under every design
		let base = allocated_bytes();
		let map: dashmap::DashMap<HashedKey, Obj, crate::NoHasher> =
			dashmap::DashMap::with_hasher(crate::NoHasher::default());
		for i in 0..n {
			map.insert(key_of(i), Obj::new(key_of(i), crate::TieredBuffer::new_fast(&value), None));
		}
		let map_only = allocated_bytes().saturating_sub(base);
		core::hint::black_box(&map);
		drop(map);

		// -- split: that map plus `CompactQueueSet`, index and all
		let base = allocated_bytes();
		let map: dashmap::DashMap<HashedKey, Obj, crate::NoHasher> =
			dashmap::DashMap::with_hasher(crate::NoHasher::default());
		let mut split: CompactQueueSet<P> = Default::default();
		for i in 0..n {
			map.insert(key_of(i), Obj::new(key_of(i), crate::TieredBuffer::new_fast(&value), None));
			split.push_front(0, key_of(i), payload);
		}
		let split_total = allocated_bytes().saturating_sub(base);
		core::hint::black_box((&map, &split));
		drop((map, split));

		// -- v4: the same map plus `ArenaQueueSet`
		let base = allocated_bytes();
		let map: dashmap::DashMap<HashedKey, Obj, crate::NoHasher> =
			dashmap::DashMap::with_hasher(crate::NoHasher::default());
		let mut v4: ArenaQueueSet<P> = Default::default();
		for i in 0..n {
			map.insert(key_of(i), Obj::new(key_of(i), crate::TieredBuffer::new_fast(&value), None));
			v4.push_front(0, key_of(i), payload);
		}
		let v4_total = allocated_bytes().saturating_sub(base);
		let v4_slab = v4.slab_capacity();
		let v4_index = v4.index_capacity();
		core::hint::black_box((&map, &v4));
		drop((map, v4));

		// -- v3: the store plus the arena, linked exactly as the worker links it
		let base = allocated_bytes();
		let store: SlottedStore<u64, crate::TieredBuffer> = SlottedStore::new();
		let mut arena: SlotArena<P> = Default::default();
		for i in 0..n {
			store.insert(key_of(i), Obj::new(key_of(i), crate::TieredBuffer::new_fast(&value), None));
			let slot = arena.push_front(key_of(i), payload);
			store.set_slot(key_of(i), slot);
		}
		let v3_total = allocated_bytes().saturating_sub(base);
		core::hint::black_box((&store, &arena));
		drop((store, arena));

		let row = per(map_only) - value_heap;
		let compact = per(split_total) - per(map_only);
		let v4_set = per(v4_total) - per(map_only);
		let arena_only = per(v3_total) - per(map_only);

		println!("\nMEASURED_POP n={n} value={vsize}  (B/object, jemalloc stats.allocated)");
		println!("  v4 slab capacity {v4_slab} for {n} slots, index capacity {v4_index} \
			 (load {:.4})", n as f64 / v4_index as f64);
		println!("  value + refcount header      {value_heap:8.2}   (common to all, subtracted below)");
		println!("  object map row (structural)  {row:8.2}   charged as OBJECT_MAP_ENTRY_OVERHEAD = 96");
		println!("  CompactQueueSet              {compact:8.2}   charged as 72");
		println!("  ArenaQueueSet                   {v4_set:8.2}   charged as 32");
		println!("  SlotArena                    {arena_only:8.2}   charged as 24");
		println!("  ---");
		println!("  split whole-design           {:8.2}", row + compact);
		println!("  v4    whole-design           {:8.2}", row + v4_set);
		println!("  v3    whole-design           {:8.2}", row + arena_only);
		println!();
	}
}
