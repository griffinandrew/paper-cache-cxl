/*
 * Copyright (c) Kia Shakiba
 *
 * This source code is licensed under the GNU AGPLv3 license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! [`CompactFrequencyChain`] with the key stored ONCE instead of twice.
//!
//! # What this is
//!
//! The frequency-ordered half of the arena conversion. Nineteen of the
//! twenty-one hybrid compact stacks moved from `CompactQueueSet` to
//! [`ArenaQueueSet`] by changing a type name, because their orders are queues
//! and `ArenaQueueSet` holds queues. The two LFU-ranked stacks could not:
//!
//! * `LfuCompactHybridStack` needs one ordered bucket per DISTINCT FREQUENCY,
//!   per tier, because eviction has to find the MINIMUM frequency. That is a
//!   `BTreeMap<u32, (head, tail)>` -- unbounded and data-dependent -- and a
//!   fixed `[u32; MAX_QUEUES]` tag cannot express it.
//!
//! * `LruLfuCompactHybridStack` additionally threads a THIRD, recency-ordered
//!   list over the same slab, for a fast tier that ranks by recency beside a
//!   slow tier that ranks by frequency. That works because a key is in exactly
//!   one tier at a time, so one `prev`/`next` pair per slot serves whichever
//!   list the key is currently in.
//!
//! So the ORDERS are kept from `CompactFrequencyChain` and the STORAGE is taken
//! from the arena. The bucket maps stay: they are O(distinct frequencies), not
//! O(objects), and were never the cost.
//!
//! # What changes, and what it is worth
//!
//! Only the index. `CompactFrequencyChain` is a 16-byte slot plus a
//! `HashMap<HashedKey, (u32, CompactEntry), NoHasher>`, and that map stores the
//! key a second time so a probe can compare it -- the same 8 bytes already in
//! the slot, in the same structure. [`KeylessIndex`] stores bare `u32` slot
//! numbers and verifies a probe against `slots[i].key`, so the index falls from
//! 56 B/object to 8.
//!
//! ```text
//! CompactFrequencyChain   slot 16 (key,prev,next)  + index 56           = 72
//! ArenaFrequencyChain     node 32 (key,prev,next,NodePayload) + index 8 = 40
//! ```
//!
//! The node grows 16 bytes because the payload moves INTO it and because it is
//! the shared [`NodePayload`] rather than a 12-byte `CompactEntry` -- the same
//! node all nineteen converted stacks now carry, so an LFU key and an LRU key
//! cost the same and no stack needs a payload of its own. Against a 56-byte
//! index that trade is 32 B/object.
//!
//! MEASURED with `measure_one_point`, release, ONE PROCESS PER POINT, at powers
//! of two, `MEASURE_POLICY=lfu-compact-hybrid`:
//!
//! ```text
//!   n        before (CompactFrequencyChain)   after (this)
//!   2^20              72.5952                   40.2100
//!   2^21              72.2962                   40.1050
//!   2^22              72.1451                   40.0518
//!   2^23              72.0707                   40.0244
//! ```
//!
//! Forty is predicted rather than fitted: the node is 32 bytes and the keyless
//! bucket array is 8 B/object at the doubling slack it holds. The residue above
//! it is a fixed intercept, not a per-object term, which is why it shrinks with
//! n. The figures are identical to `lru-compact-hybrid`'s, which is the
//! prediction the shared node makes and the check that the bucket maps really
//! are O(distinct frequencies): if they scaled with the object count they would
//! show up here as a term the queue stacks do not have.
//!
//! # What does not change
//!
//! The algorithm, method for method. Buckets are still one-per-distinct-
//! frequency, `bump` is still an O(1) unlink and relink into the adjacent
//! bucket, the minimum is still the first entry of an ordered map, and the
//! recency list is still invisible to the frequency-only path. Two differential
//! tests in this file drive this structure and `CompactFrequencyChain` through
//! the same random operation streams -- one per FACE, since the frequency stack
//! and the recency stack use the chain under different contracts -- and require
//! identical observable state at every step. That is the evidence that
//! "representation change only" is a fact about this commit rather than an
//! intention.
//!
//! [`ArenaQueueSet`]: super::arena_queue_set::ArenaQueueSet
//! [`CompactFrequencyChain`]: super::compact_frequency_chain::CompactFrequencyChain

use std::collections::BTreeMap;

use crate::{
	HashedKey,
	object::ObjectSize,
	worker::policy::policy_stack::{
		Tier,
		arena_index::{
			ArenaSlot,
			KeylessIndex,
			NIL,
			SlotVec,
			U32Vec,
			new_slot_vec,
			new_u32_vec,
		},
		arena_queue_set::NodePayload,
	},
};

// The bucket maps carry the same allocator gating as everything else here, and
// for the same reason: `get_hybrid_dram_shared_overhead` drops the
// eviction-stack DRAM charge to ZERO under `eviction_stacks_pmem`, on the
// premise the stack is not in DRAM. A map that ignored the gate would sit in
// DRAM and be charged nothing.
//
// One entry per DISTINCT frequency rather than per object, so this stays small.
#[cfg(not(feature = "eviction_stacks_pmem"))]
type BucketMap = BTreeMap<u32, (u32, u32)>;
#[cfg(feature = "eviction_stacks_pmem")]
type BucketMap = BTreeMap<u32, (u32, u32), crate::Hybrid>;

#[cfg(not(feature = "eviction_stacks_pmem"))]
fn new_bucket_maps() -> (BucketMap, BucketMap) {
	(BTreeMap::new(), BTreeMap::new())
}

#[cfg(feature = "eviction_stacks_pmem")]
fn new_bucket_maps() -> (BucketMap, BucketMap) {
	(BTreeMap::new_in(crate::Hybrid), BTreeMap::new_in(crate::Hybrid))
}

/// Frequency-ordered buckets per tier, plus a recency-ordered list, over one
/// arena slab addressed by one keyless index.
pub struct ArenaFrequencyChain {
	slots: SlotVec<NodePayload>,

	/// Bare slot numbers, verified against the slot's own key. This is the
	/// whole of the saving over `CompactFrequencyChain`; see
	/// [`arena_index`](super::arena_index).
	index: KeylessIndex,

	/// Freed slab slots, reused before the slab grows.
	free: U32Vec,

	/// frequency -> (head, tail) of that bucket's intrusive list, one map per
	/// tier. Ordered, so a tier's minimum frequency is its first entry.
	///
	/// Two bucket sets over *one* slab is what lets this replace both of
	/// `FrequencyChain`'s chains **and** the `entries` map they were paired
	/// with: a key is located in a single probe, and the slot that probe
	/// returns already carries its tier, size and frequency.
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
	/// The frequency-bucket stack (`LfuCompactHybridStack`) never calls a
	/// `recency_*` method, so for it these stay `NIL` for the structure's whole
	/// life and every other method behaves exactly as it would without them.
	recency_head: u32,
	recency_tail: u32,
}

impl Default for ArenaFrequencyChain {
	fn default() -> Self {
		let (fast_buckets, slow_buckets) = new_bucket_maps();

		ArenaFrequencyChain {
			slots: new_slot_vec(),
			index: KeylessIndex::default(),
			free: new_u32_vec(),
			fast_buckets,
			slow_buckets,
			fast_len: 0,
			slow_len: 0,
			recency_head: NIL,
			recency_tail: NIL,
		}
	}
}

/// A brand-new node for this chain.
///
/// `phys` is set equal to `tier` and kept there by every tier move below.
/// Nothing in either LFU stack reads it -- only the lazy-copy design does, and
/// only because it promotes logically and defers the byte copy -- but the
/// node's contract is that the two are equal for everyone else, and a `phys`
/// left behind at admission tier would quietly make that false.
///
/// `ts` and `queue` stay zero: recency here is `prev`/`next` and there are no
/// queues. They exist so the node is the one shape every policy shares.
fn node(size: ObjectSize, freq: u32, dram_resident: u8, tier: Tier) -> NodePayload {
	NodePayload {
		size,
		freq,
		ts: 0,
		queue: 0,
		tier: Some(tier),
		phys: Some(tier),
		dram_resident,
	}
}

impl ArenaFrequencyChain {
	/// Slab slots currently allocated. Exposed so a test can assert that
	/// construction does NOT allocate from the cache budget: these stacks grow
	/// dynamically, and an eager reservation sized from capacity was removed
	/// because it reserved far more than the eval workload can hold while still
	/// not preventing doubling on the real traces.
	pub fn slab_capacity(&self) -> usize {
		self.slots.capacity()
	}

	/// Buckets in the keyless index. Exposed for the per-object measurement,
	/// which has to know the table size to reason about its cost.
	pub fn index_capacity(&self) -> usize {
		self.index.capacity()
	}

	/// Pre-sizes the slab and the index for `objects` entries.
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
		self.index.reserve(&self.slots, objects);
	}

	pub fn len(&self) -> usize { self.fast_len + self.slow_len }
	pub fn is_empty(&self) -> bool { self.len() == 0 }
	pub fn fast_len(&self) -> usize { self.fast_len }
	pub fn slow_len(&self) -> usize { self.slow_len }

	pub fn contains(&self, key: HashedKey) -> bool {
		self.index.get(&self.slots, key) != NIL
	}

	/// One index probe, then one slab dereference.
	///
	/// `CompactFrequencyChain` returned this in a single probe, because its
	/// payload rode in the hash bucket. What is not the same as that trade is
	/// the size of the thing probed first: the index here is 8 B/object against
	/// 56 there, so the first touch is into a table seven times likelier to be
	/// cache-resident.
	pub fn get(&self, key: HashedKey) -> Option<NodePayload> {
		self.slot_of(key).map(|slot| self.slots[slot as usize].payload)
	}

	/// Admits a key at frequency 1.
	pub fn insert(&mut self, key: HashedKey, size: ObjectSize, dram_resident: u8, tier: Tier) {
		if self.contains(key) {
			return;
		}

		let slot = self.alloc_slot(key, node(size, 1, dram_resident, tier));

		self.index.insert(&self.slots, slot);
		self.link(slot, 1, tier);

		match tier {
			Tier::Fast => self.fast_len += 1,
			Tier::Slow => self.slow_len += 1,
		}
	}

	/// Moves a key to the next frequency bucket. O(1): unlink, relink.
	pub fn bump(&mut self, key: HashedKey) -> u32 {
		let Some(slot) = self.slot_of(key) else { return 0 };
		let payload = self.slots[slot as usize].payload;
		let Some(tier) = payload.tier else { return 0 };

		self.unlink(slot, payload.freq, tier);

		let next_freq = payload.freq.saturating_add(1);
		self.slots[slot as usize].payload.freq = next_freq;
		self.link(slot, next_freq, tier);

		next_freq
	}

	/// The least-frequently-used key in a tier: head of its lowest-frequency
	/// bucket. O(log D) in the number of distinct frequencies present.
	pub fn min_key(&self, tier: Tier) -> Option<HashedKey> {
		let (_, &(head, _)) = self.buckets(tier).iter().next()?;
		Some(self.slots[head as usize].key)
	}

	/// The lowest frequency present in a tier, or `None` if it is empty.
	///
	/// The promotion rule compares a slow key's new count against this: a slow
	/// key overtakes the fast tier only by *strictly* exceeding its minimum.
	pub fn min_count(&self, tier: Tier) -> Option<u32> {
		self.buckets(tier).keys().next().copied()
	}

	/// The least-frequently-used key in a tier together with its count.
	pub fn min_with_count(&self, tier: Tier) -> Option<(HashedKey, u32)> {
		let (&freq, &(head, _)) = self.buckets(tier).iter().next()?;
		Some((self.slots[head as usize].key, freq))
	}

	pub fn remove(&mut self, key: HashedKey) -> Option<NodePayload> {
		let slot = self.index.remove(&self.slots, key)?;
		let payload = self.slots[slot as usize].payload;

		match payload.tier {
			Some(Tier::Fast) => {
				self.unlink(slot, payload.freq, Tier::Fast);
				self.fast_len -= 1;
			},

			Some(Tier::Slow) => {
				self.unlink(slot, payload.freq, Tier::Slow);
				self.slow_len -= 1;
			},

			// The shared node makes `tier` optional because the 2Q and S3-FIFO
			// families legitimately have a queue with no tier of its own. Every
			// path into this chain records one, so this arm is unreachable --
			// and it is spelled out rather than papered over with `unwrap_or`,
			// which would silently unlink the slot from the wrong bucket set
			// and leave the other one holding a freed slot.
			None => {},
		}

		self.free.push(slot);

		Some(payload)
	}

	/// Moves a key between tiers, preserving its frequency. Relinks it from one
	/// bucket set into the other -- the key never moves in the slab, so its
	/// index entry and every link to it stay valid.
	pub fn set_tier(&mut self, key: HashedKey, tier: Tier) {
		let Some(slot) = self.slot_of(key) else { return };
		let payload = self.slots[slot as usize].payload;
		let Some(old_tier) = payload.tier else { return };

		if old_tier == tier {
			return;
		}

		self.unlink(slot, payload.freq, old_tier);
		self.set_tier_fields(slot, tier);
		self.link(slot, payload.freq, tier);

		match tier {
			Tier::Fast => { self.fast_len += 1; self.slow_len -= 1; },
			Tier::Slow => { self.slow_len += 1; self.fast_len -= 1; },
		}
	}

	pub fn resize(&mut self, key: HashedKey, size: ObjectSize, dram_resident: u8) {
		let Some(slot) = self.slot_of(key) else { return };
		let payload = &mut self.slots[slot as usize].payload;

		payload.size = size;
		payload.dram_resident = dram_resident;
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

	/// The slot holding `key`, or `None`. One probe into the keyless index.
	#[inline]
	fn slot_of(&self, key: HashedKey) -> Option<u32> {
		match self.index.get(&self.slots, key) {
			NIL => None,
			slot => Some(slot),
		}
	}

	fn buckets(&self, tier: Tier) -> &BucketMap {
		match tier {
			Tier::Fast => &self.fast_buckets,
			Tier::Slow => &self.slow_buckets,
		}
	}

	/// Records a tier on a slot, keeping `phys` equal to it. See [`node`].
	fn set_tier_fields(&mut self, slot: u32, tier: Tier) {
		let payload = &mut self.slots[slot as usize].payload;

		payload.tier = Some(tier);
		payload.phys = Some(tier);
	}

	/// Takes a free slab slot, or grows the slab by one. Shared by the
	/// frequency admissions and the recency ones below -- one slab, one free
	/// list, whichever list the key joins.
	fn alloc_slot(&mut self, key: HashedKey, payload: NodePayload) -> u32 {
		let node = ArenaSlot { key, prev: NIL, next: NIL, payload };

		match self.free.pop() {
			Some(slot) => {
				self.slots[slot as usize] = node;
				slot
			},

			None => {
				let slot = self.slots.len() as u32;
				assert!(slot != NIL, "ArenaFrequencyChain exceeded u32::MAX - 1 slots");
				self.slots.push(node);
				slot
			},
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

		let slot = self.alloc_slot(key, node(size, freq, dram_resident, Tier::Fast));

		self.index.insert(&self.slots, slot);
		self.recency_link_front(slot);

		self.fast_len += 1;
	}

	/// Moves an existing recency-list key to the head. O(1).
	pub fn recency_move_front(&mut self, key: HashedKey) {
		let Some(slot) = self.slot_of(key) else { return };

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
	pub fn recency_remove(&mut self, key: HashedKey) -> Option<NodePayload> {
		let slot = self.index.remove(&self.slots, key)?;
		let payload = self.slots[slot as usize].payload;

		self.recency_unlink(slot);
		self.free.push(slot);
		self.fast_len -= 1;

		Some(payload)
	}

	/// Moves the recency tail into the slow tier, into the bucket for the
	/// frequency it already carries. Returns the demoted key and its entry as
	/// it now stands.
	///
	/// This is the whole of a demotion. `FrequencyChain` needs a `pop_back`
	/// from one structure and an `insert_at` into another with the count passed
	/// across by hand; here the entry never moves in the slab, so the count is
	/// carried by construction and there is nothing to drop.
	///
	/// `CompactFrequencyChain` looked the tail's key back up in its index and
	/// bailed out if it was missing, a state it documented as impossible. It is
	/// not merely impossible but unrepresentable -- the recency list is
	/// threaded through the slots the index points at, so a listed slot IS an
	/// indexed slot -- so the tail's payload is read straight out of the slab
	/// and there is no bail-out arm to be wrong about.
	pub fn demote_recency_back(&mut self) -> Option<(HashedKey, NodePayload)> {
		if self.recency_tail == NIL {
			return None;
		}

		let slot = self.recency_tail;
		let key = self.slots[slot as usize].key;

		self.set_tier_fields(slot, Tier::Slow);
		let payload = self.slots[slot as usize].payload;

		self.recency_unlink(slot);
		self.link(slot, payload.freq, Tier::Slow);

		self.fast_len -= 1;
		self.slow_len += 1;

		Some((key, payload))
	}

	/// Moves a slow-tier key to the recency head, setting its frequency to
	/// `freq`. The whole of a promotion; `None` if the key is untracked or is
	/// not in the slow tier.
	pub fn promote_to_recency_front(&mut self, key: HashedKey, freq: u32) -> Option<NodePayload> {
		let slot = self.slot_of(key)?;
		let payload = self.slots[slot as usize].payload;

		if payload.tier != Some(Tier::Slow) {
			return None;
		}

		self.unlink(slot, payload.freq, Tier::Slow);

		self.set_tier_fields(slot, Tier::Fast);
		self.slots[slot as usize].payload.freq = freq;
		let payload = self.slots[slot as usize].payload;

		self.recency_link_front(slot);

		self.slow_len -= 1;
		self.fast_len += 1;

		Some(payload)
	}

	/// Sets a key's frequency and touches no list.
	///
	/// For a recency-list key only: its counter is carried metadata that ranks
	/// nothing, so there is no bucket to move it between. Calling this on a
	/// bucketed key would leave the buckets keyed on a stale frequency.
	pub fn set_freq(&mut self, key: HashedKey, freq: u32) {
		let Some(slot) = self.slot_of(key) else { return };
		self.slots[slot as usize].payload.freq = freq;
	}

	/// Sets a SLOW key's frequency and relinks it into that bucket.
	///
	/// Relinks even when `freq` is unchanged, which moves the key to the newest
	/// position within its bucket. That is deliberate and matches
	/// `FrequencyChain::move_to`'s unconditional remove-then-insert: at the
	/// frequency cap a further access cannot raise the count, but it still
	/// refreshes the key's standing against its equally-frequent peers.
	pub fn slow_relink_at(&mut self, key: HashedKey, freq: u32) {
		let Some(slot) = self.slot_of(key) else { return };
		let payload = self.slots[slot as usize].payload;

		if payload.tier != Some(Tier::Slow) {
			return;
		}

		self.unlink(slot, payload.freq, Tier::Slow);
		self.slots[slot as usize].payload.freq = freq;
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
	use crate::worker::policy::policy_stack::{
		arena_index::GOLDEN,
		compact_frequency_chain::CompactFrequencyChain,
	};

	/// The premise, in one assertion. The node carries the payload, so a
	/// tracked key costs 32 bytes of slab; the index carries no key, so it
	/// costs four bytes a bucket. Growth in either is paid on EVERY tracked
	/// object in both tiers.
	#[test]
	fn the_node_is_thirty_two_bytes() {
		assert_eq!(
			std::mem::size_of::<ArenaSlot<NodePayload>>(),
			32,
			"key 8 + prev 4 + next 4 + NodePayload 16 = 32",
		);
	}

	/// The index holds slot numbers and nothing else, so its whole cost is four
	/// bytes a bucket at half load. At a power-of-two population that is
	/// exactly the 8 B/object the design is built around, and it is the half of
	/// the 40 that `CompactFrequencyChain`'s 56-byte hashbrown index used to
	/// be.
	#[test]
	fn the_index_costs_eight_bytes_per_object_at_a_power_of_two_population() {
		for exponent in 8..14u32 {
			let n = 1u64 << exponent;
			let mut chain = ArenaFrequencyChain::default();

			for k in 0..n {
				chain.insert(k.wrapping_mul(GOLDEN), 100, 0, Tier::Fast);
			}

			let bytes = chain.index_capacity() * std::mem::size_of::<u32>();
			assert_eq!(
				bytes as u64 / n,
				8,
				"{n} keys sat in a {}-bucket table",
				chain.index_capacity(),
			);
		}
	}

	/// The bucket maps are the reason this structure exists rather than being
	/// an `ArenaQueueSet`, and the reason it can afford them is that they are
	/// O(DISTINCT FREQUENCIES) and not O(objects). Asserted, because if that
	/// ever stopped being true the measured 40 B/object would drift with n and
	/// nothing else here would notice.
	#[test]
	fn the_bucket_maps_are_sized_by_distinct_frequency_not_by_object_count() {
		let mut chain = ArenaFrequencyChain::default();

		for key in 0..10_000u64 {
			chain.insert(key, 100, 0, Tier::Fast);
			// three distinct counts across ten thousand keys
			for _ in 0..(key % 3) { chain.bump(key); }
		}

		assert_eq!(chain.len(), 10_000);
		assert_eq!(chain.fast_buckets.len(), 3, "one bucket per distinct frequency");
		assert!(chain.slow_buckets.is_empty());
	}

	#[test]
	fn least_frequently_used_comes_out_first() {
		let mut c = ArenaFrequencyChain::default();
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
		let mut c = ArenaFrequencyChain::default();
		for key in 1..=3u64 { c.insert(key, 10, 0, Tier::Fast); }

		assert_eq!(c.min_key(Tier::Fast), Some(1), "all at frequency 1, so oldest first");
		c.remove(1);
		assert_eq!(c.min_key(Tier::Fast), Some(2));
	}

	#[test]
	fn bump_moves_between_buckets_and_keeps_the_chain_intact() {
		let mut c = ArenaFrequencyChain::default();
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
		let mut c = ArenaFrequencyChain::default();
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
		let mut c = ArenaFrequencyChain::default();
		for key in 1..=5u64 { c.insert(key, 10, 0, Tier::Fast); }

		c.remove(3);

		assert_eq!(c.len(), 4);
		let mut seen = Vec::new();
		while let Some(k) = c.min_key(Tier::Fast) { seen.push(k); c.remove(k); }

		assert_eq!(seen, vec![1, 2, 4, 5], "the chain must survive a middle removal");
	}

	/// A tier move is the *only* thing that happens on promotion or demotion.
	///
	/// `FrequencyChain` has to `remove` from one chain and `insert_at` into the
	/// other, carrying the count across by hand. Here the entry never moves in
	/// the slab, so its frequency, size and links are all preserved by
	/// construction -- there is no count to carry and nothing to get wrong.
	#[test]
	fn promotion_is_a_tier_move_and_nothing_else() {
		let mut c = ArenaFrequencyChain::default();
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
		let mut c = ArenaFrequencyChain::default();
		for key in 1..=3u64 { c.insert(key, 100, 24, Tier::Fast); }
		c.bump(2); c.bump(2);

		assert_eq!(c.fast_len(), 3);
		assert_eq!(c.slow_len(), 0);
		assert_eq!(c.min_key(Tier::Fast), Some(1));

		c.set_tier(2, Tier::Slow);

		assert_eq!(c.fast_len(), 2);
		assert_eq!(c.slow_len(), 1);
		assert_eq!(c.get(2).unwrap().freq, 3, "frequency must survive the move");
		assert_eq!(c.get(2).unwrap().tier, Some(Tier::Slow));
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
		let mut c = ArenaFrequencyChain::default();
		c.insert(9, 500, 24, Tier::Fast);

		assert_eq!(c.get(9).unwrap().tier, Some(Tier::Fast));
		assert_eq!(c.get(9).unwrap().migrating(), 476);

		c.set_tier(9, Tier::Slow);
		c.resize(9, 800, 88);

		assert_eq!(c.get(9).unwrap().tier, Some(Tier::Slow));
		assert_eq!(c.get(9).unwrap().migrating(), 712);
	}

	/// `phys` is the one node field this chain writes that neither LFU stack
	/// reads, and the node's contract is that it equals `tier` for every design
	/// but the lazy-copy one. A tier move that left it behind would make that
	/// contract false for a key that had ever been demoted or promoted.
	#[test]
	fn a_tier_move_carries_the_physical_tier_with_it() {
		let mut c = ArenaFrequencyChain::default();
		c.insert(1, 100, 0, Tier::Fast);
		assert_eq!(c.get(1).unwrap().phys, Some(Tier::Fast));

		c.set_tier(1, Tier::Slow);
		assert_eq!(c.get(1).unwrap().phys, Some(Tier::Slow));

		c.recency_push_front(2, 100, 0, 1);
		c.demote_recency_back();
		assert_eq!(c.get(2).unwrap().phys, Some(Tier::Slow));

		c.promote_to_recency_front(2, 1);
		assert_eq!(c.get(2).unwrap().phys, Some(Tier::Fast));
	}

	// ── the distinguished recency list ────────────────────────────────────

	#[test]
	fn the_recency_list_orders_by_recency_not_frequency() {
		let mut c = ArenaFrequencyChain::default();
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
		let mut c = ArenaFrequencyChain::default();
		for key in 1..=3u64 { c.recency_push_front(key, 10, 0, 1); }

		c.recency_move_front(3);

		assert_eq!(c.recency_back(), Some(1));
		assert_eq!(c.fast_len(), 3);
	}

	#[test]
	fn demotion_carries_the_count_into_the_slow_buckets() {
		let mut c = ArenaFrequencyChain::default();
		c.recency_push_front(1, 100, 24, 1);
		c.recency_push_front(2, 200, 24, 1);
		c.set_freq(2, 7);
		c.recency_move_front(2);

		// 1 is the tail; demoting it must land it at ITS count, not 2's.
		let (key, entry) = c.demote_recency_back().unwrap();

		assert_eq!(key, 1);
		assert_eq!(entry.tier, Some(Tier::Slow));
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
		let mut c = ArenaFrequencyChain::default();
		c.recency_push_front(1, 10, 0, 1);
		c.recency_push_front(2, 10, 0, 1);
		c.demote_recency_back().unwrap(); // 1 -> slow

		let entry = c.promote_to_recency_front(1, 1).unwrap();

		assert_eq!(entry.tier, Some(Tier::Fast));
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
		let mut c = ArenaFrequencyChain::default();
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
		let mut c = ArenaFrequencyChain::default();
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
		let mut c = ArenaFrequencyChain::default();
		for key in 1..=5u64 { c.recency_push_front(key, 10, 0, 1); }

		c.recency_remove(3);

		let mut seen = Vec::new();
		while let Some(k) = c.recency_back() { seen.push(k); c.recency_remove(k); }

		assert_eq!(seen, vec![1, 2, 4, 5], "the recency list must survive a middle removal");
	}

	#[test]
	fn clear_resets_the_recency_list_too() {
		let mut c = ArenaFrequencyChain::default();
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
		let mut c = ArenaFrequencyChain::default();
		for key in 1..=10u64 { c.insert(key, 10, 0, Tier::Fast); }
		for key in 1..=5u64 { c.bump(key); }
		c.set_tier(3, Tier::Slow);
		c.remove(7);

		assert_eq!(c.recency_back(), None, "no recency link may exist on the LFU path");
		assert_eq!(c.recency_head, NIL);
		assert_eq!(c.recency_tail, NIL);
		assert_eq!(c.fast_len() + c.slow_len(), 9);
	}

	// ── the differential tests ────────────────────────────────────────────
	//
	// The claim this whole module rests on is that it is a REPRESENTATION
	// change and not an algorithm change. These two drive it and the chain it
	// replaces through the same pseudo-random operation stream and compare
	// every observable at every step. `CompactFrequencyChain` is kept in the
	// tree, test-gated and with no production caller, precisely so they can
	// exist.
	//
	// There are two of them because the chain has two FACES with two different
	// contracts, and the stacks use one each. `LfuCompactHybridStack` buckets
	// every key by frequency and never calls a `recency_*` method;
	// `LruLfuCompactHybridStack` keeps its fast tier in the recency list and
	// its slow tier in the buckets, and routes every operation by the key's
	// current tier. Driving one structure through both contracts at once is not
	// a harder test, it is an undefined one: `recency_move_front` on a bucketed
	// key splices it out of its bucket, and both implementations then corrupt
	// the same bucket in the same way. The first draft of this test did exactly
	// that and panicked identically in either -- which measures nothing.

	/// Every observable the two structures share, compared.
	fn agree(
		arena: &ArenaFrequencyChain,
		compact: &CompactFrequencyChain,
		step: u32,
		keys: &[HashedKey],
		face: &str,
	) {
		assert_eq!(arena.len(), compact.len(), "{face}: len diverged at {step}");
		assert_eq!(
			arena.fast_len(), compact.fast_len(),
			"{face}: fast_len diverged at {step}",
		);
		assert_eq!(
			arena.slow_len(), compact.slow_len(),
			"{face}: slow_len diverged at {step}",
		);
		assert_eq!(
			arena.recency_back(), compact.recency_back(),
			"{face}: recency tail diverged at {step}",
		);

		for tier in [Tier::Fast, Tier::Slow] {
			assert_eq!(
				arena.min_key(tier), compact.min_key(tier),
				"{face}: {tier:?} minimum key diverged at {step}",
			);
			assert_eq!(
				arena.min_count(tier), compact.min_count(tier),
				"{face}: {tier:?} minimum count diverged at {step}",
			);
			assert_eq!(
				arena.min_with_count(tier), compact.min_with_count(tier),
				"{face}: {tier:?} minimum pair diverged at {step}",
			);
		}

		for &k in keys {
			assert_eq!(
				arena.contains(k), compact.contains(k),
				"{face}: membership of {k} diverged at {step}",
			);

			match (arena.get(k), compact.get(k)) {
				(None, None) => {},

				(Some(a), Some(b)) => {
					assert_eq!(a.freq, b.freq, "{face}: freq of {k} diverged at {step}");
					assert_eq!(a.size, b.size, "{face}: size of {k} diverged at {step}");
					assert_eq!(
						a.tier, Some(b.tier),
						"{face}: tier of {k} diverged at {step}",
					);
					assert_eq!(
						a.dram_resident, b.dram_resident,
						"{face}: resident of {k} diverged at {step}",
					);
					assert_eq!(
						a.migrating(), b.migrating(),
						"{face}: migrating bytes of {k} diverged at {step}",
					);
				},

				_ => panic!("{face}: tracking of {k} diverged at {step}"),
			}
		}
	}

	/// xorshift rather than a dev-dependency, from a fixed seed so a failure is
	/// reproducible.
	fn stream(seed: u64) -> impl FnMut() -> u64 {
		let mut state = seed;

		move || {
			state ^= state << 13;
			state ^= state >> 7;
			state ^= state << 17;
			state
		}
	}

	/// Small enough that collisions, re-insertions and removals of absent keys
	/// all happen constantly.
	const KEYS: u64 = 64;

	/// Sweeping all 64 keys on all 40,000 steps is 2.6M debug-mode comparisons
	/// for no extra coverage: a divergence the sweep catches and the touched-key
	/// check does not still had to be created by some step, and 200 operations
	/// is not long enough for one to be created and undone.
	fn sweep(step: u32, key: HashedKey) -> Vec<HashedKey> {
		if step % 200 == 0 { (0..KEYS).collect() } else { vec![key] }
	}

	/// The face `LfuCompactHybridStack` drives: every key in a frequency
	/// bucket, ranked by count, in one of two tiers. No recency list exists on
	/// this path and both structures must leave it empty.
	#[test]
	fn the_frequency_face_agrees_with_the_compact_chain_it_replaces() {
		let mut arena = ArenaFrequencyChain::default();
		let mut compact = CompactFrequencyChain::default();
		let mut next = stream(0x2545_F491_4F6C_DD1D);

		for step in 0..40_000u32 {
			let key = next() % KEYS;
			let size = 64 + (next() % 512) as ObjectSize;
			let resident = (next() % 32) as u8;
			let tier = if next() % 2 == 0 { Tier::Fast } else { Tier::Slow };

			match next() % 6 {
				0 | 1 => {
					arena.insert(key, size, resident, tier);
					compact.insert(key, size, resident, tier);
				},

				2 | 3 => {
					assert_eq!(
						arena.bump(key), compact.bump(key),
						"bump returned a different count at {step}",
					);
				},

				4 => {
					arena.set_tier(key, tier);
					compact.set_tier(key, tier);
				},

				5 if next() % 3 == 0 => {
					arena.resize(key, size, resident);
					compact.resize(key, size, resident);
				},

				_ => {
					assert_eq!(
						arena.remove(key).is_some(),
						compact.remove(key).is_some(),
						"remove disagreed on membership at {step}",
					);
				},
			}

			agree(&arena, &compact, step, &sweep(step, key), "frequency");
			assert_eq!(arena.recency_back(), None, "the LFU path grew a recency list");
		}

		arena.clear();
		compact.clear();
		agree(&arena, &compact, u32::MAX, &(0..KEYS).collect::<Vec<_>>(), "frequency");
	}

	/// The face `LruLfuCompactHybridStack` drives: a recency-ordered fast tier
	/// and a frequency-ordered slow tier over the same slab, with every
	/// operation routed by the key's current tier exactly as that stack routes
	/// it.
	///
	/// Routing reads the ARENA's answer, which is safe because the tier of
	/// every key is compared against the compact chain's on the step before: a
	/// disagreement fails there rather than being laundered into a divergent
	/// operation stream here.
	#[test]
	fn the_recency_face_agrees_with_the_compact_chain_it_replaces() {
		let mut arena = ArenaFrequencyChain::default();
		let mut compact = CompactFrequencyChain::default();
		let mut next = stream(0x9E37_79B9_7F4A_7C15);

		for step in 0..40_000u32 {
			let key = next() % KEYS;
			let size = 64 + (next() % 512) as ObjectSize;
			let resident = (next() % 32) as u8;
			let freq = 1 + (next() % 8) as u32;

			let tier = arena.get(key).and_then(|entry| entry.tier);
			assert_eq!(
				tier, compact.get(key).map(|entry| entry.tier),
				"tier of {key} diverged before step {step} could route on it",
			);

			match next() % 8 {
				0 | 1 => {
					arena.recency_push_front(key, size, resident, freq);
					compact.recency_push_front(key, size, resident, freq);
				},

				2 if tier == Some(Tier::Fast) => {
					arena.recency_move_front(key);
					compact.recency_move_front(key);
					arena.set_freq(key, freq);
					compact.set_freq(key, freq);
				},

				3 => {
					assert_eq!(
						arena.demote_recency_back().map(|(k, e)| (k, e.freq)),
						compact.demote_recency_back().map(|(k, e)| (k, e.freq)),
						"demotion picked a different key or count at {step}",
					);
				},

				4 if tier == Some(Tier::Slow) => {
					assert_eq!(
						arena.promote_to_recency_front(key, freq).map(|e| e.freq),
						compact.promote_to_recency_front(key, freq).map(|e| e.freq),
						"promotion disagreed at {step}",
					);
				},

				5 if tier == Some(Tier::Slow) => {
					arena.slow_relink_at(key, freq);
					compact.slow_relink_at(key, freq);
				},

				6 => {
					arena.resize(key, size, resident);
					compact.resize(key, size, resident);
				},

				// Which door a key leaves by is its tier, exactly as
				// `LruLfuCompactHybridStack::remove` decides it.
				7 => match tier {
					Some(Tier::Fast) => assert_eq!(
						arena.recency_remove(key).is_some(),
						compact.recency_remove(key).is_some(),
						"recency removal disagreed at {step}",
					),

					Some(Tier::Slow) => assert_eq!(
						arena.remove(key).is_some(),
						compact.remove(key).is_some(),
						"slow removal disagreed at {step}",
					),

					None => {},
				},

				// The guarded arms above fall through when their tier
				// condition does not hold, which is itself worth exercising:
				// an untracked key must be a no-op in both. `set_freq` is for
				// recency-list keys only -- on a bucketed key it would leave
				// the bucket keyed on a stale count -- so the fallthrough only
				// takes it for a fast key.
				_ => {
					if tier == Some(Tier::Fast) {
						arena.set_freq(key, freq);
						compact.set_freq(key, freq);
					}
				},
			}

			agree(&arena, &compact, step, &sweep(step, key), "recency");
		}

		arena.clear();
		compact.clear();
		agree(&arena, &compact, u32::MAX, &(0..KEYS).collect::<Vec<_>>(), "recency");
	}
}
