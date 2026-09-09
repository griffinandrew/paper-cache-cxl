/*
 * Copyright (c) Kia Shakiba
 *
 * This source code is licensed under the GNU AGPLv3 license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! The arena's slab node, and the KEYLESS index that finds it.
//!
//! This is the whole of what the arena conversion takes off `CompactQueueSet`
//! and `CompactFrequencyChain`. Both were 72 B/object, and in both the key was
//! stored TWICE: once in the slab slot, so an eviction can name the victim it
//! just unlinked, and once in a hashbrown index, so a probe can compare. Here
//! the bucket array holds bare `u32` slot numbers and nothing else, and a probe
//! is verified against `slots[i].key` -- the copy the slot already had to carry.
//! At a power-of-two population with 2x slack that is `4 B * 2 = 8 B/object`
//! against the 56 the hashbrown index cost.
//!
//! # Why it is its own module
//!
//! Two structures now thread orders over this node, and they need different
//! orders. [`ArenaQueueSet`] holds `MAX_QUEUES` intrusive queues in a fixed
//! `[u32; 4]` of heads and tails. [`ArenaFrequencyChain`] holds one ordered
//! bucket per DISTINCT frequency, in a `BTreeMap` per tier, because LFU
//! eviction has to find the MINIMUM frequency and a four-queue tag cannot
//! express that; it also threads a third, recency-ordered list over the same
//! slab for the design whose fast tier ranks by recency and whose slow tier
//! ranks by frequency.
//!
//! The orders differ. The index does not: both are addressed by key, both
//! allocate slots out of one slab, and both verify a probe against the slot.
//! So it is shared rather than copied. Backward-shift deletion below is subtle
//! enough that a second copy would be a second chance to get it wrong, and a
//! divergence between two copies would move the measured per-object figure that
//! `object::overhead`'s `*_EVICTION_STACK_DRAM_OVERHEAD` constants encode --
//! without looking, at any call site, like a memory change at all.
//!
//! [`ArenaQueueSet`]: super::arena_queue_set::ArenaQueueSet
//! [`ArenaFrequencyChain`]: super::arena_frequency_chain::ArenaFrequencyChain
//!
//! # Deletion, which is where the bugs live
//!
//! Open addressing cannot blank a bucket on removal: every entry after it in
//! the same probe run becomes unreachable. The two repairs are tombstones plus
//! a rehash policy, or **backward-shift deletion**, and this uses the latter.
//!
//! The reason is the workload. A cache at capacity performs one removal per
//! admission, forever. Under tombstones that is one tombstone per admission
//! forever: the table's effective load rises with no bound on the count, probe
//! runs lengthen monotonically between rehashes, and the rehash -- when its
//! threshold finally trips -- is a full-table stall on the policy worker,
//! repeated for the life of the process. It also introduces a tuning knob
//! nobody can set from first principles. Backward-shift deletion restores the
//! table to precisely the state it would have been in had the removed key never
//! been inserted, so steady-state churn has no drift and no periodic stall, at
//! the cost of a short bounded shift on the delete itself. See [`erase_at`].
//!
//! [`erase_at`]: KeylessIndex::erase_at
//!
//! # Hash distribution
//!
//! `HashedKey` is already a hash, which is why `CompactQueueSet` indexes it
//! under `NoHasher` -- but hashbrown still mixes internally to derive its
//! control byte, and a bare `key & mask` here would not. The bucket is taken
//! from the HIGH bits of a Fibonacci multiply (`key * 2^64/phi >> shift`),
//! which is one `imul` and one shift and makes every input bit reach the
//! bucket.
//!
//! MEASURED on cluster26 rather than on `0..n`, which is the one input any
//! mixer looks perfect on. Successful-lookup probe lengths, `1` meaning the
//! key was in its own home bucket:
//!
//! ```text
//!                        load   mean   p50  p90  p99  p999  max
//! 2.16M keys / 2^23      0.258  1.174   1    2    3     6    16
//! 4.19M keys / 2^23      0.500  1.500   1    3    7    13    53
//! ```
//!
//! Access-weighted over 20.1M and 41.9M accesses respectively the means are
//! 1.158 and 1.453 -- slightly better than per-key, so the hot keys are not the
//! displaced ones. Both means are the closed form for linear probing,
//! `(1 + 1/(1-a))/2`, to three decimals, which says the mix leaves nothing
//! clustered on this trace.
//!
//! It also says the mix buys nothing HERE. Running the same keys through a bare
//! `key & mask` in the same table gives 1.173 and 1.500 -- indistinguishable --
//! whether the key is `RandomState::hash_one`'d as the shipped cache does it or
//! the trace's raw `u64` is used directly. cluster26's keys are already spread
//! in their low bits. The mix is kept anyway: it costs one `imul` on a path
//! that then misses to DRAM, and it is the difference between a bound that
//! holds for any key distribution and one that holds for this trace.

use crate::worker::policy::policy_stack::HashedKey;

/// Sentinel for "no slot", and for "empty bucket" in the index. `u32::MAX`
/// rather than `Option<u32>` so a slot stays 16 bytes plus its payload.
pub const NIL: u32 = u32::MAX;

/// 2^64 / phi, rounded to an odd integer. Odd is what makes the multiply a
/// bijection on `u64`, so distinct keys stay distinct before the shift.
pub(crate) const GOLDEN: u64 = 0x9E37_79B9_7F4A_7C15;

/// Smallest index table. Below this the shift arithmetic buys nothing and the
/// table is a rounding error anyway.
const MIN_BUCKETS: usize = 16;

/// One node: the links, the key that names the victim on eviction, AND the
/// payload the index used to carry.
///
/// 16 bytes for a ZST payload, 24 for every 8-byte hybrid payload in this
/// crate, and 32 for the shared `NodePayload`.
#[derive(Clone, Copy, Debug)]
pub struct ArenaSlot<P> {
	pub key: HashedKey,
	pub prev: u32,
	pub next: u32,
	pub payload: P,
}

// Under `eviction_stacks_pmem` every structure built out of these aliases is
// allocated through the crate-wide `Hybrid` allocator (the far CXL/PMEM node),
// exactly as `CompactQueueSet` and `CompactFrequencyChain` are.
//
// Not optional: `get_hybrid_dram_shared_overhead` drops the eviction-stack term
// to ZERO under that feature, on the premise the stack is not in DRAM. A slab
// that ignored these gates would sit in DRAM and be charged nothing -- silently
// wrong rather than merely unoptimised, and it would invalidate every
// far-memory placement experiment run against this tree.
#[cfg(not(feature = "eviction_stacks_pmem"))]
pub type SlotVec<P> = Vec<ArenaSlot<P>>;
#[cfg(feature = "eviction_stacks_pmem")]
pub type SlotVec<P> = Vec<ArenaSlot<P>, crate::Hybrid>;

#[cfg(not(feature = "eviction_stacks_pmem"))]
pub type U32Vec = Vec<u32>;
#[cfg(feature = "eviction_stacks_pmem")]
pub type U32Vec = Vec<u32, crate::Hybrid>;

#[cfg(not(feature = "eviction_stacks_pmem"))]
pub fn new_slot_vec<P>() -> SlotVec<P> {
	Vec::new()
}

#[cfg(feature = "eviction_stacks_pmem")]
pub fn new_slot_vec<P>() -> SlotVec<P> {
	Vec::new_in(crate::Hybrid)
}

#[cfg(not(feature = "eviction_stacks_pmem"))]
pub fn new_u32_vec() -> U32Vec {
	Vec::new()
}

#[cfg(feature = "eviction_stacks_pmem")]
pub fn new_u32_vec() -> U32Vec {
	Vec::new_in(crate::Hybrid)
}

#[cfg(not(feature = "eviction_stacks_pmem"))]
fn new_buckets(capacity: usize) -> U32Vec {
	vec![NIL; capacity]
}

#[cfg(feature = "eviction_stacks_pmem")]
fn new_buckets(capacity: usize) -> U32Vec {
	let mut buckets = Vec::with_capacity_in(capacity, crate::Hybrid);
	buckets.resize(capacity, NIL);
	buckets
}

/// An open-addressed, linear-probed table of slot numbers that holds no keys.
///
/// Every method that has to compare a key takes the owner's slab, because the
/// only copy of the key is the one in the slot. That is the point: the table
/// itself is four bytes a bucket.
///
/// It does NOT own the slab, allocate slots, or maintain any order. Which slot
/// a key gets, and what order the slots are linked in, is the owning
/// structure's business.
pub struct KeylessIndex {
	buckets: U32Vec,

	/// `64 - log2(buckets.len())`, so the bucket is the high bits of the mix.
	/// Meaningless, and never used, while `buckets` is empty.
	bucket_shift: u32,

	/// Occupied buckets. Equal to the owner's length, kept separately so the
	/// index does not depend on the order in which a caller links and indexes
	/// a slot.
	live: usize,
}

impl Default for KeylessIndex {
	fn default() -> Self {
		KeylessIndex {
			buckets: new_u32_vec(),
			bucket_shift: 63,
			live: 0,
		}
	}
}

impl KeylessIndex {
	/// Bucket a key belongs in: the high bits of a Fibonacci multiply.
	///
	/// The high bits and not the low ones because the multiply carries
	/// information upward -- bit 63 of the product depends on every bit of the
	/// key, bit 0 depends only on bit 0.
	#[inline(always)]
	pub fn home(&self, key: HashedKey) -> usize {
		(key.wrapping_mul(GOLDEN) >> self.bucket_shift) as usize
	}

	/// Buckets in the table. Exposed for the probe-length and per-object
	/// measurements, which have to know the table size to reason about a run.
	pub fn capacity(&self) -> usize {
		self.buckets.len()
	}

	/// The slot number in one bucket, or [`NIL`]. Exposed for the same
	/// measurements.
	pub fn bucket(&self, at: usize) -> u32 {
		self.buckets[at]
	}

	/// Slot holding `key`, or [`NIL`].
	#[inline]
	pub fn get<P>(&self, slots: &[ArenaSlot<P>], key: HashedKey) -> u32 {
		if self.buckets.is_empty() {
			return NIL;
		}

		let mask = self.buckets.len() - 1;
		let mut b = self.home(key);

		loop {
			let slot = self.buckets[b];

			if slot == NIL {
				return NIL;
			}

			if slots[slot as usize].key == key {
				return slot;
			}

			b = (b + 1) & mask;
		}
	}

	/// The bucket holding `key`, or `None`. Separate from [`get`] because
	/// removal needs the bucket and lookup needs the slot.
	///
	/// [`get`]: KeylessIndex::get
	#[inline]
	fn bucket_of<P>(&self, slots: &[ArenaSlot<P>], key: HashedKey) -> Option<usize> {
		if self.buckets.is_empty() {
			return None;
		}

		let mask = self.buckets.len() - 1;
		let mut b = self.home(key);

		loop {
			let slot = self.buckets[b];

			if slot == NIL {
				return None;
			}

			if slots[slot as usize].key == key {
				return Some(b);
			}

			b = (b + 1) & mask;
		}
	}

	/// Places an ALREADY-ALLOCATED slot in the table. `slots[slot].key` must
	/// already be written, since that is the only copy of the key there is.
	pub fn insert<P>(&mut self, slots: &[ArenaSlot<P>], slot: u32) {
		self.grow_for_one_more(slots);

		let key = slots[slot as usize].key;
		let mask = self.buckets.len() - 1;
		let mut b = self.home(key);

		loop {
			let occupant = self.buckets[b];

			if occupant == NIL {
				self.buckets[b] = slot;
				self.live += 1;
				return;
			}

			debug_assert!(
				slots[occupant as usize].key != key,
				"KeylessIndex indexed the same key twice",
			);

			b = (b + 1) & mask;
		}
	}

	/// Removes `key` from the table, returning its slot. Frees nothing: the
	/// slab slot is the owner's to reuse.
	pub fn remove<P>(&mut self, slots: &[ArenaSlot<P>], key: HashedKey) -> Option<u32> {
		let bucket = self.bucket_of(slots, key)?;
		let slot = self.buckets[bucket];

		self.erase_at(slots, bucket);
		self.live -= 1;

		Some(slot)
	}

	/// Backward-shift deletion (Knuth's Algorithm R for linear probing).
	///
	/// Blanking `hole` would strand every entry after it whose probe run passes
	/// through `hole`. So the run is walked forward, and each entry is pulled
	/// back into the hole when doing so does not put it BEFORE its own home
	/// bucket -- which is exactly the condition that its displacement from home
	/// is at least the distance from the hole. The hole travels with it. The
	/// walk stops at the first genuinely empty bucket, which bounds the work by
	/// the length of one probe run.
	///
	/// The result is the table that would have existed had the key never been
	/// inserted, so there is no tombstone to accumulate and no rehash to
	/// schedule.
	fn erase_at<P>(&mut self, slots: &[ArenaSlot<P>], bucket: usize) {
		let mask = self.buckets.len() - 1;

		let mut hole = bucket;
		let mut probe = hole;

		loop {
			probe = (probe + 1) & mask;

			let slot = self.buckets[probe];

			if slot == NIL {
				break;
			}

			let home = self.home(slots[slot as usize].key);

			// Displacement of the probed entry from its home, against the
			// distance it would have to travel back. Both measured cyclically,
			// which is what makes this correct across the table's wrap point.
			let displaced = probe.wrapping_sub(home) & mask;
			let distance = probe.wrapping_sub(hole) & mask;

			if displaced >= distance {
				self.buckets[hole] = slot;
				hole = probe;
			}
		}

		self.buckets[hole] = NIL;
	}

	/// Doubles the table when the next insertion would take it past half full.
	///
	/// Half full and not the 87.5% hashbrown uses, because the load factor here
	/// buys two different things at once: it is the whole memory cost of the
	/// index (4 bytes per bucket, so 8 B/object at 2x slack) AND the thing that
	/// keeps linear-probe runs short. 8 B/object is cheap enough that trading
	/// it for short runs is not a close call.
	fn grow_for_one_more<P>(&mut self, slots: &[ArenaSlot<P>]) {
		let capacity = self.buckets.len();

		if capacity != 0 && (self.live + 1) * 2 <= capacity {
			return;
		}

		let wanted = if capacity == 0 { MIN_BUCKETS } else { capacity * 2 };
		self.rehash_into(slots, wanted);
	}

	/// Rebuilds the table at `capacity` buckets, which must be a power of two.
	fn rehash_into<P>(&mut self, slots: &[ArenaSlot<P>], capacity: usize) {
		debug_assert!(capacity.is_power_of_two());
		debug_assert!(self.live * 2 <= capacity);

		let old = core::mem::replace(&mut self.buckets, new_buckets(capacity));
		self.bucket_shift = 64 - capacity.trailing_zeros();

		let mask = capacity - 1;

		for slot in old.iter().copied() {
			if slot == NIL {
				continue;
			}

			let key = slots[slot as usize].key;
			let mut b = self.home(key);

			while self.buckets[b] != NIL {
				b = (b + 1) & mask;
			}

			self.buckets[b] = slot;
		}
	}

	/// Sizes the table so `additional` more keys fit without a rehash.
	pub fn reserve<P>(&mut self, slots: &[ArenaSlot<P>], additional: usize) {
		let wanted = (self.live + additional)
			.saturating_mul(2)
			.max(MIN_BUCKETS)
			.next_power_of_two();

		if wanted > self.buckets.len() {
			self.rehash_into(slots, wanted);
		}
	}

	/// Empties the table, KEEPING its capacity, exactly as `HashMap::clear`
	/// does, so a cleared-and-refilled structure does not pay the doubling
	/// ladder twice.
	pub fn clear(&mut self) {
		self.buckets.fill(NIL);
		self.live = 0;
	}
}
