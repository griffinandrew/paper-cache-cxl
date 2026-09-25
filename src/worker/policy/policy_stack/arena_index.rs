/*
 * Copyright (c) Kia Shakiba
 *
 * This source code is licensed under the GNU AGPLv3 license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! The arena's slab node, the chunked slab that holds it, and the KEYLESS
//! index that finds it.
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
//! # The slab is chunked
//!
//! A slot id is a plain `u32` into a [`ChunkedSlab`]: a `Vec` of fixed-size
//! chunks indexed as one flat array. This is the merged store's chunked slab
//! (`merged_store.rs`, "The chunked slab") made generic over its element. Id
//! `i` is element `i & (CHUNK - 1)` of chunk `i >> CHUNK_BITS`. To grow, the
//! slab appends one chunk. Nothing is copied and no slot moves, so ids, links
//! and payloads are what they were.
//!
//! It replaced a plain doubling `Vec`, which cost the split stacks twice:
//!
//! - **Memory.** A doubling leaves up to half the capacity empty, so the 32-byte
//!   node cost 32-64 B/object across a growth cycle.
//! - **A stall.** Each doubling copied the whole slab on the policy worker. At
//!   64M objects that is a 2 GiB copy, with 6 GiB live while it runs.
//!
//! A chunk is 128 KiB ([`SLAB_CHUNK_BYTES`]), 4096 of the 32-byte node. The
//! slab now costs 32 B/object at every population, plus at most one partly
//! filled chunk. The keyless index below still doubles, which is the part of
//! the stack's per-object cost that still swings (8-16 B/object).
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

// Under `eviction_stacks_pmem` everything this module allocates goes through
// the crate-wide `Hybrid` allocator (the far CXL/PMEM node), exactly as
// `CompactQueueSet` and `CompactFrequencyChain` do. That is the slab's chunks
// and the table that holds them, the index's buckets, and the owners' free
// lists.
//
// Not optional: `get_hybrid_dram_shared_overhead` drops the eviction-stack term
// to ZERO under that feature, on the premise the stack is not in DRAM. A slab
// that ignored these gates would sit in DRAM and be charged nothing -- silently
// wrong rather than merely unoptimised, and it would invalidate every
// far-memory placement experiment run against this tree.
#[cfg(not(feature = "eviction_stacks_pmem"))]
type Chunk<T> = Vec<T>;
#[cfg(feature = "eviction_stacks_pmem")]
type Chunk<T> = Vec<T, crate::Hybrid>;

#[cfg(not(feature = "eviction_stacks_pmem"))]
type ChunkTable<T> = Vec<Chunk<T>>;
#[cfg(feature = "eviction_stacks_pmem")]
type ChunkTable<T> = Vec<Chunk<T>, crate::Hybrid>;

#[cfg(not(feature = "eviction_stacks_pmem"))]
pub type U32Vec = Vec<u32>;
#[cfg(feature = "eviction_stacks_pmem")]
pub type U32Vec = Vec<u32, crate::Hybrid>;

/// A chunk with room for exactly `slots` elements, allocated whole.
#[cfg(not(feature = "eviction_stacks_pmem"))]
fn new_chunk<T>(slots: usize) -> Chunk<T> {
	Vec::with_capacity(slots)
}

#[cfg(feature = "eviction_stacks_pmem")]
fn new_chunk<T>(slots: usize) -> Chunk<T> {
	Vec::with_capacity_in(slots, crate::Hybrid)
}

#[cfg(not(feature = "eviction_stacks_pmem"))]
fn new_chunk_table<T>() -> ChunkTable<T> {
	Vec::new()
}

#[cfg(feature = "eviction_stacks_pmem")]
fn new_chunk_table<T>() -> ChunkTable<T> {
	Vec::new_in(crate::Hybrid)
}

/// Bytes one slab chunk is sized to. 128 KiB is a jemalloc LARGE size class,
/// so a chunk of the 32-byte node is one allocation with nothing rounded on
/// top.
pub const SLAB_CHUNK_BYTES: usize = 128 * 1024;

/// log2 of the elements in one chunk of `size`-byte elements: the largest
/// power of two whose chunk fits in [`SLAB_CHUNK_BYTES`]. A power of two, so
/// an id splits into its chunk and its offset with a shift and a mask.
const fn chunk_bits(size: usize) -> u32 {
	assert!(
		size != 0 && size <= SLAB_CHUNK_BYTES,
		"a slab element must be between one byte and one chunk",
	);

	(SLAB_CHUNK_BYTES / size).ilog2()
}

/// The arena's slab: one [`ArenaSlot`] per slot id.
pub type SlotSlab<P> = ChunkedSlab<ArenaSlot<P>>;

/// A slab of `T`s addressed by id, stored as a `Vec` of fixed-size chunks.
///
/// Id `i` is element `i & (CHUNK - 1)` of chunk `i >> CHUNK_BITS`. Ids are
/// handed out in push order, as they were by the `Vec` this replaced, and
/// `Index`/`IndexMut` keep every call site reading `slots[i]`. Growth appends
/// one chunk, allocated whole. No element is ever copied or moved, so an
/// element's address holds until the slab is cleared or dropped.
///
/// `CHUNK` depends on the element: the largest power of two whose chunk fits
/// in 128 KiB. All three chunk sizes below are jemalloc size classes.
///
/// ```text
///   element                                        size    CHUNK    chunk
///   the node every hybrid stack carries            32 B     4096  128 KiB
///   ArenaSlot<()>, the faithful S3-FIFO's ghosts   16 B     8192  128 KiB
///   a slot with an 8-byte payload (tests)          24 B     4096   96 KiB
/// ```
///
/// The slab does what the `Vec` did for its owners, and only that: `len`,
/// `push`, indexing, `capacity`, `reserve` and `clear`. The owners never
/// popped, truncated or `swap_remove`d; the old `Vec` lost elements only in
/// `clear`, and so does the slab. A freed slot's id goes on its owner's free
/// list, and the slot is overwritten in place when the id is reused.
///
/// Two more `Vec` semantics are kept:
///
/// - `clear` keeps the chunks, as `Vec::clear` kept the capacity, so a wiped
///   stack refills without allocating.
/// - An id at or past `len` panics, as it did past the `Vec`'s length, even
///   when it falls inside a committed chunk.
pub struct ChunkedSlab<T> {
	/// Every committed chunk, in id order. Chunks below `len >> CHUNK_BITS`
	/// are full, the next holds the rest of `len`, and any after that are
	/// empty: kept by `clear`, and refilled in order.
	chunks: ChunkTable<T>,

	/// Elements pushed since the last `clear`: ids `0 .. len` are live.
	len: usize,
}

impl<T> ChunkedSlab<T> {
	/// log2 of [`Self::CHUNK`].
	pub const CHUNK_BITS: u32 = chunk_bits(core::mem::size_of::<T>());

	/// Elements per chunk.
	pub const CHUNK: usize = 1 << Self::CHUNK_BITS;

	const OFFSET_MASK: usize = Self::CHUNK - 1;

	/// An empty slab. Allocates nothing: the first push commits the first
	/// chunk, so a stack built for a large budget still starts at zero.
	pub fn new() -> Self {
		ChunkedSlab {
			chunks: new_chunk_table(),
			len: 0,
		}
	}

	/// Elements pushed, which is also the id the next push gets.
	#[inline]
	pub fn len(&self) -> usize {
		self.len
	}

	/// Elements the committed chunks hold. A chunk is committed whole, so
	/// this is what the slab costs: `capacity() x size_of::<T>()` bytes, plus
	/// 24 bytes of chunk table per chunk.
	pub fn capacity(&self) -> usize {
		self.chunks.len() << Self::CHUNK_BITS
	}

	/// Appends `value` at id `len()`, committing one more chunk when every
	/// committed one is full.
	#[inline]
	pub fn push(&mut self, value: T) {
		let chunk = self.len >> Self::CHUNK_BITS;

		if chunk == self.chunks.len() {
			self.chunks.push(new_chunk(Self::CHUNK));
		}

		let slots = &mut self.chunks[chunk];

		// Pushed only below `CHUNK` into a chunk allocated for `CHUNK`, so
		// this push never reallocates. That is the point of the type.
		debug_assert!(slots.len() < Self::CHUNK && slots.capacity() >= Self::CHUNK);

		slots.push(value);
		self.len += 1;
	}

	/// Commits whole chunks until `additional` more elements fit without
	/// allocating.
	pub fn reserve(&mut self, additional: usize) {
		let wanted = self.len.checked_add(additional).expect("capacity overflow");
		let chunks = wanted.div_ceil(Self::CHUNK);

		while self.chunks.len() < chunks {
			self.chunks.push(new_chunk(Self::CHUNK));
		}
	}

	/// Empties the slab and KEEPS its chunks, as `Vec::clear` keeps its
	/// capacity, so a cleared-and-refilled owner does not pay for its chunks
	/// twice.
	pub fn clear(&mut self) {
		for chunk in self.chunks.iter_mut() {
			chunk.clear();
		}

		self.len = 0;
	}
}

/// For the allocator tests: the bytes the chunk table itself asks for.
#[cfg(test)]
impl<T> ChunkedSlab<T> {
	fn table_bytes(&self) -> usize {
		self.chunks.capacity() * core::mem::size_of::<Chunk<T>>()
	}
}

impl<T> core::ops::Index<usize> for ChunkedSlab<T> {
	type Output = T;

	#[inline]
	fn index(&self, id: usize) -> &T {
		&self.chunks[id >> Self::CHUNK_BITS][id & Self::OFFSET_MASK]
	}
}

impl<T> core::ops::IndexMut<usize> for ChunkedSlab<T> {
	#[inline]
	fn index_mut(&mut self, id: usize) -> &mut T {
		&mut self.chunks[id >> Self::CHUNK_BITS][id & Self::OFFSET_MASK]
	}
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
	pub fn get<P>(&self, slots: &SlotSlab<P>, key: HashedKey) -> u32 {
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
	fn bucket_of<P>(&self, slots: &SlotSlab<P>, key: HashedKey) -> Option<usize> {
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
	pub fn insert<P>(&mut self, slots: &SlotSlab<P>, slot: u32) {
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
	pub fn remove<P>(&mut self, slots: &SlotSlab<P>, key: HashedKey) -> Option<u32> {
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
	fn erase_at<P>(&mut self, slots: &SlotSlab<P>, bucket: usize) {
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
	fn grow_for_one_more<P>(&mut self, slots: &SlotSlab<P>) {
		let capacity = self.buckets.len();

		if capacity != 0 && (self.live + 1) * 2 <= capacity {
			return;
		}

		let wanted = if capacity == 0 { MIN_BUCKETS } else { capacity * 2 };
		self.rehash_into(slots, wanted);
	}

	/// Rebuilds the table at `capacity` buckets, which must be a power of two.
	fn rehash_into<P>(&mut self, slots: &SlotSlab<P>, capacity: usize) {
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
	pub fn reserve<P>(&mut self, slots: &SlotSlab<P>, additional: usize) {
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

#[cfg(test)]
mod tests {
	use super::*;
	use crate::worker::policy::policy_stack::{Tier, arena_queue_set::NodePayload};

	type Node = ArenaSlot<NodePayload>;

	/// The node every hybrid stack carries, made distinct per id so a read
	/// from the wrong id cannot pass for the right one.
	fn node(i: usize) -> Node {
		ArenaSlot {
			key: (i as u64).wrapping_mul(GOLDEN),
			prev: i as u32,
			next: !(i as u32),
			payload: NodePayload {
				size: i as u32,
				freq: 0,
				ts: 0,
				queue: 0,
				tier: Some(Tier::Fast),
				phys: Some(Tier::Fast),
				dram_resident: 0,
			},
		}
	}

	fn slot24(i: usize) -> ArenaSlot<u64> {
		ArenaSlot { key: i as u64, prev: NIL, next: NIL, payload: i as u64 }
	}

	// -------------------------------------------------------------------
	// Shape
	// -------------------------------------------------------------------

	/// A chunk is the largest power of two of the element that fits in
	/// 128 KiB, and for every element this crate puts in a slab it is a whole
	/// jemalloc size class, so the chunk costs exactly what it holds.
	#[test]
	fn a_chunk_is_128_kib_of_the_node_and_a_whole_jemalloc_size_class() {
		assert_eq!(core::mem::size_of::<Node>(), 32);
		assert_eq!(SlotSlab::<NodePayload>::CHUNK, 4096, "4096 x 32 B = 128 KiB");
		assert_eq!(SlotSlab::<()>::CHUNK, 8192, "the 16-byte ghost slot: 8192 x 16 B = 128 KiB");
		assert_eq!(SlotSlab::<u64>::CHUNK, 4096, "a 24-byte slot: 4096 x 24 B = 96 KiB");

		for (what, bytes) in [
			("the 32-byte node", SlotSlab::<NodePayload>::CHUNK * core::mem::size_of::<Node>()),
			("the 16-byte ghost slot", SlotSlab::<()>::CHUNK * core::mem::size_of::<ArenaSlot<()>>()),
			("a 24-byte slot", SlotSlab::<u64>::CHUNK * core::mem::size_of::<ArenaSlot<u64>>()),
		] {
			assert!(
				bytes <= SLAB_CHUNK_BYTES && bytes * 2 > SLAB_CHUNK_BYTES,
				"{what}: a {bytes}-byte chunk is not the largest power of two under 128 KiB",
			);

			// SAFETY: a pure size-class computation on a non-zero size.
			let class = unsafe { tikv_jemalloc_sys::nallocx(bytes, 0) };
			assert_eq!(class, bytes, "{what}: jemalloc rounds a {bytes}-byte chunk to {class}");
		}

		assert_eq!(SlotSlab::<NodePayload>::CHUNK * core::mem::size_of::<Node>(), SLAB_CHUNK_BYTES);
	}

	// -------------------------------------------------------------------
	// Behaviour: what the `Vec` did, kept (these pass on the `Vec` too)
	// -------------------------------------------------------------------

	/// Ids are the push order, across every chunk boundary, and a write
	/// through an id lands on that id and nowhere else.
	#[test]
	fn every_id_reads_back_what_was_pushed_across_chunk_boundaries() {
		let chunk = SlotSlab::<NodePayload>::CHUNK;
		let n = 3 * chunk + 5;
		let mut slab = SlotSlab::<NodePayload>::new();

		for i in 0..n {
			assert_eq!(slab.len(), i, "len before push {i}");
			slab.push(node(i));
		}

		assert_eq!(slab.len(), n);

		for i in 0..n {
			let want = node(i);
			let got = slab[i];
			assert_eq!(
				(got.key, got.prev, got.next, got.payload),
				(want.key, want.prev, want.next, want.payload),
				"id {i} read back something else",
			);
		}

		let written = [0, chunk - 1, chunk, 2 * chunk + 1, n - 1];
		for &i in &written {
			slab[i].payload.freq = 7;
		}

		for i in 0..n {
			let want = if written.contains(&i) { 7 } else { 0 };
			assert_eq!(slab[i].payload.freq, want, "a write through an id landed at {i}");
		}
	}

	/// An id at or past `len` panics, as it did past the `Vec`'s length --
	/// inside a committed chunk as well, and after a clear, when every chunk
	/// is still committed. A `Vec` semantic, so this passes on the `Vec` too.
	#[test]
	fn an_id_at_or_past_len_panics_even_inside_a_committed_chunk() {
		let mut slab = SlotSlab::<u64>::new();
		for i in 0..10 {
			slab.push(slot24(i));
		}

		assert!(slab.capacity() > 10, "id 10 must be inside the committed capacity");
		assert_eq!(slab[9].payload, 9);

		let past = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| slab[10].payload));
		assert!(past.is_err(), "id 10 of a 10-slot slab read {past:?}");

		slab.clear();

		let cleared = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| slab[0].payload));
		assert!(cleared.is_err(), "id 0 of a cleared slab read {cleared:?}");
	}

	/// `clear` keeps the chunks, as `Vec::clear` kept the capacity, and the
	/// refill takes the same ids from 0 again. A `Vec` semantic, so this passes
	/// on the `Vec` too.
	#[test]
	fn clear_keeps_the_chunks_as_vec_clear_kept_its_capacity() {
		let chunk = SlotSlab::<NodePayload>::CHUNK;
		let n = 2 * chunk + 3;
		let mut slab = SlotSlab::<NodePayload>::new();

		for i in 0..n {
			slab.push(node(i));
		}

		let capacity = slab.capacity();

		slab.clear();
		assert_eq!(slab.len(), 0);
		assert_eq!(slab.capacity(), capacity, "clear gave chunks back");

		for i in 0..n {
			slab.push(node(i + 1_000_000));
		}

		assert_eq!(slab.len(), n);
		assert_eq!(slab.capacity(), capacity, "the refill committed chunks the slab already had");

		for i in 0..n {
			assert_eq!(slab[i].key, node(i + 1_000_000).key, "id {i} after the refill");
		}
	}

	// -------------------------------------------------------------------
	// Growth: what the `Vec` did not do
	// -------------------------------------------------------------------

	/// Growth appends ONE chunk, when the committed ones are full and not
	/// before, and never moves a slot: a chunk's address holds for the slab's
	/// life. A doubling `Vec` fails both, the capacity by the doubling and the
	/// addresses whenever its reallocation moves.
	#[test]
	fn growth_appends_one_chunk_and_moves_no_slot() {
		let chunk = SlotSlab::<NodePayload>::CHUNK;
		let mut slab = SlotSlab::<NodePayload>::new();
		assert_eq!(slab.capacity(), 0, "an empty slab commits nothing");

		let mut firsts: Vec<(usize, *const Node)> = Vec::new();

		for i in 0..8 * chunk + 1 {
			slab.push(node(i));

			assert_eq!(
				slab.capacity(),
				(i + 1).div_ceil(chunk) * chunk,
				"{} slots are in {} whole chunks and no more",
				i + 1,
				(i + 1).div_ceil(chunk),
			);

			if i % chunk == 0 {
				firsts.push((i, &slab[i] as *const Node));
			}
		}

		assert_eq!(firsts.len(), 9);

		for (i, at) in firsts {
			assert!(core::ptr::eq(at, &slab[i]), "slot {i} moved when the slab grew");
			assert_eq!(slab[i].key, node(i).key);
		}
	}

	/// `reserve` commits whole chunks, enough for `len + additional`, and
	/// nothing when they already fit.
	#[test]
	fn reserve_commits_whole_chunks_and_no_more() {
		let chunk = SlotSlab::<NodePayload>::CHUNK;
		let mut slab = SlotSlab::<NodePayload>::new();

		slab.reserve(0);
		assert_eq!(slab.capacity(), 0);

		slab.reserve(1);
		assert_eq!(slab.capacity(), chunk);

		slab.reserve(chunk);
		assert_eq!(slab.capacity(), chunk, "{chunk} more fit in the one chunk");

		slab.reserve(chunk + 1);
		assert_eq!(slab.capacity(), 2 * chunk);

		for i in 0..5 {
			slab.push(node(i));
		}

		slab.reserve(2 * chunk - 5);
		assert_eq!(slab.capacity(), 2 * chunk, "5 + {} fit in two chunks", 2 * chunk - 5);

		slab.reserve(2 * chunk - 4);
		assert_eq!(slab.capacity(), 3 * chunk);
		assert_eq!(slab.len(), 5, "reserve pushed nothing");
	}

	// -------------------------------------------------------------------
	// What the allocator sees
	// -------------------------------------------------------------------

	/// This thread's cumulative jemalloc counter `name`, in usable
	/// (size-class) bytes: `thread.allocated` or `thread.deallocated`.
	///
	/// Per thread rather than `stats.allocated`, for the reason
	/// `object::overhead`'s allocator test gives: the process-wide figure
	/// takes in the allocations and frees of every test running beside this
	/// one, and per-thread counters make the delta exact without
	/// `--test-threads=1`. They count the same usable bytes.
	fn thread_counter(name: &core::ffi::CStr) -> u64 {
		// SAFETY: reads one u64 statistic into a u64 of the size passed.
		unsafe {
			let mut v: u64 = 0;
			let mut len = core::mem::size_of::<u64>();

			let rc = tikv_jemalloc_sys::mallctl(
				name.as_ptr(),
				&mut v as *mut u64 as *mut core::ffi::c_void,
				&mut len,
				core::ptr::null_mut(),
				0,
			);

			assert_eq!(rc, 0, "{name:?} unavailable");
			v
		}
	}

	/// (allocated, deallocated) on this thread so far.
	fn counters() -> (u64, u64) {
		(thread_counter(c"thread.allocated"), thread_counter(c"thread.deallocated"))
	}

	/// What jemalloc hands out for a request of `bytes`.
	fn class(bytes: usize) -> u64 {
		match bytes {
			0 => 0,
			// SAFETY: a pure size-class computation on a non-zero size.
			n => unsafe { tikv_jemalloc_sys::nallocx(n, 0) as u64 },
		}
	}

	/// Pushing `n` slots into an empty slab allocates `ceil(n / CHUNK)`
	/// chunks of exactly 128 KiB and the table that holds them, and nothing
	/// else. A doubling `Vec` allocates its power-of-two capacity instead:
	/// 1 MiB for 16,385 nodes where this is five chunks, 640 KiB.
	#[test]
	fn pushing_n_slots_allocates_n_over_chunk_rounded_up_chunks() {
		let chunk = SlotSlab::<NodePayload>::CHUNK;
		let chunk_bytes = (chunk * core::mem::size_of::<Node>()) as u64;

		for n in [1, chunk - 1, chunk, chunk + 1, 3 * chunk, 4 * chunk + 1, 6 * chunk + 17] {
			let (a0, d0) = counters();

			let mut slab = SlotSlab::<NodePayload>::new();
			for i in 0..n {
				slab.push(node(i));
			}

			let (a1, d1) = counters();
			let live = (a1 - a0) - (d1 - d0);

			let chunks = n.div_ceil(chunk) as u64;
			let table = class(slab.table_bytes());

			assert_eq!(
				live,
				chunks * chunk_bytes + table,
				"{n} slots left {live} bytes allocated: want {chunks} chunks of {chunk_bytes} \
				 bytes and a {table}-byte chunk table",
			);

			drop(slab);
		}
	}

	/// One growth step -- a push into a slab whose chunks are all full --
	/// allocates one chunk and frees nothing but the chunk table's old buffer
	/// when the table itself grows, so no slot is copied. Taken at four full
	/// chunks, 2^14 slots, where a doubling `Vec` doubles: it allocates 1 MiB
	/// there and frees the 512 KiB it copied out of. And at six, where the
	/// table does not grow.
	#[test]
	fn one_growth_step_allocates_one_chunk_and_frees_nothing() {
		let chunk = SlotSlab::<NodePayload>::CHUNK;
		let chunk_bytes = (chunk * core::mem::size_of::<Node>()) as u64;

		for full in [4usize, 6] {
			let mut slab = SlotSlab::<NodePayload>::new();
			for i in 0..full * chunk {
				slab.push(node(i));
			}

			let table_before = class(slab.table_bytes());
			let (a0, d0) = counters();

			slab.push(node(full * chunk));

			let (a1, d1) = counters();
			let table_after = class(slab.table_bytes());

			let (table_new, table_old) = match table_after == table_before {
				true => (0, 0),
				false => (table_after, table_before),
			};

			assert_eq!(
				d1 - d0,
				table_old,
				"the step past {full} full chunks freed {} bytes: slots were copied",
				d1 - d0,
			);
			assert_eq!(
				a1 - a0,
				chunk_bytes + table_new,
				"the step past {full} full chunks allocated {} bytes, not one {chunk_bytes}-byte \
				 chunk (and a {table_new}-byte table)",
				a1 - a0,
			);
			assert_eq!(slab.capacity(), (full + 1) * chunk);
		}
	}

	/// A cleared slab refills to its old length without allocating or
	/// freeing a byte, as a cleared `Vec` did, because it kept its chunks.
	#[test]
	fn a_cleared_slab_refills_to_its_old_length_without_allocating() {
		let chunk = SlotSlab::<NodePayload>::CHUNK;
		let n = 3 * chunk + 1;
		let mut slab = SlotSlab::<NodePayload>::new();

		for i in 0..n {
			slab.push(node(i));
		}

		slab.clear();

		let (a0, d0) = counters();
		for i in 0..n {
			slab.push(node(i));
		}
		let (a1, d1) = counters();

		assert_eq!(
			(a1 - a0, d1 - d0),
			(0, 0),
			"refilling a cleared slab allocated and freed bytes",
		);
		assert_eq!(slab.len(), n);
	}
}
