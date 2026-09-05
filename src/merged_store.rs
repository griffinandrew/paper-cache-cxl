/*
 * Copyright (c) Kia Shakiba
 *
 * This source code is licensed under the GNU AGPLv3 license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! `MergedStore` -- object map, recency order and tier placement in ONE
//! structure.
//!
//! A fourth `ObjectMapRef` shape behind `merged_object_store`. A FEATURE and
//! not a `PaperPolicy` variant because the object store is a compile-time
//! choice here: `ObjectStore::get_ref` returns `impl Deref`, so the trait is
//! not object-safe and `objects` cannot be `dyn`.
//!
//! # Why
//!
//! Measured with `stats.allocated`, R^2 = 1.000000, a tiered LRU object costs
//! three independently-keyed pieces:
//!
//! ```text
//!   DashMap<HashedKey, Object>   96 B   key -> object
//!   Arc<TieredBuffer>            48 B   refcount + buffer handle
//!   LruCompactHybridStack        72 B   key -> slot -> links + payload
//! ```
//!
//! Most of that 72 is two more copies of the key -- one in the stack's slab,
//! one in its index -- present only to answer "where in the recency order is
//! this key, and which tier is it in?". Merging answers both for free: finding
//! the object IS finding its position and its tier.
//!
//! # The index is chained, not a second HashMap
//!
//! The first sharded revision kept a `HashMap<HashedKey, u32, NoHasher>` per
//! shard and measured 251.8 B/object against a DashMap control's 179.1 -- it
//! removed a 72 B/object eviction stack and added 72.7, which is not a saving.
//! The reason was structural: a `HashMap` index does not remove a hash table,
//! it MOVES one. DashMap stores the object in the bucket (41 B/bucket); that
//! design stored a `u32` in a hashbrown bucket (17 B) AND the object in a slab
//! slot, so it paid for both.
//!
//! So the index is now CacheLib's `ChainedHashTable`: a flat `Vec<u32>` of slot
//! ids, one per bucket, with the chain threaded through a `hash_next` field in
//! the slot itself -- CacheLib's `hashHook_`. 4 bytes per bucket instead of 17,
//! no second copy of the key, and no second allocation to keep sized.
//!
//! Chaining rather than open addressing is what makes this composable at all:
//! the chain link lives in a slot that already exists, so the table's marginal
//! cost is the bucket array alone. Load factor 1.0 -- mean chain length 1 --
//! rather than hashbrown's 7/8, because a chain does not degrade at high load
//! the way a probe sequence does.
//!
//! # Sharding, and memcached's two tricks
//!
//! The first revision used one `RwLock` over the whole store, on the reasoning
//! that sharding forces per-shard recency lists and so approximate order. That
//! measured +3.6% GET and +25.4% SET against DashMap on standard_web, and the
//! reasoning was only half right.
//!
//! memcached does not solve this with better locking -- it avoids the work.
//! `item_lock(hv)` is an array of mutexes indexed by hash, and
//! `ITEM_UPDATE_INTERVAL` makes `item_update` early-return unless the item is
//! older than the interval, so a key hit a million times takes the LRU lock
//! ONCE. The lock is rare because it is skipped, not because it is cheap.
//!
//! Both are here. `SHARDS` independent `{index, slab, list}` regions, and
//! `MERGED_UPDATE_INTERVAL` counted in ACCESSES rather than seconds --
//! memcached's 60 s is calibrated to wall clock, and a trace replaying 300M
//! records as fast as it can would barely reorder at all under it. The default
//! is 0, exact behaviour, so the effect is measured rather than assumed.
//!
//! ## Sharding on the HIGH bits
//!
//! The key IS the hash -- the store is `NoHasher`d -- and a bucket is selected
//! with the LOW bits of it. Sharding on the low bits too would hold them
//! constant inside a shard and drive every key in it to one bucket. Shard on
//! the high bits, which is what DashMap does and why DashMap and `NoHasher`
//! compose in the first place.
//!
//! ## Sharding without losing the order
//!
//! Each shard's list is recency-ordered within itself, so the globally
//! least-recently-used object is the MINIMUM over the shard tails. Every slot
//! carries the access counter it was last relinked at, and each shard mirrors
//! its tail's counter in a padded `AtomicU64` -- so choosing an eviction victim
//! is `SHARDS` atomic loads and then ONE shard lock, never a global lock.
//!
//! At interval 0 that is exact LRU across shards. Above 0 the order is
//! quantised by the interval, which is exactly the trade memcached makes and
//! the only approximation here.
//!
//! # Tiering
//!
//! Ported from [`LruCompactHybridStack`], whose structure this can reproduce
//! exactly: ONE recency list plus a `fast_boundary` cursor at the
//! least-recently-used FAST slot. Everything from the head up to and including
//! the boundary is fast; everything after it is slow. So a demotion is a
//! one-step walk of the cursor and never touches the list, which is why tier
//! placement here can never reorder and never evict.
//!
//! ## The tier boundary is global, by the same trick eviction uses
//!
//! An earlier revision split `fast_capacity` EVENLY across the shards and let
//! each settle its own boundary. That is not a global order: a shard holding
//! large recent objects demoted things MORE RECENT than fast objects idling in
//! another shard, and the fast tier went under-used whenever recency skewed
//! across shards.
//!
//! So the boundary is chosen the way the eviction victim is. Each shard
//! mirrors the stamp of its `fast_boundary` slot into a second padded array,
//! `fast_tails`, and the store keeps one `fast_used` total. Settling runs with
//! NO shard lock held: while the total is over the high watermark, `SHARDS`
//! relaxed loads name the shard whose boundary is oldest, that ONE shard's
//! write lock is taken, its boundary steps back one slot, the mirror is
//! republished and the lock released -- repeat until under the low watermark.
//!
//! That is exact, not approximate: every shard's fast set is a contiguous MRU
//! PREFIX of its own list, so the oldest boundary across shards IS the globally
//! least-recently-used fast object. It cannot deadlock, because a toucher
//! releases its own shard before settling and no thread ever holds two shard
//! locks. Concurrent settlers may each demote one extra object, which the
//! 0.98/0.95 hysteresis absorbs.
//!
//! Migrations accumulate per shard and `drain_tier_migrations` concatenates
//! them, so `PolicyWorker::apply_tier_migrations` performs the physical
//! `Object::set_data` moves exactly as it does for every other hybrid stack.
//!
//! # Slot recycling
//!
//! A freed slot goes on `free` and is reused whole. The `generation` counter an
//! earlier revision carried is gone: it guarded against a stale `u32` outliving
//! its slot, and with the index chained there is no `u32` handed out to anyone
//! -- `MergedRef` holds one only for as long as it holds the shard guard, which
//! is exactly as long as the slot cannot be recycled.
//!
//! # Nothing under the shard lock is O(n)
//!
//! Two paths used to be, and both stalled the API THREAD, since in this design
//! the API thread is what inserts:
//!
//!   * the slab was one `Vec<Slot>` per shard, so a growth `realloc`ed and
//!     copied every live slot;
//!   * `grow_buckets` doubled the index and re-threaded every chain by walking
//!     the WHOLE recency list.
//!
//! The slab is now a `Vec` of fixed 4096-slot CHUNKS. Growth appends a chunk;
//! nothing is copied, no slot ever moves, and a slot id is just
//! `chunk << 12 | offset`. At 56 B a slot that is 224 KiB per chunk, so an
//! empty 32-shard store costs ~7 MiB rather than the 112 MiB a 64K chunk would.
//!
//! The index grows by LINEAR HASHING: a `split` cursor and two masks, one
//! bucket rehashed per insert past the load factor. The cost of a growth step
//! is one bucket's chain -- mean length 1 -- instead of the entire list, so
//! there is no stall to move off the API thread in the first place.

use std::{
	collections::HashMap,
	ops::{Deref, DerefMut},
	sync::{
		atomic::{AtomicU64, AtomicUsize, Ordering},
		RwLock, RwLockReadGuard, RwLockWriteGuard,
	},
};

use crate::{
	object::{Object, ObjectSize},
	worker::Tier,
	CacheSize, HashedKey, NoHasher,
};

const NIL: u32 = u32::MAX;

/// 32 shards on an 8-core box matches DashMap's own `4 * ncpus` default, so
/// the comparison against it is like-for-like.
const SHARD_BITS: u32 = 5;
const SHARDS: usize = 1 << SHARD_BITS;

/// Sharded on the HIGH bits -- see the module doc. The store is `NoHasher`d, so
/// low-bit sharding would collapse each shard onto one hashbrown bucket.
#[inline]
fn shard_of(key: HashedKey) -> usize {
	(key >> (64 - SHARD_BITS)) as usize
}

/// A shard tail's access counter, on its own cache line.
///
/// Unpadded these 32 counters share four lines, and every relink of any shard's
/// tail would invalidate the line under seven other shards -- false sharing on
/// precisely the value that exists to be read without a lock.
#[repr(align(64))]
struct TailSeq(AtomicU64);

/// A shard with nothing in it, distinguishable from a real `last_access` of 0.
const EMPTY_TAIL: u64 = u64::MAX;

/// Buckets a fresh shard starts with. 32 shards x 16 x 4 B = 2 KB of baseline,
/// and the table doubles from there.
const INITIAL_BUCKETS: usize = 16;

/// Slots per slab chunk, and the width of a slot id's offset field.
///
/// The slab is a `Vec` of these chunks, so growth APPENDS one and nothing is
/// ever copied or reallocated -- the `Vec<Slot>` this replaced `realloc`ed
/// under the shard lock, on the API thread, which is the stall the chunking
/// exists to remove. A slot id is `chunk << SLAB_CHUNK_BITS | offset`, which
/// for a linearly allocated slab is just the slot's ordinal.
///
/// 4096 is picked from both ends. At 56 B a slot that is 224 KiB per chunk, so
/// 32 shards start at ~7 MiB of committed slab -- a 64K-slot chunk would make
/// an EMPTY store cost 112 MiB. And the worst-case waste is one partly-filled
/// chunk per shard, 4095 slots, bounded and independent of how large the cache
/// grows, where `Vec` doubling measured 1.40x the live count and 25% growth
/// steps 1.10x -- both proportional, both unbounded.
const SLAB_CHUNK_BITS: u32 = 12;
const SLAB_CHUNK: usize = 1 << SLAB_CHUNK_BITS;
const SLAB_OFFSET_MASK: usize = SLAB_CHUNK - 1;

/// Live entries per bucket before the table doubles. 1.0, not hashbrown's 7/8:
/// a chain degrades linearly with load where a probe sequence degrades sharply,
/// so chaining can be run full and spend the difference on memory instead.
const MAX_LOAD_NUMER: usize = 1;
const MAX_LOAD_DENOM: usize = 1;

struct Slot<K, V> {
	/// `Option` so `take` can move the object out without disturbing the slab.
	/// Free: the `NonNull` inside the object's `TieredValue` is the niche that
	/// absorbs the discriminant, so `Option<Object>` is the same size as
	/// `Object` -- 24 bytes. `object/mod.rs::layout` asserts it.
	object: Option<Object<K, V>>,
	/// The store is keyed by hash; `Object::key()` is the real key, which
	/// `key_matches` needs to make collisions safe.
	hashed: HashedKey,
	prev: u32,
	next: u32,
	/// Next slot in this key's hash-bucket chain, `NIL` at the end --
	/// CacheLib's `hashHook_`. Threading the chain through the slot is what
	/// lets the index cost 4 bytes a bucket instead of a whole hashbrown row.
	hash_next: u32,
	/// The store clock at the last relink, FULL WIDTH. Double duty: the
	/// update-interval check, and the cross-shard comparison that keeps
	/// eviction and the tier boundary picking the true LRU slot.
	///
	/// It was a truncated `u32` and compared as a wrapping difference, which
	/// is exact only while no live slot goes un-relinked for 2^32 accesses.
	/// That is four billion -- reachable by a long replay of a 306M-record
	/// trace, and silently wrong when reached. At 64 bits, one relaxed
	/// `fetch_add` per relink, a billion accesses a second would take 584
	/// years to wrap, so the stamps are simply ordered and compared as such.
	last_access: u64,
	tier: Tier,
}

/// 24 object + 8 hashed + 4 prev + 4 next + 4 hash_next + 8 last_access +
/// 1 tier = 53, padded to 56 by the object's 8-byte alignment.
///
/// The `<= 56` bound is the claim being made against the split design; the
/// EXACT assert is what catches a field silently landing in the padding and
/// then, later, pushing the slot over. `size` (u32) and `dram_resident` (u8)
/// used to sit here and are gone: both are now derived from the object, which
/// holds the value's length already -- see `Slot::migrating`.
const _: () = assert!(
	core::mem::size_of::<Slot<u64, std::sync::Arc<[u8]>>>() <= 56,
	"Slot grew past 56 bytes -- the whole point is that it is smaller than a \
	 DashMap row plus an eviction-stack row",
);

const _: () = assert!(
	core::mem::size_of::<Slot<u64, std::sync::Arc<[u8]>>>() == 56,
	"Slot is no longer exactly 56 bytes -- re-measure \
	 MERGED_STORE_STRUCTURE_OVERHEAD before changing this number",
);

impl<K, V> Slot<K, V> {
	/// A recycled or never-used slot. Linked nowhere, holding nothing.
	fn empty() -> Self {
		Slot {
			object: None,
			hashed: 0,
			prev: NIL,
			next: NIL,
			hash_next: NIL,
			last_access: 0,
			tier: Tier::Fast,
		}
	}

	/// Bytes that actually move between tiers. `Object::set_data` migrates the
	/// value buffer alone, so the key, the expiry field and the `Expiries` row
	/// stay in DRAM in either tier and must not be charged to a tier's budget.
	///
	/// DERIVED, not stored. It used to be `size - dram_resident`, two fields
	/// filled in by the worker one event after the API thread inserted the
	/// object -- so a freshly inserted object was accounted as ZERO bytes until
	/// the worker caught up. The object carries its value's length, and
	/// `size - dram_resident` was by construction `resident_value_bytes(len)`:
	/// `base_size` is `key + value + expiry (+ ttl)` and `dram_resident_size`
	/// is the same sum without the value. So this asks the allocator the same
	/// question `base_size` asks, and the two cannot drift apart.
	fn migrating(&self) -> CacheSize {
		match &self.object {
			Some(object) => {
				crate::object::overhead::resident_value_bytes(object.data_size()) as CacheSize
			},

			None => 0,
		}
	}
}

/// The chunked slab: a `Vec` of fixed-size chunks, indexed as one flat array.
///
/// `Index`/`IndexMut` on the slot id keep every call site reading like the
/// `Vec<Slot>` this replaced, which is the point -- the change is where the
/// memory comes from, not how slots are reached. Growth appends a chunk, so a
/// slot's address is stable for as long as the slot lives and no `realloc`
/// ever runs under the shard lock.
struct Slab<K, V> {
	chunks: Vec<Box<[Slot<K, V>; SLAB_CHUNK]>>,

	/// Slot ids ever handed out. Ids are allocated linearly, so this is both
	/// the next id and the high-water mark; recycled ids come off `Inner::free`
	/// and never move it.
	allocated: usize,
}

impl<K, V> Slab<K, V> {
	fn new() -> Self {
		Slab {
			chunks: Vec::new(),
			allocated: 0,
		}
	}

	/// Slots handed out so far.
	#[cfg(test)]
	fn len(&self) -> usize {
		self.allocated
	}

	/// Slots the committed chunks can hold. What `capacities()` reports as the
	/// slab's cost, since a chunk is committed whole.
	fn capacity(&self) -> usize {
		self.chunks.len() * SLAB_CHUNK
	}

	/// Takes the next id, appending a chunk when the last one is full.
	fn alloc(&mut self, fresh: Slot<K, V>) -> u32 {
		if self.allocated == self.capacity() {
			self.push_chunk();
		}

		let i = self.allocated as u32;
		self.allocated += 1;
		self[i as usize] = fresh;

		i
	}

	/// Built through a `Vec` rather than as an array literal: a
	/// `Box::new([Slot::empty(); N])` would materialise 224 KiB on the STACK
	/// first and only then move it to the heap, which is a stack overflow
	/// waiting for a thread with a small stack.
	fn push_chunk(&mut self) {
		let mut slots = Vec::with_capacity(SLAB_CHUNK);

		for _ in 0..SLAB_CHUNK {
			slots.push(Slot::empty());
		}

		let chunk: Box<[Slot<K, V>; SLAB_CHUNK]> = slots
			.into_boxed_slice()
			.try_into()
			.unwrap_or_else(|_| unreachable!("built with exactly SLAB_CHUNK slots"));

		self.chunks.push(chunk);
	}

	fn clear(&mut self) {
		self.chunks.clear();
		self.allocated = 0;
	}
}

impl<K, V> std::ops::Index<usize> for Slab<K, V> {
	type Output = Slot<K, V>;

	#[inline]
	fn index(&self, i: usize) -> &Slot<K, V> {
		debug_assert!(i < self.allocated, "slot id {i} was never handed out");

		&self.chunks[i >> SLAB_CHUNK_BITS][i & SLAB_OFFSET_MASK]
	}
}

impl<K, V> std::ops::IndexMut<usize> for Slab<K, V> {
	#[inline]
	fn index_mut(&mut self, i: usize) -> &mut Slot<K, V> {
		debug_assert!(i < self.allocated, "slot id {i} was never handed out");

		&mut self.chunks[i >> SLAB_CHUNK_BITS][i & SLAB_OFFSET_MASK]
	}
}

struct Inner<K, V> {
	/// One slot id per bucket, `NIL` when empty; the rest of the chain is in
	/// each slot's `hash_next`. `base + split` long -- NOT a power of two, see
	/// `bucket_of`.
	buckets: Vec<u32>,
	/// The power of two the low mask is taken against. `buckets.len()` sits in
	/// `[base, 2 * base)`.
	base: usize,
	/// The next bucket to split. Buckets below it have already been split this
	/// round and so are addressed with the WIDE mask.
	split: usize,
	/// Live entries, which `buckets.len()` is grown to keep up with. Not
	/// derivable from `slots.len()`, which counts recycled slots too.
	live: usize,
	slots: Slab<K, V>,
	free: Vec<u32>,

	/// Test-only: chain links walked. The linear-hashing claim -- an insert
	/// walks its own bucket's chain plus, at most, the one bucket it splits --
	/// is a cost claim, so it is asserted against a count rather than argued
	/// from the code shape.
	#[cfg(test)]
	walk: AtomicUsize,

	/// MRU end.
	head: u32,
	/// LRU end.
	tail: u32,

	/// The least-recently-used FAST slot: head..=fast_boundary is the fast
	/// tier, everything after it is slow. `NIL` when nothing is fast.
	fast_boundary: u32,

	fast_used: CacheSize,
	slow_used: CacheSize,
	fast_count: usize,

	migrations: Vec<(HashedKey, Tier)>,
}

impl<K, V> Inner<K, V> {
	fn new() -> Self {
		Inner {
			buckets: vec![NIL; INITIAL_BUCKETS],
			base: INITIAL_BUCKETS,
			split: 0,
			live: 0,
			slots: Slab::new(),
			free: Vec::new(),

			#[cfg(test)]
			walk: AtomicUsize::new(0),
			head: NIL,
			tail: NIL,
			fast_boundary: NIL,
			fast_used: 0,
			slow_used: 0,
			fast_count: 0,
			migrations: Vec::new(),
		}
	}

	/// The LOW bits pick the bucket; the shard already took the high ones, so
	/// the two selections are independent and every bucket stays reachable.
	///
	/// Two masks, because the table grows one bucket at a time: buckets below
	/// `split` have already been re-partitioned this round and answer to the
	/// WIDE mask, the rest still answer to the narrow one. Litwin's linear
	/// hashing, and the reason the table can be a length that is not a power
	/// of two.
	#[inline]
	fn bucket_of(&self, key: HashedKey) -> usize {
		let b = (key as usize) & (self.base - 1);

		match b < self.split {
			true => (key as usize) & (self.base * 2 - 1),
			false => b,
		}
	}

	/// Walk one bucket chain. Mean length 1 at load factor 1.0.
	#[inline]
	fn find(&self, key: HashedKey) -> Option<u32> {
		let mut i = self.buckets[self.bucket_of(key)];

		while i != NIL {
			#[cfg(test)]
			self.walk.fetch_add(1, Ordering::Relaxed);

			let slot = &self.slots[i as usize];

			if slot.hashed == key {
				return Some(i);
			}

			i = slot.hash_next;
		}

		None
	}

	/// Links on one bucket's chain, for the tests that bound what an insert is
	/// allowed to walk.
	#[cfg(test)]
	fn chain_len(&self, b: usize) -> usize {
		let mut i = self.buckets[b];
		let mut n = 0;

		while i != NIL {
			n += 1;
			i = self.slots[i as usize].hash_next;
		}

		n
	}

	/// Push onto the front of the key's chain, and split ONE bucket if that
	/// took the table past the load factor.
	///
	/// Front insertion is deliberate: a freshly inserted key is the one most
	/// likely to be looked up next, so it should be the first link walked.
	fn bucket_link(&mut self, i: u32) {
		let b = self.bucket_of(self.slots[i as usize].hashed);

		self.slots[i as usize].hash_next = self.buckets[b];
		self.buckets[b] = i;
		self.live += 1;

		if self.live * MAX_LOAD_DENOM > self.buckets.len() * MAX_LOAD_NUMER {
			self.split_one_bucket();
		}
	}

	/// Unlink by key, repairing the predecessor's `hash_next`.
	fn bucket_unlink(&mut self, key: HashedKey) -> Option<u32> {
		let b = self.bucket_of(key);
		let mut i = self.buckets[b];
		let mut prev = NIL;

		while i != NIL {
			let (hashed, next) = {
				let slot = &self.slots[i as usize];
				(slot.hashed, slot.hash_next)
			};

			if hashed == key {
				match prev {
					NIL => self.buckets[b] = next,
					prev => self.slots[prev as usize].hash_next = next,
				}

				self.slots[i as usize].hash_next = NIL;
				self.live -= 1;

				return Some(i);
			}

			prev = i;
			i = next;
		}

		None
	}

	/// Grow the table by ONE bucket, re-partitioning one chain.
	///
	/// This replaces a `grow_buckets` that doubled the table and re-threaded
	/// every chain by walking the whole recency list -- under the shard write
	/// lock, on the API thread, so a shard holding a million objects stalled
	/// every request to it for the length of a million-node pointer chase.
	/// Here the work is one bucket's chain, mean length 1, and it is paid by
	/// the insert that crossed the load factor.
	///
	/// Splitting bucket `split` under the wide mask sends each of its keys to
	/// either `split` or `split + base` and nowhere else, which is the whole
	/// trick: no other bucket's contents can be affected, so no other bucket
	/// has to be visited. Once every bucket of the round has been split the
	/// wide mask becomes the narrow one and the round starts again.
	///
	/// Recycled slots cannot be dragged in: only the chain is walked, and a
	/// recycled slot is off every chain -- `bucket_unlink` takes it off before
	/// it reaches the free list.
	fn split_one_bucket(&mut self) {
		let from = self.split;
		let wide = self.base * 2 - 1;

		// `push` rather than a resize: the table grows by one bucket, and the
		// `Vec`'s own amortised doubling copies a flat array of `u32`s, which
		// is a memcpy and not a walk of anything.
		self.buckets.push(NIL);

		let mut i = self.buckets[from];
		let mut stay = NIL;
		let mut moved = NIL;

		while i != NIL {
			#[cfg(test)]
			self.walk.fetch_add(1, Ordering::Relaxed);

			let (key, next) = {
				let slot = &self.slots[i as usize];
				(slot.hashed, slot.hash_next)
			};

			match (key as usize) & wide == from {
				true => {
					self.slots[i as usize].hash_next = stay;
					stay = i;
				},

				false => {
					self.slots[i as usize].hash_next = moved;
					moved = i;
				},
			}

			i = next;
		}

		self.buckets[from] = stay;
		self.buckets[from + self.base] = moved;

		self.split += 1;

		if self.split == self.base {
			self.split = 0;
			self.base *= 2;
		}
	}

	fn unlink(&mut self, i: u32) {
		let (p, n) = {
			let s = &self.slots[i as usize];
			(s.prev, s.next)
		};

		match p {
			NIL => self.head = n,
			p => self.slots[p as usize].next = n,
		}

		match n {
			NIL => self.tail = p,
			n => self.slots[n as usize].prev = p,
		}

		let s = &mut self.slots[i as usize];
		s.prev = NIL;
		s.next = NIL;
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
			old => self.slots[old as usize].prev = i,
		}

		self.head = i;
	}

	/// Reverses this slot's contribution to the tier accounting and steps the
	/// boundary back off it. Must run BEFORE `unlink`, which clears `prev`.
	fn detach_tier(&mut self, i: u32) {
		let (tier, migrating, prev) = {
			let s = &self.slots[i as usize];
			(s.tier, s.migrating(), s.prev)
		};

		// If the departing slot was the boundary, the new least-recently-used
		// fast slot is the one in front of it. When the boundary was also the
		// list tail -- every slot fast -- that is the new tail, which is the
		// same answer.
		if self.fast_boundary == i {
			self.fast_boundary = prev;
		}

		match tier {
			Tier::Fast => {
				self.fast_used = self.fast_used.saturating_sub(migrating);
				self.fast_count = self.fast_count.saturating_sub(1);
			},

			Tier::Slow => {
				self.slow_used = self.slow_used.saturating_sub(migrating);
			},
		}
	}

	/// Unlink, drop the object and return the slot to the free list.
	///
	/// Dropping the object is what RETIRES its value: `Object::drop` defers the
	/// free under an epoch pin rather than performing it, so a reader that
	/// lifted this value's pointer out from under the shard guard a moment ago
	/// and is still copying its bytes is safe. Nothing extra is needed here --
	/// and deliberately so, since this runs on the policy worker, `take` runs
	/// on the API thread, and the TTL reaper runs on a third; a per-site rule
	/// would have to be repeated at all of them.
	fn retire(&mut self, i: u32) {
		self.detach_tier(i);
		self.unlink(i);

		self.slots[i as usize].object = None;
		self.free.push(i);
	}

	/// Move to the MRU end and make fast, promoting from slow if needed.
	///
	/// Faithful port of `LruCompactHybridStack::touch_fast_key`, minus the
	/// settle: the tier boundary is now settled globally, with no shard lock
	/// held, so the caller drops this shard's guard and then calls
	/// `MergedStore::settle_tier`. A promotion that a tight budget immediately
	/// undoes therefore reports BOTH transitions, in order, rather than
	/// suppressing the first -- per-key order is preserved, so the consumer
	/// applies promote-then-demote and lands on the same final placement.
	fn touch_slot(&mut self, i: u32, now: u64) {
		let previous_tier = self.slots[i as usize].tier;
		let already_at_front = self.head == i;
		let is_boundary = self.fast_boundary == i;

		// Read the neighbour BEFORE moving: once the slot is at the front its
		// predecessor is gone, and the boundary has to step back to whatever
		// was in front of it.
		let new_boundary_if_moved = match is_boundary && !already_at_front {
			true => self.slots[i as usize].prev,
			false => NIL,
		};

		if !already_at_front {
			self.unlink(i);
			self.link_front(i);

			if is_boundary {
				self.fast_boundary = new_boundary_if_moved;
			}
		}

		self.slots[i as usize].last_access = now;

		if previous_tier != Tier::Fast {
			let migrating = self.slots[i as usize].migrating();

			self.slow_used = self.slow_used.saturating_sub(migrating);
			self.fast_used += migrating;
			self.fast_count += 1;
			self.slots[i as usize].tier = Tier::Fast;

			if self.fast_boundary == NIL {
				self.fast_boundary = i;
			}

			let key = self.slots[i as usize].hashed;
			self.migrations.push((key, Tier::Fast));
		}
	}

	/// Demotes exactly ONE slot -- the boundary, the least-recently-used fast
	/// slot in this shard -- and steps the boundary back off it.
	///
	/// Nothing is searched, and because the boundary only walks along a list
	/// this never reorders anything. Returns the bytes that left the fast tier,
	/// or `None` when the shard holds nothing fast.
	///
	/// One step per call, rather than a drain loop, because the loop now lives
	/// in `MergedStore::settle_tier` and re-chooses the shard after every step:
	/// the next victim is whichever shard's boundary is now oldest, which is
	/// what makes the demotion order global rather than per shard.
	fn demote_boundary(&mut self) -> Option<CacheSize> {
		let d = self.fast_boundary;

		if d == NIL {
			return None;
		}

		let (key, migrating, prev) = {
			let s = &self.slots[d as usize];
			(s.hashed, s.migrating(), s.prev)
		};

		self.slots[d as usize].tier = Tier::Slow;

		self.fast_used = self.fast_used.saturating_sub(migrating);
		self.fast_count = self.fast_count.saturating_sub(1);
		self.slow_used += migrating;
		self.fast_boundary = prev;

		self.migrations.push((key, Tier::Slow));

		Some(migrating)
	}

	fn tail_seq(&self) -> u64 {
		match self.tail {
			NIL => EMPTY_TAIL,
			t => self.slots[t as usize].last_access,
		}
	}

	/// The stamp of the least-recently-used FAST slot, for the second mirror.
	/// `EMPTY_TAIL` when the shard holds nothing fast, which reads as "never a
	/// candidate" in the settle loop's minimum.
	fn fast_tail_seq(&self) -> u64 {
		match self.fast_boundary {
			NIL => EMPTY_TAIL,
			b => self.slots[b as usize].last_access,
		}
	}
}

/// The tiering configuration the settle loop needs, resolved once so the loop
/// reads the store's atomics once rather than per demotion.
///
/// `per_shard_capacity` is GONE. It was `fast_capacity / SHARDS`, and settling
/// each shard against its own share is precisely what made the demotion order
/// per-shard rather than global: a shard whose recent objects are large demoted
/// objects more recent than fast objects idling in another shard, and the fast
/// tier went under-used whenever recency skewed across shards. The capacity is
/// now the whole store's, checked against one `fast_used` total.
#[derive(Clone, Copy)]
struct TierBudget {
	capacity: CacheSize,
	shared_overhead: CacheSize,
	high_ppm: u64,
	low_ppm: u64,
}

/// Watermark fractions are carried as parts-per-million so they fit an
/// `AtomicU64` without a float-atomic dance.
#[inline]
fn scale(bytes: CacheSize, ppm: u64) -> CacheSize {
	(bytes as f64 * (ppm as f64 / 1_000_000.0)) as CacheSize
}

const DEFAULT_HIGH_PPM: u64 = 980_000;
const DEFAULT_LOW_PPM: u64 = 950_000;

pub struct MergedStore<K, V> {
	shards: Box<[RwLock<Inner<K, V>>]>,

	/// Mirrors each shard's tail `last_access`, readable without that shard's
	/// lock so an eviction victim can be chosen with atomic loads alone.
	tails: Box<[TailSeq]>,

	/// The same trick for the TIER boundary: each shard mirrors the stamp of
	/// its `fast_boundary` slot, so the globally least-recently-used FAST
	/// object is found with `SHARDS` relaxed loads and no lock at all.
	///
	/// Exact, not approximate: a shard's fast set is a contiguous MRU prefix
	/// of its own list, so the oldest boundary across shards is the global LRU
	/// fast object.
	fast_tails: Box<[TailSeq]>,

	clock: AtomicU64,
	tracked: AtomicUsize,

	/// `sum(shard.fast_used)`, maintained by every locked section that changes
	/// one, so the settle loop can test the budget without taking a lock.
	/// `fast_used_matches_the_shards` pins the two together.
	fast_used: AtomicU64,

	/// Non-zero when at least one shard has migrations waiting, so the worker's
	/// per-pass drain costs one atomic load instead of `SHARDS` lock
	/// acquisitions on the overwhelmingly common empty pass.
	pending_migrations: AtomicUsize,

	/// Accesses that must elapse before a key is relinked again. 0 relinks
	/// every time, which is exact LRU. memcached's equivalent is 60 seconds.
	update_interval: u64,

	/// Fast-tier byte budget across ALL shards, settled against globally.
	fast_capacity: AtomicU64,
	shared_overhead: AtomicU64,
	high_ppm: AtomicU64,
	low_ppm: AtomicU64,
}

impl<K, V> Default for MergedStore<K, V> {
	fn default() -> Self {
		let shards = (0..SHARDS)
			.map(|_| RwLock::new(Inner::new()))
			.collect::<Vec<_>>()
			.into_boxed_slice();

		let tails = (0..SHARDS)
			.map(|_| TailSeq(AtomicU64::new(EMPTY_TAIL)))
			.collect::<Vec<_>>()
			.into_boxed_slice();

		let fast_tails = (0..SHARDS)
			.map(|_| TailSeq(AtomicU64::new(EMPTY_TAIL)))
			.collect::<Vec<_>>()
			.into_boxed_slice();

		MergedStore {
			shards,
			tails,
			fast_tails,
			clock: AtomicU64::new(0),
			tracked: AtomicUsize::new(0),
			fast_used: AtomicU64::new(0),
			pending_migrations: AtomicUsize::new(0),
			update_interval: std::env::var("MERGED_UPDATE_INTERVAL")
				.ok()
				.and_then(|v| v.parse().ok())
				.unwrap_or(0),

			// Untiered until the worker configures it: nothing can ever exceed
			// this, so `settle_tier` returns at its first comparison -- one
			// relaxed load -- and a flat build pays no tiering cost at all.
			fast_capacity: AtomicU64::new(CacheSize::MAX),
			shared_overhead: AtomicU64::new(0),
			high_ppm: AtomicU64::new(DEFAULT_HIGH_PPM),
			low_ppm: AtomicU64::new(DEFAULT_LOW_PPM),
		}
	}
}

impl<K, V> MergedStore<K, V> {
	pub fn new() -> Self {
		Self::default()
	}

	/// Installs the fast-tier budget and the per-object DRAM reservation, and
	/// settles every shard against them.
	///
	/// Called once by the policy worker when it builds its `PolicyStack` over
	/// this same `Arc`, with the values `init_policy_stack` hands the split
	/// hybrid stacks -- so the merged store is tiered on exactly the terms
	/// `LruCompactHybridStack` is.
	pub fn configure_tiering(
		&self,
		fast_capacity: CacheSize,
		shared_overhead: CacheSize,
		high_ppm: u64,
		low_ppm: u64,
	) {
		self.fast_capacity.store(fast_capacity, Ordering::Relaxed);
		self.shared_overhead.store(shared_overhead, Ordering::Relaxed);
		self.high_ppm.store(high_ppm, Ordering::Relaxed);
		self.low_ppm.store(low_ppm.min(high_ppm), Ordering::Relaxed);

		self.settle_all();
	}

	fn budget(&self) -> TierBudget {
		TierBudget {
			capacity: self.fast_capacity.load(Ordering::Relaxed),
			shared_overhead: self.shared_overhead.load(Ordering::Relaxed),
			high_ppm: self.high_ppm.load(Ordering::Relaxed),
			low_ppm: self.low_ppm.load(Ordering::Relaxed),
		}
	}

	#[inline]
	fn publish_tail(&self, shard: usize, inner: &Inner<K, V>) {
		self.tails[shard].0.store(inner.tail_seq(), Ordering::Relaxed);
	}

	#[inline]
	fn publish_fast_tail(&self, shard: usize, inner: &Inner<K, V>) {
		self.fast_tails[shard].0.store(inner.fast_tail_seq(), Ordering::Relaxed);
	}

	/// Both mirrors at once. Every locked section that can move a shard's tail
	/// or its tier boundary ends with this, so the two lock-free choices --
	/// which key to evict and which shard to demote from -- are always reading
	/// the shard's real state.
	#[inline]
	fn publish_mirrors(&self, shard: usize, inner: &Inner<K, V>) {
		self.publish_tail(shard, inner);
		self.publish_fast_tail(shard, inner);
	}

	#[inline]
	fn note_migrations(&self, inner: &Inner<K, V>) {
		if !inner.migrations.is_empty() {
			self.pending_migrations.store(1, Ordering::Relaxed);
		}
	}

	/// Applies a shard's change in `fast_used` to the store-level total.
	///
	/// Taken as a before/after pair rather than as a delta computed by each
	/// call site: the shard's own counter is the authority, so the total
	/// cannot drift from it by anyone forgetting which way a particular
	/// operation moved the bytes.
	#[inline]
	fn apply_fast_delta(&self, before: CacheSize, after: CacheSize) {
		match after.cmp(&before) {
			std::cmp::Ordering::Greater => {
				self.fast_used.fetch_add(after - before, Ordering::Relaxed);
			},

			std::cmp::Ordering::Less => {
				// Saturating: a concurrent settler may have subtracted the same
				// bytes a moment earlier, and an underflow here would wrap to
				// `u64::MAX` and demote the entire fast tier.
				let _ = self.fast_used.fetch_update(
					Ordering::Relaxed,
					Ordering::Relaxed,
					|v| Some(v.saturating_sub(before - after)),
				);
			},

			std::cmp::Ordering::Equal => {},
		}
	}

	/// The shard whose fast boundary is oldest -- the globally least-recently-
	/// used fast object -- from `SHARDS` relaxed loads and no lock.
	fn oldest_fast_shard(&self) -> Option<usize> {
		let mut best = None;
		let mut best_seq = EMPTY_TAIL;

		for (s, t) in self.fast_tails.iter().enumerate() {
			let seq = t.0.load(Ordering::Relaxed);

			if seq == EMPTY_TAIL {
				continue;
			}

			// A plain minimum: the clock is 64 bits and never wraps, so an
			// older slot's stamp is simply the smaller number.
			if best.is_none() || seq < best_seq {
				best_seq = seq;
				best = Some(s);
			}
		}

		best
	}

	/// Demote, globally, until the fast tier is back under the low watermark.
	///
	/// Runs with NO shard lock held. Each step is: 32 relaxed loads to name the
	/// shard holding the oldest fast object, that ONE shard's write lock, one
	/// boundary step, republish, unlock. So no thread ever holds two shard
	/// locks and no lock-order cycle can form -- a toucher releases its own
	/// shard before calling this.
	///
	/// Two settlers running at once may each demote one extra object, which is
	/// what the 0.98/0.95 hysteresis is for. The cost is paid by the API thread
	/// that caused the overshoot.
	fn settle_tier(&self) {
		let budget = self.budget();

		// The reservation is per LIVE object and applies across both tiers, so
		// it comes off the budget before the watermarks are taken.
		let effective = budget
			.capacity
			.saturating_sub(self.len() as CacheSize * budget.shared_overhead);

		if self.fast_used.load(Ordering::Relaxed) <= scale(effective, budget.high_ppm) {
			return;
		}

		let target = scale(effective, budget.low_ppm);

		while self.fast_used.load(Ordering::Relaxed) > target {
			let Some(s) = self.oldest_fast_shard() else {
				// Nothing anywhere is fast. Whatever is left over the target is
				// the shared-overhead reservation, not value bytes.
				break;
			};

			let mut g = self.shards[s].write().unwrap();
			let before = g.fast_used;

			// `None` when that shard's boundary went away between the load and
			// the lock -- another settler took it. Republish and re-choose.
			g.demote_boundary();

			self.apply_fast_delta(before, g.fast_used);
			self.note_migrations(&g);
			self.publish_fast_tail(s, &g);
		}
	}

	/// The worker's per-pass settle, and what `configure_tiering` and
	/// `resize_fast_tier` call. The SAME loop as an API thread's -- there is
	/// only one, so there is only one demotion order.
	fn settle_all(&self) {
		self.settle_tier();
	}

	/// Move `key` to the MRU end and make it fast.
	///
	/// The operation the design exists for: one lookup reaches the object, its
	/// position AND its tier, where the split design needs a second keyed
	/// lookup into a separate eviction stack for the last two.
	pub fn touch(&self, key: HashedKey) {
		let now = self.clock.fetch_add(1, Ordering::Relaxed);
		let s = shard_of(key);

		// memcached's `ITEM_UPDATE_INTERVAL`: a key hit repeatedly inside the
		// interval is already near enough the MRU end that relinking it buys
		// nothing, so skip under a READ lock and never block this shard's
		// readers at all.
		if self.update_interval > 0 {
			let g = self.shards[s].read().unwrap();

			let Some(i) = g.find(key) else { return };

			// A plain subtraction: the clock is 64 bits and monotonic, so the
			// stamp can only be at or behind it.
			let age = now.saturating_sub(g.slots[i as usize].last_access);

			if age < self.update_interval {
				return;
			}
		}

		{
			let mut g = self.shards[s].write().unwrap();

			let Some(i) = g.find(key) else { return };

			let before = g.fast_used;

			g.touch_slot(i, now);

			self.apply_fast_delta(before, g.fast_used);
			self.note_migrations(&g);
			self.publish_mirrors(s, &g);
		}

		// AFTER the guard is dropped -- see `settle_tier`. Holding it here
		// would let the settle take a second shard lock while holding this one.
		self.settle_tier();
	}

	/// The worker has finished processing the `Set` event for `key`: settle the
	/// tier against the bytes the insert already accounted.
	///
	/// This used to do three more things, and each was wrong once the slot
	/// stopped storing a size:
	///
	///   * it wrote `size` and `dram_resident` into the slot. Both are gone --
	///     `Slot::migrating` derives them from the object, which has held the
	///     value's length since the value became one word, so the bytes are
	///     accounted by `insert` at the moment they become reachable rather
	///     than one worker event later;
	///   * it RELINKED the slot to the MRU end, which `insert` had already
	///     done a moment earlier. Two relinks per set, the second of them
	///     redundant, both taking the shard write lock;
	///   * it bumped the clock a SECOND time, so one set consumed two stamps
	///     and a set looked, to the recency order, more recent than a get of
	///     the same age.
	///
	/// The parameters stay to match `PolicyStack::insert_resident`, whose other
	/// implementations do keep a size of their own. A key evicted between the
	/// insert and this call is simply gone, and settling is still correct.
	pub fn record_size(&self, key: HashedKey, _size: ObjectSize, _dram_resident: ObjectSize) {
		let _ = key;

		self.settle_tier();
	}

	/// The globally least-recently-used key: the minimum over the shard tails.
	///
	/// `SHARDS` relaxed atomic loads and no lock. Each shard's list is ordered
	/// within itself, so the global LRU object is necessarily some shard's
	/// tail, and the oldest of those tails is it.
	pub fn tail_key(&self) -> Option<HashedKey> {
		// A plain minimum over the stamps. This was a wrapping-difference AGE
		// comparison because `last_access` was the clock truncated to 32 bits,
		// where raw values stop being ordered once the clock wraps; at 64 bits
		// the clock does not wrap, so the older stamp is simply the smaller
		// number and the comparison is exact by construction.
		let mut best_seq = EMPTY_TAIL;
		let mut best_shard = usize::MAX;

		for (s, t) in self.tails.iter().enumerate() {
			let raw = t.0.load(Ordering::Relaxed);

			if raw == EMPTY_TAIL {
				continue;
			}

			if best_shard == usize::MAX || raw < best_seq {
				best_seq = raw;
				best_shard = s;
			}
		}

		if best_shard == usize::MAX {
			return None;
		}

		let g = self.shards[best_shard].read().unwrap();

		match g.tail {
			NIL => None,
			t => Some(g.slots[t as usize].hashed),
		}
	}

	pub fn contains_key(&self, key: &HashedKey) -> bool {
		self.shards[shard_of(*key)].read().unwrap().find(*key).is_some()
	}

	pub fn contains(&self, key: HashedKey) -> bool {
		self.contains_key(&key)
	}

	/// Remove and RETURN the object, unlinking it from the recency order and
	/// reversing its tier accounting in the same operation.
	///
	/// This is the merge paying off directly: the split design removes from the
	/// map and separately tells the stack, and when the second half is skipped
	/// the two diverge -- the failure `ERASE_FALLBACK` in `lib.rs::erase`
	/// exists to count. Here there is one structure, so the divergence has no
	/// way to occur.
	pub fn take(&self, key: &HashedKey) -> Option<Object<K, V>> {
		let s = shard_of(*key);
		let mut g = self.shards[s].write().unwrap();
		let i = g.bucket_unlink(*key)?;
		let before = g.fast_used;

		g.detach_tier(i);
		g.unlink(i);

		// Handed to the caller rather than dropped here, so the value's
		// retirement happens wherever the caller drops it -- still under a pin,
		// via `Object::drop`.
		let taken = g.slots[i as usize].object.take();
		g.free.push(i);

		self.apply_fast_delta(before, g.fast_used);
		self.publish_mirrors(s, &g);
		self.tracked.fetch_sub(1, Ordering::Relaxed);

		taken
	}

	pub fn remove_key(&self, key: HashedKey) -> bool {
		let s = shard_of(key);
		let mut g = self.shards[s].write().unwrap();

		let Some(i) = g.bucket_unlink(key) else { return false };

		let before = g.fast_used;

		g.retire(i);

		self.apply_fast_delta(before, g.fast_used);
		self.publish_mirrors(s, &g);
		self.tracked.fetch_sub(1, Ordering::Relaxed);

		true
	}

	pub fn get_ref(&self, key: &HashedKey) -> Option<MergedRef<'_, K, V>> {
		let guard = self.shards[shard_of(*key)].read().unwrap();
		let slot = guard.find(*key)?;

		Some(MergedRef { guard, slot })
	}

	pub fn get_mut_ref(&self, key: &HashedKey) -> Option<MergedRefMut<'_, K, V>> {
		let guard = self.shards[shard_of(*key)].write().unwrap();
		let slot = guard.find(*key)?;

		Some(MergedRefMut { guard, slot })
	}

	/// Insert at the MRU end, replacing any existing object for `key`.
	///
	/// Admission is unconditionally fast, matching `LruCompactHybridStack` --
	/// and matching what `PaperCache::set` physically built, since
	/// `admission_latched` is false for this store.
	pub fn insert(&self, key: HashedKey, object: Object<K, V>) -> Option<Object<K, V>> {
		let now = self.clock.fetch_add(1, Ordering::Relaxed);
		let s = shard_of(key);

		let old = {
			let mut g = self.shards[s].write().unwrap();
			let before = g.fast_used;

			let old = match g.find(key) {
				Some(i) => {
					// An overwrite can change the value's length, so the tier
					// accounting moves by the DIFFERENCE, charged to whichever
					// tier the slot is in at this instant. `touch_slot` then
					// moves the new figure to the fast tier if it was slow.
					let was = g.slots[i as usize].migrating();
					let old = g.slots[i as usize].object.replace(object);
					let now_bytes = g.slots[i as usize].migrating();

					match g.slots[i as usize].tier {
						Tier::Fast => {
							g.fast_used = (g.fast_used + now_bytes).saturating_sub(was)
						},

						Tier::Slow => {
							g.slow_used = (g.slow_used + now_bytes).saturating_sub(was)
						},
					}

					g.touch_slot(i, now);

					old
				},

				None => {
					let fresh = Slot {
						object: Some(object),
						hashed: key,
						prev: NIL,
						next: NIL,
						hash_next: NIL,
						last_access: now,
						tier: Tier::Fast,
					};

					let i = match g.free.pop() {
						Some(i) => {
							g.slots[i as usize] = fresh;
							i
						},

						// Appends a chunk when the last one is full. Nothing is
						// copied and no slot moves, so this cannot stall the
						// shard the way the old `reserve_exact` growth did.
						None => g.slots.alloc(fresh),
					};

					g.link_front(i);
					g.bucket_link(i);
					g.fast_count += 1;

					// The bytes are accounted HERE, not one worker event later:
					// the object carries its own length, so admitting it to the
					// fast tier and charging it are the same moment.
					g.fast_used += g.slots[i as usize].migrating();

					if g.fast_boundary == NIL {
						g.fast_boundary = i;
					}

					self.tracked.fetch_add(1, Ordering::Relaxed);

					None
				},
			};

			self.apply_fast_delta(before, g.fast_used);
			self.note_migrations(&g);
			self.publish_mirrors(s, &g);

			old
		};

		// Outside the guard: the settle takes one shard lock at a time and
		// this thread must not be holding another one.
		self.settle_tier();

		old
	}

	pub fn clear(&self) {
		for (s, lock) in self.shards.iter().enumerate() {
			let mut g = lock.write().unwrap();

			g.buckets.clear();
			g.buckets.resize(INITIAL_BUCKETS, NIL);
			g.base = INITIAL_BUCKETS;
			g.split = 0;
			g.live = 0;
			g.slots.clear();
			g.free.clear();
			g.head = NIL;
			g.tail = NIL;
			g.fast_boundary = NIL;
			g.fast_used = 0;
			g.slow_used = 0;
			g.fast_count = 0;
			g.migrations.clear();

			self.publish_mirrors(s, &g);
		}

		// Every slot vector dropped above retired its objects' values into this
		// thread's epoch bag. Push them out now: a `clear` is the one moment
		// the whole cache's worth of garbage appears at once, and leaving it in
		// a local bag would keep it resident until this thread happened to pin
		// enough more times to fill it.
		crate::value::flush();

		self.tracked.store(0, Ordering::Relaxed);
		self.pending_migrations.store(0, Ordering::Relaxed);
		self.fast_used.store(0, Ordering::Relaxed);
	}

	pub fn len(&self) -> usize {
		self.tracked.load(Ordering::Relaxed)
	}

	pub fn is_empty(&self) -> bool {
		self.len() == 0
	}

	/// Drains every (key, new tier) pair that crossed the fast/slow boundary
	/// since the last call, across all shards.
	pub fn drain_migrations(&self) -> Vec<(HashedKey, Tier)> {
		if self.pending_migrations.swap(0, Ordering::Relaxed) == 0 {
			return Vec::new();
		}

		let mut out = Vec::new();

		for lock in self.shards.iter() {
			let mut g = lock.write().unwrap();

			if !g.migrations.is_empty() {
				out.append(&mut g.migrations);
			}
		}

		out
	}

	pub fn resize_fast_tier(&self, size: CacheSize) {
		self.fast_capacity.store(size, Ordering::Relaxed);
		self.settle_all();
	}

	/// DRAM reserved out of the fast tier for shared per-object metadata across
	/// both tiers, so demotion bounds total DRAM and not just fast-tier values.
	pub fn dram_reserved_bytes(&self) -> CacheSize {
		self.len() as CacheSize * self.shared_overhead.load(Ordering::Relaxed)
	}

	/// The store-level total the settle loop tests, rather than a sum over the
	/// shards: the two are the same number -- `fast_used_matches_the_shards`
	/// asserts it -- and this one costs a relaxed load instead of 32 locks.
	pub fn fast_bytes_used(&self) -> CacheSize {
		self.fast_used.load(Ordering::Relaxed)
	}

	pub fn slow_bytes_used(&self) -> CacheSize {
		self.sum_shards(|g| g.slow_used)
	}

	pub fn fast_object_count(&self) -> usize {
		self.sum_shards(|g| g.fast_count as CacheSize) as usize
	}

	pub fn slow_object_count(&self) -> usize {
		self.len().saturating_sub(self.fast_object_count())
	}

	/// Gauges are read once per event-loop pass by `refresh_tier_gauges`, so
	/// they are summed under read locks rather than mirrored into atomics that
	/// every insert and demotion would then have to contend on.
	fn sum_shards<F>(&self, f: F) -> CacheSize
	where
		F: Fn(&Inner<K, V>) -> CacheSize,
	{
		self.shards.iter().map(|lock| f(&lock.read().unwrap())).sum()
	}

	/// Slab, index and free-list CAPACITIES summed across shards, for
	/// attributing a measured allocation to the three things that hold it.
	///
	/// Sharding makes this worth reporting rather than deriving: 32 slabs and
	/// 32 bucket arrays each round up to their own size class independently, so
	/// the slack is real and is not visible from the object count alone.
	///
	/// The slab figure is chunks x `SLAB_CHUNK` -- a chunk is committed whole,
	/// so that is what it costs. The index figure is the bucket `Vec`'s
	/// CAPACITY, not its length: linear hashing pushes one bucket at a time, so
	/// the `Vec`'s own doubling leaves it holding between 1x and 2x the buckets
	/// in use, and that spare is allocated memory like any other. Measuring at
	/// powers of two samples the 1x end of that range.
	pub fn capacities(&self) -> (usize, usize, usize) {
		self.shards
			.iter()
			.map(|lock| {
				let g = lock.read().unwrap();
				(g.slots.capacity(), g.buckets.capacity(), g.free.capacity())
			})
			.fold((0, 0, 0), |a, b| (a.0 + b.0, a.1 + b.1, a.2 + b.2))
	}

	/// The tier `key` is currently placed in, for tests and fidelity checks.
	pub fn tier_of(&self, key: HashedKey) -> Option<Tier> {
		let g = self.shards[shard_of(key)].read().unwrap();
		let i = g.find(key)?;

		Some(g.slots[i as usize].tier)
	}
}

/// Read handle. Holds the shard guard and re-indexes on `Deref`, matching what
/// DashMap's `Ref` gives the rest of the crate.
pub struct MergedRef<'a, K, V> {
	guard: RwLockReadGuard<'a, Inner<K, V>>,
	slot: u32,
}

impl<K, V> Deref for MergedRef<'_, K, V> {
	type Target = Object<K, V>;

	fn deref(&self) -> &Object<K, V> {
		self.guard.slots[self.slot as usize].object.as_ref().expect("live slot")
	}
}

pub struct MergedRefMut<'a, K, V> {
	guard: RwLockWriteGuard<'a, Inner<K, V>>,
	slot: u32,
}

impl<K, V> Deref for MergedRefMut<'_, K, V> {
	type Target = Object<K, V>;

	fn deref(&self) -> &Object<K, V> {
		self.guard.slots[self.slot as usize].object.as_ref().expect("live slot")
	}
}

impl<K, V> DerefMut for MergedRefMut<'_, K, V> {
	fn deref_mut(&mut self) -> &mut Object<K, V> {
		self.guard.slots[self.slot as usize].object.as_mut().expect("live slot")
	}
}

impl<K, V> MergedStore<K, V> {
	/// Overrides `MERGED_UPDATE_INTERVAL` for a single store. Builder-shaped
	/// because the interval is read once and then never written -- it is the
	/// one piece of configuration on the `touch` fast path, and keeping it a
	/// plain field rather than an atomic keeps that path to a compare.
	pub fn with_update_interval(mut self, accesses: u64) -> Self {
		self.update_interval = accesses;
		self
	}
}

#[cfg(test)]
mod tests {
	use super::*;

	type Store = MergedStore<u64, crate::BufferDRAM>;

	/// Spreads the HIGH bits, which is what selects a shard.
	fn mix(i: u64) -> HashedKey {
		i.wrapping_mul(0x9E37_79B9_7F4A_7C15)
	}

	fn tiered(fast_capacity: CacheSize) -> Store {
		let s = Store::new();
		s.configure_tiering(fast_capacity, 0, DEFAULT_HIGH_PPM, DEFAULT_LOW_PPM);
		s
	}

	/// The value is `size` bytes of real allocation, because the tier
	/// accounting is now DERIVED from the object rather than reported
	/// separately -- an object built with an empty value migrates zero bytes
	/// whatever `record_size` is told.
	fn put(s: &Store, key: HashedKey, size: ObjectSize) {
		s.insert(key, Object::new(key, &vec![0u8; size as usize], None));
		s.record_size(key, size, 0);
	}

	/// What a slot of `size` bytes of value contributes to a tier, as the store
	/// counts it: the allocator's rounded figure, not the request.
	fn migrating_bytes(size: ObjectSize) -> CacheSize {
		crate::object::overhead::resident_value_bytes(size) as CacheSize
	}

	/// Sum of what every live slot claims to be migrating, walked directly.
	fn live_migrating(s: &Store) -> CacheSize {
		s.shards
			.iter()
			.map(|lock| {
				let g = lock.read().unwrap();
				let mut total = 0;
				let mut i = g.head;

				while i != NIL {
					total += g.slots[i as usize].migrating();
					i = g.slots[i as usize].next;
				}

				total
			})
			.sum()
	}

	/// The index is `NoHasher`d, so hashbrown's bucket index IS the key's low
	/// bits. Sharding on those would put every key in a shard into one bucket
	/// and turn each shard's map into a linked list.
	#[test]
	fn shards_on_the_high_bits() {
		// Keys differing ONLY in their low bits must spread across shards...
		let low_bit_family: Vec<usize> = (0..SHARDS as u64)
			.map(|i| shard_of(0xABCD_0000_0000_0000 | i))
			.collect();

		assert!(
			low_bit_family.iter().all(|s| *s == low_bit_family[0]),
			"low bits must NOT select the shard, or the shard choice and the \
			 bucket choice would be the same bits",
		);

		// ...and the high bits must select every shard exactly once.
		let mut seen = vec![false; SHARDS];

		for i in 0..SHARDS as u64 {
			seen[shard_of(i << (64 - SHARD_BITS))] = true;
		}

		assert!(seen.into_iter().all(|b| b), "every shard must be reachable");
	}

	/// The claim sharding has to earn: per-shard lists still yield the exact
	/// global LRU order, because each shard is ordered within itself and the
	/// oldest tail is the global tail.
	#[test]
	fn eviction_order_is_exact_lru_across_shards() {
		let s = tiered(CacheSize::MAX);
		let keys: Vec<HashedKey> = (1..=500u64).map(mix).collect();

		for &k in &keys {
			put(&s, k, 128);
		}

		// Re-touch in an order unrelated to insertion, so the expected answer
		// is the touch order and not an artefact of how the slabs filled.
		let mut order = keys.clone();
		order.rotate_left(137);

		for &k in &order {
			s.touch(k);
		}

		let mut evicted = Vec::new();

		while let Some(k) = s.tail_key() {
			assert!(s.take(&k).is_some(), "nominated victim must be present");
			evicted.push(k);
		}

		assert_eq!(evicted, order, "eviction order is not exact LRU");
		assert_eq!(s.len(), 0);
	}

	/// Tier placement walks a cursor along the list and never moves a node, so
	/// a tight fast budget must not perturb the eviction order by one position.
	#[test]
	fn tiering_never_reorders() {
		let keys: Vec<HashedKey> = (1..=800u64).map(mix).collect();

		let drain = |cap: CacheSize| {
			let s = tiered(cap);

			for (n, &k) in keys.iter().enumerate() {
				put(&s, k, 256);

				// Interleave re-touches so promotions happen too, not just the
				// demotions a monotonic fill would produce.
				if n % 3 == 0 {
					s.touch(keys[n / 3]);
				}
			}

			let mut out = Vec::new();

			while let Some(k) = s.tail_key() {
				s.take(&k);
				out.push(k);
			}

			out
		};

		let untiered = drain(CacheSize::MAX);

		for cap in [4_096u64, 65_536, 1 << 20] {
			assert_eq!(drain(cap), untiered, "fast budget {cap} reordered the list");
		}
	}

	/// Every byte is in exactly one tier, and every object is counted once.
	#[test]
	fn tier_accounting_is_conserved() {
		let per_shard = 8_192u64;
		let s = tiered(per_shard * SHARDS as CacheSize);

		for i in 1..=2_000u64 {
			put(&s, mix(i), 512);

			if i % 7 == 0 {
				s.touch(mix(i / 7));
			}

			if i % 23 == 0 {
				s.remove_key(mix(i / 23));
			}
		}

		assert_eq!(
			s.fast_bytes_used() + s.slow_bytes_used(),
			live_migrating(&s),
			"tier byte totals do not add up to what the live slots hold",
		);

		assert_eq!(
			s.fast_object_count() + s.slow_object_count(),
			s.len(),
			"tier object counts do not add up to the tracked total",
		);

		// The budget is GLOBAL -- there is no per-shard share to overrun -- so
		// the bound is on the store's total.
		let capacity = per_shard * SHARDS as CacheSize;

		assert!(
			s.fast_bytes_used() <= scale(capacity, DEFAULT_HIGH_PPM),
			"the store overran its fast budget: {} > {}",
			s.fast_bytes_used(),
			scale(capacity, DEFAULT_HIGH_PPM),
		);
	}

	/// The store-level `fast_used` is what the settle loop tests without a
	/// lock, so it has to be the sum of what the shards actually hold. Every
	/// locked section that moves a shard's counter reports the difference; this
	/// is what catches one that forgets.
	#[test]
	fn fast_used_matches_the_shards() {
		let s = tiered(4_096 * SHARDS as CacheSize);

		for i in 1..=3_000u64 {
			put(&s, mix(i), 256);

			if i % 6 == 0 {
				s.touch(mix(i / 6));
			}

			if i % 11 == 0 {
				s.remove_key(mix(i / 11));
			}

			if i % 17 == 0 {
				// An overwrite with a DIFFERENT length, which is the case the
				// delta accounting in `insert` exists for.
				put(&s, mix(i / 17), 1_024);
			}

			if i % 29 == 0 {
				s.take(&mix(i / 29));
			}
		}

		let summed: CacheSize = s.sum_shards(|g| g.fast_used);

		assert_eq!(
			s.fast_bytes_used(),
			summed,
			"the store-level fast_used has drifted from the shards it mirrors",
		);
	}

	/// The fast region must stay a PREFIX of the list: head..=fast_boundary
	/// fast, everything after it slow. If that ever breaks, demotion picks the
	/// wrong victim and the tier split silently stops meaning recency.
	#[test]
	fn the_fast_region_stays_a_prefix() {
		let s = tiered(4_096 * SHARDS as CacheSize);

		for i in 1..=1_500u64 {
			put(&s, mix(i), 256);

			if i % 5 == 0 {
				s.touch(mix(i / 5));
			}
		}

		for lock in s.shards.iter() {
			let g = lock.read().unwrap();
			let mut i = g.head;
			let mut seen_slow = false;
			let mut counted_fast = 0usize;

			while i != NIL {
				match g.slots[i as usize].tier {
					Tier::Fast => {
						assert!(!seen_slow, "a fast slot sits behind a slow one");
						counted_fast += 1;
						assert!(
							!seen_slow && (g.fast_boundary != NIL),
							"fast slots exist but the boundary is unset",
						);
					},

					Tier::Slow => {
						if !seen_slow {
							// The slot just before the first slow one must be
							// the boundary.
							assert_eq!(
								g.slots[i as usize].prev,
								g.fast_boundary,
								"the boundary is not the last fast slot",
							);
						}

						seen_slow = true;
					},
				}

				i = g.slots[i as usize].next;
			}

			assert_eq!(counted_fast, g.fast_count, "fast_count disagrees with the list");
		}
	}

	/// Every drained migration must name a real transition, and the last
	/// migration for a key must agree with where that key actually ended up.
	#[test]
	fn migrations_agree_with_final_placement() {
		let s = tiered(8_192 * SHARDS as CacheSize);
		let mut last: HashMap<HashedKey, Tier, NoHasher> =
			HashMap::with_hasher(NoHasher::default());

		for i in 1..=3_000u64 {
			put(&s, mix(i), 256);

			if i % 4 == 0 {
				s.touch(mix(i / 4));
			}

			for (k, t) in s.drain_migrations() {
				last.insert(k, t);
			}
		}

		for (k, t) in s.drain_migrations() {
			last.insert(k, t);
		}

		assert!(!last.is_empty(), "a budget this tight must have produced migrations");

		let mut checked = 0usize;

		for (k, t) in &last {
			if let Some(actual) = s.tier_of(*k) {
				assert_eq!(actual, *t, "key {k:#x} was reported as {t:?} but is {actual:?}");
				checked += 1;
			}
		}

		assert!(checked > 0, "no migrated key survived to be checked");
	}

	/// memcached's `ITEM_UPDATE_INTERVAL`: a key hit repeatedly inside the
	/// interval must not be relinked, so a hot key does NOT keep taking its
	/// shard's write lock.
	#[test]
	fn the_update_interval_skips_relinks() {
		let s = Store::new().with_update_interval(1_000);
		let hot = mix(1);
		let cold = mix(2);

		put(&s, hot, 64);
		put(&s, cold, 64);

		// `cold` was inserted last and so is currently the MRU end.
		for _ in 0..100 {
			s.touch(hot);
		}

		assert_eq!(
			s.tail_key(),
			Some(hot),
			"the interval must have suppressed every relink, leaving `hot` at the LRU end",
		);

		// The same store with the interval off relinks on the first touch.
		let exact = Store::new();
		put(&exact, hot, 64);
		put(&exact, cold, 64);
		exact.touch(hot);

		assert_eq!(exact.tail_key(), Some(cold), "exact mode must relink");
	}

	/// The slab's waste is ONE PARTLY-FILLED CHUNK per shard, and no more --
	/// bounded, rather than proportional to the cache like the 1.40x `Vec`
	/// doubling gave and the 1.10x a 25% growth factor gave.
	///
	/// This was `the_slab_does_not_double`, which asserted a RATIO. A ratio is
	/// the wrong shape for a chunked slab: it is dominated by the fixed 4095
	/// slots per shard, so it is loose at 200k objects and vanishes at 20M,
	/// while the real guarantee -- bounded absolute slack -- holds at both.
	#[test]
	fn the_slab_wastes_at_most_one_chunk_per_shard() {
		let s = tiered(CacheSize::MAX);

		for i in 1..=200_000u64 {
			put(&s, mix(i), 64);
		}

		let (slab_cap, _, _) = s.capacities();
		let live = s.len();
		let bound = live + SHARDS * SLAB_CHUNK;

		assert!(
			slab_cap <= bound,
			"slab capacity {slab_cap} exceeds live {live} by more than one \
			 chunk per shard ({bound})",
		);

		// And every chunk is really committed whole, so the figure the
		// measurement harness reports is chunks x SLAB_CHUNK exactly.
		for lock in s.shards.iter() {
			let g = lock.read().unwrap();

			assert_eq!(g.slots.capacity() % SLAB_CHUNK, 0, "a chunk was part-committed");
			assert!(
				g.slots.len() <= g.slots.capacity(),
				"more slots handed out than the chunks hold",
			);
		}
	}

	/// Growth must never move a slot: a chunk is boxed, so its address is fixed
	/// for as long as it lives, and appending more chunks cannot disturb it.
	/// The `Vec<Slot>` this replaced `realloc`ed -- under the shard write lock,
	/// on the API thread -- which is the stall being removed.
	#[test]
	fn appending_a_chunk_moves_no_slot() {
		let s = tiered(CacheSize::MAX);

		// Keys with a zero top five bits all land in shard 0, so one chunk
		// boundary is crossed after 4096 inserts rather than after 4096 x 32.
		let one_shard = |i: u64| mix(i) >> SHARD_BITS;
		let first = one_shard(1);

		put(&s, first, 64);
		assert_eq!(shard_of(first), 0);

		let address = {
			let g = s.shards[0].read().unwrap();
			let i = g.find(first).expect("just inserted");
			&g.slots[i as usize] as *const Slot<u64, crate::BufferDRAM> as usize
		};

		for i in 2..=(SLAB_CHUNK as u64 * 2 + 8) {
			put(&s, one_shard(i), 8);
		}

		let g = s.shards[0].read().unwrap();

		assert!(g.slots.capacity() >= SLAB_CHUNK * 2, "the shard never grew");

		let i = g.find(first).expect("still present");

		assert_eq!(
			&g.slots[i as usize] as *const Slot<u64, crate::BufferDRAM> as usize,
			address,
			"a slot moved when the slab grew",
		);
	}

	/// The cost claim linear hashing is here to make: an insert walks its OWN
	/// bucket's chain, plus -- when it crosses the load factor -- the one
	/// bucket it splits. Nothing else. The `grow_buckets` this replaced walked
	/// the whole recency list under the shard write lock, so a shard holding
	/// half a million keys stalled every request to it for half a million
	/// pointer chases.
	#[test]
	fn an_insert_walks_one_chain_plus_the_split_bucket() {
		let s = tiered(CacheSize::MAX);

		let mut splits = 0usize;
		let mut worst = 0usize;

		// `insert` alone, not `put`: `record_size` no longer looks the key up,
		// but keeping the measurement to the one call makes what is counted
		// unambiguous.
		for i in 1..=20_000u64 {
			let key = mix(i);
			let sh = shard_of(key);

			// The two chains this insert is ALLOWED to walk, measured before it
			// runs: its own bucket, and whichever bucket is next to split.
			let (allowed, before, buckets_before) = {
				let g = s.shards[sh].read().unwrap();
				let bucket = g.bucket_of(key);
				let own = g.chain_len(bucket);

				// The bucket a split would take, plus the new slot itself when
				// the insert links it onto that same chain before splitting it.
				let split = g.chain_len(g.split) + usize::from(bucket == g.split);

				(own + split, g.walk.load(Ordering::Relaxed), g.buckets.len())
			};

			s.insert(key, Object::new(key, &[0u8; 8], None));

			let g = s.shards[sh].read().unwrap();
			let walked = g.walk.load(Ordering::Relaxed) - before;

			assert!(
				walked <= allowed,
				"an insert walked {walked} links with only {allowed} available \
				 in its own bucket and the split bucket",
			);

			if g.buckets.len() > buckets_before {
				splits += 1;
			}

			worst = worst.max(walked);
		}

		assert!(splits > 100, "the table barely grew ({splits} splits); the bound is untested");

		// The absolute claim: the per-insert cost does not scale with the
		// table. Mean chain length is 1 at load factor 1.0, so both chains are
		// short no matter how many keys the shard holds.
		assert!(
			worst < 32,
			"the worst insert walked {worst} links -- that is a stall, not a chain",
		);
	}

	/// Every live slot must be reachable from its own bucket, exactly once,
	/// and no recycled slot may be reachable at all.
	///
	/// The invariant a chained table lives or dies on. A dangling `hash_next`
	/// into a recycled slot would resurrect a dead key -- `find` would return a
	/// slot whose `hashed` happens to still match, handing the caller an object
	/// that was deleted.
	#[test]
	fn bucket_chains_hold_exactly_the_live_slots() {
		let s = tiered(CacheSize::MAX);
		let mut alive: Vec<HashedKey> = Vec::new();

		// Churn hard enough to recycle slots and to force several doublings.
		for i in 1..=4_000u64 {
			let k = mix(i);
			put(&s, k, 128);
			alive.push(k);

			if i % 3 == 0 {
				let victim = alive.remove(i as usize % alive.len());
				assert!(s.remove_key(victim), "remove of a live key failed");
			}
		}

		for &k in &alive {
			assert!(s.contains(k), "live key {k:#x} is not reachable from its bucket");
		}

		let mut chained = 0usize;

		for lock in s.shards.iter() {
			let g = lock.read().unwrap();

			// Everything on a chain must also be on the recency list.
			let mut on_list = std::collections::HashSet::new();
			let mut i = g.head;

			while i != NIL {
				assert!(on_list.insert(i), "the recency list contains a cycle");
				i = g.slots[i as usize].next;
			}

			for b in 0..g.buckets.len() {
				let mut i = g.buckets[b];
				let mut walked = 0usize;

				while i != NIL {
					assert!(
						on_list.contains(&i),
						"slot {i} is on a chain but not on the recency list -- it \
						 is recycled and would resurrect a deleted key",
					);
					assert_eq!(
						g.bucket_of(g.slots[i as usize].hashed),
						b,
						"slot {i} is in the wrong bucket",
					);

					chained += 1;
					walked += 1;
					assert!(walked <= g.live + 1, "bucket {b} chain does not terminate");
					i = g.slots[i as usize].hash_next;
				}
			}

			assert_eq!(on_list.len(), g.live, "live count disagrees with the list");
		}

		assert_eq!(chained, s.len(), "chain membership disagrees with the tracked total");
		assert_eq!(chained, alive.len(), "chain membership disagrees with what was kept");
	}

	/// The table must actually double, and carry every key across each rehash.
	#[test]
	fn growth_rehashes_without_losing_a_key() {
		let s = tiered(CacheSize::MAX);

		let start: usize = {
			let g = s.shards[0].read().unwrap();
			g.buckets.len()
		};

		let keys: Vec<HashedKey> = (1..=20_000u64).map(mix).collect();

		for &k in &keys {
			put(&s, k, 64);
		}

		let grown: usize = {
			let g = s.shards[0].read().unwrap();
			g.buckets.len()
		};

		assert!(grown > start, "the table never grew: {start} -> {grown}");

		for &k in &keys {
			assert!(s.contains(k), "key {k:#x} was lost across a rehash");
		}

		// Load factor 1.0 means buckets never exceed 2x the live count -- one
		// doubling past the trigger -- which is the memory claim being made.
		for lock in s.shards.iter() {
			let g = lock.read().unwrap();
			assert!(
				g.buckets.len() <= 2 * g.live.max(INITIAL_BUCKETS),
				"a shard over-provisioned its table: {} buckets for {} entries",
				g.buckets.len(),
				g.live,
			);
		}
	}

	/// A taken key must leave no trace on its chain.
	#[test]
	fn taking_a_key_clears_its_chain_entry() {
		let s = tiered(CacheSize::MAX);

		// Three keys deliberately sharing one bucket, so removal has to repair
		// a predecessor link rather than just a bucket head.
		let g0 = s.shards[0].read().unwrap();
		let nb = g0.buckets.len() as u64;
		drop(g0);

		let collide: Vec<HashedKey> = (0..3u64).map(|i| (i + 1) * nb).collect();

		for &k in &collide {
			assert_eq!(shard_of(k), shard_of(collide[0]), "keys must share a shard");
			put(&s, k, 64);
		}

		{
			let g = s.shards[shard_of(collide[0])].read().unwrap();
			let b = g.bucket_of(collide[0]);
			assert_eq!(
				collide.iter().filter(|k| g.bucket_of(**k) == b).count(),
				3,
				"the keys did not actually collide",
			);
		}

		// Remove the MIDDLE of the chain.
		assert!(s.take(&collide[1]).is_some());
		assert!(!s.contains(collide[1]), "a taken key is still reachable");
		assert!(s.contains(collide[0]), "removing the middle broke the chain head");
		assert!(s.contains(collide[2]), "removing the middle orphaned the tail");

		// And reinserting it must not double-link.
		put(&s, collide[1], 64);

		let g = s.shards[shard_of(collide[0])].read().unwrap();
		let b = g.bucket_of(collide[1]);
		let mut i = g.buckets[b];
		let mut hits = 0;

		while i != NIL {
			if g.slots[i as usize].hashed == collide[1] {
				hits += 1;
			}

			i = g.slots[i as usize].hash_next;
		}

		assert_eq!(hits, 1, "the reinserted key appears {hits} times on its chain");
	}

	/// `last_access` is the FULL clock, so cross-shard ordering is a plain
	/// comparison of stamps -- and stays exact across the 2^32 boundary that
	/// used to be the truncation's wrap point.
	///
	/// Two claims, because widening the field is only half of it:
	///
	///   1. the clock does not wrap within the test, and cannot wrap in
	///      practice -- one `fetch_add` per relink, so a billion accesses a
	///      second would take 584 years;
	///   2. the order the store reports is exactly the order keys were touched
	///      in, across shards, with stamps straddling 2^32.
	///
	/// This replaces `cross_shard_ordering_survives_the_32_bit_wrap`, which
	/// hand-wrote truncated stamps either side of the wrap and asserted the
	/// wrapping difference picked the older one. There is no wrap left to
	/// survive; what has to be shown now is that there is none.
	#[test]
	fn cross_shard_ordering_is_exact_and_the_clock_never_wraps() {
		let s = tiered(CacheSize::MAX);

		// Start just below the old truncation boundary, so the run crosses it.
		let start = (1u64 << 32) - 64;
		s.clock.store(start, Ordering::Relaxed);

		let keys: Vec<HashedKey> = (1..=400u64).map(mix).collect();

		for &k in &keys {
			put(&s, k, 64);
		}

		// Re-touch everything EXCEPT the first 32, in an order unrelated to
		// insertion. The untouched ones keep stamps from below 2^32 while the
		// rest are above it, so the final order the store has to report spans
		// the old truncation boundary.
		let (kept, rest) = keys.split_at(32);
		let mut order = rest.to_vec();
		order.rotate_left(197);

		for &k in &order {
			s.touch(k);
		}

		let spanned: std::collections::HashSet<usize> =
			order.iter().map(|k| shard_of(*k)).collect();

		assert!(spanned.len() > 1, "the keys must span several shards for this to say anything");

		// The stamps really do straddle the old 32-bit boundary...
		let stamps: Vec<u64> = {
			let mut v = Vec::new();

			for lock in s.shards.iter() {
				let g = lock.read().unwrap();
				let mut i = g.head;

				while i != NIL {
					v.push(g.slots[i as usize].last_access);
					i = g.slots[i as usize].next;
				}
			}

			v
		};

		assert!(
			stamps.iter().any(|v| *v < (1 << 32)) && stamps.iter().any(|v| *v >= (1 << 32)),
			"the run did not cross 2^32, so the old truncation would not have \
			 been exercised either",
		);

		// ...and the clock is nowhere near wrapping.
		let clock = s.clock.load(Ordering::Relaxed);

		assert!(clock > start, "the clock did not advance");
		assert!(
			clock < u64::MAX / 2,
			"the clock is within a factor of two of wrapping, which the stamp \
			 comparison assumes cannot happen",
		);

		// Exact global order: drain by the store's own victim choice and get
		// back the never-touched keys in insert order, then the rest in touch
		// order -- across shards, and across 2^32.
		let mut expected = kept.to_vec();
		expected.extend_from_slice(&order);

		let mut evicted = Vec::new();

		while let Some(k) = s.tail_key() {
			assert!(s.take(&k).is_some(), "nominated victim must be present");
			evicted.push(k);
		}

		assert_eq!(evicted, expected, "cross-shard order is not exact");
	}

	/// The tier boundary is GLOBAL: the fast tier holds the globally most
	/// recently used objects, not each shard's own most recent.
	///
	/// The property is stated as a separation -- every fast stamp is newer than
	/// every slow stamp -- because that is what "global" means here and it is
	/// checkable without reconstructing the whole order. Under the per-shard
	/// budget this fails outright: each shard demotes to its own share, so a
	/// quiet shard keeps old fast objects while a busy one demotes new ones.
	#[test]
	fn the_tier_boundary_is_global_across_shards() {
		let s = tiered(CacheSize::MAX);

		// Deliberately skewed: a third of the keys go to one shard, so the
		// per-shard split would give that shard far too little room and the
		// others far too much.
		let mut keys: Vec<HashedKey> = Vec::new();

		for i in 1..=1_200u64 {
			keys.push(match i % 3 {
				0 => mix(i) >> SHARD_BITS,
				_ => mix(i),
			});
		}

		for &k in &keys {
			put(&s, k, 256);
		}

		// Room for about a quarter of them.
		s.resize_fast_tier(migrating_bytes(256) * 300);

		let mut newest_slow = 0u64;
		let mut oldest_fast = u64::MAX;
		let mut fast = 0usize;

		for lock in s.shards.iter() {
			let g = lock.read().unwrap();
			let mut i = g.head;

			while i != NIL {
				let slot = &g.slots[i as usize];

				match slot.tier {
					Tier::Fast => {
						oldest_fast = oldest_fast.min(slot.last_access);
						fast += 1;
					},

					Tier::Slow => newest_slow = newest_slow.max(slot.last_access),
				}

				i = g.slots[i as usize].next;
			}
		}

		assert!(fast > 0 && fast < keys.len(), "the budget demoted everything or nothing");

		assert!(
			newest_slow < oldest_fast,
			"a slow object (stamp {newest_slow}) is more recent than a fast one \
			 (stamp {oldest_fast}) -- the boundary is per shard, not global",
		);
	}

	/// Shrinking the fast tier at runtime must demote immediately, the way
	/// `PaperCache::set_fast_tier_size` expects of every hybrid stack.
	#[test]
	fn shrinking_the_fast_tier_demotes() {
		let s = tiered(CacheSize::MAX);

		for i in 1..=1_000u64 {
			put(&s, mix(i), 512);
		}

		s.drain_migrations();
		assert_eq!(s.slow_object_count(), 0, "an unbounded fast tier demotes nothing");

		s.resize_fast_tier(64 * SHARDS as CacheSize);

		assert!(s.slow_object_count() > 0, "shrinking the budget demoted nothing");
		assert!(
			!s.drain_migrations().is_empty(),
			"demotions happened but nothing was reported for physical migration",
		);
		assert_eq!(
			s.fast_object_count() + s.slow_object_count(),
			s.len(),
			"the resize lost track of objects",
		);
	}
}

/// Measured against the same baselines, same harness: jemalloc
/// `stats.allocated`, ONE point per process, powers of two.
#[cfg(all(test, feature = "numa_jemalloc"))]
mod measure {
	use super::*;

	/// Same reader as `policy_stack::measure_overhead`, duplicated because that
	/// module is private to `worker::policy`. The `epoch` write is required:
	/// jemalloc caches these statistics per epoch.
	fn allocated_bytes() -> u64 {
		unsafe {
			let mut e: u64 = 1;
			let mut sz = core::mem::size_of::<u64>();

			tikv_jemalloc_sys::mallctl(
				c"epoch".as_ptr(),
				&mut e as *mut u64 as *mut core::ffi::c_void,
				&mut sz,
				&mut e as *mut u64 as *mut core::ffi::c_void,
				sz,
			);

			let mut allocated: usize = 0;
			let mut len = core::mem::size_of::<usize>();

			let rc = tikv_jemalloc_sys::mallctl(
				c"stats.allocated".as_ptr(),
				&mut allocated as *mut usize as *mut core::ffi::c_void,
				&mut len,
				core::ptr::null_mut(),
				0,
			);

			assert_eq!(rc, 0, "stats.allocated unavailable");
			allocated as u64
		}
	}

	/// One point per process: the slab grows a chunk at a time and the bucket
	/// array a bucket at a time, so a second point in the same process would be
	/// read off a different point on a step function. `MSTORE_N` powers of two,
	/// least-squares slope outside.
	///
	/// The slope is what `MERGED_STORE_STRUCTURE_OVERHEAD` is set from, with
	/// the size-class-rounded value cost subtracted.
	#[test]
	#[ignore]
	fn measure_merged_store_point() {
		let n: u64 = match std::env::var("MSTORE_N") {
			Ok(v) => v.parse().expect("MSTORE_N"),
			Err(_) => return,
		};

		let vsize: usize = std::env::var("MSTORE_VALUE")
			.map(|v| v.parse().expect("MSTORE_VALUE"))
			.unwrap_or(64);

		let base = allocated_bytes();
		let store: MergedStore<u64, crate::BufferDRAM> = MergedStore::new();

		for i in 0..n {
			let k = i.wrapping_mul(0x9E37_79B9_7F4A_7C15);
			store.insert(k, Object::new(k, &vec![0u8; vsize], None));
			store.record_size(k, (vsize + 16) as ObjectSize, 16);
		}

		let after = allocated_bytes();
		let held = store.len();
		core::hint::black_box(&store);

		let (slab, index, free) = store.capacities();

		println!(
			"MSTORE {} {} {} {} slab_cap={} slab_b={} index_cap={} free_cap={} slot={}",
			n,
			vsize,
			after.saturating_sub(base),
			held,
			slab,
			slab * core::mem::size_of::<Slot<u64, crate::BufferDRAM>>(),
			index,
			free,
			core::mem::size_of::<Slot<u64, crate::BufferDRAM>>(),
		);
	}

	/// The control the merged point is differenced against: the SAME objects,
	/// the same value type, the same allocator, in the map this store replaces.
	///
	/// Comparing slopes rather than absolutes is what makes the number mean
	/// something -- the value buffer and its `Arc` are identical on both sides
	/// and cancel, leaving only the structural difference. The split design's
	/// third piece, the eviction stack, is not in this control: it is the
	/// separately measured 72 B/object of `LruCompactHybridStack`, which has to
	/// be added back to get the split design's true total.
	#[test]
	#[ignore]
	fn measure_dashmap_point() {
		let n: u64 = match std::env::var("MSTORE_N") {
			Ok(v) => v.parse().expect("MSTORE_N"),
			Err(_) => return,
		};

		let vsize: usize = std::env::var("MSTORE_VALUE")
			.map(|v| v.parse().expect("MSTORE_VALUE"))
			.unwrap_or(64);

		let base = allocated_bytes();
		let map: dashmap::DashMap<HashedKey, Object<u64, crate::BufferDRAM>, NoHasher> =
			dashmap::DashMap::with_hasher(NoHasher::default());

		for i in 0..n {
			let k = i.wrapping_mul(0x9E37_79B9_7F4A_7C15);
			map.insert(k, Object::new(k, &vec![0u8; vsize], None));
		}

		let after = allocated_bytes();
		let held = map.len();
		core::hint::black_box(&map);

		println!("DMAP {} {} {} {}", n, vsize, after.saturating_sub(base), held);
	}

	/// The layout claim the saving rests on, asserted rather than asserted-in-
	/// prose: `Option<Object>` must be free, because the `NonNull` inside the
	/// object's `TieredValue` is a niche the discriminant fits in.
	#[test]
	fn the_option_in_a_slot_is_free() {
		assert_eq!(
			core::mem::size_of::<Option<Object<u64, crate::BufferDRAM>>>(),
			core::mem::size_of::<Object<u64, crate::BufferDRAM>>(),
			"Option<Object> grew: the Arc niche is no longer absorbing the discriminant",
		);
	}

	/// Everything a slot adds on top of the object it already had to store.
	#[test]
	fn print_slot_layout() {
		println!(
			"SLOTLAYOUT slot={} object={} overhead={}",
			core::mem::size_of::<Slot<u64, crate::BufferDRAM>>(),
			core::mem::size_of::<Object<u64, crate::BufferDRAM>>(),
			core::mem::size_of::<Slot<u64, crate::BufferDRAM>>()
				- core::mem::size_of::<Object<u64, crate::BufferDRAM>>(),
		);
	}
}
