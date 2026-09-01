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
//! The index is `HashMap<HashedKey, _, NoHasher>`: the key IS the hash. hashbrown
//! selects a bucket with the LOW bits of that hash, so sharding on the low bits
//! would hold them constant inside a shard and drive every key in it to one
//! bucket. Shard on the high bits, which is what DashMap does and why DashMap
//! and `NoHasher` compose in the first place.
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
//! The fast budget is split evenly across shards and each settles its own
//! against the same high/low watermarks the split stacks use. Migrations
//! accumulate per shard and `drain_tier_migrations` concatenates them, so
//! `PolicyWorker::apply_tier_migrations` performs the physical
//! `Object::set_data` moves exactly as it does for every other hybrid stack.
//!
//! # Slot recycling
//!
//! `generation` bumps on every free, so a stale index fails loudly rather than
//! silently addressing whatever now occupies the slot -- the failure
//! `pmem_collections.rs` shipped when it dropped `dlv_list`'s equivalent.

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

/// Empty shard: sorts after every real counter, so `min` skips it.
const EMPTY_TAIL: u64 = u64::MAX;

struct Slot<K, V> {
	/// `Option` so `take` can move the object out without disturbing the slab.
	/// Free: `Object` holds an `Arc`, whose non-null niche absorbs the
	/// discriminant, so `Option<Object>` is the same size as `Object`.
	object: Option<Object<K, V>>,
	/// The store is keyed by hash; `Object::key()` is the real key, which
	/// `key_matches` needs to make collisions safe.
	hashed: HashedKey,
	prev: u32,
	next: u32,
	generation: u32,
	size: ObjectSize,
	/// Part of `size` that stays in DRAM whichever tier holds the object.
	dram_resident: u8,
	tier: Tier,
	/// Access counter at the last relink. Double duty: the update-interval
	/// check, and the cross-shard comparison that keeps eviction picking the
	/// true LRU tail.
	last_access: u64,
}

impl<K, V> Slot<K, V> {
	/// Bytes that actually move between tiers. `Object::set_data` migrates the
	/// value buffer alone, so the key, the expiry field and the `Expiries` row
	/// stay in DRAM in either tier and must not be charged to a tier's budget.
	fn migrating(&self) -> CacheSize {
		(self.size as CacheSize).saturating_sub(self.dram_resident as CacheSize)
	}
}

struct Inner<K, V> {
	index: HashMap<HashedKey, u32, NoHasher>,
	slots: Vec<Slot<K, V>>,
	free: Vec<u32>,

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
			index: HashMap::with_hasher(NoHasher::default()),
			slots: Vec::new(),
			free: Vec::new(),
			head: NIL,
			tail: NIL,
			fast_boundary: NIL,
			fast_used: 0,
			slow_used: 0,
			fast_count: 0,
			migrations: Vec::new(),
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
	fn retire(&mut self, i: u32) {
		self.detach_tier(i);
		self.unlink(i);

		let s = &mut self.slots[i as usize];
		s.generation = s.generation.wrapping_add(1);
		s.object = None;

		self.free.push(i);
	}

	/// Move to the MRU end and make fast, promoting from slow if needed.
	///
	/// Faithful port of `LruCompactHybridStack::touch_fast_key`.
	fn touch_slot(&mut self, i: u32, now: u64, budget: TierBudget) {
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

		let mut promoted = false;

		if previous_tier != Tier::Fast {
			let migrating = self.slots[i as usize].migrating();

			self.slow_used = self.slow_used.saturating_sub(migrating);
			self.fast_used += migrating;
			self.fast_count += 1;
			self.slots[i as usize].tier = Tier::Fast;
			promoted = true;

			if self.fast_boundary == NIL {
				self.fast_boundary = i;
			}
		}

		self.settle_fast_tier(budget);

		// Pushed after settling and guarded on the slot still being fast: a
		// tight budget can demote it straight back out within the same settle,
		// in which case that call already pushed the correct final entry.
		if promoted && self.slots[i as usize].tier == Tier::Fast {
			let key = self.slots[i as usize].hashed;
			self.migrations.push((key, Tier::Fast));
		}
	}

	/// Demotes from the tier boundary until `fast_used` is back under the low
	/// watermark. The victim is always `fast_boundary` -- the least-recently-
	/// used fast slot -- so nothing is searched, and because the boundary only
	/// walks along a list this never reorders anything.
	fn settle_fast_tier(&mut self, budget: TierBudget) {
		let effective = budget
			.per_shard_capacity
			.saturating_sub(self.index.len() as CacheSize * budget.shared_overhead);

		if self.fast_used <= scale(effective, budget.high_ppm) {
			return;
		}

		let drain_target = scale(effective, budget.low_ppm);

		while self.fast_used > drain_target {
			let d = self.fast_boundary;

			if d == NIL {
				break;
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
		}
	}

	fn tail_seq(&self) -> u64 {
		match self.tail {
			NIL => EMPTY_TAIL,
			t => self.slots[t as usize].last_access,
		}
	}
}

/// The tiering configuration a shard needs to settle itself, resolved once by
/// the caller so no shard has to touch the store's atomics under its lock.
#[derive(Clone, Copy)]
struct TierBudget {
	per_shard_capacity: CacheSize,
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

	clock: AtomicU64,
	tracked: AtomicUsize,

	/// Non-zero when at least one shard has migrations waiting, so the worker's
	/// per-pass drain costs one atomic load instead of `SHARDS` lock
	/// acquisitions on the overwhelmingly common empty pass.
	pending_migrations: AtomicUsize,

	/// Accesses that must elapse before a key is relinked again. 0 relinks
	/// every time, which is exact LRU. memcached's equivalent is 60 seconds.
	update_interval: u64,

	/// Fast-tier byte budget across ALL shards; each settles against its share.
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

		MergedStore {
			shards,
			tails,
			clock: AtomicU64::new(0),
			tracked: AtomicUsize::new(0),
			pending_migrations: AtomicUsize::new(0),
			update_interval: std::env::var("MERGED_UPDATE_INTERVAL")
				.ok()
				.and_then(|v| v.parse().ok())
				.unwrap_or(0),

			// Untiered until the worker configures it: nothing can ever exceed
			// this, so `settle_fast_tier` returns at its first comparison and a
			// flat build pays no tiering cost at all.
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
			// Saturating rather than wrapping at `CacheSize::MAX`, which is the
			// untiered sentinel: MAX / SHARDS is still far past anything a
			// shard can hold, so the settle still short-circuits.
			per_shard_capacity: self.fast_capacity.load(Ordering::Relaxed) / SHARDS as CacheSize,
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
	fn note_migrations(&self, inner: &Inner<K, V>) {
		if !inner.migrations.is_empty() {
			self.pending_migrations.store(1, Ordering::Relaxed);
		}
	}

	fn settle_all(&self) {
		let budget = self.budget();

		for (s, lock) in self.shards.iter().enumerate() {
			let mut g = lock.write().unwrap();
			g.settle_fast_tier(budget);
			self.note_migrations(&g);
			self.publish_tail(s, &g);
		}
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

			let Some(&i) = g.index.get(&key) else { return };

			if now.saturating_sub(g.slots[i as usize].last_access) < self.update_interval {
				return;
			}
		}

		let budget = self.budget();
		let mut g = self.shards[s].write().unwrap();

		let Some(&i) = g.index.get(&key) else { return };

		g.touch_slot(i, now, budget);
		self.note_migrations(&g);
		self.publish_tail(s, &g);
	}

	/// Records `key`'s size and tier-migrating remainder, admitting it to the
	/// fast tier and settling the shard.
	///
	/// The object is already in the map by the time this runs -- the API thread
	/// inserted it and the worker is now processing the `Set` event -- so this
	/// fills in the accounting the map insert had no size to do. A key evicted
	/// in between is simply absent, and is skipped.
	pub fn record_size(&self, key: HashedKey, size: ObjectSize, dram_resident: ObjectSize) {
		// Saturating, matching `narrow_resident`: any excess is then treated as
		// migrating, which is the behaviour before this accounting existed, so
		// it degrades toward the old over-charge instead of going wrong in a
		// new way.
		let dram_resident = dram_resident.min(u8::MAX as ObjectSize) as u8;

		let now = self.clock.fetch_add(1, Ordering::Relaxed);
		let budget = self.budget();
		let s = shard_of(key);
		let mut g = self.shards[s].write().unwrap();

		let Some(&i) = g.index.get(&key) else { return };

		// Re-size first so the tier accounting below moves the right number of
		// bytes, exactly as `LruCompactHybridStack::resize_key` does.
		{
			let old_migrating = g.slots[i as usize].migrating();

			let (tier, new_migrating) = {
				let slot = &mut g.slots[i as usize];
				slot.size = size;
				slot.dram_resident = dram_resident;
				(slot.tier, slot.migrating())
			};

			let delta = new_migrating as i64 - old_migrating as i64;

			match tier {
				Tier::Fast => g.fast_used = (g.fast_used as i64 + delta).max(0) as CacheSize,
				Tier::Slow => g.slow_used = (g.slow_used as i64 + delta).max(0) as CacheSize,
			}
		}

		g.touch_slot(i, now, budget);
		self.note_migrations(&g);
		self.publish_tail(s, &g);
	}

	/// The globally least-recently-used key: the minimum over the shard tails.
	///
	/// `SHARDS` relaxed atomic loads and no lock. Each shard's list is ordered
	/// within itself, so the global LRU object is necessarily some shard's
	/// tail, and the oldest of those tails is it.
	pub fn tail_key(&self) -> Option<HashedKey> {
		let mut best = EMPTY_TAIL;
		let mut best_shard = usize::MAX;

		for (s, t) in self.tails.iter().enumerate() {
			let seq = t.0.load(Ordering::Relaxed);

			if seq < best {
				best = seq;
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
		self.shards[shard_of(*key)].read().unwrap().index.contains_key(key)
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
		let i = g.index.remove(key)?;

		g.detach_tier(i);
		g.unlink(i);

		let slot = &mut g.slots[i as usize];
		slot.generation = slot.generation.wrapping_add(1);
		let taken = slot.object.take();

		g.free.push(i);
		self.publish_tail(s, &g);
		self.tracked.fetch_sub(1, Ordering::Relaxed);

		taken
	}

	pub fn remove_key(&self, key: HashedKey) -> bool {
		let s = shard_of(key);
		let mut g = self.shards[s].write().unwrap();

		let Some(i) = g.index.remove(&key) else { return false };

		g.retire(i);
		self.publish_tail(s, &g);
		self.tracked.fetch_sub(1, Ordering::Relaxed);

		true
	}

	pub fn get_ref(&self, key: &HashedKey) -> Option<MergedRef<'_, K, V>> {
		let guard = self.shards[shard_of(*key)].read().unwrap();
		let slot = *guard.index.get(key)?;

		Some(MergedRef { guard, slot })
	}

	pub fn get_mut_ref(&self, key: &HashedKey) -> Option<MergedRefMut<'_, K, V>> {
		let guard = self.shards[shard_of(*key)].write().unwrap();
		let slot = *guard.index.get(key)?;

		Some(MergedRefMut { guard, slot })
	}

	/// Insert at the MRU end, replacing any existing object for `key`.
	///
	/// Admission is unconditionally fast, matching `LruCompactHybridStack` --
	/// and matching what `PaperCache::set` physically built, since
	/// `admission_latched` is false for this store.
	pub fn insert(&self, key: HashedKey, object: Object<K, V>) -> Option<Object<K, V>> {
		let now = self.clock.fetch_add(1, Ordering::Relaxed);
		let budget = self.budget();
		let s = shard_of(key);
		let mut g = self.shards[s].write().unwrap();

		if let Some(&i) = g.index.get(&key) {
			let old = g.slots[i as usize].object.replace(object);

			g.touch_slot(i, now, budget);
			self.note_migrations(&g);
			self.publish_tail(s, &g);

			return old;
		}

		let fresh = Slot {
			object: Some(object),
			hashed: key,
			prev: NIL,
			next: NIL,
			generation: 0,
			size: 0,
			dram_resident: 0,
			tier: Tier::Fast,
			last_access: now,
		};

		let i = match g.free.pop() {
			Some(i) => {
				let prev_gen = g.slots[i as usize].generation;
				g.slots[i as usize] = Slot { generation: prev_gen, ..fresh };
				i
			},

			None => {
				g.slots.push(fresh);
				(g.slots.len() - 1) as u32
			},
		};

		g.link_front(i);
		g.index.insert(key, i);
		g.fast_count += 1;

		if g.fast_boundary == NIL {
			g.fast_boundary = i;
		}

		// Size is still 0 here -- the worker fills it in via `record_size` when
		// it processes the `Set` event -- so this settle can only demote on
		// bytes already accounted, never on this insert's own.
		g.settle_fast_tier(budget);

		self.note_migrations(&g);
		self.publish_tail(s, &g);
		self.tracked.fetch_add(1, Ordering::Relaxed);

		None
	}

	pub fn clear(&self) {
		for (s, lock) in self.shards.iter().enumerate() {
			let mut g = lock.write().unwrap();

			g.index.clear();
			g.slots.clear();
			g.free.clear();
			g.head = NIL;
			g.tail = NIL;
			g.fast_boundary = NIL;
			g.fast_used = 0;
			g.slow_used = 0;
			g.fast_count = 0;
			g.migrations.clear();

			self.publish_tail(s, &g);
		}

		self.tracked.store(0, Ordering::Relaxed);
		self.pending_migrations.store(0, Ordering::Relaxed);
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

	pub fn fast_bytes_used(&self) -> CacheSize {
		self.sum_shards(|g| g.fast_used)
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
	/// 32 hashmaps each round up to their own size class independently, so the
	/// slack is real and is not visible from the object count alone.
	pub fn capacities(&self) -> (usize, usize, usize) {
		self.shards
			.iter()
			.map(|lock| {
				let g = lock.read().unwrap();
				(g.slots.capacity(), g.index.capacity(), g.free.capacity())
			})
			.fold((0, 0, 0), |a, b| (a.0 + b.0, a.1 + b.1, a.2 + b.2))
	}

	/// The tier `key` is currently placed in, for tests and fidelity checks.
	pub fn tier_of(&self, key: HashedKey) -> Option<Tier> {
		let g = self.shards[shard_of(key)].read().unwrap();
		let i = *g.index.get(&key)?;

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

	type Store = MergedStore<u64, Vec<u8>>;

	/// Spreads the HIGH bits, which is what selects a shard.
	fn mix(i: u64) -> HashedKey {
		i.wrapping_mul(0x9E37_79B9_7F4A_7C15)
	}

	fn tiered(fast_capacity: CacheSize) -> Store {
		let s = Store::new();
		s.configure_tiering(fast_capacity, 0, DEFAULT_HIGH_PPM, DEFAULT_LOW_PPM);
		s
	}

	fn put(s: &Store, key: HashedKey, size: ObjectSize) {
		s.insert(key, Object::new(key, Vec::new(), None));
		s.record_size(key, size, 0);
	}

	/// Sum of what every live slot claims to be migrating, walked directly.
	fn live_migrating(s: &Store) -> CacheSize {
		s.shards
			.iter()
			.map(|lock| {
				let g = lock.read().unwrap();
				g.index.values().map(|&i| g.slots[i as usize].migrating()).sum::<CacheSize>()
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

		// Each shard settles against its own share, so the bound is per shard
		// and the global one follows by summing.
		for lock in s.shards.iter() {
			let g = lock.read().unwrap();

			assert!(
				g.fast_used <= scale(per_shard, DEFAULT_HIGH_PPM),
				"a shard overran its fast budget: {} > {}",
				g.fast_used,
				scale(per_shard, DEFAULT_HIGH_PPM),
			);
		}
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

	/// A recycled slot must not be silently addressable through a stale index.
	#[test]
	fn recycled_slots_bump_their_generation() {
		let s = tiered(CacheSize::MAX);
		let k = mix(9);

		put(&s, k, 64);

		let before = {
			let g = s.shards[shard_of(k)].read().unwrap();
			let i = g.index[&k];
			g.slots[i as usize].generation
		};

		s.take(&k);
		put(&s, k, 64);

		let after = {
			let g = s.shards[shard_of(k)].read().unwrap();
			let i = g.index[&k];
			g.slots[i as usize].generation
		};

		assert_eq!(after, before.wrapping_add(1), "the freed slot kept its generation");
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

	/// One point per process: the slab and both hashmaps grow by doubling, so a
	/// second point in the same process would be read off a different point on
	/// a step function. `MSTORE_N` powers of two, least-squares slope outside.
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
		let store: MergedStore<u64, Vec<u8>> = MergedStore::new();

		for i in 0..n {
			let k = i.wrapping_mul(0x9E37_79B9_7F4A_7C15);
			store.insert(k, Object::new(k, vec![0u8; vsize], None));
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
			slab * core::mem::size_of::<Slot<u64, Vec<u8>>>(),
			index,
			free,
			core::mem::size_of::<Slot<u64, Vec<u8>>>(),
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
		let map: dashmap::DashMap<HashedKey, Object<u64, Vec<u8>>, NoHasher> =
			dashmap::DashMap::with_hasher(NoHasher::default());

		for i in 0..n {
			let k = i.wrapping_mul(0x9E37_79B9_7F4A_7C15);
			map.insert(k, Object::new(k, vec![0u8; vsize], None));
		}

		let after = allocated_bytes();
		let held = map.len();
		core::hint::black_box(&map);

		println!("DMAP {} {} {} {}", n, vsize, after.saturating_sub(base), held);
	}

	/// The layout claim the saving rests on, asserted rather than asserted-in-
	/// prose: `Option<Object>` must be free, because `Object` holds an `Arc`
	/// and the niche absorbs the discriminant.
	#[test]
	fn the_option_in_a_slot_is_free() {
		assert_eq!(
			core::mem::size_of::<Option<Object<u64, Vec<u8>>>>(),
			core::mem::size_of::<Object<u64, Vec<u8>>>(),
			"Option<Object> grew: the Arc niche is no longer absorbing the discriminant",
		);
	}

	/// Everything a slot adds on top of the object it already had to store.
	#[test]
	fn print_slot_layout() {
		println!(
			"SLOTLAYOUT slot={} object={} overhead={}",
			core::mem::size_of::<Slot<u64, Vec<u8>>>(),
			core::mem::size_of::<Object<u64, Vec<u8>>>(),
			core::mem::size_of::<Slot<u64, Vec<u8>>>()
				- core::mem::size_of::<Object<u64, Vec<u8>>>(),
		);
	}
}
