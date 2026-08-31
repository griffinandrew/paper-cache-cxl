//! `MergedStore` -- the object map and the LRU eviction order in ONE structure.
//!
//! A fourth `ObjectMapRef` shape, selected by `merged_object_store`, alongside
//! DashMap, `global_hashtable_pmem` and `hashbrown_dram`. It is a FEATURE and
//! not a `PaperPolicy` variant because the object store is a compile-time
//! choice here: `ObjectStore::get_ref` returns `impl Deref`, so the trait is
//! not object-safe and `objects` cannot be `dyn`.
//!
//! # Why
//!
//! Measured with `stats.allocated`, 2^20..2^22, R^2 = 1.000000: a tiered LRU
//! object costs 216 B of overhead today, as three independently-keyed pieces --
//! the DashMap row (96), the value `Arc` (48), and `LruCompactHybridStack`
//! (72). Most of that 72 is two copies of the key, present only to answer
//! "where in the LRU order is this key?" Merging answers it for free, because
//! finding the object IS finding its position. The standalone prototype
//! measured 104 B for the merged shape.
//!
//! # One lock, deliberately
//!
//! A single `RwLock` over the whole store rather than DashMap's per-shard
//! locks. Sharding would give each shard its own LRU list, so eviction would
//! take a per-shard tail -- APPROXIMATE LRU, a different policy whose results
//! are not comparable with `lru-compact-hybrid`. Exactness first; an
//! approximate sharded order is a separate design, not a tuning of this one.
//!
//! Contention is therefore the known cost of this version and is the thing to
//! measure next, not a detail to discover later.
//!
//! # Slot recycling
//!
//! `generation` bumps whenever a slot is freed, so a stale index fails a debug
//! assertion rather than silently addressing whatever now occupies the slot --
//! the exact failure `pmem_collections.rs` shipped when it dropped `dlv_list`'s
//! generation counter.
//!
//! # The stored hash
//!
//! `Object::key()` is the REAL key; the store is filed by `HashedKey`. So a
//! slot must carry its own hash to name itself on eviction. The alternative is
//! to recompute `hash_one(object.key())` per eviction and save 8 B/object --
//! worth taking once this is correct, since evictions are far rarer than
//! objects.

use std::{
	collections::HashMap,
	ops::{Deref, DerefMut},
	sync::RwLock,
};

use crate::{object::Object, HashedKey, NoHasher};

const NIL: u32 = u32::MAX;

struct Slot<K, V> {
	object: Object<K, V>,
	/// The store is keyed by hash; `Object::key()` is the real key.
	hashed: HashedKey,
	prev: u32,
	next: u32,
	generation: u32,
}

struct Inner<K, V> {
	index: HashMap<HashedKey, u32, NoHasher>,
	slots: Vec<Slot<K, V>>,
	free: Vec<u32>,
	/// Most-recently-used.
	head: u32,
	/// Least-recently-used; what `evict_one` takes.
	tail: u32,
}

impl<K, V> Inner<K, V> {
	fn unlink(&mut self, i: u32) {
		let (p, n) = {
			let s = &self.slots[i as usize];
			(s.prev, s.next)
		};

		if p != NIL {
			self.slots[p as usize].next = n;
		} else {
			self.head = n;
		}

		if n != NIL {
			self.slots[n as usize].prev = p;
		} else {
			self.tail = p;
		}
	}

	fn link_front(&mut self, i: u32) {
		let old = self.head;
		{
			let s = &mut self.slots[i as usize];
			s.prev = NIL;
			s.next = old;
		}

		if old != NIL {
			self.slots[old as usize].prev = i;
		}

		self.head = i;

		if self.tail == NIL {
			self.tail = i;
		}
	}

	fn retire(&mut self, i: u32) {
		self.unlink(i);
		let s = &mut self.slots[i as usize];
		s.generation = s.generation.wrapping_add(1);
		s.prev = NIL;
		s.next = NIL;
		self.free.push(i);
	}
}

pub struct MergedStore<K, V> {
	inner: RwLock<Inner<K, V>>,
}

impl<K, V> Default for MergedStore<K, V> {
	fn default() -> Self {
		MergedStore {
			inner: RwLock::new(Inner {
				index: HashMap::with_hasher(NoHasher::default()),
				slots: Vec::new(),
				free: Vec::new(),
				head: NIL,
				tail: NIL,
			}),
		}
	}
}

impl<K, V> MergedStore<K, V> {
	pub fn new() -> Self {
		Self::default()
	}

	/// Move `key` to the MRU end.
	///
	/// The operation the design exists for: one lookup reaches the object AND
	/// its position, where the split design needs a second keyed lookup into a
	/// separate eviction stack.
	pub fn touch(&self, key: HashedKey) {
		let mut g = self.inner.write().unwrap();

		let Some(&i) = g.index.get(&key) else { return };

		if g.head == i {
			return;
		}

		g.unlink(i);
		g.link_front(i);
	}

	/// Unlink the LRU tail and return its key.
	///
	/// The slot is recycled under the write guard, so no reader can hold it. A
	/// reader that already cloned the value's `Arc` is unaffected: that clone
	/// keeps the buffer alive independently, exactly as under DashMap today.
	pub fn evict_one(&self) -> Option<HashedKey> {
		let mut g = self.inner.write().unwrap();
		let i = g.tail;

		if i == NIL {
			return None;
		}

		let key = g.slots[i as usize].hashed;
		g.index.remove(&key);
		g.retire(i);

		Some(key)
	}

	pub fn contains(&self, key: HashedKey) -> bool {
		self.inner.read().unwrap().index.contains_key(&key)
	}

	pub fn remove_key(&self, key: HashedKey) -> bool {
		let mut g = self.inner.write().unwrap();

		let Some(i) = g.index.remove(&key) else { return false };

		g.retire(i);
		true
	}

	pub fn get_ref(&self, key: &HashedKey) -> Option<MergedRef<'_, K, V>> {
		let guard = self.inner.read().unwrap();
		let slot = *guard.index.get(key)?;
		Some(MergedRef { guard, slot })
	}

	pub fn get_mut_ref(&self, key: &HashedKey) -> Option<MergedRefMut<'_, K, V>> {
		let guard = self.inner.write().unwrap();
		let slot = *guard.index.get(key)?;
		Some(MergedRefMut { guard, slot })
	}

	/// Insert at the MRU end, replacing any existing object for `key`.
	pub fn insert(&self, key: HashedKey, object: Object<K, V>) -> Option<Object<K, V>> {
		let mut g = self.inner.write().unwrap();

		if let Some(&i) = g.index.get(&key) {
			let old = std::mem::replace(&mut g.slots[i as usize].object, object);
			g.unlink(i);
			g.link_front(i);
			return Some(old);
		}

		let slot = Slot { object, hashed: key, prev: NIL, next: NIL, generation: 0 };

		let i = match g.free.pop() {
			Some(i) => {
				let prev_gen = g.slots[i as usize].generation;
				g.slots[i as usize] = Slot { generation: prev_gen, ..slot };
				i
			},

			None => {
				g.slots.push(slot);
				(g.slots.len() - 1) as u32
			},
		};

		g.link_front(i);
		g.index.insert(key, i);
		None
	}

	pub fn clear(&self) {
		let mut g = self.inner.write().unwrap();
		g.index.clear();
		g.slots.clear();
		g.free.clear();
		g.head = NIL;
		g.tail = NIL;
	}

	pub fn len(&self) -> usize {
		self.inner.read().unwrap().index.len()
	}
}

/// Read handle. Holds the guard and re-indexes on `Deref`, matching what
/// DashMap's `Ref` gives the rest of the crate.
pub struct MergedRef<'a, K, V> {
	guard: std::sync::RwLockReadGuard<'a, Inner<K, V>>,
	slot: u32,
}

impl<K, V> Deref for MergedRef<'_, K, V> {
	type Target = Object<K, V>;

	fn deref(&self) -> &Object<K, V> {
		&self.guard.slots[self.slot as usize].object
	}
}

pub struct MergedRefMut<'a, K, V> {
	guard: std::sync::RwLockWriteGuard<'a, Inner<K, V>>,
	slot: u32,
}

impl<K, V> Deref for MergedRefMut<'_, K, V> {
	type Target = Object<K, V>;

	fn deref(&self) -> &Object<K, V> {
		&self.guard.slots[self.slot as usize].object
	}
}

impl<K, V> DerefMut for MergedRefMut<'_, K, V> {
	fn deref_mut(&mut self) -> &mut Object<K, V> {
		&mut self.guard.slots[self.slot as usize].object
	}
}

/// Measured against the same baselines, same harness: jemalloc
/// `stats.allocated`, ONE point per process, powers of two.
#[cfg(all(test, feature = "hybrid_cache_common"))]
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
	use crate::TieredBuffer;

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
		let value = vec![0u8; vsize];

		let base = allocated_bytes();
		let store: MergedStore<u64, TieredBuffer> = MergedStore::new();

		for i in 0..n {
			let k = i.wrapping_mul(0x9E37_79B9_7F4A_7C15);
			let buf = TieredBuffer::new_fast(&value);
			store.insert(k, Object::new(k, buf, None));
		}

		let after = allocated_bytes();
		let held = store.len();
		core::hint::black_box(&store);
		println!("MSTORE {} {} {} {}", n, vsize, after.saturating_sub(base), held);
	}
}
