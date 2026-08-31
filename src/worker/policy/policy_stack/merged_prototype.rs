//! PROTOTYPE, test-only: `LruCompactHybridStack` merged INTO the object map.
//!
//! Not wired into `init_policy_stack` and unreachable from the cache. It exists
//! to produce a measured B/object figure comparable, by the same
//! `stats.allocated` method, against the three structures it replaces.
//!
//! # What it replaces
//!
//! Today a tiered LRU object costs three independently-keyed allocations:
//!
//! ```text
//!   DashMap<HashedKey, Object>    96 B measured    key -> object
//!   Arc<TieredBuffer>             48 B measured    refcount + buffer handle
//!   LruCompactHybridStack         ~66 B            key -> slot -> links
//!                                                  (slab + index, key TWICE)
//! ```
//!
//! The stack stores the key twice purely to answer "where in the LRU order is
//! this key?" -- a question that disappears when the object and its links share
//! a slot, because finding the object IS finding its position.
//!
//! # What it keeps
//!
//! Everything `LruCompactHybridStack` needs, unchanged in meaning:
//!
//!   * `LruPayload` -- `tier`, `dram_resident`, `size`, still 8 bytes, but as
//!     FIELDS of the slot rather than the index value.
//!   * the fast/slow split. Two physical queues over one slab, `Q_FAST` and
//!     `Q_SLOW`, rather than a `fast_boundary` cursor. Main order is the
//!     concatenation, so a demotion moves one key across the seam and is a
//!     no-op on the order -- the same invariant the faithful S3-FIFO family
//!     uses, and it removes the cursor that stack maintains in six places.
//!   * the real key, because `key_matches` needs it to make hash collisions
//!     safe (`PaperCache::get`). That is why the slot's key is irreducible
//!     while the index's copy is not.
//!
//! # Two variants
//!
//! A -- slab + `HashMap<HashedKey, u32>`. One key copy, in the index.
//! B -- slab + open-addressed table of bare `u32`, resolving a probe candidate
//!      through `slots[i].key`. One key copy total, in the slot. Costs an extra
//!      indirection per probe step.

#![allow(dead_code)]

use std::sync::Arc;
use crate::{
	object::{ExpireTime, ObjectSize},
	worker::policy::policy_stack::Tier,
	CacheSize, HashedKey, NoHasher, TieredBuffer,
};

const NIL: u32 = u32::MAX;

/// Object, eviction links and tier payload in ONE slot.
///
/// Pinned so a regression is a build failure: 8 key + 8 Arc ptr + 16 expiry
/// + 4 size + 4 prev + 4 next + 1 tier + 1 dram_resident, padded to 48.
struct MergedSlot {
	key: HashedKey,
	data: Arc<TieredBuffer>,
	expiry: ExpireTime,
	size: ObjectSize,
	prev: u32,
	next: u32,
	tier: Tier,
	dram_resident: u8,
}

const _: () = assert!(
	std::mem::size_of::<MergedSlot>() == 48,
	"MergedSlot grew past 48 bytes",
);

/// Variant A: slab + a conventional key-bearing index.
pub struct MergedLruHybrid {
	index: std::collections::HashMap<HashedKey, u32, NoHasher>,
	slots: Vec<MergedSlot>,
	free: Vec<u32>,

	fast_head: u32,
	fast_tail: u32,
	slow_head: u32,
	slow_tail: u32,

	fast_capacity: CacheSize,
	fast_used: CacheSize,
	slow_used: CacheSize,
}

impl MergedLruHybrid {
	pub fn new(fast_capacity: CacheSize) -> Self {
		MergedLruHybrid {
			index: std::collections::HashMap::with_hasher(NoHasher::default()),
			slots: Vec::new(),
			free: Vec::new(),
			fast_head: NIL,
			fast_tail: NIL,
			slow_head: NIL,
			slow_tail: NIL,
			fast_capacity,
			fast_used: 0,
			slow_used: 0,
		}
	}

	pub fn len(&self) -> usize {
		self.index.len()
	}

	fn migrating(s: &MergedSlot) -> CacheSize {
		(s.size as CacheSize).saturating_sub(s.dram_resident as CacheSize)
	}

	fn push_fast_front(&mut self, i: u32) {
		let old = self.fast_head;
		self.slots[i as usize].prev = NIL;
		self.slots[i as usize].next = old;
		if old != NIL {
			self.slots[old as usize].prev = i;
		}
		self.fast_head = i;
		if self.fast_tail == NIL {
			self.fast_tail = i;
		}
	}

	/// Fast's back to slow's front -- across the seam, so the concatenated LRU
	/// order is unchanged and no eviction decision moves.
	fn demote_one(&mut self) -> bool {
		let i = self.fast_tail;
		if i == NIL {
			return false;
		}
		let p = self.slots[i as usize].prev;
		self.fast_tail = p;
		if p != NIL {
			self.slots[p as usize].next = NIL;
		} else {
			self.fast_head = NIL;
		}

		let old = self.slow_head;
		self.slots[i as usize].prev = NIL;
		self.slots[i as usize].next = old;
		if old != NIL {
			self.slots[old as usize].prev = i;
		}
		self.slow_head = i;
		if self.slow_tail == NIL {
			self.slow_tail = i;
		}

		let bytes = Self::migrating(&self.slots[i as usize]);
		self.slots[i as usize].tier = Tier::Slow;
		self.fast_used = self.fast_used.saturating_sub(bytes);
		self.slow_used += bytes;
		true
	}

	fn settle_fast(&mut self) {
		while self.fast_used > self.fast_capacity && self.demote_one() {}
	}

	pub fn insert(&mut self, key: HashedKey, data: Arc<TieredBuffer>, size: ObjectSize) {
		if let Some(&i) = self.index.get(&key) {
			self.slots[i as usize].data = data;
			return;
		}
		let slot = MergedSlot {
			key,
			data,
			expiry: None,
			size,
			prev: NIL,
			next: NIL,
			tier: Tier::Fast,
			dram_resident: 0,
		};
		let i = match self.free.pop() {
			Some(i) => {
				self.slots[i as usize] = slot;
				i
			},
			None => {
				self.slots.push(slot);
				(self.slots.len() - 1) as u32
			},
		};
		self.fast_used += Self::migrating(&self.slots[i as usize]);
		self.push_fast_front(i);
		self.index.insert(key, i);
		self.settle_fast();
	}

	/// The whole point: one lookup lands on the object AND its LRU position.
	/// No second structure, no second key comparison, no second cache miss.
	pub fn touch(&mut self, key: HashedKey) -> Option<&Arc<TieredBuffer>> {
		let i = *self.index.get(&key)?;
		Some(&self.slots[i as usize].data)
	}
}

/// Variant B: slab + open-addressed table of bare `u32` slot indices.
pub struct MergedLruHybridThin {
	table: Vec<u32>,
	mask: usize,
	slots: Vec<MergedSlot>,
	live: usize,
	head: u32,
	tail: u32,
}

impl MergedLruHybridThin {
	pub fn with_capacity(cap: usize) -> Self {
		let n = (cap * 2).next_power_of_two().max(1024);
		MergedLruHybridThin {
			table: vec![NIL; n],
			mask: n - 1,
			slots: Vec::new(),
			live: 0,
			head: NIL,
			tail: NIL,
		}
	}

	pub fn len(&self) -> usize {
		self.live
	}

	/// `HashedKey` is already a hash, so the low bits index directly. A
	/// candidate is resolved by reading `slots[c].key` -- the indirection this
	/// trades for the index's 8-byte key copy.
	fn probe(&self, key: HashedKey) -> usize {
		let mut i = (key as usize) & self.mask;
		loop {
			let c = self.table[i];
			if c == NIL || self.slots[c as usize].key == key {
				return i;
			}
			i = (i + 1) & self.mask;
		}
	}

	pub fn insert(&mut self, key: HashedKey, data: Arc<TieredBuffer>, size: ObjectSize) {
		let b = self.probe(key);
		if self.table[b] != NIL {
			return;
		}
		self.slots.push(MergedSlot {
			key,
			data,
			expiry: None,
			size,
			prev: NIL,
			next: self.head,
			tier: Tier::Fast,
			dram_resident: 0,
		});
		let i = (self.slots.len() - 1) as u32;
		if self.head != NIL {
			self.slots[self.head as usize].prev = i;
		}
		self.head = i;
		if self.tail == NIL {
			self.tail = i;
		}
		self.table[b] = i;
		self.live += 1;
	}
}

/// Same method as every other measurement here: jemalloc `stats.allocated`,
/// ONE point per process, caller samples at powers of two.
///
/// Baselines measured by the same harness, for the same 64-byte value:
///   object map row  96 B  (`measure_object_map_point`)
///   Arc header      48 B
///   flat lru-compact stack 56 B; the compact HYBRID analog measured 66 B
#[cfg(test)]
mod measure {
	use super::*;
	use crate::worker::policy::policy_stack::measure_overhead::allocated_bytes;

	#[test]
	#[ignore]
	fn measure_merged_point() {
		let n: u64 = match std::env::var("MERGED_N") {
			Ok(v) => v.parse().expect("MERGED_N"),
			Err(_) => return,
		};
		let vsize: usize = std::env::var("MERGED_VALUE")
			.map(|v| v.parse().expect("MERGED_VALUE"))
			.unwrap_or(64);
		let variant = std::env::var("MERGED_VARIANT").unwrap_or_else(|_| "a".into());
		let value = vec![0u8; vsize];

		let base = allocated_bytes();
		if variant == "b" {
			let mut m = MergedLruHybridThin::with_capacity(n as usize);
			for i in 0..n {
				let k = i.wrapping_mul(0x9E37_79B9_7F4A_7C15);
				m.insert(k, Arc::new(TieredBuffer::new_fast(&value)), vsize as ObjectSize);
			}
			let after = allocated_bytes();
			core::hint::black_box(&m);
			println!("MERGED b {} {} {}", n, vsize, after.saturating_sub(base));
		} else {
			let mut m = MergedLruHybrid::new(CacheSize::MAX / 4);
			for i in 0..n {
				let k = i.wrapping_mul(0x9E37_79B9_7F4A_7C15);
				m.insert(k, Arc::new(TieredBuffer::new_fast(&value)), vsize as ObjectSize);
			}
			let after = allocated_bytes();
			core::hint::black_box(&m);
			println!("MERGED a {} {} {}", n, vsize, after.saturating_sub(base));
		}
	}
}
