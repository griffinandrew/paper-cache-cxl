/*
 * Copyright (c) Kia Shakiba
 *
 * This source code is licensed under the GNU AGPLv3 license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! Slab-backed FIFO hybrid: `FifoHybridStack` with one structure where that
//! has two.
//!
//! `FifoHybridStack` keeps a `kwik::HashList`, which owns its own key-to-node
//! index, plus a separate `entries` map for the 8-byte payload. Two indexes,
//! one row each per object. This keeps one [`CompactQueueSet`].
//!
//! Identical to [`LruCompactHybridStack`] except that a hit does NOT reorder.
//! That is the whole of FIFO: insertion order IS eviction order, so there is no
//! `touch_fast_key`, no `update` override -- the trait default no-op is CORRECT
//! here and overriding it would silently turn this into LRU -- and an
//! `insert_resident` on an existing key only resizes, re-settling if the key is
//! fast, rather than moving it to the front.
//!
//! Everything else is shared: one queue spanning both tiers with the newest end
//! fast, `fast_boundary` naming the oldest fast key, and demotion stepping that
//! boundary one place toward the newest end per victim.

use crate::{
	object::ObjectSize,
	worker::policy::policy_stack::{
		compact_queue_set::CompactQueueSet, narrow_resident, watermarks, CacheSize,
		HashedKey, PolicyStack, Tier,
	},
	PaperPolicy,
};

/// The single recency order, in the shared queue set's slot 0.
const Q_FIFO: usize = 0;

/// Per-key bookkeeping, carried in the index value.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct FifoPayload {
	tier: Tier,
	/// The part of `size` that stays in DRAM in either tier; see `migrating`.
	dram_resident: u8,
	size: ObjectSize,
}

const _: () = assert!(
	std::mem::size_of::<FifoPayload>() == 8,
	"FifoPayload grew past 8 bytes",
);

impl FifoPayload {
	fn migrating(&self) -> CacheSize {
		(self.size as CacheSize).saturating_sub(self.dram_resident as CacheSize)
	}
}

pub struct FifoCompactHybridStack {
	list: CompactQueueSet<FifoPayload>,

	fast_capacity: CacheSize,
	fast_used: CacheSize,
	slow_used: CacheSize,

	shared_overhead: CacheSize,

	fast_count: usize,

	/// The least-recently-used FAST key: everything from the MRU end up to and
	/// including this key is fast, everything after it is slow.
	fast_boundary: Option<HashedKey>,

	migrations: Vec<(HashedKey, Tier)>,
}

impl FifoCompactHybridStack {
	pub fn new(fast_capacity: CacheSize) -> Self {
		FifoCompactHybridStack {
			list: CompactQueueSet::default(),
			fast_capacity,
			fast_used: 0,
			slow_used: 0,
			shared_overhead: 0,
			fast_count: 0,
			fast_boundary: None,
			migrations: Vec::new(),
		}
	}

	/// Per-object DRAM reserved from the fast tier for shared metadata.
	///
	pub fn with_shared_overhead(mut self, overhead: CacheSize) -> Self {
		self.shared_overhead = overhead;


		self
	}

	pub fn fast_capacity(&self) -> CacheSize {
		self.fast_capacity
	}

	fn reserved_overhead(&self) -> CacheSize {
		self.list.len() as CacheSize * self.shared_overhead
	}

	pub fn tier_of(&self, key: HashedKey) -> Option<Tier> {
		self.list.payload(key).map(|p| p.tier)
	}

	fn resize_key(&mut self, key: HashedKey, new_size: ObjectSize, new_resident: u8) {
		let Some(slot) = self.list.payload_mut(key) else { return };

		let old_migrating = slot.migrating();
		slot.size = new_size;
		slot.dram_resident = new_resident;
		let delta = slot.migrating() as i64 - old_migrating as i64;
		let tier = slot.tier;

		match tier {
			Tier::Fast => {
				self.fast_used = (self.fast_used as i64 + delta).max(0) as CacheSize;
			},

			Tier::Slow => {
				self.slow_used = (self.slow_used as i64 + delta).max(0) as CacheSize;
			},
		}
	}

	/// Demotes from the tier boundary until `fast_used` is back under the low
	/// watermark. The victim is always `fast_boundary` -- the least-recently-
	/// used fast key -- so nothing is searched.
	fn settle_fast_tier(&mut self) {
		let effective = self.fast_capacity.saturating_sub(self.reserved_overhead());

		if self.fast_used <= watermarks::high_bytes(effective) {
			return;
		}

		let drain_target = watermarks::low_bytes(effective);

		while self.fast_used > drain_target {
			let Some(demote_key) = self.fast_boundary else { break };
			let size = self.list.payload(demote_key).map(|p| p.migrating()).unwrap_or(0);
			let new_boundary = self.list.before(demote_key);

			if let Some(slot) = self.list.payload_mut(demote_key) {
				slot.tier = Tier::Slow;
			}

			self.fast_used = self.fast_used.saturating_sub(size);
			self.fast_count = self.fast_count.saturating_sub(1);
			self.slow_used += size;
			self.fast_boundary = new_boundary;

			self.migrations.push((demote_key, Tier::Slow));
		}
	}
}

impl PolicyStack for FifoCompactHybridStack {
	fn is_policy(&self, policy: &PaperPolicy) -> bool {
		matches!(policy, PaperPolicy::FifoCompactHybrid)
	}

	fn len(&self) -> usize {
		self.list.len()
	}

	fn contains(&self, key: HashedKey) -> bool {
		self.list.contains(key)
	}

	fn insert(&mut self, key: HashedKey, size: ObjectSize) {
		self.insert_resident(key, size, 0);
	}

	fn insert_resident(&mut self, key: HashedKey, size: ObjectSize, dram_resident: ObjectSize) {
		let dram_resident = narrow_resident(dram_resident);

		// An existing key is resized in place and NOT moved: insertion order is
		// eviction order. Re-settling only matters if it is fast, since only
		// then can the resize have pushed the fast tier over its watermark.
		if let Some(payload) = self.list.payload(key) {
			if payload.size != size {
				let tier = payload.tier;
				self.resize_key(key, size, dram_resident);
				if tier == Tier::Fast {
					self.settle_fast_tier();
				}
			}
			return;
		}

		self.list.push_front(Q_FIFO, key, FifoPayload { tier: Tier::Fast, dram_resident, size });
		self.fast_used += (size as CacheSize).saturating_sub(dram_resident as CacheSize);
		self.fast_count += 1;

		if self.fast_boundary.is_none() {
			self.fast_boundary = Some(key);
		}

		self.settle_fast_tier();
	}

	fn remove(&mut self, key: HashedKey) {
		let Some(slot) = self.list.payload(key) else { return };
		let size = slot.migrating();
		let tier = slot.tier;

		let new_boundary_if_needed = if tier == Tier::Fast && self.fast_boundary == Some(key) {
			self.list.before(key)
		} else {
			None
		};

		self.list.remove(Q_FIFO, key);

		match tier {
			Tier::Fast => {
				self.fast_used = self.fast_used.saturating_sub(size);
				self.fast_count = self.fast_count.saturating_sub(1);

				if self.fast_boundary == Some(key) {
					self.fast_boundary = new_boundary_if_needed;
				}
			},

			Tier::Slow => {
				self.slow_used = self.slow_used.saturating_sub(size);
			},
		}
	}

	fn clear(&mut self) {
		self.list.clear();

		self.fast_used = 0;
		self.slow_used = 0;
		self.fast_count = 0;
		self.fast_boundary = None;
		self.migrations.clear();
	}

	fn evict_one(&mut self) -> Option<HashedKey> {
		let key = self.list.back(Q_FIFO)?;
		let slot = self.list.remove(Q_FIFO, key)?;
		let size = slot.migrating();

		match slot.tier {
			Tier::Fast => {
				self.fast_used = self.fast_used.saturating_sub(size);
				self.fast_count = self.fast_count.saturating_sub(1);

				if self.fast_boundary == Some(key) {
					self.fast_boundary = self.list.back(Q_FIFO);
				}
			},

			Tier::Slow => {
				self.slow_used = self.slow_used.saturating_sub(size);
			},
		}

		Some(key)
	}

	fn resize_fast_tier(&mut self, size: CacheSize) {
		self.fast_capacity = size;
		self.settle_fast_tier();
	}

	fn drain_tier_migrations(&mut self) -> Vec<(HashedKey, Tier)> {
		std::mem::take(&mut self.migrations)
	}

	fn dram_reserved_bytes(&self) -> CacheSize {
		self.reserved_overhead()
	}

	fn fast_bytes_used(&self) -> CacheSize {
		self.fast_used
	}

	fn slow_bytes_used(&self) -> CacheSize {
		self.slow_used
	}

	fn fast_object_count(&self) -> usize {
		self.fast_count
	}

	fn slow_object_count(&self) -> usize {
		self.list.len().saturating_sub(self.fast_count)
	}
}

/// Fidelity against `FifoHybridStack`, which this stack is a compaction of.
#[cfg(all(test, feature = "fifo_hybrid_cache"))]
mod fidelity_tests {
	use super::*;
	use crate::worker::policy::policy_stack::fifo_hybrid_stack::FifoHybridStack;

	fn skewed_ops() -> Vec<(HashedKey, ObjectSize)> {
		let mut ops = Vec::new();
		let mut x: u64 = 0x243F_6A88_85A3_08D3;
		for _ in 0..20_000 {
			x ^= x << 13;
			x ^= x >> 7;
			x ^= x << 17;
			let u = (x >> 11) as f64 / (1u64 << 53) as f64;
			ops.push((((u * u * 200.0) as u64) + 1, 1024));
		}
		ops
	}

	#[test]
	fn matches_fifo_hybrid_migration_for_migration() {
		let ops = skewed_ops();
		for cap in [8_192u64, 32_768, 131_072] {
			for overhead in [0u64, 112] {
				let mut a = FifoHybridStack::new(cap).with_shared_overhead(overhead);
				let mut b = FifoCompactHybridStack::new(cap).with_shared_overhead(overhead);
				let (mut ma, mut mb) = (Vec::new(), Vec::new());

				for (k, size) in &ops {
					if a.contains(*k) { a.update(*k); } else { a.insert(*k, *size); }
					if b.contains(*k) { b.update(*k); } else { b.insert(*k, *size); }
					ma.extend(a.drain_tier_migrations());
					mb.extend(b.drain_tier_migrations());
				}

				assert_eq!(ma, mb, "migrations diverge at cap {cap} overhead {overhead}");
				for (k, _) in ops.iter().take(500) {
					assert_eq!(a.tier_of(*k), b.tier_of(*k), "tier of {k} diverges");
				}
				assert_eq!(a.fast_object_count(), b.fast_object_count());
				assert_eq!(a.fast_bytes_used(), b.fast_bytes_used());
				assert_eq!(a.slow_bytes_used(), b.slow_bytes_used());
			}
		}
	}

	/// The defining FIFO property, and the one an over-eager port would break:
	/// a hit must NOT reorder. `PolicyStack::update` has a no-op default and
	/// this stack deliberately does not override it -- overriding it would
	/// silently turn this into LRU, and every migration test above would still
	/// pass on a workload without eviction pressure.
	#[test]
	fn a_hit_does_not_reorder_or_promote() {
		let mut a = FifoHybridStack::new(4_096).with_shared_overhead(0);
		let mut b = FifoCompactHybridStack::new(4_096).with_shared_overhead(0);
		for k in 1..=4u64 {
			a.insert(k, 1024);
			b.insert(k, 1024);
		}
		a.drain_tier_migrations();
		b.drain_tier_migrations();

		// touch the oldest key repeatedly; FIFO must not rescue it
		for _ in 0..10 {
			a.update(1);
			b.update(1);
		}
		assert!(a.drain_tier_migrations().is_empty(), "baseline reordered on a hit");
		assert!(b.drain_tier_migrations().is_empty(), "compact reordered on a hit");

		// the oldest key is still the first evicted, despite being the hottest
		assert_eq!(a.evict_one(), Some(1), "baseline did not evict the oldest");
		assert_eq!(b.evict_one(), Some(1), "compact did not evict the oldest");
	}

	#[test]
	fn evicts_in_the_same_order() {
		let ops = skewed_ops();
		let mut a = FifoHybridStack::new(32_768).with_shared_overhead(112);
		let mut b = FifoCompactHybridStack::new(32_768).with_shared_overhead(112);
		for (k, size) in &ops {
			if a.contains(*k) { a.update(*k); } else { a.insert(*k, *size); }
			if b.contains(*k) { b.update(*k); } else { b.insert(*k, *size); }
			a.drain_tier_migrations();
			b.drain_tier_migrations();
		}
		let mut ea = Vec::new();
		let mut eb = Vec::new();
		while let Some(k) = a.evict_one() { ea.push(k); }
		while let Some(k) = b.evict_one() { eb.push(k); }
		assert_eq!(ea, eb, "eviction order diverges");
	}

	/// Re-inserting an existing key resizes it in place without moving it, and
	/// re-settles only when it is fast.
	#[test]
	fn reinsert_resizes_without_reordering() {
		let mut a = FifoHybridStack::new(8_192).with_shared_overhead(0);
		let mut b = FifoCompactHybridStack::new(8_192).with_shared_overhead(0);
		for k in 1..=4u64 {
			a.insert(k, 512);
			b.insert(k, 512);
		}
		a.drain_tier_migrations();
		b.drain_tier_migrations();

		a.insert(1, 4096);
		b.insert(1, 4096);
		assert_eq!(a.drain_tier_migrations(), b.drain_tier_migrations());
		assert_eq!(a.fast_bytes_used(), b.fast_bytes_used(), "resize accounting diverges");
		assert_eq!(a.evict_one(), Some(1), "the resized key should still be oldest");
		assert_eq!(b.evict_one(), Some(1));
	}

	#[test]
	fn removal_and_resize_match() {
		let ops = skewed_ops();
		let mut a = FifoHybridStack::new(32_768).with_shared_overhead(112);
		let mut b = FifoCompactHybridStack::new(32_768).with_shared_overhead(112);
		let (mut ma, mut mb) = (Vec::new(), Vec::new());

		for (i, (k, size)) in ops.iter().enumerate() {
			if a.contains(*k) { a.update(*k); } else { a.insert(*k, *size); }
			if b.contains(*k) { b.update(*k); } else { b.insert(*k, *size); }
			if i % 97 == 0 {
				let victim = (i as u64 % 200) + 1;
				a.remove(victim);
				b.remove(victim);
			}
			if i == ops.len() / 2 {
				a.resize_fast_tier(8_192);
				b.resize_fast_tier(8_192);
			}
			ma.extend(a.drain_tier_migrations());
			mb.extend(b.drain_tier_migrations());
		}

		assert_eq!(ma, mb, "migrations diverge under removal and resize");
		assert_eq!(a.len(), b.len());
		assert_eq!(a.fast_object_count(), b.fast_object_count());
	}
}
