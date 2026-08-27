/*
 * Copyright (c) Kia Shakiba
 *
 * This source code is licensed under the GNU AGPLv3 license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! Slab-backed LRU hybrid: behaviourally identical to [`LruHybridStack`], with
//! one structure where that has two.
//!
//! `LruHybridStack` keeps a `HashList` -- which owns its own index -- plus a
//! separate `entries` map holding `tier`, `size` and `dram_resident`. Both are
//! keyed by the same `HashedKey`; both hold exactly one row per object. That
//! second map is measured at **40 B/object**: all-DRAM LRU, which has one list
//! and no `entries`, costs 72 B/object, and hybrid LRU costs 112.
//!
//! The payload lives in the INDEX MAP'S VALUE, not in the slab slot, so a
//! metadata read is one probe with the payload already in the bucket rather
//! than a probe plus a dereference into the slab.
//!
//! This reverses an earlier choice here. The slot layout was picked on the
//! grounds that it was smaller, and for LRU'''s 8-byte payload it is, by 0 to 8
//! B/object depending on hash load. But measured on a quiet machine, the index
//! layout is faster on BOTH paths, not just the metadata-only one:
//!
//! ```text
//! metadata read   77.3 ns -> 41.0 ns   (-47%)
//! move_front      392.6   -> 344.1     (-12%)
//! ```
//!
//! The list operation was the one that mattered and the one nobody had
//! measured -- an earlier attempt on a loaded machine reported no separation
//! above noise. `move_front` gets faster because the slab is denser with the
//! payload removed (16-byte slots against 24), so the pointer chase touches
//! fewer cache lines. Paying up to 8 B/object for 12% on the hot path is the
//! right trade.
//!
//! Sharing `CompactQueueSet` rather than keeping a second primitive follows
//! from that: it is already a layout-B slab, and LRU is its one-queue case.

use crate::{
	object::ObjectSize,
	worker::policy::policy_stack::{
		compact_queue_set::CompactQueueSet, narrow_resident, watermarks, CacheSize,
		HashedKey, PolicyStack, Tier,
	},
	PaperPolicy,
};

/// The single recency order, in the shared queue set's slot 0.
const Q_LRU: usize = 0;

/// Per-key bookkeeping, carried in the index value.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct LruPayload {
	tier: Tier,
	/// The part of `size` that stays in DRAM in either tier; see `migrating`.
	dram_resident: u8,
	size: ObjectSize,
}

const _: () = assert!(
	std::mem::size_of::<LruPayload>() == 8,
	"LruPayload grew past 8 bytes",
);

impl LruPayload {
	fn migrating(&self) -> CacheSize {
		(self.size as CacheSize).saturating_sub(self.dram_resident as CacheSize)
	}
}

pub struct LruCompactHybridStack {
	list: CompactQueueSet<LruPayload>,

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

impl LruCompactHybridStack {
	pub fn new(fast_capacity: CacheSize) -> Self {
		LruCompactHybridStack {
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
	/// Also pre-sizes the slab, because this is the first point at which both
	/// the budget and the per-object cost are known. Every object costs
	/// `overhead` bytes of fast-tier metadata whichever tier its value sits in,
	/// so `fast_capacity / overhead` is a hard ceiling on the entry count, not
	/// an estimate. Reserving it means the slab never reallocates and never
	/// pays the copy; untouched pages are not resident, so an over-estimate
	/// costs address space rather than memory.
	pub fn with_shared_overhead(mut self, overhead: CacheSize) -> Self {
		self.shared_overhead = overhead;

		if overhead > 0 {
			// Capped: the ceiling is sound but unbounded, and reserving it
			// outright asks for petabytes at large budgets. See
			// `MAX_PREALLOC_ENTRIES`.
			let ceiling = (self.fast_capacity / overhead) as usize;
			self.list.reserve(ceiling.min(super::MAX_PREALLOC_ENTRIES));
		}

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

	/// Faithful port of `LruHybridStack::touch_fast_key`.
	fn touch_fast_key(&mut self, key: HashedKey) {
		let previous_tier = self.list.payload(key).map(|p| p.tier);

		let already_at_front = self.list.front(Q_LRU) == Some(key);
		let is_boundary = self.fast_boundary == Some(key);

		// Read the neighbour BEFORE moving: once the key is at the front its
		// predecessor is gone, and the boundary has to step back to whatever
		// was in front of it.
		let new_boundary_if_moved = if is_boundary && !already_at_front {
			self.list.before(key)
		} else {
			None
		};

		self.list.move_front(Q_LRU, key);

		if is_boundary && !already_at_front {
			self.fast_boundary = new_boundary_if_moved;
		}

		let mut promoted = false;

		if previous_tier != Some(Tier::Fast) {
			if previous_tier == Some(Tier::Slow) {
				let size = self.list.payload(key).map(|p| p.migrating()).unwrap_or(0);
				self.slow_used = self.slow_used.saturating_sub(size);
				self.fast_used += size;
				self.fast_count += 1;
				promoted = true;
			}

			if let Some(slot) = self.list.payload_mut(key) {
				slot.tier = Tier::Fast;
			}

			if self.fast_boundary.is_none() {
				self.fast_boundary = Some(key);
			}
		}

		self.settle_fast_tier();

		// Pushed after settling and guarded on the key still being fast: a
		// tight budget can demote it straight back out within the same settle,
		// in which case that call already pushed the correct final entry.
		if promoted && self.list.payload(key).map(|p| p.tier) == Some(Tier::Fast) {
			self.migrations.push((key, Tier::Fast));
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

impl PolicyStack for LruCompactHybridStack {
	fn is_policy(&self, policy: &PaperPolicy) -> bool {
		matches!(policy, PaperPolicy::LruCompactHybrid)
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

		if self.list.contains(key) {
			self.resize_key(key, size, dram_resident);
			self.touch_fast_key(key);
			return;
		}

		self.list.push_front(Q_LRU, key, LruPayload { tier: Tier::Fast, dram_resident, size });
		self.fast_used += (size as CacheSize).saturating_sub(dram_resident as CacheSize);
		self.fast_count += 1;

		if self.fast_boundary.is_none() {
			self.fast_boundary = Some(key);
		}

		self.settle_fast_tier();
	}

	fn update(&mut self, key: HashedKey) {
		if self.list.contains(key) {
			self.touch_fast_key(key);
		}
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

		self.list.remove(Q_LRU, key);

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
		let key = self.list.back(Q_LRU)?;
		let slot = self.list.remove(Q_LRU, key)?;
		let size = slot.migrating();

		match slot.tier {
			Tier::Fast => {
				self.fast_used = self.fast_used.saturating_sub(size);
				self.fast_count = self.fast_count.saturating_sub(1);

				if self.fast_boundary == Some(key) {
					self.fast_boundary = self.list.back(Q_LRU);
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

#[cfg(test)]
mod prealloc_tests {
	use super::*;

	/// Constructing with a very large budget must not try to pre-allocate the
	/// whole theoretical ceiling. The first version of `with_shared_overhead`
	/// did, and aborted the process asking for 123 PB.
	#[test]
	fn a_huge_budget_does_not_preallocate_the_ceiling() {
		let stack = LruCompactHybridStack::new(u64::MAX / 4).with_shared_overhead(190);
		assert_eq!(stack.len(), 0);
	}

	#[test]
	fn a_realistic_budget_still_reserves() {
		let stack = LruCompactHybridStack::new(4 * 1024 * 1024 * 1024).with_shared_overhead(224);
		assert_eq!(stack.len(), 0);
	}
}

/// Fidelity against `LruHybridStack`, which this stack is a compaction of.
///
/// The two must be indistinguishable: same tier for every key, same migration
/// sequence in the same order, same eviction order. A miss ratio matching on a
/// trace is necessary but not sufficient -- it would not catch a counter firing
/// on the wrong path, which is the class of defect that produced a doubled
/// demotion count on the LFU conversion.
#[cfg(all(test, feature = "lru_hybrid_cache"))]
mod fidelity_tests {
	use super::*;
	use crate::worker::policy::policy_stack::lru_hybrid_stack::LruHybridStack;

	/// 200 keys biased toward low ids: enough pressure to exercise promotion,
	/// demotion and the tier boundary.
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

	fn replay(
		cap: CacheSize,
		overhead: CacheSize,
		ops: &[(HashedKey, ObjectSize)],
	) -> (Vec<(HashedKey, Tier)>, Vec<(HashedKey, Tier)>, Vec<Option<Tier>>, Vec<Option<Tier>>) {
		let mut a = LruHybridStack::new(cap).with_shared_overhead(overhead);
		let mut b = LruCompactHybridStack::new(cap).with_shared_overhead(overhead);
		let (mut ma, mut mb) = (Vec::new(), Vec::new());

		for (k, size) in ops {
			if a.contains(*k) { a.update(*k); } else { a.insert(*k, *size); }
			if b.contains(*k) { b.update(*k); } else { b.insert(*k, *size); }
			ma.extend(a.drain_tier_migrations());
			mb.extend(b.drain_tier_migrations());
		}

		let keys: Vec<HashedKey> = ops.iter().map(|(k, _)| *k).collect();
		let ta = keys.iter().map(|k| a.tier_of(*k)).collect();
		let tb = keys.iter().map(|k| b.tier_of(*k)).collect();
		(ma, mb, ta, tb)
	}

	#[test]
	fn matches_lru_hybrid_migration_for_migration() {
		let ops = skewed_ops();
		for cap in [8_192u64, 32_768, 131_072] {
			for overhead in [0u64, 64, 224] {
				let (ma, mb, ta, tb) = replay(cap, overhead, &ops);
				assert_eq!(ta, tb, "final tiers diverge at cap {cap} overhead {overhead}");
				assert_eq!(ma, mb, "migrations diverge at cap {cap} overhead {overhead}");
			}
		}
	}

	/// Eviction order is separate from migration order and nothing above
	/// exercises it: `evict_one` walks the LRU end and prefers slow.
	#[test]
	fn evicts_in_the_same_order() {
		let ops = skewed_ops();
		let mut a = LruHybridStack::new(32_768).with_shared_overhead(224);
		let mut b = LruCompactHybridStack::new(32_768).with_shared_overhead(224);
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
		assert_eq!(a.len(), b.len());
		assert_eq!(b.len(), 0);
	}

	/// Resizing, in both directions, with BRAND-NEW keys arriving afterwards.
	///
	/// The shape matters. On the LFU conversion the equivalent test passed with
	/// a real bug present, because the workload drew from a fixed key set and
	/// every key already existed by the time the resize happened -- so nothing
	/// a resize changes was observable.
	#[test]
	fn resizes_like_lru_hybrid() {
		for (start, resized) in [(65_536u64, 65_536u64), (131_072, 32_768), (32_768, 131_072)] {
			let mut a = LruHybridStack::new(start).with_shared_overhead(224);
			let mut b = LruCompactHybridStack::new(start).with_shared_overhead(224);
			let (mut ma, mut mb) = (Vec::new(), Vec::new());

			for i in 0..4_000u64 {
				let k = (i % 200) + 1;
				if a.contains(k) { a.update(k); } else { a.insert(k, 1024); }
				if b.contains(k) { b.update(k); } else { b.insert(k, 1024); }
				ma.extend(a.drain_tier_migrations());
				mb.extend(b.drain_tier_migrations());
			}

			a.resize_fast_tier(resized);
			b.resize_fast_tier(resized);
			ma.extend(a.drain_tier_migrations());
			mb.extend(b.drain_tier_migrations());

			for i in 0..2_000u64 {
				let k = 10_000 + i;
				a.insert(k, 1024);
				b.insert(k, 1024);
				ma.extend(a.drain_tier_migrations());
				mb.extend(b.drain_tier_migrations());
			}

			assert_eq!(ma, mb, "migrations diverge resizing {start} -> {resized}");
			for i in 0..2_000u64 {
				assert_eq!(
					a.tier_of(10_000 + i),
					b.tier_of(10_000 + i),
					"tier of new key {} diverges resizing {start} -> {resized}",
					10_000 + i
				);
			}
		}
	}

	/// Removal maintains the tier boundary; nothing above removes anything.
	#[test]
	fn removal_matches_including_boundary_maintenance() {
		let ops = skewed_ops();
		let mut a = LruHybridStack::new(32_768).with_shared_overhead(224);
		let mut b = LruCompactHybridStack::new(32_768).with_shared_overhead(224);
		let (mut ma, mut mb) = (Vec::new(), Vec::new());

		for (i, (k, size)) in ops.iter().enumerate() {
			if a.contains(*k) { a.update(*k); } else { a.insert(*k, *size); }
			if b.contains(*k) { b.update(*k); } else { b.insert(*k, *size); }

			// remove a key periodically, including ones that may be the boundary
			if i % 97 == 0 {
				let victim = (i as u64 % 200) + 1;
				a.remove(victim);
				b.remove(victim);
			}

			ma.extend(a.drain_tier_migrations());
			mb.extend(b.drain_tier_migrations());
		}

		assert_eq!(ma, mb, "migrations diverge under removal");
		assert_eq!(a.len(), b.len(), "lengths diverge under removal");
		assert_eq!(a.fast_object_count(), b.fast_object_count(), "fast counts diverge");
		assert_eq!(a.fast_bytes_used(), b.fast_bytes_used(), "fast bytes diverge");
		assert_eq!(a.slow_bytes_used(), b.slow_bytes_used(), "slow bytes diverge");
	}
}
