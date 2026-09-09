/*
 * Copyright (c) Kia Shakiba
 *
 * This source code is licensed under the GNU AGPLv3 license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! Slab-backed LRU hybrid: behaviourally identical to `LruHybridStack`, with
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
//!
//! **The baseline named above no longer exists in this crate.** Every
//! non-compact hybrid stack was removed once its compact twin was shown
//! behaviourally identical at 72 B/object of eviction stack instead of 112.
//! References to it here are historical: they say what this design is a
//! compaction OF, and they are the reason the structure looks the way it
//! does. Git history holds the baseline and the differential tests that
//! proved the two agreed.

use crate::{
	object::ObjectSize,
	worker::policy::policy_stack::{
		arena_queue_set::{ArenaQueueSet, NodePayload}, narrow_resident, drain_target, CacheSize,
		HashedKey, PolicyStack, Tier,
	},
	PaperPolicy,
};

/// The single recency order, in the shared queue set's slot 0.
const Q_LRU: usize = 0;

/// Per-key bookkeeping is [`NodePayload`], the one node every policy shares.
/// This stack reads `tier`, `size` and `dram_resident`; `freq`, `ts`, `queue`
/// and `phys` belong to other policies and stay at their defaults here.
pub struct LruCompactHybridStack {
	list: ArenaQueueSet<NodePayload>,

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
			list: ArenaQueueSet::default(),
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
		self.list.payload(key).and_then(|p| p.tier)
	}

	fn resize_key(&mut self, key: HashedKey, new_size: ObjectSize, new_resident: u8) {
		let Some(slot) = self.list.payload_mut(key) else { return };

		let old_migrating = slot.migrating();
		slot.size = new_size;
		slot.dram_resident = new_resident;
		let delta = slot.migrating() as i64 - old_migrating as i64;
		let tier = slot.tier;

		match tier {
			Some(Tier::Fast) => {
				self.fast_used = (self.fast_used as i64 + delta).max(0) as CacheSize;
			},

			Some(Tier::Slow) => {
				self.slow_used = (self.slow_used as i64 + delta).max(0) as CacheSize;
			},
			// The shared node makes `tier` optional because the 2Q and S3-FIFO
			// families legitimately have a queue with no tier of its own. This
			// stack always records one, so this arm is unreachable -- and it is
			// spelled out rather than papered over with `unwrap_or`, which
			// would silently pick a tier if that ever stopped being true.
			None => {},
		}
	}

	/// Faithful port of `LruHybridStack::touch_fast_key`.
	fn touch_fast_key(&mut self, key: HashedKey) {
		let previous_tier = self.list.payload(key).and_then(|p| p.tier);

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
				slot.tier = Some(Tier::Fast);
			}

			if self.fast_boundary.is_none() {
				self.fast_boundary = Some(key);
			}
		}

		self.settle_fast_tier();

		// Pushed after settling and guarded on the key still being fast: a
		// tight budget can demote it straight back out within the same settle,
		// in which case that call already pushed the correct final entry.
		if promoted && self.list.payload(key).and_then(|p| p.tier) == Some(Tier::Fast) {
			self.migrations.push((key, Tier::Fast));
		}
	}

	/// Demotes from the tier boundary until `fast_used` is back within the
	/// effective budget. The victim is always `fast_boundary` -- the least-
	/// recently-used fast key -- so nothing is searched.
	fn settle_fast_tier(&mut self) {
		let effective = self.fast_capacity.saturating_sub(self.reserved_overhead());
		let target = drain_target::bytes(effective);

		while self.fast_used > target {
			let Some(demote_key) = self.fast_boundary else { break };
			let size = self.list.payload(demote_key).map(|p| p.migrating()).unwrap_or(0);
			let new_boundary = self.list.before(demote_key);

			if let Some(slot) = self.list.payload_mut(demote_key) {
				slot.tier = Some(Tier::Slow);
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

		self.list.push_front(Q_LRU, key, NodePayload {
			size,
			dram_resident,
			tier: Some(Tier::Fast),
			phys: Some(Tier::Fast),
			freq: 0,
			ts: 0,
			queue: 0,
		});
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

		let new_boundary_if_needed = if tier == Some(Tier::Fast) && self.fast_boundary == Some(key) {
			self.list.before(key)
		} else {
			None
		};

		self.list.remove(Q_LRU, key);

		match tier {
			Some(Tier::Fast) => {
				self.fast_used = self.fast_used.saturating_sub(size);
				self.fast_count = self.fast_count.saturating_sub(1);

				if self.fast_boundary == Some(key) {
					self.fast_boundary = new_boundary_if_needed;
				}
			},

			Some(Tier::Slow) => {
				self.slow_used = self.slow_used.saturating_sub(size);
			},
			// The shared node makes `tier` optional because the 2Q and S3-FIFO
			// families legitimately have a queue with no tier of its own. This
			// stack always records one, so this arm is unreachable -- and it is
			// spelled out rather than papered over with `unwrap_or`, which
			// would silently pick a tier if that ever stopped being true.
			None => {},
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
			Some(Tier::Fast) => {
				self.fast_used = self.fast_used.saturating_sub(size);
				self.fast_count = self.fast_count.saturating_sub(1);

				if self.fast_boundary == Some(key) {
					self.fast_boundary = self.list.back(Q_LRU);
				}
			},

			Some(Tier::Slow) => {
				self.slow_used = self.slow_used.saturating_sub(size);
			},
			// The shared node makes `tier` optional because the 2Q and S3-FIFO
			// families legitimately have a queue with no tier of its own. This
			// stack always records one, so this arm is unreachable -- and it is
			// spelled out rather than papered over with `unwrap_or`, which
			// would silently pick a tier if that ever stopped being true.
			None => {},
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
mod growth_tests {
	use super::*;

	/// These stacks grow dynamically and must not allocate from the cache
	/// budget at construction. An eager reservation sized from capacity was
	/// removed: at the standing 4 GiB fast tier it reserved 4.19M slots, which
	/// is 16x what the 16.5 KB-object eval trace can hold there, while on the
	/// real Twitter traces (~180 B objects, ~11.4M resident) it was too small
	/// to prevent doubling anyway. It also meant the shipped stack allocated
	/// something other than the 72 B/object that was measured and reported,
	/// since the measurement runs with the reservation disabled.
	///
	/// The old tests here asserted only `len() == 0`, which is true either
	/// way; neither observed capacity.
	#[test]
	fn construction_does_not_allocate_from_the_budget() {
		for budget in [u64::MAX / 4, 4 * 1024 * 1024 * 1024, 1_024] {
			let stack = LruCompactHybridStack::new(budget).with_shared_overhead(224);
			assert_eq!(stack.len(), 0, "budget {budget}: nothing is tracked yet");
			assert_eq!(
				stack.list.slab_capacity(),
				0,
				"budget {budget}: the slab must be empty at construction -- a stack \
				 that pre-sizes from capacity allocates for objects that may never \
				 arrive, and reserves a different amount than it was measured at",
			);
		}
	}

	/// The counterpart: growth still happens, it is just demand-driven.
	#[test]
	fn the_slab_grows_on_demand() {
		let mut stack = LruCompactHybridStack::new(u64::MAX / 4).with_shared_overhead(224);
		for i in 0..1_000u64 {
			stack.insert(i, 64);
		}
		assert_eq!(stack.len(), 1_000);
		assert!(
			stack.list.slab_capacity() >= 1_000,
			"the slab must have grown to hold what was inserted",
		);
	}
}
