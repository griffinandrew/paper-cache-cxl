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
		HashedKey, PolicyStack, Tier, Placement, SetEvent, fast_at_or_before, placed, prev_fast,
	},
	PaperPolicy,
};

/// The single recency order, in the shared queue set's slot 0.
const Q_LRU: usize = 0;

/// Per-key bookkeeping is [`NodePayload`], the one node every policy shares.
/// This stack reads `tier`, `size` and `dram_resident`; `freq`, `ts` and
/// `queue` belong to other policies and stay at their defaults here.
pub struct LruCompactHybridStack {
	list: ArenaQueueSet<NodePayload>,

	fast_capacity: CacheSize,
	fast_used: CacheSize,
	slow_used: CacheSize,

	shared_overhead: CacheSize,

	fast_count: usize,

	/// The least-recently-used FAST key: everything after it is slow, and
	/// every key from the MRU end up to it is fast -- but for STRUCTURAL keys
	/// (S5), which keep their place in the order with tier slow, and which
	/// every step of this cursor walks over (`prev_fast`).
	fast_boundary: Option<HashedKey>,

	migrations: Vec<(HashedKey, Tier)>,

	/// S5: the measured M the policy worker pushed (`set_dram_metadata`),
	/// reserved instead of `len x shared_overhead`; `None` under the
	/// per-object model.
	measured: Option<CacheSize>,
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
			measured: None,
		}
	}

	/// Per-object DRAM reserved from the fast tier for shared metadata.
	///
	pub fn with_shared_overhead(mut self, overhead: CacheSize) -> Self {
		self.shared_overhead = overhead;


		self
	}

	/// Metadata reservation for EVERY tracked key, fast or slow: a demotion
	/// moves the value and leaves the key's row, stack node and header in
	/// DRAM. See `PolicyStack::dram_reserved_bytes` for the rule, and for why
	/// a reservation at or over `fast_capacity` is left to saturate.
	fn reserved_overhead(&self) -> CacheSize {
		self.measured.unwrap_or(self.list.len() as CacheSize * self.shared_overhead)
	}

	/// This stack's eff (S5): the whole fast tier's budget for values, its
	/// settle's figure before the drain target.
	fn own_eff(&self) -> CacheSize {
		self.fast_capacity.saturating_sub(self.reserved_overhead())
	}

	/// Whether a value of `migrating` bytes is STRUCTURAL (S5): larger than an
	/// empty fast tier. Such a key is placed slow and never promoted while it
	/// stays that large. The stack's own check beside the client's flag, so a
	/// key the client placed normally just before eff moved is placed as the
	/// stack's own promotions would place it.
	fn structural(&self, migrating: CacheSize) -> bool {
		migrating > self.own_eff()
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

	/// Faithful port of `LruHybridStack::touch_fast_key` -- and, since S5, the
	/// structural rule: a STRUCTURAL key (its value larger than an empty fast
	/// tier) moves to the front all the same -- its place in the order -- but
	/// with tier slow. A slow one is not promoted; a fast one (an overwrite with
	/// a value too large, or an eff that shrank) leaves the fast set, pushed
	/// `(key, Slow)`: its placement changed, and a promotion of its old value
	/// may still be in flight, which this entry, behind it on the key's FIFO
	/// consumer, undoes.
	fn touch_fast_key(&mut self, key: HashedKey, structural: bool) {
		let previous_tier = self.list.payload(key).and_then(|p| p.tier);

		let already_at_front = self.list.front(Q_LRU) == Some(key);
		let is_boundary = self.fast_boundary == Some(key);

		// Read the neighbour BEFORE moving: once the key is at the front its
		// predecessor is gone, and the boundary has to step back to whatever
		// fast key was in front of it.
		let new_boundary_if_moved = if is_boundary && !already_at_front {
			prev_fast(&self.list, key)
		} else {
			None
		};

		self.list.move_front(Q_LRU, key);

		if is_boundary && !already_at_front {
			self.fast_boundary = new_boundary_if_moved;
		}

		if structural {
			if previous_tier == Some(Tier::Fast) {
				let size = self.list.payload(key).map(|p| p.migrating()).unwrap_or(0);
				self.fast_used = self.fast_used.saturating_sub(size);
				self.fast_count = self.fast_count.saturating_sub(1);
				self.slow_used += size;

				if let Some(slot) = self.list.payload_mut(key) {
					slot.tier = Some(Tier::Slow);
				}

				// Still the boundary only if it was already at the front: then
				// it was the one fast key, and none is left.
				if self.fast_boundary == Some(key) {
					self.fast_boundary = None;
				}

				self.migrations.push((key, Tier::Slow));
			}

			self.settle_fast_tier();
			return;
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
		}

		// The key is fast, at the front; with no fast key in front of it --
		// none at all, or (S5) only structural ones -- it is the boundary.
		if self.fast_boundary.is_none() {
			self.fast_boundary = Some(key);
		}

		self.settle_fast_tier();

		// Pushed after settling and guarded on the key still being fast: a
		// tight budget can demote it straight back out within the same settle,
		// in which case that call already pushed the correct final entry.
		//
		// Pushed even when the bytes are already fast, as they are on a re-`set`
		// of a slow key: the API thread built the new value in DRAM before this
		// worker saw the event, so the consumer will decline the entry. It still
		// has to be queued. Queued migrations carry no identity --
		// `apply_migration` acts on whatever object holds the key when it
		// dequeues -- so a demotion this stack decided for the old object, still
		// queued when the new one replaced it, demotes the new one; this entry,
		// behind it on the key's FIFO consumer, restores it. Pinned by
		// `overwrite_tests::an_overwrite_is_repromoted_after_a_stale_demotion`.
		if promoted && self.list.payload(key).and_then(|p| p.tier) == Some(Tier::Fast) {
			self.migrations.push((key, Tier::Fast));
		}
	}

	/// A `Set`, with the client's placement (S5): an existing key is an access
	/// (`touch_fast_key`); a new one is admitted at the front, fast -- or, when
	/// STRUCTURAL (the client's flag, or this stack's own check), slow: built
	/// slow, charged slow, never the boundary, nothing pushed (the reconcile's
	/// new-key rule covers a stale entry of the key), nothing settled (no fast
	/// byte moved). Returns the placement applied.
	fn insert_with(&mut self, key: HashedKey, size: ObjectSize, dram_resident: ObjectSize, placement: Placement) -> Placement {
		let dram_resident = narrow_resident(dram_resident);
		let migrating = (size as CacheSize).saturating_sub(dram_resident as CacheSize);
		let structural = placement == Placement::Structural || self.structural(migrating);

		if self.list.contains(key) {
			self.resize_key(key, size, dram_resident);
			self.touch_fast_key(key, structural);
			return placed(structural);
		}

		if structural {
			self.list.push_front(Q_LRU, key, NodePayload {
				size,
				dram_resident,
				tier: Some(Tier::Slow),
				freq: 0,
				ts: 0,
				queue: 0,
			});
			self.slow_used += migrating;

			return Placement::Structural;
		}

		self.list.push_front(Q_LRU, key, NodePayload {
			size,
			dram_resident,
			tier: Some(Tier::Fast),
			freq: 0,
			ts: 0,
			queue: 0,
		});
		self.fast_used += migrating;
		self.fast_count += 1;

		if self.fast_boundary.is_none() {
			self.fast_boundary = Some(key);
		}

		self.settle_fast_tier();

		Placement::Normal
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
			let new_boundary = prev_fast(&self.list, demote_key);

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
		self.insert_with(key, size, dram_resident, Placement::Normal);
	}

	fn insert_placed(
		&mut self,
		key: HashedKey,
		size: ObjectSize,
		dram_resident: ObjectSize,
		_event: SetEvent,
		placement: Placement,
	) -> Placement {
		self.insert_with(key, size, dram_resident, placement)
	}

	fn set_dram_metadata(&mut self, measured: Option<CacheSize>) {
		self.measured = measured;
	}

	fn resettle(&mut self) {
		self.settle_fast_tier();
	}

	fn update(&mut self, key: HashedKey) {
		if let Some(payload) = self.list.payload(key) {
			let structural = self.structural(payload.migrating());
			self.touch_fast_key(key, structural);
		}
	}

	fn remove(&mut self, key: HashedKey) {
		let Some(slot) = self.list.payload(key) else { return };
		let size = slot.migrating();
		let tier = slot.tier;

		let new_boundary_if_needed = if tier == Some(Tier::Fast) && self.fast_boundary == Some(key) {
			prev_fast(&self.list, key)
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

				// The boundary was the tail: the nearest fast key from the
				// new tail (S5: past any structural ones).
				if self.fast_boundary == Some(key) {
					self.fast_boundary = fast_at_or_before(&self.list, self.list.back(Q_LRU));
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

	/// `tier_of`: `settle_fast_tier` pushes every demotion and `touch_fast_key`
	/// pushes the promotion after its settle, guarded on the key still being
	/// fast -- also on a re-set, where the bytes are already fast, to follow
	/// a stale queued demotion.
	/// See `PolicyStack::placement_of`.
	fn placement_of(&self, key: HashedKey) -> Option<Tier> {
		self.tier_of(key)
	}

	fn drain_tier_migrations(&mut self) -> Vec<(HashedKey, Tier)> {
		std::mem::take(&mut self.migrations)
	}

	fn structure_bytes(&self) -> Option<crate::meta::NodeBytes> {
		Some(crate::meta::NodeBytes::stack(self.list.allocated_bytes()))
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

/// The re-promotion `touch_fast_key` queues on a re-`set` of a slow key looks
/// redundant -- the bytes are already in DRAM -- and is not. This replays the
/// one order in which it is load-bearing, a migration at a time, the way the
/// key's FIFO consumer would apply them.
///
/// Gated on `hybrid_cache_common` for `new_hybrid_object_map`, so it runs over
/// whichever object map the build selects; the stack is the split LRU stack in
/// every build.
#[cfg(all(test, feature = "hybrid_cache_common"))]
mod overwrite_tests {
	use super::*;
	use crate::{object::Object, worker::policy::migration_queue::apply_migration};

	// `MergedStore` has `insert` and `get_ref` of its own; the other maps take
	// them from the trait.
	#[cfg(not(feature = "merged_object_store"))]
	use crate::object_store::ObjectStore;

	const K: HashedKey = 7;
	const A: HashedKey = 8;
	const SIZE: ObjectSize = 1_000;

	fn fresh(key: HashedKey) -> Object<u32, crate::TieredBuffer> {
		Object::new_in(key as u32, &[0xA5; 64], Tier::Fast, None)
	}

	/// K is demoted as the LRU tail, and that demotion is decided but still
	/// queued when a re-`set` replaces K with a value built in DRAM. The
	/// queued demotion then lands on the NEW value, because migrations carry
	/// no identity; the re-promotion queued behind it must bring it back, or
	/// K's fresh value sits in the slow tier while the stack counts it fast.
	#[test]
	fn an_overwrite_is_repromoted_after_a_stale_demotion() {
		// `apply_migration` bumps the process-wide migration counters that the
		// queue tests assert exact deltas on, so this runs under their lock.
		let _serialised = crate::worker::policy::migration_test_lock::lock();

		let objects: crate::ObjectMapRef<u32, crate::TieredBuffer> = crate::new_hybrid_object_map();

		// Room for one SIZE-byte object: the second admission demotes the first.
		let mut stack = LruCompactHybridStack::new(2 * SIZE as CacheSize - 1).with_shared_overhead(0);

		for key in [K, A] {
			objects.insert(key, fresh(key)); // `set`, on the API thread
			stack.insert_resident(key, SIZE, 0); // its `Set` event, on the worker
		}

		assert_eq!(stack.tier_of(K), Some(Tier::Slow), "K should be the demoted LRU tail");

		let mut queue = stack.drain_tier_migrations();
		assert_eq!(queue, vec![(K, Tier::Slow)], "the demotion is decided, not yet applied");

		// The overwrite: an LRU `set` builds the new value in DRAM, then the
		// worker handles its `Set`.
		objects.insert(K, fresh(K));
		stack.insert_resident(K, SIZE, 0);
		queue.extend(stack.drain_tier_migrations());

		for (key, tier) in queue {
			apply_migration(&objects, key, tier);
		}

		let physical = objects.get_ref(&K).map(|object| object.value().tier());

		assert_eq!(stack.tier_of(K), Some(Tier::Fast), "an overwrite makes K the most recent key");
		assert_eq!(
			physical,
			Some(Tier::Fast),
			"K's new value was left in the slow tier while the stack counts it fast",
		);
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

/// The fast tier is charged the metadata of EVERY tracked object, not only the
/// fast ones: a demotion moves the value and leaves the row, the stack node and
/// the header in DRAM. A reservation of `fast_object_count() x shared_overhead`
/// understates DRAM by the whole slow tier's metadata, and both tests fail
/// under it.
#[cfg(test)]
mod reservation_tests {
	use super::*;

	const FAST_CAPACITY: CacheSize = 10_000;
	const OVERHEAD: CacheSize = 200;
	const SIZE: ObjectSize = 1_000;
	const N: HashedKey = 20;

	/// 20 x 200 B of metadata leaves 6_000 B for values: five of the twenty
	/// objects stay fast. Charging the fast ones alone would keep eight.
	#[test]
	fn slow_objects_are_charged_against_the_fast_tier() {
		let mut stack = LruCompactHybridStack::new(FAST_CAPACITY).with_shared_overhead(OVERHEAD);

		for key in 1..=N {
			stack.insert(key, SIZE);
		}

		assert!(
			stack.slow_object_count() > 0 && stack.fast_object_count() > 0,
			"the fixture must leave objects in both tiers to tell the rules apart",
		);
		assert_eq!(
			stack.dram_reserved_bytes(),
			N * OVERHEAD,
			"all {N} tracked objects keep their metadata in DRAM, but the reservation \
			 covers {} of them ({} fast, {} slow)",
			stack.dram_reserved_bytes() / OVERHEAD,
			stack.fast_object_count(),
			stack.slow_object_count(),
		);

		let effective = FAST_CAPACITY - N * OVERHEAD;

		assert!(
			stack.fast_bytes_used() <= drain_target::bytes(effective),
			"{} B of values are fast against {effective} B left once all {N} objects' \
			 metadata is reserved",
			stack.fast_bytes_used(),
		);
	}

	/// Metadata at, then over, the whole fast tier: the effective budget
	/// saturates at 0, every value is slow, and the stack keeps working --
	/// hits, removal and eviction included. The reservation is reported as it
	/// is, above `fast_capacity`, rather than clipped to it.
	#[test]
	fn metadata_at_or_over_the_fast_tier_leaves_no_room_for_values() {
		const OVERHEAD: CacheSize = 1_000;
		const SIZE: ObjectSize = 100;

		for n in [10, 12] {
			let mut stack =
				LruCompactHybridStack::new(FAST_CAPACITY).with_shared_overhead(OVERHEAD);

			for key in 1..=n {
				stack.insert(key, SIZE);
			}

			assert_eq!(
				stack.dram_reserved_bytes(),
				n * OVERHEAD,
				"{n} objects: the reservation must be every object's metadata, \
				 even past the fast tier",
			);
			assert_eq!(
				stack.fast_object_count(),
				0,
				"{n} objects: {} B of metadata on a {FAST_CAPACITY} B fast tier leaves \
				 no room for a value, yet {} are fast",
				n * OVERHEAD,
				stack.fast_object_count(),
			);
			assert_eq!(stack.fast_bytes_used(), 0);
			assert_eq!(stack.slow_object_count(), n as usize);

			for key in 1..=n {
				stack.update(key);
			}

			assert_eq!(stack.fast_object_count(), 0, "a hit cannot find fast room that is not there");

			stack.remove(1);
			assert_eq!(stack.evict_one(), Some(2), "the LRU order still holds");
			assert_eq!(stack.len(), n as usize - 2);
			assert_eq!(stack.dram_reserved_bytes(), (n - 2) * OVERHEAD);
		}
	}
}
