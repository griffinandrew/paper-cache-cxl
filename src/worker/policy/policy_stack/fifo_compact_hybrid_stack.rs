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
//! one row each per object. This keeps one [`ArenaQueueSet`].
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
const Q_FIFO: usize = 0;

/// Per-key bookkeeping is [`NodePayload`], the one node every policy shares.
/// This stack reads `tier`, `size` and `dram_resident`; `freq`, `ts`, `queue`
/// and `phys` belong to other policies and stay at their defaults here.
pub struct FifoCompactHybridStack {
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

	/// S5: the measured M the policy worker pushed (`set_dram_metadata`),
	/// reserved instead of `len x shared_overhead`; `None` under the
	/// per-object model.
	measured: Option<CacheSize>,
}

impl FifoCompactHybridStack {
	pub fn new(fast_capacity: CacheSize) -> Self {
		FifoCompactHybridStack {
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

	/// Takes a FAST key out of the fast set IN PLACE (S5): an overwrite with a
	/// value larger than an empty fast tier -- FIFO keeps an overwritten key
	/// where it is. Pushed `(key, Slow)`: its placement changed.
	fn demote_in_place(&mut self, key: HashedKey) {
		let Some(payload) = self.list.payload(key) else { return };
		let size = payload.migrating();

		if self.fast_boundary == Some(key) {
			self.fast_boundary = prev_fast(&self.list, key);
		}

		if let Some(slot) = self.list.payload_mut(key) {
			slot.tier = Some(Tier::Slow);
		}

		self.fast_used = self.fast_used.saturating_sub(size);
		self.fast_count = self.fast_count.saturating_sub(1);
		self.slow_used += size;

		self.migrations.push((key, Tier::Slow));
	}

	/// A `Set`, with the client's placement (S5). An existing key is resized in
	/// place and NOT moved -- insertion order is eviction order -- and, when
	/// the new value is STRUCTURAL and the key fast, taken out of the fast set
	/// in place (`demote_in_place`); re-settling matters only if it was fast
	/// and resized, as before. A new key goes to the front, fast -- or, when
	/// structural, slow, with nothing pushed or settled. Returns the placement
	/// applied.
	fn insert_with(&mut self, key: HashedKey, size: ObjectSize, dram_resident: ObjectSize, placement: Placement) -> Placement {
		let dram_resident = narrow_resident(dram_resident);
		let migrating = (size as CacheSize).saturating_sub(dram_resident as CacheSize);
		let structural = placement == Placement::Structural || self.structural(migrating);

		if let Some(payload) = self.list.payload(key) {
			let tier = payload.tier;
			let resized = payload.size != size;

			if resized {
				self.resize_key(key, size, dram_resident);
			}

			if structural && tier == Some(Tier::Fast) {
				self.demote_in_place(key);
			}

			if resized && tier == Some(Tier::Fast) {
				self.settle_fast_tier();
			}

			return placed(structural);
		}

		let tier = match structural {
			true => Tier::Slow,
			false => Tier::Fast,
		};

		self.list.push_front(Q_FIFO, key, NodePayload {
			size,
			dram_resident,
			tier: Some(tier),
			phys: Some(tier),
			freq: 0,
			ts: 0,
			queue: 0,
		});

		if structural {
			self.slow_used += migrating;
			return Placement::Structural;
		}

		self.fast_used += migrating;
		self.fast_count += 1;

		if self.fast_boundary.is_none() {
			self.fast_boundary = Some(key);
		}

		self.settle_fast_tier();

		Placement::Normal
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

	fn remove(&mut self, key: HashedKey) {
		let Some(slot) = self.list.payload(key) else { return };
		let size = slot.migrating();
		let tier = slot.tier;

		let new_boundary_if_needed = if tier == Some(Tier::Fast) && self.fast_boundary == Some(key) {
			prev_fast(&self.list, key)
		} else {
			None
		};

		self.list.remove(Q_FIFO, key);

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
		let key = self.list.back(Q_FIFO)?;
		let slot = self.list.remove(Q_FIFO, key)?;
		let size = slot.migrating();

		match slot.tier {
			Some(Tier::Fast) => {
				self.fast_used = self.fast_used.saturating_sub(size);
				self.fast_count = self.fast_count.saturating_sub(1);

				// The boundary was the tail: the nearest fast key from the
				// new tail (S5: past any structural ones).
				if self.fast_boundary == Some(key) {
					self.fast_boundary = fast_at_or_before(&self.list, self.list.back(Q_FIFO));
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

	/// `tier_of`: a key is admitted fast and only ever leaves the fast prefix
	/// through `settle_fast_tier`, which pushes the demotion; a re-set moves
	/// nothing.
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

/// The fast tier is charged the metadata of EVERY tracked object, not only the
/// fast ones. A reservation of `fast_object_count() x shared_overhead`
/// understates DRAM by the whole slow tier's metadata and fails this test.
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
		let mut stack = FifoCompactHybridStack::new(FAST_CAPACITY).with_shared_overhead(OVERHEAD);

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
}
