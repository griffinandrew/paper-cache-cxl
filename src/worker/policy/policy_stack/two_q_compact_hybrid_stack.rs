/*
 * Copyright (c) Kia Shakiba
 *
 * This source code is licensed under the GNU AGPLv3 license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! Slab-backed 2Q hybrid: behaviourally identical to `TwoQHybridStack`, with
//! one structure where that has three.
//!
//! `TwoQHybridStack` keeps two `kwik::HashList`s -- a FIFO admission queue and
//! an LRU main queue, each owning its OWN key-to-node index -- plus a separate
//! `entries` map holding the combined payload. Three indexes, for a population
//! where every key is in exactly one of the two queues.
//!
//! Here a single [`ArenaQueueSet`] holds both orders over one slab, with the
//! payload in the slot itself. A promotion out of the FIFO becomes an unlink
//! and a relink of the same slot rather than a hash-indexed removal from one
//! list and an insertion into another.
//!
//! The queue mechanics are unchanged and deliberately so. Admission lands at
//! the front of the FIFO and is entirely slow-tier; a hit there promotes to the
//! front of main and to fast; `main_boundary` names the least-recently-used
//! fast key in main, and demotion steps it one place toward the MRU end per
//! victim. Terminal eviction prefers the FIFO tail, falling back to the main
//! tail. Nothing is searched for.
//!
//! **The baseline named above no longer exists in this crate.** Every
//! non-compact hybrid stack was removed once its compact twin was shown
//! behaviourally identical and cheaper: 72 B/object of eviction stack instead of
//! 112 then, and 40 since the arena conversion (`ARENA_STACK_DRAM_OVERHEAD`).
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

/// Queue slots in the shared set. The FIFO admission queue is 0, the LRU main
/// queue is 1; a key is in exactly one of them.
const Q_FIFO: usize = 0;
const Q_MAIN: usize = 1;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
enum Queue {
	Fifo = 0,
	Main = 1,
}

impl Queue {
	/// The shared node stores `queue` as a bare `u8`, so this is the one
	/// place the tag becomes an enum again. Every write goes the other way
	/// through `Queue as u8`, which is why the last arm cannot be reached.
	#[inline]
	fn from_u8(tag: u8) -> Queue {
		match tag {
			0 => Queue::Fifo,
			1 => Queue::Main,
			_ => unreachable!("2q-compact-hybrid queue tag out of range: {tag}"),
		}
	}
}

/// Per-key bookkeeping is [`NodePayload`], the one node every policy shares.
/// This stack reads `queue`, `tier`, `size` and `dram_resident`; `freq` and `ts`
/// belong to other policies and stay at their defaults here.
///
/// `tier` is `None` while `queue == Fifo`: the FIFO is entirely slow-tier, so a
/// key there has no tier of its own to record. `queue` is a bare `u8` in the
/// shared node, so [`Queue`] converts at the boundary.
pub struct TwoQCompactHybridStack {
	queues: ArenaQueueSet<NodePayload>,

	k_in: f64,

	fifo_capacity: CacheSize,
	fifo_used: CacheSize,

	fast_capacity: CacheSize,
	fast_used: CacheSize,
	slow_used: CacheSize,

	shared_overhead: CacheSize,

	fast_count: usize,
	main_count: usize,

	/// The least-recently-used FAST key in the main queue.
	main_boundary: Option<HashedKey>,

	migrations: Vec<(HashedKey, Tier)>,

	/// S5: the measured M the policy worker pushed (`set_dram_metadata`),
	/// reserved instead of the per-object reservation; `None` under the
	/// per-object model.
	measured: Option<CacheSize>,
}

impl TwoQCompactHybridStack {
	pub fn new(k_in: f64, max_size: CacheSize, fast_capacity: CacheSize) -> Self {
		TwoQCompactHybridStack {
			queues: ArenaQueueSet::default(),
			k_in,
			fifo_capacity: (k_in * max_size as f64) as CacheSize,
			fifo_used: 0,
			fast_capacity,
			fast_used: 0,
			slow_used: 0,
			shared_overhead: 0,
			fast_count: 0,
			main_count: 0,
			main_boundary: None,
			migrations: Vec::new(),
			measured: None,
		}
	}

	pub fn with_shared_overhead(mut self, overhead: CacheSize) -> Self {
		self.shared_overhead = overhead;


		self
	}

	/// Metadata reservation for EVERY tracked key, fast or slow: a demotion
	/// moves the value and leaves the key's row, stack node and header in
	/// DRAM. See `PolicyStack::dram_reserved_bytes` for the rule, and for why
	/// a reservation at or over `fast_capacity` is left to saturate.
	fn reserved_overhead(&self) -> CacheSize {
		self.measured.unwrap_or(self.queues.len() as CacheSize * self.shared_overhead)
	}

	/// This stack's eff (S5): the whole fast tier's budget for values -- the
	/// tier's, not a segment's -- before the drain target.
	fn own_eff(&self) -> CacheSize {
		self.fast_capacity.saturating_sub(self.reserved_overhead())
	}

	/// Whether a value of `migrating` bytes is STRUCTURAL (S5): larger than an
	/// empty fast tier. Such a key is placed slow, keeps its place in the
	/// policy's order, and is never promoted while it stays that large. The
	/// stack's own check beside the client's flag, so a key the client placed
	/// normally just before eff moved is placed as the stack's own promotions
	/// would place it.
	fn structural(&self, migrating: CacheSize) -> bool {
		migrating > self.own_eff()
	}

	pub fn tier_of(&self, key: HashedKey) -> Option<Tier> {
		let payload = self.queues.payload(key)?;
		match Queue::from_u8(payload.queue) {
			Queue::Fifo => Some(Tier::Slow),
			Queue::Main => payload.tier,
		}
	}

	fn resize_key(&mut self, key: HashedKey, new_size: ObjectSize, new_resident: u8) {
		let Some(payload) = self.queues.payload_mut(key) else { return };

		let old_migrating = payload.migrating();
		payload.size = new_size;
		payload.dram_resident = new_resident;
		let delta = payload.migrating() as i64 - old_migrating as i64;
		let (queue, tier) = (Queue::from_u8(payload.queue), payload.tier);

		match (queue, tier) {
			(Queue::Fifo, _) => {
				self.fifo_used = (self.fifo_used as i64 + delta).max(0) as CacheSize;
			},

			(Queue::Main, Some(Tier::Fast)) => {
				self.fast_used = (self.fast_used as i64 + delta).max(0) as CacheSize;
			},

			(Queue::Main, Some(Tier::Slow)) => {
				self.slow_used = (self.slow_used as i64 + delta).max(0) as CacheSize;
			},

			(Queue::Main, None) => {},
		}
	}

	fn touch(&mut self, key: HashedKey, structural: bool) {
		match self.queues.payload(key).map(|p| Queue::from_u8(p.queue)) {
			Some(Queue::Fifo) => self.promote_from_fifo(key, structural),
			Some(Queue::Main) => self.touch_main_fast(key, structural),
			None => {},
		}
	}

	/// A hit in the FIFO promotes to the front of main, and to fast.
	///
	/// The slot does not move: this is an unlink from one queue and a relink
	/// into the other, where the stack this replaces removed the key from one
	/// hash-indexed list and inserted it into another.
	fn promote_from_fifo(&mut self, key: HashedKey, structural: bool) {
		let Some(payload) = self.queues.payload(key) else { return };
		let size_bytes = payload.migrating();

		// S5: a STRUCTURAL key moves to main's front all the same -- its place
		// in the order -- with tier slow, and is not promoted.
		if structural {
			self.queues.move_to_front_of(Q_FIFO, Q_MAIN, key);
			self.fifo_used = self.fifo_used.saturating_sub(size_bytes);

			if let Some(p) = self.queues.payload_mut(key) {
				p.queue = Queue::Main as u8;
				p.tier = Some(Tier::Slow);
			}

			self.slow_used += size_bytes;
			self.main_count += 1;

			self.settle_fast_tier();
			return;
		}

		self.queues.move_to_front_of(Q_FIFO, Q_MAIN, key);
		self.fifo_used = self.fifo_used.saturating_sub(size_bytes);

		if let Some(p) = self.queues.payload_mut(key) {
			p.queue = Queue::Main as u8;
			p.tier = Some(Tier::Fast);
		}

		self.fast_used += size_bytes;
		self.fast_count += 1;
		self.main_count += 1;

		if self.main_boundary.is_none() {
			self.main_boundary = Some(key);
		}

		self.settle_fast_tier();

		if self.queues.payload(key).and_then(|p| p.tier) == Some(Tier::Fast) {
			self.migrations.push((key, Tier::Fast));
		}
	}

	/// Faithful port of `TwoQHybridStack::touch_main_fast`.
	fn touch_main_fast(&mut self, key: HashedKey, structural: bool) {
		let previous_tier = self.queues.payload(key).and_then(|p| p.tier);

		let already_at_front = self.queues.front(Q_MAIN) == Some(key);
		let is_boundary = self.main_boundary == Some(key);

		// Read the neighbour BEFORE moving: once the key is at the front its
		// predecessor is gone, and the boundary must step back to whatever fast
		// key was in front of it (S5: past any structural ones).
		let new_boundary_if_moved = if is_boundary && !already_at_front {
			prev_fast(&self.queues, key)
		} else {
			None
		};

		self.queues.move_front(Q_MAIN, key);

		if is_boundary && !already_at_front {
			self.main_boundary = new_boundary_if_moved;
		}

		// S5: a STRUCTURAL key moves to the front all the same -- its place in
		// the order -- with tier slow: a slow one is not promoted, a fast one
		// leaves the fast set, pushed `(key, Slow)` (its placement changed).
		if structural {
			if previous_tier == Some(Tier::Fast) {
				let size = self.queues.payload(key).map(|p| p.migrating()).unwrap_or(0);
				self.fast_used = self.fast_used.saturating_sub(size);
				self.fast_count = self.fast_count.saturating_sub(1);
				self.slow_used += size;

				if let Some(p) = self.queues.payload_mut(key) {
					p.tier = Some(Tier::Slow);
				}

				if self.main_boundary == Some(key) {
					self.main_boundary = None;
				}

				self.migrations.push((key, Tier::Slow));
			}

			self.settle_fast_tier();
			return;
		}

		let mut promoted = false;

		if previous_tier != Some(Tier::Fast) {
			if previous_tier == Some(Tier::Slow) {
				let size = self.queues.payload(key).map(|p| p.migrating()).unwrap_or(0);
				self.slow_used = self.slow_used.saturating_sub(size);
				self.fast_used += size;
				self.fast_count += 1;
				promoted = true;
			}

			if let Some(p) = self.queues.payload_mut(key) {
				p.tier = Some(Tier::Fast);
			}
		}

		// Fast, at the front; with no fast key in front of it, the boundary.
		if self.main_boundary.is_none() {
			self.main_boundary = Some(key);
		}

		self.settle_fast_tier();

		if promoted && self.queues.payload(key).and_then(|p| p.tier) == Some(Tier::Fast) {
			self.migrations.push((key, Tier::Fast));
		}
	}

	/// Demotes from the tier boundary until `fast_used` is back within the
	/// effective budget. The victim is always `main_boundary`, so nothing is searched.
	fn settle_fast_tier(&mut self) {
		let effective = self.fast_capacity.saturating_sub(self.reserved_overhead());
		let target = drain_target::bytes(effective);

		while self.fast_used > target {
			let Some(demote_key) = self.main_boundary else { break };
			let size = self.queues.payload(demote_key).map(|p| p.migrating()).unwrap_or(0);
			let new_boundary = prev_fast(&self.queues, demote_key);

			if let Some(p) = self.queues.payload_mut(demote_key) {
				p.tier = Some(Tier::Slow);
			}

			self.fast_used = self.fast_used.saturating_sub(size);
			self.fast_count = self.fast_count.saturating_sub(1);
			self.slow_used += size;
			self.main_boundary = new_boundary;

			self.migrations.push((demote_key, Tier::Slow));
		}
	}

	/// A `Set`, with the client's placement (S5): an existing key is an access
	/// (`touch`); a new one enters the slow FIFO as it always did -- where a
	/// structural value is anyway. Returns the placement applied.
	fn insert_with(&mut self, key: HashedKey, size: ObjectSize, dram_resident: ObjectSize, placement: Placement) -> Placement {
		let dram_resident = narrow_resident(dram_resident);
		let migrating = (size as CacheSize).saturating_sub(dram_resident as CacheSize);
		let structural = placement == Placement::Structural || self.structural(migrating);

		if self.queues.contains(key) {
			self.resize_key(key, size, dram_resident);
			self.touch(key, structural);
			return placed(structural);
		}

		self.queues.push_front(Q_FIFO, key, NodePayload {
			size,
			dram_resident,
			tier: None,
			freq: 0,
			ts: 0,
			queue: Queue::Fifo as u8,
		});
		self.fifo_used += migrating;

		placed(structural)
	}

	fn evict_fifo_tail(&mut self) -> Option<HashedKey> {
		let (key, payload) = self.queues.pop_back(Q_FIFO)?;
		self.fifo_used = self.fifo_used.saturating_sub(payload.migrating());
		Some(key)
	}
}

impl PolicyStack for TwoQCompactHybridStack {
	fn is_policy(&self, policy: &PaperPolicy) -> bool {
		matches!(policy, PaperPolicy::TwoQCompactHybrid(k_in) if *k_in == self.k_in)
	}

	fn len(&self) -> usize {
		self.queues.len()
	}

	fn contains(&self, key: HashedKey) -> bool {
		self.queues.contains(key)
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

	/// S5: every settle, against the current budget (the policy worker's
	/// end-of-pass step).
	fn resettle(&mut self) {
		self.settle_fast_tier();
	}

	fn update(&mut self, key: HashedKey) {
		if let Some(payload) = self.queues.payload(key) {
			let structural = self.structural(payload.migrating());
			self.touch(key, structural);
		}
	}

	fn remove(&mut self, key: HashedKey) {
		let Some(payload) = self.queues.payload(key) else { return };
		let size = payload.migrating();

		match Queue::from_u8(payload.queue) {
			Queue::Fifo => {
				self.queues.remove(Q_FIFO, key);
				self.fifo_used = self.fifo_used.saturating_sub(size);
			},

			Queue::Main => {
				let new_boundary_if_needed =
					if payload.tier == Some(Tier::Fast) && self.main_boundary == Some(key) {
						prev_fast(&self.queues, key)
					} else {
						None
					};

				self.queues.remove(Q_MAIN, key);
				self.main_count = self.main_count.saturating_sub(1);

				match payload.tier {
					Some(Tier::Fast) => {
						self.fast_used = self.fast_used.saturating_sub(size);
						self.fast_count = self.fast_count.saturating_sub(1);

						if self.main_boundary == Some(key) {
							self.main_boundary = new_boundary_if_needed;
						}
					},

					Some(Tier::Slow) => {
						self.slow_used = self.slow_used.saturating_sub(size);
					},

					None => {},
				}
			},
		}
	}

	fn resize(&mut self, max_size: CacheSize) {
		self.fifo_capacity = (self.k_in * max_size as f64) as CacheSize;
	}

	fn clear(&mut self) {
		self.queues.clear();

		self.fifo_used = 0;
		self.fast_used = 0;
		self.slow_used = 0;
		self.fast_count = 0;
		self.main_count = 0;
		self.main_boundary = None;
		self.migrations.clear();
	}

	fn evict_one(&mut self) -> Option<HashedKey> {
		if let Some(key) = self.evict_fifo_tail() {
			return Some(key);
		}

		let (key, payload) = self.queues.pop_back(Q_MAIN)?;
		let size = payload.migrating();
		self.main_count = self.main_count.saturating_sub(1);

		match payload.tier {
			Some(Tier::Fast) => {
				self.fast_used = self.fast_used.saturating_sub(size);
				self.fast_count = self.fast_count.saturating_sub(1);

				// The boundary was main's tail: the nearest fast key from the
				// new tail (S5: past any structural ones).
				if self.main_boundary == Some(key) {
					self.main_boundary = fast_at_or_before(&self.queues, self.queues.back(Q_MAIN));
				}
			},

			Some(Tier::Slow) => {
				self.slow_used = self.slow_used.saturating_sub(size);
			},

			None => {},
		}

		Some(key)
	}

	fn resize_fast_tier(&mut self, size: CacheSize) {
		self.fast_capacity = size;
		self.settle_fast_tier();
	}

	/// `tier_of`: the admission FIFO is slow (new keys are built slow), main
	/// is placed by its tier, and every crossing is pushed.
	/// See `PolicyStack::placement_of`.
	fn placement_of(&self, key: HashedKey) -> Option<Tier> {
		self.tier_of(key)
	}

	fn drain_tier_migrations(&mut self) -> Vec<(HashedKey, Tier)> {
		std::mem::take(&mut self.migrations)
	}

	fn structure_bytes(&self) -> Option<crate::meta::NodeBytes> {
		Some(crate::meta::NodeBytes::stack(self.queues.allocated_bytes()))
	}

	fn dram_reserved_bytes(&self) -> CacheSize {
		self.reserved_overhead()
	}

	fn fast_bytes_used(&self) -> CacheSize {
		self.fast_used
	}

	fn slow_bytes_used(&self) -> CacheSize {
		self.fifo_used + self.slow_used
	}

	fn fast_object_count(&self) -> usize {
		self.fast_count
	}

	fn slow_object_count(&self) -> usize {
		self.queues.queue_len(Q_FIFO) + (self.main_count - self.fast_count)
	}

	fn needs_capacity_eviction(&self) -> bool {
		self.fifo_used > self.fifo_capacity
	}
}

/// The fast tier is charged the metadata of EVERY tracked key, the slow FIFO
/// and slow main keys as much as the fast ones. A reservation of
/// `fast_object_count() x shared_overhead` understates DRAM by every slow
/// key's metadata and fails this test.
#[cfg(test)]
mod reservation_tests {
	use super::*;

	const FAST_CAPACITY: CacheSize = 10_000;
	const OVERHEAD: CacheSize = 200;
	const SIZE: ObjectSize = 1_000;

	/// 20 admissions and 12 hits: twenty keys x 200 B leave 6_000 B for
	/// values, so five main keys stay fast, seven go slow and eight are still
	/// in the FIFO. Charging the fast ones alone would keep eight fast.
	#[test]
	fn fifo_and_slow_main_keys_are_charged_against_the_fast_tier() {
		let mut stack = TwoQCompactHybridStack::new(0.5, 1_000_000, FAST_CAPACITY)
			.with_shared_overhead(OVERHEAD);

		for key in 1..=20 {
			stack.insert(key, SIZE);
		}

		for key in 1..=12 {
			stack.update(key);
		}

		let tracked = stack.len() as CacheSize;
		let fifo = stack.queues.queue_len(Q_FIFO);
		let main_slow = stack.main_count - stack.fast_count;

		assert_eq!(tracked, 20);
		assert!(
			fifo > 0 && main_slow > 0 && stack.fast_object_count() > 0,
			"the fixture must populate every state: {fifo} in the FIFO, {} fast, \
			 {main_slow} slow in main",
			stack.fast_object_count(),
		);
		assert_eq!(
			stack.dram_reserved_bytes(),
			tracked * OVERHEAD,
			"all {tracked} tracked keys keep their metadata in DRAM, but the reservation \
			 covers {} of them ({} fast, {main_slow} slow in main, {fifo} in the FIFO)",
			stack.dram_reserved_bytes() / OVERHEAD,
			stack.fast_object_count(),
		);

		let effective = FAST_CAPACITY - tracked * OVERHEAD;

		assert!(
			stack.fast_bytes_used() <= drain_target::bytes(effective),
			"{} B of values are fast against {effective} B left once all {tracked} keys' \
			 metadata is reserved",
			stack.fast_bytes_used(),
		);
	}
}
