/*
 * Copyright (c) Kia Shakiba
 *
 * This source code is licensed under the GNU AGPLv3 license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! Slab-backed S3-FIFO hybrid: behaviourally identical to
//! `S3FifoHybridStack`, with one structure where that has three.
//!
//! `S3FifoHybridStack` keeps a one-access `HashList` and a main `HashList` --
//! each owning its OWN key-to-node index -- plus a separate `entries` map
//! holding the 8-byte payload. Every key is in exactly one of the two queues,
//! so a single [`ArenaQueueSet`] holds both orders over one slab.
//!
//! This is the family that paid the most for the payload moving OUT of the
//! index and into the slot. `mark_accessed` is the hottest per-get operation
//! here -- every hit on a main-queue key does nothing but flip a reference bit
//! -- and it touches no queue order at all. Under `CompactQueueSet` that was a
//! single probe with the payload already in the hash bucket; here it is a
//! probe into the index followed by a dereference into the slab. The 59.9 ns
//! against 97.4 ns this paragraph used to quote was measured against THAT
//! layout and no longer describes this code. The index it probes now is
//! 8 B/object rather than 56, so the first touch is into a far smaller table;
//! which way the two effects net out on this path has not been measured.
//!
//! The queue mechanics are unchanged. Admission lands at the front of the
//! one-access queue and is entirely slow-tier; a hit there promotes to the
//! front of main and to fast; a hit in main only sets the reference bit, and it
//! is eviction that acts on it -- an accessed key at the main tail is
//! reinserted at the front with the bit cleared instead of being evicted.
//! `main_boundary` names the least-recently-used fast key in main.
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
		arena_queue_set::{ArenaQueueSet, NodePayload}, narrow_resident, drain_target, CacheSize, HashedKey,
		PolicyStack, Tier, Placement, SetEvent, fast_at_or_before, placed, prev_fast,
	},
	PaperPolicy,
};

const Q_ONE_ACCESS: usize = 0;
const Q_MAIN: usize = 1;

/// Which of the two queues a key is in.
///
/// [`NodePayload::queue`] is a bare `u8`, so the enum is kept for readability
/// and converted at the boundary: `as u8` going into the payload,
/// [`Queue::from_u8`] coming back out. The discriminants are the queue indices
/// `Q_ONE_ACCESS` and `Q_MAIN` above, so the two never disagree.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
enum Queue {
	OneAccess = 0,
	Main = 1,
}

impl Queue {
	/// The read side of the `u8` boundary. Panics rather than defaulting: only
	/// this file writes the field, and it writes nothing but the two
	/// discriminants above.
	fn from_u8(raw: u8) -> Queue {
		match raw {
			0 => Queue::OneAccess,
			1 => Queue::Main,
			other => unreachable!("NodePayload::queue holds only 0 or 1 here, got {other}"),
		}
	}
}

/// Per-key bookkeeping is [`NodePayload`], the one node every policy shares.
///
/// This stack reads `size`, `dram_resident`, `queue` and `tier`, plus `freq`
/// as the S3-FIFO REFERENCE BIT: `freq != 0` is "accessed", `freq = 1` sets it
/// and `freq = 0` clears it. A reference bit is a one-bit frequency counter,
/// so nothing above 1 is ever stored here. `ts` belongs to the aging policies
/// and stays at its default.
///
/// `tier` is meaningful only while `queue == Queue::Main`: the one-access
/// queue is entirely slow-tier and its promotion is eager, so a key there
/// carries `tier: None` and needs no reference bit.
pub struct S3FifoCompactHybridStack {
	queues: ArenaQueueSet<NodePayload>,

	one_access_ratio: f64,
	one_access_capacity: CacheSize,
	one_access_used: CacheSize,

	main_capacity: CacheSize,

	fast_capacity: CacheSize,
	fast_used: CacheSize,
	slow_used: CacheSize,

	shared_overhead: CacheSize,

	fast_count: usize,
	main_count: usize,

	main_boundary: Option<HashedKey>,

	migrations: Vec<(HashedKey, Tier)>,

	/// S5: the measured M the policy worker pushed (`set_dram_metadata`),
	/// reserved instead of the per-object reservation; `None` under the
	/// per-object model.
	measured: Option<CacheSize>,
}

impl S3FifoCompactHybridStack {
	pub fn new(one_access_ratio: f64, max_size: CacheSize, fast_capacity: CacheSize) -> Self {
		S3FifoCompactHybridStack {
			queues: ArenaQueueSet::default(),
			one_access_ratio,
			one_access_capacity: (one_access_ratio * max_size as f64) as CacheSize,
			one_access_used: 0,
			main_capacity: ((1.0 - one_access_ratio) * max_size as f64) as CacheSize,
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

	/// Takes a FAST main key out of the fast set IN PLACE (S5): an overwrite
	/// with a value larger than an empty fast tier -- S3-FIFO's main queue is
	/// never reordered by an access. Pushed `(key, Slow)`: its placement
	/// changed.
	fn demote_in_place(&mut self, key: HashedKey) {
		let Some(payload) = self.queues.payload(key) else { return };
		let size = payload.migrating();

		if self.main_boundary == Some(key) {
			self.main_boundary = prev_fast(&self.queues, key);
		}

		if let Some(p) = self.queues.payload_mut(key) {
			p.tier = Some(Tier::Slow);
		}

		self.fast_used = self.fast_used.saturating_sub(size);
		self.fast_count = self.fast_count.saturating_sub(1);
		self.slow_used += size;

		self.migrations.push((key, Tier::Slow));
	}

	pub fn tier_of(&self, key: HashedKey) -> Option<Tier> {
		let payload = self.queues.payload(key)?;
		match Queue::from_u8(payload.queue) {
			Queue::OneAccess => Some(Tier::Slow),
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
			(Queue::OneAccess, _) => {
				self.one_access_used = (self.one_access_used as i64 + delta).max(0) as CacheSize;
			},

			(Queue::Main, Some(Tier::Fast)) => {
				self.fast_used = (self.fast_used as i64 + delta).max(0) as CacheSize;
			},

			(Queue::Main, Some(Tier::Slow)) => {
				self.slow_used = (self.slow_used as i64 + delta).max(0) as CacheSize;
			},

			// Unreachable: every path into the main queue records a tier.
			// This stack does produce `tier: None`, but only for one-access
			// residents, and those match the arm above.
			(Queue::Main, None) => {},
		}
	}

	fn touch(&mut self, key: HashedKey, structural: bool) {
		match self.queues.payload(key).map(|p| Queue::from_u8(p.queue)) {
			Some(Queue::OneAccess) => self.promote_from_one_access(key, structural),
			Some(Queue::Main) => self.mark_accessed(key),
			None => {},
		}
	}

	/// The hottest per-get operation in this family: no queue movement at all,
	/// just the reference bit. One index probe plus one slab dereference now
	/// that the payload lives in the slot rather than in the index value.
	fn mark_accessed(&mut self, key: HashedKey) {
		if let Some(p) = self.queues.payload_mut(key) {
			p.freq = 1;
		}
	}

	fn promote_from_one_access(&mut self, key: HashedKey, structural: bool) {
		let Some(payload) = self.queues.payload(key) else { return };
		let size_bytes = payload.migrating();

		// S5: a STRUCTURAL key moves to main's front all the same -- its place
		// in the order -- with tier slow, and is not promoted.
		if structural {
			self.queues.move_to_front_of(Q_ONE_ACCESS, Q_MAIN, key);
			self.one_access_used = self.one_access_used.saturating_sub(size_bytes);

			if let Some(p) = self.queues.payload_mut(key) {
				p.queue = Queue::Main as u8;
				p.tier = Some(Tier::Slow);
				p.freq = 0;
			}

			self.slow_used += size_bytes;
			self.main_count += 1;

			self.settle_fast_tier();
			return;
		}

		self.queues.move_to_front_of(Q_ONE_ACCESS, Q_MAIN, key);
		self.one_access_used = self.one_access_used.saturating_sub(size_bytes);

		if let Some(p) = self.queues.payload_mut(key) {
			p.queue = Queue::Main as u8;
			p.tier = Some(Tier::Fast);
			p.freq = 0;
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

	/// An accessed key at the main tail is reinserted at the front with its
	/// reference bit cleared, rather than evicted.
	fn give_second_chance(&mut self, key: HashedKey) {
		let Some(payload) = self.queues.payload(key) else { return };
		let size = payload.migrating();
		let was_fast = payload.tier == Some(Tier::Fast);
		let was_boundary = was_fast && self.main_boundary == Some(key);

		let new_boundary_if_moved = if was_boundary {
			prev_fast(&self.queues, key)
		} else {
			None
		};

		self.queues.move_front(Q_MAIN, key);

		if was_boundary {
			self.main_boundary = new_boundary_if_moved;
		}

		// S5: a STRUCTURAL key gets its second chance at the front with tier
		// slow -- never promoted; a fast one leaves the fast set, pushed.
		if self.structural(size) {
			if let Some(p) = self.queues.payload_mut(key) {
				p.tier = Some(Tier::Slow);
				p.freq = 0;
			}

			if was_fast {
				self.fast_used = self.fast_used.saturating_sub(size);
				self.fast_count = self.fast_count.saturating_sub(1);
				self.slow_used += size;

				if self.main_boundary == Some(key) {
					self.main_boundary = None;
				}

				self.migrations.push((key, Tier::Slow));
			}

			self.settle_fast_tier();
			return;
		}

		if let Some(p) = self.queues.payload_mut(key) {
			p.tier = Some(Tier::Fast);
			p.freq = 0;
		}

		if !was_fast {
			self.slow_used = self.slow_used.saturating_sub(size);
			self.fast_used += size;
			self.fast_count += 1;
		}

		if self.main_boundary.is_none() {
			self.main_boundary = Some(key);
		}

		self.settle_fast_tier();

		if self.queues.payload(key).and_then(|p| p.tier) == Some(Tier::Fast) {
			self.migrations.push((key, Tier::Fast));
		}
	}

	fn settle_fast_tier(&mut self) {
		let effective_capacity = self.fast_capacity.saturating_sub(self.reserved_overhead());
		let target = drain_target::bytes(effective_capacity);

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

	fn main_is_full(&self) -> bool {
		self.fast_used + self.slow_used >= self.main_capacity
	}

	/// A `Set`, with the client's placement (S5): an existing key is an access
	/// (`touch`: a one-access key promotes, a main key is marked -- and a fast
	/// one overwritten with a STRUCTURAL value leaves the fast set in place);
	/// a new key enters the one-access queue, which is slow, as it always did.
	/// Returns the placement applied.
	fn insert_with(&mut self, key: HashedKey, size: ObjectSize, dram_resident: ObjectSize, placement: Placement) -> Placement {
		let dram_resident = narrow_resident(dram_resident);
		let migrating = (size as CacheSize).saturating_sub(dram_resident as CacheSize);
		let structural = placement == Placement::Structural || self.structural(migrating);

		if let Some(payload) = self.queues.payload(key) {
			self.resize_key(key, size, dram_resident);

			// S5: a FAST main key overwritten with a structural value leaves the
			// fast set in place (an access never reorders main).
			if structural && Queue::from_u8(payload.queue) == Queue::Main && payload.tier == Some(Tier::Fast) {
				self.demote_in_place(key);
			}

			self.touch(key, structural);
			return placed(structural);
		}

		self.queues.push_front(
			Q_ONE_ACCESS,
			key,
			NodePayload {
				size,
				freq: 0,
				ts: 0,
				queue: Queue::OneAccess as u8,
				tier: None,
				dram_resident,
			},
		);
		self.one_access_used += migrating;

		placed(structural)
	}

	fn evict_one_access_tail(&mut self) -> Option<HashedKey> {
		let (key, payload) = self.queues.pop_back(Q_ONE_ACCESS)?;
		self.one_access_used = self.one_access_used.saturating_sub(payload.migrating());
		Some(key)
	}
}

impl PolicyStack for S3FifoCompactHybridStack {
	fn is_policy(&self, policy: &PaperPolicy) -> bool {
		matches!(policy, PaperPolicy::S3FifoCompactHybrid(r) if *r == self.one_access_ratio)
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
			Queue::OneAccess => {
				self.queues.remove(Q_ONE_ACCESS, key);
				self.one_access_used = self.one_access_used.saturating_sub(size);
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

					// Unreachable: `tier: None` is this stack's one-access
					// marker, and this match only sees main-queue keys.
					None => {},
				}
			},
		}
	}

	fn resize(&mut self, max_size: CacheSize) {
		self.one_access_capacity = (self.one_access_ratio * max_size as f64) as CacheSize;
		self.main_capacity = ((1.0 - self.one_access_ratio) * max_size as f64) as CacheSize;
	}

	fn clear(&mut self) {
		self.queues.clear();

		self.one_access_used = 0;
		self.fast_used = 0;
		self.slow_used = 0;
		self.fast_count = 0;
		self.main_count = 0;
		self.main_boundary = None;
		self.migrations.clear();
	}

	fn evict_one(&mut self) -> Option<HashedKey> {
		if !self.main_is_full() {
			if let Some(key) = self.evict_one_access_tail() {
				return Some(key);
			}
		}

		loop {
			let key = self.queues.back(Q_MAIN)?;
			let accessed = self.queues.payload(key).map(|p| p.freq != 0).unwrap_or(false);

			if accessed {
				self.give_second_chance(key);
				continue;
			}

			let payload = self.queues.remove(Q_MAIN, key);
			let size = payload.map(|p| p.migrating()).unwrap_or(0);
			let tier = payload.and_then(|p| p.tier);
			self.main_count = self.main_count.saturating_sub(1);

			match tier {
				Some(Tier::Fast) => {
					self.fast_used = self.fast_used.saturating_sub(size);
					self.fast_count = self.fast_count.saturating_sub(1);

					if self.main_boundary == Some(key) {
						// The tail was the boundary: the nearest fast key from the
						// new tail (S5: past any structural ones).
						self.main_boundary = fast_at_or_before(&self.queues, self.queues.back(Q_MAIN));
					}
				},

				Some(Tier::Slow) => {
					self.slow_used = self.slow_used.saturating_sub(size);
				},

				// Unreachable: `tier: None` is this stack's one-access
				// marker, and this key came off the main queue.
				None => {},
			}

			return Some(key);
		}
	}

	fn resize_fast_tier(&mut self, size: CacheSize) {
		self.fast_capacity = size;
		self.settle_fast_tier();
	}

	/// `tier_of`: the one-access queue is slow (new keys are built slow),
	/// main is placed by its tier, and every crossing is pushed.
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
		self.one_access_used + self.slow_used
	}

	fn fast_object_count(&self) -> usize {
		self.fast_count
	}

	fn slow_object_count(&self) -> usize {
		self.queues.queue_len(Q_ONE_ACCESS) + (self.main_count - self.fast_count)
	}

	fn needs_capacity_eviction(&self) -> bool {
		self.one_access_used > self.one_access_capacity
	}
}
