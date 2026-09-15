/*
 * Copyright (c) Kia Shakiba
 *
 * This source code is licensed under the GNU AGPLv3 license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! Slab-backed CLOCK hybrid: [`ClockCompactStack`]'s policy, tier-segmented.
//!
//! [`FifoCompactHybridStack`] with one addition -- a reference bit -- and the
//! whole of CLOCK is what the eviction path does when it finds that bit set.
//!
//! # The semantics are the flat stack's, verbatim
//!
//! Taken from `ClockCompactStack` (and, identically, from the `HashList`
//! original `ClockStack` it re-lays-out), not from a textbook:
//!
//! ```text
//!   insert, new key       push_front, bit CLEAR
//!   insert, existing key  forwarded to `update` -- the bit is SET, and
//!                         nothing moves
//!   hit                   the bit is SET, and nothing moves
//!   evict                 pop_back; bit clear -> evict it; bit set -> clear
//!                         the bit and PUSH IT TO THE FRONT, then look again
//! ```
//!
//! That last line is the one choice that matters and the one that is easy to
//! get wrong. Recycling the second-chance entry to the FRONT is CLOCK: in the
//! circular-buffer rendering the hand passes the entry and will not reach it
//! again until a full revolution, which in list terms is the head. Leaving the
//! entry WHERE IT IS and moving a separate hand past it is SIEVE -- a different
//! policy, with a different eviction order, and not what this tree's flat clock
//! stacks implement. `evict_one` below relinks.
//!
//! # What tiering adds, and why the second chance must promote
//!
//! `fast_boundary` names the oldest FAST key, and everything from the newest
//! end up to and including it is fast. That makes the fast set a contiguous
//! prefix of the one list, which is what lets a demotion be a single step of
//! the cursor. A second chance moves the key to the head -- inside that prefix
//! -- so the key has to become fast in the same operation, or a slow key would
//! sit in front of fast ones and the cursor would stop meaning anything.
//!
//! So a second chance here is exactly [`LruCompactHybridStack`]'s
//! `touch_fast_key`: relink to the front, promote, settle. CLOCK differs from
//! LRU not in what that operation does but in WHEN it happens -- LRU runs it on
//! every hit, CLOCK runs it only for the keys a hit saved from the hand.
//!
//! # Why this exists
//!
//! As the differential reference for `MergedOrder::Clock`, which is where the
//! design is actually going: the merged object store is its own eviction stack,
//! so under CLOCK a hit is one relaxed store into a reference bit in the slot's
//! tail padding, under a READ lock. This stack is the split-structure statement
//! of the same policy, so the two can be replayed against each other.

use crate::{
	object::ObjectSize,
	worker::policy::policy_stack::{
		arena_queue_set::{ArenaQueueSet, NodePayload}, narrow_resident, drain_target, CacheSize,
		HashedKey, PolicyStack, Tier,
	},
	PaperPolicy,
};

/// The single insertion-ordered queue, in the shared queue set's slot 0.
const Q_CLOCK: usize = 0;

/// Per-key bookkeeping is [`NodePayload`], the one node every policy shares.
/// This stack reads `tier`, `size`, `dram_resident` and `freq`; `ts`, `queue`
/// and `phys` belong to other policies and stay at their defaults here.
///
/// The reference bit rides in `freq`, as `freq != 0` -- the idiom the S3-FIFO
/// family already uses, and the reason this design costs not one byte more per
/// key than [`FifoCompactHybridStack`] does.
pub struct ClockCompactHybridStack {
	list: ArenaQueueSet<NodePayload>,

	fast_capacity: CacheSize,
	fast_used: CacheSize,
	slow_used: CacheSize,

	shared_overhead: CacheSize,

	fast_count: usize,

	/// The oldest FAST key: everything from the newest end up to and including
	/// this key is fast, everything after it is slow.
	fast_boundary: Option<HashedKey>,

	migrations: Vec<(HashedKey, Tier)>,
}

impl ClockCompactHybridStack {
	pub fn new(fast_capacity: CacheSize) -> Self {
		ClockCompactHybridStack {
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

	/// CLOCK's reference bit, as `freq != 0`.
	fn referenced(&self, key: HashedKey) -> bool {
		self.list.payload(key).map(|p| p.freq != 0).unwrap_or(false)
	}

	fn set_referenced(&mut self, key: HashedKey, on: bool) {
		if let Some(slot) = self.list.payload_mut(key) {
			slot.freq = u32::from(on);
		}
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

	/// The second chance: recycle `key` to the front of the queue and make it
	/// fast, exactly as `LruCompactHybridStack::touch_fast_key` does.
	///
	/// Called ONLY from `evict_one`, and only for a key whose reference bit the
	/// hand has just cleared. That is the whole difference between this policy
	/// and LRU: the same relink, reached from the eviction path rather than
	/// from every hit.
	fn recycle_to_front(&mut self, key: HashedKey) {
		let previous_tier = self.list.payload(key).and_then(|p| p.tier);

		let already_at_front = self.list.front(Q_CLOCK) == Some(key);
		let is_boundary = self.fast_boundary == Some(key);

		// Read the neighbour BEFORE moving: once the key is at the front its
		// predecessor is gone, and the boundary has to step back to whatever
		// was in front of it.
		let new_boundary_if_moved = if is_boundary && !already_at_front {
			self.list.before(key)
		} else {
			None
		};

		self.list.move_front(Q_CLOCK, key);

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
	/// effective budget. The victim is always `fast_boundary` -- the oldest
	/// fast key -- so nothing is searched.
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

impl PolicyStack for ClockCompactHybridStack {
	fn is_policy(&self, policy: &PaperPolicy) -> bool {
		matches!(policy, PaperPolicy::ClockCompactHybrid)
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

		// An existing key is resized in place and NOT moved -- insertion order
		// is the queue order -- but it IS referenced: `ClockCompactStack`'s
		// `insert` forwards an existing key straight to `update`, so writing a
		// key earns it a second chance exactly as reading it does. Re-settling
		// only matters if the key is fast, since only then can the resize have
		// pushed the fast tier over its budget.
		if let Some(payload) = self.list.payload(key) {
			let tier = payload.tier;
			let resized = payload.size != size;

			self.set_referenced(key, true);

			if resized {
				self.resize_key(key, size, dram_resident);

				if tier == Some(Tier::Fast) {
					self.settle_fast_tier();
				}
			}

			return;
		}

		self.list.push_front(Q_CLOCK, key, NodePayload {
			size,
			dram_resident,
			tier: Some(Tier::Fast),
			phys: Some(Tier::Fast),
			// A brand-new key enters UNREFERENCED, so one pass of the hand can
			// evict it. `push_front(Q, key, false)` in the flat stack.
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

	/// A cache hit: set the reference bit and nothing else.
	///
	/// The queue is not touched. That is the half of CLOCK that makes it cheap,
	/// and the half `MergedOrder::Clock` turns into a relaxed atomic store
	/// under a read lock.
	fn update(&mut self, key: HashedKey) {
		self.set_referenced(key, true);
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

		self.list.remove(Q_CLOCK, key);

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

	/// The hand. Walks from the oldest end, giving a second chance to every
	/// referenced key it passes, and evicts the first unreferenced one.
	///
	/// `ClockCompactStack::evict_one`'s loop, with the tier accounting the flat
	/// stack has nothing to do. Terminates in at most `len()` second chances:
	/// each one clears a bit, and nothing in the loop sets one.
	fn evict_one(&mut self) -> Option<HashedKey> {
		loop {
			let key = self.list.back(Q_CLOCK)?;

			if self.referenced(key) {
				// Cleared and recycled to the FRONT -- CLOCK, not SIEVE. See
				// the module doc.
				self.set_referenced(key, false);
				self.recycle_to_front(key);

				continue;
			}

			let slot = self.list.remove(Q_CLOCK, key)?;
			let size = slot.migrating();

			match slot.tier {
				Some(Tier::Fast) => {
					self.fast_used = self.fast_used.saturating_sub(size);
					self.fast_count = self.fast_count.saturating_sub(1);

					if self.fast_boundary == Some(key) {
						self.fast_boundary = self.list.back(Q_CLOCK);
					}
				},

				Some(Tier::Slow) => {
					self.slow_used = self.slow_used.saturating_sub(size);
				},
				// The shared node makes `tier` optional because the 2Q and
				// S3-FIFO families legitimately have a queue with no tier of
				// its own. This stack always records one, so this arm is
				// unreachable -- and it is spelled out rather than papered over
				// with `unwrap_or`, which would silently pick a tier if that
				// ever stopped being true.
				None => {},
			}

			return Some(key);
		}
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

/// Fidelity against the FLAT `ClockCompactStack`, which is the reference this
/// policy is defined by.
///
/// Tiering cannot reorder the queue -- demotion only steps a cursor along it --
/// so a tier-segmented CLOCK must evict in exactly the order an untiered CLOCK
/// does, given the same accesses. If it does not, the second chance is wrong:
/// the two ways to be wrong are leaving the entry in place (that is SIEVE) and
/// forgetting that an overwrite sets the bit.
#[cfg(test)]
mod flat_clock_fidelity {
	use super::*;
	use super::super::clock_compact_stack::ClockCompactStack;

	/// Big enough that nothing is ever demoted, so this test isolates the queue
	/// order from the tier boundary.
	const HUGE: CacheSize = 1 << 40;

	fn drain(stack: &mut dyn PolicyStack) -> Vec<HashedKey> {
		let mut out = Vec::new();

		while let Some(k) = stack.evict_one() {
			out.push(k);
		}

		out
	}

	/// `(key, is_update)`, the same shape `clock_compact_stack`'s own fidelity
	/// test uses -- and `is_update == false` on a key already present is an
	/// OVERWRITE, which both stacks must treat as a hit.
	fn replay(ops: &[(HashedKey, bool)], capacity: CacheSize) -> (Vec<HashedKey>, Vec<HashedKey>) {
		let mut flat = ClockCompactStack::default();
		let mut hybrid = ClockCompactHybridStack::new(capacity);

		for &(key, is_update) in ops {
			let a: &mut dyn PolicyStack = &mut flat;
			let b: &mut dyn PolicyStack = &mut hybrid;

			if is_update {
				a.update(key);
				b.update(key);
			} else {
				a.insert(key, 512);
				b.insert(key, 512);
			}

			assert_eq!(a.len(), b.len(), "len diverged at key {key}");
		}

		(drain(&mut flat), drain(&mut hybrid))
	}

	fn skewed_ops() -> Vec<(HashedKey, bool)> {
		let mut ops = Vec::new();
		let mut x: u64 = 0x243F_6A88_85A3_08D3;

		for i in 0..40_000u64 {
			x ^= x << 13;
			x ^= x >> 7;
			x ^= x << 17;
			let u = (x >> 11) as f64 / (1u64 << 53) as f64;
			let key = ((u * u * 500.0) as u64) + 1;
			ops.push((key, i % 3 == 0));
		}

		ops
	}

	#[test]
	fn evicts_in_the_same_order_as_the_flat_clock_stack() {
		let (flat, hybrid) = replay(&skewed_ops(), HUGE);

		assert_eq!(flat, hybrid, "the tiered CLOCK evicts in a different order");
		assert!(!flat.is_empty());
	}

	/// The same claim with the fast tier actually biting, which is the case
	/// that would catch a demotion or a promotion perturbing the queue.
	#[test]
	fn a_tight_fast_tier_does_not_reorder_the_queue() {
		let ops = skewed_ops();

		let (flat, hybrid) = replay(&ops, 40_000);

		assert_eq!(
			flat, hybrid,
			"a tight fast-tier budget reordered the CLOCK queue -- demotion is \
			 supposed to move a cursor, not the list",
		);
	}

	/// The SIEVE tripwire, stated as the structural fact rather than as an
	/// eviction order.
	///
	/// CLOCK and SIEVE agree on the eviction order of many short sequences --
	/// a queue that only drains gives the same answer either way -- so an
	/// order assertion is a weak test of this particular choice. The choice
	/// itself is not ambiguous at all: after a second chance the key must be at
	/// the FRONT of the queue, because `ClockCompactStack::evict_one` does
	/// `push_front`. SIEVE would leave it where it is.
	#[test]
	fn a_second_chance_moves_the_key_to_the_front() {
		let mut stack = ClockCompactHybridStack::new(HUGE);

		for key in [1u64, 2, 3] {
			stack.insert(key, 512);
		}

		assert_eq!(stack.list.front(Q_CLOCK), Some(3), "the newest key is not at the front");

		stack.update(1);

		assert_eq!(stack.evict_one(), Some(2), "the hand did not pass the referenced key");

		assert_eq!(
			stack.list.front(Q_CLOCK),
			Some(1),
			"the second-chance key was not recycled to the FRONT -- leaving it \
			 in place is SIEVE, and the flat ClockCompactStack this must match \
			 does `push_front`",
		);

		assert_eq!(stack.evict_one(), Some(3));
		assert_eq!(stack.evict_one(), Some(1));
		assert_eq!(stack.evict_one(), None);
	}

	/// An overwrite is a hit: `ClockCompactStack::insert` forwards an existing
	/// key to `update`.
	#[test]
	fn an_overwrite_sets_the_reference_bit() {
		let mut stack = ClockCompactHybridStack::new(HUGE);

		for key in [1u64, 2, 3] {
			stack.insert(key, 512);
		}

		// Same key, a DIFFERENT size, so this is a real resize as well.
		stack.insert(1, 1_024);

		assert_eq!(stack.evict_one(), Some(2), "the overwrite did not spare key 1");
		assert_eq!(stack.evict_one(), Some(3));
		assert_eq!(stack.evict_one(), Some(1));
	}
}
