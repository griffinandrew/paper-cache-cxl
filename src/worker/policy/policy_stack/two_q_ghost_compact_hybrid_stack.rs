/*
 * Copyright (c) Kia Shakiba
 *
 * This source code is licensed under the GNU AGPLv3 license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! Slab-backed 2Q ghost hybrid: `TwoQGhostHybridStack` with one structure
//! where that has three.
//!
//! Identical to [`TwoQCompactHybridStack`] plus a ghost filter: a key evicted
//! from the FIFO tail leaves a fingerprint behind, and a later admission that
//! hits it skips the FIFO entirely and enters the main queue in the fast tier.
//!
//! The ghost is NOT a queue and deliberately sits outside the slab. It stores
//! no keys and has no index -- it is a fixed power-of-two table of
//! `{fingerprint: u32, inserted_at: u32}` with an insertion-count window, the
//! design the S3-FIFO paper describes. So a key can legitimately be in the
//! ghost AND live in the index at once, and after a FIFO eviction it lives ONLY
//! in the ghost with no entry row at all.
//!
//! That last case is why `remove` clears the ghost BEFORE its early return: a
//! key with no entry row still has a fingerprint to erase, and returning early
//! would leave it to grant a spurious ghost hit later.

use crate::{
	object::ObjectSize,
	worker::policy::policy_stack::{
		compact_queue_set::CompactQueueSet, ghost_filter::GhostFilter, narrow_resident,
		watermarks, CacheSize, HashedKey,
		PolicyStack, Tier,
	},
	PaperPolicy,
};

/// Queue slots in the shared set. The FIFO admission queue is 0, the LRU main
/// queue is 1; a key is in exactly one of them.
const Q_FIFO: usize = 0;
const Q_MAIN: usize = 1;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Queue {
	Fifo,
	Main,
}

/// Combined per-key bookkeeping, carried in the index value.
///
/// `tier` is `None` while `queue == Fifo`: the FIFO is entirely slow-tier, so a
/// key there has no tier of its own to record.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct TwoQGhostPayload {
	queue: Queue,
	tier: Option<Tier>,
	/// The part of `size` that stays in DRAM in either tier; see `migrating`.
	dram_resident: u8,
	size: ObjectSize,
}

/// Pinned, exactly as `TwoQEntry` is in the stack this replaces. The payload
/// rides in the index bucket, so growth here costs bytes on every tracked key.
const _: () = assert!(
	std::mem::size_of::<TwoQGhostPayload>() == 8,
	"TwoQGhostPayload grew past 8 bytes",
);

impl TwoQGhostPayload {
	/// Bytes that actually move between tiers.
	fn migrating(&self) -> CacheSize {
		(self.size as CacheSize).saturating_sub(self.dram_resident as CacheSize)
	}
}

pub struct TwoQGhostCompactHybridStack {
	queues: CompactQueueSet<TwoQGhostPayload>,

	/// Fingerprints of keys evicted from the FIFO tail. Holds no keys and
	/// no slots, so it stays outside the slab.
	ghost: GhostFilter,

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
}

impl TwoQGhostCompactHybridStack {
	pub fn new(k_in: f64, max_size: CacheSize, fast_capacity: CacheSize) -> Self {
		// Sized from the cache's own capacity assuming a 512-byte nominal
		// object, capped at 8 Mi slots. Under-sizing only costs ghost hits.
		let ghost = GhostFilter::with_capacity(((max_size / 512) as usize).min(8 << 20));

		TwoQGhostCompactHybridStack {
			queues: CompactQueueSet::default(),
			ghost,
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
		}
	}

	pub fn with_shared_overhead(mut self, overhead: CacheSize) -> Self {
		self.shared_overhead = overhead;


		self
	}

	pub fn fast_capacity(&self) -> CacheSize {
		self.fast_capacity
	}

	fn reserved_overhead(&self) -> CacheSize {
		self.queues.len() as CacheSize * self.shared_overhead + self.ghost.dram_bytes()
	}

	fn effective_fast_capacity(&self) -> CacheSize {
		self.fast_capacity.saturating_sub(self.reserved_overhead())
	}

	pub fn is_ghost(&self, key: HashedKey) -> bool {
		self.ghost.contains(key)
	}

	/// A brand-new key whose fingerprint is in the ghost skips the FIFO and
	/// enters the main queue directly, in the fast tier.
	fn admit_via_ghost_hit(&mut self, key: HashedKey, size: ObjectSize, dram_resident: u8) {
		self.queues.push_front(
			Q_MAIN,
			key,
			TwoQGhostPayload { queue: Queue::Main, tier: Some(Tier::Fast), dram_resident, size },
		);
		self.fast_used += (size as CacheSize).saturating_sub(dram_resident as CacheSize);
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

	/// The ghost window tracks the main queue's population, so a fingerprint
	/// ages out after roughly as many insertions as the queue holds.
	fn trim_ghost(&mut self) {
		self.ghost.set_window(self.main_count);
	}

	pub fn tier_of(&self, key: HashedKey) -> Option<Tier> {
		let payload = self.queues.payload(key)?;
		match payload.queue {
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
		let (queue, tier) = (payload.queue, payload.tier);

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

	fn touch(&mut self, key: HashedKey) {
		match self.queues.payload(key).map(|p| p.queue) {
			Some(Queue::Fifo) => self.promote_from_fifo(key),
			Some(Queue::Main) => self.touch_main_fast(key),
			None => {},
		}
	}

	/// A hit in the FIFO promotes to the front of main, and to fast.
	///
	/// The slot does not move: this is an unlink from one queue and a relink
	/// into the other, where the stack this replaces removed the key from one
	/// hash-indexed list and inserted it into another.
	fn promote_from_fifo(&mut self, key: HashedKey) {
		let Some(payload) = self.queues.payload(key) else { return };
		let size_bytes = payload.migrating();

		self.queues.move_to_front_of(Q_FIFO, Q_MAIN, key);
		self.fifo_used = self.fifo_used.saturating_sub(size_bytes);

		if let Some(p) = self.queues.payload_mut(key) {
			p.queue = Queue::Main;
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
	fn touch_main_fast(&mut self, key: HashedKey) {
		let previous_tier = self.queues.payload(key).and_then(|p| p.tier);

		let already_at_front = self.queues.front(Q_MAIN) == Some(key);
		let is_boundary = self.main_boundary == Some(key);

		// Read the neighbour BEFORE moving: once the key is at the front its
		// predecessor is gone, and the boundary must step back to whatever was
		// in front of it.
		let new_boundary_if_moved = if is_boundary && !already_at_front {
			self.queues.before(key)
		} else {
			None
		};

		self.queues.move_front(Q_MAIN, key);

		if is_boundary && !already_at_front {
			self.main_boundary = new_boundary_if_moved;
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

			if self.main_boundary.is_none() {
				self.main_boundary = Some(key);
			}
		}

		self.settle_fast_tier();

		if promoted && self.queues.payload(key).and_then(|p| p.tier) == Some(Tier::Fast) {
			self.migrations.push((key, Tier::Fast));
		}
	}

	/// Demotes from the tier boundary until `fast_used` is back under the low
	/// watermark. The victim is always `main_boundary`, so nothing is searched.
	fn settle_fast_tier(&mut self) {
		let effective = self.effective_fast_capacity();

		if self.fast_used <= watermarks::high_bytes(effective) {
			return;
		}

		let low_water = watermarks::low_bytes(effective);

		while self.fast_used > low_water {
			let Some(demote_key) = self.main_boundary else { break };
			let size = self.queues.payload(demote_key).map(|p| p.migrating()).unwrap_or(0);
			let new_boundary = self.queues.before(demote_key);

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

	fn evict_fifo_tail(&mut self) -> Option<HashedKey> {
		let (key, payload) = self.queues.pop_back(Q_FIFO)?;
		self.fifo_used = self.fifo_used.saturating_sub(payload.migrating());
		self.ghost.insert(key);
		Some(key)
	}
}

impl PolicyStack for TwoQGhostCompactHybridStack {
	fn is_policy(&self, policy: &PaperPolicy) -> bool {
		matches!(policy, PaperPolicy::TwoQGhostCompactHybrid(k_in) if *k_in == self.k_in)
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
		let dram_resident = narrow_resident(dram_resident);

		if self.queues.contains(key) {
			self.resize_key(key, size, dram_resident);
			self.touch(key);
			return;
		}

		if self.ghost.contains(key) {
			self.admit_via_ghost_hit(key, size, dram_resident);
			return;
		}

		self.queues.push_front(
			Q_FIFO,
			key,
			TwoQGhostPayload { queue: Queue::Fifo, tier: None, dram_resident, size },
		);
		self.fifo_used += (size as CacheSize).saturating_sub(dram_resident as CacheSize);
	}

	fn update(&mut self, key: HashedKey) {
		if self.queues.contains(key) {
			self.touch(key);
		}
	}

	fn remove(&mut self, key: HashedKey) {
		// BEFORE the early return: after a FIFO eviction a key lives only in
		// the ghost, with no entry row to find.
		self.ghost.remove(key);

		let Some(payload) = self.queues.payload(key) else { return };
		let size = payload.migrating();

		match payload.queue {
			Queue::Fifo => {
				self.queues.remove(Q_FIFO, key);
				self.fifo_used = self.fifo_used.saturating_sub(size);
			},

			Queue::Main => {
				let new_boundary_if_needed =
					if payload.tier == Some(Tier::Fast) && self.main_boundary == Some(key) {
						self.queues.before(key)
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
		self.ghost.clear();

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

				if self.main_boundary == Some(key) {
					self.main_boundary = self.queues.back(Q_MAIN);
				}
			},

			Some(Tier::Slow) => {
				self.slow_used = self.slow_used.saturating_sub(size);
			},

			None => {},
		}

		self.trim_ghost();

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


/// Fidelity against `TwoQGhostHybridStack`.
#[cfg(all(test, feature = "two_q_ghost_hybrid_cache"))]
mod fidelity_tests {
	use super::*;
	use crate::worker::policy::policy_stack::two_q_ghost_hybrid_stack::TwoQGhostHybridStack;

	const MAX: CacheSize = 1_000_000;

	/// Churn deliberately wide enough to evict from the FIFO tail, which is the
	/// only thing that populates the ghost. A workload that never evicts would
	/// leave the ghost empty and exercise none of this variant's behaviour.
	fn churn_ops() -> Vec<(HashedKey, ObjectSize)> {
		let mut ops = Vec::new();
		let mut x: u64 = 0x243F_6A88_85A3_08D3;
		for _ in 0..20_000 {
			x ^= x << 13;
			x ^= x >> 7;
			x ^= x << 17;
			let u = (x >> 11) as f64 / (1u64 << 53) as f64;
			ops.push((((u * u * 2_000.0) as u64) + 1, 1024));
		}
		ops
	}

	#[test]
	fn matches_the_baseline_migration_for_migration() {
		let ops = churn_ops();
		for k_in in [0.1f64, 0.25] {
			for fast in [8_192u64, 65_536] {
				for overhead in [0u64, 112] {
					let mut a = TwoQGhostHybridStack::new(k_in, MAX, fast).with_shared_overhead(overhead);
					let mut b = TwoQGhostCompactHybridStack::new(k_in, MAX, fast).with_shared_overhead(overhead);
					let (mut ma, mut mb) = (Vec::new(), Vec::new());

					for (k, size) in &ops {
						if a.contains(*k) { a.update(*k); } else { a.insert(*k, *size); }
						if b.contains(*k) { b.update(*k); } else { b.insert(*k, *size); }
						// drain terminal evictions too -- they are what fills the ghost
						while a.needs_capacity_eviction() { if a.evict_one().is_none() { break } }
						while b.needs_capacity_eviction() { if b.evict_one().is_none() { break } }
						ma.extend(a.drain_tier_migrations());
						mb.extend(b.drain_tier_migrations());
					}

					assert_eq!(ma, mb, "migrations diverge k_in {k_in} fast {fast} oh {overhead}");
					assert_eq!(a.len(), b.len(), "lengths diverge");
					for (k, _) in ops.iter().take(500) {
						assert_eq!(a.tier_of(*k), b.tier_of(*k), "tier of {k} diverges");
						assert_eq!(a.is_ghost(*k), b.is_ghost(*k), "ghost membership of {k} diverges");
					}
				}
			}
		}
	}

	/// The defining behaviour: a key evicted from the FIFO tail leaves a
	/// fingerprint, and re-admitting it skips the FIFO and lands in main/fast.
	#[test]
	fn a_ghost_hit_admits_straight_to_main_and_fast() {
		let mut a = TwoQGhostHybridStack::new(0.0001, MAX, 131_072).with_shared_overhead(0);
		let mut b = TwoQGhostCompactHybridStack::new(0.0001, MAX, 131_072).with_shared_overhead(0);

		// fill the FIFO past its tiny budget so key 1 is evicted from the tail
		for k in 1..=32u64 {
			a.insert(k, 1024);
			b.insert(k, 1024);
			while a.needs_capacity_eviction() { if a.evict_one().is_none() { break } }
			while b.needs_capacity_eviction() { if b.evict_one().is_none() { break } }
		}
		assert!(a.is_ghost(1), "baseline should have ghosted the evicted key");
		assert_eq!(b.is_ghost(1), a.is_ghost(1), "ghost membership diverges");

		a.drain_tier_migrations();
		b.drain_tier_migrations();

		a.insert(1, 1024);
		b.insert(1, 1024);
		assert_eq!(a.tier_of(1), Some(Tier::Fast), "a ghost hit should admit to fast");
		assert_eq!(b.tier_of(1), a.tier_of(1), "ghost-hit tier diverges");
		assert_eq!(a.drain_tier_migrations(), b.drain_tier_migrations());
	}

	/// `remove` must clear the ghost even when the key has no entry row, which
	/// is the state a key is in after a FIFO eviction.
	#[test]
	fn remove_clears_a_ghost_with_no_entry_row() {
		let mut a = TwoQGhostHybridStack::new(0.0001, MAX, 131_072).with_shared_overhead(0);
		let mut b = TwoQGhostCompactHybridStack::new(0.0001, MAX, 131_072).with_shared_overhead(0);

		for k in 1..=32u64 {
			a.insert(k, 1024);
			b.insert(k, 1024);
			while a.needs_capacity_eviction() { if a.evict_one().is_none() { break } }
			while b.needs_capacity_eviction() { if b.evict_one().is_none() { break } }
		}
		assert!(a.is_ghost(1));
		assert!(!a.contains(1), "the key should have no entry row at this point");

		a.remove(1);
		b.remove(1);
		assert!(!a.is_ghost(1), "baseline should have cleared the fingerprint");
		assert_eq!(b.is_ghost(1), a.is_ghost(1), "ghost clearing diverges");
	}

	#[test]
	fn clear_empties_the_ghost_too() {
		let mut a = TwoQGhostHybridStack::new(0.0001, MAX, 131_072).with_shared_overhead(0);
		let mut b = TwoQGhostCompactHybridStack::new(0.0001, MAX, 131_072).with_shared_overhead(0);
		for k in 1..=32u64 {
			a.insert(k, 1024);
			b.insert(k, 1024);
			while a.needs_capacity_eviction() { if a.evict_one().is_none() { break } }
			while b.needs_capacity_eviction() { if b.evict_one().is_none() { break } }
		}
		a.clear();
		b.clear();
		assert!(!a.is_ghost(1));
		assert_eq!(b.is_ghost(1), a.is_ghost(1));
		assert_eq!(a.len(), b.len());
	}
}
