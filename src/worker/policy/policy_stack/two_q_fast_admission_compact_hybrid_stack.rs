/*
 * Copyright (c) Kia Shakiba
 *
 * This source code is licensed under the GNU AGPLv3 license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! Slab-backed 2Q fast-admission hybrid: `TwoQFastAdmissionHybridStack` with
//! one structure where that has three.
//!
//! Identical to [`TwoQCompactHybridStack`] except that the admission queue is
//! DRAM-resident rather than PMEM-resident. That single placement change
//! propagates:
//!
//! - `tier_of` reports `Fast` for a key in the FIFO, not `Slow`.
//! - The FIFO reservation is carved OUT of the fast tier, so the main queue
//!   settles against `fast_capacity - fifo_capacity - reserved_overhead`
//!   rather than against `fast_capacity - reserved_overhead`.
//! - A promotion out of the FIFO emits NO migration: the bytes are already in
//!   DRAM, so only the bookkeeping moves.
//! - `resize` must re-settle, because `fifo_capacity` scales with `max_size`
//!   and therefore changes the main queue's budget. Plain 2Q's `resize` does
//!   not need to.
//! - The byte and object counters swap sides: the FIFO counts toward fast.

use crate::{
	object::ObjectSize,
	worker::policy::policy_stack::{
		compact_queue_set::CompactQueueSet, narrow_resident, watermarks, CacheSize, HashedKey,
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
struct TwoQFaPayload {
	queue: Queue,
	tier: Option<Tier>,
	/// The part of `size` that stays in DRAM in either tier; see `migrating`.
	dram_resident: u8,
	size: ObjectSize,
}

/// Pinned, exactly as `TwoQEntry` is in the stack this replaces. The payload
/// rides in the index bucket, so growth here costs bytes on every tracked key.
const _: () = assert!(
	std::mem::size_of::<TwoQFaPayload>() == 8,
	"TwoQFaPayload grew past 8 bytes",
);

impl TwoQFaPayload {
	/// Bytes that actually move between tiers.
	fn migrating(&self) -> CacheSize {
		(self.size as CacheSize).saturating_sub(self.dram_resident as CacheSize)
	}
}

pub struct TwoQFastAdmissionCompactHybridStack {
	queues: CompactQueueSet<TwoQFaPayload>,

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

impl TwoQFastAdmissionCompactHybridStack {
	pub fn new(k_in: f64, max_size: CacheSize, fast_capacity: CacheSize) -> Self {
		TwoQFastAdmissionCompactHybridStack {
			queues: CompactQueueSet::default(),
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

	/// Also pre-sizes the slab: every object costs `overhead` bytes of
	/// fast-tier metadata whichever tier its value sits in, so
	/// `fast_capacity / overhead` bounds the entry count. Capped, because that
	/// ceiling is sound but unbounded -- see `MAX_PREALLOC_ENTRIES`.
	pub fn with_shared_overhead(mut self, overhead: CacheSize) -> Self {
		self.shared_overhead = overhead;

		if overhead > 0 {
			let ceiling = (self.fast_capacity / overhead) as usize;
			self.queues.reserve(ceiling.min(super::MAX_PREALLOC_ENTRIES));
		}

		self
	}

	pub fn fast_capacity(&self) -> CacheSize {
		self.fast_capacity
	}

	/// The main queue's share of the fast tier. The FIFO is DRAM-resident here,
	/// so its reservation is carved out of the same budget the main queue
	/// settles against -- the two compete, where in plain 2Q the FIFO is in
	/// PMEM and does not.
	fn effective_main_fast_capacity(&self) -> CacheSize {
		self.fast_capacity
			.saturating_sub(self.fifo_capacity)
			.saturating_sub(self.reserved_overhead())
	}

	fn reserved_overhead(&self) -> CacheSize {
		self.queues.len() as CacheSize * self.shared_overhead
	}

	pub fn tier_of(&self, key: HashedKey) -> Option<Tier> {
		let payload = self.queues.payload(key)?;
		match payload.queue {
			Queue::Fifo => Some(Tier::Fast),
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

		// No migration emitted: the FIFO is already DRAM, so promotion moves
		// bookkeeping rather than bytes.
		self.settle_fast_tier();
	}

	/// Faithful port of `TwoQFastAdmissionHybridStack::touch_main_fast`.
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
		let effective = self.effective_main_fast_capacity();

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
		Some(key)
	}
}

impl PolicyStack for TwoQFastAdmissionCompactHybridStack {
	fn is_policy(&self, policy: &PaperPolicy) -> bool {
		matches!(policy, PaperPolicy::TwoQFastAdmissionCompactHybrid(k_in) if *k_in == self.k_in)
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

		self.queues.push_front(
			Q_FIFO,
			key,
			TwoQFaPayload { queue: Queue::Fifo, tier: None, dram_resident, size },
		);
		self.fifo_used += (size as CacheSize).saturating_sub(dram_resident as CacheSize);
	}

	fn update(&mut self, key: HashedKey) {
		if self.queues.contains(key) {
			self.touch(key);
		}
	}

	fn remove(&mut self, key: HashedKey) {
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

		// The FIFO reservation is carved out of the fast tier, so moving it
		// changes the main queue's budget. Plain 2Q does not need this.
		self.settle_fast_tier();
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

				if self.main_boundary == Some(key) {
					self.main_boundary = self.queues.back(Q_MAIN);
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

	fn drain_tier_migrations(&mut self) -> Vec<(HashedKey, Tier)> {
		std::mem::take(&mut self.migrations)
	}

	fn dram_reserved_bytes(&self) -> CacheSize {
		self.reserved_overhead()
	}

	fn fast_bytes_used(&self) -> CacheSize {
		self.fifo_used + self.fast_used
	}

	fn slow_bytes_used(&self) -> CacheSize {
		self.slow_used
	}

	fn fast_object_count(&self) -> usize {
		self.queues.queue_len(Q_FIFO) + self.fast_count
	}

	fn slow_object_count(&self) -> usize {
		self.main_count - self.fast_count
	}

	fn needs_capacity_eviction(&self) -> bool {
		self.fifo_used > self.fifo_capacity
	}
}


/// Fidelity against `TwoQFastAdmissionHybridStack`.
#[cfg(all(test, feature = "two_q_fast_admission_hybrid_cache"))]
mod fidelity_tests {
	use super::*;
	use crate::worker::policy::policy_stack::two_q_fast_admission_hybrid_stack::TwoQFastAdmissionHybridStack;

	const MAX: CacheSize = 1_000_000;

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
	fn matches_the_baseline_migration_for_migration() {
		let ops = skewed_ops();
		for k_in in [0.1f64, 0.25, 0.5] {
			for fast in [8_192u64, 32_768, 131_072] {
				for overhead in [0u64, 112] {
					let mut a = TwoQFastAdmissionHybridStack::new(k_in, MAX, fast)
						.with_shared_overhead(overhead);
					let mut b = TwoQFastAdmissionCompactHybridStack::new(k_in, MAX, fast)
						.with_shared_overhead(overhead);
					let (mut ma, mut mb) = (Vec::new(), Vec::new());

					for (k, size) in &ops {
						if a.contains(*k) { a.update(*k); } else { a.insert(*k, *size); }
						if b.contains(*k) { b.update(*k); } else { b.insert(*k, *size); }
						ma.extend(a.drain_tier_migrations());
						mb.extend(b.drain_tier_migrations());
					}

					assert_eq!(ma, mb, "migrations diverge k_in {k_in} fast {fast} oh {overhead}");
					for (k, _) in &ops {
						assert_eq!(a.tier_of(*k), b.tier_of(*k), "tier of {k} diverges");
					}
					// the counters that swap sides in this variant
					assert_eq!(a.fast_bytes_used(), b.fast_bytes_used(), "fast bytes");
					assert_eq!(a.slow_bytes_used(), b.slow_bytes_used(), "slow bytes");
					assert_eq!(a.fast_object_count(), b.fast_object_count(), "fast objects");
					assert_eq!(a.slow_object_count(), b.slow_object_count(), "slow objects");
				}
			}
		}
	}

	/// The defining difference from plain 2Q: admission lands in DRAM, so a key
	/// in the FIFO reports Fast.
	///
	/// The capacities here are load-bearing. With `k_in = 0.25` against a
	/// 1,000,000 max_size, `fifo_capacity` is 250,000 -- larger than a 131,072
	/// fast tier -- so `effective_main_fast_capacity` saturates to ZERO and the
	/// baseline demotes the key straight back out on promotion. That is real
	/// behaviour of this variant, not a defect: its FIFO reservation is carved
	/// out of the fast tier and can swallow all of it. An earlier version of
	/// this test asserted no migration was emitted and failed against the
	/// BASELINE for exactly that reason.
	///
	/// So the assertion is equality, which is the fidelity claim, and the
	/// capacities are chosen to leave the main queue a real budget.
	#[test]
	fn admission_is_fast_and_promotion_matches_the_baseline() {
		const BIG_FAST: CacheSize = 900_000;
		let mut a = TwoQFastAdmissionHybridStack::new(0.1, MAX, BIG_FAST).with_shared_overhead(0);
		let mut b = TwoQFastAdmissionCompactHybridStack::new(0.1, MAX, BIG_FAST).with_shared_overhead(0);

		a.insert(1, 1024);
		b.insert(1, 1024);
		assert_eq!(a.tier_of(1), Some(Tier::Fast), "admission should land fast");
		assert_eq!(b.tier_of(1), a.tier_of(1));
		assert_eq!(a.drain_tier_migrations(), b.drain_tier_migrations());

		a.update(1);
		b.update(1);
		let (ma, mb) = (a.drain_tier_migrations(), b.drain_tier_migrations());
		assert_eq!(ma, mb, "promotion migrations diverge");
		assert!(ma.is_empty(), "with a real main budget, promotion moves no bytes");
		assert_eq!(a.tier_of(1), b.tier_of(1));
		assert_eq!(a.fast_object_count(), b.fast_object_count());
	}

	/// And the saturating case itself, since it is this variant's sharpest
	/// edge: when the FIFO reservation exceeds the fast tier the main budget
	/// goes to zero and everything promoted is demoted straight back.
	#[test]
	fn a_fifo_reservation_larger_than_the_fast_tier_matches_too() {
		let mut a = TwoQFastAdmissionHybridStack::new(0.25, MAX, 131_072).with_shared_overhead(0);
		let mut b = TwoQFastAdmissionCompactHybridStack::new(0.25, MAX, 131_072).with_shared_overhead(0);

		a.insert(1, 1024);
		b.insert(1, 1024);
		a.update(1);
		b.update(1);

		let (ma, mb) = (a.drain_tier_migrations(), b.drain_tier_migrations());
		assert_eq!(ma, mb, "migrations diverge under a saturated main budget");
		assert!(!ma.is_empty(), "expected the saturated budget to force a demotion");
		assert_eq!(a.tier_of(1), b.tier_of(1));
	}

	/// `resize` here changes the main queue's budget, because `fifo_capacity`
	/// is carved out of the fast tier. Plain 2Q's does not.
	#[test]
	fn resize_rescales_and_resettles_like_the_baseline() {
		let ops = skewed_ops();
		let mut a = TwoQFastAdmissionHybridStack::new(0.25, MAX, 65_536).with_shared_overhead(112);
		let mut b = TwoQFastAdmissionCompactHybridStack::new(0.25, MAX, 65_536).with_shared_overhead(112);
		let (mut ma, mut mb) = (Vec::new(), Vec::new());

		for (k, size) in ops.iter().take(8_000) {
			if a.contains(*k) { a.update(*k); } else { a.insert(*k, *size); }
			if b.contains(*k) { b.update(*k); } else { b.insert(*k, *size); }
			ma.extend(a.drain_tier_migrations());
			mb.extend(b.drain_tier_migrations());
		}

		a.resize(MAX / 4);
		b.resize(MAX / 4);
		ma.extend(a.drain_tier_migrations());
		mb.extend(b.drain_tier_migrations());

		for i in 0..2_000u64 {
			a.insert(10_000 + i, 1024);
			b.insert(10_000 + i, 1024);
			ma.extend(a.drain_tier_migrations());
			mb.extend(b.drain_tier_migrations());
		}

		assert_eq!(ma, mb, "migrations diverge across resize");
		assert_eq!(a.needs_capacity_eviction(), b.needs_capacity_eviction());
		assert_eq!(a.fast_object_count(), b.fast_object_count());
	}

	#[test]
	fn evicts_in_the_same_order() {
		let ops = skewed_ops();
		let mut a = TwoQFastAdmissionHybridStack::new(0.25, MAX, 32_768).with_shared_overhead(112);
		let mut b = TwoQFastAdmissionCompactHybridStack::new(0.25, MAX, 32_768).with_shared_overhead(112);
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
}
