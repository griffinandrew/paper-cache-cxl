/*
 * Copyright (c) Kia Shakiba
 *
 * This source code is licensed under the GNU AGPLv3 license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! Slab-backed S3-FIFO ghost hybrid: `S3FifoGhostHybridStack` with one
//! structure where that has three.
//!
//! Identical to [`S3FifoCompactHybridStack`] plus a ghost filter: a key evicted
//! from the one-access tail leaves a fingerprint, and a later admission that
//! hits it skips the one-access queue entirely and enters main in the fast tier.
//!
//! The ghost stays OUTSIDE the slab, which is correct rather than an omission.
//! It holds no keys and has no index -- a fixed power-of-two table of
//! `{fingerprint: u32, inserted_at: u32}` with an insertion-count window, the
//! design the S3-FIFO paper describes -- so it cannot live in a structure keyed
//! by slot. It is charged as a separate term alongside the per-object one, and
//! therefore does not enter this stack's per-object figure at all.

use crate::{
	object::ObjectSize,
	worker::policy::policy_stack::{
		compact_queue_set::CompactQueueSet, ghost_filter::GhostFilter, narrow_resident,
		watermarks, CacheSize, HashedKey,
		PolicyStack, Tier,
	},
	PaperPolicy,
};

const Q_ONE_ACCESS: usize = 0;
const Q_MAIN: usize = 1;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Queue {
	OneAccess,
	Main,
}

/// Combined per-key bookkeeping, carried in the index value.
///
/// `tier` and `accessed` are only meaningful while `queue == Main`: the
/// one-access queue is entirely slow-tier and its promotion is eager, so a key
/// there needs no reference bit.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct S3FifoGhostPayload {
	queue: Queue,
	tier: Option<Tier>,
	/// The part of `size` that stays in DRAM in either tier; see `migrating`.
	dram_resident: u8,
	accessed: bool,
	size: ObjectSize,
}

/// Pinned, exactly as `S3FifoEntry` is in the stack this replaces.
const _: () = assert!(
	std::mem::size_of::<S3FifoGhostPayload>() == 8,
	"S3FifoGhostPayload grew past 8 bytes",
);

impl S3FifoGhostPayload {
	fn migrating(&self) -> CacheSize {
		(self.size as CacheSize).saturating_sub(self.dram_resident as CacheSize)
	}
}

pub struct S3FifoGhostCompactHybridStack {
	queues: CompactQueueSet<S3FifoGhostPayload>,

	/// Fingerprints of keys evicted from the one-access tail. Holds no keys
	/// and no slots, so it stays outside the slab.
	ghost: GhostFilter,

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
}

impl S3FifoGhostCompactHybridStack {
	pub fn new(one_access_ratio: f64, max_size: CacheSize, fast_capacity: CacheSize) -> Self {
		// Sized from the cache's own capacity assuming a 512-byte nominal
		// object, capped at 8 Mi slots. Under-sizing only costs ghost hits.
		let ghost = GhostFilter::with_capacity(((max_size / 512) as usize).min(8 << 20));

		S3FifoGhostCompactHybridStack {
			queues: CompactQueueSet::default(),
			ghost,
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

	pub fn is_ghost(&self, key: HashedKey) -> bool {
		self.ghost.contains(key)
	}

	/// A brand-new key whose fingerprint is in the ghost skips the one-access
	/// queue and enters main directly, in the fast tier.
	fn admit_via_ghost_hit(&mut self, key: HashedKey, size: ObjectSize, dram_resident: u8) {
		self.queues.push_front(
			Q_MAIN,
			key,
			S3FifoGhostPayload {
				queue: Queue::Main,
				tier: Some(Tier::Fast),
				dram_resident,
				accessed: false,
				size,
			},
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

	/// The ghost window tracks the main queue's population. It runs only on a
	/// genuine main-queue eviction, not on a second chance.
	fn trim_ghost(&mut self) {
		self.ghost.set_window(self.main_count);
	}

	pub fn tier_of(&self, key: HashedKey) -> Option<Tier> {
		let payload = self.queues.payload(key)?;
		match payload.queue {
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
		let (queue, tier) = (payload.queue, payload.tier);

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

			(Queue::Main, None) => {},
		}
	}

	fn touch(&mut self, key: HashedKey) {
		match self.queues.payload(key).map(|p| p.queue) {
			Some(Queue::OneAccess) => self.promote_from_one_access(key),
			Some(Queue::Main) => self.mark_accessed(key),
			None => {},
		}
	}

	/// The hottest per-get operation in this family, and the reason the payload
	/// lives in the index value: one probe, no slab access, no queue movement.
	fn mark_accessed(&mut self, key: HashedKey) {
		if let Some(p) = self.queues.payload_mut(key) {
			p.accessed = true;
		}
	}

	fn promote_from_one_access(&mut self, key: HashedKey) {
		let Some(payload) = self.queues.payload(key) else { return };
		let size_bytes = payload.migrating();

		self.queues.move_to_front_of(Q_ONE_ACCESS, Q_MAIN, key);
		self.one_access_used = self.one_access_used.saturating_sub(size_bytes);

		if let Some(p) = self.queues.payload_mut(key) {
			p.queue = Queue::Main;
			p.tier = Some(Tier::Fast);
			p.accessed = false;
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
			self.queues.before(key)
		} else {
			None
		};

		self.queues.move_front(Q_MAIN, key);

		if was_boundary {
			self.main_boundary = new_boundary_if_moved;
		}

		if let Some(p) = self.queues.payload_mut(key) {
			p.tier = Some(Tier::Fast);
			p.accessed = false;
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

		if self.fast_used <= watermarks::high_bytes(effective_capacity) {
			return;
		}

		let drain_target = watermarks::low_bytes(effective_capacity);

		while self.fast_used > drain_target {
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

	fn main_is_full(&self) -> bool {
		self.fast_used + self.slow_used >= self.main_capacity
	}

	fn evict_one_access_tail(&mut self) -> Option<HashedKey> {
		let (key, payload) = self.queues.pop_back(Q_ONE_ACCESS)?;
		self.one_access_used = self.one_access_used.saturating_sub(payload.migrating());
		self.ghost.insert(key);
		Some(key)
	}
}

impl PolicyStack for S3FifoGhostCompactHybridStack {
	fn is_policy(&self, policy: &PaperPolicy) -> bool {
		matches!(policy, PaperPolicy::S3FifoGhostCompactHybrid(r) if *r == self.one_access_ratio)
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
			Q_ONE_ACCESS,
			key,
			S3FifoGhostPayload {
				queue: Queue::OneAccess,
				tier: None,
				dram_resident,
				accessed: false,
				size,
			},
		);
		self.one_access_used += (size as CacheSize).saturating_sub(dram_resident as CacheSize);
	}

	fn update(&mut self, key: HashedKey) {
		if self.queues.contains(key) {
			self.touch(key);
		}
	}

	fn remove(&mut self, key: HashedKey) {
		// BEFORE the early return: after a one-access eviction a key lives only
		// in the ghost, with no entry row to find.
		self.ghost.remove(key);

		let Some(payload) = self.queues.payload(key) else { return };
		let size = payload.migrating();

		match payload.queue {
			Queue::OneAccess => {
				self.queues.remove(Q_ONE_ACCESS, key);
				self.one_access_used = self.one_access_used.saturating_sub(size);
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
		self.one_access_capacity = (self.one_access_ratio * max_size as f64) as CacheSize;
		self.main_capacity = ((1.0 - self.one_access_ratio) * max_size as f64) as CacheSize;
	}

	fn clear(&mut self) {
		self.queues.clear();
		self.ghost.clear();

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
			let accessed = self.queues.payload(key).map(|p| p.accessed).unwrap_or(false);

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
						self.main_boundary = self.queues.back(Q_MAIN);
					}
				},

				Some(Tier::Slow) => {
					self.slow_used = self.slow_used.saturating_sub(size);
				},

				None => {},
			}

			self.trim_ghost();

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


/// Fidelity against `S3FifoGhostHybridStack`.
#[cfg(all(test, feature = "s3_fifo_ghost_hybrid_cache"))]
mod fidelity_tests {
	use super::*;
	use crate::worker::policy::policy_stack::s3_fifo_ghost_hybrid_stack::S3FifoGhostHybridStack;

	const MAX: CacheSize = 1_000_000;

	/// Wide enough to evict from the one-access tail, which is the only thing
	/// that populates a ghost. A narrower workload would leave it empty and
	/// exercise none of this variant's behaviour.
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
		for ratio in [0.1f64, 0.25] {
			for fast in [8_192u64, 65_536] {
				for overhead in [0u64, 112] {
					let mut a = S3FifoGhostHybridStack::new(ratio, MAX, fast).with_shared_overhead(overhead);
					let mut b = S3FifoGhostCompactHybridStack::new(ratio, MAX, fast).with_shared_overhead(overhead);
					let (mut ma, mut mb) = (Vec::new(), Vec::new());

					for (k, size) in &ops {
						if a.contains(*k) { a.update(*k); } else { a.insert(*k, *size); }
						if b.contains(*k) { b.update(*k); } else { b.insert(*k, *size); }
						while a.needs_capacity_eviction() { if a.evict_one().is_none() { break } }
						while b.needs_capacity_eviction() { if b.evict_one().is_none() { break } }
						ma.extend(a.drain_tier_migrations());
						mb.extend(b.drain_tier_migrations());
					}

					assert_eq!(ma, mb, "migrations diverge ratio {ratio} fast {fast} oh {overhead}");
					assert_eq!(a.len(), b.len(), "lengths diverge");
					for (k, _) in ops.iter().take(500) {
						assert_eq!(a.tier_of(*k), b.tier_of(*k), "tier of {k} diverges");
						assert_eq!(a.is_ghost(*k), b.is_ghost(*k), "ghost membership of {k} diverges");
					}
				}
			}
		}
	}

	/// A key evicted from the one-access tail leaves a fingerprint, and
	/// re-admitting it skips that queue and lands in main/fast.
	#[test]
	fn a_ghost_hit_admits_straight_to_main_and_fast() {
		let mut a = S3FifoGhostHybridStack::new(0.0001, MAX, 131_072).with_shared_overhead(0);
		let mut b = S3FifoGhostCompactHybridStack::new(0.0001, MAX, 131_072).with_shared_overhead(0);

		for k in 1..=32u64 {
			a.insert(k, 1024);
			b.insert(k, 1024);
			while a.needs_capacity_eviction() { if a.evict_one().is_none() { break } }
			while b.needs_capacity_eviction() { if b.evict_one().is_none() { break } }
		}
		assert!(a.is_ghost(1), "baseline should have ghosted the evicted key");
		assert_eq!(b.is_ghost(1), a.is_ghost(1));

		a.drain_tier_migrations();
		b.drain_tier_migrations();

		a.insert(1, 1024);
		b.insert(1, 1024);
		assert_eq!(a.tier_of(1), Some(Tier::Fast), "a ghost hit should admit to fast");
		assert_eq!(b.tier_of(1), a.tier_of(1));
		assert_eq!(a.drain_tier_migrations(), b.drain_tier_migrations());
	}

	/// `remove` must clear the ghost even with no entry row -- the state a key
	/// is in after a one-access eviction.
	#[test]
	fn remove_clears_a_ghost_with_no_entry_row() {
		let mut a = S3FifoGhostHybridStack::new(0.0001, MAX, 131_072).with_shared_overhead(0);
		let mut b = S3FifoGhostCompactHybridStack::new(0.0001, MAX, 131_072).with_shared_overhead(0);
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
		assert!(!a.is_ghost(1));
		assert_eq!(b.is_ghost(1), a.is_ghost(1));
	}

	/// Second chances must not trim the ghost window -- only a genuine
	/// main-queue eviction does.
	#[test]
	fn eviction_order_matches_including_second_chances() {
		let ops = churn_ops();
		let mut a = S3FifoGhostHybridStack::new(0.25, MAX, 32_768).with_shared_overhead(112);
		let mut b = S3FifoGhostCompactHybridStack::new(0.25, MAX, 32_768).with_shared_overhead(112);
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
		assert_eq!(b.len(), 0);
	}
}
