/*
 * Copyright (c) Kia Shakiba
 *
 * This source code is licensed under the GNU AGPLv3 license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! Slab-backed S3-FIFO ghost + lazy-demotion + fast-admission hybrid:
//! `S3FifoGhostLazyDemotionFastAdmissionHybridStack` with one structure where
//! that has three.
//!
//! Identical to [`S3FifoGhostCompactHybridStack`] plus the three things that
//! separate `S3FifoGhostLazyDemotionFastAdmissionHybridStack` from
//! `S3FifoGhostHybridStack`, all preserved byte for byte here:
//!
//! 1. **The one-access queue is FAST.** `tier_of` reports `Tier::Fast` for a
//!    one-access resident, `fast_bytes_used()`/`fast_object_count()` count it,
//!    and `slow_bytes_used()`/`slow_object_count()` no longer do. Admission is
//!    therefore a cheap DRAM write rather than a synchronous PMEM allocation
//!    on the calling thread (`hybrid_policy::admission_tier` returns `Fast`
//!    for a brand-new key under this policy).
//!
//! 2. **The two fast segments share one budget.** `one_access_capacity` is a
//!    fixed carve-out of `fast_capacity` (`raw_main_fast_capacity()`), and the
//!    shared-metadata reservation is split *proportionally* between the two
//!    segments (`reserved_shares`, following `LruSizedHybridStack`) so that
//!    `effective_one_access_capacity() + effective_main_fast_capacity() +
//!    reserved_overhead() == fast_capacity`. The main queue's demotion trigger
//!    reads `effective_main_fast_capacity()`; the one-access queue's eviction
//!    trigger reads `effective_one_access_capacity()`. `resize` re-runs
//!    `settle_fast_tier` because growing `one_access_capacity` shrinks what is
//!    left for the main queue's fast segment.
//!
//! 3. **Demotion is lazy.** `settle_fast_tier` gives a `main_boundary`
//!    candidate whose reference bit is set a reprieve -- move to the front of
//!    main with the bit cleared, walk the boundary one step, and try the next
//!    candidate -- instead of demoting it.
//!
//! And one consequence of (1) that this stack must also carry: a successful
//! promotion out of the one-access queue, and a ghost-hit admission, emit NO
//! `Tier::Fast` migration. Those keys' bytes are already physically DRAM (the
//! API layer built them Fast), so a migration would copy correct DRAM bytes
//! into a fresh DRAM buffer for nothing. `give_second_chance` keeps its push:
//! a key reaching it really can be in PMEM, so that move is real.
//!
//! The ghost stays OUTSIDE the slab for the same reason it does in
//! [`S3FifoGhostCompactHybridStack`]: it holds no keys and has no index, so it
//! cannot live in a structure keyed by slot. It is charged as a separate term
//! alongside the per-object one and does not enter this stack's per-object
//! figure.

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
/// one-access queue is entirely fast-tier in this variant and its promotion is
/// eager, so a key there needs neither.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct S3FifoGhostLazyDemotionFastAdmissionPayload {
	queue: Queue,
	tier: Option<Tier>,
	/// The part of `size` that stays in DRAM in either tier; see `migrating`.
	dram_resident: u8,
	accessed: bool,
	size: ObjectSize,
}

/// Pinned, exactly as `S3FifoEntry` is in the stack this replaces.
const _: () = assert!(
	std::mem::size_of::<S3FifoGhostLazyDemotionFastAdmissionPayload>() == 8,
	"S3FifoGhostLazyDemotionFastAdmissionPayload grew past 8 bytes",
);

impl S3FifoGhostLazyDemotionFastAdmissionPayload {
	fn migrating(&self) -> CacheSize {
		(self.size as CacheSize).saturating_sub(self.dram_resident as CacheSize)
	}
}

pub struct S3FifoGhostLazyDemotionFastAdmissionCompactHybridStack {
	queues: CompactQueueSet<S3FifoGhostLazyDemotionFastAdmissionPayload>,

	/// Fingerprints of keys evicted from the one-access tail. Holds no keys
	/// and no slots, so it stays outside the slab.
	ghost: GhostFilter,

	one_access_ratio: f64,
	one_access_capacity: CacheSize,
	one_access_used: CacheSize,

	/// The MAIN queue's total byte budget, spanning both tiers --
	/// `(1 - one_access_ratio) * max_size`. Read only by `is_main_full`, which
	/// gates `evict_one`'s one-access-tail priority. Unrelated to
	/// `raw_main_fast_capacity()`, which is carved out of `fast_capacity` and
	/// governs demotion instead.
	main_capacity: CacheSize,

	/// The configured total fast-tier (DRAM) budget, shared between the
	/// one-access queue and the main queue's fast segment.
	fast_capacity: CacheSize,
	fast_used: CacheSize,
	slow_used: CacheSize,

	shared_overhead: CacheSize,

	fast_count: usize,
	main_count: usize,

	main_boundary: Option<HashedKey>,

	migrations: Vec<(HashedKey, Tier)>,
}

impl S3FifoGhostLazyDemotionFastAdmissionCompactHybridStack {
	pub fn new(one_access_ratio: f64, max_size: CacheSize, fast_capacity: CacheSize) -> Self {
		// Sized from the cache's own capacity assuming a 512-byte nominal
		// object, capped at 8 Mi slots. Under-sizing only costs ghost hits.
		let ghost = GhostFilter::with_capacity(((max_size / 512) as usize).min(8 << 20));

		S3FifoGhostLazyDemotionFastAdmissionCompactHybridStack {
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

	fn reserved_overhead(&self) -> CacheSize {
		self.queues.len() as CacheSize * self.shared_overhead + self.ghost.dram_bytes()
	}

	/// The main queue's fast-segment budget *before* the shared-metadata
	/// reservation -- `fast_capacity` minus the one-access queue's fixed
	/// carve-out. Kept separate from `effective_main_fast_capacity` so
	/// `reserved_shares` has a reservation-free capacity to proportion against
	/// (using the effective one would be circular).
	fn raw_main_fast_capacity(&self) -> CacheSize {
		self.fast_capacity.saturating_sub(self.one_access_capacity)
	}

	/// Splits `reserved_overhead()` proportionally between this stack's two
	/// independently-capacitied FAST segments -- the one-access queue and the
	/// main queue's fast portion -- returned as `(one_access_share,
	/// main_share)`. `u128` intermediate so the product cannot overflow;
	/// remainder handed to the main segment so the two shares always re-sum
	/// exactly. `(0, 0)` if both capacities are zero.
	fn reserved_shares(&self) -> (CacheSize, CacheSize) {
		let reserved = self.reserved_overhead();

		let one_access_capacity = self.one_access_capacity;
		let main_capacity = self.raw_main_fast_capacity();
		let total_capacity = one_access_capacity + main_capacity;

		if total_capacity == 0 {
			return (0, 0);
		}

		let one_access_share =
			((reserved as u128 * one_access_capacity as u128) / total_capacity as u128) as CacheSize;
		let main_share = reserved.saturating_sub(one_access_share);

		(one_access_share, main_share)
	}

	/// The one-access queue's own byte cap after giving up its share of the
	/// shared-metadata reservation. With no reservation wired in this is the
	/// raw cap.
	fn effective_one_access_capacity(&self) -> CacheSize {
		self.one_access_capacity.saturating_sub(self.reserved_shares().0)
	}

	/// The budget actually available to the main queue's fast segment: raw
	/// `fast_capacity`, minus the one-access queue's fixed carve-out, minus
	/// this segment's share of the shared-metadata reservation. The watermarks
	/// sit on top of this number, never in place of any part of it.
	fn effective_main_fast_capacity(&self) -> CacheSize {
		self.raw_main_fast_capacity().saturating_sub(self.reserved_shares().1)
	}

	pub fn is_ghost(&self, key: HashedKey) -> bool {
		self.ghost.contains(key)
	}

	/// A brand-new key whose fingerprint is in the ghost skips the one-access
	/// queue and enters main directly, in the fast tier.
	///
	/// Emits no `Tier::Fast` migration: admission is unconditionally Fast under
	/// this policy, so the key's bytes are already DRAM. Only a
	/// `settle_fast_tier` demotion triggered by this admission can produce a
	/// migration here, and that is pushed inside `settle_fast_tier`.
	fn admit_via_ghost_hit(&mut self, key: HashedKey, size: ObjectSize, dram_resident: u8) {
		self.queues.push_front(
			Q_MAIN,
			key,
			S3FifoGhostLazyDemotionFastAdmissionPayload {
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
	}

	/// The ghost window tracks the main queue's population. It runs only on a
	/// genuine main-queue eviction, not on a second chance.
	fn trim_ghost(&mut self) {
		self.ghost.set_window(self.main_count);
	}

	pub fn tier_of(&self, key: HashedKey) -> Option<Tier> {
		let payload = self.queues.payload(key)?;
		match payload.queue {
			// The one-access queue is DRAM-resident in this variant -- the
			// single line that differs from `S3FifoGhostCompactHybridStack`.
			Queue::OneAccess => Some(Tier::Fast),
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

	/// Moves a re-accessed one-access-queue key into the main queue at
	/// `Tier::Fast`. Emits no migration for the promotion itself -- the key's
	/// bytes are already physically Fast in this variant.
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
	}

	/// An accessed key at the main tail is reinserted at the front with its
	/// reference bit cleared, rather than evicted.
	///
	/// This is the one promotion path that STILL pushes a migration: a key
	/// reaching it can genuinely be in PMEM (it was really demoted earlier), so
	/// moving it back to Fast is a physical move, not a relabeling.
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

	/// Demotes key(s) anchoring `main_boundary` once `fast_used` crosses the
	/// HIGH watermark of `effective_main_fast_capacity()`, then keeps going
	/// until it is back at or below the LOW watermark -- reference-bit gated.
	///
	/// The ceiling is `fast_capacity` minus the one-access carve-out minus this
	/// segment's proportional share of the shared-structure reservation.
	/// `effective_capacity` is read once, before the loop: a demotion only
	/// retags a payload, so neither the tracked-key count nor the ghost length
	/// -- and hence neither the reservation nor the target -- can move
	/// underneath the pass.
	fn settle_fast_tier(&mut self) {
		let effective_capacity = self.effective_main_fast_capacity();

		if self.fast_used <= watermarks::high_bytes(effective_capacity) {
			return;
		}

		let drain_target = watermarks::low_bytes(effective_capacity);

		while self.fast_used > drain_target {
			let Some(candidate) = self.main_boundary else { break };

			let accessed = self.queues.payload(candidate).map(|p| p.accessed).unwrap_or(false);

			if accessed {
				// Reprieve: fresh start at the front instead of demotion.
				let new_boundary = self.queues.before(candidate);

				self.queues.move_front(Q_MAIN, candidate);
				self.main_boundary = new_boundary;

				if let Some(p) = self.queues.payload_mut(candidate) {
					p.accessed = false;
				}

				continue;
			}

			let size = self.queues.payload(candidate).map(|p| p.migrating()).unwrap_or(0);
			let new_boundary = self.queues.before(candidate);

			if let Some(p) = self.queues.payload_mut(candidate) {
				p.tier = Some(Tier::Slow);
			}

			self.fast_used = self.fast_used.saturating_sub(size);
			self.fast_count = self.fast_count.saturating_sub(1);
			self.slow_used += size;
			self.main_boundary = new_boundary;

			self.migrations.push((candidate, Tier::Slow));
		}
	}

	/// Whether the main queue has reached its own byte budget -- the gate on
	/// `evict_one`'s one-access-tail priority.
	///
	/// `fast_used + slow_used` IS the main queue's byte total: one-access
	/// residents carry `tier: None` and move `one_access_used` alone.
	/// Deliberately not `fast_bytes_used()`, which folds `one_access_used` back
	/// in because this variant's one-access queue is DRAM too.
	fn is_main_full(&self) -> bool {
		self.fast_used + self.slow_used >= self.main_capacity
	}

	fn evict_one_access_tail(&mut self) -> Option<HashedKey> {
		let (key, payload) = self.queues.pop_back(Q_ONE_ACCESS)?;
		self.one_access_used = self.one_access_used.saturating_sub(payload.migrating());
		self.ghost.insert(key);
		Some(key)
	}
}

impl PolicyStack for S3FifoGhostLazyDemotionFastAdmissionCompactHybridStack {
	fn is_policy(&self, policy: &PaperPolicy) -> bool {
		matches!(policy, PaperPolicy::S3FifoGhostLazyDemotionFastAdmissionCompactHybrid(r) if *r == self.one_access_ratio)
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
			S3FifoGhostLazyDemotionFastAdmissionPayload {
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

		// Growing `one_access_capacity` shrinks the room left for the main
		// queue's fast segment -- catch it now rather than waiting for the next
		// unrelated insert/update, same reasoning `resize_fast_tier` has.
		self.settle_fast_tier();
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
		if !self.is_main_full() {
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
		// Total DRAM: main queue's fast segment + the one-access queue, both
		// physically Fast in this variant.
		self.fast_used + self.one_access_used
	}

	fn slow_bytes_used(&self) -> CacheSize {
		// The one-access queue no longer touches Slow/PMEM at all.
		self.slow_used
	}

	fn fast_object_count(&self) -> usize {
		self.fast_count + self.queues.queue_len(Q_ONE_ACCESS)
	}

	fn slow_object_count(&self) -> usize {
		self.main_count - self.fast_count
	}

	fn needs_capacity_eviction(&self) -> bool {
		// Against `effective_one_access_capacity()`, i.e. this segment's own cap
		// minus its proportional share of the shared-metadata reservation.
		self.one_access_used > self.effective_one_access_capacity()
	}
}


/// Fidelity against `S3FifoGhostLazyDemotionFastAdmissionHybridStack`.
#[cfg(all(test, feature = "s3_fifo_ghost_lazy_demotion_fast_admission_hybrid_cache"))]
mod fidelity_tests {
	use super::*;
	use crate::worker::policy::policy_stack::s3_fifo_ghost_lazy_demotion_fast_admission_hybrid_stack::S3FifoGhostLazyDemotionFastAdmissionHybridStack;

	type Baseline = S3FifoGhostLazyDemotionFastAdmissionHybridStack;
	type Compact = S3FifoGhostLazyDemotionFastAdmissionCompactHybridStack;

	const MAX: CacheSize = 1_000_000;

	/// Wide enough to evict from the one-access tail, which is the only thing
	/// that populates a ghost, and skewed enough to re-access main-queue keys,
	/// which is the only thing that exercises the reference bit the lazy
	/// demotion reads.
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

	/// Every observable gauge, so a divergence in accounting cannot hide behind
	/// a matching migration list.
	fn gauges(a: &Baseline, b: &Compact) {
		assert_eq!(a.len(), b.len(), "lengths diverge");
		assert_eq!(a.fast_bytes_used(), b.fast_bytes_used(), "fast bytes diverge");
		assert_eq!(a.slow_bytes_used(), b.slow_bytes_used(), "slow bytes diverge");
		assert_eq!(a.fast_object_count(), b.fast_object_count(), "fast count diverges");
		assert_eq!(a.slow_object_count(), b.slow_object_count(), "slow count diverges");
		assert_eq!(a.dram_reserved_bytes(), b.dram_reserved_bytes(), "reservation diverges");
		assert_eq!(
			a.needs_capacity_eviction(),
			b.needs_capacity_eviction(),
			"capacity-eviction trigger diverges",
		);
	}

	#[test]
	fn matches_the_baseline_migration_for_migration() {
		let ops = churn_ops();
		for ratio in [0.1f64, 0.25] {
			for fast in [8_192u64, 65_536] {
				for overhead in [0u64, 112] {
					let mut a = Baseline::new(ratio, MAX, fast).with_shared_overhead(overhead);
					let mut b = Compact::new(ratio, MAX, fast).with_shared_overhead(overhead);
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
					gauges(&a, &b);
					for (k, _) in ops.iter().take(500) {
						assert_eq!(a.tier_of(*k), b.tier_of(*k), "tier of {k} diverges");
						assert_eq!(a.is_ghost(*k), b.is_ghost(*k), "ghost membership of {k} diverges");
					}
				}
			}
		}
	}

	/// The one-access queue is FAST here, not slow: `tier_of` says so and both
	/// byte/object gauges count it on the fast side.
	#[test]
	fn the_one_access_queue_is_reported_as_fast() {
		let mut a = Baseline::new(0.25, MAX, 65_536).with_shared_overhead(0);
		let mut b = Compact::new(0.25, MAX, 65_536).with_shared_overhead(0);

		for k in 1..=8u64 {
			a.insert(k, 1024);
			b.insert(k, 1024);
		}

		assert_eq!(a.tier_of(3), Some(Tier::Fast), "baseline: one-access is fast");
		assert_eq!(b.tier_of(3), a.tier_of(3));
		assert_eq!(a.slow_bytes_used(), 0, "baseline: nothing slow yet");
		assert_eq!(a.slow_object_count(), 0);
		gauges(&a, &b);
	}

	/// A promotion out of the one-access queue emits NO migration -- the bytes
	/// are already DRAM. This is the delta from `S3FifoGhostCompactHybridStack`
	/// that a copy-paste would silently lose.
	#[test]
	fn promotion_out_of_one_access_emits_no_migration() {
		let mut a = Baseline::new(0.5, MAX, 1_000_000).with_shared_overhead(0);
		let mut b = Compact::new(0.5, MAX, 1_000_000).with_shared_overhead(0);

		a.insert(7, 1024);
		b.insert(7, 1024);
		a.drain_tier_migrations();
		b.drain_tier_migrations();

		a.update(7);
		b.update(7);

		assert_eq!(a.tier_of(7), Some(Tier::Fast));
		assert_eq!(b.tier_of(7), a.tier_of(7));
		assert!(a.drain_tier_migrations().is_empty(), "baseline emits no promotion migration");
		assert!(b.drain_tier_migrations().is_empty(), "compact must not either");
	}

	/// A ghost hit admits straight to main/fast, and likewise emits no
	/// migration in this variant.
	#[test]
	fn a_ghost_hit_admits_straight_to_main_and_fast_without_a_migration() {
		let mut a = Baseline::new(0.0001, MAX, 131_072).with_shared_overhead(0);
		let mut b = Compact::new(0.0001, MAX, 131_072).with_shared_overhead(0);

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
		let mut a = Baseline::new(0.0001, MAX, 131_072).with_shared_overhead(0);
		let mut b = Compact::new(0.0001, MAX, 131_072).with_shared_overhead(0);
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

	/// The demotion-time reference-bit reprieve: an accessed boundary key is
	/// moved to the front with its bit cleared instead of being demoted, so the
	/// pass demotes a *different* key than a non-lazy stack would.
	#[test]
	fn the_demotion_time_reprieve_matches_the_baseline() {
		// Small fast budget, large one-access ratio of 0 so everything the
		// promotions produce lands in the main queue's fast segment.
		let mut a = Baseline::new(0.0, MAX, 8_192).with_shared_overhead(0);
		let mut b = Compact::new(0.0, MAX, 8_192).with_shared_overhead(0);

		let (mut ma, mut mb) = (Vec::new(), Vec::new());

		for k in 1..=40u64 {
			a.insert(k, 512);
			b.insert(k, 512);
			// Second access promotes into main/fast.
			a.update(k);
			b.update(k);
			// Re-touch an older main resident so its reference bit is set when
			// the boundary reaches it.
			if k > 4 {
				a.update(k - 3);
				b.update(k - 3);
			}
			ma.extend(a.drain_tier_migrations());
			mb.extend(b.drain_tier_migrations());
		}

		assert!(!ma.is_empty(), "the workload must actually trigger demotions");
		assert_eq!(ma, mb, "reprieve/demotion order diverges");
		gauges(&a, &b);
		for k in 1..=40u64 {
			assert_eq!(a.tier_of(k), b.tier_of(k), "tier of {k} diverges");
		}
	}

	/// `resize` re-settles the fast tier immediately, because growing
	/// `one_access_capacity` shrinks the main queue's fast segment.
	#[test]
	fn resize_settles_the_fast_tier_immediately() {
		// one_access_capacity 10_000 out of a 32_768 fast budget leaves the
		// main queue's fast segment 22_768; the fill below settles against it.
		let mut a = Baseline::new(0.01, MAX, 32_768).with_shared_overhead(0);
		let mut b = Compact::new(0.01, MAX, 32_768).with_shared_overhead(0);

		for k in 1..=60u64 {
			a.insert(k, 512);
			b.insert(k, 512);
			// Second access promotes into main/fast, reference bit clear.
			a.update(k);
			b.update(k);
		}
		a.drain_tier_migrations();
		b.drain_tier_migrations();

		// Doubling max_size doubles one_access_capacity to 20_000, cutting the
		// main queue's fast segment to 12_768 -- which must drain on the spot
		// rather than waiting for the next unrelated insert/update.
		a.resize(MAX * 2);
		b.resize(MAX * 2);

		let ma = a.drain_tier_migrations();
		let mb = b.drain_tier_migrations();
		assert!(!ma.is_empty(), "resize must settle the fast tier on the baseline");
		assert_eq!(ma, mb, "resize-triggered demotions diverge");
		gauges(&a, &b);
	}

	/// The shared-metadata reservation is split proportionally between the two
	/// fast segments, which tightens BOTH the demotion trigger and the
	/// one-access eviction trigger.
	#[test]
	fn the_reservation_splits_between_both_fast_segments() {
		let ops = churn_ops();
		for ratio in [0.05f64, 0.4] {
			let mut a = Baseline::new(ratio, MAX, 49_152).with_shared_overhead(112);
			let mut b = Compact::new(ratio, MAX, 49_152).with_shared_overhead(112);
			let (mut ma, mut mb) = (Vec::new(), Vec::new());

			for (k, size) in ops.iter().take(6_000) {
				if a.contains(*k) { a.update(*k); } else { a.insert(*k, *size); }
				if b.contains(*k) { b.update(*k); } else { b.insert(*k, *size); }
				// Compare the trigger BEFORE draining, so a divergent
				// effective_one_access_capacity shows up directly.
				assert_eq!(
					a.needs_capacity_eviction(),
					b.needs_capacity_eviction(),
					"eviction trigger diverges at ratio {ratio}",
				);
				while a.needs_capacity_eviction() { if a.evict_one().is_none() { break } }
				while b.needs_capacity_eviction() { if b.evict_one().is_none() { break } }
				ma.extend(a.drain_tier_migrations());
				mb.extend(b.drain_tier_migrations());
			}

			assert!(!ma.is_empty(), "the workload must actually migrate");
			assert_eq!(ma, mb, "migrations diverge at ratio {ratio}");
			gauges(&a, &b);
		}
	}

	/// The one-access tail is only drained first while the main queue has room;
	/// once main is full, `evict_one` goes straight to the main tail.
	#[test]
	fn one_access_tail_is_evicted_first_only_while_main_has_room() {
		let ops = churn_ops();
		for ratio in [0.1f64, 0.25] {
			let mut a = Baseline::new(ratio, MAX, 32_768).with_shared_overhead(112);
			let mut b = Compact::new(ratio, MAX, 32_768).with_shared_overhead(112);
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
			assert_eq!(ea, eb, "eviction order diverges at ratio {ratio}");
			assert_eq!(b.len(), 0);
		}
	}

	/// A degenerate but legitimate configuration: `one_access_capacity` alone
	/// meets `fast_capacity`, so every promotion self-demotes immediately.
	#[test]
	fn zero_effective_main_capacity_demotes_every_promotion() {
		// ratio * MAX == 100_000 >= fast_capacity.
		let mut a = Baseline::new(0.1, MAX, 100_000).with_shared_overhead(0);
		let mut b = Compact::new(0.1, MAX, 100_000).with_shared_overhead(0);

		a.insert(1, 1024);
		b.insert(1, 1024);
		a.update(1);
		b.update(1);

		assert_eq!(a.tier_of(1), Some(Tier::Slow), "baseline self-demotes the promotion");
		assert_eq!(b.tier_of(1), a.tier_of(1));
		assert_eq!(a.drain_tier_migrations(), b.drain_tier_migrations());
		gauges(&a, &b);
	}

	/// `remove` and `clear` leave identical bookkeeping on both stacks.
	#[test]
	fn remove_and_clear_reset_bookkeeping() {
		let mut a = Baseline::new(0.25, MAX, 32_768).with_shared_overhead(112);
		let mut b = Compact::new(0.25, MAX, 32_768).with_shared_overhead(112);

		for k in 1..=64u64 {
			a.insert(k, 512);
			b.insert(k, 512);
			if k % 3 == 0 {
				a.update(k);
				b.update(k);
			}
		}
		for k in (1..=64u64).step_by(5) {
			a.remove(k);
			b.remove(k);
		}
		a.drain_tier_migrations();
		b.drain_tier_migrations();
		gauges(&a, &b);

		a.clear();
		b.clear();
		gauges(&a, &b);
		assert_eq!(b.len(), 0);
		assert_eq!(b.fast_bytes_used(), 0);
	}
}
