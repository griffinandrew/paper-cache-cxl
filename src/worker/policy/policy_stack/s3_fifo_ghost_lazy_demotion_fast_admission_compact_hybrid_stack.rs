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
//! 2. **The two fast segments share one budget.** `one_access_capacity` is
//!    sized from `max_size` -- the CACHE budget -- so it is taken as a
//!    carve-out of `fast_capacity` only up to what the fast tier can pay for
//!    (`raw_one_access_capacity()`); whatever is left over is the main
//!    queue's fast segment (`raw_main_fast_capacity()`). The shared-metadata
//!    reservation is then split *proportionally* between the two segments
//!    (`reserved_shares`, following `LruSizedHybridStack`) so that
//!    `effective_one_access_capacity() + effective_main_fast_capacity() +
//!    reserved_overhead() == fast_capacity`. That holds for EVERY
//!    `(one_access_ratio, max_size, fast_capacity)`, including the ones where
//!    `one_access_ratio * max_size` on its own would exceed the whole fast
//!    tier: both segments are DRAM here, so a queue capped from the cache
//!    budget would otherwise let this stack hold `max(fast_capacity,
//!    one_access_ratio * max_size)` bytes of DRAM. The main queue's demotion
//!    trigger reads `effective_main_fast_capacity()`; the one-access queue's
//!    eviction trigger reads `effective_one_access_capacity()`. `resize`
//!    re-runs `settle_fast_tier` because growing `one_access_capacity`
//!    shrinks what is left for the main queue's fast segment.
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
		arena_queue_set::{ArenaQueueSet, NodePayload}, ghost_filter::GhostFilter, narrow_resident, drain_target,
		CacheSize, HashedKey,
		PolicyStack, Tier,
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
/// and `phys` to the lazy-copy one; both stay at their defaults, `phys` set
/// equal to `tier` at construction and never read again.
///
/// `tier` is meaningful only while `queue == Queue::Main`. The one-access
/// queue is entirely fast-tier in this variant and `tier_of` reports that from
/// the queue alone, so a key there still carries `tier: None` -- the field
/// records the MAIN queue's split, and a one-access key is on neither side of
/// it. Its promotion is eager, so it needs no reference bit either.
pub struct S3FifoGhostLazyDemotionFastAdmissionCompactHybridStack {
	queues: ArenaQueueSet<NodePayload>,

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
			queues: ArenaQueueSet::default(),
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

	/// The one-access queue's carve-out as the FAST TIER can pay for it, before
	/// the shared-metadata reservation.
	///
	/// The `one_access_capacity` field is `one_access_ratio * max_size`: a slice
	/// of the CACHE budget, which says nothing about how much DRAM this stack
	/// was given. Every admission lands in the one-access queue and that queue
	/// is DRAM in this variant (point 1 in the module doc), so charging it
	/// against the cache budget lets it draw DRAM that `fast_capacity` never
	/// granted: `raw_main_fast_capacity()` saturates to 0, `settle_fast_tier`
	/// duly drains the main queue's fast segment to nothing, and then nothing
	/// at all bounds the segment that is actually over -- `settle_fast_tier`
	/// governs only the main queue, and `needs_capacity_eviction` compares
	/// `one_access_used` against a cap bigger than the whole tier. Clamping is
	/// what makes the two DRAM-resident segments sum to `fast_capacity` in that
	/// configuration instead of to `max(fast_capacity, ratio * max_size)`.
	///
	/// Computed here rather than clamped where the field is assigned because
	/// `resize_fast_tier` moves `fast_capacity` at runtime; a value fixed in
	/// `new`/`resize` would go stale the moment the budget changed. A pure
	/// no-op whenever the carve-out already fits
	/// (`one_access_ratio * max_size <= fast_capacity`), which is every
	/// configuration swept to date -- `min` returns the raw field unchanged,
	/// equality included.
	fn raw_one_access_capacity(&self) -> CacheSize {
		self.one_access_capacity.min(self.fast_capacity)
	}

	/// The main queue's fast-segment budget *before* the shared-metadata
	/// reservation -- `fast_capacity` minus the one-access queue's carve-out.
	/// Reads `raw_one_access_capacity()`, which is already clamped to the
	/// budget, so this stays a genuine remainder rather than a saturation to
	/// zero. Kept separate from `effective_main_fast_capacity` so
	/// `reserved_shares` has a reservation-free capacity to proportion against
	/// (using the effective one would be circular).
	fn raw_main_fast_capacity(&self) -> CacheSize {
		self.fast_capacity.saturating_sub(self.raw_one_access_capacity())
	}

	/// Splits `reserved_overhead()` proportionally between this stack's two
	/// independently-capacitied FAST segments -- the one-access queue and the
	/// main queue's fast portion -- returned as `(one_access_share,
	/// main_share)`. `u128` intermediate so the product cannot overflow;
	/// remainder handed to the main segment so the two shares always re-sum
	/// exactly. `(0, 0)` if both capacities are zero.
	fn reserved_shares(&self) -> (CacheSize, CacheSize) {
		let reserved = self.reserved_overhead();

		// Both terms are slices of `fast_capacity` -- the first clamped to it,
		// the second its remainder -- so `total_capacity` IS the fast tier and
		// the split apportions the real budget. Proportioning against the raw
		// `one_access_capacity` field would divide the reservation by a
		// cache-sized number that can exceed the DRAM this stack holds.
		let one_access_capacity = self.raw_one_access_capacity();
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
	/// raw cap -- `raw_one_access_capacity()`, the carve-out the fast tier can
	/// pay for, never the cache-sized `one_access_capacity` field. This is the
	/// number `needs_capacity_eviction` polices a DRAM-resident queue with, so
	/// reading the field here is what would let that queue outgrow
	/// `fast_capacity` entirely.
	fn effective_one_access_capacity(&self) -> CacheSize {
		self.raw_one_access_capacity().saturating_sub(self.reserved_shares().0)
	}

	/// The budget actually available to the main queue's fast segment: raw
	/// `fast_capacity`, minus the one-access queue's fixed carve-out, minus
	/// this segment's share of the shared-metadata reservation. The settle
	/// drains to this number, never to any part of it alone.
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
			NodePayload {
				size,
				freq: 0,
				ts: 0,
				queue: Queue::Main as u8,
				tier: Some(Tier::Fast),
				phys: Some(Tier::Fast),
				dram_resident,
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
		match Queue::from_u8(payload.queue) {
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

	fn touch(&mut self, key: HashedKey) {
		match self.queues.payload(key).map(|p| Queue::from_u8(p.queue)) {
			Some(Queue::OneAccess) => self.promote_from_one_access(key),
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

	/// Moves a re-accessed one-access-queue key into the main queue at
	/// `Tier::Fast`. Emits no migration for the promotion itself -- the key's
	/// bytes are already physically Fast in this variant.
	fn promote_from_one_access(&mut self, key: HashedKey) {
		let Some(payload) = self.queues.payload(key) else { return };
		let size_bytes = payload.migrating();

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

	/// Demotes key(s) anchoring `main_boundary` while `fast_used` exceeds
	/// `effective_main_fast_capacity()` -- reference-bit gated.
	///
	/// The ceiling is `fast_capacity` minus the one-access carve-out minus this
	/// segment's proportional share of the shared-structure reservation.
	/// `effective_capacity` is read once, before the loop: a demotion only
	/// retags a payload, so neither the tracked-key count nor the ghost length
	/// -- and hence neither the reservation nor the target -- can move
	/// underneath the pass.
	fn settle_fast_tier(&mut self) {
		let effective_capacity = self.effective_main_fast_capacity();
		let target = drain_target::bytes(effective_capacity);

		while self.fast_used > target {
			let Some(candidate) = self.main_boundary else { break };

			let accessed = self.queues.payload(candidate).map(|p| p.freq != 0).unwrap_or(false);

			if accessed {
				// Reprieve: fresh start at the front instead of demotion.
				let new_boundary = self.queues.before(candidate);

				self.queues.move_front(Q_MAIN, candidate);
				self.main_boundary = new_boundary;

				if let Some(p) = self.queues.payload_mut(candidate) {
					p.freq = 0;
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
			NodePayload {
				size,
				freq: 0,
				ts: 0,
				queue: Queue::OneAccess as u8,
				tier: None,
				phys: None,
				dram_resident,
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

		match Queue::from_u8(payload.queue) {
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
						self.main_boundary = self.queues.back(Q_MAIN);
					}
				},

				Some(Tier::Slow) => {
					self.slow_used = self.slow_used.saturating_sub(size);
				},

				// Unreachable: `tier: None` is this stack's one-access
				// marker, and this key came off the main queue.
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

#[cfg(test)]
mod fast_budget_tests {
	use super::*;

	/// Both of this stack's segments are DRAM (the one-access queue is
	/// `Tier::Fast` in this variant), so their effective caps plus the
	/// reservation they were jointly charged for ARE the fast tier -- the
	/// invariant the module doc states. An admission into a DRAM-resident queue
	/// has to draw down the DRAM budget, not the cache budget.
	///
	/// The configurations that matter are the ones where
	/// `one_access_ratio * max_size` alone exceeds `fast_capacity`:
	/// `one_access_capacity` is sized from the cache budget and nothing in
	/// `new`/`resize` relates it to the DRAM budget, so without the clamp in
	/// `raw_one_access_capacity()` the one-access queue is capped ABOVE the
	/// whole tier while `raw_main_fast_capacity()` saturates to 0 -- and
	/// `settle_fast_tier`, which governs only the main queue, cannot pull any
	/// of it back.
	#[test]
	fn the_two_fast_segments_never_exceed_the_fast_budget() {
		const MAX_SIZE: CacheSize = 12_000_000_000;
		const FAST_CAPACITY: CacheSize = 4 * 1024 * 1024 * 1024;

		for ratio in [0.0f64, 0.1, 0.25, 0.3, 0.5, 1.0] {
			let mut stack = S3FifoGhostLazyDemotionFastAdmissionCompactHybridStack::new(
				ratio,
				MAX_SIZE,
				FAST_CAPACITY,
			).with_shared_overhead(224);

			// A populated stack, so `reserved_overhead()` is a real number and
			// the proportional split is actually exercised rather than trivially
			// zero.
			for i in 0..1_000u64 {
				stack.insert(i, 4_096);
			}

			let total = stack.effective_one_access_capacity()
				+ stack.effective_main_fast_capacity()
				+ stack.reserved_overhead();

			assert!(
				total <= FAST_CAPACITY,
				"ratio {ratio}: the DRAM-resident caps sum to {total}, over the \
				 {FAST_CAPACITY} byte fast tier",
			);

			// And exactly, not merely under: `reserved_shares` hands the
			// remainder to main so the two shares re-sum to the reservation, and
			// the `saturating_sub`s in the effective accessors lose nothing
			// until a share outgrows its own segment.
			assert_eq!(
				total, FAST_CAPACITY,
				"ratio {ratio}: the split must account for the budget exactly",
			);
		}
	}

	/// The clamp must be invisible to every configuration that already fits --
	/// which is every published sweep -- so those results stay bit-identical.
	#[test]
	fn a_carve_out_that_fits_is_untouched() {
		const MAX_SIZE: CacheSize = 12_000_000_000;
		const FAST_CAPACITY: CacheSize = 4 * 1024 * 1024 * 1024;

		// 0.1 * MAX_SIZE = 1.2e9, comfortably inside the 4 GiB budget.
		let stack = S3FifoGhostLazyDemotionFastAdmissionCompactHybridStack::new(
			0.1,
			MAX_SIZE,
			FAST_CAPACITY,
		).with_shared_overhead(224);

		assert_eq!(
			stack.raw_one_access_capacity(),
			stack.one_access_capacity,
			"a carve-out under the budget must pass through unchanged",
		);
		assert_eq!(
			stack.effective_main_fast_capacity(),
			FAST_CAPACITY - stack.one_access_capacity - stack.reserved_shares().1,
			"and the main segment must still get the plain remainder",
		);
	}
}
