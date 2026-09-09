/*
 * Copyright (c) Kia Shakiba
 *
 * This source code is licensed under the GNU AGPLv3 license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! Slab-backed S3-FIFO ghost + lazy-demotion + fast-admission + MIDPOINT
//! hybrid: `S3FifoGhostLazyDemotionFastAdmissionMidpointHybridStack` with one
//! structure where that has three.
//!
//! Identical to `S3FifoGhostLazyDemotionFastAdmissionCompactHybridStack` --
//! the fast-tier one-access queue, the proportionally split shared-metadata
//! reservation, the demotion-time reference-bit reprieve, the ghost queue
//! outside the slab, and the "a promotion out of one-access or a ghost-hit
//! admission emits NO `Tier::Fast` migration" rule -- plus the single thing
//! that separates `S3FifoGhostLazyDemotionFastAdmissionMidpointHybridStack`
//! from `S3FifoGhostLazyDemotionFastAdmissionHybridStack`, preserved here
//! event for event:
//!
//! **A checkpoint roughly halfway through the SLOW portion of the main
//! queue.** The slow segment was a passive holding area: nothing looked at an
//! object there until it reached the eviction tail or was readmitted through
//! the ghost. This variant adds one more checkpoint, positioned approximately
//! halfway between the fast/slow boundary and the tail. If the object sitting
//! there has its reference bit set, it gets the exact same treatment as a
//! tail-reached second chance -- `give_second_chance`, i.e. moved to the front
//! of the fast segment with a real `Tier::Fast` migration -- instead of having
//! to survive all the way to the tail first. An object that is genuinely cold
//! at the midpoint is left alone and keeps aging normally. The check runs once
//! per `evict_one()` call, immediately before the main-queue loop, on both
//! routes into it (one-access queue empty, or main queue full and the
//! one-access tail therefore off limits).
//!
//! ## Locating "the middle" without an O(n) scan
//!
//! `slow_midpoint: Option<HashedKey>` is a cursor over one specific OBJECT,
//! maintained incrementally in O(1) amortized time using
//! [`ArenaQueueSet::before`] alone -- never by rescanning the segment, which
//! would be O(slow segment) per eviction and so O(n^2) over a cache's
//! lifetime.
//!
//! * **Growth at the front** (a demotion always retags the object already
//!   sitting where the new `main_boundary` lands, so nothing is physically
//!   inserted) and **shrinkage at the tail or from an arbitrary position** (a
//!   slow-tier eviction, an explicit `remove`, or a promotion out of the slow
//!   segment via `give_second_chance` -- including one `check_slow_midpoint`
//!   itself triggers) both push the tracked object ~0.5 positions past the
//!   true middle. `bump_midpoint_drift()` accumulates that and, every 2
//!   qualifying events, moves the cursor one step toward the front via
//!   `nudge_midpoint_toward_front` -- the only direction ever needed, since
//!   both kinds of event drift the same way.
//! * **The first demotion into an empty slow segment** seeds the cursor
//!   directly to the newly-demoted key.
//! * **The cursor's own target being removed or promoted** redirects it to the
//!   `before()` neighbor, but only if that neighbor is still Slow -- otherwise
//!   the cursor is cleared rather than left pointing into the fast segment.
//!   The redirect always runs BEFORE the key is unlinked or moved, since
//!   `before()` needs it still linked to resolve its neighbor.
//!
//! This is a heuristic trigger, not an exact median: "approximately halfway"
//! is all the mechanic needs, and the amortized correction keeps the cursor a
//! small bounded distance from the true middle without ever paying for a
//! rescan.
//!
//! The cursor and its drift counter are stack-level fields, like
//! `main_boundary` -- neither is per-object, so the per-object figure this
//! conversion exists to shrink is unchanged from the stack above.
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
pub struct S3FifoGhostLazyDemotionFastAdmissionMidpointCompactHybridStack {
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

	/// The mid-slow-segment checkpoint: one specific OBJECT roughly halfway
	/// between the fast/slow boundary and the main tail. Maintained
	/// incrementally with `before()` alone -- see the module doc.
	slow_midpoint: Option<HashedKey>,

	/// Accumulated half-position drift of `slow_midpoint` away from the true
	/// middle. One correcting step toward the front per two qualifying events.
	midpoint_drift: u8,

	migrations: Vec<(HashedKey, Tier)>,
}

impl S3FifoGhostLazyDemotionFastAdmissionMidpointCompactHybridStack {
	pub fn new(one_access_ratio: f64, max_size: CacheSize, fast_capacity: CacheSize) -> Self {
		// Sized from the cache's own capacity assuming a 512-byte nominal
		// object, capped at 8 Mi slots. Under-sizing only costs ghost hits.
		let ghost = GhostFilter::with_capacity(((max_size / 512) as usize).min(8 << 20));

		S3FifoGhostLazyDemotionFastAdmissionMidpointCompactHybridStack {
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
			slow_midpoint: None,
			midpoint_drift: 0,
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
	/// this segment's share of the shared-metadata reservation. The settle
	/// drains to this number, never to any part of it alone.
	fn effective_main_fast_capacity(&self) -> CacheSize {
		self.raw_main_fast_capacity().saturating_sub(self.reserved_shares().1)
	}

	pub fn is_ghost(&self, key: HashedKey) -> bool {
		self.ghost.contains(key)
	}

	/// Whether `key` is the object the mid-slow-segment cursor currently
	/// tracks. Mirrors the baseline's accessor of the same name.
	pub fn is_midpoint(&self, key: HashedKey) -> bool {
		self.slow_midpoint == Some(key)
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

	/// Moves the midpoint cursor one step toward the front, if possible.
	/// No-op if the cursor is empty, or if the neighbor toward the front is
	/// already Fast (the cursor has reached the boundary) -- it just stays put
	/// until growth or shrinkage makes room to move again.
	fn nudge_midpoint_toward_front(&mut self) {
		let Some(current) = self.slow_midpoint else { return };
		let Some(candidate) = self.queues.before(current) else { return };

		if self.queues.payload(candidate).and_then(|p| p.tier) == Some(Tier::Slow) {
			self.slow_midpoint = Some(candidate);
		}
	}

	/// Call after any event that changes the slow segment's size by exactly one
	/// in either direction (a demotion, a slow-tier eviction, or a
	/// promotion/removal out of the slow segment) once the cursor is already
	/// initialized. See the module doc for the "every 2 events, one step"
	/// derivation.
	fn bump_midpoint_drift(&mut self) {
		self.midpoint_drift += 1;

		if self.midpoint_drift >= 2 {
			self.midpoint_drift = 0;
			self.nudge_midpoint_toward_front();
		}
	}

	/// If `key` is currently the cursor's target, redirects it to the
	/// `before()` neighbor -- accepted only if that neighbor is still Slow,
	/// otherwise the cursor is cleared rather than left pointing into the fast
	/// segment. Must run while `key` is STILL linked in the main queue:
	/// `before()` needs that to resolve the neighbor.
	fn redirect_midpoint_before_removing(&mut self, key: HashedKey) {
		if self.slow_midpoint != Some(key) {
			return;
		}

		let new_target = self.queues.before(key).filter(|&candidate| {
			self.queues.payload(candidate).and_then(|p| p.tier) == Some(Tier::Slow)
		});

		self.slow_midpoint = new_target;
	}

	/// Checks the cursor's reference bit and, if set, gives it an early second
	/// chance -- the whole point of this variant. No-op if the slow segment is
	/// currently empty. Called once per `evict_one` pass over the main queue.
	fn check_slow_midpoint(&mut self) {
		let Some(candidate) = self.slow_midpoint else { return };
		let accessed = self.queues.payload(candidate).map(|p| p.freq != 0).unwrap_or(false);

		if accessed {
			self.give_second_chance(candidate);
		}
	}

	/// An accessed key at the main tail is reinserted at the front with its
	/// reference bit cleared, rather than evicted. Also reused verbatim by
	/// `check_slow_midpoint` for the mid-segment check: both are "promote this
	/// Slow key back to the front of Fast" with identical mechanics.
	///
	/// This is the one promotion path that STILL pushes a migration: a key
	/// reaching it can genuinely be in PMEM (it was really demoted earlier), so
	/// moving it back to Fast is a physical move, not a relabeling.
	fn give_second_chance(&mut self, key: HashedKey) {
		let Some(payload) = self.queues.payload(key) else { return };
		let size = payload.migrating();
		let was_fast = payload.tier == Some(Tier::Fast);
		let was_boundary = was_fast && self.main_boundary == Some(key);

		// The `move_front` below unlinks and relinks `key`. If it is the
		// cursor's target -- only possible while it is Slow -- redirect first,
		// while `before()` can still resolve its neighbor.
		if !was_fast {
			self.redirect_midpoint_before_removing(key);
		}

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
			self.bump_midpoint_drift();
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

			// A real demotion always grows the slow segment by exactly one, and
			// lands at its front -- see the module doc's drift derivation.
			if self.slow_midpoint.is_none() {
				self.slow_midpoint = Some(candidate);
			} else {
				self.bump_midpoint_drift();
			}
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

impl PolicyStack for S3FifoGhostLazyDemotionFastAdmissionMidpointCompactHybridStack {
	fn is_policy(&self, policy: &PaperPolicy) -> bool {
		matches!(policy, PaperPolicy::S3FifoGhostLazyDemotionFastAdmissionMidpointCompactHybrid(r) if *r == self.one_access_ratio)
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

				// Redirect the midpoint cursor BEFORE unlinking, if this key is
				// currently its target -- `before()` needs it still linked.
				let new_midpoint_if_needed =
					if payload.tier == Some(Tier::Slow) && self.slow_midpoint == Some(key) {
						self.queues.before(key).filter(|&candidate| {
							self.queues.payload(candidate).and_then(|p| p.tier) == Some(Tier::Slow)
						})
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

						if self.slow_midpoint == Some(key) {
							self.slow_midpoint = new_midpoint_if_needed;
						}

						self.bump_midpoint_drift();
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
		self.slow_midpoint = None;
		self.midpoint_drift = 0;
		self.migrations.clear();
	}

	fn evict_one(&mut self) -> Option<HashedKey> {
		if !self.is_main_full() {
			if let Some(key) = self.evict_one_access_tail() {
				return Some(key);
			}
		}

		// The mid-segment check -- see the module doc. Runs once per call,
		// exactly when this stack turns to the main queue for a real eviction,
		// and on both routes into the loop below.
		self.check_slow_midpoint();

		loop {
			let key = self.queues.back(Q_MAIN)?;
			let accessed = self.queues.payload(key).map(|p| p.freq != 0).unwrap_or(false);

			if accessed {
				self.give_second_chance(key);
				continue;
			}

			// Redirect the midpoint cursor BEFORE unlinking, if this key is
			// currently its target.
			if self.slow_midpoint == Some(key) {
				let new_target = self.queues.before(key).filter(|&candidate| {
					self.queues.payload(candidate).and_then(|p| p.tier) == Some(Tier::Slow)
				});

				self.slow_midpoint = new_target;
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
					self.bump_midpoint_drift();
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
