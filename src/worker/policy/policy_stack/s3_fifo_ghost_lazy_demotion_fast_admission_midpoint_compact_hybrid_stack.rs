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
//! [`CompactQueueSet::before`] alone -- never by rescanning the segment, which
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
struct S3FifoGhostLazyDemotionFastAdmissionMidpointPayload {
	queue: Queue,
	tier: Option<Tier>,
	/// The part of `size` that stays in DRAM in either tier; see `migrating`.
	dram_resident: u8,
	accessed: bool,
	size: ObjectSize,
}

/// Pinned, exactly as `S3FifoEntry` is in the stack this replaces.
const _: () = assert!(
	std::mem::size_of::<S3FifoGhostLazyDemotionFastAdmissionMidpointPayload>() == 8,
	"S3FifoGhostLazyDemotionFastAdmissionMidpointPayload grew past 8 bytes",
);

impl S3FifoGhostLazyDemotionFastAdmissionMidpointPayload {
	fn migrating(&self) -> CacheSize {
		(self.size as CacheSize).saturating_sub(self.dram_resident as CacheSize)
	}
}

pub struct S3FifoGhostLazyDemotionFastAdmissionMidpointCompactHybridStack {
	queues: CompactQueueSet<S3FifoGhostLazyDemotionFastAdmissionMidpointPayload>,

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
	/// this segment's share of the shared-metadata reservation. The watermarks
	/// sit on top of this number, never in place of any part of it.
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
			S3FifoGhostLazyDemotionFastAdmissionMidpointPayload {
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
		let accessed = self.queues.payload(candidate).map(|p| p.accessed).unwrap_or(false);

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
			p.accessed = false;
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
			S3FifoGhostLazyDemotionFastAdmissionMidpointPayload {
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
			let accessed = self.queues.payload(key).map(|p| p.accessed).unwrap_or(false);

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


/// Fidelity against `S3FifoGhostLazyDemotionFastAdmissionMidpointHybridStack`.
///
/// Follows the `fidelity_tests` module in
/// `s3_fifo_ghost_lazy_demotion_fast_admission_compact_hybrid_stack.rs`, plus
/// the fixtures that pin THIS variant's delta: the mid-slow-segment cursor and
/// the early second chance it triggers.
#[cfg(all(test, feature = "s3_fifo_ghost_lazy_demotion_fast_admission_midpoint_hybrid_cache"))]
mod fidelity_tests {
	use super::*;
	use crate::worker::policy::policy_stack::s3_fifo_ghost_lazy_demotion_fast_admission_midpoint_hybrid_stack::S3FifoGhostLazyDemotionFastAdmissionMidpointHybridStack;

	type Baseline = S3FifoGhostLazyDemotionFastAdmissionMidpointHybridStack;
	type Compact = S3FifoGhostLazyDemotionFastAdmissionMidpointCompactHybridStack;

	const MAX: CacheSize = 1_000_000;

	/// Wide enough to evict from the one-access tail, which is the only thing
	/// that populates a ghost, and skewed enough to re-access main-queue keys,
	/// which is the only thing that sets the reference bit both the lazy
	/// demotion and the midpoint checkpoint read.
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

	/// Which key each stack's mid-slow-segment cursor is tracking, scanned over
	/// the whole key space rather than read out of the private field -- the
	/// baseline's `slow_midpoint` is not visible from this module, only its
	/// `is_midpoint` accessor is.
	fn cursors(a: &Baseline, b: &Compact, keys: u64) -> (Option<HashedKey>, Option<HashedKey>) {
		let mut ca = None;
		let mut cb = None;

		for key in 1..=keys {
			if a.is_midpoint(key) {
				ca = Some(key);
			}
			if b.is_midpoint(key) {
				cb = Some(key);
			}
		}

		(ca, cb)
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
					let (ca, cb) = cursors(&a, &b, 2_000);
					assert_eq!(ca, cb, "midpoint cursor diverges ratio {ratio} fast {fast} oh {overhead}");
					for (k, _) in ops.iter().take(500) {
						assert_eq!(a.tier_of(*k), b.tier_of(*k), "tier of {k} diverges");
						assert_eq!(a.is_ghost(*k), b.is_ghost(*k), "ghost membership of {k} diverges");
					}
				}
			}
		}
	}

	/// The cursor is not merely equal at the end of a run: it tracks the same
	/// object throughout, which is what makes the early-promotion decisions
	/// line up rather than coincidentally agreeing.
	#[test]
	fn the_midpoint_cursor_follows_the_baseline_throughout_a_churn() {
		let ops = churn_ops();
		for ratio in [0.0f64, 0.1] {
			for fast in [8_192u64, 32_768] {
				let mut a = Baseline::new(ratio, MAX, fast).with_shared_overhead(112);
				let mut b = Compact::new(ratio, MAX, fast).with_shared_overhead(112);
				let (mut ma, mut mb) = (Vec::new(), Vec::new());

				for (i, (k, size)) in ops.iter().enumerate() {
					if a.contains(*k) { a.update(*k); } else { a.insert(*k, *size); }
					if b.contains(*k) { b.update(*k); } else { b.insert(*k, *size); }
					while a.needs_capacity_eviction() { if a.evict_one().is_none() { break } }
					while b.needs_capacity_eviction() { if b.evict_one().is_none() { break } }
					ma.extend(a.drain_tier_migrations());
					mb.extend(b.drain_tier_migrations());

					if i % 137 == 0 {
						let (ca, cb) = cursors(&a, &b, 2_000);
						assert_eq!(ca, cb, "midpoint cursor diverges at op {i} ratio {ratio} fast {fast}");
					}
				}

				assert!(!ma.is_empty(), "the workload must actually migrate");
				assert_eq!(ma, mb, "migrations diverge ratio {ratio} fast {fast}");
				let (ca, cb) = cursors(&a, &b, 2_000);
				assert_eq!(ca, cb, "midpoint cursor diverges at end ratio {ratio} fast {fast}");
				assert!(ca.is_some(), "the workload must actually seed the cursor");
				gauges(&a, &b);
			}
		}
	}

	/// Smallest `fast_capacity` that holds a fast segment of exactly `bytes`
	/// without a demotion pass either triggering on it or -- once one does
	/// trigger -- draining it. Taken from the baseline's own unit tests so the
	/// hand-traced midpoint fixtures below hold at any watermark config.
	fn capacity_holding(bytes: CacheSize, next: CacheSize) -> CacheSize {
		let mut capacity = (bytes as f64 / watermarks::low()).ceil() as CacheSize;

		while watermarks::low_bytes(capacity) < bytes {
			capacity += 1;
		}

		assert!(
			watermarks::high_bytes(capacity) < bytes + next,
			"watermark config leaves no room for this fixture",
		);

		capacity
	}

	/// The baseline's own `build_five_key_stack` fixture, driven into both
	/// stacks in lockstep. Fast segment sized to hold exactly 2 objects across
	/// a settled pass, `one_access_ratio` 0.0. Keys 1, 2, 3 get demoted oldest
	/// first as 4 and 5 arrive, leaving the slow segment [3, 2, 1]
	/// (boundary-to-tail) and the fast segment [5, 4]; after exactly 3
	/// demotions the drift correction settles the cursor on the middle
	/// element, key 2.
	fn build_five_key_pair() -> (Baseline, Compact) {
		let capacity = capacity_holding(20, 10);
		let mut a = Baseline::new(0.0, 1_000, capacity);
		let mut b = Compact::new(0.0, 1_000, capacity);

		for key in 1..=5u64 {
			a.insert(key, 10);
			b.insert(key, 10);
			a.update(key);
			b.update(key);
		}

		a.drain_tier_migrations();
		b.drain_tier_migrations();

		(a, b)
	}

	#[test]
	fn the_midpoint_cursor_seeds_and_settles_exactly_as_the_baseline() {
		let (a, b) = build_five_key_pair();

		for key in 1..=5u64 {
			assert_eq!(a.tier_of(key), b.tier_of(key), "tier of {key} diverges");
			assert_eq!(a.is_midpoint(key), b.is_midpoint(key), "midpoint of {key} diverges");
		}

		// Pinned outright, not merely mirrored.
		assert_eq!(b.tier_of(1), Some(Tier::Slow));
		assert_eq!(b.tier_of(2), Some(Tier::Slow));
		assert_eq!(b.tier_of(3), Some(Tier::Slow));
		assert_eq!(b.tier_of(4), Some(Tier::Fast));
		assert_eq!(b.tier_of(5), Some(Tier::Fast));
		assert!(b.is_midpoint(2), "expected key 2, the middle of slow segment [3, 2, 1]");
		assert!(!b.is_midpoint(1));
		assert!(!b.is_midpoint(3));
		gauges(&a, &b);
	}

	/// THE delta this variant exists for. Without the mid-segment checkpoint
	/// this fails outright: key 2 would still be Slow, key 4 would still be
	/// Fast, and the call would emit no `Tier::Fast` migration at all.
	#[test]
	fn a_reaccessed_midpoint_key_is_promoted_early_instead_of_waiting_for_the_tail() {
		let (mut a, mut b) = build_five_key_pair();
		assert!(b.is_midpoint(2));

		// Just sets the reference bit -- the same lazy-bit convention as
		// everywhere else in this design.
		a.update(2);
		b.update(2);
		assert_eq!(b.tier_of(2), Some(Tier::Slow), "a mere access must not itself migrate or reorder");
		assert_eq!(a.drain_tier_migrations(), b.drain_tier_migrations(), "an access must emit nothing");

		// `evict_one` must check the midpoint before it ever looks at the
		// tail. Key 2 is promoted (a real Slow -> Fast migration); that
		// promotion pushes `fast_used` back over capacity, cascading a real
		// demotion of the boundary (key 4, the only unaccessed fast key); the
		// call then proceeds to its own normal tail eviction of key 1.
		let ea = a.evict_one();
		let eb = b.evict_one();

		assert_eq!(ea, eb, "eviction diverges");
		assert_eq!(
			a.drain_tier_migrations(),
			b.drain_tier_migrations(),
			"the early promotion and its cascading demotion diverge",
		);

		assert_eq!(b.tier_of(2), Some(Tier::Fast), "the reaccessed midpoint key should have been promoted early");
		assert_eq!(b.tier_of(4), Some(Tier::Slow), "cascading demotion after the midpoint promotion");
		assert_eq!(eb, Some(1), "the tail is still evicted normally in the same call");
		assert!(!b.contains(1));
		assert!(!b.is_ghost(1), "main-queue tail evictions never populate the ghost queue");

		for key in 1..=5u64 {
			assert_eq!(a.tier_of(key), b.tier_of(key), "tier of {key} diverges");
			assert_eq!(a.is_midpoint(key), b.is_midpoint(key), "midpoint of {key} diverges");
		}
		gauges(&a, &b);
	}

	/// The other half of the mechanic: a cold object at the midpoint keeps
	/// aging normally and gets its one real chance at the tail.
	#[test]
	fn an_unaccessed_midpoint_key_is_left_alone() {
		let (mut a, mut b) = build_five_key_pair();
		assert!(b.is_midpoint(2));

		let ea = a.evict_one();
		let eb = b.evict_one();

		assert_eq!(ea, eb);
		assert_eq!(eb, Some(1), "straight to the normal tail eviction");
		assert_eq!(b.tier_of(2), Some(Tier::Slow), "an unaccessed midpoint key must not be promoted");
		assert_eq!(a.tier_of(2), b.tier_of(2));
		assert_eq!(a.drain_tier_migrations(), b.drain_tier_migrations());
		gauges(&a, &b);
	}

	/// An explicit `remove` of the cursor's own target redirects it to the
	/// `before()` neighbor, which here is still Slow.
	#[test]
	fn removing_the_midpoint_key_directly_redirects_the_cursor() {
		let (mut a, mut b) = build_five_key_pair();
		assert!(b.is_midpoint(2));

		a.remove(2);
		b.remove(2);

		assert!(!b.is_midpoint(2));
		assert!(b.is_midpoint(3), "cursor should redirect to the before()-neighbor still in the slow segment");
		for key in 1..=5u64 {
			assert_eq!(a.is_midpoint(key), b.is_midpoint(key), "midpoint of {key} diverges");
		}
		gauges(&a, &b);
	}

	/// The neighbor-still-Slow filter: with one slow object, evicting it must
	/// CLEAR the cursor rather than let it walk into the fast segment.
	#[test]
	fn evicting_the_only_slow_key_clears_the_midpoint_cursor() {
		let capacity = capacity_holding(10, 10);
		let mut a = Baseline::new(0.0, 1_000, capacity);
		let mut b = Compact::new(0.0, 1_000, capacity);

		for key in 1..=2u64 {
			a.insert(key, 10);
			b.insert(key, 10);
			a.update(key);
			b.update(key);
		}
		a.drain_tier_migrations();
		b.drain_tier_migrations();
		assert!(b.is_midpoint(1), "the only slow key is also the midpoint");
		assert_eq!(a.is_midpoint(1), b.is_midpoint(1));

		let ea = a.evict_one();
		let eb = b.evict_one();

		assert_eq!(ea, eb);
		assert_eq!(eb, Some(1));
		assert!(!b.is_midpoint(1), "the cursor must clear, not point into the fast segment");
		assert_eq!(a.is_midpoint(1), b.is_midpoint(1));
		assert_eq!(a.is_midpoint(2), b.is_midpoint(2));
		assert!(!b.is_midpoint(2));
		gauges(&a, &b);
	}

	/// The tail-reached second chance still works exactly as it did, and it
	/// too keeps the cursor in step.
	#[test]
	fn evict_one_gives_an_accessed_slow_key_a_second_chance() {
		let capacity = capacity_holding(10, 10);
		let mut a = Baseline::new(0.0, 1_000, capacity);
		let mut b = Compact::new(0.0, 1_000, capacity);

		for key in 1..=2u64 {
			a.insert(key, 10);
			b.insert(key, 10);
			a.update(key);
			b.update(key);
			a.drain_tier_migrations();
			b.drain_tier_migrations();
		}
		assert_eq!(b.tier_of(1), Some(Tier::Slow));

		a.update(1);
		b.update(1);
		assert_eq!(a.drain_tier_migrations(), b.drain_tier_migrations());

		let ea = a.evict_one();
		let eb = b.evict_one();

		assert_eq!(ea, eb);
		assert_eq!(eb, Some(2));
		assert_eq!(b.tier_of(1), Some(Tier::Fast));
		assert_eq!(a.tier_of(1), b.tier_of(1));
		assert_eq!(a.drain_tier_migrations(), b.drain_tier_migrations());
		assert_eq!(a.is_midpoint(1), b.is_midpoint(1));
		gauges(&a, &b);
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
	/// are already DRAM.
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
		let mut a = Baseline::new(0.0, MAX, 8_192).with_shared_overhead(0);
		let mut b = Compact::new(0.0, MAX, 8_192).with_shared_overhead(0);

		let (mut ma, mut mb) = (Vec::new(), Vec::new());

		for k in 1..=40u64 {
			a.insert(k, 512);
			b.insert(k, 512);
			a.update(k);
			b.update(k);
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
		let (ca, cb) = cursors(&a, &b, 40);
		assert_eq!(ca, cb, "midpoint cursor diverges");
		for k in 1..=40u64 {
			assert_eq!(a.tier_of(k), b.tier_of(k), "tier of {k} diverges");
		}
	}

	/// `resize` re-settles the fast tier immediately, because growing
	/// `one_access_capacity` shrinks the main queue's fast segment.
	#[test]
	fn resize_settles_the_fast_tier_immediately() {
		let mut a = Baseline::new(0.01, MAX, 32_768).with_shared_overhead(0);
		let mut b = Compact::new(0.01, MAX, 32_768).with_shared_overhead(0);

		for k in 1..=60u64 {
			a.insert(k, 512);
			b.insert(k, 512);
			a.update(k);
			b.update(k);
		}
		a.drain_tier_migrations();
		b.drain_tier_migrations();

		a.resize(MAX * 2);
		b.resize(MAX * 2);

		let ma = a.drain_tier_migrations();
		let mb = b.drain_tier_migrations();
		assert!(!ma.is_empty(), "resize must settle the fast tier on the baseline");
		assert_eq!(ma, mb, "resize-triggered demotions diverge");
		gauges(&a, &b);
		let (ca, cb) = cursors(&a, &b, 60);
		assert_eq!(ca, cb, "midpoint cursor diverges after a resize-driven pass");
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
	/// once main is full, `evict_one` goes straight to the main tail -- and the
	/// midpoint check runs on both routes.
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
		let mut a = Baseline::new(0.1, MAX, 100_000).with_shared_overhead(0);
		let mut b = Compact::new(0.1, MAX, 100_000).with_shared_overhead(0);

		a.insert(1, 1024);
		b.insert(1, 1024);
		a.update(1);
		b.update(1);

		assert_eq!(a.tier_of(1), Some(Tier::Slow), "baseline self-demotes the promotion");
		assert_eq!(b.tier_of(1), a.tier_of(1));
		assert_eq!(a.drain_tier_migrations(), b.drain_tier_migrations());
		assert!(b.is_midpoint(1), "the sole slow key seeds the cursor");
		assert_eq!(a.is_midpoint(1), b.is_midpoint(1));
		gauges(&a, &b);
	}

	/// `remove` and `clear` leave identical bookkeeping on both stacks --
	/// including the cursor and its drift counter.
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
		let (ca, cb) = cursors(&a, &b, 64);
		assert_eq!(ca, cb, "midpoint cursor diverges after removes");

		a.clear();
		b.clear();
		gauges(&a, &b);
		assert_eq!(b.len(), 0);
		assert_eq!(b.fast_bytes_used(), 0);
		let (ca, cb) = cursors(&a, &b, 64);
		assert_eq!((ca, cb), (None, None), "clear must reset the cursor on both");

		// And the drift counter, which only a fresh run can show: rebuild the
		// hand-traced fixture on the cleared stacks and check the cursor
		// settles on the same object it does from `new`.
		for key in 1..=5u64 {
			a.insert(key, 10);
			b.insert(key, 10);
			a.update(key);
			b.update(key);
		}
		a.drain_tier_migrations();
		b.drain_tier_migrations();
		let (ca, cb) = cursors(&a, &b, 5);
		assert_eq!(ca, cb, "midpoint cursor diverges after a clear-and-refill");
	}

	/// The neighbor-still-Slow filter on the DRIFT-CORRECTION path.
	///
	/// `nudge_midpoint_toward_front` and `evict_one`'s redirect apply the
	/// identical filter, but only the redirect is covered above (by
	/// `evicting_the_only_slow_key_clears_the_midpoint_cursor`). The nudge's
	/// copy only matters when a drift correction fires while the cursor is
	/// ALREADY the frontmost slow object, so `before(cursor)` is the fast/slow
	/// boundary object rather than another slow one. `churn_ops` never reaches
	/// that state: new demotions land at the slow front faster than the cursor
	/// -- one step per two events -- walks up to it, so `before(cursor)` is
	/// always Slow and the filter is a formality there. Dropping it leaves the
	/// cursor pointing into the fast segment, where the mid-segment check then
	/// promotes an object that never left DRAM.
	#[test]
	fn the_drift_correction_never_walks_the_cursor_into_the_fast_segment() {
		// The `build_five_key_pair` sizing -- a fast segment holding exactly
		// two 10-byte objects across a settled pass -- run out to nine keys, so
		// the slow segment is long enough to walk the cursor up to its own
		// front by hand.
		let capacity = capacity_holding(20, 10);
		let mut a = Baseline::new(0.0, 1_000, capacity);
		let mut b = Compact::new(0.0, 1_000, capacity);

		for key in 1..=9u64 {
			a.insert(key, 10);
			b.insert(key, 10);
			a.update(key);
			b.update(key);
		}
		a.drain_tier_migrations();
		b.drain_tier_migrations();

		// Slow segment [7, 6, 5, 4, 3, 2, 1] boundary-to-tail, fast [9, 8]; the
		// seeding demotion plus six drift events settle the cursor on key 4.
		assert_eq!(b.tier_of(7), Some(Tier::Slow), "key 7 is the frontmost slow object");
		assert_eq!(b.tier_of(8), Some(Tier::Fast), "key 8 is the fast/slow boundary object");
		assert!(b.is_midpoint(4), "expected key 4, the middle of slow segment [7 ..= 1]");
		assert_eq!(a.is_midpoint(4), b.is_midpoint(4), "midpoint of 4 diverges");

		// Two removals from the FRONT of the slow segment: each shortens the
		// segment ahead of the cursor and bumps the drift counter, and the
		// second one's correction steps the cursor onto key 5 -- now the
		// frontmost slow object, with the fast boundary directly before it.
		for key in [7u64, 6] {
			a.remove(key);
			b.remove(key);
		}

		assert!(b.is_midpoint(5), "the cursor should have reached the front of the slow segment");
		assert_eq!(a.is_midpoint(5), b.is_midpoint(5), "midpoint of 5 diverges");
		assert_eq!(
			b.queues.before(5),
			Some(8),
			"the object before the cursor must be the fast/slow boundary, or this fixture \
			 does not reach the guard",
		);

		// Two more drift events, this time from BEHIND the cursor so it stays
		// the frontmost slow object: the second fires the correction with
		// `before(cursor)` pointing squarely at the fast segment.
		for key in [1u64, 2] {
			a.remove(key);
			b.remove(key);
		}

		assert!(b.is_midpoint(5), "the correction must leave the cursor where it was");
		assert!(!b.is_midpoint(8), "the cursor must never be walked into the fast segment");
		assert_eq!(b.tier_of(8), Some(Tier::Fast));
		for key in 1..=9u64 {
			assert_eq!(a.tier_of(key), b.tier_of(key), "tier of {key} diverges");
			assert_eq!(a.is_midpoint(key), b.is_midpoint(key), "midpoint of {key} diverges");
		}
		gauges(&a, &b);

		// And the consequence, so this pins behaviour and not just the cursor's
		// identity: a cursor sitting on a FAST object turns the mid-segment
		// check into a spurious `Tier::Fast` migration of an object that is
		// already in DRAM.
		a.update(8);
		b.update(8);
		assert_eq!(a.drain_tier_migrations(), b.drain_tier_migrations(), "an access must emit nothing");

		let ea = a.evict_one();
		let eb = b.evict_one();

		assert_eq!(ea, eb, "eviction diverges");
		assert_eq!(
			a.drain_tier_migrations(),
			b.drain_tier_migrations(),
			"the mid-segment check ran against an out-of-segment cursor",
		);
		assert_eq!(b.tier_of(8), Some(Tier::Fast), "key 8 never left the fast segment");
		for key in 1..=9u64 {
			assert_eq!(a.tier_of(key), b.tier_of(key), "tier of {key} diverges");
			assert_eq!(a.is_midpoint(key), b.is_midpoint(key), "midpoint of {key} diverges");
		}
		gauges(&a, &b);
	}

	/// `is_main_full` is `>=`, not `>`: a main queue sitting EXACTLY on
	/// `main_capacity` is full, so `evict_one` goes straight to the main tail
	/// instead of draining the one-access tail first.
	///
	/// Nothing above can land on that byte. Every workload here uses 1024-byte
	/// objects while `main_capacity` is `(1 - ratio) * 1_000_000`, never a
	/// multiple of 1024, so `fast_used + slow_used` always steps OVER the
	/// budget rather than onto it and the two comparisons agree on every call
	/// the module ever makes.
	#[test]
	fn main_exactly_on_its_budget_counts_as_full() {
		const SMALL_MAX: CacheSize = 1_000;
		const RATIO: f64 = 0.5;
		const OBJECT: ObjectSize = 10;

		// A main budget of exactly 500 bytes, filled by exactly 50 objects.
		let main_capacity = ((1.0 - RATIO) * SMALL_MAX as f64) as CacheSize;
		let count = main_capacity / OBJECT as CacheSize;
		assert_eq!(
			count * OBJECT as CacheSize,
			main_capacity,
			"the fixture must land ON the budget, not near it",
		);

		// A fast budget far above anything this fixture uses, so no demotion
		// interferes and the whole main total stays on the fast side.
		const ROOMY_FAST: CacheSize = 1_000_000;

		let mut a = Baseline::new(RATIO, SMALL_MAX, ROOMY_FAST).with_shared_overhead(0);
		let mut b = Compact::new(RATIO, SMALL_MAX, ROOMY_FAST).with_shared_overhead(0);

		for key in 1..=count {
			a.insert(key, OBJECT);
			b.insert(key, OBJECT);
			a.update(key);
			b.update(key);
		}

		// One key left resident in the one-access queue -- what `evict_one`
		// takes first, and only, while main is judged not-full.
		let resident = count + 1;
		a.insert(resident, OBJECT);
		b.insert(resident, OBJECT);
		a.drain_tier_migrations();
		b.drain_tier_migrations();

		assert_eq!(a.slow_bytes_used(), 0, "no demotion should have happened");
		assert_eq!(b.slow_bytes_used(), a.slow_bytes_used());
		assert_eq!(
			a.fast_bytes_used(),
			main_capacity + OBJECT as CacheSize,
			"main should sit exactly on its budget, with one object beside it in one-access",
		);
		assert_eq!(b.fast_bytes_used(), a.fast_bytes_used());

		let ea = a.evict_one();
		let eb = b.evict_one();

		assert_eq!(ea, eb, "eviction diverges");
		assert_eq!(eb, Some(1), "a main queue exactly on its budget is full: the main tail goes first");
		assert!(b.contains(resident), "the one-access resident must not have been taken instead");
		assert!(!b.is_ghost(resident), "and so must not have been ghosted");
		assert_eq!(a.contains(resident), b.contains(resident));
		assert_eq!(a.is_ghost(resident), b.is_ghost(resident));
		assert_eq!(a.drain_tier_migrations(), b.drain_tier_migrations());
		gauges(&a, &b);

		// The control, one object short of the budget: main is genuinely not
		// full and the same call takes the one-access tail, which is what makes
		// the assertion above about the boundary rather than about the route.
		let mut c = Baseline::new(RATIO, SMALL_MAX, ROOMY_FAST).with_shared_overhead(0);
		let mut d = Compact::new(RATIO, SMALL_MAX, ROOMY_FAST).with_shared_overhead(0);

		for key in 1..count {
			c.insert(key, OBJECT);
			d.insert(key, OBJECT);
			c.update(key);
			d.update(key);
		}
		c.insert(resident, OBJECT);
		d.insert(resident, OBJECT);
		c.drain_tier_migrations();
		d.drain_tier_migrations();

		// Total fast bytes less the single one-access resident IS the main
		// total, and here it is one object short of the budget.
		assert_eq!(
			c.fast_bytes_used() - OBJECT as CacheSize,
			main_capacity - OBJECT as CacheSize,
			"the control must sit one object BELOW the budget",
		);
		assert_eq!(d.fast_bytes_used(), c.fast_bytes_used());

		let ec = c.evict_one();
		let ed = d.evict_one();

		assert_eq!(ec, ed, "eviction diverges below the budget");
		assert_eq!(ed, Some(resident), "below its budget, the one-access tail goes first");
		assert!(d.contains(1), "the main tail must be untouched below the budget");
		gauges(&c, &d);
	}

	/// The smallest `(budget, size)` pair whose single object lands `fast_used`
	/// EXACTLY on the main fast segment's high watermark, with
	/// `low_bytes(budget)` strictly below it -- so a pass that does fire has
	/// somewhere to drain to -- and `high_bytes(budget - 1)` strictly below it
	/// too, so the landing belongs to this budget alone and a byte of slack
	/// anywhere cannot reproduce it. Searched upward from the smallest budget
	/// at which the two watermarks can be a whole byte apart at all. Same shape
	/// as `capacity_holding`: derived from the configured watermarks, so the
	/// fixture below holds at any watermark config. At the defaults this is
	/// `(34, 33)`.
	fn budget_landing_exactly_on_the_high_watermark() -> (CacheSize, ObjectSize) {
		assert!(
			watermarks::high() > watermarks::low(),
			"watermark config collapses the two marks; nothing can land between them",
		);

		let mut budget = (1.0 / (watermarks::high() - watermarks::low())).ceil() as CacheSize;

		loop {
			let size = watermarks::high_bytes(budget);

			if size > 0
				&& size <= ObjectSize::MAX as CacheSize
				&& watermarks::low_bytes(budget) < size
				&& watermarks::high_bytes(budget - 1) < size
			{
				return (budget, size as ObjectSize);
			}

			budget += 1;

			assert!(budget < 1 << 20, "watermark config admits no exact landing");
		}
	}

	/// The `settle_fast_tier` TRIGGER is `<=`, not `<`: a main fast segment
	/// sitting exactly ON its high watermark has not crossed it and must not
	/// start a demotion pass.
	///
	/// No fixture above can see that byte. `churn_ops` steps `fast_used` in
	/// 1024-byte jumps against multi-kilobyte budgets, and the hand-traced
	/// 10-byte fixtures go through `capacity_holding`, which deliberately picks
	/// a capacity whose watermark falls clear of every total those fixtures
	/// reach. The two comparisons therefore agree on every call ever made.
	#[test]
	fn usage_exactly_on_the_high_watermark_does_not_demote() {
		let (budget, size) = budget_landing_exactly_on_the_high_watermark();

		// `one_access_ratio` 0.0, so the carve-out is zero and the main fast
		// segment gets `budget` whole; no shared overhead and an empty ghost,
		// so the reservation is zero too and `effective_main_fast_capacity()`
		// is exactly `budget`.
		let mut a = Baseline::new(0.0, 1_000, budget).with_shared_overhead(0);
		let mut b = Compact::new(0.0, 1_000, budget).with_shared_overhead(0);

		a.insert(1, size);
		b.insert(1, size);
		a.update(1);
		b.update(1);

		let (ma, mb) = (a.drain_tier_migrations(), b.drain_tier_migrations());

		assert_eq!(
			a.fast_bytes_used(),
			watermarks::high_bytes(budget),
			"the fixture did not land on the watermark",
		);
		assert_eq!(b.fast_bytes_used(), a.fast_bytes_used());
		assert!(ma.is_empty(), "sitting exactly ON the high watermark must not trigger a pass");
		assert_eq!(ma, mb, "a demotion pass ran at exactly the high watermark");
		assert_eq!(a.tier_of(1), Some(Tier::Fast));
		assert_eq!(b.tier_of(1), a.tier_of(1));
		assert_eq!(a.slow_object_count(), 0);
		assert_eq!(b.slow_object_count(), a.slow_object_count());
		assert!(!b.is_midpoint(1), "nothing was demoted, so nothing seeded the cursor");
		assert_eq!(a.is_midpoint(1), b.is_midpoint(1));
		gauges(&a, &b);

		// The control: a second object of the same size is well past the
		// watermark, and the pass must now fire on both stacks -- proving the
		// first object really was parked ON the trigger rather than nowhere
		// near it.
		a.insert(2, size);
		b.insert(2, size);
		a.update(2);
		b.update(2);

		let (ma, mb) = (a.drain_tier_migrations(), b.drain_tier_migrations());

		assert!(!ma.is_empty(), "crossing the watermark demoted nothing; the fixture proves nothing");
		assert_eq!(ma, mb, "the demotion pass past the watermark diverges");
		assert_eq!(a.tier_of(1), Some(Tier::Slow), "the oldest fast object should have been demoted");
		assert_eq!(b.tier_of(1), a.tier_of(1));
		gauges(&a, &b);

		let (ca, cb) = cursors(&a, &b, 2);
		assert_eq!(ca, cb, "midpoint cursor diverges after the pass");
	}

	/// The `(fast_capacity, size)` pair that separates the two ways of
	/// splitting the shared-metadata reservation. Against a one-byte
	/// one-access carve-out and a one-byte reservation the one-access share
	/// floors to zero, so the whole byte is REMAINDER: charging it to the main
	/// fast segment, as `reserved.saturating_sub(one_access_share)` does,
	/// leaves that segment a byte tighter than handing it its own floored
	/// share would. `size` is picked to sit in the one-byte window between the
	/// two resulting high watermarks, with `low_bytes` of the tighter budget
	/// below it so the pass that fires has somewhere to drain to. At the
	/// default watermarks this is `(4, 2)`.
	fn budget_separating_the_reservation_remainder() -> (CacheSize, ObjectSize) {
		let mut fast_capacity: CacheSize = 4;

		loop {
			// The main fast segment's budget with the remainder charged to it,
			// and without -- the one-access carve-out is one byte in both.
			let charged = fast_capacity - 1 - 1;
			let uncharged = fast_capacity - 1;
			let size = watermarks::high_bytes(uncharged);

			if size > watermarks::high_bytes(charged)
				&& size <= ObjectSize::MAX as CacheSize
				&& watermarks::low_bytes(charged) < size
			{
				return (fast_capacity, size as ObjectSize);
			}

			fast_capacity += 1;

			assert!(fast_capacity < 1 << 20, "watermark config never separates the two splittings");
		}
	}

	/// `reserved_shares` hands the main fast segment the whole REMAINDER of the
	/// proportional split rather than its own floored share, so the two shares
	/// re-sum to the reservation exactly. Taking the floor on both sides
	/// instead leaves the main fast segment up to a byte LOOSER than the
	/// reservation it is supposed to be paying for.
	///
	/// Every reservation fixture above runs a 112-byte overhead against
	/// kilobyte-scale capacities, where both splittings land on the same byte
	/// and the difference is arithmetically invisible. It only surfaces when
	/// the one-access share floors to zero while a whole byte of remainder is
	/// still outstanding.
	#[test]
	fn the_reservation_remainder_is_charged_to_the_main_fast_segment() {
		let (fast_capacity, size) = budget_separating_the_reservation_remainder();

		// A `one_access_capacity` of exactly one byte, so its proportional
		// share of a one-byte reservation floors to zero.
		const SMALL_MAX: CacheSize = 1_000;
		const RATIO: f64 = 0.001;

		assert_eq!(
			(RATIO * SMALL_MAX as f64) as CacheSize,
			1,
			"the fixture needs a one-byte one-access carve-out",
		);

		// One tracked key at a shared overhead of one byte, and an empty ghost:
		// `reserved_overhead()` is exactly one byte.
		let mut a = Baseline::new(RATIO, SMALL_MAX, fast_capacity).with_shared_overhead(1);
		let mut b = Compact::new(RATIO, SMALL_MAX, fast_capacity).with_shared_overhead(1);

		a.insert(1, size);
		b.insert(1, size);

		assert_eq!(a.dram_reserved_bytes(), 1, "one tracked key at one byte, ghost empty");
		assert_eq!(b.dram_reserved_bytes(), a.dram_reserved_bytes());

		a.update(1);
		b.update(1);

		let (ma, mb) = (a.drain_tier_migrations(), b.drain_tier_migrations());

		assert_eq!(
			ma,
			vec![(1u64, Tier::Slow)],
			"the remainder must tighten the main fast segment enough to demote this object",
		);
		assert_eq!(ma, mb, "the reservation split diverges");
		assert_eq!(a.tier_of(1), Some(Tier::Slow));
		assert_eq!(b.tier_of(1), a.tier_of(1));
		assert_eq!(a.fast_bytes_used(), 0);
		assert_eq!(b.fast_bytes_used(), a.fast_bytes_used());
		assert!(b.is_midpoint(1), "the demotion seeds the cursor");
		assert_eq!(a.is_midpoint(1), b.is_midpoint(1));
		gauges(&a, &b);
	}
}
