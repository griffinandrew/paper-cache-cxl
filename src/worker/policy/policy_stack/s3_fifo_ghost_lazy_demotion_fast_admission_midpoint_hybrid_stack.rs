/*
 * Copyright (c) Kia Shakiba
 *
 * This source code is licensed under the GNU AGPLv3 license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! `S3FifoGhostLazyDemotionFastAdmissionMidpointHybridStack` —
//! `S3FifoGhostLazyDemotionFastAdmissionHybridStack` with one addition: a
//! checkpoint roughly halfway through the SLOW portion of the main queue
//! that gives a reaccessed object an early second chance, instead of
//! making it wait until it reaches the eviction tail. For
//! `PaperPolicy::S3FifoGhostLazyDemotionFastAdmissionMidpointHybrid`.
//!
//! Identical to `S3FifoGhostLazyDemotionFastAdmissionHybridStack` in every
//! other respect (the fast-tier one-access queue, the ghost queue
//! lifecycle, the demotion-time reference-bit reprieve, the "contiguous
//! front run" invariant) — see that stack's module doc, and the stacks
//! beneath it, for the full picture.
//!
//! ## The new mechanic
//!
//! The slow portion of the main queue was previously a passive holding
//! area: nothing looked at an object there until it either reached the
//! eviction tail (checked by `give_second_chance`) or was promoted via a
//! ghost hit. This variant adds one more checkpoint, positioned
//! approximately halfway between the fast/slow boundary and the tail: if
//! the object currently sitting there has its reference bit set (i.e. it
//! was reaccessed after being demoted), it's given the exact same
//! treatment as a tail-reached second chance -- moved to the front of the
//! fast segment via the existing `give_second_chance` -- instead of
//! having to survive all the way to the tail first. An object that's
//! genuinely cold (bit clear) at the midpoint is left alone; it keeps
//! aging normally and will still get its one real chance at the tail.
//!
//! The check runs once per `evict_one()` call, after the one-access queue
//! has been confirmed empty (i.e. exactly when this stack is about to
//! evaluate the main queue for a real eviction) -- the same cadence
//! `give_second_chance`'s own tail check already runs at.
//!
//! ## Locating "the middle" without an O(n) scan
//!
//! The slow segment can hold hundreds of thousands of objects at the
//! scale this crate is benchmarked at, so recomputing its midpoint by
//! walking from the tail (or the boundary) on every check -- an O(slow
//! segment length) scan -- was rejected outright: called once per eviction
//! under steady-state pressure, that's O(n) per admission, i.e. O(n²) over
//! a cache's lifetime. `kwik::collections::HashList`'s `before()` (the
//! only directional-walk primitive `PmemHashList` also exposes, under
//! `eviction_stacks_pmem` -- `HashList::after()` exists but has no
//! `PmemHashList` counterpart, so a design relying on it would only work
//! for one of the two storage backends) makes a fresh full-segment walk
//! the only "exact" option available across both backends, which is
//! exactly the cost being avoided.
//!
//! Instead, `slow_midpoint: Option<HashedKey>` is a cursor maintained
//! incrementally, in O(1) amortized time, using only `before()`:
//!
//! * **Growth at the front** (a demotion always retags the object that was
//!   already sitting where the new `main_boundary` lands -- see
//!   `S3FifoHybridStack`'s module doc for why this never needs a real list
//!   insertion) and **shrinkage at the tail or from an arbitrary position**
//!   (a slow-tier eviction, or a promotion out of the slow segment via
//!   `give_second_chance`, including a promotion `check_slow_midpoint`
//!   itself triggers) both push the cursor's *tracked object* further from
//!   where the true middle currently is, at a rate of ~0.5 positions per
//!   event (worked out in full in the design notes below the trait impl).
//!   `bump_midpoint_drift()` accumulates this in a small counter and, once
//!   it reaches a full position's worth (every 2 qualifying events), moves
//!   the cursor one step toward the front via `nudge_midpoint_toward_front`
//!   -- the only direction ever needed, since both kinds of event drift the
//!   same way.
//! * **First demotion into an empty slow segment** initializes the cursor
//!   directly to the newly-demoted key (there's only one candidate).
//! * **The cursor's own target being removed or promoted** (explicit
//!   `remove()`, a slow-tier tail eviction, or a `give_second_chance`
//!   promotion) redirects it to the `before()` neighbor -- but only if
//!   that neighbor is *still Slow*; if it isn't (the cursor was one step
//!   from the boundary), the cursor is cleared instead of accidentally
//!   pointing into the fast segment. This redirect always runs *before*
//!   the key is physically unlinked, since `before()` needs it still
//!   linked to resolve its neighbor.
//!
//! This is a heuristic promotion trigger, not a correctness-critical exact
//! median -- "approximately halfway" is all the mechanic needs, and the
//! amortized correction keeps the cursor within a small, bounded distance
//! of the true middle without ever paying for a full rescan. See the
//! design notes after the `PolicyStack` impl for the arithmetic behind the
//! "every 2 events, one step" correction rate.

#[cfg(not(feature = "eviction_stacks_pmem"))]
use std::collections::HashMap;
#[cfg(feature = "eviction_stacks_pmem")]
use hashbrown::HashMap;

#[cfg(not(feature = "eviction_stacks_pmem"))]
use kwik::collections::HashList;
#[cfg(feature = "eviction_stacks_pmem")]
use super::pmem_collections::PmemHashList;

#[cfg(feature = "eviction_stacks_pmem")]
use crate::Hybrid;

use crate::{
	CacheSize,
	HashedKey,
	NoHasher,
	policy::PaperPolicy,
	object::ObjectSize,
	worker::policy::policy_stack::{PolicyStack, Tier},
};

/// Which live queue a key currently belongs to.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Queue {
	OneAccess,
	Main,
}

/// Combined per-key bookkeeping. `tier`/`accessed` are only meaningful
/// while `queue == Main`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct S3FifoEntry {
	queue: Queue,
	tier: Option<Tier>,
	size: ObjectSize,
	accessed: bool,
}

#[cfg(not(feature = "eviction_stacks_pmem"))]
type QueueList = HashList<HashedKey, NoHasher>;
#[cfg(feature = "eviction_stacks_pmem")]
type QueueList = PmemHashList<HashedKey, NoHasher>;

#[cfg(not(feature = "eviction_stacks_pmem"))]
type EntryMap = HashMap<HashedKey, S3FifoEntry, NoHasher>;
#[cfg(feature = "eviction_stacks_pmem")]
type EntryMap = HashMap<HashedKey, S3FifoEntry, NoHasher, Hybrid>;

pub struct S3FifoGhostLazyDemotionFastAdmissionMidpointHybridStack {
	one_access_queue: QueueList,
	main_queue: QueueList,
	ghost: QueueList,

	entries: EntryMap,

	one_access_ratio: f64,
	one_access_capacity: CacheSize,
	one_access_used: CacheSize,

	fast_capacity: CacheSize,
	fast_used: CacheSize,
	slow_used: CacheSize,

	fast_count: usize,
	main_count: usize,

	main_boundary: Option<HashedKey>,

	/// Cursor tracking (approximately) the middle of the slow segment --
	/// see the module doc's "Locating 'the middle'" section.
	slow_midpoint: Option<HashedKey>,
	/// Accumulates 0.5-position drift per qualifying event; reset (and the
	/// cursor nudged) every time it reaches 2.
	midpoint_drift: u8,

	migrations: Vec<(HashedKey, Tier)>,
}

impl S3FifoGhostLazyDemotionFastAdmissionMidpointHybridStack {
	#[cfg(not(feature = "eviction_stacks_pmem"))]
	fn new_collections() -> (QueueList, QueueList, QueueList, EntryMap) {
		(HashList::default(), HashList::default(), HashList::default(), HashMap::default())
	}

	#[cfg(feature = "eviction_stacks_pmem")]
	fn new_collections() -> (QueueList, QueueList, QueueList, EntryMap) {
		(
			PmemHashList::with_hasher(NoHasher::default()),
			PmemHashList::with_hasher(NoHasher::default()),
			PmemHashList::with_hasher(NoHasher::default()),
			HashMap::with_hasher_in(NoHasher::default(), Hybrid),
		)
	}

	pub fn new(one_access_ratio: f64, max_size: CacheSize, fast_capacity: CacheSize) -> Self {
		let (one_access_queue, main_queue, ghost, entries) = Self::new_collections();

		S3FifoGhostLazyDemotionFastAdmissionMidpointHybridStack {
			one_access_queue,
			main_queue,
			ghost,

			entries,

			one_access_ratio,
			one_access_capacity: (one_access_ratio * max_size as f64) as CacheSize,
			one_access_used: 0,

			fast_capacity,
			fast_used: 0,
			slow_used: 0,
			fast_count: 0,
			main_count: 0,

			main_boundary: None,
			slow_midpoint: None,
			midpoint_drift: 0,

			migrations: Vec::new(),
		}
	}

	fn effective_main_fast_capacity(&self) -> CacheSize {
		self.fast_capacity.saturating_sub(self.one_access_capacity)
	}

	pub fn tier_of(&self, key: HashedKey) -> Option<Tier> {
		let entry = self.entries.get(&key)?;

		match entry.queue {
			Queue::OneAccess => Some(Tier::Fast),
			Queue::Main => entry.tier,
		}
	}

	/// Returns `true` if `key` currently has a ghost entry. Exposed for tests.
	pub fn is_ghost(&self, key: HashedKey) -> bool {
		self.ghost.contains(&key)
	}

	/// Returns `true` if `key` is the current midpoint cursor target.
	/// Exposed for tests.
	pub fn is_midpoint(&self, key: HashedKey) -> bool {
		self.slow_midpoint == Some(key)
	}

	fn resize_key(&mut self, key: HashedKey, new_size: ObjectSize) {
		let Some(entry) = self.entries.get_mut(&key) else { return };

		let old_size = entry.size;
		entry.size = new_size;
		let delta = new_size as i64 - old_size as i64;

		match (entry.queue, entry.tier) {
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
		match self.entries.get(&key).map(|entry| entry.queue) {
			Some(Queue::OneAccess) => self.promote_from_one_access(key),
			Some(Queue::Main) => self.mark_accessed(key),
			None => {},
		}
	}

	fn mark_accessed(&mut self, key: HashedKey) {
		if let Some(entry) = self.entries.get_mut(&key) {
			entry.accessed = true;
		}
	}

	fn promote_from_one_access(&mut self, key: HashedKey) {
		let Some(entry) = self.entries.get(&key) else { return };
		let size = entry.size;
		let size_bytes = size as CacheSize;

		self.one_access_queue.remove(&key);
		self.one_access_used = self.one_access_used.saturating_sub(size_bytes);

		self.main_queue.push_front(key);
		self.entries.insert(key, S3FifoEntry {
			queue: Queue::Main,
			tier: Some(Tier::Fast),
			size,
			accessed: false,
		});
		self.fast_used += size_bytes;
		self.fast_count += 1;
		self.main_count += 1;

		if self.main_boundary.is_none() {
			self.main_boundary = Some(key);
		}

		self.settle_fast_tier();
	}

	fn admit_via_ghost_hit(&mut self, key: HashedKey, size: ObjectSize) {
		self.main_queue.push_front(key);
		self.entries.insert(key, S3FifoEntry {
			queue: Queue::Main,
			tier: Some(Tier::Fast),
			size,
			accessed: false,
		});
		self.fast_used += size as CacheSize;
		self.fast_count += 1;
		self.main_count += 1;

		if self.main_boundary.is_none() {
			self.main_boundary = Some(key);
		}

		self.settle_fast_tier();
	}

	/// Moves the midpoint cursor one step toward the front, if possible.
	/// No-op if the cursor is empty, or if the neighbor toward the front
	/// turns out to already be Fast (the cursor has reached the boundary)
	/// -- it just stays put until the next nudge, once growth or shrinkage
	/// resumes making room to move again.
	fn nudge_midpoint_toward_front(&mut self) {
		let Some(current) = self.slow_midpoint else { return };
		let Some(&candidate) = self.main_queue.before(&current) else { return };

		if self.entries.get(&candidate).and_then(|entry| entry.tier) == Some(Tier::Slow) {
			self.slow_midpoint = Some(candidate);
		}
	}

	/// Call after any event that changes the slow segment's size by
	/// exactly one in either direction (a demotion, a slow-tier eviction,
	/// or a promotion/removal out of the slow segment) once the cursor is
	/// already initialized. See the module doc's design notes for the
	/// "every 2 events, one step" derivation.
	fn bump_midpoint_drift(&mut self) {
		self.midpoint_drift += 1;

		if self.midpoint_drift >= 2 {
			self.midpoint_drift = 0;
			self.nudge_midpoint_toward_front();
		}
	}

	/// If `key` is currently the midpoint cursor's target, redirects it to
	/// the `before()` neighbor (only accepted if still Slow-tagged) before
	/// `key` is physically unlinked or moved. Must be called while `key`
	/// is still linked in `main_queue` -- `before()` needs that to resolve
	/// the neighbor.
	fn redirect_midpoint_before_removing(&mut self, key: HashedKey) {
		if self.slow_midpoint != Some(key) {
			return;
		}

		self.slow_midpoint = self.main_queue.before(&key)
			.copied()
			.filter(|candidate| self.entries.get(candidate).and_then(|entry| entry.tier) == Some(Tier::Slow));
	}

	/// Checks the midpoint cursor's reference bit and, if set, gives it an
	/// early second chance -- the whole point of this variant. No-op if
	/// the slow segment is currently empty. Called once per `evict_one`
	/// pass over the main queue.
	fn check_slow_midpoint(&mut self) {
		let Some(candidate) = self.slow_midpoint else { return };
		let accessed = self.entries.get(&candidate).map(|entry| entry.accessed).unwrap_or(false);

		if accessed {
			self.give_second_chance(candidate);
		}
	}

	/// The eviction-time second chance -- also reused directly by
	/// `check_slow_midpoint` for the new mid-segment check, since both are
	/// "promote this Slow key back to the front of Fast" with identical
	/// mechanics. A key reaching here with `tier == Some(Tier::Slow)`
	/// genuinely has PMEM-resident bytes, so the migration this produces
	/// is real, necessary work either way.
	fn give_second_chance(&mut self, key: HashedKey) {
		let Some(entry) = self.entries.get(&key).copied() else { return };
		let size = entry.size as CacheSize;
		let was_fast = entry.tier == Some(Tier::Fast);
		let was_boundary = was_fast && self.main_boundary == Some(key);

		if !was_fast {
			self.redirect_midpoint_before_removing(key);
		}

		let new_boundary_if_moved = if was_boundary {
			self.main_queue.before(&key).copied()
		} else {
			None
		};

		self.main_queue.move_front(&key);

		if was_boundary {
			self.main_boundary = new_boundary_if_moved;
		}

		if let Some(entry) = self.entries.get_mut(&key) {
			entry.tier = Some(Tier::Fast);
			entry.accessed = false;
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

		if self.entries.get(&key).and_then(|entry| entry.tier) == Some(Tier::Fast) {
			self.migrations.push((key, Tier::Fast));
		}
	}

	fn settle_fast_tier(&mut self) {
		let effective_capacity = self.effective_main_fast_capacity();

		while self.fast_used > effective_capacity {
			let Some(candidate) = self.main_boundary else { break };

			let accessed = self.entries.get(&candidate).map(|entry| entry.accessed).unwrap_or(false);

			if accessed {
				let new_boundary = self.main_queue.before(&candidate).copied();

				self.main_queue.move_front(&candidate);
				self.main_boundary = new_boundary;

				if let Some(entry) = self.entries.get_mut(&candidate) {
					entry.accessed = false;
				}

				continue;
			}

			let size = self.entries.get(&candidate).map(|entry| entry.size).unwrap_or(0) as CacheSize;
			let new_boundary = self.main_queue.before(&candidate).copied();

			if let Some(entry) = self.entries.get_mut(&candidate) {
				entry.tier = Some(Tier::Slow);
			}

			self.fast_used = self.fast_used.saturating_sub(size);
			self.fast_count = self.fast_count.saturating_sub(1);
			self.slow_used += size;
			self.main_boundary = new_boundary;

			self.migrations.push((candidate, Tier::Slow));

			// A real demotion always grows the slow segment by exactly one
			// -- see the module doc's design notes.
			if self.slow_midpoint.is_none() {
				self.slow_midpoint = Some(candidate);
			} else {
				self.bump_midpoint_drift();
			}
		}
	}

	fn evict_one_access_tail(&mut self) -> Option<HashedKey> {
		let key = self.one_access_queue.pop_back()?;
		let size = self.entries.remove(&key).map(|entry| entry.size).unwrap_or(0) as CacheSize;

		self.one_access_used = self.one_access_used.saturating_sub(size);
		self.ghost.push_front(key);

		Some(key)
	}

	fn trim_ghost(&mut self) {
		while self.ghost.len() > self.main_count {
			self.ghost.pop_back();
		}
	}
}

impl PolicyStack for S3FifoGhostLazyDemotionFastAdmissionMidpointHybridStack {
	fn is_policy(&self, policy: &PaperPolicy) -> bool {
		matches!(policy, PaperPolicy::S3FifoGhostLazyDemotionFastAdmissionMidpointHybrid(ratio) if *ratio == self.one_access_ratio)
	}

	fn len(&self) -> usize {
		self.entries.len()
	}

	fn contains(&self, key: HashedKey) -> bool {
		self.entries.contains_key(&key)
	}

	fn insert(&mut self, key: HashedKey, size: ObjectSize) {
		if self.entries.contains_key(&key) {
			self.resize_key(key, size);
			self.touch(key);
			return;
		}

		if self.ghost.contains(&key) {
			self.admit_via_ghost_hit(key, size);
			return;
		}

		self.one_access_queue.push_front(key);
		self.entries.insert(key, S3FifoEntry {
			queue: Queue::OneAccess,
			tier: None,
			size,
			accessed: false,
		});
		self.one_access_used += size as CacheSize;
	}

	fn update(&mut self, key: HashedKey) {
		if self.entries.contains_key(&key) {
			self.touch(key);
		}
	}

	fn remove(&mut self, key: HashedKey) {
		self.ghost.remove(&key);

		let Some(entry) = self.entries.remove(&key) else { return };
		let size = entry.size as CacheSize;

		match entry.queue {
			Queue::OneAccess => {
				self.one_access_queue.remove(&key);
				self.one_access_used = self.one_access_used.saturating_sub(size);
			},

			Queue::Main => {
				let new_boundary_if_needed = if entry.tier == Some(Tier::Fast) && self.main_boundary == Some(key) {
					self.main_queue.before(&key).copied()
				} else {
					None
				};

				// Redirect the midpoint cursor BEFORE unlinking, if this
				// key is currently its target -- `before()` needs the key
				// still linked.
				let new_midpoint_if_needed = if entry.tier == Some(Tier::Slow) && self.slow_midpoint == Some(key) {
					self.main_queue.before(&key).copied()
						.filter(|candidate| self.entries.get(candidate).and_then(|e| e.tier) == Some(Tier::Slow))
				} else {
					None
				};

				self.main_queue.remove(&key);
				self.main_count = self.main_count.saturating_sub(1);

				match entry.tier {
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
		self.settle_fast_tier();
	}

	fn clear(&mut self) {
		self.one_access_queue.clear();
		self.main_queue.clear();
		self.ghost.clear();
		self.entries.clear();

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
		if let Some(key) = self.evict_one_access_tail() {
			return Some(key);
		}

		// The new mid-segment check -- see the module doc. Runs once per
		// call, exactly when this stack is about to evaluate the main
		// queue for a real eviction.
		self.check_slow_midpoint();

		loop {
			let key = *self.main_queue.back()?;
			let accessed = self.entries.get(&key).map(|entry| entry.accessed).unwrap_or(false);

			if accessed {
				self.give_second_chance(key);
				continue;
			}

			// Redirect the midpoint cursor BEFORE unlinking, if this key
			// is currently its target.
			if self.slow_midpoint == Some(key) {
				let new_target = self.main_queue.before(&key)
					.copied()
					.filter(|candidate| self.entries.get(candidate).and_then(|e| e.tier) == Some(Tier::Slow));
				self.slow_midpoint = new_target;
			}

			self.main_queue.pop_back();
			let removed = self.entries.remove(&key);
			let size = removed.map(|entry| entry.size).unwrap_or(0) as CacheSize;
			let tier = removed.and_then(|entry| entry.tier);

			self.main_count = self.main_count.saturating_sub(1);

			match tier {
				Some(Tier::Fast) => {
					self.fast_used = self.fast_used.saturating_sub(size);
					self.fast_count = self.fast_count.saturating_sub(1);

					if self.main_boundary == Some(key) {
						self.main_boundary = self.main_queue.back().copied();
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

	fn fast_bytes_used(&self) -> CacheSize {
		self.fast_used + self.one_access_used
	}

	fn slow_bytes_used(&self) -> CacheSize {
		self.slow_used
	}

	fn fast_object_count(&self) -> usize {
		self.fast_count + self.one_access_queue.len()
	}

	fn slow_object_count(&self) -> usize {
		self.main_count - self.fast_count
	}

	fn needs_capacity_eviction(&self) -> bool {
		self.one_access_used > self.one_access_capacity
	}
}

// ── Design notes: the "every 2 events, one step" drift derivation ─────────
//
// Model the slow segment as a list of length N, positions 0 (nearest the
// fast/slow boundary) through N-1 (the tail). The true middle is at index
// N/2 (integer division). The cursor tracks a *specific object*, not an
// index -- its absolute index drifts as the segment mutates around it.
//
// Front-insertion (a demotion always lands its retagged object at index 0,
// since it was already sitting where the boundary lands -- nothing
// physically moves in the list): every existing object's index increases
// by 1, including the tracked one. The target index (N/2) increases by
// only 0.5 on average as N grows by 1. Net: the tracked object drifts +0.5
// positions past the true middle per event.
//
// Tail-removal (an eviction) or arbitrary-position removal (a promotion):
// the tracked object's own index is unaffected (removal at/after it, or
// the tracked object being the one removed and immediately redirected to
// its front-ward neighbor whose index is one less than the removed
// object's would have been), but N decreases by 1, so the target index
// (N/2) decreases by 0.5. Net: the tracked object again drifts +0.5
// positions past the true middle per event -- same sign as growth.
//
// Since both kinds of qualifying event drift the tracked object the same
// direction by the same magnitude, a single counter suffices: accumulate
// 1 per event, and every time it reaches 2 (i.e. every 2 events, matching
// 2 * 0.5 = 1 full position of accumulated drift), move the cursor one
// step toward the front via `before()` to cancel it out. Verified by hand
// against the small worked examples in this stack's unit tests below.

#[cfg(test)]
mod tests {
	use super::*;

	fn drain(stack: &mut S3FifoGhostLazyDemotionFastAdmissionMidpointHybridStack) -> Vec<(HashedKey, Tier)> {
		stack.drain_tier_migrations()
	}

	#[test]
	fn admission_always_lands_in_one_access_queue_fast() {
		let mut stack = S3FifoGhostLazyDemotionFastAdmissionMidpointHybridStack::new(1.0, 1_000, 1_000);

		stack.insert(1, 10);
		stack.insert(2, 10);

		assert_eq!(stack.tier_of(1), Some(Tier::Fast));
		assert_eq!(stack.tier_of(2), Some(Tier::Fast));
		assert_eq!(drain(&mut stack), Vec::new());
	}

	#[test]
	fn a_key_aging_out_without_reaccess_becomes_a_ghost_entry() {
		let mut stack = S3FifoGhostLazyDemotionFastAdmissionMidpointHybridStack::new(1.0, 1_000, 1_000);

		stack.insert(1, 10);
		drain(&mut stack);

		assert_eq!(stack.evict_one(), Some(1));
		assert!(stack.is_ghost(1));
	}

	#[test]
	fn ghost_hit_on_readmission_lands_in_fast_tier_without_a_migration() {
		let mut stack = S3FifoGhostLazyDemotionFastAdmissionMidpointHybridStack::new(0.0, 1_000, 1_000);

		stack.insert(1, 10);
		drain(&mut stack);
		stack.evict_one();
		assert!(stack.is_ghost(1));

		stack.insert(1, 10);

		assert_eq!(drain(&mut stack), Vec::new());
		assert_eq!(stack.tier_of(1), Some(Tier::Fast));
	}

	#[test]
	fn an_accessed_fast_boundary_key_is_reprieved_at_demotion_time_not_demoted() {
		let mut stack = S3FifoGhostLazyDemotionFastAdmissionMidpointHybridStack::new(0.0, 1_000, 10);

		stack.insert(1, 10);
		stack.update(1);
		drain(&mut stack);

		stack.update(1);
		assert_eq!(drain(&mut stack), Vec::new());

		stack.insert(2, 10);
		stack.update(2);
		let migrations = drain(&mut stack);

		assert_eq!(stack.tier_of(1), Some(Tier::Fast));
		assert_eq!(stack.tier_of(2), Some(Tier::Slow));
		assert_eq!(migrations, vec![(2, Tier::Slow)]);
	}

	// ── the signature new mechanic: a checkpoint mid-slow-segment ──────────

	/// Builds a stack with 5 keys admitted and promoted in order 1..=5,
	/// fast_capacity=20 (fits 2 objects), one_access_ratio=0.0. Traced by
	/// hand: keys 1, 2, 3 get demoted (oldest first) as keys 4 and 5
	/// arrive, leaving the slow segment as [3, 2, 1] (front-to-boundary to
	/// tail) and the fast segment as [5, 4]. After exactly 3 demotions the
	/// drift-correction cursor settles on the middle element, key 2.
	fn build_five_key_stack() -> S3FifoGhostLazyDemotionFastAdmissionMidpointHybridStack {
		let mut stack = S3FifoGhostLazyDemotionFastAdmissionMidpointHybridStack::new(0.0, 1_000, 20);

		for key in 1..=5u64 {
			stack.insert(key, 10);
			stack.update(key);
		}

		drain(&mut stack);
		stack
	}

	#[test]
	fn slow_midpoint_tracks_the_middle_of_the_slow_segment_as_it_grows() {
		let stack = build_five_key_stack();

		assert_eq!(stack.tier_of(1), Some(Tier::Slow));
		assert_eq!(stack.tier_of(2), Some(Tier::Slow));
		assert_eq!(stack.tier_of(3), Some(Tier::Slow));
		assert_eq!(stack.tier_of(4), Some(Tier::Fast));
		assert_eq!(stack.tier_of(5), Some(Tier::Fast));

		assert!(stack.is_midpoint(2), "expected key 2 (the middle of slow segment [3, 2, 1]) to be tracked");
		assert!(!stack.is_midpoint(1));
		assert!(!stack.is_midpoint(3));
	}

	#[test]
	fn a_reaccessed_midpoint_key_is_promoted_early_instead_of_waiting_for_the_tail() {
		let mut stack = build_five_key_stack();
		assert!(stack.is_midpoint(2));

		// Reaccess key 2 (currently Slow, sitting at the midpoint) without
		// otherwise touching anything -- just sets its reference bit, same
		// lazy-bit convention as everywhere else in this design.
		stack.update(2);
		assert_eq!(stack.tier_of(2), Some(Tier::Slow), "a mere access must not itself migrate or reorder");

		// evict_one() must check the midpoint before it ever looks at the
		// tail. Traced by hand: key 2 is promoted (a real Slow->Fast
		// migration); that promotion pushes fast_used back over capacity,
		// cascading a real demotion of the current boundary (key 4, the
		// only unaccessed fast key); the call then proceeds to its own
		// normal tail eviction of key 1 (the tail, unaccessed).
		let evicted = stack.evict_one();

		assert_eq!(stack.tier_of(2), Some(Tier::Fast), "the reaccessed midpoint key should have been promoted early");
		assert_eq!(stack.tier_of(4), Some(Tier::Slow), "cascading demotion after the midpoint promotion");
		assert_eq!(evicted, Some(1), "the tail should still be evicted normally in the same call");
		assert!(!stack.contains(1));
		assert!(!stack.is_ghost(1), "main-queue tail evictions never populate the ghost queue");
	}

	#[test]
	fn an_unaccessed_midpoint_key_is_left_alone() {
		let mut stack = build_five_key_stack();
		assert!(stack.is_midpoint(2));

		// Key 2's bit is clear (never reaccessed) -- evict_one() must not
		// promote it. It should proceed straight to the normal tail
		// eviction of key 1 instead.
		let evicted = stack.evict_one();

		assert_eq!(stack.tier_of(2), Some(Tier::Slow), "an unaccessed midpoint key must not be promoted");
		assert_eq!(evicted, Some(1));
	}

	#[test]
	fn removing_the_midpoint_key_directly_redirects_the_cursor() {
		let mut stack = build_five_key_stack();
		assert!(stack.is_midpoint(2));

		stack.remove(2);

		assert!(!stack.is_midpoint(2));
		assert!(stack.is_midpoint(3), "cursor should redirect to the before()-neighbor still in the slow segment");
	}

	#[test]
	fn evicting_the_only_slow_key_clears_the_midpoint_cursor() {
		let mut stack = S3FifoGhostLazyDemotionFastAdmissionMidpointHybridStack::new(0.0, 1_000, 10);

		stack.insert(1, 10);
		stack.update(1);
		stack.insert(2, 10);
		stack.update(2); // demotes key 1 -- the only slow key, so it's also the midpoint
		drain(&mut stack);
		assert!(stack.is_midpoint(1));

		let evicted = stack.evict_one();

		assert_eq!(evicted, Some(1));
		assert!(!stack.is_midpoint(1));
	}

	#[test]
	fn evict_one_gives_an_accessed_slow_key_a_second_chance() {
		let mut stack = S3FifoGhostLazyDemotionFastAdmissionMidpointHybridStack::new(0.0, 1_000, 10);

		stack.insert(1, 10);
		stack.update(1);
		drain(&mut stack);

		stack.insert(2, 10);
		stack.update(2);
		drain(&mut stack);
		assert_eq!(stack.tier_of(1), Some(Tier::Slow));

		stack.update(1);
		assert_eq!(drain(&mut stack), Vec::new());

		let evicted = stack.evict_one();

		assert_eq!(evicted, Some(2));
		assert_eq!(stack.tier_of(1), Some(Tier::Fast));
		assert_eq!(stack.contains(2), false);
	}

	#[test]
	fn remove_clears_ghost_entry_too() {
		let mut stack = S3FifoGhostLazyDemotionFastAdmissionMidpointHybridStack::new(1.0, 1_000, 1_000);

		stack.insert(1, 10);
		drain(&mut stack);
		stack.evict_one();
		assert!(stack.is_ghost(1));

		stack.remove(1);
		assert!(!stack.is_ghost(1));
	}

	#[test]
	fn remove_and_clear_reset_bookkeeping() {
		let mut stack = S3FifoGhostLazyDemotionFastAdmissionMidpointHybridStack::new(1.0, 1_000, 1_000);

		stack.insert(1, 10);
		stack.insert(2, 10);
		stack.update(1);
		drain(&mut stack);

		stack.remove(1);
		assert_eq!(stack.contains(1), false);

		stack.remove(2);
		assert_eq!(stack.contains(2), false);

		stack.insert(3, 10);
		stack.clear();

		assert_eq!(stack.len(), 0);
		assert_eq!(stack.tier_of(3), None);
		assert_eq!(stack.evict_one(), None);
		assert!(!stack.is_midpoint(3));
	}

	#[test]
	fn fast_and_slow_gauges_include_one_access_queue_on_the_fast_side() {
		let mut stack = S3FifoGhostLazyDemotionFastAdmissionMidpointHybridStack::new(1.0, 1_000, 1_000);

		stack.insert(1, 10);
		stack.insert(2, 10);
		drain(&mut stack);

		assert_eq!(stack.fast_bytes_used(), 20);
		assert_eq!(stack.slow_bytes_used(), 0);
		assert_eq!(stack.fast_object_count(), 2);
		assert_eq!(stack.slow_object_count(), 0);
	}
}
