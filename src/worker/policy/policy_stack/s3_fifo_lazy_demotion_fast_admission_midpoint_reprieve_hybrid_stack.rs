/*
 * Copyright (c) Kia Shakiba
 *
 * This source code is licensed under the GNU AGPLv3 license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! `S3FifoLazyDemotionFastAdmissionMidpointReprieveHybridStack` —
//! `S3FifoGhostLazyDemotionFastAdmissionMidpointHybridStack` with two
//! behavioral changes, for
//! `PaperPolicy::S3FifoLazyDemotionFastAdmissionMidpointReprieveHybrid`:
//!
//! 1. **No ghost queue.** A one-access-queue key that ages out without a
//!    second access is no longer evicted at all -- see point 2 -- so there
//!    is no longer any event that ever populates a ghost list. Rather than
//!    keep a permanently-empty structure around, the ghost list and every
//!    piece of machinery that only existed to serve it
//!    (`admit_via_ghost_hit`, `is_ghost`, `trim_ghost`, the
//!    `ghost.contains()` admission check) are removed outright.
//! 2. **The one-access queue's tail is reprieved, not evicted.** Once
//!    `one_access_used` exceeds `one_access_capacity`, the tail key is
//!    moved directly into the slow tier of the main queue -- given a full
//!    life there, promotable via the ordinary `touch()`/midpoint/tail
//!    second-chance machinery -- instead of being permanently removed from
//!    the cache.
//!
//!    Critically, this relief runs *synchronously* from `insert()`/`resize()`
//!    (a new `settle_one_access()`, mirroring `settle_fast_tier()`'s
//!    relationship to the main queue's fast/slow boundary) -- **not** through
//!    `evict_one()`/`needs_capacity_eviction()`, even though that's the
//!    mechanism the predecessor variant's `evict_one_access_tail` used. The
//!    first draft of this stack routed it through `evict_one()` and hit a
//!    real bug: `PolicyWorker::apply_evictions`'s loop calls `evict_one()`
//!    whenever `needs_capacity_eviction()` is true and unconditionally
//!    erases whatever key it returns from the *entire cache* -- and if it
//!    returns `None`, `erase()`'s own fallback evicts a *random* object
//!    instead (see its doc comment: "the policy has run out of keys to
//!    evict... fall back to evicting a random object"). A reprieve is
//!    neither of those things: nothing should be permanently removed from
//!    the cache just because the one-access queue needed relief, and
//!    `over_max_size` might not even be true at that moment. Fixed by
//!    moving the relief to the same synchronous-settle pattern this whole
//!    hybrid family already uses for its OTHER internal capacity boundary
//!    (`settle_fast_tier`), which never touches `evict_one()` either --
//!    `evict_one()` in this stack is therefore purely about the main queue
//!    (the midpoint check plus the ordinary tail loop), and
//!    `needs_capacity_eviction()` stays at the trait's default `false`.
//!
//! Otherwise identical to
//! `S3FifoGhostLazyDemotionFastAdmissionMidpointHybridStack` (the fast-tier
//! one-access queue, the demotion-time reference-bit reprieve, the
//! mid-slow-segment checkpoint) -- see that stack's module doc, and the
//! stacks beneath it, for the full picture.
//!
//! ## Two physical main-queue lists, not one list plus a boundary cursor
//!
//! Every stack in this family up to and including the predecessor keeps the
//! main queue as a *single* `HashList` with a `main_boundary: Option<HashedKey>`
//! cursor marking the oldest still-fast key -- the fast tier is then the
//! contiguous prefix from the list's head up to and including that cursor,
//! and demotion is a pure relabel (flip `tier` to `Slow`, step the cursor one
//! position toward the front via `before()`) with nothing physically moving.
//! That trick is elegant *when demotion is the only thing that ever crosses
//! the boundary*, which was true for every predecessor.
//!
//! This variant breaks that premise: a reprieve has to *insert a brand-new
//! node* at the boundary, and neither `kwik::collections::HashList` nor
//! `PmemHashList` exposes an `insert_after`/`insert_before` primitive on the
//! key-addressed API -- only `push_front`, `push_back`, `move_front`, and
//! `move_back`, all of which operate on the list's absolute ends. The first
//! implementation of this stack worked around that by walking every
//! currently-fast key (`before()` from the boundary), `push_front`ing the new
//! key, then replaying all of them through `move_front` to restore their
//! order -- correct (its ordering was verified by a dedicated unit test), but
//! **O(number of currently-fast keys) per reprieve**. That is fine at
//! unit-test scale and catastrophic at real scale: benchmarked against a real
//! trace with a 6 GB fast tier, the worker thread burned ~18 minutes of CPU
//! without completing a single run, since reprieves fire continuously and the
//! fast tier holds tens of thousands of keys.
//!
//! Splitting the main queue into two physically separate lists removes the
//! problem outright rather than optimizing around it:
//!
//! * `main_fast` -- front = newest, back = oldest fast key (the demotion
//!   candidate, previously `main_boundary`).
//! * `main_slow` -- front = newest slow key (i.e. *exactly* the fast/slow
//!   boundary position), back = oldest slow key (the eviction candidate).
//!
//! The boundary is no longer a cursor into a shared list; it's just the front
//! of `main_slow`. So a reprieve is a plain `main_slow.push_front(key)` --
//! **O(1)**, landing precisely where the old code needed an O(n) walk to put
//! it, and with no risk of corrupting a boundary pointer (there isn't one).
//! Every other boundary-crossing operation gets simpler the same way:
//! demotion is `main_fast.pop_back()` + `main_slow.push_front()`, promotion is
//! `main_slow.remove()` + `main_fast.push_front()`, and eviction is
//! `main_slow.pop_back()` (falling back to `main_fast.pop_back()` only when
//! nothing has ever been demoted). All O(1).
//!
//! This also makes the midpoint cursor strictly cleaner: because `main_slow`
//! is homogeneous, a `before()` walk inside it can never wander into
//! fast-tagged territory, so the "only accept the neighbor if it's still
//! `Tier::Slow`" filter the predecessor needed at every cursor-redirect site
//! disappears entirely.
//!
//! This is not a novel shape for this crate -- `LfuHybridStack` already keeps
//! two independent frequency chains for the same reason, and
//! `LruSizedHybridStack`'s module doc records reaching the same conclusion
//! (four homogeneous lists turned out *simpler* than any cursor-based scheme
//! once more than one segment pair was involved).
//!
//! ## Locating "the middle"
//!
//! Unchanged in substance from the predecessor: `slow_midpoint` is a cursor
//! tracking (approximately) the middle of `main_slow`, maintained in O(1)
//! amortized time via a small drift counter rather than a rescan. See the
//! design notes below the `PolicyStack` impl for the arithmetic.

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
/// while `queue == Main`. `tier` is redundant with which of the two main
/// lists the key is physically in, but kept because `tier_of()` and the
/// `PolicyWorker` migration path both want it as a cheap map lookup rather
/// than a pair of `contains()` probes.
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

pub struct S3FifoLazyDemotionFastAdmissionMidpointReprieveHybridStack {
	one_access_queue: QueueList,

	/// Main queue, fast portion. Front = newest, back = oldest (the
	/// demotion candidate).
	main_fast: QueueList,
	/// Main queue, slow portion. Front = newest slow key -- i.e. exactly
	/// the fast/slow boundary position -- back = oldest (the eviction
	/// candidate).
	main_slow: QueueList,

	entries: EntryMap,

	one_access_ratio: f64,
	one_access_capacity: CacheSize,
	one_access_used: CacheSize,

	fast_capacity: CacheSize,
	fast_used: CacheSize,
	slow_used: CacheSize,

	/// Cursor tracking (approximately) the middle of `main_slow` -- see the
	/// design notes below the `PolicyStack` impl.
	slow_midpoint: Option<HashedKey>,
	/// Accumulates 0.5-position drift per qualifying event; reset (and the
	/// cursor nudged) every time it reaches 2.
	midpoint_drift: u8,

	migrations: Vec<(HashedKey, Tier)>,
}

impl S3FifoLazyDemotionFastAdmissionMidpointReprieveHybridStack {
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
		let (one_access_queue, main_fast, main_slow, entries) = Self::new_collections();

		S3FifoLazyDemotionFastAdmissionMidpointReprieveHybridStack {
			one_access_queue,
			main_fast,
			main_slow,

			entries,

			one_access_ratio,
			one_access_capacity: (one_access_ratio * max_size as f64) as CacheSize,
			one_access_used: 0,

			fast_capacity,
			fast_used: 0,
			slow_used: 0,

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

		self.main_fast.push_front(key);
		self.entries.insert(key, S3FifoEntry {
			queue: Queue::Main,
			tier: Some(Tier::Fast),
			size,
			accessed: false,
		});
		self.fast_used += size_bytes;

		self.settle_fast_tier();
	}

	/// Moves the midpoint cursor one step toward the front of `main_slow`.
	/// No-op if the cursor is empty or already at the front. Unlike the
	/// predecessor's equivalent, no tier check is needed: `main_slow` is
	/// homogeneous, so every neighbor within it is Slow by construction.
	fn nudge_midpoint_toward_front(&mut self) {
		let Some(current) = self.slow_midpoint else { return };

		if let Some(&candidate) = self.main_slow.before(&current) {
			self.slow_midpoint = Some(candidate);
		}
	}

	/// Call after any event that changes `main_slow`'s length by exactly one
	/// in either direction (a demotion, a reprieve, a slow-tier eviction, or
	/// a promotion/removal out of the slow segment) once the cursor is
	/// already initialized.
	fn bump_midpoint_drift(&mut self) {
		self.midpoint_drift += 1;

		if self.midpoint_drift >= 2 {
			self.midpoint_drift = 0;
			self.nudge_midpoint_toward_front();
		}
	}

	/// If `key` is currently the midpoint cursor's target, redirects it to
	/// the `before()` neighbor before `key` is unlinked from `main_slow`.
	/// Must be called while `key` is still linked -- `before()` needs that
	/// to resolve the neighbor.
	fn redirect_midpoint_before_removing(&mut self, key: HashedKey) {
		if self.slow_midpoint != Some(key) {
			return;
		}

		self.slow_midpoint = self.main_slow.before(&key).copied();
	}

	/// Checks the midpoint cursor's reference bit and, if set, gives it an
	/// early second chance. No-op if the slow segment is currently empty.
	/// Called once per `evict_one` pass over the main queue.
	fn check_slow_midpoint(&mut self) {
		let Some(candidate) = self.slow_midpoint else { return };
		let accessed = self.entries.get(&candidate).map(|entry| entry.accessed).unwrap_or(false);

		if accessed {
			self.give_second_chance(candidate);
		}
	}

	/// The eviction-time second chance -- also reused directly by
	/// `check_slow_midpoint` for the mid-segment check, since both are
	/// "this key's reference bit is set, so spare it and move it to the
	/// front of the fast list" with identical mechanics.
	fn give_second_chance(&mut self, key: HashedKey) {
		let Some(entry) = self.entries.get(&key).copied() else { return };
		let size = entry.size as CacheSize;

		match entry.tier {
			// Already fast (only reachable from `evict_one`'s fast-tail
			// fallback, i.e. nothing has ever been demoted): just reorder
			// to the front, no tier change and no byte movement.
			Some(Tier::Fast) => {
				self.main_fast.move_front(&key);

				if let Some(entry) = self.entries.get_mut(&key) {
					entry.accessed = false;
				}
			},

			Some(Tier::Slow) => {
				self.redirect_midpoint_before_removing(key);

				self.main_slow.remove(&key);
				self.main_fast.push_front(key);

				if let Some(entry) = self.entries.get_mut(&key) {
					entry.tier = Some(Tier::Fast);
					entry.accessed = false;
				}

				self.slow_used = self.slow_used.saturating_sub(size);
				self.fast_used += size;

				self.bump_midpoint_drift();
			},

			None => return,
		}

		self.settle_fast_tier();

		// Only record a migration if the key actually ended up Fast -- the
		// `settle_fast_tier` above can immediately demote it right back out
		// when the fast tier is at capacity, in which case that call has
		// already pushed the correct `Tier::Slow` migration itself.
		if self.entries.get(&key).and_then(|entry| entry.tier) == Some(Tier::Fast) {
			self.migrations.push((key, Tier::Fast));
		}
	}

	/// Demotes oldest-first from `main_fast` into `main_slow` until the
	/// effective budget is met, giving any key whose reference bit is set a
	/// reprieve (moved to the front of `main_fast`, bit cleared) instead.
	/// Terminates even when every fast key's bit is set, since each reprieve
	/// clears one bit.
	fn settle_fast_tier(&mut self) {
		let effective_capacity = self.effective_main_fast_capacity();

		while self.fast_used > effective_capacity {
			let Some(candidate) = self.main_fast.back().copied() else { break };

			let accessed = self.entries.get(&candidate).map(|entry| entry.accessed).unwrap_or(false);

			if accessed {
				self.main_fast.move_front(&candidate);

				if let Some(entry) = self.entries.get_mut(&candidate) {
					entry.accessed = false;
				}

				continue;
			}

			let size = self.entries.get(&candidate).map(|entry| entry.size).unwrap_or(0) as CacheSize;

			self.main_fast.pop_back();
			self.main_slow.push_front(candidate);

			if let Some(entry) = self.entries.get_mut(&candidate) {
				entry.tier = Some(Tier::Slow);
			}

			self.fast_used = self.fast_used.saturating_sub(size);
			self.slow_used += size;

			self.migrations.push((candidate, Tier::Slow));

			// A real demotion always grows the slow segment by exactly one.
			if self.slow_midpoint.is_none() {
				self.slow_midpoint = Some(candidate);
			} else {
				self.bump_midpoint_drift();
			}
		}
	}

	/// Relieves one-access-queue pressure by moving its tail(s) to the front
	/// of `main_slow` -- which *is* the fast/slow boundary position, so this
	/// is a plain O(1) `push_front` (see the module doc's "Two physical
	/// main-queue lists" section for why this used to be an O(n) walk).
	/// Called synchronously from `insert()`/`resize()`, exactly mirroring
	/// `settle_fast_tier()`'s relationship to the fast/slow boundary. A pure
	/// internal migration: nothing is ever removed from the cache here, so
	/// this must never be routed through `evict_one()`/
	/// `needs_capacity_eviction()` -- see the module doc for the bug that
	/// caused.
	fn settle_one_access(&mut self) {
		while self.one_access_used > self.one_access_capacity {
			let Some(key) = self.one_access_queue.pop_back() else { break };
			let Some(entry) = self.entries.get(&key).copied() else { continue };
			let size = entry.size as CacheSize;

			self.one_access_used = self.one_access_used.saturating_sub(size);

			self.main_slow.push_front(key);

			if let Some(stored) = self.entries.get_mut(&key) {
				stored.queue = Queue::Main;
				stored.tier = Some(Tier::Slow);
				stored.accessed = false;
			}

			self.slow_used += size;

			self.migrations.push((key, Tier::Slow));

			// Grows the slow segment by exactly one, same as a real demotion.
			if self.slow_midpoint.is_none() {
				self.slow_midpoint = Some(key);
			} else {
				self.bump_midpoint_drift();
			}
		}
	}
}

impl PolicyStack for S3FifoLazyDemotionFastAdmissionMidpointReprieveHybridStack {
	fn is_policy(&self, policy: &PaperPolicy) -> bool {
		matches!(policy, PaperPolicy::S3FifoLazyDemotionFastAdmissionMidpointReprieveHybrid(ratio) if *ratio == self.one_access_ratio)
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

		self.one_access_queue.push_front(key);
		self.entries.insert(key, S3FifoEntry {
			queue: Queue::OneAccess,
			tier: None,
			size,
			accessed: false,
		});
		self.one_access_used += size as CacheSize;

		self.settle_one_access();
	}

	fn update(&mut self, key: HashedKey) {
		if self.entries.contains_key(&key) {
			self.touch(key);
		}
	}

	fn remove(&mut self, key: HashedKey) {
		let Some(entry) = self.entries.remove(&key) else { return };
		let size = entry.size as CacheSize;

		match entry.queue {
			Queue::OneAccess => {
				self.one_access_queue.remove(&key);
				self.one_access_used = self.one_access_used.saturating_sub(size);
			},

			Queue::Main => match entry.tier {
				Some(Tier::Fast) => {
					self.main_fast.remove(&key);
					self.fast_used = self.fast_used.saturating_sub(size);
				},

				Some(Tier::Slow) => {
					// Redirect the cursor BEFORE unlinking -- `before()`
					// needs the key still linked.
					self.redirect_midpoint_before_removing(key);

					self.main_slow.remove(&key);
					self.slow_used = self.slow_used.saturating_sub(size);

					self.bump_midpoint_drift();
				},

				None => {},
			},
		}
	}

	fn resize(&mut self, max_size: CacheSize) {
		self.one_access_capacity = (self.one_access_ratio * max_size as f64) as CacheSize;
		self.settle_one_access();
		self.settle_fast_tier();
	}

	fn clear(&mut self) {
		self.one_access_queue.clear();
		self.main_fast.clear();
		self.main_slow.clear();
		self.entries.clear();

		self.one_access_used = 0;
		self.fast_used = 0;
		self.slow_used = 0;
		self.slow_midpoint = None;
		self.midpoint_drift = 0;
		self.migrations.clear();
	}

	fn evict_one(&mut self) -> Option<HashedKey> {
		// The one-access queue never reaches here -- its own capacity
		// pressure is relieved synchronously by `settle_one_access()` (see
		// the module doc), the same way the main queue's fast/slow boundary
		// is settled by `settle_fast_tier()` rather than through eviction.
		// This is purely about the main queue: the midpoint check, then the
		// ordinary tail loop.
		self.check_slow_midpoint();

		loop {
			// The slow tail is the real eviction candidate; fall back to
			// the fast tail only when nothing has ever been demoted.
			let (key, from_slow) = match self.main_slow.back().copied() {
				Some(key) => (key, true),
				None => (self.main_fast.back().copied()?, false),
			};

			let accessed = self.entries.get(&key).map(|entry| entry.accessed).unwrap_or(false);

			if accessed {
				self.give_second_chance(key);
				continue;
			}

			if from_slow {
				// Redirect the cursor BEFORE unlinking.
				self.redirect_midpoint_before_removing(key);
				self.main_slow.pop_back();
			} else {
				self.main_fast.pop_back();
			}

			let removed = self.entries.remove(&key);
			let size = removed.map(|entry| entry.size).unwrap_or(0) as CacheSize;

			if from_slow {
				self.slow_used = self.slow_used.saturating_sub(size);
				self.bump_midpoint_drift();
			} else {
				self.fast_used = self.fast_used.saturating_sub(size);
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

	fn fast_bytes_used(&self) -> CacheSize {
		self.fast_used + self.one_access_used
	}

	fn slow_bytes_used(&self) -> CacheSize {
		self.slow_used
	}

	fn fast_object_count(&self) -> usize {
		self.main_fast.len() + self.one_access_queue.len()
	}

	fn slow_object_count(&self) -> usize {
		self.main_slow.len()
	}
}

// ── Design notes: the "every 2 events, one step" drift derivation ─────────
//
// Model `main_slow` as a list of length N, positions 0 (the front, i.e. the
// fast/slow boundary) through N-1 (the tail). The true middle is at index
// N/2 (integer division). The cursor tracks a *specific object*, not an
// index -- its absolute index drifts as the list mutates around it.
//
// Front-insertion (a demotion or a reprieve, both `main_slow.push_front`):
// every existing object's index increases by 1, including the tracked one.
// The target index (N/2) increases by only 0.5 on average as N grows by 1.
// Net: the tracked object drifts +0.5 positions past the true middle per
// event.
//
// Tail-removal (an eviction) or arbitrary-position removal (a promotion):
// the tracked object's own index is unaffected (removal at/after it, or the
// tracked object being the one removed and immediately redirected to its
// front-ward neighbor, whose index is one less than the removed object's
// would have been), but N decreases by 1, so the target index (N/2)
// decreases by 0.5. Net: the tracked object again drifts +0.5 positions past
// the true middle per event -- same sign as growth.
//
// Since both kinds of qualifying event drift the tracked object the same
// direction by the same magnitude, a single counter suffices: accumulate 1
// per event, and every time it reaches 2 (i.e. every 2 events, matching
// 2 * 0.5 = 1 full position of accumulated drift), move the cursor one step
// toward the front via `before()` to cancel it out. Verified by hand against
// the small worked examples in this stack's unit tests below.

#[cfg(test)]
mod tests {
	use super::*;

	fn drain(stack: &mut S3FifoLazyDemotionFastAdmissionMidpointReprieveHybridStack) -> Vec<(HashedKey, Tier)> {
		stack.drain_tier_migrations()
	}

	#[test]
	fn admission_always_lands_in_one_access_queue_fast() {
		let mut stack = S3FifoLazyDemotionFastAdmissionMidpointReprieveHybridStack::new(1.0, 1_000, 1_000);

		stack.insert(1, 10);
		stack.insert(2, 10);

		assert_eq!(stack.tier_of(1), Some(Tier::Fast));
		assert_eq!(stack.tier_of(2), Some(Tier::Fast));
		assert_eq!(drain(&mut stack), Vec::new());
	}

	#[test]
	fn a_key_aging_out_without_reaccess_is_moved_to_slow_instead_of_evicted() {
		// one_access_capacity = 0.01 * 1_000 = 10 -- fits exactly one 10-byte
		// key. Admitting a second pushes one_access_used to 20 > 10,
		// synchronously reprieving the oldest (key 1) from insert() itself.
		let mut stack = S3FifoLazyDemotionFastAdmissionMidpointReprieveHybridStack::new(0.01, 1_000, 1_000);

		stack.insert(1, 10);
		drain(&mut stack);
		assert_eq!(stack.tier_of(1), Some(Tier::Fast), "still in the one-access queue");

		stack.insert(2, 10);

		assert!(stack.contains(1), "the key must still be tracked, not gone");
		assert_eq!(stack.tier_of(1), Some(Tier::Slow), "aged-out key should land directly in the main queue's slow tier");
		assert_eq!(stack.tier_of(2), Some(Tier::Fast), "the newer key stays in the one-access queue");

		let migrations = drain(&mut stack);
		assert_eq!(migrations, vec![(1, Tier::Slow)], "a real Fast(DRAM)->Slow(PMEM) migration must still be recorded");
	}

	#[test]
	fn a_reprieved_key_can_still_be_promoted_by_a_later_access() {
		let mut stack = S3FifoLazyDemotionFastAdmissionMidpointReprieveHybridStack::new(0.01, 1_000, 1_000);

		// one_access_capacity = 0.01 * 1_000 = 10, fits exactly one key.
		// insert(2) pushes past it, reprieving key 1 (the oldest); insert(3)
		// pushes past it again, reprieving key 2 -- leaving key 3 sitting
		// safely in the one-access queue (untouched, under capacity) and
		// both 1 and 2 in main_slow, in that order: main_slow = [2, 1]
		// (2 at the front/freshest, 1 at the tail).
		stack.insert(1, 10);
		stack.insert(2, 10);
		stack.insert(3, 10);
		drain(&mut stack);

		assert_eq!(stack.tier_of(1), Some(Tier::Slow));
		assert_eq!(stack.tier_of(2), Some(Tier::Slow));
		assert_eq!(stack.tier_of(3), Some(Tier::Fast), "still sitting untouched in the one-access queue");

		// Re-access key 1 (the tail): sets the reference bit but must not
		// itself move or migrate it yet.
		stack.update(1);
		assert_eq!(stack.tier_of(1), Some(Tier::Slow));
		assert_eq!(drain(&mut stack), Vec::new());

		// The tail check finds key 1's bit set and gives it a second
		// chance instead of evicting it; eviction then proceeds to the
		// real (still genuinely cold) tail, key 2, in the same call.
		let evicted = stack.evict_one();

		assert_eq!(evicted, Some(2));
		assert_eq!(stack.tier_of(1), Some(Tier::Fast), "the reprieved key should have been promoted via the ordinary second chance");
	}

	#[test]
	fn reprieve_does_not_disturb_existing_fast_key_order() {
		// A comfortable one-access budget (ratio 1.0) during setup, so keys
		// 1-3 each safely survive their own insert()'s settle_one_access
		// before the very next line's update() promotes them via touch().
		// fast_capacity is set well above one_access_capacity (1_000) too --
		// effective_main_fast_capacity is fast_capacity minus
		// one_access_capacity, so leaving them equal would zero it out and
		// demote every promoted key immediately.
		let mut stack = S3FifoLazyDemotionFastAdmissionMidpointReprieveHybridStack::new(1.0, 1_000, 10_000);

		for key in 1..=3u64 {
			stack.insert(key, 10);
			stack.update(key);
		}
		drain(&mut stack);

		assert_eq!(stack.tier_of(1), Some(Tier::Fast));
		assert_eq!(stack.tier_of(2), Some(Tier::Fast));
		assert_eq!(stack.tier_of(3), Some(Tier::Fast));

		// Admit a fourth key that stays in the one-access queue (never
		// touched), then shrink the one-access budget to 0 -- forcing
		// settle_one_access to move it into main_slow synchronously, from
		// within this resize() call.
		stack.insert(4, 10);
		assert_eq!(stack.tier_of(4), Some(Tier::Fast), "still sitting untouched in the one-access queue");

		stack.resize(0);

		assert_eq!(stack.tier_of(4), Some(Tier::Slow));
		assert_eq!(drain(&mut stack), vec![(4, Tier::Slow)]);

		// The three original fast keys must all still be Fast and still in
		// their original oldest-first order -- shrink the fast budget to 0
		// and confirm every one demotes in that order, none skipped.
		stack.resize_fast_tier(0);
		let migrations = drain(&mut stack);

		assert_eq!(migrations, vec![(1, Tier::Slow), (2, Tier::Slow), (3, Tier::Slow)], "demotion order must be oldest-first, and no fast key may be silently skipped");
		assert_eq!(stack.tier_of(1), Some(Tier::Slow));
		assert_eq!(stack.tier_of(2), Some(Tier::Slow));
		assert_eq!(stack.tier_of(3), Some(Tier::Slow));
		assert_eq!(stack.fast_bytes_used(), 0);
	}

	#[test]
	fn reprieve_never_counts_toward_fast_bytes_used() {
		let mut stack = S3FifoLazyDemotionFastAdmissionMidpointReprieveHybridStack::new(0.0, 1_000, 1_000);

		stack.insert(1, 10);

		assert_eq!(stack.fast_bytes_used(), 0, "a reprieved key must never be counted as fast, even transiently");
		assert_eq!(stack.slow_bytes_used(), 10);
	}

	#[test]
	fn an_accessed_fast_boundary_key_is_reprieved_at_demotion_time_not_demoted() {
		let mut stack = S3FifoLazyDemotionFastAdmissionMidpointReprieveHybridStack::new(1.0, 1_000, 1_010);

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

	// ── the mid-slow-segment checkpoint ────────────────────────────────────

	fn build_five_key_stack() -> S3FifoLazyDemotionFastAdmissionMidpointReprieveHybridStack {
		let mut stack = S3FifoLazyDemotionFastAdmissionMidpointReprieveHybridStack::new(1.0, 1_000, 1_020);

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
	}

	#[test]
	fn a_reaccessed_midpoint_key_is_promoted_early_instead_of_waiting_for_the_tail() {
		let mut stack = build_five_key_stack();
		assert!(stack.is_midpoint(2));

		stack.update(2);
		assert_eq!(stack.tier_of(2), Some(Tier::Slow), "a mere access must not itself migrate or reorder");

		let evicted = stack.evict_one();

		assert_eq!(stack.tier_of(2), Some(Tier::Fast), "the reaccessed midpoint key should have been promoted early");
		assert_eq!(stack.tier_of(4), Some(Tier::Slow), "cascading demotion after the midpoint promotion");
		assert_eq!(evicted, Some(1), "the tail should still be evicted normally in the same call");
		assert!(!stack.contains(1));
	}

	#[test]
	fn an_unaccessed_midpoint_key_is_left_alone() {
		let mut stack = build_five_key_stack();
		assert!(stack.is_midpoint(2));

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
		assert!(stack.is_midpoint(3), "cursor should redirect to the before()-neighbor in main_slow");
	}

	#[test]
	fn evict_one_gives_an_accessed_slow_key_a_second_chance() {
		let mut stack = S3FifoLazyDemotionFastAdmissionMidpointReprieveHybridStack::new(1.0, 1_000, 1_010);

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
	fn evict_one_falls_back_to_the_fast_tail_when_nothing_has_ever_been_demoted() {
		let mut stack = S3FifoLazyDemotionFastAdmissionMidpointReprieveHybridStack::new(1.0, 1_000, 10_000);

		for key in 1..=3u64 {
			stack.insert(key, 10);
			stack.update(key);
		}
		drain(&mut stack);

		assert_eq!(stack.slow_object_count(), 0, "nothing should have been demoted yet");

		// With main_slow empty, the oldest fast key is the only candidate.
		assert_eq!(stack.evict_one(), Some(1));
		assert!(!stack.contains(1));
		assert_eq!(stack.fast_bytes_used(), 20);
	}

	#[test]
	fn remove_and_clear_reset_bookkeeping() {
		let mut stack = S3FifoLazyDemotionFastAdmissionMidpointReprieveHybridStack::new(1.0, 1_000, 1_000);

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
		let mut stack = S3FifoLazyDemotionFastAdmissionMidpointReprieveHybridStack::new(1.0, 1_000, 1_000);

		stack.insert(1, 10);
		stack.insert(2, 10);
		drain(&mut stack);

		assert_eq!(stack.fast_bytes_used(), 20);
		assert_eq!(stack.slow_bytes_used(), 0);
		assert_eq!(stack.fast_object_count(), 2);
		assert_eq!(stack.slow_object_count(), 0);
	}
}
