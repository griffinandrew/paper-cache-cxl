/*
 * Copyright (c) Kia Shakiba
 *
 * This source code is licensed under the GNU AGPLv3 license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! `S3FifoGhostLazyDemotionFastAdmissionHybridStack` —
//! `S3FifoGhostLazyDemotionHybridStack` with one change: the one-access
//! queue now lives in the FAST tier instead of the slow tier. For
//! `PaperPolicy::S3FifoGhostLazyDemotionFastAdmissionHybrid`.
//!
//! Identical to `S3FifoGhostLazyDemotionHybridStack` in every other respect
//! (ghost queue lifecycle, the demotion-time reference-bit reprieve, the
//! eviction-time second chance, the "contiguous front run" invariant) — see
//! that stack's module doc, and `S3FifoGhostHybridStack`/`S3FifoHybridStack`
//! beneath it, for the full picture.
//!
//! ## Motivation: every admission was a synchronous PMEM write
//!
//! In the base design, admission is unconditional to the *slow* tier — the
//! literal paper rule ("every new object is placed in the slow tier"). At
//! the `PaperCache::set()` API layer this means every single admission
//! (and every one-access-queue re-admission of a ghost-recycled key)
//! synchronously builds `TieredBuffer::new_slow`, i.e. a real PMEM/UMF
//! allocation on the calling thread, before the object is even in the
//! cache. Reported by the user as a real cost worth trying to avoid: this
//! variant places the one-access queue's bytes in the FAST tier instead,
//! so admission becomes a cheap DRAM write (`TieredBuffer::new_fast`) —
//! the same kind of change `lru_hybrid_cache`'s admission already gets for
//! free by always landing fast.
//!
//! Only the one-access queue moves. The main queue keeps exactly the same
//! fast/slow segmentation, demotion-time reprieve, and eviction-time
//! second chance as `S3FifoGhostLazyDemotionHybridStack` — a key still has
//! to prove itself with a second access to earn a spot in the "real",
//! frequency-durable part of the cache; the only thing that changed is
//! which physical allocator backs its bytes *while on probation* in the
//! one-access queue.
//!
//! ## Accounting: the one-access queue now competes for the SAME DRAM budget
//!
//! This is the part that has to be handled deliberately, not just
//! relabeled. In the base design, `one_access_capacity` (`one_access_ratio
//! * max_size`) and `fast_capacity` (`fast_tier_size`) are two completely
//! independent budgets — one governs a slow/PMEM queue, the other governs
//! the main queue's fast/DRAM portion. Now that the one-access queue is
//! *also* DRAM, both budgets draw from the same physical pool, and adding
//! them naively (letting each treat the full `fast_capacity` as its own)
//! would silently let real DRAM usage grow to `fast_capacity +
//! one_access_capacity` instead of the configured `fast_capacity`.
//!
//! Fixed by treating `one_access_capacity` as a fixed reservation carved
//! out of `fast_capacity` first — `effective_main_fast_capacity()` =
//! `fast_capacity.saturating_sub(one_access_capacity)` — and having
//! `settle_fast_tier` (the main queue's own demotion trigger) check
//! against that reduced number instead of raw `fast_capacity`. The
//! one-access queue's own byte cap (`needs_capacity_eviction`) is
//! unchanged -- it was always `one_access_used > one_access_capacity`,
//! independent of tier, and still is. The net result: `fast_used (main) +
//! one_access_used ≤ fast_capacity` holds by construction (modulo the same
//! kind of transient overshoot every other stack in this crate already
//! tolerates between eviction-loop passes), so the configured fast-tier
//! size remains a real, honored bound on total DRAM, not just on the main
//! queue's share of it. `resize()` (triggered when `max_size` changes,
//! which rescales `one_access_capacity`) proactively re-runs
//! `settle_fast_tier` for the same reason `resize_fast_tier` already does
//! -- growing `one_access_capacity` shrinks the room left for the main
//! queue's fast segment, and that has to be caught immediately rather than
//! waiting for the next unrelated `insert`/`update` to notice.
//!
//! A degenerate but legitimate consequence of this: if `one_access_ratio *
//! max_size` alone meets or exceeds `fast_capacity`, the main queue's fast
//! segment gets zero (or negative, saturated to zero) room, and every
//! promotion out of the one-access queue immediately self-demotes back to
//! slow. That's correct accounting given the configuration, not a bug —
//! see `zero_effective_main_capacity_demotes_every_promotion_immediately`
//! below, mirroring the equivalent documented behavior in
//! `lru_sized_hybrid_cache`.
//!
//! ## A second optimization this unlocks: no more redundant Fast→Fast copies
//!
//! `promote_from_one_access` and `admit_via_ghost_hit` no longer need to
//! push a `(key, Tier::Fast)` migration after a successful promotion. In
//! the base design that push was load-bearing: the API layer had just
//! built the key's bytes as Slow (per the always-Slow admission rule), so
//! the migration was the ONLY thing that ever physically moved them to
//! Fast DRAM. Here, admission (see `S3FifoGhostLazyDemotionFastAdmissionHybridPolicy::admission_tier`
//! in this feature's `mod.rs`) already builds every brand-new key's bytes
//! as Fast unconditionally -- including ghost hits, which are
//! indistinguishable from any other fresh `set()` at the API layer -- so a
//! one-access-queue entry's buffer is *already* physically Fast for its
//! entire lifetime in that queue, before it's ever promoted. Pushing a
//! Fast migration on a successful promotion would just make
//! `apply_tier_migrations` copy already-correct DRAM bytes into a fresh
//! DRAM buffer for no reason -- a real, avoidable cost on every single
//! second-access promotion, which is exactly the class of cost this whole
//! variant exists to cut. The one case that still needs a real migration
//! -- a promotion out of the main queue's SLOW portion via
//! `give_second_chance` -- is untouched: that key's bytes genuinely are in
//! PMEM at that point (it was really demoted there earlier), so the
//! migration is still doing real, necessary work. The demotion-time
//! reprieve and `settle_fast_tier`'s real demotions are also untouched --
//! neither of those ever needed a Fast-migration push to begin with.

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

pub struct S3FifoGhostLazyDemotionFastAdmissionHybridStack {
	one_access_queue: QueueList,
	main_queue: QueueList,
	ghost: QueueList,

	entries: EntryMap,

	one_access_ratio: f64,
	one_access_capacity: CacheSize,
	one_access_used: CacheSize,

	/// The configured total fast-tier (DRAM) budget. Shared between the
	/// one-access queue and the main queue's fast segment -- see the
	/// module doc's "Accounting" section. The main queue's own trigger
	/// checks `effective_main_fast_capacity()`, not this field directly.
	fast_capacity: CacheSize,
	fast_used: CacheSize,
	slow_used: CacheSize,

	fast_count: usize,

	/// Number of keys currently in the `Main` queue (Fast or Slow). Also
	/// used as the ghost list's size cap reference.
	main_count: usize,

	main_boundary: Option<HashedKey>,

	migrations: Vec<(HashedKey, Tier)>,
}

impl S3FifoGhostLazyDemotionFastAdmissionHybridStack {
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

		S3FifoGhostLazyDemotionFastAdmissionHybridStack {
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
			migrations: Vec::new(),
		}
	}

	/// The budget actually available to the main queue's fast segment,
	/// after reserving room for the one-access queue -- see the module
	/// doc's "Accounting" section.
	fn effective_main_fast_capacity(&self) -> CacheSize {
		self.fast_capacity.saturating_sub(self.one_access_capacity)
	}

	pub fn tier_of(&self, key: HashedKey) -> Option<Tier> {
		let entry = self.entries.get(&key)?;

		match entry.queue {
			// The one-access queue is DRAM-resident in this variant --
			// see the module doc's "Motivation" section. This is the one
			// line that differs from `S3FifoGhostLazyDemotionHybridStack`'s
			// `tier_of`.
			Queue::OneAccess => Some(Tier::Fast),
			Queue::Main => entry.tier,
		}
	}

	/// Returns `true` if `key` currently has a ghost entry. Exposed for tests.
	pub fn is_ghost(&self, key: HashedKey) -> bool {
		self.ghost.contains(&key)
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

	/// Moves a re-accessed one-access-queue key into the main queue at
	/// `Tier::Fast`. Unlike the base design, this never needs to push a
	/// migration for the promotion itself -- the key's bytes are already
	/// physically Fast (see the module doc's "no more redundant Fast→Fast
	/// copies" section) -- only `settle_fast_tier`'s own demotion push (if
	/// this promotion itself immediately overflows the budget) can produce
	/// a migration here, and that's handled inside `settle_fast_tier`.
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

	/// Admits a brand-new key directly into `main_queue` at `Tier::Fast` —
	/// the ghost-hit path. Structurally identical to
	/// `promote_from_one_access` minus the "remove from `one_access_queue`"
	/// step. Same no-redundant-migration reasoning applies: the API layer
	/// already built this key's bytes as Fast (admission is unconditional
	/// Fast in this variant), so there's nothing to migrate unless
	/// `settle_fast_tier` demotes it right back out, which pushes its own
	/// migration.
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

	/// The eviction-time second chance — completely unchanged from
	/// `S3FifoGhostLazyDemotionHybridStack`. This is the one promotion path
	/// that STILL needs its migration push: a key reaching this method by
	/// definition currently has `tier == Some(Tier::Slow)` in the common
	/// case (it was genuinely demoted to PMEM earlier), so moving it back
	/// to Fast is a real physical move, not a relabeling.
	fn give_second_chance(&mut self, key: HashedKey) {
		let Some(entry) = self.entries.get(&key).copied() else { return };
		let size = entry.size as CacheSize;
		let was_fast = entry.tier == Some(Tier::Fast);
		let was_boundary = was_fast && self.main_boundary == Some(key);

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
		}

		if self.main_boundary.is_none() {
			self.main_boundary = Some(key);
		}

		self.settle_fast_tier();

		if self.entries.get(&key).and_then(|entry| entry.tier) == Some(Tier::Fast) {
			self.migrations.push((key, Tier::Fast));
		}
	}

	/// Demotes key(s) anchoring `main_boundary` until `fast_used` fits back
	/// within `effective_main_fast_capacity()` -- reference-bit gated,
	/// exactly like `S3FifoGhostLazyDemotionHybridStack`'s version. The
	/// only difference from that stack is which capacity this checks
	/// against (see the module doc's "Accounting" section).
	fn settle_fast_tier(&mut self) {
		let effective_capacity = self.effective_main_fast_capacity();

		while self.fast_used > effective_capacity {
			let Some(candidate) = self.main_boundary else { break };

			let accessed = self.entries.get(&candidate).map(|entry| entry.accessed).unwrap_or(false);

			if accessed {
				// Reprieve: fresh start at the front instead of demotion.
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
		}
	}

	/// Pops `one_access_queue`'s tail, removes it from this stack's own
	/// bookkeeping, and remembers it in `ghost`. Only called from
	/// `evict_one`.
	fn evict_one_access_tail(&mut self) -> Option<HashedKey> {
		let key = self.one_access_queue.pop_back()?;
		let size = self.entries.remove(&key).map(|entry| entry.size).unwrap_or(0) as CacheSize;

		self.one_access_used = self.one_access_used.saturating_sub(size);
		self.ghost.push_front(key);

		Some(key)
	}

	/// Trims `ghost` down to `main_count` entries — called only from a
	/// genuine main-queue eviction, never from a second chance, a
	/// demotion-time reprieve, or `evict_one_access_tail` (which is what
	/// populates `ghost`).
	fn trim_ghost(&mut self) {
		while self.ghost.len() > self.main_count {
			self.ghost.pop_back();
		}
	}
}

impl PolicyStack for S3FifoGhostLazyDemotionFastAdmissionHybridStack {
	fn is_policy(&self, policy: &PaperPolicy) -> bool {
		matches!(policy, PaperPolicy::S3FifoGhostLazyDemotionFastAdmissionHybrid(ratio) if *ratio == self.one_access_ratio)
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
		// Unconditional and first -- see `S3FifoGhostHybridStack::remove`'s
		// doc for why.
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
					},

					None => {},
				}
			},
		}
	}

	fn resize(&mut self, max_size: CacheSize) {
		self.one_access_capacity = (self.one_access_ratio * max_size as f64) as CacheSize;

		// Growing `one_access_capacity` shrinks the room left for the main
		// queue's fast segment (see the module doc's "Accounting"
		// section) -- proactively re-check rather than waiting for the
		// next unrelated insert/update to notice, same reasoning as
		// `resize_fast_tier` already has.
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
		self.migrations.clear();
	}

	fn evict_one(&mut self) -> Option<HashedKey> {
		if let Some(key) = self.evict_one_access_tail() {
			return Some(key);
		}

		loop {
			let key = *self.main_queue.back()?;
			let accessed = self.entries.get(&key).map(|entry| entry.accessed).unwrap_or(false);

			if accessed {
				self.give_second_chance(key);
				continue;
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
		// Total DRAM: main queue's fast segment + the one-access queue,
		// both physically Fast in this variant.
		self.fast_used + self.one_access_used
	}

	fn slow_bytes_used(&self) -> CacheSize {
		// The one-access queue no longer touches Slow/PMEM at all -- unlike
		// `S3FifoGhostLazyDemotionHybridStack`, this is just the main
		// queue's slow segment.
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

#[cfg(test)]
mod tests {
	use super::*;

	fn drain(stack: &mut S3FifoGhostLazyDemotionFastAdmissionHybridStack) -> Vec<(HashedKey, Tier)> {
		stack.drain_tier_migrations()
	}

	#[test]
	fn admission_always_lands_in_one_access_queue_fast() {
		let mut stack = S3FifoGhostLazyDemotionFastAdmissionHybridStack::new(1.0, 1_000, 1_000);

		stack.insert(1, 10);
		stack.insert(2, 10);

		assert_eq!(stack.tier_of(1), Some(Tier::Fast));
		assert_eq!(stack.tier_of(2), Some(Tier::Fast));
		assert_eq!(drain(&mut stack), Vec::new());
	}

	#[test]
	fn reaccessing_a_one_access_key_promotes_it_to_main_without_a_migration() {
		// one_access_ratio=0.0 -- unlike the base design's equivalent test,
		// ratio=1.0 here would reserve the *entire* fast_capacity for the
		// one-access queue (see `zero_effective_main_capacity_...` below),
		// leaving zero room for the promoted key to actually stay fast.
		let mut stack = S3FifoGhostLazyDemotionFastAdmissionHybridStack::new(0.0, 1_000, 1_000);

		stack.insert(1, 10);
		drain(&mut stack);

		stack.update(1);

		// No migration: the key's bytes were already Fast the whole time
		// (see the module doc's "no more redundant Fast→Fast copies"
		// section) -- this is the key behavioral difference from
		// `S3FifoGhostLazyDemotionHybridStack`, whose equivalent test
		// asserts the opposite (a real migration IS produced there).
		assert_eq!(drain(&mut stack), Vec::new());
		assert_eq!(stack.tier_of(1), Some(Tier::Fast));
	}

	#[test]
	fn a_key_aging_out_without_reaccess_becomes_a_ghost_entry() {
		let mut stack = S3FifoGhostLazyDemotionFastAdmissionHybridStack::new(1.0, 1_000, 1_000);

		stack.insert(1, 10);
		drain(&mut stack);

		assert_eq!(stack.evict_one(), Some(1));
		assert!(stack.is_ghost(1));
	}

	#[test]
	fn ghost_hit_on_readmission_lands_in_fast_tier_without_a_migration() {
		// one_access_ratio=0.0 -- see the comment on
		// `reaccessing_a_one_access_key_promotes_it_to_main_without_a_migration`
		// above for why ratio=1.0 wouldn't leave room for the promoted key
		// to stay fast here.
		let mut stack = S3FifoGhostLazyDemotionFastAdmissionHybridStack::new(0.0, 1_000, 1_000);

		stack.insert(1, 10);
		drain(&mut stack);
		stack.evict_one();
		assert!(stack.is_ghost(1));

		stack.insert(1, 10);

		// No migration here either, same reasoning as the reaccess test
		// above -- the API layer already built this key's bytes as Fast.
		assert_eq!(drain(&mut stack), Vec::new());
		assert_eq!(stack.tier_of(1), Some(Tier::Fast));
	}

	// ── the signature accounting mechanic: shared DRAM budget ──────────────

	#[test]
	fn one_access_capacity_is_reserved_out_of_the_fast_budget() {
		// fast_capacity = 100, one_access_ratio reserves 40 of it
		// (0.04 * 1_000 = 40) for the one-access queue, leaving only 60
		// for the main queue's fast segment.
		let mut stack = S3FifoGhostLazyDemotionFastAdmissionHybridStack::new(0.04, 1_000, 100);
		assert_eq!(stack.effective_main_fast_capacity(), 60);

		// Promote two 50-byte keys into the main queue -- fast_used (main
		// only) would reach 100, comfortably within raw fast_capacity=100,
		// but that's over the *effective* 60-byte budget once the
		// one-access reservation is accounted for, so the older one must
		// be demoted.
		stack.insert(1, 50);
		stack.update(1);
		drain(&mut stack);

		stack.insert(2, 50);
		stack.update(2);
		let migrations = drain(&mut stack);

		assert!(migrations.iter().any(|(k, t)| *k == 1 && *t == Tier::Slow));
		assert_eq!(stack.tier_of(1), Some(Tier::Slow));
		assert_eq!(stack.tier_of(2), Some(Tier::Fast));
	}

	#[test]
	fn zero_effective_main_capacity_demotes_every_promotion_immediately() {
		// one_access_ratio alone consumes the entire fast_capacity, so the
		// main queue's fast segment has zero effective room -- every
		// promotion must self-demote right back to slow. Degenerate but
		// correct: documented in the module doc, mirrors
		// lru_sized_hybrid_cache's equivalent zero-capacity precedent.
		let mut stack = S3FifoGhostLazyDemotionFastAdmissionHybridStack::new(1.0, 1_000, 1_000);
		assert_eq!(stack.effective_main_fast_capacity(), 0);

		stack.insert(1, 10);
		stack.update(1);
		let migrations = drain(&mut stack);

		assert_eq!(migrations, vec![(1, Tier::Slow)]);
		assert_eq!(stack.tier_of(1), Some(Tier::Slow));
	}

	#[test]
	fn growing_one_access_capacity_via_resize_immediately_settles_main_fast() {
		// Start with room to spare: fast_capacity=100, one_access_ratio
		// reserves only 10 (0.01 * 1_000), leaving 90 for main-fast.
		let mut stack = S3FifoGhostLazyDemotionFastAdmissionHybridStack::new(0.01, 1_000, 100);

		stack.insert(1, 50);
		stack.update(1);
		drain(&mut stack);
		assert_eq!(stack.tier_of(1), Some(Tier::Fast));

		// Growing max_size to 10_000 grows the one-access reservation to
		// 100 (0.01 * 10_000), consuming the *entire* fast_capacity and
		// leaving 0 for main-fast -- resize() must catch this immediately
		// rather than waiting for an unrelated insert/update.
		stack.resize(10_000);
		let migrations = drain(&mut stack);

		assert_eq!(migrations, vec![(1, Tier::Slow)]);
		assert_eq!(stack.tier_of(1), Some(Tier::Slow));
	}

	// ── the inherited signature mechanic: reprieve at DEMOTION time ────────

	#[test]
	fn an_accessed_fast_boundary_key_is_reprieved_at_demotion_time_not_demoted() {
		// effective_main_fast_capacity = 10 - 0 (one_access_ratio=0) fits
		// exactly one 10-byte object at a time.
		let mut stack = S3FifoGhostLazyDemotionFastAdmissionHybridStack::new(0.0, 1_000, 10);

		stack.insert(1, 10);
		stack.update(1);
		drain(&mut stack);

		stack.update(1);
		assert_eq!(drain(&mut stack), Vec::new());

		stack.insert(2, 10);
		stack.update(2);
		let migrations = drain(&mut stack);

		assert_eq!(stack.tier_of(1), Some(Tier::Fast), "key 1 should have been reprieved, not demoted");
		assert_eq!(stack.tier_of(2), Some(Tier::Slow), "key 2 should have been demoted in key 1's place");
		assert_eq!(migrations, vec![(2, Tier::Slow)]);
	}

	#[test]
	fn evict_one_gives_an_accessed_slow_key_a_second_chance() {
		let mut stack = S3FifoGhostLazyDemotionFastAdmissionHybridStack::new(0.0, 1_000, 10);

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
		let mut stack = S3FifoGhostLazyDemotionFastAdmissionHybridStack::new(1.0, 1_000, 1_000);

		stack.insert(1, 10);
		drain(&mut stack);
		stack.evict_one();
		assert!(stack.is_ghost(1));

		stack.remove(1);
		assert!(!stack.is_ghost(1));
	}

	#[test]
	fn remove_and_clear_reset_bookkeeping() {
		let mut stack = S3FifoGhostLazyDemotionFastAdmissionHybridStack::new(1.0, 1_000, 1_000);

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
	}

	#[test]
	fn fast_and_slow_gauges_include_one_access_queue_on_the_fast_side() {
		let mut stack = S3FifoGhostLazyDemotionFastAdmissionHybridStack::new(1.0, 1_000, 1_000);

		stack.insert(1, 10);
		stack.insert(2, 10);
		drain(&mut stack);

		assert_eq!(stack.fast_bytes_used(), 20);
		assert_eq!(stack.slow_bytes_used(), 0);
		assert_eq!(stack.fast_object_count(), 2);
		assert_eq!(stack.slow_object_count(), 0);
	}
}
