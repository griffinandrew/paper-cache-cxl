/*
 * Copyright (c) Kia Shakiba
 *
 * This source code is licensed under the GNU AGPLv3 license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! `TwoQGhostHybridStack` — `TwoQHybridStack` plus a bare-key ghost queue,
//! for `PaperPolicy::TwoQGhostHybrid`.
//!
//! Identical to `TwoQHybridStack` in every other respect (see that stack's
//! module doc for the full admission/demotion/promotion/eviction rules) —
//! this file only adds what `TwoQHybridStack`'s own module doc flagged as
//! deliberately left out: a ghost queue remembering keys that aged out of
//! `fifo_queue` without a second access, so a later re-admission can be
//! trusted immediately instead of restarting from zero.
//!
//! Mirrors `s_three_fifo_stack.rs`'s existing `ghost: HashList<HashedKey>`
//! shape exactly — a bare key list, no object data, chosen over plain
//! `TwoQStack`'s heavier `a1_out` (which holds real live objects) per an
//! explicit user decision: a lightweight membership list, not a third place
//! actual bytes can live.
//!
//! ## Ghost lifecycle, matching `SThreeFifoStack`'s existing convention
//!
//! * **Added to** only by `evict_fifo_tail` (a `fifo_queue` object aging out
//!   without a second access) — never by a main-queue eviction.
//! * **Checked** by `insert`'s brand-new-key branch, before falling back to
//!   the normal `fifo_queue` admission.
//! * **Not removed immediately on a hit** — same lazy convention
//!   `SThreeFifoStack` already uses (see that file's
//!   `no_ghost_entry_routes_fresh_insertion_to_small_queue` test doc for the
//!   precedent). Only trimmed lazily, capped relative to `main_count`,
//!   during a genuine main-queue eviction (never during a `fifo_queue`
//!   eviction, which is what populates it) — and cleared outright by
//!   `remove`/`clear`.
//!
//! ## Where a ghost hit lands: fast tier, deliberately reversible
//!
//! A ghost hit is admitted directly into `main_stack` at `Tier::Fast` —
//! `admit_via_ghost_hit`, structurally identical to `promote_from_fifo`
//! minus the "remove from `fifo_queue`" step (the key was never there this
//! time). This was an explicit, acknowledged-as-arguable choice: the
//! alternative (land in the *slow* portion of `main_stack`, still having to
//! earn fast-tier promotion via a subsequent real access, the more
//! conservative reading) was flagged as possibly better and left as a
//! one-line change here (swap `Tier::Fast` for `Tier::Slow` and drop the
//! `settle_fast_tier`/fast-tier bookkeeping in `admit_via_ghost_hit`) if
//! real measurement says otherwise. Physically cheap either way: the
//! API-layer `set()` always builds a brand-new key as `TieredBuffer::
//! new_slow` regardless of ghost history (this stack has no equivalent of
//! `LfuHybridStack`'s admission-latch mirror onto `AtomicStatus` — a ghost
//! hit is corrected to `Fast` via the ordinary async migration path, the
//! same one every other promotion in this stack already uses), so a ghost
//! hit costs exactly one extra migration, not a synchronous PMEM-vs-DRAM
//! choice at the API layer.
//!
//! ## `eviction_stacks_pmem`
//!
//! `ghost` follows the same DRAM/PMEM switch as `fifo_queue`/`main_stack`/
//! `entries` — see `TwoQHybridStack`'s module doc.

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
	Fifo,
	Main,
}

/// Combined per-key bookkeeping — see `TwoQHybridStack`'s "One combined
/// per-key map" module doc section.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct TwoQEntry {
	queue: Queue,
	tier: Option<Tier>,
	size: ObjectSize,
}

#[cfg(not(feature = "eviction_stacks_pmem"))]
type QueueList = HashList<HashedKey, NoHasher>;
#[cfg(feature = "eviction_stacks_pmem")]
type QueueList = PmemHashList<HashedKey, NoHasher>;

#[cfg(not(feature = "eviction_stacks_pmem"))]
type EntryMap = HashMap<HashedKey, TwoQEntry, NoHasher>;
#[cfg(feature = "eviction_stacks_pmem")]
type EntryMap = HashMap<HashedKey, TwoQEntry, NoHasher, Hybrid>;

pub struct TwoQGhostHybridStack {
	fifo_queue: QueueList,
	main_stack: QueueList,
	ghost: QueueList,

	entries: EntryMap,

	k_in: f64,
	fifo_capacity: CacheSize,
	fifo_used: CacheSize,

	fast_capacity: CacheSize,
	fast_used: CacheSize,
	slow_used: CacheSize,

	/// Number of keys currently tagged `Tier::Fast` within `main_stack`.
	fast_count: usize,

	/// Number of keys currently in the `Main` queue (Fast or Slow). Also
	/// used as the ghost list's size cap reference (mirrors
	/// `SThreeFifoStack`'s `ghost.len() > main.stack.len()` bound).
	main_count: usize,

	/// The least-recently-used key currently tagged `Tier::Fast` within
	/// `main_stack` — the next demotion candidate. Mirrors
	/// `LruHybridStack::fast_boundary`.
	main_boundary: Option<HashedKey>,

	/// (key, new tier) pairs recorded since the last `drain_tier_migrations`.
	migrations: Vec<(HashedKey, Tier)>,
}

impl TwoQGhostHybridStack {
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

	pub fn new(k_in: f64, max_size: CacheSize, fast_capacity: CacheSize) -> Self {
		let (fifo_queue, main_stack, ghost, entries) = Self::new_collections();

		TwoQGhostHybridStack {
			fifo_queue,
			main_stack,
			ghost,

			entries,

			k_in,
			fifo_capacity: (k_in * max_size as f64) as CacheSize,
			fifo_used: 0,

			fast_capacity,
			fast_used: 0,
			slow_used: 0,
			fast_count: 0,
			main_count: 0,

			main_boundary: None,
			migrations: Vec::new(),
		}
	}

	/// Returns which queue/tier the given (currently tracked) key is in, or
	/// `None` if the key isn't tracked. Exposed for tests/diagnostics.
	pub fn tier_of(&self, key: HashedKey) -> Option<Tier> {
		let entry = self.entries.get(&key)?;

		match entry.queue {
			Queue::Fifo => Some(Tier::Slow),
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
		match self.entries.get(&key).map(|entry| entry.queue) {
			Some(Queue::Fifo) => self.promote_from_fifo(key),
			Some(Queue::Main) => self.touch_main_fast(key),
			None => {},
		}
	}

	fn promote_from_fifo(&mut self, key: HashedKey) {
		let Some(entry) = self.entries.get(&key) else { return };
		let size = entry.size;
		let size_bytes = size as CacheSize;

		self.fifo_queue.remove(&key);
		self.fifo_used = self.fifo_used.saturating_sub(size_bytes);

		self.main_stack.push_front(key);
		self.entries.insert(key, TwoQEntry { queue: Queue::Main, tier: Some(Tier::Fast), size });
		self.fast_used += size_bytes;
		self.fast_count += 1;
		self.main_count += 1;

		if self.main_boundary.is_none() {
			self.main_boundary = Some(key);
		}

		self.settle_fast_tier();

		if self.entries.get(&key).and_then(|entry| entry.tier) == Some(Tier::Fast) {
			self.migrations.push((key, Tier::Fast));
		}
	}

	/// Admits a brand-new key directly into `main_stack` at `Tier::Fast` —
	/// the ghost-hit path. Structurally identical to `promote_from_fifo`
	/// minus the "remove from `fifo_queue`" step, since the key was never
	/// there this time. See the module doc's "Where a ghost hit lands"
	/// section for why `Tier::Fast` specifically, and how to flip it.
	fn admit_via_ghost_hit(&mut self, key: HashedKey, size: ObjectSize) {
		self.main_stack.push_front(key);
		self.entries.insert(key, TwoQEntry { queue: Queue::Main, tier: Some(Tier::Fast), size });
		self.fast_used += size as CacheSize;
		self.fast_count += 1;
		self.main_count += 1;

		if self.main_boundary.is_none() {
			self.main_boundary = Some(key);
		}

		self.settle_fast_tier();

		if self.entries.get(&key).and_then(|entry| entry.tier) == Some(Tier::Fast) {
			self.migrations.push((key, Tier::Fast));
		}
	}

	fn touch_main_fast(&mut self, key: HashedKey) {
		let previous_tier = self.entries.get(&key).and_then(|entry| entry.tier);

		let already_at_front = self.main_stack.front() == Some(&key);
		let is_boundary = self.main_boundary == Some(key);

		let new_boundary_if_moved = if is_boundary && !already_at_front {
			self.main_stack.before(&key).copied()
		} else {
			None
		};

		self.main_stack.move_front(&key);

		if is_boundary && !already_at_front {
			self.main_boundary = new_boundary_if_moved;
		}

		let mut promoted = false;

		if previous_tier != Some(Tier::Fast) {
			if previous_tier == Some(Tier::Slow) {
				let size = self.entries.get(&key).map(|entry| entry.size).unwrap_or(0) as CacheSize;

				self.slow_used = self.slow_used.saturating_sub(size);
				self.fast_used += size;
				self.fast_count += 1;

				promoted = true;
			}

			if let Some(entry) = self.entries.get_mut(&key) {
				entry.tier = Some(Tier::Fast);
			}

			if self.main_boundary.is_none() {
				self.main_boundary = Some(key);
			}
		}

		self.settle_fast_tier();

		if promoted && self.entries.get(&key).and_then(|entry| entry.tier) == Some(Tier::Fast) {
			self.migrations.push((key, Tier::Fast));
		}
	}

	fn settle_fast_tier(&mut self) {
		while self.fast_used > self.fast_capacity {
			let Some(demote_key) = self.main_boundary else { break };

			let size = self.entries.get(&demote_key).map(|entry| entry.size).unwrap_or(0) as CacheSize;
			let new_boundary = self.main_stack.before(&demote_key).copied();

			if let Some(entry) = self.entries.get_mut(&demote_key) {
				entry.tier = Some(Tier::Slow);
			}

			self.fast_used = self.fast_used.saturating_sub(size);
			self.fast_count = self.fast_count.saturating_sub(1);
			self.slow_used += size;
			self.main_boundary = new_boundary;

			self.migrations.push((demote_key, Tier::Slow));
		}
	}

	/// Pops `fifo_queue`'s tail, removes it from this stack's own
	/// bookkeeping, and remembers it in `ghost` — the "aged out without a
	/// second access" case. Same "cannot self-evict from insert/resize"
	/// rationale as `TwoQHybridStack::evict_fifo_tail` — only called from
	/// `evict_one`.
	fn evict_fifo_tail(&mut self) -> Option<HashedKey> {
		let key = self.fifo_queue.pop_back()?;
		let size = self.entries.remove(&key).map(|entry| entry.size).unwrap_or(0) as CacheSize;

		self.fifo_used = self.fifo_used.saturating_sub(size);
		self.ghost.push_front(key);

		Some(key)
	}

	/// Trims `ghost` down to `main_count` entries, oldest first — called
	/// only from a genuine main-queue eviction (never from
	/// `evict_fifo_tail`, which is what populates `ghost` in the first
	/// place). Mirrors `SThreeFifoStack::evict_main`'s `while self.ghost.
	/// len() > self.main.stack.len()` cap exactly, using `main_count` as
	/// the size reference this stack already tracks.
	fn trim_ghost(&mut self) {
		while self.ghost.len() > self.main_count {
			self.ghost.pop_back();
		}
	}
}

impl PolicyStack for TwoQGhostHybridStack {
	fn is_policy(&self, policy: &PaperPolicy) -> bool {
		matches!(policy, PaperPolicy::TwoQGhostHybrid(k_in) if *k_in == self.k_in)
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

		self.fifo_queue.push_front(key);
		self.entries.insert(key, TwoQEntry { queue: Queue::Fifo, tier: None, size });
		self.fifo_used += size as CacheSize;
	}

	fn update(&mut self, key: HashedKey) {
		if self.entries.contains_key(&key) {
			self.touch(key);
		}
	}

	fn remove(&mut self, key: HashedKey) {
		// Unconditional and first: a key evicted from `fifo_queue` (via
		// `evict_fifo_tail`) has *already* been removed from `entries` by
		// the time it lives only in `ghost` -- gating this on
		// `entries.remove` succeeding (as the rest of this method's logic
		// legitimately does) would silently skip clearing a stale ghost
		// entry for exactly that case. Mirrors `SThreeFifoStack::remove`,
		// which also clears its ghost queue unconditionally.
		self.ghost.remove(&key);

		let Some(entry) = self.entries.remove(&key) else { return };
		let size = entry.size as CacheSize;

		match entry.queue {
			Queue::Fifo => {
				self.fifo_queue.remove(&key);
				self.fifo_used = self.fifo_used.saturating_sub(size);
			},

			Queue::Main => {
				let new_boundary_if_needed = if entry.tier == Some(Tier::Fast) && self.main_boundary == Some(key) {
					self.main_stack.before(&key).copied()
				} else {
					None
				};

				self.main_stack.remove(&key);
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
		self.fifo_capacity = (self.k_in * max_size as f64) as CacheSize;
	}

	fn clear(&mut self) {
		self.fifo_queue.clear();
		self.main_stack.clear();
		self.ghost.clear();
		self.entries.clear();

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

		let key = self.main_stack.pop_back()?;
		let removed = self.entries.remove(&key);
		let size = removed.map(|entry| entry.size).unwrap_or(0) as CacheSize;
		let tier = removed.and_then(|entry| entry.tier);

		self.main_count = self.main_count.saturating_sub(1);

		match tier {
			Some(Tier::Fast) => {
				self.fast_used = self.fast_used.saturating_sub(size);
				self.fast_count = self.fast_count.saturating_sub(1);

				if self.main_boundary == Some(key) {
					self.main_boundary = self.main_stack.back().copied();
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
		self.fifo_queue.len() + (self.main_count - self.fast_count)
	}

	fn needs_capacity_eviction(&self) -> bool {
		self.fifo_used > self.fifo_capacity
	}
}

#[cfg(test)]
mod tests {
	use super::*;

	fn drain(stack: &mut TwoQGhostHybridStack) -> Vec<(HashedKey, Tier)> {
		stack.drain_tier_migrations()
	}

	#[test]
	fn admission_always_lands_in_fifo_queue_slow() {
		let mut stack = TwoQGhostHybridStack::new(1.0, 1_000, 1_000);

		stack.insert(1, 10);
		stack.insert(2, 10);

		assert_eq!(stack.tier_of(1), Some(Tier::Slow));
		assert_eq!(stack.tier_of(2), Some(Tier::Slow));
		assert_eq!(drain(&mut stack), Vec::new());
	}

	#[test]
	fn reaccessing_a_fifo_key_promotes_it_to_fast() {
		let mut stack = TwoQGhostHybridStack::new(1.0, 1_000, 1_000);

		stack.insert(1, 10);
		drain(&mut stack);

		stack.update(1);
		let migrations = drain(&mut stack);

		assert_eq!(migrations, vec![(1, Tier::Fast)]);
		assert_eq!(stack.tier_of(1), Some(Tier::Fast));
	}

	#[test]
	fn a_key_aging_out_of_fifo_without_reaccess_becomes_a_ghost_entry() {
		let mut stack = TwoQGhostHybridStack::new(1.0, 1_000, 1_000);

		stack.insert(1, 10);
		drain(&mut stack);

		assert_eq!(stack.evict_one(), Some(1));
		assert_eq!(stack.contains(1), false);
		assert!(stack.is_ghost(1), "evicted fifo key should leave a ghost entry");
	}

	#[test]
	fn ghost_hit_on_readmission_lands_directly_in_fast_tier() {
		let mut stack = TwoQGhostHybridStack::new(1.0, 1_000, 1_000);

		stack.insert(1, 10);
		drain(&mut stack);
		stack.evict_one(); // key 1 ages out -> ghost
		assert!(stack.is_ghost(1));

		// Re-admission: ghost hit -> straight to Main/Fast, no fifo_queue stop.
		stack.insert(1, 10);
		let migrations = drain(&mut stack);

		assert_eq!(migrations, vec![(1, Tier::Fast)]);
		assert_eq!(stack.tier_of(1), Some(Tier::Fast));
	}

	#[test]
	fn fresh_key_with_no_ghost_history_still_lands_in_fifo_queue_slow() {
		let mut stack = TwoQGhostHybridStack::new(1.0, 1_000, 1_000);

		// No prior history for key 5 at all.
		stack.insert(5, 10);
		let migrations = drain(&mut stack);

		assert_eq!(migrations, Vec::new());
		assert_eq!(stack.tier_of(5), Some(Tier::Slow));
	}

	#[test]
	fn fifo_capacity_pressure_is_reported_not_self_evicted() {
		let mut stack = TwoQGhostHybridStack::new(1.0, 15, 1_000);

		stack.insert(1, 10);
		drain(&mut stack);
		assert_eq!(stack.needs_capacity_eviction(), false);

		stack.insert(2, 10);
		drain(&mut stack);
		assert_eq!(stack.needs_capacity_eviction(), true);

		assert_eq!(stack.evict_one(), Some(1));
		assert_eq!(stack.needs_capacity_eviction(), false);
	}

	#[test]
	fn fast_tier_pressure_within_main_queue_demotes_lru_tail() {
		let mut stack = TwoQGhostHybridStack::new(1.0, 1_000, 25);

		stack.insert(1, 10);
		stack.insert(2, 10);
		stack.update(1);
		stack.update(2);
		drain(&mut stack);

		assert_eq!(stack.fast_bytes_used(), 20);

		stack.insert(3, 10);
		stack.update(3);
		let migrations = drain(&mut stack);

		assert!(migrations.iter().any(|(k, t)| *k == 1 && *t == Tier::Slow));
		assert_eq!(stack.tier_of(1), Some(Tier::Slow));
		assert_eq!(stack.tier_of(2), Some(Tier::Fast));
		assert_eq!(stack.tier_of(3), Some(Tier::Fast));
	}

	#[test]
	fn evict_one_prefers_fifo_queue_over_main_queue() {
		let mut stack = TwoQGhostHybridStack::new(1.0, 1_000, 1_000);

		stack.insert(1, 10);
		stack.insert(2, 10);
		stack.update(2);
		drain(&mut stack);

		assert_eq!(stack.evict_one(), Some(1));
		assert_eq!(stack.tier_of(2), Some(Tier::Fast));
	}

	#[test]
	fn remove_clears_ghost_entry_too() {
		let mut stack = TwoQGhostHybridStack::new(1.0, 1_000, 1_000);

		stack.insert(1, 10);
		drain(&mut stack);
		stack.evict_one();
		assert!(stack.is_ghost(1));

		stack.remove(1);
		assert!(!stack.is_ghost(1), "remove() should clear the ghost entry too");
	}

	#[test]
	fn remove_and_clear_reset_bookkeeping() {
		let mut stack = TwoQGhostHybridStack::new(1.0, 1_000, 1_000);

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
}
