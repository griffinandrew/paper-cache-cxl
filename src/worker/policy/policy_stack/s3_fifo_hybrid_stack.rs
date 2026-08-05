/*
 * Copyright (c) Kia Shakiba
 *
 * This source code is licensed under the GNU AGPLv3 license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! `S3FifoHybridStack` — a segmented S3-FIFO stack for `PaperPolicy::S3FifoHybrid`.
//!
//! Structurally very close to `TwoQHybridStack` — a one-access FIFO queue
//! (`one_access_queue`, real objects, always entirely in the slow tier) that
//! feeds a main FIFO queue (`main_queue`, segmented fast/slow by a
//! `main_boundary` cursor, exactly like `TwoQHybridStack::main_stack`) — but
//! the *mechanism* that decides who gets to stay is different:
//!
//! * `one_access_queue` behaves exactly like `TwoQHybridStack::fifo_queue`:
//!   a re-access promotes a key **immediately, eagerly** to the front of
//!   `main_queue` at `Tier::Fast`. Anything still sitting in
//!   `one_access_queue` when it reaches the tail has therefore, by
//!   construction, never been re-accessed — so eviction from this queue is
//!   always unconditional, no bit to check.
//! * `main_queue` is **pure FIFO — never reordered on access**. Instead,
//!   every tracked `Main` key carries an `accessed: bool` "reference bit"
//!   (classic CLOCK/second-chance), set by `mark_accessed` on every touch
//!   (`get()` hit or re-`set()`), regardless of which portion (fast or
//!   slow) the key currently occupies. The bit is only ever *consulted*
//!   lazily, at the moment a key reaches the tail of `main_queue` and is
//!   about to be evicted: if set, the key is given a second chance
//!   (`give_second_chance` — reinserted at the front, retagged `Fast`, bit
//!   cleared) instead of being evicted; if clear, it's evicted for real.
//!   Demotion (fast-tier space needed) is unconditional aging of whichever
//!   key currently anchors `main_boundary` — the reference bit plays no
//!   part in demotion, only in the eviction-time second-chance check.
//!
//! This asymmetry (eager promotion for the one-access queue, lazy
//! bit-checked promotion for the main queue) is a direct reading of the
//! paper text: "if accessed again *before reaching the top* of the
//! one-access FIFO queue, they are promoted" (eager) vs. "if they *have
//! been re-accessed during this period*, they are reinserted" only
//! evaluated "after objects have traversed through both portions... and
//! are about to be evicted" (lazy, at the point of eviction).
//!
//! ## The "contiguous front run" invariant
//!
//! `main_queue` is never reordered except by two operations, both of which
//! preserve "the front `main_boundary`-worth of keys are exactly the
//! `Tier::Fast` ones, in some order; everything behind them is `Tier::Slow`":
//! insertion always happens at the front (`promote_from_one_access`,
//! `give_second_chance`) and demotion never moves anything in the list at
//! all — it only re-tags whichever key `main_boundary` currently points at
//! and walks the cursor one step toward the front (`settle_fast_tier`,
//! copied verbatim from `TwoQHybridStack`'s). A key given a second chance
//! effectively re-enters the queue as if newly promoted (moved to the
//! front) — this deliberately scrambles true insertion age in exchange for
//! matching the paper's own wording ("reinserted into the fast tier
//! portion"), exactly how a real CLOCK sweep works.
//!
//! ## No ghost queue
//!
//! Same reasoning and precedent as `TwoQHybridStack` (confirmed with the
//! user before implementing, given the source material describes exactly
//! two live queues and no ghost/ghost-hit mechanism at all): an
//! exact-membership check on every admission was judged an unwelcome added
//! cost given admission here already pays a synchronous slow-tier/PMEM
//! write. A `one_access_queue` object that ages out without a second access
//! is evicted outright, no trace kept.
//!
//! ## One combined per-key map
//!
//! Same "One combined per-key map" rationale as `TwoQHybridStack` — see
//! that stack's module doc for the full history of why this replaced
//! separate per-key maps. `S3FifoEntry` adds one field beyond
//! `TwoQEntry`: `accessed: bool`.
//!
//! ## `eviction_stacks_pmem`
//!
//! Same DRAM/PMEM backing switch as `TwoQHybridStack` — see that stack's
//! module doc.

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

/// Combined per-key bookkeeping. `tier`/`accessed` are only meaningful while
/// `queue == Main` (`tier: None`, `accessed: false` for `OneAccess` keys —
/// see the module doc for why `OneAccess` never needs a reference bit at
/// all: its promotion is eager, not lazy).
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

pub struct S3FifoHybridStack {
	one_access_queue: QueueList,
	main_queue: QueueList,

	entries: EntryMap,

	one_access_ratio: f64,
	one_access_capacity: CacheSize,
	one_access_used: CacheSize,

	fast_capacity: CacheSize,
	fast_used: CacheSize,
	slow_used: CacheSize,

	/// Number of keys currently tagged `Tier::Fast` within `main_queue`.
	/// Mirrors `TwoQHybridStack::fast_count`.
	fast_count: usize,

	/// Number of keys currently in the `Main` queue (Fast or Slow). Mirrors
	/// `TwoQHybridStack::main_count`.
	main_count: usize,

	/// The oldest key currently tagged `Tier::Fast` within `main_queue` —
	/// i.e. the next demotion candidate. `None` iff no key in `main_queue`
	/// is currently Fast. Mirrors `TwoQHybridStack::main_boundary` exactly
	/// (see the module doc's "contiguous front run" section for why this
	/// invariant holds here too, despite `main_queue` never being
	/// LRU-reordered).
	main_boundary: Option<HashedKey>,

	/// (key, new tier) pairs recorded since the last `drain_tier_migrations`.
	migrations: Vec<(HashedKey, Tier)>,
}

impl S3FifoHybridStack {
	#[cfg(not(feature = "eviction_stacks_pmem"))]
	fn new_collections() -> (QueueList, QueueList, EntryMap) {
		(HashList::default(), HashList::default(), HashMap::default())
	}

	#[cfg(feature = "eviction_stacks_pmem")]
	fn new_collections() -> (QueueList, QueueList, EntryMap) {
		(
			PmemHashList::with_hasher(NoHasher::default()),
			PmemHashList::with_hasher(NoHasher::default()),
			HashMap::with_hasher_in(NoHasher::default(), Hybrid),
		)
	}

	pub fn new(one_access_ratio: f64, max_size: CacheSize, fast_capacity: CacheSize) -> Self {
		let (one_access_queue, main_queue, entries) = Self::new_collections();

		S3FifoHybridStack {
			one_access_queue,
			main_queue,

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

	/// Returns which queue/tier the given (currently tracked) key is in, or
	/// `None` if the key isn't tracked. Exposed for tests/diagnostics.
	pub fn tier_of(&self, key: HashedKey) -> Option<Tier> {
		let entry = self.entries.get(&key)?;

		match entry.queue {
			Queue::OneAccess => Some(Tier::Slow),
			Queue::Main => entry.tier,
		}
	}

	/// Records a size change for an already-tracked key without altering its
	/// queue/tier/accessed bit, adjusting whichever counter currently
	/// applies. Mirrors `TwoQHybridStack::resize_key`.
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

	/// Treats an already-tracked key as accessed: a `OneAccess` key promotes
	/// straight and eagerly to `Main`+`Fast`; a `Main` key just has its
	/// reference bit set (`mark_accessed`) — no reordering, no migration,
	/// no tier change, until it's evaluated at eviction time.
	fn touch(&mut self, key: HashedKey) {
		match self.entries.get(&key).map(|entry| entry.queue) {
			Some(Queue::OneAccess) => self.promote_from_one_access(key),
			Some(Queue::Main) => self.mark_accessed(key),
			None => {},
		}
	}

	/// Sets a `Main`-tracked key's reference bit. The lazy half of the
	/// promotion mechanism — see the module doc.
	fn mark_accessed(&mut self, key: HashedKey) {
		if let Some(entry) = self.entries.get_mut(&key) {
			entry.accessed = true;
		}
	}

	/// Moves a `one_access_queue`-resident key to the front of `main_queue`,
	/// tagging it `Tier::Fast` with a clear reference bit. A brand-new entry
	/// into `main_queue`, so no boundary-shift bookkeeping is needed beyond
	/// setting `main_boundary` if this is the first Fast key. Mirrors
	/// `TwoQHybridStack::promote_from_fifo`.
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

		// Pushed after `settle_fast_tier` and guarded on the key still
		// being `Fast` — see `TwoQHybridStack::promote_from_fifo`'s doc for
		// why (demotions-before-promotions ordering; an extremely tight
		// fast-tier budget can self-evict this same key within the same
		// call).
		if self.entries.get(&key).and_then(|entry| entry.tier) == Some(Tier::Fast) {
			self.migrations.push((key, Tier::Fast));
		}
	}

	/// Gives a `main_queue` key currently sitting at the tail (about to be
	/// evicted) a second chance: moves it to the front, tags it
	/// `Tier::Fast`, clears its reference bit. Handles both the normal case
	/// (the key was `Slow`) and the fallback case (the key is still `Fast`
	/// because nothing has ever been demoted — see `evict_one`'s doc) via
	/// the same `before`-then-move ordering `settle_fast_tier` uses when the
	/// key being moved currently anchors `main_boundary` itself.
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

	/// Demotes the key(s) anchoring `main_boundary` — unconditional aging,
	/// no reference-bit check (the bit only ever matters at eviction time,
	/// not demotion time — see the module doc) — until `fast_used` fits
	/// back within `fast_capacity`. No low-water floor: fast-tier pressure
	/// here is only ever triggered by a promotion/second-chance or an
	/// explicit `resize_fast_tier`, never by every `set()` (admission never
	/// touches the fast tier directly). Copied verbatim from
	/// `TwoQHybridStack::settle_fast_tier` — demotion never moves anything
	/// in the list, only re-tags whichever key the cursor points at.
	fn settle_fast_tier(&mut self) {
		while self.fast_used > self.fast_capacity {
			let Some(demote_key) = self.main_boundary else { break };

			let size = self.entries.get(&demote_key).map(|entry| entry.size).unwrap_or(0) as CacheSize;
			let new_boundary = self.main_queue.before(&demote_key).copied();

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

	/// Pops and fully removes `one_access_queue`'s tail from this stack's
	/// own bookkeeping, if any — unconditional, since anything still here
	/// has (by construction of the eager-promotion rule) never been
	/// re-accessed. Same "cannot self-evict from insert/resize" rationale
	/// as `TwoQHybridStack::evict_fifo_tail` — only called from
	/// `evict_one`, never from `insert`/`resize`.
	fn evict_one_access_tail(&mut self) -> Option<HashedKey> {
		let key = self.one_access_queue.pop_back()?;
		let size = self.entries.remove(&key).map(|entry| entry.size).unwrap_or(0) as CacheSize;

		self.one_access_used = self.one_access_used.saturating_sub(size);

		Some(key)
	}
}

impl PolicyStack for S3FifoHybridStack {
	fn is_policy(&self, policy: &PaperPolicy) -> bool {
		matches!(policy, PaperPolicy::S3FifoHybrid(ratio) if *ratio == self.one_access_ratio)
	}

	fn len(&self) -> usize {
		self.entries.len()
	}

	fn contains(&self, key: HashedKey) -> bool {
		self.entries.contains_key(&key)
	}

	fn insert(&mut self, key: HashedKey, size: ObjectSize) {
		if self.entries.contains_key(&key) {
			// Existing key: track any size change, then treat as an access
			// (a re-`set()` is a genuine reference, same as `TwoQHybridStack`).
			self.resize_key(key, size);
			self.touch(key);
			return;
		}

		// Brand-new key: always admitted into the one-access queue, always
		// slow. If this pushes one_access_used over one_access_capacity,
		// `needs_capacity_eviction` reports it and `apply_evictions` drains
		// it via the real `evict_one()` path (see that method's doc for why
		// eviction can't happen here).
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
		// A shrink may push one_access_used over the new, smaller capacity;
		// `needs_capacity_eviction` reports it, `apply_evictions` drains it.
	}

	fn clear(&mut self) {
		self.one_access_queue.clear();
		self.main_queue.clear();
		self.entries.clear();

		self.one_access_used = 0;
		self.fast_used = 0;
		self.slow_used = 0;
		self.fast_count = 0;
		self.main_count = 0;
		self.main_boundary = None;
		self.migrations.clear();
	}

	/// Priority: `one_access_queue`'s tail first (unconditional — see
	/// `evict_one_access_tail`'s doc); otherwise sweeps `main_queue`'s tail,
	/// giving repeated second chances to accessed keys (classic CLOCK sweep
	/// — bounded by `main_queue`'s length, since each second chance clears
	/// that key's bit and moves it to the front, so it can't be
	/// re-examined again until a full lap) until it finds one with a clear
	/// reference bit to actually evict, or the queue is exhausted.
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
		self.fast_used
	}

	fn slow_bytes_used(&self) -> CacheSize {
		self.one_access_used + self.slow_used
	}

	fn fast_object_count(&self) -> usize {
		self.fast_count
	}

	fn slow_object_count(&self) -> usize {
		self.one_access_queue.len() + (self.main_count - self.fast_count)
	}

	fn needs_capacity_eviction(&self) -> bool {
		self.one_access_used > self.one_access_capacity
	}
}

#[cfg(test)]
mod tests {
	use super::*;

	fn drain(stack: &mut S3FifoHybridStack) -> Vec<(HashedKey, Tier)> {
		stack.drain_tier_migrations()
	}

	#[test]
	fn admission_always_lands_in_one_access_queue_slow() {
		let mut stack = S3FifoHybridStack::new(1.0, 1_000, 1_000);

		stack.insert(1, 10);
		stack.insert(2, 10);

		assert_eq!(stack.tier_of(1), Some(Tier::Slow));
		assert_eq!(stack.tier_of(2), Some(Tier::Slow));
		assert_eq!(stack.slow_bytes_used(), 20);
		assert_eq!(stack.fast_bytes_used(), 0);
		assert_eq!(drain(&mut stack), Vec::new());
	}

	#[test]
	fn reaccessing_a_one_access_key_promotes_it_eagerly_to_fast() {
		let mut stack = S3FifoHybridStack::new(1.0, 1_000, 1_000);

		stack.insert(1, 10);
		drain(&mut stack);

		stack.update(1);
		let migrations = drain(&mut stack);

		assert_eq!(migrations, vec![(1, Tier::Fast)]);
		assert_eq!(stack.tier_of(1), Some(Tier::Fast));
	}

	#[test]
	fn one_access_capacity_pressure_is_reported_not_self_evicted() {
		let mut stack = S3FifoHybridStack::new(1.0, 15, 1_000);

		stack.insert(1, 10);
		drain(&mut stack);
		assert_eq!(stack.contains(1), true);
		assert_eq!(stack.needs_capacity_eviction(), false);

		stack.insert(2, 10);
		let migrations = drain(&mut stack);

		assert_eq!(migrations, Vec::new());
		assert_eq!(stack.contains(1), true);
		assert_eq!(stack.contains(2), true);
		assert_eq!(stack.needs_capacity_eviction(), true);

		assert_eq!(stack.evict_one(), Some(1));
		assert_eq!(stack.contains(1), false);
		assert_eq!(stack.needs_capacity_eviction(), false);
		assert_eq!(stack.tier_of(2), Some(Tier::Slow));
	}

	#[test]
	fn fast_tier_pressure_within_main_queue_demotes_the_oldest() {
		let mut stack = S3FifoHybridStack::new(1.0, 1_000, 25);

		stack.insert(1, 10);
		stack.insert(2, 10);
		stack.update(1); // promote 1 -> Main/Fast
		stack.update(2); // promote 2 -> Main/Fast
		drain(&mut stack);

		assert_eq!(stack.fast_bytes_used(), 20);

		stack.insert(3, 10);
		stack.update(3); // promote 3 -> pushes fast_used to 30 > 25 -> demotes key 1
		let migrations = drain(&mut stack);

		assert!(migrations.iter().any(|(k, t)| *k == 1 && *t == Tier::Slow));
		assert_eq!(stack.tier_of(1), Some(Tier::Slow));
		assert_eq!(stack.tier_of(2), Some(Tier::Fast));
		assert_eq!(stack.tier_of(3), Some(Tier::Fast));
	}

	#[test]
	fn a_mere_access_never_reorders_the_main_queue_or_migrates() {
		// Unlike TwoQHybridStack (LRU main queue), touching a Main/Fast key
		// here must NOT produce a migration or move it in the list -- it
		// only sets the reference bit, consulted lazily at eviction time.
		let mut stack = S3FifoHybridStack::new(1.0, 1_000, 1_000);

		stack.insert(1, 10);
		stack.update(1); // promote to Main/Fast
		drain(&mut stack);

		stack.update(1); // a plain access while already Fast
		let migrations = drain(&mut stack);

		assert_eq!(migrations, Vec::new());
		assert_eq!(stack.tier_of(1), Some(Tier::Fast));
	}

	#[test]
	fn evict_one_gives_an_accessed_slow_key_a_second_chance() {
		// Fast tier fits exactly one 10-byte object at a time.
		let mut stack = S3FifoHybridStack::new(1.0, 1_000, 10);

		stack.insert(1, 10);
		stack.update(1); // promote 1 -> Fast
		drain(&mut stack);

		stack.insert(2, 10);
		stack.update(2); // promote 2 -> Fast, demotes 1 -> Slow (fast_used 20 > 10)
		drain(&mut stack);
		assert_eq!(stack.tier_of(1), Some(Tier::Slow));

		// Access key 1 while it's Slow -- lazily sets its bit, no reorder.
		stack.update(1);
		assert_eq!(drain(&mut stack), Vec::new());
		assert_eq!(stack.tier_of(1), Some(Tier::Slow));

		// Force an eviction sweep: main_queue's tail is key 1 (the only
		// Slow entry). It's accessed, so it gets a second chance (promoted
		// to Fast, which itself demotes key 2 to make room) instead of
		// being evicted; the sweep then evicts key 2 instead, since it now
		// has a clear bit.
		let evicted = stack.evict_one();

		assert_eq!(evicted, Some(2));
		assert_eq!(stack.tier_of(1), Some(Tier::Fast));
		assert_eq!(stack.contains(2), false);
	}

	#[test]
	fn evict_one_evicts_an_unaccessed_slow_tail_directly() {
		let mut stack = S3FifoHybridStack::new(1.0, 1_000, 10);

		stack.insert(1, 10);
		stack.update(1);
		drain(&mut stack);

		stack.insert(2, 10);
		stack.update(2); // demotes 1 -> Slow
		drain(&mut stack);
		assert_eq!(stack.tier_of(1), Some(Tier::Slow));

		// Key 1 was never re-accessed while Slow -- evicted directly.
		assert_eq!(stack.evict_one(), Some(1));
		assert_eq!(stack.tier_of(2), Some(Tier::Fast));
	}

	#[test]
	fn evict_one_prefers_one_access_queue_over_main_queue() {
		let mut stack = S3FifoHybridStack::new(1.0, 1_000, 1_000);

		stack.insert(1, 10); // one-access
		stack.insert(2, 10);
		stack.update(2); // promote 2 -> Main/Fast
		drain(&mut stack);

		assert_eq!(stack.evict_one(), Some(1));
		assert_eq!(stack.tier_of(2), Some(Tier::Fast));
	}

	#[test]
	fn evict_one_falls_back_to_main_fast_when_nothing_demoted_yet() {
		let mut stack = S3FifoHybridStack::new(1.0, 1_000, 25);

		stack.insert(1, 10);
		stack.insert(2, 10);
		stack.update(1);
		stack.update(2);
		drain(&mut stack);

		// one_access_queue empty; both keys are Main/Fast (nothing demoted
		// yet), neither has been accessed since being promoted -- evicts
		// the oldest (key 1) directly.
		assert_eq!(stack.evict_one(), Some(1));
		assert_eq!(stack.tier_of(2), Some(Tier::Fast));
	}

	#[test]
	fn resize_rescales_one_access_capacity_and_reports_pressure() {
		let mut stack = S3FifoHybridStack::new(0.5, 1_000, 1_000); // capacity = 500

		stack.insert(1, 100);
		stack.insert(2, 100);
		drain(&mut stack);
		assert_eq!(stack.slow_bytes_used(), 200);
		assert_eq!(stack.needs_capacity_eviction(), false);

		stack.resize(100); // capacity -> 50 -> both keys now exceed it

		assert_eq!(stack.contains(1), true);
		assert_eq!(stack.contains(2), true);
		assert_eq!(stack.needs_capacity_eviction(), true);

		assert_eq!(stack.evict_one(), Some(1));
		assert_eq!(stack.evict_one(), Some(2));
		assert_eq!(stack.needs_capacity_eviction(), false);
	}

	#[test]
	fn resize_fast_tier_shrink_triggers_demotions() {
		let mut stack = S3FifoHybridStack::new(1.0, 1_000, 1_000);

		stack.insert(1, 10);
		stack.insert(2, 10);
		stack.update(1);
		stack.update(2);
		drain(&mut stack);

		stack.resize_fast_tier(10);
		let migrations = drain(&mut stack);

		assert_eq!(migrations.len(), 1);
		assert_eq!(stack.fast_bytes_used(), 10);
	}

	#[test]
	fn remove_and_clear_reset_bookkeeping() {
		let mut stack = S3FifoHybridStack::new(1.0, 1_000, 1_000);

		stack.insert(1, 10);
		stack.insert(2, 10);
		stack.update(1); // promote 1 -> Main/Fast
		drain(&mut stack);

		stack.remove(1);
		assert_eq!(stack.contains(1), false);
		assert_eq!(stack.fast_bytes_used(), 0);

		stack.remove(2);
		assert_eq!(stack.contains(2), false);
		assert_eq!(stack.slow_bytes_used(), 0);

		stack.insert(3, 10);
		stack.clear();

		assert_eq!(stack.len(), 0);
		assert_eq!(stack.fast_bytes_used(), 0);
		assert_eq!(stack.slow_bytes_used(), 0);
		assert_eq!(stack.tier_of(3), None);
		assert_eq!(stack.evict_one(), None);
	}
}
