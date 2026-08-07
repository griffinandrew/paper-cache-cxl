/*
 * Copyright (c) Kia Shakiba
 *
 * This source code is licensed under the GNU AGPLv3 license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! `S3FifoLazyDemotionFastAdmissionSplitSlowReprieveHybridStack` —
//! `S3FifoLazyDemotionFastAdmissionMidpointReprieveHybridStack` with the
//! approximate mid-slow-segment checkpoint replaced by a real structural
//! one, for
//! `PaperPolicy::S3FifoLazyDemotionFastAdmissionSplitSlowReprieveHybrid`.
//!
//! Everything else carries over unchanged: no ghost queue, a fast-tier
//! one-access queue whose tail is *reprieved into the slow tier* rather
//! than evicted (`settle_one_access`), demotion-time reference-bit
//! reprieve, and the two-physical-list main queue (see that stack's module
//! doc for why the single-list-plus-boundary-cursor shape was abandoned).
//!
//! ## Why the previous checkpoint was replaced
//!
//! The predecessor kept a `slow_midpoint` cursor -- one key, tracked at
//! approximately the middle of the slow segment via a drift counter -- and
//! checked *that single key's* reference bit once per `evict_one()` call.
//! Benchmarked against the real traces it was indistinguishable from not
//! being there at all (largest difference: 291 hits out of 2.34M accesses,
//! i.e. 0.01%).
//!
//! The reason is structural, and it is NOT a coverage problem. (An earlier
//! draft of this doc claimed the cursor sampled too few keys to matter;
//! that was wrong, and is corrected here. In steady state the cursor holds
//! a roughly fixed index while objects flow past it, so it lands on a new
//! object each cycle and sees most objects that cross the midpoint. Its
//! coverage was fine.)
//!
//! The actual reason: **an earlier checkpoint cannot save anything the tail
//! check wouldn't.** Terminal eviction only ever removes the slow tier's
//! *tail*, so any object whose reference bit is set is already spared when
//! it arrives there. A mid-tier check changes *when* a reaccessed object
//! returns to DRAM, never *whether* it survives.
//!
//! This variant tests the one remaining hypothesis that a checkpoint could
//! still pay off: that checking *every* crossing object -- via a real
//! structural boundary rather than a tracked cursor -- gets hot objects
//! back into DRAM earlier and more uniformly, a residency/latency effect
//! rather than a survival one.
//!
//! **It does not.** Measured against the same three traces, this variant
//! produced hit counts bit-identical to the cursor version on all three,
//! while costing 2.7-11.8% on GET p99 and 1.2-6.9% on GET throughput --
//! the extra Slow->Fast migrations are pure added work on the
//! `PolicyWorker` thread and the object map's shard locks. The lineage's
//! next variant (`..._reprieve_...`, no mid-tier checkpoint at all) drops
//! the mechanic entirely. This file is retained as the record of that
//! negative result.
//!
//! ## The two slow segments
//!
//! * `slow_head` -- front = newest slow object (i.e. exactly the fast/slow
//!   boundary), back = the crossing candidate.
//! * `slow_tail` -- front = objects that just crossed, back = oldest object
//!   overall, and the only terminal-eviction candidate.
//!
//! `slow_head` is held to at most `SLOW_HEAD_RATIO` of the slow tier's
//! bytes by `settle_slow_split()`, which is where the crossing check lives:
//!
//! ```text
//! one-access queue (DRAM) ─┐
//!                          ├─> main_fast (DRAM)
//!    promotions ───────────┘        │ demotion (bit clear)
//!                                   v
//!                              slow_head (PMEM)
//!                                   │ crossing check  ── bit set ──> back to main_fast front
//!                                   v bit clear
//!                              slow_tail (PMEM)
//!                                   │ eviction check  ── bit set ──> back to main_fast front
//!                                   v bit clear
//!                                 evicted
//! ```
//!
//! Both checks share one implementation (`give_second_chance`), since both
//! are "bit is set, so move this object to the front of the fast list",
//! which is also what the demotion-boundary reprieve already did.
//!
//! Note what this buys structurally over the predecessor's cursor: no
//! approximation, no drift counter, no cursor-redirect handling at four
//! separate call sites, and the check is O(1) per crossing rather than
//! O(1) per eviction-pass-sampling-one-key. The whole midpoint apparatus
//! is simply gone.

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

/// Fraction of the slow tier's bytes `slow_head` is allowed to hold before
/// `settle_slow_split` starts pushing its tail across into `slow_tail`.
/// 0.5 puts the boundary at the slow tier's midpoint, matching what the
/// predecessor's cursor was approximating -- the difference is that this
/// one is exact and every crossing object is checked, not a sample.
const SLOW_HEAD_RATIO: f64 = 0.5;

/// Which live list a key currently sits in. Doubles as the tier tag --
/// the predecessor carried a separate `Option<Tier>` field alongside a
/// coarser queue tag, which is redundant once the slow tier is two
/// physically distinct lists.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Queue {
	OneAccess,
	Fast,
	SlowHead,
	SlowTail,
}

impl Queue {
	fn tier(self) -> Tier {
		match self {
			Queue::OneAccess | Queue::Fast => Tier::Fast,
			Queue::SlowHead | Queue::SlowTail => Tier::Slow,
		}
	}

	fn is_slow(self) -> bool {
		matches!(self, Queue::SlowHead | Queue::SlowTail)
	}
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct S3FifoEntry {
	queue: Queue,
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

pub struct S3FifoLazyDemotionFastAdmissionSplitSlowReprieveHybridStack {
	one_access_queue: QueueList,

	/// Main queue, fast portion. Front = newest, back = demotion candidate.
	main_fast: QueueList,
	/// Slow tier, newer half. Front = the fast/slow boundary, back = the
	/// crossing candidate.
	slow_head: QueueList,
	/// Slow tier, older half. Back = oldest object overall, the only
	/// terminal-eviction candidate.
	slow_tail: QueueList,

	entries: EntryMap,

	one_access_ratio: f64,
	one_access_capacity: CacheSize,
	one_access_used: CacheSize,

	fast_capacity: CacheSize,
	fast_used: CacheSize,
	slow_head_used: CacheSize,
	slow_tail_used: CacheSize,

	migrations: Vec<(HashedKey, Tier)>,
}

impl S3FifoLazyDemotionFastAdmissionSplitSlowReprieveHybridStack {
	#[cfg(not(feature = "eviction_stacks_pmem"))]
	fn new_collections() -> (QueueList, QueueList, QueueList, QueueList, EntryMap) {
		(
			HashList::default(),
			HashList::default(),
			HashList::default(),
			HashList::default(),
			HashMap::default(),
		)
	}

	#[cfg(feature = "eviction_stacks_pmem")]
	fn new_collections() -> (QueueList, QueueList, QueueList, QueueList, EntryMap) {
		(
			PmemHashList::with_hasher(NoHasher::default()),
			PmemHashList::with_hasher(NoHasher::default()),
			PmemHashList::with_hasher(NoHasher::default()),
			PmemHashList::with_hasher(NoHasher::default()),
			HashMap::with_hasher_in(NoHasher::default(), Hybrid),
		)
	}

	pub fn new(one_access_ratio: f64, max_size: CacheSize, fast_capacity: CacheSize) -> Self {
		let (one_access_queue, main_fast, slow_head, slow_tail, entries) = Self::new_collections();

		S3FifoLazyDemotionFastAdmissionSplitSlowReprieveHybridStack {
			one_access_queue,
			main_fast,
			slow_head,
			slow_tail,

			entries,

			one_access_ratio,
			one_access_capacity: (one_access_ratio * max_size as f64) as CacheSize,
			one_access_used: 0,

			fast_capacity,
			fast_used: 0,
			slow_head_used: 0,
			slow_tail_used: 0,

			migrations: Vec::new(),
		}
	}

	fn effective_main_fast_capacity(&self) -> CacheSize {
		self.fast_capacity.saturating_sub(self.one_access_capacity)
	}

	pub fn tier_of(&self, key: HashedKey) -> Option<Tier> {
		self.entries.get(&key).map(|entry| entry.queue.tier())
	}

	/// Returns `true` if `key` currently sits in the older (`slow_tail`)
	/// slow segment -- i.e. it has already survived a crossing check.
	/// Exposed for tests.
	pub fn is_in_slow_tail(&self, key: HashedKey) -> bool {
		self.entries.get(&key).map(|entry| entry.queue) == Some(Queue::SlowTail)
	}

	fn resize_key(&mut self, key: HashedKey, new_size: ObjectSize) {
		let Some(entry) = self.entries.get_mut(&key) else { return };

		let old_size = entry.size;
		entry.size = new_size;
		let delta = new_size as i64 - old_size as i64;

		let counter = match entry.queue {
			Queue::OneAccess => &mut self.one_access_used,
			Queue::Fast => &mut self.fast_used,
			Queue::SlowHead => &mut self.slow_head_used,
			Queue::SlowTail => &mut self.slow_tail_used,
		};

		*counter = (*counter as i64 + delta).max(0) as CacheSize;
	}

	fn touch(&mut self, key: HashedKey) {
		match self.entries.get(&key).map(|entry| entry.queue) {
			Some(Queue::OneAccess) => self.promote_from_one_access(key),

			// Lazy: a hit on a main-queue key only sets the reference bit.
			// It is read at three points -- the demotion boundary
			// (`settle_fast_tier`), the slow-segment crossing
			// (`settle_slow_split`), and the eviction tail (`evict_one`).
			Some(_) => self.mark_accessed(key),

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
			queue: Queue::Fast,
			size,
			accessed: false,
		});
		self.fast_used += size_bytes;

		self.settle_fast_tier();
	}

	/// Moves `key` to the front of the fast list and clears its reference
	/// bit. Shared by all three reference-bit check points (demotion
	/// boundary, slow-segment crossing, eviction tail), since all three
	/// mean the same thing: this object was reaccessed, so spare it.
	fn give_second_chance(&mut self, key: HashedKey) {
		let Some(entry) = self.entries.get(&key).copied() else { return };
		let size = entry.size as CacheSize;
		let was_slow = entry.queue.is_slow();

		match entry.queue {
			// Only reachable from `evict_one`'s fast-tail fallback (nothing
			// has ever been demoted): reorder within the fast list, no tier
			// change and no byte movement.
			Queue::Fast => {
				self.main_fast.move_front(&key);
			},

			Queue::SlowHead => {
				self.slow_head.remove(&key);
				self.slow_head_used = self.slow_head_used.saturating_sub(size);
				self.main_fast.push_front(key);
				self.fast_used += size;
			},

			Queue::SlowTail => {
				self.slow_tail.remove(&key);
				self.slow_tail_used = self.slow_tail_used.saturating_sub(size);
				self.main_fast.push_front(key);
				self.fast_used += size;
			},

			Queue::OneAccess => return,
		}

		if let Some(entry) = self.entries.get_mut(&key) {
			entry.queue = Queue::Fast;
			entry.accessed = false;
		}

		self.settle_fast_tier();

		// Only record a migration when the object genuinely crossed tiers
		// AND survived the settle above (which can demote it straight back
		// out, in which case that call already pushed the correct
		// `Tier::Slow` migration itself). A key that was already Fast needs
		// no migration at all -- unlike the predecessor, which pushed a
		// redundant Fast->Fast entry here and made `PolicyWorker` rebuild
		// an identical buffer for nothing. That waste is worth avoiding
		// specifically in this variant, where `give_second_chance` fires
		// far more often than before (every crossing, not one sampled key
		// per eviction).
		if was_slow && self.entries.get(&key).map(|entry| entry.queue) == Some(Queue::Fast) {
			self.migrations.push((key, Tier::Fast));
		}
	}

	/// Demotes oldest-first out of `main_fast` into `slow_head` until the
	/// effective fast budget is met, reprieving any key whose bit is set
	/// instead. Terminates even when every fast key's bit is set, since
	/// each reprieve clears one bit.
	///
	/// Deliberately does NOT call `settle_slow_split` -- that method calls
	/// `give_second_chance`, which calls back into here, so the two must
	/// not be mutually recursive. `settle_slow_split` is instead driven
	/// from the public trait methods, and its own loop re-checks after any
	/// nested demotion this method performs.
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
			self.slow_head.push_front(candidate);

			if let Some(entry) = self.entries.get_mut(&candidate) {
				entry.queue = Queue::SlowHead;
			}

			self.fast_used = self.fast_used.saturating_sub(size);
			self.slow_head_used += size;

			self.migrations.push((candidate, Tier::Slow));
		}
	}

	/// Holds `slow_head` to at most `SLOW_HEAD_RATIO` of the slow tier's
	/// bytes, and -- the point of this variant -- checks each object's
	/// reference bit at the moment it would cross into `slow_tail`. A set
	/// bit means the object was reaccessed since it was demoted, so it goes
	/// back to the front of the fast list instead of crossing.
	///
	/// Every crossing object is checked, which is the substantive
	/// difference from the predecessor's single-sampled-key cursor.
	///
	/// Termination: each iteration either moves an object across (strictly
	/// reducing `slow_head_used`) or promotes it out of `slow_head`
	/// entirely. A nested `settle_fast_tier` inside `give_second_chance`
	/// can push bytes back into `slow_head`, but only for keys whose bit is
	/// clear (that method reprieves the rest), and a clear-bit key at
	/// `slow_head`'s back always crosses on the following iteration.
	fn settle_slow_split(&mut self) {
		loop {
			let total = self.slow_head_used + self.slow_tail_used;

			if total == 0 || (self.slow_head_used as f64) <= total as f64 * SLOW_HEAD_RATIO {
				break;
			}

			let Some(candidate) = self.slow_head.back().copied() else { break };

			let accessed = self.entries.get(&candidate).map(|entry| entry.accessed).unwrap_or(false);

			if accessed {
				self.give_second_chance(candidate);
				continue;
			}

			let size = self.entries.get(&candidate).map(|entry| entry.size).unwrap_or(0) as CacheSize;

			self.slow_head.pop_back();
			self.slow_tail.push_front(candidate);

			if let Some(entry) = self.entries.get_mut(&candidate) {
				entry.queue = Queue::SlowTail;
			}

			self.slow_head_used = self.slow_head_used.saturating_sub(size);
			self.slow_tail_used += size;

			// No migration: both segments are the slow tier, so the bytes
			// do not move between DRAM and PMEM.
		}
	}

	/// Relieves one-access-queue pressure by moving its tail(s) to the front
	/// of `slow_head` -- the fast/slow boundary -- as an O(1) `push_front`.
	/// Called synchronously from `insert()`/`resize()`, never through
	/// `evict_one()`: nothing is removed from the cache here, and routing it
	/// through eviction would make `apply_evictions` erase a live object (or
	/// fall back to evicting a random one). See the predecessor's module doc
	/// for the full account of that bug.
	fn settle_one_access(&mut self) {
		while self.one_access_used > self.one_access_capacity {
			let Some(key) = self.one_access_queue.pop_back() else { break };
			let Some(entry) = self.entries.get(&key).copied() else { continue };
			let size = entry.size as CacheSize;

			self.one_access_used = self.one_access_used.saturating_sub(size);

			self.slow_head.push_front(key);

			if let Some(stored) = self.entries.get_mut(&key) {
				stored.queue = Queue::SlowHead;
				stored.accessed = false;
			}

			self.slow_head_used += size;

			self.migrations.push((key, Tier::Slow));
		}
	}
}

impl PolicyStack for S3FifoLazyDemotionFastAdmissionSplitSlowReprieveHybridStack {
	fn is_policy(&self, policy: &PaperPolicy) -> bool {
		matches!(policy, PaperPolicy::S3FifoLazyDemotionFastAdmissionSplitSlowReprieveHybrid(ratio) if *ratio == self.one_access_ratio)
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
			self.settle_slow_split();
			return;
		}

		self.one_access_queue.push_front(key);
		self.entries.insert(key, S3FifoEntry {
			queue: Queue::OneAccess,
			size,
			accessed: false,
		});
		self.one_access_used += size as CacheSize;

		self.settle_one_access();
		self.settle_slow_split();
	}

	fn update(&mut self, key: HashedKey) {
		if self.entries.contains_key(&key) {
			self.touch(key);
			self.settle_slow_split();
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

			Queue::Fast => {
				self.main_fast.remove(&key);
				self.fast_used = self.fast_used.saturating_sub(size);
			},

			Queue::SlowHead => {
				self.slow_head.remove(&key);
				self.slow_head_used = self.slow_head_used.saturating_sub(size);
			},

			Queue::SlowTail => {
				self.slow_tail.remove(&key);
				self.slow_tail_used = self.slow_tail_used.saturating_sub(size);
			},
		}
	}

	fn resize(&mut self, max_size: CacheSize) {
		self.one_access_capacity = (self.one_access_ratio * max_size as f64) as CacheSize;
		self.settle_one_access();
		self.settle_fast_tier();
		self.settle_slow_split();
	}

	fn clear(&mut self) {
		self.one_access_queue.clear();
		self.main_fast.clear();
		self.slow_head.clear();
		self.slow_tail.clear();
		self.entries.clear();

		self.one_access_used = 0;
		self.fast_used = 0;
		self.slow_head_used = 0;
		self.slow_tail_used = 0;
		self.migrations.clear();
	}

	fn evict_one(&mut self) -> Option<HashedKey> {
		// Crossing checks fire here, keeping the split balanced before the
		// tail is evaluated. The one-access queue is never consulted --
		// its pressure is relieved synchronously by `settle_one_access`.
		self.settle_slow_split();

		loop {
			// The oldest slow object is the real candidate; fall back
			// through slow_head, then the fast tail, only when the older
			// lists are empty (i.e. little or nothing has been demoted).
			let (key, from) = if let Some(key) = self.slow_tail.back().copied() {
				(key, Queue::SlowTail)
			} else if let Some(key) = self.slow_head.back().copied() {
				(key, Queue::SlowHead)
			} else {
				(self.main_fast.back().copied()?, Queue::Fast)
			};

			let accessed = self.entries.get(&key).map(|entry| entry.accessed).unwrap_or(false);

			if accessed {
				self.give_second_chance(key);
				continue;
			}

			let size = self.entries.get(&key).map(|entry| entry.size).unwrap_or(0) as CacheSize;

			match from {
				Queue::SlowTail => {
					self.slow_tail.pop_back();
					self.slow_tail_used = self.slow_tail_used.saturating_sub(size);
				},

				Queue::SlowHead => {
					self.slow_head.pop_back();
					self.slow_head_used = self.slow_head_used.saturating_sub(size);
				},

				Queue::Fast => {
					self.main_fast.pop_back();
					self.fast_used = self.fast_used.saturating_sub(size);
				},

				Queue::OneAccess => break,
			}

			self.entries.remove(&key);

			return Some(key);
		}

		None
	}

	fn resize_fast_tier(&mut self, size: CacheSize) {
		self.fast_capacity = size;
		self.settle_fast_tier();
		self.settle_slow_split();
	}

	fn drain_tier_migrations(&mut self) -> Vec<(HashedKey, Tier)> {
		std::mem::take(&mut self.migrations)
	}

	fn fast_bytes_used(&self) -> CacheSize {
		self.fast_used + self.one_access_used
	}

	fn slow_bytes_used(&self) -> CacheSize {
		self.slow_head_used + self.slow_tail_used
	}

	fn fast_object_count(&self) -> usize {
		self.main_fast.len() + self.one_access_queue.len()
	}

	fn slow_object_count(&self) -> usize {
		self.slow_head.len() + self.slow_tail.len()
	}
}

#[cfg(test)]
mod tests {
	use super::*;

	fn drain(stack: &mut S3FifoLazyDemotionFastAdmissionSplitSlowReprieveHybridStack) -> Vec<(HashedKey, Tier)> {
		stack.drain_tier_migrations()
	}

	#[test]
	fn admission_always_lands_in_one_access_queue_fast() {
		let mut stack = S3FifoLazyDemotionFastAdmissionSplitSlowReprieveHybridStack::new(1.0, 1_000, 1_000);

		stack.insert(1, 10);
		stack.insert(2, 10);

		assert_eq!(stack.tier_of(1), Some(Tier::Fast));
		assert_eq!(stack.tier_of(2), Some(Tier::Fast));
		assert_eq!(drain(&mut stack), Vec::new());
	}

	#[test]
	fn a_key_aging_out_without_reaccess_is_moved_to_slow_instead_of_evicted() {
		// one_access_capacity = 0.01 * 1_000 = 10 -- fits exactly one
		// 10-byte key, so admitting a second reprieves the first.
		let mut stack = S3FifoLazyDemotionFastAdmissionSplitSlowReprieveHybridStack::new(0.01, 1_000, 1_000);

		stack.insert(1, 10);
		drain(&mut stack);
		assert_eq!(stack.tier_of(1), Some(Tier::Fast), "still in the one-access queue");

		stack.insert(2, 10);

		assert!(stack.contains(1), "the key must still be tracked, not gone");
		assert_eq!(stack.tier_of(1), Some(Tier::Slow));
		assert_eq!(stack.tier_of(2), Some(Tier::Fast));
		assert_eq!(drain(&mut stack), vec![(1, Tier::Slow)]);
	}

	// ── the signature new mechanic: a real crossing checkpoint ─────────────

	/// Three keys in the slow tier, oldest-first. `one_access_ratio` of 0.0
	/// makes `settle_one_access` fire from within `insert()` itself, so each
	/// key lands in `slow_head` immediately with a CLEAR reference bit --
	/// note there is deliberately no `update()` here, unlike the equivalent
	/// helpers in this stack's predecessors: at this ratio a key never
	/// passes through the fast list, so an `update()` would only set the
	/// reference bit and defeat the point of the fixture.
	///
	/// `slow_head` is held to half the slow bytes, so 3 x 10 bytes settles
	/// at slow_head = [3] (10 bytes) / slow_tail = [2, 1] (20 bytes).
	fn build_three_slow_keys() -> S3FifoLazyDemotionFastAdmissionSplitSlowReprieveHybridStack {
		let mut stack = S3FifoLazyDemotionFastAdmissionSplitSlowReprieveHybridStack::new(0.0, 1_000, 0);

		for key in 1..=3u64 {
			stack.insert(key, 10);
		}

		drain(&mut stack);
		stack
	}

	#[test]
	fn the_split_pushes_the_older_slow_keys_into_the_tail_segment() {
		let stack = build_three_slow_keys();

		assert_eq!(stack.tier_of(1), Some(Tier::Slow));
		assert_eq!(stack.tier_of(2), Some(Tier::Slow));
		assert_eq!(stack.tier_of(3), Some(Tier::Slow));

		assert!(stack.is_in_slow_tail(1), "oldest should have crossed into the tail segment");
		assert!(stack.is_in_slow_tail(2), "second-oldest should have crossed too");
		assert!(!stack.is_in_slow_tail(3), "newest should still be in the head segment");
		assert_eq!(stack.slow_bytes_used(), 30);
	}

	#[test]
	fn a_reaccessed_key_is_promoted_at_the_crossing_instead_of_crossing() {
		// slow_head = [3], slow_tail = [2, 1]; key 3 is the next crossing
		// candidate.
		let mut stack = build_three_slow_keys();
		assert!(!stack.is_in_slow_tail(3));

		// Reaccess key 3 -- lazily, so nothing moves yet.
		stack.update(3);
		assert_eq!(stack.tier_of(3), Some(Tier::Slow), "a mere access must not itself migrate");
		drain(&mut stack);

		// Give the fast tier somewhere to promote into (it was 0 in the
		// fixture, which would have demoted the key straight back out).
		stack.resize_fast_tier(100);

		// Grow slow_head until a crossing is due: at 4 keys the split is
		// exactly balanced, at 5 it tips. Key 3 is the crossing candidate
		// and its bit is set, so it must be promoted rather than cross.
		stack.insert(4, 10);
		stack.insert(5, 10);

		assert_eq!(stack.tier_of(3), Some(Tier::Fast), "the reaccessed key should have been promoted at the crossing check");
		assert!(!stack.is_in_slow_tail(3), "and must not have crossed into the tail segment");
		assert!(drain(&mut stack).contains(&(3, Tier::Fast)), "a real Slow->Fast migration must be recorded");
	}

	#[test]
	fn an_unaccessed_key_crosses_normally() {
		let mut stack = build_three_slow_keys();

		// Key 3 is the head segment's only occupant and has a clear bit.
		// Growing the slow tier pushes it across rather than promoting it.
		assert!(!stack.is_in_slow_tail(3));

		// At 4 keys the split is exactly balanced; the 5th tips it and
		// makes key 3 the crossing candidate.
		stack.insert(4, 10);
		stack.insert(5, 10);
		drain(&mut stack);

		assert!(stack.is_in_slow_tail(3), "an unaccessed key must cross, not be promoted");
		assert_eq!(stack.tier_of(3), Some(Tier::Slow));
	}

	#[test]
	fn eviction_takes_the_slow_tail_and_still_honors_the_reference_bit() {
		let mut stack = build_three_slow_keys();

		// Key 1 is the oldest (slow_tail's back). Untouched -> evicted.
		assert_eq!(stack.evict_one(), Some(1));
		assert!(!stack.contains(1));

		// Now key 2 is the oldest; reaccess it so the tail check spares it.
		stack.update(2);
		stack.resize_fast_tier(100);

		let evicted = stack.evict_one();

		assert_eq!(stack.tier_of(2), Some(Tier::Fast), "an accessed tail key should be promoted, not evicted");
		assert_ne!(evicted, Some(2));
	}

	#[test]
	fn a_reprieved_key_can_still_be_promoted_by_a_later_access() {
		let mut stack = S3FifoLazyDemotionFastAdmissionSplitSlowReprieveHybridStack::new(0.01, 1_000, 1_000);

		stack.insert(1, 10);
		stack.insert(2, 10);
		stack.insert(3, 10);
		drain(&mut stack);

		assert_eq!(stack.tier_of(1), Some(Tier::Slow));
		assert_eq!(stack.tier_of(2), Some(Tier::Slow));

		stack.update(1);
		drain(&mut stack);

		let evicted = stack.evict_one();

		assert_eq!(stack.tier_of(1), Some(Tier::Fast), "a reprieved key stays promotable via the ordinary second chance");
		assert_ne!(evicted, Some(1));
	}

	#[test]
	fn an_accessed_fast_boundary_key_is_reprieved_at_demotion_time_not_demoted() {
		let mut stack = S3FifoLazyDemotionFastAdmissionSplitSlowReprieveHybridStack::new(1.0, 1_000, 1_010);

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

	#[test]
	fn evict_one_falls_back_to_the_fast_tail_when_nothing_has_ever_been_demoted() {
		let mut stack = S3FifoLazyDemotionFastAdmissionSplitSlowReprieveHybridStack::new(1.0, 1_000, 10_000);

		for key in 1..=3u64 {
			stack.insert(key, 10);
			stack.update(key);
		}
		drain(&mut stack);

		assert_eq!(stack.slow_object_count(), 0, "nothing should have been demoted yet");

		assert_eq!(stack.evict_one(), Some(1));
		assert!(!stack.contains(1));
		assert_eq!(stack.fast_bytes_used(), 20);
	}

	#[test]
	fn remove_handles_every_list() {
		let mut stack = S3FifoLazyDemotionFastAdmissionSplitSlowReprieveHybridStack::new(0.02, 1_000, 10_000);

		// one_access_capacity = 20 (two 10-byte keys). Land one key in each
		// list: key 2 promoted to fast by its update(), key 1 pushed out of
		// the one-access queue into slow once keys 3 and 4 fill it, and
		// keys 3/4 left sitting in the one-access queue.
		stack.insert(1, 10);
		stack.insert(2, 10);
		stack.update(2);
		stack.insert(3, 10);
		stack.insert(4, 10);
		drain(&mut stack);

		assert_eq!(stack.tier_of(1), Some(Tier::Slow), "oldest one-access key should have been reprieved into slow");
		assert_eq!(stack.tier_of(2), Some(Tier::Fast));

		stack.remove(1);
		stack.remove(2);
		stack.remove(3);
		stack.remove(4);

		assert_eq!(stack.len(), 0);
		assert_eq!(stack.fast_bytes_used(), 0);
		assert_eq!(stack.slow_bytes_used(), 0);
	}

	#[test]
	fn clear_resets_all_four_lists() {
		let mut stack = build_three_slow_keys();

		stack.clear();

		assert_eq!(stack.len(), 0);
		assert_eq!(stack.tier_of(1), None);
		assert_eq!(stack.fast_bytes_used(), 0);
		assert_eq!(stack.slow_bytes_used(), 0);
		assert_eq!(stack.evict_one(), None);
	}

	#[test]
	fn fast_and_slow_gauges_include_one_access_queue_on_the_fast_side() {
		let mut stack = S3FifoLazyDemotionFastAdmissionSplitSlowReprieveHybridStack::new(1.0, 1_000, 1_000);

		stack.insert(1, 10);
		stack.insert(2, 10);
		drain(&mut stack);

		assert_eq!(stack.fast_bytes_used(), 20);
		assert_eq!(stack.slow_bytes_used(), 0);
		assert_eq!(stack.fast_object_count(), 2);
		assert_eq!(stack.slow_object_count(), 0);
	}
}
