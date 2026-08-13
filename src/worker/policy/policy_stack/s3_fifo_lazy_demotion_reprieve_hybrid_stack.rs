/*
 * Copyright (c) Kia Shakiba
 *
 * This source code is licensed under the GNU AGPLv3 license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! `S3FifoLazyDemotionReprieveHybridStack` — a slow-tier one-access queue
//! whose aged-out keys are reprieved into the main queue rather than
//! evicted. For `PaperPolicy::S3FifoLazyDemotionReprieveHybrid`.
//!
//! This fills the one empty cell in the s3-fifo family's design matrix. Every
//! other variant pairs its one-access-queue *placement* with a fixed choice of
//! what happens to a key that ages out of it:
//!
//! | variant | one-access tier | ages out without reaccess |
//! |---|---|---|
//! | `S3FifoHybridStack` (+ghost, +lazy demotion) | slow | evicted |
//! | `...FastAdmission...` (+midpoint) | fast | evicted |
//! | `...FastAdmissionReprieve...` (+midpoint, +split slow) | fast | reprieved |
//! | **this stack** | **slow** | **reprieved** |
//!
//! ## Why the combination is interesting: the splice costs nothing
//!
//! In the fast-admission reprieve variants the one-access queue is DRAM and
//! the main queue's slow segment is PMEM, so relieving one-access pressure
//! pushes a `(key, Tier::Slow)` migration and `apply_tier_migrations`
//! performs a real `TieredBuffer::new_slow` PMEM copy for *every* aged-out
//! object.
//!
//! Here both structures are in PMEM. `settle_one_access` moves the key from
//! one list to another and emits **no migration at all** — the bytes never
//! move. That makes the reprieve strictly cheaper than the eviction it
//! replaces (which also had to do bookkeeping, and additionally dropped the
//! object).
//!
//! The cost is on the other side of the ledger, and it is the paper-literal
//! admission rule this variant keeps: every `set()` is a synchronous PMEM
//! write on the calling thread. That is precisely the cost the fast-admission
//! branch exists to avoid.
//!
//! ## Promotion is a real move again
//!
//! The mirror image of the above. `promote_from_one_access` in the
//! fast-admission variants deliberately pushes *no* migration, because the
//! bytes were already in DRAM and a `Tier::Fast` migration would have been a
//! pointless DRAM→DRAM copy. Here a one-access key really is in PMEM, so
//! promoting it is a genuine PMEM→DRAM move and must emit the migration —
//! guarded, because `settle_fast_tier` may demote the key straight back out
//! in the same call (see that function).
//!
//! ## Lazy demotion (retained)
//!
//! `settle_fast_tier` is reference-bit gated: the key anchoring the fast/slow
//! boundary is demoted only if its `accessed` bit is clear. If the bit is
//! set, the key is given a fresh start at the front of `main_fast` with the
//! bit cleared and the sweep continues to the next candidate. Termination is
//! guaranteed because each reprieve clears exactly one bit.
//!
//! This matters more here than in the fast-admission variants: a demotion in
//! this design is a real DRAM→PMEM copy, so every demotion lazy demotion
//! avoids is a copy saved outright.
//!
//! ## No ghost queue
//!
//! Nothing would populate it. A ghost list records keys evicted from the
//! one-access queue, and here no key is ever evicted from it — they are all
//! reprieved into the main queue instead. The ghost machinery is therefore
//! absent rather than present-and-always-empty, matching
//! `S3FifoLazyDemotionFastAdmissionReprieveHybridStack`.
//!
//! ## Two physical main-queue lists
//!
//! `main_fast` and `main_slow` are separate `HashList`s rather than one list
//! plus a boundary cursor: demotion is `main_fast.pop_back()` +
//! `main_slow.push_front()`, promotion is `main_slow.remove()` +
//! `main_fast.push_front()`, eviction is `main_slow.pop_back()` (falling back
//! to `main_fast.pop_back()` only when nothing has ever been demoted). All
//! O(1), and the reprieve splice targets `main_slow.push_front()` — which
//! *is* the boundary position — so it is O(1) too.
//!
//! ## `eviction_stacks_pmem`
//!
//! Same DRAM/PMEM backing switch as every other hybrid stack.

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

pub struct S3FifoLazyDemotionReprieveHybridStack {
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

	migrations: Vec<(HashedKey, Tier)>,
}

impl S3FifoLazyDemotionReprieveHybridStack {
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

		S3FifoLazyDemotionReprieveHybridStack {
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

			migrations: Vec::new(),
		}
	}

	/// The whole `fast_capacity` is available to the main queue.
	///
	/// The fast-admission variants subtract `one_access_capacity` here,
	/// because there the one-access queue is DRAM-resident and both budgets
	/// draw on the same physical pool. Here the one-access queue lives in
	/// PMEM, so it competes for nothing the main queue's fast segment wants
	/// -- reserving against it would silently shrink the DRAM tier for no
	/// reason. `one_access_capacity` still bounds the one-access queue's own
	/// (PMEM) footprint via `settle_one_access`.
	fn effective_main_fast_capacity(&self) -> CacheSize {
		self.fast_capacity
	}

	pub fn tier_of(&self, key: HashedKey) -> Option<Tier> {
		let entry = self.entries.get(&key)?;

		match entry.queue {
			Queue::OneAccess => Some(Tier::Slow),
			Queue::Main => entry.tier,
		}
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

		// Unlike the fast-admission variants -- where a one-access entry's
		// bytes are already in DRAM, so promoting it moved nothing and
		// pushing a migration would have been a pointless DRAM->DRAM copy --
		// the bytes genuinely are in PMEM here, so this needs a real
		// promotion migration.
		//
		// Guarded for the same reason `give_second_chance` guards its own:
		// `settle_fast_tier` above may have demoted this very key straight
		// back out (a zero/tiny effective budget), in which case it already
		// pushed the correct `Tier::Slow` migration. Pushing `Tier::Fast`
		// as well would be applied *after* it -- `apply_tier_migrations`
		// runs every demotion before any promotion -- leaving the bytes in
		// DRAM while the stack believes they are in PMEM.
		if self.entries.get(&key).and_then(|entry| entry.tier) == Some(Tier::Fast) {
			self.migrations.push((key, Tier::Fast));
		}
	}

	/// The eviction-time second chance, shared with the demotion-boundary
	/// reprieve: both mean "this key's reference bit is set, so spare it
	/// and move it to the front of the fast list".
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
				self.main_slow.remove(&key);
				self.main_fast.push_front(key);

				if let Some(entry) = self.entries.get_mut(&key) {
					entry.tier = Some(Tier::Fast);
					entry.accessed = false;
				}

				self.slow_used = self.slow_used.saturating_sub(size);
				self.fast_used += size;

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

			// No migration. Both the one-access queue and the main queue's
			// slow segment live in PMEM, so this splice moves the key
			// between two lists without moving a single byte -- the whole
			// point of pairing a slow-tier one-access queue with the
			// reprieve. The fast-admission reprieve variants must push a
			// `Tier::Slow` migration here (a real DRAM->PMEM copy per
			// aged-out object); this design gets the same behaviour for
			// free.
		}
	}
}

impl PolicyStack for S3FifoLazyDemotionReprieveHybridStack {
	fn is_policy(&self, policy: &PaperPolicy) -> bool {
		matches!(policy, PaperPolicy::S3FifoLazyDemotionReprieveHybrid(ratio) if *ratio == self.one_access_ratio)
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
					self.main_slow.remove(&key);
					self.slow_used = self.slow_used.saturating_sub(size);

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
		self.migrations.clear();
	}

	fn evict_one(&mut self) -> Option<HashedKey> {
		// The one-access queue never reaches here -- its own capacity
		// pressure is relieved synchronously by `settle_one_access()` (see
		// the module doc), the same way the main queue's fast/slow boundary
		// is settled by `settle_fast_tier()` rather than through eviction.
		// This is purely the main queue's ordinary tail loop.
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
				self.main_slow.pop_back();
			} else {
				self.main_fast.pop_back();
			}

			let removed = self.entries.remove(&key);
			let size = removed.map(|entry| entry.size).unwrap_or(0) as CacheSize;

			if from_slow {
				self.slow_used = self.slow_used.saturating_sub(size);
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

	// The one-access queue counts toward the SLOW gauges here, not the fast
	// ones. The fast-admission variants add `one_access_used` to
	// `fast_bytes_used` because their one-access queue really is DRAM; this
	// variant's is PMEM, so attributing it to the fast tier would over-report
	// DRAM usage by the whole one-access budget and under-report PMEM by the
	// same amount. `tier_of` already reports `Tier::Slow` for these keys --
	// these gauges must agree with it.
	fn fast_bytes_used(&self) -> CacheSize {
		self.fast_used
	}

	fn slow_bytes_used(&self) -> CacheSize {
		self.slow_used + self.one_access_used
	}

	fn fast_object_count(&self) -> usize {
		self.main_fast.len()
	}

	fn slow_object_count(&self) -> usize {
		self.main_slow.len() + self.one_access_queue.len()
	}
}

#[cfg(test)]
mod tests {
	use super::*;

	fn drain(stack: &mut S3FifoLazyDemotionReprieveHybridStack) -> Vec<(HashedKey, Tier)> {
		stack.drain_tier_migrations()
	}

	#[test]
	// ── the set protocol ──────────────────────────────────────────────
	//
	// `insert()` is what a `set()` routes to, and its behaviour depends
	// entirely on whether the key is already tracked. These three cases are
	// the protocol, pinned explicitly rather than inferred from the
	// admission/promotion tests below -- they are what differs from the
	// fast-admission variants, where a brand-new set lands in DRAM.

	#[test]
	fn set_of_a_brand_new_key_lands_in_the_one_access_queue_in_the_slow_tier() {
		let mut stack = S3FifoLazyDemotionReprieveHybridStack::new(1.0, 1_000, 1_000);

		stack.insert(1, 10);

		assert_eq!(stack.tier_of(1), Some(Tier::Slow), "paper-literal admission: a new object goes to the slow tier");
		assert_eq!(stack.slow_bytes_used(), 10);
		assert_eq!(stack.fast_bytes_used(), 0, "a brand-new set must never touch the DRAM budget");
		assert_eq!(
			drain(&mut stack), Vec::new(),
			"no migration: the API layer already built the value as Slow (see this feature's admission_tier)",
		);
	}

	#[test]
	fn set_of_an_existing_one_access_key_counts_as_a_reaccess_and_promotes_it() {
		// A set is an access. A key sitting in the one-access queue that is
		// set again has therefore demonstrated reuse, and is promoted to the
		// main queue's fast segment exactly as a get would promote it.
		let mut stack = S3FifoLazyDemotionReprieveHybridStack::new(1.0, 1_000, 1_000);

		stack.insert(1, 10);
		drain(&mut stack);
		assert_eq!(stack.tier_of(1), Some(Tier::Slow));

		stack.insert(1, 10);

		assert_eq!(stack.tier_of(1), Some(Tier::Fast), "a re-set is a reaccess, so it promotes");
		assert_eq!(
			drain(&mut stack), vec![(1, Tier::Fast)],
			"a real PMEM->DRAM migration: unlike the fast-admission variants the bytes genuinely move here",
		);
	}

	#[test]
	fn set_of_an_existing_main_slow_key_marks_the_bit_without_migrating() {
		// Once in the main queue, a set behaves like any other access under
		// lazy promotion: it sets the reference bit and nothing else. The
		// key is only returned to DRAM later, by the demotion-boundary
		// reprieve or the eviction-time second chance.
		let mut stack = S3FifoLazyDemotionReprieveHybridStack::new(0.01, 1_000, 10);

		stack.insert(1, 10);
		stack.update(1);          // promote into main_fast
		stack.insert(2, 10);
		stack.update(2);          // promotes 2, demoting 1 to main_slow
		drain(&mut stack);
		assert_eq!(stack.tier_of(1), Some(Tier::Slow), "precondition: key 1 is in the main queue's slow segment");

		stack.insert(1, 10);

		assert_eq!(stack.tier_of(1), Some(Tier::Slow), "a set on a main-queue key does not itself promote");
		assert_eq!(drain(&mut stack), Vec::new(), "and moves no bytes");
	}

	fn admission_always_lands_in_one_access_queue_slow() {
		let mut stack = S3FifoLazyDemotionReprieveHybridStack::new(1.0, 1_000, 1_000);

		stack.insert(1, 10);
		stack.insert(2, 10);

		// Slow, not Fast: this variant's one-access queue lives in PMEM.
		assert_eq!(stack.tier_of(1), Some(Tier::Slow));
		assert_eq!(stack.tier_of(2), Some(Tier::Slow));
		assert_eq!(drain(&mut stack), Vec::new());
	}

	#[test]
	fn a_key_aging_out_without_reaccess_is_moved_to_slow_instead_of_evicted() {
		// one_access_capacity = 0.01 * 1_000 = 10 -- fits exactly one 10-byte
		// key. Admitting a second pushes one_access_used to 20 > 10,
		// synchronously reprieving the oldest (key 1) from insert() itself.
		let mut stack = S3FifoLazyDemotionReprieveHybridStack::new(0.01, 1_000, 1_000);

		stack.insert(1, 10);
		drain(&mut stack);
		assert_eq!(stack.tier_of(1), Some(Tier::Slow), "still in the one-access queue (which is slow here)");

		stack.insert(2, 10);

		assert!(stack.contains(1), "the key must still be tracked, not gone");
		assert_eq!(stack.tier_of(1), Some(Tier::Slow), "aged-out key should land directly in the main queue's slow tier");
		assert_eq!(stack.tier_of(2), Some(Tier::Slow), "the newer key stays in the one-access queue (slow)");

		let migrations = drain(&mut stack);
		assert_eq!(migrations, Vec::new(), "no migration: one-access queue and main_slow are both PMEM, so the splice moves no bytes");
	}

	#[test]
	fn a_reprieved_key_can_still_be_promoted_by_a_later_access() {
		let mut stack = S3FifoLazyDemotionReprieveHybridStack::new(0.01, 1_000, 1_000);

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
		assert_eq!(stack.tier_of(3), Some(Tier::Slow), "still sitting untouched in the one-access queue (slow)");

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
		// fast_capacity is generous so promoted keys stay put. Note this
		// variant does NOT subtract one_access_capacity from it -- the
		// one-access queue is PMEM and competes for nothing here.
		let mut stack = S3FifoLazyDemotionReprieveHybridStack::new(1.0, 1_000, 10_000);

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
		assert_eq!(stack.tier_of(4), Some(Tier::Slow), "still sitting untouched in the one-access queue (slow)");

		stack.resize(0);

		assert_eq!(stack.tier_of(4), Some(Tier::Slow));
		assert_eq!(drain(&mut stack), Vec::new(), "slow->slow splice: no migration");

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
		let mut stack = S3FifoLazyDemotionReprieveHybridStack::new(0.0, 1_000, 1_000);

		stack.insert(1, 10);

		assert_eq!(stack.fast_bytes_used(), 0, "a reprieved key must never be counted as fast, even transiently");
		assert_eq!(stack.slow_bytes_used(), 10);
	}

	#[test]
	fn an_accessed_fast_boundary_key_is_reprieved_at_demotion_time_not_demoted() {
		let mut stack = S3FifoLazyDemotionReprieveHybridStack::new(1.0, 1_000, 10);

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

	fn build_five_key_stack() -> S3FifoLazyDemotionReprieveHybridStack {
		let mut stack = S3FifoLazyDemotionReprieveHybridStack::new(1.0, 1_000, 20);

		for key in 1..=5u64 {
			stack.insert(key, 10);
			stack.update(key);
		}

		drain(&mut stack);
		stack
	}

	#[test]
	fn evict_one_gives_an_accessed_slow_key_a_second_chance() {
		let mut stack = S3FifoLazyDemotionReprieveHybridStack::new(1.0, 1_000, 10);

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
		let mut stack = S3FifoLazyDemotionReprieveHybridStack::new(1.0, 1_000, 10_000);

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
		let mut stack = S3FifoLazyDemotionReprieveHybridStack::new(1.0, 1_000, 1_000);

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
	fn fast_and_slow_gauges_count_the_one_access_queue_on_the_slow_side() {
		// The inverse of the fast-admission variants' equivalent test. Two
		// keys sitting in the one-access queue are PMEM-resident here, so
		// they must show up in the slow gauges and leave the DRAM gauges at
		// zero -- otherwise `*_hybrid_stats()` would report a fast tier that
		// is entirely PMEM.
		let mut stack = S3FifoLazyDemotionReprieveHybridStack::new(1.0, 1_000, 1_000);

		stack.insert(1, 10);
		stack.insert(2, 10);
		drain(&mut stack);

		assert_eq!(stack.fast_bytes_used(), 0);
		assert_eq!(stack.slow_bytes_used(), 20);
		assert_eq!(stack.fast_object_count(), 0);
		assert_eq!(stack.slow_object_count(), 2);
	}
}
