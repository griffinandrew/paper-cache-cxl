/*
 * Copyright (c) Kia Shakiba
 *
 * This source code is licensed under the GNU AGPLv3 license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! `FifoHybridStack` — a single, segmented FIFO queue for `PaperPolicy::FifoHybrid`.
//!
//! One insertion-ordered list backs both tiers. The fast tier is the maximal
//! prefix of the list (starting from the head/newest end) whose cumulative
//! byte size fits within `fast_capacity`; everything else is the slow tier.
//! A brand-new key is admitted at the front (bottom of the fast tier);
//! whenever that pushes `fast_used` over `fast_capacity`, the oldest fast key
//! (tracked via `fast_boundary`, so no scan is needed) is demoted.
//! `evict_one` always pops the absolute tail (oldest key overall), which —
//! once any demotion has occurred — is always in the slow tier.
//!
//! ## No promotion, ever — this is the defining difference from `LruHybridStack`
//!
//! The paper's FIFO-hybrid spec has no promotion policy at all: "objects age
//! through the queue in insertion order... and are never reordered
//! regardless of subsequent accesses." Concretely this means:
//!
//! - `update()` (called on a cache `get()` hit, via `record_access`'s default
//!   `if hit { self.update(key); }` composition) is **deliberately left as
//!   the `PolicyStack` trait's default no-op body** — not overridden here at
//!   all, unlike every sibling hybrid stack (`LruHybridStack`/
//!   `LfuHybridStack`), which override it to reorder/promote on a hit. A hit
//!   on a slow-tier key must never migrate it back to fast. This exactly
//!   matches this crate's own plain (non-hybrid) `FifoStack`
//!   (`worker/policy/policy_stack/fifo_stack.rs`), which also never
//!   overrides `update()`. **Do not add an override here that does
//!   anything** — if a future refactor pass assumes every other stack in
//!   this directory overrides `update()` and this one "forgot" to, that
//!   assumption is wrong.
//! - `insert()` on an *existing* key (a `set()` overwrite) never repositions
//!   it in `queue` and never changes its tier, regardless of which tier it
//!   currently occupies — only the size accounting for whichever tier it's
//!   already in is corrected if the byte length changed. Contrast with
//!   `LruHybridStack::insert`, which unconditionally treats an existing key
//!   as "touch to front, promote if slow."
//!
//! ## No shared-DRAM-overhead reservation or low-water headroom (yet)
//!
//! Per an explicit product decision, this is a pure implementation of the
//! paper's spec: `settle_fast_tier` triggers once `fast_used > fast_capacity`
//! and drains back down to exactly `fast_capacity` — no
//! `with_shared_overhead`/`reserved_overhead` term (contrast
//! `LruHybridStack`/`LfuHybridStack`, which reserve DRAM for the shared
//! object hashtable + eviction stacks out of the fast-tier budget) and no
//! `FAST_TIER_LOW_WATER_RATIO`-style drain-below-ceiling headroom (contrast
//! `LruHybridStack`, which reintroduced a small one for burst-write safety).
//! `LruHybridStack`'s module doc explains why those were added there: real
//! usage measurements showed real DRAM overshooting `fast_tier_size` and
//! concurrent `set()` bursts landing just over budget between worker passes.
//! Those measurements have not yet been run for `fifo_hybrid_cache`; if a
//! follow-up measurement shows the same issues here, port the same fixes
//! over (see this crate's planning notes for the exact follow-up steps).
//!
//! This stack only tracks *order and tier membership*; it does not move any
//! bytes itself. `PolicyWorker` drains `drain_tier_migrations` after each
//! `insert` call and performs the actual `TieredBuffer` reallocation against
//! the shared object map (see `Object::set_data`).
//!
//! ## One combined per-key map, not two
//!
//! Every tracked key needs both a tier and a size, and nearly every
//! operation here touches both together — matching `LruHybridStack`'s
//! `entries: HashMap<HashedKey, FifoEntry>` (`FifoEntry { tier, size }`)
//! consolidation (see that stack's module doc for the history of why this
//! collapsed from two separate maps).

#[cfg(not(feature = "eviction_stacks_pmem"))]
use std::collections::HashMap;
#[cfg(feature = "eviction_stacks_pmem")]
use hashbrown::HashMap;

#[cfg(not(feature = "eviction_stacks_pmem"))]
use kwik::collections::HashList;
#[cfg(feature = "eviction_stacks_pmem")]
use super::pmem_collections::PmemHashList;

// Eviction-stack metadata is allocated through the same crate-wide `Hybrid`
// alias (`HybridObjects`, UMF/TBB, NUMA node 1) that `BufferPMEM`/other PMEM
// features already use -- previously routed through a separate,
// jemalloc_cxl-backed `EvictionStackAllocator`, removed for depending on an
// allocator with no stability track record under real concurrent load (see
// `jemalloc_cxl_slow_tier`'s removal notes in `CLAUDE.md`).
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

// The insertion-ordered queue and per-key map are DRAM-backed by default.
// When `eviction_stacks_pmem` is enabled, they are instead allocated in the
// slow tier (PMEM, via `crate::Hybrid`) — co-located with the slow-tier
// object bytes — exactly the way `LruHybridStack` switches to PMEM
// collections under that flag. The method surface of the PMEM
// `PmemHashList`/`hashbrown::HashMap` variants matches the DRAM
// `HashList`/`std::collections::HashMap` ones used below, so the stack logic
// itself is identical for both backings. Only the transient `migrations`
// scratch and the scalar counters stay in DRAM.
#[cfg(not(feature = "eviction_stacks_pmem"))]
type QueueList = HashList<HashedKey, NoHasher>;
#[cfg(feature = "eviction_stacks_pmem")]
type QueueList = PmemHashList<HashedKey, NoHasher>;

/// Combined per-key bookkeeping: tier and size. See the module doc's "One
/// combined per-key map" section for why this is a single map.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct FifoEntry {
	tier: Tier,
	size: ObjectSize,
}

#[cfg(not(feature = "eviction_stacks_pmem"))]
type EntryMap = HashMap<HashedKey, FifoEntry, NoHasher>;
#[cfg(feature = "eviction_stacks_pmem")]
type EntryMap = HashMap<HashedKey, FifoEntry, NoHasher, Hybrid>;

pub struct FifoHybridStack {
	/// Insertion order: front = newest admission, back = oldest.
	queue: QueueList,
	entries: EntryMap,

	fast_capacity: CacheSize,
	fast_used: CacheSize,
	slow_used: CacheSize,

	/// Number of keys currently tagged `Tier::Fast`. Kept alongside
	/// `fast_used` (bytes) so `fast_object_count`/`slow_object_count` don't
	/// need an O(n) scan over `entries`.
	fast_count: usize,

	/// The oldest key currently tagged `Tier::Fast` — i.e. the next
	/// candidate for demotion. `None` iff no key is currently Fast. Because
	/// the fast tier is always a contiguous prefix of `queue` (starting from
	/// the head), this single key is enough to find the demotion candidate
	/// in O(1) instead of scanning the list.
	fast_boundary: Option<HashedKey>,

	/// (key, new tier) pairs recorded since the last `drain_tier_migrations`.
	migrations: Vec<(HashedKey, Tier)>,
}

impl FifoHybridStack {
	/// Constructs the (queue, entry map) pair, DRAM- or PMEM-backed
	/// depending on `eviction_stacks_pmem`.
	#[cfg(not(feature = "eviction_stacks_pmem"))]
	fn new_collections() -> (QueueList, EntryMap) {
		(HashList::default(), HashMap::default())
	}

	#[cfg(feature = "eviction_stacks_pmem")]
	fn new_collections() -> (QueueList, EntryMap) {
		(
			PmemHashList::with_hasher(NoHasher::default()),
			HashMap::with_hasher_in(NoHasher::default(), Hybrid),
		)
	}

	pub fn new(fast_capacity: CacheSize) -> Self {
		let (queue, entries) = Self::new_collections();

		FifoHybridStack {
			queue,
			entries,

			fast_capacity,
			fast_used: 0,
			slow_used: 0,
			fast_count: 0,

			fast_boundary: None,
			migrations: Vec::new(),
		}
	}

	/// The configured fast-tier byte budget.
	pub fn fast_capacity(&self) -> CacheSize {
		self.fast_capacity
	}

	/// Returns the tier the given (currently tracked) key is in, or `None`
	/// if the key isn't tracked. Exposed for tests/diagnostics.
	pub fn tier_of(&self, key: HashedKey) -> Option<Tier> {
		self.entries.get(&key).map(|entry| entry.tier)
	}

	/// Records a size change for an already-tracked key without altering its
	/// tier or position, adjusting whichever tier's used-bytes counter
	/// currently applies.
	fn resize_key(&mut self, key: HashedKey, new_size: ObjectSize) {
		let Some(entry) = self.entries.get_mut(&key) else { return };

		let old_size = entry.size;
		entry.size = new_size;
		let delta = new_size as i64 - old_size as i64;

		match entry.tier {
			Tier::Fast => {
				self.fast_used = (self.fast_used as i64 + delta).max(0) as CacheSize;
			},

			Tier::Slow => {
				self.slow_used = (self.slow_used as i64 + delta).max(0) as CacheSize;
			},
		}
	}

	/// Demotes the oldest fast key(s), triggered once `fast_used` exceeds
	/// `fast_capacity`, drained back down to exactly `fast_capacity` (no
	/// low-water headroom — see the module doc's "No shared-DRAM-overhead
	/// reservation or low-water headroom" section for why). Demotion is the
	/// only response; capacity here never evicts (terminal eviction stays
	/// governed solely by `max_size`, handled by `PolicyWorker::apply_evictions`).
	fn settle_fast_tier(&mut self) {
		if self.fast_used <= self.fast_capacity {
			return;
		}

		while self.fast_used > self.fast_capacity {
			let Some(demote_key) = self.fast_boundary else { break };

			let size = self.entries.get(&demote_key).map(|entry| entry.size).unwrap_or(0) as CacheSize;
			let new_boundary = self.queue.before(&demote_key).copied();

			if let Some(entry) = self.entries.get_mut(&demote_key) {
				entry.tier = Tier::Slow;
			}

			self.fast_used = self.fast_used.saturating_sub(size);
			self.fast_count = self.fast_count.saturating_sub(1);
			self.slow_used += size;
			self.fast_boundary = new_boundary;

			self.migrations.push((demote_key, Tier::Slow));
		}
	}
}

impl PolicyStack for FifoHybridStack {
	fn is_policy(&self, policy: &PaperPolicy) -> bool {
		matches!(policy, PaperPolicy::FifoHybrid)
	}

	fn len(&self) -> usize {
		self.queue.len()
	}

	fn contains(&self, key: HashedKey) -> bool {
		self.queue.contains(&key)
	}

	fn insert(&mut self, key: HashedKey, size: ObjectSize) {
		if let Some(&FifoEntry { tier, size: old_size }) = self.entries.get(&key) {
			// Existing key: FIFO has no promotion/reordering at all — a
			// `set()` overwrite must never move this key's position in
			// `queue` and must never change its tier. Only correct
			// whichever tier's byte accounting applies if the size changed.
			if old_size != size {
				self.resize_key(key, size);

				// A larger value now resident in Fast can itself push
				// fast_used over budget, the same way a fresh admission
				// would — but note settle_fast_tier only ever demotes the
				// current fast_boundary (the oldest Fast key), which may or
				// may not be this key; this key itself never moves as a
				// *direct* effect of this branch.
				if tier == Tier::Fast {
					self.settle_fast_tier();
				}
			}

			return;
		}

		// Brand-new key: admitted at the bottom of the fast tier (newest
		// end of the queue), per the paper's admission rule.
		self.queue.push_front(key);
		self.entries.insert(key, FifoEntry { tier: Tier::Fast, size });
		self.fast_used += size as CacheSize;
		self.fast_count += 1;

		if self.fast_boundary.is_none() {
			self.fast_boundary = Some(key);
		}

		self.settle_fast_tier();
	}

	// `update()` is deliberately NOT overridden here — it stays the
	// `PolicyStack` trait's default no-op body. See the module doc's "No
	// promotion, ever" section: a cache `get()` hit must never reorder this
	// key in `queue` or change its tier. This is the single most
	// load-bearing design decision in this file — do not "fix" this into
	// doing something.

	fn remove(&mut self, key: HashedKey) {
		let entry = self.entries.remove(&key);
		let size = entry.map(|entry| entry.size).unwrap_or(0) as CacheSize;
		let tier = entry.map(|entry| entry.tier);

		let new_boundary_if_needed = if tier == Some(Tier::Fast) && self.fast_boundary == Some(key) {
			self.queue.before(&key).copied()
		} else {
			None
		};

		self.queue.remove(&key);

		match tier {
			Some(Tier::Fast) => {
				self.fast_used = self.fast_used.saturating_sub(size);
				self.fast_count = self.fast_count.saturating_sub(1);

				if self.fast_boundary == Some(key) {
					self.fast_boundary = new_boundary_if_needed;
				}
			},

			Some(Tier::Slow) => {
				self.slow_used = self.slow_used.saturating_sub(size);
			},

			None => {},
		}
	}

	fn clear(&mut self) {
		self.queue.clear();
		self.entries.clear();

		self.fast_used = 0;
		self.slow_used = 0;
		self.fast_count = 0;
		self.fast_boundary = None;
		self.migrations.clear();
	}

	fn evict_one(&mut self) -> Option<HashedKey> {
		let key = self.queue.pop_back()?;
		let entry = self.entries.remove(&key);
		let size = entry.map(|entry| entry.size).unwrap_or(0) as CacheSize;

		match entry.map(|entry| entry.tier) {
			Some(Tier::Fast) => {
				self.fast_used = self.fast_used.saturating_sub(size);
				self.fast_count = self.fast_count.saturating_sub(1);

				// The tail of the whole list can only be Fast-tagged if
				// every tracked key is still Fast (no demotion has ever
				// happened), in which case the boundary must have equaled
				// this key too. The new tail, if any, is then still Fast.
				if self.fast_boundary == Some(key) {
					self.fast_boundary = self.queue.back().copied();
				}
			},

			Some(Tier::Slow) => {
				self.slow_used = self.slow_used.saturating_sub(size);
			},

			None => {},
		}

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
		self.slow_used
	}

	fn fast_object_count(&self) -> usize {
		self.fast_count
	}

	fn slow_object_count(&self) -> usize {
		self.entries.len() - self.fast_count
	}
}

#[cfg(test)]
mod tests {
	use super::*;

	fn drain(stack: &mut FifoHybridStack) -> Vec<(HashedKey, Tier)> {
		stack.drain_tier_migrations()
	}

	#[test]
	fn evict_one_terminates_when_both_keys_are_immediately_demoted() {
		// Reproduces the `apply_evictions` scenario where fast_capacity is
		// tiny relative to object sizes, so *both* inserted keys demote to
		// slow immediately (fast_boundary bounces to None each time).
		let mut stack = FifoHybridStack::new(4);

		stack.insert(1, 19);
		stack.insert(2, 19);
		drain(&mut stack);

		assert_eq!(stack.tier_of(1), Some(Tier::Slow));
		assert_eq!(stack.tier_of(2), Some(Tier::Slow));
		assert_eq!(stack.len(), 2);

		assert_eq!(stack.evict_one(), Some(1));
		assert_eq!(stack.len(), 1);
		assert_eq!(stack.evict_one(), Some(2));
		assert_eq!(stack.len(), 0);
		assert_eq!(stack.evict_one(), None);
	}

	#[test]
	fn admission_always_lands_fast() {
		let mut stack = FifoHybridStack::new(1_000);

		stack.insert(1, 10);
		stack.insert(2, 10);

		assert_eq!(stack.tier_of(1), Some(Tier::Fast));
		assert_eq!(stack.tier_of(2), Some(Tier::Fast));
		assert_eq!(stack.fast_bytes_used(), 20);
		assert_eq!(stack.slow_bytes_used(), 0);
	}

	#[test]
	fn fast_tier_pressure_demotes_oldest_tail() {
		let mut stack = FifoHybridStack::new(25);

		stack.insert(1, 10); // fast: [1]
		stack.insert(2, 10); // fast: [2, 1]
		drain(&mut stack);

		stack.insert(3, 10); // pushes fast_used to 30 > 25 -> demotes key 1 (oldest)
		let migrations = drain(&mut stack);

		assert_eq!(migrations, vec![(1, Tier::Slow)]);
		assert_eq!(stack.tier_of(1), Some(Tier::Slow));
		assert_eq!(stack.tier_of(2), Some(Tier::Fast));
		assert_eq!(stack.tier_of(3), Some(Tier::Fast));
		assert_eq!(stack.fast_bytes_used(), 20);
		assert_eq!(stack.slow_bytes_used(), 10);
	}

	#[test]
	fn hit_on_slow_key_does_not_migrate_or_reorder() {
		let mut stack = FifoHybridStack::new(25);

		stack.insert(1, 10);
		stack.insert(2, 10);
		stack.insert(3, 10); // demotes 1 -> slow
		drain(&mut stack);

		assert_eq!(stack.tier_of(1), Some(Tier::Slow));

		// A hit (record_access / update) on a slow key must be a total
		// no-op: no migration, no reorder, no tier change -- this is the
		// defining difference from `LruHybridStack`'s equivalent test, where
		// the same setup would promote key 1 back to fast.
		stack.record_access(1, true);
		let migrations = drain(&mut stack);

		assert_eq!(migrations, Vec::new());
		assert_eq!(stack.tier_of(1), Some(Tier::Slow));
		assert_eq!(stack.tier_of(2), Some(Tier::Fast));
		assert_eq!(stack.tier_of(3), Some(Tier::Fast));
	}

	#[test]
	fn overwriting_an_existing_key_does_not_reposition_it_in_the_queue() {
		let mut stack = FifoHybridStack::new(1_000);

		stack.insert(1, 10); // oldest
		stack.insert(2, 10);
		stack.insert(3, 10); // newest

		// Under LRU semantics, re-inserting key 1 here would move it to the
		// front (MRU), making key 2 the next demotion candidate. Under FIFO,
		// key 1 must remain the oldest despite being freshly re-set.
		stack.insert(1, 15);
		assert_eq!(stack.tier_of(1), Some(Tier::Fast));
		assert_eq!(stack.fast_bytes_used(), 10 + 10 + 15);

		// key 1 is still the oldest by insertion order, not key 3.
		assert_eq!(stack.evict_one(), Some(1));
	}

	#[test]
	fn object_counts_track_tier_membership() {
		let mut stack = FifoHybridStack::new(15);

		stack.insert(1, 10); // fast
		stack.insert(2, 10); // demotes 1 -> slow, fast holds 2
		drain(&mut stack);

		assert_eq!(stack.fast_object_count(), 1);
		assert_eq!(stack.slow_object_count(), 1);

		stack.remove(1);
		assert_eq!(stack.fast_object_count(), 1);
		assert_eq!(stack.slow_object_count(), 0);
	}

	#[test]
	fn evict_one_only_removes_from_slow_tier_once_demotions_have_happened() {
		let mut stack = FifoHybridStack::new(15);

		stack.insert(1, 10); // fast
		stack.insert(2, 10); // demotes 1 -> slow, fast holds 2
		drain(&mut stack);

		assert_eq!(stack.tier_of(1), Some(Tier::Slow));
		assert_eq!(stack.tier_of(2), Some(Tier::Fast));

		// Tail of the whole list is the slow key (1), since it was demoted
		// and is the oldest.
		assert_eq!(stack.evict_one(), Some(1));
		assert_eq!(stack.tier_of(2), Some(Tier::Fast));
		assert_eq!(stack.slow_bytes_used(), 0);
	}

	#[test]
	fn evict_one_falls_back_to_fast_tail_when_everything_still_fits() {
		let mut stack = FifoHybridStack::new(1_000);

		stack.insert(1, 10);
		stack.insert(2, 10);
		drain(&mut stack);

		// Nothing has ever been demoted; the whole list is Fast.
		assert_eq!(stack.evict_one(), Some(1));
		assert_eq!(stack.tier_of(2), Some(Tier::Fast));
		assert_eq!(stack.fast_bytes_used(), 10);
	}

	#[test]
	fn zero_fast_capacity_demotes_immediately() {
		let mut stack = FifoHybridStack::new(0);

		stack.insert(1, 10);
		let migrations = drain(&mut stack);

		assert_eq!(migrations, vec![(1, Tier::Slow)]);
		assert_eq!(stack.tier_of(1), Some(Tier::Slow));
		assert_eq!(stack.fast_bytes_used(), 0);
		assert_eq!(stack.slow_bytes_used(), 10);
	}

	#[test]
	fn resizing_an_existing_key_adjusts_the_correct_tier_counter() {
		let mut stack = FifoHybridStack::new(1_000);

		stack.insert(1, 10);
		drain(&mut stack);
		assert_eq!(stack.fast_bytes_used(), 10);

		// re-`set()` with a larger value: still fast, counter adjusted, no
		// reposition.
		stack.insert(1, 30);
		assert_eq!(stack.tier_of(1), Some(Tier::Fast));
		assert_eq!(stack.fast_bytes_used(), 30);
	}

	#[test]
	fn shrinking_fast_tier_at_runtime_triggers_demotions() {
		let mut stack = FifoHybridStack::new(1_000);

		stack.insert(1, 100);
		stack.insert(2, 100);
		drain(&mut stack);

		// fast_capacity shrinks to 150; fast_used starts at 200; demoting
		// the oldest key (1, 100 bytes) alone lands at 100, under 150 (no
		// low-water headroom here, so it stops as soon as it's under budget).
		stack.resize_fast_tier(150);
		let migrations = drain(&mut stack);

		assert_eq!(migrations, vec![(1, Tier::Slow)]);
		assert_eq!(stack.tier_of(2), Some(Tier::Fast));
		assert_eq!(stack.fast_bytes_used(), 100);
	}

	#[test]
	fn remove_updates_boundary_and_counters() {
		let mut stack = FifoHybridStack::new(1_000);

		stack.insert(1, 10);
		stack.insert(2, 10);
		drain(&mut stack);

		stack.remove(2);
		assert_eq!(stack.contains(2), false);
		assert_eq!(stack.fast_bytes_used(), 10);

		// The remaining fast key (1) must still be demotable correctly.
		stack.resize_fast_tier(0);
		let migrations = drain(&mut stack);
		assert_eq!(migrations, vec![(1, Tier::Slow)]);
	}

	#[test]
	fn clear_resets_all_state() {
		let mut stack = FifoHybridStack::new(15);

		stack.insert(1, 10);
		stack.insert(2, 10);
		drain(&mut stack);

		stack.clear();

		assert_eq!(stack.len(), 0);
		assert_eq!(stack.fast_bytes_used(), 0);
		assert_eq!(stack.slow_bytes_used(), 0);
		assert_eq!(stack.tier_of(1), None);
		assert_eq!(stack.evict_one(), None);
	}
}
