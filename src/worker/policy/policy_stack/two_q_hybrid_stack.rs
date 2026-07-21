/*
 * Copyright (c) Kia Shakiba
 *
 * This source code is licensed under the GNU AGPLv3 license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! `TwoQHybridStack` — a segmented 2Q stack for `PaperPolicy::TwoQHybrid`.
//!
//! Two live queues, matching the paper text directly (unlike this crate's
//! plain `TwoQStack`, which has a heavier three-*live*-queue shape with a
//! real-object `a1_out` overflow queue): `fifo_queue`, a one-access FIFO
//! queue holding real objects that is always entirely in the slow tier, and
//! `main_stack`, a recency-ordered LRU queue segmented fast/slow exactly
//! like `LruHybridStack::stack`.
//!
//! Admission always lands in `fifo_queue` (byte-capped at `fifo_capacity =
//! k_in * max_size`, mirroring how plain `TwoQStack` sizes `a1_in`). A hit
//! on a `fifo_queue`-resident key promotes it straight to the top of
//! `main_stack` at `Tier::Fast`. A `fifo_queue` object that ages out
//! without a second access is evicted outright — no ghost/re-admission
//! memory is kept (see `CLAUDE.md`'s `two_q_hybrid_cache` section for why:
//! an exact-membership ghost check on every admission was flagged as an
//! unwelcome cost given every admission here already pays a synchronous
//! slow-tier/PMEM write; a probabilistic structure is the right tool to
//! revisit this and is left as future work).
//!
//! Note this stack never evicts on its own: `insert`/`resize` only update
//! `fifo_used`, and `needs_capacity_eviction` reports when it has exceeded
//! `fifo_capacity` — the caller (`PolicyWorker::apply_evictions`) is the
//! one that actually removes the object, via the same `evict_one()` +
//! `erase()` pairing it already uses for overall-`max_size` pressure (see
//! `evict_fifo_tail`'s doc comment for why: a `PolicyStack` has no
//! reference to the shared object map, so it cannot safely evict on its
//! own).
//! Once inside `main_stack`, an object behaves exactly like
//! `LruHybridStack`: a fast-tier hit just reorders; a slow-tier hit
//! promotes (and may cascade a demotion); fast-tier pressure demotes the
//! LRU tail down to the slow tier.
//!
//! `fifo_capacity` (sized by the policy-embedded `k_in`, fixed at
//! construction, rescaled on `resize()`) and `fast_capacity` (the main
//! queue's fast/slow split, via `fast_tier_size`/`set_fast_tier_size`,
//! freely adjustable at runtime) are two independent sizing knobs.
//!
//! Eviction priority: `fifo_queue`'s tail first, then `main_stack`'s slow
//! tail, falling back to `main_stack`'s fast tail only if nothing has ever
//! been demoted there yet (same fallback `LruHybridStack::evict_one` has).
//! This reconciles the paper's two eviction clauses into one rule:
//! sacrificing still-unproven FIFO objects before ever touching the proven
//! main queue reproduces both stated behaviors.
//!
//! ## One combined per-key map, not three
//!
//! Every tracked key needs a queue tag (`Fifo`/`Main`), a size, and — only
//! while in `Main` — a tier. An earlier version of this stack tracked these
//! in three separate maps (`queue`, `main_tiers`, `sizes`), mirroring how it
//! was originally built by extending `LruHybridStack`'s (`tiers`+`sizes`)
//! shape with a third map bolted on for the extra Fifo/Main dimension.
//! Checking every call site showed no operation ever wants just one of these
//! in isolation — `insert` touches queue+size together, `remove` touches all
//! three, etc. — so they're now one `entries: HashMap<HashedKey, TwoQEntry>`
//! (`TwoQEntry { queue, tier: Option<Tier>, size }`, `tier: None` iff
//! `queue == Fifo`). This eliminates two of the three hashtable-structural
//! overhead charges per tracked object (see `object/overhead.rs`'s
//! `TwoQHybrid` arm) and removes an entire class of possible desync bug
//! (a key present in one map but not another) by construction, since there
//! is now only one map for a key to be present or absent from. The one
//! `Some -> None` case, `main_tiers.len()` (used by `slow_object_count`), no
//! longer comes for free from the map itself, so a `main_count` counter
//! tracks it explicitly, mirroring the existing `fast_count` pattern.
//!
//! ## `eviction_stacks_pmem`
//!
//! Both live queues (`fifo_queue`, `main_stack`) and the combined per-key
//! `entries` map are DRAM-backed by default. When `eviction_stacks_pmem` is
//! enabled, they are instead allocated in the slow tier (PMEM, via
//! `crate::Hybrid`) — the same switch `LruHybridStack`/`LfuHybridStack` make
//! under this flag. The `PmemHashList`/`hashbrown::HashMap` variants expose
//! the same method surface as the DRAM `HashList`/`std::collections::
//! HashMap` ones used below, so the stack logic itself is identical for both
//! backings; only the transient `migrations` scratch and the scalar
//! counters stay in DRAM.

#[cfg(not(feature = "eviction_stacks_pmem"))]
use std::collections::HashMap;
#[cfg(feature = "eviction_stacks_pmem")]
use hashbrown::HashMap;

#[cfg(not(feature = "eviction_stacks_pmem"))]
use kwik::collections::HashList;
#[cfg(feature = "eviction_stacks_pmem")]
use super::pmem_collections::PmemHashList;

// `Hybrid` here is `crate::allocator::EvictionStackAllocator` (jemalloc_cxl's
// CXL/NUMA arena mechanism) -- a different type from the crate-level
// `Hybrid` alias used by `BufferPMEM`/other PMEM features. Kept under this
// local name only to minimize the diff against the call sites below.
#[cfg(feature = "eviction_stacks_pmem")]
use crate::allocator::EvictionStackAllocator as Hybrid;

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

/// Combined per-key bookkeeping: which queue, which tier (only meaningful
/// while `queue == Main`), and the object's size. See the module doc's "One
/// combined per-key map" section for why this replaced three separate maps.
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

pub struct TwoQHybridStack {
	fifo_queue: QueueList,
	main_stack: QueueList,

	entries: EntryMap,

	k_in: f64,
	fifo_capacity: CacheSize,
	fifo_used: CacheSize,

	fast_capacity: CacheSize,
	fast_used: CacheSize,
	slow_used: CacheSize,

	/// Number of keys currently tagged `Tier::Fast` within `main_stack`.
	/// Kept alongside `fast_used` so `fast_object_count`/`slow_object_count`
	/// don't need an O(n) scan over `entries` — mirrors
	/// `LruHybridStack::fast_count`.
	fast_count: usize,

	/// Number of keys currently in the `Main` queue (Fast or Slow). Kept
	/// explicitly since `entries.len()` now covers *both* queues; before the
	/// three-maps-to-one consolidation this was `main_tiers.len()`, free
	/// from that map's own length.
	main_count: usize,

	/// The least-recently-used key currently tagged `Tier::Fast` within
	/// `main_stack` — i.e. the next demotion candidate. `None` iff no key in
	/// `main_stack` is currently Fast. Mirrors `LruHybridStack::fast_boundary`.
	main_boundary: Option<HashedKey>,

	/// (key, new tier) pairs recorded since the last `drain_tier_migrations`.
	migrations: Vec<(HashedKey, Tier)>,
}

impl TwoQHybridStack {
	/// Constructs the (fifo list, main list, entry map) triple, DRAM- or
	/// PMEM-backed depending on `eviction_stacks_pmem`.
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

	pub fn new(k_in: f64, max_size: CacheSize, fast_capacity: CacheSize) -> Self {
		let (fifo_queue, main_stack, entries) = Self::new_collections();

		TwoQHybridStack {
			fifo_queue,
			main_stack,

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

	/// Records a size change for an already-tracked key without altering its
	/// queue/tier, adjusting whichever counter currently applies.
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

	/// Treats an already-tracked key as accessed: a `Fifo` key promotes
	/// straight to `Main`+`Fast`; a `Main` key is handled by
	/// `touch_main_fast` (reorder if already Fast, promote if Slow).
	fn touch(&mut self, key: HashedKey) {
		match self.entries.get(&key).map(|entry| entry.queue) {
			Some(Queue::Fifo) => self.promote_from_fifo(key),
			Some(Queue::Main) => self.touch_main_fast(key),
			None => {},
		}
	}

	/// Moves a `fifo_queue`-resident key to the front of `main_stack`,
	/// tagging it `Tier::Fast`. A brand-new entry into `main_stack`, so no
	/// `before`/boundary bookkeeping is needed beyond setting `main_boundary`
	/// if this is the first Fast key.
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

		// Pushed *after* `settle_fast_tier` (which pushes any demotions this
		// promotion itself triggered), not before: `apply_tier_migrations`
		// applies a stack's migrations in push order, so pushing the
		// promotion first would apply its DRAM allocation before the
		// corresponding demotion's DRAM free -- a transient window with both
		// copies resident. Guarded on the key still being `Fast`: an
		// extremely tight budget can demote it straight back out within the
		// same `settle_fast_tier` call (self-eviction), in which case that
		// call already pushed the correct final `(key, Tier::Slow)` entry.
		if self.entries.get(&key).and_then(|entry| entry.tier) == Some(Tier::Fast) {
			self.migrations.push((key, Tier::Fast));
		}
	}

	/// Moves an already-`Main`-tracked key to the front of `main_stack`,
	/// promoting it to `Tier::Fast` if it was `Slow`, then settles the fast
	/// tier. Mirrors `LruHybridStack::touch_fast_key` exactly, scoped to
	/// `main_stack`.
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

		// See `promote_from_fifo`'s doc for why this is pushed after
		// `settle_fast_tier` and guarded on the key still being `Fast`.
		if promoted && self.entries.get(&key).and_then(|entry| entry.tier) == Some(Tier::Fast) {
			self.migrations.push((key, Tier::Fast));
		}
	}

	/// Demotes the least-recently-used fast key(s) within `main_stack` until
	/// `fast_used` fits back within `fast_capacity`. Unlike
	/// `LruHybridStack`, drains to exactly `fast_capacity` (no low-water
	/// floor): fast-tier pressure here is only ever triggered by a
	/// promotion or an explicit `resize_fast_tier`, never by every `set()`.
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

	/// Pops and fully removes `fifo_queue`'s tail from this stack's own
	/// bookkeeping (the "reached the top without re-access" key), if any.
	/// Used by `evict_one`'s FIFO-first priority.
	///
	/// Deliberately **not** called from `insert`/`resize` to self-evict
	/// under `k_in`-driven `fifo_capacity` pressure: a `PolicyStack` has no
	/// reference to the shared object map or `status`, so it can only ever
	/// update its own bookkeeping here — it cannot actually remove the
	/// object from the cache or adjust accounted size. Doing so anyway
	/// would silently desync this stack's view of the world from the real
	/// object map (the object would linger forever, untracked). Real
	/// removal always has to go through `PolicyWorker::apply_evictions`'s
	/// `evict_one()` + `erase()` pairing, which is why `fifo_capacity`
	/// pressure is instead surfaced via `needs_capacity_eviction` below —
	/// `apply_evictions` polls that and keeps calling `evict_one()`
	/// (through the correct removal path) until it's satisfied.
	fn evict_fifo_tail(&mut self) -> Option<HashedKey> {
		let key = self.fifo_queue.pop_back()?;
		let size = self.entries.remove(&key).map(|entry| entry.size).unwrap_or(0) as CacheSize;

		self.fifo_used = self.fifo_used.saturating_sub(size);

		Some(key)
	}
}

impl PolicyStack for TwoQHybridStack {
	fn is_policy(&self, policy: &PaperPolicy) -> bool {
		matches!(policy, PaperPolicy::TwoQHybrid(k_in) if *k_in == self.k_in)
	}

	fn len(&self) -> usize {
		self.entries.len()
	}

	fn contains(&self, key: HashedKey) -> bool {
		self.entries.contains_key(&key)
	}

	fn insert(&mut self, key: HashedKey, size: ObjectSize) {
		if self.entries.contains_key(&key) {
			// Existing key: track any size change, then treat as an access.
			self.resize_key(key, size);
			self.touch(key);
			return;
		}

		// Brand-new key: always admitted into the FIFO queue, always slow.
		// If this pushes fifo_used over fifo_capacity, `needs_capacity_eviction`
		// will report it and `apply_evictions` will drain it via `evict_one`
		// (see that method's doc comment for why eviction can't happen here).
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
		// A shrink may push fifo_used over the new, smaller fifo_capacity;
		// `needs_capacity_eviction` reports it, `apply_evictions` drains it.
	}

	fn clear(&mut self) {
		self.fifo_queue.clear();
		self.main_stack.clear();
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

				// The tail of main_stack can only be Fast-tagged if every
				// tracked Main key is still Fast (no demotion has ever
				// happened), in which case the boundary must have equaled
				// this key too. The new tail, if any, is then still Fast.
				if self.main_boundary == Some(key) {
					self.main_boundary = self.main_stack.back().copied();
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

	fn drain(stack: &mut TwoQHybridStack) -> Vec<(HashedKey, Tier)> {
		stack.drain_tier_migrations()
	}

	#[test]
	fn admission_always_lands_in_fifo_queue_slow() {
		let mut stack = TwoQHybridStack::new(1.0, 1_000, 1_000);

		stack.insert(1, 10);
		stack.insert(2, 10);

		assert_eq!(stack.tier_of(1), Some(Tier::Slow));
		assert_eq!(stack.tier_of(2), Some(Tier::Slow));
		assert_eq!(stack.slow_bytes_used(), 20);
		assert_eq!(stack.fast_bytes_used(), 0);
		assert_eq!(drain(&mut stack), Vec::new());
	}

	#[test]
	fn reaccessing_a_fifo_key_promotes_it_to_fast() {
		let mut stack = TwoQHybridStack::new(1.0, 1_000, 1_000);

		stack.insert(1, 10);
		drain(&mut stack);

		stack.update(1);
		let migrations = drain(&mut stack);

		assert_eq!(migrations, vec![(1, Tier::Fast)]);
		assert_eq!(stack.tier_of(1), Some(Tier::Fast));
	}

	#[test]
	fn fifo_capacity_pressure_is_reported_not_self_evicted() {
		// k_in=1.0 against max_size=15 -> fifo_capacity fits exactly one
		// 10-byte object.
		let mut stack = TwoQHybridStack::new(1.0, 15, 1_000);

		stack.insert(1, 10);
		drain(&mut stack);
		assert_eq!(stack.contains(1), true);
		assert_eq!(stack.needs_capacity_eviction(), false);

		// New key exceeds fifo_capacity. The stack cannot evict on its own
		// (see `evict_fifo_tail`'s doc comment) -- both keys remain tracked,
		// and `needs_capacity_eviction` reports the pressure so the caller
		// (`apply_evictions`) drains it via the real `evict_one()` path.
		stack.insert(2, 10);
		let migrations = drain(&mut stack);

		assert_eq!(migrations, Vec::new());
		assert_eq!(stack.contains(1), true);
		assert_eq!(stack.contains(2), true);
		assert_eq!(stack.needs_capacity_eviction(), true);

		// Simulates what `apply_evictions` does when it observes
		// `needs_capacity_eviction() == true`: keep calling `evict_one()`
		// (the FIFO tail, key 1) until satisfied.
		assert_eq!(stack.evict_one(), Some(1));
		assert_eq!(stack.contains(1), false);
		assert_eq!(stack.needs_capacity_eviction(), false);
		assert_eq!(stack.tier_of(2), Some(Tier::Slow));
	}

	#[test]
	fn fast_tier_pressure_within_main_queue_demotes_lru_tail() {
		let mut stack = TwoQHybridStack::new(1.0, 1_000, 25);

		stack.insert(1, 10);
		stack.insert(2, 10);
		stack.update(1); // promote 1 -> Main/Fast
		stack.update(2); // promote 2 -> Main/Fast
		drain(&mut stack);

		assert_eq!(stack.fast_bytes_used(), 20);

		stack.insert(3, 10);
		stack.update(3); // promote 3 -> pushes fast_used to 30 > 25 -> demotes key 1 (LRU)
		let migrations = drain(&mut stack);

		assert!(migrations.iter().any(|(k, t)| *k == 1 && *t == Tier::Slow));
		assert_eq!(stack.tier_of(1), Some(Tier::Slow));
		assert_eq!(stack.tier_of(2), Some(Tier::Fast));
		assert_eq!(stack.tier_of(3), Some(Tier::Fast));
	}

	#[test]
	fn promotion_within_main_can_cascade_a_demotion() {
		let mut stack = TwoQHybridStack::new(1.0, 1_000, 25);

		stack.insert(1, 10);
		stack.insert(2, 10);
		stack.insert(3, 10);
		stack.update(1);
		stack.update(2);
		stack.update(3); // demotes 1 (fast_used 30 > 25)
		drain(&mut stack);

		assert_eq!(stack.tier_of(1), Some(Tier::Slow));

		// Accessing the slow key promotes it back to fast, which may itself
		// demote the new fast-tier LRU tail (key 2).
		stack.update(1);
		let migrations = drain(&mut stack);

		assert_eq!(stack.tier_of(1), Some(Tier::Fast));
		// Demotion is applied before the promotion that triggered it, so a
		// promotion never has its DRAM write applied before the
		// corresponding demotion's DRAM free (see `touch_main_fast`'s doc).
		assert_eq!(migrations, vec![(2, Tier::Slow), (1, Tier::Fast)]);
		assert_eq!(stack.tier_of(2), Some(Tier::Slow));
	}

	#[test]
	fn evict_one_prefers_fifo_queue_over_main_queue() {
		let mut stack = TwoQHybridStack::new(1.0, 1_000, 1_000);

		stack.insert(1, 10); // fifo
		stack.insert(2, 10);
		stack.update(2); // promote 2 -> Main/Fast
		drain(&mut stack);

		assert_eq!(stack.evict_one(), Some(1));
		assert_eq!(stack.tier_of(2), Some(Tier::Fast));
	}

	#[test]
	fn evict_one_falls_back_to_main_slow_then_main_fast() {
		let mut stack = TwoQHybridStack::new(1.0, 1_000, 25);

		stack.insert(1, 10);
		stack.insert(2, 10);
		stack.update(1);
		stack.update(2);
		drain(&mut stack);

		// fifo_queue empty; both keys are Main/Fast (nothing demoted yet).
		assert_eq!(stack.evict_one(), Some(1));
		assert_eq!(stack.tier_of(2), Some(Tier::Fast));
	}

	#[test]
	fn resize_rescales_fifo_capacity_and_reports_pressure() {
		let mut stack = TwoQHybridStack::new(0.5, 1_000, 1_000); // fifo_capacity = 500

		stack.insert(1, 100);
		stack.insert(2, 100);
		drain(&mut stack);
		assert_eq!(stack.slow_bytes_used(), 200);
		assert_eq!(stack.needs_capacity_eviction(), false);

		// Shrink overall max_size to 100 -> fifo_capacity = 50 -> both keys
		// now exceed it (200 > 50), reported via needs_capacity_eviction
		// rather than self-evicted (see `evict_fifo_tail`'s doc comment).
		stack.resize(100);

		assert_eq!(stack.contains(1), true);
		assert_eq!(stack.contains(2), true);
		assert_eq!(stack.needs_capacity_eviction(), true);

		assert_eq!(stack.evict_one(), Some(1));
		assert_eq!(stack.evict_one(), Some(2));
		assert_eq!(stack.needs_capacity_eviction(), false);
	}

	#[test]
	fn resize_fast_tier_shrink_triggers_demotions() {
		let mut stack = TwoQHybridStack::new(1.0, 1_000, 1_000);

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
		let mut stack = TwoQHybridStack::new(1.0, 1_000, 1_000);

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
