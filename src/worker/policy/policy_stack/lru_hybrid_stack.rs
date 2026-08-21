/*
 * Copyright (c) Kia Shakiba
 *
 * This source code is licensed under the GNU AGPLv3 license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! `LruHybridStack` — a single, segmented LRU queue for `PaperPolicy::LruHybrid`.
//!
//! One recency-ordered list backs both tiers. The fast tier is the maximal
//! prefix of the list (starting from the head/MRU end) whose cumulative byte
//! size fits within `fast_capacity`; everything else is the slow tier. Every
//! admission or access moves its key to the front and marks it `Tier::Fast`;
//! whenever that pushes `fast_used` over `fast_capacity`, the least recently
//! used fast key (tracked via `fast_boundary`, so no scan is needed) is
//! demoted. `evict_one` always pops the absolute LRU tail, which — once any
//! demotion has occurred — is always in the slow tier.
//!
//! ## High/low watermarks (shared with every other hybrid stack)
//!
//! `settle_fast_tier` triggers once `fast_used` exceeds
//! `watermarks::high_bytes` of the effective budget, and once triggered
//! drains down to `watermarks::low_bytes` of that budget rather than back to
//! exactly the ceiling. Both ratios live in `super::watermarks`
//! (`DEFAULT_HIGH` / `DEFAULT_LOW`, currently 0.98 / 0.95 -- those constants
//! are authoritative, not this sentence) and are shared by every hybrid
//! stack, so the
//! trigger/drain tradeoff is tuned in one place. Setting both to `1.0`
//! restores the original trigger-and-drain-at-the-ceiling behaviour exactly.
//!
//! The *effective budget* itself is unchanged by this: it is still
//! `fast_capacity` minus `reserved_overhead()`, and the watermarks are
//! applied on top of that value rather than in place of it.
//!
//! This supersedes the stack-local `FAST_TIER_LOW_WATER_RATIO` (0.98), which
//! is retained only as documentation of the previous value. An earlier
//! version of this stack had a
//! larger (90%) low-water floor, removed at the user's explicit request
//! ("keeping the 10% high water mark in the lru implementation hurts
//! performance so get rid of it") because idle headroom cost usable space
//! for no correctness benefit at the time. It was reintroduced, deliberately
//! smaller, for a *different* reason: `PaperCache::set()` writes a new
//! object's `TieredBuffer` to DRAM synchronously at the API layer, before
//! this stack (running on the background `PolicyWorker` thread) even sees
//! the corresponding event — so a burst of concurrent `set()` calls can
//! transiently push real DRAM usage above what the stack's own bookkeeping
//! shows, in the window between that physical write and the worker's next
//! pass. Draining slightly below the ceiling on each settle leaves that
//! burst some room to land in before the *next* settle needs to trigger
//! again. This headroom is not itself what bounds that window — see
//! `PolicyWorker::apply_tier_migrations`'s per-event (not per-batch)
//! scheduling for the change that actually shrinks it — it only reduces how
//! close to the edge the tier sits between settles, and is applied only to
//! `LruHybridStack` (not `LfuHybridStack`, which doesn't re-settle on every
//! admission the way this stack does).
//!
//! This stack only tracks *order and tier membership*; it does not move any
//! bytes itself. `PolicyWorker` drains `drain_tier_migrations` after each
//! `insert`/`update` call and performs the actual `TieredBuffer`
//! reallocation against the shared object map (see `Object::set_data`).
//!
//! ## One combined per-key map, not two
//!
//! Every tracked key needs both a tier and a size, and nearly every
//! operation here touches both together (`resize_key` adjusts whichever
//! tier's counter applies to the size change; `touch_fast_key`/
//! `settle_fast_tier`/`remove`/`evict_one` all read the size right after
//! reading or writing the tier). An earlier version kept these in two
//! separate maps (`sizes`, `tiers`); they're now one
//! `entries: HashMap<HashedKey, LruEntry>` (`LruEntry { tier, size }`),
//! matching `TwoQHybridStack`'s equivalent consolidation. This removes one
//! of the two hashtable-structural-overhead charges per tracked object (see
//! `object/overhead.rs`'s `LruHybrid` arm) and removes the possibility of a
//! key being present in one map but not the other by construction.

#[cfg(not(feature = "eviction_stacks_pmem"))]
use std::collections::HashMap;
#[cfg(feature = "eviction_stacks_pmem")]
use hashbrown::HashMap;

#[cfg(not(feature = "eviction_stacks_pmem"))]
use kwik::collections::HashList;
#[cfg(feature = "eviction_stacks_pmem")]
use super::pmem_collections::PmemHashList;

// Eviction-stack metadata is allocated through the same crate-wide `Hybrid`
// alias (`numa_alloc::SlowObjects`, node-1-bound jemalloc arenas) that
// `BufferPMEM` and the other PMEM features use, so the stacks land on the
// same node as the slow-tier values they index.
#[cfg(feature = "eviction_stacks_pmem")]
use crate::Hybrid;

use crate::{
	CacheSize,
	HashedKey,
	NoHasher,
	policy::PaperPolicy,
	object::ObjectSize,
	worker::policy::policy_stack::{PolicyStack, Tier, watermarks},
};

/// Superseded by the shared `super::watermarks` pair — `settle_fast_tier`
/// now drains to `watermarks::low_bytes` of the effective budget instead of
/// this stack-local ratio. Retained (unused) purely to document the value
/// this stack ran with before the watermarks were centralised: a 2% burst
/// safety margin, deliberately much smaller than the 90% low-water floor an
/// even earlier version used (and which was removed for hurting
/// performance), sized only to give concurrent `set()` calls some landing
/// room rather than to reduce demotion-pass frequency on its own. The shared
/// defaults are more aggressive on purpose; see `super::watermarks`.
#[allow(dead_code)]
const FAST_TIER_LOW_WATER_RATIO: f64 = 0.98;

// The recency list and per-key map are DRAM-backed by default. When
// `eviction_stacks_pmem` is enabled, they are instead allocated in the slow
// tier (PMEM, via `crate::Hybrid`) — co-located with the slow-tier object
// bytes — exactly the way plain `LruStack` switches to PMEM collections under
// that flag. The method surface of the PMEM `PmemHashList`/`hashbrown::HashMap`
// variants matches the DRAM `HashList`/`std::collections::HashMap` ones used
// below, so the stack logic itself is identical for both backings. Only the
// transient `migrations` scratch and the scalar counters stay in DRAM.
#[cfg(not(feature = "eviction_stacks_pmem"))]
type RecencyList = HashList<HashedKey, NoHasher>;
#[cfg(feature = "eviction_stacks_pmem")]
type RecencyList = PmemHashList<HashedKey, NoHasher>;

/// Combined per-key bookkeeping: tier and size. See the module doc's "One
/// combined per-key map" section for why this replaced two separate maps.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct LruEntry {
	tier: Tier,
	size: ObjectSize,
}

#[cfg(not(feature = "eviction_stacks_pmem"))]
type EntryMap = HashMap<HashedKey, LruEntry, NoHasher>;
#[cfg(feature = "eviction_stacks_pmem")]
type EntryMap = HashMap<HashedKey, LruEntry, NoHasher, Hybrid>;

pub struct LruHybridStack {
	stack: RecencyList,
	entries: EntryMap,

	fast_capacity: CacheSize,
	fast_used: CacheSize,
	slow_used: CacheSize,

	/// Approximate per-object DRAM cost of the shared structures (object
	/// hashtable + eviction stacks) that hold an entry for every object of
	/// both tiers. Reserved out of `fast_capacity` in `settle_fast_tier` so
	/// the fast-tier budget bounds total DRAM (values + shared metadata), not
	/// just fast-tier values. `0` unless set via `with_shared_overhead` (so
	/// unit tests exercising the pure value-budget behavior are unaffected).
	shared_overhead: CacheSize,

	/// Number of keys currently tagged `Tier::Fast`. Kept alongside
	/// `fast_used` (bytes) so `fast_object_count`/`slow_object_count` don't
	/// need an O(n) scan over `entries`.
	fast_count: usize,

	/// The least-recently-used key currently tagged `Tier::Fast` — i.e. the
	/// next candidate for demotion. `None` iff no key is currently Fast.
	/// Because the fast tier is always a contiguous prefix of `stack`
	/// (starting from the head), this single key is enough to find the
	/// demotion candidate in O(1) instead of scanning the list.
	fast_boundary: Option<HashedKey>,

	/// (key, new tier) pairs recorded since the last `drain_tier_migrations`.
	migrations: Vec<(HashedKey, Tier)>,
}

impl LruHybridStack {
	/// Constructs the (recency list, entry map) pair, DRAM- or PMEM-backed
	/// depending on `eviction_stacks_pmem`.
	#[cfg(not(feature = "eviction_stacks_pmem"))]
	fn new_collections() -> (RecencyList, EntryMap) {
		(HashList::default(), HashMap::default())
	}

	#[cfg(feature = "eviction_stacks_pmem")]
	fn new_collections() -> (RecencyList, EntryMap) {
		(
			PmemHashList::with_hasher(NoHasher::default()),
			HashMap::with_hasher_in(NoHasher::default(), Hybrid),
		)
	}

	pub fn new(fast_capacity: CacheSize) -> Self {
		let (stack, entries) = Self::new_collections();

		LruHybridStack {
			stack,
			entries,

			fast_capacity,
			fast_used: 0,
			slow_used: 0,
			shared_overhead: 0,
			fast_count: 0,

			fast_boundary: None,
			migrations: Vec::new(),
		}
	}

	/// Sets the approximate per-object shared-structure DRAM overhead (object
	/// hashtable + eviction stacks) reserved out of the fast-tier budget. See
	/// `crate::object::overhead::get_hybrid_dram_shared_overhead`. Builder-style
	/// so `init_policy_stack` can wire it in without disturbing `new`'s
	/// signature (unit tests keep the default `0`).
	pub fn with_shared_overhead(mut self, overhead: CacheSize) -> Self {
		self.shared_overhead = overhead;
		self
	}

	/// The configured fast-tier byte budget.
	pub fn fast_capacity(&self) -> CacheSize {
		self.fast_capacity
	}

	/// Total DRAM currently reserved for shared per-object metadata across
	/// both tiers (`tracked object count × shared_overhead`). Subtracted from
	/// `fast_capacity` to form the effective value-byte budget in
	/// `settle_fast_tier`.
	fn reserved_overhead(&self) -> CacheSize {
		self.stack.len() as CacheSize * self.shared_overhead
	}

	/// Returns the tier the given (currently tracked) key is in, or `None`
	/// if the key isn't tracked. Exposed for tests/diagnostics.
	pub fn tier_of(&self, key: HashedKey) -> Option<Tier> {
		self.entries.get(&key).map(|entry| entry.tier)
	}

	/// Records a size change for an already-tracked key without altering its
	/// tier, adjusting whichever tier's used-bytes counter currently applies.
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

	/// Moves an already-tracked key to the front of the recency list,
	/// promoting it to `Tier::Fast` if it was in the slow tier, then settles
	/// the fast tier (demoting as needed). Used by both `insert` (on an
	/// existing key — a `set()` always re-admits to the fast tier) and
	/// `update` (a fast-or-slow hit).
	fn touch_fast_key(&mut self, key: HashedKey) {
		let previous_tier = self.entries.get(&key).map(|entry| entry.tier);

		let already_at_front = self.stack.front() == Some(&key);
		let is_boundary = self.fast_boundary == Some(key);

		// If this key is the current fast-tier boundary and is about to
		// move away from its spot, the key immediately ahead of it (toward
		// the head) becomes the new boundary once it moves. Must be
		// captured *before* the move, since `before` reads current list
		// structure.
		let new_boundary_if_moved = if is_boundary && !already_at_front {
			self.stack.before(&key).copied()
		} else {
			None
		};

		self.stack.move_front(&key);

		if is_boundary && !already_at_front {
			self.fast_boundary = new_boundary_if_moved;
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
				entry.tier = Tier::Fast;
			}

			if self.fast_boundary.is_none() {
				self.fast_boundary = Some(key);
			}
		}

		self.settle_fast_tier();

		// Pushed *after* `settle_fast_tier` (which pushes any demotions this
		// promotion itself triggered) rather than before: `apply_tier_
		// migrations` applies a stack's migrations in order, physically
		// reallocating each one, so pushing the promotion first meant a
		// promotion that pushed `fast_used` over budget had its DRAM
		// allocation applied *before* the corresponding demotion's DRAM
		// free -- a real, reported transient window where both the
		// promoted object's new DRAM copy and a not-yet-demoted victim's
		// old DRAM copy were resident simultaneously. Guarded on the key
		// still being `Fast` afterward: an extremely tight budget can demote
		// this same key straight back out within the same `settle_fast_tier`
		// call (self-eviction, e.g. a fast tier that fits nothing) -- in
		// that case `settle_fast_tier` already pushed the correct final
		// `(key, Tier::Slow)` entry and no separate `Fast` entry should
		// follow it.
		if promoted && self.entries.get(&key).map(|entry| entry.tier) == Some(Tier::Fast) {
			self.migrations.push((key, Tier::Fast));
		}
	}

	/// Demotes the least-recently-used fast key(s), *triggered* once
	/// `fast_used` exceeds `watermarks::high_bytes` of the *effective* value
	/// budget (`fast_capacity` minus the DRAM reserved for shared per-object
	/// metadata — hashtable + eviction stacks — across both tiers; this makes
	/// the fast-tier budget bound total DRAM, not just fast-tier values,
	/// saturating to 0 when the shared metadata alone meets/exceeds
	/// `fast_capacity`), but *drained* down to `watermarks::low_bytes` of that
	/// same effective budget rather than back to the ceiling exactly — see the
	/// module doc's "High/low watermarks" section, and `super::watermarks` for
	/// the ratios and their env overrides. Demotion is the only response; the
	/// DRAM budget never evicts (terminal eviction stays governed solely by
	/// `max_size`).
	fn settle_fast_tier(&mut self) {
		// The effective budget is unchanged — capacity minus the shared
		// per-object metadata reservation. The watermarks are applied *on top
		// of* that value, never in place of it.
		let effective = self.fast_capacity.saturating_sub(self.reserved_overhead());

		if self.fast_used <= watermarks::high_bytes(effective) {
			return;
		}

		let drain_target = watermarks::low_bytes(effective);

		while self.fast_used > drain_target {
			let Some(demote_key) = self.fast_boundary else { break };

			let size = self.entries.get(&demote_key).map(|entry| entry.size).unwrap_or(0) as CacheSize;
			let new_boundary = self.stack.before(&demote_key).copied();

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

impl PolicyStack for LruHybridStack {
	fn is_policy(&self, policy: &PaperPolicy) -> bool {
		matches!(policy, PaperPolicy::LruHybrid)
	}

	fn len(&self) -> usize {
		self.stack.len()
	}

	fn contains(&self, key: HashedKey) -> bool {
		self.stack.contains(&key)
	}

	fn insert(&mut self, key: HashedKey, size: ObjectSize) {
		if self.stack.contains(&key) {
			// Existing key: track any size change, then treat as an access.
			// Per the admission policy, a `set()` always places the object
			// at the top of the fast tier, promoting it if it was slow.
			self.resize_key(key, size);
			self.touch_fast_key(key);
			return;
		}

		self.stack.push_front(key);
		self.entries.insert(key, LruEntry { tier: Tier::Fast, size });
		self.fast_used += size as CacheSize;
		self.fast_count += 1;

		if self.fast_boundary.is_none() {
			self.fast_boundary = Some(key);
		}

		self.settle_fast_tier();
	}

	fn update(&mut self, key: HashedKey) {
		if self.entries.contains_key(&key) {
			self.touch_fast_key(key);
		}
	}

	fn remove(&mut self, key: HashedKey) {
		let entry = self.entries.remove(&key);
		let size = entry.map(|entry| entry.size).unwrap_or(0) as CacheSize;
		let tier = entry.map(|entry| entry.tier);

		let new_boundary_if_needed = if tier == Some(Tier::Fast) && self.fast_boundary == Some(key) {
			self.stack.before(&key).copied()
		} else {
			None
		};

		self.stack.remove(&key);

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
		self.stack.clear();
		self.entries.clear();

		self.fast_used = 0;
		self.slow_used = 0;
		self.fast_count = 0;
		self.fast_boundary = None;
		self.migrations.clear();
	}

	fn evict_one(&mut self) -> Option<HashedKey> {
		let key = self.stack.pop_back()?;
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
					self.fast_boundary = self.stack.back().copied();
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

	fn drain(stack: &mut LruHybridStack) -> Vec<(HashedKey, Tier)> {
		stack.drain_tier_migrations()
	}

	/// Smallest `fast_capacity` whose high watermark is at least `bytes`, so a
	/// test can fill the fast tier to exactly `bytes` without tripping a
	/// demotion pass at whatever ratios are configured. The watermarks are
	/// process-global (`OnceLock` + env), so tests derive their expectations
	/// from `watermarks::high()`/`low()` instead of setting the env vars.
	fn capacity_admitting(bytes: CacheSize) -> CacheSize {
		let mut capacity = (bytes as f64 / watermarks::high()).ceil() as CacheSize;

		// Guard against `high_bytes`'s `as u64` truncation landing a byte short
		// for some ratio/rounding combinations.
		while watermarks::high_bytes(capacity) < bytes {
			capacity += 1;
		}

		capacity
	}

	#[test]
	fn evict_one_terminates_when_both_keys_are_immediately_demoted() {
		// Reproduces the `apply_evictions` scenario where fast_capacity is
		// tiny relative to object sizes, so *both* inserted keys demote to
		// slow immediately (fast_boundary bounces to None each time).
		let mut stack = LruHybridStack::new(4);

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
		let mut stack = LruHybridStack::new(1_000);

		stack.insert(1, 10);
		stack.insert(2, 10);

		assert_eq!(stack.tier_of(1), Some(Tier::Fast));
		assert_eq!(stack.tier_of(2), Some(Tier::Fast));
		assert_eq!(stack.fast_bytes_used(), 20);
		assert_eq!(stack.slow_bytes_used(), 0);
	}

	#[test]
	fn fast_tier_pressure_demotes_lru_tail() {
		// Sized so two 10-byte objects sit at/below the high watermark and the
		// third pushes past it, whatever ratios are configured. How *many*
		// objects the resulting pass demotes depends on the low watermark, so
		// assert the victim order (LRU end inward) rather than a fixed count.
		let capacity = capacity_admitting(20);
		let mut stack = LruHybridStack::new(capacity);

		stack.insert(1, 10); // fast: [1]
		stack.insert(2, 10); // fast: [2, 1]
		drain(&mut stack);
		assert_eq!(stack.fast_bytes_used(), 20);

		stack.insert(3, 10); // pushes fast_used to 30, past the high watermark
		let migrations = drain(&mut stack);

		// Victims are taken from the LRU end inward (1, then 2, ...) and the
		// just-inserted MRU key is never among them.
		assert!(!migrations.is_empty());
		assert!(migrations.iter().all(|(_, tier)| *tier == Tier::Slow));

		let demoted: Vec<HashedKey> = migrations.iter().map(|(key, _)| *key).collect();
		assert_eq!(demoted, (1..=demoted.len() as HashedKey).collect::<Vec<HashedKey>>());

		assert_eq!(stack.tier_of(1), Some(Tier::Slow));

		// The MRU key survives unless the low watermark is tight enough to
		// empty the tier outright.
		if watermarks::low_bytes(capacity) >= 10 {
			assert_eq!(stack.tier_of(3), Some(Tier::Fast));
		}

		// The pass drained to the low watermark, and no bytes were lost.
		assert!(stack.fast_bytes_used() <= watermarks::low_bytes(capacity));
		assert_eq!(stack.fast_bytes_used() + stack.slow_bytes_used(), 30);
	}

	#[test]
	fn accessing_a_slow_key_promotes_it_and_may_demote_another() {
		// `capacity_admitting(20)` puts the high watermark in [20, 21], so
		// four 10-byte objects always overshoot it and key 1 (the LRU tail) is
		// always the first victim -- i.e. always slow by the time it is
		// re-accessed below, at any configured ratio.
		let capacity = capacity_admitting(20);
		let mut stack = LruHybridStack::new(capacity);

		stack.insert(1, 10);
		stack.insert(2, 10);
		stack.insert(3, 10); // demotes from the LRU end, starting with key 1
		stack.insert(4, 10);
		drain(&mut stack);

		assert_eq!(stack.tier_of(1), Some(Tier::Slow));

		// Accessing the slow key promotes it back to the fast tier, which
		// pushes fast_used past the high watermark and so demotes the fast-tier
		// LRU tail behind it.
		let before = stack.fast_bytes_used();

		stack.update(1);
		let migrations = drain(&mut stack);

		assert_eq!(stack.tier_of(1), Some(Tier::Fast));

		// Demotions are applied before the promotion that triggered them, so a
		// promotion never has its DRAM write applied before the corresponding
		// demotion's DRAM free (see `touch_fast_key`'s doc). The promotion is
		// therefore always the *last* entry, and everything ahead of it is a
		// demotion of some other key.
		assert_eq!(migrations.last(), Some(&(1, Tier::Fast)));
		assert!(
			migrations[..migrations.len() - 1]
				.iter()
				.all(|(key, tier)| *tier == Tier::Slow && *key != 1)
		);

		// Whenever the promotion carries the tier past the high watermark --
		// which it does at the default ratios -- it must have demoted something,
		// and the resulting pass must have drained to the low watermark. When it
		// does not, the promotion is the only migration and the tier is left
		// wherever it already sat.
		if before + 10 > watermarks::high_bytes(capacity) {
			assert!(migrations.len() > 1);
			assert!(stack.fast_bytes_used() <= watermarks::low_bytes(capacity));
		} else {
			assert_eq!(migrations.len(), 1);
		}

		assert_eq!(stack.fast_bytes_used() + stack.slow_bytes_used(), 40);
	}

	#[test]
	fn promoted_key_is_absent_from_slow_afterward_and_demoted_key_from_fast() {
		let mut stack = LruHybridStack::new(25);

		stack.insert(1, 10);
		stack.insert(2, 10);
		stack.insert(3, 10); // demotes 1
		drain(&mut stack);

		stack.update(1); // promotes 1, likely demotes 2
		drain(&mut stack);

		// A key can never be tagged as being in both tiers simultaneously —
		// `tier_of` returns exactly one Tier per key by construction.
		assert_ne!(stack.tier_of(1), Some(Tier::Slow));
		assert_ne!(stack.tier_of(2), Some(Tier::Fast));
	}

	#[test]
	fn object_counts_track_tier_membership() {
		let mut stack = LruHybridStack::new(15);

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
		let mut stack = LruHybridStack::new(15);

		stack.insert(1, 10); // fast
		stack.insert(2, 10); // demotes 1 -> slow, fast holds 2
		drain(&mut stack);

		assert_eq!(stack.tier_of(1), Some(Tier::Slow));
		assert_eq!(stack.tier_of(2), Some(Tier::Fast));

		// Tail of the whole list is the slow key (1), since it was demoted
		// and never re-accessed.
		assert_eq!(stack.evict_one(), Some(1));
		assert_eq!(stack.tier_of(2), Some(Tier::Fast));
		assert_eq!(stack.slow_bytes_used(), 0);
	}

	#[test]
	fn evict_one_falls_back_to_fast_tail_when_everything_still_fits() {
		let mut stack = LruHybridStack::new(1_000);

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
		let mut stack = LruHybridStack::new(0);

		stack.insert(1, 10);
		let migrations = drain(&mut stack);

		assert_eq!(migrations, vec![(1, Tier::Slow)]);
		assert_eq!(stack.tier_of(1), Some(Tier::Slow));
		assert_eq!(stack.fast_bytes_used(), 0);
		assert_eq!(stack.slow_bytes_used(), 10);
	}

	#[test]
	fn resizing_an_existing_key_adjusts_the_correct_tier_counter() {
		let mut stack = LruHybridStack::new(1_000);

		stack.insert(1, 10);
		drain(&mut stack);
		assert_eq!(stack.fast_bytes_used(), 10);

		// re-`set()` with a larger value: still fast, counter adjusted.
		stack.insert(1, 30);
		assert_eq!(stack.tier_of(1), Some(Tier::Fast));
		assert_eq!(stack.fast_bytes_used(), 30);
	}

	#[test]
	fn shrinking_fast_tier_at_runtime_triggers_demotions() {
		let mut stack = LruHybridStack::new(1_000);

		stack.insert(1, 100);
		stack.insert(2, 100);
		drain(&mut stack);

		// effective=150. fast_used starts at 200, above the high watermark
		// (150 * 0.95 = 142), so the shrink triggers a pass; demoting the LRU
		// tail (key 1, 100 bytes) alone lands at 100, already under the low
		// watermark (150 * 0.75 = 112), so the pass stops there.
		stack.resize_fast_tier(150);
		let migrations = drain(&mut stack);

		assert_eq!(migrations, vec![(1, Tier::Slow)]);
		assert_eq!(stack.tier_of(2), Some(Tier::Fast));
		assert_eq!(stack.fast_bytes_used(), 100);
		assert!(stack.fast_bytes_used() <= watermarks::low_bytes(150));
	}

	#[test]
	fn settle_drains_to_low_water_target_with_headroom() {
		const CAPACITY: CacheSize = 1_000;

		let high = watermarks::high_bytes(CAPACITY);
		let low = watermarks::low_bytes(CAPACITY);

		let mut stack = LruHybridStack::new(CAPACITY);

		// Fill to exactly the high watermark in 1-byte objects. The trigger is
		// strict (`fast_used > high_bytes`), so landing *on* it demotes nothing.
		for key in 1..=high {
			stack.insert(key, 1);
		}
		drain(&mut stack);
		assert_eq!(stack.fast_bytes_used(), high);

		// One more byte trips the trigger; the pass then drains all the way to
		// the low watermark rather than merely back under the ceiling, demoting
		// far more than the single object that would have been the bare minimum
		// (see the module doc's "High/low watermarks" section).
		stack.insert(high + 1, 1);
		let migrations = drain(&mut stack);

		assert_eq!(stack.fast_bytes_used(), low);
		assert_eq!(migrations.len() as CacheSize, high + 1 - low);
		assert_eq!(migrations.first(), Some(&(1, Tier::Slow)));
		assert!(migrations.iter().all(|(_, tier)| *tier == Tier::Slow));
	}

	#[test]
	fn remove_updates_boundary_and_counters() {
		let mut stack = LruHybridStack::new(1_000);

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
	fn shared_overhead_reserves_dram_and_demotes_earlier() {
		// Without overhead, two 40-byte values fit in a 100-byte fast tier.
		let mut plain = LruHybridStack::new(100);
		plain.insert(1, 40);
		plain.insert(2, 40);
		drain(&mut plain);
		assert_eq!(plain.tier_of(1), Some(Tier::Fast));
		assert_eq!(plain.tier_of(2), Some(Tier::Fast));

		// With a 30-byte per-object shared reservation, the second insert
		// reserves 2 × 30 = 60, leaving an effective value budget of 40 --
		// tighter than the raw values (80) fit. The watermarks apply on top of
		// that reduced effective budget, not the raw capacity. Both objects end
		// up demoted: 80 exceeds the high watermark (40 * 0.95 = 38), and after
		// the LRU tail (key 1) demotes, fast_used (40) is still above the low
		// watermark (40 * 0.75 = 30), so settle continues and demotes key 2 as
		// well, landing at 0. This is the low-water drain interacting with the
		// DRAM-cap reservation at small scale -- with only two same-sized
		// objects there's no smaller demotion available to land closer to the
		// target without emptying the tier.
		let mut stack = LruHybridStack::new(100).with_shared_overhead(30);
		stack.insert(1, 40);
		stack.insert(2, 40);
		let migrations = drain(&mut stack);

		assert_eq!(migrations, vec![(1, Tier::Slow), (2, Tier::Slow)]);
		assert_eq!(stack.tier_of(1), Some(Tier::Slow));
		assert_eq!(stack.tier_of(2), Some(Tier::Slow));
		assert_eq!(stack.fast_bytes_used(), 0);
	}

	#[test]
	fn shared_overhead_exceeding_capacity_demotes_all_but_never_evicts() {
		// One object's shared reservation (100) already exceeds the whole
		// fast budget (50): the effective value budget saturates to 0, so the
		// object demotes to slow immediately on admission.
		let mut stack = LruHybridStack::new(50).with_shared_overhead(100);
		stack.insert(1, 10);
		let migrations = drain(&mut stack);

		assert_eq!(migrations, vec![(1, Tier::Slow)]);
		assert_eq!(stack.tier_of(1), Some(Tier::Slow));
		assert_eq!(stack.fast_bytes_used(), 0);

		// Demotion is the only response — the object is still tracked (the
		// DRAM budget never evicts; `needs_capacity_eviction` stays default).
		assert_eq!(stack.len(), 1);
		assert!(!stack.needs_capacity_eviction());
	}

	// ---------------------------------------------------------------------
	// Shared high/low watermarks (`super::watermarks`).
	//
	// The ratios are process-global (`OnceLock`, seeded once from
	// `FAST_TIER_HIGH_WATERMARK` / `FAST_TIER_LOW_WATERMARK`), so these tests
	// cannot set the env vars for themselves without racing every other test in
	// the binary. They compute their expectations from `watermarks::high()` /
	// `watermarks::low()` instead, and therefore hold at any configured ratio
	// pair -- including the `1.0` / `1.0` setting that restores the original
	// drain-to-the-ceiling behaviour.
	// ---------------------------------------------------------------------

	#[test]
	fn usage_just_below_the_high_watermark_does_not_trigger_a_pass() {
		const CAPACITY: CacheSize = 1_000;

		let high = watermarks::high_bytes(CAPACITY);
		let mut stack = LruHybridStack::new(CAPACITY);

		// Fill to exactly the high watermark in 1-byte objects. The trigger is
		// strict (`fast_used > high_bytes`), so sitting *on* the watermark --
		// and every byte below it -- must demote nothing.
		for key in 1..=high {
			stack.insert(key, 1);
			assert!(
				stack.drain_tier_migrations().is_empty(),
				"demoted at {} bytes, below the high watermark of {}",
				key,
				high,
			);
		}

		assert_eq!(stack.fast_bytes_used(), high);
		assert_eq!(stack.slow_bytes_used(), 0);
		assert_eq!(stack.fast_object_count(), high as usize);
		assert_eq!(stack.slow_object_count(), 0);
	}

	#[test]
	fn usage_above_the_high_watermark_triggers_a_pass() {
		const CAPACITY: CacheSize = 1_000;

		let high = watermarks::high_bytes(CAPACITY);
		let mut stack = LruHybridStack::new(CAPACITY);

		for key in 1..=high {
			stack.insert(key, 1);
		}
		drain(&mut stack);
		assert_eq!(stack.fast_bytes_used(), high);

		// The very first byte past the watermark fires the pass.
		stack.insert(high + 1, 1);
		let migrations = drain(&mut stack);

		assert!(!migrations.is_empty());
		assert!(migrations.iter().all(|(_, tier)| *tier == Tier::Slow));
		assert_eq!(migrations.first(), Some(&(1, Tier::Slow)));
	}

	#[test]
	fn a_triggered_pass_drains_to_the_low_watermark_not_the_ceiling() {
		const CAPACITY: CacheSize = 1_000;

		let high = watermarks::high_bytes(CAPACITY);
		let low = watermarks::low_bytes(CAPACITY);
		let mut stack = LruHybridStack::new(CAPACITY);

		for key in 1..=high {
			stack.insert(key, 1);
		}
		drain(&mut stack);

		stack.insert(high + 1, 1);
		let migrations = drain(&mut stack);

		// 1-byte objects, so the drain lands exactly on the low watermark
		// rather than merely somewhere at or below it.
		assert_eq!(stack.fast_bytes_used(), low);
		assert!(stack.fast_bytes_used() <= watermarks::low_bytes(CAPACITY));

		// Draining back to the ceiling would have demoted exactly one object
		// and left the tier at `high`; the low watermark instead demotes the
		// whole `high - low` band in a single pass.
		assert_eq!(migrations.len() as CacheSize, high + 1 - low);

		if low < high {
			assert!(stack.fast_bytes_used() < high);
			assert!(stack.fast_bytes_used() < CAPACITY);
			assert!(migrations.len() > 1);
		}
	}

	#[test]
	fn counters_stay_consistent_across_a_watermark_pass() {
		const CAPACITY: CacheSize = 1_000;

		let high = watermarks::high_bytes(CAPACITY);
		let low = watermarks::low_bytes(CAPACITY);
		let total = high + 1;

		let mut stack = LruHybridStack::new(CAPACITY);

		for key in 1..=total {
			stack.insert(key, 1);
		}
		let migrations = drain(&mut stack);

		let demoted = migrations.len() as CacheSize;

		// Byte counters: every inserted byte is accounted to exactly one tier,
		// and the split matches the number of demotions the pass emitted.
		assert_eq!(stack.fast_bytes_used(), low);
		assert_eq!(stack.slow_bytes_used(), demoted);
		assert_eq!(stack.fast_bytes_used() + stack.slow_bytes_used(), total);

		// Object counts: 1-byte objects, so each count equals its tier's bytes,
		// and together they cover every tracked key exactly once.
		assert_eq!(stack.fast_object_count() as CacheSize, low);
		assert_eq!(stack.slow_object_count() as CacheSize, demoted);
		assert_eq!(stack.fast_object_count() + stack.slow_object_count(), total as usize);
		assert_eq!(stack.len(), total as usize);

		// Per-key tiers agree with the counters: the demoted keys are exactly
		// the LRU-end prefix, and nothing was dropped or double-counted.
		for (key, tier) in &migrations {
			assert_eq!(stack.tier_of(*key), Some(*tier));
		}

		let fast_keys = (1..=total).filter(|key| stack.tier_of(*key) == Some(Tier::Fast)).count();
		let slow_keys = (1..=total).filter(|key| stack.tier_of(*key) == Some(Tier::Slow)).count();

		assert_eq!(fast_keys, stack.fast_object_count());
		assert_eq!(slow_keys, stack.slow_object_count());
	}

	#[test]
	fn clear_resets_all_state() {
		let mut stack = LruHybridStack::new(15);

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
