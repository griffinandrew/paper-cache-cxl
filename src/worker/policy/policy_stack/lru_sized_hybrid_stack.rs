/*
 * Copyright (c) Kia Shakiba
 *
 * This source code is licensed under the GNU AGPLv3 license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! `LruSizedHybridStack` — `PaperPolicy::LruSizedHybrid`: `LruHybridStack`'s
//! recency-segmented fast/slow LRU queue, but with the fast tier's (and the
//! slow tier's) *bookkeeping* further split into two size-routed segments
//! ("small"/"large") by a runtime-configurable byte threshold.
//!
//! ## Four independent, homogeneous recency lists — not a boundary cursor
//!
//! `LruHybridStack` tracks fast and slow membership as a single combined
//! recency list plus a `fast_boundary` cursor marking where the fast prefix
//! ends, because there is exactly one fast segment and one slow segment
//! sharing that one list. Here there are two independent fast sources
//! (`small_fast`, `large_fast`) each feeding its own independent slow
//! destination (`small_slow`, `large_slow`) — a cursor trick doesn't
//! generalize to that shape. Instead each of the four lists is fully
//! homogeneous (100% one segment's keys), so each list's own tail is
//! directly its own demotion/eviction candidate; no cursor is needed
//! anywhere. This ends up *simpler* than `LruHybridStack`'s bookkeeping, not
//! more complex, despite tracking four lists instead of one.
//!
//! ## Classification input: `ObjectSize` (base_size), not raw value length
//!
//! `PolicyStack::insert`'s only size parameter is the same `ObjectSize`
//! (key + value + expiry-slot bytes) every other stack already budgets
//! against — there is no second, raw-`value.len()` channel into this stack.
//! Threading one through would mean changing `PolicyStack::insert`'s
//! signature for all nine other stacks for a benefit that's only ever a
//! small, near-constant offset (key size plus a fixed expiry-slot cost) near
//! the threshold boundary. `classify` compares against this same `ObjectSize`
//! throughout.
//!
//! ## Admission, promotion, and reclassification are all "touch_fast"
//!
//! `LruHybridStack::touch_fast_key` already promotes *any* touched existing
//! key straight to fast, whichever tier it was previously in — a `set()`
//! overwrite is never treated differently from a cache hit. This design's
//! `touch_fast` keeps that rule unchanged and just adds "which of the two
//! fast segments" on top of it: reclassifying an existing key from one fast
//! segment to the other (an overwrite whose new size crosses the threshold)
//! and promoting a slow-resident key back to fast both funnel through the
//! same method, both landing wherever `classify` currently says the key's
//! *current* size belongs. A same-tier fast→fast move never emits a
//! `(key, Tier)` migration (both segments are physically `TieredBuffer::
//! Fast`); only a genuine slow→fast crossing does — same guard
//! `LruHybridStack::touch_fast_key` uses for its own self-eviction edge case,
//! reused here.
//!
//! ## Eviction priority
//!
//! `evict_one` always prefers a real slow-tier candidate: if both
//! `small_slow`/`large_slow` are non-empty, whichever currently holds *more*
//! objects is preferred (a cheap `len()`-based proxy for "probably has the
//! older tail", avoiding real cross-list timestamps — this crate tracks
//! recency purely via list position everywhere else too). Only if *both*
//! slow lists are empty (nothing has ever been demoted) does eviction fall
//! back to whichever fast segment is furthest over its own budget by ratio —
//! a direct port of `LruHybridStack`'s own documented last-resort fallback
//! ("evict from fast only if nothing has ever been demoted yet"), not a new
//! invention. Neither slow list carries an independent capacity of its own —
//! terminal eviction is still governed purely by the caller's overall
//! `status.used_size() > max_size` check (see `PolicyWorker::
//! apply_evictions`), exactly like `LruHybridStack`'s single slow tier today.
//!
//! ## Shared DRAM-reservation overhead
//!
//! `shared_overhead` (see `crate::object::overhead::
//! get_hybrid_dram_shared_overhead`) is reserved proportionally between the
//! two fast segments' capacities (`reserved_shares`), not charged in full
//! against each independently — the underlying per-object metadata cost is
//! real only once, and double-charging it would waste usable fast-tier
//! budget for no reason. The two slow lists carry no capacity, so they have
//! nothing to reserve against.

#[cfg(not(feature = "eviction_stacks_pmem"))]
use std::collections::HashMap;
#[cfg(feature = "eviction_stacks_pmem")]
use hashbrown::HashMap;

#[cfg(not(feature = "eviction_stacks_pmem"))]
use kwik::collections::HashList;
#[cfg(feature = "eviction_stacks_pmem")]
use super::pmem_collections::PmemHashList;

// See `LruHybridStack`'s identical note: eviction-stack metadata is
// allocated through the same crate-wide `Hybrid` alias (`HybridObjects`,
// UMF/TBB, NUMA node 1) that `BufferPMEM`/other PMEM features already use.
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

/// Fraction of the effective fast-segment budget `settle_small_fast`/
/// `settle_large_fast` drain down to once triggered — identical constant and
/// rationale to `LruHybridStack`'s (see that module doc's "Low-water
/// headroom" section): `PaperCache::set()` writes new object bytes to DRAM
/// synchronously at the API layer before this stack (on the background
/// `PolicyWorker` thread) ever sees the corresponding event, so a burst of
/// concurrent `set()`s can transiently overshoot either segment's own
/// last-known bookkeeping between worker polls. Applied identically to both
/// segments — the race this protects against doesn't care which segment a
/// burst's objects land in.
const FAST_TIER_LOW_WATER_RATIO: f64 = 0.98;

// DRAM-backed by default; under `eviction_stacks_pmem`, all four recency
// lists and the combined entry map move to PMEM together (co-located with
// the slow-tier object bytes), exactly like `LruHybridStack`'s bookkeeping
// does — independent of which *tier* a given tracked key's value bytes are
// actually in.
#[cfg(not(feature = "eviction_stacks_pmem"))]
type RecencyList = HashList<HashedKey, NoHasher>;
#[cfg(feature = "eviction_stacks_pmem")]
type RecencyList = PmemHashList<HashedKey, NoHasher>;

/// Which of the four lists a key is currently tracked in.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum SizeQueue {
	SmallFast,
	LargeFast,
	SmallSlow,
	LargeSlow,
}

/// Combined per-key bookkeeping: which list and what size. One map, not
/// four, following the same consolidation `LruHybridStack`/`TwoQHybridStack`
/// already use (see `object/overhead.rs`'s `LruSizedHybrid` arm).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct SizedEntry {
	queue: SizeQueue,
	size: ObjectSize,
}

#[cfg(not(feature = "eviction_stacks_pmem"))]
type EntryMap = HashMap<HashedKey, SizedEntry, NoHasher>;
#[cfg(feature = "eviction_stacks_pmem")]
type EntryMap = HashMap<HashedKey, SizedEntry, NoHasher, Hybrid>;

pub struct LruSizedHybridStack {
	small_fast: RecencyList,
	large_fast: RecencyList,
	small_slow: RecencyList,
	large_slow: RecencyList,
	entries: EntryMap,

	small_capacity: CacheSize,
	large_capacity: CacheSize,
	size_threshold: CacheSize,

	small_fast_used: CacheSize,
	large_fast_used: CacheSize,
	small_slow_used: CacheSize,
	large_slow_used: CacheSize,

	small_fast_count: usize,
	large_fast_count: usize,
	small_slow_count: usize,
	large_slow_count: usize,

	/// Approximate per-object DRAM cost of the shared structures (object
	/// hashtable + eviction stacks), reserved proportionally between the two
	/// fast segments' capacities — see the module doc. `0` unless set via
	/// `with_shared_overhead`.
	shared_overhead: CacheSize,

	/// (key, new tier) pairs recorded since the last `drain_tier_migrations`.
	migrations: Vec<(HashedKey, Tier)>,
}

impl LruSizedHybridStack {
	#[cfg(not(feature = "eviction_stacks_pmem"))]
	fn new_collections() -> (RecencyList, RecencyList, RecencyList, RecencyList, EntryMap) {
		(
			HashList::default(),
			HashList::default(),
			HashList::default(),
			HashList::default(),
			HashMap::default(),
		)
	}

	#[cfg(feature = "eviction_stacks_pmem")]
	fn new_collections() -> (RecencyList, RecencyList, RecencyList, RecencyList, EntryMap) {
		(
			PmemHashList::with_hasher(NoHasher::default()),
			PmemHashList::with_hasher(NoHasher::default()),
			PmemHashList::with_hasher(NoHasher::default()),
			PmemHashList::with_hasher(NoHasher::default()),
			HashMap::with_hasher_in(NoHasher::default(), Hybrid),
		)
	}

	pub fn new(small_capacity: CacheSize, large_capacity: CacheSize, size_threshold: CacheSize) -> Self {
		let (small_fast, large_fast, small_slow, large_slow, entries) = Self::new_collections();

		LruSizedHybridStack {
			small_fast,
			large_fast,
			small_slow,
			large_slow,
			entries,

			small_capacity,
			large_capacity,
			size_threshold,

			small_fast_used: 0,
			large_fast_used: 0,
			small_slow_used: 0,
			large_slow_used: 0,

			small_fast_count: 0,
			large_fast_count: 0,
			small_slow_count: 0,
			large_slow_count: 0,

			shared_overhead: 0,
			migrations: Vec::new(),
		}
	}

	/// Sets the approximate per-object shared-structure DRAM overhead. See
	/// `crate::object::overhead::get_hybrid_dram_shared_overhead`.
	/// Builder-style so `init_policy_stack` can wire it in without
	/// disturbing `new`'s signature (unit tests keep the default `0`).
	pub fn with_shared_overhead(mut self, overhead: CacheSize) -> Self {
		self.shared_overhead = overhead;
		self
	}

	/// The configured SMALL fast segment's byte budget.
	pub fn small_capacity(&self) -> CacheSize {
		self.small_capacity
	}

	/// The configured LARGE fast segment's byte budget.
	pub fn large_capacity(&self) -> CacheSize {
		self.large_capacity
	}

	/// The current small/large size-classification threshold.
	pub fn size_threshold(&self) -> CacheSize {
		self.size_threshold
	}

	/// Returns the tier the given (currently tracked) key is in, or `None`
	/// if the key isn't tracked. Exposed for integration diagnostics/tests,
	/// mirroring `LruHybridStack::tier_of`.
	pub fn tier_of(&self, key: HashedKey) -> Option<Tier> {
		self.queue_of(key).map(|queue| match queue {
			SizeQueue::SmallFast | SizeQueue::LargeFast => Tier::Fast,
			SizeQueue::SmallSlow | SizeQueue::LargeSlow => Tier::Slow,
		})
	}

	/// Returns which of the four lists the given (currently tracked) key is
	/// in, or `None` if the key isn't tracked. Only used by this file's own
	/// unit tests.
	fn queue_of(&self, key: HashedKey) -> Option<SizeQueue> {
		self.entries.get(&key).map(|entry| entry.queue)
	}

	/// `true` if `size` classifies as the SMALL segment (`size <
	/// size_threshold`), `false` for LARGE.
	fn classify(&self, size: ObjectSize) -> bool {
		(size as CacheSize) < self.size_threshold
	}

	/// Splits the total reserved shared-structure DRAM cost
	/// (`tracked object count × shared_overhead`, across all four lists —
	/// shared metadata scales with everything tracked, not just one
	/// segment) proportionally between the two fast segments' capacities.
	/// `(0, 0)` if both capacities are zero (nothing to proportion against).
	fn reserved_shares(&self) -> (CacheSize, CacheSize) {
		let reserved = self.entries.len() as CacheSize * self.shared_overhead;
		let total_capacity = self.small_capacity + self.large_capacity;

		if total_capacity == 0 {
			return (0, 0);
		}

		let small_share = ((reserved as u128 * self.small_capacity as u128) / total_capacity as u128) as CacheSize;
		let large_share = reserved.saturating_sub(small_share);

		(small_share, large_share)
	}

	fn effective_small(&self) -> CacheSize {
		self.small_capacity.saturating_sub(self.reserved_shares().0)
	}

	fn effective_large(&self) -> CacheSize {
		self.large_capacity.saturating_sub(self.reserved_shares().1)
	}

	/// Records a size change for an already-tracked key without altering
	/// its queue, adjusting whichever counter currently applies.
	fn resize_key(&mut self, key: HashedKey, new_size: ObjectSize) {
		let Some(entry) = self.entries.get_mut(&key) else { return };

		let old_size = entry.size;
		entry.size = new_size;
		let delta = new_size as i64 - old_size as i64;

		match entry.queue {
			SizeQueue::SmallFast => {
				self.small_fast_used = (self.small_fast_used as i64 + delta).max(0) as CacheSize;
			},
			SizeQueue::LargeFast => {
				self.large_fast_used = (self.large_fast_used as i64 + delta).max(0) as CacheSize;
			},
			SizeQueue::SmallSlow => {
				self.small_slow_used = (self.small_slow_used as i64 + delta).max(0) as CacheSize;
			},
			SizeQueue::LargeSlow => {
				self.large_slow_used = (self.large_slow_used as i64 + delta).max(0) as CacheSize;
			},
		}
	}

	fn remove_from_small_fast(&mut self, key: HashedKey, size: ObjectSize) {
		self.small_fast.remove(&key);
		self.small_fast_used = self.small_fast_used.saturating_sub(size as CacheSize);
		self.small_fast_count = self.small_fast_count.saturating_sub(1);
	}

	fn remove_from_large_fast(&mut self, key: HashedKey, size: ObjectSize) {
		self.large_fast.remove(&key);
		self.large_fast_used = self.large_fast_used.saturating_sub(size as CacheSize);
		self.large_fast_count = self.large_fast_count.saturating_sub(1);
	}

	fn remove_from_small_slow(&mut self, key: HashedKey, size: ObjectSize) {
		self.small_slow.remove(&key);
		self.small_slow_used = self.small_slow_used.saturating_sub(size as CacheSize);
		self.small_slow_count = self.small_slow_count.saturating_sub(1);
	}

	fn remove_from_large_slow(&mut self, key: HashedKey, size: ObjectSize) {
		self.large_slow.remove(&key);
		self.large_slow_used = self.large_slow_used.saturating_sub(size as CacheSize);
		self.large_slow_count = self.large_slow_count.saturating_sub(1);
	}

	fn add_to_small_fast(&mut self, key: HashedKey, size: ObjectSize) {
		self.small_fast.push_front(key);
		self.small_fast_used += size as CacheSize;
		self.small_fast_count += 1;
	}

	fn add_to_large_fast(&mut self, key: HashedKey, size: ObjectSize) {
		self.large_fast.push_front(key);
		self.large_fast_used += size as CacheSize;
		self.large_fast_count += 1;
	}

	/// Moves an already-tracked key to the front of whichever fast segment
	/// its *current* size classifies as, promoting it if it was in either
	/// slow list, or reclassifying it if it was in the other fast segment —
	/// see the module doc's "Admission, promotion, and reclassification"
	/// section. Used by both `insert` (an existing key — a `set()` always
	/// re-admits to fast) and `update` (a fast-or-slow hit).
	fn touch_fast(&mut self, key: HashedKey) {
		let Some(entry) = self.entries.get(&key).copied() else { return };
		let target_small = self.classify(entry.size);
		let was_slow = matches!(entry.queue, SizeQueue::SmallSlow | SizeQueue::LargeSlow);

		match (entry.queue, target_small) {
			(SizeQueue::SmallFast, true) => {
				self.small_fast.move_front(&key);
				self.settle_small_fast();
				return;
			},

			(SizeQueue::LargeFast, false) => {
				self.large_fast.move_front(&key);
				self.settle_large_fast();
				return;
			},

			(SizeQueue::SmallFast, false) => self.remove_from_small_fast(key, entry.size),
			(SizeQueue::LargeFast, true) => self.remove_from_large_fast(key, entry.size),
			(SizeQueue::SmallSlow, _) => self.remove_from_small_slow(key, entry.size),
			(SizeQueue::LargeSlow, _) => self.remove_from_large_slow(key, entry.size),
		}

		let target_queue = if target_small { SizeQueue::SmallFast } else { SizeQueue::LargeFast };

		if target_small {
			self.add_to_small_fast(key, entry.size);
		} else {
			self.add_to_large_fast(key, entry.size);
		}

		if let Some(entry) = self.entries.get_mut(&key) {
			entry.queue = target_queue;
		}

		if target_small {
			self.settle_small_fast();
		} else {
			self.settle_large_fast();
		}

		// Only a genuine slow->fast promotion needs a migration -- a
		// fast<->fast reclassification never crosses the Tier boundary
		// (both segments are physically TieredBuffer::Fast). Pushed after
		// `settle_*` (which may push demotions this same promotion
		// triggered) and guarded on the key still being in the target
		// queue, for the same reasons `LruHybridStack::touch_fast_key`
		// documents: demotions must apply before the promotion that
		// triggered them, and an extremely tight target segment can demote
		// this same key straight back out within the settle call above
		// (self-eviction), in which case no separate Fast entry should
		// follow it.
		if was_slow && self.entries.get(&key).map(|entry| entry.queue) == Some(target_queue) {
			self.migrations.push((key, Tier::Fast));
		}
	}

	/// Demotes the SMALL fast segment's LRU tail(s) into `small_slow`,
	/// triggered only once `small_fast_used` genuinely exceeds its effective
	/// budget, drained down to `FAST_TIER_LOW_WATER_RATIO` of that budget —
	/// see `LruHybridStack::settle_fast_tier`'s identical shape/rationale.
	fn settle_small_fast(&mut self) {
		let effective = self.effective_small();

		if self.small_fast_used <= effective {
			return;
		}

		let drain_target = (effective as f64 * FAST_TIER_LOW_WATER_RATIO) as CacheSize;

		while self.small_fast_used > drain_target {
			let Some(demote_key) = self.small_fast.pop_back() else { break };
			let size = self.entries.get(&demote_key).map(|entry| entry.size).unwrap_or(0) as CacheSize;

			self.small_fast_used = self.small_fast_used.saturating_sub(size);
			self.small_fast_count = self.small_fast_count.saturating_sub(1);

			self.small_slow.push_front(demote_key);
			self.small_slow_used += size;
			self.small_slow_count += 1;

			if let Some(entry) = self.entries.get_mut(&demote_key) {
				entry.queue = SizeQueue::SmallSlow;
			}

			self.migrations.push((demote_key, Tier::Slow));
		}
	}

	/// LARGE-segment counterpart of `settle_small_fast`, demoting into
	/// `large_slow`.
	fn settle_large_fast(&mut self) {
		let effective = self.effective_large();

		if self.large_fast_used <= effective {
			return;
		}

		let drain_target = (effective as f64 * FAST_TIER_LOW_WATER_RATIO) as CacheSize;

		while self.large_fast_used > drain_target {
			let Some(demote_key) = self.large_fast.pop_back() else { break };
			let size = self.entries.get(&demote_key).map(|entry| entry.size).unwrap_or(0) as CacheSize;

			self.large_fast_used = self.large_fast_used.saturating_sub(size);
			self.large_fast_count = self.large_fast_count.saturating_sub(1);

			self.large_slow.push_front(demote_key);
			self.large_slow_used += size;
			self.large_slow_count += 1;

			if let Some(entry) = self.entries.get_mut(&demote_key) {
				entry.queue = SizeQueue::LargeSlow;
			}

			self.migrations.push((demote_key, Tier::Slow));
		}
	}

	/// Last-resort eviction fallback, only reachable when both slow lists
	/// are empty (nothing has ever been demoted) — see the module doc's
	/// "Eviction priority" section. Evicts from whichever fast segment is
	/// furthest over its own budget by ratio (`used / capacity`, treating a
	/// zero-capacity segment with any usage as infinitely over), ties (and
	/// the both-empty-fast case) going to small.
	fn evict_fast_fallback(&mut self) -> Option<HashedKey> {
		if self.small_fast_count == 0 && self.large_fast_count == 0 {
			return None;
		}

		let ratio = |used: CacheSize, capacity: CacheSize| -> f64 {
			if capacity == 0 {
				if used > 0 { f64::INFINITY } else { 0.0 }
			} else {
				used as f64 / capacity as f64
			}
		};

		let pick_small = if self.small_fast_count == 0 {
			false
		} else if self.large_fast_count == 0 {
			true
		} else {
			ratio(self.small_fast_used, self.small_capacity) >= ratio(self.large_fast_used, self.large_capacity)
		};

		if pick_small {
			let key = self.small_fast.pop_back()?;
			let size = self.entries.remove(&key).map(|entry| entry.size).unwrap_or(0) as CacheSize;
			self.small_fast_used = self.small_fast_used.saturating_sub(size);
			self.small_fast_count = self.small_fast_count.saturating_sub(1);
			Some(key)
		} else {
			let key = self.large_fast.pop_back()?;
			let size = self.entries.remove(&key).map(|entry| entry.size).unwrap_or(0) as CacheSize;
			self.large_fast_used = self.large_fast_used.saturating_sub(size);
			self.large_fast_count = self.large_fast_count.saturating_sub(1);
			Some(key)
		}
	}
}

impl PolicyStack for LruSizedHybridStack {
	fn is_policy(&self, policy: &PaperPolicy) -> bool {
		matches!(policy, PaperPolicy::LruSizedHybrid)
	}

	fn len(&self) -> usize {
		self.entries.len()
	}

	fn contains(&self, key: HashedKey) -> bool {
		self.entries.contains_key(&key)
	}

	fn insert(&mut self, key: HashedKey, size: ObjectSize) {
		if self.entries.contains_key(&key) {
			// Existing key: track any size change, then treat as an
			// access -- a `set()` always re-admits to fast, reclassifying
			// between segments if the new size crosses the threshold.
			self.resize_key(key, size);
			self.touch_fast(key);
			return;
		}

		if self.classify(size) {
			self.small_fast.push_front(key);
			self.entries.insert(key, SizedEntry { queue: SizeQueue::SmallFast, size });
			self.small_fast_used += size as CacheSize;
			self.small_fast_count += 1;
			self.settle_small_fast();
		} else {
			self.large_fast.push_front(key);
			self.entries.insert(key, SizedEntry { queue: SizeQueue::LargeFast, size });
			self.large_fast_used += size as CacheSize;
			self.large_fast_count += 1;
			self.settle_large_fast();
		}
	}

	fn update(&mut self, key: HashedKey) {
		if self.entries.contains_key(&key) {
			self.touch_fast(key);
		}
	}

	fn remove(&mut self, key: HashedKey) {
		let Some(entry) = self.entries.remove(&key) else { return };

		match entry.queue {
			SizeQueue::SmallFast => {
				self.small_fast.remove(&key);
				self.small_fast_used = self.small_fast_used.saturating_sub(entry.size as CacheSize);
				self.small_fast_count = self.small_fast_count.saturating_sub(1);
			},
			SizeQueue::LargeFast => {
				self.large_fast.remove(&key);
				self.large_fast_used = self.large_fast_used.saturating_sub(entry.size as CacheSize);
				self.large_fast_count = self.large_fast_count.saturating_sub(1);
			},
			SizeQueue::SmallSlow => {
				self.small_slow.remove(&key);
				self.small_slow_used = self.small_slow_used.saturating_sub(entry.size as CacheSize);
				self.small_slow_count = self.small_slow_count.saturating_sub(1);
			},
			SizeQueue::LargeSlow => {
				self.large_slow.remove(&key);
				self.large_slow_used = self.large_slow_used.saturating_sub(entry.size as CacheSize);
				self.large_slow_count = self.large_slow_count.saturating_sub(1);
			},
		}
	}

	fn clear(&mut self) {
		self.small_fast.clear();
		self.large_fast.clear();
		self.small_slow.clear();
		self.large_slow.clear();
		self.entries.clear();

		self.small_fast_used = 0;
		self.large_fast_used = 0;
		self.small_slow_used = 0;
		self.large_slow_used = 0;

		self.small_fast_count = 0;
		self.large_fast_count = 0;
		self.small_slow_count = 0;
		self.large_slow_count = 0;

		self.migrations.clear();
	}

	fn evict_one(&mut self) -> Option<HashedKey> {
		if self.small_slow_count == 0 && self.large_slow_count == 0 {
			return self.evict_fast_fallback();
		}

		let pick_small = if self.small_slow_count == 0 {
			false
		} else if self.large_slow_count == 0 {
			true
		} else {
			self.small_slow_count >= self.large_slow_count
		};

		if pick_small {
			let key = self.small_slow.pop_back()?;
			let size = self.entries.remove(&key).map(|entry| entry.size).unwrap_or(0) as CacheSize;
			self.small_slow_used = self.small_slow_used.saturating_sub(size);
			self.small_slow_count = self.small_slow_count.saturating_sub(1);
			Some(key)
		} else {
			let key = self.large_slow.pop_back()?;
			let size = self.entries.remove(&key).map(|entry| entry.size).unwrap_or(0) as CacheSize;
			self.large_slow_used = self.large_slow_used.saturating_sub(size);
			self.large_slow_count = self.large_slow_count.saturating_sub(1);
			Some(key)
		}
	}

	/// Resizes the SMALL fast segment. The LARGE segment uses
	/// `resize_large_fast_tier` instead.
	fn resize_fast_tier(&mut self, size: CacheSize) {
		self.small_capacity = size;
		self.settle_small_fast();
	}

	fn resize_large_fast_tier(&mut self, size: CacheSize) {
		self.large_capacity = size;
		self.settle_large_fast();
	}

	fn resize_size_threshold(&mut self, size: CacheSize) {
		self.size_threshold = size;
	}

	fn drain_tier_migrations(&mut self) -> Vec<(HashedKey, Tier)> {
		std::mem::take(&mut self.migrations)
	}

	fn fast_bytes_used(&self) -> CacheSize {
		self.small_fast_used + self.large_fast_used
	}

	fn slow_bytes_used(&self) -> CacheSize {
		self.small_slow_used + self.large_slow_used
	}

	fn fast_object_count(&self) -> usize {
		self.small_fast_count + self.large_fast_count
	}

	fn slow_object_count(&self) -> usize {
		self.small_slow_count + self.large_slow_count
	}

	fn small_fast_bytes_used(&self) -> CacheSize {
		self.small_fast_used
	}

	fn large_fast_bytes_used(&self) -> CacheSize {
		self.large_fast_used
	}

	fn small_fast_object_count(&self) -> usize {
		self.small_fast_count
	}

	fn large_fast_object_count(&self) -> usize {
		self.large_fast_count
	}

	fn small_slow_bytes_used(&self) -> CacheSize {
		self.small_slow_used
	}

	fn large_slow_bytes_used(&self) -> CacheSize {
		self.large_slow_used
	}

	fn small_slow_object_count(&self) -> usize {
		self.small_slow_count
	}

	fn large_slow_object_count(&self) -> usize {
		self.large_slow_count
	}
}

#[cfg(test)]
mod tests {
	use super::*;

	fn drain(stack: &mut LruSizedHybridStack) -> Vec<(HashedKey, Tier)> {
		stack.drain_tier_migrations()
	}

	#[test]
	fn admission_routes_small_and_large_values_to_their_segments() {
		let mut stack = LruSizedHybridStack::new(1_000, 1_000, 20);

		stack.insert(1, 10); // small (10 < 20)
		stack.insert(2, 30); // large (30 >= 20)

		assert_eq!(stack.queue_of(1), Some(SizeQueue::SmallFast));
		assert_eq!(stack.queue_of(2), Some(SizeQueue::LargeFast));
		assert_eq!(stack.small_fast_bytes_used(), 10);
		assert_eq!(stack.large_fast_bytes_used(), 30);
	}

	#[test]
	fn small_segment_pressure_demotes_lru_tail_without_touching_large_segment() {
		let mut stack = LruSizedHybridStack::new(15, 1_000, 20);

		stack.insert(1, 10); // small: small_fast = [1]
		stack.insert(2, 10); // small: 20 > 15 -> demotes 1
		let migrations = drain(&mut stack);

		assert_eq!(migrations, vec![(1, Tier::Slow)]);
		assert_eq!(stack.tier_of(1), Some(Tier::Slow));
		assert_eq!(stack.tier_of(2), Some(Tier::Fast));

		stack.insert(3, 500); // large, fits comfortably -- large segment untouched
		let migrations = drain(&mut stack);

		assert!(migrations.is_empty());
		assert_eq!(stack.tier_of(3), Some(Tier::Fast));
	}

	#[test]
	fn large_segment_pressure_demotes_lru_tail_without_touching_small_segment() {
		let mut stack = LruSizedHybridStack::new(1_000, 25, 5);

		stack.insert(1, 10); // large (10 >= 5)
		stack.insert(2, 10); // large: 20 <= 25, no trigger yet
		stack.insert(3, 10); // large: 30 > 25 -> demotes 1 (LRU tail)
		let migrations = drain(&mut stack);

		assert_eq!(migrations, vec![(1, Tier::Slow)]);
		assert_eq!(stack.tier_of(1), Some(Tier::Slow));
		assert_eq!(stack.tier_of(2), Some(Tier::Fast));
		assert_eq!(stack.tier_of(3), Some(Tier::Fast));

		stack.insert(4, 1); // small, fits comfortably -- small segment untouched
		let migrations = drain(&mut stack);

		assert!(migrations.is_empty());
		assert_eq!(stack.tier_of(4), Some(Tier::Fast));
	}

	#[test]
	fn promotion_from_small_slow_routes_back_to_small_fast() {
		let mut stack = LruSizedHybridStack::new(15, 1_000, 20);

		stack.insert(1, 10);
		stack.insert(2, 10); // demotes 1 -> small_slow
		drain(&mut stack);
		assert_eq!(stack.tier_of(1), Some(Tier::Slow));

		stack.update(1); // promotes 1, which re-triggers small settle, demoting 2
		let migrations = drain(&mut stack);

		// Demotion before the promotion that triggered it, same ordering
		// invariant `LruHybridStack::touch_fast_key` documents.
		assert_eq!(migrations, vec![(2, Tier::Slow), (1, Tier::Fast)]);
		assert_eq!(stack.tier_of(1), Some(Tier::Fast));
		assert_eq!(stack.tier_of(2), Some(Tier::Slow));
	}

	#[test]
	fn promotion_from_large_slow_routes_back_to_large_fast() {
		let mut stack = LruSizedHybridStack::new(1_000, 25, 5);

		stack.insert(1, 10);
		stack.insert(2, 10);
		stack.insert(3, 10); // demotes 1 -> large_slow
		drain(&mut stack);
		assert_eq!(stack.tier_of(1), Some(Tier::Slow));

		stack.update(1); // promotes 1, re-triggers large settle, demoting 2
		let migrations = drain(&mut stack);

		assert_eq!(migrations, vec![(2, Tier::Slow), (1, Tier::Fast)]);
		assert_eq!(stack.tier_of(1), Some(Tier::Fast));
		assert_eq!(stack.tier_of(2), Some(Tier::Slow));
		assert_eq!(stack.tier_of(3), Some(Tier::Fast));
	}

	#[test]
	fn overwrite_reclassifies_into_the_other_fast_segment_without_a_migration() {
		let mut stack = LruSizedHybridStack::new(1_000, 1_000, 20);

		stack.insert(1, 10); // small
		drain(&mut stack);
		assert_eq!(stack.queue_of(1), Some(SizeQueue::SmallFast));

		stack.insert(1, 50); // overwrite with a larger value -> now large
		let migrations = drain(&mut stack);

		assert!(migrations.is_empty()); // fast<->fast never crosses Tier
		assert_eq!(stack.queue_of(1), Some(SizeQueue::LargeFast));
		assert_eq!(stack.small_fast_bytes_used(), 0);
		assert_eq!(stack.large_fast_bytes_used(), 50);
	}

	#[test]
	fn overwrite_within_same_segment_resizes_and_reorders() {
		let mut stack = LruSizedHybridStack::new(1_000, 1_000, 20);

		stack.insert(1, 10);
		drain(&mut stack);
		assert_eq!(stack.small_fast_bytes_used(), 10);

		stack.insert(1, 15); // still small (15 < 20), same segment
		assert_eq!(stack.queue_of(1), Some(SizeQueue::SmallFast));
		assert_eq!(stack.small_fast_bytes_used(), 15);
	}

	#[test]
	fn evict_one_prefers_the_slow_list_with_more_objects() {
		let mut stack = LruSizedHybridStack::new(0, 0, 20);

		stack.insert(1, 5);  // small -> demotes immediately to small_slow
		stack.insert(2, 5);  // small -> demotes immediately to small_slow
		stack.insert(3, 30); // large -> demotes immediately to large_slow
		drain(&mut stack);

		assert_eq!(stack.small_slow_object_count(), 2);
		assert_eq!(stack.large_slow_object_count(), 1);

		// small_slow has more objects (2 vs 1) -> its tail evicts first.
		assert_eq!(stack.evict_one(), Some(1));
	}

	#[test]
	fn evict_one_falls_back_to_large_when_both_slow_lists_are_empty_and_large_is_more_over_budget() {
		let mut stack = LruSizedHybridStack::new(100, 200, 20);

		stack.insert(1, 10); // small, ratio = 10/100 = 0.1
		stack.insert(2, 50); // large, ratio = 50/200 = 0.25

		assert_eq!(stack.small_slow_object_count(), 0);
		assert_eq!(stack.large_slow_object_count(), 0);

		assert_eq!(stack.evict_one(), Some(2));
	}

	#[test]
	fn evict_one_falls_back_to_small_when_both_slow_lists_are_empty_and_small_is_more_over_budget() {
		let mut stack = LruSizedHybridStack::new(20, 200, 5);

		stack.insert(1, 4);  // small, ratio = 4/20 = 0.2
		stack.insert(2, 10); // large, ratio = 10/200 = 0.05

		assert_eq!(stack.evict_one(), Some(1));
	}

	#[test]
	fn shared_overhead_reservation_splits_proportionally_between_segments() {
		// Equal capacities (100/100) -> an equal 40/40 split of the 80-byte
		// reservation (2 tracked objects x 40/object) is easy to verify:
		// effective_small = effective_large = 60.
		let mut stack = LruSizedHybridStack::new(100, 100, 50).with_shared_overhead(40);

		stack.insert(1, 10); // small, 10 <= 60, no trigger
		stack.insert(2, 90); // large, 90 > 60 -> triggers demotion
		let migrations = drain(&mut stack);

		assert_eq!(migrations, vec![(2, Tier::Slow)]);
		assert_eq!(stack.tier_of(1), Some(Tier::Fast));
		assert_eq!(stack.tier_of(2), Some(Tier::Slow));
	}

	#[test]
	fn resize_fast_tier_resizes_small_segment_only() {
		let mut stack = LruSizedHybridStack::new(1_000, 1_000, 20);

		stack.insert(1, 10);  // small
		stack.insert(2, 100); // large
		drain(&mut stack);

		stack.resize_fast_tier(5); // shrinks SMALL only
		let migrations = drain(&mut stack);

		assert_eq!(migrations, vec![(1, Tier::Slow)]);
		assert_eq!(stack.tier_of(2), Some(Tier::Fast));
	}

	#[test]
	fn resize_large_fast_tier_resizes_large_segment_only() {
		let mut stack = LruSizedHybridStack::new(1_000, 1_000, 20);

		stack.insert(1, 10);  // small
		stack.insert(2, 100); // large
		drain(&mut stack);

		stack.resize_large_fast_tier(5); // shrinks LARGE only
		let migrations = drain(&mut stack);

		assert_eq!(migrations, vec![(2, Tier::Slow)]);
		assert_eq!(stack.tier_of(1), Some(Tier::Fast));
	}

	#[test]
	fn resize_size_threshold_only_affects_future_classification() {
		let mut stack = LruSizedHybridStack::new(1_000, 1_000, 5);

		stack.insert(1, 10); // large (10 >= 5)
		assert_eq!(stack.queue_of(1), Some(SizeQueue::LargeFast));

		stack.resize_size_threshold(20);
		// Existing key is NOT retroactively reclassified.
		assert_eq!(stack.queue_of(1), Some(SizeQueue::LargeFast));

		// A brand-new admission with the same size now lands small.
		stack.insert(2, 10);
		assert_eq!(stack.queue_of(2), Some(SizeQueue::SmallFast));
	}

	#[test]
	fn zero_small_capacity_demotes_immediately_on_admission() {
		let mut stack = LruSizedHybridStack::new(0, 1_000, 20);

		stack.insert(1, 10); // small
		let migrations = drain(&mut stack);

		assert_eq!(migrations, vec![(1, Tier::Slow)]);
		assert_eq!(stack.tier_of(1), Some(Tier::Slow));
	}

	#[test]
	fn zero_large_capacity_demotes_immediately_on_admission() {
		let mut stack = LruSizedHybridStack::new(1_000, 0, 5);

		stack.insert(1, 10); // large
		let migrations = drain(&mut stack);

		assert_eq!(migrations, vec![(1, Tier::Slow)]);
		assert_eq!(stack.tier_of(1), Some(Tier::Slow));
	}

	#[test]
	fn remove_updates_counters_for_whichever_list_the_key_is_in() {
		let mut stack = LruSizedHybridStack::new(1_000, 1_000, 20);

		stack.insert(1, 10);  // small
		stack.insert(2, 100); // large
		drain(&mut stack);

		stack.remove(1);
		assert_eq!(stack.contains(1), false);
		assert_eq!(stack.small_fast_bytes_used(), 0);
		assert_eq!(stack.large_fast_bytes_used(), 100);

		stack.remove(2);
		assert_eq!(stack.large_fast_bytes_used(), 0);
		assert_eq!(stack.len(), 0);
	}

	#[test]
	fn clear_resets_all_four_lists_and_counters() {
		let mut stack = LruSizedHybridStack::new(15, 25, 20);

		stack.insert(1, 10);  // small
		stack.insert(2, 10);  // small, demotes 1
		stack.insert(3, 100); // large
		drain(&mut stack);

		stack.clear();

		assert_eq!(stack.len(), 0);
		assert_eq!(stack.fast_bytes_used(), 0);
		assert_eq!(stack.slow_bytes_used(), 0);
		assert_eq!(stack.tier_of(1), None);
		assert_eq!(stack.evict_one(), None);
	}

	#[test]
	fn combined_gauges_sum_both_segments() {
		let mut stack = LruSizedHybridStack::new(1_000, 1_000, 20);

		stack.insert(1, 10);  // small
		stack.insert(2, 100); // large
		drain(&mut stack);

		assert_eq!(stack.fast_bytes_used(), 110);
		assert_eq!(stack.fast_object_count(), 2);
		assert_eq!(stack.small_fast_bytes_used(), 10);
		assert_eq!(stack.large_fast_bytes_used(), 100);
	}
}
