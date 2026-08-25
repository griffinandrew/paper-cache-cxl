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
// allocated through the same crate-wide `Hybrid` alias (`SlowObjects`,
// jemalloc, NUMA node 1) that `BufferPMEM`/other PMEM features already use.
#[cfg(feature = "eviction_stacks_pmem")]
use crate::Hybrid;

use crate::{
	CacheSize,
	HashedKey,
	NoHasher,
	policy::PaperPolicy,
	object::ObjectSize,
	worker::policy::policy_stack::{PolicyStack, Tier, narrow_resident, watermarks},
};

/// SUPERSEDED by the shared `super::watermarks` high/low pair, and
/// deliberately no longer read by `settle_small_fast`/`settle_large_fast`.
/// Kept defined (and explicitly allowed to be dead) as the record of the
/// ratio this file used before the switch.
///
/// It was a flat 2% burst margin: `PaperCache::set()` writes new object bytes
/// to DRAM synchronously at the API layer before this stack (on the
/// background `PolicyWorker` thread) ever sees the corresponding event, so a
/// burst of concurrent `set()`s can transiently overshoot either segment's
/// own last-known bookkeeping between worker polls. `watermarks::low()`
/// subsumes that role, and `watermarks::high()` adds the trigger hysteresis
/// the flat ratio never had. Set `FAST_TIER_HIGH_WATERMARK=1.0` with
/// `FAST_TIER_LOW_WATERMARK=0.98` to reproduce the old behaviour exactly.
#[allow(dead_code)]
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

	/// Part of `size` that stays in DRAM in either tier; see `migrating`.
	dram_resident: u8,

	size: ObjectSize,
}

impl SizedEntry {
	/// The bytes that actually move between tiers when this object migrates.
	///
	/// Deliberately distinct from `size` (`base_size`), which remains the input
	/// to `classify`: the small/large split is a property of the whole object
	/// as the cache accounts for it, not of its value alone -- see the module
	/// doc's "Classification input" section. Only the byte counters change.
	#[inline]
	fn migrating(&self) -> CacheSize {
		(self.size as CacheSize).saturating_sub(self.dram_resident as CacheSize)
	}
}

/// `dram_resident` was meant to occupy padding the entry already had.
/// If this ever fails, the field is costing 4 more bytes on *every* tracked
/// object in both tiers, which defeats the point of storing it per entry.
const _: () = assert!(
	std::mem::size_of::<SizedEntry>() == 8,
	"SizedEntry grew past 8 bytes",
);

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

		let old_migrating = entry.migrating();
		entry.size = new_size;
		let delta = entry.migrating() as i64 - old_migrating as i64;
		let queue = entry.queue;

		match queue {
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

	fn remove_from_small_fast(&mut self, key: HashedKey, size: CacheSize) {
		self.small_fast.remove(&key);
		self.small_fast_used = self.small_fast_used.saturating_sub(size);
		self.small_fast_count = self.small_fast_count.saturating_sub(1);
	}

	fn remove_from_large_fast(&mut self, key: HashedKey, size: CacheSize) {
		self.large_fast.remove(&key);
		self.large_fast_used = self.large_fast_used.saturating_sub(size);
		self.large_fast_count = self.large_fast_count.saturating_sub(1);
	}

	fn remove_from_small_slow(&mut self, key: HashedKey, size: CacheSize) {
		self.small_slow.remove(&key);
		self.small_slow_used = self.small_slow_used.saturating_sub(size);
		self.small_slow_count = self.small_slow_count.saturating_sub(1);
	}

	fn remove_from_large_slow(&mut self, key: HashedKey, size: CacheSize) {
		self.large_slow.remove(&key);
		self.large_slow_used = self.large_slow_used.saturating_sub(size);
		self.large_slow_count = self.large_slow_count.saturating_sub(1);
	}

	fn add_to_small_fast(&mut self, key: HashedKey, size: CacheSize) {
		self.small_fast.push_front(key);
		self.small_fast_used += size;
		self.small_fast_count += 1;
	}

	fn add_to_large_fast(&mut self, key: HashedKey, size: CacheSize) {
		self.large_fast.push_front(key);
		self.large_fast_used += size;
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

			(SizeQueue::SmallFast, false) => self.remove_from_small_fast(key, entry.migrating()),
			(SizeQueue::LargeFast, true) => self.remove_from_large_fast(key, entry.migrating()),
			(SizeQueue::SmallSlow, _) => self.remove_from_small_slow(key, entry.migrating()),
			(SizeQueue::LargeSlow, _) => self.remove_from_large_slow(key, entry.migrating()),
		}

		let target_queue = if target_small { SizeQueue::SmallFast } else { SizeQueue::LargeFast };

		if target_small {
			self.add_to_small_fast(key, entry.migrating());
		} else {
			self.add_to_large_fast(key, entry.migrating());
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
	/// triggered only once `small_fast_used` crosses the shared *high*
	/// watermark of its effective budget, then drained in one pass down to
	/// the shared *low* watermark of that same budget — see
	/// `super::watermarks` for why the pair exists (draining to the exact
	/// ceiling pinned the segment at 100% and turned every subsequent
	/// admission into a one-object migration batch) and
	/// `LruHybridStack::settle_fast_tier` for the identical shape/rationale.
	///
	/// `effective_small()` — the configured capacity minus this segment's
	/// proportional share of the reserved shared-structure overhead — remains
	/// the budget in play: the watermarks scale that effective value, they
	/// never replace it. It is also loop-invariant here, since
	/// `reserved_shares()` counts *tracked* entries and a demotion only
	/// changes which list an entry is in, never whether it is tracked.
	fn settle_small_fast(&mut self) {
		let effective = self.effective_small();

		if self.small_fast_used <= watermarks::high_bytes(effective) {
			return;
		}

		let drain_target = watermarks::low_bytes(effective);

		while self.small_fast_used > drain_target {
			let Some(demote_key) = self.small_fast.pop_back() else { break };
			let size = self.entries.get(&demote_key).map(|entry| entry.migrating()).unwrap_or(0);

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
	/// `large_slow`. Same shared high/low watermark pair, taken against
	/// `effective_large()` instead.
	fn settle_large_fast(&mut self) {
		let effective = self.effective_large();

		if self.large_fast_used <= watermarks::high_bytes(effective) {
			return;
		}

		let drain_target = watermarks::low_bytes(effective);

		while self.large_fast_used > drain_target {
			let Some(demote_key) = self.large_fast.pop_back() else { break };
			let size = self.entries.get(&demote_key).map(|entry| entry.migrating()).unwrap_or(0);

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
			let size = self.entries.remove(&key).map(|entry| entry.migrating()).unwrap_or(0);
			self.small_fast_used = self.small_fast_used.saturating_sub(size);
			self.small_fast_count = self.small_fast_count.saturating_sub(1);
			Some(key)
		} else {
			let key = self.large_fast.pop_back()?;
			let size = self.entries.remove(&key).map(|entry| entry.migrating()).unwrap_or(0);
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
		self.insert_resident(key, size, 0);
	}

	fn insert_resident(&mut self, key: HashedKey, size: ObjectSize, dram_resident: ObjectSize) {
		let dram_resident = narrow_resident(dram_resident);
		let migrating = (size as CacheSize).saturating_sub(dram_resident as CacheSize);
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
			self.entries.insert(key, SizedEntry { queue: SizeQueue::SmallFast, dram_resident, size });
			self.small_fast_used += migrating;
			self.small_fast_count += 1;
			self.settle_small_fast();
		} else {
			self.large_fast.push_front(key);
			self.entries.insert(key, SizedEntry { queue: SizeQueue::LargeFast, dram_resident, size });
			self.large_fast_used += migrating;
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
				self.small_fast_used = self.small_fast_used.saturating_sub(entry.migrating());
				self.small_fast_count = self.small_fast_count.saturating_sub(1);
			},
			SizeQueue::LargeFast => {
				self.large_fast.remove(&key);
				self.large_fast_used = self.large_fast_used.saturating_sub(entry.migrating());
				self.large_fast_count = self.large_fast_count.saturating_sub(1);
			},
			SizeQueue::SmallSlow => {
				self.small_slow.remove(&key);
				self.small_slow_used = self.small_slow_used.saturating_sub(entry.migrating());
				self.small_slow_count = self.small_slow_count.saturating_sub(1);
			},
			SizeQueue::LargeSlow => {
				self.large_slow.remove(&key);
				self.large_slow_used = self.large_slow_used.saturating_sub(entry.migrating());
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
			let size = self.entries.remove(&key).map(|entry| entry.migrating()).unwrap_or(0);
			self.small_slow_used = self.small_slow_used.saturating_sub(size);
			self.small_slow_count = self.small_slow_count.saturating_sub(1);
			Some(key)
		} else {
			let key = self.large_slow.pop_back()?;
			let size = self.entries.remove(&key).map(|entry| entry.migrating()).unwrap_or(0);
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
		stack.insert(2, 10); // small: 20 > high_bytes(15) -> demotes 1
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
		let mut stack = LruSizedHybridStack::new(1_000, 30, 5);

		stack.insert(1, 10); // large (10 >= 5)
		stack.insert(2, 10); // large: 20 <= high_bytes(30) = 28, no trigger yet
		stack.insert(3, 10); // large: 30 > 28 -> drains to low_bytes(30) = 22, demoting 1
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
		let mut stack = LruSizedHybridStack::new(1_000, 30, 5);

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

		stack.insert(1, 10); // small, 10 <= high_bytes(80), no trigger
		stack.insert(2, 90); // large, 90 > high_bytes(60) -> triggers demotion
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

	// ---- Shared fast-tier high/low watermarks (see `super::watermarks`) ----
	//
	// Written against `watermarks::high_bytes`/`low_bytes` rather than
	// hard-coded byte counts, so they hold at whatever
	// `FAST_TIER_HIGH_WATERMARK`/`FAST_TIER_LOW_WATERMARK` the process is
	// configured with. Deliberately NO `std::env::set_var` in here: both
	// ratios are cached in a process-wide `OnceLock` on first read, so a test
	// that set them would race every other test in the same binary.

	/// Byte budget the watermark tests below size their segments against.
	/// Large enough that `high_bytes` and `low_bytes` stay many 10-byte
	/// objects apart at any sane ratio.
	const WATERMARK_CAPACITY: CacheSize = 10_000;

	/// Admits 10-byte SMALL objects under keys `1..` until one more would
	/// push `small_fast_used` past `target`. Returns the number admitted
	/// (which is also the highest key used).
	fn fill_small_to(stack: &mut LruSizedHybridStack, target: CacheSize) -> HashedKey {
		let mut key: HashedKey = 0;

		while stack.small_fast_bytes_used() + 10 <= target {
			key += 1;
			stack.insert(key, 10);
		}

		key
	}

	#[test]
	fn usage_up_to_the_high_watermark_triggers_no_demotion() {
		let mut stack = LruSizedHybridStack::new(WATERMARK_CAPACITY, WATERMARK_CAPACITY, 20);
		let high = watermarks::high_bytes(WATERMARK_CAPACITY);
		let admitted = fill_small_to(&mut stack, high);

		assert!(admitted > 0);
		assert!(stack.small_fast_bytes_used() <= high);

		// The trigger is strictly `>`, so sitting at (or just under) the high
		// watermark must stay completely quiet.
		assert!(drain(&mut stack).is_empty());
		assert_eq!(stack.small_slow_object_count(), 0);
		assert_eq!(stack.small_slow_bytes_used(), 0);
		assert_eq!(stack.small_fast_object_count(), admitted as usize);
	}

	#[test]
	fn usage_above_the_high_watermark_triggers_a_demotion_pass() {
		let mut stack = LruSizedHybridStack::new(WATERMARK_CAPACITY, WATERMARK_CAPACITY, 20);
		let admitted = fill_small_to(&mut stack, watermarks::high_bytes(WATERMARK_CAPACITY));

		assert!(drain(&mut stack).is_empty());

		// `fill_small_to` stops one object short of crossing, so this
		// admission puts usage strictly above the high watermark; the pass
		// runs synchronously inside `insert`.
		stack.insert(admitted + 1, 10);
		let migrations = drain(&mut stack);

		assert!(!migrations.is_empty(), "crossing the high watermark must trigger a pass");
		assert!(migrations.iter().all(|(_, tier)| *tier == Tier::Slow));
		// A pass always drains from the LRU end, so key 1 goes first.
		assert_eq!(migrations.first(), Some(&(1, Tier::Slow)));
	}

	#[test]
	fn a_triggered_pass_drains_to_the_low_watermark_not_merely_to_the_ceiling() {
		let high = watermarks::high_bytes(WATERMARK_CAPACITY);
		let low = watermarks::low_bytes(WATERMARK_CAPACITY);

		let mut stack = LruSizedHybridStack::new(WATERMARK_CAPACITY, WATERMARK_CAPACITY, 20);
		let admitted = fill_small_to(&mut stack, high);
		drain(&mut stack);

		stack.insert(admitted + 1, 10);
		let migrations = drain(&mut stack);

		assert!(stack.small_fast_bytes_used() <= low);

		// Not merely back under the ceiling: unless the low watermark is
		// configured at 1.0, the segment ends the pass strictly below its
		// full effective budget.
		if low < WATERMARK_CAPACITY {
			assert!(stack.small_fast_bytes_used() < WATERMARK_CAPACITY);
		}

		// Which is the whole point of the pair: one pass demotes a batch, not
		// just the single object that happened to cross the line.
		if low + 10 < high {
			assert!(migrations.len() > 1);
		}
	}

	#[test]
	fn counters_stay_consistent_after_a_watermark_triggered_pass() {
		let mut stack = LruSizedHybridStack::new(WATERMARK_CAPACITY, WATERMARK_CAPACITY, 20);
		let admitted = fill_small_to(&mut stack, watermarks::high_bytes(WATERMARK_CAPACITY));

		stack.insert(admitted + 1, 10);

		let total = (admitted + 1) as usize;
		let migrations = drain(&mut stack);

		assert!(!migrations.is_empty());

		// Nothing lost, nothing double-counted: every admitted key is still
		// tracked, in exactly one of the two SMALL lists.
		assert_eq!(stack.len(), total);
		assert_eq!(stack.small_fast_object_count() + stack.small_slow_object_count(), total);
		assert_eq!(
			stack.small_fast_bytes_used() + stack.small_slow_bytes_used(),
			total as CacheSize * 10,
		);

		// The per-demotion bookkeeping ran exactly once per migrated object.
		assert_eq!(migrations.len(), stack.small_slow_object_count());
		assert_eq!(stack.small_slow_bytes_used(), migrations.len() as CacheSize * 10);

		// The LARGE segment was never touched.
		assert_eq!(stack.large_fast_bytes_used(), 0);
		assert_eq!(stack.large_slow_bytes_used(), 0);
		assert_eq!(stack.large_fast_object_count(), 0);
		assert_eq!(stack.large_slow_object_count(), 0);

		// Combined gauges still agree with the per-segment ones.
		assert_eq!(stack.fast_bytes_used(), stack.small_fast_bytes_used());
		assert_eq!(stack.slow_bytes_used(), stack.small_slow_bytes_used());
		assert_eq!(stack.fast_object_count(), stack.small_fast_object_count());
		assert_eq!(stack.slow_object_count(), stack.small_slow_object_count());
	}

	#[test]
	fn large_segment_uses_the_same_watermarks_against_its_own_budget() {
		let high = watermarks::high_bytes(WATERMARK_CAPACITY);
		let low = watermarks::low_bytes(WATERMARK_CAPACITY);

		// Threshold 5 => every 10-byte object classifies LARGE.
		let mut stack = LruSizedHybridStack::new(WATERMARK_CAPACITY, WATERMARK_CAPACITY, 5);
		let mut key: HashedKey = 0;

		while stack.large_fast_bytes_used() + 10 <= high {
			key += 1;
			stack.insert(key, 10);
		}

		assert!(drain(&mut stack).is_empty());
		assert_eq!(stack.large_slow_object_count(), 0);

		stack.insert(key + 1, 10);
		let migrations = drain(&mut stack);

		assert!(!migrations.is_empty());
		assert_eq!(migrations.first(), Some(&(1, Tier::Slow)));
		assert!(stack.large_fast_bytes_used() <= low);
		assert_eq!(migrations.len(), stack.large_slow_object_count());
		assert_eq!(
			stack.large_fast_bytes_used() + stack.large_slow_bytes_used(),
			(key + 1) as CacheSize * 10,
		);
		assert_eq!(stack.small_fast_bytes_used(), 0);
		assert_eq!(stack.small_slow_object_count(), 0);
	}

	#[test]
	fn watermarks_scale_the_effective_budget_not_the_raw_capacity() {
		// 1_000/1_000 capacities with a 400-byte/object shared reservation:
		// at two tracked objects that is 800 bytes, split 400/400, so each
		// segment's *effective* budget is 600 -- the value the watermarks
		// have to be taken against.
		let mut stack = LruSizedHybridStack::new(1_000, 1_000, 50).with_shared_overhead(400);

		stack.insert(1, 10); // small, nowhere near its own watermark
		assert!(drain(&mut stack).is_empty());

		// One byte over the high watermark of the EFFECTIVE budget, but
		// comfortably under the high watermark of the raw capacity -- so a
		// demotion here can only come from the reservation being preserved.
		let size = watermarks::high_bytes(600) + 1;
		assert!(size <= watermarks::high_bytes(1_000));

		stack.insert(2, size as ObjectSize);
		let migrations = drain(&mut stack);

		assert_eq!(stack.effective_large(), 600);
		assert_eq!(migrations, vec![(2, Tier::Slow)]);
		assert_eq!(stack.tier_of(1), Some(Tier::Fast));
		assert_eq!(stack.tier_of(2), Some(Tier::Slow));
		assert_eq!(stack.large_fast_bytes_used(), 0);
		assert_eq!(stack.large_slow_bytes_used(), size);
	}
}
