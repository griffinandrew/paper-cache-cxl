/*
 * Copyright (c) Kia Shakiba
 *
 * This source code is licensed under the GNU AGPLv3 license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! `LfuHybridStack` — a frequency-segmented LFU stack for `PaperPolicy::LfuHybrid`.
//!
//! Two independent frequency-bucket chains (`FrequencyChain`, an adapted
//! copy of `LfuStack`'s classic O(1) LFU structure — see that file) back the
//! fast and slow tiers respectively. A key lives in exactly one chain at a
//! time; `entries` records which tier (and its size). Unlike `LruHybridStack`
//! (one shared recency list, fast/slow split by list position), LFU's
//! fast/slow boundary is a *frequency* threshold, not a position, so two
//! chains — each queryable for its own minimum frequency in O(1) — are the
//! natural fit.
//!
//! Admission checks fast-tier capacity directly, matching the paper's
//! admission rule literally: while the fast tier has room, a brand-new key
//! is admitted there at frequency 1; once `fast_used` would exceed
//! `fast_capacity`, every new key goes straight to the slow chain instead
//! — it never touches the fast chain and never displaces an existing fast
//! resident. (An earlier "always admit fast, let settle demote if needed"
//! design relied on ties-within-a-frequency-bucket breaking
//! LRU-within-frequency to decide who gets demoted once the fast tier was
//! full — but that could demote an *older* resident instead of the
//! newcomer, which is not what the paper specifies: "every new object is
//! admitted into the slow tier" means the new object specifically, not
//! whichever key loses a tie-break.) `lib.rs`'s `set()` still always
//! synchronously builds `TieredBuffer::new_fast` for a brand-new key
//! regardless of which tier admission ultimately assigns it — when
//! admission decides slow, that disagreement is exactly what produces a
//! `(key, Tier::Slow)` migration, physically corrected by
//! `PolicyWorker::apply_tier_migrations` the same way a demotion is.
//!
//! ## Admission latches shut once the fast tier has ever reached capacity
//!
//! A raw byte-capacity check alone (`fast_used + size <= admit_effective`)
//! is not sufficient to keep admission honoring frequency order over time,
//! because `settle_fast_tier`'s demotion granularity is per-*object*, not
//! per-*byte*: demoting one lowest-frequency fast object to cover a small
//! promotion overage can free far more bytes than the overage itself (e.g.
//! demoting a 90-byte object to cover a 5-byte overage leaves 85 bytes of
//! slack). A brand-new, frequency-1 key admitted purely because that slack
//! exists bypasses the "prove yourself via promotion from slow" path
//! entirely — even when every current fast resident already has frequency
//! ≥ 2 — which does not honor LFU ordering.
//!
//! To close this, `fast_tier_latched: bool` permanently closes brand-new-key
//! admission to the fast tier the first time capacity is genuinely reached
//! (a new admission that doesn't fit, or a demotion firing inside
//! `settle_fast_tier`): once latched, *every* subsequent brand-new key goes
//! straight to slow regardless of any later byte slack, and can only reach
//! the fast tier by earning a promotion. The latch resets on `clear()`
//! (full state reset) and on `resize_fast_tier` *growing* the budget (a
//! deliberate capacity increase should be immediately usable by new
//! admissions, not gated behind promotions).
//!
//! A previously-considered alternative — gating admission on whether the
//! fast chain's *current minimum frequency* is still 1, rather than a
//! one-time latch — was rejected for now: it still permits a brand-new key
//! to ride in alongside any surviving untouched frequency-1 resident, which
//! does not fully honor LFU order either. Left as a documented option for a
//! future revision if the one-time latch proves too coarse in practice.
//!
//! A slow-tier access (`update`) bumps that key's frequency in the slow
//! chain; if the new count strictly exceeds the fast chain's current
//! minimum (or the fast chain is empty), the key is promoted — moved to the
//! fast chain, preserving its accumulated frequency via `insert_at` — which
//! may itself trigger `settle_fast_tier` to demote the (new) fast minimum.
//!
//! Unlike `LruHybridStack`, `settle_fast_tier` here drains exactly back to
//! `fast_capacity` (no low-water headroom): demotion pressure in this stack
//! is only triggered by a promotion or an explicit `resize_fast_tier`, not
//! by every admission, so the thrashing concern that motivated LRU-hybrid's
//! headroom largely doesn't apply.
//!
//! ## One combined per-key map, not two
//!
//! Every tracked key needs both a tier and a size, and nearly every
//! operation here touches both together. An earlier version kept these in
//! two separate maps (`tiers`, `sizes`); they're now one
//! `entries: HashMap<HashedKey, LfuEntry>` (`LfuEntry { tier, size }`),
//! matching `LruHybridStack`'s and `TwoQHybridStack`'s equivalent
//! consolidation. This removes one of the two hashtable-structural-overhead
//! charges per tracked object (see `object/overhead.rs`'s `LfuHybrid` arm)
//! and removes the possibility of a key being present in one map but not the
//! other by construction. (`fast_chain`/`slow_chain` are unrelated to this
//! consolidation — they're the frequency-ordered structures, not per-key
//! tier/size bookkeeping.)

#[cfg(not(feature = "eviction_stacks_pmem"))]
use std::collections::HashMap;
#[cfg(feature = "eviction_stacks_pmem")]
use hashbrown::HashMap;

#[cfg(not(feature = "eviction_stacks_pmem"))]
use dlv_list::{VecList, Index};
#[cfg(not(feature = "eviction_stacks_pmem"))]
use kwik::collections::HashList;

#[cfg(feature = "eviction_stacks_pmem")]
use super::pmem_collections::{PmemVecList, PmemHashList, PmemIndex};

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

// The two frequency-bucket chains and the per-key map are DRAM-backed by
// default. When `eviction_stacks_pmem` is enabled, they are instead allocated
// in the slow tier (PMEM, via `crate::Hybrid`), mirroring how plain `LfuStack`
// switches to `PmemVecList`/`PmemHashList` under that flag. The PMEM and DRAM
// collection variants share a method surface, so the stack logic below is
// identical for both. Only the transient `migrations`/`pending_demotions`
// scratch and scalar counters stay in DRAM.
#[cfg(not(feature = "eviction_stacks_pmem"))]
type ChainIndex = Index<CountStack>;
#[cfg(feature = "eviction_stacks_pmem")]
type ChainIndex = PmemIndex;

/// Combined per-key bookkeeping: tier and size. See the module doc's "One
/// combined per-key map" section for why this replaced two separate maps.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct LfuEntry {
	tier: Tier,
	size: ObjectSize,
}

#[cfg(not(feature = "eviction_stacks_pmem"))]
type EntryMap = HashMap<HashedKey, LfuEntry, NoHasher>;
#[cfg(feature = "eviction_stacks_pmem")]
type EntryMap = HashMap<HashedKey, LfuEntry, NoHasher, Hybrid>;

/// A classic O(1) LFU frequency-bucket chain: an ascending-by-count linked
/// list of `CountStack` buckets (each holding every key at that exact
/// frequency, itself recency-ordered so ties break LRU-within-frequency),
/// plus an index from key to its current bucket. Adapted from `LfuStack`
/// (`lfu_stack.rs`) with one addition needed here but not there: `insert_at`,
/// which places a key at an *arbitrary* existing count rather than always
/// starting at 1 or advancing by exactly 1 — needed when a key crosses from
/// the other chain carrying its already-accumulated frequency.
#[cfg(not(feature = "eviction_stacks_pmem"))]
#[derive(Default)]
struct FrequencyChain {
	index_map: HashMap<HashedKey, ChainIndex, NoHasher>,
	count_stacks: VecList<CountStack>,
}

#[cfg(feature = "eviction_stacks_pmem")]
struct FrequencyChain {
	index_map: HashMap<HashedKey, ChainIndex, NoHasher, Hybrid>,
	count_stacks: PmemVecList<CountStack>,
}

#[cfg(feature = "eviction_stacks_pmem")]
impl Default for FrequencyChain {
	fn default() -> Self {
		FrequencyChain {
			index_map: HashMap::with_hasher_in(NoHasher::default(), Hybrid),
			count_stacks: PmemVecList::new(),
		}
	}
}

#[cfg(not(feature = "eviction_stacks_pmem"))]
struct CountStack {
	count: u32,
	stack: HashList<HashedKey, NoHasher>,
}

#[cfg(feature = "eviction_stacks_pmem")]
struct CountStack {
	count: u32,
	stack: PmemHashList<HashedKey, NoHasher>,
}

impl CountStack {
	#[cfg(not(feature = "eviction_stacks_pmem"))]
	fn new(count: u32) -> Self {
		CountStack {
			count,
			stack: HashList::with_hasher(NoHasher::default()),
		}
	}

	#[cfg(feature = "eviction_stacks_pmem")]
	fn new(count: u32) -> Self {
		CountStack {
			count,
			stack: PmemHashList::with_hasher(NoHasher::default()),
		}
	}

	fn is_empty(&self) -> bool {
		self.stack.is_empty()
	}

	fn push(&mut self, key: HashedKey) {
		self.stack.push_front(key);
	}

	fn pop(&mut self) -> HashedKey {
		self.stack.pop_back().unwrap()
	}

	fn remove(&mut self, key: HashedKey) {
		self.stack.remove(&key).unwrap();
	}
}

impl FrequencyChain {
	fn len(&self) -> usize {
		self.index_map.len()
	}

	/// The lowest frequency currently present in this chain, or `None` if
	/// the chain is empty. O(1) — just the head bucket's count.
	fn min_count(&self) -> Option<u32> {
		self.count_stacks.front().map(|count_stack| count_stack.count)
	}

	/// Inserts a brand-new key at frequency 1. Mirrors `LfuStack::insert`'s
	/// new-key branch. Returns the assigned count (always 1).
	fn insert_new(&mut self, key: HashedKey) -> u32 {
		if self.count_stacks.front().is_none_or(|count_stack| count_stack.count != 1) {
			self.count_stacks.push_front(CountStack::new(1));
		}

		let count_stack_index = self.count_stacks.front_index().unwrap();
		let count_stack = self.count_stacks.get_mut(count_stack_index).unwrap();

		count_stack.push(key);
		self.index_map.insert(key, count_stack_index);

		1
	}

	/// Moves an already-tracked key to the next-higher frequency bucket.
	/// Mirrors `LfuStack::update`. Returns the new count, or `0` if the key
	/// isn't tracked by this chain (callers are expected to only bump keys
	/// they know are present).
	fn bump(&mut self, key: HashedKey) -> u32 {
		let Some(count_stack_index) = self.index_map.get(&key).copied() else {
			return 0;
		};

		let prev_count_stack = self.count_stacks.get_mut(count_stack_index).unwrap();
		let prev_count = prev_count_stack.count;

		prev_count_stack.remove(key);
		let prev_is_empty = prev_count_stack.is_empty();

		if let Some(next_count_stack_index) = self.count_stacks.get_next_index(count_stack_index) {
			let next_count_stack = self.count_stacks.get_mut(next_count_stack_index).unwrap();

			if next_count_stack.count == prev_count + 1 {
				next_count_stack.push(key);
				self.index_map.insert(key, next_count_stack_index);

				if prev_is_empty {
					self.count_stacks.remove(count_stack_index);
				}

				return prev_count + 1;
			}
		}

		let mut new_count_stack = CountStack::new(prev_count + 1);
		new_count_stack.push(key);

		let new_count_stack_index = self.count_stacks.insert_after(count_stack_index, new_count_stack);
		self.index_map.insert(key, new_count_stack_index);

		if prev_is_empty {
			self.count_stacks.remove(count_stack_index);
		}

		prev_count + 1
	}

	/// Places `key` directly into the bucket for an arbitrary existing
	/// `count`, creating that bucket (in sorted position) if it doesn't
	/// already exist. Needed when a promoted/demoted key crosses chains
	/// carrying an accumulated frequency that may not be adjacent to
	/// anything already in this chain — unlike `bump`'s O(1) adjacent-bucket
	/// check, this requires a linear scan to find or create the correctly
	/// sorted bucket. Accepted as O(distinct frequencies in this chain);
	/// expected small in practice since the fast tier is DRAM-budget-limited.
	fn insert_at(&mut self, key: HashedKey, count: u32) {
		let mut cursor = self.count_stacks.front_index();

		while let Some(index) = cursor {
			let count_stack = self.count_stacks.get(index).unwrap();

			if count_stack.count == count {
				let count_stack = self.count_stacks.get_mut(index).unwrap();
				count_stack.push(key);
				self.index_map.insert(key, index);
				return;
			}

			if count_stack.count > count {
				let mut new_count_stack = CountStack::new(count);
				new_count_stack.push(key);

				let new_index = self.count_stacks.insert_before(index, new_count_stack);
				self.index_map.insert(key, new_index);
				return;
			}

			cursor = self.count_stacks.get_next_index(index);
		}

		let mut new_count_stack = CountStack::new(count);
		new_count_stack.push(key);

		let new_index = self.count_stacks.push_back(new_count_stack);
		self.index_map.insert(key, new_index);
	}

	fn remove(&mut self, key: HashedKey) {
		let Some(count_stack_index) = self.index_map.remove(&key) else {
			return;
		};

		let count_stack = self.count_stacks.get_mut(count_stack_index).unwrap();
		count_stack.remove(key);

		if count_stack.is_empty() {
			self.count_stacks.remove(count_stack_index);
		}
	}

	/// Removes and returns the lowest-frequency key (ties broken LRU-within-
	/// frequency) along with the count it held, or `None` if the chain is
	/// empty. Mirrors `LfuStack::evict_one`.
	fn pop_min(&mut self) -> Option<(HashedKey, u32)> {
		let count_stack_index = self.count_stacks.front_index()?;
		let count_stack = self.count_stacks.get_mut(count_stack_index)?;

		let key = count_stack.pop();
		let count = count_stack.count;

		self.index_map.remove(&key);

		if count_stack.is_empty() {
			self.count_stacks.remove(count_stack_index);
		}

		Some((key, count))
	}

	fn clear(&mut self) {
		self.index_map.clear();
		self.count_stacks.clear();
	}
}

pub struct LfuHybridStack {
	fast_chain: FrequencyChain,
	slow_chain: FrequencyChain,

	entries: EntryMap,

	fast_capacity: CacheSize,
	fast_used: CacheSize,
	slow_used: CacheSize,

	/// Approximate per-object DRAM cost of the shared structures (object
	/// hashtable + eviction stacks) that hold an entry for every object of
	/// both tiers. Reserved out of `fast_capacity` in `settle_fast_tier` so
	/// the fast-tier budget bounds total DRAM (values + shared metadata), not
	/// just fast-tier values. `0` unless set via `with_shared_overhead`.
	shared_overhead: CacheSize,

	/// (key, new tier) pairs recorded since the last `drain_tier_migrations`.
	/// `Tier::Slow` entries here can be either a genuine demotion (via
	/// `settle_fast_tier`) or a fresh admission routed directly to slow
	/// (via `insert`, once the fast tier was already full) — both need the
	/// same physical `Object::set_data` correction, but only the former
	/// should count toward `demotions`; see `pending_demotions`.
	migrations: Vec<(HashedKey, Tier)>,

	/// Count of genuine `settle_fast_tier` demotions since the last
	/// `drain_demotions` — kept separate from `migrations.len()` /
	/// `Tier::Slow` entries specifically so a fresh admission-to-slow isn't
	/// miscounted as a demotion (see `drain_demotions`'s trait doc).
	pending_demotions: u64,

	/// Once `true`, every brand-new key is admitted straight to slow,
	/// regardless of any leftover fast-tier byte slack. Set the first time
	/// fast-tier capacity is genuinely reached (a failed admission or a
	/// `settle_fast_tier` demotion); reset by `clear()` and by
	/// `resize_fast_tier` growing the budget. See the module doc for why a
	/// one-time latch is needed on top of the raw byte check.
	fast_tier_latched: bool,
}

impl LfuHybridStack {
	/// Constructs the entry map, DRAM- or PMEM-backed depending on
	/// `eviction_stacks_pmem`.
	#[cfg(not(feature = "eviction_stacks_pmem"))]
	fn new_collections() -> EntryMap {
		HashMap::default()
	}

	#[cfg(feature = "eviction_stacks_pmem")]
	fn new_collections() -> EntryMap {
		HashMap::with_hasher_in(NoHasher::default(), Hybrid)
	}

	pub fn new(fast_capacity: CacheSize) -> Self {
		LfuHybridStack {
			fast_chain: FrequencyChain::default(),
			slow_chain: FrequencyChain::default(),

			entries: Self::new_collections(),

			fast_capacity,
			fast_used: 0,
			slow_used: 0,
			shared_overhead: 0,

			migrations: Vec::new(),
			pending_demotions: 0,
			fast_tier_latched: false,
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
		self.entries.len() as CacheSize * self.shared_overhead
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

	/// Bumps an already-slow-tier key's frequency and, if the new count
	/// strictly exceeds the fast chain's current minimum (or the fast chain
	/// is empty), promotes it — moving it to the fast chain at its new,
	/// accumulated count. A tie does *not* promote (spec: "exceeds"), which
	/// avoids promote/demote ping-pong between equal-frequency neighbors.
	///
	/// Returns `Some(key)` if it promoted (the caller is responsible for
	/// calling `settle_fast_tier` and then pushing this key's own `Fast`
	/// migration entry *afterward* — see `maybe_promote`'s callers for why:
	/// pushing it here, before the caller's `settle_fast_tier` call, would
	/// let a promotion that pushes `fast_used` over budget have its DRAM
	/// allocation physically applied before the corresponding demotion's
	/// DRAM free, since `apply_tier_migrations` applies a batch in push
	/// order).
	fn maybe_promote(&mut self, key: HashedKey) -> Option<HashedKey> {
		let new_count = self.slow_chain.bump(key);
		let fast_min = self.fast_chain.min_count();

		let should_promote = match fast_min {
			None => true,
			Some(min) => new_count > min,
		};

		if !should_promote {
			return None;
		}

		let size = self.entries.get(&key).map(|entry| entry.size).unwrap_or(0) as CacheSize;

		self.slow_chain.remove(key);
		self.slow_used = self.slow_used.saturating_sub(size);

		self.fast_chain.insert_at(key, new_count);
		if let Some(entry) = self.entries.get_mut(&key) {
			entry.tier = Tier::Fast;
		}
		self.fast_used += size;

		Some(key)
	}

	/// Demotes the lowest-frequency fast key(s) until `fast_used` fits back
	/// within the *effective* value budget: `fast_capacity` minus the DRAM
	/// reserved for shared per-object metadata (hashtable + eviction stacks)
	/// across both tiers, so the fast-tier budget bounds total DRAM, not just
	/// fast-tier values (when the shared metadata alone meets/exceeds
	/// `fast_capacity` the effective budget saturates to 0, draining every fast
	/// value to slow). Demotion is the only response; the DRAM budget never
	/// evicts (terminal eviction stays governed solely by `max_size`). No
	/// low-water floor — see the module doc.
	fn settle_fast_tier(&mut self) {
		let effective = self.fast_capacity.saturating_sub(self.reserved_overhead());

		while self.fast_used > effective {
			let Some((demote_key, count)) = self.fast_chain.pop_min() else { break };

			let size = self.entries.get(&demote_key).map(|entry| entry.size).unwrap_or(0) as CacheSize;

			self.slow_chain.insert_at(demote_key, count);
			if let Some(entry) = self.entries.get_mut(&demote_key) {
				entry.tier = Tier::Slow;
			}

			self.fast_used = self.fast_used.saturating_sub(size);
			self.slow_used += size;

			self.migrations.push((demote_key, Tier::Slow));
			self.pending_demotions += 1;

			// A demotion firing at all means fast-tier capacity was
			// genuinely reached — latch shut brand-new admission (see the
			// module doc).
			self.fast_tier_latched = true;
		}
	}
}

impl PolicyStack for LfuHybridStack {
	fn is_policy(&self, policy: &PaperPolicy) -> bool {
		matches!(policy, PaperPolicy::LfuHybrid)
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
			// (matches `LfuStack::insert`'s existing-key delegation to
			// `update`).
			self.resize_key(key, size);

			let promoted_key = match self.entries.get(&key).map(|entry| entry.tier) {
				Some(Tier::Fast) => {
					self.fast_chain.bump(key);
					None
				},

				Some(Tier::Slow) => self.maybe_promote(key),

				None => None,
			};

			self.settle_fast_tier();

			// Pushed after `settle_fast_tier` -- see `maybe_promote`'s doc.
			// Guarded on the key still being `Fast`: an extremely tight
			// budget can demote it straight back out within the same
			// `settle_fast_tier` call (self-eviction), in which case that
			// call already pushed the correct final `(key, Tier::Slow)`
			// entry and no separate `Fast` entry should follow it.
			if let Some(k) = promoted_key {
				if self.entries.get(&k).map(|entry| entry.tier) == Some(Tier::Fast) {
					self.migrations.push((k, Tier::Fast));
				}
			}
			return;
		}

		// Brand-new key: admitted to the fast chain only while there's room
		// *and* the fast tier has never yet reached capacity — see the
		// module doc for why a one-time latch is needed on top of the raw
		// byte check (byte slack left over from an object-granular demotion
		// would otherwise let a frequency-1 newcomer bypass promotion).
		if self.fast_tier_latched {
			self.slow_chain.insert_new(key);
			self.entries.insert(key, LfuEntry { tier: Tier::Slow, size });
			self.slow_used += size as CacheSize;

			self.migrations.push((key, Tier::Slow));
			return;
		}

		// The capacity checked is the *effective* value budget (`fast_capacity`
		// minus the DRAM reserved for shared per-object metadata), so admission
		// honors the same total-DRAM bound as demotion. The `+ 1` reserves for
		// the new object's own shared metadata, which is DRAM-resident whether
		// it lands fast or slow. (Unlike `LruHybridStack`, LFU doesn't call
		// `settle_fast_tier` on a fresh admission, so the budget check has to
		// happen here.)
		let admit_effective = self.fast_capacity
			.saturating_sub((self.entries.len() as CacheSize + 1) * self.shared_overhead);

		if self.fast_used + size as CacheSize <= admit_effective {
			self.fast_chain.insert_new(key);
			self.entries.insert(key, LfuEntry { tier: Tier::Fast, size });
			self.fast_used += size as CacheSize;
		} else {
			self.slow_chain.insert_new(key);
			self.entries.insert(key, LfuEntry { tier: Tier::Slow, size });
			self.slow_used += size as CacheSize;

			self.migrations.push((key, Tier::Slow));

			// Capacity was genuinely reached — latch shut for all future
			// brand-new admissions (see the module doc).
			self.fast_tier_latched = true;
		}
	}

	fn update(&mut self, key: HashedKey) {
		match self.entries.get(&key).map(|entry| entry.tier) {
			Some(Tier::Fast) => {
				self.fast_chain.bump(key);
			},

			Some(Tier::Slow) => {
				let promoted_key = self.maybe_promote(key);
				self.settle_fast_tier();

				// See `maybe_promote`'s doc for why this is pushed after
				// `settle_fast_tier`, and why it's guarded on the key still
				// being `Fast` (self-eviction case).
				if let Some(k) = promoted_key {
					if self.entries.get(&k).map(|entry| entry.tier) == Some(Tier::Fast) {
						self.migrations.push((k, Tier::Fast));
					}
				}
			},

			None => {},
		}
	}

	fn remove(&mut self, key: HashedKey) {
		let entry = self.entries.remove(&key);
		let size = entry.map(|entry| entry.size).unwrap_or(0) as CacheSize;

		match entry.map(|entry| entry.tier) {
			Some(Tier::Fast) => {
				self.fast_chain.remove(key);
				self.fast_used = self.fast_used.saturating_sub(size);
			},

			Some(Tier::Slow) => {
				self.slow_chain.remove(key);
				self.slow_used = self.slow_used.saturating_sub(size);
			},

			None => {},
		}
	}

	fn clear(&mut self) {
		self.fast_chain.clear();
		self.slow_chain.clear();
		self.entries.clear();

		self.fast_used = 0;
		self.slow_used = 0;
		self.migrations.clear();
		self.pending_demotions = 0;
		self.fast_tier_latched = false;
	}

	fn evict_one(&mut self) -> Option<HashedKey> {
		if let Some((key, _count)) = self.slow_chain.pop_min() {
			let size = self.entries.remove(&key).map(|entry| entry.size).unwrap_or(0) as CacheSize;
			self.slow_used = self.slow_used.saturating_sub(size);
			return Some(key);
		}

		// Slow chain empty (e.g. `fast_capacity == max_size`, so nothing has
		// ever been demoted): fall back to the fast chain's minimum, mirrors
		// `LruHybridStack::evict_one`'s fallback for the same situation.
		let (key, _count) = self.fast_chain.pop_min()?;
		let size = self.entries.remove(&key).map(|entry| entry.size).unwrap_or(0) as CacheSize;
		self.fast_used = self.fast_used.saturating_sub(size);

		Some(key)
	}

	fn resize_fast_tier(&mut self, size: CacheSize) {
		// Growing the budget is a deliberate decision to make more capacity
		// available; the fresh room should be immediately usable by new
		// admissions rather than gated behind promotions, so unlatch. A
		// shrink (or no-op resize) leaves the latch as-is — `settle_fast_tier`
		// below will naturally re-latch it if the shrink forces a demotion.
		if size > self.fast_capacity {
			self.fast_tier_latched = false;
		}

		self.fast_capacity = size;
		self.settle_fast_tier();
	}

	fn drain_tier_migrations(&mut self) -> Vec<(HashedKey, Tier)> {
		std::mem::take(&mut self.migrations)
	}

	fn drain_demotions(&mut self) -> u64 {
		std::mem::take(&mut self.pending_demotions)
	}

	fn admission_latched(&self) -> bool {
		self.fast_tier_latched
	}

	fn fast_bytes_used(&self) -> CacheSize {
		self.fast_used
	}

	fn slow_bytes_used(&self) -> CacheSize {
		self.slow_used
	}

	fn fast_object_count(&self) -> usize {
		self.fast_chain.len()
	}

	fn slow_object_count(&self) -> usize {
		self.slow_chain.len()
	}
}

#[cfg(test)]
mod tests {
	use super::*;

	fn drain(stack: &mut LfuHybridStack) -> Vec<(HashedKey, Tier)> {
		stack.drain_tier_migrations()
	}

	#[test]
	fn admission_always_lands_fast() {
		let mut stack = LfuHybridStack::new(1_000);

		stack.insert(1, 10);
		stack.insert(2, 10);

		assert_eq!(stack.tier_of(1), Some(Tier::Fast));
		assert_eq!(stack.tier_of(2), Some(Tier::Fast));
		assert_eq!(stack.fast_bytes_used(), 20);
		assert_eq!(stack.slow_bytes_used(), 0);
	}

	#[test]
	fn admission_once_fast_is_full_goes_directly_to_slow() {
		// Fast capacity fits exactly one 10-byte key.
		let mut stack = LfuHybridStack::new(10);

		stack.insert(1, 10);
		drain(&mut stack);
		assert_eq!(stack.tier_of(1), Some(Tier::Fast));

		// Fast tier is now full; the new key is admitted straight to slow
		// -- key 1 (the existing resident) is untouched, matching the
		// paper's admission rule literally ("every new object is admitted
		// into the slow tier", not "whichever key loses a tie-break").
		stack.insert(2, 10);
		let migrations = drain(&mut stack);

		assert_eq!(migrations, vec![(2, Tier::Slow)]);
		assert_eq!(stack.tier_of(1), Some(Tier::Fast));
		assert_eq!(stack.tier_of(2), Some(Tier::Slow));
	}

	#[test]
	fn promotion_pressure_demotes_the_lowest_frequency_fast_key() {
		// Admission alone can never overflow the fast tier under the fixed
		// capacity check (new keys route around it once full), so demotion
		// pressure here has to come from a promotion instead.
		let mut stack = LfuHybridStack::new(20); // fits exactly two 10-byte keys

		stack.insert(1, 10);
		stack.insert(2, 10);
		drain(&mut stack);

		// Bump key 1's frequency so key 2 is unambiguously the fast tier's
		// minimum.
		stack.update(1);
		drain(&mut stack);

		stack.insert(3, 10); // fast tier full -> admitted directly to slow
		drain(&mut stack);
		assert_eq!(stack.tier_of(3), Some(Tier::Slow));

		// Bump key 3 past the fast minimum (key 2, count 1) -> promotes,
		// which needs to demote key 2 to make room.
		stack.update(3);
		let migrations = drain(&mut stack);

		assert!(migrations.iter().any(|(k, t)| *k == 3 && *t == Tier::Fast));
		assert!(migrations.iter().any(|(k, t)| *k == 2 && *t == Tier::Slow));
		assert_eq!(stack.tier_of(1), Some(Tier::Fast));
		assert_eq!(stack.tier_of(2), Some(Tier::Slow));
		assert_eq!(stack.tier_of(3), Some(Tier::Fast));
	}

	#[test]
	fn slow_key_promotes_once_frequency_strictly_exceeds_fast_minimum() {
		// Plenty of headroom (1_000) so promotion doesn't also cascade a
		// demotion — that combined behavior is covered separately by
		// `promotion_can_cascade_a_demotion`. This test isolates the
		// threshold check itself.
		let mut stack = LfuHybridStack::new(1_000);

		stack.insert(1, 10); // fast, count 1
		drain(&mut stack);

		// Manually place key 2 into the slow chain at count 1, bypassing
		// admission (which would land it fast, not slow) so the promotion
		// path can be exercised directly and deterministically.
		stack.slow_chain.insert_new(2);
		stack.entries.insert(2, LfuEntry { tier: Tier::Slow, size: 10 });
		stack.slow_used += 10;

		assert_eq!(stack.fast_chain.min_count(), Some(1));

		// One access brings key 2 to count 2, strictly exceeding the fast
		// minimum (1) -> promotes.
		stack.update(2);
		let migrations = drain(&mut stack);

		assert_eq!(migrations, vec![(2, Tier::Fast)]);
		assert_eq!(stack.tier_of(2), Some(Tier::Fast));
		assert_ne!(stack.tier_of(2), Some(Tier::Slow));
	}

	#[test]
	fn tie_with_fast_minimum_does_not_promote() {
		let mut stack = LfuHybridStack::new(1_000);

		stack.insert(1, 10); // fast, count 1
		stack.update(1); // fast, count 2 -> fast_min == 2

		// Manually place key 2 into the slow chain at count 1, so a single
		// real `update` bump (1 -> 2) lands it exactly on the fast
		// minimum (2), not strictly past it.
		stack.slow_chain.insert_new(2);
		stack.entries.insert(2, LfuEntry { tier: Tier::Slow, size: 10 });
		stack.slow_used += 10;

		assert_eq!(stack.fast_chain.min_count(), Some(2));

		stack.update(2); // bumps slow key 2 to count 2 -> ties fast_min
		let migrations = drain(&mut stack);

		assert_eq!(migrations, Vec::new(), "a tie must not promote");
		assert_eq!(stack.tier_of(2), Some(Tier::Slow));
	}

	#[test]
	fn promotion_can_cascade_a_demotion() {
		let mut stack = LfuHybridStack::new(20); // fits exactly two 10-byte keys

		stack.insert(1, 10);
		stack.insert(2, 10);
		stack.insert(3, 10); // fast tier full -> admitted directly to slow
		drain(&mut stack);

		assert_eq!(stack.tier_of(3), Some(Tier::Slow));
		assert_eq!(stack.tier_of(1), Some(Tier::Fast));
		assert_eq!(stack.tier_of(2), Some(Tier::Fast));

		// Bump key 3 past the fast minimum (count 1, tied between 1 and 2)
		// so it promotes; fast is already full, so the promotion must
		// demote whichever fast key is now the minimum (tie -> LRU).
		stack.update(3);
		let migrations = drain(&mut stack);

		assert!(migrations.iter().any(|(k, t)| *k == 3 && *t == Tier::Fast));
		assert!(migrations.iter().any(|(_, t)| *t == Tier::Slow));

		assert_eq!(stack.tier_of(3), Some(Tier::Fast));
		// Exactly one of {1, 2} should now be slow.
		let now_slow = [1, 2].into_iter()
			.filter(|k| stack.tier_of(*k) == Some(Tier::Slow))
			.count();
		assert_eq!(now_slow, 1);
	}

	#[test]
	fn evict_one_prefers_slow_falls_back_to_fast_when_slow_is_empty() {
		let mut stack = LfuHybridStack::new(1_000);

		stack.insert(1, 10);
		stack.insert(2, 10);
		drain(&mut stack);

		// Nothing has ever been demoted; slow chain is empty.
		assert_eq!(stack.evict_one(), Some(1));
		assert_eq!(stack.tier_of(2), Some(Tier::Fast));
	}

	#[test]
	fn evict_one_removes_from_slow_when_present() {
		let mut stack = LfuHybridStack::new(20);

		stack.insert(1, 10);
		stack.insert(2, 10);
		stack.insert(3, 10); // fast tier full -> key 3 admitted directly to slow
		drain(&mut stack);

		assert_eq!(stack.tier_of(3), Some(Tier::Slow));

		assert_eq!(stack.evict_one(), Some(3));
		assert_eq!(stack.slow_bytes_used(), 0);
	}

	#[test]
	fn resize_fast_tier_shrink_triggers_demotions() {
		let mut stack = LfuHybridStack::new(1_000);

		stack.insert(1, 10);
		stack.insert(2, 10);
		drain(&mut stack);

		stack.resize_fast_tier(10);
		let migrations = drain(&mut stack);

		assert_eq!(migrations.len(), 1);
		assert_eq!(stack.fast_bytes_used(), 10);
	}

	#[test]
	fn resize_fast_tier_grow_creates_headroom_for_next_promotion() {
		let mut stack = LfuHybridStack::new(20);

		stack.insert(1, 10);
		stack.insert(2, 10);
		stack.insert(3, 10); // fast tier full -> key 3 admitted directly to slow
		drain(&mut stack);

		assert_eq!(stack.tier_of(3), Some(Tier::Slow));

		stack.resize_fast_tier(1_000); // plenty of headroom now
		drain(&mut stack);

		stack.update(3); // bump to count 2, exceeds fast_min (1) -> promotes
		let migrations = drain(&mut stack);

		// With headroom, promotion should not need to demote anyone else.
		assert_eq!(migrations, vec![(3, Tier::Fast)]);
		assert_eq!(stack.tier_of(3), Some(Tier::Fast));
	}

	#[test]
	fn insert_at_preserves_an_arbitrary_accumulated_count() {
		let mut chain = FrequencyChain::default();

		chain.insert_new(1); // count 1
		chain.bump(1); // count 2
		chain.bump(1); // count 3

		let mut other = FrequencyChain::default();
		other.insert_new(2); // count 1

		// Move key 1 (count 3) into `other`, which has no bucket near 3.
		other.insert_at(1, 3);

		assert_eq!(other.min_count(), Some(1)); // key 2 still the minimum
		assert_eq!(other.len(), 2);

		// Popping the minimum should yield key 2 first (count 1), not key 1.
		assert_eq!(other.pop_min(), Some((2, 1)));
		assert_eq!(other.pop_min(), Some((1, 3)));
	}

	#[test]
	fn shared_overhead_reserves_dram_at_admission() {
		// Without overhead, two 40-byte values both fit in a 100-byte fast
		// tier (admission check is against the raw budget).
		let mut plain = LfuHybridStack::new(100);
		plain.insert(1, 40);
		plain.insert(2, 40);
		drain(&mut plain);
		assert_eq!(plain.tier_of(1), Some(Tier::Fast));
		assert_eq!(plain.tier_of(2), Some(Tier::Fast));

		// With a 30-byte per-object shared reservation, admitting the second
		// key reserves (1 existing + 1 new) × 30 = 60, leaving an effective
		// value budget of 40; 40 (used) + 40 (new) > 40, so it is admitted
		// straight to the slow tier.
		let mut stack = LfuHybridStack::new(100).with_shared_overhead(30);
		stack.insert(1, 40);
		stack.insert(2, 40);
		let migrations = drain(&mut stack);

		assert_eq!(migrations, vec![(2, Tier::Slow)]);
		assert_eq!(stack.tier_of(1), Some(Tier::Fast));
		assert_eq!(stack.tier_of(2), Some(Tier::Slow));
	}

	#[test]
	fn shared_overhead_exceeding_capacity_routes_admission_to_slow_never_evicts() {
		// One object's shared reservation (100) already exceeds the whole
		// fast budget (50): the effective admission budget saturates to 0, so
		// the object is admitted straight to slow.
		let mut stack = LfuHybridStack::new(50).with_shared_overhead(100);
		stack.insert(1, 10);
		let migrations = drain(&mut stack);

		assert_eq!(migrations, vec![(1, Tier::Slow)]);
		assert_eq!(stack.tier_of(1), Some(Tier::Slow));
		assert_eq!(stack.fast_bytes_used(), 0);

		// Demotion/slow-admission is the only response — still tracked, no
		// eviction (the DRAM budget never evicts).
		assert_eq!(stack.len(), 1);
		assert!(!stack.needs_capacity_eviction());
	}

	#[test]
	fn remove_and_clear_reset_bookkeeping() {
		let mut stack = LfuHybridStack::new(1_000);

		stack.insert(1, 10);
		stack.insert(2, 10);
		drain(&mut stack);

		stack.remove(1);
		assert_eq!(stack.contains(1), false);
		assert_eq!(stack.fast_bytes_used(), 10);

		stack.clear();
		assert_eq!(stack.len(), 0);
		assert_eq!(stack.fast_bytes_used(), 0);
		assert_eq!(stack.slow_bytes_used(), 0);
		assert_eq!(stack.tier_of(2), None);
		assert_eq!(stack.evict_one(), None);
	}

	#[test]
	fn admission_latches_shut_once_capacity_is_reached_even_after_headroom_reopens() {
		// Reproduces the object-granular-demotion headroom gap: two 50-byte
		// keys fill a 100-byte fast tier; promoting a third (5-byte) slow key
		// demotes one of them (LRU-within-tie), leaving 45 bytes of slack even
		// though the tier has "reached capacity" in the LFU sense. A raw byte
		// check alone would let a brand-new frequency-1 key back into that
		// slack; the latch must block it once every current fast resident is
		// frequency >= 2.
		let mut stack = LfuHybridStack::new(100);

		stack.insert(1, 50); // A, freq 1
		stack.insert(2, 50); // B, freq 1
		drain(&mut stack);
		assert_eq!(stack.fast_bytes_used(), 100);

		stack.insert(3, 5); // C, freq 1 -> capacity full -> slow, latches
		drain(&mut stack);
		assert_eq!(stack.tier_of(3), Some(Tier::Slow));

		stack.update(3); // C: freq 1 -> 2, promotes, demotes the freq-1 LRU tie
		drain(&mut stack);

		// Demotion left byte slack (some fast resident's size exceeded the
		// promotion's overage).
		assert!(stack.fast_bytes_used() < 100, "demotion should have overshot the exact overage");

		// A brand-new key must go straight to slow: the latch is already
		// tripped, regardless of the leftover slack.
		stack.insert(4, 5);
		let migrations = drain(&mut stack);

		assert_eq!(migrations, vec![(4, Tier::Slow)]);
		assert_eq!(stack.tier_of(4), Some(Tier::Slow));
	}

	#[test]
	fn growing_fast_tier_via_resize_unlatches_admission() {
		let mut stack = LfuHybridStack::new(10);

		stack.insert(1, 10); // fills the tiny fast tier
		drain(&mut stack);

		stack.insert(2, 10); // doesn't fit -> slow, latches
		drain(&mut stack);
		assert_eq!(stack.tier_of(2), Some(Tier::Slow));

		// A deliberate capacity increase should immediately reopen direct
		// admission, not force new keys to wait on a promotion.
		stack.resize_fast_tier(1_000);
		drain(&mut stack);

		stack.insert(3, 10);
		let migrations = drain(&mut stack);

		assert_eq!(migrations, Vec::new(), "key 3 should have been admitted directly to fast");
		assert_eq!(stack.tier_of(3), Some(Tier::Fast));
	}

	#[test]
	fn shrinking_fast_tier_via_resize_does_not_unlatch() {
		let mut stack = LfuHybridStack::new(1_000);

		stack.insert(1, 10);
		drain(&mut stack);

		// Never reached capacity yet -- latch still open.
		stack.resize_fast_tier(500);
		drain(&mut stack);

		stack.insert(2, 10);
		let migrations = drain(&mut stack);
		assert_eq!(stack.tier_of(2), Some(Tier::Fast), "shrinking alone must not spuriously latch");
		assert_eq!(migrations, Vec::new());
	}

	#[test]
	fn clear_resets_the_latch() {
		let mut stack = LfuHybridStack::new(10);

		stack.insert(1, 10);
		stack.insert(2, 10); // doesn't fit -> slow, latches
		drain(&mut stack);
		assert_eq!(stack.tier_of(2), Some(Tier::Slow));

		stack.clear();

		stack.insert(3, 10);
		let migrations = drain(&mut stack);
		assert_eq!(stack.tier_of(3), Some(Tier::Fast), "clear() should reset the latch");
		assert_eq!(migrations, Vec::new());
	}
}
