/*
 * Copyright (c) Kia Shakiba
 *
 * This source code is licensed under the GNU AGPLv3 license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! `LruLfuHybridStack` — recency in the fast tier, frequency in the slow
//! tier, for `PaperPolicy::LruLfuHybrid`.
//!
//! Every other hybrid stack in this crate orders both tiers by the *same*
//! metric: `LruHybridStack` is literally one recency list with a boundary
//! cursor marking the fast/slow cut, and `LfuHybridStack` runs two chains
//! that both rank by frequency. This stack is the first where the two tiers
//! rank by different metrics, which changes what is and isn't expressible.
//!
//! ## Why split the metrics
//!
//! The two tiers have different jobs. The fast tier (DRAM) is small and
//! holds the active working set, where recency is the right signal for
//! short-term reuse and costs one list splice per access. The slow tier
//! (PMEM) is far larger and holds a long cold tail, where LRU is close to
//! meaningless — its tail position is dominated by scans and one-hit-wonders
//! and says little about whether an object deserves to survive. LFU is
//! scan-resistant and actually identifies the popular cold objects worth
//! promoting, and the unpopular ones worth evicting.
//!
//! In one line: **frequency is the admission control *into* DRAM; recency is
//! the retention policy *within* DRAM.**
//!
//! ## Rules
//!
//! - **Admission**: new object → fast tier, at the recency head, frequency 1.
//!   Admitting to the slow tier instead would make every `set()` a
//!   synchronous PMEM allocation; `two_q_hybrid` vs `two_q_fast_admission`
//!   measured that at 2.15x SET mean / 2.72x SET p99 (see `CLAUDE.md`), and
//!   the traces this is aimed at are 80–94% SETs.
//! - **Demotion**: fast tier's LRU tail → slow chain, **carrying its
//!   accumulated frequency** (`FrequencyChain::insert_at`). This is why the
//!   fast tier counts frequency at all despite not ordering by it — see
//!   "Why the fast tier counts frequency" below.
//! - **Promotion**: a slow object whose frequency *reaches* `promote_k` moves
//!   to the fast tier's recency head, and its counter **resets**.
//! - **Eviction**: the slow chain's minimum-frequency key (ties broken
//!   least-recently-touched, matching `LfuStack`'s convention), falling back
//!   to the fast tier's LRU tail when nothing has ever been demoted — the
//!   same last-resort fallback every hybrid stack here has.
//!
//! ## Why promotion is a fixed threshold, not a cross-tier comparison
//!
//! `LfuHybridStack` promotes when a slow object's frequency exceeds the fast
//! tier's *minimum* frequency. That rule is unavailable here: the fast tier
//! is recency-ordered and has no O(1)-queryable minimum frequency. Making it
//! queryable would mean maintaining a frequency chain over the fast tier
//! *in addition to* its recency list — roughly doubling the fast tier's
//! structural cost to answer a question a fixed threshold answers for free.
//!
//! `promote_k` is that threshold, and it is an **absolute frequency**, not a
//! count of accesses since demotion. That distinction is load-bearing and is
//! easy to get wrong (this file's own tests did, first time round):
//!
//! - A key admitted and never accessed demotes carrying frequency 1, so it
//!   needs `promote_k - 1` further accesses to come back. **`promote_k == 2`
//!   therefore behaves exactly like `promote_k == 1` for such a key** — one
//!   slow access promotes it, which is `LruHybridStack`'s rule. The first
//!   threshold that actually filters anything is **3**.
//! - A key that was genuinely hot before it demoted arrives in the slow tier
//!   at or near the cap, and promotes on its very next access regardless of
//!   `promote_k`. This is deliberate, and it is the same property that makes
//!   carried frequency worth having: an object with demonstrated popularity
//!   needs only one confirmation that it is still in use, while a
//!   one-hit-wonder has to earn the whole climb.
//!
//! Sharpening this to "accesses since demotion" would need a second per-key
//! counter (or the demotion-time frequency stored alongside), which would
//! push `LruLfuEntry` past 8 bytes — see below for why that matters. If
//! threshold tuning proves unsatisfying, the fast-tier frequency index is the
//! documented next step.
//!
//! ## Why the fast tier counts frequency
//!
//! If demoted objects entered the slow chain at frequency 1, everything the
//! fast tier learned would be discarded: an object hit 50 times in DRAM would
//! land indistinguishable from a one-hit-wonder demoted in the same pass —
//! and those are precisely the objects most likely to be referenced again.
//! So the fast tier maintains a counter it does not rank by; it is carried
//! metadata, handed to `insert_at` on demotion.
//!
//! ## Why the counter is capped, and why the cap is small
//!
//! [`FREQUENCY_CAP`] pays twice.
//!
//! **Memory.** `LruEntry { tier, size }` in `LruHybridStack` measures 8 bytes
//! (`u8` + `u32`), pairing with the 8-byte `HashedKey` to exactly 16 — the
//! figure `object/overhead.rs`'s DRAM-reservation constants are derived from.
//! Adding a `u32` counter would push the entry to `4+4+1 = 9` → padded to 12
//! → a 24-byte pair, +8 bytes on *every* object in *both* tiers. A `u16`
//! counter gives `4+2+1 = 7` → padded back to 8, so the pair stays 16 and
//! this design costs no more per-object DRAM than `LruHybridStack`.
//!
//! **Demotion cost.** `FrequencyChain::insert_at` is a *linear scan* over
//! count buckets, and every demotion performs one into the large slow chain.
//! Its cost is O(distinct frequency values present), which a cap bounds
//! directly: with counts confined to `1..=FREQUENCY_CAP` the chain can never
//! hold more than `FREQUENCY_CAP` buckets. Capping is what keeps demotion
//! cheap, not merely what keeps the entry small.
//!
//! Pair the cap with reset-on-promotion (an object spends its credit earning
//! DRAM). Without it a repeatedly-promoted object accumulates an unassailable
//! count and becomes effectively un-evictable — ordinary LFU ossification,
//! and worse on multi-day traces whose popularity distribution shifts.
//!
//! ## A `set()` is an access, not an automatic promotion
//!
//! This is a deliberate divergence from `LruHybridStack`, where a `set()` on
//! an existing key *always* re-admits it to the fast tier. Here an overwrite
//! goes through the same frequency gate a read does: it bumps the counter and
//! promotes only if that crosses `promote_k`.
//!
//! The reason is that the gate would otherwise be porous on exactly the
//! workloads this is aimed at. On a trace that is 80–94% SETs, "any set
//! promotes" means nearly everything reaches DRAM without ever demonstrating
//! reuse, and the slow tier's frequency ordering never gets to filter
//! anything. Keeping writes inside the gate is what makes "frequency is the
//! admission control into DRAM" true rather than aspirational.
//!
//! Consequence for the API layer: `hybrid_policy::admission_tier` must
//! look up an *existing* key's current tier (like `fifo_hybrid_cache`'s does)
//! rather than answering purely from "is this key new", so an overwrite of a
//! slow-tier key is written straight to PMEM instead of being written to DRAM
//! and corrected afterward.
//!
//! ## Structure
//!
//! Two homogeneous per-tier structures and one combined per-key map:
//!
//! ```text
//! fast_stack:  RecencyList      // fast tier only, recency-ordered
//! slow_chain:  FrequencyChain   // slow tier only, frequency-ordered
//! entries:     EntryMap         // { tier, size, freq } — 8 bytes
//! ```
//!
//! Note what is *absent*: `LruHybridStack`'s `fast_boundary` cursor. Because
//! the slow tier is no longer part of the same list, there is no cut to
//! track — the fast list's own tail *is* the demotion candidate, in O(1),
//! and none of the boundary-repair logic that `touch_fast_key`/`remove`/
//! `evict_one` need there exists here. `LruSizedHybridStack` reached the same
//! conclusion from the other direction: homogeneous per-tier structures beat
//! cursor tricks.
//!
//! Like every stack here, this one tracks *order and tier membership only*.
//! It moves no bytes; `PolicyWorker` drains `drain_tier_migrations` and
//! performs the real `TieredBuffer` reallocation (see `Object::set_data`).

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
// alias (`SlowObjects`, node-1 jemalloc arenas) that `BufferPMEM`/other PMEM
// features already use.
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

/// Superseded by [`watermarks`]: `settle_fast_tier` now triggers at
/// `watermarks::high_bytes` of the effective budget and drains to
/// `watermarks::low_bytes` of it, shared with every other hybrid stack and
/// tunable at runtime.
///
/// Kept only as the historical record of what this file used to do -- a 2%
/// shave off the ceiling, on the rationale that admission here lands in the
/// fast tier and is written to DRAM synchronously by `PaperCache::set()`
/// before this stack ever sees the event, so a burst of concurrent `set()`
/// calls can transiently exceed what the stack's own bookkeeping shows.
/// `watermarks`' default low ratio subsumes that burst margin; setting
/// `FAST_TIER_HIGH_WATERMARK=1.0` and `FAST_TIER_LOW_WATERMARK=0.98`
/// reproduces this constant's exact behaviour.
#[allow(dead_code)]
const FAST_TIER_LOW_WATER_RATIO: f64 = 0.98;

/// Maximum value the per-object frequency counter saturates at. See the
/// module doc's "Why the counter is capped" section: this bounds both the
/// entry size (so the `(HashedKey, LruLfuEntry)` pair stays 16 bytes) and
/// `FrequencyChain::insert_at`'s linear scan (so the slow chain can never
/// hold more than this many buckets).
///
/// 16 is generous for the discrimination LFU actually needs — S3-FIFO caps
/// its equivalent counter at 3 — while leaving room for a `promote_k` well
/// above the useful range.
const FREQUENCY_CAP: u16 = 16;

#[cfg(not(feature = "eviction_stacks_pmem"))]
type RecencyList = HashList<HashedKey, NoHasher>;
#[cfg(feature = "eviction_stacks_pmem")]
type RecencyList = PmemHashList<HashedKey, NoHasher>;

#[cfg(not(feature = "eviction_stacks_pmem"))]
type ChainIndex = Index<CountStack>;
#[cfg(feature = "eviction_stacks_pmem")]
type ChainIndex = PmemIndex;

/// Combined per-key bookkeeping. Field order is chosen so the struct packs to
/// 8 bytes (`u32` + `u16` + `u8` = 7, padded to 8), keeping the
/// `(HashedKey, LruLfuEntry)` pair at 16 — see the module doc.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct LruLfuEntry {/// Part of `size` that stays in DRAM in either tier; see `migrating`.
	dram_resident: u8,

	size: ObjectSize,
	/// Accesses accumulated, saturating at [`FREQUENCY_CAP`]. Meaningful in
	/// both tiers: it ranks the key while slow, and is carried across on
	/// demotion while fast. Reset to 1 on promotion.
	freq: u16,
	tier: Tier,
}

impl LruLfuEntry {
/// The bytes that actually move between tiers when this object migrates.
	///
	/// `size` is `base_size`, which also counts the DRAM-resident remainder --
	/// the key and expiry field (inline in the object map) plus the `Expiries`
	/// entry when a TTL is set. `Object::set_data` replaces the value buffer
	/// alone, so none of that moves, and the key and expiry are already inside
	/// `shared_overhead`. Charging them to the tier counters double-counted
	/// every fast-tier object and made demotion appear to free DRAM it did not.
	#[inline]
	fn migrating(&self) -> CacheSize {
		(self.size as CacheSize).saturating_sub(self.dram_resident as CacheSize)
	}
}

/// `dram_resident` was meant to occupy padding the entry already had.
/// If this ever fails, the field is costing 4 more bytes on *every* tracked
/// object in both tiers, which defeats the point of storing it per entry.
const _: () = assert!(
	std::mem::size_of::<LruLfuEntry>() == 8,
	"LruLfuEntry grew past 8 bytes",
);


#[cfg(not(feature = "eviction_stacks_pmem"))]
type EntryMap = HashMap<HashedKey, LruLfuEntry, NoHasher>;
#[cfg(feature = "eviction_stacks_pmem")]
type EntryMap = HashMap<HashedKey, LruLfuEntry, NoHasher, Hybrid>;

/// A classic O(1) LFU frequency-bucket chain: an ascending-by-count linked
/// list of `CountStack` buckets (each holding every key at that exact
/// frequency, itself recency-ordered so ties break LRU-within-frequency),
/// plus an index from key to its current bucket.
///
/// Adapted from `LfuHybridStack`'s chain of the same name. Only one tier
/// needs one here, so the operations that stack uses to compare two chains
/// (`min_count`) are absent; `insert_at` — placing a key at an arbitrary
/// accumulated frequency when it crosses tiers — is the one this design
/// depends on most, since every demotion performs one.
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
		self.stack.remove(&key);
	}
}

impl FrequencyChain {
	fn len(&self) -> usize {
		self.index_map.len()
	}

	fn contains(&self, key: HashedKey) -> bool {
		self.index_map.contains_key(&key)
	}

	/// Places `key` into the bucket for an arbitrary `count`, creating that
	/// bucket in sorted position if absent. Linear in the number of distinct
	/// counts present, which [`FREQUENCY_CAP`] bounds — see the module doc.
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

	/// Moves an already-tracked key from its current bucket to `count`'s,
	/// keeping the chain ordered after the caller has bumped the key's
	/// counter. A no-op if the key isn't in this chain.
	fn move_to(&mut self, key: HashedKey, count: u32) {
		if !self.index_map.contains_key(&key) {
			return;
		}

		self.remove(key);
		self.insert_at(key, count);
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

	/// Removes and returns the lowest-frequency key, ties broken
	/// least-recently-touched.
	fn pop_min(&mut self) -> Option<HashedKey> {
		let count_stack_index = self.count_stacks.front_index()?;
		let count_stack = self.count_stacks.get_mut(count_stack_index)?;

		let key = count_stack.pop();
		self.index_map.remove(&key);

		if count_stack.is_empty() {
			self.count_stacks.remove(count_stack_index);
		}

		Some(key)
	}

	fn clear(&mut self) {
		self.index_map.clear();
		self.count_stacks.clear();
	}
}

pub struct LruLfuHybridStack {
	/// Fast tier, recency-ordered. Its back is always the demotion candidate.
	fast_stack: RecencyList,
	/// Slow tier, frequency-ordered. Its minimum is always the eviction
	/// candidate.
	slow_chain: FrequencyChain,

	entries: EntryMap,

	/// Absolute frequency a slow-tier key must reach to earn the fast tier —
	/// *not* a count of accesses since it was demoted. Values below 3 do not
	/// filter a never-accessed key at all; see the module doc's "Why
	/// promotion is a fixed threshold" section.
	promote_k: u16,

	fast_capacity: CacheSize,
	fast_used: CacheSize,
	slow_used: CacheSize,

	/// Approximate per-object DRAM cost of the shared structures (object
	/// hashtable + eviction stacks) that hold an entry for every object of
	/// both tiers. Reserved out of `fast_capacity` in `settle_fast_tier` so
	/// the fast-tier budget bounds total DRAM rather than just fast-tier
	/// values. `0` unless set via `with_shared_overhead`.
	shared_overhead: CacheSize,

	/// (key, new tier) pairs recorded since the last `drain_tier_migrations`.
	migrations: Vec<(HashedKey, Tier)>,
}

impl LruLfuHybridStack {
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

	/// `promote_k` is the absolute frequency a slow object must reach to be
	/// promoted (see the module doc — it is not a count of accesses since
	/// demotion, and values below 3 do not filter a never-accessed key). It
	/// is clamped to at least 1 (0 would make every slow object instantly
	/// promotable before it was ever accessed) and at most [`FREQUENCY_CAP`]
	/// (a threshold above the cap could never be reached, silently disabling
	/// promotion entirely).
	pub fn new(fast_capacity: CacheSize, promote_k: u16) -> Self {
		let (fast_stack, entries) = Self::new_collections();

		LruLfuHybridStack {
			fast_stack,
			slow_chain: FrequencyChain::default(),
			entries,

			promote_k: promote_k.clamp(1, FREQUENCY_CAP),

			fast_capacity,
			fast_used: 0,
			slow_used: 0,
			shared_overhead: 0,

			migrations: Vec::new(),
		}
	}

	/// Sets the approximate per-object shared-structure DRAM overhead
	/// reserved out of the fast-tier budget. See
	/// `crate::object::overhead::get_hybrid_dram_shared_overhead`.
	pub fn with_shared_overhead(mut self, overhead: CacheSize) -> Self {
		self.shared_overhead = overhead;
		self
	}

	/// The configured fast-tier byte budget.
	pub fn fast_capacity(&self) -> CacheSize {
		self.fast_capacity
	}

	/// The configured promotion threshold, after clamping.
	pub fn promote_k(&self) -> u16 {
		self.promote_k
	}

	fn reserved_overhead(&self) -> CacheSize {
		self.entries.len() as CacheSize * self.shared_overhead
	}

	/// Returns the tier the given (currently tracked) key is in, or `None`
	/// if the key isn't tracked. Exposed for tests/diagnostics.
	pub fn tier_of(&self, key: HashedKey) -> Option<Tier> {
		self.entries.get(&key).map(|entry| entry.tier)
	}

	/// Returns the key's current frequency counter. Exposed for tests.
	pub fn frequency_of(&self, key: HashedKey) -> Option<u16> {
		self.entries.get(&key).map(|entry| entry.freq)
	}

	/// Records a size change for an already-tracked key without altering its
	/// tier, adjusting whichever tier's used-bytes counter applies.
	/// `new_resident` refreshes the entry's DRAM-resident remainder: a re-set
	/// can add or drop a TTL, which changes it by the `Expiries` entry's cost.
	/// Without this the entry keeps its old remainder and every later
	/// migration moves the wrong number of bytes.
	fn resize_key(&mut self, key: HashedKey, new_size: ObjectSize, new_resident: u8) {
		let Some(entry) = self.entries.get_mut(&key) else { return };

		let old_migrating = entry.migrating();
		entry.size = new_size;
		entry.dram_resident = new_resident;
		let delta = entry.migrating() as i64 - old_migrating as i64;

		match entry.tier {
			Tier::Fast => {
				self.fast_used = (self.fast_used as i64 + delta).max(0) as CacheSize;
			},

			Tier::Slow => {
				self.slow_used = (self.slow_used as i64 + delta).max(0) as CacheSize;
			},
		}
	}

	/// Saturating counter bump. Returns the new value.
	fn bump_frequency(&mut self, key: HashedKey) -> u16 {
		let Some(entry) = self.entries.get_mut(&key) else { return 0 };

		entry.freq = entry.freq.saturating_add(1).min(FREQUENCY_CAP);
		entry.freq
	}

	/// Handles an access (read or overwrite) to a key already in the fast
	/// tier: pure recency, plus the carried counter bump. Never migrates.
	fn touch_fast(&mut self, key: HashedKey) {
		self.fast_stack.move_front(&key);
		self.bump_frequency(key);
	}

	/// Handles an access to a key in the slow tier: bump its counter, keep
	/// the frequency chain ordered, and promote if that crossed
	/// `promote_k`.
	fn touch_slow(&mut self, key: HashedKey) {
		let freq = self.bump_frequency(key);

		if freq < self.promote_k {
			// Still earning its way in — reorder within the slow chain only.
			self.slow_chain.move_to(key, freq as u32);
			return;
		}

		self.promote(key);
	}

	/// Moves a slow-tier key to the fast tier's recency head, resetting its
	/// counter (it spent that credit earning DRAM — see the module doc).
	fn promote(&mut self, key: HashedKey) {
		let Some(entry) = self.entries.get_mut(&key) else { return };

		let size = entry.migrating();
		entry.tier = Tier::Fast;
		entry.freq = 1;

		self.slow_chain.remove(key);
		self.fast_stack.push_front(key);

		self.slow_used = self.slow_used.saturating_sub(size);
		self.fast_used += size;

		self.settle_fast_tier();

		// Pushed *after* `settle_fast_tier`, matching `LruHybridStack::
		// touch_fast_key`: `apply_tier_migrations` applies a batch's
		// demotions before its promotions, but within the stack the ordering
		// still matters for the guard below — an extremely tight budget can
		// demote this very key straight back out in the same settle, in
		// which case `settle_fast_tier` has already pushed the correct final
		// `(key, Tier::Slow)` entry and no `Fast` entry should follow it.
		if self.entries.get(&key).map(|entry| entry.tier) == Some(Tier::Fast) {
			self.migrations.push((key, Tier::Fast));
		}
	}

	/// Demotes the least-recently-used fast key(s) into the slow chain,
	/// *triggered* once `fast_used` exceeds [`watermarks::high_bytes`] of the
	/// effective value budget (`fast_capacity` minus the DRAM reserved for
	/// shared per-object metadata across both tiers) but *drained* all the
	/// way down to [`watermarks::low_bytes`] of it.
	///
	/// The effective budget itself is unchanged: the watermarks apply *on top
	/// of* the shared-overhead reservation, they do not replace it. Triggering
	/// below the ceiling instead of at it is what makes a pass demote a
	/// *batch* rather than the single object each admission displaced when the
	/// tier sat pinned at 100% utilisation -- see the `watermarks` module doc.
	///
	/// Each demoted key enters the slow chain at its accumulated frequency,
	/// not at 1 — that carry is the whole reason the fast tier counts.
	/// Demotion is the only response; the DRAM budget never evicts (terminal
	/// eviction stays governed solely by `max_size`).
	fn settle_fast_tier(&mut self) {
		let effective = self.fast_capacity.saturating_sub(self.reserved_overhead());

		if self.fast_used <= watermarks::high_bytes(effective) {
			return;
		}

		let drain_target = watermarks::low_bytes(effective);

		while self.fast_used > drain_target {
			let Some(demote_key) = self.fast_stack.pop_back() else { break };

			let (size, freq) = match self.entries.get_mut(&demote_key) {
				Some(entry) => {
					entry.tier = Tier::Slow;
					(entry.migrating(), entry.freq)
				},

				// Untracked key in the recency list should be impossible;
				// drop it rather than looping forever on it.
				None => continue,
			};

			self.fast_used = self.fast_used.saturating_sub(size);
			self.slow_used += size;
			self.slow_chain.insert_at(demote_key, freq as u32);

			self.migrations.push((demote_key, Tier::Slow));
		}
	}
}

impl PolicyStack for LruLfuHybridStack {
	fn is_policy(&self, policy: &PaperPolicy) -> bool {
		matches!(policy, PaperPolicy::LruLfuHybrid(_))
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
		if self.entries.contains_key(&key) {
			// An overwrite is an access, not an automatic promotion — see
			// the module doc's "A `set()` is an access" section.
			self.resize_key(key, size, dram_resident);
			self.update(key);
			return;
		}

		self.fast_stack.push_front(key);
		self.entries.insert(key, LruLfuEntry { dram_resident, size, freq: 1, tier: Tier::Fast });
		self.fast_used += (size as CacheSize).saturating_sub(dram_resident as CacheSize);

		self.settle_fast_tier();
	}

	fn update(&mut self, key: HashedKey) {
		match self.entries.get(&key).map(|entry| entry.tier) {
			Some(Tier::Fast) => self.touch_fast(key),
			Some(Tier::Slow) => self.touch_slow(key),
			None => {},
		}
	}

	fn remove(&mut self, key: HashedKey) {
		let Some(entry) = self.entries.remove(&key) else { return };

		let size = entry.migrating();

		match entry.tier {
			Tier::Fast => {
				self.fast_stack.remove(&key);
				self.fast_used = self.fast_used.saturating_sub(size);
			},

			Tier::Slow => {
				self.slow_chain.remove(key);
				self.slow_used = self.slow_used.saturating_sub(size);
			},
		}
	}

	fn clear(&mut self) {
		self.fast_stack.clear();
		self.slow_chain.clear();
		self.entries.clear();

		self.fast_used = 0;
		self.slow_used = 0;
		self.migrations.clear();
	}

	fn evict_one(&mut self) -> Option<HashedKey> {
		// Slow tier first, by minimum frequency. The fast-tier fallback only
		// applies when nothing has ever been demoted — the same last-resort
		// path every hybrid stack here keeps.
		let key = match self.slow_chain.pop_min() {
			Some(key) => key,
			None => self.fast_stack.pop_back()?,
		};

		if let Some(entry) = self.entries.remove(&key) {
			let size = entry.migrating();

			match entry.tier {
				Tier::Fast => {
					// Popped off the recency list already by the fallback
					// branch above.
					self.fast_used = self.fast_used.saturating_sub(size);
				},

				Tier::Slow => {
					self.slow_used = self.slow_used.saturating_sub(size);
				},
			}
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
		self.fast_stack.len()
	}

	fn slow_object_count(&self) -> usize {
		self.slow_chain.len()
	}
}

#[cfg(test)]
mod tests {
	use super::*;

	const K: u16 = 2;

	fn drain(stack: &mut LruLfuHybridStack) -> Vec<(HashedKey, Tier)> {
		stack.drain_tier_migrations()
	}

	/// A fast-tier capacity that leaves `target` bytes sitting comfortably
	/// *above* `settle_fast_tier`'s low-water drain floor.
	///
	/// Sizing a test's capacity to exactly what should survive is a trap:
	/// `settle_fast_tier` triggers at `watermarks::high_bytes` but drains all
	/// the way to `watermarks::low_bytes`, so at these tiny capacities the
	/// drain floor lands *below* what the intended survivors need and
	/// cascades an extra demotion the test never meant to exercise. (Both of
	/// this module's cascade/counter tests were originally written with a
	/// bare `100` and failed for exactly that reason.) `LruHybridStack`'s
	/// tests carry the same helper for the same reason.
	///
	/// Derived from `watermarks::low()` rather than a literal ratio so these
	/// tests track whatever high/low pair is configured.
	///
	/// Its callers are coarse-grained -- 50-byte objects against a ~100-byte
	/// budget -- so "two survive, the third triggers a pass" is only
	/// expressible while `high() / low() < 1.5`: below that no capacity
	/// satisfies both `low_bytes(cap) >= target` and
	/// `high_bytes(cap) < target + 50` at once. That covers the defaults
	/// (0.98/0.95) and the drain-to-ceiling restore (1.0/1.0). The watermark
	/// tests at the bottom of this module are granular enough to hold at any
	/// pair, and are the ones that pin the watermark semantics themselves.
	fn low_water_safe(target: CacheSize) -> CacheSize {
		(target as f64 / watermarks::low()).ceil() as CacheSize + 1
	}

	// ── admission ─────────────────────────────────────────────────────────

	#[test]
	fn admission_always_lands_fast_at_frequency_one() {
		let mut stack = LruLfuHybridStack::new(1_000, K);

		stack.insert(1, 10);

		assert_eq!(stack.tier_of(1), Some(Tier::Fast));
		assert_eq!(stack.frequency_of(1), Some(1));
		assert!(drain(&mut stack).is_empty(), "admission needs no migration");
	}

	#[test]
	fn entry_packs_to_eight_bytes() {
		// The whole reason the counter is u16 rather than u32 — see the
		// module doc. If this ever fails, `object/overhead.rs`'s
		// LRU_LFU_HYBRID_EVICTION_STACK_DRAM_OVERHEAD needs re-deriving.
		assert_eq!(std::mem::size_of::<LruLfuEntry>(), 8);
		assert_eq!(std::mem::size_of::<(HashedKey, LruLfuEntry)>(), 16);
	}

	// ── demotion ──────────────────────────────────────────────────────────

	#[test]
	fn fast_pressure_demotes_the_lru_tail() {
		// `low_water_safe` so the first two keys sit under the high
		// watermark (a bare `100` puts them over it, demoting key 1 before
		// the third admission this test is actually about) and the third
		// admission triggers exactly one demotion.
		let mut stack = LruLfuHybridStack::new(low_water_safe(100), K);

		stack.insert(1, 50);
		stack.insert(2, 50);
		drain(&mut stack);

		// 1 is the LRU tail; admitting a third key must demote it.
		stack.insert(3, 50);

		assert_eq!(stack.tier_of(1), Some(Tier::Slow));
		assert_eq!(stack.tier_of(3), Some(Tier::Fast));

		let migrations = drain(&mut stack);
		assert!(migrations.contains(&(1, Tier::Slow)), "got {migrations:?}");
	}

	#[test]
	fn demotion_carries_the_accumulated_frequency() {
		let mut stack = LruLfuHybridStack::new(100, 99);

		stack.insert(1, 50);
		// Bump 1 well above a fresh admission's frequency. promote_k is
		// clamped to FREQUENCY_CAP, above these bumps, so 1 stays fast.
		stack.update(1);
		stack.update(1);
		assert_eq!(stack.frequency_of(1), Some(3));

		stack.insert(2, 50);
		stack.insert(3, 50);
		drain(&mut stack);

		// 1 demoted, and must have carried its count rather than resetting.
		assert_eq!(stack.tier_of(1), Some(Tier::Slow));
		assert_eq!(stack.frequency_of(1), Some(3));
	}

	#[test]
	fn a_demoted_object_outranks_a_one_hit_wonder_in_the_slow_tier() {
		// The payoff of carrying frequency: eviction must prefer the object
		// that was never popular, not the one that merely cooled off.
		let mut stack = LruLfuHybridStack::new(100, 99);

		stack.insert(1, 50);
		stack.update(1);
		stack.update(1); // freq 3 — genuinely hot while fast

		stack.insert(2, 50); // freq 1 — one-hit wonder
		stack.insert(3, 50); // forces a demotion
		stack.insert(4, 50); // forces another
		drain(&mut stack);

		assert_eq!(stack.tier_of(1), Some(Tier::Slow));
		assert_eq!(stack.tier_of(2), Some(Tier::Slow));

		// Minimum frequency in the slow chain is key 2, not key 1.
		assert_eq!(stack.evict_one(), Some(2));
	}

	// ── promotion ─────────────────────────────────────────────────────────

	#[test]
	fn a_slow_access_below_the_absolute_threshold_does_not_promote() {
		// k = 2 is NOT a filter for a never-accessed key: it demotes at
		// frequency 1, so a single access already reaches the threshold.
		// This is the documented off-by-one that makes 3 the first
		// meaningful value.
		let mut lenient = LruLfuHybridStack::new(100, 2);

		lenient.insert(1, 50);
		lenient.insert(2, 50);
		lenient.insert(3, 50);
		drain(&mut lenient);
		assert_eq!(lenient.tier_of(1), Some(Tier::Slow));

		lenient.update(1);
		assert_eq!(
			lenient.tier_of(1),
			Some(Tier::Fast),
			"at k = 2 a single slow access suffices — same as lru_hybrid_cache",
		);

		let mut strict = LruLfuHybridStack::new(100, 3);
		strict.insert(1, 50);
		strict.insert(2, 50);
		strict.insert(3, 50);
		drain(&mut strict);
		assert_eq!(strict.tier_of(1), Some(Tier::Slow));

		strict.update(1); // freq 1 -> 2, still below k=3
		assert_eq!(strict.tier_of(1), Some(Tier::Slow));
		assert!(drain(&mut strict).is_empty(), "no migration below threshold");

		strict.update(1); // freq 2 -> 3, meets k
		assert_eq!(strict.tier_of(1), Some(Tier::Fast));
	}

	#[test]
	fn crossing_the_threshold_promotes_and_resets_the_counter() {
		let mut stack = LruLfuHybridStack::new(100, 2);

		stack.insert(1, 50);
		stack.insert(2, 50);
		stack.insert(3, 50);
		drain(&mut stack);
		assert_eq!(stack.tier_of(1), Some(Tier::Slow));

		stack.update(1);

		assert_eq!(stack.tier_of(1), Some(Tier::Fast));
		assert_eq!(stack.frequency_of(1), Some(1), "counter resets on promotion");

		let migrations = drain(&mut stack);
		assert!(migrations.contains(&(1, Tier::Fast)), "got {migrations:?}");
	}

	#[test]
	fn promotion_can_cascade_a_demotion() {
		// Sized so exactly one key demotes, leaving the fast tier full with
		// two — so the promotion below has to push someone back out.
		let mut stack = LruLfuHybridStack::new(low_water_safe(100), 2);

		stack.insert(1, 50);
		stack.insert(2, 50);
		stack.insert(3, 50);
		drain(&mut stack);

		// 1 is slow; 2 and 3 fill the fast tier.
		assert_eq!(stack.tier_of(1), Some(Tier::Slow));
		assert_eq!(stack.tier_of(2), Some(Tier::Fast));
		assert_eq!(stack.tier_of(3), Some(Tier::Fast));

		stack.update(1); // promotes 1, which must push someone out

		let migrations = drain(&mut stack);
		assert!(migrations.contains(&(1, Tier::Fast)), "got {migrations:?}");
		assert!(
			migrations.iter().any(|(_, tier)| *tier == Tier::Slow),
			"promotion should have cascaded a demotion; got {migrations:?}",
		);
	}

	#[test]
	fn an_overwrite_goes_through_the_same_frequency_gate_as_a_read() {
		// The deliberate divergence from LruHybridStack: a set() on a
		// slow-tier key must not promote it outright.
		let mut stack = LruLfuHybridStack::new(100, 3);

		stack.insert(1, 50);
		stack.insert(2, 50);
		stack.insert(3, 50);
		drain(&mut stack);
		assert_eq!(stack.tier_of(1), Some(Tier::Slow));

		stack.insert(1, 50); // an overwrite, freq 1 -> 2, still below k=3

		assert_eq!(stack.tier_of(1), Some(Tier::Slow), "a set must not bypass the gate");
		assert_eq!(stack.frequency_of(1), Some(2));

		stack.insert(1, 50); // freq 2 -> 3, meets k
		assert_eq!(stack.tier_of(1), Some(Tier::Fast));
	}

	// ── eviction ──────────────────────────────────────────────────────────

	#[test]
	fn evict_one_prefers_the_slow_minimum_frequency() {
		let mut stack = LruLfuHybridStack::new(100, 99);

		stack.insert(1, 50);
		stack.insert(2, 50);
		stack.insert(3, 50); // demotes 1
		stack.insert(4, 50); // demotes 2
		drain(&mut stack);

		// Both slow at freq 1; ties break least-recently-touched, so the
		// earlier-demoted key goes first.
		assert_eq!(stack.evict_one(), Some(1));
		assert_eq!(stack.evict_one(), Some(2));
	}

	#[test]
	fn evict_one_falls_back_to_the_fast_tail_when_slow_is_empty() {
		let mut stack = LruLfuHybridStack::new(1_000, K);

		stack.insert(1, 10);
		stack.insert(2, 10);
		drain(&mut stack);

		// Nothing was ever demoted, so the slow chain is empty.
		assert_eq!(stack.slow_object_count(), 0);
		assert_eq!(stack.evict_one(), Some(1), "LRU tail of the fast tier");
	}

	#[test]
	fn evict_one_on_an_empty_stack_is_none() {
		let mut stack = LruLfuHybridStack::new(1_000, K);
		assert_eq!(stack.evict_one(), None);
	}

	// ── bookkeeping ───────────────────────────────────────────────────────

	#[test]
	fn remove_updates_the_right_tier_counters() {
		let mut stack = LruLfuHybridStack::new(low_water_safe(100), 99);

		stack.insert(1, 50);
		stack.insert(2, 50);
		stack.insert(3, 50); // demotes exactly 1
		drain(&mut stack);

		assert_eq!(stack.slow_bytes_used(), 50);
		stack.remove(1);
		assert_eq!(stack.slow_bytes_used(), 0);
		assert_eq!(stack.slow_object_count(), 0);

		let fast_before = stack.fast_bytes_used();
		stack.remove(2);
		assert_eq!(stack.fast_bytes_used(), fast_before - 50);
		assert!(!stack.contains(2));
	}

	#[test]
	fn gauges_track_both_structures() {
		let mut stack = LruLfuHybridStack::new(100, 99);

		stack.insert(1, 50);
		stack.insert(2, 50);
		stack.insert(3, 50);
		drain(&mut stack);

		assert_eq!(stack.len(), 3);
		assert_eq!(
			stack.fast_object_count() + stack.slow_object_count(),
			3,
			"every tracked key is in exactly one structure",
		);
		assert_eq!(stack.fast_bytes_used() + stack.slow_bytes_used(), 150);
	}

	#[test]
	fn clear_resets_everything() {
		let mut stack = LruLfuHybridStack::new(100, K);

		stack.insert(1, 50);
		stack.insert(2, 50);
		stack.insert(3, 50);

		stack.clear();

		assert_eq!(stack.len(), 0);
		assert_eq!(stack.fast_bytes_used(), 0);
		assert_eq!(stack.slow_bytes_used(), 0);
		assert_eq!(stack.fast_object_count(), 0);
		assert_eq!(stack.slow_object_count(), 0);
		assert!(drain(&mut stack).is_empty());
	}

	#[test]
	fn resize_fast_tier_shrink_demotes_and_grow_creates_headroom() {
		let mut stack = LruLfuHybridStack::new(1_000, 99);

		stack.insert(1, 50);
		stack.insert(2, 50);
		drain(&mut stack);
		assert_eq!(stack.slow_object_count(), 0);

		stack.resize_fast_tier(50);
		let migrations = drain(&mut stack);
		assert!(!migrations.is_empty(), "shrinking should demote");
		assert!(stack.slow_object_count() > 0);

		stack.resize_fast_tier(1_000);
		assert!(drain(&mut stack).is_empty(), "growing demotes nothing");
	}

	#[test]
	fn frequency_saturates_at_the_cap() {
		let mut stack = LruLfuHybridStack::new(1_000, 99);

		stack.insert(1, 10);
		for _ in 0..(FREQUENCY_CAP as usize * 3) {
			stack.update(1);
		}

		assert_eq!(stack.frequency_of(1), Some(FREQUENCY_CAP));
	}

	#[test]
	fn promote_k_is_clamped_into_range() {
		assert_eq!(LruLfuHybridStack::new(100, 0).promote_k(), 1);
		assert_eq!(LruLfuHybridStack::new(100, 5).promote_k(), 5);
		assert_eq!(
			LruLfuHybridStack::new(100, FREQUENCY_CAP + 10).promote_k(),
			FREQUENCY_CAP,
			"a threshold above the cap would disable promotion entirely",
		);
	}

	#[test]
	fn shared_overhead_reserves_dram_and_demotes_earlier() {
		// Without the reservation, two 40-byte values fit a 100-byte tier.
		let mut plain = LruLfuHybridStack::new(100, 99);
		plain.insert(1, 40);
		plain.insert(2, 40);
		drain(&mut plain);
		assert_eq!(plain.tier_of(1), Some(Tier::Fast));
		assert_eq!(plain.tier_of(2), Some(Tier::Fast));

		// With a 30-byte per-object reservation, two objects reserve 60,
		// leaving 40 of effective value budget — tighter than 80 fits.
		let mut reserved = LruLfuHybridStack::new(100, 99).with_shared_overhead(30);
		reserved.insert(1, 40);
		reserved.insert(2, 40);
		drain(&mut reserved);
		assert!(
			reserved.slow_object_count() > 0,
			"the shared-metadata reservation should have forced a demotion",
		);
	}

	// ── watermarks ────────────────────────────────────────────────────────

	/// Fast-tier capacity for the watermark tests. Large relative to
	/// `WM_UNIT` so a triggered pass demotes a real batch and the drain
	/// target is observable rather than rounded away by one chunky object.
	const WM_CAPACITY: CacheSize = 10_000;

	/// Per-object size for the watermark tests. Small enough that
	/// `fast_used` can land within one object of the low watermark.
	const WM_UNIT: ObjectSize = 10;

	#[test]
	fn usage_below_the_high_watermark_does_not_demote() {
		let mut stack = LruLfuHybridStack::new(WM_CAPACITY, K);
		let high = watermarks::high_bytes(WM_CAPACITY);

		stack.insert(1, (high - 1) as ObjectSize);

		assert_eq!(stack.slow_object_count(), 0, "below the watermark demotes nothing");
		assert!(drain(&mut stack).is_empty(), "no migration below the watermark");

		// The trigger is strictly greater-than, so landing exactly *on* the
		// watermark must still demote nothing.
		stack.insert(2, 1);

		assert_eq!(stack.fast_bytes_used(), high);
		assert_eq!(stack.tier_of(1), Some(Tier::Fast));
		assert_eq!(stack.tier_of(2), Some(Tier::Fast));
		assert_eq!(stack.slow_object_count(), 0, "at the watermark is not over it");
		assert!(drain(&mut stack).is_empty(), "no migration at the watermark");
	}

	#[test]
	fn usage_above_the_high_watermark_triggers_a_pass() {
		let mut stack = LruLfuHybridStack::new(WM_CAPACITY, K);
		let high = watermarks::high_bytes(WM_CAPACITY);

		stack.insert(1, high as ObjectSize);
		assert!(drain(&mut stack).is_empty());

		// One byte past the high watermark is enough to trigger.
		stack.insert(2, 1);

		assert_eq!(stack.tier_of(1), Some(Tier::Slow), "the LRU tail goes first");

		let migrations = drain(&mut stack);
		assert!(
			migrations.contains(&(1, Tier::Slow)),
			"crossing the high watermark must trigger a pass; got {migrations:?}",
		);
	}

	#[test]
	fn a_triggered_pass_drains_to_the_low_watermark() {
		let mut stack = LruLfuHybridStack::new(WM_CAPACITY, K);

		let high = watermarks::high_bytes(WM_CAPACITY);
		let low = watermarks::low_bytes(WM_CAPACITY);
		let unit = WM_UNIT as CacheSize;

		// Fill right up to -- but not over -- the trigger.
		let mut key: HashedKey = 0;

		while stack.fast_bytes_used() + unit <= high {
			key += 1;
			stack.insert(key, WM_UNIT);
		}

		assert_eq!(stack.slow_object_count(), 0, "nothing has crossed the trigger yet");
		assert!(drain(&mut stack).is_empty());

		let before = stack.fast_bytes_used();

		key += 1;
		stack.insert(key, WM_UNIT); // crosses the high watermark

		assert!(
			stack.fast_bytes_used() <= low,
			"a triggered pass must drain to the low watermark ({low}), not merely back to the ceiling ({high}); fast_used = {}",
			stack.fast_bytes_used(),
		);
		assert!(
			stack.fast_bytes_used() + unit > low,
			"and must stop the moment it reaches it rather than over-draining",
		);

		// The whole point of the low watermark: one pass demotes a batch,
		// not just the single object the admission displaced.
		let expected = (before + unit - low).div_ceil(unit);

		let migrations = drain(&mut stack);
		let demoted = migrations
			.iter()
			.filter(|(_, tier)| *tier == Tier::Slow)
			.count() as CacheSize;

		assert_eq!(demoted, expected, "got {} migrations", migrations.len());
		assert_eq!(stack.slow_object_count() as CacheSize, expected);
	}

	#[test]
	fn counters_stay_consistent_across_a_watermark_pass() {
		// Comfortably more than the tier holds, so several passes run.
		let count: HashedKey = WM_CAPACITY / WM_UNIT as CacheSize + 200;

		let mut stack = LruLfuHybridStack::new(WM_CAPACITY, K);

		for key in 1..=count {
			stack.insert(key, WM_UNIT);
		}
		drain(&mut stack);

		let unit = WM_UNIT as CacheSize;

		assert!(stack.slow_object_count() > 0, "the run must have triggered a pass");

		assert_eq!(stack.len(), count as usize);
		assert_eq!(
			stack.fast_object_count() + stack.slow_object_count(),
			count as usize,
			"every tracked key is in exactly one structure",
		);
		assert_eq!(
			stack.fast_bytes_used() + stack.slow_bytes_used(),
			count * unit,
			"no bytes lost or double-counted across the demotions",
		);
		assert_eq!(
			stack.fast_bytes_used(),
			stack.fast_object_count() as CacheSize * unit,
			"fast bytes must equal the fast object count times the unit size",
		);
		assert_eq!(
			stack.slow_bytes_used(),
			stack.slow_object_count() as CacheSize * unit,
			"slow bytes must equal the slow object count times the unit size",
		);
		assert!(
			stack.fast_bytes_used() <= watermarks::high_bytes(WM_CAPACITY),
			"the tier must be settled below the trigger after the last insert",
		);
	}
}
