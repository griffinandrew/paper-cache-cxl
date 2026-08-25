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
//! ## Shared-DRAM-overhead reservation, then shared watermarks
//!
//! Like `LruHybridStack`/`LfuHybridStack`/`LruSizedHybridStack`, this stack
//! reserves DRAM for the shared per-object metadata — the object hashtable
//! entry plus this stack's own eviction-stack bookkeeping — out of the
//! fast-tier budget, so that budget bounds *total* DRAM rather than just
//! fast-tier values. `LruHybridStack`'s module doc explains why: real usage
//! measurements showed real DRAM overshooting `fast_tier_size`, because
//! neither the hashtable nor the eviction stack is counted in `fast_used`.
//!
//! Concretely: `with_shared_overhead` sets a per-tracked-object byte figure
//! (`crate::object::overhead::get_hybrid_dram_shared_overhead`, `0` unless
//! wired in — unit tests exercising the pure value-budget behaviour are
//! unaffected), `reserved_overhead()` multiplies it by the *tracked* object
//! count, and `settle_fast_tier` subtracts that from `fast_capacity` to get
//! the effective value-byte budget. The count is every tracked key, not just
//! the fast-tier ones: this stack keeps exactly one `queue` node and one
//! `entries` slot per key regardless of which tier that key's *data* sits in,
//! and those live in DRAM (unless `eviction_stacks_pmem` relocates them, in
//! which case the caller's figure drops that term).
//!
//! On top of that effective budget sits the crate-wide fast-tier watermark
//! pair (`super::watermarks`), shared with every other hybrid stack:
//! `settle_fast_tier` triggers only once `fast_used` exceeds
//! `watermarks::high_bytes(effective)`, and a triggered pass then drains all
//! the way down to `watermarks::low_bytes(effective)`. The composition order
//! is load-bearing and matches `LruHybridStack`'s exactly — the reservation
//! comes out of the capacity *first*, and the watermarks are ratios of what
//! is left, never of the raw `fast_capacity`. That replaces
//! the original pure-paper-spec rule ("trigger at `fast_capacity`, drain back
//! to exactly `fast_capacity`"), which pinned the tier at 100% utilisation
//! and made all but the first admission demote exactly one object — see
//! `super::watermarks`' doc for the migration-batch-size measurements behind
//! the change. It also supersedes the per-stack
//! `FAST_TIER_LOW_WATER_RATIO`-style headroom `LruHybridStack` had grown for
//! burst-write safety. Setting `FAST_TIER_HIGH_WATERMARK=1.0` and
//! `FAST_TIER_LOW_WATERMARK=1.0` restores the original drain-to-ceiling
//! behaviour here exactly — drain-to-the-*effective*-ceiling, that is; the
//! metadata reservation is not a watermark and does not switch off with
//! them.
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
//! `entries: HashMap<HashedKey, FifoEntry>` (`FifoEntry { dram_resident, tier, size }`)
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
	worker::policy::policy_stack::{PolicyStack, Tier, narrow_resident, watermarks},
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
struct FifoEntry {tier: Tier,
	/// Part of `size` that stays in DRAM in either tier; see `migrating`.
	dram_resident: u8,

	size: ObjectSize,
}

impl FifoEntry {
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
	std::mem::size_of::<FifoEntry>() == 8,
	"FifoEntry grew past 8 bytes",
);


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
	///
	/// Counts *every* tracked key, fast and slow alike: a demotion flips a
	/// `FifoEntry`'s tier tag and moves `fast_boundary`, it never removes the
	/// key's `queue` node or its `entries` slot, so the DRAM this is reserving
	/// against does not shrink when an object's data moves to the slow tier.
	/// Only `remove`/`evict_one`/`clear` release it. (`queue.len()` is the
	/// same count `entries.len()` would give — the two are inserted into and
	/// removed from together in every path.)
	fn reserved_overhead(&self) -> CacheSize {
		self.queue.len() as CacheSize * self.shared_overhead
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

		let old_migrating = entry.migrating();
		entry.size = new_size;
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

	/// Demotes the oldest fast key(s), triggered once `fast_used` exceeds the
	/// shared HIGH watermark of the *effective* fast-tier budget, then drained
	/// all the way down to the shared LOW watermark of that same effective
	/// budget (see `super::watermarks`, and the module doc's
	/// "Shared-DRAM-overhead reservation, then shared watermarks" section).
	///
	/// The effective budget is `fast_capacity` minus `reserved_overhead()` —
	/// the DRAM held by the shared per-object metadata (object hashtable +
	/// eviction stacks) for every tracked object of both tiers — saturating to
	/// `0` when that metadata alone meets or exceeds `fast_capacity`. The
	/// watermarks are ratios applied *on top of* that reduced value, never in
	/// place of it: the drain target is `low_bytes(capacity - reserved)`, not
	/// `low_bytes(capacity)`.
	///
	/// Demotion is the only response; capacity here never evicts (terminal
	/// eviction stays governed solely by `max_size`, handled by
	/// `PolicyWorker::apply_evictions`).
	fn settle_fast_tier(&mut self) {
		// Capacity minus the shared per-object metadata reservation. The
		// watermarks below are applied to this, never to `fast_capacity`.
		let effective = self.fast_capacity.saturating_sub(self.reserved_overhead());

		if self.fast_used <= watermarks::high_bytes(effective) {
			return;
		}

		let drain_target = watermarks::low_bytes(effective);

		while self.fast_used > drain_target {
			let Some(demote_key) = self.fast_boundary else { break };

			let size = self.entries.get(&demote_key).map(|entry| entry.migrating()).unwrap_or(0);
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
		self.insert_resident(key, size, 0);
	}

	fn insert_resident(&mut self, key: HashedKey, size: ObjectSize, dram_resident: ObjectSize) {
		let dram_resident = narrow_resident(dram_resident);
		if let Some(&FifoEntry { tier, size: old_size , .. }) = self.entries.get(&key) {
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
		self.entries.insert(key, FifoEntry { dram_resident, tier: Tier::Fast, size });
		self.fast_used += (size as CacheSize).saturating_sub(dram_resident as CacheSize);
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
		let size = entry.map(|entry| entry.migrating()).unwrap_or(0);
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
		let size = entry.map(|entry| entry.migrating()).unwrap_or(0);

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

	/// Fast capacity for the watermark tests. Large relative to
	/// `WATERMARK_TEST_OBJECT_SIZE` so the floor-rounding inside
	/// `watermarks::high_bytes`/`low_bytes` is immaterial and a triggered pass
	/// has plenty of demotion candidates to walk.
	const WATERMARK_TEST_CAPACITY: CacheSize = 1_000_000;

	/// Uniform object size for the watermark tests. Uniform sizes make the
	/// post-pass `fast_used` a closed form, so every expectation below is
	/// derived from `watermarks::high()`/`low()` rather than hardcoded and
	/// holds at any configured ratio pair. Deriving is the only option here:
	/// the watermarks are resolved once per process through `OnceLock`, so a
	/// test setting `FAST_TIER_HIGH_WATERMARK`/`FAST_TIER_LOW_WATERMARK`
	/// itself would race every other test in the same binary.
	const WATERMARK_TEST_OBJECT_SIZE: ObjectSize = 1_000;

	/// Admits keys `1..=n` of `WATERMARK_TEST_OBJECT_SIZE` bytes each until
	/// `fast_used` sits at the largest whole-object multiple still `<=` the
	/// high watermark — i.e. right on the no-trigger side of the boundary —
	/// and returns `n`. Because that leaves under one object of headroom, the
	/// caller's next same-sized admission is guaranteed to land strictly
	/// above the watermark and provoke exactly one pass.
	fn fill_to_high_watermark(stack: &mut FifoHybridStack) -> HashedKey {
		let size = WATERMARK_TEST_OBJECT_SIZE as CacheSize;
		let count = watermarks::high_bytes(stack.fast_capacity()) / size;

		assert!(
			count >= 2,
			"WATERMARK_TEST_CAPACITY is too small for the configured watermark ratios",
		);

		for key in 1..=count {
			stack.insert(key, WATERMARK_TEST_OBJECT_SIZE);
		}

		count
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
		let capacity: CacheSize = 25;
		let high = watermarks::high_bytes(capacity);
		let low = watermarks::low_bytes(capacity);

		let mut stack = FifoHybridStack::new(capacity);

		stack.insert(1, 10); // fast: [1]
		stack.insert(2, 10); // fast: [2, 1]
		drain(&mut stack);

		// This test is about the single pass that key 3's admission provokes,
		// so the first two admissions have to stay on the no-trigger side of
		// the high watermark. True for the shipped ratios; a hand-tuned
		// FAST_TIER_HIGH_WATERMARK under 0.8 would need a bigger capacity.
		assert!(20 <= high, "the first two admissions must not trigger a pass");
		assert_eq!(stack.fast_object_count(), 2);

		// fast_used 30 exceeds the high watermark at any ratio (it is above
		// the ceiling itself), so exactly one pass runs.
		stack.insert(3, 10);
		let migrations = drain(&mut stack);

		// Objects are uniformly 10 bytes and demotion walks the queue
		// oldest-first, so the demoted set is the prefix of [1, 2, 3] that
		// brings fast_used down to the LOW watermark — under the old
		// drain-to-ceiling rule this would always have been just key 1.
		let expected_demotions = (30 - low).div_ceil(10).min(3);
		let expected = (1..=expected_demotions)
			.map(|key| (key, Tier::Slow))
			.collect::<Vec<(HashedKey, Tier)>>();

		assert_eq!(migrations, expected);
		assert_eq!(stack.tier_of(1), Some(Tier::Slow));
		assert!(stack.fast_bytes_used() <= low);
		assert_eq!(stack.fast_bytes_used() + stack.slow_bytes_used(), 30);
	}

	#[test]
	fn hit_on_slow_key_does_not_migrate_or_reorder() {
		let mut stack = FifoHybridStack::new(25);

		stack.insert(1, 10);
		stack.insert(2, 10);
		stack.insert(3, 10); // fast_used 30 is over the ceiling, so over the
		                     // high watermark at any ratio -> key 1 (oldest)
		                     // is always the first thing demoted
		drain(&mut stack);

		assert_eq!(stack.tier_of(1), Some(Tier::Slow));

		// Whatever the configured watermarks drained above, a hit
		// (record_access / update) on a slow key must be a total no-op: no
		// migration, no reorder, no tier change -- this is the defining
		// difference from `LruHybridStack`'s equivalent test, where the same
		// setup would promote key 1 back to fast.
		let tiers_before = [stack.tier_of(1), stack.tier_of(2), stack.tier_of(3)];
		let fast_before = stack.fast_bytes_used();
		let slow_before = stack.slow_bytes_used();

		stack.record_access(1, true);
		let migrations = drain(&mut stack);

		assert_eq!(migrations, Vec::new());
		assert_eq!(stack.tier_of(1), Some(Tier::Slow));
		assert_eq!(
			[stack.tier_of(1), stack.tier_of(2), stack.tier_of(3)],
			tiers_before,
		);
		assert_eq!(stack.fast_bytes_used(), fast_before);
		assert_eq!(stack.slow_bytes_used(), slow_before);
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
		stack.insert(2, 10); // fast_used 20 is over the ceiling, so over the
		                     // high watermark at any ratio -> a pass runs and
		                     // demotes key 1 (oldest) first
		drain(&mut stack);

		assert_eq!(stack.tier_of(1), Some(Tier::Slow));

		// How far the configured low watermark drains past key 1 is
		// irrelevant to what these counters must agree with: a straight
		// recount of tier membership.
		fn recount(stack: &FifoHybridStack, tier: Tier) -> usize {
			[1, 2].iter().filter(|key| stack.tier_of(**key) == Some(tier)).count()
		}

		assert_eq!(stack.fast_object_count(), recount(&stack, Tier::Fast));
		assert_eq!(stack.slow_object_count(), recount(&stack, Tier::Slow));
		assert_eq!(stack.fast_object_count() + stack.slow_object_count(), 2);

		// Removing the slow key takes exactly one off the slow count and
		// leaves the fast count alone.
		let fast_before = stack.fast_object_count();
		let slow_before = stack.slow_object_count();

		stack.remove(1);

		assert_eq!(stack.fast_object_count(), fast_before);
		assert_eq!(stack.slow_object_count(), slow_before - 1);
	}

	#[test]
	fn evict_one_only_removes_from_slow_tier_once_demotions_have_happened() {
		let mut stack = FifoHybridStack::new(15);

		stack.insert(1, 10); // fast
		stack.insert(2, 10); // over the high watermark at any ratio -> demotes
		                     // key 1 (oldest) first
		drain(&mut stack);

		assert_eq!(stack.tier_of(1), Some(Tier::Slow));

		let slow_before = stack.slow_bytes_used();

		// Tail of the whole list is the slow key (1), since it was demoted
		// and is the oldest.
		assert_eq!(stack.evict_one(), Some(1));
		assert_eq!(stack.tier_of(1), None);
		assert_eq!(stack.slow_bytes_used(), slow_before - 10);
		assert_eq!(stack.len(), 1);
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

		// fast_capacity shrinks to 150 while fast_used is 200, which is over
		// the new ceiling and so over its high watermark at any ratio. The
		// eager `settle_fast_tier` then drains oldest-first down to the new
		// LOW watermark rather than merely back under 150.
		let low = watermarks::low_bytes(150);

		stack.resize_fast_tier(150);
		let migrations = drain(&mut stack);

		let expected_demotions = (200 - low).div_ceil(100).min(2);
		let expected = (1..=expected_demotions)
			.map(|key| (key, Tier::Slow))
			.collect::<Vec<(HashedKey, Tier)>>();

		assert_eq!(migrations, expected);
		assert_eq!(stack.tier_of(1), Some(Tier::Slow));
		assert!(stack.fast_bytes_used() <= low);
		assert_eq!(stack.fast_bytes_used() + stack.slow_bytes_used(), 200);
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

	// ---- shared fast-tier watermarks (`super::watermarks`) -------------
	//
	// Every expectation below is derived from `watermarks::high_bytes` /
	// `low_bytes` instead of being hardcoded, so these hold at any configured
	// ratio pair -- including `FAST_TIER_HIGH_WATERMARK=FAST_TIER_LOW_WATERMARK=1.0`,
	// which restores the original drain-to-ceiling behaviour.

	#[test]
	fn usage_below_the_high_watermark_triggers_no_demotion() {
		let mut stack = FifoHybridStack::new(WATERMARK_TEST_CAPACITY);
		let high = watermarks::high_bytes(WATERMARK_TEST_CAPACITY);

		let count = fill_to_high_watermark(&mut stack);
		let used = count * WATERMARK_TEST_OBJECT_SIZE as CacheSize;

		assert!(used <= high);
		assert_eq!(drain(&mut stack), Vec::new());
		assert_eq!(stack.fast_bytes_used(), used);
		assert_eq!(stack.slow_bytes_used(), 0);
		assert_eq!(stack.fast_object_count(), count as usize);
		assert_eq!(stack.slow_object_count(), 0);

		// Top the tier up to *exactly* the high watermark. The trigger is
		// `fast_used > high_bytes`, so sitting precisely on the watermark must
		// still be a no-op. (The remainder is under one object, hence within
		// `ObjectSize`.)
		let remainder = high - used;

		if remainder > 0 {
			stack.insert(count + 1, remainder as ObjectSize);
			assert_eq!(stack.fast_bytes_used(), high);
		}

		assert_eq!(drain(&mut stack), Vec::new());
		assert_eq!(stack.slow_bytes_used(), 0);
		assert_eq!(stack.slow_object_count(), 0);
	}

	#[test]
	fn usage_above_the_high_watermark_triggers_a_demotion_pass() {
		let mut stack = FifoHybridStack::new(WATERMARK_TEST_CAPACITY);
		let high = watermarks::high_bytes(WATERMARK_TEST_CAPACITY);

		let count = fill_to_high_watermark(&mut stack);
		assert_eq!(drain(&mut stack), Vec::new());

		// The fill stopped within one object of the watermark, so one more
		// whole object necessarily lands strictly above it.
		let used_after_admission = (count + 1) * WATERMARK_TEST_OBJECT_SIZE as CacheSize;
		assert!(
			used_after_admission > high,
			"the provoking admission must cross the high watermark",
		);

		stack.insert(count + 1, WATERMARK_TEST_OBJECT_SIZE);
		let migrations = drain(&mut stack);

		// Demotion walks the queue oldest-first, so a pass always starts at
		// key 1.
		assert!(!migrations.is_empty(), "crossing the high watermark must trigger a pass");
		assert_eq!(migrations.first(), Some(&(1, Tier::Slow)));
		assert_eq!(stack.tier_of(1), Some(Tier::Slow));
		assert_eq!(stack.tier_of(count + 1), Some(Tier::Fast));
	}

	#[test]
	fn a_triggered_pass_drains_to_the_low_watermark_not_the_ceiling() {
		let size = WATERMARK_TEST_OBJECT_SIZE as CacheSize;
		let low = watermarks::low_bytes(WATERMARK_TEST_CAPACITY);

		let mut stack = FifoHybridStack::new(WATERMARK_TEST_CAPACITY);

		let count = fill_to_high_watermark(&mut stack);
		drain(&mut stack);

		stack.insert(count + 1, WATERMARK_TEST_OBJECT_SIZE);
		let migrations = drain(&mut stack);

		// Uniform object sizes, so the pass demotes the oldest
		// `ceil(excess / size)` keys and lands on a predictable byte count.
		// `used_before_pass > high_bytes >= low_bytes`, so `excess > 0`.
		let used_before_pass = (count + 1) * size;
		let excess = used_before_pass - low;
		let expected_demotions = excess.div_ceil(size).min(count + 1);

		assert_eq!(migrations.len() as u64, expected_demotions);
		assert_eq!(
			stack.fast_bytes_used(),
			used_before_pass - expected_demotions * size,
		);
		assert!(stack.fast_bytes_used() <= low, "the pass must reach the low watermark");

		// ...and to the LOW watermark, not merely back under the ceiling:
		// whenever the watermark gap is wider than a single object this
		// demotes a whole batch, where the old drain-to-`fast_capacity` rule
		// would have demoted exactly one.
		if excess > size {
			assert!(expected_demotions > 1);
			assert!(migrations.len() > 1);
			assert!(stack.fast_bytes_used() < WATERMARK_TEST_CAPACITY);
		}
	}

	#[test]
	fn a_triggered_pass_leaves_byte_and_object_counters_consistent() {
		let size = WATERMARK_TEST_OBJECT_SIZE as CacheSize;

		let mut stack = FifoHybridStack::new(WATERMARK_TEST_CAPACITY);

		let count = fill_to_high_watermark(&mut stack);
		drain(&mut stack);

		stack.insert(count + 1, WATERMARK_TEST_OBJECT_SIZE);
		let migrations = drain(&mut stack);

		// Recount from scratch: the cached counters must agree with a full
		// walk of every admitted key's tier.
		let total_keys = count + 1;
		let mut fast_objects: u64 = 0;
		let mut slow_objects: u64 = 0;

		for key in 1..=total_keys {
			match stack.tier_of(key) {
				Some(Tier::Fast) => fast_objects += 1,
				Some(Tier::Slow) => slow_objects += 1,
				None => panic!("key {key} should still be tracked after a demotion pass"),
			}
		}

		// A demotion pass never evicts or drops anything.
		assert_eq!(stack.len() as u64, total_keys);

		assert_eq!(stack.fast_object_count() as u64, fast_objects);
		assert_eq!(stack.slow_object_count() as u64, slow_objects);
		assert_eq!(stack.fast_bytes_used(), fast_objects * size);
		assert_eq!(stack.slow_bytes_used(), slow_objects * size);
		assert_eq!(
			stack.fast_bytes_used() + stack.slow_bytes_used(),
			total_keys * size,
		);

		// Exactly one migration per demoted object -- the per-demotion
		// bookkeeping ran once each, not once for the whole batch. (The fill
		// provably emitted none, so every slow key came from this one pass.)
		assert_eq!(migrations.len() as u64, slow_objects);
		assert!(migrations.iter().all(|(_, tier)| *tier == Tier::Slow));
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

	// ---- shared per-object DRAM overhead reservation --------------------
	//
	// `with_shared_overhead` reserves `tracked keys x share` bytes out of
	// `fast_capacity` *before* the watermarks are applied, so the fast-tier
	// budget bounds total DRAM (values + the shared object hashtable + this
	// stack's own `queue`/`entries` bookkeeping) instead of just values.
	//
	// Every test above builds its stack with a plain `FifoHybridStack::new`,
	// which leaves `shared_overhead` at its default `0` -- `reserved_overhead`
	// is then `0` and `settle_fast_tier`'s effective budget is exactly
	// `fast_capacity`, so none of them is affected by this feature and none
	// needed rescaling.
	//
	// As with the watermark tests, every expectation here is derived from
	// `watermarks::high_bytes` / `low_bytes` rather than hardcoded, so they
	// hold at any configured ratio pair.

	/// Per-tracked-key shared-metadata reservation for the tests below. Equal
	/// to `WATERMARK_TEST_OBJECT_SIZE`, so `reserved_overhead()` is exactly
	/// "one object's worth of bytes per tracked key" and the arithmetic stays
	/// in whole objects.
	const OVERHEAD_TEST_SHARE: CacheSize = WATERMARK_TEST_OBJECT_SIZE as CacheSize;

	/// Number of uniform objects admitted by the single-pass tests below.
	const OVERHEAD_TEST_COUNT: u64 = 100;

	/// Per-key reservation for the single-pass tests: 100 keys x 200 bytes =
	/// 20_000 bytes reserved against a 100_000-byte ceiling, i.e. 20 whole
	/// objects' worth. Deliberately far more than one object, which is what
	/// makes the reservation's effect on the *drain target* (not just on the
	/// trigger) observable at any sane low ratio.
	const OVERHEAD_TEST_DRAIN_SHARE: CacheSize = 200;

	/// Ceiling the single-pass tests resize down to: exactly the value bytes
	/// admitted (`OVERHEAD_TEST_COUNT * WATERMARK_TEST_OBJECT_SIZE`), so the
	/// whole demotion batch is attributable to the watermark gap plus the
	/// reservation.
	const OVERHEAD_TEST_DRAIN_CAPACITY: CacheSize = 100_000;

	/// Ceiling the single-pass tests admit *under*: large enough that the
	/// admissions provably never trigger a pass of their own, so the pass the
	/// caller provokes with `resize_fast_tier` is the only one that ever ran.
	const OVERHEAD_TEST_FILL_CAPACITY: CacheSize = 10_000_000;

	/// Admits `OVERHEAD_TEST_COUNT` uniform objects into a stack carrying
	/// `share` bytes of per-key reservation, under a ceiling far too large to
	/// trigger anything, and returns it primed for exactly one
	/// `resize_fast_tier` pass.
	fn primed_for_one_pass(share: CacheSize) -> FifoHybridStack {
		let size = WATERMARK_TEST_OBJECT_SIZE as CacheSize;

		// Precondition: not even the *last* admission crosses the high
		// watermark. `fast_used` only grows and the reservation only grows
		// (shrinking the effective budget), so the trigger condition is
		// monotonic in the admission count -- failing it at the last
		// admission means it failed at every earlier one too.
		let effective = OVERHEAD_TEST_FILL_CAPACITY.saturating_sub(OVERHEAD_TEST_COUNT * share);

		assert!(
			OVERHEAD_TEST_COUNT * size <= watermarks::high_bytes(effective),
			"OVERHEAD_TEST_FILL_CAPACITY is too small for the configured high watermark",
		);

		let mut stack = FifoHybridStack::new(OVERHEAD_TEST_FILL_CAPACITY)
			.with_shared_overhead(share);

		for key in 1..=OVERHEAD_TEST_COUNT {
			stack.insert(key, WATERMARK_TEST_OBJECT_SIZE);
		}

		assert_eq!(drain(&mut stack), Vec::new(), "the priming fill must not demote anything");
		assert_eq!(stack.fast_bytes_used(), OVERHEAD_TEST_COUNT * size);
		assert_eq!(stack.slow_bytes_used(), 0);

		stack
	}

	#[test]
	fn the_reservation_counts_every_tracked_key_of_both_tiers() {
		let mut stack = FifoHybridStack::new(WATERMARK_TEST_CAPACITY)
			.with_shared_overhead(OVERHEAD_TEST_SHARE);

		assert_eq!(stack.reserved_overhead(), 0);

		stack.insert(1, WATERMARK_TEST_OBJECT_SIZE);
		assert_eq!(stack.reserved_overhead(), OVERHEAD_TEST_SHARE);

		stack.insert(2, WATERMARK_TEST_OBJECT_SIZE);
		assert_eq!(stack.reserved_overhead(), 2 * OVERHEAD_TEST_SHARE);

		// 2_000 value bytes against a 1_000_000-byte ceiling: nothing settles
		// yet (same tolerance `fill_to_high_watermark`'s `count >= 2` assert
		// already relies on).
		assert_eq!(drain(&mut stack), Vec::new());
		assert_eq!(stack.fast_object_count(), 2);

		// Demoting a key does NOT release its reservation: its `queue` node
		// and its `entries` slot are both still there, still in DRAM. Only the
		// object's *data* moved tiers.
		stack.resize_fast_tier(0);
		drain(&mut stack);

		assert_eq!(stack.fast_object_count(), 0);
		assert_eq!(stack.slow_object_count(), 2);
		assert_eq!(stack.reserved_overhead(), 2 * OVERHEAD_TEST_SHARE);

		// Actually dropping the key is what releases it.
		stack.remove(1);
		assert_eq!(stack.reserved_overhead(), OVERHEAD_TEST_SHARE);

		stack.clear();
		assert_eq!(stack.reserved_overhead(), 0);
	}

	#[test]
	fn shared_overhead_reserves_dram_and_demotes_earlier() {
		let size = WATERMARK_TEST_OBJECT_SIZE as CacheSize;

		// A fill that provably stops on the no-trigger side of the high
		// watermark when the whole capacity is available for values.
		let mut plain = FifoHybridStack::new(WATERMARK_TEST_CAPACITY);
		let count = fill_to_high_watermark(&mut plain);

		assert_eq!(drain(&mut plain), Vec::new());
		assert_eq!(plain.fast_bytes_used(), count * size);
		assert_eq!(plain.slow_bytes_used(), 0);
		assert_eq!(plain.slow_object_count(), 0);

		// The identical admission sequence, but with every tracked key also
		// reserving one object's worth of DRAM for shared metadata.
		let mut stack = FifoHybridStack::new(WATERMARK_TEST_CAPACITY)
			.with_shared_overhead(OVERHEAD_TEST_SHARE);

		for key in 1..=count {
			stack.insert(key, WATERMARK_TEST_OBJECT_SIZE);
		}

		let migrations = drain(&mut stack);

		let reserved = count * OVERHEAD_TEST_SHARE;
		let effective = WATERMARK_TEST_CAPACITY.saturating_sub(reserved);

		// Precondition, and the proof that `migrations` cannot be empty: if
		// nothing had demoted, the final admission's settle would have seen
		// `count * size` value bytes against an effective budget of
		// `capacity - reserved` -- strictly above that budget's high
		// watermark -- and would have demoted. So the reservation demotes
		// exactly where the plain stack above did not.
		assert!(
			count * size > watermarks::high_bytes(effective),
			"OVERHEAD_TEST_SHARE is too small to bite at the configured watermarks",
		);

		assert!(
			!migrations.is_empty(),
			"the reservation must demote where the unreserved stack did not",
		);

		// Demotion walks the queue oldest-first, and FIFO never reorders, so a
		// pass always starts at key 1.
		assert_eq!(migrations.first(), Some(&(1, Tier::Slow)));
		assert_eq!(stack.tier_of(1), Some(Tier::Slow));
		assert!(stack.slow_bytes_used() > 0);
		assert!(stack.fast_bytes_used() < plain.fast_bytes_used());

		// Nothing was evicted and no bytes went missing: demotion is the only
		// response, so the reservation is still the full tracked count.
		assert_eq!(stack.len() as u64, count);
		assert_eq!(stack.reserved_overhead(), reserved);
		assert_eq!(stack.fast_bytes_used() + stack.slow_bytes_used(), count * size);

		// Final-state invariant: after the last admission `fast_used` sits at
		// or below the high watermark of the *effective* budget -- either that
		// admission did not trigger, or it drained to the low watermark, which
		// is never above the high one.
		assert!(stack.fast_bytes_used() <= watermarks::high_bytes(effective));
	}

	#[test]
	fn a_triggered_pass_drains_to_the_low_watermark_of_the_reserved_budget() {
		let size = WATERMARK_TEST_OBJECT_SIZE as CacheSize;
		let used_before = OVERHEAD_TEST_COUNT * size;

		let mut stack = primed_for_one_pass(OVERHEAD_TEST_DRAIN_SHARE);

		// A demotion never drops a tracked key, so the reservation is fixed
		// for the whole pass and the pass's arithmetic is closed-form.
		let reserved = OVERHEAD_TEST_COUNT * OVERHEAD_TEST_DRAIN_SHARE;
		let effective = OVERHEAD_TEST_DRAIN_CAPACITY.saturating_sub(reserved);
		let drain_target = watermarks::low_bytes(effective);

		assert_eq!(stack.reserved_overhead(), reserved);
		assert!(
			used_before > watermarks::high_bytes(effective),
			"the resize must cross the high watermark of the effective budget",
		);

		stack.resize_fast_tier(OVERHEAD_TEST_DRAIN_CAPACITY);
		let migrations = drain(&mut stack);

		// Uniform sizes and oldest-first demotion: the pass demotes exactly
		// `ceil(excess / size)` keys, starting at key 1. (`used_before >
		// high_bytes(effective) >= low_bytes(effective)`, so `excess > 0`.)
		let expected_demotions = (used_before - drain_target)
			.div_ceil(size)
			.min(OVERHEAD_TEST_COUNT);

		let expected = (1..=expected_demotions)
			.map(|key| (key, Tier::Slow))
			.collect::<Vec<(HashedKey, Tier)>>();

		assert_eq!(migrations, expected);
		assert_eq!(stack.fast_bytes_used(), used_before - expected_demotions * size);
		assert!(stack.fast_bytes_used() <= drain_target, "the pass must reach the low watermark");
		assert_eq!(stack.fast_bytes_used() + stack.slow_bytes_used(), used_before);

		// The composition claim: the target is `low_bytes(fast_capacity -
		// reserved)`, NOT `low_bytes(fast_capacity)`. The reservation comes
		// out of the ceiling first; the watermark is a ratio of what is left.
		let unreserved_target = watermarks::low_bytes(OVERHEAD_TEST_DRAIN_CAPACITY);

		assert_eq!(
			drain_target,
			watermarks::low_bytes(OVERHEAD_TEST_DRAIN_CAPACITY - reserved),
		);

		// The 20_000-byte reservation is worth 20 whole objects at the
		// ceiling, hence at least one whole object at any low ratio >= 0.05.
		assert!(
			unreserved_target - drain_target >= size,
			"OVERHEAD_TEST_DRAIN_SHARE is too small at the configured low watermark",
		);

		// ...so applying the watermark to the raw capacity, or applying the
		// reservation after the watermark instead of before it, would have
		// demoted strictly fewer objects than this pass did.
		let unreserved_demotions = (used_before - unreserved_target)
			.div_ceil(size)
			.min(OVERHEAD_TEST_COUNT);

		assert!(
			expected_demotions > unreserved_demotions,
			"reserving DRAM must deepen the drain, not merely shift the trigger",
		);
	}

	#[test]
	fn a_triggered_pass_under_reservation_leaves_counters_consistent() {
		let size = WATERMARK_TEST_OBJECT_SIZE as CacheSize;

		let mut stack = primed_for_one_pass(OVERHEAD_TEST_DRAIN_SHARE);

		stack.resize_fast_tier(OVERHEAD_TEST_DRAIN_CAPACITY);
		let migrations = drain(&mut stack);

		assert!(!migrations.is_empty(), "the resize must trigger a pass");

		// Recount from scratch: the cached counters must agree with a full
		// walk of every admitted key's tier.
		let mut fast_objects: u64 = 0;
		let mut slow_objects: u64 = 0;

		for key in 1..=OVERHEAD_TEST_COUNT {
			match stack.tier_of(key) {
				Some(Tier::Fast) => fast_objects += 1,
				Some(Tier::Slow) => slow_objects += 1,
				None => panic!("key {key} should still be tracked after a demotion pass"),
			}
		}

		// A demotion pass never evicts or drops anything.
		assert_eq!(stack.len() as u64, OVERHEAD_TEST_COUNT);

		assert_eq!(stack.fast_object_count() as u64, fast_objects);
		assert_eq!(stack.slow_object_count() as u64, slow_objects);
		assert_eq!(stack.fast_bytes_used(), fast_objects * size);
		assert_eq!(stack.slow_bytes_used(), slow_objects * size);
		assert_eq!(
			stack.fast_bytes_used() + stack.slow_bytes_used(),
			OVERHEAD_TEST_COUNT * size,
		);

		// Exactly one migration per demoted object (the priming fill provably
		// emitted none), and the reservation itself is untouched by the pass.
		assert_eq!(migrations.len() as u64, slow_objects);
		assert!(migrations.iter().all(|(_, tier)| *tier == Tier::Slow));
		assert_eq!(
			stack.reserved_overhead(),
			OVERHEAD_TEST_COUNT * OVERHEAD_TEST_DRAIN_SHARE,
		);
	}

	#[test]
	fn shared_overhead_exceeding_capacity_demotes_all_but_never_evicts() {
		// One key's shared reservation (100) already exceeds the whole fast
		// budget (50): the effective value budget saturates to 0, so the
		// object demotes to slow immediately on admission at any watermark
		// ratio (`high_bytes(0) == low_bytes(0) == 0`).
		let mut stack = FifoHybridStack::new(50).with_shared_overhead(100);

		stack.insert(1, 10);
		let migrations = drain(&mut stack);

		assert_eq!(migrations, vec![(1, Tier::Slow)]);
		assert_eq!(stack.tier_of(1), Some(Tier::Slow));
		assert_eq!(stack.fast_bytes_used(), 0);
		assert_eq!(stack.slow_bytes_used(), 10);

		// Demotion is the only response -- the object is still tracked. The
		// DRAM budget never evicts; terminal eviction stays governed by
		// `max_size`, so `needs_capacity_eviction` keeps its default `false`.
		assert_eq!(stack.len(), 1);
		assert!(!stack.needs_capacity_eviction());
	}

	#[test]
	fn a_set_overwrite_under_reservation_does_not_double_count_the_key() {
		// FIFO's existing-key branch never touches `queue`, so re-`set()`ing a
		// tracked key must leave the reservation exactly where it was -- the
		// key still occupies one `queue` node and one `entries` slot.
		let mut stack = FifoHybridStack::new(WATERMARK_TEST_CAPACITY)
			.with_shared_overhead(OVERHEAD_TEST_SHARE);

		stack.insert(1, 10);
		stack.insert(2, 10);
		drain(&mut stack);

		let reserved_before = stack.reserved_overhead();
		assert_eq!(reserved_before, 2 * OVERHEAD_TEST_SHARE);

		stack.insert(1, 20);

		assert_eq!(stack.len(), 2);
		assert_eq!(stack.reserved_overhead(), reserved_before);
		assert_eq!(stack.fast_bytes_used(), 30);

		// ...and key 1 is still the oldest, exactly as without a reservation.
		assert_eq!(stack.evict_one(), Some(1));
		assert_eq!(stack.reserved_overhead(), OVERHEAD_TEST_SHARE);
	}
}
