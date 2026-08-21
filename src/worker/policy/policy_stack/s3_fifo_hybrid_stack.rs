/*
 * Copyright (c) Kia Shakiba
 *
 * This source code is licensed under the GNU AGPLv3 license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! `S3FifoHybridStack` — a segmented S3-FIFO stack for `PaperPolicy::S3FifoHybrid`.
//!
//! Structurally very close to `TwoQHybridStack` — a one-access FIFO queue
//! (`one_access_queue`, real objects, always entirely in the slow tier) that
//! feeds a main FIFO queue (`main_queue`, segmented fast/slow by a
//! `main_boundary` cursor, exactly like `TwoQHybridStack::main_stack`) — but
//! the *mechanism* that decides who gets to stay is different:
//!
//! * `one_access_queue` behaves exactly like `TwoQHybridStack::fifo_queue`:
//!   a re-access promotes a key **immediately, eagerly** to the front of
//!   `main_queue` at `Tier::Fast`. Anything still sitting in
//!   `one_access_queue` when it reaches the tail has therefore, by
//!   construction, never been re-accessed — so eviction from this queue is
//!   always unconditional, no bit to check.
//! * `main_queue` is **pure FIFO — never reordered on access**. Instead,
//!   every tracked `Main` key carries an `accessed: bool` "reference bit"
//!   (classic CLOCK/second-chance), set by `mark_accessed` on every touch
//!   (`get()` hit or re-`set()`), regardless of which portion (fast or
//!   slow) the key currently occupies. The bit is only ever *consulted*
//!   lazily, at the moment a key reaches the tail of `main_queue` and is
//!   about to be evicted: if set, the key is given a second chance
//!   (`give_second_chance` — reinserted at the front, retagged `Fast`, bit
//!   cleared) instead of being evicted; if clear, it's evicted for real.
//!   Demotion (fast-tier space needed) is unconditional aging of whichever
//!   key currently anchors `main_boundary` — the reference bit plays no
//!   part in demotion, only in the eviction-time second-chance check.
//!
//! This asymmetry (eager promotion for the one-access queue, lazy
//! bit-checked promotion for the main queue) is a direct reading of the
//! paper text: "if accessed again *before reaching the top* of the
//! one-access FIFO queue, they are promoted" (eager) vs. "if they *have
//! been re-accessed during this period*, they are reinserted" only
//! evaluated "after objects have traversed through both portions... and
//! are about to be evicted" (lazy, at the point of eviction).
//!
//! ## The "contiguous front run" invariant
//!
//! `main_queue` is never reordered except by two operations, both of which
//! preserve "the front `main_boundary`-worth of keys are exactly the
//! `Tier::Fast` ones, in some order; everything behind them is `Tier::Slow`":
//! insertion always happens at the front (`promote_from_one_access`,
//! `give_second_chance`) and demotion never moves anything in the list at
//! all — it only re-tags whichever key `main_boundary` currently points at
//! and walks the cursor one step toward the front (`settle_fast_tier`,
//! copied verbatim from `TwoQHybridStack`'s). A key given a second chance
//! effectively re-enters the queue as if newly promoted (moved to the
//! front) — this deliberately scrambles true insertion age in exchange for
//! matching the paper's own wording ("reinserted into the fast tier
//! portion"), exactly how a real CLOCK sweep works.
//!
//! ## No ghost queue
//!
//! Same reasoning and precedent as `TwoQHybridStack` (confirmed with the
//! user before implementing, given the source material describes exactly
//! two live queues and no ghost/ghost-hit mechanism at all): an
//! exact-membership check on every admission was judged an unwelcome added
//! cost given admission here already pays a synchronous slow-tier/PMEM
//! write. A `one_access_queue` object that ages out without a second access
//! is evicted outright, no trace kept.
//!
//! ## One combined per-key map
//!
//! Same "One combined per-key map" rationale as `TwoQHybridStack` — see
//! that stack's module doc for the full history of why this replaced
//! separate per-key maps. `S3FifoEntry` adds one field beyond
//! `TwoQEntry`: `accessed: bool`.
//!
//! ## Shared-metadata DRAM reservation
//!
//! `entries`, both queue lists and the global object hashtable all live in
//! DRAM (node 0), but `fast_used` counts object *values* only, so none of
//! that bookkeeping is visible to the fast-tier budget. `shared_overhead`
//! (set via `with_shared_overhead`) is the approximate per-tracked-key cost
//! of those shared structures; `settle_fast_tier` reserves
//! `tracked keys x shared_overhead` out of `fast_capacity` so the fast-tier
//! budget bounds total DRAM rather than just fast-tier values. Same
//! mechanism as `LruHybridStack`'s.
//!
//! It is charged against *every* tracked key, not just the `Tier::Fast`
//! ones: a `one_access_queue` resident's value sits in the slow tier, but
//! its `one_access_queue` node, its `entries` slot and its hashtable entry
//! are all still DRAM. This matches `LruHybridStack`'s `stack.len()` and
//! `LruSizedHybridStack`'s `entries.len()`. A key occupies exactly one of
//! the two queue lists at a time (`promote_from_one_access` removes it from
//! `one_access_queue` before pushing it onto `main_queue`), so the per-key
//! cost is one list node plus one `entries` slot regardless of where it is.
//!
//! There is only one independently-capacitied *fast* segment here —
//! `fast_capacity`, the fast portion of `main_queue` — so the whole
//! reservation is charged to it. `one_access_capacity` bounds slow-tier
//! bytes (`one_access_used` feeds `slow_bytes_used`), not DRAM, so unlike
//! `LruSizedHybridStack` there is no second fast budget to split the
//! reservation proportionally between.
//!
//! ## `eviction_stacks_pmem`
//!
//! Same DRAM/PMEM backing switch as `TwoQHybridStack` — see that stack's
//! module doc.

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
	worker::policy::policy_stack::{PolicyStack, Tier, watermarks},
};

/// Which live queue a key currently belongs to.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Queue {
	OneAccess,
	Main,
}

/// Combined per-key bookkeeping. `tier`/`accessed` are only meaningful while
/// `queue == Main` (`tier: None`, `accessed: false` for `OneAccess` keys —
/// see the module doc for why `OneAccess` never needs a reference bit at
/// all: its promotion is eager, not lazy).
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

pub struct S3FifoHybridStack {
	one_access_queue: QueueList,
	main_queue: QueueList,

	entries: EntryMap,

	one_access_ratio: f64,
	one_access_capacity: CacheSize,
	one_access_used: CacheSize,

	fast_capacity: CacheSize,
	fast_used: CacheSize,
	slow_used: CacheSize,

	/// Approximate per-object DRAM cost of the shared structures (object
	/// hashtable + this stack's own `entries`/queue-list bookkeeping) that
	/// hold an entry for every tracked key of both tiers. Reserved out of
	/// `fast_capacity` in `settle_fast_tier` so the fast-tier budget bounds
	/// total DRAM (values + shared metadata), not just fast-tier values. `0`
	/// unless set via `with_shared_overhead` (so unit tests exercising the
	/// pure value-budget behavior are unaffected).
	shared_overhead: CacheSize,

	/// Number of keys currently tagged `Tier::Fast` within `main_queue`.
	/// Mirrors `TwoQHybridStack::fast_count`.
	fast_count: usize,

	/// Number of keys currently in the `Main` queue (Fast or Slow). Mirrors
	/// `TwoQHybridStack::main_count`.
	main_count: usize,

	/// The oldest key currently tagged `Tier::Fast` within `main_queue` —
	/// i.e. the next demotion candidate. `None` iff no key in `main_queue`
	/// is currently Fast. Mirrors `TwoQHybridStack::main_boundary` exactly
	/// (see the module doc's "contiguous front run" section for why this
	/// invariant holds here too, despite `main_queue` never being
	/// LRU-reordered).
	main_boundary: Option<HashedKey>,

	/// (key, new tier) pairs recorded since the last `drain_tier_migrations`.
	migrations: Vec<(HashedKey, Tier)>,
}

impl S3FifoHybridStack {
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

	pub fn new(one_access_ratio: f64, max_size: CacheSize, fast_capacity: CacheSize) -> Self {
		let (one_access_queue, main_queue, entries) = Self::new_collections();

		S3FifoHybridStack {
			one_access_queue,
			main_queue,

			entries,

			one_access_ratio,
			one_access_capacity: (one_access_ratio * max_size as f64) as CacheSize,
			one_access_used: 0,

			fast_capacity,
			fast_used: 0,
			slow_used: 0,
			shared_overhead: 0,
			fast_count: 0,
			main_count: 0,

			main_boundary: None,
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

	/// The configured fast-tier byte budget, before the shared-metadata
	/// reservation is taken out of it.
	pub fn fast_capacity(&self) -> CacheSize {
		self.fast_capacity
	}

	/// Total DRAM currently reserved for shared per-object metadata across
	/// both tiers (`tracked object count x shared_overhead`). Subtracted from
	/// `fast_capacity` to form the effective value-byte budget in
	/// `settle_fast_tier`.
	///
	/// Counts `entries` — i.e. *every* tracked key, in either queue — not
	/// just the `Tier::Fast` ones: a `one_access_queue` resident's value is
	/// slow-tier, but its list node, its `entries` slot and its hashtable
	/// entry are DRAM all the same. It is therefore loop-invariant inside
	/// `settle_fast_tier`: a demotion only retags an entry, it never stops
	/// tracking one.
	fn reserved_overhead(&self) -> CacheSize {
		self.entries.len() as CacheSize * self.shared_overhead
	}

	/// Returns which queue/tier the given (currently tracked) key is in, or
	/// `None` if the key isn't tracked. Exposed for tests/diagnostics.
	pub fn tier_of(&self, key: HashedKey) -> Option<Tier> {
		let entry = self.entries.get(&key)?;

		match entry.queue {
			Queue::OneAccess => Some(Tier::Slow),
			Queue::Main => entry.tier,
		}
	}

	/// Records a size change for an already-tracked key without altering its
	/// queue/tier/accessed bit, adjusting whichever counter currently
	/// applies. Mirrors `TwoQHybridStack::resize_key`.
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

	/// Treats an already-tracked key as accessed: a `OneAccess` key promotes
	/// straight and eagerly to `Main`+`Fast`; a `Main` key just has its
	/// reference bit set (`mark_accessed`) — no reordering, no migration,
	/// no tier change, until it's evaluated at eviction time.
	fn touch(&mut self, key: HashedKey) {
		match self.entries.get(&key).map(|entry| entry.queue) {
			Some(Queue::OneAccess) => self.promote_from_one_access(key),
			Some(Queue::Main) => self.mark_accessed(key),
			None => {},
		}
	}

	/// Sets a `Main`-tracked key's reference bit. The lazy half of the
	/// promotion mechanism — see the module doc.
	fn mark_accessed(&mut self, key: HashedKey) {
		if let Some(entry) = self.entries.get_mut(&key) {
			entry.accessed = true;
		}
	}

	/// Moves a `one_access_queue`-resident key to the front of `main_queue`,
	/// tagging it `Tier::Fast` with a clear reference bit. A brand-new entry
	/// into `main_queue`, so no boundary-shift bookkeeping is needed beyond
	/// setting `main_boundary` if this is the first Fast key. Mirrors
	/// `TwoQHybridStack::promote_from_fifo`.
	fn promote_from_one_access(&mut self, key: HashedKey) {
		let Some(entry) = self.entries.get(&key) else { return };
		let size = entry.size;
		let size_bytes = size as CacheSize;

		self.one_access_queue.remove(&key);
		self.one_access_used = self.one_access_used.saturating_sub(size_bytes);

		self.main_queue.push_front(key);
		self.entries.insert(key, S3FifoEntry {
			queue: Queue::Main,
			tier: Some(Tier::Fast),
			size,
			accessed: false,
		});
		self.fast_used += size_bytes;
		self.fast_count += 1;
		self.main_count += 1;

		if self.main_boundary.is_none() {
			self.main_boundary = Some(key);
		}

		self.settle_fast_tier();

		// Pushed after `settle_fast_tier` and guarded on the key still
		// being `Fast` — see `TwoQHybridStack::promote_from_fifo`'s doc for
		// why (demotions-before-promotions ordering; an extremely tight
		// fast-tier budget can self-evict this same key within the same
		// call).
		if self.entries.get(&key).and_then(|entry| entry.tier) == Some(Tier::Fast) {
			self.migrations.push((key, Tier::Fast));
		}
	}

	/// Gives a `main_queue` key currently sitting at the tail (about to be
	/// evicted) a second chance: moves it to the front, tags it
	/// `Tier::Fast`, clears its reference bit. Handles both the normal case
	/// (the key was `Slow`) and the fallback case (the key is still `Fast`
	/// because nothing has ever been demoted — see `evict_one`'s doc) via
	/// the same `before`-then-move ordering `settle_fast_tier` uses when the
	/// key being moved currently anchors `main_boundary` itself.
	fn give_second_chance(&mut self, key: HashedKey) {
		let Some(entry) = self.entries.get(&key).copied() else { return };
		let size = entry.size as CacheSize;
		let was_fast = entry.tier == Some(Tier::Fast);
		let was_boundary = was_fast && self.main_boundary == Some(key);

		let new_boundary_if_moved = if was_boundary {
			self.main_queue.before(&key).copied()
		} else {
			None
		};

		self.main_queue.move_front(&key);

		if was_boundary {
			self.main_boundary = new_boundary_if_moved;
		}

		if let Some(entry) = self.entries.get_mut(&key) {
			entry.tier = Some(Tier::Fast);
			entry.accessed = false;
		}

		if !was_fast {
			self.slow_used = self.slow_used.saturating_sub(size);
			self.fast_used += size;
			self.fast_count += 1;
		}

		if self.main_boundary.is_none() {
			self.main_boundary = Some(key);
		}

		self.settle_fast_tier();

		if self.entries.get(&key).and_then(|entry| entry.tier) == Some(Tier::Fast) {
			self.migrations.push((key, Tier::Fast));
		}
	}

	/// Demotes the key(s) anchoring `main_boundary` — unconditional aging,
	/// no reference-bit check (the bit only ever matters at eviction time,
	/// not demotion time — see the module doc) — until `fast_used` is back
	/// under the shared *low* watermark, but only once it has crossed the
	/// shared *high* watermark in the first place (see `super::watermarks`).
	///
	/// The effective ceiling this stack works against is `fast_capacity`
	/// minus [`Self::reserved_overhead`] — the DRAM held by the shared
	/// per-object metadata (object hashtable + this stack's own `entries`
	/// and queue-list bookkeeping) for every tracked key of *both* tiers,
	/// saturating to 0 when that metadata alone meets or exceeds
	/// `fast_capacity`. That reservation is what makes this budget bound
	/// total DRAM rather than just fast-tier values. `one_access_queue` is
	/// slow-tier here (counted in `slow_bytes_used`, bounded separately by
	/// `one_access_capacity` via `needs_capacity_eviction`), so nothing
	/// *further* is carved out of the DRAM budget the way the
	/// `fast_admission` variants carve out their one-access reservation. The
	/// watermarks are applied *on top of* that effective value, never in
	/// place of it — they change only *when* a pass fires and *how far* it
	/// drains, not the budget itself.
	///
	/// Previously this drained to exactly `fast_capacity`, which pinned the
	/// tier at 100% utilisation and made essentially every promotion demote
	/// exactly one object (see the `watermarks` module doc). Setting both
	/// `FAST_TIER_HIGH_WATERMARK` and `FAST_TIER_LOW_WATERMARK` to `1.0`
	/// restores that behaviour byte-for-byte.
	///
	/// Per-demotion bookkeeping is deliberately untouched: each demoted
	/// object still retags its entry, still moves `fast_used`/`fast_count`/
	/// `slow_used` by its own size, still walks `main_boundary` one step
	/// toward the front, and still emits exactly one `Tier::Slow` migration.
	/// Fast-tier pressure is still only ever triggered by a
	/// promotion/second-chance or an explicit `resize_fast_tier`, never by
	/// every `set()` (admission never touches the fast tier directly).
	/// Structurally identical to `TwoQHybridStack::settle_fast_tier` —
	/// demotion never moves anything in the list, only re-tags whichever key
	/// the cursor points at.
	fn settle_fast_tier(&mut self) {
		// Capacity minus the shared per-object metadata reservation; the
		// watermarks below scale *this* value, never the raw capacity.
		let effective_capacity = self.fast_capacity.saturating_sub(self.reserved_overhead());

		if self.fast_used <= watermarks::high_bytes(effective_capacity) {
			return;
		}

		let drain_target = watermarks::low_bytes(effective_capacity);

		while self.fast_used > drain_target {
			let Some(demote_key) = self.main_boundary else { break };

			let size = self.entries.get(&demote_key).map(|entry| entry.size).unwrap_or(0) as CacheSize;
			let new_boundary = self.main_queue.before(&demote_key).copied();

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

	/// Pops and fully removes `one_access_queue`'s tail from this stack's
	/// own bookkeeping, if any — unconditional, since anything still here
	/// has (by construction of the eager-promotion rule) never been
	/// re-accessed. Same "cannot self-evict from insert/resize" rationale
	/// as `TwoQHybridStack::evict_fifo_tail` — only called from
	/// `evict_one`, never from `insert`/`resize`.
	fn evict_one_access_tail(&mut self) -> Option<HashedKey> {
		let key = self.one_access_queue.pop_back()?;
		let size = self.entries.remove(&key).map(|entry| entry.size).unwrap_or(0) as CacheSize;

		self.one_access_used = self.one_access_used.saturating_sub(size);

		Some(key)
	}
}

impl PolicyStack for S3FifoHybridStack {
	fn is_policy(&self, policy: &PaperPolicy) -> bool {
		matches!(policy, PaperPolicy::S3FifoHybrid(ratio) if *ratio == self.one_access_ratio)
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
			// (a re-`set()` is a genuine reference, same as `TwoQHybridStack`).
			self.resize_key(key, size);
			self.touch(key);
			return;
		}

		// Brand-new key: always admitted into the one-access queue, always
		// slow. If this pushes one_access_used over one_access_capacity,
		// `needs_capacity_eviction` reports it and `apply_evictions` drains
		// it via the real `evict_one()` path (see that method's doc for why
		// eviction can't happen here).
		self.one_access_queue.push_front(key);
		self.entries.insert(key, S3FifoEntry {
			queue: Queue::OneAccess,
			tier: None,
			size,
			accessed: false,
		});
		self.one_access_used += size as CacheSize;
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

			Queue::Main => {
				let new_boundary_if_needed = if entry.tier == Some(Tier::Fast) && self.main_boundary == Some(key) {
					self.main_queue.before(&key).copied()
				} else {
					None
				};

				self.main_queue.remove(&key);
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
		self.one_access_capacity = (self.one_access_ratio * max_size as f64) as CacheSize;
		// A shrink may push one_access_used over the new, smaller capacity;
		// `needs_capacity_eviction` reports it, `apply_evictions` drains it.
	}

	fn clear(&mut self) {
		self.one_access_queue.clear();
		self.main_queue.clear();
		self.entries.clear();

		self.one_access_used = 0;
		self.fast_used = 0;
		self.slow_used = 0;
		self.fast_count = 0;
		self.main_count = 0;
		self.main_boundary = None;
		self.migrations.clear();
	}

	/// Priority: `one_access_queue`'s tail first (unconditional — see
	/// `evict_one_access_tail`'s doc); otherwise sweeps `main_queue`'s tail,
	/// giving repeated second chances to accessed keys (classic CLOCK sweep
	/// — bounded by `main_queue`'s length, since each second chance clears
	/// that key's bit and moves it to the front, so it can't be
	/// re-examined again until a full lap) until it finds one with a clear
	/// reference bit to actually evict, or the queue is exhausted.
	fn evict_one(&mut self) -> Option<HashedKey> {
		if let Some(key) = self.evict_one_access_tail() {
			return Some(key);
		}

		loop {
			let key = *self.main_queue.back()?;
			let accessed = self.entries.get(&key).map(|entry| entry.accessed).unwrap_or(false);

			if accessed {
				self.give_second_chance(key);
				continue;
			}

			self.main_queue.pop_back();
			let removed = self.entries.remove(&key);
			let size = removed.map(|entry| entry.size).unwrap_or(0) as CacheSize;
			let tier = removed.and_then(|entry| entry.tier);

			self.main_count = self.main_count.saturating_sub(1);

			match tier {
				Some(Tier::Fast) => {
					self.fast_used = self.fast_used.saturating_sub(size);
					self.fast_count = self.fast_count.saturating_sub(1);

					if self.main_boundary == Some(key) {
						self.main_boundary = self.main_queue.back().copied();
					}
				},

				Some(Tier::Slow) => {
					self.slow_used = self.slow_used.saturating_sub(size);
				},

				None => {},
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

	fn fast_bytes_used(&self) -> CacheSize {
		self.fast_used
	}

	fn slow_bytes_used(&self) -> CacheSize {
		self.one_access_used + self.slow_used
	}

	fn fast_object_count(&self) -> usize {
		self.fast_count
	}

	fn slow_object_count(&self) -> usize {
		self.one_access_queue.len() + (self.main_count - self.fast_count)
	}

	fn needs_capacity_eviction(&self) -> bool {
		self.one_access_used > self.one_access_capacity
	}
}

#[cfg(test)]
mod tests {
	use super::*;

	fn drain(stack: &mut S3FifoHybridStack) -> Vec<(HashedKey, Tier)> {
		stack.drain_tier_migrations()
	}

	/// `insert` + `update` -- the insert-into-`one_access_queue`-then-promote
	/// pairing every fast-tier test in this module already uses inline, since
	/// this stack never admits a fresh key straight into `main_queue`'s fast
	/// tier.
	fn promote(stack: &mut S3FifoHybridStack, key: HashedKey, size: ObjectSize) {
		stack.insert(key, size);
		stack.update(key);
	}

	/// Smallest fast-tier capacity whose *low* watermark still leaves room for
	/// `bytes`. Lets the fast-tier tests state their expectations in whole
	/// objects instead of hard-coded byte thresholds, so they hold at whatever
	/// `FAST_TIER_HIGH_WATERMARK`/`FAST_TIER_LOW_WATERMARK` pair is configured
	/// rather than only at the default ratios. (The watermarks are
	/// process-global `OnceLock`s, so a test cannot set the env vars itself
	/// without racing every other test in the binary.) The `while` loop absorbs
	/// the truncation in `watermarks::low_bytes`' `as u64` cast, which a bare
	/// `ceil()` on its own can land a byte short of.
	fn capacity_holding(bytes: CacheSize) -> CacheSize {
		let mut capacity = (bytes as f64 / watermarks::low()).ceil() as CacheSize;

		while watermarks::low_bytes(capacity) < bytes {
			capacity += 1;
		}

		capacity
	}

	/// Smallest fast-tier capacity whose *high* watermark still admits `bytes`
	/// — so `capacity_admitting(bytes) - 1` is the largest budget at which
	/// `bytes` resident is guaranteed to trip a demotion pass, at whatever
	/// `FAST_TIER_HIGH_WATERMARK` is configured. Mirrors `LruHybridStack`'s
	/// helper of the same name; used by the shared-overhead tests to pin the
	/// trigger point exactly rather than hard-coding a byte threshold. The
	/// `while` loop absorbs the truncation in `watermarks::high_bytes`' `as
	/// u64` cast, which a bare `ceil()` on its own can land a byte short of.
	fn capacity_admitting(bytes: CacheSize) -> CacheSize {
		let mut capacity = (bytes as f64 / watermarks::high()).ceil() as CacheSize;

		while watermarks::high_bytes(capacity) < bytes {
			capacity += 1;
		}

		capacity
	}

	/// A stack holding `count` promoted `size`-byte keys (so `one_access_queue`
	/// is empty and every key is `Main`/`Fast`), built at `fast_capacity` and
	/// with its migration log already drained. Callers pass a `fast_capacity`
	/// far above every threshold so the fill itself settles nothing, then drive
	/// one deterministic pass with `resize_fast_tier`.
	fn filled(
		fast_capacity: CacheSize,
		overhead: CacheSize,
		count: CacheSize,
		size: ObjectSize,
	) -> S3FifoHybridStack {
		let mut stack = S3FifoHybridStack::new(1.0, 1_000_000, fast_capacity)
			.with_shared_overhead(overhead);

		for key in 1..=count {
			promote(&mut stack, key, size);
		}

		drain(&mut stack);
		stack
	}

	#[test]
	fn admission_always_lands_in_one_access_queue_slow() {
		let mut stack = S3FifoHybridStack::new(1.0, 1_000, 1_000);

		stack.insert(1, 10);
		stack.insert(2, 10);

		assert_eq!(stack.tier_of(1), Some(Tier::Slow));
		assert_eq!(stack.tier_of(2), Some(Tier::Slow));
		assert_eq!(stack.slow_bytes_used(), 20);
		assert_eq!(stack.fast_bytes_used(), 0);
		assert_eq!(drain(&mut stack), Vec::new());
	}

	#[test]
	fn reaccessing_a_one_access_key_promotes_it_eagerly_to_fast() {
		let mut stack = S3FifoHybridStack::new(1.0, 1_000, 1_000);

		stack.insert(1, 10);
		drain(&mut stack);

		stack.update(1);
		let migrations = drain(&mut stack);

		assert_eq!(migrations, vec![(1, Tier::Fast)]);
		assert_eq!(stack.tier_of(1), Some(Tier::Fast));
	}

	#[test]
	fn one_access_capacity_pressure_is_reported_not_self_evicted() {
		let mut stack = S3FifoHybridStack::new(1.0, 15, 1_000);

		stack.insert(1, 10);
		drain(&mut stack);
		assert_eq!(stack.contains(1), true);
		assert_eq!(stack.needs_capacity_eviction(), false);

		stack.insert(2, 10);
		let migrations = drain(&mut stack);

		assert_eq!(migrations, Vec::new());
		assert_eq!(stack.contains(1), true);
		assert_eq!(stack.contains(2), true);
		assert_eq!(stack.needs_capacity_eviction(), true);

		assert_eq!(stack.evict_one(), Some(1));
		assert_eq!(stack.contains(1), false);
		assert_eq!(stack.needs_capacity_eviction(), false);
		assert_eq!(stack.tier_of(2), Some(Tier::Slow));
	}

	#[test]
	fn fast_tier_pressure_within_main_queue_demotes_the_oldest() {
		// Sized so both 10-byte objects sit under the low watermark while the
		// third crosses the high one -- i.e. the triggered pass demotes exactly
		// the oldest fast key. (Was a hard-coded 25: correct back when a pass
		// drained to the ceiling, but a 25-byte ceiling now triggers at 23 and
		// drains to 18, which would take key 2 with it.)
		let fast_capacity = capacity_holding(20);

		let mut stack = S3FifoHybridStack::new(1.0, 1_000, fast_capacity);

		stack.insert(1, 10);
		stack.insert(2, 10);
		stack.update(1); // promote 1 -> Main/Fast
		stack.update(2); // promote 2 -> Main/Fast
		drain(&mut stack);

		assert_eq!(stack.fast_bytes_used(), 20);

		stack.insert(3, 10);
		stack.update(3); // promote 3 -> pushes fast_used past the high watermark
		let migrations = drain(&mut stack);

		assert!(migrations.iter().any(|(k, t)| *k == 1 && *t == Tier::Slow));
		assert_eq!(stack.tier_of(1), Some(Tier::Slow));
		assert_eq!(stack.tier_of(2), Some(Tier::Fast));
		assert_eq!(stack.tier_of(3), Some(Tier::Fast));
		assert!(stack.fast_bytes_used() <= watermarks::low_bytes(fast_capacity));
	}

	#[test]
	fn a_mere_access_never_reorders_the_main_queue_or_migrates() {
		// Unlike TwoQHybridStack (LRU main queue), touching a Main/Fast key
		// here must NOT produce a migration or move it in the list -- it
		// only sets the reference bit, consulted lazily at eviction time.
		let mut stack = S3FifoHybridStack::new(1.0, 1_000, 1_000);

		stack.insert(1, 10);
		stack.update(1); // promote to Main/Fast
		drain(&mut stack);

		stack.update(1); // a plain access while already Fast
		let migrations = drain(&mut stack);

		assert_eq!(migrations, Vec::new());
		assert_eq!(stack.tier_of(1), Some(Tier::Fast));
	}

	#[test]
	fn evict_one_gives_an_accessed_slow_key_a_second_chance() {
		// Fast tier settles at exactly one 10-byte object at a time. (Was a
		// hard-coded 10: correct back when a pass drained to the ceiling, but a
		// 10-byte ceiling now triggers at 9 and drains to 7, so even a lone
		// resident object would be demoted straight back out.)
		let mut stack = S3FifoHybridStack::new(1.0, 1_000, capacity_holding(10));

		stack.insert(1, 10);
		stack.update(1); // promote 1 -> Fast
		drain(&mut stack);

		stack.insert(2, 10);
		stack.update(2); // promote 2 -> Fast, demotes 1 -> Slow (past the high watermark)
		drain(&mut stack);
		assert_eq!(stack.tier_of(1), Some(Tier::Slow));

		// Access key 1 while it's Slow -- lazily sets its bit, no reorder.
		stack.update(1);
		assert_eq!(drain(&mut stack), Vec::new());
		assert_eq!(stack.tier_of(1), Some(Tier::Slow));

		// Force an eviction sweep: main_queue's tail is key 1 (the only
		// Slow entry). It's accessed, so it gets a second chance (promoted
		// to Fast, which itself demotes key 2 to make room) instead of
		// being evicted; the sweep then evicts key 2 instead, since it now
		// has a clear bit.
		let evicted = stack.evict_one();

		assert_eq!(evicted, Some(2));
		assert_eq!(stack.tier_of(1), Some(Tier::Fast));
		assert_eq!(stack.contains(2), false);
	}

	#[test]
	fn evict_one_evicts_an_unaccessed_slow_tail_directly() {
		// Same watermark-derived "one object stays resident" capacity as
		// `evict_one_gives_an_accessed_slow_key_a_second_chance` above.
		let mut stack = S3FifoHybridStack::new(1.0, 1_000, capacity_holding(10));

		stack.insert(1, 10);
		stack.update(1);
		drain(&mut stack);

		stack.insert(2, 10);
		stack.update(2); // demotes 1 -> Slow
		drain(&mut stack);
		assert_eq!(stack.tier_of(1), Some(Tier::Slow));

		// Key 1 was never re-accessed while Slow -- evicted directly.
		assert_eq!(stack.evict_one(), Some(1));
		assert_eq!(stack.tier_of(2), Some(Tier::Fast));
	}

	#[test]
	fn evict_one_prefers_one_access_queue_over_main_queue() {
		let mut stack = S3FifoHybridStack::new(1.0, 1_000, 1_000);

		stack.insert(1, 10); // one-access
		stack.insert(2, 10);
		stack.update(2); // promote 2 -> Main/Fast
		drain(&mut stack);

		assert_eq!(stack.evict_one(), Some(1));
		assert_eq!(stack.tier_of(2), Some(Tier::Fast));
	}

	#[test]
	fn evict_one_falls_back_to_main_fast_when_nothing_demoted_yet() {
		// Both 10-byte objects must stay Fast, so the capacity has to hold 20
		// bytes below even the *low* watermark -- nothing may trigger here.
		let mut stack = S3FifoHybridStack::new(1.0, 1_000, capacity_holding(20));

		stack.insert(1, 10);
		stack.insert(2, 10);
		stack.update(1);
		stack.update(2);
		drain(&mut stack);

		// one_access_queue empty; both keys are Main/Fast (nothing demoted
		// yet), neither has been accessed since being promoted -- evicts
		// the oldest (key 1) directly.
		assert_eq!(stack.evict_one(), Some(1));
		assert_eq!(stack.tier_of(2), Some(Tier::Fast));
	}

	#[test]
	fn resize_rescales_one_access_capacity_and_reports_pressure() {
		let mut stack = S3FifoHybridStack::new(0.5, 1_000, 1_000); // capacity = 500

		stack.insert(1, 100);
		stack.insert(2, 100);
		drain(&mut stack);
		assert_eq!(stack.slow_bytes_used(), 200);
		assert_eq!(stack.needs_capacity_eviction(), false);

		stack.resize(100); // capacity -> 50 -> both keys now exceed it

		assert_eq!(stack.contains(1), true);
		assert_eq!(stack.contains(2), true);
		assert_eq!(stack.needs_capacity_eviction(), true);

		assert_eq!(stack.evict_one(), Some(1));
		assert_eq!(stack.evict_one(), Some(2));
		assert_eq!(stack.needs_capacity_eviction(), false);
	}

	#[test]
	fn resize_fast_tier_shrink_triggers_demotions() {
		let mut stack = S3FifoHybridStack::new(1.0, 1_000, 1_000);

		stack.insert(1, 10);
		stack.insert(2, 10);
		stack.update(1);
		stack.update(2);
		drain(&mut stack);

		// Shrink to a capacity whose low watermark still holds one of the two
		// 10-byte objects, so the pass demotes exactly one (was a bare 10,
		// which under the watermarks drains the tier empty instead).
		let shrunk = capacity_holding(10);

		stack.resize_fast_tier(shrunk);
		let migrations = drain(&mut stack);

		assert_eq!(migrations.len(), 1);
		assert_eq!(stack.fast_bytes_used(), 10);
		assert!(stack.fast_bytes_used() <= watermarks::low_bytes(shrunk));
	}

	#[test]
	fn remove_and_clear_reset_bookkeeping() {
		let mut stack = S3FifoHybridStack::new(1.0, 1_000, 1_000);

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

	// ---- fast-tier watermarks (`super::watermarks`) ----

	/// (a) The trigger is a strict `>`, so usage sitting right *on* the high
	/// watermark -- the largest usage that is not over it -- must leave the
	/// tier completely alone, even though it is well above the low watermark.
	#[test]
	fn fast_usage_at_the_high_watermark_triggers_no_demotion() {
		let fast_capacity: CacheSize = 1_000;
		let high = watermarks::high_bytes(fast_capacity);

		let mut stack = S3FifoHybridStack::new(1.0, 100_000, fast_capacity);

		// Two objects summing to exactly the high watermark.
		promote(&mut stack, 1, (high - 1) as ObjectSize);
		promote(&mut stack, 2, 1);

		let migrations = drain(&mut stack);

		assert_eq!(stack.fast_bytes_used(), high);
		assert!(
			!migrations.iter().any(|(_, tier)| *tier == Tier::Slow),
			"usage at the high watermark must not trigger a demotion pass, got {migrations:?}",
		);
		assert_eq!(stack.tier_of(1), Some(Tier::Fast));
		assert_eq!(stack.tier_of(2), Some(Tier::Fast));
		assert!(stack.fast_bytes_used() >= watermarks::low_bytes(fast_capacity));
		// `one_access_queue` is drained by `promote`, so the slow side is
		// genuinely empty rather than merely holding the pre-promotion copies.
		assert_eq!(stack.slow_bytes_used(), 0);
	}

	/// (b) One byte past the high watermark -- the smallest possible overshoot
	/// -- must fire a pass, and it must take `main_queue`'s oldest fast key
	/// rather than the key that just arrived.
	#[test]
	fn fast_usage_above_the_high_watermark_triggers_a_demotion_pass() {
		let fast_capacity: CacheSize = 1_000;
		let high = watermarks::high_bytes(fast_capacity);

		let mut stack = S3FifoHybridStack::new(1.0, 100_000, fast_capacity);

		promote(&mut stack, 1, high as ObjectSize);
		assert!(
			!drain(&mut stack).iter().any(|(_, tier)| *tier == Tier::Slow),
			"filling exactly to the high watermark must not demote anything yet",
		);

		promote(&mut stack, 2, 1);
		let migrations = drain(&mut stack);

		assert!(
			migrations.contains(&(1, Tier::Slow)),
			"usage past the high watermark must trigger a demotion pass, got {migrations:?}",
		);
		assert_eq!(stack.tier_of(1), Some(Tier::Slow));
		assert_eq!(stack.tier_of(2), Some(Tier::Fast));
		assert!(stack.fast_bytes_used() <= watermarks::low_bytes(fast_capacity));
	}

	/// (c) A triggered pass keeps going down to the *low* watermark, not just
	/// back under the ceiling. With the defaults this drains 960 -> 750 across
	/// 21 demotions; the pre-watermark drain-to-ceiling loop would have stopped
	/// after a single one, at 950.
	#[test]
	fn a_triggered_pass_drains_all_the_way_to_the_low_watermark() {
		let fast_capacity: CacheSize = 1_000;
		let size: ObjectSize = 10;
		let bytes = size as CacheSize;

		let high = watermarks::high_bytes(fast_capacity);
		let low = watermarks::low_bytes(fast_capacity);

		let mut stack = S3FifoHybridStack::new(1.0, 100_000, fast_capacity);

		// Exactly one object past the high watermark, so precisely one pass
		// fires -- with plenty of resident objects for it to chew through
		// before it reaches the low watermark.
		let count = high / bytes + 1;

		for key in 1..=count {
			promote(&mut stack, key, size);
		}

		let migrations = drain(&mut stack);
		let demoted = migrations.iter().filter(|(_, tier)| *tier == Tier::Slow).count() as CacheSize;

		// The pass halts at the first whole-object multiple at or below the low
		// watermark -- well under `fast_capacity`, which is where the old loop
		// would have left it.
		let expected_used = low - low % bytes;

		assert_eq!(stack.fast_bytes_used(), expected_used);
		assert!(stack.fast_bytes_used() <= low);
		assert_eq!(demoted, (count * bytes - expected_used) / bytes);

		// The whole point of the low watermark: the pass keeps going well past
		// the ceiling the old rule stopped at. Skipped when the low watermark
		// is configured back to 1.0 (drain-to-ceiling).
		if watermarks::low() < 1.0 {
			assert!(stack.fast_bytes_used() < fast_capacity);
		}

		// Demotion order is oldest-fast-key-first, exactly as before the
		// watermarks -- only the number of them per pass changed.
		assert_eq!(stack.tier_of(1), Some(Tier::Slow));
		assert_eq!(stack.tier_of(count), Some(Tier::Fast));
	}

	/// (d) Every byte counter and object count still agrees with the per-key
	/// tier tags once a full watermark drain has run -- the same per-demotion
	/// bookkeeping must have run once per demoted object, no more and no less.
	#[test]
	fn byte_and_object_counters_stay_consistent_across_a_watermark_drain() {
		let fast_capacity: CacheSize = 1_000;
		let size: ObjectSize = 10;
		let bytes = size as CacheSize;

		let count = watermarks::high_bytes(fast_capacity) / bytes + 1;

		let mut stack = S3FifoHybridStack::new(1.0, 100_000, fast_capacity);

		for key in 1..=count {
			promote(&mut stack, key, size);
		}

		drain(&mut stack);

		let fast_objects = stack.fast_object_count() as CacheSize;
		let slow_objects = stack.slow_object_count() as CacheSize;

		// Nothing was inserted, evicted or resized mid-pass, so every object is
		// still tracked, still `size` bytes, and still on exactly one side of
		// the fast/slow line. `one_access_queue` is empty here (every key was
		// promoted out of it), so `slow_*` is purely the demoted main-queue
		// tail.
		assert!(fast_objects > 0 && slow_objects > 0);
		assert_eq!(fast_objects + slow_objects, count);
		assert_eq!(stack.len() as CacheSize, count);

		assert_eq!(stack.fast_bytes_used(), fast_objects * bytes);
		assert_eq!(stack.slow_bytes_used(), slow_objects * bytes);
		assert_eq!(stack.fast_bytes_used() + stack.slow_bytes_used(), count * bytes);

		// And the aggregate counts agree with the per-key tier tags.
		let tagged_fast = (1..=count).filter(|key| stack.tier_of(*key) == Some(Tier::Fast)).count();
		let tagged_slow = (1..=count).filter(|key| stack.tier_of(*key) == Some(Tier::Slow)).count();

		assert_eq!(tagged_fast as CacheSize, fast_objects);
		assert_eq!(tagged_slow as CacheSize, slow_objects);
	}

	// ---- shared-metadata DRAM reservation (`with_shared_overhead`) ----
	//
	// Every test above constructs the stack with plain `new`, which leaves
	// `shared_overhead` at 0, so `reserved_overhead()` is 0 and the effective
	// budget is exactly `fast_capacity` — i.e. none of them change behaviour
	// under this feature, which is why none of them needed rescaling.

	/// `reserved_overhead` charges every *tracked* key, whichever queue it is
	/// in: a `one_access_queue` resident's value is slow-tier, but its list
	/// node, its `entries` slot and its hashtable entry are DRAM all the same.
	/// Moving a key between the two queues must not change the charge — a key
	/// occupies exactly one queue list at a time.
	#[test]
	fn reserved_overhead_charges_every_tracked_key_in_both_queues() {
		const OVERHEAD: CacheSize = 7;

		// Capacity far above any threshold, so nothing settles here.
		let mut stack = S3FifoHybridStack::new(1.0, 1_000_000, 1_000_000)
			.with_shared_overhead(OVERHEAD);

		assert_eq!(stack.fast_capacity(), 1_000_000);
		assert_eq!(stack.reserved_overhead(), 0);

		stack.insert(1, 10); // one_access_queue, Tier::Slow -- still charged
		assert_eq!(stack.tier_of(1), Some(Tier::Slow));
		assert_eq!(stack.reserved_overhead(), OVERHEAD);

		stack.update(1); // promoted into main_queue/Fast: same key, same charge
		assert_eq!(stack.tier_of(1), Some(Tier::Fast));
		assert_eq!(stack.reserved_overhead(), OVERHEAD);

		stack.insert(2, 10);
		assert_eq!(stack.reserved_overhead(), 2 * OVERHEAD);

		stack.remove(1);
		assert_eq!(stack.reserved_overhead(), OVERHEAD);

		stack.clear();
		assert_eq!(stack.reserved_overhead(), 0);

		// A stack built without `with_shared_overhead` reserves nothing, which
		// is exactly why every pre-existing test in this module is unaffected.
		let mut plain = S3FifoHybridStack::new(1.0, 1_000, 1_000);

		plain.insert(1, 10);
		plain.update(1);

		assert_eq!(plain.reserved_overhead(), 0);
	}

	/// The reservation shrinks the effective fast budget, so a stack carrying
	/// shared overhead demotes where an otherwise identical one — same
	/// capacity, same objects — does not.
	#[test]
	fn shared_overhead_reserves_dram_and_demotes_earlier() {
		let size: ObjectSize = 10;
		let bytes = size as CacheSize;
		let count: CacheSize = 8;
		let total = count * bytes; // 80 bytes of values, all Main/Fast
		let overhead: CacheSize = 30;
		let reserved = count * overhead; // 240 bytes of shared DRAM metadata

		// One byte under the smallest budget whose high watermark admits all
		// `total` resident bytes: `total` against exactly this effective budget
		// is guaranteed to trip a pass at any configured ratios. Adding
		// `reserved` back on gives the raw capacity, which is therefore
		// `reserved` bytes clear of the trigger point on its own.
		let effective = capacity_admitting(total) - 1;
		let capacity = effective + reserved;

		// Same capacity, no reservation. `capacity >= capacity_admitting(total)`
		// (since `reserved >= 1`), so nothing ever crosses the high watermark.
		let mut plain = S3FifoHybridStack::new(1.0, 1_000_000, capacity);

		for key in 1..=count {
			promote(&mut plain, key, size);
		}

		let plain_migrations = drain(&mut plain);

		assert!(
			!plain_migrations.iter().any(|(_, tier)| *tier == Tier::Slow),
			"without the reservation this capacity must not demote, got {plain_migrations:?}",
		);
		assert_eq!(plain.fast_bytes_used(), total);
		assert_eq!(plain.slow_bytes_used(), 0);

		// Same capacity, 30 bytes reserved per tracked key.
		let mut stack = S3FifoHybridStack::new(1.0, 1_000_000, capacity)
			.with_shared_overhead(overhead);

		for key in 1..=count {
			promote(&mut stack, key, size);
		}

		let migrations = drain(&mut stack);

		assert_eq!(stack.fast_capacity(), capacity);
		assert_eq!(stack.reserved_overhead(), reserved);

		// Composition: the pass drains to `low_bytes(capacity - reserved)`, NOT
		// `low_bytes(capacity)` — the `plain` stack above shows the raw
		// capacity's watermarks leave all `total` bytes resident. It halts at
		// the first whole-object multiple at or below that target.
		let target = watermarks::low_bytes(capacity - reserved);
		let expected_used = target - target % bytes;

		assert_eq!(stack.fast_bytes_used(), expected_used);
		assert!(
			expected_used < total,
			"the reservation must have forced at least one demotion",
		);

		let demoted = migrations.iter().filter(|(_, tier)| *tier == Tier::Slow).count() as CacheSize;

		assert_eq!(demoted, (total - expected_used) / bytes);

		// Oldest-fast-key-first, exactly as without the reservation — only the
		// budget the pass measures against changed.
		assert_eq!(stack.tier_of(1), Some(Tier::Slow));

		if expected_used > 0 {
			assert_eq!(stack.tier_of(count), Some(Tier::Fast));
		}
	}

	/// The reservation moves *both* watermark thresholds onto the effective
	/// budget, not just the drain target: usage sitting exactly on
	/// `high_bytes(capacity - reserved)` still leaves the tier alone, one byte
	/// less of raw capacity fires a pass, and that pass drains to
	/// `low_bytes(capacity - reserved)`. Neither capacity does anything at all
	/// without the reservation.
	#[test]
	fn overhead_moves_both_watermarks_onto_the_effective_budget() {
		let size: ObjectSize = 10;
		let bytes = size as CacheSize;
		let count: CacheSize = 4;
		let total = count * bytes; // 40
		let overhead: CacheSize = 30;
		let reserved = count * overhead; // 120

		// The raw capacity whose effective budget is exactly
		// `capacity_admitting(total)` -- the trigger point, inclusive.
		let at_threshold = capacity_admitting(total) + reserved;

		let mut stack = filled(1_000_000, overhead, count, size);

		assert_eq!(stack.fast_bytes_used(), total);
		assert_eq!(stack.reserved_overhead(), reserved);

		stack.resize_fast_tier(at_threshold);
		let migrations = drain(&mut stack);

		assert_eq!(
			migrations,
			Vec::new(),
			"usage exactly at the effective budget's high watermark must not demote",
		);
		assert_eq!(stack.fast_bytes_used(), total);

		// One byte less of raw capacity is one byte less of effective budget,
		// which puts usage past that same high watermark.
		stack.resize_fast_tier(at_threshold - 1);
		let migrations = drain(&mut stack);

		let target = watermarks::low_bytes(at_threshold - 1 - reserved);
		let expected_used = target - target % bytes;

		assert!(
			!migrations.is_empty(),
			"one byte past the effective high watermark must fire a pass",
		);
		assert!(migrations.iter().all(|(_, tier)| *tier == Tier::Slow));
		assert_eq!(stack.fast_bytes_used(), expected_used);
		assert!(expected_used < total);
		assert_eq!(stack.tier_of(1), Some(Tier::Slow));

		// Both capacities are >= `capacity_admitting(total)`, so without the
		// reservation neither one demotes a single object.
		let mut plain = filled(1_000_000, 0, count, size);

		plain.resize_fast_tier(at_threshold);
		assert_eq!(drain(&mut plain), Vec::new());

		plain.resize_fast_tier(at_threshold - 1);
		assert_eq!(drain(&mut plain), Vec::new());
		assert_eq!(plain.fast_bytes_used(), total);
		assert_eq!(plain.reserved_overhead(), 0);
	}

	/// Every byte counter and object count still agrees with the per-key tier
	/// tags once an overhead-reserved drain has run: the same per-demotion
	/// bookkeeping ran once per demoted object, no more and no less, and the
	/// reservation demoted without evicting anything.
	#[test]
	fn counters_stay_consistent_across_an_overhead_reserved_drain() {
		let size: ObjectSize = 10;
		let bytes = size as CacheSize;
		let count: CacheSize = 8;
		let total = count * bytes;
		let overhead: CacheSize = 30;
		let reserved = count * overhead;

		let capacity = (capacity_admitting(total) - 1) + reserved;

		let mut stack = S3FifoHybridStack::new(1.0, 1_000_000, capacity)
			.with_shared_overhead(overhead);

		for key in 1..=count {
			promote(&mut stack, key, size);
		}

		drain(&mut stack);

		let fast_objects = stack.fast_object_count() as CacheSize;
		let slow_objects = stack.slow_object_count() as CacheSize;

		// The DRAM reservation demotes; it never evicts.
		assert_eq!(stack.len() as CacheSize, count);
		assert_eq!(fast_objects + slow_objects, count);
		assert!(slow_objects > 0, "the reservation must have demoted something");
		assert!(!stack.needs_capacity_eviction());

		// `one_access_queue` is empty (every key was promoted out of it), so
		// `slow_bytes_used` is purely the demoted main-queue tail.
		assert_eq!(stack.fast_bytes_used(), fast_objects * bytes);
		assert_eq!(stack.slow_bytes_used(), slow_objects * bytes);
		assert_eq!(stack.fast_bytes_used() + stack.slow_bytes_used(), total);

		let tagged_fast = (1..=count).filter(|key| stack.tier_of(*key) == Some(Tier::Fast)).count();
		let tagged_slow = (1..=count).filter(|key| stack.tier_of(*key) == Some(Tier::Slow)).count();

		assert_eq!(tagged_fast as CacheSize, fast_objects);
		assert_eq!(tagged_slow as CacheSize, slow_objects);
	}

	/// One key's reservation alone exceeding the whole fast budget saturates
	/// the effective budget to 0: the key demotes the moment it is promoted,
	/// and is still tracked afterwards — the DRAM budget demotes, it never
	/// evicts (terminal eviction stays governed solely by `max_size` /
	/// `one_access_capacity`).
	#[test]
	fn shared_overhead_exceeding_capacity_demotes_all_but_never_evicts() {
		let mut stack = S3FifoHybridStack::new(1.0, 1_000, 50).with_shared_overhead(100);

		stack.insert(1, 10);
		assert_eq!(drain(&mut stack), Vec::new()); // admission never touches the fast tier
		assert_eq!(stack.reserved_overhead(), 100);

		stack.update(1); // promotes to Fast, then settles against an effective budget of 0
		let migrations = drain(&mut stack);

		assert_eq!(migrations, vec![(1, Tier::Slow)]);
		assert_eq!(stack.tier_of(1), Some(Tier::Slow));
		assert_eq!(stack.fast_bytes_used(), 0);
		assert_eq!(stack.slow_bytes_used(), 10);

		// Demotion was the only response.
		assert_eq!(stack.len(), 1);
		assert_eq!(stack.fast_object_count(), 0);
		assert_eq!(stack.slow_object_count(), 1);
		assert!(!stack.needs_capacity_eviction());
	}
}
