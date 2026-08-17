/*
 * Copyright (c) Kia Shakiba
 *
 * This source code is licensed under the GNU AGPLv3 license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! `TwoQFastAdmissionReprieveHybridStack` —
//! `TwoQFastAdmissionHybridStack` with one behavioral change, for
//! `PaperPolicy::TwoQFastAdmissionReprieveHybrid`: **a one-access key that
//! ages out of the FIFO queue without a second access is reprieved into the
//! slow tier instead of being evicted outright.**
//!
//! Everything else is identical to `TwoQFastAdmissionHybridStack` — fast-tier
//! admission, the FIFO reservation carved out of `fast_capacity`, the main
//! queue's LRU fast/slow segmentation, no ghost queue. See that stack's
//! module doc for all of it.
//!
//! ## Where a reprieved key lands, and why it is the *bottom*
//!
//! `settle_fifo_queue` splices the aged-out key onto the **back of
//! `main_stack`** — the absolute LRU tail, i.e. the next terminal-eviction
//! candidate — tagged `Tier::Slow`.
//!
//! This is deliberately weaker than the equivalent in
//! `s3_fifo_lazy_demotion_fast_admission_reprieve_hybrid_stack`, which splices
//! to the *front* of its slow segment (`main_slow.push_front`), giving a
//! reprieved key a full traversal of the slow tier before it can be evicted.
//! Two reasons for the difference:
//!
//! 1. **Rank.** The main queue here is LRU-ordered, and its slow segment holds
//!    keys that were promoted to fast at least once and later demoted. A key
//!    aging out of the one-access queue has demonstrated *no* reuse at all, so
//!    ranking it above proven-but-cold objects would invert the ordering the
//!    main queue exists to maintain. The bottom is where an unproven object
//!    belongs.
//! 2. **Cost.** `push_back` is O(1) on the existing single `main_stack` list.
//!    The s3-fifo variant needed to insert at the fast/slow *boundary*, which
//!    `HashList`/`PmemHashList` cannot do (only `push_front`/`push_back`/
//!    `move_front`/`move_back` are exposed) — its first implementation walked
//!    every fast key per reprieve and burned ~18 minutes of worker CPU on a
//!    real trace without completing a run, which is what forced that stack's
//!    two-physical-lists restructure. Landing at the back needs no such
//!    restructure and cannot corrupt `main_boundary`.
//!
//! The tradeoff to be aware of when reading results: a reprieved key sitting
//! at the LRU tail may be evicted very soon under steady capacity pressure,
//! having cost a real DRAM→PMEM copy on the way there. Whether that copy buys
//! enough extra hits to pay for itself is exactly the question this variant
//! exists to answer — a null result here would be a real finding, not a bug.
//! If it *is* null, the front-of-slow placement above is the natural next
//! thing to try, and would need this stack's `main_stack` split in two the way
//! the s3-fifo variant's was.
//!
//! ## The boundary invariant survives a back-splice
//!
//! `main_boundary` marks the LRU-most `Tier::Fast` key, and the design relies
//! on fast keys forming a contiguous prefix from the list's head. Pushing a
//! `Tier::Slow` key onto the absolute back preserves that: everything fast is
//! still in front of everything slow, and `main_boundary` still points at the
//! same key it did before. No boundary update is needed, which is the other
//! reason this placement is O(1).
//!
//! ## The reprieve must NOT run through `evict_one()`
//!
//! `settle_fifo_queue` is called **synchronously from `insert()`/`resize()`**,
//! mirroring `settle_fast_tier`'s relationship to the fast/slow boundary — not
//! surfaced via `needs_capacity_eviction()`/`evict_one()` the way
//! `TwoQFastAdmissionHybridStack` surfaces the same pressure.
//!
//! That difference is load-bearing, and both halves of it are lessons this
//! crate already learned the hard way:
//!
//! * `PolicyWorker::apply_evictions` unconditionally *erases* whatever key
//!   `evict_one()` returns from the entire cache, and if it returns `None`,
//!   `erase()` falls back to evicting a **random** object. A reprieve is
//!   neither of those — nothing should leave the cache just because the FIFO
//!   queue needed relief, and `over_max_size` may not even be true at that
//!   moment. (`s3_fifo_lazy_demotion_fast_admission_reprieve`'s first draft
//!   routed a reprieve through `evict_one()` and hit exactly this.)
//! * The converse rule still holds: a `PolicyStack` may never *remove* a key
//!   on its own, because it cannot touch the object map or `AtomicStatus` —
//!   `TwoQHybridStack`'s first draft did, and permanently desynced the stack
//!   from the real object map (`has()` kept returning `true` for keys the
//!   stack had "forgotten"). That rule is not violated here precisely
//!   *because* a reprieve removes nothing: the key stays in `entries` and in
//!   the object map, and only moves between two of this stack's own lists.
//!
//! So `needs_capacity_eviction()` returns to the trait's default `false`, and
//! `evict_one()` becomes purely about the main queue — with one last-resort
//! FIFO fallback (see its doc) that exists only to avoid handing
//! `apply_evictions` a `None` it would answer with a random eviction.

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
// features already use.
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
	Fifo,
	Main,
}

/// Combined per-key bookkeeping: which queue, which tier (only meaningful
/// while `queue == Main` — a `Fifo` key is always physically Fast in this
/// design, so it needs no stored tier), and the object's size.
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

pub struct TwoQFastAdmissionReprieveHybridStack {
	fifo_queue: QueueList,
	main_stack: QueueList,

	entries: EntryMap,

	k_in: f64,

	/// The FIFO queue's own byte budget. Unlike `TwoQHybridStack`, this is a
	/// reservation carved out of `fast_capacity` (both are DRAM now) — see
	/// `effective_main_fast_capacity`.
	fifo_capacity: CacheSize,
	fifo_used: CacheSize,

	/// Total fast-tier (DRAM) budget, covering BOTH the FIFO queue and the
	/// main queue's fast segment.
	fast_capacity: CacheSize,

	/// Bytes held by `main_stack` keys tagged `Tier::Fast`. Does NOT include
	/// `fifo_used`, even though both are physically DRAM — see
	/// `fast_bytes_used`, which sums them for reporting.
	fast_used: CacheSize,
	slow_used: CacheSize,

	/// Number of keys currently tagged `Tier::Fast` within `main_stack`.
	fast_count: usize,

	/// Number of keys currently in the `Main` queue (Fast or Slow).
	main_count: usize,

	/// The least-recently-used key currently tagged `Tier::Fast` within
	/// `main_stack` — i.e. the next demotion candidate. `None` iff no key in
	/// `main_stack` is currently Fast.
	main_boundary: Option<HashedKey>,

	/// (key, new tier) pairs recorded since the last `drain_tier_migrations`.
	migrations: Vec<(HashedKey, Tier)>,
}

impl TwoQFastAdmissionReprieveHybridStack {
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

		TwoQFastAdmissionReprieveHybridStack {
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

	/// How much of `fast_capacity` the main queue's fast segment may use,
	/// after the FIFO queue's reservation is carved out.
	///
	/// Saturating rather than panicking on `fifo_capacity > fast_capacity`:
	/// that is a legitimate (if degenerate) configuration — see the module
	/// doc — and it means "the main queue gets no fast segment", not an
	/// error.
	fn effective_main_fast_capacity(&self) -> CacheSize {
		self.fast_capacity.saturating_sub(self.fifo_capacity)
	}

	/// Returns which tier the given (currently tracked) key is in, or `None`
	/// if the key isn't tracked. Exposed for tests/diagnostics.
	///
	/// The `Fifo` arm is the one line that differs from `TwoQHybridStack`'s
	/// equivalent: a one-access key is physically Fast here, not Slow.
	pub fn tier_of(&self, key: HashedKey) -> Option<Tier> {
		let entry = self.entries.get(&key)?;

		match entry.queue {
			Queue::Fifo => Some(Tier::Fast),
			Queue::Main => entry.tier,
		}
	}

	/// Records a size change for an already-tracked key without altering its
	/// queue/tier, adjusting whichever counter currently applies.
	///
	/// Callers must re-settle the fast tier afterwards when the key is
	/// `Fifo`-resident and grew: `fifo_used` is a DRAM reservation here, so
	/// growing it shrinks the main queue's effective budget. Both current
	/// callers do (`insert` explicitly, `update` via `touch`).
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
	/// tagging it `Tier::Fast`.
	///
	/// Emits **no** `(key, Tier::Fast)` migration, unlike
	/// `TwoQHybridStack::promote_from_fifo`: the key's bytes were already
	/// physically Fast (admission built them that way), so this is a
	/// bookkeeping move between two DRAM-resident structures, not a data
	/// move. See the module doc's "A migration this design no longer needs".
	///
	/// It can still *cause* migrations: the bytes move out of the FIFO
	/// reservation and into the main-fast budget, so `settle_fast_tier` below
	/// may have to demote other keys to make room — including, at a
	/// tight-enough budget, this very key straight back out again, which that
	/// call records correctly as a genuine `(key, Tier::Slow)`.
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
	}

	/// Moves an already-`Main`-tracked key to the front of `main_stack`,
	/// promoting it to `Tier::Fast` if it was `Slow`, then settles the fast
	/// tier. Unchanged from `TwoQHybridStack`: a slow→fast move here IS a
	/// real data movement (PMEM→DRAM), so it still emits a migration.
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

		// Pushed *after* `settle_fast_tier` (which pushes any demotions this
		// promotion itself triggered), not before: `apply_tier_migrations`
		// applies demotions before promotions, and within each phase in push
		// order, so pushing the promotion first would risk its DRAM
		// allocation landing before the corresponding demotion's DRAM free.
		// Guarded on the key still being `Fast`: a tight budget can demote it
		// straight back out within the same `settle_fast_tier` call, in which
		// case that call already pushed the correct final `(key, Tier::Slow)`.
		if promoted && self.entries.get(&key).and_then(|entry| entry.tier) == Some(Tier::Fast) {
			self.migrations.push((key, Tier::Fast));
		}
	}

	/// Demotes the least-recently-used fast key(s) within `main_stack` until
	/// `fast_used` fits back within [`Self::effective_main_fast_capacity`].
	///
	/// The capacity this checks against is the one substantive difference
	/// from `TwoQHybridStack::settle_fast_tier`, which checks raw
	/// `fast_capacity` — correct there, where the FIFO queue is PMEM and
	/// competes for nothing.
	fn settle_fast_tier(&mut self) {
		let effective = self.effective_main_fast_capacity();

		while self.fast_used > effective {
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

	/// Relieves FIFO-queue pressure by **reprieving** its tail into the slow
	/// tier rather than evicting it: the aged-out key is spliced onto the
	/// back of `main_stack` (the LRU tail) tagged `Tier::Slow`, and stays in
	/// the cache.
	///
	/// This is the one behavioral difference from
	/// `TwoQFastAdmissionHybridStack`, where the same pressure is reported
	/// via `needs_capacity_eviction()` and drained by `apply_evictions`
	/// removing the key outright. See the module doc for why a reprieve must
	/// run synchronously here instead, and why the back (rather than the
	/// front of the slow segment) is the right landing spot.
	///
	/// Safe to do inside the stack precisely because nothing is removed: the
	/// key stays in `entries` and in the shared object map, moving only
	/// between two of this stack's own lists. The `(key, Tier::Slow)`
	/// migration it records is a real DRAM->PMEM copy that `PolicyWorker`
	/// applies, exactly as for an ordinary demotion.
	fn settle_fifo_queue(&mut self) {
		while self.fifo_used > self.fifo_capacity {
			let Some(key) = self.fifo_queue.pop_back() else { break };
			let Some(entry) = self.entries.get(&key).copied() else { continue };
			let size = entry.size as CacheSize;

			self.fifo_used = self.fifo_used.saturating_sub(size);

			// Back of the list: still behind every fast key, so the
			// "fast keys are a contiguous prefix" invariant `main_boundary`
			// depends on is preserved and the boundary needs no update.
			self.main_stack.push_back(key);

			if let Some(stored) = self.entries.get_mut(&key) {
				stored.queue = Queue::Main;
				stored.tier = Some(Tier::Slow);
			}

			self.slow_used += size;
			self.main_count += 1;

			self.migrations.push((key, Tier::Slow));
		}
	}

	/// Pops and fully removes `fifo_queue`'s tail from this stack's own
	/// bookkeeping. Unlike `TwoQFastAdmissionHybridStack`, this is **not**
	/// the normal path for an aged-out one-access key -- that is
	/// `settle_fifo_queue` above, which reprieves rather than evicts. This
	/// exists only as `evict_one`'s last resort when the main queue is
	/// entirely empty; see there.
	fn evict_fifo_tail(&mut self) -> Option<HashedKey> {
		let key = self.fifo_queue.pop_back()?;
		let size = self.entries.remove(&key).map(|entry| entry.size).unwrap_or(0) as CacheSize;

		self.fifo_used = self.fifo_used.saturating_sub(size);

		Some(key)
	}
}

impl PolicyStack for TwoQFastAdmissionReprieveHybridStack {
	fn is_policy(&self, policy: &PaperPolicy) -> bool {
		matches!(policy, PaperPolicy::TwoQFastAdmissionReprieveHybrid(k_in) if *k_in == self.k_in)
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
			// `touch` settles the fast tier on every path, so a FIFO-resident
			// key that grew is covered without an extra call here.
			self.resize_key(key, size);
			self.touch(key);
			return;
		}

		// Brand-new key: admitted into the FIFO queue, which is FAST here.
		// If this pushes fifo_used over fifo_capacity, `settle_fifo_queue`
		// below reprieves the queue's tail into the slow tier -- nothing is
		// evicted.
		self.fifo_queue.push_front(key);
		self.entries.insert(key, TwoQEntry { queue: Queue::Fifo, tier: None, size });
		self.fifo_used += size as CacheSize;

		// Relieve FIFO pressure immediately, by reprieving the tail into the
		// slow tier rather than reporting the pressure for `apply_evictions`
		// to remove -- the defining difference from
		// `TwoQFastAdmissionHybridStack`. See `settle_fifo_queue`.
		self.settle_fifo_queue();

		// Deliberately does NOT re-settle the *fast* tier. The reservation
		// carved out of `fast_capacity` is the fixed `fifo_capacity`, not the
		// live `fifo_used`, so the main queue's effective budget doesn't move
		// when the FIFO queue fills -- only `resize`/`resize_fast_tier` can
		// change it. (A reprieve moves bytes from `fifo_used` to `slow_used`,
		// leaving `fast_used` untouched, so it cannot create fast-tier
		// pressure either.)
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

		// Re-settle, which `TwoQHybridStack::resize` has no need to do:
		// `fifo_capacity` is carved out of `fast_capacity` here, so growing
		// the cache grows the FIFO reservation and shrinks the main queue's
		// effective fast budget. Catch that now rather than at whatever
		// unrelated `insert`/`update` happens to come next.
		self.settle_fast_tier();

		// A shrink also lowers `fifo_capacity`, which may leave the FIFO
		// queue over budget; reprieve its tail down to fit rather than
		// reporting eviction pressure.
		self.settle_fifo_queue();
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

	/// Terminal eviction takes the main queue's LRU tail -- which, thanks to
	/// `settle_fifo_queue`, is where reprieved one-access keys land, so an
	/// object that never demonstrated reuse is still the first thing to go.
	///
	/// The FIFO queue is only touched as a last resort, when the main queue
	/// is completely empty. That fallback is not about eviction policy: it
	/// exists so this method never returns `None` while the stack still holds
	/// keys, because `apply_evictions` answers a `None` by evicting a
	/// **random** object (see `erase`'s doc). Reaching it requires every
	/// tracked key to still be in the one-access queue while overall
	/// `max_size` is already exceeded.
	fn evict_one(&mut self) -> Option<HashedKey> {
		if self.main_stack.len() == 0 {
			return self.evict_fifo_tail();
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

	/// Both DRAM-resident structures, summed: the FIFO queue plus the main
	/// queue's fast segment. The mirror image of `TwoQHybridStack`, where
	/// `fifo_used` counts toward the *slow* total instead.
	fn fast_bytes_used(&self) -> CacheSize {
		self.fifo_used + self.fast_used
	}

	fn slow_bytes_used(&self) -> CacheSize {
		self.slow_used
	}

	fn fast_object_count(&self) -> usize {
		self.fifo_queue.len() + self.fast_count
	}

	fn slow_object_count(&self) -> usize {
		self.main_count - self.fast_count
	}
}


#[cfg(test)]
mod tests {
	use super::*;

	fn drain(stack: &mut TwoQFastAdmissionReprieveHybridStack) -> Vec<(HashedKey, Tier)> {
		stack.drain_tier_migrations()
	}

	/// Unchanged from `TwoQFastAdmissionHybridStack`: admission is still a
	/// plain DRAM write into the one-access queue.
	#[test]
	fn admission_still_lands_in_the_fifo_queue_in_the_fast_tier() {
		let mut stack = TwoQFastAdmissionReprieveHybridStack::new(0.5, 1_000, 1_000);

		stack.insert(1, 10);

		assert_eq!(stack.tier_of(1), Some(Tier::Fast));
		assert!(drain(&mut stack).is_empty());
	}

	/// The defining behavior: an aged-out one-access key survives, in slow.
	#[test]
	fn an_aged_out_one_access_key_is_reprieved_not_evicted() {
		// fifo_capacity = 0.1 * 1_000 = 100, so a third 50-byte key pushes
		// the queue over and the tail must be reprieved.
		let mut stack = TwoQFastAdmissionReprieveHybridStack::new(0.1, 1_000, 1_000);

		stack.insert(1, 50);
		stack.insert(2, 50);
		assert!(drain(&mut stack).is_empty());

		stack.insert(3, 50);

		// Key 1 was the FIFO tail. It is still tracked -- not removed --
		// and now lives in the slow tier.
		assert_eq!(stack.len(), 3);
		assert_eq!(stack.tier_of(1), Some(Tier::Slow));
		assert_eq!(stack.tier_of(2), Some(Tier::Fast));
		assert_eq!(stack.tier_of(3), Some(Tier::Fast));

		// And it produced a real DRAM->PMEM migration.
		assert_eq!(drain(&mut stack), vec![(1, Tier::Slow)]);
	}

	#[test]
	fn a_reprieve_moves_bytes_from_the_fast_gauges_to_the_slow_gauges() {
		let mut stack = TwoQFastAdmissionReprieveHybridStack::new(0.1, 1_000, 1_000);

		stack.insert(1, 50);
		stack.insert(2, 50);

		assert_eq!(stack.fast_bytes_used(), 100);
		assert_eq!(stack.slow_bytes_used(), 0);

		stack.insert(3, 50);

		assert_eq!(stack.fast_bytes_used(), 100);
		assert_eq!(stack.slow_bytes_used(), 50);
		assert_eq!(stack.fast_object_count(), 2);
		assert_eq!(stack.slow_object_count(), 1);
	}

	/// A reprieved key lands at the LRU tail, so it is the next thing
	/// evicted -- the placement decision this variant makes.
	#[test]
	fn a_reprieved_key_lands_at_the_bottom_and_is_evicted_first() {
		let mut stack = TwoQFastAdmissionReprieveHybridStack::new(0.1, 1_000, 1_000);

		// Key 1 proves itself and moves into the main queue as Fast.
		stack.insert(1, 50);
		stack.update(1);

		// Keys 2-4 stay unproven. fifo_capacity is 100, and the trigger is
		// strictly `fifo_used > fifo_capacity`, so three 50-byte keys are
		// needed to push it over -- key 1 left the queue when it was
		// promoted, so it no longer counts toward `fifo_used`.
		stack.insert(2, 50);
		stack.insert(3, 50);
		stack.insert(4, 50);

		drain(&mut stack);

		assert_eq!(stack.tier_of(2), Some(Tier::Slow));

		// The reprieved key goes before the proven one, despite being
		// added to the main queue later.
		assert_eq!(stack.evict_one(), Some(2));
		assert_eq!(stack.evict_one(), Some(1));
	}

	/// Pushing a Slow key onto the back must not disturb `main_boundary`,
	/// which the fast/slow prefix invariant depends on.
	#[test]
	fn a_reprieve_does_not_corrupt_the_fast_slow_boundary() {
		let mut stack = TwoQFastAdmissionReprieveHybridStack::new(0.1, 1_000, 1_000);

		// Two proven keys in the main queue, both Fast.
		stack.insert(1, 50);
		stack.update(1);
		stack.insert(2, 50);
		stack.update(2);
		drain(&mut stack);

		assert_eq!(stack.main_boundary, Some(1));

		// A reprieve appends a Slow key behind them.
		stack.insert(3, 50);
		stack.insert(4, 50);
		stack.insert(5, 50);
		drain(&mut stack);

		// Boundary is untouched, and both proven keys are still Fast.
		assert_eq!(stack.main_boundary, Some(1));
		assert_eq!(stack.tier_of(1), Some(Tier::Fast));
		assert_eq!(stack.tier_of(2), Some(Tier::Fast));

		// Fast-tier demotion still works correctly afterwards.
		stack.resize_fast_tier(50 + stack.fifo_capacity);
		let migrations = drain(&mut stack);

		assert_eq!(migrations, vec![(1, Tier::Slow)]);
		assert_eq!(stack.main_boundary, Some(2));
	}

	/// A reprieved key is a full main-queue citizen: re-accessing it
	/// promotes it back to fast like any other slow key.
	#[test]
	fn a_reprieved_key_can_still_be_promoted_by_a_later_access() {
		let mut stack = TwoQFastAdmissionReprieveHybridStack::new(0.1, 1_000, 1_000);

		stack.insert(1, 50);
		stack.insert(2, 50);
		stack.insert(3, 50);
		drain(&mut stack);

		assert_eq!(stack.tier_of(1), Some(Tier::Slow));

		stack.update(1);

		assert_eq!(stack.tier_of(1), Some(Tier::Fast));
		assert_eq!(drain(&mut stack), vec![(1, Tier::Fast)]);
	}

	/// The reprieve runs inside the stack, so the stack must never report
	/// eviction pressure -- see the module doc for why routing it through
	/// `evict_one()` was a real bug in the s3-fifo equivalent.
	#[test]
	fn capacity_pressure_is_never_reported_for_eviction() {
		let mut stack = TwoQFastAdmissionReprieveHybridStack::new(0.1, 1_000, 1_000);

		for key in 1..=10 {
			stack.insert(key, 50);
			assert!(
				!stack.needs_capacity_eviction(),
				"FIFO pressure must be relieved by reprieve, never reported as eviction",
			);
		}

		// Nothing was lost along the way.
		assert_eq!(stack.len(), 10);
	}

	/// Shrinking `max_size` shrinks `fifo_capacity`, which must reprieve
	/// rather than leave the queue over budget.
	#[test]
	fn shrinking_max_size_reprieves_down_to_the_new_budget() {
		let mut stack = TwoQFastAdmissionReprieveHybridStack::new(0.5, 1_000, 1_000);

		// fifo_capacity 500: four 100-byte keys fit comfortably.
		for key in 1..=4 {
			stack.insert(key, 100);
		}

		assert!(drain(&mut stack).is_empty());
		assert_eq!(stack.fifo_used, 400);

		// max_size 400 => fifo_capacity 200, so two keys must be reprieved.
		stack.resize(400);

		assert_eq!(stack.fifo_used, 200);
		assert_eq!(drain(&mut stack), vec![(1, Tier::Slow), (2, Tier::Slow)]);
		assert_eq!(stack.len(), 4);
	}

	/// The last-resort fallback: with nothing in the main queue, `evict_one`
	/// must still return a key rather than `None`, which `apply_evictions`
	/// would answer with a random eviction.
	#[test]
	fn eviction_falls_back_to_the_fifo_queue_when_the_main_queue_is_empty() {
		let mut stack = TwoQFastAdmissionReprieveHybridStack::new(1.0, 1_000, 1_000);

		// k_in 1.0 keeps the FIFO queue within budget, so nothing is ever
		// reprieved and the main queue stays empty.
		stack.insert(1, 50);
		stack.insert(2, 50);

		assert_eq!(stack.slow_object_count(), 0);

		assert_eq!(stack.evict_one(), Some(1));
		assert_eq!(stack.evict_one(), Some(2));
		assert_eq!(stack.evict_one(), None);
		assert_eq!(stack.len(), 0);
	}

	#[test]
	fn removing_a_reprieved_key_releases_its_slow_bytes() {
		let mut stack = TwoQFastAdmissionReprieveHybridStack::new(0.1, 1_000, 1_000);

		stack.insert(1, 50);
		stack.insert(2, 50);
		stack.insert(3, 50);
		drain(&mut stack);

		assert_eq!(stack.slow_bytes_used(), 50);

		stack.remove(1);

		assert_eq!(stack.slow_bytes_used(), 0);
		assert_eq!(stack.slow_object_count(), 0);
		assert_eq!(stack.len(), 2);
	}

	#[test]
	fn clear_resets_every_counter() {
		let mut stack = TwoQFastAdmissionReprieveHybridStack::new(0.1, 1_000, 1_000);

		for key in 1..=5 {
			stack.insert(key, 50);
		}

		stack.clear();

		assert_eq!(stack.len(), 0);
		assert_eq!(stack.fast_bytes_used(), 0);
		assert_eq!(stack.slow_bytes_used(), 0);
		assert_eq!(stack.fast_object_count(), 0);
		assert_eq!(stack.slow_object_count(), 0);
		assert!(drain(&mut stack).is_empty());
	}

	#[test]
	fn is_policy_matches_only_its_own_variant_and_k_in() {
		let stack = TwoQFastAdmissionReprieveHybridStack::new(0.1, 1_000, 300);

		assert!(stack.is_policy(&PaperPolicy::TwoQFastAdmissionReprieveHybrid(0.1)));
		assert!(!stack.is_policy(&PaperPolicy::TwoQFastAdmissionReprieveHybrid(0.2)));
		assert!(!stack.is_policy(&PaperPolicy::TwoQFastAdmissionHybrid(0.1)));
		assert!(!stack.is_policy(&PaperPolicy::TwoQHybrid(0.1)));
	}
}
