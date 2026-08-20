/*
 * Copyright (c) Kia Shakiba
 *
 * This source code is licensed under the GNU AGPLv3 license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! `S3FifoGhostHybridStack` — `S3FifoHybridStack` plus a bare-key ghost
//! queue, for `PaperPolicy::S3FifoGhostHybrid`.
//!
//! Identical to `S3FifoHybridStack` in every other respect (see that
//! stack's module doc for the full admission/demotion/promotion/eviction
//! rules, including the "contiguous front run" invariant and the eager
//! one-access-queue-promotion vs. lazy main-queue-reference-bit asymmetry)
//! — this file only adds a ghost queue remembering keys that aged out of
//! `one_access_queue` without a second access, mirroring
//! `s_three_fifo_stack.rs`'s own `ghost: HashList<HashedKey>` shape exactly
//! (the plain, non-hybrid `SThreeFifoStack` already in this crate already
//! has a ghost queue of precisely this bare-key shape, so this brings the
//! hybrid version in line with its own plain-policy counterpart).
//!
//! ## Ghost lifecycle, matching `SThreeFifoStack`'s existing convention
//!
//! * **Added to** only by `evict_one_access_tail` (an `one_access_queue`
//!   object aging out without a second access) — never by a main-queue
//!   eviction, matching `SThreeFifoStack::evict_small` (adds to ghost) vs.
//!   `evict_main` (only trims it) exactly.
//! * **Checked** by `insert`'s brand-new-key branch, before falling back to
//!   the normal `one_access_queue` admission.
//! * **Not removed immediately on a hit** — same lazy convention
//!   `SThreeFifoStack` uses. Only trimmed lazily, capped relative to
//!   `main_count`, during a genuine main-queue eviction (the reference-bit-
//!   clear branch, never the second-chance branch) — and cleared outright
//!   by `remove`/`clear`.
//!
//! ## Where a ghost hit lands: fast tier, deliberately reversible
//!
//! Same choice, same rationale, and same easy reversal as
//! `TwoQGhostHybridStack`'s module doc describes — `admit_via_ghost_hit` is
//! structurally identical to `promote_from_one_access` minus the "remove
//! from `one_access_queue`" step. See that file's module doc for the full
//! reasoning; not repeated here.
//!
//! ## Shared-structure DRAM reservation
//!
//! `settle_fast_tier` works against `fast_capacity` minus
//! `reserved_overhead()` — the DRAM the shared object hashtable and this
//! stack's own eviction-stack bookkeeping occupy for every *tracked* key
//! (exactly one `HashList` node, in whichever of `one_access_queue`/
//! `main_queue` the key currently sits in, plus its one `entries` map slot)
//! — so the fast-tier budget bounds total DRAM rather than just fast-tier
//! values. Keys of *both* tiers are charged: those entries are DRAM
//! wherever the object's own data lives. Same builder
//! (`with_shared_overhead`), same `saturating_sub`, and same composition
//! with the watermarks as `LruHybridStack`; and because `one_access_queue`
//! is slow-tier here, this stack has a single fast-capacity segment, so the
//! whole reservation lands on it instead of being split proportionally the
//! way `LruSizedHybridStack` splits its two.
//!
//! `ghost` is charged *separately*, as its own term: its entries are bare
//! keys for objects that are no longer in the cache and no longer in
//! `entries` at all, so that cost scales with the ghost queue's own length
//! (`ghost.len()`, capped at `main_count` by `trim_ghost` on genuine
//! main-queue evictions), not with the tracked-key count — a per-tracked-key
//! constant cannot model it. That term reads the shared
//! `crate::object::overhead::GHOST_ENTRY_DRAM_OVERHEAD` constant *directly*
//! rather than taking a builder-supplied value: every ghost variant keeps the
//! same bare-key `HashList<HashedKey>`, so the cost is the same for all of
//! them and there is nothing for a caller to configure. It is therefore
//! charged unconditionally — including on a stack built without
//! `with_shared_overhead`, since ghost DRAM is occupied whether or not the
//! per-tracked-key term happens to be wired in. Only the per-tracked-key
//! term defaults to `0`, so unit tests constructing the stack directly see
//! the pure value-byte budget plus whatever ghost entries they created.
//!
//! ## `eviction_stacks_pmem`
//!
//! `ghost` follows the same DRAM/PMEM switch as `one_access_queue`/
//! `main_queue`/`entries` — see `S3FifoHybridStack`'s module doc. That switch
//! is why `GHOST_ENTRY_DRAM_OVERHEAD` is itself `cfg`-selected: when the
//! eviction stacks are in PMEM, a ghost entry costs the fast tier nothing and
//! the term drops to `0`.

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
	object::{
		ObjectSize,
		overhead::GHOST_ENTRY_DRAM_OVERHEAD,
	},
	worker::policy::policy_stack::{PolicyStack, Tier, watermarks},
};

/// Which live queue a key currently belongs to.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Queue {
	OneAccess,
	Main,
}

/// Combined per-key bookkeeping. `tier`/`accessed` are only meaningful
/// while `queue == Main` — see `S3FifoHybridStack`'s module doc for why
/// `OneAccess` never needs a reference bit at all.
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

pub struct S3FifoGhostHybridStack {
	one_access_queue: QueueList,
	main_queue: QueueList,
	ghost: QueueList,

	entries: EntryMap,

	one_access_ratio: f64,
	one_access_capacity: CacheSize,
	one_access_used: CacheSize,

	fast_capacity: CacheSize,
	fast_used: CacheSize,
	slow_used: CacheSize,

	/// Approximate per-object DRAM cost of the shared structures (object
	/// hashtable + eviction stacks) that hold an entry for every *tracked*
	/// key of both tiers. Reserved out of `fast_capacity` in
	/// `settle_fast_tier`. `0` unless set via `with_shared_overhead` (so
	/// unit tests exercising the pure value-budget behavior are unaffected).
	shared_overhead: CacheSize,

	fast_count: usize,

	/// Number of keys currently in the `Main` queue (Fast or Slow). Also
	/// used as the ghost list's size cap reference.
	main_count: usize,

	main_boundary: Option<HashedKey>,

	migrations: Vec<(HashedKey, Tier)>,
}

impl S3FifoGhostHybridStack {
	#[cfg(not(feature = "eviction_stacks_pmem"))]
	fn new_collections() -> (QueueList, QueueList, QueueList, EntryMap) {
		(HashList::default(), HashList::default(), HashList::default(), HashMap::default())
	}

	#[cfg(feature = "eviction_stacks_pmem")]
	fn new_collections() -> (QueueList, QueueList, QueueList, EntryMap) {
		(
			PmemHashList::with_hasher(NoHasher::default()),
			PmemHashList::with_hasher(NoHasher::default()),
			PmemHashList::with_hasher(NoHasher::default()),
			HashMap::with_hasher_in(NoHasher::default(), Hybrid),
		)
	}

	pub fn new(one_access_ratio: f64, max_size: CacheSize, fast_capacity: CacheSize) -> Self {
		let (one_access_queue, main_queue, ghost, entries) = Self::new_collections();

		S3FifoGhostHybridStack {
			one_access_queue,
			main_queue,
			ghost,

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
	/// hashtable + this stack's eviction-stack bookkeeping) reserved out of
	/// the fast-tier budget. See
	/// `crate::object::overhead::get_hybrid_dram_shared_overhead`.
	/// Builder-style so `init_policy_stack` can wire it in without disturbing
	/// `new`'s signature (unit tests keep the default `0`).
	pub fn with_shared_overhead(mut self, overhead: CacheSize) -> Self {
		self.shared_overhead = overhead;
		self
	}

	/// Total DRAM currently reserved out of `fast_capacity` for metadata. Two
	/// independent terms:
	///
	/// * `tracked key count × shared_overhead` (every key in `entries` — i.e.
	///   both tiers, and both `one_access_queue` and `main_queue` residents,
	///   since a key holds exactly one list node plus one `entries` slot
	///   wherever its data lives); and
	/// * `ghost length × GHOST_ENTRY_DRAM_OVERHEAD` — the bare-key list node
	///   a *ghost* key occupies. A ghost key names an object that is no
	///   longer in the cache: it has no `entries` row and no object-hashtable
	///   slot, so no per-tracked-key constant can express its cost (which is
	///   also why the shared constant is gated on `eviction_stacks_pmem`
	///   alone, never on `global_hashtable_pmem`).
	///
	/// The ghost term is read straight off the shared constant rather than
	/// configured per stack — every ghost variant keeps the same
	/// `HashList<HashedKey>`, so the cost is identical for all of them — and
	/// it is charged whether or not `shared_overhead` was ever set: the two
	/// quantities are independent, and that DRAM is occupied either way.
	///
	/// Loop-invariant within a `settle_fast_tier` pass: a demotion only
	/// retags an entry's `tier`, it never changes whether a key is tracked
	/// nor the length of `ghost`.
	fn reserved_overhead(&self) -> CacheSize {
		self.entries.len() as CacheSize * self.shared_overhead
			+ self.ghost.len() as CacheSize * (GHOST_ENTRY_DRAM_OVERHEAD as CacheSize)
	}

	pub fn tier_of(&self, key: HashedKey) -> Option<Tier> {
		let entry = self.entries.get(&key)?;

		match entry.queue {
			Queue::OneAccess => Some(Tier::Slow),
			Queue::Main => entry.tier,
		}
	}

	/// Returns `true` if `key` currently has a ghost entry. Exposed for tests.
	pub fn is_ghost(&self, key: HashedKey) -> bool {
		self.ghost.contains(&key)
	}

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

	fn touch(&mut self, key: HashedKey) {
		match self.entries.get(&key).map(|entry| entry.queue) {
			Some(Queue::OneAccess) => self.promote_from_one_access(key),
			Some(Queue::Main) => self.mark_accessed(key),
			None => {},
		}
	}

	fn mark_accessed(&mut self, key: HashedKey) {
		if let Some(entry) = self.entries.get_mut(&key) {
			entry.accessed = true;
		}
	}

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

		if self.entries.get(&key).and_then(|entry| entry.tier) == Some(Tier::Fast) {
			self.migrations.push((key, Tier::Fast));
		}
	}

	/// Admits a brand-new key directly into `main_queue` at `Tier::Fast` —
	/// the ghost-hit path. Structurally identical to
	/// `promote_from_one_access` minus the "remove from `one_access_queue`"
	/// step. See the module doc's "Where a ghost hit lands" section.
	fn admit_via_ghost_hit(&mut self, key: HashedKey, size: ObjectSize) {
		self.main_queue.push_front(key);
		self.entries.insert(key, S3FifoEntry {
			queue: Queue::Main,
			tier: Some(Tier::Fast),
			size,
			accessed: false,
		});
		self.fast_used += size as CacheSize;
		self.fast_count += 1;
		self.main_count += 1;

		if self.main_boundary.is_none() {
			self.main_boundary = Some(key);
		}

		self.settle_fast_tier();

		if self.entries.get(&key).and_then(|entry| entry.tier) == Some(Tier::Fast) {
			self.migrations.push((key, Tier::Fast));
		}
	}

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

	/// Demotes `main_queue`'s oldest fast-tier keys until the fast tier is
	/// back under the shared *low* watermark -- but only once usage has
	/// crossed the shared *high* watermark in the first place.
	///
	/// The effective ceiling this stack works against is `fast_capacity`
	/// minus `reserved_overhead()` -- the DRAM the object hashtable and this
	/// stack's own bookkeeping occupy for every tracked key, plus the bare-key
	/// list node every *ghost* key occupies (always charged, at the shared
	/// `GHOST_ENTRY_DRAM_OVERHEAD`), saturating to 0 when that metadata alone
	/// meets or exceeds `fast_capacity`. That reservation is the only thing
	/// carved out of the budget: unlike the `fast_admission` variants,
	/// `one_access_queue` here is slow-tier (it is counted in
	/// `slow_bytes_used` and bounded separately by `one_access_capacity` via
	/// `needs_capacity_eviction`), so
	/// this stack has exactly one fast-capacity segment and the whole
	/// reservation lands on it rather than being split proportionally the way
	/// `LruSizedHybridStack` splits its two.
	///
	/// The `watermarks` helpers are applied *on top of* that effective value
	/// -- the reservation shapes the ceiling, the watermarks only shape when
	/// a pass fires and how far it drains.
	///
	/// Previously this drained to exactly `fast_capacity`, which pinned the
	/// tier at 100% utilisation and made essentially every admission demote
	/// exactly one object (see the `watermarks` module doc). Setting both
	/// `FAST_TIER_HIGH_WATERMARK` and `FAST_TIER_LOW_WATERMARK` to `1.0`
	/// restores that behaviour byte-for-byte.
	///
	/// Per-demotion bookkeeping is deliberately untouched: each demoted
	/// object still retags its entry, still moves `fast_used`/`fast_count`/
	/// `slow_used` by its own size, still walks `main_boundary` one step
	/// toward the front, and still emits exactly one `Tier::Slow` migration.
	fn settle_fast_tier(&mut self) {
		// Capacity minus the shared per-object metadata reservation. The
		// watermarks are applied *on top of* this value, never in place of it.
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

	/// Pops `one_access_queue`'s tail, removes it from this stack's own
	/// bookkeeping, and remembers it in `ghost`. Only called from
	/// `evict_one`.
	fn evict_one_access_tail(&mut self) -> Option<HashedKey> {
		let key = self.one_access_queue.pop_back()?;
		let size = self.entries.remove(&key).map(|entry| entry.size).unwrap_or(0) as CacheSize;

		self.one_access_used = self.one_access_used.saturating_sub(size);
		self.ghost.push_front(key);

		Some(key)
	}

	/// Trims `ghost` down to `main_count` entries — called only from a
	/// genuine main-queue eviction (the reference-bit-clear branch), never
	/// from a second chance or from `evict_one_access_tail` (which is what
	/// populates `ghost`). Mirrors `SThreeFifoStack::evict_main`'s cap.
	fn trim_ghost(&mut self) {
		while self.ghost.len() > self.main_count {
			self.ghost.pop_back();
		}
	}
}

impl PolicyStack for S3FifoGhostHybridStack {
	fn is_policy(&self, policy: &PaperPolicy) -> bool {
		matches!(policy, PaperPolicy::S3FifoGhostHybrid(ratio) if *ratio == self.one_access_ratio)
	}

	fn len(&self) -> usize {
		self.entries.len()
	}

	fn contains(&self, key: HashedKey) -> bool {
		self.entries.contains_key(&key)
	}

	fn insert(&mut self, key: HashedKey, size: ObjectSize) {
		if self.entries.contains_key(&key) {
			self.resize_key(key, size);
			self.touch(key);
			return;
		}

		if self.ghost.contains(&key) {
			self.admit_via_ghost_hit(key, size);
			return;
		}

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
		// Unconditional and first: a key evicted from `one_access_queue`
		// (via `evict_one_access_tail`) has *already* been removed from
		// `entries` by the time it lives only in `ghost` -- gating this on
		// `entries.remove` succeeding (as the rest of this method's logic
		// legitimately does) would silently skip clearing a stale ghost
		// entry for exactly that case. Mirrors `SThreeFifoStack::remove`,
		// which also clears its ghost queue unconditionally.
		self.ghost.remove(&key);

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
	}

	fn clear(&mut self) {
		self.one_access_queue.clear();
		self.main_queue.clear();
		self.ghost.clear();
		self.entries.clear();

		self.one_access_used = 0;
		self.fast_used = 0;
		self.slow_used = 0;
		self.fast_count = 0;
		self.main_count = 0;
		self.main_boundary = None;
		self.migrations.clear();
	}

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

			self.trim_ghost();

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

	fn drain(stack: &mut S3FifoGhostHybridStack) -> Vec<(HashedKey, Tier)> {
		stack.drain_tier_migrations()
	}

	/// `insert` + `update` -- the insert-into-`one_access_queue`-then-promote
	/// pairing every fast-tier test in this module already uses, since this
	/// stack never admits a fresh (non-ghost) key straight into `main_queue`'s
	/// fast tier.
	fn promote(stack: &mut S3FifoGhostHybridStack, key: HashedKey, size: ObjectSize) {
		stack.insert(key, size);
		stack.update(key);
	}

	/// Smallest fast-tier capacity whose *low* watermark still leaves room for
	/// `bytes`. Lets the fast-tier tests state their expectations in whole
	/// objects instead of hard-coded byte thresholds, so they hold at whatever
	/// `FAST_TIER_HIGH_WATERMARK`/`FAST_TIER_LOW_WATERMARK` pair is configured
	/// rather than only at the 0.95/0.75 defaults. The `while` loop absorbs the
	/// truncation in `watermarks::low_bytes`' `as u64` cast, which a bare
	/// `ceil()` on its own can land a byte short of.
	fn capacity_holding(bytes: CacheSize) -> CacheSize {
		let mut capacity = (bytes as f64 / watermarks::low()).ceil() as CacheSize;

		while watermarks::low_bytes(capacity) < bytes {
			capacity += 1;
		}

		capacity
	}

	#[test]
	fn admission_always_lands_in_one_access_queue_slow() {
		let mut stack = S3FifoGhostHybridStack::new(1.0, 1_000, 1_000);

		stack.insert(1, 10);
		stack.insert(2, 10);

		assert_eq!(stack.tier_of(1), Some(Tier::Slow));
		assert_eq!(stack.tier_of(2), Some(Tier::Slow));
		assert_eq!(drain(&mut stack), Vec::new());
	}

	#[test]
	fn reaccessing_a_one_access_key_promotes_it_eagerly_to_fast() {
		let mut stack = S3FifoGhostHybridStack::new(1.0, 1_000, 1_000);

		stack.insert(1, 10);
		drain(&mut stack);

		stack.update(1);
		let migrations = drain(&mut stack);

		assert_eq!(migrations, vec![(1, Tier::Fast)]);
		assert_eq!(stack.tier_of(1), Some(Tier::Fast));
	}

	#[test]
	fn a_key_aging_out_without_reaccess_becomes_a_ghost_entry() {
		let mut stack = S3FifoGhostHybridStack::new(1.0, 1_000, 1_000);

		stack.insert(1, 10);
		drain(&mut stack);

		assert_eq!(stack.evict_one(), Some(1));
		assert!(stack.is_ghost(1));
	}

	#[test]
	fn ghost_hit_on_readmission_lands_directly_in_fast_tier() {
		let mut stack = S3FifoGhostHybridStack::new(1.0, 1_000, 1_000);

		stack.insert(1, 10);
		drain(&mut stack);
		stack.evict_one();
		assert!(stack.is_ghost(1));

		stack.insert(1, 10);
		let migrations = drain(&mut stack);

		assert_eq!(migrations, vec![(1, Tier::Fast)]);
		assert_eq!(stack.tier_of(1), Some(Tier::Fast));
	}

	#[test]
	fn fresh_key_with_no_ghost_history_still_lands_in_one_access_queue_slow() {
		let mut stack = S3FifoGhostHybridStack::new(1.0, 1_000, 1_000);

		stack.insert(5, 10);
		let migrations = drain(&mut stack);

		assert_eq!(migrations, Vec::new());
		assert_eq!(stack.tier_of(5), Some(Tier::Slow));
	}

	#[test]
	fn a_mere_access_never_reorders_the_main_queue_or_migrates() {
		let mut stack = S3FifoGhostHybridStack::new(1.0, 1_000, 1_000);

		stack.insert(1, 10);
		stack.update(1);
		drain(&mut stack);

		stack.update(1);
		let migrations = drain(&mut stack);

		assert_eq!(migrations, Vec::new());
		assert_eq!(stack.tier_of(1), Some(Tier::Fast));
	}

	#[test]
	fn evict_one_gives_an_accessed_slow_key_a_second_chance() {
		// Sized so a triggered pass drains to a low watermark that still holds
		// one of these two 10-byte objects -- i.e. exactly one demotion, the
		// oldest fast key. (Was a hard-coded 10: correct back when a pass
		// drained to the ceiling, but under the watermarks a 10-byte ceiling
		// triggers at 9 and drains to 7, so even a lone resident object would
		// be demoted straight back out.)
		let mut stack = S3FifoGhostHybridStack::new(1.0, 1_000, capacity_holding(10));

		stack.insert(1, 10);
		stack.update(1);
		drain(&mut stack);

		stack.insert(2, 10);
		stack.update(2);
		drain(&mut stack);
		assert_eq!(stack.tier_of(1), Some(Tier::Slow));

		stack.update(1);
		assert_eq!(drain(&mut stack), Vec::new());

		let evicted = stack.evict_one();

		assert_eq!(evicted, Some(2));
		assert_eq!(stack.tier_of(1), Some(Tier::Fast));
		assert_eq!(stack.contains(2), false);
		// Main-queue evictions never populate the ghost list -- only
		// one_access_queue evictions do (mirrors SThreeFifoStack::evict_main,
		// which only trims, vs. evict_small, which adds).
		assert!(!stack.is_ghost(1));
		assert!(!stack.is_ghost(2));
	}

	#[test]
	fn remove_clears_ghost_entry_too() {
		let mut stack = S3FifoGhostHybridStack::new(1.0, 1_000, 1_000);

		stack.insert(1, 10);
		drain(&mut stack);
		stack.evict_one();
		assert!(stack.is_ghost(1));

		stack.remove(1);
		assert!(!stack.is_ghost(1));
	}

	#[test]
	fn remove_and_clear_reset_bookkeeping() {
		let mut stack = S3FifoGhostHybridStack::new(1.0, 1_000, 1_000);

		stack.insert(1, 10);
		stack.insert(2, 10);
		stack.update(1);
		drain(&mut stack);

		stack.remove(1);
		assert_eq!(stack.contains(1), false);

		stack.remove(2);
		assert_eq!(stack.contains(2), false);

		stack.insert(3, 10);
		stack.clear();

		assert_eq!(stack.len(), 0);
		assert_eq!(stack.tier_of(3), None);
		assert_eq!(stack.evict_one(), None);
	}

	/// (a) The trigger is a strict `>`, so usage sitting right *on* the high
	/// watermark -- the largest usage that is not over it -- must leave the
	/// tier completely alone.
	#[test]
	fn fast_usage_at_the_high_watermark_triggers_no_demotion() {
		let fast_capacity: CacheSize = 1_000;
		let high = watermarks::high_bytes(fast_capacity);

		let mut stack = S3FifoGhostHybridStack::new(1.0, 100_000, fast_capacity);

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

		let mut stack = S3FifoGhostHybridStack::new(1.0, 100_000, fast_capacity);

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

		let mut stack = S3FifoGhostHybridStack::new(1.0, 100_000, fast_capacity);

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

		let mut stack = S3FifoGhostHybridStack::new(1.0, 100_000, fast_capacity);

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

	// ---------------------------------------------------------------------
	// Shared-structure DRAM reservation: the per-tracked-key term
	// (`with_shared_overhead`) and the per-ghost-entry term (the shared
	// `GHOST_ENTRY_DRAM_OVERHEAD` constant, charged unconditionally).
	//
	// Every other test in this module constructs the stack without the
	// builder, so its per-tracked-key term is `0`; the ghost term is charged
	// regardless, but only three of those tests ever populate `ghost` at all
	// (`a_key_aging_out_...`, `ghost_hit_on_readmission_...`,
	// `remove_clears_ghost_entry_too`), and each of them runs against a
	// 1_000-byte budget holding a single 10-byte object -- far too slack for
	// one 44-byte ghost node to change any outcome.
	//
	// Like the watermark tests above, these derive their capacities from
	// `capacity_holding` / `watermarks::*` rather than hard-coding the
	// 0.95/0.75 defaults.
	// ---------------------------------------------------------------------

	#[test]
	fn shared_overhead_reserves_dram_and_demotes_earlier() {
		let size: ObjectSize = 10;
		let bytes = size as CacheSize;
		let overhead: CacheSize = 20;

		// The budget the two-key reservation (2 x 20 = 40) leaves behind: still
		// wide enough for the low watermark to hold one 10-byte object, so a
		// triggered pass stops after demoting exactly the older one.
		let effective = capacity_holding(bytes);
		let capacity = effective + 2 * overhead;

		// Same capacity, no reservation: both objects stay fast.
		let mut plain = S3FifoGhostHybridStack::new(1.0, 100_000, capacity);

		promote(&mut plain, 1, size);
		promote(&mut plain, 2, size);

		let plain_migrations = drain(&mut plain);

		assert!(
			!plain_migrations.iter().any(|(_, tier)| *tier == Tier::Slow),
			"without a reservation the raw capacity holds both objects, got {plain_migrations:?}",
		);
		assert_eq!(plain.fast_bytes_used(), 2 * bytes);
		assert_eq!(plain.tier_of(1), Some(Tier::Fast));
		assert_eq!(plain.tier_of(2), Some(Tier::Fast));

		// With the reservation, the same capacity gives up 20 bytes per tracked
		// key, and the watermarks then apply to whatever is left.
		let mut stack = S3FifoGhostHybridStack::new(1.0, 100_000, capacity)
			.with_shared_overhead(overhead);

		promote(&mut stack, 1, size);
		assert!(
			!drain(&mut stack).iter().any(|(_, tier)| *tier == Tier::Slow),
			"one tracked key reserves only 20 of the budget, which still holds the object",
		);

		// The second key takes the reservation to 40, dropping the effective
		// budget to `effective` -- too tight for both objects at once.
		promote(&mut stack, 2, size);
		let migrations = drain(&mut stack);

		assert!(
			migrations.contains(&(1, Tier::Slow)),
			"the reservation must demote the oldest fast key, got {migrations:?}",
		);
		assert_eq!(stack.tier_of(1), Some(Tier::Slow));
		assert_eq!(stack.tier_of(2), Some(Tier::Fast));
		assert_eq!(stack.fast_bytes_used(), bytes);
		assert!(stack.fast_bytes_used() <= watermarks::low_bytes(effective));

		// Demotion, never eviction: both keys are still tracked.
		assert_eq!(stack.len(), 2);
		assert!(!stack.needs_capacity_eviction());
	}

	#[test]
	fn shared_overhead_exceeding_capacity_demotes_all_but_never_evicts() {
		// One tracked key's reservation (100) already exceeds the whole fast
		// budget (50): the effective value budget saturates to 0, so the object
		// is demoted the moment it is promoted into the main queue.
		let mut stack = S3FifoGhostHybridStack::new(1.0, 100_000, 50).with_shared_overhead(100);

		promote(&mut stack, 1, 10);
		let migrations = drain(&mut stack);

		// No `(1, Tier::Fast)` migration is emitted at all: `settle_fast_tier`
		// runs inside `promote_from_one_access`, which only records the
		// promotion if the key survived the pass still tagged `Fast`.
		assert_eq!(migrations, vec![(1, Tier::Slow)]);
		assert_eq!(stack.tier_of(1), Some(Tier::Slow));
		assert_eq!(stack.fast_bytes_used(), 0);

		assert_eq!(stack.len(), 1);
		assert!(!stack.needs_capacity_eviction());
	}

	#[test]
	fn reservation_counts_every_tracked_key_not_just_fast_tier_ones() {
		let size: ObjectSize = 10;
		let bytes = size as CacheSize;
		let overhead: CacheSize = 20;

		// 3 tracked keys x 20 = 60 of the 65-byte budget, leaving 5 -- under
		// any watermark pair too little for a single 10-byte fast object.
		let capacity: CacheSize = 65;

		let mut stack = S3FifoGhostHybridStack::new(1.0, 100_000, capacity)
			.with_shared_overhead(overhead);

		promote(&mut stack, 1, size);
		drain(&mut stack);
		assert_eq!(stack.tier_of(1), Some(Tier::Fast));

		// Keys 2 and 3 never leave `one_access_queue`, so they are slow-tier --
		// but their hashtable/list/`entries` bookkeeping is DRAM all the same,
		// and the reservation must charge for them too.
		stack.insert(2, size);
		stack.insert(3, size);

		assert_eq!(stack.tier_of(2), Some(Tier::Slow));
		assert_eq!(stack.tier_of(3), Some(Tier::Slow));

		// `insert` alone never settles, so the tightened budget is acted on at
		// the next settle; re-applying the same capacity is the smallest
		// trigger available.
		stack.resize_fast_tier(capacity);
		let migrations = drain(&mut stack);

		assert_eq!(migrations, vec![(1, Tier::Slow)]);
		assert_eq!(stack.tier_of(1), Some(Tier::Slow));
		assert_eq!(stack.fast_bytes_used(), 0);
		assert_eq!(stack.len(), 3);

		// The identical sequence without a reservation leaves key 1 fast: it is
		// the three-key charge, not the capacity, that demoted it.
		let mut plain = S3FifoGhostHybridStack::new(1.0, 100_000, capacity);

		promote(&mut plain, 1, size);
		plain.insert(2, size);
		plain.insert(3, size);
		plain.resize_fast_tier(capacity);
		drain(&mut plain);

		assert_eq!(plain.tier_of(1), Some(Tier::Fast));
		assert_eq!(plain.fast_bytes_used(), bytes);
	}

	/// A ghost entry is DRAM held for a key that is *no longer tracked* -- no
	/// `entries` row, no `one_access_queue`/`main_queue` node, and no
	/// object-hashtable slot -- so it is charged as its own term against
	/// `ghost.len()` instead of being folded into the per-tracked-key
	/// constant.
	///
	/// The stack here is built *without* `with_shared_overhead`, which is the
	/// regression the old builder-with-a-`0`-default shape allowed: ghost DRAM
	/// is occupied whether or not the per-key term happens to be configured,
	/// so the two must not be coupled.
	///
	/// Stated against `GHOST_ENTRY_DRAM_OVERHEAD` itself rather than a
	/// hard-coded 44, so it holds under `eviction_stacks_pmem` too (where the
	/// ghost list is PMEM-resident and that constant is `0`).
	#[test]
	fn ghost_entries_are_charged_without_any_shared_overhead() {
		let ghost_cost = GHOST_ENTRY_DRAM_OVERHEAD as CacheSize;

		// Deliberately no `with_shared_overhead`: the per-tracked-key term is
		// `0` for the whole test, and every byte below is the ghost term.
		let mut stack = S3FifoGhostHybridStack::new(1.0, 100_000, 10_000);

		stack.insert(1, 10);
		assert_eq!(stack.len(), 1);
		assert_eq!(stack.reserved_overhead(), 0);

		// Key 1 ages out of `one_access_queue` into `ghost`: the tracked row is
		// gone, the ghost node is not -- and neither is the DRAM it occupies.
		assert_eq!(stack.evict_one(), Some(1));
		assert_eq!(stack.len(), 0);
		assert!(stack.is_ghost(1));
		assert_eq!(stack.reserved_overhead(), ghost_cost);

		// A second ghost key doubles the term: it scales with `ghost.len()`,
		// and `trim_ghost` has not run (only a genuine main-queue eviction
		// trims, and neither eviction here was one).
		stack.insert(2, 10);
		assert_eq!(stack.evict_one(), Some(2));
		assert!(stack.is_ghost(2));
		assert_eq!(stack.reserved_overhead(), 2 * ghost_cost);

		// A ghost hit really does occupy both structures at once under the lazy
		// `trim_ghost` convention -- key 1 is tracked again *and* still a ghost
		// -- but with no per-key term configured only the two ghost nodes are
		// charged.
		stack.insert(1, 10);
		drain(&mut stack);

		assert_eq!(stack.tier_of(1), Some(Tier::Fast));
		assert!(stack.is_ghost(1));
		assert_eq!(stack.reserved_overhead(), 2 * ghost_cost);

		// `remove` clears the ghost node, and its charge with it.
		stack.remove(1);
		assert!(!stack.is_ghost(1));
		assert_eq!(stack.reserved_overhead(), ghost_cost);

		stack.remove(2);
		assert_eq!(stack.reserved_overhead(), 0);
	}

	/// The two terms are independent and simply add: a stack that *does* wire
	/// in the per-tracked-key term still charges the ghost term on top, and
	/// each moves only with its own population.
	#[test]
	fn ghost_and_shared_overhead_terms_compose() {
		const OVERHEAD: CacheSize = 64;

		let ghost_cost = GHOST_ENTRY_DRAM_OVERHEAD as CacheSize;

		let mut stack = S3FifoGhostHybridStack::new(1.0, 100_000, 10_000)
			.with_shared_overhead(OVERHEAD);

		stack.insert(1, 10);
		assert_eq!(stack.reserved_overhead(), OVERHEAD);

		// Tracked -> ghost: the per-key term is handed back, the ghost term
		// takes its place.
		assert_eq!(stack.evict_one(), Some(1));
		assert_eq!(stack.reserved_overhead(), ghost_cost);

		// Ghost hit: tracked again, still a ghost, charged for both.
		stack.insert(1, 10);
		drain(&mut stack);

		assert_eq!(stack.tier_of(1), Some(Tier::Fast));
		assert!(stack.is_ghost(1));
		assert_eq!(stack.reserved_overhead(), OVERHEAD + ghost_cost);

		// A second *tracked* key moves only the per-key term.
		stack.insert(2, 10);
		assert_eq!(stack.reserved_overhead(), 2 * OVERHEAD + ghost_cost);
	}

	/// A ghost backlog on its own -- with no per-tracked-key term configured
	/// at all -- is enough DRAM to reserve the fast tier out from under a
	/// resident object. `trim_ghost` runs only on a genuine main-queue
	/// eviction, so a run of `one_access_queue` evictions grows `ghost`
	/// unopposed.
	///
	/// Behavioural, so it is scoped to the DRAM-resident configuration: under
	/// `eviction_stacks_pmem` the ghost list costs the fast tier nothing and
	/// there is nothing here to observe.
	#[cfg(not(feature = "eviction_stacks_pmem"))]
	#[test]
	fn a_ghost_backlog_alone_can_force_a_demotion() {
		let size: ObjectSize = 10;
		let bytes = size as CacheSize;
		let ghost_cost = GHOST_ENTRY_DRAM_OVERHEAD as CacheSize;

		// A budget whose *low* watermark still holds the single 10-byte object,
		// so with an empty ghost list nothing is demoted...
		let capacity = capacity_holding(bytes);

		// ...and the number of ghost nodes whose DRAM alone reserves the whole
		// of it, leaving an effective budget of 0 whatever the watermarks are
		// configured to.
		let ghosts = (capacity + ghost_cost - 1) / ghost_cost;

		let mut stack = S3FifoGhostHybridStack::new(1.0, 100_000, capacity);

		promote(&mut stack, 1, size);

		assert_eq!(drain(&mut stack), vec![(1, Tier::Fast)]);
		assert_eq!(stack.fast_bytes_used(), bytes);
		assert_eq!(stack.reserved_overhead(), 0);

		// Each key is admitted into `one_access_queue` and aged straight back
		// out of it, which is the one path that populates `ghost`.
		for key in 2..=ghosts + 1 {
			stack.insert(key, size);
			assert_eq!(stack.evict_one(), Some(key));
			assert!(stack.is_ghost(key));
		}

		// Key 1 is still the only *tracked* key, and it is charged nothing:
		// the whole reservation is ghost DRAM.
		assert_eq!(stack.len(), 1);
		assert_eq!(stack.reserved_overhead(), ghosts * ghost_cost);
		assert!(stack.reserved_overhead() >= capacity);

		// `insert`/`evict_one` never settle, so the tightened budget is acted
		// on at the next settle; re-applying the same capacity is the smallest
		// trigger available.
		stack.resize_fast_tier(capacity);

		assert_eq!(drain(&mut stack), vec![(1, Tier::Slow)]);
		assert_eq!(stack.tier_of(1), Some(Tier::Slow));
		assert_eq!(stack.fast_bytes_used(), 0);
		assert_eq!(stack.slow_bytes_used(), bytes);

		// Demotion, never eviction: the key is still tracked.
		assert_eq!(stack.len(), 1);
		assert!(!stack.needs_capacity_eviction());
	}

	#[test]
	fn counters_stay_consistent_across_a_reserved_budget_drain() {
		let size: ObjectSize = 10;
		let bytes = size as CacheSize;
		let overhead: CacheSize = 4;
		let count: CacheSize = 12;

		// The effective budget once all 12 keys are tracked (12 x 4 = 48
		// reserved). Its low watermark still holds 50 bytes of values, so the
		// drained tier stays non-empty and both sides can be checked.
		let effective = capacity_holding(50);
		let capacity = effective + count * overhead;

		let mut stack = S3FifoGhostHybridStack::new(1.0, 100_000, capacity)
			.with_shared_overhead(overhead);

		// Each promotion both adds 10 bytes of values and tightens the budget
		// by another 4, so several passes fire along the way.
		for key in 1..=count {
			promote(&mut stack, key, size);
		}

		drain(&mut stack);

		let fast_objects = stack.fast_object_count() as CacheSize;
		let slow_objects = stack.slow_object_count() as CacheSize;

		// The tier came to rest at or under the high watermark of the
		// *reserved* budget (a pass either never fired, or drained past the
		// low watermark, which is lower still) -- and strictly below the raw
		// byte total, i.e. a pass genuinely did fire.
		assert!(stack.fast_bytes_used() <= watermarks::high_bytes(effective));
		assert!(stack.fast_bytes_used() < count * bytes);

		// Nothing was inserted, evicted or resized mid-pass, so every object is
		// still tracked, still `size` bytes, and still on exactly one side of
		// the fast/slow line. `one_access_queue` is empty (every key was
		// promoted out of it), so `slow_*` is purely the demoted main-queue
		// tail.
		assert!(fast_objects > 0 && slow_objects > 0);
		assert_eq!(fast_objects + slow_objects, count);
		assert_eq!(stack.len() as CacheSize, count);

		assert_eq!(stack.fast_bytes_used(), fast_objects * bytes);
		assert_eq!(stack.slow_bytes_used(), slow_objects * bytes);
		assert_eq!(stack.fast_bytes_used() + stack.slow_bytes_used(), count * bytes);

		let tagged_fast = (1..=count).filter(|key| stack.tier_of(*key) == Some(Tier::Fast)).count();
		let tagged_slow = (1..=count).filter(|key| stack.tier_of(*key) == Some(Tier::Slow)).count();

		assert_eq!(tagged_fast as CacheSize, fast_objects);
		assert_eq!(tagged_slow as CacheSize, slow_objects);
	}

	#[test]
	fn overhead_composes_with_the_watermarks_rather_than_replacing_them() {
		let size: ObjectSize = 10;
		let bytes = size as CacheSize;
		let overhead: CacheSize = 5;
		let count: CacheSize = 10;

		// A capacity whose low watermark sits well below the 100 bytes of
		// values, so a single pass -- triggered deterministically by the
		// `resize_fast_tier` below rather than by admissions along the way --
		// has plenty to chew through in both arms.
		let capacity = capacity_holding(60);
		let reserved = count * overhead;
		let effective = capacity - reserved;

		let low_raw = watermarks::low_bytes(capacity);
		let low_effective = watermarks::low_bytes(effective);

		assert!(
			low_effective < low_raw,
			"the two candidate drain targets must actually differ for this test to mean anything",
		);

		// No reservation: the pass drains to the low watermark of the raw
		// capacity (rounded down to a whole object, every object being 10
		// bytes).
		let mut plain = S3FifoGhostHybridStack::new(1.0, 100_000, 100_000);

		for key in 1..=count {
			promote(&mut plain, key, size);
		}
		drain(&mut plain);
		assert_eq!(plain.fast_bytes_used(), count * bytes);

		plain.resize_fast_tier(capacity);
		drain(&mut plain);

		assert_eq!(plain.fast_bytes_used(), low_raw - low_raw % bytes);

		// With the reservation the target is the low watermark of
		// `capacity - reserved`, not of `capacity`: the watermark is applied to
		// the capacity that remains *after* the reservation.
		let mut stack = S3FifoGhostHybridStack::new(1.0, 100_000, 100_000)
			.with_shared_overhead(overhead);

		for key in 1..=count {
			promote(&mut stack, key, size);
		}
		drain(&mut stack);
		assert_eq!(stack.fast_bytes_used(), count * bytes);

		stack.resize_fast_tier(capacity);
		drain(&mut stack);

		assert_eq!(stack.fast_bytes_used(), low_effective - low_effective % bytes);
		assert!(stack.fast_bytes_used() < plain.fast_bytes_used());
		assert_eq!(stack.len() as CacheSize, count);
	}
}
