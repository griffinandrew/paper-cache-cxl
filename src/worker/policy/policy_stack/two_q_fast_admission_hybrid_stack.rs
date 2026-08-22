/*
 * Copyright (c) Kia Shakiba
 *
 * This source code is licensed under the GNU AGPLv3 license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! `TwoQFastAdmissionHybridStack` — `TwoQHybridStack` with one change: the
//! one-access FIFO queue lives in the FAST tier instead of the slow tier.
//! For `PaperPolicy::TwoQFastAdmissionHybrid`.
//!
//! Identical to `TwoQHybridStack` in every other respect (two live queues,
//! no ghost queue, main-queue LRU segmentation, FIFO-tail-first eviction
//! priority, one combined per-key `entries` map) — see that stack's module
//! doc for the full picture. Only the *physical placement* of the
//! one-access queue's bytes differs; the logical queue structure is
//! untouched.
//!
//! ## Motivation: every admission was a synchronous PMEM write
//!
//! In `two_q_hybrid_cache` admission is unconditionally to the *slow* tier —
//! the literal paper rule ("every new object is placed in the one-access
//! FIFO queue in the slow tier"). At the `PaperCache::set()` API layer that
//! means every single admission synchronously builds
//! `TieredBuffer::new_slow`, i.e. a real PMEM allocation on the calling
//! thread, before the object is even in the cache. This variant places the
//! one-access queue's bytes in the FAST tier instead, so admission becomes a
//! cheap DRAM write (`TieredBuffer::new_fast`) — the same change
//! `s3_fifo_ghost_lazy_demotion_fast_admission_hybrid_cache` already makes
//! to the s3-fifo family, and for the same reason.
//!
//! Only the one-access queue moves. The main queue keeps exactly the same
//! fast/slow segmentation and demotion behavior — a key still has to prove
//! itself with a second access to earn a place in the "real", recency-durable
//! part of the cache; the only thing that changed is which physical allocator
//! backs its bytes *while on probation* in the one-access queue.
//!
//! ## Accounting: the one-access queue now competes for the SAME DRAM budget
//!
//! This is the part that has to be handled deliberately rather than
//! relabeled. In `TwoQHybridStack`, `fifo_capacity` (`k_in * max_size`) and
//! `fast_capacity` (`fast_tier_size`) are two completely independent budgets
//! — one governs a slow/PMEM queue, the other the main queue's fast/DRAM
//! portion. Now that the FIFO queue is *also* DRAM, both draw from the same
//! physical pool, and leaving them independent would silently let real DRAM
//! usage grow to `fast_capacity + fifo_capacity` instead of the configured
//! `fast_capacity`.
//!
//! Fixed by treating `fifo_capacity` as a fixed reservation carved out of
//! `fast_capacity` first — [`Self::effective_main_fast_capacity`] =
//! `fast_capacity.saturating_sub(fifo_capacity)` — and having
//! `settle_fast_tier` (the main queue's demotion trigger) apply its shared
//! high/low watermarks to that reduced number instead of to raw
//! `fast_capacity` -- the watermarks sit on top of the effective value, they
//! never replace it (see `super::watermarks`). The FIFO queue's own byte
//! cap (`needs_capacity_eviction`) is unchanged: it was always `fifo_used >
//! fifo_capacity`, independent of tier, and still is. The net result is
//! `fast_used (main) + fifo_used <= fast_capacity` by construction (modulo
//! the same transient overshoot between eviction-loop passes every other
//! stack in this crate already tolerates), so the configured fast-tier size
//! stays a real bound on total DRAM rather than a bound on the main queue's
//! share of it. The watermarks only ever lower the point at which the main
//! queue settles, so that bound stays exactly as valid as before -- it is
//! tighter now, never looser.
//!
//! Note the reservation is the **fixed `fifo_capacity`, not the live
//! `fifo_used`**. Charging live usage would bound total DRAM more tightly,
//! but it would also make the main queue's budget breathe with FIFO
//! occupancy — churning demotions and promotions as the FIFO queue fills and
//! drains. A fixed reservation keeps the main queue's budget stable, and is
//! what `s3_fifo_ghost_lazy_demotion_fast_admission_hybrid_stack` does. Two
//! consequences follow:
//!
//! * **Admission never demotes anyone by itself.** `insert`ing a brand-new
//!   key moves no capacity, so — exactly as in `TwoQHybridStack` — it does
//!   not re-settle the fast tier. (`fifo_used <= fifo_capacity` is enforced
//!   separately, via `needs_capacity_eviction`.) See
//!   `admission_does_not_move_the_main_queues_budget`.
//! * **`resize()` must re-settle, which `TwoQHybridStack::resize` need not.**
//!   `fifo_capacity` is derived from `max_size`, so a `resize` that grows the
//!   cache grows the FIFO reservation and shrinks the main queue's effective
//!   fast budget — which has to be caught immediately, not whenever the next
//!   unrelated `insert`/`update` happens to notice. (`resize_fast_tier`
//!   already re-settled.)
//!
//! A degenerate but legitimate consequence: if `k_in * max_size` alone meets
//! or exceeds `fast_capacity`, the main queue's fast segment gets zero room
//! (saturated), and every promotion out of the FIFO queue immediately
//! self-demotes back to slow. That is correct accounting for that
//! configuration, not a bug — see
//! `zero_effective_main_capacity_demotes_every_promotion_immediately` below,
//! mirroring the equivalent documented behavior in
//! `s3_fifo_ghost_lazy_demotion_fast_admission_hybrid_stack` and
//! `lru_sized_hybrid_cache`. It is worth knowing when picking `k_in`:
//! `fifo_capacity` scales with `max_size` while the budget it is carved out
//! of is `fast_tier_size`, which is typically a small fraction of `max_size`,
//! so a `k_in` that was unremarkable under `two_q_hybrid_cache` can consume
//! most of the DRAM budget here.
//!
//! ## Shared DRAM-reservation overhead
//!
//! The object hashtable and this stack's own eviction-stack bookkeeping are
//! DRAM too, and neither is counted in `fifo_used`/`fast_used`. Left
//! unreserved, the fast tier's real DRAM footprint would be `fast_capacity`
//! *plus* `tracked keys × per-key metadata`. So `shared_overhead` (see
//! `crate::object::overhead::get_hybrid_dram_shared_overhead`) is carved out
//! of the budget as well, via [`Self::reserved_overhead`], and the result is
//! what [`Self::effective_main_fast_capacity`] returns.
//!
//! It is charged against **every tracked key**, not just fast-tier ones: a
//! demoted main-queue key's *values* move to PMEM, but its `entries` slot
//! and its `main_stack` node stay in DRAM regardless. Same convention as
//! `LruHybridStack` (`stack.len()`) and `LruSizedHybridStack`
//! (`entries.len()`).
//!
//! ### Why the whole reservation lands on the main queue
//!
//! This stack has two fast segments, and `LruSizedHybridStack` splits its
//! reservation proportionally between its two (`reserved_shares`). That
//! split does not carry over here, for two reasons:
//!
//! * **There is a single enclosing budget.** Over there, `small_capacity`
//!   and `large_capacity` are independent budgets whose *sum* is the DRAM
//!   total, so a reservation has to be divided between them to be charged
//!   exactly once rather than twice. Here `fifo_capacity` is carved *out of*
//!   `fast_capacity` and the main queue gets the residual, so subtracting
//!   the reservation from that residual alone already lowers the total by
//!   exactly `reserved`, once:
//!   `fifo_used + fast_used + reserved <= fifo_capacity + (fast_capacity −
//!   fifo_capacity − reserved) + reserved = fast_capacity`. A proportional
//!   split would reserve the same total, just distributed differently.
//! * **The FIFO segment cannot demote, only evict.** Its sole enforcement
//!   point is `needs_capacity_eviction`, which drives real *eviction* —
//!   objects leaving the cache entirely. Charging a metadata reservation
//!   there would let bookkeeping cost drop user data, which the hybrid
//!   stacks explicitly refuse to do (`LruHybridStack`: demotion is the only
//!   response, "the DRAM budget never evicts"; terminal eviction stays
//!   governed solely by `max_size`). Nor is there a demotion path to use
//!   instead — a one-access key is physically Fast *by definition* in this
//!   design. The FIFO share is therefore un-enforceable, and is borne by the
//!   only segment that can shed bytes; per the arithmetic above that is
//!   exactly right for the total, not an approximation.
//!
//! The reservation composes *underneath* the watermarks rather than beside
//! them: `settle_fast_tier` still applies `watermarks::high_bytes` /
//! `low_bytes` to [`Self::effective_main_fast_capacity`], which is now
//! `fast_capacity − fifo_capacity − reserved_overhead()`. Both subtractions
//! saturate, so a reservation at or above what the FIFO carve-out leaves
//! means "the main queue gets no fast segment" — the same legitimate
//! degenerate outcome as an oversized `k_in`, not an error. See
//! `shared_overhead_exceeding_the_main_budget_demotes_all_but_never_evicts`.
//!
//! ## A migration this design no longer needs
//!
//! `promote_from_fifo` no longer pushes a `(key, Tier::Fast)` migration. In
//! `TwoQHybridStack` that push was load-bearing: the API layer had built the
//! key's bytes as Slow (per the always-slow admission rule), so the migration
//! was the only thing that ever physically moved them into fast DRAM. Here,
//! admission already builds every brand-new key's bytes as Fast (see
//! `hybrid_policy::admission_tier` in this feature's
//! `mod.rs`), and the FIFO→main promotion is Fast→Fast — the bytes are
//! already exactly where they need to be, so emitting a migration would copy
//! a value onto itself.
//!
//! Note what this does *not* remove: a FIFO→main promotion moves bytes out
//! of the FIFO reservation and into the main-fast budget, which can push
//! `fast_used` past the effective capacity and demote *someone else*. So a
//! promotion here can still produce migrations — just only `Tier::Slow` ones,
//! never a matching `Tier::Fast`. `PolicyWorker::apply_tier_migrations`
//! already handles an arbitrary `Vec` per drain, so no worker-side change is
//! needed for this shape.

#[cfg(not(feature = "eviction_stacks_pmem"))]
use std::collections::HashMap;
#[cfg(feature = "eviction_stacks_pmem")]
use hashbrown::HashMap;

#[cfg(not(feature = "eviction_stacks_pmem"))]
use kwik::collections::HashList;
#[cfg(feature = "eviction_stacks_pmem")]
use super::pmem_collections::PmemHashList;

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
	worker::policy::policy_stack::{PolicyStack, Tier, watermarks},
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

pub struct TwoQFastAdmissionHybridStack {
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

	/// Approximate per-object DRAM cost of the shared structures (object
	/// hashtable + eviction stacks) that hold an entry for every tracked key
	/// of both tiers. Reserved out of the main queue's share of
	/// `fast_capacity` — see `effective_main_fast_capacity` and the module
	/// doc's "Shared DRAM-reservation overhead" — so that budget bounds
	/// total DRAM (values + shared metadata), not just values. `0` unless
	/// set via `with_shared_overhead`, so unit tests exercising the pure
	/// value-budget behaviour are unaffected.
	shared_overhead: CacheSize,

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

impl TwoQFastAdmissionHybridStack {
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

		TwoQFastAdmissionHybridStack {
			fifo_queue,
			main_stack,

			entries,

			k_in,
			fifo_capacity: (k_in * max_size as f64) as CacheSize,
			fifo_used: 0,

			fast_capacity,
			shared_overhead: 0,
			fast_used: 0,
			slow_used: 0,
			fast_count: 0,
			main_count: 0,

			main_boundary: None,
			migrations: Vec::new(),
		}
	}

	/// Sets the approximate per-object shared-structure DRAM overhead (object
	/// hashtable + eviction stacks) reserved out of the fast-tier budget. See
	/// `crate::object::overhead::get_hybrid_dram_shared_overhead`.
	/// Builder-style so `init_policy_stack` can wire it in without disturbing
	/// `new`'s signature (unit tests keep the default `0`).
	pub fn with_shared_overhead(mut self, overhead: CacheSize) -> Self {
		self.shared_overhead = overhead;
		self
	}

	/// Total DRAM currently reserved for shared per-object metadata across
	/// every tracked key (`entries.len() × shared_overhead`), whichever queue
	/// and whichever tier each of them is in — the hashtable slot and the
	/// `HashList` node are DRAM even for a demoted key. Subtracted, in full,
	/// from the main queue's share of `fast_capacity` in
	/// `effective_main_fast_capacity`; see the module doc for why none of it
	/// is charged against `fifo_capacity`.
	fn reserved_overhead(&self) -> CacheSize {
		self.entries.len() as CacheSize * self.shared_overhead
	}

	/// How much of `fast_capacity` the main queue's fast segment may use,
	/// after the FIFO queue's reservation and the shared per-object metadata
	/// reservation are both carved out.
	///
	/// Saturating rather than panicking when the two carve-outs meet or
	/// exceed `fast_capacity`: that is a legitimate (if degenerate)
	/// configuration — see the module doc — and it means "the main queue gets
	/// no fast segment", not an error.
	///
	/// The watermarks in `settle_fast_tier` are applied *on top of* this
	/// value, never in place of it.
	fn effective_main_fast_capacity(&self) -> CacheSize {
		self.fast_capacity
			.saturating_sub(self.fifo_capacity)
			.saturating_sub(self.reserved_overhead())
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

	/// Demotes the least-recently-used fast key(s) within `main_stack` once
	/// `fast_used` exceeds the HIGH watermark of
	/// [`Self::effective_main_fast_capacity`], then drains down to the LOW
	/// watermark rather than merely back under the ceiling.
	///
	/// The capacity the watermarks are applied *to* is the one substantive
	/// difference from `TwoQHybridStack::settle_fast_tier`, which applies
	/// them to raw `fast_capacity` — correct there, where the FIFO queue is
	/// PMEM and competes for nothing. The watermarks sit on top of that
	/// effective value; they never replace it.
	///
	/// That effective value now nets out *two* carve-outs, not one: the FIFO
	/// queue's fixed `fifo_capacity` and the shared per-object metadata
	/// reservation (`reserved_overhead()` — object hashtable + eviction
	/// stacks, charged for every tracked key of both tiers). Composition
	/// order is: reservations first, watermarks on the remainder. Demotion
	/// stays the only response to either; neither ever evicts.
	///
	/// Draining below the ceiling instead of exactly to it is what turns the
	/// steady state from "every promotion demotes exactly one object" into
	/// occasional multi-object batches — see [`watermarks`] for the full
	/// rationale, and for how to restore the old drain-to-ceiling behaviour
	/// exactly (`FAST_TIER_HIGH_WATERMARK=1.0`, `FAST_TIER_LOW_WATERMARK=1.0`).
	///
	/// Nothing else moves: the per-demotion bookkeeping in the loop below
	/// (tier tag, `fast_used`/`slow_used`, `fast_count`, the boundary walk,
	/// the migration push) is unchanged and still runs exactly once per
	/// demoted object.
	fn settle_fast_tier(&mut self) {
		let effective = self.effective_main_fast_capacity();

		// Trigger only once usage is past the high watermark...
		if self.fast_used <= watermarks::high_bytes(effective) {
			return;
		}

		// ...but once triggered, drain all the way down to the low one.
		let drain_target = watermarks::low_bytes(effective);

		while self.fast_used > drain_target {
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
	/// Deliberately **not** called from `insert`/`resize` to self-evict under
	/// `fifo_capacity` pressure: a `PolicyStack` has no reference to the
	/// shared object map or `status`, so it can only update its own
	/// bookkeeping here — it cannot remove the object from the cache or
	/// adjust accounted size, and doing so anyway would silently desync this
	/// stack from the real object map. Real removal always goes through
	/// `PolicyWorker::apply_evictions`'s `evict_one()` + `erase()` pairing,
	/// which is why `fifo_capacity` pressure is surfaced via
	/// `needs_capacity_eviction` instead. (This bug was found the hard way in
	/// `TwoQHybridStack` — see `CLAUDE.md`.)
	fn evict_fifo_tail(&mut self) -> Option<HashedKey> {
		let key = self.fifo_queue.pop_back()?;
		let size = self.entries.remove(&key).map(|entry| entry.size).unwrap_or(0) as CacheSize;

		self.fifo_used = self.fifo_used.saturating_sub(size);

		Some(key)
	}
}

impl PolicyStack for TwoQFastAdmissionHybridStack {
	fn is_policy(&self, policy: &PaperPolicy) -> bool {
		matches!(policy, PaperPolicy::TwoQFastAdmissionHybrid(k_in) if *k_in == self.k_in)
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
		// If this pushes fifo_used over fifo_capacity, `needs_capacity_eviction`
		// reports it and `apply_evictions` drains it via `evict_one` (see
		// `evict_fifo_tail`'s doc for why eviction can't happen here).
		self.fifo_queue.push_front(key);
		self.entries.insert(key, TwoQEntry { queue: Queue::Fifo, tier: None, size });
		self.fifo_used += size as CacheSize;

		// Deliberately does NOT re-settle the fast tier, despite admission
		// now consuming DRAM. The reservation carved out of `fast_capacity`
		// is the fixed `fifo_capacity`, not the live `fifo_used`, so the main
		// queue's effective budget doesn't move when the FIFO queue fills --
		// only `resize`/`resize_fast_tier` can change it. Charging live
		// `fifo_used` instead would make the main queue's budget breathe with
		// FIFO occupancy and churn demotions/promotions as the queue fills
		// and drains; a fixed reservation keeps it stable, and is what
		// `s3_fifo_ghost_lazy_demotion_fast_admission_hybrid_stack` does too.
		// `fifo_used <= fifo_capacity` is enforced separately, via
		// `needs_capacity_eviction`.
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

		// A shrink may also push fifo_used over the new, smaller
		// fifo_capacity; `needs_capacity_eviction` reports it and
		// `apply_evictions` drains it.
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

	fn needs_capacity_eviction(&self) -> bool {
		self.fifo_used > self.fifo_capacity
	}
}

#[cfg(test)]
mod tests {
	use super::*;

	fn drain(stack: &mut TwoQFastAdmissionHybridStack) -> Vec<(HashedKey, Tier)> {
		stack.drain_tier_migrations()
	}

	/// One `settle_fast_tier` pass over `count` equal-sized keys: how many are
	/// left tagged `Fast` afterwards.
	///
	/// The expectations below are derived from this rather than hard-coded so
	/// that every test holds at any configured `FAST_TIER_HIGH_WATERMARK` /
	/// `FAST_TIER_LOW_WATERMARK` pair. They cannot simply pin the ratios: the
	/// watermarks are process-global `OnceLock`s read once per process, so a
	/// test that set them would race every other test in the binary.
	fn settled(effective: CacheSize, size: CacheSize, count: usize) -> usize {
		let high = watermarks::high_bytes(effective);
		let low = watermarks::low_bytes(effective);

		let mut fast = count;

		if fast as CacheSize * size > high {
			while fast > 0 && fast as CacheSize * size > low {
				fast -= 1;
			}
		}

		fast
	}

	/// Mirrors a run of equal-sized keys promoted into the main queue one at a
	/// time: element `i` is how many are left tagged `Fast` once the `i + 1`th
	/// promotion has settled.
	fn fast_key_counts(effective: CacheSize, size: CacheSize, promotions: usize) -> Vec<usize> {
		let mut fast = 0usize;
		let mut counts = Vec::with_capacity(promotions);

		for _ in 0..promotions {
			fast = settled(effective, size, fast + 1);
			counts.push(fast);
		}

		counts
	}

	/// How many equal-sized keys are left `Fast` after `promotions` of them
	/// have been promoted one at a time.
	fn fast_keys_after(effective: CacheSize, size: CacheSize, promotions: usize) -> usize {
		fast_key_counts(effective, size, promotions)[promotions - 1]
	}

	/// The headline difference from `TwoQHybridStack`: a brand-new key is
	/// Fast, not Slow, so `set()` never pays a synchronous PMEM allocation.
	#[test]
	fn admission_lands_in_the_fifo_queue_in_the_fast_tier() {
		let mut stack = TwoQFastAdmissionHybridStack::new(0.5, 1_000, 1_000);

		stack.insert(1, 10);
		stack.insert(2, 10);

		assert_eq!(stack.tier_of(1), Some(Tier::Fast));
		assert_eq!(stack.tier_of(2), Some(Tier::Fast));

		// Admission is already physically correct, so nothing to migrate.
		assert!(drain(&mut stack).is_empty());
	}

	#[test]
	fn admission_counts_toward_the_fast_tier_gauges() {
		let mut stack = TwoQFastAdmissionHybridStack::new(0.5, 1_000, 1_000);

		stack.insert(1, 10);
		stack.insert(2, 30);

		assert_eq!(stack.fast_bytes_used(), 40);
		assert_eq!(stack.slow_bytes_used(), 0);
		assert_eq!(stack.fast_object_count(), 2);
		assert_eq!(stack.slow_object_count(), 0);
	}

	/// A FIFO→main promotion is a bookkeeping move between two DRAM
	/// structures, so it must not emit a migration — emitting one would copy
	/// the value onto itself.
	#[test]
	fn promotion_out_of_the_fifo_queue_emits_no_migration() {
		let mut stack = TwoQFastAdmissionHybridStack::new(0.5, 1_000, 1_000);

		stack.insert(1, 10);
		drain(&mut stack);

		stack.update(1);

		assert_eq!(stack.tier_of(1), Some(Tier::Fast));
		assert!(drain(&mut stack).is_empty());

		// Still one fast object, now counted via the main queue rather than
		// the FIFO queue.
		assert_eq!(stack.fast_object_count(), 1);
		assert_eq!(stack.fast_bytes_used(), 10);
	}

	/// The FIFO reservation is carved out of `fast_capacity`, so the main
	/// queue's usable fast budget is the remainder.
	#[test]
	fn effective_main_capacity_excludes_the_fifo_reservation() {
		// k_in 0.25 of max_size 1_000 => fifo_capacity 250, out of a 1_000
		// byte fast tier => 750 left for the main queue.
		let stack = TwoQFastAdmissionHybridStack::new(0.25, 1_000, 1_000);

		assert_eq!(stack.effective_main_fast_capacity(), 750);
	}

	/// Total DRAM (FIFO + main-fast) must stay within `fast_capacity`, which
	/// only holds because `settle_fast_tier` applies its watermarks to the
	/// *effective* budget.
	#[test]
	fn main_queue_demotes_once_it_exceeds_the_effective_budget() {
		// fifo_capacity = 100, fast_capacity = 300 => main gets 200.
		let mut stack = TwoQFastAdmissionHybridStack::new(0.1, 1_000, 300);
		let effective = stack.effective_main_fast_capacity();

		assert_eq!(effective, 200);

		for key in 1..=3 {
			stack.insert(key, 100);
			stack.update(key);
		}

		drain(&mut stack);

		// Three 100-byte keys promoted into a 200-byte effective budget: the
		// LRU-most one(s) must have been pushed out to slow. Exactly how many
		// survive is the low watermark's call.
		let fast_keys = fast_keys_after(effective, 100, 3);

		assert!(fast_keys < 3, "a 200-byte budget cannot hold three 100-byte keys");
		assert_eq!(stack.fast_used, fast_keys as CacheSize * 100);
		assert_eq!(stack.fast_count, fast_keys);

		// The survivors are the most recently promoted keys.
		let first_fast = (3 - fast_keys) as HashedKey + 1;

		for key in 1..=3u64 {
			let expected = if key >= first_fast { Tier::Fast } else { Tier::Slow };

			assert_eq!(stack.tier_of(key), Some(expected), "key {key}");
		}

		// And the reported DRAM total stays within the configured fast tier.
		assert!(stack.fast_bytes_used() <= 300);
	}

	/// A demotion moves real bytes DRAM→PMEM, so unlike a FIFO promotion it
	/// must emit a migration.
	#[test]
	fn demotion_out_of_the_main_queue_emits_a_slow_migration() {
		let mut stack = TwoQFastAdmissionHybridStack::new(0.1, 1_000, 300);
		let effective = stack.effective_main_fast_capacity();

		for key in 1..=3 {
			stack.insert(key, 100);
			stack.update(key);
		}

		let migrations = drain(&mut stack);

		// Demotions come off the LRU end, i.e. in promotion order.
		let demoted = (3 - fast_keys_after(effective, 100, 3)) as HashedKey;
		let expected = (1..=demoted).map(|key| (key, Tier::Slow)).collect::<Vec<_>>();

		assert!(demoted >= 1);
		assert_eq!(migrations, expected);
	}

	/// A promotion here can produce a demotion with no matching `Tier::Fast`
	/// entry — the shape that only exists because admission is already fast.
	#[test]
	fn a_fifo_promotion_can_demote_someone_else_without_promoting_itself() {
		let mut stack = TwoQFastAdmissionHybridStack::new(0.1, 1_000, 300);
		let effective = stack.effective_main_fast_capacity();

		for key in 1..=2 {
			stack.insert(key, 100);
			stack.update(key);
		}

		drain(&mut stack);

		// A third key: admitted fast, then re-accessed to promote it. Its own
		// promotion emits nothing, but it displaces the LRU end of the main
		// queue's fast segment out of the 200-byte effective budget.
		stack.insert(3, 100);
		stack.update(3);

		let migrations = drain(&mut stack);

		let counts = fast_key_counts(effective, 100, 3);
		let already_demoted = (2 - counts[1]) as HashedKey;
		let now_demoted = (3 - counts[2]) as HashedKey;

		let expected = ((already_demoted + 1)..=now_demoted)
			.map(|key| (key, Tier::Slow))
			.collect::<Vec<_>>();

		assert!(!expected.is_empty(), "the third promotion must displace someone");
		assert_eq!(migrations, expected);

		// The promotion itself never emits a matching `Tier::Fast` entry.
		assert!(!migrations.iter().any(|(_, tier)| *tier == Tier::Fast));
		assert_eq!(
			stack.tier_of(3),
			Some(if counts[2] > 0 { Tier::Fast } else { Tier::Slow }),
		);
	}

	/// Re-accessing a demoted main-queue key is a real PMEM→DRAM move.
	#[test]
	fn re_accessing_a_slow_main_key_promotes_it_with_a_migration() {
		let mut stack = TwoQFastAdmissionHybridStack::new(0.1, 1_000, 300);
		let effective = stack.effective_main_fast_capacity();

		for key in 1..=3 {
			stack.insert(key, 100);
			stack.update(key);
		}

		drain(&mut stack);
		assert_eq!(stack.tier_of(1), Some(Tier::Slow));

		let before = fast_keys_after(effective, 100, 3);
		let after = settled(effective, 100, before + 1);

		stack.update(1);

		let migrations = drain(&mut stack);

		if after > 0 {
			// Key 1 is now the most recently used, so a pass triggered by its
			// own promotion reaches it last: it survives, and its PMEM->DRAM
			// move is reported.
			assert_eq!(stack.tier_of(1), Some(Tier::Fast));
			assert!(migrations.contains(&(1, Tier::Fast)));
		} else {
			// Degenerate watermarks: the pass drains the segment completely
			// and takes key 1 straight back out, in which case
			// `touch_main_fast`'s guard correctly suppresses the stale
			// `Tier::Fast` entry.
			assert_eq!(stack.tier_of(1), Some(Tier::Slow));
			assert!(!migrations.contains(&(1, Tier::Fast)));
		}

		// One `Tier::Slow` entry per object the pass had to displace.
		let demoted = migrations.iter().filter(|(_, tier)| *tier == Tier::Slow).count();

		assert_eq!(demoted, before + 1 - after);

		if demoted > 0 {
			// Demotions come off the LRU end, so the first one is whichever
			// fast key was the demotion candidate before key 1 rejoined.
			assert_eq!(migrations.first(), Some(&((3 - before) as HashedKey + 1, Tier::Slow)));
		}
	}

	/// The reservation is the fixed `fifo_capacity`, not the live
	/// `fifo_used`, so the main queue's budget does not move as the FIFO
	/// queue fills — admission never demotes anyone by itself.
	///
	/// This pins down a decision that could reasonably have gone the other
	/// way (charging live `fifo_used` would bound total DRAM more tightly,
	/// at the cost of a main-queue budget that breathes with FIFO occupancy
	/// and churns migrations as it does). If that decision is ever revisited,
	/// this test is the one that should fail.
	#[test]
	fn admission_does_not_move_the_main_queues_budget() {
		// fifo_capacity 100, fast_capacity 300 => a fixed 200 for the main
		// queue, whether the FIFO queue is empty or full.
		let mut stack = TwoQFastAdmissionHybridStack::new(0.1, 1_000, 300);

		for key in 1..=2 {
			stack.insert(key, 100);
			stack.update(key);
		}

		drain(&mut stack);

		assert_eq!(stack.effective_main_fast_capacity(), 200);

		// Wherever the watermarks left the main queue, a brand-new admission
		// must not disturb it -- so snapshot rather than hard-code.
		let tiers_before = (1..=2).map(|key| stack.tier_of(key)).collect::<Vec<_>>();
		let fast_used_before = stack.fast_used;
		let slow_used_before = stack.slow_used;

		// A brand-new admission fills the FIFO queue to its cap. The main
		// queue's budget does not move, so nothing there may change.
		stack.insert(3, 100);

		assert_eq!(stack.effective_main_fast_capacity(), 200);
		assert!(drain(&mut stack).is_empty());
		assert_eq!(stack.fast_used, fast_used_before);
		assert_eq!(stack.slow_used, slow_used_before);
		assert_eq!((1..=2).map(|key| stack.tier_of(key)).collect::<Vec<_>>(), tiers_before);

		// The new key itself is FIFO-resident, hence Fast.
		assert_eq!(stack.tier_of(3), Some(Tier::Fast));
	}

	/// The flip side of the fixed reservation: the main queue cannot borrow
	/// the FIFO queue's reserved bytes even while the FIFO queue is empty.
	/// This is what keeps total DRAM within `fast_capacity` once the FIFO
	/// queue does fill.
	#[test]
	fn the_fifo_reservation_is_held_even_while_the_fifo_queue_is_empty() {
		let mut stack = TwoQFastAdmissionHybridStack::new(0.1, 1_000, 300);
		let effective = stack.effective_main_fast_capacity();

		for key in 1..=3 {
			stack.insert(key, 100);
			stack.update(key);
		}

		drain(&mut stack);

		// All three were promoted out of the FIFO queue, so it is empty --
		// yet the main queue still only gets 200 of the 300 byte fast tier,
		// and the watermarks hold it below even that.
		assert_eq!(stack.fifo_used, 0);
		assert_eq!(stack.fast_used, fast_keys_after(effective, 100, 3) as CacheSize * 100);
		assert!(stack.fast_used <= watermarks::high_bytes(effective));

		// 300 bytes of keys never fit in a 200-byte budget at any watermark,
		// so the LRU-most key is demoted regardless of how they are tuned.
		assert_eq!(stack.tier_of(1), Some(Tier::Slow));
	}

	/// `resize` changes `fifo_capacity`, which moves the main queue's
	/// effective budget — the reason this stack re-settles there and
	/// `TwoQHybridStack` doesn't.
	#[test]
	fn growing_max_size_grows_the_fifo_reservation_and_re_settles() {
		// k_in 0.5: fifo_capacity tracks max_size closely, so a resize moves
		// the effective main budget a lot.
		let mut stack = TwoQFastAdmissionHybridStack::new(0.5, 200, 400);

		// fifo_capacity 100 => 300 for the main queue.
		assert_eq!(stack.effective_main_fast_capacity(), 300);

		for key in 1..=3 {
			stack.insert(key, 100);
			stack.update(key);
		}

		drain(&mut stack);

		let before = fast_keys_after(300, 100, 3);

		assert_eq!(stack.fast_used, before as CacheSize * 100);

		// max_size 600 => fifo_capacity 300 => only 100 left for the main
		// queue, so the shrunken budget must demote immediately.
		stack.resize(600);

		let migrations = drain(&mut stack);

		assert_eq!(stack.effective_main_fast_capacity(), 100);

		let after = settled(100, 100, before);

		assert!(after <= before);
		assert_eq!(stack.fast_used, after as CacheSize * 100);

		// Demotions keep coming off the LRU end in promotion order, picking up
		// wherever the pre-resize settles left off.
		let expected = (((3 - before) as HashedKey + 1)..=((3 - after) as HashedKey))
			.map(|key| (key, Tier::Slow))
			.collect::<Vec<_>>();

		assert_eq!(migrations, expected);
	}

	/// Degenerate but legitimate: a FIFO reservation at or above the whole
	/// fast tier leaves the main queue nothing.
	#[test]
	fn zero_effective_main_capacity_demotes_every_promotion_immediately() {
		// fifo_capacity = 1.0 * 1_000 = 1_000 == fast_capacity => 0 left.
		let mut stack = TwoQFastAdmissionHybridStack::new(1.0, 1_000, 1_000);

		assert_eq!(stack.effective_main_fast_capacity(), 0);

		stack.insert(1, 10);
		assert_eq!(stack.tier_of(1), Some(Tier::Fast));

		stack.update(1);

		assert_eq!(stack.tier_of(1), Some(Tier::Slow));
		assert_eq!(drain(&mut stack), vec![(1, Tier::Slow)]);
	}

	/// FIFO-tail-first, same as `TwoQHybridStack` — but the bytes freed are
	/// DRAM now, not PMEM.
	#[test]
	fn eviction_takes_the_fifo_tail_first() {
		let mut stack = TwoQFastAdmissionHybridStack::new(0.5, 1_000, 1_000);

		stack.insert(1, 10);
		stack.insert(2, 10);
		stack.update(2); // promote 2 into the main queue

		stack.insert(3, 10);

		// 1 is the FIFO tail (3 was pushed to the front), so it goes first.
		assert_eq!(stack.evict_one(), Some(1));
		assert_eq!(stack.evict_one(), Some(3));

		// Only the main queue is left.
		assert_eq!(stack.evict_one(), Some(2));
		assert_eq!(stack.evict_one(), None);
	}

	#[test]
	fn fifo_capacity_pressure_is_reported_not_self_evicted() {
		let mut stack = TwoQFastAdmissionHybridStack::new(0.1, 1_000, 1_000);

		assert!(!stack.needs_capacity_eviction());

		stack.insert(1, 60);
		stack.insert(2, 60);

		// 120 > fifo_capacity 100, but the stack must not have removed
		// anything itself -- only reported the pressure.
		assert!(stack.needs_capacity_eviction());
		assert_eq!(stack.len(), 2);
	}

	#[test]
	fn removing_a_fifo_key_releases_its_fast_tier_bytes() {
		let mut stack = TwoQFastAdmissionHybridStack::new(0.5, 1_000, 1_000);

		stack.insert(1, 40);
		assert_eq!(stack.fast_bytes_used(), 40);

		stack.remove(1);

		assert_eq!(stack.fast_bytes_used(), 0);
		assert_eq!(stack.fast_object_count(), 0);
		assert_eq!(stack.len(), 0);
	}

	#[test]
	fn clear_resets_every_counter() {
		let mut stack = TwoQFastAdmissionHybridStack::new(0.1, 1_000, 300);

		for key in 1..=3 {
			stack.insert(key, 100);
			stack.update(key);
		}

		stack.clear();

		assert_eq!(stack.len(), 0);
		assert_eq!(stack.fast_bytes_used(), 0);
		assert_eq!(stack.slow_bytes_used(), 0);
		assert_eq!(stack.fast_object_count(), 0);
		assert_eq!(stack.slow_object_count(), 0);
		assert!(!stack.needs_capacity_eviction());
		assert!(drain(&mut stack).is_empty());
	}

	#[test]
	fn is_policy_matches_only_its_own_variant_and_k_in() {
		let stack = TwoQFastAdmissionHybridStack::new(0.1, 1_000, 300);

		assert!(stack.is_policy(&PaperPolicy::TwoQFastAdmissionHybrid(0.1)));
		assert!(!stack.is_policy(&PaperPolicy::TwoQFastAdmissionHybrid(0.2)));
		assert!(!stack.is_policy(&PaperPolicy::TwoQHybrid(0.1)));
	}

	/// Watermark test rig: fifo_capacity 1_000 out of an 11_000 byte fast
	/// tier leaves the main queue a round 10_000, so `high_bytes` /
	/// `low_bytes` land on exact byte counts whatever the ratios are.
	fn watermark_stack() -> TwoQFastAdmissionHybridStack {
		let stack = TwoQFastAdmissionHybridStack::new(0.1, 10_000, 11_000);

		assert_eq!(stack.effective_main_fast_capacity(), 10_000);

		stack
	}

	/// Sitting just under the high watermark triggers nothing at all -- the
	/// old rule would only have left the tier alone below the ceiling.
	#[test]
	fn usage_just_below_the_high_watermark_triggers_no_demotion() {
		let mut stack = watermark_stack();
		let effective = stack.effective_main_fast_capacity();
		let high = watermarks::high_bytes(effective);

		assert!(high > 1, "watermark config leaves no room for this test");

		// A single key sized one byte under the high watermark, promoted into
		// the main queue.
		stack.insert(1, (high - 1) as ObjectSize);
		stack.update(1);

		assert_eq!(stack.fast_used, high - 1);
		assert_eq!(stack.tier_of(1), Some(Tier::Fast));
		assert!(drain(&mut stack).is_empty());
		assert_eq!(stack.slow_used, 0);
		assert_eq!(stack.slow_object_count(), 0);
	}

	/// Sitting exactly *on* the high watermark still triggers nothing (the
	/// test is `>`, not `>=`); one byte past it does.
	#[test]
	fn usage_above_the_high_watermark_triggers_a_pass() {
		let mut stack = watermark_stack();
		let effective = stack.effective_main_fast_capacity();
		let high = watermarks::high_bytes(effective);
		let low = watermarks::low_bytes(effective);

		assert!(high >= 1, "watermark config leaves no room for this test");

		stack.insert(1, high as ObjectSize);
		stack.update(1);

		assert_eq!(stack.fast_used, high);
		assert!(drain(&mut stack).is_empty(), "exactly on the high watermark must not trigger");

		// One more byte tips it over.
		stack.insert(2, 1);
		stack.update(2);

		let migrations = drain(&mut stack);

		// Key 1 is the LRU-most fast key, so it is the one that goes.
		assert_eq!(migrations, vec![(1, Tier::Slow)]);
		assert_eq!(stack.tier_of(1), Some(Tier::Slow));
		assert_eq!(stack.tier_of(2), Some(Tier::Fast));
		assert!(stack.fast_used <= low);
	}

	/// The point of the change: a triggered pass drains to the LOW watermark,
	/// not merely back under the ceiling it used to settle at.
	#[test]
	fn a_triggered_pass_drains_to_the_low_watermark() {
		let mut stack = watermark_stack();
		let effective = stack.effective_main_fast_capacity();
		let high = watermarks::high_bytes(effective);
		let low = watermarks::low_bytes(effective);
		let size: CacheSize = 100;

		// Fill with the largest whole number of 100-byte keys that still sits
		// at or below the high watermark: no pass has run yet.
		let filled = high / size;

		assert!(filled >= 1, "watermark config leaves no room for this test");

		for key in 1..=filled {
			stack.insert(key, size as ObjectSize);
			stack.update(key);
		}

		assert_eq!(stack.fast_used, filled * size);
		assert!(drain(&mut stack).is_empty());

		// One more key tips it past the high watermark.
		stack.insert(filled + 1, size as ObjectSize);
		stack.update(filled + 1);

		let migrations = drain(&mut stack);

		assert!(!migrations.is_empty(), "crossing the high watermark must trigger a pass");

		// Drained past the ceiling, all the way down to the low watermark...
		assert!(stack.fast_used <= low);
		assert_eq!(stack.fast_used, low / size * size);
		assert!(stack.fast_used < effective || low >= effective);

		// ...and stopped as soon as it got under, rather than emptying the
		// segment: one fewer demotion would have left it above the target.
		assert!(stack.fast_used + size > low);

		// The whole pass is one batch, which is the behaviour being bought:
		// one `Tier::Slow` entry per demoted object, off the LRU end in
		// promotion order.
		let demoted = (filled + 1) - stack.fast_count as CacheSize;

		assert_eq!(migrations.len() as CacheSize, demoted);
		assert_eq!(migrations, (1..=demoted).map(|key| (key, Tier::Slow)).collect::<Vec<_>>());
	}

	/// Every byte and every object is still accounted for exactly once after a
	/// multi-object drain: the per-demotion bookkeeping did not change, only
	/// how many times it runs per pass.
	#[test]
	fn counters_stay_consistent_after_a_watermark_pass() {
		let mut stack = watermark_stack();
		let effective = stack.effective_main_fast_capacity();
		let size: CacheSize = 100;
		let promoted = watermarks::high_bytes(effective) / size + 1;

		assert!(promoted >= 2, "watermark config leaves no room for this test");

		for key in 1..=promoted {
			stack.insert(key, size as ObjectSize);
			stack.update(key);
		}

		// A brand-new key left sitting in the FIFO queue, to pin that the pass
		// touched neither it nor the FIFO accounting.
		stack.insert(promoted + 1, 40);

		let migrations = drain(&mut stack);
		let demoted = migrations.len() as CacheSize;

		assert!(demoted >= 1, "the pass must have run");
		assert!(migrations.iter().all(|(_, tier)| *tier == Tier::Slow));

		// Bytes: every promoted key is either main-fast or main-slow, and the
		// FIFO key is neither.
		assert_eq!(stack.fast_used, (promoted - demoted) * size);
		assert_eq!(stack.slow_used, demoted * size);
		assert_eq!(stack.fifo_used, 40);
		assert_eq!(stack.fast_used + stack.slow_used, promoted * size);

		// Counts: `fast_count` covers main-fast only, `main_count` every
		// promoted key.
		assert_eq!(stack.fast_count as CacheSize, promoted - demoted);
		assert_eq!(stack.main_count as CacheSize, promoted);
		assert_eq!(stack.len() as CacheSize, promoted + 1);

		// Reported gauges add the FIFO queue back into the fast side.
		assert_eq!(stack.fast_bytes_used(), (promoted - demoted) * size + 40);
		assert_eq!(stack.slow_bytes_used(), demoted * size);
		assert_eq!(stack.fast_object_count() as CacheSize, promoted - demoted + 1);
		assert_eq!(stack.slow_object_count() as CacheSize, demoted);

		// The per-key tier tags agree with those counters.
		let fast_tagged = (1..=promoted).filter(|key| stack.tier_of(*key) == Some(Tier::Fast)).count();
		let slow_tagged = (1..=promoted).filter(|key| stack.tier_of(*key) == Some(Tier::Slow)).count();

		assert_eq!(fast_tagged as CacheSize, promoted - demoted);
		assert_eq!(slow_tagged as CacheSize, demoted);

		// The boundary is the LRU-most still-fast key, or `None` if the pass
		// drained the segment completely.
		assert_eq!(
			stack.main_boundary,
			if stack.fast_count > 0 { Some(demoted + 1) } else { None },
		);

		// And total DRAM is still within the configured fast tier.
		assert!(stack.fast_bytes_used() <= 11_000);
	}

	// ---------------------------------------------------------------------
	// Shared DRAM-reservation overhead (`with_shared_overhead`).
	//
	// Every test above constructs the stack WITHOUT `with_shared_overhead`,
	// so it legitimately sees `shared_overhead == 0` and
	// `reserved_overhead() == 0` -- their effective capacities, and therefore
	// every expectation they make, are unchanged by this feature.
	//
	// Like the watermark tests, these derive their expectations from
	// `watermarks::high_bytes`/`low_bytes` (via `settled`/`fast_keys_after`)
	// rather than hard-coding ratios, so they hold at any configured
	// watermark pair.
	// ---------------------------------------------------------------------

	/// The reservation scales with *tracked* keys, wherever they live: a
	/// FIFO-resident key, a main-fast key and a demoted main-slow key each
	/// own a DRAM `entries` slot and a DRAM `HashList` node, so each is
	/// charged. Moving between queues or tiers moves no reservation;
	/// untracking hands it back.
	#[test]
	fn the_reservation_covers_every_tracked_key_not_just_fast_ones() {
		// fifo_capacity 100 out of a 1_000-byte fast tier => a raw 900 for
		// the main queue before any metadata is charged.
		let mut stack = TwoQFastAdmissionHybridStack::new(0.1, 1_000, 1_000)
			.with_shared_overhead(30);

		assert_eq!(stack.effective_main_fast_capacity(), 900);

		// A FIFO-resident key is tracked exactly like a main-queue one.
		stack.insert(1, 10);
		assert_eq!(stack.effective_main_fast_capacity(), 870);

		stack.insert(2, 10);
		stack.insert(3, 10);
		assert_eq!(stack.len(), 3);
		assert_eq!(stack.effective_main_fast_capacity(), 810);

		// Promotion moves a key between the two lists without changing how
		// many keys are tracked, so the reservation does not move either.
		stack.update(1);
		assert_eq!(stack.len(), 3);
		assert_eq!(stack.effective_main_fast_capacity(), 810);

		// Neither does demotion: shrinking the fast tier to the FIFO
		// reservation plus nothing leaves a zero-byte main segment, which
		// pushes key 1 out to slow -- and its metadata is still charged.
		stack.resize_fast_tier(100);

		assert_eq!(stack.tier_of(1), Some(Tier::Slow));
		assert_eq!(stack.len(), 3);
		assert_eq!(stack.effective_main_fast_capacity(), 0);

		// Untracking a key, on the other hand, does hand its share back:
		// 2 keys x 30 = 60 reserved instead of 90.
		stack.remove(3);
		stack.resize_fast_tier(1_000);

		assert_eq!(stack.len(), 2);
		assert_eq!(stack.effective_main_fast_capacity(), 840);
	}

	/// The headline: two stacks with identical capacities running an
	/// identical workload part ways once one of them charges shared per-key
	/// DRAM metadata against the budget. The charging one demotes strictly
	/// earlier. Modelled on `LruHybridStack`'s
	/// `shared_overhead_reserves_dram_and_demotes_earlier`.
	#[test]
	fn shared_overhead_reserves_dram_and_demotes_earlier() {
		// fifo_capacity 500 out of a 1_500-byte fast tier => a raw 1_000 for
		// the main queue in both stacks.
		let mut plain = TwoQFastAdmissionHybridStack::new(0.5, 1_000, 1_500);
		let mut charged = TwoQFastAdmissionHybridStack::new(0.5, 1_000, 1_500)
			.with_shared_overhead(300);

		// Admit all three first, so `entries.len()` -- and therefore the
		// reservation -- is fixed for the whole promotion run and the
		// `fast_keys_after` model applies exactly.
		for key in 1..=3 {
			plain.insert(key, 100);
			charged.insert(key, 100);
		}

		assert_eq!(plain.effective_main_fast_capacity(), 1_000);
		// 3 keys x 300 = 900 reserved out of the same raw 1_000.
		assert_eq!(charged.effective_main_fast_capacity(), 100);

		for key in 1..=3 {
			plain.update(key);
			charged.update(key);
		}

		let plain_fast = fast_keys_after(1_000, 100, 3);
		let charged_fast = fast_keys_after(100, 100, 3);

		assert!(
			charged_fast < plain_fast,
			"watermark config leaves no room for this test",
		);

		assert_eq!(plain.fast_count, plain_fast);
		assert_eq!(plain.fast_used, plain_fast as CacheSize * 100);

		assert_eq!(charged.fast_count, charged_fast);
		assert_eq!(charged.fast_used, charged_fast as CacheSize * 100);

		// Demotions come off the LRU end in promotion order in both stacks;
		// the charged one simply has more of them.
		let plain_migrations = drain(&mut plain);
		let charged_migrations = drain(&mut charged);

		assert_eq!(
			plain_migrations,
			(1..=(3 - plain_fast) as HashedKey).map(|key| (key, Tier::Slow)).collect::<Vec<_>>(),
		);
		assert_eq!(
			charged_migrations,
			(1..=(3 - charged_fast) as HashedKey).map(|key| (key, Tier::Slow)).collect::<Vec<_>>(),
		);
		assert!(charged_migrations.len() > plain_migrations.len());

		// Neither stack evicted anything -- demotion is the only response.
		assert_eq!(plain.len(), 3);
		assert_eq!(charged.len(), 3);
	}

	/// Composition order: the reservation comes off the capacity FIRST, and
	/// the watermarks are applied to what is left. Pinned by a run that the
	/// unreserved budget would not even have triggered, yet which drains to
	/// `low_bytes(capacity - reserved)`.
	#[test]
	fn shared_overhead_composes_underneath_the_watermarks() {
		// fifo_capacity 15_000 out of a 35_000-byte fast tier => a raw 20_000
		// for the main queue; 120 tracked keys at 100 bytes of shared
		// metadata each reserve 12_000 of that, leaving a round 8_000.
		let mut stack = TwoQFastAdmissionHybridStack::new(0.5, 30_000, 35_000)
			.with_shared_overhead(100);

		let size: CacheSize = 100;
		let raw_main: CacheSize = 20_000;

		for key in 1..=120 {
			stack.insert(key, size as ObjectSize);
		}

		assert_eq!(stack.len(), 120);
		// 12_000 of FIFO bytes against a 15_000 cap: no eviction pressure,
		// so nothing here is confounded by the FIFO queue's own budget.
		assert!(!stack.needs_capacity_eviction());

		let effective = stack.effective_main_fast_capacity();

		assert_eq!(effective, 8_000);

		let high = watermarks::high_bytes(effective);
		let low = watermarks::low_bytes(effective);
		let filled = high / size;

		assert!(filled >= 1, "watermark config leaves no room for this test");
		assert!(filled + 1 <= 120, "watermark config leaves no room for this test");

		// Promotions never change `entries.len()`, so `effective` is constant
		// for the whole run.
		for key in 1..=filled {
			stack.update(key);
		}

		assert_eq!(stack.fast_used, filled * size);
		assert_eq!(stack.effective_main_fast_capacity(), effective);
		assert!(drain(&mut stack).is_empty(), "at or below the high watermark must not trigger");

		// One more promotion tips it past the high watermark OF THE RESERVED
		// budget.
		stack.update(filled + 1);

		let migrations = drain(&mut stack);

		assert!(!migrations.is_empty(), "crossing the reserved high watermark must trigger a pass");

		// It drained to `low_bytes(capacity - reserved)`, and stopped as soon
		// as it got under rather than emptying the segment.
		assert_eq!(stack.fast_used, low / size * size);
		assert!(stack.fast_used <= low);
		assert!(stack.fast_used + size > low);

		// ...and NOT to `low_bytes(capacity)`. Against the unreserved raw
		// budget this run sits below even the HIGH watermark, so no pass
		// would have run at all -- the reservation is the only reason any of
		// these demotions happened, and the target it drained to is strictly
		// tighter than the unreserved low watermark.
		assert!(
			(filled + 1) * size <= watermarks::high_bytes(raw_main),
			"the unreserved budget must not have triggered a pass here",
		);
		// The two candidate drain targets are genuinely different numbers, so
		// landing on the reserved one is a real distinction rather than a
		// coincidence of rounding.
		assert!(
			low / size * size < watermarks::low_bytes(raw_main) / size * size,
			"watermark config leaves no room for this test",
		);

		// Total DRAM -- both fast structures' values plus every tracked key's
		// shared metadata -- is within the configured fast tier, which is the
		// whole point of the reservation.
		assert!(stack.fast_bytes_used() + stack.len() as CacheSize * 100 <= 35_000);
	}

	/// Every byte and every object is still accounted for exactly once after
	/// a reservation-triggered multi-object drain, and the reservation itself
	/// is unchanged by the pass (demotion untracks nothing).
	#[test]
	fn counters_stay_consistent_after_a_reserved_pass() {
		let mut stack = TwoQFastAdmissionHybridStack::new(0.5, 30_000, 35_000)
			.with_shared_overhead(100);

		let size: CacheSize = 100;

		for key in 1..=120 {
			stack.insert(key, size as ObjectSize);
		}

		let effective = stack.effective_main_fast_capacity();

		assert_eq!(effective, 8_000);

		let promoted = watermarks::high_bytes(effective) / size + 1;

		assert!(promoted >= 2, "watermark config leaves no room for this test");
		assert!(promoted <= 120, "watermark config leaves no room for this test");

		for key in 1..=promoted {
			stack.update(key);
		}

		let migrations = drain(&mut stack);
		let demoted = migrations.len() as CacheSize;

		assert!(demoted >= 1, "the pass must have run");
		assert!(migrations.iter().all(|(_, tier)| *tier == Tier::Slow));
		assert_eq!(migrations, (1..=demoted).map(|key| (key, Tier::Slow)).collect::<Vec<_>>());

		// The keys never promoted are still FIFO-resident, untouched by the
		// pass.
		let still_fifo = 120 - promoted;

		// Bytes: every promoted key is main-fast or main-slow; the rest are
		// neither.
		assert_eq!(stack.fast_used, (promoted - demoted) * size);
		assert_eq!(stack.slow_used, demoted * size);
		assert_eq!(stack.fifo_used, still_fifo * size);
		assert_eq!(stack.fast_used + stack.slow_used, promoted * size);

		// Counts: `fast_count` covers main-fast only, `main_count` every
		// promoted key, `len()` every tracked key.
		assert_eq!(stack.fast_count as CacheSize, promoted - demoted);
		assert_eq!(stack.main_count as CacheSize, promoted);
		assert_eq!(stack.len(), 120);

		// Reported gauges add the FIFO queue back into the fast side.
		assert_eq!(stack.fast_bytes_used(), (promoted - demoted) * size + still_fifo * size);
		assert_eq!(stack.slow_bytes_used(), demoted * size);
		assert_eq!(stack.fast_object_count() as CacheSize, promoted - demoted + still_fifo);
		assert_eq!(stack.slow_object_count() as CacheSize, demoted);

		// The per-key tier tags agree with those counters.
		let fast_tagged = (1..=promoted).filter(|key| stack.tier_of(*key) == Some(Tier::Fast)).count();
		let slow_tagged = (1..=promoted).filter(|key| stack.tier_of(*key) == Some(Tier::Slow)).count();

		assert_eq!(fast_tagged as CacheSize, promoted - demoted);
		assert_eq!(slow_tagged as CacheSize, demoted);

		// The boundary is the LRU-most still-fast key, or `None` if the pass
		// drained the segment completely.
		assert_eq!(
			stack.main_boundary,
			if stack.fast_count > 0 { Some(demoted + 1) } else { None },
		);

		// Demotion untracks nothing, so the reservation -- and the effective
		// budget it produces -- is exactly where it started.
		assert_eq!(stack.effective_main_fast_capacity(), effective);

		// And total DRAM, metadata included, is still within the fast tier.
		assert!(stack.fast_bytes_used() + stack.len() as CacheSize * 100 <= 35_000);
	}

	/// A reservation big enough to swallow the main queue's whole share
	/// saturates to a zero-byte fast segment and demotes everything -- but it
	/// must never turn into eviction pressure. `needs_capacity_eviction` is
	/// the FIFO queue's own byte cap and nothing else; the reservation is
	/// deliberately not charged against it (see the module doc).
	#[test]
	fn shared_overhead_exceeding_the_main_budget_demotes_all_but_never_evicts() {
		// fifo_capacity 100 out of a 500-byte fast tier leaves the main queue
		// 400; one tracked key's 1_000-byte reservation already exceeds that.
		let mut stack = TwoQFastAdmissionHybridStack::new(0.1, 1_000, 500)
			.with_shared_overhead(1_000);

		stack.insert(1, 10);

		assert_eq!(stack.effective_main_fast_capacity(), 0);

		// Admission is still Fast: the FIFO queue's own budget is untouched
		// by the reservation, which is exactly what keeps it from evicting.
		assert_eq!(stack.tier_of(1), Some(Tier::Fast));
		assert!(drain(&mut stack).is_empty());
		assert!(!stack.needs_capacity_eviction());

		// Promotion lands in a zero-byte budget, so it demotes straight back
		// out -- a genuine DRAM->PMEM move, hence a real migration.
		stack.update(1);

		assert_eq!(stack.tier_of(1), Some(Tier::Slow));
		assert_eq!(drain(&mut stack), vec![(1, Tier::Slow)]);
		assert_eq!(stack.fast_used, 0);

		// Demotion is the only response: the key is still tracked and there
		// is still no eviction pressure.
		assert_eq!(stack.len(), 1);
		assert!(!stack.needs_capacity_eviction());
	}
}
