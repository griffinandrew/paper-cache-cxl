/*
 * Copyright (c) Kia Shakiba
 *
 * This source code is licensed under the GNU AGPLv3 license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! `S3FifoGhostLazyDemotionFastAdmissionHybridStack` —
//! `S3FifoGhostLazyDemotionHybridStack` with one change: the one-access
//! queue now lives in the FAST tier instead of the slow tier. For
//! `PaperPolicy::S3FifoGhostLazyDemotionFastAdmissionHybrid`.
//!
//! Identical to `S3FifoGhostLazyDemotionHybridStack` in every other respect
//! (ghost queue lifecycle, the demotion-time reference-bit reprieve, the
//! eviction-time second chance, the "contiguous front run" invariant) — see
//! that stack's module doc, and `S3FifoGhostHybridStack`/`S3FifoHybridStack`
//! beneath it, for the full picture.
//!
//! ## Motivation: every admission was a synchronous PMEM write
//!
//! In the base design, admission is unconditional to the *slow* tier — the
//! literal paper rule ("every new object is placed in the slow tier"). At
//! the `PaperCache::set()` API layer this means every single admission
//! (and every one-access-queue re-admission of a ghost-recycled key)
//! synchronously builds `TieredBuffer::new_slow`, i.e. a real PMEM/UMF
//! allocation on the calling thread, before the object is even in the
//! cache. Reported by the user as a real cost worth trying to avoid: this
//! variant places the one-access queue's bytes in the FAST tier instead,
//! so admission becomes a cheap DRAM write (`TieredBuffer::new_fast`) —
//! the same kind of change `lru_hybrid_cache`'s admission already gets for
//! free by always landing fast.
//!
//! Only the one-access queue moves. The main queue keeps exactly the same
//! fast/slow segmentation, demotion-time reprieve, and eviction-time
//! second chance as `S3FifoGhostLazyDemotionHybridStack` — a key still has
//! to prove itself with a second access to earn a spot in the "real",
//! frequency-durable part of the cache; the only thing that changed is
//! which physical allocator backs its bytes *while on probation* in the
//! one-access queue.
//!
//! ## Accounting: the one-access queue now competes for the SAME DRAM budget
//!
//! This is the part that has to be handled deliberately, not just
//! relabeled. In the base design, `one_access_capacity` (`one_access_ratio
//! * max_size`) and `fast_capacity` (`fast_tier_size`) are two completely
//! independent budgets — one governs a slow/PMEM queue, the other governs
//! the main queue's fast/DRAM portion. Now that the one-access queue is
//! *also* DRAM, both budgets draw from the same physical pool, and adding
//! them naively (letting each treat the full `fast_capacity` as its own)
//! would silently let real DRAM usage grow to `fast_capacity +
//! one_access_capacity` instead of the configured `fast_capacity`.
//!
//! Fixed by treating `one_access_capacity` as a fixed reservation carved
//! out of `fast_capacity` first — `effective_main_fast_capacity()` =
//! `fast_capacity.saturating_sub(one_access_capacity)` — and having
//! `settle_fast_tier` (the main queue's own demotion trigger) check
//! against that reduced number instead of raw `fast_capacity`. The
//! one-access queue keeps its own byte cap (`needs_capacity_eviction`),
//! which was always `one_access_used > one_access_capacity`, independent
//! of tier; it is now taken against `effective_one_access_capacity()` --
//! that same cap minus this segment's proportional share of the
//! shared-metadata reservation described in the next section. With no
//! reservation wired in, the two are identical byte for byte.
//! The net result: `fast_used (main) +
//! one_access_used ≤ fast_capacity` holds by construction (modulo the same
//! kind of transient overshoot every other stack in this crate already
//! tolerates between eviction-loop passes), so the configured fast-tier
//! size remains a real, honored bound on total DRAM, not just on the main
//! queue's share of it. The shared high/low watermarks (see the
//! `watermarks` module doc) are layered *on top of* that effective number
//! rather than replacing it, so the bound is if anything tighter: a
//! demotion pass now fires at `high * effective_main_fast_capacity()` and
//! drains to `low * effective_main_fast_capacity()`.
//! `resize()` (triggered when `max_size` changes,
//! which rescales `one_access_capacity`) proactively re-runs
//! `settle_fast_tier` for the same reason `resize_fast_tier` already does
//! -- growing `one_access_capacity` shrinks the room left for the main
//! queue's fast segment, and that has to be caught immediately rather than
//! waiting for the next unrelated `insert`/`update` to notice.
//!
//! A degenerate but legitimate consequence of this: if `one_access_ratio *
//! max_size` alone meets or exceeds `fast_capacity`, the main queue's fast
//! segment gets zero (or negative, saturated to zero) room, and every
//! promotion out of the one-access queue immediately self-demotes back to
//! slow. That's correct accounting given the configuration, not a bug —
//! see `zero_effective_main_capacity_demotes_every_promotion_immediately`
//! below, mirroring the equivalent documented behavior in
//! `lru_sized_hybrid_cache`.
//!
//! ## Shared-structure DRAM reservation
//!
//! On top of the split above, the fast-tier budget also has to cover the
//! DRAM that the shared object hashtable and this stack's own
//! eviction-stack bookkeeping occupy: neither is counted in `fast_used`/
//! `one_access_used`, and both are real DRAM. Same mechanism as
//! `LruHybridStack` -- a per-tracked-key `shared_overhead` (wired in by
//! `init_policy_stack` from
//! `crate::object::overhead::get_hybrid_dram_shared_overhead`, `0`
//! otherwise), multiplied by the tracked-key count and subtracted from the
//! fast-tier budget *before* the watermarks are applied to what is left.
//!
//! A tracked key occupies exactly ONE `HashList` node -- in
//! `one_access_queue` *or* `main_queue`, never both (`insert` pushes to
//! `one_access_queue`; `promote_from_one_access` removes from it *before*
//! pushing to `main_queue`; `admit_via_ghost_hit` only ever pushes to
//! `main_queue`; `give_second_chance` and `settle_fast_tier` only
//! `move_front`/retag *within* `main_queue`) -- plus exactly one `entries`
//! slot. Keys of *both* tiers are charged: those two structures are DRAM
//! wherever the object's own data happens to live.
//!
//! Unlike `S3FifoGhostHybridStack` (whose one-access queue is slow-tier,
//! leaving it a single fast-capacity segment), this variant has *two*
//! independently-capacitied fast segments -- `one_access_capacity` and the
//! main queue's `fast_capacity - one_access_capacity`. The reservation is
//! therefore split proportionally between them (`reserved_shares`,
//! following `LruSizedHybridStack`) rather than charged in full to each
//! (which would over-reserve by a factor of two) or in full to just one
//! (which would leave the other segment paying nothing toward metadata it
//! is equally responsible for). With the split,
//! `effective_one_access_capacity() + effective_main_fast_capacity() +
//! reserved_overhead() == fast_capacity` exactly, so the configured
//! fast-tier size bounds total DRAM -- values *and* shared metadata --
//! rather than values alone.
//!
//! The two segments enforce their budgets differently, and that carries
//! over unchanged: the main queue's share is enforced by *demotion*
//! (`settle_fast_tier`), the one-access queue's by *eviction*
//! (`needs_capacity_eviction` -- that queue has never had a slow half to
//! demote into). So its share of the reservation shortens probation
//! slightly rather than demoting anything, and in the degenerate case where
//! the share swallows the whole one-access cap, every admission is evicted
//! straight into `ghost` on the next capacity pass. Same class of
//! documented, configuration-driven degeneracy as
//! `zero_effective_main_capacity_demotes_every_promotion_immediately` above.
//!
//! `ghost` is charged *separately*, exactly as in `S3FifoGhostHybridStack`:
//! its entries are bare keys for objects that are no longer in the cache and
//! no longer in `entries` at all, so that cost scales with the ghost queue's
//! own length (`ghost.len()`, capped at `main_count` by `trim_ghost` on
//! genuine main-queue evictions), not with the tracked-key count -- a
//! per-tracked-key constant cannot model it, and a ghost key has no
//! object-hashtable slot to charge either.
//!
//! That per-ghost-entry price is the shared
//! `crate::object::overhead::GHOST_ENTRY_DRAM_OVERHEAD`, read directly by
//! `reserved_overhead` rather than injected through a builder. Every
//! ghost-keeping stack in this module holds its ghost queue in the same
//! `HashList<HashedKey>`, so the figure is not something a caller could
//! sensibly vary -- and a caller-supplied field defaulting to `0` meant a
//! construction site that forgot to wire it silently reserved nothing at all
//! for a queue that really does occupy DRAM. Only `shared_overhead` is still
//! caller-supplied and still defaults to `0`, so a unit test constructing the
//! stack directly sees the pure value-byte budget right up until it puts
//! something in `ghost`.
//!
//! ## A second optimization this unlocks: no more redundant Fast→Fast copies
//!
//! `promote_from_one_access` and `admit_via_ghost_hit` no longer need to
//! push a `(key, Tier::Fast)` migration after a successful promotion. In
//! the base design that push was load-bearing: the API layer had just
//! built the key's bytes as Slow (per the always-Slow admission rule), so
//! the migration was the ONLY thing that ever physically moved them to
//! Fast DRAM. Here, admission (see `S3FifoGhostLazyDemotionFastAdmissionHybridPolicy::admission_tier`
//! in this feature's `mod.rs`) already builds every brand-new key's bytes
//! as Fast unconditionally -- including ghost hits, which are
//! indistinguishable from any other fresh `set()` at the API layer -- so a
//! one-access-queue entry's buffer is *already* physically Fast for its
//! entire lifetime in that queue, before it's ever promoted. Pushing a
//! Fast migration on a successful promotion would just make
//! `apply_tier_migrations` copy already-correct DRAM bytes into a fresh
//! DRAM buffer for no reason -- a real, avoidable cost on every single
//! second-access promotion, which is exactly the class of cost this whole
//! variant exists to cut. The one case that still needs a real migration
//! -- a promotion out of the main queue's SLOW portion via
//! `give_second_chance` -- is untouched: that key's bytes genuinely are in
//! PMEM at that point (it was really demoted there earlier), so the
//! migration is still doing real, necessary work. The demotion-time
//! reprieve and `settle_fast_tier`'s real demotions are also untouched --
//! neither of those ever needed a Fast-migration push to begin with.

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
	object::{ObjectSize, overhead::GHOST_ENTRY_DRAM_OVERHEAD},
	worker::policy::policy_stack::{PolicyStack, Tier, watermarks},
};

/// Which live queue a key currently belongs to.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Queue {
	OneAccess,
	Main,
}

/// Combined per-key bookkeeping. `tier`/`accessed` are only meaningful
/// while `queue == Main`.
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

pub struct S3FifoGhostLazyDemotionFastAdmissionHybridStack {
	one_access_queue: QueueList,
	main_queue: QueueList,
	ghost: QueueList,

	entries: EntryMap,

	one_access_ratio: f64,
	one_access_capacity: CacheSize,
	one_access_used: CacheSize,

	/// The configured total fast-tier (DRAM) budget. Shared between the
	/// one-access queue and the main queue's fast segment -- see the
	/// module doc's "Accounting" section. The main queue's own trigger
	/// checks `effective_main_fast_capacity()`, not this field directly.
	fast_capacity: CacheSize,
	fast_used: CacheSize,
	slow_used: CacheSize,

	/// Approximate per-object DRAM cost of the shared structures (object
	/// hashtable + this stack's eviction-stack bookkeeping) that hold an
	/// entry for every *tracked* key of both tiers. Reserved out of
	/// `fast_capacity` -- split proportionally between the two fast
	/// segments by `reserved_shares` -- see the module doc's
	/// "Shared-structure DRAM reservation" section. `0` unless set via
	/// `with_shared_overhead` (so unit tests exercising the pure
	/// value-budget behavior are unaffected).
	shared_overhead: CacheSize,

	fast_count: usize,

	/// Number of keys currently in the `Main` queue (Fast or Slow). Also
	/// used as the ghost list's size cap reference.
	main_count: usize,

	main_boundary: Option<HashedKey>,

	migrations: Vec<(HashedKey, Tier)>,
}

impl S3FifoGhostLazyDemotionFastAdmissionHybridStack {
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

		S3FifoGhostLazyDemotionFastAdmissionHybridStack {
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

	/// Total DRAM currently reserved out of `fast_capacity` for metadata:
	/// `tracked key count × shared_overhead` (every key in `entries` — i.e.
	/// both tiers, and both `one_access_queue` and `main_queue` residents,
	/// since a key holds exactly one list node plus one `entries` slot
	/// wherever its data lives) `+ ghost length ×
	/// GHOST_ENTRY_DRAM_OVERHEAD`.
	///
	/// The two terms are wholly independent. The ghost one is neither opt-in
	/// nor caller-configurable: it reads
	/// `crate::object::overhead::GHOST_ENTRY_DRAM_OVERHEAD` directly (`44`
	/// for the `HashList<HashedKey>` node every ghost-keeping stack uses;
	/// `0` under `eviction_stacks_pmem`, where that list is not DRAM at all),
	/// so a ghost entry is charged whether or not `with_shared_overhead` was
	/// ever called. It cannot be folded into `shared_overhead` and must never
	/// carry `HASHTABLE_ENTRY_OVERHEAD`, because a ghost names a key that is
	/// no longer in the cache: it has no `entries` row and no object-hashtable
	/// slot for a per-tracked-key constant to ride along on.
	///
	/// Loop-invariant within a `settle_fast_tier` pass: a demotion only
	/// retags an entry's `tier`, it never changes whether a key is tracked
	/// nor the length of `ghost`.
	fn reserved_overhead(&self) -> CacheSize {
		self.entries.len() as CacheSize * self.shared_overhead
			+ self.ghost.len() as CacheSize * (GHOST_ENTRY_DRAM_OVERHEAD as CacheSize)
	}

	/// The main queue's fast-segment budget *before* the shared-metadata
	/// reservation — `fast_capacity` minus the one-access queue's fixed
	/// carve-out. Kept separate from `effective_main_fast_capacity` so
	/// `reserved_shares` has a reservation-free capacity to proportion
	/// against (using the effective one would be circular).
	fn raw_main_fast_capacity(&self) -> CacheSize {
		self.fast_capacity.saturating_sub(self.one_access_capacity)
	}

	/// Splits `reserved_overhead()` proportionally between this stack's two
	/// independently-capacitied FAST segments — the one-access queue and the
	/// main queue's fast portion — by their respective capacities, returned
	/// as `(one_access_share, main_share)`. The shared structures hold an
	/// entry for everything tracked, not just one segment's residents, so
	/// neither segment may be charged the full amount; splitting keeps
	/// `effective_one_access_capacity() + effective_main_fast_capacity() +
	/// reserved_overhead() == fast_capacity`. Same construction as
	/// `LruSizedHybridStack::reserved_shares` (`u128` intermediate so the
	/// product cannot overflow, remainder handed to the main segment so the
	/// two shares always re-sum exactly). `(0, 0)` if both capacities are
	/// zero (nothing to proportion against).
	fn reserved_shares(&self) -> (CacheSize, CacheSize) {
		let reserved = self.reserved_overhead();

		let one_access_capacity = self.one_access_capacity;
		let main_capacity = self.raw_main_fast_capacity();
		let total_capacity = one_access_capacity + main_capacity;

		if total_capacity == 0 {
			return (0, 0);
		}

		let one_access_share =
			((reserved as u128 * one_access_capacity as u128) / total_capacity as u128) as CacheSize;
		let main_share = reserved.saturating_sub(one_access_share);

		(one_access_share, main_share)
	}

	/// The one-access queue's own byte cap after giving up its share of the
	/// shared-metadata reservation. `needs_capacity_eviction` checks against
	/// this rather than raw `one_access_capacity`; with no reservation wired
	/// in they are the same number.
	fn effective_one_access_capacity(&self) -> CacheSize {
		self.one_access_capacity.saturating_sub(self.reserved_shares().0)
	}

	/// The budget actually available to the main queue's fast segment: raw
	/// `fast_capacity`, minus the one-access queue's fixed carve-out (the
	/// module doc's "Accounting" section), minus this segment's share of the
	/// shared-metadata reservation (its "Shared-structure DRAM reservation"
	/// section). This is the value `settle_fast_tier` applies the watermarks
	/// to — they sit on top of it, never in place of it.
	fn effective_main_fast_capacity(&self) -> CacheSize {
		self.raw_main_fast_capacity().saturating_sub(self.reserved_shares().1)
	}

	pub fn tier_of(&self, key: HashedKey) -> Option<Tier> {
		let entry = self.entries.get(&key)?;

		match entry.queue {
			// The one-access queue is DRAM-resident in this variant --
			// see the module doc's "Motivation" section. This is the one
			// line that differs from `S3FifoGhostLazyDemotionHybridStack`'s
			// `tier_of`.
			Queue::OneAccess => Some(Tier::Fast),
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

	/// Moves a re-accessed one-access-queue key into the main queue at
	/// `Tier::Fast`. Unlike the base design, this never needs to push a
	/// migration for the promotion itself -- the key's bytes are already
	/// physically Fast (see the module doc's "no more redundant Fast→Fast
	/// copies" section) -- only `settle_fast_tier`'s own demotion push (if
	/// this promotion itself immediately overflows the budget) can produce
	/// a migration here, and that's handled inside `settle_fast_tier`.
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
	}

	/// Admits a brand-new key directly into `main_queue` at `Tier::Fast` —
	/// the ghost-hit path. Structurally identical to
	/// `promote_from_one_access` minus the "remove from `one_access_queue`"
	/// step. Same no-redundant-migration reasoning applies: the API layer
	/// already built this key's bytes as Fast (admission is unconditional
	/// Fast in this variant), so there's nothing to migrate unless
	/// `settle_fast_tier` demotes it right back out, which pushes its own
	/// migration.
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
	}

	/// The eviction-time second chance — completely unchanged from
	/// `S3FifoGhostLazyDemotionHybridStack`. This is the one promotion path
	/// that STILL needs its migration push: a key reaching this method by
	/// definition currently has `tier == Some(Tier::Slow)` in the common
	/// case (it was genuinely demoted to PMEM earlier), so moving it back
	/// to Fast is a real physical move, not a relabeling.
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

	/// Demotes key(s) anchoring `main_boundary` once `fast_used` crosses the
	/// fast tier's HIGH watermark, then keeps going until it is back at or
	/// below the LOW watermark -- reference-bit gated, exactly like
	/// `S3FifoGhostLazyDemotionHybridStack`'s version.
	///
	/// The effective ceiling is `effective_main_fast_capacity()`:
	/// `fast_capacity`, minus the one-access queue's fixed carve-out (the
	/// module doc's "Accounting" section — still what distinguishes this
	/// stack from its predecessors), minus this segment's proportional share
	/// of the shared-structure DRAM reservation (that doc's
	/// "Shared-structure DRAM reservation" section), which makes the
	/// fast-tier budget bound total DRAM rather than just fast-tier values
	/// and saturates the ceiling to 0 when the metadata alone meets or
	/// exceeds what the carve-out left behind.
	///
	/// The `watermarks` helpers are applied *on top of* that fully reduced
	/// number, never in place of any part of it: a pass fires at
	/// `high_bytes(effective)` and drains to `low_bytes(effective)`, so the
	/// drain target is `low_bytes(capacity - reserved)`, not
	/// `low_bytes(capacity)`. `effective_capacity` is read once, before the
	/// loop: a demotion only retags an entry, so neither the tracked-key
	/// count nor `ghost.len()` — and hence neither the reservation nor the
	/// target — can move underneath the pass.
	///
	/// Previously this drained to exactly `effective_main_fast_capacity()`,
	/// which pinned the tier at 100% utilisation and made essentially every
	/// promotion demote exactly one object (see the `watermarks` module
	/// doc). Setting both `FAST_TIER_HIGH_WATERMARK` and
	/// `FAST_TIER_LOW_WATERMARK` to `1.0` restores that behaviour
	/// byte-for-byte.
	///
	/// Per-demotion bookkeeping is deliberately untouched: each demoted
	/// object still retags its entry, still moves `fast_used`/`fast_count`/
	/// `slow_used` by its own size, still walks `main_boundary` one step
	/// toward the front, and still emits exactly one `Tier::Slow` migration
	/// -- and the demotion-time reference-bit reprieve still gets first
	/// refusal on every candidate it reaches.
	fn settle_fast_tier(&mut self) {
		let effective_capacity = self.effective_main_fast_capacity();

		if self.fast_used <= watermarks::high_bytes(effective_capacity) {
			return;
		}

		let drain_target = watermarks::low_bytes(effective_capacity);

		while self.fast_used > drain_target {
			let Some(candidate) = self.main_boundary else { break };

			let accessed = self.entries.get(&candidate).map(|entry| entry.accessed).unwrap_or(false);

			if accessed {
				// Reprieve: fresh start at the front instead of demotion.
				let new_boundary = self.main_queue.before(&candidate).copied();

				self.main_queue.move_front(&candidate);
				self.main_boundary = new_boundary;

				if let Some(entry) = self.entries.get_mut(&candidate) {
					entry.accessed = false;
				}

				continue;
			}

			let size = self.entries.get(&candidate).map(|entry| entry.size).unwrap_or(0) as CacheSize;
			let new_boundary = self.main_queue.before(&candidate).copied();

			if let Some(entry) = self.entries.get_mut(&candidate) {
				entry.tier = Some(Tier::Slow);
			}

			self.fast_used = self.fast_used.saturating_sub(size);
			self.fast_count = self.fast_count.saturating_sub(1);
			self.slow_used += size;
			self.main_boundary = new_boundary;

			self.migrations.push((candidate, Tier::Slow));
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
	/// genuine main-queue eviction, never from a second chance, a
	/// demotion-time reprieve, or `evict_one_access_tail` (which is what
	/// populates `ghost`).
	fn trim_ghost(&mut self) {
		while self.ghost.len() > self.main_count {
			self.ghost.pop_back();
		}
	}
}

impl PolicyStack for S3FifoGhostLazyDemotionFastAdmissionHybridStack {
	fn is_policy(&self, policy: &PaperPolicy) -> bool {
		matches!(policy, PaperPolicy::S3FifoGhostLazyDemotionFastAdmissionHybrid(ratio) if *ratio == self.one_access_ratio)
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
		// Unconditional and first -- see `S3FifoGhostHybridStack::remove`'s
		// doc for why.
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

		// Growing `one_access_capacity` shrinks the room left for the main
		// queue's fast segment (see the module doc's "Accounting"
		// section) -- proactively re-check rather than waiting for the
		// next unrelated insert/update to notice, same reasoning as
		// `resize_fast_tier` already has.
		self.settle_fast_tier();
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
		// Total DRAM: main queue's fast segment + the one-access queue,
		// both physically Fast in this variant.
		self.fast_used + self.one_access_used
	}

	fn slow_bytes_used(&self) -> CacheSize {
		// The one-access queue no longer touches Slow/PMEM at all -- unlike
		// `S3FifoGhostLazyDemotionHybridStack`, this is just the main
		// queue's slow segment.
		self.slow_used
	}

	fn fast_object_count(&self) -> usize {
		self.fast_count + self.one_access_queue.len()
	}

	fn slow_object_count(&self) -> usize {
		self.main_count - self.fast_count
	}

	fn needs_capacity_eviction(&self) -> bool {
		// Against `effective_one_access_capacity()`, i.e. this segment's own
		// cap minus its proportional share of the shared-metadata
		// reservation -- see the module doc's "Shared-structure DRAM
		// reservation" section. Identical to the raw cap whenever no
		// reservation is wired in.
		self.one_access_used > self.effective_one_access_capacity()
	}
}

#[cfg(test)]
mod tests {
	use super::*;

	/// The per-ghost-entry DRAM price, straight from the shared constant --
	/// `44`, or `0` under `eviction_stacks_pmem`, where the ghost list is not
	/// DRAM at all. The reservation arithmetic below is stated against this
	/// symbol rather than a literal so it stays correct under either gating.
	const GHOST: CacheSize = GHOST_ENTRY_DRAM_OVERHEAD as CacheSize;

	fn drain(stack: &mut S3FifoGhostLazyDemotionFastAdmissionHybridStack) -> Vec<(HashedKey, Tier)> {
		stack.drain_tier_migrations()
	}

	/// `insert` + `update` -- the insert-into-`one_access_queue`-then-promote
	/// pairing every fast-tier test in this module already uses, since a fresh
	/// (non-ghost) key never lands straight in `main_queue`'s fast segment.
	/// Leaves `one_access_queue` empty, so `fast_bytes_used()` /
	/// `fast_object_count()` report the main queue's fast segment alone even
	/// though this variant folds the one-access queue into both.
	fn promote(stack: &mut S3FifoGhostLazyDemotionFastAdmissionHybridStack, key: HashedKey, size: ObjectSize) {
		stack.insert(key, size);
		stack.update(key);
	}

	/// Smallest *effective* main-fast budget whose low watermark still leaves
	/// room for `bytes`. Lets the fast-tier tests state their expectations in
	/// whole objects instead of hard-coded byte thresholds, so they hold at
	/// whatever `FAST_TIER_HIGH_WATERMARK`/`FAST_TIER_LOW_WATERMARK` pair is
	/// configured rather than only at the 0.95/0.75 defaults. The `while` loop
	/// absorbs the truncation in `watermarks::low_bytes`' `as u64` cast, which
	/// a bare `ceil()` on its own can land a byte short of.
	fn capacity_holding(bytes: CacheSize) -> CacheSize {
		let mut capacity = (bytes as f64 / watermarks::low()).ceil() as CacheSize;

		while watermarks::low_bytes(capacity) < bytes {
			capacity += 1;
		}

		capacity
	}

	/// The watermark tests below all share this shape: a caller-chosen
	/// *effective* main-fast budget sitting behind a non-zero one-access
	/// reservation (0.01 * 100_000 = 1_000), so they also pin down that the
	/// watermarks are applied on top of `effective_main_fast_capacity()`
	/// rather than raw `fast_capacity`.
	const WATERMARK_RATIO: f64 = 0.01;
	const WATERMARK_MAX_SIZE: CacheSize = 100_000;
	const WATERMARK_RESERVED: CacheSize = 1_000;

	fn watermark_stack(effective: CacheSize) -> S3FifoGhostLazyDemotionFastAdmissionHybridStack {
		let stack = S3FifoGhostLazyDemotionFastAdmissionHybridStack::new(
			WATERMARK_RATIO, WATERMARK_MAX_SIZE, effective + WATERMARK_RESERVED,
		);

		assert_eq!(stack.effective_main_fast_capacity(), effective);

		stack
	}

	#[test]
	fn admission_always_lands_in_one_access_queue_fast() {
		let mut stack = S3FifoGhostLazyDemotionFastAdmissionHybridStack::new(1.0, 1_000, 1_000);

		stack.insert(1, 10);
		stack.insert(2, 10);

		assert_eq!(stack.tier_of(1), Some(Tier::Fast));
		assert_eq!(stack.tier_of(2), Some(Tier::Fast));
		assert_eq!(drain(&mut stack), Vec::new());
	}

	#[test]
	fn reaccessing_a_one_access_key_promotes_it_to_main_without_a_migration() {
		// one_access_ratio=0.0 -- unlike the base design's equivalent test,
		// ratio=1.0 here would reserve the *entire* fast_capacity for the
		// one-access queue (see `zero_effective_main_capacity_...` below),
		// leaving zero room for the promoted key to actually stay fast.
		let mut stack = S3FifoGhostLazyDemotionFastAdmissionHybridStack::new(0.0, 1_000, 1_000);

		stack.insert(1, 10);
		drain(&mut stack);

		stack.update(1);

		// No migration: the key's bytes were already Fast the whole time
		// (see the module doc's "no more redundant Fast→Fast copies"
		// section) -- this is the key behavioral difference from
		// `S3FifoGhostLazyDemotionHybridStack`, whose equivalent test
		// asserts the opposite (a real migration IS produced there).
		assert_eq!(drain(&mut stack), Vec::new());
		assert_eq!(stack.tier_of(1), Some(Tier::Fast));
	}

	#[test]
	fn a_key_aging_out_without_reaccess_becomes_a_ghost_entry() {
		let mut stack = S3FifoGhostLazyDemotionFastAdmissionHybridStack::new(1.0, 1_000, 1_000);

		stack.insert(1, 10);
		drain(&mut stack);

		assert_eq!(stack.evict_one(), Some(1));
		assert!(stack.is_ghost(1));
	}

	#[test]
	fn ghost_hit_on_readmission_lands_in_fast_tier_without_a_migration() {
		// one_access_ratio=0.0 -- see the comment on
		// `reaccessing_a_one_access_key_promotes_it_to_main_without_a_migration`
		// above for why ratio=1.0 wouldn't leave room for the promoted key
		// to stay fast here.
		let mut stack = S3FifoGhostLazyDemotionFastAdmissionHybridStack::new(0.0, 1_000, 1_000);

		stack.insert(1, 10);
		drain(&mut stack);
		stack.evict_one();
		assert!(stack.is_ghost(1));

		stack.insert(1, 10);

		// No migration here either, same reasoning as the reaccess test
		// above -- the API layer already built this key's bytes as Fast.
		assert_eq!(drain(&mut stack), Vec::new());
		assert_eq!(stack.tier_of(1), Some(Tier::Fast));
	}

	// ── the signature accounting mechanic: shared DRAM budget ──────────────

	#[test]
	fn one_access_capacity_is_reserved_out_of_the_fast_budget() {
		// one_access_ratio reserves 40 (0.04 * 1_000) of fast_capacity for
		// the one-access queue; the rest is the main queue's effective
		// budget, sized to hold exactly one of the two 50-byte keys below
		// its low watermark so a triggered pass demotes exactly one (see
		// `capacity_holding`). Was a hard-coded 100/60 pair: correct back
		// when a pass drained to the ceiling, but under the watermarks a
		// 60-byte effective budget triggers at 57 and drains to 45, which
		// would take both keys.
		let effective = capacity_holding(50);
		let mut stack = S3FifoGhostLazyDemotionFastAdmissionHybridStack::new(0.04, 1_000, effective + 40);
		assert_eq!(stack.effective_main_fast_capacity(), effective);

		// Promote two 50-byte keys into the main queue -- fast_used (main
		// only) reaches 100, comfortably within raw fast_capacity's own high
		// watermark, but that's over the *effective* budget's once the
		// one-access reservation is accounted for, so the older one must
		// be demoted.
		stack.insert(1, 50);
		stack.update(1);
		drain(&mut stack);

		stack.insert(2, 50);
		stack.update(2);
		let migrations = drain(&mut stack);

		assert!(migrations.iter().any(|(k, t)| *k == 1 && *t == Tier::Slow));
		assert_eq!(stack.tier_of(1), Some(Tier::Slow));
		assert_eq!(stack.tier_of(2), Some(Tier::Fast));
	}

	#[test]
	fn zero_effective_main_capacity_demotes_every_promotion_immediately() {
		// one_access_ratio alone consumes the entire fast_capacity, so the
		// main queue's fast segment has zero effective room -- every
		// promotion must self-demote right back to slow. Degenerate but
		// correct: documented in the module doc, mirrors
		// lru_sized_hybrid_cache's equivalent zero-capacity precedent.
		let mut stack = S3FifoGhostLazyDemotionFastAdmissionHybridStack::new(1.0, 1_000, 1_000);
		assert_eq!(stack.effective_main_fast_capacity(), 0);

		stack.insert(1, 10);
		stack.update(1);
		let migrations = drain(&mut stack);

		assert_eq!(migrations, vec![(1, Tier::Slow)]);
		assert_eq!(stack.tier_of(1), Some(Tier::Slow));
	}

	#[test]
	fn growing_one_access_capacity_via_resize_immediately_settles_main_fast() {
		// Start with room to spare: fast_capacity=100, one_access_ratio
		// reserves only 10 (0.01 * 1_000), leaving 90 for main-fast.
		let mut stack = S3FifoGhostLazyDemotionFastAdmissionHybridStack::new(0.01, 1_000, 100);

		stack.insert(1, 50);
		stack.update(1);
		drain(&mut stack);
		assert_eq!(stack.tier_of(1), Some(Tier::Fast));

		// Growing max_size to 10_000 grows the one-access reservation to
		// 100 (0.01 * 10_000), consuming the *entire* fast_capacity and
		// leaving 0 for main-fast -- resize() must catch this immediately
		// rather than waiting for an unrelated insert/update.
		stack.resize(10_000);
		let migrations = drain(&mut stack);

		assert_eq!(migrations, vec![(1, Tier::Slow)]);
		assert_eq!(stack.tier_of(1), Some(Tier::Slow));
	}

	// ── the inherited signature mechanic: reprieve at DEMOTION time ────────

	#[test]
	fn an_accessed_fast_boundary_key_is_reprieved_at_demotion_time_not_demoted() {
		// effective_main_fast_capacity = fast_capacity - 0
		// (one_access_ratio=0), sized so a triggered pass drains to a low
		// watermark that still holds one of these two 10-byte objects --
		// i.e. exactly one demotion. (Was a hard-coded 10: correct back when
		// a pass drained to the ceiling, but under the watermarks a 10-byte
		// ceiling triggers at 9 and drains to 7, so even a lone resident
		// object would be demoted straight back out.)
		let mut stack = S3FifoGhostLazyDemotionFastAdmissionHybridStack::new(0.0, 1_000, capacity_holding(10));

		stack.insert(1, 10);
		stack.update(1);
		drain(&mut stack);

		stack.update(1);
		assert_eq!(drain(&mut stack), Vec::new());

		stack.insert(2, 10);
		stack.update(2);
		let migrations = drain(&mut stack);

		assert_eq!(stack.tier_of(1), Some(Tier::Fast), "key 1 should have been reprieved, not demoted");
		assert_eq!(stack.tier_of(2), Some(Tier::Slow), "key 2 should have been demoted in key 1's place");
		assert_eq!(migrations, vec![(2, Tier::Slow)]);
	}

	#[test]
	fn evict_one_gives_an_accessed_slow_key_a_second_chance() {
		// Same watermark-aware sizing as the reprieve test above.
		let mut stack = S3FifoGhostLazyDemotionFastAdmissionHybridStack::new(0.0, 1_000, capacity_holding(10));

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
	}

	#[test]
	fn remove_clears_ghost_entry_too() {
		let mut stack = S3FifoGhostLazyDemotionFastAdmissionHybridStack::new(1.0, 1_000, 1_000);

		stack.insert(1, 10);
		drain(&mut stack);
		stack.evict_one();
		assert!(stack.is_ghost(1));

		stack.remove(1);
		assert!(!stack.is_ghost(1));
	}

	#[test]
	fn remove_and_clear_reset_bookkeeping() {
		let mut stack = S3FifoGhostLazyDemotionFastAdmissionHybridStack::new(1.0, 1_000, 1_000);

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

	#[test]
	fn fast_and_slow_gauges_include_one_access_queue_on_the_fast_side() {
		let mut stack = S3FifoGhostLazyDemotionFastAdmissionHybridStack::new(1.0, 1_000, 1_000);

		stack.insert(1, 10);
		stack.insert(2, 10);
		drain(&mut stack);

		assert_eq!(stack.fast_bytes_used(), 20);
		assert_eq!(stack.slow_bytes_used(), 0);
		assert_eq!(stack.fast_object_count(), 2);
		assert_eq!(stack.slow_object_count(), 0);
	}

	// ── the new mechanic: fast-tier high/low watermarks ────────────────────

	/// (a) The trigger is a strict `>`, so usage sitting right *on* the high
	/// watermark -- the largest usage that is not over it -- must leave the
	/// tier completely alone.
	#[test]
	fn fast_usage_at_the_high_watermark_triggers_no_demotion() {
		let effective: CacheSize = 1_000;
		let high = watermarks::high_bytes(effective);

		let mut stack = watermark_stack(effective);

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
		assert_eq!(stack.slow_object_count(), 0);
	}

	/// (b) One byte past the high watermark -- the smallest possible overshoot
	/// -- must fire a pass, and it must take `main_queue`'s oldest fast key
	/// rather than the key that just arrived.
	#[test]
	fn fast_usage_above_the_high_watermark_triggers_a_demotion_pass() {
		let effective: CacheSize = 1_000;
		let high = watermarks::high_bytes(effective);

		let mut stack = watermark_stack(effective);

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
		assert!(stack.fast_bytes_used() <= watermarks::low_bytes(effective));
	}

	/// (c) A triggered pass keeps going down to the *low* watermark, not just
	/// back under the ceiling. With the defaults this drains 960 -> 750 across
	/// 21 demotions; the pre-watermark drain-to-ceiling loop would have stopped
	/// after a single one, at 950.
	#[test]
	fn a_triggered_pass_drains_all_the_way_to_the_low_watermark() {
		let effective: CacheSize = 1_000;
		let size: ObjectSize = 10;
		let bytes = size as CacheSize;

		let high = watermarks::high_bytes(effective);
		let low = watermarks::low_bytes(effective);

		let mut stack = watermark_stack(effective);

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
		// watermark -- well under `effective_main_fast_capacity()`, which is
		// where the old loop would have left it.
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
		let effective: CacheSize = 1_000;
		let size: ObjectSize = 10;
		let bytes = size as CacheSize;

		let count = watermarks::high_bytes(effective) / bytes + 1;

		let mut stack = watermark_stack(effective);

		for key in 1..=count {
			promote(&mut stack, key, size);
		}

		drain(&mut stack);

		let fast_objects = stack.fast_object_count() as CacheSize;
		let slow_objects = stack.slow_object_count() as CacheSize;

		// Nothing was inserted, evicted or resized mid-pass, so every object is
		// still tracked, still `size` bytes, and still on exactly one side of
		// the fast/slow line. `one_access_queue` is empty here (every key was
		// promoted out of it), so `fast_*` is purely the main queue's fast
		// segment and `slow_*` purely its demoted tail.
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

	// ── the new mechanic: shared-structure DRAM reservation ────────────────
	//
	// Every test above constructs the stack without `with_shared_overhead`, so
	// its per-tracked-key term is legitimately `0`. The ghost term is *not*
	// opt-in -- it is charged from `GHOST_ENTRY_DRAM_OVERHEAD` unconditionally
	// -- but every test above either leaves `ghost` empty or runs with enough
	// budget slack that a single ghost entry cannot change its outcome, which
	// is why none of them needed rescaling. The tests below opt into the
	// per-tracked-key term explicitly.
	//
	// `one_access_ratio` is `0.0` wherever the arithmetic needs to be exact:
	// that makes `one_access_capacity` zero, so `reserved_shares()` puts the
	// whole reservation on the main fast segment and the numbers below are the
	// undivided ones. The proportional split gets its own tests further down.

	/// Promotes `size`-byte keys one at a time until the reservation-tightened
	/// budget finally fires a demotion pass, then returns `(tracked key count,
	/// that pass's migrations)`. Each promotion adds `size` bytes of value AND
	/// one key's worth of reservation, so the effective budget closes on the
	/// usage from both sides -- which is the point: it is the reservation, not
	/// the raw capacity, that eventually trips the watermark.
	fn fill_until_reservation_fires(
		stack: &mut S3FifoGhostLazyDemotionFastAdmissionHybridStack,
		size: ObjectSize,
	) -> (CacheSize, Vec<(HashedKey, Tier)>) {
		let mut count: CacheSize = 0;

		loop {
			count += 1;
			assert!(count <= 10_000, "a demotion pass should have fired long before this");

			promote(stack, count, size);
			let migrations = stack.drain_tier_migrations();

			if migrations.iter().any(|(_, tier)| *tier == Tier::Slow) {
				return (count, migrations);
			}
		}
	}

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
		let mut plain = S3FifoGhostLazyDemotionFastAdmissionHybridStack::new(0.0, 100_000, capacity);

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
		assert_eq!(plain.effective_main_fast_capacity(), capacity);

		// With the reservation, the same capacity gives up 20 bytes per tracked
		// key, and the watermarks then apply to whatever is left.
		let mut stack = S3FifoGhostLazyDemotionFastAdmissionHybridStack::new(0.0, 100_000, capacity)
			.with_shared_overhead(overhead);

		promote(&mut stack, 1, size);
		assert!(
			!drain(&mut stack).iter().any(|(_, tier)| *tier == Tier::Slow),
			"one tracked key reserves only 20 of the budget, which still holds the object",
		);
		assert_eq!(stack.effective_main_fast_capacity(), capacity - overhead);

		// The second key takes the reservation to 40, dropping the effective
		// budget to `effective` -- too tight to hold both objects at once.
		promote(&mut stack, 2, size);
		let migrations = drain(&mut stack);

		assert_eq!(stack.effective_main_fast_capacity(), effective);
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
		let mut stack = S3FifoGhostLazyDemotionFastAdmissionHybridStack::new(0.0, 100_000, 50)
			.with_shared_overhead(100);

		promote(&mut stack, 1, 10);
		let migrations = drain(&mut stack);

		assert_eq!(stack.effective_main_fast_capacity(), 0);

		// No `(1, Tier::Fast)` migration is emitted at all -- this variant never
		// pushes one for a promotion out of the one-access queue (the key's
		// bytes were already Fast), so the demotion is the only entry.
		assert_eq!(migrations, vec![(1, Tier::Slow)]);
		assert_eq!(stack.tier_of(1), Some(Tier::Slow));
		assert_eq!(stack.fast_bytes_used(), 0);

		assert_eq!(stack.len(), 1);
		assert!(!stack.needs_capacity_eviction());
	}

	#[test]
	fn reservation_counts_every_tracked_key_not_just_fast_tier_ones() {
		let size: ObjectSize = 10;
		let overhead: CacheSize = 20;
		let capacity: CacheSize = 105;

		let mut stack = S3FifoGhostLazyDemotionFastAdmissionHybridStack::new(0.0, 100_000, capacity)
			.with_shared_overhead(overhead);

		for key in 1..=4 {
			promote(&mut stack, key, size);
		}

		drain(&mut stack);

		// Whatever the pass demoted, all four keys are still tracked and all
		// four are still charged: their hashtable and eviction-stack entries are
		// DRAM regardless of which tier their *data* sits in.
		assert_eq!(stack.len(), 4);
		assert_eq!(stack.reserved_overhead(), 4 * overhead);
		assert_eq!(stack.effective_main_fast_capacity(), capacity - 4 * overhead);
		assert!(stack.slow_object_count() > 0, "the pass must have demoted something");
		assert_eq!(
			stack.fast_object_count() + stack.slow_object_count(),
			4,
			"one_access_queue is empty here, so these two partition the tracked keys",
		);

		// Direct proof the SLOW keys were being charged: dropping one of them
		// gives exactly one key's worth of budget back.
		assert_eq!(stack.tier_of(1), Some(Tier::Slow));
		stack.remove(1);

		assert_eq!(stack.tier_of(1), None);
		assert_eq!(stack.reserved_overhead(), 3 * overhead);
		assert_eq!(stack.effective_main_fast_capacity(), capacity - 3 * overhead);
	}

	#[test]
	fn shared_overhead_reservation_splits_proportionally_between_fast_segments() {
		// one_access_capacity = 0.5 * 200 = 100, so the main queue's raw fast
		// segment is 200 - 100 = 100 too: an even split that is easy to check.
		let overhead: CacheSize = 20;

		let mut plain = S3FifoGhostLazyDemotionFastAdmissionHybridStack::new(0.5, 200, 200);
		let mut stack = S3FifoGhostLazyDemotionFastAdmissionHybridStack::new(0.5, 200, 200)
			.with_shared_overhead(overhead);

		for key in 1..=2 {
			promote(&mut plain, key, 1);
			promote(&mut stack, key, 1);
		}

		drain(&mut plain);
		drain(&mut stack);

		// No reservation: each segment keeps its whole configured capacity.
		assert_eq!(plain.reserved_overhead(), 0);
		assert_eq!(plain.reserved_shares(), (0, 0));
		assert_eq!(plain.effective_one_access_capacity(), 100);
		assert_eq!(plain.effective_main_fast_capacity(), 100);

		// 2 tracked keys x 20 = 40, split 20/20 by the two equal capacities --
		// NOT 40 charged to each, which would over-reserve by a factor of two.
		assert_eq!(stack.reserved_overhead(), 2 * overhead);
		assert_eq!(stack.reserved_shares(), (20, 20));
		assert_eq!(stack.effective_one_access_capacity(), 80);
		assert_eq!(stack.effective_main_fast_capacity(), 80);

		// The identity the split exists to preserve: the two effective budgets
		// plus the reservation still add back up to the configured fast tier.
		assert_eq!(
			stack.effective_one_access_capacity()
				+ stack.effective_main_fast_capacity()
				+ stack.reserved_overhead(),
			200,
		);
	}

	#[test]
	fn one_access_share_of_the_reservation_tightens_its_own_eviction_trigger() {
		// The same even split as the test above, scaled up 1_000x:
		// one_access_capacity = 0.5 * 200_000 = 100_000, so the main queue's
		// raw fast segment is 100_000 too. Five 18_000-byte keys sitting in the
		// one-access queue come to 90_000 bytes: comfortably under the raw
		// 100_000-byte cap, but over the 50_000 left once this segment gives up
		// its half of the 5 x 20_000 = 100_000-byte reservation.
		//
		// The scaling is what keeps the eviction sequence pinned. Each eviction
		// below now also creates a ghost entry, and the ghost term is charged
		// unconditionally, so the reservation no longer shrinks by a clean
		// `overhead` per eviction. At this scale `GHOST` is far too small to
		// move the stopping point either way, so the sequence is the same
		// whether or not `eviction_stacks_pmem` zeroes it out.
		let overhead: CacheSize = 20_000;

		let mut plain = S3FifoGhostLazyDemotionFastAdmissionHybridStack::new(0.5, 200_000, 200_000);
		let mut stack = S3FifoGhostLazyDemotionFastAdmissionHybridStack::new(0.5, 200_000, 200_000)
			.with_shared_overhead(overhead);

		for key in 1..=5 {
			plain.insert(key, 18_000);
			stack.insert(key, 18_000);
		}

		assert_eq!(plain.effective_one_access_capacity(), 100_000);
		assert!(!plain.needs_capacity_eviction());

		// `ghost` is still empty here, so this is the per-tracked-key term
		// alone.
		assert_eq!(stack.reserved_overhead(), 5 * overhead);
		assert_eq!(stack.reserved_shares(), (50_000, 50_000));
		assert_eq!(stack.effective_one_access_capacity(), 50_000);
		assert!(stack.needs_capacity_eviction());

		// It converges: each eviction untracks a key, handing back 20_000 of
		// reservation (10_000 of it to this segment), while adding one ghost
		// entry, which takes `GHOST` of it back (`GHOST / 2` to this segment).
		// 90_000 > 50_000 evicts key 1 (-> 72_000 used, cap 60_000 - GHOST/2),
		// 72_000 > that evicts key 2 (-> 54_000 used, cap 70_000 - GHOST), and
		// 54_000 is comfortably under it, so the loop stops. Pure integer
		// arithmetic -- no watermark is involved on this path.
		let mut evicted = Vec::new();

		while stack.needs_capacity_eviction() {
			evicted.push(stack.evict_one().expect("the one-access queue is non-empty"));
		}

		assert_eq!(evicted, vec![1, 2]);
		assert_eq!(stack.len(), 3);
		assert!(stack.is_ghost(1) && stack.is_ghost(2));

		// Three tracked keys and two ghost entries, each charged on its own
		// terms.
		assert_eq!(stack.reserved_overhead(), 3 * overhead + 2 * GHOST);
		assert_eq!(stack.effective_one_access_capacity(), 70_000 - GHOST);
	}

	/// The ghost term keys off `ghost.len()`, the per-tracked-key term off
	/// `entries.len()`, and the two are independent addends -- which is the
	/// whole reason the ghost cost cannot be folded into `shared_overhead`.
	#[test]
	fn the_ghost_term_scales_with_ghost_length_not_the_tracked_key_count() {
		let size: ObjectSize = 10;
		let overhead: CacheSize = 20;

		// Roomy enough that nothing here demotes: this test is about what the
		// reservation *is*, not about what it triggers.
		let capacity: CacheSize = 10_000;

		// (a) One tracked key and no ghost entries: the per-tracked-key term
		// alone, with the ghost term contributing nothing.
		let mut tracked_only = S3FifoGhostLazyDemotionFastAdmissionHybridStack::new(0.0, 100_000, capacity)
			.with_shared_overhead(overhead);

		promote(&mut tracked_only, 1, size);
		drain(&mut tracked_only);

		assert_eq!(tracked_only.len(), 1);
		assert_eq!(tracked_only.reserved_overhead(), overhead);
		assert_eq!(tracked_only.effective_main_fast_capacity(), capacity - overhead);

		// (b) The mirror image: one ghost entry and nothing tracked at all. The
		// evicted key is gone from `entries` (`len()` is 0) and never had an
		// object-hashtable slot, which is exactly why a per-tracked-key
		// constant cannot model this cost -- and its list node is still real
		// DRAM, so it is still charged.
		let mut ghost_only = S3FifoGhostLazyDemotionFastAdmissionHybridStack::new(0.0, 100_000, capacity)
			.with_shared_overhead(overhead);

		ghost_only.insert(9, size);
		assert_eq!(ghost_only.evict_one(), Some(9));
		assert!(ghost_only.is_ghost(9));
		assert_eq!(ghost_only.len(), 0);
		drain(&mut ghost_only);

		assert_eq!(ghost_only.reserved_overhead(), GHOST);
		assert_eq!(ghost_only.effective_main_fast_capacity(), capacity - GHOST);

		// (c) Both at once, as two independent addends.
		promote(&mut ghost_only, 1, size);
		drain(&mut ghost_only);

		assert_eq!(ghost_only.len(), 1);
		assert_eq!(ghost_only.reserved_overhead(), overhead + GHOST);

		// A second ghost adds exactly one more `GHOST` and leaves the
		// per-tracked-key term exactly where it was.
		ghost_only.insert(8, size);
		assert_eq!(ghost_only.evict_one(), Some(8));

		assert_eq!(ghost_only.len(), 1);
		assert_eq!(ghost_only.reserved_overhead(), overhead + 2 * GHOST);
		assert_eq!(ghost_only.effective_main_fast_capacity(), capacity - overhead - 2 * GHOST);

		// And dropping a ghost key hands its share straight back.
		ghost_only.remove(9);

		assert!(!ghost_only.is_ghost(9));
		assert_eq!(ghost_only.reserved_overhead(), overhead + GHOST);
		assert_eq!(ghost_only.effective_main_fast_capacity(), capacity - overhead - GHOST);
	}

	/// The regression the old shape allowed: the ghost price used to be a
	/// caller-supplied field defaulting to `0`, so a stack whose construction
	/// site forgot the builder that set it -- which is exactly what
	/// `init_policy_stack` did -- reserved nothing whatsoever for a ghost queue
	/// that really does occupy DRAM. It now comes straight from
	/// `GHOST_ENTRY_DRAM_OVERHEAD`, so it is charged independently of the
	/// per-tracked-key term, including when that term is `0` because
	/// `with_shared_overhead` was never called.
	#[test]
	fn ghost_entries_are_charged_even_when_shared_overhead_is_zero() {
		let size: ObjectSize = 10;
		let bytes = size as CacheSize;

		// A budget whose low watermark holds exactly one `size`-byte object
		// *after* a single ghost entry has been paid for.
		let effective = capacity_holding(bytes);
		let capacity = effective + GHOST;

		// Deliberately no `with_shared_overhead`: the per-tracked-key term is
		// `0` for this stack's entire lifetime.
		let mut stack = S3FifoGhostLazyDemotionFastAdmissionHybridStack::new(0.0, 100_000, capacity);

		assert_eq!(stack.reserved_overhead(), 0, "nothing tracked, and no ghost entries yet");
		assert_eq!(stack.effective_main_fast_capacity(), capacity);

		stack.insert(9, size);
		assert_eq!(stack.evict_one(), Some(9));
		assert!(stack.is_ghost(9));
		assert_eq!(stack.len(), 0);
		drain(&mut stack);

		// The point of the test: zero tracked keys, zero `shared_overhead`, and
		// DRAM is reserved anyway. (Under `eviction_stacks_pmem` `GHOST` is `0`
		// and `capacity == effective`: the ghost list lives in PMEM there, so it
		// genuinely costs the DRAM tier nothing and every assertion below still
		// holds with the reservation empty.)
		assert_eq!(stack.reserved_overhead(), GHOST);
		assert_eq!(stack.effective_main_fast_capacity(), effective);

		// And it is a real budget, not just a number: the watermarks are taken
		// against what the ghost entry left behind, so the second of two
		// `size`-byte objects can no longer sit fast beside the first.
		promote(&mut stack, 1, size);
		assert!(
			!drain(&mut stack).iter().any(|(_, tier)| *tier == Tier::Slow),
			"one object still fits under the low watermark of the ghost-reduced budget",
		);
		assert_eq!(stack.tier_of(1), Some(Tier::Fast));

		promote(&mut stack, 2, size);
		let migrations = drain(&mut stack);

		assert!(
			migrations.contains(&(1, Tier::Slow)),
			"the pass must run against `fast_capacity - ghost reservation`, got {migrations:?}",
		);
		assert_eq!(stack.tier_of(1), Some(Tier::Slow));
		assert_eq!(stack.tier_of(2), Some(Tier::Fast));
		assert_eq!(stack.fast_bytes_used(), bytes);
		assert!(stack.fast_bytes_used() <= watermarks::low_bytes(effective));

		// Demotion, never eviction -- and the ghost entry is still charged
		// alongside the two now-tracked keys.
		assert_eq!(stack.len(), 2);
		assert_eq!(stack.reserved_overhead(), GHOST);
	}

	/// The composition order that matters: the watermarks are taken against
	/// `fast_capacity - reserved`, so a triggered pass drains to
	/// `low_bytes(capacity - reserved)` -- never `low_bytes(capacity)`.
	#[test]
	fn the_drain_target_is_the_low_watermark_of_capacity_minus_the_reservation() {
		let capacity: CacheSize = 10_000;
		let overhead: CacheSize = 100;
		let size: ObjectSize = 100;
		let bytes = size as CacheSize;

		let mut stack = S3FifoGhostLazyDemotionFastAdmissionHybridStack::new(0.0, 100_000, capacity)
			.with_shared_overhead(overhead);

		let (count, _) = fill_until_reservation_fires(&mut stack, size);

		let effective = capacity - count * overhead;
		let low = watermarks::low_bytes(effective);

		assert_eq!(stack.reserved_overhead(), count * overhead);
		assert_eq!(stack.effective_main_fast_capacity(), effective);

		// Every object is `bytes`, so the pass halts at the first whole-object
		// multiple at or below the low watermark of the EFFECTIVE budget.
		let expected_used = low - low % bytes;

		assert_eq!(stack.fast_bytes_used(), expected_used);
		assert!(stack.fast_bytes_used() <= low);

		// Taking the watermark first and subtracting the reservation afterwards
		// -- the wrong composition order -- would have stopped at the strictly
		// larger `low_bytes(capacity)`; the same fill against the same raw
		// capacity with no reservation does not demote a single object.
		assert!(low < watermarks::low_bytes(capacity));

		let mut plain = S3FifoGhostLazyDemotionFastAdmissionHybridStack::new(0.0, 100_000, capacity);

		for key in 1..=count {
			promote(&mut plain, key, size);
		}

		assert!(
			!drain(&mut plain).iter().any(|(_, tier)| *tier == Tier::Slow),
			"the raw capacity alone holds this fill -- only the reservation makes it demote",
		);
		assert_eq!(plain.fast_bytes_used(), count * bytes);
	}

	#[test]
	fn counters_stay_consistent_across_a_reservation_triggered_pass() {
		let capacity: CacheSize = 10_000;
		let overhead: CacheSize = 100;
		let size: ObjectSize = 100;
		let bytes = size as CacheSize;

		let mut stack = S3FifoGhostLazyDemotionFastAdmissionHybridStack::new(0.0, 100_000, capacity)
			.with_shared_overhead(overhead);

		let (count, migrations) = fill_until_reservation_fires(&mut stack, size);
		let demoted = migrations.iter().filter(|(_, tier)| *tier == Tier::Slow).count() as CacheSize;

		let fast_objects = stack.fast_object_count() as CacheSize;
		let slow_objects = stack.slow_object_count() as CacheSize;

		// This is the first pass, so everything Slow was demoted by it -- once
		// each, no more and no less. Nothing was inserted, evicted or resized
		// mid-pass, and `one_access_queue` is empty (every key was promoted out
		// of it), so `fast_*`/`slow_*` partition the tracked keys exactly.
		assert!(slow_objects > 0);
		assert_eq!(demoted, slow_objects);
		assert_eq!(fast_objects + slow_objects, count);
		assert_eq!(stack.len() as CacheSize, count);

		assert_eq!(stack.fast_bytes_used(), fast_objects * bytes);
		assert_eq!(stack.slow_bytes_used(), slow_objects * bytes);
		assert_eq!(stack.fast_bytes_used() + stack.slow_bytes_used(), count * bytes);

		// The reservation is a loop invariant of the pass: a demotion retags an
		// entry, it never untracks a key nor touches `ghost`, so the effective
		// budget the pass drained against is the one still in force afterwards.
		assert_eq!(stack.reserved_overhead(), count * overhead);
		assert_eq!(stack.effective_main_fast_capacity(), capacity - count * overhead);
		assert!(stack.fast_bytes_used() <= watermarks::low_bytes(stack.effective_main_fast_capacity()));

		// And the aggregate counts agree with the per-key tier tags.
		let tagged_fast = (1..=count).filter(|key| stack.tier_of(*key) == Some(Tier::Fast)).count();
		let tagged_slow = (1..=count).filter(|key| stack.tier_of(*key) == Some(Tier::Slow)).count();

		assert_eq!(tagged_fast as CacheSize, fast_objects);
		assert_eq!(tagged_slow as CacheSize, slow_objects);
	}
}
