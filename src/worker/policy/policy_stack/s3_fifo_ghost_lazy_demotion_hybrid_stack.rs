/*
 * Copyright (c) Kia Shakiba
 *
 * This source code is licensed under the GNU AGPLv3 license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! `S3FifoGhostLazyDemotionHybridStack` — `S3FifoGhostHybridStack` with one
//! change: demotion is now reference-bit gated too, not just eviction. For
//! `PaperPolicy::S3FifoGhostLazyDemotionHybrid`.
//!
//! Identical to `S3FifoGhostHybridStack` in every other respect (ghost
//! queue lifecycle, admission/promotion/eviction rules, the "contiguous
//! front run" invariant) — see that stack's module doc, and
//! `S3FifoHybridStack`'s beneath it, for the full picture. The only change
//! is `settle_fast_tier`.
//!
//! ## Lazy demotion: the whole point of this variant
//!
//! The base S3-FIFO design (both hybrid variants above this one) is
//! classic "quick demotion, lazy promotion": `settle_fast_tier` demotes the
//! oldest fast key *unconditionally* — the reference bit is never
//! consulted there, only at eviction time. This variant makes demotion
//! reference-bit gated too: before actually demoting the key anchoring
//! `main_boundary`, its `accessed` bit is checked.
//!
//! * **Bit set** — the key was touched since being promoted. It is given a
//!   fresh start right here instead of being demoted: moved to the front
//!   of the fast portion, bit cleared, `Tier` and all fast/slow accounting
//!   left untouched (it was already `Tier::Fast` and stays `Tier::Fast` —
//!   this is a reprieve, not a promotion, so no migration is produced).
//!   The sweep then continues to the next-oldest fast key (the new
//!   `main_boundary`) and re-evaluates.
//! * **Bit clear** — demoted for real, exactly as the base design already
//!   does (unconditional aging once reached).
//!
//! In other words: S3-FIFO's own tagline becomes "lazy demotion, lazy
//! promotion" — the reference bit now gates *both* tier transitions
//! instead of only the eviction-time one. The eviction-time
//! `give_second_chance` (protecting a *slow* key that gets touched again
//! before it reaches the tail) is completely unchanged and still matters:
//! the two mechanisms protect different things (an unfairly-demoted fast
//! key here; an unfairly-evicted slow key there) and compose naturally.
//!
//! **Termination.** Each reprieve moves that key to the front and clears
//! its bit, so it cannot be re-examined as a demotion candidate again
//! until every other currently-fast key has had its own turn first (the
//! sweep only ever walks toward the back via `main_boundary`/`before`).
//! Bounded by `fast_count` reprieves per call before either a real
//! demotion happens or `fast_used` drops back to the fast tier's drain
//! target (see `settle_fast_tier` for the shared high/low watermarks that
//! set that target).
//!
//! **Deliberately not implemented via `give_second_chance`.** That method
//! itself calls `settle_fast_tier` at its own end (needed for *its* caller,
//! `evict_one`, since a promotion out of the slow tier can itself need to
//! free room). Reusing it here would recurse `settle_fast_tier` calling
//! `give_second_chance` calling `settle_fast_tier` for every reprieved key
//! — correct, but needlessly indirect and recursive for what's a pure
//! in-place reordering with no tier change. The reprieve arm below is a
//! trimmed-down inline copy: no `was_fast`/`!was_fast` accounting branch
//! (the key is always already fast here), no trailing migration push (no
//! tier changed).
//!
//! ## Eviction priority: the one-access tail first, but only while the
//! main queue has room
//!
//! Unchanged from `S3FifoGhostHybridStack`, and now mirroring
//! `SThreeFifoStack::evict_one` exactly:
//!
//! ```text
//! if !main_is_full() {
//!     // prioritize evicting from the one-access queue when possible
//!     if let Some(key) = evict_one_access_tail() { return Some(key) }
//! }
//! // ...otherwise the main-queue CLOCK sweep below
//! ```
//!
//! "Full" is `main_used >= main_capacity`, where `main_capacity` is
//! `(1 - one_access_ratio) * max_size` — the exact complement of
//! `one_access_capacity`, recomputed alongside it in `resize` — and
//! `main_used` is `fast_used + slow_used`. That is the same `used >= max`
//! test `SThreeFifoStack`'s per-queue `Stack::is_full` applies to its own
//! `main`; see `main_is_full` for why `fast_used + slow_used` is exactly
//! the main queue's byte total here, this variant's demotion-time reprieve
//! included.
//!
//! This gate was previously absent: the one-access tail was drained
//! *unconditionally* and there was no main-queue capacity concept at all.
//! Unlike `TwoQHybridStack`, which documents and argues its own eviction
//! priority, nothing in this family ever justified that divergence — it was
//! unexamined inheritance, and it is now corrected to match the plain,
//! non-hybrid policy these stacks are hybrids of.
//!
//! It leaves the ghost lifecycle alone: `ghost` is still populated by
//! `evict_one_access_tail` alone, so a call that finds the main queue full
//! simply produces no ghost entry — the same split `SThreeFifoStack`
//! already has between `evict_small` (adds) and `evict_main` (only trims).
//!
//! One consequence worth stating plainly, since it is shared with the plain
//! stack rather than introduced here: `one_access_ratio == 1.0` leaves
//! `main_capacity` at `0`, so the main queue reads as full from the outset
//! and the one-access tail is never preferred.
//! `SThreeFifoStack::new(1.0, _)` behaves identically.
//!
//! ## Shared-metadata DRAM reservation
//!
//! The fast tier is DRAM (NUMA node 0), but `fast_used` counts object
//! *values* only. The shared object hashtable and this stack's own
//! bookkeeping live in DRAM too and are invisible to `fast_used`, so
//! demoting purely against `fast_capacity` lets the tier's real DRAM
//! footprint overrun its budget. Two terms are therefore reserved out of
//! `fast_capacity` *before* the watermarks are applied (see
//! `reserved_overhead`):
//!
//! * **Per tracked key** — `shared_overhead`, charged against *every* key
//!   in `entries` regardless of tier, since a slow-tier object still owns a
//!   DRAM hashtable slot, a DRAM `entries` row and a DRAM queue-list node.
//!   A tracked key sits in exactly one of `one_access_queue`/`main_queue`
//!   at a time (`promote_from_one_access` removes it from the former before
//!   pushing it onto the latter, and `insert`/`admit_via_ghost_hit` each
//!   push onto exactly one of them), so that is one list node, never two.
//! * **Per ghost entry** — [`GHOST_ENTRY_DRAM_OVERHEAD`], charged against
//!   `ghost.len()`. Ghost entries are *bare keys* for objects that are
//!   neither in the cache nor in `entries` (`evict_one_access_tail` drops
//!   the `entries` row and pushes the key onto `ghost`), so no
//!   per-tracked-key term can model them — `ghost` scales with its own
//!   length, which `trim_ghost` caps at `main_count`.
//!
//! The ghost term is deliberately *not* caller-configurable: every ghost
//! variant keeps the same `HashList<HashedKey>`, so its per-entry cost is
//! the one shared constant and `reserved_overhead` reads it directly. That
//! also makes it unconditional — a ghost entry holds its DRAM whether or
//! not `with_shared_overhead` was ever called, and the constant is already
//! `0` under `eviction_stacks_pmem`, where the ghost list is PMEM-resident.
//! Only `shared_overhead` still defaults to `0`, so a stack built without
//! that builder reserves for its ghost entries and nothing else.

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
	worker::policy::policy_stack::{PolicyStack, Tier, narrow_resident, watermarks},
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
struct S3FifoEntry {queue: Queue,
	tier: Option<Tier>,
	/// Part of `size` that stays in DRAM in either tier; see `migrating`.
	dram_resident: u8,

	size: ObjectSize,
	accessed: bool,
}

impl S3FifoEntry {
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
	std::mem::size_of::<S3FifoEntry>() == 8,
	"S3FifoEntry grew past 8 bytes",
);


#[cfg(not(feature = "eviction_stacks_pmem"))]
type QueueList = HashList<HashedKey, NoHasher>;
#[cfg(feature = "eviction_stacks_pmem")]
type QueueList = PmemHashList<HashedKey, NoHasher>;

#[cfg(not(feature = "eviction_stacks_pmem"))]
type EntryMap = HashMap<HashedKey, S3FifoEntry, NoHasher>;
#[cfg(feature = "eviction_stacks_pmem")]
type EntryMap = HashMap<HashedKey, S3FifoEntry, NoHasher, Hybrid>;

pub struct S3FifoGhostLazyDemotionHybridStack {
	one_access_queue: QueueList,
	main_queue: QueueList,
	ghost: QueueList,

	entries: EntryMap,

	one_access_ratio: f64,
	one_access_capacity: CacheSize,
	one_access_used: CacheSize,

	/// Byte budget for `main_queue`, `(1 - one_access_ratio) * max_size` —
	/// the exact complement of `one_access_capacity`, computed the same way
	/// in both `new` and `resize`. Mirrors `SThreeFifoStack`'s
	/// `main.max_size`. Read only by `main_is_full`, which gates
	/// `evict_one`'s one-access-first priority; unlike
	/// `one_access_capacity` it never drives `needs_capacity_eviction`.
	main_capacity: CacheSize,

	fast_capacity: CacheSize,
	fast_used: CacheSize,
	slow_used: CacheSize,

	/// Approximate per-*tracked-key* DRAM cost of the shared structures (the
	/// object hashtable + this stack's eviction-stack bookkeeping) that hold
	/// an entry for every object of both tiers. Reserved out of
	/// `fast_capacity` in `settle_fast_tier` so the fast-tier budget bounds
	/// total DRAM (values + shared metadata), not just fast-tier values. `0`
	/// unless set via `with_shared_overhead` (so unit tests exercising the
	/// pure value-budget behaviour are unaffected).
	shared_overhead: CacheSize,

	fast_count: usize,

	/// Number of keys currently in the `Main` queue (Fast or Slow). Also
	/// used as the ghost list's size cap reference.
	main_count: usize,

	main_boundary: Option<HashedKey>,

	migrations: Vec<(HashedKey, Tier)>,
}

impl S3FifoGhostLazyDemotionHybridStack {
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

		S3FifoGhostLazyDemotionHybridStack {
			one_access_queue,
			main_queue,
			ghost,

			entries,

			one_access_ratio,
			one_access_capacity: (one_access_ratio * max_size as f64) as CacheSize,
			one_access_used: 0,
			main_capacity: ((1.0 - one_access_ratio) * max_size as f64) as CacheSize,

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

	/// Sets the approximate per-tracked-key shared-structure DRAM overhead
	/// (object hashtable + eviction stacks) reserved out of the fast-tier
	/// budget. See `crate::object::overhead::get_hybrid_dram_shared_overhead`.
	/// Builder-style so `init_policy_stack` can wire it in without disturbing
	/// `new`'s signature (unit tests keep the default `0`).
	pub fn with_shared_overhead(mut self, overhead: CacheSize) -> Self {
		self.shared_overhead = overhead;
		self
	}

	/// The configured fast-tier byte budget, before any reservation.
	pub fn fast_capacity(&self) -> CacheSize {
		self.fast_capacity
	}

	/// Total DRAM currently reserved for shared metadata: one
	/// `shared_overhead` per *tracked* key (both tiers — a slow-tier object
	/// still owns its DRAM hashtable slot, `entries` row and queue-list node,
	/// and a tracked key is in exactly one queue list at a time) plus one
	/// [`GHOST_ENTRY_DRAM_OVERHEAD`] per bare-key `ghost` entry (which no
	/// per-tracked-key term can express, since those keys are not in
	/// `entries`).
	///
	/// The two terms are independent: the ghost one is read straight from the
	/// shared constant rather than configured by the caller, so ghost DRAM is
	/// reserved even on a stack that was never given a `shared_overhead`
	/// (`GHOST_ENTRY_DRAM_OVERHEAD` is itself `0` under
	/// `eviction_stacks_pmem`, where the ghost list is PMEM-resident, which is
	/// the only condition that drops the term).
	fn reserved_overhead(&self) -> CacheSize {
		self.entries.len() as CacheSize * self.shared_overhead
			+ self.ghost.len() as CacheSize * (GHOST_ENTRY_DRAM_OVERHEAD as CacheSize)
	}

	/// The fast-tier *value*-byte budget actually available: `fast_capacity`
	/// minus [`Self::reserved_overhead`], saturating at 0. This is the value
	/// `settle_fast_tier` applies the watermarks to. Exposed for tests.
	pub fn effective_fast_capacity(&self) -> CacheSize {
		self.fast_capacity.saturating_sub(self.reserved_overhead())
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
		let dram_resident = entry.dram_resident;
		// Tier arithmetic moves only what migrates; `size` still rebuilds the entry.
		let size_bytes = entry.migrating();

		self.one_access_queue.remove(&key);
		self.one_access_used = self.one_access_used.saturating_sub(size_bytes);

		self.main_queue.push_front(key);
		self.entries.insert(key, S3FifoEntry { dram_resident, queue: Queue::Main,
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
	/// step.
	fn admit_via_ghost_hit(&mut self, key: HashedKey, size: ObjectSize, dram_resident: u8) {
		self.main_queue.push_front(key);
		self.entries.insert(key, S3FifoEntry { dram_resident, queue: Queue::Main,
			tier: Some(Tier::Fast),
			size,
			accessed: false,
		});
		self.fast_used += (size as CacheSize).saturating_sub(dram_resident as CacheSize);
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

	/// The eviction-time second chance — completely unchanged from
	/// `S3FifoGhostHybridStack`. Protects a *slow* key that gets touched
	/// again before it reaches the tail; independent of (and still
	/// necessary alongside) `settle_fast_tier`'s demotion-time reprieve
	/// below, which protects *fast* keys instead.
	fn give_second_chance(&mut self, key: HashedKey) {
		let Some(entry) = self.entries.get(&key).copied() else { return };
		let size = entry.migrating();
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

	/// Demotes key(s) anchoring `main_boundary` until the fast tier is back
	/// under the shared *low* watermark -- but only once usage has crossed the
	/// shared *high* watermark in the first place, and still reference-bit
	/// gated per candidate rather than unconditional. See the module doc's
	/// "Lazy demotion" section for the gating itself; it is unchanged, and it
	/// is still the one mechanic that differs from `S3FifoGhostHybridStack`.
	///
	/// The effective ceiling this stack works against is `fast_capacity`
	/// minus [`Self::reserved_overhead`] -- the DRAM held by the shared
	/// per-object metadata (hashtable + eviction stacks) across both tiers,
	/// plus the bare-key `ghost` entries -- saturating to 0 when that
	/// metadata alone meets or exceeds `fast_capacity`. This is what makes the
	/// fast-tier budget bound total DRAM rather than just fast-tier values;
	/// see the module doc's "Shared-metadata DRAM reservation" section. There
	/// is no *one-access* reservation on top of that: `one_access_queue` is
	/// slow-tier here (it holds brand-new keys at `Tier::Slow` and is bounded
	/// separately by `one_access_capacity`), so unlike the `fast_admission`
	/// variants there is only the one fast segment and the reservation is
	/// charged to it in full rather than split between segments. The
	/// `watermarks` helpers are applied *on top of* that effective value --
	/// they change only when a pass fires and how far it drains, never the
	/// budget itself.
	///
	/// Previously this drained to exactly `fast_capacity`, which pinned the
	/// tier at 100% utilisation and made essentially every promotion demote
	/// exactly one object (see the `watermarks` module doc). Setting both
	/// `FAST_TIER_HIGH_WATERMARK` and `FAST_TIER_LOW_WATERMARK` to `1.0`
	/// restores that behaviour byte-for-byte.
	///
	/// Per-demotion bookkeeping is deliberately untouched: each demoted object
	/// still retags its entry, still moves `fast_used`/`fast_count`/
	/// `slow_used` by its own size, still walks `main_boundary` one step
	/// toward the front, and still emits exactly one `Tier::Slow` migration --
	/// and a reprieved candidate still changes none of them. A pass simply
	/// walks further before it stops.
	///
	/// Termination is unaffected: the reprieve arm still moves the candidate
	/// to the front with its bit cleared and still walks `main_boundary`
	/// strictly toward the front, so it is bounded by `fast_count` reprieves
	/// before either a real demotion happens or the boundary runs out.
	fn settle_fast_tier(&mut self) {
		// Capacity minus the shared per-object metadata reservation. The
		// watermarks are applied *on top of* this value, never in place of
		// it.
		let effective_capacity = self.effective_fast_capacity();

		if self.fast_used <= watermarks::high_bytes(effective_capacity) {
			return;
		}

		let drain_target = watermarks::low_bytes(effective_capacity);

		while self.fast_used > drain_target {
			let Some(candidate) = self.main_boundary else { break };

			let accessed = self.entries.get(&candidate).map(|entry| entry.accessed).unwrap_or(false);

			if accessed {
				// Reprieve: fresh start at the front instead of demotion.
				// Same before-then-move ordering `give_second_chance` uses.
				// No fast/slow accounting change -- the key was already
				// Fast and stays Fast -- and no migration (no tier
				// changed), which is exactly why this isn't just a call to
				// `give_second_chance` (see the module doc).
				let new_boundary = self.main_queue.before(&candidate).copied();

				self.main_queue.move_front(&candidate);
				self.main_boundary = new_boundary;

				if let Some(entry) = self.entries.get_mut(&candidate) {
					entry.accessed = false;
				}

				continue;
			}

			let size = self.entries.get(&candidate).map(|entry| entry.migrating()).unwrap_or(0);
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

	/// Whether `main_queue` is at or over its byte budget — the gate on
	/// `evict_one`'s one-access-first priority. Mirrors
	/// `SThreeFifoStack::Stack::is_full`'s `used_size >= max_size`, applied
	/// to the same quantity.
	///
	/// `main_used` is exactly `fast_used + slow_used`. A `one_access_queue`
	/// resident is admitted with `tier: None` and its bytes go to
	/// `one_access_used` alone (`insert`); `resize_key`'s `Queue::OneAccess`
	/// arm keeps them there. `fast_used`/`slow_used` move only on the
	/// main-queue paths — `promote_from_one_access` (which hands the bytes
	/// over from `one_access_used`), `admit_via_ghost_hit` (a brand-new key
	/// admitted straight into `main_queue`, so its bytes never pass through
	/// `one_access_used` at all), `give_second_chance`, `settle_fast_tier`'s
	/// demotion arm, `remove`'s `Queue::Main` arm and `evict_one`'s own
	/// main-queue arm. No one-access byte is ever counted here, and every
	/// main-queue byte is, on exactly one side of the fast/slow line.
	///
	/// This variant's demotion-time reprieve does not disturb that: it only
	/// reorders `main_queue` and clears a reference bit, moving no bytes
	/// between `fast_used` and `slow_used` and changing their sum not at all
	/// — which is exactly why a reprieved key still counts toward the main
	/// queue being full.
	///
	/// `ghost` contributes nothing: its entries are bare keys for objects
	/// no longer in the cache, and `evict_one_access_tail` hands their bytes
	/// back to `one_access_used` before pushing them onto it. (Ghost DRAM is
	/// accounted for separately, against the *fast tier*, by
	/// `reserved_overhead` — a different budget entirely.)
	fn main_is_full(&self) -> bool {
		self.fast_used + self.slow_used >= self.main_capacity
	}

	/// Pops `one_access_queue`'s tail, removes it from this stack's own
	/// bookkeeping, and remembers it in `ghost`. Only called from
	/// `evict_one`, and only when `main_is_full()` is false.
	fn evict_one_access_tail(&mut self) -> Option<HashedKey> {
		let key = self.one_access_queue.pop_back()?;
		let size = self.entries.remove(&key).map(|entry| entry.migrating()).unwrap_or(0);

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

impl PolicyStack for S3FifoGhostLazyDemotionHybridStack {
	fn is_policy(&self, policy: &PaperPolicy) -> bool {
		matches!(policy, PaperPolicy::S3FifoGhostLazyDemotionHybrid(ratio) if *ratio == self.one_access_ratio)
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
			self.resize_key(key, size, dram_resident);
			self.touch(key);
			return;
		}

		if self.ghost.contains(&key) {
			self.admit_via_ghost_hit(key, size, dram_resident);
			return;
		}

		self.one_access_queue.push_front(key);
		self.entries.insert(key, S3FifoEntry { dram_resident, queue: Queue::OneAccess,
			tier: None,
			size,
			accessed: false,
		});
		self.one_access_used += (size as CacheSize).saturating_sub(dram_resident as CacheSize);
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
		let size = entry.migrating();

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
		self.main_capacity = ((1.0 - self.one_access_ratio) * max_size as f64) as CacheSize;
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

	/// Priority: `one_access_queue`'s tail first, but only while
	/// `main_queue` is not full (`main_is_full`) — exactly
	/// `SThreeFifoStack::evict_one`'s gate, see the module doc's "Eviction
	/// priority" section. Otherwise, and whenever `one_access_queue` is
	/// empty, sweeps `main_queue`'s tail with the usual second-chance check.
	fn evict_one(&mut self) -> Option<HashedKey> {
		if !self.main_is_full() {
			// prioritize evicting from the one-access queue when possible
			if let Some(key) = self.evict_one_access_tail() {
				return Some(key);
			}
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
			let size = removed.map(|entry| entry.migrating()).unwrap_or(0);
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

	fn dram_reserved_bytes(&self) -> CacheSize {
		self.reserved_overhead()
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

	fn drain(stack: &mut S3FifoGhostLazyDemotionHybridStack) -> Vec<(HashedKey, Tier)> {
		stack.drain_tier_migrations()
	}

	/// `insert` + `update` -- the admit-into-`one_access_queue`-then-promote
	/// pairing every fast-tier test in this module already uses, since this
	/// stack never admits a fresh (non-ghost) key straight into `main_queue`'s
	/// fast tier.
	fn promote(stack: &mut S3FifoGhostLazyDemotionHybridStack, key: HashedKey, size: ObjectSize) {
		stack.insert(key, size);
		stack.update(key);
	}

	/// Smallest fast-tier capacity whose *low* watermark still leaves room for
	/// `bytes`. Lets the fast-tier tests state their expectations in whole
	/// objects instead of hard-coded byte thresholds, so they hold at whatever
	/// `FAST_TIER_HIGH_WATERMARK`/`FAST_TIER_LOW_WATERMARK` pair is configured
	/// rather than only at the default ratios. The `while` loop absorbs the
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
		let mut stack = S3FifoGhostLazyDemotionHybridStack::new(1.0, 1_000, 1_000);

		stack.insert(1, 10);
		stack.insert(2, 10);

		assert_eq!(stack.tier_of(1), Some(Tier::Slow));
		assert_eq!(stack.tier_of(2), Some(Tier::Slow));
		assert_eq!(drain(&mut stack), Vec::new());
	}

	#[test]
	fn reaccessing_a_one_access_key_promotes_it_eagerly_to_fast() {
		let mut stack = S3FifoGhostLazyDemotionHybridStack::new(1.0, 1_000, 1_000);

		stack.insert(1, 10);
		drain(&mut stack);

		stack.update(1);
		let migrations = drain(&mut stack);

		assert_eq!(migrations, vec![(1, Tier::Fast)]);
		assert_eq!(stack.tier_of(1), Some(Tier::Fast));
	}

	#[test]
	fn a_key_aging_out_without_reaccess_becomes_a_ghost_entry() {
		// Ratio 0.5 over twice the max size keeps `one_access_capacity` at its
		// original value while leaving `main_capacity` non-degenerate: at a
		// ratio of 1.0 it would be 0, `main_is_full()` would be true from the
		// outset, and `evict_one` could never reach the one-access tail this
		// test needs it to age out.
		let mut stack = S3FifoGhostLazyDemotionHybridStack::new(0.5, 2_000, 1_000);

		stack.insert(1, 10);
		drain(&mut stack);

		assert_eq!(stack.evict_one(), Some(1));
		assert!(stack.is_ghost(1));
	}

	#[test]
	fn ghost_hit_on_readmission_lands_directly_in_fast_tier() {
		// Ratio 0.5 over twice the max size keeps `one_access_capacity` at its
		// original value while leaving `main_capacity` non-degenerate: at a
		// ratio of 1.0 it would be 0, `main_is_full()` would be true from the
		// outset, and `evict_one` could never reach the one-access tail this
		// test needs it to age out.
		let mut stack = S3FifoGhostLazyDemotionHybridStack::new(0.5, 2_000, 1_000);

		stack.insert(1, 10);
		drain(&mut stack);
		stack.evict_one();
		assert!(stack.is_ghost(1));

		stack.insert(1, 10);
		let migrations = drain(&mut stack);

		assert_eq!(migrations, vec![(1, Tier::Fast)]);
		assert_eq!(stack.tier_of(1), Some(Tier::Fast));
	}

	// ── the signature mechanic: reprieve at DEMOTION time ──────────────────

	#[test]
	fn an_accessed_fast_boundary_key_is_reprieved_at_demotion_time_not_demoted() {
		// Sized so the fast tier holds exactly one of these 10-byte objects
		// across a triggered pass: a second one trips the high watermark, and
		// the pass then drains to a low watermark that still fits one. (Was a
		// hard-coded 10: correct back when a pass triggered at, and drained to,
		// the ceiling, but under the watermarks a 10-byte ceiling triggers on
		// the very first object and drains the tier empty.) Assumes a
		// watermark pair narrow enough that two of these objects still trip
		// the high watermark -- i.e. `high()/low() < 2`, comfortably true of
		// the 0.98/0.95 defaults.
		let mut stack = S3FifoGhostLazyDemotionHybridStack::new(1.0, 1_000, capacity_holding(10));

		stack.insert(1, 10);
		stack.update(1); // promote 1 -> Fast (main_boundary = 1)
		drain(&mut stack);

		// Touch key 1 again while it's still Fast -- sets its bit, no
		// reorder (same lazy-bit convention as the base design).
		stack.update(1);
		assert_eq!(drain(&mut stack), Vec::new());

		// Promoting key 2 pushes fast_used to 20 > 10 -- settle_fast_tier
		// must demote *someone*. In the base S3FifoGhostHybridStack this
		// would demote key 1 unconditionally. Here, key 1's bit is set, so
		// it gets reprieved (moved to the front, bit cleared, stays Fast)
		// instead -- and the sweep must find someone else. Key 2 itself
		// becomes the new (and only remaining) boundary candidate; its bit
		// is clear (just promoted, never touched again), so IT gets
		// demoted instead.
		stack.insert(2, 10);
		stack.update(2);
		let migrations = drain(&mut stack);

		assert_eq!(stack.tier_of(1), Some(Tier::Fast), "key 1 should have been reprieved, not demoted");
		assert_eq!(stack.tier_of(2), Some(Tier::Slow), "key 2 should have been demoted in key 1's place");

		// Only a genuine tier change produces a migration -- key 1's
		// reprieve is not one (it was already Fast and stays Fast), so the
		// only migration this call produces is key 2's real demotion. Key
		// 2's own would-be promotion migration is suppressed too, since by
		// the time `promote_from_one_access` checks its tier after
		// `settle_fast_tier` ran, key 2 has already been demoted back to
		// Slow in its place -- net effect: key 2 never shows up as Fast at
		// all, only as the demotion.
		assert_eq!(migrations, vec![(2, Tier::Slow)]);
	}

	#[test]
	fn fast_tier_pressure_demotes_the_oldest_when_unaccessed() {
		// Base-design-equivalent behavior when nothing has been reaccessed:
		// demotion is still effectively "unconditional" (every candidate's
		// bit is clear), same outcome as S3FifoHybridStack's own test.
		// Sized so a triggered pass drains to a low watermark that still holds
		// two of these three 10-byte objects -- i.e. exactly one demotion, the
		// oldest fast key. (Was a hard-coded 25: correct back when a pass
		// drained to the ceiling, but under the watermarks a 25-byte ceiling
		// drains to 18 and takes key 2 down with key 1.) The premise -- three
		// objects trip the high watermark while two still fit under the low
		// one -- needs `high()/low() < 1.5`, again comfortably true of the
		// 0.98/0.95 defaults. The watermark-specific tests at the bottom of
		// this module derive their expectations instead, so they hold at any
		// configured pair.
		let mut stack = S3FifoGhostLazyDemotionHybridStack::new(1.0, 1_000, capacity_holding(20));

		stack.insert(1, 10);
		stack.insert(2, 10);
		stack.update(1);
		stack.update(2);
		drain(&mut stack);

		assert_eq!(stack.fast_bytes_used(), 20);

		stack.insert(3, 10);
		stack.update(3);
		let migrations = drain(&mut stack);

		assert!(migrations.iter().any(|(k, t)| *k == 1 && *t == Tier::Slow));
		assert_eq!(stack.tier_of(1), Some(Tier::Slow));
		assert_eq!(stack.tier_of(2), Some(Tier::Fast));
		assert_eq!(stack.tier_of(3), Some(Tier::Fast));
	}

	#[test]
	fn evict_one_gives_an_accessed_slow_key_a_second_chance() {
		// Same one-object-wide fast tier as the reprieve test above, sized
		// against the low watermark for the same reason: the second chance
		// promotes key 1 back to fast, and that has to survive the
		// `settle_fast_tier` call `give_second_chance` makes on its way out.
		// Same `high()/low() < 2` assumption as that test.
		let mut stack = S3FifoGhostLazyDemotionHybridStack::new(1.0, 1_000, capacity_holding(10));

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

	/// The one-access-first half of `evict_one`'s priority rule: while
	/// `main_queue` is under its budget the one-access tail goes first, even
	/// though `main_queue` also holds an evictable key — and, this being a
	/// ghost variant, that eviction is what leaves a ghost entry behind.
	///
	/// Paired with `evict_one_evicts_the_main_tail_once_the_main_queue_is_full`
	/// below: same ratio, same capacities, same key sizes — the only
	/// difference is how many bytes sit in `main_queue`. Together they pin
	/// the gate itself rather than either branch alone.
	#[test]
	fn evict_one_prefers_one_access_queue_while_the_main_queue_has_room() {
		// one_access_capacity = 20, main_capacity = 20.
		let mut stack = S3FifoGhostLazyDemotionHybridStack::new(0.5, 40, 1_000);

		stack.insert(1, 10); // one-access
		stack.insert(2, 10);
		stack.update(2); // promote 2 -> Main/Fast
		drain(&mut stack);

		// main_used = fast_used + slow_used = 10 < main_capacity = 20.
		assert!(!stack.main_is_full());

		assert_eq!(stack.evict_one(), Some(1));
		assert!(stack.is_ghost(1), "a one-access eviction is what populates `ghost`");
		assert_eq!(stack.tier_of(2), Some(Tier::Fast));
	}

	/// The other half: once `main_queue` is at its budget the gate closes and
	/// `evict_one` goes straight to the main-queue sweep, leaving the
	/// one-access tail alone — exactly what `SThreeFifoStack::evict_one` does
	/// when `self.main.is_full()`.
	///
	/// Before this gate existed, `evict_one` drained the one-access tail
	/// unconditionally: this test would have evicted key 3 and left a ghost
	/// entry for it, instead of evicting main's key 1 and leaving `ghost`
	/// empty.
	///
	/// Both main-queue keys are promoted rather than reprieved, so their
	/// reference bits are clear and this variant's demotion-time reprieve
	/// stays out of the way — the gate is the only thing under test.
	#[test]
	fn evict_one_evicts_the_main_tail_once_the_main_queue_is_full() {
		// Same fixture as the test above, and a fast tier far too slack to
		// demote anything.
		let mut stack = S3FifoGhostLazyDemotionHybridStack::new(0.5, 40, 1_000);

		stack.insert(1, 10);
		stack.update(1); // promote 1 -> Main/Fast (main_queue tail)
		stack.insert(2, 10);
		stack.update(2); // promote 2 -> Main/Fast (main_queue front)
		stack.insert(3, 10); // one-access
		drain(&mut stack);

		assert!(stack.main_is_full());
		assert_eq!(stack.tier_of(3), Some(Tier::Slow));

		// The main queue's oldest key, not the one-access tail.
		assert_eq!(stack.evict_one(), Some(1));
		assert!(stack.contains(3), "the one-access tail must be left alone while main is full");
		assert_eq!(stack.tier_of(2), Some(Tier::Fast));

		// A main-queue eviction never populates `ghost` -- only
		// `evict_one_access_tail` does, which this call never reached.
		assert!(!stack.is_ghost(1));
		assert!(!stack.is_ghost(3));

		// That eviction took main back under its budget, so the gate reopens
		// and the one-access tail is preferred again -- ghost entry and all.
		assert!(!stack.main_is_full());
		assert_eq!(stack.evict_one(), Some(3));
		assert!(stack.is_ghost(3));
	}

	#[test]
	fn remove_clears_ghost_entry_too() {
		// Ratio 0.5 over twice the max size keeps `one_access_capacity` at its
		// original value while leaving `main_capacity` non-degenerate: at a
		// ratio of 1.0 it would be 0, `main_is_full()` would be true from the
		// outset, and `evict_one` could never reach the one-access tail this
		// test needs it to age out.
		let mut stack = S3FifoGhostLazyDemotionHybridStack::new(0.5, 2_000, 1_000);

		stack.insert(1, 10);
		drain(&mut stack);
		stack.evict_one();
		assert!(stack.is_ghost(1));

		stack.remove(1);
		assert!(!stack.is_ghost(1));
	}

	#[test]
	fn remove_and_clear_reset_bookkeeping() {
		let mut stack = S3FifoGhostLazyDemotionHybridStack::new(1.0, 1_000, 1_000);

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

	// -- shared fast-tier high/low watermarks -------------------------------

	/// (a) The trigger is a strict `>`, so usage sitting right *on* the high
	/// watermark -- the largest usage that is not over it -- must leave the
	/// tier completely alone.
	#[test]
	fn fast_usage_at_the_high_watermark_triggers_no_demotion() {
		let fast_capacity: CacheSize = 1_000;
		let high = watermarks::high_bytes(fast_capacity);

		let mut stack = S3FifoGhostLazyDemotionHybridStack::new(1.0, 100_000, fast_capacity);

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
		assert_eq!(stack.slow_bytes_used(), 0);
	}

	/// (b) One byte past the high watermark -- the smallest possible overshoot
	/// -- must fire a pass, and it must take the oldest fast key rather than
	/// the key that just arrived. Neither key has been touched since being
	/// promoted, so no reference bit is set and the pass demotes for real
	/// instead of reprieving.
	#[test]
	fn fast_usage_above_the_high_watermark_triggers_a_demotion_pass() {
		let fast_capacity: CacheSize = 1_000;
		let high = watermarks::high_bytes(fast_capacity);

		let mut stack = S3FifoGhostLazyDemotionHybridStack::new(1.0, 100_000, fast_capacity);

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

		let mut stack = S3FifoGhostLazyDemotionHybridStack::new(1.0, 100_000, fast_capacity);

		// Exactly one object past the high watermark, so precisely one pass
		// fires -- with plenty of resident objects for it to chew through
		// before it reaches the low watermark. Every one of them is freshly
		// promoted with a clear reference bit, so none is reprieved.
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

		let mut stack = S3FifoGhostLazyDemotionHybridStack::new(1.0, 100_000, fast_capacity);

		for key in 1..=count {
			promote(&mut stack, key, size);
		}

		drain(&mut stack);

		let fast_objects = stack.fast_object_count() as CacheSize;
		let slow_objects = stack.slow_object_count() as CacheSize;

		// Nothing was inserted, evicted or resized mid-pass, so every object is
		// still tracked, still `size` bytes, and still on exactly one side of
		// the fast/slow line. `one_access_queue` is empty here -- every key was
		// promoted out of it -- so `slow_bytes_used`'s `one_access_used` term
		// contributes nothing and the slow side is purely demoted objects.
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

	/// This variant's signature mechanic is untouched by the watermarks: a
	/// candidate whose reference bit is set is still reprieved (front, bit
	/// cleared, tier and accounting untouched, no migration) and the sweep
	/// still moves on to the next-oldest fast key. It just has further to walk
	/// now that a pass drains to the low watermark instead of the ceiling.
	#[test]
	fn a_watermark_drain_still_reprieves_accessed_boundary_keys() {
		let fast_capacity: CacheSize = 1_000;
		let size: ObjectSize = 10;
		let bytes = size as CacheSize;

		let high = watermarks::high_bytes(fast_capacity);
		let low = watermarks::low_bytes(fast_capacity);

		let mut stack = S3FifoGhostLazyDemotionHybridStack::new(1.0, 100_000, fast_capacity);

		let count = high / bytes + 1;

		// Fill to the high watermark without tripping it.
		for key in 1..count {
			promote(&mut stack, key, size);
		}
		drain(&mut stack);

		// Set the reference bit on the three oldest fast keys -- the first three
		// demotion candidates the pass will reach. Marking is lazy: no reorder,
		// no tier change, no migration.
		for key in 1..=3 {
			stack.update(key);
		}
		assert_eq!(drain(&mut stack), Vec::new());

		// One more object trips the high watermark and fires the pass.
		promote(&mut stack, count, size);
		let migrations = drain(&mut stack);

		for key in 1..=3 {
			assert_eq!(
				stack.tier_of(key), Some(Tier::Fast),
				"key {key} should have been reprieved, not demoted",
			);
			assert!(
				!migrations.contains(&(key, Tier::Slow)),
				"a reprieve is not a tier change and must not emit a migration, got {migrations:?}",
			);
		}

		// The first candidate with a clear bit is demoted for real, and the pass
		// still runs all the way down to the low watermark.
		assert!(migrations.contains(&(4, Tier::Slow)));
		assert_eq!(stack.fast_bytes_used(), low - low % bytes);
		assert!(stack.fast_bytes_used() <= low);
	}

	// -- shared-metadata DRAM reservation -----------------------------------
	//
	// Every test above this line builds the stack without
	// `with_shared_overhead`, so the per-tracked-key term is `0` there. The
	// ghost term is not caller-configurable and so is always live, but the
	// three tests above that leave a ghost entry behind
	// (`a_key_aging_out_without_reaccess_becomes_a_ghost_entry`,
	// `ghost_hit_on_readmission_lands_directly_in_fast_tier`,
	// `remove_clears_ghost_entry_too`) all run a single 10-byte object against
	// a 1_000-byte fast tier, so shrinking that budget by one
	// `GHOST_ENTRY_DRAM_OVERHEAD` leaves it far above the high watermark: no
	// pass can fire and none of them needed adjusting. Every other test above
	// keeps `ghost` empty, so its effective capacity is `fast_capacity`
	// exactly.

	/// The per-*ghost-entry* charge as a `CacheSize`. Stated against the shared
	/// constant rather than hard-coded so these tests stay correct under
	/// `eviction_stacks_pmem`, where the ghost list is PMEM-resident and the
	/// constant is `0`.
	const GHOST_BYTES: CacheSize = GHOST_ENTRY_DRAM_OVERHEAD as CacheSize;

	/// The number of promoted `bytes`-sized objects at which a stack carrying
	/// a per-tracked-key reservation of `overhead` first trips its
	/// (reservation-shrunk) high watermark: the smallest `k` with
	/// `k * bytes > high_bytes(fast_capacity - k * overhead)`. The left side
	/// rises in `k` while the right side falls, so this is also the *first*
	/// `k` at which any pass fires at all -- every earlier promotion is
	/// guaranteed to have left the tier alone. Derived from the `watermarks`
	/// helpers rather than hard-coded so these tests hold at whatever
	/// `FAST_TIER_HIGH_WATERMARK`/`FAST_TIER_LOW_WATERMARK` pair the process
	/// was seeded with.
	fn first_firing_count(fast_capacity: CacheSize, bytes: CacheSize, overhead: CacheSize) -> CacheSize {
		let mut count: CacheSize = 1;

		while count * bytes <= watermarks::high_bytes(fast_capacity.saturating_sub(count * overhead)) {
			count += 1;
		}

		count
	}

	/// (1) The reservation is charged against *every tracked key*, not just
	/// fast-tier ones: a key still sitting in `one_access_queue` (slow) owns a
	/// DRAM hashtable slot, an `entries` row and a queue-list node exactly as a
	/// fast key does, so it shrinks the effective fast-tier budget too.
	#[test]
	fn shared_overhead_shrinks_effective_capacity_for_every_tracked_key() {
		const OVERHEAD: CacheSize = 64;

		let mut stack = S3FifoGhostLazyDemotionHybridStack::new(1.0, 100_000, 1_000)
			.with_shared_overhead(OVERHEAD);

		assert_eq!(stack.fast_capacity(), 1_000);
		assert_eq!(stack.effective_fast_capacity(), 1_000);

		// Tracked but still slow (one-access queue) -- charged all the same.
		stack.insert(1, 10);
		assert_eq!(stack.len(), 1);
		assert_eq!(stack.tier_of(1), Some(Tier::Slow));
		assert_eq!(stack.effective_fast_capacity(), 1_000 - OVERHEAD);

		// Promoting it to fast does not change the charge -- same one key,
		// same shared metadata, only the value bytes moved tier.
		stack.update(1);
		assert_eq!(stack.tier_of(1), Some(Tier::Fast));
		assert_eq!(stack.len(), 1);
		assert_eq!(stack.effective_fast_capacity(), 1_000 - OVERHEAD);

		promote(&mut stack, 2, 10);
		assert_eq!(stack.len(), 2);
		assert_eq!(stack.effective_fast_capacity(), 1_000 - 2 * OVERHEAD);

		// Untracking a key hands its reservation back.
		stack.remove(2);
		assert_eq!(stack.len(), 1);
		assert_eq!(stack.effective_fast_capacity(), 1_000 - OVERHEAD);

		// None of this was pressure enough to demote: 20 value bytes against
		// an effective 872 (and its high watermark) never came close.
		assert_eq!(stack.tier_of(1), Some(Tier::Fast));
		drain(&mut stack);
	}

	/// (2) The ghost term really is a separate axis: `ghost` holds bare keys
	/// for objects that are *not* in `entries`, so its DRAM scales with
	/// `ghost.len()` and no per-tracked-key constant can express it. Stated
	/// against `GHOST_ENTRY_DRAM_OVERHEAD` (via `GHOST_BYTES`) rather than a
	/// hard-coded 44, since the stack now reads that constant directly.
	#[test]
	fn ghost_entries_are_reserved_separately_from_tracked_keys() {
		const OVERHEAD: CacheSize = 64;

		// Ratio 0.5 over twice the max size keeps `one_access_capacity` at its
		// original value while leaving `main_capacity` non-degenerate: at a
		// ratio of 1.0 it would be 0, `main_is_full()` would be true from the
		// outset, and `evict_one` could never reach the one-access tail this
		// test needs it to age out.
		let mut stack = S3FifoGhostLazyDemotionHybridStack::new(0.5, 200_000, 1_000)
			.with_shared_overhead(OVERHEAD);

		stack.insert(1, 10);
		drain(&mut stack);
		assert_eq!(stack.effective_fast_capacity(), 1_000 - OVERHEAD);

		// Ageing key 1 out of the one-access queue drops its `entries` row --
		// so the per-tracked-key term goes away -- but leaves a bare-key ghost
		// entry behind, which only the ghost term covers.
		assert_eq!(stack.evict_one(), Some(1));
		assert!(stack.is_ghost(1));
		assert_eq!(stack.len(), 0);
		assert_eq!(stack.effective_fast_capacity(), 1_000 - GHOST_BYTES);

		// A ghost hit re-tracks the key *without* consuming its ghost entry
		// (`admit_via_ghost_hit` deliberately leaves `ghost` alone -- only
		// `remove` and `trim_ghost` ever shrink it), so both terms apply at
		// once and the two charges stack.
		stack.insert(1, 10);
		drain(&mut stack);
		assert_eq!(stack.tier_of(1), Some(Tier::Fast));
		assert!(stack.is_ghost(1));
		assert_eq!(stack.effective_fast_capacity(), 1_000 - OVERHEAD - GHOST_BYTES);

		// Dropping the key clears both.
		stack.remove(1);
		assert!(!stack.is_ghost(1));
		assert_eq!(stack.len(), 0);
		assert_eq!(stack.effective_fast_capacity(), 1_000);
	}

	/// (2a) The two terms are independent, and the ghost one is charged even
	/// when the per-tracked-key term was never configured: a ghost entry
	/// occupies its DRAM regardless of whether the caller remembered
	/// `with_shared_overhead`. This is the regression the old caller-configured
	/// ghost term allowed: it defaulted to `0`, so it silently reserved nothing
	/// at all unless the construction site remembered to set it.
	#[test]
	fn ghost_entries_are_reserved_without_any_shared_overhead() {
		// No builder at all, so `shared_overhead` keeps its `0` default.
		// Ratio 0.5 over twice the max size keeps `one_access_capacity` at its
		// original value while leaving `main_capacity` non-degenerate: at a
		// ratio of 1.0 it would be 0, `main_is_full()` would be true from the
		// outset, and `evict_one` could never reach the one-access tail this
		// test needs it to age out.
		let mut stack = S3FifoGhostLazyDemotionHybridStack::new(0.5, 200_000, 1_000);

		stack.insert(1, 10);
		stack.insert(2, 10);
		drain(&mut stack);

		// Two tracked keys, no ghosts: the per-key term being `0` means
		// nothing is reserved yet.
		assert_eq!(stack.len(), 2);
		assert_eq!(stack.effective_fast_capacity(), 1_000);

		// Ageing both out of the one-access queue drops their `entries` rows
		// (oldest first) and leaves two bare-key ghosts behind.
		assert_eq!(stack.evict_one(), Some(1));
		assert_eq!(stack.evict_one(), Some(2));
		assert_eq!(stack.len(), 0);
		assert!(stack.is_ghost(1) && stack.is_ghost(2));

		// Nothing is tracked and `shared_overhead` is `0`, so this reservation
		// is the ghost term alone -- and it scales with `ghost.len()`.
		assert_eq!(stack.effective_fast_capacity(), 1_000 - 2 * GHOST_BYTES);

		stack.remove(1);
		assert!(!stack.is_ghost(1));
		assert_eq!(stack.effective_fast_capacity(), 1_000 - GHOST_BYTES);
	}

	/// (2b) And that unconfigured ghost reservation really does drive
	/// `settle_fast_tier`: with `shared_overhead` left at `0`, the ghost term
	/// is the only thing shrinking the budget, and a pass fires against -- and
	/// drains to -- the shrunken value. Expectations are derived from
	/// `GHOST_BYTES` and the configured watermarks, so this holds at any
	/// watermark pair and under `eviction_stacks_pmem` (where `GHOST_BYTES` is
	/// `0` and the effective capacity is simply the raw one).
	#[test]
	fn a_ghost_only_reservation_still_drives_a_demotion_pass() {
		let fast_capacity: CacheSize = 1_000;
		let size: ObjectSize = 10;
		let bytes = size as CacheSize;
		let ghosts: CacheSize = 5;

		// Ratio 0.5 over twice the max size keeps `one_access_capacity` at its
		// original value while leaving `main_capacity` non-degenerate: at a
		// ratio of 1.0 it would be 0, `main_is_full()` would be true from the
		// outset, and `evict_one` could never reach the one-access tail this
		// test needs it to age out.
		let mut stack = S3FifoGhostLazyDemotionHybridStack::new(0.5, 200_000, fast_capacity);

		// Fill the fast tier to exactly the raw high watermark (whole objects
		// only, so at most it) -- not over it, so no pass fires during the
		// fill. Every key is freshly promoted with a clear reference bit, so
		// none of them can be reprieved later either.
		let count = watermarks::high_bytes(fast_capacity) / bytes;

		for key in 1..=count {
			promote(&mut stack, key, size);
		}

		assert!(
			!drain(&mut stack).iter().any(|(_, tier)| *tier == Tier::Slow),
			"filling to the high watermark must not demote anything yet",
		);
		assert_eq!(stack.fast_bytes_used(), count * bytes);
		assert_eq!(stack.effective_fast_capacity(), fast_capacity);

		// Now manufacture ghost entries: admit a key into `one_access_queue`
		// and age it straight back out. Neither step touches `fast_used` nor
		// calls `settle_fast_tier`, so the tier is untouched -- only the
		// reservation grows.
		for key in 1_001..=(1_000 + ghosts) {
			stack.insert(key, size);
			assert_eq!(stack.evict_one(), Some(key));
			assert!(stack.is_ghost(key));
		}

		let effective = fast_capacity - ghosts * GHOST_BYTES;

		assert_eq!(stack.len() as CacheSize, count);
		assert_eq!(stack.fast_bytes_used(), count * bytes);
		assert_eq!(stack.effective_fast_capacity(), effective);

		// One more promoted object trips the high watermark of that shrunken
		// budget and fires the pass.
		promote(&mut stack, count + 1, size);
		let migrations = drain(&mut stack);

		let low = watermarks::low_bytes(effective);

		assert!(
			migrations.contains(&(1, Tier::Slow)),
			"the pass must demote the oldest fast key, got {migrations:?}",
		);

		// The pass halts at the first whole-object multiple at or below the
		// *effective* low watermark -- the ghost term is the entire difference
		// between this target and `low_bytes(fast_capacity)`.
		assert_eq!(stack.fast_bytes_used(), low - low % bytes);
		assert!(stack.fast_bytes_used() <= low);
		assert!(
			GHOST_BYTES == 0 || low < watermarks::low_bytes(fast_capacity),
			"a ghost entry that costs DRAM must tighten the drain target",
		);

		// One demotion per object moved, no more and no less.
		let demoted = migrations.iter().filter(|(_, tier)| *tier == Tier::Slow).count() as CacheSize;

		assert_eq!(demoted, ((count + 1) * bytes - stack.fast_bytes_used()) / bytes);

		// Demotion never untracks and never consumes a ghost, so the
		// reservation the pass computed is still the one in force.
		assert_eq!(stack.len() as CacheSize, count + 1);
		assert_eq!(stack.effective_fast_capacity(), effective);
	}

	/// (3) Same capacity, same workload, one difference: the stack that
	/// reserves DRAM for shared metadata demotes strictly earlier than the one
	/// that does not.
	#[test]
	fn shared_overhead_demotes_earlier_than_an_unreserved_stack() {
		let capacity = capacity_holding(20);

		// Unreserved, two 10-byte objects sit at or under the *low* watermark
		// by construction, so no pass can fire.
		let mut plain = S3FifoGhostLazyDemotionHybridStack::new(1.0, 100_000, capacity);
		promote(&mut plain, 1, 10);
		promote(&mut plain, 2, 10);
		drain(&mut plain);

		assert_eq!(plain.fast_bytes_used(), 20);
		assert_eq!(plain.tier_of(1), Some(Tier::Fast));
		assert_eq!(plain.tier_of(2), Some(Tier::Fast));

		// Smallest per-key reservation that makes those same two keys overflow
		// the effective budget. Derived from the configured watermarks so this
		// holds at any ratio pair, not just the default ratios.
		let mut overhead: CacheSize = 1;

		while watermarks::high_bytes(capacity.saturating_sub(2 * overhead)) >= 20 {
			overhead += 1;
		}

		let mut stack = S3FifoGhostLazyDemotionHybridStack::new(1.0, 100_000, capacity)
			.with_shared_overhead(overhead);

		promote(&mut stack, 1, 10);
		drain(&mut stack);
		assert_eq!(
			stack.tier_of(1), Some(Tier::Fast),
			"one reserved key's worth of overhead must still leave room for its own 10 bytes",
		);

		promote(&mut stack, 2, 10);
		let migrations = drain(&mut stack);

		// The value bytes (20) are identical to `plain`'s and still fit the raw
		// capacity -- the reservation (2 x overhead) is the whole difference.
		assert_eq!(stack.effective_fast_capacity(), capacity - 2 * overhead);
		assert!(
			migrations.contains(&(1, Tier::Slow)),
			"the reserved stack must demote where the plain one did not, got {migrations:?}",
		);
		assert_eq!(stack.tier_of(1), Some(Tier::Slow));
		assert!(stack.fast_bytes_used() <= watermarks::low_bytes(stack.effective_fast_capacity()));

		// Note this also pins the drain *target* to the effective budget: had
		// the loop aimed at `low_bytes(capacity)` (>= 20 by `capacity_holding`)
		// instead, `fast_used` of 20 would not have exceeded it and key 1 would
		// still be Fast.
	}

	/// (4) Reservation and watermarks compose in the documented order: a pass
	/// fires at `high_bytes(capacity - reserved)` and drains to
	/// `low_bytes(capacity - reserved)` -- never to `low_bytes(capacity)`.
	#[test]
	fn a_pass_drains_to_the_low_watermark_of_the_effective_capacity() {
		let fast_capacity: CacheSize = 1_000;
		let size: ObjectSize = 10;
		let bytes = size as CacheSize;
		let overhead: CacheSize = 20;

		let count = first_firing_count(fast_capacity, bytes, overhead);
		let effective = fast_capacity.saturating_sub(count * overhead);

		let mut stack = S3FifoGhostLazyDemotionHybridStack::new(1.0, 100_000, fast_capacity)
			.with_shared_overhead(overhead);

		// Every promotion before the firing one leaves the tier alone.
		for key in 1..count {
			promote(&mut stack, key, size);
			assert_eq!(
				stack.fast_bytes_used(), key * bytes,
				"no pass may fire before object {count}",
			);
		}

		promote(&mut stack, count, size);
		let migrations = drain(&mut stack);

		// Demotion never untracks, so the reservation is the same one the pass
		// itself computed.
		assert_eq!(stack.len() as CacheSize, count);
		assert_eq!(stack.effective_fast_capacity(), effective);

		let low = watermarks::low_bytes(effective);

		// The pass halts at the first whole-object multiple at or below the
		// *effective* low watermark.
		assert_eq!(stack.fast_bytes_used(), low - low % bytes);
		assert!(stack.fast_bytes_used() <= low);

		// That target is strictly tighter than the raw capacity's -- and the
		// same fill would not even have tripped the raw high watermark, so
		// without the reservation nothing would have moved at all.
		assert!(low < watermarks::low_bytes(fast_capacity));
		assert!(count * bytes <= watermarks::high_bytes(fast_capacity));

		let demoted = migrations.iter().filter(|(_, tier)| *tier == Tier::Slow).count() as CacheSize;

		assert_eq!(demoted, (count * bytes - stack.fast_bytes_used()) / bytes);
	}

	/// (5) Every byte counter and object count still agrees with the per-key
	/// tier tags after a pass triggered by the reservation rather than by raw
	/// value pressure -- the per-demotion bookkeeping ran once per demoted
	/// object, no more and no less.
	#[test]
	fn counters_stay_consistent_across_a_reservation_triggered_pass() {
		let fast_capacity: CacheSize = 1_000;
		let size: ObjectSize = 10;
		let bytes = size as CacheSize;
		let overhead: CacheSize = 20;

		let count = first_firing_count(fast_capacity, bytes, overhead);

		let mut stack = S3FifoGhostLazyDemotionHybridStack::new(1.0, 100_000, fast_capacity)
			.with_shared_overhead(overhead);

		for key in 1..=count {
			promote(&mut stack, key, size);
		}

		drain(&mut stack);

		let fast_objects = stack.fast_object_count() as CacheSize;
		let slow_objects = stack.slow_object_count() as CacheSize;

		// Nothing was inserted, evicted or resized mid-pass, so every object is
		// still tracked, still `size` bytes, and still on exactly one side of
		// the fast/slow line. `one_access_queue` is empty -- every key was
		// promoted out of it -- so `slow_bytes_used`'s `one_access_used` term
		// contributes nothing and the slow side is purely demoted objects.
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

		// Demotion moves value bytes between tiers; it does not free shared
		// metadata, so the reservation is untouched by the pass.
		assert_eq!(stack.effective_fast_capacity(), fast_capacity - count * overhead);
	}

	/// (6) A reservation that alone meets or exceeds the fast budget saturates
	/// the effective capacity to 0: everything demotes, nothing is evicted
	/// (the DRAM budget is a demotion target, never a data-dropping ceiling --
	/// terminal eviction stays governed by `max_size`).
	#[test]
	fn overhead_exceeding_capacity_demotes_all_but_never_evicts() {
		let mut stack = S3FifoGhostLazyDemotionHybridStack::new(1.0, 100_000, 50)
			.with_shared_overhead(100);

		promote(&mut stack, 1, 10);
		let migrations = drain(&mut stack);

		assert_eq!(stack.effective_fast_capacity(), 0);

		// The would-be `(1, Fast)` promotion migration is suppressed: by the
		// time `promote_from_one_access` re-checks the tier, `settle_fast_tier`
		// has already put the key back in the slow tier.
		assert_eq!(migrations, vec![(1, Tier::Slow)]);
		assert_eq!(stack.tier_of(1), Some(Tier::Slow));
		assert_eq!(stack.fast_bytes_used(), 0);

		// Still tracked -- demotion is the only response.
		assert_eq!(stack.len(), 1);
		assert!(!stack.needs_capacity_eviction());
	}

	/// (7) This variant's signature mechanic survives the reservation intact: a
	/// candidate whose reference bit is set is still reprieved (front, bit
	/// cleared, tier and accounting untouched, no migration), and the pass --
	/// now aimed at the *effective* low watermark -- simply walks past it to
	/// the next candidate.
	#[test]
	fn a_reservation_triggered_pass_still_reprieves_accessed_boundary_keys() {
		let fast_capacity: CacheSize = 1_000;
		let size: ObjectSize = 10;
		let bytes = size as CacheSize;
		let overhead: CacheSize = 20;

		let count = first_firing_count(fast_capacity, bytes, overhead);
		let effective = fast_capacity.saturating_sub(count * overhead);
		let low = watermarks::low_bytes(effective);

		assert!(
			count >= 5,
			"this test needs at least three keys to reprieve plus a fourth to demote",
		);

		let mut stack = S3FifoGhostLazyDemotionHybridStack::new(1.0, 100_000, fast_capacity)
			.with_shared_overhead(overhead);

		// Fill to just short of the firing point.
		for key in 1..count {
			promote(&mut stack, key, size);
		}

		drain(&mut stack);
		assert_eq!(stack.fast_bytes_used(), (count - 1) * bytes);

		// Set the reference bit on the three oldest fast keys -- the first
		// three demotion candidates the pass will reach. Marking is lazy: no
		// reorder, no tier change, no migration.
		for key in 1..=3 {
			stack.update(key);
		}

		assert_eq!(drain(&mut stack), Vec::new());

		// One more object trips the reservation-shrunk high watermark.
		promote(&mut stack, count, size);
		let migrations = drain(&mut stack);

		for key in 1..=3 {
			assert_eq!(
				stack.tier_of(key), Some(Tier::Fast),
				"key {key} should have been reprieved, not demoted",
			);
			assert!(
				!migrations.contains(&(key, Tier::Slow)),
				"a reprieve is not a tier change and must not emit a migration, got {migrations:?}",
			);
		}

		// The first candidate with a clear bit is demoted for real, and the
		// pass still runs all the way down to the effective low watermark.
		assert!(migrations.contains(&(4, Tier::Slow)));
		assert_eq!(stack.fast_bytes_used(), low - low % bytes);
		assert!(stack.fast_bytes_used() <= low);
	}
}
