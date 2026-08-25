/*
 * Copyright (c) Kia Shakiba
 *
 * This source code is licensed under the GNU AGPLv3 license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! `S3FifoLazyDemotionReprieveHybridStack` — a slow-tier one-access queue
//! whose aged-out keys are reprieved into the main queue rather than
//! evicted. For `PaperPolicy::S3FifoLazyDemotionReprieveHybrid`.
//!
//! This fills the one empty cell in the s3-fifo family's design matrix. Every
//! other variant pairs its one-access-queue *placement* with a fixed choice of
//! what happens to a key that ages out of it:
//!
//! | variant | one-access tier | ages out without reaccess |
//! |---|---|---|
//! | `S3FifoHybridStack` (+ghost, +lazy demotion) | slow | evicted |
//! | `...FastAdmission...` (+midpoint) | fast | evicted |
//! | `...FastAdmissionReprieve...` (+midpoint, +split slow) | fast | reprieved |
//! | **this stack** | **slow** | **reprieved** |
//!
//! ## Why the combination is interesting: the splice costs nothing
//!
//! In the fast-admission reprieve variants the one-access queue is DRAM and
//! the main queue's slow segment is PMEM, so relieving one-access pressure
//! pushes a `(key, Tier::Slow)` migration and `apply_tier_migrations`
//! performs a real `TieredBuffer::new_slow` PMEM copy for *every* aged-out
//! object.
//!
//! Here both structures are in PMEM. `settle_one_access` moves the key from
//! one list to another and emits **no migration at all** — the bytes never
//! move. That makes the reprieve strictly cheaper than the eviction it
//! replaces (which also had to do bookkeeping, and additionally dropped the
//! object).
//!
//! The cost is on the other side of the ledger, and it is the paper-literal
//! admission rule this variant keeps: every `set()` is a synchronous PMEM
//! write on the calling thread. That is precisely the cost the fast-admission
//! branch exists to avoid.
//!
//! ## Promotion is a real move again
//!
//! The mirror image of the above. `promote_from_one_access` in the
//! fast-admission variants deliberately pushes *no* migration, because the
//! bytes were already in DRAM and a `Tier::Fast` migration would have been a
//! pointless DRAM→DRAM copy. Here a one-access key really is in PMEM, so
//! promoting it is a genuine PMEM→DRAM move and must emit the migration —
//! guarded, because `settle_fast_tier` may demote the key straight back out
//! in the same call (see that function).
//!
//! ## Lazy demotion (retained)
//!
//! `settle_fast_tier` is reference-bit gated: the key anchoring the fast/slow
//! boundary is demoted only if its `accessed` bit is clear. If the bit is
//! set, the key is given a fresh start at the front of `main_fast` with the
//! bit cleared and the sweep continues to the next candidate. Termination is
//! guaranteed because each reprieve clears exactly one bit.
//!
//! This matters more here than in the fast-admission variants: a demotion in
//! this design is a real DRAM→PMEM copy, so every demotion lazy demotion
//! avoids is a copy saved outright.
//!
//! ## No ghost queue
//!
//! Nothing would populate it. A ghost list records keys evicted from the
//! one-access queue, and here no key is ever evicted from it — they are all
//! reprieved into the main queue instead. The ghost machinery is therefore
//! absent rather than present-and-always-empty, matching
//! `S3FifoLazyDemotionFastAdmissionReprieveHybridStack`.
//!
//! ## Two physical main-queue lists
//!
//! `main_fast` and `main_slow` are separate `HashList`s rather than one list
//! plus a boundary cursor: demotion is `main_fast.pop_back()` +
//! `main_slow.push_front()`, promotion is `main_slow.remove()` +
//! `main_fast.push_front()`, eviction is `main_slow.pop_back()` (falling back
//! to `main_fast.pop_back()` only when nothing has ever been demoted). All
//! O(1), and the reprieve splice targets `main_slow.push_front()` — which
//! *is* the boundary position — so it is O(1) too.
//!
//! ## Shared-metadata DRAM reservation
//!
//! The object hashtable and this stack's own bookkeeping live in DRAM but are
//! not part of `fast_used`, so the fast tier's real DRAM footprint would
//! exceed its budget unless that metadata is reserved out of it.
//! `shared_overhead` (see
//! `crate::object::overhead::get_hybrid_dram_shared_overhead`) is the
//! per-tracked-key cost, `reserved_overhead()` multiplies it by
//! `entries.len()`, and `effective_main_fast_capacity()` subtracts the
//! product from `fast_capacity`.
//!
//! Charged against *every* tracked key, not just the fast-tier ones: a
//! one-access key's value is PMEM, but its object-hashtable entry and its
//! `one_access_queue`/`entries` nodes are DRAM exactly like a `main_fast`
//! key's.
//!
//! Charged in full against `fast_capacity` rather than split the way
//! `LruSizedHybridStack` splits its own between two segments:
//! `one_access_capacity` is a second capacity, but it bounds a PMEM-resident
//! queue, so `main_fast` is this stack's only DRAM segment and there is no
//! second DRAM budget to apportion a share to.
//!
//! ## `eviction_stacks_pmem`
//!
//! Same DRAM/PMEM backing switch as every other hybrid stack.

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
	worker::policy::policy_stack::{PolicyStack, Tier, narrow_resident, watermarks},
};

/// Which live queue a key currently belongs to.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Queue {
	OneAccess,
	Main,
}

/// Combined per-key bookkeeping. `tier`/`accessed` are only meaningful
/// while `queue == Main`. `tier` is redundant with which of the two main
/// lists the key is physically in, but kept because `tier_of()` and the
/// `PolicyWorker` migration path both want it as a cheap map lookup rather
/// than a pair of `contains()` probes.
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

pub struct S3FifoLazyDemotionReprieveHybridStack {
	one_access_queue: QueueList,

	/// Main queue, fast portion. Front = newest, back = oldest (the
	/// demotion candidate).
	main_fast: QueueList,
	/// Main queue, slow portion. Front = newest slow key -- i.e. exactly
	/// the fast/slow boundary position -- back = oldest (the eviction
	/// candidate).
	main_slow: QueueList,

	entries: EntryMap,

	one_access_ratio: f64,
	one_access_capacity: CacheSize,
	one_access_used: CacheSize,

	fast_capacity: CacheSize,
	fast_used: CacheSize,
	slow_used: CacheSize,

	/// Approximate per-object DRAM cost of the shared structures (object
	/// hashtable + eviction stacks) that hold an entry for every *tracked*
	/// key of both tiers. Reserved out of `fast_capacity` by
	/// `effective_main_fast_capacity()` so the fast-tier budget bounds total
	/// DRAM (values + shared metadata), not just fast-tier values. `0` unless
	/// set via `with_shared_overhead`, so every unit test below that exercises
	/// the pure value-budget behaviour is unaffected.
	shared_overhead: CacheSize,

	migrations: Vec<(HashedKey, Tier)>,
}

impl S3FifoLazyDemotionReprieveHybridStack {
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
		let (one_access_queue, main_fast, main_slow, entries) = Self::new_collections();

		S3FifoLazyDemotionReprieveHybridStack {
			one_access_queue,
			main_fast,
			main_slow,

			entries,

			one_access_ratio,
			one_access_capacity: (one_access_ratio * max_size as f64) as CacheSize,
			one_access_used: 0,

			fast_capacity,
			fast_used: 0,
			slow_used: 0,
			shared_overhead: 0,

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

	/// Total DRAM currently reserved for shared per-object metadata:
	/// `tracked key count * shared_overhead`.
	///
	/// `entries.len()` counts *every* tracked key -- the one-access queue and
	/// both main-queue lists alike -- not just the fast ones. A one-access
	/// key's value is PMEM, but its object-hashtable entry and its
	/// `one_access_queue`/`entries` nodes are DRAM exactly like a `main_fast`
	/// key's, so it is charged too. Matches `LruHybridStack`'s `stack.len()`
	/// and `LruSizedHybridStack`'s `entries.len()`.
	///
	/// A key occupies exactly *one* of the three lists at any instant -- every
	/// transition removes before it inserts (`promote_from_one_access`:
	/// `one_access_queue.remove` then `main_fast.push_front`;
	/// `settle_one_access`: `one_access_queue.pop_back` then
	/// `main_slow.push_front`; `settle_fast_tier`: `main_fast.pop_back` then
	/// `main_slow.push_front`; `give_second_chance`: `main_slow.remove` then
	/// `main_fast.push_front`) -- so the per-key constant charges one
	/// `HashList` node, not three.
	///
	/// There is no ghost-queue term to add here: this variant has no ghost
	/// queue at all (see the module doc's "No ghost queue" section -- nothing
	/// would ever populate one, since no key is evicted from the one-access
	/// queue, only reprieved).
	fn reserved_overhead(&self) -> CacheSize {
		self.entries.len() as CacheSize * self.shared_overhead
	}

	/// The whole `fast_capacity`, less the shared per-object metadata
	/// reservation, is available to the main queue.
	///
	/// The fast-admission variants subtract `one_access_capacity` here,
	/// because there the one-access queue is DRAM-resident and both budgets
	/// draw on the same physical pool. Here the one-access queue lives in
	/// PMEM, so it competes for nothing the main queue's fast segment wants
	/// -- reserving against it would silently shrink the DRAM tier for no
	/// reason. `one_access_capacity` still bounds the one-access queue's own
	/// (PMEM) footprint via `settle_one_access`.
	///
	/// That is also why `reserved_overhead()` is charged here in *full*,
	/// rather than split proportionally the way `LruSizedHybridStack` splits
	/// its own between two fast segments: `main_fast` is this stack's only
	/// DRAM segment. `one_access_capacity` is a second capacity, but it
	/// governs a PMEM-resident queue, so there is no second DRAM budget for a
	/// share of the reservation to go to.
	///
	/// Saturating: once the shared metadata alone meets or exceeds
	/// `fast_capacity` the effective value budget is `0`, and every main-queue
	/// key demotes -- never evicts, demotion being `settle_fast_tier`'s only
	/// response.
	fn effective_main_fast_capacity(&self) -> CacheSize {
		self.fast_capacity.saturating_sub(self.reserved_overhead())
	}

	pub fn tier_of(&self, key: HashedKey) -> Option<Tier> {
		let entry = self.entries.get(&key)?;

		match entry.queue {
			Queue::OneAccess => Some(Tier::Slow),
			Queue::Main => entry.tier,
		}
	}

	fn resize_key(&mut self, key: HashedKey, new_size: ObjectSize) {
		let Some(entry) = self.entries.get_mut(&key) else { return };

		let old_migrating = entry.migrating();
		entry.size = new_size;
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

		self.main_fast.push_front(key);
		self.entries.insert(key, S3FifoEntry { dram_resident, queue: Queue::Main,
			tier: Some(Tier::Fast),
			size,
			accessed: false,
		});
		self.fast_used += size_bytes;

		self.settle_fast_tier();

		// Unlike the fast-admission variants -- where a one-access entry's
		// bytes are already in DRAM, so promoting it moved nothing and
		// pushing a migration would have been a pointless DRAM->DRAM copy --
		// the bytes genuinely are in PMEM here, so this needs a real
		// promotion migration.
		//
		// Guarded for the same reason `give_second_chance` guards its own:
		// `settle_fast_tier` above may have demoted this very key straight
		// back out (a zero/tiny effective budget), in which case it already
		// pushed the correct `Tier::Slow` migration. Pushing `Tier::Fast`
		// as well would be applied *after* it -- `apply_tier_migrations`
		// runs every demotion before any promotion -- leaving the bytes in
		// DRAM while the stack believes they are in PMEM.
		if self.entries.get(&key).and_then(|entry| entry.tier) == Some(Tier::Fast) {
			self.migrations.push((key, Tier::Fast));
		}
	}

	/// The eviction-time second chance, shared with the demotion-boundary
	/// reprieve: both mean "this key's reference bit is set, so spare it
	/// and move it to the front of the fast list".
	fn give_second_chance(&mut self, key: HashedKey) {
		let Some(entry) = self.entries.get(&key).copied() else { return };
		let size = entry.migrating();

		match entry.tier {
			// Already fast (only reachable from `evict_one`'s fast-tail
			// fallback, i.e. nothing has ever been demoted): just reorder
			// to the front, no tier change and no byte movement.
			Some(Tier::Fast) => {
				self.main_fast.move_front(&key);

				if let Some(entry) = self.entries.get_mut(&key) {
					entry.accessed = false;
				}
			},

			Some(Tier::Slow) => {
				self.main_slow.remove(&key);
				self.main_fast.push_front(key);

				if let Some(entry) = self.entries.get_mut(&key) {
					entry.tier = Some(Tier::Fast);
					entry.accessed = false;
				}

				self.slow_used = self.slow_used.saturating_sub(size);
				self.fast_used += size;

				},

			None => return,
		}

		self.settle_fast_tier();

		// Only record a migration if the key actually ended up Fast -- the
		// `settle_fast_tier` above can immediately demote it right back out
		// when the fast tier is at capacity, in which case that call has
		// already pushed the correct `Tier::Slow` migration itself.
		if self.entries.get(&key).and_then(|entry| entry.tier) == Some(Tier::Fast) {
			self.migrations.push((key, Tier::Fast));
		}
	}

	/// Demotes oldest-first from `main_fast` into `main_slow` until the fast
	/// tier is back under the shared *low* watermark -- but only once usage
	/// has crossed the shared *high* watermark in the first place. Any key
	/// whose reference bit is set is still given a reprieve (moved to the
	/// front of `main_fast`, bit cleared) instead of being demoted.
	/// Terminates even when every fast key's bit is set, since each reprieve
	/// clears one bit.
	///
	/// The effective ceiling is exactly whatever
	/// `effective_main_fast_capacity()` says it is -- for this variant
	/// `fast_capacity` minus the shared per-object metadata reservation, with
	/// nothing carved out for the one-access queue (PMEM-resident, so it takes
	/// nothing from the DRAM budget -- see that method). The `watermarks`
	/// helpers are applied *on top of* that same effective value; they decide
	/// only when a pass fires and how far it drains, never what the budget is.
	///
	/// Previously this drained to exactly the effective capacity, which
	/// pinned the tier at 100% utilisation and made essentially every
	/// promotion demote exactly one object (see the `watermarks` module doc).
	/// Setting both `FAST_TIER_HIGH_WATERMARK` and `FAST_TIER_LOW_WATERMARK`
	/// to `1.0` restores that behaviour byte-for-byte.
	///
	/// Per-demotion bookkeeping is deliberately untouched: each demoted
	/// object still retags its entry, still moves between the two physical
	/// lists, still moves `fast_used`/`slow_used` by its own size, and still
	/// emits exactly one `Tier::Slow` migration.
	fn settle_fast_tier(&mut self) {
		let effective_capacity = self.effective_main_fast_capacity();

		if self.fast_used <= watermarks::high_bytes(effective_capacity) {
			return;
		}

		let drain_target = watermarks::low_bytes(effective_capacity);

		while self.fast_used > drain_target {
			let Some(candidate) = self.main_fast.back().copied() else { break };

			let accessed = self.entries.get(&candidate).map(|entry| entry.accessed).unwrap_or(false);

			if accessed {
				self.main_fast.move_front(&candidate);

				if let Some(entry) = self.entries.get_mut(&candidate) {
					entry.accessed = false;
				}

				continue;
			}

			let size = self.entries.get(&candidate).map(|entry| entry.migrating()).unwrap_or(0);

			self.main_fast.pop_back();
			self.main_slow.push_front(candidate);

			if let Some(entry) = self.entries.get_mut(&candidate) {
				entry.tier = Some(Tier::Slow);
			}

			self.fast_used = self.fast_used.saturating_sub(size);
			self.slow_used += size;

			self.migrations.push((candidate, Tier::Slow));
		}
	}

	/// Relieves one-access-queue pressure by moving its tail(s) to the front
	/// of `main_slow` -- which *is* the fast/slow boundary position, so this
	/// is a plain O(1) `push_front` (see the module doc's "Two physical
	/// main-queue lists" section for why this used to be an O(n) walk).
	/// Called synchronously from `insert()`/`resize()`, exactly mirroring
	/// `settle_fast_tier()`'s relationship to the fast/slow boundary. A pure
	/// internal migration: nothing is ever removed from the cache here, so
	/// this must never be routed through `evict_one()`/
	/// `needs_capacity_eviction()` -- see the module doc for the bug that
	/// caused.
	fn settle_one_access(&mut self) {
		while self.one_access_used > self.one_access_capacity {
			let Some(key) = self.one_access_queue.pop_back() else { break };
			let Some(entry) = self.entries.get(&key).copied() else { continue };
			let size = entry.migrating();

			self.one_access_used = self.one_access_used.saturating_sub(size);

			self.main_slow.push_front(key);

			if let Some(stored) = self.entries.get_mut(&key) {
				stored.queue = Queue::Main;
				stored.tier = Some(Tier::Slow);
				stored.accessed = false;
			}

			self.slow_used += size;

			// No migration. Both the one-access queue and the main queue's
			// slow segment live in PMEM, so this splice moves the key
			// between two lists without moving a single byte -- the whole
			// point of pairing a slow-tier one-access queue with the
			// reprieve. The fast-admission reprieve variants must push a
			// `Tier::Slow` migration here (a real DRAM->PMEM copy per
			// aged-out object); this design gets the same behaviour for
			// free.
		}
	}
}

impl PolicyStack for S3FifoLazyDemotionReprieveHybridStack {
	fn is_policy(&self, policy: &PaperPolicy) -> bool {
		matches!(policy, PaperPolicy::S3FifoLazyDemotionReprieveHybrid(ratio) if *ratio == self.one_access_ratio)
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
			self.resize_key(key, size);
			self.touch(key);
			return;
		}

		self.one_access_queue.push_front(key);
		self.entries.insert(key, S3FifoEntry { dram_resident, queue: Queue::OneAccess,
			tier: None,
			size,
			accessed: false,
		});
		self.one_access_used += (size as CacheSize).saturating_sub(dram_resident as CacheSize);

		self.settle_one_access();
	}

	fn update(&mut self, key: HashedKey) {
		if self.entries.contains_key(&key) {
			self.touch(key);
		}
	}

	fn remove(&mut self, key: HashedKey) {
		let Some(entry) = self.entries.remove(&key) else { return };
		let size = entry.migrating();

		match entry.queue {
			Queue::OneAccess => {
				self.one_access_queue.remove(&key);
				self.one_access_used = self.one_access_used.saturating_sub(size);
			},

			Queue::Main => match entry.tier {
				Some(Tier::Fast) => {
					self.main_fast.remove(&key);
					self.fast_used = self.fast_used.saturating_sub(size);
				},

				Some(Tier::Slow) => {
					self.main_slow.remove(&key);
					self.slow_used = self.slow_used.saturating_sub(size);

						},

				None => {},
			},
		}
	}

	fn resize(&mut self, max_size: CacheSize) {
		self.one_access_capacity = (self.one_access_ratio * max_size as f64) as CacheSize;
		self.settle_one_access();
		self.settle_fast_tier();
	}

	fn clear(&mut self) {
		self.one_access_queue.clear();
		self.main_fast.clear();
		self.main_slow.clear();
		self.entries.clear();

		self.one_access_used = 0;
		self.fast_used = 0;
		self.slow_used = 0;
		self.migrations.clear();
	}

	fn evict_one(&mut self) -> Option<HashedKey> {
		// The one-access queue never reaches here -- its own capacity
		// pressure is relieved synchronously by `settle_one_access()` (see
		// the module doc), the same way the main queue's fast/slow boundary
		// is settled by `settle_fast_tier()` rather than through eviction.
		// This is purely the main queue's ordinary tail loop.
		loop {
			// The slow tail is the real eviction candidate; fall back to
			// the fast tail only when nothing has ever been demoted.
			let (key, from_slow) = match self.main_slow.back().copied() {
				Some(key) => (key, true),
				None => (self.main_fast.back().copied()?, false),
			};

			let accessed = self.entries.get(&key).map(|entry| entry.accessed).unwrap_or(false);

			if accessed {
				self.give_second_chance(key);
				continue;
			}

			if from_slow {
				self.main_slow.pop_back();
			} else {
				self.main_fast.pop_back();
			}

			let removed = self.entries.remove(&key);
			let size = removed.map(|entry| entry.migrating()).unwrap_or(0);

			if from_slow {
				self.slow_used = self.slow_used.saturating_sub(size);
				} else {
				self.fast_used = self.fast_used.saturating_sub(size);
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

	// The one-access queue counts toward the SLOW gauges here, not the fast
	// ones. The fast-admission variants add `one_access_used` to
	// `fast_bytes_used` because their one-access queue really is DRAM; this
	// variant's is PMEM, so attributing it to the fast tier would over-report
	// DRAM usage by the whole one-access budget and under-report PMEM by the
	// same amount. `tier_of` already reports `Tier::Slow` for these keys --
	// these gauges must agree with it.
	fn fast_bytes_used(&self) -> CacheSize {
		self.fast_used
	}

	fn slow_bytes_used(&self) -> CacheSize {
		self.slow_used + self.one_access_used
	}

	fn fast_object_count(&self) -> usize {
		self.main_fast.len()
	}

	fn slow_object_count(&self) -> usize {
		self.main_slow.len() + self.one_access_queue.len()
	}
}

#[cfg(test)]
mod tests {
	use super::*;

	fn drain(stack: &mut S3FifoLazyDemotionReprieveHybridStack) -> Vec<(HashedKey, Tier)> {
		stack.drain_tier_migrations()
	}

	/// Admits `key` and immediately re-accesses it -- this variant's only route
	/// into the fast tier. A brand-new set lands in the (PMEM) one-access
	/// queue; the second access is what runs `promote_from_one_access`, and
	/// with it `settle_fast_tier`. The promotion clears the reference bit, so
	/// keys built this way are all genuinely demotable and the lazy-demotion
	/// reprieve stays out of the fast-tier watermark tests' way.
	fn promote(stack: &mut S3FifoLazyDemotionReprieveHybridStack, key: HashedKey, size: ObjectSize) {
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
	// ── the set protocol ──────────────────────────────────────────────
	//
	// `insert()` is what a `set()` routes to, and its behaviour depends
	// entirely on whether the key is already tracked. These three cases are
	// the protocol, pinned explicitly rather than inferred from the
	// admission/promotion tests below -- they are what differs from the
	// fast-admission variants, where a brand-new set lands in DRAM.

	#[test]
	fn set_of_a_brand_new_key_lands_in_the_one_access_queue_in_the_slow_tier() {
		let mut stack = S3FifoLazyDemotionReprieveHybridStack::new(1.0, 1_000, 1_000);

		stack.insert(1, 10);

		assert_eq!(stack.tier_of(1), Some(Tier::Slow), "paper-literal admission: a new object goes to the slow tier");
		assert_eq!(stack.slow_bytes_used(), 10);
		assert_eq!(stack.fast_bytes_used(), 0, "a brand-new set must never touch the DRAM budget");
		assert_eq!(
			drain(&mut stack), Vec::new(),
			"no migration: the API layer already built the value as Slow (see this feature's admission_tier)",
		);
	}

	#[test]
	fn set_of_an_existing_one_access_key_counts_as_a_reaccess_and_promotes_it() {
		// A set is an access. A key sitting in the one-access queue that is
		// set again has therefore demonstrated reuse, and is promoted to the
		// main queue's fast segment exactly as a get would promote it.
		let mut stack = S3FifoLazyDemotionReprieveHybridStack::new(1.0, 1_000, 1_000);

		stack.insert(1, 10);
		drain(&mut stack);
		assert_eq!(stack.tier_of(1), Some(Tier::Slow));

		stack.insert(1, 10);

		assert_eq!(stack.tier_of(1), Some(Tier::Fast), "a re-set is a reaccess, so it promotes");
		assert_eq!(
			drain(&mut stack), vec![(1, Tier::Fast)],
			"a real PMEM->DRAM migration: unlike the fast-admission variants the bytes genuinely move here",
		);
	}

	#[test]
	fn set_of_an_existing_main_slow_key_marks_the_bit_without_migrating() {
		// Once in the main queue, a set behaves like any other access under
		// lazy promotion: it sets the reference bit and nothing else. The
		// key is only returned to DRAM later, by the demotion-boundary
		// reprieve or the eviction-time second chance.
		let mut stack = S3FifoLazyDemotionReprieveHybridStack::new(0.01, 1_000, 10);

		stack.insert(1, 10);
		stack.update(1);          // promote into main_fast
		stack.insert(2, 10);
		stack.update(2);          // promotes 2, demoting 1 to main_slow
		drain(&mut stack);
		assert_eq!(stack.tier_of(1), Some(Tier::Slow), "precondition: key 1 is in the main queue's slow segment");

		stack.insert(1, 10);

		assert_eq!(stack.tier_of(1), Some(Tier::Slow), "a set on a main-queue key does not itself promote");
		assert_eq!(drain(&mut stack), Vec::new(), "and moves no bytes");
	}

	#[test]
	fn admission_always_lands_in_one_access_queue_slow() {
		let mut stack = S3FifoLazyDemotionReprieveHybridStack::new(1.0, 1_000, 1_000);

		stack.insert(1, 10);
		stack.insert(2, 10);

		// Slow, not Fast: this variant's one-access queue lives in PMEM.
		assert_eq!(stack.tier_of(1), Some(Tier::Slow));
		assert_eq!(stack.tier_of(2), Some(Tier::Slow));
		assert_eq!(drain(&mut stack), Vec::new());
	}

	#[test]
	fn a_key_aging_out_without_reaccess_is_moved_to_slow_instead_of_evicted() {
		// one_access_capacity = 0.01 * 1_000 = 10 -- fits exactly one 10-byte
		// key. Admitting a second pushes one_access_used to 20 > 10,
		// synchronously reprieving the oldest (key 1) from insert() itself.
		let mut stack = S3FifoLazyDemotionReprieveHybridStack::new(0.01, 1_000, 1_000);

		stack.insert(1, 10);
		drain(&mut stack);
		assert_eq!(stack.tier_of(1), Some(Tier::Slow), "still in the one-access queue (which is slow here)");

		stack.insert(2, 10);

		assert!(stack.contains(1), "the key must still be tracked, not gone");
		assert_eq!(stack.tier_of(1), Some(Tier::Slow), "aged-out key should land directly in the main queue's slow tier");
		assert_eq!(stack.tier_of(2), Some(Tier::Slow), "the newer key stays in the one-access queue (slow)");

		let migrations = drain(&mut stack);
		assert_eq!(migrations, Vec::new(), "no migration: one-access queue and main_slow are both PMEM, so the splice moves no bytes");
	}

	#[test]
	fn a_reprieved_key_can_still_be_promoted_by_a_later_access() {
		let mut stack = S3FifoLazyDemotionReprieveHybridStack::new(0.01, 1_000, 1_000);

		// one_access_capacity = 0.01 * 1_000 = 10, fits exactly one key.
		// insert(2) pushes past it, reprieving key 1 (the oldest); insert(3)
		// pushes past it again, reprieving key 2 -- leaving key 3 sitting
		// safely in the one-access queue (untouched, under capacity) and
		// both 1 and 2 in main_slow, in that order: main_slow = [2, 1]
		// (2 at the front/freshest, 1 at the tail).
		stack.insert(1, 10);
		stack.insert(2, 10);
		stack.insert(3, 10);
		drain(&mut stack);

		assert_eq!(stack.tier_of(1), Some(Tier::Slow));
		assert_eq!(stack.tier_of(2), Some(Tier::Slow));
		assert_eq!(stack.tier_of(3), Some(Tier::Slow), "still sitting untouched in the one-access queue (slow)");

		// Re-access key 1 (the tail): sets the reference bit but must not
		// itself move or migrate it yet.
		stack.update(1);
		assert_eq!(stack.tier_of(1), Some(Tier::Slow));
		assert_eq!(drain(&mut stack), Vec::new());

		// The tail check finds key 1's bit set and gives it a second
		// chance instead of evicting it; eviction then proceeds to the
		// real (still genuinely cold) tail, key 2, in the same call.
		let evicted = stack.evict_one();

		assert_eq!(evicted, Some(2));
		assert_eq!(stack.tier_of(1), Some(Tier::Fast), "the reprieved key should have been promoted via the ordinary second chance");
	}

	#[test]
	fn reprieve_does_not_disturb_existing_fast_key_order() {
		// A comfortable one-access budget (ratio 1.0) during setup, so keys
		// 1-3 each safely survive their own insert()'s settle_one_access
		// before the very next line's update() promotes them via touch().
		// fast_capacity is generous so promoted keys stay put. Note this
		// variant does NOT subtract one_access_capacity from it -- the
		// one-access queue is PMEM and competes for nothing here.
		let mut stack = S3FifoLazyDemotionReprieveHybridStack::new(1.0, 1_000, 10_000);

		for key in 1..=3u64 {
			stack.insert(key, 10);
			stack.update(key);
		}
		drain(&mut stack);

		assert_eq!(stack.tier_of(1), Some(Tier::Fast));
		assert_eq!(stack.tier_of(2), Some(Tier::Fast));
		assert_eq!(stack.tier_of(3), Some(Tier::Fast));

		// Admit a fourth key that stays in the one-access queue (never
		// touched), then shrink the one-access budget to 0 -- forcing
		// settle_one_access to move it into main_slow synchronously, from
		// within this resize() call.
		stack.insert(4, 10);
		assert_eq!(stack.tier_of(4), Some(Tier::Slow), "still sitting untouched in the one-access queue (slow)");

		stack.resize(0);

		assert_eq!(stack.tier_of(4), Some(Tier::Slow));
		assert_eq!(drain(&mut stack), Vec::new(), "slow->slow splice: no migration");

		// The three original fast keys must all still be Fast and still in
		// their original oldest-first order -- shrink the fast budget to 0
		// and confirm every one demotes in that order, none skipped.
		stack.resize_fast_tier(0);
		let migrations = drain(&mut stack);

		assert_eq!(migrations, vec![(1, Tier::Slow), (2, Tier::Slow), (3, Tier::Slow)], "demotion order must be oldest-first, and no fast key may be silently skipped");
		assert_eq!(stack.tier_of(1), Some(Tier::Slow));
		assert_eq!(stack.tier_of(2), Some(Tier::Slow));
		assert_eq!(stack.tier_of(3), Some(Tier::Slow));
		assert_eq!(stack.fast_bytes_used(), 0);
	}

	#[test]
	fn reprieve_never_counts_toward_fast_bytes_used() {
		let mut stack = S3FifoLazyDemotionReprieveHybridStack::new(0.0, 1_000, 1_000);

		stack.insert(1, 10);

		assert_eq!(stack.fast_bytes_used(), 0, "a reprieved key must never be counted as fast, even transiently");
		assert_eq!(stack.slow_bytes_used(), 10);
	}

	#[test]
	fn an_accessed_fast_boundary_key_is_reprieved_at_demotion_time_not_demoted() {
		// Sized so a triggered pass drains to a low watermark that still holds
		// one of these two 10-byte objects: the accessed key is reprieved and
		// exactly one object -- the un-accessed one -- is demoted. (Was a
		// hard-coded 10, correct back when a pass drained to the ceiling; under
		// the watermarks a 10-byte ceiling triggers at 9 and drains to 7, so
		// even a lone resident object would be demoted straight back out and
		// there would be no boundary key left to reprieve.)
		let mut stack = S3FifoLazyDemotionReprieveHybridStack::new(1.0, 1_000, capacity_holding(10));

		stack.insert(1, 10);
		stack.update(1);
		drain(&mut stack);

		stack.update(1);
		assert_eq!(drain(&mut stack), Vec::new());

		stack.insert(2, 10);
		stack.update(2);
		let migrations = drain(&mut stack);

		assert_eq!(stack.tier_of(1), Some(Tier::Fast));
		assert_eq!(stack.tier_of(2), Some(Tier::Slow));
		assert_eq!(migrations, vec![(2, Tier::Slow)]);
	}

	fn build_five_key_stack() -> S3FifoLazyDemotionReprieveHybridStack {
		let mut stack = S3FifoLazyDemotionReprieveHybridStack::new(1.0, 1_000, 20);

		for key in 1..=5u64 {
			stack.insert(key, 10);
			stack.update(key);
		}

		drain(&mut stack);
		stack
	}

	#[test]
	fn evict_one_gives_an_accessed_slow_key_a_second_chance() {
		// Same re-sizing as the reprieve test above: the low watermark must
		// still hold one 10-byte object, or the second-chance key would be
		// demoted straight back out by the `settle_fast_tier` that
		// `give_second_chance` runs.
		let mut stack = S3FifoLazyDemotionReprieveHybridStack::new(1.0, 1_000, capacity_holding(10));

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
	fn evict_one_falls_back_to_the_fast_tail_when_nothing_has_ever_been_demoted() {
		let mut stack = S3FifoLazyDemotionReprieveHybridStack::new(1.0, 1_000, 10_000);

		for key in 1..=3u64 {
			stack.insert(key, 10);
			stack.update(key);
		}
		drain(&mut stack);

		assert_eq!(stack.slow_object_count(), 0, "nothing should have been demoted yet");

		// With main_slow empty, the oldest fast key is the only candidate.
		assert_eq!(stack.evict_one(), Some(1));
		assert!(!stack.contains(1));
		assert_eq!(stack.fast_bytes_used(), 20);
	}

	#[test]
	fn remove_and_clear_reset_bookkeeping() {
		let mut stack = S3FifoLazyDemotionReprieveHybridStack::new(1.0, 1_000, 1_000);

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
	fn fast_and_slow_gauges_count_the_one_access_queue_on_the_slow_side() {
		// The inverse of the fast-admission variants' equivalent test. Two
		// keys sitting in the one-access queue are PMEM-resident here, so
		// they must show up in the slow gauges and leave the DRAM gauges at
		// zero -- otherwise `hybrid_stats()` would report a fast tier that
		// is entirely PMEM.
		let mut stack = S3FifoLazyDemotionReprieveHybridStack::new(1.0, 1_000, 1_000);

		stack.insert(1, 10);
		stack.insert(2, 10);
		drain(&mut stack);

		assert_eq!(stack.fast_bytes_used(), 0);
		assert_eq!(stack.slow_bytes_used(), 20);
		assert_eq!(stack.fast_object_count(), 0);
		assert_eq!(stack.slow_object_count(), 2);
	}

	/// (a) The trigger is a strict `>`, so usage sitting right *on* the high
	/// watermark -- the largest usage that is not over it -- must leave the
	/// tier completely alone.
	#[test]
	fn fast_usage_at_the_high_watermark_triggers_no_demotion() {
		let fast_capacity: CacheSize = 1_000;
		let high = watermarks::high_bytes(fast_capacity);

		let mut stack = S3FifoLazyDemotionReprieveHybridStack::new(1.0, 100_000, fast_capacity);

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
		// `promote` empties the one-access queue, which this variant counts on
		// the slow side, so the slow gauge is genuinely zero rather than merely
		// holding the pre-promotion copies.
		assert_eq!(stack.slow_bytes_used(), 0);
		assert_eq!(stack.slow_object_count(), 0);
	}

	/// (b) One byte past the high watermark -- the smallest possible overshoot
	/// -- must fire a pass, and it must take `main_fast`'s oldest key rather
	/// than the key that just arrived.
	#[test]
	fn fast_usage_above_the_high_watermark_triggers_a_demotion_pass() {
		let fast_capacity: CacheSize = 1_000;
		let high = watermarks::high_bytes(fast_capacity);

		let mut stack = S3FifoLazyDemotionReprieveHybridStack::new(1.0, 100_000, fast_capacity);

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

		let mut stack = S3FifoLazyDemotionReprieveHybridStack::new(1.0, 100_000, fast_capacity);

		// Exactly one object past the high watermark, so precisely one pass
		// fires -- with plenty of resident objects for it to chew through before
		// it reaches the low watermark.
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
		// ...and genuinely past the old stopping point, not merely back under
		// the ceiling. The two coincide only in the degenerate
		// `FAST_TIER_LOW_WATERMARK=1.0` configuration, which exists precisely to
		// restore the old drain-to-ceiling behaviour.
		assert!(
			stack.fast_bytes_used() < fast_capacity || watermarks::low() == 1.0,
			"a pass must drain down to the low watermark, not merely under the {}-byte ceiling",
			fast_capacity,
		);
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

		let mut stack = S3FifoLazyDemotionReprieveHybridStack::new(1.0, 100_000, fast_capacity);

		for key in 1..=count {
			promote(&mut stack, key, size);
		}

		let migrations = drain(&mut stack);
		let demoted = migrations.iter().filter(|(_, tier)| *tier == Tier::Slow).count() as CacheSize;

		let fast_objects = stack.fast_object_count() as CacheSize;
		let slow_objects = stack.slow_object_count() as CacheSize;

		// Nothing was inserted, evicted or resized mid-pass, so every object is
		// still tracked, still `size` bytes, and still on exactly one side of the
		// fast/slow line. The one-access queue is empty here (every key was
		// promoted out of it), so the slow gauges -- which count it in this
		// variant -- are purely the demoted main-queue tail.
		assert!(fast_objects > 0 && slow_objects > 0);
		assert_eq!(fast_objects + slow_objects, count);
		assert_eq!(stack.len() as CacheSize, count);

		assert_eq!(stack.fast_bytes_used(), fast_objects * bytes);
		assert_eq!(stack.slow_bytes_used(), slow_objects * bytes);
		assert_eq!(stack.fast_bytes_used() + stack.slow_bytes_used(), count * bytes);

		// Exactly one demotion migration per object that actually moved.
		assert_eq!(demoted, slow_objects);

		// And the aggregate counts agree with the per-key tier tags.
		let tagged_fast = (1..=count).filter(|key| stack.tier_of(*key) == Some(Tier::Fast)).count();
		let tagged_slow = (1..=count).filter(|key| stack.tier_of(*key) == Some(Tier::Slow)).count();

		assert_eq!(tagged_fast as CacheSize, fast_objects);
		assert_eq!(tagged_slow as CacheSize, slow_objects);
	}

	// -- shared-metadata DRAM reservation ------------------------------
	//
	// `shared_overhead` is `0` in every test above -- none of them call
	// `with_shared_overhead` -- so `effective_main_fast_capacity()` is still
	// exactly `fast_capacity` there and not one of their expectations moves.
	// The five below are the only tests that turn the reservation on.

	/// The reservation shrinks the effective budget, and a stack carrying it
	/// demotes where an otherwise identical stack without it does not.
	#[test]
	fn shared_overhead_reserves_dram_and_demotes_earlier() {
		const CAPACITY: CacheSize = 1_000;
		const OVERHEAD: CacheSize = 200;

		// Two tracked keys reserve 2 x 200 = 400, dropping the effective value
		// budget from 1_000 to 600. A payload sized one byte past 600's high
		// watermark -- but still comfortably inside 1_000's -- can only be
		// demoted if the reservation is honoured.
		let size = (watermarks::high_bytes(600) + 1) as ObjectSize;
		assert!(
			size as CacheSize + 1 <= watermarks::high_bytes(CAPACITY),
			"the payload must still fit the un-reserved budget, or the contrast below proves nothing",
		);

		// Without the reservation both keys stay in DRAM.
		let mut plain = S3FifoLazyDemotionReprieveHybridStack::new(1.0, 100_000, CAPACITY);

		promote(&mut plain, 1, 1);
		drain(&mut plain);
		promote(&mut plain, 2, size);

		assert_eq!(plain.effective_main_fast_capacity(), CAPACITY, "no reservation configured");
		assert_eq!(drain(&mut plain), vec![(2, Tier::Fast)], "only the promotion; nothing demoted");
		assert_eq!(plain.tier_of(1), Some(Tier::Fast));
		assert_eq!(plain.tier_of(2), Some(Tier::Fast));

		// With it, the very same second promotion trips a pass.
		let mut stack = S3FifoLazyDemotionReprieveHybridStack::new(1.0, 100_000, CAPACITY)
			.with_shared_overhead(OVERHEAD);

		promote(&mut stack, 1, 1);

		assert_eq!(stack.effective_main_fast_capacity(), CAPACITY - OVERHEAD, "one tracked key so far");
		assert_eq!(stack.tier_of(1), Some(Tier::Fast), "1 byte is nowhere near an 800-byte budget");
		drain(&mut stack);

		promote(&mut stack, 2, size);
		let migrations = drain(&mut stack);

		assert_eq!(stack.effective_main_fast_capacity(), CAPACITY - 2 * OVERHEAD);
		// The pass drains to `low_bytes(600)`, and the surviving key 2 -- at
		// `high_bytes(600) + 1` bytes, with `low() <= high()` -- is still above
		// that, so the oldest key goes first and key 2 follows it straight out.
		assert_eq!(migrations, vec![(1, Tier::Slow), (2, Tier::Slow)]);
		assert_eq!(stack.tier_of(1), Some(Tier::Slow));
		assert_eq!(stack.tier_of(2), Some(Tier::Slow));
		assert_eq!(stack.fast_bytes_used(), 0);
		assert_eq!(stack.len(), 2, "demotion, never eviction");
	}

	/// The reservation composes *with* the watermarks rather than replacing
	/// them: a triggered pass drains to `low_bytes(capacity - reserved)`, not
	/// to `low_bytes(capacity)`.
	#[test]
	fn a_reserved_pass_drains_to_the_low_watermark_of_the_reserved_budget() {
		const CAPACITY: CacheSize = 690;
		const OVERHEAD: CacheSize = 25;
		const COUNT: CacheSize = 20;

		let size: ObjectSize = 10;
		let bytes = size as CacheSize;

		// Built under a budget nothing can trip, so the tracked count -- and
		// with it the 20 x 25 = 500-byte reservation -- is fully settled before
		// the single pass under test fires.
		let mut stack = S3FifoLazyDemotionReprieveHybridStack::new(1.0, 100_000, 1_000_000)
			.with_shared_overhead(OVERHEAD);

		for key in 1..=COUNT {
			promote(&mut stack, key, size);
		}
		drain(&mut stack);

		assert_eq!(stack.fast_bytes_used(), COUNT * bytes);
		assert_eq!(stack.slow_object_count(), 0, "nothing demoted during setup");

		// 690 - 500 = 190 of effective budget against 200 bytes of value. 200
		// exceeds `high_bytes(190)` at *any* watermark ratio (that is at most
		// 190), so exactly one pass fires here.
		stack.resize_fast_tier(CAPACITY);
		let migrations = drain(&mut stack);

		let effective = CAPACITY - COUNT * OVERHEAD;

		assert_eq!(effective, 190);
		assert_eq!(stack.effective_main_fast_capacity(), effective);

		// The pass halts at the first whole-object multiple at or below the
		// *reserved* budget's low watermark.
		let drain_target = watermarks::low_bytes(effective);
		let expected_used = drain_target - drain_target % bytes;

		assert_eq!(stack.fast_bytes_used(), expected_used);
		assert!(stack.fast_bytes_used() <= drain_target);

		// The load-bearing pair. Had the watermarks been taken against the raw
		// capacity instead, this pass would not have fired at all (200 bytes is
		// already under `high_bytes(690)`), and even once fired it would have
		// stopped hundreds of bytes higher.
		assert!(
			COUNT * bytes <= watermarks::high_bytes(CAPACITY),
			"without the reservation this pass would never have triggered",
		);
		assert!(
			expected_used < watermarks::low_bytes(CAPACITY),
			"drained to {} bytes, which is not below low_bytes({}) = {} -- the reservation missed the drain target",
			expected_used,
			CAPACITY,
			watermarks::low_bytes(CAPACITY),
		);

		let demoted = migrations.iter().filter(|(_, tier)| *tier == Tier::Slow).count() as CacheSize;

		assert!(demoted > 0);
		assert_eq!(demoted, (COUNT * bytes - expected_used) / bytes);
	}

	/// Every counter still agrees with the per-key tier tags after a pass that
	/// fired because of the reservation: the same per-demotion bookkeeping ran
	/// once per demoted object, no more and no less.
	#[test]
	fn counters_stay_consistent_across_a_reserved_watermark_pass() {
		const CAPACITY: CacheSize = 690;
		const OVERHEAD: CacheSize = 25;
		const COUNT: CacheSize = 20;

		let size: ObjectSize = 10;
		let bytes = size as CacheSize;

		let mut stack = S3FifoLazyDemotionReprieveHybridStack::new(1.0, 100_000, 1_000_000)
			.with_shared_overhead(OVERHEAD);

		for key in 1..=COUNT {
			promote(&mut stack, key, size);
		}
		drain(&mut stack);

		stack.resize_fast_tier(CAPACITY);
		let migrations = drain(&mut stack);
		let demoted = migrations.iter().filter(|(_, tier)| *tier == Tier::Slow).count() as CacheSize;

		let fast_objects = stack.fast_object_count() as CacheSize;
		let slow_objects = stack.slow_object_count() as CacheSize;

		// Nothing was inserted, evicted, resized or re-accessed mid-pass.
		// `promote` emptied the one-access queue, so the slow gauges -- which
		// count it in this variant -- are purely the demoted main-queue tail.
		assert!(fast_objects > 0 && slow_objects > 0);
		assert_eq!(fast_objects + slow_objects, COUNT);
		assert_eq!(stack.len() as CacheSize, COUNT, "the reservation demotes; it never evicts");

		assert_eq!(stack.fast_bytes_used(), fast_objects * bytes);
		assert_eq!(stack.slow_bytes_used(), slow_objects * bytes);
		assert_eq!(stack.fast_bytes_used() + stack.slow_bytes_used(), COUNT * bytes);

		assert_eq!(demoted, slow_objects, "exactly one migration per object that actually moved");

		let tagged_fast = (1..=COUNT).filter(|key| stack.tier_of(*key) == Some(Tier::Fast)).count();
		let tagged_slow = (1..=COUNT).filter(|key| stack.tier_of(*key) == Some(Tier::Slow)).count();

		assert_eq!(tagged_fast as CacheSize, fast_objects);
		assert_eq!(tagged_slow as CacheSize, slow_objects);

		// And the reservation is still exactly what the (unchanged) tracked
		// count says: demotion moves keys between lists, it does not untrack
		// them, so the effective budget must not have drifted mid-pass.
		assert_eq!(stack.effective_main_fast_capacity(), CAPACITY - COUNT * OVERHEAD);
	}

	/// A one-access key's *value* is PMEM and adds nothing to `fast_used`, but
	/// its object-hashtable entry and its `one_access_queue`/`entries` nodes
	/// are DRAM all the same -- so `reserved_overhead` counts it
	/// (`entries.len()`, not `main_fast.len()`).
	#[test]
	fn one_access_keys_are_charged_even_though_their_values_live_in_pmem() {
		const CAPACITY: CacheSize = 1_000;
		const OVERHEAD: CacheSize = 100;

		let mut stack = S3FifoLazyDemotionReprieveHybridStack::new(1.0, 100_000, CAPACITY)
			.with_shared_overhead(OVERHEAD);

		promote(&mut stack, 1, 300);
		drain(&mut stack);

		assert_eq!(stack.tier_of(1), Some(Tier::Fast));
		assert_eq!(stack.effective_main_fast_capacity(), CAPACITY - OVERHEAD, "one tracked key");

		// Seven admissions that are never re-accessed, so they all sit
		// untouched in the (PMEM) one-access queue.
		for key in 2..=8u64 {
			stack.insert(key, 10);
		}

		assert_eq!(stack.fast_bytes_used(), 300, "not one of them added a DRAM value byte");
		assert_eq!(stack.slow_object_count(), 7, "all still in the one-access queue");
		assert_eq!(
			stack.effective_main_fast_capacity(), CAPACITY - 8 * OVERHEAD,
			"...but all eight tracked keys are charged, so the DRAM budget is 1_000 - 800 = 200",
		);

		// `insert` settles the one-access queue, not the fast tier, so the
		// squeezed budget only bites at the next `settle_fast_tier`.
		// Re-applying the same capacity is the cheapest way to run one.
		stack.resize_fast_tier(CAPACITY);
		let migrations = drain(&mut stack);

		// 300 > high_bytes(200) at any watermark ratio, so the pass fires and
		// the main queue's only fast resident is demoted -- by keys whose own
		// bytes never entered DRAM.
		assert_eq!(migrations, vec![(1, Tier::Slow)]);
		assert_eq!(stack.tier_of(1), Some(Tier::Slow));
		assert_eq!(stack.fast_bytes_used(), 0);
		assert_eq!(stack.len(), 8, "every key is still tracked");
	}

	/// One key's reservation alone exceeding the whole fast budget saturates
	/// the effective value budget to `0`: the key demotes on promotion and is
	/// never evicted.
	#[test]
	fn shared_overhead_exceeding_capacity_demotes_all_but_never_evicts() {
		let mut stack = S3FifoLazyDemotionReprieveHybridStack::new(1.0, 100_000, 50)
			.with_shared_overhead(100);

		promote(&mut stack, 1, 10);
		let migrations = drain(&mut stack);

		assert_eq!(stack.effective_main_fast_capacity(), 0, "50 saturating_sub 100");
		// One migration, not two: `promote_from_one_access`' guard sees the
		// key already retagged `Slow` by the `settle_fast_tier` it just ran and
		// correctly suppresses its own `Tier::Fast` push.
		assert_eq!(migrations, vec![(1, Tier::Slow)]);
		assert_eq!(stack.tier_of(1), Some(Tier::Slow));
		assert_eq!(stack.fast_bytes_used(), 0);
		assert_eq!(stack.slow_bytes_used(), 10);

		// Still tracked: the DRAM budget only ever demotes (terminal eviction
		// stays governed solely by `max_size`).
		assert_eq!(stack.len(), 1);
		assert!(!stack.needs_capacity_eviction());
	}
}
