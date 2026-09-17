/*
 * Copyright (c) Kia Shakiba
 *
 * This source code is licensed under the GNU AGPLv3 license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! `PolicyStack` over [`crate::merged_store::MergedStore`] -- the eviction
//! stack manager's view of a store that IS its own eviction stack.
//!
//! Every other stack in this directory owns a structure keyed by `HashedKey`
//! that sits beside the object map. This one owns nothing: it holds the same
//! `Arc` the object map is behind and forwards to it. That makes it the only
//! stack whose bookkeeping cannot drift from the map, and the only one where a
//! `PolicyStack` method and an `ObjectStore` method can be the same operation.
//!
//! # The one surprising method
//!
//! `evict_one` does NOT remove anything.
//!
//! `PolicyWorker::apply_evictions` pairs `policy_stack.evict_one()` with
//! `erase(objects, .., Some(EraseKey::Hashed(key)))`, on the assumption that
//! those two touch different structures: the stack drops its own row, then
//! `erase` drops the map's. Here they are one structure, so doing both would
//! mean the slot is already gone by the time `erase` looks for it --
//! `MergedStore::take` would return `None`, `erase` would report
//! `KeyNotFound`, and `apply_evictions` would `continue` without ever
//! decrementing `status`. The loop would then spin on a cache it believes is
//! still over capacity, freeing nothing.
//!
//! So `evict_one` NOMINATES the victim -- the globally oldest key by the
//! store's order, `SHARDS` atomic loads and no lock -- and the removal happens
//! exactly once, inside `erase`'s `take`, which unlinks it from that order and
//! reverses its tier accounting in the same operation.
//!
//! # The other two, which are the same mistake in reverse
//!
//! `remove` and `clear` are no-ops.
//!
//! For every other stack these drop the stack's OWN bookkeeping while the
//! object map keeps its entry, and the worker calls them AFTER the API thread
//! has already erased from the map -- `PaperCache::del` erases synchronously
//! and only then broadcasts `Del`; `PaperCache::wipe` calls `objects.clear()`
//! synchronously and only then broadcasts `Wipe`. Two structures, two removals,
//! one per structure.
//!
//! Here the map IS the stack, so the API thread's erase already did the whole
//! job and the worker's call is a SECOND removal of the same thing. That is not
//! merely redundant. Both events are queued, and the worker may not drain them
//! for up to a poll interval, so a `set()` landing in the window is destroyed:
//!
//! ```text
//!   API thread:     wipe()          -> map emptied, Wipe queued
//!   API thread:     set(k, v)       -> k live in the map, status counts it
//!   worker thread:  handle_wipe()   -> stack.clear() -> k DESTROYED
//!   API thread:     get(k)          -> KeyNotFound, though set() returned Ok
//! ```
//!
//! and `status` can never be corrected, because `del(k)` now finds nothing to
//! erase -- so `used_size` climbs permanently toward `max_size` on a cache that
//! is holding less than it thinks. `handle_expire` already carries exactly this
//! guard (`object_exists`) for exactly this race against the TTL worker;
//! `handle_del` and `handle_wipe` did not need one until the map and the stack
//! became the same structure.
//!
//! So the rule for this handle is uniform: a method that would MUTATE the map
//! does nothing, because the caller already mutated it. `evict_one` nominates
//! rather than removes; `remove` and `clear` do nothing at all. Only the
//! methods that add information the map does not already have --
//! `insert_resident`'s size and tier accounting, `update`'s relink -- do work.
//!
//! # The order
//!
//! `update` is also where the eviction ORDER lives, because the order is
//! nothing but what a hit does to the map's own list. This handle resolves the
//! configured policy to a `MergedOrder` once, in `new`, installs it on the
//! store and keeps a copy: under `Lru` a hit relinks and promotes, under `Fifo`
//! it does nothing, which is the same nothing `FifoCompactHybridStack` gets
//! from not overriding `PolicyStack::update`, under `Clock` it sets a reference
//! bit and nothing else, and under `Lfu` it bumps a frequency counter and may
//! promote. A policy whose order the store does not implement is an ERROR: it
//! is refused, `new` returns `CacheError::PolicyNotImplemented`, and the cache
//! fails to construct -- see `new`.
//!
//! `Clock` is the one that changes the LOCK a hit takes. `Lru` runs
//! `MergedStore::touch`, which takes the shard WRITE lock to relink -- measured
//! as the merged store reaching 1.24x the DashMap arm's service time at sixteen
//! clients. `Clock` runs `MergedStore::mark_referenced`, which takes the READ
//! lock and performs one relaxed store; the relink it skips is paid for later,
//! by the eviction hand, under a write lock that path was taking anyway.
//!
//! # Tiering
//!
//! The seven tiering methods forward to the store rather than taking their
//! trait defaults, so the merged store is a genuine hybrid: it demotes at the
//! same drain target, emits the same `(key, Tier)` migrations for
//! `apply_tier_migrations` to physically perform, and publishes the same
//! gauges. Comparing it against `lru-compact-hybrid` is therefore
//! like-for-like, which comparing the untiered prototype against a tiered
//! stack was not.
//!
//! One order pays a cost the others do not, and it should be measured rather
//! than hidden. Under `Lfu` the store decides a brand-new key's tier itself
//! (`MergedStore::lfu_admission_tier`) while `PaperCache::set` has already
//! BUILT the bytes in DRAM -- `admission_latched()` here is deliberately false,
//! so `hybrid_policy::admission_tier` returns `Fast` for every new key. So a
//! latched merged-LFU run emits one corrective `(key, Slow)` migration per
//! admission. That is a real throughput cost on the SET path, of exactly the
//! kind the split stack's latched branch avoids by trusting a mirror -- and
//! trusting that mirror is what makes the split stack wrong under a burst. If
//! the cost proves dominant the fix is to let `admission_tier` read this
//! store's own latch directly (it already receives the store as `objects`),
//! which removes the mirror from the path rather than tolerating it.

use crate::{
	error::CacheError,
	merged_store::{MergedOrder, MergedStore},
	object::ObjectSize,
	worker::policy::policy_stack::{CacheSize, HashedKey, PolicyStack, Tier},
	PaperPolicy,
};

use std::sync::Arc;

/// `MergedStore::configure_tiering` still takes a high/low pair in parts per
/// million. The split stacks hold ONE continuous threshold, so both marks go to
/// the same `drain_target` ratio -- which makes the store arm above it and
/// drain back to it, with no band in between.
#[cfg(feature = "hybrid_cache_common")]
fn drain_target_ppm() -> u64 {
	(crate::worker::policy::policy_stack::drain_target::ratio() * 1_000_000.0) as u64
}

pub struct MergedStackHandle<K, V> {
	store: Arc<MergedStore<K, V>>,

	/// The configured policy, reported verbatim by `is_policy`. The merged
	/// store is a build-time object-map shape rather than a policy, so it
	/// answers to whichever policy the cache was configured with and never
	/// triggers a stack reconstruction.
	policy: PaperPolicy,

	/// The order the store was actually put into, which is the same value the
	/// store holds. Kept here as well so `update` -- the hottest method on this
	/// type, one call per cache hit -- answers from a plain field instead of an
	/// atomic load through the `Arc`.
	order: MergedOrder,
}

impl<K, V> MergedStackHandle<K, V> {
	/// Builds the stack over the same `Arc` the object map is behind, and --
	/// for a hybrid policy -- installs the fast-tier budget on the terms
	/// `init_policy_stack` gives the split hybrid stacks.
	///
	/// A flat policy leaves the store's fast capacity at its untiered
	/// sentinel, so `settle_fast_tier` short-circuits at its first comparison
	/// and a flat build pays nothing for the tiering machinery.
	pub fn new(
		store: Arc<MergedStore<K, V>>,
		policy: PaperPolicy,
		max_size: CacheSize,
	) -> Result<Self, CacheError> {
		#[cfg(feature = "hybrid_cache_common")]
		if policy.is_hybrid() {
			store.configure_tiering(
				// Same default fast-tier budget as every hybrid stack: 20% of
				// the overall cache size, runtime-adjustable afterward through
				// `resize_fast_tier`.
				(max_size as f64 * 0.2) as CacheSize,
				crate::object::overhead::get_hybrid_dram_shared_overhead(&policy) as CacheSize,
				drain_target_ppm(),
				drain_target_ppm(),
			);
		}

		let _ = max_size;

		// The merged store's eviction order IS the object map's own link
		// structure, so it implements the orders it has been taught and no
		// others -- recency, insertion order, CLOCK and now LFU. `is_policy`
		// still answers to the configured policy so nothing tries to
		// reconstruct a stack that has no separate existence.
		//
		// Anything else is REFUSED. There used to be a fallback here: a
		// `eprintln!` and then `MergedOrder::Lru`, so a merged build asked for
		// `lfu-compact-hybrid` ran LRU under an LFU label and reported a
		// plausible miss ratio filed under the wrong policy name. That is
		// deleted. A policy this build cannot honour is an error and the cache
		// does not come up.
		//
		// The reasoning that put the old warning on STDERR rather than through
		// `log` is worth keeping, because it is the argument for an error: the
		// warning before it was a `log::warn!`, and this crate installs no
		// logger anywhere (`log` is a dependency and `warn!` is used across the
		// tree, but nothing ever calls `set_logger`), so in the one binary
		// where this mislabelling could actually happen the warning had never
		// once been printed. A channel nobody reads cannot carry a warning that
		// matters -- so this one is not a warning at all.
		let order = MergedOrder::from_policy(&policy)?;

		store.set_order(order);

		Ok(MergedStackHandle { store, policy, order })
	}
}

impl<K, V> PolicyStack for MergedStackHandle<K, V>
where
	K: Send + Sync,
	V: Send + Sync,
{
	fn is_policy(&self, policy: &PaperPolicy) -> bool {
		*policy == self.policy
	}

	fn len(&self) -> usize {
		self.store.len()
	}

	fn contains(&self, key: HashedKey) -> bool {
		self.store.contains(key)
	}

	fn insert(&mut self, key: HashedKey, size: ObjectSize) {
		self.insert_resident(key, size, 0);
	}

	/// The object is already linked at the MRU end -- the API thread's
	/// `ObjectStore::insert` did that, since in this design inserting into the
	/// map IS inserting into the stack. What only the worker knows is the size
	/// and the DRAM-resident remainder, so that is what this records, and the
	/// shard settles against its fast budget once it has them.
	fn insert_resident(&mut self, key: HashedKey, size: ObjectSize, dram_resident: ObjectSize) {
		self.store.record_size(key, size, dram_resident);
	}

	/// A cache hit.
	///
	/// Under FIFO this is a no-op, which is the whole of the policy -- and it
	/// is the same no-op `FifoCompactHybridStack` gets by NOT overriding
	/// `PolicyStack::update`, whose default body is empty. The store's `touch`
	/// refuses under FIFO as well; the check is repeated here so the hot path
	/// does not pay for a call through the `Arc` to find that out.
	fn update(&mut self, key: HashedKey) {
		match self.order {
			MergedOrder::Lru => self.store.touch(key),

			// CLOCK's hit, and the reason this seam is worth having: one
			// relaxed store under the shard's READ lock. `mark_referenced` is
			// called directly rather than through `touch` so the hot path does
			// not re-test an order this handle already resolved at startup --
			// and `touch` refuses under CLOCK in any case, for the same reason
			// it refuses under FIFO.
			MergedOrder::Clock => self.store.mark_referenced(key),

			// LFU's hit, called directly for the same reason CLOCK's is. It is
			// the OPPOSITE trade: `bump` takes the shard WRITE lock, because a
			// frequency bump moves the slot between two bucket chains and there
			// is no read-lock formulation that leaves the buckets truthful.
			// LFU therefore does not inherit CLOCK's win, and this arm is where
			// that cost is paid -- see `MergedOrder::Lfu`.
			MergedOrder::Lfu => self.store.bump(key),

			MergedOrder::Fifo => {},
		}
	}

	/// Deliberate no-op -- see the module doc's second surprising-method note.
	fn remove(&mut self, _key: HashedKey) {}

	/// Deliberate no-op, for the same reason as `remove`.
	fn clear(&mut self) {}

	/// Nominates the victim WITHOUT removing it -- see the module doc. The
	/// removal is `erase`'s `take`, which is the same operation on the same
	/// structure.
	fn evict_one(&mut self) -> Option<HashedKey> {
		self.store.tail_key()
	}

	fn resize_fast_tier(&mut self, size: CacheSize) {
		self.store.resize_fast_tier(size);
	}

	fn drain_tier_migrations(&mut self) -> Vec<(HashedKey, Tier)> {
		self.store.drain_migrations()
	}

	fn dram_reserved_bytes(&self) -> CacheSize {
		self.store.dram_reserved_bytes()
	}

	fn fast_bytes_used(&self) -> CacheSize {
		self.store.fast_bytes_used()
	}

	fn slow_bytes_used(&self) -> CacheSize {
		self.store.slow_bytes_used()
	}

	fn fast_object_count(&self) -> usize {
		self.store.fast_object_count()
	}

	fn slow_object_count(&self) -> usize {
		self.store.slow_object_count()
	}
}

/// The acceptance test for design note 1f: the tier boundary is GLOBAL.
///
/// Every other tiering test in `merged_store` states a property of the merged
/// store alone. This one states a FIDELITY property, which is the only kind
/// that can fail for the reason 1f exists: it replays one sequence into
/// `MergedStackHandle` (32 shards, boundary reconstructed from stamps) and into
/// `LruCompactHybridStack` (one list, boundary by construction) and requires
/// the two to place every key in the same tier.
///
/// # Why it is built out of a SIZE skew, not just a recency skew
///
/// The budget that was replaced split `fast_capacity` EVENLY across the shards.
/// An even split of BYTES is only correct when the bytes are spread evenly, so
/// the way to break it is to make one shard hold far more bytes than another
/// while holding the MORE RECENT objects:
///
///   * shard 0 -- 200 COLD SMALL objects, inserted first, 512 B each;
///   * shard 1 -- 100 RECENT LARGE objects, inserted second, 8192 B each.
///
/// Globally the victims must be the oldest fast objects, which are all in
/// shard 0, and not one recent large object may be demoted. Under the per-shard
/// budget shard 1's share is `fast_capacity / 32` = 27,500 B -- room for three
/// large objects -- so it demotes about 97 of the 100 MOST RECENT objects in
/// the cache while shard 0 keeps objects that are older than every one of them.
/// The two placements are then nothing like each other, which is the point:
/// this test fails loudly on the pre-Phase-2 code and was checked doing so.
///
/// # Why the two structures are comparable at all
///
/// Both are fed the identical `(key, size, dram_resident)` calls through
/// `PolicyStack`, both settle at the same points (unconditional fast admission,
/// then a settle), and both hold the tier at the same `drain_target`, so
/// nothing but the boundary rule differs. The sizes are exact jemalloc size
/// classes, so the merged store's size-class-rounded `migrating()` and the
/// split stack's raw `size - dram_resident` are the same number and the
/// accounting cannot drift for a reason unrelated to tiering.
#[cfg(all(test, feature = "lru_compact_hybrid_cache"))]
mod global_demotion_fidelity {
	use super::*;

	use crate::{
		object::Object,
		worker::policy::policy_stack::{
			lru_compact_hybrid_stack::LruCompactHybridStack,
		},
		BufferDRAM,
	};

	use std::collections::HashSet;

	type Store = MergedStore<u64, BufferDRAM>;
	type Handle = MergedStackHandle<u64, BufferDRAM>;

	/// `MergedStore::shard_of` reads the top five bits of the key.
	const SHARD_BITS: u32 = 5;
	const SHARDS: u64 = 1 << SHARD_BITS;

	const COLD_SHARD: u64 = 0;
	const HOT_SHARD: u64 = 1;

	/// Both are exact jemalloc size classes, so `nallocx` rounds neither and
	/// the two structures charge the identical number of bytes per object.
	///
	/// These are ITEM sizes -- what the object COSTS -- and `value_len` below
	/// turns each into the value length that produces it.
	const SMALL: ObjectSize = 512;
	const LARGE: ObjectSize = 8192;

	const N_SMALL: u64 = 200;
	const N_LARGE: u64 = 100;

	/// Chosen so the budget bites in the middle of the cold objects: at the end
	/// of the sequence 120 of the 200 small objects are slow, 80 are still
	/// fast, and all 100 large ones are fast. A capacity that demoted all or
	/// none of the small objects would let a per-shard rule agree by accident.
	const FAST_CAPACITY: CacheSize = 880_000;

	fn mix(i: u64) -> HashedKey {
		i.wrapping_mul(0x9E37_79B9_7F4A_7C15)
	}

	/// A key that lands in shard `s`: the mixed value shifted clear of the
	/// shard field, then the shard written into it.
	fn in_shard(s: u64, i: u64) -> HashedKey {
		assert!(s < SHARDS);
		(mix(i) >> SHARD_BITS) | (s << (64 - SHARD_BITS))
	}

	/// Feeds ONE sequence to both structures.
	///
	/// The merged handle needs the object in the map first: in that design
	/// inserting into the map IS inserting into the stack, and
	/// `insert_resident` only settles. The split stack owns its own row, so
	/// `insert_resident` is the whole insert. After that both see the identical
	/// call, in the identical order.
	fn feed(
		store: &Arc<Store>,
		merged: &mut Handle,
		split: &mut LruCompactHybridStack,
		seq: &[(HashedKey, ObjectSize)],
	) {
		for &(key, size) in seq {
			let value = vec![0u8; value_len(size) as usize];

			store.insert(key, Object::new(key, &value, None));
			merged.insert_resident(key, size, 0);
			split.insert_resident(key, size, 0);
		}
	}

	/// The value LENGTH whose WHOLE ITEM costs exactly `item` bytes.
	///
	/// The constants above name what each object must be CHARGED, because the
	/// budget is expressed in those bytes and the scenario is built on them.
	/// The merged store does not take that figure from the caller: it derives
	/// it from the object, through `Slot::migrating` -> `resident_object_bytes`
	/// -- so under `fused_value` a 512-BYTE VALUE is a 536-byte item that
	/// rounds to 640, and feeding the reference stack 512 charged the two
	/// structures differently. The tier comparison then stopped being about
	/// ORDER, which is the only thing it exists to check.
	///
	/// Subtracting the header makes the item land back on the class in BOTH
	/// layouts: 488 + 24 = 512 fused, and 488 rounds to 512 split. Every
	/// capacity and count in this module is therefore unchanged.
	fn value_len(item: ObjectSize) -> ObjectSize {
		let len = item - crate::object::overhead::value_header_bytes::<u64>();

		assert_eq!(
			crate::object::overhead::resident_object_bytes::<u64>(len),
			item,
			"a {len}-byte value does not make an item of exactly {item} bytes, \
			 so the merged store and the reference stack are being charged \
			 different numbers and this fixture is not comparing orders",
		);

		len
	}

	#[test]
	fn demotion_victims_are_the_globally_oldest_fast_objects() {
		let cold: Vec<HashedKey> = (1..=N_SMALL).map(|i| in_shard(COLD_SHARD, i)).collect();
		let hot: Vec<HashedKey> =
			(1..=N_LARGE).map(|i| in_shard(HOT_SHARD, 1_000_000 + i)).collect();

		// The skew is the whole experiment, so assert it rather than trusting
		// the mixer: distinct keys, and each group entirely in its own shard.
		let distinct: HashSet<HashedKey> = cold.iter().chain(hot.iter()).copied().collect();
		assert_eq!(distinct.len(), cold.len() + hot.len(), "two keys collided");

		for &k in &cold {
			assert_eq!(k >> (64 - SHARD_BITS), COLD_SHARD, "a cold key left its shard");
		}

		for &k in &hot {
			assert_eq!(k >> (64 - SHARD_BITS), HOT_SHARD, "a hot key left its shard");
		}

		// Cold first, so every small object is older than every large one.
		let mut seq: Vec<(HashedKey, ObjectSize)> =
			cold.iter().map(|&k| (k, SMALL)).collect();

		seq.extend(hot.iter().map(|&k| (k, LARGE)));

		let store = Arc::new(Store::new());
		let mut merged =
			Handle::new(store.clone(), PaperPolicy::LruCompactHybrid, FAST_CAPACITY * 5)
				.expect("lru-compact-hybrid is implemented");

		// Override whatever `Handle::new` derived from `max_size`. The
		// experiment needs an exact budget and no per-object reservation, on
		// both sides -- `LruCompactHybridStack::new` leaves its own shared
		// overhead at zero.
		store.configure_tiering(
			FAST_CAPACITY,
			0,
			drain_target_ppm(),
			drain_target_ppm(),
		);

		let mut split = LruCompactHybridStack::new(FAST_CAPACITY);

		feed(&store, &mut merged, &mut split, &seq);

		// 1. The fidelity claim, key by key. This is what a per-shard budget
		//    cannot satisfy, and it is checked before the readable summaries
		//    below so a failure names the first key that differs.
		for &(key, _) in &seq {
			assert_eq!(
				store.tier_of(key),
				split.tier_of(key),
				"key {key:#018x} is in a different tier in the merged store than \
				 in the single-list stack fed the same sequence",
			);
		}

		// 2. The recent large objects are the newest things in the cache, so a
		//    GLOBAL boundary cannot touch one while any older object is fast.
		let demoted_hot = hot.iter().filter(|&&k| store.tier_of(k) == Some(Tier::Slow)).count();

		assert_eq!(
			demoted_hot, 0,
			"{demoted_hot} of the {N_LARGE} most recent objects were demoted while \
			 older objects stayed fast -- the boundary is per shard, not global",
		);

		// 3. The victims are an oldest-first PREFIX of the cold objects. A set
		//    comparison would pass on any subset of the right size; a prefix is
		//    the actual LRU claim.
		let demoted_cold: Vec<usize> = cold
			.iter()
			.enumerate()
			.filter(|&(_, &k)| store.tier_of(k) == Some(Tier::Slow))
			.map(|(i, _)| i)
			.collect();

		assert!(!demoted_cold.is_empty(), "nothing was demoted -- the budget never bit");
		assert!(
			demoted_cold.len() < cold.len(),
			"every cold object was demoted -- the budget bit so hard the order is \
			 untested",
		);
		assert_eq!(
			demoted_cold,
			(0..demoted_cold.len()).collect::<Vec<usize>>(),
			"the victims are not the oldest cold objects, so they were not chosen \
			 by global recency",
		);

		// 4. The gauges the tiering manager reads agree too, which is what makes
		//    the two stacks interchangeable to the worker rather than merely
		//    agreeing about tiers.
		assert_eq!(
			store.fast_bytes_used(),
			split.fast_bytes_used(),
			"the two stacks disagree about how many fast bytes they hold",
		);
		assert_eq!(
			store.fast_object_count(),
			split.fast_object_count(),
			"the two stacks disagree about how many fast objects they hold",
		);
		assert_eq!(
			store.slow_object_count(),
			split.slow_object_count(),
			"the two stacks disagree about how many slow objects they hold",
		);
	}

	/// The same skew, but the cold shard is then TOUCHED back to the MRU end.
	///
	/// Demotion has to follow the order rather than the shard: once the small
	/// objects are the most recent, the large ones become the oldest fast
	/// objects and the victims must switch shards entirely. A boundary settled
	/// per shard cannot move bytes across shards at all, so it cannot do this.
	#[test]
	fn touching_the_cold_shard_moves_the_victims_to_the_other_shard() {
		let cold: Vec<HashedKey> = (1..=N_SMALL).map(|i| in_shard(COLD_SHARD, i)).collect();
		let hot: Vec<HashedKey> =
			(1..=N_LARGE).map(|i| in_shard(HOT_SHARD, 1_000_000 + i)).collect();

		let mut seq: Vec<(HashedKey, ObjectSize)> =
			cold.iter().map(|&k| (k, SMALL)).collect();

		seq.extend(hot.iter().map(|&k| (k, LARGE)));

		let store = Arc::new(Store::new());
		let mut merged =
			Handle::new(store.clone(), PaperPolicy::LruCompactHybrid, FAST_CAPACITY * 5)
				.expect("lru-compact-hybrid is implemented");

		store.configure_tiering(
			FAST_CAPACITY,
			0,
			drain_target_ppm(),
			drain_target_ppm(),
		);

		let mut split = LruCompactHybridStack::new(FAST_CAPACITY);

		feed(&store, &mut merged, &mut split, &seq);

		// Every cold key, oldest first, so the whole small group ends up newer
		// than the whole large group -- and promoting them back into the fast
		// tier is itself what pushes the large ones out.
		for &k in &cold {
			merged.update(k);
			split.update(k);
		}

		for &(key, _) in &seq {
			assert_eq!(
				store.tier_of(key),
				split.tier_of(key),
				"key {key:#018x} diverged after the touches",
			);
		}

		let slow_hot = hot.iter().filter(|&&k| store.tier_of(k) == Some(Tier::Slow)).count();

		assert!(
			slow_hot > 0,
			"the now-oldest large objects were not demoted, so the boundary did \
			 not follow the order across shards",
		);
		assert_eq!(
			store.fast_bytes_used(),
			split.fast_bytes_used(),
			"the two stacks disagree about fast bytes after the touches",
		);
	}
}

/// The acceptance test for `MergedOrder::Fifo`: it is the SAME order
/// [`FifoCompactHybridStack`] implements, key for key, step for step.
///
/// # Why this test and not a property of the merged store alone
///
/// "FIFO" is not a property a single structure can be checked against -- any
/// self-consistent order looks fine from inside. It is a claim that this store
/// and the reference FIFO stack, fed one sequence, agree. So the sequence is
/// replayed into both and the two are compared at every step on tier placement,
/// and at the end on the whole eviction order and on the gauges the tiering
/// manager reads.
///
/// # What each ingredient of the sequence is for
///
/// * **Hits, on fast keys AND on demoted slow ones.** This is the whole of
///   FIFO, and the only thing that can break it. Under recency a hit relinks to
///   the MRU end and promotes out of the slow tier; either one diverges from
///   the reference stack immediately -- a promotion shows up in `tier_of` at
///   the next step, a relink in the eviction order at the end. The reference
///   stack does not override `PolicyStack::update` at all, so on its side a hit
///   is the trait's empty default; the merged store has to arrive at that same
///   nothing from the other direction, by refusing to do what its own link
///   structure makes easy.
///
/// * **An overwrite of a key that is still fast, to a larger size.** The subtle
///   one. An overwrite has to do HALF of what a touch does -- the bytes really
///   did change and some tier has to be charged for them -- and none of the
///   other half: no relink, no promotion, and above all no new `last_access`,
///   which under FIFO is not a recency stamp but this object's POSITION in the
///   queue. Growing a fast key also pushes the fast tier over its budget, so
///   the settle that follows demotes older objects and the boundary has to
///   land in the same place on both sides.
///
/// * **An overwrite of a key that has already been demoted.** The same delta,
///   charged to the SLOW tier this time, and still no promotion -- the case
///   where forgetting the order costs a wrong tier rather than a wrong
///   position.
///
/// * **An overwrite that does not change the size.** The reference stack
///   returns early without even re-settling; the merged store cannot, because
///   the object was physically replaced. The two must still agree.
///
/// * **Keys spread over three shards.** A single-shard sequence would compare
///   one list against one list and never exercise `tail_key`'s minimum over the
///   shard tails, which is the part of the merged store that has to reconstruct
///   a global FIFO order out of 32 independent ones.
///
/// # Why the two are comparable at all
///
/// Identical `(key, size, dram_resident)` calls through `PolicyStack`, the same
/// unconditional fast admission, the same `drain_target`, and no per-object
/// shared overhead on either side. The sizes are exact jemalloc size classes,
/// so the merged store's size-class-rounded `migrating()` and the reference
/// stack's raw `size - dram_resident` are the same number and the accounting
/// cannot drift for a reason unrelated to the order.
#[cfg(all(test, feature = "fifo_compact_hybrid_cache"))]
mod fifo_order_fidelity {
	use super::*;

	use crate::{
		object::Object,
		worker::policy::policy_stack::{
			fifo_compact_hybrid_stack::FifoCompactHybridStack,
		},
		BufferDRAM,
	};

	use std::collections::HashSet;

	type Store = MergedStore<u64, BufferDRAM>;
	type Handle = MergedStackHandle<u64, BufferDRAM>;

	/// `MergedStore::shard_of` reads the top five bits of the key.
	const SHARD_BITS: u32 = 5;
	const SHARDS: u64 = 1 << SHARD_BITS;

	/// All exact jemalloc size classes, so `nallocx` rounds none of them and
	/// the two structures charge the identical number of bytes per object.
	///
	/// These are ITEM sizes -- what the object COSTS -- and `value_len` below
	/// turns each into the value length that produces it.
	const SMALL: ObjectSize = 512;
	const MEDIUM: ObjectSize = 1024;
	const LARGE: ObjectSize = 8192;

	const N_KEYS: u64 = 240;

	/// Room for roughly 117 of the 240 small objects, so the budget bites in
	/// the MIDDLE of the sequence: a capacity that demoted all or none of them
	/// would let a wrong order agree by accident.
	const FAST_CAPACITY: CacheSize = 60_000;

	fn mix(i: u64) -> HashedKey {
		i.wrapping_mul(0x9E37_79B9_7F4A_7C15)
	}

	/// A key that lands in shard `s`: the mixed value shifted clear of the
	/// shard field, then the shard written into it.
	fn in_shard(s: u64, i: u64) -> HashedKey {
		assert!(s < SHARDS);
		(mix(i) >> SHARD_BITS) | (s << (64 - SHARD_BITS))
	}

	/// The n-th key of the sequence, round-robin over three shards.
	fn key_at(n: u64) -> HashedKey {
		in_shard(n % 3, n)
	}

	/// The value LENGTH whose WHOLE ITEM costs exactly `item` bytes.
	///
	/// The constants above name what each object must be CHARGED, because the
	/// budget is expressed in those bytes and the scenario is built on them.
	/// The merged store does not take that figure from the caller: it derives
	/// it from the object, through `Slot::migrating` -> `resident_object_bytes`
	/// -- so under `fused_value` a 512-BYTE VALUE is a 536-byte item that
	/// rounds to 640, and feeding the reference stack 512 charged the two
	/// structures differently. The tier comparison then stopped being about
	/// ORDER, which is the only thing it exists to check.
	///
	/// Subtracting the header makes the item land back on the class in BOTH
	/// layouts: 488 + 24 = 512 fused, and 488 rounds to 512 split. Every
	/// capacity and count in this module is therefore unchanged.
	fn value_len(item: ObjectSize) -> ObjectSize {
		let len = item - crate::object::overhead::value_header_bytes::<u64>();

		assert_eq!(
			crate::object::overhead::resident_object_bytes::<u64>(len),
			item,
			"a {len}-byte value does not make an item of exactly {item} bytes, \
			 so the merged store and the reference stack are being charged \
			 different numbers and this fixture is not comparing orders",
		);

		len
	}

	/// Spelled out rather than inferred from whether the key happens to be
	/// present: an `Insert` that silently became an overwrite, or an
	/// `Overwrite` whose key had gone, would quietly delete the case the
	/// sequence exists to cover.
	#[derive(Clone, Copy, Debug)]
	enum Op {
		Insert(HashedKey, ObjectSize),
		Hit(HashedKey),
		Overwrite(HashedKey, ObjectSize),
	}

	/// One op, to both structures.
	///
	/// The merged handle needs the object in the map first: in that design
	/// inserting into the map IS inserting into the stack, and
	/// `insert_resident` only settles. The split stack owns its own row, so
	/// `insert_resident` is the whole insert. An overwrite is the same pair of
	/// calls -- which is the point, since neither side is told which it is.
	fn apply(
		store: &Arc<Store>,
		merged: &mut Handle,
		split: &mut FifoCompactHybridStack,
		op: Op,
	) {
		match op {
			Op::Insert(key, size) => {
				assert!(!store.contains(key), "Insert of a key already present");
				assert!(!split.contains(key), "Insert of a key already present");

				store.insert(key, Object::new(key, &vec![0u8; value_len(size) as usize], None));
				merged.insert_resident(key, size, 0);
				split.insert_resident(key, size, 0);
			},

			Op::Hit(key) => {
				assert!(store.contains(key), "Hit on a key that is not present");
				assert!(split.contains(key), "Hit on a key that is not present");

				merged.update(key);
				split.update(key);
			},

			Op::Overwrite(key, size) => {
				assert!(store.contains(key), "Overwrite of a key that is not present");
				assert!(split.contains(key), "Overwrite of a key that is not present");

				store.insert(key, Object::new(key, &vec![0u8; value_len(size) as usize], None));
				merged.insert_resident(key, size, 0);
				split.insert_resident(key, size, 0);
			},
		}
	}

	/// The sequence. Built rather than written out so the interesting ops sit
	/// at positions the budget has already bitten at.
	fn sequence() -> (Vec<HashedKey>, Vec<Op>) {
		let keys: Vec<HashedKey> = (0..N_KEYS).map(key_at).collect();

		let distinct: HashSet<HashedKey> = keys.iter().copied().collect();
		assert_eq!(distinct.len(), keys.len(), "two keys collided");

		let mut ops = Vec::new();

		for (n, &key) in keys.iter().enumerate() {
			ops.push(Op::Insert(key, SMALL));

			// Hits on keys already in the cache, from both ends of the queue:
			// `n / 4` is old enough to have been demoted once the budget bites,
			// `n` is the newest object there is. Under recency the first would
			// be PROMOTED and the second RELINKED; under FIFO neither moves.
			if n % 3 == 0 {
				ops.push(Op::Hit(keys[n / 4]));
				ops.push(Op::Hit(key));
			}
		}

		// A fast key -- one of the newest -- grown to 16x its size. Charged to
		// the fast tier, and large enough that the settle that follows demotes
		// a run of older objects.
		ops.push(Op::Overwrite(keys[(N_KEYS - 3) as usize], LARGE));

		// A key demoted long ago, resized. Charged to the SLOW tier, and it
		// must not be promoted by having been written.
		ops.push(Op::Overwrite(keys[2], MEDIUM));

		// And an overwrite that changes nothing, which the reference stack
		// short-circuits and the merged store cannot.
		ops.push(Op::Overwrite(keys[5], SMALL));

		// Hits after the overwrites, so a relink introduced by an overwrite
		// cannot be masked by the sequence ending there.
		ops.push(Op::Hit(keys[(N_KEYS - 3) as usize]));
		ops.push(Op::Hit(keys[2]));

		(keys, ops)
	}

	fn build() -> (Arc<Store>, Handle, FifoCompactHybridStack) {
		let store = Arc::new(Store::new());
		let merged =
			Handle::new(store.clone(), PaperPolicy::FifoCompactHybrid, FAST_CAPACITY * 5)
				.expect("fifo-compact-hybrid is implemented");

		// Override whatever `Handle::new` derived from `max_size`: the
		// experiment needs an exact budget and no per-object reservation on
		// either side, and `FifoCompactHybridStack::new` leaves its own shared
		// overhead at zero.
		store.configure_tiering(FAST_CAPACITY, 0, drain_target_ppm(), drain_target_ppm());

		let split = FifoCompactHybridStack::new(FAST_CAPACITY);

		(store, merged, split)
	}

	/// The policy string really does reach the store's order, rather than
	/// being stored and compared and never dispatched on -- which is what it
	/// did before FIFO existed.
	#[test]
	fn the_policy_selects_the_order() {
		let store = Arc::new(Store::new());
		let _fifo = Handle::new(store.clone(), PaperPolicy::FifoCompactHybrid, 1 << 20)
			.expect("fifo-compact-hybrid is implemented");

		assert_eq!(
			store.order(),
			crate::merged_store::MergedOrder::Fifo,
			"fifo-compact-hybrid did not select the FIFO order",
		);

		let store = Arc::new(Store::new());
		let _lru = Handle::new(store.clone(), PaperPolicy::LruCompactHybrid, 1 << 20)
			.expect("lru-compact-hybrid is implemented");

		assert_eq!(
			store.order(),
			crate::merged_store::MergedOrder::Lru,
			"lru-compact-hybrid did not select the LRU order",
		);

		// A policy the merged store does not implement is an ERROR, not a quiet
		// substitution.
		//
		// This assertion used to be the opposite of itself -- "an unimplemented
		// policy must fall back to recency, not to nothing" -- and it was wrong
		// by POLICY rather than by accident: what it pinned was a run that
		// produced a plausible miss ratio filed under a policy name it was not
		// honouring. `from_policy` has no fallback now and `new` is fallible,
		// so a mislabelled run cannot be constructed at all.
		//
		// The probe had to change with it. It was `LfuCompactHybrid`, which is
		// exactly the policy that stopped being unimplemented, so leaving it
		// would have made this test pass for the wrong reason.
		// `TwoQCompactHybrid` is the right replacement: still unimplemented,
		// and still HYBRID, so it keeps exercising the `policy.is_hybrid()` /
		// `configure_tiering` path that runs in `new` before the order is
		// resolved.
		let store = Arc::new(Store::new());

		// Pre-set to something that is NOT the old fallback, so "the refusal
		// installed nothing" is checkable. Against the default this assertion
		// could not tell an untouched store from one the fallback had just set
		// to recency.
		store.set_order(crate::merged_store::MergedOrder::Fifo);

		let refused = Handle::new(store.clone(), PaperPolicy::TwoQCompactHybrid(0.2), 1 << 20);

		assert!(
			matches!(
				refused,
				Err(CacheError::PolicyNotImplemented(PaperPolicy::TwoQCompactHybrid(..))),
			),
			"an unimplemented policy must fail to construct AND name itself, \
			 rather than falling back to another order",
		);

		assert_eq!(
			store.order(),
			crate::merged_store::MergedOrder::Fifo,
			"the refused construction installed an order anyway -- `set_order` \
			 must not run when `from_policy` fails",
		);
	}

	#[test]
	fn fifo_matches_the_reference_stack_key_for_key() {
		let (keys, ops) = sequence();
		let (store, mut merged, mut split) = build();

		// 1. Tier placement, after EVERY op, so a failure names the step that
		//    introduced the divergence rather than the end state.
		for (n, &op) in ops.iter().enumerate() {
			apply(&store, &mut merged, &mut split, op);

			for &key in &keys {
				if !store.contains(key) {
					continue;
				}

				assert_eq!(
					store.tier_of(key),
					split.tier_of(key),
					"after step {n} ({op:?}) key {key:#018x} is in a different \
					 tier in the merged store than in the reference FIFO stack",
				);
			}
		}

		// The sequence has to have actually demoted things, or every tier
		// comparison above was `Some(Fast) == Some(Fast)` and said nothing.
		assert!(
			store.slow_object_count() > 0 && store.fast_object_count() > 0,
			"the budget demoted everything or nothing, so the order is untested",
		);

		// 2. The gauges the tiering manager reads, which is what makes the two
		//    interchangeable to the worker rather than merely agreeing about
		//    tiers.
		assert_eq!(
			store.fast_bytes_used(),
			split.fast_bytes_used(),
			"the two stacks disagree about how many fast bytes they hold",
		);
		assert_eq!(
			store.slow_bytes_used(),
			split.slow_bytes_used(),
			"the two stacks disagree about how many slow bytes they hold",
		);
		assert_eq!(
			store.fast_object_count(),
			split.fast_object_count(),
			"the two stacks disagree about how many fast objects they hold",
		);
		assert_eq!(
			store.slow_object_count(),
			split.slow_object_count(),
			"the two stacks disagree about how many slow objects they hold",
		);

		// 3. The eviction order, to the last key. `evict_one` on the merged
		//    handle only NOMINATES -- the removal is `take`, exactly as
		//    `apply_evictions` pairs them -- where the reference stack's
		//    `evict_one` removes.
		let mut merged_order = Vec::new();

		while let Some(key) = merged.evict_one() {
			assert!(store.take(&key).is_some(), "nominated victim was not present");
			merged_order.push(key);
		}

		let mut split_order = Vec::new();

		while let Some(key) = split.evict_one() {
			split_order.push(key);
		}

		assert_eq!(
			merged_order, split_order,
			"the merged store and the reference FIFO stack evict in different \
			 orders",
		);

		// 4. And that shared order is insertion order, stated directly rather
		//    than only relative to the reference stack -- so the test still
		//    means something if both were wrong in the same way.
		assert_eq!(
			merged_order, keys,
			"eviction order is not insertion order, so something reordered on a \
			 hit or on an overwrite",
		);

		assert_eq!(store.len(), 0, "the drain left objects behind");
	}
}

/// The acceptance test for `MergedOrder::Clock`: it is the SAME order the flat
/// `ClockCompactStack` implements, key for key, step for step.
///
/// # Two references, because there are two claims
///
/// * [`ClockCompactHybridStack`] -- the split-structure statement of the same
///   policy. Fed the identical `PolicyStack` calls, it must agree with the
///   merged store on the tier of every live key after every step, on the four
///   gauges the tiering manager reads, and on the whole eviction order. This is
///   the claim that 32 shard lists plus a stamp reconstruct one global CLOCK
///   order.
///
/// * `ClockCompactStack` -- the FLAT stack serving `PaperPolicy::ClockCompact`,
///   which has no tiers and no shards at all. The order those two agree on must
///   also be the order it produces from the same accesses. This is the claim
///   that neither of the tiered structures quietly invented a policy: tiering
///   moves a cursor along the queue and is not allowed to reorder it, so an
///   untiered CLOCK is the arbiter of what the order IS.
///
/// # What each ingredient of the sequence is for
///
/// * **A key hit exactly once and then never again.** The minimal second
///   chance: it must survive exactly one pass of the hand and be evicted on the
///   next. A bit that was never set, or never cleared, both show here.
///
/// * **A key hit over and over.** Its bit is set again after every clear, so it
///   survives pass after pass and leaves last. Under LRU it would also leave
///   last, which is why the FLAT comparison matters -- it pins the positions of
///   everything else, which LRU would have shuffled.
///
/// * **An overwrite of an existing key.** `ClockCompactStack::insert` forwards
///   an existing key to `update`, so an overwrite SETS THE BIT and moves
///   nothing. Three of them here: one growing a still-fast key 16x (charged to
///   the fast tier, and large enough that the settle after it demotes a run of
///   older objects), one resizing a long-demoted key (charged to the slow tier,
///   and not promoted by having been written), and one that changes no bytes at
///   all -- which the reference stack short-circuits and the merged store
///   cannot.
///
/// * **Hits on demoted slow keys.** A CLOCK hit must not promote. The promotion
///   happens later, when the hand grants the second chance, and only then.
///
/// * **Keys spread over three shards.** A single-shard sequence would compare
///   one list against one list and never exercise `tail_key`'s minimum over the
///   shard tails -- the part of the merged store that has to reconstruct a
///   global order out of 32 independent ones, and the part a second chance has
///   to keep coherent by restamping.
///
/// # Where the tiers are actually tested
///
/// Not during the op sequence. Until the hand runs, CLOCK places tiers exactly
/// as FIFO does, because a hit moves nothing -- so the per-step comparison
/// below is a FIFO check with a reference bit riding along. The CLOCK-specific
/// placement happens DURING THE DRAIN, where every second chance promotes its
/// key back into the fast tier and the settle that follows demotes someone
/// else. So the tiers and the gauges are compared after every eviction too, not
/// only at the end.
#[cfg(all(test, feature = "clock_compact_hybrid_cache"))]
mod clock_order_fidelity {
	use super::*;

	use crate::{
		object::Object,
		worker::policy::policy_stack::{
			clock_compact_hybrid_stack::ClockCompactHybridStack,
			clock_compact_stack::ClockCompactStack,
		},
		BufferDRAM,
	};

	use std::collections::HashSet;

	type Store = MergedStore<u64, BufferDRAM>;
	type Handle = MergedStackHandle<u64, BufferDRAM>;

	/// `MergedStore::shard_of` reads the top five bits of the key.
	const SHARD_BITS: u32 = 5;
	const SHARDS: u64 = 1 << SHARD_BITS;

	/// All exact jemalloc size classes, so `nallocx` rounds none of them and
	/// the two tiered structures charge the identical number of bytes per
	/// object.
	///
	/// These are ITEM sizes -- what the object COSTS -- and `value_len` below
	/// turns each into the value length that produces it.
	const SMALL: ObjectSize = 512;
	const MEDIUM: ObjectSize = 1024;
	const LARGE: ObjectSize = 8192;

	const N_KEYS: u64 = 240;

	/// Room for roughly 117 of the 240 small objects, so the budget bites in
	/// the MIDDLE of the sequence: a capacity that demoted all or none of them
	/// would let a wrong order agree by accident.
	const FAST_CAPACITY: CacheSize = 60_000;

	/// Hit exactly ONCE, immediately after it is inserted, and never mentioned
	/// again. Its reference bit is therefore set for the whole run.
	const ONE_HIT: usize = 7;

	/// Never hit at all, and inserted immediately after `ONE_HIT` -- so under
	/// pure FIFO it outlives it, and the reference bit is the only thing that
	/// can reverse the two.
	const NEVER_HIT: usize = 8;

	/// Never HIT, but overwritten exactly once. `ClockCompactStack::insert`
	/// forwards an existing key to `update`, so the overwrite is the only thing
	/// in the whole run that can set this key's bit.
	///
	/// It exists because the first version of this test did not cover the
	/// overwrite at all: all three overwrites landed on keys the generic hit
	/// rule had already referenced, so deleting the bit from the merged store's
	/// overwrite path changed nothing and the test still passed. This key and
	/// the one below are the pair that closes that hole.
	const OVERWRITE_ONLY: usize = 9;

	/// Never hit and never overwritten, and inserted immediately after
	/// `OVERWRITE_ONLY` -- the control for it, exactly as `NEVER_HIT` is the
	/// control for `ONE_HIT`.
	const NEVER_TOUCHED: usize = 10;

	/// Hit at every opportunity, so its bit is always set when the hand
	/// arrives.
	const MANY_HITS: usize = 3;

	/// Indices the generic hit rule below must leave alone, or "hit exactly
	/// once" and "never hit" would not be true of them: `keys[n / 4]` reaches
	/// EVERY index eventually, since any four consecutive integers contain a
	/// multiple of three.
	fn reserved(i: usize) -> bool {
		i == ONE_HIT
			|| i == NEVER_HIT
			|| i == OVERWRITE_ONLY
			|| i == NEVER_TOUCHED
			|| i == MANY_HITS
	}

	fn mix(i: u64) -> HashedKey {
		i.wrapping_mul(0x9E37_79B9_7F4A_7C15)
	}

	/// A key that lands in shard `s`: the mixed value shifted clear of the
	/// shard field, then the shard written into it.
	fn in_shard(s: u64, i: u64) -> HashedKey {
		assert!(s < SHARDS);
		(mix(i) >> SHARD_BITS) | (s << (64 - SHARD_BITS))
	}

	/// The n-th key of the sequence, round-robin over three shards.
	fn key_at(n: u64) -> HashedKey {
		in_shard(n % 3, n)
	}

	/// The value LENGTH whose WHOLE ITEM costs exactly `item` bytes.
	///
	/// The constants above name what each object must be CHARGED, because the
	/// budget is expressed in those bytes and the scenario is built on them.
	/// The merged store does not take that figure from the caller: it derives
	/// it from the object, through `Slot::migrating` -> `resident_object_bytes`
	/// -- so under `fused_value` a 512-BYTE VALUE is a 536-byte item that
	/// rounds to 640, and feeding the reference stack 512 charged the two
	/// structures differently. The tier comparison then stopped being about
	/// ORDER, which is the only thing it exists to check.
	///
	/// Subtracting the header makes the item land back on the class in BOTH
	/// layouts: 488 + 24 = 512 fused, and 488 rounds to 512 split. Every
	/// capacity and count in this module is therefore unchanged.
	fn value_len(item: ObjectSize) -> ObjectSize {
		let len = item - crate::object::overhead::value_header_bytes::<u64>();

		assert_eq!(
			crate::object::overhead::resident_object_bytes::<u64>(len),
			item,
			"a {len}-byte value does not make an item of exactly {item} bytes, \
			 so the merged store and the reference stack are being charged \
			 different numbers and this fixture is not comparing orders",
		);

		len
	}

	/// Spelled out rather than inferred from whether the key happens to be
	/// present: an `Insert` that silently became an overwrite, or an
	/// `Overwrite` whose key had gone, would quietly delete the case the
	/// sequence exists to cover.
	#[derive(Clone, Copy, Debug)]
	enum Op {
		Insert(HashedKey, ObjectSize),
		Hit(HashedKey),
		Overwrite(HashedKey, ObjectSize),
	}

	/// One op, to all three structures.
	///
	/// The merged handle needs the object in the map first: in that design
	/// inserting into the map IS inserting into the stack, and
	/// `insert_resident` only settles. The two split stacks own their own rows,
	/// so `insert_resident` is the whole insert. An overwrite is the same pair
	/// of calls -- which is the point, since no side is told which it is.
	fn apply(
		store: &Arc<Store>,
		merged: &mut Handle,
		split: &mut ClockCompactHybridStack,
		flat: &mut ClockCompactStack,
		op: Op,
	) {
		match op {
			Op::Insert(key, size) => {
				assert!(!store.contains(key), "Insert of a key already present");
				assert!(!split.contains(key), "Insert of a key already present");

				store.insert(key, Object::new(key, &vec![0u8; value_len(size) as usize], None));
				merged.insert_resident(key, size, 0);
				split.insert_resident(key, size, 0);
				flat.insert(key, size);
			},

			Op::Hit(key) => {
				assert!(store.contains(key), "Hit on a key that is not present");
				assert!(split.contains(key), "Hit on a key that is not present");

				merged.update(key);
				split.update(key);
				flat.update(key);
			},

			Op::Overwrite(key, size) => {
				assert!(store.contains(key), "Overwrite of a key that is not present");
				assert!(split.contains(key), "Overwrite of a key that is not present");

				store.insert(key, Object::new(key, &vec![0u8; value_len(size) as usize], None));
				merged.insert_resident(key, size, 0);
				split.insert_resident(key, size, 0);
				flat.insert(key, size);
			},
		}
	}

	/// The sequence. Built rather than written out so the interesting ops sit
	/// at positions the budget has already bitten at.
	fn sequence() -> (Vec<HashedKey>, Vec<Op>) {
		let keys: Vec<HashedKey> = (0..N_KEYS).map(key_at).collect();

		let distinct: HashSet<HashedKey> = keys.iter().copied().collect();
		assert_eq!(distinct.len(), keys.len(), "two keys collided");

		let mut ops = Vec::new();

		for (n, &key) in keys.iter().enumerate() {
			ops.push(Op::Insert(key, SMALL));

			// The key that is hit exactly once, as early as it can be.
			if n == ONE_HIT {
				ops.push(Op::Hit(key));
			}

			// Hits on keys already in the cache, from both ends of the queue:
			// `n / 4` is old enough to have been demoted once the budget bites,
			// `n` is the newest object there is. Neither may move, and the
			// demoted one may not be promoted.
			if n % 3 == 0 {
				if !reserved(n / 4) {
					ops.push(Op::Hit(keys[n / 4]));
				}

				if !reserved(n) {
					ops.push(Op::Hit(key));
				}

				// And the key that is hit over and over.
				if n > MANY_HITS {
					ops.push(Op::Hit(keys[MANY_HITS]));
				}
			}
		}

		// A fast key -- one of the newest -- grown to 16x its size. Charged to
		// the fast tier, and large enough that the settle that follows demotes
		// a run of older objects.
		ops.push(Op::Overwrite(keys[(N_KEYS - 3) as usize], LARGE));

		// A key demoted long ago, resized. Charged to the SLOW tier, and it
		// must not be promoted by having been written.
		ops.push(Op::Overwrite(keys[2], MEDIUM));

		// And an overwrite that changes nothing, which the reference stack
		// short-circuits and the merged store cannot. It still sets the bit.
		ops.push(Op::Overwrite(keys[5], SMALL));

		// The overwrite that is the ONLY access its key ever gets, so the
		// reference bit it sets is load-bearing and nothing else can supply it.
		ops.push(Op::Overwrite(keys[OVERWRITE_ONLY], MEDIUM));

		// Hits after the overwrites, so a relink introduced by an overwrite
		// cannot be masked by the sequence ending there.
		ops.push(Op::Hit(keys[(N_KEYS - 3) as usize]));
		ops.push(Op::Hit(keys[2]));

		// The three reserved keys really are what they are called. Asserted
		// rather than read off the loop above, because the generic rule reaches
		// far more keys than it looks like it does -- an earlier revision of
		// this test believed `keys[8]` was never hit and it was hit at step 33.
		let hits_on = |i: usize| {
			ops.iter()
				.filter(|op| matches!(op, Op::Hit(k) if *k == keys[i]))
				.count()
		};

		let writes_on = |i: usize| {
			ops.iter()
				.filter(|op| matches!(op, Op::Overwrite(k, _) if *k == keys[i]))
				.count()
		};

		assert_eq!(hits_on(ONE_HIT), 1, "the hit-once key was hit a different number of times");
		assert_eq!(hits_on(NEVER_HIT), 0, "the never-hit key was hit");
		assert_eq!(hits_on(OVERWRITE_ONLY), 0, "the overwrite-only key was hit");
		assert_eq!(writes_on(OVERWRITE_ONLY), 1, "the overwrite-only key was not overwritten once");
		assert_eq!(hits_on(NEVER_TOUCHED), 0, "the untouched key was hit");
		assert_eq!(writes_on(NEVER_TOUCHED), 0, "the untouched key was overwritten");
		assert!(
			hits_on(MANY_HITS) > 10,
			"the repeatedly-hit key was hit {} times, which is not repeatedly",
			hits_on(MANY_HITS),
		);

		(keys, ops)
	}

	fn build() -> (Arc<Store>, Handle, ClockCompactHybridStack, ClockCompactStack) {
		let store = Arc::new(Store::new());
		let merged =
			Handle::new(store.clone(), PaperPolicy::ClockCompactHybrid, FAST_CAPACITY * 5)
				.expect("clock-compact-hybrid is implemented");

		// Override whatever `Handle::new` derived from `max_size`: the
		// experiment needs an exact budget and no per-object reservation on
		// either side, and `ClockCompactHybridStack::new` leaves its own shared
		// overhead at zero.
		store.configure_tiering(FAST_CAPACITY, 0, drain_target_ppm(), drain_target_ppm());

		let split = ClockCompactHybridStack::new(FAST_CAPACITY);

		(store, merged, split, ClockCompactStack::default())
	}

	/// `clock-compact-hybrid` really does reach the store's order, and the two
	/// orders that were already there are untouched by its arrival.
	#[test]
	fn the_policy_selects_the_clock_order() {
		let store = Arc::new(Store::new());
		let _clock = Handle::new(store.clone(), PaperPolicy::ClockCompactHybrid, 1 << 20)
			.expect("clock-compact-hybrid is implemented");

		assert_eq!(
			store.order(),
			crate::merged_store::MergedOrder::Clock,
			"clock-compact-hybrid did not select the CLOCK order",
		);

		let store = Arc::new(Store::new());
		let _fifo = Handle::new(store.clone(), PaperPolicy::FifoCompactHybrid, 1 << 20)
			.expect("fifo-compact-hybrid is implemented");

		assert_eq!(
			store.order(),
			crate::merged_store::MergedOrder::Fifo,
			"adding CLOCK moved fifo-compact-hybrid off the FIFO order",
		);

		let store = Arc::new(Store::new());
		let _lru = Handle::new(store.clone(), PaperPolicy::LruCompactHybrid, 1 << 20)
			.expect("lru-compact-hybrid is implemented");

		assert_eq!(
			store.order(),
			crate::merged_store::MergedOrder::Lru,
			"adding CLOCK moved lru-compact-hybrid off the LRU order",
		);
	}

	#[test]
	fn clock_matches_the_reference_stack_key_for_key() {
		let (keys, ops) = sequence();
		let (store, mut merged, mut split, mut flat) = build();

		// 1. Tier placement, after EVERY op, so a failure names the step that
		//    introduced the divergence rather than the end state.
		for (n, &op) in ops.iter().enumerate() {
			apply(&store, &mut merged, &mut split, &mut flat, op);

			for &key in &keys {
				if !store.contains(key) {
					continue;
				}

				assert_eq!(
					store.tier_of(key),
					split.tier_of(key),
					"after step {n} ({op:?}) key {key:#018x} is in a different \
					 tier in the merged store than in the reference CLOCK stack",
				);
			}
		}

		// The sequence has to have actually demoted things, or every tier
		// comparison above was `Some(Fast) == Some(Fast)` and said nothing.
		assert!(
			store.slow_object_count() > 0 && store.fast_object_count() > 0,
			"the budget demoted everything or nothing, so the order is untested",
		);

		// 2. The gauges the tiering manager reads, which is what makes the two
		//    interchangeable to the worker rather than merely agreeing about
		//    tiers.
		assert_eq!(
			store.fast_bytes_used(),
			split.fast_bytes_used(),
			"the two stacks disagree about how many fast bytes they hold",
		);
		assert_eq!(
			store.slow_bytes_used(),
			split.slow_bytes_used(),
			"the two stacks disagree about how many slow bytes they hold",
		);
		assert_eq!(
			store.fast_object_count(),
			split.fast_object_count(),
			"the two stacks disagree about how many fast objects they hold",
		);
		assert_eq!(
			store.slow_object_count(),
			split.slow_object_count(),
			"the two stacks disagree about how many slow objects they hold",
		);

		// 3. The eviction order, to the last key -- AND the tier of every
		//    survivor after every eviction, which is where CLOCK's own tier
		//    behaviour lives: a second chance promotes its key back into the
		//    fast tier and the settle that follows demotes someone else.
		//
		//    `evict_one` on the merged handle only NOMINATES -- the removal is
		//    `take`, exactly as `apply_evictions` pairs them -- where the
		//    reference stacks' `evict_one` removes.
		let mut merged_order = Vec::new();
		let mut split_order = Vec::new();
		let mut promotions_during_the_drain = 0;

		loop {
			let before_fast = store.fast_object_count();

			let m = merged.evict_one();
			let s = split.evict_one();

			assert_eq!(
				m, s,
				"the merged store and the reference CLOCK stack nominated \
				 different victims at position {}",
				merged_order.len(),
			);

			let Some(key) = m else { break };

			assert!(store.take(&key).is_some(), "nominated victim was not present");

			merged_order.push(key);
			split_order.push(s.expect("checked equal to m"));

			// A second chance PROMOTES, so the fast count can go UP across an
			// eviction. Counting it proves the drain exercised the hand rather
			// than walking a queue of cleared bits.
			if store.fast_object_count() > before_fast {
				promotions_during_the_drain += 1;
			}

			for &k in &keys {
				if !store.contains(k) {
					continue;
				}

				assert_eq!(
					store.tier_of(k),
					split.tier_of(k),
					"after evicting {key:#018x} key {k:#018x} is in a different \
					 tier in the merged store than in the reference CLOCK stack",
				);
			}

			assert_eq!(
				store.fast_bytes_used(),
				split.fast_bytes_used(),
				"fast bytes diverged after evicting {key:#018x}",
			);
			assert_eq!(
				store.fast_object_count(),
				split.fast_object_count(),
				"fast object count diverged after evicting {key:#018x}",
			);
		}

		assert_eq!(merged_order, split_order, "the two orders diverged");
		assert_eq!(store.len(), 0, "the drain left objects behind");

		assert!(
			promotions_during_the_drain > 0,
			"no second chance ever promoted a key back into the fast tier, so \
			 the drain never exercised the hand",
		);

		// 4. And that shared order is the FLAT `ClockCompactStack`'s, stated
		//    directly rather than only relative to the tiered reference -- so
		//    the test still means something if both tiered structures were
		//    wrong in the same way. Tiering moves a cursor along the queue and
		//    is not allowed to reorder it, so the untiered stack is the arbiter
		//    of what CLOCK's order IS.
		let mut flat_order = Vec::new();

		while let Some(key) = flat.evict_one() {
			flat_order.push(key);
		}

		assert_eq!(
			merged_order, flat_order,
			"the merged store under CLOCK does not evict in the order the flat \
			 ClockCompactStack does",
		);

		// 5. And it is NOT insertion order, which is what FIFO would give and
		//    what this whole sequence would collapse to if the reference bit
		//    were never read.
		assert_ne!(
			merged_order, keys,
			"eviction order is exactly insertion order, so no key was ever \
			 given a second chance and CLOCK ran as FIFO",
		);

		// 6. What a reference bit actually buys, stated on the two keys the
		//    sequence reserved for it. `ONE_HIT` was inserted BEFORE
		//    `NEVER_HIT`, so under FIFO -- and under this same store with the
		//    bit ignored -- it would leave first. One hit reverses them, and
		//    exactly one hit is all it was given.
		let pos = |k: HashedKey| merged_order.iter().position(|&x| x == k).unwrap();

		assert!(
			pos(keys[ONE_HIT]) > pos(keys[NEVER_HIT]),
			"the key hit once did not outlive the key inserted after it, so its \
			 reference bit bought it nothing",
		);

		// The same claim for a WRITE, which the flat stack says is an access
		// too. This is the pair the first version of the test was missing.
		assert!(
			pos(keys[OVERWRITE_ONLY]) > pos(keys[NEVER_TOUCHED]),
			"the overwritten key did not outlive the key inserted after it, so \
			 an overwrite did not set the reference bit -- which is what \
			 `ClockCompactStack::insert` forwarding to `update` means",
		);

		// And NOT more than that, which is the honest half of the claim and the
		// difference between CLOCK and LRU: the bit is one bit, so a key hit
		// two hundred times is spared exactly as often as a key hit once. The
		// repeatedly-hit key is older than the once-hit key and the hand
		// reaches it first, so it is recycled first and therefore evicted
		// first. A test that asserted the opposite would be asserting LRU.
		assert!(
			pos(keys[MANY_HITS]) < pos(keys[ONE_HIT]),
			"the repeatedly-hit key outlived the once-hit key, which means \
			 something is counting hits -- CLOCK has one bit, not a counter",
		);
	}
}

/// The acceptance test for `MergedOrder::Lfu`: it is the SAME order
/// [`LfuCompactHybridStack`] implements, key for key, step for step.
///
/// # Two claims, and -- unlike CLOCK's -- they cannot be checked together
///
/// * [`LfuCompactHybridStack`] -- the split-structure statement of this policy.
///   Fed the identical `PolicyStack` calls it must agree with the merged store
///   on the tier of every live key after every step, on the four gauges the
///   tiering manager reads, and on the whole eviction order. This is the claim
///   that 32 shards of frequency buckets reconstruct one global LFU order.
///
/// * `LfuCompactStack` -- the FLAT stack serving `PaperPolicy::LfuCompact`,
///   with no tiers and no shards, which is the arbiter of what the order IS.
///
/// The CLOCK module checks both against ONE sequence. That is not available
/// here, and the reason is worth stating because it looks like a gap. Tiered
/// LFU evicts SLOW FIRST (`LfuCompactHybridStack::evict_one`), and under this
/// policy that genuinely reorders against untiered LFU: once admission latches,
/// newcomers enter the slow tier at frequency 1 while earlier keys sit fast at
/// frequency 1, so the tiered stacks evict the slow one and the flat stack
/// evicts the older fast one. For CLOCK the two never diverge -- tiering moves
/// a cursor along the queue and the globally oldest object is already a slow
/// one whenever anything is slow -- which is why that module can compare all
/// three at once and this one cannot.
///
/// So the flat comparison is made on its own, UNTIERED, in
/// `an_untiered_merged_store_evicts_in_the_flat_lfu_order`. That splits the two
/// questions cleanly: is the order LFU (flat, untiered), and does tiering place
/// and demote the way the reference does (tiered, against the hybrid stack).
/// Folding them together would have meant asserting something false.
///
/// # What each ingredient of the sequence is for
///
/// * **A key hit over and over, and a key hit exactly once.** The pair that
///   distinguishes a COUNTER from a bit. The CLOCK module asserts the opposite
///   of what this one does on the same fixture shape -- there the
///   repeatedly-hit key leaves FIRST, because one bit spares a key exactly as
///   often however many times it was hit -- and getting the two tests to
///   contradict each other is the clearest available statement that the order
///   really is dispatched on.
///
/// * **An overwrite of a key that is never otherwise accessed.** Both
///   references treat a re-insert as a hit, so the overwrite is the only thing
///   that can raise this key's count, and its control was inserted immediately
///   after it.
///
/// * **Hits on demoted slow keys.** The promotion rule: a slow key rejoins the
///   fast tier only by STRICTLY exceeding the fast minimum, so a tie must not
///   promote.
///
/// * **Overwrites that grow a fast key, resize a long-demoted one, and change
///   no bytes at all.** The byte accounting is order-independent and has to
///   stay so while the access it also represents is counted.
///
/// * **Keys spread over three shards.** A single-shard sequence would never
///   exercise the cross-shard `(freq, stamp)` minimum, which is the part of
///   this order that has to reconstruct one global answer out of 32.
///
/// # Why the two are comparable at all
///
/// Identical `(key, size, dram_resident)` calls through `PolicyStack`, the same
/// `drain_target`, no per-object shared overhead on either side, and sizes that
/// are exact jemalloc size classes -- so the merged store's size-class-rounded
/// `migrating()` and the reference's raw `size - dram_resident` are the same
/// number and the accounting cannot drift for a reason unrelated to the order.
///
/// [`LfuCompactHybridStack`]: super::lfu_compact_hybrid_stack::LfuCompactHybridStack
#[cfg(all(test, feature = "lfu_compact_hybrid_cache"))]
mod lfu_order_fidelity {
	use super::*;

	use crate::{
		object::Object,
		worker::policy::policy_stack::{
			lfu_compact_hybrid_stack::LfuCompactHybridStack,
			lfu_compact_stack::LfuCompactStack,
		},
		BufferDRAM,
	};

	use std::collections::HashSet;

	type Store = MergedStore<u64, BufferDRAM>;
	type Handle = MergedStackHandle<u64, BufferDRAM>;

	/// `MergedStore::shard_of` reads the top five bits of the key.
	const SHARD_BITS: u32 = 5;
	const SHARDS: u64 = 1 << SHARD_BITS;

	/// All exact jemalloc size classes -- see `value_len`.
	const SMALL: ObjectSize = 512;
	const MEDIUM: ObjectSize = 1024;
	const LARGE: ObjectSize = 8192;

	const N_KEYS: u64 = 240;

	/// Room for roughly 117 of the 240 small objects, so the budget bites in
	/// the MIDDLE of the sequence: a capacity that demoted all or none of them
	/// would let a wrong order agree by accident.
	const FAST_CAPACITY: CacheSize = 60_000;

	/// Hit at every opportunity, so its count climbs far above everything
	/// else's and it must leave LAST. This is the key whose position the CLOCK
	/// module asserts the opposite about.
	const MANY_HITS: usize = 3;

	/// Hit exactly ONCE, immediately after it is inserted, and never mentioned
	/// again -- so its count is 2 for the whole run.
	const ONE_HIT: usize = 7;

	/// Never hit at all, and inserted immediately after `ONE_HIT` -- the
	/// control for it. Count 1 forever, so it must leave first of the two.
	const NEVER_HIT: usize = 8;

	/// Never HIT, but overwritten exactly once. Both references forward an
	/// existing key to `update`, so the overwrite is the only thing in the
	/// whole run that can raise this key's count.
	const OVERWRITE_ONLY: usize = 9;

	/// Never hit and never overwritten, inserted immediately after
	/// `OVERWRITE_ONLY` -- its control.
	const NEVER_TOUCHED: usize = 10;

	/// Indices the generic hit rule must leave alone, or none of the claims
	/// above would be true of them: `keys[n / 4]` reaches EVERY index
	/// eventually, since any four consecutive integers contain a multiple of
	/// three.
	fn reserved(i: usize) -> bool {
		i == MANY_HITS
			|| i == ONE_HIT
			|| i == NEVER_HIT
			|| i == OVERWRITE_ONLY
			|| i == NEVER_TOUCHED
	}

	fn mix(i: u64) -> HashedKey {
		i.wrapping_mul(0x9E37_79B9_7F4A_7C15)
	}

	/// A key that lands in shard `s`: the mixed value shifted clear of the
	/// shard field, then the shard written into it.
	fn in_shard(s: u64, i: u64) -> HashedKey {
		assert!(s < SHARDS);
		(mix(i) >> SHARD_BITS) | (s << (64 - SHARD_BITS))
	}

	/// The n-th key of the sequence, round-robin over three shards.
	fn key_at(n: u64) -> HashedKey {
		in_shard(n % 3, n)
	}

	/// The value LENGTH whose WHOLE ITEM costs exactly `item` bytes.
	///
	/// The merged store derives what to charge from the object, through
	/// `Slot::migrating` -> `resident_object_bytes`, while the reference stack
	/// is told a figure -- so feeding the reference `item` while the store
	/// rounded something else would charge the two differently and the
	/// comparison would stop being about ORDER.
	fn value_len(item: ObjectSize) -> ObjectSize {
		let len = item - crate::object::overhead::value_header_bytes::<u64>();

		assert_eq!(
			crate::object::overhead::resident_object_bytes::<u64>(len),
			item,
			"a {len}-byte value does not make an item of exactly {item} bytes, \
			 so the merged store and the reference stack are being charged \
			 different numbers and this fixture is not comparing orders",
		);

		len
	}

	/// Spelled out rather than inferred from whether the key happens to be
	/// present: an `Insert` that silently became an overwrite, or an
	/// `Overwrite` whose key had gone, would quietly delete the case the
	/// sequence exists to cover.
	#[derive(Clone, Copy, Debug)]
	enum Op {
		Insert(HashedKey, ObjectSize),
		Hit(HashedKey),
		Overwrite(HashedKey, ObjectSize),
	}

	/// One op, to both structures.
	///
	/// The merged handle needs the object in the map first: in that design
	/// inserting into the map IS inserting into the stack, and
	/// `insert_resident` only settles. The split stack owns its own row, so
	/// `insert_resident` is the whole insert. An overwrite is the same pair of
	/// calls -- which is the point, since neither side is told which it is.
	fn apply(
		store: &Arc<Store>,
		merged: &mut Handle,
		split: &mut LfuCompactHybridStack,
		op: Op,
	) {
		match op {
			Op::Insert(key, size) => {
				assert!(!store.contains(key), "Insert of a key already present");
				assert!(!split.contains(key), "Insert of a key already present");

				store.insert(key, Object::new(key, &vec![0u8; value_len(size) as usize], None));
				merged.insert_resident(key, size, 0);
				split.insert_resident(key, size, 0);
			},

			Op::Hit(key) => {
				assert!(store.contains(key), "Hit on a key that is not present");
				assert!(split.contains(key), "Hit on a key that is not present");

				merged.update(key);
				split.update(key);
			},

			Op::Overwrite(key, size) => {
				assert!(store.contains(key), "Overwrite of a key that is not present");
				assert!(split.contains(key), "Overwrite of a key that is not present");

				store.insert(key, Object::new(key, &vec![0u8; value_len(size) as usize], None));
				merged.insert_resident(key, size, 0);
				split.insert_resident(key, size, 0);
			},
		}
	}

	/// The sequence. Built rather than written out so the interesting ops sit
	/// at positions the budget has already bitten at.
	fn sequence() -> (Vec<HashedKey>, Vec<Op>) {
		let keys: Vec<HashedKey> = (0..N_KEYS).map(key_at).collect();

		let distinct: HashSet<HashedKey> = keys.iter().copied().collect();
		assert_eq!(distinct.len(), keys.len(), "two keys collided");

		let mut ops = Vec::new();

		for (n, &key) in keys.iter().enumerate() {
			ops.push(Op::Insert(key, SMALL));

			// The key that is hit exactly once, as early as it can be.
			if n == ONE_HIT {
				ops.push(Op::Hit(key));
			}

			// Hits on keys already in the cache, from both ends of the order:
			// `n / 4` is old enough to have been demoted once the budget bites
			// -- so it exercises the promotion rule -- and `n` is the newest
			// object there is.
			if n % 3 == 0 {
				if !reserved(n / 4) {
					ops.push(Op::Hit(keys[n / 4]));
				}

				if !reserved(n) {
					ops.push(Op::Hit(key));
				}

				if n > MANY_HITS {
					ops.push(Op::Hit(keys[MANY_HITS]));
				}
			}
		}

		// A fast key -- one of the newest -- grown to 16x its size. Charged to
		// the fast tier, and large enough that the settle that follows demotes
		// a run of lower-frequency objects.
		ops.push(Op::Overwrite(keys[(N_KEYS - 3) as usize], LARGE));

		// A key demoted long ago, resized. Charged to the SLOW tier, and its
		// count rises, so it may or may not promote -- both sides have to reach
		// the same answer.
		ops.push(Op::Overwrite(keys[2], MEDIUM));

		// An overwrite that changes nothing, which the reference stack
		// short-circuits on size and the merged store cannot. It still counts.
		ops.push(Op::Overwrite(keys[5], SMALL));

		// The overwrite that is the ONLY access its key ever gets.
		ops.push(Op::Overwrite(keys[OVERWRITE_ONLY], MEDIUM));

		// Hits after the overwrites, so a count introduced by an overwrite
		// cannot be masked by the sequence ending there.
		ops.push(Op::Hit(keys[(N_KEYS - 3) as usize]));
		ops.push(Op::Hit(keys[2]));

		// The reserved keys really are what they are called. Asserted rather
		// than read off the loop above, because the generic rule reaches far
		// more keys than it looks like it does.
		let hits_on = |i: usize| {
			ops.iter().filter(|op| matches!(op, Op::Hit(k) if *k == keys[i])).count()
		};

		let writes_on = |i: usize| {
			ops.iter().filter(|op| matches!(op, Op::Overwrite(k, _) if *k == keys[i])).count()
		};

		assert_eq!(hits_on(ONE_HIT), 1, "the hit-once key was hit a different number of times");
		assert_eq!(hits_on(NEVER_HIT), 0, "the never-hit key was hit");
		assert_eq!(hits_on(OVERWRITE_ONLY), 0, "the overwrite-only key was hit");
		assert_eq!(
			writes_on(OVERWRITE_ONLY), 1,
			"the overwrite-only key was not overwritten exactly once",
		);
		assert_eq!(hits_on(NEVER_TOUCHED), 0, "the untouched key was hit");
		assert_eq!(writes_on(NEVER_TOUCHED), 0, "the untouched key was overwritten");
		assert!(
			hits_on(MANY_HITS) > 10,
			"the repeatedly-hit key was hit {} times, which is not repeatedly",
			hits_on(MANY_HITS),
		);

		(keys, ops)
	}

	fn build() -> (Arc<Store>, Handle, LfuCompactHybridStack) {
		let store = Arc::new(Store::new());
		let merged = Handle::new(store.clone(), PaperPolicy::LfuCompactHybrid, FAST_CAPACITY * 5)
			.expect("lfu-compact-hybrid is implemented");

		// Override whatever `Handle::new` derived from `max_size`: the
		// experiment needs an exact budget and no per-object reservation on
		// either side, and `LfuCompactHybridStack::new` leaves its own shared
		// overhead at zero.
		store.configure_tiering(FAST_CAPACITY, 0, drain_target_ppm(), drain_target_ppm());

		let split = LfuCompactHybridStack::new(FAST_CAPACITY);

		(store, merged, split)
	}

	/// All three LFU spellings reach the store's order, and the three orders
	/// that were already there are untouched by its arrival.
	#[test]
	fn the_policy_selects_the_lfu_order() {
		for policy in [
			PaperPolicy::Lfu,
			PaperPolicy::LfuCompact,
			PaperPolicy::LfuCompactHybrid,
		] {
			let store = Arc::new(Store::new());
			let _lfu = Handle::new(store.clone(), policy, 1 << 20)
				.expect("every lfu spelling is implemented");

			assert_eq!(
				store.order(),
				crate::merged_store::MergedOrder::Lfu,
				"{policy} did not select the LFU order",
			);
		}

		for (policy, want) in [
			(PaperPolicy::LruCompactHybrid, crate::merged_store::MergedOrder::Lru),
			(PaperPolicy::FifoCompactHybrid, crate::merged_store::MergedOrder::Fifo),
			(PaperPolicy::ClockCompactHybrid, crate::merged_store::MergedOrder::Clock),
		] {
			let store = Arc::new(Store::new());
			let _other = Handle::new(store.clone(), policy, 1 << 20)
				.expect("the three earlier orders are implemented");

			assert_eq!(store.order(), want, "adding LFU moved {policy} off its order");
		}
	}

	#[test]
	fn lfu_matches_the_reference_stack_key_for_key() {
		let (keys, ops) = sequence();
		let (store, mut merged, mut split) = build();

		// 1. Tier placement, after EVERY op, so a failure names the step that
		//    introduced the divergence rather than the end state. This is what
		//    catches an admission rule that differs: the merged store decides a
		//    new key's tier itself, and getting that wrong shows up here one op
		//    after it happens.
		for (n, &op) in ops.iter().enumerate() {
			apply(&store, &mut merged, &mut split, op);

			for &key in &keys {
				if !store.contains(key) {
					continue;
				}

				assert_eq!(
					store.tier_of(key),
					split.tier_of(key),
					"after step {n} ({op:?}) key {key:#018x} is in a different \
					 tier in the merged store than in the reference LFU stack",
				);
			}
		}

		// The sequence has to have actually demoted things, or every tier
		// comparison above was `Some(Fast) == Some(Fast)` and said nothing.
		assert!(
			store.slow_object_count() > 0 && store.fast_object_count() > 0,
			"the budget demoted everything or nothing, so the order is untested",
		);

		// 2. The gauges the tiering manager reads, which is what makes the two
		//    interchangeable to the worker rather than merely agreeing about
		//    tiers.
		assert_eq!(
			store.fast_bytes_used(),
			split.fast_bytes_used(),
			"the two stacks disagree about how many fast bytes they hold",
		);
		assert_eq!(
			store.slow_bytes_used(),
			split.slow_bytes_used(),
			"the two stacks disagree about how many slow bytes they hold",
		);
		assert_eq!(
			store.fast_object_count(),
			split.fast_object_count(),
			"the two stacks disagree about how many fast objects they hold",
		);
		assert_eq!(
			store.slow_object_count(),
			split.slow_object_count(),
			"the two stacks disagree about how many slow objects they hold",
		);

		// 3. The eviction order, to the last key, with the tier of every
		//    survivor checked after every eviction.
		//
		//    `evict_one` on the merged handle only NOMINATES -- the removal is
		//    `take`, exactly as `apply_evictions` pairs them -- where the
		//    reference stack's `evict_one` removes.
		let mut merged_order = Vec::new();
		let mut split_order = Vec::new();

		loop {
			let m = merged.evict_one();
			let s = split.evict_one();

			assert_eq!(
				m, s,
				"the merged store and the reference LFU stack nominated \
				 different victims at position {}",
				merged_order.len(),
			);

			let Some(key) = m else { break };

			assert!(store.take(&key).is_some(), "nominated victim was not present");

			merged_order.push(key);
			split_order.push(s.expect("checked equal to m"));

			for &k in &keys {
				if !store.contains(k) {
					continue;
				}

				assert_eq!(
					store.tier_of(k),
					split.tier_of(k),
					"after evicting {key:#018x} key {k:#018x} is in a different \
					 tier in the merged store than in the reference LFU stack",
				);
			}

			assert_eq!(
				store.fast_bytes_used(),
				split.fast_bytes_used(),
				"fast bytes diverged after evicting {key:#018x}",
			);
			assert_eq!(
				store.fast_object_count(),
				split.fast_object_count(),
				"fast object count diverged after evicting {key:#018x}",
			);
		}

		assert_eq!(merged_order, split_order, "the two orders diverged");
		assert_eq!(store.len(), 0, "the drain left objects behind");

		// 4. The order is NOT insertion order, which is what it would collapse
		//    to if `freq` were never read -- i.e. if this were FIFO wearing an
		//    LFU label, which is the substitution this whole commit deletes.
		assert_ne!(
			merged_order, keys,
			"eviction order is exactly insertion order, so no count was ever \
			 consulted and LFU ran as FIFO",
		);

		// 5. What a COUNTER buys, on the keys the sequence reserved for it.
		let pos = |k: HashedKey| merged_order.iter().position(|&x| x == k).unwrap();

		// The claim that separates this order from CLOCK, and the CLOCK module
		// asserts its exact opposite on the same fixture shape: there the
		// repeatedly-hit key leaves FIRST, because one bit spares a key as
		// often whether it was hit once or two hundred times.
		assert!(
			pos(keys[MANY_HITS]) > pos(keys[ONE_HIT]),
			"the repeatedly-hit key did not outlive the key hit once, so hits \
			 are not being COUNTED -- that is CLOCK's single bit, not LFU",
		);

		// One hit is still worth something: `ONE_HIT` was inserted BEFORE
		// `NEVER_HIT`, so under insertion order it would leave first.
		assert!(
			pos(keys[ONE_HIT]) > pos(keys[NEVER_HIT]),
			"the key hit once did not outlive the key inserted after it, so its \
			 count bought it nothing",
		);

		// And the same claim for a WRITE, which both references say is an
		// access too.
		assert!(
			pos(keys[OVERWRITE_ONLY]) > pos(keys[NEVER_TOUCHED]),
			"the overwritten key did not outlive the key inserted after it, so \
			 an overwrite did not count as an access",
		);
	}

	/// The flat comparison, UNTIERED -- see the module doc for why it cannot be
	/// folded into the test above.
	///
	/// With no fast-tier budget nothing is ever demoted, so slow-preference
	/// never fires and the merged store's order must be exactly the order the
	/// flat `LfuCompactStack` produces from the same accesses. That stack is
	/// the arbiter of what LFU's order IS: it has no tiers and no shards, so
	/// agreeing with it is the claim that neither of the tiered structures
	/// quietly invented a policy.
	///
	/// It also pins the tie-break, which is the half of exactness that a
	/// single-shard fixture could not test: every key that is never hit stays
	/// at frequency 1, so the whole of that bucket has to come out in
	/// insertion order ACROSS SHARDS, ordered by the `(freq, stamp)` pair the
	/// mirrors carry.
	#[test]
	fn an_untiered_merged_store_evicts_in_the_flat_lfu_order() {
		let (keys, ops) = sequence();

		let store = Arc::new(Store::new());

		// `LfuCompact` is the FLAT spelling, so `Handle::new` skips
		// `configure_tiering` entirely and the store keeps its untiered
		// sentinel capacity -- which is also the check that a flat LFU policy
		// resolves to the same order as the hybrid one.
		let mut merged = Handle::new(store.clone(), PaperPolicy::LfuCompact, 1 << 30)
			.expect("lfu-compact is implemented");

		let mut flat = LfuCompactStack::default();

		for &op in &ops {
			match op {
				Op::Insert(key, size) | Op::Overwrite(key, size) => {
					store.insert(
						key,
						Object::new(key, &vec![0u8; value_len(size) as usize], None),
					);
					merged.insert_resident(key, size, 0);
					flat.insert(key, size);
				},

				Op::Hit(key) => {
					merged.update(key);
					flat.update(key);
				},
			}
		}

		// Nothing was demoted, which is the precondition for this comparison
		// to mean what it says.
		assert_eq!(
			store.slow_object_count(), 0,
			"an untiered store demoted something, so slow-preference is in play \
			 and the flat stack is no longer the right reference",
		);

		let mut merged_order = Vec::new();

		while let Some(key) = merged.evict_one() {
			assert!(store.take(&key).is_some(), "nominated victim was not present");
			merged_order.push(key);
		}

		let mut flat_order = Vec::new();

		while let Some(key) = flat.evict_one() {
			flat_order.push(key);
		}

		assert_eq!(
			merged_order, flat_order,
			"the merged store under LFU does not evict in the order the flat \
			 LfuCompactStack does",
		);

		assert_ne!(merged_order, keys, "eviction order is exactly insertion order");
		assert_eq!(store.len(), 0, "the drain left objects behind");
	}
}
