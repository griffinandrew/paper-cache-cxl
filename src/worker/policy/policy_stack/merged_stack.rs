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
//! from not overriding `PolicyStack::update`. A policy whose order the store
//! does not implement is reported on stderr and run as LRU -- see `new`.
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

use crate::{
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
	) -> Self {
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
		// others -- today recency and insertion order. `is_policy` still
		// answers to the configured policy so nothing tries to reconstruct a
		// stack that has no separate existence, which means a merged build
		// asked for, say, `lfu-compact-hybrid` runs LRU under an LFU label.
		//
		// `MergedOrder::from_policy` returning `None` is exactly that case, and
		// it is reported on STDERR rather than through `log`: the warning that
		// used to be here was a `log::warn!`, the server installs no logger, so
		// in the one binary where this mislabelling can actually happen the
		// warning had never once been printed. A mislabelled run is worse than
		// a failed one -- it produces a plausible miss ratio filed under the
		// wrong policy name -- so it is worth a line the user cannot miss.
		let order = match MergedOrder::from_policy(&policy) {
			Some(order) => order,

			None => {
				eprintln!(
					"WARNING: merged_object_store implements LRU and FIFO only; \
					 running it as {policy} executes LRU and reports LRU \
					 behaviour under that policy's name",
				);

				MergedOrder::Lru
			},
		};

		store.set_order(order);

		MergedStackHandle { store, policy, order }
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
			let value = vec![0u8; size as usize];

			store.insert(key, Object::new(key, &value, None));
			merged.insert_resident(key, size, 0);
			split.insert_resident(key, size, 0);
		}
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
			Handle::new(store.clone(), PaperPolicy::LruCompactHybrid, FAST_CAPACITY * 5);

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
			Handle::new(store.clone(), PaperPolicy::LruCompactHybrid, FAST_CAPACITY * 5);

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

				store.insert(key, Object::new(key, &vec![0u8; size as usize], None));
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

				store.insert(key, Object::new(key, &vec![0u8; size as usize], None));
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
			Handle::new(store.clone(), PaperPolicy::FifoCompactHybrid, FAST_CAPACITY * 5);

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
		let _fifo = Handle::new(store.clone(), PaperPolicy::FifoCompactHybrid, 1 << 20);

		assert_eq!(
			store.order(),
			crate::merged_store::MergedOrder::Fifo,
			"fifo-compact-hybrid did not select the FIFO order",
		);

		let store = Arc::new(Store::new());
		let _lru = Handle::new(store.clone(), PaperPolicy::LruCompactHybrid, 1 << 20);

		assert_eq!(
			store.order(),
			crate::merged_store::MergedOrder::Lru,
			"lru-compact-hybrid did not select the LRU order",
		);

		// An order the merged store does not implement falls back to recency --
		// loudly, on stderr, which is the part that cannot be asserted here.
		let store = Arc::new(Store::new());
		let _mislabelled = Handle::new(store.clone(), PaperPolicy::LfuCompactHybrid, 1 << 20);

		assert_eq!(
			store.order(),
			crate::merged_store::MergedOrder::Lru,
			"an unimplemented policy must fall back to recency, not to nothing",
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
