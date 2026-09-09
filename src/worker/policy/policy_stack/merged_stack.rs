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
//! So `evict_one` NOMINATES the victim -- the globally least-recently-used key,
//! `SHARDS` atomic loads and no lock -- and the removal happens exactly once,
//! inside `erase`'s `take`, which unlinks it from the recency order and
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
//! # Tiering
//!
//! The seven tiering methods forward to the store rather than taking their
//! trait defaults, so the merged store is a genuine hybrid: it demotes at the
//! same ceiling, emits the same `(key, Tier)` migrations for
//! `apply_tier_migrations` to physically perform, and publishes the same
//! gauges. Comparing it against `lru-compact-hybrid` is therefore
//! like-for-like, which comparing the untiered prototype against a tiered
//! stack was not.

use crate::{
	merged_store::MergedStore,
	object::ObjectSize,
	worker::policy::policy_stack::{CacheSize, HashedKey, PolicyStack, Tier},
	PaperPolicy,
};

use std::sync::Arc;

/// `MergedStore::configure_tiering` still takes a high/low pair in parts per
/// million. The split stacks drain to exactly their ceiling, so the merged
/// store is configured the same way: both marks at the ceiling.
const DRAIN_TO_CEILING_PPM: u64 = 1_000_000;

pub struct MergedStackHandle<K, V> {
	store: Arc<MergedStore<K, V>>,

	/// The configured policy, reported verbatim by `is_policy`. The merged
	/// store is a build-time object-map shape rather than a policy, so it
	/// answers to whichever policy the cache was configured with and never
	/// triggers a stack reconstruction.
	policy: PaperPolicy,
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
				DRAIN_TO_CEILING_PPM,
				DRAIN_TO_CEILING_PPM,
			);
		}

		let _ = max_size;

		// The merged store implements ONE eviction order -- recency -- because
		// that order is the object map's own link structure. `is_policy` still
		// answers to the configured policy so nothing tries to reconstruct a
		// stack that has no separate existence, which means a merged build asked
		// for, say, `lfu-compact-hybrid` would run LRU under an LFU label. Say so
		// loudly rather than reporting a miss ratio against the wrong name.
		if !matches!(
			policy,
			PaperPolicy::Lru | PaperPolicy::LruCompact,
		) && !format!("{policy}").starts_with("lru") {
			log::warn!(
				"merged_object_store implements LRU; running it as {policy} will \
				 report LRU behaviour under that policy's name",
			);
		}

		MergedStackHandle { store, policy }
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

	fn update(&mut self, key: HashedKey) {
		self.store.touch(key);
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
/// then a settle), and both drain to exactly the same ceiling, so nothing
/// but the boundary rule differs. The sizes are exact jemalloc size
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
			DRAIN_TO_CEILING_PPM,
			DRAIN_TO_CEILING_PPM,
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
			DRAIN_TO_CEILING_PPM,
			DRAIN_TO_CEILING_PPM,
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
