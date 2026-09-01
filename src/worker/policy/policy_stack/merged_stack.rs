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
//! # Tiering
//!
//! The seven tiering methods forward to the store rather than taking their
//! trait defaults, so the merged store is a genuine hybrid: it demotes at the
//! same watermarks, emits the same `(key, Tier)` migrations for
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
			use crate::worker::policy::policy_stack::watermarks;

			let to_ppm = |f: f64| (f * 1_000_000.0) as u64;

			store.configure_tiering(
				// Same default fast-tier budget as every hybrid stack: 20% of
				// the overall cache size, runtime-adjustable afterward through
				// `resize_fast_tier`.
				(max_size as f64 * 0.2) as CacheSize,
				crate::object::overhead::get_hybrid_dram_shared_overhead(&policy) as CacheSize,
				to_ppm(watermarks::high()),
				to_ppm(watermarks::low()),
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

	fn remove(&mut self, key: HashedKey) {
		self.store.remove_key(key);
	}

	fn clear(&mut self) {
		self.store.clear();
	}

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
