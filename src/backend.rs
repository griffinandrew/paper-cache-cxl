/*
 * Copyright (c) Kia Shakiba
 *
 * This source code is licensed under the GNU AGPLv3 license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! Backend abstraction for the `PaperCache` object store.
//!
//! [`CacheMap`] is a thin trait that each compiled `ObjectMapRef<K, V>` type
//! (DashMap, RwLock-wrapped HashMap, or RwLock-wrapped FlatMap) must
//! implement.  The shared `impl PaperCache` block in `lib.rs` uses this trait
//! so that methods like `get`, `del`, `has`, `peek`, `ttl`, `size`, `wipe`,
//! `resize`, `policy`, `broadcast`, and `hash_key` only need to be written
//! once.

use std::sync::{Arc, RwLock};

use crate::{
	HashedKey,
	object::Object,
};

// ── Public trait ─────────────────────────────────────────────────────────────

/// Abstracts read / write / clear operations over the compiled
/// `ObjectMapRef<K, V>` backend so that the shared `impl PaperCache` block
/// can call them without knowing the concrete map type.
pub trait CacheMap<K, V> {
	/// Call `f` with a shared reference to the *live* (non-expired, matching)
	/// object stored under `hashed_key`.  Returns `None` when no such object
	/// exists.
	fn cm_with_object<R, F>(&self, hashed_key: HashedKey, key: &K, f: F) -> Option<R>
	where
		K: Eq,
		F: FnOnce(&Object<K, V>) -> R;

	/// Call `f` with a mutable reference to the *live* (non-expired, matching)
	/// object stored under `hashed_key`.  Returns `None` when no such object
	/// exists.
	fn cm_with_object_mut<R, F>(&self, hashed_key: HashedKey, key: &K, f: F) -> Option<R>
	where
		K: Eq,
		F: FnOnce(&mut Object<K, V>) -> R;

	/// Clear every object from the map.
	fn cm_clear(&self);
}

// ── DashMap backend (default / all_dram / key_value_pmem without RwLock) ─────

#[cfg(not(any(
	feature = "global_hashtable_pmem",
	feature = "global_flatmap_dram",
	feature = "global_flatmap_pmem",
	feature = "hashbrown_dram",
)))]
impl<K, V> CacheMap<K, V>
	for Arc<dashmap::DashMap<HashedKey, Object<K, V>, crate::NoHasher>>
{
	fn cm_with_object<R, F>(&self, hashed_key: HashedKey, key: &K, f: F) -> Option<R>
	where
		K: Eq,
		F: FnOnce(&Object<K, V>) -> R,
	{
		match self.get(&hashed_key) {
			Some(obj) if obj.key_matches(key) && !obj.is_expired() => Some(f(&*obj)),
			_ => None,
		}
	}

	fn cm_with_object_mut<R, F>(&self, hashed_key: HashedKey, key: &K, f: F) -> Option<R>
	where
		K: Eq,
		F: FnOnce(&mut Object<K, V>) -> R,
	{
		match self.get_mut(&hashed_key) {
			Some(mut obj) if obj.key_matches(key) && !obj.is_expired() => Some(f(&mut *obj)),
			_ => None,
		}
	}

	fn cm_clear(&self) {
		self.clear();
	}
}

// ── RwLock-wrapped backends (HashMap + FlatMap) ───────────────────────────────

/// Helper for the map type stored *inside* an `Arc<RwLock<M>>`.
///
/// Implementations are compiled only for the relevant feature flag so
/// exactly one `impl` is active at a time.
trait InnerMap<K, V> {
	fn inner_get(&self, key: &HashedKey) -> Option<&Object<K, V>>;
	fn inner_get_mut(&mut self, key: &HashedKey) -> Option<&mut Object<K, V>>;
	fn inner_clear(&mut self);
}

/// Blanket `CacheMap` impl for every `Arc<RwLock<M>>` whose inner map
/// implements `InnerMap<K, V>`.
impl<K, V, M> CacheMap<K, V> for Arc<RwLock<M>>
where
	M: InnerMap<K, V>,
{
	fn cm_with_object<R, F>(&self, hashed_key: HashedKey, key: &K, f: F) -> Option<R>
	where
		K: Eq,
		F: FnOnce(&Object<K, V>) -> R,
	{
		let guard = self.read().unwrap();
		match guard.inner_get(&hashed_key) {
			Some(obj) if obj.key_matches(key) && !obj.is_expired() => Some(f(obj)),
			_ => None,
		}
	}

	fn cm_with_object_mut<R, F>(&self, hashed_key: HashedKey, key: &K, f: F) -> Option<R>
	where
		K: Eq,
		F: FnOnce(&mut Object<K, V>) -> R,
	{
		let mut guard = self.write().unwrap();
		let obj = guard.inner_get_mut(&hashed_key)?;
		if !obj.key_matches(key) || obj.is_expired() {
			return None;
		}
		Some(f(obj))
	}

	fn cm_clear(&self) {
		self.write().unwrap().inner_clear();
	}
}

// ── InnerMap impls ────────────────────────────────────────────────────────────

// 1. PMEM HashMap  (global_hashtable_pmem, no FlatMap)
#[cfg(all(feature = "global_hashtable_pmem", not(feature = "global_flatmap_pmem")))]
impl<K, V> InnerMap<K, V>
	for hashbrown::HashMap<
		HashedKey,
		Object<K, V>,
		std::hash::BuildHasherDefault<nohash_hasher::NoHashHasher<HashedKey>>,
		crate::allocator::Hybrid,
	>
{
	fn inner_get(&self, key: &HashedKey) -> Option<&Object<K, V>> {
		self.get(key)
	}

	fn inner_get_mut(&mut self, key: &HashedKey) -> Option<&mut Object<K, V>> {
		self.get_mut(key)
	}

	fn inner_clear(&mut self) {
		self.clear();
	}
}

// 2. DRAM FlatMap  (global_flatmap_dram)
#[cfg(feature = "global_flatmap_dram")]
impl<K, V> InnerMap<K, V>
	for crate::flatmap::FlatMapWithHasher<HashedKey, Object<K, V>, crate::NoHasher>
{
	fn inner_get(&self, key: &HashedKey) -> Option<&Object<K, V>> {
		self.get(key)
	}

	fn inner_get_mut(&mut self, key: &HashedKey) -> Option<&mut Object<K, V>> {
		self.get_mut(key)
	}

	fn inner_clear(&mut self) {
		self.clear();
	}
}

// 3. PMEM FlatMap  (global_flatmap_pmem)
#[cfg(feature = "global_flatmap_pmem")]
impl<K, V> InnerMap<K, V>
	for crate::flatmap::FlatMapWithHasher<
		HashedKey,
		Object<K, V>,
		crate::NoHasher,
		crate::allocator::Hybrid,
	>
{
	fn inner_get(&self, key: &HashedKey) -> Option<&Object<K, V>> {
		self.get(key)
	}

	fn inner_get_mut(&mut self, key: &HashedKey) -> Option<&mut Object<K, V>> {
		self.get_mut(key)
	}

	fn inner_clear(&mut self) {
		self.clear();
	}
}

// 4. DRAM hashbrown HashMap  (hashbrown_dram)
#[cfg(feature = "hashbrown_dram")]
impl<K, V> InnerMap<K, V>
	for hashbrown::HashMap<
		HashedKey,
		Object<K, V>,
		std::hash::BuildHasherDefault<nohash_hasher::NoHashHasher<HashedKey>>,
	>
{
	fn inner_get(&self, key: &HashedKey) -> Option<&Object<K, V>> {
		self.get(key)
	}

	fn inner_get_mut(&mut self, key: &HashedKey) -> Option<&mut Object<K, V>> {
		self.get_mut(key)
	}

	fn inner_clear(&mut self) {
		self.clear();
	}
}
