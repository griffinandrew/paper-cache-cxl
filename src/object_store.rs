/*
 * Copyright (c) Kia Shakiba
 *
 * This source code is licensed under the GNU AGPLv3 license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! Abstracts over the two remaining object-map storage shapes
//! (`ObjectMapRef<K, V>`'s two cfg'd arms in `lib.rs`) behind one trait, so
//! `PaperCache`'s `get`/`set`/`del`/`has`/`peek`/`ttl`/`size`/`wipe` bodies
//! can be written once instead of once per shape.
//!
//! With the FlatMap backend removed, exactly two shapes remain:
//!
//! - **Shape A** (default: `all_dram`, or `key_value_pmem` without
//!   `global_hashtable_pmem`): `Arc<DashMap<HashedKey, Object<K, V>,
//!   NoHasher>>`. DashMap shards its own locking internally, so
//!   `get`/`get_mut` need no external guard.
//! - **Shape B** (`global_hashtable_pmem`, `hashbrown_dram`, or
//!   `key_value_pmem` + `global_hashtable_pmem` together):
//!   `Arc<RwLock<HashMap<HashedKey, Object<K, V>, NoHasher, A>>>`, generic
//!   over the allocator `A` (`Hybrid` for `global_hashtable_pmem`, the
//!   default `Global` allocator for `hashbrown_dram`) -- one impl covers
//!   both, since they only differ in `A`.
//!
//! `get_ref`/`get_mut` return `impl Deref`/`DerefMut` rather than a boxed
//! trait object so the common case (DashMap, which already returns a
//! `Deref`-able `Ref`/`RefMut` guard) costs nothing extra; Shape B wraps its
//! `RwLock` guard in a small helper type that re-indexes on `Deref` so the
//! two shapes can share one return type shape (return-position `impl Trait`
//! in traits, stable since Rust 1.75).

use std::hash::BuildHasherDefault;
// `hashbrown::HashMap`'s allocator parameter is bounded by
// `allocator_api2`'s `Allocator` trait (not the nightly `std::alloc::
// Allocator`) -- this crate's `Hybrid` allocators all implement that one
// (see `src/allocator.rs`'s `allocator_api2::alloc::Allocator` impls).
use allocator_api2::alloc::Allocator;
use std::ops::{Deref, DerefMut};
use std::sync::RwLock;

use dashmap::DashMap;
use hashbrown::HashMap;
use nohash_hasher::NoHashHasher;

use crate::{HashedKey, NoHasher};
use crate::object::Object;

/// Common operations `PaperCache`'s generic impl blocks need from the
/// object map, independent of whether it's backed by a `DashMap` or an
/// externally-locked `HashMap`.
pub trait ObjectStore<K, V> {
	/// Returns a read-only handle to the object at `key`, if present.
	fn get_ref(&self, key: &HashedKey) -> Option<impl Deref<Target = Object<K, V>> + '_>;

	/// Returns a mutable handle to the object at `key`, if present.
	fn get_mut_ref(&self, key: &HashedKey) -> Option<impl DerefMut<Target = Object<K, V>> + '_>;

	/// Inserts `object` at `key`, returning the previous object if one
	/// existed.
	fn insert(&self, key: HashedKey, object: Object<K, V>) -> Option<Object<K, V>>;

	/// Removes and returns every object, resetting the store to empty.
	fn clear(&self);

	/// Returns the number of objects currently tracked.
	fn len(&self) -> usize;
}

// ---------------------------------------------------------------------
// Shape A: DashMap (internally sharded, no external lock needed)
// ---------------------------------------------------------------------

impl<K, V> ObjectStore<K, V> for DashMap<HashedKey, Object<K, V>, NoHasher> {
	fn get_ref(&self, key: &HashedKey) -> Option<impl Deref<Target = Object<K, V>> + '_> {
		self.get(key)
	}

	fn get_mut_ref(&self, key: &HashedKey) -> Option<impl DerefMut<Target = Object<K, V>> + '_> {
		self.get_mut(key)
	}

	fn insert(&self, key: HashedKey, object: Object<K, V>) -> Option<Object<K, V>> {
		DashMap::insert(self, key, object)
	}

	fn clear(&self) {
		DashMap::clear(self)
	}

	fn len(&self) -> usize {
		DashMap::len(self)
	}
}

// ---------------------------------------------------------------------
// Shape B: RwLock<HashMap<..., A>>, generic over the allocator -- covers
// both `global_hashtable_pmem` (A = Hybrid) and `hashbrown_dram`
// (A = std::alloc::Global, via HashMap's own default) in one impl.
// ---------------------------------------------------------------------

/// Read guard for Shape B: holds the `RwLock` read guard and re-indexes on
/// `Deref` so callers see a plain `&Object<K, V>`, matching what DashMap's
/// `Ref` already gives them for free.
pub struct RwLockObjectRef<'a, K, V, A: Allocator> {
	guard: std::sync::RwLockReadGuard<'a, HashMap<HashedKey, Object<K, V>, NoHasher, A>>,
	key: HashedKey,
}

impl<'a, K, V, A: Allocator> Deref for RwLockObjectRef<'a, K, V, A> {
	type Target = Object<K, V>;

	fn deref(&self) -> &Object<K, V> {
		// Presence was already confirmed by `get_ref` before this was
		// constructed; the guard is held for the wrapper's whole lifetime,
		// so the entry cannot have been removed in the meantime.
		self.guard.get(&self.key).expect("key present at construction")
	}
}

/// Write-guard analogue of [`RwLockObjectRef`].
pub struct RwLockObjectMut<'a, K, V, A: Allocator> {
	guard: std::sync::RwLockWriteGuard<'a, HashMap<HashedKey, Object<K, V>, NoHasher, A>>,
	key: HashedKey,
}

impl<'a, K, V, A: Allocator> Deref for RwLockObjectMut<'a, K, V, A> {
	type Target = Object<K, V>;

	fn deref(&self) -> &Object<K, V> {
		self.guard.get(&self.key).expect("key present at construction")
	}
}

impl<'a, K, V, A: Allocator> DerefMut for RwLockObjectMut<'a, K, V, A> {
	fn deref_mut(&mut self) -> &mut Object<K, V> {
		self.guard.get_mut(&self.key).expect("key present at construction")
	}
}

impl<K, V, A: Allocator> ObjectStore<K, V>
	for RwLock<HashMap<HashedKey, Object<K, V>, BuildHasherDefault<NoHashHasher<HashedKey>>, A>>
{
	fn get_ref(&self, key: &HashedKey) -> Option<impl Deref<Target = Object<K, V>> + '_> {
		let guard = self.read().unwrap();

		if guard.contains_key(key) {
			Some(RwLockObjectRef { guard, key: *key })
		} else {
			None
		}
	}

	fn get_mut_ref(&self, key: &HashedKey) -> Option<impl DerefMut<Target = Object<K, V>> + '_> {
		let guard = self.write().unwrap();

		if guard.contains_key(key) {
			Some(RwLockObjectMut { guard, key: *key })
		} else {
			None
		}
	}

	fn insert(&self, key: HashedKey, object: Object<K, V>) -> Option<Object<K, V>> {
		self.write().unwrap().insert(key, object)
	}

	fn clear(&self) {
		self.write().unwrap().clear()
	}

	fn len(&self) -> usize {
		self.read().unwrap().len()
	}
}
