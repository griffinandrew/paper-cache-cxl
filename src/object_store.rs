/*
 * Copyright (c) Kia Shakiba
 *
 * This source code is licensed under the GNU AGPLv3 license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! The object-map storage shape (`ObjectMapRef<K, V>` in `lib.rs`) behind a
//! trait, so `PaperCache`'s `get`/`set`/`del`/`has`/`peek`/`ttl`/`size`/`wipe`
//! bodies are written once, generic over the value buffer.
//!
//! The one shape, in `all_dram` and `key_value_pmem` builds:
//! `Arc<DashMap<HashedKey, Object<K, V>, NoHasher>>`. DashMap shards its own
//! locking internally, so `get`/`get_mut` need no external guard. (The
//! `RwLock<HashMap>` shape that `hashbrown_dram` and `global_hashtable_pmem`
//! selected was removed in R2.)
//!
//! `get_ref`/`get_mut` return `impl Deref`/`DerefMut` rather than a boxed
//! trait object, so DashMap's `Ref`/`RefMut` guards cost nothing extra
//! (return-position `impl Trait` in traits, stable since Rust 1.75).

use std::ops::{Deref, DerefMut};

use dashmap::DashMap;

use crate::{CacheSize, HashedKey, NoHasher, Tier};
use crate::object::{Object, ObjectSize};
use crate::status::Cleared;

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

	/// Removes every object, resetting the store to empty, and returns what it
	/// removed: the objects, and the base bytes `base_size` gives each
	/// (`OverheadManager::base_size`), for `AtomicStatus::clear` to take off.
	/// Each object is counted under the lock that removes it, so an insert
	/// racing the clear is counted exactly when it is removed: one landing in
	/// a part of the map the clear has already emptied stays, and is not in
	/// the count. A different name from `DashMap`'s inherent `clear`, which
	/// method resolution would otherwise pick, and which counts nothing.
	fn clear_counted(&self, base_size: impl Fn(&Object<K, V>) -> ObjectSize) -> Cleared;

	/// Returns the number of objects currently tracked.
	fn len(&self) -> usize;

	/// Calls `f(key, tier, len)` for every object: its hashed key, the tier
	/// its value's bytes are in (the value's tag) and the value's length. The
	/// placement audit's walk (`PaperCache::placement_audit`), a diagnostic:
	/// it holds each shard's read lock while it reads that shard, and calls
	/// `f` under it, so `f` must not touch the store.
	fn for_each_value(&self, f: impl FnMut(HashedKey, Tier, ObjectSize));
}

// ---------------------------------------------------------------------
// DashMap (internally sharded, no external lock needed)
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

	/// Shard by shard, each under its write lock (`retain`), each object
	/// counted as it is dropped.
	fn clear_counted(&self, base_size: impl Fn(&Object<K, V>) -> ObjectSize) -> Cleared {
		let mut cleared = Cleared::default();

		DashMap::retain(self, |_, object| {
			cleared.objects += 1;
			cleared.base_bytes += base_size(object) as CacheSize;

			false
		});

		cleared
	}

	fn len(&self) -> usize {
		DashMap::len(self)
	}

	fn for_each_value(&self, mut f: impl FnMut(HashedKey, Tier, ObjectSize)) {
		for entry in self.iter() {
			let object = entry.value();

			f(*entry.key(), object.value().tier(), object.data_size());
		}
	}
}
