/*
 * Copyright (c) Kia Shakiba
 *
 * This source code is licensed under the GNU AGPLv3 license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! What `erase` removes an object with, per object map: the one place the
//! DashMap and the merged store differ on a delete, an eviction or a reap.
//!
//! The check and the removal happen under ONE lock, so a hash collision does
//! not remove the wrong object and a reap does not take an object that is live
//! again (`EraseKey::Original` and `EraseKey::Expired`). Checking through
//! `get_ref` and then removing would release the lock in between, and a `set`
//! landing there would have its new object removed.

use dashmap::{
	DashMap,
	mapref::entry::Entry,
};
use log::error;

use crate::{CacheError, HashedKey, NoHasher};
use crate::object::Object;

/// The removal half of an object map. See the module doc.
pub trait Removal<K, V> {
	/// Removes and returns the object at `key` if `pred` holds of it, the check
	/// and the removal under one lock.
	fn take_if(&self, key: &HashedKey, pred: impl FnOnce(&Object<K, V>) -> bool) -> Option<Object<K, V>>;

	/// Removes and returns the object at `key`, live or not: capacity
	/// eviction's path (`EraseKey::Hashed`), where the stack chose the victim
	/// and it must go. A store whose index is its eviction order answers `None`
	/// for a key its policy worker has not linked.
	fn take_evict(&self, key: &HashedKey) -> Option<Object<K, V>>;

	/// The key `erase` evicts when it was given none -- the stack ran out of
	/// candidates while the map did not, a stack behind its map. `Internal` if
	/// there is none to name. `fallbacks` is the calling cache's own count of
	/// such evictions (`migstats::Stats::erase_fallbacks`), which a store that
	/// takes this path counts in.
	fn fallback_victim(&self, fallbacks: &std::sync::atomic::AtomicU64) -> Result<HashedKey, CacheError>;
}

impl<K, V> Removal<K, V> for DashMap<HashedKey, Object<K, V>, NoHasher> {
	/// Under the shard's write lock, which the check and the removal share.
	///
	/// Through `entry`, not `remove_if`: `entry` reserves room for a new
	/// object before it looks the key up, so an eviction can grow the victim's
	/// shard table, and `tests/dram_metadata_identity.rs` holds the measured
	/// DRAM to exactly those steps.
	fn take_if(&self, key: &HashedKey, pred: impl FnOnce(&Object<K, V>) -> bool) -> Option<Object<K, V>> {
		let Entry::Occupied(entry) = self.entry(*key) else {
			return None;
		};

		pred(entry.get()).then(|| entry.remove())
	}

	/// Through `entry` too -- see `take_if`.
	fn take_evict(&self, key: &HashedKey) -> Option<Object<K, V>> {
		let Entry::Occupied(entry) = self.entry(*key) else {
			return None;
		};

		Some(entry.remove())
	}

	/// Whatever object the map iterates first.
	///
	/// INSTRUMENTATION: this path removes an object from the MAP without
	/// informing the eviction STACK, which is exactly the shape of the observed
	/// map>stack divergence. Counted so the hypothesis is testable rather than
	/// plausible: in the cache's own `fallbacks` and in the process-wide
	/// `crate::ERASE_FALLBACK`.
	fn fallback_victim(&self, fallbacks: &std::sync::atomic::AtomicU64) -> Result<HashedKey, CacheError> {
		crate::ERASE_FALLBACK.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
		fallbacks.fetch_add(1, std::sync::atomic::Ordering::Relaxed);

		// The stack has run out of keys to evict while the map has not (a stack
		// behind its map), so this falls back to evicting a random object.
		let Some(object) = self.iter().next() else {
			error!("Object store is empty with non-zero used size");
			return Err(CacheError::Internal);
		};

		Ok(object.key().to_owned())
	}
}

#[cfg(feature = "merged_object_store")]
impl<K, V> Removal<K, V> for crate::merged_store::MergedStore<K, V> {
	fn take_if(&self, key: &HashedKey, pred: impl FnOnce(&Object<K, V>) -> bool) -> Option<Object<K, V>> {
		crate::merged_store::MergedStore::take_if(self, key, pred)
	}

	fn take_evict(&self, key: &HashedKey) -> Option<Object<K, V>> {
		crate::merged_store::MergedStore::take_evict(self, key)
	}

	/// None: the eviction order is the store itself, which cannot run behind
	/// its own map.
	fn fallback_victim(&self, _fallbacks: &std::sync::atomic::AtomicU64) -> Result<HashedKey, CacheError> {
		Err(CacheError::Internal)
	}
}
