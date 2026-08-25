/*
 * Copyright (c) Kia Shakiba
 *
 * This source code is licensed under the GNU AGPLv3 license found in the
 * LICENSE file in the root directory of this source tree.
 */

pub mod overhead;

use std::{
	mem,
	sync::Arc,
	time::{Instant, Duration},
};

use typesize::TypeSize;

pub type ObjectSize = u32;
pub type ExpireTime = Option<Instant>;

#[derive(Clone)]
pub struct Object<K, V> {
	/// The key stored in DRAM.  Present only when `key_pmem_value_pmem` is
	/// **not** enabled; when the feature is active the key lives exclusively
	/// in persistent memory via `_key_pmem` below.
	#[cfg(not(feature = "key_pmem_value_pmem"))]
	key: K,

	/// When `key_pmem_value_pmem` is enabled the key is owned here, allocated
	/// directly in persistent memory via the Hybrid allocator.  There is no
	/// separate DRAM copy of the key in this configuration.
	#[cfg(feature = "key_pmem_value_pmem")]
	_key_pmem: Box<K, crate::Hybrid>,

	data: Arc<V>,
	expiry: ExpireTime,
}

impl<K, V> Object<K, V> {
	/// Create a new Object.
	///
	/// When `key_pmem_value_pmem` is **not** enabled this method has no
	/// additional bounds.
	#[cfg(not(feature = "key_pmem_value_pmem"))]
	pub fn new(key: K, data: V, ttl: Option<u32>) -> Self {
		let expiry = match ttl {
			Some(0) | None => None,
			Some(ttl) => Some(get_expiry_from_ttl(ttl)),
		};

		Object {
			key,
			data: Arc::new(data),
			expiry,
		}
	}

	/// Create a new Object.
	///
	/// When `key_pmem_value_pmem` is enabled the key is moved directly into a
	/// `Box` allocated in persistent memory via the Hybrid allocator.  No DRAM
	/// copy of the key is retained.
	#[cfg(feature = "key_pmem_value_pmem")]
	pub fn new(key: K, data: V, ttl: Option<u32>) -> Self {
		use crate::Hybrid;

		let expiry = match ttl {
			Some(0) | None => None,
			Some(ttl) => Some(get_expiry_from_ttl(ttl)),
		};

		Object {
			_key_pmem: Box::new_in(key, Hybrid),
			data: Arc::new(data),
			expiry,
		}
	}

	/// Create a new Object with an explicit expiry time.
	///
	/// When `key_pmem_value_pmem` is **not** enabled this method has no
	/// additional bounds.
	#[cfg(not(feature = "key_pmem_value_pmem"))]
	pub fn with_expiry(key: K, data: V, expiry: ExpireTime) -> Self {
		Object {
			key,
			data: Arc::new(data),
			expiry,
		}
	}

	/// Create a new Object with an explicit expiry time.
	///
	/// When `key_pmem_value_pmem` is enabled the key is moved directly into
	/// PMEM; no DRAM copy is retained.
	#[cfg(feature = "key_pmem_value_pmem")]
	pub fn with_expiry(key: K, data: V, expiry: ExpireTime) -> Self {
		use crate::Hybrid;

		Object {
			_key_pmem: Box::new_in(key, Hybrid),
			data: Arc::new(data),
			expiry,
		}
	}

	pub fn data(&self) -> Arc<V> {
		self.data.clone()
	}

	/// Replaces this object's data in place, leaving `key` and `expiry`
	/// untouched.
	///
	/// Used by `lru_hybrid_cache` to physically migrate an object's bytes
	/// between tiers (e.g. `TieredBuffer::Fast` <-> `TieredBuffer::Slow`)
	/// without disturbing its TTL or key.
	pub fn set_data(&mut self, data: V) {
		self.data = Arc::new(data);
	}

	/// Return a reference to the key.
	///
	/// Without `key_pmem_value_pmem` the key lives in DRAM.
	/// With `key_pmem_value_pmem` the key lives in PMEM and is accessed via
	/// the `_key_pmem` box; no DRAM copy exists.
	/// The key's own byte cost, as `base_size` counts it.
	pub fn key_size(&self) -> ObjectSize
	where
		K: TypeSize,
	{
		self.key().get_size() as ObjectSize
	}

	#[cfg(not(feature = "key_pmem_value_pmem"))]
	pub fn key(&self) -> &K {
		&self.key
	}

	#[cfg(feature = "key_pmem_value_pmem")]
	pub fn key(&self) -> &K {
		&self._key_pmem
	}

	/// Check whether this object's key matches the given key.
	///
	/// When `key_pmem_value_pmem` is enabled the comparison reads the key from
	/// PMEM, ensuring that set/get/delete operations all verify against the
	/// PMEM-resident copy.
	#[cfg(not(feature = "key_pmem_value_pmem"))]
	pub fn key_matches(&self, key: &K) -> bool
	where
		K: Eq,
	{
		self.key.eq(key)
	}

	#[cfg(feature = "key_pmem_value_pmem")]
	pub fn key_matches(&self, key: &K) -> bool
	where
		K: Eq,
	{
		(*self._key_pmem).eq(key)
	}

	#[cfg(not(feature = "key_pmem_value_pmem"))]
	fn total_size(&self) -> ObjectSize
	where
		K: TypeSize,
		V: TypeSize,
	{
		(
			self.key.get_size()
				+ self.data.get_size()
				+ mem::size_of::<ExpireTime>()
		) as ObjectSize
	}

	#[cfg(feature = "key_pmem_value_pmem")]
	fn total_size(&self) -> ObjectSize
	where
		K: TypeSize,
		V: TypeSize,
	{
		(
			(*self._key_pmem).get_size()
				+ self.data.get_size()
				+ mem::size_of::<ExpireTime>()
		) as ObjectSize
	}

	pub fn expiry(&self) -> ExpireTime {
		self.expiry
	}

	pub fn is_expired(&self) -> bool {
		self.expiry.is_some_and(|expiry| expiry <= Instant::now())
	}

	pub fn expires(&mut self, ttl: Option<u32>) {
		self.expiry = match ttl {
			Some(0) | None => None,
			Some(ttl) => Some(get_expiry_from_ttl(ttl)),
		};
	}
}

pub fn get_expiry_from_ttl(ttl: u32) -> Instant {
	Instant::now() + Duration::from_secs(ttl.into())
}

