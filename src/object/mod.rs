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
	key: K,
	data: Arc<V>,

	expiry: ExpireTime,

	/// When `key_pmem_value_pmem` is enabled, a copy of the key is allocated in
	/// persistent memory via the Hybrid allocator so that both key data and
	/// value data each reside in PMEM.  The field is prefixed with `_` because
	/// its purpose is solely to own the PMEM allocation for the key; the
	/// live copy of the key used for lookups is the `key` field above.
	#[cfg(feature = "key_pmem_value_pmem")]
	_key_pmem: Box<K, crate::Hybrid>,
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
	/// When `key_pmem_value_pmem` is enabled the key is cloned into a
	/// Hybrid-allocated PMEM region so that it resides in persistent memory
	/// alongside the value.  This requires `K: Clone` but contains no unsafe
	/// code.
	#[cfg(feature = "key_pmem_value_pmem")]
	pub fn new(key: K, data: V, ttl: Option<u32>) -> Self
	where
		K: Clone,
	{
		use crate::Hybrid;

		let expiry = match ttl {
			Some(0) | None => None,
			Some(ttl) => Some(get_expiry_from_ttl(ttl)),
		};

		let _key_pmem = Box::new_in(key.clone(), Hybrid);

		Object {
			key,
			data: Arc::new(data),
			expiry,
			_key_pmem,
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
	/// When `key_pmem_value_pmem` is enabled the key is cloned into PMEM.
	/// Requires `K: Clone`; contains no unsafe code.
	#[cfg(feature = "key_pmem_value_pmem")]
	pub fn with_expiry(key: K, data: V, expiry: ExpireTime) -> Self
	where
		K: Clone,
	{
		use crate::Hybrid;

		let _key_pmem = Box::new_in(key.clone(), Hybrid);

		Object {
			key,
			data: Arc::new(data),
			expiry,
			_key_pmem,
		}
	}

	pub fn data(&self) -> Arc<V> {
		self.data.clone()
	}

	pub fn key(&self) -> &K {
		&self.key
	}

	pub fn key_matches(&self, key: &K) -> bool
	where
		K: Eq,
	{
		self.key.eq(key)
	}

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

// Specialized Default implementations for Object to support FlatMap initialization.
// These create dummy objects with empty buffers that will never be read
// (only used for FlatMap empty bucket initialization where hash = 0).

// For BufferDRAM (Box<[u8]>)
impl<K: Default> Default for Object<K, Box<[u8]>> {
	fn default() -> Self {
		#[cfg(feature = "key_pmem_value_pmem")]
		let _key_pmem = {
			use crate::Hybrid;
			Box::new_in(K::default(), Hybrid)
		};

		Object {
			key: K::default(),
			data: Arc::new(Vec::new().into_boxed_slice()),
			expiry: None,

			#[cfg(feature = "key_pmem_value_pmem")]
			_key_pmem,
		}
	}
}

// For BufferPMEM (Box<[u8], Hybrid>) need to mod...
#[cfg(any(feature = "key_value_pmem", feature = "global_flatmap_pmem"))]
impl<K: Default> Default for Object<K, Box<[u8], crate::Hybrid>> {
	fn default() -> Self {
		use crate::Hybrid;
		let vec: Vec<u8, Hybrid> = Vec::new_in(Hybrid);

		// For `key_pmem_value_pmem`, allocate a default key in PMEM for the
		// dummy sentinel object used to initialise empty FlatMap buckets.
		#[cfg(feature = "key_pmem_value_pmem")]
		let _key_pmem = Box::new_in(K::default(), Hybrid);

		Object {
			key: K::default(),
			data: Arc::new(vec.into_boxed_slice()),
			expiry: None,

			#[cfg(feature = "key_pmem_value_pmem")]
			_key_pmem,
		}
	}
}
