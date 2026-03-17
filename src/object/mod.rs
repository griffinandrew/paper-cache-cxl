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

	/// When `key_pmem_value_pmem` is enabled, the key's raw bytes are also allocated
	/// in persistent memory via the Hybrid allocator so that both key data and
	/// value data each reside in their own `BufferPMEM`.
	#[cfg(feature = "key_pmem_value_pmem")]
	key_buf: Box<[u8], crate::Hybrid>,
}

impl<K, V> Object<K, V> {
	pub fn new(key: K, data: V, ttl: Option<u32>) -> Self {
		let expiry = match ttl {
			Some(0) | None => None,
			Some(ttl) => Some(get_expiry_from_ttl(ttl)),
		};

		// Allocate the key's raw bytes into PMEM before `key` is moved into
		// the struct.
		#[cfg(feature = "key_pmem_value_pmem")]
		let key_buf = Self::alloc_key_buf(&key);

		Object {
			key,
			data: Arc::new(data),

			expiry,

			#[cfg(feature = "key_pmem_value_pmem")]
			key_buf,
		}
	}

	/// Create a new Object with an explicit expiry time
	pub fn with_expiry(key: K, data: V, expiry: ExpireTime) -> Self {
		// Allocate the key's raw bytes into PMEM before `key` is moved into
		// the struct.
		#[cfg(feature = "key_pmem_value_pmem")]
		let key_buf = Self::alloc_key_buf(&key);

		Object {
			key,
			data: Arc::new(data),
			expiry,

			#[cfg(feature = "key_pmem_value_pmem")]
			key_buf,
		}
	}

	/// Allocates the raw bytes of `key` into persistent memory via the Hybrid
	/// allocator, returning a `Box<[u8], Hybrid>` that keeps the key data
	/// in PMEM alongside the value `BufferPMEM`.
	///
	/// # Safety note
	/// The byte slice is obtained by reinterpreting the memory of `key` as
	/// `[u8]`.  This is sound for any fully-initialised `K: Sized` because we
	/// only copy the bit pattern.  Types that contain padding bytes will have
	/// those padding bytes (which may be uninitialized) included in the PMEM
	/// copy; for typical key types such as `u32` there is no padding.
	#[cfg(feature = "key_pmem_value_pmem")]
	fn alloc_key_buf(key: &K) -> Box<[u8], crate::Hybrid> {
		use crate::Hybrid;
		// SAFETY: `key` is a reference to a fully-initialised value of type
		// `K`.  Reinterpreting its bytes as `[u8]` is sound – we are merely
		// copying the bit pattern, not constructing any value that could be
		// invalid.
		let key_bytes = unsafe {
			std::slice::from_raw_parts(
				key as *const K as *const u8,
				std::mem::size_of::<K>(),
			)
		};
		key_bytes.to_vec_in(Hybrid).into_boxed_slice()
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
		let key_buf = {
			use crate::Hybrid;
			Vec::<u8, Hybrid>::new_in(Hybrid).into_boxed_slice()
		};

		Object {
			key: K::default(),
			data: Arc::new(Vec::new().into_boxed_slice()),
			expiry: None,

			#[cfg(feature = "key_pmem_value_pmem")]
			key_buf,
		}
	}
}

// For BufferPMEM (Box<[u8], Hybrid>) need to mod...
#[cfg(any(feature = "key_value_pmem", feature = "global_flatmap_pmem"))]
impl<K: Default> Default for Object<K, Box<[u8], crate::Hybrid>> {
	fn default() -> Self {
		use crate::Hybrid;
		let vec: Vec<u8, Hybrid> = Vec::new_in(Hybrid);

		// For `key_pmem_value_pmem`, create an empty key_buf in PMEM for the dummy
		// sentinel object used to initialise empty FlatMap buckets.
		#[cfg(feature = "key_pmem_value_pmem")]
		let key_buf = Vec::<u8, Hybrid>::new_in(Hybrid).into_boxed_slice();

		Object {
			key: K::default(),
			data: Arc::new(vec.into_boxed_slice()),
			expiry: None,

			#[cfg(feature = "key_pmem_value_pmem")]
			key_buf,
		}
	}
}
