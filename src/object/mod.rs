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

#[cfg(feature = "allocator_api")]
use crate::allocator::HybridObjects as Hybrid;

#[cfg(feature = "allocator_api")]
pub struct Object<K, V> {
	key: Box<K, Hybrid>,
	data: Arc<V>,

	expiry: ExpireTime,
}

#[cfg(not(feature = "allocator_api"))]
pub struct Object<K, V> {
	key: K,
	data: Arc<V>,

	expiry: ExpireTime,
}

#[cfg(feature = "allocator_api")]
impl<K, V> Object<K, V> {
	pub fn new(key: K, data: V, ttl: Option<u32>) -> Self {
		let expiry = match ttl {
			Some(0) | None => None,
			Some(ttl) => Some(get_expiry_from_ttl(ttl)),
		};

		Object {
			key: Box::new_in(key, Hybrid),
			data: Arc::new(data),

			expiry,
		}
	}

	pub fn data(&self) -> Arc<V> {
		self.data.clone()
	}

	pub fn key_matches(&self, key: &K) -> bool
	where
		K: Eq,
	{
		self.key.as_ref().eq(key)
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

	// Helper method to get a pointer to the key for tier checking
	pub fn key_ptr(&self) -> *const K {
		&**self.key as *const K
	}
}

#[cfg(not(feature = "allocator_api"))]
impl<K, V> Object<K, V> {
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

	pub fn data(&self) -> Arc<V> {
		self.data.clone()
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
