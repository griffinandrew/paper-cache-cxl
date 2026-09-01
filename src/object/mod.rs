/*
 * Copyright (c) Kia Shakiba
 *
 * This source code is licensed under the GNU AGPLv3 license found in the
 * LICENSE file in the root directory of this source tree.
 */

pub mod overhead;

use std::{
	mem,
	time::{Instant, Duration},
};

use typesize::TypeSize;

pub type ObjectSize = u32;
/// Expiry as a tick count, where one tick is one second since a process-global
/// base instant, plus one.
///
/// Was `Option<Instant>` -- **16 bytes**, a quarter of the merged store's
/// 64-byte slot, carrying nanosecond precision that a cache TTL has no use for:
/// the smallest TTL the API accepts is one second, and no trace in the corpus
/// sets a TTL at all.
///
/// The `+1` is what buys the 4 bytes: `Option<NonZeroU32>` uses zero as its
/// `None`, so a stored tick must never be zero -- and an object set during the
/// first second of process life otherwise would be.
///
/// u32 seconds is 136 years of uptime, against a base taken once per process.
pub type ExpireTime = Option<std::num::NonZeroU32>;

/// The instant tick 1 corresponds to. Taken once, on first use.
fn tick_base() -> Instant {
	static BASE: std::sync::OnceLock<Instant> = std::sync::OnceLock::new();

	*BASE.get_or_init(Instant::now)
}

/// The current tick. Never zero, so it is always a valid `NonZeroU32`.
pub fn now_ticks() -> u32 {
	// Saturating rather than wrapping: at 136 years of uptime this pins every
	// object as already expired, which is wrong but bounded. Wrapping would
	// make old objects look freshly set.
	tick_base().elapsed().as_secs().min(u32::MAX as u64 - 1) as u32 + 1
}

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

	data: crate::shared::Shared<V>,
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
			data: crate::shared::Shared::new(data),
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
			data: crate::shared::Shared::new(data),
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
			data: crate::shared::Shared::new(data),
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
			data: crate::shared::Shared::new(data),
			expiry,
		}
	}

	pub fn data(&self) -> crate::shared::Shared<V> {
		self.data.clone()
	}

	/// Replaces this object's data in place, leaving `key` and `expiry`
	/// untouched.
	///
	/// Used by `lru_hybrid_cache` to physically migrate an object's bytes
	/// between tiers (e.g. `TieredBuffer::Fast` <-> `TieredBuffer::Slow`)
	/// without disturbing its TTL or key.
	pub fn set_data(&mut self, data: V) {
		self.data = crate::shared::Shared::new(data);
	}

	/// Return a reference to the key.
	///
	/// Without `key_pmem_value_pmem` the key lives in DRAM.
	/// With `key_pmem_value_pmem` the key lives in PMEM and is accessed via
	/// the `_key_pmem` box; no DRAM copy exists.
	/// The value buffer's own byte cost.
	///
	/// Separated from `key_size` because the two are corrected differently:
	/// the key and expiry are already inside `shared_overhead`, which applies
	/// its own resident factor, while the value is scaled in `base_size`.
	pub fn data_size(&self) -> ObjectSize
	where
		V: TypeSize,
	{
		self.data.get_size() as ObjectSize
	}

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
		self.expiry.is_some_and(|expiry| expiry.get() <= now_ticks())
	}

	pub fn expires(&mut self, ttl: Option<u32>) {
		self.expiry = match ttl {
			Some(0) | None => None,
			Some(ttl) => Some(get_expiry_from_ttl(ttl)),
		};
	}
}

pub fn get_expiry_from_ttl(ttl: u32) -> std::num::NonZeroU32 {
	// `now_ticks()` is >= 1 and `saturating_add` cannot reach zero from it, so
	// the `NonZeroU32` is always valid.
	std::num::NonZeroU32::new(now_ticks().saturating_add(ttl))
		.expect("now_ticks() is never zero")
}

