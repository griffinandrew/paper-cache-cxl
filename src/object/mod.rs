/*
 * Copyright (c) Kia Shakiba
 *
 * This source code is licensed under the GNU AGPLv3 license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! The cached object: a key, its value, and its expiry, in 24 bytes.
//!
//! ## v5 variant B: the value is one word, and its length lives here
//!
//! The value used to be a `Shared<TieredBuffer>` -- an 8-byte handle onto a
//! 32-byte refcounted allocation wrapping a 24-byte two-variant enum wrapping
//! a 16-byte fat `Box<[u8]>`. It is now a [`TieredValue`]: eight bytes,
//! pointing straight at the value's bytes, with the tier in bit 0.
//!
//! That moves the length out of the value and into this struct, which is
//! variant B and costs nothing: `Object` already had a four-byte hole after
//! `expiry` (verified with `-Zprint-type-sizes` on the base commit), and
//! `len: u32` lands in it. The struct stays 24 bytes, and `Option<Object>`
//! stays 24 bytes because the `NonNull` inside `TieredValue` is still the
//! niche -- which is what keeps the merged store's slot at 56.
//!
//! ## This type OWNS its value, and dropping it defers the free
//!
//! [`TieredValue`] is `Copy` and has no destructor: a copy of one is a BORROW
//! of the allocation, which is exactly what lets a reader lift the pointer out
//! from under a shard lock and copy the bytes with the lock released. The
//! owner is this struct, and there is exactly one owner per allocation.
//!
//! So `Object`'s `Drop` is where a value's life ends, and it ends by DEFERRAL
//! under a crossbeam-epoch pin rather than by an immediate free -- see
//! [`crate::value::defer_free`]. Putting it here rather than at each removal
//! site is what makes the trap in the brief ("every path that drops a value
//! ... THE TTL WORKER REAPS FROM ITS OWN THREAD AND MUST PIN TOO") unreachable
//! by construction: a set overwrite, an eviction, a TTL reap, a `wipe`, a
//! `MergedStore::take`/`retire`/`clear`, and dropping the cache itself all
//! destroy an `Object`, and all of them therefore pin and defer. There is no
//! site to forget.
//!
//! The one value that leaves without its `Object` being dropped is the one
//! [`Object::set_data`] hands back, which is the migration swap; that caller
//! defers it under the pin it already holds.

pub mod overhead;

use std::{marker::PhantomData, time::Instant};

use crate::{Tier, value::TieredValue};

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

	/// The value's bytes, and its tier, in one word. OWNED: see the module
	/// documentation on `Drop`.
	value: TieredValue,

	/// The value's length in bytes.
	///
	/// `TieredValue` is only the pointer half, so this is the half that makes
	/// `bytes()` and the deallocation layout well defined. It lives in the
	/// padding `expiry` used to leave behind, so it is free.
	len: u32,

	expiry: ExpireTime,

	/// Which cache SHAPE this object belongs to (`BufferDRAM`, `BufferPMEM`,
	/// or `TieredBuffer`). Zero-sized: since v5 every shape stores the same
	/// `TieredValue`, and `V` survives only to keep the shape-specific
	/// `PaperCache` impl blocks disjoint and to pick the admission tier at
	/// compile time -- see `crate::value::ValueShape`.
	///
	/// `fn() -> V` rather than `V` so this is unconditionally `Send + Sync`
	/// and covariant, whatever `V` is.
	_shape: PhantomData<fn() -> V>,
}

impl<K, V> Object<K, V> {
	/// Creates an object whose value is copied into the FAST tier.
	///
	/// The default tier rather than a required argument because that is what
	/// every all-DRAM shape and every test wants; the two callers that choose
	/// use [`Object::new_in`].
	pub fn new(key: K, bytes: &[u8], ttl: Option<u32>) -> Self {
		Self::new_in(key, bytes, Tier::Fast, ttl)
	}

	/// Creates an object whose value is copied into `tier`.
	pub fn new_in(key: K, bytes: &[u8], tier: Tier, ttl: Option<u32>) -> Self {
		let expiry = match ttl {
			Some(0) | None => None,
			Some(ttl) => Some(get_expiry_from_ttl(ttl)),
		};

		Self::with_expiry_in(key, bytes, tier, expiry)
	}

	/// Creates an object with an explicit expiry time, value in the fast tier.
	pub fn with_expiry(key: K, bytes: &[u8], expiry: ExpireTime) -> Self {
		Self::with_expiry_in(key, bytes, Tier::Fast, expiry)
	}

	/// Creates an object with an explicit expiry time and an explicit tier.
	///
	/// The one constructor: every other one funnels here, so there is exactly
	/// one place a value allocation is paired with the `len` that will later
	/// free it.
	pub fn with_expiry_in(key: K, bytes: &[u8], tier: Tier, expiry: ExpireTime) -> Self {
		// `TieredValue::new_in` refuses a length that does not fit a `u32`, so
		// this cast cannot truncate.
		let len = bytes.len() as u32;
		let value = TieredValue::new_in(bytes, tier);

		Object {
			#[cfg(not(feature = "key_pmem_value_pmem"))]
			key,
			#[cfg(feature = "key_pmem_value_pmem")]
			_key_pmem: Box::new_in(key, crate::Hybrid),

			value,
			len,
			expiry,
			_shape: PhantomData,
		}
	}

	/// This object's value: the pointer and its tier, nothing else.
	///
	/// A `&TieredValue` rather than a copy so the borrow checker keeps the
	/// handle tied to the object that owns it; call [`TieredValue::raw`] on it
	/// for the migration identity check.
	pub fn value(&self) -> &TieredValue {
		&self.value
	}

	/// The value's length in bytes.
	pub fn len(&self) -> u32 {
		self.len
	}

	/// The value's bytes.
	///
	/// Safe, unlike [`TieredValue::as_slice`], because this object owns the
	/// allocation and supplies the matching length itself. The returned slice
	/// borrows `self`, so a caller holding a shard guard keeps the guard for
	/// as long as it is copying -- which is why the read paths in `lib.rs`
	/// take the pointer and the length out FIRST and then drop the guard,
	/// rather than calling this across the copy.
	pub fn bytes(&self) -> &[u8] {
		// SAFETY: `self.len` is the length `self.value` was created with (both
		// are written together in `with_expiry_in` and replaced together in
		// `set_data`), and the value cannot have been freed while `self` is
		// alive -- `Drop` is the only thing that frees it.
		unsafe { self.value.as_slice(self.len) }
	}

	/// Lifts this object's value out from under the shard guard, tying it to
	/// an epoch pin instead.
	///
	/// This is the read path's whole trick, and it is SAFE. The returned
	/// [`ValueRef`] borrows `guard` rather than `self`, so the caller can drop
	/// the shard guard -- ending a critical section that would otherwise span
	/// a multi-kilobyte, possibly PMEM-backed copy -- and still read the bytes
	/// afterwards.
	///
	/// Sound because of what the two arguments prove between them. `&self`
	/// proves the value has not been retired: retirement is `Object::drop`,
	/// which cannot run while this borrow is live. `guard` proves that once it
	/// IS retired, the free is deferred behind a pin taken no later than this
	/// one, so it cannot run until the caller drops the guard.
	pub fn snapshot<'g>(&self, guard: &'g crossbeam_epoch::Guard) -> crate::value::ValueRef<'g> {
		// SAFETY: per the paragraph above -- `self.len` is this value's exact
		// length, and `self` being borrowed means the value is live now, so
		// `guard` covers it from here on.
		unsafe { crate::value::ValueRef::new(guard, self.value, self.len) }
	}

	/// Replaces this object's value in place, leaving `key` and `expiry`
	/// untouched, and RETURNS the old value and its length.
	///
	/// Used to physically migrate an object's bytes between tiers without
	/// disturbing its TTL or key.
	///
	/// The old value is handed back rather than freed here because freeing it
	/// is not this type's decision to make: a reader may be copying those
	/// bytes right now with no lock held, so the free has to be DEFERRED under
	/// the caller's epoch guard -- the same guard whose pin is what makes the
	/// caller's identity check immune to ABA. Returning it also makes the
	/// obligation impossible to overlook: the value is `#[must_use]`, so
	/// dropping it on the floor is a warning rather than a silent leak.
	#[must_use = "the old value must be freed -- defer it under an epoch guard"]
	pub fn set_data(&mut self, value: TieredValue, len: u32) -> (TieredValue, u32) {
		let old = (self.value, self.len);

		self.value = value;
		self.len = len;

		old
	}

	/// The value buffer's own byte cost.
	///
	/// Separated from `key_size` because the two are corrected differently:
	/// the key and expiry are already inside `shared_overhead`, which applies
	/// its own resident factor, while the value is scaled in `base_size`.
	pub fn data_size(&self) -> ObjectSize {
		self.len
	}

	/// The key's own byte cost, as `base_size` counts it.
	pub fn key_size(&self) -> ObjectSize
	where
		K: typesize::TypeSize,
	{
		use typesize::TypeSize;
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

/// Deep copy: a clone is a SEPARATE allocation in the same tier.
///
/// It has to be. The value is a bare pointer with no reference count, so a
/// shallow copy would hand two owners the same allocation and the second drop
/// would be a double free. Written by hand rather than derived for exactly
/// that reason -- `#[derive(Clone)]` on this struct would compile and would be
/// wrong.
///
/// `V` carries no bound: it is `PhantomData`.
impl<K: Clone, V> Clone for Object<K, V> {
	fn clone(&self) -> Self {
		Self::with_expiry_in(
			self.key().clone(),
			self.bytes(),
			self.value.tier(),
			self.expiry,
		)
	}
}

/// Ends the value's life, by deferral rather than by freeing.
///
/// See the module documentation: this is the single point every removal path
/// funnels through, which is why none of them has to remember to pin.
impl<K, V> Drop for Object<K, V> {
	fn drop(&mut self) {
		crate::value::defer_free(self.value, self.len);
	}
}

pub fn get_expiry_from_ttl(ttl: u32) -> std::num::NonZeroU32 {
	// `now_ticks()` is >= 1 and `saturating_add` cannot reach zero from it, so
	// the `NonZeroU32` is always valid.
	std::num::NonZeroU32::new(now_ticks().saturating_add(ttl))
		.expect("now_ticks() is never zero")
}

#[cfg(test)]
mod layout {
	use super::*;

	/// The whole point of variant B, asserted rather than assumed.
	///
	/// 24 bytes for `{ key: u64, value: TieredValue, len: u32, expiry }`, and
	/// `Option<Object>` the same 24 -- the `NonNull` inside `TieredValue` is
	/// the niche. The second assertion is the one the merged store's 56-byte
	/// slot depends on; lose the niche and every slot grows by 8.
	#[test]
	fn an_object_is_twenty_four_bytes_and_the_option_is_free() {
		type O = Object<u64, crate::value::BufferDRAM>;

		assert_eq!(
			core::mem::size_of::<O>(),
			24,
			"Object must be {{key 8, value 8, len 4, expiry 4}} with no padding",
		);

		assert_eq!(
			core::mem::size_of::<Option<O>>(),
			core::mem::size_of::<O>(),
			"Option<Object> must be niche-optimised into the value pointer",
		);
	}

	/// The length is free: it lives in the hole `expiry` left behind, so
	/// adding it did not grow the struct past the key + pointer + expiry it
	/// held before.
	#[test]
	fn the_length_costs_nothing() {
		type O = Object<u64, crate::value::BufferDRAM>;

		let without_len = core::mem::size_of::<u64>()
			+ core::mem::size_of::<crate::value::TieredValue>()
			+ core::mem::size_of::<ExpireTime>();

		assert_eq!(
			core::mem::size_of::<O>(),
			without_len.next_multiple_of(8),
			"len: u32 must fit the alignment padding, not add to it",
		);
	}

	/// A round trip through the accessors the read paths use.
	#[test]
	fn an_object_round_trips_its_bytes_and_length() {
		let object = Object::<u64, crate::value::BufferDRAM>::new(7, b"hello world", None);

		assert_eq!(object.len(), 11);
		assert_eq!(object.bytes(), b"hello world");
		assert_eq!(object.data_size(), 11);
		assert!(object.value().is_fast());
	}

	/// A clone must be a DIFFERENT allocation, or the second drop double-frees.
	#[test]
	fn a_clone_is_a_separate_allocation() {
		let a = Object::<u64, crate::value::BufferDRAM>::new(1, b"abcd", None);
		let b = a.clone();

		assert_eq!(a.bytes(), b.bytes(), "the bytes must match");
		assert_ne!(
			a.value().raw(),
			b.value().raw(),
			"but they must not be the same allocation",
		);
		assert_eq!(a.value().tier(), b.value().tier(), "and the tier must survive");
	}

	/// `set_data` must hand the old value back rather than leaking or freeing
	/// it, and must leave the key and expiry alone.
	#[test]
	fn set_data_returns_the_old_value_and_keeps_the_rest() {
		let mut object = Object::<u64, crate::value::BufferDRAM>::new(3, b"old", Some(60));
		let expiry = object.expiry();
		let old_raw = object.value().raw();

		let replacement = TieredValue::new_slow(b"newer");
		let (old, old_len) = object.set_data(replacement, 5);

		assert_eq!(old.raw(), old_raw, "the handle handed back must be the old one");
		assert_eq!(old_len, 3);
		assert_eq!(object.bytes(), b"newer");
		assert_eq!(object.len(), 5);
		assert!(object.value().is_slow(), "the new tier must stick");
		assert_eq!(*object.key(), 3, "the key must not move");
		assert_eq!(object.expiry(), expiry, "nor the expiry");

		// This test owns the old value now, exactly as `apply_migration` does.
		crate::value::defer_free(old, old_len);
	}

	/// Dropping an object must RETIRE its value, not free it and not leak it.
	///
	/// Asserted through the global counter with `>=` rather than `==`: the
	/// counter is process-wide and every other test in the suite retires values
	/// too. What is deterministic is that our own drop contributes at least
	/// one, and that is exactly the claim -- a leak would contribute none.
	#[test]
	fn dropping_an_object_retires_its_value() {
		use std::sync::atomic::Ordering::Relaxed;

		let before = crate::value::VALUE_FREES_DEFERRED.load(Relaxed);

		drop(Object::<u64, crate::value::BufferDRAM>::new(11, b"retire me", None));

		assert!(
			crate::value::VALUE_FREES_DEFERRED.load(Relaxed) > before,
			"Object::drop must hand the value to the epoch; a value dropped \
			 without one of these is leaked outright",
		);
	}

	/// Deferral is not enough on its own: the garbage has to actually come
	/// back. This drops objects and flushes until the RUN counter moves,
	/// which is the end-to-end claim that epoch reclamation is wired up
	/// rather than merely invoked.
	///
	/// The bound is enormous relative to what it needs (crossbeam's local bag
	/// holds a few dozen deferrals and an epoch advances in three steps), so a
	/// slow collector still passes and only a broken one fails. It cannot hang.
	#[test]
	fn retired_values_are_eventually_reclaimed() {
		use std::sync::atomic::Ordering::Relaxed;

		let before = crate::value::VALUE_FREES_RUN.load(Relaxed);

		for _ in 0..200 {
			for key in 0..100u64 {
				drop(Object::<u64, crate::value::BufferDRAM>::new(
					key,
					&[0u8; 64],
					None,
				));
			}

			crate::value::flush();

			if crate::value::VALUE_FREES_RUN.load(Relaxed) > before {
				return;
			}
		}

		panic!(
			"20,000 retired values and 200 flushes ran none of them -- deferred \
			 frees are accumulating without ever being collected",
		);
	}

	/// A zero-length value is still a real, unique, freeable allocation --
	/// `TieredValue` rounds the layout up to one byte for precisely this.
	#[test]
	fn an_empty_value_is_still_an_object() {
		let object = Object::<u64, crate::value::BufferDRAM>::new(9, b"", None);

		assert_eq!(object.len(), 0);
		assert!(object.bytes().is_empty());
		assert!(!object.value().raw().is_null());
	}
}
