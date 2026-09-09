/*
 * Copyright (c) Kia Shakiba
 *
 * This source code is licensed under the GNU AGPLv3 license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! The cached object: an eight-byte handle, and nothing else.
//!
//! ## Everything moved into the value header
//!
//! This struct used to be `{ key, value, len, expiry }` -- 24 bytes sitting
//! inside the object map's row, so the row was 8 (hashed key) + 24 + a control
//! byte. All four now live in the refcounted DRAM header
//! [`crate::value::ValueHeader`], and what is left here is the handle onto it:
//!
//! ```text
//!   Object                      8 B   (a TieredValue, i.e. one Arc pointer)
//!   map row  = 8 + 8 + 1 = 17 B       (was 8 + 24 + 1 = 33)
//! ```
//!
//! The row is the most expensive place in this design to keep a byte -- it is
//! measured at ~80 B/object of allocation for a 33-byte row, because 32
//! independently-doubling `DashMap` shard tables amortise badly -- so moving
//! 16 bytes out of it and into an allocation that is charged once is the point
//! of the change.
//!
//! `V` survives as a zero-sized `PhantomData` marker, exactly as before: it
//! keeps the shape-specific `PaperCache` impl blocks disjoint and picks the
//! admission tier at compile time. See [`crate::value::ValueShape`].
//!
//! ## There is no `Drop` here any more
//!
//! There used to be, and it was load-bearing: the value was a bare pointer, so
//! this struct was the single point every removal path funnelled through, and
//! it deferred the free under a `crossbeam_epoch` pin.
//!
//! The refcount replaces all of that. Dropping an `Object` drops its
//! `TieredValue`, which decrements; the last handle to go frees the header and
//! the bytes it owns. A set overwrite, an eviction, a TTL reap, a `wipe`,
//! `MergedStore::take`, dropping the cache -- all of them still funnel through
//! one place, it is just `Arc`'s place now rather than ours. In particular a
//! reader that lifted the value out from under the shard lock holds a strong
//! reference, so a writer that unpublishes mid-copy cannot free underneath it.
//!
//! ## `Clone` is shallow now, and that is the correct meaning
//!
//! It had to be a deep copy while the value was an unowned pointer, because a
//! shallow copy would have been a double free. With a refcount, two handles
//! onto one value ARE the same object -- the bytes are immutable, so they can
//! never disagree about its contents -- and copying them would be a silent
//! waste of a whole value allocation.

pub mod overhead;

use std::{marker::PhantomData, time::Instant};

use crate::{Tier, value::TieredValue};

pub type ObjectSize = u32;

/// Expiry as a tick count, where one tick is one second since a process-global
/// base instant, plus one.
///
/// Was `Option<Instant>` -- **16 bytes**, carrying nanosecond precision that a
/// cache TTL has no use for: the smallest TTL the API accepts is one second,
/// and no trace in the corpus sets a TTL at all.
///
/// The `+1` is what buys the 4 bytes: `Option<NonZeroU32>` uses zero as its
/// `None`, so a stored tick must never be zero -- and an object set during the
/// first second of process life otherwise would be. That same encoding is what
/// lets the header store it as a plain `AtomicU32` with `0` meaning `None`.
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
	/// The whole object: key, bytes, length, tier and expiry, behind one
	/// refcounted DRAM header. See the module documentation.
	value: TieredValue<K>,

	/// Which cache SHAPE this object belongs to (`BufferDRAM`, `BufferPMEM`,
	/// or `TieredBuffer`). Zero-sized: every shape stores the same
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
	/// every all-DRAM shape and every test wants; the callers that choose use
	/// [`Object::new_in`].
	pub fn new(key: K, bytes: &[u8], ttl: Option<u32>) -> Self {
		Self::new_in(key, bytes, Tier::Fast, ttl)
	}

	/// Creates an object whose value is copied into `tier`.
	pub fn new_in(key: K, bytes: &[u8], tier: Tier, ttl: Option<u32>) -> Self {
		Self::with_expiry_in(key, bytes, tier, expiry_from_ttl(ttl))
	}

	/// Creates an object with an explicit expiry time, value in the fast tier.
	pub fn with_expiry(key: K, bytes: &[u8], expiry: ExpireTime) -> Self {
		Self::with_expiry_in(key, bytes, Tier::Fast, expiry)
	}

	/// Creates an object with an explicit expiry time and an explicit tier.
	///
	/// The one constructor: every other one funnels here.
	pub fn with_expiry_in(key: K, bytes: &[u8], tier: Tier, expiry: ExpireTime) -> Self {
		Object {
			value: TieredValue::new_in(key, bytes, tier, expiry),
			_shape: PhantomData,
		}
	}

	/// Wraps an existing value handle, sharing its header.
	///
	/// Used by the migration path, which builds the replacement header outside
	/// the shard guard and installs it inside.
	pub fn from_value(value: TieredValue<K>) -> Self {
		Object { value, _shape: PhantomData }
	}

	/// This object's value handle.
	pub fn value(&self) -> &TieredValue<K> {
		&self.value
	}

	/// The value's length in bytes.
	pub fn len(&self) -> u32 {
		self.value.len()
	}

	pub fn is_empty(&self) -> bool {
		self.value.is_empty()
	}

	/// The value's bytes.
	///
	/// The returned slice borrows `self`. A caller that wants to read the
	/// bytes with the shard guard RELEASED must take a [`Object::snapshot`]
	/// instead, which is an owned handle rather than a borrow.
	pub fn bytes(&self) -> &[u8] {
		self.value.bytes()
	}

	/// Lifts this object's value out from under the shard guard.
	///
	/// This is the read path's whole trick, and with a refcount it is a
	/// one-liner: the returned handle owns a strong reference, so the caller
	/// can drop the shard guard -- ending a critical section that would
	/// otherwise span a multi-kilobyte, possibly CXL-backed copy -- and still
	/// read the bytes afterwards. Whoever replaces this object in the map
	/// meanwhile decrements a count that is not yet zero, and frees nothing.
	///
	/// It used to take a `crossbeam_epoch::Guard` and return a `ValueRef`
	/// borrowing the pin. The guarantee is the same; the mechanism is one
	/// atomic increment instead of a pin, and it cannot be forgotten because
	/// the handle IS the proof.
	pub fn snapshot(&self) -> TieredValue<K> {
		self.value.clone()
	}

	/// Replaces this object's value, and RETURNS the old handle.
	///
	/// Used to physically migrate an object's bytes between tiers. The
	/// replacement carries the key and expiry across -- see
	/// [`TieredValue::migrated_to`] -- so this is not "keep the key and swap
	/// the bytes" as it was when those fields lived in this struct; it is a
	/// whole new header, installed atomically from the map's point of view.
	///
	/// The old handle is returned rather than dropped here so the caller can
	/// count the migration and, if it wants, observe whether it held the last
	/// reference. Dropping it is now the correct and complete way to retire
	/// it, so this is no longer `#[must_use]`: there is no obligation left to
	/// forget.
	pub fn set_data(&mut self, value: TieredValue<K>) -> TieredValue<K> {
		std::mem::replace(&mut self.value, value)
	}

	/// The value buffer's own byte cost.
	///
	/// Separated from `key_size` because the two are corrected differently:
	/// the key and expiry are already inside `shared_overhead`, which applies
	/// its own resident factor, while the value is scaled in `base_size`.
	pub fn data_size(&self) -> ObjectSize {
		self.value.len()
	}

	/// The key's own byte cost, as `base_size` counts it.
	pub fn key_size(&self) -> ObjectSize
	where
		K: typesize::TypeSize,
	{
		use typesize::TypeSize;
		self.key().get_size() as ObjectSize
	}

	pub fn key(&self) -> &K {
		self.value.key()
	}

	/// Whether this object's key matches the given key.
	///
	/// The object map is keyed on a 64-bit hash, so this is what makes a hash
	/// collision harmless. The key lives in the DRAM header whatever tier the
	/// bytes are in, so this never crosses the interconnect.
	pub fn key_matches(&self, key: &K) -> bool
	where
		K: Eq,
	{
		self.value.key_matches(key)
	}

	pub fn expiry(&self) -> ExpireTime {
		self.value.expiry()
	}

	pub fn is_expired(&self) -> bool {
		self.expiry().is_some_and(|expiry| expiry.get() <= now_ticks())
	}

	/// Sets or clears this object's TTL.
	///
	/// `&mut self` for source compatibility with every caller, though the
	/// header's expiry is atomic and does not require exclusivity. Visible to
	/// every handle onto the same header, which is correct: they are the same
	/// object.
	pub fn expires(&mut self, ttl: Option<u32>) {
		self.value.set_expiry(expiry_from_ttl(ttl));
	}
}

/// Shallow: a clone SHARES the value allocation.
///
/// See the module documentation. This was a deep copy while the value was a
/// bare pointer with no count; with a refcount, sharing is both correct and
/// what the callers actually mean.
impl<K, V> Clone for Object<K, V> {
	fn clone(&self) -> Self {
		Object {
			value: self.value.clone(),
			_shape: PhantomData,
		}
	}
}

/// Turns a TTL in seconds into an expiry tick.
fn expiry_from_ttl(ttl: Option<u32>) -> ExpireTime {
	match ttl {
		Some(0) | None => None,
		Some(ttl) => Some(get_expiry_from_ttl(ttl)),
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

	/// The point of moving the key and the expiry into the header, asserted
	/// rather than assumed.
	///
	/// 8 bytes -- one `Arc` pointer -- and `Option<Object>` the same 8, because
	/// the `NonNull` inside the `Arc` is the niche. The second assertion is
	/// what keeps the merged store's slot small; lose the niche and every slot
	/// grows by 8.
	#[test]
	fn an_object_is_one_pointer_and_the_option_is_free() {
		type O = Object<u64, crate::value::BufferDRAM>;

		assert_eq!(
			core::mem::size_of::<O>(),
			core::mem::size_of::<usize>(),
			"Object must be exactly the value handle; everything else lives in \
			 the header",
		);

		assert_eq!(
			core::mem::size_of::<Option<O>>(),
			core::mem::size_of::<O>(),
			"Option<Object> must be niche-optimised into the value pointer",
		);
	}

	/// A round trip through the accessors the read paths use.
	#[test]
	fn an_object_round_trips_its_bytes_and_length() {
		let object = Object::<u64, crate::value::BufferDRAM>::new(7, b"hello world", None);

		assert_eq!(*object.key(), 7);
		assert_eq!(object.len(), 11);
		assert_eq!(object.bytes(), b"hello world");
		assert_eq!(object.data_size(), 11);
		assert!(object.value().is_fast());
	}

	/// A clone SHARES the allocation -- the opposite of what this test
	/// asserted before the refcount, and the reason `Clone` could stop being
	/// hand-written as a deep copy.
	#[test]
	fn a_clone_shares_the_allocation() {
		let object = Object::<u64, crate::value::BufferDRAM>::new(1, b"payload", None);
		let copy = object.clone();

		assert!(
			TieredValue::ptr_eq(object.value(), copy.value()),
			"a clone must share the header, not duplicate it",
		);
		assert_eq!(object.value().strong_count(), 2);
		assert_eq!(copy.bytes(), b"payload");

		drop(copy);
		assert_eq!(object.value().strong_count(), 1);
	}

	/// A snapshot outlives the object it came from, which is exactly what the
	/// read path relies on when it drops the shard guard before copying.
	#[test]
	fn a_snapshot_keeps_the_value_alive_past_its_object() {
		let object = Object::<u64, crate::value::BufferDRAM>::new(2, b"still here", None);
		let snapshot = object.snapshot();

		drop(object);

		assert_eq!(snapshot.bytes(), b"still here");
		assert_eq!(snapshot.strong_count(), 1);
	}

	/// `set_data` swaps the header and hands the old one back; the key and the
	/// expiry travel with the new header rather than staying behind.
	#[test]
	fn set_data_returns_the_old_value() {
		let mut object = Object::<u64, crate::value::BufferDRAM>::new(3, b"old", None);
		let before = object.snapshot();

		let replacement = object.value().migrated_to(Tier::Slow);
		let old = object.set_data(replacement);

		assert!(TieredValue::ptr_eq(&old, &before), "the old handle comes back");
		assert_eq!(old.bytes(), b"old");
		assert_eq!(object.bytes(), b"old", "a migration copies the bytes across");
		assert_eq!(*object.key(), 3, "and carries the key");
		assert!(object.value().is_slow());
	}

	/// A TTL set on one handle is visible through another, because they are
	/// the same object.
	#[test]
	fn an_expiry_is_shared_between_handles() {
		let mut object = Object::<u64, crate::value::BufferDRAM>::new(4, b"x", None);
		let snapshot = object.snapshot();

		assert!(object.expiry().is_none());
		object.expires(Some(60));

		assert!(snapshot.expiry().is_some());
		assert_eq!(object.expiry(), snapshot.expiry());
	}

	/// Dropping the last handle frees the bytes. Counted, so the assertion is
	/// about the allocator rather than about the absence of a crash.
	#[test]
	fn dropping_the_last_handle_frees_the_value() {
		// Asserts on THIS value's own strong count, not on the process-global
		// `VALUE_FREES`. That counter is bumped by every test that drops a
		// value, and cargo runs tests in parallel, so reading it as a delta
		// raced everything else in the binary -- observed failing four runs in
		// eight, reporting +2 where it required 0. Serialising the tests that
		// read it does not help either, because the ones that MOVE it are all
		// the others.
		let object = Object::<u64, crate::value::BufferDRAM>::new(5, b"transient", None);
		let snapshot = object.snapshot();

		assert_eq!(
			snapshot.strong_count(),
			2,
			"the object and the snapshot are both handles onto one allocation",
		);

		drop(object);

		assert_eq!(
			snapshot.strong_count(),
			1,
			"the snapshot still holds a reference, so nothing may be freed yet",
		);

		// Reading through the surviving handle is the proof the allocation is
		// still live: against a premature free this is a use-after-free, which
		// is what Miri and the sanitisers are pointed at.
		assert_eq!(snapshot.bytes(), b"transient");

		// Dropping the last handle takes the count to zero, and reaching zero
		// IS the free -- there is no deferral left to get wrong, which is the
		// property this test was written to pin.
		drop(snapshot);
	}

	/// A zero-length value is still a distinct, addressable object.
	#[test]
	fn an_empty_value_is_still_an_object() {
		let object = Object::<u64, crate::value::BufferDRAM>::new(6, b"", None);

		assert_eq!(object.len(), 0);
		assert!(object.is_empty());
		assert_eq!(object.bytes(), b"");
		assert_eq!(*object.key(), 6);
	}
}
