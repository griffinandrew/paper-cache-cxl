/*
 * Copyright (c) Kia Shakiba
 *
 * This source code is licensed under the GNU AGPLv3 license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! A strong-count-only reference-counted pointer, for the value buffer.
//!
//! `Arc<T>` allocates an `ArcInner` carrying BOTH a strong and a weak count.
//! Measured on this crate's `TieredBuffer` (see `tiered_buffer::layout`):
//!
//! ```text
//!   strong           8
//!   weak             8   <- nothing in this tree ever creates a Weak
//!   TieredBuffer    24
//!                  ---
//!                   40   -> jemalloc rounds up to its 48-byte size class
//! ```
//!
//! Dropping the weak count takes the allocation to 32 bytes -- one size class
//! down -- and saves 16 B on EVERY cached object. At the measured 143 B of DRAM
//! metadata per object that is a 11% cut, and it is enough to put a 100-byte-
//! object trace like cluster19 under the ~135 B/object at which its fast tier
//! still has usable capacity after metadata.
//!
//! What is kept is exactly what the cache depends on:
//!
//!   * cheap `clone`, so a reader can take a handle and RELEASE the shard lock
//!     before touching the bytes. That is what lets `apply_migration` build a
//!     destination buffer with no lock held, which is the whole reason tier
//!     copies do not serialise against readers.
//!   * `ptr_eq`, the identity check that makes a migration safe to apply: a
//!     value replaced by a `set` mid-copy is rejected rather than overwritten.
//!     The strong reference held across the copy also makes that check immune
//!     to ABA, since the allocation cannot be freed and its address cannot be
//!     recycled while we hold it.

use std::{
	marker::PhantomData,
	ops::Deref,
	ptr::NonNull,
	sync::atomic::{fence, AtomicUsize, Ordering},
};

#[repr(C)]
struct Inner<T> {
	strong: AtomicUsize,
	data: T,
}

pub struct Shared<T> {
	ptr: NonNull<Inner<T>>,

	/// Marks this as OWNING an `Inner<T>`, so drop-check and variance behave
	/// the way `Arc` does rather than the way a bare pointer would.
	_owns: PhantomData<Inner<T>>,
}

// Same bounds `Arc` carries, and for the same reasons: sharing a handle shares
// `&T` across threads, and the last handle may run `T`'s destructor on whatever
// thread happens to hold it.
unsafe impl<T: Send + Sync> Send for Shared<T> {}
unsafe impl<T: Send + Sync> Sync for Shared<T> {}

impl<T> Shared<T> {
	pub fn new(data: T) -> Self {
		let boxed = Box::new(Inner { strong: AtomicUsize::new(1), data });

		Shared {
			// `Box::into_raw` never returns null.
			ptr: unsafe { NonNull::new_unchecked(Box::into_raw(boxed)) },
			_owns: PhantomData,
		}
	}

	/// Whether two handles name the SAME allocation.
	pub fn ptr_eq(a: &Self, b: &Self) -> bool {
		std::ptr::eq(a.ptr.as_ptr(), b.ptr.as_ptr())
	}

	pub fn strong_count(this: &Self) -> usize {
		unsafe { (*this.ptr.as_ptr()).strong.load(Ordering::Acquire) }
	}
}

impl<T> Clone for Shared<T> {
	fn clone(&self) -> Self {
		// `Relaxed` is sufficient. We already hold a handle, so the count
		// cannot reach zero underneath us, and that existing handle is what
		// orders this thread against the allocation.
		unsafe { (*self.ptr.as_ptr()).strong.fetch_add(1, Ordering::Relaxed) };

		Shared { ptr: self.ptr, _owns: PhantomData }
	}
}

impl<T> Drop for Shared<T> {
	fn drop(&mut self) {
		if unsafe { (*self.ptr.as_ptr()).strong.fetch_sub(1, Ordering::Release) } != 1 {
			return;
		}

		// The `Release` above publishes this thread's writes to the value. The
		// `Acquire` fence makes every OTHER thread's writes visible before the
		// destructor runs, so the last handle can safely drop what all of them
		// touched. Same ordering pair `std`'s `Arc` uses, for the same reason.
		fence(Ordering::Acquire);

		drop(unsafe { Box::from_raw(self.ptr.as_ptr()) });
	}
}

impl<T> Deref for Shared<T> {
	type Target = T;

	fn deref(&self) -> &T {
		unsafe { &(*self.ptr.as_ptr()).data }
	}
}

impl<T> AsRef<T> for Shared<T> {
	fn as_ref(&self) -> &T {
		self
	}
}

impl<T: std::fmt::Debug> std::fmt::Debug for Shared<T> {
	fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
		std::fmt::Debug::fmt(&**self, f)
	}
}

#[cfg(test)]
mod tests {
	use super::*;
	use std::sync::atomic::AtomicU32;

	/// The reason this type exists, asserted rather than assumed.
	#[test]
	fn the_allocation_is_one_word_smaller_than_arcs() {
		assert_eq!(
			core::mem::size_of::<Inner<[u8; 24]>>(),
			32,
			"a 24-byte payload must allocate 32 bytes, not the 40 an ArcInner \
			 needs -- that is the whole saving",
		);
		assert_eq!(core::mem::size_of::<Shared<[u8; 24]>>(), 8, "the handle is a pointer");
	}

	#[test]
	fn clones_share_and_the_last_drop_frees() {
		static DROPS: AtomicU32 = AtomicU32::new(0);
		struct Tracked;
		impl Drop for Tracked {
			fn drop(&mut self) {
				DROPS.fetch_add(1, Ordering::Relaxed);
			}
		}

		let a = Shared::new(Tracked);
		let b = a.clone();
		let c = b.clone();

		assert_eq!(Shared::strong_count(&a), 3);
		assert!(Shared::ptr_eq(&a, &c), "clones must name the same allocation");

		drop(a);
		drop(b);
		assert_eq!(DROPS.load(Ordering::Relaxed), 0, "not the last handle yet");

		drop(c);
		assert_eq!(DROPS.load(Ordering::Relaxed), 1, "the last handle must drop the value");
	}

	/// `ptr_eq` is the migration safety check, so distinct allocations holding
	/// EQUAL values must still compare unequal.
	#[test]
	fn ptr_eq_is_identity_not_equality() {
		let a = Shared::new(vec![1u8, 2, 3]);
		let b = Shared::new(vec![1u8, 2, 3]);

		assert_eq!(*a, *b, "the values are equal");
		assert!(!Shared::ptr_eq(&a, &b), "but they are not the same allocation");
	}

	#[test]
	fn handles_survive_being_moved_between_threads() {
		let a = Shared::new(vec![7u8; 64]);
		let b = a.clone();

		let t = std::thread::spawn(move || {
			assert_eq!(b.len(), 64);
			drop(b);
		});

		t.join().unwrap();
		assert_eq!(Shared::strong_count(&a), 1);
		assert_eq!(a.len(), 64);
	}
}
