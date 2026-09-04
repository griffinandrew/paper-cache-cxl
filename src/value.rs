/*
 * Copyright (c) Kia Shakiba
 *
 * This source code is licensed under the GNU AGPLv3 license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! `TieredValue` -- an entire cached value in ONE eight-byte word.
//!
//! This is the v5 replacement for `Shared<TieredBuffer>`, and it is the ONLY
//! module in the crate that is allowed to contain unsafe value code. Every
//! allocation, every free, and every reinterpretation of a value pointer lives
//! here; nothing outside this file may construct a value pointer, and nothing
//! outside this file may free one.
//!
//! ## What the eight bytes replace
//!
//! ```text
//!   Shared<TieredBuffer>  handle          8   -> the pointer we keep
//!   Shared inner: strong count            8   -> gone (epoch reclamation)
//!   TieredBuffer enum discriminant        8   -> gone (the low tag bit)
//!   Box<[u8]> fat-pointer length          8   -> gone (`len: u32` in Object)
//!   ...plus the allocation holding them  32   -> gone entirely
//! ```
//!
//! `TieredBuffer` is a two-variant enum wrapping a fat `Box<[u8]>`, so it is
//! 24 bytes with an EIGHT-byte discriminant, and `Shared` boxes it behind a
//! 32-byte refcounted `Inner`. A `TieredValue` is the value's bytes and
//! nothing else: one `NonNull<u8>` to the bytes, with the tier in bit 0.
//!
//! ## The tag bit, and why it is free
//!
//! Every value is allocated with `Layout::from_size_align(len.max(1), 8)`, so
//! the address is always 8-aligned and its low THREE bits are always zero.
//! Bit 0 carries the tier (`0` = fast, `1` = slow); bits 1 and 2 are left
//! spare. jemalloc's smallest size class is 8 bytes and 8-aligned, so asking
//! for 8-byte alignment moves nothing into a larger class -- `nallocx(8) == 8`
//! is asserted by `numa_alloc`'s own rounding harness. The alignment is what
//! makes the tag free, so `new_in` debug-asserts it on every allocation and
//! `values_are_eight_aligned_so_the_tag_bit_is_free` asserts it in tests.
//!
//! `len.max(1)` is not cosmetic: a zero-sized allocation is not required to
//! return a unique address, and a value's address IS its identity for the
//! migration check below.
//!
//! ## Lifetime: there is no `Drop`, deliberately
//!
//! `TieredValue` does NOT implement `Drop`, and it is `Copy`. A copy is a
//! BORROW of the allocation, never ownership of it: it is what lets a reader
//! lift the pointer out from under a shard lock, release the lock, and only
//! then copy the bytes. Exactly one operation frees -- [`TieredValue::free`],
//! which is `unsafe` and carries the obligation that no copy is used again.
//!
//! In v5 that obligation is discharged by crossbeam-epoch: a writer replacing
//! a value defers `free` under the shard write guard, and the free runs only
//! once every reader pin that could have observed the old pointer has ended.
//! That is step 2; this module only supplies the primitive.
//!
//! ## Identity survives, so migration stays safe
//!
//! `Shared::ptr_eq` was the check that made an in-flight tier copy safe to
//! apply: a value replaced by a `set()` mid-copy is rejected rather than
//! overwritten. That check becomes [`TieredValue::raw`] equality, and it is
//! exact for the same reason it was before -- the old allocation cannot be
//! freed, and therefore its address cannot be recycled, while the migrating
//! consumer's own epoch pin is live. `raw` returns the UNTAGGED address, so
//! it is the identity of the allocation rather than of the handle.
//!
//! ## Routing: the tier bit names the allocator that must free it
//!
//! Fast values come from the global allocator -- or, under
//! `segregated_value_arena`, from `numa_alloc::FastValues`, which is a
//! separate arena set with its own tcache and is NOT the global allocator.
//! Slow values come from `numa_alloc::SlowObjects` (node-1 arenas). A value
//! is frequently freed on a different thread from the one that made it, so
//! the routing cannot depend on thread-local state -- only on the tag bit.
//! [`free`](TieredValue::free) is the single place that decision is made, and
//! the tests below assert it three independent ways, none of which subsumes
//! another:
//!
//! 1. A counting wrapper around the four allocator entry points. It proves
//!    which entry point was CALLED, so a free routed by anything other than
//!    the tag moves the wrong counter.
//! 2. An arena-pool check (`arenas.lookup`) that names the jemalloc arena
//!    each value's bytes actually came from. This is the only one that
//!    catches the `segregated_value_arena` trap: `FastValues` and the global
//!    allocator are both `mbind`ed to physical node 0, so a `fast_alloc` that
//!    wrongly called `std::alloc::alloc` lands on the right NODE and passes
//!    the counters (which sit inside `fast_alloc` and count whatever it
//!    calls) -- only the arena index differs.
//! 3. A NUMA placement check asking the kernel which node each value's pages
//!    landed on. Ground truth about physical placement, which the arena index
//!    only implies.

use std::{alloc::Layout, ptr::NonNull};

use crate::Tier;

/// Bit 0 of the value word: set means the slow tier, clear means the fast one.
const SLOW_BIT: usize = 0b1;

/// Alignment demanded of every value allocation, which is what keeps the low
/// three bits of the address available for tagging.
const VALUE_ALIGN: usize = 8;

/// A cached value: a pointer to its bytes, with its tier in the low bit.
///
/// See the module documentation. Exactly 8 bytes, and `Option<TieredValue>` is
/// 8 bytes too -- the `NonNull` is the niche, which is what keeps
/// `Option<Object>` at 24 bytes in the merged store's slot.
#[repr(transparent)]
#[derive(Clone, Copy)]
pub struct TieredValue {
	/// The address of the value bytes, with the tier OR-ed into bit 0. Never
	/// dereference this directly -- go through [`TieredValue::raw`], which
	/// strips the tag.
	word: NonNull<u8>,
}

// The bytes behind the pointer are immutable for the whole life of the value
// (a "modification" allocates a new value and defers the old one), so handing
// a `TieredValue` to another thread hands out `&[u8]` and nothing more. The
// same bounds `Shared<TieredBuffer>` carried, for the same reason -- and the
// free may likewise run on whichever thread the epoch advance lands on, which
// is precisely why `free` routes on the tag rather than on thread state.
unsafe impl Send for TieredValue {}
unsafe impl Sync for TieredValue {}

impl TieredValue {
	/// Creates a fast-tier (DRAM) value by copying `bytes`.
	#[inline]
	pub fn new_fast(bytes: &[u8]) -> Self {
		Self::new_in(bytes, Tier::Fast)
	}

	/// Creates a slow-tier (PMEM/CXL) value by copying `bytes`.
	#[inline]
	pub fn new_slow(bytes: &[u8]) -> Self {
		Self::new_in(bytes, Tier::Slow)
	}

	/// Creates a value in `tier` by copying `bytes`.
	///
	/// # Panics
	///
	/// If `bytes.len()` does not fit a `u32`. Variant B stores the length in
	/// the `Object` as a `u32`, and that length is what [`free`] and
	/// [`as_slice`] are handed later -- so a length that cannot round-trip
	/// through `u32` has to be refused HERE, where it is still a panic, rather
	/// than silently truncated into a mismatched deallocation layout.
	///
	/// [`free`]: TieredValue::free
	/// [`as_slice`]: TieredValue::as_slice
	pub fn new_in(bytes: &[u8], tier: Tier) -> Self {
		assert!(
			u32::try_from(bytes.len()).is_ok(),
			"a cached value must fit a u32 length; got {} bytes",
			bytes.len(),
		);

		let layout = value_layout(bytes.len() as u32);

		// SAFETY: `value_layout` never yields a zero-sized layout, which is
		// the only precondition either allocator entry point has.
		let raw = unsafe {
			match tier {
				Tier::Fast => fast_alloc(layout),
				Tier::Slow => slow_alloc(layout),
			}
		};

		let Some(ptr) = NonNull::new(raw) else {
			std::alloc::handle_alloc_error(layout);
		};

		debug_assert_eq!(
			ptr.as_ptr().addr() % VALUE_ALIGN,
			0,
			"the allocator returned an address that is not {VALUE_ALIGN}-aligned, \
			 so the tier tag would alias the address itself",
		);

		// SAFETY: `ptr` owns at least `bytes.len()` freshly allocated bytes
		// (`value_layout` only ever rounds a length UP, to 1), and a fresh
		// allocation cannot overlap the caller's slice.
		unsafe {
			std::ptr::copy_nonoverlapping(bytes.as_ptr(), ptr.as_ptr(), bytes.len());
		}

		TieredValue { word: tag(ptr, tier) }
	}

	/// Which tier this value's bytes physically live in.
	#[inline]
	pub fn tier(&self) -> Tier {
		if self.word.as_ptr().addr() & SLOW_BIT == 0 {
			Tier::Fast
		} else {
			Tier::Slow
		}
	}

	/// Whether this value currently lives in the fast (DRAM) tier.
	#[inline]
	pub fn is_fast(&self) -> bool {
		matches!(self.tier(), Tier::Fast)
	}

	/// Whether this value currently lives in the slow (PMEM/CXL) tier.
	#[inline]
	pub fn is_slow(&self) -> bool {
		matches!(self.tier(), Tier::Slow)
	}

	/// The address of the value bytes, with the tier tag stripped.
	///
	/// This is the value's IDENTITY, and the migration check is
	/// `old.raw() == current.raw()`. It is exact rather than merely likely:
	/// the old allocation cannot be freed -- and so its address cannot be
	/// recycled into a different value -- while the migrating consumer's epoch
	/// pin is live, which rules out ABA. Compare these, never the bytes: two
	/// distinct allocations holding equal content are NOT the same value, and
	/// treating them as such is what would let a migration overwrite a `set`.
	#[inline]
	pub fn raw(&self) -> *mut u8 {
		let untagged = self.word.as_ptr().map_addr(|addr| addr & !SLOW_BIT);

		debug_assert_eq!(
			untagged.addr() % VALUE_ALIGN,
			0,
			"a value address must stay {VALUE_ALIGN}-aligned; bits 1-2 of the \
			 word are reserved and must never be set",
		);

		untagged
	}

	/// The value's bytes.
	///
	/// # Safety
	///
	/// `len` MUST be the exact length this value was created with, and the
	/// value must not have been freed. Variant B keeps that length in the
	/// `Object` beside the pointer; this type is only the pointer half, so it
	/// cannot check the caller. The returned slice borrows `self`, which is
	/// what keeps a reader's handle alive for as long as it is copying bytes.
	#[inline]
	pub unsafe fn as_slice(&self, len: u32) -> &[u8] {
		// SAFETY: by the contract above, `raw()` points at `len` initialised
		// bytes in one allocation. `raw()` is non-null and 8-aligned even when
		// `len` is 0, which is what `from_raw_parts` requires for the empty
		// case.
		unsafe { std::slice::from_raw_parts(self.raw(), len as usize) }
	}

	/// Frees the value, returning its bytes to the allocator its TIER names.
	///
	/// The routing is the whole point: under `segregated_value_arena` a fast
	/// value came from `numa_alloc::FastValues`, not from the global
	/// allocator, and a slow value always came from `numa_alloc::SlowObjects`.
	/// Only the tag bit knows which, and the tag bit travels with the pointer
	/// -- so this is correct on whichever thread the epoch advance runs it.
	///
	/// # Safety
	///
	/// `len` MUST be the exact length this value was created with, or the
	/// deallocation layout will not match the allocation layout. After this
	/// returns, NO copy of this `TieredValue` may be used again -- and copies
	/// are cheap and easy to make, so in v5 the only correct caller is a
	/// `crossbeam_epoch` deferred closure created under the shard write guard
	/// that unpublished the pointer.
	#[inline]
	pub unsafe fn free(self, len: u32) {
		let layout = value_layout(len);
		let ptr = self.raw();

		// SAFETY: by the contract above `ptr` came from the allocator this
		// arm names, with exactly `layout`.
		unsafe {
			match self.tier() {
				Tier::Fast => fast_dealloc(ptr, layout),
				Tier::Slow => slow_dealloc(ptr, layout),
			}
		}
	}
}

impl std::fmt::Debug for TieredValue {
	fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
		f.debug_struct("TieredValue")
			.field("tier", &self.tier())
			.field("raw", &self.raw())
			.finish()
	}
}

/// The layout every value of `len` bytes is allocated and freed with.
///
/// `len.max(1)` keeps the allocation addressable and, more importantly,
/// UNIQUE -- see the module documentation on identity. `VALUE_ALIGN` is what
/// reserves the low bits for the tag.
#[inline]
fn value_layout(len: u32) -> Layout {
	// `len` is a `u32` and `VALUE_ALIGN` is 8, so the rounded size cannot
	// overflow an `isize` on any target this crate builds for; the `expect`
	// documents that rather than guarding a reachable case.
	Layout::from_size_align((len as usize).max(1), VALUE_ALIGN)
		.expect("a u32 length can always be laid out with 8-byte alignment")
}

/// Folds `tier` into bit 0 of an 8-aligned value address.
#[inline]
fn tag(ptr: NonNull<u8>, tier: Tier) -> NonNull<u8> {
	match tier {
		Tier::Fast => ptr,

		// SAFETY: setting a bit cannot turn a non-null address into a null
		// one, so the result is still a valid `NonNull`.
		Tier::Slow => unsafe {
			NonNull::new_unchecked(ptr.as_ptr().map_addr(|addr| addr | SLOW_BIT))
		},
	}
}

// ---------------------------------------------------------------------------
// allocator routing -- the four entry points, and nothing else in the crate
// may call an allocator for a value
// ---------------------------------------------------------------------------

/// # Safety
///
/// `layout` must have a non-zero size.
#[inline]
unsafe fn fast_alloc(layout: Layout) -> *mut u8 {
	#[cfg(test)]
	route_counts::FAST_ALLOCS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);

	// SAFETY: non-zero size, per the contract above.
	#[cfg(not(feature = "segregated_value_arena"))]
	return unsafe { std::alloc::alloc(layout) };

	// SAFETY: as above. `FastValues` is a distinct arena set from the global
	// allocator, so a value allocated here MUST come back to `fast_dealloc`.
	#[cfg(feature = "segregated_value_arena")]
	return unsafe { std::alloc::GlobalAlloc::alloc(&crate::numa_alloc::FastValues, layout) };
}

/// # Safety
///
/// `ptr` must have come from [`fast_alloc`] with exactly `layout`.
#[inline]
unsafe fn fast_dealloc(ptr: *mut u8, layout: Layout) {
	#[cfg(test)]
	route_counts::FAST_FREES.fetch_add(1, std::sync::atomic::Ordering::Relaxed);

	// SAFETY: per the contract above.
	#[cfg(not(feature = "segregated_value_arena"))]
	unsafe {
		std::alloc::dealloc(ptr, layout)
	};

	// SAFETY: per the contract above.
	#[cfg(feature = "segregated_value_arena")]
	unsafe {
		std::alloc::GlobalAlloc::dealloc(&crate::numa_alloc::FastValues, ptr, layout)
	};
}

/// # Safety
///
/// `layout` must have a non-zero size.
#[inline]
unsafe fn slow_alloc(layout: Layout) -> *mut u8 {
	#[cfg(test)]
	route_counts::SLOW_ALLOCS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);

	// SAFETY: non-zero size, per the contract above.
	unsafe { std::alloc::GlobalAlloc::alloc(&crate::numa_alloc::SlowObjects, layout) }
}

/// # Safety
///
/// `ptr` must have come from [`slow_alloc`] with exactly `layout`.
#[inline]
unsafe fn slow_dealloc(ptr: *mut u8, layout: Layout) {
	#[cfg(test)]
	route_counts::SLOW_FREES.fetch_add(1, std::sync::atomic::Ordering::Relaxed);

	// SAFETY: per the contract above.
	unsafe { std::alloc::GlobalAlloc::dealloc(&crate::numa_alloc::SlowObjects, ptr, layout) }
}

/// The counting wrapper the routing tests assert against.
///
/// Counting inside the four functions that NAME a concrete allocator, rather
/// than inside `free`, is what makes the assertion meaningful: `free`'s only
/// job is to pick an arm from the tag, so a mis-routed free moves the wrong
/// counter and the test fails. Compiled only under `cfg(test)`, so the release
/// path is the bare allocator call.
///
/// jemalloc cannot make this assertion for us -- it looks the owning arena up
/// from the extent, so returning a node-1 block through the node-0 entry point
/// is silently tolerated rather than reported.
#[cfg(test)]
pub(crate) mod route_counts {
	use std::sync::atomic::{AtomicU64, Ordering};

	pub(crate) static FAST_ALLOCS: AtomicU64 = AtomicU64::new(0);
	pub(crate) static FAST_FREES: AtomicU64 = AtomicU64::new(0);
	pub(crate) static SLOW_ALLOCS: AtomicU64 = AtomicU64::new(0);
	pub(crate) static SLOW_FREES: AtomicU64 = AtomicU64::new(0);

	#[derive(Clone, Copy, Debug, PartialEq, Eq)]
	pub(crate) struct Counts {
		pub fast_allocs: u64,
		pub fast_frees: u64,
		pub slow_allocs: u64,
		pub slow_frees: u64,
	}

	pub(crate) fn snapshot() -> Counts {
		Counts {
			fast_allocs: FAST_ALLOCS.load(Ordering::Relaxed),
			fast_frees: FAST_FREES.load(Ordering::Relaxed),
			slow_allocs: SLOW_ALLOCS.load(Ordering::Relaxed),
			slow_frees: SLOW_FREES.load(Ordering::Relaxed),
		}
	}
}

#[cfg(test)]
mod tests {
	use super::*;
	use std::sync::{Mutex, MutexGuard};

	/// EVERY test below that allocates a `TieredValue` must hold this, not
	/// just the two that read the counters.
	///
	/// The counters are process-global and the test runner is parallel by
	/// default, so a sibling test allocating one slow value between a
	/// snapshot and its successor moves the count under the reader and the
	/// routing assertion fails on a delta of 2. That is precisely what
	/// happened the first time this was written with the lock on the counting
	/// tests alone -- an intermittent failure in four of eight feature sets
	/// and a clean pass in the other four, which is the worst possible signal.
	/// Nothing outside this module allocates a `TieredValue` yet, so
	/// serialising the module is the whole fix; when something does, it will
	/// need its own answer.
	///
	/// Poisoning is stepped over on purpose: a panicking test is already a
	/// failure, and letting it cascade into every other test's error message
	/// only hides which one broke.
	fn routing_lock() -> MutexGuard<'static, ()> {
		static LOCK: Mutex<()> = Mutex::new(());
		LOCK.lock().unwrap_or_else(|poisoned| poisoned.into_inner())
	}

	/// The saving this whole type exists for, asserted rather than assumed.
	#[test]
	fn a_value_is_one_word_and_the_option_is_free() {
		assert_eq!(
			core::mem::size_of::<TieredValue>(),
			8,
			"a value is a tagged pointer and nothing else -- 8 bytes, not the \
			 8 + 32 a Shared<TieredBuffer> handle plus its Inner cost",
		);
		assert_eq!(core::mem::align_of::<TieredValue>(), 8, "it is a pointer");

		assert_eq!(
			core::mem::size_of::<Option<TieredValue>>(),
			8,
			"the NonNull must stay the niche: Option<Object> is what keeps a \
			 merged-store slot at 56 bytes, and it only does so if the value \
			 pointer costs nothing to make optional",
		);
	}

	#[test]
	fn fast_values_round_trip_their_bytes() {
		let _guard = routing_lock();

		let value = TieredValue::new_fast(b"hello");

		assert!(value.is_fast());
		assert!(!value.is_slow());
		assert_eq!(value.tier(), Tier::Fast);
		assert_eq!(unsafe { value.as_slice(5) }, b"hello");

		unsafe { value.free(5) };
	}

	#[test]
	fn slow_values_round_trip_their_bytes() {
		let _guard = routing_lock();

		let value = TieredValue::new_slow(b"world!");

		assert!(value.is_slow());
		assert!(!value.is_fast());
		assert_eq!(value.tier(), Tier::Slow);
		assert_eq!(unsafe { value.as_slice(6) }, b"world!");

		unsafe { value.free(6) };
	}

	/// Long enough to cross a page and a few size classes, so the round trip
	/// is not just testing the first cache line of a fresh allocation.
	#[test]
	fn values_round_trip_at_every_size_in_both_tiers() {
		let _guard = routing_lock();

		for len in [0usize, 1, 2, 3, 7, 8, 9, 15, 16, 17, 63, 64, 100, 128, 1023, 1024, 4096] {
			let bytes: Vec<u8> = (0..len).map(|i| (i % 251) as u8).collect();

			for tier in [Tier::Fast, Tier::Slow] {
				let value = TieredValue::new_in(&bytes, tier);

				assert_eq!(value.tier(), tier, "len {len}: wrong tier");
				assert_eq!(
					unsafe { value.as_slice(len as u32) },
					&bytes[..],
					"len {len} in {tier:?}: bytes did not survive the copy",
				);

				unsafe { value.free(len as u32) };
			}
		}
	}

	/// The tag is only free because the address never uses the low bits.
	#[test]
	fn values_are_eight_aligned_so_the_tag_bit_is_free() {
		let _guard = routing_lock();

		for len in [0usize, 1, 2, 3, 5, 7, 8, 9, 12, 13, 16, 17, 31, 100, 1000, 4097] {
			let bytes = vec![0xA5u8; len];

			for tier in [Tier::Fast, Tier::Slow] {
				let value = TieredValue::new_in(&bytes, tier);

				assert_eq!(
					value.raw().addr() % VALUE_ALIGN,
					0,
					"len {len} in {tier:?}: allocation is not 8-aligned, so the \
					 tier tag would corrupt the address",
				);
				assert!(!value.raw().is_null(), "len {len} in {tier:?}: null value address");

				// The word is the address plus the tag, and NOTHING else:
				// bits 1 and 2 are reserved and must stay clear.
				let expected_tag = match tier {
					Tier::Fast => 0,
					Tier::Slow => SLOW_BIT,
				};

				assert_eq!(
					value.word.as_ptr().addr(),
					value.raw().addr() | expected_tag,
					"len {len} in {tier:?}: the word is not exactly address | tag",
				);
				assert_eq!(
					value.word.as_ptr().addr() & 0b110,
					0,
					"len {len} in {tier:?}: bits 1-2 are reserved and must be clear",
				);

				unsafe { value.free(len as u32) };
			}
		}
	}

	/// A zero-length value is the case where an implementation is most likely
	/// to hand back a null, a dangling `align` sentinel, or a shared address.
	#[test]
	fn a_zero_length_value_is_still_a_tagged_non_null_pointer() {
		let _guard = routing_lock();

		let fast = TieredValue::new_fast(&[]);
		let slow = TieredValue::new_slow(&[]);

		for (value, tier) in [(fast, Tier::Fast), (slow, Tier::Slow)] {
			assert!(!value.raw().is_null(), "{tier:?}: a zero-length value must not be null");
			assert_eq!(value.raw().addr() % VALUE_ALIGN, 0, "{tier:?}: must stay 8-aligned");
			assert_eq!(value.tier(), tier, "{tier:?}: the tag must survive a zero length");
			assert_eq!(unsafe { value.as_slice(0) }, b"", "{tier:?}: must read back empty");
		}

		// And they must be distinct allocations, because the address is the
		// identity the migration check compares.
		assert_ne!(
			fast.raw(),
			slow.raw(),
			"zero-length values must still get unique addresses -- that is why \
			 the layout is len.max(1) and not len",
		);

		let another_fast = TieredValue::new_fast(&[]);
		assert_ne!(fast.raw(), another_fast.raw(), "two zero-length values must not alias");

		unsafe {
			fast.free(0);
			slow.free(0);
			another_fast.free(0);
		}
	}

	/// `raw()` replaces `Shared::ptr_eq`, so it must be identity, not equality.
	#[test]
	fn raw_is_identity_not_content_equality() {
		let _guard = routing_lock();

		let a = TieredValue::new_fast(b"same bytes");
		let b = TieredValue::new_fast(b"same bytes");

		assert_eq!(unsafe { a.as_slice(10) }, unsafe { b.as_slice(10) }, "the bytes are equal");
		assert_ne!(a.raw(), b.raw(), "but they are not the same value");

		// A copy of a handle IS the same value -- that is what a migrating
		// consumer holds across the copy.
		let copy = a;
		assert_eq!(a.raw(), copy.raw(), "a copy must name the same allocation");

		unsafe {
			a.free(10);
			b.free(10);
		}
	}

	/// `free` must pick its allocator from the TAG. The counters sit inside
	/// the four functions that name a concrete allocator, so a free that went
	/// to the wrong one moves the wrong counter and this fails.
	#[test]
	fn free_routes_to_the_allocator_its_tier_names() {
		let _guard = routing_lock();

		let before = route_counts::snapshot();
		let fast = TieredValue::new_fast(b"fast");
		let after_alloc = route_counts::snapshot();

		assert_eq!(
			after_alloc.fast_allocs,
			before.fast_allocs + 1,
			"new_fast must allocate from the fast allocator",
		);
		assert_eq!(
			after_alloc.slow_allocs, before.slow_allocs,
			"new_fast must not touch the slow allocator",
		);

		unsafe { fast.free(4) };
		let after_free = route_counts::snapshot();

		assert_eq!(
			after_free.fast_frees,
			before.fast_frees + 1,
			"a fast value must be returned to the fast allocator",
		);
		assert_eq!(
			after_free.slow_frees, before.slow_frees,
			"a fast value must NEVER be returned to the slow allocator",
		);

		let before = route_counts::snapshot();
		let slow = TieredValue::new_slow(b"slow");
		let after_alloc = route_counts::snapshot();

		assert_eq!(
			after_alloc.slow_allocs,
			before.slow_allocs + 1,
			"new_slow must allocate from the slow allocator",
		);
		assert_eq!(
			after_alloc.fast_allocs, before.fast_allocs,
			"new_slow must not touch the fast allocator",
		);

		unsafe { slow.free(4) };
		let after_free = route_counts::snapshot();

		assert_eq!(
			after_free.slow_frees,
			before.slow_frees + 1,
			"a slow value must be returned to the slow allocator",
		);
		assert_eq!(
			after_free.fast_frees, before.fast_frees,
			"a slow value must NEVER be returned to the fast allocator -- under \
			 segregated_value_arena that is a cross-arena free",
		);
	}

	/// The free routinely happens on a different thread from the allocation:
	/// an epoch advance runs a deferred closure on whichever thread happens to
	/// collect. Routing must therefore depend on the tag alone, never on
	/// thread-local allocator state.
	#[test]
	fn free_routes_correctly_from_a_foreign_thread() {
		let _guard = routing_lock();

		let before = route_counts::snapshot();

		let fast = TieredValue::new_fast(b"made here, freed there");
		let slow = TieredValue::new_slow(b"made here, freed there");
		let len = "made here, freed there".len() as u32;

		std::thread::spawn(move || unsafe {
			slow.free(len);
			fast.free(len);
		})
		.join()
		.expect("the freeing thread must not panic");

		let after = route_counts::snapshot();

		assert_eq!(
			after.fast_frees,
			before.fast_frees + 1,
			"the fast value must still go back to the fast allocator off-thread",
		);
		assert_eq!(
			after.slow_frees,
			before.slow_frees + 1,
			"the slow value must still go back to the slow allocator off-thread",
		);
	}

	/// The counting wrapper proves which entry point was CALLED; this proves
	/// the memory actually landed where the tier claims, by asking the kernel.
	///
	/// jemalloc will silently accept a block returned through the wrong entry
	/// point, so neither check subsumes the other.
	///
	/// Correct under `--test-threads=1` and under the default runner alike:
	/// `node_of` answers for one address via `get_mempolicy`, unlike
	/// `numa_alloc`'s `placement_check`, which diffs process-wide
	/// `/proc/self/numa_maps` and therefore needs the process to itself.
	///
	/// Skipped under `stock_jemalloc`, where the global allocator is
	/// deliberately unbound and so has no node to assert.
	#[test]
	#[cfg(not(feature = "stock_jemalloc"))]
	fn values_land_on_the_numa_node_their_tier_names() {
		use crate::numa_alloc::{self, tests::node_of, NODE_FAST, NODE_SLOW};

		assert!(numa_alloc::init(), "the node-0 and node-1 arena pools must build");

		#[cfg(feature = "segregated_value_arena")]
		assert!(
			numa_alloc::init_node(numa_alloc::NODE_FAST_VALUES),
			"the segregated value pool must build",
		);

		let _guard = routing_lock();

		// Big enough to be genuinely faulted in by the copy: `get_mempolicy`
		// reports the node of a PAGE, and an untouched page has none.
		let bytes = vec![0x5Au8; 8192];

		let fast = TieredValue::new_fast(&bytes);
		let slow = TieredValue::new_slow(&bytes);

		let fast_node = node_of(fast.raw());
		let slow_node = node_of(slow.raw());

		// Under `segregated_value_arena` the fast tier draws from
		// `NODE_FAST_VALUES`, a separate ARENA set that is still mbind-ed to
		// the physical fast node -- so the expected node is `NODE_FAST` either
		// way. See `numa_alloc::physical_node`.
		assert_eq!(
			fast_node, NODE_FAST as i32,
			"a fast value must be on node {NODE_FAST}, not node {fast_node}",
		);
		assert_eq!(
			slow_node, NODE_SLOW as i32,
			"a slow value must be on node {NODE_SLOW}, not node {slow_node}",
		);

		unsafe {
			fast.free(bytes.len() as u32);
			slow.free(bytes.len() as u32);
		}
	}

	/// Interleaving equal-sized allocations across tiers on one thread is the
	/// shortest path to a size-class cache handing a node-1 block back for a
	/// node-0 request. Each value must keep its own address and its own tag.
	#[test]
	fn interleaved_tiers_keep_distinct_addresses_and_tags() {
		let _guard = routing_lock();

		const N: usize = 64;
		const LEN: u32 = 64;

		let bytes = vec![0x11u8; LEN as usize];
		let mut values = Vec::with_capacity(N * 2);

		for _ in 0..N {
			values.push((TieredValue::new_fast(&bytes), Tier::Fast));
			values.push((TieredValue::new_slow(&bytes), Tier::Slow));
		}

		let mut seen = std::collections::HashSet::new();

		for (value, tier) in &values {
			assert_eq!(value.tier(), *tier, "a value forgot its tier");
			assert_eq!(unsafe { value.as_slice(LEN) }, &bytes[..], "a value lost its bytes");
			assert!(seen.insert(value.raw()), "two live values share an address");
		}

		for (value, _) in values {
			unsafe { value.free(LEN) };
		}
	}

	/// The alignment must be REQUESTED, not merely received.
	///
	/// `values_are_eight_aligned_so_the_tag_bit_is_free` asserts the outcome,
	/// and that is not enough on its own: jemalloc's smallest size class is 8
	/// bytes and 8-aligned, so dropping `VALUE_ALIGN` from the layout entirely
	/// leaves every address 8-aligned anyway and that test still passes. It was
	/// checked -- mutating the layout's alignment to 1 is invisible to every
	/// other test in this file. The tag would then rest on an allocator
	/// accident rather than on a contract, and `free` would be handed a layout
	/// that no longer matches the one `new_in` allocated with the moment either
	/// side is edited independently. Hence the layout itself is the assertion.
	#[test]
	fn the_value_layout_always_demands_eight_byte_alignment() {
		for len in [0u32, 1, 2, 3, 7, 8, 9, 15, 16, 100, 4096, u32::MAX] {
			let layout = value_layout(len);

			assert_eq!(
				layout.align(),
				VALUE_ALIGN,
				"len {len}: the layout must DEMAND {VALUE_ALIGN}-byte alignment, so \
				 the low bits are reserved by contract rather than by luck",
			);

			// `len.max(1)`: a zero-sized allocation need not return a unique
			// address, and the address IS the migration identity.
			assert_eq!(
				layout.size(),
				(len as usize).max(1),
				"len {len}: a value's layout is len.max(1), never len",
			);
		}
	}

	/// Which POOL served the bytes -- the check `node_of` structurally cannot
	/// make.
	///
	/// Under `segregated_value_arena` the fast allocator is
	/// `numa_alloc::FastValues`, NOT the global allocator, and BOTH are
	/// `mbind`ed to physical node 0 (see `numa_alloc::physical_node`). So a
	/// `fast_alloc` that wrongly called `std::alloc::alloc` would place its
	/// bytes on exactly the right node and
	/// `values_land_on_the_numa_node_their_tier_names` would pass; so would the
	/// counting wrapper, which sits inside `fast_alloc` and counts whatever it
	/// calls. `arenas.lookup` names the owning arena, and the three pools hold
	/// disjoint arena sets (asserted by `numa_alloc`'s own
	/// `arena_lookup_answers_for_each_pool`), so this is the check that fails.
	///
	/// Both a tcache-sized and an extent-sized value, because a leak between
	/// pools is likeliest on the cached path, where a cache bin is indexed by
	/// size class alone.
	///
	/// Skipped under `stock_jemalloc`, whose global allocator is deliberately
	/// unbound and so belongs to no pool.
	#[test]
	#[cfg(not(feature = "stock_jemalloc"))]
	fn each_tier_draws_from_the_arena_pool_its_tier_names() {
		use crate::numa_alloc::{
			self, NODE_FAST, NODE_FAST_VALUES, NODE_SLOW,
			tests::{arena_of, pool_arenas},
		};

		assert!(numa_alloc::init(), "the node-0 and node-1 arena pools must build");

		// The fast tier's pool is the whole point of the feature.
		let fast_pool = if cfg!(feature = "segregated_value_arena") {
			assert!(
				numa_alloc::init_node(NODE_FAST_VALUES),
				"the segregated value pool must build",
			);
			NODE_FAST_VALUES
		} else {
			NODE_FAST
		};

		let _guard = routing_lock();

		let fast_arenas = pool_arenas(fast_pool);
		let slow_arenas = pool_arenas(NODE_SLOW);

		assert!(!fast_arenas.is_empty(), "pool {fast_pool} reported no arenas");
		assert!(!slow_arenas.is_empty(), "pool {NODE_SLOW} reported no arenas");

		// 64 B is served from a tcache; 64 KiB is above the default
		// `opt.tcache_max` of 32 KiB and so comes straight from an extent.
		for len in [64u32, 65_536] {
			let bytes = vec![0x7Eu8; len as usize];

			let fast = TieredValue::new_fast(&bytes);
			let slow = TieredValue::new_slow(&bytes);

			let fast_arena = arena_of(fast.raw())
				.expect("arenas.lookup must be available in this jemalloc build");
			let slow_arena = arena_of(slow.raw())
				.expect("arenas.lookup must be available in this jemalloc build");

			assert!(
				fast_arenas.contains(&fast_arena),
				"len {len}: a fast value came from arena {fast_arena}, which is not \
				 in pool {fast_pool}'s set {fast_arenas:?} -- under \
				 segregated_value_arena the fast allocator is FastValues, not the \
				 global allocator",
			);
			assert!(
				slow_arenas.contains(&slow_arena),
				"len {len}: a slow value came from arena {slow_arena}, which is not \
				 in pool {NODE_SLOW}'s set {slow_arenas:?}",
			);

			unsafe {
				fast.free(len);
				slow.free(len);
			}
		}
	}
}
