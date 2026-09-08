/*
 * Copyright (c) Kia Shakiba
 *
 * This source code is licensed under the GNU AGPLv3 license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! `TieredValue` -- a cached value as ONE refcounted DRAM header pointing at
//! bytes that live in whichever tier the object is currently placed in.
//!
//! This is the only module in the crate allowed to contain unsafe value code.
//! Every allocation, every free, and every reinterpretation of a value pointer
//! lives here.
//!
//! ## The shape, and what each part of it is for
//!
//! ```text
//!   Object                 TieredValue = Arc<ValueHeader<K>>      8 B handle
//!                            |
//!                            v
//!   DRAM, always      +---------------------------+
//!                     | strong count           8  |   std Arc
//!                     | weak count             8  |   (unused; see below)
//!                     | key: K                 8  |   the collision check
//!                     | bytes: ValueBytes      8  |   tagged ptr, tier in bit 0
//!                     | len: u32               4  |
//!                     | expiry: AtomicU32      4  |
//!                     +---------------------------+   40 -> 48 size class
//!                            |
//!                            v
//!   fast OR slow tier  [ the value's bytes ]          this is what migrates
//! ```
//!
//! Two allocations, and the split between them is the whole design:
//!
//!   * the HEADER IS ALWAYS IN DRAM. The key, the expiry and the tier tag are
//!     metadata every hot path touches -- a collision check, a TTL sweep, a
//!     tier query -- and none of them should cross the interconnect just
//!     because the object's bytes happen to be in CXL.
//!   * only the BYTES are tiered. A migration allocates new bytes in the
//!     target tier, builds a new header around them, and swaps the `Arc`.
//!
//! ## Why a refcount rather than epoch reclamation
//!
//! A reader needs the bytes to stay alive while it copies them with the shard
//! lock released. Two ways to buy that: an epoch pin (v5 phase 1) or a strong
//! reference. This is the strong reference. `get` clones the `Arc` under the
//! shard guard, drops the guard, and copies; the value cannot be freed while
//! that clone is live, whoever else replaces it in the map.
//!
//! The cost is one atomic increment and one decrement per read. What it buys
//! back is the deletion of the entire pin discipline -- `crossbeam_epoch`, the
//! per-pass flushes, `defer_free`'s unchecked `len` precondition, and the
//! torn-read stress harness that existed to prove all of it -- and an exact
//! migration identity check that needs no ABA argument at all: the migrating
//! thread HOLDS a strong reference to the value it snapshotted, so that
//! allocation cannot be freed and its address cannot be recycled, full stop.
//!
//! ## Why `triomphe::Arc`, and why NOT `ThinArc`
//!
//! An `Arc` over a SIZED `T` is already a thin 8-byte pointer -- the
//! fat-pointer problem only arises for `Arc<[u8]>`, and the bytes are not
//! inside the header. So no DST trickery is needed and `ThinArc` is the wrong
//! tool: what it exists to provide, a thin handle onto a runtime-length slice,
//! is exactly the thing this design deliberately does not want, because a
//! header fused to its bytes would have to live in the bytes' tier.
//!
//! `triomphe::Arc` rather than `std::sync::Arc` for one reason, which is
//! worth 16 bytes an object:
//!
//! ```text
//!   std::sync::Arc   strong 8 + weak 8 + header 24 = 40  -> 48 B size class
//!   triomphe::Arc    strong 8            + header 24 = 32 -> 32 B size class
//! ```
//!
//! std's `weak` count is never used here and never can be -- nothing in this
//! crate holds a weak reference to a value -- and paying for it spills the
//! allocation into jemalloc's next class. `triomphe` is the same `Arc` without
//! it (`ArcInner { count, data }`), with the same `MAX_REFCOUNT` overflow
//! guard, maintained and in production in Servo and rust-analyzer. Preferred
//! over reviving this crate's own deleted `Shared` for exactly that reason:
//! the same 32 bytes, with none of the hand-rolled unsafe -- and `Shared`'s
//! own history here is the argument, since its first version shipped without
//! an overflow guard at all.
//!
//! Its one hard limitation -- it allocates from the GLOBAL allocator, with no
//! hook -- is what rules it out for the value bytes, and is a non-issue for
//! the header, which is meant to be in DRAM. The global allocator IS the DRAM
//! allocator here. The bytes never pass through `triomphe`; they keep this
//! module's own `fast_alloc`/`slow_alloc` routing.
//!
//! ## The tag bit, and why it is free
//!
//! Every value's BYTES are allocated with
//! `Layout::from_size_align(len.max(1), 8)`, so the address is 8-aligned and
//! its low three bits are always zero. Bit 0 carries the tier (`0` = fast,
//! `1` = slow); bits 1 and 2 are spare. jemalloc's smallest size class is 8
//! bytes and 8-aligned, so demanding 8-byte alignment moves nothing into a
//! larger class.
//!
//! `len.max(1)` is not cosmetic: a zero-sized allocation is not required to
//! return a unique address.
//!
//! ## Routing: the tier bit names the allocator that must free it
//!
//! Fast values come from the global allocator -- or, under
//! `segregated_value_arena`, from `numa_alloc::FastValues`, which is a
//! separate arena set with its own tcache and is NOT the global allocator.
//! Slow values come from `numa_alloc::SlowObjects` (node-1 arenas). A value is
//! frequently freed on a different thread from the one that made it -- the
//! last `Arc` to drop may be any reader's -- so the routing cannot depend on
//! thread-local state, only on the tag bit. [`ValueBytes::free`] is the single
//! place that decision is made, and the tests below assert it three
//! independent ways, none of which subsumes another:
//!
//! 1. A counting wrapper around the four allocator entry points. It proves
//!    which entry point was CALLED, so a free routed by anything other than
//!    the tag moves the wrong counter.
//! 2. An arena-pool check (`arenas.lookup`) naming the jemalloc arena each
//!    value's bytes actually came from. This is the only one that catches the
//!    `segregated_value_arena` trap: `FastValues` and the global allocator are
//!    both `mbind`ed to physical node 0, so a `fast_alloc` that wrongly called
//!    `std::alloc::alloc` lands on the right NODE and passes the counters --
//!    only the arena index differs.
//! 3. A NUMA placement check asking the kernel which node each value's pages
//!    landed on. Ground truth about physical placement, which the arena index
//!    only implies.

use std::{
	alloc::Layout,
	ptr::NonNull,
	sync::atomic::{AtomicU32, AtomicU64, Ordering},
};

/// The refcount. `triomphe::Arc` rather than `std::sync::Arc` because std's
/// carries a `weak` count this design never uses, and those 8 bytes are the
/// difference between the header fitting jemalloc's 32-byte class exactly and
/// spilling into the 48. See the module documentation.
use triomphe::Arc;

use crate::{Tier, object::ExpireTime};

/// Bit 0 of the byte-pointer word: set means the slow tier, clear the fast one.
const SLOW_BIT: usize = 0b1;

/// Alignment demanded of every value allocation, which is what keeps the low
/// three bits of the address available for tagging.
const VALUE_ALIGN: usize = 8;

// ---------------------------------------------------------------------------
// the bytes: a tagged pointer, owned by the header
// ---------------------------------------------------------------------------

/// The value's bytes: a pointer to them, with their tier in the low bit.
///
/// Not public API. Exactly one [`ValueHeader`] owns each of these, and that
/// header's `Drop` is the only thing that frees it -- so unlike v5 phase 1's
/// bare pointer there is no deferral obligation to discharge anywhere else.
#[repr(transparent)]
#[derive(Clone, Copy)]
pub(crate) struct ValueBytes {
	/// The address of the bytes, with the tier OR-ed into bit 0. Never
	/// dereference directly -- go through [`ValueBytes::raw`].
	word: NonNull<u8>,
}

// The bytes are immutable for the whole life of the allocation (a
// "modification" allocates a new value), so sharing one across threads hands
// out `&[u8]` and nothing more. The free may run on whichever thread drops the
// last `Arc`, which is precisely why `free` routes on the tag rather than on
// thread state.
unsafe impl Send for ValueBytes {}
unsafe impl Sync for ValueBytes {}

impl ValueBytes {
	/// Allocates `tier` bytes and copies `bytes` into them.
	///
	/// # Panics
	///
	/// If `bytes.len()` does not fit a `u32`. The length is stored in the
	/// header as a `u32` and is what [`free`](ValueBytes::free) is handed
	/// later, so a length that cannot round-trip has to be refused HERE, where
	/// it is still a panic, rather than silently truncated into a mismatched
	/// deallocation layout.
	fn new_in(bytes: &[u8], tier: Tier) -> Self {
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

		ValueBytes { word: tag(ptr, tier) }
	}

	/// Which tier these bytes physically live in.
	#[inline]
	fn tier(&self) -> Tier {
		if self.word.as_ptr().addr() & SLOW_BIT == 0 {
			Tier::Fast
		} else {
			Tier::Slow
		}
	}

	/// The address of the bytes, with the tier tag stripped.
	#[inline]
	fn raw(&self) -> *mut u8 {
		let untagged = self.word.as_ptr().map_addr(|addr| addr & !SLOW_BIT);

		debug_assert_eq!(
			untagged.addr() % VALUE_ALIGN,
			0,
			"a value address must stay {VALUE_ALIGN}-aligned; bits 1-2 of the \
			 word are reserved and must never be set",
		);

		untagged
	}

	/// # Safety
	///
	/// `len` must be the exact length these bytes were created with, and they
	/// must not have been freed.
	#[inline]
	unsafe fn as_slice(&self, len: u32) -> &[u8] {
		// SAFETY: by the contract above, `raw()` points at `len` initialised
		// bytes in one allocation. `raw()` is non-null and 8-aligned even when
		// `len` is 0, which is what `from_raw_parts` requires for that case.
		unsafe { std::slice::from_raw_parts(self.raw(), len as usize) }
	}

	/// Returns the bytes to the allocator their TIER names.
	///
	/// # Safety
	///
	/// `len` must be the exact length these bytes were created with, or the
	/// deallocation layout will not match the allocation layout, and nothing
	/// may use this `ValueBytes` afterwards. The only caller is
	/// `ValueHeader::drop`, which owns both halves and runs exactly once.
	#[inline]
	unsafe fn free(self, len: u32) {
		let layout = value_layout(len);
		let ptr = self.raw();

		VALUE_FREES.fetch_add(1, Ordering::Relaxed);

		// SAFETY: by the contract above `ptr` came from the allocator this arm
		// names, with exactly `layout`.
		unsafe {
			match self.tier() {
				Tier::Fast => fast_dealloc(ptr, layout),
				Tier::Slow => slow_dealloc(ptr, layout),
			}
		}
	}
}

// ---------------------------------------------------------------------------
// the header: DRAM, refcounted, owns the bytes
// ---------------------------------------------------------------------------

/// Everything about a cached value except its bytes.
///
/// Allocated in DRAM by `Arc`, and freed -- along with the bytes it owns --
/// when the last [`TieredValue`] handle drops. `#[repr(C)]` so the field order
/// above the size assertions is the field order the compiler uses.
#[repr(C)]
pub struct ValueHeader<K> {
	/// The real key, kept for the hash-collision check. The object map is
	/// keyed on a 64-bit hash, so this is what distinguishes two keys that
	/// collide.
	#[cfg(not(feature = "key_pmem_value_pmem"))]
	key: K,

	/// Under `key_pmem_value_pmem` the key is owned in persistent memory
	/// instead, with no DRAM copy -- the header itself stays in DRAM, only the
	/// key's own allocation moves.
	#[cfg(feature = "key_pmem_value_pmem")]
	key: Box<K, crate::Hybrid>,

	/// The value's bytes and their tier. Owned: see `Drop`.
	bytes: ValueBytes,

	/// The bytes' length. The half that makes `as_slice` and the deallocation
	/// layout well defined.
	len: u32,

	/// The expiry tick, or `0` for "never expires" -- the same encoding
	/// [`ExpireTime`]'s `Option<NonZeroU32>` niche uses.
	///
	/// Atomic because the header is SHARED and `PaperCache::ttl` sets a TTL on
	/// a live object. A four-byte store is the whole operation, so this costs
	/// nothing over a plain field and avoids rebuilding the header -- which,
	/// since the header owns the bytes, would mean copying the value to change
	/// its TTL.
	expiry: AtomicU32,
}

/// Frees the bytes. The header itself is freed by `Arc`.
///
/// This is the single point every removal path funnels through -- a set
/// overwrite, an eviction, a TTL reap, a `wipe`, dropping the cache -- because
/// all of them drop an `Object`, which drops its handle. There is no site to
/// forget, and no deferral to get wrong: the refcount decides when.
impl<K> Drop for ValueHeader<K> {
	fn drop(&mut self) {
		// SAFETY: `bytes` and `len` are written together in
		// `TieredValue::new_in` and never separately afterwards, and this runs
		// exactly once, when the last handle drops.
		unsafe { self.bytes.free(self.len) }
	}
}

// ---------------------------------------------------------------------------
// the handle
// ---------------------------------------------------------------------------

/// A cached value: an eight-byte handle onto a DRAM header that owns tiered
/// bytes. See the module documentation.
///
/// Cloning is a refcount bump, not a copy -- which is what lets a reader lift
/// the value out from under the shard lock and copy the bytes with the lock
/// released.
pub struct TieredValue<K> {
	/// `triomphe::Arc`: strong count only, so the header allocation is 32
	/// bytes rather than 40. See the module doc.
	inner: Arc<ValueHeader<K>>,
}

impl<K> Clone for TieredValue<K> {
	/// A refcount bump. SHALLOW, and correct: the bytes are immutable, so two
	/// handles onto one allocation observe the same value forever.
	///
	/// This is the opposite of the pre-Arc `Object::clone`, which had to deep
	/// copy because the value was a bare pointer with no count and a shallow
	/// copy would have been a double free.
	#[inline]
	fn clone(&self) -> Self {
		TieredValue { inner: Arc::clone(&self.inner) }
	}
}

impl<K> TieredValue<K> {
	/// Builds a value: a DRAM header, and `bytes` copied into `tier`.
	pub fn new_in(key: K, bytes: &[u8], tier: Tier, expiry: ExpireTime) -> Self {
		// `ValueBytes::new_in` refuses a length that does not fit a `u32`, so
		// this cast cannot truncate.
		let len = bytes.len() as u32;

		TieredValue {
			inner: Arc::new(ValueHeader {
				#[cfg(not(feature = "key_pmem_value_pmem"))]
				key,
				#[cfg(feature = "key_pmem_value_pmem")]
				key: Box::new_in(key, crate::Hybrid),

				bytes: ValueBytes::new_in(bytes, tier),
				len,
				expiry: AtomicU32::new(expiry.map_or(0, |tick| tick.get())),
			}),
		}
	}

	/// Builds a value in the fast (DRAM) tier.
	#[inline]
	pub fn new_fast(key: K, bytes: &[u8], expiry: ExpireTime) -> Self {
		Self::new_in(key, bytes, Tier::Fast, expiry)
	}

	/// Builds a value in the slow (PMEM/CXL) tier.
	#[inline]
	pub fn new_slow(key: K, bytes: &[u8], expiry: ExpireTime) -> Self {
		Self::new_in(key, bytes, Tier::Slow, expiry)
	}

	/// The same value's bytes, re-copied into `tier`, carrying the key and the
	/// CURRENT expiry across.
	///
	/// This is a physical tier migration: a fresh header around fresh bytes,
	/// which the caller then swaps in under the shard guard after checking
	/// [`TieredValue::ptr_eq`] against the handle it snapshotted. Building it
	/// OUTSIDE the guard is the point -- the byte copy is the expensive part
	/// and may be a CXL write.
	pub fn migrated_to(&self, tier: Tier) -> Self
	where
		K: Clone,
	{
		Self::new_in(self.key().clone(), self.bytes(), tier, self.expiry())
	}

	/// The real key, for the hash-collision check.
	#[inline]
	pub fn key(&self) -> &K {
		#[cfg(not(feature = "key_pmem_value_pmem"))]
		return &self.inner.key;

		// Under `key_pmem_value_pmem` the key is a `Box<K, Hybrid>`, so the
		// comparison reads it from persistent memory -- which is the point of
		// that feature, and the reason this deref is not elided.
		#[cfg(feature = "key_pmem_value_pmem")]
		return &self.inner.key;
	}

	/// Whether this value's key is `key`. The check that makes a 64-bit hash
	/// collision harmless.
	#[inline]
	pub fn key_matches(&self, key: &K) -> bool
	where
		K: Eq,
	{
		self.key().eq(key)
	}

	/// The value's bytes.
	///
	/// Safe, because the header owns both the pointer and the length and this
	/// borrow keeps the header alive.
	#[inline]
	pub fn bytes(&self) -> &[u8] {
		// SAFETY: `len` is the length `bytes` was created with -- they are
		// written together in `new_in` and never separately -- and the header
		// cannot have been dropped while `self` holds a strong reference.
		unsafe { self.inner.bytes.as_slice(self.inner.len) }
	}

	/// The value's length in bytes.
	#[inline]
	pub fn len(&self) -> u32 {
		self.inner.len
	}

	#[inline]
	pub fn is_empty(&self) -> bool {
		self.inner.len == 0
	}

	/// Which tier the BYTES live in. The header is always DRAM.
	#[inline]
	pub fn tier(&self) -> Tier {
		self.inner.bytes.tier()
	}

	#[inline]
	pub fn is_fast(&self) -> bool {
		matches!(self.tier(), Tier::Fast)
	}

	#[inline]
	pub fn is_slow(&self) -> bool {
		matches!(self.tier(), Tier::Slow)
	}

	/// The expiry tick, or `None` if this value never expires.
	#[inline]
	pub fn expiry(&self) -> ExpireTime {
		std::num::NonZeroU32::new(self.inner.expiry.load(Ordering::Relaxed))
	}

	/// Sets the expiry. Visible to every handle onto this header, which is
	/// correct: they are the same object.
	#[inline]
	pub fn set_expiry(&self, expiry: ExpireTime) {
		self.inner
			.expiry
			.store(expiry.map_or(0, |tick| tick.get()), Ordering::Relaxed);
	}

	/// Whether the two handles name the SAME header allocation.
	///
	/// This is the migration identity check, and it is exact rather than
	/// merely likely: the caller holds a strong reference to the handle it
	/// snapshotted, so that allocation cannot be freed and its address cannot
	/// be recycled into a different value. Compare these, never the bytes --
	/// two distinct allocations holding equal content are NOT the same value,
	/// and treating them as such is what would let a migration overwrite a
	/// concurrent `set`.
	#[inline]
	pub fn ptr_eq(a: &Self, b: &Self) -> bool {
		Arc::ptr_eq(&a.inner, &b.inner)
	}

	/// The header's address, as an opaque identity for logging and tests.
	#[inline]
	pub fn raw(&self) -> *const ValueHeader<K> {
		Arc::as_ptr(&self.inner)
	}

	/// The raw tagged word naming this value's BYTES -- address with the tier
	/// in bit 0. Test-only: the tag discipline is asserted against it, and
	/// nothing in the release path should ever need the tagged form.
	#[cfg(test)]
	pub(crate) fn tagged_word(&self) -> usize {
		self.inner.bytes.word.as_ptr().addr()
	}

	/// How many handles currently name this header. Tests and diagnostics
	/// only.
	#[inline]
	pub fn strong_count(&self) -> usize {
		Arc::count(&self.inner)
	}
}

impl<K> std::fmt::Debug for TieredValue<K> {
	fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
		f.debug_struct("TieredValue")
			.field("tier", &self.tier())
			.field("len", &self.len())
			.field("header", &self.raw())
			.finish()
	}
}

/// The layout every value of `len` bytes is allocated and freed with.
///
/// `len.max(1)` keeps the allocation addressable and, more importantly,
/// UNIQUE. `VALUE_ALIGN` is what reserves the low bits for the tag.
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
	route_counts::bump(&route_counts::FAST_ALLOCS);

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
	route_counts::bump(&route_counts::FAST_FREES);

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
	route_counts::bump(&route_counts::SLOW_ALLOCS);

	// SAFETY: non-zero size, per the contract above.
	unsafe { std::alloc::GlobalAlloc::alloc(&crate::numa_alloc::SlowObjects, layout) }
}

/// # Safety
///
/// `ptr` must have come from [`slow_alloc`] with exactly `layout`.
#[inline]
unsafe fn slow_dealloc(ptr: *mut u8, layout: Layout) {
	#[cfg(test)]
	route_counts::bump(&route_counts::SLOW_FREES);

	// SAFETY: per the contract above.
	unsafe { std::alloc::GlobalAlloc::dealloc(&crate::numa_alloc::SlowObjects, ptr, layout) }
}

// ---------------------------------------------------------------------------
// lifetime: the refcount, and what is left to count
// ---------------------------------------------------------------------------

/// Value byte-allocations actually returned to an allocator.
///
/// v5 phase 1 needed a PAIR of counters here -- deferred and run -- because
/// epoch reclamation put an unbounded-looking gap between the two, and the gap
/// was the cache's un-reclaimed footprint. A refcount has no such gap: the
/// last handle to drop frees, synchronously, on that thread. So there is one
/// counter, and `deferred == run` is not a property that needs asserting
/// because there is nothing to defer.
pub static VALUE_FREES: AtomicU64 = AtomicU64::new(0);

// ---------------------------------------------------------------------------
// value shapes -- what is left of the old `ValueBuffer`
// ---------------------------------------------------------------------------

/// Which tier a non-hybrid cache shape builds its values in.
///
/// This is all that survives of the `ValueBuffer` trait. That trait existed to
/// abstract over two DIFFERENT value types -- `BufferDRAM = Box<[u8]>` and
/// `BufferPMEM = Box<[u8], Hybrid>` -- so `set()` could build one without
/// knowing which. Since v5 there is only one value type, [`TieredValue`], and
/// the two shapes differ in exactly one bit: which allocator their values come
/// from. So the trait collapses to a single associated constant, and
/// `from_bytes` collapses to `TieredValue::new_in(bytes, V::TIER)`.
///
/// The marker types below have no values and no fields; `V` survives purely as
/// a compile-time selector.
///
/// ## Why the shapes are still distinct types
///
/// The brief's preferred route was to delete this trait outright and let every
/// shape name `TieredValue` directly. That does not compile in the
/// all-features build, and the reason is structural rather than incidental:
/// `PaperCache`'s flat impl block is `impl<K, V, S> PaperCache<K, V, S> where
/// V: ValueShape` and the hybrid one is `impl<K, S> PaperCache<K, TieredBuffer,
/// S>`, and EVERY hybrid feature enables `key_value_pmem`, so both blocks are
/// compiled together in any build that has one. Collapsing the two `V`s to one
/// type makes those two blocks overlap -- a duplicate-inherent-impl error on
/// `get`, `set`, `peek` and the rest -- and merging them is not a rename: the
/// flat `new` accepts an all-DRAM policy that the hybrid `new` rejects, so the
/// 43-feature test suite would start failing at construction. Keeping one
/// zero-sized marker per shape preserves the disjointness for free, and the
/// value type really is single: both shapes store a `TieredValue`.
pub trait ValueShape: 'static + Send + Sync {
	/// The tier this shape's values are allocated in.
	const TIER: Tier;
}

/// The all-DRAM cache shape: values in the fast tier.
///
/// Under `segregated_value_arena` that means `numa_alloc::FastValues` rather
/// than the global allocator -- the routing lives in [`TieredValue`], so this
/// marker does not have to know, which is the whole reason the old
/// `BufferDRAM` type alias was feature-dependent and this one is not.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct BufferDRAM;

impl ValueShape for BufferDRAM {
	const TIER: Tier = Tier::Fast;
}

/// The PMEM/CXL cache shape: values in the slow tier (`numa_alloc::
/// SlowObjects`), matching what `BufferPMEM = Box<[u8], Hybrid>` allocated.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct BufferPMEM;

impl ValueShape for BufferPMEM {
	const TIER: Tier = Tier::Slow;
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
///
/// PER THREAD, not process-global. They were global until `Object` started
/// storing a `TieredValue`, at which point every other test in the crate began
/// allocating and freeing values too -- concurrently, on the default test
/// runner -- and the two counting tests here started failing about one run in
/// three on a delta of exactly one. Thread-local counters make the assertions
/// exact again regardless of what the rest of the suite is doing, and they
/// make the off-thread test STRONGER rather than weaker: it now reads the
/// counters on the thread that actually performed the free, which is the
/// thread whose routing decision is in question.
pub(crate) mod route_counts {
	use std::cell::Cell;

	thread_local! {
		pub(crate) static FAST_ALLOCS: Cell<u64> = const { Cell::new(0) };
		pub(crate) static FAST_FREES: Cell<u64> = const { Cell::new(0) };
		pub(crate) static SLOW_ALLOCS: Cell<u64> = const { Cell::new(0) };
		pub(crate) static SLOW_FREES: Cell<u64> = const { Cell::new(0) };
	}

	#[derive(Clone, Copy, Debug, PartialEq, Eq)]
	pub(crate) struct Counts {
		pub fast_allocs: u64,
		pub fast_frees: u64,
		pub slow_allocs: u64,
		pub slow_frees: u64,
	}

	/// One more on this thread's counter.
	pub(crate) fn bump(counter: &'static std::thread::LocalKey<Cell<u64>>) {
		// `try_with`: a value can be freed during thread teardown, after this
		// thread's locals have been destroyed. Counting is diagnostic, so
		// losing the increment there is correct behaviour, not a reason to
		// abort the free.
		let _ = counter.try_with(|c| c.set(c.get() + 1));
	}

	/// This thread's counters.
	pub(crate) fn snapshot() -> Counts {
		let read = |c: &'static std::thread::LocalKey<Cell<u64>>| c.with(|c| c.get());

		Counts {
			fast_allocs: read(&FAST_ALLOCS),
			fast_frees: read(&FAST_FREES),
			slow_allocs: read(&SLOW_ALLOCS),
			slow_frees: read(&SLOW_FREES),
		}
	}
}

#[cfg(test)]
mod tests {
	use super::*;
	use std::sync::{Mutex, MutexGuard};

	/// The key every test below stores. Its value is irrelevant to what these
	/// tests assert -- they are about the BYTES allocation and its tier -- but
	/// the header carries a key now, so one has to be supplied.
	const KEY: u64 = 0xC0FFEE;

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
			core::mem::size_of::<TieredValue<u64>>(),
			8,
			"a value is a tagged pointer and nothing else -- 8 bytes, not the \
			 8 + 32 a Shared<TieredBuffer> handle plus its Inner cost",
		);
		assert_eq!(core::mem::align_of::<TieredValue<u64>>(), 8, "it is a pointer");

		assert_eq!(
			core::mem::size_of::<Option<TieredValue<u64>>>(),
			8,
			"the NonNull must stay the niche: Option<Object> is what keeps a \
			 merged-store slot at 56 bytes, and it only does so if the value \
			 pointer costs nothing to make optional",
		);
	}

	#[test]
	fn fast_values_round_trip_their_bytes() {
		let _guard = routing_lock();

		let value = TieredValue::new_fast(KEY, b"hello", None);

		assert!(value.is_fast());
		assert!(!value.is_slow());
		assert_eq!(value.tier(), Tier::Fast);
		assert_eq!(value.bytes(), b"hello");

		drop(value);
	}

	#[test]
	fn slow_values_round_trip_their_bytes() {
		let _guard = routing_lock();

		let value = TieredValue::new_slow(KEY, b"world!", None);

		assert!(value.is_slow());
		assert!(!value.is_fast());
		assert_eq!(value.tier(), Tier::Slow);
		assert_eq!(value.bytes(), b"world!");

		drop(value);
	}

	/// Long enough to cross a page and a few size classes, so the round trip
	/// is not just testing the first cache line of a fresh allocation.
	#[test]
	fn values_round_trip_at_every_size_in_both_tiers() {
		let _guard = routing_lock();

		for len in [0usize, 1, 2, 3, 7, 8, 9, 15, 16, 17, 63, 64, 100, 128, 1023, 1024, 4096] {
			let bytes: Vec<u8> = (0..len).map(|i| (i % 251) as u8).collect();

			for tier in [Tier::Fast, Tier::Slow] {
				let value = TieredValue::new_in(KEY, &bytes, tier, None);

				assert_eq!(value.tier(), tier, "len {len}: wrong tier");
				assert_eq!(
					value.bytes(),
					&bytes[..],
					"len {len} in {tier:?}: bytes did not survive the copy",
				);

				drop(value);
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
				let value = TieredValue::new_in(KEY, &bytes, tier, None);

				assert_eq!(
					value.bytes().as_ptr().addr() % VALUE_ALIGN,
					0,
					"len {len} in {tier:?}: allocation is not 8-aligned, so the \
					 tier tag would corrupt the address",
				);
				assert!(!value.bytes().as_ptr().is_null(), "len {len} in {tier:?}: null value address");

				// The word is the address plus the tag, and NOTHING else:
				// bits 1 and 2 are reserved and must stay clear.
				let expected_tag = match tier {
					Tier::Fast => 0,
					Tier::Slow => SLOW_BIT,
				};

				assert_eq!(
					value.tagged_word(),
					value.bytes().as_ptr().addr() | expected_tag,
					"len {len} in {tier:?}: the word is not exactly address | tag",
				);
				assert_eq!(
					value.tagged_word() & 0b110,
					0,
					"len {len} in {tier:?}: bits 1-2 are reserved and must be clear",
				);

				drop(value);
			}
		}
	}

	/// A zero-length value is the case where an implementation is most likely
	/// to hand back a null, a dangling `align` sentinel, or a shared address.
	#[test]
	fn a_zero_length_value_is_still_a_tagged_non_null_pointer() {
		let _guard = routing_lock();

		let fast = TieredValue::new_fast(KEY, &[], None);
		let slow = TieredValue::new_slow(KEY, &[], None);

		// Cloned, not moved: `TieredValue` is an owning Arc handle now, and
		// the assertions below still need `fast` and `slow`. A clone is a
		// refcount bump onto the same allocation, so the addresses compared
		// afterwards are the same addresses this loop checked.
		for (value, tier) in [(fast.clone(), Tier::Fast), (slow.clone(), Tier::Slow)] {
			assert!(!value.bytes().as_ptr().is_null(), "{tier:?}: a zero-length value must not be null");
			assert_eq!(value.bytes().as_ptr().addr() % VALUE_ALIGN, 0, "{tier:?}: must stay 8-aligned");
			assert_eq!(value.tier(), tier, "{tier:?}: the tag must survive a zero length");
			assert_eq!(value.bytes(), b"", "{tier:?}: must read back empty");
		}

		// And they must be distinct allocations, because the address is the
		// identity the migration check compares.
		assert_ne!(
			fast.bytes().as_ptr(),
			slow.bytes().as_ptr(),
			"zero-length values must still get unique addresses -- that is why \
			 the layout is len.max(1) and not len",
		);

		let another_fast = TieredValue::new_fast(KEY, &[], None);
		assert_ne!(fast.bytes().as_ptr(), another_fast.bytes().as_ptr(), "two zero-length values must not alias");

		drop(fast);
		drop(slow);
		drop(another_fast);
	}

	/// `raw()` replaces `Shared::ptr_eq`, so it must be identity, not equality.
	#[test]
	fn raw_is_identity_not_content_equality() {
		let _guard = routing_lock();

		let a = TieredValue::new_fast(KEY, b"same bytes", None);
		let b = TieredValue::new_fast(KEY, b"same bytes", None);

		assert_eq!(a.bytes(), b.bytes(), "the bytes are equal");
		assert_ne!(a.bytes().as_ptr(), b.bytes().as_ptr(), "but they are not the same value");

		// A copy of a handle IS the same value -- that is what a migrating
		// consumer holds across the copy. It is a refcount bump now rather
		// than a bitwise copy, so it is also an assertion that cloning shares
		// rather than duplicates.
		let copy = a.clone();
		assert!(TieredValue::ptr_eq(&a, &copy), "a clone must name the same header");
		assert_eq!(a.bytes().as_ptr(), copy.bytes().as_ptr(), "and the same allocation");
		assert_eq!(a.strong_count(), 2);

		drop(a);
		drop(b);
	}

	/// `free` must pick its allocator from the TAG. The counters sit inside
	/// the four functions that name a concrete allocator, so a free that went
	/// to the wrong one moves the wrong counter and this fails.
	#[test]
	fn free_routes_to_the_allocator_its_tier_names() {
		let _guard = routing_lock();

		let before = route_counts::snapshot();
		let fast = TieredValue::new_fast(KEY, b"fast", None);
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

		drop(fast);
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
		let slow = TieredValue::new_slow(KEY, b"slow", None);
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

		drop(slow);
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

		let fast = TieredValue::new_fast(KEY, b"made here, freed there", None);
		let slow = TieredValue::new_slow(KEY, b"made here, freed there", None);
		let len = "made here, freed there".len() as u32;

		// The counters are read INSIDE the freeing thread -- they are
		// thread-local, and that thread is the one whose routing decision is
		// under test. A fresh thread starts at zero, so the deltas are the
		// absolute counts.
		let counted = std::thread::spawn(move || {
			let before = route_counts::snapshot();

			// SAFETY: both values were created just above, are `len` bytes
			// long, and no other copy of either exists.
		drop(slow);
		drop(fast);

			let after = route_counts::snapshot();

			(after.fast_frees - before.fast_frees, after.slow_frees - before.slow_frees)
		})
		.join()
		.expect("the freeing thread must not panic");

		assert_eq!(
			counted,
			(1, 1),
			"off-thread, each value must still be returned to the allocator its \
			 TAG names -- one fast free and one slow free, not two of either",
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

		let fast = TieredValue::new_fast(KEY, &bytes, None);
		let slow = TieredValue::new_slow(KEY, &bytes, None);

		let fast_node = node_of(fast.bytes().as_ptr());
		let slow_node = node_of(slow.bytes().as_ptr());

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

		// No manual free. `free` moved onto the private `ValueBytes` and is
		// called by `ValueHeader::drop`, so `fast` and `slow` going out of
		// scope here IS the teardown -- and freeing by hand as well would be
		// a double free. This block predates the Arc and did not survive it.
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
			values.push((TieredValue::new_fast(KEY, &bytes, None), Tier::Fast));
			values.push((TieredValue::new_slow(KEY, &bytes, None), Tier::Slow));
		}

		let mut seen = std::collections::HashSet::new();

		for (value, tier) in &values {
			assert_eq!(value.tier(), *tier, "a value forgot its tier");
			assert_eq!(value.bytes(), &bytes[..], "a value lost its bytes");
			assert!(seen.insert(value.bytes().as_ptr()), "two live values share an address");
		}

		for (value, _) in values {
			drop(value);
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

			let fast = TieredValue::new_fast(KEY, &bytes, None);
			let slow = TieredValue::new_slow(KEY, &bytes, None);

			let fast_arena = arena_of(fast.bytes().as_ptr())
				.expect("arenas.lookup must be available in this jemalloc build");
			let slow_arena = arena_of(slow.bytes().as_ptr())
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

		drop(fast);
		drop(slow);
		}
	}
}
