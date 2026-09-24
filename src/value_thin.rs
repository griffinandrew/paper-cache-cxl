/*
 * Copyright (c) Kia Shakiba
 *
 * This source code is licensed under the GNU AGPLv3 license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! A THIN DRAM HEADER IN FRONT OF A TIERED ITEM -- selected by `thin_header`.
//!
//! `TieredValue` -- a cached value as a refcounted DRAM word naming ONE
//! allocation that holds the length, the expiry, the key and the bytes, in
//! whichever tier the object is currently placed in.
//!
//! This is the only module in the crate allowed to contain unsafe value code.
//! Every allocation, every free, and every reinterpretation of a value pointer
//! lives here.
//!
//! ## The shape
//!
//! ```text
//!   Object          TieredValue = triomphe::Arc<ValueHeader<K>>   8 B handle
//!                     |
//!                     v
//!   DRAM, always    +--------------------------+
//!                   | strong count          8  |   triomphe's ArcInner
//!                   | item: tagged ptr      8  |   tier in bit 0
//!                   +--------------------------+   16 -> 16 B size class
//!                     |
//!                     v
//!   fast OR slow    +--------------------------+   ItemHeader<K>
//!   tier            | len: u32              4  |
//!                   | expiry: AtomicU32     4  |
//!                   | key: K                8  |   (a u64 key)
//!                   +--------------------------+   bytes_offset::<u64>() = 16
//!                   | the value's bytes ...    |
//!                   +--------------------------+   nallocx(16 + len)
//! ```
//!
//! The DRAM half keeps exactly the two things a read of a SLOW object must not
//! pay the interconnect for:
//!
//!   * the COUNT. A `get` bumps it under the shard guard and drops it after the
//!     copy. Both are atomic read-modify-writes, which neither pipeline nor
//!     prefetch, so on the far node they serialise where a streaming copy would
//!     not. This is the cost that sank `fused_value` (see `value.rs`, "THE
//!     READ PATH"), and the reason the count is not in the item here.
//!   * the TIER TAG, which every path reads first to route -- the free, the
//!     policy's tier query, the migration's "already there?" check.
//!
//! Everything else tiers WITH the bytes, so a demotion takes it out of DRAM:
//! the default layout leaves a 32-byte header behind for every slow object
//! (`value.rs`), this one leaves 16.
//!
//! ## What the item costs
//!
//! SIZE CLASSES. The item is `nallocx(bytes_offset::<K>() + len)` rather than
//! the default layout's `nallocx(len)`, and any prefix in front of a
//! class-aligned value costs the whole next class: `nallocx(4096)` is 4096,
//! `nallocx(16 + 4096)` is 5120. Per object, DRAM header plus item, against the
//! default layout's `32 + nallocx(len)`, for a `u64` key:
//!
//! ```text
//!   value      default    thin_header
//!      64      32 +   64   16 +    80   =    96 both
//!     100      32 +  112   16 +   128   =   144 both
//!    1000      32 + 1024   16 +  1024   =  1056 vs  1040
//!    1024      32 + 1024   16 +  1280   =  1056 vs  1296
//!    4096      32 + 4096   16 +  5120   =  4128 vs  5136
//!    8192      32 + 8192   16 + 10240   =  8224 vs 10256
//! ```
//!
//! The cliff is the price of co-locating ANY metadata with the bytes, and
//! `fused_value` pays it with a 24-byte prefix. Which way the total goes is a
//! property of the workload's value sizes, not of this module, so it is
//! measured rather than asserted here -- `object::overhead`'s
//! `an_object_costs_what_the_accounting_says_it_costs` holds the accounting to
//! whatever the allocator actually hands out.
//!
//! REMOTE METADATA. On a slow object the key, the length and the expiry are on
//! the far node, so the collision check, `is_expired` and `len` each read the
//! item's first cache line there -- the same line a hit's copy starts on, so a
//! hit pays it once, but a get that REJECTS a slow object (expired, or a hash
//! collision) now pays one remote line it did not pay before. So does the
//! last free, which reads `len` to rebuild the deallocation layout.
//!
//! ## The key
//!
//! The key is in the item either way, in one of two shapes:
//!
//!   * AS A `K`, the layout drawn above (`ItemHeader<K>`), for any key that is
//!     all of itself -- a `u64`, or any POD key. Its bytes tier with the value.
//!   * AS BYTES, for a `String`, `Vec<u8>` or `Box<[u8]>` key. Stored as a `K`,
//!     such a key would put only its 24- or 16-byte handle in the item and leave
//!     its characters in their own global -- DRAM -- allocation: the key would
//!     not tier, and every migration would allocate and copy it again. So the
//!     item holds the characters themselves (`BytesItemHeader`):
//!
//! ```text
//!   fast OR slow    +--------------------------+   BytesItemHeader
//!   tier            | len: u32              4  |
//!                   | expiry: AtomicU32     4  |
//!                   | key_len: u32          4  |
//!                   +--------------------------+
//!                   | the key's bytes ...      |
//!                   +--------------------------+   padded to VALUE_ALIGN
//!                   | the value's bytes ...    |
//!                   +--------------------------+   nallocx(align8(12 + key_len) + len)
//! ```
//!
//! A 42-byte key and a 100-byte value (cluster19's means) are one
//! `nallocx(56 + 100) = 160` item; stored as a `String` they were
//! `nallocx(32 + 100) = 160` plus the key's own `nallocx(42) = 48`, which
//! stayed in DRAM whatever the object's tier.
//!
//! Which shape a key gets is decided by its TYPE (`key_as_bytes`), and each
//! item records it in bit 1 of the header's tagged word, so the free -- which
//! runs from `Drop` and so cannot ask the key's type anything -- knows which
//! layout to rebuild. The map row keeps the key's 64-bit hash either way; the
//! item's key is what makes a collision on that hash harmless.
//!
//! `key_pmem_value_pmem`, which puts the key in its own persistent-memory box,
//! is refused at compile time in `lib.rs`: the key's placement is the whole
//! point of this layout.
//!
//! ## Shared with `value.rs`
//!
//! The refcount rather than epoch reclamation, `triomphe::Arc` rather than
//! `std::sync::Arc` (its count is 8 bytes with no `weak`, so the DRAM half is
//! 16 bytes rather than 24 -> 32), the tag bit, and routing the free on the
//! tag: the reasoning is in `value.rs` and holds unchanged. The one difference
//! in the tag is WHAT it points at -- the item, whose bytes are a fixed offset
//! in, rather than the bytes themselves.

use std::{
	alloc::Layout,
	any::{Any, TypeId},
	marker::PhantomData,
	ptr::NonNull,
	sync::atomic::{AtomicU32, AtomicU64, Ordering},
};

/// The refcount, and the only thing besides the tag that stays in DRAM.
use triomphe::Arc;

use crate::{Tier, object::ExpireTime};

/// Bit 0 of the item-pointer word: set means the slow tier, clear the fast one.
const SLOW_BIT: usize = 0b1;

/// Bit 1 of the item-pointer word: set means the item holds its key as BYTES
/// (`BytesItemHeader`), clear that it holds a `K` (`ItemHeader<K>`). See the
/// module documentation, "The key". Bit 2 stays reserved.
const KEY_BYTES_BIT: usize = 0b10;

/// Every flag the word may carry -- what `ValueHeader::item` strips.
const TAG_BITS: usize = SLOW_BIT | KEY_BYTES_BIT;

/// Alignment demanded of every item allocation, which is what keeps the low
/// three bits of the address available for tagging -- and what the bytes keep
/// at `bytes_offset`, since that offset is rounded to it.
const VALUE_ALIGN: usize = 8;

// ---------------------------------------------------------------------------
// the item: ONE tiered allocation holding the metadata and the bytes
// ---------------------------------------------------------------------------

/// Everything about a cached value except its count and its tier -- and in the
/// same allocation, at [`bytes_offset`], its bytes.
///
/// `#[repr(C)]` so the field order below is the field order the compiler uses,
/// which is what lets `bytes_offset` be a function of `size_of` alone, and
/// what keeps a `u64` key's header at exactly 16 bytes (asserted below).
#[repr(C)]
pub struct ItemHeader<K> {
	/// The value's length in bytes. The half that makes the tail slice and the
	/// deallocation layout well defined.
	len: u32,

	/// The expiry tick, or `0` for "never expires" -- the same encoding
	/// [`ExpireTime`]'s `Option<NonZeroU32>` niche uses.
	///
	/// Atomic because the item is SHARED and `PaperCache::ttl` sets a TTL on a
	/// live object. A four-byte store is the whole operation, and rebuilding
	/// the item instead would mean copying the value.
	expiry: AtomicU32,

	/// The real key, kept for the hash-collision check. The object map is keyed
	/// on a 64-bit hash, so this is what distinguishes two keys that collide.
	key: K,
	// The value's bytes follow, at `bytes_offset::<K>()`.
}

/// Where the value bytes start, measured from the head of the item.
///
/// Rounded up to `VALUE_ALIGN` so the bytes keep the eight-byte alignment they
/// have as their own allocation under the default layout.
///
/// Sized from [`ItemHeader`], the TIERED half -- never from [`ValueHeader`],
/// which is the DRAM half and is not in this allocation at all. The two are
/// deliberately distinct types, and the assertions below the handle pin both
/// sizes, so reaching for the wrong one fails the build.
///
/// `pub(crate)` because it is also the ACCOUNTING's business: for a key held
/// as a `K` the item is one allocation of `bytes_offset::<K>() + len`, which
/// `item_prefix_bytes` reports and `object::overhead::resident_item_bytes`
/// rounds.
#[inline]
pub(crate) const fn bytes_offset<K>() -> usize {
	let header = std::mem::size_of::<ItemHeader<K>>();

	(header + VALUE_ALIGN - 1) & !(VALUE_ALIGN - 1)
}

/// The layout of one whole item: metadata, padding, then `len` bytes.
///
/// Never zero-sized, because the metadata alone is eight bytes. That is what
/// gives two zero-length values distinct addresses without the default
/// layout's `len.max(1)`.
///
/// Aligned to the stricter of `VALUE_ALIGN` and the key's own alignment, so a
/// key type wider than eight bytes stays correctly aligned inside the item.
fn item_layout<K>(len: u32) -> Layout {
	let align = std::mem::align_of::<ItemHeader<K>>().max(VALUE_ALIGN);

	// `len` is a `u32` and the metadata is a handful of words, so on the
	// 64-bit targets this crate builds for the sum cannot overflow an
	// `isize`; the `expect` documents that rather than guarding a reachable
	// case.
	Layout::from_size_align(bytes_offset::<K>() + len as usize, align)
		.expect("a u32 length can always be laid out behind an item header")
		.pad_to_align()
}

// ---------------------------------------------------------------------------
// an item that holds its key as BYTES -- a `String`, `Vec<u8>` or `Box<[u8]>`
// ---------------------------------------------------------------------------

/// The two fields EVERY item starts with, whichever shape it holds its key in.
///
/// `ItemHeader<K>` and [`BytesItemHeader`] both begin with exactly these, in
/// this order, under `#[repr(C)]` (asserted below), so the length and the
/// expiry are read through this one type from either -- without claiming a
/// `K` exists in an item that holds bytes, which a `&ItemHeader<K>` would.
#[repr(C)]
struct ItemPrefix {
	len: u32,
	expiry: AtomicU32,
}

/// The metadata of an item that holds its key as bytes: an [`ItemPrefix`] and
/// the key's length. The key's bytes follow it, then padding to
/// `VALUE_ALIGN`, then the value's bytes at [`key_bytes_value_offset`].
#[repr(C)]
struct BytesItemHeader {
	len: u32,
	expiry: AtomicU32,
	key_len: u32,
}

/// Where a bytes-key item's value starts: after the 12-byte header and the
/// key, rounded up to `VALUE_ALIGN` as [`bytes_offset`] is.
#[inline]
const fn key_bytes_value_offset(key_len: usize) -> usize {
	(std::mem::size_of::<BytesItemHeader>() + key_len + VALUE_ALIGN - 1) & !(VALUE_ALIGN - 1)
}

/// The layout of one whole bytes-key item. The header's fields are all 4-byte,
/// so `VALUE_ALIGN` is the stricter alignment, as in [`item_layout`].
fn key_bytes_item_layout(key_len: usize, len: u32) -> Layout {
	Layout::from_size_align(key_bytes_value_offset(key_len) + len as usize, VALUE_ALIGN)
		.expect("a u32 key and a u32 length can always be laid out behind an item header")
		.pad_to_align()
}

const _: () = assert!(std::mem::offset_of!(ItemPrefix, len) == 0);
const _: () = assert!(std::mem::offset_of!(ItemPrefix, expiry) == 4);
const _: () = assert!(std::mem::offset_of!(ItemHeader<u64>, len) == 0);
const _: () = assert!(std::mem::offset_of!(ItemHeader<u64>, expiry) == 4);
const _: () = assert!(std::mem::offset_of!(ItemHeader<[u64; 8]>, expiry) == 4);
const _: () = assert!(std::mem::offset_of!(BytesItemHeader, len) == 0);
const _: () = assert!(std::mem::offset_of!(BytesItemHeader, expiry) == 4);
const _: () = assert!(std::mem::size_of::<BytesItemHeader>() == 12);

/// The key's bytes, if `K` is a type whose item holds its key as bytes: one
/// that owns a heap buffer the item would otherwise only point at. `None` for
/// every other key, which the item holds as a `K`.
///
/// Decided by the key's TYPE. Once `K` is monomorphised the `TypeId`
/// comparisons behind `downcast_ref` are constants, so this costs nothing at
/// run time.
fn key_as_bytes<K: 'static>(key: &K) -> Option<&[u8]> {
	let key = key as &dyn Any;

	if let Some(key) = key.downcast_ref::<String>() {
		return Some(key.as_bytes());
	}

	if let Some(key) = key.downcast_ref::<Vec<u8>>() {
		return Some(key);
	}

	key.downcast_ref::<Box<[u8]>>().map(|key| &**key)
}

/// Rebuilds an owned key from the bytes an item holds for it: the inverse of
/// [`key_as_bytes`], for the types it accepts.
///
/// # Panics
///
/// If `K` is not one of those types -- which cannot happen for an item that
/// set `KEY_BYTES_BIT`, since only [`key_as_bytes`] sets it.
fn key_from_bytes<K: 'static>(bytes: &[u8]) -> K {
	let id = TypeId::of::<K>();

	let key: Box<dyn Any> = if id == TypeId::of::<String>() {
		// They were a `String`'s `as_bytes` when the item was built.
		Box::new(String::from_utf8(bytes.to_vec()).expect("a String key's bytes are UTF-8"))
	} else if id == TypeId::of::<Vec<u8>>() {
		Box::new(bytes.to_vec())
	} else if id == TypeId::of::<Box<[u8]>>() {
		Box::new(Box::<[u8]>::from(bytes))
	} else {
		unreachable!("only the key types key_as_bytes accepts are held as bytes")
	};

	*key.downcast::<K>().expect("rebuilt as the key type it was asked for")
}

// ---------------------------------------------------------------------------
// the DRAM header: refcounted by `Arc`, owns the item
// ---------------------------------------------------------------------------

/// The DRAM half of a value: the tagged address of its item, and nothing else.
///
/// Allocated by `triomphe::Arc`, which puts its strong count in front of it,
/// so the whole DRAM allocation is `count 8 + word 8 = 16` bytes -- jemalloc's
/// 16-byte class exactly. Freed, and its item with it, when the last
/// [`TieredValue`] handle drops.
///
/// Not `Clone`: it OWNS the item the way a `Box` owns its pointee, and a copy
/// would free it twice. Sharing goes through the `Arc` around it.
pub struct ValueHeader<K> {
	/// The item's address, with the tier OR-ed into bit 0 and, for an item that
	/// holds its key as bytes, `KEY_BYTES_BIT` into bit 1. Never dereference
	/// directly -- go through [`ValueHeader::item_addr`].
	word: NonNull<u8>,

	/// This header owns an `ItemHeader<K>`, and through it a `K`: that is what
	/// drop check and the auto traits have to see.
	_item: PhantomData<ItemHeader<K>>,
}

// Ownership semantics are a `Box<ItemHeader<K>>`'s, so the bounds are too.
// Sending the header moves the key it owns to another thread, which needs
// `K: Send`; sharing it hands out `&K` to several threads, which needs
// `K: Sync`. `triomphe::Arc<T>` is `Send` and `Sync` only when `T` is both, so
// a handle crosses threads exactly when `K: Send + Sync` -- asserted, both
// ways and for each bound, by `the_handle_crosses_threads_only_when_its_key_can`.
// The `K: Send` bound is the one `Arc` cannot supply: the last handle can drop
// on any thread, and it drops the key there. The item's
// mutable state is one atomic, and the bytes are immutable for the life of the
// allocation (a "modification" allocates a new item).
unsafe impl<K: Send> Send for ValueHeader<K> {}
unsafe impl<K: Sync> Sync for ValueHeader<K> {}

impl<K> ValueHeader<K> {
	/// Which tier the ITEM lives in. The header itself is always DRAM.
	#[inline]
	fn tier(&self) -> Tier {
		if self.word.as_ptr().addr() & SLOW_BIT == 0 {
			Tier::Fast
		} else {
			Tier::Slow
		}
	}

	/// Whether the item holds its key as bytes (`BytesItemHeader`) rather than
	/// as a `K` (`ItemHeader<K>`). Read from the DRAM word, like the tier.
	#[inline]
	fn holds_key_bytes(&self) -> bool {
		self.word.as_ptr().addr() & KEY_BYTES_BIT != 0
	}

	/// The item's address, with the tag bits stripped.
	#[inline]
	fn item_addr(&self) -> *mut u8 {
		let untagged = self.word.as_ptr().map_addr(|addr| addr & !TAG_BITS);

		debug_assert_eq!(
			untagged.addr() % VALUE_ALIGN,
			0,
			"an item address must stay {VALUE_ALIGN}-aligned; bit 2 of the word \
			 is reserved and must never be set",
		);

		untagged
	}

	/// The item, as the `ItemHeader<K>` it is when it holds its key as a `K`.
	/// Only meaningful when [`ValueHeader::holds_key_bytes`] is false.
	#[inline]
	fn item(&self) -> *mut ItemHeader<K> {
		debug_assert!(!self.holds_key_bytes(), "a bytes-key item has no ItemHeader<K>");

		self.item_addr().cast::<ItemHeader<K>>()
	}
}

/// Frees the item -- the key's own destructor included -- back to the
/// allocator its tier names. The DRAM half is freed by `Arc`.
///
/// This is the single point every removal path funnels through -- a set
/// overwrite, an eviction, a TTL reap, a `wipe`, dropping the cache -- because
/// all of them drop an `Object`, which drops its handle. The refcount decides
/// when.
impl<K> Drop for ValueHeader<K> {
	fn drop(&mut self) {
		/// Returns the item's memory to its tier's allocator when dropped:
		/// after the key's destructor, or while unwinding out of it. The key
		/// lives INSIDE the allocation, so it has to be dropped before the
		/// free -- the reverse of `value.rs`, which frees the bytes and then
		/// drops a key that is not in them -- and without this guard a
		/// panicking `K::drop` would skip the free and leak the item.
		struct Free {
			ptr: *mut u8,
			layout: Layout,
			tier: Tier,
		}

		impl Drop for Free {
			fn drop(&mut self) {
				VALUE_FREES.fetch_add(1, Ordering::Relaxed);

				// SAFETY: `ptr` came from the allocator `tier` names, with
				// exactly `layout` -- see where the guard is built.
				unsafe {
					match self.tier {
						Tier::Fast => fast_dealloc(self.ptr, self.layout),
						Tier::Slow => slow_dealloc(self.ptr, self.layout),
					}
				}
			}
		}

		// An item holding its key as bytes has no key to drop -- the bytes are
		// plain data -- so its layout is all there is to rebuild.
		if self.holds_key_bytes() {
			let item = self.item_addr().cast::<BytesItemHeader>();

			// SAFETY: `Arc` runs this exactly once, when the last handle drops,
			// so nothing else can reach the item. `new_with_key_bytes` fully
			// initialised the `BytesItemHeader` before this header existed, and
			// the two lengths it wrote are the ones `key_bytes_item_layout` was
			// given for the allocation.
			let (key_len, len) = unsafe { ((*item).key_len, (*item).len) };

			drop(Free {
				ptr: item.cast::<u8>(),
				layout: key_bytes_item_layout(key_len as usize, len),
				tier: self.tier(),
			});

			return;
		}

		let item = self.item();

		// SAFETY: `Arc` runs this exactly once, when the last handle drops, so
		// nothing else can reach the item. `new_in` fully initialised the
		// `ItemHeader` before this header existed, and the `len` it wrote is
		// the one `item_layout` was given for the allocation, so the layout
		// rebuilt here is the allocation's. `len` is read BEFORE
		// `drop_in_place`, which leaves the header uninitialised.
		let free = Free {
			ptr: item.cast::<u8>(),
			layout: item_layout::<K>(unsafe { (*item).len }),
			tier: self.tier(),
		};

		// SAFETY: as above; the item is initialised and unreachable, and
		// `free` deallocates only once this has returned or unwound.
		unsafe { std::ptr::drop_in_place(item) };

		drop(free);
	}
}

// The sizes this layout exists for, pinned at compile time for the key the
// cache actually uses. The DRAM half is one word (triomphe adds the 8-byte
// count), the tiered metadata is 16 bytes, and the bytes start right after it.
const _: () = assert!(std::mem::size_of::<ValueHeader<u64>>() == 8);
const _: () = assert!(std::mem::size_of::<ItemHeader<u64>>() == 16);
const _: () = assert!(bytes_offset::<u64>() == 16);

// ---------------------------------------------------------------------------
// the handle
// ---------------------------------------------------------------------------

/// A cached value: an eight-byte handle onto a 16-byte DRAM header, which owns
/// one tiered item holding the metadata and the bytes. See the module
/// documentation.
///
/// Cloning is a refcount bump in DRAM, not a copy, and touches nothing on the
/// item's tier -- which is what lets a reader lift a slow value out from under
/// the shard lock without a remote atomic.
pub struct TieredValue<K> {
	inner: Arc<ValueHeader<K>>,
}

impl<K> Clone for TieredValue<K> {
	/// A refcount bump. SHALLOW, and correct: the bytes are immutable, so two
	/// handles onto one item observe the same value forever.
	#[inline]
	fn clone(&self) -> Self {
		TieredValue { inner: Arc::clone(&self.inner) }
	}
}

impl<K> TieredValue<K> {
	/// Builds a value: one item on `tier` holding the metadata, the key and a
	/// copy of `bytes`, and a DRAM header naming it.
	///
	/// A `String`, `Vec<u8>` or `Box<[u8]>` key goes into the item as its
	/// bytes, and its own buffer is freed here; any other key goes in as a `K`.
	/// See the module documentation, "The key".
	///
	/// # Panics
	///
	/// If `bytes.len()` does not fit a `u32`. The length is stored as a `u32`
	/// and is what the deallocation layout is rebuilt from, so a length that
	/// cannot round-trip has to be refused HERE, where it is still a panic,
	/// rather than silently truncated into a mismatched free.
	pub fn new_in(key: K, bytes: &[u8], tier: Tier, expiry: ExpireTime) -> Self
	where
		K: 'static,
	{
		if let Some(key_bytes) = key_as_bytes(&key) {
			return Self::new_with_key_bytes(key_bytes, bytes, tier, expiry);
		}

		Self::new_with_key_value(key, bytes, tier, expiry)
	}

	/// Builds an item that holds its key as BYTES (`BytesItemHeader`): the
	/// key's bytes and the value's, in one allocation on `tier`.
	///
	/// # Panics
	///
	/// If either length does not fit a `u32`, for the reason [`TieredValue::new_in`]
	/// gives.
	fn new_with_key_bytes(key: &[u8], bytes: &[u8], tier: Tier, expiry: ExpireTime) -> Self {
		assert!(
			u32::try_from(bytes.len()).is_ok(),
			"a cached value must fit a u32 length; got {} bytes",
			bytes.len(),
		);
		assert!(
			u32::try_from(key.len()).is_ok(),
			"a cached key must fit a u32 length; got {} bytes",
			key.len(),
		);

		let len = bytes.len() as u32;
		let layout = key_bytes_item_layout(key.len(), len);

		// SAFETY: the layout is never zero-sized -- the header alone is twelve
		// bytes -- which is the only precondition either allocator entry point
		// has.
		let raw = unsafe {
			match tier {
				Tier::Fast => fast_alloc(layout),
				Tier::Slow => slow_alloc(layout),
			}
		};

		let Some(ptr) = NonNull::new(raw) else {
			std::alloc::handle_alloc_error(layout)
		};

		debug_assert_eq!(
			ptr.as_ptr().addr() % VALUE_ALIGN,
			0,
			"the allocator returned an address that is not {VALUE_ALIGN}-aligned, \
			 so bits 0-1 are not free for the tags",
		);

		// SAFETY: `ptr` names `layout.size()` writable bytes, 8-aligned, which
		// is at least `key_bytes_value_offset(key.len()) + len`: the header,
		// then the key at 12, then the value at the key's rounded end. A fresh
		// allocation cannot overlap either of the caller's slices. All three
		// writes finish before any header, and so any handle, exists.
		unsafe {
			ptr.as_ptr().cast::<BytesItemHeader>().write(BytesItemHeader {
				len,
				expiry: AtomicU32::new(expiry.map_or(0, |tick| tick.get())),
				key_len: key.len() as u32,
			});

			std::ptr::copy_nonoverlapping(
				key.as_ptr(),
				ptr.as_ptr().add(std::mem::size_of::<BytesItemHeader>()),
				key.len(),
			);

			std::ptr::copy_nonoverlapping(
				bytes.as_ptr(),
				ptr.as_ptr().add(key_bytes_value_offset(key.len())),
				len as usize,
			);
		}

		TieredValue {
			inner: Arc::new(ValueHeader {
				word: tag_key_bytes(tag(ptr, tier)),
				_item: PhantomData,
			}),
		}
	}

	/// Builds an item that holds its key as a `K` (`ItemHeader<K>`).
	fn new_with_key_value(key: K, bytes: &[u8], tier: Tier, expiry: ExpireTime) -> Self {
		assert!(
			u32::try_from(bytes.len()).is_ok(),
			"a cached value must fit a u32 length; got {} bytes",
			bytes.len(),
		);

		let len = bytes.len() as u32;
		let layout = item_layout::<K>(len);

		// SAFETY: `item_layout` is never zero-sized -- the metadata alone is
		// eight bytes -- which is the only precondition either allocator entry
		// point has.
		let raw = unsafe {
			match tier {
				Tier::Fast => fast_alloc(layout),
				Tier::Slow => slow_alloc(layout),
			}
		};

		let Some(ptr) = NonNull::new(raw) else {
			std::alloc::handle_alloc_error(layout)
		};

		debug_assert_eq!(
			ptr.as_ptr().addr() % VALUE_ALIGN,
			0,
			"the allocator returned an address that is not {VALUE_ALIGN}-aligned, \
			 so bit 0 is not free for the tier tag",
		);

		// SAFETY: `ptr` names `layout.size()` writable bytes aligned for
		// `ItemHeader<K>`, and that size is at least `bytes_offset::<K>() +
		// len`. A fresh allocation cannot overlap the caller's slice. Both
		// writes finish before any header, and so any handle, exists.
		unsafe {
			ptr.as_ptr().cast::<ItemHeader<K>>().write(ItemHeader {
				len,
				expiry: AtomicU32::new(expiry.map_or(0, |tick| tick.get())),
				key,
			});

			std::ptr::copy_nonoverlapping(
				bytes.as_ptr(),
				ptr.as_ptr().add(bytes_offset::<K>()),
				len as usize,
			);
		}

		TieredValue {
			inner: Arc::new(ValueHeader { word: tag(ptr, tier), _item: PhantomData }),
		}
	}

	/// Builds a value in the fast (DRAM) tier.
	#[inline]
	pub fn new_fast(key: K, bytes: &[u8], expiry: ExpireTime) -> Self
	where
		K: 'static,
	{
		Self::new_in(key, bytes, Tier::Fast, expiry)
	}

	/// Builds a value in the slow (PMEM/CXL) tier.
	#[inline]
	pub fn new_slow(key: K, bytes: &[u8], expiry: ExpireTime) -> Self
	where
		K: 'static,
	{
		Self::new_in(key, bytes, Tier::Slow, expiry)
	}

	/// The item's metadata as an `ItemHeader<K>`, borrowed for as long as this
	/// handle lives. Only for an item that holds its key as a `K`.
	#[inline]
	fn item(&self) -> &ItemHeader<K> {
		// SAFETY: `self` holds a strong reference, so the header and the item
		// it owns are alive, and `new_with_key_value` initialised the whole
		// `ItemHeader<K>` before any handle existed -- `ValueHeader::item`
		// asserts this is such an item. Only the atomic `expiry` is ever
		// written afterwards.
		unsafe { &*self.inner.item() }
	}

	/// The length and expiry every item starts with, whichever shape it holds
	/// its key in.
	#[inline]
	fn prefix(&self) -> &ItemPrefix {
		// SAFETY: `self` holds a strong reference, so the item is alive, and
		// both item shapes begin with an initialised `ItemPrefix` at offset 0
		// (asserted beside `ItemPrefix`). Only the atomic `expiry` is ever
		// written after construction.
		unsafe { &*self.inner.item_addr().cast::<ItemPrefix>() }
	}

	/// The same value's bytes, re-copied into `tier`, carrying the key and the
	/// CURRENT expiry across.
	///
	/// This is a physical tier migration: a fresh item behind a fresh header,
	/// which the caller then swaps in under the shard guard after checking
	/// [`TieredValue::ptr_eq`] against the handle it snapshotted. Building it
	/// OUTSIDE the guard is the point -- the copy is the expensive part and may
	/// be a CXL write.
	///
	/// A key held as bytes is copied as bytes: no `K` is rebuilt, and nothing
	/// is allocated but the new item and its header.
	pub fn migrated_to(&self, tier: Tier) -> Self
	where
		K: Clone,
	{
		if let Some(key) = self.key_bytes() {
			return Self::new_with_key_bytes(key, self.bytes(), tier, self.expiry());
		}

		Self::new_with_key_value(self.key().clone(), self.bytes(), tier, self.expiry())
	}

	/// The real key, for an item that holds it as a `K`. On the item's tier.
	///
	/// An item that holds its key as bytes (a `String`, `Vec<u8>` or
	/// `Box<[u8]>` key) has no `K` in it to lend: compare with
	/// [`TieredValue::key_matches`], rebuild with [`TieredValue::key_owned`].
	///
	/// # Panics
	///
	/// If this item holds its key as bytes.
	#[inline]
	#[track_caller]
	pub fn key(&self) -> &K {
		assert!(
			!self.inner.holds_key_bytes(),
			"this item holds its key as bytes, so there is no K to borrow; \
			 use key_matches or key_owned",
		);

		&self.item().key
	}

	/// The key's bytes, for an item that holds its key as bytes; `None` for
	/// one that holds a `K`. On the item's tier.
	#[inline]
	pub(crate) fn key_bytes(&self) -> Option<&[u8]> {
		if !self.inner.holds_key_bytes() {
			return None;
		}

		let item = self.inner.item_addr();

		// SAFETY: `self` holds a strong reference, so the item is alive, and
		// `new_with_key_bytes` wrote a `BytesItemHeader` and exactly `key_len`
		// key bytes right behind it before any handle existed; neither is ever
		// written again. The pointer is derived from the item's raw address, so
		// it carries the whole allocation's provenance.
		unsafe {
			let key_len = (*item.cast::<BytesItemHeader>()).key_len as usize;

			Some(std::slice::from_raw_parts(
				item.add(std::mem::size_of::<BytesItemHeader>()),
				key_len,
			))
		}
	}

	/// The key, owned: rebuilt from its bytes for an item that holds it as
	/// bytes, cloned for one that holds a `K`.
	pub fn key_owned(&self) -> K
	where
		K: Clone + 'static,
	{
		match self.key_bytes() {
			Some(bytes) => key_from_bytes(bytes),
			None => self.key().clone(),
		}
	}

	/// Whether this value's key is `key`. The check that makes a 64-bit hash
	/// collision harmless.
	#[inline]
	pub fn key_matches(&self, key: &K) -> bool
	where
		K: Eq + 'static,
	{
		match self.key_bytes() {
			// Only a byte-string key type builds an item that holds bytes, so
			// `key` is one too, and equal bytes are equal keys.
			Some(stored) => key_as_bytes(key) == Some(stored),
			None => self.key().eq(key),
		}
	}

	/// The key's byte cost as `object::overhead::base_size` counts it.
	///
	/// For a key held as a `K`, what `TypeSize` says, as under every layout.
	/// For a key held as bytes, the bytes are INSIDE the item, which
	/// `resident_item_bytes` rounds together with the value; `base_size` then
	/// adds `key_size + expiry` on top, as it does for every layout, and
	/// `get_policy_overhead` takes `DOUBLE_COUNTED_IN_BASE_SIZE` -- a
	/// `HashedKey` and an expiry -- back off. Charging a `HashedKey` here makes
	/// that pair cancel exactly, so the key is counted once, in the item, and
	/// `size - dram_resident` is the item: what a migration moves.
	pub fn key_accounted_size(&self) -> usize
	where
		K: typesize::TypeSize,
	{
		if self.inner.holds_key_bytes() {
			return std::mem::size_of::<crate::HashedKey>();
		}

		self.key().get_size()
	}

	/// The item's bytes in front of the value -- its metadata, its key when the
	/// key is held as bytes, and padding -- which is what the item costs
	/// beyond the value. [`bytes_offset`] for a key held as a `K`.
	#[inline]
	pub fn item_prefix_bytes(&self) -> usize {
		match self.key_bytes() {
			Some(key) => key_bytes_value_offset(key.len()),
			None => bytes_offset::<K>(),
		}
	}

	/// The value's bytes.
	///
	/// Safe, because the item owns both the length and the tail and this
	/// borrow keeps the item alive.
	#[inline]
	pub fn bytes(&self) -> &[u8] {
		let len = self.prefix().len;

		// SAFETY: the tail was written with exactly `len` bytes when the item
		// was built, at `item_prefix_bytes()` -- `bytes_offset::<K>()` or
		// `key_bytes_value_offset(key_len)`, whichever shape it is -- and is
		// never written again, in the same allocation as the metadata this
		// handle keeps alive. The pointer is derived from the item's raw
		// address, so it carries the whole allocation's provenance. It is
		// non-null and `VALUE_ALIGN`-aligned even when `len` is 0, which is
		// what `from_raw_parts` requires then.
		unsafe {
			std::slice::from_raw_parts(
				self.inner.item_addr().add(self.item_prefix_bytes()),
				len as usize,
			)
		}
	}

	/// The value's length in bytes. On the item's tier.
	#[inline]
	pub fn len(&self) -> u32 {
		self.prefix().len
	}

	#[inline]
	pub fn is_empty(&self) -> bool {
		self.len() == 0
	}

	/// Which tier the item -- metadata, key and bytes -- lives in. Read from
	/// the DRAM header, so it costs no remote access.
	#[inline]
	pub fn tier(&self) -> Tier {
		self.inner.tier()
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
		std::num::NonZeroU32::new(self.prefix().expiry.load(Ordering::Relaxed))
	}

	/// Sets the expiry. Visible to every handle onto this item, which is
	/// correct: they are the same object.
	#[inline]
	pub fn set_expiry(&self, expiry: ExpireTime) {
		self.prefix()
			.expiry
			.store(expiry.map_or(0, |tick| tick.get()), Ordering::Relaxed);
	}

	/// Whether the two handles name the SAME header allocation.
	///
	/// This is the migration identity check, and it is exact rather than
	/// merely likely: the caller holds a strong reference to the handle it
	/// snapshotted, so that header cannot be freed and its address cannot be
	/// recycled into a different value. A migration builds a new header as
	/// well as a new item, so comparing headers is comparing values. Never
	/// compare the bytes -- equal content in two allocations is NOT the same
	/// value, and treating it as such would let a migration overwrite a
	/// concurrent `set`.
	#[inline]
	pub fn ptr_eq(a: &Self, b: &Self) -> bool {
		Arc::ptr_eq(&a.inner, &b.inner)
	}

	/// The DRAM header's address, as an opaque identity for logging and tests.
	#[inline]
	pub fn raw(&self) -> *const ValueHeader<K> {
		Arc::as_ptr(&self.inner)
	}

	/// The raw tagged word naming this value's ITEM -- address with the tier
	/// in bit 0 and `KEY_BYTES_BIT` in bit 1. Test-only: the tag discipline is
	/// asserted against it, and nothing in the release path should ever need
	/// the tagged form.
	#[cfg(test)]
	pub(crate) fn tagged_word(&self) -> usize {
		self.inner.word.as_ptr().addr()
	}

	/// The item's untagged address. Test-only, for placement assertions.
	#[cfg(test)]
	fn item_ptr(&self) -> *const ItemHeader<K> {
		self.inner.item_addr().cast::<ItemHeader<K>>()
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
			.field("item", &self.inner.item_addr())
			.field("key_as_bytes", &self.inner.holds_key_bytes())
			.finish()
	}
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

/// Folds `KEY_BYTES_BIT` into an item word: the item holds its key as bytes.
#[inline]
fn tag_key_bytes(word: NonNull<u8>) -> NonNull<u8> {
	// SAFETY: setting a bit cannot turn a non-null address into a null one,
	// so the result is still a valid `NonNull`.
	unsafe { NonNull::new_unchecked(word.as_ptr().map_addr(|addr| addr | KEY_BYTES_BIT)) }
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
	use std::{num::NonZeroU32, sync::MutexGuard};

	/// The key most tests below store. Its value is irrelevant to what they
	/// assert, but the item carries a key, so one has to be supplied.
	const KEY: u64 = 0xC0FFEE;

	/// EVERY test below that allocates a `TieredValue` holds this. The crate-
	/// wide lock, not a private one: these tests race the `VALUE_FREES` and
	/// `PENDING_DEMOTE` delta tests in other modules, and a lock only they took
	/// would not serialise against those. See `value.rs` for the intermittent
	/// failure that made it necessary.
	fn routing_lock() -> MutexGuard<'static, ()> {
		crate::global_counter_lock()
	}

	fn other(tier: Tier) -> Tier {
		match tier {
			Tier::Fast => Tier::Slow,
			Tier::Slow => Tier::Fast,
		}
	}

	/// The handle stays one word, and `Option` of it stays free.
	#[test]
	fn a_value_is_one_word_and_the_option_is_free() {
		assert_eq!(core::mem::size_of::<TieredValue<u64>>(), 8, "a value is one Arc pointer");
		assert_eq!(core::mem::align_of::<TieredValue<u64>>(), 8, "it is a pointer");

		assert_eq!(
			core::mem::size_of::<Option<TieredValue<u64>>>(),
			8,
			"the Arc's non-null pointer must stay the niche: Option<Object> is \
			 what keeps a merged-store slot at its size",
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
		assert_eq!(*value.key(), KEY);
	}

	#[test]
	fn slow_values_round_trip_their_bytes() {
		let _guard = routing_lock();

		let value = TieredValue::new_slow(KEY, b"world!", None);

		assert!(value.is_slow());
		assert!(!value.is_fast());
		assert_eq!(value.tier(), Tier::Slow);
		assert_eq!(value.bytes(), b"world!");
		assert_eq!(*value.key(), KEY);
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
				assert_eq!(value.len() as usize, len, "len {len} in {tier:?}: wrong length");
				assert_eq!(
					value.bytes(),
					&bytes[..],
					"len {len} in {tier:?}: bytes did not survive the copy",
				);
			}
		}
	}

	/// The key, the length and the expiry all live on the item's tier now, so
	/// each hop rebuilds all three there. Lengths include the two that fill a
	/// size class exactly once the 16-byte header is added (1008 -> 1024,
	/// 4080 -> 4096), where the tail ends on the allocation's last byte, and
	/// the powers of two that spill into the next class.
	#[test]
	fn key_len_and_expiry_survive_fast_slow_fast() {
		let _guard = routing_lock();

		let expiry = NonZeroU32::new(0x00C0_FFEE);

		for len in [0usize, 1, 63, 1008, 4080, 4096, 8192] {
			let bytes: Vec<u8> = (0..len).map(|i| (i * 7 % 251) as u8).collect();
			let key = 0xDEAD_0000 + len as u64;

			let fast = TieredValue::new_fast(key, &bytes, expiry);
			let slow = fast.migrated_to(Tier::Slow);
			let back = slow.migrated_to(Tier::Fast);

			for (hop, value, tier) in [
				("built", &fast, Tier::Fast),
				("demoted", &slow, Tier::Slow),
				("promoted", &back, Tier::Fast),
			] {
				assert_eq!(value.tier(), tier, "len {len}, {hop}: wrong tier");
				assert_eq!(*value.key(), key, "len {len}, {hop}: the key did not travel");
				assert!(value.key_matches(&key), "len {len}, {hop}: key_matches disagrees with key()");
				assert_eq!(value.len() as usize, len, "len {len}, {hop}: the length did not travel");
				assert_eq!(value.expiry(), expiry, "len {len}, {hop}: the expiry did not travel");
				assert_eq!(value.bytes(), &bytes[..], "len {len}, {hop}: the bytes did not travel");
			}

			assert!(!TieredValue::ptr_eq(&fast, &slow), "len {len}: a migration builds a new header");
			assert!(!TieredValue::ptr_eq(&slow, &back), "len {len}: and so does the way back");

			// Each item owns its own expiry: clearing it on one copy is not
			// visible through another, and IS visible through a clone.
			let alias = slow.clone();
			slow.set_expiry(None);
			assert_eq!(alias.expiry(), None, "len {len}: a clone shares the item");
			assert_eq!(fast.expiry(), expiry, "len {len}: a migrated copy does not");
		}
	}

	/// The item owns its key, so the key's destructor runs once per ITEM --
	/// when the last handle onto that item drops -- and never for a handle
	/// that is not the last. A `drop_in_place` missing from `Drop` leaks every
	/// key's own heap allocation; one run per handle would double-free it.
	#[test]
	fn a_key_with_drop_glue_is_dropped_exactly_once_per_item() {
		use std::sync::atomic::AtomicUsize;

		static DROPS: AtomicUsize = AtomicUsize::new(0);

		#[derive(Debug, PartialEq, Eq)]
		struct Counted(u64);

		impl Clone for Counted {
			fn clone(&self) -> Self {
				Counted(self.0)
			}
		}

		impl Drop for Counted {
			fn drop(&mut self) {
				DROPS.fetch_add(1, Ordering::Relaxed);
			}
		}

		let _guard = routing_lock();
		let drops = || DROPS.load(Ordering::Relaxed);
		let start = drops();

		for tier in [Tier::Fast, Tier::Slow] {
			let base = drops();
			let value = TieredValue::new_in(Counted(7), b"payload", tier, None);

			drop(value.clone());
			assert_eq!(drops(), base, "{tier:?}: dropping a handle that is not the last dropped the key");

			let moved = value.migrated_to(other(tier));
			assert_eq!(drops(), base, "{tier:?}: a migration clones the key, it drops nothing");

			drop(value);
			assert_eq!(drops(), base + 1, "{tier:?}: the last handle drops its item's key exactly once");
			// `.0`, not `== &Counted(7)`: a temporary `Counted` would count its own drop.
			assert_eq!(moved.key().0, 7, "{tier:?}: the copy owns its own key");

			drop(moved);
			assert_eq!(drops(), base + 2, "{tier:?}: and the copy's, exactly once");
		}

		assert_eq!(drops() - start, 4);

		// A heap-owning key round-trips through both tiers intact. A `String`
		// is held as its bytes, not as a `K` -- see
		// `a_string_key_is_held_as_bytes_inside_the_item` -- so there is no
		// destructor of its own left in the item to count; what is left to
		// check is that it survives the hops.
		let key = String::from("a key long enough to own a heap allocation");
		let fast = TieredValue::new_fast(key.clone(), b"v", None);
		let slow = fast.migrated_to(Tier::Slow);
		drop(fast);
		let back = slow.migrated_to(Tier::Fast);
		drop(slow);

		assert_eq!(back.key_owned(), key);
		assert!(back.key_matches(&key));
	}

	/// A key wider than the metadata, or narrower than nothing, must still
	/// land aligned inside the item, and the bytes must still start after it.
	/// `u64` is the only key the cache instantiates today, so these are the
	/// layouts nothing else exercises.
	#[test]
	fn keys_of_any_alignment_keep_the_item_well_formed() {
		#[repr(align(16))]
		#[derive(Clone, Debug, PartialEq, Eq)]
		struct Wide(u128);

		#[repr(align(64))]
		#[derive(Clone, Debug, PartialEq, Eq)]
		struct Line([u8; 64]);

		#[derive(Clone, Debug, PartialEq, Eq)]
		struct Unit;

		assert_eq!(bytes_offset::<Unit>(), 8, "len + expiry, and a key that takes no room");
		assert_eq!(bytes_offset::<Wide>(), 32, "8 of metadata, padded to the key's 16");
		assert_eq!(bytes_offset::<Line>(), 128, "8 of metadata, padded to the key's 64");

		fn check<K: Clone + Eq + std::fmt::Debug + 'static>(key: K) {
			let name = std::any::type_name::<K>();
			let expiry = NonZeroU32::new(9);

			// The alignment must be REQUESTED: jemalloc hands a 16-aligned
			// address to every class a `Wide` item lands in, so an item layout
			// that dropped the key's alignment would still receive it, and the
			// placement checks below would pass by luck. Asserted on the layout
			// itself, as the u64 case is in
			// `the_item_layout_always_demands_eight_byte_alignment`.
			let demanded = std::mem::align_of::<K>().max(VALUE_ALIGN);

			for len in [0u32, 1, 100, 4096] {
				assert_eq!(
					item_layout::<K>(len).align(),
					demanded,
					"{name}, len {len}: the item layout must demand the key's alignment",
				);
			}

			for len in [0usize, 1, 100, 4096] {
				let bytes = vec![0x3Cu8; len];

				for tier in [Tier::Fast, Tier::Slow] {
					let built = TieredValue::new_in(key.clone(), &bytes, tier, expiry);
					let moved = built.migrated_to(other(tier));

					for value in [&built, &moved] {
						let item = value.item_ptr().addr();
						let at = format!("{name}, len {len}, {:?}", value.tier());

						assert_eq!(item % demanded, 0, "{at}: item misaligned");
						assert_eq!(
							(value.key() as *const K).addr() % std::mem::align_of::<K>(),
							0,
							"{at}: the key is not aligned for its type",
						);
						assert_eq!(
							value.bytes().as_ptr().addr(),
							item + bytes_offset::<K>(),
							"{at}: the bytes are not at bytes_offset",
						);
						assert!(
							bytes_offset::<K>() >= std::mem::size_of::<ItemHeader<K>>(),
							"{at}: the bytes would overlap the key",
						);
						assert_eq!(value.bytes().as_ptr().addr() % VALUE_ALIGN, 0, "{at}: bytes misaligned");
						assert_eq!(value.tagged_word() & 0b110, 0, "{at}: bits 1-2 are reserved");
						assert_eq!(value.key(), &key, "{at}: key");
						assert_eq!(value.len() as usize, len, "{at}: len");
						assert_eq!(value.expiry(), expiry, "{at}: expiry");
						assert_eq!(value.bytes(), &bytes[..], "{at}: bytes");
					}
				}
			}
		}

		let _guard = routing_lock();

		check(Unit);
		check(Wide(0x0123_4567_89AB_CDEF_0011_2233_4455_6677));
		check(Line([0x5Du8; 64]));
	}

	/// `Send` and `Sync` are hand-written for the header, so they are checked
	/// here in both directions, at compile time. The negative half uses the
	/// ambiguity trick: `some_item` resolves only if exactly one impl of
	/// `AmbiguousIf*` applies, which is the case only when the bound is NOT
	/// met, so a handle that gains a bound it should not have stops this
	/// module compiling.
	///
	/// One key per way a bound can be lost, because `Arc` supplies half of
	/// them itself: `Arc<T>: Send` needs `T: Send + Sync`, so a lost `Sync`
	/// bound is exposed by a key that is `Send` only (`Cell`), and a lost
	/// `Send` bound only by a key that is `Sync` only (`MutexGuard`) -- a
	/// `Rc` key, being neither, is still refused by the bound that remains.
	#[test]
	fn the_handle_crosses_threads_only_when_its_key_can() {
		fn send_and_sync<T: Send + Sync>() {}

		send_and_sync::<TieredValue<u64>>();
		send_and_sync::<TieredValue<String>>();

		trait AmbiguousIfSend<A> {
			fn some_item() {}
		}
		impl<T: ?Sized> AmbiguousIfSend<()> for T {}
		impl<T: ?Sized + Send> AmbiguousIfSend<u8> for T {}

		trait AmbiguousIfSync<A> {
			fn some_item() {}
		}
		impl<T: ?Sized> AmbiguousIfSync<()> for T {}
		impl<T: ?Sized + Sync> AmbiguousIfSync<u8> for T {}

		// Neither Send nor Sync.
		let _ = <TieredValue<std::rc::Rc<u8>> as AmbiguousIfSend<_>>::some_item;
		let _ = <TieredValue<std::rc::Rc<u8>> as AmbiguousIfSync<_>>::some_item;

		// Send but not Sync: two handles on two threads would both read the
		// same `&Cell`, so the handle may cross neither way.
		let _ = <TieredValue<std::cell::Cell<u64>> as AmbiguousIfSend<_>>::some_item;
		let _ = <TieredValue<std::cell::Cell<u64>> as AmbiguousIfSync<_>>::some_item;

		// Sync but not Send: the last handle may drop on any thread, and a
		// guard dropped off its locking thread unlocks a mutex it does not own.
		let _ = <TieredValue<std::sync::MutexGuard<'static, u8>> as AmbiguousIfSend<_>>::some_item;
		let _ = <TieredValue<std::sync::MutexGuard<'static, u8>> as AmbiguousIfSync<_>>::some_item;
	}

	/// A key whose destructor panics must not take the item with it: the
	/// memory still goes back to its tier's allocator while the panic unwinds.
	/// The key lives inside the allocation, so it is dropped BEFORE the free,
	/// and without a guard an unwinding `K::drop` skips the free.
	#[test]
	fn a_panicking_key_destructor_still_frees_the_item() {
		struct Bomb;

		impl Drop for Bomb {
			fn drop(&mut self) {
				panic!("a key destructor that unwinds (expected by this test)");
			}
		}

		let _guard = routing_lock();

		for tier in [Tier::Fast, Tier::Slow] {
			let value = TieredValue::new_in(Bomb, b"payload", tier, None);
			let before = route_counts::snapshot();

			let unwound = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| drop(value)));
			assert!(unwound.is_err(), "{tier:?}: the key's destructor should have panicked");

			let after = route_counts::snapshot();
			let frees = match tier {
				Tier::Fast => (after.fast_frees - before.fast_frees, after.slow_frees - before.slow_frees),
				Tier::Slow => (after.slow_frees - before.slow_frees, after.fast_frees - before.fast_frees),
			};

			assert_eq!(frees, (1, 0), "{tier:?}: the item must be freed, to its own tier, despite the panic");
		}
	}

	/// The tag is only free because the item address never uses the low bits,
	/// and it rides on the ITEM, whose bytes are a fixed offset in.
	#[test]
	fn values_are_eight_aligned_so_the_tag_bit_is_free() {
		let _guard = routing_lock();

		for len in [0usize, 1, 2, 3, 5, 7, 8, 9, 12, 13, 16, 17, 31, 100, 1000, 4097] {
			let bytes = vec![0xA5u8; len];

			for tier in [Tier::Fast, Tier::Slow] {
				let value = TieredValue::new_in(KEY, &bytes, tier, None);
				let item = value.item_ptr().addr();

				assert_eq!(item % VALUE_ALIGN, 0, "len {len} in {tier:?}: the item is not 8-aligned");
				assert_eq!(
					value.bytes().as_ptr().addr() % VALUE_ALIGN,
					0,
					"len {len} in {tier:?}: the bytes are not 8-aligned",
				);

				let expected_tag = match tier {
					Tier::Fast => 0,
					Tier::Slow => SLOW_BIT,
				};

				assert_eq!(
					value.tagged_word(),
					item | expected_tag,
					"len {len} in {tier:?}: the word is not exactly item address | tag",
				);
				assert_eq!(
					value.bytes().as_ptr().addr(),
					item + bytes_offset::<u64>(),
					"len {len} in {tier:?}: the bytes must sit at a FIXED offset into \
					 the item, or `bytes()` and the allocation disagree",
				);
				assert_eq!(
					value.tagged_word() & 0b110,
					0,
					"len {len} in {tier:?}: bits 1-2 are reserved and must be clear",
				);
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

		for (value, tier) in [(fast.clone(), Tier::Fast), (slow.clone(), Tier::Slow)] {
			assert_eq!(value.bytes().as_ptr().addr() % VALUE_ALIGN, 0, "{tier:?}: must stay 8-aligned");
			assert_eq!(value.tier(), tier, "{tier:?}: the tag must survive a zero length");
			assert_eq!(value.bytes(), b"", "{tier:?}: must read back empty");
			assert_eq!(*value.key(), KEY, "{tier:?}: the key is still there");
		}

		assert_ne!(
			fast.item_ptr(),
			slow.item_ptr(),
			"zero-length values must still get unique items -- an item is never \
			 zero-sized, because the metadata is in it",
		);

		let another_fast = TieredValue::new_fast(KEY, &[], None);
		assert_ne!(fast.item_ptr(), another_fast.item_ptr(), "two zero-length values must not alias");
	}

	/// `ptr_eq` is identity, not equality.
	#[test]
	fn raw_is_identity_not_content_equality() {
		let _guard = routing_lock();

		let a = TieredValue::new_fast(KEY, b"same bytes", None);
		let b = TieredValue::new_fast(KEY, b"same bytes", None);

		assert_eq!(a.bytes(), b.bytes(), "the bytes are equal");
		assert!(!TieredValue::ptr_eq(&a, &b), "but they are not the same value");
		assert_ne!(a.raw(), b.raw());

		let copy = a.clone();
		assert!(TieredValue::ptr_eq(&a, &copy), "a clone must name the same header");
		assert_eq!(a.bytes().as_ptr(), copy.bytes().as_ptr(), "and the same item");
		assert_eq!(a.strong_count(), 2);
	}

	/// The free must pick its allocator from the TAG. The counters sit inside
	/// the four functions that name a concrete allocator, so a free that went
	/// to the wrong one moves the wrong counter and this fails. Only the ITEM
	/// goes through them: the DRAM header is `Arc`'s, from the global
	/// allocator, and is not counted.
	#[test]
	fn free_routes_to_the_allocator_its_tier_names() {
		let _guard = routing_lock();

		for tier in [Tier::Fast, Tier::Slow] {
			let before = route_counts::snapshot();
			let value = TieredValue::new_in(KEY, b"routed", tier, None);
			let after_alloc = route_counts::snapshot();

			drop(value);
			let after_free = route_counts::snapshot();

			let (allocs, frees) = match tier {
				Tier::Fast => (
					(after_alloc.fast_allocs - before.fast_allocs, after_alloc.slow_allocs - before.slow_allocs),
					(after_free.fast_frees - before.fast_frees, after_free.slow_frees - before.slow_frees),
				),
				Tier::Slow => (
					(after_alloc.slow_allocs - before.slow_allocs, after_alloc.fast_allocs - before.fast_allocs),
					(after_free.slow_frees - before.slow_frees, after_free.fast_frees - before.fast_frees),
				),
			};

			assert_eq!(allocs, (1, 0), "{tier:?}: one item from its own tier's allocator, none from the other");
			assert_eq!(
				frees,
				(1, 0),
				"{tier:?}: the item must go back to the allocator its tag names -- \
				 under segregated_value_arena anything else is a cross-arena free",
			);
		}
	}

	/// The last handle routinely drops on a different thread from the one that
	/// built the value, so routing must depend on the tag alone. The counters
	/// are read INSIDE the freeing thread, whose routing is the one under test.
	#[test]
	fn free_routes_correctly_from_a_foreign_thread() {
		let _guard = routing_lock();

		let fast = TieredValue::new_fast(KEY, b"made here, freed there", None);
		let slow = TieredValue::new_slow(KEY, b"made here, freed there", None);

		let counted = std::thread::spawn(move || {
			let before = route_counts::snapshot();

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
			"off-thread, each item must still be returned to the allocator its \
			 TAG names -- one fast free and one slow free, not two of either",
		);
	}

	/// The counting wrapper proves which entry point was CALLED; this proves
	/// the memory landed where the tier claims, by asking the kernel.
	///
	/// Skipped under `stock_jemalloc`, where the global allocator is
	/// deliberately unbound and so has no node to assert.
	#[test]
	#[cfg(not(feature = "stock_jemalloc"))]
	fn values_land_on_the_numa_node_their_tier_names() {
		use crate::numa_alloc::{self, NODE_FAST, NODE_SLOW, tests::node_of};

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

		assert_eq!(fast_node, NODE_FAST as i32, "a fast value must be on node {NODE_FAST}, not {fast_node}");
		assert_eq!(slow_node, NODE_SLOW as i32, "a slow value must be on node {NODE_SLOW}, not {slow_node}");
	}

	/// The property this layout exists for, asked of the kernel: a slow value
	/// keeps its COUNT in DRAM and puts its METADATA -- length, expiry, key --
	/// on the slow node with its bytes. The default layout passes the first
	/// half and fails the second; `fused_value` the other way round.
	///
	/// Skipped under `stock_jemalloc`, as above.
	#[test]
	#[cfg(not(feature = "stock_jemalloc"))]
	fn a_slow_value_keeps_its_count_in_dram_and_its_item_on_the_slow_node() {
		use crate::numa_alloc::{self, NODE_FAST, NODE_SLOW, tests::node_of};

		assert!(numa_alloc::init(), "the node-0 and node-1 arena pools must build");

		#[cfg(feature = "segregated_value_arena")]
		assert!(
			numa_alloc::init_node(numa_alloc::NODE_FAST_VALUES),
			"the segregated value pool must build",
		);

		let _guard = routing_lock();

		let bytes = vec![0x5Au8; 8192];
		let slow = TieredValue::new_slow(KEY, &bytes, None);

		let header_node = node_of(slow.raw() as *const u8);
		let item_node = node_of(slow.item_ptr() as *const u8);
		let key_node = node_of(slow.key() as *const u64 as *const u8);
		let bytes_node = node_of(slow.bytes().as_ptr());
		// The item, the key and the first bytes share the item's first page, so
		// the checks above are really one; the last byte is on another page.
		let tail_node = node_of(&slow.bytes()[bytes.len() - 1]);

		assert_eq!(
			header_node, NODE_FAST as i32,
			"the count and the tag must stay in DRAM: header on node {header_node}",
		);
		assert_eq!(item_node, NODE_SLOW as i32, "the length and expiry must tier: item on node {item_node}");
		assert_eq!(key_node, NODE_SLOW as i32, "the key must tier: key on node {key_node}");
		assert_eq!(bytes_node, NODE_SLOW as i32, "the bytes must tier: bytes on node {bytes_node}");
		assert_eq!(tail_node, NODE_SLOW as i32, "all of them: the last page is on node {tail_node}");

		// And the way back: promoting moves the item home, and the new header
		// is DRAM like every header.
		let fast = slow.migrated_to(Tier::Fast);

		assert_eq!(node_of(fast.raw() as *const u8), NODE_FAST as i32, "a promoted header is in DRAM");
		assert_eq!(node_of(fast.item_ptr() as *const u8), NODE_FAST as i32, "a promoted item is in DRAM");
	}

	/// A cluster19 key, 43 bytes: what a real Twitter trace hands the cache.
	const TRACE_KEY: &str = "Y~S288lhSR8~88Syh688S6C28UUR6U6~hSh6hhylh~l";

	/// A `String` key is held as its bytes INSIDE the item: the item is the only
	/// allocation on the value path, the key's bytes sit in it between the
	/// header and the value, and the item says so in bit 1 of its word.
	#[test]
	fn a_string_key_is_held_as_bytes_inside_the_item() {
		let _guard = routing_lock();

		let key = String::from(TRACE_KEY);

		for tier in [Tier::Fast, Tier::Slow] {
			let before = route_counts::snapshot();
			let value = TieredValue::new_in(key.clone(), b"payload", tier, None);
			let after = route_counts::snapshot();

			let allocs = (after.fast_allocs - before.fast_allocs) + (after.slow_allocs - before.slow_allocs);
			assert_eq!(allocs, 1, "{tier:?}: one item, and nothing else, on the value path");
			assert_ne!(value.tagged_word() & KEY_BYTES_BIT, 0, "{tier:?}: the word must say the item holds bytes");
			assert_eq!(value.tagged_word() & 0b100, 0, "{tier:?}: bit 2 stays reserved");

			let item = value.item_ptr().addr();
			let size = key_bytes_item_layout(key.len(), 7).size();
			let stored = value.key_bytes().expect("a String key is held as bytes");

			assert_eq!(stored, key.as_bytes(), "{tier:?}: the key's bytes");
			assert_eq!(stored.as_ptr().addr(), item + 12, "{tier:?}: right behind the 12-byte header");
			assert_eq!(value.item_prefix_bytes(), 56, "{tier:?}: 12 + 43, rounded up to 8");
			assert_eq!(value.bytes().as_ptr().addr(), item + 56, "{tier:?}: the value right after the key");
			assert!(56 + 7 <= size, "{tier:?}: all of it inside the item's {size} bytes");
			assert_eq!(value.bytes(), b"payload");
			assert_eq!(value.len(), 7);

			let mut flipped = key.clone().into_bytes();
			flipped[42] ^= 1;

			assert!(value.key_matches(&key), "{tier:?}: its own key");
			assert!(!value.key_matches(&String::from_utf8(flipped).unwrap()), "{tier:?}: one byte off");
			assert!(!value.key_matches(&key[..42].to_string()), "{tier:?}: a prefix of the key");
			assert!(!value.key_matches(&format!("{key}~")), "{tier:?}: an extension of the key");
			assert_eq!(value.key_owned(), key, "{tier:?}: rebuilt intact");
		}
	}

	/// Every key length the header's rounding treats differently, against value
	/// lengths either side of a size class, through a demotion and a promotion.
	/// Each hop rebuilds the item on the new tier; nothing but the tier may
	/// change.
	#[test]
	fn string_keys_of_every_length_survive_fast_slow_fast() {
		let _guard = routing_lock();

		let expiry = NonZeroU32::new(0x00C0_FFEE);

		for key_len in [0usize, 1, 3, 4, 5, 11, 12, 13, 42, 43, 44, 250] {
			let key: String = (0..key_len).map(|i| char::from(b'!' + (i % 90) as u8)).collect();

			for len in [0usize, 1, 63, 100, 4040, 4096] {
				let bytes: Vec<u8> = (0..len).map(|i| (i * 7 % 251) as u8).collect();

				let fast = TieredValue::new_fast(key.clone(), &bytes, expiry);
				let slow = fast.migrated_to(Tier::Slow);
				let back = slow.migrated_to(Tier::Fast);

				for (hop, value, tier) in [
					("built", &fast, Tier::Fast),
					("demoted", &slow, Tier::Slow),
					("promoted", &back, Tier::Fast),
				] {
					let at = format!("key {key_len} B, value {len} B, {hop}");

					assert_eq!(value.tier(), tier, "{at}: wrong tier");
					assert_eq!(value.key_bytes(), Some(key.as_bytes()), "{at}: the key did not travel");
					assert!(value.key_matches(&key), "{at}: key_matches");
					assert_eq!(value.key_owned(), key, "{at}: key_owned");
					assert_eq!(value.len() as usize, len, "{at}: the length did not travel");
					assert_eq!(value.expiry(), expiry, "{at}: the expiry did not travel");
					assert_eq!(value.bytes(), &bytes[..], "{at}: the bytes did not travel");
					assert_eq!(value.bytes().as_ptr().addr() % VALUE_ALIGN, 0, "{at}: the value must stay 8-aligned");
					assert_eq!(value.item_prefix_bytes(), key_bytes_value_offset(key_len), "{at}: prefix");
				}
			}
		}
	}

	/// The other two byte-string key types take the same path, including a key
	/// that is not UTF-8, which only a `String` would reject; a POD key still
	/// takes the other one.
	#[test]
	fn byte_vector_and_boxed_slice_keys_are_held_as_bytes_too() {
		let _guard = routing_lock();

		let raw = vec![0xFFu8, 0x00, 0x80, b'k', 0xC3];

		let vec_key = TieredValue::new_slow(raw.clone(), b"v", None);
		assert_ne!(vec_key.tagged_word() & KEY_BYTES_BIT, 0);
		assert_eq!(vec_key.key_bytes(), Some(&raw[..]));
		assert!(vec_key.key_matches(&raw));
		assert_eq!(vec_key.migrated_to(Tier::Fast).key_owned(), raw);

		let boxed: Box<[u8]> = raw.clone().into_boxed_slice();
		let box_key = TieredValue::new_fast(boxed.clone(), b"v", None);
		assert_ne!(box_key.tagged_word() & KEY_BYTES_BIT, 0);
		assert!(box_key.key_matches(&boxed));
		assert_eq!(box_key.migrated_to(Tier::Slow).key_owned(), boxed);

		assert_eq!(TieredValue::new_fast(KEY, b"v", None).tagged_word() & KEY_BYTES_BIT, 0);
	}

	/// `key()` has no `K` to lend for a key held as bytes, and says so rather
	/// than reading one that is not there.
	#[test]
	#[should_panic(expected = "holds its key as bytes")]
	fn borrowing_a_key_held_as_bytes_panics() {
		let _guard = routing_lock();

		let value = TieredValue::new_fast(String::from(TRACE_KEY), b"v", None);
		let _ = value.key();
	}

	/// A bytes-key item is freed to its own tier exactly once, however many
	/// handles and hops it went through -- the `Drop` path that reads the key's
	/// length from the item to rebuild the layout.
	#[test]
	fn a_bytes_key_item_frees_to_its_tier_exactly_once() {
		let _guard = routing_lock();

		let before = route_counts::snapshot();
		let fast = TieredValue::new_fast(String::from(TRACE_KEY), &[7u8; 300], None);
		drop(fast.clone());
		let slow = fast.migrated_to(Tier::Slow);
		drop(fast);
		let alias = slow.clone();
		drop(slow);
		let mid = route_counts::snapshot();
		drop(alias);
		let after = route_counts::snapshot();

		assert_eq!(mid.fast_allocs - before.fast_allocs, 1, "one fast item");
		assert_eq!(mid.slow_allocs - before.slow_allocs, 1, "one slow item");
		assert_eq!(mid.fast_frees - before.fast_frees, 1, "the fast item, freed when its last handle went");
		assert_eq!(mid.slow_frees - before.slow_frees, 0, "the slow item still has a handle");
		assert_eq!(after.slow_frees - mid.slow_frees, 1, "and is freed once when that goes");
	}

	/// The point of holding a key as bytes, asked of the kernel: a slow item's
	/// key is on the slow node with its value, not in DRAM.
	///
	/// Skipped under `stock_jemalloc`, as above.
	#[test]
	#[cfg(not(feature = "stock_jemalloc"))]
	fn a_slow_string_keys_bytes_are_on_the_slow_node() {
		use crate::numa_alloc::{self, NODE_FAST, NODE_SLOW, tests::node_of};

		assert!(numa_alloc::init(), "the node-0 and node-1 arena pools must build");

		#[cfg(feature = "segregated_value_arena")]
		assert!(
			numa_alloc::init_node(numa_alloc::NODE_FAST_VALUES),
			"the segregated value pool must build",
		);

		let _guard = routing_lock();

		let key = "k".repeat(250);
		let slow = TieredValue::new_slow(key.clone(), &vec![0x5Au8; 8192], None);
		let stored = slow.key_bytes().expect("a String key is held as bytes");

		assert_eq!(node_of(slow.raw() as *const u8), NODE_FAST as i32, "the header stays in DRAM");
		assert_eq!(node_of(stored.as_ptr()), NODE_SLOW as i32, "the key's first byte is on the slow node");
		assert_eq!(node_of(&stored[stored.len() - 1]), NODE_SLOW as i32, "and its last");

		let fast = slow.migrated_to(Tier::Fast);
		let stored = fast.key_bytes().expect("a String key is held as bytes");

		assert_eq!(node_of(stored.as_ptr()), NODE_FAST as i32, "a promoted key comes home with its item");
		assert_eq!(fast.key_owned(), key);
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
			assert!(seen.insert(value.item_ptr()), "two live values share an item");
		}
	}

	/// The alignment must be REQUESTED, not merely received: jemalloc's
	/// smallest class is 8-aligned anyway, so an outcome check alone would
	/// pass with the alignment dropped from the layout. See `value.rs`.
	#[test]
	fn the_item_layout_always_demands_eight_byte_alignment() {
		for len in [0u32, 1, 2, 3, 7, 8, 9, 15, 16, 100, 4096, u32::MAX] {
			let layout = item_layout::<u64>(len);
			let offset = bytes_offset::<u64>();

			assert_eq!(
				layout.align(),
				VALUE_ALIGN,
				"len {len}: the layout must DEMAND {VALUE_ALIGN}-byte alignment, so \
				 the low bits are reserved by contract rather than by luck",
			);
			assert_eq!(offset % VALUE_ALIGN, 0, "len {len}: the bytes must start {VALUE_ALIGN}-aligned");
			assert!(layout.size() >= offset + len as usize, "len {len}: the layout must hold metadata AND bytes");
			assert!(
				layout.size() - (offset + len as usize) < VALUE_ALIGN,
				"len {len}: the layout must not waste a whole alignment unit",
			);
			assert_eq!(layout.size() % VALUE_ALIGN, 0, "len {len}: `pad_to_align` must leave whole units");
		}
	}

	/// Which POOL served each allocation -- the check `node_of` structurally
	/// cannot make. Under `segregated_value_arena` the fast allocator is
	/// `numa_alloc::FastValues`, and it and the global allocator are both bound
	/// to physical node 0, so only the arena index tells them apart. See
	/// `value.rs` for the full argument.
	///
	/// The DRAM header is checked too: it is `Arc`'s, so it must come from the
	/// global allocator's pool whichever tier the item is in -- never from the
	/// slow pool, and never from the segregated value pool.
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

		let fast_pool = if cfg!(feature = "segregated_value_arena") {
			assert!(numa_alloc::init_node(NODE_FAST_VALUES), "the segregated value pool must build");
			NODE_FAST_VALUES
		} else {
			NODE_FAST
		};

		let _guard = routing_lock();

		let global_arenas = pool_arenas(NODE_FAST);
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

			let arena = |ptr: *const u8| {
				arena_of(ptr).expect("arenas.lookup must be available in this jemalloc build")
			};

			let fast_arena = arena(fast.item_ptr() as *const u8);
			let slow_arena = arena(slow.item_ptr() as *const u8);

			assert!(
				fast_arenas.contains(&fast_arena),
				"len {len}: a fast item came from arena {fast_arena}, not in pool \
				 {fast_pool}'s set {fast_arenas:?}",
			);
			assert!(
				slow_arenas.contains(&slow_arena),
				"len {len}: a slow item came from arena {slow_arena}, not in pool \
				 {NODE_SLOW}'s set {slow_arenas:?}",
			);

			for (value, tier) in [(&fast, Tier::Fast), (&slow, Tier::Slow)] {
				let header_arena = arena(value.raw() as *const u8);

				assert!(
					global_arenas.contains(&header_arena),
					"len {len}, {tier:?}: the DRAM header came from arena \
					 {header_arena}, not the global allocator's pool {global_arenas:?}",
				);
			}
		}
	}
}
