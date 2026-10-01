/*
 * Copyright (c) Kia Shakiba
 *
 * This source code is licensed under the GNU AGPLv3 license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! A set admitted BEFORE its value is read: the library half of the server's
//! SET permit (fast-tier backpressure plan S9).
//!
//! `PaperCache::set` takes the value as a slice, so a server has to read a
//! request's body off its socket into a buffer of its own before the cache can
//! even decide whether to take it -- a buffer no budget covers, and a copy
//! into the cache's allocation after it. Three steps replace that:
//!
//! ```text
//!   reserve_set(key, len, ttl, deadline)   admission: everything `set` decides
//!        |                                  from the key and the LENGTH -- the size
//!        v                                  checks, the metadata cap, the tier, the
//!   SetPermit                               structural check, and for a value to be
//!        |                                  built fast the byte gate, which WAITS
//!        | fill()                           until demotions free room (or the
//!        v                                  deadline passes). Nothing is allocated.
//!   PendingSet                             the value's allocation, in the tier the
//!        |                                  permit decided, UNINITIALIZED: written in
//!        | write / read_exact_from          place, with no zero-fill (for a slow
//!        v                                  value a write to CXL nobody asked for).
//!   commit()                               inserts exactly as `set` does.
//! ```
//!
//! While the byte gate holds a set the body is still in the kernel's socket
//! buffer, so TCP flow control slows that one client and no cache DRAM is held
//! for it. `set` is itself these steps -- `reserve_set` with no deadline, the
//! value copied from its slice into the allocation, and `commit` -- so the two
//! paths cannot drift apart (T14, which only calls `set`, covers the shared
//! decisions).
//!
//! # A key borrowed from the request
//!
//! The key is the first thing a server reads, and a server that keeps a buffer
//! for it per connection has it as a `&[u8]`, not as a `K`.
//! [`PaperCache::reserve_set_borrowed`] is `reserve_set` for such a key (a cache
//! keyed by `Box<[u8]>`, `Vec<u8>` or `String`, see [`KeyBytes`]): admission is
//! decided from the key's hash and length, so nothing is built to ask; under
//! `thin_header` the bytes are copied from the slice into the item when
//! [`SetPermit::fill`] allocates it, and no key buffer is allocated at all;
//! under the default layout the header stores a `K`, and `fill` builds one from
//! the bytes, once. The permit borrows the slice until `fill`.
//!
//! # Abandoning a set
//!
//! A client that disconnects mid-value is the normal case here, not an
//! exception. Dropping a [`SetPermit`] releases the bytes the gate reserved for
//! it; dropping a [`PendingSet`] at any point after `fill` frees the allocation
//! to its tier and refunds P (the allocation was charged when it was made),
//! exactly once; in neither case is anything inserted or sent. The same holds if
//! `commit` returns an error, and when a panic unwinds through any of them.
//!
//! # The deadline
//!
//! `reserve_set` waits at most until its `deadline`. A wait ends there with the
//! error the lane it waits in gives when its own watchdog gives up --
//! [`CacheError::FastTierStalled`] in the bytes lane (the fast tier stayed full),
//! [`CacheError::MetadataOverflow`] in the metadata lane (`EvictToFit` found no
//! room in time) -- but never with an [`OnStall`] action: a set
//! that is out of time is not built slow or admitted over the budget, the caller
//! is told. A deadline that has already passed means no waiting at all: a set
//! that would wait fails at once. The gate's own watchdog (`stall_window`) still
//! applies before the deadline, and with `GateMode::Off`, or while the tier has
//! room, a reservation does not wait and the deadline is never read.
//!
//! # The slot
//!
//! A [`PendingSet`] hands out its value's bytes as `&mut [MaybeUninit<u8>]`
//! ([`PendingSet::unfilled`]), the part not yet written, and counts what has
//! been (`advance`). That is stable Rust and independent of how the bytes
//! arrive; [`PendingSet::read_exact_from`] is the common case done safely --
//! `Read::read_buf_exact` into a `BorrowedBuf` over the slot, so a body is
//! received straight into the allocation (by a reader that implements
//! `read_buf` itself, as std's sockets, files and slices do; the trait's
//! default zero-fills what it is handed first) -- and [`PendingSet::write`]
//! copies a slice. `commit` refuses (it panics) a value that is not written whole: the
//! type cannot say so statically for a reader that fills it in pieces, and an
//! uninitialized byte must never be published.
//!
//! # Concurrency
//!
//! A server serves many connections, and the byte gate's fast path admits a
//! set without looking at P, so setters that are already in flight when the
//! tier fills overshoot by a value each. The near band `N` is widened to
//! `(concurrency_hint + live setters) x value_hint` for that
//! (`GateConfig::concurrency_hint`); [`PaperCache::register_setter`] counts the
//! live setters, one per accepted connection, so a server does not have to know
//! its connections when it builds the cache. The count has an effect only when
//! `GateConfig::value_hint` is set (the typical value size), and none when no
//! setter is registered.

use std::{
	hash::{BuildHasher, Hash},
	io::{self, BorrowedBuf, Read},
	mem::MaybeUninit,
	sync::Arc,
	time::Instant,
};

use typesize::TypeSize;

use crate::{
	CacheError,
	HashedKey,
	KeyBytes,
	PaperCache,
	StatusRef,
	TieredBuffer,
	Tier,
	gate::{Admission, Sizes},
	object::{Object, expiry_from_ttl},
	value::{TieredValue, UninitValue},
	worker::Placement,
};

impl<K, S> PaperCache<K, TieredBuffer, S>
where
	K: 'static + Eq + Hash + TypeSize + Clone + Send + Sync,
	S: Default + Clone + BuildHasher,
{
	/// Admits a set before its value is read: everything [`PaperCache::set`]
	/// decides from the key, the value's LENGTH `len` and its `ttl` -- the size
	/// checks, the metadata cap, the tier, structural slow placement and, for a
	/// value to be built in the fast tier, the byte gate -- and returns the
	/// [`SetPermit`] to fill. Nothing is allocated or sent. See the [module
	/// documentation](self) for the three steps.
	///
	/// A set the byte gate holds waits here, FIFO, until demotions free room, and
	/// at most until `deadline`. `ttl` is the TTL the set will carry: it sizes
	/// the admission. A server that reads the TTL after the value (the wire's
	/// order is key, value, TTL) passes `None` and gives the real one with
	/// [`PendingSet::set_ttl`] before it commits.
	///
	/// # Errors
	///
	/// As [`PaperCache::set`], from the checks it makes before building:
	/// [`CacheError::ExceedingValueSize`], [`CacheError::ZeroValueSize`],
	/// [`CacheError::MetadataOverflow`], [`CacheError::FastTierStalled`] (for
	/// the watchdog's `OnStall::Error`, and for the `deadline` passing while the
	/// set waits for room in the fast tier), [`CacheError::Internal`].
	/// `MetadataOverflow` is also what a `deadline` that passes while the set
	/// waits in the metadata lane (`MetadataOverflow::EvictToFit`) returns.
	pub fn reserve_set(
		&self,
		key: K,
		len: usize,
		ttl: Option<u32>,
		deadline: Instant,
	) -> Result<SetPermit<'_, K, S>, CacheError> {
		let admission = self.begin_set_until(self.key_figures(&key), len, ttl, Some(deadline))?;

		Ok(SetPermit { cache: self, key: PermitKey::Owned(key), admission })
	}

	/// Allocates the value `admission` decided on, UNINITIALIZED, in its tier --
	/// charged to P -- and releases the byte gate's reservation (the bytes are
	/// P's now). The one place a set's value is made, for `set` and for a
	/// permit's `fill`.
	pub(crate) fn allocate(&self, admission: Admission<'_>, key: PermitKey<'_, K>) -> PendingSet<'_, K, S> {
		let Admission { hashed, tier, placement, len, ttl, sizes, reservation } = admission;

		let value = match key {
			PermitKey::Owned(key) => {
				debug_assert_eq!(self.hash_key(&key), hashed, "a value is built for the key begin_set checked");

				TieredValue::new_uninit_in(key, len, tier)
			},

			// The bytes are copied into the item (`thin_header`), or a `K` is
			// built from them (the default layout), here and nowhere else.
			PermitKey::Borrowed(bytes) => TieredValue::new_uninit_in_bytes(bytes, len, tier),
		};

		// B2: the value is allocated -- charged to P if fast -- so the bytes the
		// byte gate held for it go back (waking the lane's head if anyone waits).
		drop(reservation);

		PendingSet { cache: self, value, filled: 0, hashed, placement, ttl, reserved_ttl: ttl, sizes }
	}

	/// Counts one more live setter -- a connection a server has accepted -- into
	/// the near band of the byte gate, until the returned guard drops (the
	/// connection closes). The near band `N = B - max(near_frac x eff,
	/// concurrency_hint x value_hint)` is widened by the setters that are live
	/// at the policy worker's next pass (`(concurrency_hint + live setters) x
	/// value_hint`), so a fast-path set that is in flight when the tier fills
	/// overshoots the budget by at most the bytes the band leaves room for.
	///
	/// With no setter registered nothing changes, and with `value_hint` 0 -- the
	/// default -- a registered setter widens nothing: set
	/// [`GateConfig::value_hint`](crate::GateConfig) (or
	/// `PAPER_GATE_VALUE_HINT_BYTES`) to the typical value size.
	pub fn register_setter(&self) -> SetterGuard {
		let status = Arc::clone(&self.status);

		status.gate().add_setter();
		status.kick_policy_worker();

		SetterGuard { status }
	}

	/// The setters registered and not yet released ([`PaperCache::register_setter`]).
	#[must_use]
	pub fn live_setters(&self) -> u32 {
		self.status.gate().setters()
	}
}

impl<K, S> PaperCache<K, TieredBuffer, S>
where
	K: KeyBytes,
	S: Default + Clone + BuildHasher,
{
	/// [`PaperCache::reserve_set`] with the key given as its bytes: for a cache
	/// keyed by `Box<[u8]>`, `Vec<u8>` or `String`. See the [module
	/// documentation](self), "A key borrowed from the request". The permit
	/// borrows `key` until [`SetPermit::fill`], which copies it where it
	/// belongs.
	///
	/// # Errors
	///
	/// As `reserve_set`, and [`CacheError::InvalidKey`] if no key of this cache's
	/// type holds the bytes (a `String` cache, bytes that are not UTF-8).
	pub fn reserve_set_borrowed<'c>(
		&'c self,
		key: &'c [u8],
		len: usize,
		ttl: Option<u32>,
		deadline: Instant,
	) -> Result<SetPermit<'c, K, S>, CacheError> {
		let admission = self.begin_set_borrowed(key, len, ttl, Some(deadline))?;

		Ok(SetPermit { cache: self, key: PermitKey::Borrowed(key), admission })
	}
}

/// A set the cache has admitted whose value has not been read: what
/// [`PaperCache::reserve_set`] returns. Dropping it abandons the set -- the
/// bytes the byte gate reserved for it are released, nothing was allocated or
/// inserted. [`SetPermit::fill`] allocates the value.
#[must_use = "a permit that is dropped abandons the set"]
pub struct SetPermit<'c, K, S> {
	cache: &'c PaperCache<K, TieredBuffer, S>,
	key: PermitKey<'c, K>,
	admission: Admission<'c>,
}

/// A set's key between its admission and the allocation of its value: the key
/// the caller gave, or -- for a cache of byte-string keys -- the bytes of one,
/// borrowed from the request (`PaperCache::reserve_set_borrowed`).
pub(crate) enum PermitKey<'k, K> {
	Owned(K),
	Borrowed(&'k [u8]),
}

impl<'c, K, S> SetPermit<'c, K, S>
where
	K: 'static + Eq + Hash + TypeSize + Clone + Send + Sync,
	S: Default + Clone + BuildHasher,
{
	/// The value's length in bytes, as reserved.
	#[must_use]
	pub fn len(&self) -> usize {
		self.admission.len
	}

	#[must_use]
	pub fn is_empty(&self) -> bool {
		self.admission.len == 0
	}

	/// The tier the value will be built in: the fast tier, or the slow tier --
	/// the design's own choice, a value larger than an empty fast tier
	/// (structural slow placement), or a set that was diverted.
	#[must_use]
	pub fn tier(&self) -> Tier {
		self.admission.tier
	}

	/// Allocates the value in that tier, its bytes UNINITIALIZED, charges it
	/// to P and releases the byte gate's reservation, and returns the
	/// [`PendingSet`] to write it into.
	pub fn fill(self) -> PendingSet<'c, K, S> {
		self.cache.allocate(self.admission, self.key)
	}
}

/// A value allocated in its tier, its bytes (partly) unwritten, and everything
/// its commit needs: what [`SetPermit::fill`] returns. Write all
/// [`PendingSet::len`] bytes -- [`PendingSet::read_exact_from`],
/// [`PendingSet::write`], or [`PendingSet::unfilled`] and
/// [`PendingSet::advance`] -- then [`PendingSet::commit`].
///
/// Dropping it abandons the set, at any point: the allocation is freed to its
/// tier and P is refunded what it was charged, exactly once, and nothing is
/// inserted or sent.
#[must_use = "a pending set that is dropped abandons the set"]
pub struct PendingSet<'c, K, S> {
	cache: &'c PaperCache<K, TieredBuffer, S>,
	value: UninitValue<K>,

	/// Bytes written: the front of the value.
	filled: usize,

	hashed: HashedKey,
	placement: Placement,

	/// The TTL the set will commit with, and the one it was admitted under.
	ttl: Option<u32>,
	reserved_ttl: Option<u32>,

	/// What admission computed (`base` and `resident` for `reserved_ttl`).
	sizes: Sizes,
}

impl<K, S> PendingSet<'_, K, S>
where
	K: 'static + Eq + Hash + TypeSize + Clone + Send + Sync,
	S: Default + Clone + BuildHasher,
{
	/// The value's length in bytes, as reserved.
	#[must_use]
	pub fn len(&self) -> usize {
		self.value.len()
	}

	#[must_use]
	pub fn is_empty(&self) -> bool {
		self.value.len() == 0
	}

	/// Bytes written so far.
	#[must_use]
	pub fn filled(&self) -> usize {
		self.filled
	}

	/// Whether every byte has been written: what `commit` requires.
	#[must_use]
	pub fn is_full(&self) -> bool {
		self.filled == self.value.len()
	}

	/// The tier the value was allocated in.
	#[must_use]
	pub fn tier(&self) -> Tier {
		self.value.tier()
	}

	/// The bytes not yet written, uninitialized. Write a prefix of it, then
	/// [`PendingSet::advance`].
	pub fn unfilled(&mut self) -> &mut [MaybeUninit<u8>] {
		let filled = self.filled;

		&mut self.value.bytes_mut()[filled..]
	}

	/// Marks `n` more bytes, the front of [`PendingSet::unfilled`], as written.
	///
	/// # Safety
	///
	/// Those `n` bytes must have been initialized, and `n` must not exceed what
	/// is unfilled.
	pub unsafe fn advance(&mut self, n: usize) {
		debug_assert!(n <= self.value.len() - self.filled, "advanced past the end of the value");

		self.filled += n;
	}

	/// Copies `bytes` into the next bytes of the value.
	///
	/// # Panics
	///
	/// If `bytes` is longer than what is unfilled.
	pub fn write(&mut self, bytes: &[u8]) {
		let slot = self.unfilled();

		assert!(bytes.len() <= slot.len(), "{} bytes written to a value with {} unfilled", bytes.len(), slot.len());

		// SAFETY: `slot` has room for `bytes.len()` bytes (just checked), a fresh
		// allocation the caller's slice cannot overlap, and a `MaybeUninit<u8>`
		// has a `u8`'s layout.
		unsafe {
			std::ptr::copy_nonoverlapping(bytes.as_ptr(), slot.as_mut_ptr().cast::<u8>(), bytes.len());
			self.advance(bytes.len());
		}
	}

	/// Reads exactly the unfilled bytes from `reader` into the value, in place:
	/// `Read::read_buf_exact` into a `BorrowedBuf` over the slot, so nothing is
	/// staged, and no zero-fill is written first if the reader implements
	/// `read_buf` itself, as std's sockets, files and slices do (the trait's
	/// default zero-fills what it is handed before it reads). The bytes a
	/// reader that fails part way did deliver stay counted (it is the caller's
	/// to drop or to read on, after a `WouldBlock`); `UnexpectedEof` if the
	/// reader ends first.
	pub fn read_exact_from<R: Read + ?Sized>(&mut self, reader: &mut R) -> io::Result<()> {
		let (read, n) = {
			let mut buf = BorrowedBuf::from(self.unfilled());
			let read = reader.read_buf_exact(buf.unfilled());

			(read, buf.len())
		};

		// SAFETY: `BorrowedBuf` counts exactly the bytes its reader initialized,
		// from the front of the slot, and cannot count more than the slot has.
		unsafe { self.advance(n) };

		read
	}

	/// Changes the TTL the set will commit with: for a server that reads the TTL
	/// after the value's bytes. Admission was sized for the TTL `reserve_set`
	/// was given, so when this is a different one the commit takes the
	/// object's own sizes -- the DRAM charge for a TTL'd object comes and goes --
	/// and refuses the set with [`CacheError::ExceedingValueSize`] if they put
	/// it over the eviction threshold; a TTL the same as the reserved one
	/// changes nothing.
	pub fn set_ttl(&mut self, ttl: Option<u32>) {
		self.ttl = ttl;
	}

	/// Builds the value around its bytes and publishes it: inserted, sized,
	/// its `Set` sent with the placement admission decided -- exactly what
	/// `PaperCache::set` does after it has built its value.
	///
	/// # Panics
	///
	/// If the value is not written whole ([`PendingSet::is_full`]): an
	/// uninitialized byte is never published. The set is abandoned as the panic
	/// unwinds.
	///
	/// # Errors
	///
	/// [`CacheError::ExceedingValueSize`] if a different TTL ([`PendingSet::set_ttl`])
	/// put the set over the eviction threshold; [`CacheError::Internal`] if a
	/// worker could not be told. The value is dropped, and P refunded, either way.
	pub fn commit(self) -> Result<(), CacheError> {
		let PendingSet { cache, value, filled, hashed, placement, ttl, reserved_ttl, sizes } = self;

		assert_eq!(filled, value.len(), "a value is committed only once every byte of it is written");

		// SAFETY: every byte of the value was written (just asserted), through
		// `unfilled` and `advance`, `write` or `read_exact_from`.
		let value = unsafe { value.assume_init(expiry_from_ttl(ttl)) };
		let object = Object::<K, TieredBuffer>::from_value(value);

		let (base, resident) = match ttl == reserved_ttl {
			true => (sizes.base, sizes.resident),

			false => (
				cache.overhead_manager.base_size(&object),
				cache.overhead_manager.dram_resident_size(&object),
			),
		};

		if base != sizes.base && cache.status.exceeds_eviction_threshold(base) {
			return Err(CacheError::ExceedingValueSize);
		}

		cache.publish_set(hashed, object, placement, base, resident)
	}
}

/// One live setter, counted into the byte gate's near band for as long as it
/// lives: what [`PaperCache::register_setter`] returns. It holds nothing of the
/// cache but its status, so it may be kept in a connection's state and dropped
/// from any thread, and a cache dropped before it is no matter.
#[must_use = "a setter is counted only while its guard lives"]
pub struct SetterGuard {
	status: StatusRef,
}

impl Drop for SetterGuard {
	fn drop(&mut self) {
		self.status.gate().remove_setter();
		self.status.kick_policy_worker();
	}
}
