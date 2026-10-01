/*
 * Copyright (c) Griffin Andrew
 *
 * This source code is licensed under the GNU AGPLv3 license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! Byte-string keys, used BORROWED: the `*_borrowed` methods of `PaperCache`.
//!
//! A server reads a request's key off its socket into a buffer it keeps for
//! the connection. `PaperCache::get(&K)` and `set(K, ..)` want a `K`, so it
//! built one -- an allocation per request -- only for the cache to hash it,
//! compare it, and, on a set, copy its bytes into the item and free it. The
//! `*_borrowed` methods take the key as `&[u8]` instead, for a cache whose key
//! type is one of three byte strings: `Box<[u8]>`, `Vec<u8>` or `String`.
//!
//! What a borrowed key does, per layout:
//!
//! * `get`, `has`, `peek`, `ttl`, `size`, `del`: hash the bytes, find the
//!   object, compare the bytes. No `K` is built, under either layout.
//! * `set`, under `thin_header`: the item holds the key as bytes
//!   (`value_thin.rs`, "The key"), so they are copied from the caller's slice
//!   into the item, once. No `K` is built.
//! * `set`, under the default layout: the key is stored in the DRAM header as
//!   a `K`, so one is built from the bytes, once, when the value is allocated
//!   -- and not before, so a set the cache refuses builds nothing.
//!
//! # Why a trait of its own
//!
//! `Borrow<[u8]>` would be the std spelling, but `String` is not one: it hashes
//! as a `str` (its bytes, then a `0xff`), not as a `[u8]` (a length, then its
//! bytes), so a lookup by `&[u8]` would hash somewhere else than the key was
//! stored. And no std conversion builds all three from `&[u8]`: a `String`
//! needs UTF-8. [`KeyBytes`] says each of the three once: how its owned key
//! hashes, which byte strings it can hold, and what the accounting charges for
//! it. It is sealed: it is implemented for exactly the types `thin_header`
//! holds as bytes (`value_thin.rs`, `key_as_bytes`), and a test holds the two
//! lists equal.
//!
//! The one equation everything rests on, held by tests for every key type and
//! more than one hasher: `K::hash_key_bytes(hasher, bytes) ==
//! hasher.hash_one(&key)` for the `key` that holds `bytes`.

use std::hash::{BuildHasher, Hash};

use typesize::TypeSize;

use crate::HashedKey;

mod sealed {
	pub trait Sealed {}

	impl Sealed for Box<[u8]> {}
	impl Sealed for Vec<u8> {}
	impl Sealed for String {}
}

/// A cache key type that is a byte string: `Box<[u8]>`, `Vec<u8>` or `String`.
/// See the module documentation. Sealed.
pub trait KeyBytes: sealed::Sealed + 'static + Eq + Hash + TypeSize + Clone + Send + Sync {
	/// The key's bytes.
	fn key_bytes(&self) -> &[u8];

	/// Whether some key of this type holds exactly `bytes`: every byte string
	/// for `Box<[u8]>` and `Vec<u8>`, valid UTF-8 for `String`. A lookup of
	/// bytes that no key holds is a miss; a set of them is
	/// [`CacheError::InvalidKey`](crate::CacheError::InvalidKey).
	fn holds(bytes: &[u8]) -> bool;

	/// The key that holds `bytes`, or `None` if none does ([`KeyBytes::holds`]).
	fn from_key_bytes(bytes: &[u8]) -> Option<Self>;

	/// `hasher.hash_one(&key)`, for the `key` that holds `bytes`, without
	/// building it. If no key holds them the value is only an address that
	/// nothing is stored at: the lookup then misses on the comparison.
	fn hash_key_bytes<S: BuildHasher>(hasher: &S, bytes: &[u8]) -> HashedKey;

	/// `TypeSize::get_size` of the key `from_key_bytes` builds for `len` bytes
	/// (its handle and its exactly-sized buffer): what the default layout
	/// charges for a key it stores as a `K`.
	fn accounted_size(len: usize) -> usize;
}

impl KeyBytes for Box<[u8]> {
	#[inline]
	fn key_bytes(&self) -> &[u8] {
		self
	}

	#[inline]
	fn holds(_bytes: &[u8]) -> bool {
		true
	}

	#[inline]
	fn from_key_bytes(bytes: &[u8]) -> Option<Self> {
		Some(Box::from(bytes))
	}

	/// A `[u8]` hashes as its length, then its bytes -- as `Box<[u8]>` does.
	#[inline]
	fn hash_key_bytes<S: BuildHasher>(hasher: &S, bytes: &[u8]) -> HashedKey {
		hasher.hash_one(bytes)
	}

	#[inline]
	fn accounted_size(len: usize) -> usize {
		std::mem::size_of::<Self>() + len
	}
}

impl KeyBytes for Vec<u8> {
	#[inline]
	fn key_bytes(&self) -> &[u8] {
		self
	}

	#[inline]
	fn holds(_bytes: &[u8]) -> bool {
		true
	}

	#[inline]
	fn from_key_bytes(bytes: &[u8]) -> Option<Self> {
		Some(bytes.to_vec())
	}

	/// A `Vec<u8>` hashes as its slice: as `Box<[u8]>`.
	#[inline]
	fn hash_key_bytes<S: BuildHasher>(hasher: &S, bytes: &[u8]) -> HashedKey {
		hasher.hash_one(bytes)
	}

	#[inline]
	fn accounted_size(len: usize) -> usize {
		std::mem::size_of::<Self>() + len
	}
}

impl KeyBytes for String {
	#[inline]
	fn key_bytes(&self) -> &[u8] {
		self.as_bytes()
	}

	#[inline]
	fn holds(bytes: &[u8]) -> bool {
		std::str::from_utf8(bytes).is_ok()
	}

	#[inline]
	fn from_key_bytes(bytes: &[u8]) -> Option<Self> {
		std::str::from_utf8(bytes).ok().map(str::to_owned)
	}

	/// A `String` hashes as a `str`, which is not how a `[u8]` hashes (a
	/// terminator after the bytes, not a length before them). Hashing the
	/// `&str` is by construction what `String` does, whatever the hasher.
	#[inline]
	fn hash_key_bytes<S: BuildHasher>(hasher: &S, bytes: &[u8]) -> HashedKey {
		match std::str::from_utf8(bytes) {
			Ok(key) => hasher.hash_one(key),

			// No `String` holds these bytes, so nothing is stored under any
			// hash for them to match; any address will do.
			Err(_) => hasher.hash_one(bytes),
		}
	}

	#[inline]
	fn accounted_size(len: usize) -> usize {
		std::mem::size_of::<Self>() + len
	}
}

/// The key's bytes if `K` is one of the types that implement [`KeyBytes`], for
/// the code that has `K: 'static` and no more (`thin_header`'s item holds
/// exactly these as bytes). Decided by the TYPE: once `K` is monomorphised the
/// `TypeId` comparisons behind `downcast_ref` are constants.
#[cfg(feature = "thin_header")]
pub(crate) fn key_as_bytes<K: 'static>(key: &K) -> Option<&[u8]> {
	let key = key as &dyn std::any::Any;

	if let Some(key) = key.downcast_ref::<String>() {
		return Some(key.key_bytes());
	}

	if let Some(key) = key.downcast_ref::<Vec<u8>>() {
		return Some(key.key_bytes());
	}

	key.downcast_ref::<Box<[u8]>>().map(|key| key.key_bytes())
}

/// Whether `K` is one of the types that implement [`KeyBytes`]: whether
/// `key_as_bytes` would answer `Some` for a `K`.
#[cfg(all(feature = "thin_header", feature = "hybrid_cache_common"))]
pub(crate) fn is_key_bytes_type<K: 'static>() -> bool {
	use std::any::TypeId;

	let id = TypeId::of::<K>();

	id == TypeId::of::<String>() || id == TypeId::of::<Vec<u8>>() || id == TypeId::of::<Box<[u8]>>()
}

/// Rebuilds an owned key from the bytes it was made of: the inverse of
/// `key_as_bytes`, for the types it accepts.
///
/// # Panics
///
/// If `K` is not one of those types, or `K` is `String` and the bytes are not
/// UTF-8 -- neither can happen for bytes an item took from a key, nor for the
/// bytes of a set that was admitted (`KeyBytes::holds` is checked first).
#[cfg(any(feature = "thin_header", feature = "hybrid_cache_common"))]
pub(crate) fn key_from_bytes<K: 'static>(bytes: &[u8]) -> K {
	use std::any::Any;

	// Not a `Box<dyn Any>` of the key, which is an allocation of its own that
	// only an optimizer that sees through it removes: a slot of the key's type,
	// filled through the one downcast that fits.
	let mut built: Option<K> = None;
	let slot = &mut built as &mut dyn Any;

	if let Some(slot) = slot.downcast_mut::<Option<String>>() {
		*slot = Some(String::from_key_bytes(bytes).expect("a String key's bytes are UTF-8"));
	} else if let Some(slot) = slot.downcast_mut::<Option<Vec<u8>>>() {
		*slot = Some(Vec::<u8>::from_key_bytes(bytes).expect("every byte string is a Vec<u8> key"));
	} else if let Some(slot) = slot.downcast_mut::<Option<Box<[u8]>>>() {
		*slot = Some(<Box<[u8]>>::from_key_bytes(bytes).expect("every byte string is a Box<[u8]> key"));
	}

	built.expect("only the key types KeyBytes names are built from bytes")
}

#[cfg(test)]
mod tests {
	use std::hash::{BuildHasherDefault, DefaultHasher, Hasher, RandomState};

	use super::*;

	/// A hasher that is not a mixing function: it keeps every `write` it is
	/// given, in order, so that two keys hash alike only if they feed the
	/// hasher the same calls. (`RandomState` agrees only when the bytes do; this
	/// agrees only when the CALLS do, which is what `Hash` promises.)
	#[derive(Default)]
	struct Recorder(Vec<u8>);

	impl Hasher for Recorder {
		fn write(&mut self, bytes: &[u8]) {
			self.0.extend_from_slice(&(bytes.len() as u64).to_le_bytes());
			self.0.extend_from_slice(bytes);
		}

		fn write_u8(&mut self, i: u8) {
			self.0.extend_from_slice(&[b'u', i]);
		}

		fn write_usize(&mut self, i: usize) {
			self.0.extend_from_slice(b"z");
			self.0.extend_from_slice(&i.to_le_bytes());
		}

		fn finish(&self) -> u64 {
			// FNV-1a over the record.
			self.0.iter().fold(0xcbf2_9ce4_8422_2325u64, |hash, byte| {
				(hash ^ u64::from(*byte)).wrapping_mul(0x0100_0000_01b3)
			})
		}
	}

	type RecordingState = BuildHasherDefault<Recorder>;

	fn samples() -> Vec<Vec<u8>> {
		let mut samples: Vec<Vec<u8>> = vec![
			Vec::new(),
			b"k".to_vec(),
			b"key-0001".to_vec(),
			b"a key of nineteen b".to_vec(),
			"caf\u{e9} \u{1f980}".as_bytes().to_vec(),
		];

		samples.push((0..=255u8).filter(|byte| byte.is_ascii_graphic()).collect());
		samples.push(vec![b'x'; 4096]);

		samples
	}

	fn hash_agrees<K: KeyBytes, S: BuildHasher>(hasher: &S) {
		for bytes in samples() {
			let key = K::from_key_bytes(&bytes).expect("a valid sample");

			assert_eq!(K::hash_key_bytes(hasher, &bytes), hasher.hash_one(&key), "{} bytes", bytes.len());
			assert_eq!(key.key_bytes(), &bytes[..]);
			assert!(K::holds(&bytes));
		}
	}

	/// The equation the whole borrowed API rests on, for each key type and for
	/// a hasher that tells calls apart as well as for the real ones.
	#[test]
	fn a_borrowed_key_hashes_as_the_owned_key_does() {
		let random = RandomState::new();
		let recording = RecordingState::default();
		let default = BuildHasherDefault::<DefaultHasher>::default();

		hash_agrees::<Box<[u8]>, _>(&random);
		hash_agrees::<Vec<u8>, _>(&random);
		hash_agrees::<String, _>(&random);

		hash_agrees::<Box<[u8]>, _>(&recording);
		hash_agrees::<Vec<u8>, _>(&recording);
		hash_agrees::<String, _>(&recording);

		hash_agrees::<Box<[u8]>, _>(&default);
		hash_agrees::<Vec<u8>, _>(&default);
		hash_agrees::<String, _>(&default);
	}

	/// The reason the trait exists: a `String` does not hash as the `[u8]` of
	/// its bytes, so the byte-slice hash would have looked a `String` up in the
	/// wrong place. (`Vec<u8>` and `Box<[u8]>` do hash as one.)
	#[test]
	fn a_string_hashes_apart_from_its_bytes_and_the_byte_keys_do_not() {
		let recording = RecordingState::default();
		let bytes: &[u8] = b"key-0001";

		assert_ne!(recording.hash_one(bytes), recording.hash_one(String::from("key-0001")));
		assert_eq!(recording.hash_one(bytes), recording.hash_one(Vec::from(bytes)));
		assert_eq!(recording.hash_one(bytes), recording.hash_one(Box::<[u8]>::from(bytes)));
	}

	#[test]
	fn a_string_holds_only_utf8() {
		let invalid: &[u8] = &[b'a', 0xff, b'b'];

		assert!(!String::holds(invalid));
		assert_eq!(String::from_key_bytes(invalid), None);

		assert!(Vec::<u8>::holds(invalid));
		assert!(<Box<[u8]>>::holds(invalid));
		assert_eq!(Vec::<u8>::from_key_bytes(invalid).as_deref(), Some(invalid));

		// A hash is still produced, and is not the hash of any `String`.
		let hasher = RandomState::new();
		assert_eq!(String::hash_key_bytes(&hasher, invalid), hasher.hash_one(invalid));
	}

	/// The default layout's charge for a key stored as a `K` is `TypeSize`'s;
	/// the borrowed set computes it from the length, before any `K` exists.
	#[test]
	fn the_accounted_size_of_a_key_is_what_typesize_says_of_the_key_built() {
		fn agrees<K: KeyBytes>() {
			for bytes in samples() {
				let key = K::from_key_bytes(&bytes).expect("a valid sample");

				assert_eq!(K::accounted_size(bytes.len()), key.get_size(), "{} bytes", bytes.len());
			}
		}

		agrees::<Box<[u8]>>();
		agrees::<Vec<u8>>();
		agrees::<String>();
	}

	/// `key_from_bytes`, which the code that has only `K: 'static` calls, is
	/// the trait's `from_key_bytes`.
	#[cfg(any(feature = "thin_header", feature = "hybrid_cache_common"))]
	#[test]
	fn the_untyped_rebuild_is_the_traits() {
		let bytes = b"key-0001".as_slice();

		assert_eq!(key_from_bytes::<String>(bytes), "key-0001");
		assert_eq!(key_from_bytes::<Vec<u8>>(bytes), bytes);
		assert_eq!(&*key_from_bytes::<Box<[u8]>>(bytes), bytes);
	}

	/// `thin_header` holds exactly the types `KeyBytes` names as bytes.
	#[cfg(feature = "thin_header")]
	#[test]
	fn thin_header_holds_as_bytes_the_types_key_bytes_names() {
		assert_eq!(key_as_bytes(&String::from("ab")), Some(b"ab".as_slice()));
		assert_eq!(key_as_bytes(&b"ab".to_vec()), Some(b"ab".as_slice()));
		assert_eq!(key_as_bytes(&Box::<[u8]>::from(b"ab".as_slice())), Some(b"ab".as_slice()));

		assert_eq!(key_as_bytes(&7u64), None);
		assert_eq!(key_as_bytes(&[1u8, 2]), None);
	}

	#[cfg(all(feature = "thin_header", feature = "hybrid_cache_common"))]
	#[test]
	fn the_type_check_names_the_same_three_types() {
		assert!(is_key_bytes_type::<String>());
		assert!(is_key_bytes_type::<Vec<u8>>());
		assert!(is_key_bytes_type::<Box<[u8]>>());

		assert!(!is_key_bytes_type::<u64>());
		assert!(!is_key_bytes_type::<&'static str>());
	}
}
