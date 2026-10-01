/*
 * Copyright (c) Griffin Andrew
 *
 * This source code is licensed under the GNU AGPLv3 license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! The borrowed-key API (`crate::key_bytes`): `PaperCache`'s `*_borrowed`
//! methods, which take the key of a byte-string cache as `&[u8]`.
//!
//! What they promise, and the test that holds each:
//!
//! * the same entry is found by either spelling of its key, for all three key
//!   types, in a flat and in a tiered cache (`an_entry_is_found_by_either_spelling_of_its_key`,
//!   `a_tiered_entry_is_found_by_either_spelling_of_its_key`);
//! * the hash of a borrowed key is the owned key's (`a_borrowed_key_hashes_as_the_owned_key_does`);
//! * a set by a borrowed key is a set: the same objects, accounting and P
//!   (`a_borrowed_set_is_a_set`), and the figures admission computes from a key's
//!   length are the ones of the key (`the_figures_of_a_borrowed_key_are_the_keys`);
//! * under `thin_header` a borrowed set allocates no key, under either layout it
//!   builds at most the one the header stores, and a refused one builds none
//!   (`the_borrowed_paths_allocate_no_key_they_do_not_need`);
//! * a `String` cache holds no key that is not UTF-8
//!   (`a_string_cache_refuses_a_key_that_is_not_utf8`);
//! * an abandoned permit or pending set of a borrowed key leaves nothing behind
//!   (`an_abandoned_borrowed_set_is_refunded_once`).
//!
//! The tests that build a tiered cache run alone in a child process, as the
//! gate's do: P is process-global.

use std::{
	thread,
	time::{Duration, Instant},
};

use super::s5_gate_tests::{LEN, M0, TIER, gated, p};
use super::test_support::{alone_in, wait_for};

use crate::gate::{self, OnStall, test_hooks};
use crate::key_bytes::KeyBytes;
use crate::object::{Object, ObjectSize};
use crate::value::{BufferDRAM, TieredValue};
use crate::{CacheError, CacheTierSize, PaperCache, PaperPolicy, Tier, TieredBuffer, phys};

use Tier::{Fast, Slow};

fn alone(test: &str, body: impl FnOnce()) {
	alone_in(module_path!(), test, body);
}

fn soon() -> Instant {
	Instant::now() + Duration::from_secs(30)
}

/// A byte-string key type, made from a number: its length varies with it.
trait Keyed: KeyBytes + std::fmt::Debug {
	fn make(n: u64) -> Self;
	fn name() -> &'static str;
}

fn bytes_of_key(n: u64) -> Vec<u8> {
	format!("key-{n}-{}", "k".repeat((n % 41) as usize)).into_bytes()
}

// Built through `from_key_bytes`, so that the buffer is exactly the key's: a
// `Vec` or a `String` made by `format!` has spare capacity, which the accounting
// charges (`TypeSize`), and the key built from borrowed bytes has none.
impl Keyed for Box<[u8]> {
	fn make(n: u64) -> Self {
		Self::from_key_bytes(&bytes_of_key(n)).expect("any bytes")
	}

	fn name() -> &'static str {
		"Box<[u8]>"
	}
}

impl Keyed for Vec<u8> {
	fn make(n: u64) -> Self {
		Self::from_key_bytes(&bytes_of_key(n)).expect("any bytes")
	}

	fn name() -> &'static str {
		"Vec<u8>"
	}
}

impl Keyed for String {
	fn make(n: u64) -> Self {
		Self::from_key_bytes(&bytes_of_key(n)).expect("ASCII")
	}

	fn name() -> &'static str {
		"String"
	}
}

fn value_of(n: u64) -> Vec<u8> {
	vec![(n % 251) as u8 + 1; 64 + ((n * 7_919) % 4_000) as usize]
}

fn ttl_of(n: u64) -> Option<u32> {
	(n % 3 == 0).then_some(3_600)
}

fn flat<K: Keyed>() -> PaperCache<K, BufferDRAM> {
	PaperCache::<K, BufferDRAM>::new(64 << 20, &[PaperPolicy::LfuCompact], PaperPolicy::LfuCompact)
		.expect("a flat cache")
}

/// A tiered cache with the byte gate on, `TIER` of fast tier.
fn tiered<K: Keyed>() -> PaperCache<K, TieredBuffer> {
	let cache = PaperCache::<K, TieredBuffer>::new_with_gate(
		256 << 20,
		CacheTierSize::Bytes(TIER),
		PaperPolicy::LruCompactHybrid,
		gated(Duration::from_secs(30), OnStall::Error),
	)
	.expect("a tiered cache");

	wait_for("the byte gate to enable", Duration::from_secs(10), || {
		cache.hybrid_stats().gate_state == gate::GateState::Enabled
	});

	cache
}

fn quiesce<K: Keyed>(cache: &PaperCache<K, TieredBuffer>) {
	let gate = cache.status.gate();
	let passes = gate.passes();

	wait_for("two worker passes", Duration::from_secs(10), || gate.passes() >= passes + 2);
	wait_for("the migrations to land", Duration::from_secs(10), || phys::pending_migrations() == (0, 0));
}

// ---------------------------------------------------------------------------
// the same entry, either spelling

const KEYS: u64 = 60;

/// Sets `KEYS` entries -- the even ones by owned key, the odd by borrowed --
/// then reads, sizes, re-TTLs and deletes them by both spellings, and checks
/// that each spelling sees what the other did. `set_owned`/`set_borrowed`
/// are the cache's own `set` and `set_borrowed`: this part is shared by the
/// flat and the tiered cache.
fn exercise<K: Keyed, V: crate::cache_shape::CacheShape>(
	cache: &PaperCache<K, V>,
	set_owned: impl Fn(K, &[u8], Option<u32>) -> Result<(), CacheError>,
	set_borrowed: impl Fn(&[u8], &[u8], Option<u32>) -> Result<(), CacheError>,
) {
	let name = K::name();

	for n in 0..KEYS {
		let (key, bytes) = (K::make(n), bytes_of_key(n));

		match n % 2 {
			0 => set_owned(key, &value_of(n), ttl_of(n)).expect("an owned set"),
			_ => set_borrowed(&bytes, &value_of(n), ttl_of(n)).expect("a borrowed set"),
		}
	}

	assert_eq!(cache.status().expect("status").num_objects(), KEYS, "{name}: every set made an entry");

	let mut out = vec![9u8; 3];

	for n in 0..KEYS {
		let (key, bytes) = (K::make(n), bytes_of_key(n));

		assert!(key.key_bytes() == &bytes[..], "{name}: the key is its bytes");

		// Every read, by both spellings.
		assert_eq!(cache.get(&key).expect("get"), value_of(n), "{name}: get {n}");
		assert_eq!(cache.get_borrowed(&bytes).expect("get_borrowed"), value_of(n), "{name}: get_borrowed {n}");

		assert_eq!(cache.peek(&key), cache.peek_borrowed(&bytes), "{name}: peek {n}");
		assert!(cache.has(&key) && cache.has_borrowed(&bytes), "{name}: has {n}");
		assert_eq!(cache.size(&key), cache.size_borrowed(&bytes), "{name}: size {n}");

		cache.get_into_borrowed(&bytes, &mut out).expect("get_into_borrowed");
		assert_eq!(out, value_of(n), "{name}: get_into_borrowed {n}");

		// Keys that are not it: a prefix, an extension, and one byte off.
		let mut longer = bytes.clone();
		longer.push(b'~');

		let mut off = bytes.clone();
		off[0] ^= 1;

		for other in [&bytes[..bytes.len() - 1], &longer[..], &off[..]] {
			assert!(!cache.has_borrowed(other), "{name}: {n}: {other:?} is not the key");
			assert_eq!(cache.get_borrowed(other), Err(CacheError::KeyNotFound), "{name}: {n}");
		}
	}

	assert_eq!(cache.get_borrowed(b"never-set"), Err(CacheError::KeyNotFound));
	assert_eq!(cache.size_borrowed(b"never-set"), Err(CacheError::KeyNotFound));
	assert_eq!(cache.ttl_borrowed(b"never-set", Some(5)), Err(CacheError::KeyNotFound));
	assert_eq!(cache.del_borrowed(b"never-set"), Err(CacheError::KeyNotFound));
	assert!(!cache.has_borrowed(b"never-set"));
	assert_eq!(cache.peek_borrowed(b"never-set"), Err(CacheError::KeyNotFound));

	// A TTL set by one spelling is seen by the other, and charged: an object with
	// an expiry is bigger than one without.
	for n in (0..KEYS).filter(|n| n % 5 == 1) {
		let (key, bytes) = (K::make(n), bytes_of_key(n));

		let before = cache.size(&key).expect("size");

		cache.ttl_borrowed(&bytes, Some(7_200)).expect("ttl_borrowed");

		let with = cache.size_borrowed(&bytes).expect("size_borrowed");

		assert_eq!(cache.size(&key), Ok(with), "{name}: {n}: the owned spelling sees the TTL");

		if ttl_of(n).is_none() {
			assert!(with > before, "{name}: {n}: an expiry is charged");
		}

		if ttl_of(n).is_some() {
			assert_eq!(with, before, "{name}: {n}: a TTL for one that had a TTL changes nothing");
		}

		cache.ttl(&key, None).expect("ttl");

		let plain = cache.size_borrowed(&bytes).expect("size_borrowed");

		assert!(plain < with, "{name}: {n}: and its removal by the owned spelling");

		if ttl_of(n).is_none() {
			assert_eq!(plain, before, "{name}: {n}: back to what it was");
		}
	}

	// Deleting by one spelling deletes it for the other; every third by bytes, every
	// third by key, and the rest stay.
	let mut left = KEYS;

	for n in 0..KEYS {
		let (key, bytes) = (K::make(n), bytes_of_key(n));

		match n % 3 {
			0 => {
				cache.del_borrowed(&bytes).expect("del_borrowed");
				assert_eq!(cache.del(&key), Err(CacheError::KeyNotFound), "{name}: {n}: gone for the owned spelling");
				left -= 1;
			},

			1 => {
				cache.del(&key).expect("del");
				assert_eq!(cache.del_borrowed(&bytes), Err(CacheError::KeyNotFound), "{name}: {n}: gone for the borrowed one");
				left -= 1;
			},

			_ => {},
		}

		assert_eq!(cache.has(&key), cache.has_borrowed(&bytes), "{name}: {n}");
		assert_eq!(cache.has(&key), n % 3 == 2, "{name}: {n}: only the third kind is left");
	}

	assert_eq!(cache.status().expect("status").num_objects(), left, "{name}: the deletes took their entries off the count");
	assert_eq!(cache.status().expect("status").num_objects(), (0..KEYS).filter(|n| n % 3 == 2).count() as u64);
}

fn flat_entries<K: Keyed>() {
	let cache = flat::<K>();

	exercise(&cache, |key, value, ttl| cache.set(key, value, ttl), |key, value, ttl| cache.set_borrowed(key, value, ttl));
}

/// The same entry is found by either spelling of its key: a key set by `set` is
/// read, sized, re-TTLed and deleted by its bytes and the other way round, in a
/// flat cache, for all three key types, and a key that is a prefix, an
/// extension or one byte off is not it. Red with a `String` hashed as its bytes
/// (`stringashash`: no borrowed lookup finds an owned set), with the comparison
/// only of a prefix (`prefixmatch`), and with `ttl_borrowed` charging nothing.
#[test]
fn an_entry_is_found_by_either_spelling_of_its_key() {
	flat_entries::<Box<[u8]>>();
	flat_entries::<Vec<u8>>();
	flat_entries::<String>();
}

fn tiered_entries<K: Keyed>() {
	let _m = test_hooks::override_m(M0);
	let cache = tiered::<K>();

	exercise(&cache, |key, value, ttl| cache.set(key, value, ttl), |key, value, ttl| cache.set_borrowed(key, value, ttl));

	// The tier an entry is in is the same question by either spelling; a value
	// larger than the whole fast tier is placed slow, and found as well.
	let big = vec![7u8; (TIER as usize) * 2];

	cache.set_borrowed(b"big-by-bytes", &big, None).expect("a structural slow set");
	cache.set(K::make(1_000), &big, None).expect("another");

	assert_eq!(cache.tier_of_borrowed(b"big-by-bytes"), Some(Slow));
	assert_eq!(cache.tier_of(&K::make(1_000)), cache.tier_of_borrowed(&bytes_of_key(1_000)));
	assert_eq!(cache.get_borrowed(b"big-by-bytes").expect("the slow value"), big);

	quiesce(&cache);

	for n in (0..KEYS).filter(|n| n % 3 == 2) {
		assert_eq!(cache.tier_of(&K::make(n)), cache.tier_of_borrowed(&bytes_of_key(n)), "{}: {n}", K::name());
		assert!(cache.tier_of_borrowed(&bytes_of_key(n)).is_some());
	}

	assert_eq!(cache.tier_of_borrowed(b"never-set"), None);
}

/// [`an_entry_is_found_by_either_spelling_of_its_key`] in a tiered cache, whose
/// set is the admission path, and whose entries are in a tier (`tier_of`).
#[test]
fn a_tiered_entry_is_found_by_either_spelling_of_its_key() {
	alone("a_tiered_entry_is_found_by_either_spelling_of_its_key", || {
		tiered_entries::<Box<[u8]>>();
		tiered_entries::<Vec<u8>>();
		tiered_entries::<String>();
	});
}

// ---------------------------------------------------------------------------
// the hash

fn hashes<K: Keyed>() {
	let flat = flat::<K>();

	for n in (0..KEYS).chain([1_000_000, u64::MAX]) {
		let (key, bytes) = (K::make(n), bytes_of_key(n));

		assert_eq!(flat.hash_key(&key), flat.hash_key_bytes(&bytes), "{}: {n}", K::name());
	}

	assert_eq!(flat.hash_key(&K::make(0)), flat.hash_key_bytes(&bytes_of_key(0)));

	// The empty key is a key.
	let empty = K::from_key_bytes(b"").expect("the empty key");

	assert_eq!(flat.hash_key(&empty), flat.hash_key_bytes(b""));
}

/// The hash of a borrowed key is the owned key's, in the cache's own hasher
/// (a `RandomState`, which differs per cache): so an entry is where either
/// spelling looks. (`key_bytes::tests` holds it for hashers that tell calls
/// apart.)
#[test]
fn a_borrowed_key_hashes_as_the_owned_key_does() {
	hashes::<Box<[u8]>>();
	hashes::<Vec<u8>>();
	hashes::<String>();
}

// ---------------------------------------------------------------------------
// a hash collision

/// A hasher that hashes everything to one value: every key collides, so the
/// comparison of the key -- not its hash -- is all that tells keys apart.
#[derive(Default)]
struct Constant;

impl std::hash::Hasher for Constant {
	fn write(&mut self, _bytes: &[u8]) {}

	fn finish(&self) -> u64 {
		7
	}
}

type Collide = std::hash::BuildHasherDefault<Constant>;

/// What a lookup by bytes must answer for keys that are not the stored one but
/// hash where it is stored.
fn assert_collisions_miss<K: Keyed, V: crate::cache_shape::CacheShape>(cache: &PaperCache<K, V, Collide>) {
	let name = K::name();
	let bytes = bytes_of_key(5);

	assert_eq!(cache.hash_key(&K::make(5)), cache.hash_key_bytes(b"another key"), "{name}: every key hashes alike");

	let mut longer = bytes.clone();
	longer.push(b'~');

	let mut off = bytes.clone();
	off[0] ^= 1;

	let before = cache.size_borrowed(&bytes).expect("the stored key's size");
	let mut out = vec![1u8];

	for other in [&bytes[..bytes.len() - 1], &longer[..], &off[..], &b""[..]] {
		assert!(!cache.has_borrowed(other), "{name}: {other:?} is not the key");
		assert_eq!(cache.get_borrowed(other), Err(CacheError::KeyNotFound), "{name}: {other:?}");
		assert_eq!(cache.peek_borrowed(other), Err(CacheError::KeyNotFound), "{name}: {other:?}");
		assert_eq!(cache.size_borrowed(other), Err(CacheError::KeyNotFound), "{name}: {other:?}");
		assert_eq!(cache.ttl_borrowed(other, Some(9)), Err(CacheError::KeyNotFound), "{name}: {other:?}");
		assert_eq!(cache.get_into_borrowed(other, &mut out), Err(CacheError::KeyNotFound), "{name}: {other:?}");
		assert_eq!(cache.del_borrowed(other), Err(CacheError::KeyNotFound), "{name}: {other:?}");
	}

	// The stored key is as it was: there, by both spellings, with no TTL set by a
	// key that is not it, and not deleted by one.
	assert_eq!(out, [1u8], "a miss leaves the caller's buffer alone");
	assert_eq!(cache.get_borrowed(&bytes).expect("still there"), [1, 2, 3], "{name}");
	assert_eq!(cache.get(&K::make(5)).expect("still there"), [1, 2, 3], "{name}");
	assert_eq!(cache.size_borrowed(&bytes), Ok(before), "{name}: and no TTL was set on it");
	assert_eq!(cache.status().expect("status").num_objects(), 1, "{name}");
}

/// Keys that collide are told apart by their bytes, whichever way they are
/// asked: with every key hashed alike, no lookup, TTL or delete by bytes finds
/// or touches the one stored key unless its bytes are the key's -- not a prefix
/// of it, an extension, one byte off or empty. Red with the comparison a prefix
/// test (`prefixmatch`) or none (`matchany`).
#[test]
fn a_borrowed_key_that_collides_is_not_the_key() {
	fn flat_collisions<K: Keyed>() {
		let cache = PaperCache::<K, BufferDRAM, Collide>::with_hasher(
			64 << 20,
			&[PaperPolicy::LfuCompact],
			PaperPolicy::LfuCompact,
			Collide::default(),
		)
		.expect("a flat cache");

		cache.set_borrowed(&bytes_of_key(5), &[1, 2, 3], None).expect("a set");
		assert_collisions_miss(&cache);
	}

	flat_collisions::<Box<[u8]>>();
	flat_collisions::<Vec<u8>>();
	flat_collisions::<String>();
}

/// `a_borrowed_key_that_collides_is_not_the_key` in a tiered cache.
#[test]
fn a_tiered_borrowed_key_that_collides_is_not_the_key() {
	alone("a_tiered_borrowed_key_that_collides_is_not_the_key", || {
		let _m = test_hooks::override_m(M0);

		fn tiered_collisions<K: Keyed>() {
			let cache = PaperCache::<K, TieredBuffer, Collide>::with_hasher(
				256 << 20,
				CacheTierSize::Bytes(TIER),
				PaperPolicy::LruCompactHybrid,
				Collide::default(),
			)
			.expect("a tiered cache");

			cache.set_borrowed(&bytes_of_key(5), &[1, 2, 3], None).expect("a set");
			assert_collisions_miss(&cache);
		}

		tiered_collisions::<Box<[u8]>>();
		tiered_collisions::<Vec<u8>>();
		tiered_collisions::<String>();
	});
}

// ---------------------------------------------------------------------------
// the figures

fn figures<K: Keyed>() {
	let cache = flat::<K>();

	for n in 0..KEYS {
		let (key, bytes) = (K::make(n), bytes_of_key(n));

		for len in [1usize, 17, 64, 100, 1_000, 4_096, 5_000, 70_000] {
			for ttl in [None, Some(0), Some(60)] {
				let by_key = cache.overhead_manager.base_size_for(&key, len, ttl);
				let by_bytes = cache.overhead_manager.base_size_with(
					TieredValue::<K>::key_accounted_size_for_bytes(bytes.len()),
					TieredValue::<K>::item_prefix_bytes_for_bytes(bytes.len()),
					len,
					ttl,
				);

				assert_eq!(by_bytes, by_key, "{}: {n}: base size of {len} B, ttl {ttl:?}", K::name());

				// And it is the base size of the object the set builds.
				let object = Object::<K, BufferDRAM>::new_in_bytes(&bytes, &vec![1u8; len], Fast, ttl);

				assert_eq!(Some(cache.overhead_manager.base_size(&object)), by_key, "{}: {n}: {len} B", K::name());
				assert!(object.key_matches_bytes(&bytes));
				assert!(object.key_matches(&key));
			}

			assert_eq!(
				cache.overhead_manager.dram_resident_size_with(TieredValue::<K>::key_accounted_size_for_bytes(bytes.len()), Some(60)),
				cache.overhead_manager.dram_resident_size_for(&key, Some(60)),
			);

			assert_eq!(
				phys::value_charge_with(TieredValue::<K>::item_prefix_bytes_for_bytes(bytes.len()), len as u32),
				phys::value_charge_for(&key, len as u32),
				"{}: {n}: P's charge for {len} B",
				K::name(),
			);
		}
	}
}

/// What admission is given for a borrowed key -- the item's prefix and the
/// key's accounted size, from its length -- are what the key itself gives, so
/// the sizes, the DRAM residency and P's charge of a borrowed set are the
/// owned set's. Red with a prefix of 0 (`prefixzero`) or a key size of
/// `size_of::<K>()` (`keysizeheader`).
#[test]
fn the_figures_of_a_borrowed_key_are_the_keys() {
	figures::<Box<[u8]>>();
	figures::<Vec<u8>>();
	figures::<String>();
}

/// A value built from borrowed bytes -- in one go (`new_in_bytes`), or
/// allocated and then filled (`new_uninit_in_bytes`) -- is the value built from
/// the key: the key, the bytes, the expiry and the item's prefix, in either tier.
fn built_values<K: Keyed>() {
	for tier in [Fast, Slow] {
		for n in 0..KEYS {
			let (key, bytes, value) = (K::make(n), bytes_of_key(n), value_of(n));
			let expiry = crate::object::expiry_from_ttl(ttl_of(n));

			let owned = TieredValue::<K>::new_in(key.clone(), &value, tier, expiry);
			let direct = TieredValue::<K>::new_in_bytes(&bytes, &value, tier, expiry);

			let mut uninit = TieredValue::<K>::new_uninit_in_bytes(&bytes, value.len(), tier);

			for (to, from) in uninit.bytes_mut().iter_mut().zip(&value) {
				to.write(*from);
			}

			// SAFETY: every byte of the value was just written.
			let filled = unsafe { uninit.assume_init(expiry) };

			for built in [&direct, &filled] {
				assert_eq!(built.key_owned(), key, "{}: {tier:?}: {n}", K::name());
				assert!(built.key_matches_bytes(&bytes) && built.key_matches(&key));
				assert_eq!(built.bytes(), owned.bytes());
				assert_eq!(built.expiry(), owned.expiry());
				assert_eq!(built.tier(), tier);
				assert_eq!(built.item_prefix_bytes(), owned.item_prefix_bytes());
				assert_eq!(built.key_accounted_size(), owned.key_accounted_size());
			}

			assert!(!filled.key_matches_bytes(&bytes_of_key(n + 1)));
		}
	}
}

/// See `built_values`.
#[test]
fn a_value_built_from_borrowed_bytes_is_the_value_built_from_the_key() {
	alone("a_value_built_from_borrowed_bytes_is_the_value_built_from_the_key", || {
		built_values::<Box<[u8]>>();
		built_values::<Vec<u8>>();
		built_values::<String>();

		assert_eq!(phys::fast_bytes_signed(), 0, "every value built here was freed, and refunded");
	});
}

// ---------------------------------------------------------------------------
// a set is a set

#[derive(Clone, Copy, Debug)]
enum By {
	/// `set`, with the owned key.
	Owned,

	/// `set_borrowed`.
	Borrowed,

	/// `reserve_set_borrowed`, `fill`, `write`, `commit`.
	Permit,
}

/// Everything caches that did the same sets must agree on, once idle.
#[derive(Debug, PartialEq)]
struct Outcome {
	objects: Vec<(u64, Option<Tier>, Vec<u8>, ObjectSize)>,
	live: u64,
	used: u64,
	fast_objects: u64,
	slow_objects: u64,
	fast_bytes_used: u64,
	slow_bytes_used: u64,
	structural_slow_sets: u64,
	p: i64,
}

fn run<K: Keyed>(by: By) -> Outcome {
	let cache = tiered::<K>();

	for n in 0..200u64 {
		let (key, bytes, value, ttl) = (K::make(n), bytes_of_key(n), value_of(n), ttl_of(n));

		// Every tenth is larger than the fast tier: structural slow placement.
		let value = if n % 10 == 7 { vec![3u8; TIER as usize + 1_000] } else { value };

		match by {
			By::Owned => cache.set(key, &value, ttl).expect("set"),
			By::Borrowed => cache.set_borrowed(&bytes, &value, ttl).expect("set_borrowed"),

			By::Permit => {
				let permit = cache.reserve_set_borrowed(&bytes, value.len(), ttl, soon()).expect("a permit");

				assert_eq!(permit.len(), value.len());

				let mut pending = permit.fill();

				match n % 2 {
					0 => pending.read_exact_from(&mut &value[..]).expect("the body"),
					_ => pending.write(&value),
				}

				pending.commit().expect("commit");
			},
		}
	}

	quiesce(&cache);

	let stats = cache.hybrid_stats();
	let mut objects = Vec::new();

	for n in 0..200u64 {
		let (key, bytes) = (K::make(n), bytes_of_key(n));

		objects.push((
			n,
			cache.tier_of(&key),
			cache.peek(&key).expect("a live key"),
			cache.size(&key).expect("a size"),
		));

		assert_eq!(cache.tier_of_borrowed(&bytes), objects[n as usize].1);
	}

	let outcome = Outcome {
		objects,
		live: cache.status.live_num_objects(),
		used: cache.status.used_size(&PaperPolicy::LruCompactHybrid),
		fast_objects: stats.fast_objects,
		slow_objects: stats.slow_objects,
		fast_bytes_used: stats.fast_bytes_used,
		slow_bytes_used: stats.slow_bytes_used,
		structural_slow_sets: stats.structural_slow_sets,
		p: phys::fast_bytes_signed(),
	};

	assert_eq!(outcome.p as u64, outcome.fast_bytes_used, "P is the stack's fast bytes ({by:?})");
	assert!(outcome.fast_objects > 0 && outcome.slow_objects > 0, "both tiers hold keys: {outcome:?}");
	assert!(outcome.structural_slow_sets > 0, "the big values were placed slow: {outcome:?}");

	drop(cache);
	assert_eq!(phys::fast_bytes_signed(), 0, "every allocation was refunded when the cache dropped ({by:?})");

	outcome
}

/// A set by a borrowed key is a set: the same 200 sets -- varying in length,
/// every third with a TTL, every tenth larger than the whole fast tier -- by
/// `set`, by `set_borrowed` and by `reserve_set_borrowed` + `fill` + `commit`,
/// into a gated cache whose tier is a fraction of the data, leave the same
/// objects (bytes, tier, size), the same status (count and used bytes), the
/// same stack and the same P, for all three key types. Red with the item's
/// prefix of a borrowed key taken as 0 (`prefixzero`), with the charge of
/// P taken from anything but the allocation (`nocharge`), and with the key
/// size taken as the header's (`keysizeheader`).
#[test]
fn a_borrowed_set_is_a_set() {
	alone("a_borrowed_set_is_a_set", || {
		let _m = test_hooks::override_m(M0);

		let a = run::<Box<[u8]>>(By::Owned);

		assert_eq!(a, run::<Box<[u8]>>(By::Borrowed), "Box<[u8]>: set_borrowed is not set");
		assert_eq!(a, run::<Box<[u8]>>(By::Permit), "Box<[u8]>: the permit is not set");

		let a = run::<Vec<u8>>(By::Owned);

		assert_eq!(a, run::<Vec<u8>>(By::Borrowed), "Vec<u8>: set_borrowed is not set");
		assert_eq!(a, run::<Vec<u8>>(By::Permit), "Vec<u8>: the permit is not set");

		let a = run::<String>(By::Owned);

		assert_eq!(a, run::<String>(By::Borrowed), "String: set_borrowed is not set");
		assert_eq!(a, run::<String>(By::Permit), "String: the permit is not set");
	});
}

// ---------------------------------------------------------------------------
// allocations

#[cfg(not(feature = "stock_jemalloc"))]
mod allocations {
	use super::*;

	use crate::numa_alloc::thread_allocs;

	const RUNS: u64 = 80;

	/// The key `n` of the `family`, as bytes: fresh for every call, 30 bytes, so
	/// that a key buffer is a real allocation.
	fn fresh(family: u64, n: u64) -> Vec<u8> {
		format!("fresh-key-{family:02}-{n:06}-xxxxxxxx").into_bytes()
	}

	/// The fewest allocations `call` made on this thread, with the bytes of a
	/// fresh key of the `family`, in any of `RUNS` calls. The key is made before
	/// the count starts. The most a call can allocate beyond its own is a block
	/// of the workers' channel (one in 31 sends) or a step of the object map;
	/// the fewest is what the path itself needs.
	fn fewest(family: u64, mut call: impl FnMut(&[u8])) -> u64 {
		(0..RUNS)
			.map(|run| {
				let key = fresh(family, run);
				let before = thread_allocs();

				call(&key);

				thread_allocs() - before
			})
			.min()
			.expect("runs")
	}

	fn counts<K: Keyed, V: crate::cache_shape::CacheShape>(
		cache: &PaperCache<K, V>,
		set_owned: impl Fn(K, &[u8]),
		set_borrowed: impl Fn(&[u8], &[u8]),
		thin: bool,
	) {
		let value = vec![5u8; 300];

		// A set: by the owned key built from the bytes at the call (the key
		// buffer is one allocation), and by the bytes.
		let owned = fewest(1, |bytes| {
			let key = K::from_key_bytes(bytes).expect("a key");

			set_owned(key, &value);
		});

		let borrowed = fewest(2, |bytes| set_borrowed(bytes, &value));

		if thin {
			assert_eq!(borrowed + 1, owned, "{}: under thin_header a borrowed set allocates no key", K::name());
		} else {
			assert_eq!(borrowed, owned, "{}: under the default layout a borrowed set builds the one key the header stores", K::name());
		}

		// A get: the key is not stored, so by the bytes nothing is built at all.
		let owned = fewest(1, |bytes| {
			let key = K::from_key_bytes(bytes).expect("a key");

			cache.get(&key).expect("a hit");
		});

		let borrowed = fewest(1, |bytes| {
			cache.get_borrowed(bytes).expect("a hit");
		});

		assert_eq!(borrowed + 1, owned, "{}: a borrowed get builds no key", K::name());

		// And the others that look an entry up.
		let owned = fewest(2, |bytes| {
			let key = K::from_key_bytes(bytes).expect("a key");

			assert!(cache.has(&key));
			assert!(cache.size(&key).is_ok());
		});

		let borrowed = fewest(2, |bytes| {
			assert!(cache.has_borrowed(bytes));
			assert!(cache.size_borrowed(bytes).is_ok());
		});

		assert_eq!(borrowed + 1, owned, "{}: a borrowed has and size build no key", K::name());
	}

	fn flat_counts<K: Keyed>() {
		let cache = flat::<K>();

		counts(
			&cache,
			|key, value| cache.set(key, value, None).expect("set"),
			|key, value| cache.set_borrowed(key, value, None).expect("set_borrowed"),
			cfg!(feature = "thin_header"),
		);
	}

	/// What a flat cache's borrowed set and lookups allocate, against the same
	/// calls with the key built first: one allocation fewer (the key buffer) in
	/// every borrowed lookup, and under `thin_header` in the set too -- where
	/// the item holds the key as bytes. Under the default layout the set builds
	/// the key the header stores, once, so it allocates what `set` does. Red
	/// with the borrowed paths building an owned key to call the owned ones
	/// (`buildkey`).
	#[test]
	fn the_borrowed_paths_allocate_no_key_they_do_not_need_in_a_flat_cache() {
		flat_counts::<Box<[u8]>>();
		flat_counts::<Vec<u8>>();
		flat_counts::<String>();
	}

	fn tiered_counts<K: Keyed>() {
		let cache = tiered::<K>();

		counts(
			&cache,
			|key, value| cache.set(key, value, None).expect("set"),
			|key, value| cache.set_borrowed(key, value, None).expect("set_borrowed"),
			cfg!(feature = "thin_header"),
		);

		// The three steps by a borrowed key allocate what `set_borrowed` does.
		let value = vec![5u8; 300];
		let by_set = fewest(3, |bytes| cache.set_borrowed(bytes, &value, None).expect("set_borrowed"));

		let by_permit = fewest(4, |bytes| {
			let permit = cache.reserve_set_borrowed(bytes, value.len(), None, soon()).expect("a permit");
			let mut pending = permit.fill();

			pending.write(&value);
			pending.commit().expect("commit");
		});

		assert_eq!(by_permit, by_set, "{}: the permit is the set", K::name());

		// A set the cache refuses builds nothing, under either layout: the key is
		// built when the value is allocated, not when the set is admitted.
		let refused = fewest(5, |bytes| {
			assert!(matches!(
				cache.reserve_set_borrowed(bytes, usize::MAX >> 8, None, soon()),
				Err(CacheError::ExceedingValueSize),
			));
		});

		assert_eq!(refused, 0, "{}: a refused borrowed set allocates nothing", K::name());

		let abandoned = fewest(6, |bytes| {
			drop(cache.reserve_set_borrowed(bytes, 300, None, soon()).expect("a permit"));
		});

		assert_eq!(abandoned, 0, "{}: an abandoned permit allocates nothing", K::name());
	}

	/// The tiered cache's: the same, and the three steps, and a refused or
	/// abandoned reservation allocates nothing at all -- under the default
	/// layout too, which builds its key only when the value is allocated.
	#[test]
	fn the_borrowed_paths_allocate_no_key_they_do_not_need_in_a_tiered_cache() {
		alone("allocations::the_borrowed_paths_allocate_no_key_they_do_not_need_in_a_tiered_cache", || {
			let _m = test_hooks::override_m(M0);

			tiered_counts::<Box<[u8]>>();
			tiered_counts::<Vec<u8>>();
			tiered_counts::<String>();
		});
	}
}

// ---------------------------------------------------------------------------
// keys no cache holds

/// A `String` cache holds no key that is not UTF-8: a set of one is refused
/// (`InvalidKey`, before anything is allocated or counted), a lookup of one is a
/// miss, and a cache of byte keys takes any bytes. Red with the check missing
/// (`utf8unchecked`: a `String` is built over bytes that are not a string).
#[test]
fn a_string_cache_refuses_a_key_that_is_not_utf8() {
	alone("a_string_cache_refuses_a_key_that_is_not_utf8", || {
		let _m = test_hooks::override_m(M0);
		let bad: &[u8] = &[b'a', 0xff, 0xfe, b'z'];

		// One tiered cache at a time: the byte gate disables itself beside another.
		{
			let strings = tiered::<String>();

			assert_eq!(strings.set_borrowed(bad, &[1, 2, 3], None), Err(CacheError::InvalidKey));
			assert!(matches!(strings.reserve_set_borrowed(bad, 3, None, soon()), Err(CacheError::InvalidKey)));

			assert!(!strings.has_borrowed(bad));
			assert_eq!(strings.get_borrowed(bad), Err(CacheError::KeyNotFound));
			assert_eq!(strings.peek_borrowed(bad), Err(CacheError::KeyNotFound));
			assert_eq!(strings.size_borrowed(bad), Err(CacheError::KeyNotFound));
			assert_eq!(strings.ttl_borrowed(bad, Some(3)), Err(CacheError::KeyNotFound));
			assert_eq!(strings.del_borrowed(bad), Err(CacheError::KeyNotFound));
			assert_eq!(strings.tier_of_borrowed(bad), None);

			assert_eq!(strings.status().expect("status").num_objects(), 0, "nothing was inserted");
			assert_eq!(phys::fast_bytes_signed(), 0, "and nothing is charged");

			// The empty key, and a key that is valid UTF-8 and not ASCII, are keys.
			strings.set_borrowed(b"", &[9], None).expect("the empty key");
			strings.set_borrowed("caf\u{e9}".as_bytes(), &[8], None).expect("a non-ASCII key");

			assert_eq!(strings.get(&String::new()).expect("the empty key"), [9]);
			assert_eq!(strings.get_borrowed("caf\u{e9}".as_bytes()).expect("non-ASCII"), [8]);
			assert_eq!(strings.get(&String::from("caf\u{e9}")).expect("non-ASCII"), [8]);
		}

		// Bytes of a key that is not a string are a key of the byte caches.
		{
			let boxes = tiered::<Box<[u8]>>();

			boxes.set_borrowed(bad, &[1, 2, 3], None).expect("any bytes");

			assert_eq!(boxes.get_borrowed(bad).expect("a hit"), [1, 2, 3]);
			assert_eq!(boxes.get(&Box::<[u8]>::from(bad)).expect("a hit"), [1, 2, 3]);
		}

		// A flat cache has no gate: the same, for all three.
		let flat_strings = flat::<String>();
		let vecs = flat::<Vec<u8>>();

		assert_eq!(flat_strings.set_borrowed(bad, &[1, 2, 3], None), Err(CacheError::InvalidKey));
		assert!(!flat_strings.has_borrowed(bad));
		assert_eq!(flat_strings.status().expect("status").num_objects(), 0);

		vecs.set_borrowed(bad, &[4], None).expect("any bytes");
		assert_eq!(vecs.get(&bad.to_vec()).expect("a hit"), [4]);
	});
}

// ---------------------------------------------------------------------------
// abandoning

/// A permit or a pending set of a borrowed key that is dropped leaves nothing
/// behind: no entry, no count, P as it was (the allocation was charged when it
/// was made, and is refunded once), for each key type. Red with the charge of
/// a borrowed allocation not made (`nocharge`: P goes below where it was) or
/// not refunded (`norefund`).
#[test]
fn an_abandoned_borrowed_set_is_refunded_once() {
	alone("an_abandoned_borrowed_set_is_refunded_once", || {
		fn check<K: Keyed>() {
			let cache = tiered::<K>();
			let key = bytes_of_key(5);
			let p0 = p();

			// A permit dropped before its fill: nothing allocated.
			let permit = cache.reserve_set_borrowed(&key, LEN, None, soon()).expect("a permit");
			assert_eq!(permit.tier(), Fast);
			drop(permit);

			assert_eq!(p(), p0, "{}: a permit allocates nothing", K::name());

			// A pending set dropped, unwritten, half written, or written whole.
			for written in [0, LEN / 2, LEN] {
				let permit = cache.reserve_set_borrowed(&key, LEN, None, soon()).expect("a permit");
				let mut pending = permit.fill();

				assert!(p() > p0, "{}: the allocation is charged to P", K::name());

				pending.write(&vec![1u8; written]);
				drop(pending);

				assert_eq!(p(), p0, "{}: refunded once ({written} written)", K::name());
				assert_eq!(cache.status().expect("status").num_objects(), 0, "{}: nothing inserted", K::name());
				assert!(!cache.has_borrowed(&key));
			}

			// And the key is still set by the same bytes after.
			cache.set_borrowed(&key, &vec![2u8; LEN], None).expect("a set");
			assert_eq!(cache.get_borrowed(&key).expect("a hit"), vec![2u8; LEN]);
			assert_eq!(cache.status().expect("status").num_objects(), 1);
		}

		let _m = test_hooks::override_m(M0);

		check::<Box<[u8]>>();
		check::<Vec<u8>>();
		check::<String>();
	});
}

// ---------------------------------------------------------------------------
// the deadline

/// A borrowed reservation honours its deadline as `reserve_set` does: against
/// a tier held full, one with a deadline 300 ms off fails at it with the bytes
/// lane's error -- not the watchdog's -- and one whose deadline has passed
/// fails at once; neither leaves a reservation, a charge or an entry.
#[test]
fn a_borrowed_reservation_honours_its_deadline() {
	alone("a_borrowed_reservation_honours_its_deadline", || {
		let _flush = test_hooks::no_flush();
		let _m = test_hooks::override_m(M0);
		let cache = tiered::<Box<[u8]>>();
		let gate = cache.status.gate();
		let _pause = test_hooks::pause_consumers();

		// Nine-byte keys, so the charge of one value is the same for all of them.
		let key = |n: u64| format!("k{n:08}").into_bytes();
		let v = phys::value_charge_with(TieredValue::<Box<[u8]>>::item_prefix_bytes_for_bytes(9), LEN as u32);
		let b = cache.hybrid_stats().band_b;

		assert!(b > 0, "the gate publishes its close level");

		let mut next = 0u64;

		while p() + v <= b {
			assert!(next < 10_000, "P never reached the close level");
			cache.set_borrowed(&key(next), &vec![next as u8; LEN], None).expect("a fill set is admitted at once");
			next += 1;
		}

		let (p0, count) = (p(), cache.status.live_num_objects());

		// (a) A deadline 300 ms off.
		let started = Instant::now();
		let next_key = key(next);
		let result = cache.reserve_set_borrowed(&next_key, LEN, None, started + Duration::from_millis(300));
		let took = started.elapsed();

		assert!(matches!(result, Err(CacheError::FastTierStalled)), "(a): {:?}", result.err());
		assert!(took >= Duration::from_millis(300), "(a): gave up after {took:?}, before its deadline");
		assert!(took < Duration::from_millis(300) + Duration::from_secs(2), "(a): gave up after {took:?}");
		assert_eq!(cache.hybrid_stats().gate_stall_errors, 0, "(a): the deadline's error is not the watchdog's");

		// (b) A deadline that has already passed.
		let past = Instant::now();
		thread::sleep(Duration::from_millis(2));

		let started = Instant::now();
		let result = cache.reserve_set_borrowed(&next_key, LEN, None, past);

		assert!(matches!(result, Err(CacheError::FastTierStalled)), "(b): {:?}", result.err());
		assert!(started.elapsed() < Duration::from_millis(200), "(b): waited {:?} though out of time", started.elapsed());

		assert_eq!((gate.reserved(), p(), cache.status.live_num_objects()), (0, p0, count), "nothing left behind");
		assert!(!cache.has_borrowed(&key(next)));
	});
}
