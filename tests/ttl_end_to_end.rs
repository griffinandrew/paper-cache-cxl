/*
 * Copyright (c) Kia Shakiba
 *
 * This source code is licensed under the GNU AGPLv3 license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! End-to-end coverage that a TTL actually expires an object.
//!
//! The existing tests cover the two halves separately and neither catches a
//! break in the whole path: `expiries.rs` drives `Expiries` directly without a
//! cache, and `ttl_is_preserved_across_a_set` only checks that a TTL survives
//! a `set`, never that it fires. So a cache that accepts TTLs and silently
//! never expires anything passes both.
//!
//! There are two independent pathways and they are tested separately, because
//! either can work while the other is broken:
//!
//!   * the **read** path -- `Object::is_expired` filters a lapsed object out
//!     of a `get`, whether or not any worker has run;
//!   * the **reap** path -- `TtlWorker` pops the deadline, erases the object
//!     from the map and notifies `PolicyWorker`, which is what actually
//!     returns the bytes and the object-map slot.
//!
//! Run with nightly:
//!   cargo +nightly test --test ttl_end_to_end --features lru_compact_hybrid_cache

#![cfg(feature = "lru_compact_hybrid_cache")]

use std::{thread, time::Duration};

use paper_cache::{CacheTierSize, PaperCache, PaperPolicy, TieredBuffer};

/// Long enough for `TtlWorker` to notice. Its loop sleeps 1 ms while anything
/// is due within 2 s and 1000 ms otherwise, so a second past the deadline is
/// already generous; five is slack for a loaded machine.
const SETTLE: Duration = Duration::from_secs(5);

fn cache() -> PaperCache<u32, TieredBuffer> {
	PaperCache::<u32, TieredBuffer>::new(
		1_000_000,
		CacheTierSize::Bytes(500_000),
		PaperPolicy::LruCompactHybrid,
	)
	.expect("cache should construct")
}

#[test]
fn an_object_is_readable_before_its_ttl_elapses() {
	let cache = cache();

	cache.set(1u32, b"value", Some(60)).expect("set should succeed");

	assert_eq!(
		cache.get(&1u32).expect("a live object should be returned"),
		b"value",
	);
}

/// The read path. Independent of any worker: `is_expired` compares the stored
/// tick against `now_ticks()` on every read.
#[test]
fn an_expired_object_is_not_returned_by_a_read() {
	let cache = cache();

	cache.set(1u32, b"value", Some(1)).expect("set should succeed");
	assert!(cache.get(&1u32).is_ok(), "should hit before the ttl elapses");

	thread::sleep(SETTLE);

	assert!(
		cache.get(&1u32).is_err(),
		"a lapsed object must not be served: Object::is_expired did not fire",
	);
}

/// The reap path. This is the one that returns the bytes and the object-map
/// slot; without it a cache can look correct to readers while every expired
/// object still occupies capacity.
#[test]
fn the_ttl_worker_reaps_an_expired_object_from_the_object_map() {
	let cache = cache();

	cache.set(1u32, b"value", Some(1)).expect("set should succeed");

	let before = cache.status().expect("status").num_objects();
	assert_eq!(before, 1, "the object should be resident before it expires");

	thread::sleep(SETTLE);

	let after = cache.status().expect("status").num_objects();

	assert_eq!(
		after, 0,
		"TtlWorker did not reap the expired object: num_objects went {before} -> {after}",
	);
}

/// The bytes have to come back too, not just the count.
#[test]
fn reaping_returns_the_objects_bytes_to_the_cache() {
	let cache = cache();

	for key in 0u32..1_000 {
		cache.set(key, &[0u8; 256], Some(1)).expect("set should succeed");
	}

	let before = cache.status().expect("status").used_size();
	assert!(before > 0, "the cache should be holding bytes");

	thread::sleep(SETTLE);

	let after = cache.status().expect("status").used_size();

	assert_eq!(
		cache.status().expect("status").num_objects(),
		0,
		"every object had the same ttl and all should have been reaped",
	);

	assert!(
		after < before,
		"used_size did not fall after reaping: {before} -> {after}",
	);
}

/// A control. If this fails alongside the others, objects are disappearing for
/// some reason other than expiry and the tests above prove nothing.
#[test]
fn an_object_with_no_ttl_is_never_reaped() {
	let cache = cache();

	cache.set(1u32, b"value", None).expect("set should succeed");

	thread::sleep(SETTLE);

	assert_eq!(
		cache.status().expect("status").num_objects(),
		1,
		"an object with no ttl must survive",
	);
	assert!(cache.get(&1u32).is_ok(), "and must still be readable");
}

/// A TTL far in the future must not be treated as already lapsed -- the
/// failure mode if the tick arithmetic ever wraps or compares the wrong way
/// round.
#[test]
fn a_distant_ttl_does_not_expire_early() {
	let cache = cache();

	cache.set(1u32, b"value", Some(86_400)).expect("set should succeed");

	thread::sleep(SETTLE);

	assert_eq!(
		cache.status().expect("status").num_objects(),
		1,
		"a day-long ttl must not fire within seconds",
	);
	assert!(cache.get(&1u32).is_ok());
}

/// The benchmark expires nothing while these tests pass, and the one
/// difference is sustained load. If this reproduces, the defect is in the
/// worker under a live event stream rather than in the expiry logic.
#[test]
fn expiry_still_fires_while_the_cache_is_under_read_load() {
	use std::sync::{Arc, atomic::{AtomicBool, Ordering}};

	let cache = Arc::new(cache_big());

	for key in 0u32..100_000 {
		cache.set(key, &[0u8; 512], Some(2)).expect("set should succeed");
	}

	let before = cache.status().expect("status").num_objects();
	assert_eq!(before, 100_000, "all objects should be resident");

	// Hammer reads the way the benchmark's replay loop does.
	let stop = Arc::new(AtomicBool::new(false));
	let readers: Vec<_> = (0..2)
		.map(|_| {
			let cache = Arc::clone(&cache);
			let stop = Arc::clone(&stop);
			std::thread::spawn(move || {
				let mut k = 0u32;
				while !stop.load(Ordering::Relaxed) {
					let _ = cache.get(&(k % 100_000));
					k = k.wrapping_add(1);
				}
			})
		})
		.collect();

	std::thread::sleep(Duration::from_secs(10));
	stop.store(true, Ordering::Relaxed);
	for r in readers {
		r.join().unwrap();
	}

	let after = cache.status().expect("status").num_objects();

	assert_eq!(
		after, 0,
		"under read load the TtlWorker reaped nothing: {before} -> {after}",
	);
}

fn cache_big() -> PaperCache<u32, TieredBuffer> {
	PaperCache::<u32, TieredBuffer>::new(
		1_000_000_000,
		CacheTierSize::Bytes(250_000_000),
		PaperPolicy::LruCompactHybrid,
	)
	.expect("cache should construct")
}
