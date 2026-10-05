/*
 * Copyright (c) Griffin Andrew
 *
 * This source code is licensed under the GNU AGPLv3 license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! M counts the heap bytes behind a byte-string key (`meta::DramMetadata::keys`).
//!
//! Under the default layout a `String`, `Vec<u8>` or `Box<[u8]>` key is a `K`
//! in the DRAM header (counted: `headers`) that points at an allocation of its
//! own, which nothing counted. It is now charged `nallocx(len)` when its object
//! enters the map and refunded when it leaves. Under `thin_header` the key is
//! inside the item (counted with the value, P) and M charges nothing; a `u64`
//! key is inline in either layout.
//!
//! One file for every build: the figure expected is `charge(len)`, which is 0
//! under `thin_header`. What each test holds:
//!
//! * a new key raises M's `keys` by exactly its size class, per key type, and a
//!   `u64` key by nothing (`a_new_key_is_charged_its_size_class`);
//! * delete, capacity eviction, TTL expiry, overwrite and a wipe refund it
//!   (`every_way_out_refunds_the_charge`);
//! * migration copies the key and charges nothing
//!   (`migration_does_not_charge_the_copied_key`);
//! * a set refused, or abandoned before its fill, charges nothing
//!   (`a_refused_or_abandoned_set_charges_nothing`).

use std::time::{Duration, Instant};

use super::test_support::wait_for;

use crate::key_bytes::KeyBytes;
use crate::meta::usable;
use crate::{CacheTierSize, PaperCache, PaperPolicy, TieredBuffer};

/// What one key of `len` bytes is charged in this build.
fn charge(len: usize) -> u64 {
	match cfg!(feature = "thin_header") {
		true => 0,
		false => usable(len, 1),
	}
}

trait Keyed: KeyBytes + std::fmt::Debug {
	fn name() -> &'static str;
}

impl Keyed for Box<[u8]> {
	fn name() -> &'static str {
		"Box<[u8]>"
	}
}

impl Keyed for Vec<u8> {
	fn name() -> &'static str {
		"Vec<u8>"
	}
}

impl Keyed for String {
	fn name() -> &'static str {
		"String"
	}
}

/// Key `n`, `len` bytes: ASCII, so every key type holds it.
fn bytes(n: u64, len: usize) -> Vec<u8> {
	let mut key = format!("{n:08}").into_bytes();
	key.resize(len.max(8), b'k');
	key
}

fn key<K: Keyed>(n: u64, len: usize) -> K {
	K::from_key_bytes(&bytes(n, len)).expect("ASCII")
}

fn tiered<K: Keyed>(max: u64, tier: u64) -> PaperCache<K, TieredBuffer> {
	PaperCache::<K, TieredBuffer>::new(max, CacheTierSize::Bytes(tier), PaperPolicy::LruCompactHybrid)
		.expect("a tiered cache")
}

fn soon() -> Duration {
	Duration::from_secs(60)
}

/// The counter, the published `keys` and the total, once the worker has
/// published `want`: they agree at quiescence.
fn settled(status: &crate::status::AtomicStatus, want: u64, what: &str) {
	wait_for(what, soon(), || status.dram_metadata().keys == want);

	assert_eq!(status.key_heap_bytes(), want, "{what}: the status's count");

	let m = status.dram_metadata();

	assert_eq!(m.keys, want, "{what}: the published keys");
	assert_eq!(m.total(), m.map + m.stack + m.headers + m.keys, "{what}: keys are in M");
}

/// Waits for the live count to stop moving (evictions and migrations settle).
fn quiet(status: &crate::status::AtomicStatus) -> u64 {
	let mut last = (u64::MAX, Instant::now());

	loop {
		let live = status.live_num_objects();

		if live == last.0 {
			if last.1.elapsed() > Duration::from_millis(600) {
				return live;
			}
		} else {
			last = (live, Instant::now());
		}

		std::thread::sleep(Duration::from_millis(20));
	}
}

fn lens() -> [usize; 8] {
	[1, 8, 9, 24, 40, 47, 100, 1_000]
}

#[test]
fn a_new_key_is_charged_its_size_class() {
	fn check<K: Keyed + 'static>() {
		let cache = tiered::<K>(256 << 20, 64 << 20);
		let mut total = 0u64;

		settled(&cache.status, 0, "empty");

		for (i, len) in lens().into_iter().enumerate() {
			let k = key::<K>(i as u64, len);
			let len = len.max(8);

			cache.set(k, &[7u8; 64], None).expect("a set");
			total += charge(len);

			assert_eq!(cache.status.key_heap_bytes(), total, "{}: a {len}-byte key", K::name());
			settled(&cache.status, total, K::name());
		}

		// 48 B, the size class of a 40-byte key: what the cluster23 gap is made of.
		if !cfg!(feature = "thin_header") {
			assert_eq!(charge(40), 48);
		}
	}

	check::<String>();
	check::<Box<[u8]>>();
	check::<Vec<u8>>();

	// An inline key holds nothing on the heap.
	let cache = tiered::<Vec<u8>>(1 << 28, 1 << 26);
	drop(cache);

	let cache = PaperCache::<u64, TieredBuffer>::new(256 << 20, CacheTierSize::Bytes(64 << 20), PaperPolicy::LruCompactHybrid)
		.expect("a tiered cache");

	for n in 0..100u64 {
		cache.set(n, &[1u8; 64], None).expect("a set");
	}

	settled(&cache.status, 0, "u64 keys");
	assert_eq!(cache.status.live_num_objects(), 100);
}

#[test]
fn every_way_out_refunds_the_charge() {
	fn check<K: Keyed + 'static>() {
		let n = K::name();
		let cache = tiered::<K>(256 << 20, 64 << 20);

		let fill = |cache: &PaperCache<K, TieredBuffer>, ttl: Option<u32>| {
			for (i, len) in lens().into_iter().enumerate() {
				cache.set(key::<K>(i as u64, len), &[7u8; 64], ttl).expect("a set");
			}

			lens().into_iter().map(|len| charge(len.max(8))).sum::<u64>()
		};

		let one_cycle = fill(&cache, None);
		settled(&cache.status, one_cycle, n);

		// Overwrite: the same keys again. The map holds the new object's key, so the sum is the same.
		assert_eq!(fill(&cache, None), one_cycle);
		settled(&cache.status, one_cycle, &format!("{n}: overwritten"));

		// Overwrite with a TTL, then without: still the same keys.
		fill(&cache, Some(3_600));
		settled(&cache.status, one_cycle, &format!("{n}: overwritten with a ttl"));

		// Delete, one at a time.
		let mut left = one_cycle;

		for (i, len) in lens().into_iter().enumerate() {
			cache.del(&key::<K>(i as u64, len)).expect("a delete");
			left -= charge(len.max(8));

			assert_eq!(cache.status.key_heap_bytes(), left, "{n}: after deleting key {i}");
		}

		settled(&cache.status, 0, &format!("{n}: deleted"));

		// A delete of an absent key refunds nothing.
		assert!(cache.del(&key::<K>(99, 24)).is_err());
		settled(&cache.status, 0, &format!("{n}: absent delete"));

		// Expiry: set with a TTL, never touched again; the reaper takes them.
		fill(&cache, Some(1));
		settled(&cache.status, one_cycle, &format!("{n}: ttl set"));
		wait_for(&format!("{n}: expiry"), soon(), || cache.status.live_num_objects() == 0);
		settled(&cache.status, 0, &format!("{n}: expired"));

		// A wipe.
		fill(&cache, None);
		settled(&cache.status, one_cycle, &format!("{n}: refilled"));
		cache.wipe().expect("a wipe");
		settled(&cache.status, 0, &format!("{n}: wiped"));

		// Capacity eviction: far more than the cache holds, fixed-size keys.
		let small = tiered::<K>(16 << 20, 12 << 20);

		// A set the metadata cap refuses is a set that charged nothing.
		for i in 0..10_000u64 {
			let _ = small.set(key::<K>(i, 12), &[3u8; 4_096], None);
		}

		let live = quiet(&small.status);
		let s = small.hybrid_stats();
		assert!(
			s.evictions > 0,
			"{n}: the cache evicted: evictions {} fast {} slow {} overflows {} used {}",
			s.evictions, s.fast_objects, s.slow_objects, s.metadata_overflows, small.status.used_size(&PaperPolicy::LruCompactHybrid),
		);

		assert!(live > 0 && live < 10_000, "{n}: {live} live");
		settled(&small.status, live * charge(12), &format!("{n}: after evictions"));

		small.wipe().expect("a wipe");
		settled(&small.status, 0, &format!("{n}: evicted then wiped"));
	}

	check::<String>();
	check::<Box<[u8]>>();
	check::<Vec<u8>>();
}

#[test]
fn migration_does_not_charge_the_copied_key() {
	fn check<K: Keyed + 'static>() {
		let n = K::name();
		let cache = tiered::<K>(64 << 20, 8 << 20);

		for i in 0..6_000u64 {
			cache.set(key::<K>(i, 12), &[5u8; 4_096], None).expect("a set");
		}

		// Reads promote what they hit in the slow tier.
		for i in 0..2_000u64 {
			let _ = cache.get(&key::<K>(i, 12));
		}

		let stats = cache.hybrid_stats();

		assert!(stats.demotions > 0, "{n}: demotions happened");

		let live = quiet(&cache.status);
		let stats = cache.hybrid_stats();

		assert_eq!(live, 6_000, "{n}: nothing was evicted");
		assert!(stats.slow_objects > 0 && stats.fast_objects > 0, "{n}: both tiers hold objects");
		settled(&cache.status, live * charge(12), &format!("{n}: after migrations"));

		// Deleting everything, wherever it landed, refunds exactly once.
		for i in 0..6_000u64 {
			cache.del(&key::<K>(i, 12)).expect("a delete");
		}

		settled(&cache.status, 0, &format!("{n}: deleted after migrations"));
	}

	check::<String>();
	check::<Box<[u8]>>();
	check::<Vec<u8>>();
}

#[cfg(any(feature = "key_value_pmem", feature = "all_dram"))]
#[test]
fn a_refused_or_abandoned_set_charges_nothing() {
	use crate::CacheError;

	fn check<K: Keyed + 'static>() {
		let n = K::name();
		let cache = tiered::<K>(256 << 20, 64 << 20);
		let k = bytes(5, 40);

		// An abandoned permit, and a pending set dropped unwritten, half written and whole.
		let deadline = Instant::now() + soon();

		drop(cache.reserve_set_borrowed(&k, 256, None, deadline).expect("a permit"));
		assert_eq!(cache.status.key_heap_bytes(), 0, "{n}: a permit");

		for written in [0usize, 128, 256] {
			let mut pending = cache.reserve_set_borrowed(&k, 256, None, deadline).expect("a permit").fill();

			pending.write(&vec![1u8; written]);
			drop(pending);
			assert_eq!(cache.status.key_heap_bytes(), 0, "{n}: a pending set, {written} written");
		}

		// Refused before any key is built: too large, a key that is not UTF-8 for a String.
		assert!(cache.set_borrowed(&k, &vec![0u8; 512 << 20], None).is_err());

		if K::name() == "String" {
			assert!(matches!(cache.set_borrowed(&[0xff, 0xfe], &[1], None), Err(CacheError::InvalidKey)));
		}

		assert_eq!(cache.status.key_heap_bytes(), 0, "{n}: refused sets");
		settled(&cache.status, 0, &format!("{n}: refused sets published"));

		// And the committed one is charged once, as a set of the owned key is.
		cache.set_borrowed(&k, &[2u8; 256], None).expect("a set");
		settled(&cache.status, charge(40), &format!("{n}: a borrowed set"));

		cache.set_borrowed(&k, &[3u8; 256], None).expect("an overwrite");
		settled(&cache.status, charge(40), &format!("{n}: a borrowed overwrite"));

		cache.del_borrowed(&k).expect("a delete");
		settled(&cache.status, 0, &format!("{n}: a borrowed delete"));
	}

	check::<String>();
	check::<Box<[u8]>>();
	check::<Vec<u8>>();
}

/// A flat cache keeps the same count (it publishes no M, but the status is the
/// store's own books).
#[cfg(any(feature = "key_value_pmem", feature = "all_dram"))]
#[test]
fn a_flat_cache_keeps_the_same_count() {
	use crate::BufferDRAM;

	let cache = PaperCache::<String, BufferDRAM>::new(64 << 20, &[PaperPolicy::LfuCompact], PaperPolicy::LfuCompact)
		.expect("a flat cache");

	for i in 0..50u64 {
		cache.set(key::<String>(i, 40), &[1u8; 64], None).expect("a set");
	}

	assert_eq!(cache.status.key_heap_bytes(), 50 * charge(40));

	for i in 0..25u64 {
		cache.del(&key::<String>(i, 40)).expect("a delete");
	}

	assert_eq!(cache.status.key_heap_bytes(), 25 * charge(40));
	cache.wipe().expect("a wipe");
	wait_for("flat wipe", soon(), || cache.status.key_heap_bytes() == 0);
}
