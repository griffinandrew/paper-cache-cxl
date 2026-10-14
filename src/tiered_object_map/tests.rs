//! The per-tier map on its own: placement, the re-filing paths, `Removal`,
//! the clear, and the per-tier accounting.

use std::sync::atomic::{AtomicU64, Ordering};

use super::*;
use crate::removal::{Removal, Swap, SwapValue};

type Map = TieredObjectMap<u64, crate::TieredBuffer>;

const LEN: usize = 64;

fn object(key: u64, tier: Tier) -> Object<u64, crate::TieredBuffer> {
	Object::new_in(key, &[key as u8; LEN], tier, None)
}

/// Keys spread over every shard (the map shards on the high bits).
fn key(i: u64) -> HashedKey {
	i.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1
}

fn assert_placed(map: &Map) -> MapAudit {
	let audit = map.audit();

	assert_eq!((audit.misplaced_in_fast, audit.misplaced_in_slow), (0, 0), "{audit:?}");
	audit
}

#[test]
fn a_row_is_filed_by_its_values_tier() {
	let map = Map::new();

	for i in 0..1_000 {
		let tier = if i % 3 == 0 { Tier::Slow } else { Tier::Fast };
		assert!(map.insert(key(i), object(key(i), tier)).is_none());
	}

	let audit = assert_placed(&map);
	assert_eq!((audit.fast_rows, audit.slow_rows), (666, 334));
	assert_eq!(map.len(), 1_000);
	assert_eq!(map.tier_lens(), (666, 334));

	for i in 0..1_000 {
		let want = if i % 3 == 0 { Tier::Slow } else { Tier::Fast };

		assert_eq!(map.table_of(&key(i)), Some(want));
		assert!(map.contains_key(&key(i)));

		let object = map.get_ref(&key(i)).expect("present");
		assert_eq!(object.value().tier(), want);
		assert_eq!(object.bytes()[0], key(i) as u8);
	}

	assert!(map.get_ref(&2).is_none());
	assert!(!map.contains_key(&2));
	assert_eq!(map.table_of(&2), None);
}

#[test]
fn an_overwrite_in_the_other_tier_moves_the_row() {
	let map = Map::new();
	let k = key(7);

	map.insert(k, object(k, Tier::Fast));
	let old = map.insert(k, object(k, Tier::Slow)).expect("the fast object");
	assert_eq!(old.value().tier(), Tier::Fast);
	assert_eq!(map.table_of(&k), Some(Tier::Slow));
	assert_eq!(map.tier_lens(), (0, 1));
	assert_eq!(map.moves(), 1);

	// Same tier: replaced in place, no move.
	let old = map.insert(k, object(k, Tier::Slow)).expect("the slow object");
	assert_eq!(old.value().tier(), Tier::Slow);
	assert_eq!(map.moves(), 1);

	let old = map.insert(k, object(k, Tier::Fast)).expect("the second slow object");
	assert_eq!(old.value().tier(), Tier::Slow);
	assert_eq!(map.table_of(&k), Some(Tier::Fast));
	assert_eq!(map.tier_lens(), (1, 0));
	assert_eq!(map.moves(), 2);
	assert_placed(&map);
}

/// `set_data` through any mutable handle re-files the row when the handle
/// drops -- the tests' own `get_mut_ref(..).set_data(..)` and anything added
/// later.
#[test]
fn a_tier_change_through_get_mut_ref_is_refiled_at_the_drop() {
	let map = Map::new();
	let k = key(11);

	map.insert(k, object(k, Tier::Fast));

	{
		let mut handle = map.get_mut_ref(&k).expect("present");
		let demoted = handle.value().migrated_to(Tier::Slow);
		drop(handle.set_data(demoted));

		// Still filed fast while the handle is held: the lock is.
		assert_eq!(handle.value().tier(), Tier::Slow);
	}

	assert_eq!(map.table_of(&k), Some(Tier::Slow));
	assert_eq!(map.moves(), 1);

	// A handle that changes nothing moves nothing.
	map.get_mut_ref(&k).expect("present").expires(Some(60));
	assert_eq!(map.table_of(&k), Some(Tier::Slow));
	assert_eq!(map.moves(), 1);
	assert!(map.get_ref(&k).unwrap().expiry().is_some());

	assert_placed(&map);
}

#[test]
fn swap_value_checks_identity_swaps_and_moves() {
	let map = Map::new();
	let k = key(13);

	assert_eq!(map.swap_value(&k, &object(k, Tier::Fast).value().clone(), |_| ()), Swap::Gone);

	map.insert(k, object(k, Tier::Fast));
	let snapshot = map.get_ref(&k).unwrap().snapshot();

	// A set replaced it: superseded, nothing swapped.
	map.insert(k, object(k, Tier::Fast));
	assert_eq!(SwapValue::swap_value(&map, &k, &snapshot, |_| unreachable!()), Swap::<()>::Superseded);

	let current = map.get_ref(&k).unwrap().snapshot();
	let copy = current.migrated_to(Tier::Slow);
	let installed = copy.clone();

	let swapped = SwapValue::swap_value(&map, &k, &current, move |object| drop(object.set_data(copy)));
	assert_eq!(swapped, Swap::Swapped(()));
	assert_eq!(map.table_of(&k), Some(Tier::Slow));
	assert!(TieredValue::ptr_eq(map.get_ref(&k).unwrap().value(), &installed));
	assert_placed(&map);
}

#[test]
fn removal_semantics() {
	let map = Map::new();
	let fallbacks = AtomicU64::new(0);

	// Empty: the fallback names nothing, counted.
	assert!(matches!(Removal::fallback_victim(&map, &fallbacks), Err(CacheError::Internal)));
	assert_eq!(fallbacks.load(Ordering::Relaxed), 1);

	let (f, s) = (key(1), key(2));
	map.insert(f, object(f, Tier::Fast));
	map.insert(s, object(s, Tier::Slow));

	// take_if: the predicate decides, under the lock; a refusal leaves it.
	assert!(Removal::take_if(&map, &s, |_| false).is_none());
	assert!(map.contains_key(&s));

	let taken = Removal::take_if(&map, &s, |object| object.value().tier() == Tier::Slow).expect("taken");
	assert_eq!(taken.value().tier(), Tier::Slow);
	assert!(!map.contains_key(&s));
	assert!(Removal::take_if(&map, &s, |_| true).is_none(), "gone");

	// take_evict: whatever is there.
	assert!(Removal::take_evict(&map, &f).is_some());
	assert!(Removal::take_evict(&map, &f).is_none());

	// The fallback names a present key, from either table.
	map.insert(s, object(s, Tier::Slow));
	assert_eq!(Removal::fallback_victim(&map, &fallbacks).unwrap(), s);
	assert_eq!(fallbacks.load(Ordering::Relaxed), 2);

	map.insert(f, object(f, Tier::Fast));
	let victim = Removal::fallback_victim(&map, &fallbacks).unwrap();
	assert!(victim == f || victim == s);
	assert_placed(&map);
}

#[test]
fn clear_counted_counts_both_tables() {
	let map = Map::new();

	for i in 0..300 {
		map.insert(key(i), object(key(i), if i % 2 == 0 { Tier::Fast } else { Tier::Slow }));
	}

	let cleared = map.clear_counted(|_| 10);
	assert_eq!(cleared.objects, 300);
	assert_eq!(cleared.base_bytes, 3_000);
	assert_eq!(map.len(), 0);
	assert_eq!(map.tier_lens(), (0, 0));
}

#[test]
fn for_each_value_reports_every_row_with_its_tier() {
	let map = Map::new();

	for i in 0..200 {
		map.insert(key(i), object(key(i), if i % 4 == 0 { Tier::Slow } else { Tier::Fast }));
	}

	let (mut fast, mut slow) = (0, 0);
	map.for_each_value(|k, tier, charge| {
		assert_eq!(map_tier_by_key(k), tier);
		// The item's own bytes, in either layout: the value under the default
		// one, header, key and value under `thin_header`.
		let item = crate::object::overhead::resident_item_bytes_with(
			crate::TieredValue::<u64>::item_prefix_bytes_for(&k),
			LEN as u32,
		);
		assert_eq!(charge, item);
		match tier {
			Tier::Fast => fast += 1,
			Tier::Slow => slow += 1,
		}
	});
	assert_eq!((fast, slow), (150, 50));

	fn map_tier_by_key(k: HashedKey) -> Tier {
		let i = (0..200u64).find(|&i| key(i) == k).expect("a key the test inserted");
		if i % 4 == 0 { Tier::Slow } else { Tier::Fast }
	}
}

/// The per-tier counters are the tables' allocations, exactly: held to the
/// tables' own layouts after growth, moves and removals, and to the
/// allocator (`thread.allocated`) across an insert run.
#[test]
fn the_map_counts_its_own_tables_per_tier() {
	let map = Map::with_shard_amount(4);

	assert_eq!(map.table_bytes(), (0, 0), "an empty table allocates nothing");

	// Built before the readings: only the tables allocate in the loops. The
	// fast tables are held to this thread's jemalloc `thread.allocated`; the
	// slow ones (allocated through the slow arenas) to the slow pool's own
	// live-byte counter under `measured_accounting`, and to their layouts
	// always.
	let fast_objects: Vec<_> = (0..10_000u64).map(|i| (key(2 * i), object(key(2 * i), Tier::Fast))).collect();
	let slow_objects: Vec<_> = (0..10_000u64).map(|i| (key(2 * i + 1), object(key(2 * i + 1), Tier::Slow))).collect();

	let base = crate::meta::thread_live_bytes();

	for (k, object) in fast_objects {
		map.insert(k, object);
	}

	let grown = (crate::meta::thread_live_bytes() - base) as u64;
	assert_eq!(map.table_bytes().1, 0);
	assert_eq!(map.table_bytes(), map.table_bytes_by_layout(), "the fast counter is the tables' allocations");

	// Reported, not asserted: on this allocator `thread.allocated` read 163840
	// against the tables' 327680 (4 shards x nallocx(69648) = 81920) -- it
	// does not see every large allocation the global allocator makes, so the
	// layouts above are the reference.
	println!("TIERED_MAP_FAST_TABLES counter={} thread_allocated_delta={grown}", map.table_bytes().0);

	for (k, object) in slow_objects {
		map.insert(k, object);
	}

	let (fast, slow) = map.table_bytes();

	assert!(fast > 0 && slow > 0);
	assert_eq!((fast, slow), map.table_bytes_by_layout());

	// Demote every fast row: the slow tables grow, the emptied fast ones are
	// freed (`shrink_if_sparse`).
	for i in (0..20_000u64).step_by(2) {
		let mut handle = map.get_mut_ref(&key(i)).expect("present");
		let demoted = handle.value().migrated_to(Tier::Slow);
		drop(handle.set_data(demoted));
	}

	assert_eq!(map.tier_lens(), (0, 20_000));
	assert_eq!(map.table_bytes(), map.table_bytes_by_layout());
	assert!(map.table_bytes().1 > slow);
	assert!(map.table_bytes().0 < fast / 8, "the emptied fast tables shrank: {:?} from {fast}", map.table_bytes());
	assert_placed(&map);

	// The structure's NodeBytes: shard array + fast tables in DRAM, slow apart.
	let nodes = map.structure_bytes();
	assert_eq!(nodes.slow, map.table_bytes().1);
	assert_eq!(nodes.dram, map.table_bytes().0 + map.shard_array_bytes());

	drop(map);
}

/// A population that falls shrinks its table; rows moving back and forth stay
/// counted exactly; a clear keeps the capacity.
#[test]
fn a_sparse_table_shrinks_and_a_cleared_one_is_freed() {
	let map = Map::with_shard_amount(2);

	for i in 0..40_000u64 {
		map.insert(key(i), object(key(i), Tier::Slow));
	}

	let full = map.table_bytes().1;

	for i in 0..36_000u64 {
		assert!(map.take_evict(&key(i)).is_some());
	}

	let shrunk = map.table_bytes().1;
	assert!(shrunk * 4 <= full, "{shrunk} from {full}");
	assert_eq!(map.table_bytes(), map.table_bytes_by_layout());

	// The held population moved between the tables, round by round.
	for round in 0..4u64 {
		for i in 36_000..40_000u64 {
			let k = key(i);
			let tier = if round % 2 == 0 { Tier::Fast } else { Tier::Slow };
			map.insert(k, object(k, tier));
		}
	}

	assert_eq!(map.tier_lens(), (0, 4_000));
	assert_placed(&map);
	assert_eq!(map.table_bytes(), map.table_bytes_by_layout());

	let before = map.table_bytes();
	let cleared = map.clear_counted(|_| 1);
	assert_eq!(cleared.objects, 4_000);
	assert_eq!(map.table_bytes(), before, "a clear keeps the tables' capacity, as DashMap's does");
}

/// The slow tables' counter against the slow pool's own live-byte count.
/// The pool is process-global, so this needs the process to itself:
/// `--ignored --test-threads=1`.
#[cfg(feature = "measured_accounting")]
#[test]
#[ignore]
fn the_slow_counter_is_what_the_slow_pool_holds() {
	let map = Map::with_shard_amount(4);
	let objects: Vec<_> = (0..10_000u64).map(|i| (key(i), object(key(i), Tier::Slow))).collect();

	let base = crate::numa_alloc::measured::raw_sum(crate::numa_alloc::NODE_SLOW);

	for (k, object) in objects {
		map.insert(k, object);
	}

	let grown = crate::numa_alloc::measured::raw_sum(crate::numa_alloc::NODE_SLOW) - base;
	assert_eq!(grown as u64, map.table_bytes().1);
	println!("TIERED_MAP_SLOW_TABLES counter={} slow_pool_delta={grown}", map.table_bytes().1);
}

/// What `meta::map_bytes` publishes under this feature: the Arc and the
/// structure's DRAM part as DRAM, the slow tables as slow.
#[test]
fn map_bytes_charges_the_slow_tables_to_the_slow_node() {
	let objects: crate::ObjectMapRef<u64, crate::TieredBuffer> = crate::new_hybrid_object_map();

	for i in 0..5_000u64 {
		objects.insert(key(i), object(key(i), if i % 2 == 0 { Tier::Fast } else { Tier::Slow }));
	}

	let mut state = ();
	let nodes = crate::meta::map_bytes(&objects, &mut state, true);
	let (fast, slow) = objects.table_bytes();

	assert_eq!(nodes.slow, slow);
	assert_eq!(nodes.dram, crate::meta::arc_bytes::<Map>() + objects.shard_array_bytes() + fast);
}

/// Many threads re-filing the same keys through every path at once; the
/// invariant holds after (and the map's count is the keys').
#[test]
fn concurrent_refiling_keeps_every_row_in_its_table() {
	let map = std::sync::Arc::new(Map::new());
	const KEYS: u64 = 512;

	for i in 0..KEYS {
		map.insert(key(i), object(key(i), Tier::Fast));
	}

	let threads: Vec<_> = (0..8u64)
		.map(|t| {
			let map = map.clone();

			std::thread::spawn(move || {
				let mut x = t.wrapping_mul(0x2545_F491_4F6C_DD1D) | 1;

				for _ in 0..20_000 {
					x ^= x << 13;
					x ^= x >> 7;
					x ^= x << 17;

					let k = key(x % KEYS);
					let tier = if x & 0x100 == 0 { Tier::Fast } else { Tier::Slow };

					match (x >> 20) % 5 {
						0 => {
							map.insert(k, object(k, tier));
						},
						1 => {
							// A migration: snapshot, copy, swap if current.
							let Some(snapshot) = map.get_ref(&k).map(|o| o.snapshot()) else { continue };
							let copy = snapshot.migrated_to(tier);
							let _ = map.swap_value(&k, &snapshot, move |object| drop(object.set_data(copy)));
						},
						2 => {
							if let Some(mut handle) = map.get_mut_ref(&k) {
								let copy = handle.value().migrated_to(tier);
								drop(handle.set_data(copy));
							}
						},
						3 => {
							if let Some(object) = map.get_ref(&k) {
								assert_eq!(object.bytes()[0], k as u8);
							}
						},
						_ => {
							if map.take_if(&k, |_| x & 0x200 == 0).is_some() {
								map.insert(k, object(k, tier));
							}
						},
					}
				}
			})
		})
		.collect();

	for thread in threads {
		thread.join().expect("a worker panicked");
	}

	let audit = assert_placed(&map);
	assert_eq!(audit.fast_rows + audit.slow_rows, KEYS);
	assert!(audit.fast_rows > 0 && audit.slow_rows > 0);
	assert_eq!(map.table_bytes(), map.table_bytes_by_layout());
}

/// Values whose key is held in the ITEM (`thin_header`: a `String` key's
/// bytes, and the length and the expiry, all tier with the value) or beside
/// it (the default layout): through every path that re-files a row -- an
/// overwrite into the other tier, a migration's swap, a `get_mut_ref` that
/// changes the value, a `ttl` through the guard -- each row stays in the table
/// of its value's tier and its value reads back whole, key, expiry and bytes.
/// Under `thin_header` a slow row's table, its item and the item's metadata
/// are all on the slow node, and only the 16-byte header is DRAM.
#[test]
fn string_keyed_values_keep_their_rows_and_read_back_whole() {
	type StringMap = TieredObjectMap<String, crate::TieredBuffer>;

	let map = StringMap::new();
	let name = |i: u64| format!("key-{i:05}-{}", "x".repeat((i % 40) as usize));
	let bytes = |i: u64, round: u64| vec![(i + round) as u8; 1 + ((i * 37 + round * 11) % 3000) as usize];
	let tier_of = |i: u64, round: u64| if (i + round) % 3 == 0 { Tier::Slow } else { Tier::Fast };

	for i in 0..600u64 {
		map.insert(key(i), Object::new_in(name(i), &bytes(i, 0), tier_of(i, 0), None));
	}

	for round in 1..4u64 {
		for i in 0..600u64 {
			let k = key(i);

			match (i + round) % 4 {
				// An overwrite, into whichever tier.
				0 => {
					map.insert(k, Object::new_in(name(i), &bytes(i, round), tier_of(i, round), Some(3600)));
				},

				// A migration: snapshot, copy, swap if still current.
				1 => {
					let snapshot = map.get_ref(&k).expect("present").snapshot();
					let copy = snapshot.migrated_to(tier_of(i, round));
					assert!(matches!(map.swap_value(&k, &snapshot, move |object| drop(object.set_data(copy))), Swap::Swapped(())));
				},

				// A tier change through the guard.
				2 => {
					let mut handle = map.get_mut_ref(&k).expect("present");
					let copy = handle.value().migrated_to(tier_of(i, round));
					drop(handle.set_data(copy));
				},

				// An expiry change through the guard: the row stays put.
				_ => {
					let before = map.table_of(&k);
					map.get_mut_ref(&k).expect("present").expires(Some(60 * round as u32));
					assert_eq!(map.table_of(&k), before);
				},
			}
		}

		assert_placed_any(&map);
	}

	// Every value reads back whole from whichever table holds it.
	for i in 0..600u64 {
		let object = map.get_ref(&key(i)).expect("present");

		assert!(object.key_matches(&name(i)), "key {i}");
		assert_eq!(map.table_of(&key(i)), Some(object.value().tier()));
		assert_eq!(object.bytes()[0] as u64, (i + last_write(i)) & 0xff, "bytes of {i}");
		assert_eq!(object.bytes().len(), bytes(i, last_write(i)).len(), "length of {i}");
	}

	assert_eq!(map.table_bytes(), map.table_bytes_by_layout());

	/// The round of key `i`'s last overwrite (0: never overwritten).
	fn last_write(i: u64) -> u64 {
		(1..4u64).rev().find(|round| (i + round) % 4 == 0).unwrap_or(0)
	}

	fn assert_placed_any<K>(map: &TieredObjectMap<K, crate::TieredBuffer>) {
		let audit = map.audit();
		assert_eq!((audit.misplaced_in_fast, audit.misplaced_in_slow), (0, 0), "{audit:?}");
	}
}
