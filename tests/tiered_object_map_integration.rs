/*
 * Copyright (c) Kia Shakiba
 *
 * This source code is licensed under the GNU AGPLv3 license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! `tiered_object_map` on a real cache: concurrent clients against the policy
//! worker's demotions, promotions and evictions; at quiescence every map row is
//! in the table of its value's tier, the map's per-table counts are the
//! values' own (`placement_audit`, counted independently from the values'
//! tags), and the published metadata charges the slow tables to the slow node.
//!
//!   cargo test --test tiered_object_map_integration --features lru_compact_hybrid_cache,tiered_object_map

mod common;

#[cfg(all(feature = "tiered_object_map", feature = "lru_compact_hybrid_cache"))]
mod tiered_map {
	use std::{sync::Arc, time::Duration};

	use paper_cache::{CacheTierSize, PaperCache, PaperPolicy, TieredBuffer, tiered_object_map::TieredObjectMap};

	const VALUE: usize = 1_024;

	fn churn(cache: &Arc<PaperCache<u64, TieredBuffer>>, keys: u64, ops: u64) {
		let threads: Vec<_> = (0..8u64)
			.map(|t| {
				let cache = cache.clone();

				std::thread::spawn(move || {
					let mut x = (t + 1).wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;

					for _ in 0..ops {
						x ^= x << 13;
						x ^= x >> 7;
						x ^= x << 17;

						let key = x % keys;

						match (x >> 32) % 10 {
							0 => {
								let _ = cache.del(&key);
							},
							1..=4 => {
								let _ = cache.set(key, &[(key as u8) | 1; VALUE], None);
							},
							_ => {
								if let Ok(value) = cache.get(&key) {
									assert_eq!(value[0], (key as u8) | 1);
								}
							},
						}
					}
				})
			})
			.collect();

		for thread in threads {
			thread.join().expect("a client panicked");
		}
	}

	fn check_at_quiescence(cache: &PaperCache<u64, TieredBuffer>) {
		// Quiescent: the worker has drained, the migrations have landed, and
		// the published metadata has caught up with the map.
		let (audit, placement) = crate::common::settle(
			Duration::from_secs(120),
			Duration::from_millis(100),
			|| {
				let audit = cache.object_map_audit();
				let placement = cache.placement_audit().expect("worker alive");
				let key = (audit.fast_rows, audit.slow_rows, placement.fast, placement.slow, cache.dram_metadata());

				(true, key, (audit, placement))
			},
			|(audit, placement)| format!("never settled: {audit:?} {placement:?}"),
		);

		assert_eq!((audit.misplaced_in_fast, audit.misplaced_in_slow), (0, 0), "{audit:?}");
		assert_eq!(audit.fast_rows, placement.fast, "fast rows vs fast values: {audit:?} {placement:?}");
		assert_eq!(audit.slow_rows, placement.slow, "slow rows vs slow values: {audit:?} {placement:?}");
		assert_eq!(audit.fast_rows + audit.slow_rows, cache.status().unwrap().num_objects());

		// The accounting: the map's DRAM part (its Arc, shard array and fast
		// tables) is M's map term; the slow tables are reported apart.
		let published = cache.dram_metadata();
		let structure = cache.object_map_bytes();
		let arc = paper_cache::meta::arc_bytes::<TieredObjectMap<u64, TieredBuffer>>();

		assert_eq!(published.map, arc + structure.dram, "{published:?} {structure:?}");
		assert!(published.slow >= structure.slow, "{published:?} {structure:?}");

		if audit.slow_rows > 0 {
			assert!(structure.slow > 0, "slow rows with no slow table bytes");
		}
	}

	/// No eviction: the fast tier holds about half the data.
	#[test]
	fn rows_follow_their_values_through_demotion_and_promotion() {
		let cache = Arc::new(
			PaperCache::<u64, TieredBuffer>::new(64 << 20, CacheTierSize::Bytes(2 << 20), PaperPolicy::LruCompactHybrid)
				.expect("cache"),
		);

		churn(&cache, 4_096, 40_000);
		check_at_quiescence(&cache);

		let audit = cache.object_map_audit();
		assert!(audit.fast_rows > 0 && audit.slow_rows > 0, "both tables in use: {audit:?}");
	}

	/// Capacity eviction too: the cache holds about a third of the keys.
	#[test]
	fn rows_follow_their_values_under_eviction() {
		let cache = Arc::new(
			PaperCache::<u64, TieredBuffer>::new(3 << 20, CacheTierSize::Bytes(1 << 20), PaperPolicy::LruCompactHybrid)
				.expect("cache"),
		);

		churn(&cache, 8_192, 40_000);
		check_at_quiescence(&cache);
		assert!(cache.hybrid_stats().evictions > 0, "the run never evicted");
	}

	/// A wipe empties both tables.
	#[test]
	fn a_wipe_empties_both_tables() {
		let cache = Arc::new(
			PaperCache::<u64, TieredBuffer>::new(64 << 20, CacheTierSize::Bytes(1 << 20), PaperPolicy::LruCompactHybrid)
				.expect("cache"),
		);

		churn(&cache, 4_096, 10_000);
		cache.wipe().expect("wipe");

		assert!(crate::common::wait_until(Duration::from_secs(30), || {
			let audit = cache.object_map_audit();
			audit.fast_rows == 0 && audit.slow_rows == 0
		}));
	}
}
