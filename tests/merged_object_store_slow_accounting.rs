/*
 * Copyright (c) Kia Shakiba
 *
 * This source code is licensed under the GNU AGPLv3 license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! The merged object store's slow tier: modelled against measured.
//!
//! `hybrid_stats().slow_bytes_used` is what the store CHARGES to the slow tier;
//! `numa_alloc::measured::slow_allocated()` is what the slow allocator actually
//! handed out. Nothing but cached values is allocated on the slow node, so once
//! the cache is idle the two must be the same number. This is the slow row of
//! paper-server's `*** MEASURED vs MODELLED ***` report, pinned.
//!
//! It was not the same number before `split_tier_migrations`. A hit on a key
//! the fast budget could not hold queued a promotion and, from the settle
//! right behind it, a demotion of the same key, and the worker dispatched every
//! demotion in a drain before any promotion. The pair ran backwards, the bytes
//! went back to DRAM, and the store went on charging them to the slow tier:
//! 6,877 objects and 793,088 B, ratio 0.944, on the cluster99 golden trace.
//!
//! ONE test, in a binary of its own, on purpose: the counter is process-global,
//! so a second cache alive at the same time -- another test on another thread
//! -- would be counted too. The caches below run one after another, each
//! dropped (and its workers joined) before the next is built.
//!
//! ```text
//! cargo +nightly test --test merged_object_store_slow_accounting \
//!     --features lru_compact_hybrid_cache,merged_object_store,measured_accounting
//! ```

#![cfg(all(
	feature = "merged_object_store",
	feature = "measured_accounting",
	feature = "lru_compact_hybrid_cache",
))]

use std::time::{Duration, Instant};

use paper_cache::{
	numa_alloc::measured, CacheTierSize, GateConfig, HybridStats, MetadataModel, PaperCache, PaperPolicy,
	TieredBuffer,
};

const KEYS: u64 = 3_000;

/// Small enough that the per-object metadata reservation (78 B an object in
/// this build) takes most of the budget by the end of the fill -- 234,000 of
/// its 245,760 B -- so most hits land on slow keys. It was 64 KiB, which the
/// reservation filled a third of the way through, until S5: with the key
/// ceiling (per-object: F / 78 B) that tier refuses keys past 840, and it
/// must admit all 3,000. (The regime it reached -- every hit a promotion the
/// settle undid -- is T8's, in the lib.)
const FAST_TIER: u64 = 240 * 1024;

/// Varied, never a size class on the nose, never empty.
fn value(key: u64, round: u64) -> Vec<u8> {
	vec![key as u8; 24 + ((key * 37 + round * 101) % 900) as usize]
}

/// Waits until nothing moves: the gauges have caught up with every live
/// object, and neither they nor the allocator's counter have changed across
/// several polls. The worker applies migrations asynchronously, and a hit
/// that has not been processed yet is not a hit this test has checked.
fn idle(cache: &PaperCache<u64, TieredBuffer>, live: u64) -> (u64, HybridStats) {
	let deadline = Instant::now() + Duration::from_secs(60);
	let mut last = None;
	let mut unchanged = 0;

	loop {
		let sample = (measured::slow_allocated(), cache.hybrid_stats());
		let tracked = sample.1.fast_objects + sample.1.slow_objects;

		match tracked == live && Some(sample) == last {
			true => unchanged += 1,
			false => unchanged = 0,
		}

		if unchanged == 5 {
			return sample;
		}

		assert!(
			Instant::now() < deadline,
			"the cache never went idle ({tracked} of {live} objects tracked): {sample:?}",
		);

		last = Some(sample);
		std::thread::sleep(Duration::from_millis(50));
	}
}

#[test]
fn slow_bytes_measured_equal_slow_bytes_modelled() {
	for policy in [
		PaperPolicy::LruCompactHybrid,
		PaperPolicy::LfuCompactHybrid,
		PaperPolicy::FifoCompactHybrid,
		PaperPolicy::ClockCompactHybrid,
	] {
		let before = measured::slow_allocated();

		let (measured_slow, stats) = {
			// The per-object model (S5), whose arithmetic the tier above is
			// sized in: under the measured one this tier is smaller than the
			// store's own structures, and its key ceiling would refuse every
			// key.
			let mut gate = GateConfig::default();
			gate.metadata_model = MetadataModel::PerObject;

			let cache = PaperCache::<u64, TieredBuffer>::new_with_gate(
				64 * 1024 * 1024,
				CacheTierSize::Bytes(FAST_TIER),
				policy,
				gate,
			)
			.expect("the merged store implements this policy");

			// A read-through fill, hitting older keys all the way: most of
			// them are slow by the time they are hit.
			for key in 0..KEYS {
				cache.set(key, &value(key, 0), None).expect("set");

				for back in [1, 5, 50, 500] {
					if key >= back {
						cache.get(&(key - back)).expect("a live key hits");
					}
				}
			}

			// Overwrites that change the size, of fast and slow keys alike, and
			// the hits that follow them.
			for key in (0..KEYS).step_by(7) {
				cache.set(key, &value(key, 1), None).expect("overwrite");
				cache.get(&key).expect("an overwritten key hits");
			}

			// And removals, so a freed slot's bytes must leave the tier too.
			let mut live = KEYS;

			for key in (3..KEYS).step_by(11) {
				cache.del(&key).expect("del");
				live -= 1;
			}

			idle(&cache, live)
		};

		let modelled = stats.slow_bytes_used;
		let measured = measured_slow - before;

		assert!(modelled > 0, "{policy}: the fast tier must have demoted: {stats:?}");
		assert_eq!(
			measured,
			modelled,
			"{policy}: the slow allocator holds {measured} B but the store charges {modelled} B \
			 to the slow tier (drift {:+} B): {stats:?}",
			measured as i64 - modelled as i64,
		);
	}
}
