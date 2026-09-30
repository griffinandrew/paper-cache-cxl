/*
 * Copyright (c) Kia Shakiba
 *
 * This source code is licensed under the GNU AGPLv3 license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! Helpers the integration test binaries share (`mod common;`). Each binary
//! compiles its own copy and uses a subset, hence the blanket `dead_code`.
//!
//! Child processes: the process-global counters (P, the migration statistics,
//! the gate's accounting, the allocator's arenas) outlive a cache, so a test
//! that runs several policies runs each in a child copy of the test binary
//! ([`each_alone`]), and a test that must not see another's leavings runs
//! alone in one ([`alone`]) -- "each policy run resets all cache state", and
//! only a fresh process resets all of it.

#![allow(dead_code)]

use std::{
	process::Command,
	thread,
	time::{Duration, Instant},
};

const CHILD: &str = "PAPER_TEST_CHILD";

/// The child's case, for a test run once per case: an index into its list.
const CASE: &str = "PAPER_TEST_CASE";

/// How long a child may run before it is killed and its test failed.
const DEADLINE: Duration = Duration::from_secs(900);

fn in_child() -> bool {
	std::env::var_os(CHILD).is_some_and(|value| value == "1")
}

/// Polls `predicate` every 20 ms until it holds (`true`) or `timeout` passes
/// (`false`).
pub fn wait_until(timeout: Duration, mut predicate: impl FnMut() -> bool) -> bool {
    let deadline = Instant::now() + timeout;

    loop {
        if predicate() {
            return true;
        }

        if Instant::now() > deadline {
            return false;
        }

        thread::sleep(Duration::from_millis(20));
    }
}

/// Runs `body` in a child process in which `test`, of `module`
/// (`module_path!()`), is the only test, killed at `DEADLINE`. The parent
/// passes only if the child ran exactly that one test and it passed.
pub fn alone(module: &str, test: &str, body: impl FnOnce()) {
	if in_child() {
		body();
		return;
	}

	run_in_child(module, test, None);
}

/// `test`, of `module`, once per case of `cases` -- a policy, a design -- each
/// in a child process of its own, in which it is the only test and only that
/// case runs `body`.
pub fn each_alone<T: Copy>(module: &str, test: &str, cases: impl AsRef<[T]>, mut body: impl FnMut(T)) {
	let cases = cases.as_ref();

	if in_child() {
		let index: usize = std::env::var(CASE)
			.expect("a per-case child is told its case")
			.parse()
			.expect("the case is an index");

		body(cases[index]);
		return;
	}

	for index in 0..cases.len() {
		run_in_child(module, test, Some(index));
	}
}

fn run_in_child(module: &str, test: &str, case: Option<usize>) {
	// libtest names a test by its path without the crate.
	let name = match module.split_once("::") {
		Some((_, module)) => format!("{module}::{test}"),
		None => test.to_owned(),
	};

	let tag = case.map_or_else(String::new, |index| format!("-{index}"));
	let path = std::env::temp_dir().join(format!("paper-test-{}-{test}{tag}.out", std::process::id()));
	let file = std::fs::File::create(&path).expect("the child's output file");

	let mut command = Command::new(std::env::current_exe().expect("this test binary"));

	command
		.args([name.as_str(), "--exact", "--test-threads=1", "--nocapture"])
		.env(CHILD, "1")
		.stdout(file.try_clone().expect("the output file, twice"))
		.stderr(file);

	if let Some(index) = case {
		command.env(CASE, index.to_string());
	}

	let mut child = command.spawn().expect("could not re-run this test binary");
	let start = Instant::now();

	let status = loop {
		match child.try_wait().expect("the child's status") {
			Some(status) => break Some(status),

			// Killed through its own handle: never a process found by name.
			None if start.elapsed() > DEADLINE => {
				let _ = child.kill();
				let _ = child.wait();
				break None;
			},

			None => thread::sleep(Duration::from_millis(10)),
		}
	};

	let output = std::fs::read_to_string(&path).unwrap_or_default();
	let _ = std::fs::remove_file(&path);

	let what = match case {
		Some(index) => format!("{name}, case {index}"),
		None => name.clone(),
	};

	assert!(
		status.is_some_and(|status| status.success()) && output.contains("test result: ok. 1 passed;"),
		"{what}, run alone in a child process ({}):\n{output}",
		status.map_or_else(|| format!("killed after {DEADLINE:?}"), |status| status.to_string()),
	);

	// The child's diagnostics, for a run with --nocapture.
	eprintln!("--- {what}, alone:\n{output}");
}

/// The tests every tiered design's integration binary runs, and the helpers
/// they and the design's own tests share, written once. Invoked inside the
/// binary's `mod hybrid_cache_tests`, so the tests keep their names:
///
/// ```ignore
/// crate::common::hybrid_suite! {
///     policy: PaperPolicy::LruCompactHybrid,
///     value_len: 1024,
///     demoted_of_two: (1u32, 2u32),
///     ttl_demotion: true,
/// }
/// ```
///
/// * `policy`: the design under test.
/// * `value_len`: the length of a demotion test's ~1 KB values. 1008 where a
///   fused value's header must round the item to exactly 1 KiB (see FIFO's).
/// * `demoted_of_two`: `(demoted, kept)`: which of two keys set in turn into
///   a tier that holds one the design demotes -- the older in an ordered
///   design, the newcomer in LFU, whose admission ranks by frequency.
/// * `ttl_demotion`: whether the generic `ttl_survives_a_demotion` applies. LFU
///   admits a filler slow and needs a slow one hit to promote the TTL'd key
///   back, so it keeps a test of its own.
///
/// It defines `wait_until`, `ensure_pmem_allocator_warm`, `MIGRATION_TIMEOUT`,
/// `VALUE_LEN`, `value`, `TTL_FAST_TIER` and `DEMOTES_ONE_OF_TWO` for the
/// design's own tests, and needs `use paper_cache::{CacheError, CacheTierSize,
/// PaperCache, PaperPolicy, Tier, TieredBuffer}` in scope only for those.
///
/// The all-tier-crossing tests exercise the real slow-node allocator, whose
/// first use in the process pays a one-time pool init and prewarm (observed
/// ~45 s in a sandbox): `ensure_pmem_allocator_warm()` pays it synchronously
/// at the start of every test that crosses tiers -- the process-wide `Once`
/// behind it makes only the first call wait.
#[allow(unused_macros)]
macro_rules! hybrid_suite {
    (
        policy: $policy:expr,
        value_len: $len:expr,
        demoted_of_two: ($demoted:literal, $kept:literal),
        ttl_demotion: $ttl:tt $(,)?
    ) => {
        use $crate::common::wait_until;

        /// Forces the one-time slow-allocator pool init/prewarm to complete
        /// before a test's own timing-sensitive assertions begin, with the
        /// metadata reservation off (mechanics tests at toy scales; see
        /// `get_hybrid_dram_shared_overhead`) -- the variable first, as every
        /// test here: a cache built before it would use the measured metadata
        /// model, whose key ceiling refuses keys on a tier smaller than the
        /// cache's own structures, and whether a sibling test had set it yet
        /// was a race.
        fn ensure_pmem_allocator_warm() {
            unsafe { std::env::set_var("PAPER_DISABLE_SHARED_OVERHEAD", "1") };
            let cache = PaperCache::<u32, TieredBuffer>::new(1_048_576, CacheTierSize::Bytes(1), $policy)
                .expect("warm-up cache should construct");

            cache.set(0u32, b"warm", None).expect("warm-up set should succeed");

            let ready = wait_until(std::time::Duration::from_secs(90), || {
                cache.tier_of(&0u32) == Some(Tier::Slow)
            });
            assert!(ready, "PMEM allocator warm-up should complete within 90s");
        }

        const MIGRATION_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(10);

        // The fast-tier budget also reserves a per-object DRAM cost for the
        // shared object hashtable + eviction stacks (tens of bytes per object
        // -- see `object/overhead.rs::get_hybrid_dram_shared_overhead`). To
        // keep the byte-sized fast-tier budgets below behaving intuitively
        // (~value-sized) rather than being dominated by that reservation, the
        // demotion tests use ~1 KB values, so the reservation is a small
        // fraction and the fast-tier sizes have a wide, robust margin.
        const VALUE_LEN: usize = $len;

        fn value(seed: u8) -> Vec<u8> {
            vec![seed; VALUE_LEN]
        }

        /// A tier that holds one of these values and not two.
        const DEMOTES_ONE_OF_TWO: u64 = 1_600;

        /// A TTL'd object's `base_size` (via `get_ttl_overhead`) is larger by
        /// tens of bytes than a `None`-ttl one's, so a fast tier sized only for
        /// `None`-ttl objects is too tight for a single ttl'd object: promoting
        /// it can immediately trip the settle again and re-demote the very key
        /// just promoted. Sized to hold ~2 objects, so a ttl'd key demotes
        /// under filler pressure yet can still be observed as `Fast`.
        const TTL_FAST_TIER: u64 = 2_600;

        #[test]
        fn admission_always_lands_in_fast_tier() {
            ensure_pmem_allocator_warm();

            let cache = PaperCache::<u32, TieredBuffer>::new(1_048_576, CacheTierSize::Bytes(1_048_576), $policy)
                .expect("cache should construct");

            cache.set(1u32, b"hello world", None).expect("set should succeed");

            // Admission is synchronous (the object is inserted as `TieredBuffer::
            // Fast` directly inside `set()`, before the WorkerEvent is even
            // broadcast), so this doesn't need `wait_until`.
            assert_eq!(cache.tier_of(&1u32), Some(Tier::Fast));
            assert_eq!(cache.get(&1u32).unwrap(), b"hello world");
        }

        $crate::common::ttl_survives_a_demotion!($ttl, $policy);

        #[test]
        fn terminal_eviction_only_removes_from_slow_tier_and_is_counted() {
            ensure_pmem_allocator_warm();

            // A small overall cache with a tiny fast tier: every object demotes
            // to slow almost immediately, and once total usage exceeds max_size
            // the slow tier's next victim must be evicted (never the fast tier,
            // which by construction holds only the newest-admitted key).
            let cache = PaperCache::<u32, TieredBuffer>::new(256, CacheTierSize::Bytes(10), $policy)
                .expect("cache should construct");

            for key in 1u32..=10 {
                let _ = cache.set(key, b"payload bytes", None);
            }

            let evicted = wait_until(MIGRATION_TIMEOUT, || {
                cache.hybrid_stats().evictions >= 1
            });
            assert!(evicted, "at least one terminal eviction should have occurred");

            // Give the worker a moment to settle so the count below is stable.
            std::thread::sleep(std::time::Duration::from_millis(200));

            let stats = cache.hybrid_stats();
            let present = (1u32..=10).filter(|key| cache.has(key)).count() as u64;

            // Every key is accounted for exactly once: either still present
            // (in fast or slow -- doesn't matter which) or evicted. None should
            // be silently lost, and none double-counted.
            assert_eq!(present + stats.evictions, 10);

            // Every evicted key is fully gone, never left dangling in a tier.
            for key in 1u32..=10 {
                if !cache.has(&key) {
                    assert_eq!(cache.tier_of(&key), None);
                }
            }
        }

        #[test]
        fn set_fast_tier_size_takes_effect_at_runtime() {
            ensure_pmem_allocator_warm();

            let cache = PaperCache::<u32, TieredBuffer>::new(1_048_576, CacheTierSize::Bytes(1_048_576), $policy)
                .expect("cache should construct");

            cache.set(1u32, b"first value 123", None).expect("set should succeed");
            assert_eq!(cache.tier_of(&1u32), Some(Tier::Fast));
            assert_eq!(cache.fast_tier_size(), 1_048_576);

            // Shrink the fast tier drastically; the existing key should demote
            // even without any further access, once the worker applies the
            // resize (the stack settles eagerly on `resize_fast_tier`).
            cache.set_fast_tier_size(CacheTierSize::Bytes(1)).expect("resize should succeed");
            assert_eq!(cache.fast_tier_size(), 1);

            let demoted = wait_until(MIGRATION_TIMEOUT, || {
                cache.tier_of(&1u32) == Some(Tier::Slow)
            });
            assert!(demoted, "shrinking the fast tier should demote the existing key");
        }

        #[test]
        fn zero_fast_tier_size_is_rejected() {
            let result = PaperCache::<u32, TieredBuffer>::new(1_024, CacheTierSize::Bytes(0), $policy);
            assert!(matches!(result, Err(CacheError::InvalidFastTierSize)));
        }

        #[test]
        fn fast_tier_size_exceeding_max_size_is_rejected() {
            let result = PaperCache::<u32, TieredBuffer>::new(1_024, CacheTierSize::Bytes(2_048), $policy);
            assert!(matches!(result, Err(CacheError::InvalidFastTierSize)));
        }

        #[test]
        fn zero_max_size_is_rejected() {
            let result = PaperCache::<u32, TieredBuffer>::new(0, CacheTierSize::Bytes(100), $policy);
            assert!(matches!(result, Err(CacheError::ZeroCacheSize)));
        }

        #[test]
        fn tiny_fast_tier_demotes_everything_almost_immediately() {
            ensure_pmem_allocator_warm();

            let cache = PaperCache::<u32, TieredBuffer>::new(1_048_576, CacheTierSize::Bytes(1), $policy)
                .expect("cache should construct");

            cache.set(1u32, b"a value", None).expect("set should succeed");

            let demoted = wait_until(MIGRATION_TIMEOUT, || {
                cache.tier_of(&1u32) == Some(Tier::Slow)
            });
            assert!(demoted, "a 1-byte fast tier should demote any real value almost immediately");
            assert_eq!(cache.get(&1u32).unwrap(), b"a value");
        }

        #[test]
        fn del_removes_key_from_whichever_tier_it_is_in() {
            ensure_pmem_allocator_warm();

            // One key per tier is the whole point, so the fast tier has to hold
            // one of these ~1 KB values and not two (the reservation is per
            // object: a 15-byte payload against a 40-byte budget fitted both).
            let cache = PaperCache::<u32, TieredBuffer>::new(1_048_576, CacheTierSize::Bytes(DEMOTES_ONE_OF_TWO), $policy)
                .expect("cache should construct");

            cache.set(1u32, &value(0xA1), None).expect("set should succeed");
            cache.set(2u32, &value(0xB2), None).expect("set should succeed");
            assert!(wait_until(MIGRATION_TIMEOUT, || cache.tier_of(&$demoted) == Some(Tier::Slow)));
            assert_eq!(cache.tier_of(&$kept), Some(Tier::Fast));

            cache.del(&1u32).expect("del should succeed");
            assert!(!cache.has(&1u32));
            assert_eq!(cache.tier_of(&1u32), None);

            cache.del(&2u32).expect("del should succeed");
            assert!(!cache.has(&2u32));
            assert_eq!(cache.tier_of(&2u32), None);
        }

        #[test]
        fn wipe_clears_both_tiers() {
            ensure_pmem_allocator_warm();

            // "Both tiers" requires one key actually sitting in each: the same
            // sizing as `del_removes_key_from_whichever_tier_it_is_in`.
            let cache = PaperCache::<u32, TieredBuffer>::new(1_048_576, CacheTierSize::Bytes(DEMOTES_ONE_OF_TWO), $policy)
                .expect("cache should construct");

            cache.set(1u32, &value(0xA1), None).expect("set should succeed");
            cache.set(2u32, &value(0xB2), None).expect("set should succeed");
            assert!(wait_until(MIGRATION_TIMEOUT, || cache.tier_of(&$demoted) == Some(Tier::Slow)));
            assert_eq!(cache.tier_of(&$kept), Some(Tier::Fast));

            cache.wipe().expect("wipe should succeed");

            assert!(!cache.has(&1u32));
            assert!(!cache.has(&2u32));
            assert_eq!(cache.tier_of(&1u32), None);
            assert_eq!(cache.tier_of(&2u32), None);
        }
    };
}

/// `ttl_survives_a_demotion` for the designs that admit the TTL'd key fast and
/// demote it under filler pressure; `false` for LFU, whose own version admits
/// fillers slow (see `hybrid_suite!`).
#[allow(unused_macros)]
macro_rules! ttl_survives_a_demotion {
    (true, $policy:expr) => {
        #[test]
        fn ttl_survives_a_demotion() {
            ensure_pmem_allocator_warm();

            let cache = PaperCache::<u32, TieredBuffer>::new(1_048_576, CacheTierSize::Bytes(TTL_FAST_TIER), $policy)
                .expect("cache should construct");

            // A *short* TTL here (comparable to `MIGRATION_TIMEOUT`) would make
            // this test racy against `tier_of` itself, which treats an expired
            // object as absent (`None`), same as `get`/`has`: if the object
            // expired before the migration was observed, the `wait_until` below
            // would spin until its own timeout with no way to tell "never
            // migrated" from "migrated but already expired". A TTL comfortably
            // longer than any plausible migration latency avoids that; the
            // assertions below still prove the *original* deadline survived
            // rather than being reset or dropped.
            let ttl_secs = 5u32;
            let set_at = std::time::Instant::now();
            cache.set(1u32, &value(0xC1), Some(ttl_secs)).expect("set should succeed");

            for key in 2u32..=4 {
                cache.set(key, &value(key as u8), None).expect("set should succeed");
            }

            assert!(wait_until(MIGRATION_TIMEOUT, || cache.tier_of(&1u32) == Some(Tier::Slow)));

            // If `Object::set_data` (the migration) had reset or dropped
            // `expiry`, the key would already be gone or immortal here.
            assert!(cache.has(&1u32), "key should still be alive right after migrating");

            // Sleep past the *original* deadline (measured from `set`, not from
            // the migration), proving the original clock kept ticking through
            // the tier move rather than being restarted or cleared.
            let remaining = std::time::Duration::from_millis(ttl_secs as u64 * 1000 + 500)
                .saturating_sub(set_at.elapsed());
            std::thread::sleep(remaining);

            assert!(matches!(cache.get(&1u32), Err(CacheError::KeyNotFound)));
            assert!(!cache.has(&1u32));
        }
    };

    (false, $policy:expr) => {};
}

/// The tests of a design whose queue is an insertion order and whose hit never
/// promotes by itself: FIFO, and CLOCK, whose hit only sets a reference bit --
/// the second chance is the eviction hand's. Invoked inside the binary's `mod
/// hybrid_cache_tests`, after `hybrid_suite!`, for the helpers.
#[allow(unused_macros)]
macro_rules! insertion_order_suite {
    ($policy:expr) => {
        #[test]
        fn fast_tier_pressure_demotes_oldest_object_with_real_data_movement() {
            ensure_pmem_allocator_warm();

            // A fast tier sized to hold ~1 of these ~1 KB values guarantees the
            // first (oldest) key demotes once the second is admitted.
            let cache = PaperCache::<u32, TieredBuffer>::new(1_048_576, CacheTierSize::Bytes(DEMOTES_ONE_OF_TWO), $policy)
                .expect("cache should construct");

            cache.set(1u32, &value(0xA1), None).expect("set should succeed");
            assert_eq!(cache.tier_of(&1u32), Some(Tier::Fast));

            cache.set(2u32, &value(0xB2), None).expect("set should succeed");

            let demoted = wait_until(MIGRATION_TIMEOUT, || {
                cache.tier_of(&1u32) == Some(Tier::Slow)
            });
            assert!(demoted, "key 1 (oldest) should have demoted to the slow tier");

            // Real data movement, not a copy: the key is gone from the fast
            // tier's accounting entirely -- there is only one object map, so
            // "gone from fast" and "present in slow" are the same fact checked
            // two ways.
            assert_ne!(cache.tier_of(&1u32), Some(Tier::Fast));

            // Value survives the physical move intact.
            assert_eq!(cache.get(&1u32).unwrap(), value(0xA1));

            let stats = cache.hybrid_stats();
            assert!(stats.demotions >= 1);
            assert_eq!(stats.promotions, 0);
        }

        #[test]
        fn cascading_demotion_on_repeated_admission_is_handled() {
            ensure_pmem_allocator_warm();

            // There is no promotion to cascade a demotion from -- cascades here
            // come from repeated *new-key* admission into a fast tier sized for
            // ~1 object instead. Exercises the "more than one migration per
            // call" path (the stack's `settle_fast_tier` loop).
            let cache = PaperCache::<u32, TieredBuffer>::new(1_048_576, CacheTierSize::Bytes(DEMOTES_ONE_OF_TWO), $policy)
                .expect("cache should construct");

            for key in 1u32..=4 {
                cache.set(key, &value(key as u8), None).expect("set should succeed");
            }

            assert!(wait_until(MIGRATION_TIMEOUT, || cache.tier_of(&1u32) == Some(Tier::Slow)));
            assert!(wait_until(MIGRATION_TIMEOUT, || cache.tier_of(&2u32) == Some(Tier::Slow)));
            assert!(wait_until(MIGRATION_TIMEOUT, || cache.tier_of(&3u32) == Some(Tier::Slow)));
            assert_eq!(cache.tier_of(&4u32), Some(Tier::Fast));

            for key in 1u32..=4 {
                assert_eq!(cache.get(&key).unwrap(), value(key as u8));
            }
        }

        #[test]
        fn slow_tier_hit_does_not_promote_and_object_stays_slow() {
            ensure_pmem_allocator_warm();

            // The defining difference from LRU: a hit on a slow-tier key must
            // never migrate it back to fast by itself.
            let cache = PaperCache::<u32, TieredBuffer>::new(1_048_576, CacheTierSize::Bytes(DEMOTES_ONE_OF_TWO), $policy)
                .expect("cache should construct");

            cache.set(1u32, &value(0xA1), None).expect("set should succeed");
            cache.set(2u32, &value(0xB2), None).expect("set should succeed");
            assert!(wait_until(MIGRATION_TIMEOUT, || cache.tier_of(&1u32) == Some(Tier::Slow)));

            // A get() on the slow-tier key must never promote it.
            assert_eq!(cache.get(&1u32).unwrap(), value(0xA1));

            // No "wait_until true" assertion is possible for a negative claim;
            // sleep comfortably longer than a real migration would take and
            // confirm the tier never changed.
            std::thread::sleep(std::time::Duration::from_millis(500));
            assert_eq!(cache.tier_of(&1u32), Some(Tier::Slow));

            let stats = cache.hybrid_stats();
            assert_eq!(stats.promotions, 0);
        }

        #[test]
        fn overwriting_an_existing_key_does_not_reposition_it_in_the_queue() {
            ensure_pmem_allocator_warm();

            // Overwriting a key that is still in the fast tier must not
            // reposition it (unlike LRU, which would move it to MRU), and a
            // subsequent demotion must still pick the same oldest key, not
            // whichever key was most recently overwritten.
            //
            // The budget holds exactly three objects and refuses a fourth. It is
            // expressed in what an object COSTS, which is why `VALUE_LEN` is
            // sized to make that cost a round 1 KiB in both value layouts.
            let cache = PaperCache::<u32, TieredBuffer>::new(1_048_576, CacheTierSize::Bytes(3_400), $policy)
                .expect("cache should construct");

            cache.set(1u32, &value(0x11), None).expect("set should succeed"); // oldest
            cache.set(2u32, &value(0x22), None).expect("set should succeed");
            cache.set(3u32, &value(0x33), None).expect("set should succeed"); // newest

            // Overwrite the oldest key. Under LRU semantics this would move it
            // to MRU; here it must stay the oldest.
            cache.set(1u32, &value(0x11), None).expect("overwrite should succeed");
            assert_eq!(cache.tier_of(&1u32), Some(Tier::Fast), "overwrite must not change tier");

            // A fourth admission that forces exactly one demotion should demote
            // key 1 (still oldest by insertion order), never key 2 or key 3.
            cache.set(4u32, &value(0x44), None).expect("set should succeed");

            assert!(wait_until(MIGRATION_TIMEOUT, || cache.tier_of(&1u32) == Some(Tier::Slow)));
            assert_eq!(cache.tier_of(&2u32), Some(Tier::Fast));
            assert_eq!(cache.tier_of(&3u32), Some(Tier::Fast));
            assert_eq!(cache.tier_of(&4u32), Some(Tier::Fast));

            assert_eq!(cache.get(&1u32).unwrap(), value(0x11));
        }
    };
}

#[allow(unused_imports)]
pub(crate) use {hybrid_suite, insertion_order_suite, ttl_survives_a_demotion};
