/*
 * Copyright (c) Kia Shakiba
 *
 * This source code is licensed under the GNU AGPLv3 license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! Backpressure plan S8: the per-cache statistics -- the MIGSTATS counters
//! (the migration queue's depth and dispositions, the batch-size histograms,
//! the reconcile's counters, the eviction-fallback count) are owned by each
//! cache's status and exported through `HybridStats`, where until S8 they were
//! process-global statics that every cache in a process added to.

use std::time::Duration;

use super::*;
use super::test_support::{alone_in_with_env, wait_for};

use crate::gate::{GateConfig, GateMode, GateState, MetadataModel, MetadataOverflow, OnStall};
use crate::{CacheTierSize, HybridStats, PaperCache, TieredBuffer};

type Cache = PaperCache<u64, TieredBuffer>;

/// A tiered cache that demotes: a 16 KiB fast tier under a cap of 1 MiB, with
/// the per-object metadata model (a cache this small is otherwise "metadata
/// bound" and tiers nothing).
fn demoting_cache() -> Cache {
	let mut config = GateConfig::default();
	config.metadata_model = MetadataModel::PerObject;

	Cache::new_with_gate(1 << 20, CacheTierSize::Bytes(16 << 10), PaperPolicy::LruCompactHybrid, config)
		.expect("a tiered cache")
}

fn fill(cache: &Cache, keys: std::ops::Range<u64>) {
	for key in keys {
		cache.set(key, &[key as u8; 1_024], None).expect("a set");
	}
}

/// Every set handled and every migration it queued finished: two whole passes
/// of the policy worker after the sets, then nothing pending in the queue.
fn settle(cache: &Cache) {
	let passes = cache.status.policy_worker_passes();

	wait_for("two more passes of the policy worker", Duration::from_secs(10), || {
		cache.status.kick_policy_worker();
		cache.status.policy_worker_passes() >= passes + 2
	});

	wait_for("the queue to drain", Duration::from_secs(10), || {
		let stats = cache.hybrid_stats();

		stats.pending_demote == 0 && stats.pending_promote == 0
	});
}

/// The migration statistics of a snapshot, whole -- every counter the MIGSTATS
/// lines print and `HybridStats` exports for them, the histograms flattened --
/// but for the eviction passes' own (`evict_calls`, `evict_hist`): the worker
/// ends one at every poll, idle or not, so they move with time and not with
/// work.
fn block(stats: &HybridStats) -> Vec<u64> {
	let mut block = vec![
		stats.queue_depth_max,
		stats.burst_max,
		stats.pending_demote,
		stats.pending_promote,
		stats.pending_demote_max,
		stats.pending_promote_max,
		stats.pending_net_max,
		stats.mig_applied,
		stats.mig_gone,
		stats.mig_declined,
		stats.mig_superseded,
		stats.mig_calls,
		stats.demo_tot,
		stats.promo_tot,
		stats.evict_tot,
		stats.coalesced_tot,
		stats.reconcile_set_to_fast,
		stats.reconcile_set_to_slow,
		stats.reconcile_get_to_fast,
		stats.reconcile_set_new_key,
		stats.reconcile_get_heal_skipped,
		stats.reconcile_queued_to_fast,
		stats.reconcile_queued_to_slow,
		stats.reconcile_applied_to_fast,
		stats.reconcile_applied_to_slow,
		stats.erase_fallbacks,
	];

	block.extend(stats.demo_hist);
	block.extend(stats.promo_hist);

	block
}

/// S8: each cache reports its OWN migrations. One that demotes is built and
/// driven; a second cache built beside it starts at zero -- it reports none of
/// the first's -- and its work moves none of the first's counters; and one
/// built after both are gone starts at zero too ("each policy run should reset
/// all cache state": until S8 these were process-global statics, so the second
/// cache's totals included the first's). Every migration ended one of four
/// ways, and each of the cache's own completions is one of its promotions,
/// demotions or landed correctives. Red with the block shared by every cache
/// (`sharedstats`).
#[test]
fn each_cache_reports_only_its_own_migrations() {
	let _serialised = migration_test_lock::lock();

	let completed = |stats: &HybridStats| {
		stats.promotions + stats.demotions + stats.reconcile_applied_to_fast + stats.reconcile_applied_to_slow
	};

	let a = demoting_cache();

	assert_eq!(block(&a.hybrid_stats()), vec![0; block(&HybridStats::default()).len()], "a new cache starts at zero");

	fill(&a, 0..64);
	settle(&a);

	let first = a.hybrid_stats();

	assert!(first.mig_calls > 0 && first.demo_tot > 0, "the first cache demoted: {first:?}");
	assert!(first.mig_applied > 0, "and moved bytes: {first:?}");
	assert_eq!(first.mig_applied, completed(&first), "every move that landed is a completion");
	assert_eq!(first.demo_hist.iter().sum::<u64>(), first.mig_calls, "one histogram entry per drain");
	assert_eq!(first.promo_hist.iter().sum::<u64>(), first.mig_calls);

	// A second cache, beside it: nothing of the first's.
	let b = demoting_cache();

	assert_eq!(block(&b.hybrid_stats()), vec![0; block(&HybridStats::default()).len()], "a second cache starts at zero");

	fill(&b, 100..132);
	settle(&b);

	let second = b.hybrid_stats();

	assert!(second.mig_calls > 0 && second.demo_tot > 0, "the second cache demoted too: {second:?}");
	assert_eq!(second.mig_applied, completed(&second));

	// Its work moved none of the first's.
	assert_eq!(block(&a.hybrid_stats()), block(&first), "the first cache's counters are its own");

	// Both gone: a third starts at zero.
	drop(a);
	drop(b);

	let c = demoting_cache();

	assert_eq!(block(&c.hybrid_stats()), vec![0; block(&HybridStats::default()).len()], "a cache built after the others starts at zero");
}

/// S8: the eviction loop's fallback (the stack named no victim, so `erase`
/// evicted an arbitrary map entry) is counted per cache -- the DIVERGE line's
/// `fallback` -- and in the process-wide `ERASE_FALLBACK`, which stays. Set up
/// by putting objects in the map the stack never heard of. The DashMap store
/// only: the merged store's index IS its eviction order and has no fallback.
#[cfg(not(feature = "merged_object_store"))]
#[test]
fn the_eviction_fallback_is_counted_per_cache() {
	use crate::object::Object;
	use super::test_support::tiered_worker;

	const POLICY: PaperPolicy = PaperPolicy::LruCompactHybrid;

	let _serialised = migration_test_lock::lock();

	let objects: ObjectMapRef<u64, TieredBuffer> = crate::new_hybrid_object_map();
	let (mut worker, status, overhead_manager) = tiered_worker(objects.clone(), 1 << 30, POLICY, true, false);

	let add = |key: u64| {
		let object = Object::<u64, TieredBuffer>::new(key, &[0u8; 64], None);
		let base = overhead_manager.base_size(&object);

		objects.insert(key, object);
		status.update_base_used_size(base as i64);
		status.incr_num_objects();
	};

	add(0);

	let per_object = status.used_size(&POLICY);

	assert_eq!(per_object, overhead_manager.total_size(&objects.get_ref(&0).unwrap()) as CacheSize);

	// A cap of four objects, six in the map, none of them in the stack: the
	// pass evicts to 98% of four -- three -- each by the fallback.
	status.set_max_size(4 * per_object);

	for key in 1..6 {
		add(key);
	}

	assert_eq!(worker.policy_stack.len(), 0, "the stack knows none of them");

	let global = crate::ERASE_FALLBACK.load(std::sync::atomic::Ordering::Relaxed);

	worker.apply_evictions().expect("a pass");

	assert_eq!(objects.len(), 3, "evicted to 98% of the cap");
	assert_eq!(status.migstats().erase_fallbacks.load(std::sync::atomic::Ordering::Relaxed), 3);
	assert_eq!(status.hybrid_stats().erase_fallbacks, 3);
	assert!(crate::ERASE_FALLBACK.load(std::sync::atomic::Ordering::Relaxed) >= global + 3, "and the process-wide total moved with it");

	// Another cache's count is its own.
	let other_objects: ObjectMapRef<u64, TieredBuffer> = crate::new_hybrid_object_map();
	let (_other, other_status, _) = tiered_worker(other_objects, 1 << 30, POLICY, true, false);

	assert_eq!(other_status.hybrid_stats().erase_fallbacks, 0);
}

/// The tiered cache the gate-configuration tests build: a plain constructor, so
/// it takes the environment's configuration.
fn plain_cache() -> Cache {
	Cache::new(1 << 20, CacheTierSize::Bytes(64 << 10), PaperPolicy::LruCompactHybrid).expect("a tiered cache")
}

/// S8: the `PAPER_GATE_*` environment configures a cache built WITHOUT a gate
/// configuration, and never one built with one -- `new_with_gate` (and
/// `set_gate_config` after it) is used as given, however the environment is
/// set. The precedence the gate module documents, through real caches, in a
/// child process that has the variables from the start (they are read once per
/// process).
#[test]
fn the_environment_configures_a_plain_constructor_and_never_an_explicit_configuration() {
	alone_in_with_env(
		module_path!(),
		"the_environment_configures_a_plain_constructor_and_never_an_explicit_configuration",
		&[
			("PAPER_GATE_MODE", "block"),
			("PAPER_GATE_ON_STALL", "divert"),
			("PAPER_GATE_ON_METADATA_OVERFLOW", "evict_to_fit"),
			("PAPER_GATE_METADATA_MODEL", "per_object"),
			("PAPER_GATE_STALL_WINDOW_MS", "750"),
			("PAPER_GATE_POLL_INTERVAL_US", "500"),
			("PAPER_GATE_SLACK_BYTES", "4096"),
			("PAPER_GATE_NEAR_FRAC", "0.02"),
		],
		|| {
			let mut expected = GateConfig::default();

			expected.mode = GateMode::Block;
			expected.on_stall = OnStall::Divert;
			expected.on_metadata_overflow = MetadataOverflow::EvictToFit;
			expected.metadata_model = MetadataModel::PerObject;
			expected.stall_window = Duration::from_millis(750);
			expected.poll_interval = Duration::from_micros(500);
			expected.slack = 4_096;
			expected.near_frac = 0.02;

			// A plain constructor: the defaults with the environment on top.
			let plain = plain_cache();

			assert_eq!(plain.gate_config(), expected);
			assert_eq!(plain.hybrid_stats().metadata_model, MetadataModel::PerObject, "and the worker runs it");

			// An explicit configuration: exactly what was given.
			let mut given = GateConfig::default();
			given.stall_window = Duration::from_millis(40);

			let explicit = Cache::new_with_gate(1 << 20, CacheTierSize::Bytes(64 << 10), PaperPolicy::LruCompactHybrid, given)
				.expect("a tiered cache");

			assert_eq!(explicit.gate_config(), given, "the environment is not consulted for it");
			assert_eq!(explicit.gate_config().mode, GateConfig::default().mode);

			let defaults = Cache::new_with_gate(1 << 20, CacheTierSize::Bytes(64 << 10), PaperPolicy::LruCompactHybrid, GateConfig::default())
				.expect("a tiered cache");

			assert_eq!(defaults.gate_config(), GateConfig::default(), "not even the defaults are overlaid");

			// And at runtime.
			plain.set_gate_config(given).expect("a valid configuration");
			assert_eq!(plain.gate_config(), given);
		},
	);
}

/// S8: a variable that does not parse, or that the configuration could not
/// run with, is ignored -- the default stays, the valid ones still apply -- and
/// the cache builds.
#[test]
fn an_invalid_environment_value_is_ignored_and_the_valid_ones_apply() {
	alone_in_with_env(
		module_path!(),
		"an_invalid_environment_value_is_ignored_and_the_valid_ones_apply",
		&[
			("PAPER_GATE_MODE", "off"),
			("PAPER_GATE_STALL_WINDOW_MS", "soon"),
			("PAPER_GATE_NEAR_FRAC", "1.5"),
			("PAPER_GATE_ON_STALL", "admit_over"),
		],
		|| {
			let mut expected = GateConfig::default();

			expected.mode = GateMode::Off;
			expected.on_stall = OnStall::AdmitOver;

			assert_eq!(plain_cache().gate_config(), expected);
		},
	);
}

/// S8: a drain target that leaves no room for the default near band
/// (`FAST_TIER_DRAIN_TARGET` of 0.995 against 0.01) already fails
/// `validate` on its bands -- the plain constructor starts with the byte gate
/// disabled (`GateState::Bands`) rather than refusing -- so a variable that
/// leaves the configuration failing in that way and no other is not refused
/// for a failure that was already there, while one that makes it invalid
/// otherwise is; and a valid configuration is not made invalid on its bands
/// (the mode asked to `block` when the bands cannot hold is ignored). In a
/// child process with the drain target from the start (read once per process).
#[test]
fn a_variable_is_not_refused_for_a_failure_that_was_already_there() {
	alone_in_with_env(
		module_path!(),
		"a_variable_is_not_refused_for_a_failure_that_was_already_there",
		&[("FAST_TIER_DRAIN_TARGET", "0.995"), ("PAPER_GATE_MODE", "block"), ("PAPER_GATE_STALL_WINDOW_MS", "100")],
		|| {
			// The default of a release build: `Block`, with a near band of 1%.
			let mut base = GateConfig::default();
			base.mode = GateMode::Block;

			assert!(!base.bands_hold(), "0.995 + 0.01 leaves no room");

			let with = |change: fn(&mut GateConfig)| {
				let mut config = base;
				change(&mut config);
				config
			};

			assert!(base.admits(&with(|config| config.stall_window = Duration::from_millis(100))), "an unrelated variable");
			assert!(base.admits(&with(|config| config.near_frac = 0.008)), "a near band that fails the same way");
			assert!(!base.admits(&with(|config| config.poll_interval = Duration::ZERO)), "but not one invalid otherwise");
			assert!(!base.admits(&with(|config| config.near_frac = 1.5)));

			let mut off = base;
			off.mode = GateMode::Off;

			assert!(off.validate().is_ok());
			assert!(!off.admits(&base), "a valid configuration is not made one whose bands fail");

			// The test build's default is `Off`, so through the environment the mode
			// is ignored (with a note) and the window applies.
			let mut expected = GateConfig::default();
			expected.stall_window = Duration::from_millis(100);

			let cache = plain_cache();

			assert_eq!(cache.gate_config(), expected);
			assert_eq!(cache.hybrid_stats().gate_state, GateState::Off);
		},
	);
}
