/*
 * Copyright (c) Kia Shakiba
 *
 * This source code is licensed under the GNU AGPLv3 license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! Backpressure plan S5, commit B2: the byte gate, through real tiered caches
//! (design section 7.2, as the liveness and semantics reviews amended it).
//!
//! P is process-global and this binary runs its tests on parallel threads, so
//! every test here runs ALONE, in a child copy of the test binary ([`alone`]):
//! its cache is the only one alive -- the gate runs only for a cache that is
//! P's only user -- and P is that cache's. The child is killed at a hard
//! deadline, so a red run that hangs (a watchdog or a wake-up removed) fails
//! rather than wedging the suite. Timing bounds are several times what a
//! green run takes and named for what they detect: none is a measurement.
//!
//! The tests hold M_model at a constant (`test_hooks::override_m`, under the
//! measured model, so the stacks settle on it too): the levels stay where the
//! test put them however many keys it sets. `fill` then sets values, with the
//! migration consumers paused so nothing frees meanwhile, until one more would
//! not fit the close level B: the next fast set waits. Every test turns off
//! the test builds' consumer flush after each migration batch
//! (`test_hooks::no_flush`), which would spin the worker while the consumers
//! are paused and stop its passes -- the gate's.

use std::{
	sync::{
		Mutex,
		atomic::{AtomicU64, Ordering},
	},
	thread,
	time::{Duration, Instant},
};

use super::*;

use crate::gate::{GateConfig, GateMode, GateState, MetadataModel, OnStall, test_hooks};
// The merged store answers `get_ref` with an inherent method.
#[cfg(not(feature = "merged_object_store"))]
use crate::object_store::ObjectStore;
use crate::{CacheTierSize, PaperCache, TieredBuffer, phys};

use Tier::{Fast, Slow};

type Cache = PaperCache<u64, TieredBuffer>;

/// The fast tier, the M held under it, and so eff = 960 KiB.
const TIER: CacheSize = 1 << 20;
const M0: u64 = 64 * 1024;

/// The value length most sets use, and a cache size with room for every
/// value: nothing is evicted by size.
const LEN: usize = 4_096;
const MAX: CacheSize = 256 << 20;

/// What exact P may read under the truth: one shard's unfolded balance for
/// each thread that can fold at once -- a client and the two consumers (the
/// liveness review's e_fold).
const E_FOLD: CacheSize = 3 * phys::FOLD_BYTES as CacheSize;

const CHILD: &str = "PAPER_GATE_TEST_CHILD";

/// How long a child may run before it is killed and its test failed.
const DEADLINE: Duration = Duration::from_secs(90);

/// Runs `body` in a child process in which `test`, of `module`
/// (`module_path!()`), is the only test, killed at `DEADLINE`. The parent
/// passes only if the child ran exactly that one test and it passed. Also
/// T14's gate half (`s4_tests`).
pub(super) fn alone_in(module: &str, test: &str, body: impl FnOnce()) {
	if std::env::var_os(CHILD).is_some_and(|value| value == "1") {
		body();
		return;
	}

	// libtest names a test by its path without the crate.
	let (_, module) = module.split_once("::").expect("a module path");
	let name = format!("{module}::{test}");
	let path = std::env::temp_dir().join(format!("paper-gate-{}-{test}.out", std::process::id()));
	let file = std::fs::File::create(&path).expect("the child's output file");

	let mut child = std::process::Command::new(std::env::current_exe().expect("this test binary"))
		.args([name.as_str(), "--exact", "--test-threads=1", "--nocapture"])
		.env(CHILD, "1")
		// The consumer count the timing bounds assume (the default).
		.env("MIGRATION_QUEUE_THREADS", "2")
		.stdout(file.try_clone().expect("the output file, twice"))
		.stderr(file)
		.spawn()
		.expect("could not re-run this test binary");

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

	assert!(
		status.is_some_and(|status| status.success()) && output.contains("test result: ok. 1 passed;"),
		"{name}, run alone in a child process ({}):\n{output}",
		status.map_or_else(|| format!("killed after {DEADLINE:?}"), |status| status.to_string()),
	);

	// The child's diagnostics, for a run with --nocapture.
	eprintln!("--- {name}, alone:\n{output}");
}

fn alone(test: &str, body: impl FnOnce()) {
	alone_in(module_path!(), test, body);
}

/// Polls `done` every millisecond until it holds, failing after `deadline`.
fn wait_for(what: &str, deadline: Duration, mut done: impl FnMut() -> bool) {
	let start = Instant::now();

	while !done() {
		assert!(start.elapsed() < deadline, "{what} did not happen within {deadline:?}");
		thread::sleep(Duration::from_millis(1));
	}
}

/// The byte gate on, with `window` and `on_stall`; the measured model, whose M
/// `override_m` holds. The lib's test builds default to `Off` (P is shared
/// with every test running beside them), so each test opts in here.
fn gated(window: Duration, on_stall: OnStall) -> GateConfig {
	let mut config = GateConfig::default();
	config.mode = GateMode::Block;
	config.metadata_model = MetadataModel::Measured;
	config.stall_window = window;
	config.on_stall = on_stall;
	config
}

fn build(policy: PaperPolicy, config: GateConfig) -> Cache {
	Cache::new_with_gate(MAX, CacheTierSize::Bytes(TIER), policy, config).expect("a tiered cache")
}

/// `build`, once the worker has enabled the byte gate.
fn cache(policy: PaperPolicy, config: GateConfig) -> Cache {
	let cache = build(policy, config);
	wait_for("the byte gate to enable", Duration::from_secs(10), || cache.hybrid_stats().gate_state == GateState::Enabled);
	cache
}

/// What one `LEN`-byte value charges P.
fn v() -> CacheSize {
	phys::value_charge::<u64>(LEN as u32)
}

/// P, exact, as the gate reads it.
fn p() -> CacheSize {
	phys::fast_bytes_signed().max(0) as CacheSize
}

fn value(key: u64) -> Vec<u8> {
	vec![key as u8; LEN]
}

/// Sets new keys from `next` while one more value fits the close level: with
/// the consumers paused and M held, nothing moves P or B but these sets, so
/// after it the next fast set of a `LEN`-byte value waits. Returns the next
/// key. For the designs that admit new keys fast (not LFU, whose latch sends
/// them slow once the tier is full).
fn fill(cache: &Cache, mut next: u64) -> u64 {
	let b = cache.hybrid_stats().band_b;
	let limit = next + 10_000;

	assert!(b > 0, "the gate publishes its close level");

	while p() + v() <= b {
		assert!(next < limit, "P never reached the close level");
		cache.set(next, &value(next), None).expect("a fill set is admitted at once");
		next += 1;
	}

	next
}

/// Waits until the worker has handled every event sent before this call --
/// two of its gate passes begun since -- and every migration it queued has
/// landed.
fn quiesce(cache: &Cache) {
	let gate = cache.status.gate();
	let passes = gate.passes();

	wait_for("two worker passes", Duration::from_secs(10), || gate.passes() >= passes + 2);
	wait_for("the migrations to land", Duration::from_secs(10), || cache.migrations_in_flight() == 0);
}

fn waiters(cache: &Cache) -> u64 {
	cache.hybrid_stats().waiters
}

// ---------------------------------------------------------------------------
// Holding the budget, and waking

/// T1: a burst over the fast tier is held to its budget. The consumers
/// paused, one client sets 1,000 values of 4 KiB into a 1 MiB tier (about
/// four times the tier); they resume after 100 ms, inside the watchdog's
/// window. Every set succeeds, some after waiting, and P -- read exact after
/// each set -- never exceeds B by more than a value and the fold error, and a
/// set that found P past N kicked the worker. Reds: the gate admitting
/// everything (`nogate`: P runs to about 4 MiB); no near kick (`nonearkick`).
#[test]
fn t1_a_burst_is_held_to_the_budget() {
	alone("t1_a_burst_is_held_to_the_budget", || {
		let _flush = test_hooks::no_flush();
		let _m = test_hooks::override_m(M0);
		let cache = cache(PaperPolicy::LruCompactHybrid, gated(Duration::from_secs(5), OnStall::Error));
		let b = cache.hybrid_stats().band_b;

		let pause = test_hooks::pause_consumers();
		let peak = AtomicU64::new(0);

		thread::scope(|scope| {
			let client = scope.spawn(|| {
				for key in 0..1_000 {
					cache.set(key, &value(key), None).expect("every set of the burst succeeds");
					peak.fetch_max(p(), Ordering::Relaxed);
					thread::sleep(Duration::from_micros(20));
				}
			});

			thread::sleep(Duration::from_millis(100));
			drop(pause);
			client.join().expect("the client");
		});

		let stats = cache.hybrid_stats();
		let peak = peak.load(Ordering::Relaxed);
		let bound = b + v() + E_FOLD;

		eprintln!("T1: B {b}, v {}, P peaked at {peak} (bound {bound}); {} waits, at most {} waiting", v(), stats.gate_waits, stats.max_waiters);
		assert!(stats.gate_waits > 0, "no set waited: the burst never met the gate");
		assert!(peak <= bound, "P peaked at {peak} B, over the budget's bound {bound} B");
		assert!(stats.near_kicks > 0, "P passed N and no set kicked the worker");
	});
}

/// T2: a waiting set is woken by a migration consumer's landed demotion, not
/// by its poll: the poll is 10 s and the worker's per-pass notify is off, so
/// only the consumer's wake-up admits it inside the bound. Red without it
/// (`nonotify`): about 10 s.
#[test]
fn t2_a_blocked_set_is_woken_by_a_consumer_demotion() {
	alone("t2_a_blocked_set_is_woken_by_a_consumer_demotion", || {
		let _flush = test_hooks::no_flush();
		let _m = test_hooks::override_m(M0);
		let mut config = gated(Duration::from_secs(60), OnStall::Error);
		config.poll_interval = Duration::from_secs(10);
		let cache = cache(PaperPolicy::LruCompactHybrid, config);

		let pause = test_hooks::pause_consumers();
		let next = fill(&cache, 0);
		let _quiet = test_hooks::suppress_pass_notify();

		thread::scope(|scope| {
			let waiter = scope.spawn(|| {
				cache.set(next, &value(next), None).expect("the waiting set is admitted");
				Instant::now()
			});

			wait_for("the set to wait", Duration::from_secs(5), || waiters(&cache) == 1);

			// Well into its first 10 s park.
			thread::sleep(Duration::from_millis(50));

			let resumed = Instant::now();
			drop(pause);

			let took = waiter.join().expect("the waiter").duration_since(resumed);

			eprintln!("T2: admitted {took:?} after the consumers resumed");
			assert!(took < Duration::from_secs(1), "admitted {took:?} after the consumers resumed: not by their wake-up");
		});
	});
}

/// T3: gets never wait at the gate. With a set waiting -- the gate CLOSED, the
/// consumers paused -- 1,000 gets of stored keys finish at once, and the set
/// still waits. Red with gets held while the gate is closed (`gateonget`).
#[test]
fn t3_gets_never_wait() {
	alone("t3_gets_never_wait", || {
		let _flush = test_hooks::no_flush();
		let _m = test_hooks::override_m(M0);
		let cache = cache(PaperPolicy::LruCompactHybrid, gated(Duration::from_secs(60), OnStall::Error));

		let pause = test_hooks::pause_consumers();
		let next = fill(&cache, 0);

		thread::scope(|scope| {
			let waiter = scope.spawn(|| cache.set(next, &value(next), None));

			wait_for("the set to wait", Duration::from_secs(5), || waiters(&cache) == 1);

			let start = Instant::now();

			for n in 0..1_000 {
				let key = n % next;
				assert_eq!(cache.get(&key).expect("a stored key"), value(key));
			}

			let took = start.elapsed();

			assert_eq!(waiters(&cache), 1, "the set still waits");
			drop(pause);
			waiter.join().expect("the waiter").expect("the waiting set is admitted");

			assert!(took < Duration::from_secs(1), "1,000 gets took {took:?} while a set waited");
		});
	});
}

/// T11: waiters are admitted in the order they queued. Eight sets queue one
/// at a time (each once the one before it is counted waiting); the consumers
/// then land one demotion at a time, room for one value each. The lane logs
/// its admissions by join order (`test_admissions`). Red with the lane's
/// newest at its front (`lifo`): the order reversed.
#[test]
fn t11_waiters_are_admitted_in_fifo_order() {
	alone("t11_waiters_are_admitted_in_fifo_order", || {
		let _flush = test_hooks::no_flush();
		let _m = test_hooks::override_m(M0);
		let cache = cache(PaperPolicy::LruCompactHybrid, gated(Duration::from_secs(60), OnStall::Error));

		let pause = test_hooks::pause_consumers();
		let next = fill(&cache, 0);

		thread::scope(|scope| {
			let mut sets = Vec::new();

			for i in 0..8 {
				let cache = &cache;
				sets.push(scope.spawn(move || cache.set(next + i, &value(next + i), None).expect("every waiter is admitted")));
				wait_for("the set to queue", Duration::from_secs(5), || waiters(cache) == i + 1);
			}

			let _pace = test_hooks::pace_consumers(Duration::from_millis(10));
			drop(pause);

			for set in sets {
				set.join().expect("a waiter");
			}
		});

		assert_eq!(*cache.status.gate().test_admissions.lock(), (0..8).collect::<Vec<u64>>(), "admitted out of join order");
	});
}

/// The oversize path: a value larger than the settled tier's headroom
/// `B - S`, though no larger than an empty tier, waits for a settled tier and
/// is admitted only once `P <= S` with nothing reserved -- so P peaks at
/// `S + v`, never `B + v`. Red with it admitted by the normal rule
/// (`oversizeasnormal`: `P + v <= B`, which a tier resting at S never meets):
/// FastTierStalled.
#[test]
fn the_oversize_path_waits_for_a_settled_tier() {
	alone("the_oversize_path_waits_for_a_settled_tier", || {
		let _flush = test_hooks::no_flush();
		let _m = test_hooks::override_m(M0);
		let cache = cache(PaperPolicy::LruCompactHybrid, gated(Duration::from_secs(5), OnStall::Error));

		let s = cache.hybrid_stats();
		let big = 64 * 1024;
		let vb = phys::value_charge::<u64>(big as u32);

		assert!(vb > s.band_b - s.band_s && vb <= s.effective_fast_capacity, "a {vb}-byte value is oversize: B - S = {}", s.band_b - s.band_s);

		let pause = test_hooks::pause_consumers();
		let next = fill(&cache, 0);

		thread::scope(|scope| {
			let waiter = scope.spawn(|| {
				cache.set(next, &vec![7u8; big], None).expect("the oversize set is admitted on a settled tier");
				p()
			});

			wait_for("the oversize set to wait", Duration::from_secs(5), || waiters(&cache) == 1);
			drop(pause);

			let after = waiter.join().expect("the waiter");

			assert_eq!(cache.hybrid_stats().oversize_admits, 1);
			assert_eq!(cache.status.gate().reserved(), 0, "the oversize reservation released");
			assert!(after <= s.band_s + vb + E_FOLD, "P {after} B right after the admission, over S + v");
		});

		assert_eq!(cache.tier_of(&next), Some(Fast), "built fast");
	});
}

// ---------------------------------------------------------------------------
// The watchdog

/// T12: bytes nothing can free -- fast values pinned by readers' handles --
/// honour each `on_stall` once the watchdog's window passes with nothing
/// freed. The fill's values are pinned, and the demotions queued for them
/// land before the set (they refund nothing: the pins hold the old copies).
/// `Error`: FastTierStalled; `Divert`: built slow; `AdmitOver`: built fast,
/// over the budget, its reservation released once built. Reds: no watchdog
/// (`nowatchdog`: the set waits until the child is killed); the admit-over's
/// reservation leaked (`admitoverleak`).
#[test]
fn t12_unfreeable_pinned_bytes_honour_each_on_stall() {
	alone("t12_unfreeable_pinned_bytes_honour_each_on_stall", || {
		let _flush = test_hooks::no_flush();
		let _m = test_hooks::override_m(M0);
		let window = Duration::from_millis(300);

		for on_stall in [OnStall::Error, OnStall::Divert, OnStall::AdmitOver] {
			let cache = cache(PaperPolicy::LruCompactHybrid, gated(window, on_stall));

			let pause = test_hooks::pause_consumers();
			let next = fill(&cache, 0);

			// A reader's handle on every stored value keeps its fast copy alive
			// after a demotion replaces it.
			let pins: Vec<_> = (0..next)
				.filter_map(|key| cache.objects.get_ref(&cache.hash_key(&key)).map(|object| object.snapshot()))
				.collect();

			drop(pause);
			quiesce(&cache);

			let start = Instant::now();
			let result = cache.set(next, &value(next), None);
			let took = start.elapsed();
			let stats = cache.hybrid_stats();

			eprintln!("T12 {on_stall:?}: {result:?} after {took:?}; {} stalls", stats.gate_stalls);
			assert!(took >= window && took < window + Duration::from_secs(3), "{on_stall:?}: acted after {took:?}");
			assert_eq!(stats.gate_stalls, 1, "{on_stall:?}");

			match on_stall {
				OnStall::Error => {
					assert!(matches!(result, Err(CacheError::FastTierStalled)), "{result:?}");
					assert_eq!(stats.gate_stall_errors, 1);
					assert!(!cache.has(&next), "a refused set stores nothing");
				},

				OnStall::Divert => {
					result.expect("a diverted set succeeds");
					assert_eq!((stats.divert_sets, stats.divert_bytes), (1, v()));
					assert_eq!(cache.tier_of(&next), Some(Slow), "built slow");
				},

				OnStall::AdmitOver => {
					result.expect("an admitted-over set succeeds");
					assert_eq!((stats.admit_over_sets, stats.admit_over_bytes), (1, v()));
					assert_eq!(cache.tier_of(&next), Some(Fast), "built fast, over the budget");
					assert_eq!(cache.status.gate().reserved(), 0, "the admit-over's reservation released");
				},
			}

			drop(pins);
		}
	});
}

/// T13: a diverted key -- built slow because the fast tier was stalled, and
/// placed by its policy (LRU: fast, a new key) -- is not corrected toward fast
/// at its `Set`: it LAGS, as the audit reports, until its first slow-served
/// hit heals it; an untouched one costs no copy. `stall_window` 0 acts at
/// once, so both sets divert as soon as they would wait. Red with a diverted
/// `Set` reconciled as a normal one (`reconcilediverted`): both promoted at
/// their Sets.
#[test]
fn t13_a_diverted_key_is_healed_on_its_first_slow_hit_and_untouched_costs_nothing() {
	alone("t13_a_diverted_key_is_healed_on_its_first_slow_hit_and_untouched_costs_nothing", || {
		let _flush = test_hooks::no_flush();
		let _m = test_hooks::override_m(M0);
		let cache = cache(PaperPolicy::LruCompactHybrid, gated(Duration::ZERO, OnStall::Divert));

		let pause = test_hooks::pause_consumers();
		let next = fill(&cache, 0);
		let (touched, untouched) = (next, next + 1);

		for key in [touched, untouched] {
			cache.set(key, &value(key), None).expect("a diverted set succeeds");
		}

		drop(pause);
		quiesce(&cache);

		let before = cache.hybrid_stats();
		let audit = cache.placement_audit().expect("a tiered cache answers the audit");

		assert_eq!(before.divert_sets, 2);
		assert_eq!(before.reconcile_set_to_fast, 0, "a corrective toward fast at a diverted Set");
		assert_eq!((audit.lagging, audit.stranded), (2, 0), "{audit:?}");
		assert_eq!((cache.tier_of(&touched), cache.tier_of(&untouched)), (Some(Slow), Some(Slow)));

		assert_eq!(cache.get(&touched).expect("a hit"), value(touched));
		quiesce(&cache);

		let after = cache.hybrid_stats();

		assert_eq!(after.reconcile_applied_to_fast - before.reconcile_applied_to_fast, 1, "its first slow hit healed it");
		assert_eq!(cache.tier_of(&touched), Some(Fast));
		assert_eq!(cache.tier_of(&untouched), Some(Slow), "untouched: no copy");
		assert_eq!(cache.placement_audit().expect("an audit").lagging, 1);
	});
}

/// T19: the watchdog tells a stuck gate from a slow one. (a) The consumers
/// paused -- a dead consumer, nothing freed -- a waiting set errs with
/// FastTierStalled once the window passes. (b) The tier five values deeper
/// over B than one landing frees, the consumers slowed to one landing per
/// 200 ms each: the set waits longer than the window -- each landing is
/// progress -- and succeeds. Reds: `nowatchdog` ((a) waits until the child is
/// killed); `freedoff` (neither FREED nor a landed demotion counts as
/// progress: (b) errs).
#[test]
fn t19_a_stuck_gate_errs_within_the_window_and_a_slow_one_keeps_waiting() {
	alone("t19_a_stuck_gate_errs_within_the_window_and_a_slow_one_keeps_waiting", || {
		let _flush = test_hooks::no_flush();
		let _m = test_hooks::override_m(M0);
		let window = Duration::from_millis(500);

		{
			let cache = cache(PaperPolicy::LruCompactHybrid, gated(window, OnStall::Error));
			let _pause = test_hooks::pause_consumers();
			let next = fill(&cache, 0);

			let start = Instant::now();
			let result = cache.set(next, &value(next), None);
			let took = start.elapsed();

			eprintln!("T19 (a): {result:?} after {took:?}");
			assert!(matches!(result, Err(CacheError::FastTierStalled)), "(a): {result:?}");
			assert!(took >= window && took < window + Duration::from_secs(2), "(a): stalled after {took:?}");
		}

		{
			let cache = cache(PaperPolicy::LruCompactHybrid, gated(window, OnStall::Error));
			let pause = test_hooks::pause_consumers();
			let next = fill(&cache, 0);

			// Five values' worth over B: the set waits for about six landings.
			let _deeper = test_hooks::override_m(M0 + 5 * v());
			let _pace = test_hooks::pace_consumers(Duration::from_millis(200));
			drop(pause);

			let start = Instant::now();
			cache.set(next, &value(next), None).expect("(b): demotions landing keep the set waiting");
			let took = start.elapsed();

			eprintln!("T19 (b): admitted after {took:?}");
			assert!(took > window, "(b): admitted after {took:?}, inside one window: the test did not span one");
			assert_eq!(cache.hybrid_stats().gate_stalls, 0, "(b): a slow gate called stuck");
		}
	});
}

/// T19b: a waiting set whose policy worker dies returns `Internal` at once --
/// the worker's exit guard marks it gone and wakes every waiter -- not
/// `FastTierStalled` after the window. Red without the guard (`noguard`): a
/// dead worker is then only HUNG -- no pass, no event -- which the watchdog
/// calls a stall after five 60 s windows, so the set waits until the child is
/// killed.
#[test]
fn t19b_a_dead_worker_fails_a_waiting_set_with_internal() {
	alone("t19b_a_dead_worker_fails_a_waiting_set_with_internal", || {
		let _flush = test_hooks::no_flush();
		let _m = test_hooks::override_m(M0);
		let cache = cache(PaperPolicy::LruCompactHybrid, gated(Duration::from_secs(60), OnStall::Error));

		let _pause = test_hooks::pause_consumers();
		let next = fill(&cache, 0);

		thread::scope(|scope| {
			let waiter = scope.spawn(|| cache.set(next, &value(next), None));

			wait_for("the set to wait", Duration::from_secs(5), || waiters(&cache) == 1);

			let killed = Instant::now();
			cache.status.gate().test_panic_on_pass.store(true, Ordering::Relaxed);
			cache.status.kick_policy_worker();

			let result = waiter.join().expect("the waiter");

			assert!(matches!(result, Err(CacheError::Internal)), "{result:?}");
			assert!(killed.elapsed() < Duration::from_secs(2), "Internal {:?} after the worker died", killed.elapsed());
		});
	});
}

/// A late worker is not a stall (the liveness review): with nothing freed and
/// the worker held for three windows, a waiting set does not err -- the
/// watchdog also needs the worker's passes, each with its resettle, since the
/// window began -- and it errs once the worker runs passes that free nothing.
/// Held three windows, it is late, not hung (five). Red with the worker
/// counted as caught up at once (`nocatchup`): it errs while the worker is
/// held; and with the window NOT restarted at catch-up (`norestart`): it errs
/// at once on release, not a window later.
#[test]
fn a_late_worker_is_not_a_stall() {
	alone("a_late_worker_is_not_a_stall", || {
		let _flush = test_hooks::no_flush();
		let _m = test_hooks::override_m(M0);
		let window = Duration::from_millis(300);
		let cache = cache(PaperPolicy::LruCompactHybrid, gated(window, OnStall::Error));

		let _pause = test_hooks::pause_consumers();
		let next = fill(&cache, 0);

		let hold = test_hooks::hold_workers();

		// The worker at its hold.
		thread::sleep(Duration::from_millis(20));

		thread::scope(|scope| {
			let waiter = scope.spawn(|| (cache.set(next, &value(next), None), Instant::now()));

			thread::sleep(3 * window);

			let released = Instant::now();
			drop(hold);

			let (result, at) = waiter.join().expect("the waiter");

			assert!(matches!(result, Err(CacheError::FastTierStalled)), "{result:?}");
			assert!(at >= released, "stalled {:?} BEFORE the worker was released", released.saturating_duration_since(at));
			assert!(
				at >= released + window / 2,
				"stalled {:?} after the worker was released: its window did not start again at catch-up",
				at.saturating_duration_since(released),
			);
		});
	});
}

/// A stall ends at the first byte freed (the liveness review): the worker owns
/// the flag and clears it once anything is freed, so after a stuck spell a
/// later burst waits -- and succeeds -- rather than erring after the short
/// probe. Red with the worker never clearing it (`stickystall`).
#[test]
fn a_stall_ends_at_the_first_byte_freed() {
	alone("a_stall_ends_at_the_first_byte_freed", || {
		let _flush = test_hooks::no_flush();
		let _m = test_hooks::override_m(M0);
		let cache = cache(PaperPolicy::LruCompactHybrid, gated(Duration::from_millis(300), OnStall::Error));

		let pause = test_hooks::pause_consumers();
		let next = fill(&cache, 0);

		let result = cache.set(next, &value(next), None);
		assert!(matches!(result, Err(CacheError::FastTierStalled)), "{result:?}");
		assert!(cache.status.gate().stalled(), "the gate is stalled");

		drop(pause);
		wait_for("the stall to end", Duration::from_secs(5), || !cache.status.gate().stalled());

		// 200 more values into a full tier: each waits for a landing, none errs.
		for key in next..next + 200 {
			cache.set(key, &value(key), None).expect("a set after the stall ended");
		}

		let stats = cache.hybrid_stats();
		assert_eq!((stats.gate_stalls, stats.gate_stall_errors), (1, 1));
	});
}

/// A rise of the close level does not end a stall (fix 3's worker half, the
/// review of this commit): with the consumers paused nothing is freed and P
/// stays put; M_model falling by 48 KiB raises B, and through the worker's
/// next passes the gate stays stalled -- only a byte freed (or P below where it
/// stalled) ends it, as the consumers' landings then do. Red with a rise of B
/// ending the stall, as B2's rule did (`bclear`).
#[test]
fn a_rise_of_the_close_level_does_not_end_a_stall() {
	alone("a_rise_of_the_close_level_does_not_end_a_stall", || {
		let _flush = test_hooks::no_flush();
		let _m = test_hooks::override_m(M0);
		let cache = cache(PaperPolicy::LruCompactHybrid, gated(Duration::from_millis(300), OnStall::Error));

		let pause = test_hooks::pause_consumers();
		let next = fill(&cache, 0);

		let result = cache.set(next, &value(next), None);
		assert!(matches!(result, Err(CacheError::FastTierStalled)), "{result:?}");
		assert!(cache.status.gate().stalled(), "the gate is stalled");

		let b = cache.hybrid_stats().band_b;
		let _lower = test_hooks::override_m(M0 / 4);
		wait_for("B to rise", Duration::from_secs(5), || cache.hybrid_stats().band_b > b);

		let passes = cache.status.gate().passes();
		wait_for("two more passes", Duration::from_secs(5), || cache.status.gate().passes() >= passes + 2);

		eprintln!("B rose {b} -> {} with P {}", cache.hybrid_stats().band_b, p());
		assert!(cache.status.gate().stalled(), "a rise of B ended the stall with nothing freed");

		drop(pause);
		wait_for("the stall to end at a byte freed", Duration::from_secs(5), || !cache.status.gate().stalled());
	});
}

// ---------------------------------------------------------------------------
// What moves under a waiter

/// A waiter decides its tier again at every wake (the liveness review): when
/// eff falls below its value while it waits -- the fast tier shrunk under it,
/// or M grown to fill it -- it leaves the lane as a structural set, built and
/// placed slow, and succeeds. Red without the re-decision (`nowakeplace`): it
/// waits for room that cannot come, and errs.
#[test]
fn a_waiter_whose_value_no_longer_fits_leaves_structural() {
	alone("a_waiter_whose_value_no_longer_fits_leaves_structural", || {
		let _flush = test_hooks::no_flush();
		let _m = test_hooks::override_m(M0);

		for shrink in ["the fast tier", "M"] {
			let cache = cache(PaperPolicy::LruCompactHybrid, gated(Duration::from_secs(2), OnStall::Error));
			let _pause = test_hooks::pause_consumers();
			let next = fill(&cache, 0);

			thread::scope(|scope| {
				let mut sets = Vec::new();

				for i in 0..3 {
					let cache = &cache;
					sets.push(scope.spawn(move || cache.set(next + i, &value(next + i), None)));
					wait_for("the set to queue", Duration::from_secs(5), || waiters(cache) == i + 1);
				}

				let structural = cache.hybrid_stats().structural_slow_sets;

				// eff 1 KiB, or 0: smaller than every waiter's value.
				let _full = match shrink {
					"M" => Some(test_hooks::override_m(TIER)),

					_ => {
						cache.set_fast_tier_size(CacheTierSize::Bytes(M0 + 1024)).expect("a resize");
						None
					},
				};

				for set in sets {
					set.join().expect("a waiter").unwrap_or_else(|error| panic!("{shrink}: {error:?}"));
				}

				assert_eq!(cache.hybrid_stats().structural_slow_sets - structural, 3, "{shrink}");
			});

			for i in 0..3 {
				assert_eq!(cache.tier_of(&(next + i)), Some(Slow), "{shrink}: built slow");
			}
		}
	});
}

/// A table step is absorbed (design 3.8): M jumps by a quarter of eff at a
/// pass, as a hash table doubling would move it, so the tier is suddenly over
/// its close level; the resettle queues the demotions that make room, and a
/// stream of sets waits through it, none erring, while the over-budget
/// integral records the excursion. The consumers are paused for 50 ms after
/// the step, with a set waiting: the integral counts WHOLE byte-seconds, and
/// ~220 KB over the budget makes one only after ~4 ms, which the consumers can
/// beat. Red without the pass end's resettle (`noresettle`): nothing demotes
/// until a set is admitted, and none is.
#[test]
fn a_table_step_is_absorbed_without_a_stall() {
	alone("a_table_step_is_absorbed_without_a_stall", || {
		let _flush = test_hooks::no_flush();
		let _m = test_hooks::override_m(M0);
		let cache = cache(PaperPolicy::LruCompactHybrid, gated(Duration::from_millis(500), OnStall::Error));

		let pause = test_hooks::pause_consumers();
		let next = fill(&cache, 0);
		drop(pause);
		quiesce(&cache);

		let before = cache.hybrid_stats();
		let pause = test_hooks::pause_consumers();
		let _step = test_hooks::override_m(M0 + before.effective_fast_capacity / 4);

		wait_for("the step's publication", Duration::from_secs(5), || cache.hybrid_stats().band_b < before.band_b);

		thread::scope(|scope| {
			let sets = scope.spawn(|| {
				for key in next..next + 100 {
					cache.set(key, &value(key), None).expect("a set through the table step");
				}
			});

			wait_for("a set to wait", Duration::from_secs(5), || waiters(&cache) >= 1);
			thread::sleep(Duration::from_millis(50));
			drop(pause);
			sets.join().expect("the setter");
		});

		let stats = cache.hybrid_stats();

		assert!(stats.gate_waits > before.gate_waits, "no set waited: the step never closed the gate");
		assert_eq!(stats.gate_stalls, 0);
		assert!(stats.over_budget_byte_seconds > before.over_budget_byte_seconds, "the integral missed the excursion");
	});
}

/// A wipe and a grown tier release the waiters at once -- they re-check when
/// the worker empties the cache or publishes the larger eff, not at their
/// next poll (10 s here, with the worker's per-pass notify off); a shrunk
/// tier closes the gate. Reds: `nowipenotify`, `nogrownotify` (each waits out
/// its poll).
#[test]
fn wipe_and_resize_release_waiters() {
	alone("wipe_and_resize_release_waiters", || {
		let _flush = test_hooks::no_flush();
		let _m = test_hooks::override_m(M0);
		let mut config = gated(Duration::from_secs(60), OnStall::Error);
		config.poll_interval = Duration::from_secs(10);

		for release in ["a wipe", "a grow"] {
			let cache = cache(PaperPolicy::LruCompactHybrid, config);
			let pause = test_hooks::pause_consumers();
			let next = fill(&cache, 0);
			let quiet = test_hooks::suppress_pass_notify();

			thread::scope(|scope| {
				let waiter = scope.spawn(|| cache.set(next, &value(next), None));

				wait_for("the set to wait", Duration::from_secs(5), || waiters(&cache) == 1);
				thread::sleep(Duration::from_millis(50));

				let start = Instant::now();

				match release {
					"a wipe" => cache.wipe().expect("a wipe"),
					_ => cache.set_fast_tier_size(CacheTierSize::Bytes(2 * TIER)).expect("a resize"),
				}

				waiter.join().expect("the waiter").unwrap_or_else(|error| panic!("{release}: {error:?}"));
				assert!(start.elapsed() < Duration::from_secs(2), "admitted {:?} after {release}", start.elapsed());
			});

			drop(quiet);

			if release == "a grow" {
				// Shrunk to half: P is over the new close level, and a set waits.
				cache.set_fast_tier_size(CacheTierSize::Bytes(TIER / 2)).expect("a resize");
				wait_for("the smaller levels", Duration::from_secs(5), || cache.hybrid_stats().band_b < p());

				thread::scope(|scope| {
					let set = scope.spawn(|| cache.set(u64::MAX, &value(1), None));

					wait_for("a set to wait on the shrunk tier", Duration::from_secs(5), || waiters(&cache) == 1);
					drop(pause);
					set.join().expect("the waiter").expect("admitted once demotions land");
				});
			}
		}
	});
}

/// A permit's reservation survives a wipe (the liveness review): the wipe
/// resets the gate's counters, never its reservation, so the permit's release
/// after it takes back exactly what it added -- the reservation returns to 0
/// -- and sets are admitted after it. Red with the wipe resetting the
/// reservation (`wipereserved`).
#[test]
fn a_wipe_under_an_outstanding_reservation_wraps_nothing() {
	alone("a_wipe_under_an_outstanding_reservation_wraps_nothing", || {
		let _flush = test_hooks::no_flush();
		let _m = test_hooks::override_m(M0);
		let cache = cache(PaperPolicy::LruCompactHybrid, gated(Duration::from_secs(5), OnStall::Error));
		let gate = cache.status.gate();
		let n = cache.hybrid_stats().band_n;

		let pause = test_hooks::pause_consumers();
		let mut next = 0;

		// To the near level: the next admission reserves its bytes.
		while p() + v() <= n {
			cache.set(next, &value(next), None).expect("a set under N");
			next += 1;
		}

		let permit = cache.begin_set(&next, LEN, None).expect("admitted, reserving");

		assert_eq!(permit.reservation.bytes(), v());
		assert_eq!(gate.reserved(), v());

		cache.wipe().expect("a wipe");
		assert_eq!(gate.reserved(), v(), "the wipe left the reservation");

		drop(permit);
		assert_eq!(gate.reserved(), 0, "released exactly once");

		drop(pause);

		for key in 0..100 {
			cache.set(key, &value(key), None).expect("a set after the wipe");
		}

		assert_eq!(gate.reserved(), 0);
	});
}

// ---------------------------------------------------------------------------
// When the gate does not run

/// The byte gate runs only for the cache that is P's only user (design
/// 3.9.8): a second tiered cache disables it -- at once, through the live
/// count's epoch, before the first cache's worker passes again -- and its
/// waiter is admitted ungated; dropped, the gate is enabled again at the
/// worker's next pass. Red without the epoch (`noepoch`): nothing disables it
/// while the worker is held, and the waiter waits on.
#[test]
fn the_gate_disables_itself_beside_another_cache() {
	alone("the_gate_disables_itself_beside_another_cache", || {
		let _flush = test_hooks::no_flush();
		let _m = test_hooks::override_m(M0);
		let config = gated(Duration::from_secs(60), OnStall::Error);
		let cache = cache(PaperPolicy::LruCompactHybrid, config);

		let _pause = test_hooks::pause_consumers();
		let next = fill(&cache, 0);

		thread::scope(|scope| {
			let waiter = scope.spawn(|| cache.set(next, &value(next), None));

			wait_for("the set to wait", Duration::from_secs(5), || waiters(&cache) == 1);

			let hold = test_hooks::hold_workers();
			thread::sleep(Duration::from_millis(20));

			// Built without its worker running: its construction publishes.
			let other = build(PaperPolicy::LruCompactHybrid, config);

			waiter.join().expect("the waiter").expect("admitted ungated beside another cache");

			let stats = cache.hybrid_stats();
			assert_eq!(stats.gate_state, GateState::NotSole);
			assert!(stats.gate_disabled_sets >= 1);

			// Released before `other` drops: its drop joins its worker.
			drop(hold);
			drop(other);
		});

		wait_for("the gate to enable again", Duration::from_secs(5), || cache.hybrid_stats().gate_state == GateState::Enabled);
	});
}

/// Gate off is B1 (the gate-off golden): under `GateMode::Off` a burst through
/// the tier touches nothing of the byte gate -- no slow path, no wait, no
/// reservation, no counter -- and P runs over the budget as it did before B2.
#[test]
fn mode_off_is_b1() {
	alone("mode_off_is_b1", || {
		let _flush = test_hooks::no_flush();
		let _m = test_hooks::override_m(M0);
		let mut config = gated(Duration::from_secs(2), OnStall::Error);
		config.mode = GateMode::Off;

		let cache = build(PaperPolicy::LruCompactHybrid, config);
		let pause = test_hooks::pause_consumers();

		for key in 0..600 {
			cache.set(key, &value(key), None).expect("an ungated set");
		}

		let peak = p();
		drop(pause);

		let s = cache.hybrid_stats();

		assert_eq!(s.gate_state, GateState::Off);
		assert_eq!(
			(s.gate_slow_paths, s.gate_waits, s.gate_disabled_sets, s.near_kicks, s.oversize_admits, s.max_waiters, s.reserved_bytes),
			(0, 0, 0, 0, 0, 0, 0),
		);
		assert_eq!((s.band_s, s.band_n, s.band_b), (0, 0, 0), "no levels published");
		assert!(peak > 2 * TIER, "ungated, P reached only {peak} B");
	});
}

/// The designs whose settles do not bound their DRAM run ungated (design 0.6,
/// 3.9.8): the lazy-copy LRU (plan P6) and the faithful S3-FIFO fast-admission
/// pair, whose small queue is not clamped to the tier (Q7) -- `Ungated`,
/// whatever else is alive, so this one needs no child process. Red without
/// the exception (`noungated`).
#[cfg(not(feature = "merged_object_store"))]
#[test]
fn designs_whose_settles_do_not_bound_their_dram_run_ungated() {
	let mut config = GateConfig::default();
	config.mode = GateMode::Block;
	config.metadata_model = MetadataModel::PerObject;

	for policy in [
		PaperPolicy::LruLazyCopyCompactHybrid,
		PaperPolicy::S3FifoFaithfulFastAdmissionCompactHybrid(0.1),
		PaperPolicy::S3FifoFaithfulFastAdmissionReprieveCompactHybrid(0.1),
	] {
		let cache = build(policy, config);
		wait_for(&format!("{policy} to run ungated"), Duration::from_secs(10), || cache.hybrid_stats().gate_state == GateState::Ungated);
	}
}

// ---------------------------------------------------------------------------
// Liveness

/// Liveness: four paced clients, 2,000 ops each (70% set, 20% get, 10% del)
/// over their own 500 keys, on a 1 MiB tier with the two migration consumers
/// running, in every order the build's store has: all finish, none errs, sets
/// waited (in the orders that admit new keys fast), and P -- read after every
/// set -- stays within the bound for four setters (section 5): a value per
/// setter past the check, a fold in flight per thread that folds -- the four
/// clients and the two consumers -- and the promotion copies of the gets,
/// which the gate does not hold, landed before their paired demotions
/// (Delta_promo, allowed 16 values here).
#[test]
fn concurrent_clients_finish_under_the_gate() {
	alone("concurrent_clients_finish_under_the_gate", || {
		let _flush = test_hooks::no_flush();
		let _m = test_hooks::override_m(M0);

		for policy in [
			PaperPolicy::LruCompactHybrid,
			PaperPolicy::FifoCompactHybrid,
			PaperPolicy::ClockCompactHybrid,
			PaperPolicy::LfuCompactHybrid,
		] {
			let cache = cache(policy, gated(Duration::from_secs(5), OnStall::Error));
			let b = cache.hybrid_stats().band_b;
			let peak = AtomicU64::new(0);
			let errors = Mutex::new(Vec::new());
			let start = Instant::now();

			thread::scope(|scope| {
				for client in 0..4u64 {
					let (cache, peak, errors) = (&cache, &peak, &errors);

					scope.spawn(move || {
						let mut rng = client.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;

						for op in 0..2_000 {
							rng = rng.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1_442_695_040_888_963_407);
							let key = client * 10_000 + (rng >> 33) % 500;

							match (rng >> 20) % 10 {
								0..7 => match cache.set(key, &value(key), None) {
									Ok(()) => {
										peak.fetch_max(p(), Ordering::Relaxed);
									},

									Err(error) => errors.lock().unwrap().push(format!("set {key}: {error:?}")),
								},

								7..9 => {
									let _ = cache.get(&key);
								},

								_ => {
									let _ = cache.del(&key);
								},
							}

							if op % 16 == 15 {
								thread::sleep(Duration::from_micros(100));
							}
						}
					});
				}
			});

			let took = start.elapsed();
			let stats = cache.hybrid_stats();
			let peak = peak.load(Ordering::Relaxed);
			let bound = b + (4 + 16) * v() + 6 * phys::FOLD_BYTES as CacheSize;

			eprintln!(
				"liveness {policy}: {took:?}, {} waits (longest {} us), P peaked at {peak} (bound {bound}), {} errors",
				stats.gate_waits,
				stats.gate_wait_ns_max / 1_000,
				errors.lock().unwrap().len(),
			);

			assert!(errors.lock().unwrap().is_empty(), "{policy}: {:?}", errors.lock().unwrap());
			assert!(took < Duration::from_secs(15), "{policy}: took {took:?}");
			assert!(peak <= bound, "{policy}: P peaked at {peak} B, over {bound} B");

			if policy != PaperPolicy::LfuCompactHybrid {
				assert!(stats.gate_waits > 0, "{policy}: no set waited");
			}
		}
	});
}

// ---------------------------------------------------------------------------
// Commit C: what the implementation reviews found untested or wrong

/// A worker BEHIND its channel is not a stall (the correctness review): the
/// Sets that filled the tier sit behind ~3,000 gets the worker handles at
/// 1 ms each, so nothing is settled or freed for ~3 s -- six windows -- and a
/// set waiting meanwhile must wait, not err. Only the worker's whole passes
/// count, and the window starts again once it has caught up. FIFO, whose hits
/// migrate nothing, so the gets free nothing either. Red with a pass counted
/// every 1,024 events of a batch while a set waits, as B2's mid-batch gate
/// step did (`midbatchpass`).
#[test]
fn a_backlogged_worker_is_not_a_stall() {
	alone("a_backlogged_worker_is_not_a_stall", || {
		let _flush = test_hooks::no_flush();
		let _m = test_hooks::override_m(M0);
		let window = Duration::from_millis(500);
		let cache = cache(PaperPolicy::FifoCompactHybrid, gated(window, OnStall::Error));

		// At rest: P at the settle target, the consumers running.
		let pause = test_hooks::pause_consumers();
		let next = fill(&cache, 0);
		drop(pause);
		quiesce(&cache);

		// The worker slowed to 1 ms an event, then ~3 s of gets queued ahead.
		let _slow = test_hooks::slow_worker(Duration::from_millis(1));

		for n in 0..3_000 {
			let _ = cache.get(&(n % next));
		}

		// A burst over B: its first sets fit, their Sets queue behind the gets,
		// and the next set waits for the worker to reach them.
		let start = Instant::now();

		for key in next..next + 20 {
			cache.set(key, &value(key), None).expect("a set behind a worker's backlog waits and succeeds");
		}

		let took = start.elapsed();
		let stats = cache.hybrid_stats();

		eprintln!("backlog: 20 sets took {took:?}, {} waits, {} stalls", stats.gate_waits, stats.gate_stalls);
		assert!(stats.gate_waits > 0, "no set waited");
		assert_eq!(stats.gate_stalls, 0, "a backlog called a stall");
		assert!(took > 2 * window, "the burst took {took:?}: the backlog did not outlast two windows");
	});
}

/// A HUNG worker is a stall (the correctness review): with nothing freed and
/// the worker held -- no pass ended, no event handled -- a waiting set errs
/// after five windows, while the worker is still held. (A late worker, held
/// for less, is not: `a_late_worker_is_not_a_stall`.) Red without the hung
/// rule (`nohung`): it waits until the worker is released, at 3 s.
#[test]
fn a_hung_worker_is_a_stall_after_five_windows() {
	alone("a_hung_worker_is_a_stall_after_five_windows", || {
		let _flush = test_hooks::no_flush();
		let _m = test_hooks::override_m(M0);
		let window = Duration::from_millis(200);
		let cache = cache(PaperPolicy::LruCompactHybrid, gated(window, OnStall::Error));

		let _pause = test_hooks::pause_consumers();
		let next = fill(&cache, 0);

		thread::scope(|scope| {
			let hold = test_hooks::hold_workers();

			// The worker at its hold, and released after 3 s whatever happens.
			thread::sleep(Duration::from_millis(20));
			scope.spawn(move || {
				thread::sleep(Duration::from_secs(3));
				drop(hold);
			});

			let start = Instant::now();
			let result = cache.set(next, &value(next), None);
			let took = start.elapsed();

			eprintln!("hung: {result:?} after {took:?}");
			assert!(matches!(result, Err(CacheError::FastTierStalled)), "{result:?}");
			assert!(took >= 5 * window && took < Duration::from_secs(2), "acted after {took:?}, not at five windows of a held worker");
		});
	});
}

/// A worker that MOVES ONCE and then hangs is hung too (the review of this
/// commit): the hung clock runs from the last pass or event the waiter saw, not
/// from its window's start. On a gate set by hand, alone (the frees the
/// watchdog counts are the process's): the head waits; the worker ends ONE
/// pass -- fewer than the two that make it caught up -- or publishes one
/// 64-event step, and then nothing; five windows later the head stalls. Red
/// with the hung rule anchored at the window's start (`hungbaseline`): after
/// one pass the head waits for good.
#[test]
fn a_worker_that_moves_once_and_then_hangs_is_a_stall() {
	alone("a_worker_that_moves_once_and_then_hangs_is_a_stall", || {
		use crate::gate::{Bands, Gate, Published, Waiter, Watch};

		// This process's one tiered registration, so the gate may run.
		let _registered = phys::LiveRegistration::tiered_cache();
		let window = Duration::from_millis(100);

		for step in ["one pass", "one 64-event step"] {
			let gate = Gate::default();

			let mut config = GateConfig::default();
			config.mode = GateMode::Block;
			config.stall_window = window;
			gate.set_config(config);

			let eff: CacheSize = 10_000_000;
			let bands = Bands { s: 9_800_000, n: 9_900_000, b: 10_000_000 };

			gate.publish(
				Published { model: MetadataModel::PerObject, m_model: 0, eff, eff_small: eff, eff_large: eff, k_max: u64::MAX, bands: Some(bands) },
				|| 0,
				|| 0,
			);
			gate.worker_pass(GateState::Enabled, phys::gate_epoch());

			let config = gate.config();
			let mut head = Waiter::enqueue(&gate);
			assert!(head.is_head());
			assert!(matches!(head.watch(&config), Watch::Park(_)), "{step}: a new waiter stalled");

			// The worker's one sign of life after the window began, then nothing.
			match step {
				"one pass" => gate.end_pass(),
				_ => gate.set_worker_progress(64),
			}

			let moved = Instant::now();

			let at = loop {
				match head.watch(&config) {
					Watch::Stalled => break moved.elapsed(),

					Watch::Park(park) => {
						assert!(
							moved.elapsed() < 10 * window,
							"{step}: the head still waits {:?} after the worker's last sign of life",
							moved.elapsed(),
						);
						thread::sleep(park);
					},
				}
			};

			eprintln!("moves once, then hangs ({step}): stalled {at:?} after");
			assert!(at >= 5 * window, "{step}: stalled {at:?} after the worker's last sign of life, before five windows");
			assert!(gate.stalled(), "{step}: the gate is marked stalled");
			drop(head);
		}
	});
}

/// Churn that frees no fast byte is not progress (the correctness review):
/// under the per-object model M_model moves by omega with every key, so every
/// delete raises the close level B a little. The fast tier here is stuck --
/// every value pinned by a reader's handle, M raised three quarters of the tier
/// -- and a client deletes a key every 20 ms, freeing nothing in DRAM (the
/// pins hold every copy). A waiting set must still err after its window. Red
/// with a rise of B counted as progress, as B2 did (`bprogress`): the set
/// waits for as long as the churn lasts, ~4 s.
#[test]
fn churn_that_frees_no_fast_byte_is_not_progress() {
	alone("churn_that_frees_no_fast_byte_is_not_progress", || {
		let _flush = test_hooks::no_flush();
		let window = Duration::from_millis(300);
		let mut config = gated(window, OnStall::Error);
		config.metadata_model = MetadataModel::PerObject;
		let cache = cache(PaperPolicy::LruCompactHybrid, config);

		for key in 0..200 {
			cache.set(key, &value(key), None).expect("under the budget");
		}

		quiesce(&cache);

		let pins: Vec<_> = (0..200)
			.filter_map(|key| cache.objects.get_ref(&cache.hash_key(&key)).map(|object| object.snapshot()))
			.collect();

		let _step = test_hooks::raise_m(3 * TIER / 4);
		wait_for("the step's publication", Duration::from_secs(5), || cache.hybrid_stats().band_b + v() < p());
		quiesce(&cache);

		let stop = std::sync::atomic::AtomicBool::new(false);

		thread::scope(|scope| {
			scope.spawn(|| {
				let mut key = 0;

				while !stop.load(Ordering::Relaxed) {
					let _ = cache.del(&key);
					key = (key + 1) % 200;
					thread::sleep(Duration::from_millis(20));
				}
			});

			let start = Instant::now();
			let result = cache.set(1_000, &value(1_000), None);
			let took = start.elapsed();
			stop.store(true, Ordering::Relaxed);

			eprintln!("churn: {result:?} after {took:?}");
			assert!(matches!(result, Err(CacheError::FastTierStalled)), "{result:?}");
			assert!(took < window + Duration::from_secs(2), "erred after {took:?}: the churn kept it waiting");
		});

		drop(pins);
	});
}

/// While a stall is unresolved, a newcomer acts after the short probe, not a
/// whole window: a stuck state does not cost every set its full window
/// (the liveness review). Red with the probe gone (`noprobe`): the second set
/// waits its whole second.
#[test]
fn a_set_during_a_stall_acts_after_the_probe() {
	alone("a_set_during_a_stall_acts_after_the_probe", || {
		let _flush = test_hooks::no_flush();
		let _m = test_hooks::override_m(M0);
		let window = Duration::from_secs(1);
		let cache = cache(PaperPolicy::LruCompactHybrid, gated(window, OnStall::Error));

		let _pause = test_hooks::pause_consumers();
		let next = fill(&cache, 0);

		let first = cache.set(next, &value(next), None);
		assert!(matches!(first, Err(CacheError::FastTierStalled)), "{first:?}");
		assert!(cache.status.gate().stalled());

		let start = Instant::now();
		let second = cache.set(next + 1, &value(next + 1), None);
		let took = start.elapsed();

		eprintln!("probe: {second:?} after {took:?}");
		assert!(matches!(second, Err(CacheError::FastTierStalled)), "{second:?}");
		assert!(took < Duration::from_millis(500), "the newcomer acted after {took:?}, not after the probe");
	});
}

/// Strict FIFO (the test review): a set that could be admitted still queues
/// behind a waiter that cannot. An OVERSIZE set waits for a settled tier (P at
/// or under S) while P sits just above S, with room under B for small values;
/// three small sets arriving meanwhile queue behind it, and it is admitted
/// first. Reds: newcomers not held behind the lane (`nofifo`); the lane not
/// closing the gate (`noclosed`).
#[test]
fn newcomers_queue_behind_a_waiting_oversize_set() {
	alone("newcomers_queue_behind_a_waiting_oversize_set", || {
		let _flush = test_hooks::no_flush();
		let _m = test_hooks::override_m(M0);
		let cache = cache(PaperPolicy::LruCompactHybrid, gated(Duration::from_secs(30), OnStall::Error));
		let s = cache.hybrid_stats();
		let small = 1_024;

		let pause = test_hooks::pause_consumers();
		let mut next = 0;

		// Just above the settle target, with room under B for the small ones.
		while p() <= s.band_s {
			cache.set(next, &value(next), None).expect("a set under S");
			next += 1;
		}

		assert!(p() + 3 * phys::value_charge::<u64>(small as u32) <= s.band_b, "no room under B for the small sets");

		thread::scope(|scope| {
			let (big, cache_ref) = (next, &cache);
			let oversize = scope.spawn(move || cache_ref.set(big, &vec![7u8; 64 * 1024], None));
			wait_for("the oversize set to wait", Duration::from_secs(5), || waiters(&cache) == 1);

			let mut smalls = Vec::new();

			for i in 0..3 {
				let (cache, key) = (&cache, next + 1 + i);
				smalls.push(scope.spawn(move || cache.set(key, &vec![1u8; small], None)));
				wait_for("a small set to queue behind it", Duration::from_secs(5), || waiters(cache) == 2 + i);
			}

			drop(pause);
			oversize.join().expect("the oversize set").expect("admitted once the tier settled");

			for set in smalls {
				set.join().expect("a small set").expect("admitted after it");
			}
		});

		assert_eq!(*cache.status.gate().test_admissions.lock(), vec![0, 1, 2, 3], "admitted out of join order");
	});
}

/// Turning the gate off releases its waiters at once, not at the worker's next
/// pass (`set_gate_config`): with the worker held, a waiting set is admitted
/// ungated as soon as the configuration says Off. Red without the release
/// (`nooffrelease`).
#[test]
fn turning_the_gate_off_releases_its_waiters_at_once() {
	alone("turning_the_gate_off_releases_its_waiters_at_once", || {
		let _flush = test_hooks::no_flush();
		let _m = test_hooks::override_m(M0);
		let cache = cache(PaperPolicy::LruCompactHybrid, gated(Duration::from_secs(60), OnStall::Error));

		let _pause = test_hooks::pause_consumers();
		let next = fill(&cache, 0);

		thread::scope(|scope| {
			let waiter = scope.spawn(|| cache.set(next, &value(next), None));
			wait_for("the set to wait", Duration::from_secs(5), || waiters(&cache) == 1);

			// Dropped before the scope joins the waiter, even unwinding.
			let _hold = test_hooks::hold_workers();
			thread::sleep(Duration::from_millis(20));

			let mut off = cache.gate_config();
			off.mode = GateMode::Off;
			cache.set_gate_config(off).expect("a valid configuration");

			wait_for("the waiter's release", Duration::from_secs(2), || waiters(&cache) == 0);
			waiter.join().expect("the waiter").expect("admitted ungated");
			assert_eq!(cache.hybrid_stats().gate_state, GateState::Off);
		});
	});
}

/// A diverted OVERWRITE is not promoted at its `Set` (the test review): a slow
/// key overwritten while the tier is stalled is built slow and placed by LRU at
/// the front, fast -- the promotion the stack queues for it there is dropped
/// (`drain_reconciled`), so it lags until its first slow-served hit heals it.
/// Red without the drop (`noretain`): the promotion lands at the Set.
#[test]
fn a_diverted_overwrite_is_not_promoted_at_its_set() {
	alone("a_diverted_overwrite_is_not_promoted_at_its_set", || {
		let _flush = test_hooks::no_flush();
		let _m = test_hooks::override_m(M0);
		let cache = cache(PaperPolicy::LruCompactHybrid, gated(Duration::ZERO, OnStall::Divert));

		// The oldest keys demoted and landed: key 0 is slow.
		let pause = test_hooks::pause_consumers();
		let next = fill(&cache, 0);
		drop(pause);
		quiesce(&cache);
		assert_eq!(cache.tier_of(&0), Some(Slow));

		// The tier full again, and key 0 overwritten: it would wait, so it diverts.
		let pause = test_hooks::pause_consumers();
		fill(&cache, next);
		cache.set(0, &value(0), None).expect("a diverted overwrite succeeds");
		assert_eq!(cache.hybrid_stats().divert_sets, 1);
		drop(pause);
		quiesce(&cache);

		assert_eq!(cache.tier_of(&0), Some(Slow), "promoted at its Set");
		assert!(cache.placement_audit().expect("an audit").lagging >= 1);

		assert_eq!(cache.get(&0).expect("a hit"), value(0));
		quiesce(&cache);
		assert_eq!(cache.tier_of(&0), Some(Fast), "healed on its first slow hit");
	});
}

/// Bytes freed by DELETES keep a waiter waiting: with the consumers paused no
/// demotion lands, and a client deletes a fast key every 200 ms -- each frees
/// its bytes (FREED, which the lane's first waiter turns on) -- while a set of
/// three values' size waits for room: longer than its window, and it
/// succeeds. Red with the first waiter not watching FREED (`nowatchfirst`):
/// nothing counts as progress, and it errs.
#[test]
fn bytes_freed_by_deletes_keep_a_waiter_waiting() {
	alone("bytes_freed_by_deletes_keep_a_waiter_waiting", || {
		let _flush = test_hooks::no_flush();
		let _m = test_hooks::override_m(M0);
		let window = Duration::from_millis(500);
		let cache = cache(PaperPolicy::LruCompactHybrid, gated(window, OnStall::Error));

		let _pause = test_hooks::pause_consumers();
		let next = fill(&cache, 0);

		thread::scope(|scope| {
			let waiter = scope.spawn(|| {
				let start = Instant::now();
				(cache.set(next, &vec![1u8; 3 * LEN], None), start.elapsed())
			});

			wait_for("the set to wait", Duration::from_secs(5), || waiters(&cache) == 1);

			// Every key is still physically fast: nothing has landed.
			for key in 0..next {
				if waiters(&cache) == 0 {
					break;
				}

				thread::sleep(Duration::from_millis(200));
				cache.del(&key).expect("a delete");
			}

			let (result, took) = waiter.join().expect("the waiter");

			eprintln!("deletes: {result:?} after {took:?}");
			result.expect("freed bytes keep it waiting, and it is admitted");
			assert!(took > window, "admitted after {took:?}: the test did not span a window");
		});

		assert_eq!(cache.hybrid_stats().gate_stalls, 0);
	});
}

/// The fast path and the fold hook, on a tier large enough that the near flag
/// can be clear (`approx + E < N` needs N above E = 2 MiB): sets under N - E
/// take the one-load path (no slow path counted); then, with the worker held
/// so that no publication can set the flag, a burst past N - E is caught by
/// the FOLDS alone -- the flag set, the exact path taken -- and P stays within
/// the budget. Reds: no fold hook on a charge (`nohook`); no fast path
/// (`nofastpath`).
#[test]
fn a_large_tier_takes_the_fast_path_and_its_folds_set_near() {
	alone("a_large_tier_takes_the_fast_path_and_its_folds_set_near", || {
		let _flush = test_hooks::no_flush();
		let _m = test_hooks::override_m(1 << 20);
		let tier: CacheSize = 64 << 20;
		let cache = Cache::new_with_gate(MAX, CacheTierSize::Bytes(tier), PaperPolicy::LruCompactHybrid, gated(Duration::from_secs(5), OnStall::Error))
			.expect("a tiered cache");
		wait_for("the byte gate to enable", Duration::from_secs(10), || cache.hybrid_stats().gate_state == GateState::Enabled);

		let s = cache.hybrid_stats();
		let len = 64 * 1024;
		let vb = phys::value_charge::<u64>(len as u32);
		let e = phys::FOLD_ERROR as CacheSize;
		let data = vec![5u8; len];

		let _pause = test_hooks::pause_consumers();
		let mut key = 0;

		// Under N - E, with a page of margin: the fast path.
		while p() + vb + e + (1 << 20) < s.band_n {
			cache.set(key, &data, None).expect("a set under N - E");
			key += 1;
		}

		let under = cache.hybrid_stats();
		assert!(key > 100, "only {key} sets under N - E");
		assert_eq!(under.gate_slow_paths, s.gate_slow_paths, "sets under N - E took the exact path");

		// Past N - E with no publication possible: the folds set the flag.
		let _hold = test_hooks::hold_workers();
		thread::sleep(Duration::from_millis(20));

		while p() + vb <= s.band_n {
			cache.set(key, &data, None).expect("a set under N");
			key += 1;
		}

		let over = cache.hybrid_stats();
		eprintln!("large tier: {key} sets, slow paths {} -> {}, P {} (N {}, B {})", under.gate_slow_paths, over.gate_slow_paths, p(), s.band_n, s.band_b);
		assert!(over.gate_slow_paths > under.gate_slow_paths, "no set past N - E took the exact path: the folds never set NEAR");
		assert!(p() <= s.band_b + vb + E_FOLD);
	});
}

/// The byte decision on a gate set by hand, alone (P and the live counts are
/// the process's): a small value takes the fast path whatever P reads, with no
/// slow path counted; a value over the page figure takes the exact path --
/// admitted under B, and an OVERSIZE one only at P <= S with nothing reserved
/// (with the near flag CLEAR, which no tier in the other tests has); behind a
/// waiter a newcomer waits and the head does not. Reds: no fast path
/// (`nofastpath`); newcomers not held behind the lane (`nofifo`).
#[test]
fn the_byte_decision_on_a_hand_set_gate() {
	alone("the_byte_decision_on_a_hand_set_gate", || {
		use crate::gate::{Bands, Bytes, Gate, Published, Waiter};

		// This process's one tiered registration, so the gate may run.
		let _registered = phys::LiveRegistration::tiered_cache();
		let gate = Gate::default();

		let mut config = GateConfig::default();
		config.mode = GateMode::Block;
		gate.set_config(config);

		let eff: CacheSize = 10_000_000;
		let bands = Bands { s: 9_800_000, n: 9_900_000, b: 10_000_000 };

		gate.publish(
			Published { model: MetadataModel::PerObject, m_model: 0, eff, eff_small: eff, eff_large: eff, k_max: u64::MAX, bands: Some(bands) },
			|| 0,
			|| 0,
		);
		gate.worker_pass(GateState::Enabled, phys::gate_epoch());

		let admits = |bytes: Bytes<'_>| matches!(bytes, Bytes::Admit(_));
		let slow = || gate.stats().gate_slow_paths;

		// The fast path: one load, whatever P says.
		assert!(admits(gate.admit_bytes(4_096, false, || i64::MAX, || {})));
		assert_eq!(slow(), 0, "a small value took the exact path");

		// Over the page figure (B - N): the exact path, admitted under B.
		assert!(admits(gate.admit_bytes(150_000, false, || 9_000_000, || {})));
		assert!(!admits(gate.admit_bytes(150_000, false, || 9_900_000, || {})), "P + v over B");

		// OVERSIZE (v > B - S), the near flag clear: only on a settled tier.
		assert!(!admits(gate.admit_bytes(300_000, false, || 9_800_001, || {})), "oversize over S");
		assert!(admits(gate.admit_bytes(300_000, false, || 9_800_000, || {})), "oversize at S");
		assert_eq!(slow(), 4);

		// Behind a waiter: a newcomer waits, the head does not.
		let head = Waiter::enqueue(&gate);
		assert!(head.is_head());
		assert!(!admits(gate.admit_bytes(4_096, false, || 0, || {})), "a newcomer overtook the lane");
		assert!(admits(gate.admit_bytes(4_096, true, || 0, || {})), "the head was held");
		drop(head);

		assert_eq!(gate.reserved(), 0, "every reservation released");
	});
}

/// A FLAT cache with fast values disables the byte gate too: its values are in
/// P as well (design 3.9.8, the test review). With the workers held, the
/// gate's next exact-path set sees the live counts move and turns it off.
/// Red with P's sole-user check blind to flat caches (`noflat`).
#[test]
fn a_flat_fast_cache_beside_disables_the_gate() {
	alone("a_flat_fast_cache_beside_disables_the_gate", || {
		let _flush = test_hooks::no_flush();
		let _m = test_hooks::override_m(M0);
		let cache = cache(PaperPolicy::LruCompactHybrid, gated(Duration::from_secs(5), OnStall::Error));

		let hold = test_hooks::hold_workers();
		thread::sleep(Duration::from_millis(20));

		let flat = PaperCache::<u64, crate::BufferDRAM>::new(1 << 20, &[PaperPolicy::Lru], PaperPolicy::Lru)
			.expect("a flat cache with fast values");

		// A fast set on the 1 MiB tier takes the exact path, which reads the
		// live counts.
		let checked = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
			cache.set(0, &value(0), None).expect("a set");
			assert_eq!(cache.hybrid_stats().gate_state, GateState::NotSole);
		}));

		// Released before the flat cache drops, pass or fail: its drop joins
		// its worker (a failure used to hang to the child's deadline).
		drop(hold);
		drop(flat);

		if let Err(panic) = checked {
			std::panic::resume_unwind(panic);
		}

		wait_for("the gate to enable again", Duration::from_secs(5), || cache.hybrid_stats().gate_state == GateState::Enabled);
	});
}
