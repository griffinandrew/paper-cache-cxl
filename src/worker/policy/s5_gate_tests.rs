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
		.args([name.as_str(), "--exact", "--test-threads=1"])
		.env(CHILD, "1")
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
/// each set -- never exceeds B by more than a value and the fold error. Red
/// with the gate admitting everything (`nogate`): P runs to about 4 MiB.
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
/// over the budget. Red without the watchdog (`nowatchdog`): the set waits
/// until the child is killed.
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
/// progress -- and succeeds. Reds: `notimeout` ((a) waits until the child is
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
/// `FastTierStalled` after the window. Red without the guard (`noguard`): the
/// set waits out its whole 60 s window (the passes the worker made before it
/// died count toward it) and errs `FastTierStalled`.
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
/// Red without that condition (`nopasses`): it errs while the worker is held.
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
			assert!(at >= released, "stalled {:?} BEFORE the worker was released", released - at);
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
			assert!(took < Duration::from_secs(30), "{policy}: took {took:?}");
			assert!(peak <= bound, "{policy}: P peaked at {peak} B, over {bound} B");

			if policy != PaperPolicy::LfuCompactHybrid {
				assert!(stats.gate_waits > 0, "{policy}: no set waited");
			}
		}
	});
}
