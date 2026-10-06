/*
 * Copyright (c) Kia Shakiba
 *
 * This source code is licensed under the GNU AGPLv3 license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! The promotion gate (`gate::PromotionGate`, `GateConfig::promo_frac`): a
//! migration consumer copies a promotion into the fast tier only while
//! `P + v <= min(promo_frac x eff, S)`, otherwise it declines it before the
//! copy, and the policy worker retries or drops the declined key
//! (`PromotionRetry`).
//!
//! P is process-global, so every test here runs ALONE in a child copy of the
//! test binary (`alone`). The levels are chosen so the arithmetic is exact:
//! eff = 2 n v and `promo_frac` 0.5 give a promotion level of exactly n
//! values, so a test can put P on either side of the boundary by one value,
//! and the boundary itself -- `P + v == level` -- is admitted.

use std::{
	collections::HashMap,
	sync::{Arc, Mutex},
};

use super::*;
use super::migration_queue::apply_migration_with;
use super::reconcile_tests::{Objects, Worker, bytes_tier, drain_and_apply};
use super::test_support::{alone_in, tiered_worker};

use crate::{
	CacheSize,
	gate::{self, Gate, GateConfig, MetadataModel, Published, PromotionGate},
	object::Object,
	phys,
};

use Tier::{Fast, Slow};

/// A value's length; `v()` is what P charges for it.
const LEN: usize = 1_000;

/// The promotion level, in values.
const N: u64 = 20;

fn alone(test: &str, body: impl FnOnce()) {
	alone_in(module_path!(), test, body);
}

fn v() -> u64 {
	phys::value_charge::<u64>(LEN as u32)
}

fn p() -> u64 {
	phys::fast_bytes_signed().max(0) as u64
}

/// eff such that `promo_frac` 0.5 puts the promotion level at exactly `N`
/// values.
fn eff() -> CacheSize {
	2 * N * v()
}

fn config(mode: PromotionGate) -> GateConfig {
	let mut config = GateConfig::default();

	config.promotion_gate = mode;
	config.promo_frac = 0.5;
	config.metadata_model = MetadataModel::Measured;

	config
}

/// Publishes eff and the bands the way the worker does when the byte gate is
/// enabled: what sets the promotion level.
fn publish(gate: &Gate, config: GateConfig, enabled: bool) {
	let eff = eff();
	let bands = enabled.then(|| gate::bands(eff, &config));

	gate.set_config(config);

	gate.publish(
		Published { model: MetadataModel::Measured, m_model: 0, eff, eff_small: eff, eff_large: eff, k_max: u64::MAX, bands },
		|| 0,
		|| 0,
	);
}

fn insert(objects: &Objects, key: HashedKey, tier: Tier) {
	objects.insert(key, Object::new_in(key, &vec![key as u8; LEN], tier, None));
}

/// Keys `1..=count`, fast: P is `count` values.
fn fill(objects: &Objects, count: u64) {
	for key in 1..=count {
		insert(objects, key, Fast);
	}
}

fn status() -> StatusRef {
	Arc::new(
		crate::status::AtomicStatus::new(1_000_000, &[PaperPolicy::LfuCompactHybrid], PaperPolicy::LfuCompactHybrid).unwrap(),
	)
}

fn objects() -> Objects {
	crate::new_hybrid_object_map()
}

fn apply(objects: &Objects, status: &StatusRef, key: HashedKey, tier: Tier) -> bool {
	apply_migration_with(objects, key, tier, status.migstats(), Some(status.gate()))
}

/// The promotion gate is a per-consumer check of exact P, so it holds to the
/// byte. Level `N` values: with P at `N - 1` values a promotion fits exactly
/// (`P + v == level` is admitted); one more value of P and it is declined,
/// BEFORE the copy -- the bytes stay slow, P does not move, the decline is
/// counted with its bytes and the key is kept for the worker -- and it lands
/// once a value is freed. A demotion is never held. Reds: no check
/// (`nocheck`), `<` for `<=` (`strict`), the charge taken as 0 (`nocharge`),
/// the check on every tier (`bothtiers`), a refusal not kept (`nokeep`).
#[test]
fn t1_a_promotion_is_held_to_the_level_to_the_byte() {
	alone("t1_a_promotion_is_held_to_the_level_to_the_byte", || {
		let objects = objects();
		let status = status();
		let gate = status.gate();

		publish(gate, config(PromotionGate::On), true);
		assert_eq!(gate.promo_level(), N * v(), "the level is promo_frac of eff");

		fill(&objects, N - 1);
		insert(&objects, 100, Slow);
		insert(&objects, 101, Slow);
		assert_eq!(p(), (N - 1) * v());

		// P + v == level: admitted.
		assert!(apply(&objects, &status, 100, Fast), "a promotion that fits to the byte lands");
		assert_eq!(bytes_tier(&objects, 100), Fast);
		assert_eq!(p(), N * v());
		assert_eq!(gate.stats().promo_gated, 0);

		// One byte over: declined before the copy.
		let before = p();

		assert!(!apply(&objects, &status, 101, Fast), "a promotion past the level is declined");
		assert_eq!(bytes_tier(&objects, 101), Slow, "no copy was made");
		assert_eq!(p(), before, "P did not move");

		let stats = gate.stats();

		assert_eq!((stats.promo_gated, stats.promo_gated_bytes), (1, v()));
		assert_eq!(gate.take_declined(), vec![(101, v() as u32)], "the key is kept for the worker");
		assert!(gate.take_declined().is_empty(), "and handed over once");

		// A demotion is never held, whatever P is.
		for key in 1000..1004 {
			insert(&objects, key, Fast);
		}

		assert!(p() > gate.promo_level());
		assert!(apply(&objects, &status, 1000, Slow), "a demotion is not gated");
		assert_eq!(gate.stats().promo_gated, 1);

		// Room: back under the level by a value, it lands.
		for key in [1001, 1002, 1003] {
			assert!(apply(&objects, &status, key, Slow));
		}

		assert!(apply(&objects, &status, 1, Slow));
		assert!(apply(&objects, &status, 101, Fast), "it lands once there is room");
		assert_eq!(p(), N * v());
		assert_eq!(gate.stats().promo_gated, 1, "and was not declined again");
	});
}

/// Within one drain, demotions are dispatched before promotions, so the bytes
/// they free are free when the promotions are charged: with P exactly at the
/// level, two promotions listed FIRST in the drain still both land, after the
/// two demotions listed behind them. Reds: promotions dispatched first
/// (`promofirst`) -- both declined.
#[test]
fn t2_demotions_are_dispatched_before_promotions() {
	alone("t2_demotions_are_dispatched_before_promotions", || {
		let objects = objects();
		let (worker, status, _) = tiered_worker(objects.clone(), 1 << 30, PaperPolicy::LruCompactHybrid, true, false);

		publish(status.gate(), config(PromotionGate::On), true);

		// A, B (keys 1, 2) and the fillers are fast: P is exactly the level.
		fill(&objects, N);
		insert(&objects, 100, Slow);
		insert(&objects, 101, Slow);
		assert_eq!(p(), N * v());

		worker.apply_migration_batches(vec![(100u64, Fast), (101, Fast), (1, Slow), (2, Slow)], false);

		assert_eq!([100, 101].map(|key| bytes_tier(&objects, key)), [Fast, Fast], "both promotions landed");
		assert_eq!([1, 2].map(|key| bytes_tier(&objects, key)), [Slow, Slow]);
		assert_eq!(status.gate().stats().promo_gated, 0, "none was declined: the demotions were first");
		assert_eq!(p(), N * v());
	});
}

/// The levels. With the gate `On` and the byte gate enabled, the level is
/// `promo_frac x eff`, clamped to S; with the gate `Off` or `Observe`, or with
/// the byte gate not enabled, there is none and nothing is ever declined.
/// Reds: S for the level (`slevel`), the mode ignored (`modeignored`).
#[test]
fn t3_the_level_follows_frac_eff_and_s() {
	alone("t3_the_level_follows_frac_eff_and_s", || {
		let gate = Gate::default();

		publish(&gate, config(PromotionGate::On), true);
		assert_eq!(gate.promo_level(), N * v());
		assert_eq!(gate.promotion_gate(), PromotionGate::On);

		// Above S: clamped to it.
		let mut high = config(PromotionGate::On);
		high.promo_frac = 1.0;
		publish(&gate, high, true);
		assert_eq!(gate.promo_level(), gate.bands().s, "never above the settle target");
		assert!(gate.promo_level() < eff());

		for mode in [PromotionGate::Off, PromotionGate::Observe] {
			publish(&gate, config(mode), true);
			assert_eq!(gate.promo_level(), u64::MAX, "{mode:?}: no level");
			assert_eq!(gate.promotion_gate(), mode);
			assert!(gate.promotion_admit(7, u64::MAX / 2), "{mode:?}: never declines");
		}

		// The byte gate not enabled: off whatever the configuration says.
		publish(&gate, config(PromotionGate::On), false);
		assert_eq!((gate.promotion_gate(), gate.promo_level()), (PromotionGate::Off, u64::MAX));
		assert!(gate.promotion_admit(7, u64::MAX / 2));
		assert!(gate.take_declined().is_empty());

		// The defaults.
		let default = GateConfig::default();

		assert_eq!(default.promotion_gate, PromotionGate::Off, "off unless asked for");
		assert_eq!(default.promo_frac, 0.90);
		assert!(default.validate().is_ok());

		for bad in [0.0, -0.1, 1.5, f64::NAN, f64::INFINITY] {
			let mut config = GateConfig::default();
			config.promo_frac = bad;

			assert!(config.validate().is_err(), "promo_frac {bad}");
		}
	});
}

/// With the gate `Off` a promotion is never declined and nothing is counted
/// or observed; `Observe` declines nothing either but counts what the
/// consumers saw of P against B -- the peak and the integral of the overshoot.
/// Reds: `Off` gating (`offgates`), `Observe` observing nothing (`noobserve`),
/// `Off` observing (`offobserves`).
#[test]
fn t4_off_changes_nothing_and_observe_only_counts() {
	alone("t4_off_changes_nothing_and_observe_only_counts", || {
		for (mode, observed) in [(PromotionGate::Off, false), (PromotionGate::Observe, true)] {
			let objects = objects();
			let status = status();
			let gate = status.gate();

			publish(gate, config(mode), true);

			// P well over the promotion level, and over B for the observation.
			fill(&objects, 2 * N + 5);
			insert(&objects, 100, Slow);

			assert!(apply(&objects, &status, 100, Fast), "{mode:?}: a promotion past the level lands");
			assert_eq!(bytes_tier(&objects, 100), Fast);

			// Builds more samples a gap apart, so the integral has a width.
			std::thread::sleep(std::time::Duration::from_millis(5));
			insert(&objects, 101, Slow);
			assert!(apply(&objects, &status, 101, Fast));

			let stats = gate.stats();
			let over = p() - gate.bands().b;

			assert_eq!((stats.promo_gated, stats.promo_gated_bytes), (0, 0), "{mode:?}");
			assert!(gate.take_declined().is_empty());

			match observed {
				true => {
					assert!(stats.phys_obs >= 2, "{mode:?}: sampled at each copy");
					assert_eq!(stats.phys_over_b_max, over, "the peak P - B (at the last copy)");
					assert!(stats.phys_over_b_byte_us > 0, "an overshoot held for a while has an integral");
				},

				false => assert_eq!((stats.phys_obs, stats.phys_over_b_max, stats.phys_over_b_byte_us), (0, 0, 0), "off observes nothing"),
			}
		}
	});
}

/// What the stack believes, shared with the test: the tier it places each
/// key in (`placement_of`, what the retry asks) and the next drain.
#[derive(Default)]
struct Belief {
	placed: HashMap<HashedKey, Tier>,
	script: Vec<(HashedKey, Tier)>,

	/// The `placement_of` probes answered: what the retry costs the stack.
	probes: u64,
}

struct PlacedStack(Arc<Mutex<Belief>>);

impl PolicyStack for PlacedStack {
	fn is_policy(&self, policy: &PaperPolicy) -> bool {
		matches!(policy, PaperPolicy::LruCompact)
	}

	fn len(&self) -> usize {
		self.0.lock().unwrap().placed.len()
	}

	fn contains(&self, key: HashedKey) -> bool {
		self.0.lock().unwrap().placed.contains_key(&key)
	}

	fn insert(&mut self, key: HashedKey, _size: crate::object::ObjectSize) {
		self.0.lock().unwrap().placed.entry(key).or_insert(Slow);
	}

	fn remove(&mut self, key: HashedKey) {
		self.0.lock().unwrap().placed.remove(&key);
	}

	fn clear(&mut self) {
		self.0.lock().unwrap().placed.clear();
	}

	fn evict_one(&mut self) -> Option<HashedKey> {
		None
	}

	fn drain_tier_migrations(&mut self) -> Vec<(HashedKey, Tier)> {
		std::mem::take(&mut self.0.lock().unwrap().script)
	}

	fn placement_of(&self, key: HashedKey) -> Option<Tier> {
		let mut belief = self.0.lock().unwrap();

		belief.probes += 1;
		belief.placed.get(&key).copied()
	}
}

/// What a worker driven by hand needs: the objects, the gate published `On`
/// with the level at `N` values, a stack whose beliefs the test controls, and
/// the inline path (no queue), so every step is deterministic.
fn rig() -> (Worker, Objects, StatusRef, Arc<Mutex<Belief>>) {
	let objects = objects();
	let (mut worker, status, _) = tiered_worker(objects.clone(), 1 << 30, PaperPolicy::LruCompactHybrid, true, false);
	let belief = Arc::new(Mutex::new(Belief::default()));

	worker.policy_stack = Box::new(PlacedStack(belief.clone()));
	worker.promo_retry.interval = std::time::Duration::ZERO;
	publish(status.gate(), config(PromotionGate::On), true);

	(worker, objects, status, belief)
}

/// A promotion the stack decides is queued; a consumer declines it; and the
/// worker settles it, whichever way the stack moves meanwhile:
///
///   * KEPT while P leaves no room for the copy, and nothing is queued;
///   * RETRIED as a corrective once there is room -- and it lands, so the
///     key's bytes are where the stack places it;
///   * DROPPED once the stack no longer places the key fast (counted);
///   * DROPPED, not duplicated, when the same drain carries a newer entry of
///     the stack for the key (which lands by itself).
///
/// Reds: the room not asked (`noroom`: queued at once and declined again), the smallest-copy shortcut
/// removed (`nomin`: the stack is probed with no room),
/// the placement not asked (`noplacement`: a demoted key promoted), the drain
/// not asked (`nodrain`: the key queued twice), a refusal not kept (`nokeep`).
#[test]
fn t5_a_declined_promotion_is_kept_retried_or_dropped_and_ends_consistent() {
	alone("t5_a_declined_promotion_is_kept_retried_or_dropped_and_ends_consistent", || {
		let (mut worker, objects, status, belief) = rig();
		let gate = status.gate();

		// P is exactly at the level: no promotion fits.
		fill(&objects, N);
		insert(&objects, 100, Slow);
		belief.lock().unwrap().placed.insert(100, Fast);
		belief.lock().unwrap().script = vec![(100, Fast)];

		drain_and_apply(&mut worker);

		assert_eq!(bytes_tier(&objects, 100), Slow, "declined: the stack says fast, the bytes are slow");
		assert_eq!(gate.stats().promo_gated, 1);

		// KEPT: the worker takes the decline, and there is no room.
		drain_and_apply(&mut worker);

		let stats = gate.stats();

		assert_eq!((stats.promo_retried, stats.promo_dropped, stats.promo_retry_pending), (0, 0, 1));
		assert_eq!(bytes_tier(&objects, 100), Slow);
		assert_eq!(stats.promo_gated, 1, "nothing was queued, so nothing was declined again");
		assert_eq!(belief.lock().unwrap().probes, 0, "with no room for the smallest copy, no key was even looked at");

		// RETRIED: a value freed, the corrective lands.
		assert!(apply(&objects, &status, 1, Slow));
		drain_and_apply(&mut worker);

		let stats = gate.stats();

		assert_eq!(bytes_tier(&objects, 100), Fast, "logical and physical placement agree again");
		assert_eq!((stats.promo_retried, stats.promo_dropped, stats.promo_retry_pending), (1, 0, 0));
		assert_eq!(p(), N * v());

		// DROPPED: declined, then the stack demotes the key.
		insert(&objects, 101, Slow);
		belief.lock().unwrap().placed.insert(101, Fast);
		belief.lock().unwrap().script = vec![(101, Fast)];
		drain_and_apply(&mut worker);
		assert_eq!(gate.stats().promo_gated, 2);

		belief.lock().unwrap().placed.insert(101, Slow);
		assert!(apply(&objects, &status, 2, Slow));
		drain_and_apply(&mut worker);

		let stats = gate.stats();

		assert_eq!(bytes_tier(&objects, 101), Slow, "a key the stack no longer places fast is not promoted");
		assert_eq!((stats.promo_retried, stats.promo_dropped, stats.promo_retry_pending), (1, 1, 0));

		// DROPPED, not duplicated: declined, then room, and the same drain
		// carries the stack's own newer promotion of the key.
		insert(&objects, 2000, Fast);
		assert_eq!(p(), N * v());
		insert(&objects, 102, Slow);
		belief.lock().unwrap().placed.insert(102, Fast);
		belief.lock().unwrap().script = vec![(102, Fast)];
		drain_and_apply(&mut worker);
		drain_and_apply(&mut worker);
		assert_eq!(gate.stats().promo_retry_pending, 1);

		assert!(apply(&objects, &status, 3, Slow));
		belief.lock().unwrap().script = vec![(102, Fast)];
		drain_and_apply(&mut worker);

		let stats = gate.stats();

		assert_eq!(bytes_tier(&objects, 102), Fast, "the stack's entry landed");
		assert_eq!((stats.promo_retried, stats.promo_dropped, stats.promo_retry_pending), (1, 2, 0), "the retry did not queue it again");

		// ROOM FOR ONE: two declined, a value freed, and exactly one is queued
		// -- the other waits for the next -- so a call never queues more than
		// fits.
		for key in [103, 104] {
			insert(&objects, key, Slow);
			belief.lock().unwrap().placed.insert(key, Fast);
		}

		belief.lock().unwrap().script = vec![(103, Fast), (104, Fast)];
		drain_and_apply(&mut worker);
		drain_and_apply(&mut worker);
		assert_eq!((gate.stats().promo_retry_pending, gate.stats().promo_gated), (2, 5));

		assert!(apply(&objects, &status, 4, Slow));
		drain_and_apply(&mut worker);

		let stats = gate.stats();

		assert_eq!([103, 104].map(|key| bytes_tier(&objects, key)).iter().filter(|tier| **tier == Fast).count(), 1, "one fits");
		assert_eq!((stats.promo_retried, stats.promo_retry_pending, stats.promo_gated), (2, 1, 5), "one queued, one kept, none declined again");

		assert!(apply(&objects, &status, 5, Slow));
		drain_and_apply(&mut worker);

		let stats = gate.stats();

		assert_eq!([103, 104].map(|key| bytes_tier(&objects, key)), [Fast, Fast]);
		assert_eq!((stats.promo_retried, stats.promo_retry_pending, stats.promo_dropped), (3, 0, 2));
	});
}

/// An overflow of the declined list is dropped and counted, never grown past
/// its cap, and the worker's retry list is bounded the same way.
#[test]
fn t6_the_declined_list_is_bounded() {
	alone("t6_the_declined_list_is_bounded", || {
		let gate = Gate::default();

		publish(&gate, config(PromotionGate::On), true);

		// P at the level: everything is declined.
		let objects = objects();
		fill(&objects, N);

		for key in 0..(gate::PROMO_DECLINED_CAP as u64 + 10) {
			assert!(!gate.promotion_admit(key, v()));
		}

		let stats = gate.stats();

		assert_eq!(stats.promo_gated, gate::PROMO_DECLINED_CAP as u64 + 10);
		assert_eq!((stats.promo_dropped, stats.promo_dropped_overflow), (10, 10));
		assert_eq!(gate.take_declined().len(), gate::PROMO_DECLINED_CAP);
	});
}

/// End to end, through a real cache with its consumers and worker (LFU, a
/// 1 MiB tier, the gate `On` at 0.90): after a burst of sets has left the tier
/// at its settle target (above the promotion level), a READ-ONLY phase of hits
/// on slow keys -- every promotion the stack decides, each with the demotion
/// that pays for it -- cannot take P above where it was, because nothing but a
/// promotion adds to it, and a promotion is copied only under the level: P,
/// sampled flat out meanwhile, never rises past `max(P at the start, level)`,
/// and the consumers declined promotions (the premise: the tier started above
/// the level). Then room appears (100 fast keys deleted) and the declined keys
/// land by themselves, with no further hit: the placement audit finds nothing
/// lagging and the worker's retry set is empty. Reds: no check (`nocheck`),
/// a refusal not kept or not retried (`nokeep`, `noroom`).
#[test]
fn t7_reads_do_not_overshoot_through_promotions_and_the_declined_land_when_room_appears() {
	use super::s5_gate_tests::{cache, gated, value, M0};
	use crate::gate::{OnStall, test_hooks};
	use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
	use std::time::Duration;

	alone("t7_reads_do_not_overshoot_through_promotions_and_the_declined_land_when_room_appears", || {
		let _m = test_hooks::override_m(M0);
		let mut config = gated(Duration::from_secs(5), OnStall::Error);

		config.promotion_gate = PromotionGate::On;
		config.promo_frac = 0.90;

		let cache = cache(PaperPolicy::LfuCompactHybrid, config);
		let gate = cache.status.gate();
		let level = gate.promo_level();

		let quiesce = || {
			let passes = gate.passes();

			super::test_support::wait_for("two worker passes", Duration::from_secs(10), || gate.passes() >= passes + 2);
			super::test_support::wait_for("the migrations to land", Duration::from_secs(10), || cache.migrations_in_flight() == 0);
		};

		assert!(level > 0 && level < gate.bands().s, "the level is under the settle target");

		for key in 0..400u64 {
			cache.set(key, &value(key), None).expect("a set of the burst");
		}

		quiesce();

		let start = p();
		let peak = AtomicU64::new(start);
		let done = AtomicBool::new(false);

		assert!(start > level, "the tier rests above the promotion level ({start} against {level})");

		std::thread::scope(|scope| {
			let sampler = scope.spawn(|| {
				// An exact read of P sums sixteen shards one after another, so a
				// sample taken while copies move bytes between shards can be torn
				// high; a level P really sits at survives three reads running.
				while !done.load(Ordering::Relaxed) {
					peak.fetch_max(p().min(p()).min(p()), Ordering::Relaxed);
				}
			});

			for _ in 0..3 {
				for key in 150..400u64 {
					let _ = cache.get(&key);
				}
			}

			quiesce();
			done.store(true, Ordering::Relaxed);
			sampler.join().expect("the sampler");
		});

		let stats = cache.hybrid_stats();

		assert!(stats.promo_gated > 0, "the consumers declined promotions: the premise of the test");
		assert!(
			peak.load(Ordering::Relaxed) <= start.max(level) + v(),
			"P rose through promotions: peak {} against start {start} and level {level}",
			peak.load(Ordering::Relaxed),
		);

		// Room: 100 of the hot keys (fast: the read phase promoted them, the
		// lowest-count keys it demoted) are deleted. Nothing is hit again.
		let lagging = cache.placement_audit().expect("an audit").lagging;

		assert!(lagging > 0, "keys the stack places fast are in the slow tier: the declined");

		for key in 150..250u64 {
			cache.del(&key).expect("a hot key");
		}

		super::test_support::wait_for("the declined promotions to land", Duration::from_secs(20), || {
			quiesce();

			let audit = cache.placement_audit().expect("an audit");
			let s = cache.hybrid_stats();

			audit.is_clean() && s.promo_retry_pending == 0
		});

		let stats = cache.hybrid_stats();

		assert!(stats.promo_retried > 0, "the worker queued them again");
		assert!(p() <= level + v(), "and the copies stayed under the level");
	});
}

/// The retry is throttled: `drain_reconciled` runs after every event, and
/// looking at the declined keys reads P (sixteen cache lines the consumers
/// write), which at one read per hit slowed the worker enough for its event
/// channel to back up in the first benchmark (16x the migration backlog).
/// Within its interval a call does nothing -- not even take the declined keys
/// -- and the next one after it does. Reds: no throttle (`nothrottle`), the
/// idle early return removed (`noidle`: a flag left hot).
#[test]
fn t8_the_retry_is_throttled() {
	alone("t8_the_retry_is_throttled", || {
		let (mut worker, objects, status, belief) = rig();
		let gate = status.gate();

		worker.promo_retry.interval = std::time::Duration::from_secs(3600);

		fill(&objects, N);
		insert(&objects, 100, Slow);
		belief.lock().unwrap().placed.insert(100, Fast);
		belief.lock().unwrap().script = vec![(100, Fast)];
		drain_and_apply(&mut worker);
		assert_eq!(gate.stats().promo_gated, 1);

		// The first look at the declined key: taken, kept (no room).
		drain_and_apply(&mut worker);
		assert_eq!(gate.stats().promo_retry_pending, 1);

		// Room, but inside the interval: nothing is looked at.
		assert!(apply(&objects, &status, 1, Slow));
		drain_and_apply(&mut worker);
		assert_eq!((gate.stats().promo_retried, bytes_tier(&objects, 100)), (0, Slow), "throttled");
		assert_eq!(belief.lock().unwrap().probes, 0);

		// The interval over: it lands.
		worker.promo_retry.interval = std::time::Duration::ZERO;
		drain_and_apply(&mut worker);
		assert_eq!((gate.stats().promo_retried, bytes_tier(&objects, 100)), (1, Fast));

		// Nothing waiting: the poll flag is down.
		drain_and_apply(&mut worker);
		assert!(!worker.promo_retry.hot, "an idle retry leaves no flag up");
	});
}
