/*
 * Copyright (c) Kia Shakiba
 *
 * This source code is licensed under the GNU AGPLv3 license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! Backpressure plan S4: the merged store does its policy work on the policy
//! worker, as the DashMap stores do -- the client publishes, the worker links,
//! charges, places, settles and retires.
//!
//! Every test here runs in every unit build, over the build's own store,
//! unless it is gated to the merged builds (the races only that store can
//! have). They drive the worker by hand, as its loop drives it, with
//! `reconcile_tests`' harness: `publish` is exactly `PaperCache::set`'s client
//! half, `publish_del` `del`'s, `handle` the worker's handling of a `Set`.
//!
//!   * T14 (`t14`): the uniform differential. One scripted sequence per
//!     order -- tiered and flat -- with the two per-object constants that
//!     differ by design forced equal, every op's drains, victims, placements
//!     and stats recorded as text. With `PAPER_T14_DIR` set each test writes
//!     its file there, and the bp-s4 runner requires the files of the
//!     DashMap, merged and hashbrown builds to be identical (and the three
//!     thin-header builds'). T14b (S4's follow-ups) is the same differential
//!     without the gauge refresh after every op, with TTL reaps, bursts of
//!     sets published before the worker takes them, and wipes.
//!   * T15, through the worker: a client's set and delete move nothing the
//!     stack reports until the worker takes the event (the store-level T15
//!     is in `merged_store`'s tests).
//!   * T8: a promotion its own settle undoes queues only the settle's entry.
//!   * U12: the LFU latch is published with the event that moves it.
//!   * U15: LFU counts its settle's demotions, not a refused admission.
//!   * U9: the CLOCK hand stops at its budget in both stores.
//!   * U7: `wipe` is done by the worker, and returns when it is; with the
//!     worker gone it clears the cache itself; and a wipe racing clients
//!     leaves the status counting exactly what the map holds.
//!   * The races of design section 6, event orders arranged by hand.
//!   * A concurrent workload on real caches, then every charge checked.

use std::collections::HashMap;
use std::time::{Duration, Instant};

use super::*;
use super::reconcile_tests::{
	Objects, Worker, FAST, LEN, assert_settled, bytes_tier, correctives, drain_and_apply,
	evict_one_key, fill, handle, make_worker, of, placement, publish, publish_del, set,
};

// What only the merged store's races need.
#[cfg(feature = "merged_object_store")]
use super::reconcile_tests::Published;
#[cfg(feature = "merged_object_store")]
use std::num::NonZeroU32;
#[cfg(feature = "merged_object_store")]
use crate::object::Object;

use crate::hybrid_policy::admission_tier;
use crate::object::overhead::{OverheadManager, test_overheads};
use crate::status::AtomicStatus;
use crate::{CacheTierSize, PaperCache, TieredBuffer};

// The DashMap and hashbrown maps answer these through the trait; the merged
// store with inherent methods.
#[cfg(not(feature = "merged_object_store"))]
use crate::object_store::ObjectStore;

use Tier::{Fast, Slow};

/// The four orders the merged store implements, as tiered policies.
const TIERED: [PaperPolicy; 4] = [
	PaperPolicy::LruCompactHybrid,
	PaperPolicy::FifoCompactHybrid,
	PaperPolicy::ClockCompactHybrid,
	PaperPolicy::LfuCompactHybrid,
];

fn stack(worker: &Worker) -> &dyn PolicyStack {
	worker.policy_stack.as_deref().expect("a stack")
}

/// What the stack reports of its tiers: fast and slow bytes and objects, and
/// the metadata reservation.
fn gauges(worker: &Worker) -> (CacheSize, CacheSize, usize, usize, CacheSize) {
	let stack = stack(worker);

	(
		stack.fast_bytes_used(),
		stack.slow_bytes_used(),
		stack.fast_object_count(),
		stack.slow_object_count(),
		stack.dram_reserved_bytes(),
	)
}

/// The merged store's own oracle (`MergedStore::verify_charges`): invariant I,
/// the link count and, at quiescence, no DEAD or unlinked slot. Nothing to
/// check in the DashMap stores, whose client writes only the map.
fn charges_exact(objects: &Objects, quiescent: bool) {
	#[cfg(feature = "merged_object_store")]
	objects.verify_charges(quiescent);

	#[cfg(not(feature = "merged_object_store"))]
	let _ = (objects, quiescent);
}

/// What one `len`-byte value is charged to a tier (`Slot::migrating`, the
/// DashMap stacks' `size - dram_resident`).
#[cfg(feature = "merged_object_store")]
fn charge(len: usize) -> CacheSize {
	crate::object::overhead::resident_object_bytes::<u64>(len as ObjectSize) as CacheSize
}

/// `publish` of a value that has ALREADY EXPIRED (its expiry is tick 1), for
/// the TTL reaper's cases.
#[cfg(feature = "merged_object_store")]
fn publish_expired(worker: &Worker, objects: &Objects, key: HashedKey, len: usize) -> Published {
	let object = Object::with_expiry_in(key, &vec![key as u8; len], Fast, NonZeroU32::new(1));
	let base_size = worker.overhead_manager.base_size(&object);
	let resident = worker.overhead_manager.dram_resident_size(&object);
	let mark = worker.status.migration_in_flight().mark(key);

	let previous = objects.insert(key, object).map(|old| {
		let old_size = worker.overhead_manager.base_size(&old);
		worker.status.update_base_used_size(base_size as i64 - old_size as i64);
		old_size
	});

	if previous.is_none() {
		worker.status.incr_num_objects();
		worker.status.update_base_used_size(base_size as i64);
	}

	Published { base_size, resident, built: Fast, previous, mark }
}

/// The TTL reaper's erase of a due key -- what `TtlWorker::reap_due` does
/// before it sends `Expire`.
#[cfg(feature = "merged_object_store")]
fn reap(worker: &Worker, objects: &Objects, key: HashedKey) {
	let _ = erase(objects, &worker.status, &worker.overhead_manager, Some(EraseKey::Expired(key)));
}

/// T15 through the worker, in both stores: a client's set -- of a new key,
/// and over a placed fast and a placed slow one, resized -- leaves every
/// gauge the stack reports, and the key's placement, as they were until the
/// worker takes its `Set`; a client's delete leaves the gauges as they were
/// until the worker takes its `Del`. So in the DashMap stores, whose client
/// writes only the map; in the merged store it is S4 (red before it: the
/// client's insert linked, charged and settled, and its delete uncharged).
#[test]
fn t15_a_clients_set_and_delete_move_nothing_the_stack_reports_until_the_worker_takes_them() {
	let _serialised = migration_test_lock::lock();

	for policy in TIERED {
		let (mut worker, objects) = make_worker(policy);

		// Past the tier: fast and slow keys, and the LFU latched.
		fill(&mut worker, &objects, 1..=24, LEN);

		let fast_key = (1..=24).rev().find(|&k| placement(&worker, k) == Some(Fast)).expect("a fast key");
		let slow_key = (1..=24).find(|&k| placement(&worker, k) == Some(Slow)).expect("a slow key");

		const K: HashedKey = 100;

		let before = gauges(&worker);
		let published = publish(&worker.status, &worker.overhead_manager, &objects, K, LEN, Fast);

		assert_eq!(gauges(&worker), before, "{policy}: a new key's publish moved the stack's gauges");
		assert_eq!(placement(&worker, K), None, "{policy}: a new key is placed before its Set");

		handle(&mut worker, K, published);
		drain_and_apply(&mut worker);

		for key in [fast_key, slow_key] {
			let before = (gauges(&worker), placement(&worker, key));
			let built = bytes_tier(&objects, key);
			let published = publish(&worker.status, &worker.overhead_manager, &objects, key, LEN + 700, built);

			assert_eq!(
				(gauges(&worker), placement(&worker, key)),
				before,
				"{policy}: an overwrite's publish moved the stack's gauges or the placement",
			);

			handle(&mut worker, key, published);
			drain_and_apply(&mut worker);
		}

		let before = gauges(&worker);
		publish_del(&worker.status, &worker.overhead_manager, &objects, fast_key);

		assert_eq!(gauges(&worker), before, "{policy}: a delete moved the stack's gauges before its Del");

		worker.handle_del(fast_key);
		drain_and_apply(&mut worker);

		assert_settled(&mut worker);
		charges_exact(&objects, true);
	}
}

/// T8, in both stores: a touch whose promotion its own settle undoes queues
/// ONLY the settle's entry -- never `(k, Fast)` followed by `(k, Slow)`, a
/// pair the split turns into one entry in a drain and the queue into a round
/// trip across drains. The DashMap stacks push a promotion after the settle
/// and only if the key is still fast; the merged store did it the other way
/// round until S4 put its settle on the worker, where the check against the
/// log costs no lock. One case per touch path, with a fast tier of one byte,
/// so no value fits: an LRU hit and an LRU overwrite of a slow key, a CLOCK
/// second chance of a referenced slow key, an LFU hit and an LFU overwrite.
/// Each event's raw drain (before the split) holds `[Slow]` for the key, its
/// bytes stay slow, and nothing heals it.
#[test]
fn t8_a_promotion_its_own_settle_undoes_queues_only_the_settles_entry() {
	let _serialised = migration_test_lock::lock();

	const K: HashedKey = 1;
	const X: HashedKey = 2;

	let tiny = |worker: &mut Worker| {
		worker.handle_resize_fast_tier(1);
		drain_and_apply(worker);
	};

	for (policy, overwrite) in [
		(PaperPolicy::LruCompactHybrid, false),
		(PaperPolicy::LruCompactHybrid, true),
		(PaperPolicy::LfuCompactHybrid, false),
		(PaperPolicy::LfuCompactHybrid, true),
	] {
		let (mut worker, objects) = make_worker(policy);

		set(&mut worker, &objects, K, LEN, Fast);
		drain_and_apply(&mut worker);
		tiny(&mut worker);
		assert_eq!((placement(&worker, K), bytes_tier(&objects, K)), (Some(Slow), Slow), "{policy}: demoted");

		let drain = match overwrite {
			false => {
				worker.handle_get(K, Some(Slow));
				drain_and_apply(&mut worker)
			},

			true => {
				let built = admission_tier(policy, K, &worker.status, &objects);
				set(&mut worker, &objects, K, LEN, built);
				drain_and_apply(&mut worker)
			},
		};

		let what = if overwrite { "overwrite" } else { "hit" };

		assert_eq!(of(&drain, K), vec![Slow], "{policy} {what}: the settle's entry alone, no promote-then-demote pair");
		assert_eq!(correctives(&drain, K), vec![], "{policy} {what}: no heal, no corrective");
		assert_eq!(bytes_tier(&objects, K), Slow, "{policy} {what}: the bytes stay slow");
		assert_settled(&mut worker);
		charges_exact(&objects, true);
	}

	// CLOCK: the hand's second chance promotes a referenced slow key, which
	// its settle demotes again; the unreferenced key behind it is evicted.
	let (mut worker, objects) = make_worker(PaperPolicy::ClockCompactHybrid);

	set(&mut worker, &objects, K, LEN, Fast);
	drain_and_apply(&mut worker);
	set(&mut worker, &objects, X, LEN, Fast);
	drain_and_apply(&mut worker);
	tiny(&mut worker);

	worker.handle_get(K, Some(Slow));
	drain_and_apply(&mut worker);

	// `evict_one_key`, keeping the drain of the pass.
	let used = worker.status.used_size(&worker.status.policy());
	worker.status.set_max_size(used - 1);
	worker.apply_evictions(&mut Vec::new()).expect("an eviction pass");
	worker.status.set_max_size(1 << 30);
	let drain = drain_and_apply(&mut worker);

	assert_eq!(placement(&worker, X), None, "the unreferenced key was the victim");
	assert_eq!(of(&drain, K), vec![Slow], "clock second chance: the settle's entry alone");
	assert_eq!(bytes_tier(&objects, K), Slow);
	assert_settled(&mut worker);
	charges_exact(&objects, true);
}

/// U12, in both stores: the LFU latch is published with the `Set` that shuts
/// it -- right after the worker handles it, with no pass in between. It used
/// to be published once per pass (`refresh_tier_gauges`), and the merged
/// store did not publish it at all. Red with the per-event publication off
/// (`latchperpass`).
#[test]
fn u12_the_latch_is_published_with_the_set_that_shuts_it() {
	let _serialised = migration_test_lock::lock();

	let (mut worker, objects) = make_worker(PaperPolicy::LfuCompactHybrid);

	let mut latched_at = None;

	// New keys, each built fast and handled -- and NO `refresh_tier_gauges`,
	// which `fill` would call before every set.
	for key in 1..=32 {
		let published = publish(&worker.status, &worker.overhead_manager, &objects, key, LEN, Fast);
		handle(&mut worker, key, published);

		let latched = stack(&worker).admission_latched();

		assert_eq!(
			worker.status.hybrid_admission_latched(),
			latched,
			"after the Set of key {key}, with no pass since, the published latch is not the stack's",
		);

		drain_and_apply(&mut worker);

		if latched {
			latched_at = Some(key);
			break;
		}
	}

	assert!(latched_at.is_some(), "twice the fast tier never latched the stack");
	assert_settled(&mut worker);
}

/// U12's point, in both stores: once the worker has handled the `Set` that
/// latched, the client builds the next new LFU key slow, where the stack
/// places it, and its `Set` queues nothing at all. The merged store's client
/// built every new LFU key fast (its handle never published the latch) and
/// queued a corrective per admission. Red with the merged latch unpublished
/// (`nolatchpub`) and with the per-event publication off (`latchperpass`).
#[test]
fn u12_a_new_key_set_after_the_latch_is_built_slow_and_needs_no_corrective() {
	let _serialised = migration_test_lock::lock();

	let (mut worker, objects) = make_worker(PaperPolicy::LfuCompactHybrid);

	for key in 1..=32 {
		let published = publish(&worker.status, &worker.overhead_manager, &objects, key, LEN, Fast);
		handle(&mut worker, key, published);
		drain_and_apply(&mut worker);
	}

	assert!(stack(&worker).admission_latched(), "twice the fast tier latched the stack");

	const K: HashedKey = 100;

	// What `PaperCache::set` builds it in now -- no pass has run.
	let built = admission_tier(PaperPolicy::LfuCompactHybrid, K, &worker.status, &objects);
	assert_eq!(built, Slow, "a new key after the latching Set is built slow");

	set(&mut worker, &objects, K, LEN, built);
	assert_eq!(placement(&worker, K), Some(Slow));

	let drain = drain_and_apply(&mut worker);
	assert_eq!(of(&drain, K), vec![], "built where it is placed: nothing queued for it");
	assert_settled(&mut worker);
}

/// U15, in both stores: LFU counts the demotions its SETTLE decides
/// (`drain_demotions`), not every slow landing -- an admission refused to the
/// slow tier queues `(k, Slow)` for a value the client built fast, and that
/// landing displaces nothing. The merged store counted its landings until
/// S4 (`inline_demotion_accounting`); green there before, because its
/// refusal was a client-side corrective, and red with the landings counted
/// (`inline`) now that the refusal is the store's own entry.
#[test]
fn u15_lfu_counts_settle_demotions_not_a_refused_admission() {
	let _serialised = migration_test_lock::lock();

	let (mut worker, objects) = make_worker(PaperPolicy::LfuCompactHybrid);

	fill(&mut worker, &objects, 1..=4, LEN);

	let before = worker.status.hybrid_stats().demotions;
	worker.drained = Some(Vec::new());

	// Too big for what is left of the tier: refused to slow, queued, latched.
	const K: HashedKey = 100;
	set(&mut worker, &objects, K, 15 * 1024, Fast);
	worker.apply_tier_migrations();
	assert_eq!((placement(&worker, K), bytes_tier(&objects, K)), (Some(Slow), Slow), "refused to slow");

	// A shrink whose settle demotes every fast key.
	worker.handle_resize_fast_tier(1);
	worker.apply_tier_migrations();

	let drained = worker.drained.take().expect("recording");
	let settled = drained.iter().filter(|(k, t, o)| *k != K && *t == Slow && *o == MigrationOrigin::Stack).count() as u64;

	assert!(settled > 0, "the shrink demoted nothing");
	assert_eq!(
		worker.status.hybrid_stats().demotions - before,
		settled,
		"demotions are the settle's decisions, not the refused admission's landing",
	);
	assert_settled(&mut worker);
}

/// U9, in both stores: a CLOCK hand that has used up its budget of second
/// chances evicts the tail whatever its bit -- `clock_hand_budget`, forced
/// to 0 here, which no real sequence reaches. The DashMap stack's hand had no
/// cap (red there without it: `nohandcap`); the merged store's had one.
#[test]
fn u9_the_clock_hand_stops_at_its_budget() {
	let _serialised = migration_test_lock::lock();

	const K: HashedKey = 1;
	const X: HashedKey = 2;

	for budget in [None, Some(0)] {
		let (mut worker, objects) = make_worker(PaperPolicy::ClockCompactHybrid);

		set(&mut worker, &objects, K, LEN, Fast);
		drain_and_apply(&mut worker);
		set(&mut worker, &objects, X, LEN, Fast);
		drain_and_apply(&mut worker);

		// K, the older, referenced.
		worker.handle_get(K, Some(Fast));
		drain_and_apply(&mut worker);

		match budget {
			None => evict_one_key(&mut worker),
			Some(b) => super::policy_stack::hand_budget_override::with(b, || evict_one_key(&mut worker)),
		}

		let victim = match budget {
			None => X,
			Some(_) => K,
		};

		assert_eq!(placement(&worker, victim), None, "budget {budget:?}: the victim");
		assert!(objects.get_ref(&victim).is_none(), "budget {budget:?}: the victim left the map");
		assert!(placement(&worker, K ^ X ^ victim).is_some(), "budget {budget:?}: the other key stayed");
		assert_settled(&mut worker);
	}
}

/// Waits for `done`, up to `deadline`.
fn wait_for(what: &str, deadline: Duration, mut done: impl FnMut() -> bool) {
	let start = Instant::now();

	while !done() {
		assert!(start.elapsed() < deadline, "{what}");
		std::thread::sleep(Duration::from_millis(1));
	}
}

/// U7, in both stores: `wipe` is the policy worker's -- the map, the stack,
/// the status and the gauges -- and returns when it is done. So at once after
/// it the tier gauges read an empty cache and the audit finds nothing live;
/// and a set and a get right after work. The gauges used to wait for the
/// worker's next pass. Red with the wait removed (`asyncwipe`).
#[test]
fn u7_wipe_returns_after_the_worker_cleared_everything() {
	let _serialised = migration_test_lock::lock();

	for policy in [PaperPolicy::LruCompactHybrid, PaperPolicy::LfuCompactHybrid] {
		let cache = PaperCache::<u64, TieredBuffer>::new(1 << 20, CacheTierSize::Bytes(FAST), policy)
			.expect("a hybrid cache");

		for key in 0..40u64 {
			cache.set(key, &[key as u8; LEN], None).expect("set");
		}

		wait_for("the worker never took the sets", Duration::from_secs(10), || {
			let s = cache.hybrid_stats();
			s.fast_objects + s.slow_objects == 40
		});

		cache.wipe().expect("wipe");

		let s = cache.hybrid_stats();

		assert_eq!(
			(s.fast_objects, s.slow_objects, s.fast_bytes_used, s.slow_bytes_used, s.fast_metadata_bytes),
			(0, 0, 0, 0, 0),
			"{policy}: the gauges right after wipe()",
		);
		assert_eq!(cache.status().expect("status").used_size(), 0, "{policy}: used size after wipe()");
		assert_eq!(cache.placement_audit().expect("an audit").live, 0, "{policy}: live values after wipe()");

		cache.set(7, &[7; LEN], None).expect("a set after the wipe");
		assert_eq!(cache.get(&7).expect("a get after the wipe"), vec![7; LEN]);
	}
}

/// U7, in both stores: the wipe of a cache whose worker is parked on its
/// idle poll (a new cache, no set yet: up to 1 s) returns promptly -- the
/// wait for the worker's answer kicks it -- and so does the audit. Red
/// without the kicks (`nokick`). The bound is 600 ms: the worker has been
/// parked for 100-200 ms when the call is made, so an unkicked call waits
/// 800 ms or more, and a loaded machine gets room it did not have at 200.
#[test]
fn u7_wipe_and_audit_of_an_idle_cache_do_not_wait_out_the_idle_poll() {
	let _serialised = migration_test_lock::lock();

	let cache = PaperCache::<u64, TieredBuffer>::new(1 << 20, CacheTierSize::Bytes(FAST), PaperPolicy::LruCompactHybrid)
		.expect("a hybrid cache");

	// Parked: a pass has run, and none follows for 100 ms.
	let parked = |cache: &PaperCache<u64, TieredBuffer>| {
		wait_for("the worker's first pass", Duration::from_secs(10), || cache.status.policy_worker_passes() > 0);

		loop {
			let passes = cache.status.policy_worker_passes();
			std::thread::sleep(Duration::from_millis(100));

			if cache.status.policy_worker_passes() == passes {
				return;
			}
		}
	};

	parked(&cache);
	let start = Instant::now();
	cache.wipe().expect("wipe");
	let wipe = start.elapsed();

	parked(&cache);
	let start = Instant::now();
	cache.placement_audit().expect("an audit");
	let audit = start.elapsed();

	assert!(wipe < Duration::from_millis(600), "wipe() of an idle cache took {wipe:?}: it waited out the worker's idle poll");
	assert!(audit < Duration::from_millis(600), "placement_audit() of an idle cache took {audit:?}: it waited out the worker's idle poll");
}

/// U7, in both stores: a `Set` the worker handled before the `Wipe` cannot be
/// left live and untracked -- the worker clears the map with the stack. When
/// the client cleared the map and the worker the stack, a second client's
/// value published between the client's clear and the worker's handling of
/// its `Wipe` -- its `Set` handled first -- stayed in the map, which the
/// stack no longer tracked. Red with the worker clearing the stack alone
/// (`wipestackonly`).
#[test]
fn u7_a_set_handled_before_the_wipe_is_not_left_untracked() {
	let _serialised = migration_test_lock::lock();

	for policy in TIERED {
		let (mut worker, objects) = make_worker(policy);

		const K: HashedKey = 1;
		set(&mut worker, &objects, K, LEN, Fast);
		drain_and_apply(&mut worker);
		assert!(placement(&worker, K).is_some());

		worker.handle_wipe(None);

		assert!(objects.get_ref(&K).is_none(), "{policy}: the map still holds the key the stack dropped");
		assert_eq!(placement(&worker, K), None, "{policy}: the stack still places the key");
		assert_eq!(stack(&worker).len(), 0);
		assert_eq!(worker.status.used_size(&policy), 0, "{policy}: the status still counts it");
		charges_exact(&objects, true);
	}
}

/// U7's fallback, in both stores: with the policy worker gone -- ended here by
/// a `Shutdown` and joined, as a worker that died would be -- `wipe()` does not
/// wait for an answer that cannot come: the event is dropped with the worker's
/// channel, and with it the answer's sender, so the wait errs at once; it then
/// clears the map and the status itself, and reports the failure. Red with the
/// fallback clearing nothing (`nofallbackclear`).
#[test]
fn wipe_clears_the_cache_itself_when_the_policy_worker_is_gone() {
	let _serialised = migration_test_lock::lock();

	let policy = PaperPolicy::LruCompactHybrid;
	let mut cache = PaperCache::<u64, TieredBuffer>::new(1 << 20, CacheTierSize::Bytes(FAST), policy)
		.expect("a hybrid cache");

	for key in 0..40u64 {
		cache.set(key, &[key as u8; LEN], None).expect("set");
	}

	wait_for("the worker never took the sets", Duration::from_secs(10), || {
		let s = cache.hybrid_stats();
		s.fast_objects + s.slow_objects == 40
	});

	// Every worker ends, the policy worker among them, and is joined: its
	// channel is closed.
	cache.workers.send(WorkerEvent::Shutdown).expect("the shutdown");

	for handle in cache.worker_handles.drain(..) {
		let _ = handle.join();
	}

	assert_eq!((cache.objects.len(), cache.status.live_num_objects()), (40, 40), "nothing cleared yet");

	let start = Instant::now();
	let wiped = cache.wipe();

	assert!(matches!(wiped, Err(CacheError::Internal)), "the wipe reports the dead worker: {wiped:?}");
	assert!(start.elapsed() < Duration::from_secs(5), "the wipe waited {:?} for a dead worker", start.elapsed());
	assert_eq!(cache.objects.len(), 0, "the fallback left objects in the map");
	assert_eq!(
		(cache.status.live_num_objects(), cache.status.used_size(&policy)),
		(0, 0),
		"the fallback left the status counting them",
	);
}

/// S4's wipe/status race, in both stores: client threads setting and
/// deleting -- each over its own keys, so no two threads race on a key --
/// while the cache is wiped over and over for 400 ms and its fast tier
/// resized; then, quiet, the status counts exactly what the map holds: the
/// object count is the map's, `used_size` its base bytes plus the per-object
/// overhead; the merged store's charges are exact and every value linked
/// (`verify_charges`); the audit is clean. The worker's wipe STORED 0 into the
/// status after its clear, so a set landing in a shard the clear had emptied
/// -- its status update before the store -- stayed live and uncounted (its
/// removal then wrapped `base_used_size`), and one the clear removed -- its
/// update after the store -- stayed counted. Now the wipe subtracts what the
/// clear removed. Red with the store (`wipestorezero`).
///
/// The clients pause 100 us every 16 operations. Unpaced they outrun the
/// policy worker, the event channel grows without bound (gigabytes), and every
/// wipe waits for the whole backlog ahead of it.
#[test]
fn a_wipe_racing_client_sets_and_deletes_leaves_the_status_exact() {
	let _serialised = migration_test_lock::lock();

	const CLIENTS: u64 = 4;
	const KEYS_EACH: u64 = 512;
	const RACE: Duration = Duration::from_millis(400);

	for policy in TIERED {
		let cache = std::sync::Arc::new(
			PaperCache::<u64, TieredBuffer>::new(1 << 20, CacheTierSize::Bytes(256 * 1024), policy)
				.expect("a hybrid cache"),
		);

		let stop = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));

		let clients: Vec<_> = (0..CLIENTS)
			.map(|t| {
				let (cache, stop) = (cache.clone(), stop.clone());

				std::thread::spawn(move || {
					let mut x = 0x2545_F491_4F6C_DD1Du64 ^ (t + 1).wrapping_mul(0x9E37_79B9_7F4A_7C15);
					let mut ops = 0u64;

					while !stop.load(std::sync::atomic::Ordering::Relaxed) {
						x ^= x << 13;
						x ^= x >> 7;
						x ^= x << 17;

						let key = t * KEYS_EACH + x % KEYS_EACH;
						let len = 64 + (x >> 20) as usize % 960;

						match (x >> 40) % 8 {
							0..=5 => { let _ = cache.set(key, &vec![key as u8; len], None); },
							_ => { let _ = cache.del(&key); },
						}

						ops += 1;

						if ops % 16 == 0 {
							std::thread::sleep(Duration::from_micros(100));
						}
					}

					ops
				})
			})
			.collect();

		let start = Instant::now();
		let mut wipes = 0usize;

		while start.elapsed() < RACE {
			cache.wipe().expect("wipe");
			wipes += 1;

			if wipes % 10 == 0 {
				let size = if wipes % 20 == 0 { 128 * 1024 } else { 256 * 1024 };
				cache.set_fast_tier_size(CacheTierSize::Bytes(size)).expect("a resize");
			}
		}

		stop.store(true, std::sync::atomic::Ordering::Relaxed);

		let ops: u64 = clients.into_iter().map(|client| client.join().expect("a client thread panicked")).sum();

		// Quiet: the audit is handled after every event before it; two passes
		// more and the worker's last eviction pass is done too.
		cache.placement_audit().expect("an audit");
		let passes = cache.status.policy_worker_passes();
		wait_for("two more passes", Duration::from_secs(10), || cache.status.policy_worker_passes() >= passes + 2);

		let objects = &cache.objects;
		let (mut live, mut base) = (0u64, 0 as CacheSize);

		for key in 0..CLIENTS * KEYS_EACH {
			if let Some(object) = objects.get_ref(&cache.hash_key(&key)) {
				live += 1;
				base += cache.overhead_manager.base_size(&object) as CacheSize;
			}
		}

		let what = format!("{policy}: {wipes} wipes racing {ops} client ops");

		assert!(wipes >= 20, "{what}: too few wipes to race anything");

		assert_eq!(objects.len() as u64, live, "{what}: the map's count is its keys'");
		assert_eq!(cache.status.live_num_objects(), live, "{what}: the status's object count is the map's");
		assert_eq!(
			cache.status.used_size(&policy),
			base + live * crate::object::overhead::get_policy_overhead(&policy) as CacheSize,
			"{what}: used_size is the map's base bytes plus the per-object overhead",
		);

		#[cfg(feature = "merged_object_store")]
		cache.objects.verify_charges(true);

		let audit = cache.placement_audit().expect("an audit");
		assert!(audit.is_clean(), "{what}: {audit:?}");
	}
}

/// Race R1, in both stores: a value a client published and whose `Set` the
/// worker has not taken is in the map and not placed -- the audit reports it
/// untracked, and the stack neither counts nor places it -- until the `Set`
/// is handled, after which the audit is clean. In the merged store it used
/// to be linked, charged and placed by the client's insert. Red with an
/// unlinked slot answering its tier (`unlinkedplaced`).
#[test]
fn r1_a_value_published_and_never_set_is_unlinked_and_untracked() {
	let _serialised = migration_test_lock::lock();

	for policy in TIERED {
		let (mut worker, objects) = make_worker(policy);

		const K: HashedKey = 1;
		let published = publish(&worker.status, &worker.overhead_manager, &objects, K, LEN, Fast);

		let audit = worker.placement_audit();
		assert_eq!((audit.live, audit.untracked), (1, 1), "{policy}: {audit:?}");
		assert_eq!(placement(&worker, K), None, "{policy}: placed before its Set");
		assert_eq!(stack(&worker).len(), 0, "{policy}: counted before its Set");

		handle(&mut worker, K, published);
		drain_and_apply(&mut worker);

		assert_eq!(placement(&worker, K), Some(Fast));
		assert_settled(&mut worker);
		charges_exact(&objects, true);
	}
}

/// Race R8, in both stores: a hit the worker handles before the key's `Set`
/// (two client threads can order them so) moves nothing -- the DashMap
/// stacks' `update` of a key they do not track is a no-op, and the merged
/// store's touch, reference bit and bump skip an unlinked slot. Then the
/// `Set` links it as a new key. Red with an unlinked slot touched
/// (`touchunlinked`).
#[test]
fn r8_a_get_handled_before_its_keys_set_moves_nothing() {
	let _serialised = migration_test_lock::lock();

	for policy in TIERED {
		let (mut worker, objects) = make_worker(policy);

		fill(&mut worker, &objects, 1..=4, LEN);

		const K: HashedKey = 100;
		let published = publish(&worker.status, &worker.overhead_manager, &objects, K, LEN, Fast);

		let before = gauges(&worker);
		worker.handle_get(K, Some(Fast));
		let drain = drain_and_apply(&mut worker);

		assert_eq!(gauges(&worker), before, "{policy}: the early hit moved the stack");
		assert!(drain.is_empty(), "{policy}: the early hit queued {drain:?}");
		charges_exact(&objects, false);

		handle(&mut worker, K, published);
		drain_and_apply(&mut worker);

		assert_eq!(placement(&worker, K), Some(Fast), "{policy}: admitted as a new key");
		assert_settled(&mut worker);
		charges_exact(&objects, true);
	}
}

/// Race R5, in both stores: `set v1; del; set v2` of one key all published
/// before the worker takes any of them. The DashMap stack inserts at `Set(v1)`,
/// removes at `Del` and inserts afresh at `Set(v2)`: a NEW key -- frequency 1,
/// reference bit clear, the newest position. The merged store linked the
/// only slot left (v2's) at `Set(v1)`; `Set(v2)` then finds it linked and
/// RE-ADMITS it, and ends in the same state. Checked through the eviction
/// order: under LFU the key is the victim ahead of a key hit once before it;
/// under CLOCK its bit is clear when the hand reaches it; and under LRU and
/// FIFO, with another key's set published in the middle (`set k; set x; del
/// k; set k`), it is newer than that key. Red with a fresh `Set` on a linked
/// slot doing nothing (`freshnoop`) or taken as an access (`freshaccess`).
#[test]
fn r5_a_del_and_reset_before_the_worker_counts_one_admission() {
	let _serialised = migration_test_lock::lock();

	const K: HashedKey = 1;
	const A: HashedKey = 2;
	const X: HashedKey = 3;

	let reset_before_the_worker = |worker: &mut Worker, objects: &Objects, between: Option<HashedKey>| {
		let v1 = publish(&worker.status, &worker.overhead_manager, objects, K, LEN, Fast);
		let x = between.map(|x| (x, publish(&worker.status, &worker.overhead_manager, objects, x, LEN, Fast)));
		publish_del(&worker.status, &worker.overhead_manager, objects, K);
		let v2 = publish(&worker.status, &worker.overhead_manager, objects, K, LEN + 300, Fast);

		handle(worker, K, v1);
		drain_and_apply(worker);

		if let Some((x, published)) = x {
			handle(worker, x, published);
			drain_and_apply(worker);
		}

		worker.handle_del(K);
		drain_and_apply(worker);
		handle(worker, K, v2);
		drain_and_apply(worker);
	};

	// LFU: A, hit once, has frequency 2; K, re-admitted, has 1 -- the victim.
	let (mut worker, objects) = make_worker(PaperPolicy::LfuCompactHybrid);
	set(&mut worker, &objects, A, LEN, Fast);
	drain_and_apply(&mut worker);
	worker.handle_get(A, Some(Fast));
	drain_and_apply(&mut worker);
	reset_before_the_worker(&mut worker, &objects, None);
	evict_one_key(&mut worker);
	assert_eq!((placement(&worker, K), placement(&worker, A).is_some()), (None, true), "LFU: the re-admitted key is at frequency 1");
	assert_settled(&mut worker);
	charges_exact(&objects, true);

	// CLOCK: Y oldest, then K; evict Y; set Z; K is then the tail, and its
	// bit is clear, so it goes before Z.
	let (mut worker, objects) = make_worker(PaperPolicy::ClockCompactHybrid);
	const Y: HashedKey = 4;
	const Z: HashedKey = 5;
	set(&mut worker, &objects, Y, LEN, Fast);
	drain_and_apply(&mut worker);
	reset_before_the_worker(&mut worker, &objects, None);
	evict_one_key(&mut worker);
	assert_eq!(placement(&worker, Y), None, "CLOCK: the oldest key goes first");
	set(&mut worker, &objects, Z, LEN, Fast);
	drain_and_apply(&mut worker);
	evict_one_key(&mut worker);
	assert_eq!((placement(&worker, K), placement(&worker, Z).is_some()), (None, true), "CLOCK: the re-admitted key's bit is clear");
	assert_settled(&mut worker);
	charges_exact(&objects, true);

	// LRU and FIFO, with X published between the set and the re-set: K is
	// newer than X, so X goes first.
	for policy in [PaperPolicy::LruCompactHybrid, PaperPolicy::FifoCompactHybrid] {
		let (mut worker, objects) = make_worker(policy);
		reset_before_the_worker(&mut worker, &objects, Some(X));
		evict_one_key(&mut worker);
		assert_eq!((placement(&worker, X), placement(&worker, K).is_some()), (None, true), "{policy}: the re-set key is the newer");
		assert_settled(&mut worker);
		charges_exact(&objects, true);
	}
}

/// Race R12, in both stores: a value published before the worker's wipe,
/// whose `Set` is behind the `Wipe` in the channel. The worker clears it with
/// the map, and its `Set` then finds nothing in the merged store: nothing
/// linked, nothing charged. (A DashMap stack inserts it anyway -- a ghost
/// entry over an empty map, evicted later as `KeyNotFound`; the map is empty
/// in both.)
#[test]
fn r12_a_wipe_during_a_pending_link_leaves_nothing_linked() {
	let _serialised = migration_test_lock::lock();

	for policy in TIERED {
		let (mut worker, objects) = make_worker(policy);

		const K: HashedKey = 1;
		let published = publish(&worker.status, &worker.overhead_manager, &objects, K, LEN, Fast);

		worker.handle_wipe(None);
		handle(&mut worker, K, published);
		drain_and_apply(&mut worker);

		assert!(objects.get_ref(&K).is_none(), "{policy}: the wipe left the value in the map");

		#[cfg(feature = "merged_object_store")]
		{
			assert_eq!(placement(&worker, K), None, "{policy}: the Set after the wipe linked something");
			assert_eq!((objects.linked(), objects.len()), (0, 0), "{policy}");
			charges_exact(&objects, true);
		}
	}
}

// ---- The merged store's own races (section 6), and its oracle. ------------

/// A key in merged shard `shard` (`merged_store::shard_of` reads the top five
/// bits), distinct for distinct `i`.
#[cfg(feature = "merged_object_store")]
fn in_shard(shard: u64, i: u64) -> HashedKey {
	(i.wrapping_mul(0x9E37_79B9_7F4A_7C15) >> 5) | (shard << 59)
}

/// Race R2: an overwrite before the link. v1 is published, then v2 over it,
/// both before the worker takes `Set(v1)`. The client records no unfolded
/// bytes for an unlinked slot, and the link charges the slot's CURRENT object
/// -- v2 -- once; `Set(v2)` is then an ordinary overwrite. Red with unfolded
/// bytes recorded for an unlinked slot too (`unfoldunlinked`): charged twice;
/// and with the link charging the event's size (`linkeventsize`): v1's.
#[cfg(feature = "merged_object_store")]
#[test]
fn r2_an_overwrite_before_the_link_charges_the_newest_value_once() {
	let _serialised = migration_test_lock::lock();

	let (mut worker, objects) = make_worker(PaperPolicy::LruCompactHybrid);

	const K: HashedKey = 1;
	let v1 = publish(&worker.status, &worker.overhead_manager, &objects, K, LEN, Fast);
	let v2 = publish(&worker.status, &worker.overhead_manager, &objects, K, 3 * LEN, Fast);

	handle(&mut worker, K, v1);
	assert_eq!(objects.fast_bytes_used(), charge(3 * LEN), "the link charged the newest value");
	charges_exact(&objects, false);

	handle(&mut worker, K, v2);
	drain_and_apply(&mut worker);

	assert_eq!(objects.fast_bytes_used(), charge(3 * LEN), "charged once");
	assert_settled(&mut worker);
	charges_exact(&objects, true);
}

/// Race R3: two clients' `Set`s of one key handled out of order -- the
/// overwrite's first (it links the slot), the fresh one second (it finds the
/// slot linked and re-admits it). Linked once, charged the live value.
#[cfg(feature = "merged_object_store")]
#[test]
fn r3_sets_handled_out_of_order_link_once_and_charge_the_live_value() {
	let _serialised = migration_test_lock::lock();

	let (mut worker, objects) = make_worker(PaperPolicy::LruCompactHybrid);

	const K: HashedKey = 1;
	let v1 = publish(&worker.status, &worker.overhead_manager, &objects, K, LEN, Fast);
	let v2 = publish(&worker.status, &worker.overhead_manager, &objects, K, 2 * LEN, Fast);
	assert!(v1.fresh() && !v2.fresh());

	handle(&mut worker, K, v2);
	drain_and_apply(&mut worker);
	handle(&mut worker, K, v1);
	drain_and_apply(&mut worker);

	assert_eq!((objects.linked(), objects.fast_bytes_used()), (1, charge(2 * LEN)));
	assert_settled(&mut worker);
	charges_exact(&objects, true);
}

/// Race R3 through the new-key rule's fence: a `Replaced` `Set` the worker
/// handles while its slot is still UNLINKED -- the overwrite's `Set` taken
/// before the fresh one's -- is fenced like a new key when its bucket moved
/// since its mark, for the store does not place the key before the event
/// (`placement_of` None) although the insert replaced a value. K's old value
/// is demoted by another key's admission, and the worker takes that drain;
/// before it lands, the clients delete K and set v1 (fresh) and v2 over it.
/// The demotion then lands on v2, the value the key holds, after both marks.
/// The worker takes the `Del` (the DEAD slot) and then Set(v2): it links the
/// slot fast, where v2 was built, and queues nothing; fenced, the rule queues
/// `(K, Fast)` last and v2 ends where it is placed. Then Set(v1), fresh on the
/// linked slot: a re-admission, fenced too, whose corrective declines. Red
/// with the fence's `placement_of` clause removed (`nonfresh`, S3's): v2 is
/// not fresh, is not fenced, and stays in CXL, placed fast.
#[cfg(feature = "merged_object_store")]
#[test]
fn r3_a_replaced_set_on_an_unlinked_slot_is_fenced_like_a_new_key() {
	let _serialised = migration_test_lock::lock();

	let (mut worker, objects) = make_worker(PaperPolicy::LruCompactHybrid);

	const K: HashedKey = 1;
	// Twice the fast tier: its admission demotes K and itself.
	const X: HashedKey = 7_777;

	set(&mut worker, &objects, K, LEN, Fast);
	drain_and_apply(&mut worker);
	assert_eq!((placement(&worker, K), bytes_tier(&objects, K)), (Some(Fast), Fast));

	// The worker takes the admission's drain, and has not applied it yet.
	set(&mut worker, &objects, X, 2 * FAST as usize, Fast);
	let (inline, held) = worker.drain_reconciled().expect("a stack");
	assert_eq!(of(&held, K), vec![Slow], "the admission demoted K");

	// The clients: `del(K)`, then `set(K, v1)` -- new -- and `set(K, v2)`
	// over it, both built in DRAM as LRU admits.
	let status = worker.status.clone();
	let overhead_manager = worker.overhead_manager.clone();

	publish_del(&status, &overhead_manager, &objects, K);
	let v1 = publish(&status, &overhead_manager, &objects, K, LEN, Fast);
	let v2 = publish(&status, &overhead_manager, &objects, K, LEN + 100, Fast);
	assert!(v1.fresh() && !v2.fresh(), "v1 was new to the map, v2 replaced it");

	// The drain lands: K's demotion moves v2, after both marks.
	worker.apply_migration_batches(held, inline);
	assert_eq!(bytes_tier(&objects, K), Slow, "the stale demotion moved v2");

	// The worker: the `Del`, then Set(v2) on the unlinked slot.
	worker.handle_del(K);
	drain_and_apply(&mut worker);
	assert_eq!(placement(&worker, K), None, "v2 is unlinked: the store does not place K");

	handle(&mut worker, K, v2);
	assert_eq!(placement(&worker, K), Some(Fast), "linked fast, where v2 was built, with no push");
	let observed = worker.observed.clone();

	let drain = drain_and_apply(&mut worker);
	assert_eq!(
		observed,
		vec![Observed::Built { key: K, built: Fast, fence: true }],
		"fenced: v2 replaced a value, but the store did not place K before its Set",
	);
	assert_eq!(correctives(&drain, K), vec![Fast], "the new-key rule's corrective (built where placed)");
	assert_eq!(bytes_tier(&objects, K), Fast, "v2 is where the store placed it");

	// Set(v1), fresh on the linked slot: re-admitted; its corrective declines.
	handle(&mut worker, K, v1);
	drain_and_apply(&mut worker);

	assert_eq!((placement(&worker, K), bytes_tier(&objects, K)), (Some(Fast), Fast));
	assert_eq!(objects.get_ref(&K).map(|o| o.data_size() as usize), Some(LEN + 100), "v2 is the value");
	assert_settled(&mut worker);
	charges_exact(&objects, true);
}

/// Race R4: a value deleted -- or reaped -- before the worker links it is
/// freed by the client at once (nothing was charged or linked); its `Set`
/// then finds nothing, and its `Del` (`Expire`) no DEAD slot.
#[cfg(feature = "merged_object_store")]
#[test]
fn r4_a_slot_deleted_or_reaped_before_its_link_is_never_linked_or_charged() {
	let _serialised = migration_test_lock::lock();

	for reaped in [false, true] {
		let (mut worker, objects) = make_worker(PaperPolicy::LruCompactHybrid);

		const K: HashedKey = 1;

		let published = match reaped {
			false => {
				let p = publish(&worker.status, &worker.overhead_manager, &objects, K, LEN, Fast);
				publish_del(&worker.status, &worker.overhead_manager, &objects, K);
				p
			},

			true => {
				let p = publish_expired(&worker, &objects, K, LEN);
				reap(&worker, &objects, K);
				p
			},
		};

		assert!(objects.get_ref(&K).is_none());

		handle(&mut worker, K, published);

		match reaped {
			false => worker.handle_del(K),
			true => worker.handle_expire(K),
		}

		drain_and_apply(&mut worker);

		assert_eq!((objects.linked(), objects.len(), objects.fast_bytes_used()), (0, 0, 0), "reaped {reaped}");
		assert_settled(&mut worker);
		charges_exact(&objects, true);
	}
}

/// Race R6: an overwritten slot moved before its `Set`. K -- the oldest fast
/// key, in one shard -- is overwritten and grown by a client; before the
/// worker takes that `Set`, another key's admission, in ANOTHER shard,
/// settles and demotes K. The settle's step in K's shard folds the client's
/// bytes first, so the demotion moves K's CURRENT size and both tiers stay
/// exact; then K's own `Set` promotes it. Red with no fold (`nofold`) and with
/// the fold only in `worker_set`'s own section (`foldonlyinset`).
#[cfg(feature = "merged_object_store")]
#[test]
fn r6_an_overwritten_slot_moved_before_its_set_keeps_the_totals_exact() {
	let _serialised = migration_test_lock::lock();

	let (mut worker, objects) = make_worker(PaperPolicy::LruCompactHybrid);

	let k = in_shard(3, 1);
	set(&mut worker, &objects, k, LEN, Fast);
	drain_and_apply(&mut worker);

	for i in 2..=10 {
		set(&mut worker, &objects, in_shard(7, i), LEN, Fast);
		drain_and_apply(&mut worker);
	}

	assert_eq!(placement(&worker, k), Some(Fast), "k fast, and the oldest");

	let grown = publish(&worker.status, &worker.overhead_manager, &objects, k, 4 * LEN, Fast);

	// Another shard's admission, big enough to demote k.
	let x = in_shard(9, 11);
	set(&mut worker, &objects, x, 6 * LEN, Fast);
	let drain = drain_and_apply(&mut worker);

	assert_eq!(of(&drain, k), vec![Slow], "the admission's settle demoted k");
	charges_exact(&objects, false);

	handle(&mut worker, k, grown);
	drain_and_apply(&mut worker);

	assert_settled(&mut worker);
	charges_exact(&objects, true);
}

/// Race R7: a value the worker has not linked is never evicted -- `erase`'s
/// eviction arm (`take_evict`) refuses it, `KeyNotFound`, and an eviction
/// pass over nothing linked stops rather than spins; once linked, it goes.
/// Red with the refusal off (`evictunlinked`).
#[cfg(feature = "merged_object_store")]
#[test]
fn r7_eviction_never_takes_an_unlinked_value() {
	let _serialised = migration_test_lock::lock();

	let (mut worker, objects) = make_worker(PaperPolicy::LruCompactHybrid);

	const U: HashedKey = 1;
	let published = publish(&worker.status, &worker.overhead_manager, &objects, U, LEN, Fast);

	assert!(
		matches!(
			erase(&objects, &worker.status, &worker.overhead_manager, Some(EraseKey::Hashed(U))),
			Err(CacheError::KeyNotFound),
		),
		"an unlinked value was evicted",
	);
	assert!(objects.get_ref(&U).is_some());

	// Over the cache's size with nothing linked: the pass stops.
	let used = worker.status.used_size(&worker.status.policy());
	worker.status.set_max_size(used - 1);
	worker.apply_evictions(&mut Vec::new()).expect("an eviction pass");
	assert!(objects.get_ref(&U).is_some(), "the pass took an unlinked value");

	handle(&mut worker, U, published);
	worker.apply_evictions(&mut Vec::new()).expect("an eviction pass");
	worker.status.set_max_size(1 << 30);
	drain_and_apply(&mut worker);

	assert!(objects.get_ref(&U).is_none(), "linked, it is evicted");
	assert_settled(&mut worker);
	charges_exact(&objects, true);
}

/// The eviction pass over a merged store with nothing linked stops (R7) --
/// silently while published, unlinked values remain, whose `Set`s are behind
/// this pass (it never evicts them) -- and with an error when the store is
/// REALLY empty: `used_size` over the cache's size with nothing in the map is
/// an accounting bug, not a backlog. Red with the error dropped
/// (`silentempty`, S4's break) and raised for unlinked values too
/// (`loudunlinked`, the break before S4).
#[cfg(feature = "merged_object_store")]
#[test]
fn an_eviction_pass_with_nothing_linked_errs_only_over_an_empty_store() {
	let _serialised = migration_test_lock::lock();

	let (mut worker, objects) = make_worker(PaperPolicy::LruCompactHybrid);

	const U: HashedKey = 1;
	publish(&worker.status, &worker.overhead_manager, &objects, U, LEN, Fast);

	// Only an unlinked value, over the size: the pass stops, silently.
	let used = worker.status.used_size(&worker.status.policy());
	worker.status.set_max_size(used - 1);
	worker.apply_evictions(&mut Vec::new()).expect("an eviction pass");

	assert!(objects.get_ref(&U).is_some(), "the pass took an unlinked value");
	assert_eq!(worker.nothing_left_to_evict, 0, "an unlinked value is a backlog, not an error");

	// Deleted before its link, it is freed at once: the store is empty. The
	// status says a MiB is held: the pass stops, and errs.
	publish_del(&worker.status, &worker.overhead_manager, &objects, U);
	assert_eq!(objects.len(), 0);

	worker.status.update_base_used_size(1 << 20);
	worker.status.set_max_size(1 << 19);
	worker.apply_evictions(&mut Vec::new()).expect("an eviction pass");

	assert_eq!(worker.nothing_left_to_evict, 1, "a store with nothing in it and used_size over the size is an error");

	worker.status.update_base_used_size(-(1i64 << 20));
	worker.status.set_max_size(1 << 30);
	worker.handle_del(U);
	drain_and_apply(&mut worker);
	assert_settled(&mut worker);
	charges_exact(&objects, true);
}

/// Race R9: a DEAD slot at the tail -- deleted by a client, its `Del` not yet
/// handled -- is not nominated: the nominator retires it in place and names
/// the next key. The `Del` then finds nothing to retire. Red with the DEAD
/// tail nominated (`nominatedead`).
#[cfg(feature = "merged_object_store")]
#[test]
fn r9_a_dead_slot_at_the_tail_is_retired_once_by_the_evictor() {
	let _serialised = migration_test_lock::lock();

	for policy in TIERED {
		let (mut worker, objects) = make_worker(policy);

		const K: HashedKey = 1;
		const X: HashedKey = 2;
		set(&mut worker, &objects, K, LEN, Fast);
		drain_and_apply(&mut worker);
		set(&mut worker, &objects, X, LEN, Fast);
		drain_and_apply(&mut worker);

		publish_del(&worker.status, &worker.overhead_manager, &objects, K);

		let victim = worker.policy_stack.as_mut().expect("a stack").evict_one();
		assert_eq!(victim, Some(X), "{policy}: the DEAD tail was nominated");
		assert_eq!(objects.linked(), 1, "{policy}: the DEAD slot was retired");
		charges_exact(&objects, false);

		worker.handle_del(K);
		drain_and_apply(&mut worker);

		assert_eq!(objects.linked(), 1, "{policy}: the Del retired something else");
		assert_settled(&mut worker);
		charges_exact(&objects, true);
	}
}

/// Race R10: a DEAD slot where the settle's victim would be -- the fast
/// boundary (LRU, FIFO), or the LFU fast minimum -- is RETIRED, not demoted:
/// a `(k, Slow)` for it would land on whatever value holds the key next.
/// Under LFU the retire latches admission, as a demotion does. Red with the
/// DEAD boundary demoted (`demotedead`), and for LFU below.
#[cfg(feature = "merged_object_store")]
#[test]
fn r10_a_dead_slot_at_the_tier_boundary_is_retired_not_demoted() {
	let _serialised = migration_test_lock::lock();

	for policy in [PaperPolicy::LruCompactHybrid, PaperPolicy::FifoCompactHybrid] {
		let (mut worker, objects) = make_worker(policy);

		const K: HashedKey = 1;
		set(&mut worker, &objects, K, LEN, Fast);
		drain_and_apply(&mut worker);

		for key in 2..=8 {
			set(&mut worker, &objects, key, LEN, Fast);
			drain_and_apply(&mut worker);
		}

		assert_eq!(placement(&worker, K), Some(Fast), "{policy}: K fast, at the boundary");

		publish_del(&worker.status, &worker.overhead_manager, &objects, K);

		// An admission whose settle reaches the boundary.
		set(&mut worker, &objects, 100, 12 * LEN, Fast);
		let drain = drain_and_apply(&mut worker);

		assert_eq!(of(&drain, K), vec![], "{policy}: the DEAD boundary was demoted: {drain:?}");
		assert!(drain.iter().any(|&(key, tier, _)| key != K && tier == Slow), "{policy}: the settle demoted nothing");

		worker.handle_del(K);
		drain_and_apply(&mut worker);
		assert_settled(&mut worker);
		charges_exact(&objects, true);
	}

	// LFU: the DEAD slot is the fast MINIMUM (`demote_freq_min`). `k` at
	// frequency 1 in one shard, `a` hit twice in another. The clients delete
	// `k` and shrink `a`, and a resize puts the tier over its watermark on
	// the stale totals. The settle's first step folds `k`'s shard, is still
	// over (`a`'s shrink is pending in its own shard), and finds the DEAD
	// minimum: it retires it, queues no `(k, Slow)` and LATCHES -- the loop
	// runs only while the tier is over its target, and the DashMap stack
	// demotes the deleted key there and latches. The next step folds `a`'s
	// shard, is under the target, and stops (the re-check after the fold), so
	// the latch is the retire's alone. Red with the DEAD minimum demoted
	// (`demotedeadlfu`), with the retire not latching (`retirenolatch`) and
	// without the re-check (`norecheck`: `a` is demoted).
	let (mut worker, objects) = make_worker(PaperPolicy::LfuCompactHybrid);

	let k = in_shard(3, 1);
	let a = in_shard(5, 2);

	set(&mut worker, &objects, k, LEN, Fast);
	drain_and_apply(&mut worker);
	set(&mut worker, &objects, a, 8 * LEN, Fast);
	drain_and_apply(&mut worker);

	for _ in 0..2 {
		worker.handle_get(a, Some(Fast));
		drain_and_apply(&mut worker);
	}

	assert_eq!((placement(&worker, k), placement(&worker, a)), (Some(Fast), Some(Fast)), "LFU: both fast");
	assert!(!stack(&worker).admission_latched(), "LFU: not latched yet");

	publish_del(&worker.status, &worker.overhead_manager, &objects, k);
	let shrunk = publish(&worker.status, &worker.overhead_manager, &objects, a, LEN, Fast);

	// Four of `LEN`'s items of value budget: over the watermark on the stale
	// totals (`k`'s and `a`'s old bytes), under the target once both are
	// folded (`a`'s new bytes alone).
	let omega = stack(&worker).dram_reserved_bytes() / objects.linked() as CacheSize;
	worker.handle_resize_fast_tier(4 * charge(LEN) + 2 * omega);

	assert!(
		stack(&worker).admission_latched() && worker.status.hybrid_admission_latched(),
		"LFU: the retire of the DEAD minimum latched admission, and the latch is published",
	);

	let drain = drain_and_apply(&mut worker);

	assert_eq!(of(&drain, k), vec![], "LFU: the DEAD minimum was demoted: {drain:?}");
	assert_eq!(of(&drain, a), vec![], "LFU: the fold brought the tier under, yet `a` was demoted: {drain:?}");
	assert_eq!(objects.linked(), 1, "LFU: the DEAD minimum was retired");

	worker.handle_del(k);
	handle(&mut worker, a, shrunk);
	drain_and_apply(&mut worker);
	assert_settled(&mut worker);
	charges_exact(&objects, true);
}

/// Race R11: a `Del` -- and a reap's `Expire` -- racing a re-set. The
/// client's delete left a DEAD slot and its re-set a live one on the same
/// chain; the worker links the live one, then retires the DEAD one, never
/// the live one. The `Expire` is taken whatever the map holds (the merged
/// handle's `remove` is a retire), where the DashMap stacks' is guarded on
/// the key being gone. Red with the live slot retired (`retirelive`) and
/// with the guard kept (`expireguard`: the DEAD slot is left behind).
#[cfg(feature = "merged_object_store")]
#[test]
fn r11_a_del_or_reap_racing_a_reset_retires_the_dead_slot_not_the_live_one() {
	let _serialised = migration_test_lock::lock();

	for reaped in [false, true] {
		let (mut worker, objects) = make_worker(PaperPolicy::LruCompactHybrid);

		const K: HashedKey = 1;

		match reaped {
			false => set(&mut worker, &objects, K, LEN, Fast),

			true => {
				let p = publish_expired(&worker, &objects, K, LEN);
				handle(&mut worker, K, p);
			},
		}

		drain_and_apply(&mut worker);

		match reaped {
			false => publish_del(&worker.status, &worker.overhead_manager, &objects, K),
			true => reap(&worker, &objects, K),
		}

		let v2 = publish(&worker.status, &worker.overhead_manager, &objects, K, 2 * LEN, Fast);
		assert!(v2.fresh());

		handle(&mut worker, K, v2);
		drain_and_apply(&mut worker);

		match reaped {
			false => worker.handle_del(K),
			true => worker.handle_expire(K),
		}

		drain_and_apply(&mut worker);

		assert_eq!(placement(&worker, K), Some(Fast), "reaped {reaped}: the live value was retired");
		assert_eq!(objects.get_ref(&K).map(|o| o.data_size() as usize), Some(2 * LEN));
		assert_settled(&mut worker);
		charges_exact(&objects, true);
	}
}

/// A reader never sees a DEAD slot: after a client's delete, with its `Del`
/// not handled, the value is gone to every reader -- `get_ref`,
/// `contains_key`, the tier, and `admission_tier`'s physical read (a new
/// key). Red with `find` returning DEAD slots (`deadvisible`).
#[cfg(feature = "merged_object_store")]
#[test]
fn a_reader_never_sees_a_dead_slot() {
	let _serialised = migration_test_lock::lock();

	let (mut worker, objects) = make_worker(PaperPolicy::FifoCompactHybrid);

	const K: HashedKey = 1;
	set(&mut worker, &objects, K, LEN, Fast);
	drain_and_apply(&mut worker);
	worker.handle_resize_fast_tier(1);
	drain_and_apply(&mut worker);
	assert_eq!(bytes_tier(&objects, K), Slow);

	publish_del(&worker.status, &worker.overhead_manager, &objects, K);

	assert!(objects.get_ref(&K).is_none());
	assert!(!objects.contains_key(&K));
	assert_eq!(objects.tier_of(K), None);
	assert_eq!(
		admission_tier(PaperPolicy::FifoCompactHybrid, K, &worker.status, &objects),
		Fast,
		"a deleted key is built as a new one, not in its old tier",
	);

	worker.handle_del(K);
	drain_and_apply(&mut worker);
	assert_settled(&mut worker);
	charges_exact(&objects, true);
}

/// A `del` of a linked value that has EXPIRED sends no `Del`: its `erase`
/// removes the value -- which goes DEAD -- and answers `KeyNotFound`, so
/// `PaperCache::del` returns before its broadcast. The DEAD slot's event is
/// the TTL reaper's `Expire` for the same due entry, whose erase finds
/// nothing and which is sent all the same.
#[cfg(feature = "merged_object_store")]
#[test]
fn a_del_of_an_expired_linked_value_is_retired_by_its_reap() {
	let _serialised = migration_test_lock::lock();

	let (mut worker, objects) = make_worker(PaperPolicy::LruCompactHybrid);

	const K: HashedKey = 1;
	let published = publish_expired(&worker, &objects, K, LEN);
	handle(&mut worker, K, published);
	drain_and_apply(&mut worker);
	assert_eq!(objects.linked(), 1);

	let deleted = erase(&objects, &worker.status, &worker.overhead_manager, Some(EraseKey::Original(&K, K)));
	assert!(matches!(deleted, Err(CacheError::KeyNotFound)), "an expired value's del is not found");
	assert_eq!((objects.linked(), objects.len()), (1, 0), "the value went DEAD, and no Del follows");

	reap(&worker, &objects, K);
	worker.handle_expire(K);
	drain_and_apply(&mut worker);

	assert_eq!(objects.linked(), 0, "the reap's Expire retired it");
	assert_settled(&mut worker);
	charges_exact(&objects, true);
}

/// Every order under a concurrent workload -- four client threads per cache
/// setting, overwriting, deleting, getting and setting with a 1 s TTL over a
/// small key space -- in a TIGHT cache (evictions throughout, the fast tier
/// settling on almost every set) and a ROOMY one (no eviction, no settle: only
/// the `Del`s and `Expire`s retire what the clients delete); and then, at
/// quiescence, every charge exact: invariant I and the link count
/// (`verify_charges`: no DEAD slot, no unlinked value, nothing unfolded,
/// `linked == len`), the gauges equal to the shards, and a clean audit. What
/// the hand-arranged races cannot reach: a missed fold, a DEAD slot left
/// behind, a link lost.
#[cfg(feature = "merged_object_store")]
#[test]
fn a_concurrent_workload_leaves_every_charge_exact_at_quiescence() {
	let _serialised = migration_test_lock::lock();

	let caches: Vec<_> = TIERED
		.iter()
		.flat_map(|&policy| [(policy, 512 * 1024, 64 * 1024), (policy, 8 << 20, 4 << 20)])
		.map(|(policy, max, fast)| {
			std::thread::spawn(move || {
				let cache = std::sync::Arc::new(
					PaperCache::<u64, TieredBuffer>::new(max, CacheTierSize::Bytes(fast), policy)
						.expect("a hybrid cache"),
				);

				let clients: Vec<_> = (0..4u64)
					.map(|t| {
						let cache = cache.clone();

						std::thread::spawn(move || {
							let mut x = 0x2545_F491_4F6C_DD1Du64 ^ (t + 1).wrapping_mul(0x9E37_79B9_7F4A_7C15);

							for _ in 0..4_000 {
								x ^= x << 13;
								x ^= x >> 7;
								x ^= x << 17;

								let key = x % 384;
								let len = 100 + (x >> 20) as usize % 3_000;

								match (x >> 40) % 10 {
									0..=3 => { let _ = cache.set(key, &vec![key as u8; len], None); },
									4 => { let _ = cache.set(key, &vec![key as u8; len], Some(1)); },
									5 => { let _ = cache.del(&key); },
									_ => { let _ = cache.get(&key); },
								}
							}
						})
					})
					.collect();

				for client in clients {
					client.join().expect("a client thread panicked");
				}

				(policy, max, cache)
			})
		})
		.collect::<Vec<_>>()
		.into_iter()
		.map(|thread| thread.join().expect("a cache thread panicked"))
		.collect();

	// Past every TTL, so the reaper has taken them all and sent its `Expire`s.
	std::thread::sleep(Duration::from_millis(2_500));

	for (policy, max, cache) in caches {
		// The audit is handled after every event before it; two passes more
		// and the worker's last eviction pass is done too.
		let audit = cache.placement_audit().expect("an audit");
		let passes = cache.status.policy_worker_passes();
		wait_for("two more passes", Duration::from_secs(10), || cache.status.policy_worker_passes() >= passes + 2);

		cache.objects.verify_charges(true);

		let audit_again = cache.placement_audit().expect("an audit");
		assert!(audit_again.is_clean(), "{policy} max {max}: {audit_again:?} (first: {audit:?})");
	}
}

/// T14, the uniform differential: one script per order, run in every unit
/// build with the build's own store, recorded op by op as text.
///
/// Identical in every build: the worker driven by hand with the migration
/// queue OFF (landings inline, deterministic); the two per-object constants
/// that differ between the stores by design forced equal for the whole run
/// (`test_overheads`: the fast-tier reservation, omega, and the per-object
/// overhead `used_size` adds); a 24 KiB fast tier in a 64 KiB cache; 96 keys
/// spread over the merged shards -- some three times what the cache holds, so
/// sets evict throughout; item sizes that are exact jemalloc classes
/// (`value_len`), so both stores charge the same bytes; `MERGED_UPDATE_INTERVAL`
/// unset. Each op is what the event loop does for one event: the client half,
/// the worker's handler, the event's drain applied, then the batch end -- the
/// eviction pass, its drain, the gauges. Recorded per op: the event's drain
/// and the batch end's (keys as indices, tiers F/S, origins s/r), the victims
/// in order, every live key's placement, and the cache's own stats (none of
/// the process-global ones). Checked per op in every build: a clean audit,
/// the stack counting exactly the live keys, the merged store's charges, and
/// no promote-then-demote pair for the key a hit or an overwrite touched.
///
/// With `PAPER_T14_DIR` set, each test writes `<dir>/<name>.txt` and prints its
/// hash; the bp-s4 runner diffs the D, M and H files (and TD, TM, TH). The
/// LFU script opens with 32 sets built so that the 32nd admission's MIGRATING
/// bytes fit the gate and its BASE size does not -- the case where the merged
/// store's gate used to add the other one.
mod t14 {
	use super::*;
	use super::super::reconcile_tests::Published;
	use crate::object::Object;
	use std::num::NonZeroU32;

	const OMEGA: ObjectSize = 64;
	const PER_OBJECT: ObjectSize = 100;
	const FAST_TIER: CacheSize = 24 * 1024;
	const MAX_SIZE: CacheSize = 64 * 1024;
	const KEYS: u64 = 96;
	const OPS: usize = 600;
	const ITEMS: [ObjectSize; 9] = [512, 640, 768, 1024, 1280, 1536, 2048, 3072, 4096];

	fn key(i: u64) -> HashedKey {
		(i + 1).wrapping_mul(0x9E37_79B9_7F4A_7C15)
	}

	/// The value length whose whole item costs exactly `item` bytes.
	fn value_len(item: ObjectSize) -> usize {
		let len = item - crate::object::overhead::value_header_bytes::<u64>();

		assert_eq!(
			crate::object::overhead::resident_object_bytes::<u64>(len),
			item,
			"a {len}-byte value is not an item of exactly {item} bytes",
		);

		len as usize
	}

	struct Rng(u64);

	impl Rng {
		fn next(&mut self) -> u64 {
			self.0 = self.0.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1_442_695_040_888_963_407);
			self.0 >> 33
		}

		fn below(&mut self, n: u64) -> u64 {
			self.next() % n
		}
	}

	fn tier(tier: Tier) -> char {
		match tier {
			Fast => 'F',
			Slow => 'S',
		}
	}

	struct Run {
		policy: PaperPolicy,
		tiered: bool,
		/// T14b: no gauge refresh after an op -- the stack's own gauges are
		/// recorded instead -- and TTL reaps, bursts and wipes among the ops.
		b: bool,
		worker: Worker,
		objects: Objects,
		index: HashMap<HashedKey, u64>,
		lines: Vec<String>,
	}

	/// `publish` of a value built with `expiry` -- `PaperCache::set` with a TTL.
	fn publish_object(worker: &Worker, objects: &Objects, key: HashedKey, len: usize, built: Tier, expiry: Option<NonZeroU32>) -> Published {
		let object = Object::with_expiry_in(key, &vec![key as u8; len], built, expiry);
		let base_size = worker.overhead_manager.base_size(&object);
		let resident = worker.overhead_manager.dram_resident_size(&object);
		let mark = worker.status.migration_in_flight().mark(key);

		let previous = objects.insert(key, object).map(|old| worker.overhead_manager.base_size(&old));

		match previous {
			Some(old) => worker.status.update_base_used_size(base_size as i64 - old as i64),

			None => {
				worker.status.incr_num_objects();
				worker.status.update_base_used_size(base_size as i64);
			},
		}

		Published { base_size, resident, built, previous, mark }
	}

	impl Run {
		fn new(policy: PaperPolicy, tiered: bool, b: bool) -> Run {
			assert!(
				std::env::var_os("MERGED_UPDATE_INTERVAL").is_none(),
				"T14 compares exact orders: MERGED_UPDATE_INTERVAL must be unset",
			);

			let (_tx, rx) = crossbeam_channel::unbounded::<WorkerEvent>();

			let objects: Objects = crate::new_hybrid_object_map();
			let status = Arc::new(AtomicStatus::new(MAX_SIZE, &[policy], policy).unwrap());
			let overhead_manager = Arc::new(OverheadManager::new(&status));

			let mut worker = match tiered {
				true => PolicyWorker::new_with_tier_migration(rx, objects.clone(), status, overhead_manager).unwrap(),
				false => PolicyWorker::new(rx, objects.clone(), status, overhead_manager, None).unwrap(),
			};

			worker.migration_queue = None;
			worker.evicted = Some(Vec::new());
			worker.drained = Some(Vec::new());

			if tiered {
				worker.handle_resize_fast_tier(FAST_TIER);
				worker.apply_tier_migrations();
			}

			let index = (0..KEYS).map(|i| (key(i), i)).collect();

			Run { policy, tiered, b, worker, objects, index, lines: Vec::new() }
		}

		fn live(&self, i: u64) -> bool {
			self.objects.get_ref(&key(i)).is_some()
		}

		/// In the map with its TTL passed: not reaped yet (T14b).
		fn expired(&self, i: u64) -> bool {
			self.objects.get_ref(&key(i)).is_some_and(|object| object.is_expired())
		}

		fn names(&self, keys: &[HashedKey]) -> String {
			keys.iter().map(|k| format!("k{}", self.index[k])).collect::<Vec<_>>().join(",")
		}

		fn entries(&self, drain: &[TaggedMigration]) -> String {
			drain
				.iter()
				.map(|&(k, t, o)| {
					let origin = match o {
						MigrationOrigin::Stack => 's',
						MigrationOrigin::Reconcile => 'r',
					};

					format!("k{}{}{}", self.index[&k], tier(t), origin)
				})
				.collect::<Vec<_>>()
				.join(",")
		}

		/// The client's set, as `PaperCache::set` does it, then its event.
		fn set(&mut self, i: u64, item: ObjectSize) -> String {
			let k = key(i);
			let built = admission_tier(self.policy, k, &self.worker.status, &self.objects);
			let published = publish(&self.worker.status, &self.worker.overhead_manager, &self.objects, k, value_len(item), built);
			let kind = if published.fresh() { "set" } else { "overwrite" };

			handle(&mut self.worker, k, published);

			format!("{kind} k{i} {item} built {}", tier(built))
		}

		/// The client's get, as `PaperCache::get` does it, then its event.
		fn get(&mut self, i: u64) -> String {
			let k = key(i);
			let served = self.objects
				.get_ref(&k)
				.filter(|object| !object.is_expired())
				.map(|object| object.value().tier());

			match served {
				Some(t) => {
					self.worker.status.incr_hits();
					self.worker.status.incr_served_hit(t);
				},

				None => self.worker.status.incr_misses(),
			}

			self.worker.handle_get(k, served);

			match served {
				Some(t) => format!("get k{i} hit {}", tier(t)),
				None => format!("get k{i} miss"),
			}
		}

		fn del(&mut self, i: u64) -> String {
			let k = key(i);
			publish_del(&self.worker.status, &self.worker.overhead_manager, &self.objects, k);
			self.worker.status.incr_dels();
			self.worker.handle_del(k);

			format!("del k{i}")
		}

		/// T14b: a set whose value has ALREADY expired (its expiry is tick 1):
		/// until the reaper takes it it is in the map and the stack, and every
		/// get of it misses.
		fn set_expired(&mut self, i: u64, item: ObjectSize) -> String {
			let k = key(i);
			let built = admission_tier(self.policy, k, &self.worker.status, &self.objects);
			let published = publish_object(&self.worker, &self.objects, k, value_len(item), built, NonZeroU32::new(1));
			let kind = if published.fresh() { "set" } else { "overwrite" };

			handle(&mut self.worker, k, published);

			format!("{kind} k{i} {item} expired built {}", tier(built))
		}

		/// T14b: the TTL reaper's take of an expired value, then the worker's
		/// `Expire`. (Not a re-set between them: a `Set` handled before the
		/// `Expire` it follows is an accepted difference between the stores.)
		fn reap(&mut self, i: u64) -> String {
			let k = key(i);
			let _ = erase(&self.objects, &self.worker.status, &self.worker.overhead_manager, Some(EraseKey::Expired(k)));
			self.worker.handle_expire(k);

			format!("reap k{i}")
		}

		/// T14b: sets of new keys all published before the worker takes the
		/// first -- a worker backlog, one client -- then each `Set` handled
		/// with its own drain, as the event loop drains after every event (the
		/// last one's is the step's). The client builds every one with the
		/// latch and tiers published before the burst, so a `Set` inside it
		/// that latches LFU leaves the later keys built fast and placed slow:
		/// the reconcile's correctives. In the merged store the later keys are
		/// UNLINKED while the earlier ones are linked, charged and settled.
		fn burst(&mut self, keys: &[(u64, ObjectSize)]) -> String {
			let published: Vec<(u64, ObjectSize, Tier, Published)> = keys
				.iter()
				.map(|&(i, item)| {
					let k = key(i);
					let built = admission_tier(self.policy, k, &self.worker.status, &self.objects);
					let published = publish(&self.worker.status, &self.worker.overhead_manager, &self.objects, k, value_len(item), built);

					(i, item, built, published)
				})
				.collect();

			let mut what = Vec::new();
			let last = published.len() - 1;

			for (n, (i, item, built, published)) in published.into_iter().enumerate() {
				handle(&mut self.worker, key(i), published);

				if n < last {
					self.worker.apply_tier_migrations();
				}

				what.push(format!("k{i} {item} built {}", tier(built)));
			}

			format!("burst {}", what.join(" "))
		}

		/// One op: `op` is its client half and handler; then the event's
		/// drain, the batch end, and the record.
		fn step(&mut self, n: usize, op: impl FnOnce(&mut Run) -> String, touched: Option<u64>) {
			let what = op(self);

			self.worker.apply_tier_migrations();
			let drain = self.worker.drained.replace(Vec::new()).expect("recording");

			self.worker.apply_evictions(&mut Vec::new()).expect("an eviction pass");
			let evicted = self.worker.evicted.replace(Vec::new()).expect("recording");

			self.worker.apply_tier_migrations();
			let drain2 = self.worker.drained.replace(Vec::new()).expect("recording");

			// T14b leaves the gauges and the latch as the events published them.
			if !self.b {
				self.worker.refresh_tier_gauges();
			}

			// T8's property, on a real script: the key a hit or an overwrite
			// touched is never promoted and then demoted in one event's drain.
			if let Some(i) = touched {
				let mine: Vec<Tier> = drain.iter().filter(|e| e.0 == key(i)).map(|e| e.1).collect();

				assert!(
					!mine.windows(2).any(|w| w == [Fast, Slow]),
					"{} op {n} ({what}): a promote-then-demote pair for k{i}: {mine:?}",
					self.policy,
				);
			}

			let live: Vec<u64> = (0..KEYS).filter(|&i| self.live(i)).collect();

			let placements = live
				.iter()
				.map(|&i| {
					let p = stack(&self.worker).placement_of(key(i)).map_or('-', tier);
					format!("k{i}={p}")
				})
				.collect::<Vec<_>>()
				.join(" ");

			let s = self.worker.status.hybrid_stats();
			let used = self.worker.status.used_size(&self.policy);

			// A flat cache has no tiers: a flat DashMap stack places nothing
			// and gauges nothing, while the merged store tags its slots fast
			// either way. Its record is the order alone -- the victims, the
			// live keys -- and the accounted size. T14b records the stack's own
			// gauges, which no refresh has copied into the status.
			if self.b {
				let (fb, sb, fo, so, meta) = gauges(&self.worker);

				self.lines.push(format!(
					"op {n} {what} | drain [{}] | evicted [{}] | drain2 [{}] | placement {placements} | stack fo={fo} so={so} fb={fb} sb={sb} meta={meta} | stats promo={} demo={} evict={} rafast={} raslow={} fasthits={} slowhits={} latched={} used={used} live={}",
					self.entries(&drain),
					self.names(&evicted),
					self.entries(&drain2),
					s.promotions,
					s.demotions,
					s.evictions,
					s.reconcile_applied_to_fast,
					s.reconcile_applied_to_slow,
					s.fast_hits,
					s.slow_hits,
					self.worker.status.hybrid_admission_latched(),
					live.len(),
				));
			} else if !self.tiered {
				let keys = live.iter().map(|i| format!("k{i}")).collect::<Vec<_>>().join(" ");

				self.lines.push(format!(
					"op {n} {what} | drain [{}] | evicted [{}] | drain2 [{}] | live {keys} | used={used}",
					self.entries(&drain),
					self.names(&evicted),
					self.entries(&drain2),
				));
			} else {
			self.lines.push(format!(
				"op {n} {what} | drain [{}] | evicted [{}] | drain2 [{}] | placement {placements} | stats fo={} so={} fb={} sb={} meta={} promo={} demo={} evict={} rafast={} raslow={} fasthits={} slowhits={} latched={} used={used} live={}",
				self.entries(&drain),
				self.names(&evicted),
				self.entries(&drain2),
				s.fast_objects,
				s.slow_objects,
				s.fast_bytes_used,
				s.slow_bytes_used,
				s.fast_metadata_bytes,
				s.promotions,
				s.demotions,
				s.evictions,
				s.reconcile_applied_to_fast,
				s.reconcile_applied_to_slow,
				s.fast_hits,
				s.slow_hits,
				self.worker.status.hybrid_admission_latched(),
				live.len(),
			));
			}

			// Checked in every build, after every op.
			assert_eq!(stack(&self.worker).len(), live.len(), "{} op {n} ({what}): the stack does not count the live keys", self.policy);

			if self.tiered {
				let audit = self.worker.placement_audit();
				assert!(audit.is_clean(), "{} op {n} ({what}): {audit:?}", self.policy);
			}

			charges_exact(&self.objects, true);
		}
	}

	/// The script: the LFU prefix, then `OPS` ops drawn from a fixed LCG --
	/// T14's mix, or with `b` T14b's (`script_b_op`).
	fn script(name: &str, policy: PaperPolicy, tiered: bool, seed: u64, b: bool) {
		// Its inline landings count in the process-wide migration counters,
		// which other tests assert exact deltas on under this lock.
		let _serialised = migration_test_lock::lock();
		let _overheads = test_overheads::set(OMEGA, PER_OBJECT);

		let mut run = Run::new(policy, tiered, b);
		let mut rng = Rng(seed);
		let mut n = 0;

		// 31 keys of 640 and 768 bytes, alternating, then a 768 whose
		// migrating bytes fill the LFU gate exactly: 21,760 + 768 = 22,528 =
		// 24,576 - 32 x 64, while its base size is 12 bytes more.
		for i in 0..32u64 {
			let item = if i % 2 == 0 && i < 31 { 640 } else { 768 };
			run.step(n, |run| run.set(i, item), None);
			n += 1;
		}

		while n < OPS {
			let r = rng.below(100);
			let i = rng.below(KEYS);
			let item = ITEMS[rng.below(ITEMS.len() as u64) as usize];

			if b {
				script_b_op(&mut run, &mut rng, n, r, i, item);
				n += 1;
				continue;
			}

			match r {
				// A set of an absent key (or, if it is live, an overwrite).
				0..40 => run.step(n, |run| run.set(i, item), None),

				// An overwrite of a live key: half resized, half the same size.
				40..55 => {
					let resized = rng.below(2) == 0;

					match (0..KEYS).map(|d| (i + d) % KEYS).find(|&j| run.live(j)) {
						Some(j) => {
							let same = run.objects.get_ref(&key(j)).map(|o| {
								crate::object::overhead::resident_object_bytes::<u64>(o.data_size() as ObjectSize)
							});
							let item = if resized { item } else { same.expect("live") };

							run.step(n, |run| run.set(j, item), Some(j));
						},

						None => run.step(n, |run| run.set(i, item), None),
					}
				},

				// A get: a hit if the key is live, else a miss.
				55..85 => run.step(n, |run| run.get(i), Some(i)),

				// A delete of a live key.
				85..93 => match (0..KEYS).map(|d| (i + d) % KEYS).find(|&j| run.live(j)) {
					Some(j) => run.step(n, |run| run.del(j), None),
					None => run.step(n, |run| run.get(i), Some(i)),
				},

				// The fast tier resized within [12, 32] KiB: shrinks and grows
				// (a grow unlatches LFU).
				93..97 if tiered => {
					let size = (12 + rng.below(21)) * 1024;

					run.step(n, |run| {
						run.worker.handle_resize_fast_tier(size);
						format!("resize_fast_tier {size}")
					}, None);
				},

				// The cache's size: shrunk by up to a quarter, or restored.
				_ => {
					let size = match rng.below(2) {
						0 => MAX_SIZE - rng.below(MAX_SIZE / 4),
						_ => MAX_SIZE,
					};

					run.step(n, |run| {
						run.worker.status.set_max_size(size);
						run.worker.handle_resize(size);
						format!("resize {size}")
					}, None);
				},
			}

			n += 1;
		}

		let text = run.lines.join("\n") + "\n";

		let mut hash: u64 = 0xcbf2_9ce4_8422_2325;

		for byte in text.bytes() {
			hash ^= byte as u64;
			hash = hash.wrapping_mul(0x0100_0000_01b3);
		}

		println!("T14 {name} ops={OPS} fnv64={hash:016x}");

		if let Some(dir) = std::env::var_os("PAPER_T14_DIR") {
			let path = std::path::Path::new(&dir).join(format!("{name}.txt"));
			std::fs::write(&path, &text).unwrap_or_else(|e| panic!("writing {path:?}: {e}"));
		}
	}

	/// One T14b op. Over T14's mix: sets whose value has already expired, the
	/// reaper's take of one and its `Expire`, bursts of three new keys
	/// published before the worker takes the first, and wipes; deletes only of
	/// unexpired values (a `del` of an expired one sends no `Del`, and until
	/// its `Expire` the DashMap stack charges it while the merged store has
	/// folded it out). Every op is still a sequence the stores agree on: the
	/// burst's keys are new and distinct, and nothing is deleted or
	/// overwritten inside it (the lag-only differences R2/R4 and (a)).
	fn script_b_op(run: &mut Run, rng: &mut Rng, n: usize, r: u64, i: u64, item: ObjectSize) {
		let next = |run: &Run, from: u64, want: &dyn Fn(&Run, u64) -> bool| {
			(0..KEYS).map(|d| (from + d) % KEYS).find(|&j| want(run, j))
		};

		let live_unexpired = |run: &Run, j: u64| run.live(j) && !run.expired(j);

		match r {
			// A set: new, or an overwrite (of an expired value too).
			0..30 => run.step(n, |run| run.set(i, item), None),

			// A set whose value has already expired, of an absent key.
			30..38 => match next(run, i, &|run, j| !run.live(j)) {
				Some(j) => run.step(n, |run| run.set_expired(j, item), None),
				None => run.step(n, |run| run.get(i), Some(i)),
			},

			// An overwrite of an unexpired value: half resized, half not.
			38..48 => {
				let resized = rng.below(2) == 0;

				match next(run, i, &live_unexpired) {
					Some(j) => {
						let same = run.objects.get_ref(&key(j)).map(|o| {
							crate::object::overhead::resident_object_bytes::<u64>(o.data_size() as ObjectSize)
						});
						let item = if resized { item } else { same.expect("live") };

						run.step(n, |run| run.set(j, item), Some(j));
					},

					None => run.step(n, |run| run.set(i, item), None),
				}
			},

			// A get: a hit if live and unexpired, else a miss.
			48..72 => run.step(n, |run| run.get(i), Some(i)),

			// A delete of an unexpired value.
			72..79 => match next(run, i, &live_unexpired) {
				Some(j) => run.step(n, |run| run.del(j), None),
				None => run.step(n, |run| run.get(i), Some(i)),
			},

			// The reaper takes an expired value; its `Expire`.
			79..87 => match next(run, i, &|run, j| run.expired(j)) {
				Some(j) => run.step(n, |run| run.reap(j), None),
				None => run.step(n, |run| run.get(i), Some(i)),
			},

			// Three new keys, published before the worker takes the first.
			87..92 => {
				let mut keys = Vec::new();
				let mut j = i;

				while keys.len() < 3 {
					match next(run, j, &|run, k| !run.live(k) && !keys.iter().any(|&(x, _)| x == k)) {
						Some(k) => {
							keys.push((k, ITEMS[rng.below(ITEMS.len() as u64) as usize]));
							j = (k + 1) % KEYS;
						},

						None => break,
					}
				}

				match keys.is_empty() {
					false => run.step(n, |run| run.burst(&keys), None),
					true => run.step(n, |run| run.get(i), Some(i)),
				}
			},

			// The fast tier resized within [12, 32] KiB.
			92..95 => {
				let size = (12 + rng.below(21)) * 1024;

				run.step(n, |run| {
					run.worker.handle_resize_fast_tier(size);
					format!("resize_fast_tier {size}")
				}, None);
			},

			// The cache's size: shrunk by up to a quarter, or restored.
			95..99 => {
				let size = match rng.below(2) {
					0 => MAX_SIZE - rng.below(MAX_SIZE / 4),
					_ => MAX_SIZE,
				};

				run.step(n, |run| {
					run.worker.status.set_max_size(size);
					run.worker.handle_resize(size);
					format!("resize {size}")
				}, None);
			},

			// A wipe, the worker's.
			_ => run.step(n, |run| {
				run.worker.handle_wipe(None);
				"wipe".to_string()
			}, None),
		}
	}

	#[test]
	fn t14_lru_scripts_match_across_stores() {
		script("lru", PaperPolicy::LruCompactHybrid, true, 11, false);
	}

	#[test]
	fn t14_fifo_scripts_match_across_stores() {
		script("fifo", PaperPolicy::FifoCompactHybrid, true, 12, false);
	}

	#[test]
	fn t14_clock_scripts_match_across_stores() {
		script("clock", PaperPolicy::ClockCompactHybrid, true, 13, false);
	}

	#[test]
	fn t14_lfu_scripts_match_across_stores() {
		script("lfu", PaperPolicy::LfuCompactHybrid, true, 14, false);
	}

	/// T14b (S4's follow-ups): the same differential without the gauge refresh
	/// after every op -- so the LFU latch a client builds with is what the
	/// events published (U12), not what a refresh copied -- and with TTL reaps
	/// (U8's `Expire`), worker-backlog bursts (unlinked values in the merged
	/// store at a `Set`, and the reconcile's correctives when a burst latches
	/// LFU) and wipes (U7). Files `<order>-b.txt`.
	#[test]
	fn t14b_lru_scripts_match_across_stores() {
		script("lru-b", PaperPolicy::LruCompactHybrid, true, 31, true);
	}

	#[test]
	fn t14b_fifo_scripts_match_across_stores() {
		script("fifo-b", PaperPolicy::FifoCompactHybrid, true, 32, true);
	}

	#[test]
	fn t14b_clock_scripts_match_across_stores() {
		script("clock-b", PaperPolicy::ClockCompactHybrid, true, 33, true);
	}

	#[test]
	fn t14b_lfu_scripts_match_across_stores() {
		script("lfu-b", PaperPolicy::LfuCompactHybrid, true, 34, true);
	}

	/// The flat caches -- the other half of the merged store's use: the same
	/// scripts with no tiers, against the flat DashMap stacks
	/// (`LruCompactStack`, ...). Only the order is compared: victims, and the
	/// stack counting the live keys.
	#[test]
	fn t14_flat_scripts_match_across_stores() {
		for (name, policy, seed) in [
			("flat-lru", PaperPolicy::LruCompact, 21),
			("flat-fifo", PaperPolicy::FifoCompact, 22),
			("flat-clock", PaperPolicy::ClockCompact, 23),
			("flat-lfu", PaperPolicy::LfuCompact, 24),
		] {
			script(name, policy, false, seed, false);
		}
	}
}
