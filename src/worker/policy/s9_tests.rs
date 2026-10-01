/*
 * Copyright (c) Kia Shakiba
 *
 * This source code is licensed under the GNU AGPLv3 license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! Backpressure plan S9, the library half of the server's SET permit
//! (`crate::permit`; T16): a set admitted before its value is read
//! (`PaperCache::reserve_set`), the value's allocation filled in place
//! (`SetPermit::fill`, `PendingSet`), and committed.
//!
//! P is process-global, so the tests that read it run ALONE, each in a child
//! copy of the test binary (`alone_in`), as the byte gate's do; the tests whose
//! gate must run and hold use that module's helpers (the consumers paused, M
//! held, `fill` to the close level). Every test runs in every unit build, so
//! both value layouts (with `String` keys held as bytes under `thin_header`)
//! and both object stores are covered.

use std::{
	io,
	panic::{self, AssertUnwindSafe},
	thread,
	time::{Duration, Instant},
};

use super::*;
use super::s5_gate_tests::{LEN, M0, TIER, cache, fill, gated, p, v, value, waiters};
use super::s5_tests::{cache_with, evict_to_fit};
use super::test_support::{alone_in, wait_for};

use crate::gate::{self, GateConfig, GateMode, MetadataModel, OnStall, test_hooks};
use crate::{CacheTierSize, PaperCache, PendingSet, TieredBuffer, phys};

use Tier::{Fast, Slow};

type Cache = PaperCache<u64, TieredBuffer>;

fn alone(test: &str, body: impl FnOnce()) {
	alone_in(module_path!(), test, body);
}

/// A deadline far enough off that a reservation which does not wait never
/// meets it.
fn soon() -> Instant {
	Instant::now() + Duration::from_secs(30)
}

// ---------------------------------------------------------------------------
// reserve + fill + commit is set

/// A key type per shape the item can hold a key in: a `u64` (held as a `K`) and
/// a `String` (held as BYTES inside the item under `thin_header`, whose size
/// depends on its length), each with a length that varies with the key.
trait Keyed: 'static + Eq + std::hash::Hash + typesize::TypeSize + Clone + Send + Sync {
	fn make(n: u64) -> Self;
}

impl Keyed for u64 {
	fn make(n: u64) -> u64 {
		n
	}
}

impl Keyed for String {
	fn make(n: u64) -> String {
		format!("key-{n}-{}", "k".repeat((n % 41) as usize))
	}
}

/// Waits until the worker has handled every event sent before this call --
/// two of its gate passes begun since -- and every migration it queued has
/// landed (`s5_gate_tests::quiesce`, for a cache of any key).
fn quiesce<K: Keyed>(cache: &PaperCache<K, TieredBuffer>) {
	let gate = cache.status.gate();
	let passes = gate.passes();

	wait_for("two worker passes", Duration::from_secs(10), || gate.passes() >= passes + 2);
	wait_for("the migrations to land", Duration::from_secs(10), || phys::pending_migrations() == (0, 0));
}

/// Everything two caches that did the same sets must agree on, once idle.
#[derive(Debug, PartialEq)]
struct Outcome {
	/// Per key: where its bytes are, and the bytes themselves.
	objects: Vec<(u64, Option<Tier>, Vec<u8>)>,

	/// The status: objects, and the bytes and metadata they are charged.
	live: u64,
	used: u64,

	/// The stack's, as the worker built it from each set's `Set` event.
	fast_objects: u64,
	slow_objects: u64,
	fast_bytes_used: u64,
	slow_bytes_used: u64,
	structural_slow_sets: u64,

	/// P, exact: and it is the stack's fast bytes, whichever path built them.
	p: i64,
}

const KEYS: u64 = 300;

fn len_of(n: u64) -> usize {
	LEN / 2 + ((n * 7_919) % (LEN as u64)) as usize
}

fn bytes_of(n: u64) -> Vec<u8> {
	vec![(n % 251) as u8 + 1; len_of(n)]
}

fn ttl_of(n: u64) -> Option<u32> {
	(n % 3 == 0).then_some(3_600)
}

/// `KEYS` sets of varying length -- every third with a TTL -- into a fresh
/// tiered cache with the gate on, a tier a quarter of the data, and the
/// consumers running, by `set` or by the three steps; the cache at rest.
fn run<K: Keyed>(by_permit: bool) -> Outcome {
	let cache = {
		let cache = PaperCache::<K, TieredBuffer>::new_with_gate(
			256 << 20,
			CacheTierSize::Bytes(TIER),
			PaperPolicy::LruCompactHybrid,
			gated(Duration::from_secs(30), OnStall::Error),
		)
		.expect("a tiered cache");

		wait_for("the byte gate to enable", Duration::from_secs(10), || cache.hybrid_stats().gate_state == gate::GateState::Enabled);
		cache
	};

	for n in 0..KEYS {
		let (key, bytes, ttl) = (K::make(n), bytes_of(n), ttl_of(n));

		if !by_permit {
			cache.set(key, &bytes, ttl).expect("a set");
			continue;
		}

		let permit = cache.reserve_set(key, bytes.len(), ttl, soon()).expect("a permit");
		assert_eq!(permit.len(), bytes.len());

		let tier = permit.tier();
		let mut pending = permit.fill();

		assert_eq!((pending.len(), pending.filled(), pending.is_full()), (bytes.len(), 0, false));
		assert_eq!(pending.tier(), tier, "the value is allocated in the tier admission decided");

		// The three ways to write the slot, by key: a reader (the server's),
		// two copies, and the raw slot.
		match n % 3 {
			0 => pending.read_exact_from(&mut &bytes[..]).expect("the body"),

			1 => {
				let (head, tail) = bytes.split_at(bytes.len() / 3);

				pending.write(head);
				assert_eq!(pending.filled(), head.len());
				pending.write(tail);
			},

			_ => {
				let slot = pending.unfilled();
				assert_eq!(slot.len(), bytes.len());

				for (to, from) in slot.iter_mut().zip(&bytes) {
					to.write(*from);
				}

				// SAFETY: every byte of the slot was just written.
				unsafe { pending.advance(bytes.len()) };
			},
		}

		assert!(pending.is_full());
		pending.commit().expect("a commit");
	}

	quiesce(&cache);

	let stats = cache.hybrid_stats();
	let mut objects = Vec::new();

	for n in 0..KEYS {
		let key = K::make(n);

		objects.push((n, cache.tier_of(&key), cache.peek(&key).expect("a live key")));
	}

	let outcome = Outcome {
		objects,
		live: cache.status.live_num_objects(),
		used: cache.status.used_size(&PaperPolicy::LruCompactHybrid),
		fast_objects: stats.fast_objects,
		slow_objects: stats.slow_objects,
		fast_bytes_used: stats.fast_bytes_used,
		slow_bytes_used: stats.slow_bytes_used,
		structural_slow_sets: stats.structural_slow_sets,
		p: phys::fast_bytes_signed(),
	};

	assert_eq!(outcome.p as u64, outcome.fast_bytes_used, "P is the stack's fast bytes");
	assert!(outcome.fast_objects > 0 && outcome.slow_objects > 0, "both tiers hold keys: {outcome:?}");

	drop(cache);
	assert_eq!(phys::fast_bytes_signed(), 0, "every allocation was refunded when the cache dropped");

	outcome
}

/// T16: `set` and reserve + fill + commit are one path. The same 300 sets of
/// varying length, every third with a TTL, through each, into a gated cache
/// whose tier is a quarter of the data: the objects (bytes, tier), what the
/// status charges them, what the stack tracks and P are the same, for a
/// `u64` key and for a `String` key (held as bytes in the item under
/// `thin_header`). Red with the permit's commit building its figures from
/// anything but what admission decided (`sizesfromlen`: the stack's bytes part
/// from P's).
#[test]
fn reserve_fill_commit_is_set() {
	alone("reserve_fill_commit_is_set", || {
		let _m = test_hooks::override_m(M0);

		let a = run::<u64>(false);
		let b = run::<u64>(true);

		assert_eq!(a, b, "u64 keys: the three steps are not `set`");

		let a = run::<String>(false);
		let b = run::<String>(true);

		assert_eq!(a, b, "String keys: the three steps are not `set`");
	});
}

// ---------------------------------------------------------------------------
// abandoning a set

/// A permit dropped before it is filled releases the bytes the gate reserved
/// for it, once, and inserts nothing: P, the object count, the sets counter
/// and the reservation are as they were, and the key is not there. Red with
/// the permit's reservation leaked (`leakreservation`: it stays held and the
/// gate's close level drifts down by a value) or released twice (`doublerelease`).
#[test]
fn a_permit_dropped_before_its_fill_releases_its_reservation_and_inserts_nothing() {
	alone("a_permit_dropped_before_its_fill_releases_its_reservation_and_inserts_nothing", || {
		let _flush = test_hooks::no_flush();
		let _m = test_hooks::override_m(M0);
		let cache = cache(PaperPolicy::LruCompactHybrid, gated(Duration::from_secs(5), OnStall::Error));
		let gate = cache.status.gate();
		let n = cache.hybrid_stats().band_n;

		let _pause = test_hooks::pause_consumers();
		let mut next = 0;

		// To the near level: the next admission reserves its bytes.
		while p() + v() <= n {
			cache.set(next, &value(next), None).expect("a set under N");
			next += 1;
		}

		let (p0, count, used) = (p(), cache.status.live_num_objects(), cache.status.used_size(&PaperPolicy::LruCompactHybrid));

		let permit = cache.reserve_set(next, LEN, None, soon()).expect("admitted, reserving");

		assert_eq!(permit.tier(), Fast);
		assert_eq!(gate.reserved(), v(), "the permit holds the bytes the gate reserved");
		assert_eq!(p(), p0, "nothing is allocated, so P has not moved");

		drop(permit);

		assert_eq!(gate.reserved(), 0, "released, exactly once");
		assert_eq!(cache.hybrid_stats().reserved_bytes, 0);
		assert_eq!(p(), p0);
		assert_eq!(cache.status.live_num_objects(), count, "nothing was inserted");
		assert_eq!(cache.status.used_size(&PaperPolicy::LruCompactHybrid), used);
		assert!(cache.peek(&next).is_err(), "the key is not in the cache");

		// And the gate works after it: the same reservation, taken and used.
		let permit = cache.reserve_set(next, LEN, None, soon()).expect("admitted, reserving");

		assert_eq!(gate.reserved(), v());

		let mut pending = permit.fill();

		assert_eq!(gate.reserved(), 0, "the allocation is P's now: the reservation went back at fill");
		assert_eq!(p(), p0 + v());

		pending.write(&value(next));
		pending.commit().expect("a commit");

		assert_eq!(p(), p0 + v(), "committed: still the one charge");
		assert_eq!(cache.peek(&next).expect("the committed key"), value(next));
		assert_eq!(gate.reserved(), 0);
	});
}

/// A value dropped after its fill -- unwritten, half written, or written
/// whole but never committed -- frees its allocation and refunds P exactly
/// once, and inserts nothing; a slow value was never charged, so nothing is
/// refunded for it. P returns to exactly where it was: a double refund would
/// take it below, a missed one leave it above. Red with the refund missing
/// (`norefund`) or doubled (`doublerefund`).
#[test]
fn a_value_dropped_after_its_fill_refunds_p_exactly_once_and_inserts_nothing() {
	alone("a_value_dropped_after_its_fill_refunds_p_exactly_once_and_inserts_nothing", || {
		let _flush = test_hooks::no_flush();
		let _m = test_hooks::override_m(M0);
		let cache = cache(PaperPolicy::LruCompactHybrid, gated(Duration::from_secs(5), OnStall::Error));
		let gate = cache.status.gate();

		for (what, written) in [("unwritten", 0), ("half written", LEN / 2), ("written whole, not committed", LEN)] {
			let (p0, count, used) = (p(), cache.status.live_num_objects(), cache.status.used_size(&PaperPolicy::LruCompactHybrid));
			let sets = cache.status().expect("the status").total_sets();

			let permit = cache.reserve_set(1, LEN, None, soon()).expect("a permit");
			assert_eq!(permit.tier(), Fast);

			let mut pending = permit.fill();

			assert_eq!(p(), p0 + v(), "{what}: charged when allocated");
			assert_eq!(gate.reserved(), 0, "{what}: the reservation went back at the fill");

			pending.write(&value(1)[..written]);
			assert_eq!(pending.filled(), written);

			drop(pending);

			assert_eq!(p(), p0, "{what}: refunded, once");
			assert_eq!(cache.status.live_num_objects(), count, "{what}: nothing inserted");
			assert_eq!(cache.status.used_size(&PaperPolicy::LruCompactHybrid), used, "{what}");
			assert_eq!(cache.status().expect("the status").total_sets(), sets, "{what}: no set was counted");
			assert!(cache.peek(&1).is_err(), "{what}: the key is not in the cache");
		}

		// A slow value: structural, larger than an empty fast tier.
		let big = 2 * TIER as usize;
		let p0 = p();
		let permit = cache.reserve_set(2, big, None, soon()).expect("a permit for a value that is built slow");

		assert_eq!(permit.tier(), Slow);

		let mut pending = permit.fill();

		pending.write(&vec![9u8; big / 2]);
		assert_eq!(p(), p0, "a slow value charges P nothing");

		drop(pending);

		assert_eq!(p(), p0, "and refunds it nothing");
		assert!(cache.peek(&2).is_err());
		assert_eq!(gate.reserved(), 0);
	});
}

/// A value is committed only once every byte of it is written: `commit`
/// refuses -- panics -- a value that is not, and the set is abandoned as the
/// panic unwinds (P refunded, nothing inserted). A reader that ends early
/// leaves what it delivered counted, and the value can be read on and
/// committed. Red with commit accepting a part-written value
/// (`commitunfilled`: uninitialized bytes are published).
#[test]
fn a_value_is_committed_only_once_it_is_written_whole() {
	alone("a_value_is_committed_only_once_it_is_written_whole", || {
		let mut config = GateConfig::default();
		config.metadata_model = MetadataModel::PerObject;
		let cache = Cache::new_with_gate(256 << 20, CacheTierSize::Bytes(TIER), PaperPolicy::LruCompactHybrid, config).expect("a tiered cache");

		// (a) Part written: the commit panics, and nothing of it stays.
		let p0 = p();
		let mut pending = cache.reserve_set(1, 100, None, soon()).expect("a permit").fill();

		pending.write(&[1u8; 60]);

		let outcome = panic::catch_unwind(AssertUnwindSafe(|| pending.commit()));
		let message = *outcome.expect_err("the commit of a part-written value panics").downcast::<String>().expect("a message");

		assert!(message.contains("every byte"), "{message}");
		assert_eq!(p(), p0, "the abandoned value was refunded");
		assert!(cache.peek(&1).is_err());
		assert_eq!(cache.status.live_num_objects(), 0);

		// (b) A reader that ends early: what it delivered stays, and the rest
		// can be read on.
		let mut pending = cache.reserve_set(2, 100, None, soon()).expect("a permit").fill();
		let error = pending.read_exact_from(&mut &[7u8; 40][..]).expect_err("the reader ended first");

		assert_eq!(error.kind(), io::ErrorKind::UnexpectedEof);
		assert_eq!((pending.filled(), pending.is_full()), (40, false));

		pending.read_exact_from(&mut &[8u8; 60][..]).expect("the rest");
		assert!(pending.is_full());
		pending.commit().expect("a commit");

		let mut expected = vec![7u8; 40];
		expected.extend([8u8; 60]);

		assert_eq!(cache.peek(&2).expect("the committed key"), expected);

		// (c) An empty value is whole at once.
		let pending = cache.reserve_set(3, 0, None, soon()).expect("a permit").fill();

		assert!(pending.is_full() && pending.is_empty());
		pending.commit().expect("an empty value");
		assert_eq!(cache.peek(&3).expect("the empty value"), Vec::<u8>::new());
	});
}

/// Setters running together, a good part of whose sets are abandoned at a
/// random point -- the permit dropped, the value dropped unwritten, half
/// written, or whole but never committed -- leave nothing behind: the gate
/// holds no reservation, P is the stack's fast bytes and returns to 0 when
/// the cache drops, and the cache holds exactly the keys that were committed,
/// each with its bytes. Red with a reservation that is leaked or released
/// twice, or an allocation that is not refunded exactly once
/// (`leakreservation`, `norefund`, `doublerefund`), under contention too.
#[test]
fn setters_that_abandon_at_random_leave_nothing_behind() {
	alone("setters_that_abandon_at_random_leave_nothing_behind", || {
		let _m = test_hooks::override_m(M0);
		let cache = cache(PaperPolicy::LruCompactHybrid, gated(Duration::from_secs(30), OnStall::Error));
		let gate = cache.status.gate();
		let committed = std::sync::Mutex::new(Vec::new());

		const SETTERS: u64 = 4;
		const SETS: u64 = 120;

		thread::scope(|scope| {
			for setter in 0..SETTERS {
				let (cache, committed) = (&cache, &committed);

				scope.spawn(move || {
					let _live = cache.register_setter();

					for n in 0..SETS {
						let key = setter * 1_000 + n;
						let bytes = vec![(key % 251) as u8 + 1; LEN / 2 + (key as usize * 37) % LEN];
						let fate = (key * 7 + setter) % 9;

						let permit = cache.reserve_set(key, bytes.len(), None, Instant::now() + Duration::from_secs(20)).expect("a permit");

						if fate == 0 {
							drop(permit);
							continue;
						}

						let mut pending = permit.fill();

						match fate {
							1 => {},
							2 => pending.write(&bytes[..bytes.len() / 2]),
							3 => pending.write(&bytes),

							_ => {
								pending.write(&bytes);
								pending.commit().expect("a commit");
								committed.lock().expect("the list").push((key, bytes));
								continue;
							},
						}

						drop(pending);
					}
				});
			}
		});

		quiesce(&cache);

		let committed = committed.into_inner().expect("the list");
		let stats = cache.hybrid_stats();

		assert!(committed.len() > 150 && committed.len() < (SETTERS * SETS) as usize, "{} committed", committed.len());
		assert_eq!(gate.reserved(), 0, "no reservation is left held");
		assert_eq!(cache.status.live_num_objects(), committed.len() as u64, "exactly the committed keys");
		assert_eq!(p(), stats.fast_bytes_used, "P is the stack's fast bytes");
		assert!(stats.demotions > 0, "the gate held sets and demotions freed them");

		for (key, bytes) in &committed {
			assert_eq!(&cache.peek(key).expect("a committed key"), bytes, "key {key}");
		}

		assert_eq!(cache.live_setters(), 0);

		drop(cache);
		assert_eq!(phys::fast_bytes_signed(), 0, "everything allocated was refunded");
	});
}

// ---------------------------------------------------------------------------
// the deadline

/// A reserve that waits for room in a full fast tier ends at its deadline --
/// not before, and not at the gate's own watchdog (ten seconds away) -- with
/// `FastTierStalled`, the bytes lane's error, and leaves nothing behind: the
/// lane, the reservation, P and the object count are as they were. It is never
/// built slow or admitted over the budget, whatever `on_stall` says; a
/// deadline that has already passed waits not at all; and one that is far
/// enough off is met by the room demotions free. Red with the deadline unread
/// (`nodeadline`: the reserve waits out the minute) or read only at the
/// watchdog's stall.
#[test]
fn a_reserve_past_its_deadline_fails_while_the_tier_is_held_full() {
	alone("a_reserve_past_its_deadline_fails_while_the_tier_is_held_full", || {
		let _flush = test_hooks::no_flush();
		let _m = test_hooks::override_m(M0);
		let original = gated(Duration::from_secs(10), OnStall::Error);
		let cache = cache(PaperPolicy::LruCompactHybrid, original);
		let gate = cache.status.gate();

		let pause = test_hooks::pause_consumers();
		let next = fill(&cache, 0);
		let (p0, count) = (p(), cache.status.live_num_objects());

		// (a) A deadline 300 ms off, with the tier held full.
		let started = Instant::now();
		let result = cache.reserve_set(next, LEN, None, started + Duration::from_millis(300));
		let took = started.elapsed();

		assert!(matches!(result, Err(CacheError::FastTierStalled)), "(a): the bytes lane's error: {:?}", result.err());
		assert!(took >= Duration::from_millis(300), "(a): gave up after {took:?}, before its deadline");
		assert!(took < Duration::from_millis(300) + Duration::from_secs(2), "(a): gave up after {took:?}");

		let stats = cache.hybrid_stats();

		assert_eq!(stats.gate_stall_errors, 0, "(a): the deadline's error is not the watchdog's");
		assert_eq!(waiters(&cache), 0, "(a): the lane is empty");
		assert_eq!((gate.reserved(), p(), cache.status.live_num_objects()), (0, p0, count), "(a): nothing left behind");

		// (b) A deadline that has already passed.
		let past = Instant::now();
		thread::sleep(Duration::from_millis(2));

		let started = Instant::now();
		let result = cache.reserve_set(next, LEN, None, past);

		assert!(matches!(result, Err(CacheError::FastTierStalled)), "(b): {:?}", result.err());
		assert!(started.elapsed() < Duration::from_millis(200), "(b): waited {:?} though out of time", started.elapsed());

		// (c) Not built slow, not admitted over the budget, under either `on_stall`.
		for on_stall in [OnStall::Divert, OnStall::AdmitOver] {
			let mut config = cache.gate_config();
			config.on_stall = on_stall;
			cache.set_gate_config(config).expect("a valid configuration");

			let started = Instant::now();
			let result = cache.reserve_set(next, LEN, None, started + Duration::from_millis(100));

			assert!(matches!(result, Err(CacheError::FastTierStalled)), "(c) {on_stall:?}: {:?}", result.err());
			assert!(started.elapsed() >= Duration::from_millis(100), "(c) {on_stall:?}: out of time early");

			let stats = cache.hybrid_stats();

			assert_eq!((stats.divert_sets, stats.admit_over_sets), (0, 0), "(c) {on_stall:?}");
			assert_eq!((gate.reserved(), p()), (0, p0), "(c) {on_stall:?}");
		}

		// (e) `stall_window` 0 never waits and acts at once -- here, builds slow --
		// unless the deadline has passed: then the error, not the action.
		let mut config = original;
		config.stall_window = Duration::ZERO;
		config.on_stall = OnStall::Divert;
		cache.set_gate_config(config).expect("a valid configuration");

		let result = cache.reserve_set(next, LEN, None, past);

		assert!(matches!(result, Err(CacheError::FastTierStalled)), "(e): out of time: {:?}", result.err());

		let diverted = cache.reserve_set(next, LEN, None, soon()).expect("(e): diverted at once, time left");

		assert_eq!(diverted.tier(), Slow, "(e)");
		drop(diverted);
		cache.set_gate_config(original).expect("the configuration back");

		// (d) A deadline far enough off is met by the room the consumers free.
		let resumed = Instant::now();

		thread::scope(|scope| {
			let reserve = scope.spawn(|| cache.reserve_set(next, LEN, None, resumed + Duration::from_secs(20)).map(|permit| {
				let mut pending = permit.fill();

				pending.write(&value(next));
				pending.commit()
			}));

			wait_for("the reserve to wait", Duration::from_secs(5), || waiters(&cache) == 1);
			drop(pause);

			reserve.join().expect("the setter").expect("admitted once demotions freed room").expect("committed");
		});

		assert_eq!(cache.peek(&next).expect("the committed key"), value(next));
	});
}

/// The metadata lane's deadline: a new key at the ceiling that waits for the
/// worker to evict (`EvictToFit`) ends at its deadline with `MetadataOverflow`,
/// the error that lane gives when its own watchdog gives up -- here with the
/// worker held and a window of ten seconds -- and one whose deadline has passed
/// asks the worker for nothing. Red with the deadline unread in `await_room`
/// (`noroomdeadline`).
#[test]
fn a_reserve_past_its_deadline_in_the_metadata_lane_fails_with_metadata_overflow() {
	alone("a_reserve_past_its_deadline_in_the_metadata_lane_fails_with_metadata_overflow", || {
		let _overheads = crate::object::overhead::test_overheads::set(64, 100);

		let mut config = evict_to_fit();
		config.stall_window = Duration::from_secs(10);

		// The ceiling is 64 keys (a 4 KiB tier, 64 B each).
		let cache = cache_with(PaperPolicy::LruCompactHybrid, 4_096, config);

		for key in 0..64u64 {
			cache.set(key, &[key as u8; 100], None).expect("under the ceiling");
		}

		wait_for("the worker taking the sets", Duration::from_secs(10), || {
			let stats = cache.hybrid_stats();

			stats.fast_objects + stats.slow_objects == 64
		});

		let hold = test_hooks::hold_workers();

		// (a) The worker cannot answer: the deadline ends the wait.
		let started = Instant::now();
		let result = cache.reserve_set(64, 100, None, started + Duration::from_millis(300));
		let took = started.elapsed();

		assert!(matches!(result, Err(CacheError::MetadataOverflow)), "(a): {:?}", result.err());
		assert!(took >= Duration::from_millis(300) && took < Duration::from_millis(300) + Duration::from_secs(2), "(a): gave up after {took:?}");
		assert_eq!(cache.hybrid_stats().make_room_requests, 1, "(a): the head asked the worker once");

		// (b) Out of time before it begins: no request is made.
		let past = Instant::now();
		thread::sleep(Duration::from_millis(2));

		let started = Instant::now();
		let result = cache.reserve_set(65, 100, None, past);

		assert!(matches!(result, Err(CacheError::MetadataOverflow)), "(b): {:?}", result.err());
		assert!(started.elapsed() < Duration::from_millis(200), "(b): waited {:?}", started.elapsed());
		assert_eq!(cache.hybrid_stats().make_room_requests, 1, "(b): the worker was asked for nothing");

		drop(hold);

		// An overwrite adds no metadata: it is admitted whatever the deadline.
		let permit = cache.reserve_set(30, 100, None, Instant::now()).expect("an overwrite never waits");

		assert_eq!(permit.len(), 100);
	});
}

// ---------------------------------------------------------------------------
// a TTL learned after the value

/// The wire's order is key, value, TTL: a server reserves without the TTL and
/// gives it with `set_ttl` once it has read it. The set then carries it -- it
/// expires; the DRAM charged for a TTL'd object is the object's; one reserved
/// with a TTL and committed without loses it -- and what admission computed
/// without it is rechecked: a value that fit without a TTL and does not fit
/// with the TTL's 64 bytes is refused at its commit with `ExceedingValueSize`,
/// its allocation refunded.
#[test]
fn a_ttl_given_after_the_value_is_the_ttl_of_the_set() {
	alone("a_ttl_given_after_the_value_is_the_ttl_of_the_set", || {
		let mut config = GateConfig::default();
		config.metadata_model = MetadataModel::PerObject;
		let cache = Cache::new_with_gate(256 << 20, CacheTierSize::Bytes(TIER), PaperPolicy::LruCompactHybrid, config).expect("a tiered cache");
		let policy = PaperPolicy::LruCompactHybrid;
		let used = || cache.status.used_size(&policy);

		let via_permit = |key: u64, reserved: Option<u32>, given: Option<Option<u32>>| {
			let mut pending = cache.reserve_set(key, 1_000, reserved, soon()).expect("a permit").fill();

			pending.write(&[key as u8; 1_000]);

			if let Some(ttl) = given {
				pending.set_ttl(ttl);
			}

			pending.commit()
		};

		// What a plain `set` charges, with and without a TTL.
		let before = used();
		cache.set(1, &[1u8; 1_000], None).expect("a set");
		let plain = used() - before;

		let before = used();
		cache.set(2, &[2u8; 1_000], Some(3_600)).expect("a set with a TTL");
		let with_ttl = used() - before;

		assert!(with_ttl > plain, "a TTL'd object costs more ({with_ttl} against {plain})");

		// A TTL given late is charged as one reserved with it.
		let before = used();
		via_permit(3, None, Some(Some(3_600))).expect("reserved without, given a TTL");
		assert_eq!(used() - before, with_ttl, "a late TTL is the TTL'd object's charge");

		let before = used();
		via_permit(4, Some(3_600), Some(None)).expect("reserved with, committed without");
		assert_eq!(used() - before, plain, "a TTL taken away is the plain charge");

		let before = used();
		via_permit(5, Some(3_600), None).expect("reserved with, committed with");
		assert_eq!(used() - before, with_ttl, "the reserved TTL stands when none is given");

		// It expires: a one-second TTL given after the value.
		via_permit(6, None, Some(Some(1))).expect("a one-second TTL");
		assert!(cache.get(&6).is_ok(), "alive at once");

		thread::sleep(Duration::from_millis(2_200));

		assert!(cache.get(&6).is_err(), "gone after its TTL");
		assert!(cache.get(&3).is_ok(), "the one-hour TTL is not");

		// The recheck. The workers are held while the cap is moved, so that no
		// pass evicts what the cache holds for a cap this small.
		let status = &cache.status;
		let manager = &cache.overhead_manager;
		let normal = status.max_size();

		let plain_base = manager.base_size_for(&7u64, 1_000, None).expect("a size");
		let ttl_base = manager.base_size_for(&7u64, 1_000, Some(3_600)).expect("a size");

		assert!(ttl_base > plain_base);

		let mut pending = cache.reserve_set(7, 1_000, None, soon()).expect("a permit").fill();

		pending.write(&[7u8; 1_000]);

		let hold = test_hooks::hold_workers();

		// The smallest cap at which the plain value fits: the TTL's 64 bytes tip it.
		let cap = (1..).find(|cap| {
			status.set_max_size(*cap);
			!status.exceeds_eviction_threshold(plain_base)
		}).expect("a cap the plain value fits");

		assert!(status.exceeds_eviction_threshold(ttl_base), "at cap {cap} the TTL's {} bytes tip it", ttl_base - plain_base);

		let (p0, count) = (p(), cache.status.live_num_objects());

		pending.set_ttl(Some(3_600));

		let refused = pending.commit();

		status.set_max_size(normal);

		assert_eq!(refused, Err(CacheError::ExceedingValueSize), "refused at its commit");
		assert_eq!(p(), p0 - phys::value_charge::<u64>(1_000), "its allocation was refunded");
		assert_eq!(cache.status.live_num_objects(), count, "nothing was inserted");
		assert!(cache.peek(&7).is_err());

		// A TTL reserved with is refused at the reserve, as a set is.
		status.set_max_size(cap);
		let at_reserve = cache.reserve_set(7, 1_000, Some(3_600), soon()).map(|_| ());
		status.set_max_size(normal);

		assert_eq!(at_reserve, Err(CacheError::ExceedingValueSize), "the same value, reserved with its TTL");

		// Without the TTL the value fits at that cap.
		let mut pending = cache.reserve_set(7, 1_000, None, soon()).expect("a permit").fill();

		pending.write(&[7u8; 1_000]);
		status.set_max_size(cap);

		let committed = pending.commit();

		status.set_max_size(normal);
		drop(hold);

		assert_eq!(committed, Ok(()));
		assert!(cache.peek(&7).is_ok());
	});
}

// ---------------------------------------------------------------------------
// the concurrency hint

/// The setters a server registers widen the byte gate's near band, as the
/// configuration's own hint does: the published `N` is `B - max(near_frac x
/// eff, (concurrency_hint + live setters) x value_hint)`, clamped into `[S +
/// 1, B]` (`gate::bands`), republished at the worker's next pass -- which a
/// registration and a release both ask for. With none registered it is the
/// band the configuration gives; with `value_hint` 0 -- the default -- a
/// registered setter widens nothing; and the count cannot go below zero. Red
/// with the count not read where the levels are published (`nosetters`).
#[test]
fn registered_setters_widen_the_near_band_and_a_release_narrows_it() {
	alone("registered_setters_widen_the_near_band_and_a_release_narrows_it", || {
		let _flush = test_hooks::no_flush();
		let _m = test_hooks::override_m(M0);

		let mut config = gated(Duration::from_secs(5), OnStall::Error);
		config.value_hint = 4 << 10;

		let cache = cache(PaperPolicy::LruCompactHybrid, config);
		let eff = cache.status.gate().eff();
		let n_for = |hint: u32| gate::bands(eff, &GateConfig { concurrency_hint: hint, ..config }).n;

		assert_eq!(cache.live_setters(), 0);
		assert_eq!(cache.hybrid_stats().band_n, n_for(0), "no setter: the configuration's band");

		let expect = |hint: u32, what: &str| {
			wait_for(what, Duration::from_secs(10), || cache.hybrid_stats().band_n == n_for(hint));
		};

		let first = cache.register_setter();
		let second = cache.register_setter();
		let third = cache.register_setter();

		assert_eq!(cache.live_setters(), 3);
		expect(3, "N to widen for three setters");
		assert!(n_for(3) < n_for(0), "three 4 KiB setters are wider than the 1% band: {} against {}", n_for(3), n_for(0));

		drop(second);
		assert_eq!(cache.live_setters(), 2);
		expect(2, "N to narrow for two setters");

		// The configuration's own hint counts with them.
		let mut with_hint = config;
		with_hint.concurrency_hint = 4;
		cache.set_gate_config(with_hint).expect("a valid configuration");

		let n_with = |hint: u32| gate::bands(eff, &GateConfig { concurrency_hint: hint, ..with_hint }).n;

		wait_for("N to count the configured hint with the setters", Duration::from_secs(10), || cache.hybrid_stats().band_n == n_with(6));

		// So many that the band clamps above the settle target.
		let crowd: Vec<_> = (0..1_000).map(|_| cache.register_setter()).collect();

		wait_for("N to clamp", Duration::from_secs(10), || cache.hybrid_stats().band_n == n_with(1_006));
		assert_eq!(cache.hybrid_stats().band_n, cache.hybrid_stats().band_s + 1, "clamped just above S");

		drop(crowd);
		drop(first);
		drop(third);
		assert_eq!(cache.live_setters(), 0);

		wait_for("N to return", Duration::from_secs(10), || cache.hybrid_stats().band_n == n_with(4));

		cache.set_gate_config(config).expect("the configuration back");
		expect(0, "N to return to the configuration's band");

		// The count saturates: a release with nothing registered stays at zero.
		cache.status.gate().remove_setter();
		assert_eq!(cache.live_setters(), 0);

		// With no value hint a setter widens nothing, whatever the count.
		let mut bare = config;
		bare.value_hint = 0;
		cache.set_gate_config(bare).expect("a valid configuration");

		let n0 = gate::bands(eff, &bare).n;

		wait_for("N for no value hint", Duration::from_secs(10), || cache.hybrid_stats().band_n == n0);

		let _crowd: Vec<_> = (0..50).map(|_| cache.register_setter()).collect();
		let passes = cache.status.gate().passes();

		wait_for("two passes", Duration::from_secs(10), || cache.status.gate().passes() >= passes + 2);
		assert_eq!(cache.hybrid_stats().band_n, n0, "fifty setters widen nothing without a value hint");
	});
}

/// The count is nothing but a number the levels are computed from: with no
/// setter registered `bands_for` is `bands`, exactly, and the gate that
/// counts them adds each to the configuration's hint, saturating.
#[test]
fn with_no_setter_the_levels_are_the_configurations() {
	let gate = crate::gate::Gate::default();
	let mut config = GateConfig::default();
	config.mode = GateMode::Block;
	config.value_hint = 4 << 10;

	for hint in [0u32, 1, 8, 1_000] {
		config.concurrency_hint = hint;

		for eff in [0u64, 1 << 20, 960 << 10, 64 << 30] {
			assert_eq!(gate.bands_for(eff, &config), gate::bands(eff, &config), "hint {hint}, eff {eff}");
		}
	}

	config.concurrency_hint = u32::MAX;
	gate.add_setter();

	assert_eq!(gate.bands_for(1 << 20, &config), gate::bands(1 << 20, &config), "the sum saturates");
	assert_eq!(gate.setters(), 1);

	gate.remove_setter();
	gate.remove_setter();

	assert_eq!(gate.setters(), 0, "never below zero");
}

// The permit and the pending set are for the thread that serves a request, and
// move between threads (a server hands a connection around): `Send` for a
// sendable key, checked at compile time.
#[allow(dead_code)]
fn the_permit_types_are_send() {
	fn is_send<T: Send>() {}

	is_send::<crate::SetPermit<'static, u64, std::hash::RandomState>>();
	is_send::<PendingSet<'static, String, std::hash::RandomState>>();
	is_send::<crate::SetterGuard>();
}
