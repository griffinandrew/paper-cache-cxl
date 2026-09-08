/*
 * Copyright (c) Kia Shakiba
 *
 * This source code is licensed under the GNU AGPLv3 license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! The concurrency gate for v5's value lifetime -- now the refcount's, and it
//! matters MORE than it did under the epoch pin, not less.
//!
//! Every other test in the tree checks a value's bytes from one thread, and so
//! cannot see the defect this design exists to prevent: a reader that has
//! dropped the shard guard and is part-way through copying a value while a
//! writer unpublishes it and a migration swaps it. That window is the whole
//! reason `Object::snapshot` hands back an OWNED
//! [`crate::value::TieredValue`] -- a strong reference -- rather than a borrow
//! of the map row, and it is invisible to a single-threaded test.
//!
//! The mechanism changed under this harness; the window did not. It used to be
//! held open by a `crossbeam_epoch` pin and closed by `defer_free`; it is now
//! held open by a refcount and closed by the last handle's drop. Both designs
//! stand or fall on the same question -- can a reader that let go of the shard
//! lock still be reading bytes some other thread has freed -- and this is the
//! only test in the tree that asks it concurrently. What a refcount buys is
//! that the answer is now structural rather than disciplinary: the proof is the
//! handle itself, not a rule every call site has to remember. What it does not
//! buy is exemption from being checked, because "the read path takes a strong
//! reference" is a claim about the read path, not about `Arc`.
//!
//! So this harness runs the window continuously for ten seconds:
//!
//!   * 8 reader threads spin on `get_into` over 1024 keys,
//!   * one thread flaps the fast tier between big and small, which demotes en
//!     masse and lets the readers' own hits promote back, and
//!   * one thread re-sets every key with a rotating, DISTINCT marker byte.
//!
//! Every value is `VALUE_LEN` copies of ONE marker byte, so any read whose
//! buffer is not uniform saw two different values -- i.e. the bytes moved or
//! were recycled under it. That is the torn read. A buffer that is uniform but
//! whose byte is not a marker at all (zeroed pages, say) is caught too.
//!
//! The second half of the gate is the opposite failure: a value that is retired
//! and never freed. That half is where the bookkeeping changed, and it changed
//! by getting SIMPLER and STRICTER at once:
//!
//!   * There is no deferral gap left to bound. Under epoch reclamation a
//!     retirement and its free were two events with an unbounded-looking
//!     interval between them, so the gate could only assert that the pair of
//!     counters had converged after a flush and that whatever remained fitted
//!     inside a per-thread bag (62 objects). A refcount frees on the thread
//!     that drops the last handle, synchronously, so once the load stops and
//!     the threads have joined there is no residue to allow for at all --
//!     `crate::value::VALUE_FREES` is the one counter, and the bound is zero.
//!   * What replaces "deferred == ran" is a RETIREMENT LEDGER, which the old
//!     shape could not have asserted. Every successful `set` displaces exactly
//!     one published value, and every migration that `Object::set_data`
//!     actually applied displaces exactly one more; both are unreachable
//!     afterwards, so both must have been freed by the time the round comes to
//!     rest. See [`check_round`] for why this is stated as `>=` and what
//!     accounts for the excess.
//!   * The `stats.allocated` criterion is kept exactly as it was. It is the
//!     only thing here that would catch a leak with no counter attached to it
//!     -- a reference cycle, or a handle parked somewhere that never drops.
//!
//! ## Why `#[ignore]`
//!
//! `stats.allocated` and `VALUE_FREES` are PROCESS-WIDE. Cargo runs the lib
//! tests concurrently in one process, so any other test running alongside this
//! one would allocate and free into the same numbers. This harness therefore
//! owns its process, the same rule every `measure_*` harness in the tree
//! follows:
//!
//! ```text
//! cargo +nightly test --lib --features lru_compact_hybrid_cache -- \
//!     --ignored --exact --test-threads=1 --nocapture \
//!     value_stress::concurrent_readers_never_see_a_torn_value_and_nothing_leaks
//! ```
//!
//! ## A pre-existing panic this harness surfaces
//!
//! In a DEBUG build the policy worker's migration consumer panics with
//! `attempt to add with overflow` inside `PendingOnDrop` (`worker/policy/
//! mod.rs`) under this load. `MigrationQueue::push` calls `record_pending`
//! AFTER `send` succeeds, so a consumer can receive the item and run the
//! decrement before the producer's increment, wrapping the counter to
//! `u64::MAX`. It is instrumentation only, it is not v5's, and it reproduced
//! unchanged on the base commit 9480852 with this same harness. Release builds
//! do not check the overflow, so the gate run uses `--release`; a debug run
//! loses its migration consumer part-way and its memory figures are not
//! comparable.
//!
//! ## Proving it can fail
//!
//! A stress test that has never failed proves nothing, and "the refcount makes
//! tearing impossible" is a reason to keep the test rather than to retire it:
//! it is the claim under test, not a premise. Two one-line mutations, each
//! outside `src/value.rs`, break a different half of the gate.
//!
//! TEARING. Make the read path's snapshot stop being a strong reference --
//! `Object::snapshot` (src/object/mod.rs) returning a forged handle rather
//! than a clone:
//!
//! ```ignore
//! pub fn snapshot(&self) -> TieredValue<K> {
//!     // MUTATION: a bitwise copy of the handle, with NO increment.
//!     unsafe { std::ptr::read(&self.value) }
//! }
//! ```
//!
//! This is the exact analogue of the mutation the epoch version was verified
//! with (`Object::drop` freeing inline instead of deferring), and it fails the
//! same way: the snapshot's drop decrements a count it never incremented, so
//! the header and its bytes go back to the allocator while the map still
//! publishes them and while a reader is still copying them. `SCRATCH_BUFFERS`
//! then dirties the block, and round 3 reports `torn` and `alien` reads.
//!
//! LEAKING. Make a retirement not retire -- in `migration_queue::
//! apply_migration` (src/worker/policy/mod.rs), where the superseded value is
//! unpublished under the write guard:
//!
//! ```ignore
//! // MUTATION: was `drop(superseded)`.
//! std::mem::forget(superseded);
//! ```
//!
//! The count never reaches zero, so `VALUE_FREES` stops tracking the
//! migrations the cache reports having applied -- the retirement ledger below
//! fails first, on the round in which it happens, naming the shortfall -- and
//! `stats.allocated` climbs by a round's worth of leaked values, which the 1%
//! criterion fails on independently.
//!
//! Revert either before trusting a green run.

use std::sync::{
	Arc,
	atomic::{AtomicBool, AtomicU64, Ordering},
};

use std::time::{Duration, Instant};

use crate::{CacheTierSize, PaperCache, PaperPolicy, Tier, TieredBuffer};

/// Bytes jemalloc has handed to the application, from `stats.allocated`.
///
/// A deliberate copy of
/// `worker::policy::policy_stack::measure_overhead::allocated_bytes`: that
/// module sits under a private `mod worker`, so it is not reachable from a
/// sibling of `worker` however public the function itself is. The two must stay
/// in step -- in particular the `epoch` write, without which jemalloc returns
/// the statistics it cached at the previous epoch rather than current ones.
fn allocated_bytes() -> u64 {
	unsafe {
		let mut e: u64 = 1;
		let mut elen = core::mem::size_of::<u64>();
		tikv_jemalloc_sys::mallctl(
			c"epoch".as_ptr(),
			&mut e as *mut u64 as *mut core::ffi::c_void,
			&mut elen as *mut usize,
			&mut e as *mut u64 as *mut core::ffi::c_void,
			core::mem::size_of::<u64>(),
		);
		let mut allocated: usize = 0;
		let mut len = core::mem::size_of::<usize>();
		let rc = tikv_jemalloc_sys::mallctl(
			c"stats.allocated".as_ptr(),
			&mut allocated as *mut usize as *mut core::ffi::c_void,
			&mut len as *mut usize,
			core::ptr::null_mut(),
			0,
		);
		assert_eq!(
			rc, 0,
			"stats.allocated unavailable -- tikv-jemalloc-sys needs features = [\"stats\"]"
		);
		allocated as u64
	}
}

/// Keys touched by every thread. The gate's figure.
const KEYS: u32 = 1024;

/// 8 KiB: an exact jemalloc size class (no rounding to reason about), and long
/// enough that a reader's copy spans a wide enough window for a mid-copy free
/// to actually be observable rather than theoretically possible.
const VALUE_LEN: usize = 8192;

const READERS: usize = 8;

/// Microseconds a reader pauses after each `get_into`, and the overwriter after
/// each `set`.
///
/// The harness is PACED, not free-running, and the reason is not politeness.
/// Every `get` and every `set` broadcasts a `WorkerEvent` down an UNBOUNDED
/// channel to the single policy worker. Eight readers spinning flat out
/// enqueue faster than that worker can consume -- ~22M events in ten seconds --
/// and the resulting backlog is over a gigabyte that does not drain when the
/// load stops. That backlog is real (see the gate report), but it is a
/// worker-throughput property, not a value-reclamation one, and left in it
/// swamps the `stats.allocated` comparison by four orders of magnitude.
///
/// So the load is held at a rate the worker services, and the memory
/// comparison then measures what it is meant to. Overridable for sweeps;
/// `STRESS_READ_PACE_US=0` reproduces the free-running backlog.
fn pace_us(var: &str, default: u64) -> u64 {
	std::env::var(var).ok().and_then(|v| v.parse().ok()).unwrap_or(default)
}

fn pace(micros: u64) {
	if micros > 0 {
		std::thread::sleep(Duration::from_micros(micros));
	}
}

const RUN_FOR: Duration = Duration::from_secs(10);

/// The marker alphabet. Values are `VALUE_LEN` copies of one of these, so a
/// non-uniform buffer means two values, and a uniform buffer of some OTHER byte
/// means memory that was never a value of ours.
const MARKERS: [u8; 8] = [0x11, 0x33, 0x55, 0x77, 0x99, 0xbb, 0xdd, 0xf1];

/// Deliberately NOT a marker. The overwriter dirties freshly freed blocks with
/// this, so a value freed under a reader gets REWRITTEN rather than left
/// intact -- see `SCRATCH_BUFFERS`.
const SCRATCH_BYTE: u8 = 0xab;

/// How many same-sized buffers the overwriter allocates and dirties right after
/// each `set`.
///
/// Without this the harness cannot see a use-after-free at all, and that is a
/// measured statement rather than a worry: with the epoch design's `Object::
/// drop` deliberately changed to free inline, a full run reported `torn=0`.
/// The same holds for the refcount's equivalent mutation (a `snapshot` that
/// forges a handle instead of cloning one -- see the module doc), because the
/// failure it produces is the same one: a live block freed early. The reason
/// is timing.
/// An 8 KiB copy takes well under a microsecond; the freed block goes to the
/// freeing thread's tcache and is not handed out again until that thread's next
/// allocation, ~20 us later. By then every reader has finished, so the stale
/// bytes it copied were still intact and still a valid marker -- a real
/// use-after-free that no assertion could see.
///
/// Dirtying the block immediately closes that gap. It is not artificial: any
/// server doing other work in the same process recycles a freed block within
/// microseconds. All it does is make a defect that is already there visible.
const SCRATCH_BUFFERS: usize = 4;

/// Comfortably larger than `KEYS * VALUE_LEN` (8 MiB) plus metadata, so nothing
/// is ever evicted and the population at the end is the population at the
/// start -- which is what makes the `stats.allocated` comparison meaningful.
const MAX_SIZE: u64 = 256 * 1024 * 1024;

/// Holds every key in the fast tier.
const FAST_BIG: u64 = 128 * 1024 * 1024;

/// Holds a few dozen values, so raising and lowering between this and
/// `FAST_BIG` migrates most of the population each way.
const FAST_SMALL: u64 = 256 * 1024;

/// What the value allocator has actually freed, and what the cache says it
/// retired, sampled so that the difference is meaningful.
///
/// The ORDER of the two reads is the whole content of this function, and it is
/// the reason it exists rather than two inline loads at each end of a round.
/// The invariant [`check_round`] asserts is one-directional -- frees must not
/// fall SHORT of retirements -- so each sample is biased in the direction that
/// cannot manufacture a false failure:
///
///   * At the START of a window, read `VALUE_FREES` FIRST and the migration
///     counters second. A migration completing in between contributes its free
///     to the window's delta and its count to the baseline, so the delta gains
///     a free and no retirement.
///   * At the END, read the migration counters FIRST and `VALUE_FREES` second,
///     for the mirror-image reason.
///
/// Either way is only sound because `apply_migration` drops the superseded
/// value BEFORE it increments its counter, never after: at any instant, frees
/// already performed are ahead of migrations already counted, never behind. A
/// sampler that read them the other way round would see a straggler as a
/// shortfall, and the gate would fail on timing rather than on a leak.
fn ledger(cache: &PaperCache<u32, TieredBuffer>, at: Sample) -> (u64, u64) {
	let read_frees = || crate::value::VALUE_FREES.load(Ordering::Relaxed);

	let read_migrations = || {
		let stats = cache.hybrid_stats();
		stats.promotions + stats.demotions
	};

	match at {
		Sample::Start => {
			let frees = read_frees();

			(frees, read_migrations())
		},

		Sample::End => {
			let migrations = read_migrations();

			(read_frees(), migrations)
		},
	}
}

/// Which end of a round a [`ledger`] reading is being taken at. A named pair
/// rather than a `bool`, because the two differ only in the order of two
/// loads and a caller passing the wrong one would be invisible at the call
/// site.
#[derive(Clone, Copy)]
enum Sample {
	Start,
	End,
}

/// A uniform buffer of a known marker.
fn value(marker: u8) -> Vec<u8> {
	vec![marker; VALUE_LEN]
}

/// How many keys are in each tier.
///
/// The two tiers draw from DIFFERENT jemalloc arenas (`numa_alloc`), and a
/// value that ends in the slow tier is not the same number of allocated bytes
/// as the same value in the fast tier. So the memory comparison is only valid
/// between two points with the SAME census -- otherwise it measures where the
/// population happens to be sitting rather than whether anything leaked.
fn census(cache: &PaperCache<u32, TieredBuffer>) -> (u32, u32) {
	let mut fast = 0;
	let mut slow = 0;

	for key in 0..KEYS {
		match cache.tier_of(&key) {
			Some(Tier::Fast) => fast += 1,
			Some(Tier::Slow) => slow += 1,
			None => {},
		}
	}

	(fast, slow)
}

/// Drags the whole population back into the fast tier, which is where the
/// baseline was taken. A slow-tier value only promotes when it is HIT, so this
/// hits every key until the census says they all came home.
fn refill_fast_tier(cache: &PaperCache<u32, TieredBuffer>, budget: Duration) -> bool {
	wait_until(budget, || {
		for key in 0..KEYS {
			let _ = cache.get(&key);
		}

		census(cache) == (KEYS, 0)
	})
}

/// Counts shared by the threads. Nothing here fails a thread on the spot: a
/// panic inside a spawned thread would leave the others spinning and the
/// harness reporting a join error instead of the actual defect, so every
/// observation is tallied and asserted on the main thread after the join.
#[derive(Default)]
struct Tally {
	reads: AtomicU64,
	misses: AtomicU64,
	/// Buffers holding more than one distinct byte: the torn read.
	torn: AtomicU64,
	/// Uniform buffers whose byte was never written by this test.
	alien: AtomicU64,
	/// Buffers of the wrong length.
	wrong_len: AtomicU64,
	sets: AtomicU64,
	flaps: AtomicU64,
	demotions_seen: AtomicU64,
}

/// What one round of the load saw. A plain snapshot, so the round can hand its
/// counters back after its threads are gone.
struct Seen {
	reads: u64,
	misses: u64,
	torn: u64,
	alien: u64,
	wrong_len: u64,
	sets: u64,
	flaps: u64,
	demotion_waves: u64,
	census: (u32, u32),
	quiesced: bool,
	/// `stats.allocated` once the round had come to rest.
	allocated: u64,

	/// Value byte-allocations returned to an allocator during the round --
	/// `crate::value::VALUE_FREES`, differenced across it.
	frees: u64,

	/// Tier migrations the cache reports as PHYSICALLY APPLIED during the
	/// round (promotions + demotions), differenced across it.
	///
	/// Each one ran `Object::set_data` and therefore displaced exactly one
	/// published value -- the counters are incremented if and only if the swap
	/// happened, which is what makes them usable as a retirement count rather
	/// than as an intent count. See `apply_migration_batches`.
	migrations: u64,
}

/// One pass of the load: spawn, run, stop, join, drain the backlog, put the
/// population back in the tier the baseline found it in, quiesce.
fn run_round(
	cache: &Arc<PaperCache<u32, TieredBuffer>>,
	round: u32,
	read_pace: u64,
	set_pace: u64,
	scratch: bool,
) -> Seen {
	println!("ROUND {round} pacing read={read_pace}us set={set_pace}us");

	let (frees_before, migrations_before) = ledger(cache, Sample::Start);

	let tally = Arc::new(Tally::default());
	let stop = Arc::new(AtomicBool::new(false));
	let mut threads = Vec::new();

	// ── 8 readers ─────────────────────────────────────────────────────────
	for reader in 0..READERS {
		let cache = Arc::clone(cache);
		let tally = tally.clone();
		let stop = stop.clone();

		threads.push(std::thread::spawn(move || {
			let mut buf: Vec<u8> = Vec::with_capacity(VALUE_LEN);

			// Each reader starts at a different key and strides by a number
			// coprime with KEYS, so the readers do not march in lockstep
			// through the same key at the same moment.
			let mut key = (reader as u32) * 37;

			while !stop.load(Ordering::Relaxed) {
				key = key.wrapping_add(7) % KEYS;

				match cache.get_into(&key, &mut buf) {
					Ok(()) => {
						tally.reads.fetch_add(1, Ordering::Relaxed);

						if buf.len() != VALUE_LEN {
							tally.wrong_len.fetch_add(1, Ordering::Relaxed);
							continue;
						}

						let first = buf[0];

						// Torn: the buffer is not one value.
						if buf.iter().any(|b| *b != first) {
							tally.torn.fetch_add(1, Ordering::Relaxed);
						} else if !MARKERS.contains(&first) {
							// Uniform, but not any value this test wrote.
							tally.alien.fetch_add(1, Ordering::Relaxed);
						}
					},

					Err(_) => {
						tally.misses.fetch_add(1, Ordering::Relaxed);
					},
				}

				pace(read_pace);
			}
		}));
	}

	// ── the migration flapper: demote en masse, then let hits promote ─────
	{
		let cache = Arc::clone(cache);
		let tally = tally.clone();
		let stop = stop.clone();

		threads.push(std::thread::spawn(move || {
			let mut small = true;

			while !stop.load(Ordering::Relaxed) {
				let size = if small { FAST_SMALL } else { FAST_BIG };

				if cache.set_fast_tier_size(CacheTierSize::Bytes(size)).is_ok() {
					tally.flaps.fetch_add(1, Ordering::Relaxed);
				}

				small = !small;

				// Long enough for a wave of migrations to actually be applied,
				// short enough to get many waves into ten seconds.
				std::thread::sleep(Duration::from_millis(150));

				if cache.tier_of(&0u32) == Some(Tier::Slow) {
					tally.demotions_seen.fetch_add(1, Ordering::Relaxed);
				}
			}
		}));
	}

	// ── the overwriter: a fresh, distinct marker every pass ───────────────
	{
		let cache = Arc::clone(cache);
		let tally = tally.clone();
		let stop = stop.clone();

		threads.push(std::thread::spawn(move || {
			let mut i: u64 = 0;

			while !stop.load(Ordering::Relaxed) {
				let key = (i % u64::from(KEYS)) as u32;
				let marker = MARKERS[((i / u64::from(KEYS)) as usize) % MARKERS.len()];

				if cache.set(key, &value(marker), None).is_ok() {
					tally.sets.fetch_add(1, Ordering::Relaxed);
				}

				i += 1;

				// Hand the block that `set` just released straight back to the
				// allocator and dirty it. See `SCRATCH_BUFFERS`. Off in the
				// measured rounds: it is a detector for the tearing round, and
				// its churn only adds variance to a byte comparison.
				for _ in 0..if scratch { SCRATCH_BUFFERS } else { 0 } {
					let scratch = vec![SCRATCH_BYTE; VALUE_LEN];
					core::hint::black_box(&scratch);
				}

				pace(set_pace);
			}
		}));
	}

	// `STRESS_SECS` is a diagnostic knob, not a tuning one: a leak scales with
	// the length of the run and a steady-state high-water mark does not, so
	// the same harness at 5s / 10s / 20s tells the two apart.
	let run_for = match std::env::var("STRESS_SECS").ok().and_then(|v| v.parse().ok()) {
		Some(secs) => Duration::from_secs(secs),
		None => RUN_FOR,
	};

	std::thread::sleep(run_for);
	stop.store(true, Ordering::Relaxed);

	for thread in threads {
		thread.join().expect("no stress thread should panic");
	}

	// ── settle, then account ──────────────────────────────────────────────
	// Order matters here, and getting it wrong silently invalidates the whole
	// comparison. The event channel still holds the run's backlog, INCLUDING
	// the flapper's last `ResizeFastTier(FAST_SMALL)`. Refilling the fast tier
	// before that backlog drains just means the queued shrink lands afterwards
	// and demotes the population straight back out -- which is what an earlier
	// version of this harness did, and it read as a 4.8 MB "leak".
	//
	// So: drain first, then resize, then refill, then quiesce again.
	let (drained, _) = settle(cache, Duration::from_secs(300));

	cache
		.set_fast_tier_size(CacheTierSize::Bytes(FAST_BIG))
		.expect("restoring the fast tier should succeed");

	assert_eq!(
		cache.fast_tier_size(),
		FAST_BIG,
		"the fast tier budget should be back where the baseline had it"
	);

	assert!(
		refill_fast_tier(cache, Duration::from_secs(120)),
		"the whole population should return to the fast tier before the comparison"
	);

	let (rested, allocated) = settle(cache, Duration::from_secs(120));
	let quiesced = drained && rested;

	// After the join and after the settle, so every value this round retired
	// has had its last handle dropped: the readers that were holding snapshots
	// are gone, and the migration queue has drained.
	let (frees_after, migrations_after) = ledger(cache, Sample::End);

	let seen = Seen {
		reads: tally.reads.load(Ordering::Relaxed),
		misses: tally.misses.load(Ordering::Relaxed),
		torn: tally.torn.load(Ordering::Relaxed),
		alien: tally.alien.load(Ordering::Relaxed),
		wrong_len: tally.wrong_len.load(Ordering::Relaxed),
		sets: tally.sets.load(Ordering::Relaxed),
		flaps: tally.flaps.load(Ordering::Relaxed),
		demotion_waves: tally.demotions_seen.load(Ordering::Relaxed),
		census: census(cache),
		quiesced,
		allocated,
		frees: frees_after - frees_before,
		migrations: migrations_after - migrations_before,
	};

	println!(
		"ROUND {round} reads={} misses={} torn={} alien={} wrong_len={} sets={} flaps={} \
		 waves={} census={:?} quiesced={} allocated={} frees={} migrations={}",
		seen.reads,
		seen.misses,
		seen.torn,
		seen.alien,
		seen.wrong_len,
		seen.sets,
		seen.flaps,
		seen.demotion_waves,
		seen.census,
		seen.quiesced,
		seen.allocated,
		seen.frees,
		seen.migrations,
	);

	seen
}

/// Everything a round must have got right before its bytes mean anything.
fn check_round(seen: &Seen, round: u32, census_before: (u32, u32)) {
	// The point of the whole exercise.
	assert_eq!(seen.torn, 0, "round {round}: a read saw more than one value's bytes in one buffer");
	assert_eq!(seen.alien, 0, "round {round}: a read saw a uniform buffer this test never wrote");
	assert_eq!(seen.wrong_len, 0, "round {round}: a read returned a buffer of the wrong length");

	// Not vacuous: the readers really did hammer the window, the overwriter
	// really did free underneath them, and values really did move between
	// tiers while they read.
	assert!(seen.reads > 100_000, "round {round}: only {} reads", seen.reads);
	assert!(seen.sets > u64::from(KEYS), "round {round}: the overwriter did not lap the key space");
	assert!(seen.demotion_waves > 0, "round {round}: the migration path was never exercised");

	assert!(seen.quiesced, "round {round}: the cache never came to rest");

	// The two tiers draw from different arenas, so a byte comparison across a
	// census change measures where the population sits, not whether anything
	// leaked.
	assert_eq!(
		seen.census, census_before,
		"round {round}: the population is not where the baseline found it"
	);

	// THE RETIREMENT LEDGER, which is what "deferred == ran once settled" turned
	// into once there was no deferral to account for.
	//
	// Every successful `set` overwrote a key that was already populated -- the
	// census assertion just above is what underwrites that, since it says the
	// whole key space survived the round -- so each one displaced exactly one
	// published value. Every applied migration displaced exactly one more. None
	// of those values is reachable afterwards and no thread that could have
	// been holding a snapshot of one is still alive, so every one of them must
	// have been freed.
	//
	// `>=` rather than `==` because two further things legitimately free a
	// value without any counter here naming it, and both are real work rather
	// than slack in the assertion:
	//
	//   * a migration copy that lost the `ptr_eq` identity check. It was built
	//     and then never published, so it is freed by the thread that built it
	//     and counted as a migration by nobody -- that is `MIG_SUPERSEDED`,
	//     which lives under a private `mod worker` and is not readable here.
	//   * the last handle on a value some other path retired, e.g. one the
	//     settle loop's own reads had lifted out.
	//
	// The direction that matters is the one that is closed: a leak can only
	// make frees fall SHORT, and it does so proportionally to the load, so a
	// single forgotten handle per migration is thousands of frees missing here
	// long before it is a percent of `stats.allocated`.
	let retired = seen.sets + seen.migrations;

	assert!(
		seen.frees >= retired,
		"round {round}: {} values were freed but {retired} were retired ({} sets + {} \
		 applied migrations) -- {} retirements never reached their allocator, so a handle \
		 outlived the value it named",
		seen.frees,
		seen.sets,
		seen.migrations,
		retired - seen.frees,
	);
}

#[test]
#[ignore = "10s of stress plus a PMEM warm-up, and it reads process-wide jemalloc statistics, so it must own the process"]
fn concurrent_readers_never_see_a_torn_value_and_nothing_leaks() {
	let cache = Arc::new(
		PaperCache::<u32, TieredBuffer>::new(
			MAX_SIZE,
			CacheTierSize::Bytes(FAST_BIG),
			PaperPolicy::LruCompactHybrid,
		)
		.expect("cache should construct"),
	);

	// Populate, then force the one-time PMEM pool init/prewarm to happen NOW,
	// by demoting the whole population once and promoting it back. Paying that
	// cost before the baseline is taken keeps it out of the memory comparison,
	// and paying it before the run keeps a ~45s first-touch stall out of the
	// ten seconds of stress.
	for key in 0..KEYS {
		cache
			.set(key, &value(MARKERS[0]), None)
			.expect("initial set should succeed");
	}

	cache
		.set_fast_tier_size(CacheTierSize::Bytes(FAST_SMALL))
		.expect("shrinking the fast tier should succeed");

	let warm = wait_until(Duration::from_secs(120), || {
		cache.tier_of(&0u32) == Some(Tier::Slow)
	});
	assert!(warm, "the slow tier should have taken a value within 120s");

	cache
		.set_fast_tier_size(CacheTierSize::Bytes(FAST_BIG))
		.expect("restoring the fast tier should succeed");

	for key in 0..KEYS {
		let _ = cache.get(&key);
	}

	assert!(
		refill_fast_tier(&cache, Duration::from_secs(60)),
		"the whole population should be back in the fast tier before the baseline"
	);

	let (rested, allocated_before) = settle(&cache, Duration::from_secs(60));
	assert!(rested, "the cache should quiesce before the baseline is taken");

	// The baseline.
	let census_before = census(&cache);


	// ── rounds 1 and 2: PACED, and they answer the memory question ────────
	//
	// Paced because every `get` and `set` broadcasts a `WorkerEvent` down an
	// unbounded channel to the single policy worker; free-running readers
	// enqueue faster than it consumes and leave a gigabyte of backlog that has
	// nothing to do with values. At this pace the worker keeps up and the
	// figure means what it says.
	let read_pace = pace_us("STRESS_READ_PACE_US", 100);
	let set_pace = pace_us("STRESS_SET_PACE_US", 20);

	let first = run_round(&cache, 1, read_pace, set_pace, false);
	let allocated_after_first = first.allocated;

	// Identical load, so anything it ADDS is not a one-time high-water mark.
	let second = run_round(&cache, 2, read_pace, set_pace, false);
	let allocated_after_second = second.allocated;

	let first_delta = allocated_after_first as i64 - allocated_before as i64;
	let second_delta = allocated_after_second as i64 - allocated_after_first as i64;
	let budget = (allocated_after_first / 100) as i64;

	println!(
		"STRESS allocated base={allocated_before} after1={allocated_after_first} \
		 after2={allocated_after_second}"
	);
	println!(
		"STRESS one_time_delta={first_delta} leak_delta={second_delta} budget(1%)={budget}"
	);
	println!(
		"STRESS retired r1={} r2={} frees r1={} r2={}",
		first.sets + first.migrations,
		second.sets + second.migrations,
		first.frees,
		second.frees,
	);

	check_round(&first, 1, census_before);
	check_round(&second, 2, census_before);

	// THE LEAK CRITERION, and the reason there are two rounds rather than one.
	//
	// Round 1 raises `stats.allocated` by roughly 5 MB whatever the run length
	// -- the same figure at 5s, 10s and 20s, which is what `STRESS_SECS` is for
	// -- because the first pass through this load faults in per-thread caches,
	// shard tables and channel blocks that then stay. That is a one-time
	// high-water mark and not a leak, so it is PRINTED (`one_time_delta`) and
	// not asserted on; holding round 1 to 1% would be asserting that jemalloc's
	// steady state is reached before it has been reached.
	//
	// An identical second round adds essentially nothing on top: round 1's
	// garbage was reclaimed before round 2 ran, so the figure does not CLIMB.
	// Retained garbage would put another round's worth on it -- round 2 alone
	// retires over five gigabytes of values, so even a hundredth of a percent
	// of retention is far past this budget.
	//
	// One-directional on purpose. A leak can only push the figure up; a round
	// that comes to rest slightly BELOW the previous one has simply shed more
	// of the first round's high-water mark, which is not a leak and must not
	// be reported as one.
	assert!(
		second_delta <= budget,
		"a second identical round added {second_delta} B, past 1% ({budget} B) \
		 -- garbage from the first round was not reclaimed"
	);

	// ── round 3: FREE-RUNNING, and it answers the tearing question ────────
	//
	// A paced reader spends ~1% of its time inside a copy, so the odds that a
	// free lands mid-copy are ~1% of what they could be. Unpaced, the readers
	// copy back to back and the window is open essentially all the time. It
	// runs LAST because it leaves the worker backlog the paced rounds exist to
	// avoid, and nothing is measured in bytes after it.
	let torn_hunt = run_round(&cache, 3, 0, 0, true);

	assert_eq!(torn_hunt.torn, 0, "round 3: a read saw more than one value's bytes in one buffer");
	assert_eq!(torn_hunt.alien, 0, "round 3: a read saw a uniform buffer this test never wrote");
	assert_eq!(torn_hunt.wrong_len, 0, "round 3: a read returned a buffer of the wrong length");
	assert!(torn_hunt.reads > 1_000_000, "round 3: only {} reads", torn_hunt.reads);
	assert!(torn_hunt.sets > u64::from(KEYS), "round 3: the overwriter did not lap the key space");

	// The ledger holds here too, and is worth asserting even though this round's
	// bytes are not. It needs only that the load has stopped and its threads
	// have been joined -- not that the cache has come to rest -- because a
	// migration still sitting in the queue has not incremented a counter
	// either, so an undrained backlog can only make `retired` smaller. This is
	// the round with by far the most retirements, so it is the round where a
	// forgotten handle shows up largest.
	let retired = torn_hunt.sets + torn_hunt.migrations;

	assert!(
		torn_hunt.frees >= retired,
		"round 3: {} values were freed but {retired} were retired ({} sets + {} applied \
		 migrations)",
		torn_hunt.frees,
		torn_hunt.sets,
		torn_hunt.migrations,
	);
}

/// Polls `predicate` until it holds or the timeout expires. Bounded by
/// construction: it cannot hang.
fn wait_until(timeout: Duration, mut predicate: impl FnMut() -> bool) -> bool {
	let deadline = Instant::now() + timeout;

	loop {
		if predicate() {
			return true;
		}

		if Instant::now() > deadline {
			return false;
		}

		std::thread::sleep(Duration::from_millis(20));
	}
}

/// Runs the cache forward until it is QUIESCENT, or the budget expires.
///
/// Quiescent used to mean two separate things. The first of them is gone with
/// the epoch pin: there is no `flush` to call and no per-thread bag of
/// deferrals to push, because the last handle to drop frees on the spot. An
/// idle worker sitting on un-run garbage was a real hazard of that design and
/// is not a state this one has.
///
/// What is left is the second: `stats.allocated` has stopped moving. Eight
/// readers spin far faster
/// than the single policy worker consumes, so at the moment they stop there is
/// a large backlog of `WorkerEvent`s sitting in an unbounded channel. That
/// backlog is transient -- the worker drains it -- but it is hundreds of
/// megabytes while it lasts, and it is nothing to do with value reclamation.
/// Comparing against the baseline before it drains measures the backlog. So
/// this waits for the figure to hold still first.
///
/// Bounded by construction: it cannot hang. Returns whether it quiesced and
/// the lowest `stats.allocated` it saw while settling.
fn settle(cache: &PaperCache<u32, TieredBuffer>, budget: Duration) -> (bool, u64) {
	let started = Instant::now();
	let deadline = started + budget;
	let mut samples: Vec<u64> = Vec::new();
	let mut next_sample = Instant::now();

	loop {
		// A burst rather than a single get, so the worker gets a full pass in.
		// It still pokes the worker into an event-loop pass -- which is what
		// drains the migration queue -- even though there is no longer a bag of
		// deferrals for that pass to flush.
		for _ in 0..64 {
			let _ = cache.get(&0u32);
		}

		if Instant::now() >= next_sample {
			let allocated = allocated_bytes();
			let frees = crate::value::VALUE_FREES.load(Ordering::Relaxed);

			println!(
				"SETTLE t={:.0}s allocated={allocated} frees={frees}",
				started.elapsed().as_secs_f64(),
			);

			samples.push(allocated);
			next_sample = Instant::now() + Duration::from_secs(1);

			// Five one-second samples, the last three of them within 0.1% of
			// one another. The figure the caller gets is the MINIMUM of them,
			// not the last: a channel block still in flight or a thread's
			// tcache still warm shows up as an upward transient, and the low
			// point is the closest thing to "what is actually retained".
			if samples.len() >= 5 {
				let tail = &samples[samples.len() - 3..];
				let spread = tail.iter().max().unwrap() - tail.iter().min().unwrap();

				if spread <= tail[0] / 1000 {
					return (true, *samples.iter().min().unwrap_or(&allocated));
				}
			}
		}

		if Instant::now() > deadline {
			return (false, samples.iter().min().copied().unwrap_or_else(allocated_bytes));
		}

		std::thread::sleep(Duration::from_millis(20));
	}
}
