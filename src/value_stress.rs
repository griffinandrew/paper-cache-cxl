/*
 * Copyright (c) Kia Shakiba
 *
 * This source code is licensed under the GNU AGPLv3 license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! The concurrency gate for v5's epoch-based value reclamation.
//!
//! Every other test in the tree checks a value's bytes from one thread, and so
//! cannot see the defect this design exists to prevent: a reader that has
//! dropped the shard guard and is part-way through copying a value while a
//! writer unpublishes and frees it. That window is the whole reason values are
//! retired through [`crate::value::defer_free`] rather than freed inline, and
//! it is invisible to a single-threaded test.
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
//! The second half of the gate is the opposite failure: garbage that is
//! deferred and never runs. After the threads join, the harness quiesces, calls
//! [`crate::value::flush`], and requires both that every deferral has executed
//! (`VALUE_FREES_DEFERRED == VALUE_FREES_RUN`) and that jemalloc
//! `stats.allocated` is back within 1% of where it was before the run.
//!
//! ## Why `#[ignore]`
//!
//! `stats.allocated` and the two free counters are PROCESS-WIDE. Cargo runs the
//! lib tests concurrently in one process, so any other test running alongside
//! this one would allocate into the same numbers. This harness therefore owns
//! its process, the same rule every `measure_*` harness in the tree follows:
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
//! `attempt to add with overflow` at `worker/policy/mod.rs:185` under this load.
//! `MigrationQueue::push` calls `record_pending` AFTER `send` succeeds, so a
//! consumer can receive the item and run `PendingOnDrop` -- the decrement --
//! before the producer's increment, wrapping the counter to `u64::MAX`. It is
//! instrumentation only, it is not v5's, and it reproduces unchanged on the
//! base commit 9480852 with this same harness. Release builds do not check the
//! overflow, so the gate run uses `--release`; a debug run loses its migration
//! consumer part-way and its memory figures are not comparable.
//!
//! ## Proving it can fail
//!
//! A stress test that has never failed proves nothing. This one was verified by
//! changing `Object::drop` (src/object/mod.rs) from `defer_free(..)` to an
//! inline `unsafe { self.value.free(self.len) }` and re-running: see the gate
//! report for what it caught. Revert before trusting a green run.

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
/// measured statement rather than a worry: with `Object::drop` deliberately
/// changed to free inline, a full run reported `torn=0`. The reason is timing.
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

/// The most retired values that may still be un-run once the cache is idle.
///
/// crossbeam-epoch keeps deferrals in a per-thread bag and only pushes that bag
/// to the global queue when it fills (62 objects on a 64-bit target) or when
/// that thread calls `flush`. So every thread that retires values but does not
/// flush can be sitting on up to a bag of them for as long as it stays idle.
/// That is a BOUNDED residue, which is what `defer_free`'s contract promises;
/// a genuine leak grows without limit and blows straight through this.
///
/// Sized for a handful of such threads. The observed residue and which thread
/// holds it are printed either way.
const OUTSTANDING_BOUND: u64 = 62 * 8;

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

	let (rested, allocated) = settle(cache, Duration::from_secs(120));
	let quiesced = drained && rested;

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
	};

	println!(
		"ROUND {round} reads={} misses={} torn={} alien={} wrong_len={} sets={} flaps={} \
		 waves={} census={:?} quiesced={} allocated={}",
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

	let deferred = crate::value::VALUE_FREES_DEFERRED.load(Ordering::Relaxed);
	let ran = crate::value::VALUE_FREES_RUN.load(Ordering::Relaxed);
	let outstanding = deferred - ran;

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
		"STRESS deferred={deferred} ran={ran} outstanding={outstanding} \
		 outstanding_bytes={}",
		outstanding * VALUE_LEN as u64
	);

	check_round(&first, 1, census_before);
	check_round(&second, 2, census_before);

	// Whatever is still outstanding is a bounded bag residue, not a leak.
	assert!(
		outstanding <= OUTSTANDING_BOUND,
		"{outstanding} deferred frees are still outstanding, past the {OUTSTANDING_BOUND} \
		 that a bounded per-thread bag residue can account for -- that is a leak"
	);

	// THE LEAK CRITERION. An identical second round adds essentially nothing:
	// round 1's deferred garbage was reclaimed before round 2 ran, so the
	// figure does not CLIMB. Retained garbage would put another round's worth
	// on top -- round 2 alone retires over five gigabytes of values, so even a
	// hundredth of a percent of retention is far past this budget.
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
/// Quiescent means two separate things, and both are needed.
///
/// First, every deferral has executed. [`crate::value::flush`] pushes THIS
/// thread's bag and tries to advance the global epoch; the `get` pokes the
/// policy worker into an event-loop pass, which is what makes the worker flush
/// ITS bag. An idle worker never pins again on its own, so its garbage would
/// otherwise sit there -- exactly the hazard `flush` exists for.
///
/// Second, `stats.allocated` has stopped moving. Eight readers spin far faster
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
		for _ in 0..64 {
			let _ = cache.get(&0u32);
		}

		crate::value::flush();

		if Instant::now() >= next_sample {
			let allocated = allocated_bytes();
			let deferred = crate::value::VALUE_FREES_DEFERRED.load(Ordering::Relaxed);
			let ran = crate::value::VALUE_FREES_RUN.load(Ordering::Relaxed);

			println!(
				"SETTLE t={:.0}s allocated={allocated} outstanding={}",
				started.elapsed().as_secs_f64(),
				deferred - ran,
			);

			samples.push(allocated);
			next_sample = Instant::now() + Duration::from_secs(1);

			// Three consecutive one-second samples within 0.1% of one
			// another. Deliberately NOT "and nothing outstanding": a small
			// residue is expected and is measured separately below -- see
			// `OUTSTANDING_BOUND`.
			let _ = (deferred, ran);

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
