/*
 * Copyright (c) Kia Shakiba
 *
 * This source code is licensed under the GNU AGPLv3 license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! Helpers the worker's test modules share.
//!
//! Child processes: the process-global counters (the migration statistics, P
//! and the gate's accounting, the allocator's arenas) outlive a cache, so a
//! test that must not see another's leavings runs in a child copy of the test
//! binary ([`alone_in`]), and a test that runs several policies runs each in a
//! child of its own ([`each_alone`]) -- "each policy run resets all cache
//! state", and only a fresh process resets all of it.

use std::{
	thread,
	time::{Duration, Instant},
};

#[cfg(feature = "hybrid_cache_common")]
use std::sync::Arc;

#[cfg(feature = "hybrid_cache_common")]
use crossbeam_channel::unbounded;

#[cfg(feature = "hybrid_cache_common")]
use crate::{
	CacheSize, HashedKey, ObjectMapRef, OverheadManagerRef, PaperPolicy, StatusRef, TieredBuffer,
	gate::{self, Verdict},
	object::{ObjectSize, overhead::OverheadManager},
	status::AtomicStatus,
	worker::WorkerEvent,
};

#[cfg(feature = "hybrid_cache_common")]
use super::{
	PolicyWorker, Tier,
	policy_stack::PolicyStack,
	reconcile_tests::{Objects, Worker, handle, publish},
};

const CHILD: &str = "PAPER_GATE_TEST_CHILD";

/// How long a child may run before it is killed and its test failed.
const DEADLINE: Duration = Duration::from_secs(90);

/// The child's case, for a test run once per case ([`each_alone_in`]): an index
/// into the test's list.
const CASE: &str = "PAPER_GATE_TEST_CASE";

fn in_child() -> bool {
	std::env::var_os(CHILD).is_some_and(|value| value == "1")
}

/// Runs `body` in a child process in which `test`, of `module`
/// (`module_path!()`), is the only test, killed at `DEADLINE`. The parent
/// passes only if the child ran exactly that one test and it passed. Also
/// T14's gate half (`s4_tests`).
#[cfg(feature = "hybrid_cache_common")]
pub(super) fn alone_in(module: &str, test: &str, body: impl FnOnce()) {
	alone_in_with_env(module, test, &[], body);
}

/// `alone_in`, with `env` set in the child: for a setting the crate reads once
/// per process (`EVICTION_HIGH_WATERMARK` and the like, memoised in a
/// `OnceLock`), which a test can only choose by starting a process of its own
/// with it. The parent's environment is untouched.
#[cfg(feature = "hybrid_cache_common")]
pub(super) fn alone_in_with_env(module: &str, test: &str, env: &[(&str, &str)], body: impl FnOnce()) {
	if in_child() {
		body();
		return;
	}

	run_in_child(module, test, None, env);
}

/// `test`, of `module`, once per case of `cases` -- a policy, a design, an
/// order -- each in a child process of its own, in which it is the only test
/// and only that case runs `body`. "Each policy run resets all cache state":
/// the process-global counters (the migration statistics, P and the gate's
/// accounting, the allocator's arenas) outlive a cache, so one process keeps
/// what the previous policy left; only a fresh process resets them. The parent
/// passes only if every child ran exactly that one test and it passed.
pub(super) fn each_alone_in<T: Copy>(module: &str, test: &str, cases: impl AsRef<[T]>, mut body: impl FnMut(T)) {
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
		run_in_child(module, test, Some(index), &[]);
	}
}

/// Runs the child copy of this test binary for `alone_in` and `each_alone_in`,
/// with `env` added to its environment.
fn run_in_child(module: &str, test: &str, case: Option<usize>, env: &[(&str, &str)]) {
	// libtest names a test by its path without the crate.
	let (_, module) = module.split_once("::").expect("a module path");
	let name = format!("{module}::{test}");
	let tag = case.map_or_else(String::new, |index| format!("-{index}"));
	let path = std::env::temp_dir().join(format!("paper-gate-{}-{test}{tag}.out", std::process::id()));
	let file = std::fs::File::create(&path).expect("the child's output file");

	let mut command = std::process::Command::new(std::env::current_exe().expect("this test binary"));

	command
		.args([name.as_str(), "--exact", "--test-threads=1", "--nocapture"])
		.env(CHILD, "1")
		// The consumer count the timing bounds assume (the default).
		.env("MIGRATION_QUEUE_THREADS", "2")
		.stdout(file.try_clone().expect("the output file, twice"))
		.stderr(file);

	for (name, value) in env {
		command.env(name, value);
	}

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

/// Polls `done` every millisecond until it holds, failing after `deadline`
/// (a hang detector, not the measurement).
pub(super) fn wait_for(what: &str, deadline: Duration, done: impl FnMut() -> bool) {
	wait_for_every(Duration::from_millis(1), what, deadline, done);
}

/// `wait_for`, polling every `poll`: a test that times what it waits for is
/// bounded by its own poll.
pub(super) fn wait_for_every(poll: Duration, what: &str, deadline: Duration, mut done: impl FnMut() -> bool) {
	let start = Instant::now();

	while !done() {
		assert!(start.elapsed() < deadline, "{what} did not happen within {deadline:?}");
		thread::sleep(poll);
	}
}

/// The policy stack of a hand-driven worker.
#[cfg(feature = "hybrid_cache_common")]
pub(super) fn stack<K, V>(worker: &PolicyWorker<K, V>) -> &dyn PolicyStack {
	&*worker.policy_stack
}

/// Waits for the worker's first pass, then CHECKS the premise the kick tests
/// rest on rather than assuming it: with nothing queued and no set ever seen,
/// that pass chose the LONG poll, so no other pass runs in the next 100 ms (on
/// the SHORT poll about a hundred would). `poll` is the interval the first
/// wait polls at. Returns the pass count to measure from.
#[cfg(feature = "hybrid_cache_common")]
pub(super) fn parked_on_the_long_poll(status: &AtomicStatus, poll: Duration) -> u64 {
	wait_for_every(poll, "the worker's first pass", Duration::from_secs(10), || {
		status.policy_worker_passes() >= 1
	});

	let passes = status.policy_worker_passes();

	thread::sleep(Duration::from_millis(100));

	assert_eq!(
		status.policy_worker_passes(),
		passes,
		"the idle worker ran another pass within 100 ms of its first: it is not \
		 parked on the {:?} poll, so a kick would prove nothing",
		super::LONG_POLLING_DURATION,
	);

	passes
}

/// A worker driven by hand over `objects`: its status (a cache of `max_size`
/// running `policy`), its overhead manager and, unless `queued` is false, its
/// migration queue. `per_object` pins the per-object metadata model the hand-
/// driven tests are written against (S5).
///
/// The worker never runs on a thread here -- the event channel is dropped
/// with this call -- so a test calls its handlers itself.
#[cfg(feature = "hybrid_cache_common")]
pub(super) fn tiered_worker<K>(
	objects: ObjectMapRef<K, TieredBuffer>,
	max_size: CacheSize,
	policy: PaperPolicy,
	per_object: bool,
	queued: bool,
) -> (PolicyWorker<K, TieredBuffer>, StatusRef, OverheadManagerRef)
where
	K: 'static + Eq + Clone + typesize::TypeSize + Send + Sync,
{
	let (_tx, rx) = unbounded::<WorkerEvent>();

	let status = Arc::new(AtomicStatus::new(max_size, &[policy], policy).unwrap());
	let overhead_manager = Arc::new(OverheadManager::new(&status));

	if per_object {
		status.pin_per_object();
	}

	let mut worker = PolicyWorker::new_with_tier_migration(
		rx,
		objects,
		status.clone(),
		overhead_manager.clone(),
	).unwrap();

	if !queued {
		// Dropping the queue joins its consumers, so nothing is left in
		// flight and `apply_migration_batches` takes the synchronous path.
		worker.migration_queue = None;
	}

	(worker, status, overhead_manager)
}

/// `PaperCache::begin_set`'s decision for a `len`-byte value of `key`, from
/// the sizes it computes before allocating.
#[cfg(feature = "hybrid_cache_common")]
pub(super) fn decide(worker: &Worker, objects: &Objects, key: HashedKey, len: usize) -> Result<Verdict, crate::CacheError> {
	let sizes = gate::Sizes {
		base: worker.overhead_manager.base_size_for(&key, len, None).expect("a length in range"),
		resident: worker.overhead_manager.dram_resident_size_for(&key, None),
		value: crate::phys::value_charge::<u64>(len as ObjectSize),
	};

	gate::decide(&worker.status, objects, key, &sizes, false)
}

/// A client's set through the decision -- built where it says, its `Set`
/// carrying its placement -- then the worker's handling of the `Set`, as
/// `PaperCache::set` makes it since S5. The decision must admit. Returns it.
#[cfg(feature = "hybrid_cache_common")]
pub(super) fn client_set(worker: &mut Worker, objects: &Objects, key: HashedKey, len: usize) -> (Tier, super::Placement) {
	let Ok(Verdict::Admit { tier, placement }) = decide(worker, objects, key, len) else {
		panic!("{}: key {key} was not admitted", worker.status.policy());
	};

	let mut published = publish(&worker.status, &worker.overhead_manager, objects, key, len, tier);
	published.placement = placement;

	handle(worker, key, published);

	(tier, placement)
}

/// A stack that stands in for a real one where a test needs a particular
/// answer of it and nothing else: it tracks the keys it is inserted (in
/// order, evicting the oldest first), reports the drain it was given, and --
/// with a `budget` -- insists on draining to that many objects.
#[cfg(feature = "hybrid_cache_common")]
pub(super) struct FakeStack {
	keys: std::collections::VecDeque<HashedKey>,
	migrations: Vec<(HashedKey, Tier)>,
	budget: usize,
}

#[cfg(feature = "hybrid_cache_common")]
impl FakeStack {
	/// Hands the worker ONE scripted drain and nothing else, so an exact entry
	/// sequence can go through `apply_tier_migrations` -- the path production
	/// takes, `split_tier_migrations` included -- without coaxing a real stack
	/// into emitting it.
	pub(super) fn scripted(migrations: Vec<(HashedKey, Tier)>) -> Self {
		FakeStack { keys: Default::default(), migrations, budget: usize::MAX }
	}

	/// A stack whose *internal* sub-structure is over its own budget -- the
	/// `needs_capacity_eviction` case: it insists on draining to `budget`
	/// objects however much room the cache as a whole still has.
	/// `TwoQCompactHybridStack`'s `k_in`-derived fifo budget is the real
	/// instance; this stands in for it because that one needs a hybrid design
	/// compiled in and a fast tier configured, neither of which the condition
	/// (or the watermark that must stay off it) has anything to do with.
	pub(super) fn over_budget(budget: usize) -> Self {
		FakeStack { keys: Default::default(), migrations: Vec::new(), budget }
	}
}

#[cfg(feature = "hybrid_cache_common")]
impl PolicyStack for FakeStack {
	fn is_policy(&self, policy: &PaperPolicy) -> bool {
		matches!(policy, PaperPolicy::LruCompact)
	}

	fn len(&self) -> usize {
		self.keys.len()
	}

	fn contains(&self, key: HashedKey) -> bool {
		self.keys.contains(&key)
	}

	fn insert(&mut self, key: HashedKey, _size: ObjectSize) {
		self.keys.push_back(key);
	}

	fn remove(&mut self, key: HashedKey) {
		self.keys.retain(|existing| *existing != key);
	}

	fn clear(&mut self) {
		self.keys.clear();
	}

	fn evict_one(&mut self) -> Option<HashedKey> {
		self.keys.pop_front()
	}

	fn drain_tier_migrations(&mut self) -> Vec<(HashedKey, Tier)> {
		std::mem::take(&mut self.migrations)
	}

	fn needs_capacity_eviction(&self) -> bool {
		self.keys.len() > self.budget
	}
}

/// `each_alone_in` for the calling module's `test`: `each_alone!("name", CASES,
/// |case| { .. })`.
macro_rules! each_alone {
	($test:literal, $cases:expr, $body:expr) => {
		$crate::worker::policy::test_support::each_alone_in(module_path!(), $test, $cases, $body)
	};
}

pub(super) use each_alone;
