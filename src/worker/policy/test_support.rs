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
	if in_child() {
		body();
		return;
	}

	run_in_child(module, test, None);
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
		run_in_child(module, test, Some(index));
	}
}

/// Runs the child copy of this test binary for `alone_in` and `each_alone_in`.
fn run_in_child(module: &str, test: &str, case: Option<usize>) {
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

/// `each_alone_in` for the calling module's `test`: `each_alone!("name", CASES,
/// |case| { .. })`.
macro_rules! each_alone {
	($test:literal, $cases:expr, $body:expr) => {
		$crate::worker::policy::test_support::each_alone_in(module_path!(), $test, $cases, $body)
	};
}

pub(super) use each_alone;
