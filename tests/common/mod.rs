/*
 * Copyright (c) Kia Shakiba
 *
 * This source code is licensed under the GNU AGPLv3 license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! Helpers the integration test binaries share (`mod common;`). Each binary
//! compiles its own copy and uses a subset, hence the blanket `dead_code`.
//!
//! Child processes: the process-global counters (P, the migration statistics,
//! the gate's accounting, the allocator's arenas) outlive a cache, so a test
//! that runs several policies runs each in a child copy of the test binary
//! ([`each_alone`]), and a test that must not see another's leavings runs
//! alone in one ([`alone`]) -- "each policy run resets all cache state", and
//! only a fresh process resets all of it.

#![allow(dead_code)]

use std::{
	process::Command,
	thread,
	time::{Duration, Instant},
};

const CHILD: &str = "PAPER_TEST_CHILD";

/// The child's case, for a test run once per case: an index into its list.
const CASE: &str = "PAPER_TEST_CASE";

/// How long a child may run before it is killed and its test failed.
const DEADLINE: Duration = Duration::from_secs(900);

fn in_child() -> bool {
	std::env::var_os(CHILD).is_some_and(|value| value == "1")
}

/// Runs `body` in a child process in which `test`, of `module`
/// (`module_path!()`), is the only test, killed at `DEADLINE`. The parent
/// passes only if the child ran exactly that one test and it passed.
pub fn alone(module: &str, test: &str, body: impl FnOnce()) {
	if in_child() {
		body();
		return;
	}

	run_in_child(module, test, None);
}

/// `test`, of `module`, once per case of `cases` -- a policy, a design -- each
/// in a child process of its own, in which it is the only test and only that
/// case runs `body`.
pub fn each_alone<T: Copy>(module: &str, test: &str, cases: impl AsRef<[T]>, mut body: impl FnMut(T)) {
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

fn run_in_child(module: &str, test: &str, case: Option<usize>) {
	// libtest names a test by its path without the crate.
	let name = match module.split_once("::") {
		Some((_, module)) => format!("{module}::{test}"),
		None => test.to_owned(),
	};

	let tag = case.map_or_else(String::new, |index| format!("-{index}"));
	let path = std::env::temp_dir().join(format!("paper-test-{}-{test}{tag}.out", std::process::id()));
	let file = std::fs::File::create(&path).expect("the child's output file");

	let mut command = Command::new(std::env::current_exe().expect("this test binary"));

	command
		.args([name.as_str(), "--exact", "--test-threads=1", "--nocapture"])
		.env(CHILD, "1")
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
