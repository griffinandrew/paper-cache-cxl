/*
 * Copyright (c) Kia Shakiba
 *
 * This source code is licensed under the GNU AGPLv3 license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! Microbenchmarks of the value layouts' get and set paths. `#[ignore]`d:
//! each one is a measurement, run alone in a release build, under
//! `--test-threads=1`, with the process confined to node-0 CPUs. The layout is
//! the build's (`thin_header` or not), so each is run once per build. Results
//! are printed as `VBENCH ...` lines, one per configuration.
//!
//! * `get_bench` -- `PaperCache::get` hit latency through the real
//!   read path, on a flat cache whose values are all fast (`BufferDRAM`) or
//!   all slow (`BufferPMEM`, node 2), per value size, per prefetch setting
//!   (`crate::read_path`; `VBENCH_LINES`, a comma list of line counts and
//!   `a<CAP>` for `auto:CAP`, default `0,8,a16,a64`), keys drawn uniformly at random from a working set far
//!   above the LLC. Knob combinations are interleaved in rounds so drift and
//!   neighbours' load land on all of them alike.
//! * `set_parts_bench` -- a set's steps, timed apart with the TSC: the
//!   value's allocation, the copy, the header (`assume_init`), the map insert,
//!   the old object's accounting reads (what `publish_set` reads of the
//!   replaced object), an event send, the old object's drop.
//!   On a bare object map with explicit tiers, so nothing but the layout
//!   differs.
//! * `set_cache_bench` -- `PaperCache::set` on a real LRU hybrid cache:
//!   new keys, then overwrites, with the fast tier holding everything or a
//!   small fast tier (demotions, the byte gate).
//!
//! Environment: `VBENCH_CPUS` (comma list, node-0 CPUs; the first is the GET
//! bench's), `VBENCH_SIZES`, `VBENCH_WS_MB`, `VBENCH_OPS`, `VBENCH_THREADS`,
//! `VBENCH_TIERS` (get: `slow,fast`), `VBENCH_LINES` (get).

#![allow(clippy::all)]

use std::{
	hint::black_box,
	sync::{Arc, Barrier},
	time::{Duration, Instant},
};

use crate::{
	CacheTierSize, HashedKey, PaperCache, PaperPolicy, Tier, TieredBuffer,
	object::Object,
	value::{BufferDRAM, BufferPMEM, TieredValue, ValueShape},
};

const LAYOUT: &str = if cfg!(feature = "thin_header") { "thin" } else { "default" };

#[inline(always)]
fn tsc() -> u64 {
	// SAFETY: rdtsc and lfence have no preconditions.
	unsafe {
		core::arch::x86_64::_mm_lfence();
		let t = core::arch::x86_64::_rdtsc();
		core::arch::x86_64::_mm_lfence();
		t
	}
}

/// TSC ticks per nanosecond, measured against the monotonic clock.
fn ticks_per_ns() -> f64 {
	let (t0, i0) = (tsc(), Instant::now());
	std::thread::sleep(Duration::from_millis(200));
	let (t1, i1) = (tsc(), Instant::now());

	(t1 - t0) as f64 / (i1 - i0).as_nanos() as f64
}

fn pin(cpu: usize) {
	// SAFETY: a zeroed cpu_set_t is a valid empty set; the calls only read it.
	unsafe {
		let mut set: libc::cpu_set_t = std::mem::zeroed();
		libc::CPU_SET(cpu, &mut set);
		let rc = libc::sched_setaffinity(0, std::mem::size_of::<libc::cpu_set_t>(), &set);
		assert_eq!(rc, 0, "sched_setaffinity({cpu})");
	}
}

fn env_list(name: &str, default: &[usize]) -> Vec<usize> {
	std::env::var(name)
		.ok()
		.map(|v| v.split(',').filter_map(|x| x.trim().parse().ok()).collect())
		.unwrap_or_else(|| default.to_vec())
}

fn env_usize(name: &str, default: usize) -> usize {
	std::env::var(name).ok().and_then(|v| v.trim().parse().ok()).unwrap_or(default)
}

fn cpus() -> Vec<usize> {
	env_list("VBENCH_CPUS", &[2, 3, 4, 5, 6, 7, 8, 9, 10])
}

fn splitmix(x: u64) -> u64 {
	let mut z = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
	z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
	z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
	z ^ (z >> 31)
}

struct Rng(u64);

impl Rng {
	fn next(&mut self) -> u64 {
		self.0 = self.0.wrapping_add(1);
		splitmix(self.0)
	}
}

/// mean, p50, p90, p99 in ns of `ticks`.
fn summary(ticks: &mut [u64], tpn: f64) -> (f64, f64, f64, f64) {
	if ticks.is_empty() {
		return (0.0, 0.0, 0.0, 0.0);
	}

	ticks.sort_unstable();
	let n = ticks.len();
	let mean = ticks.iter().sum::<u64>() as f64 / n as f64 / tpn;
	let q = |p: f64| ticks[((n as f64 * p) as usize).min(n - 1)] as f64 / tpn;

	(mean, q(0.50), q(0.90), q(0.99))
}

/// A Fisher-Yates permutation of `0..n`.
fn permutation(n: usize, seed: u64) -> Vec<u64> {
	let mut v: Vec<u64> = (0..n as u64).collect();
	let mut rng = Rng(seed);

	for i in (1..n).rev() {
		let j = (rng.next() % (i as u64 + 1)) as usize;
		v.swap(i, j);
	}

	v
}

// ---------------------------------------------------------------------------
// GET
// ---------------------------------------------------------------------------

/// (label, lines, auto) per prefetch setting, from `VBENCH_LINES`: a comma
/// list of line counts (`0` is off) and `a<CAP>` for `auto:CAP`.
fn get_combos() -> Vec<(String, u32, bool)> {
	let list = std::env::var("VBENCH_LINES").unwrap_or_else(|_| "0,8,a16,a64".into());

	list.split(',')
		.filter_map(|x| {
			let x = x.trim();
			match x.strip_prefix('a') {
				Some(cap) => cap.parse::<u32>().ok().map(|c| (format!("auto{c}"), c, true)),
				None => x.parse::<u32>().ok().map(|l| (format!("pf{l}"), l, false)),
			}
		})
		.collect()
}

fn get_one<V: ValueShape>(tier: Tier, size: usize, tpn: f64) {
	let ws = env_usize("VBENCH_WS_MB", 1024) << 20;
	let ops = env_usize("VBENCH_OPS", 300_000);
	let rounds = 6;
	let n = (ws / size).clamp(50_000, 4_000_000);
	let cpu = cpus()[0];

	let cache = PaperCache::<u64, V>::new(64 << 30, &[PaperPolicy::LruCompact], PaperPolicy::LruCompact).unwrap();
	let value = vec![0xA5u8; size];

	for k in 0..n as u64 {
		cache.set(k, &value, None).unwrap();
	}

	// The policy worker takes its `Set`s; let it drain before timing.
	std::thread::sleep(Duration::from_secs(2));

	pin(cpu);

	let combos = get_combos();
	// Fast-tier values are prefetched only with `fast_too`: on for them here.
	let fast_too = matches!(tier, Tier::Fast);
	let mut samples: Vec<Vec<u64>> = vec![Vec::with_capacity(ops); combos.len()];
	let mut rng = Rng(0xC0FFEE ^ size as u64);
	let mut misses = 0u64;

	for _ in 0..rounds {
		for (c, &(_, lines, auto)) in combos.iter().enumerate() {
			crate::read_path::set_knobs_for_test(lines, auto, fast_too);

			for _ in 0..ops / rounds {
				let key = rng.next() % n as u64;
				let t0 = tsc();
				let got = cache.get(&key);
				let t1 = tsc();

				misses += got.is_err() as u64;
				let _ = black_box(got);
				samples[c].push(t1 - t0);
			}
		}
	}

	crate::read_path::set_knobs_for_test(0, false, false);

	for (c, (label, ..)) in combos.iter().enumerate() {
		let (mean, p50, p90, p99) = summary(&mut samples[c], tpn);

		println!(
			"VBENCH GET layout={LAYOUT} tier={tier:?} size={size} combo={label} n={n} ops={} mean={mean:.0} p50={p50:.0} p90={p90:.0} p99={p99:.0} misses={misses}",
			samples[c].len(),
		);
	}

	drop(cache);
	std::thread::sleep(Duration::from_millis(500));
}

#[test]
#[ignore = "measurement; release, --test-threads=1, node-0 CPUs"]
fn get_bench() {
	let tpn = ticks_per_ns();
	println!("VBENCH TSC ticks_per_ns={tpn:.4}");

	let sizes = env_list("VBENCH_SIZES", &[64, 256, 1024, 4096, 16384]);
	let tiers = std::env::var("VBENCH_TIERS").unwrap_or_else(|_| "slow,fast".into());

	for tier in tiers.split(',') {
		for &size in &sizes {
			match tier {
				"fast" => get_one::<BufferDRAM>(Tier::Fast, size, tpn),
				_ => get_one::<BufferPMEM>(Tier::Slow, size, tpn),
			}
		}
	}
}

// ---------------------------------------------------------------------------
// SET, by parts
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug)]
enum Scenario {
	NewFast,
	NewSlow,
	OwFastToFast,
	OwSlowToFast,
	OwSlowToSlow,
}

impl Scenario {
	fn tiers(self) -> (Option<Tier>, Tier) {
		match self {
			Scenario::NewFast => (None, Tier::Fast),
			Scenario::NewSlow => (None, Tier::Slow),
			Scenario::OwFastToFast => (Some(Tier::Fast), Tier::Fast),
			Scenario::OwSlowToFast => (Some(Tier::Slow), Tier::Fast),
			Scenario::OwSlowToSlow => (Some(Tier::Slow), Tier::Slow),
		}
	}
}

const PARTS: [&str; 7] = ["alloc", "copy", "header", "insert", "oldread", "send", "olddrop"];

type Map = crate::ObjectMapRef<u64, TieredBuffer>;

fn set_parts_one(threads: usize, size: usize, scenario: Scenario, tpn: f64) {
	let ws = env_usize("VBENCH_WS_MB", 1024) << 20;
	let per_thread = (ws / size / threads).clamp(20_000, 1_000_000);
	let ops = env_usize("VBENCH_OPS", 300_000).min(per_thread);
	let (old_tier, new_tier) = scenario.tiers();
	let cpus = cpus();

	let map: Map = crate::new_hybrid_object_map();
	let value = vec![0x5Au8; size];

	// Old objects for the overwrite scenarios, each thread's range its own.
	if let Some(old) = old_tier {
		for k in 0..(per_thread * threads) as u64 {
			map.insert(splitmix(k), Object::from_value(TieredValue::new_in(k, &value, old, None)));
		}
	}

	let (tx, rx) = crossbeam_channel::unbounded::<[u64; 8]>();
	let drain_cpu = cpus[threads.min(cpus.len() - 1)];
	let drain = std::thread::spawn(move || {
		pin(drain_cpu);
		let mut n = 0u64;
		while let Ok(m) = rx.recv() {
			n = n.wrapping_add(m[0]);
		}
		n
	});

	let barrier = Arc::new(Barrier::new(threads));
	let mut handles = Vec::new();

	for t in 0..threads {
		let map = Arc::clone(&map);
		let tx = tx.clone();
		let barrier = Arc::clone(&barrier);
		let value = value.clone();
		let cpu = cpus[t % cpus.len()];

		handles.push(std::thread::spawn(move || {
			pin(cpu);

			let base = (t * per_thread) as u64;
			// New keys: above every old key. Overwrites: a permutation of this
			// thread's old keys, each overwritten once, cold.
			let keys: Vec<u64> = match old_tier {
				None => (0..ops as u64).map(|i| (1u64 << 40) + base + i).collect(),
				Some(_) => permutation(per_thread, 0xBEEF + t as u64).into_iter().take(ops).map(|i| base + i).collect(),
			};

			let mut parts = vec![Vec::with_capacity(ops); PARTS.len()];
			let mut total = Vec::with_capacity(ops);

			barrier.wait();

			for &k in &keys {
				let hashed: HashedKey = splitmix(k);

				let t0 = tsc();
				let mut uninit = TieredValue::<u64>::new_uninit_in(k, size, new_tier);
				let t1 = tsc();
				// SAFETY: `value` is `size` bytes, the slot's length.
				unsafe {
					std::ptr::copy_nonoverlapping(value.as_ptr(), uninit.bytes_mut().as_mut_ptr().cast::<u8>(), size);
				}
				let t2 = tsc();
				// SAFETY: every byte was written just above.
				let object = Object::<u64, TieredBuffer>::from_value(unsafe { uninit.assume_init(None) });
				let t3 = tsc();
				let old = map.insert(hashed, object);
				let t4 = tsc();
				// What `publish_set` reads of the old object
				// (`OverheadManager::replaced_size_and_expiry`): the word's
				// hint, and the item only when the hint is missing or says
				// the item may carry an expiry.
				let info = old.as_ref().map(|o| match o.value().accounting_hint() {
					Some((item, false)) => (o.key_size() + item, None),
					_ => (o.key_size() + o.len(), o.expiry()),
				});
				black_box(&info);
				let t5 = tsc();
				let _ = tx.send([hashed, size as u64, 0, 0, 0, 0, 0, 0]);
				let t6 = tsc();
				drop(old);
				let t7 = tsc();

				for (i, d) in [t1 - t0, t2 - t1, t3 - t2, t4 - t3, t5 - t4, t6 - t5, t7 - t6].into_iter().enumerate() {
					parts[i].push(d);
				}
				total.push(t7 - t0);
			}

			(parts, total)
		}));
	}

	drop(tx);

	let mut parts = vec![Vec::new(); PARTS.len()];
	let mut total = Vec::new();

	for h in handles {
		let (p, t) = h.join().unwrap();
		for (i, v) in p.into_iter().enumerate() {
			parts[i].extend(v);
		}
		total.extend(t);
	}

	black_box(drain.join().unwrap());

	let mut line = format!(
		"VBENCH SETPARTS layout={LAYOUT} threads={threads} size={size} scenario={scenario:?} ops={}",
		total.len()
	);

	for (i, name) in PARTS.iter().enumerate() {
		let (mean, ..) = summary(&mut parts[i], tpn);
		line += &format!(" {name}={mean:.0}");
	}

	let (mean, p50, _, p99) = summary(&mut total, tpn);
	line += &format!(" total={mean:.0} p50={p50:.0} p99={p99:.0}");
	println!("{line}");

	drop(map);
	std::thread::sleep(Duration::from_millis(300));
}

#[test]
#[ignore = "measurement; release, --test-threads=1, node-0 CPUs"]
fn set_parts_bench() {
	let tpn = ticks_per_ns();
	println!("VBENCH TSC ticks_per_ns={tpn:.4}");

	let sizes = env_list("VBENCH_SIZES", &[64, 256, 1024, 4096, 16384]);
	let threads = env_list("VBENCH_THREADS", &[1, 8]);

	for &t in &threads {
		for &size in &sizes {
			for scenario in [
				Scenario::NewFast,
				Scenario::NewSlow,
				Scenario::OwFastToFast,
				Scenario::OwSlowToFast,
				Scenario::OwSlowToSlow,
			] {
				set_parts_one(t, size, scenario, tpn);
			}
		}
	}
}

// ---------------------------------------------------------------------------
// SET, through a real LRU hybrid cache
// ---------------------------------------------------------------------------

fn set_cache_one(threads: usize, size: usize, small_fast: bool, tpn: f64) {
	let ws = env_usize("VBENCH_WS_MB", 1024) << 20;
	let per_thread = (ws / size / threads).clamp(20_000, 1_000_000 / threads);
	let cpus = cpus();
	let max: u64 = 32 << 30;
	// `small`: room for every object's DRAM metadata (~320 B covers either
	// layout's per-object reservation) and a quarter of the values, so most
	// values are demoted and an overwrite usually replaces a slow one.
	let total = (per_thread * threads) as u64;
	let fast: u64 = if small_fast { total * 320 + total * size as u64 / 4 } else { max };

	let cache = Arc::new(
		PaperCache::<u64, TieredBuffer>::new(max, CacheTierSize::Bytes(fast), PaperPolicy::LruCompactHybrid).unwrap(),
	);

	let mut result = Vec::new();

	for phase in ["new", "overwrite"] {
		let barrier = Arc::new(Barrier::new(threads));
		let mut handles = Vec::new();

		for t in 0..threads {
			let cache = Arc::clone(&cache);
			let barrier = Arc::clone(&barrier);
			let cpu = cpus[t % cpus.len()];
			let value = vec![0x3Cu8; size];

			handles.push(std::thread::spawn(move || {
				pin(cpu);

				let base = (t * per_thread) as u64;
				let keys: Vec<u64> = match phase {
					"new" => (0..per_thread as u64).map(|i| base + i).collect(),
					_ => permutation(per_thread, 0xFACE + t as u64).into_iter().map(|i| base + i).collect(),
				};
				let mut lat = Vec::with_capacity(keys.len());
				let mut errs = 0u64;

				barrier.wait();

				for &k in &keys {
					let t0 = tsc();
					errs += cache.set(k, &value, None).is_err() as u64;
					lat.push(tsc() - t0);
				}

				(lat, errs)
			}));
		}

		let mut lat = Vec::new();
		let mut errs = 0;
		for h in handles {
			let (l, e) = h.join().unwrap();
			lat.extend(l);
			errs += e;
		}

		let (mean, p50, p90, p99) = summary(&mut lat, tpn);
		result.push(format!("{phase}_mean={mean:.0} {phase}_p50={p50:.0} {phase}_p90={p90:.0} {phase}_p99={p99:.0} {phase}_errs={errs}"));

		// Let the worker land its demotions before the overwrites.
		std::thread::sleep(Duration::from_secs(2));

		let h = cache.hybrid_stats();
		result.push(format!(
			"{phase}_fast_objects={} {phase}_slow_objects={} {phase}_promotions={} {phase}_demotions={} {phase}_gate_waits={} {phase}_gate_wait_ms={}",
			h.fast_objects,
			h.slow_objects,
			h.promotions,
			h.demotions,
			h.gate_waits,
			h.gate_wait_ns_total / 1_000_000,
		));
	}

	println!(
		"VBENCH SETCACHE layout={LAYOUT} threads={threads} size={size} fast={} fast_bytes={fast} per_thread={per_thread} {}",
		if small_fast { "small" } else { "all" },
		result.join(" "),
	);

	drop(cache);
	std::thread::sleep(Duration::from_millis(500));
}

#[test]
#[ignore = "measurement; release, --test-threads=1, node-0 CPUs"]
fn set_cache_bench() {
	let tpn = ticks_per_ns();
	println!("VBENCH TSC ticks_per_ns={tpn:.4}");

	let sizes = env_list("VBENCH_SIZES", &[64, 256, 1024, 4096, 16384]);
	let threads = env_list("VBENCH_THREADS", &[1, 8]);

	for &t in &threads {
		for &size in &sizes {
			for small_fast in [false, true] {
				set_cache_one(t, size, small_fast, tpn);
			}
		}
	}
}
