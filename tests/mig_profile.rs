/*
 * Copyright (c) Griffin Andrew
 *
 * This source code is licensed under the GNU AGPLv3 license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! What slows the migration consumers down, in process: a Zipf GET/SET
//! workload on one tiered cache, in three windows, each with its own
//! `MIGPROF` block (`paper_cache::migprof`):
//!
//!   1. `mixed`   -- GETs and SETs, migrations running: the consumers' copy
//!                   rates, lock waits, superseded copies, and the worker's
//!                   busy fraction against the queue depth (slow, or starved?).
//!   2. `get_on`  -- GETs only, migrations running.
//!   3. `get_off` -- GETs only, migrations PAUSED: no migration traffic, so
//!                   `get_slow` here against window 2 is what migrations cost
//!                   a slow GET (p99 on vs off).
//!
//! Needs `MIG_PROFILE=1`. On a machine with no slow node, add `EMU_SLOW=1`
//! (and its `EMU_SLOW_*` knobs) and label the results emulated. Never set
//! `EMU_SLOW` on the CXL machine.
//!
//!   MIG_PROFILE=1 cargo +nightly test --release --features lru_compact_hybrid_cache \
//!     --test mig_profile -- --ignored --nocapture
//!
//! Knobs (`MP_*`, defaults in brackets): `MP_POLICY` [lru-compact-hybrid],
//! `MP_KEYS` [200000], `MP_VALUE` [4096] bytes, `MP_FAST_FRAC` [0.25] of the
//! working set, `MP_ZIPF` [0.99], `MP_CLIENTS` [2], `MP_SET_FRAC` [0.1] of
//! the mixed window's ops, `MP_SECS` [10] per window. `MIGRATION_QUEUE_THREADS`
//! and the rest of the cache's own knobs apply as usual.

#[cfg(feature = "hybrid_cache_common")]
mod mig_profile {
	use std::{
		str::FromStr,
		sync::{
			Arc,
			atomic::{AtomicBool, AtomicU64, Ordering},
		},
		thread,
		time::{Duration, Instant},
	};

	use paper_cache::{CacheTierSize, PaperCache, PaperPolicy, TieredBuffer};

	fn env<T: FromStr>(name: &str, default: T) -> T {
		std::env::var(name).ok().and_then(|v| v.parse().ok()).unwrap_or(default)
	}

	struct Zipf {
		cdf: Vec<f64>,
	}

	impl Zipf {
		fn new(n: usize, alpha: f64) -> Self {
			let mut cdf = Vec::with_capacity(n);
			let mut sum = 0.0;

			for rank in 1..=n {
				sum += 1.0 / (rank as f64).powf(alpha);
				cdf.push(sum);
			}

			cdf.iter_mut().for_each(|c| *c /= sum);

			Zipf { cdf }
		}

		fn sample(&self, u: f64) -> u64 {
			self.cdf.partition_point(|&c| c < u) as u64
		}
	}

	struct Rng(u64);

	impl Rng {
		fn next(&mut self) -> u64 {
			self.0 ^= self.0 << 13;
			self.0 ^= self.0 >> 7;
			self.0 ^= self.0 << 17;
			self.0
		}

		fn unit(&mut self) -> f64 {
			(self.next() >> 11) as f64 / (1u64 << 53) as f64
		}
	}

	/// The rank -> key permutation: hot keys spread over the hash shards.
	fn key_of(rank: u64) -> u64 {
		rank.wrapping_mul(0x9E37_79B9_7F4A_7C15)
	}

	fn window(
		name: &str,
		cache: &Arc<PaperCache<u64, TieredBuffer>>,
		zipf: &Arc<Zipf>,
		clients: usize,
		set_frac: f64,
		value: usize,
		secs: u64,
	) {
		cache.migprof_reset();

		let stop = Arc::new(AtomicBool::new(false));
		let ops = Arc::new(AtomicU64::new(0));
		let start = Instant::now();

		let handles: Vec<_> = (0..clients)
			.map(|c| {
				let (cache, zipf, stop, ops) = (cache.clone(), zipf.clone(), stop.clone(), ops.clone());

				thread::spawn(move || {
					let mut rng = Rng(0x2545_F491_4F6C_DD1D ^ ((c as u64 + 1) << 32));
					let payload = vec![c as u8; value];
					let mut done = 0u64;

					while !stop.load(Ordering::Relaxed) {
						for _ in 0..256 {
							let key = key_of(zipf.sample(rng.unit()));

							if set_frac > 0.0 && rng.unit() < set_frac {
								let _ = cache.set(key, &payload, None);
							} else if cache.get(&key).is_err() {
								let _ = cache.set(key, &payload, None);
							}
						}

						done += 256;
					}

					ops.fetch_add(done, Ordering::Relaxed);
				})
			})
			.collect();

		thread::sleep(Duration::from_secs(secs));
		stop.store(true, Ordering::Relaxed);
		handles.into_iter().for_each(|h| h.join().unwrap());

		let elapsed = start.elapsed().as_secs_f64();
		let stats = cache.hybrid_stats();

		println!(
			"MIGPROF_WINDOW name={name} ops_per_s={:.0} promotions={} demotions={} fast_objects={} slow_objects={}",
			ops.load(Ordering::Relaxed) as f64 / elapsed,
			stats.promotions,
			stats.demotions,
			stats.fast_objects,
			stats.slow_objects,
		);

		for line in cache.migprof_lines() {
			println!("{name}: {line}");
		}
	}

	#[test]
	#[ignore]
	fn mig_profile() {
		assert!(
			std::env::var("MIG_PROFILE").as_deref() == Ok("1"),
			"run with MIG_PROFILE=1",
		);

		let policy = PaperPolicy::from_str(&env("MP_POLICY", "lru-compact-hybrid".to_string())).expect("MP_POLICY");
		let keys: usize = env("MP_KEYS", 200_000);
		let value: usize = env("MP_VALUE", 4096);
		let fast_frac: f64 = env("MP_FAST_FRAC", 0.25);
		let alpha: f64 = env("MP_ZIPF", 0.99);
		let clients: usize = env("MP_CLIENTS", 2);
		let set_frac: f64 = env("MP_SET_FRAC", 0.1);
		let secs: u64 = env("MP_SECS", 10);

		// Room for every key (no capacity evictions muddying the windows),
		// and a fast tier of `fast_frac` of the values.
		let working_set = (keys * (value + 256)) as u64;
		let max_size = working_set * 3 / 2;
		let fast = (working_set as f64 * fast_frac) as u64;

		println!(
			"MIGPROF_CONFIG policy={policy} keys={keys} value={value} fast_bytes={fast} max_size={max_size} zipf={alpha} clients={clients} set_frac={set_frac} secs={secs} consumers={}",
			std::env::var("MIGRATION_QUEUE_THREADS").unwrap_or_else(|_| "2 (default)".into()),
		);

		let cache = Arc::new(
			PaperCache::<u64, TieredBuffer>::new(max_size, CacheTierSize::Bytes(fast), policy).expect("cache"),
		);

		let zipf = Arc::new(Zipf::new(keys, alpha));
		let payload = vec![7u8; value];

		for rank in 0..keys as u64 {
			cache.set(key_of(rank), &payload, None).expect("load");
		}

		// Let the load's demotions land before measuring.
		thread::sleep(Duration::from_secs(2));

		window("mixed", &cache, &zipf, clients, set_frac, value, secs);
		window("get_on", &cache, &zipf, clients, 0.0, value, secs);

		cache.set_migrations_paused(true);
		window("get_off", &cache, &zipf, clients, 0.0, value, secs);
		cache.set_migrations_paused(false);
	}
}
