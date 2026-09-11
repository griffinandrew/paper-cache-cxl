/*
 * Copyright (c) Kia Shakiba
 *
 * This source code is licensed under the GNU AGPLv3 license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! A PaperCache server: the tiered hybrid cache behind a real socket.
//!
//! The benchmark has only ever driven `PaperCache` in-process, which measures
//! the cache and nothing else. A deployed cache is reached over a socket, and
//! everything that boundary adds -- syscalls, protocol framing, a copy in each
//! direction, the TCP stack -- lands on every request. This serves the same
//! cache over the wire so that cost is measurable rather than assumed.
//!
//! ## The protocol
//!
//! This speaks the PaperCache wire protocol, as implemented by the published
//! `paper-client` crate (1.10.1) and its `paper-utils` (1.2.5) framing, so the
//! stock client can drive it unmodified. The encoding is:
//!
//! ```text
//!   integers   little-endian, fixed width
//!   bool       b'!' (33) = true, b'?' (63) = false
//!   buf/str    u32 length prefix, then that many bytes
//!   request    [command: u8] followed by that command's arguments
//!   response   [ok: bool] then, if ok, the command's payload
//!   error      [false] [code: u8]; code 0 means a CACHE error follows as a
//!              second u8, any other code is a SERVER error
//! ```
//!
//! There is no length prefix on a request as a whole, so a reader cannot skip
//! a command it does not understand -- it has to decode each one to know where
//! the next begins. An unknown command byte is therefore fatal to the
//! connection, and is answered with an error before the socket is dropped.
//!
//! On accept the SERVER speaks first, sending a bare `[ok: bool]` handshake
//! before any command is read. `PaperClient::new` blocks on that byte, so a
//! server that waits for the client instead deadlocks every connection.
//!
//! ## Keys
//!
//! The protocol carries keys as strings, but this server parses each one to a
//! `u64` and runs `PaperCache<u64, TieredBuffer>`.
//!
//! That is deliberate, and it is about comparability rather than convenience.
//! The point of this binary is an A/B against the in-process benchmark, whose
//! traces are `u64` keyed -- and paper-benchmark commit 0df0ba9 ("Pass the
//! trace's u64 keys through instead of round-tripping them as strings")
//! specifically removed string keys from that path. Running `String` keys here
//! would change per-object metadata on the cache side, so a slower result
//! could be the socket or could be the wider key, with no way to tell them
//! apart. Parsing to `u64` keeps everything except the transport identical.
//!
//! The cost of that choice is that a non-numeric key is rejected. Serving
//! arbitrary keys means `PaperCache<String, TieredBuffer>`, which is a change
//! of one type parameter and the two `parse_key` call sites -- but it is not
//! the same experiment.

use std::{
	fmt::Write as _,
	io::{self, BufReader, BufWriter, IoSlice, Read, Write},
	net::{TcpListener, TcpStream},
	process,
	sync::{
		atomic::{AtomicU64, Ordering},
		Arc,
	},
	thread,
	time::{Duration, Instant},
};

#[cfg(not(feature = "all_dram"))]
use paper_cache::{
	CacheError, CacheSize, CacheTierSize, PaperCache, PaperPolicy, TieredBuffer,
};

#[cfg(feature = "all_dram")]
use paper_cache::{BufferDRAM, CacheError, CacheSize, PaperCache, PaperPolicy};

/// Command bytes, from `paper_utils::command::CommandByte`.
mod command {
	pub const PING: u8 = 0;
	pub const VERSION: u8 = 1;
	pub const AUTH: u8 = 2;
	pub const GET: u8 = 3;
	pub const SET: u8 = 4;
	pub const DEL: u8 = 5;
	pub const HAS: u8 = 6;
	pub const PEEK: u8 = 7;
	pub const TTL: u8 = 8;
	pub const SIZE: u8 = 9;
	pub const WIPE: u8 = 10;
	pub const RESIZE: u8 = 11;
	pub const POLICY: u8 = 12;
	pub const STATS: u8 = 13;
}

const TRUE_INDICATOR: u8 = 33; // b'!'
const FALSE_INDICATOR: u8 = 63; // b'?'

/// Server-level error codes. 0 is reserved: it tells the client a cache error
/// code follows.
mod server_error {
	pub const UNAUTHORIZED: u8 = 3;
}

/// Cache error codes, from `PaperCacheError::from_code`.
fn cache_error_code(err: &CacheError) -> u8 {
	match err {
		CacheError::KeyNotFound => 1,
		CacheError::ZeroValueSize => 2,
		CacheError::ExceedingValueSize => 3,
		CacheError::ZeroCacheSize => 4,
		CacheError::UnconfiguredPolicy => 5,
		CacheError::InvalidPolicy => 6,
		_ => 0,
	}
}

/// The server's own view of what the cache costs, excluding the socket.
///
/// A client can only ever time a round trip: two syscalls, the protocol frames,
/// a copy each way and the TCP stack, wrapped around an operation the
/// in-process benchmark resolves at a few hundred nanoseconds. Those numbers
/// answer "what does a deployed cache cost", which is a fair question but not
/// the same question, and they cannot be compared against an in-process result.
///
/// So the server times the cache call and nothing else -- `Instant::now()`
/// immediately before `cache.get(..)` and `elapsed()` immediately after, with
/// the frame already parsed and the response not yet written. What is left out
/// is deliberate: reading the request, parsing the key, writing the reply and
/// flushing it are all outside the span.
///
/// The counters are per-command and lock-free, so instrumenting costs two
/// `Instant::now()` calls and three relaxed atomic adds on the hot path. That
/// is not nothing at this timescale (order 20-40 ns on x86), and it is charged
/// to the server, not to the cache: it sits outside the timed span except for
/// the clock reads themselves, which bracket it. Treat the reported mean as the
/// cache call plus one clock read, and compare like with like.
///
/// # Reading the SET figure
///
/// A cache that is still FILLING charges every set a first-touch page fault per
/// value page: `fast_alloc` hands back memory the process has never touched,
/// while `get`'s `to_vec` recycles a hot tcache block. Measured here at 4 KiB
/// values -- 3935 ns/set while filling, 1087 ns once keys are overwritten and
/// memory recycles, against 1154 ns for a get. So set is not slower than get;
/// filling is, by 3.6x. Warm to steady state before believing a set number.
mod selfstats {
	use std::sync::atomic::{AtomicU64, Ordering};

	/// Command bytes run 0..=13, so one slot each covers every command.
	pub const SLOTS: usize = 16;

	/// Log-ish buckets: exact below 16 ns, then four sub-buckets per octave,
	/// i.e. ~25% resolution. Enough for percentiles without a lock or a
	/// reservoir, and the MEAN is exact regardless (count and total are).
	pub const BUCKETS: usize = 256;

	pub struct SelfStats {
		count: [AtomicU64; SLOTS],
		total_ns: [AtomicU64; SLOTS],
		hist: Vec<AtomicU64>,
	}

	impl SelfStats {
		pub fn new() -> Self {
			SelfStats {
				count: std::array::from_fn(|_| AtomicU64::new(0)),
				total_ns: std::array::from_fn(|_| AtomicU64::new(0)),
				hist: (0..SLOTS * BUCKETS).map(|_| AtomicU64::new(0)).collect(),
			}
		}

		pub fn record(&self, cmd: u8, nanos: u64) {
			let slot = cmd as usize;
			if slot >= SLOTS {
				return;
			}

			self.count[slot].fetch_add(1, Ordering::Relaxed);
			self.total_ns[slot].fetch_add(nanos, Ordering::Relaxed);
			self.hist[slot * BUCKETS + bucket(nanos)].fetch_add(1, Ordering::Relaxed);
		}

		pub fn count(&self, cmd: u8) -> u64 {
			self.count[cmd as usize].load(Ordering::Relaxed)
		}

		pub fn mean_ns(&self, cmd: u8) -> f64 {
			let n = self.count(cmd);
			match n {
				0 => 0.0,
				n => self.total_ns[cmd as usize].load(Ordering::Relaxed) as f64 / n as f64,
			}
		}

		/// Lower bound of the bucket the requested quantile falls in. Reported
		/// as a bound rather than an interpolated value because the buckets are
		/// ~25% wide and interpolating would imply precision the histogram does
		/// not have.
		pub fn quantile_ns(&self, cmd: u8, q: f64) -> u64 {
			let total = self.count(cmd);
			if total == 0 {
				return 0;
			}

			let target = (total as f64 * q).ceil() as u64;
			let base = cmd as usize * BUCKETS;
			let mut seen = 0u64;

			for b in 0..BUCKETS {
				seen += self.hist[base + b].load(Ordering::Relaxed);
				if seen >= target {
					return bucket_low(b);
				}
			}

			bucket_low(BUCKETS - 1)
		}
	}

	pub fn bucket(ns: u64) -> usize {
		if ns < 16 {
			return ns as usize;
		}

		let e = 63 - ns.leading_zeros() as usize;
		let sub = ((ns >> (e - 2)) & 0b11) as usize;

		(16 + (e - 4) * 4 + sub).min(BUCKETS - 1)
	}

	pub fn bucket_low(b: usize) -> u64 {
		if b < 16 {
			return b as u64;
		}

		let e = (b - 16) / 4 + 4;
		let sub = ((b - 16) % 4) as u64;

		(1u64 << e) | (sub << (e - 2))
	}
}

use selfstats::SelfStats;

/// Times ONE cache call and nothing around it. The request is already parsed
/// and the response is not yet written when this runs.
fn timed<T>(stats: &SelfStats, cmd: u8, call: impl FnOnce() -> T) -> T {
	let started = Instant::now();
	let out = call();
	stats.record(cmd, started.elapsed().as_nanos() as u64);
	out
}

/// Slot for GET misses, outside the command range so it needs no command byte.
///
/// A miss does no copy and no allocation, so folding it into the GET average
/// deflates that average in proportion to the miss ratio. The in-process
/// benchmark times only hits (`handle_read_through` calls `store_get_time`
/// solely on `Ok`), and these figures exist to be compared against those.
const SLOT_GET_MISS: u8 = 14;

/// The command byte the server answers its OWN statistics on.
///
/// Deliberately outside the upstream protocol's 0..=13 range: a stock
/// `paper-client` will never send it, and the server's answer is a plain text
/// buffer rather than the fixed STATS frame, which has no room for tier fields
/// and whose policy strings upstream cannot parse anyway.
const COMMAND_SELF_STATS: u8 = 200;

fn render_self_stats(stats: &SelfStats, cache: &Cache) -> String {
	let mut out = String::new();

	out.push_str("*** SERVER-SIDE CACHE LATENCY (socket excluded) ***\n\n");
	out.push_str("op      count            mean       p50       p90       p99      p999\n");

	for (name, cmd) in [
		("get(hit)", command::GET),
		("get(miss)", SLOT_GET_MISS),
		("set", command::SET),
		("del", command::DEL),
		("has", command::HAS),
		("peek", command::PEEK),
		("ttl", command::TTL),
		("size", command::SIZE),
	] {
		let n = stats.count(cmd);
		if n == 0 {
			continue;
		}

		let _ = writeln!(
			out,
			"{name:<6} {n:>10} {:>13.1}ns {:>7}ns {:>7}ns {:>7}ns {:>7}ns",
			stats.mean_ns(cmd),
			stats.quantile_ns(cmd, 0.50),
			stats.quantile_ns(cmd, 0.90),
			stats.quantile_ns(cmd, 0.99),
			stats.quantile_ns(cmd, 0.999),
		);
	}

	out.push_str("\npercentiles are bucket LOWER BOUNDS (~25% wide); the mean is exact.\n");
	out.push_str(
		"get(hit) is the figure comparable with the in-process benchmark, which times\n\
		 only hits. A miss neither copies nor allocates, so averaging the two together\n\
		 would understate the cost of a get by the miss ratio.\n\
		 A set on a FILLING cache pays a first-touch page fault per value page (3.6x at\n\
		 4 KiB); warm to steady state before comparing set against get.\n",
	);

	// The tier figures are the whole point of reporting here rather than over
	// the wire: the protocol's STATS frame has no room for them, so a stock
	// client cannot see promotions, demotions or the fast/slow split at all.
	if let Ok(status) = cache.status() {
		let _ = writeln!(out, "\n*** CACHE ***\n");
		let _ = writeln!(out, "objects        {}", status.num_objects());
		let _ = writeln!(out, "used size      {} B", status.used_size());
		let _ = writeln!(out, "max size       {} B", status.max_size());
		let _ = writeln!(out, "miss ratio     {:.4}", status.miss_ratio());
		let _ = writeln!(out, "gets/sets/dels {}/{}/{}",
			status.total_gets(), status.total_sets(), status.total_dels());
		let _ = writeln!(out, "rss            {} B (hwm {} B)", status.rss(), status.hwm());
	}

	#[cfg(not(feature = "all_dram"))]
	{
		let tier = cache.hybrid_stats();

		let _ = writeln!(out, "\n*** TIERS ***\n");
		let _ = writeln!(out, "fast tier size {} B", cache.fast_tier_size());
		let _ = writeln!(out, "fast           {} objects, {} B",
			tier.fast_objects, tier.fast_bytes_used);
		let _ = writeln!(out, "               + {} B reserved for per-object metadata",
			tier.fast_metadata_bytes);
		let _ = writeln!(out, "slow           {} objects, {} B",
			tier.slow_objects, tier.slow_bytes_used);
		let _ = writeln!(out, "promotions     {}", tier.promotions);
		let _ = writeln!(out, "demotions      {}", tier.demotions);
		let _ = writeln!(out, "evictions      {}", tier.evictions);
	}

	// A flat build has no tiers to report, and says so rather than printing
	// a table of zeroes that reads like a tiered run that never migrated.
	#[cfg(feature = "all_dram")]
	{
		let _ = writeln!(out, "\n*** TIERS ***\n");
		let _ = writeln!(out, "flat all-DRAM build: no tiers, no migrations, \
			nothing on the slow node.");
	}

	// SHADOW. What the fitted constants predict, beside what the allocator
	// actually handed out. Nothing reads these to make a decision -- the whole
	// point is to characterise the drift before anything depends on it.
	//
	// Expect measured > modelled on DRAM, and by a knowable amount: the counter
	// sees every node-0 allocation in the process, which is the object map's
	// bucket arrays, the eviction-stack arenas, the value headers, ghost
	// entries, the expiry index AND this server's own per-connection buffers.
	// The modelled figure is `fast_bytes_used + fast_metadata_bytes`, where the
	// second term is a per-object constant times the tracked count. The gap
	// between them IS the question.
	#[cfg(feature = "measured_accounting")]
	{
		use paper_cache::numa_alloc::measured;

		let measured_dram = measured::dram_allocated();
		let measured_slow = measured::slow_allocated();

		// The modelled side is shape-dependent; the measured side is not,
		// which is exactly why the two arms are comparable at all. For a flat
		// build the model IS `used_size`, and the slow pool must read zero --
		// if it does not, something placed bytes off-node and the build is not
		// the all-DRAM baseline it claims to be.
		#[cfg(not(feature = "all_dram"))]
		let (modelled_dram, modelled_slow, tracked, object_bytes, metadata_bytes) = {
			let tier = cache.hybrid_stats();
			(
				(tier.fast_bytes_used + tier.fast_metadata_bytes) as u64,
				tier.slow_bytes_used as u64,
				(tier.fast_objects + tier.slow_objects) as u64,
				tier.fast_bytes_used as u64,
				tier.fast_metadata_bytes as u64,
			)
		};

		#[cfg(feature = "all_dram")]
		let (modelled_dram, modelled_slow, tracked, object_bytes, metadata_bytes) = {
			let (objects, used) = cache
				.status()
				.map(|s| (s.num_objects() as u64, s.used_size() as u64))
				.unwrap_or((0, 0));
			(used, 0u64, objects, used, 0u64)
		};

		let _ = writeln!(out, "\n*** MEASURED vs MODELLED (nothing acts on this) ***\n");
		let _ = writeln!(
			out,
			"{:<10} {:>18} {:>18} {:>14} {:>8}",
			"pool", "modelled B", "measured B", "drift B", "ratio",
		);

		for (name, modelled, measured_bytes) in [
			("dram", modelled_dram, measured_dram),
			("slow", modelled_slow, measured_slow),
		] {
			let drift = measured_bytes as i64 - modelled as i64;
			let ratio = match modelled {
				0 => 0.0,
				m => measured_bytes as f64 / m as f64,
			};

			let _ = writeln!(
				out,
				"{name:<10} {modelled:>18} {measured_bytes:>18} {drift:>+14} {ratio:>8.3}",
			);
		}

		// Per-object, which is the form the fitted constants are written in and
		// therefore the only form in which the two are directly comparable.
		if tracked > 0 {
			let _ = writeln!(
				out,
				"\nper tracked object ({tracked}): modelled metadata {:.1} B, \
				 measured DRAM less object bytes {:.1} B",
				metadata_bytes as f64 / tracked as f64,
				(measured_dram as f64 - object_bytes as f64) / tracked as f64,
			);
		}

		let _ = writeln!(
			out,
			"\nmeasured counts EVERY node-0 allocation in this process, including \
			 this\nserver's own connection buffers -- it is an upper bound on the \
			 cache's DRAM,\nnot an attribution of it.",
		);
	}

	out
}

#[cfg(not(feature = "all_dram"))]
type Cache = PaperCache<u64, TieredBuffer>;

#[cfg(feature = "all_dram")]
type Cache = PaperCache<u64, BufferDRAM>;

fn main() {
	let config = match Config::from_args() {
		Ok(config) => config,
		Err(message) => {
			eprintln!("{message}");
			process::exit(2);
		},
	};

	// The flat constructor takes the set of CONFIGURED policies rather than a
	// tier size -- there is no tier to size. Only the running policy is
	// configured, so a POLICY command still has nothing to switch to, and the
	// two arms refuse it identically.
	#[cfg(feature = "all_dram")]
	let built = PaperCache::<u64, BufferDRAM>::new(
		config.max_size,
		&[config.policy],
		config.policy,
	);

	#[cfg(not(feature = "all_dram"))]
	let built = PaperCache::<u64, TieredBuffer>::new(
		config.max_size,
		CacheTierSize::Bytes(config.fast_tier_size),
		config.policy,
	);

	let cache = match built {
		Ok(cache) => Arc::new(cache),
		Err(err) => {
			eprintln!("could not construct cache: {err}");
			process::exit(1);
		},
	};

	let listener = match TcpListener::bind(&config.bind) {
		Ok(listener) => listener,
		Err(err) => {
			eprintln!("could not bind {}: {err}", config.bind);
			process::exit(1);
		},
	};

	println!("paper-server listening on {}", config.bind);
	println!("  policy         {}", config.policy);
	println!("  max size       {} B", config.max_size);
	println!("  fast tier      {} B", config.fast_tier_size);
	println!("  auth           {}", if config.auth.is_some() { "required" } else { "disabled" });

	let connections = Arc::new(AtomicU64::new(0));
	let stats = Arc::new(SelfStats::new());

	// Optional periodic dump, so a long run leaves a trace without anyone
	// having to ask for it. A background thread rather than a signal handler:
	// this crate takes no signal dependency, and the reporter needs no
	// cooperation from the connection threads.
	if let Some(interval) = config.stats_interval {
		let stats = Arc::clone(&stats);
		let cache = Arc::clone(&cache);

		thread::spawn(move || loop {
			thread::sleep(Duration::from_secs(interval));
			eprint!("{}", render_self_stats(&stats, &cache));
		});
	}

	for stream in listener.incoming() {
		let stream = match stream {
			Ok(stream) => stream,
			Err(err) => {
				eprintln!("accept failed: {err}");
				continue;
			},
		};

		// Latency is the point of this binary, and Nagle would batch small
		// responses into 40ms stalls that have nothing to do with the cache.
		if let Err(err) = stream.set_nodelay(true) {
			eprintln!("could not set TCP_NODELAY: {err}");
		}

		let cache = Arc::clone(&cache);
		let connections = Arc::clone(&connections);
		let stats = Arc::clone(&stats);
		let auth = config.auth.clone();

		connections.fetch_add(1, Ordering::Relaxed);

		thread::spawn(move || {
			if let Err(err) = serve(stream, &cache, &stats, auth.as_deref()) {
				// A client hanging up mid-command is ordinary, not an error
				// worth reporting.
				if err.kind() != io::ErrorKind::UnexpectedEof
					&& err.kind() != io::ErrorKind::ConnectionReset
				{
					eprintln!("connection ended: {err}");
				}
			}

			connections.fetch_sub(1, Ordering::Relaxed);
		});
	}
}

/// One connection, start to finish. Thread per connection: the benchmark
/// drives `-c N` clients, so N connections means N server threads, and each
/// one blocks on its own socket.
fn serve(
	stream: TcpStream,
	cache: &Cache,
	stats: &SelfStats,
	auth: Option<&str>,
) -> io::Result<()> {
	let mut reader = BufReader::new(stream.try_clone()?);
	let mut writer = BufWriter::new(stream);

	// The server speaks first. `PaperClient::new` blocks reading this byte.
	write_bool(&mut writer, true)?;
	writer.flush()?;

	let mut authorized = auth.is_none();

	loop {
		let command = match read_u8(&mut reader) {
			Ok(command) => command,
			// A clean hangup between commands is how every client exits.
			Err(err) if err.kind() == io::ErrorKind::UnexpectedEof => return Ok(()),
			Err(err) => return Err(err),
		};

		// Every command except AUTH itself needs authorization first.
		if !authorized && command != command::AUTH {
			write_server_error(&mut writer, server_error::UNAUTHORIZED)?;
			writer.flush()?;
			continue;
		}

		match command {
			command::PING => {
				write_bool(&mut writer, true)?;
			},

			command::VERSION => {
				write_ok_buf(&mut writer, cache.version().as_bytes())?;
			},

			command::AUTH => {
				let token = read_string(&mut reader)?;

				match auth {
					Some(expected) if token == expected => {
						authorized = true;
						write_bool(&mut writer, true)?;
					},
					// No auth configured: accept any token rather than
					// rejecting a client that offers one.
					None => {
						authorized = true;
						write_bool(&mut writer, true)?;
					},
					Some(_) => write_server_error(&mut writer, server_error::UNAUTHORIZED)?,
				}
			},

			command::GET => {
				let key = read_string(&mut reader)?;

				match parse_key(&key) {
					Some(key) => {
						// Timed inline rather than through `timed` so the
						// outcome can pick the slot: hits and misses are
						// different operations and must not share an average.
						let started = Instant::now();
						let outcome = cache.get(&key);
						let nanos = started.elapsed().as_nanos() as u64;

						stats.record(
							if outcome.is_ok() { command::GET } else { SLOT_GET_MISS },
							nanos,
						);

						match outcome {
							Ok(value) => {
								write_ok_buf(&mut writer, &value)?;
							},
							Err(err) => write_cache_error(&mut writer, &err)?,
						}
					},
					None => write_cache_error(&mut writer, &CacheError::KeyNotFound)?,
				}
			},

			command::SET => {
				let key = read_string(&mut reader)?;
				let value = read_buf(&mut reader)?;
				let ttl = read_u32(&mut reader)?;

				// The wire has no null TTL; 0 is "no expiry".
				let ttl = if ttl == 0 { None } else { Some(ttl) };

				match parse_key(&key) {
					Some(key) => match timed(stats, command::SET, || cache.set(key, &value, ttl)) {
						Ok(()) => write_bool(&mut writer, true)?,
						Err(err) => write_cache_error(&mut writer, &err)?,
					},
					None => write_cache_error(&mut writer, &CacheError::Internal)?,
				}
			},

			command::DEL => {
				let key = read_string(&mut reader)?;

				match parse_key(&key) {
					Some(key) => match timed(stats, command::DEL, || cache.del(&key)) {
						Ok(()) => write_bool(&mut writer, true)?,
						Err(err) => write_cache_error(&mut writer, &err)?,
					},
					None => write_cache_error(&mut writer, &CacheError::KeyNotFound)?,
				}
			},

			command::HAS => {
				let key = read_string(&mut reader)?;
				let has = parse_key(&key)
					.is_some_and(|key| timed(stats, command::HAS, || cache.has(&key)));

				write_bool(&mut writer, true)?;
				write_bool(&mut writer, has)?;
			},

			command::PEEK => {
				let key = read_string(&mut reader)?;

				match parse_key(&key) {
					Some(key) => match timed(stats, command::PEEK, || cache.peek(&key)) {
						Ok(value) => {
							write_ok_buf(&mut writer, &value)?;
						},
						Err(err) => write_cache_error(&mut writer, &err)?,
					},
					None => write_cache_error(&mut writer, &CacheError::KeyNotFound)?,
				}
			},

			command::TTL => {
				let key = read_string(&mut reader)?;
				let ttl = read_u32(&mut reader)?;
				let ttl = if ttl == 0 { None } else { Some(ttl) };

				match parse_key(&key) {
					Some(key) => match timed(stats, command::TTL, || cache.ttl(&key, ttl)) {
						Ok(()) => write_bool(&mut writer, true)?,
						Err(err) => write_cache_error(&mut writer, &err)?,
					},
					None => write_cache_error(&mut writer, &CacheError::KeyNotFound)?,
				}
			},

			command::SIZE => {
				let key = read_string(&mut reader)?;

				match parse_key(&key) {
					Some(key) => match timed(stats, command::SIZE, || cache.size(&key)) {
						Ok(size) => {
							write_bool(&mut writer, true)?;
							write_u32(&mut writer, size as u32)?;
						},
						Err(err) => write_cache_error(&mut writer, &err)?,
					},
					None => write_cache_error(&mut writer, &CacheError::KeyNotFound)?,
				}
			},

			COMMAND_SELF_STATS => {
				write_ok_buf(&mut writer, render_self_stats(stats, cache).as_bytes())?;
			},

			command::WIPE => match cache.wipe() {
				Ok(()) => write_bool(&mut writer, true)?,
				Err(err) => write_cache_error(&mut writer, &err)?,
			},

			command::RESIZE => {
				let size = read_u64(&mut reader)?;

				match cache.resize(size) {
					Ok(()) => write_bool(&mut writer, true)?,
					Err(err) => write_cache_error(&mut writer, &err)?,
				}
			},

			command::POLICY => {
				// The policy arrives as a string; the tiered cache has no
				// runtime policy setter (its `policy` method lives on the
				// non-tiered impl, which `TieredBuffer` does not satisfy), so
				// this is refused rather than silently ignored.
				let _policy = read_string(&mut reader)?;
				write_cache_error(&mut writer, &CacheError::InvalidPolicy)?;
			},

			command::STATS => {
				// Layout, from `Command::parse_stats_stream`: three sizes,
				// three counters, the miss ratio, then a u32-counted list of
				// policy strings, the active policy, the auto flag, uptime.
				//
				// Caveat worth knowing before trusting this against the stock
				// client: it parses each policy string with ITS `PaperPolicy`,
				// which is upstream 1.10.1 and has never heard of this fork's
				// hybrid designs. `lru-hybrid` and friends will fail to parse
				// client-side. Nothing on the benchmark's hot path calls
				// STATS, so this is faithful to the protocol rather than
				// useful to that client.
				match cache.status() {
					Ok(status) => {
						write_bool(&mut writer, true)?;

						write_u64(&mut writer, status.max_size())?;
						write_u64(&mut writer, status.used_size())?;
						write_u64(&mut writer, status.num_objects())?;

						write_u64(&mut writer, status.total_gets())?;
						write_u64(&mut writer, status.total_sets())?;
						write_u64(&mut writer, status.total_dels())?;

						write_f64(&mut writer, status.miss_ratio())?;

						let policies = status.policies();
						write_u32(&mut writer, policies.len() as u32)?;

						for policy in policies {
							write_buf(&mut writer, policy.to_string().as_bytes())?;
						}

						write_buf(&mut writer, status.policy().to_string().as_bytes())?;
						write_bool(&mut writer, status.is_auto_policy())?;
						write_u64(&mut writer, status.uptime())?;
					},
					Err(err) => write_cache_error(&mut writer, &err)?,
				}
			},

			unknown => {
				// Requests are not length-prefixed, so the rest of this
				// command cannot be skipped and the stream position is lost.
				// Report it, then let the connection close.
				write_server_error(&mut writer, 1)?;
				writer.flush()?;

				return Err(io::Error::new(
					io::ErrorKind::InvalidData,
					format!("unknown command byte {unknown}; connection desynchronized"),
				));
			},
		}

		// One flush per command: the client is blocked on this response, so
		// buffering past it would deadlock rather than batch.
		writer.flush()?;
	}
}

/// The protocol's keys are strings; this cache is `u64` keyed. See the module
/// doc for why.
fn parse_key(key: &str) -> Option<u64> {
	key.parse::<u64>().ok()
}

// ---- wire reads ---------------------------------------------------------

fn read_u8<R: Read>(reader: &mut R) -> io::Result<u8> {
	let mut buf = [0u8; 1];
	reader.read_exact(&mut buf)?;
	Ok(buf[0])
}

fn read_u32<R: Read>(reader: &mut R) -> io::Result<u32> {
	let mut buf = [0u8; 4];
	reader.read_exact(&mut buf)?;
	Ok(u32::from_le_bytes(buf))
}

fn read_u64<R: Read>(reader: &mut R) -> io::Result<u64> {
	let mut buf = [0u8; 8];
	reader.read_exact(&mut buf)?;
	Ok(u64::from_le_bytes(buf))
}

fn read_buf<R: Read>(reader: &mut R) -> io::Result<Vec<u8>> {
	let size = read_u32(reader)? as usize;
	let mut buf = vec![0u8; size];
	reader.read_exact(&mut buf)?;
	Ok(buf)
}

fn read_string<R: Read>(reader: &mut R) -> io::Result<String> {
	let buf = read_buf(reader)?;
	String::from_utf8(buf).map_err(|_| io::Error::new(io::ErrorKind::InvalidData, "key is not utf-8"))
}

// ---- wire writes --------------------------------------------------------

fn write_bool<W: Write>(writer: &mut W, value: bool) -> io::Result<()> {
	writer.write_all(&[if value { TRUE_INDICATOR } else { FALSE_INDICATOR }])
}

fn write_u32<W: Write>(writer: &mut W, value: u32) -> io::Result<()> {
	writer.write_all(&value.to_le_bytes())
}

fn write_u64<W: Write>(writer: &mut W, value: u64) -> io::Result<()> {
	writer.write_all(&value.to_le_bytes())
}

fn write_f64<W: Write>(writer: &mut W, value: f64) -> io::Result<()> {
	writer.write_all(&value.to_le_bytes())
}

fn write_buf<W: Write>(writer: &mut W, value: &[u8]) -> io::Result<()> {
	write_u32(writer, value.len() as u32)?;
	writer.write_all(value)
}

/// `[ok][len][payload]` in ONE vectored write.
///
/// The obvious spelling -- five bytes of prefix through the `BufWriter`, then
/// `write_all` for the payload -- costs an extra TCP segment on every payload
/// that does not fit the buffer. `BufWriter` passes a write larger than its
/// capacity straight through to the socket, so the prefix flushes on its own
/// first and the payload follows as a second `sendto`.
///
/// That is not a rounding error. Profiled against memcached 1.6.45 on the same
/// box at a 16,439 B mean value: 1.74 `sendto` per hit against its 1.00
/// `sendmsg`, the client doing 1.98 `read`s instead of 1.00, +27% TCP segments
/// in both directions. It switches on at exactly 8,188 bytes, which is the
/// 8 KiB buffer capacity minus this five-byte prefix -- at 8,187 B this server
/// measured 618 ns FASTER than memcached, at 8,188 B 5,665 ns slower, while
/// memcached was flat across the same byte. 73% of that trace's hits are above
/// the boundary, so it was worth about 4.4 us per operation.
///
/// `write_vectored` on a `BufWriter` wrapping a `TcpStream` flushes and then
/// forwards to `writev`, because `TcpStream::is_write_vectored` is true, so the
/// whole response leaves as one syscall and one segment. This is the shape
/// memcached has always had: one `sendmsg` over an iovec pointing straight at
/// the stored bytes.
fn write_ok_buf<W: Write>(writer: &mut W, payload: &[u8]) -> io::Result<()> {
	let mut prefix = [0u8; 5];
	prefix[0] = TRUE_INDICATOR;
	prefix[1..].copy_from_slice(&(payload.len() as u32).to_le_bytes());

	let mut slices = [IoSlice::new(&prefix), IoSlice::new(payload)];
	let mut bufs: &mut [IoSlice<'_>] = &mut slices;

	// `write_vectored` is allowed to return short, so advance past whatever
	// went out and go again. `advance_slices` drops fully written slices and
	// trims the partial one.
	while !bufs.is_empty() {
		match writer.write_vectored(bufs) {
			Ok(0) => return Err(io::ErrorKind::WriteZero.into()),
			Ok(n) => IoSlice::advance_slices(&mut bufs, n),
			Err(ref err) if err.kind() == io::ErrorKind::Interrupted => {},
			Err(err) => return Err(err),
		}
	}

	Ok(())
}

fn write_cache_error<W: Write>(writer: &mut W, err: &CacheError) -> io::Result<()> {
	write_bool(writer, false)?;
	writer.write_all(&[0u8])?; // 0 => a cache error code follows
	writer.write_all(&[cache_error_code(err)])
}

fn write_server_error<W: Write>(writer: &mut W, code: u8) -> io::Result<()> {
	write_bool(writer, false)?;
	writer.write_all(&[code])
}

// ---- configuration ------------------------------------------------------

struct Config {
	bind: String,
	max_size: CacheSize,
	fast_tier_size: CacheSize,
	policy: PaperPolicy,
	auth: Option<String>,
	stats_interval: Option<u64>,
}

impl Config {
	/// Hand-rolled rather than clap: this crate is a library and does not
	/// depend on an argument parser, and adding one for six flags would put it
	/// in the dependency tree of everything that links the cache.
	fn from_args() -> Result<Self, String> {
		let mut bind = "127.0.0.1:3145".to_string();
		let mut max_size: CacheSize = 24 * 1024 * 1024 * 1024;
		let mut fast_tier_size: CacheSize = 4 * 1024 * 1024 * 1024;
		// `lru-hybrid` and the other 18 non-compact designs were dropped in
		// f1f5d88; the compact one is the only LRU hybrid left.
		let mut policy_str = "lru-compact-hybrid".to_string();
		let mut auth = None;
		let mut stats_interval = None;

		let mut args = std::env::args().skip(1);

		while let Some(arg) = args.next() {
			let mut value = || {
				args.next()
					.ok_or_else(|| format!("{arg} requires a value"))
			};

			match arg.as_str() {
				"--bind" => bind = value()?,
				"--max-size" => {
					max_size = value()?
						.parse()
						.map_err(|_| "--max-size must be a byte count".to_string())?
				},
				"--fast-tier-size" => {
					fast_tier_size = value()?
						.parse()
						.map_err(|_| "--fast-tier-size must be a byte count".to_string())?
				},
				"--policy" => policy_str = value()?,
				"--auth" => auth = Some(value()?),
				"--stats-interval" => {
					stats_interval = Some(value()?.parse().map_err(|_| {
						"--stats-interval must be a whole number of seconds".to_string()
					})?)
				},
				"-h" | "--help" => {
					println!("{USAGE}");
					process::exit(0);
				},
				other => return Err(format!("unknown argument {other}\n\n{USAGE}")),
			}
		}

		let policy = policy_str
			.parse::<PaperPolicy>()
			.map_err(|_| format!("invalid policy {policy_str:?}"))?;

		// A flat design is allowed, and is the ONLY honest all-DRAM baseline.
		//
		// Setting a hybrid policy's fast tier equal to its max size does NOT
		// give a DRAM-only cache. The fast tier is drained continuously to
		// drain_target::ratio() of its capacity while eviction only fires at
		// 1.00, so the slow tier becomes a victim queue holding the last 2%
		// and every eviction is routed through a demotion. Measured on
		// low_alpha_cold at 6 GB with lru-compact-hybrid: 119 MB and 6,684
		// objects left on node 1, and demotions 2,515,180 == evictions
		// 2,484,317 + promotions 30,863, an exact conservation identity.
		//
		// So the guard is a warning, not an error. PaperCache::new accepts any
		// policy; the tier report just shows an empty slow tier for a flat one.
		#[cfg(not(feature = "all_dram"))]
		if !policy.is_hybrid() {
			return Err(format!(
				"{policy_str:?} is not a hybrid design; build with --features \
				all_dram to serve it as a flat DRAM cache",
			));
		}

		#[cfg(feature = "all_dram")]
		if policy.is_hybrid() {
			return Err(format!(
				"{policy_str:?} is a hybrid design; this all_dram build has no \
				slow tier to place anything in",
			));
		}

		Ok(Config { bind, max_size, fast_tier_size, policy, auth, stats_interval })
	}
}

const USAGE: &str = "\
paper-server -- the PaperCache tiered cache, served over TCP

USAGE:
    paper-server [OPTIONS]

OPTIONS:
    --bind <ADDR:PORT>     Address to listen on [default: 127.0.0.1:3145]
                           Use 0.0.0.0:3145 to accept connections from other
                           machines.
    --max-size <BYTES>     Overall cache capacity [default: 25769803776 (24 GiB)]
    --fast-tier-size <B>   Fast (DRAM) tier capacity [default: 4294967296 (4 GiB)]
    --policy <POLICY>      Hybrid policy string, e.g. lru-compact-hybrid,
                           s3-fifo-faithful-compact-hybrid-0.1
                           [default: lru-compact-hybrid]
    --auth <TOKEN>         Require this token via the AUTH command
    --stats-interval <S>   Print server-side cache latency to stderr every S
                           seconds. Command byte 200 returns the same report
                           on demand.
    -h, --help             Print this help
";
