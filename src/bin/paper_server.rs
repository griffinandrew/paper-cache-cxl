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
	io::{self, BufReader, BufWriter, Read, Write},
	net::{TcpListener, TcpStream},
	process,
	sync::{
		atomic::{AtomicU64, Ordering},
		Arc,
	},
	thread,
};

use paper_cache::{
	CacheError, CacheSize, CacheTierSize, PaperCache, PaperPolicy, TieredBuffer,
};

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

type Cache = PaperCache<u64, TieredBuffer>;

fn main() {
	let config = match Config::from_args() {
		Ok(config) => config,
		Err(message) => {
			eprintln!("{message}");
			process::exit(2);
		},
	};

	let cache = match PaperCache::<u64, TieredBuffer>::new(
		config.max_size,
		CacheTierSize::Bytes(config.fast_tier_size),
		config.policy,
	) {
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
		let auth = config.auth.clone();

		connections.fetch_add(1, Ordering::Relaxed);

		thread::spawn(move || {
			if let Err(err) = serve(stream, &cache, auth.as_deref()) {
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
				write_bool(&mut writer, true)?;
				write_buf(&mut writer, cache.version().as_bytes())?;
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
					Some(key) => match cache.get(&key) {
						Ok(value) => {
							write_bool(&mut writer, true)?;
							write_buf(&mut writer, &value)?;
						},
						Err(err) => write_cache_error(&mut writer, &err)?,
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
					Some(key) => match cache.set(key, &value, ttl) {
						Ok(()) => write_bool(&mut writer, true)?,
						Err(err) => write_cache_error(&mut writer, &err)?,
					},
					None => write_cache_error(&mut writer, &CacheError::Internal)?,
				}
			},

			command::DEL => {
				let key = read_string(&mut reader)?;

				match parse_key(&key) {
					Some(key) => match cache.del(&key) {
						Ok(()) => write_bool(&mut writer, true)?,
						Err(err) => write_cache_error(&mut writer, &err)?,
					},
					None => write_cache_error(&mut writer, &CacheError::KeyNotFound)?,
				}
			},

			command::HAS => {
				let key = read_string(&mut reader)?;
				let has = parse_key(&key).is_some_and(|key| cache.has(&key));

				write_bool(&mut writer, true)?;
				write_bool(&mut writer, has)?;
			},

			command::PEEK => {
				let key = read_string(&mut reader)?;

				match parse_key(&key) {
					Some(key) => match cache.peek(&key) {
						Ok(value) => {
							write_bool(&mut writer, true)?;
							write_buf(&mut writer, value.as_ref().as_ref())?;
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
					Some(key) => match cache.ttl(&key, ttl) {
						Ok(()) => write_bool(&mut writer, true)?,
						Err(err) => write_cache_error(&mut writer, &err)?,
					},
					None => write_cache_error(&mut writer, &CacheError::KeyNotFound)?,
				}
			},

			command::SIZE => {
				let key = read_string(&mut reader)?;

				match parse_key(&key) {
					Some(key) => match cache.size(&key) {
						Ok(size) => {
							write_bool(&mut writer, true)?;
							write_u32(&mut writer, size as u32)?;
						},
						Err(err) => write_cache_error(&mut writer, &err)?,
					},
					None => write_cache_error(&mut writer, &CacheError::KeyNotFound)?,
				}
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
}

impl Config {
	/// Hand-rolled rather than clap: this crate is a library and does not
	/// depend on an argument parser, and adding one for six flags would put it
	/// in the dependency tree of everything that links the cache.
	fn from_args() -> Result<Self, String> {
		let mut bind = "127.0.0.1:3145".to_string();
		let mut max_size: CacheSize = 24 * 1024 * 1024 * 1024;
		let mut fast_tier_size: CacheSize = 4 * 1024 * 1024 * 1024;
		let mut policy_str = "lru-hybrid".to_string();
		let mut auth = None;

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

		if !policy.is_hybrid() {
			return Err(format!(
				"{policy_str:?} is not a hybrid design; this server serves the tiered cache",
			));
		}

		Ok(Config { bind, max_size, fast_tier_size, policy, auth })
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
    --policy <POLICY>      Hybrid policy string, e.g. lru-hybrid,
                           s3-fifo-hybrid-0.1 [default: lru-hybrid]
    --auth <TOKEN>         Require this token via the AUTH command
    -h, --help             Print this help
";
