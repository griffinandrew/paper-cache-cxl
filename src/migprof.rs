/*
 * Copyright (c) Griffin Andrew
 *
 * This source code is licensed under the GNU AGPLv3 license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! What slows the migration consumers down: a profile of where their time
//! goes, the policy worker's, and the client GETs' by the tier they were
//! served from. PROFILING ONLY, and off unless `MIG_PROFILE=1`: off, every hook
//! below is one load of a `LazyLock<bool>`.
//!
//! The questions it answers, and the field that answers each:
//!
//!   * Is one consumer's copy rate the limit? `copy_bytes / copy_ns` per
//!     consumer and direction (`MIGPROF consumer=..`) against the slow tier's
//!     peak. `copy_ns` is `migrated_to` -- the destination allocation AND the
//!     copy -- so a slow allocator shows up here too; `alloc` is not split out.
//!   * Do the consumers wait on clients? `lookup_ns` (the shard READ guard, for
//!     the snapshot) and `swap_ns` (the shard WRITE guard, compare and swap).
//!   * How much copying is thrown away? `superseded`: a `set` replaced the
//!     value mid-copy, the copy is dropped. Nothing retries it -- the entry is
//!     finished; the stack re-emits only if it decides to again.
//!   * Slow, or starved? `busy_ns` per consumer against the receive idle the
//!     `MIGSTATS mq_*` line already reports, and the policy worker's own busy
//!     fraction with its event backlog and the queue depth it left behind
//!     (`MIGPROF worker ..`): a worker near 100% busy with a backlog while the
//!     queue sits empty is starvation, not slow consumers.
//!   * Do migrations slow the clients? `MIGPROF get_fast` / `get_slow`: a
//!     latency histogram of `get` by served tier. Compare a window with
//!     migrations running against one with them paused
//!     (`PaperCache::set_migrations_paused`).
//!
//! # The emulated slow tier
//!
//! `EMU_SLOW=1` makes every slow-tier access -- a migration's copy (a
//! promotion reads the slow tier, a demotion writes it), a GET served slow, a
//! set built slow -- also pay a modelled cost, spun on the calling thread
//! (a core stalled on far memory is not free either):
//!
//!   done = max(link_done, now + bytes / EMU_SLOW_THREAD_GBPS) + EMU_SLOW_LAT_NS
//!
//! where `link_done` is a FCFS reservation on ONE shared link of
//! `EMU_SLOW_LINK_GBPS`, so clients and migrations contend for it. Writes cost
//! `EMU_SLOW_WRITE_FACTOR` times their bytes. It is a model for machines with
//! no slow node: it tells you whether the instrumentation separates the
//! causes, not what a real CXL device does. Never set it on the CXL machine.

use std::sync::LazyLock;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::cell::Cell;
use std::time::Instant;

use crate::worker::Tier;

/// `MIG_PROFILE=1`, read once.
pub(crate) static ENABLED: LazyLock<bool> = LazyLock::new(|| env_flag("MIG_PROFILE"));

#[inline]
pub(crate) fn enabled() -> bool {
	*ENABLED
}

fn env_flag(name: &str) -> bool {
	std::env::var(name).is_ok_and(|v| v == "1" || v.eq_ignore_ascii_case("true"))
}

fn env_f64(name: &str, default: f64) -> f64 {
	std::env::var(name).ok().and_then(|v| v.parse().ok()).unwrap_or(default)
}

/// Process-wide clock origin for the emulated link.
static ORIGIN: LazyLock<Instant> = LazyLock::new(Instant::now);

#[inline]
fn now_ns() -> u64 {
	ORIGIN.elapsed().as_nanos() as u64
}

// ---------------------------------------------------------------------------
// the emulated slow tier
// ---------------------------------------------------------------------------

struct Emu {
	/// Bytes per ns: the shared link, and one thread alone.
	link: f64,
	thread: f64,
	lat_ns: u64,
	write_factor: f64,
}

static EMU: LazyLock<Option<Emu>> = LazyLock::new(|| {
	if !env_flag("EMU_SLOW") {
		return None;
	}

	// GB/s is bytes per ns.
	let emu = Emu {
		link: env_f64("EMU_SLOW_LINK_GBPS", 16.0),
		thread: env_f64("EMU_SLOW_THREAD_GBPS", 4.0),
		lat_ns: env_f64("EMU_SLOW_LAT_NS", 250.0) as u64,
		write_factor: env_f64("EMU_SLOW_WRITE_FACTOR", 1.5),
	};

	eprintln!(
		"MIGPROF emu_slow=1 link_gbps={} thread_gbps={} lat_ns={} write_factor={}",
		emu.link, emu.thread, emu.lat_ns, emu.write_factor,
	);

	Some(emu)
});

/// The emulated link's next free instant, in `now_ns` time.
static LINK_FREE: AtomicU64 = AtomicU64::new(0);

/// Emulated link time waited for, process-wide: queueing behind other users of
/// the link, beyond this access's own service time.
pub(crate) static EMU_QUEUE_NS: AtomicU64 = AtomicU64::new(0);

/// Charges one slow-tier access of `bytes` (see the module doc). Nothing
/// without `EMU_SLOW=1`.
#[inline]
pub(crate) fn emu_charge(bytes: usize, write: bool) {
	if let Some(emu) = EMU.as_ref() {
		emu_charge_slow(emu, bytes, write);
	}
}

#[cold]
fn emu_charge_slow(emu: &Emu, bytes: usize, write: bool) {
	let now = now_ns();
	let cost = bytes as f64 * if write { emu.write_factor } else { 1.0 };
	let service = (cost / emu.link) as u64;

	let mut free = LINK_FREE.load(Ordering::Relaxed);

	let start = loop {
		let start = free.max(now);

		match LINK_FREE.compare_exchange_weak(free, start + service, Ordering::Relaxed, Ordering::Relaxed) {
			Ok(_) => break start,
			Err(current) => free = current,
		}
	};

	EMU_QUEUE_NS.fetch_add(start - now, Ordering::Relaxed);

	let done = (start + service).max(now + (cost / emu.thread) as u64) + emu.lat_ns;

	while now_ns() < done {
		std::hint::spin_loop();
	}
}

// ---------------------------------------------------------------------------
// histograms
// ---------------------------------------------------------------------------

/// Four sub-buckets per octave of ns, up to 2^32 ns.
pub(crate) const HB: usize = 128;

fn hbucket(ns: u64) -> usize {
	if ns < 4 {
		return ns as usize;
	}

	let octave = 63 - ns.leading_zeros() as usize;
	let sub = ((ns >> (octave - 2)) & 3) as usize;

	(octave * 4 + sub).min(HB - 1)
}

/// The lower edge of bucket `b`.
fn hbucket_floor(b: usize) -> u64 {
	if b < 4 {
		return b as u64;
	}

	let octave = b / 4;
	let sub = (b % 4) as u64;

	(1u64 << octave) + (sub << (octave - 2))
}

/// Client threads write these, so they are sharded by thread.
const HSHARDS: usize = 8;

#[repr(align(128))]
pub(crate) struct Hist {
	buckets: [AtomicU64; HB],
	total_ns: AtomicU64,
}

impl Default for Hist {
	fn default() -> Self {
		Hist { buckets: std::array::from_fn(|_| AtomicU64::new(0)), total_ns: AtomicU64::new(0) }
	}
}

impl Hist {
	fn record(&self, ns: u64) {
		self.buckets[hbucket(ns)].fetch_add(1, Ordering::Relaxed);
		self.total_ns.fetch_add(ns, Ordering::Relaxed);
	}

	fn reset(&self) {
		for bucket in &self.buckets {
			bucket.store(0, Ordering::Relaxed);
		}

		self.total_ns.store(0, Ordering::Relaxed);
	}
}

#[derive(Default)]
pub(crate) struct ShardedHist {
	shards: [Hist; HSHARDS],
}

thread_local! {
	static HSHARD: Cell<usize> = const { Cell::new(usize::MAX) };
	/// The consumer slot of this thread, set once by a migration consumer;
	/// `INLINE_SLOT` everywhere else (the policy worker's inline path).
	static SLOT: Cell<usize> = const { Cell::new(INLINE_SLOT) };
}

static NEXT_HSHARD: AtomicU64 = AtomicU64::new(0);

impl ShardedHist {
	fn record(&self, ns: u64) {
		let shard = HSHARD.with(|cell| {
			if cell.get() == usize::MAX {
				cell.set(NEXT_HSHARD.fetch_add(1, Ordering::Relaxed) as usize % HSHARDS);
			}

			cell.get()
		});

		self.shards[shard].record(ns);
	}

	fn reset(&self) {
		self.shards.iter().for_each(Hist::reset);
	}

	/// `(count, mean_ns, p50, p90, p99, p999)`; percentiles are bucket floors.
	fn summary(&self) -> (u64, u64, u64, u64, u64, u64) {
		let mut merged = [0u64; HB];
		let mut total = 0u64;

		for shard in &self.shards {
			for (b, bucket) in shard.buckets.iter().enumerate() {
				merged[b] += bucket.load(Ordering::Relaxed);
			}

			total += shard.total_ns.load(Ordering::Relaxed);
		}

		let count: u64 = merged.iter().sum();

		let pct = |q: f64| -> u64 {
			if count == 0 {
				return 0;
			}

			let target = ((count as f64) * q).ceil().max(1.0) as u64;
			let mut seen = 0;

			for (b, n) in merged.iter().enumerate() {
				seen += n;

				if seen >= target {
					return hbucket_floor(b);
				}
			}

			hbucket_floor(HB - 1)
		};

		(count, total.checked_div(count).unwrap_or(0), pct(0.50), pct(0.90), pct(0.99), pct(0.999))
	}
}

// ---------------------------------------------------------------------------
// the profile
// ---------------------------------------------------------------------------

/// Consumer slots; the last is the policy worker's inline path
/// (`MIGRATION_QUEUE_THREADS=0`) and anything else not a consumer.
pub(crate) const SLOTS: usize = 16;
const INLINE_SLOT: usize = SLOTS - 1;

#[derive(Default)]
pub(crate) struct Dir {
	/// Copies built (the swap attempted), their bytes, and where the time went.
	n: AtomicU64,
	bytes: AtomicU64,
	lookup_ns: AtomicU64,
	copy_ns: AtomicU64,
	swap_ns: AtomicU64,
	/// Built and thrown away: the swap found the value replaced.
	superseded: AtomicU64,
	superseded_bytes: AtomicU64,
	/// Swap time histogram: the shard write guard's wait shows up in its tail.
	swap_hist_max_ns: AtomicU64,
}

#[derive(Default)]
#[repr(align(128))]
pub(crate) struct Slot {
	/// [Fast, Slow]: promotions, demotions.
	dir: [Dir; 2],
	/// Time from an item's receipt to its disposition, every item.
	busy_ns: AtomicU64,
	items: AtomicU64,
	/// Time held by `set_migrations_paused`.
	paused_ns: AtomicU64,
}

#[derive(Default)]
#[repr(align(128))]
pub(crate) struct WorkerProf {
	passes: AtomicU64,
	/// A pass, from its first event to the end of its instrumentation: work.
	busy_ns: AtomicU64,
	/// `delay_event_loop`: parked or sleeping.
	idle_ns: AtomicU64,
	events: AtomicU64,
	/// The event channel's length at each pass's start.
	backlog_sum: AtomicU64,
	backlog_max: AtomicU64,
	/// Migrations pending (queued, not finished) at each pass's end.
	mq_depth_sum: AtomicU64,
	/// Passes that started with >= `STARVE_BACKLOG` events waiting AND found
	/// the migration queue empty: the consumers had nothing while the worker
	/// was behind.
	starved_passes: AtomicU64,
}

const STARVE_BACKLOG: u64 = 64;

/// One cache's profile, in its `MigStats`.
#[derive(Default)]
pub struct Prof {
	slots: [Slot; SLOTS],
	worker: WorkerProf,
	get_fast: ShardedHist,
	get_slow: ShardedHist,
	paused: AtomicBool,
	/// `t_ms` of the last reset, so a report can give rates.
	since_ns: AtomicU64,
}

fn dir_index(tier: Tier) -> usize {
	match tier {
		Tier::Fast => 0,
		Tier::Slow => 1,
	}
}

/// What a consumer thread calls once, so `apply_migration_with` knows its slot.
pub(crate) fn set_consumer_slot(index: usize) {
	SLOT.with(|cell| cell.set(index % INLINE_SLOT));
}

fn slot_index() -> usize {
	SLOT.with(Cell::get)
}

/// A span's start, only when profiling.
#[inline]
pub(crate) fn start() -> Option<Instant> {
	enabled().then(Instant::now)
}

#[inline]
fn ns_since(start: Instant) -> u64 {
	start.elapsed().as_nanos() as u64
}

impl Prof {
	/// A migration's three phases, as `apply_migration_with` timed them:
	/// `swapped` false is a superseded copy.
	pub(crate) fn record_migration(&self, tier: Tier, bytes: usize, lookup_ns: u64, copy_ns: u64, swap_ns: u64, swapped: bool) {
		let dir = &self.slots[slot_index()].dir[dir_index(tier)];

		dir.n.fetch_add(1, Ordering::Relaxed);
		dir.bytes.fetch_add(bytes as u64, Ordering::Relaxed);
		dir.lookup_ns.fetch_add(lookup_ns, Ordering::Relaxed);
		dir.copy_ns.fetch_add(copy_ns, Ordering::Relaxed);
		dir.swap_ns.fetch_add(swap_ns, Ordering::Relaxed);
		dir.swap_hist_max_ns.fetch_max(swap_ns, Ordering::Relaxed);

		if !swapped {
			dir.superseded.fetch_add(1, Ordering::Relaxed);
			dir.superseded_bytes.fetch_add(bytes as u64, Ordering::Relaxed);
		}
	}

	pub(crate) fn record_item(&self, start: Option<Instant>) {
		if let Some(start) = start {
			let slot = &self.slots[slot_index()];
			slot.busy_ns.fetch_add(ns_since(start), Ordering::Relaxed);
			slot.items.fetch_add(1, Ordering::Relaxed);
		}
	}

	/// Holds the calling consumer while the cache's migrations are paused.
	pub(crate) fn wait_while_paused(&self) {
		if !self.paused.load(Ordering::Acquire) {
			return;
		}

		let start = Instant::now();

		while self.paused.load(Ordering::Acquire) {
			std::thread::sleep(std::time::Duration::from_micros(100));
		}

		self.slots[slot_index()].paused_ns.fetch_add(ns_since(start), Ordering::Relaxed);
	}

	pub fn set_paused(&self, paused: bool) {
		self.paused.store(paused, Ordering::Release);
	}

	pub(crate) fn record_get(&self, tier: Tier, start: Option<Instant>) {
		if let Some(start) = start {
			let ns = ns_since(start);

			match tier {
				Tier::Fast => self.get_fast.record(ns),
				Tier::Slow => self.get_slow.record(ns),
			}
		}
	}

	pub(crate) fn record_pass(&self, busy: Option<Instant>, events: usize, backlog: usize, mq_depth: u64) {
		let Some(busy) = busy else { return };
		let worker = &self.worker;
		let backlog = backlog as u64;

		worker.passes.fetch_add(1, Ordering::Relaxed);
		worker.busy_ns.fetch_add(ns_since(busy), Ordering::Relaxed);
		worker.events.fetch_add(events as u64, Ordering::Relaxed);
		worker.backlog_sum.fetch_add(backlog, Ordering::Relaxed);
		worker.backlog_max.fetch_max(backlog, Ordering::Relaxed);
		worker.mq_depth_sum.fetch_add(mq_depth, Ordering::Relaxed);

		if backlog >= STARVE_BACKLOG && mq_depth == 0 {
			worker.starved_passes.fetch_add(1, Ordering::Relaxed);
		}
	}

	pub(crate) fn record_idle(&self, idle: Option<Instant>) {
		if let Some(idle) = idle {
			self.worker.idle_ns.fetch_add(ns_since(idle), Ordering::Relaxed);
		}
	}

	/// Zeroes every counter: the start of a measurement window.
	pub fn reset(&self) {
		let zero = |a: &AtomicU64| a.store(0, Ordering::Relaxed);

		for slot in &self.slots {
			for dir in &slot.dir {
				for a in [&dir.n, &dir.bytes, &dir.lookup_ns, &dir.copy_ns, &dir.swap_ns, &dir.superseded, &dir.superseded_bytes, &dir.swap_hist_max_ns] {
					zero(a);
				}
			}

			zero(&slot.busy_ns);
			zero(&slot.items);
			zero(&slot.paused_ns);
		}

		let w = &self.worker;

		for a in [&w.passes, &w.busy_ns, &w.idle_ns, &w.events, &w.backlog_sum, &w.backlog_max, &w.mq_depth_sum, &w.starved_passes] {
			zero(a);
		}

		self.get_fast.reset();
		self.get_slow.reset();
		EMU_QUEUE_NS.store(0, Ordering::Relaxed);
		self.since_ns.store(now_ns(), Ordering::Relaxed);
	}

	/// The `MIGPROF` lines; none unless `MIG_PROFILE=1`. `key=value`, `t_ms`
	/// last, like `MIGSTATS`. Rates are GB/s (bytes per ns); `pct` fields are
	/// percent of the window (`window_ms`, since the last reset or start).
	pub fn lines(&self, t_ms: u64) -> Vec<String> {
		if !enabled() {
			return Vec::new();
		}

		let load = |a: &AtomicU64| a.load(Ordering::Relaxed);
		let window_ns = now_ns().saturating_sub(load(&self.since_ns)).max(1);
		let gbps = |bytes: u64, ns: u64| if ns == 0 { 0.0 } else { bytes as f64 / ns as f64 };
		let per = |ns: u64, n: u64| ns.checked_div(n).unwrap_or(0);
		let mut lines = Vec::new();

		lines.push(format!("MIGPROF window_ms={} emu_queue_ms={} t_ms={t_ms}", window_ns / 1_000_000, load(&EMU_QUEUE_NS) / 1_000_000));

		for (index, slot) in self.slots.iter().enumerate() {
			let items = load(&slot.items);

			if items == 0 && slot.dir.iter().all(|dir| load(&dir.n) == 0) {
				continue;
			}

			let name = if index == INLINE_SLOT { "inline".to_string() } else { index.to_string() };
			let busy = load(&slot.busy_ns);

			for (dir, label) in slot.dir.iter().zip(["promote", "demote"]) {
				let n = load(&dir.n);

				if n == 0 {
					continue;
				}

				let (bytes, copy) = (load(&dir.bytes), load(&dir.copy_ns));

				lines.push(format!(
					"MIGPROF consumer={name} dir={label} n={n} bytes={bytes} copy_gbps={:.2} lookup_ns_avg={} copy_ns_avg={} swap_ns_avg={} swap_ns_max={} superseded={} superseded_bytes={} t_ms={t_ms}",
					gbps(bytes, copy),
					per(load(&dir.lookup_ns), n),
					per(copy, n),
					per(load(&dir.swap_ns), n),
					load(&dir.swap_hist_max_ns),
					load(&dir.superseded),
					load(&dir.superseded_bytes),
				));
			}

			let moved: u64 = slot.dir.iter().map(|dir| load(&dir.bytes)).sum();

			lines.push(format!(
				"MIGPROF consumer={name} items={items} busy_pct={:.1} paused_pct={:.1} effective_gbps={:.2} t_ms={t_ms}",
				100.0 * busy as f64 / window_ns as f64,
				100.0 * load(&slot.paused_ns) as f64 / window_ns as f64,
				gbps(moved, busy),
			));
		}

		let w = &self.worker;
		let passes = load(&w.passes);
		let (busy, idle) = (load(&w.busy_ns), load(&w.idle_ns));

		lines.push(format!(
			"MIGPROF worker passes={passes} busy_pct={:.1} events={} backlog_avg={} backlog_max={} mq_depth_avg={} starved_passes={} t_ms={t_ms}",
			100.0 * busy as f64 / (busy + idle).max(1) as f64,
			load(&w.events),
			per(load(&w.backlog_sum), passes),
			load(&w.backlog_max),
			per(load(&w.mq_depth_sum), passes),
			load(&w.starved_passes),
		));

		for (label, hist) in [("get_fast", &self.get_fast), ("get_slow", &self.get_slow)] {
			let (n, mean, p50, p90, p99, p999) = hist.summary();

			lines.push(format!("MIGPROF {label} n={n} mean_ns={mean} p50_ns={p50} p90_ns={p90} p99_ns={p99} p999_ns={p999} t_ms={t_ms}"));
		}

		lines
	}
}

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn buckets_are_monotone_and_floors_round_trip() {
		let mut last = 0;

		for ns in [0u64, 1, 3, 4, 5, 7, 8, 100, 1_000, 65_535, 1 << 20, 1 << 31] {
			let b = hbucket(ns);
			assert!(b >= last, "{ns}");
			assert!(hbucket_floor(b) <= ns, "{ns}: floor {}", hbucket_floor(b));
			assert_eq!(hbucket(hbucket_floor(b)), b, "{ns}");
			last = b;
		}
	}
}
