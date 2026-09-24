/*
 * Copyright (c) Kia Shakiba
 *
 * This source code is licensed under the GNU AGPLv3 license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! PHYS_FAST: the bytes PHYSICALLY allocated in the fast tier's value pool,
//! counted at the allocation and the free rather than at a policy decision.
//!
//! ## Why this exists
//!
//! Every tier gauge the cache reports is INTENT. A stack moves a key's bytes
//! out of `fast_used` the moment it DECIDES to demote it, and the copy that
//! makes it true runs later, on a migration consumer (`migration_queue`'s doc:
//! completion-time accounting would make a settle drain the whole tier before
//! seeing its own decisions register). Nothing measured what is actually
//! sitting in DRAM, and the gap is not small: on cluster35 merged CLOCK every
//! stack gauge read 5.26-5.28e9 B (`fast_dram_total` ~5.27 GB for a 5,120 MiB
//! tier) while node 0 physically peaked at 14,167 MiB (the matrix's numastat
//! log). This counter is the measurement the backpressure plan's gate (S5)
//! will act on. In this step nothing reads it to make a decision.
//!
//! ## What P counts
//!
//! The sum, over every LIVE fast-tier value allocation in the process, of
//! [`value_charge`] of that allocation: an allocation is charged once when
//! `TieredValue::new_in` builds it in `Tier::Fast` and refunded once when the
//! drop that frees it runs (its last handle's). So P includes everything
//! physically in the value pool, whatever the stack believes about it:
//!
//!   * unsettled sets -- a value built in DRAM before the worker has seen it;
//!   * pending demotions -- still in DRAM, already out of `fast_used`;
//!   * in-flight migration copies -- a promotion's DRAM copy is charged by
//!     `migrated_to` on the CONSUMER thread before the swap, and a demotion's
//!     old DRAM copy stays charged until the swap drops it;
//!   * superseded copies -- a copy whose swap lost to a `set` is refunded when
//!     the consumer drops it, an overwritten value when its handle drops;
//!   * snapshots held by readers -- a `get` that lifted the value out from
//!     under the shard guard keeps it charged until its copy finishes.
//!
//! It does NOT count the DRAM value headers, the object map, the stacks or
//! anything else the per-object reservation models -- that is `L * omega` in
//! the budget arithmetic -- nor allocator slack and retained pages (process
//! RSS, printed beside it on the MEMTS line).
//!
//! ## The unit: the stacks' own
//!
//! [`value_charge`] is `object::overhead::resident_object_bytes::<K>(len)`,
//! THE function the stacks' per-object figure comes from, so at quiescence P
//! equals the sum of the stacks' `fast_used`:
//!
//! ```text
//!   layout        allocation          P charges              the stacks charge
//!   split         len.max(1) @ 8      nallocx(len)           DashMap: base_size - dram_resident_size
//!   value.rs                                                   = nallocx(len); merged: Slot::migrating,
//!                                                              the same call
//!   thin_header   offset+len @ 8      nallocx(offset+len)    the same, in both stores: dram_resident_size
//!   value_thin                                                 keeps the split arm ON PURPOSE, so the
//!                                                              difference is exactly the item
//!   fused_value   offset+len @ 8      nallocx(offset+len)    merged: the same; DashMap: base_size,
//!   value_fused                                                since dram_resident_size is 0 there
//! ```
//!
//! In every build the full suites run (split and thin_header; DashMap,
//! hashbrown and merged) the charge IS the stacks' figure. The one pair where
//! it is not is `fused_value` with a DashMap-family store: that stack charges
//! `base_size`, which re-adds the key and the 4-byte expiry (and
//! `get_ttl_overhead()` for a TTL'd object) although both are inside the item
//! -- `base_size`'s own doc says so, and `get_policy_overhead` takes them back
//! off for the FLAT budget only. P does not copy that. A charge has to be a
//! function of the allocation alone, or charge and refund stop pairing (a
//! `ttl()` can change the expiry between them). There `fast_used = P +
//! fast_objects * (key_size + 4)`, plus 64 per TTL'd fast object, and the
//! identity test asserts that relation instead.
//!
//! The rounding is `nallocx(n, 0)`, not the `Layout`'s `(size, 8)`: identical
//! for every non-zero size (jemalloc's classes are multiples of 8), and for a
//! zero-length split value the stacks charge 0 where the allocator hands out
//! its 8-byte minimum. `numa_alloc::measured` counts the allocator's figure,
//! so under `measured_accounting` + `segregated_value_arena` P equals the
//! change in `measured::allocated(NODE_FAST_VALUES)` exactly while no
//! zero-length fast value is live, and trails it by 8 per one otherwise
//! (split layout only: a thin or fused item is never zero-sized).
//!
//! ## Clones, and `enable_tiering_manager`
//!
//! A `TieredValue` clone is a refcount bump in all three layouts (`Arc` in
//! the split and thin layouts, the item's own count under fusing): it
//! allocates nothing and charges nothing, and the allocation it keeps alive
//! stays charged until the LAST handle drops, which is when it leaves the
//! pool. So every path that clones an `Object` -- a reader's snapshot, a
//! migration's snapshot, `Object::clone` -- is exact by construction. The
//! legacy copy-based manager needs no `cfg` exclusion for the same reason,
//! and two more: the hybrid constructors never build one (only the flat impl
//! blocks do), and its DRAM side-copy is a plain `Box<[u8]>` from the global
//! allocator, not a `TieredValue`, so it is neither charged nor refunded.
//! That copy is DRAM outside the value pool, like the object map, and is
//! not what P measures.
//!
//! ## Process-global
//!
//! Like `numa_alloc::measured`, this is one counter per PROCESS: the charge
//! site is the value constructor, which does not know which cache it belongs
//! to. It counts every fast `TieredValue` -- a flat (`BufferDRAM`) cache in a
//! hybrid build builds its values through the same `new_in` and is counted
//! too. So P describes one cache only while that cache is the only one alive.
//! [`live_tiered_caches`] counts the tiered caches (up in the two hybrid
//! constructors, down when the cache is dropped), and a consumer -- the S5
//! gate in particular -- must refuse or disable itself unless it reads 1. The
//! server builds exactly one cache, and so does the benchmark's in-process
//! mode (one cache shared by every client); a harness that builds more sees
//! it in this count.
//!
//! ## Cost
//!
//! Per fast value allocation and per fast value free: one `nallocx` (a
//! size-class lookup, the same call `base_size` already makes on every set),
//! one read of the allocator's per-thread arena slot, and one relaxed
//! `fetch_add` on a cache-line-padded shard chosen by that slot -- no lock,
//! and no line shared by threads holding different slots. A shard is folded
//! into the shared `approx` word only when its magnitude reaches
//! [`FOLD_BYTES`], once per 128 KiB of one thread's net drift (every ~8
//! allocations of a 16 KiB value, every ~128 of a 1 KiB one). A slow value
//! pays one tier-tag branch. Arithmetic, not a measurement.
//!
//! Not compiled without `hybrid_cache_common`: a build with no tiers pays
//! nothing.

use std::{
	sync::{
		OnceLock,
		atomic::{AtomicI64, AtomicU64, Ordering},
	},
	time::{Duration, Instant},
};

/// Shards per counter, keyed by `numa_alloc`'s per-thread arena slot -- the
/// key `numa_alloc::measured` already shards on, so no second thread-local.
pub const SHARDS: usize = 16;

/// A shard is folded into `approx` once its magnitude reaches this, in EITHER
/// direction: a free often lands on a different thread's shard than its
/// charge (a consumer frees what a client allocated), so a shard drifts
/// negative as readily as positive.
pub const FOLD_BYTES: i64 = 128 * 1024;

/// One cache line per atomic: without it sixteen shards share two lines and
/// the sharding buys nothing.
#[repr(align(64))]
struct Padded<T>(T);

/// A sharded signed byte counter, with a folded running total and a peak.
///
/// Only the SUM of the shards and `approx` means anything; an individual
/// shard is routinely negative. A type rather than bare statics so the
/// arithmetic can be tested on a private instance without touching the
/// process-global one.
pub(crate) struct Counter {
	shards: [Padded<AtomicI64>; SHARDS],
	approx: Padded<AtomicI64>,
	max: Padded<AtomicI64>,
}

impl Counter {
	pub(crate) const fn new() -> Self {
		Counter {
			shards: [const { Padded(AtomicI64::new(0)) }; SHARDS],
			approx: Padded(AtomicI64::new(0)),
			max: Padded(AtomicI64::new(0)),
		}
	}

	/// Adds `delta` to shard `shard % SHARDS`, folding that shard once it
	/// reaches `FOLD_BYTES` in magnitude.
	#[inline]
	pub(crate) fn add(&self, shard: usize, delta: i64) {
		let cell = &self.shards[shard % SHARDS].0;
		let now = cell.fetch_add(delta, Ordering::Relaxed).wrapping_add(delta);

		if now >= FOLD_BYTES || now <= -FOLD_BYTES {
			self.fold(cell);
		}
	}

	/// Moves a shard's whole balance into `approx`. `swap` takes everything,
	/// adds that raced in after the one that triggered the fold included, so
	/// nothing is lost or counted twice. A concurrent `exact` can miss the
	/// swapped amount for the two instructions between the swap and the add
	/// -- a torn read, like any other cross-shard sum here.
	#[cold]
	#[inline(never)]
	fn fold(&self, cell: &AtomicI64) {
		let taken = cell.swap(0, Ordering::Relaxed);
		self.approx.0.fetch_add(taken, Ordering::Relaxed);
		self.observe_max(self.exact());
	}

	/// Every shard plus `approx`: the counter's value, exact whenever no
	/// charge, refund or fold is in flight, a torn read otherwise (like
	/// `measured::allocated`). Signed: a torn read can be momentarily
	/// negative, and a test looking for drift must see a persistent one.
	pub(crate) fn exact(&self) -> i64 {
		let shards: i64 = self.shards.iter().map(|s| s.0.load(Ordering::Relaxed)).sum();

		shards + self.approx.0.load(Ordering::Relaxed)
	}

	/// One load. Differs from `exact` by the unfolded shard balances, each
	/// under `FOLD_BYTES` in magnitude once its last add returns, so by less
	/// than `SHARDS * FOLD_BYTES` = 2 MiB.
	pub(crate) fn approx(&self) -> i64 {
		self.approx.0.load(Ordering::Relaxed)
	}

	pub(crate) fn observe_max(&self, value: i64) {
		self.max.0.fetch_max(value, Ordering::Relaxed);
	}

	/// The largest value seen at a fold or an `observe_max` (the policy worker
	/// calls it every pass). A LOWER BOUND on the true peak: a burst that
	/// rises and falls between two observations without folding a shard is
	/// never seen.
	pub(crate) fn max(&self) -> i64 {
		self.max.0.load(Ordering::Relaxed)
	}
}

static PHYS_FAST: Counter = Counter::new();

#[inline]
fn shard() -> usize {
	crate::numa_alloc::arena_slot() as usize % SHARDS
}

/// The figure P charges for one value allocation of `len` bytes keyed by `K`:
/// `resident_object_bytes::<K>(len)`, the stacks' own unit. See the module doc
/// for the one build pair (fused + DashMap) whose stacks charge more.
#[inline]
pub fn value_charge<K>(len: u32) -> u64 {
	crate::object::overhead::resident_object_bytes::<K>(len) as u64
}

/// Charges one fast value allocation. Called once per allocation, by
/// `TieredValue::new_in`, once the value exists.
#[inline]
pub(crate) fn charge(bytes: u64) {
	PHYS_FAST.add(shard(), bytes as i64);
}

/// Refunds one fast value allocation. Called once per allocation, by the drop
/// that frees it.
#[inline]
pub(crate) fn refund(bytes: u64) {
	PHYS_FAST.add(shard(), -(bytes as i64));
}

/// P (exact), clamped at zero for reporting.
pub fn fast_bytes() -> u64 {
	PHYS_FAST.exact().max(0) as u64
}

/// P, signed and unclamped: what a drift test reads, since the clamp would
/// hide a persistent negative drift (a refund with no charge).
pub fn fast_bytes_signed() -> i64 {
	PHYS_FAST.exact()
}

/// The folded total alone: one load, within 2 MiB of [`fast_bytes_signed`].
/// For a check that can afford that error on a hot path; nothing uses it yet.
pub fn fast_bytes_approx() -> i64 {
	PHYS_FAST.approx()
}

/// The peak of P seen at a fold or a policy-worker pass -- a LOWER BOUND on
/// the true peak (see `Counter::max`). Process-global and never reset.
pub fn fast_bytes_max() -> u64 {
	PHYS_FAST.max().max(0) as u64
}

/// Reads P and folds it into the peak: the policy worker's per-pass sample.
pub(crate) fn observe() -> i64 {
	let phys = PHYS_FAST.exact();
	PHYS_FAST.observe_max(phys);
	phys
}

/// Migration-queue entries handed to the consumers and not yet finished, as
/// `(demotions, promotions)`. Process-global, like the queue's own counters.
/// Entry counts only: queue entries carry no sizes, so pending BYTES wait for
/// a later step. Always `(0, 0)` under `MIGRATION_QUEUE_THREADS=0`, where
/// nothing is queued.
pub fn pending_migrations() -> (u64, u64) {
	crate::worker::pending_migrations()
}

// ---------------------------------------------------------------------------
// live tiered caches
// ---------------------------------------------------------------------------

static LIVE_TIERED_CACHES: AtomicU64 = AtomicU64::new(0);

/// Tiered caches alive in this process. P describes one cache only while this
/// reads 1 -- see the module doc.
pub fn live_tiered_caches() -> u64 {
	LIVE_TIERED_CACHES.load(Ordering::Relaxed)
}

/// One tiered cache's place in the live count: counted when built, uncounted
/// when dropped. Held by the cache's `AtomicStatus`, which is freed when the
/// last owner lets go -- the cache itself, after it has joined its workers --
/// so the count falls exactly when the cache is gone, and a constructor that
/// fails after registering uncounts on the way out.
pub(crate) struct LiveRegistration {
	count: &'static AtomicU64,
}

impl LiveRegistration {
	pub(crate) fn tiered_cache() -> Self {
		Self::in_count(&LIVE_TIERED_CACHES)
	}

	fn in_count(count: &'static AtomicU64) -> Self {
		count.fetch_add(1, Ordering::Relaxed);
		LiveRegistration { count }
	}
}

impl Drop for LiveRegistration {
	fn drop(&mut self) {
		self.count.fetch_sub(1, Ordering::Relaxed);
	}
}

// ---------------------------------------------------------------------------
// the policy worker's per-pass instrumentation
// ---------------------------------------------------------------------------

/// Byte-nanoseconds the fast tier spent over its budget during one pass
/// interval: `max(0, P + L * omega - F) * dt`, with a torn negative P read as
/// zero.
///
/// `P + L * omega` is the fast tier's physical DRAM as the budget models it --
/// the value bytes actually there plus the per-object reservation -- and `F`
/// the budget. The sample is taken at the END of the interval and held across
/// it (a right Riemann sum). The worker passes every 1 ms while sets arrive,
/// so under load an interval is short and the error small; when idle it
/// parks for up to 1 s, and nothing kicks it on the set path yet (S5's gate
/// will), so the pass that first sees a burst after a quiet spell charges its
/// whole idle interval at the burst's level. The integral therefore
/// OVER-states by at most one long poll per burst; it never under-states an
/// excursion that lasts past a pass.
pub(crate) fn over_budget_increment(
	phys: i64,
	live: u64,
	omega: u64,
	fast_capacity: u64,
	dt: Duration,
) -> u128 {
	let used = (phys.max(0) as u64).saturating_add(live.saturating_mul(omega));
	let over = used.saturating_sub(fast_capacity);

	over as u128 * dt.as_nanos()
}

/// Whole byte-seconds in a byte-nanosecond total, saturating.
pub(crate) fn byte_seconds(byte_ns: u128) -> u64 {
	(byte_ns / 1_000_000_000).min(u64::MAX as u128) as u64
}

/// What the policy worker keeps between passes for the integral and the MEMTS
/// line. One per worker, so per cache, unlike P.
pub(crate) struct PassInstrument {
	last_pass: Instant,
	over_budget_byte_ns: u128,
	last_memts: Option<Instant>,
}

impl PassInstrument {
	/// `now` is the cache's start: the first pass's interval runs from it.
	pub(crate) fn new(now: Instant) -> Self {
		PassInstrument {
			last_pass: now,
			over_budget_byte_ns: 0,
			last_memts: None,
		}
	}

	/// Accounts one pass ending at `now`, with `phys` physically in the fast
	/// value pool, and returns the integral so far in whole byte-seconds.
	/// `phys` is an argument rather than a read so the arithmetic can be
	/// driven with a synthetic P.
	pub(crate) fn pass(
		&mut self,
		now: Instant,
		phys: i64,
		live: u64,
		omega: u64,
		fast_capacity: u64,
	) -> u64 {
		let dt = now.saturating_duration_since(self.last_pass);
		self.last_pass = now;
		self.over_budget_byte_ns += over_budget_increment(phys, live, omega, fast_capacity, dt);

		byte_seconds(self.over_budget_byte_ns)
	}

	/// Whether this pass prints a MEMTS line, recording it if it does.
	pub(crate) fn memts_due(&mut self, now: Instant) -> bool {
		let due = memts_due(memts_enabled(), self.last_memts, now);

		if due {
			self.last_memts = Some(now);
		}

		due
	}
}

/// Minimum spacing of MEMTS lines. The worker passes every 1 ms under load; a
/// line per pass would be most of what the instrumentation costs.
pub(crate) const MEMTS_INTERVAL: Duration = Duration::from_millis(250);

/// `PAPER_MEMTS` present in the environment (any value), read once.
pub(crate) fn memts_enabled() -> bool {
	static ENABLED: OnceLock<bool> = OnceLock::new();

	*ENABLED.get_or_init(|| std::env::var_os("PAPER_MEMTS").is_some())
}

/// Pure: whether a pass at `now` prints, given when the last line was printed.
pub(crate) fn memts_due(enabled: bool, last: Option<Instant>, now: Instant) -> bool {
	enabled && last.is_none_or(|last| now.saturating_duration_since(last) >= MEMTS_INTERVAL)
}

/// One MEMTS line's fields -- only what exists at this step. The gate's bands,
/// waiters and pinned bytes come with the gate and are left OUT until then,
/// not printed as zeros that would read like a gate that never engaged.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct MemtsSample {
	/// Milliseconds since the policy workers' shared origin: the first cache
	/// built in this process, i.e. THIS cache in a one-cache process. The same
	/// origin as the `t_ms` on the MIGSTATS and DIVERGE lines.
	pub t_ms: u64,
	/// Wall clock, milliseconds since the Unix epoch.
	pub wall_ms: u64,
	/// P, exact and signed.
	pub phys: i64,
	/// `AtomicStatus::effective_fast_capacity`: F - L * omega.
	pub eff: u64,
	/// The stack's intent gauge.
	pub fast_used: u64,
	pub fast_metadata_bytes: u64,
	pub over_budget_byte_seconds: u64,
	/// Pending demotion entries minus pending promotion entries, now.
	pub pending_net: i64,
	/// Events waiting on the policy worker's channel.
	pub backlog: usize,
	pub live_tiered_caches: u64,
	/// `VmRSS` from `/proc/self/status`, kB; `None` if unreadable.
	pub vmrss_kb: Option<u64>,
}

/// Pure. `key=value` pairs after a `MEMTS ` prefix, in a fixed order, so a
/// parser can take them by regex and later steps can append.
pub(crate) fn format_memts(sample: &MemtsSample) -> String {
	let vmrss = match sample.vmrss_kb {
		Some(kb) => kb.to_string(),
		None => "na".to_owned(),
	};

	format!(
		"MEMTS t_ms={} wall_ms={} phys={} eff={} fast_used={} fast_metadata_bytes={} \
		 over_budget_byte_seconds={} pending_net={} backlog={} live_tiered_caches={} vmrss_kb={}",
		sample.t_ms,
		sample.wall_ms,
		sample.phys,
		sample.eff,
		sample.fast_used,
		sample.fast_metadata_bytes,
		sample.over_budget_byte_seconds,
		sample.pending_net,
		sample.backlog,
		sample.live_tiered_caches,
		vmrss,
	)
}

/// Milliseconds since the Unix epoch.
pub(crate) fn wall_ms() -> u64 {
	std::time::SystemTime::now()
		.duration_since(std::time::UNIX_EPOCH)
		.map_or(0, |d| d.as_millis() as u64)
}

/// This process's `VmRSS`, in kB.
pub(crate) fn vmrss_kb() -> Option<u64> {
	parse_vmrss_kb(&std::fs::read_to_string("/proc/self/status").ok()?)
}

fn parse_vmrss_kb(status: &str) -> Option<u64> {
	status
		.lines()
		.find_map(|line| line.strip_prefix("VmRSS:"))
		.and_then(|rest| rest.split_whitespace().next())
		.and_then(|kb| kb.parse().ok())
}

#[cfg(test)]
mod tests {
	use std::{
		sync::atomic::{AtomicU64, Ordering},
		time::{Duration, Instant},
	};

	use super::*;

	#[test]
	fn shards_sum_exactly_and_an_unfolded_shard_stays_out_of_approx() {
		let c = Counter::new();

		c.add(0, 1_000);
		c.add(1, -400);
		c.add(SHARDS + 1, -100); // the same shard as 1

		assert_eq!(c.exact(), 500);
		assert_eq!(c.approx(), 0, "nothing reached FOLD_BYTES, so nothing folded");
		assert_eq!(c.max(), 0, "no fold and no observation: nothing sampled the peak");
	}

	#[test]
	fn a_shard_folds_at_the_threshold_in_either_direction() {
		let c = Counter::new();

		c.add(2, FOLD_BYTES - 1);
		assert_eq!(c.approx(), 0, "one byte short of the threshold stays in the shard");

		c.add(2, 1);
		assert_eq!(c.approx(), FOLD_BYTES, "reaching it moves the whole shard");
		assert_eq!(c.exact(), FOLD_BYTES);

		// A NEGATIVE shard -- frees landing on a thread that charged nothing --
		// folds too, and where the bytes sit never changes the sum.
		c.add(3, -(FOLD_BYTES + 5));
		assert_eq!(c.approx(), -5);
		assert_eq!(c.exact(), -5);

		c.add(4, 7);
		c.add(5, -3);
		assert_eq!(c.exact(), -1);
		assert_eq!(c.approx(), -5, "the unfolded remainder is in exact, not approx");
		assert!((c.exact() - c.approx()).abs() < SHARDS as i64 * FOLD_BYTES);
	}

	#[test]
	fn the_max_is_sampled_at_folds_and_observations_and_is_a_lower_bound() {
		let c = Counter::new();

		// A spike that never folds and is never observed is invisible.
		c.add(0, FOLD_BYTES / 2);
		c.add(1, -(FOLD_BYTES / 2));
		assert_eq!(c.max(), 0);

		// A fold samples the whole sum, not just the shard it folds.
		c.add(0, 10);
		c.add(2, FOLD_BYTES);
		assert_eq!(c.exact(), FOLD_BYTES + 10);
		assert_eq!(c.max(), FOLD_BYTES + 10);

		c.add(3, -1_000);
		c.observe_max(c.exact());
		assert_eq!(c.max(), FOLD_BYTES + 10, "a lower reading never lowers the peak");

		c.add(3, 2_000);
		c.observe_max(c.exact());
		assert_eq!(c.max(), FOLD_BYTES + 1_010);
	}

	#[test]
	fn over_budget_integrates_only_the_excess_over_the_interval() {
		const SEC: Duration = Duration::from_secs(1);
		const NS: u128 = 1_000_000_000;

		// P + L * omega = 1_000 + 10 * 10 = 1_100 against F = 1_000, for 2 s.
		assert_eq!(over_budget_increment(1_000, 10, 10, 1_000, 2 * SEC), 200 * NS);
		// At or under the budget: nothing.
		assert_eq!(over_budget_increment(900, 10, 10, 1_000, SEC), 0);
		assert_eq!(over_budget_increment(1_000, 0, 10, 1_000, SEC), 0);
		// A torn negative P reads as zero, never as headroom.
		assert_eq!(over_budget_increment(-5_000, 200, 10, 1_000, SEC), 1_000 * NS);
		// No time, no area.
		assert_eq!(over_budget_increment(1_000_000, 0, 0, 0, Duration::ZERO), 0);
		// Saturating: an absurd L * omega does not wrap round into headroom.
		assert_eq!(
			over_budget_increment(0, u64::MAX, 2, 0, Duration::from_nanos(1)),
			u64::MAX as u128,
		);
	}

	#[test]
	fn the_pass_accumulator_integrates_over_the_time_since_the_previous_pass() {
		let start = Instant::now();
		let mut acc = PassInstrument::new(start);

		// 1.5 s at 400 B over the budget ...
		let t1 = start + Duration::from_millis(1_500);
		assert_eq!(acc.pass(t1, 1_400, 0, 0, 1_000), 600);

		// ... 0.5 s under it ...
		let t2 = t1 + Duration::from_millis(500);
		assert_eq!(acc.pass(t2, 500, 0, 0, 1_000), 600, "under budget adds nothing");

		// ... then 1 s at 1_000 B over, half of it the reservation.
		let t3 = t2 + Duration::from_secs(1);
		assert_eq!(acc.pass(t3, 1_500, 50, 10, 1_000), 1_600);

		// A second pass at the same instant adds nothing, however far over.
		assert_eq!(acc.pass(t3, i64::MAX, 0, 0, 0), 1_600);
		assert_eq!(byte_seconds(2_999_999_999), 2);
	}

	#[test]
	fn memts_formats_every_field_in_order_after_its_prefix() {
		let sample = MemtsSample {
			t_ms: 1_250,
			wall_ms: 1_790_000_000_123,
			phys: -64,
			eff: 5_000_000,
			fast_used: 4_900_000,
			fast_metadata_bytes: 120_000,
			over_budget_byte_seconds: 42,
			pending_net: -3,
			backlog: 17,
			live_tiered_caches: 1,
			vmrss_kb: Some(123_456),
		};

		assert_eq!(
			format_memts(&sample),
			"MEMTS t_ms=1250 wall_ms=1790000000123 phys=-64 eff=5000000 fast_used=4900000 \
			 fast_metadata_bytes=120000 over_budget_byte_seconds=42 pending_net=-3 backlog=17 \
			 live_tiered_caches=1 vmrss_kb=123456",
		);

		let unread = MemtsSample { vmrss_kb: None, ..sample };
		assert!(format_memts(&unread).ends_with(" vmrss_kb=na"));
		assert!(!format_memts(&sample).contains('\n'));
	}

	#[test]
	fn memts_is_absent_without_the_env_var_and_rate_limited_with_it() {
		let now = Instant::now();

		assert!(!memts_due(false, None, now), "disabled: never, not even the first pass");
		assert!(!memts_due(false, Some(now - Duration::from_secs(10)), now));

		assert!(memts_due(true, None, now), "enabled: the first pass prints");
		assert!(!memts_due(true, Some(now - Duration::from_millis(249)), now));
		assert!(memts_due(true, Some(now - MEMTS_INTERVAL), now));

		// The switch itself. The suites run without PAPER_MEMTS, so the
		// worker's gate reads false and no pass prints. Skipped rather than
		// failed if someone runs them with it set.
		if std::env::var_os("PAPER_MEMTS").is_none() {
			assert!(!memts_enabled());

			let mut pass = PassInstrument::new(now);
			assert!(!pass.memts_due(now));
			assert!(!pass.memts_due(now + Duration::from_secs(5)));
		}
	}

	#[test]
	fn vmrss_is_parsed_from_proc_status() {
		let status = "Name:\tpaper\nVmPeak:\t  999 kB\nVmRSS:\t   12345 kB\nRssAnon:\t 1 kB\n";
		assert_eq!(parse_vmrss_kb(status), Some(12_345));
		assert_eq!(parse_vmrss_kb("Name:\tx\n"), None);

		#[cfg(target_os = "linux")]
		assert!(vmrss_kb().is_some_and(|kb| kb > 0));
	}

	#[test]
	fn a_live_registration_counts_up_and_down() {
		static COUNT: AtomicU64 = AtomicU64::new(0);

		let a = LiveRegistration::in_count(&COUNT);
		assert_eq!(COUNT.load(Ordering::Relaxed), 1);

		let b = LiveRegistration::in_count(&COUNT);
		assert_eq!(COUNT.load(Ordering::Relaxed), 2);

		drop(a);
		assert_eq!(COUNT.load(Ordering::Relaxed), 1);

		drop(b);
		assert_eq!(COUNT.load(Ordering::Relaxed), 0);
	}

	/// Served-tier hits end to end, through a real cache. FIFO never promotes,
	/// so once its oldest key has been demoted every hit on it is served from
	/// the slow tier, while the newest key stays in the fast prefix. A miss and
	/// a peek count in neither.
	#[test]
	fn a_hit_is_counted_by_the_tier_it_was_served_from() {
		use crate::{CacheTierSize, PaperCache, Tier, TieredBuffer, policy::PaperPolicy};

		// This cache DEMOTES, through a migration queue that bumps the
		// process-wide disposition counters the queue tests assert exact
		// deltas on, so it runs under their lock. Declared first, so it is
		// released last: after the cache has dropped and joined its consumers.
		let _serialised = crate::worker::migration_test_lock::lock();

		let cache = PaperCache::<u64, TieredBuffer>::new(
			1 << 20,
			CacheTierSize::Bytes(8 << 10),
			PaperPolicy::FifoCompactHybrid,
		)
		.expect("a FIFO hybrid cache");

		for key in 1..=16u64 {
			cache.set(key, &[key as u8; 1_000], None).expect("set");
		}

		let deadline = Instant::now() + Duration::from_secs(10);
		while cache.tier_of(&1) != Some(Tier::Slow) {
			assert!(Instant::now() < deadline, "key 1 was never demoted");
			std::thread::sleep(Duration::from_millis(5));
		}

		assert_eq!(cache.tier_of(&16), Some(Tier::Fast), "the newest key is in the fast prefix");

		cache.get(&16).expect("a hit");
		cache.get(&1).expect("a hit");

		#[cfg(not(feature = "enable_tiering_manager"))]
		cache.get_into(&1, &mut Vec::new()).expect("a hit");

		cache.peek(&1).expect("a peek");
		cache.peek(&16).expect("a peek");
		assert!(cache.get(&999).is_err());

		let slow = if cfg!(feature = "enable_tiering_manager") { 1 } else { 2 };
		let stats = cache.hybrid_stats();
		assert_eq!((stats.fast_hits, stats.slow_hits), (1, slow));
	}
}
