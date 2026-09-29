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
//! log). This counter is the measurement the backpressure plan's gate acts
//! on: since S5's commit B2 the byte gate (`crate::gate`) holds a tiered
//! cache's fast sets to `P + M_model <= F + slack`, reading it here.
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
//! One more condition, on the DashMap-family stacks in the split and thin
//! layouts: they keep the DRAM-resident remainder of an object -- `key_size +
//! 4` (the expiry field), `+ 64` for a TTL'd object (`get_ttl_overhead`) -- in
//! a spare `u8` of the stack entry (`narrow_resident`), saturating at 255, and
//! charge `base_size` less THAT. So their `fast_used` equals P only while
//! `key_size + 4 (+ 64 with a TTL) <= 255`: a longer key leaves its saturated
//! excess in `fast_used` on top of P, per fast object. The merged store does
//! not narrow, and a `u64` key's remainder is 12 (76 with a TTL).
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
//! to. It counts every fast `TieredValue`, not only a tiered cache's: a FLAT
//! cache whose values are fast (`PaperCache<K, BufferDRAM>` in a hybrid
//! build) builds them through the same `new_in`. Two live counts say what
//! else P holds:
//!
//!   * [`live_tiered_caches`] -- tiered caches, counted in the two hybrid
//!     constructors and uncounted when the cache is dropped;
//!   * [`live_flat_fast_caches`] -- flat caches whose values are fast
//!     (`V::TIER == Tier::Fast`), counted in the two flat constructors and
//!     uncounted when dropped. A `BufferPMEM` cache's values are slow and
//!     charge nothing, so it is not counted.
//!
//! P describes ONE cache only while `live_tiered_caches() == 1` AND
//! `live_flat_fast_caches() == 0`. A consumer -- the S5 gate in particular --
//! must check both, and refuse or disable itself otherwise. The server builds
//! exactly one cache, and so does the benchmark's in-process mode (one cache
//! shared by every client); a harness that builds more sees it in these
//! counts. A value built outside any cache (a test's `TieredValue::new_fast`)
//! is in P and in neither count.
//!
//! ## Cost
//!
//! Per fast value allocation and per fast value free: one `nallocx` (a
//! size-class lookup, the same call `base_size` already makes on every set),
//! one read of the allocator's per-thread arena slot, and one relaxed
//! `fetch_add` on a cache-line-padded shard chosen by that slot -- no lock.
//! A shard is not one thread's line, though. `numa_alloc` hands slots out
//! round-robin, `0..32` (`MAX_ARENAS_PER_NODE`), in the order threads first
//! allocate through a bound arena, and the shard is the slot `% 16`
//! ([`SHARDS`]): the 1st and the 17th thread to allocate share shard 0, the
//! 2nd and the 18th shard 1, and so on, and a thread that has never allocated
//! through a bound arena reads `u32::MAX`, which is shard 15, beside slots 15
//! and 31. So at most 16 threads can have a line each, and only while no
//! other thread's slot is congruent to theirs mod 16; past that, threads
//! share lines -- contention, never a wrong sum. A shard is folded into the
//! shared `approx` word only when its magnitude reaches [`FOLD_BYTES`], once
//! per 128 KiB of one shard's net drift (every ~8 allocations of a 16 KiB
//! value, every ~128 of a 1 KiB one). A slow value pays one tier-tag branch.
//! Arithmetic, not a measurement.
//!
//! The byte gate (S5, commit B2) adds, per fast refund, one relaxed load of
//! `GATE_WATCH` -- and while a set waits for fast bytes (or a stall is
//! unresolved) one relaxed `fetch_add` on the same shard's line of `FREED`,
//! the watchdog's count of bytes freed -- and, per FOLD, a read lock of
//! `GATE_HOOK` and the enabled gate's near-flag re-evaluation (a load, and a
//! `fetch_or`/`fetch_and` of the gate word only when the flag changes).
//!
//! Not compiled without `hybrid_cache_common`: a build with no tiers pays
//! nothing.

use std::{
	sync::{
		Arc,
		OnceLock,
		atomic::{AtomicI64, AtomicU64, Ordering},
	},
	time::{Duration, Instant},
};

/// Shards per counter, keyed by `numa_alloc`'s per-thread arena slot -- the
/// key `numa_alloc::measured` already shards on, so no second thread-local.
/// Slots run `0..32` and are reduced `% SHARDS`, so slots `s` and `s + 16`
/// share a shard, and a thread with no slot yet (`u32::MAX`) uses the last:
/// see the module doc's Cost section.
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
	/// reaches `FOLD_BYTES` in magnitude. Returns `approx` after the fold when
	/// this add folded -- where the byte gate re-evaluates its near flag (S5
	/// B2, `gate_hook`) -- and `None` otherwise.
	#[inline]
	pub(crate) fn add(&self, shard: usize, delta: i64) -> Option<i64> {
		let cell = &self.shards[shard % SHARDS].0;
		let now = cell.fetch_add(delta, Ordering::Relaxed).wrapping_add(delta);

		if now >= FOLD_BYTES || now <= -FOLD_BYTES {
			return Some(self.fold(cell));
		}

		None
	}

	/// Moves a shard's whole balance into `approx`. `swap` takes everything,
	/// adds that raced in after the one that triggered the fold included, so
	/// nothing is lost or counted twice in the sum.
	///
	/// A concurrent reader can still catch the move half done: the balance
	/// has left the shard and not yet reached `approx`. The add is `Release`
	/// and `exact` loads `approx` FIRST, with `Acquire`, then the shards: a
	/// reader whose `approx` load sees this add therefore also sees the swap
	/// when it reads the shard. So a fold in flight can be MISSED by a read
	/// (the balance in neither place it looked), never counted twice (in
	/// both). See `exact`. Returns `approx` as this fold left it.
	#[cold]
	#[inline(never)]
	fn fold(&self, cell: &AtomicI64) -> i64 {
		let taken = cell.swap(0, Ordering::Relaxed);
		let approx = self.approx.0.fetch_add(taken, Ordering::Release).wrapping_add(taken);
		self.observe_max(self.exact());
		approx
	}

	/// Every shard plus `approx`: the counter's value, exact whenever no
	/// charge, refund or fold is in flight, a torn read otherwise (like
	/// `measured::allocated`). Signed: a torn read can be momentarily
	/// negative, and a test looking for drift must see a persistent one.
	///
	/// `approx` is loaded FIRST, with `Acquire` (paired with `fold`'s
	/// `Release`), and the shards after it. In the other order a fold landing
	/// between the reads was counted twice -- its balance read in the shard
	/// before the swap and again in `approx` after the add -- an
	/// over-statement of a whole shard's balance (>= `FOLD_BYTES`) that the
	/// peak then kept. In this order a fold in flight can only be missed.
	///
	/// A read of 17 words is still not a snapshot. A value refunded on a
	/// shard read early and another charged on a shard read late can both be
	/// counted although they were never live at the same instant, and the
	/// reverse can count neither. So one read can differ from P at every
	/// instant of the read by up to the bytes charged and refunded on other
	/// threads while it runs -- a few allocations, over 17 loads.
	pub(crate) fn exact(&self) -> i64 {
		self.exact_with(|| {})
	}

	/// `exact`, running `between` after `approx` is loaded and before the
	/// shards are: the window the order is about. `exact` passes a no-op,
	/// which compiles away; the ordering test passes a fold.
	#[inline(always)]
	fn exact_with(&self, between: impl FnOnce()) -> i64 {
		let approx = self.approx.0.load(Ordering::Acquire);

		between();

		let shards: i64 = self.shards.iter().map(|s| s.0.load(Ordering::Relaxed)).sum();

		approx + shards
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
	/// calls it every pass). A lower bound on the true peak, up to one
	/// sample's torn read: a burst that rises and falls between two samples
	/// without folding a shard is never seen, and a fold in flight during a
	/// sample is missed rather than counted twice, but a sample is not a
	/// snapshot and can exceed P at every instant of its own read by the
	/// bytes charged and refunded while it reads (see `exact`). So `max <=
	/// true peak + one read's churn`.
	pub(crate) fn max(&self) -> i64 {
		self.max.0.load(Ordering::Relaxed)
	}
}

static PHYS_FAST: Counter = Counter::new();

/// This thread's shard: its arena slot `% SHARDS` (see [`SHARDS`]).
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
	if let Some(approx) = PHYS_FAST.add(shard(), bytes as i64) {
		gate_hook(approx);
	}
}

/// Refunds one fast value allocation. Called once per allocation, by the drop
/// that frees it. While a byte gate watches (`GATE_WATCH`), the bytes also
/// count in `FREED`, its watchdog's evidence that something was freed.
#[inline]
pub(crate) fn refund(bytes: u64) {
	let shard = shard();

	if let Some(approx) = PHYS_FAST.add(shard, -(bytes as i64)) {
		gate_hook(approx);
	}

	if GATE_WATCH.load(Ordering::Relaxed) != 0 {
		FREED[shard % SHARDS].0.fetch_add(bytes, Ordering::Relaxed);
	}
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

/// The peak of P seen at a fold or a policy-worker pass -- a lower bound on
/// the true peak up to one sample's torn read (see `Counter::max`).
/// Process-global, and never reset outside tests ([`reset_fast_bytes_max`]).
pub fn fast_bytes_max() -> u64 {
	PHYS_FAST.max().max(0) as u64
}

/// TEST SUPPORT: forgets the peak (sets it to 0), so a test can require a
/// fresh sample. T9 does at every check: the peak is process-global, so
/// without this one left by an earlier cache or phase satisfies the check.
/// Nothing in the crate calls it, and a harness must not -- the peak is the
/// process's only while nothing does.
#[doc(hidden)]
pub fn reset_fast_bytes_max() {
	PHYS_FAST.max.0.store(0, Ordering::Relaxed);
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

static LIVE_FLAT_FAST_CACHES: AtomicU64 = AtomicU64::new(0);

/// Tiered caches alive in this process. P describes one cache only while this
/// reads 1 and [`live_flat_fast_caches`] reads 0 -- see the module doc.
pub fn live_tiered_caches() -> u64 {
	LIVE_TIERED_CACHES.load(Ordering::Relaxed)
}

/// Flat caches whose values are FAST (`V::TIER == Tier::Fast`, i.e.
/// `PaperCache<K, BufferDRAM>`) alive in this process. Their values are in P
/// too, so P describes one tiered cache only while this reads 0 and
/// [`live_tiered_caches`] reads 1 -- see the module doc.
pub fn live_flat_fast_caches() -> u64 {
	LIVE_FLAT_FAST_CACHES.load(Ordering::Relaxed)
}

/// One cache's place in a live count: counted when built, uncounted
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

	pub(crate) fn flat_fast_cache() -> Self {
		Self::in_count(&LIVE_FLAT_FAST_CACHES)
	}

	fn in_count(count: &'static AtomicU64) -> Self {
		count.fetch_add(1, Ordering::Relaxed);
		GATE_EPOCH.fetch_add(1, Ordering::AcqRel);
		LiveRegistration { count }
	}
}

impl Drop for LiveRegistration {
	fn drop(&mut self) {
		self.count.fetch_sub(1, Ordering::Relaxed);
		GATE_EPOCH.fetch_add(1, Ordering::AcqRel);
	}
}

// ---------------------------------------------------------------------------
// the byte gate's hooks (S5, commit B2)
// ---------------------------------------------------------------------------

/// `E`, the byte gate's bound on `approx`'s error: every shard's unfolded
/// balance is under `FOLD_BYTES` in magnitude once its last add has returned,
/// so `P < approx + E` (design 3.9.2-3.9.3) -- up to the adds still in flight.
pub(crate) const FOLD_ERROR: i64 = SHARDS as i64 * FOLD_BYTES;

/// How many reasons a byte gate has to watch for freed bytes: one per
/// non-empty bytes lane and one per unresolved stall (`gate::Gate`). While it
/// is non-zero every fast refund also counts its bytes in `FREED`.
static GATE_WATCH: AtomicU64 = AtomicU64::new(0);

/// Bytes refunded while a gate watched, sharded like P. Monotonic, and only
/// its ADVANCE means anything: the watchdog's "something was freed".
static FREED: [Padded<AtomicU64>; SHARDS] = [const { Padded(AtomicU64::new(0)) }; SHARDS];

/// Moved by every change of a live count (`LiveRegistration`): a byte gate
/// that sees it move re-reads whether its cache is still P's only user at
/// once, not at its worker's next pass (design 3.9.8).
static GATE_EPOCH: AtomicU64 = AtomicU64::new(0);

/// The enabled byte gate's shared state, against which every fold
/// re-evaluates the gate's near flag (`gate::GateShared::on_fold`). At most one
/// gate is enabled at a time -- enabling requires its cache to be P's only
/// user -- so one slot. A read lock per fold, never taken with another lock
/// held by this module; the write lock only by a policy worker installing or
/// clearing its own gate.
static GATE_HOOK: std::sync::RwLock<Option<Arc<crate::gate::GateShared>>> = std::sync::RwLock::new(None);

/// A byte gate starts (`true`) or stops (`false`) watching for freed bytes.
pub(crate) fn watch_freed(on: bool) {
	match on {
		true => GATE_WATCH.fetch_add(1, Ordering::AcqRel),
		false => GATE_WATCH.fetch_sub(1, Ordering::AcqRel),
	};
}

/// Bytes refunded while a gate watched (`FREED`), summed over its shards.
pub(crate) fn freed() -> u64 {
	FREED.iter().map(|shard| shard.0.load(Ordering::Relaxed)).fold(0, u64::wrapping_add)
}

/// `GATE_EPOCH`: moved by every live-count change.
pub(crate) fn gate_epoch() -> u64 {
	GATE_EPOCH.load(Ordering::Acquire)
}

/// P describes ONE tiered cache: exactly one is alive and no flat cache with
/// fast values -- the byte gate's condition (see the module doc).
pub(crate) fn sole_fast_user() -> bool {
	live_tiered_caches() == 1 && live_flat_fast_caches() == 0
}

/// Installs `shared` as the fold hook: its policy worker enabled its gate.
pub(crate) fn install_gate_hook(shared: &Arc<crate::gate::GateShared>) {
	*GATE_HOOK.write().unwrap_or_else(|poisoned| poisoned.into_inner()) = Some(shared.clone());
}

/// Clears the fold hook if it is `shared`'s: its gate was disabled or its
/// worker exited. Compare-and-clear, so a gate going away never removes
/// another's.
pub(crate) fn clear_gate_hook(shared: &Arc<crate::gate::GateShared>) {
	let mut hook = GATE_HOOK.write().unwrap_or_else(|poisoned| poisoned.into_inner());

	if hook.as_ref().is_some_and(|installed| Arc::ptr_eq(installed, shared)) {
		*hook = None;
	}
}

/// A fold of P left `approx`: the enabled gate, if any, re-evaluates its near
/// flag (design 3.9.3).
#[cold]
#[inline(never)]
fn gate_hook(approx: i64) {
	let hook = GATE_HOOK.read().unwrap_or_else(|poisoned| poisoned.into_inner());

	if let Some(shared) = hook.as_ref() {
		shared.on_fold(approx);
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
/// the whole budget. The integrand is defined on those three, NOT as `P -
/// eff`: `eff = F - L * omega` saturates at 0, so once `L * omega > F` the
/// integrand is `P + (L * omega - F)`, more than `P - eff = P`. (MEMTS prints
/// `phys` and `eff`; their difference is the integrand only while `L * omega
/// <= F`.)
///
/// A RIGHT Riemann sum: each pass samples at the END of its interval and
/// charges that level for the whole interval. So an excursion over the budget
/// that begins inside an interval is charged from the interval's start (its
/// head over-stated), and one that ends inside an interval is charged nothing
/// for that interval, since the pass closing it sees the tier back under (its
/// tail dropped). Each error is at most one poll interval of the excursion:
/// 1 ms while sets are recent (a set within the last 5 s, or a set waiting
/// in the byte gate), 1 s otherwise. Since S5 the first set after an idle
/// spell kicks the worker, so a burst is seen within its first pass.
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

/// `PAPER_MEMTS=1` in the environment, read once. Only `1` turns the line
/// on, as only `1` sets `PAPER_DISABLE_SHARED_OVERHEAD`: `PAPER_MEMTS=0`, an
/// empty value or any other string leaves it off.
pub(crate) fn memts_enabled() -> bool {
	static ENABLED: OnceLock<bool> = OnceLock::new();

	*ENABLED.get_or_init(|| memts_switch(std::env::var_os("PAPER_MEMTS").as_deref()))
}

/// Pure: whether a `PAPER_MEMTS` value (`None` when unset) turns MEMTS on.
fn memts_switch(value: Option<&std::ffi::OsStr>) -> bool {
	value.is_some_and(|value| value == "1")
}

/// Pure: whether a pass at `now` prints, given when the last line was printed.
pub(crate) fn memts_due(enabled: bool, last: Option<Instant>, now: Instant) -> bool {
	enabled && last.is_none_or(|last| now.saturating_duration_since(last) >= MEMTS_INTERVAL)
}

/// One MEMTS line's fields -- only what exists at this step. The byte gate's
/// levels, waiters and reservations are in since S5's commit B2 (last, as
/// every field added after the line's first version); pinned bytes come with
/// the server's permits (S9) and are left OUT until then, not printed as
/// zeros that would read like a figure that exists.
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
	/// `AtomicStatus::effective_fast_capacity`: F - L * omega, saturating at
	/// 0. So `phys - eff` is the over-budget integrand only while `L * omega
	/// <= F`; past that `eff` reads 0 and the integrand is larger (see
	/// `over_budget_increment`).
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
	/// Flat caches with fast values alive in this process: with
	/// `live_tiered_caches`, what says whether `phys` is one cache's.
	pub live_flat_fast_caches: u64,
	/// M (S5a, `AtomicStatus::dram_metadata_bytes`): the bytes the cache's own
	/// DRAM metadata structures hold, beside `fast_metadata_bytes`'s model.
	/// Last, like every field added after the line's first version.
	pub meta: u64,
	/// The byte gate's levels (S5 B2, `gate::bands`): the settle target S, the
	/// near level N and the close level B, as last published; 0 while the
	/// gate is not enabled.
	pub s: u64,
	pub n: u64,
	pub b: u64,
	/// Sets waiting in the gate's bytes lane, and the bytes admitted sets
	/// hold reserved until their values are built.
	pub waiters: u64,
	pub reserved: u64,
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
		 over_budget_byte_seconds={} pending_net={} backlog={} live_tiered_caches={} vmrss_kb={} \
		 live_flat_fast_caches={} meta={} s={} n={} b={} waiters={} reserved={}",
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
		sample.live_flat_fast_caches,
		sample.meta,
		sample.s,
		sample.n,
		sample.b,
		sample.waiters,
		sample.reserved,
	)
}

// ---------------------------------------------------------------------------
// the placement audit
// ---------------------------------------------------------------------------

/// Where every live value's bytes ARE -- its tag -- against where the policy
/// stack PLACES its key (`PolicyStack::placement_of`): the answer to
/// `PaperCache::placement_audit`, a DIAGNOSTIC (backpressure plan S3).
///
/// Bytes are [`value_charge`]s, the stacks' own unit and P's, so in a
/// one-cache process at quiescence `fast_bytes` is P. Every live value is
/// counted once in `fast`/`slow`, by where its bytes are, and at most once
/// more in one of the three mismatch classes:
///
///   * `stranded` -- bytes in the FAST tier, the stack places them SLOW: DRAM
///     the stack believes it gave up, which no settle will ever free (it
///     counts them in `slow_used`, not `fast_used`);
///   * `lagging` -- bytes in the SLOW tier, the stack places them FAST:
///     charged to the fast budget, served at CXL speed until the worker's
///     heal sees a hit served from the slow tier, with nothing of its bucket
///     in flight, and queues the promotion;
///   * `untracked` -- the stack does not know the key (`placement_of` is
///     `None`): a key the stack lost, or a value whose `Set` the worker has
///     not taken yet.
///
/// The worker lands the MIGRATIONS it has decided before it walks -- its
/// pending drain, and a flush of the migration consumers -- so a value merely
/// waiting on a queued migration is not counted as a mismatch: what is counted
/// is what nothing in flight will fix. It does not run the EVICTION pass the
/// event may have fallen before (the audit is handled mid-batch, where its
/// event is), so a cache over its size is audited with the values that pass
/// will evict. EXACT ONLY AT CLIENT QUIESCENCE: clients running during the
/// walk move values under it, and a value whose `Set` the worker has not
/// taken yet is untracked. Nothing here is exported in `HybridStats`.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct PlacementAudit {
	/// Live values walked.
	pub live: u64,

	/// Values whose bytes are in the fast tier, and their charge.
	pub fast: u64,
	pub fast_bytes: u64,

	/// Values whose bytes are in the slow tier, and their charge.
	pub slow: u64,
	pub slow_bytes: u64,

	/// In the fast tier, placed slow.
	pub stranded: u64,
	pub stranded_bytes: u64,

	/// In the slow tier, placed fast.
	pub lagging: u64,
	pub lagging_bytes: u64,

	/// Not tracked by the stack at all, wherever the bytes are.
	pub untracked: u64,
	pub untracked_bytes: u64,
}

impl PlacementAudit {
	/// Counts one live value: the tier its bytes are in, where the stack
	/// places it, and its charge.
	pub(crate) fn record(&mut self, physical: crate::Tier, placement: Option<crate::Tier>, bytes: u64) {
		use crate::Tier::{Fast, Slow};

		self.live += 1;

		match physical {
			Fast => {
				self.fast += 1;
				self.fast_bytes += bytes;
			},

			Slow => {
				self.slow += 1;
				self.slow_bytes += bytes;
			},
		}

		match (physical, placement) {
			(_, None) => {
				self.untracked += 1;
				self.untracked_bytes += bytes;
			},

			(Fast, Some(Slow)) => {
				self.stranded += 1;
				self.stranded_bytes += bytes;
			},

			(Slow, Some(Fast)) => {
				self.lagging += 1;
				self.lagging_bytes += bytes;
			},

			(Fast, Some(Fast)) | (Slow, Some(Slow)) => {},
		}
	}

	/// No value is stranded, lagging or untracked.
	pub fn is_clean(&self) -> bool {
		self.stranded == 0 && self.lagging == 0 && self.untracked == 0
	}
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

	/// Every live value lands in exactly one of fast/slow by its tag, and in
	/// at most one mismatch class by the placement; bytes are summed as given.
	#[test]
	fn the_audit_classifies_by_tag_against_placement_and_sums_the_charges() {
		use crate::Tier::{Fast, Slow};

		let mut a = PlacementAudit::default();
		assert!(a.is_clean(), "an empty audit is clean");

		a.record(Fast, Some(Fast), 100);
		a.record(Slow, Some(Slow), 200);
		assert!(a.is_clean(), "placed where the bytes are: clean");

		a.record(Fast, Some(Slow), 1_000);
		a.record(Slow, Some(Fast), 20_000);
		a.record(Slow, Some(Fast), 30_000);
		a.record(Fast, None, 400_000);
		a.record(Slow, None, 5_000_000);

		assert_eq!(
			a,
			PlacementAudit {
				live: 7,
				fast: 3,
				fast_bytes: 100 + 1_000 + 400_000,
				slow: 4,
				slow_bytes: 200 + 20_000 + 30_000 + 5_000_000,
				stranded: 1,
				stranded_bytes: 1_000,
				lagging: 2,
				lagging_bytes: 50_000,
				untracked: 2,
				untracked_bytes: 5_400_000,
			},
		);
		assert!(!a.is_clean());
	}

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

	/// The order `exact` reads in, pinned by running a fold in the one window
	/// that matters: after `approx` is loaded and before the shards are. The
	/// fold moves shard 3's balance into `approx`, so a reader that already
	/// has `approx` and then finds the shard emptied misses the balance (in
	/// neither place it looked). The old order -- shards, then `approx` --
	/// found it in both: 2 * FOLD_BYTES - 1 + 100 here, above every value the
	/// counter ever held. Single-threaded, so deterministic; the
	/// Acquire/Release pairing that carries the same guarantee between
	/// threads is argued in `fold`'s doc and cannot be forced here.
	#[test]
	fn a_fold_between_the_reads_is_missed_never_counted_twice() {
		let c = Counter::new();

		c.add(3, FOLD_BYTES - 1);
		c.add(4, 100);
		assert_eq!(c.approx(), 0, "nothing folded yet");

		let before = c.exact();
		let torn = c.exact_with(|| {
			c.add(3, 1); // reaches FOLD_BYTES: folds shard 3
		});
		let after = c.exact();

		assert_eq!((before, after), (FOLD_BYTES + 99, FOLD_BYTES + 100));
		assert_eq!(c.approx(), FOLD_BYTES, "the add inside the window folded shard 3");
		assert_eq!(torn, 100, "the balance in flight is missed, not read in the shard and in approx");
		assert!(torn <= before.max(after), "a torn read never exceeds what the counter held");
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
			live_flat_fast_caches: 2,
			meta: 131_072,
			s: 4_900_000,
			n: 4_950_000,
			b: 5_000_000,
			waiters: 3,
			reserved: 4_096,
		};

		assert_eq!(
			format_memts(&sample),
			"MEMTS t_ms=1250 wall_ms=1790000000123 phys=-64 eff=5000000 fast_used=4900000 \
			 fast_metadata_bytes=120000 over_budget_byte_seconds=42 pending_net=-3 backlog=17 \
			 live_tiered_caches=1 vmrss_kb=123456 live_flat_fast_caches=2 meta=131072 \
			 s=4900000 n=4950000 b=5000000 waiters=3 reserved=4096",
		);

		let unread = MemtsSample { vmrss_kb: None, ..sample };
		assert!(format_memts(&unread).contains(" vmrss_kb=na "));
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

		// The switch itself. The suites run without PAPER_MEMTS=1, so the
		// worker's gate reads false and no pass prints. Skipped rather than
		// failed if someone runs them with it set.
		if !memts_switch(std::env::var_os("PAPER_MEMTS").as_deref()) {
			assert!(!memts_enabled());

			let mut pass = PassInstrument::new(now);
			assert!(!pass.memts_due(now));
			assert!(!pass.memts_due(now + Duration::from_secs(5)));
		}
	}

	/// Only `PAPER_MEMTS=1` turns the line on, as only `1` sets
	/// `PAPER_DISABLE_SHARED_OVERHEAD`.
	#[test]
	fn memts_is_switched_on_by_exactly_1() {
		use std::ffi::OsStr;

		assert!(memts_switch(Some(OsStr::new("1"))));
		assert!(!memts_switch(None), "unset: off");

		for off in ["", "0", "true", "yes", "on", " 1", "1 ", "01", "11"] {
			assert!(!memts_switch(Some(OsStr::new(off))), "PAPER_MEMTS={off:?} must leave MEMTS off");
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

		let epoch = gate_epoch();
		let a = LiveRegistration::in_count(&COUNT);
		assert_eq!(COUNT.load(Ordering::Relaxed), 1);
		assert!(gate_epoch() > epoch, "a registration moves the byte gate's epoch (S5 B2)");

		let b = LiveRegistration::in_count(&COUNT);
		assert_eq!(COUNT.load(Ordering::Relaxed), 2);

		drop(a);
		assert_eq!(COUNT.load(Ordering::Relaxed), 1);

		let epoch = gate_epoch();
		drop(b);
		assert_eq!(COUNT.load(Ordering::Relaxed), 0);
		assert!(gate_epoch() > epoch, "and so does its drop");
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

		// The per-object metadata model (S5): at this toy fast tier the MEASURED
		// M of the cache's own structures would leave the strict key ceiling
		// no room, and every new key would fail with `MetadataOverflow`.
		let _per_object = crate::object::overhead::test_overheads::per_object();

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
