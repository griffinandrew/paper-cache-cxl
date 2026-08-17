/*
 * Copyright (c) Kia Shakiba
 *
 * This source code is licensed under the GNU AGPLv3 license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! Statistics for `two_q_fast_admission_reprieve_hybrid_cache`.
//!
//! The counters backing this snapshot live directly on `AtomicStatus`
//! (`status.rs`) rather than in a separate atomics struct here — same
//! rationale as every other hybrid design's stats module: `AtomicStatus` is
//! already the one shared, per-cache structure both `PaperCache` and
//! `PolicyWorker` hold (`status: StatusRef`), so no new field is needed on
//! `PaperCache` itself, which matters because `PaperCache`'s struct
//! definition in `lib.rs` is shared across every value type and adding a
//! field there would ripple into every other constructor in the file.

/// A point-in-time snapshot of `two_q_fast_admission_reprieve_hybrid_cache`
/// statistics.
///
/// Assembled by `AtomicStatus::two_q_fast_admission_hybrid_stats` and
/// returned from `PaperCache::two_q_fast_admission_hybrid_stats`. The same
/// seven fields are also reachable design-neutrally via
/// `PaperCache::hybrid_stats` (see `crate::hybrid_stats`).
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct TwoQFastAdmissionReprieveHybridStats {
	/// Objects moved from the main queue's slow portion into its fast
	/// portion.
	///
	/// Unlike `two_q_hybrid_cache`, this does **not** count FIFO→main
	/// promotions: the one-access queue is already fast here, so that
	/// transition moves no bytes and emits no migration. A rising
	/// `promotions` therefore means specifically "a demoted, previously
	/// proven object was re-accessed", not "a new object proved itself".
	pub promotions: u64,

	/// Objects moved into the slow tier: the main queue's fast portion
	/// demoting under fast-tier pressure, **plus** one-access keys reprieved
	/// out of the FIFO queue when it exceeds its budget. Both are real
	/// DRAM→PMEM copies, which is why they share a counter; the reprieve
	/// stream is typically the larger of the two.
	///
	/// Note the main queue's budget here is `fast_tier_size` minus the
	/// one-access queue's fixed `k_in * max_size` reservation, not the whole
	/// fast tier — so demotions begin earlier, at a lower `fast_bytes_used`,
	/// than an equivalently-configured `two_q_hybrid_cache` would show.
	pub demotions: u64,

	/// Terminal removals from the cache — the main queue's LRU tail under
	/// overall capacity pressure.
	///
	/// Unlike `two_q_fast_admission_hybrid_cache`, a one-access key aging out
	/// of the FIFO queue is **not** counted here, because it is not evicted:
	/// it is reprieved into the slow tier (and counted in `demotions`
	/// instead). It can of course be evicted later, once it reaches the main
	/// queue's tail — which, since a reprieve lands it exactly there, is
	/// often soon.
	pub evictions: u64,

	/// Bytes currently in DRAM, covering **both** the one-access FIFO queue
	/// and the main queue's fast segment. (In `two_q_hybrid_cache` the FIFO
	/// queue counts toward `slow_bytes_used` instead — that difference is
	/// the entire point of this design.)
	pub fast_bytes_used: u64,

	/// Bytes currently in PMEM: the main queue's slow segment only.
	pub slow_bytes_used: u64,

	/// Objects currently in DRAM, across the FIFO queue and the main
	/// queue's fast segment.
	pub fast_objects: u64,

	/// Objects currently in PMEM: the main queue's slow segment only.
	pub slow_objects: u64,
}
