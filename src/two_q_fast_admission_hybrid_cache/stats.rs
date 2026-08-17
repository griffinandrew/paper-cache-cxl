/*
 * Copyright (c) Kia Shakiba
 *
 * This source code is licensed under the GNU AGPLv3 license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! Statistics for `two_q_fast_admission_hybrid_cache`.
//!
//! The counters backing this snapshot live directly on `AtomicStatus`
//! (`status.rs`) rather than in a separate atomics struct here — same
//! rationale as every other hybrid design's stats module: `AtomicStatus` is
//! already the one shared, per-cache structure both `PaperCache` and
//! `PolicyWorker` hold (`status: StatusRef`), so no new field is needed on
//! `PaperCache` itself, which matters because `PaperCache`'s struct
//! definition in `lib.rs` is shared across every value type and adding a
//! field there would ripple into every other constructor in the file.

/// A point-in-time snapshot of `two_q_fast_admission_hybrid_cache`
/// statistics.
///
/// Assembled by `AtomicStatus::two_q_fast_admission_hybrid_stats` and
/// returned from `PaperCache::two_q_fast_admission_hybrid_stats`. The same
/// seven fields are also reachable design-neutrally via
/// `PaperCache::hybrid_stats` (see `crate::hybrid_stats`).
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct TwoQFastAdmissionHybridStats {
	/// Objects moved from the main queue's slow portion into its fast
	/// portion.
	///
	/// Unlike `two_q_hybrid_cache`, this does **not** count FIFO→main
	/// promotions: the one-access queue is already fast here, so that
	/// transition moves no bytes and emits no migration. A rising
	/// `promotions` therefore means specifically "a demoted, previously
	/// proven object was re-accessed", not "a new object proved itself".
	pub promotions: u64,

	/// Objects moved from the main queue's fast portion to its slow portion
	/// under fast-tier pressure.
	///
	/// Note the main queue's budget here is `fast_tier_size` minus the
	/// one-access queue's fixed `k_in * max_size` reservation, not the whole
	/// fast tier — so demotions begin earlier, at a lower `fast_bytes_used`,
	/// than an equivalently-configured `two_q_hybrid_cache` would show.
	pub demotions: u64,

	/// Terminal removals from the cache — the FIFO queue's tail aging out
	/// without a second access, or the main queue's slow tail under overall
	/// capacity pressure.
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
