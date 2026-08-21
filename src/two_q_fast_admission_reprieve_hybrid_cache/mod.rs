/*
 * Copyright (c) Kia Shakiba
 *
 * This source code is licensed under the GNU AGPLv3 license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! Single-instance, segmented-2Q hybrid cache with a **fast-tier** one-access
//! queue.
//!
//! Same architecture as every other hybrid design here — one
//! `PaperCache<K, TieredBuffer>`, not two composed instances — and the same
//! 2Q object flow as `two_q_hybrid_cache`, with exactly one change: the
//! one-access FIFO queue's bytes live in the fast (DRAM) tier rather than the
//! slow (PMEM) tier.
//!
//! * Admission: every new object is placed in the one-access FIFO queue,
//!   **in the fast tier** — a plain DRAM write, not a synchronous PMEM
//!   allocation on the calling thread.
//! * Demotion: the LRU tail of the main queue's fast portion moves to its
//!   slow portion when fast-tier space is needed — where "needed" is
//!   measured against `fast_tier_size` minus the FIFO queue's reservation,
//!   not the whole fast tier.
//! * Promotion: a re-accessed FIFO object moves to the top of the main
//!   queue's fast portion (a bookkeeping move — the bytes are already in
//!   DRAM); a re-accessed slow main-queue object moves to the fast portion
//!   (a real PMEM→DRAM data move).
//! * **Reprieve** (the difference from `two_q_fast_admission_hybrid_cache`):
//!   a one-access object that ages out of the FIFO queue without a second
//!   access is spliced onto the *bottom* of the main queue in the slow tier
//!   rather than evicted — it survives, but ranked below every proven object,
//!   as the next thing to go.
//! * Eviction: the main queue's LRU tail (which is where reprieved objects
//!   land), so unproven objects are still sacrificed first.
//!
//! ## Why this design exists
//!
//! Relative to `two_q_fast_admission_hybrid_cache`, the bet is that a
//! one-hit-wonder is worth one asynchronous DRAM→PMEM copy on the chance it
//! is reaccessed, instead of being dropped for free. Note the copy is paid on
//! the `PolicyWorker` thread, not the `set()` path, so SET latency should be
//! unaffected — the cost shows up as worker CPU, PMEM write volume and slow-
//! tier occupancy. Because a reprieved object lands at the LRU tail it may be
//! evicted very soon under steady pressure, so whether the copy pays for
//! itself is genuinely open; see `TwoQFastAdmissionReprieveHybridStack`'s
//! module doc for the placement rationale and the alternative.
//!
//! `two_q_hybrid_cache` implements the paper's admission rule literally
//! ("every new object is placed in the one-access FIFO queue in the slow
//! tier"), which makes every single `set()` pay a synchronous PMEM
//! allocation before the object is even in the cache. That is the intended
//! cost of only ever spending DRAM on proven-hot objects — but it is a real,
//! measured cost. This variant trades it the other way, exactly as
//! `s3_fifo_ghost_lazy_demotion_fast_admission_hybrid_cache` does for the
//! s3-fifo family.
//!
//! The trade is not free: the FIFO queue's byte budget (`k_in * max_size`) is
//! now a reservation **carved out of** `fast_tier_size`, so DRAM that used to
//! be available to proven-hot main-queue objects is instead held by objects
//! with no demonstrated reuse. Since `fifo_capacity` scales with `max_size`
//! while the budget it comes out of is `fast_tier_size` — typically a small
//! fraction of `max_size` — a `k_in` that was unremarkable under
//! `two_q_hybrid_cache` can consume most of the DRAM budget here. See
//! `TwoQFastAdmissionHybridStack`'s module doc for the full accounting.
//!
//! A live object's bytes exist in exactly one tier's allocation at a time —
//! see [`crate::tiered_buffer::TieredBuffer`] and `Object::set_data`, which
//! together make promotion/demotion an in-place data move rather than a copy.
//!
//! The policy stack lives at
//! `worker::policy::policy_stack::two_q_fast_admission_reprieve_hybrid_stack::TwoQFastAdmissionReprieveHybridStack`
//! (`PaperPolicy::TwoQFastAdmissionReprieveHybrid`) and `PolicyWorker` performs the
//! actual tier migrations it reports, recording counters directly on
//! `AtomicStatus` (see the `stats` module doc for why).

mod stats;

pub use crate::tiered_buffer::TieredBuffer;
pub use stats::TwoQFastAdmissionReprieveHybridStats;


