/*
 * Copyright (c) Kia Shakiba
 *
 * This source code is licensed under the GNU AGPLv3 license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! The **full**, three-queue 2Q hybrid cache, with a fast-tier probation
//! queue.
//!
//! Same architecture as every other hybrid design here — one
//! `PaperCache<K, TieredBuffer>`, not two composed instances — but this is
//! the only design whose *queue algorithm* matches `PaperPolicy::TwoQ`'s.
//! `two_q_hybrid_cache` and its fast-admission/reprieve/ghost siblings
//! implement **Simplified 2Q**: a one-hit FIFO admission filter in front of
//! an LRU. This one implements the three-queue algorithm the paper calls
//! "full 2Q", which is a different policy — different queue count, different
//! FIFO-hit rule, different promotion signal, different eviction order, and
//! one more parameter.
//!
//! ## Three live queues
//!
//! * `a1_in` — the probation FIFO every new object is admitted into, capped
//!   at `k_in * max_size`, living in the **fast** (DRAM) tier. Admission is
//!   a plain DRAM write, not a synchronous PMEM allocation on the calling
//!   thread.
//! * `a1_out` — the overflow FIFO an object ages into when it leaves
//!   `a1_in` without a second access, capped at `k_out * max_size`, living
//!   in the **slow** (PMEM) tier. It holds **real resident objects, not
//!   ghosts**: an `a1_out` member is still in the cache, still counted by
//!   `has()`/`len()`, and a hit on it is a genuine cache hit served from
//!   PMEM.
//! * `am` — the main LRU of proven objects, tier-segmented exactly like
//!   `lru_hybrid_cache`'s single queue.
//!
//! ## Object flow
//!
//! * **Admission**: into `a1_in`, in the fast tier.
//! * **A hit in `a1_in`**: *nothing at all*. This is 2Q's central rule — a
//!   burst of references to a just-loaded object (a scan touching a page
//!   twice) must not buy promotion — and it is exactly where
//!   `two_q_hybrid_cache` diverges, promoting on that event instead.
//! * **`a1_in` overflow**: the tail is **demoted** into `a1_out` (a real
//!   DRAM→PMEM copy, paid on the `PolicyWorker` thread). Nothing is evicted.
//! * **A hit in `a1_out`**: promotes the object to `am`'s MRU end in the
//!   fast tier — a real PMEM→DRAM move. This is 2Q's promotion signal.
//! * **A hit in `am`**: the usual LRU reorder, plus promotion to the fast
//!   segment if it had been demoted.
//! * **Demotion within `am`**: the LRU tail of the fast segment moves to the
//!   slow segment when DRAM is needed.
//! * **Eviction**: `a1_out`'s tail first, then `a1_in`'s tail, then `am`'s
//!   LRU tail — verbatim from `PaperPolicy::TwoQ`. Terminal eviction
//!   therefore frees PMEM first and DRAM last.
//!
//! ## `k_out` is a live parameter, for the first time
//!
//! `PaperPolicy::TwoQ` writes its `a1_out` budget and never reads it. Here
//! it is what `needs_capacity_eviction` reports, so
//! `2q-full-fast-admission-hybrid-<k_in>-<k_out>` is the only two-parameter
//! policy string in the tree. Note the two are denominated against different
//! physical tiers: `k_in * max_size` is a reservation carved out of
//! `fast_tier_size` (DRAM), while `k_out * max_size` bounds PMEM. Pick
//! `k_in` against `fast_tier_size`, not against `max_size` — see
//! `TwoQFullFastAdmissionHybridStack`'s module doc for the full accounting
//! and for the degenerate configuration where `a1_in`'s reservation swallows
//! the whole DRAM budget.
//!
//! A live object's bytes exist in exactly one tier's allocation at a time —
//! see [`crate::tiered_buffer::TieredBuffer`] and `Object::set_data`, which
//! together make promotion/demotion an in-place data move rather than a copy.
//!
//! The policy stack lives at
//! `worker::policy::policy_stack::two_q_full_fast_admission_hybrid_stack::TwoQFullFastAdmissionHybridStack`
//! (`PaperPolicy::TwoQFullFastAdmissionHybrid`) and `PolicyWorker` performs
//! the actual tier migrations it reports, recording counters directly on
//! `AtomicStatus` (see the `stats` module doc for why).

mod stats;

pub use crate::tiered_buffer::TieredBuffer;
pub use stats::TwoQFullFastAdmissionHybridStats;
