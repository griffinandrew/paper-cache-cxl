/*
 * Copyright (c) Kia Shakiba
 *
 * This source code is licensed under the GNU AGPLv3 license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! Single-instance, segmented-S3-FIFO hybrid cache with a ghost queue, a
//! demotion-time reference-bit gate, AND a fast-tier-resident one-access
//! queue.
//!
//! Identical architecture and eviction/promotion rules to
//! `s3_fifo_ghost_lazy_demotion_hybrid_cache` — see that module's docs for
//! the full design (the ghost queue's lifecycle, the "contiguous front run"
//! invariant, the demotion-time reprieve) — plus one change: the one-access
//! queue now lives in the FAST tier instead of the slow tier, so admission
//! is a cheap DRAM write instead of a synchronous PMEM write. See
//! `worker::policy::policy_stack::s3_fifo_ghost_lazy_demotion_fast_admission_hybrid_stack`'s
//! module doc for the full mechanics, including how the one-access queue's
//! byte budget is now reserved out of the fast-tier size rather than being
//! an independent PMEM-side budget.

mod stats;

pub use crate::tiered_buffer::TieredBuffer;
pub use stats::S3FifoGhostLazyDemotionFastAdmissionHybridStats;


