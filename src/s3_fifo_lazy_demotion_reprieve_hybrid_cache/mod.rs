/*
 * Copyright (c) Kia Shakiba
 *
 * This source code is licensed under the GNU AGPLv3 license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! Single-instance, segmented-S3-FIFO hybrid cache with a demotion-time
//! reference-bit gate, a fast-tier-resident one-access queue, a
//! mid-slow-segment reference-bit checkpoint, and no ghost queue.
//!
//! Identical architecture and admission/eviction rules to
//! `s3_fifo_ghost_lazy_demotion_fast_admission_midpoint_hybrid_cache` — see
//! that module's docs for the full design — with two changes: there is no
//! ghost queue (removed entirely, since nothing ever populates it here),
//! and a one-access-queue key that ages out without a second access is
//! spliced directly into the slow tier of the main queue instead of being
//! evicted. See
//! `worker::policy::policy_stack::s3_fifo_lazy_demotion_reprieve_hybrid_stack`'s
//! module doc for the full mechanics, including the O(number of currently-
//! fast keys) splice technique used to place the reprieved key exactly
//! adjacent to the fast/slow boundary without ever tagging it Fast.

mod stats;

pub use crate::tiered_buffer::TieredBuffer;
pub use stats::S3FifoLazyDemotionReprieveHybridStats;


