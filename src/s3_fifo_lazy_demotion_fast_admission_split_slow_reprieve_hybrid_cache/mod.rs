/*
 * Copyright (c) Kia Shakiba
 *
 * This source code is licensed under the GNU AGPLv3 license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! Single-instance, segmented-S3-FIFO hybrid cache with a demotion-time
//! reference-bit gate, a fast-tier-resident one-access queue, a two-segment
//! slow tier with a reference-bit check at the crossing, and no ghost
//! queue.
//!
//! Identical architecture and admission/eviction rules to
//! `s3_fifo_lazy_demotion_fast_admission_midpoint_reprieve_hybrid_cache` — see
//! that module's docs for the full design — with one change: the
//! approximate mid-slow-segment cursor is replaced by splitting the slow
//! tier into two physical FIFO segments, with every object's reference bit
//! checked as it crosses between them. See
//! `worker::policy::policy_stack::s3_fifo_lazy_demotion_fast_admission_split_slow_reprieve_hybrid_stack`'s
//! module doc for the full mechanics, including why the predecessor's
//! sampled-cursor checkpoint was replaced.

mod stats;

pub use crate::tiered_buffer::TieredBuffer;
pub use stats::S3FifoLazyDemotionFastAdmissionSplitSlowReprieveHybridStats;


