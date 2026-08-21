/*
 * Copyright (c) Kia Shakiba
 *
 * This source code is licensed under the GNU AGPLv3 license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! Single-instance, segmented-S3-FIFO hybrid cache with a ghost queue, a
//! demotion-time reference-bit gate, a fast-tier-resident one-access
//! queue, AND a mid-slow-segment reference-bit checkpoint.
//!
//! Identical architecture and admission/eviction rules to
//! `s3_fifo_ghost_lazy_demotion_fast_admission_hybrid_cache` — see that
//! module's docs for the full design — plus one addition: a checkpoint
//! roughly halfway through the SLOW portion of the main queue gives a
//! reaccessed object an early second chance, instead of making it wait
//! until it reaches the eviction tail. See
//! `worker::policy::policy_stack::s3_fifo_ghost_lazy_demotion_fast_admission_midpoint_hybrid_stack`'s
//! module doc for the full mechanics, including the O(1)-amortized cursor
//! that locates "the middle" without an O(n) scan.

mod stats;

pub use crate::tiered_buffer::TieredBuffer;
pub use stats::S3FifoGhostLazyDemotionFastAdmissionMidpointHybridStats;


