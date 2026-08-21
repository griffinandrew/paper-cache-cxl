/*
 * Copyright (c) Kia Shakiba
 *
 * This source code is licensed under the GNU AGPLv3 license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! Single-instance, segmented-S3-FIFO hybrid cache with a ghost queue AND a
//! demotion-time reference-bit gate.
//!
//! Identical architecture and admission/promotion/eviction rules to
//! `s3_fifo_ghost_hybrid_cache` — see that module's docs for the full design
//! (including the bare-key ghost queue's lifecycle and the "contiguous
//! front run" invariant) — plus one change: demotion (moving a fast
//! main-queue key to the slow tier under fast-tier pressure) is now
//! reference-bit gated too, not just eviction. See
//! `worker::policy::policy_stack::s3_fifo_ghost_lazy_demotion_hybrid_stack`'s
//! module doc for the full mechanics ("lazy demotion, lazy promotion").

mod stats;

pub use crate::tiered_buffer::TieredBuffer;
pub use stats::S3FifoGhostLazyDemotionHybridStats;


