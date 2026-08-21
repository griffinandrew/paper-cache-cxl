/*
 * Copyright (c) Kia Shakiba
 *
 * This source code is licensed under the GNU AGPLv3 license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! Single-instance, segmented-S3-FIFO hybrid cache with a ghost queue.
//!
//! Identical architecture and admission/demotion/promotion/eviction rules
//! to `s3_fifo_hybrid_cache` — see that module's docs for the full design
//! (including the "contiguous front run" invariant and the eager
//! one-access-queue-promotion vs. lazy main-queue-reference-bit asymmetry)
//! — plus a bare-key ghost queue remembering objects that aged out of the
//! one-access queue without a second access, so a later re-admission is
//! trusted immediately (lands directly in the main queue's fast tier)
//! instead of restarting from the one-access queue. See
//! `worker::policy::policy_stack::s3_fifo_ghost_hybrid_stack`'s module doc
//! for the full ghost-queue mechanics and lifecycle.

mod stats;

pub use crate::tiered_buffer::TieredBuffer;
pub use stats::S3FifoGhostHybridStats;


