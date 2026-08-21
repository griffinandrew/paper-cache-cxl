/*
 * Copyright (c) Kia Shakiba
 *
 * This source code is licensed under the GNU AGPLv3 license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! Single-instance, segmented-2Q hybrid cache with a ghost queue.
//!
//! Identical architecture and admission/demotion/promotion/eviction rules
//! to `two_q_hybrid_cache` — see that module's docs for the full design —
//! plus a bare-key ghost queue remembering objects that aged out of the
//! one-access FIFO queue without a second access, so a later re-admission
//! is trusted immediately (lands directly in the main queue's fast tier)
//! instead of restarting from the FIFO queue. See
//! `worker::policy::policy_stack::two_q_ghost_hybrid_stack`'s module doc
//! for the full ghost-queue mechanics, lifecycle, and the "where a ghost
//! hit lands" design note.

mod stats;

pub use crate::tiered_buffer::TieredBuffer;
pub use stats::TwoQGhostHybridStats;


