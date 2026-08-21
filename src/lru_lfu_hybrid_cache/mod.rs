/*
 * Copyright (c) Kia Shakiba
 *
 * This source code is licensed under the GNU AGPLv3 license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! Single-instance hybrid cache with a different eviction discipline per
//! tier: **recency (LRU) in the fast tier, frequency (LFU) in the slow
//! tier**.
//!
//! Like the other hybrids here this is **one** `PaperCache<K, TieredBuffer>`,
//! not two composed instances, and a live object's bytes exist in exactly one
//! tier's allocation at a time (see [`crate::tiered_buffer::TieredBuffer`]
//! and `Object::set_data`). What is new is that the two tiers no longer rank
//! by the same metric, which is what makes this design distinct rather than a
//! reparameterization of `lru_hybrid_cache`:
//!
//! - **Admission**: new object → fast tier, recency head, frequency 1.
//! - **Demotion**: fast tier's LRU tail → slow tier, carrying its
//!   accumulated frequency.
//! - **Promotion**: a slow-tier object reaching `promote_k` accesses → fast
//!   tier's recency head, counter reset.
//! - **Eviction**: the slow tier's minimum-frequency object.
//!
//! In one line: frequency is the admission control *into* DRAM; recency is
//! the retention policy *within* DRAM.
//!
//! The full derivation — why promotion is a fixed threshold rather than the
//! cross-tier frequency comparison `lfu_hybrid_cache` uses, why the fast tier
//! counts a frequency it does not rank by, why that counter is capped, and
//! why an overwrite goes through the same gate a read does — lives in
//! `worker::policy::policy_stack::lru_lfu_hybrid_stack`'s module doc
//! (`PaperPolicy::LruLfuHybrid`).

mod stats;

pub use crate::tiered_buffer::TieredBuffer;
pub use stats::LruLfuHybridStats;


