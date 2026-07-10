/*
 * Copyright (c) Kia Shakiba
 *
 * This source code is licensed under the GNU AGPLv3 license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! Single-instance, segmented-LRU hybrid cache.
//!
//! Unlike `hybridcache` (`S3FifoHybridCache`), which composes two independent
//! `PaperCache` instances, `lru_hybrid_cache` is **one** `PaperCache<K,
//! TieredBuffer>`. The fast (DRAM) and slow (PMEM) tiers are a single logical
//! LRU queue segmented by a byte-budgeted boundary: every new object is
//! admitted at the top of the fast tier; as objects age past the boundary
//! they are demoted (physically moved) to the slow tier; accessing a
//! slow-tier object promotes (physically moves) it back to the top of the
//! fast tier; when overall cache capacity is exhausted, the least recently
//! accessed slow-tier object is evicted.
//!
//! A live object's bytes exist in exactly one tier's allocation at a time —
//! see [`buffer::TieredBuffer`] and `Object::set_data`, which together make
//! promotion/demotion an in-place data move rather than a copy.
//!
//! The policy stack lives at
//! `worker::policy::policy_stack::lru_hybrid_stack::LruHybridStack`
//! (`PaperPolicy::LruHybrid`) and `PolicyWorker` performs the actual tier
//! migrations it reports, recording counters directly on `AtomicStatus`
//! (see `stats` module docs for why).

mod buffer;
mod stats;

pub use buffer::TieredBuffer;
pub use stats::LruHybridStats;
