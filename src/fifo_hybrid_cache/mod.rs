/*
 * Copyright (c) Kia Shakiba
 *
 * This source code is licensed under the GNU AGPLv3 license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! Single-instance, segmented-FIFO hybrid cache.
//!
//! Same one-`PaperCache<K, TieredBuffer>` architecture as `lru_hybrid_cache`/
//! `lfu_hybrid_cache`/`two_q_hybrid_cache` (contrast with `hybridcache`,
//! which composes two independent `PaperCache` instances) — but with **no
//! promotion policy at all**. Every new object is admitted at the bottom of
//! the fast tier; objects age strictly by insertion order and are never
//! reordered by subsequent access or overwrite; when fast-tier space is
//! needed, the oldest fast-tier object demotes to the slow tier; when
//! overall cache capacity is exhausted, the oldest slow-tier object is
//! evicted. A `get()` hit never migrates or reorders anything, since FIFO's
//! eviction order depends only on insertion order, never on access.
//!
//! A live object's bytes exist in exactly one tier's allocation at a time —
//! see [`crate::tiered_buffer::TieredBuffer`] and `Object::set_data`, which
//! together make demotion an in-place data move rather than a copy.
//! `TieredBuffer` itself lives in the crate-root `tiered_buffer` module
//! (shared with `lru_hybrid_cache`/`lfu_hybrid_cache`/`two_q_hybrid_cache`,
//! which need the identical type — see that module's doc comment for why).
//!
//! The policy stack lives at
//! `worker::policy::policy_stack::fifo_hybrid_stack::FifoHybridStack`
//! (`PaperPolicy::FifoHybrid`) and `PolicyWorker` performs the actual tier
//! migrations it reports, recording counters directly on `AtomicStatus`
//! (see `stats` module docs for why).

mod stats;

pub use crate::tiered_buffer::TieredBuffer;
pub use stats::FifoHybridStats;
