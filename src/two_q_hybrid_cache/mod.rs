/*
 * Copyright (c) Kia Shakiba
 *
 * This source code is licensed under the GNU AGPLv3 license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! Single-instance, segmented-2Q hybrid cache.
//!
//! Same overall architecture as `lru_hybrid_cache`/`lfu_hybrid_cache` — one
//! `PaperCache<K, TieredBuffer>`, not two composed instances — but the
//! object flow follows the 2Q algorithm:
//!
//! * Admission: every new object is placed in a one-access FIFO queue that
//!   lives entirely in the slow tier.
//! * Demotion: the LRU tail of the main queue's fast-tier portion moves to
//!   the top of its slow-tier portion when fast-tier space is needed.
//! * Promotion: a re-accessed FIFO-queue object moves straight to the top
//!   of the main queue's fast-tier portion; a re-accessed main-queue
//!   slow-tier object moves to the top of the fast-tier portion.
//! * Eviction: the FIFO queue's tail is sacrificed first (an object that
//!   ages out without a second access), falling back to the main queue's
//!   slow tail once the FIFO queue is empty.
//!
//! Unlike classic 2Q (and this crate's own plain `PaperPolicy::TwoQ`), no
//! ghost queue is kept for objects that age out of the FIFO queue — an
//! exact-membership check on every admission (which already pays a
//! synchronous slow-tier/PMEM write here) was judged an unwelcome added
//! cost; see `CLAUDE.md`'s `two_q_hybrid_cache` section for the reasoning
//! and the probabilistic-structure alternative left as future work.
//!
//! A live object's bytes exist in exactly one tier's allocation at a time —
//! see [`crate::tiered_buffer::TieredBuffer`] and `Object::set_data`, which
//! together make promotion/demotion an in-place data move rather than a
//! copy. `TieredBuffer` itself lives in the crate-root `tiered_buffer`
//! module, shared with `lru_hybrid_cache`/`lfu_hybrid_cache` (all three
//! hybrid-cache features are mutually exclusive — see `lib.rs`'s
//! `compile_error!` guards — since each defines its own inherent-method
//! `PaperCache<K, TieredBuffer, S>` impl block).
//!
//! The policy stack lives at
//! `worker::policy::policy_stack::two_q_hybrid_stack::TwoQHybridStack`
//! (`PaperPolicy::TwoQHybrid`) and `PolicyWorker` performs the actual tier
//! migrations it reports, recording counters directly on `AtomicStatus`
//! (see `stats` module docs for why).

mod stats;

pub use crate::tiered_buffer::TieredBuffer;
pub use stats::TwoQHybridStats;


