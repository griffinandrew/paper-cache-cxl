/*
 * Copyright (c) Kia Shakiba
 *
 * This source code is licensed under the GNU AGPLv3 license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! Single-instance, segmented-S3-FIFO hybrid cache.
//!
//! Same overall architecture as `lru_hybrid_cache`/`lfu_hybrid_cache`/
//! `two_q_hybrid_cache` — one `PaperCache<K, TieredBuffer>`, not two
//! composed instances. Structurally closest to `two_q_hybrid_cache` (a
//! one-access FIFO queue always in the slow tier, feeding a main queue
//! segmented fast/slow), but the main queue's promotion mechanism is the
//! classic S3-FIFO/CLOCK "lazy, reference-bit-checked" one rather than
//! `two_q_hybrid_cache`'s "eager, reorder-on-every-touch" LRU one:
//!
//! * Admission: every new object is placed at the bottom of the one-access
//!   FIFO queue in the slow tier.
//! * Demotion: the oldest object in the main queue's fast-tier portion
//!   moves to the slow-tier portion when fast-tier space is needed —
//!   unconditional aging, independent of whether the object has been
//!   accessed.
//! * Promotion: a one-access-queue object that is re-accessed is moved
//!   *immediately* (eagerly) to the bottom of the main queue's fast-tier
//!   portion. A main-queue object that is re-accessed only has a reference
//!   bit set — it moves nowhere until it reaches the top of the main
//!   queue's slow-tier portion and is about to be evicted, at which point
//!   (if the bit is set) it is reinserted at the bottom of the fast-tier
//!   portion instead of being evicted (a "second chance"), and the bit is
//!   cleared.
//! * Eviction: the oldest object at the top of the one-access queue is
//!   evicted if it was never re-accessed (always true for anything still
//!   there, by construction of the eager one-access promotion rule); the
//!   oldest object at the top of the main queue is evicted once its
//!   reference bit is found clear during the sweep described above.
//!
//! No ghost queue is kept for one-access-queue objects that age out — same
//! reasoning and precedent as `two_q_hybrid_cache` (confirmed with the user
//! before implementing): an exact-membership check on every admission was
//! judged an unwelcome added cost given admission here already pays a
//! synchronous slow-tier/PMEM write.
//!
//! A live object's bytes exist in exactly one tier's allocation at a time —
//! see [`crate::tiered_buffer::TieredBuffer`] and `Object::set_data`, which
//! together make promotion/demotion an in-place data move rather than a
//! copy. `TieredBuffer` itself lives in the crate-root `tiered_buffer`
//! module, shared with the other hybrid-cache features (all mutually
//! exclusive — see `lib.rs`'s `compile_error!` guards — since each defines
//! its own inherent-method `PaperCache<K, TieredBuffer, S>` impl block).
//!
//! The policy stack lives at
//! `worker::policy::policy_stack::s3_fifo_hybrid_stack::S3FifoHybridStack`
//! (`PaperPolicy::S3FifoHybrid`) and `PolicyWorker` performs the actual tier
//! migrations it reports, recording counters directly on `AtomicStatus`
//! (see `stats` module docs for why).

mod stats;

pub use crate::tiered_buffer::TieredBuffer;
pub use stats::S3FifoHybridStats;


