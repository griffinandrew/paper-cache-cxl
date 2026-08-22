/*
 * Copyright (c) Kia Shakiba
 *
 * This source code is licensed under the GNU AGPLv3 license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! Single-instance, segmented-LFU hybrid cache.
//!
//! Same overall architecture as `lru_hybrid_cache` — **one**
//! `PaperCache<K, TieredBuffer>`, not two composed instances — but the
//! fast/slow boundary is frequency-ordered rather than recency-ordered:
//!
//! * Admission: while the fast tier has capacity, new objects are admitted
//!   into the fast tier. Once the fast tier is full, every new object is,
//!   by definition, the least frequently accessed object, so it lands in
//!   the slow tier — see `LfuHybridStack`'s doc comment for how this is
//!   achieved as an emergent result of "always admit fast, let settle
//!   demote if needed" rather than a special-cased admission check.
//! * Demotion: the least frequently accessed fast-tier object moves to the
//!   slow tier when fast-tier space is needed.
//! * Promotion: a slow-tier object moves to the fast tier once its access
//!   frequency strictly exceeds the minimum frequency among fast-tier
//!   residents — which may itself demote the (new) fast-tier minimum.
//! * Eviction: the least frequently accessed slow-tier object is removed
//!   when overall cache capacity is exhausted.
//!
//! A live object's bytes exist in exactly one tier's allocation at a time —
//! see [`crate::tiered_buffer::TieredBuffer`] and `Object::set_data`, which
//! together make promotion/demotion an in-place data move rather than a
//! copy. `TieredBuffer` itself lives in the crate-root `tiered_buffer`
//! module, shared unchanged by every hybrid design.
//!
//! The policy stack lives at
//! `worker::policy::policy_stack::lfu_hybrid_stack::LfuHybridStack`
//! (`PaperPolicy::LfuHybrid`) and `PolicyWorker` performs the actual tier
//! migrations it reports, recording counters directly on `AtomicStatus`
//! (see `stats` module docs for why).
//!
//! NOTE: this module is now only a shim -- a `TieredBuffer` re-export and a
//! `<Design>HybridStats` alias of `HybridStats`. All 18 designs share the two
//! `impl<K, S> PaperCache<K, TieredBuffer, S>` blocks in `lib.rs` and are
//! selected at runtime by the `PaperPolicy` passed to `new()`. The design
//! description above is still accurate; the module structure it implies is not.

mod stats;

pub use crate::tiered_buffer::TieredBuffer;
pub use stats::LfuHybridStats;


