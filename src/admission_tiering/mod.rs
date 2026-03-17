/*
 * Copyright (c) Kia Shakiba
 *
 * This source code is licensed under the GNU AGPLv3 license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! Admission Tiering Module
//!
//! Provides [`AdmissionTierCache`] — a two-tier cache backed by inner
//! PaperCache instances.  Each tier's PaperCache manages its own eviction-
//! policy worker so that `get` events flow into the policy stacks automatically.
//!
//! Enable with the `admission_tiering` feature flag.
//!
//! # Architecture
//!
//! ```text
//!   set(key, value)
//!        │
//!        ▼
//!   ┌──────────────────┐           ┌───────────────────┐
//!   │  DRAM PaperCache │  eviction │  Far PaperCache   │  eviction
//!   │  (hot, 2Q)       │ ────────► │  (shadow, LRU)    │ ──────────► (deleted)
//!   └──────────────────┘           └───────────────────┘
//!        │ WorkerEvent::Get              │ WorkerEvent::Get
//!        ▼                              ▼
//!   policy stack updated           policy stack updated
//! ```
//!
//! - **`set`**: new objects go to both DRAM and far (shadow copy).
//! - **DRAM eviction**: handled automatically by the DRAM PaperCache worker.
//! - **Far-memory access**: hot far-memory objects are promoted back to DRAM.
//! - **Far-memory eviction**: handled automatically by the far PaperCache worker.

pub mod manager;
// two_q is kept as a standalone module; its tests remain valid as unit tests
// for the 2Q algorithm independent of the tier cache.
mod two_q;

pub use manager::{AdmissionTierCache, AdmissionTierConfig, AdmissionTierStats};
