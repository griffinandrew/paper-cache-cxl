/*
 * Copyright (c) Kia Shakiba
 *
 * This source code is licensed under the GNU AGPLv3 license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! Admission Tiering Module
//!
//! Provides [`AdmissionTierCache`] — a standalone, independent two-tier cache
//! that implements an admission policy with a DRAM hot tier and a far-memory
//! cold tier. Each tier is governed by its own 2Q eviction structure.
//!
//! Enable with the `admission_tiering` feature flag.
//!
//! # Architecture
//!
//! ```text
//!   set(key, value)
//!        │
//!        ▼
//!   ┌──────────┐   eviction    ┌──────────────┐   eviction
//!   │  DRAM    │  ──────────►  │  Far Memory  │  ──────────► (deleted)
//!   │  (hot)   │               │  (cold)      │
//!   │  2Q LRU  │  ◄──────────  │  2Q LRU      │
//!   └──────────┘   promotion   └──────────────┘
//!        │                            │
//!        ▼                            ▼
//!   get() fast path            get() slow path
//! ```
//!
//! - **`set`**: new objects go to DRAM only.
//! - **DRAM eviction**: coldest DRAM objects move to far memory.
//! - **Far-memory access**: hot far-memory objects are promoted back to DRAM.
//! - **Far-memory eviction**: coldest far-memory objects are permanently removed.

pub mod manager;
mod two_q;

pub use manager::{AdmissionTierCache, AdmissionTierConfig, AdmissionTierStats};
