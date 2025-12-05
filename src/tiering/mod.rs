/*
 * Copyright (c) Kia Shakiba
 *
 * This source code is licensed under the GNU AGPLv3 license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! Tiering Manager Module
//! 
//! This module provides functionality to manage objects between DRAM and PMEM tiers.
//! Objects in DRAM are copies of those in PMEM, with PMEM serving as the source of truth.
//! The tiering manager uses existing eviction stacks to determine which objects to promote
//! to DRAM or demote from DRAM based on a configurable threshold.

pub mod manager;

pub use manager::TieringManager;
pub use manager::TieringConfig;
pub use manager::TieringStats;
