/*
 * Copyright (c) Kia Shakiba
 *
 * This source code is licensed under the GNU AGPLv3 license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! Integration tests for the `fifo_compact_hybrid_cache` feature.
//!
//! A deliberate near-copy of the baseline suite. This stack is a compaction of
//! that one and must be behaviourally indistinguishable from it, so it answers
//! the same behavioural questions rather than a reduced set.
//!
//! Unit tests cannot substitute: they drive the stack directly and never place
//! any bytes, so they cannot see a placement or validation bug. Porting these
//! suites to the first four conversions immediately found four real defects
//! that every unit and fidelity test had passed.
//!
//! Run with nightly (required for `allocator_api` via `key_value_pmem`):
//!   cargo +nightly test --test fifo_compact_hybrid_cache_integration --features fifo_compact_hybrid_cache
//!
//! This feature is **one** `PaperCache<K, TieredBuffer>` instance (not two
//! composed `PaperCache`s), so `tier_of` reads the tier directly off the
//! single object map. Modeled on
//! `tests/hybrid_cache_integration.rs`, with promotion-specific tests
//! dropped (FIFO has no promotion policy at all) and two FIFO-defining tests
//! added instead.
//!
//! What is tested:
//!   * Admission always lands in the fast tier
//!   * Fast-tier pressure demotes the oldest object to the slow tier, and
//!     `tier_of` confirms it is gone from the fast tier (real data movement,
//!     not a copy)
//!   * A slow-tier hit does **not** promote the key — it stays slow (the
//!     defining difference from `lru_compact_hybrid_cache`)
//!   * Overwriting an existing key never repositions it or changes its tier
//!     (exercises the tier-aware `set()` fix needed since FIFO's overwrite
//!     rule differs from LRU's)
//!   * TTL set before a demotion is still correctly enforced after
//!   * Terminal eviction only ever removes the slow-tier oldest object and
//!     is counted in `hybrid_stats().evictions`
//!   * `set_fast_tier_size` takes effect at runtime
//!   * Zero/invalid/tiny fast-tier-size edge cases
//!
//! All tier-crossing tests exercise the real `Hybrid`/UMF PMEM allocator (no
//! shortcuts): the very first PMEM allocation in the whole test process
//! triggers a one-time NUMA-node pool init + prewarm that can take on the
//! order of a minute on first touch (observed ~45s in this sandbox) — see
//! `allocator.rs`'s `HybridObjects`. `ensure_pmem_allocator_warm()` below
//! forces that one-time cost to be paid synchronously at the start of every
//! test — since it's backed by the same process-wide `Once`, only the very
//! first call actually waits ~45s; every other call returns almost
//! immediately once the allocator is warm.

mod common;

#[cfg(feature = "fifo_compact_hybrid_cache")]
mod hybrid_cache_tests {
    use paper_cache::{PaperPolicy, PaperCache, TieredBuffer, CacheTierSize, Tier, CacheError};

    crate::common::hybrid_suite! {
        policy: PaperPolicy::FifoCompactHybrid,
        value_len: 1008,
        demoted_of_two: (1u32, 2u32),
        ttl_demotion: true,
    }

    crate::common::insertion_order_suite!(PaperPolicy::FifoCompactHybrid);
}
