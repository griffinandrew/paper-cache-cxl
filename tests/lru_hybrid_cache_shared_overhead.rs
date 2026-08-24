/*
 * Copyright (c) Kia Shakiba
 *
 * This source code is licensed under the GNU AGPLv3 license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! Tests of the shared-metadata DRAM reservation, with the reservation ON.
//!
//! Run with:
//!   cargo +nightly test --test lru_hybrid_cache_shared_overhead --features lru_hybrid_cache
//!
//! This is a separate test binary -- and therefore a separate PROCESS -- on
//! purpose. `get_hybrid_dram_shared_overhead` reads the process-global
//! `PAPER_DISABLE_SHARED_OVERHEAD` at every cache construction, and the main
//! integration binary sets it to "1" from `ensure_pmem_allocator_warm()` so
//! its toy-scale mechanics tests get value-only semantics. Tests of the
//! reservation itself cannot share that process: flipping the variable back
//! races every sibling test constructing a cache on another thread. Here the
//! variable is simply never set, so every construction in this binary gets
//! the production default.
//!
//! The demotion fixture below is self-calibrating: it measures the accounted
//! `ObjectSize` of its own payload via `cache.size()`, then chooses a
//! fast-tier budget that puts the values at ~85% of it -- safely below the
//! 98% high watermark, so with the reservation zeroed NO demotion could
//! occur. Any demotion the test then observes is attributable only to the
//! per-object metadata reservation.

#[cfg(feature = "lru_hybrid_cache")]
mod shared_overhead_tests {
    use paper_cache::{PaperCache, PaperPolicy, TieredBuffer, CacheTierSize};

    const TIMEOUT: std::time::Duration = std::time::Duration::from_secs(20);
    const PAYLOAD: &[u8] = b"shared overhead probe";
    const N: u32 = 400;

    fn wait_until(timeout: std::time::Duration, mut predicate: impl FnMut() -> bool) -> bool {
        let deadline = std::time::Instant::now() + timeout;
        loop {
            if predicate() {
                return true;
            }
            if std::time::Instant::now() > deadline {
                return false;
            }
            std::thread::sleep(std::time::Duration::from_millis(20));
        }
    }

    /// The stack-accounted bytes of one `PAYLOAD` object, read back through
    /// the `fast_bytes_used` gauge of a throwaway cache whose budgets are far
    /// too large for anything to migrate. Deliberately NOT `cache.size()`:
    /// that figure embeds the per-object overhead charge (measured: 126 vs
    /// the stack's 41 for this payload), while the fast-tier watermark
    /// compares the stack's own value-byte accounting against the budget --
    /// calibrating on anything else makes the 85% claim below false.
    fn accounted_size() -> u64 {
        let probe = PaperCache::<u32, TieredBuffer>::new(
            1_000_000,
            CacheTierSize::Bytes(100_000),
            PaperPolicy::LruHybrid,
        )
        .expect("probe cache should construct");
        probe.set(1u32, PAYLOAD, None).expect("probe set");
        assert!(
            wait_until(TIMEOUT, || probe.hybrid_stats().fast_bytes_used > 0),
            "probe gauge never refreshed"
        );
        probe.hybrid_stats().fast_bytes_used
    }

    /// A fast-tier budget below one object's metadata reservation must admit
    /// no bytes at all. (Port of the test that previously lived in the main
    /// integration binary and had to flip the env var around itself.)
    #[test]
    fn reservation_is_active_by_default() {
        let cache = PaperCache::<u32, TieredBuffer>::new(
            1_000_000,
            CacheTierSize::Bytes(40),
            PaperPolicy::LruHybrid,
        )
        .expect("cache should construct");

        for key in 1u32..=3 {
            cache.set(key, b"tiny value bytes", None).expect("set");
        }
        cache.get(&1u32).expect("get");

        // Give a would-be promotion ample time, then require it did NOT
        // happen: 40 bytes cannot hold one object's per-object metadata.
        std::thread::sleep(std::time::Duration::from_millis(1500));
        let stats = cache.hybrid_stats();
        assert_eq!(
            stats.fast_bytes_used, 0,
            "fast tier admitted bytes despite a budget below one object's metadata overhead"
        );
    }

    /// With values alone at ~85% of the budget (below the 98% trigger), any
    /// demotion can only come from the metadata reservation -- and it must
    /// demote, never evict.
    #[test]
    fn reservation_forces_demotion_values_alone_would_not() {
        let s = accounted_size();

        // ceil(N*s / 0.85): values occupy <= 85% of the budget by construction.
        let budget = (u64::from(N) * s * 100).div_ceil(85);
        assert!(
            u64::from(N) * s * 100 <= budget * 85,
            "fixture arithmetic drifted: {N} objects of {s} accounted bytes exceed 85% of {budget}"
        );

        let cache = PaperCache::<u32, TieredBuffer>::new(
            1_000_000,
            CacheTierSize::Bytes(budget),
            PaperPolicy::LruHybrid,
        )
        .expect("cache should construct");

        for key in 1u32..=N {
            cache.set(key, PAYLOAD, None).expect("set should succeed");
        }

        assert!(
            wait_until(TIMEOUT, || cache.hybrid_stats().demotions >= 1),
            "values sit at 85% of the fast budget, so only the metadata \
             reservation can trigger demotion -- and none was observed"
        );

        // The reservation responds with demotions only: nothing is evicted,
        // every key survives.
        let stats = cache.hybrid_stats();
        assert_eq!(stats.evictions, 0, "the DRAM cap must demote, never evict");
        let present = (1u32..=N).filter(|key| cache.has(key)).count();
        assert_eq!(present, N as usize, "every key must survive a reservation-driven demotion");
    }
}
