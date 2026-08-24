/*
 * Copyright (c) Kia Shakiba
 *
 * This source code is licensed under the GNU AGPLv3 license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! Tests of the shared-metadata DRAM reservation under `LfuHybrid`, with the
//! reservation ON. Separate binary/process on purpose -- see
//! `lru_hybrid_cache_shared_overhead.rs`'s module doc for why this cannot
//! live in the main integration binary (its warm-up helper sets
//! `PAPER_DISABLE_SHARED_OVERHEAD=1` process-wide).
//!
//! Run with:
//!   cargo +nightly test --test lfu_hybrid_cache_shared_overhead --features lfu_hybrid_cache
//!
//! LFU differs from LRU here: admission checks fast-tier capacity directly,
//! so reservation pressure shows up as brand-new keys ROUTED straight to the
//! slow tier (and the admission latch closing), not as demotions of existing
//! residents. The self-calibrating fixture puts value bytes at ~85% of the
//! budget, so with the reservation zeroed every key would fit fast and
//! `slow_objects` would stay 0 forever.

#[cfg(feature = "lfu_hybrid_cache")]
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
            PaperPolicy::LfuHybrid,
        )
        .expect("probe cache should construct");
        probe.set(1u32, PAYLOAD, None).expect("probe set");
        assert!(
            wait_until(TIMEOUT, || probe.hybrid_stats().fast_bytes_used > 0),
            "probe gauge never refreshed"
        );
        probe.hybrid_stats().fast_bytes_used
    }

    /// With values alone at ~85% of the budget, only the metadata reservation
    /// can fill the fast tier -- so some admissions must be routed straight
    /// to the slow tier, and nothing may be evicted.
    #[test]
    fn reservation_routes_admissions_to_slow_values_alone_would_fit() {
        let s = accounted_size();
        let budget = (u64::from(N) * s * 100).div_ceil(85);
        assert!(
            u64::from(N) * s * 100 <= budget * 85,
            "fixture arithmetic drifted: {N} objects of {s} accounted bytes exceed 85% of {budget}"
        );

        let cache = PaperCache::<u32, TieredBuffer>::new(
            1_000_000,
            CacheTierSize::Bytes(budget),
            PaperPolicy::LfuHybrid,
        )
        .expect("cache should construct");

        for key in 1u32..=N {
            cache.set(key, PAYLOAD, None).expect("set should succeed");
        }

        assert!(
            wait_until(TIMEOUT, || cache.hybrid_stats().slow_objects >= 1),
            "values sit at 85% of the fast budget, so only the metadata \
             reservation can fill the fast tier -- yet no key was routed slow"
        );

        let stats = cache.hybrid_stats();
        assert_eq!(stats.evictions, 0, "reservation pressure must route/demote, never evict");
        let present = (1u32..=N).filter(|key| cache.has(key)).count();
        assert_eq!(present, N as usize, "every key must survive reservation pressure");
    }
}
