/*
 * Copyright (c) Kia Shakiba
 *
 * This source code is licensed under the GNU AGPLv3 license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! Tests of the shared-metadata DRAM reservation under `LruSizedHybrid`, with
//! the reservation ON. Separate binary/process on purpose -- see
//! `lru_hybrid_cache_shared_overhead.rs`'s module doc for why this cannot
//! live in the main integration binary.
//!
//! Run with:
//!   cargo +nightly test --test lru_sized_hybrid_cache_shared_overhead --features lru_sized_hybrid_cache
//!
//! The reservation is split proportionally between the two fast segments by
//! capacity (`LruSizedHybridStack::reserved_shares`), so the fixture gives
//! the large segment a minimal 1-byte capacity to concentrate the
//! reservation on the small segment, whose values alone sit at ~85% of its
//! budget.

#[cfg(feature = "lru_sized_hybrid_cache")]
mod shared_overhead_tests {
    use paper_cache::{PaperCache, TieredBuffer, CacheTierSize};

    const TIMEOUT: std::time::Duration = std::time::Duration::from_secs(20);
    const PAYLOAD: &[u8] = b"shared overhead probe";
    const N: u32 = 400;
    // Far above the payload's accounted size, so every object classifies small.
    const SIZE_THRESHOLD: u64 = 1_000;

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
        let probe = PaperCache::<u32, TieredBuffer>::new_sized(
            1_000_000,
            CacheTierSize::Bytes(100_000),
            CacheTierSize::Bytes(100_000),
            CacheTierSize::Bytes(SIZE_THRESHOLD),
        )
        .expect("probe cache should construct");
        probe.set(1u32, PAYLOAD, None).expect("probe set");
        assert!(
            wait_until(TIMEOUT, || probe.hybrid_stats().fast_bytes_used > 0),
            "probe gauge never refreshed"
        );
        probe.hybrid_stats().fast_bytes_used
    }

    /// With small-segment values alone at ~85% of the small budget, only the
    /// reservation share carried by that segment can force demotion -- and it
    /// must demote, never evict.
    #[test]
    fn reservation_forces_demotion_values_alone_would_not() {
        let s = accounted_size();
        assert!(s < SIZE_THRESHOLD, "payload must classify small for this fixture");

        let budget = (u64::from(N) * s * 100).div_ceil(85);
        assert!(
            u64::from(N) * s * 100 <= budget * 85,
            "fixture arithmetic drifted: {N} objects of {s} accounted bytes exceed 85% of {budget}"
        );

        // A minimal (but non-zero, since 0 is rejected) large-segment
        // capacity concentrates the reservation almost entirely on the small
        // segment (`LruSizedHybridStack::reserved_shares` splits it in
        // proportion to each segment's capacity).
        let cache = PaperCache::<u32, TieredBuffer>::new_sized(
            1_000_000,
            CacheTierSize::Bytes(budget),
            CacheTierSize::Bytes(1),
            CacheTierSize::Bytes(SIZE_THRESHOLD),
        )
        .expect("cache should construct");

        for key in 1u32..=N {
            cache.set(key, PAYLOAD, None).expect("set should succeed");
        }

        assert!(
            wait_until(TIMEOUT, || cache.hybrid_stats().demotions >= 1),
            "small-segment values sit at 85% of that segment's budget, so only \
             the metadata reservation can trigger demotion -- none was observed"
        );

        let stats = cache.hybrid_stats();
        assert_eq!(stats.evictions, 0, "the DRAM cap must demote, never evict");
        let present = (1u32..=N).filter(|key| cache.has(key)).count();
        assert_eq!(present, N as usize, "every key must survive a reservation-driven demotion");
    }
}
