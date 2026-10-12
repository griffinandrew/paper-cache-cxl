/*
 * Copyright (c) Kia Shakiba
 *
 * This source code is licensed under the GNU AGPLv3 license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! Tests of the shared-metadata DRAM reservation under
//! `LruSizedCompactHybrid`, with the reservation ON. Separate binary/process
//! on purpose -- see `lru_compact_hybrid_cache_shared_overhead.rs`'s module
//! doc for why this cannot live in the main integration binary.
//!
//! A deliberate near-copy of the baseline suite. This stack is a compaction of
//! that one and must be behaviourally indistinguishable from it, so it answers
//! the same behavioural questions rather than a reduced set.
//!
//! Constructed through `new_sized_compact_with_gate` rather than `new_sized`:
//! the size-split designs take three sizing scalars and are rejected by
//! `PaperCache::new`, so each has its own constructor -- and that constructor
//! has to seed the policy this feature actually registers. One that hardcoded
//! some other size-split policy would leave this variant registered at every
//! dispatch site and still unconstructible.
//!
//! Run with:
//!   cargo +nightly test --test lru_sized_compact_hybrid_cache_shared_overhead --features lru_sized_compact_hybrid_cache
//!
//! The reservation is split proportionally between the two fast segments by
//! capacity (`LruSizedCompactHybridStack::reserved_shares`), so the fixture gives
//! the large segment a minimal 1-byte capacity to concentrate the
//! reservation on the small segment, whose values alone sit at ~85% of its
//! budget -- below the 95% settle target the fixture was sized at (E1b's
//! default; the default is 0.85 since), which the test runs under in a child
//! process given it (`common::alone_with_env`), as its LRU and LFU siblings do.
//!
//! Since S5 the default reservation is the MEASURED model's -- the cache's own
//! structures' bytes, M -- and a new key whose metadata would not fit the fast
//! tier is refused (`MetadataOverflow`): the key ceiling. This fixture's
//! arithmetic is the PER-OBJECT reservation (`L x omega`), so, as its siblings
//! have since S5, it builds its caches with that model and keeps its key count
//! under that model's ceiling (`N x omega <= F`, F being both segments) with a
//! payload large enough for it. Under the measured model its 15 KB tier
//! (sized for the 21 B payload it had) was smaller than an empty cache's own
//! structures (33,136 B measured, DashMap build), so the ceiling was 0 and
//! every set was refused.
//!
//! Not built under `merged_object_store`: the merged store refuses
//! `LruSizedCompactHybrid` at construction (`PolicyNotImplemented`;
//! `merged_object_store_policy_refusal.rs`), so there is no reservation of
//! this design's to test there -- cfg'd out as `phys_fast_identity.rs` does
//! the designs that store refuses.

mod common;

#[cfg(all(feature = "lru_sized_compact_hybrid_cache", not(feature = "merged_object_store")))]
mod shared_overhead_tests {
    use paper_cache::{CacheTierSize, GateConfig, MetadataModel, PaperCache, TieredBuffer};

    const TIMEOUT: std::time::Duration = std::time::Duration::from_secs(20);
    /// 300 B (was 21), as in the siblings: a value large enough that N keys'
    /// per-object reservation fits the tier their values need (S5's key
    /// ceiling), and small enough that it still pushes the values over the
    /// settle target: `budget` checks both.
    const PAYLOAD: &[u8] = &[7u8; 300];
    const N: u32 = 400;
    // Far above the payload's accounted size, so every object classifies small.
    const SIZE_THRESHOLD: u64 = 1_000;
    /// The large segment's capacity: minimal, but non-zero, since 0 is
    /// rejected.
    const LARGE: u64 = 1;

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

    /// This binary's per-object fixture's model (S5's fallback).
    fn per_object() -> GateConfig {
        let mut gate = GateConfig::default();
        gate.metadata_model = MetadataModel::PerObject;
        gate
    }

    /// `(s, omega)`: the stack-accounted bytes of one `PAYLOAD` object --
    /// read back through the `fast_bytes_used` gauge of a throwaway cache
    /// whose budgets are far too large for anything to migrate -- and the
    /// per-object reservation it made for that one key. Deliberately NOT
    /// `cache.size()`: that figure embeds the per-object overhead charge,
    /// while the fast-tier watermark compares the stack's own value-byte
    /// accounting against the budget -- calibrating on anything else makes
    /// the 85% claim below false.
    fn accounted() -> (u64, u64) {
        let probe = PaperCache::<u32, TieredBuffer>::new_sized_compact_with_gate(
            1_048_576,
            CacheTierSize::Bytes(131_072),
            CacheTierSize::Bytes(131_072),
            CacheTierSize::Bytes(SIZE_THRESHOLD),
            per_object(),
        )
        .expect("probe cache should construct");
        probe.set(1u32, PAYLOAD, None).expect("probe set");
        assert!(
            wait_until(TIMEOUT, || probe.hybrid_stats().fast_bytes_used > 0),
            "probe gauge never refreshed"
        );
        let stats = probe.hybrid_stats();
        (stats.fast_bytes_used, stats.fast_metadata_bytes)
    }

    /// The small segment's budget: values alone at <= 85% of it (below the
    /// 95% settle target), the key count under the key ceiling, and the
    /// small segment's share of the reservation large enough to push the
    /// values over the target.
    fn budget(s: u64, omega: u64) -> u64 {
        let n = u64::from(N);
        let budget = (n * s * 100).div_ceil(85);

        assert!(
            n * s * 100 <= budget * 85,
            "fixture arithmetic drifted: {N} objects of {s} accounted bytes exceed 85% of {budget}"
        );
        assert!(
            n * omega <= budget,
            "fixture: {N} keys' reservation ({} B) exceeds the {budget} B tier -- the key ceiling would refuse keys",
            n * omega,
        );

        // `reserved_shares`: the reservation, split by capacity.
        let small_share = n * omega * budget / (budget + LARGE);

        assert!(
            n * s * 100 > (budget - small_share) * 95,
            "fixture: with {small_share} B of the {N} x {omega} B reservation on the small segment the values \
             still fit the settle target of its {budget} B budget"
        );

        budget
    }

    /// With small-segment values alone at ~85% of the small budget, only the
    /// reservation share carried by that segment can force demotion -- and it
    /// must demote, never evict. The per-object model; every key under the
    /// ceiling (`budget`).
    #[test]
    fn reservation_forces_demotion_values_alone_would_not() {
        // At the 0.95 drain target `budget` was sized against, in a child
        // given it: at the 0.85 default the values alone would sit AT the
        // settle target rather than safely under it.
        crate::common::alone_with_env(module_path!(), "reservation_forces_demotion_values_alone_would_not", crate::common::AT_FIXTURE_DRAIN_TARGET, || {
            let (s, omega) = accounted();
            assert!(s < SIZE_THRESHOLD, "payload must classify small for this fixture");

            let budget = budget(s, omega);

            // A minimal large-segment capacity concentrates the reservation
            // almost entirely on the small segment
            // (`LruSizedCompactHybridStack::reserved_shares` splits it in
            // proportion to each segment's capacity).
            let cache = PaperCache::<u32, TieredBuffer>::new_sized_compact_with_gate(
                1_048_576,
                CacheTierSize::Bytes(budget),
                CacheTierSize::Bytes(LARGE),
                CacheTierSize::Bytes(SIZE_THRESHOLD),
                per_object(),
            )
            .expect("cache should construct");

            crate::common::assert_at_fixture_drain_target(|| {
                let stats = cache.hybrid_stats();
                (stats.effective_fast_capacity, stats.band_s)
            });

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
        });
    }
}
