/*
 * Copyright (c) Kia Shakiba
 *
 * This source code is licensed under the GNU AGPLv3 license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! Tests of the shared-metadata DRAM reservation under `LfuCompactHybrid`, with the
//! reservation ON. Separate binary/process on purpose -- see
//! `lru_compact_hybrid_cache_shared_overhead.rs`'s module doc for why this cannot
//! live in the main integration binary (its warm-up helper sets
//! `PAPER_DISABLE_SHARED_OVERHEAD=1` process-wide).
//!
//! Run with:
//!   cargo +nightly test --test lfu_compact_hybrid_cache_shared_overhead --features lfu_compact_hybrid_cache
//!
//! LFU differs from LRU here: admission checks fast-tier capacity directly,
//! so reservation pressure shows up as brand-new keys ROUTED straight to the
//! slow tier (and the admission latch closing), not as demotions of existing
//! residents. The self-calibrating fixture puts value bytes at ~85% of the
//! budget, so with the reservation zeroed every key would fit fast and
//! `slow_objects` would stay 0 forever.
//!
//! Since S5 the default reservation is the MEASURED model's -- the cache's own
//! structures' bytes, M -- and a new key whose metadata would not fit the
//! fast tier is refused (`MetadataOverflow`): the key ceiling. The fixtures
//! below whose arithmetic is the PER-OBJECT reservation (`L x omega`) build
//! their caches with that model, S5's fallback, and keep their key count
//! under its ceiling (`N x omega <= F`) with a payload large enough for it;
//! the measured model's settle is the lib's
//! `the_measured_model_settles_on_the_published_m`.

#[cfg(feature = "lfu_compact_hybrid_cache")]
mod shared_overhead_tests {
    use paper_cache::{CacheTierSize, GateConfig, MetadataModel, PaperCache, PaperPolicy, TieredBuffer};

    const TIMEOUT: std::time::Duration = std::time::Duration::from_secs(20);
    /// 300 B (was 21): a value large enough that N keys' per-object
    /// reservation fits the tier their values need (S5's key ceiling), and
    /// small enough that it still pushes the values over the settle target
    /// at the merged store's omega (78 B): `budget` checks both.
    const PAYLOAD: &[u8] = &[7u8; 300];
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

    /// This binary's per-object fixtures' model (S5's fallback).
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
    fn accounted(policy: PaperPolicy) -> (u64, u64) {
        let probe = PaperCache::<u32, TieredBuffer>::new_with_gate(
            4_194_304,
            CacheTierSize::Bytes(1_048_576),
            policy,
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

    /// The fixture's budget: values alone at <= 85% of it (below the 98%
    /// settle target), the key count under the key ceiling, and the
    /// reservation large enough to push the values over the target.
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
        assert!(
            n * s * 100 > (budget - n * omega) * 98,
            "fixture: with {N} x {omega} B reserved the values still fit the settle target of {budget} B"
        );

        budget
    }

    /// With values alone at ~85% of the budget, only the metadata reservation
    /// can fill the fast tier -- so some admissions must be routed straight
    /// to the slow tier, and nothing may be evicted.
    #[test]
    fn reservation_routes_admissions_to_slow_values_alone_would_fit() {
        let (s, omega) = accounted(PaperPolicy::LfuCompactHybrid);
        let budget = budget(s, omega);

        let cache = PaperCache::<u32, TieredBuffer>::new_with_gate(
            1_048_576,
            CacheTierSize::Bytes(budget),
            PaperPolicy::LfuCompactHybrid,
            per_object(),
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
