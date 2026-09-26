/*
 * Copyright (c) Kia Shakiba
 *
 * This source code is licensed under the GNU AGPLv3 license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! Tests of the shared-metadata DRAM reservation, with the reservation ON.
//!
//! Run with:
//!   cargo +nightly test --test lru_compact_hybrid_cache_shared_overhead --features lru_compact_hybrid_cache
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
//!
//! Since S5 the default reservation is the MEASURED model's -- the cache's own
//! structures' bytes, M -- and a new key whose metadata would not fit the
//! fast tier is refused (`MetadataOverflow`): the key ceiling. The fixtures
//! below whose arithmetic is the PER-OBJECT reservation (`L x omega`) build
//! their caches with that model, S5's fallback, and keep their key count
//! under its ceiling (`N x omega <= F`) with a payload large enough for it;
//! the measured model's settle is the lib's
//! `the_measured_model_settles_on_the_published_m`.

#[cfg(feature = "lru_compact_hybrid_cache")]
mod shared_overhead_tests {
    use paper_cache::{CacheError, CacheTierSize, GateConfig, MetadataModel, PaperCache, PaperPolicy, TieredBuffer};

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

    /// A fast-tier budget below one object's metadata must admit no bytes at
    /// all -- and since S5 no KEY: with the reservation on by default (the
    /// measured model), a 40 B tier cannot hold one key's metadata, so the
    /// key ceiling is 0 and every new key is refused with `MetadataOverflow`,
    /// nothing stored. (Before S5 the keys were stored, none of their bytes
    /// in DRAM.) Port of the test that previously lived in the main
    /// integration binary and had to flip the env var around itself.
    #[test]
    fn reservation_is_active_by_default() {
        let cache = PaperCache::<u32, TieredBuffer>::new(
            1_048_576,
            CacheTierSize::Bytes(40),
            PaperPolicy::LruCompactHybrid,
        )
        .expect("cache should construct");

        for key in 1u32..=3 {
            assert_eq!(
                cache.set(key, b"tiny value bytes", None),
                Err(CacheError::MetadataOverflow),
                "key {key}: a 40 B tier holds no key's metadata",
            );
        }
        assert!(cache.get(&1u32).is_err(), "a refused key was stored");

        let stats = cache.hybrid_stats();
        assert_eq!(
            stats.fast_bytes_used, 0,
            "fast tier admitted bytes despite a budget below one object's metadata overhead"
        );
        assert_eq!(stats.fast_objects + stats.slow_objects, 0, "nothing was stored");
        assert_eq!(stats.metadata_overflows, 3);
    }

    /// With values alone at ~85% of the budget (below the 98% trigger), any
    /// demotion can only come from the metadata reservation -- and it must
    /// demote, never evict. The per-object model; every key under the
    /// ceiling (`budget`).
    #[test]
    fn reservation_forces_demotion_values_alone_would_not() {
        let (s, omega) = accounted(PaperPolicy::LruCompactHybrid);
        let budget = budget(s, omega);

        let cache = PaperCache::<u32, TieredBuffer>::new_with_gate(
            1_048_576,
            CacheTierSize::Bytes(budget),
            PaperPolicy::LruCompactHybrid,
            per_object(),
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
