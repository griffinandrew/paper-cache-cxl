/*
 * Copyright (c) Kia Shakiba
 *
 * This source code is licensed under the GNU AGPLv3 license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! Integration tests for the `clock_compact_hybrid_cache` feature.
//!
//! CLOCK is FIFO plus a reference bit: a hit only sets the bit, nothing moves,
//! and the eviction hand gives a referenced key a second chance -- clears the
//! bit and recycles the key to the front of the queue, promoting it -- before
//! it evicts the first unreferenced one. So this binary runs the generic tiered
//! suite (`common::hybrid_suite!`) and the insertion-order suite it shares with
//! FIFO, and adds the one thing that separates CLOCK from FIFO: a referenced
//! key survives an eviction an unreferenced one does not.
//!
//! Unit tests cannot substitute: they drive the stack directly and never place
//! any bytes, so they cannot see a placement or validation bug.
//!
//! Run with nightly (required for `allocator_api` via `key_value_pmem`):
//!   cargo +nightly test --test clock_compact_hybrid_cache_integration --features clock_compact_hybrid_cache
//!
//! All tier-crossing tests exercise the real slow-node allocator (no
//! shortcuts); see `common::hybrid_suite!` for the one-time warm-up cost.

mod common;

#[cfg(feature = "clock_compact_hybrid_cache")]
mod hybrid_cache_tests {
    use paper_cache::{PaperPolicy, PaperCache, TieredBuffer, CacheTierSize, Tier, CacheError};

    crate::common::hybrid_suite! {
        policy: PaperPolicy::ClockCompactHybrid,
        value_len: 1008,
        demoted_of_two: (1u32, 2u32),
        ttl_demotion: true,
    }

    crate::common::insertion_order_suite!(PaperPolicy::ClockCompactHybrid);

    /// The second chance, end to end. Three keys fill a cache whose fast tier
    /// is too small for any of them (each is built slow); key 1, the oldest, is
    /// hit, which sets its reference bit and moves nothing. A fourth key then
    /// pushes the cache over its size, and the eviction hand walks from the
    /// oldest end: key 1 is referenced -- spared, its bit cleared -- so key 2,
    /// the next oldest, is the victim. FIFO would have evicted key 1.
    #[test]
    fn a_referenced_key_gets_a_second_chance_at_eviction() {
        ensure_pmem_allocator_warm();

        let cache = PaperCache::<u32, TieredBuffer>::new(1_048_576, CacheTierSize::Bytes(10), PaperPolicy::ClockCompactHybrid)
            .expect("cache should construct");

        for key in 1u32..=3 {
            cache.set(key, &value(key as u8), None).expect("set should succeed");
        }

        // Every value is larger than the whole fast tier: built and kept slow.
        for key in 1u32..=3 {
            assert_eq!(cache.tier_of(&key), Some(Tier::Slow), "key {key} is structural");
        }

        // The hit sets key 1's bit. The worker handles it before the next
        // set's event, in the order the events were sent.
        assert_eq!(cache.get(&1u32).unwrap(), value(1));

        // Room for what is held now and half an object more: one more set puts
        // the cache over its size by half an object, and one eviction cures it.
        let used = cache.status().expect("status").used_size();
        cache.resize(used + used / 6).expect("resize should succeed");

        cache.set(4u32, &value(4), None).expect("set should succeed");

        assert!(
            wait_until(MIGRATION_TIMEOUT, || cache.hybrid_stats().evictions >= 1),
            "the fourth set should have forced one eviction",
        );

        // Let the worker settle so the count below is stable.
        std::thread::sleep(std::time::Duration::from_millis(300));

        assert_eq!(cache.hybrid_stats().evictions, 1, "exactly one victim");
        assert!(!cache.has(&2u32), "key 2, the oldest unreferenced key, was the victim");
        assert!(cache.has(&1u32), "key 1 was referenced and must have been spared");
        assert!(cache.has(&3u32));
        assert!(cache.has(&4u32));
    }
}
