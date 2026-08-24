/*
 * Copyright (c) Kia Shakiba
 *
 * This source code is licensed under the GNU AGPLv3 license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! Does a demotion actually move bytes onto the slow NUMA node?
//!
//! Run with:
//!   cargo +nightly test --test lru_hybrid_cache_real_dram --features lru_hybrid_cache
//!
//! Every other capacity assertion in the suite reads the policy stack's own
//! `fast_used` counter back through `hybrid_stats().fast_bytes_used` and
//! compares it against the budget that same counter drives -- so a migration
//! that is *counted* but never physically performed satisfies all of them.
//! This test uses `numa_alloc::resident_pages_per_node()`, which walks
//! `/proc/self/numa_maps` and totals resident pages per node: kernel ground
//! truth, independent of anything the cache believes about itself.
//!
//! ## Why this is its own binary, and why only node 1 is asserted
//!
//! `numa_maps` is process-wide, so a sibling test allocating on another
//! thread would land in the same reading. Hence one test, one binary.
//!
//! The measurement is a *delta*, and a delta only counts pages that became
//! resident during the window. jemalloc retains freed pages and hands them
//! back out, so a second workload in the same process largely reuses pages
//! already resident and its delta collapses. Measured across three
//! back-to-back workloads in one process:
//!
//! ```text
//!   workload   node1_delta / slow_bytes_used   node0_delta / payload
//!   first                  1.114                       1.339
//!   second                 1.026                       0.799
//!   third                  0.498                       0.000
//! ```
//!
//! Two things follow. The node-0 figure is not assertable at all: LruHybrid
//! admits every object to DRAM before demoting it, so the whole payload
//! passes through node 0, and jemalloc's retention means node-0 residency
//! does not fall back afterwards -- it ranged 0.00x to 1.34x of the payload
//! with the implementation working correctly. Only node 1 carries a usable
//! signal, and the bound below is set at half of the reported slow bytes to
//! stay clear of the page-reuse effect while still failing decisively if the
//! bytes never move.
//!
//! ## Environment
//!
//! This needs a real two-node NUMA machine. On a single-node host
//! `/proc/self/numa_maps` emits no `N1=` fields at all, so node 1 reads zero
//! and the assertion fails. It fails loudly rather than passing vacuously,
//! which is the intended behaviour: a green run must mean the tiering was
//! observed, never that the observation was unavailable.

#[cfg(feature = "lru_hybrid_cache")]
mod real_dram_tests {
    use paper_cache::{PaperCache, PaperPolicy, TieredBuffer, CacheTierSize};
    use paper_cache::numa_alloc::resident_pages_per_node;

    const PAGE_BYTES: u64 = 4096;
    const OBJECTS: u32 = 4_000;
    const VALUE_LEN: usize = 4096;
    const FAST_TIER_BYTES: u64 = 819_200; // ~5% of the payload: most of it must demote
    const TIMEOUT: std::time::Duration = std::time::Duration::from_secs(60);

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

    #[test]
    fn demoted_bytes_actually_land_on_the_slow_numa_node() {
        // Pay the one-time allocator init before sampling, so its pages are
        // not counted in the delta. Deliberately tiny: a large warm-up would
        // pre-map pages the measured workload would then reuse.
        {
            let warm = PaperCache::<u32, TieredBuffer>::new(
                1_000_000,
                CacheTierSize::Bytes(1_000),
                PaperPolicy::LruHybrid,
            )
            .expect("warm-up cache should construct");
            warm.set(1u32, &[0u8; 64], None).expect("warm-up set");
        }
        std::thread::sleep(std::time::Duration::from_millis(500));

        let (node0_before, node1_before) =
            resident_pages_per_node().expect("should read /proc/self/numa_maps");

        let cache = PaperCache::<u32, TieredBuffer>::new(
            200_000_000, // far above the payload: nothing is evicted, only demoted
            CacheTierSize::Bytes(FAST_TIER_BYTES),
            PaperPolicy::LruHybrid,
        )
        .expect("cache should construct");

        let payload = vec![0xABu8; VALUE_LEN];
        for key in 1..=OBJECTS {
            cache.set(key, &payload, None).expect("set should succeed");
        }

        // Most of the payload has to leave DRAM at this budget.
        assert!(
            wait_until(TIMEOUT, || {
                cache.hybrid_stats().demotions >= u64::from(OBJECTS) / 2
            }),
            "fewer than half the objects were demoted; fixture no longer forces migration"
        );
        // Let the standing pool drain the tail of the batch.
        std::thread::sleep(std::time::Duration::from_millis(2_000));

        let (node0_after, node1_after) =
            resident_pages_per_node().expect("should read /proc/self/numa_maps");
        let stats = cache.hybrid_stats();

        let node1_delta = node1_after.saturating_sub(node1_before) * PAGE_BYTES;
        let node0_delta = node0_after.saturating_sub(node0_before) * PAGE_BYTES;

        assert_eq!(stats.evictions, 0, "fixture should demote, not evict");
        assert!(
            stats.slow_bytes_used > 0,
            "precondition: the stack should report bytes in the slow tier"
        );

        // The claim under test, and the reason it is worth a whole binary:
        // every other assertion in the suite would still pass if the slow
        // tier were a fiction. Two mutants confirm that.
        //
        //   1. Make the migrate closure decline every migration. The stack
        //      still reports these slow bytes; the 485-test lib suite still
        //      passes; only this assertion fires. (The integration suite does
        //      catch this one, via tier_of.)
        //   2. Point `SlowObjects` at the fast node, so the tier tags, the
        //      counters and `tier_of` all stay correct while every "PMEM"
        //      byte is really in DRAM. 485 lib tests and all 17 integration
        //      tests pass. This is the only test in the crate that fails.
        //
        // The second is the one that matters: it is the whole premise of the
        // fork silently not happening, and nothing else notices.
        assert!(
            node1_delta >= stats.slow_bytes_used / 2,
            "stack reports {} slow bytes but node 1 only gained {} bytes \
             ({} pages): the slow tier is not physically on the slow node -- \
             the demotions were either counted without being performed, or \
             performed into the wrong node (node 0 gained {} bytes)",
            stats.slow_bytes_used,
            node1_delta,
            node1_delta / PAGE_BYTES,
            node0_delta,
        );
    }
}
