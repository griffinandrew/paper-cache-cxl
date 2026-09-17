//! The owner's hard requirement, EXECUTED rather than argued: under
//! `merged_object_store`, a policy the store cannot implement must fail at
//! construction instead of silently running a different eviction order.
//!
//! Every other check of this lives at handle level. This one builds a real
//! `PaperCache`, which is the only thing that proves the error propagates out
//! through `MergedStackHandle::new` -> `PolicyWorker::new` -> `WorkerFanout::new`
//! -> `PaperCache::new` rather than being swallowed on the way.
#![cfg(all(feature = "merged_object_store", feature = "lfu_compact_hybrid_cache"))]

use paper_cache::{CacheTierSize, PaperCache, PaperPolicy, TieredBuffer};

const MAX: u64 = 64 * 1_048_576;

/// LFU is implemented now, so it must construct and report itself as LFU.
#[test]
fn lfu_constructs_under_the_merged_store() {
    let cache = PaperCache::<u64, TieredBuffer>::new(
        MAX,
        CacheTierSize::Mib(16),
        PaperPolicy::LfuCompactHybrid,
    );
    assert!(
        cache.is_ok(),
        "merged LFU must construct now that the order exists: {:?}",
        cache.err(),
    );
}

/// A policy the merged store does not implement must be REFUSED. Before this
/// change it constructed happily and ran LRU under the wrong label.
#[test]
fn an_unimplemented_policy_is_refused_not_substituted() {
    let cache = PaperCache::<u64, TieredBuffer>::new(
        MAX,
        CacheTierSize::Mib(16),
        PaperPolicy::TwoQCompactHybrid(0.25),
    );
    let err = match cache {
        Ok(_) => panic!(
            "BUG: an unimplemented policy constructed a cache -- the silent \
             LRU fallback is still reachable from PaperCache::new",
        ),
        Err(e) => e,
    };
    let msg = format!("{err}").to_lowercase();
    assert!(
        msg.contains("implement"),
        "the error should say the policy is not implemented, got: {msg}",
    );
}
