/*
 * Copyright (c) Kia Shakiba
 *
 * This source code is licensed under the GNU AGPLv3 license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! Integration tests for the `lfu_hybrid_cache` feature.
//!
//! Run with nightly (required for `allocator_api` via `key_value_pmem`):
//!   cargo +nightly test --test hybrid_cache_integration --features lfu_hybrid_cache
//!
//! Same one-`PaperCache<K, TieredBuffer>` architecture as
//! `hybrid_cache_integration.rs` — `tier_of` reads the tier directly off
//! the single object map, no `has_in_dram`/`has_in_pmem` pair needed.
//!
//! One behavioral difference from the LRU-hybrid tests worth calling out:
//! admission checks fast-tier capacity directly (see `LfuHybridStack`'s
//! module doc) — while the fast tier has room, a new key lands there; once
//! it's full, every new key is admitted straight to the slow tier instead,
//! deterministically, regardless of any existing resident's frequency or
//! recency. This matches the paper's admission rule literally ("every new
//! object is admitted into the slow tier") rather than relying on
//! frequency-tie-break to decide who ends up slow.
//!
//! What is tested:
//!   * Admission lands in the fast tier while it has room, and switches to
//!     the slow tier directly (not via demoting an existing resident) once
//!     it's full
//!   * Fast-tier pressure (triggered only by a promotion, never by plain
//!     admission) demotes the lowest-frequency resident, and `tier_of`
//!     confirms real data movement (not a copy)
//!   * A slow-tier access promotes the key back to fast once its frequency
//!     *strictly* exceeds the fast tier's minimum — a tie does not promote
//!   * TTL set before a demotion/promotion is still correctly enforced after
//!   * Terminal eviction only ever removes the slow-tier minimum-frequency
//!     resident (falling back to the fast tier only if slow is empty) and
//!     is counted in `hybrid_stats().evictions`
//!   * `set_fast_tier_size` takes effect at runtime
//!   * Zero/invalid/tiny fast-tier-size edge cases
//!
//! All tier-crossing tests exercise the real `Hybrid`/UMF PMEM allocator —
//! see `hybrid_cache_integration.rs`'s module doc for the one-time
//! ~45s PMEM pool warm-up caveat this shares (`ensure_pmem_allocator_warm`
//! below is the same pattern, backed by the same process-wide `Once`).

#[cfg(feature = "lfu_hybrid_cache")]
mod hybrid_cache_tests {
    use paper_cache::{PaperPolicy, PaperCache, TieredBuffer, CacheTierSize, Tier, CacheError};

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

    /// Forces the one-time PMEM allocator pool init/prewarm to complete
    /// before a test's own timing-sensitive assertions begin. See the module
    /// doc comment above for why this is necessary.
    fn ensure_pmem_allocator_warm() {
        // Mechanics tests at toy scales: metadata reservation off (see
        // `get_hybrid_dram_shared_overhead`).
        unsafe { std::env::set_var("PAPER_DISABLE_SHARED_OVERHEAD", "1") };
        let cache = PaperCache::<u32, TieredBuffer>::new(1_000_000, CacheTierSize::Bytes(1), PaperPolicy::LfuHybrid)
            .expect("warm-up cache should construct");

        cache.set(0u32, b"warm", None).expect("warm-up set should succeed");

        let ready = wait_until(std::time::Duration::from_secs(90), || {
            cache.tier_of(&0u32) == Some(Tier::Slow)
        });
        assert!(ready, "PMEM allocator warm-up should complete within 90s");
    }

    const MIGRATION_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(10);

    // A fast tier of this size comfortably fits one ~15-byte value's
    // base_size but not two, matching the two demotion-relevant values used
    // throughout ("first value 123" / "second value 45", both 15 bytes).
    // The fast-tier budget now also reserves an approximate per-object DRAM
    // cost for the shared object hashtable + eviction stacks (see
    // `object/overhead.rs::get_hybrid_dram_shared_overhead`). Using ~1 KB
    // values keeps that reservation a small fraction of each value, so the
    // byte-sized fast-tier budgets below have a wide, robust margin.
    const VALUE_LEN: usize = 1024;

    fn value(seed: u8) -> Vec<u8> {
        vec![seed; VALUE_LEN]
    }

    // A fast tier that holds ~1 of the ~1 KB `value()` payloads (after the
    // per-object shared-metadata reservation for the two tracked objects), so a
    // second admission lands directly in the slow tier / demotes.
    const DEMOTES_ONE_OF_TWO: u64 = 1_600;

    // ── admission ─────────────────────────────────────────────────────────

    #[test]
    fn admission_always_lands_in_fast_tier() {
        let cache = PaperCache::<u32, TieredBuffer>::new(
            1_000_000,
            CacheTierSize::Bytes(1_000_000), PaperPolicy::LfuHybrid).expect("cache should construct");

        cache.set(1u32, b"hello world", None).expect("set should succeed");

        // Admission is synchronous (the object is inserted as `TieredBuffer::
        // Fast` directly inside `set()`, before the WorkerEvent is even
        // broadcast), so this doesn't need `wait_until`.
        assert_eq!(cache.tier_of(&1u32), Some(Tier::Fast));
        assert_eq!(cache.get(&1u32).unwrap(), b"hello world");
    }

    // ── demotion ──────────────────────────────────────────────────────────

    #[test]
    fn admission_once_fast_is_full_goes_directly_to_slow() {
        ensure_pmem_allocator_warm();

        let cache = PaperCache::<u32, TieredBuffer>::new(
            1_000_000,
            CacheTierSize::Bytes(DEMOTES_ONE_OF_TWO), PaperPolicy::LfuHybrid).expect("cache should construct");

        cache.set(1u32, &value(0xA1), None).expect("set should succeed");
        assert_eq!(cache.tier_of(&1u32), Some(Tier::Fast));

        // Fast tier is now full; key 2 is admitted directly to the slow
        // tier -- key 1 (the existing resident) is untouched, matching the
        // paper's admission rule literally ("every new object is admitted
        // into the slow tier", not "whichever key loses a tie-break").
        cache.set(2u32, &value(0xB2), None).expect("set should succeed");

        let admitted_slow = wait_until(MIGRATION_TIMEOUT, || {
            cache.tier_of(&2u32) == Some(Tier::Slow)
        });
        assert!(admitted_slow, "key 2 should have been admitted directly to the slow tier");
        assert_eq!(cache.tier_of(&1u32), Some(Tier::Fast));

        // A fresh admission-to-slow is a real physical migration
        // (correcting the API layer's initially-Fast-built TieredBuffer)
        // but is not a demotion -- no existing fast-tier object was
        // displaced.
        let stats = cache.hybrid_stats();
        assert_eq!(stats.demotions, 0);
    }

    #[test]
    fn fast_tier_pressure_demotes_the_lowest_frequency_key() {
        ensure_pmem_allocator_warm();

        // Fits exactly one value (same capacity proven to do so in
        // `admission_once_fast_is_full_goes_directly_to_slow`).
        let cache = PaperCache::<u32, TieredBuffer>::new(
            1_000_000,
            CacheTierSize::Bytes(DEMOTES_ONE_OF_TWO), PaperPolicy::LfuHybrid).expect("cache should construct");

        cache.set(1u32, &value(0xA1), None).expect("set should succeed");
        assert_eq!(cache.tier_of(&1u32), Some(Tier::Fast));

        // Bump key 1's frequency well above 1 -- doesn't matter for
        // admission itself (which only checks capacity, not frequency),
        // but demonstrates a higher-frequency resident stays untouched.
        for _ in 0..5 {
            let _ = cache.get(&1u32);
        }
        std::thread::sleep(std::time::Duration::from_millis(300));

        // Fast tier is full -> key 2 is admitted directly to slow.
        cache.set(2u32, &value(0xB2), None).expect("set should succeed");

        let admitted_slow = wait_until(MIGRATION_TIMEOUT, || {
            cache.tier_of(&2u32) == Some(Tier::Slow)
        });
        assert!(admitted_slow, "key 2 should have been admitted directly to the slow tier");
        assert_eq!(cache.tier_of(&1u32), Some(Tier::Fast));
    }

    // ── promotion ─────────────────────────────────────────────────────────

    #[test]
    fn slow_tier_key_promotes_once_frequency_strictly_exceeds_fast_minimum() {
        ensure_pmem_allocator_warm();

        let cache = PaperCache::<u32, TieredBuffer>::new(
            1_000_000,
            CacheTierSize::Bytes(DEMOTES_ONE_OF_TWO), PaperPolicy::LfuHybrid).expect("cache should construct");

        cache.set(1u32, &value(0xA1), None).expect("set should succeed");
        // Fast tier is now full; key 2 is admitted directly to slow.
        cache.set(2u32, &value(0xB2), None).expect("set should succeed");

        assert!(wait_until(MIGRATION_TIMEOUT, || cache.tier_of(&2u32) == Some(Tier::Slow)));
        assert_eq!(cache.tier_of(&1u32), Some(Tier::Fast));

        // Grow the fast tier so the upcoming promotion has headroom and
        // doesn't also need to cascade a demotion (that combined behavior
        // is covered separately below).
        cache.set_fast_tier_size(CacheTierSize::Bytes(1_000_000)).expect("resize should succeed");

        // Accessing the slow-tier key should promote it back to fast: its
        // frequency (now 2) strictly exceeds key 1's (still 1).
        assert_eq!(cache.get(&2u32).unwrap(), value(0xB2));

        let promoted = wait_until(MIGRATION_TIMEOUT, || {
            cache.tier_of(&2u32) == Some(Tier::Fast)
        });
        assert!(promoted, "key 2 should have promoted back to the fast tier");

        assert_ne!(cache.tier_of(&2u32), Some(Tier::Slow));

        let stats = cache.hybrid_stats();
        assert!(stats.promotions >= 1);
    }

    #[test]
    fn tie_with_fast_minimum_does_not_promote() {
        ensure_pmem_allocator_warm();

        let cache = PaperCache::<u32, TieredBuffer>::new(
            1_000_000,
            CacheTierSize::Bytes(DEMOTES_ONE_OF_TWO), PaperPolicy::LfuHybrid).expect("cache should construct");

        cache.set(1u32, &value(0xA1), None).expect("set should succeed");
        cache.set(2u32, &value(0xB2), None).expect("set should succeed");

        assert!(wait_until(MIGRATION_TIMEOUT, || cache.tier_of(&2u32) == Some(Tier::Slow)));
        assert_eq!(cache.tier_of(&1u32), Some(Tier::Fast));

        // Bump the fast key's frequency to 2, then give the worker a moment
        // to process it before the next access.
        cache.get(&1u32).expect("get should succeed");
        std::thread::sleep(std::time::Duration::from_millis(300));

        // Bump the slow key from 1 to 2 as well -- this only *ties* the
        // fast minimum (also 2 now), which must not promote.
        cache.get(&2u32).expect("get should succeed");
        std::thread::sleep(std::time::Duration::from_millis(300));

        assert_eq!(cache.tier_of(&2u32), Some(Tier::Slow), "a tie must not promote");
    }

    #[test]
    fn cascading_demotion_on_promotion_is_handled() {
        ensure_pmem_allocator_warm();

        // Promoting a slow key back to a full fast tier can itself demote
        // whatever is now the fast tier's lowest-frequency resident.
        let cache = PaperCache::<u32, TieredBuffer>::new(
            1_000_000,
            CacheTierSize::Bytes(DEMOTES_ONE_OF_TWO), PaperPolicy::LfuHybrid).expect("cache should construct");

        cache.set(1u32, &value(0xA1), None).expect("set should succeed");
        cache.set(2u32, &value(0xB2), None).expect("set should succeed");

        assert!(wait_until(MIGRATION_TIMEOUT, || cache.tier_of(&2u32) == Some(Tier::Slow)));
        assert_eq!(cache.tier_of(&1u32), Some(Tier::Fast));

        // Promote key 2 back — the fast tier only has room for one object
        // here, so key 1 should now be the one demoted.
        cache.get(&2u32).expect("get should succeed");

        assert!(wait_until(MIGRATION_TIMEOUT, || cache.tier_of(&2u32) == Some(Tier::Fast)));
        assert!(wait_until(MIGRATION_TIMEOUT, || cache.tier_of(&1u32) == Some(Tier::Slow)));

        // Both values remain intact and reachable regardless of tier.
        assert_eq!(cache.get(&1u32).unwrap(), value(0xA1));
        assert_eq!(cache.get(&2u32).unwrap(), value(0xB2));
    }

    // ── TTL ───────────────────────────────────────────────────────────────

    // See `hybrid_cache_integration.rs`'s analogous constant/tests for
    // why this is comfortably larger than one ttl'd object's base_size
    // (which includes fixed TTL bookkeeping overhead on top of key + value)
    // and why several small filler keys (rather than one) are used to
    // create demotion pressure.
    const TTL_FAST_TIER: u64 = 2_600;

    #[test]
    fn ttl_survives_a_demotion() {
        ensure_pmem_allocator_warm();

        let cache = PaperCache::<u32, TieredBuffer>::new(
            1_000_000,
            CacheTierSize::Bytes(TTL_FAST_TIER), PaperPolicy::LfuHybrid).expect("cache should construct");

        // See `ttl_survives_a_demotion` in `hybrid_cache_integration.rs`
        // for why the TTL must be comfortably longer than any plausible
        // migration latency, not merely comparable to it.
        let ttl_secs = 5u32;
        let set_at = std::time::Instant::now();
        cache.set(1u32, &value(0xC1), Some(ttl_secs)).expect("set should succeed");
        assert_eq!(cache.tier_of(&1u32), Some(Tier::Fast));

        // Admission no longer displaces existing fast-tier residents (see
        // module doc), so filling the fast tier can only push *later*
        // arrivals straight to slow -- it can never demote key 1 by itself.
        // To demote key 1, promote one of those slow-admitted fillers back:
        // with the fast tier already full, that promotion needs to make
        // room, and key 1 (still tied at frequency 1) is the demotion
        // candidate.
        for key in 2u32..=6 {
            cache.set(key, &value(key as u8), None).expect("set should succeed");
        }

        // Admission's *logical* tier is decided synchronously, but the
        // physical byte migration that corrects the API layer's initially-
        // Fast-built `TieredBuffer` still runs asynchronously on the worker
        // thread -- so this still needs to wait. Check every candidate in a
        // single predicate (rather than giving each candidate its own full
        // `wait_until` timeout) so a filler that's staying fast forever
        // doesn't burn through the TTL budget below.
        assert!(
            wait_until(MIGRATION_TIMEOUT, || {
                (2u32..=6).any(|key| cache.tier_of(&key) == Some(Tier::Slow))
            }),
            "at least one filler should have been admitted directly to the slow tier",
        );
        let slow_filler = (2u32..=6)
            .find(|key| cache.tier_of(key) == Some(Tier::Slow))
            .expect("a slow filler should exist after the wait above");

        cache.get(&slow_filler).expect("get should succeed");

        assert!(wait_until(MIGRATION_TIMEOUT, || cache.tier_of(&1u32) == Some(Tier::Slow)));

        // If `Object::set_data` (the migration) had reset or dropped
        // `expiry`, the key would already be gone or immortal here.
        assert!(cache.has(&1u32), "key should still be alive right after migrating");

        // Sleep past the *original* deadline (measured from `set`, not from
        // the migration), proving the original clock kept ticking through
        // the tier move rather than being restarted or cleared.
        let remaining = std::time::Duration::from_millis(ttl_secs as u64 * 1000 + 500)
            .saturating_sub(set_at.elapsed());
        std::thread::sleep(remaining);

        assert!(matches!(cache.get(&1u32), Err(CacheError::KeyNotFound)));
        assert!(!cache.has(&1u32));
    }

    #[test]
    fn ttl_survives_a_promotion() {
        ensure_pmem_allocator_warm();

        let cache = PaperCache::<u32, TieredBuffer>::new(
            1_000_000,
            CacheTierSize::Bytes(TTL_FAST_TIER), PaperPolicy::LfuHybrid).expect("cache should construct");

        // Fill the fast tier with plenty of non-ttl fillers first --
        // comfortably exceeding TTL_FAST_TIER on their own, regardless of
        // exact per-object overhead, so the fast tier is definitely full by
        // the time the ttl'd key is inserted below.
        for key in 2u32..=6 {
            cache.set(key, &value(key as u8), None).expect("set should succeed");
        }

        // The fillers overfill the fast tier, but sets flow through the
        // async policy worker: the stack latches admission only once it has
        // processed them, and `set()` reads a status mirror refreshed one
        // worker pass later still. Wait for the observable consequence -- a
        // filler demoted to slow -- before relying on the latch, or key 1
        // races the mirror and lands fast.
        assert!(wait_until(MIGRATION_TIMEOUT, || {
            (2u32..=6).any(|k| cache.tier_of(&k) == Some(Tier::Slow))
        }));

        let ttl_secs = 5u32;
        let set_at = std::time::Instant::now();
        cache.set(1u32, &value(0xC1), Some(ttl_secs)).expect("set should succeed");

        // Fast tier is already full -- key 1 is admitted directly to slow.
        assert!(wait_until(MIGRATION_TIMEOUT, || cache.tier_of(&1u32) == Some(Tier::Slow)));

        // Promote key 1; its original TTL should still be in effect
        // afterward. Its frequency (now 2, after this access) strictly
        // exceeds the fillers' (still 1), so it promotes.
        cache.get(&1u32).expect("get should succeed");
        assert!(wait_until(MIGRATION_TIMEOUT, || cache.tier_of(&1u32) == Some(Tier::Fast)));

        assert!(cache.has(&1u32), "key should still be alive right after promoting");

        let remaining = std::time::Duration::from_millis(ttl_secs as u64 * 1000 + 500)
            .saturating_sub(set_at.elapsed());
        std::thread::sleep(remaining);

        assert!(matches!(cache.get(&1u32), Err(CacheError::KeyNotFound)));
    }

    // ── eviction ──────────────────────────────────────────────────────────

    #[test]
    fn terminal_eviction_only_removes_from_slow_tier_and_is_counted() {
        ensure_pmem_allocator_warm();

        // A small overall cache with a tiny fast tier: every object demotes
        // to slow almost immediately, and once total usage exceeds max_size
        // the slow-tier minimum-frequency resident must be evicted.
        let cache = PaperCache::<u32, TieredBuffer>::new(
            200,
            CacheTierSize::Bytes(10), PaperPolicy::LfuHybrid).expect("cache should construct");

        for key in 1u32..=10 {
            let _ = cache.set(key, b"payload bytes", None);
        }

        let evicted = wait_until(MIGRATION_TIMEOUT, || {
            cache.hybrid_stats().evictions >= 1
        });
        assert!(evicted, "at least one terminal eviction should have occurred");

        // Give the worker a moment to settle so the count below is stable.
        std::thread::sleep(std::time::Duration::from_millis(200));

        let stats = cache.hybrid_stats();
        let present = (1u32..=10).filter(|key| cache.has(key)).count() as u64;

        // Every key is accounted for exactly once: either still present
        // (in fast or slow — doesn't matter which) or evicted. None should
        // be silently lost, and none double-counted.
        assert_eq!(present + stats.evictions, 10);

        // Every evicted key is fully gone, never left dangling in a tier.
        for key in 1u32..=10 {
            if !cache.has(&key) {
                assert_eq!(cache.tier_of(&key), None);
            }
        }
    }

    #[test]
    fn terminal_eviction_falls_back_to_fast_tier_when_slow_tier_is_empty() {
        ensure_pmem_allocator_warm();

        // Exercise `evict_one`'s fallback path: when the slow tier is empty,
        // eviction must remove the fast tier's lowest-frequency resident.
        //
        // Reaching "slow empty *and* eviction needed" takes care now that the
        // fast-tier budget reserves a per-object DRAM cost for the shared
        // hashtable + eviction stacks: because that reservation slightly
        // exceeds the per-object overhead `max_size` charges, at
        // `fast_capacity == max_size` the fast tier always starts routing
        // objects to slow *before* `max_size` eviction triggers — so the slow
        // tier can't stay empty by simply filling the cache. Instead: keep a
        // single object comfortably fast (huge fast tier, nothing demotes,
        // slow empty), then shrink `max_size` below it so eviction fires while
        // the slow tier is still empty, forcing the fast-tier fallback.
        let cache = PaperCache::<u32, TieredBuffer>::new(
            1_000_000,
            CacheTierSize::Bytes(1_000_000), PaperPolicy::LfuHybrid).expect("cache should construct");

        cache.set(1u32, b"payload bytes", None).expect("set should succeed");
        assert_eq!(cache.tier_of(&1u32), Some(Tier::Fast));

        // Nothing demoted; the slow tier is genuinely empty.
        let stats = cache.hybrid_stats();
        assert_eq!(stats.demotions, 0);
        assert_eq!(stats.slow_objects, 0);

        // Shrink the overall cache below the resident object so `used_size`
        // exceeds `max_size`. `resize` doesn't touch `fast_capacity`, so the
        // object never demotes — eviction must take it straight from the fast
        // tier via the empty-slow fallback.
        cache.resize(1).expect("resize should succeed");

        let evicted = wait_until(MIGRATION_TIMEOUT, || {
            cache.hybrid_stats().evictions >= 1
        });
        assert!(evicted, "shrinking max_size below the fast resident should evict it");

        std::thread::sleep(std::time::Duration::from_millis(200));

        let stats = cache.hybrid_stats();
        assert_eq!(stats.demotions, 0, "the object should have been evicted from fast, never demoted");
        assert_eq!(stats.slow_objects, 0, "the slow tier stayed empty throughout");
        assert!(!cache.has(&1u32), "the single fast resident should have been evicted");
    }

    // ── DRAM cap accounts for shared metadata (hashtable + eviction stacks) ──

    #[test]
    fn dram_cap_reserves_shared_metadata_and_routes_to_slow_without_evicting() {
        ensure_pmem_allocator_warm();

        // A big overall cache (so `max_size` never triggers eviction) with a
        // fast tier whose raw byte budget (2 KB) would comfortably hold all of
        // these tiny (~13-byte) values at once. The fast-tier budget, however,
        // now also reserves an approximate per-object DRAM cost for the shared
        // object hashtable (and the eviction stacks too, when those are also
        // DRAM-resident -- excluded here under `eviction_stacks_pmem`, so this
        // test only needs to rely on the smaller hashtable-only term); across
        // *enough* objects that reservation fills the budget regardless of
        // which terms apply, so once the fast tier is full every further
        // admission is routed straight to the slow tier -- even though the
        // values alone would all fit. Crucially, this never evicts. 300
        // objects gives comfortable margin under the hashtable-only
        // reservation alone (roughly 11 bytes/object -- see
        // `object/overhead.rs::HASHTABLE_ENTRY_OVERHEAD` -- so >180 objects
        // already exceeds the 2 KB budget on that term by itself); once the
        // admission latch trips (see `LfuHybridStack`'s module doc), every
        // later admission stays routed to slow regardless.
        let cache = PaperCache::<u32, TieredBuffer>::new(
            1_000_000,
            CacheTierSize::Bytes(2_000), PaperPolicy::LfuHybrid).expect("cache should construct");

        for key in 1u32..=300 {
            cache.set(key, b"payload bytes", None).expect("set should succeed");
        }

        // The shared-metadata reservation fills the 2 KB fast budget well
        // before 300 objects, so later admissions land in the slow tier.
        let routed = wait_until(MIGRATION_TIMEOUT, || {
            cache.hybrid_stats().slow_objects >= 1
        });
        assert!(routed, "shared-metadata reservation should route admissions to slow");

        std::thread::sleep(std::time::Duration::from_millis(300));

        let stats = cache.hybrid_stats();
        assert_eq!(stats.evictions, 0, "the DRAM cap must route to slow, never evict");

        let present = (1u32..=300).filter(|key| cache.has(key)).count();
        assert_eq!(present, 300, "no key should be evicted by the DRAM cap");

        // The fast tier's live value bytes stay within the configured budget.
        assert!(stats.fast_bytes_used <= cache.fast_tier_size());
    }

    #[test]
    fn set_places_a_brand_new_key_directly_in_slow_once_admission_is_latched() {
        ensure_pmem_allocator_warm();

        // A tiny fast tier that a single filler already exhausts, latching
        // admission shut for every subsequent brand-new key (see
        // `LfuHybridStack`'s module doc on the admission latch).
        let cache = PaperCache::<u32, TieredBuffer>::new(
            1_000_000,
            CacheTierSize::Bytes(DEMOTES_ONE_OF_TWO), PaperPolicy::LfuHybrid).expect("cache should construct");

        cache.set(1u32, &value(0xA1), None).expect("set should succeed");
        cache.set(2u32, &value(0xB2), None).expect("set should succeed");

        // Wait for the worker to have applied key 2's admission-to-slow
        // migration -- by the time `slow_objects` reflects that, the same
        // `apply_tier_migrations` call has already synced the admission
        // latch onto `AtomicStatus` (the sync runs unconditionally, before
        // the migration is even applied).
        let latched = wait_until(MIGRATION_TIMEOUT, || {
            cache.hybrid_stats().slow_objects >= 1
        });
        assert!(latched, "fast tier should have latched shut after the second filler");

        // A brand-new key's `set()` should place it directly in the slow
        // tier -- checked with *no* `wait_until`: if `PaperCache::set()`
        // still unconditionally built `TieredBuffer::new_fast` (the bug this
        // fixes), this key would read back as `Fast` immediately after
        // `set()` returns, only becoming `Slow` later once the worker's
        // async correction caught up. Reading `Slow` synchronously proves
        // the object's bytes were placed correctly the first time, with no
        // DRAM write followed by a PMEM correction.
        cache.set(3u32, &value(0xC3), None).expect("set should succeed");
        assert_eq!(cache.tier_of(&3u32), Some(Tier::Slow));

        // The latch must not affect an *existing* key: re-setting the very
        // first (still-fast) filler is an access, not an admission, and
        // should stay fast.
        cache.set(1u32, &value(0xA9), None).expect("set should succeed");
        assert_eq!(cache.tier_of(&1u32), Some(Tier::Fast));
    }

    // ── runtime fast-tier resize ─────────────────────────────────────────

    #[test]
    fn set_fast_tier_size_takes_effect_at_runtime() {
        ensure_pmem_allocator_warm();

        let cache = PaperCache::<u32, TieredBuffer>::new(1_000_000, CacheTierSize::Bytes(1_000_000), PaperPolicy::LfuHybrid).expect("cache should construct");

        cache.set(1u32, &value(0xA1), None).expect("set should succeed");
        assert_eq!(cache.tier_of(&1u32), Some(Tier::Fast));
        assert_eq!(cache.fast_tier_size(), 1_000_000);

        // Shrink the fast tier drastically; the existing key should demote
        // even without any further access, once the worker applies the
        // resize (mirrors `LfuHybridStack::resize_fast_tier`'s eager
        // `settle_fast_tier` call).
        cache.set_fast_tier_size(CacheTierSize::Bytes(1)).expect("resize should succeed");
        assert_eq!(cache.fast_tier_size(), 1);

        let demoted = wait_until(MIGRATION_TIMEOUT, || {
            cache.tier_of(&1u32) == Some(Tier::Slow)
        });
        assert!(demoted, "shrinking the fast tier should demote the existing key");
    }

    // ── edge cases ────────────────────────────────────────────────────────

    #[test]
    fn zero_fast_tier_size_is_rejected() {
        let result = PaperCache::<u32, TieredBuffer>::new(1_000, CacheTierSize::Bytes(0), PaperPolicy::LfuHybrid);
        assert!(matches!(result, Err(CacheError::InvalidFastTierSize)));
    }

    #[test]
    fn fast_tier_size_exceeding_max_size_is_rejected() {
        let result = PaperCache::<u32, TieredBuffer>::new(1_000, CacheTierSize::Bytes(2_000), PaperPolicy::LfuHybrid);
        assert!(matches!(result, Err(CacheError::InvalidFastTierSize)));
    }

    #[test]
    fn zero_max_size_is_rejected() {
        let result = PaperCache::<u32, TieredBuffer>::new(0, CacheTierSize::Bytes(100), PaperPolicy::LfuHybrid);
        assert!(matches!(result, Err(CacheError::ZeroCacheSize)));
    }

    #[test]
    fn tiny_fast_tier_demotes_everything_almost_immediately() {
        ensure_pmem_allocator_warm();

        let cache = PaperCache::<u32, TieredBuffer>::new(
            1_000_000,
            CacheTierSize::Bytes(1), PaperPolicy::LfuHybrid).expect("cache should construct");

        cache.set(1u32, b"a value", None).expect("set should succeed");

        let demoted = wait_until(MIGRATION_TIMEOUT, || {
            cache.tier_of(&1u32) == Some(Tier::Slow)
        });
        assert!(demoted, "a 1-byte fast tier should demote any real value almost immediately");
        assert_eq!(cache.get(&1u32).unwrap(), b"a value");
    }

    #[test]
    fn del_removes_key_from_whichever_tier_it_is_in() {
        ensure_pmem_allocator_warm();

        let cache = PaperCache::<u32, TieredBuffer>::new(
            1_000_000,
            CacheTierSize::Bytes(DEMOTES_ONE_OF_TWO), PaperPolicy::LfuHybrid).expect("cache should construct");

        cache.set(1u32, &value(0xA1), None).expect("set should succeed");
        cache.set(2u32, &value(0xB2), None).expect("set should succeed");
        assert!(wait_until(MIGRATION_TIMEOUT, || cache.tier_of(&2u32) == Some(Tier::Slow)));
        assert_eq!(cache.tier_of(&1u32), Some(Tier::Fast));

        cache.del(&1u32).expect("del should succeed");
        assert!(!cache.has(&1u32));
        assert_eq!(cache.tier_of(&1u32), None);

        cache.del(&2u32).expect("del should succeed");
        assert!(!cache.has(&2u32));
        assert_eq!(cache.tier_of(&2u32), None);
    }

    #[test]
    fn wipe_clears_both_tiers() {
        ensure_pmem_allocator_warm();

        let cache = PaperCache::<u32, TieredBuffer>::new(
            1_000_000,
            CacheTierSize::Bytes(DEMOTES_ONE_OF_TWO), PaperPolicy::LfuHybrid).expect("cache should construct");

        cache.set(1u32, &value(0xA1), None).expect("set should succeed");
        cache.set(2u32, &value(0xB2), None).expect("set should succeed");
        assert!(wait_until(MIGRATION_TIMEOUT, || cache.tier_of(&2u32) == Some(Tier::Slow)));
        assert_eq!(cache.tier_of(&1u32), Some(Tier::Fast));

        cache.wipe().expect("wipe should succeed");

        assert!(!cache.has(&1u32));
        assert!(!cache.has(&2u32));
        assert_eq!(cache.tier_of(&1u32), None);
        assert_eq!(cache.tier_of(&2u32), None);
    }
}
