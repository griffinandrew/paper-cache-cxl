/*
 * Copyright (c) Kia Shakiba
 *
 * This source code is licensed under the GNU AGPLv3 license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! Integration tests for the `hybridcache` feature.
//!
//! Run with:
//!   cargo test --test hybridcache_integration --features hybridcache

#[cfg(feature = "hybridcache")]
mod hybridcache_tests {
use paper_cache::hybridcache::{HybridCacheConfig, HybridCacheStats, S3FifoHybridCache};

/// Small cache with plenty of headroom so no evictions occur during setup.
fn make_cache() -> S3FifoHybridCache<u32> {
let config = HybridCacheConfig {
total_size: 500_000,
small_ratio: 0.1,
..Default::default()
};
S3FifoHybridCache::<u32>::new(config).expect("failed to create S3FifoHybridCache")
}

// ── basic operations ──────────────────────────────────────────────────────

#[test]
fn test_initial_state_is_empty() {
let cache = make_cache();
assert_eq!(cache.stats(), HybridCacheStats::default());
assert!(!cache.has(&0u32));
assert!(cache.get(&0u32).is_err());
}

#[test]
fn test_basic_set_and_get() {
let cache = make_cache();
let payload: Vec<u8> = (0u8..64).collect();
cache.set(42u32, &payload, None).expect("set failed");
let result = cache.get(&42u32).expect("get failed");
assert_eq!(result, payload);
}

#[test]
fn test_has_present_and_absent() {
let cache = make_cache();
cache.set(1u32, &[0xAA; 32], None).expect("set failed");
assert!(cache.has(&1u32));
assert!(!cache.has(&2u32));
}

#[test]
fn test_get_missing_key_returns_error() {
let cache = make_cache();
let err = cache.get(&99u32).expect_err("expected KeyNotFound");
assert_eq!(err, paper_cache::CacheError::KeyNotFound);
}

// ── tier routing ──────────────────────────────────────────────────────────

/// New items are inserted into the small tier; the first `get` is a
/// small-tier hit.
#[test]
fn test_new_items_start_in_small_tier() {
let cache = make_cache();
cache.set(10u32, &[1u8; 64], None).expect("set failed");
cache.get(&10u32).expect("get failed");
let stats = cache.stats();
assert_eq!(stats.small_hits, 1);
assert_eq!(stats.main_hits, 0);
}

/// `set` always lands in the small tier – never directly in main.
#[test]
fn test_set_goes_to_small_tier_only() {
let cache = make_cache();
for i in 0u32..10 {
cache.set(i, &[i as u8; 32], None).expect("set failed");
}
for i in 0u32..10 {
cache.get(&i).expect("get failed");
}
let stats = cache.stats();
assert_eq!(stats.main_hits, 0);
assert_eq!(stats.small_hits, 10);
}

// ── delete ────────────────────────────────────────────────────────────────

#[test]
fn test_del_from_small_tier() {
let cache = make_cache();
cache.set(60u32, &[0u8; 32], None).expect("set failed");
assert!(cache.has(&60u32));
cache.del(&60u32).expect("del failed");
assert!(!cache.has(&60u32));
assert!(cache.get(&60u32).is_err());
}

#[test]
fn test_del_missing_key_returns_error() {
let cache = make_cache();
let err = cache.del(&999u32).expect_err("expected KeyNotFound");
assert_eq!(err, paper_cache::CacheError::KeyNotFound);
}

// ── wipe ──────────────────────────────────────────────────────────────────

#[test]
fn test_wipe_clears_small_tier() {
let cache = make_cache();
cache.set(80u32, &[1u8; 32], None).expect("set failed");
cache.set(81u32, &[2u8; 32], None).expect("set failed");
cache.wipe().expect("wipe failed");
assert!(!cache.has(&80u32));
assert!(!cache.has(&81u32));
}

// ── stats ─────────────────────────────────────────────────────────────────

#[test]
fn test_miss_counter_increments() {
let cache = make_cache();
let _ = cache.get(&1u32);
let _ = cache.get(&2u32);
let _ = cache.get(&3u32);
assert_eq!(cache.stats().misses, 3);
}

#[test]
fn test_stats_consistency() {
let cache = make_cache();
cache.set(90u32, &[0u8; 32], None).expect("set failed");
cache.set(91u32, &[0u8; 32], None).expect("set failed");
cache.get(&90u32).expect("get 90-1 failed");
cache.get(&90u32).expect("get 90-2 failed");
cache.get(&91u32).expect("get 91-1 failed");
let _ = cache.get(&999u32);
let stats = cache.stats();
assert_eq!(stats.small_hits, 3);
assert_eq!(stats.misses, 1);
assert_eq!(stats.main_hits, 0);
}

// ── configuration ─────────────────────────────────────────────────────────

#[test]
fn test_zero_total_size_returns_error() {
let config = HybridCacheConfig { total_size: 0, ..Default::default() };
assert!(S3FifoHybridCache::<u32>::new(config).is_err());
}

// ── eviction-driven persistence ───────────────────────────────────────────

/// Items evicted from the small tier should appear in the main tier.
///
/// Overfill the small tier (50 × 128 B into a 3 KB small cache) and wait
/// for the PolicyWorker eviction callbacks to propagate items to main.
/// Then verify at least one item can be retrieved from main (small miss +
/// main hit → `main_hits` counter increases).
#[test]
fn test_eviction_writes_to_main_tier() {
let config = HybridCacheConfig {
total_size: 30_000,
small_ratio: 0.1, // ~3 KB small, ~27 KB main
..Default::default()
};
let cache = S3FifoHybridCache::<u32>::new(config).expect("new failed");

let payload = vec![0xABu8; 128];
for i in 0u32..50 {
cache.set(i, &payload, None).expect("set failed");
}

// Give the PolicyWorker background thread time to process all set events
// and fire eviction callbacks so items reach the main tier.
std::thread::sleep(std::time::Duration::from_millis(500));

// Try each key; look for the first one whose `get` goes to the main tier
// (small miss + main hit: main_hits counter increases).
let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
let mut found_in_main = false;
'outer: loop {
for key in 0u32..50 {
let before = cache.stats().main_hits;
if cache.get(&key).is_ok() && cache.stats().main_hits > before {
found_in_main = true;
break 'outer;
}
}
if std::time::Instant::now() > deadline {
break;
}
std::thread::sleep(std::time::Duration::from_millis(20));
}

assert!(
found_in_main,
"expected at least one key to be found in the main tier after small-tier eviction"
);
}
}
