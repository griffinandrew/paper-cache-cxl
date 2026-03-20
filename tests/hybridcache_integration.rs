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

	/// Convenience constructor with a small total capacity to keep tests fast.
	fn make_cache() -> S3FifoHybridCache<u32> {
		let config = HybridCacheConfig {
			total_size: 100_000, // 100 KB
			small_ratio: 0.1,
			freq_threshold: 2,
			..Default::default()
		};
		S3FifoHybridCache::<u32>::new(config).expect("failed to create S3FifoHybridCache")
	}

	// ── basic operations ─────────────────────────────────────────────────────

	/// A freshly created cache should be empty.
	#[test]
	fn test_initial_state_is_empty() {
		let cache = make_cache();

		// Stats should be all-zero before any operation.
		assert_eq!(cache.stats(), HybridCacheStats::default());

		// Lookups on an empty cache should all miss.
		assert!(!cache.has(&0u32));
		assert!(cache.get(&0u32).is_err());
	}

	/// `set` + `get` round-trip returns the exact bytes that were stored.
	#[test]
	fn test_basic_set_and_get() {
		let cache = make_cache();
		let payload: Vec<u8> = (0u8..64).collect();

		cache.set(42u32, &payload, None).expect("set failed");

		let result = cache.get(&42u32).expect("get failed");
		assert_eq!(result, payload, "round-trip data mismatch");
	}

	/// `has` returns true for an inserted key and false for an unknown key.
	#[test]
	fn test_has_present_and_absent() {
		let cache = make_cache();

		cache.set(1u32, &[0xAA; 32], None).expect("set failed");

		assert!(cache.has(&1u32), "inserted key should be present");
		assert!(!cache.has(&2u32), "unknown key should be absent");
	}

	/// `get` on an absent key returns `KeyNotFound`.
	#[test]
	fn test_get_missing_key_returns_error() {
		let cache = make_cache();
		let err = cache.get(&99u32).expect_err("expected KeyNotFound");
		assert_eq!(err, paper_cache::CacheError::KeyNotFound);
	}

	// ── tier routing ─────────────────────────────────────────────────────────

	/// New items should reside in the small tier: the first access is counted
	/// as a small-tier hit.
	#[test]
	fn test_new_items_start_in_small_tier() {
		let cache = make_cache();

		cache.set(10u32, &[1u8; 64], None).expect("set failed");

		// First get: should be a small-tier hit.
		cache.get(&10u32).expect("get failed");

		let stats = cache.stats();
		assert_eq!(stats.small_hits, 1, "expected 1 small-tier hit");
		assert_eq!(stats.main_hits, 0, "expected 0 main-tier hits before promotion");
		assert_eq!(stats.promotions, 0, "no promotion yet");
	}

	/// Reaching `freq_threshold` accesses in the small tier triggers promotion
	/// to the main tier.
	#[test]
	fn test_promotion_after_freq_threshold() {
		let cache = make_cache(); // freq_threshold = 2

		cache.set(20u32, &[2u8; 64], None).expect("set failed");

		// First access: freq becomes 1 – no promotion.
		cache.get(&20u32).expect("get failed");
		assert_eq!(cache.stats().promotions, 0);

		// Second access: freq reaches threshold (2) – promotion.
		cache.get(&20u32).expect("get failed");
		assert_eq!(cache.stats().promotions, 1, "expected 1 promotion");
	}

	/// After promotion, subsequent accesses are served from the main tier.
	#[test]
	fn test_post_promotion_hits_served_from_main() {
		let cache = make_cache(); // freq_threshold = 2

		cache.set(30u32, &[3u8; 64], None).expect("set failed");

		// Drive past threshold: two small-tier accesses trigger promotion.
		cache.get(&30u32).expect("get 1 failed");
		cache.get(&30u32).expect("get 2 failed (promotion happens here)");

		// Third access: item is now in main tier.
		cache.get(&30u32).expect("get 3 failed");

		let stats = cache.stats();
		assert_eq!(stats.promotions, 1);
		assert!(stats.main_hits >= 1, "expected at least 1 main-tier hit after promotion");
	}

	/// Data read after promotion must equal the originally stored bytes.
	#[test]
	fn test_data_integrity_after_promotion() {
		let cache = make_cache();
		let expected: Vec<u8> = (0u8..128).collect();

		cache.set(40u32, &expected, None).expect("set failed");

		// Trigger promotion.
		cache.get(&40u32).expect("get 1 failed");
		cache.get(&40u32).expect("get 2 failed"); // promotion

		let result = cache.get(&40u32).expect("get 3 failed");
		assert_eq!(result, expected, "data integrity mismatch after promotion");
	}

	// ── re-insertion of hot items ─────────────────────────────────────────────

	/// Re-inserting a key that already lives in the main tier should update
	/// the value in the main tier (not demote it back to small).
	#[test]
	fn test_update_hot_item_stays_in_main() {
		let cache = make_cache();

		cache.set(50u32, &[0xFF; 32], None).expect("set failed");

		// Promote.
		cache.get(&50u32).expect("get 1 failed");
		cache.get(&50u32).expect("get 2 failed");

		assert_eq!(cache.stats().promotions, 1);

		// Re-insert with a new value.
		cache.set(50u32, &[0xAB; 32], None).expect("re-set failed");

		// Should be served from main tier with the new value.
		let result = cache.get(&50u32).expect("get after re-set failed");
		assert_eq!(result, vec![0xABu8; 32], "updated value mismatch");

		// Promotions counter should not have increased.
		assert_eq!(cache.stats().promotions, 1, "no extra promotion on re-insert");
	}

	// ── delete ────────────────────────────────────────────────────────────────

	/// `del` removes a key that lives in the small tier.
	#[test]
	fn test_del_from_small_tier() {
		let cache = make_cache();

		cache.set(60u32, &[0u8; 32], None).expect("set failed");
		assert!(cache.has(&60u32));

		cache.del(&60u32).expect("del failed");
		assert!(!cache.has(&60u32));
		assert!(cache.get(&60u32).is_err());
	}

	/// `del` removes a key that has been promoted to the main tier.
	#[test]
	fn test_del_from_main_tier() {
		let cache = make_cache();

		cache.set(70u32, &[0u8; 32], None).expect("set failed");

		// Promote.
		cache.get(&70u32).expect("get 1 failed");
		cache.get(&70u32).expect("get 2 failed");

		assert!(cache.has(&70u32));
		cache.del(&70u32).expect("del failed");
		assert!(!cache.has(&70u32));
	}

	/// `del` on an absent key returns `KeyNotFound`.
	#[test]
	fn test_del_missing_key_returns_error() {
		let cache = make_cache();
		let err = cache.del(&999u32).expect_err("expected KeyNotFound");
		assert_eq!(err, paper_cache::CacheError::KeyNotFound);
	}

	// ── wipe ─────────────────────────────────────────────────────────────────

	/// `wipe` clears both tiers and removes all frequency counters.
	#[test]
	fn test_wipe_clears_both_tiers() {
		let cache = make_cache();

		cache.set(80u32, &[1u8; 32], None).expect("set failed");
		cache.set(81u32, &[2u8; 32], None).expect("set failed");

		// Promote 80.
		cache.get(&80u32).expect("get 1 failed");
		cache.get(&80u32).expect("get 2 failed");

		cache.wipe().expect("wipe failed");

		assert!(!cache.has(&80u32), "80 should be gone after wipe");
		assert!(!cache.has(&81u32), "81 should be gone after wipe");
		assert!(cache.get(&80u32).is_err());
		assert!(cache.get(&81u32).is_err());
	}

	// ── stats ─────────────────────────────────────────────────────────────────

	/// Miss counter increments on each failed lookup.
	#[test]
	fn test_miss_counter_increments() {
		let cache = make_cache();

		let _ = cache.get(&1u32);
		let _ = cache.get(&2u32);
		let _ = cache.get(&3u32);

		assert_eq!(cache.stats().misses, 3);
	}

	/// Small-hit and promotion counters are consistent with observed behaviour.
	#[test]
	fn test_stats_consistency() {
		let cache = make_cache(); // freq_threshold = 2

		cache.set(90u32, &[0u8; 32], None).expect("set failed");
		cache.set(91u32, &[0u8; 32], None).expect("set failed");

		// Access 90 twice → promote.
		cache.get(&90u32).expect("get 90-1 failed");
		cache.get(&90u32).expect("get 90-2 failed");

		// Access 91 once → stays in small.
		cache.get(&91u32).expect("get 91-1 failed");

		let stats = cache.stats();
		// At least 3 small-tier hits (90×2 + 91×1).
		assert!(stats.small_hits >= 3, "expected >= 3 small hits, got {}", stats.small_hits);
		assert_eq!(stats.promotions, 1, "expected exactly 1 promotion");
	}

	// ── configuration ─────────────────────────────────────────────────────────

	/// A custom `freq_threshold` of 1 should promote on the very first access.
	#[test]
	fn test_custom_freq_threshold_of_one() {
		let config = HybridCacheConfig {
			total_size: 100_000,
			small_ratio: 0.1,
			freq_threshold: 1,
			..Default::default()
		};
		let cache = S3FifoHybridCache::<u32>::new(config).expect("new failed");

		cache.set(100u32, &[0xCC; 32], None).expect("set failed");

		// First access should trigger immediate promotion.
		cache.get(&100u32).expect("get failed");

		assert_eq!(cache.stats().promotions, 1);
	}

	/// `ZeroCacheSize` is returned when `total_size` is 0.
	#[test]
	fn test_zero_total_size_returns_error() {
		let config = HybridCacheConfig {
			total_size: 0,
			..Default::default()
		};
		let result = S3FifoHybridCache::<u32>::new(config);
		assert!(result.is_err(), "expected error for zero total_size");
	}
}
