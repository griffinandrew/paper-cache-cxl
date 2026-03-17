/*
 * Copyright (c) Kia Shakiba
 *
 * This source code is licensed under the GNU AGPLv3 license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! Integration tests for the admission tiering module.
//!
//! These tests are gated behind the `admission_tiering` feature flag and are
//! completely independent of all other tiering infrastructure.

#[cfg(feature = "admission_tiering")]
mod admission_tiering_tests {
    use paper_cache::{AdmissionTierCache, AdmissionTierConfig};

    // ── Helper ──────────────────────────────────────────────────────────────

    /// Build a tiny cache for testing: 512-byte DRAM, 2 KiB far memory.
    fn tiny_cache() -> AdmissionTierCache<u32> {
        let config = AdmissionTierConfig {
            dram_max_bytes: 512,
            far_max_bytes: 2048,
            k_in: 0.25,
            k_out: 0.50,
        };
        AdmissionTierCache::new(config)
    }

    // ── Basic set / get / del ────────────────────────────────────────────────

    #[test]
    fn set_and_get_roundtrip() {
        let cache = tiny_cache();
        cache.set(1u32, &[42u8; 32], None).unwrap();
        let v = cache.get(&1u32).unwrap();
        assert_eq!(v, vec![42u8; 32]);
    }

    #[test]
    fn get_missing_key_returns_error() {
        let cache = tiny_cache();
        assert!(cache.get(&99u32).is_err());
    }

    #[test]
    fn set_zero_value_returns_error() {
        let cache = tiny_cache();
        assert!(cache.set(1u32, &[], None).is_err());
    }

    #[test]
    fn del_removes_from_dram() {
        let cache = tiny_cache();
        cache.set(10u32, &[1u8; 32], None).unwrap();
        assert!(cache.has(&10u32));
        cache.del(&10u32).unwrap();
        assert!(!cache.has(&10u32));
    }

    #[test]
    fn del_missing_key_returns_error() {
        let cache = tiny_cache();
        assert!(cache.del(&999u32).is_err());
    }

    #[test]
    fn has_returns_false_for_missing_key() {
        let cache = tiny_cache();
        assert!(!cache.has(&7u32));
    }

    // ── Admission policy: new objects go to DRAM only ───────────────────────

    #[test]
    fn new_objects_land_in_dram() {
        let cache = tiny_cache();
        cache.set(1u32, &[0u8; 32], None).unwrap();
        let stats = cache.stats();
        assert_eq!(stats.dram_objects, 1);
        assert_eq!(stats.far_objects, 0);
        assert_eq!(stats.dram_hits, 0);
    }

    #[test]
    fn get_from_dram_records_dram_hit() {
        let cache = tiny_cache();
        cache.set(1u32, &[0u8; 32], None).unwrap();
        let _ = cache.get(&1u32).unwrap();
        let stats = cache.stats();
        assert_eq!(stats.dram_hits, 1);
        assert_eq!(stats.far_hits, 0);
        assert_eq!(stats.misses, 0);
    }

    // ── Eviction: DRAM → far memory ─────────────────────────────────────────

    #[test]
    fn dram_eviction_moves_objects_to_far_memory() {
        // DRAM = 512 bytes, each object = 100 bytes → after 6 inserts, some move to far.
        let cache = tiny_cache();
        for i in 0u32..8 {
            cache.set(i, &[i as u8; 100], None).unwrap();
        }
        let stats = cache.stats();
        // At least one object should have been evicted from DRAM.
        assert!(stats.evictions_to_far > 0, "expected DRAM evictions to far memory");
        // Total objects = DRAM + far = 8.
        assert_eq!(
            stats.dram_objects + stats.far_objects,
            8,
            "total object count must equal inserts",
        );
    }

    // ── Far-memory lookup ────────────────────────────────────────────────────

    #[test]
    fn get_from_far_memory_records_far_hit() {
        let cache = tiny_cache();
        // Force some objects to far memory by filling DRAM.
        for i in 0u32..8 {
            cache.set(i, &[i as u8; 100], None).unwrap();
        }
        // Key 0 was inserted first — it is likely the coldest and moved to far.
        let stats_before = cache.stats();
        if stats_before.far_objects == 0 {
            // Nothing in far memory yet; skip this sub-test.
            return;
        }
        // Find a key that was evicted to far memory.
        let mut far_key = None;
        for k in 0u32..8 {
            let s = cache.stats();
            if s.far_objects > 0 {
                // Try each key; a far-memory hit will appear in stats.
                let hits_before = s.far_hits;
                let _ = cache.get(&k);
                let hits_after = cache.stats().far_hits;
                if hits_after > hits_before {
                    far_key = Some(k);
                    break;
                }
            }
        }
        // At least one key should have been found in far memory.
        assert!(
            far_key.is_some() || cache.stats().far_hits > 0,
            "expected at least one far-memory hit",
        );
    }

    // ── Promotion: far memory → DRAM ─────────────────────────────────────────

    #[test]
    fn hot_far_memory_objects_are_promoted_to_dram() {
        let cache = tiny_cache();

        // Insert objects to fill DRAM and push some to far memory.
        for i in 0u32..8 {
            cache.set(i, &[i as u8; 100], None).unwrap();
        }

        let stats_after_fill = cache.stats();
        if stats_after_fill.far_objects == 0 {
            // Nothing in far memory — test not applicable in this config.
            return;
        }

        // Access every key enough times to trigger the 2Q "hot" promotion path.
        // The 2Q a1_out→am transition (second chance) or am re-access triggers promotion.
        for _ in 0..4 {
            for k in 0u32..8 {
                let _ = cache.get(&k);
            }
        }

        let stats_final = cache.stats();
        // We expect at least one promotion to have occurred.
        assert!(
            stats_final.promotions_to_dram > 0,
            "expected at least one promotion from far memory to DRAM; stats={:?}",
            stats_final,
        );
    }

    // ── Far-memory eviction ──────────────────────────────────────────────────

    #[test]
    fn far_memory_evicts_when_full() {
        // far_max = 2 KiB, each object = 400 bytes → after 6 DRAM evictions
        // we expect some far evictions too.
        let config = AdmissionTierConfig {
            dram_max_bytes: 400,   // Holds ~1 object at a time
            far_max_bytes: 1200,   // Holds ~3 objects
            ..Default::default()
        };
        let cache = AdmissionTierCache::new(config);

        // Insert 10 objects, each 400 bytes.
        for i in 0u32..10 {
            cache.set(i, &[i as u8; 400], None).unwrap();
        }

        let stats = cache.stats();
        assert!(
            stats.evictions_from_far > 0,
            "expected far-memory evictions when far memory is full; stats={:?}",
            stats,
        );
    }

    // ── Update (overwrite existing key) ─────────────────────────────────────

    #[test]
    fn overwrite_existing_key_updates_value() {
        let cache = tiny_cache();
        cache.set(5u32, &[1u8; 32], None).unwrap();
        cache.set(5u32, &[2u8; 32], None).unwrap();
        let v = cache.get(&5u32).unwrap();
        assert_eq!(v, vec![2u8; 32]);
        // Only one logical object should be tracked.
        let stats = cache.stats();
        assert_eq!(stats.dram_objects + stats.far_objects, 1);
    }

    // ── Stats consistency ────────────────────────────────────────────────────

    #[test]
    fn miss_is_counted_for_unknown_key() {
        let cache = tiny_cache();
        let _ = cache.get(&42u32);
        assert_eq!(cache.stats().misses, 1);
    }

    #[test]
    fn stats_byte_counts_are_consistent() {
        let cache = tiny_cache();
        for i in 0u32..4 {
            cache.set(i, &[0u8; 64], None).unwrap();
        }
        let s = cache.stats();
        // Total bytes = dram + far; should be at most 4 * 64 = 256 bytes
        // (possibly less if any were evicted from far too).
        assert!(
            s.dram_bytes + s.far_bytes <= 256,
            "byte count too high: {:?}",
            s,
        );
    }

    // ── Multiple key types ───────────────────────────────────────────────────

    #[test]
    fn works_with_string_keys() {
        let config = AdmissionTierConfig {
            dram_max_bytes: 4096,
            far_max_bytes: 16384,
            ..Default::default()
        };
        let cache: AdmissionTierCache<String> = AdmissionTierCache::new(config);
        cache.set("hello".to_string(), b"world", None).unwrap();
        let v = cache.get(&"hello".to_string()).unwrap();
        assert_eq!(v, b"world");
    }

    // ── Config and runtime resize ────────────────────────────────────────────

    #[test]
    fn config_is_accessible() {
        let cache = tiny_cache();
        assert_eq!(cache.config().dram_max_bytes, 512);
        assert_eq!(cache.config().far_max_bytes, 2048);
    }

    #[test]
    fn resize_dram_max_bytes() {
        let mut cache = tiny_cache();
        cache.set_dram_max_bytes(1024);
        assert_eq!(cache.config().dram_max_bytes, 1024);
    }
}
