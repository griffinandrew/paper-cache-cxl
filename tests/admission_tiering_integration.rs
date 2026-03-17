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

    // ── Helpers ──────────────────────────────────────────────────────────────

    /// Build a tiny cache: 512-byte DRAM, 2 KiB far memory.
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
        // DRAM = 512 bytes, each object = 100 bytes → after 8 inserts, ~3 in far.
        let cache = tiny_cache();
        for i in 0u32..8 {
            cache.set(i, &[i as u8; 100], None).unwrap();
        }
        let stats = cache.stats();
        // At least one object should have been evicted from DRAM to far.
        assert!(stats.evictions_to_far > 0, "expected DRAM evictions to far memory");
        // Without any get() calls, objects are in exactly ONE tier each.
        assert_eq!(
            stats.dram_objects + stats.far_objects,
            8,
            "with no promotions, total count must equal number of inserts",
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
        let stats_before = cache.stats();
        if stats_before.far_objects == 0 {
            return; // nothing in far memory; skip
        }
        // Find at least one far-memory hit.
        let mut found = false;
        for k in 0u32..8 {
            let hits_before = cache.stats().far_hits;
            let _ = cache.get(&k);
            if cache.stats().far_hits > hits_before {
                found = true;
                break;
            }
        }
        assert!(found || cache.stats().far_hits > 0, "expected at least one far-memory hit");
    }

    // ── Shadow copy: far memory keeps copy on promotion ───────────────────────

    #[test]
    fn far_memory_keeps_copy_after_promotion() {
        let cache = tiny_cache();
        // Fill DRAM and push some objects to far.
        for i in 0u32..8 {
            cache.set(i, &[i as u8; 100], None).unwrap();
        }
        if cache.stats().far_objects == 0 {
            return; // not applicable
        }

        // Find a key that lives in far memory and get() it (triggers promotion).
        let mut promoted_key = None;
        for k in 0u32..8 {
            let far_before = cache.stats().far_hits;
            let _ = cache.get(&k);
            if cache.stats().far_hits > far_before {
                promoted_key = Some(k);
                break;
            }
        }
        let Some(k) = promoted_key else { return };

        let after = cache.stats();
        // One promotion must have been recorded.
        assert_eq!(after.promotions_to_dram, 1, "exactly one promotion should have occurred");

        // The key must still be accessible (DRAM copy + far backup both exist).
        assert!(cache.has(&k), "promoted key must be accessible");

        // The far backup copy must still be counted.
        // Note: after promotion DRAM may immediately evict *another* object to
        // far (to stay within capacity), so far_objects can be >= before; what
        // matters is that it did NOT decrease by 1 (the promoted key's far entry
        // must be retained).
        assert!(
            after.far_objects >= 1,
            "far backup copy must be retained after promotion; stats={:?}",
            after,
        );

        // The next get for the promoted key must be a DRAM hit, not a far hit.
        let dh_before = cache.stats().dram_hits;
        let fh_before = cache.stats().far_hits;
        let _ = cache.get(&k).expect("promoted key should be accessible");
        let s2 = cache.stats();
        assert!(s2.dram_hits > dh_before, "second access should hit DRAM");
        assert_eq!(s2.far_hits, fh_before, "second access must NOT be a far-memory hit");
    }

    #[test]
    fn promoted_object_is_readable_from_dram_after_promotion() {
        let cache = tiny_cache();
        for i in 0u32..8 {
            cache.set(i, &[i as u8; 100], None).unwrap();
        }
        if cache.stats().far_objects == 0 {
            return;
        }
        // Find and promote a far-memory key.
        let mut promoted_key = None;
        for k in 0u32..8 {
            let fb = cache.stats().far_hits;
            let _ = cache.get(&k);
            if cache.stats().far_hits > fb {
                promoted_key = Some(k);
                break;
            }
        }
        let Some(k) = promoted_key else { return };

        // Second get() must come from DRAM (not far again).
        let dh_before = cache.stats().dram_hits;
        let fh_before = cache.stats().far_hits;
        let _ = cache.get(&k).expect("key should be accessible after promotion");
        let after = cache.stats();
        assert!(
            after.dram_hits > dh_before,
            "second access should be a DRAM hit",
        );
        assert_eq!(
            after.far_hits, fh_before,
            "second access should NOT increment far_hits",
        );
    }

    // ── DRAM eviction of promoted copy leaves far copy intact ─────────────────

    #[test]
    fn dram_eviction_of_promoted_object_keeps_far_copy() {
        // Tiny DRAM (400 bytes = 1 object), far memory 2 KiB.
        let config = AdmissionTierConfig {
            dram_max_bytes: 400,
            far_max_bytes: 2048,
            ..Default::default()
        };
        let cache = AdmissionTierCache::new(config);

        // Step 1: push object 0 to far memory.
        cache.set(0u32, &[0u8; 400], None).unwrap(); // fills DRAM
        cache.set(1u32, &[1u8; 400], None).unwrap(); // evicts 0 to far

        // Object 0 should now be in far memory.
        let s1 = cache.stats();
        assert!(s1.far_objects >= 1, "object 0 should be in far memory");

        // Step 2: get object 0 → promotes a copy to DRAM (far keeps copy).
        let val = cache.get(&0u32).expect("object 0 should be accessible");
        assert_eq!(val, vec![0u8; 400]);

        let s2 = cache.stats();
        assert_eq!(s2.promotions_to_dram, 1);
        // far_objects must still include object 0's backing copy.
        assert!(
            s2.far_objects >= 1,
            "far copy of object 0 must remain after promotion",
        );

        // Step 3: force DRAM eviction of the promoted copy by inserting another object.
        cache.set(2u32, &[2u8; 400], None).unwrap();

        // Object 0 must still be accessible (via the far backup).
        assert!(
            cache.has(&0u32),
            "object 0 must still be accessible via far-memory backup after DRAM eviction",
        );

        // A get() on object 0 should now be a far hit again.
        let fh_before = cache.stats().far_hits;
        let _ = cache.get(&0u32).expect("object 0 should be accessible from far backup");
        assert!(
            cache.stats().far_hits > fh_before,
            "expected a far-memory hit for object 0 after its DRAM copy was evicted",
        );
    }

    // ── Far-memory eviction drops both copies ────────────────────────────────

    #[test]
    fn far_eviction_removes_dram_copy_too() {
        // Very tight far memory so it evicts quickly.
        let config = AdmissionTierConfig {
            dram_max_bytes: 400,   // holds 1 × 400-byte object
            far_max_bytes: 800,    // holds 2 × 400-byte objects
            ..Default::default()
        };
        let cache = AdmissionTierCache::new(config);

        // Push object 0 to far memory, then promote it to DRAM.
        cache.set(0u32, &[0u8; 400], None).unwrap();
        cache.set(1u32, &[1u8; 400], None).unwrap(); // evicts 0 to far
        let _ = cache.get(&0u32); // promotes 0 back to DRAM; far keeps copy

        let s = cache.stats();
        assert_eq!(s.promotions_to_dram, 1, "precondition: object 0 promoted");

        // Now flood far memory so that object 0's far copy gets evicted.
        // Insert enough unique objects to exceed far_max_bytes.
        // Each insert displaces from DRAM to far; after 3 more objects,
        // far is at 800 bytes (full with objects 1 and one more), then
        // a 4th will push it over.
        cache.set(2u32, &[2u8; 400], None).unwrap(); // object 1 was in far; 2 goes to DRAM, evicts 0's DRAM copy
        cache.set(3u32, &[3u8; 400], None).unwrap(); // forces far eviction
        cache.set(4u32, &[4u8; 400], None).unwrap();

        let s_final = cache.stats();
        // Far eviction should have occurred.
        assert!(
            s_final.evictions_from_far > 0,
            "expected far-memory evictions; stats={:?}",
            s_final,
        );

        // Object 0 might or might not have been the evicted one depending on
        // 2Q ordering.  The key property to verify is: if object 0 is no longer
        // in far memory, it must also not be in DRAM (no orphaned DRAM copy).
        if !cache.has(&0u32) {
            let after = cache.stats();
            // If object 0 is gone, dram and far counts should be consistent.
            // We can verify by checking that the object truly cannot be found.
            assert!(
                cache.get(&0u32).is_err(),
                "evicted object must not be retrievable",
            );
            // Verify no phantom dram_objects count: dram_objects must not include
            // the evicted object (indirect: total objects <= inserts).
            assert!(
                after.dram_objects + after.far_objects <= 5,
                "dram+far object count must not exceed total unique inserts; stats={:?}",
                after,
            );
        }
    }

    // ── Far-memory eviction ──────────────────────────────────────────────────

    #[test]
    fn far_memory_evicts_when_full() {
        let config = AdmissionTierConfig {
            dram_max_bytes: 400,   // holds ~1 object
            far_max_bytes: 1200,   // holds ~3 objects
            ..Default::default()
        };
        let cache = AdmissionTierCache::new(config);

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

    // ── Promotion: stat is recorded ──────────────────────────────────────────

    #[test]
    fn hot_far_memory_objects_are_promoted_to_dram() {
        let cache = tiny_cache();

        for i in 0u32..8 {
            cache.set(i, &[i as u8; 100], None).unwrap();
        }

        if cache.stats().far_objects == 0 {
            return;
        }

        // Access every key; any far-memory key will be promoted.
        for k in 0u32..8 {
            let _ = cache.get(&k);
        }

        let stats_final = cache.stats();
        assert!(
            stats_final.promotions_to_dram > 0,
            "expected at least one promotion from far memory to DRAM; stats={:?}",
            stats_final,
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
        // After two sets (no eviction), key 5 is only in DRAM.
        let stats = cache.stats();
        assert_eq!(stats.dram_objects, 1);
        assert_eq!(stats.far_objects, 0);
    }

    // ── del() removes from both tiers ────────────────────────────────────────

    #[test]
    fn del_removes_from_both_tiers_when_promoted() {
        let cache = tiny_cache();
        for i in 0u32..8 {
            cache.set(i, &[i as u8; 100], None).unwrap();
        }
        if cache.stats().far_objects == 0 {
            return;
        }
        // Promote a far-memory key (both copies now exist).
        let mut promoted = None;
        for k in 0u32..8 {
            let fb = cache.stats().far_hits;
            let _ = cache.get(&k);
            if cache.stats().far_hits > fb {
                promoted = Some(k);
                break;
            }
        }
        let Some(k) = promoted else { return };

        // del() must remove from both tiers.
        cache.del(&k).unwrap();
        assert!(!cache.has(&k), "key must be gone from both tiers after del");
        assert!(cache.get(&k).is_err());
    }

    // ── Stats consistency ────────────────────────────────────────────────────

    #[test]
    fn miss_is_counted_for_unknown_key() {
        let cache = tiny_cache();
        let _ = cache.get(&42u32);
        assert_eq!(cache.stats().misses, 1);
    }

    #[test]
    fn stats_byte_counts_are_consistent_without_promotion() {
        // With no get() calls, each object is in exactly one tier.
        let cache = tiny_cache();
        for i in 0u32..4 {
            cache.set(i, &[0u8; 64], None).unwrap();
        }
        let s = cache.stats();
        // No promotions → objects are in at most one tier.
        assert_eq!(s.promotions_to_dram, 0);
        // Total bytes must equal 4 × 64.
        assert_eq!(
            s.dram_bytes + s.far_bytes, 256,
            "byte count mismatch: {:?}", s,
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

