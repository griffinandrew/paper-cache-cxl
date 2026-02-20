/* 


/* 
 * Copyright (c) Kia Shakiba
 *
 * This source code is licensed under the GNU AGPLv3 license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! Multitiering Manager Module
//!
//! This module provides a completely independent 3-state tiering policy for CXL/PMEM memory
//! hierarchies. Unlike the 2-state `TieringManager`, objects here move through three tiers:
//!
//! 1. **PmemOnly (Cold)**: Object exists entirely in PMEM.
//! 2. **DramPtrToPmem (Warm)**: Metadata/pointer is cached in DRAM for fast routing;
//!    the payload remains in PMEM (zero-copy promotion).
//! 3. **DramAndPmem (Hot)**: Both metadata and payload are physically copied into DRAM.
//!
//! # Eviction (2Q Flow)
//!
//! - **Hot full**: oldest `DramAndPmem` objects are demoted **directly to `PmemOnly`** (never to Warm).
//! - **Warm full**: oldest `DramPtrToPmem` objects are demoted to `PmemOnly`.
//!
//! Byte-based capacity limits (`warm_capacity_bytes` and `hot_capacity_bytes`) are enforced
//! independently for each tier using an LRU (Least Recently Used) strategy.
//!
//! # Thread Safety
//!
//! All mutable state is protected by `std::sync::RwLock`. Tier transitions and byte-counter
//! updates are performed under write locks to avoid counter drift.

use std::{
    collections::{HashMap, VecDeque},
    sync::{Arc, RwLock},
};

use crate::{HashedKey, object::ObjectSize};

/// Configuration for the `MultitieringManager`.
#[derive(Debug, Clone)]
pub struct MultitieringConfig {
    /// Maximum byte capacity for the Warm (`DramPtrToPmem`) tier.
    pub warm_capacity_bytes: u64,

    /// Maximum byte capacity for the Hot (`DramAndPmem`) tier.
    pub hot_capacity_bytes: u64,

    /// Number of accesses before a Cold object is promoted to Warm.
    pub warm_threshold: u64,

    /// Number of accesses before a Warm object is promoted to Hot.
    pub hot_threshold: u64,
}

impl Default for MultitieringConfig {
    fn default() -> Self {
        MultitieringConfig {
            warm_capacity_bytes: 512 * 1024 * 1024, // 512 MB
            hot_capacity_bytes: 1024 * 1024 * 1024,  // 512 MB
            warm_threshold: 2,
            hot_threshold: 4, 
        }
    }
}

/// Statistics tracked by the `MultitieringManager`.
#[derive(Debug, Clone, Default)]
pub struct MultitieringStats {
    /// Total bytes currently in the Warm (`DramPtrToPmem`) tier.
    pub warm_size: u64,

    /// Total bytes currently in the Hot (`DramAndPmem`) tier.
    pub hot_size: u64,

    /// Number of objects in the Cold (`PmemOnly`) tier.
    pub cold_objects: u64,

    /// Number of objects in the Warm (`DramPtrToPmem`) tier.
    pub warm_objects: u64,

    /// Number of objects in the Hot (`DramAndPmem`) tier.
    pub hot_objects: u64,

    /// Total number of promotions (Cold→Warm or Warm→Hot).
    pub promotions: u64,

    /// Total number of demotions (any tier → Cold).
    pub demotions: u64,
}

/// The three-state tier used by `MultitieringManager`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Tier {
    /// Object exists only in PMEM (Cold).
    PmemOnly,

    /// Object metadata/pointer is in DRAM; payload remains in PMEM (Warm, zero-copy).
    DramPtrToPmem,

    /// Both metadata and payload are physically copied/cached in DRAM (Hot).
    DramAndPmem,
}

/// Per-object tracking information.
#[derive(Debug)]
struct MultiObjectInfo {
    tier: Tier,
    size: ObjectSize,
    access_count: u64,
}

/// A completely independent 3-state memory tiering manager.
///
/// Manages object placement across Cold (PMEM), Warm (DRAM pointer, zero-copy), and
/// Hot (DRAM copy) tiers with byte-based LRU eviction.
///
/// This struct is designed to operate alongside — but completely independently of —
/// the existing `TieringManager`. No shared state or feature-flag coupling exists
/// between the two.
pub struct MultitieringManager<K, V> {
    config: Arc<RwLock<MultitieringConfig>>,
    stats: Arc<RwLock<MultitieringStats>>,

    /// Per-object tier info, keyed by hashed key.
    object_info: Arc<RwLock<HashMap<HashedKey, MultiObjectInfo>>>,

    /// LRU order for the Warm tier (front = oldest/LRU, back = most-recently used).
    warm_lru: Arc<RwLock<VecDeque<HashedKey>>>,

    /// LRU order for the Hot tier (front = oldest/LRU, back = most-recently used).
    hot_lru: Arc<RwLock<VecDeque<HashedKey>>>,

    _phantom: std::marker::PhantomData<(K, V)>,
}

impl<K, V> MultitieringManager<K, V> {
    /// Creates a new `MultitieringManager` with the given configuration.
    pub fn new(config: MultitieringConfig) -> Self {
        MultitieringManager {
            config: Arc::new(RwLock::new(config)),
            stats: Arc::new(RwLock::new(MultitieringStats::default())),
            object_info: Arc::new(RwLock::new(HashMap::new())),
            warm_lru: Arc::new(RwLock::new(VecDeque::new())),
            hot_lru: Arc::new(RwLock::new(VecDeque::new())),
            _phantom: std::marker::PhantomData,
        }
    }

    /// Creates a new `MultitieringManager` with default configuration.
    pub fn with_defaults() -> Self {
        Self::new(MultitieringConfig::default())
    }

    /// Registers a new object as `PmemOnly` (Cold).
    pub fn register_object(&self, key: HashedKey, size: ObjectSize) {
        let mut info_map = self.object_info.write().unwrap();
        info_map.insert(
            key,
            MultiObjectInfo {
                tier: Tier::PmemOnly,
                size,
                access_count: 0,
            },
        );
        let mut stats = self.stats.write().unwrap();
        stats.cold_objects += 1;
    }

    /// Records an access to an object and returns the tier transition that should occur,
    /// if any: `Some(Tier::DramPtrToPmem)` to promote Cold→Warm, or
    /// `Some(Tier::DramAndPmem)` to promote Warm→Hot.
    pub fn record_access(&self, key: HashedKey) -> Option<Tier> {
        let config = self.config.read().unwrap();
        let warm_threshold = config.warm_threshold;
        let hot_threshold = config.hot_threshold;
        drop(config);

        let mut info_map = self.object_info.write().unwrap();
        if let Some(info) = info_map.get_mut(&key) {
            info.access_count += 1;
            match info.tier {
                Tier::PmemOnly => {
                    if info.access_count >= warm_threshold {
                        return Some(Tier::DramPtrToPmem);
                    }
                }
                Tier::DramPtrToPmem => {
                    if info.access_count >= hot_threshold {
                        return Some(Tier::DramAndPmem);
                    }
                }
                Tier::DramAndPmem => {
                    // Already hot; update LRU position.
                    let mut hot_lru = self.hot_lru.write().unwrap();
                    lru_touch(&mut hot_lru, key);
                }
            }
        }
        None
    }

    /// Promotes a Cold (`PmemOnly`) object to Warm (`DramPtrToPmem`).
    ///
    /// This is a **zero-copy** operation — no payload bytes are allocated or moved.
    /// Only the metadata pointer is registered in the Warm tier.
    ///
    /// If the warm tier is over capacity after promotion, the oldest warm object is
    /// demoted directly to Cold (`PmemOnly`).
    ///
    /// Returns `true` if the promotion succeeded.
    pub fn promote_to_warm(&self, key: HashedKey) -> bool {
        let mut info_map = self.object_info.write().unwrap();

        let size = match info_map.get(&key) {
            Some(info) if info.tier == Tier::PmemOnly => info.size,
            _ => return false,
        };

        // Transition Cold → Warm.
        info_map.get_mut(&key).unwrap().tier = Tier::DramPtrToPmem;
        drop(info_map);

        {
            let mut warm_lru = self.warm_lru.write().unwrap();
            warm_lru.push_back(key);
        }

        {
            let mut stats = self.stats.write().unwrap();
            stats.cold_objects = stats.cold_objects.saturating_sub(1);
            stats.warm_objects += 1;
            stats.warm_size += size as u64;
            stats.promotions += 1;
        }

        // Enforce warm capacity: evict oldest warm objects to Cold.
        self.enforce_warm_capacity();

        true
    }

    /// Promotes a Warm (`DramPtrToPmem`) object to Hot (`DramAndPmem`).
    ///
    /// This operation **physically copies** the object payload into DRAM.
    ///
    /// If the hot tier is over capacity after promotion, the oldest hot object is
    /// demoted **directly to Cold** (`PmemOnly`) — never to Warm.
    ///
    /// Returns `true` if the promotion succeeded.
    pub fn promote_to_hot(&self, key: HashedKey) -> bool {
        let mut info_map = self.object_info.write().unwrap();

        let size = match info_map.get(&key) {
            Some(info) if info.tier == Tier::DramPtrToPmem => info.size,
            _ => return false,
        };

        // Transition Warm → Hot.
        info_map.get_mut(&key).unwrap().tier = Tier::DramAndPmem;
        drop(info_map);

        // Remove from warm LRU.
        {
            let mut warm_lru = self.warm_lru.write().unwrap();
            lru_remove(&mut warm_lru, key);
        }

        // Add to hot LRU.
        {
            let mut hot_lru = self.hot_lru.write().unwrap();
            hot_lru.push_back(key);
        }

        {
            let mut stats = self.stats.write().unwrap();
            stats.warm_objects = stats.warm_objects.saturating_sub(1);
            stats.warm_size = stats.warm_size.saturating_sub(size as u64);
            stats.hot_objects += 1;
            stats.hot_size += size as u64;
            stats.promotions += 1;
        }

        // Enforce hot capacity: evict oldest hot objects directly to Cold.
        self.enforce_hot_capacity();

        true
    }

    /// Removes an object from all tracking (called on cache deletion).
    pub fn remove_object(&self, key: HashedKey) {
        let mut info_map = self.object_info.write().unwrap();

        if let Some(info) = info_map.remove(&key) {
            let mut stats = self.stats.write().unwrap();
            match info.tier {
                Tier::PmemOnly => {
                    stats.cold_objects = stats.cold_objects.saturating_sub(1);
                }
                Tier::DramPtrToPmem => {
                    drop(stats);
                    drop(info_map);
                    let mut warm_lru = self.warm_lru.write().unwrap();
                    lru_remove(&mut warm_lru, key);
                    let mut stats = self.stats.write().unwrap();
                    stats.warm_objects = stats.warm_objects.saturating_sub(1);
                    stats.warm_size = stats.warm_size.saturating_sub(info.size as u64);
                    return;
                }
                Tier::DramAndPmem => {
                    drop(stats);
                    drop(info_map);
                    let mut hot_lru = self.hot_lru.write().unwrap();
                    lru_remove(&mut hot_lru, key);
                    let mut stats = self.stats.write().unwrap();
                    stats.hot_objects = stats.hot_objects.saturating_sub(1);
                    stats.hot_size = stats.hot_size.saturating_sub(info.size as u64);
                    return;
                }
            }
        }
    }

    /// Clears all tiering state (for cache wipe).
    pub fn clear(&self) {
        self.object_info.write().unwrap().clear();
        self.warm_lru.write().unwrap().clear();
        self.hot_lru.write().unwrap().clear();
        *self.stats.write().unwrap() = MultitieringStats::default();
    }

    /// Returns a snapshot of current tiering statistics.
    pub fn stats(&self) -> MultitieringStats {
        self.stats.read().unwrap().clone()
    }

    /// Returns the current tier for the given key, if tracked.
    pub fn tier_of(&self, key: &HashedKey) -> Option<Tier> {
        self.object_info.read().unwrap().get(key).map(|i| i.tier)
    }

    // ── Internal helpers ────────────────────────────────────────────────────────

    /// Enforces `warm_capacity_bytes` by demoting the oldest Warm objects to Cold
    /// until the warm tier is within capacity.
    fn enforce_warm_capacity(&self) {
        loop {
            let over_capacity = {
                let config = self.config.read().unwrap();
                let stats = self.stats.read().unwrap();
                stats.warm_size > config.warm_capacity_bytes
            };
            if !over_capacity {
                break;
            }

            // Pop the oldest warm object.
            let oldest = {
                let mut warm_lru = self.warm_lru.write().unwrap();
                warm_lru.pop_front()
            };

            let Some(evict_key) = oldest else { break };
            self.demote_to_cold_internal(evict_key, Tier::DramPtrToPmem);
        }
    }

    /// Enforces `hot_capacity_bytes` by demoting the oldest Hot objects **directly to Cold**
    /// until the hot tier is within capacity.
    fn enforce_hot_capacity(&self) {
        loop {
            let over_capacity = {
                let config = self.config.read().unwrap();
                let stats = self.stats.read().unwrap();
                stats.hot_size > config.hot_capacity_bytes
            };
            if !over_capacity {
                break;
            }

            // Pop the oldest hot object.
            let oldest = {
                let mut hot_lru = self.hot_lru.write().unwrap();
                hot_lru.pop_front()
            };

            let Some(evict_key) = oldest else { break };
            self.demote_to_cold_internal(evict_key, Tier::DramAndPmem);
        }
    }

    /// Demotes an object that is currently in `expected_tier` to `PmemOnly`.
    ///
    /// This is the shared demotion path used by both `enforce_warm_capacity` and
    /// `enforce_hot_capacity`. The caller is responsible for already having removed
    /// the key from the respective LRU queue.
    fn demote_to_cold_internal(&self, key: HashedKey, expected_tier: Tier) {
        let mut info_map = self.object_info.write().unwrap();
        if let Some(info) = info_map.get_mut(&key) {
            if info.tier != expected_tier {
                return;
            }
            let size = info.size as u64;
            info.tier = Tier::PmemOnly;
            // Reset access count so the object can be re-promoted via the normal path.
            info.access_count = 0;

            let mut stats = self.stats.write().unwrap();
            match expected_tier {
                Tier::DramPtrToPmem => {
                    stats.warm_objects = stats.warm_objects.saturating_sub(1);
                    stats.warm_size = stats.warm_size.saturating_sub(size);
                }
                Tier::DramAndPmem => {
                    stats.hot_objects = stats.hot_objects.saturating_sub(1);
                    stats.hot_size = stats.hot_size.saturating_sub(size);
                }
                Tier::PmemOnly => {}
            }
            stats.cold_objects += 1;
            stats.demotions += 1;
        }
    }


        /// Gets a TieringObject from the DRAM cache if it exists there
    /// Returns a reference to the TieringObject<K> stored in DRAM
    #[cfg(all(feature = "key_value_pmem", not(feature = "tiering_hashtable_pmem")))]
    pub fn get_from_dram(&self, key: &HashedKey) -> Option<impl std::ops::Deref<Target = TieringObject<K>> + '_> {
        self.dram_cache.get(key)
    }

    #[cfg(any(
        all(feature = "key_value_pmem", feature = "tiering_hashtable_pmem"),
        feature = "alloc_api_exp"
    ))]
    pub fn get_from_dram(&self, key: &HashedKey) -> Option<Arc<TieringObject<K>>> {
        self.dram_cache
            .read()
            .unwrap()
            .get(key)
            .map(|obj| Arc::new(obj.clone()))
    }
    
}

/// Removes the first occurrence of `key` from `deque` (O(n) linear scan).
///
/// For production deployments with very large working sets, this can be replaced
/// with an intrusive doubly-linked list (e.g., using `dlv-list`) combined with a
/// `HashMap<HashedKey, Index>` to achieve O(1) removals.
fn lru_remove(deque: &mut VecDeque<HashedKey>, key: HashedKey) {
    if let Some(pos) = deque.iter().position(|&k| k == key) {
        deque.remove(pos);
    }
}

/// Moves an existing entry to the back of the deque (most recently used).
/// If the key is not present, it is pushed to the back.
///
/// Like `lru_remove`, this is O(n) due to the linear scan. See that function's
/// documentation for guidance on upgrading to an O(1) implementation.
fn lru_touch(deque: &mut VecDeque<HashedKey>, key: HashedKey) {
    lru_remove(deque, key);
    deque.push_back(key);
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_manager(warm_cap: u64, hot_cap: u64) -> MultitieringManager<u64, Vec<u8>> {
        MultitieringManager::new(MultitieringConfig {
            warm_capacity_bytes: warm_cap,
            hot_capacity_bytes: hot_cap,
            warm_threshold: 1,
            hot_threshold: 2,
        })
    }

    #[test]
    fn test_multitier_manager_creation() {
        let mgr = MultitieringManager::<u64, Vec<u8>>::with_defaults();
        let stats = mgr.stats();
        assert_eq!(stats.warm_size, 0);
        assert_eq!(stats.hot_size, 0);
        assert_eq!(stats.cold_objects, 0);
        assert_eq!(stats.promotions, 0);
        assert_eq!(stats.demotions, 0);
    }

    #[test]
    fn test_register_object() {
        let mgr = MultitieringManager::<u64, Vec<u8>>::with_defaults();
        mgr.register_object(1, 100);
        let stats = mgr.stats();
        assert_eq!(stats.cold_objects, 1);
        assert_eq!(stats.warm_objects, 0);
        assert_eq!(stats.hot_objects, 0);
        assert_eq!(mgr.tier_of(&1), Some(Tier::PmemOnly));
    }

    #[test]
    fn test_promote_to_warm() {
        let mgr = MultitieringManager::<u64, Vec<u8>>::with_defaults();
        mgr.register_object(1, 100);
        assert!(mgr.promote_to_warm(1));
        let stats = mgr.stats();
        assert_eq!(stats.warm_objects, 1);
        assert_eq!(stats.warm_size, 100);
        assert_eq!(stats.cold_objects, 0);
        assert_eq!(stats.promotions, 1);
        assert_eq!(mgr.tier_of(&1), Some(Tier::DramPtrToPmem));
    }

    #[test]
    fn test_promote_to_hot() {
        let mgr = MultitieringManager::<u64, Vec<u8>>::with_defaults();
        mgr.register_object(1, 100);
        mgr.promote_to_warm(1);
        assert!(mgr.promote_to_hot(1));
        let stats = mgr.stats();
        assert_eq!(stats.hot_objects, 1);
        assert_eq!(stats.hot_size, 100);
        assert_eq!(stats.warm_objects, 0);
        assert_eq!(stats.warm_size, 0);
        assert_eq!(stats.promotions, 2);
        assert_eq!(mgr.tier_of(&1), Some(Tier::DramAndPmem));
    }

    #[test]
    fn test_remove_object() {
        let mgr = MultitieringManager::<u64, Vec<u8>>::with_defaults();
        mgr.register_object(1, 100);
        mgr.promote_to_warm(1);
        mgr.remove_object(1);
        assert_eq!(mgr.tier_of(&1), None);
        let stats = mgr.stats();
        assert_eq!(stats.warm_objects, 0);
        assert_eq!(stats.warm_size, 0);
    }

    #[test]
    fn test_clear() {
        let mgr = MultitieringManager::<u64, Vec<u8>>::with_defaults();
        for i in 1_u64..=5 {
            mgr.register_object(i, 100);
            mgr.promote_to_warm(i);
        }
        mgr.clear();
        let stats = mgr.stats();
        assert_eq!(stats.warm_size, 0);
        assert_eq!(stats.hot_size, 0);
        assert_eq!(stats.cold_objects, 0);
        assert_eq!(stats.warm_objects, 0);
        assert_eq!(stats.hot_objects, 0);
    }

    /// Validates byte-threshold evictions for both the Warm and Hot tiers.
    ///
    /// Warm tier:
    ///   - warm_capacity_bytes = 250 bytes, each object = 100 bytes.
    ///   - After inserting objects 1, 2, 3 into warm (total 300 B > 250 B), object 1
    ///     (the oldest) should be demoted back to PmemOnly.
    ///
    /// Hot tier:
    ///   - hot_capacity_bytes = 150 bytes, each object = 100 bytes.
    ///   - After promoting objects 2 and 3 to hot (total 200 B > 150 B), object 2
    ///     (the oldest hot) should be demoted **directly to PmemOnly** — not to Warm.
    ///
    /// Throughout the test, `stats.warm_size` and `stats.hot_size` must remain accurate.
    #[test]
    fn test_multitiering_byte_threshold_evictions() {
        // warm_capacity = 250 bytes → can hold two 100-byte objects before evicting.
        // hot_capacity  = 150 bytes → can hold one  100-byte object before evicting.
        let mgr: MultitieringManager<u64, Vec<u8>> = MultitieringManager::new(MultitieringConfig {
            warm_capacity_bytes: 250,
            hot_capacity_bytes: 150,
            warm_threshold: 1,
            hot_threshold: 2,
        });

        // ── Phase 1: fill the Warm tier ────────────────────────────────────────
        // Insert objects 1, 2, 3 (each 100 bytes) into Cold, then promote to Warm.
        for key in [1_u64, 2, 3] {
            mgr.register_object(key, 100);
            assert!(mgr.promote_to_warm(key), "promote_to_warm({key}) should succeed");
        }

        // After promoting 3 objects the warm tier holds 300 B, which exceeds 250 B.
        // The LRU eviction should have demoted object 1 (inserted first) back to Cold.
        assert_eq!(
            mgr.tier_of(&1),
            Some(Tier::PmemOnly),
            "object 1 should be demoted to PmemOnly after warm overflow"
        );
        assert_eq!(
            mgr.tier_of(&2),
            Some(Tier::DramPtrToPmem),
            "object 2 should remain in Warm"
        );
        assert_eq!(
            mgr.tier_of(&3),
            Some(Tier::DramPtrToPmem),
            "object 3 should remain in Warm"
        );

        {
            let stats = mgr.stats();
            assert_eq!(stats.warm_size, 200, "warm_size should be 200 after overflow eviction");
            assert_eq!(stats.warm_objects, 2);
            assert_eq!(stats.cold_objects, 1, "object 1 must be cold");
            assert_eq!(stats.demotions, 1, "exactly one demotion should have occurred");
        }

        // ── Phase 2: fill the Hot tier ─────────────────────────────────────────
        // Promote objects 2 and 3 from Warm to Hot.
        assert!(mgr.promote_to_hot(2), "promote_to_hot(2) should succeed");

        {
            let stats = mgr.stats();
            assert_eq!(stats.hot_size, 100, "hot_size should be 100 after first hot promotion");
            assert_eq!(stats.hot_objects, 1);
            assert_eq!(stats.warm_size, 100, "warm_size should drop to 100");
            assert_eq!(stats.warm_objects, 1);
        }

        assert!(mgr.promote_to_hot(3), "promote_to_hot(3) should succeed");

        // After promoting 2 and 3 to hot (total 200 B > 150 B), object 2 (oldest hot)
        // must be demoted **directly to PmemOnly** — NOT to DramPtrToPmem.
        assert_eq!(
            mgr.tier_of(&2),
            Some(Tier::PmemOnly),
            "object 2 should be demoted directly to PmemOnly after hot overflow"
        );
        assert_eq!(
            mgr.tier_of(&3),
            Some(Tier::DramAndPmem),
            "object 3 should remain in Hot"
        );

        {
            let stats = mgr.stats();
            assert_eq!(stats.hot_size, 100, "hot_size should be 100 after hot overflow eviction");
            assert_eq!(stats.hot_objects, 1);
            assert_eq!(
                stats.warm_size, 0,
                "warm_size must be 0 (object 3 moved from Warm to Hot)"
            );
            assert_eq!(stats.warm_objects, 0);
            // demotions: 1 (warm overflow) + 1 (hot overflow) = 2
            assert_eq!(stats.demotions, 2, "two demotions should have occurred total");
        }

        // ── Phase 3: verify Warm tier stats after multiple promotions/demotions ─
        // Register and promote a fresh object to verify counters are still accurate.
        mgr.register_object(4, 50);
        mgr.promote_to_warm(4);

        {
            let stats = mgr.stats();
            // warm holds only key 4 (50 B < 250 B) → no further eviction
            assert_eq!(stats.warm_size, 50, "warm_size should be 50 after inserting key 4");
            assert_eq!(stats.warm_objects, 1);
            // hot still holds key 3 (100 B < 150 B)
            assert_eq!(stats.hot_size, 100, "hot_size should remain 100");
            assert_eq!(stats.hot_objects, 1);
        }
    }
}


*/


/*
 * Copyright (c) Kia Shakiba
 *
 * This source code is licensed under the GNU AGPLv3 license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::{
    collections::HashMap,
    sync::{Arc, RwLock},
    time::{Instant, Duration},
};

use dashmap::DashMap;
use typesize::TypeSize;
use nohash_hasher::NoHashHasher;
use std::hash::BuildHasherDefault;

use crate::{
    HashedKey,
    object::{Object, ObjectSize},
};

use super::object::TieringObject;

pub type NoHasher = BuildHasherDefault<NoHashHasher<HashedKey>>;

#[cfg(any(feature = "key_value_pmem", feature = "alloc_api_exp", feature = "tiering_hashtable_pmem"))]
use crate::allocator::HybridObjects as Hybrid;

#[cfg(any(feature = "key_value_pmem", feature = "alloc_api_exp"))]
use hashbrown::HashMap as hashtable;

/// Statistics tracked by the MultitieringManager.
#[derive(Debug, Clone, Default)]
pub struct MultitieringStats {
    pub warm_size: u64,
    pub hot_size: u64,
    pub cold_objects: u64,
    pub warm_objects: u64,
    pub hot_objects: u64,
    pub promotions: u64,
    pub demotions: u64,
}

#[derive(Debug, Clone)]
pub struct MultitieringConfig {
    pub warm_capacity_bytes: u64,
    pub hot_capacity_bytes: u64,
    pub warm_threshold: u64,
    pub hot_threshold: u64,
    pub evaluation_interval: Duration,
    /// High water mark for the warm tier (0.0–1.0). Demotion starts when warm usage exceeds
    /// `warm_capacity_bytes * warm_high_water_mark`.
    pub warm_high_water_mark: f64,
    /// Low water mark for the warm tier (0.0–1.0). Demotion stops when warm usage drops below
    /// `warm_capacity_bytes * warm_low_water_mark`.
    pub warm_low_water_mark: f64,
    /// High water mark for the hot tier (0.0–1.0). Demotion starts when hot usage exceeds
    /// `hot_capacity_bytes * hot_high_water_mark`.
    pub hot_high_water_mark: f64,
    /// Low water mark for the hot tier (0.0–1.0). Demotion stops when hot usage drops below
    /// `hot_capacity_bytes * hot_low_water_mark`.
    pub hot_low_water_mark: f64,
}

impl Default for MultitieringConfig {
    fn default() -> Self {
        MultitieringConfig {
            warm_capacity_bytes: 512 * 1024 * 1024,
            hot_capacity_bytes: 1024 * 1024 * 1024,
            warm_threshold: 2,
            hot_threshold: 4,
            evaluation_interval: Duration::from_secs(5),  // matches TieringWorker default
            warm_high_water_mark: 0.95,
            warm_low_water_mark: 0.7,
            hot_high_water_mark: 0.9,
            hot_low_water_mark: 0.7,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Tier {
    PmemOnly,
    DramPtrToPmem,
    DramAndPmem,
}

struct MultiObjectInfo {
    tier: Tier,
    size: ObjectSize,
    access_count: u64,
    last_access: Instant,
}

pub struct MultitieringManager<K, V> {
    config: Arc<RwLock<MultitieringConfig>>,
    stats: Arc<RwLock<MultitieringStats>>,
    object_info: Arc<RwLock<HashMap<HashedKey, MultiObjectInfo>>>,

    /// DRAM cache for hot objects (DashMap when no PMEM hashtable allocator is in use).
    #[cfg(not(any(feature = "tiering_hashtable_pmem", feature = "alloc_api_exp")))]
    dram_cache: Arc<DashMap<HashedKey, TieringObject<K>, NoHasher>>,

    /// DRAM cache backed by a PMEM-aware hashmap when a PMEM hashtable allocator is enabled.
    #[cfg(any(feature = "tiering_hashtable_pmem", feature = "alloc_api_exp"))]
    dram_cache: Arc<RwLock<hashtable<HashedKey, TieringObject<K>, NoHasher, Hybrid>>>,

    _phantom: std::marker::PhantomData<(K, V)>,
}

/// General Implementation (No AsRef<[u8]> bound required)
impl<K, V> MultitieringManager<K, V>
where
    K: TypeSize + Clone,
{
    pub fn new(config: MultitieringConfig) -> Self {
        #[cfg(not(any(feature = "tiering_hashtable_pmem", feature = "alloc_api_exp")))]
        let dram_cache = Arc::new(DashMap::with_hasher(NoHasher::default()));
        #[cfg(any(feature = "tiering_hashtable_pmem", feature = "alloc_api_exp"))]
        let dram_cache = Arc::new(RwLock::new(hashtable::with_hasher_in(NoHasher::default(), Hybrid)));

        MultitieringManager {
            config: Arc::new(RwLock::new(config)),
            stats: Arc::new(RwLock::new(MultitieringStats::default())),
            object_info: Arc::new(RwLock::new(HashMap::new())),
            dram_cache,
            _phantom: std::marker::PhantomData,
        }
    }

    /// Creates a new `MultitieringManager` with default configuration.
    pub fn with_defaults() -> Self {
        Self::new(MultitieringConfig::default())
    }

    pub fn register_object(&self, key: HashedKey, size: ObjectSize) {
        let mut info_map = self.object_info.write().unwrap();
        info_map.insert(key, MultiObjectInfo {
            tier: Tier::PmemOnly,
            size,
            access_count: 0,
            last_access: Instant::now(),
        });
        self.stats.write().unwrap().cold_objects += 1;
    }

    /// Records an access to an object, updating its access count and last-access timestamp.
    pub fn record_access(&self, key: HashedKey) {
        let mut info_map = self.object_info.write().unwrap();
        if let Some(info) = info_map.get_mut(&key) {
            info.access_count += 1;
            info.last_access = Instant::now();
        }
    }

    pub fn stats(&self) -> MultitieringStats {
        self.stats.read().unwrap().clone()
    }

    /// Returns the current tier for the given key, if tracked.
    pub fn tier_of(&self, key: &HashedKey) -> Option<Tier> {
        self.object_info.read().unwrap().get(key).map(|i| i.tier)
    }

    /// Returns keys that should be promoted and their target tier, based on access counts.
    pub fn get_keys_to_promote(&self) -> Vec<(HashedKey, Tier)> {
        let info_map = self.object_info.read().unwrap();
        let config = self.config.read().unwrap();
        let mut to_promote = Vec::new();

        for (key, info) in info_map.iter() {
            match info.tier {
                Tier::PmemOnly if info.access_count >= config.warm_threshold => {
                    to_promote.push((*key, Tier::DramPtrToPmem));
                }
                Tier::DramPtrToPmem if info.access_count >= config.hot_threshold => {
                    to_promote.push((*key, Tier::DramAndPmem));
                }
                _ => {}
            }
        }
        to_promote
    }

    /// Returns keys that should be demoted based on high/low water marks.
    ///
    /// When a tier's used bytes exceed `capacity * high_water_mark`, the oldest (LRU) objects
    /// are collected for demotion until usage would fall below `capacity * low_water_mark`.
    pub fn get_keys_to_demote(&self) -> Vec<(HashedKey, Tier)> {
        let info_map = self.object_info.read().unwrap();
        let stats = self.stats.read().unwrap();
        let config = self.config.read().unwrap();
        let mut keys_to_demote = Vec::new();

        let hot_high_water = (config.hot_capacity_bytes as f64 * config.hot_high_water_mark) as u64;
        let hot_low_water = (config.hot_capacity_bytes as f64 * config.hot_low_water_mark) as u64;
        if stats.hot_size > hot_high_water {
            let mut hot_objects: Vec<_> = info_map.iter()
                .filter(|(_, info)| info.tier == Tier::DramAndPmem)
                .collect();
            hot_objects.sort_by_key(|(_, info)| info.last_access);

            let mut current_hot_size = stats.hot_size;
            for (key, info) in hot_objects {
                if current_hot_size <= hot_low_water { break; }
                keys_to_demote.push((*key, Tier::DramAndPmem));
                current_hot_size = current_hot_size.saturating_sub(info.size as u64);
            }
        }

        let warm_high_water = (config.warm_capacity_bytes as f64 * config.warm_high_water_mark) as u64;
        let warm_low_water = (config.warm_capacity_bytes as f64 * config.warm_low_water_mark) as u64;
        if stats.warm_size > warm_high_water {
            let mut warm_objects: Vec<_> = info_map.iter()
                .filter(|(_, info)| info.tier == Tier::DramPtrToPmem)
                .collect();
            warm_objects.sort_by_key(|(_, info)| info.last_access);

            let mut current_warm_size = stats.warm_size;
            for (key, info) in warm_objects {
                if current_warm_size <= warm_low_water { break; }
                keys_to_demote.push((*key, Tier::DramPtrToPmem));
                current_warm_size = current_warm_size.saturating_sub(info.size as u64);
            }
        }

        keys_to_demote
    }

    /// Promotes a Cold (`PmemOnly`) object to Warm (`DramPtrToPmem`).
    /// Returns `true` if the promotion succeeded.
    pub fn promote_to_warm(&self, key: HashedKey) -> bool {
        let mut info_map = self.object_info.write().unwrap();
        if let Some(info) = info_map.get_mut(&key) {
            if info.tier == Tier::PmemOnly {
                info.tier = Tier::DramPtrToPmem;
                info.access_count = 0;
                let mut stats = self.stats.write().unwrap();
                stats.warm_size += info.size as u64;
                stats.warm_objects += 1;
                stats.cold_objects = stats.cold_objects.saturating_sub(1);
                stats.promotions += 1;
                return true;
            }
        }
        false
    }

    /// Demotes an object from any DRAM tier directly to Cold (`PmemOnly`).
    pub fn demote_to_cold(&self, key: HashedKey) {
        let mut info_map = self.object_info.write().unwrap();
        if let Some(info) = info_map.get_mut(&key) {
            let old_tier = info.tier;
            if old_tier == Tier::DramAndPmem {
                #[cfg(not(any(feature = "tiering_hashtable_pmem", feature = "alloc_api_exp")))]
                self.dram_cache.remove(&key);
                #[cfg(any(feature = "tiering_hashtable_pmem", feature = "alloc_api_exp"))]
                self.dram_cache.write().unwrap().remove(&key);
            }

            info.tier = Tier::PmemOnly;
            info.access_count = 0;

            let mut stats = self.stats.write().unwrap();
            match old_tier {
                Tier::DramAndPmem => {
                    stats.hot_size = stats.hot_size.saturating_sub(info.size as u64);
                    stats.hot_objects = stats.hot_objects.saturating_sub(1);
                }
                Tier::DramPtrToPmem => {
                    stats.warm_size = stats.warm_size.saturating_sub(info.size as u64);
                    stats.warm_objects = stats.warm_objects.saturating_sub(1);
                }
                _ => return,
            }
            stats.cold_objects += 1;
            stats.demotions += 1;
        }
    }

    /// Removes an object from all tracking (called on cache deletion).
    pub fn remove_object(&self, key: HashedKey) {
        let info = {
            let mut info_map = self.object_info.write().unwrap();
            info_map.remove(&key)
        };
        if let Some(info) = info {
            if info.tier == Tier::DramAndPmem {
                #[cfg(not(any(feature = "tiering_hashtable_pmem", feature = "alloc_api_exp")))]
                self.dram_cache.remove(&key);
                #[cfg(any(feature = "tiering_hashtable_pmem", feature = "alloc_api_exp"))]
                self.dram_cache.write().unwrap().remove(&key);
            }
            let mut stats = self.stats.write().unwrap();
            match info.tier {
                Tier::PmemOnly => {
                    stats.cold_objects = stats.cold_objects.saturating_sub(1);
                }
                Tier::DramPtrToPmem => {
                    stats.warm_objects = stats.warm_objects.saturating_sub(1);
                    stats.warm_size = stats.warm_size.saturating_sub(info.size as u64);
                }
                Tier::DramAndPmem => {
                    stats.hot_objects = stats.hot_objects.saturating_sub(1);
                    stats.hot_size = stats.hot_size.saturating_sub(info.size as u64);
                }
            }
        }
    }

    /// Gets a hot object from the DRAM cache if it exists there.
    #[cfg(not(any(feature = "tiering_hashtable_pmem", feature = "alloc_api_exp")))]
    pub fn get_from_dram(&self, key: &HashedKey) -> Option<impl std::ops::Deref<Target = TieringObject<K>> + '_> {
        self.dram_cache.get(key)
    }

    /// Gets a hot object from the DRAM cache if it exists there.
    #[cfg(any(feature = "tiering_hashtable_pmem", feature = "alloc_api_exp"))]
    pub fn get_from_dram(&self, key: &HashedKey) -> Option<Arc<TieringObject<K>>> {
        self.dram_cache.read().unwrap().get(key).map(|obj| Arc::new(obj.clone()))
    }

    /// Clears all tiering state (for cache wipe).
    pub fn clear(&self) {
        self.object_info.write().unwrap().clear();
        #[cfg(not(any(feature = "tiering_hashtable_pmem", feature = "alloc_api_exp")))]
        self.dram_cache.clear();
        #[cfg(any(feature = "tiering_hashtable_pmem", feature = "alloc_api_exp"))]
        self.dram_cache.write().unwrap().clear();
        *self.stats.write().unwrap() = MultitieringStats::default();
    }

    /// Returns the configured periodic evaluation interval.
    pub fn evaluation_interval(&self) -> Duration {
        self.config.read().unwrap().evaluation_interval
    }

    /// Promotes a Hot object (metadata only, no data copy) — for testing tier logic
    /// without needing a real `Object<K, V>` value.
    ///
    /// **Warning**: This method does **not** place any data into the DRAM cache. Calling
    /// `get_from_dram()` for a key promoted via this method will return `None`. It is
    /// intended solely for unit and integration tests that exercise tier-tracking and
    /// eviction logic. Production code should use `promote_to_hot(&key, &object)` instead.
    pub fn promote_to_hot_no_data(&self, key: HashedKey) -> bool {
        let mut info_map = self.object_info.write().unwrap();
        if let Some(info) = info_map.get_mut(&key) {
            if info.tier != Tier::DramAndPmem {
                let old_tier = info.tier;
                info.tier = Tier::DramAndPmem;
                info.access_count = 0;
                let mut stats = self.stats.write().unwrap();
                stats.hot_size += info.size as u64;
                stats.hot_objects += 1;
                if old_tier == Tier::DramPtrToPmem {
                    stats.warm_size = stats.warm_size.saturating_sub(info.size as u64);
                    stats.warm_objects = stats.warm_objects.saturating_sub(1);
                } else {
                    stats.cold_objects = stats.cold_objects.saturating_sub(1);
                }
                stats.promotions += 1;
                return true;
            }
        }
        false
    }
}

/// Specialized implementation that requires `V: AsRef<[u8]>` in order to physically
/// copy object payload bytes into the DRAM cache.
impl<K, V> MultitieringManager<K, V>
where
    K: TypeSize + Clone,
    V: TypeSize + Clone + AsRef<[u8]>,
{
    /// Promotes an object to Hot (`DramAndPmem`), physically copying its payload into DRAM.
    ///
    /// Returns `true` if the promotion succeeded.
    pub fn promote_to_hot(&self, key: HashedKey, object: &Object<K, V>) -> bool {
        let mut info_map = self.object_info.write().unwrap();
        if let Some(info) = info_map.get_mut(&key) {
            if info.tier != Tier::DramAndPmem {
                let data_arc = object.data();
                let bytes: &[u8] = data_arc.as_ref().as_ref();
                let dram_obj = TieringObject::with_expiry(
                    object.key().clone(),
                    bytes.to_vec().into_boxed_slice(),
                    object.expiry(),
                );

                #[cfg(not(any(feature = "tiering_hashtable_pmem", feature = "alloc_api_exp")))]
                self.dram_cache.insert(key, dram_obj);
                #[cfg(any(feature = "tiering_hashtable_pmem", feature = "alloc_api_exp"))]
                self.dram_cache.write().unwrap().insert(key, dram_obj);

                let old_tier = info.tier;
                info.tier = Tier::DramAndPmem;
                info.access_count = 0;

                let mut stats = self.stats.write().unwrap();
                stats.hot_size += info.size as u64;
                stats.hot_objects += 1;
                if old_tier == Tier::DramPtrToPmem {
                    stats.warm_size = stats.warm_size.saturating_sub(info.size as u64);
                    stats.warm_objects = stats.warm_objects.saturating_sub(1);
                } else {
                    stats.cold_objects = stats.cold_objects.saturating_sub(1);
                }
                stats.promotions += 1;
                return true;
            }
        }
        false
    }
}