/*
 * Copyright (c) Kia Shakiba
 *
 * This source code is licensed under the GNU AGPLv3 license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::collections::HashSet;
use parking_lot::RwLock;

use crate::{
    HashedKey,
    CacheSize,
};

/// TieringManager tracks which objects should be in DRAM vs PMEM-only
/// using information from the eviction stacks.
/// 
/// The DRAM tier acts as a cache of hot objects that also exist in PMEM.
/// All data is stored in PMEM, but frequently accessed data is also kept in DRAM.
pub struct TieringManager {
    /// Set of objects currently in DRAM tier
    dram_objects: RwLock<HashSet<HashedKey>>,
    /// Maximum number of objects that can be in DRAM
    dram_capacity: RwLock<usize>,
}

impl TieringManager {
    /// Create a new TieringManager with the given DRAM capacity
    pub fn new(dram_capacity: usize) -> Self {
        TieringManager {
            dram_objects: RwLock::new(HashSet::new()),
            dram_capacity: RwLock::new(dram_capacity),
        }
    }

    /// Check if an object is in the DRAM tier
    pub fn is_in_dram(&self, key: HashedKey) -> bool {
        self.dram_objects.read().contains(&key)
    }

    /// Promote an object to DRAM tier
    /// Returns true if the object was promoted, false if already in DRAM or DRAM is full
    pub fn promote_to_dram(&self, key: HashedKey) -> bool {
        let mut dram = self.dram_objects.write();
        let capacity = *self.dram_capacity.read();
        
        if dram.contains(&key) {
            return false; // Already in DRAM
        }
        
        if dram.len() >= capacity {
            return false; // DRAM is full
        }
        
        dram.insert(key);
        true
    }

    /// Demote an object from DRAM tier (keep in PMEM only)
    /// Returns true if the object was demoted, false if not in DRAM
    pub fn demote_from_dram(&self, key: HashedKey) -> bool {
        self.dram_objects.write().remove(&key)
    }

    /// Evict the coldest object from DRAM to make room for a hot object
    /// Returns the key that was evicted, or None if DRAM is empty
    pub fn evict_from_dram(&self, eviction_candidate: HashedKey) -> Option<HashedKey> {
        let mut dram = self.dram_objects.write();
        if dram.remove(&eviction_candidate) {
            Some(eviction_candidate)
        } else {
            None
        }
    }

    /// Get the number of objects currently in DRAM
    pub fn dram_size(&self) -> usize {
        self.dram_objects.read().len()
    }

    /// Get the DRAM capacity
    pub fn dram_capacity(&self) -> usize {
        *self.dram_capacity.read()
    }

    /// Update DRAM capacity based on cache size
    pub fn update_capacity(&self, cache_size: CacheSize, ratio: f64) {
        let new_capacity = ((cache_size as f64) * ratio) as usize;
        *self.dram_capacity.write() = new_capacity.max(1);
    }

    /// Clear all DRAM tier tracking
    pub fn clear(&self) {
        self.dram_objects.write().clear();
    }

    /// Remove a key from tracking (when object is deleted from cache)
    pub fn remove(&self, key: HashedKey) {
        self.dram_objects.write().remove(&key);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_promotion_and_demotion() {
        let manager = TieringManager::new(2);
        
        assert!(manager.promote_to_dram(1));
        assert!(manager.is_in_dram(1));
        
        assert!(manager.promote_to_dram(2));
        assert!(manager.is_in_dram(2));
        
        // DRAM is full
        assert!(!manager.promote_to_dram(3));
        
        // Demote one object
        assert!(manager.demote_from_dram(1));
        assert!(!manager.is_in_dram(1));
        
        // Now we can promote
        assert!(manager.promote_to_dram(3));
        assert!(manager.is_in_dram(3));
    }

    #[test]
    fn test_capacity_update() {
        let manager = TieringManager::new(2);
        
        manager.promote_to_dram(1);
        manager.promote_to_dram(2);
        
        assert_eq!(manager.dram_size(), 2);
        
        manager.update_capacity(1000, 0.5);
        assert_eq!(manager.dram_capacity(), 500);
    }
}
