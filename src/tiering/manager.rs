/*
 * Copyright (c) Kia Shakiba
 *
 * This source code is licensed under the GNU AGPLv3 license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::{
    collections::{HashMap, HashSet},
    sync::{Arc, RwLock},
};

use crate::{
    HashedKey,
    object::ObjectSize,
};

/// Configuration for the tiering manager
#[derive(Debug, Clone)]
pub struct TieringConfig {
    /// Threshold for DRAM usage in bytes
    /// When DRAM usage exceeds this threshold, cold objects are demoted from DRAM to PMEM
    pub dram_threshold: u64,
    
    /// High water mark as percentage of threshold (0.0 to 1.0)
    /// When DRAM usage exceeds threshold * high_water_mark, start demoting
    pub high_water_mark: f64,
    
    /// Low water mark as percentage of threshold (0.0 to 1.0)
    /// Continue demoting until DRAM usage falls below threshold * low_water_mark
    pub low_water_mark: f64,
}

impl Default for TieringConfig {
    fn default() -> Self {
        TieringConfig {
            dram_threshold: 1_073_741_824, // 1 GB default
            high_water_mark: 0.9,
            low_water_mark: 0.7,
        }
    }
}

/// Statistics for the tiering manager
#[derive(Debug, Clone, Default)]
pub struct TieringStats {
    /// Number of objects currently in DRAM
    pub dram_objects: u64,
    
    /// Total size of objects in DRAM (bytes)
    pub dram_size: u64,
    
    /// Number of promotions from PMEM to DRAM
    pub promotions: u64,
    
    /// Number of demotions from DRAM to PMEM
    pub demotions: u64,
    
    /// Number of objects only in PMEM
    pub pmem_only_objects: u64,
}

/// Tier location for an object
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Tier {
    /// Object exists in both DRAM (copy) and PMEM (source of truth)
    DramAndPmem,
    
    /// Object exists only in PMEM
    PmemOnly,
}

/// Information about an object's tiering status
#[derive(Debug, Clone)]
struct ObjectTierInfo {
    /// Current tier location
    tier: Tier,
    
    /// Size of the object in bytes
    size: ObjectSize,
    
    /// Access count (for LFU-like promotion/demotion)
    access_count: u64,
    
    /// Last access timestamp (for LRU-like promotion/demotion)
    last_access: std::time::Instant,
}

/// Tiering Manager
/// 
/// Manages object placement between DRAM and PMEM tiers.
/// DRAM contains hot copies of objects while PMEM is the source of truth.
/// 
/// ## Promotion Policy
/// Objects are promoted to DRAM after being accessed more than once (hardcoded threshold).
/// This simple heuristic ensures that only objects with repeated access are promoted.
/// Future versions may make this configurable via TieringConfig.
pub struct TieringManager {
    config: Arc<RwLock<TieringConfig>>,
    stats: Arc<RwLock<TieringStats>>,
    
    /// Tracking information for each object
    object_info: Arc<RwLock<HashMap<HashedKey, ObjectTierInfo>>>,
    
    /// Set of objects currently in DRAM (for fast lookup)
    dram_objects: Arc<RwLock<HashSet<HashedKey>>>,
}

impl TieringManager {
    /// Creates a new TieringManager with the given configuration
    pub fn new(config: TieringConfig) -> Self {
        TieringManager {
            config: Arc::new(RwLock::new(config)),
            stats: Arc::new(RwLock::new(TieringStats::default())),
            object_info: Arc::new(RwLock::new(HashMap::new())),
            dram_objects: Arc::new(RwLock::new(HashSet::new())),
        }
    }
    
    /// Creates a new TieringManager with default configuration
    pub fn with_defaults() -> Self {
        Self::new(TieringConfig::default())
    }
    
    /// Registers a new object in PMEM
    pub fn register_object(&self, key: HashedKey, size: ObjectSize) {
        let mut info_map = self.object_info.write().unwrap();
        
        info_map.insert(key, ObjectTierInfo {
            tier: Tier::PmemOnly,
            size,
            access_count: 0,
            last_access: std::time::Instant::now(),
        });
        
        let mut stats = self.stats.write().unwrap();
        stats.pmem_only_objects += 1;
    }
    
    /// Records an access to an object
    /// Returns true if the object should be promoted to DRAM
    /// 
    /// ## Promotion Heuristic
    /// Objects are promoted after 2 accesses (hardcoded threshold).
    /// First access: counted but not promoted
    /// Second+ access: suggests promotion
    pub fn record_access(&self, key: HashedKey) -> bool {
        let mut info_map = self.object_info.write().unwrap();
        
        if let Some(info) = info_map.get_mut(&key) {
            info.access_count += 1;
            info.last_access = std::time::Instant::now();
            
            // Decide if promotion is needed
            match info.tier {
                Tier::PmemOnly => {
                    // Check if we should promote based on access count
                    // Simple heuristic: promote if accessed more than once
                    info.access_count > 1
                }
                Tier::DramAndPmem => {
                    // Already in DRAM
                    false
                }
            }
        } else {
            false
        }
    }
    
    /// Promotes an object to DRAM (creates a copy in DRAM)
    /// Returns true if promotion was successful
    pub fn promote_to_dram(&self, key: HashedKey) -> bool {
        let mut info_map = self.object_info.write().unwrap();
        
        if let Some(info) = info_map.get_mut(&key) {
            if info.tier == Tier::PmemOnly {
                let config = self.config.read().unwrap();
                let mut stats = self.stats.write().unwrap();
                let new_dram_size = stats.dram_size + info.size as u64;
                
                // Check if promotion would exceed threshold
                if new_dram_size <= config.dram_threshold {
                    info.tier = Tier::DramAndPmem;
                    
                    let mut dram_objects = self.dram_objects.write().unwrap();
                    dram_objects.insert(key);
                    
                    stats.dram_size = new_dram_size;
                    stats.dram_objects += 1;
                    stats.pmem_only_objects -= 1;
                    stats.promotions += 1;
                    
                    return true;
                }
            }
        }
        
        false
    }
    
    /// Demotes an object from DRAM (removes the DRAM copy, keeps PMEM)
    /// Returns true if demotion was successful
    pub fn demote_from_dram(&self, key: HashedKey) -> bool {
        let mut info_map = self.object_info.write().unwrap();
        
        if let Some(info) = info_map.get_mut(&key) {
            if info.tier == Tier::DramAndPmem {
                info.tier = Tier::PmemOnly;
                
                let mut dram_objects = self.dram_objects.write().unwrap();
                dram_objects.remove(&key);
                
                let mut stats = self.stats.write().unwrap();
                stats.dram_size = stats.dram_size.saturating_sub(info.size as u64);
                stats.dram_objects = stats.dram_objects.saturating_sub(1);
                stats.pmem_only_objects += 1;
                stats.demotions += 1;
                
                return true;
            }
        }
        
        false
    }
    
    /// Checks if DRAM threshold has been exceeded and returns keys to demote
    /// Uses LRU (Least Recently Used) strategy for demotion
    pub fn get_keys_to_demote(&self) -> Vec<HashedKey> {
        let config = self.config.read().unwrap();
        let stats = self.stats.read().unwrap();
        let high_water = (config.dram_threshold as f64 * config.high_water_mark) as u64;
        let low_water = (config.dram_threshold as f64 * config.low_water_mark) as u64;
        
        if stats.dram_size <= high_water {
            return Vec::new();
        }
        
        drop(stats);
        drop(config);
        
        // Find objects to demote to bring usage below low water mark
        let info_map = self.object_info.read().unwrap();
        let dram_objects = self.dram_objects.read().unwrap();
        
        let mut dram_object_info: Vec<(HashedKey, &ObjectTierInfo)> = dram_objects
            .iter()
            .filter_map(|key| info_map.get(key).map(|info| (*key, info)))
            .collect();
        
        // Sort by last access time (LRU)
        dram_object_info.sort_by_key(|(_, info)| info.last_access);
        
        let config = self.config.read().unwrap();
        let stats = self.stats.read().unwrap();
        let mut current_size = stats.dram_size;
        let mut keys_to_demote = Vec::new();
        let low_water = (config.dram_threshold as f64 * config.low_water_mark) as u64;
        
        for (key, info) in dram_object_info {
            if current_size <= low_water {
                break;
            }
            
            keys_to_demote.push(key);
            current_size = current_size.saturating_sub(info.size as u64);
        }
        
        keys_to_demote
    }
    
    /// Removes an object from tracking (when it's deleted from cache)
    pub fn remove_object(&self, key: HashedKey) {
        let mut info_map = self.object_info.write().unwrap();
        
        if let Some(info) = info_map.remove(&key) {
            let mut stats = self.stats.write().unwrap();
            
            match info.tier {
                Tier::DramAndPmem => {
                    let mut dram_objects = self.dram_objects.write().unwrap();
                    dram_objects.remove(&key);
                    
                    stats.dram_size = stats.dram_size.saturating_sub(info.size as u64);
                    stats.dram_objects = stats.dram_objects.saturating_sub(1);
                }
                Tier::PmemOnly => {
                    stats.pmem_only_objects = stats.pmem_only_objects.saturating_sub(1);
                }
            }
        }
    }
    
    /// Gets current tiering statistics
    pub fn stats(&self) -> TieringStats {
        self.stats.read().unwrap().clone()
    }
    
    /// Checks if an object is currently in DRAM
    pub fn is_in_dram(&self, key: &HashedKey) -> bool {
        self.dram_objects.read().unwrap().contains(key)
    }
    
    /// Updates the DRAM threshold
    pub fn set_dram_threshold(&self, threshold: u64) {
        let mut config = self.config.write().unwrap();
        config.dram_threshold = threshold;
    }
    
    /// Gets the current DRAM threshold
    pub fn dram_threshold(&self) -> u64 {
        self.config.read().unwrap().dram_threshold
    }
    
    /// Clears all tiering information (for cache wipe)
    pub fn clear(&self) {
        let mut info_map = self.object_info.write().unwrap();
        info_map.clear();
        
        let mut dram_objects = self.dram_objects.write().unwrap();
        dram_objects.clear();
        
        let mut stats = self.stats.write().unwrap();
        *stats = TieringStats::default();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_tiering_manager_creation() {
        let manager = TieringManager::with_defaults();
        let stats = manager.stats();
        
        assert_eq!(stats.dram_objects, 0);
        assert_eq!(stats.dram_size, 0);
        assert_eq!(stats.promotions, 0);
        assert_eq!(stats.demotions, 0);
    }
    
    #[test]
    fn test_register_object() {
        let manager = TieringManager::with_defaults();
        
        manager.register_object(1, 100);
        
        let stats = manager.stats();
        assert_eq!(stats.pmem_only_objects, 1);
        assert_eq!(stats.dram_objects, 0);
    }
    
    #[test]
    fn test_promote_to_dram() {
        let manager = TieringManager::with_defaults();
        
        manager.register_object(1, 100);
        assert!(manager.promote_to_dram(1));
        
        let stats = manager.stats();
        assert_eq!(stats.dram_objects, 1);
        assert_eq!(stats.dram_size, 100);
        assert_eq!(stats.promotions, 1);
        assert!(manager.is_in_dram(&1));
    }
    
    #[test]
    fn test_demote_from_dram() {
        let manager = TieringManager::with_defaults();
        
        manager.register_object(1, 100);
        manager.promote_to_dram(1);
        assert!(manager.demote_from_dram(1));
        
        let stats = manager.stats();
        assert_eq!(stats.dram_objects, 0);
        assert_eq!(stats.dram_size, 0);
        assert_eq!(stats.demotions, 1);
        assert!(!manager.is_in_dram(&1));
    }
    
    #[test]
    fn test_threshold_enforcement() {
        let config = TieringConfig {
            dram_threshold: 200,
            high_water_mark: 0.9,
            low_water_mark: 0.7,
        };
        let manager = TieringManager::new(config);
        
        // Promote two objects, total 200 bytes (at threshold)
        manager.register_object(1, 100);
        manager.register_object(2, 100);
        
        assert!(manager.promote_to_dram(1));
        assert!(manager.promote_to_dram(2));
        
        // Try to promote a third object (would exceed threshold)
        manager.register_object(3, 100);
        assert!(!manager.promote_to_dram(3));
        
        let stats = manager.stats();
        assert_eq!(stats.dram_objects, 2);
    }
    
    #[test]
    fn test_remove_object() {
        let manager = TieringManager::with_defaults();
        
        manager.register_object(1, 100);
        manager.promote_to_dram(1);
        manager.remove_object(1);
        
        let stats = manager.stats();
        assert_eq!(stats.dram_objects, 0);
        assert_eq!(stats.dram_size, 0);
    }
    
    #[test]
    fn test_access_recording() {
        let manager = TieringManager::with_defaults();
        
        manager.register_object(1, 100);
        
        // First access - should not promote yet
        assert!(!manager.record_access(1));
        
        // Second access - should suggest promotion
        assert!(manager.record_access(1));
    }
}
