/*
 * Copyright (c) Kia Shakiba
 *
 * This source code is licensed under the GNU AGPLv3 license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use parking_lot::RwLock;

use crate::HashedKey;

const PROMOTION_THRESHOLD: u32 = 2;

/// TieringManager tracks access patterns and manages promotion/demotion decisions
pub struct TieringManager {
    /// Current DRAM size in bytes
    dram_size: AtomicU64,
    
    /// High water mark for DRAM (trigger demotion)
    high_water_mark: u64,
    
    /// Low water mark for DRAM (stop demotion)
    low_water_mark: u64,
    
    /// Access counts for objects
    access_counts: RwLock<HashMap<HashedKey, u32>>,
    
    /// Objects currently in DRAM
    dram_objects: RwLock<HashMap<HashedKey, u64>>, // key -> size
    
    /// Objects pending promotion
    pending_promotion: RwLock<HashMap<HashedKey, ()>>,
}

impl TieringManager {
    pub fn new(high_water_mark: u64, low_water_mark: u64) -> Self {
        TieringManager {
            dram_size: AtomicU64::new(0),
            high_water_mark,
            low_water_mark,
            access_counts: RwLock::new(HashMap::new()),
            dram_objects: RwLock::new(HashMap::new()),
            pending_promotion: RwLock::new(HashMap::new()),
        }
    }
    
    /// Record an access to an object
    pub fn record_access(&self, key: HashedKey) -> u32 {
        let mut counts = self.access_counts.write();
        let count = counts.entry(key).or_insert(0);
        *count += 1;
        *count
    }
    
    /// Check if an object should be promoted
    pub fn should_promote(&self, key: HashedKey, access_count: u32) -> bool {
        if access_count < PROMOTION_THRESHOLD {
            return false;
        }
        
        let dram_objects = self.dram_objects.read();
        if dram_objects.contains_key(&key) {
            return false; // Already in DRAM
        }
        
        let pending = self.pending_promotion.read();
        if pending.contains_key(&key) {
            return false; // Already pending promotion
        }
        
        true
    }
    
    /// Mark an object as pending promotion
    pub fn mark_pending_promotion(&self, key: HashedKey) {
        let mut pending = self.pending_promotion.write();
        pending.insert(key, ());
    }
    
    /// Promote an object to DRAM
    pub fn promote_to_dram(&self, key: HashedKey, size: u64) {
        let mut dram_objects = self.dram_objects.write();
        dram_objects.insert(key, size);
        self.dram_size.fetch_add(size, Ordering::SeqCst);
        
        // Remove from pending promotion
        let mut pending = self.pending_promotion.write();
        pending.remove(&key);
    }
    
    /// Demote an object from DRAM
    pub fn demote_from_dram(&self, key: HashedKey) {
        let mut dram_objects = self.dram_objects.write();
        if let Some(size) = dram_objects.remove(&key) {
            self.dram_size.fetch_sub(size, Ordering::SeqCst);
        }
    }
    
    /// Get current DRAM size
    pub fn dram_size(&self) -> u64 {
        self.dram_size.load(Ordering::SeqCst)
    }
    
    /// Check if demotion is needed
    pub fn needs_demotion(&self) -> bool {
        self.dram_size() > self.high_water_mark
    }
    
    /// Get keys to demote (coldest objects)
    pub fn get_keys_to_demote(&self) -> Vec<HashedKey> {
        let current_size = self.dram_size();
        if current_size <= self.high_water_mark {
            return Vec::new();
        }
        
        let target_size = self.low_water_mark;
        let bytes_to_free = current_size - target_size;
        
        let dram_objects = self.dram_objects.read();
        let access_counts = self.access_counts.read();
        
        // Sort objects by access count (ascending - coldest first)
        let mut objects: Vec<_> = dram_objects.iter()
            .map(|(k, s)| {
                let count = access_counts.get(k).copied().unwrap_or(0);
                (*k, *s, count)
            })
            .collect();
        
        objects.sort_by_key(|(_, _, count)| *count);
        
        // Select objects to demote
        let mut keys_to_demote = Vec::new();
        let mut freed_bytes = 0u64;
        
        for (key, size, _) in objects {
            keys_to_demote.push(key);
            freed_bytes += size;
            
            if freed_bytes >= bytes_to_free {
                break;
            }
        }
        
        keys_to_demote
    }
    
    /// Demote until low water mark is reached
    pub fn demote_until_low_water(&self) -> Vec<HashedKey> {
        let keys = self.get_keys_to_demote();
        for key in &keys {
            self.demote_from_dram(*key);
        }
        keys
    }
}
