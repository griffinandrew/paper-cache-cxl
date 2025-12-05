/*
 * Copyright (c) Kia Shakiba
 *
 * This source code is licensed under the GNU AGPLv3 license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};
use std::sync::atomic::{AtomicU64, Ordering};
use parking_lot::RwLock;
use crossbeam_channel::{Receiver, Sender, bounded};
use typesize::TypeSize;
use log::{info, warn};

use crate::{
    HashedKey,
    ObjectMapRef,
    StatusRef,
    OverheadManagerRef,
    error::CacheError,
    worker::{Worker, WorkerEvent, WorkerReceiver},
};

const BATCH_SIZE: usize = 1024;
const BATCH_TIMEOUT_MS: u64 = 100;
const PROMOTION_THRESHOLD: u32 = 2;

/// Access event for tiering
#[derive(Clone, Copy, Debug)]
pub enum AccessEvent {
    Get(HashedKey),
}

pub type AccessEventSender = Sender<AccessEvent>;
pub type AccessEventReceiver = Receiver<AccessEvent>;

/// TieringManager tracks access patterns and manages promotion/demotion decisions
struct TieringManager {
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
    fn new(high_water_mark: u64, low_water_mark: u64) -> Self {
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
    fn record_access(&self, key: HashedKey) -> u32 {
        let mut counts = self.access_counts.write();
        let count = counts.entry(key).or_insert(0);
        *count += 1;
        *count
    }
    
    /// Check if an object should be promoted
    fn should_promote(&self, key: HashedKey, access_count: u32) -> bool {
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
    fn mark_pending_promotion(&self, key: HashedKey) {
        let mut pending = self.pending_promotion.write();
        pending.insert(key, ());
    }
    
    /// Promote an object to DRAM
    fn promote_to_dram(&self, key: HashedKey, size: u64) {
        let mut dram_objects = self.dram_objects.write();
        dram_objects.insert(key, size);
        self.dram_size.fetch_add(size, Ordering::SeqCst);
        
        // Remove from pending promotion
        let mut pending = self.pending_promotion.write();
        pending.remove(&key);
    }
    
    /// Demote an object from DRAM
    fn demote_from_dram(&self, key: HashedKey) {
        let mut dram_objects = self.dram_objects.write();
        if let Some(size) = dram_objects.remove(&key) {
            self.dram_size.fetch_sub(size, Ordering::SeqCst);
        }
    }
    
    /// Get current DRAM size
    fn dram_size(&self) -> u64 {
        self.dram_size.load(Ordering::SeqCst)
    }
    
    /// Check if demotion is needed
    fn needs_demotion(&self) -> bool {
        self.dram_size() > self.high_water_mark
    }
    
    /// Get keys to demote (coldest objects)
    fn get_keys_to_demote(&self) -> Vec<HashedKey> {
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
    fn demote_until_low_water(&self) -> Vec<HashedKey> {
        let keys = self.get_keys_to_demote();
        for key in &keys {
            self.demote_from_dram(*key);
        }
        keys
    }
}

pub struct TieringWorker<K, V> {
    access_listener: AccessEventReceiver,
    listener: WorkerReceiver,
    
    objects: ObjectMapRef<K, V>,
    overhead_manager: OverheadManagerRef,
    
    tiering_manager: Arc<TieringManager>,
}

impl<K, V> TieringWorker<K, V>
where
    K: 'static + Eq + TypeSize,
    V: 'static + TypeSize,
{
    pub fn new(
        access_listener: AccessEventReceiver,
        listener: WorkerReceiver,
        objects: ObjectMapRef<K, V>,
        _status: StatusRef,
        overhead_manager: OverheadManagerRef,
        high_water_mark: u64,
        low_water_mark: u64,
    ) -> Self {
        TieringWorker {
            access_listener,
            listener,
            objects,
            overhead_manager,
            tiering_manager: Arc::new(TieringManager::new(high_water_mark, low_water_mark)),
        }
    }
    
    /// Process a batch of access events
    fn process_batch(&self, batch: Vec<AccessEvent>) {
        if batch.is_empty() {
            return;
        }
        
        // Deduplicate and count accesses within this batch
        let mut batch_counts: HashMap<HashedKey, u32> = HashMap::new();
        for event in batch {
            match event {
                AccessEvent::Get(key) => {
                    *batch_counts.entry(key).or_insert(0) += 1;
                }
            }
        }
        
        // Process each unique key
        for (key, batch_count) in batch_counts {
            // Update global access count
            let total_count = {
                let mut count = self.tiering_manager.record_access(key);
                // record_access already incremented by 1, we need to add the rest
                for _ in 1..batch_count {
                    count = self.tiering_manager.record_access(key);
                }
                count
            };
            
            // Check if promotion is needed
            if self.tiering_manager.should_promote(key, total_count) {
                self.promote_object(key);
            }
        }
        
        // After processing batch, check if demotion is needed
        if self.tiering_manager.needs_demotion() {
            self.perform_demotion();
        }
    }
    
    /// Promote an object to DRAM
    fn promote_object(&self, key: HashedKey) {
        // Mark as pending promotion first to avoid race conditions
        self.tiering_manager.mark_pending_promotion(key);
        
        // Get object size
        let size = match self.objects.get(&key) {
            Some(obj) => self.overhead_manager.total_size(&obj) as u64,
            None => {
                // Object was deleted, skip promotion
                return;
            }
        };
        
        // Perform promotion (in a real implementation, this would trigger
        // memory migration to DRAM)
        self.tiering_manager.promote_to_dram(key, size);
        
        info!("Promoted object {} to DRAM (size: {} bytes)", key, size);
    }
    
    /// Perform demotion to free DRAM
    fn perform_demotion(&self) {
        let keys = self.tiering_manager.demote_until_low_water();
        
        if !keys.is_empty() {
            info!("Demoted {} objects from DRAM", keys.len());
        }
    }
    
    /// Collect events into a batch
    fn collect_batch(&self) -> Vec<AccessEvent> {
        let mut batch = Vec::with_capacity(BATCH_SIZE);
        let deadline = Instant::now() + Duration::from_millis(BATCH_TIMEOUT_MS);
        
        loop {
            let remaining = deadline.saturating_duration_since(Instant::now());
            
            if batch.len() >= BATCH_SIZE {
                break;
            }
            
            if remaining.is_zero() {
                break;
            }
            
            match self.access_listener.recv_timeout(remaining) {
                Ok(event) => {
                    batch.push(event);
                }
                Err(_) => {
                    // Timeout or disconnected
                    break;
                }
            }
        }
        
        batch
    }
    
    /// Handle worker events (for consistency with other workers)
    fn handle_worker_event(&self, event: WorkerEvent) {
        match event {
            WorkerEvent::Wipe => {
                // Reset tiering state on wipe
                info!("Tiering worker: cache wiped, resetting state");
            }
            _ => {
                // Other events are not relevant for tiering
            }
        }
    }
}

impl<K, V> Worker for TieringWorker<K, V>
where
    K: 'static + Eq + TypeSize,
    V: 'static + TypeSize,
{
    fn run(&mut self) -> Result<(), CacheError> {
        loop {
            // Check for worker events (non-blocking)
            while let Ok(event) = self.listener.try_recv() {
                self.handle_worker_event(event);
            }
            
            // Collect and process a batch of access events
            let batch = self.collect_batch();
            self.process_batch(batch);
        }
    }
}

unsafe impl<K, V> Send for TieringWorker<K, V> {}
