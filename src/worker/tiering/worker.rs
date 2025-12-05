/*
 * Copyright (c) Kia Shakiba
 *
 * This source code is licensed under the GNU AGPLv3 license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};
use crossbeam_channel::{Receiver, bounded};
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

use super::TieringManager;

const BATCH_SIZE: usize = 1024;
const BATCH_TIMEOUT_MS: u64 = 100;

/// Access event for tiering
#[derive(Clone, Copy, Debug)]
pub enum AccessEvent {
    Get(HashedKey),
}

pub type AccessEventReceiver = Receiver<AccessEvent>;

pub struct TieringWorker<K, V> {
    access_listener: AccessEventReceiver,
    worker_listener: WorkerReceiver,
    
    objects: ObjectMapRef<K, V>,
    status: StatusRef,
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
        worker_listener: WorkerReceiver,
        objects: ObjectMapRef<K, V>,
        status: StatusRef,
        overhead_manager: OverheadManagerRef,
        high_water_mark: u64,
        low_water_mark: u64,
    ) -> Self {
        TieringWorker {
            access_listener,
            worker_listener,
            objects,
            status,
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
            while let Ok(event) = self.worker_listener.try_recv() {
                self.handle_worker_event(event);
            }
            
            // Collect and process a batch of access events
            let batch = self.collect_batch();
            self.process_batch(batch);
        }
    }
}

unsafe impl<K, V> Send for TieringWorker<K, V> {}
