/*
 * Copyright (c) Kia Shakiba
 *
 * This source code is licensed under the GNU AGPLv3 license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::{
    time::Duration,
    sync::Arc,
};

use typesize::TypeSize;
use crossbeam_channel::Receiver;
use log::{info, debug};

use crate::{
    ObjectMapRef,
    StatusRef,
    OverheadManagerRef,
    error::CacheError,
    tiering::TieringManager,
    worker::{
        Worker,
        WorkerEvent,
    },
};

/// Interval for periodic tiering decisions (migration checks)
const TIERING_INTERVAL: Duration = Duration::from_secs(5);

pub struct TieringWorker<K, V> {
    listener: Receiver<WorkerEvent>,
    
    #[allow(dead_code)]
    objects: ObjectMapRef<K, V>,
    #[allow(dead_code)]
    status: StatusRef,
    #[allow(dead_code)]
    overhead_manager: OverheadManagerRef,
    
    tiering_manager: Arc<TieringManager>,
}

impl<K, V> TieringWorker<K, V>
where
    K: 'static + Eq + TypeSize,
    V: 'static + TypeSize,
{
    pub fn new(
        listener: Receiver<WorkerEvent>,
        objects: ObjectMapRef<K, V>,
        status: StatusRef,
        overhead_manager: OverheadManagerRef,
        tiering_manager: Arc<TieringManager>,
    ) -> Self {
        TieringWorker {
            listener,
            objects,
            status,
            overhead_manager,
            tiering_manager,
        }
    }
    
    /// Process events from the worker manager
    fn process_event(&self, event: WorkerEvent) {
        match event {
            WorkerEvent::Get(hashed_key, hit) => {
                if hit {
                    // Record access and check if we should promote
                    if self.tiering_manager.record_access(hashed_key) {
                        // Object should be promoted to DRAM
                        if self.tiering_manager.promote_to_dram(hashed_key) {
                            debug!("Promoted object {} to DRAM", hashed_key);
                        }
                    }
                }
            }
            
            WorkerEvent::Set(hashed_key, base_size, _expiry, old_object_info) => {
                if old_object_info.is_none() {
                    // New object - register it in PMEM tier
                    self.tiering_manager.register_object(hashed_key, base_size);
                }
                // For updates, the object is already tracked; no need to re-register
            }
            
            WorkerEvent::Del(hashed_key, _expiry) => {
                // Remove object from tiering tracking
                self.tiering_manager.remove_object(hashed_key);
            }
            
            WorkerEvent::Wipe => {
                // Clear all tiering information
                self.tiering_manager.clear();
            }
            
            WorkerEvent::Resize(_max_size) => {
                // Potentially update DRAM threshold based on new cache size
                // For now, keep the existing threshold
            }
            
            _ => {
                // Other events (Ttl, Policy) don't directly affect tiering
            }
        }
    }
    
    /// Perform periodic tiering decisions (demotion checks)
    fn periodic_tiering(&self) {
        // Check if we need to demote any objects from DRAM
        let keys_to_demote = self.tiering_manager.get_keys_to_demote();
        
        if !keys_to_demote.is_empty() {
            info!("Demoting {} objects from DRAM to PMEM", keys_to_demote.len());
            
            for key in keys_to_demote {
                if self.tiering_manager.demote_from_dram(key) {
                    debug!("Demoted object {} from DRAM", key);
                }
            }
        }
    }
}

impl<K, V> Worker for TieringWorker<K, V>
where
    Self: 'static + Send,
    K: Eq + TypeSize,
    V: TypeSize,
{
    fn run(&mut self) -> Result<(), CacheError> {
        let mut last_periodic = std::time::Instant::now();
        
        loop {
            // Process all pending events with a timeout
            let timeout = TIERING_INTERVAL.saturating_sub(last_periodic.elapsed());
            
            if let Ok(event) = self.listener.recv_timeout(timeout) {
                self.process_event(event);
                
                // Process any additional events that are immediately available
                for event in self.listener.try_iter() {
                    self.process_event(event);
                }
            }
            
            // Perform periodic tiering decisions
            if last_periodic.elapsed() >= TIERING_INTERVAL {
                self.periodic_tiering();
                last_periodic = std::time::Instant::now();
            }
        }
    }
}

unsafe impl<K, V> Send for TieringWorker<K, V> {}
