/*
 * Copyright (c) Kia Shakiba
 *
 * This source code is licensed under the GNU AGPLv3 license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! Tiering Worker
//!
//! This module implements the TieringWorker which integrates the TieringManager
//! into the worker manager workflow. The worker:
//!
//! - Receives WorkerEvents (Get, Set, Del, etc.) from the worker manager
//! - Tracks object access patterns in coordination with the TieringManager
//! - Periodically triggers promotion/demotion decisions
//! - Maintains data consistency across both tiers
//!
//! The TieringWorker runs as a background thread alongside PolicyWorker and TtlWorker,
//! processing events in real-time and making periodic tiering decisions.

use std::{sync::Arc, time::Duration};

use crossbeam_channel::Receiver;
use log::{debug, info};
use typesize::TypeSize;

use crate::{
    ObjectMapRef,
    OverheadManagerRef,
    StatusRef,
    error::CacheError,
    //tiering::TieringManager,
    worker::{Worker, WorkerEvent},
};

#[cfg(feature = "enable_tiering_manager")]
use crate::tiering::TieringManager;

/// Interval for periodic tiering decisions (migration checks)
const TIERING_INTERVAL: Duration = Duration::from_secs(5);

pub struct TieringWorker<K, V> {
    listener: Receiver<WorkerEvent>,

    objects: ObjectMapRef<K, V>,
    #[allow(dead_code)]
    status: StatusRef,
    #[allow(dead_code)]
    overhead_manager: OverheadManagerRef,

    tiering_manager: Arc<TieringManager<K, V>>,
}

#[cfg(all(
    feature = "key_value_pmem", 
    not(feature = "global_hashtable_pmem"),
    not(feature = "sets_dram") // <-- Add this exclusion
))]
impl<K, V> TieringWorker<K, V>
where
    K: 'static + Eq + TypeSize + Clone,
    V: 'static + TypeSize + Clone + AsRef<[u8]>,
{
    pub fn new(
        listener: Receiver<WorkerEvent>,
        objects: ObjectMapRef<K, V>,
        status: StatusRef,
        overhead_manager: OverheadManagerRef,
        tiering_manager: Arc<TieringManager<K, V>>,
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
                        // Object should be promoted to DRAM - copy the Object
                        if let Some(object_ref) = self.objects.get(&hashed_key) {
                            if self
                                .tiering_manager
                                .promote_to_dram_with_object(hashed_key, &*object_ref)
                            {
                                debug!("Promoted object {} to DRAM", hashed_key);
                            }
                        }
                    }
                }
            }

            WorkerEvent::Promote(hashed_key) => {
                if let Some(object_ref) = self.objects.get(&hashed_key) {
                    if self
                        .tiering_manager
                        .promote_to_dram_with_object(hashed_key, &*object_ref)
                    {
                        debug!("Promoted object {} to DRAM (ghost hit)", hashed_key);
                    }
                }
            }

            WorkerEvent::Set(hashed_key, base_size, _expiry, old_object_info) => {
                if old_object_info.is_none() {
                    // New object - register it in PMEM tier
                    self.tiering_manager.register_object(hashed_key, base_size);
                } else {
                    // Object updated - update DRAM copy if it exists
                    if let Some(object_ref) = self.objects.get(&hashed_key) {
                        self.tiering_manager
                            .update_dram_copy(hashed_key, &*object_ref);
                    }
                }
            }

            WorkerEvent::Del(hashed_key, _expiry) => {
                // Remove object from tiering tracking (this also removes from DRAM cache)
                self.tiering_manager.remove_object(hashed_key);
            }

            WorkerEvent::Wipe => {
                // Clear all tiering information (including DRAM cache)
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
            info!(
                "Demoting {} objects from DRAM to PMEM",
                keys_to_demote.len()
            );

            for key in keys_to_demote {
                if self.tiering_manager.demote_from_dram(key) {
                    debug!("Demoted object {} from DRAM", key);
                }
            }
        }
    }
}

#[cfg(all(feature = "key_value_pmem", feature = "global_hashtable_pmem"))]
impl<K, V> TieringWorker<K, V>
where
    K: 'static + Eq + TypeSize + Clone,
    V: 'static + TypeSize + Clone + AsRef<[u8]>,
{
    pub fn new(
        listener: Receiver<WorkerEvent>,
        objects: ObjectMapRef<K, V>,
        status: StatusRef,
        overhead_manager: OverheadManagerRef,
        tiering_manager: Arc<TieringManager<K, V>>,
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
                        // Object should be promoted to DRAM - copy the Object
                        //if let Some(object_ref) = self.objects.get(&hashed_key) {
                        if let Some(object_ref) = self.objects.read().unwrap().get(&hashed_key) {
                            if self
                                .tiering_manager
                                .promote_to_dram_with_object(hashed_key, &*object_ref)
                            {
                                debug!("Promoted object {} to DRAM", hashed_key);
                            }
                        }
                    }
                }
            }

            WorkerEvent::Promote(hashed_key) => {
                if let Some(object_ref) = self.objects.read().unwrap().get(&hashed_key) {
                    if self
                        .tiering_manager
                        .promote_to_dram_with_object(hashed_key, &*object_ref)
                    {
                        debug!("Promoted object {} to DRAM (ghost hit)", hashed_key);
                    }
                }
            }

            WorkerEvent::Set(hashed_key, base_size, _expiry, old_object_info) => {
                if old_object_info.is_none() {
                    // New object - register it in PMEM tier
                    self.tiering_manager.register_object(hashed_key, base_size);
                } else {
                    // Object updated - update DRAM copy if it exists
                    //if let Some(object_ref) = self.objects.get(&hashed_key) {
                    if let Some(object_ref) = self.objects.read().unwrap().get(&hashed_key) {
                        self.tiering_manager
                            .update_dram_copy(hashed_key, &*object_ref);
                    }
                }
            }

            WorkerEvent::Del(hashed_key, _expiry) => {
                // Remove object from tiering tracking (this also removes from DRAM cache)
                self.tiering_manager.remove_object(hashed_key);
            }

            WorkerEvent::Wipe => {
                // Clear all tiering information (including DRAM cache)
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
            info!(
                "Demoting {} objects from DRAM to PMEM",
                keys_to_demote.len()
            );

            for key in keys_to_demote {
                if self.tiering_manager.demote_from_dram(key) {
                    debug!("Demoted object {} from DRAM", key);
                }
            }
        }
    }
}

#[cfg(all(
    feature = "key_value_pmem",
    feature = "sets_dram",
    not(feature = "global_hashtable_pmem")
))]
impl<K, V> TieringWorker<K, V>
where
    K: 'static + Eq + TypeSize + Clone,
    V: 'static + TypeSize + Clone + AsRef<[u8]>,
{
    pub fn new(
        listener: Receiver<WorkerEvent>,
        objects: ObjectMapRef<K, V>,
        status: StatusRef,
        overhead_manager: OverheadManagerRef,
        tiering_manager: Arc<TieringManager<K, V>>,
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
                        // Object should be promoted to DRAM - copy the Object
                        if let Some(object_ref) = self.objects.get(&hashed_key) {
                            if self
                                .tiering_manager
                                .promote_to_dram_with_object(hashed_key, &*object_ref)
                            {
                                debug!("Promoted object {} to DRAM", hashed_key);
                            }
                        }
                    }
                }
            }

            WorkerEvent::Promote(hashed_key) => {
                if let Some(object_ref) = self.objects.get(&hashed_key) {
                    if self
                        .tiering_manager
                        .promote_to_dram_with_object(hashed_key, &*object_ref)
                    {
                        debug!("Promoted object {} to DRAM (ghost hit)", hashed_key);
                    }
                }
            }

            WorkerEvent::Set(hashed_key, base_size, _expiry, old_object_info) => {
                if old_object_info.is_none() {
                    // New object - register it in PMEM tier
                    //self.tiering_manager.register_object(hashed_key, base_size);
                    //self.tiering_manager.set_tiering_for_new_object(&hashed_key);
                } else {
                    // Object updated - update DRAM copy if it exists
                    if let Some(object_ref) = self.objects.get(&hashed_key) {
                        self.tiering_manager
                            .update_dram_copy(hashed_key, &*object_ref);
                    }
                }
            }

            WorkerEvent::Del(hashed_key, _expiry) => {
                // Remove object from tiering tracking (this also removes from DRAM cache)
                self.tiering_manager.remove_object(hashed_key);
            }

            WorkerEvent::Wipe => {
                // Clear all tiering information (including DRAM cache)
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
            info!(
                "Demoting {} objects from DRAM to PMEM",
                keys_to_demote.len()
            );

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
    K: Eq + TypeSize + Clone,
    V: TypeSize + Clone + AsRef<[u8]>,
{
    fn run(&mut self) -> Result<(), CacheError> {
        let mut last_periodic = std::time::Instant::now();

        loop {
            // Calculate timeout, ensuring we don't busy wait
            let elapsed = last_periodic.elapsed();
            let timeout = if elapsed >= TIERING_INTERVAL {
                Duration::from_millis(1) // Minimal timeout to avoid busy waiting
            } else {
                TIERING_INTERVAL - elapsed
            };

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

            //println!("SIZE: {}", self.tiering_manager.stats().dram_size);
            //println!("OBJS: {}", self.tiering_manager.stats().dram_objects);
            //println!("PROS: {}", self.tiering_manager.stats().promotions);
            //println!("DEMS: {}", self.tiering_manager.stats().demotions);
            //println!("PMEM: {}", self.tiering_manager.stats().pmem_only_objects);
        }
    }
}

unsafe impl<K, V> Send for TieringWorker<K, V> {}
