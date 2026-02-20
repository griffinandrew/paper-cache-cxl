/*
 * Copyright (c) Kia Shakiba
 *
 * This source code is licensed under the GNU AGPLv3 license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! Multitiering Worker
//!
//! Background worker that integrates `MultitieringManager` into the worker-manager
//! event loop. Mirrors the structure of `TieringWorker` but routes to the 3-state
//! `MultitieringManager` instead of the 2-state `TieringManager`.
//!
//! Objects are moved between tiers at a configurable time interval (matching the
//! behaviour of `TieringWorker`). On each get-event the access count is updated;
//! the actual promotion/demotion decisions happen in the periodic sweep.

use std::{
    sync::Arc,
    time::Duration,
};

use typesize::TypeSize;
use crossbeam_channel::Receiver;
use log::{info, debug};

use crate::{
    ObjectMapRef,
    StatusRef,
    OverheadManagerRef,
    error::CacheError,
    worker::{Worker, WorkerEvent},
    tiering::MultitieringManager,
    tiering::multitier_manager::Tier,
};

pub struct MultitieringWorker<K, V> {
    listener: Receiver<WorkerEvent>,

    objects: ObjectMapRef<K, V>,
    #[allow(dead_code)]
    status: StatusRef,
    #[allow(dead_code)]
    overhead_manager: OverheadManagerRef,

    tiering_manager: Arc<MultitieringManager<K, V>>,
}

#[cfg(not(any(feature = "alloc_api_exp", feature = "global_hashtable_pmem")))]
impl<K, V> MultitieringWorker<K, V>
where
    K: 'static + Eq + TypeSize + Clone,
    V: 'static + TypeSize + Clone + AsRef<[u8]>,
{
    pub fn new(
        listener: Receiver<WorkerEvent>,
        objects: ObjectMapRef<K, V>,
        status: StatusRef,
        overhead_manager: OverheadManagerRef,
        tiering_manager: Arc<MultitieringManager<K, V>>,
    ) -> Self {
        MultitieringWorker {
            listener,
            objects,
            status,
            overhead_manager,
            tiering_manager,
        }
    }

    fn process_event(&self, event: WorkerEvent) {
        match event {
            WorkerEvent::Get(hashed_key, hit) => {
                if hit {
                    // Record the access; tier moves happen in the periodic sweep.
                    self.tiering_manager.record_access(hashed_key);
                }
            }

            WorkerEvent::Set(hashed_key, base_size, _expiry, old_object_info) => {
                if old_object_info.is_none() {
                    // New object — register in the Cold tier.
                    self.tiering_manager.register_object(hashed_key, base_size);
                }
                // Updates to existing objects don't change tier placement.
            }

            WorkerEvent::Del(hashed_key, _expiry) => {
                self.tiering_manager.remove_object(hashed_key);
            }

            WorkerEvent::Wipe => {
                self.tiering_manager.clear();
            }

            WorkerEvent::Resize(_) | WorkerEvent::Ttl(..) | WorkerEvent::Policy(_) => {
                // These events do not affect multitiering placement.
            }
        }
    }

    /// Periodic sweep: promote eligible objects and demote objects that exceed watermarks.
    fn periodic_tiering(&self) {
        // ── Promotions ───────────────────────────────────────────────────────────
        let to_promote = self.tiering_manager.get_keys_to_promote();
        for (key, target_tier) in to_promote {
            match target_tier {
                Tier::DramPtrToPmem => {
                    if self.tiering_manager.promote_to_warm(key) {
                        debug!("Multitiering: promoted key {} to Warm (DramPtrToPmem)", key);
                    }
                }
                Tier::DramAndPmem => {
                    if let Some(object_ref) = self.objects.get(&key) {
                        if self.tiering_manager.promote_to_hot(key, &*object_ref) {
                            debug!("Multitiering: promoted key {} to Hot (DramAndPmem)", key);
                        }
                    }
                }
                Tier::PmemOnly => {}
            }
        }

        // ── Demotions ────────────────────────────────────────────────────────────
        let to_demote = self.tiering_manager.get_keys_to_demote();
        if !to_demote.is_empty() {
            info!("Multitiering: demoting {} objects to Cold", to_demote.len());
            for (key, _from_tier) in to_demote {
                self.tiering_manager.demote_to_cold(key);
                debug!("Multitiering: demoted key {} to Cold (PmemOnly)", key);
            }
        }
    }
}

#[cfg(any(feature = "alloc_api_exp", feature = "global_hashtable_pmem"))]
impl<K, V> MultitieringWorker<K, V>
where
    K: 'static + Eq + TypeSize + Clone,
    V: 'static + TypeSize + Clone + AsRef<[u8]>,
{
    pub fn new(
        listener: Receiver<WorkerEvent>,
        objects: ObjectMapRef<K, V>,
        status: StatusRef,
        overhead_manager: OverheadManagerRef,
        tiering_manager: Arc<MultitieringManager<K, V>>,
    ) -> Self {
        MultitieringWorker {
            listener,
            objects,
            status,
            overhead_manager,
            tiering_manager,
        }
    }

    fn process_event(&self, event: WorkerEvent) {
        match event {
            WorkerEvent::Get(hashed_key, hit) => {
                if hit {
                    self.tiering_manager.record_access(hashed_key);
                }
            }

            WorkerEvent::Set(hashed_key, base_size, _expiry, old_object_info) => {
                if old_object_info.is_none() {
                    self.tiering_manager.register_object(hashed_key, base_size);
                }
            }

            WorkerEvent::Del(hashed_key, _expiry) => {
                self.tiering_manager.remove_object(hashed_key);
            }

            WorkerEvent::Wipe => {
                self.tiering_manager.clear();
            }

            WorkerEvent::Resize(_) | WorkerEvent::Ttl(..) | WorkerEvent::Policy(_) => {}
        }
    }

    fn periodic_tiering(&self) {
        let to_promote = self.tiering_manager.get_keys_to_promote();
        for (key, target_tier) in to_promote {
            match target_tier {
                Tier::DramPtrToPmem => {
                    if self.tiering_manager.promote_to_warm(key) {
                        debug!("Multitiering: promoted key {} to Warm (DramPtrToPmem)", key);
                    }
                }
                Tier::DramAndPmem => {
                    if let Some(object_ref) = self.objects.read().unwrap().get(&key) {
                        if self.tiering_manager.promote_to_hot(key, &*object_ref) {
                            debug!("Multitiering: promoted key {} to Hot (DramAndPmem)", key);
                        }
                    }
                }
                Tier::PmemOnly => {}
            }
        }

        let to_demote = self.tiering_manager.get_keys_to_demote();
        if !to_demote.is_empty() {
            info!("Multitiering: demoting {} objects to Cold", to_demote.len());
            for (key, _from_tier) in to_demote {
                self.tiering_manager.demote_to_cold(key);
                debug!("Multitiering: demoted key {} to Cold (PmemOnly)", key);
            }
        }
    }
}

impl<K, V> Worker for MultitieringWorker<K, V>
where
    Self: 'static + Send,
    K: Eq + TypeSize + Clone,
    V: TypeSize + Clone + AsRef<[u8]>,
{
    fn run(&mut self) -> Result<(), CacheError> {
        let interval = self.tiering_manager.evaluation_interval();
        let mut last_periodic = std::time::Instant::now();

        loop {
            // Calculate how long until the next periodic sweep.
            let elapsed = last_periodic.elapsed();
            let timeout = if elapsed >= interval {
                Duration::from_millis(1)
            } else {
                interval - elapsed
            };

            if let Ok(event) = self.listener.recv_timeout(timeout) {
                self.process_event(event);
                // Drain any additional events queued since the last recv.
                for event in self.listener.try_iter() {
                    self.process_event(event);
                }
            }

            if last_periodic.elapsed() >= interval {
                self.periodic_tiering();
                last_periodic = std::time::Instant::now();
            }
        }
    }
}

unsafe impl<K, V> Send for MultitieringWorker<K, V> {}

