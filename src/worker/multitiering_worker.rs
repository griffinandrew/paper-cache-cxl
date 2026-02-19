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

use std::sync::Arc;

use typesize::TypeSize;
use crossbeam_channel::Receiver;
use log::debug;

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

impl<K, V> MultitieringWorker<K, V>
where
    K: 'static + Eq + TypeSize + Clone,
    V: 'static + TypeSize + Clone,
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
                    if let Some(target_tier) = self.tiering_manager.record_access(hashed_key) {
                        match target_tier {
                            Tier::DramPtrToPmem => {
                                // Zero-copy promotion: only metadata moves to DRAM.
                                if self.tiering_manager.promote_to_warm(hashed_key) {
                                    debug!("Multitiering: promoted key {} to Warm (DramPtrToPmem)", hashed_key);
                                }
                            }
                            Tier::DramAndPmem => {
                                // Full copy promotion: payload is physically copied to DRAM.
                                if self.tiering_manager.promote_to_hot(hashed_key) {
                                    debug!("Multitiering: promoted key {} to Hot (DramAndPmem)", hashed_key);
                                }
                            }
                            Tier::PmemOnly => {}
                        }
                    }
                }
            }

            WorkerEvent::Set(hashed_key, base_size, _expiry, old_object_info) => {
                if old_object_info.is_none() {
                    // New object — register in Cold tier.
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
}

impl<K, V> Worker for MultitieringWorker<K, V>
where
    Self: 'static + Send,
    K: Eq + TypeSize + Clone,
    V: TypeSize + Clone,
{
    fn run(&mut self) -> Result<(), CacheError> {
        loop {
            let Ok(event) = self.listener.recv() else {
                return Ok(());
            };
            self.process_event(event);
            // Drain any additional events queued since the last recv.
            for event in self.listener.try_iter() {
                self.process_event(event);
            }
        }
    }
}

unsafe impl<K, V> Send for MultitieringWorker<K, V> {}
