/*
 * Copyright (c) Kia Shakiba
 *
 * This source code is licensed under the GNU AGPLv3 license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! Shared worker pool for `S3FifoHybridCache`.
//!
//! Instead of each `PaperCache` tier spawning its own `WorkerManager`,
//! `PolicyWorker`, and `TtlWorker` (8–10 threads total), the hybrid cache uses
//! a single `HybridWorkerPool` that spawns ~4 threads:
//!
//! - 1 dispatcher thread (routes `TaggedWorkerEvent` by `CacheId`)
//! - 1 `PolicyWorker` for the small DRAM tier (S3-FIFO + eviction callback)
//! - 1 `PolicyWorker` for the far PMEM tier (LRU)
//! - 2 `TtlWorker` instances (one per tier, routed by `CacheId`)
//!
//! Demotion and promotion worker threads are spawned separately by
//! `S3FifoHybridCache` after construction, as they need access to the
//! fully-constructed PaperCache instances.

use std::{
    sync::Arc,
    thread,
};

use crossbeam_channel::{Sender, Receiver, unbounded};
use typesize::TypeSize;
use log::error;

use crate::{
    ObjectMapRef,
    StatusRef,
    OverheadManagerRef,
    BufferDRAM,
    BufferPMEM,
    error::CacheError,
    worker::{
        Worker,
        WorkerEvent,
        WorkerReceiver,
        PolicyWorker,
        TtlWorker,
        register_worker,
        CacheId,
        TaggedWorkerEvent,
        HybridWorkerSender,
    },
};

// ── HybridWorkerPool ──────────────────────────────────────────────────────────

/// Shared worker pool for `S3FifoHybridCache`.
///
/// Spawns a dispatcher thread that routes events to tier-specific workers.
/// Each `PaperCache` tier is given a sender handle (`small_sender` or
/// `far_sender`) that wraps outgoing `WorkerEvent` with its `CacheId`.
pub struct HybridWorkerPool {
    /// Sender given to the small DRAM PaperCache.
    pub small_sender: HybridWorkerSender,

    /// Sender given to the far PMEM PaperCache.
    pub far_sender: HybridWorkerSender,
}

impl HybridWorkerPool {
    /// Creates a new `HybridWorkerPool` and spawns all worker threads.
    ///
    /// Returns the pool with two sender handles (one per tier).
    ///
    /// # Arguments
    ///
    /// - `small_objects`, `small_status`, `small_overhead_manager`: refs for the small DRAM tier
    /// - `far_objects`, `far_status`, `far_overhead_manager`: refs for the far PMEM tier
    /// - `eviction_callback`: fired by the small tier's `PolicyWorker` on eviction
    ///
    /// # Errors
    ///
    /// Returns [`CacheError`] if worker initialization fails.
    pub fn new<K>(
        small_objects: &ObjectMapRef<K, BufferDRAM>,
        small_status: &StatusRef,
        small_overhead_manager: &OverheadManagerRef,
        far_objects: &ObjectMapRef<K, BufferPMEM>,
        far_status: &StatusRef,
        far_overhead_manager: &OverheadManagerRef,
        eviction_callback: Box<dyn for<'a> Fn(crate::HashedKey, Arc<BufferDRAM>, &'a K) + Send + Sync>,
    ) -> Result<Self, CacheError>
    where
        K: 'static + Eq + TypeSize + std::fmt::Debug + Clone + Send + Sync,
    {
        // ── Dispatcher: single inbound channel, routes by CacheId ────────────

        let (dispatcher_tx, dispatcher_rx) = unbounded::<TaggedWorkerEvent>();

        // ── PolicyWorker channels (per-tier, incompatible internal state) ────

        let (small_policy_tx, small_policy_rx): (Sender<WorkerEvent>, Receiver<WorkerEvent>) = unbounded();
        let (far_policy_tx, far_policy_rx): (Sender<WorkerEvent>, Receiver<WorkerEvent>) = unbounded();

        // ── TtlWorker channels (per-tier) ─────────────────────────────────────

        let (small_ttl_tx, small_ttl_rx): (Sender<WorkerEvent>, Receiver<WorkerEvent>) = unbounded();
        let (far_ttl_tx, far_ttl_rx): (Sender<WorkerEvent>, Receiver<WorkerEvent>) = unbounded();

        // ── Spawn PolicyWorker for small DRAM tier ────────────────────────────

        register_worker(PolicyWorker::<K, BufferDRAM>::new_with_eviction_callback(
            small_policy_rx,
            small_objects.clone(),
            small_status.clone(),
            small_overhead_manager.clone(),
            eviction_callback,
        )?);

        // ── Spawn PolicyWorker for far PMEM tier ──────────────────────────────

        register_worker(PolicyWorker::<K, BufferPMEM>::new(
            far_policy_rx,
            far_objects.clone(),
            far_status.clone(),
            far_overhead_manager.clone(),
        )?);

        // ── Spawn TtlWorker for small DRAM tier ───────────────────────────────

        register_worker(TtlWorker::<K, BufferDRAM>::new(
            small_ttl_rx,
            small_objects.clone(),
            small_status.clone(),
            small_overhead_manager.clone(),
        ));

        // ── Spawn TtlWorker for far PMEM tier ─────────────────────────────────

        register_worker(TtlWorker::<K, BufferPMEM>::new(
            far_ttl_rx,
            far_objects.clone(),
            far_status.clone(),
            far_overhead_manager.clone(),
        ));

        // ── Spawn dispatcher thread ───────────────────────────────────────────

        thread::Builder::new()
            .name("hybrid-dispatcher".to_string())
            .spawn(move || {
                loop {
                    let Ok(tagged_event) = dispatcher_rx.recv() else {
                        return;
                    };

                    let (policy_tx, ttl_tx) = match tagged_event.cache_id {
                        CacheId::Small => (&small_policy_tx, &small_ttl_tx),
                        CacheId::Far => (&far_policy_tx, &far_ttl_tx),
                    };

                    // Route to PolicyWorker
                    if let Err(err) = policy_tx.try_send(tagged_event.event.clone()) {
                        error!("Hybrid dispatcher could not send to PolicyWorker: {err:?}");
                        return;
                    }

                    // Route to TtlWorker
                    if let Err(err) = ttl_tx.try_send(tagged_event.event) {
                        error!("Hybrid dispatcher could not send to TtlWorker: {err:?}");
                        return;
                    }
                }
            })
            .map_err(|e| {
                error!("Failed to spawn hybrid dispatcher thread: {e}");
                CacheError::Internal
            })?;

        // ── Sender handles for each tier ──────────────────────────────────────

        let small_sender = dispatcher_tx.clone();
        let far_sender = dispatcher_tx;

        Ok(HybridWorkerPool {
            small_sender,
            far_sender,
        })
    }
}
