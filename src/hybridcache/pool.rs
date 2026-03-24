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
//! a single `HybridWorkerPool` that spawns ~5 threads:
//!
//! - 1 dispatcher thread (routes `TaggedWorkerEvent` by `CacheId`)
//! - 1 `PolicyWorker` for the small DRAM tier (S3-FIFO + eviction callback)
//! - 1 `PolicyWorker` for the far PMEM tier (LRU)
//! - 1 `TtlWorker` shared across both tiers (partitioned by `CacheId`)
//! - N promotion worker threads (bounded channel for PMEM→DRAM reinsertion)
//!
//! Demotion (DRAM eviction → PMEM write) is handled by a single bounded channel
//! and worker thread, replacing the previous ad-hoc `thread::spawn` approach.

use std::{
    sync::Arc,
    thread,
    time::{Instant, Duration},
};

use crossbeam_channel::{Sender, Receiver, unbounded, bounded};
use typesize::TypeSize;
use log::error;

use crate::{
    ObjectMapRef,
    StatusRef,
    OverheadManagerRef,
    BufferDRAM,
    BufferPMEM,
    EraseKey,
    erase,
    error::CacheError,
    worker::{
        Worker,
        WorkerEvent,
        WorkerSender,
        WorkerReceiver,
        PolicyWorker,
        register_worker,
        CacheId,
        TaggedWorkerEvent,
    },
};

// ── Configuration ─────────────────────────────────────────────────────────────

/// Configuration for the `HybridWorkerPool`.
///
/// Controls the capacities of the bounded demotion and promotion channels, and
/// the number of promotion worker threads.
#[derive(Debug, Clone)]
pub struct HybridWorkerPoolConfig {
    /// Capacity of the bounded demotion channel (DRAM→PMEM migration queue).
    ///
    /// The eviction callback enqueues `(key, value_bytes)` without blocking.
    /// If the channel is full, the demotion is dropped (counted in
    /// `dropped_demotions`).
    ///
    /// Defaults to 512.
    pub demotion_channel_capacity: usize,

    /// Capacity of the bounded promotion channel (PMEM→DRAM reinsertion queue).
    ///
    /// A far-tier hit enqueues `(key, value_bytes)` here without blocking.
    /// If the channel is full, the promotion is skipped (counted in
    /// `dropped_promotions`).
    ///
    /// Defaults to 1024.
    pub promotion_channel_capacity: usize,

    /// Number of promotion worker threads.
    ///
    /// Each thread drains `promotion_rx` and re-inserts items into the small
    /// DRAM tier.
    ///
    /// Defaults to 2.
    pub promotion_threads: usize,
}

impl Default for HybridWorkerPoolConfig {
    fn default() -> Self {
        HybridWorkerPoolConfig {
            demotion_channel_capacity: 512,
            promotion_channel_capacity: 1024,
            promotion_threads: 2,
        }
    }
}

// ── Task types ────────────────────────────────────────────────────────────────

pub struct DemotionTask<K> {
    pub key: K,
    pub value: Vec<u8>,
}

pub struct PromotionTask<K> {
    pub key: K,
    pub value: Vec<u8>,
}

// ── HybridWorkerPool ──────────────────────────────────────────────────────────

/// Shared worker pool for `S3FifoHybridCache`.
///
/// Owns the dispatcher thread, policy workers, TTL worker, and demotion/promotion
/// worker threads. Each `PaperCache` tier is given a sender handle that wraps
/// outgoing `WorkerEvent` with its `CacheId`.
pub struct HybridWorkerPool<K>
where
    K: 'static + Eq + TypeSize + std::fmt::Debug + Clone + Send + Sync,
{
    /// Sender given to the small DRAM PaperCache.
    /// Wraps events with `CacheId::Small` before sending to the dispatcher.
    pub small_sender: WorkerSender,

    /// Sender given to the far PMEM PaperCache.
    /// Wraps events with `CacheId::Far` before sending to the dispatcher.
    pub far_sender: WorkerSender,

    /// Bounded channel for async DRAM→PMEM demotions.
    pub demotion_tx: Sender<DemotionTask<K>>,

    /// Bounded channel for async PMEM→DRAM promotions.
    pub promotion_tx: Sender<PromotionTask<K>>,

    /// Phantom data to associate the pool with the key type.
    _phantom: std::marker::PhantomData<K>,
}

impl<K> HybridWorkerPool<K>
where
    K: 'static + Eq + TypeSize + std::fmt::Debug + Clone + Send + Sync,
{
    /// Creates a new `HybridWorkerPool` and spawns all worker threads.
    ///
    /// Returns the pool and two sender handles (one per tier).
    ///
    /// # Arguments
    ///
    /// - `small_objects`, `small_status`, `small_overhead_manager`: refs for the small DRAM tier
    /// - `far_objects`, `far_status`, `far_overhead_manager`: refs for the far PMEM tier
    /// - `eviction_callback`: fired by the small tier's `PolicyWorker` on eviction
    /// - `config`: pool configuration (channel capacities, thread counts)
    ///
    /// # Errors
    ///
    /// Returns [`CacheError`] if worker initialization fails.
    pub fn new(
        small_objects: &ObjectMapRef<K, BufferDRAM>,
        small_status: &StatusRef,
        small_overhead_manager: &OverheadManagerRef,
        far_objects: &ObjectMapRef<K, BufferPMEM>,
        far_status: &StatusRef,
        far_overhead_manager: &OverheadManagerRef,
        eviction_callback: Box<dyn for<'a> Fn(crate::HashedKey, Arc<BufferDRAM>, &'a K) + Send + Sync>,
        config: HybridWorkerPoolConfig,
    ) -> Result<Self, CacheError> {
        // ── Dispatcher: single inbound channel, routes by CacheId ────────────

        let (dispatcher_tx, dispatcher_rx) = unbounded::<TaggedWorkerEvent>();

        // ── PolicyWorker channels (per-tier, incompatible internal state) ────

        let (small_policy_tx, small_policy_rx): (Sender<WorkerEvent>, Receiver<WorkerEvent>) = unbounded();
        let (far_policy_tx, far_policy_rx): (Sender<WorkerEvent>, Receiver<WorkerEvent>) = unbounded();

        // ── Shared TtlWorker channel ──────────────────────────────────────────
        //
        // The TtlWorker receives events from both tiers. We'll handle the
        // dual-ObjectMap problem by creating a hybrid-aware TtlWorker that
        // receives `TaggedWorkerEvent` and dispatches erasure to the correct tier.
        //
        // For now, we'll spawn TWO TtlWorkers (one per tier) and route TTL events
        // by CacheId. This is simpler and matches the existing architecture.

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

        register_worker(crate::worker::TtlWorker::<K, BufferDRAM>::new(
            small_ttl_rx,
            small_objects.clone(),
            small_status.clone(),
            small_overhead_manager.clone(),
        ));

        // ── Spawn TtlWorker for far PMEM tier ─────────────────────────────────

        register_worker(crate::worker::TtlWorker::<K, BufferPMEM>::new(
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

        // ── Demotion and promotion channels ───────────────────────────────────

        let (demotion_tx, demotion_rx) = bounded::<DemotionTask<K>>(config.demotion_channel_capacity);
        let (promotion_tx, promotion_rx) = bounded::<PromotionTask<K>>(config.promotion_channel_capacity);

        // ── Sender handles for each tier ──────────────────────────────────────
        //
        // Each tier's PaperCache will be given a sender that wraps outgoing
        // `WorkerEvent` with its `CacheId` before sending to the dispatcher.

        let small_sender = dispatcher_tx.clone();
        let far_sender = dispatcher_tx;

        Ok(HybridWorkerPool {
            small_sender,
            far_sender,
            demotion_tx,
            promotion_tx,
            _phantom: std::marker::PhantomData,
        })
    }

    /// Spawns the demotion worker thread.
    ///
    /// The demotion worker drains `demotion_rx` and writes each task to the
    /// far PMEM tier using `far.set(k, &val, None)`.
    ///
    /// This must be called AFTER the pool is constructed so that the worker
    /// can access `far` (which is created outside the pool).
    pub fn spawn_demotion_worker(
        demotion_rx: Receiver<DemotionTask<K>>,
        far: Arc<crate::PaperCache<K, BufferPMEM>>,
        in_flight_demotions: Arc<dashmap::DashSet<K>>,
        demoted_keys: Arc<dashmap::DashSet<K>>,
        demotions_counter: Arc<std::sync::atomic::AtomicU64>,
        dropped_demotions_counter: Arc<std::sync::atomic::AtomicU64>,
    ) {
        thread::Builder::new()
            .name("hybrid-demotion".to_string())
            .spawn(move || {
                while let Ok(task) = demotion_rx.recv() {
                    if far.set(task.key.clone(), &task.value, None).is_ok() {
                        in_flight_demotions.remove(&task.key);
                        demoted_keys.insert(task.key);
                        demotions_counter.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    } else {
                        in_flight_demotions.remove(&task.key);
                        dropped_demotions_counter.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    }
                }
            })
            .expect("failed to spawn demotion worker thread");
    }

    /// Spawns `n` promotion worker threads.
    ///
    /// Each promotion worker drains `promotion_rx` and re-inserts items into
    /// the small DRAM tier using `small.set(k, &val, None)`.
    ///
    /// This must be called AFTER the pool is constructed so that the workers
    /// can access `small` (which is created outside the pool).
    pub fn spawn_promotion_workers(
        promotion_rx: Receiver<PromotionTask<K>>,
        small: Arc<crate::PaperCache<K, BufferDRAM>>,
        promotions_counter: Arc<std::sync::atomic::AtomicU64>,
        n: usize,
    ) {
        for i in 0..n {
            let rx = promotion_rx.clone();
            let small_clone = Arc::clone(&small);
            let counter_clone = Arc::clone(&promotions_counter);
            thread::Builder::new()
                .name(format!("hybrid-promotion-{}", i))
                .spawn(move || {
                    while let Ok(task) = rx.recv() {
                        let _ = small_clone.set(task.key, &task.value, None);
                        counter_clone.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    }
                })
                .expect("failed to spawn promotion worker thread");
        }
    }
}
