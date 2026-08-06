/*
 * Copyright (c) Kia Shakiba
 *
 * This source code is licensed under the GNU AGPLv3 license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::sync::Arc;
use typesize::TypeSize;
use crossbeam_channel::{unbounded, TryRecvError};
use log::error;

use crate::{
	ObjectMapRef,
	StatusRef,
	OverheadManagerRef,
	error::CacheError,
	worker::{
		Worker,
		WorkerEvent,
		WorkerSender,
		WorkerReceiver,
		WorkerHandles,
		PolicyWorker,
		TtlWorker,
		register_worker,
	},
};

#[cfg(all(feature = "key_value_pmem", feature = "enable_tiering_manager"))]
use crate::{
	tiering::TieringManager,
	worker::TieringWorker,
};

#[cfg(any(feature = "lru_hybrid_cache", feature = "lfu_hybrid_cache", feature = "two_q_hybrid_cache", feature = "fifo_hybrid_cache", feature = "lru_sized_hybrid_cache", feature = "s3_fifo_hybrid_cache", feature = "two_q_ghost_hybrid_cache", feature = "s3_fifo_ghost_hybrid_cache", feature = "s3_fifo_ghost_lazy_demotion_hybrid_cache", feature = "s3_fifo_ghost_lazy_demotion_fast_admission_hybrid_cache"))]
use crate::worker::policy::Tier;

pub struct WorkerManager {
	listener: WorkerReceiver,
	workers: Arc<Box<[WorkerSender]>>,
}

impl Worker for WorkerManager {
	// Busy-polls via `try_recv` rather than blocking on `recv`. Measured
	// directly (real `perf record` profiling against paper-benchmark-cxl,
	// both a PMEM-mixed and an all-DRAM config): at this crate's real
	// single-client request rates, the gap between successive get()/set()
	// calls consistently exceeds crossbeam's internal spin/yield backoff
	// window, so this thread's blocking `recv()` was parking via a real
	// OS futex between nearly every event -- ~30%+ of total CPU cycles
	// went to futex_wait/futex_wake/sched_yield machinery in both configs,
	// not to any allocator-specific cost. `PolicyWorker`/`TtlWorker` never
	// had this problem: both already drain via `try_iter()` (non-blocking)
	// in their own `run()` loops -- this was the one remaining blocking
	// wait in the whole worker architecture.
	//
	// Trade-off, accepted deliberately: this thread now spins continuously
	// rather than sleeping between events, permanently consuming one CPU
	// core even when the cache is idle. Only worth it because there's a
	// core to spare in this crate's real deployment/benchmark shape (a
	// handful of worker threads on an 8-core machine) -- if that's ever
	// not true, this trades a real latency win for a real throughput/power
	// cost elsewhere.
	fn run(&mut self) -> Result<(), CacheError> {
		loop {
			let event = match self.listener.try_recv() {
				Ok(event) => event,
				Err(TryRecvError::Empty) => {
					std::hint::spin_loop();
					continue;
				},
				Err(TryRecvError::Disconnected) => return Ok(()),
			};

			// Forward first (every sub-worker needs its own copy of
			// `Shutdown` to stop its own loop -- see each `Worker::run`
			// impl), *then* stop ourselves. Don't join the sub-workers
			// here: `PaperCache::drop` already holds their `JoinHandle`s
			// directly (see `WorkerHandles`, returned from `WorkerManager::
			// new`/`new_with_tier_migration` at construction time) and
			// joins them itself after this thread's own handle, so a
			// second join here would be redundant, not incorrect, but
			// pointless complexity.
			let is_shutdown = matches!(event, WorkerEvent::Shutdown);

			for worker in self.workers.iter() {
				if let Err(err) = worker.try_send(event.clone()) {
					error!("Could not send event to worker: {err:?}");
					return Err(CacheError::Internal);
				}
			}

			if is_shutdown {
				return Ok(());
			}
		}
	}
}

impl WorkerManager {
	#[cfg(all(feature = "key_value_pmem", feature = "enable_tiering_manager"))]
	pub fn new<K, V>(
		listener: WorkerReceiver,
		objects: &ObjectMapRef<K, V>,
		status: &StatusRef,
		overhead_manager: &OverheadManagerRef,
		tiering_manager: &Arc<TieringManager<K, V>>,
	) -> Result<(Self, WorkerHandles), CacheError>
	where
		K: 'static + Eq + TypeSize + Clone,
		V: 'static + TypeSize + Clone + AsRef<[u8]>,
	{
		let (policy_worker, policy_listener) = unbounded();
		let (ttl_worker, ttl_listener) = unbounded();
		let (tiering_worker, tiering_listener) = unbounded();

		let mut handles: WorkerHandles = Vec::new();

		handles.push(register_worker(PolicyWorker::<K, V>::new(
			policy_listener,
			objects.clone(),
			status.clone(),
			overhead_manager.clone(),
			Some(tiering_worker.clone()),
		)?));

		handles.push(register_worker(TtlWorker::<K, V>::new(
			ttl_listener,
			objects.clone(),
			status.clone(),
			overhead_manager.clone(),
		)));

		handles.push(register_worker(TieringWorker::<K, V>::new(
			tiering_listener,
			objects.clone(),
			status.clone(),
			overhead_manager.clone(),
			tiering_manager.clone(),
		)));

		let workers: Arc<Box<[WorkerSender]>> = Arc::new(Box::new([
			policy_worker,
			ttl_worker,
			tiering_worker,
		]));

		let manager = WorkerManager {
			listener,
			workers,
		};

		Ok((manager, handles))
	}

	#[cfg(all(not(all(feature = "key_value_pmem", feature = "enable_tiering_manager"))))]
	pub fn new<K, V>(
		listener: WorkerReceiver,
		objects: &ObjectMapRef<K, V>,
		status: &StatusRef,
		overhead_manager: &OverheadManagerRef,
	) -> Result<(Self, WorkerHandles), CacheError>
	where
		K: 'static + Eq + TypeSize,
		V: 'static + TypeSize + Clone,
	{
		let (policy_worker, policy_listener) = unbounded();
		let (ttl_worker, ttl_listener) = unbounded();

		let mut handles: WorkerHandles = Vec::new();

		handles.push(register_worker(PolicyWorker::<K, V>::new(
			policy_listener,
			objects.clone(),
			status.clone(),
			overhead_manager.clone(),
			None,
		)?));

		handles.push(register_worker(TtlWorker::<K, V>::new(
			ttl_listener,
			objects.clone(),
			status.clone(),
			overhead_manager.clone(),
		)));

		let workers: Arc<Box<[WorkerSender]>> = Arc::new(Box::new([
			policy_worker,
			ttl_worker,
		]));

		let manager = WorkerManager {
			listener,
			workers,
		};

		Ok((manager, handles))
	}

	/// Creates a `WorkerManager` whose policy worker physically migrates
	/// object bytes between tiers whenever `PaperPolicy::LruHybrid`,
	/// `PaperPolicy::LfuHybrid`, `PaperPolicy::TwoQHybrid`, or
	/// `PaperPolicy::FifoHybrid` reports a promotion or demotion. `migrate`
	/// reallocates a value into the target tier's representation (e.g.
	/// `TieredBuffer::new_fast`/`new_slow`). Promotion/demotion/eviction
	/// counters and gauges are recorded directly on the shared `status`
	/// (backing `PaperCache::lru_hybrid_stats`/`lfu_hybrid_stats`/
	/// `two_q_hybrid_stats`/`fifo_hybrid_stats`), so no separate stats
	/// parameter is needed here.
	#[cfg(any(feature = "lru_hybrid_cache", feature = "lfu_hybrid_cache", feature = "two_q_hybrid_cache", feature = "fifo_hybrid_cache", feature = "lru_sized_hybrid_cache", feature = "s3_fifo_hybrid_cache", feature = "two_q_ghost_hybrid_cache", feature = "s3_fifo_ghost_hybrid_cache", feature = "s3_fifo_ghost_lazy_demotion_hybrid_cache", feature = "s3_fifo_ghost_lazy_demotion_fast_admission_hybrid_cache"))]
	pub fn new_with_tier_migration<K, V>(
		listener: WorkerReceiver,
		objects: &ObjectMapRef<K, V>,
		status: &StatusRef,
		overhead_manager: &OverheadManagerRef,
		migrate: Box<dyn Fn(&V, Tier) -> V + Send + Sync>,
	) -> Result<(Self, WorkerHandles), CacheError>
	where
		K: 'static + Eq + TypeSize,
		V: 'static + TypeSize,
	{
		let (policy_worker, policy_listener) = unbounded();
		let (ttl_worker, ttl_listener) = unbounded();

		let mut handles: WorkerHandles = Vec::new();

		handles.push(register_worker(PolicyWorker::<K, V>::new_with_tier_migration(
			policy_listener,
			objects.clone(),
			status.clone(),
			overhead_manager.clone(),
			migrate,
		)?));

		handles.push(register_worker(TtlWorker::<K, V>::new(
			ttl_listener,
			objects.clone(),
			status.clone(),
			overhead_manager.clone(),
		)));

		let workers: Arc<Box<[WorkerSender]>> = Arc::new(Box::new([
			policy_worker,
			ttl_worker,
		]));

		let manager = WorkerManager {
			listener,
			workers,
		};

		Ok((manager, handles))
	}
}

unsafe impl Send for WorkerManager {}
