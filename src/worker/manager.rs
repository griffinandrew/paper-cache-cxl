/*
 * Copyright (c) Kia Shakiba
 *
 * This source code is licensed under the GNU AGPLv3 license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::sync::Arc;
use typesize::TypeSize;
use crossbeam_channel::unbounded;
use log::error;

use crate::{
	ObjectMapRef,
	StatusRef,
	OverheadManagerRef,
	error::CacheError,
	worker::{
		Worker,
		WorkerSender,
		WorkerReceiver,
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

#[cfg(any(feature = "lru_hybrid_cache", feature = "lfu_hybrid_cache", feature = "two_q_hybrid_cache"))]
use crate::worker::policy::Tier;

pub struct WorkerManager {
	listener: WorkerReceiver,
	workers: Arc<Box<[WorkerSender]>>,
}

impl Worker for WorkerManager {
	fn run(&mut self) -> Result<(), CacheError> {
		loop {
			let Ok(event) = self.listener.recv() else {
				return Ok(());
			};

			for worker in self.workers.iter() {
				if let Err(err) = worker.try_send(event.clone()) {
					error!("Could not send event to worker: {err:?}");
					return Err(CacheError::Internal);
				}
			}
		}
	}
}

impl WorkerManager {
	/// `sets_dram` variant: PolicyWorker receives a TieringManager reference so
	/// it can intercept evictions of DramOnly keys and force-persist them first.
	/// No TieringWorker is spawned because tiering is driven by the backfill path.
	#[cfg(all(feature = "key_value_pmem", feature = "enable_tiering_manager", feature = "sets_dram"))]
	pub fn new<K, V>(
		listener: WorkerReceiver,
		objects: &ObjectMapRef<K, V>,
		status: &StatusRef,
		overhead_manager: &OverheadManagerRef,
		tiering_manager: &Arc<TieringManager<K, V>>,
	) -> Result<Self, CacheError>
	where
		K: 'static + Eq + TypeSize + Clone,
		V: 'static + TypeSize + Clone + AsRef<[u8]>,
	{
		let (policy_worker, policy_listener) = unbounded();
		let (ttl_worker, ttl_listener) = unbounded();

		register_worker(PolicyWorker::<K, V>::new_with_tiering(
			policy_listener,
			objects.clone(),
			status.clone(),
			overhead_manager.clone(),
			None,
			tiering_manager.clone(),
		)?);

		register_worker(TtlWorker::<K, V>::new(
			ttl_listener,
			objects.clone(),
			status.clone(),
			overhead_manager.clone(),
		));

		let workers: Arc<Box<[WorkerSender]>> = Arc::new(Box::new([
			policy_worker,
			ttl_worker,
		]));

		let manager = WorkerManager {
			listener,
			workers,
		};

		Ok(manager)
	}

	#[cfg(all(feature = "key_value_pmem", feature = "enable_tiering_manager", not(feature = "sets_dram")))]
	pub fn new<K, V>(
		listener: WorkerReceiver,
		objects: &ObjectMapRef<K, V>,
		status: &StatusRef,
		overhead_manager: &OverheadManagerRef,
		tiering_manager: &Arc<TieringManager<K, V>>,
	) -> Result<Self, CacheError>
	where
		K: 'static + Eq + TypeSize + Clone,
		V: 'static + TypeSize + Clone + AsRef<[u8]>,
	{
		let (policy_worker, policy_listener) = unbounded();
		let (ttl_worker, ttl_listener) = unbounded();
		let (tiering_worker, tiering_listener) = unbounded();

		register_worker(PolicyWorker::<K, V>::new(
			policy_listener,
			objects.clone(),
			status.clone(),
			overhead_manager.clone(),
			Some(tiering_worker.clone()),
		)?);

		register_worker(TtlWorker::<K, V>::new(
			ttl_listener,
			objects.clone(),
			status.clone(),
			overhead_manager.clone(),
		));

		register_worker(TieringWorker::<K, V>::new(
			tiering_listener,
			objects.clone(),
			status.clone(),
			overhead_manager.clone(),
			tiering_manager.clone(),
		));

		let workers: Arc<Box<[WorkerSender]>> = Arc::new(Box::new([
			policy_worker,
			ttl_worker,
			tiering_worker,
		]));

		let manager = WorkerManager {
			listener,
			workers,
		};

		Ok(manager)
	}

	#[cfg(all(not(all(feature = "key_value_pmem", feature = "enable_tiering_manager"))))]
	pub fn new<K, V>(
		listener: WorkerReceiver,
		objects: &ObjectMapRef<K, V>,
		status: &StatusRef,
		overhead_manager: &OverheadManagerRef,
	) -> Result<Self, CacheError>
	where
		K: 'static + Eq + TypeSize,
		V: 'static + TypeSize + Clone,
	{
		let (policy_worker, policy_listener) = unbounded();
		let (ttl_worker, ttl_listener) = unbounded();

		register_worker(PolicyWorker::<K, V>::new(
			policy_listener,
			objects.clone(),
			status.clone(),
			overhead_manager.clone(),
			None,
		)?);

		register_worker(TtlWorker::<K, V>::new(
			ttl_listener,
			objects.clone(),
			status.clone(),
			overhead_manager.clone(),
		));

		let workers: Arc<Box<[WorkerSender]>> = Arc::new(Box::new([
			policy_worker,
			ttl_worker,
		]));

		let manager = WorkerManager {
			listener,
			workers,
		};

		Ok(manager)
	}

	/// Creates a `WorkerManager` where the policy worker fires `eviction_callback`
	/// for each evicted item.  Used by `hybridcache` to propagate evictions from
	/// the small DRAM tier to the far PMEM tier.
	#[cfg(feature = "hybridcache")]
	pub fn new_with_eviction_callback<K, V>(
		listener: WorkerReceiver,
		objects: &ObjectMapRef<K, V>,
		status: &StatusRef,
		overhead_manager: &OverheadManagerRef,
		eviction_callback: Box<dyn for<'a> Fn(crate::HashedKey, std::sync::Arc<V>, &'a K) + Send + Sync>,
		promotion_tx: Option<WorkerSender>,
	) -> Result<Self, CacheError>
	where
		K: 'static + Eq + TypeSize + Clone,
		V: 'static + TypeSize + Clone,
	{
		let (policy_worker, policy_listener) = unbounded();
		let (ttl_worker, ttl_listener) = unbounded();

		register_worker(PolicyWorker::<K, V>::new_with_eviction_callback(
			policy_listener,
			objects.clone(),
			status.clone(),
			overhead_manager.clone(),
			promotion_tx,
			eviction_callback,
		)?);

		register_worker(TtlWorker::<K, V>::new(
			ttl_listener,
			objects.clone(),
			status.clone(),
			overhead_manager.clone(),
		));

		let workers: Arc<Box<[WorkerSender]>> = Arc::new(Box::new([
			policy_worker,
			ttl_worker,
		]));

		let manager = WorkerManager {
			listener,
			workers,
		};

		Ok(manager)
	}

	/// Creates a `WorkerManager` whose policy worker physically migrates
	/// object bytes between tiers whenever `PaperPolicy::LruHybrid`,
	/// `PaperPolicy::LfuHybrid`, or `PaperPolicy::TwoQHybrid` reports a
	/// promotion or demotion. `migrate` reallocates a value into the target
	/// tier's representation (e.g. `TieredBuffer::new_fast`/`new_slow`).
	/// Promotion/demotion/eviction counters and gauges are recorded directly
	/// on the shared `status` (backing `PaperCache::lru_hybrid_stats`/
	/// `lfu_hybrid_stats`/`two_q_hybrid_stats`), so no separate stats
	/// parameter is needed here.
	#[cfg(any(feature = "lru_hybrid_cache", feature = "lfu_hybrid_cache", feature = "two_q_hybrid_cache"))]
	pub fn new_with_tier_migration<K, V>(
		listener: WorkerReceiver,
		objects: &ObjectMapRef<K, V>,
		status: &StatusRef,
		overhead_manager: &OverheadManagerRef,
		migrate: Box<dyn Fn(&V, Tier) -> V + Send + Sync>,
	) -> Result<Self, CacheError>
	where
		K: 'static + Eq + TypeSize,
		V: 'static + TypeSize,
	{
		let (policy_worker, policy_listener) = unbounded();
		let (ttl_worker, ttl_listener) = unbounded();

		register_worker(PolicyWorker::<K, V>::new_with_tier_migration(
			policy_listener,
			objects.clone(),
			status.clone(),
			overhead_manager.clone(),
			migrate,
		)?);

		register_worker(TtlWorker::<K, V>::new(
			ttl_listener,
			objects.clone(),
			status.clone(),
			overhead_manager.clone(),
		));

		let workers: Arc<Box<[WorkerSender]>> = Arc::new(Box::new([
			policy_worker,
			ttl_worker,
		]));

		let manager = WorkerManager {
			listener,
			workers,
		};

		Ok(manager)
	}
}

unsafe impl Send for WorkerManager {}
