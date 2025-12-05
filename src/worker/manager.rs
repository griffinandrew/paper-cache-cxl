/*
 * Copyright (c) Kia Shakiba
 *
 * This source code is licensed under the GNU AGPLv3 license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::sync::Arc;
use typesize::TypeSize;
use crossbeam_channel::{unbounded, bounded, Sender};
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
		TieringWorker,
		AccessEvent,
		register_worker,
	},
};

pub struct WorkerManager {
	listener: WorkerReceiver,
	workers: Arc<Box<[WorkerSender]>>,
	pub access_event_sender: Sender<AccessEvent>,
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
	pub fn new<K, V>(
		listener: WorkerReceiver,
		objects: &ObjectMapRef<K, V>,
		status: &StatusRef,
		overhead_manager: &OverheadManagerRef,
	) -> Result<Self, CacheError>
	where
		K: 'static + Eq + TypeSize,
		V: 'static + TypeSize,
	{
		let (policy_worker, policy_listener) = unbounded();
		let (ttl_worker, ttl_listener) = unbounded();
		
		// Create tiering worker with bounded channel
		let (access_event_sender, access_event_receiver) = bounded(10_000);
		let (tiering_worker_sender, tiering_worker_listener) = unbounded();
		
		// For now, use reasonable defaults for water marks
		// High water mark: 80% of max cache size
		// Low water mark: 60% of max cache size
		let max_size = status.max_size();
		let high_water_mark = (max_size as f64 * 0.8) as u64;
		let low_water_mark = (max_size as f64 * 0.6) as u64;

		register_worker(PolicyWorker::<K, V>::new(
			policy_listener,
			objects.clone(),
			status.clone(),
			overhead_manager.clone(),
		)?);

		register_worker(TtlWorker::<K, V>::new(
			ttl_listener,
			objects.clone(),
			status.clone(),
			overhead_manager.clone(),
		));
		
		register_worker(TieringWorker::<K, V>::new(
			access_event_receiver,
			tiering_worker_listener,
			objects.clone(),
			status.clone(),
			overhead_manager.clone(),
			high_water_mark,
			low_water_mark,
		));

		let workers: Arc<Box<[WorkerSender]>> = Arc::new(Box::new([
			policy_worker,
			ttl_worker,
			tiering_worker_sender,
		]));

		let manager = WorkerManager {
			listener,
			workers,
			access_event_sender,
		};

		Ok(manager)
	}
}

unsafe impl Send for WorkerManager {}
