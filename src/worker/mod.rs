/*
 * Copyright (c) Kia Shakiba
 *
 * This source code is licensed under the GNU AGPLv3 license found in the
 * LICENSE file in the root directory of this source tree.
 */

mod manager;
mod policy;
mod ttl;

#[cfg(all(feature = "key_value_pmem", feature = "enable_tiering_manager"))]
mod tiering;

use std::thread;
use crossbeam_channel::{Sender, Receiver};

use crate::{
	CacheSize,
	HashedKey,
	error::CacheError,
	object::{ObjectSize, ExpireTime},
	policy::PaperPolicy,
};

/// Discriminant for tagging worker events in the hybrid cache.
///
/// Used by `HybridWorkerPool` to route events to the correct tier-specific
/// `PolicyWorker` (Small vs Far) while sharing worker threads.
#[cfg(feature = "hybridcache")]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CacheId {
	Small,
	Far,
}

/// Tagged worker event for the hybrid cache.
///
/// The dispatcher reads these and routes `event` to the appropriate worker
/// channel based on `cache_id`.
#[cfg(feature = "hybridcache")]
#[derive(Clone)]
pub struct TaggedWorkerEvent {
	pub cache_id: CacheId,
	pub event: WorkerEvent,
}

pub type WorkerSender = Sender<WorkerEvent>;
pub type WorkerReceiver = Receiver<WorkerEvent>;

#[cfg(feature = "hybridcache")]
pub type HybridWorkerSender = Sender<TaggedWorkerEvent>;

#[derive(Clone)]
pub enum WorkerEvent {
	Get(HashedKey, bool),
	Set(HashedKey, ObjectSize, ExpireTime, Option<(ObjectSize, ExpireTime)>),
	Del(HashedKey, ExpireTime),

	Ttl(HashedKey, ExpireTime, ExpireTime),

	Wipe,

	Resize(CacheSize),
	Policy(PaperPolicy),
}

pub trait Worker
where
	Self: 'static + Send,
{
	fn run(&mut self) -> Result<(), CacheError>;
}

pub fn register_worker(mut worker: impl Worker) {
	thread::spawn(move || worker.run());
}

pub use crate::worker::{
	manager::WorkerManager,
	policy::PolicyWorker,
	ttl::TtlWorker,
};

#[cfg(all(feature = "key_value_pmem", feature = "enable_tiering_manager"))]
pub use crate::worker::tiering::TieringWorker;
