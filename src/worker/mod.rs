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

pub type WorkerSender = Sender<WorkerEvent>;
pub type WorkerReceiver = Receiver<WorkerEvent>;

#[derive(Clone)]
pub enum WorkerEvent {
	Get(HashedKey, bool),
	Promote(HashedKey),
	Set(HashedKey, ObjectSize, ExpireTime, Option<(ObjectSize, ExpireTime)>),
	Del(HashedKey, ExpireTime),

	Ttl(HashedKey, ExpireTime, ExpireTime),

	Wipe,

	Resize(CacheSize),
	/// Runtime-adjusts the fast-tier byte budget for `lru_hybrid_cache`
	/// (`PaperPolicy::LruHybrid`). No-op for every other policy stack; see
	/// `PolicyStack::resize_fast_tier`.
	ResizeFastTier(CacheSize),
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

// Flattens `worker::policy::Tier` (itself a `pub(crate)` re-export of the
// private `policy_stack` submodule's `Tier`, see `worker/policy/mod.rs`) so
// `lib.rs` can re-export it further as a fully public
// `PaperCache::tier_of`/`lru_hybrid_cache` return type.
#[cfg(feature = "lru_hybrid_cache")]
pub use crate::worker::policy::Tier;
