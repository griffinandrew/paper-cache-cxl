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

use std::thread::{self, JoinHandle};
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

/// Join handles for every background thread transitively spawned on behalf
/// of a single `PaperCache` instance -- collected at construction time (see
/// `WorkerManager::new`/`new_with_tier_migration`'s return type and each
/// `PaperCache::new`/`with_hasher`'s call site) and joined in `PaperCache`'s
/// `Drop` impl after signalling `WorkerEvent::Shutdown`.
pub type WorkerHandles = Vec<JoinHandle<Result<(), CacheError>>>;

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
	/// (`PaperPolicy::LruHybrid`) / `lfu_hybrid_cache` (`PaperPolicy::LfuHybrid`)
	/// / `two_q_hybrid_cache` (`PaperPolicy::TwoQHybrid`) / `fifo_hybrid_cache`
	/// (`PaperPolicy::FifoHybrid`). No-op for every other policy stack; see
	/// `PolicyStack::resize_fast_tier`.
	ResizeFastTier(CacheSize),
	/// Runtime-adjusts the LARGE fast segment's byte budget for
	/// `lru_sized_hybrid_cache` (`PaperPolicy::LruSizedHybrid`) specifically
	/// -- the SMALL segment reuses `ResizeFastTier` above. No-op for every
	/// other policy stack; see `PolicyStack::resize_large_fast_tier`.
	ResizeLargeFastTier(CacheSize),
	/// Runtime-adjusts the small/large size-classification threshold for
	/// `lru_sized_hybrid_cache`. No-op for every other policy stack; see
	/// `PolicyStack::resize_size_threshold`.
	ResizeSizeThreshold(CacheSize),
	Policy(PaperPolicy),

	/// Tells a worker to stop its event loop and return. Sent exactly once,
	/// by `PaperCache::drop`, and cascaded onward by `WorkerManager` to its
	/// own sub-workers -- see that type's `run` impl. Every `Worker::run`
	/// loop must actually check for this and return on receipt; before this
	/// was added, no worker loop had any exit condition at all (they ran
	/// until the process itself terminated), which meant a `PaperCache`
	/// being dropped never actually stopped its background threads before
	/// returning -- those threads could still be mid-allocation when the
	/// process's own exit-time global-allocator teardown ran concurrently
	/// with them, a real, reproduced SIGSEGV inside a UMF/TBB pool's own
	/// teardown code racing a still-live `PolicyWorker` thread's `tbb_malloc`
	/// call. See `PaperCache`'s `Drop` impl for the send-then-join sequence
	/// this variant exists to support.
	Shutdown,
}

pub trait Worker
where
	Self: 'static + Send,
{
	fn run(&mut self) -> Result<(), CacheError>;
}

pub fn register_worker(mut worker: impl Worker) -> JoinHandle<Result<(), CacheError>> {
	thread::spawn(move || worker.run())
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
// `lib.rs` can re-export it further as a fully public `PaperCache::tier_of`/
// `lru_hybrid_cache`/`lfu_hybrid_cache`/`two_q_hybrid_cache`/
// `fifo_hybrid_cache`/`lru_sized_hybrid_cache` return type.
#[cfg(any(feature = "lru_hybrid_cache", feature = "lfu_hybrid_cache", feature = "two_q_hybrid_cache", feature = "fifo_hybrid_cache", feature = "lru_sized_hybrid_cache", feature = "s3_fifo_hybrid_cache", feature = "two_q_ghost_hybrid_cache", feature = "s3_fifo_ghost_hybrid_cache", feature = "s3_fifo_ghost_lazy_demotion_hybrid_cache", feature = "s3_fifo_ghost_lazy_demotion_fast_admission_hybrid_cache", feature = "s3_fifo_ghost_lazy_demotion_fast_admission_midpoint_hybrid_cache", feature = "s3_fifo_lazy_demotion_fast_admission_midpoint_reprieve_hybrid_cache", feature = "s3_fifo_lazy_demotion_fast_admission_reprieve_hybrid_cache", feature = "s3_fifo_lazy_demotion_fast_admission_split_slow_reprieve_hybrid_cache"))]
pub use crate::worker::policy::Tier;
