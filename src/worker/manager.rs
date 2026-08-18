/*
 * Copyright (c) Kia Shakiba
 *
 * This source code is licensed under the GNU AGPLv3 license found in the
 * LICENSE file in the root directory of this source tree.
 */

use typesize::TypeSize;
use crossbeam_channel::unbounded;
use log::error;

use crate::{
	ObjectMapRef,
	StatusRef,
	OverheadManagerRef,
	error::CacheError,
	worker::{
		WorkerEvent,
		WorkerSender,
		WorkerHandles,
		EventMask,
		Events,
		PolicyWorker,
		TtlWorker,
		register_worker,
	},
};

#[cfg(all(feature = "key_value_pmem", feature = "enable_tiering_manager"))]
use std::sync::Arc;

#[cfg(all(feature = "key_value_pmem", feature = "enable_tiering_manager"))]
use crate::{
	tiering::TieringManager,
	worker::TieringWorker,
};

#[cfg(any(feature = "lru_hybrid_cache", feature = "lfu_hybrid_cache", feature = "two_q_hybrid_cache", feature = "two_q_fast_admission_hybrid_cache", feature = "two_q_fast_admission_reprieve_hybrid_cache", feature = "fifo_hybrid_cache", feature = "lru_sized_hybrid_cache", feature = "s3_fifo_hybrid_cache", feature = "two_q_ghost_hybrid_cache", feature = "s3_fifo_ghost_hybrid_cache", feature = "s3_fifo_ghost_lazy_demotion_hybrid_cache", feature = "s3_fifo_ghost_lazy_demotion_fast_admission_hybrid_cache", feature = "s3_fifo_ghost_lazy_demotion_fast_admission_midpoint_hybrid_cache", feature = "s3_fifo_lazy_demotion_fast_admission_midpoint_reprieve_hybrid_cache", feature = "s3_fifo_lazy_demotion_fast_admission_reprieve_hybrid_cache", feature = "s3_fifo_lazy_demotion_fast_admission_split_slow_reprieve_hybrid_cache", feature = "s3_fifo_lazy_demotion_reprieve_hybrid_cache", feature = "lru_lfu_hybrid_cache"))]
use crate::worker::policy::Tier;

/// Routes each `WorkerEvent` to the background workers that consume it.
///
/// This used to be a background thread of its own: `PaperCache` pushed every
/// event into *its* channel, and it looped popping events back out, cloning
/// each one, and pushing the copies into each sub-worker's channel. That cost
/// every cache operation two channel round trips instead of one, and the pop
/// side had no good options -- blocking `recv()` parked on a real futex
/// between nearly every event (~30%+ of process cycles in `futex_wait`/
/// `futex_wake`/`sched_yield`, measured under `perf` against
/// paper-benchmark-cxl), so it was changed to a non-blocking spin, which
/// traded that for a thread hammering the head/tail cache lines of the very
/// channel the API threads were producing into, plus a permanently occupied
/// core.
///
/// Fanning out inline on the calling thread removes the whole dilemma: no
/// futex, no spin, no extra core, and one channel push per interested worker
/// instead of one push, one pop, and one push per interested worker. With
/// `Events`' per-worker masks a `Get` -- the dominant event in a read-heavy
/// workload -- now reaches exactly one channel (`PolicyWorker`'s), so a cache
/// read costs a single uncontended push.
///
/// Measured on an 8-core box, 20k x 4 KiB working set, 2M reads, median of 5
/// (gets/sec, before -> after):
///
/// | client threads | before | after |
/// |---|---|---|
/// | 1 | 684,195 | 685,608 |
/// | 4 | 2,482,037 | 2,456,834 |
/// | 8 | 3,184,438 | 4,223,833 |
///
/// The shape of that is the point: the win arrives exactly when the machine
/// is saturated, because that is when a dedicated fan-out thread stops being
/// free work on a spare core and starts being a core taken away from the
/// request path. Below saturation the extra core absorbed the fan-out, so
/// doing it inline is a wash to marginally negative (the 4-thread column is
/// -1%, at the edge of this harness's run-to-run spread).
///
/// One property is deliberately given up: events no longer pass through a
/// single serialization point, so two *different* API threads' events can
/// reach two different sub-workers in opposite relative orders. Nothing
/// depends on that. `PolicyWorker` and `TtlWorker` consume disjoint concerns
/// (eviction ordering vs. expiry bookkeeping), each individual channel is
/// still FIFO so a single thread's own events stay ordered relative to each
/// other, and two threads racing on the same key were already unordered --
/// a thread can be preempted between its object-map write and its broadcast,
/// so the old single-channel funnel never implied the events arrived in the
/// order the map was actually mutated.
pub struct WorkerFanout {
	/// Each sub-worker's sender paired with the set of `WorkerEvent` variants
	/// its own `run` loop actually has an arm for -- see `Events`' per-worker
	/// subscription constants. Anything outside a worker's mask is dropped
	/// here rather than cloned, sent, and discarded on the far side.
	workers: Box<[(WorkerSender, EventMask)]>,
}

impl WorkerFanout {
	/// Delivers `event` to every sub-worker subscribed to its variant.
	///
	/// Runs on the caller's thread -- this is the hot path for `get`/`set`,
	/// so the per-worker mask test exists to keep an uninterested worker from
	/// costing a clone and a channel push.
	pub fn send(&self, event: WorkerEvent) -> Result<(), CacheError> {
		let bit = event.mask_bit();

		// Hand the event itself to the last subscriber and clone only for the
		// ones before it, so the common case -- a `Get`, which after the mask
		// filter has exactly one subscriber -- costs a plain move, exactly as
		// it did when this was a single unconditional send into the manager's
		// channel. Deferring by one keeps that without needing to know the
		// subscriber count up front.
		let mut pending: Option<&WorkerSender> = None;

		for (worker, mask) in self.workers.iter() {
			if bit & mask == 0 {
				continue;
			}

			if let Some(previous) = pending.replace(worker) {
				Self::deliver(previous, event.clone())?;
			}
		}

		match pending {
			Some(last) => Self::deliver(last, event),
			None => Ok(()),
		}
	}

	fn deliver(worker: &WorkerSender, event: WorkerEvent) -> Result<(), CacheError> {
		if let Err(err) = worker.try_send(event) {
			error!("Could not send event to worker: {err:?}");
			return Err(CacheError::Internal);
		}

		Ok(())
	}

	#[cfg(all(feature = "key_value_pmem", feature = "enable_tiering_manager"))]
	pub fn new<K, V>(
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
			policy_worker.clone(),
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

		let workers: Box<[(WorkerSender, EventMask)]> = Box::new([
			(policy_worker, Events::POLICY_WORKER),
			(ttl_worker, Events::TTL_WORKER),
			(tiering_worker, Events::TIERING_WORKER),
		]);

		Ok((WorkerFanout { workers }, handles))
	}

	#[cfg(all(not(all(feature = "key_value_pmem", feature = "enable_tiering_manager"))))]
	pub fn new<K, V>(
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
			policy_worker.clone(),
			objects.clone(),
			status.clone(),
			overhead_manager.clone(),
		)));

		let workers: Box<[(WorkerSender, EventMask)]> = Box::new([
			(policy_worker, Events::POLICY_WORKER),
			(ttl_worker, Events::TTL_WORKER),
		]);

		Ok((WorkerFanout { workers }, handles))
	}

	/// Creates a `WorkerFanout` whose policy worker physically migrates
	/// object bytes between tiers whenever `PaperPolicy::LruHybrid`,
	/// `PaperPolicy::LfuHybrid`, `PaperPolicy::TwoQHybrid`, or
	/// `PaperPolicy::FifoHybrid` reports a promotion or demotion. `migrate`
	/// reallocates a value into the target tier's representation (e.g.
	/// `TieredBuffer::new_fast`/`new_slow`). Promotion/demotion/eviction
	/// counters and gauges are recorded directly on the shared `status`
	/// (backing `PaperCache::lru_hybrid_stats`/`lfu_hybrid_stats`/
	/// `two_q_hybrid_stats`/`fifo_hybrid_stats`), so no separate stats
	/// parameter is needed here.
	#[cfg(any(feature = "lru_hybrid_cache", feature = "lfu_hybrid_cache", feature = "two_q_hybrid_cache", feature = "two_q_fast_admission_hybrid_cache", feature = "two_q_fast_admission_reprieve_hybrid_cache", feature = "fifo_hybrid_cache", feature = "lru_sized_hybrid_cache", feature = "s3_fifo_hybrid_cache", feature = "two_q_ghost_hybrid_cache", feature = "s3_fifo_ghost_hybrid_cache", feature = "s3_fifo_ghost_lazy_demotion_hybrid_cache", feature = "s3_fifo_ghost_lazy_demotion_fast_admission_hybrid_cache", feature = "s3_fifo_ghost_lazy_demotion_fast_admission_midpoint_hybrid_cache", feature = "s3_fifo_lazy_demotion_fast_admission_midpoint_reprieve_hybrid_cache", feature = "s3_fifo_lazy_demotion_fast_admission_reprieve_hybrid_cache", feature = "s3_fifo_lazy_demotion_fast_admission_split_slow_reprieve_hybrid_cache", feature = "s3_fifo_lazy_demotion_reprieve_hybrid_cache", feature = "lru_lfu_hybrid_cache"))]
	pub fn new_with_tier_migration<K, V>(
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
			policy_worker.clone(),
			objects.clone(),
			status.clone(),
			overhead_manager.clone(),
		)));

		let workers: Box<[(WorkerSender, EventMask)]> = Box::new([
			(policy_worker, Events::POLICY_WORKER),
			(ttl_worker, Events::TTL_WORKER),
		]);

		Ok((WorkerFanout { workers }, handles))
	}
}

