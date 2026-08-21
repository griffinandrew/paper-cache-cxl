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
/// `WorkerFanout::new`/`new_with_tier_migration`'s return type and each
/// `PaperCache::new`/`with_hasher`'s call site) and joined in `PaperCache`'s
/// `Drop` impl after signalling `WorkerEvent::Shutdown`.
pub type WorkerHandles = Vec<JoinHandle<Result<(), CacheError>>>;

#[derive(Clone)]
pub enum WorkerEvent {
	Get(HashedKey, bool),
	Promote(HashedKey),
	Set(HashedKey, ObjectSize, ExpireTime, Option<(ObjectSize, ExpireTime)>),
	Del(HashedKey, ExpireTime),

	/// A `TtlWorker` reap: the object at this key expired, and that worker has
	/// already removed it from the object map and decremented `AtomicStatus`
	/// via its own `erase` call. Sent point-to-point to `PolicyWorker` (see
	/// `TtlWorker::notify_expired`), not through `WorkerFanout`.
	///
	/// Kept distinct from `Del` for two reasons: the sender is a background
	/// worker rather than an API call that just succeeded, and the receiver
	/// therefore has to re-check the object map before acting on it -- see
	/// `PolicyWorker::handle_expire`.
	///
	/// Before this existed `TtlWorker` reaped silently. `erase` only touches
	/// the object map and the size counters, so a reaped key stayed in the
	/// policy stack's recency/frequency structures *and* kept counting its
	/// bytes toward the hybrid stacks' `fast_used`/`slow_used`. `used_size()`
	/// stayed correct, so global eviction pressure was right, but
	/// `settle_fast_tier` was choosing demotions against an inflated
	/// `fast_used` and so demoted earlier than the fast tier's live contents
	/// warranted. The stack only self-corrected once the phantom key reached
	/// the eviction tail, where `evict_one()` popped it and `erase` answered
	/// `KeyNotFound` -- which on a TTL-dominated workload (every object
	/// expiring rather than being evicted) could be a very long time, or
	/// never.
	Expire(HashedKey),

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
	/// by `PaperCache::drop`, fanned out to every sub-worker by
	/// `WorkerFanout::send` like any other event. Every `Worker::run`
	/// loop must actually check for this and return on receipt; before this
	/// was added, no worker loop had any exit condition at all (they ran
	/// until the process itself terminated), which meant a `PaperCache`
	/// being dropped never actually stopped its background threads before
	/// returning -- those threads could still be mid-allocation when the
	/// process's own exit-time global-allocator teardown ran concurrently
	/// with them, a real, reproduced SIGSEGV inside a jemalloc pool's own
	/// teardown code racing a still-live `PolicyWorker` thread's allocations
	/// call. See `PaperCache`'s `Drop` impl for the send-then-join sequence
	/// this variant exists to support.
	Shutdown,
}

/// Bitmask over [`WorkerEvent`]'s variants. `WorkerFanout` pairs one of
/// these with each sub-worker's sender (see its `new*` constructors) and
/// skips fanning an event out to a worker whose mask doesn't include it.
///
/// Before this existed the fan-out cloned and forwarded *every* event to
/// *every* sub-worker. `TtlWorker`'s run loop has no `Get` arm at all (it
/// falls through to `_ => {}`), so it was receiving -- and being woken by --
/// a copy of every single cache read. In a read-heavy workload `Get` is the
/// overwhelming majority of all events, so that doubled the channel traffic
/// generated by the one event type that matters most for GET latency, purely
/// to hand a second thread work it always discarded.
pub type EventMask = u16;

/// Per-variant bits for [`EventMask`], plus one subscription mask per
/// sub-worker.
///
/// Each worker's mask must list exactly the variants its own `Worker::run`
/// match has a real arm for. Adding an arm to a worker without adding its bit
/// here silently drops that event -- treat the two as a single edit.
pub struct Events;

impl Events {
	pub const GET: EventMask = 1 << 0;
	pub const PROMOTE: EventMask = 1 << 1;
	pub const SET: EventMask = 1 << 2;
	pub const DEL: EventMask = 1 << 3;
	pub const EXPIRE: EventMask = 1 << 12;
	pub const TTL: EventMask = 1 << 4;
	pub const WIPE: EventMask = 1 << 5;
	pub const RESIZE: EventMask = 1 << 6;
	pub const RESIZE_FAST_TIER: EventMask = 1 << 7;
	pub const RESIZE_LARGE_FAST_TIER: EventMask = 1 << 8;
	pub const RESIZE_SIZE_THRESHOLD: EventMask = 1 << 9;
	pub const POLICY: EventMask = 1 << 10;
	pub const SHUTDOWN: EventMask = 1 << 11;

	/// `PolicyWorker`. Note the two omissions: `Promote` is delivered to the
	/// tiering worker directly through `PolicyWorker`'s own `promotion_tx`,
	/// never routed back through the manager, and `Ttl` has no arm in the
	/// policy loop (an expiry change doesn't reorder or resize anything the
	/// policy stack tracks).
	pub const POLICY_WORKER: EventMask = Self::GET
		| Self::SET
		| Self::DEL
		| Self::EXPIRE
		| Self::WIPE
		| Self::RESIZE
		| Self::RESIZE_FAST_TIER
		| Self::RESIZE_LARGE_FAST_TIER
		| Self::RESIZE_SIZE_THRESHOLD
		| Self::POLICY
		| Self::SHUTDOWN;

	/// `TtlWorker` -- expiry bookkeeping only. Reads never change an object's
	/// expiry, so `Get` (the dominant event in a read-heavy workload) is
	/// deliberately absent. `Expire` is absent because this worker *emits* it
	/// (to `PolicyWorker`, point-to-point) rather than consuming it.
	pub const TTL_WORKER: EventMask = Self::SET
		| Self::DEL
		| Self::TTL
		| Self::WIPE
		| Self::SHUTDOWN;

	/// `TieringWorker` (`enable_tiering_manager`). Unlike the policy worker
	/// this one *does* consume `Get` -- hotness counting is its whole job --
	/// and `Promote`, which `PolicyWorker` forwards to it on a ghost hit.
	#[cfg(all(feature = "key_value_pmem", feature = "enable_tiering_manager"))]
	pub const TIERING_WORKER: EventMask = Self::GET
		| Self::PROMOTE
		| Self::SET
		| Self::DEL
		| Self::WIPE
		| Self::RESIZE
		| Self::SHUTDOWN;
}

impl WorkerEvent {
	/// This event's single [`EventMask`] bit, for testing against a
	/// sub-worker's subscription mask.
	#[must_use]
	pub const fn mask_bit(&self) -> EventMask {
		match self {
			WorkerEvent::Get(..) => Events::GET,
			WorkerEvent::Promote(..) => Events::PROMOTE,
			WorkerEvent::Set(..) => Events::SET,
			WorkerEvent::Del(..) => Events::DEL,
			WorkerEvent::Expire(..) => Events::EXPIRE,
			WorkerEvent::Ttl(..) => Events::TTL,
			WorkerEvent::Wipe => Events::WIPE,
			WorkerEvent::Resize(..) => Events::RESIZE,
			WorkerEvent::ResizeFastTier(..) => Events::RESIZE_FAST_TIER,
			WorkerEvent::ResizeLargeFastTier(..) => Events::RESIZE_LARGE_FAST_TIER,
			WorkerEvent::ResizeSizeThreshold(..) => Events::RESIZE_SIZE_THRESHOLD,
			WorkerEvent::Policy(..) => Events::POLICY,
			WorkerEvent::Shutdown => Events::SHUTDOWN,
		}
	}
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
	manager::WorkerFanout,
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
#[cfg(feature = "hybrid_cache_common")]
pub use crate::worker::policy::Tier;
