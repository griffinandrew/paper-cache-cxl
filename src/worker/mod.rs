/*
 * Copyright (c) Kia Shakiba
 *
 * This source code is licensed under the GNU AGPLv3 license found in the
 * LICENSE file in the root directory of this source tree.
 */

mod manager;
mod policy;
mod ttl;

use std::thread::{self, JoinHandle};
use crossbeam_channel::{Sender, Receiver};

use crate::{
	CacheSize,
	HashedKey,
	error::CacheError,
	object::{ObjectSize, ExpireTime},
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
	/// `(key, served)`: `served` is `Some(tier)` on a hit -- the tier of the
	/// value the hit copied, read off the snapshot's tag -- and `None` on a
	/// miss.
	///
	/// The tier is what the policy worker's heal needs (backpressure plan S3):
	/// a hit served from the SLOW tier on a key the stack places fast is a
	/// value left in CXL with nothing queued to move it, and the worker queues
	/// its promotion. Only the client knows which copy it read, so it travels
	/// with the event, in the byte the old `bool` took.
	Get(HashedKey, Option<Tier>),
	/// `(key, base_size, dram_resident, expiry, previous (base_size, expiry),
	/// built, mark)`
	///
	/// `dram_resident` is the part of `base_size` that never migrates (see
	/// `OverheadManager::dram_resident_size`). It travels with the event because
	/// only the caller holds the `Object` it is derived from.
	///
	/// `built` is the tier the new value's bytes were allocated in (its
	/// `value().tier()`), which the policy worker's reconcile compares with the
	/// tier the stack places the key in once it has handled the event (S3):
	/// the client chose it from a mirror or a physical read the worker may
	/// already have moved past, and the stack does not otherwise know it.
	///
	/// `mark` is the LANDED count of the key's migration bucket
	/// (`migration_queue::InFlight::mark`), which the client read BEFORE it
	/// published the value. Against the bucket as the worker finds it, it
	/// tells whether any migration of the bucket was in flight, or landed,
	/// after the value was published -- the new-key rule
	/// (`PolicyWorker::handle_set`). 0 from a flat cache, which queues no
	/// migrations; in the byte padding the variant already had.
	///
	/// `placement` (S5): `Structural` when the value was larger than an empty
	/// fast tier and was built slow for that reason (`gate::decide`); every
	/// stack then places the key slow (`PolicyStack::insert_placed`). `Normal`
	/// from a flat cache. Also in the padding: the event stays 40 bytes.
	Set(HashedKey, ObjectSize, ObjectSize, ExpireTime, Option<(ObjectSize, ExpireTime)>, Tier, u32, Placement),
	Del(HashedKey, ExpireTime),

	/// A `TtlWorker` reap: the object at this key expired, and that worker has
	/// already removed it from the object map and decremented `AtomicStatus`
	/// via its own `erase` call. Sent point-to-point to `PolicyWorker` (see
	/// `TtlWorker::notify_expired`), not through `WorkerFanout`.
	///
	/// Kept distinct from `Del` for two reasons: the sender is a background
	/// worker rather than an API call that just succeeded, and the receiver
	/// therefore has to re-check the object map before acting on it -- see
	/// `PolicyWorker::handle_expire`. (Not for the merged store, whose stack
	/// retires only the DEAD slot the reap left, never a live one.)
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

	/// Empty the cache. The POLICY WORKER does it -- the object map, the
	/// stack, the status counters and the tier gauges, in that order
	/// (`PolicyWorker::handle_wipe`) -- and then answers on the sender, for
	/// which `PaperCache::wipe` waits: so a `Set` the worker handled before
	/// the `Wipe` cannot leave a live key its stack no longer tracks, and the
	/// merged store's worker-owned state (its link count, its latch, its
	/// retired slots) has one writer. The TTL worker clears its own state and
	/// ignores the sender. `None` from a test that sends the raw event.
	///
	/// Still `Clone` (each subscriber gets a clone of the sender), and still 40
	/// bytes: a crossbeam `Sender` is 16, with a niche for the `None`.
	Wipe(Option<Sender<()>>),

	Resize(CacheSize),
	/// Runtime-adjusts the fast-tier byte budget for every hybrid design --
	/// `lru_compact_hybrid_cache` (`PaperPolicy::LruCompactHybrid`),
	/// `lfu_compact_hybrid_cache` (`PaperPolicy::LfuCompactHybrid`),
	/// `two_q_compact_hybrid_cache` (`PaperPolicy::TwoQCompactHybrid`),
	/// `fifo_compact_hybrid_cache` (`PaperPolicy::FifoCompactHybrid`) and the
	/// rest. No-op for every other policy stack; see
	/// `PolicyStack::resize_fast_tier`.
	ResizeFastTier(CacheSize),
	/// Runtime-adjusts the LARGE fast segment's byte budget for
	/// `lru_sized_compact_hybrid_cache` (`PaperPolicy::LruSizedCompactHybrid`)
	/// specifically -- the SMALL segment reuses `ResizeFastTier` above. No-op
	/// for every other policy stack; see `PolicyStack::resize_large_fast_tier`.
	ResizeLargeFastTier(CacheSize),
	/// Runtime-adjusts the small/large size-classification threshold for
	/// `lru_sized_compact_hybrid_cache`. No-op for every other policy stack;
	/// see `PolicyStack::resize_size_threshold`.
	ResizeSizeThreshold(CacheSize),

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

	/// S5, `MetadataOverflow::EvictToFit`: the metadata lane's head asks the
	/// policy worker to evict the policy's own victims until a new key fits
	/// under the key ceiling (`PolicyWorker::handle_make_room`); the worker
	/// answers through the gate (`Gate::answer_make_room`). To the policy
	/// worker only.
	#[cfg(feature = "hybrid_cache_common")]
	MakeRoom(u64),

	/// DIAGNOSTIC: the policy worker lands every migration it has decided,
	/// walks the object map, classifies each live value by where its bytes
	/// are against where the stack places it, and replies. Sent only by
	/// `PaperCache::placement_audit`, point to point in effect (only the
	/// policy worker subscribes), and it blocks that worker for the flush and
	/// the walk; handled where it falls in a batch, before that batch's
	/// eviction pass, and exact only at client quiescence -- see
	/// `phys::PlacementAudit`.
	#[cfg(feature = "hybrid_cache_common")]
	Audit(Sender<crate::phys::PlacementAudit>),
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
	pub const SET: EventMask = 1 << 2;
	pub const DEL: EventMask = 1 << 3;
	pub const EXPIRE: EventMask = 1 << 12;
	pub const TTL: EventMask = 1 << 4;
	pub const WIPE: EventMask = 1 << 5;
	pub const RESIZE: EventMask = 1 << 6;
	pub const RESIZE_FAST_TIER: EventMask = 1 << 7;
	pub const RESIZE_LARGE_FAST_TIER: EventMask = 1 << 8;
	pub const RESIZE_SIZE_THRESHOLD: EventMask = 1 << 9;
	pub const SHUTDOWN: EventMask = 1 << 11;
	pub const AUDIT: EventMask = 1 << 13;
	pub const MAKE_ROOM: EventMask = 1 << 14;

	/// `PolicyWorker`. Note the omission: `Ttl` has no arm in the policy loop
	/// (an expiry change doesn't reorder or resize anything the policy stack
	/// tracks).
	pub const POLICY_WORKER: EventMask = Self::GET
		| Self::SET
		| Self::DEL
		| Self::EXPIRE
		| Self::WIPE
		| Self::RESIZE
		| Self::RESIZE_FAST_TIER
		| Self::RESIZE_LARGE_FAST_TIER
		| Self::RESIZE_SIZE_THRESHOLD
		| Self::SHUTDOWN
		| Self::AUDIT
		| Self::MAKE_ROOM;

	/// `TtlWorker` -- expiry bookkeeping only. Reads never change an object's
	/// expiry, so `Get` (the dominant event in a read-heavy workload) is
	/// deliberately absent. `Expire` is absent because this worker *emits* it
	/// (to `PolicyWorker`, point-to-point) rather than consuming it.
	pub const TTL_WORKER: EventMask = Self::SET
		| Self::DEL
		| Self::TTL
		| Self::WIPE
		| Self::SHUTDOWN;
}

impl WorkerEvent {
	/// This event's single [`EventMask`] bit, for testing against a
	/// sub-worker's subscription mask.
	#[must_use]
	pub const fn mask_bit(&self) -> EventMask {
		match self {
			WorkerEvent::Get(..) => Events::GET,
			WorkerEvent::Set(..) => Events::SET,
			WorkerEvent::Del(..) => Events::DEL,
			WorkerEvent::Expire(..) => Events::EXPIRE,
			WorkerEvent::Ttl(..) => Events::TTL,
			WorkerEvent::Wipe(..) => Events::WIPE,
			WorkerEvent::Resize(..) => Events::RESIZE,
			WorkerEvent::ResizeFastTier(..) => Events::RESIZE_FAST_TIER,
			WorkerEvent::ResizeLargeFastTier(..) => Events::RESIZE_LARGE_FAST_TIER,
			WorkerEvent::ResizeSizeThreshold(..) => Events::RESIZE_SIZE_THRESHOLD,
			WorkerEvent::Shutdown => Events::SHUTDOWN,
			#[cfg(feature = "hybrid_cache_common")]
			WorkerEvent::Audit(..) => Events::AUDIT,
			#[cfg(feature = "hybrid_cache_common")]
			WorkerEvent::MakeRoom(..) => Events::MAKE_ROOM,
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
	thread::spawn(move || {
		// Bind this worker's own allocations and stack growth to a node. On by
		// default (node 0); `PAPER_BIND_WORKERS=off` disables, `=1` targets the
		// slow node. Per-thread, so no other thread is affected.
		#[cfg(feature = "numa_jemalloc")]
		crate::numa_alloc::bind_worker_thread_if_configured();

		worker.run()
	})
}

pub use crate::worker::{
	manager::WorkerFanout,
	policy::PolicyWorker,
	ttl::TtlWorker,
};

// Flattens `worker::policy::Tier` (itself a re-export of the private
// `policy_stack` submodule's `Tier`, see `worker/policy/mod.rs`) so `lib.rs`
// can re-export it further as the fully public `PaperCache::tier_of` return
// type, shared by every hybrid design.
//
// Unconditional: the merged store tags every slot with a `Tier` whether or
// not a hybrid feature is on, and `value::TieredValue` -- which is compiled
// in every configuration -- carries one in the low bit of its pointer. There
// is no configuration left that does not need the name.
pub use crate::worker::policy::Tier;

// A set's placement byte (S5), for `gate` and the set path.
pub use crate::worker::policy::Placement;

// The settle target's ratio, for the byte gate's levels (S5 B2).
#[cfg(feature = "hybrid_cache_common")]
pub(crate) use crate::worker::policy::drain_target;

// The capacity-eviction watermarks, for `AtomicStatus`: the one snapshot the
// policy worker's passes and the client's size check read.
pub(crate) use crate::worker::policy::eviction_watermarks::Watermarks;

// The lock every unit test that drives a migration holds, for `crate::phys`'s
// served-hit test, which builds a real demoting cache. See its doc.
#[cfg(all(test, feature = "hybrid_cache_common"))]
pub(crate) use crate::worker::policy::migration_test_lock;

// The migration queue's pending entry counts, summed over every cache in the
// process, for `crate::phys`.
#[cfg(feature = "hybrid_cache_common")]
pub(crate) use crate::worker::policy::migration_queue::pending as pending_migrations;

// One cache's migration and eviction statistics (S8), which `AtomicStatus`
// owns for its policy worker, its migration consumers and `hybrid_stats`.
pub(crate) use crate::worker::policy::migstats::Stats as MigStats;
#[cfg(feature = "hybrid_cache_common")]
pub(crate) use crate::worker::policy::migstats::{NB as MIGSTATS_BUCKETS, read as migstats_read};

// The migration pipeline's per-key-bucket in-flight and landed counts, which
// `AtomicStatus` holds for the client, the worker and the consumers.
#[cfg(feature = "hybrid_cache_common")]
pub(crate) use crate::worker::policy::migration_queue::InFlight;

// The tagged drain entry, for the merged store's migration log; what a `Set`
// did to the map, for its `worker_set`; and the CLOCK hand's budget.
#[cfg(feature = "merged_object_store")]
pub(crate) use crate::worker::policy::{MigrationOrigin, SetEvent, TaggedMigration, clock_hand_budget};
