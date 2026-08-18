/*
 * Copyright (c) Kia Shakiba
 *
 * This source code is licensed under the GNU AGPLv3 license found in the
 * LICENSE file in the root directory of this source tree.
 */

mod policy_stack;
mod mini_stack;
mod event;
mod trace;

use std::{
	thread,
	sync::Arc,
	time::{Instant, Duration},
	io::{Seek, SeekFrom},
	collections::VecDeque,
};

use typesize::TypeSize;
use parking_lot::RwLock;
use crossbeam_channel::{Sender, Receiver, unbounded};
use log::{info, warn, error};
use kwik::fmt;

// Gated exactly as the `object_store` module itself is (see `lib.rs`) rather
// than on the hybrid features that were its original users, because
// `handle_expire` needs it on any build that has it available. A build that
// selects no storage feature at all (e.g. bare `eviction_stacks_pmem`) has no
// `object_store` module to import; `PolicyWorker::object_exists` carries a
// second body for that case.
#[cfg(any(feature = "all_dram", feature = "key_value_pmem", feature = "global_hashtable_pmem", feature = "hashbrown_dram"))]
use crate::object_store::ObjectStore;

use crate::{
	CacheSize,
	HashedKey,
	ObjectMapRef,
	StatusRef,
	OverheadManagerRef,
	EraseKey,
	erase,
	error::CacheError,
	policy::PaperPolicy,
	object::ObjectSize,
	worker::{
		Worker,
		WorkerEvent,
		WorkerSender,
		WorkerReceiver,
		register_worker,
		policy::{
			mini_stack::MiniStackManager,
			event::{StackEvent, TraceEvent},
			trace::{TraceWorker, TraceFragment},
			policy_stack::{AccessOutcome, PolicyStack, init_policy_stack},
		},
	},
};

// Re-exported (fully `pub`, not `pub(crate)`) so sibling modules (e.g.
// `worker::manager`) can name `Tier` without reaching into the private
// `policy_stack` submodule directly, *and* so it can flow all the way out
// to `PaperCache::tier_of`'s public return type via `worker::Tier` /
// `crate::Tier` (see `worker/mod.rs` and `lib.rs`).
#[cfg(any(feature = "lru_hybrid_cache", feature = "lfu_hybrid_cache", feature = "two_q_hybrid_cache", feature = "two_q_fast_admission_hybrid_cache", feature = "two_q_fast_admission_reprieve_hybrid_cache", feature = "fifo_hybrid_cache", feature = "lru_sized_hybrid_cache", feature = "s3_fifo_hybrid_cache", feature = "two_q_ghost_hybrid_cache", feature = "s3_fifo_ghost_hybrid_cache", feature = "s3_fifo_ghost_lazy_demotion_hybrid_cache", feature = "s3_fifo_ghost_lazy_demotion_fast_admission_hybrid_cache", feature = "s3_fifo_ghost_lazy_demotion_fast_admission_midpoint_hybrid_cache", feature = "s3_fifo_lazy_demotion_fast_admission_midpoint_reprieve_hybrid_cache", feature = "s3_fifo_lazy_demotion_fast_admission_reprieve_hybrid_cache", feature = "s3_fifo_lazy_demotion_fast_admission_split_slow_reprieve_hybrid_cache", feature = "s3_fifo_lazy_demotion_reprieve_hybrid_cache"))]
pub use policy_stack::Tier;

// the polling value must be a power of 2
const RECONSTRUCT_POLICY_POLLING: usize = 1_048_576;

const AUTO_POLICY_DURATION: Duration = Duration::from_secs(3_600);
const SET_RECENCY_DURATION: Duration = Duration::from_secs(5);
const SHORT_POLLING_DURATION: Duration = Duration::from_millis(1);
const LONG_POLLING_DURATION: Duration = Duration::from_secs(1);

pub struct PolicyWorker<K, V> {
	listener: Receiver<WorkerEvent>,

	objects: ObjectMapRef<K, V>,
	status: StatusRef,
	overhead_manager: OverheadManagerRef,

	policy_stack: Option<Box<dyn PolicyStack>>,

	trace_fragments: Arc<RwLock<VecDeque<TraceFragment>>>,
	/// Sender into `TraceWorker`, or `None` when access tracing is switched
	/// off entirely (see `trace_is_useful`).
	///
	/// The trace exists for exactly one purpose: replaying past accesses to
	/// rebuild a *different* policy's stack after a live policy switch (see
	/// `handle_policy` -> `reconstruct_policy_stack`). A cache configured with
	/// a single policy -- which is every hybrid cache, and any `paper-server`
	/// instance pinned to one eviction policy -- can never perform that
	/// switch, so every byte it records is written and never read.
	///
	/// Leaving it on wasn't free: each cache read produced a second channel
	/// send from this thread into `TraceWorker`, which then copied a 13-byte
	/// chunk per hit into an on-disk temp file and flushed it once a second.
	/// At this crate's real request rates that is tens of MB/s of pure write
	/// amplification, plus a whole extra thread competing for cores with the
	/// GET path, in service of a reconstruction that can never be requested.
	trace_worker: Option<Sender<StackEvent>>,
	/// `TraceWorker`'s own thread handle -- `None` after `WorkerEvent::
	/// Shutdown` has already been handled once (joined and taken; see the
	/// `run` loop's `Shutdown` arm), `Some` otherwise. Owned here (not by
	/// `WorkerManager`/`PaperCache` directly) because `TraceWorker` is
	/// itself spawned from inside `PolicyWorker::new`, not from the
	/// `WorkerManager::new*` call sites those two collect handles from --
	/// joining it here, before this worker's own `run` returns, means
	/// `PaperCache`'s top-level `WorkerHandles` list doesn't need to know
	/// about this nested worker at all.
	trace_handle: Option<thread::JoinHandle<Result<(), CacheError>>>,

	mini_stack_manager: MiniStackManager,
	mini_index: Option<usize>,
	current_policy: Arc<RwLock<PaperPolicy>>,

	last_auto_policy_time: Option<Instant>,
	last_set_time: Option<Instant>,

	promotion_tx: Option<WorkerSender>,

	/// Reallocates a value into the target tier's representation (e.g.
	/// `TieredBuffer::new_fast`/`new_slow`). Used by `lru_hybrid_cache` /
	/// `lfu_hybrid_cache` to physically move an object's bytes when
	/// `LruHybridStack`/`LfuHybridStack` reports a tier migration; `None`
	/// for every other policy/value type. Promotion/demotion/eviction
	/// counters and gauges are recorded directly on the shared `status`
	/// (see `apply_tier_migrations`), not a separate field.
	#[cfg(any(feature = "lru_hybrid_cache", feature = "lfu_hybrid_cache", feature = "two_q_hybrid_cache", feature = "two_q_fast_admission_hybrid_cache", feature = "two_q_fast_admission_reprieve_hybrid_cache", feature = "fifo_hybrid_cache", feature = "lru_sized_hybrid_cache", feature = "s3_fifo_hybrid_cache", feature = "two_q_ghost_hybrid_cache", feature = "s3_fifo_ghost_hybrid_cache", feature = "s3_fifo_ghost_lazy_demotion_hybrid_cache", feature = "s3_fifo_ghost_lazy_demotion_fast_admission_hybrid_cache", feature = "s3_fifo_ghost_lazy_demotion_fast_admission_midpoint_hybrid_cache", feature = "s3_fifo_lazy_demotion_fast_admission_midpoint_reprieve_hybrid_cache", feature = "s3_fifo_lazy_demotion_fast_admission_reprieve_hybrid_cache", feature = "s3_fifo_lazy_demotion_fast_admission_split_slow_reprieve_hybrid_cache", feature = "s3_fifo_lazy_demotion_reprieve_hybrid_cache"))]
	tier_migration_fn: Option<Box<dyn Fn(&V, Tier) -> V + Send + Sync>>,
}

impl<K, V> Worker for PolicyWorker<K, V>
where
	Self: 'static + Send,
	K: Eq + TypeSize,
	V: TypeSize,
{
	fn run(&mut self) -> Result<(), CacheError> {
		let (
			policy_reconstruct_tx,
			policy_reconstruct_rx,
		) = unbounded::<Box<dyn PolicyStack>>();

		let policy_reconstruct_tx = Arc::new(policy_reconstruct_tx);
		let mut buffered_events = Vec::<StackEvent>::new();

		// Drained into and reused across iterations rather than re-collected
		// into a fresh `Vec` each pass. The collect is needed at all only
		// because `try_iter()` borrows `self.listener` while the loop body
		// needs `&mut self`; keeping one buffer alive means a steady-state
		// poll allocates nothing, instead of allocating (and, under bursty
		// load, repeatedly growing) a new one every millisecond.
		let mut events = Vec::<WorkerEvent>::new();

		loop {
			events.clear();
			events.extend(self.listener.try_iter());

			let mut has_current_set = false;

			for event in events.drain(..) {
				match event {
					WorkerEvent::Get(key, hit) => self.handle_get(key, hit),

					WorkerEvent::Set(key, size, _, _) => {
						self.handle_set(key, size);
						has_current_set = true;
					},

					WorkerEvent::Del(key, _) => self.handle_del(key),
					WorkerEvent::Expire(key) => self.handle_expire(key),
					WorkerEvent::Wipe => self.handle_wipe(),
					WorkerEvent::Resize(max_size) => self.handle_resize(max_size),
					WorkerEvent::ResizeFastTier(size) => self.handle_resize_fast_tier(size),
					WorkerEvent::ResizeLargeFastTier(size) => self.handle_resize_large_fast_tier(size),
					WorkerEvent::ResizeSizeThreshold(size) => self.handle_resize_size_threshold(size),

					WorkerEvent::Policy(policy) => {
						self.handle_policy(policy, policy_reconstruct_tx.clone());
					},

					WorkerEvent::Shutdown => {
						// Cascade to our own child worker before stopping
						// ourselves -- see `StackEvent::Shutdown`'s doc
						// comment. Best-effort: if `TraceWorker` already
						// exited on its own (e.g. a prior error return),
						// the send is simply a no-op and the join returns
						// immediately.
						if let Some(trace_worker) = &self.trace_worker {
							let _ = trace_worker.send(StackEvent::Shutdown);
						}

						if let Some(handle) = self.trace_handle.take() {
							let _ = handle.join();
						}

						return Ok(());
					},

					_ => {},
				}

				// Skipped entirely when tracing is off (see `trace_worker`'s
				// doc comment) -- not just the send, but deriving the
				// `StackEvent` in the first place. This is the per-access cost
				// that a single-policy cache was paying for a replay it can
				// never perform.
				if let Some(trace_worker) = &self.trace_worker {
					if let Some(stack_event) = StackEvent::maybe_from_worker_event(&event) {
						if self.policy_stack.is_some() {
							if let Err(err) = trace_worker.send(stack_event) {
								error!("Could not send stack event to trace worker: {err:?}");
								return Err(CacheError::Internal);
							}
						} else {
							buffered_events.push(stack_event);
						}
					}
				}

				// Applied per-event rather than once after the whole batch
				// drains: a `set()` writes its `TieredBuffer` to DRAM
				// synchronously at the API layer, before this worker even
				// sees the event, so the only latency this loop controls is
				// how soon a demotion decision made *during* this batch gets
				// physically executed (moving bytes to PMEM). Migrating
				// per-event shrinks that window from "however long the rest
				// of this batch takes to process" down to one event, at the
				// cost of potentially more, smaller `apply_tier_migrations`
				// calls under heavy concurrent load — cheap to call when
				// there's nothing to migrate (an early-return on an empty
				// drain), so this isn't a meaningful throughput cost.
				//
				// Tested reverting this to once-per-batch while investigating
				// why real DRAM usage doesn't track fast_tier_size (see
				// CLAUDE.md): made no measurable difference (200K-object
				// scale: batched 4030.4/4039.1 MB vs. per-event's original
				// 3959.0/3983.3 MB — within normal run-to-run noise). The
				// allocator-level retention behavior responsible for that gap
				// is independent of this loop's migration granularity.
				#[cfg(any(feature = "lru_hybrid_cache", feature = "lfu_hybrid_cache", feature = "two_q_hybrid_cache", feature = "two_q_fast_admission_hybrid_cache", feature = "two_q_fast_admission_reprieve_hybrid_cache", feature = "fifo_hybrid_cache", feature = "lru_sized_hybrid_cache", feature = "s3_fifo_hybrid_cache", feature = "two_q_ghost_hybrid_cache", feature = "s3_fifo_ghost_hybrid_cache", feature = "s3_fifo_ghost_lazy_demotion_hybrid_cache", feature = "s3_fifo_ghost_lazy_demotion_fast_admission_hybrid_cache", feature = "s3_fifo_ghost_lazy_demotion_fast_admission_midpoint_hybrid_cache", feature = "s3_fifo_lazy_demotion_fast_admission_midpoint_reprieve_hybrid_cache", feature = "s3_fifo_lazy_demotion_fast_admission_reprieve_hybrid_cache", feature = "s3_fifo_lazy_demotion_fast_admission_split_slow_reprieve_hybrid_cache", feature = "s3_fifo_lazy_demotion_reprieve_hybrid_cache"))]
				self.apply_tier_migrations();
			}

			self.apply_buffered_events(&buffered_events, &policy_reconstruct_rx);
			self.flush_buffered_events(&mut buffered_events)?;
			self.apply_evictions(&mut buffered_events)?;

			// `apply_evictions` runs every outer-loop iteration regardless
			// of whether `events` was non-empty (unlike the per-event call
			// above, gated on there being an event to process at all) --
			// this matters because `S3FifoHybridStack::evict_one` can push
			// a real (key, Tier::Fast) migration as a side effect of its
			// eviction sweep (`give_second_chance`, the CLOCK-style "reused
			// second chance" mechanic: a key found with its reference bit
			// set gets promoted instead of evicted). Every other hybrid
			// stack's `evict_one` only ever pops-and-removes for a real
			// eviction -- it never touches `self.migrations` -- so this call
			// was previously safe to omit here; it's a correctness
			// requirement now that at least one stack's eviction sweep can
			// produce a promotion. Cheap early-return when there's nothing
			// to migrate, same as the per-event call above.
			#[cfg(any(feature = "lru_hybrid_cache", feature = "lfu_hybrid_cache", feature = "two_q_hybrid_cache", feature = "two_q_fast_admission_hybrid_cache", feature = "two_q_fast_admission_reprieve_hybrid_cache", feature = "fifo_hybrid_cache", feature = "lru_sized_hybrid_cache", feature = "s3_fifo_hybrid_cache", feature = "two_q_ghost_hybrid_cache", feature = "s3_fifo_ghost_hybrid_cache", feature = "s3_fifo_ghost_lazy_demotion_hybrid_cache", feature = "s3_fifo_ghost_lazy_demotion_fast_admission_hybrid_cache", feature = "s3_fifo_ghost_lazy_demotion_fast_admission_midpoint_hybrid_cache", feature = "s3_fifo_lazy_demotion_fast_admission_midpoint_reprieve_hybrid_cache", feature = "s3_fifo_lazy_demotion_fast_admission_reprieve_hybrid_cache", feature = "s3_fifo_lazy_demotion_fast_admission_split_slow_reprieve_hybrid_cache", feature = "s3_fifo_lazy_demotion_reprieve_hybrid_cache"))]
			self.apply_tier_migrations();

			// Once per pass, after every migration and eviction this pass
			// could produce has already been applied -- so what gets published
			// here is exactly as current as a per-event refresh would have
			// left it, without putting those stores on the per-read path. See
			// `refresh_tier_gauges`.
			#[cfg(any(feature = "lru_hybrid_cache", feature = "lfu_hybrid_cache", feature = "two_q_hybrid_cache", feature = "two_q_fast_admission_hybrid_cache", feature = "two_q_fast_admission_reprieve_hybrid_cache", feature = "fifo_hybrid_cache", feature = "lru_sized_hybrid_cache", feature = "s3_fifo_hybrid_cache", feature = "two_q_ghost_hybrid_cache", feature = "s3_fifo_ghost_hybrid_cache", feature = "s3_fifo_ghost_lazy_demotion_hybrid_cache", feature = "s3_fifo_ghost_lazy_demotion_fast_admission_hybrid_cache", feature = "s3_fifo_ghost_lazy_demotion_fast_admission_midpoint_hybrid_cache", feature = "s3_fifo_lazy_demotion_fast_admission_midpoint_reprieve_hybrid_cache", feature = "s3_fifo_lazy_demotion_fast_admission_reprieve_hybrid_cache", feature = "s3_fifo_lazy_demotion_fast_admission_split_slow_reprieve_hybrid_cache", feature = "s3_fifo_lazy_demotion_reprieve_hybrid_cache"))]
			self.refresh_tier_gauges();

			let now = Instant::now();

			if let Some(policy) = self.perform_auto_policy(now, has_current_set) {
				self.status.set_auto_policy(policy)?;
				self.handle_policy(policy, policy_reconstruct_tx.clone());
			}

			self.delay_event_loop(now, has_current_set);
		}
	}
}

impl<K, V> PolicyWorker<K, V>
where
	K: Eq + TypeSize,
	V: TypeSize,
{
	pub fn new(
		listener: WorkerReceiver,
		objects: ObjectMapRef<K, V>,
		status: StatusRef,
		overhead_manager: OverheadManagerRef,
		promotion_tx: Option<WorkerSender>,
	) -> Result<Self, CacheError> {
		let max_cache_size = status.max_size();

		let mini_stacks = MiniStackManager::new(
			status.policies(),
			max_cache_size,
		);

		let policy = status.policy();
		let policy_stack = init_policy_stack(policy, max_cache_size);

		let trace_fragments = Arc::new(RwLock::new(VecDeque::new()));
		let (trace_worker, trace_handle) = spawn_trace_worker(
			trace_is_useful(&status),
			&trace_fragments,
			status.max_size(),
		)?;

		let worker = PolicyWorker {
			listener,

			objects,
			status,
			overhead_manager,

			policy_stack: Some(policy_stack),

			trace_fragments,
			trace_worker,
			trace_handle,

			mini_stack_manager: mini_stacks,
			mini_index: None,

			current_policy: Arc::new(RwLock::new(policy)),

			last_auto_policy_time: None,
			last_set_time: None,

			promotion_tx,

			#[cfg(any(feature = "lru_hybrid_cache", feature = "lfu_hybrid_cache", feature = "two_q_hybrid_cache", feature = "two_q_fast_admission_hybrid_cache", feature = "two_q_fast_admission_reprieve_hybrid_cache", feature = "fifo_hybrid_cache", feature = "lru_sized_hybrid_cache", feature = "s3_fifo_hybrid_cache", feature = "two_q_ghost_hybrid_cache", feature = "s3_fifo_ghost_hybrid_cache", feature = "s3_fifo_ghost_lazy_demotion_hybrid_cache", feature = "s3_fifo_ghost_lazy_demotion_fast_admission_hybrid_cache", feature = "s3_fifo_ghost_lazy_demotion_fast_admission_midpoint_hybrid_cache", feature = "s3_fifo_lazy_demotion_fast_admission_midpoint_reprieve_hybrid_cache", feature = "s3_fifo_lazy_demotion_fast_admission_reprieve_hybrid_cache", feature = "s3_fifo_lazy_demotion_fast_admission_split_slow_reprieve_hybrid_cache", feature = "s3_fifo_lazy_demotion_reprieve_hybrid_cache"))]
			tier_migration_fn: None,
		};

		Ok(worker)
	}

	/// Constructs a `PolicyWorker` that physically migrates object bytes
	/// between tiers whenever `PaperPolicy::LruHybrid`'s `LruHybridStack`,
	/// `PaperPolicy::LfuHybrid`'s `LfuHybridStack`,
	/// `PaperPolicy::TwoQHybrid`'s `TwoQHybridStack`, or
	/// `PaperPolicy::FifoHybrid`'s `FifoHybridStack` reports a promotion or
	/// demotion (see `apply_tier_migrations`).
	///
	/// `migrate` reallocates a value into the representation for the given
	/// `Tier` (e.g. `TieredBuffer::new_fast`/`new_slow`). Promotion/demotion/
	/// eviction counters and the current tier gauges are recorded directly
	/// on `status` (see `apply_tier_migrations`), which is why this
	/// constructor needs no separate stats parameter.
	#[cfg(any(feature = "lru_hybrid_cache", feature = "lfu_hybrid_cache", feature = "two_q_hybrid_cache", feature = "two_q_fast_admission_hybrid_cache", feature = "two_q_fast_admission_reprieve_hybrid_cache", feature = "fifo_hybrid_cache", feature = "lru_sized_hybrid_cache", feature = "s3_fifo_hybrid_cache", feature = "two_q_ghost_hybrid_cache", feature = "s3_fifo_ghost_hybrid_cache", feature = "s3_fifo_ghost_lazy_demotion_hybrid_cache", feature = "s3_fifo_ghost_lazy_demotion_fast_admission_hybrid_cache", feature = "s3_fifo_ghost_lazy_demotion_fast_admission_midpoint_hybrid_cache", feature = "s3_fifo_lazy_demotion_fast_admission_midpoint_reprieve_hybrid_cache", feature = "s3_fifo_lazy_demotion_fast_admission_reprieve_hybrid_cache", feature = "s3_fifo_lazy_demotion_fast_admission_split_slow_reprieve_hybrid_cache", feature = "s3_fifo_lazy_demotion_reprieve_hybrid_cache"))]
	pub fn new_with_tier_migration(
		listener: WorkerReceiver,
		objects: ObjectMapRef<K, V>,
		status: StatusRef,
		overhead_manager: OverheadManagerRef,
		migrate: Box<dyn Fn(&V, Tier) -> V + Send + Sync>,
	) -> Result<Self, CacheError> {
		let max_cache_size = status.max_size();

		// Hybrid caches (`lru_hybrid_cache`/`lfu_hybrid_cache`/`two_q_hybrid_cache`,
		// the only callers of this constructor) are always constructed with a
		// single-element policies list containing only their own fixed policy
		// (e.g. `[PaperPolicy::LruHybrid]`) and never with `PaperPolicy::Auto`
		// -- there is no public API to switch a hybrid cache's policy at all
		// (see `PaperCache<K, TieredBuffer, S>`'s design: "no `policy()`
		// method, there's only one policy"). `perform_auto_policy` already
		// short-circuits on `!self.status.is_auto_policy()` before ever
		// consulting a mini stack, so passing the real (single-entry)
		// policies list here only bought a self-referential `MiniStack` that
		// received a sampled copy of every get/set/del/resize/wipe/eviction
		// event (`MiniStackManager`'s `handle_*` methods aren't gated by
		// `is_auto_policy` at all) and could never possibly be switched to.
		// An empty policies list here means zero `MiniStack`s are ever
		// constructed, so those `handle_*` calls become true no-ops. Safe
		// even for the eviction-during-reconstruction path
		// (`apply_mini_evictions`/`MiniStackManager::get_eviction`, the one
		// place that indexes directly into the mini-stacks slice): it only
		// runs when `self.mini_index` is `Some`, which is only ever set by
		// the explicit-policy-switch handler elsewhere in this file --
		// unreachable here for the same reason auto-switching is.
		let mini_stacks = MiniStackManager::new(&[], max_cache_size);

		let policy = status.policy();
		let policy_stack = init_policy_stack(policy, max_cache_size);

		// A hybrid cache is always constructed with a single fixed policy and
		// exposes no way to switch it, so `trace_is_useful` is always false
		// here -- no `TraceWorker` thread, and no per-access trace writes.
		let trace_fragments = Arc::new(RwLock::new(VecDeque::new()));
		let (trace_worker, trace_handle) = spawn_trace_worker(
			trace_is_useful(&status),
			&trace_fragments,
			status.max_size(),
		)?;

		let worker = PolicyWorker {
			listener,

			objects,
			status,
			overhead_manager,

			policy_stack: Some(policy_stack),

			trace_fragments,
			trace_worker,
			trace_handle,

			mini_stack_manager: mini_stacks,
			mini_index: None,

			current_policy: Arc::new(RwLock::new(policy)),

			last_auto_policy_time: None,
			last_set_time: None,

			// None of `LruHybridStack`, `LfuHybridStack`, or `TwoQHybridStack`
			// ever emits `AccessOutcome::GhostHit`, so no ghost-hit-driven
			// promotion channel is needed for any of them.
			promotion_tx: None,

			tier_migration_fn: Some(migrate),
		};

		Ok(worker)
	}

	fn handle_get(&mut self, key: HashedKey, hit: bool) {
		if let Some(stack) = &mut self.policy_stack {
			if let AccessOutcome::GhostHit = stack.record_access(key, hit) {
				debug_assert!(self.promotion_tx.is_some(), "promotion channel must exist for ghost hits");
				if let Some(tx) = &self.promotion_tx {
					let _ = tx.try_send(WorkerEvent::Promote(key));
				}
			}
		}

		self.mini_stack_manager.handle_get(key);
	}

	fn handle_set(&mut self, key: HashedKey, size: ObjectSize) {
		if let Some(stack) = &mut self.policy_stack {
			stack.insert(key, size);
		}

		self.mini_stack_manager.handle_set(key, size);
	}

	fn handle_del(&mut self, key: HashedKey) {
		if let Some(stack) = &mut self.policy_stack {
			stack.remove(key);
		}

		self.mini_stack_manager.handle_del(key);
	}

	/// Drops a key the `TtlWorker` has already reaped out of the object map.
	///
	/// Same stack bookkeeping as `handle_del` -- the object is gone either
	/// way, and the policy stack should not go on ranking it or counting its
	/// bytes -- behind one guard `handle_del` does not need.
	///
	/// The guard exists because a reap is not synchronous with this worker.
	/// `TtlWorker` erases the object and then sends `Expire`; nothing stops a
	/// `set()` on that same key from landing in between, in which case the map
	/// entry this event refers to has already been replaced by a live one.
	/// Removing the key from the stack then would desync in the opposite
	/// direction -- an object present in the map but absent from the stack,
	/// which is strictly worse than the staleness being fixed here, since such
	/// an object can never be chosen for eviction and its bytes go
	/// unaccounted for in the hybrid stacks' tier gauges for as long as it
	/// lives.
	///
	/// Re-reading the map here rather than at send time is what makes this
	/// safe: both the re-set's `WorkerEvent::Set` and this event land in the
	/// same single-consumer channel, so whichever order they arrive in, the
	/// map lookup performed *at the moment this event is handled* agrees with
	/// the stack state this worker is about to produce.
	fn handle_expire(&mut self, key: HashedKey) {
		if self.object_exists(key) {
			// Re-set between the reap and this notification -- the key is
			// live again, and its `Set` event owns the stack entry now.
			return;
		}

		self.handle_del(key);
	}

	/// Whether the object map still holds `key`.
	///
	/// Two bodies because the `object_store` module -- and so the
	/// `ObjectStore` trait that supplies `get_ref` -- is itself gated on a
	/// storage feature being selected (`lib.rs`). A build that selects none
	/// (bare `eviction_stacks_pmem`, say) still resolves `ObjectMapRef` to the
	/// default `DashMap` shape, which answers this directly.
	#[cfg(any(feature = "all_dram", feature = "key_value_pmem", feature = "global_hashtable_pmem", feature = "hashbrown_dram"))]
	fn object_exists(&self, key: HashedKey) -> bool {
		self.objects.get_ref(&key).is_some()
	}

	#[cfg(not(any(feature = "all_dram", feature = "key_value_pmem", feature = "global_hashtable_pmem", feature = "hashbrown_dram")))]
	fn object_exists(&self, key: HashedKey) -> bool {
		self.objects.contains_key(&key)
	}

	fn handle_resize(&mut self, size: CacheSize) {
		if let Some(stack) = &mut self.policy_stack {
			stack.resize(size);
		}

		self.mini_stack_manager.handle_resize(size);
	}

	/// Runtime-adjusts the fast-tier byte budget (`lru_hybrid_cache`). No-op
	/// for every other policy stack. May itself trigger demotions, drained by
	/// `apply_tier_migrations` on the next pass through the event loop.
	fn handle_resize_fast_tier(&mut self, size: CacheSize) {
		if let Some(stack) = &mut self.policy_stack {
			stack.resize_fast_tier(size);
		}
	}

	/// Runtime-adjusts the LARGE fast segment's byte budget
	/// (`lru_sized_hybrid_cache` specifically). No-op for every other policy
	/// stack. May itself trigger demotions, drained by `apply_tier_migrations`
	/// on the next pass through the event loop.
	fn handle_resize_large_fast_tier(&mut self, size: CacheSize) {
		if let Some(stack) = &mut self.policy_stack {
			stack.resize_large_fast_tier(size);
		}
	}

	/// Runtime-adjusts the small/large size-classification threshold
	/// (`lru_sized_hybrid_cache`). No-op for every other policy stack.
	fn handle_resize_size_threshold(&mut self, size: CacheSize) {
		if let Some(stack) = &mut self.policy_stack {
			stack.resize_size_threshold(size);
		}
	}

	fn handle_policy(
		&mut self,
		policy: PaperPolicy,
		policy_reconstruct_tx: Arc<Sender<Box<dyn PolicyStack>>>,
	) {
		if policy.is_auto() || policy == *self.current_policy.read() {
			return;
		}

		// Defensive: reconstruction replays the access trace, so without one
		// there is nothing to rebuild the new stack from. Bail before the
		// teardown below rather than after -- clearing `policy_stack` for a
		// reconstruction that can never deliver would leave this worker with
		// no stack at all, permanently. Unreachable in practice: tracing is
		// only off when a single policy is configured, and both callers
		// (`WorkerEvent::Policy`, itself validated against the configured
		// policy list by `PaperCache::policy()`, and `perform_auto_policy`,
		// which picks from that same list) can then only ever name the policy
		// already running, which the equality check above already caught.
		if self.trace_worker.is_none() {
			warn!("Ignoring switch to {policy}: policy reconstruction is disabled");
			return;
		}

		info!(
			"Switching policy {} to {policy}",
			self.current_policy.read(),
		);

		*self.current_policy.write() = policy;

		let mini_index = self.mini_stack_manager.get_index(&policy);

		self.policy_stack = None;
		self.mini_index = Some(mini_index);

		let max_cache_size = self.status.max_size();
		let current_policy = self.current_policy.clone();
		let trace_fragments = self.trace_fragments.clone();

		thread::spawn(move || {
			info!("Reconstructing {policy} stack");
			let now = Instant::now();

			let reconstruction_result = reconstruct_policy_stack(
				policy,
				max_cache_size,
				current_policy.clone(),
				trace_fragments.clone(),
			);

			if let Ok(stack) = reconstruction_result {
				// check to make sure the configured policy was not modified
				// before sending the reconstructed stack
				if policy == *current_policy.read() {
					info!(
						"{policy} stack reconstructed with {} object(s) in {:?}",
						fmt::number(stack.len()),
						now.elapsed(),
					);

					let _ = policy_reconstruct_tx.send(stack);
				} else {
					warn!("The policy changed during reconstruction");
				}
			}
		});
	}

	fn handle_wipe(&mut self) {
		if let Some(stack) = &mut self.policy_stack {
			stack.clear();
		}

		self.mini_stack_manager.handle_wipe();
	}

	/// Drains tier migrations recorded by the policy stack (only
	/// `LruHybridStack` ever produces any) since the last call, and
	/// physically reallocates each migrated object's bytes into the target
	/// tier's representation via `Object::set_data`. Also refreshes the live
	/// gauge fields in `lru_hybrid_stats`. No-op unless this worker was built
	/// via `new_with_tier_migration` (`tier_migration_fn` is `None`
	/// otherwise, regardless of the active policy).
	///
	/// `lru_hybrid_cache`, `lfu_hybrid_cache`, `two_q_hybrid_cache`, and
	/// `fifo_hybrid_cache` are mutually exclusive features (see `lib.rs`'s
	/// `compile_error!` guards), so exactly one of this method and its three
	/// siblings below ever compiles.
	///
	/// Physical migrations are applied inline in this policy-worker thread,
	/// sequentially, in two phases: every demotion in this batch is applied
	/// before any promotion in the same batch begins. That ordering is a
	/// correctness requirement, not an optimization -- a promotion's new
	/// fast-tier DRAM allocation must never run ahead of a demotion that
	/// exists specifically to free room for it. (An earlier version spread
	/// these copies across `rayon`'s worker pool; that was removed as a
	/// measured performance loss -- concurrent PMEM allocations contend on
	/// the underlying allocator -- see CLAUDE.md.)
	#[cfg(feature = "lru_hybrid_cache")]
	fn apply_tier_migrations(&mut self) {
		let Some(stack) = &mut self.policy_stack else { return };
		let migrations = stack.drain_tier_migrations();

		if !migrations.is_empty() {
			if let Some(migrate) = &self.tier_migration_fn {
				let (demotions, promotions): (Vec<_>, Vec<_>) = migrations
					.into_iter()
					.partition(|(_, tier)| *tier == Tier::Slow);

				let objects = &self.objects;
				let status = &self.status;

				// Build the destination buffer with NO object-map guard
				// held. `migrate` is a real allocation plus a full byte
				// copy of the value -- a PMEM write on demotion, a PMEM
				// read on promotion -- and at this crate's real object
				// sizes (~16 KB average on the benchmark traces) that is
				// microseconds, not nanoseconds. Holding the shard's
				// *write* guard across it stalls every concurrent `get()`
				// that hashes to the same shard, which surfaces as GET tail
				// latency rather than a uniform slowdown.
				//
				// `Object::data()` is only an `Arc` refcount bump, and the
				// `Arc` keeps the source bytes alive independently of the
				// map, so the snapshot below is safe to use unlocked.
				let apply_physical = |(key, tier): (HashedKey, Tier)| {
					let Some(old_data) = objects.get_ref(&key).map(|object| object.data()) else {
						return;
					};

					let new_data = migrate(&old_data, tier);

					// Re-acquire only to swap the pointer. The `ptr_eq`
					// guard matters: `PaperCache::set()` runs on the API
					// thread and can replace this entry while the copy
					// above is in flight, and writing `new_data` over a
					// *replacement* value would resurrect the bytes of the
					// value it replaced. If the object changed (or was
					// evicted), the migration is stale -- drop it, and let
					// the stack's next event re-derive the correct tier.
					if let Some(mut object) = objects.get_mut_ref(&key) {
						if Arc::ptr_eq(&object.data(), &old_data) {
							object.set_data(new_data);
						}
					}
				};

				demotions.into_iter().for_each(|entry| {
					apply_physical(entry);
					status.record_lru_hybrid_demotion();
				});

				promotions.into_iter().for_each(|entry| {
					apply_physical(entry);
					status.record_lru_hybrid_promotion();
				});
			}
		}
	}

	/// Republishes the active hybrid stack's tier gauges onto
	/// `AtomicStatus`, backing this feature's `*_hybrid_stats()`
	/// accessor.
	///
	/// Split out of `apply_tier_migrations` and called once per pass of
	/// the event loop rather than once per event. These are pure gauges --
	/// a snapshot of state the stack already owns -- so republishing them
	/// after each batch reports exactly the same values as republishing
	/// after each event; only the write frequency changes. That frequency
	/// mattered: it put four virtual calls and four atomic stores into
	/// `AtomicStatus` on the path of every single cache read, and those
	/// stores land in the same struct the API threads are concurrently
	/// incrementing their hit/miss counters in.
	///
	/// Still unconditional (not gated on a migration having happened):
	/// that gate is what let these gauges go stale indefinitely -- see the
	/// note in `apply_tier_migrations` -- and removing it is what fixed
	/// them. This only moves *when* the refresh runs, not *whether*.
	#[cfg(feature = "lru_hybrid_cache")]
	fn refresh_tier_gauges(&mut self) {
		// Refreshed unconditionally (not gated on `migrations` being
		// non-empty): gating it on "a migration just happened" left
		// `lru_hybrid_stats`'s
		// `fast_objects`/`slow_objects`/`fast_bytes_used`/`slow_bytes_used`
		// able to go stale and never catch up to the stack's true state --
		// e.g. the tail of a large insert burst that happens to land without
		// triggering one further demotion left these gauges permanently
		// short of the real tracked count, discovered while diagnosing a
		// reported DRAM-usage gap (the gauges themselves turned out fine;
		// only their refresh cadence was wrong).
		if let Some(stack) = &self.policy_stack {
			self.status.set_lru_hybrid_gauges(
				stack.fast_bytes_used(),
				stack.slow_bytes_used(),
				stack.fast_object_count() as u64,
				stack.slow_object_count() as u64,
			);
		}
	}

	/// `lfu_hybrid_cache` counterpart of the `lru_hybrid_cache` method above
	/// — same overall shape, draining `LfuHybridStack`'s migrations instead
	/// and recording to the `lfu_hybrid_*` counters/gauges on `status`, but
	/// with one difference: a `Tier::Slow` migration here isn't always a
	/// genuine demotion — `LfuHybridStack::insert` can also route a fresh
	/// admission directly to slow once the fast tier is already full, which
	/// still needs the same physical `Object::set_data` correction (the
	/// object was initially built as `Fast` by the API layer) but must not
	/// be counted as a demotion (no existing fast-tier object was
	/// displaced). So every migration is still applied physically here, but
	/// the demotion *count* comes from `drain_demotions` instead of being
	/// inferred per-entry from `Tier::Slow`.
	///
	/// Also mirrors `stack.admission_latched()` onto `AtomicStatus` on every
	/// call (unconditionally, before the empty-migrations early return —
	/// this method already runs after every event, per-event, so this keeps
	/// the mirror fresh regardless of whether *this particular* event
	/// produced a migration) so `PaperCache::set()`, running on the
	/// API-calling thread with no direct access to this worker-owned stack,
	/// can build a brand-new key's `TieredBuffer` in the correct tier up
	/// front instead of always guessing fast.
	///
	/// Same inline-sequential, demotions-before-promotions shape as the
	/// `lru_hybrid_cache` sibling above (see its comment): every
	/// `Tier::Slow` entry in this batch -- genuine demotions and
	/// fresh-admission-to-slow corrections alike -- is applied before any
	/// `Tier::Fast` (promotion) entry, so a promotion never allocates
	/// fast-tier DRAM ahead of the demotion(s) that were supposed to make
	/// room for it.
	#[cfg(feature = "lfu_hybrid_cache")]
	fn apply_tier_migrations(&mut self) {
		let Some(stack) = &mut self.policy_stack else { return };

		let migrations = stack.drain_tier_migrations();

		if !migrations.is_empty() {
			if let Some(migrate) = &self.tier_migration_fn {
				let (demotions, promotions): (Vec<_>, Vec<_>) = migrations
					.into_iter()
					.partition(|(_, tier)| *tier == Tier::Slow);

				let objects = &self.objects;
				let status = &self.status;

				// Build the destination buffer with NO object-map guard
				// held. `migrate` is a real allocation plus a full byte
				// copy of the value -- a PMEM write on demotion, a PMEM
				// read on promotion -- and at this crate's real object
				// sizes (~16 KB average on the benchmark traces) that is
				// microseconds, not nanoseconds. Holding the shard's
				// *write* guard across it stalls every concurrent `get()`
				// that hashes to the same shard, which surfaces as GET tail
				// latency rather than a uniform slowdown.
				//
				// `Object::data()` is only an `Arc` refcount bump, and the
				// `Arc` keeps the source bytes alive independently of the
				// map, so the snapshot below is safe to use unlocked.
				let apply_physical = |(key, tier): (HashedKey, Tier)| {
					let Some(old_data) = objects.get_ref(&key).map(|object| object.data()) else {
						return;
					};

					let new_data = migrate(&old_data, tier);

					// Re-acquire only to swap the pointer. The `ptr_eq`
					// guard matters: `PaperCache::set()` runs on the API
					// thread and can replace this entry while the copy
					// above is in flight, and writing `new_data` over a
					// *replacement* value would resurrect the bytes of the
					// value it replaced. If the object changed (or was
					// evicted), the migration is stale -- drop it, and let
					// the stack's next event re-derive the correct tier.
					if let Some(mut object) = objects.get_mut_ref(&key) {
						if Arc::ptr_eq(&object.data(), &old_data) {
							object.set_data(new_data);
						}
					}
				};

				// Not counted per-entry here -- unlike the other two
				// hybrids, a `Tier::Slow` entry isn't always a genuine
				// demotion (see the doc comment above this method's
				// declaration); the real count comes from
				// `stack.drain_demotions()` below, unchanged from before.
				demotions.into_iter().for_each(&apply_physical);

				promotions.into_iter().for_each(|entry| {
					apply_physical(entry);
					status.record_lfu_hybrid_promotion();
				});
			}

			let demotions = stack.drain_demotions();
			self.status.record_lfu_hybrid_demotions(demotions);
		}
	}

	/// Republishes the active hybrid stack's tier gauges onto
	/// `AtomicStatus`, backing this feature's `*_hybrid_stats()`
	/// accessor.
	///
	/// Split out of `apply_tier_migrations` and called once per pass of
	/// the event loop rather than once per event. These are pure gauges --
	/// a snapshot of state the stack already owns -- so republishing them
	/// after each batch reports exactly the same values as republishing
	/// after each event; only the write frequency changes. That frequency
	/// mattered: it put four virtual calls and four atomic stores into
	/// `AtomicStatus` on the path of every single cache read, and those
	/// stores land in the same struct the API threads are concurrently
	/// incrementing their hit/miss counters in.
	///
	/// Still unconditional (not gated on a migration having happened):
	/// that gate is what let these gauges go stale indefinitely -- see the
	/// note in `apply_tier_migrations` -- and removing it is what fixed
	/// them. This only moves *when* the refresh runs, not *whether*.
	#[cfg(feature = "lfu_hybrid_cache")]
	fn refresh_tier_gauges(&mut self) {
		// Mirrored alongside the gauges for the same reason, and with the
		// same safety: the latch is one-way (it closes when the fast tier
		// first fills and never reopens except on `clear`), so a mirror
		// that lags by one pass of the event loop only means a handful of
		// sets around the transition build `Fast` and get corrected --
		// exactly the behaviour that existed before the latch, for a
		// moment, rather than any incorrect placement.
		if let Some(stack) = &self.policy_stack {
			self.status.set_lfu_hybrid_admission_latched(stack.admission_latched());
		}

		// Refreshed unconditionally -- see the `lru_hybrid_cache` sibling's
		// comment on this same pattern for why gating it on `migrations`
		// being non-empty let these gauges go stale and never catch up.
		if let Some(stack) = &self.policy_stack {
			self.status.set_lfu_hybrid_gauges(
				stack.fast_bytes_used(),
				stack.slow_bytes_used(),
				stack.fast_object_count() as u64,
				stack.slow_object_count() as u64,
			);
		}
	}

	/// `two_q_hybrid_cache` counterpart of the two methods above — identical
	/// shape, draining `TwoQHybridStack`'s migrations instead and recording
	/// to the `two_q_hybrid_*` counters/gauges on `status`.
	///
	/// Same inline-sequential, demotions-before-promotions shape as the
	/// `lru_hybrid_cache` sibling above (see its comment): all demotions
	/// in this batch are applied before any promotion in the same batch
	/// begins.
	#[cfg(feature = "two_q_hybrid_cache")]
	fn apply_tier_migrations(&mut self) {
		let Some(stack) = &mut self.policy_stack else { return };
		let migrations = stack.drain_tier_migrations();

		if !migrations.is_empty() {
			if let Some(migrate) = &self.tier_migration_fn {
				let (demotions, promotions): (Vec<_>, Vec<_>) = migrations
					.into_iter()
					.partition(|(_, tier)| *tier == Tier::Slow);

				let objects = &self.objects;
				let status = &self.status;

				// Build the destination buffer with NO object-map guard
				// held. `migrate` is a real allocation plus a full byte
				// copy of the value -- a PMEM write on demotion, a PMEM
				// read on promotion -- and at this crate's real object
				// sizes (~16 KB average on the benchmark traces) that is
				// microseconds, not nanoseconds. Holding the shard's
				// *write* guard across it stalls every concurrent `get()`
				// that hashes to the same shard, which surfaces as GET tail
				// latency rather than a uniform slowdown.
				//
				// `Object::data()` is only an `Arc` refcount bump, and the
				// `Arc` keeps the source bytes alive independently of the
				// map, so the snapshot below is safe to use unlocked.
				let apply_physical = |(key, tier): (HashedKey, Tier)| {
					let Some(old_data) = objects.get_ref(&key).map(|object| object.data()) else {
						return;
					};

					let new_data = migrate(&old_data, tier);

					// Re-acquire only to swap the pointer. The `ptr_eq`
					// guard matters: `PaperCache::set()` runs on the API
					// thread and can replace this entry while the copy
					// above is in flight, and writing `new_data` over a
					// *replacement* value would resurrect the bytes of the
					// value it replaced. If the object changed (or was
					// evicted), the migration is stale -- drop it, and let
					// the stack's next event re-derive the correct tier.
					if let Some(mut object) = objects.get_mut_ref(&key) {
						if Arc::ptr_eq(&object.data(), &old_data) {
							object.set_data(new_data);
						}
					}
				};

				demotions.into_iter().for_each(|entry| {
					apply_physical(entry);
					status.record_two_q_hybrid_demotion();
				});

				promotions.into_iter().for_each(|entry| {
					apply_physical(entry);
					status.record_two_q_hybrid_promotion();
				});
			}
		}
	}

	/// Republishes the active hybrid stack's tier gauges onto
	/// `AtomicStatus`, backing this feature's `*_hybrid_stats()`
	/// accessor.
	///
	/// Split out of `apply_tier_migrations` and called once per pass of
	/// the event loop rather than once per event. These are pure gauges --
	/// a snapshot of state the stack already owns -- so republishing them
	/// after each batch reports exactly the same values as republishing
	/// after each event; only the write frequency changes. That frequency
	/// mattered: it put four virtual calls and four atomic stores into
	/// `AtomicStatus` on the path of every single cache read, and those
	/// stores land in the same struct the API threads are concurrently
	/// incrementing their hit/miss counters in.
	///
	/// Still unconditional (not gated on a migration having happened):
	/// that gate is what let these gauges go stale indefinitely -- see the
	/// note in `apply_tier_migrations` -- and removing it is what fixed
	/// them. This only moves *when* the refresh runs, not *whether*.
	#[cfg(feature = "two_q_hybrid_cache")]
	fn refresh_tier_gauges(&mut self) {
		// Refreshed unconditionally -- see the `lru_hybrid_cache` sibling's
		// comment on this same pattern for why gating it on `migrations`
		// being non-empty let these gauges go stale and never catch up.
		if let Some(stack) = &self.policy_stack {
			self.status.set_two_q_hybrid_gauges(
				stack.fast_bytes_used(),
				stack.slow_bytes_used(),
				stack.fast_object_count() as u64,
				stack.slow_object_count() as u64,
			);
		}
	}

	/// `two_q_fast_admission_hybrid_cache` counterpart of the methods above — identical
	/// shape, draining `TwoQFastAdmissionHybridStack`'s migrations instead and recording
	/// to the `two_q_fast_admission_hybrid_*` counters/gauges on `status`.
	///
	/// Note this design's `promotions` counter only ever counts genuine
	/// slow->fast moves: its FIFO->main promotion is Fast->Fast and emits
	/// no migration at all (see `TwoQFastAdmissionHybridStack`'s module
	/// doc), so nothing reaches here to be counted for it.
	///
	/// Same inline-sequential, demotions-before-promotions shape as the
	/// `lru_hybrid_cache` sibling above (see its comment): all demotions
	/// in this batch are applied before any promotion in the same batch
	/// begins.
	#[cfg(feature = "two_q_fast_admission_hybrid_cache")]
	fn apply_tier_migrations(&mut self) {
		let Some(stack) = &mut self.policy_stack else { return };
		let migrations = stack.drain_tier_migrations();

		if !migrations.is_empty() {
			if let Some(migrate) = &self.tier_migration_fn {
				let (demotions, promotions): (Vec<_>, Vec<_>) = migrations
					.into_iter()
					.partition(|(_, tier)| *tier == Tier::Slow);

				let objects = &self.objects;
				let status = &self.status;

				// Build the destination buffer with NO object-map guard
				// held. `migrate` is a real allocation plus a full byte
				// copy of the value -- a PMEM write on demotion, a PMEM
				// read on promotion -- and at this crate's real object
				// sizes (~16 KB average on the benchmark traces) that is
				// microseconds, not nanoseconds. Holding the shard's
				// *write* guard across it stalls every concurrent `get()`
				// that hashes to the same shard, which surfaces as GET tail
				// latency rather than a uniform slowdown.
				//
				// `Object::data()` is only an `Arc` refcount bump, and the
				// `Arc` keeps the source bytes alive independently of the
				// map, so the snapshot below is safe to use unlocked.
				let apply_physical = |(key, tier): (HashedKey, Tier)| {
					let Some(old_data) = objects.get_ref(&key).map(|object| object.data()) else {
						return;
					};

					let new_data = migrate(&old_data, tier);

					// Re-acquire only to swap the pointer. The `ptr_eq`
					// guard matters: `PaperCache::set()` runs on the API
					// thread and can replace this entry while the copy
					// above is in flight, and writing `new_data` over a
					// *replacement* value would resurrect the bytes of the
					// value it replaced. If the object changed (or was
					// evicted), the migration is stale -- drop it, and let
					// the stack's next event re-derive the correct tier.
					if let Some(mut object) = objects.get_mut_ref(&key) {
						if Arc::ptr_eq(&object.data(), &old_data) {
							object.set_data(new_data);
						}
					}
				};

				demotions.into_iter().for_each(|entry| {
					apply_physical(entry);
					status.record_two_q_fast_admission_hybrid_demotion();
				});

				promotions.into_iter().for_each(|entry| {
					apply_physical(entry);
					status.record_two_q_fast_admission_hybrid_promotion();
				});
			}
		}
	}

	/// Republishes the active hybrid stack's tier gauges onto
	/// `AtomicStatus`, backing this feature's `*_hybrid_stats()`
	/// accessor.
	///
	/// Split out of `apply_tier_migrations` and called once per pass of
	/// the event loop rather than once per event. These are pure gauges --
	/// a snapshot of state the stack already owns -- so republishing them
	/// after each batch reports exactly the same values as republishing
	/// after each event; only the write frequency changes. That frequency
	/// mattered: it put four virtual calls and four atomic stores into
	/// `AtomicStatus` on the path of every single cache read, and those
	/// stores land in the same struct the API threads are concurrently
	/// incrementing their hit/miss counters in.
	///
	/// Still unconditional (not gated on a migration having happened):
	/// that gate is what let these gauges go stale indefinitely -- see the
	/// note in `apply_tier_migrations` -- and removing it is what fixed
	/// them. This only moves *when* the refresh runs, not *whether*.
	#[cfg(feature = "two_q_fast_admission_hybrid_cache")]
	fn refresh_tier_gauges(&mut self) {
		// Refreshed unconditionally -- see the `lru_hybrid_cache` sibling's
		// comment on this same pattern for why gating it on `migrations`
		// being non-empty let these gauges go stale and never catch up.
		if let Some(stack) = &self.policy_stack {
			self.status.set_two_q_fast_admission_hybrid_gauges(
				stack.fast_bytes_used(),
				stack.slow_bytes_used(),
				stack.fast_object_count() as u64,
				stack.slow_object_count() as u64,
			);
		}
	}

	/// `two_q_fast_admission_reprieve_hybrid_cache` counterpart of the methods above — identical
	/// shape, draining `TwoQFastAdmissionHybridStack`'s migrations instead and recording
	/// to the `two_q_fast_admission_reprieve_hybrid_*` counters/gauges on `status`.
	///
	/// Note `demotions` here covers both real fast->slow demotions AND
	/// one-access keys reprieved out of the FIFO queue -- both are
	/// `Tier::Slow` migrations carrying a real DRAM->PMEM copy, and the
	/// stack does not distinguish them.
	///
	/// Note this design's `promotions` counter only ever counts genuine
	/// slow->fast moves: its FIFO->main promotion is Fast->Fast and emits
	/// no migration at all (see `TwoQFastAdmissionHybridStack`'s module
	/// doc), so nothing reaches here to be counted for it.
	///
	/// Same inline-sequential, demotions-before-promotions shape as the
	/// `lru_hybrid_cache` sibling above (see its comment): all demotions
	/// in this batch are applied before any promotion in the same batch
	/// begins.
	#[cfg(feature = "two_q_fast_admission_reprieve_hybrid_cache")]
	fn apply_tier_migrations(&mut self) {
		let Some(stack) = &mut self.policy_stack else { return };
		let migrations = stack.drain_tier_migrations();

		if !migrations.is_empty() {
			if let Some(migrate) = &self.tier_migration_fn {
				let (demotions, promotions): (Vec<_>, Vec<_>) = migrations
					.into_iter()
					.partition(|(_, tier)| *tier == Tier::Slow);

				let objects = &self.objects;
				let status = &self.status;

				// Build the destination buffer with NO object-map guard
				// held. `migrate` is a real allocation plus a full byte
				// copy of the value -- a PMEM write on demotion, a PMEM
				// read on promotion -- and at this crate's real object
				// sizes (~16 KB average on the benchmark traces) that is
				// microseconds, not nanoseconds. Holding the shard's
				// *write* guard across it stalls every concurrent `get()`
				// that hashes to the same shard, which surfaces as GET tail
				// latency rather than a uniform slowdown.
				//
				// `Object::data()` is only an `Arc` refcount bump, and the
				// `Arc` keeps the source bytes alive independently of the
				// map, so the snapshot below is safe to use unlocked.
				let apply_physical = |(key, tier): (HashedKey, Tier)| {
					let Some(old_data) = objects.get_ref(&key).map(|object| object.data()) else {
						return;
					};

					let new_data = migrate(&old_data, tier);

					// Re-acquire only to swap the pointer. The `ptr_eq`
					// guard matters: `PaperCache::set()` runs on the API
					// thread and can replace this entry while the copy
					// above is in flight, and writing `new_data` over a
					// *replacement* value would resurrect the bytes of the
					// value it replaced. If the object changed (or was
					// evicted), the migration is stale -- drop it, and let
					// the stack's next event re-derive the correct tier.
					if let Some(mut object) = objects.get_mut_ref(&key) {
						if Arc::ptr_eq(&object.data(), &old_data) {
							object.set_data(new_data);
						}
					}
				};

				demotions.into_iter().for_each(|entry| {
					apply_physical(entry);
					status.record_two_q_fast_admission_reprieve_hybrid_demotion();
				});

				promotions.into_iter().for_each(|entry| {
					apply_physical(entry);
					status.record_two_q_fast_admission_reprieve_hybrid_promotion();
				});
			}
		}
	}

	/// Republishes the active hybrid stack's tier gauges onto
	/// `AtomicStatus`, backing this feature's `*_hybrid_stats()`
	/// accessor.
	///
	/// Split out of `apply_tier_migrations` and called once per pass of
	/// the event loop rather than once per event. These are pure gauges --
	/// a snapshot of state the stack already owns -- so republishing them
	/// after each batch reports exactly the same values as republishing
	/// after each event; only the write frequency changes. That frequency
	/// mattered: it put four virtual calls and four atomic stores into
	/// `AtomicStatus` on the path of every single cache read, and those
	/// stores land in the same struct the API threads are concurrently
	/// incrementing their hit/miss counters in.
	///
	/// Still unconditional (not gated on a migration having happened):
	/// that gate is what let these gauges go stale indefinitely -- see the
	/// note in `apply_tier_migrations` -- and removing it is what fixed
	/// them. This only moves *when* the refresh runs, not *whether*.
	#[cfg(feature = "two_q_fast_admission_reprieve_hybrid_cache")]
	fn refresh_tier_gauges(&mut self) {
		// Refreshed unconditionally -- see the `lru_hybrid_cache` sibling's
		// comment on this same pattern for why gating it on `migrations`
		// being non-empty let these gauges go stale and never catch up.
		if let Some(stack) = &self.policy_stack {
			self.status.set_two_q_fast_admission_reprieve_hybrid_gauges(
				stack.fast_bytes_used(),
				stack.slow_bytes_used(),
				stack.fast_object_count() as u64,
				stack.slow_object_count() as u64,
			);
		}
	}

	/// `fifo_hybrid_cache` counterpart of the three methods above — same
	/// overall shape, draining `FifoHybridStack`'s migrations instead and
	/// recording to the `fifo_hybrid_*` counters/gauges on `status`.
	/// `FifoHybridStack` never emits a `Tier::Fast` entry at all (no
	/// promotion path exists — see that stack's module doc), so the
	/// `promotions` half of the demotions/promotions partition below is
	/// always empty in practice; kept structurally symmetric with the other
	/// three siblings anyway (cheap early exit, matches this file's
	/// established style) rather than special-cased away.
	///
	/// Same inline-sequential, demotions-before-promotions shape as the
	/// `lru_hybrid_cache` sibling above (see its comment), though it never
	/// actually matters here since there are no promotions to order against.
	#[cfg(feature = "fifo_hybrid_cache")]
	fn apply_tier_migrations(&mut self) {
		let Some(stack) = &mut self.policy_stack else { return };
		let migrations = stack.drain_tier_migrations();

		if !migrations.is_empty() {
			if let Some(migrate) = &self.tier_migration_fn {
				let (demotions, promotions): (Vec<_>, Vec<_>) = migrations
					.into_iter()
					.partition(|(_, tier)| *tier == Tier::Slow);

				let objects = &self.objects;
				let status = &self.status;

				// Build the destination buffer with NO object-map guard
				// held. `migrate` is a real allocation plus a full byte
				// copy of the value -- a PMEM write on demotion, a PMEM
				// read on promotion -- and at this crate's real object
				// sizes (~16 KB average on the benchmark traces) that is
				// microseconds, not nanoseconds. Holding the shard's
				// *write* guard across it stalls every concurrent `get()`
				// that hashes to the same shard, which surfaces as GET tail
				// latency rather than a uniform slowdown.
				//
				// `Object::data()` is only an `Arc` refcount bump, and the
				// `Arc` keeps the source bytes alive independently of the
				// map, so the snapshot below is safe to use unlocked.
				let apply_physical = |(key, tier): (HashedKey, Tier)| {
					let Some(old_data) = objects.get_ref(&key).map(|object| object.data()) else {
						return;
					};

					let new_data = migrate(&old_data, tier);

					// Re-acquire only to swap the pointer. The `ptr_eq`
					// guard matters: `PaperCache::set()` runs on the API
					// thread and can replace this entry while the copy
					// above is in flight, and writing `new_data` over a
					// *replacement* value would resurrect the bytes of the
					// value it replaced. If the object changed (or was
					// evicted), the migration is stale -- drop it, and let
					// the stack's next event re-derive the correct tier.
					if let Some(mut object) = objects.get_mut_ref(&key) {
						if Arc::ptr_eq(&object.data(), &old_data) {
							object.set_data(new_data);
						}
					}
				};

				demotions.into_iter().for_each(|entry| {
					apply_physical(entry);
					status.record_fifo_hybrid_demotion();
				});

				// Unreachable in practice -- FifoHybridStack never emits
				// Tier::Fast -- but applied physically anyway for the same
				// structural-symmetry reason noted above. No
				// `record_fifo_hybrid_promotion()` call: `fifo_hybrid_stats`'s
				// `promotions` field stays permanently 0 by construction.
				promotions.into_iter().for_each(apply_physical);
			}
		}
	}

	/// Republishes the active hybrid stack's tier gauges onto
	/// `AtomicStatus`, backing this feature's `*_hybrid_stats()`
	/// accessor.
	///
	/// Split out of `apply_tier_migrations` and called once per pass of
	/// the event loop rather than once per event. These are pure gauges --
	/// a snapshot of state the stack already owns -- so republishing them
	/// after each batch reports exactly the same values as republishing
	/// after each event; only the write frequency changes. That frequency
	/// mattered: it put four virtual calls and four atomic stores into
	/// `AtomicStatus` on the path of every single cache read, and those
	/// stores land in the same struct the API threads are concurrently
	/// incrementing their hit/miss counters in.
	///
	/// Still unconditional (not gated on a migration having happened):
	/// that gate is what let these gauges go stale indefinitely -- see the
	/// note in `apply_tier_migrations` -- and removing it is what fixed
	/// them. This only moves *when* the refresh runs, not *whether*.
	#[cfg(feature = "fifo_hybrid_cache")]
	fn refresh_tier_gauges(&mut self) {
		// Refreshed unconditionally -- see the `lru_hybrid_cache` sibling's
		// comment on this same pattern for why gating it on `migrations`
		// being non-empty let these gauges go stale and never catch up.
		if let Some(stack) = &self.policy_stack {
			self.status.set_fifo_hybrid_gauges(
				stack.fast_bytes_used(),
				stack.slow_bytes_used(),
				stack.fast_object_count() as u64,
				stack.slow_object_count() as u64,
			);
		}
	}

	/// `lru_sized_hybrid_cache` counterpart of the four methods above —
	/// same overall shape (drain, partition demotions/promotions on
	/// `Tier::Slow`, demotions fully before promotions, sequential,
	/// gauges refreshed unconditionally) as the `lru_hybrid_cache`
	/// sibling, since every `Tier::Slow`/`Tier::Fast` migration
	/// `LruSizedHybridStack` emits is unambiguously a genuine
	/// demotion/promotion (no `LfuHybridStack`-style admission-latch
	/// ambiguity — see `drain_demotions`'s doc). The only real
	/// difference is the gauge call itself: `LruSizedHybridStack`
	/// tracks four segments, not two, so `set_lru_sized_hybrid_gauges`
	/// takes the four granular trait-method reads instead of two.
	#[cfg(feature = "lru_sized_hybrid_cache")]
	fn apply_tier_migrations(&mut self) {
		let Some(stack) = &mut self.policy_stack else { return };
		let migrations = stack.drain_tier_migrations();

		if !migrations.is_empty() {
			if let Some(migrate) = &self.tier_migration_fn {
				let (demotions, promotions): (Vec<_>, Vec<_>) = migrations
					.into_iter()
					.partition(|(_, tier)| *tier == Tier::Slow);

				let objects = &self.objects;
				let status = &self.status;

				// Build the destination buffer with NO object-map guard
				// held. `migrate` is a real allocation plus a full byte
				// copy of the value -- a PMEM write on demotion, a PMEM
				// read on promotion -- and at this crate's real object
				// sizes (~16 KB average on the benchmark traces) that is
				// microseconds, not nanoseconds. Holding the shard's
				// *write* guard across it stalls every concurrent `get()`
				// that hashes to the same shard, which surfaces as GET tail
				// latency rather than a uniform slowdown.
				//
				// `Object::data()` is only an `Arc` refcount bump, and the
				// `Arc` keeps the source bytes alive independently of the
				// map, so the snapshot below is safe to use unlocked.
				let apply_physical = |(key, tier): (HashedKey, Tier)| {
					let Some(old_data) = objects.get_ref(&key).map(|object| object.data()) else {
						return;
					};

					let new_data = migrate(&old_data, tier);

					// Re-acquire only to swap the pointer. The `ptr_eq`
					// guard matters: `PaperCache::set()` runs on the API
					// thread and can replace this entry while the copy
					// above is in flight, and writing `new_data` over a
					// *replacement* value would resurrect the bytes of the
					// value it replaced. If the object changed (or was
					// evicted), the migration is stale -- drop it, and let
					// the stack's next event re-derive the correct tier.
					if let Some(mut object) = objects.get_mut_ref(&key) {
						if Arc::ptr_eq(&object.data(), &old_data) {
							object.set_data(new_data);
						}
					}
				};

				demotions.into_iter().for_each(|entry| {
					apply_physical(entry);
					status.record_lru_sized_hybrid_demotion();
				});

				promotions.into_iter().for_each(|entry| {
					apply_physical(entry);
					status.record_lru_sized_hybrid_promotion();
				});
			}
		}
	}

	/// Republishes the active hybrid stack's tier gauges onto
	/// `AtomicStatus`, backing this feature's `*_hybrid_stats()`
	/// accessor.
	///
	/// Split out of `apply_tier_migrations` and called once per pass of
	/// the event loop rather than once per event. These are pure gauges --
	/// a snapshot of state the stack already owns -- so republishing them
	/// after each batch reports exactly the same values as republishing
	/// after each event; only the write frequency changes. That frequency
	/// mattered: it put four virtual calls and four atomic stores into
	/// `AtomicStatus` on the path of every single cache read, and those
	/// stores land in the same struct the API threads are concurrently
	/// incrementing their hit/miss counters in.
	///
	/// Still unconditional (not gated on a migration having happened):
	/// that gate is what let these gauges go stale indefinitely -- see the
	/// note in `apply_tier_migrations` -- and removing it is what fixed
	/// them. This only moves *when* the refresh runs, not *whether*.
	#[cfg(feature = "lru_sized_hybrid_cache")]
	fn refresh_tier_gauges(&mut self) {
		// Refreshed unconditionally -- see the `lru_hybrid_cache` sibling's
		// comment on this same pattern for why gating it on `migrations`
		// being non-empty let these gauges go stale and never catch up.
		if let Some(stack) = &self.policy_stack {
			self.status.set_lru_sized_hybrid_gauges(
				stack.small_fast_bytes_used(),
				stack.large_fast_bytes_used(),
				stack.small_slow_bytes_used(),
				stack.large_slow_bytes_used(),
				stack.small_fast_object_count() as u64,
				stack.large_fast_object_count() as u64,
				stack.small_slow_object_count() as u64,
				stack.large_slow_object_count() as u64,
			);
		}
	}

	/// `s3_fifo_hybrid_cache` counterpart of the methods above — identical
	/// shape to `two_q_hybrid_cache`'s sibling, draining
	/// `S3FifoHybridStack`'s migrations instead and recording to the
	/// `s3_fifo_hybrid_*` counters/gauges on `status`. No `LfuHybridStack`-
	/// style admission-latch ambiguity here (admission always lands in the
	/// one-access queue unconditionally, never fast), so every
	/// `Tier::Slow`/`Tier::Fast` entry is unambiguously a genuine
	/// demotion/promotion — including the ones produced by
	/// `give_second_chance`'s lazy, reference-bit-gated promotion, which is
	/// still just a `Tier::Fast` migration from this method's point of view.
	///
	/// Same inline-sequential, demotions-before-promotions shape as the
	/// `lru_hybrid_cache` sibling above (see its comment): every demotion in
	/// this batch is applied before any promotion in the same batch begins.
	#[cfg(feature = "s3_fifo_hybrid_cache")]
	fn apply_tier_migrations(&mut self) {
		let Some(stack) = &mut self.policy_stack else { return };
		let migrations = stack.drain_tier_migrations();

		if !migrations.is_empty() {
			if let Some(migrate) = &self.tier_migration_fn {
				let (demotions, promotions): (Vec<_>, Vec<_>) = migrations
					.into_iter()
					.partition(|(_, tier)| *tier == Tier::Slow);

				let objects = &self.objects;
				let status = &self.status;

				// Build the destination buffer with NO object-map guard
				// held. `migrate` is a real allocation plus a full byte
				// copy of the value -- a PMEM write on demotion, a PMEM
				// read on promotion -- and at this crate's real object
				// sizes (~16 KB average on the benchmark traces) that is
				// microseconds, not nanoseconds. Holding the shard's
				// *write* guard across it stalls every concurrent `get()`
				// that hashes to the same shard, which surfaces as GET tail
				// latency rather than a uniform slowdown.
				//
				// `Object::data()` is only an `Arc` refcount bump, and the
				// `Arc` keeps the source bytes alive independently of the
				// map, so the snapshot below is safe to use unlocked.
				let apply_physical = |(key, tier): (HashedKey, Tier)| {
					let Some(old_data) = objects.get_ref(&key).map(|object| object.data()) else {
						return;
					};

					let new_data = migrate(&old_data, tier);

					// Re-acquire only to swap the pointer. The `ptr_eq`
					// guard matters: `PaperCache::set()` runs on the API
					// thread and can replace this entry while the copy
					// above is in flight, and writing `new_data` over a
					// *replacement* value would resurrect the bytes of the
					// value it replaced. If the object changed (or was
					// evicted), the migration is stale -- drop it, and let
					// the stack's next event re-derive the correct tier.
					if let Some(mut object) = objects.get_mut_ref(&key) {
						if Arc::ptr_eq(&object.data(), &old_data) {
							object.set_data(new_data);
						}
					}
				};

				demotions.into_iter().for_each(|entry| {
					apply_physical(entry);
					status.record_s3_fifo_hybrid_demotion();
				});

				promotions.into_iter().for_each(|entry| {
					apply_physical(entry);
					status.record_s3_fifo_hybrid_promotion();
				});
			}
		}
	}

	/// Republishes the active hybrid stack's tier gauges onto
	/// `AtomicStatus`, backing this feature's `*_hybrid_stats()`
	/// accessor.
	///
	/// Split out of `apply_tier_migrations` and called once per pass of
	/// the event loop rather than once per event. These are pure gauges --
	/// a snapshot of state the stack already owns -- so republishing them
	/// after each batch reports exactly the same values as republishing
	/// after each event; only the write frequency changes. That frequency
	/// mattered: it put four virtual calls and four atomic stores into
	/// `AtomicStatus` on the path of every single cache read, and those
	/// stores land in the same struct the API threads are concurrently
	/// incrementing their hit/miss counters in.
	///
	/// Still unconditional (not gated on a migration having happened):
	/// that gate is what let these gauges go stale indefinitely -- see the
	/// note in `apply_tier_migrations` -- and removing it is what fixed
	/// them. This only moves *when* the refresh runs, not *whether*.
	#[cfg(feature = "s3_fifo_hybrid_cache")]
	fn refresh_tier_gauges(&mut self) {
		// Refreshed unconditionally -- see the `lru_hybrid_cache` sibling's
		// comment on this same pattern for why gating it on `migrations`
		// being non-empty let these gauges go stale and never catch up.
		if let Some(stack) = &self.policy_stack {
			self.status.set_s3_fifo_hybrid_gauges(
				stack.fast_bytes_used(),
				stack.slow_bytes_used(),
				stack.fast_object_count() as u64,
				stack.slow_object_count() as u64,
			);
		}
	}

	/// `two_q_ghost_hybrid_cache` counterpart of the methods above —
	/// identical shape to `two_q_hybrid_cache`'s sibling. A ghost-queue hit
	/// (`TwoQGhostHybridStack::admit_via_ghost_hit`) produces an ordinary
	/// `Tier::Fast` migration, no different from any other promotion here.
	#[cfg(feature = "two_q_ghost_hybrid_cache")]
	fn apply_tier_migrations(&mut self) {
		let Some(stack) = &mut self.policy_stack else { return };
		let migrations = stack.drain_tier_migrations();

		if !migrations.is_empty() {
			if let Some(migrate) = &self.tier_migration_fn {
				let (demotions, promotions): (Vec<_>, Vec<_>) = migrations
					.into_iter()
					.partition(|(_, tier)| *tier == Tier::Slow);

				let objects = &self.objects;
				let status = &self.status;

				// Build the destination buffer with NO object-map guard
				// held. `migrate` is a real allocation plus a full byte
				// copy of the value -- a PMEM write on demotion, a PMEM
				// read on promotion -- and at this crate's real object
				// sizes (~16 KB average on the benchmark traces) that is
				// microseconds, not nanoseconds. Holding the shard's
				// *write* guard across it stalls every concurrent `get()`
				// that hashes to the same shard, which surfaces as GET tail
				// latency rather than a uniform slowdown.
				//
				// `Object::data()` is only an `Arc` refcount bump, and the
				// `Arc` keeps the source bytes alive independently of the
				// map, so the snapshot below is safe to use unlocked.
				let apply_physical = |(key, tier): (HashedKey, Tier)| {
					let Some(old_data) = objects.get_ref(&key).map(|object| object.data()) else {
						return;
					};

					let new_data = migrate(&old_data, tier);

					// Re-acquire only to swap the pointer. The `ptr_eq`
					// guard matters: `PaperCache::set()` runs on the API
					// thread and can replace this entry while the copy
					// above is in flight, and writing `new_data` over a
					// *replacement* value would resurrect the bytes of the
					// value it replaced. If the object changed (or was
					// evicted), the migration is stale -- drop it, and let
					// the stack's next event re-derive the correct tier.
					if let Some(mut object) = objects.get_mut_ref(&key) {
						if Arc::ptr_eq(&object.data(), &old_data) {
							object.set_data(new_data);
						}
					}
				};

				demotions.into_iter().for_each(|entry| {
					apply_physical(entry);
					status.record_two_q_ghost_hybrid_demotion();
				});

				promotions.into_iter().for_each(|entry| {
					apply_physical(entry);
					status.record_two_q_ghost_hybrid_promotion();
				});
			}
		}
	}

	/// Republishes the active hybrid stack's tier gauges onto
	/// `AtomicStatus`, backing this feature's `*_hybrid_stats()`
	/// accessor.
	///
	/// Split out of `apply_tier_migrations` and called once per pass of
	/// the event loop rather than once per event. These are pure gauges --
	/// a snapshot of state the stack already owns -- so republishing them
	/// after each batch reports exactly the same values as republishing
	/// after each event; only the write frequency changes. That frequency
	/// mattered: it put four virtual calls and four atomic stores into
	/// `AtomicStatus` on the path of every single cache read, and those
	/// stores land in the same struct the API threads are concurrently
	/// incrementing their hit/miss counters in.
	///
	/// Still unconditional (not gated on a migration having happened):
	/// that gate is what let these gauges go stale indefinitely -- see the
	/// note in `apply_tier_migrations` -- and removing it is what fixed
	/// them. This only moves *when* the refresh runs, not *whether*.
	#[cfg(feature = "two_q_ghost_hybrid_cache")]
	fn refresh_tier_gauges(&mut self) {
		if let Some(stack) = &self.policy_stack {
			self.status.set_two_q_ghost_hybrid_gauges(
				stack.fast_bytes_used(),
				stack.slow_bytes_used(),
				stack.fast_object_count() as u64,
				stack.slow_object_count() as u64,
			);
		}
	}

	/// `s3_fifo_ghost_hybrid_cache` counterpart of the methods above —
	/// identical shape to `s3_fifo_hybrid_cache`'s sibling, including the
	/// eviction-sweep-triggered `give_second_chance` promotions (see that
	/// sibling's doc). A ghost-queue hit produces an ordinary `Tier::Fast`
	/// migration, no different from any other promotion here.
	#[cfg(feature = "s3_fifo_ghost_hybrid_cache")]
	fn apply_tier_migrations(&mut self) {
		let Some(stack) = &mut self.policy_stack else { return };
		let migrations = stack.drain_tier_migrations();

		if !migrations.is_empty() {
			if let Some(migrate) = &self.tier_migration_fn {
				let (demotions, promotions): (Vec<_>, Vec<_>) = migrations
					.into_iter()
					.partition(|(_, tier)| *tier == Tier::Slow);

				let objects = &self.objects;
				let status = &self.status;

				// Build the destination buffer with NO object-map guard
				// held. `migrate` is a real allocation plus a full byte
				// copy of the value -- a PMEM write on demotion, a PMEM
				// read on promotion -- and at this crate's real object
				// sizes (~16 KB average on the benchmark traces) that is
				// microseconds, not nanoseconds. Holding the shard's
				// *write* guard across it stalls every concurrent `get()`
				// that hashes to the same shard, which surfaces as GET tail
				// latency rather than a uniform slowdown.
				//
				// `Object::data()` is only an `Arc` refcount bump, and the
				// `Arc` keeps the source bytes alive independently of the
				// map, so the snapshot below is safe to use unlocked.
				let apply_physical = |(key, tier): (HashedKey, Tier)| {
					let Some(old_data) = objects.get_ref(&key).map(|object| object.data()) else {
						return;
					};

					let new_data = migrate(&old_data, tier);

					// Re-acquire only to swap the pointer. The `ptr_eq`
					// guard matters: `PaperCache::set()` runs on the API
					// thread and can replace this entry while the copy
					// above is in flight, and writing `new_data` over a
					// *replacement* value would resurrect the bytes of the
					// value it replaced. If the object changed (or was
					// evicted), the migration is stale -- drop it, and let
					// the stack's next event re-derive the correct tier.
					if let Some(mut object) = objects.get_mut_ref(&key) {
						if Arc::ptr_eq(&object.data(), &old_data) {
							object.set_data(new_data);
						}
					}
				};

				demotions.into_iter().for_each(|entry| {
					apply_physical(entry);
					status.record_s3_fifo_ghost_hybrid_demotion();
				});

				promotions.into_iter().for_each(|entry| {
					apply_physical(entry);
					status.record_s3_fifo_ghost_hybrid_promotion();
				});
			}
		}
	}

	/// Republishes the active hybrid stack's tier gauges onto
	/// `AtomicStatus`, backing this feature's `*_hybrid_stats()`
	/// accessor.
	///
	/// Split out of `apply_tier_migrations` and called once per pass of
	/// the event loop rather than once per event. These are pure gauges --
	/// a snapshot of state the stack already owns -- so republishing them
	/// after each batch reports exactly the same values as republishing
	/// after each event; only the write frequency changes. That frequency
	/// mattered: it put four virtual calls and four atomic stores into
	/// `AtomicStatus` on the path of every single cache read, and those
	/// stores land in the same struct the API threads are concurrently
	/// incrementing their hit/miss counters in.
	///
	/// Still unconditional (not gated on a migration having happened):
	/// that gate is what let these gauges go stale indefinitely -- see the
	/// note in `apply_tier_migrations` -- and removing it is what fixed
	/// them. This only moves *when* the refresh runs, not *whether*.
	#[cfg(feature = "s3_fifo_ghost_hybrid_cache")]
	fn refresh_tier_gauges(&mut self) {
		if let Some(stack) = &self.policy_stack {
			self.status.set_s3_fifo_ghost_hybrid_gauges(
				stack.fast_bytes_used(),
				stack.slow_bytes_used(),
				stack.fast_object_count() as u64,
				stack.slow_object_count() as u64,
			);
		}
	}

	/// `s3_fifo_ghost_lazy_demotion_hybrid_cache` counterpart of the method
	/// above — identical shape, since the demotion-time reference-bit gate
	/// this variant adds is entirely internal to the stack's own
	/// `settle_fast_tier` (see that stack's module doc): the migrations this
	/// drains and the way they're applied physically are unchanged, only
	/// the stack's *decision* of which key ends up in `demotions` differs.
	#[cfg(feature = "s3_fifo_ghost_lazy_demotion_hybrid_cache")]
	fn apply_tier_migrations(&mut self) {
		let Some(stack) = &mut self.policy_stack else { return };
		let migrations = stack.drain_tier_migrations();

		if !migrations.is_empty() {
			if let Some(migrate) = &self.tier_migration_fn {
				let (demotions, promotions): (Vec<_>, Vec<_>) = migrations
					.into_iter()
					.partition(|(_, tier)| *tier == Tier::Slow);

				let objects = &self.objects;
				let status = &self.status;

				// Build the destination buffer with NO object-map guard
				// held. `migrate` is a real allocation plus a full byte
				// copy of the value -- a PMEM write on demotion, a PMEM
				// read on promotion -- and at this crate's real object
				// sizes (~16 KB average on the benchmark traces) that is
				// microseconds, not nanoseconds. Holding the shard's
				// *write* guard across it stalls every concurrent `get()`
				// that hashes to the same shard, which surfaces as GET tail
				// latency rather than a uniform slowdown.
				//
				// `Object::data()` is only an `Arc` refcount bump, and the
				// `Arc` keeps the source bytes alive independently of the
				// map, so the snapshot below is safe to use unlocked.
				let apply_physical = |(key, tier): (HashedKey, Tier)| {
					let Some(old_data) = objects.get_ref(&key).map(|object| object.data()) else {
						return;
					};

					let new_data = migrate(&old_data, tier);

					// Re-acquire only to swap the pointer. The `ptr_eq`
					// guard matters: `PaperCache::set()` runs on the API
					// thread and can replace this entry while the copy
					// above is in flight, and writing `new_data` over a
					// *replacement* value would resurrect the bytes of the
					// value it replaced. If the object changed (or was
					// evicted), the migration is stale -- drop it, and let
					// the stack's next event re-derive the correct tier.
					if let Some(mut object) = objects.get_mut_ref(&key) {
						if Arc::ptr_eq(&object.data(), &old_data) {
							object.set_data(new_data);
						}
					}
				};

				demotions.into_iter().for_each(|entry| {
					apply_physical(entry);
					status.record_s3_fifo_ghost_lazy_demotion_hybrid_demotion();
				});

				promotions.into_iter().for_each(|entry| {
					apply_physical(entry);
					status.record_s3_fifo_ghost_lazy_demotion_hybrid_promotion();
				});
			}
		}
	}

	/// Republishes the active hybrid stack's tier gauges onto
	/// `AtomicStatus`, backing this feature's `*_hybrid_stats()`
	/// accessor.
	///
	/// Split out of `apply_tier_migrations` and called once per pass of
	/// the event loop rather than once per event. These are pure gauges --
	/// a snapshot of state the stack already owns -- so republishing them
	/// after each batch reports exactly the same values as republishing
	/// after each event; only the write frequency changes. That frequency
	/// mattered: it put four virtual calls and four atomic stores into
	/// `AtomicStatus` on the path of every single cache read, and those
	/// stores land in the same struct the API threads are concurrently
	/// incrementing their hit/miss counters in.
	///
	/// Still unconditional (not gated on a migration having happened):
	/// that gate is what let these gauges go stale indefinitely -- see the
	/// note in `apply_tier_migrations` -- and removing it is what fixed
	/// them. This only moves *when* the refresh runs, not *whether*.
	#[cfg(feature = "s3_fifo_ghost_lazy_demotion_hybrid_cache")]
	fn refresh_tier_gauges(&mut self) {
		if let Some(stack) = &self.policy_stack {
			self.status.set_s3_fifo_ghost_lazy_demotion_hybrid_gauges(
				stack.fast_bytes_used(),
				stack.slow_bytes_used(),
				stack.fast_object_count() as u64,
				stack.slow_object_count() as u64,
			);
		}
	}

	/// `s3_fifo_ghost_lazy_demotion_fast_admission_hybrid_cache`
	/// counterpart of the method above — identical shape. The only
	/// difference this variant makes (the one-access queue's bytes are
	/// already Fast at admission time, so `promote_from_one_access`/
	/// `admit_via_ghost_hit` no longer produce a Fast migration for the
	/// common case -- see that stack's module doc) is entirely internal to
	/// `drain_tier_migrations`'s own output; the physical-apply/counting
	/// logic here is unchanged.
	#[cfg(feature = "s3_fifo_ghost_lazy_demotion_fast_admission_hybrid_cache")]
	fn apply_tier_migrations(&mut self) {
		let Some(stack) = &mut self.policy_stack else { return };
		let migrations = stack.drain_tier_migrations();

		if !migrations.is_empty() {
			if let Some(migrate) = &self.tier_migration_fn {
				let (demotions, promotions): (Vec<_>, Vec<_>) = migrations
					.into_iter()
					.partition(|(_, tier)| *tier == Tier::Slow);

				let objects = &self.objects;
				let status = &self.status;

				// Build the destination buffer with NO object-map guard
				// held. `migrate` is a real allocation plus a full byte
				// copy of the value -- a PMEM write on demotion, a PMEM
				// read on promotion -- and at this crate's real object
				// sizes (~16 KB average on the benchmark traces) that is
				// microseconds, not nanoseconds. Holding the shard's
				// *write* guard across it stalls every concurrent `get()`
				// that hashes to the same shard, which surfaces as GET tail
				// latency rather than a uniform slowdown.
				//
				// `Object::data()` is only an `Arc` refcount bump, and the
				// `Arc` keeps the source bytes alive independently of the
				// map, so the snapshot below is safe to use unlocked.
				let apply_physical = |(key, tier): (HashedKey, Tier)| {
					let Some(old_data) = objects.get_ref(&key).map(|object| object.data()) else {
						return;
					};

					let new_data = migrate(&old_data, tier);

					// Re-acquire only to swap the pointer. The `ptr_eq`
					// guard matters: `PaperCache::set()` runs on the API
					// thread and can replace this entry while the copy
					// above is in flight, and writing `new_data` over a
					// *replacement* value would resurrect the bytes of the
					// value it replaced. If the object changed (or was
					// evicted), the migration is stale -- drop it, and let
					// the stack's next event re-derive the correct tier.
					if let Some(mut object) = objects.get_mut_ref(&key) {
						if Arc::ptr_eq(&object.data(), &old_data) {
							object.set_data(new_data);
						}
					}
				};

				demotions.into_iter().for_each(|entry| {
					apply_physical(entry);
					status.record_s3_fifo_ghost_lazy_demotion_fast_admission_hybrid_demotion();
				});

				promotions.into_iter().for_each(|entry| {
					apply_physical(entry);
					status.record_s3_fifo_ghost_lazy_demotion_fast_admission_hybrid_promotion();
				});
			}
		}
	}

	/// Republishes the active hybrid stack's tier gauges onto
	/// `AtomicStatus`, backing this feature's `*_hybrid_stats()`
	/// accessor.
	///
	/// Split out of `apply_tier_migrations` and called once per pass of
	/// the event loop rather than once per event. These are pure gauges --
	/// a snapshot of state the stack already owns -- so republishing them
	/// after each batch reports exactly the same values as republishing
	/// after each event; only the write frequency changes. That frequency
	/// mattered: it put four virtual calls and four atomic stores into
	/// `AtomicStatus` on the path of every single cache read, and those
	/// stores land in the same struct the API threads are concurrently
	/// incrementing their hit/miss counters in.
	///
	/// Still unconditional (not gated on a migration having happened):
	/// that gate is what let these gauges go stale indefinitely -- see the
	/// note in `apply_tier_migrations` -- and removing it is what fixed
	/// them. This only moves *when* the refresh runs, not *whether*.
	#[cfg(feature = "s3_fifo_ghost_lazy_demotion_fast_admission_hybrid_cache")]
	fn refresh_tier_gauges(&mut self) {
		if let Some(stack) = &self.policy_stack {
			self.status.set_s3_fifo_ghost_lazy_demotion_fast_admission_hybrid_gauges(
				stack.fast_bytes_used(),
				stack.slow_bytes_used(),
				stack.fast_object_count() as u64,
				stack.slow_object_count() as u64,
			);
		}
	}

	/// `s3_fifo_ghost_lazy_demotion_fast_admission_midpoint_hybrid_cache`
	/// counterpart of the method above — identical shape. The new
	/// mid-slow-segment checkpoint this variant adds only decides *which*
	/// key ends up in a migration (via the stack's own `settle_fast_tier`/
	/// `give_second_chance`/`check_slow_midpoint`), not how migrations get
	/// drained, applied, or counted here.
	#[cfg(feature = "s3_fifo_ghost_lazy_demotion_fast_admission_midpoint_hybrid_cache")]
	fn apply_tier_migrations(&mut self) {
		let Some(stack) = &mut self.policy_stack else { return };
		let migrations = stack.drain_tier_migrations();

		if !migrations.is_empty() {
			if let Some(migrate) = &self.tier_migration_fn {
				let (demotions, promotions): (Vec<_>, Vec<_>) = migrations
					.into_iter()
					.partition(|(_, tier)| *tier == Tier::Slow);

				let objects = &self.objects;
				let status = &self.status;

				// Build the destination buffer with NO object-map guard
				// held. `migrate` is a real allocation plus a full byte
				// copy of the value -- a PMEM write on demotion, a PMEM
				// read on promotion -- and at this crate's real object
				// sizes (~16 KB average on the benchmark traces) that is
				// microseconds, not nanoseconds. Holding the shard's
				// *write* guard across it stalls every concurrent `get()`
				// that hashes to the same shard, which surfaces as GET tail
				// latency rather than a uniform slowdown.
				//
				// `Object::data()` is only an `Arc` refcount bump, and the
				// `Arc` keeps the source bytes alive independently of the
				// map, so the snapshot below is safe to use unlocked.
				let apply_physical = |(key, tier): (HashedKey, Tier)| {
					let Some(old_data) = objects.get_ref(&key).map(|object| object.data()) else {
						return;
					};

					let new_data = migrate(&old_data, tier);

					// Re-acquire only to swap the pointer. The `ptr_eq`
					// guard matters: `PaperCache::set()` runs on the API
					// thread and can replace this entry while the copy
					// above is in flight, and writing `new_data` over a
					// *replacement* value would resurrect the bytes of the
					// value it replaced. If the object changed (or was
					// evicted), the migration is stale -- drop it, and let
					// the stack's next event re-derive the correct tier.
					if let Some(mut object) = objects.get_mut_ref(&key) {
						if Arc::ptr_eq(&object.data(), &old_data) {
							object.set_data(new_data);
						}
					}
				};

				demotions.into_iter().for_each(|entry| {
					apply_physical(entry);
					status.record_s3_fifo_ghost_lazy_demotion_fast_admission_midpoint_hybrid_demotion();
				});

				promotions.into_iter().for_each(|entry| {
					apply_physical(entry);
					status.record_s3_fifo_ghost_lazy_demotion_fast_admission_midpoint_hybrid_promotion();
				});
			}
		}
	}

	/// Republishes the active hybrid stack's tier gauges onto
	/// `AtomicStatus`, backing this feature's `*_hybrid_stats()`
	/// accessor.
	///
	/// Split out of `apply_tier_migrations` and called once per pass of
	/// the event loop rather than once per event. These are pure gauges --
	/// a snapshot of state the stack already owns -- so republishing them
	/// after each batch reports exactly the same values as republishing
	/// after each event; only the write frequency changes. That frequency
	/// mattered: it put four virtual calls and four atomic stores into
	/// `AtomicStatus` on the path of every single cache read, and those
	/// stores land in the same struct the API threads are concurrently
	/// incrementing their hit/miss counters in.
	///
	/// Still unconditional (not gated on a migration having happened):
	/// that gate is what let these gauges go stale indefinitely -- see the
	/// note in `apply_tier_migrations` -- and removing it is what fixed
	/// them. This only moves *when* the refresh runs, not *whether*.
	#[cfg(feature = "s3_fifo_ghost_lazy_demotion_fast_admission_midpoint_hybrid_cache")]
	fn refresh_tier_gauges(&mut self) {
		if let Some(stack) = &self.policy_stack {
			self.status.set_s3_fifo_ghost_lazy_demotion_fast_admission_midpoint_hybrid_gauges(
				stack.fast_bytes_used(),
				stack.slow_bytes_used(),
				stack.fast_object_count() as u64,
				stack.slow_object_count() as u64,
			);
		}
	}

	/// Same shape as the `s3_fifo_ghost_lazy_demotion_fast_admission_midpoint_hybrid_cache`
	/// sibling above -- this variant's differences (no ghost queue, the
	/// one-access queue's own capacity pressure settled synchronously
	/// instead of through eviction) are entirely internal to the stack;
	/// migrations still drain, apply, and count exactly the same way here.
	#[cfg(feature = "s3_fifo_lazy_demotion_fast_admission_midpoint_reprieve_hybrid_cache")]
	fn apply_tier_migrations(&mut self) {
		let Some(stack) = &mut self.policy_stack else { return };
		let migrations = stack.drain_tier_migrations();

		if !migrations.is_empty() {
			if let Some(migrate) = &self.tier_migration_fn {
				let (demotions, promotions): (Vec<_>, Vec<_>) = migrations
					.into_iter()
					.partition(|(_, tier)| *tier == Tier::Slow);

				let objects = &self.objects;
				let status = &self.status;

				// Build the destination buffer with NO object-map guard
				// held. `migrate` is a real allocation plus a full byte
				// copy of the value -- a PMEM write on demotion, a PMEM
				// read on promotion -- and at this crate's real object
				// sizes (~16 KB average on the benchmark traces) that is
				// microseconds, not nanoseconds. Holding the shard's
				// *write* guard across it stalls every concurrent `get()`
				// that hashes to the same shard, which surfaces as GET tail
				// latency rather than a uniform slowdown.
				//
				// `Object::data()` is only an `Arc` refcount bump, and the
				// `Arc` keeps the source bytes alive independently of the
				// map, so the snapshot below is safe to use unlocked.
				let apply_physical = |(key, tier): (HashedKey, Tier)| {
					let Some(old_data) = objects.get_ref(&key).map(|object| object.data()) else {
						return;
					};

					let new_data = migrate(&old_data, tier);

					// Re-acquire only to swap the pointer. The `ptr_eq`
					// guard matters: `PaperCache::set()` runs on the API
					// thread and can replace this entry while the copy
					// above is in flight, and writing `new_data` over a
					// *replacement* value would resurrect the bytes of the
					// value it replaced. If the object changed (or was
					// evicted), the migration is stale -- drop it, and let
					// the stack's next event re-derive the correct tier.
					if let Some(mut object) = objects.get_mut_ref(&key) {
						if Arc::ptr_eq(&object.data(), &old_data) {
							object.set_data(new_data);
						}
					}
				};

				demotions.into_iter().for_each(|entry| {
					apply_physical(entry);
					status.record_s3_fifo_lazy_demotion_fast_admission_midpoint_reprieve_hybrid_demotion();
				});

				promotions.into_iter().for_each(|entry| {
					apply_physical(entry);
					status.record_s3_fifo_lazy_demotion_fast_admission_midpoint_reprieve_hybrid_promotion();
				});
			}
		}
	}

	/// Republishes the active hybrid stack's tier gauges onto
	/// `AtomicStatus`, backing this feature's `*_hybrid_stats()`
	/// accessor.
	///
	/// Split out of `apply_tier_migrations` and called once per pass of
	/// the event loop rather than once per event. These are pure gauges --
	/// a snapshot of state the stack already owns -- so republishing them
	/// after each batch reports exactly the same values as republishing
	/// after each event; only the write frequency changes. That frequency
	/// mattered: it put four virtual calls and four atomic stores into
	/// `AtomicStatus` on the path of every single cache read, and those
	/// stores land in the same struct the API threads are concurrently
	/// incrementing their hit/miss counters in.
	///
	/// Still unconditional (not gated on a migration having happened):
	/// that gate is what let these gauges go stale indefinitely -- see the
	/// note in `apply_tier_migrations` -- and removing it is what fixed
	/// them. This only moves *when* the refresh runs, not *whether*.
	#[cfg(feature = "s3_fifo_lazy_demotion_fast_admission_midpoint_reprieve_hybrid_cache")]
	fn refresh_tier_gauges(&mut self) {
		if let Some(stack) = &self.policy_stack {
			self.status.set_s3_fifo_lazy_demotion_fast_admission_midpoint_reprieve_hybrid_gauges(
				stack.fast_bytes_used(),
				stack.slow_bytes_used(),
				stack.fast_object_count() as u64,
				stack.slow_object_count() as u64,
			);
		}
	}

	#[cfg(feature = "s3_fifo_lazy_demotion_fast_admission_reprieve_hybrid_cache")]
	fn apply_tier_migrations(&mut self) {
		let Some(stack) = &mut self.policy_stack else { return };
		let migrations = stack.drain_tier_migrations();

		if !migrations.is_empty() {
			if let Some(migrate) = &self.tier_migration_fn {
				let (demotions, promotions): (Vec<_>, Vec<_>) = migrations
					.into_iter()
					.partition(|(_, tier)| *tier == Tier::Slow);

				let objects = &self.objects;
				let status = &self.status;

				// Build the destination buffer with NO object-map guard
				// held. `migrate` is a real allocation plus a full byte
				// copy of the value -- a PMEM write on demotion, a PMEM
				// read on promotion -- and at this crate's real object
				// sizes (~16 KB average on the benchmark traces) that is
				// microseconds, not nanoseconds. Holding the shard's
				// *write* guard across it stalls every concurrent `get()`
				// that hashes to the same shard, which surfaces as GET tail
				// latency rather than a uniform slowdown.
				//
				// `Object::data()` is only an `Arc` refcount bump, and the
				// `Arc` keeps the source bytes alive independently of the
				// map, so the snapshot below is safe to use unlocked.
				let apply_physical = |(key, tier): (HashedKey, Tier)| {
					let Some(old_data) = objects.get_ref(&key).map(|object| object.data()) else {
						return;
					};

					let new_data = migrate(&old_data, tier);

					// Re-acquire only to swap the pointer. The `ptr_eq`
					// guard matters: `PaperCache::set()` runs on the API
					// thread and can replace this entry while the copy
					// above is in flight, and writing `new_data` over a
					// *replacement* value would resurrect the bytes of the
					// value it replaced. If the object changed (or was
					// evicted), the migration is stale -- drop it, and let
					// the stack's next event re-derive the correct tier.
					if let Some(mut object) = objects.get_mut_ref(&key) {
						if Arc::ptr_eq(&object.data(), &old_data) {
							object.set_data(new_data);
						}
					}
				};

				demotions.into_iter().for_each(|entry| {
					apply_physical(entry);
					status.record_s3_fifo_lazy_demotion_fast_admission_reprieve_hybrid_demotion();
				});

				promotions.into_iter().for_each(|entry| {
					apply_physical(entry);
					status.record_s3_fifo_lazy_demotion_fast_admission_reprieve_hybrid_promotion();
				});
			}
		}
	}

	#[cfg(feature = "s3_fifo_lazy_demotion_reprieve_hybrid_cache")]
	fn apply_tier_migrations(&mut self) {
		let Some(stack) = &mut self.policy_stack else { return };
		let migrations = stack.drain_tier_migrations();

		if !migrations.is_empty() {
			if let Some(migrate) = &self.tier_migration_fn {
				let (demotions, promotions): (Vec<_>, Vec<_>) = migrations
					.into_iter()
					.partition(|(_, tier)| *tier == Tier::Slow);

				let objects = &self.objects;
				let status = &self.status;

				// Build the destination buffer with NO object-map guard
				// held. `migrate` is a real allocation plus a full byte
				// copy of the value -- a PMEM write on demotion, a PMEM
				// read on promotion -- and at this crate's real object
				// sizes (~16 KB average on the benchmark traces) that is
				// microseconds, not nanoseconds. Holding the shard's
				// *write* guard across it stalls every concurrent `get()`
				// that hashes to the same shard, which surfaces as GET tail
				// latency rather than a uniform slowdown.
				//
				// `Object::data()` is only an `Arc` refcount bump, and the
				// `Arc` keeps the source bytes alive independently of the
				// map, so the snapshot below is safe to use unlocked.
				let apply_physical = |(key, tier): (HashedKey, Tier)| {
					let Some(old_data) = objects.get_ref(&key).map(|object| object.data()) else {
						return;
					};

					let new_data = migrate(&old_data, tier);

					// Re-acquire only to swap the pointer. The `ptr_eq`
					// guard matters: `PaperCache::set()` runs on the API
					// thread and can replace this entry while the copy
					// above is in flight, and writing `new_data` over a
					// *replacement* value would resurrect the bytes of the
					// value it replaced. If the object changed (or was
					// evicted), the migration is stale -- drop it, and let
					// the stack's next event re-derive the correct tier.
					if let Some(mut object) = objects.get_mut_ref(&key) {
						if Arc::ptr_eq(&object.data(), &old_data) {
							object.set_data(new_data);
						}
					}
				};

				demotions.into_iter().for_each(|entry| {
					apply_physical(entry);
					status.record_s3_fifo_lazy_demotion_reprieve_hybrid_demotion();
				});

				promotions.into_iter().for_each(|entry| {
					apply_physical(entry);
					status.record_s3_fifo_lazy_demotion_reprieve_hybrid_promotion();
				});
			}
		}
	}

	/// Republishes the active hybrid stack's tier gauges onto
	/// `AtomicStatus`, backing this feature's `*_hybrid_stats()`
	/// accessor.
	///
	/// Split out of `apply_tier_migrations` and called once per pass of
	/// the event loop rather than once per event. These are pure gauges --
	/// a snapshot of state the stack already owns -- so republishing them
	/// after each batch reports exactly the same values as republishing
	/// after each event; only the write frequency changes. That frequency
	/// mattered: it put four virtual calls and four atomic stores into
	/// `AtomicStatus` on the path of every single cache read, and those
	/// stores land in the same struct the API threads are concurrently
	/// incrementing their hit/miss counters in.
	///
	/// Still unconditional (not gated on a migration having happened):
	/// that gate is what let these gauges go stale indefinitely -- see the
	/// note in `apply_tier_migrations` -- and removing it is what fixed
	/// them. This only moves *when* the refresh runs, not *whether*.
	#[cfg(feature = "s3_fifo_lazy_demotion_fast_admission_reprieve_hybrid_cache")]
	fn refresh_tier_gauges(&mut self) {
		if let Some(stack) = &self.policy_stack {
			self.status.set_s3_fifo_lazy_demotion_fast_admission_reprieve_hybrid_gauges(
				stack.fast_bytes_used(),
				stack.slow_bytes_used(),
				stack.fast_object_count() as u64,
				stack.slow_object_count() as u64,
			);
		}
	}

	#[cfg(feature = "s3_fifo_lazy_demotion_reprieve_hybrid_cache")]
	fn refresh_tier_gauges(&mut self) {
		if let Some(stack) = &self.policy_stack {
			self.status.set_s3_fifo_lazy_demotion_reprieve_hybrid_gauges(
				stack.fast_bytes_used(),
				stack.slow_bytes_used(),
				stack.fast_object_count() as u64,
				stack.slow_object_count() as u64,
			);
		}
	}

	#[cfg(feature = "s3_fifo_lazy_demotion_fast_admission_split_slow_reprieve_hybrid_cache")]
	fn apply_tier_migrations(&mut self) {
		let Some(stack) = &mut self.policy_stack else { return };
		let migrations = stack.drain_tier_migrations();

		if !migrations.is_empty() {
			if let Some(migrate) = &self.tier_migration_fn {
				let (demotions, promotions): (Vec<_>, Vec<_>) = migrations
					.into_iter()
					.partition(|(_, tier)| *tier == Tier::Slow);

				let objects = &self.objects;
				let status = &self.status;

				// Build the destination buffer with NO object-map guard
				// held. `migrate` is a real allocation plus a full byte
				// copy of the value -- a PMEM write on demotion, a PMEM
				// read on promotion -- and at this crate's real object
				// sizes (~16 KB average on the benchmark traces) that is
				// microseconds, not nanoseconds. Holding the shard's
				// *write* guard across it stalls every concurrent `get()`
				// that hashes to the same shard, which surfaces as GET tail
				// latency rather than a uniform slowdown.
				//
				// `Object::data()` is only an `Arc` refcount bump, and the
				// `Arc` keeps the source bytes alive independently of the
				// map, so the snapshot below is safe to use unlocked.
				let apply_physical = |(key, tier): (HashedKey, Tier)| {
					let Some(old_data) = objects.get_ref(&key).map(|object| object.data()) else {
						return;
					};

					let new_data = migrate(&old_data, tier);

					// Re-acquire only to swap the pointer. The `ptr_eq`
					// guard matters: `PaperCache::set()` runs on the API
					// thread and can replace this entry while the copy
					// above is in flight, and writing `new_data` over a
					// *replacement* value would resurrect the bytes of the
					// value it replaced. If the object changed (or was
					// evicted), the migration is stale -- drop it, and let
					// the stack's next event re-derive the correct tier.
					if let Some(mut object) = objects.get_mut_ref(&key) {
						if Arc::ptr_eq(&object.data(), &old_data) {
							object.set_data(new_data);
						}
					}
				};

				demotions.into_iter().for_each(|entry| {
					apply_physical(entry);
					status.record_s3_fifo_lazy_demotion_fast_admission_split_slow_reprieve_hybrid_demotion();
				});

				promotions.into_iter().for_each(|entry| {
					apply_physical(entry);
					status.record_s3_fifo_lazy_demotion_fast_admission_split_slow_reprieve_hybrid_promotion();
				});
			}
		}
	}

	/// Republishes the active hybrid stack's tier gauges onto
	/// `AtomicStatus`, backing this feature's `*_hybrid_stats()`
	/// accessor.
	///
	/// Split out of `apply_tier_migrations` and called once per pass of
	/// the event loop rather than once per event. These are pure gauges --
	/// a snapshot of state the stack already owns -- so republishing them
	/// after each batch reports exactly the same values as republishing
	/// after each event; only the write frequency changes. That frequency
	/// mattered: it put four virtual calls and four atomic stores into
	/// `AtomicStatus` on the path of every single cache read, and those
	/// stores land in the same struct the API threads are concurrently
	/// incrementing their hit/miss counters in.
	///
	/// Still unconditional (not gated on a migration having happened):
	/// that gate is what let these gauges go stale indefinitely -- see the
	/// note in `apply_tier_migrations` -- and removing it is what fixed
	/// them. This only moves *when* the refresh runs, not *whether*.
	#[cfg(feature = "s3_fifo_lazy_demotion_fast_admission_split_slow_reprieve_hybrid_cache")]
	fn refresh_tier_gauges(&mut self) {
		if let Some(stack) = &self.policy_stack {
			self.status.set_s3_fifo_lazy_demotion_fast_admission_split_slow_reprieve_hybrid_gauges(
				stack.fast_bytes_used(),
				stack.slow_bytes_used(),
				stack.fast_object_count() as u64,
				stack.slow_object_count() as u64,
			);
		}
	}

	fn apply_buffered_events(
		&mut self,
		buffered_events: &[StackEvent],
		policy_reconstruct_rx: &Receiver<Box<dyn PolicyStack>>,
	) {
		for mut stack in policy_reconstruct_rx.try_iter() {
			for event in buffered_events {
				match event {
					StackEvent::Get(key) => stack.update(*key),
					StackEvent::Set(key, size) => stack.insert(*key, *size),
					StackEvent::Del(key) => stack.remove(*key),
					StackEvent::Wipe => stack.clear(),
					StackEvent::Resize(size) => stack.resize(*size),

					// Never actually buffered -- `Shutdown` is sent
					// directly to `trace_worker`, not derived from a
					// `WorkerEvent` via `maybe_from_worker_event` (the only
					// thing that populates `buffered_events`). Exhaustive
					// match still needs an arm.
					StackEvent::Shutdown => {},
				}
			}

			info!("Policy switch complete");

			self.policy_stack = Some(stack);
			self.mini_index = None;
		}
	}

	fn flush_buffered_events(
		&self,
		buffered_events: &mut Vec<StackEvent>,
	) -> Result<(), CacheError> {
		// Nothing ever buffers when tracing is off (both producers -- the run
		// loop's stack-event derivation and `apply_evictions`' eviction
		// record -- are gated on the same `Option`), so this is a plain
		// no-op rather than a silent drop.
		let Some(trace_worker) = &self.trace_worker else {
			return Ok(());
		};

		if self.mini_index.is_some() {
			// the mini policy is still running so stack events should be buffered
			// until the full stack is reconstructed
			return Ok(());
		}

		for event in buffered_events.iter() {
			if let Err(err) = trace_worker.send(event.clone()) {
				error!("Could not send buffered event to trace worker: {err:?}");
				return Err(CacheError::Internal);
			}
		}

		buffered_events.clear();

		Ok(())
	}

	fn apply_evictions(
		&mut self,
		buffered_events: &mut Vec<StackEvent>,
	) -> Result<(), CacheError> {
		if let Some(index) = self.mini_index {
			self.apply_mini_evictions(index, buffered_events);
			return Ok(());
		}

		let policy = self.current_policy.read();
		let max_cache_size = self.status.max_size();

		loop {
			let over_max_size = self.status.used_size(&policy) > max_cache_size;

			// `len() > 0` guards against ever looping forever on a stack
			// whose `needs_capacity_eviction` stays true despite having
			// nothing left to evict (which would indicate an accounting
			// bug in the stack, not a real pending eviction).
			let needs_capacity_eviction = self.policy_stack.as_ref()
				.is_some_and(|stack| stack.len() > 0 && stack.needs_capacity_eviction());

			if !over_max_size && !needs_capacity_eviction {
				break;
			}

			let Some(policy_stack) = self.policy_stack.as_mut() else {
				error!("No active policy or mini stack");
				return Err(CacheError::Internal);
			};

			let maybe_key = policy_stack
				.evict_one()
				.map(|key| EraseKey::Hashed(key));

			let erase_result = erase(
				&self.objects,
				&self.status,
				&self.overhead_manager,
				maybe_key,
			);

			let Ok((key, _evicted_obj)) = erase_result else {
				continue;
			};

			#[cfg(feature = "lru_hybrid_cache")]
			if *policy == PaperPolicy::LruHybrid {
				self.status.record_lru_hybrid_eviction();
			}

			#[cfg(feature = "lfu_hybrid_cache")]
			if *policy == PaperPolicy::LfuHybrid {
				self.status.record_lfu_hybrid_eviction();
			}

			#[cfg(feature = "two_q_hybrid_cache")]
			if matches!(*policy, PaperPolicy::TwoQHybrid(_)) {
				self.status.record_two_q_hybrid_eviction();
			}

			#[cfg(feature = "two_q_fast_admission_hybrid_cache")]
			if matches!(*policy, PaperPolicy::TwoQFastAdmissionHybrid(_)) {
				self.status.record_two_q_fast_admission_hybrid_eviction();
			}

			#[cfg(feature = "two_q_fast_admission_reprieve_hybrid_cache")]
			if matches!(*policy, PaperPolicy::TwoQFastAdmissionReprieveHybrid(_)) {
				self.status.record_two_q_fast_admission_reprieve_hybrid_eviction();
			}

			#[cfg(feature = "fifo_hybrid_cache")]
			if *policy == PaperPolicy::FifoHybrid {
				self.status.record_fifo_hybrid_eviction();
			}

			#[cfg(feature = "lru_sized_hybrid_cache")]
			if *policy == PaperPolicy::LruSizedHybrid {
				self.status.record_lru_sized_hybrid_eviction();
			}

			#[cfg(feature = "s3_fifo_hybrid_cache")]
			if matches!(*policy, PaperPolicy::S3FifoHybrid(_)) {
				self.status.record_s3_fifo_hybrid_eviction();
			}

			#[cfg(feature = "two_q_ghost_hybrid_cache")]
			if matches!(*policy, PaperPolicy::TwoQGhostHybrid(_)) {
				self.status.record_two_q_ghost_hybrid_eviction();
			}

			#[cfg(feature = "s3_fifo_ghost_hybrid_cache")]
			if matches!(*policy, PaperPolicy::S3FifoGhostHybrid(_)) {
				self.status.record_s3_fifo_ghost_hybrid_eviction();
			}

			#[cfg(feature = "s3_fifo_ghost_lazy_demotion_hybrid_cache")]
			if matches!(*policy, PaperPolicy::S3FifoGhostLazyDemotionHybrid(_)) {
				self.status.record_s3_fifo_ghost_lazy_demotion_hybrid_eviction();
			}

			#[cfg(feature = "s3_fifo_ghost_lazy_demotion_fast_admission_hybrid_cache")]
			if matches!(*policy, PaperPolicy::S3FifoGhostLazyDemotionFastAdmissionHybrid(_)) {
				self.status.record_s3_fifo_ghost_lazy_demotion_fast_admission_hybrid_eviction();
			}

			#[cfg(feature = "s3_fifo_ghost_lazy_demotion_fast_admission_midpoint_hybrid_cache")]
			if matches!(*policy, PaperPolicy::S3FifoGhostLazyDemotionFastAdmissionMidpointHybrid(_)) {
				self.status.record_s3_fifo_ghost_lazy_demotion_fast_admission_midpoint_hybrid_eviction();
			}

			#[cfg(feature = "s3_fifo_lazy_demotion_fast_admission_midpoint_reprieve_hybrid_cache")]
			if matches!(*policy, PaperPolicy::S3FifoLazyDemotionFastAdmissionMidpointReprieveHybrid(_)) {
				self.status.record_s3_fifo_lazy_demotion_fast_admission_midpoint_reprieve_hybrid_eviction();
			}

			#[cfg(feature = "s3_fifo_lazy_demotion_fast_admission_reprieve_hybrid_cache")]
			if matches!(*policy, PaperPolicy::S3FifoLazyDemotionFastAdmissionReprieveHybrid(_)) {
				self.status.record_s3_fifo_lazy_demotion_fast_admission_reprieve_hybrid_eviction();
			}

			#[cfg(feature = "s3_fifo_lazy_demotion_reprieve_hybrid_cache")]
			if matches!(*policy, PaperPolicy::S3FifoLazyDemotionReprieveHybrid(_)) {
				self.status.record_s3_fifo_lazy_demotion_reprieve_hybrid_eviction();
			}

			#[cfg(feature = "s3_fifo_lazy_demotion_fast_admission_split_slow_reprieve_hybrid_cache")]
			if matches!(*policy, PaperPolicy::S3FifoLazyDemotionFastAdmissionSplitSlowReprieveHybrid(_)) {
				self.status.record_s3_fifo_lazy_demotion_fast_admission_split_slow_reprieve_hybrid_eviction();
			}

			// Only recorded when something can replay it. Without this gate an
			// eviction-heavy workload would keep pushing into a `Vec` that
			// `flush_buffered_events` now clears without sending, which is
			// merely wasted work -- but wasted work inside the eviction loop,
			// which is precisely the loop GET latency waits on.
			if self.trace_worker.is_some() {
				buffered_events.push(StackEvent::Del(key));
			}
		}

		Ok(())
	}

	fn apply_mini_evictions(
		&mut self,
		mini_index: usize,
		buffered_events: &mut Vec<StackEvent>,
	) {
		let max_cache_size = self.status.max_size();
		let policy = self.current_policy.read();
		let mut evictions = Vec::<HashedKey>::new();

		while self.status.used_size(&policy) > max_cache_size {
			let maybe_key = self.mini_stack_manager
				.get_eviction(mini_index)
				.map(|key| EraseKey::Hashed(key));

			let erase_result = erase(
				&self.objects,
				&self.status,
				&self.overhead_manager,
				maybe_key,
			);

			let Ok((key, _)) = erase_result else {
				continue;
			};

			evictions.push(key);
			buffered_events.push(StackEvent::Del(key));
		}

		self.mini_stack_manager.apply_evictions(mini_index, evictions);
	}

	fn perform_auto_policy(&mut self, now: Instant, has_current_set: bool) -> Option<PaperPolicy> {
		if has_current_set || !self.status.is_auto_policy() || self.mini_index.is_some() {
			// don't switch the policy while (any of):
			// * there is recent set activity
			// * the auto policy is not configured
			// * a stack is being reconstructed
			return None;
		}

		let should_poll_policy = self.last_auto_policy_time
			.is_none_or(|last_auto_policy_time| now - last_auto_policy_time > AUTO_POLICY_DURATION);

		if !should_poll_policy {
			return None;
		}

		self.last_auto_policy_time = Some(now);
		self.mini_stack_manager.get_optimal_policy(&self.current_policy.read())
	}

	/// Parks this thread between polls.
	///
	/// The sleep is unconditional, including when the poll just processed a
	/// full batch and more work is already queued. That looks like it should
	/// be wrong -- the backlog visibly grows -- but it is load-bearing, and
	/// measured: this thread shares its cores with the request path, and
	/// `try_iter()` drains everything that accumulated during the sleep, so
	/// sleeping costs staleness (bounded by the polling interval) but not
	/// throughput. Skipping the sleep while work remains turns this loop into
	/// a spin that competes with the clients it exists to serve. On an 8-core
	/// box with 8 client threads and a 20k x 4 KiB working set, that cost
	/// 4.20M -> 3.17M gets/sec; `thread::yield_now()` in place of the sleep
	/// measured the same as the spin (3.15M), because with every client
	/// thread runnable a yield returns almost immediately.
	fn delay_event_loop(&mut self, now: Instant, has_current_set: bool) {
		let has_recent_set = self.last_set_time
			.is_some_and(|last_set_time| now - last_set_time <= SET_RECENCY_DURATION);

		if has_current_set {
			self.last_set_time = Some(now);
		}

		let delay = if has_recent_set {
			SHORT_POLLING_DURATION
		} else {
			LONG_POLLING_DURATION
		};

		thread::sleep(delay);
	}
}

/// Whether this cache can ever actually *use* an access trace.
///
/// The trace's only consumer is `reconstruct_policy_stack`, which replays it
/// to rebuild a different policy's stack after a live policy switch. A switch
/// requires a second policy to switch *to*: `PaperCache::policy()` rejects
/// anything outside the configured `policies` list, and `handle_policy`
/// early-returns when the requested policy already matches the current one.
/// So with a single configured policy -- every hybrid cache, and any
/// single-policy `paper-server` deployment -- reconstruction is unreachable
/// and every trace write is dead weight on the hot path.
///
/// `PaperPolicy::Auto` doesn't change this: it drives `perform_auto_policy`,
/// which picks from the same `policies` list via the mini stacks, so a
/// one-entry list can only ever "switch" to the policy already running.
fn trace_is_useful(status: &StatusRef) -> bool {
	status.policies().len() > 1
}

/// Spawns `TraceWorker` and seeds it with the cache's starting size, or
/// returns `(None, None)` when tracing is off (see `trace_is_useful`).
///
/// The initial `Resize` matters for reconstruction accuracy: a replay has to
/// know the size the cache was at when the recorded accesses happened.
fn spawn_trace_worker(
	enabled: bool,
	trace_fragments: &Arc<RwLock<VecDeque<TraceFragment>>>,
	max_size: CacheSize,
) -> Result<
	(Option<Sender<StackEvent>>, Option<thread::JoinHandle<Result<(), CacheError>>>),
	CacheError,
> {
	if !enabled {
		return Ok((None, None));
	}

	let (trace_worker, trace_listener) = unbounded();

	let trace_handle = register_worker(TraceWorker::new(
		trace_listener,
		trace_fragments.clone(),
	));

	if let Err(err) = trace_worker.send(StackEvent::Resize(max_size)) {
		error!("Could not send initial cache size to trace worker: {err:?}");
		return Err(CacheError::Internal);
	}

	Ok((Some(trace_worker), Some(trace_handle)))
}

fn reconstruct_policy_stack(
	policy: PaperPolicy,
	max_size: CacheSize,
	current_policy: Arc<RwLock<PaperPolicy>>,
	trace_fragments: Arc<RwLock<VecDeque<TraceFragment>>>,
) -> Result<Box<dyn PolicyStack>, CacheError> {
	let mut stack = init_policy_stack(policy, max_size);

	for fragment in trace_fragments.read().iter() {
		let mut fragment_modifiers = fragment.lock();
		let fragment_reader = &mut fragment_modifiers.0;

		let initial_position = match fragment_reader.stream_position() {
			Ok(position) => position,

			Err(err) => {
				error!("Could not get trace fragment initial stream position: {err:?}");
				return Err(CacheError::Internal);
			},
		};

		// start reading the file from the beginning
		if let Err(err) = fragment_reader.rewind() {
			error!("Could not rewind trace fragment: {err:?}");
			return Err(CacheError::Internal);
		}

		for (index, event) in fragment_reader.iter().enumerate() {
			if index & (RECONSTRUCT_POLICY_POLLING - 1) == 0 && policy != *current_policy.read() {
				// every RECONSTRUCT_POLICY_POLLING events, check if the currently
				// configured policy is still the policy we're reconstructing and
				// if it's not, move the reader back to its original position in
				// the file and terminate the reconstruction
				if let Err(err) = fragment_reader.seek(SeekFrom::Start(initial_position)) {
					error!("Could not seek within trace fragment: {err:?}");
				}

				return Err(CacheError::Internal);
			}

			match event {
				TraceEvent::Get(key) => stack.update(key),
				TraceEvent::Set(key, size) => stack.insert(key, size),
				TraceEvent::Del(key) => stack.remove(key),
				TraceEvent::Resize(size) => stack.resize(size),
			}
		}

		// ensure the underlying trace fragment is returned back to its original
		// position (this is mostly just a sanity check as reading the file should
		// already return it to the end which should be the orignal position)
		if let Err(err) = fragment_reader.seek(SeekFrom::Start(initial_position)) {
			error!("Could not seek within trace fragment: {err:?}");
			return Err(CacheError::Internal);
		}
	}

	Ok(stack)
}

unsafe impl<K, V> Send for PolicyWorker<K, V>
where
	K: TypeSize,
	V: TypeSize,
{}

#[cfg(all(test, feature = "lru_hybrid_cache"))]
mod lru_hybrid_tests {
	use super::*;

	use crate::{
		NoHasher,
		object::Object,
		status::AtomicStatus,
		object::overhead::{OverheadManager, get_hybrid_dram_shared_overhead},
	};

	type TestBuffer = Box<[u8]>;

	// The per-object shared-structure DRAM overhead `init_policy_stack` now
	// reserves out of the fast-tier budget (via `with_shared_overhead`). Tests
	// that size the fast tier to a small, exact byte budget must add headroom
	// for this reservation so their intended fast/slow boundary still holds.
	fn shared_overhead() -> CacheSize {
		get_hybrid_dram_shared_overhead(&PaperPolicy::LruHybrid) as CacheSize
	}

	// `LruHybridStack::settle_fast_tier` now also drains to
	// `FAST_TIER_LOW_WATER_RATIO` (0.98, private to that module -- duplicated
	// here) of the effective budget rather than back to exactly the ceiling.
	// At the razor-thin margins these tests intentionally use ("fits exactly
	// one object"), shaving 2% off can land the drain target *below* what a
	// single already-resident object needs, cascading an extra demotion that
	// isn't the scenario under test. Scaling a target byte count up by this
	// factor keeps that object comfortably above the drain target even after
	// the shave.
	fn low_water_safe(target: CacheSize) -> CacheSize {
		(target as f64 / 0.98).ceil() as CacheSize
	}

	// Exercises the real `PolicyWorker` migration pipeline end to end using a
	// plain `Box<[u8]>` value type and a trivial "migrate" closure — this
	// tests the generic wiring (LruHybridStack -> drain_tier_migrations ->
	// apply_tier_migrations -> Object::set_data -> AtomicStatus counters)
	// without needing the real `Hybrid`/UMF PMEM allocator that
	// `TieredBuffer::new_slow` depends on.
	fn make_worker(max_size: CacheSize) -> (
		PolicyWorker<u32, TestBuffer>,
		ObjectMapRef<u32, TestBuffer>,
		StatusRef,
		OverheadManagerRef,
	) {
		let (_tx, rx) = unbounded::<WorkerEvent>();

		let objects: ObjectMapRef<u32, TestBuffer> =
			crate::new_hybrid_object_map();

		let status = Arc::new(
			AtomicStatus::new(max_size, &[PaperPolicy::LruHybrid], PaperPolicy::LruHybrid).unwrap(),
		);

		let overhead_manager = Arc::new(OverheadManager::new(&status));

		// Tags each byte buffer by overwriting its last byte with a marker so
		// tests can confirm the object's bytes were actually replaced, not
		// just relabeled. Must NOT change the buffer's length: the real
		// `TieredBuffer` migration doesn't either, and both `AtomicStatus`'s
		// base_used_size and `LruHybridStack`'s own per-key size bookkeeping
		// assume a migration never changes an object's accounted size (see
		// CLAUDE.md's `lru_hybrid_cache` plan). Growing the buffer here
		// previously desynced `base_used_size` between insert-time and
		// erase-time, wrapping it to a huge value and hanging
		// `apply_evictions`'s eviction loop forever.
		let migrate: Box<dyn Fn(&TestBuffer, Tier) -> TestBuffer + Send + Sync> =
			Box::new(|bytes, tier| {
				let marker: u8 = match tier {
					Tier::Fast => 0xFA,
					Tier::Slow => 0x50,
				};

				let mut v = bytes.to_vec();
				if let Some(last) = v.last_mut() {
					*last = marker;
				}
				v.into_boxed_slice()
			});

		let worker = PolicyWorker::new_with_tier_migration(
			rx,
			objects.clone(),
			status.clone(),
			overhead_manager.clone(),
			migrate,
		).unwrap();

		(worker, objects, status, overhead_manager)
	}

	// Mirrors what `PaperCache::set()` does to shared state (object map +
	// status accounting, using the *same* `overhead_manager.base_size` the
	// production `set()` path computes and broadcasts) before calling
	// `handle_set`, so both `status.used_size()` and the policy stack's own
	// byte accounting behave realistically for tests exercising
	// `apply_evictions` / tier-boundary math.
	fn insert(
		objects: &ObjectMapRef<u32, TestBuffer>,
		status: &StatusRef,
		overhead_manager: &OverheadManagerRef,
		worker: &mut PolicyWorker<u32, TestBuffer>,
		key: HashedKey,
		size: usize,
	) {
		let object = Object::new(key as u32, vec![0u8; size].into_boxed_slice(), None);
		let base_size = overhead_manager.base_size(&object);

		objects.insert(key, object);
		status.update_base_used_size(base_size as i64);
		status.incr_num_objects();
		worker.handle_set(key, base_size);
	}

	// `overhead_manager.base_size` includes fixed key/expiry overhead on top
	// of the raw byte length, so "insert a 15-byte object" doesn't account
	// for exactly 15 bytes in the stack. Compute the real base_size of a
	// representative object so tests can set an exact fast-tier capacity
	// (via `handle_resize_fast_tier`, bypassing the 20%-of-max_size default)
	// without depending on the precise overhead constant.
	fn base_size_of(overhead_manager: &OverheadManagerRef, size: usize) -> ObjectSize {
		let probe = Object::new(0u32, vec![0u8; size].into_boxed_slice(), None);
		overhead_manager.base_size(&probe)
	}

	#[test]
	fn demotion_physically_replaces_object_bytes_and_updates_stats() {
		let (mut worker, objects, status, overhead_manager) = make_worker(1_000);

		// Fits exactly one ~15-byte object but not two. The fast budget must
		// also cover the reserved shared-metadata overhead for both tracked
		// objects (2 × shared), on top of one object's value bytes + 1,
		// scaled up via `low_water_safe` so the 2% low-water headroom (see
		// `FAST_TIER_LOW_WATER_RATIO`) doesn't shave the drain target below
		// what the single surviving object needs.
		worker.handle_resize_fast_tier(
			low_water_safe(base_size_of(&overhead_manager, 15) as CacheSize + 2 * shared_overhead() + 1),
		);

		insert(&objects, &status, &overhead_manager, &mut worker, 1, 15); // fast
		insert(&objects, &status, &overhead_manager, &mut worker, 2, 10); // demotes key 1

		worker.apply_tier_migrations();
		worker.refresh_tier_gauges();

		let snapshot = status.lru_hybrid_stats();
		assert_eq!(snapshot.demotions, 1);
		assert_eq!(snapshot.promotions, 0);
		assert_eq!(snapshot.fast_objects, 1);
		assert_eq!(snapshot.slow_objects, 1);

		// The demoted key's bytes were physically replaced by `migrate`
		// with the Slow-tagged version.
		let data = objects.get_ref(&1).unwrap().data();
		assert_eq!(data.last(), Some(&0x50));
	}

	#[test]
	fn access_promotes_a_slow_key_and_may_cascade_a_demotion() {
		let (mut worker, objects, status, overhead_manager) = make_worker(1_000);

		// Fits exactly one ~15-byte object but not two (plus reserved
		// shared-metadata overhead for both tracked objects), scaled up via
		// `low_water_safe` so the 2% low-water headroom doesn't cascade an
		// extra demotion of the single object this scenario expects to
		// survive the promotion.
		worker.handle_resize_fast_tier(
			low_water_safe(base_size_of(&overhead_manager, 15) as CacheSize + 2 * shared_overhead() + 1),
		);

		insert(&objects, &status, &overhead_manager, &mut worker, 1, 15);
		insert(&objects, &status, &overhead_manager, &mut worker, 2, 10); // demotes 1
		worker.apply_tier_migrations();
		worker.refresh_tier_gauges();

		// Accessing the now-slow key promotes it, which may itself demote
		// key 2 (the new fast-tier LRU tail).
		worker.handle_get(1, true);
		worker.apply_tier_migrations();
		worker.refresh_tier_gauges();

		let snapshot = status.lru_hybrid_stats();
		assert_eq!(snapshot.promotions, 1);
		assert_eq!(snapshot.demotions, 2);

		let data_1 = objects.get_ref(&1).unwrap().data();
		assert_eq!(data_1.last(), Some(&0xFA));
	}

	#[test]
	fn eviction_under_lru_hybrid_policy_is_recorded_in_stats() {
		// A tiny max_size relative to per-object policy overhead guarantees
		// apply_evictions has to remove at least one object; exactly how
		// many depends on LruHybrid's fixed per-object overhead, so assert
		// self-consistency (evictions == objects actually removed) rather
		// than a hardcoded count.
		let (mut worker, objects, status, overhead_manager) = make_worker(20);

		insert(&objects, &status, &overhead_manager, &mut worker, 1, 15);
		insert(&objects, &status, &overhead_manager, &mut worker, 2, 15);
		worker.apply_tier_migrations();
		worker.refresh_tier_gauges();

		let mut buffered_events = Vec::new();
		worker.apply_evictions(&mut buffered_events).unwrap();

		let evictions = status.lru_hybrid_stats().evictions;
		assert!(evictions >= 1);
		assert_eq!(objects.len() as u64, 2 - evictions);
	}

	#[test]
	fn no_migration_fn_is_a_safe_no_op() {
		// A plain `new()`-constructed worker (no tier_migration_fn) should
		// never panic when `apply_tier_migrations` is called, even though
		// its policy stack may still be LruHybrid via `init_policy_stack`.
		let (_tx, rx) = unbounded::<WorkerEvent>();
		let objects: ObjectMapRef<u32, TestBuffer> =
			crate::new_hybrid_object_map();
		let status = Arc::new(
			AtomicStatus::new(100, &[PaperPolicy::LruHybrid], PaperPolicy::LruHybrid).unwrap(),
		);
		let overhead_manager = Arc::new(OverheadManager::new(&status));

		let mut worker = PolicyWorker::<u32, TestBuffer>::new(
			rx,
			objects.clone(),
			status.clone(),
			overhead_manager.clone(),
			None,
		).unwrap();

		insert(&objects, &status, &overhead_manager, &mut worker, 1, 15);
		insert(&objects, &status, &overhead_manager, &mut worker, 2, 10);

		worker.apply_tier_migrations();
		worker.refresh_tier_gauges();
	}
}

#[cfg(all(test, feature = "lfu_hybrid_cache"))]
mod lfu_hybrid_tests {
	use super::*;

	use crate::{
		NoHasher,
		object::Object,
		status::AtomicStatus,
		object::overhead::{OverheadManager, get_hybrid_dram_shared_overhead},
	};

	type TestBuffer = Box<[u8]>;

	// Per-object shared-structure DRAM overhead reserved out of the fast-tier
	// budget by `init_policy_stack` (`with_shared_overhead`). Tests sizing the
	// fast tier to a small exact budget add headroom for it — note the
	// reservation scales with the *total* tracked object count, so a test with
	// N tracked objects must budget N × this on top of the values it wants
	// resident.
	fn shared_overhead() -> CacheSize {
		get_hybrid_dram_shared_overhead(&PaperPolicy::LfuHybrid) as CacheSize
	}

	// Same rationale as `lru_hybrid_tests::make_worker`: exercises the real
	// `PolicyWorker` migration pipeline end to end using a plain `Box<[u8]>`
	// value type and a trivial "migrate" closure, without needing the real
	// `Hybrid`/UMF PMEM allocator that `TieredBuffer::new_slow` depends on.
	fn make_worker(max_size: CacheSize) -> (
		PolicyWorker<u32, TestBuffer>,
		ObjectMapRef<u32, TestBuffer>,
		StatusRef,
		OverheadManagerRef,
	) {
		let (_tx, rx) = unbounded::<WorkerEvent>();

		let objects: ObjectMapRef<u32, TestBuffer> =
			crate::new_hybrid_object_map();

		let status = Arc::new(
			AtomicStatus::new(max_size, &[PaperPolicy::LfuHybrid], PaperPolicy::LfuHybrid).unwrap(),
		);

		let overhead_manager = Arc::new(OverheadManager::new(&status));

		// See `lru_hybrid_tests::make_worker` for why this must never change
		// the buffer's byte length (a migration that does desyncs
		// `base_used_size` and can hang `apply_evictions`'s loop forever).
		let migrate: Box<dyn Fn(&TestBuffer, Tier) -> TestBuffer + Send + Sync> =
			Box::new(|bytes, tier| {
				let marker: u8 = match tier {
					Tier::Fast => 0xFA,
					Tier::Slow => 0x50,
				};

				let mut v = bytes.to_vec();
				if let Some(last) = v.last_mut() {
					*last = marker;
				}
				v.into_boxed_slice()
			});

		let worker = PolicyWorker::new_with_tier_migration(
			rx,
			objects.clone(),
			status.clone(),
			overhead_manager.clone(),
			migrate,
		).unwrap();

		(worker, objects, status, overhead_manager)
	}

	fn insert(
		objects: &ObjectMapRef<u32, TestBuffer>,
		status: &StatusRef,
		overhead_manager: &OverheadManagerRef,
		worker: &mut PolicyWorker<u32, TestBuffer>,
		key: HashedKey,
		size: usize,
	) {
		let object = Object::new(key as u32, vec![0u8; size].into_boxed_slice(), None);
		let base_size = overhead_manager.base_size(&object);

		objects.insert(key, object);
		status.update_base_used_size(base_size as i64);
		status.incr_num_objects();
		worker.handle_set(key, base_size);
	}

	fn base_size_of(overhead_manager: &OverheadManagerRef, size: usize) -> ObjectSize {
		let probe = Object::new(0u32, vec![0u8; size].into_boxed_slice(), None);
		overhead_manager.base_size(&probe)
	}

	#[test]
	fn admission_once_fast_is_full_goes_directly_to_slow_and_updates_stats() {
		let (mut worker, objects, status, overhead_manager) = make_worker(1_000);

		// Fits exactly one 15-byte object's value (plus that object's own
		// reserved shared-metadata overhead), so a second key can't be admitted
		// fast.
		worker.handle_resize_fast_tier(
			base_size_of(&overhead_manager, 15) as CacheSize + shared_overhead(),
		);

		insert(&objects, &status, &overhead_manager, &mut worker, 1, 15); // fast, count 1
		// Fast tier is now full; key 2 is admitted straight to slow -- key 1
		// (the existing resident) is untouched, matching the paper's
		// admission rule literally.
		insert(&objects, &status, &overhead_manager, &mut worker, 2, 10);

		worker.apply_tier_migrations();
		worker.refresh_tier_gauges();

		let snapshot = status.lfu_hybrid_stats();
		assert_eq!(snapshot.demotions, 0);
		assert_eq!(snapshot.promotions, 0);
		assert_eq!(snapshot.fast_objects, 1);
		assert_eq!(snapshot.slow_objects, 1);

		// Key 2's bytes were physically built as Slow via the migration;
		// key 1 was never migrated (correctly Fast from the start).
		let data_2 = objects.get_ref(&2).unwrap().data();
		assert_eq!(data_2.last(), Some(&0x50));
	}

	#[test]
	fn promotion_pressure_demotes_the_lowest_frequency_fast_key_and_updates_stats() {
		let (mut worker, objects, status, overhead_manager) = make_worker(1_000);

		// Fits exactly two ~15-byte object values, not three. Budget for two
		// values plus the reserved shared-metadata overhead of all three
		// tracked objects (so the third's promotion demotes exactly one).
		worker.handle_resize_fast_tier(
			base_size_of(&overhead_manager, 15) as CacheSize * 2 + 3 * shared_overhead(),
		);

		insert(&objects, &status, &overhead_manager, &mut worker, 1, 15);
		insert(&objects, &status, &overhead_manager, &mut worker, 2, 15);
		worker.apply_tier_migrations();
		worker.refresh_tier_gauges();

		// Bump key 1 so key 2 is unambiguously the fast tier's minimum.
		worker.handle_get(1, true);
		worker.apply_tier_migrations();
		worker.refresh_tier_gauges();

		// Fast tier is full -> key 3 is admitted directly to slow.
		insert(&objects, &status, &overhead_manager, &mut worker, 3, 15);
		worker.apply_tier_migrations();
		worker.refresh_tier_gauges();

		// Bump key 3 past the fast minimum (key 2, count 1) -> promotes,
		// which must demote key 2 to make room.
		worker.handle_get(3, true);
		worker.apply_tier_migrations();
		worker.refresh_tier_gauges();

		let snapshot = status.lfu_hybrid_stats();
		assert_eq!(snapshot.promotions, 1);
		assert_eq!(snapshot.demotions, 1);

		let data_3 = objects.get_ref(&3).unwrap().data();
		assert_eq!(data_3.last(), Some(&0xFA));

		let data_2 = objects.get_ref(&2).unwrap().data();
		assert_eq!(data_2.last(), Some(&0x50));
	}

	#[test]
	fn eviction_under_lfu_hybrid_policy_is_recorded_in_stats() {
		// A tiny max_size relative to per-object policy overhead guarantees
		// apply_evictions has to remove at least one object; assert
		// self-consistency (evictions == objects actually removed) rather
		// than a hardcoded count, same as the `lru_hybrid_cache` analog.
		let (mut worker, objects, status, overhead_manager) = make_worker(20);

		insert(&objects, &status, &overhead_manager, &mut worker, 1, 15);
		insert(&objects, &status, &overhead_manager, &mut worker, 2, 15);
		worker.apply_tier_migrations();
		worker.refresh_tier_gauges();

		let mut buffered_events = Vec::new();
		worker.apply_evictions(&mut buffered_events).unwrap();

		let evictions = status.lfu_hybrid_stats().evictions;
		assert!(evictions >= 1);
		assert_eq!(objects.len() as u64, 2 - evictions);
	}

	#[test]
	fn no_migration_fn_is_a_safe_no_op() {
		// A plain `new()`-constructed worker (no tier_migration_fn) should
		// never panic when `apply_tier_migrations` is called, even though
		// its policy stack may still be LfuHybrid via `init_policy_stack`.
		let (_tx, rx) = unbounded::<WorkerEvent>();
		let objects: ObjectMapRef<u32, TestBuffer> =
			crate::new_hybrid_object_map();
		let status = Arc::new(
			AtomicStatus::new(100, &[PaperPolicy::LfuHybrid], PaperPolicy::LfuHybrid).unwrap(),
		);
		let overhead_manager = Arc::new(OverheadManager::new(&status));

		let mut worker = PolicyWorker::<u32, TestBuffer>::new(
			rx,
			objects.clone(),
			status.clone(),
			overhead_manager.clone(),
			None,
		).unwrap();

		insert(&objects, &status, &overhead_manager, &mut worker, 1, 15);
		insert(&objects, &status, &overhead_manager, &mut worker, 2, 10);

		worker.apply_tier_migrations();
		worker.refresh_tier_gauges();
	}
}

#[cfg(all(test, feature = "fifo_hybrid_cache"))]
mod fifo_hybrid_tests {
	use super::*;

	use crate::{
		NoHasher,
		object::Object,
		status::AtomicStatus,
		object::overhead::OverheadManager,
	};

	type TestBuffer = Box<[u8]>;

	// Same rationale as `lru_hybrid_tests::make_worker`/`two_q_hybrid_tests::
	// make_worker`: exercises the real `PolicyWorker` migration pipeline end
	// to end using a plain `Box<[u8]>` value type and a trivial "migrate"
	// closure, without needing the real `Hybrid`/UMF PMEM allocator that
	// `TieredBuffer::new_slow` depends on. `FifoHybridStack` has no
	// shared-overhead reservation (pure paper-spec implementation — see that
	// stack's module doc), so unlike `lru_hybrid_tests`/`lfu_hybrid_tests`
	// there's no `shared_overhead`/`low_water_safe` helper needed here.
	fn make_worker(max_size: CacheSize) -> (
		PolicyWorker<u32, TestBuffer>,
		ObjectMapRef<u32, TestBuffer>,
		StatusRef,
		OverheadManagerRef,
	) {
		let (_tx, rx) = unbounded::<WorkerEvent>();

		let objects: ObjectMapRef<u32, TestBuffer> =
			crate::new_hybrid_object_map();

		let status = Arc::new(
			AtomicStatus::new(max_size, &[PaperPolicy::FifoHybrid], PaperPolicy::FifoHybrid).unwrap(),
		);

		let overhead_manager = Arc::new(OverheadManager::new(&status));

		// See `lru_hybrid_tests::make_worker` for why this must never change
		// the buffer's byte length (a migration that does desyncs
		// `base_used_size` and can hang `apply_evictions`'s loop forever).
		let migrate: Box<dyn Fn(&TestBuffer, Tier) -> TestBuffer + Send + Sync> =
			Box::new(|bytes, tier| {
				let marker: u8 = match tier {
					Tier::Fast => 0xFA,
					Tier::Slow => 0x50,
				};

				let mut v = bytes.to_vec();
				if let Some(last) = v.last_mut() {
					*last = marker;
				}
				v.into_boxed_slice()
			});

		let worker = PolicyWorker::new_with_tier_migration(
			rx,
			objects.clone(),
			status.clone(),
			overhead_manager.clone(),
			migrate,
		).unwrap();

		(worker, objects, status, overhead_manager)
	}

	fn insert(
		objects: &ObjectMapRef<u32, TestBuffer>,
		status: &StatusRef,
		overhead_manager: &OverheadManagerRef,
		worker: &mut PolicyWorker<u32, TestBuffer>,
		key: HashedKey,
		size: usize,
	) {
		let object = Object::new(key as u32, vec![0u8; size].into_boxed_slice(), None);
		let base_size = overhead_manager.base_size(&object);

		objects.insert(key, object);
		status.update_base_used_size(base_size as i64);
		status.incr_num_objects();
		worker.handle_set(key, base_size);
	}

	fn base_size_of(overhead_manager: &OverheadManagerRef, size: usize) -> ObjectSize {
		let probe = Object::new(0u32, vec![0u8; size].into_boxed_slice(), None);
		overhead_manager.base_size(&probe)
	}

	#[test]
	fn demotion_physically_replaces_object_bytes_and_updates_stats() {
		let (mut worker, objects, status, overhead_manager) = make_worker(1_000);

		// Fits exactly one ~15-byte object but not two.
		worker.handle_resize_fast_tier(base_size_of(&overhead_manager, 15) as CacheSize + 1);

		insert(&objects, &status, &overhead_manager, &mut worker, 1, 15); // fast
		insert(&objects, &status, &overhead_manager, &mut worker, 2, 10); // demotes key 1 (oldest)

		worker.apply_tier_migrations();
		worker.refresh_tier_gauges();

		let snapshot = status.fifo_hybrid_stats();
		assert_eq!(snapshot.demotions, 1);
		assert_eq!(snapshot.promotions, 0);
		assert_eq!(snapshot.fast_objects, 1);
		assert_eq!(snapshot.slow_objects, 1);

		// The demoted key's bytes were physically replaced by `migrate`
		// with the Slow-tagged version.
		let data = objects.get_ref(&1).unwrap().data();
		assert_eq!(data.last(), Some(&0x50));
	}

	#[test]
	fn hit_on_slow_key_does_not_migrate_or_reorder() {
		// Replaces the other hybrids' "access promotes a slow key" test:
		// FIFO has no promotion policy at all, so this test asserts the
		// opposite — a hit on a slow-tier key must never migrate it back.
		let (mut worker, objects, status, overhead_manager) = make_worker(1_000);

		worker.handle_resize_fast_tier(base_size_of(&overhead_manager, 15) as CacheSize + 1);

		insert(&objects, &status, &overhead_manager, &mut worker, 1, 15);
		insert(&objects, &status, &overhead_manager, &mut worker, 2, 10); // demotes 1
		worker.apply_tier_migrations();
		worker.refresh_tier_gauges();

		let snapshot_before = status.fifo_hybrid_stats();
		assert_eq!(snapshot_before.demotions, 1);

		// A "hit" on the now-slow key must be a total no-op.
		worker.handle_get(1, true);
		worker.apply_tier_migrations();
		worker.refresh_tier_gauges();

		let snapshot_after = status.fifo_hybrid_stats();
		assert_eq!(snapshot_after.demotions, snapshot_before.demotions);
		assert_eq!(snapshot_after.promotions, 0);
		assert_eq!(snapshot_after.fast_objects, snapshot_before.fast_objects);
		assert_eq!(snapshot_after.slow_objects, snapshot_before.slow_objects);

		// Bytes are still tagged Slow -- never re-migrated back to Fast.
		let data_1 = objects.get_ref(&1).unwrap().data();
		assert_eq!(data_1.last(), Some(&0x50));
	}

	#[test]
	fn eviction_under_fifo_hybrid_policy_is_recorded_in_stats() {
		// A tiny max_size relative to per-object policy overhead guarantees
		// apply_evictions has to remove at least one object; assert
		// self-consistency (evictions == objects actually removed) rather
		// than a hardcoded count, same as the other hybrids' analogs.
		let (mut worker, objects, status, overhead_manager) = make_worker(20);

		insert(&objects, &status, &overhead_manager, &mut worker, 1, 15);
		insert(&objects, &status, &overhead_manager, &mut worker, 2, 15);
		worker.apply_tier_migrations();
		worker.refresh_tier_gauges();

		let mut buffered_events = Vec::new();
		worker.apply_evictions(&mut buffered_events).unwrap();

		let evictions = status.fifo_hybrid_stats().evictions;
		assert!(evictions >= 1);
		assert_eq!(objects.len() as u64, 2 - evictions);
	}

	#[test]
	fn no_migration_fn_is_a_safe_no_op() {
		// A plain `new()`-constructed worker (no tier_migration_fn) should
		// never panic when `apply_tier_migrations` is called, even though
		// its policy stack may still be FifoHybrid via `init_policy_stack`.
		let (_tx, rx) = unbounded::<WorkerEvent>();
		let objects: ObjectMapRef<u32, TestBuffer> =
			crate::new_hybrid_object_map();
		let status = Arc::new(
			AtomicStatus::new(100, &[PaperPolicy::FifoHybrid], PaperPolicy::FifoHybrid).unwrap(),
		);
		let overhead_manager = Arc::new(OverheadManager::new(&status));

		let mut worker = PolicyWorker::<u32, TestBuffer>::new(
			rx,
			objects.clone(),
			status.clone(),
			overhead_manager.clone(),
			None,
		).unwrap();

		insert(&objects, &status, &overhead_manager, &mut worker, 1, 15);
		insert(&objects, &status, &overhead_manager, &mut worker, 2, 10);

		worker.apply_tier_migrations();
		worker.refresh_tier_gauges();
	}
}

#[cfg(all(test, feature = "two_q_fast_admission_hybrid_cache"))]
mod two_q_fast_admission_hybrid_tests {
	use super::*;

	use crate::{
		NoHasher,
		object::Object,
		status::AtomicStatus,
		object::overhead::OverheadManager,
	};

	type TestBuffer = Box<[u8]>;

	/// `k_in` is deliberately tiny (not `1.0`, unlike `two_q_hybrid_tests`):
	/// here the FIFO budget is carved *out of* the fast tier, so a large
	/// `k_in` would leave the main queue no fast segment at all and every
	/// promotion would self-demote. `0.001` of these tests' `max_size`
	/// rounds to a 0- or 1-byte reservation, keeping
	/// `effective_main_fast_capacity` essentially equal to whatever
	/// `handle_resize_fast_tier` sets — which is what these tests want to
	/// exercise.
	fn make_worker(max_size: CacheSize) -> (
		PolicyWorker<u32, TestBuffer>,
		ObjectMapRef<u32, TestBuffer>,
		StatusRef,
		OverheadManagerRef,
	) {
		let (_tx, rx) = unbounded::<WorkerEvent>();

		let objects: ObjectMapRef<u32, TestBuffer> =
			crate::new_hybrid_object_map();

		let policy = PaperPolicy::TwoQFastAdmissionHybrid(0.001);
		let status = Arc::new(
			AtomicStatus::new(max_size, &[policy], policy).unwrap(),
		);

		let overhead_manager = Arc::new(OverheadManager::new(&status));

		let migrate: Box<dyn Fn(&TestBuffer, Tier) -> TestBuffer + Send + Sync> =
			Box::new(|bytes, tier| {
				let marker: u8 = match tier {
					Tier::Fast => 0xFA,
					Tier::Slow => 0x50,
				};

				let mut v = bytes.to_vec();
				if let Some(last) = v.last_mut() {
					*last = marker;
				}
				v.into_boxed_slice()
			});

		let worker = PolicyWorker::new_with_tier_migration(
			rx,
			objects.clone(),
			status.clone(),
			overhead_manager.clone(),
			migrate,
		).unwrap();

		(worker, objects, status, overhead_manager)
	}

	fn insert(
		objects: &ObjectMapRef<u32, TestBuffer>,
		status: &StatusRef,
		overhead_manager: &OverheadManagerRef,
		worker: &mut PolicyWorker<u32, TestBuffer>,
		key: HashedKey,
		size: usize,
	) {
		let object = Object::new(key as u32, vec![0u8; size].into_boxed_slice(), None);
		let base_size = overhead_manager.base_size(&object);

		objects.insert(key, object);
		status.update_base_used_size(base_size as i64);
		status.incr_num_objects();
		worker.handle_set(key, base_size);
	}

	fn base_size_of(overhead_manager: &OverheadManagerRef, size: usize) -> ObjectSize {
		let probe = Object::new(0u32, vec![0u8; size].into_boxed_slice(), None);
		overhead_manager.base_size(&probe)
	}

	/// The whole point of this design: a brand-new key is admitted fast and
	/// stays fast, with no migration to correct it. Contrast
	/// `two_q_hybrid_tests::admission_lands_slow_and_promotion_physically_moves_bytes`.
	#[test]
	fn admission_lands_fast_and_needs_no_migration() {
		let (mut worker, objects, status, overhead_manager) = make_worker(1_000);

		worker.handle_resize_fast_tier(base_size_of(&overhead_manager, 15) as CacheSize * 4);

		insert(&objects, &status, &overhead_manager, &mut worker, 1, 15);
		worker.apply_tier_migrations();
		worker.refresh_tier_gauges();

		let snapshot = status.two_q_fast_admission_hybrid_stats();
		assert_eq!(snapshot.promotions, 0);
		assert_eq!(snapshot.demotions, 0);
		assert_eq!(snapshot.fast_objects, 1);
		assert_eq!(snapshot.slow_objects, 0);

		// The API layer built these bytes Fast and nothing rewrote them, so
		// the migrate closure's marker was never applied.
		let data = objects.get_ref(&1).unwrap().data();
		assert_ne!(data.last(), Some(&0x50));
		assert_ne!(data.last(), Some(&0xFA));
	}

	/// A FIFO->main promotion is Fast->Fast, so it must not rewrite bytes or
	/// bump the promotions counter -- the optimization this design unlocks.
	#[test]
	fn promotion_out_of_the_fifo_queue_moves_no_bytes() {
		let (mut worker, objects, status, overhead_manager) = make_worker(1_000);

		worker.handle_resize_fast_tier(base_size_of(&overhead_manager, 15) as CacheSize * 4);

		insert(&objects, &status, &overhead_manager, &mut worker, 1, 15);
		worker.handle_get(1, true);
		worker.apply_tier_migrations();
		worker.refresh_tier_gauges();

		let snapshot = status.two_q_fast_admission_hybrid_stats();
		assert_eq!(snapshot.promotions, 0);
		assert_eq!(snapshot.demotions, 0);
		assert_eq!(snapshot.fast_objects, 1);

		let data = objects.get_ref(&1).unwrap().data();
		assert_ne!(data.last(), Some(&0xFA));
	}

	/// Main-queue fast-tier pressure still moves real bytes DRAM->PMEM.
	#[test]
	fn demotion_physically_replaces_object_bytes_and_updates_stats() {
		let (mut worker, objects, status, overhead_manager) = make_worker(1_000);

		// Room in the main queue for exactly one ~15-byte object.
		worker.handle_resize_fast_tier(base_size_of(&overhead_manager, 15) as CacheSize + 1);

		// Both keys need a second access to reach the main queue.
		insert(&objects, &status, &overhead_manager, &mut worker, 1, 15);
		worker.handle_get(1, true);
		insert(&objects, &status, &overhead_manager, &mut worker, 2, 10);
		worker.handle_get(2, true);

		worker.apply_tier_migrations();
		worker.refresh_tier_gauges();

		let snapshot = status.two_q_fast_admission_hybrid_stats();
		assert_eq!(snapshot.demotions, 1);
		assert_eq!(snapshot.slow_objects, 1);

		// Key 1 was the main queue's LRU tail, so it took the demotion.
		let data = objects.get_ref(&1).unwrap().data();
		assert_eq!(data.last(), Some(&0x50));
	}

	/// Re-accessing a demoted main-queue key is a genuine PMEM->DRAM move,
	/// so unlike the FIFO promotion above it does bump `promotions`.
	#[test]
	fn re_accessing_a_demoted_key_promotes_it_with_a_real_byte_move() {
		let (mut worker, objects, status, overhead_manager) = make_worker(1_000);

		worker.handle_resize_fast_tier(base_size_of(&overhead_manager, 15) as CacheSize + 1);

		insert(&objects, &status, &overhead_manager, &mut worker, 1, 15);
		worker.handle_get(1, true);
		insert(&objects, &status, &overhead_manager, &mut worker, 2, 10);
		worker.handle_get(2, true);
		worker.apply_tier_migrations();
		worker.refresh_tier_gauges();

		assert_eq!(status.two_q_fast_admission_hybrid_stats().demotions, 1);

		worker.handle_get(1, true);
		worker.apply_tier_migrations();
		worker.refresh_tier_gauges();

		let snapshot = status.two_q_fast_admission_hybrid_stats();
		assert_eq!(snapshot.promotions, 1);

		let data_1 = objects.get_ref(&1).unwrap().data();
		assert_eq!(data_1.last(), Some(&0xFA));
	}

	#[test]
	fn eviction_under_two_q_fast_admission_hybrid_policy_is_recorded_in_stats() {
		// A tiny max_size relative to per-object policy overhead guarantees
		// apply_evictions has to remove at least one object; assert
		// self-consistency rather than a hardcoded count, same as the other
		// hybrids' analogs.
		let (mut worker, objects, status, overhead_manager) = make_worker(20);

		insert(&objects, &status, &overhead_manager, &mut worker, 1, 15);
		insert(&objects, &status, &overhead_manager, &mut worker, 2, 15);
		worker.apply_tier_migrations();
		worker.refresh_tier_gauges();

		let mut buffered_events = Vec::new();
		worker.apply_evictions(&mut buffered_events).unwrap();

		let evictions = status.two_q_fast_admission_hybrid_stats().evictions;
		assert!(evictions >= 1);
		assert_eq!(objects.len() as u64, 2 - evictions);
	}

	/// `hybrid_stats()` (the design-neutral accessor) must agree with this
	/// design's own named accessor.
	#[test]
	fn hybrid_stats_mirrors_the_named_accessor() {
		let (mut worker, objects, status, overhead_manager) = make_worker(1_000);

		worker.handle_resize_fast_tier(base_size_of(&overhead_manager, 15) as CacheSize + 1);

		insert(&objects, &status, &overhead_manager, &mut worker, 1, 15);
		worker.handle_get(1, true);
		insert(&objects, &status, &overhead_manager, &mut worker, 2, 10);
		worker.handle_get(2, true);
		worker.apply_tier_migrations();
		worker.refresh_tier_gauges();

		let named = status.two_q_fast_admission_hybrid_stats();
		let common = status.hybrid_stats();

		assert_eq!(common.promotions, named.promotions);
		assert_eq!(common.demotions, named.demotions);
		assert_eq!(common.evictions, named.evictions);
		assert_eq!(common.fast_bytes_used, named.fast_bytes_used);
		assert_eq!(common.slow_bytes_used, named.slow_bytes_used);
		assert_eq!(common.fast_objects, named.fast_objects);
		assert_eq!(common.slow_objects, named.slow_objects);
		assert!(common.demotions > 0, "test should have produced a demotion to compare");
	}
}

#[cfg(all(test, feature = "two_q_fast_admission_reprieve_hybrid_cache"))]
mod two_q_fast_admission_reprieve_hybrid_tests {
	use super::*;

	use crate::{
		NoHasher,
		object::Object,
		status::AtomicStatus,
		object::overhead::OverheadManager,
	};

	type TestBuffer = Box<[u8]>;

	/// `k_in` is deliberately tiny (not `1.0`, unlike `two_q_hybrid_tests`):
	/// here the FIFO budget is carved *out of* the fast tier, so a large
	/// `k_in` would leave the main queue no fast segment at all and every
	/// promotion would self-demote. `0.001` of these tests' `max_size`
	/// rounds to a 0- or 1-byte reservation, keeping
	/// `effective_main_fast_capacity` essentially equal to whatever
	/// `handle_resize_fast_tier` sets — which is what these tests want to
	/// exercise.
	fn make_worker(max_size: CacheSize) -> (
		PolicyWorker<u32, TestBuffer>,
		ObjectMapRef<u32, TestBuffer>,
		StatusRef,
		OverheadManagerRef,
	) {
		let (_tx, rx) = unbounded::<WorkerEvent>();

		let objects: ObjectMapRef<u32, TestBuffer> =
			crate::new_hybrid_object_map();

		let policy = PaperPolicy::TwoQFastAdmissionReprieveHybrid(0.001);
		let status = Arc::new(
			AtomicStatus::new(max_size, &[policy], policy).unwrap(),
		);

		let overhead_manager = Arc::new(OverheadManager::new(&status));

		let migrate: Box<dyn Fn(&TestBuffer, Tier) -> TestBuffer + Send + Sync> =
			Box::new(|bytes, tier| {
				let marker: u8 = match tier {
					Tier::Fast => 0xFA,
					Tier::Slow => 0x50,
				};

				let mut v = bytes.to_vec();
				if let Some(last) = v.last_mut() {
					*last = marker;
				}
				v.into_boxed_slice()
			});

		let worker = PolicyWorker::new_with_tier_migration(
			rx,
			objects.clone(),
			status.clone(),
			overhead_manager.clone(),
			migrate,
		).unwrap();

		(worker, objects, status, overhead_manager)
	}

	fn insert(
		objects: &ObjectMapRef<u32, TestBuffer>,
		status: &StatusRef,
		overhead_manager: &OverheadManagerRef,
		worker: &mut PolicyWorker<u32, TestBuffer>,
		key: HashedKey,
		size: usize,
	) {
		let object = Object::new(key as u32, vec![0u8; size].into_boxed_slice(), None);
		let base_size = overhead_manager.base_size(&object);

		objects.insert(key, object);
		status.update_base_used_size(base_size as i64);
		status.incr_num_objects();
		worker.handle_set(key, base_size);
	}

	fn base_size_of(overhead_manager: &OverheadManagerRef, size: usize) -> ObjectSize {
		let probe = Object::new(0u32, vec![0u8; size].into_boxed_slice(), None);
		overhead_manager.base_size(&probe)
	}

	/// `max_size` used by every test below except the eviction one, chosen so
	/// `k_in` (0.001) yields a FIFO reservation of 1,000 bytes -- roomy enough
	/// for the ~15-byte objects these tests admit, so nothing is reprieved
	/// unless a test means to trigger it.
	const TEST_MAX_SIZE: CacheSize = 1_000_000;

	/// Bytes reserved for the one-access queue at `TEST_MAX_SIZE`.
	const FIFO_CAPACITY: CacheSize = 1_000;

	/// Sets the fast tier so the *main queue* gets exactly `bytes` of it.
	///
	/// Necessary because `settle_fast_tier` checks
	/// `fast_capacity - fifo_capacity`, not raw `fast_capacity`: passing the
	/// desired main-queue budget straight to `handle_resize_fast_tier` (as the
	/// non-reprieve variant's tests do, where the same subtraction applies but
	/// the reservation is negligible at their sizes) would leave the main
	/// queue nothing and demote everything.
	fn set_main_fast_capacity(worker: &mut PolicyWorker<u32, TestBuffer>, bytes: CacheSize) {
		worker.handle_resize_fast_tier(FIFO_CAPACITY + bytes);
	}

	/// The whole point of this design: a brand-new key is admitted fast and
	/// stays fast, with no migration to correct it. Contrast
	/// `two_q_hybrid_tests::admission_lands_slow_and_promotion_physically_moves_bytes`.
	#[test]
	fn admission_lands_fast_and_needs_no_migration() {
		let (mut worker, objects, status, overhead_manager) = make_worker(TEST_MAX_SIZE);

		set_main_fast_capacity(&mut worker, base_size_of(&overhead_manager, 15) as CacheSize * 4);

		insert(&objects, &status, &overhead_manager, &mut worker, 1, 15);
		worker.apply_tier_migrations();
		worker.refresh_tier_gauges();

		let snapshot = status.two_q_fast_admission_reprieve_hybrid_stats();
		assert_eq!(snapshot.promotions, 0);
		assert_eq!(snapshot.demotions, 0);
		assert_eq!(snapshot.fast_objects, 1);
		assert_eq!(snapshot.slow_objects, 0);

		// The API layer built these bytes Fast and nothing rewrote them, so
		// the migrate closure's marker was never applied.
		let data = objects.get_ref(&1).unwrap().data();
		assert_ne!(data.last(), Some(&0x50));
		assert_ne!(data.last(), Some(&0xFA));
	}

	/// A FIFO->main promotion is Fast->Fast, so it must not rewrite bytes or
	/// bump the promotions counter -- the optimization this design unlocks.
	#[test]
	fn promotion_out_of_the_fifo_queue_moves_no_bytes() {
		let (mut worker, objects, status, overhead_manager) = make_worker(TEST_MAX_SIZE);

		set_main_fast_capacity(&mut worker, base_size_of(&overhead_manager, 15) as CacheSize * 4);

		insert(&objects, &status, &overhead_manager, &mut worker, 1, 15);
		worker.handle_get(1, true);
		worker.apply_tier_migrations();
		worker.refresh_tier_gauges();

		let snapshot = status.two_q_fast_admission_reprieve_hybrid_stats();
		assert_eq!(snapshot.promotions, 0);
		assert_eq!(snapshot.demotions, 0);
		assert_eq!(snapshot.fast_objects, 1);

		let data = objects.get_ref(&1).unwrap().data();
		assert_ne!(data.last(), Some(&0xFA));
	}

	/// Main-queue fast-tier pressure still moves real bytes DRAM->PMEM.
	#[test]
	fn demotion_physically_replaces_object_bytes_and_updates_stats() {
		let (mut worker, objects, status, overhead_manager) = make_worker(TEST_MAX_SIZE);

		// Room in the main queue for exactly one ~15-byte object.
		set_main_fast_capacity(&mut worker, base_size_of(&overhead_manager, 15) as CacheSize + 1);

		// Both keys need a second access to reach the main queue.
		insert(&objects, &status, &overhead_manager, &mut worker, 1, 15);
		worker.handle_get(1, true);
		insert(&objects, &status, &overhead_manager, &mut worker, 2, 10);
		worker.handle_get(2, true);

		worker.apply_tier_migrations();
		worker.refresh_tier_gauges();

		let snapshot = status.two_q_fast_admission_reprieve_hybrid_stats();
		assert_eq!(snapshot.demotions, 1);
		assert_eq!(snapshot.slow_objects, 1);

		// Key 1 was the main queue's LRU tail, so it took the demotion.
		let data = objects.get_ref(&1).unwrap().data();
		assert_eq!(data.last(), Some(&0x50));
	}

	/// Re-accessing a demoted main-queue key is a genuine PMEM->DRAM move,
	/// so unlike the FIFO promotion above it does bump `promotions`.
	#[test]
	fn re_accessing_a_demoted_key_promotes_it_with_a_real_byte_move() {
		let (mut worker, objects, status, overhead_manager) = make_worker(TEST_MAX_SIZE);

		set_main_fast_capacity(&mut worker, base_size_of(&overhead_manager, 15) as CacheSize + 1);

		insert(&objects, &status, &overhead_manager, &mut worker, 1, 15);
		worker.handle_get(1, true);
		insert(&objects, &status, &overhead_manager, &mut worker, 2, 10);
		worker.handle_get(2, true);
		worker.apply_tier_migrations();
		worker.refresh_tier_gauges();

		assert_eq!(status.two_q_fast_admission_reprieve_hybrid_stats().demotions, 1);

		worker.handle_get(1, true);
		worker.apply_tier_migrations();
		worker.refresh_tier_gauges();

		let snapshot = status.two_q_fast_admission_reprieve_hybrid_stats();
		assert_eq!(snapshot.promotions, 1);

		let data_1 = objects.get_ref(&1).unwrap().data();
		assert_eq!(data_1.last(), Some(&0xFA));
	}

	#[test]
	fn eviction_under_two_q_fast_admission_reprieve_hybrid_policy_is_recorded_in_stats() {
		// A tiny max_size relative to per-object policy overhead guarantees
		// apply_evictions has to remove at least one object; assert
		// self-consistency rather than a hardcoded count, same as the other
		// hybrids' analogs.
		let (mut worker, objects, status, overhead_manager) = make_worker(20);

		insert(&objects, &status, &overhead_manager, &mut worker, 1, 15);
		insert(&objects, &status, &overhead_manager, &mut worker, 2, 15);
		worker.apply_tier_migrations();
		worker.refresh_tier_gauges();

		let mut buffered_events = Vec::new();
		worker.apply_evictions(&mut buffered_events).unwrap();

		let evictions = status.two_q_fast_admission_reprieve_hybrid_stats().evictions;
		assert!(evictions >= 1);
		assert_eq!(objects.len() as u64, 2 - evictions);
	}

	/// The reprieve, end to end through the worker: an aged-out one-access
	/// key must have its bytes physically rewritten by `migrate` with the
	/// Slow marker, while remaining present in the object map.
	#[test]
	fn a_reprieved_key_has_its_bytes_physically_moved_and_stays_present() {
		let (mut worker, objects, status, overhead_manager) = make_worker(TEST_MAX_SIZE);

		// A generous main-queue budget: this test is about FIFO-capacity
		// pressure, not main-queue demotion.
		set_main_fast_capacity(&mut worker, base_size_of(&overhead_manager, 15) as CacheSize * 60);

		// k_in is 0.001 of max_size 1_000_000 => fifo_capacity 1_000, so a
		// handful of ~15-byte objects overflows it and forces a reprieve.
		for key in 1..=40u64 {
			insert(&objects, &status, &overhead_manager, &mut worker, key, 15);
		}

		worker.apply_tier_migrations();
		worker.refresh_tier_gauges();

		let snapshot = status.two_q_fast_admission_reprieve_hybrid_stats();

		assert!(snapshot.demotions > 0, "FIFO pressure should have reprieved");
		assert_eq!(snapshot.evictions, 0, "a reprieve must not evict");
		assert!(snapshot.slow_objects > 0);

		// Every key is still in the object map -- a reprieve removes nothing.
		assert_eq!(objects.len(), 40);

		// The reprieved (oldest) key's bytes carry the Slow marker.
		let data = objects.get_ref(&1).unwrap().data();
		assert_eq!(data.last(), Some(&0x50));
	}

	/// The stack relieves FIFO pressure itself, so it must never ask
	/// `apply_evictions` to do it -- see the stack's module doc for why
	/// routing a reprieve through `evict_one()` is a real bug.
	#[test]
	fn fifo_pressure_is_never_reported_as_eviction_pressure() {
		let (mut worker, objects, status, overhead_manager) = make_worker(TEST_MAX_SIZE);

		set_main_fast_capacity(&mut worker, base_size_of(&overhead_manager, 15) as CacheSize * 60);

		for key in 1..=40u64 {
			insert(&objects, &status, &overhead_manager, &mut worker, key, 15);
		}

		worker.apply_tier_migrations();

		let mut buffered_events = Vec::new();
		worker.apply_evictions(&mut buffered_events).unwrap();

		// max_size is nowhere near exceeded and FIFO pressure is relieved
		// internally, so the eviction loop should have removed nothing.
		assert_eq!(objects.len(), 40);
		assert_eq!(status.two_q_fast_admission_reprieve_hybrid_stats().evictions, 0);
	}

	/// `hybrid_stats()` (the design-neutral accessor) must agree with this
	/// design's own named accessor.
	#[test]
	fn hybrid_stats_mirrors_the_named_accessor() {
		let (mut worker, objects, status, overhead_manager) = make_worker(TEST_MAX_SIZE);

		set_main_fast_capacity(&mut worker, base_size_of(&overhead_manager, 15) as CacheSize + 1);

		insert(&objects, &status, &overhead_manager, &mut worker, 1, 15);
		worker.handle_get(1, true);
		insert(&objects, &status, &overhead_manager, &mut worker, 2, 10);
		worker.handle_get(2, true);
		worker.apply_tier_migrations();
		worker.refresh_tier_gauges();

		let named = status.two_q_fast_admission_reprieve_hybrid_stats();
		let common = status.hybrid_stats();

		assert_eq!(common.promotions, named.promotions);
		assert_eq!(common.demotions, named.demotions);
		assert_eq!(common.evictions, named.evictions);
		assert_eq!(common.fast_bytes_used, named.fast_bytes_used);
		assert_eq!(common.slow_bytes_used, named.slow_bytes_used);
		assert_eq!(common.fast_objects, named.fast_objects);
		assert_eq!(common.slow_objects, named.slow_objects);
		assert!(common.demotions > 0, "test should have produced a demotion to compare");
	}
}

#[cfg(all(test, feature = "two_q_hybrid_cache"))]
mod two_q_hybrid_tests {
	use super::*;

	use crate::{
		NoHasher,
		object::Object,
		status::AtomicStatus,
		object::overhead::OverheadManager,
	};

	type TestBuffer = Box<[u8]>;

	// k_in=1.0 keeps fifo_capacity == max_size, so these tests can focus on
	// fast-tier (main-queue) behavior via `handle_resize_fast_tier` without
	// incidentally tripping fifo-capacity pressure. Same rationale as
	// `lru_hybrid_tests`/`lfu_hybrid_tests::make_worker` otherwise.
	fn make_worker(max_size: CacheSize) -> (
		PolicyWorker<u32, TestBuffer>,
		ObjectMapRef<u32, TestBuffer>,
		StatusRef,
		OverheadManagerRef,
	) {
		let (_tx, rx) = unbounded::<WorkerEvent>();

		let objects: ObjectMapRef<u32, TestBuffer> =
			crate::new_hybrid_object_map();

		let policy = PaperPolicy::TwoQHybrid(1.0);
		let status = Arc::new(
			AtomicStatus::new(max_size, &[policy], policy).unwrap(),
		);

		let overhead_manager = Arc::new(OverheadManager::new(&status));

		let migrate: Box<dyn Fn(&TestBuffer, Tier) -> TestBuffer + Send + Sync> =
			Box::new(|bytes, tier| {
				let marker: u8 = match tier {
					Tier::Fast => 0xFA,
					Tier::Slow => 0x50,
				};

				let mut v = bytes.to_vec();
				if let Some(last) = v.last_mut() {
					*last = marker;
				}
				v.into_boxed_slice()
			});

		let worker = PolicyWorker::new_with_tier_migration(
			rx,
			objects.clone(),
			status.clone(),
			overhead_manager.clone(),
			migrate,
		).unwrap();

		(worker, objects, status, overhead_manager)
	}

	fn insert(
		objects: &ObjectMapRef<u32, TestBuffer>,
		status: &StatusRef,
		overhead_manager: &OverheadManagerRef,
		worker: &mut PolicyWorker<u32, TestBuffer>,
		key: HashedKey,
		size: usize,
	) {
		let object = Object::new(key as u32, vec![0u8; size].into_boxed_slice(), None);
		let base_size = overhead_manager.base_size(&object);

		objects.insert(key, object);
		status.update_base_used_size(base_size as i64);
		status.incr_num_objects();
		worker.handle_set(key, base_size);
	}

	fn base_size_of(overhead_manager: &OverheadManagerRef, size: usize) -> ObjectSize {
		let probe = Object::new(0u32, vec![0u8; size].into_boxed_slice(), None);
		overhead_manager.base_size(&probe)
	}

	#[test]
	fn admission_lands_slow_and_promotion_physically_moves_bytes() {
		let (mut worker, objects, status, overhead_manager) = make_worker(1_000);

		worker.handle_resize_fast_tier(base_size_of(&overhead_manager, 15) as CacheSize);

		insert(&objects, &status, &overhead_manager, &mut worker, 1, 15); // fifo, slow
		worker.apply_tier_migrations();
		worker.refresh_tier_gauges();

		// Admission produces no migration (physical and logical tier already
		// agree — see decision 3 in CLAUDE.md's `two_q_hybrid_cache` section),
		// so the gauges (only refreshed when a migration is drained) are
		// still at their initial zero here; only `promotions` is meaningful
		// to check at this point.
		let snapshot = status.two_q_hybrid_stats();
		assert_eq!(snapshot.promotions, 0);

		let data = objects.get_ref(&1).unwrap().data();
		assert_eq!(data.last(), Some(&0u8)); // untouched: admission builds no migration

		// Accessing the FIFO key promotes it straight to Main/Fast.
		worker.handle_get(1, true);
		worker.apply_tier_migrations();
		worker.refresh_tier_gauges();

		let snapshot = status.two_q_hybrid_stats();
		assert_eq!(snapshot.promotions, 1);
		assert_eq!(snapshot.fast_objects, 1);

		let data = objects.get_ref(&1).unwrap().data();
		assert_eq!(data.last(), Some(&0xFA));
	}

	#[test]
	fn promotion_can_cascade_a_demotion_within_main_queue() {
		let (mut worker, objects, status, overhead_manager) = make_worker(1_000);

		worker.handle_resize_fast_tier(base_size_of(&overhead_manager, 15) as CacheSize);

		insert(&objects, &status, &overhead_manager, &mut worker, 1, 15);
		insert(&objects, &status, &overhead_manager, &mut worker, 2, 15);
		worker.apply_tier_migrations();
		worker.refresh_tier_gauges();

		worker.handle_get(1, true); // promote 1 -> Main/Fast
		worker.apply_tier_migrations();
		worker.refresh_tier_gauges();

		// Fast tier only fits one 15-byte object, so promoting key 2 must
		// demote key 1 back down.
		worker.handle_get(2, true);
		worker.apply_tier_migrations();
		worker.refresh_tier_gauges();

		let snapshot = status.two_q_hybrid_stats();
		assert_eq!(snapshot.promotions, 2);
		assert_eq!(snapshot.demotions, 1);

		let data_2 = objects.get_ref(&2).unwrap().data();
		assert_eq!(data_2.last(), Some(&0xFA));
	}

	#[test]
	fn eviction_under_two_q_hybrid_policy_is_recorded_in_stats() {
		let (mut worker, objects, status, overhead_manager) = make_worker(20);

		insert(&objects, &status, &overhead_manager, &mut worker, 1, 15);
		insert(&objects, &status, &overhead_manager, &mut worker, 2, 15);
		worker.apply_tier_migrations();
		worker.refresh_tier_gauges();

		let mut buffered_events = Vec::new();
		worker.apply_evictions(&mut buffered_events).unwrap();

		let evictions = status.two_q_hybrid_stats().evictions;
		assert!(evictions >= 1);
		assert_eq!(objects.len() as u64, 2 - evictions);
	}

	#[test]
	fn no_migration_fn_is_a_safe_no_op() {
		let (_tx, rx) = unbounded::<WorkerEvent>();
		let objects: ObjectMapRef<u32, TestBuffer> =
			crate::new_hybrid_object_map();

		let policy = PaperPolicy::TwoQHybrid(1.0);
		let status = Arc::new(
			AtomicStatus::new(100, &[policy], policy).unwrap(),
		);
		let overhead_manager = Arc::new(OverheadManager::new(&status));

		let mut worker = PolicyWorker::<u32, TestBuffer>::new(
			rx,
			objects.clone(),
			status.clone(),
			overhead_manager.clone(),
			None,
		).unwrap();

		insert(&objects, &status, &overhead_manager, &mut worker, 1, 15);
		insert(&objects, &status, &overhead_manager, &mut worker, 2, 10);

		worker.apply_tier_migrations();
		worker.refresh_tier_gauges();
	}
}

#[cfg(all(test, feature = "lru_sized_hybrid_cache"))]
mod lru_sized_hybrid_tests {
	use super::*;

	use crate::{
		NoHasher,
		object::Object,
		status::AtomicStatus,
		object::overhead::{OverheadManager, get_hybrid_dram_shared_overhead},
	};

	type TestBuffer = Box<[u8]>;

	// See `lru_hybrid_tests::shared_overhead`'s identical rationale.
	#[allow(dead_code)]
	fn shared_overhead() -> CacheSize {
		get_hybrid_dram_shared_overhead(&PaperPolicy::LruSizedHybrid) as CacheSize
	}

	// See `lru_hybrid_tests::low_water_safe`'s identical rationale --
	// `LruSizedHybridStack` reuses the same `FAST_TIER_LOW_WATER_RATIO`
	// (0.98) for both segments.
	fn low_water_safe(target: CacheSize) -> CacheSize {
		(target as f64 / 0.98).ceil() as CacheSize
	}

	// Mirrors `lru_hybrid_tests::make_worker` exactly, seeded with
	// `PaperPolicy::LruSizedHybrid` instead.
	fn make_worker(max_size: CacheSize) -> (
		PolicyWorker<u32, TestBuffer>,
		ObjectMapRef<u32, TestBuffer>,
		StatusRef,
		OverheadManagerRef,
	) {
		let (_tx, rx) = unbounded::<WorkerEvent>();

		let objects: ObjectMapRef<u32, TestBuffer> =
			crate::new_hybrid_object_map();

		let status = Arc::new(
			AtomicStatus::new(max_size, &[PaperPolicy::LruSizedHybrid], PaperPolicy::LruSizedHybrid).unwrap(),
		);

		let overhead_manager = Arc::new(OverheadManager::new(&status));

		let migrate: Box<dyn Fn(&TestBuffer, Tier) -> TestBuffer + Send + Sync> =
			Box::new(|bytes, tier| {
				let marker: u8 = match tier {
					Tier::Fast => 0xFA,
					Tier::Slow => 0x50,
				};

				let mut v = bytes.to_vec();
				if let Some(last) = v.last_mut() {
					*last = marker;
				}
				v.into_boxed_slice()
			});

		let worker = PolicyWorker::new_with_tier_migration(
			rx,
			objects.clone(),
			status.clone(),
			overhead_manager.clone(),
			migrate,
		).unwrap();

		(worker, objects, status, overhead_manager)
	}

	// Mirrors `lru_hybrid_tests::insert` exactly.
	fn insert(
		objects: &ObjectMapRef<u32, TestBuffer>,
		status: &StatusRef,
		overhead_manager: &OverheadManagerRef,
		worker: &mut PolicyWorker<u32, TestBuffer>,
		key: HashedKey,
		size: usize,
	) {
		let object = Object::new(key as u32, vec![0u8; size].into_boxed_slice(), None);
		let base_size = overhead_manager.base_size(&object);

		objects.insert(key, object);
		status.update_base_used_size(base_size as i64);
		status.incr_num_objects();
		worker.handle_set(key, base_size);
	}

	// Mirrors `lru_hybrid_tests::base_size_of` exactly.
	fn base_size_of(overhead_manager: &OverheadManagerRef, size: usize) -> ObjectSize {
		let probe = Object::new(0u32, vec![0u8; size].into_boxed_slice(), None);
		overhead_manager.base_size(&probe)
	}

	#[test]
	fn demotion_physically_replaces_object_bytes_and_updates_stats() {
		let (mut worker, objects, status, overhead_manager) = make_worker(1_000_000);

		// Route everything to the small segment and make the large
		// segment's capacity dwarf the small one's, so the shared-metadata
		// reservation's share against small stays negligible (rounds to 0)
		// -- see `LruSizedHybridStack::reserved_shares`. Lets this test
		// reuse the same simple math `lru_hybrid_tests`'s equivalent uses.
		worker.handle_resize_size_threshold(1_000_000);
		worker.handle_resize_large_fast_tier(1_000_000);
		worker.handle_resize_fast_tier(
			low_water_safe(base_size_of(&overhead_manager, 15) as CacheSize + 1),
		);

		insert(&objects, &status, &overhead_manager, &mut worker, 1, 15); // fast
		insert(&objects, &status, &overhead_manager, &mut worker, 2, 10); // demotes key 1

		worker.apply_tier_migrations();
		worker.refresh_tier_gauges();

		let snapshot = status.lru_sized_hybrid_stats();
		assert_eq!(snapshot.demotions, 1);
		assert_eq!(snapshot.promotions, 0);
		assert_eq!(snapshot.fast_objects, 1);
		assert_eq!(snapshot.slow_objects, 1);
		assert_eq!(snapshot.small_fast_objects, 1);
		assert_eq!(snapshot.small_slow_objects, 1);
		assert_eq!(snapshot.large_fast_objects, 0);
		assert_eq!(snapshot.large_slow_objects, 0);

		let data = objects.get_ref(&1).unwrap().data();
		assert_eq!(data.last(), Some(&0x50));
	}

	#[test]
	fn access_promotes_a_slow_key_and_may_cascade_a_demotion() {
		let (mut worker, objects, status, overhead_manager) = make_worker(1_000_000);

		worker.handle_resize_size_threshold(1_000_000);
		worker.handle_resize_large_fast_tier(1_000_000);
		worker.handle_resize_fast_tier(
			low_water_safe(base_size_of(&overhead_manager, 15) as CacheSize + 1),
		);

		insert(&objects, &status, &overhead_manager, &mut worker, 1, 15);
		insert(&objects, &status, &overhead_manager, &mut worker, 2, 10); // demotes 1
		worker.apply_tier_migrations();
		worker.refresh_tier_gauges();

		worker.handle_get(1, true);
		worker.apply_tier_migrations();
		worker.refresh_tier_gauges();

		let snapshot = status.lru_sized_hybrid_stats();
		assert_eq!(snapshot.promotions, 1);
		assert_eq!(snapshot.demotions, 2);

		let data_1 = objects.get_ref(&1).unwrap().data();
		assert_eq!(data_1.last(), Some(&0xFA));
	}

	#[test]
	fn eviction_under_lru_sized_hybrid_policy_is_recorded_in_stats() {
		let (mut worker, objects, status, overhead_manager) = make_worker(20);

		insert(&objects, &status, &overhead_manager, &mut worker, 1, 15);
		insert(&objects, &status, &overhead_manager, &mut worker, 2, 15);
		worker.apply_tier_migrations();
		worker.refresh_tier_gauges();

		let mut buffered_events = Vec::new();
		worker.apply_evictions(&mut buffered_events).unwrap();

		let evictions = status.lru_sized_hybrid_stats().evictions;
		assert!(evictions >= 1);
		assert_eq!(objects.len() as u64, 2 - evictions);
	}

	#[test]
	fn no_migration_fn_is_a_safe_no_op() {
		let (_tx, rx) = unbounded::<WorkerEvent>();
		let objects: ObjectMapRef<u32, TestBuffer> =
			crate::new_hybrid_object_map();
		let status = Arc::new(
			AtomicStatus::new(100, &[PaperPolicy::LruSizedHybrid], PaperPolicy::LruSizedHybrid).unwrap(),
		);
		let overhead_manager = Arc::new(OverheadManager::new(&status));

		let mut worker = PolicyWorker::<u32, TestBuffer>::new(
			rx,
			objects.clone(),
			status.clone(),
			overhead_manager.clone(),
			None,
		).unwrap();

		insert(&objects, &status, &overhead_manager, &mut worker, 1, 15);
		insert(&objects, &status, &overhead_manager, &mut worker, 2, 10);

		worker.apply_tier_migrations();
		worker.refresh_tier_gauges();
	}

	#[test]
	fn small_and_large_segments_demote_independently() {
		let (mut worker, objects, status, overhead_manager) = make_worker(1_000_000);

		let small_base = base_size_of(&overhead_manager, 15) as CacheSize;
		let large_base = base_size_of(&overhead_manager, 200) as CacheSize;

		worker.handle_resize_size_threshold((small_base + large_base) / 2);
		// Generous capacities up front -- both objects land and stay fast.
		worker.handle_resize_fast_tier(10_000);
		worker.handle_resize_large_fast_tier(10_000);

		insert(&objects, &status, &overhead_manager, &mut worker, 1, 15);  // small
		insert(&objects, &status, &overhead_manager, &mut worker, 2, 200); // large
		worker.apply_tier_migrations();
		worker.refresh_tier_gauges();

		assert_eq!(status.lru_sized_hybrid_stats().small_fast_objects, 1);
		assert_eq!(status.lru_sized_hybrid_stats().large_fast_objects, 1);

		// Shrinking only the SMALL segment's budget demotes key 1 without
		// touching key 2 in the large segment.
		worker.handle_resize_fast_tier(0);
		worker.apply_tier_migrations();
		worker.refresh_tier_gauges();

		let snapshot = status.lru_sized_hybrid_stats();
		assert_eq!(snapshot.demotions, 1);
		assert_eq!(snapshot.small_slow_objects, 1);
		assert_eq!(snapshot.large_fast_objects, 1);
		assert_eq!(snapshot.large_slow_objects, 0);
	}

	#[test]
	fn reclassification_on_overwrite_moves_segments_without_a_migration() {
		let (mut worker, objects, status, overhead_manager) = make_worker(1_000_000);

		let small_base = base_size_of(&overhead_manager, 15) as CacheSize;
		let large_base = base_size_of(&overhead_manager, 200) as CacheSize;

		worker.handle_resize_size_threshold((small_base + large_base) / 2);
		worker.handle_resize_fast_tier(10_000);
		worker.handle_resize_large_fast_tier(10_000);

		insert(&objects, &status, &overhead_manager, &mut worker, 1, 15); // small
		worker.apply_tier_migrations();
		worker.refresh_tier_gauges();
		assert_eq!(status.lru_sized_hybrid_stats().small_fast_objects, 1);

		let before = status.lru_sized_hybrid_stats();

		// A re-`handle_set` for the same key with a larger size mirrors what
		// `PaperCache::set()` does on a real overwrite (recompute base_size,
		// call handle_set again) -- reclassifies key 1 into the large
		// segment.
		worker.handle_set(1, large_base as ObjectSize);
		worker.apply_tier_migrations();
		worker.refresh_tier_gauges();

		let after = status.lru_sized_hybrid_stats();
		assert_eq!(after.small_fast_objects, 0);
		assert_eq!(after.large_fast_objects, 1);
		// A fast<->fast reclassification never crosses Tier, so it's
		// invisible to the promotion/demotion counters.
		assert_eq!(after.promotions, before.promotions);
		assert_eq!(after.demotions, before.demotions);
	}
}
