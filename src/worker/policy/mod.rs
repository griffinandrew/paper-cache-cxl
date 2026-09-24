/*
 * Copyright (c) Kia Shakiba
 *
 * This source code is licensed under the GNU AGPLv3 license found in the
 * LICENSE file in the root directory of this source tree.
 */


/// Persistent migration queue: a standing pool that drains physical tier
/// copies continuously, decoupled from batch boundaries.
///
/// `parallel_migration` below fans a single *batch* out across a pool, which
/// measurement showed cannot help this workload: 99.4% of demotion volume
/// arrives as single-object batches (37M calls of exactly 1 object against 9
/// calls of >=16K), so there is nothing to fan out and the threshold check is
/// pure overhead. The work is genuinely fine-grained, not genuinely serial --
/// profiling attributes ~52% of the saturated `PolicyWorker` thread to the
/// copies (~37% `__memmove_avx_unaligned_erms` plus ~15% surrounding closure).
///
/// This module captures that work regardless of how it arrives: the worker
/// pushes `(key, tier)` pairs onto an unbounded channel and returns
/// immediately, and N consumer threads perform the allocate-copy-swap. The
/// queue entries are 16 bytes, so even a deep backlog costs little next to
/// the values themselves.
///
/// Correctness is unchanged from the inline path, which already did the copy
/// with no map guard held: consumers re-acquire the shard only to swap the
/// pointer, and the `Arc::ptr_eq` guard still rejects a migration whose value
/// was replaced while the copy was in flight. Two consumers racing the same
/// key is the same situation -- one swap wins, the other's `ptr_eq` fails and
/// it drops its copy.
///
/// Demotion/promotion counters live on the consumers, not on the worker.
/// Counting at enqueue time counted *intents*: every entry bumped a counter
/// and then still had three ways to move nothing at all -- the object was
/// gone, `migrate` declined, or the `Arc::ptr_eq` guard rejected a copy taken
/// from a superseded value. One measured run reported 17,946,549 demotions
/// against the 3,894,278 `migstats` saw drained, a 4.6x overstatement. The
/// consumer is the only thread that knows whether `Object::set_data` ran, so
/// the consumer is the thread that counts.
///
/// That costs less, not more: each consumer accumulates completions in plain
/// local `u64`s and flushes them with a single `fetch_add` whenever its
/// channel momentarily drains (plus once when the loop exits), so a burst of
/// N migrations pays one atomic instead of N. Dropping the worker's per-entry
/// increment on top of that makes the whole change a net *removal* of atomic
/// operations.
///
/// On by default: `MIGRATION_QUEUE_THREADS` sets the consumer count, and 0
/// disables the queue entirely, in which case migrations apply inline on the
/// worker exactly as before.
#[cfg(feature = "hybrid_cache_common")]
pub mod migration_queue {
	use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
	use std::sync::{Arc, OnceLock};
	use std::thread::JoinHandle;

	use crossbeam_channel::{Sender, unbounded};

	use crate::object_store::ObjectStore;
	use crate::{HashedKey, ObjectMapRef, StatusRef};
	use crate::worker::policy::policy_stack::Tier;

	/// Default consumer count.
	///
	/// Measured on the benchmark traces (cluster12, 15 GB cache / 5 GB fast
	/// tier): 1 and 2 consumers both hold `used_size` exactly at the
	/// configured cap where the inline path overshot it 2.6x, at the best
	/// observed SET latency (~1.6us). 3 and above also hold the cap but cost
	/// latency, and shift which objects the cache retains -- at 4 consumers
	/// the same byte budget holds 21% more (smaller) objects, moving the miss
	/// ratio by ~6 points. 2 is chosen over 1 for a little headroom on a
	/// busier host while staying below that shift.
	pub const DEFAULT_THREADS: usize = 2;

	/// Consumer count. `MIGRATION_QUEUE_THREADS=0` disables the queue
	/// entirely and leaves every migration inline on the worker, which is the
	/// behaviour that predates this module.
	pub fn threads() -> usize {
		static THREADS: OnceLock<usize> = OnceLock::new();

		*THREADS.get_or_init(|| {
			std::env::var("MIGRATION_QUEUE_THREADS")
				.ok()
				.and_then(|value| value.parse::<usize>().ok())
				.unwrap_or(DEFAULT_THREADS)
		})
	}

	/// Increments the completion counter however the loop body exits, so a
	/// migration skipped because its object vanished still counts as done.
	/// High-water mark of `enqueued - processed`, i.e. the deepest the pool
	/// has ever fallen behind. Purely observational and process-global
	/// (unlike the per-queue counters that drive `flush`), so a run can
	/// report whether the queue ever actually backed up.
	pub static DEPTH_MAX: AtomicU64 = AtomicU64::new(0);

	/// Largest single batch ever handed to the consumers, in entries.
	///
	/// `DEPTH_MAX` alone cannot separate the two ways a queue gets deep: one
	/// big slug, or sustained overproduction. When the consumers keep up,
	/// `DEPTH_MAX` is set by the largest BURST and this is the number that
	/// explains it; when they do not, `DEPTH_MAX` runs far past any burst and
	/// the queue is throughput-bound instead. Mean batch size cannot tell them
	/// apart -- it was identical (2.8) across two designs whose `DEPTH_MAX`
	/// differed 3.4x.
	pub static BURST_MAX: AtomicU64 = AtomicU64::new(0);

	/// Migrations enqueued but not yet applied, split by destination tier.
	///
	/// These are the PHYSICAL mis-placement: a pending `Tier::Slow` entry is an
	/// object the policy stack already counts as slow and has already removed
	/// from `fast_used`, whose bytes are still in DRAM. A pending `Tier::Fast`
	/// is the reverse.
	///
	/// The stack cannot account any other way -- `settle_fast_tier` reads
	/// `fast_used` to decide whether to keep demoting, so completion-time
	/// accounting would make it drain the entire fast tier in one pass, never
	/// seeing its own decisions register. So `fast_used` is INTENT by
	/// necessity, and these two counters are the gap between intent and
	/// placement.
	///
	/// That gap is not only a reporting error. It biases latency: an object the
	/// stack believes is in Optane but which is physically still in DRAM is
	/// SERVED FROM DRAM, so a tiered run reports more fast-tier-speed hits than
	/// its own tier assignment implies, and the bias grows with the backlog.
	pub static PENDING_DEMOTE: AtomicU64 = AtomicU64::new(0);
	pub static PENDING_PROMOTE: AtomicU64 = AtomicU64::new(0);
	pub static PENDING_DEMOTE_MAX: AtomicU64 = AtomicU64::new(0);
	pub static PENDING_PROMOTE_MAX: AtomicU64 = AtomicU64::new(0);

	/// Peak of `PENDING_DEMOTE - PENDING_PROMOTE`, sampled together.
	///
	/// The DRAM overrun is the NET mis-placement -- pending demotions are bytes
	/// still in DRAM, pending promotions are bytes still in Optane, and they
	/// cancel. Subtracting the two separate high-water marks is wrong: they
	/// need not peak at the same instant, so their difference is neither the
	/// peak of the difference nor the difference at any single moment. This
	/// samples both counters at one point and takes the maximum of the result.
	pub static PENDING_NET_MAX: AtomicU64 = AtomicU64::new(0);

	/// The three ways `apply_migration` moves nothing, counted separately.
	///
	/// Together they measure how much of the queue is WASTED work -- entries
	/// the consumers pay a dequeue, a lock and a lookup for, and which copy no
	/// bytes. `GONE` is an object evicted while its migration sat in the queue;
	/// `DECLINED` is one already in the tier the entry asks for, which happens
	/// when an earlier entry for the same key already moved it; `SUPERSEDED` is
	/// a value replaced by a `set` mid-copy.
	///
	/// DECLINED counts less than it used to. `split_tier_migrations` already
	/// drops, inside one drain, every entry that a later entry for the other
	/// tier supersedes, so a key reversed within a drain no longer reaches the
	/// queue as a pair. What reaches DECLINED now is chiefly redundancy ACROSS
	/// drains (an entry for a key that an entry from an earlier drain already
	/// moved), same-tier duplicates within a drain (kept on purpose: the
	/// second declines), and the overwrite-restore no-op -- a `(k, Fast)` that
	/// finds the value a `set` built in DRAM. GONE, and the cross-drain part of
	/// DECLINED, are what a key-keyed pending map spanning drains would still
	/// drop before dispatch.
	pub static MIG_GONE: AtomicU64 = AtomicU64::new(0);
	pub static MIG_DECLINED: AtomicU64 = AtomicU64::new(0);
	pub static MIG_SUPERSEDED: AtomicU64 = AtomicU64::new(0);
	pub static MIG_APPLIED: AtomicU64 = AtomicU64::new(0);


	/// Decrements the pending counter for `tier` however the consumer loop body
	/// exits. Same reasoning as `CountOnDrop`: an entry leaves the queue whether
	/// or not `apply_migration` moved anything, and a leaked increment here
	/// would make the backlog look permanent.
	struct PendingOnDrop(Tier);

	impl Drop for PendingOnDrop {
		fn drop(&mut self) {
			let counter = match self.0 {
				Tier::Fast => &PENDING_PROMOTE,
				Tier::Slow => &PENDING_DEMOTE,
			};

			counter.fetch_sub(1, Ordering::Release);
		}
	}

	/// Records one enqueued migration against its tier and updates that tier's
	/// high-water mark.
	fn record_pending(tier: Tier) {
		let (counter, peak) = match tier {
			Tier::Fast => (&PENDING_PROMOTE, &PENDING_PROMOTE_MAX),
			Tier::Slow => (&PENDING_DEMOTE, &PENDING_DEMOTE_MAX),
		};

		peak.fetch_max(counter.fetch_add(1, Ordering::Release) + 1, Ordering::Relaxed);

		// Both counters read at one point, so the difference is a real
		// instantaneous net rather than a difference of two unrelated peaks.
		let net = PENDING_DEMOTE
			.load(Ordering::Acquire)
			.saturating_sub(PENDING_PROMOTE.load(Ordering::Acquire));

		PENDING_NET_MAX.fetch_max(net, Ordering::Relaxed);
	}

	struct CountOnDrop<'a>(&'a Arc<AtomicU64>);

	impl Drop for CountOnDrop<'_> {
		fn drop(&mut self) {
			self.0.fetch_add(1, Ordering::Release);
		}
	}

	/// Test-only rendezvous at the one point a concurrent `ttl()` can still be
	/// lost: after `migrated_to` has read the expiry and built the copy, before
	/// the write guard is taken for the swap. Keyed, because `apply_migration`
	/// runs from every migration test in the process and an unkeyed hook would
	/// park whichever migration reached it first.
	#[cfg(test)]
	pub(crate) mod after_copy {
		use std::sync::Mutex;

		use crossbeam_channel::{Receiver, Sender, unbounded};

		use crate::HashedKey;

		static PARK: Mutex<Option<(HashedKey, Sender<()>, Receiver<()>)>> = Mutex::new(None);

		/// Arms the rendezvous for `key`: returns (entered, release).
		pub(crate) fn arm(key: HashedKey) -> (Receiver<()>, Sender<()>) {
			let (entered_tx, entered_rx) = unbounded();
			let (release_tx, release_rx) = unbounded();

			*PARK.lock().unwrap() = Some((key, entered_tx, release_rx));

			(entered_rx, release_tx)
		}

		pub(crate) fn arrive(key: HashedKey) {
			let armed = {
				let mut park = PARK.lock().unwrap();

				match park.as_ref() {
					Some((armed_key, _, _)) if *armed_key == key => park.take(),
					_ => None,
				}
			};

			if let Some((_, entered, release)) = armed {
				let _ = entered.send(());
				let _ = release.recv();
			}
		}
	}

	/// Performs one physical tier migration: snapshot, rebuild, swap.
	///
	/// Returns `true` if and only if `Object::set_data` actually ran -- the one
	/// point at which a migration has moved anything. The three ways it returns
	/// `false` (the object is gone, `migrate` declined because the value is
	/// already in the requested tier, or the value was replaced while the copy
	/// was in flight) are all legitimate no-ops that displaced nothing, so none
	/// of them may be counted as a promotion or a demotion.
	///
	/// Shared by the consumer threads and by `apply_tier_migrations`'
	/// synchronous path (`MIGRATION_QUEUE_THREADS=0`) so the two cannot drift
	/// apart on either the swap or what counts as a completion.
	pub(crate) fn apply_migration<K: Clone, V>(
		objects: &ObjectMapRef<K, V>,
		key: HashedKey,
		tier: Tier,
	) -> bool {
		// The snapshot is a STRONG REFERENCE, and it is what makes every step
		// below sound. It replaces the epoch pin this function used to take,
		// and it is strictly simpler because the proof is the handle itself
		// rather than a discipline the caller has to maintain:
		//
		//   * the source bytes stay readable with NO shard guard held, which
		//     is what stops a multi-KB (possibly CXL) copy from serialising
		//     against readers. Whoever replaces this object meanwhile
		//     decrements a count that is not yet zero, and frees nothing.
		//   * the identity check is immune to ABA for the same reason. The old
		//     header cannot be freed while this handle is live, so its address
		//     cannot be recycled into a different value, and `ptr_eq` is
		//     therefore EXACT rather than merely probable. Comparing bytes
		//     would not be: a `set` that wrote identical content is a
		//     different value and must be rejected.
		let Some(old_value) = objects.get_ref(&key).map(|object| object.snapshot()) else {
			MIG_GONE.fetch_add(1, Ordering::Relaxed);
			return false;
		};

		// Declined: already in the requested tier, nothing to move.
		//
		// This used to be a caller-supplied `migrate` closure, boxed into the
		// worker and cloned into the migration queue, because building the
		// replacement needed the shape-specific `TieredBuffer::new_fast` /
		// `new_slow`. It does not any more: a value knows its own tier and can
		// copy itself into another one, carrying its key and its current
		// expiry, so the whole plumbing collapses to these four lines.
		if old_value.tier() == tier {
			MIG_DECLINED.fetch_add(1, Ordering::Relaxed);
			return false;
		}

		let new_value = old_value.migrated_to(tier);

		// The expiry the copy carried, read here -- outside the guard -- so
		// the swap below only has to compare. `new_value` is private to this
		// thread, so this cannot change before the swap.
		let copied_expiry = new_value.expiry();

		#[cfg(test)]
		after_copy::arrive(key);

		// Check-and-act under one shard write lock: any writer must take the
		// same lock, so nothing can replace the value between the comparison
		// and the swap.
		if let Some(mut object) = objects.get_mut_ref(&key) {
			if crate::TieredValue::ptr_eq(object.value(), &old_value) {
				// Carry the LIVE expiry across, here, under the guard.
				// `migrated_to` read it before the copy with no guard held, so
				// a `ttl()` that landed during the copy is on the old value
				// only, and publishing the copy as built would undo it: a TTL
				// extended mid-copy expires early, a cleared one comes back,
				// a shortened one outlives its deadline. `ttl()` stores under
				// this same write guard, so nothing can change the expiry
				// between this read and the swap, and `new_value` is not yet
				// published, so the store is private to this thread. (Review
				// finding values-1 / correctness-lib-4.)
				//
				// Stored only when it differs from what the copy carried.
				// Under `thin_header` the expiry is in the item, and the new
				// item is on the TARGET tier, so an unconditional store would
				// be a CXL write under the shard write lock on every
				// demotion; only a `ttl()` that raced the copy needs one. The
				// read of the old expiry is still a CXL read on a promotion
				// there, and that one cannot move: it is the check.
				let live_expiry = old_value.expiry();

				if live_expiry != copied_expiry {
					new_value.set_expiry(live_expiry);
				}

				let superseded = object.set_data(new_value);

				// Unpublished under the write guard. Dropping the handle is
				// the whole retirement: if a reader lifted this value out a
				// moment ago it still holds a reference and the free waits
				// for it, and if not the count reaches zero here and the
				// value goes back to its allocator immediately -- one
				// allocation or two, depending on `fused_value`. No deferral,
				// and nothing for a later epoch advance to run.
				drop(superseded);

				MIG_APPLIED.fetch_add(1, Ordering::Relaxed);
				return true;
			}
		}

		// Superseded, or the object vanished between the two lookups. The copy
		// was NEVER PUBLISHED -- no other thread has ever seen it -- so
		// dropping it here is its only decrement and frees it outright.
		drop(new_value);

		MIG_SUPERSEDED.fetch_add(1, Ordering::Relaxed);
		false
	}

	pub struct MigrationQueue {
		/// One channel per consumer, indexed by `key % senders.len()`.
		///
		/// A single shared channel with N consumers would let two migrations
		/// for the *same* key be applied out of order: a demote and a
		/// following promote can be picked up concurrently by different
		/// consumers, and whichever wins the shard lock last is the one whose
		/// `Arc::ptr_eq` fails and gets discarded. Each swap is individually
		/// correct -- the value is never corrupted, since `ptr_eq` still
		/// rejects any copy taken from a superseded value -- but the survivor
		/// can be the *older* decision, leaving the object physically in a
		/// tier the policy stack no longer believes it is in. That does not
		/// self-heal: the stack already records the newer tier, so it has no
		/// reason to re-emit a migration for that key.
		///
		/// Routing by key removes the hazard by construction. Every migration
		/// for a given key lands in exactly one channel, and a channel is
		/// drained FIFO by exactly one consumer, so per-key emission order is
		/// preserved end to end. Different keys still proceed in parallel,
		/// which is where the throughput comes from.
		///
		/// Emptied by `Drop` before joining: dropping the senders is what
		/// makes each consumer's `recv` return `Err` and its loop exit.
		senders: Vec<Sender<(HashedKey, Tier)>>,
		handles: Vec<JoinHandle<()>>,

		/// Items handed to the pool, and items the pool has finished with.
		/// `flush` waits for the second to catch up to the first. Both count
		/// *dispositions*, not successful swaps: a migration whose object was
		/// evicted or superseded is finished as far as the queue is concerned,
		/// so it must be counted or `flush` would never return.
		enqueued: AtomicU64,
		processed: Arc<AtomicU64>,

		/// Whether a *completed* `Tier::Slow` migration is a demotion for the
		/// policy currently running, i.e. `PolicyStack::
		/// inline_demotion_accounting`. Mirrored here because the consumers have
		/// no access to the stack, and the answer is a property of the policy
		/// rather than of the move: the LFU-style design lands new objects fast
		/// unconditionally and corrects them to slow, which needs the same
		/// physical `set_data` but displaces nothing, so it is not a demotion in
		/// the paper's sense (that design reports its real demotions through
		/// `PolicyStack::drain_demotions` instead).
		///
		/// Written once per `apply_tier_migrations` pass by the worker and read
		/// once per completed slow move by a consumer, both `Relaxed`: a single
		/// hot, uncontended cache line, and the channel send/recv the value
		/// travels alongside already orders the two.
		demotion_accounting: Arc<AtomicBool>,
	}

	impl MigrationQueue {
		/// Returns `None` when `threads == 0`, i.e. the queue is disabled.
		pub fn spawn<K, V>(
			objects: ObjectMapRef<K, V>,
			threads: usize,
			status: StatusRef,
		) -> Option<Self>
		where
			K: 'static + Eq + Clone + Send + Sync,
			V: 'static + Send + Sync,
		{
			if threads == 0 {
				return None;
			}

			let processed = Arc::new(AtomicU64::new(0));

			// Seeded with the `PolicyStack` trait default; the first
			// `apply_tier_migrations` pass overwrites it before it can push
			// anything, so this value is never actually read.
			let demotion_accounting = Arc::new(AtomicBool::new(true));

			let mut senders = Vec::with_capacity(threads);
			let mut handles = Vec::with_capacity(threads);

			for index in 0..threads {
				// Per-consumer channel rather than one shared queue -- see the
				// ordering note on `senders`.
				let (sender, receiver) = unbounded::<(HashedKey, Tier)>();

				let objects = objects.clone();
				let processed = processed.clone();
				let status = status.clone();
				let demotion_accounting = demotion_accounting.clone();

				let handle = std::thread::Builder::new()
					.name(format!("mig-{index}"))
					.spawn(move || {
						// Same binding as the policy worker: these threads do the
						// allocate-copy-swap for every migration, so their own
						// placement matters more than anything else here.
						#[cfg(feature = "numa_jemalloc")]
						crate::numa_alloc::bind_worker_thread_if_configured();

						// Completions are tallied here rather than at enqueue
						// time on the worker: `apply_migration` returns `true`
						// only when `Object::set_data` actually ran, so a
						// migration whose object vanished, whose `migrate`
						// declined, or whose `Arc::ptr_eq` guard rejected a
						// superseded copy contributes nothing.
						//
						// Plain non-atomic locals, flushed once per burst. The
						// loop already pays one atomic per item for
						// `CountOnDrop`; a second per-item `fetch_add` would
						// double that, which is exactly what this avoids.
						let mut completed_promotions: u64 = 0;
						let mut completed_demotions: u64 = 0;

						while let Ok((key, tier)) = receiver.recv() {
							// Counted on every path out of this iteration.
							let _done = CountOnDrop(&processed);
							let _pending = PendingOnDrop(tier);

							if apply_migration(&objects, key, tier) {
								match tier {
									Tier::Fast => completed_promotions += 1,

									// Not every completed slow move is a demotion
									// -- see `demotion_accounting`'s doc on the
									// struct. A `Relaxed` load off a line this
									// thread already owns.
									Tier::Slow => {
										if demotion_accounting.load(Ordering::Relaxed) {
											completed_demotions += 1;
										}
									},
								}
							}

							// Flush when the burst momentarily drains.
							// `is_empty` is only consulted when there is
							// something to flush, so a burst that completes
							// nothing adds no work at all.
							//
							// Placed before `_done` drops, so the `Release`
							// increment of `processed` publishes these counters
							// too: a `flush()` that has observed `processed`
							// (`Acquire`) is guaranteed to see them.
							if (completed_promotions | completed_demotions) != 0 && receiver.is_empty() {
								status.record_hybrid_promotions(completed_promotions);
								status.record_hybrid_demotions(completed_demotions);

								completed_promotions = 0;
								completed_demotions = 0;

								// And push this thread's epoch bag out, for the same
								// reason the policy worker flushes once per event-loop
								// pass: a completed migration retires the value it
								// displaced, and this thread then blocks in `recv()`
								// holding up to a bag's worth of it (62 objects) until
								// its next burst. Bounded, so it was never a leak -- but
								// at 8 KiB values that is half a megabyte of garbage
								// sitting in an idle thread, which is what `flush` is for.
								//
								// The condition is already exactly right: nothing
								// completed means nothing was retired, and a declined
								// migration frees its copy outright rather than deferring.
							}
						}

						// Channel closed: publish whatever the last burst left.
						status.record_hybrid_promotions(completed_promotions);
						status.record_hybrid_demotions(completed_demotions);
					});

				match handle {
					Ok(handle) => {
						handles.push(handle);
						senders.push(sender);
					},
					// Partial spawn is still usable: `push` shards over
					// however many consumers actually started, so ordering
					// still holds -- just with less parallelism.
					Err(_) => break,
				}
			}

			if handles.is_empty() {
				return None;
			}

			Some(MigrationQueue {
				senders,
				handles,
				enqueued: AtomicU64::new(0),
				processed,
				demotion_accounting,
			})
		}

		/// Hands one migration to the consumer that owns this key.
		///
		/// `HashedKey` is already a hash, so the low bits are well distributed
		/// and a modulo is an adequate shard selector. A send failure means
		/// that consumer is gone, which only happens during shutdown; the
		/// stack state is already correct either way, so the copy is dropped.
		pub fn push(&self, item: (HashedKey, Tier)) {
			let Some(first) = self.senders.first() else {
				return;
			};

			// Charge the pending counter BEFORE publishing, not after.
			//
			// `send` is the publication point: the instant it returns, a
			// consumer parked in `recv` owns the item and can run its entire
			// iteration -- including `PendingOnDrop`'s `fetch_sub` -- before
			// this thread reaches the next statement. Charging afterwards let
			// that decrement run before its own increment, which took the
			// counter from 0 to `u64::MAX` and made the `+ 1` inside
			// `record_pending` panic on the checked add in a debug build (and
			// wrap silently in release). The panic killed the policy worker,
			// and a dead policy worker is what strands the TTL worker's
			// `Shutdown` and hangs `PaperCache::drop` in `join` -- the two
			// defects are one chain, not two independent ones.
			//
			// The counters were never mismatched: one increment site, one
			// decrement site, exactly one of each per accepted item. Only the
			// order was wrong. A refused send therefore has to refund the
			// charge, which `PendingOnDrop`'s own `Drop` already knows how to
			// do -- constructing and dropping one is the decrement.
			record_pending(item.1);

			// Single consumer: one FIFO channel already preserves global
			// order, so there is nothing to shard and the modulo is skipped.
			// Sharding only does work when there is more than one consumer to
			// distribute across.
			if self.senders.len() == 1 {
				match first.send(item).is_ok() {
					true => self
						.record_depth(self.enqueued.fetch_add(1, Ordering::Release) + 1),
					false => drop(PendingOnDrop(item.1)),
				}

				return;
			}

			let shard = (item.0 % self.senders.len() as HashedKey) as usize;

			match self.senders[shard].send(item).is_ok() {
				true => self
					.record_depth(self.enqueued.fetch_add(1, Ordering::Release) + 1),
				false => drop(PendingOnDrop(item.1)),
			}
		}

		/// Updates the global high-water mark from an enqueue count.
		fn record_depth(&self, enqueued: u64) {
			let depth = enqueued.saturating_sub(self.processed.load(Ordering::Acquire));

			DEPTH_MAX.fetch_max(depth, Ordering::Relaxed);
		}

		/// Tells the consumers whether a completed `Tier::Slow` move counts as a
		/// demotion for the policy stack that is about to hand them work -- see
		/// `demotion_accounting`'s doc on the struct.
		///
		/// One `Relaxed` store per `apply_tier_migrations` pass, not per
		/// migration. The value is constant for the life of a hybrid cache (a
		/// hybrid cache exposes no way to switch policy), so this is a cheap way
		/// to keep the consumers policy-agnostic rather than a hot write.
		pub fn set_demotion_accounting(&self, enabled: bool) {
			self.demotion_accounting.store(enabled, Ordering::Relaxed);
		}

		/// Blocks until every migration handed to the pool so far has been
		/// applied or discarded.
		///
		/// With the queue enabled `apply_tier_migrations` returns as soon as
		/// the batch is handed off, so the policy stack's tier tags are
		/// up to date before the bytes have physically moved. Callers that
		/// need the two to agree -- tests asserting on buffer contents, or a
		/// caller about to measure tier residency -- call this first.
		pub fn flush(&self) {
			let target = self.enqueued.load(Ordering::Acquire);

			while self.processed.load(Ordering::Acquire) < target {
				std::thread::yield_now();
			}
		}
	}

	impl Drop for MigrationQueue {
		fn drop(&mut self) {
			// Close the channel first so consumers finish the backlog and
			// then exit, rather than being detached mid-copy.
			self.senders.clear();

			for handle in self.handles.drain(..) {
				let _ = handle.join();
			}
		}
	}

	#[cfg(test)]
	mod tests {
		use super::*;
		use crossbeam_channel::bounded;

		/// A rendezvous channel parks the producer inside `push` at exactly the
		/// send, which makes the charge-then-publish ordering observable with no
		/// race at all: while the producer is parked, the item is not yet visible
		/// to any consumer, so the pending counter must already show it.
		///
		/// Against a `push` that charges AFTER the send this fails on the first
		/// assertion, and it fails deterministically rather than flakily.
		/// NEEDS THE PROCESS TO ITSELF, and is `#[ignore]`d for it. Run with
		/// `--ignored --exact --test-threads=1`.
		///
		/// `PENDING_DEMOTE` is process-global and this reads it as a delta, but
		/// the threads that move it are other tests' migration CONSUMERS, not
		/// their test bodies -- so a shared lock between tests does not
		/// serialise anything that matters. Observed failing four runs in ten
		/// with `left: 1, right: 2`: a consumer elsewhere decremented between
		/// the baseline read and the check. The assertion is exact on purpose,
		/// because a `>=` form would pass against the unfixed `push` whenever
		/// another test happened to leave the counter non-zero, which is the
		/// entire discrimination this test exists for.
		#[test]
		#[ignore]
		fn push_charges_pending_before_the_item_can_reach_a_consumer() {
			let (sender, receiver) = bounded::<(HashedKey, Tier)>(0);

			let queue = MigrationQueue {
				senders: vec![sender],
				handles: Vec::new(),
				enqueued: AtomicU64::new(0),
				processed: Arc::new(AtomicU64::new(0)),
				demotion_accounting: Arc::new(AtomicBool::new(true)),
			};

			// These counters are process-global, so measure this push as a delta
			// and take the baseline while the queue is quiescent -- which means
			// holding off every other test that moves them, since cargo runs
			// them in parallel.
			let _serial = crate::global_counter_lock();

			let before = PENDING_DEMOTE.load(Ordering::Acquire);

			let parked = Arc::new(AtomicBool::new(false));
			let signal = parked.clone();

			let producer = std::thread::spawn(move || {
				signal.store(true, Ordering::Release);
				queue.push((7, Tier::Slow))
			});

			// The producer signals immediately before `push`, and the rendezvous
			// nothing has received from is the only place `push` can block, so
			// once the flag is up it is parked at the send within a few hundred
			// nanoseconds. The sleep is many orders of magnitude more than that.
			while !parked.load(Ordering::Acquire) {
				std::thread::yield_now();
			}

			std::thread::sleep(std::time::Duration::from_millis(250));

			let charged = PENDING_DEMOTE.load(Ordering::Acquire);

			// Checked BEFORE the decrement below, so an unfixed `push` is reported
			// as the ordering bug it is rather than as a mystery panic on the
			// producer thread -- and so a failing run does not leave the global
			// counter wrapped underneath every other test in this binary.
			assert_eq!(
				charged,
				before + 1,
				"`push` published the item before charging it: a consumer that \
				 dequeues inside that window decrements a counter which was never \
				 incremented, wrapping it to u64::MAX and panicking the next enqueue",
			);

			// Now take the item and pay the charge back exactly as the consumer
			// loop does. This is the unmatched-decrement sequence itself: against
			// the unfixed `push` it wraps the counter and the producer's own
			// `record_pending` panics on the checked add, so the join below fails
			// too. Against the fixed one the pair is balanced and the counter is
			// left exactly as this test found it.
			let (key, tier) = receiver
				.recv()
				.expect("the producer must still be parked in `send`");

			drop(PendingOnDrop(tier));

			producer.join().expect("`push` must not panic");

			assert_eq!(key, 7);
			assert_eq!(tier, Tier::Slow);
			assert_eq!(PENDING_DEMOTE.load(Ordering::Acquire), before);
		}
	}

}

/// Optional parallel application of tier-migration batches.
///
/// `apply_tier_migrations` builds each destination buffer with `migrate`
/// (a real allocation plus a full byte copy of the value) and then swaps the
/// pointer under the object map's shard guard. Profiling the saturated
/// `PolicyWorker` thread on the benchmark traces attributes ~52% of its time
/// to that work -- ~37% in `__memmove_avx_unaligned_erms` for the copies
/// themselves plus ~15% in the surrounding closure -- while the remaining
/// ~38% is hash-table mutation on the single policy stack, which is
/// inherently serial and stays on the worker.
///
/// The copies, unlike the stack mutation, share no mutable state: `migrate`
/// is `Fn + Send + Sync`, the object map is a `DashMap`, and the demotion /
/// promotion counters are atomics. So a batch can be fanned out across a
/// dedicated pool.
///
/// Compiled in unconditionally and gated at run time on batch length, since
/// the win is entirely batch-size dependent: every hybrid stack holds its fast
/// tier at a single continuous threshold, so a settle moves only what the event
/// that triggered it displaced and the overwhelming majority of calls carry
/// 0 or 1 object, where a fan-out would be pure overhead. Only batches at or above
/// [`threshold`] go to the pool; everything else runs inline exactly as
/// before.
pub mod parallel_migration {
	use std::sync::OnceLock;

	use rayon::ThreadPool;

	use crate::HashedKey;
	use crate::worker::policy::policy_stack::Tier;

	/// Zero, i.e. parallel application is off by default: every batch
	/// runs inline on the worker regardless of size. Set
	/// `PARALLEL_MIGRATION_THRESHOLD` to a non-zero value to re-enable
	/// the fan-out.
	///
	/// A measured dead end rather than a tuned value. The batch-size
	/// distribution leaves nothing to fan out: 99.4% of demotion volume
	/// arrives as single-object batches, so a threshold low enough to
	/// engage at all would mostly have the pool paying dispatch cost to
	/// hand one object to one thread. The ~626K-object passes this
	/// module was built for only appeared under a wide high/low drain band
	/// (since removed), and were far too rare to pay back the machinery -- so the value is 0,
	/// not a compromise picked somewhere between 1 and that size.
	///
	/// Superseded by the `migration_queue` module above, which takes the
	/// same win without depending on batch size: a standing pool of
	/// consumers sharded by key hash drains migrations off the worker as
	/// they are produced, one object at a time.
	pub const DEFAULT_THRESHOLD: usize = 0;

	/// Pool size when parallel application does engage.
	pub const DEFAULT_THREADS: usize = 4;

	static THRESHOLD: OnceLock<usize> = OnceLock::new();
	static POOL: OnceLock<Option<ThreadPool>> = OnceLock::new();

	/// `PARALLEL_MIGRATION_THRESHOLD=0` disables parallel application
	/// entirely -- every batch runs inline, which is exactly the behaviour
	/// that predates this module.
	pub fn threshold() -> usize {
		*THRESHOLD.get_or_init(|| {
			std::env::var("PARALLEL_MIGRATION_THRESHOLD")
				.ok()
				.and_then(|value| value.parse::<usize>().ok())
				.unwrap_or(DEFAULT_THRESHOLD)
		})
	}

	/// Dedicated pool rather than rayon's global one: migration is latency
	/// -sensitive background work and should not queue behind, or be starved
	/// by, anything else that happens to use rayon. Threads are named
	/// `mig-N` so they are identifiable in `perf`/`top`. Returns `None` if
	/// the pool could not be built, in which case callers fall back to
	/// inline application rather than failing the migration.
	fn pool() -> Option<&'static ThreadPool> {
		POOL.get_or_init(|| {
			let threads = std::env::var("PARALLEL_MIGRATION_THREADS")
				.ok()
				.and_then(|value| value.parse::<usize>().ok())
				.filter(|threads| *threads > 0)
				.unwrap_or(DEFAULT_THREADS);

			rayon::ThreadPoolBuilder::new()
				.num_threads(threads)
				.thread_name(|index| format!("mig-{index}"))
				.build()
				.ok()
		})
		.as_ref()
	}

	/// Applies `apply` to every entry of `batch`, on the pool when the batch
	/// is large enough to be worth the fan-out and inline otherwise.
	///
	/// Returns how many entries `apply` reported as *completed* (`true`).
	/// Tallied by the iterator rather than by a shared counter the closure
	/// bumps: the serial path adds up a plain local and rayon's `count`
	/// reduces per-thread partials, so counting costs no atomic per entry on
	/// either path -- the caller pays one `fetch_add` for the whole batch.
	pub fn apply_batch<F>(batch: Vec<(HashedKey, Tier)>, apply: F) -> u64
	where
		F: Fn((HashedKey, Tier)) -> bool + Send + Sync,
	{
		let parallel_threshold = threshold();

		if parallel_threshold == 0 || batch.len() < parallel_threshold {
			return batch.into_iter().filter(|entry| apply(*entry)).count() as u64;
		}

		let Some(pool) = pool() else {
			return batch.into_iter().filter(|entry| apply(*entry)).count() as u64;
		};

		use rayon::prelude::*;
		pool.install(|| batch.into_par_iter().filter(|entry| apply(*entry)).count() as u64)
	}
}

/// TEMPORARY DIAGNOSTIC: batch-size histograms for tier migrations and
/// evictions, to decide whether parallelising the migration copies is
/// worthwhile on a given trace. Buckets are log2: [0]=0, [1]=1, [2]=2-3,
/// [3]=4-7 ... [15]=16384+. Dumped to stderr periodically.
pub mod migstats {
	use std::sync::atomic::{AtomicU64, Ordering};
	use std::sync::OnceLock;
	use std::time::Instant;
	const NB: usize = 16;
	pub static DEMO: [AtomicU64; NB] = [const { AtomicU64::new(0) }; NB];
	pub static PROMO: [AtomicU64; NB] = [const { AtomicU64::new(0) }; NB];
	pub static EVICT: [AtomicU64; NB] = [const { AtomicU64::new(0) }; NB];
	pub static DEMO_TOT: AtomicU64 = AtomicU64::new(0);
	pub static PROMO_TOT: AtomicU64 = AtomicU64::new(0);
	pub static EVICT_TOT: AtomicU64 = AtomicU64::new(0);

	/// Drained migration entries `split_tier_migrations` dropped because a
	/// later entry in the SAME drain named the same key for the OTHER tier --
	/// the intents that never reach `DEMO`/`PROMO` above. So a drain's entries
	/// are `DEMO_TOT + PROMO_TOT + COALESCED_TOT`, and a promote-then-demote
	/// pair for one key shows up here as 1 rather than as a promote copy
	/// undone by a demote copy. A same-tier duplicate is not dropped: it is
	/// counted in `DEMO`/`PROMO` like any other entry, and declines.
	pub static COALESCED_TOT: AtomicU64 = AtomicU64::new(0);
	pub static CALLS: AtomicU64 = AtomicU64::new(0);
	static START: OnceLock<Instant> = OnceLock::new();
	static LAST_DUMP_MS: AtomicU64 = AtomicU64::new(0);
	pub static ECALLS: AtomicU64 = AtomicU64::new(0);
	fn bucket(n: usize) -> usize {
		if n == 0 { return 0; }
		let b = (usize::BITS - n.leading_zeros()) as usize;
		if b >= NB { NB - 1 } else { b }
	}
	pub fn rec(h: &[AtomicU64; NB], tot: &AtomicU64, n: usize) {
		h[bucket(n)].fetch_add(1, Ordering::Relaxed);
		tot.fetch_add(n as u64, Ordering::Relaxed);
	}
	/// Wall-clock interval between periodic dumps.
	const DUMP_INTERVAL_MS: u64 = 10_000;

	/// Reading the clock on every call would cost more than the
	/// instrumentation measures -- a full cluster12 lfu run makes ~445M
	/// `tick` calls -- so the clock is consulted once per this many calls.
	/// Cheap enough to stay in the hot path, frequent enough that a variant
	/// making relatively few migration calls still dumps regularly.
	const CLOCK_CHECK_MASK: u64 = 0xFFF;

	fn maybe_dump(counter_value: u64) {
		if counter_value & CLOCK_CHECK_MASK != 0 {
			return;
		}

		let start = START.get_or_init(Instant::now);
		let now_ms = start.elapsed().as_millis() as u64;
		let last = LAST_DUMP_MS.load(Ordering::Relaxed);

		if now_ms.saturating_sub(last) < DUMP_INTERVAL_MS {
			return;
		}

		// Whichever thread wins the swap does the dump; the others skip it
		// rather than interleaving four `eprintln!`s into the same stderr.
		if LAST_DUMP_MS
			.compare_exchange(last, now_ms, Ordering::Relaxed, Ordering::Relaxed)
			.is_ok()
		{
			dump();
		}
	}

	/// Emits the final totals.
	///
	/// The periodic path above is for progress, not totals: it was previously
	/// keyed on `CALLS % 5_000_000`, which meant a variant making fewer than
	/// 5M migration calls dumped exactly once -- at call #0, before any work
	/// had happened -- and its "totals" were a snapshot of an empty run. That
	/// produced a reported `queue_depth_max=0` and `demo_tot=94,519` for a
	/// 530M-record lru run, both meaningless. Called on worker shutdown so
	/// every run ends with real numbers regardless of its call volume.
	pub fn dump_final() {
		dump();
	}

	pub fn tick() {
		maybe_dump(CALLS.fetch_add(1, Ordering::Relaxed));
	}

	pub fn etick() {
		maybe_dump(ECALLS.fetch_add(1, Ordering::Relaxed));
	}
	pub fn dump() {
		let f = |h: &[AtomicU64; NB]| (0..NB)
			.map(|i| h[i].load(Ordering::Relaxed).to_string())
			.collect::<Vec<_>>().join(",");
		#[cfg(feature = "hybrid_cache_common")]
		eprintln!(
			"MIGSTATS queue_depth_max={} burst_max={} pending_demote_max={} pending_promote_max={}",
			super::migration_queue::DEPTH_MAX.load(Ordering::Relaxed),
			super::migration_queue::BURST_MAX.load(Ordering::Relaxed),
			super::migration_queue::PENDING_DEMOTE_MAX.load(Ordering::Relaxed),
			super::migration_queue::PENDING_PROMOTE_MAX.load(Ordering::Relaxed),
		);

		#[cfg(feature = "hybrid_cache_common")]
		eprintln!(
			"MIGSTATS pending_net_max={}",
			super::migration_queue::PENDING_NET_MAX.load(Ordering::Relaxed),
		);

		#[cfg(feature = "hybrid_cache_common")]
		eprintln!(
			"MIGSTATS applied={} gone={} declined={} superseded={}",
			super::migration_queue::MIG_APPLIED.load(Ordering::Relaxed),
			super::migration_queue::MIG_GONE.load(Ordering::Relaxed),
			super::migration_queue::MIG_DECLINED.load(Ordering::Relaxed),
			super::migration_queue::MIG_SUPERSEDED.load(Ordering::Relaxed),
		);

		// `coalesced_tot` goes LAST: scripts read this line as `key=value`
		// pairs, and appending keeps every existing field where it was.
		eprintln!("MIGSTATS mig_calls={} evict_calls={} demo_tot={} promo_tot={} evict_tot={} coalesced_tot={}",
			CALLS.load(Ordering::Relaxed), ECALLS.load(Ordering::Relaxed),
			DEMO_TOT.load(Ordering::Relaxed), PROMO_TOT.load(Ordering::Relaxed),
			EVICT_TOT.load(Ordering::Relaxed), COALESCED_TOT.load(Ordering::Relaxed));
		eprintln!("MIGSTATS demo={}", f(&DEMO));
		eprintln!("MIGSTATS promo={}", f(&PROMO));
		eprintln!("MIGSTATS evict={}", f(&EVICT));
	}
}

/// Capacity-eviction watermarks, for `apply_evictions`' `over_max_size` loop.
///
/// That loop drains to exactly `max_size` one object at a time, so a cache
/// sitting at capacity re-enters the whole eviction machinery on *every*
/// subsequent set to free a single object -- the same batch-of-one shape a
/// high/low band once batched for fast-tier demotions (that band is gone; the
/// hybrid stacks now hold one continuous threshold, `policy_stack::drain_target`,
/// at 0.98 of their budget). With watermarks a pass
/// arms only once usage crosses `high * max_size`, and then drains to
/// `low * max_size` in one go.
///
/// THE DEFAULTS ARE 1.0/1.0 AND MUST STAY THAT WAY. Both thresholds are then
/// `max_size` exactly and the loop is the pre-watermark
/// `while used_size > max_size`, object for object. Every published sweep
/// (`results/policy_sweep_110_cells.md` and the others beside it) was measured
/// against a cache that settles exactly at the cap; a default that evicted
/// deeper would leave those numbers parsing perfectly and describing code that
/// no longer exists. The feature is strictly opt-in, via
/// `EVICTION_HIGH_WATERMARK` / `EVICTION_LOW_WATERMARK`.
///
/// Deliberately its own pair, and not shared with the hybrid stacks'
/// fast-tier settle. That settle is a SINGLE threshold -- `drain_target`, 0.98
/// of the effective fast-tier budget -- armed and drained to the same number,
/// so the tier steady-states just under its budget with 2% of burst headroom.
/// It is not a band, and it says nothing about how far past `max_size` the
/// cache as a whole may be trimmed.
pub mod eviction_watermarks {
	use std::sync::OnceLock;

	use crate::CacheSize;

	pub const DEFAULT_HIGH: f64 = 1.0;
	pub const DEFAULT_LOW: f64 = 1.0;

	static HIGH: OnceLock<f64> = OnceLock::new();
	static LOW: OnceLock<f64> = OnceLock::new();

	fn read(var: &str, default: f64) -> f64 {
		std::env::var(var)
			.ok()
			.and_then(|v| v.parse::<f64>().ok())
			.filter(|v| *v > 0.0 && *v <= 1.0)
			.unwrap_or(default)
	}

	/// Fraction of `max_size` above which a capacity-eviction pass arms.
	/// `1.0` (the default) arms at the cap, exactly as before this existed.
	pub fn high() -> f64 {
		*HIGH.get_or_init(|| read("EVICTION_HIGH_WATERMARK", DEFAULT_HIGH))
	}

	/// Fraction of `max_size` an armed pass drains down to. Clamped to at
	/// most `high()` -- see `clamped_low`.
	pub fn low() -> f64 {
		*LOW.get_or_init(|| clamped_low(high(), read("EVICTION_LOW_WATERMARK", DEFAULT_LOW)))
	}

	/// Clamps the drain target to the trigger point so a misconfigured pair
	/// cannot invert. Inverted, a pass would arm at the high mark and
	/// immediately find itself already under its own drain target: it would
	/// evict nothing at all and the cache would sit permanently above
	/// `high * max_size` with no way back down.
	///
	/// Factored out of `low()` rather than written inline the way the
	/// fast-tier pair does it because `OnceLock` memoises the env read for the
	/// life of the process, which leaves the clamp itself unreachable from a
	/// test that does not want to configure every other test in the binary.
	pub(crate) fn clamped_low(high: f64, low: f64) -> f64 {
		if low > high { high } else { low }
	}

	/// Turns a watermark into a byte threshold on `max_size`.
	///
	/// `>= 1.0` returns `max_size` untouched instead of round-tripping it
	/// through `f64`, and that short-circuit is the whole 1.0-default
	/// guarantee: an unconfigured build must perform *the same comparison* it
	/// performed before this module existed, not one that agrees with it up to
	/// a rounding step. `u64 -> f64` is lossy above 2^53 bytes, and the `as`
	/// truncation could otherwise land a threshold a byte under the cap --
	/// either of which is a silent change in eviction depth.
	pub(crate) fn scaled(max_size: CacheSize, watermark: f64) -> CacheSize {
		if watermark >= 1.0 {
			return max_size;
		}

		(max_size as f64 * watermark) as CacheSize
	}

	/// A `(high, low)` pair, snapshotted per `PolicyWorker`.
	///
	/// Snapshotted rather than read from the statics at each call so a test
	/// can drive a non-default pair: `OnceLock` memoises the first env read
	/// for the whole test binary, so a test that set the vars itself would
	/// silently configure -- or be configured by -- every other test in the
	/// process, depending on which one ran first.
	#[derive(Clone, Copy, Debug, PartialEq)]
	pub struct Watermarks {
		high: f64,
		low: f64,
	}

	impl Watermarks {
		/// Builds a pair with `low` clamped to at most `high`.
		pub fn new(high: f64, low: f64) -> Self {
			Watermarks {
				high,
				low: clamped_low(high, low),
			}
		}

		/// The process-wide pair, from the environment or the 1.0 defaults.
		/// Goes through `new` so the configured path and the test path cannot
		/// drift apart on the clamp (`low()` has already applied it, and the
		/// clamp is idempotent).
		pub fn from_env() -> Self {
			Watermarks::new(high(), low())
		}

		/// `(trigger, drain target)` in bytes. Both are exactly `max_size` at
		/// the defaults.
		pub fn bytes(&self, max_size: CacheSize) -> (CacheSize, CacheSize) {
			(scaled(max_size, self.high), scaled(max_size, self.low))
		}
	}
}

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

	/// Capacity-eviction watermarks for `apply_evictions`, snapshotted at
	/// construction (see `eviction_watermarks::Watermarks`). `1.0`/`1.0`
	/// unless `EVICTION_HIGH_WATERMARK`/`EVICTION_LOW_WATERMARK` say
	/// otherwise, which is the pre-watermark drain-to-exactly-`max_size`
	/// loop.
	eviction_watermarks: eviction_watermarks::Watermarks,

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
	/// `TieredBuffer::new_fast`/`new_slow`). Used by the hybrid designs
	/// (`lru_compact_hybrid_cache`, `lfu_compact_hybrid_cache`, ...) to
	/// physically move an object's bytes when their stack reports a tier
	/// migration; `None` for every other policy/value type. Promotion,
	/// demotion and eviction counters and gauges are recorded directly on the
	/// shared `status` (see `apply_tier_migrations`), not a separate field.

	/// Whether this worker physically migrates object bytes between tiers.
	///
	/// Was `tier_migration_fn: Option<Arc<dyn Fn(..) -> Option<TieredValue>>>` --
	/// a boxed per-shape constructor for the destination buffer, cloned into
	/// the migration queue so both paths built values the same way. A value can
	/// now copy itself into another tier (`TieredValue::migrated_to`), so all
	/// that survives of it is the question it also answered: is this a hybrid
	/// build, or a policy with no tiers to move anything between?
	#[cfg(feature = "hybrid_cache_common")]
	tier_migration: bool,

	/// Standing pool draining physical tier copies off the worker thread.
	/// `None` unless `MIGRATION_QUEUE_THREADS` is non-zero -- see
	/// [`migration_queue`].
	#[cfg(feature = "hybrid_cache_common")]
	migration_queue: Option<migration_queue::MigrationQueue>,
}

impl<K, V> Worker for PolicyWorker<K, V>
where
	Self: 'static + Send,
	K: Eq + Clone + TypeSize + Send + Sync,
	V: Send + Sync,
{
	fn run(&mut self) -> Result<(), CacheError> {
		// Published before the first pass, so a kick can find this thread
		// from the moment it could be parked -- see
		// `AtomicStatus::kick_policy_worker`. Overwrites, rather than keeping
		// the first, so a status ever handed to a second worker would wake
		// the live one and not a thread that has exited.
		self.status.set_policy_worker_thread(thread::current());

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

					WorkerEvent::Set(key, size, resident, _, _) => {
						self.handle_set(key, size, resident);
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

						// Real totals, whatever this run's call volume was.
						migstats::dump_final();

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
				//
				// STAYS PER-EVENT, for the latency reason above. It used to
				// have a second one: `apply_migration_batches` PARTITIONS its
				// batch into demotions and promotions and applies all of the
				// first before any of the second, and per-key emission order
				// did not survive that -- a key promoted and then demoted in
				// one drain arrived demote-then-promote and ended up
				// physically in DRAM while the policy stack recorded it as
				// slow, which does not self-heal (`MigrationQueue`'s doc: the
				// stack already believes the newer tier, so it never
				// re-emits). Draining per event never actually closed that
				// hole: ONE event can emit both entries for one key (the
				// merged store's touch queues its promotion before the settle
				// that may demote the key again), and the drain after
				// `apply_evictions` below collects a whole eviction loop's
				// decisions, the merged CLOCK hand's second chances among
				// them. `split_tier_migrations` now closes it for a drain of
				// any width: the partition drops every entry that a later
				// entry for the other tier supersedes, so no key reaches both
				// halves.
				//
				// The reason it was expensive is gone regardless:
				// `MergedStore::drain_migrations` used to take a WRITE lock on
				// all 32 shards whenever anything anywhere was pending. It now
				// consults a per-shard dirty bitmask and locks only the shards
				// that actually migrated -- one relaxed load when nothing has.
				#[cfg(feature = "hybrid_cache_common")]
				self.apply_tier_migrations();
			}

			self.apply_buffered_events(&buffered_events, &policy_reconstruct_rx);
			self.flush_buffered_events(&mut buffered_events)?;
			self.apply_evictions(&mut buffered_events)?;

			// INSTRUMENTATION: the invariant the failing run violated. Sampled on
			// the same cadence as MIGSTATS so the two can be correlated.
			{
				use std::sync::atomic::Ordering;
				static TICK: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
				if TICK.fetch_add(1, Ordering::Relaxed) % 4_096 == 0 {
					let stack_len = self.policy_stack.as_ref().map_or(0, |s| s.len());
					let map_len = self.status.live_num_objects() as usize;
					let (fo, so, fb, sb) = self.policy_stack.as_ref()
						.map_or((0, 0, 0, 0), |s| (
							s.fast_object_count(), s.slow_object_count(),
							s.fast_bytes_used(), s.slow_bytes_used(),
						));
					eprintln!(
						"DIVERGE map={map_len} stack={stack_len} delta={} fallback={} fast_obj={fo} slow_obj={so} fast_b={fb} slow_b={sb}",
						map_len as i64 - stack_len as i64,
						crate::ERASE_FALLBACK.load(Ordering::Relaxed),
					);
				}
			}

			// `apply_evictions` runs every outer-loop iteration regardless
			// of whether `events` was non-empty (unlike the per-event call
			// above, gated on there being an event to process at all) --
			// this matters because `S3FifoCompactHybridStack::evict_one` can
			// push a real (key, Tier::Fast) migration as a side effect of its
			// eviction sweep (`give_second_chance`, the CLOCK-style "reused
			// second chance" mechanic: a key found with its reference bit
			// set gets promoted instead of evicted). Every other hybrid
			// stack's `evict_one` only ever pops-and-removes for a real
			// eviction -- it never touches `self.migrations` -- so this call
			// was previously safe to omit here; it's a correctness
			// requirement now that at least one stack's eviction sweep can
			// produce a promotion. Cheap early-return when there's nothing
			// to migrate, same as the per-event call above.
			#[cfg(feature = "hybrid_cache_common")]
			self.apply_tier_migrations();

			// Once per pass, after every migration and eviction this pass
			// could produce has already been applied -- so what gets published
			// here is exactly as current as a per-event refresh would have
			// left it, without putting those stores on the per-read path. See
			// `refresh_tier_gauges`.
			#[cfg(feature = "hybrid_cache_common")]
			self.refresh_tier_gauges();

			// Once per pass: push this thread's retired values into the global
			// garbage queue and try to advance the epoch.
			//
			// This worker is where most values die -- every eviction and every
			// superseded migration retires one -- and a deferral sits in the
			// LOCAL bag of the thread that made it until that thread pins
			// enough more times to fill the bag. Between two bursts of
			// evictions this thread sleeps in `delay_event_loop`, so without
			// this the cache's real footprint would stay a whole burst above
			// what it reports, for as long as the lull lasts. Cheap when there
			// is nothing to flush.

			let now = Instant::now();

			if let Some(policy) = self.perform_auto_policy(now, has_current_set) {
				self.status.set_auto_policy(policy)?;
				self.handle_policy(policy, policy_reconstruct_tx.clone());
			}

			// The pass is complete: what a test waits on to see that a kick
			// reached this thread.
			#[cfg(test)]
			self.status.record_policy_worker_pass();

			self.delay_event_loop(now, has_current_set);
		}
	}
}

impl<K, V> PolicyWorker<K, V>
where
	// `Send + Sync` are required by `parallel_migration::apply_batch`, which
	// fans a large migration batch out across a pool; they hold already for
	// every real instantiation, since the worker owns the object map on its
	// own thread. `Clone` is required because a tier migration now rebuilds the
	// value header around the key it copies from the old one -- see
	// `TieredValue::migrated_to`.
	K: 'static + Eq + Clone + TypeSize + Send + Sync,
	V: 'static + Send + Sync,
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
		#[cfg(not(feature = "merged_object_store"))]
		let policy_stack = init_policy_stack(policy, max_cache_size);

		// The merged store is both the object map and the eviction stack, so
		// the stack is built over the SAME `Arc` rather than allocated
		// alongside it. `init_policy_stack` is bypassed entirely: there is no
		// second structure for it to construct.
		// Fallible, and the `?` is the whole of the ripple: a merged build asked
		// for a policy whose order the store does not implement now fails to
		// CONSTRUCT rather than quietly running a different order. Both
		// enclosing constructors already returned `Result<Self, CacheError>`,
		// and every caller above them -- `WorkerFanout::new`,
		// `PaperCache::new` / `new_sized_compact` -- already propagates, so the
		// error surfaces where the cache is built and never at first use.
		#[cfg(feature = "merged_object_store")]
		let policy_stack: Box<dyn PolicyStack> = Box::new(
			policy_stack::merged_stack::MergedStackHandle::new(
				objects.clone(),
				policy,
				max_cache_size,
			)?,
		);

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
			eviction_watermarks: eviction_watermarks::Watermarks::from_env(),

			trace_fragments,
			trace_worker,
			trace_handle,

			mini_stack_manager: mini_stacks,
			mini_index: None,

			current_policy: Arc::new(RwLock::new(policy)),

			last_auto_policy_time: None,
			last_set_time: None,

			promotion_tx,

			#[cfg(feature = "hybrid_cache_common")]
			#[cfg(feature = "hybrid_cache_common")]
			tier_migration: false,

			#[cfg(feature = "hybrid_cache_common")]
			migration_queue: None,
		};

		Ok(worker)
	}

	/// Constructs a `PolicyWorker` that physically migrates object bytes
	/// between tiers whenever `PaperPolicy::LruCompactHybrid`'s
	/// `LruCompactHybridStack`, `PaperPolicy::LfuCompactHybrid`'s
	/// `LfuCompactHybridStack`, `PaperPolicy::TwoQCompactHybrid`'s
	/// `TwoQCompactHybridStack`, `PaperPolicy::FifoCompactHybrid`'s
	/// `FifoCompactHybridStack` -- or any other hybrid design's stack --
	/// reports a promotion or demotion (see `apply_tier_migrations`).
	///
	/// `migrate` reallocates a value into the representation for the given
	/// `Tier` (e.g. `TieredBuffer::new_fast`/`new_slow`). Promotion/demotion/
	/// eviction counters and the current tier gauges are recorded directly
	/// on `status` (see `apply_tier_migrations`), which is why this
	/// constructor needs no separate stats parameter.
	#[cfg(feature = "hybrid_cache_common")]
	pub fn new_with_tier_migration(
		listener: WorkerReceiver,
		objects: ObjectMapRef<K, V>,
		status: StatusRef,
		overhead_manager: OverheadManagerRef,
	) -> Result<Self, CacheError> {
		let max_cache_size = status.max_size();

		// Hybrid caches (the only callers of this constructor) are always
		// constructed with a single-element policies list containing only
		// their own fixed policy (e.g. `[PaperPolicy::LruCompactHybrid]`)
		// and never with `PaperPolicy::Auto` -- there is no public API to
		// switch a hybrid cache's policy at all
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
		#[cfg(not(feature = "merged_object_store"))]
		let policy_stack = init_policy_stack(policy, max_cache_size);

		// The merged store is both the object map and the eviction stack, so
		// the stack is built over the SAME `Arc` rather than allocated
		// alongside it. `init_policy_stack` is bypassed entirely: there is no
		// second structure for it to construct.
		// Fallible, and the `?` is the whole of the ripple: a merged build asked
		// for a policy whose order the store does not implement now fails to
		// CONSTRUCT rather than quietly running a different order. Both
		// enclosing constructors already returned `Result<Self, CacheError>`,
		// and every caller above them -- `WorkerFanout::new`,
		// `PaperCache::new` / `new_sized_compact` -- already propagates, so the
		// error surfaces where the cache is built and never at first use.
		#[cfg(feature = "merged_object_store")]
		let policy_stack: Box<dyn PolicyStack> = Box::new(
			policy_stack::merged_stack::MergedStackHandle::new(
				objects.clone(),
				policy,
				max_cache_size,
			)?,
		);

		// A hybrid cache is always constructed with a single fixed policy and
		// exposes no way to switch it, so `trace_is_useful` is always false
		// here -- no `TraceWorker` thread, and no per-access trace writes.
		let trace_fragments = Arc::new(RwLock::new(VecDeque::new()));
		let (trace_worker, trace_handle) = spawn_trace_worker(
			trace_is_useful(&status),
			&trace_fragments,
			status.max_size(),
		)?;

		#[cfg(feature = "hybrid_cache_common")]
		let migration_queue = migration_queue::MigrationQueue::spawn(
			objects.clone(),
			migration_queue::threads(),
			status.clone(),
		);

		let worker = PolicyWorker {
			listener,

			objects,
			status,
			overhead_manager,

			policy_stack: Some(policy_stack),
			eviction_watermarks: eviction_watermarks::Watermarks::from_env(),

			trace_fragments,
			trace_worker,
			trace_handle,

			mini_stack_manager: mini_stacks,
			mini_index: None,

			current_policy: Arc::new(RwLock::new(policy)),

			last_auto_policy_time: None,
			last_set_time: None,

			// None of `LruCompactHybridStack`, `LfuCompactHybridStack`, or
			// `TwoQCompactHybridStack` ever emits
			// `AccessOutcome::GhostHit`, so no ghost-hit-driven promotion
			// channel is needed for any of them.
			promotion_tx: None,

			tier_migration: true,

			#[cfg(feature = "hybrid_cache_common")]
			migration_queue,
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

	/// `dram_resident` is the part of `size` that never migrates; the policy
	/// stack needs it to keep `fast_used` / `slow_used` to migrating bytes.
	/// The mini stacks model policy behaviour, not tier occupancy, so they
	/// keep taking the full `base_size`.
	fn handle_set(&mut self, key: HashedKey, size: ObjectSize, dram_resident: ObjectSize) {
		if let Some(stack) = &mut self.policy_stack {
			stack.insert_resident(key, size, dram_resident);
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
			// Live: re-set between the reap and this notification, or live
			// all along and left in place by the reap (`EraseKey::Expired`).
			// Either way the stack entry belongs to that live object.
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

	/// Runtime-adjusts the fast-tier byte budget. Honoured by every hybrid
	/// design's stack; a no-op for every non-hybrid one. May itself trigger
	/// demotions, drained by `apply_tier_migrations` on the next pass through
	/// the event loop.
	fn handle_resize_fast_tier(&mut self, size: CacheSize) {
		if let Some(stack) = &mut self.policy_stack {
			stack.resize_fast_tier(size);
		}
	}

	/// Runtime-adjusts the LARGE fast segment's byte budget
	/// (`lru_sized_compact_hybrid_cache` specifically). No-op for every other
	/// policy stack. May itself trigger demotions, drained by
	/// `apply_tier_migrations` on the next pass through the event loop.
	fn handle_resize_large_fast_tier(&mut self, size: CacheSize) {
		if let Some(stack) = &mut self.policy_stack {
			stack.resize_large_fast_tier(size);
		}
	}

	/// Runtime-adjusts the small/large size-classification threshold
	/// (`lru_sized_compact_hybrid_cache`). No-op for every other policy stack.
	fn handle_resize_size_threshold(&mut self, size: CacheSize) {
		if let Some(stack) = &mut self.policy_stack {
			stack.resize_size_threshold(size);
		}
	}

	#[cfg_attr(feature = "merged_object_store", allow(unreachable_code))]
	fn handle_policy(
		&mut self,
		policy: PaperPolicy,
		policy_reconstruct_tx: Arc<Sender<Box<dyn PolicyStack>>>,
	) {
		#[cfg(feature = "merged_object_store")]
		{
			// Reconstruction replays a trace into a NEW stack and swaps it in.
			// There is nothing to swap here: the stack is a handle on the object
			// map, and a second one would be a second view of the same data, not
			// a rebuilt structure. Unreachable in practice anyway -- a merged
			// build runs one fixed policy -- but silent divergence is exactly
			// what this design exists to make impossible, so it is refused
			// rather than left to the trace-worker guard below.
			let _ = policy_reconstruct_tx;
			warn!("Ignoring switch to {policy}: the merged store has one policy");
			return;
		}
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

	/// Applies every tier migration the policy stack has accumulated.
	///
	/// One body for all hybrid designs; the two per-design accounting
	/// differences live on the stack itself. `inline_demotion_accounting`
	/// is `false` only for the LFU-style design, whose `Tier::Slow` entries
	/// are not always genuine demotions -- its true count arrives through
	/// `drain_demotions` (zero for every other stack). The FIFO design
	/// never emits `Tier::Fast`, so its promotion counter stays 0 through
	/// the same per-entry path rather than by special case.
	#[cfg(feature = "hybrid_cache_common")]
	fn apply_tier_migrations(&mut self) {
		let (inline_demotion_accounting, migrations) = {
			let Some(stack) = &mut self.policy_stack else { return };

			(stack.inline_demotion_accounting(), stack.drain_tier_migrations())
		};

		// Handed over whole: `apply_migration_batches` does the split itself
		// (`split_tier_migrations`), and taking the drain unsplit is what
		// makes that the only way in.
		if !migrations.is_empty() {
			self.apply_migration_batches(migrations, inline_demotion_accounting);
		}

		let drained_demotions = match &mut self.policy_stack {
			Some(stack) => stack.drain_demotions(),
			None => 0,
		};

		if drained_demotions > 0 {
			self.status.record_hybrid_demotions(drained_demotions);
		}
	}

	/// Applies one pass's worth of drained migrations and records the ones
	/// that physically completed.
	///
	/// A migration is counted if and only if `Object::set_data` ran. The
	/// counters used to fire once per entry handed to `apply_physical`, which
	/// with the standing queue on (the default) meant they counted *enqueued
	/// intents*: the entry had not moved a byte yet, and had three remaining
	/// ways never to (see `migration_queue::apply_migration`). Measured
	/// overstatement on one benchmark run was 4.6x. So the queued path leaves
	/// the counting to the consumer that performs the swap, and the
	/// synchronous path takes the completion count back from `apply_batch`.
	///
	/// Split out of [`Self::apply_tier_migrations`] so this accounting can be
	/// unit-tested against hand-built batches, instead of having to coax a
	/// real policy stack into emitting each of the four outcomes.
	///
	/// Takes one drain UNSPLIT and splits it itself, with
	/// [`split_tier_migrations`]. It used to take the two halves already
	/// partitioned, which left the ordering hazard with every caller: a key
	/// promoted then demoted in one drain was applied demote-first -- the
	/// demote declined against a value that was still slow, the promote then
	/// copied it into DRAM -- and the stack went on counting it slow with
	/// nothing left to move it back. The split drops every entry that a later
	/// entry for the other tier supersedes, so no key is in both halves and
	/// the order between them cannot matter to any key; demotions still go
	/// first so DRAM is freed before it is claimed. Every caller, the tests
	/// included, goes through it, because there is no other way in.
	#[cfg(feature = "hybrid_cache_common")]
	fn apply_migration_batches(
		&self,
		migrations: Vec<(HashedKey, Tier)>,
		inline_demotion_accounting: bool,
	) {
		if !self.tier_migration {
			return;
		}

		let (demotions, promotions, coalesced) = split_tier_migrations(migrations);

		if coalesced > 0 {
			migstats::COALESCED_TOT.fetch_add(coalesced as u64, std::sync::atomic::Ordering::Relaxed);
		}

		migration_queue::BURST_MAX.fetch_max(
			(demotions.len() + promotions.len()) as u64,
			std::sync::atomic::Ordering::Relaxed,
		);

		migstats::rec(&migstats::DEMO, &migstats::DEMO_TOT, demotions.len());
		migstats::rec(&migstats::PROMO, &migstats::PROMO_TOT, promotions.len());
		migstats::tick();

		let objects = &self.objects;
		let status = &self.status;
		let migration_queue = self.migration_queue.as_ref();

		// The consumers cannot see the policy stack, so mirror the one bit of
		// it they need before handing them anything to count.
		if let Some(queue) = migration_queue {
			queue.set_demotion_accounting(inline_demotion_accounting);
		}

		// Build the destination buffer with NO object-map guard held -- see
		// the pre-unification history for the full latency reasoning;
		// `Object::data()` is an `Arc` refcount bump and keeps the source
		// bytes alive unlocked.
		let apply_physical = |(key, tier): (HashedKey, Tier)| -> bool {
			if let Some(queue) = migration_queue {
				queue.push((key, tier));

				// Handed off, not applied: nothing has moved yet, and the
				// consumer that eventually moves it does the counting.
				return false;
			}

			migration_queue::apply_migration(objects, key, tier)
		};

		let completed_demotions = parallel_migration::apply_batch(
			demotions,
			|entry| apply_physical(entry),
		);

		let completed_promotions = parallel_migration::apply_batch(
			promotions,
			|entry| apply_physical(entry),
		);

		// At most one atomic per direction per pass (and none at all when the
		// count is zero, which with the queue on is every pass), where the old
		// code paid one per entry regardless of outcome.
		if inline_demotion_accounting {
			status.record_hybrid_demotions(completed_demotions);
		}

		status.record_hybrid_promotions(completed_promotions);

		// Tests assert on tier residency and on the counters right after
		// triggering a migration; the standing consumer pool applies
		// asynchronously, so drain it before returning.
		#[cfg(test)]
		if let Some(queue) = migration_queue {
			queue.flush();
		}
	}

	/// Mirrors the stack's tier gauges (and the LFU admission latch) onto
	/// the shared status.
	///
	/// Refreshed unconditionally, not gated on "a migration just happened":
	/// gating let the gauges go permanently stale after insert bursts that
	/// ended without one further demotion. The four-segment gauges are read
	/// through trait methods that default to zero, so only the size-split
	/// design ever populates them; likewise `admission_latched` is `false`
	/// for every stack but the LFU-style one.
	#[cfg(feature = "hybrid_cache_common")]
	fn refresh_tier_gauges(&mut self) {
		if let Some(stack) = &self.policy_stack {
			self.status.set_hybrid_admission_latched(stack.admission_latched());

			self.status.set_hybrid_gauges(
				stack.fast_bytes_used(),
				stack.slow_bytes_used(),
				stack.fast_object_count() as u64,
				stack.slow_object_count() as u64,
				stack.dram_reserved_bytes(),
			);

			self.status.set_hybrid_sized_gauges(
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

	fn apply_buffered_events(
		&mut self,
		buffered_events: &[StackEvent],
		policy_reconstruct_rx: &Receiver<Box<dyn PolicyStack>>,
	) {
		for mut stack in policy_reconstruct_rx.try_iter() {
			for event in buffered_events {
				match event {
					StackEvent::Get(key) => stack.update(*key),
					StackEvent::Set(key, size, resident) =>
						stack.insert_resident(*key, *size, *resident),
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

		// `trigger_size` arms a capacity pass; `drain_target` is how far that
		// pass then goes. Both are `max_cache_size` at the 1.0/1.0 default,
		// which is what keeps an unconfigured build on the exact loop that
		// predates the eviction watermarks.
		let (trigger_size, drain_target) = self.eviction_watermarks.bytes(max_cache_size);

		let mut _evicted_this_call: usize = 0;

		// Hysteresis latch. Without it the pass would re-check its own trigger
		// every iteration and stop the moment usage fell a byte back under the
		// high mark -- which is not a batch at all, it is the old
		// evict-exactly-one behaviour wearing a threshold, and it would
		// oscillate across the high mark once per set.
		//
		// Armed by the capacity condition alone, never by
		// `needs_capacity_eviction`: a stack draining its own internal
		// sub-budget (`TwoQCompactHybridStack`'s `k_in`-derived fifo budget)
		// must not drag the whole cache down to the low mark as a side
		// effect.
		let mut draining = false;

		loop {
			let used_size = self.status.used_size(&policy);

			let over_max_size = match draining {
				false => used_size > trigger_size,
				true => used_size > drain_target,
			};

			draining |= over_max_size;

			// `len() > 0` guards against ever looping forever on a stack
			// whose `needs_capacity_eviction` stays true despite having
			// nothing left to evict (which would indicate an accounting
			// bug in the stack, not a real pending eviction).
			let needs_capacity_eviction = self.policy_stack.as_ref()
				.is_some_and(|stack| stack.len() > 0 && stack.needs_capacity_eviction());

			if !over_max_size && !needs_capacity_eviction {
				migstats::rec(&migstats::EVICT, &migstats::EVICT_TOT, _evicted_this_call);
				migstats::etick();
				break;
			}

			let Some(policy_stack) = self.policy_stack.as_mut() else {
				error!("No active policy or mini stack");
				return Err(CacheError::Internal);
			};

			let maybe_key = policy_stack
				.evict_one()
				.map(|key| EraseKey::Hashed(key));

			// A split design can legitimately have an empty stack over a
			// non-empty map -- that divergence is what `erase`'s `None`
			// fallback exists to clean up, by evicting an arbitrary map entry.
			// The merged store cannot diverge, so `None` here means the store
			// is genuinely empty and `erase` can only fail. Without this the
			// loop `continue`s on unchanged state forever.
			#[cfg(feature = "merged_object_store")]
			if maybe_key.is_none() {
				error!("Nothing left to evict with used_size still over max");
				break;
			}

			let erase_result = erase(
				&self.objects,
				&self.status,
				&self.overhead_manager,
				maybe_key,
			);

			let Ok((key, _evicted_obj)) = erase_result else {
				continue;
			};
			_evicted_this_call += 1;

			#[cfg(feature = "hybrid_cache_common")]
			if policy.is_hybrid() {
				self.status.record_hybrid_eviction();
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
	/// The wait is unconditional, including when the poll just processed a
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
	///
	/// A PARK, not a sleep: `thread::park_timeout` with nobody unparking
	/// returns at the same timeout `thread::sleep` did, so the measured
	/// behaviour above is unchanged, but the thread can now be woken early by
	/// `AtomicStatus::kick_policy_worker`. Nothing in production kicks yet.
	/// The two ways a park can return early both cost one extra pass and
	/// nothing else: a spurious wakeup, and a stale unpark token -- a kick
	/// that landed while this thread was mid-pass is kept by the thread and
	/// consumed by the next park, which returns at once. Neither can lose
	/// work: the pass it causes drains whatever is queued, exactly as a timed
	/// wakeup would.
	///
	/// How long it waits is `polling_delay`'s decision, which counts the sets
	/// THIS pass handled. It used to consult only the sets before it, so the
	/// first pass to see a burst after SET_RECENCY_DURATION of quiet chose the
	/// long poll again, and whatever the burst sent after that pass -- its
	/// later sets, their demotions, their settles -- waited up to another
	/// second. The wait BEFORE that first pass is not this function's to
	/// shorten: a worker parked on the long poll when a burst begins sleeps
	/// it out unless it is kicked (see `AtomicStatus::kick_policy_worker`).
	fn delay_event_loop(&mut self, now: Instant, has_current_set: bool) {
		let delay = polling_delay(now, self.last_set_time, has_current_set);

		if has_current_set {
			self.last_set_time = Some(now);
		}

		thread::park_timeout(delay);
	}
}

/// How long the policy worker waits before its next pass.
///
/// SHORT while sets are arriving -- this pass handled one, or the last one
/// was within SET_RECENCY_DURATION -- and LONG only once they have stopped.
/// `has_current_set` is the half that was missing: the recency test alone
/// reads `last_set_time` from BEFORE this pass, so the pass that first saw a
/// burst after 5 s of quiet (or the cache's very first sets) parked on the
/// 1 s poll again, and the rest of the burst waited another second. It
/// cannot shorten the wait before that pass, which a worker already parked on
/// the long poll sleeps out -- see `AtomicStatus::kick_policy_worker`.
///
/// Pure, and taking the clock as an argument, so the decision can be tested
/// without a running worker or a real 5 s wait.
fn polling_delay(now: Instant, last_set_time: Option<Instant>, has_current_set: bool) -> Duration {
	let has_recent_set = last_set_time
		.is_some_and(|last_set_time| now - last_set_time <= SET_RECENCY_DURATION);

	match has_recent_set || has_current_set {
		true => SHORT_POLLING_DURATION,
		false => LONG_POLLING_DURATION,
	}
}

/// Splits one drain into the demotions and the promotions
/// `apply_migration_batches` applies -- every demotion, then every
/// promotion -- dropping each entry whose key has a LATER entry for the OTHER
/// tier in the same drain. Returns `(demotions, promotions, dropped)`.
///
/// Each half keeps drain order. The contract, pinned exhaustively by
/// `migration_split_tests` over every drain of up to five entries on two
/// keys:
///
///   * no key is in both halves: every entry kept for a key is in the tier
///     of that key's last entry;
///   * so applying the demotions and then the promotions leaves every key in
///     the tier of its LAST entry in drain order -- where applying the whole
///     drain in order leaves it -- in at most one copy per key;
///   * `dropped` counts only the entries a later entry for the other tier
///     superseded. Same-tier duplicates are kept: the partition cannot
///     reorder them against each other, and the second one declines.
///
/// # Why the partition needs it
///
/// `apply_migration` carries no identity: it acts on whatever object holds
/// the key when it runs, and declines when that object is already in the
/// requested tier. So a key's entries applied in order leave its value in
/// the last entry's tier -- and applied demotions-first they need not.
/// `[(k, Fast), (k, Slow)]` on a slow value -- promoted, then demoted again,
/// net intent Slow -- went out Slow-then-Fast: the demote declined against a
/// value still in CXL, the promote copied it into DRAM, and the stack counted
/// it slow from then on with nothing left to move it. The mirror,
/// `[(k, Slow), (k, Fast)]` on a fast value, ended in the right tier but
/// through a round trip to CXL and back. Only a key with entries in both
/// halves can be reordered by the partition, and of such a key's entries
/// this keeps only those after its last entry for the other tier.
///
/// The overwrite case is the same rule. A stack queues `(k, Slow)` for an OLD
/// object; a `set` replaces it with a new one built in DRAM; the stack's
/// re-promotion queues `(k, Fast)` so that the stale demotion, landing on the
/// new object, is undone (`LruCompactHybridStack::touch_fast_key`, the merged
/// store's `touch_slot`). When both are in one drain the Slow is dropped and
/// the Fast declines against the DRAM-built value -- exactly the intended end
/// state, with neither copy. When they are in different drains the migration
/// queue's per-key FIFO still applies them in order, as before.
///
/// # What it does to the accounting
///
/// Completions are counted where `Object::set_data` runs, so dropping an
/// entry that would have declined changes no counter, and dropping a pair
/// that would have been undone removes its promotion and its demotion
/// together -- copies that ended where they started. The exception is the
/// LFU-style design (`inline_demotion_accounting() == false`): it never
/// counts a completed slow move, and its demotions are tallied by
/// `drain_demotions` when its settle DECIDES them, so a demotion dropped
/// here because a later promotion in the same drain supersedes it stays
/// counted although nothing moved. Lazy copy emits only PHYSICAL intents
/// (its logical demotions queue nothing), and for those the last one is
/// equally the answer. Dropped entries are counted in
/// `migstats::COALESCED_TOT`; `DEMO`/`PROMO`, `BURST_MAX` and the
/// `PENDING_*` gauges see only what is kept.
///
/// # Cost
///
/// A one-sided drain -- all demotions or all promotions, which includes
/// every drain of 0 or 1 entries, the common case (`migration_queue`'s doc
/// measured 99.4% of demotion volume arriving one object at a time) -- has
/// nothing the partition could reorder. It costs one read of each entry's
/// tier and is returned whole as the non-empty half: nothing is allocated,
/// hashed or copied, less than the plain partition this replaced. A mixed
/// drain pays for the two halves, ONE map sized to the SMALLER half (16
/// bytes a bucket: the key and two `u32` positions), an insert and a lookup
/// per entry of the smaller half, and one lookup per entry of the larger
/// half. `HashedKey` is already a hash, so the map's hasher is the crate's
/// pass-through `NoHasher`. Mixed bursts can still be large: the faithful
/// S3-FIFO's main-queue sweep pushes a promotion per requeued slow key and
/// its settle a demotion for each, so its bursts size the map to up to half
/// the drain.
#[cfg(feature = "hybrid_cache_common")]
fn split_tier_migrations(
	migrations: Vec<(HashedKey, Tier)>,
) -> (Vec<(HashedKey, Tier)>, Vec<(HashedKey, Tier)>, usize) {
	let slow = migrations.iter().filter(|(_, tier)| *tier == Tier::Slow).count();
	let fast = migrations.len() - slow;

	// One-sided: nothing to reorder, so nothing to drop.
	if fast == 0 {
		return (migrations, Vec::new(), 0);
	}

	if slow == 0 {
		return (Vec::new(), migrations, 0);
	}

	// Positions are `u32` to keep a bucket at 16 bytes. The largest drains
	// recorded are millions of entries (HYBRID_CACHES.md); four billion would
	// be a 64 GiB `Vec`.
	assert!(
		u32::try_from(migrations.len()).is_ok(),
		"a drain of {} entries overflows a u32 position",
		migrations.len(),
	);

	let smaller = if slow <= fast { Tier::Slow } else { Tier::Fast };

	// Per key of the smaller half: its last position there, and the end of
	// its entries in the larger half -- one past the last of them, 0 while it
	// has none there.
	let mut last: std::collections::HashMap<HashedKey, (u32, u32), crate::NoHasher> =
		std::collections::HashMap::with_capacity_and_hasher(slow.min(fast), Default::default());

	for (i, &(key, tier)) in migrations.iter().enumerate() {
		if tier == smaller {
			last.entry(key).or_insert((0, 0)).0 = i as u32;
		}
	}

	let mut demotions = Vec::with_capacity(slow);
	let mut promotions = Vec::with_capacity(fast);

	let (smaller_half, larger_half) = match smaller {
		Tier::Slow => (&mut demotions, &mut promotions),
		Tier::Fast => (&mut promotions, &mut demotions),
	};

	// The larger half, one lookup per entry, which records where the key's
	// entries here end AND decides this one: dropped if the key has a later
	// entry in the smaller half.
	for (i, &(key, tier)) in migrations.iter().enumerate() {
		if tier == smaller {
			continue;
		}

		let i = i as u32;

		let superseded = match last.get_mut(&key) {
			Some((smaller_last, larger_end)) => {
				*larger_end = i + 1;
				*smaller_last > i
			},
			None => false,
		};

		if !superseded {
			larger_half.push((key, tier));
		}
	}

	// The smaller half, now that every key's end in the larger half is known:
	// kept if all of the key's entries there come before this one.
	for (i, &(key, tier)) in migrations.iter().enumerate() {
		if tier != smaller {
			continue;
		}

		if last[&key].1 <= i as u32 {
			smaller_half.push((key, tier));
		}
	}

	let dropped = migrations.len() - demotions.len() - promotions.len();

	(demotions, promotions, dropped)
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
{}

/// Serialises every test in this file that performs a tier migration.
///
/// The `MIG_*` dispositions in [`migration_queue`] are process-global
/// `AtomicU64`s and the test runner is parallel by default, so a sibling test
/// applying one migration between a snapshot and its successor moves the
/// count under the reader and an exact-delta assertion fails by one. Both
/// migration modules below hold this, not just the ones that read the
/// counters -- a module that only *perturbs* them is exactly as damaging as
/// one that reads them.
///
/// Nothing else in the crate's unit tests drives a migration: `apply_migration`
/// is reached only from a `MigrationQueue` consumer or from
/// `apply_migration_batches`, and the hybrid caches `lib.rs` builds in its own
/// tests are all configured with a fast tier as large as the whole cache, so
/// they never demote. When something does, it will need its own answer.
///
/// Poisoning is stepped over on purpose: a panicking test is already a
/// failure, and letting it cascade into every other test's error message only
/// hides which one broke.
#[cfg(all(test, feature = "hybrid_cache_common"))]
mod migration_test_lock {
	use std::sync::{Mutex, MutexGuard};

	pub(super) fn lock() -> MutexGuard<'static, ()> {
		static LOCK: Mutex<()> = Mutex::new(());

		LOCK.lock().unwrap_or_else(|poisoned| poisoned.into_inner())
	}
}

/// What a migration is now observable BY.
///
/// These tests used to hand the queue a `marker_migrate` closure that stamped
/// the destination buffer's last byte, so "which copy won" could be read back
/// out of the bytes, and a `DECLINE_SENTINEL` first byte to make that closure
/// return `None`. There is no such hook any more: the copy is
/// [`crate::TieredValue::migrated_to`], which a value performs on itself, and
/// it is a faithful copy with nowhere to stamp a marker.
///
/// Nothing is lost, and two things are gained:
///
///   * the destination TIER (`value.tier()`) is a migration's entire
///     observable effect, and it is the property the marker byte was standing
///     in for in the first place;
///   * HEADER IDENTITY ([`crate::TieredValue::ptr_eq`]) is strictly sharper
///     than the marker was. A marker could only say "some copy tagged for this
///     tier is here"; `ptr_eq` says exactly *which allocation* is installed, so
///     "an applied migration installs a new header" and "a declined one leaves
///     the original in place" become assertions rather than inferences. It is
///     exact rather than merely likely, too: a test holding a handle to the
///     header it snapshotted keeps that allocation alive, so its address cannot
///     be recycled underneath the comparison.
///   * the declined path needs no sentinel at all -- asking for the tier a
///     value is already in is precisely what production declines.
///
/// The value bytes are still asserted on every path, because a migration that
/// moved the tier but corrupted the payload would otherwise pass.
#[cfg(all(test, feature = "hybrid_cache_common"))]
mod migration_queue_tests {
	use super::*;

	use std::sync::atomic::Ordering;

	use super::migration_queue::{
		MIG_APPLIED,
		MIG_DECLINED,
		MIG_GONE,
		MIG_SUPERSEDED,
		MigrationQueue,
	};
	use super::migration_test_lock;
	use crate::TieredValue;
	use crate::object::Object;
	use crate::object_store::ObjectStore;
	use crate::status::AtomicStatus;

	/// The cache SHAPE under test. Never `Box<[u8]>`: that is a
	/// feature-dependent alias, and since v5 `V` is a zero-sized marker
	/// rather than the value type -- every value is a `TieredValue`.
	type TestBuffer = crate::TieredBuffer;

	const BUFFER_LEN: usize = 8;

	/// The four ways `apply_migration` can finish, read together so a test can
	/// assert the whole disposition of a batch rather than one arm of it.
	#[derive(Clone, Copy, Debug, PartialEq, Eq)]
	struct Dispositions {
		applied: u64,
		gone: u64,
		declined: u64,
		superseded: u64,
	}

	fn dispositions() -> Dispositions {
		Dispositions {
			applied: MIG_APPLIED.load(Ordering::Relaxed),
			gone: MIG_GONE.load(Ordering::Relaxed),
			declined: MIG_DECLINED.load(Ordering::Relaxed),
			superseded: MIG_SUPERSEDED.load(Ordering::Relaxed),
		}
	}

	/// The dispositions recorded since `before`. Exact, because every test
	/// that touches these counters holds `migration_test_lock`.
	fn since(before: Dispositions) -> Dispositions {
		let now = dispositions();

		Dispositions {
			applied: now.applied - before.applied,
			gone: now.gone - before.gone,
			declined: now.declined - before.declined,
			superseded: now.superseded - before.superseded,
		}
	}

	fn make_status() -> crate::StatusRef {
		Arc::new(
			AtomicStatus::new(
				1_000_000,
				&[PaperPolicy::LruCompactHybrid],
				PaperPolicy::LruCompactHybrid,
			)
			.unwrap(),
		)
	}

	fn make_objects() -> ObjectMapRef<u32, TestBuffer> {
		crate::new_hybrid_object_map()
	}

	/// Admits `key` into `tier` with every byte set to `fill`.
	///
	/// The fill is per-key in the multi-key cases, so a migration that copied
	/// the wrong object's bytes into the right object's slot is caught rather
	/// than passing as "the tier is correct".
	fn insert(
		objects: &ObjectMapRef<u32, TestBuffer>,
		key: HashedKey,
		tier: Tier,
		fill: u8,
	) {
		let object = Object::new_in(key as u32, &vec![fill; BUFFER_LEN], tier, None);
		objects.insert(key, object);
	}

	fn tier_of(objects: &ObjectMapRef<u32, TestBuffer>, key: HashedKey) -> Tier {
		objects.get_ref(&key).unwrap().value().tier()
	}

	fn bytes_of(objects: &ObjectMapRef<u32, TestBuffer>, key: HashedKey) -> Vec<u8> {
		objects.get_ref(&key).unwrap().bytes().to_vec()
	}

	/// A handle onto whatever header is installed for `key` right now.
	///
	/// Owning it is the point: it keeps that allocation alive, so a later
	/// `ptr_eq` against it cannot be fooled by a recycled address.
	fn header_of(objects: &ObjectMapRef<u32, TestBuffer>, key: HashedKey) -> TieredValue<u32> {
		objects.get_ref(&key).unwrap().snapshot()
	}

	/// Per-key ordering, which is the entire reason the queue shards by key
	/// rather than sharing one channel across its consumers.
	///
	/// Both halves are sharp because the second entry of each pair only has an
	/// effect in one of the two possible orders: a demote-then-promote pair
	/// applied backwards leaves the value Slow (the promote is declined
	/// against a still-Fast value, then the demote moves it), and a
	/// promote-then-demote pair applied backwards leaves it Fast. So the final
	/// tier alone distinguishes the orders, and the disposition counts say
	/// which entry did the work.
	#[test]
	fn per_key_demote_then_promote_applies_in_order() {
		let _serialised = migration_test_lock::lock();

		let objects = make_objects();
		let key: HashedKey = 1;
		insert(&objects, key, Tier::Fast, 0xA1);

		let admitted = header_of(&objects, key);

		// Two consumers, so the per-key channel sharding is actually in play
		// (a single consumer preserves global order trivially).
		let queue = MigrationQueue::spawn(objects.clone(), 2, make_status()).unwrap();

		// Demote then promote: the promote is the newer decision, so the value
		// must physically end Fast. Both land in the same shard's FIFO
		// channel, which is exactly the ordering the sharding exists to
		// guarantee.
		let before = dispositions();

		queue.push((key, Tier::Slow));
		queue.push((key, Tier::Fast));
		queue.flush();

		assert_eq!(tier_of(&objects, key), Tier::Fast);
		assert_eq!(
			since(before),
			Dispositions { applied: 2, gone: 0, declined: 0, superseded: 0 },
			"both entries moved the value; neither was declined, which is what \
			 the reverse order would have produced",
		);

		// Two real copies happened, so neither header can be the admitted one.
		let after_promote = header_of(&objects, key);
		assert!(
			!TieredValue::ptr_eq(&after_promote, &admitted),
			"an applied migration installs a NEW header -- it does not edit the \
			 one that was there",
		);

		// A migration is a faithful copy: the tier moved, the payload did not.
		assert_eq!(bytes_of(&objects, key), vec![0xA1; BUFFER_LEN]);

		// Mirror: promote then demote must end Slow, and this time the first
		// entry is the declined one.
		let before = dispositions();

		queue.push((key, Tier::Fast));
		queue.push((key, Tier::Slow));
		queue.flush();

		assert_eq!(tier_of(&objects, key), Tier::Slow);
		assert_eq!(
			since(before),
			Dispositions { applied: 1, gone: 0, declined: 1, superseded: 0 },
			"the promote is a no-op against an already-Fast value and the demote \
			 does the work; the reverse order would have applied both",
		);
		assert_eq!(bytes_of(&objects, key), vec![0xA1; BUFFER_LEN]);
	}

	/// `flush` counts *dispositions*, not successful swaps. All three ways a
	/// migration can finish without moving a byte must still be counted, or
	/// `flush` would never return.
	///
	/// The declined case is where the identity check earns its keep: a
	/// declined entry must leave the ORIGINAL header in place, not an
	/// identical-looking copy of it.
	#[test]
	fn flush_returns_only_after_every_disposition() {
		let _serialised = migration_test_lock::lock();

		let objects = make_objects();

		insert(&objects, 1, Tier::Fast, 0x11);
		insert(&objects, 2, Tier::Fast, 0x22);
		// Key 3 is deliberately never inserted.

		let untouched = header_of(&objects, 2);
		let before = dispositions();

		let queue = Arc::new(MigrationQueue::spawn(objects.clone(), 2, make_status()).unwrap());

		queue.push((1, Tier::Slow)); // applied
		queue.push((3, Tier::Fast)); // object absent from the map: skipped, still counted
		queue.push((2, Tier::Fast)); // already Fast: declined, still counted

		// Run `flush` on its own thread so a completion count missed on the
		// skipped or declined path shows up as a clean assertion failure
		// rather than a test process spinning forever. The timeout is a
		// failure detector only: on a correct queue the send arrives as soon
		// as the three dispositions are counted, with no timing dependence.
		let (done_tx, done_rx) = unbounded::<()>();

		let flusher = {
			let queue = queue.clone();

			thread::spawn(move || {
				queue.flush();
				let _ = done_tx.send(());
			})
		};

		assert!(
			done_rx.recv_timeout(Duration::from_secs(10)).is_ok(),
			"flush() never returned: a skipped or declined migration was not counted",
		);
		flusher.join().unwrap();

		assert_eq!(
			since(before),
			Dispositions { applied: 1, gone: 1, declined: 1, superseded: 0 },
			"one of each: the migration that moved a value, the one whose object \
			 had gone, and the one already in the tier it was asked for",
		);

		// The applied migration landed, payload intact...
		assert_eq!(tier_of(&objects, 1), Tier::Slow);
		assert_eq!(bytes_of(&objects, 1), vec![0x11; BUFFER_LEN]);

		// ...and the declined object was not merely left with equal bytes, it
		// was left with the SAME header. A decline that had gone on to build a
		// copy and swap it in would be invisible to a bytes comparison and is
		// caught here.
		assert!(
			TieredValue::ptr_eq(&header_of(&objects, 2), &untouched),
			"a declined migration must not install anything",
		);
		assert_eq!(tier_of(&objects, 2), Tier::Fast);
		assert_eq!(bytes_of(&objects, 2), vec![0x22; BUFFER_LEN]);
	}

	/// A key whose `Clone` parks the thread cloning it -- see
	/// [`a_superseded_migration_is_dropped_by_the_identity_guard`].
	struct ParkingKey(u32);

	/// Identity is the id alone; the parking is a side channel, not part of
	/// what makes two keys equal.
	impl PartialEq for ParkingKey {
		fn eq(&self, other: &Self) -> bool {
			self.0 == other.0
		}
	}

	impl Eq for ParkingKey {}

	impl Clone for ParkingKey {
		fn clone(&self) -> Self {
			park::arrive();
			ParkingKey(self.0)
		}
	}

	/// A one-shot rendezvous the next `ParkingKey::clone` walks into.
	mod park {
		use std::sync::Mutex;

		use crossbeam_channel::{Receiver, Sender, unbounded};

		static PARK: Mutex<Option<(Sender<()>, Receiver<()>)>> = Mutex::new(None);

		/// Arms the next clone to park. Returns the "it arrived" receiver and
		/// the "carry on" sender.
		pub(super) fn arm() -> (Receiver<()>, Sender<()>) {
			let (entered_tx, entered_rx) = unbounded();
			let (release_tx, release_rx) = unbounded();

			*PARK.lock().unwrap() = Some((entered_tx, release_rx));

			(entered_rx, release_tx)
		}

		/// Parks if armed, disarming as it goes so only the FIRST clone stops.
		/// The lock is held only long enough to take the rendezvous out, never
		/// across the wait.
		pub(super) fn arrive() {
			let armed = PARK.lock().unwrap().take();

			if let Some((entered, release)) = armed {
				let _ = entered.send(());
				let _ = release.recv();
			}
		}
	}

	/// Review finding values-1: a `ttl()` that lands after the copy has read
	/// the expiry, but before the swap, must survive the swap. Parks the
	/// migration at exactly that point. The `ParkingKey` hook cannot reach it:
	/// `migrated_to` clones the key BEFORE it reads the expiry, so a `ttl()`
	/// made during that park is picked up by the copy anyway.
	#[test]
	fn a_ttl_set_while_a_migration_copies_survives_the_swap() {
		let _serialised = migration_test_lock::lock();

		const KEY: HashedKey = 0xA11C_E5ED;

		let objects: ObjectMapRef<u32, TestBuffer> = crate::new_hybrid_object_map();
		objects.insert(KEY, Object::new_in(KEY as u32, &[0x11u8; BUFFER_LEN], Tier::Fast, None));

		let (entered, release) = super::migration_queue::after_copy::arm(KEY);
		let migrating = objects.clone();
		let migration = std::thread::spawn(move || {
			super::migration_queue::apply_migration(&migrating, KEY, Tier::Slow)
		});

		entered
			.recv_timeout(Duration::from_secs(10))
			.expect("the migration never reached the post-copy park point");

		// What `PaperCache::ttl` does: under the write guard, set a TTL on the
		// live object. The copy already read "no TTL".
		objects.get_mut_ref(&KEY).unwrap().expires(Some(3_600));

		release.send(()).unwrap();
		assert!(migration.join().unwrap(), "the migration should have been applied");

		let live = objects.get_ref(&KEY).unwrap();

		assert_eq!(live.value().tier(), Tier::Slow, "the migration moved the value");
		assert!(live.expiry().is_some(), "the TTL set during the copy was lost by the swap");
	}

	/// A migration computed from a value that was replaced mid-copy must be
	/// discarded -- the concurrent `set` wins, not the older copy.
	///
	/// This is the most valuable case in this module, so it is worth being
	/// precise about what changed underneath it and what did not.
	///
	/// The SHAPE is unchanged: park a consumer between its snapshot and its
	/// swap, replace the value from the test thread while it is parked, then
	/// let it finish and check which value survived.
	///
	/// The MECHANISM had to change, because the parking used to hang off the
	/// `migrate` closure the queue was handed, and there is no such closure any
	/// more. It now hangs off `K::clone`: [`crate::TieredValue::migrated_to`]
	/// rebuilds the header around a clone of the key, and it does so after the
	/// snapshot and before the swap, which is exactly the window. If that ever
	/// stops being true the rendezvous below times out with a message saying
	/// so, rather than hanging.
	///
	/// The GUARANTEE is stronger than it was. The snapshot is now a strong
	/// reference rather than an epoch pin, so the header the consumer
	/// snapshotted cannot be freed while it is parked and its address cannot be
	/// recycled into a different value -- `ptr_eq` is therefore exact rather
	/// than merely improbable to fool. The old test had to be careful to
	/// `defer_free` the displaced value for precisely that reason; here
	/// dropping the displaced handle is the whole retirement, and it frees
	/// nothing while the parked consumer still holds a reference.
	#[test]
	fn a_superseded_migration_is_dropped_by_the_identity_guard() {
		let _serialised = migration_test_lock::lock();

		const KEY: HashedKey = 1;
		const ID: u32 = 1;

		let objects: ObjectMapRef<ParkingKey, TestBuffer> = crate::new_hybrid_object_map();

		objects.insert(
			KEY,
			Object::new_in(ParkingKey(ID), &[0x11u8; BUFFER_LEN], Tier::Fast, None),
		);

		let snapshotted = objects.get_ref(&KEY).unwrap().snapshot();
		let before = dispositions();

		let (entered, release) = park::arm();

		// One consumer, so exactly one thread can be parked and the entry
		// cannot be picked up by a second.
		let queue = MigrationQueue::spawn(objects.clone(), 1, make_status()).unwrap();

		queue.push((KEY, Tier::Slow));

		entered.recv_timeout(Duration::from_secs(10)).expect(
			"the migration never reached the parked key clone: if \
			 TieredValue::migrated_to no longer clones the key between the \
			 snapshot and the swap, this test needs a new parking point -- the \
			 window it is testing still exists either way",
		);

		// The consumer holds its snapshot and is building the copy, with no
		// map guard held. Replacing the value here is exactly the interleaving
		// of a concurrent `set()` racing an in-flight migration.
		let replacement = TieredValue::new_fast(ParkingKey(ID), &[0x99u8; BUFFER_LEN], None);
		let installed = replacement.clone();

		{
			let displaced = objects.get_mut_ref(&KEY).unwrap().set_data(replacement);

			// Exactly what `set()` does with the value it displaces. Dropping
			// the handle is the complete retirement now -- the parked consumer
			// still holds a strong reference, so nothing is freed and the
			// address it is about to be compared against cannot be recycled.
			drop(displaced);
		}

		release.send(()).unwrap();
		queue.flush();

		let current = header_of_parking(&objects, KEY);

		assert!(
			TieredValue::ptr_eq(&current, &installed),
			"the concurrent set's value must survive; the migration was computed \
			 from a superseded snapshot",
		);
		assert!(
			!TieredValue::ptr_eq(&current, &snapshotted),
			"and the value the consumer snapshotted is gone from the map",
		);
		assert_eq!(current.bytes(), &[0x99u8; BUFFER_LEN][..]);
		assert_eq!(
			current.tier(),
			Tier::Fast,
			"the Slow copy must never have been published",
		);

		assert_eq!(
			since(before),
			Dispositions { applied: 0, gone: 0, declined: 0, superseded: 1 },
			"the guard rejected it, which is a superseded disposition and not an \
			 applied one",
		);
	}

	/// `header_of` for the parking-key map. Same one-liner, different `K`.
	fn header_of_parking(
		objects: &ObjectMapRef<ParkingKey, TestBuffer>,
		key: HashedKey,
	) -> TieredValue<ParkingKey> {
		objects.get_ref(&key).unwrap().snapshot()
	}

	/// Every entry reaches its own key's consumer and moves that key's own
	/// bytes.
	///
	/// Each key is admitted into one tier and asked for the other, so all 32
	/// entries are genuine moves rather than declines. The admitted tier flips
	/// every *two* keys while the shard is the key's parity, so each of the two
	/// consumers sees both directions. Each key's fill byte is its own, so a
	/// migration that copied the wrong object's bytes fails here rather than
	/// passing on the tier alone.
	#[test]
	fn migrations_for_different_keys_all_complete_across_shards() {
		let _serialised = migration_test_lock::lock();

		let objects = make_objects();

		let keys: Vec<HashedKey> = (0..32).collect();

		let admitted_tier = |key: HashedKey| {
			if (key / 2) % 2 == 0 { Tier::Fast } else { Tier::Slow }
		};

		let requested_tier = |key: HashedKey| match admitted_tier(key) {
			Tier::Fast => Tier::Slow,
			Tier::Slow => Tier::Fast,
		};

		let fill = |key: HashedKey| (key as u8).wrapping_add(1);

		for &key in &keys {
			insert(&objects, key, admitted_tier(key), fill(key));
		}

		let admitted: Vec<TieredValue<u32>> =
			keys.iter().map(|&key| header_of(&objects, key)).collect();

		let before = dispositions();

		let queue = MigrationQueue::spawn(objects.clone(), 2, make_status()).unwrap();

		for &key in &keys {
			queue.push((key, requested_tier(key)));
		}

		queue.flush();

		assert_eq!(
			since(before),
			Dispositions { applied: 32, gone: 0, declined: 0, superseded: 0 },
			"every entry was a real move",
		);

		for (index, &key) in keys.iter().enumerate() {
			assert_eq!(tier_of(&objects, key), requested_tier(key), "key {key}");
			assert_eq!(bytes_of(&objects, key), vec![fill(key); BUFFER_LEN], "key {key}");
			assert!(
				!TieredValue::ptr_eq(&header_of(&objects, key), &admitted[index]),
				"key {key}: a completed migration installs a new header",
			);
		}
	}
}

/// Completion accounting for tier migrations.
///
/// Exercised against hand-built batches rather than through a policy stack:
/// the property under test -- a promotion or demotion is counted if and only
/// if `Object::set_data` actually ran -- belongs to the application path, and
/// coaxing a real stack into emitting each of the four possible outcomes
/// (completed, object gone, already in the requested tier, guard rejected)
/// would be far more fragile than handing them over directly.
///
/// Every test runs both ways round: `queued = true` keeps the standing
/// consumer pool (the default configuration, and where the enqueued-intent
/// overcount came from), `queued = false` drops it so migrations apply
/// synchronously exactly as `MIGRATION_QUEUE_THREADS=0` would.
///
/// Gated on `hybrid_cache_common` rather than on one design's feature so the
/// cases run under whichever hybrid design is compiled in. The worker's own
/// policy stack is deliberately a plain `Lru` one: it is never consulted
/// here, and it is the one policy `init_policy_stack` builds under every
/// feature combination.
///
/// These used to build the worker with a `migrate` closure that stamped the
/// destination buffer, and set `decline = true` to make that closure return
/// `None` for everything. Neither exists: the copy is
/// `TieredValue::migrated_to` and the declined case is reached by asking for
/// the tier the value is already in, which is what production declines on. So
/// the "did the swap really happen" assertion is now the object's TIER, and
/// -- for the declined case, where the tier does not change by definition --
/// its header identity.
/// The merged store's twin of `lru_compact_hybrid_stack::overwrite_tests`: the
/// re-promotion `MergedStore::touch_slot` queues on an overwrite of a slow key
/// is load-bearing even though `set` already built the value in DRAM.
///
/// Lives here rather than in `merged_store.rs` because `apply_migration` is
/// private to this module tree.
#[cfg(all(test, feature = "merged_object_store", feature = "hybrid_cache_common"))]
mod merged_overwrite_tests {
	use std::sync::Arc;

	use super::{Tier, migration_queue::apply_migration};
	use crate::{
		HashedKey,
		merged_store::{MergedOrder, MergedStore},
		object::Object,
	};

	const K: HashedKey = 0x51;
	const A: HashedKey = 0x52;

	fn fresh(key: HashedKey) -> Object<u64, crate::TieredBuffer> {
		Object::new_in(key, &[0xA5; 256], Tier::Fast, None)
	}

	/// K is demoted as the LRU tail; that demotion is still queued when an
	/// overwrite replaces K with a value built in DRAM, so it lands on the NEW
	/// value. The re-promotion `touch_slot` queues behind it must restore it.
	#[test]
	fn an_overwrite_is_repromoted_after_a_stale_demotion() {
		// `apply_migration` bumps the process-wide migration counters that the
		// queue tests assert exact deltas on, so this runs under their lock.
		let _serialised = super::migration_test_lock::lock();

		let objects: crate::ObjectMapRef<u64, crate::TieredBuffer> = Arc::new(MergedStore::new());
		objects.set_order(MergedOrder::Lru);

		// Measure one object's tier charge untiered, then size the fast tier
		// to hold exactly one: the second admission demotes the first.
		objects.insert(K, fresh(K));
		let one = objects.fast_bytes_used();
		objects.configure_tiering(2 * one - 1, 0, 1_000_000, 1_000_000);
		objects.insert(A, fresh(A));

		assert_eq!(objects.tier_of(K), Some(Tier::Slow), "K should be the demoted LRU tail");

		let mut queue = objects.drain_migrations();
		assert_eq!(queue, vec![(K, Tier::Slow)], "the demotion is decided, not yet applied");

		// The overwrite: an LRU `set` builds the new value in DRAM, then inserts.
		objects.insert(K, fresh(K));
		queue.extend(objects.drain_migrations());

		// The key's consumer applies its entries in emission order.
		for (key, tier) in queue {
			apply_migration(&objects, key, tier);
		}

		let physical = objects.get_ref(&K).map(|object| object.value().tier());

		assert_eq!(objects.tier_of(K), Some(Tier::Fast), "an overwrite makes K the most recent key");
		assert_eq!(
			physical,
			Some(Tier::Fast),
			"K's new value was left in the slow tier while the store counts it fast",
		);
	}
}

/// The merged store's slow tier holds exactly the bytes it charges to the slow
/// tier, once the worker has applied what it drained.
///
/// Driven through the worker the way its event loop drives it -- `handle_get`
/// for a hit, then `apply_tier_migrations` -- with the store's own `touch` and
/// settle producing the drain, rather than a scripted one. The stranding
/// `split_tier_migrations` removed lived between the two: the store's
/// migration list was right, and the worker applied it backwards. With a fast
/// budget that cannot hold a hit key, the hit's promotion and the settle's
/// demotion of the same key went out demote-first, and the bytes stayed in
/// DRAM while `slow_used` counted them -- 6,877 objects, 793,088 B, the whole
/// slow-tier drift on the cluster99 golden trace (4 MiB fast tier).
///
/// Each case runs with the consumer queue (the default) and without it
/// (`MIGRATION_QUEUE_THREADS=0`); both apply the split drain.
#[cfg(all(test, feature = "merged_object_store", feature = "hybrid_cache_common"))]
mod merged_placement_tests {
	use std::sync::Arc;

	use crossbeam_channel::unbounded;

	use super::{PolicyWorker, Tier, WorkerEvent, migration_test_lock};
	use crate::{
		CacheSize, HashedKey, ObjectMapRef, PaperPolicy, TieredBuffer,
		merged_store::MergedStore,
		object::{Object, overhead::{OverheadManager, resident_object_bytes}},
		status::AtomicStatus,
	};

	type Objects = ObjectMapRef<u64, TieredBuffer>;

	/// Spreads keys across shards, which select on the HIGH bits.
	fn mix(i: u64) -> HashedKey {
		i.wrapping_mul(0x9E37_79B9_7F4A_7C15)
	}

	/// A worker whose stack is the merged store's own, as a merged build
	/// constructs it, with the fast budget then set to `fast_capacity` and no
	/// metadata reservation, so the budget is in value bytes alone.
	fn make_worker(queued: bool, fast_capacity: CacheSize) -> (PolicyWorker<u64, TieredBuffer>, Objects) {
		let (_tx, rx) = unbounded::<WorkerEvent>();

		let objects: Objects = Arc::new(MergedStore::new());
		let policy = PaperPolicy::LruCompactHybrid;
		let status = Arc::new(AtomicStatus::new(1 << 30, &[policy], policy).unwrap());
		let overhead_manager = Arc::new(OverheadManager::new(&status));

		let mut worker = PolicyWorker::new_with_tier_migration(
			rx,
			objects.clone(),
			status,
			overhead_manager,
		).unwrap();

		objects.configure_tiering(fast_capacity, 0, 1_000_000, 1_000_000);

		if !queued {
			worker.migration_queue = None;
		}

		(worker, objects)
	}

	/// What `PaperCache::set` does under LRU: build the bytes in DRAM and
	/// insert, which admits fast and settles on this thread.
	fn set(objects: &Objects, key: HashedKey, len: usize) {
		objects.insert(key, Object::new_in(key, &vec![key as u8; len], Tier::Fast, None));
	}

	/// Every key's bytes are in the tier the store charges it to, and each
	/// tier's byte total is exactly what those objects cost -- the per-object
	/// form of "modelled == measured".
	fn assert_placement_matches_the_model(objects: &Objects, keys: &[HashedKey]) {
		let mut fast: CacheSize = 0;
		let mut slow: CacheSize = 0;

		for &key in keys {
			let logical = objects.tier_of(key).expect("live key");

			let (physical, bytes) = objects
				.get_ref(&key)
				.map(|object| {
					let bytes = resident_object_bytes::<u64>(object.data_size()) as CacheSize;
					(object.value().tier(), bytes)
				})
				.expect("live key");

			assert_eq!(
				physical,
				logical,
				"key {key:#x}: the store charges it to {logical:?} but its bytes are {physical:?}",
			);

			match physical {
				Tier::Fast => fast += bytes,
				Tier::Slow => slow += bytes,
			}
		}

		assert_eq!(objects.slow_bytes_used(), slow, "slow_used is not the slow tier's bytes");
		assert_eq!(objects.fast_bytes_used(), fast, "fast_used is not the fast tier's bytes");
	}

	/// The minimal case: a budget that holds nothing, one key, one hit.
	#[test]
	fn a_promotion_undone_by_its_own_settle_leaves_the_bytes_slow() {
		for queued in [true, false] {
			let _serialised = migration_test_lock::lock();

			let (mut worker, objects) = make_worker(queued, 0);
			let key = mix(1);

			set(&objects, key, 100);
			worker.apply_tier_migrations();
			assert_placement_matches_the_model(&objects, &[key]);

			worker.handle_get(key, true);
			worker.apply_tier_migrations();

			assert_eq!(objects.tier_of(key), Some(Tier::Slow), "queued = {queued}");
			assert_placement_matches_the_model(&objects, &[key]);
		}
	}

	/// Many keys, repeated hits, and a budget that holds a few objects, so
	/// some promotions stand and some are undone at once -- applied per event,
	/// and every eight events, since the merged store's API threads push
	/// migrations between the worker's drains and a batch can span events.
	#[test]
	fn slow_bytes_are_where_the_store_counts_them_under_churn() {
		const KEYS: u64 = 256;
		const LEN: usize = 200;

		for queued in [true, false] {
			for events_per_drain in [1, 8] {
				let _serialised = migration_test_lock::lock();

				let budget = 6 * resident_object_bytes::<u64>(LEN as u32) as CacheSize;
				let (mut worker, objects) = make_worker(queued, budget);
				let keys: Vec<HashedKey> = (1..=KEYS).map(mix).collect();

				let mut events = 0;
				let mut event = |worker: &mut PolicyWorker<u64, TieredBuffer>| {
					events += 1;

					if events % events_per_drain == 0 {
						worker.apply_tier_migrations();
					}
				};

				for (n, &key) in keys.iter().enumerate() {
					set(&objects, key, LEN);
					event(&mut worker);

					// Re-hit a spread of older keys, most of them slow by now.
					for back in [1, 7, 31] {
						if n >= back {
							worker.handle_get(keys[n - back], true);
							event(&mut worker);
						}
					}
				}

				worker.apply_tier_migrations();

				assert!(objects.slow_bytes_used() > 0, "the budget must have demoted");
				assert!(objects.fast_bytes_used() > 0, "and must still hold something");
				assert_placement_matches_the_model(&objects, &keys);
			}
		}
	}
}

#[cfg(all(test, feature = "hybrid_cache_common"))]
mod migration_accounting_tests {
	use super::*;

	use super::migration_test_lock;
	use crate::TieredValue;
	use crate::{
		object::Object,
		object::overhead::OverheadManager,
		status::AtomicStatus,
	};

	/// The cache SHAPE under test. Never `Box<[u8]>`: that is a
	/// feature-dependent alias, and since v5 `V` is a zero-sized marker
	/// rather than the value type -- every value is a `TieredValue`.
	type TestBuffer = crate::TieredBuffer;

	const BUFFER_LEN: usize = 8;

	fn make_worker(queued: bool) -> (
		PolicyWorker<u32, TestBuffer>,
		ObjectMapRef<u32, TestBuffer>,
		StatusRef,
	) {
		let (_tx, rx) = unbounded::<WorkerEvent>();

		let objects: ObjectMapRef<u32, TestBuffer> = crate::new_hybrid_object_map();

		let status = Arc::new(
			AtomicStatus::new(1_000, &[PaperPolicy::Lru], PaperPolicy::Lru).unwrap(),
		);

		let overhead_manager = Arc::new(OverheadManager::new(&status));

		let mut worker = PolicyWorker::new_with_tier_migration(
			rx,
			objects.clone(),
			status.clone(),
			overhead_manager,
		).unwrap();

		if !queued {
			// Dropping the queue joins its consumers, so nothing is left in
			// flight and `apply_migration_batches` takes the synchronous path.
			worker.migration_queue = None;
		}

		(worker, objects, status)
	}

	fn insert(objects: &ObjectMapRef<u32, TestBuffer>, key: HashedKey, tier: Tier) {
		objects.insert(
			key,
			Object::new_in(key as u32, &vec![0u8; BUFFER_LEN], tier, None),
		);
	}

	fn tier_of(objects: &ObjectMapRef<u32, TestBuffer>, key: HashedKey) -> Tier {
		objects.get_ref(&key).unwrap().value().tier()
	}

	fn header_of(objects: &ObjectMapRef<u32, TestBuffer>, key: HashedKey) -> TieredValue<u32> {
		objects.get_ref(&key).unwrap().snapshot()
	}

	#[test]
	fn completed_migrations_are_counted_once_in_the_right_direction() {
		for queued in [true, false] {
			let _serialised = migration_test_lock::lock();

			let (worker, objects, status) = make_worker(queued);

			// Admitted into the tier each is about to be moved OUT of, so
			// neither entry can be declined.
			insert(&objects, 1, Tier::Fast);
			insert(&objects, 2, Tier::Slow);

			worker.apply_migration_batches(
				vec![(1, Tier::Slow), (2, Tier::Fast)],
				true,
			);

			let stats = status.hybrid_stats();

			assert_eq!(stats.demotions, 1, "queued = {queued}");
			assert_eq!(stats.promotions, 1, "queued = {queued}");

			// Both swaps really happened, so both counters describe a physical
			// copy rather than an intent to make one.
			assert_eq!(tier_of(&objects, 1), Tier::Slow, "queued = {queued}");
			assert_eq!(tier_of(&objects, 2), Tier::Fast, "queued = {queued}");
		}
	}

	#[test]
	fn migration_whose_object_was_removed_is_not_counted() {
		for queued in [true, false] {
			let _serialised = migration_test_lock::lock();

			let (worker, objects, status) = make_worker(queued);

			insert(&objects, 1, Tier::Fast);
			insert(&objects, 2, Tier::Slow);

			// Removed between the stack emitting the migration and the copy
			// being applied -- the `objects.get_ref` miss.
			objects.clear();

			worker.apply_migration_batches(
				vec![(1, Tier::Slow), (2, Tier::Fast)],
				true,
			);

			let stats = status.hybrid_stats();

			assert_eq!(stats.demotions, 0, "queued = {queued}");
			assert_eq!(stats.promotions, 0, "queued = {queued}");
		}
	}

	/// A migration into the tier the value is already in moves nothing, and so
	/// counts as nothing.
	///
	/// This is the case the old `decline` flag stood in for, reached now the
	/// way production reaches it. The tier cannot change by definition here, so
	/// the assertion that nothing happened has to be header identity: a
	/// "decline" that had gone on to build a copy and swap it in would leave
	/// the tier and the bytes looking exactly right.
	#[test]
	fn a_migration_into_the_tier_the_value_is_already_in_is_not_counted() {
		for queued in [true, false] {
			let _serialised = migration_test_lock::lock();

			let (worker, objects, status) = make_worker(queued);

			insert(&objects, 1, Tier::Slow);
			insert(&objects, 2, Tier::Fast);

			let untouched_demotion = header_of(&objects, 1);
			let untouched_promotion = header_of(&objects, 2);

			worker.apply_migration_batches(
				vec![(1, Tier::Slow), (2, Tier::Fast)],
				true,
			);

			let stats = status.hybrid_stats();

			assert_eq!(stats.demotions, 0, "queued = {queued}");
			assert_eq!(stats.promotions, 0, "queued = {queued}");

			// Nothing was installed for either key.
			assert!(
				TieredValue::ptr_eq(&header_of(&objects, 1), &untouched_demotion),
				"queued = {queued}",
			);
			assert!(
				TieredValue::ptr_eq(&header_of(&objects, 2), &untouched_promotion),
				"queued = {queued}",
			);

			assert_eq!(tier_of(&objects, 1), Tier::Slow, "queued = {queued}");
			assert_eq!(tier_of(&objects, 2), Tier::Fast, "queued = {queued}");
		}
	}

	/// The LFU-style design's `inline_demotion_accounting() == false`: its
	/// `Tier::Slow` entries include admission corrections, which need the same
	/// physical `set_data` but displace nothing. A completed slow move must
	/// therefore leave the demotion counter alone, and the real count must
	/// still arrive through `drain_demotions` without being double-counted.
	#[test]
	fn slow_moves_are_not_demotions_when_inline_accounting_is_off() {
		for queued in [true, false] {
			let _serialised = migration_test_lock::lock();

			let (worker, objects, status) = make_worker(queued);

			insert(&objects, 1, Tier::Fast);
			insert(&objects, 2, Tier::Slow);

			worker.apply_migration_batches(
				vec![(1, Tier::Slow), (2, Tier::Fast)],
				false,
			);

			let stats = status.hybrid_stats();

			// The slow move physically happened...
			assert_eq!(tier_of(&objects, 1), Tier::Slow, "queued = {queued}");

			// ...and still was not counted as a demotion.
			assert_eq!(stats.demotions, 0, "queued = {queued}");

			// Promotions carry no such ambiguity.
			assert_eq!(stats.promotions, 1, "queued = {queued}");

			// `drain_demotions` is that design's only source of demotions;
			// `apply_tier_migrations` forwards it verbatim through this same
			// counter, and the completed slow move above has not inflated it.
			status.record_hybrid_demotions(3);

			assert_eq!(status.hybrid_stats().demotions, 3, "queued = {queued}");
		}
	}

	/// A stack that hands the worker ONE scripted drain and nothing else, so an
	/// exact entry sequence can go through `apply_tier_migrations` -- the path
	/// production takes, `split_tier_migrations` included -- without coaxing
	/// a real stack into emitting it.
	struct ScriptedDrain {
		migrations: Vec<(HashedKey, Tier)>,
	}

	impl PolicyStack for ScriptedDrain {
		fn is_policy(&self, policy: &PaperPolicy) -> bool {
			matches!(policy, PaperPolicy::Lru)
		}

		fn len(&self) -> usize {
			0
		}

		fn contains(&self, _key: HashedKey) -> bool {
			false
		}

		fn insert(&mut self, _key: HashedKey, _size: ObjectSize) {}

		fn remove(&mut self, _key: HashedKey) {}

		fn clear(&mut self) {}

		fn evict_one(&mut self) -> Option<HashedKey> {
			None
		}

		fn drain_tier_migrations(&mut self) -> Vec<(HashedKey, Tier)> {
			std::mem::take(&mut self.migrations)
		}
	}

	/// Runs one scripted drain through `apply_tier_migrations` and reports
	/// the copies it made: `MIG_APPLIED`'s delta, exact under
	/// `migration_test_lock`.
	fn apply_scripted(
		worker: &mut PolicyWorker<u32, TestBuffer>,
		migrations: Vec<(HashedKey, Tier)>,
	) -> u64 {
		use std::sync::atomic::Ordering;

		let before = migration_queue::MIG_APPLIED.load(Ordering::Relaxed);

		worker.policy_stack = Some(Box::new(ScriptedDrain { migrations }));
		worker.apply_tier_migrations();

		migration_queue::MIG_APPLIED.load(Ordering::Relaxed) - before
	}

	/// A slow value whose key is promoted and then demoted again in ONE drain
	/// -- net intent Slow -- ends slow, and nothing is copied.
	///
	/// Without the split dropping the superseded promotion, the
	/// demotions-first partition applied the pair backwards: the demote
	/// declined against a value still slow, the promote copied it into DRAM,
	/// and the stack counted it slow from then on. Run on both paths, one test
	/// each, because the synchronous path (`MIGRATION_QUEUE_THREADS=0`) shares
	/// the partition with the queued one.
	fn promotion_undone_in_one_drain(queued: bool) {
		let _serialised = migration_test_lock::lock();

		let (mut worker, objects, status) = make_worker(queued);

		insert(&objects, 1, Tier::Slow);
		let admitted = header_of(&objects, 1);

		let coalesced_before = migstats::COALESCED_TOT.load(std::sync::atomic::Ordering::Relaxed);

		let copies = apply_scripted(&mut worker, vec![(1, Tier::Fast), (1, Tier::Slow)]);

		assert_eq!(
			tier_of(&objects, 1),
			Tier::Slow,
			"queued = {queued}: the value was left in DRAM while the stack's last word \
			 on it was Slow -- stranded",
		);
		assert_eq!(copies, 0, "queued = {queued}: a promotion copy was made and undone");
		assert!(
			TieredValue::ptr_eq(&header_of(&objects, 1), &admitted),
			"queued = {queued}: a copy was installed",
		);

		let stats = status.hybrid_stats();

		assert_eq!(stats.promotions, 0, "queued = {queued}");
		assert_eq!(stats.demotions, 0, "queued = {queued}");

		// Process-global and bumped by any worker in the binary, so only a
		// lower bound is exact.
		assert!(
			migstats::COALESCED_TOT.load(std::sync::atomic::Ordering::Relaxed) > coalesced_before,
			"queued = {queued}: the dropped entry was not counted",
		);
	}

	#[test]
	fn a_promotion_undone_in_one_drain_leaves_the_value_slow_and_uncopied_through_the_queue() {
		promotion_undone_in_one_drain(true);
	}

	#[test]
	fn a_promotion_undone_in_one_drain_leaves_the_value_slow_and_uncopied_when_applied_inline() {
		promotion_undone_in_one_drain(false);
	}

	/// The mirror: a fast value demoted and then promoted again in one drain
	/// ends fast, with no demote copy and no promote copy back. Without the
	/// split dropping the superseded demotion the placement came out right --
	/// demote first was the right order here -- but through a round trip to
	/// CXL and back.
	fn demotion_undone_in_one_drain(queued: bool) {
		let _serialised = migration_test_lock::lock();

		let (mut worker, objects, status) = make_worker(queued);

		insert(&objects, 1, Tier::Fast);
		let admitted = header_of(&objects, 1);

		let copies = apply_scripted(&mut worker, vec![(1, Tier::Slow), (1, Tier::Fast)]);

		assert_eq!(tier_of(&objects, 1), Tier::Fast, "queued = {queued}");
		assert_eq!(copies, 0, "queued = {queued}: a round trip through CXL was made");
		assert!(
			TieredValue::ptr_eq(&header_of(&objects, 1), &admitted),
			"queued = {queued}: a copy was installed",
		);

		let stats = status.hybrid_stats();

		assert_eq!(stats.promotions, 0, "queued = {queued}");
		assert_eq!(stats.demotions, 0, "queued = {queued}");
	}

	#[test]
	fn a_demotion_undone_in_one_drain_leaves_the_value_fast_and_uncopied_through_the_queue() {
		demotion_undone_in_one_drain(true);
	}

	#[test]
	fn a_demotion_undone_in_one_drain_leaves_the_value_fast_and_uncopied_when_applied_inline() {
		demotion_undone_in_one_drain(false);
	}
}

/// `split_tier_migrations` on hand-built drains: each half in drain order, no
/// key in both, and every key ending in the tier of its last entry.
#[cfg(all(test, feature = "hybrid_cache_common"))]
mod migration_split_tests {
	use super::*;

	type Drain = Vec<(HashedKey, Tier)>;

	#[test]
	fn a_promotion_then_a_demotion_of_one_key_leaves_only_the_demotion() {
		assert_eq!(
			split_tier_migrations(vec![(7, Tier::Fast), (7, Tier::Slow)]),
			(vec![(7, Tier::Slow)], vec![], 1),
		);
	}

	#[test]
	fn a_demotion_then_a_promotion_of_one_key_leaves_only_the_promotion() {
		assert_eq!(
			split_tier_migrations(vec![(7, Tier::Slow), (7, Tier::Fast)]),
			(vec![], vec![(7, Tier::Fast)], 1),
		);
	}

	#[test]
	fn distinct_keys_keep_drain_order_within_each_half() {
		assert_eq!(
			split_tier_migrations(vec![(1, Tier::Slow), (2, Tier::Fast), (3, Tier::Slow), (4, Tier::Fast)]),
			(vec![(1, Tier::Slow), (3, Tier::Slow)], vec![(2, Tier::Fast), (4, Tier::Fast)], 0),
		);
	}

	#[test]
	fn an_empty_or_single_entry_drain_is_untouched() {
		assert_eq!(split_tier_migrations(Vec::new()), (vec![], vec![], 0));
		assert_eq!(split_tier_migrations(vec![(9, Tier::Slow)]), (vec![(9, Tier::Slow)], vec![], 0));
		assert_eq!(split_tier_migrations(vec![(9, Tier::Fast)]), (vec![], vec![(9, Tier::Fast)], 0));
	}

	/// All demotions or all promotions, repeated keys included: returned
	/// whole as the one half -- the same allocation, so nothing was copied,
	/// let alone hashed -- with nothing dropped.
	#[test]
	fn a_one_sided_drain_is_returned_whole_with_nothing_dropped() {
		for tier in [Tier::Slow, Tier::Fast] {
			let drain: Drain = vec![(1, tier), (2, tier), (1, tier), (3, tier)];
			let expected = drain.clone();
			let allocation = drain.as_ptr();

			let (demotions, promotions, dropped) = split_tier_migrations(drain);

			let (whole, other) = match tier {
				Tier::Slow => (demotions, promotions),
				Tier::Fast => (promotions, demotions),
			};

			assert_eq!(whole, expected, "{tier:?}");
			assert!(std::ptr::eq(whole.as_ptr(), allocation), "{tier:?}: the drain was copied");
			assert!(other.is_empty(), "{tier:?}");
			assert_eq!(dropped, 0, "{tier:?}");
		}
	}

	/// A key repeated in ONE tier of a mixed drain keeps every entry: the
	/// partition cannot reorder them against each other, and the second
	/// declines.
	#[test]
	fn same_tier_duplicates_in_a_mixed_drain_are_kept() {
		assert_eq!(
			split_tier_migrations(vec![(1, Tier::Fast), (2, Tier::Slow), (1, Tier::Fast)]),
			(vec![(2, Tier::Slow)], vec![(1, Tier::Fast), (1, Tier::Fast)], 0),
		);
	}

	/// Interleaved keys, two of them reversed within the drain: only the
	/// entries no later entry for the other tier supersedes are kept, in
	/// drain order. Once with the demotions the smaller half and once with
	/// the promotions, since the map is keyed on whichever is smaller.
	#[test]
	fn interleaved_reversals_keep_only_what_nothing_later_supersedes() {
		// Demotions the smaller half, 3 of 7.
		assert_eq!(
			split_tier_migrations(vec![
				(1, Tier::Fast),
				(2, Tier::Slow),
				(1, Tier::Slow),
				(3, Tier::Fast),
				(2, Tier::Fast),
				(1, Tier::Fast),
				(4, Tier::Slow),
			]),
			(vec![(4, Tier::Slow)], vec![(3, Tier::Fast), (2, Tier::Fast), (1, Tier::Fast)], 3),
		);

		// Promotions the smaller half, 2 of 7.
		assert_eq!(
			split_tier_migrations(vec![
				(1, Tier::Slow),
				(2, Tier::Slow),
				(1, Tier::Fast),
				(3, Tier::Slow),
				(2, Tier::Fast),
				(1, Tier::Slow),
				(4, Tier::Slow),
			]),
			(vec![(3, Tier::Slow), (1, Tier::Slow), (4, Tier::Slow)], vec![(2, Tier::Fast)], 3),
		);
	}

	/// `apply_migration`'s rule on a two-key placement (keys 1 and 2) -- an
	/// entry moves its key to the entry's tier, or declines if the key is
	/// there already -- returning the final placement and each key's copies.
	fn apply<'a>(
		mut placement: [Tier; 2],
		entries: impl IntoIterator<Item = &'a (HashedKey, Tier)>,
	) -> ([Tier; 2], [u32; 2]) {
		let mut copies = [0; 2];

		for &(key, tier) in entries {
			let k = (key - 1) as usize;

			if placement[k] != tier {
				placement[k] = tier;
				copies[k] += 1;
			}
		}

		(placement, copies)
	}

	/// The contract on one drain, checked from its definition rather than
	/// against a second copy of the walk.
	fn check(drain: &Drain) {
		let (demotions, promotions, dropped) = split_tier_migrations(drain.clone());

		assert!(demotions.iter().all(|&(_, tier)| tier == Tier::Slow), "{drain:?}: {demotions:?}");
		assert!(promotions.iter().all(|&(_, tier)| tier == Tier::Fast), "{drain:?}: {promotions:?}");

		for &(key, _) in &demotions {
			assert!(
				!promotions.iter().any(|&(k, _)| k == key),
				"{drain:?}: key {key} is in both halves: {demotions:?} / {promotions:?}",
			);
		}

		// Kept if and only if no later entry for the key names the other
		// tier; each half in drain order, same-tier duplicates included.
		let kept: Drain = drain
			.iter()
			.enumerate()
			.filter(|&(i, &(key, tier))| !drain[i + 1..].iter().any(|&(k, t)| k == key && t != tier))
			.map(|(_, &entry)| entry)
			.collect();

		let kept_in = |tier: Tier| -> Drain {
			kept.iter().copied().filter(|&(_, t)| t == tier).collect()
		};

		assert_eq!(demotions, kept_in(Tier::Slow), "{drain:?}: the demotions");
		assert_eq!(promotions, kept_in(Tier::Fast), "{drain:?}: the promotions");
		assert_eq!(dropped, drain.len() - kept.len(), "{drain:?}: the dropped count");

		for start in [
			[Tier::Slow, Tier::Slow],
			[Tier::Slow, Tier::Fast],
			[Tier::Fast, Tier::Slow],
			[Tier::Fast, Tier::Fast],
		] {
			let (in_order, _) = apply(start, drain);
			let (split, copies) = apply(start, demotions.iter().chain(&promotions));

			for (k, key) in [1, 2].into_iter().enumerate() {
				// Where the key's last entry in drain order puts it.
				let last = drain
					.iter()
					.rev()
					.find(|&&(entry_key, _)| entry_key == key)
					.map_or(start[k], |&(_, tier)| tier);

				assert_eq!(in_order[k], last, "{drain:?} from {start:?}: key {key}, applied in order");
				assert_eq!(
					split[k],
					last,
					"{drain:?} from {start:?}: key {key} ended {:?}, its last entry says {last:?}",
					split[k],
				);
				assert!(
					copies[k] <= 1,
					"{drain:?} from {start:?}: key {key} was copied {} times",
					copies[k],
				);
			}
		}
	}

	/// Every drain of up to five entries over two keys and both tiers --
	/// 1 + 4 + 16 + 64 + 256 + 1,024 = 1,365 of them, one-sided and mixed,
	/// with either half the smaller -- against the contract: the halves hold
	/// only their own tier and share no key; an entry is kept if and only if
	/// no later entry for its key names the other tier, in drain order within
	/// its half; `dropped` counts the rest; and from every starting placement
	/// of the two keys, applying the demotions and then the promotions ends
	/// each key where applying the drain in order does -- its last entry's
	/// tier -- copying it at most once.
	#[test]
	fn every_drain_of_up_to_five_entries_on_two_keys_keeps_the_contract() {
		const ENTRIES: [(HashedKey, Tier); 4] = [
			(1, Tier::Slow),
			(1, Tier::Fast),
			(2, Tier::Slow),
			(2, Tier::Fast),
		];

		let mut drains = 0;

		for len in 0..=5u32 {
			for code in 0..4usize.pow(len) {
				let drain: Drain = (0..len)
					.map(|p| ENTRIES[code / 4usize.pow(p) % 4])
					.collect();

				check(&drain);
				drains += 1;
			}
		}

		assert_eq!(drains, 1_365);
	}
}

/// `polling_delay`, the decision `delay_event_loop` parks on, with the clock
/// supplied rather than waited for.
#[cfg(test)]
mod polling_delay_tests {
	use super::*;

	/// The idle trap: the first pass to see a set after a quiet spell longer
	/// than the recency window must poll SHORT. It used to poll LONG, because
	/// the recency test read only the sets before this pass, and the burst
	/// then waited up to a second for its settle.
	#[test]
	fn the_first_pass_to_see_a_set_after_a_long_idle_polls_short() {
		let last_set = Instant::now();
		let now = last_set + 4 * SET_RECENCY_DURATION;

		assert_eq!(polling_delay(now, Some(last_set), true), SHORT_POLLING_DURATION);
	}

	/// The same trap at start-up: a cache's very first sets.
	#[test]
	fn the_first_pass_to_see_any_set_at_all_polls_short() {
		assert_eq!(polling_delay(Instant::now(), None, true), SHORT_POLLING_DURATION);
	}

	/// Inside the window, with or without a set in this pass, up to and
	/// including its edge.
	#[test]
	fn a_pass_inside_the_recency_window_polls_short() {
		let last_set = Instant::now();

		for elapsed in [Duration::ZERO, SET_RECENCY_DURATION / 2, SET_RECENCY_DURATION] {
			for has_current_set in [false, true] {
				assert_eq!(
					polling_delay(last_set + elapsed, Some(last_set), has_current_set),
					SHORT_POLLING_DURATION,
					"{elapsed:?} after the last set, has_current_set = {has_current_set}",
				);
			}
		}
	}

	/// Only an idle worker -- no set in this pass, none within the window --
	/// polls LONG.
	#[test]
	fn an_idle_pass_with_no_current_set_polls_long() {
		let last_set = Instant::now();

		for elapsed in [SET_RECENCY_DURATION + Duration::from_millis(1), 4 * SET_RECENCY_DURATION] {
			assert_eq!(
				polling_delay(last_set + elapsed, Some(last_set), false),
				LONG_POLLING_DURATION,
				"{elapsed:?} after the last set",
			);
		}

		assert_eq!(polling_delay(Instant::now(), None, false), LONG_POLLING_DURATION);
	}
}

/// `AtomicStatus::kick_policy_worker`, and the idle poll it interrupts,
/// against a real worker thread.
///
/// Gated on `hybrid_cache_common` only for the object-map constructor, as
/// `migration_accounting_tests` is; nothing here is hybrid.
#[cfg(all(test, feature = "hybrid_cache_common"))]
mod policy_worker_kick_tests {
	use super::*;

	use crate::{object::overhead::OverheadManager, status::AtomicStatus};

	/// Polls `done` every 100 us until it holds, failing after `deadline`.
	/// The deadline is a hang detector, not the measurement.
	fn wait_for(what: &str, deadline: Duration, mut done: impl FnMut() -> bool) {
		let start = Instant::now();

		while !done() {
			assert!(start.elapsed() < deadline, "{what} did not happen within {deadline:?}");
			thread::sleep(Duration::from_micros(100));
		}
	}

	type WorkerHandle = thread::JoinHandle<Result<(), CacheError>>;

	/// A real worker, on its own thread, for a cache that has seen nothing.
	fn spawn_worker() -> (Sender<WorkerEvent>, StatusRef, WorkerHandle) {
		let (tx, rx) = unbounded::<WorkerEvent>();

		let objects: ObjectMapRef<u32, crate::TieredBuffer> = crate::new_hybrid_object_map();

		let status: StatusRef = Arc::new(
			AtomicStatus::new(1_000_000, &[PaperPolicy::Lru], PaperPolicy::Lru).unwrap(),
		);

		let overhead_manager = Arc::new(OverheadManager::new(&status));

		let worker = PolicyWorker::<u32, crate::TieredBuffer>::new(
			rx,
			objects,
			status.clone(),
			overhead_manager,
			None,
		).unwrap();

		(tx, status, register_worker(worker))
	}

	/// Waits for the worker's first pass, then CHECKS the premise both tests
	/// rest on rather than assuming it: with nothing queued and no set ever
	/// seen, that pass chose the LONG poll, so no other pass runs in the next
	/// 100 ms (on the SHORT poll about a hundred would). Returns the pass
	/// count to measure from.
	fn parked_on_the_long_poll(status: &StatusRef) -> u64 {
		wait_for("the worker's first pass", Duration::from_secs(10), || {
			status.policy_worker_passes() >= 1
		});

		let passes = status.policy_worker_passes();

		thread::sleep(Duration::from_millis(100));

		assert_eq!(
			status.policy_worker_passes(),
			passes,
			"the idle worker ran another pass within 100 ms of its first: it is not \
			 parked on the {LONG_POLLING_DURATION:?} poll, so a kick would prove nothing",
		);

		passes
	}

	/// Stops the worker -- kicked, so the shutdown does not wait out a poll
	/// either -- and checks that it exited cleanly.
	fn shut_down(tx: Sender<WorkerEvent>, status: &StatusRef, handle: WorkerHandle) {
		tx.send(WorkerEvent::Shutdown).unwrap();
		status.kick_policy_worker();

		assert!(
			handle.join().expect("the worker thread panicked").is_ok(),
			"the worker returned an error",
		);
	}

	/// A worker parked on the LONG poll -- idle, no set ever seen -- runs a
	/// pass as soon as it is kicked, not when its second is up.
	///
	/// Not timing-fragile in the direction that matters. That the worker is
	/// parked is checked first: it ran no pass for 100 ms after its first. If
	/// this thread is then descheduled and the worker is somehow not parked
	/// when the kick lands, the kick leaves an unpark token and the park
	/// returns at once, which is FASTER. The bound is 200 ms against a 1 s
	/// poll, so what fails it is a worker that slept through the kick, not a
	/// slow scheduler.
	///
	/// The printed `KICK_LATENCY` is an UPPER bound set by this test's own
	/// 100 us poll of the pass counter (a sleep, which overshoots), not a
	/// measurement of the unpark.
	#[test]
	fn a_kick_wakes_a_worker_parked_on_the_long_poll() {
		let (tx, status, handle) = spawn_worker();

		let passes = parked_on_the_long_poll(&status);
		let kicked = Instant::now();

		status.kick_policy_worker();

		wait_for("a pass after the kick", Duration::from_secs(10), || {
			status.policy_worker_passes() > passes
		});

		let latency = kicked.elapsed();

		println!("KICK_LATENCY {latency:?} (an upper bound: the pass counter is polled every 100 us)");

		shut_down(tx, &status, handle);

		assert!(
			latency < Duration::from_millis(200),
			"the kicked worker took {latency:?} to run a pass: it slept through the \
			 kick to the end of its {LONG_POLLING_DURATION:?} poll",
		);
	}

	/// The idle fix's WIRING, which `polling_delay_tests` cannot see: `run`
	/// must hand `delay_event_loop` the sets THIS pass handled, and
	/// `delay_event_loop` must poll on them.
	///
	/// A worker idle on the LONG poll is given a set and kicked. The pass the
	/// kick starts handles the set and must choose the SHORT poll, so the
	/// pass after it follows within milliseconds: two passes, well inside
	/// 200 ms of the kick. Choosing LONG there -- what the worker did before
	/// the decision counted the current pass -- puts the second pass a whole
	/// poll later.
	#[test]
	fn the_pass_that_handles_the_first_set_after_an_idle_spell_polls_short() {
		let (tx, status, handle) = spawn_worker();

		let passes = parked_on_the_long_poll(&status);

		// Queued BEFORE the kick, so the pass the kick starts takes it.
		tx.send(WorkerEvent::Set(1, 64, 0, None, None)).unwrap();

		let kicked = Instant::now();

		status.kick_policy_worker();

		// Timed separately, so a failure names its cause: the first pass is
		// the kick's (`a_kick_wakes_a_worker_parked_on_the_long_poll`), the
		// second is the poll the set's pass chose.
		wait_for("a pass after the kick", Duration::from_secs(10), || {
			status.policy_worker_passes() > passes
		});

		let first = kicked.elapsed();

		wait_for("a second pass after the kick", Duration::from_secs(10), || {
			status.policy_worker_passes() >= passes + 2
		});

		let second = kicked.elapsed();

		shut_down(tx, &status, handle);

		assert!(
			first < Duration::from_millis(200),
			"the kicked worker took {first:?} to run the pass that handles the set: it \
			 slept through the kick",
		);
		assert!(
			second < Duration::from_millis(200),
			"the pass after the one that handled the set came {second:?} after the kick: \
			 that pass parked on the {LONG_POLLING_DURATION:?} poll",
		);
	}
}


/// The capacity-eviction watermark (`eviction_watermarks`), against the real
/// `apply_evictions` loop.
///
/// Gated on `hybrid_cache_common` for the same reason
/// `migration_accounting_tests` is -- the object-map constructor these build
/// on lives behind it -- but nothing here is hybrid. The stack is a plain
/// `Lru` one: the policy `init_policy_stack` builds under every feature
/// combination, and the one whose `needs_capacity_eviction` is always
/// `false`, so the capacity cases below isolate the `over_max_size`
/// condition. The one case that does need the other condition brings its own
/// stack.
#[cfg(all(test, feature = "hybrid_cache_common"))]
mod capacity_watermark_tests {
	use super::*;

	use super::eviction_watermarks::{DEFAULT_HIGH, DEFAULT_LOW, Watermarks, clamped_low};

	use crate::{
		object::Object,
		object::overhead::{OverheadManager, get_policy_overhead},
		status::AtomicStatus,
	};

	/// The cache SHAPE under test. Never `Box<[u8]>`: that is a
	/// feature-dependent alias, and since v5 `V` is a zero-sized marker
	/// rather than the value type -- every value is a `TieredValue`.
	type TestBuffer = crate::TieredBuffer;

	const TEST_POLICY: PaperPolicy = PaperPolicy::Lru;
	const VALUE_BYTES: usize = 16;

	/// Cap the status is constructed with, replaced by `make_worker` as soon
	/// as it can measure one object -- see there.
	const PLACEHOLDER_MAX_SIZE: CacheSize = 1_000_000;

	/// What one test object actually costs `used_size`: its `base_size` plus
	/// the per-object policy overhead `used_size` adds on top of every object
	/// it counts. Measured rather than hardcoded, because both terms are
	/// tuned constants elsewhere in the tree and have moved before.
	fn per_object_size(overhead_manager: &OverheadManagerRef) -> CacheSize {
		let probe = Object::<u32, TestBuffer>::new(0u32, &vec![0u8; VALUE_BYTES], None);
		(overhead_manager.base_size(&probe) + get_policy_overhead(&TEST_POLICY)) as CacheSize
	}

	fn used(status: &StatusRef) -> CacheSize {
		status.used_size(&TEST_POLICY)
	}

	/// Builds a worker whose cap is exactly `objects_at_max` objects' worth of
	/// accounted bytes, so every threshold below is an exact object count
	/// instead of a rounded byte figure that could hide an off-by-one
	/// settling point.
	///
	/// The cap can only be sized once an `OverheadManager` exists to measure
	/// an object with, and that needs a status that already carries a cap --
	/// hence the placeholder, then `set_max_size`. Nothing goes stale:
	/// `apply_evictions` reads `max_size()` on entry, and `LruStack` ignores
	/// `resize` altogether.
	fn make_worker(objects_at_max: u64) -> (
		PolicyWorker<u32, TestBuffer>,
		ObjectMapRef<u32, TestBuffer>,
		StatusRef,
		OverheadManagerRef,
	) {
		let (_tx, rx) = unbounded::<WorkerEvent>();

		let objects: ObjectMapRef<u32, TestBuffer> = crate::new_hybrid_object_map();

		let status = Arc::new(
			AtomicStatus::new(PLACEHOLDER_MAX_SIZE, &[TEST_POLICY], TEST_POLICY).unwrap(),
		);

		let overhead_manager = Arc::new(OverheadManager::new(&status));

		status.set_max_size(objects_at_max * per_object_size(&overhead_manager));

		let worker = PolicyWorker::<u32, TestBuffer>::new(
			rx,
			objects.clone(),
			status.clone(),
			overhead_manager.clone(),
			None,
		).unwrap();

		(worker, objects, status, overhead_manager)
	}

	/// Mirrors what `PaperCache::set()` does to shared state before calling
	/// `handle_set`, exactly as the other test modules in this file do, so
	/// `status.used_size()` and the policy stack's own bookkeeping agree.
	fn insert(
		objects: &ObjectMapRef<u32, TestBuffer>,
		status: &StatusRef,
		overhead_manager: &OverheadManagerRef,
		worker: &mut PolicyWorker<u32, TestBuffer>,
		key: HashedKey,
	) {
		let object = Object::new(key as u32, &vec![0u8; VALUE_BYTES], None);
		let base_size = overhead_manager.base_size(&object);
		let dram_resident = overhead_manager.dram_resident_size(&object);

		objects.insert(key, object);
		status.update_base_used_size(base_size as i64);
		status.incr_num_objects();
		worker.handle_set(key, base_size, dram_resident);
	}

	fn fill(
		objects: &ObjectMapRef<u32, TestBuffer>,
		status: &StatusRef,
		overhead_manager: &OverheadManagerRef,
		worker: &mut PolicyWorker<u32, TestBuffer>,
		keys: std::ops::RangeInclusive<HashedKey>,
	) {
		for key in keys {
			insert(objects, status, overhead_manager, worker, key);
		}
	}

	/// A stack whose *internal* sub-structure is over its own budget -- the
	/// `needs_capacity_eviction` case. It insists on draining to `budget`
	/// objects however much room the cache as a whole still has.
	/// `TwoQCompactHybridStack`'s `k_in`-derived fifo budget is the real
	/// instance; this stands in for it because that one needs a hybrid design
	/// compiled in and a fast tier configured, neither of which this condition
	/// (or the watermark that must stay off it) has anything to do with.
	struct SubBudgetStack {
		keys: VecDeque<HashedKey>,
		budget: usize,
	}

	impl PolicyStack for SubBudgetStack {
		fn is_policy(&self, policy: &PaperPolicy) -> bool {
			matches!(policy, PaperPolicy::Lru)
		}

		fn len(&self) -> usize {
			self.keys.len()
		}

		fn contains(&self, key: HashedKey) -> bool {
			self.keys.contains(&key)
		}

		fn insert(&mut self, key: HashedKey, _size: ObjectSize) {
			self.keys.push_back(key);
		}

		fn remove(&mut self, key: HashedKey) {
			self.keys.retain(|existing| *existing != key);
		}

		fn clear(&mut self) {
			self.keys.clear();
		}

		fn evict_one(&mut self) -> Option<HashedKey> {
			self.keys.pop_front()
		}

		fn needs_capacity_eviction(&self) -> bool {
			self.keys.len() > self.budget
		}
	}

	#[test]
	fn default_watermarks_are_exactly_max_size_at_every_cache_size() {
		assert_eq!(DEFAULT_HIGH, 1.0);
		assert_eq!(DEFAULT_LOW, 1.0);

		let defaults = Watermarks::new(DEFAULT_HIGH, DEFAULT_LOW);

		// Includes caps past f64's exact-integer range: a `u64 -> f64 -> u64`
		// round trip loses the low bits above 2^53, and a threshold a single
		// byte off `max_size` is a silent change in eviction depth -- the one
		// thing the default may never be.
		for max_size in [0u64, 1, 1_000, (1u64 << 53) + 1, u64::MAX] {
			assert_eq!(defaults.bytes(max_size), (max_size, max_size));
		}
	}

	#[test]
	fn default_watermarks_settle_the_cache_at_exactly_max_size() {
		let (mut worker, objects, status, overhead_manager) = make_worker(16);
		let per_object = per_object_size(&overhead_manager);
		let max_size = status.max_size();

		// The wiring half of the 1.0 guarantee: a worker built with neither
		// var set carries the defaults. Skipped rather than failed when the
		// run itself exports an override, which is a deliberate configuration
		// and not a regression.
		if std::env::var("EVICTION_HIGH_WATERMARK").is_err()
			&& std::env::var("EVICTION_LOW_WATERMARK").is_err()
		{
			assert_eq!(
				worker.eviction_watermarks,
				Watermarks::new(DEFAULT_HIGH, DEFAULT_LOW),
			);
		}

		// Pinned so the case still tests 1.0/1.0 semantics under such a run.
		worker.eviction_watermarks = Watermarks::new(DEFAULT_HIGH, DEFAULT_LOW);

		fill(&objects, &status, &overhead_manager, &mut worker, 1..=18);
		assert_eq!(used(&status), 18 * per_object);

		let mut buffered_events = Vec::new();
		worker.apply_evictions(&mut buffered_events).unwrap();

		// Exactly at the cap, not one object under it: the pre-watermark loop
		// stops the instant `used_size` is no longer *over* `max_size`, and
		// every published sweep in `results/` was measured against a cache
		// that settles right there.
		assert_eq!(used(&status), max_size);
		assert_eq!(used(&status), 16 * per_object);
		assert_eq!(objects.len() as u64, 16);
	}

	#[test]
	fn a_triggered_pass_drains_past_max_size_and_does_not_rearm_under_the_high_mark() {
		let (mut worker, objects, status, overhead_manager) = make_worker(16);
		let per_object = per_object_size(&overhead_manager);
		let max_size = status.max_size();

		// 0.75 and 0.5 are exact in binary floating point, so the marks are
		// exactly 12 and 8 objects' worth and no rounding step can hide an
		// off-by-one settling point.
		worker.eviction_watermarks = Watermarks::new(0.75, 0.5);
		assert_eq!(
			worker.eviction_watermarks.bytes(max_size),
			(12 * per_object, 8 * per_object),
		);

		fill(&objects, &status, &overhead_manager, &mut worker, 1..=18);
		assert!(used(&status) > max_size);

		let mut buffered_events = Vec::new();
		worker.apply_evictions(&mut buffered_events).unwrap();

		// One pass, and it went well past `max_size` -- the pre-watermark loop
		// would have stopped at exactly 16 objects' worth.
		assert_eq!(used(&status), 8 * per_object);
		assert_eq!(objects.len() as u64, 8);
		assert!(used(&status) < max_size);

		// Back up to 11 objects: above the drain target, below the trigger.
		fill(&objects, &status, &overhead_manager, &mut worker, 19..=21);
		worker.apply_evictions(&mut buffered_events).unwrap();

		// Nothing evicted at all. A pass that re-armed anywhere under the high
		// mark would drain to 8 again on every set, which is the batch-of-one
		// behaviour the watermark exists to get rid of.
		assert_eq!(used(&status), 11 * per_object);
		assert_eq!(objects.len() as u64, 11);
	}

	#[test]
	fn an_inverted_watermark_pair_is_clamped_to_the_high_mark() {
		assert_eq!(clamped_low(0.75, 0.875), 0.75);
		assert_eq!(clamped_low(0.75, 0.5), 0.5);

		let (mut worker, objects, status, overhead_manager) = make_worker(16);
		let per_object = per_object_size(&overhead_manager);
		let max_size = status.max_size();

		// `low > high`. Unclamped, a pass would arm at 12 objects' worth and
		// find itself already under a 14-objects'-worth drain target: it would
		// evict nothing and leave the cache parked above the high mark with no
		// way back down.
		worker.eviction_watermarks = Watermarks::new(0.75, 0.875);
		assert_eq!(
			worker.eviction_watermarks.bytes(max_size),
			(12 * per_object, 12 * per_object),
		);

		fill(&objects, &status, &overhead_manager, &mut worker, 1..=18);

		let mut buffered_events = Vec::new();
		worker.apply_evictions(&mut buffered_events).unwrap();

		assert_eq!(used(&status), 12 * per_object);
		assert_eq!(objects.len() as u64, 12);
	}

	#[test]
	fn an_internal_capacity_eviction_is_not_extended_to_the_low_watermark() {
		let (mut worker, objects, status, overhead_manager) = make_worker(16);
		let per_object = per_object_size(&overhead_manager);

		// Marks that would drain to 8 objects' worth if the internal condition
		// were ever allowed to arm them.
		worker.eviction_watermarks = Watermarks::new(0.75, 0.5);
		worker.policy_stack = Some(Box::new(SubBudgetStack {
			keys: VecDeque::new(),
			budget: 2,
		}));

		fill(&objects, &status, &overhead_manager, &mut worker, 1..=4);

		let mut buffered_events = Vec::new();
		worker.apply_evictions(&mut buffered_events).unwrap();

		// Four objects is far below even the low mark, so the capacity
		// condition never fires: the stack drains to its own budget and stops
		// there, instead of being pulled all the way down to `low * max_size`
		// by a watermark that has nothing to say about its sub-structure.
		assert_eq!(objects.len() as u64, 2);
		assert_eq!(used(&status), 2 * per_object);
	}
}
