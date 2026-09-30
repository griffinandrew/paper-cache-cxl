/*
 * Copyright (c) Kia Shakiba
 *
 * This source code is licensed under the GNU AGPLv3 license found in the
 * LICENSE file in the root directory of this source tree.
 */


/// Persistent migration queue: a standing pool that drains physical tier
/// copies continuously, decoupled from batch boundaries.
///
/// Fanning a single *batch* out across a pool (the removed `parallel_migration`
/// module) was measured not to help this workload: 99.4% of demotion volume
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
	use crate::worker::policy::policy_stack::{MigrationEntry, MigrationOrigin, TaggedMigration, Tier};

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

	/// `(PENDING_DEMOTE, PENDING_PROMOTE)`: entries handed to the consumers
	/// and not finished yet. Read by `crate::phys::pending_migrations` and the
	/// MEMTS line's `pending_net`.
	pub fn pending() -> (u64, u64) {
		(PENDING_DEMOTE.load(Ordering::Acquire), PENDING_PROMOTE.load(Ordering::Acquire))
	}

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


	/// Migrations IN FLIGHT, and migrations LANDED, per key bucket: what the
	/// policy worker knows of what a queued migration may yet do, or has just
	/// done, to a key's value -- with no identity on the queue's entries. It
	/// is what the reconcile's two rules read (`Observed` in this file): the
	/// new-key rule (a key re-admitted while a migration of its bucket was in
	/// flight, or landed after its value was published, gets a corrective that
	/// lands last) and the heal rule (a slow-served hit is healed only while
	/// nothing of its bucket is in flight).
	///
	/// `IN_FLIGHT_BUCKETS` words, indexed by the key's low bits: `HashedKey` is already a
	/// hash. Each word packs two 32-bit counts:
	///
	///   * low half, PENDING: entries for keys of this bucket handed to the
	///     consumers (`MigrationQueue::push`, one per entry, after
	///     `split_tier_migrations`) and not finished. A consumer finishes an
	///     entry once `apply_migration` has returned, however it ended --
	///     applied, declined, gone or superseded -- so an entry being copied
	///     still counts. Zero whenever the queue is idle -- unless a consumer
	///     thread has died: the entries still in its channel are never
	///     finished, so their buckets stay busy for the life of the cache.
	///     That is the same class as the queue's `processed` count, which
	///     then never catches up with `enqueued`, so a `flush` would not
	///     return;
	///   * high half, LANDED: entries of this bucket that MOVED a value
	///     (`apply_migration` returned `true`), on a consumer or inline
	///     (`MIGRATION_QUEUE_THREADS=0`, which never has anything pending but
	///     lands migrations all the same). Wraps at 2^32 and is only ever
	///     compared for equality.
	///
	/// Cost: two atomic read-modify-writes per queued entry -- the hand-off's
	/// increment and the finish, one RMW however the entry ended (landed:
	/// `+2^32 - 1`, one more landed and one fewer pending, the pending half
	/// never borrowing because its increment happened-before, as the channel
	/// orders it; not landed: `-1`) -- and one per inline landing. Readers pay
	/// one load. 128 KiB per tiered cache, built on first use.
	///
	/// Two keys in one bucket make it look busier than either key is: a false
	/// positive, costing one corrective that declines or one heal deferred to
	/// a later hit -- never a missed fence.
	///
	/// Under a BACKLOG that is the common case, not a collision. With D
	/// entries of distinct keys in flight, a given bucket is busy with
	/// probability about 1 - e^(-D/16384): 63% at D = 16k, 95% at 50k. The
	/// new-key rule's fence then fires for most fresh sets -- each a
	/// corrective that usually declines, and that is handed to the consumers
	/// and counted in `PENDING_*` like any entry -- and the heal is
	/// effectively off until the backlog drains: most slow-served hits find
	/// their bucket busy (`migstats::RECONCILE_GET_HEAL_SKIPPED` counts them).
	pub struct InFlight {
		words: Box<[AtomicU64]>,
	}

	/// Buckets in `InFlight`: 2^14 words, 128 KiB.
	pub const IN_FLIGHT_BUCKETS: usize = 1 << 14;

	const PENDING: u64 = 1;
	const LANDED: u64 = 1 << 32;

	impl InFlight {
		pub fn new() -> Self {
			InFlight {
				words: (0..IN_FLIGHT_BUCKETS).map(|_| AtomicU64::new(0)).collect(),
			}
		}

		#[inline]
		fn word(&self, key: HashedKey) -> &AtomicU64 {
			&self.words[key as usize & (IN_FLIGHT_BUCKETS - 1)]
		}

		/// One entry for `key` handed to the consumers. Charged before the
		/// send, as `record_pending` is, so no finish can precede it; the
		/// channel orders the two.
		#[inline]
		pub(crate) fn handed(&self, key: HashedKey) {
			self.word(key).fetch_add(PENDING, Ordering::Relaxed);
		}

		/// A refused send: the hand-off undone.
		#[inline]
		fn refund(&self, key: HashedKey) {
			self.word(key).fetch_sub(PENDING, Ordering::Release);
		}

		/// A consumer is done with one entry for `key`; `landed`: it moved the
		/// value. `Release`: a reader that sees this sees the swap.
		#[inline]
		pub(crate) fn finished(&self, key: HashedKey, landed: bool) {
			match landed {
				true => self.word(key).fetch_add(LANDED - PENDING, Ordering::Release),
				false => self.word(key).fetch_sub(PENDING, Ordering::Release),
			};
		}

		/// A migration of `key` landed inline, on the worker
		/// (`MIGRATION_QUEUE_THREADS=0`): nothing was pending.
		#[inline]
		pub(crate) fn landed_inline(&self, key: HashedKey) {
			self.word(key).fetch_add(LANDED, Ordering::Release);
		}

		/// The client's MARK: the landed count of `key`'s bucket, read BEFORE
		/// the client publishes a value, and carried by the value's `Set`.
		#[inline]
		pub fn mark(&self, key: HashedKey) -> u32 {
			(self.word(key).load(Ordering::Acquire) >> 32) as u32
		}

		/// Entries of `key`'s bucket in flight now.
		#[inline]
		pub fn pending(&self, key: HashedKey) -> u32 {
			self.word(key).load(Ordering::Acquire) as u32
		}

		/// Whether a migration of `key`'s bucket is in flight now, or has
		/// landed since `mark` was read.
		#[inline]
		pub fn moved_since(&self, key: HashedKey, mark: u32) -> bool {
			let word = self.word(key).load(Ordering::Acquire);

			word as u32 != 0 || (word >> 32) as u32 != mark
		}

		/// Every bucket's pending count, summed: 0 once every entry handed to
		/// the consumers has finished, which is what a flush or a quiescent
		/// cache must show -- unless a consumer thread has died with entries
		/// still in its channel, which are never finished (see the struct's
		/// doc). A diagnostic (one load per bucket).
		pub fn total_pending(&self) -> u64 {
			self.words.iter().map(|word| word.load(Ordering::Acquire) as u32 as u64).sum()
		}
	}

	impl Default for InFlight {
		fn default() -> Self {
			Self::new()
		}
	}

	/// Finishes one entry in `InFlight` however the consumer loop body exits
	/// -- `PendingOnDrop`'s reasoning -- with `landed` set once
	/// `apply_migration` has returned `true`. Declared after the other two
	/// guards, so dropped before them: by the time `processed` counts the
	/// entry (and a `flush` returns), its bucket shows it finished.
	struct Finish<'a> {
		in_flight: &'a InFlight,
		key: HashedKey,
		landed: bool,
	}

	impl Drop for Finish<'_> {
		fn drop(&mut self) {
			self.in_flight.finished(self.key, self.landed);
		}
	}

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
		senders: Vec<Sender<TaggedMigration>>,
		handles: Vec<JoinHandle<()>>,

		/// The cache's per-bucket in-flight and landed counts (`InFlight`):
		/// charged by `push`, finished by the consumers.
		in_flight: Arc<InFlight>,

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
			let in_flight = status.migration_in_flight().clone();

			// Seeded with the `PolicyStack` trait default; the first
			// `apply_tier_migrations` pass overwrites it before it can push
			// anything, so this value is never actually read.
			let demotion_accounting = Arc::new(AtomicBool::new(true));

			let mut senders = Vec::with_capacity(threads);
			let mut handles = Vec::with_capacity(threads);

			for index in 0..threads {
				// Per-consumer channel rather than one shared queue -- see the
				// ordering note on `senders`.
				let (sender, receiver) = unbounded::<TaggedMigration>();

				let objects = objects.clone();
				let processed = processed.clone();
				let in_flight = in_flight.clone();
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

						// Completed CORRECTIVES (`MigrationOrigin::Reconcile`),
						// by destination: counted apart, never as promotions
						// or demotions -- a corrective moves bytes to where
						// the stack already placed the key and displaces
						// nothing.
						let mut reconciled_to_fast: u64 = 0;
						let mut reconciled_to_slow: u64 = 0;

						while let Ok((key, tier, origin)) = receiver.recv() {
							// Test builds: a gate test's paused or paced consumers.
							#[cfg(test)]
							crate::gate::test_hooks::consumer_wait();

							// Counted on every path out of this iteration.
							let _done = CountOnDrop(&processed);
							let _pending = PendingOnDrop(tier);
							let mut finish = Finish { in_flight: &in_flight, key, landed: false };

							if apply_migration(&objects, key, tier) {
								finish.landed = true;

								// S5 B2: a landed demotion freed fast bytes (the old
								// copy went when `apply_migration` returned, unless a
								// reader holds it): a set waiting for them is woken
								// now. One load when none waits.
								if tier == Tier::Slow {
									status.gate().note_demotion();
								}

								match (origin, tier) {
									(MigrationOrigin::Stack, Tier::Fast) => completed_promotions += 1,

									// Not every completed slow move is a demotion
									// -- see `demotion_accounting`'s doc on the
									// struct. A `Relaxed` load off a line this
									// thread already owns.
									(MigrationOrigin::Stack, Tier::Slow) => {
										if demotion_accounting.load(Ordering::Relaxed) {
											completed_demotions += 1;
										}
									},

									(MigrationOrigin::Reconcile, Tier::Fast) => reconciled_to_fast += 1,
									(MigrationOrigin::Reconcile, Tier::Slow) => reconciled_to_slow += 1,
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
							if (completed_promotions | completed_demotions | reconciled_to_fast | reconciled_to_slow) != 0
								&& receiver.is_empty()
							{
								status.record_hybrid_promotions(completed_promotions);
								status.record_hybrid_demotions(completed_demotions);
								status.record_reconcile_applied(reconciled_to_fast, reconciled_to_slow);

								completed_promotions = 0;
								completed_demotions = 0;
								reconciled_to_fast = 0;
								reconciled_to_slow = 0;

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
						status.record_reconcile_applied(reconciled_to_fast, reconciled_to_slow);
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
				in_flight,
				enqueued: AtomicU64::new(0),
				processed,
				demotion_accounting,
			})
		}

		/// Hands one migration to the consumer that owns this key, tagged with
		/// its origin (an untagged `(key, tier)` is a stack entry), and counts
		/// it in flight in the key's bucket (`InFlight`).
		///
		/// `HashedKey` is already a hash, so the low bits are well distributed
		/// and a modulo is an adequate shard selector. A send failure means
		/// that consumer is gone, which only happens during shutdown; the
		/// stack state is already correct either way, so the copy is dropped.
		pub fn push(&self, item: impl MigrationEntry) {
			let item = item.tagged();

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
			self.in_flight.handed(item.0);

			// Single consumer: one FIFO channel already preserves global
			// order, so there is nothing to shard and the modulo is skipped.
			// Sharding only does work when there is more than one consumer to
			// distribute across.
			if self.senders.len() == 1 {
				match first.send(item).is_ok() {
					true => self
						.record_depth(self.enqueued.fetch_add(1, Ordering::Release) + 1),
					false => self.refuse(item),
				}

				return;
			}

			let shard = (item.0 % self.senders.len() as HashedKey) as usize;

			match self.senders[shard].send(item).is_ok() {
				true => self
					.record_depth(self.enqueued.fetch_add(1, Ordering::Release) + 1),
				false => self.refuse(item),
			}
		}

		/// A refused send: both charges `push` made, refunded.
		fn refuse(&self, item: TaggedMigration) {
			drop(PendingOnDrop(item.1));
			self.in_flight.refund(item.0);
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
		/// migration. The value is constant for the life of a hybrid cache (its
		/// policy is fixed when it is built), so this is a cheap way
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
			let (sender, receiver) = bounded::<TaggedMigration>(0);
			let in_flight = Arc::new(InFlight::new());

			let queue = MigrationQueue {
				senders: vec![sender],
				handles: Vec::new(),
				in_flight: in_flight.clone(),
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
			let (key, tier, _origin) = receiver
				.recv()
				.expect("the producer must still be parked in `send`");

			drop(PendingOnDrop(tier));
			in_flight.finished(key, false);
			assert_eq!(in_flight.total_pending(), 0, "the bucket's charge was paid back too");

			producer.join().expect("`push` must not panic");

			assert_eq!(key, 7);
			assert_eq!(tier, Tier::Slow);
			assert_eq!(PENDING_DEMOTE.load(Ordering::Acquire), before);
		}
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
	/// are `DEMO_TOT + PROMO_TOT + COALESCED_TOT` plus the correctives kept,
	/// `RECONCILE_QUEUED_TO_SLOW + RECONCILE_QUEUED_TO_FAST` (below), and a
	/// promote-then-demote pair for one key shows up here as 1 rather than as
	/// a promote copy undone by a demote copy. A same-tier duplicate is not
	/// dropped: it is counted in `DEMO`/`PROMO` (or, a corrective, in
	/// `RECONCILE_QUEUED_*`) like any other entry, and declines.
	pub static COALESCED_TOT: AtomicU64 = AtomicU64::new(0);

	/// Corrective migrations the policy worker's reconcile QUEUED (S3; see
	/// `Observed`): a `set` whose value was built in the slow tier for a key
	/// the stack places fast (`SET_TO_FAST`) or the reverse (`SET_TO_SLOW`),
	/// and a hit served from the slow tier on a key the stack places fast --
	/// the heal (`GET_TO_FAST`). Each is one queue entry pushed behind
	/// whatever the stack queued for the key; one that finds the bytes
	/// already moved is declined by its consumer, so these count intents, as
	/// `DEMO`/`PROMO` do. Process-global, like everything here: printed on a
	/// MIGSTATS line of their own and exported by `HybridStats`.
	#[cfg(feature = "hybrid_cache_common")]
	pub static RECONCILE_SET_TO_FAST: AtomicU64 = AtomicU64::new(0);
	#[cfg(feature = "hybrid_cache_common")]
	pub static RECONCILE_SET_TO_SLOW: AtomicU64 = AtomicU64::new(0);
	#[cfg(feature = "hybrid_cache_common")]
	pub static RECONCILE_GET_TO_FAST: AtomicU64 = AtomicU64::new(0);

	/// Correctives the NEW-KEY RULE alone queued (`PolicyWorker::handle_set`):
	/// a key (re-)admitted as new while a migration of its bucket was in
	/// flight, or had landed since the value was published, and whose value
	/// was built where the stack places it -- queued only so that it lands
	/// LAST, behind whatever stale entry of the key may still be queued. A
	/// corrective the built tier asked for anyway is `SET_TO_*`. Intents.
	#[cfg(feature = "hybrid_cache_common")]
	pub static RECONCILE_SET_NEW_KEY: AtomicU64 = AtomicU64::new(0);

	/// Hits served from the slow tier whose HEAL the worker skipped because
	/// something of the key's in-flight bucket was busy (`Observed`'s heal
	/// rule): no `placement_of` probe and no corrective. An UPPER BOUND on the
	/// heals skipped, not a count of them -- the placement is not read for
	/// these hits (not reading it is the point), so a hit on a key placed
	/// slow, or on a key whose own promotion is what is in flight, is counted
	/// although it needed no heal. Under a backlog it is most slow-served
	/// hits: with D entries in flight a bucket is busy with probability about
	/// 1 - e^(-D/16384) (`migration_queue::InFlight`), and heals are then
	/// effectively off until the backlog drains. Not a corrective, so in no
	/// `RECONCILE_QUEUED_*` sum.
	#[cfg(feature = "hybrid_cache_common")]
	pub static RECONCILE_GET_HEAL_SKIPPED: AtomicU64 = AtomicU64::new(0);

	/// Reconcile-origin entries (`MigrationOrigin::Reconcile`: the worker's
	/// correctives, in every store) handed on
	/// after `split_tier_migrations`, by destination -- their own intent
	/// counters, kept OUT of `DEMO`/`PROMO`, which count the stacks' policy
	/// decisions. A drain's entries are `DEMO_TOT + PROMO_TOT +
	/// RECONCILE_QUEUED_TO_SLOW + RECONCILE_QUEUED_TO_FAST + COALESCED_TOT`.
	#[cfg(feature = "hybrid_cache_common")]
	pub static RECONCILE_QUEUED_TO_FAST: AtomicU64 = AtomicU64::new(0);
	#[cfg(feature = "hybrid_cache_common")]
	pub static RECONCILE_QUEUED_TO_SLOW: AtomicU64 = AtomicU64::new(0);

	/// Reconcile-origin entries that LANDED -- moved a value's bytes -- on a
	/// consumer or inline, by destination. Never promotions or demotions: a
	/// corrective moves bytes to where the stack already placed the key, and
	/// displaces nothing. Process-global; `HybridStats::reconcile_applied_*`
	/// carries each cache's own.
	#[cfg(feature = "hybrid_cache_common")]
	pub static RECONCILE_APPLIED_TO_FAST: AtomicU64 = AtomicU64::new(0);
	#[cfg(feature = "hybrid_cache_common")]
	pub static RECONCILE_APPLIED_TO_SLOW: AtomicU64 = AtomicU64::new(0);

	/// `(RECONCILE_SET_TO_FAST, RECONCILE_SET_TO_SLOW, RECONCILE_GET_TO_FAST,
	/// RECONCILE_SET_NEW_KEY, RECONCILE_GET_HEAL_SKIPPED)`.
	#[cfg(feature = "hybrid_cache_common")]
	pub fn reconciled() -> (u64, u64, u64, u64, u64) {
		(
			RECONCILE_SET_TO_FAST.load(Ordering::Relaxed),
			RECONCILE_SET_TO_SLOW.load(Ordering::Relaxed),
			RECONCILE_GET_TO_FAST.load(Ordering::Relaxed),
			RECONCILE_SET_NEW_KEY.load(Ordering::Relaxed),
			RECONCILE_GET_HEAL_SKIPPED.load(Ordering::Relaxed),
		)
	}

	/// Counts reconcile-origin entries handed on after the split.
	#[cfg(feature = "hybrid_cache_common")]
	pub(crate) fn reconcile_queued(to_fast: usize, to_slow: usize) {
		if to_fast != 0 {
			RECONCILE_QUEUED_TO_FAST.fetch_add(to_fast as u64, Ordering::Relaxed);
		}

		if to_slow != 0 {
			RECONCILE_QUEUED_TO_SLOW.fetch_add(to_slow as u64, Ordering::Relaxed);
		}
	}

	/// Counts reconcile-origin entries that landed.
	#[cfg(feature = "hybrid_cache_common")]
	pub(crate) fn reconcile_applied(to_fast: u64, to_slow: u64) {
		if to_fast != 0 {
			RECONCILE_APPLIED_TO_FAST.fetch_add(to_fast, Ordering::Relaxed);
		}

		if to_slow != 0 {
			RECONCILE_APPLIED_TO_SLOW.fetch_add(to_slow, Ordering::Relaxed);
		}
	}

	pub static CALLS: AtomicU64 = AtomicU64::new(0);
	static START: OnceLock<Instant> = OnceLock::new();
	static LAST_DUMP_MS: AtomicU64 = AtomicU64::new(0);

	/// The origin of every instrumentation line's `t_ms` -- MIGSTATS here,
	/// DIVERGE and MEMTS in the worker loop -- so the three series share one
	/// clock. Set by the first `PolicyWorker` built in the process
	/// (`mark_origin`), i.e. when the first cache is constructed: in the
	/// one-cache process the server and the benchmark run, `t_ms` is the time
	/// since that cache started. `START` above paces the periodic dump and is
	/// left as it was.
	static ORIGIN: OnceLock<Instant> = OnceLock::new();

	pub fn mark_origin() {
		ORIGIN.get_or_init(Instant::now);
	}

	/// Milliseconds since `ORIGIN`.
	pub fn t_ms() -> u64 {
		ORIGIN.get_or_init(Instant::now).elapsed().as_millis() as u64
	}
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

		// One timestamp for the whole block, appended as the LAST field of
		// every line -- the same rule `coalesced_tot` follows below, so every
		// existing field keeps its place for a `key=value` reader.
		let t_ms = t_ms();

		#[cfg(feature = "hybrid_cache_common")]
		eprintln!(
			"MIGSTATS queue_depth_max={} burst_max={} pending_demote_max={} pending_promote_max={} t_ms={t_ms}",
			super::migration_queue::DEPTH_MAX.load(Ordering::Relaxed),
			super::migration_queue::BURST_MAX.load(Ordering::Relaxed),
			super::migration_queue::PENDING_DEMOTE_MAX.load(Ordering::Relaxed),
			super::migration_queue::PENDING_PROMOTE_MAX.load(Ordering::Relaxed),
		);

		#[cfg(feature = "hybrid_cache_common")]
		eprintln!(
			"MIGSTATS pending_net_max={} t_ms={t_ms}",
			super::migration_queue::PENDING_NET_MAX.load(Ordering::Relaxed),
		);

		#[cfg(feature = "hybrid_cache_common")]
		eprintln!(
			"MIGSTATS applied={} gone={} declined={} superseded={} t_ms={t_ms}",
			super::migration_queue::MIG_APPLIED.load(Ordering::Relaxed),
			super::migration_queue::MIG_GONE.load(Ordering::Relaxed),
			super::migration_queue::MIG_DECLINED.load(Ordering::Relaxed),
			super::migration_queue::MIG_SUPERSEDED.load(Ordering::Relaxed),
		);

		// `coalesced_tot` goes LAST but for `t_ms`: scripts read this line as
		// `key=value` pairs, and appending keeps every existing field where it
		// was.
		eprintln!("MIGSTATS mig_calls={} evict_calls={} demo_tot={} promo_tot={} evict_tot={} coalesced_tot={} t_ms={t_ms}",
			CALLS.load(Ordering::Relaxed), ECALLS.load(Ordering::Relaxed),
			DEMO_TOT.load(Ordering::Relaxed), PROMO_TOT.load(Ordering::Relaxed),
			EVICT_TOT.load(Ordering::Relaxed), COALESCED_TOT.load(Ordering::Relaxed));
		eprintln!("MIGSTATS demo={} t_ms={t_ms}", f(&DEMO));
		eprintln!("MIGSTATS promo={} t_ms={t_ms}", f(&PROMO));
		eprintln!("MIGSTATS evict={} t_ms={t_ms}", f(&EVICT));

		// The reconcile (S3): a line of its own, LAST, so every line above
		// keeps its fields and its place; `t_ms` last, as on every line. The
		// correctives the worker queued, by reason; every reconcile-origin
		// entry handed on after the split, which `demo`/`promo` above no
		// longer count; those
		// that landed, which are not promotions or demotions; and, appended
		// after them so each keeps its place, the slow-served hits whose heal
		// a busy bucket skipped.
		#[cfg(feature = "hybrid_cache_common")]
		{
			let (set_to_fast, set_to_slow, get_to_fast, set_new_key, get_heal_skipped) = reconciled();

			eprintln!(
				"MIGSTATS reconcile_set_to_fast={set_to_fast} reconcile_set_to_slow={set_to_slow} \
				 reconcile_get_to_fast={get_to_fast} reconcile_set_new_key={set_new_key} \
				 reconcile_queued_to_fast={} reconcile_queued_to_slow={} \
				 reconcile_applied_to_fast={} reconcile_applied_to_slow={} \
				 reconcile_get_heal_skipped={get_heal_skipped} t_ms={t_ms}",
				RECONCILE_QUEUED_TO_FAST.load(Ordering::Relaxed),
				RECONCILE_QUEUED_TO_SLOW.load(Ordering::Relaxed),
				RECONCILE_APPLIED_TO_FAST.load(Ordering::Relaxed),
				RECONCILE_APPLIED_TO_SLOW.load(Ordering::Relaxed),
			);
		}
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

use std::{
	thread,
	time::{Instant, Duration},
};

use typesize::TypeSize;
use crossbeam_channel::{Sender, Receiver};

// Gated exactly as the `object_store` module itself is (see `lib.rs`) rather
// than on the hybrid features that were its original users, because
// `handle_expire` needs it on any build that has it available. A build that
// selects no storage feature at all (e.g. bare `eviction_stacks_pmem`) has no
// `object_store` module to import; `PolicyWorker::object_exists` carries a
// second body for that case.
#[cfg(any(feature = "all_dram", feature = "key_value_pmem"))]
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
	object::ObjectSize,
	worker::{
		Worker,
		WorkerEvent,
		WorkerReceiver,
		policy::policy_stack::PolicyStack,
	},
};

// The split stacks' constructor: a merged build's stack is the object map itself.
#[cfg(not(feature = "merged_object_store"))]
use crate::worker::policy::policy_stack::init_policy_stack;

// The policy, for the tiered paths (the byte gate's state, the size-split
// shares) and the tests.
#[cfg(feature = "hybrid_cache_common")]
use crate::policy::PaperPolicy;

// What the tiered test modules below take through `use super::*`.
#[cfg(all(test, feature = "hybrid_cache_common"))]
use std::sync::Arc;
#[cfg(all(test, feature = "hybrid_cache_common"))]
use crossbeam_channel::unbounded;
#[cfg(all(test, feature = "hybrid_cache_common"))]
use crate::worker::register_worker;

// Re-exported (fully `pub`, not `pub(crate)`) so sibling modules (e.g.
// `worker::manager`) can name `Tier` without reaching into the private
// `policy_stack` submodule directly, *and* so it can flow all the way out
// to `PaperCache::tier_of`'s public return type via `worker::Tier` /
// `crate::Tier` (see `worker/mod.rs` and `lib.rs`).
pub use policy_stack::Tier;

// What a `Set` did to the object map (`PolicyStack::insert_set`), for the
// merged store's `worker_set`; and the CLOCK hand's shared budget.
pub use policy_stack::SetEvent;

// Where a set's value was built, and why (S5): the `Set`'s placement byte.
pub use policy_stack::Placement;

// The settle target's ratio, for the byte gate's levels (S5, `gate::bands`).
#[cfg(feature = "hybrid_cache_common")]
pub(crate) use policy_stack::drain_target;
#[cfg(feature = "merged_object_store")]
pub(crate) use policy_stack::clock_hand_budget;

// The tagged drain entry: who queued a migration travels with it to the
// consumer that counts it (see `MigrationOrigin`).
#[cfg(any(feature = "hybrid_cache_common", feature = "merged_object_store"))]
pub use policy_stack::{MigrationEntry, MigrationOrigin, TaggedMigration};

/// What a client observed of a key's bytes, carried to the policy worker by
/// its event: the tier a `set` BUILT the new value in (`WorkerEvent::Set`), or
/// that a hit was SERVED from the slow tier (`WorkerEvent::Get`). Once the
/// stack has handled the event, `PolicyWorker::drain_reconciled` compares it
/// with where the stack places the key (`PolicyStack::placement_of`) and
/// queues a corrective migration -- tagged `MigrationOrigin::Reconcile` --
/// where they disagree and nothing the stack queued for the event already
/// moves the bytes there: the RECONCILE (backpressure plan S3, P3/U12).
///
/// Why the worker, and why every set. The client picks a value's tier before
/// the worker sees the set, in `hybrid_policy::admission_tier`, which reads
/// either a MIRROR the worker publishes once per pass (the LFU admission
/// latch) or the key's current PHYSICAL tier (FIFO, CLOCK, LRU-LFU and the
/// S3-FIFO designs keep an existing key where it is), and both can be stale by
/// the time the value is in the map:
///
///   * the LFU stale latch: a burst of new keys outruns the mirror, so keys
///     the latched stack admits slow -- emitting nothing, because it trusts
///     the build -- are built in DRAM (S2's burst: 147 values physically
///     fast, the stack counting 111 of them slow, nothing pending; measured
///     earlier on the split path, 7,999 of 8,000 objects in DRAM while the
///     stack reported 5,966 slow and a compliant `fast_bytes_used`);
///   * race (a): a migration of the key lands, or loses its swap, between
///     `admission_tier`'s read and the insert, so the new value is built in
///     the tier the stack has just moved the key out of -- and the merged
///     store's overwrite branch queued nothing for it.
///
/// The stack says where the bytes should be; the event says where they were
/// built. Comparing the two on the worker fixes both stores the same way and
/// fixes the latch without recording the built tier in the stack. In both
/// stores the worker is where a new key is placed (the merged store's client
/// only publishes it -- `MergedStore::insert`), so no earlier point knows the
/// placement, and the reconcile's corrective, in the `Set`'s own drain, is the
/// only one.
///
/// Why a corrective is safe to queue whenever the two disagree. It goes behind
/// whatever is already queued for the key, on the key's one FIFO consumer
/// (`MigrationQueue`), and at the END of this event's drain, so
/// `split_tier_migrations` keeps it over an earlier entry of the drain for
/// the other tier: the last intent wins, and it is the newest there is (the
/// placement is read after the drain; anything decided later is queued
/// later, and wins in its turn). One that finds the bytes already there -- an
/// earlier entry moved them, or it duplicates a promotion still pending -- is
/// DECLINED by `apply_migration`: a dequeue and a lookup, no copy. One that
/// finds the value replaced mid-copy is SUPERSEDED, and the replacement's own
/// `Set` is reconciled in its turn.
///
/// # The tracked invariant
///
/// Queued migrations carry no identity: an entry acts on whatever value holds
/// its key when it is dequeued. What makes that sound is this invariant: FOR
/// A KEY THE STACK PLACES, THE LAST ENTRY QUEUED FOR IT THAT IS STILL IN
/// FLIGHT NAMES ITS PLACEMENT. Every design pushes a tier change of a key it
/// tracks in the call that makes it (the `placement_of` contract, checked at
/// quiescence by T9's audit and, between quiescent points, by T9's zero
/// reconcile deltas over its one-at-a-time phases); every corrective names
/// the placement read after the drain; and every (re-)admission either
/// pushes its own entry or -- whenever anything of its bucket may be stale --
/// gets the new-key corrective below. So a value published under a tracked
/// key ends at the placement once the key's in-flight entries have landed,
/// whatever they found, and an OVERWRITE needs only the plain rule (its
/// built tier against the placement). The overwrite-restore argument --
/// `touch_slot`'s and `ArenaHybridStack::touch_to_front`'s `(k, Fast)`
/// queued behind a stale demotion of the old value -- is this invariant for
/// the case where the stack pushes the re-placement itself, and still holds.
///
/// The placement changes no push accompanies are all UN-TRACKINGS: `del`
/// (`handle_del`), an eviction (`evict_one`), a TTL reap (`handle_expire`,
/// then `handle_del`) and a wipe (`clear`). Each leaves the key's in-flight
/// entries acting on whatever value holds the key next -- the new-key case.
/// A resize changes placements only through evictions and the settle, which
/// pushes (`resize_fast_tier`), and a flat stack has no tiers.
///
/// # The new-key rule (review M1)
///
/// A key (re-)admitted as NEW has its placement decided without a push
/// whenever the stack admits it where the client built it -- 2Q's and
/// S3-FIFO's slow admission queues, the latched LFU, any design's fast
/// admission of a value built fast. Then no entry queued after a stale one
/// says where the fresh value belongs, and the stale one moves it for good:
///
///   (i)   2Q: `k` placed fast, its bytes still slow. A hit is served slow,
///         then `del(k)` and `set(k, v2)`, built slow. The worker takes the
///         hit first and heals `(k, Fast)`, which lands on v2; `Set(v2)` admits
///         `k` to the slow FIFO, where v2 was built, and queues nothing: v2 is
///         stranded in DRAM, charged slow, and no hit ever looks at it (a
///         fast-served hit is not looked up);
///   (ii)  LFU: v1 built slow under a stale mirror and admitted fast, so the
///         reconcile queues `(k, Fast)`; `k` is evicted; v2 is set built slow
///         and admitted slow (latched); the corrective lands on v2: stranded;
///   (iii) a stale STACK `(k, Fast)` still queued when `k` is deleted or
///         evicted and re-set into a slow admission queue: the same.
///
/// The rule: when the worker handles a `Set` for a key the stack did not
/// place before it (`PolicyWorker::handle_set` says how that is told), and a
/// migration of the key's bucket is IN FLIGHT when the event is handled or
/// has LANDED since the client published the value (`InFlight::moved_since`,
/// against the mark the client read before publishing), `(k, placement)` is
/// appended to this event's drain EVEN IF the value was built there -- unless
/// the drain's own last entry for the key already names it. The worker takes
/// events in channel order and hands each event's drain to the queue before
/// the next, so this entry is queued after every entry computed from an
/// earlier event and after every stale stack entry; the key's FIFO consumer
/// applies it last, and the split keeps it (the drain's last for the key).
/// The fresh value ends where the stack placed it: (i) to (iii) are all this.
///
/// Why "in flight OR landed since the mark", not the in-flight count alone:
/// an entry can land on v2 AFTER the client published it and BEFORE the
/// worker reaches its `Set`. In (i) the heal is pushed while the worker
/// handles the hit, ahead of the `Set` in the channel, and it may well have
/// finished by the time the `Set` is handled; then nothing is in flight, and
/// only the landed count shows that a value of the bucket moved after v2 was
/// published. The client reads its mark BEFORE it publishes, and an entry
/// that lands on v2 dequeued v2 after the publish, so its landing is ordered
/// after the mark: it is either still in flight when the worker looks, or in
/// the landed half. With neither, no entry of the bucket has touched v2 and
/// none can -- nothing of the key is queued -- so v2 is where it was built,
/// and the plain rule suffices. The inline path (`MIGRATION_QUEUE_THREADS=0`)
/// has nothing in flight but the same window -- the heal of (i) applied
/// inline, while the worker handles the hit, onto an already-published v2 --
/// so its landings are counted too.
///
/// A quiet bucket with an unmoved mark needs nothing: no stale entry of the
/// key exists. A collision -- another key of the bucket in flight or landed
/// -- costs one corrective that declines. In aggregate that is not small
/// under a BACKLOG: with D entries in flight a bucket is busy with
/// probability about 1 - e^(-D/16384) (63% at D = 16k, 95% at 50k;
/// `InFlight`), so the fence then fires for most fresh sets -- each a
/// corrective that usually declines, counted in `PENDING_*` like any entry
/// (and in `RECONCILE_SET_NEW_KEY` when only the rule asked for it).
///
/// # The heal (review M2)
///
/// A hit served from the slow tier on a key the stack places FAST is a value
/// in CXL the stack counts fast: a promotion still pending, or a value moved
/// behind the stack's back. It is healed -- `(k, Fast)` appended -- only when
/// NOTHING OF ITS BUCKET IS IN FLIGHT. If something is, that decides: by the
/// tracked invariant the key's last in-flight entry names its placement, so
/// a promotion is already on its way; if the bucket is busy only with
/// another key, a later slow hit heals it once the bucket is quiet. A heal is
/// itself in flight until it finishes, so at most one heal per bucket is in
/// flight at a time, and the heals no longer scale with hit rate x consumer
/// lag (review: T7's burst queued 39 `get -> fast` on 147 keys, mostly
/// declined duplicates, each lengthening the queue that made the next hit
/// slow). A fast-served hit is not looked up at all.
///
/// The backlog that fences most fresh sets (above) turns the heal
/// effectively OFF until it drains: most slow-served hits then find their
/// bucket busy. Each is counted in `RECONCILE_GET_HEAL_SKIPPED` -- every
/// slow-served hit skipped for a busy bucket, so an upper bound on the heals
/// skipped: the placement is not read for them, and a hit on a key placed
/// slow, or on a key whose own promotion is what is in flight, needed none.
///
/// # Cost
///
/// Two atomic RMWs per queued entry (`InFlight`: the hand-off and the
/// finish) and one per inline landing. One load per `Set` on the client (the
/// mark, before the insert) and one on the worker (`moved_since`), plus a
/// `placement_of` probe before the stack's insert only for a `Set` whose map
/// insert replaced a value while its bucket moved. One load per slow-served
/// hit (and, when its bucket is busy, no `placement_of` -- one relaxed
/// increment of the skipped-heal count instead). Nothing per fast-served hit
/// or miss.
#[cfg(feature = "hybrid_cache_common")]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Observed {
	/// A `set` built the key's new value in `built`. `fence`: the new-key rule
	/// applies -- the stack did not place the key before this event, and a
	/// migration of its bucket was in flight or landed after the value was
	/// published.
	Built { key: HashedKey, built: Tier, fence: bool },
	/// A hit on the key was served from the slow tier. `quiet`: nothing of the
	/// key's bucket was in flight -- the heal rule.
	ServedSlow { key: HashedKey, quiet: bool },
	/// A `set` the byte gate DIVERTED (S5 B2, `OnStall::Divert`): its value was
	/// built slow because the fast tier was stalled, and the key is placed by
	/// its policy. Reconciled as a slow build, except never toward fast: a
	/// diverted key placed fast LAGS -- in CXL, charged to the fast budget --
	/// until the heal promotes it on its first slow-served hit, so an untouched
	/// one costs no copy; and a promotion the stack queued for it while
	/// handling this `Set` is dropped from the drain (`drain_reconciled`).
	Diverted { key: HashedKey, fence: bool },
}

#[cfg(feature = "hybrid_cache_common")]
impl Observed {
	fn key(self) -> HashedKey {
		match self {
			Observed::Built { key, .. } | Observed::ServedSlow { key, .. } | Observed::Diverted { key, .. } => key,
		}
	}
}

/// Why a corrective was queued, which is the counter it is counted in.
#[cfg(feature = "hybrid_cache_common")]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Reason {
	/// Where the set leaves the bytes -- the drain's last entry for the key,
	/// else the built tier -- is not the placement: `RECONCILE_SET_TO_*`.
	Built,
	/// The new-key rule alone: built where placed, queued to land last:
	/// `RECONCILE_SET_NEW_KEY`.
	NewKey,
	/// The heal: `RECONCILE_GET_TO_FAST`.
	Heal,
}

#[cfg(feature = "hybrid_cache_common")]
impl Reason {
	/// Counts one corrective queued for this reason, toward `tier`.
	fn count(self, tier: Tier) {
		use std::sync::atomic::Ordering::Relaxed;

		let counter = match (self, tier) {
			(Reason::Built, Tier::Fast) => &migstats::RECONCILE_SET_TO_FAST,
			(Reason::Built, Tier::Slow) => &migstats::RECONCILE_SET_TO_SLOW,
			(Reason::NewKey, _) => &migstats::RECONCILE_SET_NEW_KEY,
			(Reason::Heal, _) => &migstats::RECONCILE_GET_TO_FAST,
		};

		counter.fetch_add(1, Relaxed);
	}
}

/// The corrective migration for one observation, if it needs one: `drain` is
/// what the stack queued while handling the event, `placement` where it
/// places the key after it (`None`: it does not track the key, and nothing is
/// corrected).
///
/// This event leaves the bytes where the stack's LAST entry for the key in
/// `drain` sends them, or, if it queued none, where the client built them.
/// A `Built` observation is corrected toward the placement whenever that
/// differs, and -- the new-key rule, when `fence` is set -- also when it does
/// not, unless the drain's last entry for the key names the placement (that
/// entry is last already). A `ServedSlow` one is healed only when `quiet`,
/// the placement is FAST and nothing in the drain promotes the key already
/// -- the heal never demotes.
#[cfg(feature = "hybrid_cache_common")]
fn corrective<E: MigrationEntry>(
	drain: &[E],
	observed: Observed,
	placement: Option<Tier>,
) -> Option<(HashedKey, Tier, Reason)> {
	let placement = placement?;
	let key = observed.key();

	let queued = drain.iter().rev().find(|entry| entry.key() == key).map(|entry| entry.tier());

	match observed {
		Observed::Built { built, fence, .. } => {
			if queued.unwrap_or(built) != placement {
				Some((key, placement, Reason::Built))
			} else if fence && queued.is_none() {
				Some((key, placement, Reason::NewKey))
			} else {
				None
			}
		},

		Observed::ServedSlow { quiet, .. } => {
			(quiet && placement == Tier::Fast && queued.unwrap_or(Tier::Slow) == Tier::Slow)
				.then_some((key, Tier::Fast, Reason::Heal))
		},

		// Built slow on purpose (S5 B2): never corrected toward fast here -- the
		// heal does that on the key's first slow-served hit -- and toward slow
		// exactly as any slow build.
		Observed::Diverted { fence, .. } => match placement {
			Tier::Fast => None,
			Tier::Slow => corrective(drain, Observed::Built { key, built: Tier::Slow, fence }, Some(placement)),
		},
	}
}

const SET_RECENCY_DURATION: Duration = Duration::from_secs(5);
const SHORT_POLLING_DURATION: Duration = Duration::from_millis(1);
const LONG_POLLING_DURATION: Duration = Duration::from_secs(1);

pub struct PolicyWorker<K, V> {
	listener: Receiver<WorkerEvent>,

	objects: ObjectMapRef<K, V>,
	status: StatusRef,
	overhead_manager: OverheadManagerRef,

	policy_stack: Box<dyn PolicyStack>,

	/// Capacity-eviction watermarks for `apply_evictions`, snapshotted at
	/// construction (see `eviction_watermarks::Watermarks`). `1.0`/`1.0`
	/// unless `EVICTION_HIGH_WATERMARK`/`EVICTION_LOW_WATERMARK` say
	/// otherwise, which is the pre-watermark drain-to-exactly-`max_size`
	/// loop.
	eviction_watermarks: eviction_watermarks::Watermarks,

	last_set_time: Option<Instant>,

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

	/// The per-pass PHYS instrumentation's state: the over-budget integral and
	/// the MEMTS rate limit. See `instrument_pass`.
	#[cfg(feature = "hybrid_cache_common")]
	phys_pass: crate::phys::PassInstrument,

	/// S5a: what this worker keeps to publish M, the bytes the cache's own
	/// DRAM metadata structures hold -- the map's per-shard reading, one
	/// header's size, the stack's part as last published. See
	/// `publish_metadata`.
	#[cfg(feature = "hybrid_cache_common")]
	metadata: WorkerMetadata,

	/// S5: what this worker keeps to publish the gate's figures -- the
	/// high-water object count, the M last pushed into the stack, the model
	/// and the metadata cap last published, and the events handled. See
	/// `publish_gate`.
	#[cfg(feature = "hybrid_cache_common")]
	gate_pass: GatePass,

	/// What the event just handled observed of a key's bytes -- a `Set`'s
	/// built tier, a slow-served hit -- for the next `apply_tier_migrations`
	/// to reconcile (see `Observed`). The run loop drains after every event,
	/// so it holds at most one entry there; kept as a `Vec` so a test that
	/// handles several events before one drain loses none. Its capacity is
	/// reused: steady state allocates nothing.
	#[cfg(feature = "hybrid_cache_common")]
	observed: Vec<Observed>,

	/// Test builds flush the migration consumers after every batch, so a test
	/// sees a migration land when the call that queued it returns. A test
	/// that parks a consumer and keeps driving the worker turns this off.
	#[cfg(all(test, feature = "hybrid_cache_common"))]
	test_flush: bool,

	/// Test builds: when `Some`, every key `apply_evictions` evicts, in order
	/// -- T14 compares victims across stores.
	#[cfg(test)]
	evicted: Option<Vec<HashedKey>>,

	/// Test builds, merged store: how many eviction passes stopped over a
	/// REALLY empty store with `used_size` still over the cache's size -- the
	/// error `apply_evictions` logs (`log` has no logger in this crate's own
	/// binaries, so a test reads this instead).
	#[cfg(all(test, feature = "merged_object_store"))]
	nothing_left_to_evict: u32,

	/// Test builds: when `Some`, every drain `apply_tier_migrations` applies,
	/// reconcile included, in order -- T14 compares them across stores.
	#[cfg(all(test, feature = "hybrid_cache_common"))]
	drained: Option<Vec<TaggedMigration>>,
}

/// What the policy worker keeps to publish M (S5a; `publish_metadata`).
#[cfg(feature = "hybrid_cache_common")]
struct WorkerMetadata {
	/// The object map's worker-side reading (`crate::meta::MapState`): the
	/// DashMap's per-shard table sizes and headroom, nothing for the merged
	/// store, which counts itself.
	map: crate::meta::MapState,

	/// One DRAM value header's usable bytes, for this cache's key type and
	/// the build's layout (`value::dram_header_bytes`).
	header_bytes: u64,

	/// The stack's structures as last published (`PolicyStack::
	/// structure_bytes`, its box not included): an event that changes them
	/// republishes at once. Loads only, so it is compared after every event.
	structures: crate::meta::NodeBytes,

	/// When the whole map was last re-read.
	full_refresh: Instant,
}

/// What the policy worker keeps to publish the gate's figures (S5;
/// `publish_gate`).
#[cfg(feature = "hybrid_cache_common")]
#[derive(Default)]
struct GatePass {
	/// `L_hw`: the most objects the cache has held -- the key ceiling's reuse
	/// floor (`gate::key_ceiling`). Never lowered, not even by a wipe (the
	/// tables keep their capacity; the merged slab's chunks come back to at
	/// most what they were at `L_hw`), except: reset to the live count when
	/// the model changes, or when the metadata cap `F - floor` DECREASES (a
	/// refill re-creates per-object bytes the smaller cap may not hold).
	l_hw: u64,

	/// The M last pushed into the stack (`PolicyStack::set_dram_metadata`);
	/// `None` under the per-object model.
	pushed: Option<CacheSize>,

	/// The model and the metadata cap of the last publication.
	model: Option<crate::gate::MetadataModel>,
	c_meta: CacheSize,

	/// Events handled, for the gate's progress figure.
	events: u64,

	/// The measured M is outside 2x of `L * omega` (`metadata_model_divergence`
	/// counts the entries into it).
	diverged: bool,
}

/// How often `publish_metadata` re-reads the whole map regardless of its
/// write counts: the bound on how long a table growth the worker cannot count
/// (a `del` of an absent key) stays out of M. Every 256 DashMap shards'
/// `try_read` at most ten times a second.
#[cfg(feature = "hybrid_cache_common")]
const METADATA_FULL_REFRESH: Duration = Duration::from_millis(100);

#[cfg(feature = "hybrid_cache_common")]
impl WorkerMetadata {
	fn new<K>() -> Self {
		WorkerMetadata {
			map: Default::default(),
			header_bytes: crate::value::dram_header_bytes::<K>(),
			structures: crate::meta::NodeBytes::default(),
			full_refresh: Instant::now(),
		}
	}
}

/// What one step of `PolicyWorker::evict_victim` did.
enum Victim {
	/// A victim of the stack's order, removed from the map.
	Evicted,

	/// The stack named a key the map no longer holds (it is gone from the
	/// stack now); nothing was removed.
	Missed,

	/// Nothing to evict: the stack named none (and no fallback was asked for,
	/// or the merged store has nothing linked).
	Exhausted,
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

		// S5: when this function ends -- by returning or by unwinding -- a set
		// waiting on this worker (the metadata lane) sees it and returns
		// `CacheError::Internal` rather than waiting out its window.
		#[cfg(feature = "hybrid_cache_common")]
		let _gone = crate::gate::WorkerGoneGuard::new(&self.status);

		// Drained into and reused across iterations rather than re-collected
		// into a fresh `Vec` each pass. The collect is needed at all only
		// because `try_iter()` borrows `self.listener` while the loop body
		// needs `&mut self`; keeping one buffer alive means a steady-state
		// poll allocates nothing, instead of allocating (and, under bursty
		// load, repeatedly growing) a new one every millisecond.
		let mut events = Vec::<WorkerEvent>::new();

		loop {
			// Test builds: a gate test holding the workers.
			#[cfg(all(test, feature = "hybrid_cache_common"))]
			crate::gate::test_hooks::worker_hold();

			events.clear();
			events.extend(self.listener.try_iter());

			let mut has_current_set = false;

			for event in events.drain(..) {
				// Test builds: a gate test slowing the worker down.
				#[cfg(all(test, feature = "hybrid_cache_common"))]
				crate::gate::test_hooks::event_delay();

				match event {
					WorkerEvent::Get(key, served) => self.handle_get(key, served),

					WorkerEvent::Set(key, size, resident, _, previous, built, mark, placement) => {
						self.handle_set(key, size, resident, built, previous.map(|(size, _)| size), mark, placement);
						has_current_set = true;
					},

					WorkerEvent::Del(key, _) => self.handle_del(key),
					WorkerEvent::Expire(key) => self.handle_expire(key),

					WorkerEvent::Wipe(ack) => self.handle_wipe(ack.as_ref()),
					WorkerEvent::Resize(max_size) => self.handle_resize(max_size),
					WorkerEvent::ResizeFastTier(size) => self.handle_resize_fast_tier(size),
					WorkerEvent::ResizeLargeFastTier(size) => self.handle_resize_large_fast_tier(size),
					WorkerEvent::ResizeSizeThreshold(size) => self.handle_resize_size_threshold(size),

					WorkerEvent::Shutdown => {
						// Real totals, whatever this run's call volume was.
						migstats::dump_final();

						return Ok(());
					},

					#[cfg(feature = "hybrid_cache_common")]
					WorkerEvent::Audit(reply) => {
						// A requester that stopped waiting is no loss.
						let _ = reply.send(self.placement_audit());
					},

					#[cfg(feature = "hybrid_cache_common")]
					WorkerEvent::MakeRoom(request) => self.handle_make_room(request),

					_ => {},
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
				// hole: ONE event could emit both entries for one key (the
				// merged store's touch queued its promotion before the settle
				// that could demote the key again, until its settle moved to
				// this thread), and the drain after `apply_evictions` below
				// collects a whole eviction loop's decisions.
				// `split_tier_migrations` now closes it for a drain of any
				// width: the partition drops every entry that a later entry
				// for the other tier supersedes, so no key reaches both
				// halves.
				//
				// The reason it was expensive is gone regardless: the merged
				// store's drain used to take a WRITE lock on all 32 shards
				// whenever anything anywhere was pending. Its migrations are
				// now one log the stack owns, and a drain is a `mem::take`.
				#[cfg(feature = "hybrid_cache_common")]
				self.apply_tier_migrations();

				// S5a: an event that changed the stack's structures republishes
				// M now rather than at the end of a pass that may be long.
				#[cfg(feature = "hybrid_cache_common")]
				self.publish_metadata_if_the_stack_changed();

				// S5: a set waiting on this worker (the metadata lane) sees it
				// working through the events ahead of its request.
				#[cfg(feature = "hybrid_cache_common")]
				{
					self.gate_pass.events += 1;

					if self.gate_pass.events % 64 == 0 {
						self.status.gate().set_worker_progress(self.gate_pass.events);
					}
				}
			}

			self.apply_evictions()?;

			// INSTRUMENTATION: the invariant the failing run violated. Sampled on
			// the same cadence as MIGSTATS so the two can be correlated.
			{
				use std::sync::atomic::Ordering;
				static TICK: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
				if TICK.fetch_add(1, Ordering::Relaxed) % 4_096 == 0 {
					let stack_len = self.policy_stack.len();
					let map_len = self.status.live_num_objects() as usize;
					let s = &self.policy_stack;
					let (fo, so, fb, sb) = (
						s.fast_object_count(), s.slow_object_count(),
						s.fast_bytes_used(), s.slow_bytes_used(),
					);
					eprintln!(
						"DIVERGE map={map_len} stack={stack_len} delta={} fallback={} fast_obj={fo} slow_obj={so} fast_b={fb} slow_b={sb} t_ms={}",
						map_len as i64 - stack_len as i64,
						crate::ERASE_FALLBACK.load(Ordering::Relaxed),
						migstats::t_ms(),
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

			// S5a: M, after this pass's sets, migrations and evictions, and
			// before the MEMTS line that prints it.
			#[cfg(feature = "hybrid_cache_common")]
			self.publish_metadata(false);

			// S5: eff, the key ceiling and the stack's M from it -- and every
			// stack's settles re-run against them (`publish_gate`).
			#[cfg(feature = "hybrid_cache_common")]
			{
				self.publish_gate();
				self.status.gate().set_worker_progress(self.gate_pass.events);

				// A WHOLE pass ended -- this batch's events, the evictions, the
				// publication and its resettle -- what the byte gate's watchdog
				// counts (`gate::STALL_PASSES`); no other publication is one.
				self.status.gate().end_pass();
			}

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

			// After the gauges, so the MEMTS line and the integral see this
			// pass's migrations and evictions. See `instrument_pass`.
			#[cfg(feature = "hybrid_cache_common")]
			self.instrument_pass(now);

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
	// `Send + Sync`: the worker holds the object map on its own thread (and a
	// tiered cache's migration consumers share it); they hold for every real
	// instantiation. `Clone` is required because a tier migration now rebuilds the
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
	) -> Result<Self, CacheError> {
		// The first worker built in the process fixes the `t_ms` origin.
		migstats::mark_origin();

		let max_cache_size = status.max_size();

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

		let worker = PolicyWorker {
			listener,

			objects,
			status,
			overhead_manager,

			policy_stack,
			eviction_watermarks: eviction_watermarks::Watermarks::from_env(),

			last_set_time: None,

			#[cfg(feature = "hybrid_cache_common")]
			#[cfg(feature = "hybrid_cache_common")]
			tier_migration: false,

			#[cfg(feature = "hybrid_cache_common")]
			migration_queue: None,

			#[cfg(feature = "hybrid_cache_common")]
			phys_pass: crate::phys::PassInstrument::new(Instant::now()),

			#[cfg(feature = "hybrid_cache_common")]
			metadata: WorkerMetadata::new::<K>(),

			#[cfg(feature = "hybrid_cache_common")]
			gate_pass: GatePass::default(),

			#[cfg(feature = "hybrid_cache_common")]
			observed: Vec::new(),

			#[cfg(all(test, feature = "hybrid_cache_common"))]
			test_flush: true,

			#[cfg(test)]
			evicted: None,

			#[cfg(all(test, feature = "merged_object_store"))]
			nothing_left_to_evict: 0,

			#[cfg(all(test, feature = "hybrid_cache_common"))]
			drained: None,
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
		// The first worker built in the process fixes the `t_ms` origin.
		migstats::mark_origin();

		let max_cache_size = status.max_size();

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

			policy_stack,
			eviction_watermarks: eviction_watermarks::Watermarks::from_env(),

			last_set_time: None,

			tier_migration: true,

			#[cfg(feature = "hybrid_cache_common")]
			migration_queue,

			#[cfg(feature = "hybrid_cache_common")]
			phys_pass: crate::phys::PassInstrument::new(Instant::now()),

			#[cfg(feature = "hybrid_cache_common")]
			metadata: WorkerMetadata::new::<K>(),

			#[cfg(feature = "hybrid_cache_common")]
			gate_pass: GatePass::default(),

			#[cfg(feature = "hybrid_cache_common")]
			observed: Vec::new(),

			#[cfg(all(test, feature = "hybrid_cache_common"))]
			test_flush: true,

			#[cfg(test)]
			evicted: None,

			#[cfg(all(test, feature = "merged_object_store"))]
			nothing_left_to_evict: 0,

			#[cfg(all(test, feature = "hybrid_cache_common"))]
			drained: None,
		};

		// M from the start: an empty map's tables and arrays, an empty stack,
		// no headers -- before the first `Set` can move it. And the gate's
		// figures from it (S5), before the constructor returns the cache.
		let mut worker = worker;
		worker.publish_metadata(true);
		worker.publish_gate();

		Ok(worker)
	}

	/// `served` is the tier a hit's value was served from, `None` on a miss.
	fn handle_get(&mut self, key: HashedKey, served: Option<Tier>) {
		self.policy_stack.record_access(key, served.is_some());

		// An LFU slow hit settles, and a settle can latch.
		#[cfg(feature = "hybrid_cache_common")]
		self.publish_admission_latch();

		// The heal (see `Observed`): only a hit served from the SLOW tier is
		// checked against the stack's placement, and only while nothing of its
		// bucket is in flight -- one load. A fast-served hit and a miss cost
		// this one comparison and nothing else.
		#[cfg(feature = "hybrid_cache_common")]
		if self.tier_migration && served == Some(Tier::Slow) {
			let quiet = self.status.migration_in_flight().pending(key) == 0;

			self.observed.push(Observed::ServedSlow { key, quiet });
		}
	}

	/// `dram_resident` is the part of `size` that never migrates; the policy
	/// stack needs it to keep `fast_used` / `slow_used` to migrating bytes.
	///
	/// `previous` is the base size of the value the map insert replaced
	/// (`None`: it replaced nothing). It tells the stack what the insert did
	/// (`SetEvent`: `Fresh`, or `Replaced` and whether the size changed) --
	/// which only the merged store needs, its index being the map the client
	/// already wrote (`MergedStore::worker_set`) -- and it is the new-key
	/// rule's `fresh`.
	///
	/// `built` is the tier the value's bytes were allocated in, which the
	/// next `apply_tier_migrations` reconciles against the stack's placement
	/// (see `Observed`) -- on every set, since the client's choice can be
	/// stale whatever the design.
	///
	/// `fresh` and `mark` (the landed count of the key's migration bucket the
	/// client read before publishing) are the new-key rule's inputs: the rule
	/// applies when the stack did not place the key BEFORE this event and the
	/// bucket moved since the mark (`Observed`). So it is decided here, before
	/// the stack handles the set. "Did not place" is `fresh`, or -- for a map
	/// insert that replaced a value -- the stack's own `placement_of` being
	/// `None`: a `del` on one thread racing a `set` on another can reach the
	/// worker between two sets of the key and untrack it while a value is
	/// live, and the second set then admits the key without a push although
	/// it replaced a value. That probe is taken only when the bucket moved. It
	/// is exact for both stores: the merged store's `placement_of` is `None`
	/// until the worker links a value, as a DashMap stack's is until it
	/// inserts the key. A `fresh` set the stack still tracks (a re-set that
	/// reached the worker before the `Del` or reap it follows, or -- merged --
	/// one an earlier `Set` of the key linked on its behalf) is treated as
	/// new: one declined corrective at most.
	///
	/// The LFU latch this handling may have moved is published at once
	/// (`publish_admission_latch`).
	///
	/// `placement` is where the client placed the value (S5): `Structural`
	/// when it was larger than an empty fast tier and built slow for that;
	/// the stack places such a key slow (`PolicyStack::insert_placed`). A
	/// `Normal` set the stack's own check made structural -- eff moved between
	/// the client's decision and this -- is counted.
	fn handle_set(
		&mut self,
		key: HashedKey,
		size: ObjectSize,
		dram_resident: ObjectSize,
		built: Tier,
		previous: Option<ObjectSize>,
		mark: u32,
		placement: Placement,
	) {
		// The value this `Set` published is gone: an eviction pass (or a
		// MakeRoom) took it between the client's map insert and this event, or a
		// delete or reap on another thread did. The client inserts before it
		// broadcasts, so an absent key means removed since, and any later event
		// for it is behind this one. Admitting it would track an object the map
		// no longer holds, which nothing would ever remove (T9's 2Q full
		// fast-admission "never quiesced": stack 44 against map 43). The merged
		// store needs no check: it never evicts a value the worker has not
		// linked.
		#[cfg(not(feature = "merged_object_store"))]
		if !self.object_exists(key) {
			// S5a: the client's insert may still have grown the map's table.
			#[cfg(feature = "hybrid_cache_common")]
			if self.tier_migration {
				crate::meta::map_write(&self.objects, &mut self.metadata.map, key);
			}

			return;
		}

		let fresh = previous.is_none();

		#[cfg(feature = "hybrid_cache_common")]
		let fence = self.tier_migration
			&& self.status.migration_in_flight().moved_since(key, mark)
			&& (fresh
				|| self.policy_stack.placement_of(key).is_none());

		let event = match previous {
			None => SetEvent::Fresh,
			Some(previous) => SetEvent::Replaced { resized: previous != size },
		};

		let applied = self.policy_stack.insert_placed(key, size, dram_resident, event, placement);

		#[cfg(feature = "hybrid_cache_common")]
		if applied == Placement::Structural && placement != Placement::Structural {
			self.status.gate().count_structural_placement();
		}

		#[cfg(not(feature = "hybrid_cache_common"))]
		let _ = applied;

		#[cfg(feature = "hybrid_cache_common")]
		self.publish_admission_latch();

		#[cfg(feature = "hybrid_cache_common")]
		if self.tier_migration {
			self.observed.push(match placement {
				Placement::Diverted => Observed::Diverted { key, fence },
				_ => Observed::Built { key, built, fence },
			});

			// S5a: this set's insert may have grown the map's table (any
			// insert can, at the load limit -- see `crate::meta::ShardState`).
			crate::meta::map_write(&self.objects, &mut self.metadata.map, key);
		}

		#[cfg(not(feature = "hybrid_cache_common"))]
		let _ = (built, fresh, mark);
	}

	fn handle_del(&mut self, key: HashedKey) {
		self.policy_stack.remove(key);

		// S5a: the delete looked the key up with `entry`, which reserves a
		// slot in its DashMap shard first (`crate::meta::ShardState`).
		#[cfg(feature = "hybrid_cache_common")]
		if self.tier_migration {
			crate::meta::map_write(&self.objects, &mut self.metadata.map, key);
		}
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
	///
	/// The merged store's handle is the exception (`remove_is_retire`): its
	/// `remove` retires a DEAD slot of the key -- the one the reap left on the
	/// list -- and never the live one, so it is called whatever the map holds.
	/// Guarded, it would leave that DEAD slot behind whenever the key was set
	/// again before this event.
	fn handle_expire(&mut self, key: HashedKey) {
		let live = self.object_exists(key);

		// Live: re-set between the reap and this notification, or live all
		// along and left in place by the reap (`EraseKey::Expired`). Either
		// way the entry belongs to that live object.
		if !live || self.policy_stack.remove_is_retire() {
			self.policy_stack.remove(key);
		}

		// S5a: the reap looked the key up with `entry`, as a delete does.
		#[cfg(feature = "hybrid_cache_common")]
		if self.tier_migration {
			crate::meta::map_write(&self.objects, &mut self.metadata.map, key);
		}
	}

	/// Whether the object map still holds `key`.
	///
	/// Two bodies because the `object_store` module -- and so the
	/// `ObjectStore` trait that supplies `get_ref` -- is itself gated on a
	/// storage feature being selected (`lib.rs`). A build that selects none
	/// (bare `eviction_stacks_pmem`, say) still resolves `ObjectMapRef` to the
	/// default `DashMap` shape, which answers this directly.
	#[cfg(any(feature = "all_dram", feature = "key_value_pmem"))]
	fn object_exists(&self, key: HashedKey) -> bool {
		self.objects.get_ref(&key).is_some()
	}

	#[cfg(not(any(feature = "all_dram", feature = "key_value_pmem")))]
	fn object_exists(&self, key: HashedKey) -> bool {
		self.objects.contains_key(&key)
	}

	fn handle_resize(&mut self, size: CacheSize) {
		self.policy_stack.resize(size);
	}

	/// Runtime-adjusts the fast-tier byte budget. Honoured by every hybrid
	/// design's stack; a no-op for every non-hybrid one. May itself trigger
	/// demotions, drained by `apply_tier_migrations` on the next pass through
	/// the event loop.
	fn handle_resize_fast_tier(&mut self, size: CacheSize) {
		self.policy_stack.resize_fast_tier(size);

		// A grow unlatches LFU admission; a shrink's settle can latch it.
		#[cfg(feature = "hybrid_cache_common")]
		self.publish_admission_latch();
	}

	/// Runtime-adjusts the LARGE fast segment's byte budget
	/// (`lru_sized_compact_hybrid_cache` specifically). No-op for every other
	/// policy stack. May itself trigger demotions, drained by
	/// `apply_tier_migrations` on the next pass through the event loop.
	fn handle_resize_large_fast_tier(&mut self, size: CacheSize) {
		self.policy_stack.resize_large_fast_tier(size);
	}

	/// Runtime-adjusts the small/large size-classification threshold
	/// (`lru_sized_compact_hybrid_cache`). No-op for every other policy stack.
	fn handle_resize_size_threshold(&mut self, size: CacheSize) {
		self.policy_stack.resize_size_threshold(size);
	}

	/// Empties the cache, on this thread, and then answers `ack` -- which
	/// `PaperCache::wipe` waits on. The object map and, at once, the status
	/// (`AtomicStatus::clear`: what the map's clear removed is SUBTRACTED
	/// from the object count and the base size, so a client's insert racing
	/// the clear stays counted exactly -- removed and taken off, or live and
	/// kept -- and every counter reset, the LFU latch mirror among them);
	/// then the stack and, on a tiered cache, the tier gauges
	/// and the latch, republished from the empty stack: so when `wipe`
	/// returns, `hybrid_stats` already reads an empty cache, and a `Set` this
	/// worker handled before the `Wipe` cannot have left a live key its stack
	/// no longer tracks (the map is cleared with it). Under the merged store
	/// the map IS the stack, and its worker-owned state -- the link count, the
	/// latch, the DEAD slots -- has one writer.
	///
	/// What can still diverge is a value published before this clear whose
	/// `Set` is behind the `Wipe` in the channel: cleared with the map here,
	/// its `Set` then finds nothing in the merged store; a DashMap stack
	/// inserts it anyway, a ghost entry evicted later as `KeyNotFound`.
	fn handle_wipe(&mut self, ack: Option<&Sender<()>>) {
		let overhead_manager = &self.overhead_manager;
		let cleared = self.objects.clear_counted(|object| overhead_manager.base_size(object));

		self.status.clear(cleared);

		self.policy_stack.clear();

		#[cfg(feature = "hybrid_cache_common")]
		{
			self.observed.clear();
			self.refresh_tier_gauges();

			// S5a: every table kept its capacity (DashMap, the
			// merged buckets), the merged slab freed its chunks, the headers
			// went with their values. Re-read all of it.
			self.publish_metadata(true);

			// S5: and the gate's figures from it: L = 0 opens the key ceiling.
			self.publish_gate();

			// B2: every waiter re-checks against the emptied tier now, not at
			// its next poll.
			self.status.gate().wake_waiters();
		}

		// A client that stopped waiting is no loss.
		if let Some(ack) = ack {
			let _ = ack.send(());
		}
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
	///
	/// The drain carries the reconcile's correctives for whatever the event
	/// just handled observed (`drain_reconciled`).
	#[cfg(feature = "hybrid_cache_common")]
	fn apply_tier_migrations(&mut self) {
		let (inline_demotion_accounting, migrations) = self.drain_reconciled();

		#[cfg(test)]
		if let Some(drained) = &mut self.drained {
			drained.extend_from_slice(&migrations);
		}

		// Handed over whole: `apply_migration_batches` does the split itself
		// (`split_tier_migrations`), and taking the drain unsplit is what
		// makes that the only way in.
		if !migrations.is_empty() {
			self.apply_migration_batches(migrations, inline_demotion_accounting);
		}

		let drained_demotions = self.policy_stack.drain_demotions();

		if drained_demotions > 0 {
			self.status.record_hybrid_demotions(drained_demotions);
		}
	}

	/// The stack's drain, tagged (`drain_tagged_migrations`), with the
	/// reconcile's correctives for this event's observations APPENDED and
	/// tagged `MigrationOrigin::Reconcile` (see `Observed` for the whole
	/// argument), and whether completed slow moves count as demotions for
	/// this stack.
	///
	/// The drain is taken FIRST and the placement read after it, so the
	/// placement is at least as new as every intent the drain carries: read
	/// the other way round a corrective could be computed from a placement
	/// OLDER than an entry it is then appended behind, and win. (Every stack
	/// now decides only on this thread, so nothing can push between the two;
	/// the order costs nothing and keeps the argument local.) A corrective is
	/// appended exactly when this event leaves the bytes somewhere other than
	/// the placement: the stack's own last entry for the key in this drain if
	/// it queued one, else where the client observed them (`corrective`). So a
	/// stack that already queued the move -- LRU's re-promotion on a re-set,
	/// LFU's admission to slow before it latches -- is not sent a duplicate.
	/// The new-key rule appends one even when the bytes were built at the
	/// placement, unless the drain's last entry for the key names it.
	///
	/// Cost, per `Set`: one `placement_of` -- a probe of the stack's index;
	/// on the merged store a shard READ lock and a probe -- plus a reverse scan
	/// of this event's drain, which is empty in the common case and never
	/// longer than the work the stack just did. The same per hit served from
	/// the slow tier; nothing per fast-served hit or miss (`handle_get`). It
	/// is a second lookup rather than a tier the set's handling hands back:
	/// `insert_resident` returns nothing, and no stack holds its final
	/// placement at its end in one place -- the settle after an insert can
	/// move the key itself -- so handing it back would mean a new return
	/// value through every stack, the flat ones included, to save one probe.
	#[cfg(feature = "hybrid_cache_common")]
	fn drain_reconciled(&mut self) -> (bool, Vec<TaggedMigration>) {
		let stack = &mut self.policy_stack;

		let mut migrations = stack.drain_tagged_migrations();

		// A diverted set's value stays in CXL until its first slow-served hit
		// (S5 B2): a promotion the stack queued for the key while handling its
		// `Set` -- an overwrite's re-promotion, say -- is dropped, as no
		// corrective toward fast is queued for it. What the stack places is
		// unchanged: the key lags, and the audit counts it.
		for observed in &self.observed {
			if let Observed::Diverted { key, .. } = *observed {
				migrations.retain(|&(k, tier, origin)| !(k == key && tier == Tier::Fast && origin == MigrationOrigin::Stack));
			}
		}

		for observed in self.observed.drain(..) {
			// A slow hit whose bucket is busy is not healed, wherever it is
			// placed: no probe. Counted as a skipped heal -- an upper bound,
			// the placement not being read.
			if let Observed::ServedSlow { quiet: false, .. } = observed {
				migstats::RECONCILE_GET_HEAL_SKIPPED.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
				continue;
			}

			let placement = stack.placement_of(observed.key());

			if let Some((key, tier, reason)) = corrective(&migrations, observed, placement) {
				reason.count(tier);
				migrations.push((key, tier, MigrationOrigin::Reconcile));
			}
		}

		(stack.inline_demotion_accounting(), migrations)
	}

	/// DIAGNOSTIC -- `WorkerEvent::Audit`, i.e. `PaperCache::placement_audit`:
	/// every live value's bytes against where the stack places its key.
	///
	/// Lands everything already decided first: the stack's pending drain goes
	/// out, reconcile included, and the consumers are flushed. A value that is
	/// merely waiting on a queued migration is therefore not reported, and
	/// what is reported is misplacement nothing in flight will fix. Then one
	/// walk of the object map with one `placement_of` per value
	/// (`for_each_value`: under each shard's read lock, except the merged
	/// store, which buffers a shard and classifies after releasing it --
	/// `placement_of` there takes that lock). The worker handles nothing else
	/// meanwhile: no event, no eviction, and the flush waits out the
	/// consumers' whole backlog. It runs where the `Audit` event falls in its
	/// batch, BEFORE the eviction pass that ends the batch, so a cache over
	/// its size is walked with the values that pass will evict; and a
	/// client's `Set` still behind it in the channel is untracked -- exact
	/// only at client quiescence.
	#[cfg(feature = "hybrid_cache_common")]
	fn placement_audit(&mut self) -> crate::phys::PlacementAudit {
		self.apply_tier_migrations();

		if let Some(queue) = &self.migration_queue {
			queue.flush();
		}

		let mut audit = crate::phys::PlacementAudit::default();
		let stack = &*self.policy_stack;

		self.objects.for_each_value(|key, tier, len| {
			audit.record(
				tier,
				stack.placement_of(key),
				crate::phys::value_charge::<K>(len),
			);
		});

		audit
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
	/// synchronous path counts the completions itself.
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
	///
	/// Each entry keeps its origin (`MigrationOrigin`) through the split and
	/// the queue. A reconcile-origin entry -- a corrective -- is not counted
	/// in `DEMO`/`PROMO` but in `RECONCILE_QUEUED_TO_*`, and its completion
	/// not as a promotion or a demotion but in `reconcile_applied_*`. Every
	/// landing, inline or on a consumer, is counted in the key's `InFlight`
	/// bucket, for the new-key rule.
	#[cfg(feature = "hybrid_cache_common")]
	fn apply_migration_batches<E: MigrationEntry>(
		&self,
		migrations: Vec<E>,
		inline_demotion_accounting: bool,
	) {
		use std::sync::atomic::{AtomicU64, Ordering::Relaxed};

		if !self.tier_migration {
			return;
		}

		let (demotions, promotions, coalesced) = split_tier_migrations(migrations);

		if coalesced > 0 {
			migstats::COALESCED_TOT.fetch_add(coalesced as u64, Relaxed);
		}

		migration_queue::BURST_MAX.fetch_max((demotions.len() + promotions.len()) as u64, Relaxed);

		// The stacks' decisions and the correctives, counted apart.
		let is_corrective = |entry: &&E| entry.origin() == MigrationOrigin::Reconcile;
		let corrective_demotions = demotions.iter().filter(is_corrective).count();
		let corrective_promotions = promotions.iter().filter(is_corrective).count();

		migstats::rec(&migstats::DEMO, &migstats::DEMO_TOT, demotions.len() - corrective_demotions);
		migstats::rec(&migstats::PROMO, &migstats::PROMO_TOT, promotions.len() - corrective_promotions);
		migstats::reconcile_queued(corrective_promotions, corrective_demotions);
		migstats::tick();

		let objects = &self.objects;
		let status = &self.status;
		let migration_queue = self.migration_queue.as_ref();

		// The consumers cannot see the policy stack, so mirror the one bit of
		// it they need before handing them anything to count.
		if let Some(queue) = migration_queue {
			queue.set_demotion_accounting(inline_demotion_accounting);
		}

		// Correctives that landed inline, by destination (fast, slow): counted
		// apart from the stack's completions, which the two filters below count.
		let inline_correctives = [AtomicU64::new(0), AtomicU64::new(0)];

		// Build the destination buffer with NO object-map guard held -- see
		// the pre-unification history for the full latency reasoning;
		// `Object::data()` is an `Arc` refcount bump and keeps the source
		// bytes alive unlocked.
		let apply_physical = |entry: E| -> bool {
			if let Some(queue) = migration_queue {
				queue.push(entry);

				// Handed off, not applied: nothing has moved yet, and the
				// consumer that eventually moves it does the counting.
				return false;
			}

			let (key, tier, origin) = entry.tagged();

			if !migration_queue::apply_migration(objects, key, tier) {
				return false;
			}

			// Inline, nothing was pending, but the landing is the new-key
			// rule's business all the same (`Observed`).
			status.migration_in_flight().landed_inline(key);

			match origin {
				MigrationOrigin::Stack => true,

				MigrationOrigin::Reconcile => {
					inline_correctives[(tier == Tier::Slow) as usize].fetch_add(1, Relaxed);
					false
				},
			}
		};

		let completed_demotions = demotions.into_iter().filter(|&entry| apply_physical(entry)).count() as u64;

		let completed_promotions = promotions.into_iter().filter(|&entry| apply_physical(entry)).count() as u64;

		// At most one atomic per direction per pass (and none at all when the
		// count is zero, which with the queue on is every pass), where the old
		// code paid one per entry regardless of outcome.
		if inline_demotion_accounting {
			status.record_hybrid_demotions(completed_demotions);
		}

		status.record_hybrid_promotions(completed_promotions);

		let [to_fast, to_slow] = inline_correctives.map(AtomicU64::into_inner);

		if (to_fast | to_slow) != 0 {
			status.record_reconcile_applied(to_fast, to_slow);
		}

		// Tests assert on tier residency and on the counters right after
		// triggering a migration; the standing consumer pool applies
		// asynchronously, so drain it before returning -- unless the test
		// parks a consumer and drives on (`test_flush`).
		#[cfg(test)]
		if let (Some(queue), true) = (migration_queue, self.test_flush && !crate::gate::test_hooks::flush_off()) {
			queue.flush();
		}
	}

	/// Publishes the stack's LFU admission latch into `status`, where
	/// `hybrid_policy::admission_tier` reads it to decide which tier a client
	/// builds a NEW key in -- right after every stack call that can move it
	/// (`handle_set`, `handle_get`, `handle_resize_fast_tier`, `handle_wipe`,
	/// and once a pass through `refresh_tier_gauges`), so it trails the stack
	/// by the event backlog alone, not by a pass. This worker is its only
	/// writer. A key a client built before the worker reached the `Set` that
	/// latched was built fast and is placed slow: the reconcile of its own
	/// `Set` queues the corrective, as for any stale build.
	///
	/// One load of the cell, and a store only when it differs -- compared
	/// with the cell itself, not a copy kept here, so a status cleared
	/// behind the worker's back is republished.
	#[cfg(feature = "hybrid_cache_common")]
	fn publish_admission_latch(&self) {
		let stack = &self.policy_stack;
		let latched = stack.admission_latched();

		if self.status.hybrid_admission_latched() != latched {
			self.status.set_hybrid_admission_latched(latched);
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
		self.publish_admission_latch();

		let stack = &self.policy_stack;
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

	/// The physical fast tier, once per pass, on a TIERED cache's worker (a
	/// flat cache has no fast tier to measure): samples P into its peak, adds
	/// this interval's `max(0, P + L * omega - F) * dt` to the over-budget
	/// integral and publishes it, and -- with `PAPER_MEMTS` set -- prints a
	/// MEMTS line at most every 250 ms. Reporting only: nothing here feeds a
	/// decision (the gate that will is S5).
	///
	/// Per pass that is 17 loads for P (one `Acquire`, 16 relaxed), one
	/// `fetch_max`, a few status loads and a multiply; the MEMTS line (a
	/// `/proc/self/status` read and one `eprintln!`) only when enabled and
	/// due.
	#[cfg(feature = "hybrid_cache_common")]
	fn instrument_pass(&mut self, now: Instant) {
		if !self.tier_migration {
			return;
		}

		let phys = crate::phys::observe();

		// S5: the integrand is `P + M_model - F`, M_model being the metadata
		// figure eff takes off F (the measured M, or the per-object model's) --
		// passed as one "object" of `M_model` bytes.
		let over_budget_byte_seconds = self.phys_pass.pass(
			now,
			phys,
			1,
			self.status.gate().m_model(),
			self.status.whole_fast_tier_capacity(),
		);

		self.status.set_hybrid_over_budget_byte_seconds(over_budget_byte_seconds);

		if !self.phys_pass.memts_due(now) {
			return;
		}

		let stats = self.status.hybrid_stats();
		let (pending_demote, pending_promote) = migration_queue::pending();
		let gate = self.status.gate();
		let bands = gate.bands();

		let sample = crate::phys::MemtsSample {
			t_ms: migstats::t_ms(),
			wall_ms: crate::phys::wall_ms(),
			phys,
			eff: self.status.effective_fast_capacity(),
			fast_used: stats.fast_bytes_used,
			fast_metadata_bytes: stats.fast_metadata_bytes,
			over_budget_byte_seconds,
			pending_net: pending_demote as i64 - pending_promote as i64,
			backlog: self.listener.len(),
			live_tiered_caches: crate::phys::live_tiered_caches(),
			vmrss_kb: crate::phys::vmrss_kb(),
			live_flat_fast_caches: crate::phys::live_flat_fast_caches(),
			meta: self.status.dram_metadata_bytes(),
			s: bands.s,
			n: bands.n,
			b: bands.b,
			waiters: gate.bytes_lane.len() as u64,
			reserved: gate.reserved(),
		};

		eprintln!("{}", crate::phys::format_memts(&sample));
	}

	/// S5a: publishes M, the bytes the cache's own DRAM metadata structures
	/// hold (`crate::meta`), into the status, on a TIERED cache's worker (a
	/// flat cache has no fast tier to budget):
	///
	///   * the map: `crate::meta::map_bytes` -- the merged store's own count,
	///     or the DashMap shards re-read where
	///     the writes counted since their last read (`Set`, `Del` and `Expire`
	///     events, the worker's own evictions) could have made them
	///     reallocate, every one when `all`;
	///   * the stack: its `structure_bytes`, and the box it lives in;
	///   * the headers: live objects times one header's usable size.
	///
	/// Called at construction and after a wipe with `all`, at the end of every
	/// pass, and after any event that changed the stack's structures. The live
	/// count is the status', which a client's insert moves at once; the map's
	/// part trails it by up to a pass, as a DashMap shard's growth is seen at
	/// the end of the pass that handled the `Set` that grew it.
	///
	/// And every `METADATA_FULL_REFRESH` the whole map is re-read whatever the
	/// counts say, for the writes the worker cannot count: a `del` of an
	/// absent key, whose `entry` lookup reserves in its shard and sends no
	/// event (`crate::meta::ShardState`), and the eviction fallback that
	/// erases an arbitrary map entry when the stack names no victim.
	#[cfg(feature = "hybrid_cache_common")]
	fn publish_metadata(&mut self, all: bool) {
		if !self.tier_migration {
			return;
		}

		let now = Instant::now();
		let all = all || now.saturating_duration_since(self.metadata.full_refresh) >= METADATA_FULL_REFRESH;

		if all {
			self.metadata.full_refresh = now;
		}

		let map = crate::meta::map_bytes(&self.objects, &mut self.metadata.map, all);
		let structures = self.stack_structures();

		// The box the stack lives in: DRAM, whatever node its structures are on.
		let boxed = crate::meta::box_bytes_of_val(&*self.policy_stack);
		let headers = self.status.live_num_objects().saturating_mul(self.metadata.header_bytes);

		self.metadata.structures = structures;

		self.status.set_dram_metadata(crate::meta::DramMetadata {
			map: map.dram,
			stack: structures.dram + boxed,
			headers,
			slow: map.slow + structures.slow,
		});
	}

	/// S5: the gate's publication, once per pass (and at construction, after a
	/// wipe and around a `MakeRoom`), on a TIERED cache's worker:
	///
	///   1. `M_model`: the measured M (`AtomicStatus::dram_metadata_bytes`,
	///      just published) or, per-object, the stack's own reservation
	///      (`dram_reserved_bytes`: `len x omega`, plus a ghost's DRAM);
	///   2. under the measured model the stack is given the same M
	///      (`set_dram_metadata`), so its settles reserve what eff takes off;
	///   3. eff = `F - M_model` (the size-split design's two class figures
	///      beside it, split as its stack splits them), `L_hw`, and the key
	///      ceiling (`gate::key_ceiling`), published with the `META_NEAR` flag
	///      (`Gate::publish`);
	///   4. the model's sanity check: M against `L * omega`;
	///   5. `resettle`: every settle of the stack against the budget just
	///      published, and what it queues applied at once -- so after every
	///      pass each design rests at or under its drain target, whatever its
	///      new-key path does (the LFU latch, the slow admission queues).
	///
	/// B2, the byte gate: its state (`byte_gate_state`) and, while it is
	/// enabled, its levels (`gate::bands`) go out with eff; after the resettle
	/// the gate's pass (`Gate::worker_pass`): the fold hook installed or
	/// cleared, waiters released when the gate turns off, a stall ended once
	/// anything was freed, the head notified. The watchdog's whole-pass count
	/// is `Gate::end_pass`, which the run loop calls after this; a MakeRoom's,
	/// a wipe's or the constructor's publication is not a pass.
	#[cfg(feature = "hybrid_cache_common")]
	fn publish_gate(&mut self) {
		use crate::gate::{self, GateState, MetadataModel};

		if !self.tier_migration {
			return;
		}

		#[cfg(test)]
		if self.status.gate().test_panic_on_pass.load(std::sync::atomic::Ordering::Relaxed) {
			panic!("test: the policy worker dies at a gate pass");
		}

		// The live counts are read after the epoch, so a registration after
		// this reading moves the epoch past what the gate records.
		let epoch = crate::phys::gate_epoch();
		let state = self.byte_gate_state();

		let gate = self.status.gate();
		let config = gate.config();
		let model = gate.model();
		let floor = config.metadata_floor;
		let status = &self.status;
		let pass = &mut self.gate_pass;

		let stack = &mut self.policy_stack;

		let fast_before = (stack.fast_bytes_used(), stack.fast_object_count());
		let l_pub = status.live_num_objects();

		if pass.model != Some(model) {
			if model == MetadataModel::PerObject && pass.pushed.take().is_some() {
				stack.set_dram_metadata(None);
			}

			pass.model = Some(model);
			pass.l_hw = l_pub;
		}

		let m_model = match model {
			MetadataModel::Measured => status.dram_metadata_bytes(),
			MetadataModel::PerObject => stack.dram_reserved_bytes(),
		};

		// Test builds: M held where a gate test put it (a table step).
		#[cfg(test)]
		let m_model = gate::test_hooks::m_override().unwrap_or(m_model) + gate::test_hooks::m_offset();

		if model == MetadataModel::Measured && pass.pushed != Some(m_model) {
			stack.set_dram_metadata(Some(m_model));
			pass.pushed = Some(m_model);
		}

		let whole = status.whole_fast_tier_capacity();
		let eff = whole.saturating_sub(m_model);

		let (eff_small, eff_large) = match status.policy() {
			PaperPolicy::LruSizedCompactHybrid => {
				let small = status.fast_tier_capacity();
				let large = status.hybrid_large_fast_capacity();
				let (small_share, large_share) = gate::size_split_shares(m_model, small, large);

				(small.saturating_sub(small_share), large.saturating_sub(large_share))
			},

			_ => (eff, eff),
		};

		let c_meta = whole.saturating_sub(floor);

		if c_meta < pass.c_meta {
			pass.l_hw = l_pub;
		}

		pass.c_meta = c_meta;
		pass.l_hw = pass.l_hw.max(l_pub);

		let omega = status.hybrid_shared_overhead();
		let k_max = gate::key_ceiling(model, omega, c_meta, m_model, stack.len() as u64, l_pub, pass.l_hw);

		let bands = (state == GateState::Enabled).then(|| gate::bands(eff, &config));

		gate.publish(
			gate::Published { model, m_model, eff, eff_small, eff_large, k_max, bands },
			|| status.live_num_objects(),
			crate::phys::fast_bytes_approx,
		);

		// The model's sanity check: the measured M against `L * omega`,
		// counted when it leaves 2x either way at a population where the
		// per-object figure means something.
		let per_object = l_pub.saturating_mul(omega);
		let measured = status.dram_metadata_bytes();
		let diverged = l_pub > 10_000
			&& omega > 0
			&& (measured > per_object.saturating_mul(2) || measured.saturating_mul(2) < per_object);

		if diverged && !pass.diverged {
			gate.count_divergence();
		}

		pass.diverged = diverged;

		stack.resettle();

		let fast_after = (stack.fast_bytes_used(), stack.fast_object_count());

		self.apply_tier_migrations();

		if fast_after != fast_before {
			self.refresh_tier_gauges();
		}

		// B2: after the resettle queued its demotions.
		self.status.gate().worker_pass(state, epoch);
	}

	/// The byte gate's state for this cache now (S5 B2, design 3.9.8): `Off`
	/// by its configuration; `Bands` when the settle target could not be below
	/// the near level; `Ungated` for the designs whose settles do not bound
	/// their DRAM -- the faithful S3-FIFO fast-admission pair, whose small
	/// queue is not clamped to the tier (Q7); `NotSole` while the cache
	/// is not P's only user; `Enabled` otherwise.
	#[cfg(feature = "hybrid_cache_common")]
	fn byte_gate_state(&self) -> crate::gate::GateState {
		use crate::gate::{GateMode, GateState};

		let config = self.status.gate().config();

		if config.mode == GateMode::Off {
			return GateState::Off;
		}

		if !config.bands_hold() {
			return GateState::Bands;
		}

		if matches!(
			self.status.policy(),
			PaperPolicy::S3FifoFaithfulFastAdmissionCompactHybrid(..)
				| PaperPolicy::S3FifoFaithfulFastAdmissionReprieveCompactHybrid(..)
		) {
			return GateState::Ungated;
		}

		match crate::phys::sole_fast_user() {
			true => GateState::Enabled,
			false => GateState::NotSole,
		}
	}

	/// S5, `MetadataOverflow::EvictToFit`: the metadata lane's head asked for
	/// room. With the figures republished first (the events ahead of the
	/// request are handled), evicts the policy's own victims -- the stack's
	/// `evict_one` and `erase`, the eviction pass's victim step, so both
	/// stores take the same victims -- until the object count is under the
	/// key ceiling for every new key waiting, at most `MAKE_ROOM_BATCH` at a
	/// time; applies what the evictions queued, republishes, and answers
	/// through the gate. No victim at all (an empty stack): it answers 0, and
	/// the waiting set fails with `MetadataOverflow` at once.
	///
	/// Why this is "the inserting set pays for it": the set that needs room
	/// does not return until its victims are gone -- the eviction is inside its
	/// latency, as Redis's `performEvictions` runs inside the command that
	/// needs memory. Only the thread differs: this one, which owns the stack
	/// and is the one place a victim is chosen the same way in both stores.
	#[cfg(feature = "hybrid_cache_common")]
	fn handle_make_room(&mut self, request: u64) {
		#[cfg(test)]
		if self.status.gate().test_panic_on_make_room.load(std::sync::atomic::Ordering::Relaxed) {
			panic!("test: the policy worker dies at a MakeRoom");
		}

		self.publish_metadata(false);
		self.publish_gate();

		let waiting = self.status.gate().meta_lane.len().max(1) as u64;
		let deficit = self.status
			.live_num_objects()
			.saturating_add(waiting)
			.saturating_sub(self.status.gate().k_max())
			.min(crate::gate::MAKE_ROOM_BATCH);

		let mut evicted = 0;
		let mut tries = self.policy_stack.len() as u64 + deficit;

		while evicted < deficit && tries > 0 {
			tries -= 1;

			match self.evict_victim(false) {
				Ok(Victim::Evicted) => evicted += 1,
				Ok(Victim::Missed) => {},
				Ok(Victim::Exhausted) | Err(_) => break,
			}
		}

		self.apply_tier_migrations();
		self.publish_metadata(false);
		self.publish_gate();
		self.refresh_tier_gauges();

		self.status.gate().answer_make_room(request, evicted);
	}

	/// The stack's own structures by node (`PolicyStack::structure_bytes`;
	/// every tiered design meters itself).
	#[cfg(feature = "hybrid_cache_common")]
	fn stack_structures(&self) -> crate::meta::NodeBytes {
		self.policy_stack.structure_bytes().unwrap_or_default()
	}

	/// S5a: republishes M when the event just handled changed the stack's
	/// structures -- a slab chunk, an index doubling, a free list's buffer, a
	/// bucket map's node. A virtual call and a handful of loads when it did
	/// not.
	#[cfg(feature = "hybrid_cache_common")]
	fn publish_metadata_if_the_stack_changed(&mut self) {
		if self.tier_migration && self.stack_structures() != self.metadata.structures {
			self.publish_metadata(false);
		}
	}

	/// One victim of the stack's own order, removed from the map: `evict_one`
	/// and `erase` -- the eviction pass's victim step, and `MakeRoom`'s (S5),
	/// so both take victims the same way in both stores. `fallback`: when the
	/// stack names no victim, let `erase` evict an arbitrary map entry (a
	/// DashMap stack behind its map -- the eviction pass's last resort);
	/// `MakeRoom` does not.
	fn evict_victim(&mut self, fallback: bool) -> Result<Victim, CacheError> {
		let maybe_key = self.policy_stack
			.evict_one()
			.map(|key| EraseKey::Hashed(key));

		// S5a: `erase` looks the victim up with `entry`, which reserves a
		// slot in its DashMap shard first (`crate::meta::ShardState`).
		#[cfg(feature = "hybrid_cache_common")]
		if self.tier_migration {
			if let Some(EraseKey::Hashed(victim)) = &maybe_key {
				crate::meta::map_write(&self.objects, &mut self.metadata.map, *victim);
			}
		}

		if maybe_key.is_none() && !fallback {
			return Ok(Victim::Exhausted);
		}

		// A split design can legitimately have an empty stack over a
		// non-empty map -- that divergence is what `erase`'s `None`
		// fallback exists to clean up, by evicting an arbitrary map entry.
		// The merged store has no such fallback: `None` means nothing
		// LINKED is left, while `used_size` still counts what clients have
		// published and this worker has not linked yet -- values whose
		// `Set` arrives after this pass, which never evicts an unlinked
		// value (`take_evict`). The pass stops, silently; the next links
		// them and evicts. Without this the loop `continue`s on unchanged
		// state forever. A store with NOTHING in it -- no unlinked value
		// either -- and `used_size` still over the size is an accounting
		// bug, not a backlog: the pass stops and says so, as it did before
		// unlinked values existed.
		#[cfg(feature = "merged_object_store")]
		if maybe_key.is_none() {
			if self.objects.len() == 0 {
				log::error!("Nothing left to evict with used_size still over max");

				#[cfg(test)]
				{
					self.nothing_left_to_evict += 1;
				}
			}

			return Ok(Victim::Exhausted);
		}

		let erase_result = erase(
			&self.objects,
			&self.status,
			&self.overhead_manager,
			maybe_key,
		);

		let Ok((_key, _evicted_obj)) = erase_result else {
			return Ok(Victim::Missed);
		};

		#[cfg(test)]
		if let Some(evicted) = &mut self.evicted {
			evicted.push(_key);
		}

		#[cfg(feature = "hybrid_cache_common")]
		if self.status.policy().is_hybrid() {
			self.status.record_hybrid_eviction();
		}

		Ok(Victim::Evicted)
	}

	fn apply_evictions(&mut self) -> Result<(), CacheError> {
		// The cache's one policy, for `used_size`'s per-object overhead.
		let policy = self.status.policy();
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
			let needs_capacity_eviction = self.policy_stack.len() > 0 && self.policy_stack.needs_capacity_eviction();

			if !over_max_size && !needs_capacity_eviction {
				migstats::rec(&migstats::EVICT, &migstats::EVICT_TOT, _evicted_this_call);
				migstats::etick();
				break;
			}

			match self.evict_victim(true)? {
				Victim::Evicted => {},
				Victim::Missed => continue,
				Victim::Exhausted => break,
			}

			_evicted_this_call += 1;
		}

		Ok(())
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
	///
	/// The long poll is IDLE (S5): before it this thread sets the gate's idle
	/// bit, fences, and parks only if its channel is still empty -- the
	/// worker's half of the Dekker pair with `PaperCache::kick_idle_worker`,
	/// whose set wrote the channel, fenced, and reads the bit. So a set either
	/// finds the bit and wakes this thread, or this thread finds the set and
	/// does not park: the first set after an idle spell is taken at once, in
	/// both stores. Nor does it park long while a set waits in either of the
	/// gate's lanes: it polls SHORT then (S5 B2).
	fn delay_event_loop(&mut self, now: Instant, has_current_set: bool) {
		let delay = polling_delay(now, self.last_set_time, has_current_set);

		// S5 B2: SHORT while a set waits in either lane -- every pass resettles
		// and notifies the head (design 3.9.10(3)).
		#[cfg(feature = "hybrid_cache_common")]
		let delay = match self.status.gate().waiting() {
			true => SHORT_POLLING_DURATION,
			false => delay,
		};

		if has_current_set {
			self.last_set_time = Some(now);
		}

		#[cfg(feature = "hybrid_cache_common")]
		if delay == LONG_POLLING_DURATION {
			use std::sync::atomic::{fence, Ordering};

			let gate = self.status.gate();

			gate.worker_idle.store(true, Ordering::Relaxed);

			#[cfg(test)]
			gate.test_idle_spells.fetch_add(1, Ordering::Relaxed);

			fence(Ordering::SeqCst);

			if self.listener.is_empty() && !gate.waiting() {
				thread::park_timeout(delay);
			}

			gate.worker_idle.store(false, Ordering::Relaxed);

			return;
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
/// new object, is undone (`ArenaHybridStack::touch_to_front`, the merged
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
/// counted although nothing moved. Dropped entries are counted in
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
///
/// # Tags
///
/// Generic over the entry, so a tagged entry (`TaggedMigration`) keeps its
/// origin: entries are kept or dropped whole, never rebuilt. So of a stack
/// entry and a corrective for one key in one drain, the one kept is the LAST
/// -- the reconcile appends its correctives at the end of the drain, and
/// appends one only toward the other tier than the key's last entry (or with
/// none for the key) -- and it keeps its own tag.
#[cfg(feature = "hybrid_cache_common")]
fn split_tier_migrations<E: MigrationEntry>(migrations: Vec<E>) -> (Vec<E>, Vec<E>, usize) {
	let slow = migrations.iter().filter(|entry| entry.tier() == Tier::Slow).count();
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

	for (i, entry) in migrations.iter().enumerate() {
		if entry.tier() == smaller {
			last.entry(entry.key()).or_insert((0, 0)).0 = i as u32;
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
	for (i, entry) in migrations.iter().enumerate() {
		if entry.tier() == smaller {
			continue;
		}

		let i = i as u32;

		let superseded = match last.get_mut(&entry.key()) {
			Some((smaller_last, larger_end)) => {
				*larger_end = i + 1;
				*smaller_last > i
			},
			None => false,
		};

		if !superseded {
			larger_half.push(*entry);
		}
	}

	// The smaller half, now that every key's end in the larger half is known:
	// kept if all of the key's entries there come before this one.
	for (i, entry) in migrations.iter().enumerate() {
		if entry.tier() != smaller {
			continue;
		}

		if last[&entry.key()].1 <= i as u32 {
			smaller_half.push(*entry);
		}
	}

	let dropped = migrations.len() - demotions.len() - promotions.len();

	(demotions, promotions, dropped)
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
/// Two tests outside these modules drive migrations and hold it too, for the
/// same reason: `arena_hybrid_stack`'s stale-demotion test, and
/// `phys::tests::a_hit_is_counted_by_the_tier_it_was_served_from`, whose real
/// FIFO cache demotes through its own queue (it reaches the lock through the
/// `crate::worker::migration_test_lock` re-export, and holds it until the
/// cache has dropped and joined its consumers). Without it that test's ten
/// demotions landed inside `flush_returns_only_after_every_disposition`'s
/// window once in twelve lib runs: `applied: 11` against the expected 1.
/// Otherwise nothing in the crate's unit tests drives a migration:
/// `apply_migration` is reached only from a `MigrationQueue` consumer or from
/// `apply_migration_batches`, and the hybrid caches `lib.rs` builds in its own
/// tests are all configured with a fast tier as large as the whole cache, so
/// they never demote. The next test that does needs the lock as well.
///
/// Poisoning is stepped over on purpose: a panicking test is already a
/// failure, and letting it cascade into every other test's error message only
/// hides which one broke.
#[cfg(all(test, feature = "hybrid_cache_common"))]
pub(crate) mod migration_test_lock {
	use std::sync::{Mutex, MutexGuard};

	pub(crate) fn lock() -> MutexGuard<'static, ()> {
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

		let status = make_status();
		let in_flight = status.migration_in_flight().clone();
		let marks = [1, 2, 3].map(|key| in_flight.mark(key));

		let queue = Arc::new(MigrationQueue::spawn(objects.clone(), 2, status).unwrap());

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

		// Every entry finished in its bucket, however it ended; only the one
		// that moved a value LANDED (the new-key rule's other half).
		assert_eq!(in_flight.total_pending(), 0, "applied, gone and declined all finish");
		assert_eq!(
			[1, 2, 3].map(|key| in_flight.mark(key)),
			[marks[0].wrapping_add(1), marks[1], marks[2]],
			"only the applied migration landed",
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
		let status = make_status();
		let in_flight = status.migration_in_flight().clone();
		let mark = in_flight.mark(KEY);
		let queue = MigrationQueue::spawn(objects.clone(), 1, status).unwrap();

		queue.push((KEY, Tier::Slow));

		entered.recv_timeout(Duration::from_secs(10)).expect(
			"the migration never reached the parked key clone: if \
			 TieredValue::migrated_to no longer clones the key between the \
			 snapshot and the swap, this test needs a new parking point -- the \
			 window it is testing still exists either way",
		);

		assert_eq!(in_flight.pending(KEY), 1, "an entry being copied is in flight");

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

		assert_eq!(in_flight.total_pending(), 0, "a superseded entry finishes too");
		assert_eq!(in_flight.mark(KEY), mark, "and did not land");
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
/// The merged store's twin of `arena_hybrid_stack::overwrite_tests`: the
/// re-promotion the worker queues after an overwrite's settle
/// (`MergedStore::worker_set`) is load-bearing even though `set` already built
/// the value in DRAM.
///
/// Lives here rather than in `merged_store.rs` because `apply_migration` is
/// private to this module tree.
#[cfg(all(test, feature = "merged_object_store", feature = "hybrid_cache_common"))]
mod merged_overwrite_tests {
	use std::sync::Arc;

	use super::{SetEvent, Tier, migration_queue::apply_migration};
	use crate::{
		HashedKey,
		merged_store::{MergedOrder, MergedStore, MigrationLog},
		object::Object,
	};

	const K: HashedKey = 0x51;
	const A: HashedKey = 0x52;

	fn fresh(key: HashedKey) -> Object<u64, crate::TieredBuffer> {
		Object::new_in(key, &[0xA5; 256], Tier::Fast, None)
	}

	/// K is demoted as the LRU tail; that demotion is still queued when an
	/// overwrite replaces K with a value built in DRAM, so it lands on the NEW
	/// value. The re-promotion the worker queues behind it, after the settle
	/// of the overwrite's `Set`, must restore it.
	#[test]
	fn an_overwrite_is_repromoted_after_a_stale_demotion() {
		// `apply_migration` bumps the process-wide migration counters that the
		// queue tests assert exact deltas on, so this runs under their lock.
		let _serialised = super::migration_test_lock::lock();

		let objects: crate::ObjectMapRef<u64, crate::TieredBuffer> = Arc::new(MergedStore::new());
		objects.set_order(MergedOrder::Lru);

		// The policy worker's log: each client insert below is followed by the
		// worker's handling of its `Set`.
		let mut log = MigrationLog::default();

		// Measure one object's tier charge untiered, then size the fast tier
		// to hold exactly one: the second admission demotes the first.
		objects.insert(K, fresh(K));
		objects.worker_set(K, 256, SetEvent::Fresh, &mut log);
		let one = objects.fast_bytes_used();
		objects.configure_tiering(2 * one - 1, 0, 1_000_000, 1_000_000);
		objects.insert(A, fresh(A));
		objects.worker_set(A, 256, SetEvent::Fresh, &mut log);

		assert_eq!(objects.tier_of(K), Some(Tier::Slow), "K should be the demoted LRU tail");

		let mut queue = log.take_untagged();
		assert_eq!(queue, vec![(K, Tier::Slow)], "the demotion is decided, not yet applied");

		// The overwrite: an LRU `set` builds the new value in DRAM, then
		// inserts; then the worker takes its `Set`.
		objects.insert(K, fresh(K));
		objects.worker_set(K, 256, SetEvent::Replaced { resized: false }, &mut log);
		queue.extend(log.take_untagged());

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

	use super::{PolicyWorker, Tier, migration_test_lock};
	use crate::{
		CacheSize, HashedKey, ObjectMapRef, PaperPolicy, TieredBuffer,
		merged_store::MergedStore,
		object::{Object, overhead::resident_object_bytes},
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
		let objects: Objects = Arc::new(MergedStore::new());

		// The per-object model this module was written against (S5).
		let (worker, _status, _overhead_manager) = super::test_support::tiered_worker(
			objects.clone(),
			1 << 30,
			PaperPolicy::LruCompactHybrid,
			true,
			queued,
		);

		objects.configure_tiering(fast_capacity, 0, 1_000_000, 1_000_000);

		(worker, objects)
	}

	/// What `PaperCache::set` does under LRU -- build the bytes in DRAM and
	/// insert -- and then the worker's handling of its `Set`, which admits the
	/// key fast and settles.
	fn set(worker: &mut PolicyWorker<u64, TieredBuffer>, objects: &Objects, key: HashedKey, len: usize) {
		let object = Object::new_in(key, &vec![key as u8; len], Tier::Fast, None);
		let base_size = worker.overhead_manager.base_size(&object);
		let resident = worker.overhead_manager.dram_resident_size(&object);

		let previous = objects
			.insert(key, object)
			.map(|old| worker.overhead_manager.base_size(&old));

		worker.handle_set(key, base_size, resident, Tier::Fast, previous, 0, crate::worker::Placement::Normal);
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

			set(&mut worker, &objects, key, 100);
			worker.apply_tier_migrations();
			assert_placement_matches_the_model(&objects, &[key]);

			let served = objects.get_ref(&key).map(|object| object.value().tier());
			worker.handle_get(key, served);
			worker.apply_tier_migrations();

			assert_eq!(objects.tier_of(key), Some(Tier::Slow), "queued = {queued}");
			assert_placement_matches_the_model(&objects, &[key]);
		}
	}

	/// Many keys, repeated hits, and a budget that holds a few objects, so
	/// some promotions stand and some are undone at once -- applied per event,
	/// and every eight events, since a batch can span events.
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
					set(&mut worker, &objects, key, LEN);
					event(&mut worker);

					// Re-hit a spread of older keys, most of them slow by now.
					for back in [1, 7, 31] {
						if n >= back {
							let key = keys[n - back];
							let served = objects.get_ref(&key).map(|object| object.value().tier());
							worker.handle_get(key, served);
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
	use crate::object::Object;

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
		let objects: ObjectMapRef<u32, TestBuffer> = crate::new_hybrid_object_map();

		let (worker, status, _overhead_manager) = super::test_support::tiered_worker(
			objects.clone(),
			1_000,
			PaperPolicy::LruCompact,
			false,
			queued,
		);

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
			objects.clear_counted(|_| 0);

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

	/// Runs one scripted drain through `apply_tier_migrations` and reports
	/// the copies it made: `MIG_APPLIED`'s delta, exact under
	/// `migration_test_lock`.
	fn apply_scripted(
		worker: &mut PolicyWorker<u32, TestBuffer>,
		migrations: Vec<(HashedKey, Tier)>,
	) -> u64 {
		use std::sync::atomic::Ordering;

		let before = migration_queue::MIG_APPLIED.load(Ordering::Relaxed);

		worker.policy_stack = Box::new(super::test_support::FakeStack::scripted(migrations));
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
		assert_eq!(split_tier_migrations(Vec::<(HashedKey, Tier)>::new()), (vec![], vec![], 0));
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

	/// The origin tag survives the split: a tagged drain keeps exactly the
	/// entries -- the same positions -- that the untagged one does, each with
	/// its own tag. So of a stack entry and a corrective for one key in one
	/// drain, the LAST is kept with its tag: every drain of up to four
	/// entries on two keys, over both tags.
	#[test]
	fn the_origin_tag_survives_the_split_on_the_entry_kept() {
		use MigrationOrigin::{Reconcile, Stack};

		assert_eq!(
			split_tier_migrations(vec![(7, Tier::Slow, Stack), (8, Tier::Fast, Stack), (7, Tier::Fast, Reconcile)]),
			(vec![], vec![(8, Tier::Fast, Stack), (7, Tier::Fast, Reconcile)], 1),
			"the stack's demotion, then the reconcile's promotion: the corrective is kept, as one",
		);
		assert_eq!(
			split_tier_migrations(vec![(7, Tier::Fast, Reconcile), (7, Tier::Slow, Stack)]),
			(vec![(7, Tier::Slow, Stack)], vec![], 1),
			"a corrective, then a later decision of the stack's: the decision is kept",
		);

		/// An entry that remembers where it was in its drain.
		#[derive(Clone, Copy)]
		struct Probe(HashedKey, Tier, usize);

		impl MigrationEntry for Probe {
			fn key(&self) -> HashedKey {
				self.0
			}

			fn tier(&self) -> Tier {
				self.1
			}

			fn origin(&self) -> MigrationOrigin {
				Stack
			}
		}

		const ENTRIES: [TaggedMigration; 8] = [
			(1, Tier::Slow, Stack),
			(1, Tier::Fast, Stack),
			(2, Tier::Slow, Stack),
			(2, Tier::Fast, Stack),
			(1, Tier::Slow, Reconcile),
			(1, Tier::Fast, Reconcile),
			(2, Tier::Slow, Reconcile),
			(2, Tier::Fast, Reconcile),
		];

		let mut drains = 0;

		for len in 0..=4u32 {
			for code in 0..8usize.pow(len) {
				let drain: Vec<TaggedMigration> =
					(0..len).map(|p| ENTRIES[code / 8usize.pow(p) % 8]).collect();

				let probes = drain.iter().enumerate().map(|(i, e)| Probe(e.0, e.1, i)).collect();
				let (slow, fast, dropped) = split_tier_migrations::<Probe>(probes);
				let at = |half: Vec<Probe>| half.into_iter().map(|p| drain[p.2]).collect::<Vec<_>>();

				assert_eq!(split_tier_migrations(drain.clone()), (at(slow), at(fast), dropped), "{drain:?}");
				drains += 1;
			}
		}

		assert_eq!(drains, 4_681);
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
	fn wait_for(what: &str, deadline: Duration, done: impl FnMut() -> bool) {
		super::test_support::wait_for_every(Duration::from_micros(100), what, deadline, done);
	}

	type WorkerHandle = thread::JoinHandle<Result<(), CacheError>>;

	/// A real worker, on its own thread, for a cache that has seen nothing.
	fn spawn_worker() -> (Sender<WorkerEvent>, StatusRef, WorkerHandle) {
		let (tx, rx) = unbounded::<WorkerEvent>();

		let objects: ObjectMapRef<u32, crate::TieredBuffer> = crate::new_hybrid_object_map();

		let status: StatusRef = Arc::new(
			AtomicStatus::new(1_000_000, &[PaperPolicy::LruCompact], PaperPolicy::LruCompact).unwrap(),
		);

		let overhead_manager = Arc::new(OverheadManager::new(&status));

		let worker = PolicyWorker::<u32, crate::TieredBuffer>::new(
			rx,
			objects,
			status.clone(),
			overhead_manager,
		).unwrap();

		(tx, status, register_worker(worker))
	}

	/// The premise both tests rest on, checked (see
	/// `test_support::parked_on_the_long_poll`), the first pass polled every 100
	/// us. Returns the pass count to measure from.
	fn parked_on_the_long_poll(status: &StatusRef) -> u64 {
		super::test_support::parked_on_the_long_poll(status, Duration::from_micros(100))
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
		tx.send(WorkerEvent::Set(1, 64, 0, None, None, Tier::Fast, 0, Placement::Normal)).unwrap();

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

	const TEST_POLICY: PaperPolicy = PaperPolicy::LruCompact;
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
	/// `apply_evictions` reads `max_size()` on entry, and `LruCompactStack` ignores
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
		let built = object.value().tier();

		objects.insert(key, object);
		status.update_base_used_size(base_size as i64);
		status.incr_num_objects();
		worker.handle_set(key, base_size, dram_resident, built, None, 0, Placement::Normal);
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

		worker.apply_evictions().unwrap();

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

		worker.apply_evictions().unwrap();

		// One pass, and it went well past `max_size` -- the pre-watermark loop
		// would have stopped at exactly 16 objects' worth.
		assert_eq!(used(&status), 8 * per_object);
		assert_eq!(objects.len() as u64, 8);
		assert!(used(&status) < max_size);

		// Back up to 11 objects: above the drain target, below the trigger.
		fill(&objects, &status, &overhead_manager, &mut worker, 19..=21);
		worker.apply_evictions().unwrap();

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

		worker.apply_evictions().unwrap();

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
		worker.policy_stack = Box::new(super::test_support::FakeStack::over_budget(2));

		fill(&objects, &status, &overhead_manager, &mut worker, 1..=4);

		// The merged store evicts only what its policy worker has linked
		// (`take_evict`), and the stub stands in for that worker's handle:
		// link what the stub was given, as the handle's `insert_set` would.
		#[cfg(feature = "merged_object_store")]
		for key in 1..=4 {
			objects.worker_set(
				key,
				0,
				crate::worker::SetEvent::Fresh,
				&mut crate::merged_store::MigrationLog::default(),
			);
		}

		worker.apply_evictions().unwrap();

		// Four objects is far below even the low mark, so the capacity
		// condition never fires: the stack drains to its own budget and stops
		// there, instead of being pulled all the way down to `low * max_size`
		// by a watermark that has nothing to say about its sub-structure.
		assert_eq!(objects.len() as u64, 2);
		assert_eq!(used(&status), 2 * per_object);
	}
}

/// PHYS_FAST through the transient states T9's quiescent checks cannot see: a
/// migration parked between building its copy and swapping it in (the
/// `migration_queue::after_copy` rendezvous), and a reader's snapshot of a
/// value that is then overwritten. P counts ALLOCATIONS, so in both it must
/// count every live fast copy -- published or not, installed or superseded
/// -- and drop each one exactly when its last handle does.
///
/// P is PROCESS-GLOBAL, and this binary runs every lib test on parallel
/// threads, many of them building fast values, so an exact assertion on P
/// here would race all of them. Each test therefore re-runs ITSELF, alone,
/// in a child copy of this test binary (`--exact`, with
/// `PAPER_PHYS_TRANSIENT_CHILD=1` telling the child to run the body instead
/// of spawning again), where nothing else builds a value; the parent passes
/// only if the child ran exactly that one test and it passed. The rendezvous
/// is a `cfg(test)` hook inside the lib, which is why these cannot move to a
/// binary of their own the way T9 did.
#[cfg(all(test, feature = "hybrid_cache_common"))]
mod phys_transient_tests {
	use std::time::Duration;

	use super::*;
	use super::migration_queue::{after_copy, apply_migration};
	use crate::object::Object;
	// The merged store answers these calls with inherent methods.
	#[cfg(not(feature = "merged_object_store"))]
	use crate::object_store::ObjectStore;
	use crate::{phys, TieredValue};

	const CHILD: &str = "PAPER_PHYS_TRANSIENT_CHILD";

	type Objects = ObjectMapRef<u32, crate::TieredBuffer>;

	/// Runs `body` in a child process in which `test` is the only test.
	fn alone(test: &str, body: impl FnOnce()) {
		if std::env::var_os(CHILD).is_some_and(|value| value == "1") {
			body();
			return;
		}

		// libtest names a test by its path without the crate.
		let (_, module) = module_path!().split_once("::").expect("a module path");
		let name = format!("{module}::{test}");

		let out = std::process::Command::new(std::env::current_exe().expect("this test binary"))
			.args([name.as_str(), "--exact", "--test-threads=1"])
			.env(CHILD, "1")
			.output()
			.expect("could not re-run this test binary");

		let stdout = String::from_utf8_lossy(&out.stdout);
		let stderr = String::from_utf8_lossy(&out.stderr);

		assert!(
			out.status.success() && stdout.contains("test result: ok. 1 passed;"),
			"{name}, run alone in a child process ({}):\n--- stdout\n{stdout}\n--- stderr\n{stderr}",
			out.status,
		);
	}

	/// P, relative to `p0`.
	fn p(p0: i64) -> i64 {
		phys::fast_bytes_signed() - p0
	}

	fn charge(len: usize) -> i64 {
		phys::value_charge::<u32>(len as u32) as i64
	}

	fn tier_of(objects: &Objects, key: HashedKey) -> Tier {
		objects.get_ref(&key).unwrap().value().tier()
	}

	/// Starts `apply_migration(key -> tier)` on a thread of its own and
	/// returns once it is parked after building its copy, before the swap.
	fn park(
		objects: &Objects,
		key: HashedKey,
		tier: Tier,
	) -> (std::thread::JoinHandle<bool>, crossbeam_channel::Sender<()>) {
		let (entered, release) = after_copy::arm(key);
		let migrating = objects.clone();
		let migration = std::thread::spawn(move || apply_migration(&migrating, key, tier));

		entered
			.recv_timeout(Duration::from_secs(10))
			.expect("the migration never reached the post-copy park point");

		(migration, release)
	}

	/// Three migrations, each parked after its copy is built:
	///
	///   1. a promotion, applied: its DRAM copy is counted before it is
	///      published, while the map still holds the CXL original, and stays
	///      the one counted copy after the swap;
	///   2. a promotion superseded by a set while parked: the parked DRAM copy
	///      and the set's DRAM value are BOTH counted, and the copy is
	///      refunded when the consumer drops it unpublished;
	///   3. a demotion: its DRAM original stays counted until the swap drops
	///      it.
	#[test]
	fn a_parked_migration_keeps_every_live_fast_copy_counted() {
		alone("a_parked_migration_keeps_every_live_fast_copy_counted", || {
			const PROMOTED: HashedKey = 0x0001_F457;
			const SUPERSEDED: HashedKey = 0x0002_F457;
			const DEMOTED: HashedKey = 0x0003_F457;
			// Distinct size classes, so a copy charged at the wrong size or
			// refunded for the wrong value cannot cancel out.
			const LEN: usize = 3_000;
			const SET_LEN: usize = 5_000;
			const DEMOTED_LEN: usize = 9_000;

			let objects: Objects = crate::new_hybrid_object_map();
			let p0 = phys::fast_bytes_signed();

			// 1. A promotion, applied.
			objects.insert(PROMOTED, Object::new_in(PROMOTED as u32, &[0x11; LEN], Tier::Slow, None));
			assert_eq!(p(p0), 0, "a CXL value charges nothing");

			let (migration, release) = park(&objects, PROMOTED, Tier::Fast);
			assert_eq!(tier_of(&objects, PROMOTED), Tier::Slow, "parked before the swap");
			assert_eq!(p(p0), charge(LEN), "P counts the promotion's DRAM copy before it is published");

			release.send(()).unwrap();
			assert!(migration.join().unwrap(), "the promotion was applied");
			assert_eq!(tier_of(&objects, PROMOTED), Tier::Fast);
			assert_eq!(p(p0), charge(LEN), "one DRAM copy, now the installed one; the CXL original's free is not P's");

			let base = p(p0);

			// 2. A promotion superseded by a set while it is parked.
			objects.insert(SUPERSEDED, Object::new_in(SUPERSEDED as u32, &[0x22; LEN], Tier::Slow, None));
			assert_eq!(p(p0), base);

			let (migration, release) = park(&objects, SUPERSEDED, Tier::Fast);
			assert_eq!(p(p0), base + charge(LEN), "the parked copy");

			// What `set` does with a key it overwrites: install the new value
			// under the write guard and drop the displaced handle.
			let fresh = TieredValue::new_fast(SUPERSEDED as u32, &[0x33; SET_LEN], None);
			drop(objects.get_mut_ref(&SUPERSEDED).unwrap().set_data(fresh));
			assert_eq!(
				p(p0),
				base + charge(LEN) + charge(SET_LEN),
				"BOTH DRAM copies are counted: the parked promotion's and the set's",
			);

			release.send(()).unwrap();
			assert!(!migration.join().unwrap(), "the set superseded the promotion");
			assert_eq!(
				p(p0),
				base + charge(SET_LEN),
				"the superseded copy was refunded when the consumer dropped it unpublished",
			);

			let base = p(p0);

			// 3. A demotion.
			objects.insert(DEMOTED, Object::new_in(DEMOTED as u32, &[0x44; DEMOTED_LEN], Tier::Fast, None));
			assert_eq!(p(p0), base + charge(DEMOTED_LEN));

			let (migration, release) = park(&objects, DEMOTED, Tier::Slow);
			assert_eq!(tier_of(&objects, DEMOTED), Tier::Fast, "parked before the swap");
			assert_eq!(
				p(p0),
				base + charge(DEMOTED_LEN),
				"a demotion's DRAM original stays counted while its CXL copy is built",
			);

			release.send(()).unwrap();
			assert!(migration.join().unwrap(), "the demotion was applied");
			assert_eq!(tier_of(&objects, DEMOTED), Tier::Slow);
			assert_eq!(p(p0), base, "the swap dropped the DRAM original");

			drop(objects);
			assert_eq!(p(p0), 0, "every fast allocation was refunded");
		});
	}

	/// What `get` does: lift a strong handle out under the shard guard, then
	/// copy with no guard held. A set that overwrites the key meanwhile
	/// displaces the value, but the snapshot keeps it allocated -- and P
	/// counted -- until the reader lets go.
	#[test]
	fn a_readers_snapshot_keeps_an_overwritten_fast_value_counted() {
		alone("a_readers_snapshot_keeps_an_overwritten_fast_value_counted", || {
			const KEY: HashedKey = 0x0004_F457;
			const OLD_LEN: usize = 3_000;
			const NEW_LEN: usize = 5_000;

			let objects: Objects = crate::new_hybrid_object_map();
			let p0 = phys::fast_bytes_signed();

			objects.insert(KEY, Object::new_in(KEY as u32, &[0x55; OLD_LEN], Tier::Fast, None));
			assert_eq!(p(p0), charge(OLD_LEN));

			let snapshot = objects.get_ref(&KEY).map(|object| object.snapshot()).unwrap();

			let fresh = TieredValue::new_fast(KEY as u32, &[0x66; NEW_LEN], None);
			drop(objects.get_mut_ref(&KEY).unwrap().set_data(fresh));
			assert_eq!(
				p(p0),
				charge(OLD_LEN) + charge(NEW_LEN),
				"the reader's snapshot keeps the overwritten value counted",
			);

			assert_eq!(snapshot.bytes(), &[0x55; OLD_LEN][..], "the reader still copies the old bytes");
			drop(snapshot);
			assert_eq!(p(p0), charge(NEW_LEN), "the snapshot was the last handle: its drop refunded it");

			drop(objects);
			assert_eq!(p(p0), 0);
		});
	}
}


/// The reconcile and the heal (backpressure plan S3; see `Observed`), the
/// new-key rule and the heal rule, the correctives' own counters, and the
/// placement audit, driven through the worker the way its event loop drives
/// it -- `handle_set` / `handle_get`, then the drain -- over the build's own
/// store: the split stacks in the DashMap builds, the merged
/// store in the merged builds, with orders both implement (LRU, FIFO, LFU);
/// 2Q, which only the split store implements, in the split builds.
///
/// `drain_and_apply` hands back the drain it applied, tagged, so a test can
/// require EXACTLY one entry for a key -- not merely a right final placement,
/// which a duplicate would reach too -- and tell a corrective from a stack
/// decision.
///
/// The races are made deterministic with `migration_queue::after_copy`: a
/// migration is parked after its copy is built and before its swap. Either
/// the worker's apply -- which flushes the queue in test builds -- runs on a
/// thread of its own meanwhile, so the test thread can play the client, or
/// the test turns the flush off (`test_flush`) and drives the worker on while
/// a consumer is parked (`park`), which is how a stale entry is kept queued
/// across the events that make it stale. A test that needs a consumer is
/// SKIPPED -- loudly, and only when `MIGRATION_QUEUE_THREADS` is 0 -- where
/// migrations apply inline (`queue_or_skip`).
// Helpers the test modules below share: child processes, one per test or per policy.
#[cfg(all(test, any(feature = "hybrid_cache_common", feature = "merged_object_store")))]
mod test_support;

// Backpressure plan S4: the merged store's policy work on the policy worker,
// and the uniform differential (T14) over both stores.
#[cfg(all(test, feature = "hybrid_cache_common"))]
mod s4_tests;

// Backpressure plan S5, commit B1: the admission path -- the size checks, the
// metadata cap, structural slow placement, every settle on eff, the kick.
#[cfg(all(test, feature = "hybrid_cache_common"))]
mod s5_tests;

// Backpressure plan S5, commit B2: the byte gate, through real tiered caches,
// each test alone in a child process.
#[cfg(all(test, feature = "hybrid_cache_common"))]
mod s5_gate_tests;

// S4's follow-ups: a flat cache over the merged store through a real worker,
// reaper and wipe -- in every merged build, the flat-merged one included.
#[cfg(all(test, feature = "merged_object_store"))]
mod flat_merged_tests;

#[cfg(all(test, feature = "hybrid_cache_common"))]
mod reconcile_tests {
	use std::time::Duration;

	use super::*;
	use super::migration_queue::after_copy;
	#[cfg(not(feature = "merged_object_store"))]
	use super::test_support::each_alone;
	use crate::hybrid_policy::admission_tier;
	use crate::object::Object;
	// The merged store answers these calls with inherent methods.
	#[cfg(not(feature = "merged_object_store"))]
	use crate::object_store::ObjectStore;
	use crate::phys::{PlacementAudit, value_charge};
	use crate::TieredBuffer;
	use MigrationOrigin::Reconcile;
	use Tier::{Fast, Slow};

	pub(super) type Objects = ObjectMapRef<u64, TieredBuffer>;
	pub(super) type Worker = PolicyWorker<u64, TieredBuffer>;

	/// The fast tier: about fifteen of the tests' values, once the per-object
	/// reservation is off.
	pub(super) const FAST: CacheSize = 16 * 1024;
	pub(super) const LEN: usize = 1_000;

	/// An empty drain, typed.
	const NONE: &[(HashedKey, Tier)] = &[];

	pub(super) fn make_worker(policy: PaperPolicy) -> (Worker, Objects) {
		let objects: Objects = crate::new_hybrid_object_map();

		// The per-object model these tests were written against (S5).
		let (mut worker, _status, _overhead_manager) =
			super::test_support::tiered_worker(objects.clone(), 1 << 30, policy, true, true);

		worker.handle_resize_fast_tier(FAST);
		worker.apply_tier_migrations();

		(worker, objects)
	}

	/// A `Set` event's fields, as the client produced them.
	#[derive(Clone, Copy, Debug)]
	pub(super) struct Published {
		pub(super) base_size: ObjectSize,
		pub(super) resident: ObjectSize,
		pub(super) built: Tier,
		/// The base size of the value the map insert replaced, `None` if it
		/// replaced nothing.
		pub(super) previous: Option<ObjectSize>,
		/// The key's bucket's landed count, read before the insert.
		pub(super) mark: u32,
		/// Where the client placed the value (S5): `Normal` unless a test
		/// says otherwise, or the admission decision (`gate::decide`) did.
		pub(super) placement: Placement,
	}

	impl Published {
		/// The map insert replaced nothing.
		pub(super) fn fresh(&self) -> bool {
			self.previous.is_none()
		}
	}

	/// What `PaperCache::set` does before its broadcast: read the mark, build
	/// in `built`, insert, account.
	pub(super) fn publish(
		status: &StatusRef,
		overhead_manager: &OverheadManagerRef,
		objects: &Objects,
		key: HashedKey,
		len: usize,
		built: Tier,
	) -> Published {
		let object = Object::new_in(key, &vec![key as u8; len], built, None);
		let base_size = overhead_manager.base_size(&object);
		let resident = overhead_manager.dram_resident_size(&object);
		let mark = status.migration_in_flight().mark(key);

		let previous = match objects.insert(key, object) {
			Some(old) => {
				let old_size = overhead_manager.base_size(&old);

				status.update_base_used_size(base_size as i64 - old_size as i64);

				Some(old_size)
			},

			None => {
				status.incr_num_objects();
				status.update_base_used_size(base_size as i64);

				None
			},
		};

		Published { base_size, resident, built, previous, mark, placement: Placement::Normal }
	}

	/// The worker's handling of a published value's `Set`.
	pub(super) fn handle(worker: &mut Worker, key: HashedKey, set: Published) {
		worker.handle_set(key, set.base_size, set.resident, set.built, set.previous, set.mark, set.placement);
	}

	/// What `PaperCache::del` does before its broadcast: the client's erase.
	/// In the merged store a linked value goes DEAD, for the worker's
	/// `handle_del` to retire; in the DashMap stores the map entry goes.
	pub(super) fn publish_del(status: &StatusRef, overhead_manager: &OverheadManagerRef, objects: &Objects, key: HashedKey) {
		erase(objects, status, overhead_manager, Some(EraseKey::Original(&key, key))).expect("the key is live");
	}

	/// A whole set: `publish`, then the worker's handling of its `Set`.
	pub(super) fn set(worker: &mut Worker, objects: &Objects, key: HashedKey, len: usize, built: Tier) {
		let published = publish(&worker.status, &worker.overhead_manager, objects, key, len, built);

		handle(worker, key, published);
	}

	/// A settle that demotes every fast key -- what these tests' admission of
	/// a value twice the fast tier did, until S5 made such a value STRUCTURAL
	/// (built and placed slow, no settle): the fast tier shrunk to one byte.
	/// `restore_fast` puts it back; a grow moves nothing in LRU or FIFO.
	pub(super) fn shrink_fast(worker: &mut Worker) {
		worker.handle_resize_fast_tier(1);
	}

	pub(super) fn restore_fast(worker: &mut Worker) {
		worker.handle_resize_fast_tier(FAST);
	}

	/// The drain the event loop would apply after the event just handled --
	/// the stack's entries and the reconcile's -- applied, and returned.
	pub(super) fn drain_and_apply(worker: &mut Worker) -> Vec<TaggedMigration> {
		let (inline, drain) = worker.drain_reconciled();
		worker.apply_migration_batches(drain.clone(), inline);
		drain
	}

	/// A drain's entries for one key, in order.
	pub(super) fn of(drain: &[TaggedMigration], key: HashedKey) -> Vec<Tier> {
		drain.iter().filter(|(k, _, _)| *k == key).map(|(_, tier, _)| *tier).collect()
	}

	/// A drain's CORRECTIVES for one key (`MigrationOrigin::Reconcile`), in
	/// order.
	pub(super) fn correctives(drain: &[TaggedMigration], key: HashedKey) -> Vec<Tier> {
		drain
			.iter()
			.filter(|(k, _, origin)| *k == key && *origin == Reconcile)
			.map(|(_, tier, _)| *tier)
			.collect()
	}

	/// Where `key`'s bytes are: its value's tag.
	pub(super) fn bytes_tier(objects: &Objects, key: HashedKey) -> Tier {
		objects.get_ref(&key).expect("a live key").value().tier()
	}

	pub(super) fn placement(worker: &Worker, key: HashedKey) -> Option<Tier> {
		worker.policy_stack.placement_of(key)
	}

	fn other(tier: Tier) -> Tier {
		match tier {
			Fast => Slow,
			Slow => Fast,
		}
	}

	/// Moves `key`'s bytes to `tier` behind the stack's back -- what a stale
	/// or a lost migration leaves.
	fn move_bytes(objects: &Objects, key: HashedKey, tier: Tier) {
		let mut object = objects.get_mut_ref(&key).expect("a live key");
		let moved = object.value().migrated_to(tier);

		drop(object.set_data(moved));
	}

	/// New keys set the way a client with an UP-TO-DATE mirror sets them: each
	/// built where `admission_tier` says once the worker has published its
	/// gauges, and each `Set` drained before the next.
	pub(super) fn fill(worker: &mut Worker, objects: &Objects, keys: std::ops::RangeInclusive<HashedKey>, len: usize) {
		for key in keys {
			let built = built_by_the_client(worker, objects, key);

			set(worker, objects, key, len, built);
			drain_and_apply(worker);
		}
	}

	/// The tier `PaperCache::set` would build `key`'s value in now, with the
	/// worker's gauges -- the latch mirror -- up to date.
	pub(super) fn built_by_the_client(worker: &mut Worker, objects: &Objects, key: HashedKey) -> Tier {
		worker.refresh_tier_gauges();

		admission_tier(worker.status.policy(), key, &worker.status, objects)
	}

	/// Whether `worker` has a migration queue. A test that parks a consumer
	/// needs one; without it -- with `MIGRATION_QUEUE_THREADS=0`, and only then
	/// -- the test is SKIPPED, and says so, rather than passing silently.
	pub(super) fn queue_or_skip(worker: &Worker, test: &str) -> bool {
		if worker.migration_queue.is_some() {
			return true;
		}

		let threads = std::env::var("MIGRATION_QUEUE_THREADS").ok();

		assert_eq!(
			threads.as_deref().and_then(|value| value.parse::<usize>().ok()),
			Some(0),
			"{test}: no migration queue, yet MIGRATION_QUEUE_THREADS is {threads:?}",
		);

		eprintln!("SKIPPED {test}: MIGRATION_QUEUE_THREADS=0, migrations apply inline");

		false
	}

	/// The `n`th key after `key` on the same migration consumer (the queue
	/// shards by key modulo its consumer count), in another in-flight bucket.
	fn same_consumer(key: HashedKey, n: HashedKey) -> HashedKey {
		key + n * migration_queue::threads() as HashedKey
	}

	/// Parks the consumer that owns the live key `j`: a raw migration of `j`
	/// to its other tier, stopped by `after_copy` after its copy. Everything
	/// queued for that consumer behind it waits. `unpark` releases it.
	fn park(worker: &Worker, objects: &Objects, j: HashedKey) -> crossbeam_channel::Sender<()> {
		let (entered, release) = after_copy::arm(j);

		worker.migration_queue.as_ref().expect("a queue").push((j, other(bytes_tier(objects, j))));

		entered
			.recv_timeout(Duration::from_secs(10))
			.expect("the parking migration never reached the park point");

		release
	}

	/// Releases `park`'s consumer, moves `j` back to `j_tier` and waits for
	/// every consumer to finish.
	fn unpark(worker: &Worker, j: HashedKey, j_tier: Tier, release: crossbeam_channel::Sender<()>) {
		let queue = worker.migration_queue.as_ref().expect("a queue");

		release.send(()).unwrap();
		queue.push((j, j_tier));
		queue.flush();
	}

	/// Evicts one key through the worker's own eviction pass: the cache's size
	/// set one byte under what it holds, then restored.
	pub(super) fn evict_one_key(worker: &mut Worker) {
		let used = worker.status.used_size(&worker.status.policy());

		worker.status.set_max_size(used - 1);
		worker.apply_evictions().expect("an eviction pass");
		worker.status.set_max_size(1 << 30);
		drain_and_apply(worker);
	}

	/// The in-flight buckets balance once the consumers are idle, and nothing
	/// is misplaced.
	pub(super) fn assert_settled(worker: &mut Worker) {
		if let Some(queue) = &worker.migration_queue {
			queue.flush();
		}

		assert_eq!(
			worker.status.migration_in_flight().total_pending(),
			0,
			"every entry handed to the consumers finished, and was counted finished",
		);

		let audit = worker.placement_audit();
		assert!(audit.is_clean(), "{audit:?}");
	}


	/// Race E (T9's 2Q full fast-admission "never quiesced", stack 44 against
	/// map 43): an eviction pass between a re-set's publish -- its map insert
	/// and its bytes' accounting, both before its broadcast -- and its `Set`
	/// takes the very key being re-set (the stack's next victim, still known by
	/// its OLD value), and its erase removes the NEW value. The late `Set` must
	/// not admit the key: the stack would track an object the map no longer
	/// holds, and nothing would ever remove it. Likewise a `del` on another
	/// thread that erases a set's value and whose `Del` the worker handles first.
	/// Pre-existing (an S4 diagnostic that widened the gap hit it). Red without
	/// the map check in `handle_set` (`setnomapcheck`).
	#[cfg(not(feature = "merged_object_store"))]
	#[test]
	fn a_set_whose_value_was_taken_before_its_event_admits_nothing() {
		// Its worker's consumers apply migrations: the process-global counters
		// other tests take exact deltas of.
		let _serialised = migration_test_lock::lock();

		each_alone!("a_set_whose_value_was_taken_before_its_event_admits_nothing", [PaperPolicy::LruCompactHybrid, PaperPolicy::TwoQFullFastAdmissionCompactHybrid(0.25, 0.5)], |policy| {
			let (mut worker, objects) = make_worker(policy);
			fill(&mut worker, &objects, 1..=40, LEN);
			assert_settled(&mut worker);

			// The eviction: key 1 is the stack's next victim (the LRU tail; 2Q's
			// a1_out tail), re-set bigger, then an eviction pass before its Set.
			const K: HashedKey = 1;
			let published = publish(&worker.status, &worker.overhead_manager, &objects, K, 2 * LEN, Fast);
			evict_one_key(&mut worker);
			assert!(objects.get_ref(&K).is_none(), "{policy}: the eviction pass took the re-set key");

			handle(&mut worker, K, published);
			drain_and_apply(&mut worker);
			assert_eq!(placement(&worker, K), None, "{policy}: the late Set admitted a key the map no longer holds");

			// The delete: another thread erases a re-set's value; its Del first.
			const J: HashedKey = 30;
			let published = publish(&worker.status, &worker.overhead_manager, &objects, J, LEN, Fast);
			publish_del(&worker.status, &worker.overhead_manager, &objects, J);
			worker.handle_del(J);
			handle(&mut worker, J, published);
			drain_and_apply(&mut worker);
			assert_eq!(placement(&worker, J), None, "{policy}: a Set behind its value's delete admitted the key");

			assert_eq!(
				worker.policy_stack.len() as u64,
				worker.status.live_num_objects(),
				"{policy}: the stack tracks exactly the map's keys",
			);
			assert_settled(&mut worker);
		});
	}

	#[test]
	fn a_set_is_corrected_toward_its_placement_only_when_nothing_queued_lands_it_there() {
		const K: HashedKey = 7;
		const J: HashedKey = 8;

		let built = |tier| Observed::Built { key: K, built: tier, fence: false };

		// Built where it is placed: nothing.
		assert_eq!(corrective(NONE, built(Fast), Some(Fast)), None);
		assert_eq!(corrective(NONE, built(Slow), Some(Slow)), None);

		// Built elsewhere, nothing queued for the key: one entry, toward the
		// placement -- another key's entries do not count.
		assert_eq!(corrective(NONE, built(Fast), Some(Slow)), Some((K, Slow, Reason::Built)));
		assert_eq!(corrective(&[(J, Fast)], built(Slow), Some(Fast)), Some((K, Fast, Reason::Built)));

		// The stack queued the move itself (LFU's admission to slow before it
		// latches; LRU's re-promotion on a re-set), tagged or not: no second
		// entry.
		assert_eq!(corrective(&[(K, Slow)], built(Fast), Some(Slow)), None);
		assert_eq!(corrective(&[(K, Fast)], built(Fast), Some(Fast)), None);
		assert_eq!(corrective(&[(K, Slow, Reconcile)], built(Fast), Some(Slow)), None);

		// The key's LAST entry is where the event leaves the bytes.
		assert_eq!(corrective(&[(K, Fast), (K, Slow)], built(Fast), Some(Slow)), None);
		assert_eq!(
			corrective(&[(K, Slow), (K, Fast)], built(Slow), Some(Slow)),
			Some((K, Slow, Reason::Built)),
		);

		// A key the stack does not track is not corrected.
		assert_eq!(corrective(NONE, built(Fast), None), None);
	}

	/// The new-key rule's own arm (review M1): a key re-admitted while its
	/// bucket moved gets its placement queued LAST even when its value was
	/// built there -- unless the drain's last entry for it already is that.
	#[test]
	fn the_new_key_rule_queues_the_placement_last_even_when_the_value_was_built_there() {
		const K: HashedKey = 7;

		let fenced = |tier| Observed::Built { key: K, built: tier, fence: true };

		assert_eq!(corrective(NONE, fenced(Slow), Some(Slow)), Some((K, Slow, Reason::NewKey)));
		assert_eq!(corrective(NONE, fenced(Fast), Some(Fast)), Some((K, Fast, Reason::NewKey)));

		// Built elsewhere: the plain rule's corrective, counted as such.
		assert_eq!(corrective(NONE, fenced(Fast), Some(Slow)), Some((K, Slow, Reason::Built)));

		// The drain's own last entry for the key names the placement: it is
		// queued after everything in flight already.
		assert_eq!(corrective(&[(K, Slow)], fenced(Fast), Some(Slow)), None);
		assert_eq!(corrective(&[(K, Fast), (K, Slow)], fenced(Slow), Some(Slow)), None);

		// One naming the other tier is overruled, by the plain rule.
		assert_eq!(corrective(&[(K, Fast)], fenced(Slow), Some(Slow)), Some((K, Slow, Reason::Built)));

		assert_eq!(corrective(NONE, fenced(Fast), None), None, "untracked: nothing");
	}

	#[test]
	fn a_slow_served_hit_is_healed_only_toward_a_fast_placement_and_only_when_its_bucket_is_quiet() {
		const K: HashedKey = 7;

		let hit = Observed::ServedSlow { key: K, quiet: true };

		assert_eq!(corrective(NONE, hit, Some(Fast)), Some((K, Fast, Reason::Heal)), "placed fast: promoted");
		assert_eq!(corrective(NONE, hit, Some(Slow)), None, "placed slow: where it belongs");
		assert_eq!(corrective(&[(K, Fast)], hit, Some(Fast)), None, "the hit's own promotion: no second");
		assert_eq!(
			corrective(&[(K, Slow)], hit, Some(Fast)),
			Some((K, Fast, Reason::Heal)),
			"the placement wins",
		);
		assert_eq!(corrective(NONE, hit, None), None, "untracked: nothing");

		// Something of the key's bucket in flight decides instead (review M2).
		let busy = Observed::ServedSlow { key: K, quiet: false };
		assert_eq!(corrective(NONE, busy, Some(Fast)), None, "in flight: no heal");
	}

	/// The LFU stale latch, in both stores. The stack has latched -- a new key
	/// goes slow -- but the client read the latch open and built the new value
	/// in DRAM. The latched branch queues nothing (it trusts the build), in
	/// the DashMap stack and the merged store's worker alike, and the
	/// reconcile's is the one entry. Red with the reconcile off.
	#[test]
	fn a_new_key_built_fast_after_the_lfu_stack_latched_gets_exactly_one_demotion() {
		let _serialised = migration_test_lock::lock();

		let (mut worker, objects) = make_worker(PaperPolicy::LfuCompactHybrid);

		// Twice what the fast tier holds: the stack latches part-way.
		fill(&mut worker, &objects, 1..=32, LEN);
		assert_eq!(placement(&worker, 32), Some(Slow), "latched: a new key is placed slow");
		assert!(worker.placement_audit().is_clean(), "an up-to-date mirror strands nothing");

		const K: HashedKey = 100;
		set(&mut worker, &objects, K, LEN, Fast);
		assert_eq!(placement(&worker, K), Some(Slow));

		let drain = drain_and_apply(&mut worker);

		assert_eq!(of(&drain, K), vec![Slow], "exactly one entry, toward the placement");
		assert_eq!(correctives(&drain, K), vec![Slow], "a corrective, not a stack decision");
		assert_eq!(bytes_tier(&objects, K), Slow, "and the bytes follow it");
		assert_settled(&mut worker);
	}

	/// The other direction, in both stores: FIFO admits a new key to the fast
	/// prefix. Built in the slow tier, it gets exactly one promotion; built in
	/// DRAM, no entry at all.
	#[test]
	fn a_new_key_built_slow_where_the_stack_places_it_fast_gets_exactly_one_promotion() {
		let _serialised = migration_test_lock::lock();

		let (mut worker, objects) = make_worker(PaperPolicy::FifoCompactHybrid);

		const K: HashedKey = 1;
		set(&mut worker, &objects, K, LEN, Slow);
		assert_eq!(placement(&worker, K), Some(Fast), "FIFO admits a new key fast");

		let drain = drain_and_apply(&mut worker);
		assert_eq!(of(&drain, K), vec![Fast]);
		assert_eq!(bytes_tier(&objects, K), Fast);

		const J: HashedKey = 2;
		set(&mut worker, &objects, J, LEN, Fast);

		assert_eq!(of(&drain_and_apply(&mut worker), J), vec![], "built where placed: nothing");
		assert_eq!(bytes_tier(&objects, J), Fast);
		assert_settled(&mut worker);
	}

	/// A move queued once is not queued twice, in both stores. The LFU stack
	/// -- the DashMap one, or the merged store's worker -- admits a key that
	/// does not fit the tier to the SLOW tier before it has latched, and
	/// queues `(key, Slow)` for the value the client built in DRAM. The
	/// reconcile finds that entry in the drain -- the drain scan -- and adds
	/// none: red with the scan off in both stores (`nodrainscan`).
	#[test]
	fn a_move_the_stack_queues_itself_is_not_queued_twice() {
		let _serialised = migration_test_lock::lock();

		let (mut worker, objects) = make_worker(PaperPolicy::LfuCompactHybrid);

		fill(&mut worker, &objects, 1..=4, LEN);
		assert_eq!(placement(&worker, 4), Some(Fast), "well inside the tier: not latched");

		const K: HashedKey = 100;
		set(&mut worker, &objects, K, 15 * 1024, Fast);
		assert_eq!(placement(&worker, K), Some(Slow), "it does not fit: admitted slow");

		let drain = drain_and_apply(&mut worker);

		assert_eq!(of(&drain, K), vec![Slow], "one entry for the key, not two");
		assert_eq!(bytes_tier(&objects, K), Slow);
		assert_settled(&mut worker);
	}

	/// A latched new key is placed by the WORKER, and corrected in its own
	/// Set's drain, in both stores. It replaces the test that pinned the
	/// merged store's client-side corrective: the client no longer decides a
	/// new key's tier in any store, so there is nothing to queue before the
	/// worker takes the `Set` -- the decision itself waits for the worker.
	/// The LFU stack has latched; the client builds `K` fast (a latch read
	/// before the worker published it). Before the worker takes the `Set`
	/// nothing is queued and the key is not placed; after it, `K` is placed
	/// slow and exactly one `(K, Slow)` corrective is in that event's drain.
	#[test]
	fn a_latched_new_key_is_placed_by_the_worker_in_its_own_sets_drain() {
		let _serialised = migration_test_lock::lock();

		let (mut worker, objects) = make_worker(PaperPolicy::LfuCompactHybrid);

		// Twice what the fast tier holds: the stack latches part-way.
		fill(&mut worker, &objects, 1..=32, LEN);
		assert_eq!(placement(&worker, 32), Some(Slow), "latched: a new key is placed slow");

		const K: HashedKey = 100;
		let published = publish(&worker.status, &worker.overhead_manager, &objects, K, LEN, Fast);
		assert!(published.fresh());

		assert_eq!(placement(&worker, K), None, "published, not yet placed");
		assert_eq!(
			worker.policy_stack.drain_tagged_migrations(),
			vec![],
			"nothing is queued before the worker takes the Set",
		);

		handle(&mut worker, K, published);
		assert_eq!(placement(&worker, K), Some(Slow), "placed slow by the worker, latched");

		let drain = drain_and_apply(&mut worker);
		assert_eq!(of(&drain, K), vec![Slow], "exactly one entry, in the Set's own drain");
		assert_eq!(correctives(&drain, K), vec![Slow], "the reconcile's");
		assert_eq!(bytes_tier(&objects, K), Slow);
		assert_settled(&mut worker);
	}

	/// The heal, in both stores: a value whose bytes are in the slow tier while
	/// the stack places it fast -- moved behind the stack's back here -- is
	/// promoted once by a hit served from the slow tier. A hit the client
	/// served fast, and a miss, are not looked at.
	#[test]
	fn a_slow_served_hit_on_a_key_placed_fast_is_promoted_once() {
		let _serialised = migration_test_lock::lock();

		let (mut worker, objects) = make_worker(PaperPolicy::LruCompactHybrid);

		const K: HashedKey = 1;
		set(&mut worker, &objects, K, LEN, Fast);
		drain_and_apply(&mut worker);

		move_bytes(&objects, K, Slow);

		let audit = worker.placement_audit();
		assert_eq!(
			(audit.lagging, audit.lagging_bytes, audit.stranded, audit.untracked),
			(1, value_charge::<u64>(LEN as u32), 0, 0),
			"{audit:?}",
		);

		worker.handle_get(K, Some(Fast));
		worker.handle_get(K + 1, None);
		assert!(worker.observed.is_empty(), "a fast-served hit and a miss observe nothing");
		assert_eq!(of(&drain_and_apply(&mut worker), K), vec![]);

		worker.handle_get(K, Some(bytes_tier(&objects, K)));

		let drain = drain_and_apply(&mut worker);
		assert_eq!(of(&drain, K), vec![Fast], "one promotion");
		assert_eq!(correctives(&drain, K), vec![Fast], "the heal's");
		assert_eq!(bytes_tier(&objects, K), Fast);
		assert_settled(&mut worker);
	}

	/// A slow-served hit the stack promotes ITSELF is not healed as well, in
	/// both stores: LRU's hit promotes a demoted key, and that entry is the one.
	#[test]
	fn a_slow_served_hit_the_stack_promotes_itself_is_promoted_once() {
		let _serialised = migration_test_lock::lock();

		let (mut worker, objects) = make_worker(PaperPolicy::LruCompactHybrid);

		const K: HashedKey = 1;
		fill(&mut worker, &objects, K..=24, LEN);
		assert_eq!((placement(&worker, K), bytes_tier(&objects, K)), (Some(Slow), Slow), "demoted");

		worker.handle_get(K, Some(Slow));

		let drain = drain_and_apply(&mut worker);
		assert_eq!(of(&drain, K), vec![Fast], "the stack's entry, alone");
		assert_eq!(correctives(&drain, K), vec![], "a stack decision");
		assert_eq!(bytes_tier(&objects, K), Fast);
		assert_settled(&mut worker);
	}

	/// Review M1 (i), in the split builds (2Q, which the merged store does
	/// not implement): `K` placed fast in 2Q's main, its bytes still slow (as
	/// while its promotion is queued -- moved behind the stack's back here). A
	/// hit is served slow; the client deletes `K` and sets it again, built
	/// slow as 2Q admits a new key. The worker takes the hit first and heals
	/// `(K, Fast)` -- nothing of the bucket was in flight -- which lands on
	/// the FRESH value; the `Set` then admits `K` to the slow FIFO, where it
	/// was built. Nothing is in flight any more, but a migration of the bucket
	/// LANDED after the client's mark: the new-key rule queues `(K, Slow)`
	/// last, and the value ends where the stack placed it. Red with the rule
	/// off (`nofence`) and with the in-flight count alone (`pendingonly`):
	/// "stranded". No consumer is parked, so it runs inline too
	/// (`MIGRATION_QUEUE_THREADS=0`), where the heal lands on the worker.
	#[cfg(not(feature = "merged_object_store"))]
	#[test]
	fn m1_i_a_heal_landing_on_a_deleted_and_reset_2q_key_is_undone_at_its_set() {
		let _serialised = migration_test_lock::lock();

		let (mut worker, objects) = make_worker(PaperPolicy::TwoQCompactHybrid(0.25));

		const K: HashedKey = 1;

		let built = built_by_the_client(&mut worker, &objects, K);
		assert_eq!(built, Slow, "2Q builds a new key slow");
		set(&mut worker, &objects, K, LEN, built);
		drain_and_apply(&mut worker);

		// A hit in the FIFO promotes K to main, and fast.
		worker.handle_get(K, Some(Slow));
		drain_and_apply(&mut worker);
		assert_eq!((placement(&worker, K), bytes_tier(&objects, K)), (Some(Fast), Fast));

		move_bytes(&objects, K, Slow);

		// The client: a hit served slow, then `del(K)`, `set(K)` -- new, so
		// built slow -- before the worker has taken any of it.
		let served = bytes_tier(&objects, K);
		let status = worker.status.clone();
		let overhead_manager = worker.overhead_manager.clone();

		publish_del(&status, &overhead_manager, &objects, K);
		let built = built_by_the_client(&mut worker, &objects, K);
		assert_eq!(built, Slow);
		let published = publish(&status, &overhead_manager, &objects, K, LEN, built);
		assert!(published.fresh());

		// The worker: the hit, healed onto the fresh value.
		worker.handle_get(K, Some(served));
		assert_eq!(correctives(&drain_and_apply(&mut worker), K), vec![Fast], "the heal");
		assert_eq!(bytes_tier(&objects, K), Fast, "it landed on the FRESH value");
		assert_eq!(worker.status.migration_in_flight().pending(K), 0, "and nothing is in flight");

		worker.handle_del(K);
		drain_and_apply(&mut worker);

		handle(&mut worker, K, published);
		assert_eq!(placement(&worker, K), Some(Slow), "admitted to the slow FIFO, where it was built");

		let drain = drain_and_apply(&mut worker);
		assert_eq!(bytes_tier(&objects, K), Slow, "the fresh value is where the stack placed it");
		assert_settled(&mut worker);
		assert_eq!(correctives(&drain, K), vec![Slow], "the new-key rule's corrective");
	}

	/// Review M1 (ii), in both stores: LFU's value v1 is built slow under a
	/// stale latched mirror and admitted FAST by the open stack, so the
	/// reconcile queues a corrective `(K, Fast)` -- and here it waits
	/// behind a parked migration on its consumer. `K` is EVICTED; the stack
	/// latches; v2 is set, built slow, and admitted slow with no push. The
	/// parked consumer is released and the stale corrective lands on v2; only
	/// then does the worker take v2's `Set`: nothing in flight, but the bucket
	/// landed since the mark, so the new-key rule queues `(K, Slow)` last. Red
	/// with the rule off (`nofence`) and with the in-flight count alone
	/// (`pendingonly`).
	#[test]
	fn m1_ii_a_corrective_landing_on_an_evicted_and_reset_lfu_key_is_undone_at_its_set() {
		let _serialised = migration_test_lock::lock();

		let (mut worker, objects) = make_worker(PaperPolicy::LfuCompactHybrid);

		if !queue_or_skip(&worker, "m1_ii_a_corrective_landing_on_an_evicted_and_reset_lfu_key_is_undone_at_its_set") {
			return;
		}

		const K: HashedKey = 1;
		let j = same_consumer(K, 4);

		// J: read often, so K -- read never -- is the key an eviction takes.
		set(&mut worker, &objects, j, LEN, Fast);
		drain_and_apply(&mut worker);
		for _ in 0..3 {
			worker.handle_get(j, Some(Fast));
			drain_and_apply(&mut worker);
		}

		worker.test_flush = false;
		let j_tier = bytes_tier(&objects, j);
		let release = park(&worker, &objects, j);

		// v1: built slow, admitted fast -- one corrective, queued behind J's.
		set(&mut worker, &objects, K, LEN, Slow);
		let drain = drain_and_apply(&mut worker);
		assert_eq!(placement(&worker, K), Some(Fast), "the open stack admits v1 fast");
		assert_eq!(correctives(&drain, K), vec![Fast]);
		assert_eq!(worker.status.migration_in_flight().pending(K), 1, "queued behind the parked one");

		evict_one_key(&mut worker);
		assert_eq!(placement(&worker, K), None, "K, the least frequent, was evicted");
		assert!(placement(&worker, j).is_some(), "J was not");

		// Fresh keys until the stack latches.
		fill(&mut worker, &objects, 100..=131, LEN);
		assert_eq!(placement(&worker, 131), Some(Slow), "latched");

		// v2, built slow: as the client builds a new key under the latch the
		// worker published, in both stores.
		assert_eq!(built_by_the_client(&mut worker, &objects, K), Slow, "the latched mirror builds slow");
		let published = publish(&worker.status, &worker.overhead_manager, &objects, K, LEN, Slow);
		assert!(published.fresh());

		unpark(&worker, j, j_tier, release);
		assert_eq!(bytes_tier(&objects, K), Fast, "the stale corrective promoted the FRESH value");

		handle(&mut worker, K, published);
		assert_eq!(placement(&worker, K), Some(Slow), "admitted slow, latched, with no push");

		let drain = drain_and_apply(&mut worker);

		worker.test_flush = true;
		worker.migration_queue.as_ref().expect("a queue").flush();
		assert_eq!(bytes_tier(&objects, K), Slow, "the fresh value is where the stack placed it");
		assert_settled(&mut worker);
		assert_eq!(correctives(&drain, K), vec![Slow], "the new-key rule's corrective");
	}

	/// Review M1 (iii), in both stores, the rule's IN-FLIGHT path: a stale
	/// STACK promotion. `K` is slow in a latched LFU; a hit promotes it, and
	/// the stack's `(K, Fast)` waits behind a parked migration. The client
	/// deletes `K` and sets it again, built slow, and the worker takes the
	/// `Set` WHILE the promotion is still queued: admitted slow with no push,
	/// but its bucket is in flight, so the new-key rule queues `(K, Slow)`
	/// behind the stale promotion. Released, the promotion lands on the fresh
	/// value and the corrective takes it back. Red with the rule off
	/// (`nofence`); green with the in-flight count alone (`pendingonly`),
	/// which is this path.
	#[test]
	fn m1_iii_a_stale_stack_promotion_of_a_deleted_key_does_not_strand_its_slow_readmission() {
		let _serialised = migration_test_lock::lock();

		let (mut worker, objects) = make_worker(PaperPolicy::LfuCompactHybrid);

		if !queue_or_skip(&worker, "m1_iii_a_stale_stack_promotion_of_a_deleted_key_does_not_strand_its_slow_readmission") {
			return;
		}

		fill(&mut worker, &objects, 1..=32, LEN);

		const K: HashedKey = 32;
		assert_eq!((placement(&worker, K), bytes_tier(&objects, K)), (Some(Slow), Slow), "latched");

		let j = K - 4 * migration_queue::threads() as HashedKey;

		worker.test_flush = false;
		let j_tier = bytes_tier(&objects, j);
		let release = park(&worker, &objects, j);

		// Hits until the stack promotes K; its entry waits behind J's.
		let mut promoted = Vec::new();
		for _ in 0..8 {
			worker.handle_get(K, Some(Slow));
			promoted.extend(of(&drain_and_apply(&mut worker), K));

			if placement(&worker, K) == Some(Fast) {
				break;
			}
		}
		assert_eq!(promoted, vec![Fast], "the stack's promotion, once");
		assert!(worker.status.migration_in_flight().pending(K) >= 1, "queued behind the parked one");

		// The client: `del(K)`, `set(K)`, built slow as the client builds it
		// under the latch the worker published.
		let status = worker.status.clone();
		let overhead_manager = worker.overhead_manager.clone();

		publish_del(&status, &overhead_manager, &objects, K);
		assert_eq!(built_by_the_client(&mut worker, &objects, K), Slow, "the latched mirror builds slow");
		let published = publish(&status, &overhead_manager, &objects, K, LEN, Slow);

		worker.handle_del(K);
		drain_and_apply(&mut worker);

		handle(&mut worker, K, published);
		assert_eq!(placement(&worker, K), Some(Slow), "admitted slow, with no push");
		let drain = drain_and_apply(&mut worker);

		unpark(&worker, j, j_tier, release);

		worker.test_flush = true;
		assert_eq!(bytes_tier(&objects, K), Slow, "the fresh value is where the stack placed it");
		assert_settled(&mut worker);
		assert_eq!(correctives(&drain, K), vec![Slow], "the new-key rule's corrective, behind the stale promotion");
	}

	/// Race (b), in both stores, now fixed at the re-set's `Set` rather than
	/// at the key's first slow hit: `K`'s demotion is queued behind `J`'s,
	/// which is parked; the client deletes `K` and sets it again -- new, built
	/// in DRAM; released, the stale `(K, Slow)` demotes the FRESH value. The
	/// worker takes the `Set` after that: LRU admits `K` fast with no push,
	/// nothing is in flight, and a migration of the bucket landed since the
	/// mark -- the new-key rule queues `(K, Fast)`. Red with the rule off
	/// (`nofence`, which leaves `K` lagging for the heal) and with the
	/// in-flight count alone (`pendingonly`).
	#[test]
	fn race_b_a_stale_demotion_of_a_deleted_and_reset_key_is_undone_at_its_set() {
		let _serialised = migration_test_lock::lock();

		let (mut worker, objects) = make_worker(PaperPolicy::LruCompactHybrid);

		if !queue_or_skip(&worker, "race_b_a_stale_demotion_of_a_deleted_and_reset_key_is_undone_at_its_set") {
			return;
		}

		const J: HashedKey = 64;
		let k = same_consumer(J, 4);

		set(&mut worker, &objects, J, LEN, Fast);
		set(&mut worker, &objects, k, LEN, Fast);
		drain_and_apply(&mut worker);
		assert_eq!((placement(&worker, J), placement(&worker, k)), (Some(Fast), Some(Fast)));

		let status = worker.status.clone();
		let overhead_manager = worker.overhead_manager.clone();

		let (entered, release) = after_copy::arm(J);

		// A settle that demotes J and K: the fast tier shrunk (S5: a value
		// twice the tier, this fixture until then, is placed slow with no
		// settle), then restored once its drain is applied.
		let admitting = std::thread::spawn(move || {
			shrink_fast(&mut worker);
			// Returns once the consumers are done: after the release.
			let drain = drain_and_apply(&mut worker);
			restore_fast(&mut worker);
			(worker, drain)
		});

		entered
			.recv_timeout(Duration::from_secs(10))
			.expect("J's demotion never reached the park point");

		// The client: `del(K)`, then `set(K)` -- new, so built in DRAM.
		publish_del(&status, &overhead_manager, &objects, k);
		let published = publish(&status, &overhead_manager, &objects, k, LEN, Fast);

		release.send(()).unwrap();
		let (mut worker, drain) = admitting.join().unwrap();
		assert_eq!(of(&drain, k), vec![Slow], "K's demotion was queued behind J's");
		assert_eq!(bytes_tier(&objects, k), Slow, "the stale demotion moved the FRESH value");

		// The worker takes the del and the set only now.
		worker.handle_del(k);
		handle(&mut worker, k, published);
		assert_eq!(placement(&worker, k), Some(Fast), "the stack admitted the fresh K fast");

		let drain = drain_and_apply(&mut worker);
		assert_eq!(bytes_tier(&objects, k), Fast, "the fresh value is where the stack placed it");
		assert_settled(&mut worker);
		assert_eq!(correctives(&drain, k), vec![Fast], "the new-key rule's corrective");
	}

	/// The new-key rule's NON-FRESH branch, in the split builds (in the merged
	/// store the racing `Del` retires the DEAD slot the delete left and never
	/// the live one, so it cannot untrack a live key): a `del` on one thread
	/// races two sets on
	/// another and reaches the worker BETWEEN their `Set`s, untracking a key
	/// whose value is live. `K`'s v1 is placed fast. The clients delete v1,
	/// set v2 -- fresh -- and v3 over it, NOT fresh, v3's mark read before
	/// its insert. The worker takes Set(v2), then another client's admission
	/// of twice the fast tier, whose settle demotes `K`: that entry lands on
	/// v3, the value holding the key, after v3's mark (since S5 a shrink of the
	/// fast tier: that value is structural, placed slow with no settle). Then
	/// the `Del`, which
	/// untracks `K` while v3 is live, then Set(v3): LRU re-admits `K` fast,
	/// where v3 was built, with no push, and nothing is in flight. v3's
	/// insert replaced a value, but the stack did not place `K` before the
	/// event (`placement_of` None) and a migration of its bucket landed since
	/// the mark, so the rule queues `(K, Fast)` last and v3 ends where the
	/// stack placed it. Red with that `placement_of` clause removed
	/// (`nonfresh`): `fresh` alone does not fence v3, which is left in CXL,
	/// placed fast. No consumer is parked, so it runs inline too.
	#[cfg(not(feature = "merged_object_store"))]
	#[test]
	fn a_set_that_replaced_a_value_a_racing_del_untracked_is_fenced_like_a_new_key() {
		let _serialised = migration_test_lock::lock();

		let (mut worker, objects) = make_worker(PaperPolicy::LruCompactHybrid);

		const K: HashedKey = 1;

		set(&mut worker, &objects, K, LEN, Fast);
		drain_and_apply(&mut worker);
		assert_eq!((placement(&worker, K), bytes_tier(&objects, K)), (Some(Fast), Fast));

		// The clients: `del(K)` on one thread; `set(K, v2)` and `set(K, v3)`
		// on another, both built in DRAM as LRU admits.
		let status = worker.status.clone();
		let overhead_manager = worker.overhead_manager.clone();

		publish_del(&status, &overhead_manager, &objects, K);
		let v2 = publish(&status, &overhead_manager, &objects, K, LEN, Fast);
		let v3 = publish(&status, &overhead_manager, &objects, K, LEN + 100, Fast);
		assert!(v2.fresh() && !v3.fresh(), "v2 was new to the map, v3 replaced it");

		// The worker: Set(v2), `K` still tracked.
		handle(&mut worker, K, v2);
		drain_and_apply(&mut worker);
		assert_eq!(placement(&worker, K), Some(Fast));

		// A settle demotes `K`: the entry lands on v3. (The fast tier shrunk,
		// then restored: S5 places this fixture's old trigger -- another
		// client's admission of a value twice the tier -- slow, with no
		// settle.)
		shrink_fast(&mut worker);
		let drain = drain_and_apply(&mut worker);
		restore_fast(&mut worker);
		assert_eq!(of(&drain, K), vec![Slow], "the settle demoted K");
		assert_eq!(bytes_tier(&objects, K), Slow, "and the demotion moved v3");

		// The racing `Del`, then Set(v3).
		worker.handle_del(K);
		drain_and_apply(&mut worker);
		assert_eq!(placement(&worker, K), None, "the del untracked K, v3 live");
		assert_eq!(worker.status.migration_in_flight().pending(K), 0, "nothing in flight: the landed path");

		handle(&mut worker, K, v3);
		assert_eq!(placement(&worker, K), Some(Fast), "re-admitted fast, where v3 was built, with no push");
		let observed = worker.observed.clone();

		let drain = drain_and_apply(&mut worker);
		assert_eq!(bytes_tier(&objects, K), Fast, "v3 is where the stack placed it");
		assert_settled(&mut worker);
		assert_eq!(
			observed,
			vec![Observed::Built { key: K, built: Fast, fence: true }],
			"fenced: v3 replaced a value, but the stack did not place K before its Set",
		);
		assert_eq!(correctives(&drain, K), vec![Fast], "the new-key rule's corrective (built where placed)");
	}

	/// Review M2, in both stores: slow-served hits while a promotion of their
	/// key's bucket is in flight queue at most ONE heal. `K` is placed fast
	/// with its bytes slow, and the consumer that owns it is parked: its first
	/// slow hit heals (nothing was in flight), and the heal waits; twenty more
	/// hits, served slow meanwhile, queue nothing. `M` is demoted, and a hit
	/// promotes it -- the stack's own entry, parked the same way -- and twenty
	/// slow hits after it queue no heal at all. Counted off the drains, not
	/// off the process-global `RECONCILE_GET_TO_FAST`, which other tests'
	/// caches move concurrently. Red with the bucket check off (`noquiet`):
	/// 21 and 20. The 40 hits that found their bucket busy are each counted a
	/// skipped heal -- `reconcile_get_heal_skipped`, process-global too, so
	/// asserted as at least 40. Red with that count off (`noskipcount`).
	#[test]
	fn m2_slow_hits_while_their_buckets_promotion_is_in_flight_queue_at_most_one_heal() {
		let _serialised = migration_test_lock::lock();

		let (mut worker, objects) = make_worker(PaperPolicy::LruCompactHybrid);

		if !queue_or_skip(&worker, "m2_slow_hits_while_their_buckets_promotion_is_in_flight_queue_at_most_one_heal") {
			return;
		}

		fill(&mut worker, &objects, 1..=24, LEN);

		const M: HashedKey = 1;
		assert_eq!((placement(&worker, M), bytes_tier(&objects, M)), (Some(Slow), Slow), "demoted");

		let threads = migration_queue::threads() as HashedKey;
		let consumer = |key: HashedKey| key % threads == M % threads;

		let k = (2..=24).rev().find(|&key| consumer(key) && placement(&worker, key) == Some(Fast))
			.expect("a fast key on M's consumer");
		let j = (2..=24).find(|&key| consumer(key) && key != k).expect("a third key on M's consumer");

		worker.test_flush = false;
		let j_tier = bytes_tier(&objects, j);
		let release = park(&worker, &objects, j);

		let skipped_before = worker.status.hybrid_stats().reconcile_get_heal_skipped;

		// K: one heal, then its own heal in flight.
		move_bytes(&objects, k, Slow);

		let mut heals = 0;
		for _ in 0..21 {
			worker.handle_get(k, Some(Slow));
			heals += correctives(&drain_and_apply(&mut worker), k).len();
		}
		assert_eq!(heals, 1, "the first slow hit heals; the rest find it in flight");

		// M: the stack's promotion in flight, and no heal at all.
		worker.handle_get(M, Some(Slow));
		let drain = drain_and_apply(&mut worker);
		assert_eq!((of(&drain, M), correctives(&drain, M)), (vec![Fast], vec![]), "the stack promotes M");

		let mut heals = 0;
		for _ in 0..20 {
			worker.handle_get(M, Some(Slow));
			heals += correctives(&drain_and_apply(&mut worker), M).len();
		}
		assert_eq!(heals, 0, "M's promotion is in flight: it decides");

		let skipped = worker.status.hybrid_stats().reconcile_get_heal_skipped - skipped_before;
		assert!(skipped >= 40, "the 40 slow hits on busy buckets are counted skipped heals: {skipped}");

		unpark(&worker, j, j_tier, release);

		worker.test_flush = true;
		assert_settled(&mut worker);
		assert_eq!((bytes_tier(&objects, k), bytes_tier(&objects, M)), (Fast, Fast));
	}

	/// Review m4, in both stores: a corrective that lands is counted in the
	/// cache's `reconcile_applied_*`, not as a promotion or a demotion. The
	/// heal promotes `K`; a FIFO overwrite built fast on a key placed slow is
	/// demoted by the reconcile (FIFO counts completed slow moves as
	/// demotions: `inline_demotion_accounting`). Red with the tag dropped
	/// (`notag`): the promotion and the demotion are counted.
	#[test]
	fn a_landed_corrective_is_counted_apart_from_promotions_and_demotions() {
		let _serialised = migration_test_lock::lock();

		// To fast: the heal.
		let (mut worker, objects) = make_worker(PaperPolicy::LruCompactHybrid);

		const K: HashedKey = 1;
		set(&mut worker, &objects, K, LEN, Fast);
		drain_and_apply(&mut worker);
		move_bytes(&objects, K, Slow);

		let before = worker.status.hybrid_stats();
		worker.handle_get(K, Some(Slow));
		let drain = drain_and_apply(&mut worker);
		assert_settled(&mut worker);
		let after = worker.status.hybrid_stats();

		assert_eq!(
			(after.promotions - before.promotions, after.reconcile_applied_to_fast - before.reconcile_applied_to_fast),
			(0, 1),
			"the heal landed: (promotions, correctives to fast)",
		);
		assert_eq!((of(&drain, K), correctives(&drain, K)), (vec![Fast], vec![Fast]), "one entry, the heal");

		// To slow: FIFO, an overwrite built fast on a key placed slow.
		let (mut worker, objects) = make_worker(PaperPolicy::FifoCompactHybrid);

		fill(&mut worker, &objects, 1..=24, LEN);
		assert_eq!((placement(&worker, K), bytes_tier(&objects, K)), (Some(Slow), Slow), "demoted");

		let before = worker.status.hybrid_stats();
		set(&mut worker, &objects, K, LEN + 100, Fast);
		let drain = drain_and_apply(&mut worker);
		assert_settled(&mut worker);
		let after = worker.status.hybrid_stats();

		assert_eq!(
			(after.demotions - before.demotions, after.reconcile_applied_to_slow - before.reconcile_applied_to_slow),
			(0, 1),
			"the corrective landed: (demotions, correctives to slow)",
		);
		assert_eq!((of(&drain, K), correctives(&drain, K)), (vec![Slow], vec![Slow]), "one entry, the corrective");
	}

	/// Race (a), promotion side, deterministically, in both stores (LFU keeps
	/// an existing key's PHYSICAL tier on a re-set). A hit promotes slow `K`,
	/// and the promotion is parked after its copy. The client overwrites `K`:
	/// `admission_tier` reads the bytes' tier -- still slow -- and the new
	/// value is built there; released, the parked swap loses to the overwrite.
	/// The stack places `K` fast, its bytes are slow, nothing is queued: the
	/// reconcile of the overwrite's `Set` queues the one promotion that fixes
	/// it. The swap LOSING stands in for a swap landing between the read and
	/// the insert, which no hook can pause: either way the new value is built
	/// from a read the migration has made stale, and ends in the same state.
	/// Red with the reconcile off.
	#[test]
	fn race_a_an_overwrite_that_beats_its_keys_promotion_is_promoted_by_the_reconcile() {
		let _serialised = migration_test_lock::lock();

		let (mut worker, objects) = make_worker(PaperPolicy::LfuCompactHybrid);

		if !queue_or_skip(&worker, "race_a_an_overwrite_that_beats_its_keys_promotion_is_promoted_by_the_reconcile") {
			return;
		}

		fill(&mut worker, &objects, 1..=32, LEN);

		const K: HashedKey = 32;
		assert_eq!((placement(&worker, K), bytes_tier(&objects, K)), (Some(Slow), Slow));

		let status = worker.status.clone();
		let overhead_manager = worker.overhead_manager.clone();

		let (entered, release) = after_copy::arm(K);

		let hit = {
			let objects = objects.clone();

			std::thread::spawn(move || {
				worker.handle_get(K, Some(bytes_tier(&objects, K)));
				let drain = drain_and_apply(&mut worker);
				(worker, drain)
			})
		};

		entered
			.recv_timeout(Duration::from_secs(10))
			.expect("K's promotion never reached the park point");

		let built = admission_tier(PaperPolicy::LfuCompactHybrid, K, &status, &objects);
		assert_eq!(built, Slow, "admission_tier read the bytes' tier before the swap");

		let published = publish(&status, &overhead_manager, &objects, K, LEN + 500, built);

		release.send(()).unwrap();
		let (mut worker, drain) = hit.join().unwrap();

		assert_eq!(of(&drain, K), vec![Fast], "the hit promoted K");
		assert_eq!(bytes_tier(&objects, K), Slow, "its swap lost to the overwrite");
		assert_eq!(placement(&worker, K), Some(Fast));

		handle(&mut worker, K, published);

		assert_eq!(of(&drain_and_apply(&mut worker), K), vec![Fast], "exactly one corrective");
		assert_eq!(bytes_tier(&objects, K), Fast);
		assert_settled(&mut worker);
	}

	/// Race (a), demotion side, in both stores (FIFO keeps an existing key's
	/// physical tier on a re-set): a settle demotes `K` (since S5 a shrink of
	/// the fast tier: the admission of a value twice the tier this used is
	/// placed slow, with no settle), parked after its copy; the client overwrites `K` reading its bytes still fast, and builds
	/// the new value in DRAM; the demotion's swap loses. The stack places `K`
	/// slow with its bytes in DRAM -- stranded -- until the reconcile of the
	/// overwrite's `Set` demotes it. Red with the reconcile off.
	#[test]
	fn race_a_an_overwrite_that_beats_its_keys_demotion_is_demoted_by_the_reconcile() {
		let _serialised = migration_test_lock::lock();

		let (mut worker, objects) = make_worker(PaperPolicy::FifoCompactHybrid);

		if !queue_or_skip(&worker, "race_a_an_overwrite_that_beats_its_keys_demotion_is_demoted_by_the_reconcile") {
			return;
		}

		const K: HashedKey = 1;

		fill(&mut worker, &objects, K..=4, LEN);
		assert_eq!((placement(&worker, K), bytes_tier(&objects, K)), (Some(Fast), Fast));

		let status = worker.status.clone();
		let overhead_manager = worker.overhead_manager.clone();

		let (entered, release) = after_copy::arm(K);

		// A settle that demotes K: the fast tier shrunk, then restored (S5
		// places this fixture's old trigger, a value twice the tier, slow with
		// no settle).
		let admitting = std::thread::spawn(move || {
			shrink_fast(&mut worker);
			let drain = drain_and_apply(&mut worker);
			restore_fast(&mut worker);
			(worker, drain)
		});

		entered
			.recv_timeout(Duration::from_secs(10))
			.expect("K's demotion never reached the park point");

		let built = admission_tier(PaperPolicy::FifoCompactHybrid, K, &status, &objects);
		assert_eq!(built, Fast, "admission_tier read the bytes' tier before the swap");

		let published = publish(&status, &overhead_manager, &objects, K, LEN + 500, built);

		release.send(()).unwrap();
		let (mut worker, drain) = admitting.join().unwrap();

		assert_eq!(of(&drain, K), vec![Slow], "the settle demoted K");
		assert_eq!(bytes_tier(&objects, K), Fast, "its swap lost to the overwrite");
		assert_eq!(placement(&worker, K), Some(Slow));

		handle(&mut worker, K, published);

		assert_eq!(of(&drain_and_apply(&mut worker), K), vec![Slow], "exactly one corrective");
		assert_eq!(bytes_tier(&objects, K), Slow);
		assert_settled(&mut worker);
	}

	/// The heal end to end, in both stores: through a real cache, so the
	/// served tier is the one the cache reads off its snapshot and sends --
	/// through `get` for one key and `get_into` for another, each a producer
	/// of its own. Two values moved to the slow tier behind the stack's back
	/// are seen lagging by `PaperCache::placement_audit`; one `get` of the
	/// first and one `get_into` of the second -- each served from the slow
	/// tier -- and the worker promotes both. Red when either producer reports
	/// its hit served fast (`servedfast_get`, `servedfast_getinto`).
	#[test]
	fn a_get_served_from_the_slow_tier_heals_its_value_through_the_cache() {
		let _serialised = migration_test_lock::lock();

		// The per-object metadata model (S5): at this toy fast tier the MEASURED
		// M of the cache's own structures would leave the strict key ceiling
		// no room, and every new key would fail with `MetadataOverflow`.
		let _per_object = crate::object::overhead::test_overheads::per_object();

		let cache = crate::PaperCache::<u64, TieredBuffer>::new(
			1 << 20,
			crate::CacheTierSize::Bytes(FAST),
			PaperPolicy::LruCompactHybrid,
		)
		.expect("an LRU hybrid cache");

		const K: u64 = 1;
		const L: u64 = 2;
		cache.set(K, &[1u8; LEN], None).expect("set");
		cache.set(L, &[2u8; LEN], None).expect("set");

		let wait = |what: &str, done: &dyn Fn() -> bool| {
			let deadline = std::time::Instant::now() + Duration::from_secs(10);

			while !done() {
				assert!(std::time::Instant::now() < deadline, "{what}");
				std::thread::sleep(Duration::from_millis(1));
			}
		};

		wait("the worker never took the sets", &|| cache.hybrid_stats().fast_objects == 2);
		assert_eq!((cache.tier_of(&K), cache.tier_of(&L)), (Some(Fast), Some(Fast)));

		move_bytes(&cache.objects, cache.hash_key(&K), Slow);
		move_bytes(&cache.objects, cache.hash_key(&L), Slow);

		let audit = cache.placement_audit().expect("an audit");
		assert_eq!((audit.lagging, audit.stranded), (2, 0), "{audit:?}");

		cache.get(&K).expect("a hit");
		let mut out = Vec::new();
		cache.get_into(&L, &mut out).expect("a hit");
		assert_eq!(out, vec![2u8; LEN]);

		wait("the slow-served get was never healed", &|| cache.tier_of(&K) == Some(Fast));
		wait("the slow-served get_into was never healed", &|| cache.tier_of(&L) == Some(Fast));
		assert!(cache.placement_audit().expect("an audit").is_clean());
		assert_eq!(cache.migrations_in_flight(), 0, "and the buckets balance");
	}

	/// The audit's classes and units, in both stores. After a workload that
	/// demoted some keys every key agrees; then one value placed slow is moved
	/// to DRAM and one placed fast to CXL behind the stack's back, and -- in
	/// the split stores, where the map and the stack are two structures -- one
	/// value is put in the map the stack never saw. Each is reported once with
	/// its own charge, and the per-tier totals are the tags' own. Red with
	/// `PlacementAudit::record`'s stranded and lagging arms swapped
	/// (`auditswap`), and in the split builds with its untracked arm dropped
	/// (`auditnountracked`). In the merged store the untracked value is one
	/// the client published and the worker has not linked.
	#[test]
	fn the_audit_reports_each_misplaced_value_once_with_its_charge() {
		let _serialised = migration_test_lock::lock();

		let (mut worker, objects) = make_worker(PaperPolicy::LruCompactHybrid);

		// Distinct lengths, so a value counted in the wrong class shows.
		let lens: Vec<(HashedKey, usize)> =
			(1..=24).map(|key| (key, 300 + 97 * key as usize)).collect();

		for &(key, len) in &lens {
			set(&mut worker, &objects, key, len, Fast);
			drain_and_apply(&mut worker);
		}

		let clean = worker.placement_audit();
		assert!(clean.is_clean(), "{clean:?}");
		assert!(clean.fast > 0 && clean.slow > 0, "the workload kept some fast and demoted some: {clean:?}");

		let charge = |len: usize| value_charge::<u64>(len as u32);

		let stranded = *lens.iter().find(|(key, _)| placement(&worker, *key) == Some(Slow)).unwrap();
		let lagging = *lens.iter().rev().find(|(key, _)| placement(&worker, *key) == Some(Fast)).unwrap();

		move_bytes(&objects, stranded.0, Fast);
		move_bytes(&objects, lagging.0, Slow);

		let untracked = {
			const U: HashedKey = 1_000;
			const U_LEN: usize = 5_000;

			objects.insert(U, Object::new_in(U, &[0u8; U_LEN], Fast, None));
			Some(charge(U_LEN))
		};

		let (mut fast, mut fast_bytes, mut slow, mut slow_bytes) = (0, 0, 0, 0);

		for &(key, len) in &lens {
			match bytes_tier(&objects, key) {
				Fast => (fast, fast_bytes) = (fast + 1, fast_bytes + charge(len)),
				Slow => (slow, slow_bytes) = (slow + 1, slow_bytes + charge(len)),
			}
		}

		if let Some(bytes) = untracked {
			(fast, fast_bytes) = (fast + 1, fast_bytes + bytes);
		}

		assert_eq!(
			worker.placement_audit(),
			PlacementAudit {
				live: lens.len() as u64 + untracked.is_some() as u64,
				fast,
				fast_bytes,
				slow,
				slow_bytes,
				stranded: 1,
				stranded_bytes: charge(stranded.1),
				lagging: 1,
				lagging_bytes: charge(lagging.1),
				untracked: untracked.is_some() as u64,
				untracked_bytes: untracked.unwrap_or(0),
			},
		);
	}
}
