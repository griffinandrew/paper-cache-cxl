/*
 * Copyright (c) Kia Shakiba
 *
 * This source code is licensed under the GNU AGPLv3 license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! Slab-backed split-slow reprieve hybrid: behaviourally identical to
//! [`S3FifoLazyDemotionFastAdmissionSplitSlowReprieveHybridStack`], with one
//! structure where that has five.
//!
//! The stack this compacts keeps FOUR `kwik::HashList`s -- `one_access_queue`,
//! `main_fast`, `slow_head`, `slow_tail` -- each owning its OWN key-to-node
//! index, plus a separate `entries` map holding the 8-byte payload. A key is in
//! exactly one of the four at any moment, which is precisely the condition a
//! single [`CompactQueueSet`] needs: one slab of 16-byte link-only slots, one
//! index whose VALUE carries both the slot number and the payload, and a queue
//! tag inside that payload.
//!
//! ## Four queues, not a boundary marker
//!
//! This is the first stack in the S3-FIFO family to need all four of
//! `compact_queue_set::MAX_QUEUES`. The split slow segment is NOT a cursor into
//! one list -- the predecessor's `slow_midpoint` was exactly that and was
//! deliberately replaced by a real structural boundary (see the baseline's
//! module doc for the negative result that motivated it). `slow_head` and
//! `slow_tail` are two physically distinct FIFO orders with independent byte
//! counters, and `settle_slow_split` moves objects between them one at a time
//! while checking each one's reference bit. Nothing about that is expressible
//! as a marker inside a single queue, so it gets its own slot:
//!
//! ```text
//! Q_ONE_ACCESS  one-access queue (DRAM)  ─┐
//!                                         ├─> Q_FAST  main_fast (DRAM)
//!               promotions ───────────────┘        │ demotion (bit clear)
//!                                                  v
//!                                    Q_SLOW_HEAD  slow_head (PMEM)
//!                                                  │ crossing check
//!                                                  v  (bit clear)
//!                                    Q_SLOW_TAIL  slow_tail (PMEM)
//!                                                  v  (bit clear)
//!                                                evicted
//! ```
//!
//! Both reference-bit checkpoints -- the crossing and the eviction tail -- and
//! the demotion-boundary reprieve share one implementation
//! (`give_second_chance`), exactly as in the baseline.
//!
//! Splitting the slow tier costs nothing in the slab: a key still occupies one
//! slot and one index bucket whichever of the four orders it is threaded into,
//! so the per-object figure is the same 72 bytes every other converted queue
//! stack measures. Moving between segments is an unlink plus a relink -- a
//! handful of `u32` writes -- where the baseline pays a hash-indexed remove
//! from one `HashList` and an insert into another.
//!
//! ## Why the payload stays in the index value
//!
//! `mark_accessed` is the hottest per-get operation in this family: every hit
//! on a key outside the one-access queue does nothing but flip a reference bit,
//! and touches no queue order at all. This variant makes that even more
//! pronounced, since the bit is now read at three points rather than two. With
//! the payload in the slab it would cost a dereference on every such get for
//! nothing; in the index value it is a single probe. Measured, 59.9 ns against
//! 97.4 ns.
//!
//! ## Everything else carries over unchanged
//!
//! * The one-access queue is FAST (`Queue::OneAccess.tier() == Tier::Fast`),
//!   so `fast_bytes_used`/`fast_object_count` count it and admission is a DRAM
//!   write. A promotion out of it therefore emits NO migration -- the bytes
//!   are already DRAM.
//! * Its tail is *reprieved into `slow_head`* rather than evicted
//!   (`settle_one_access`), synchronously from `insert`/`resize`, never through
//!   `evict_one`.
//! * Demotion is lazy and reference-bit gated, under the shared fast-tier
//!   watermarks.
//! * The shared per-tracked-key metadata reservation is split PROPORTIONALLY
//!   between the two independently-capacitied fast segments (`reserved_shares`),
//!   never charged in full to each. The two slow segments carry no capacity of
//!   their own and reserve nothing.
//! * `needs_capacity_eviction` is deliberately NOT overridden, matching the
//!   baseline: one-access pressure is relieved by `settle_one_access`, not by
//!   the eviction loop.

use crate::{
	object::ObjectSize,
	worker::policy::policy_stack::{
		compact_queue_set::CompactQueueSet, narrow_resident, watermarks, CacheSize, HashedKey,
		PolicyStack, Tier,
	},
	PaperPolicy,
};

const Q_ONE_ACCESS: usize = 0;
const Q_FAST: usize = 1;
const Q_SLOW_HEAD: usize = 2;
const Q_SLOW_TAIL: usize = 3;

/// Fraction of the slow tier's bytes `slow_head` is allowed to hold before
/// `settle_slow_split` starts pushing its tail across into `slow_tail`. 0.5
/// puts the boundary at the slow tier's midpoint, exactly as in the baseline.
const SLOW_HEAD_RATIO: f64 = 0.5;

/// Which of the four live orders a key currently sits in. Doubles as the tier
/// tag: with the slow tier physically split there is no longer any need for a
/// separate `Option<Tier>` field alongside a coarser queue tag, which is why
/// this payload has one fewer field than `S3FifoPayload` and still packs to 8.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Queue {
	OneAccess,
	Fast,
	SlowHead,
	SlowTail,
}

impl Queue {
	fn tier(self) -> Tier {
		match self {
			Queue::OneAccess | Queue::Fast => Tier::Fast,
			Queue::SlowHead | Queue::SlowTail => Tier::Slow,
		}
	}

	fn is_slow(self) -> bool {
		matches!(self, Queue::SlowHead | Queue::SlowTail)
	}
}

/// Combined per-key bookkeeping, carried in the index value.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct SplitSlowPayload {
	queue: Queue,
	/// The part of `size` that stays in DRAM in either tier; see `migrating`.
	dram_resident: u8,
	accessed: bool,
	size: ObjectSize,
}

/// Pinned, exactly as `S3FifoEntry` is in the stack this replaces.
const _: () = assert!(
	std::mem::size_of::<SplitSlowPayload>() == 8,
	"SplitSlowPayload grew past 8 bytes",
);

impl SplitSlowPayload {
	/// The bytes that actually move between tiers when this object migrates.
	/// `size` is `base_size`, which also counts the DRAM-resident remainder
	/// (key + expiry field, already inside `shared_overhead`); charging those
	/// to the tier counters double-counted every fast-tier object.
	fn migrating(&self) -> CacheSize {
		(self.size as CacheSize).saturating_sub(self.dram_resident as CacheSize)
	}
}

pub struct S3FifoLazyDemotionFastAdmissionSplitSlowReprieveCompactHybridStack {
	/// All four orders -- one-access, main-fast, slow-head, slow-tail -- over a
	/// single slab. `MAX_QUEUES` is 4, which this uses in full.
	queues: CompactQueueSet<SplitSlowPayload>,

	one_access_ratio: f64,
	one_access_capacity: CacheSize,
	one_access_used: CacheSize,

	fast_capacity: CacheSize,
	fast_used: CacheSize,
	slow_head_used: CacheSize,
	slow_tail_used: CacheSize,

	/// Approximate per-*tracked-key* DRAM cost of the shared structures,
	/// reserved proportionally between the two fast segments' capacities.
	/// `0` unless set via `with_shared_overhead`.
	shared_overhead: CacheSize,

	migrations: Vec<(HashedKey, Tier)>,
}

impl S3FifoLazyDemotionFastAdmissionSplitSlowReprieveCompactHybridStack {
	pub fn new(one_access_ratio: f64, max_size: CacheSize, fast_capacity: CacheSize) -> Self {
		S3FifoLazyDemotionFastAdmissionSplitSlowReprieveCompactHybridStack {
			queues: CompactQueueSet::default(),
			one_access_ratio,
			one_access_capacity: (one_access_ratio * max_size as f64) as CacheSize,
			one_access_used: 0,
			fast_capacity,
			fast_used: 0,
			slow_head_used: 0,
			slow_tail_used: 0,
			shared_overhead: 0,
			migrations: Vec::new(),
		}
	}

	/// Also pre-sizes the slab from the DRAM-budget ceiling, capped -- see
	/// `MAX_PREALLOC_ENTRIES` for why the ceiling alone is not safe.
	pub fn with_shared_overhead(mut self, overhead: CacheSize) -> Self {
		self.shared_overhead = overhead;

		if overhead > 0 {
			let ceiling = (self.fast_capacity / overhead) as usize;
			self.queues.reserve(ceiling.min(super::MAX_PREALLOC_ENTRIES));
		}

		self
	}

	/// The CONFIGURED (pre-reservation) main-fast budget: `fast_capacity` with
	/// the one-access queue's carve-out removed. The proportioning basis for
	/// `reserved_shares`, deliberately not what `settle_fast_tier` settles
	/// against.
	fn main_fast_capacity(&self) -> CacheSize {
		self.fast_capacity.saturating_sub(self.one_access_capacity)
	}

	/// Total DRAM reserved for shared per-object metadata across both tiers.
	/// A key occupies exactly one slab slot plus one index bucket no matter
	/// which of the four orders it is in, so a demotion or a crossing does not
	/// change this value -- which is what makes it loop-invariant inside
	/// `settle_fast_tier` and `settle_one_access`.
	fn reserved_overhead(&self) -> CacheSize {
		self.queues.len() as CacheSize * self.shared_overhead
	}

	/// Splits `reserved_overhead()` proportionally between the two
	/// independently-capacitied fast segments, as `(one_access, main_fast)`.
	/// `u128` intermediate so the product cannot overflow; the remainder goes
	/// to the main segment so the two shares re-sum exactly.
	fn reserved_shares(&self) -> (CacheSize, CacheSize) {
		let reserved = self.reserved_overhead();

		let one_access_capacity = self.one_access_capacity;
		let main_fast_capacity = self.main_fast_capacity();
		let total_capacity = one_access_capacity + main_fast_capacity;

		if total_capacity == 0 {
			return (0, 0);
		}

		let one_access_share = ((reserved as u128 * one_access_capacity as u128)
			/ total_capacity as u128) as CacheSize;
		let main_fast_share = reserved.saturating_sub(one_access_share);

		(one_access_share, main_fast_share)
	}

	/// The one-access queue's byte budget once its share of the shared
	/// metadata reservation is carved out. Settled against by
	/// `settle_one_access`.
	fn effective_one_access_capacity(&self) -> CacheSize {
		self.one_access_capacity.saturating_sub(self.reserved_shares().0)
	}

	/// The main queue's fast-portion byte budget once its share of the shared
	/// metadata reservation is carved out. The watermarks sit on top of this.
	fn effective_main_fast_capacity(&self) -> CacheSize {
		self.main_fast_capacity().saturating_sub(self.reserved_shares().1)
	}

	pub fn tier_of(&self, key: HashedKey) -> Option<Tier> {
		self.queues.payload(key).map(|payload| payload.queue.tier())
	}

	/// Returns `true` if `key` currently sits in the older (`slow_tail`) slow
	/// segment -- i.e. it has already survived a crossing check. Exposed for
	/// tests, exactly as on the baseline.
	pub fn is_in_slow_tail(&self, key: HashedKey) -> bool {
		self.queues.payload(key).map(|payload| payload.queue) == Some(Queue::SlowTail)
	}

	/// `new_resident` refreshes the entry's DRAM-resident remainder: a re-set
	/// can add or drop a TTL, which changes it by the `Expiries` entry's cost.
	fn resize_key(&mut self, key: HashedKey, new_size: ObjectSize, new_resident: u8) {
		let Some(payload) = self.queues.payload_mut(key) else { return };

		let old_migrating = payload.migrating();
		payload.size = new_size;
		payload.dram_resident = new_resident;
		let delta = payload.migrating() as i64 - old_migrating as i64;
		let queue = payload.queue;

		let counter = match queue {
			Queue::OneAccess => &mut self.one_access_used,
			Queue::Fast => &mut self.fast_used,
			Queue::SlowHead => &mut self.slow_head_used,
			Queue::SlowTail => &mut self.slow_tail_used,
		};

		*counter = (*counter as i64 + delta).max(0) as CacheSize;
	}

	fn touch(&mut self, key: HashedKey) {
		match self.queues.payload(key).map(|p| p.queue) {
			Some(Queue::OneAccess) => self.promote_from_one_access(key),

			// Lazy: a hit on any main-queue key only sets the reference bit.
			// It is read at three points -- the demotion boundary
			// (`settle_fast_tier`), the slow-segment crossing
			// (`settle_slow_split`), and the eviction tail (`evict_one`).
			Some(_) => self.mark_accessed(key),

			None => {},
		}
	}

	/// The hottest per-get operation in this family, and the reason the payload
	/// lives in the index value: one probe, no slab access, no queue movement.
	fn mark_accessed(&mut self, key: HashedKey) {
		if let Some(p) = self.queues.payload_mut(key) {
			p.accessed = true;
		}
	}

	/// Moves a re-accessed one-access key to the front of the fast list.
	/// Emits NO migration: the one-access queue is fast-tier here, so the
	/// key's bytes are already physically DRAM.
	fn promote_from_one_access(&mut self, key: HashedKey) {
		let Some(payload) = self.queues.payload(key) else { return };
		let size_bytes = payload.migrating();

		self.queues.move_to_front_of(Q_ONE_ACCESS, Q_FAST, key);
		self.one_access_used = self.one_access_used.saturating_sub(size_bytes);

		if let Some(p) = self.queues.payload_mut(key) {
			p.queue = Queue::Fast;
			p.accessed = false;
		}

		self.fast_used += size_bytes;

		self.settle_fast_tier();
	}

	/// Moves `key` to the front of the fast list and clears its reference bit.
	/// Shared by all three reference-bit check points (demotion boundary,
	/// slow-segment crossing, eviction tail), since all three mean the same
	/// thing: this object was reaccessed, so spare it.
	fn give_second_chance(&mut self, key: HashedKey) {
		let Some(payload) = self.queues.payload(key) else { return };
		let size = payload.migrating();
		let was_slow = payload.queue.is_slow();

		match payload.queue {
			// Only reachable from `evict_one`'s fast-tail fallback (nothing has
			// ever been demoted): reorder within the fast list, no tier change
			// and no byte movement.
			Queue::Fast => {
				self.queues.move_front(Q_FAST, key);
			},

			Queue::SlowHead => {
				self.queues.move_to_front_of(Q_SLOW_HEAD, Q_FAST, key);
				self.slow_head_used = self.slow_head_used.saturating_sub(size);
				self.fast_used += size;
			},

			Queue::SlowTail => {
				self.queues.move_to_front_of(Q_SLOW_TAIL, Q_FAST, key);
				self.slow_tail_used = self.slow_tail_used.saturating_sub(size);
				self.fast_used += size;
			},

			Queue::OneAccess => return,
		}

		if let Some(p) = self.queues.payload_mut(key) {
			p.queue = Queue::Fast;
			p.accessed = false;
		}

		self.settle_fast_tier();

		// Only record a migration when the object genuinely crossed tiers AND
		// survived the settle above (which can demote it straight back out, in
		// which case that call already pushed the correct `Tier::Slow`
		// migration itself). A key that was already Fast needs no migration at
		// all -- a redundant Fast->Fast entry would make `PolicyWorker` rebuild
		// an identical buffer for nothing, and `give_second_chance` fires far
		// more often in this variant (every crossing).
		if was_slow && self.queues.payload(key).map(|p| p.queue) == Some(Queue::Fast) {
			self.migrations.push((key, Tier::Fast));
		}
	}

	/// Demotes oldest-first out of the fast list into `slow_head` under the
	/// shared fast-tier watermarks, reprieving any key whose bit is set instead.
	/// Terminates even when every fast key's bit is set, since each reprieve
	/// clears one bit.
	///
	/// `effective_main_fast_capacity()` is loop-invariant -- a demotion moves a
	/// key from one order to another but leaves it in the index, so
	/// `reserved_shares()` cannot shift underneath the loop -- which is why it
	/// is read once up front.
	///
	/// Deliberately does NOT call `settle_slow_split`: that method calls
	/// `give_second_chance`, which calls back into here, so the two must not be
	/// mutually recursive. `settle_slow_split` is driven from the public trait
	/// methods instead, and its own loop re-checks after any nested demotion.
	fn settle_fast_tier(&mut self) {
		let effective_capacity = self.effective_main_fast_capacity();

		if self.fast_used <= watermarks::high_bytes(effective_capacity) {
			return;
		}

		let low_water = watermarks::low_bytes(effective_capacity);

		while self.fast_used > low_water {
			let Some(candidate) = self.queues.back(Q_FAST) else { break };

			let accessed = self.queues.payload(candidate).map(|p| p.accessed).unwrap_or(false);

			if accessed {
				self.queues.move_front(Q_FAST, candidate);

				if let Some(p) = self.queues.payload_mut(candidate) {
					p.accessed = false;
				}

				continue;
			}

			let size = self.queues.payload(candidate).map(|p| p.migrating()).unwrap_or(0);

			self.queues.move_to_front_of(Q_FAST, Q_SLOW_HEAD, candidate);

			if let Some(p) = self.queues.payload_mut(candidate) {
				p.queue = Queue::SlowHead;
			}

			self.fast_used = self.fast_used.saturating_sub(size);
			self.slow_head_used += size;

			self.migrations.push((candidate, Tier::Slow));
		}
	}

	/// Holds `slow_head` to at most `SLOW_HEAD_RATIO` of the slow tier's bytes,
	/// and -- the point of this variant -- checks each object's reference bit at
	/// the moment it would cross into `slow_tail`. A set bit means the object
	/// was reaccessed since it was demoted, so it goes back to the front of the
	/// fast list instead of crossing.
	///
	/// Termination: each iteration either moves an object across (strictly
	/// reducing `slow_head_used`) or promotes it out of `slow_head` entirely. A
	/// nested `settle_fast_tier` inside `give_second_chance` can push bytes back
	/// into `slow_head`, but only for keys whose bit is clear, and a clear-bit
	/// key at `slow_head`'s back always crosses on the following iteration.
	fn settle_slow_split(&mut self) {
		loop {
			let total = self.slow_head_used + self.slow_tail_used;

			if total == 0 || (self.slow_head_used as f64) <= total as f64 * SLOW_HEAD_RATIO {
				break;
			}

			let Some(candidate) = self.queues.back(Q_SLOW_HEAD) else { break };

			let accessed = self.queues.payload(candidate).map(|p| p.accessed).unwrap_or(false);

			if accessed {
				self.give_second_chance(candidate);
				continue;
			}

			let size = self.queues.payload(candidate).map(|p| p.migrating()).unwrap_or(0);

			self.queues.move_to_front_of(Q_SLOW_HEAD, Q_SLOW_TAIL, candidate);

			if let Some(p) = self.queues.payload_mut(candidate) {
				p.queue = Queue::SlowTail;
			}

			self.slow_head_used = self.slow_head_used.saturating_sub(size);
			self.slow_tail_used += size;

			// No migration: both segments are the slow tier, so the bytes do
			// not move between DRAM and PMEM.
		}
	}

	/// Relieves one-access-queue pressure by moving its tail(s) to the front of
	/// `slow_head` -- the fast/slow boundary. Called synchronously from
	/// `insert()`/`resize()`, never through `evict_one()`: nothing is removed
	/// from the cache here, and routing it through eviction would make
	/// `apply_evictions` erase a live object.
	///
	/// No watermarks here, deliberately: this segment relieves pressure
	/// synchronously rather than through a `PolicyWorker` migration batch, so
	/// there is no batch-of-one cost for a low-water drain to amortise away.
	fn settle_one_access(&mut self) {
		let effective_capacity = self.effective_one_access_capacity();

		while self.one_access_used > effective_capacity {
			let Some(key) = self.queues.back(Q_ONE_ACCESS) else { break };
			let Some(payload) = self.queues.payload(key) else { break };
			let size = payload.migrating();

			self.one_access_used = self.one_access_used.saturating_sub(size);

			self.queues.move_to_front_of(Q_ONE_ACCESS, Q_SLOW_HEAD, key);

			if let Some(p) = self.queues.payload_mut(key) {
				p.queue = Queue::SlowHead;
				p.accessed = false;
			}

			self.slow_head_used += size;

			self.migrations.push((key, Tier::Slow));
		}
	}
}

impl PolicyStack for S3FifoLazyDemotionFastAdmissionSplitSlowReprieveCompactHybridStack {
	fn is_policy(&self, policy: &PaperPolicy) -> bool {
		matches!(policy, PaperPolicy::S3FifoLazyDemotionFastAdmissionSplitSlowReprieveCompactHybrid(ratio) if *ratio == self.one_access_ratio)
	}

	fn len(&self) -> usize {
		self.queues.len()
	}

	fn contains(&self, key: HashedKey) -> bool {
		self.queues.contains(key)
	}

	fn insert(&mut self, key: HashedKey, size: ObjectSize) {
		self.insert_resident(key, size, 0);
	}

	fn insert_resident(&mut self, key: HashedKey, size: ObjectSize, dram_resident: ObjectSize) {
		let dram_resident = narrow_resident(dram_resident);

		if self.queues.contains(key) {
			self.resize_key(key, size, dram_resident);
			self.touch(key);
			self.settle_slow_split();
			return;
		}

		self.queues.push_front(
			Q_ONE_ACCESS,
			key,
			SplitSlowPayload {
				queue: Queue::OneAccess,
				dram_resident,
				accessed: false,
				size,
			},
		);
		self.one_access_used += (size as CacheSize).saturating_sub(dram_resident as CacheSize);

		self.settle_one_access();
		self.settle_slow_split();
	}

	fn update(&mut self, key: HashedKey) {
		if self.queues.contains(key) {
			self.touch(key);
			self.settle_slow_split();
		}
	}

	fn remove(&mut self, key: HashedKey) {
		let Some(payload) = self.queues.payload(key) else { return };
		let size = payload.migrating();

		match payload.queue {
			Queue::OneAccess => {
				self.queues.remove(Q_ONE_ACCESS, key);
				self.one_access_used = self.one_access_used.saturating_sub(size);
			},

			Queue::Fast => {
				self.queues.remove(Q_FAST, key);
				self.fast_used = self.fast_used.saturating_sub(size);
			},

			Queue::SlowHead => {
				self.queues.remove(Q_SLOW_HEAD, key);
				self.slow_head_used = self.slow_head_used.saturating_sub(size);
			},

			Queue::SlowTail => {
				self.queues.remove(Q_SLOW_TAIL, key);
				self.slow_tail_used = self.slow_tail_used.saturating_sub(size);
			},
		}
	}

	fn resize(&mut self, max_size: CacheSize) {
		self.one_access_capacity = (self.one_access_ratio * max_size as f64) as CacheSize;
		self.settle_one_access();
		self.settle_fast_tier();
		self.settle_slow_split();
	}

	fn clear(&mut self) {
		self.queues.clear();

		self.one_access_used = 0;
		self.fast_used = 0;
		self.slow_head_used = 0;
		self.slow_tail_used = 0;
		self.migrations.clear();
	}

	fn evict_one(&mut self) -> Option<HashedKey> {
		// Crossing checks fire here, keeping the split balanced before the tail
		// is evaluated. The one-access queue is never consulted -- its pressure
		// is relieved synchronously by `settle_one_access`.
		self.settle_slow_split();

		loop {
			// The oldest slow object is the real candidate; fall back through
			// slow_head, then the fast tail, only when the older orders are
			// empty (i.e. little or nothing has been demoted).
			let (key, from) = if let Some(key) = self.queues.back(Q_SLOW_TAIL) {
				(key, Queue::SlowTail)
			} else if let Some(key) = self.queues.back(Q_SLOW_HEAD) {
				(key, Queue::SlowHead)
			} else {
				(self.queues.back(Q_FAST)?, Queue::Fast)
			};

			let accessed = self.queues.payload(key).map(|p| p.accessed).unwrap_or(false);

			if accessed {
				self.give_second_chance(key);
				continue;
			}

			let size = self.queues.payload(key).map(|p| p.migrating()).unwrap_or(0);

			match from {
				Queue::SlowTail => {
					self.queues.remove(Q_SLOW_TAIL, key);
					self.slow_tail_used = self.slow_tail_used.saturating_sub(size);
				},

				Queue::SlowHead => {
					self.queues.remove(Q_SLOW_HEAD, key);
					self.slow_head_used = self.slow_head_used.saturating_sub(size);
				},

				Queue::Fast => {
					self.queues.remove(Q_FAST, key);
					self.fast_used = self.fast_used.saturating_sub(size);
				},

				Queue::OneAccess => break,
			}

			return Some(key);
		}

		None
	}

	fn resize_fast_tier(&mut self, size: CacheSize) {
		self.fast_capacity = size;
		self.settle_fast_tier();
		self.settle_slow_split();
	}

	fn drain_tier_migrations(&mut self) -> Vec<(HashedKey, Tier)> {
		std::mem::take(&mut self.migrations)
	}

	fn dram_reserved_bytes(&self) -> CacheSize {
		self.reserved_overhead()
	}

	fn fast_bytes_used(&self) -> CacheSize {
		self.fast_used + self.one_access_used
	}

	fn slow_bytes_used(&self) -> CacheSize {
		self.slow_head_used + self.slow_tail_used
	}

	fn fast_object_count(&self) -> usize {
		self.queues.queue_len(Q_FAST) + self.queues.queue_len(Q_ONE_ACCESS)
	}

	fn slow_object_count(&self) -> usize {
		self.queues.queue_len(Q_SLOW_HEAD) + self.queues.queue_len(Q_SLOW_TAIL)
	}
}

/// Fidelity against `S3FifoLazyDemotionFastAdmissionSplitSlowReprieveHybridStack`,
/// which this stack is a compaction of.
#[cfg(all(test, feature = "s3_fifo_lazy_demotion_fast_admission_split_slow_reprieve_hybrid_cache"))]
mod fidelity_tests {
	use super::*;
	use crate::worker::policy::policy_stack::s3_fifo_lazy_demotion_fast_admission_split_slow_reprieve_hybrid_stack::S3FifoLazyDemotionFastAdmissionSplitSlowReprieveHybridStack;

	type Baseline = S3FifoLazyDemotionFastAdmissionSplitSlowReprieveHybridStack;
	type Compact = S3FifoLazyDemotionFastAdmissionSplitSlowReprieveCompactHybridStack;

	const MAX: CacheSize = 1_000_000;

	/// Repeats are essential: a key only leaves the one-access queue on a
	/// SECOND access, and the reference bit only matters on a third, so a
	/// workload without reuse would exercise neither the demotion reprieve, the
	/// crossing checkpoint, nor the eviction-tail second chance.
	fn skewed_ops() -> Vec<(HashedKey, ObjectSize)> {
		let mut ops = Vec::new();
		let mut x: u64 = 0x243F_6A88_85A3_08D3;
		for _ in 0..20_000 {
			x ^= x << 13;
			x ^= x >> 7;
			x ^= x << 17;
			let u = (x >> 11) as f64 / (1u64 << 53) as f64;
			ops.push((((u * u * 200.0) as u64) + 1, 1024));
		}
		ops
	}

	/// Every observable gauge, so a divergence in accounting cannot hide behind
	/// a matching migration list.
	fn gauges(a: &Baseline, b: &Compact) {
		assert_eq!(a.len(), b.len(), "lengths diverge");
		assert_eq!(a.fast_bytes_used(), b.fast_bytes_used(), "fast bytes diverge");
		assert_eq!(a.slow_bytes_used(), b.slow_bytes_used(), "slow bytes diverge");
		assert_eq!(a.fast_object_count(), b.fast_object_count(), "fast count diverges");
		assert_eq!(a.slow_object_count(), b.slow_object_count(), "slow count diverges");
		assert_eq!(a.dram_reserved_bytes(), b.dram_reserved_bytes(), "reservation diverges");
		assert_eq!(
			a.needs_capacity_eviction(),
			b.needs_capacity_eviction(),
			"capacity-eviction trigger diverges",
		);
	}

	/// Identical migration sequence AND order, identical tiers, and identical
	/// slow-segment membership, across several capacities and reservations.
	#[test]
	fn matches_the_baseline_migration_for_migration() {
		let ops = skewed_ops();

		for ratio in [0.0f64, 0.1, 0.25, 0.5, 1.0] {
			for fast in [8_192u64, 131_072] {
				for overhead in [0u64, 112] {
					let mut a = Baseline::new(ratio, MAX, fast).with_shared_overhead(overhead);
					let mut b = Compact::new(ratio, MAX, fast).with_shared_overhead(overhead);
					let (mut ma, mut mb) = (Vec::new(), Vec::new());

					for (k, size) in &ops {
						if a.contains(*k) { a.update(*k); } else { a.insert(*k, *size); }
						if b.contains(*k) { b.update(*k); } else { b.insert(*k, *size); }
						ma.extend(a.drain_tier_migrations());
						mb.extend(b.drain_tier_migrations());
					}

					assert_eq!(
						ma, mb,
						"migrations diverge at ratio {ratio} fast {fast} overhead {overhead}",
					);
					gauges(&a, &b);

					for k in 1..=200u64 {
						assert_eq!(
							a.tier_of(k), b.tier_of(k),
							"tier of {k} diverges at ratio {ratio} fast {fast} overhead {overhead}",
						);
						assert_eq!(
							a.is_in_slow_tail(k), b.is_in_slow_tail(k),
							"slow segment of {k} diverges at ratio {ratio} fast {fast} overhead {overhead}",
						);
					}
				}
			}
		}
	}

	/// Eviction is where the reference bit is acted on for the third time: an
	/// accessed key at the slow tail is promoted back to the fast list instead
	/// of evicted, which reorders three of the four queues mid-eviction.
	#[test]
	fn evicts_in_the_same_order_including_second_chances() {
		let ops = skewed_ops();

		for (ratio, fast) in [(0.1f64, 32_768u64), (0.25, 8_192), (0.5, 131_072)] {
			let mut a = Baseline::new(ratio, MAX, fast).with_shared_overhead(112);
			let mut b = Compact::new(ratio, MAX, fast).with_shared_overhead(112);

			for (k, size) in &ops {
				if a.contains(*k) { a.update(*k); } else { a.insert(*k, *size); }
				if b.contains(*k) { b.update(*k); } else { b.insert(*k, *size); }
				a.drain_tier_migrations();
				b.drain_tier_migrations();
			}

			let (mut ea, mut eb) = (Vec::new(), Vec::new());
			let (mut ma, mut mb) = (Vec::new(), Vec::new());

			while let Some(k) = a.evict_one() {
				ea.push(k);
				ma.extend(a.drain_tier_migrations());
			}
			while let Some(k) = b.evict_one() {
				eb.push(k);
				mb.extend(b.drain_tier_migrations());
			}

			assert_eq!(ea, eb, "eviction order diverges at ratio {ratio} fast {fast}");
			assert_eq!(ma, mb, "eviction-time migrations diverge at ratio {ratio} fast {fast}");
			// The one-access queue is never an eviction source here, so what is
			// left over is exactly what the baseline leaves over.
			gauges(&a, &b);
		}
	}

	#[test]
	fn removal_matches_across_all_four_queues() {
		let ops = skewed_ops();
		let mut a = Baseline::new(0.25, MAX, 32_768).with_shared_overhead(112);
		let mut b = Compact::new(0.25, MAX, 32_768).with_shared_overhead(112);
		let (mut ma, mut mb) = (Vec::new(), Vec::new());

		for (i, (k, size)) in ops.iter().enumerate() {
			if a.contains(*k) { a.update(*k); } else { a.insert(*k, *size); }
			if b.contains(*k) { b.update(*k); } else { b.insert(*k, *size); }

			if i % 97 == 0 {
				let victim = (i as u64 % 200) + 1;
				a.remove(victim);
				b.remove(victim);
			}

			ma.extend(a.drain_tier_migrations());
			mb.extend(b.drain_tier_migrations());
		}

		assert_eq!(ma, mb, "migrations diverge under removal");
		gauges(&a, &b);

		for k in 1..=200u64 {
			assert_eq!(a.tier_of(k), b.tier_of(k), "tier of {k} diverges under removal");
			assert_eq!(a.is_in_slow_tail(k), b.is_in_slow_tail(k), "segment of {k} diverges");
		}
	}

	/// Resize in both directions with BRAND-NEW keys afterwards. The shape is
	/// load-bearing: the equivalent test on the LFU conversion passed with a
	/// real bug present because the workload had no new keys after the resize.
	/// `resize` here re-settles all three segments, which `resize_fast_tier`
	/// does not do in the same order.
	#[test]
	fn resizes_like_the_baseline() {
		for (start, resized) in [(65_536u64, 65_536u64), (131_072, 32_768), (32_768, 131_072)] {
			let mut a = Baseline::new(0.25, MAX, start).with_shared_overhead(112);
			let mut b = Compact::new(0.25, MAX, start).with_shared_overhead(112);
			let (mut ma, mut mb) = (Vec::new(), Vec::new());

			for i in 0..4_000u64 {
				let k = (i % 200) + 1;
				if a.contains(k) { a.update(k); } else { a.insert(k, 1024); }
				if b.contains(k) { b.update(k); } else { b.insert(k, 1024); }
				ma.extend(a.drain_tier_migrations());
				mb.extend(b.drain_tier_migrations());
			}

			a.resize_fast_tier(resized);
			b.resize_fast_tier(resized);
			a.resize(MAX / 2);
			b.resize(MAX / 2);
			ma.extend(a.drain_tier_migrations());
			mb.extend(b.drain_tier_migrations());
			gauges(&a, &b);

			for i in 0..2_000u64 {
				let k = 10_000 + i;
				a.insert(k, 1024);
				b.insert(k, 1024);
				ma.extend(a.drain_tier_migrations());
				mb.extend(b.drain_tier_migrations());
			}

			assert_eq!(ma, mb, "migrations diverge resizing {start} -> {resized}");
			gauges(&a, &b);

			for i in 0..2_000u64 {
				assert_eq!(a.tier_of(10_000 + i), b.tier_of(10_000 + i));
				assert_eq!(a.is_in_slow_tail(10_000 + i), b.is_in_slow_tail(10_000 + i));
			}
		}
	}

	/// The registration surface, which a copied parser gets subtly wrong: the
	/// policy string is the baseline's with `compact` spliced in, it round
	/// trips, `is_hybrid` claims it, the stack answers to it and NOT to the
	/// baseline's variant, and -- because this is a reprieve design that sizes
	/// no queue at `1 - ratio` -- it keeps the baseline's INCLUSIVE upper
	/// bound rather than the s3-fifo family's exclusive one.
	#[test]
	fn the_policy_string_round_trips_and_keeps_the_baselines_bound() {
		for ratio in [0.0f64, 0.1, 1.0] {
			let base = PaperPolicy::S3FifoLazyDemotionFastAdmissionSplitSlowReprieveHybrid(ratio);
			let compact =
				PaperPolicy::S3FifoLazyDemotionFastAdmissionSplitSlowReprieveCompactHybrid(ratio);

			assert_eq!(
				base.to_string().replace("-hybrid-", "-compact-hybrid-"),
				compact.to_string(),
				"the compact policy string must be the baseline's with `compact` spliced in",
			);
			assert_eq!(
				compact.to_string().parse::<PaperPolicy>(),
				Ok(compact),
				"the compact policy string must round trip",
			);
			assert!(compact.is_hybrid(), "the compact policy must be a recognised hybrid design");

			let stack = Compact::new(ratio, MAX, 1_000);
			assert!(stack.is_policy(&compact), "the stack must answer to its own variant");
			assert!(!stack.is_policy(&base), "and must NOT answer to the baseline's variant");
		}

		assert!(
			"s3-fifo-lazy-demotion-fast-admission-split-slow-reprieve-compact-hybrid-1.5"
				.parse::<PaperPolicy>()
				.is_err(),
			"a ratio above 1 must still be rejected",
		);
	}

	// ── the signature new mechanic: a real crossing checkpoint ─────────────

	/// Three 10-byte keys admitted at `ratio == 0.0`, so `settle_one_access`
	/// fires from inside `insert()` itself and each lands in `slow_head` with a
	/// CLEAR reference bit. `slow_head` is held to half the slow tier's bytes,
	/// so 3 x 10 settles at slow_head = [3] / slow_tail = [2, 1].
	fn build_three_slow_keys() -> (Baseline, Compact) {
		let mut a = Baseline::new(0.0, MAX, 0);
		let mut b = Compact::new(0.0, MAX, 0);

		for key in 1..=3u64 {
			a.insert(key, 10);
			b.insert(key, 10);
		}

		a.drain_tier_migrations();
		b.drain_tier_migrations();
		(a, b)
	}

	/// Pins the split itself. Without `settle_slow_split` every slow key would
	/// stay in `slow_head` and every `is_in_slow_tail` below would be false.
	#[test]
	fn the_split_pushes_the_older_slow_keys_into_the_tail_segment() {
		let (a, b) = build_three_slow_keys();

		for k in 1..=3u64 {
			assert_eq!(a.tier_of(k), Some(Tier::Slow));
			assert_eq!(b.tier_of(k), a.tier_of(k));
			assert_eq!(b.is_in_slow_tail(k), a.is_in_slow_tail(k), "segment of {k} diverges");
		}

		assert!(b.is_in_slow_tail(1), "oldest should have crossed into the tail segment");
		assert!(b.is_in_slow_tail(2), "second-oldest should have crossed too");
		assert!(!b.is_in_slow_tail(3), "newest should still be in the head segment");
		assert_eq!(b.slow_bytes_used(), 30);
		gauges(&a, &b);
	}

	/// Pins the crossing CHECKPOINT -- the mechanic this variant exists for. A
	/// reaccessed key at `slow_head`'s back is promoted straight back to the
	/// fast list instead of crossing, and that promotion is a real Slow->Fast
	/// migration. Would fail outright if `settle_slow_split`'s reference-bit
	/// branch had not been ported.
	#[test]
	fn a_reaccessed_key_is_promoted_at_the_crossing_instead_of_crossing() {
		let (mut a, mut b) = build_three_slow_keys();
		assert!(!b.is_in_slow_tail(3));

		// Reaccess key 3 -- lazily, so nothing moves yet.
		a.update(3);
		b.update(3);
		assert_eq!(a.tier_of(3), Some(Tier::Slow), "a mere access must not itself migrate");
		assert_eq!(b.tier_of(3), a.tier_of(3));
		a.drain_tier_migrations();
		b.drain_tier_migrations();

		// Give the fast tier somewhere to promote into (it was 0 in the
		// fixture, which would have demoted the key straight back out).
		a.resize_fast_tier(100);
		b.resize_fast_tier(100);

		// At 4 keys the split is exactly balanced, at 5 it tips: key 3 becomes
		// the crossing candidate with its bit set.
		a.insert(4, 10);
		b.insert(4, 10);
		a.insert(5, 10);
		b.insert(5, 10);

		assert_eq!(
			a.tier_of(3), Some(Tier::Fast),
			"baseline: the reaccessed key is promoted at the crossing check",
		);
		assert_eq!(b.tier_of(3), a.tier_of(3));
		assert!(!b.is_in_slow_tail(3), "and must not have crossed into the tail segment");

		let ma = a.drain_tier_migrations();
		let mb = b.drain_tier_migrations();
		assert_eq!(ma, mb, "crossing-checkpoint migrations diverge");
		assert!(mb.contains(&(3, Tier::Fast)), "a real Slow->Fast migration must be recorded");
		gauges(&a, &b);
	}

	/// The complement: a clear-bit key crosses rather than being promoted.
	#[test]
	fn an_unaccessed_key_crosses_normally() {
		let (mut a, mut b) = build_three_slow_keys();
		assert!(!b.is_in_slow_tail(3));

		a.insert(4, 10);
		b.insert(4, 10);
		a.insert(5, 10);
		b.insert(5, 10);
		a.drain_tier_migrations();
		b.drain_tier_migrations();

		assert!(a.is_in_slow_tail(3), "baseline: an unaccessed key must cross");
		assert!(b.is_in_slow_tail(3), "an unaccessed key must cross, not be promoted");
		assert_eq!(b.tier_of(3), Some(Tier::Slow));
		gauges(&a, &b);
	}

	/// `remove()` is the one public entry point that deliberately does NOT
	/// re-balance the split, so it can leave `slow_head` over its ratio. Both
	/// `resize_fast_tier` and `resize` must settle it on the way out -- the
	/// baseline calls `settle_slow_split` last in each, and dropping either
	/// call leaves the two stacks in different segment states.
	#[test]
	fn resize_fast_tier_rebalances_a_split_left_skewed_by_a_removal() {
		let (mut a, mut b) = build_three_slow_keys();

		// slow_head = [3] (10 B), slow_tail = [2, 1] (20 B). Emptying the tail
		// segment leaves the head holding 100% of the slow bytes.
		for victim in [1u64, 2] {
			a.remove(victim);
			b.remove(victim);
		}
		assert!(!a.is_in_slow_tail(3), "baseline: removal must not itself re-balance");
		assert!(!b.is_in_slow_tail(3), "removal must not itself re-balance");
		assert_eq!(a.slow_bytes_used(), 10);
		assert_eq!(b.slow_bytes_used(), a.slow_bytes_used());

		a.resize_fast_tier(50);
		b.resize_fast_tier(50);

		assert!(a.is_in_slow_tail(3), "baseline: resize_fast_tier re-balances the split");
		assert!(b.is_in_slow_tail(3), "resize_fast_tier must re-balance the split");
		assert_eq!(a.drain_tier_migrations(), b.drain_tier_migrations());
		gauges(&a, &b);
	}

	/// The same obligation on `resize`, which settles all three segments and
	/// ends with the split.
	#[test]
	fn resize_rebalances_a_split_left_skewed_by_a_removal() {
		let (mut a, mut b) = build_three_slow_keys();

		for victim in [1u64, 2] {
			a.remove(victim);
			b.remove(victim);
		}
		assert!(!a.is_in_slow_tail(3));
		assert!(!b.is_in_slow_tail(3));

		a.resize(MAX);
		b.resize(MAX);

		assert!(a.is_in_slow_tail(3), "baseline: resize re-balances the split");
		assert!(b.is_in_slow_tail(3), "resize must re-balance the split");
		assert_eq!(a.drain_tier_migrations(), b.drain_tier_migrations());
		gauges(&a, &b);
	}

	/// A re-`insert` of an existing key resizes its byte counter, which can
	/// skew the split on its own, so that path settles it too before
	/// returning. Here the re-set also sets the key's reference bit, so the
	/// crossing check promotes it rather than pushing it across.
	#[test]
	fn a_re_set_that_skews_the_split_settles_it_before_returning() {
		let (mut a, mut b) = build_three_slow_keys();

		// Somewhere to promote into; the split is balanced, so this settles
		// nothing.
		a.resize_fast_tier(1_000);
		b.resize_fast_tier(1_000);
		a.drain_tier_migrations();
		b.drain_tier_migrations();
		assert_eq!(a.tier_of(3), Some(Tier::Slow));
		assert!(!a.is_in_slow_tail(3));

		// slow_head = [3] at 10 B, slow_tail = [2, 1] at 20 B. Re-setting key
		// 3 at 100 B puts the head segment at 100 of 120 slow bytes.
		a.insert(3, 100);
		b.insert(3, 100);

		assert_eq!(
			a.tier_of(3), Some(Tier::Fast),
			"baseline: the re-set settles the split, and the crossing check promotes key 3",
		);
		assert_eq!(b.tier_of(3), a.tier_of(3));

		let (ma, mb) = (a.drain_tier_migrations(), b.drain_tier_migrations());
		assert_eq!(ma, mb, "re-set migrations diverge");
		assert!(mb.contains(&(3, Tier::Fast)));
		gauges(&a, &b);
	}

	/// A re-set of a key in the OLDER slow segment must charge its byte delta
	/// to `slow_tail`, not to `slow_head`. The two counters are exactly what
	/// the ratio is computed from, so mixing them silently moves the boundary
	/// without moving any object.
	#[test]
	fn a_re_set_of_a_slow_tail_key_is_charged_to_the_tail_segment() {
		let (mut a, mut b) = build_three_slow_keys();

		a.resize_fast_tier(1_000);
		b.resize_fast_tier(1_000);
		a.drain_tier_migrations();
		b.drain_tier_migrations();

		// slow_head = [3] at 10 B, slow_tail = [2, 1] at 20 B. Growing key 1 --
		// the tail segment's back -- leaves the head segment at 10 of 120 slow
		// bytes, far under the ratio, so nothing may cross.
		a.insert(1, 100);
		b.insert(1, 100);

		assert!(a.is_in_slow_tail(1));
		assert!(b.is_in_slow_tail(1));
		assert!(
			!a.is_in_slow_tail(3),
			"baseline: the head segment must not have been pushed across",
		);
		assert!(
			!b.is_in_slow_tail(3),
			"the byte delta belongs to slow_tail, so the head segment must not move",
		);
		assert_eq!(a.slow_bytes_used(), 120);
		assert_eq!(b.slow_bytes_used(), a.slow_bytes_used());
		assert_eq!(a.drain_tier_migrations(), b.drain_tier_migrations());
		gauges(&a, &b);
	}

	/// `evict_one` runs the crossing check BEFORE evaluating the tail. That
	/// ordering is observable only once a `remove()` has skewed the split: the
	/// crossing then promotes a re-accessed `slow_head` key the tail check
	/// would never have looked at.
	#[test]
	fn evict_one_runs_the_crossing_check_before_the_tail() {
		let mut a = Baseline::new(0.0, MAX, 1_000);
		let mut b = Compact::new(0.0, MAX, 1_000);

		// ratio 0.0: every insert is reprieved into slow_head immediately, and
		// the split settles to slow_head = [5, 4] / slow_tail = [3, 2, 1].
		for k in 1..=5u64 {
			a.insert(k, 10);
			b.insert(k, 10);
		}

		// Set the reference bit on slow_head's BACK. The split is balanced
		// here, so nothing moves yet.
		a.update(4);
		b.update(4);
		assert_eq!(a.tier_of(4), Some(Tier::Slow), "a mere access must not migrate");
		assert_eq!(b.tier_of(4), a.tier_of(4));
		assert!(!a.is_in_slow_tail(4), "key 4 must still be in the head segment");
		assert!(!b.is_in_slow_tail(4));

		// Drain the tail segment without settling -- `remove` deliberately does
		// not re-balance -- leaving slow_head past its ratio.
		for victim in [1u64, 2] {
			a.remove(victim);
			b.remove(victim);
		}
		a.drain_tier_migrations();
		b.drain_tier_migrations();

		let (ea, eb) = (a.evict_one(), b.evict_one());
		assert_eq!(ea, eb, "eviction victim diverges");
		assert_eq!(ea, Some(3), "the oldest slow-tail key is still the victim");

		assert_eq!(
			a.tier_of(4), Some(Tier::Fast),
			"baseline: the crossing check promoted key 4 on the way in",
		);
		assert_eq!(b.tier_of(4), a.tier_of(4));

		let (ma, mb) = (a.drain_tier_migrations(), b.drain_tier_migrations());
		assert_eq!(ma, mb, "crossing-before-tail migrations diverge");
		assert!(mb.contains(&(4, Tier::Fast)));
		gauges(&a, &b);
	}

	/// Terminal eviction takes the slow TAIL, and still honours the reference
	/// bit there.
	#[test]
	fn eviction_takes_the_slow_tail_and_still_honors_the_reference_bit() {
		let (mut a, mut b) = build_three_slow_keys();

		assert_eq!(a.evict_one(), Some(1));
		assert_eq!(b.evict_one(), Some(1));
		assert!(!b.contains(1));

		a.update(2);
		b.update(2);
		a.resize_fast_tier(100);
		b.resize_fast_tier(100);

		let ea = a.evict_one();
		let eb = b.evict_one();

		assert_eq!(ea, eb, "eviction victim diverges after a tail second chance");
		assert_eq!(a.tier_of(2), Some(Tier::Fast), "baseline: an accessed tail key is promoted");
		assert_eq!(b.tier_of(2), a.tier_of(2));
		assert_ne!(eb, Some(2));
		gauges(&a, &b);
	}

	/// `give_second_chance` is also reachable from `evict_one`'s FAST-tail
	/// fallback, when nothing has ever been demoted. That case is a pure
	/// reorder inside one queue: no tier change, no bytes moved, and --
	/// crucially -- NO migration, which is what the `was_slow` guard is for. A
	/// stack that pushed a redundant Fast->Fast entry here would make
	/// `PolicyWorker` rebuild a byte-identical DRAM buffer for nothing.
	#[test]
	fn a_second_chance_inside_the_fast_list_emits_no_migration() {
		// Fast tier far larger than the workload, so `settle_fast_tier` never
		// demotes and both slow segments stay empty -- which is the only way
		// `evict_one` reaches its fast-tail fallback at all.
		let mut a = Baseline::new(0.5, MAX, MAX);
		let mut b = Compact::new(0.5, MAX, MAX);

		for k in 1..=2u64 {
			a.insert(k, 10);
			b.insert(k, 10);
			a.update(k);
			b.update(k);
		}

		// Re-access the older one, so the fast tail carries a set bit.
		a.update(1);
		b.update(1);

		assert_eq!(a.slow_bytes_used(), 0, "baseline: nothing was ever demoted");
		assert_eq!(b.slow_bytes_used(), 0);
		a.drain_tier_migrations();
		b.drain_tier_migrations();

		let (ea, eb) = (a.evict_one(), b.evict_one());
		assert_eq!(ea, eb, "eviction victim diverges on the fast-tail fallback");
		assert_eq!(ea, Some(2), "the reprieved key must not be the victim");

		let (ma, mb) = (a.drain_tier_migrations(), b.drain_tier_migrations());
		assert_eq!(ma, mb, "fast-tail second-chance migrations diverge");
		assert!(mb.is_empty(), "a reorder inside the fast list must emit no migration");
		assert_eq!(a.tier_of(1), Some(Tier::Fast));
		assert_eq!(b.tier_of(1), a.tier_of(1));
		gauges(&a, &b);
	}

	/// Fast admission: the one-access queue is FAST, and a promotion out of it
	/// emits NO migration because the bytes are already DRAM. Pins the other
	/// delta a copy from `S3FifoCompactHybridStack` would silently lose.
	#[test]
	fn the_one_access_queue_is_fast_and_its_promotion_emits_no_migration() {
		let mut a = Baseline::new(0.5, MAX, 1_000_000);
		let mut b = Compact::new(0.5, MAX, 1_000_000);

		a.insert(7, 1024);
		b.insert(7, 1024);

		assert_eq!(a.tier_of(7), Some(Tier::Fast), "baseline: one-access is fast");
		assert_eq!(b.tier_of(7), a.tier_of(7));
		assert_eq!(a.slow_bytes_used(), 0, "baseline: nothing slow yet");

		let (admit_a, admit_b) = (a.drain_tier_migrations(), b.drain_tier_migrations());
		assert_eq!(admit_a, admit_b);
		assert!(admit_b.is_empty(), "admission itself must emit nothing");
		gauges(&a, &b);

		a.update(7);
		b.update(7);

		assert_eq!(a.tier_of(7), Some(Tier::Fast));
		assert_eq!(b.tier_of(7), a.tier_of(7));

		let (promo_a, promo_b) = (a.drain_tier_migrations(), b.drain_tier_migrations());
		assert_eq!(promo_a, promo_b);
		assert!(
			promo_b.is_empty(),
			"promotion out of the one-access queue must emit nothing -- the bytes are already DRAM",
		);
		assert_eq!(a.fast_object_count(), 1);
		gauges(&a, &b);
	}

	/// The one-access tail is REPRIEVED into `slow_head`, never evicted: the
	/// key stays tracked and a `Tier::Slow` migration is recorded.
	#[test]
	fn the_one_access_tail_is_reprieved_into_the_slow_head_not_evicted() {
		// one_access_capacity = 0.01 * 1_000 = 10 -- one 10-byte key exactly.
		let mut a = Baseline::new(0.01, 1_000, 1_000);
		let mut b = Compact::new(0.01, 1_000, 1_000);

		a.insert(1, 10);
		b.insert(1, 10);
		a.drain_tier_migrations();
		b.drain_tier_migrations();
		assert_eq!(a.tier_of(1), Some(Tier::Fast));
		assert_eq!(b.tier_of(1), a.tier_of(1));

		a.insert(2, 10);
		b.insert(2, 10);

		assert!(b.contains(1), "the key must still be tracked, not gone");
		assert_eq!(a.tier_of(1), Some(Tier::Slow));
		assert_eq!(b.tier_of(1), a.tier_of(1));
		assert_eq!(a.tier_of(2), Some(Tier::Fast));
		assert_eq!(b.tier_of(2), a.tier_of(2));
		assert_eq!(a.drain_tier_migrations(), b.drain_tier_migrations());
		gauges(&a, &b);
	}
}
