/*
 * Copyright (c) Kia Shakiba
 *
 * This source code is licensed under the GNU AGPLv3 license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! Slab-backed split-slow reprieve hybrid: behaviourally identical to
//! `S3FifoLazyDemotionFastAdmissionSplitSlowReprieveHybridStack`, with one
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
//! * Demotion is lazy and reference-bit gated, draining to the effective
//!   fast-tier budget.
//! * The shared per-tracked-key metadata reservation is split PROPORTIONALLY
//!   between the two independently-capacitied fast segments (`reserved_shares`),
//!   never charged in full to each. The two slow segments carry no capacity of
//!   their own and reserve nothing.
//! * `needs_capacity_eviction` is deliberately NOT overridden, matching the
//!   baseline: one-access pressure is relieved by `settle_one_access`, not by
//!   the eviction loop.
//!
//! **The baseline named above no longer exists in this crate.** Every
//! non-compact hybrid stack was removed once its compact twin was shown
//! behaviourally identical at 72 B/object of eviction stack instead of 112.
//! References to it here are historical: they say what this design is a
//! compaction OF, and they are the reason the structure looks the way it
//! does. Git history holds the baseline and the differential tests that
//! proved the two agreed.

use crate::{
	object::ObjectSize,
	worker::policy::policy_stack::{
		arena_queue_set::{ArenaQueueSet, NodePayload}, narrow_resident, CacheSize,
		HashedKey, PolicyStack, Tier,
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
/// tag: with the slow tier physically split, the queue alone says which tier a
/// key is in, which is why this stack never needed a tier field of its own.
///
/// The shared node stores this as a plain `u8`, so the enum is kept purely for
/// readability and converted at that one boundary: `Queue as u8` on the way in,
/// [`Queue::from_u8`] on the way out. The discriminants are pinned to the `Q_*`
/// queue indices they name, so the tag and the slot it is threaded into are the
/// same number.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
enum Queue {
	OneAccess = Q_ONE_ACCESS as u8,
	Fast = Q_FAST as u8,
	SlowHead = Q_SLOW_HEAD as u8,
	SlowTail = Q_SLOW_TAIL as u8,
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

	/// The inverse of `as u8`. `NodePayload::queue` is only ever written here
	/// from a `Queue as u8`, so the catch-all arm is unreachable and `SlowTail`
	/// is the only value it could stand for.
	fn from_u8(tag: u8) -> Queue {
		match tag {
			0 => Queue::OneAccess,
			1 => Queue::Fast,
			2 => Queue::SlowHead,
			_ => Queue::SlowTail,
		}
	}
}

/// Per-key bookkeeping is [`NodePayload`], the one node every policy shares.
///
/// This stack reads `queue` (as [`Queue`]), `freq`, `size` and `dram_resident`.
/// `freq` carries the S3-FIFO REFERENCE BIT: set is `freq = 1` and tested as
/// `freq != 0`, since a reference bit is a one-bit frequency counter.
///
/// The QUEUE remains the tier's single source of truth here -- this design's
/// whole point is that the split slow tier makes a separate tier field
/// redundant -- but `tier` and `phys` are kept in step with it at every queue
/// write (`Some(queue.tier())`), so the shared node never carries a tier that
/// contradicts the order the key is actually threaded into. `tier` is therefore
/// never `None` in this stack, and no match here has a `None` arm to spell out.
/// `ts` belongs to other policies and stays at its default.
pub struct S3FifoLazyDemotionFastAdmissionSplitSlowReprieveCompactHybridStack {
	/// All four orders -- one-access, main-fast, slow-head, slow-tail -- over a
	/// single slab. `MAX_QUEUES` is 4, which this uses in full.
	queues: ArenaQueueSet<NodePayload>,

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
			queues: ArenaQueueSet::default(),
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

	pub fn with_shared_overhead(mut self, overhead: CacheSize) -> Self {
		self.shared_overhead = overhead;


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
	/// metadata reservation is carved out. The settle drains to this.
	fn effective_main_fast_capacity(&self) -> CacheSize {
		self.main_fast_capacity().saturating_sub(self.reserved_shares().1)
	}

	pub fn tier_of(&self, key: HashedKey) -> Option<Tier> {
		self.queues.payload(key).map(|payload| Queue::from_u8(payload.queue).tier())
	}

	/// Returns `true` if `key` currently sits in the older (`slow_tail`) slow
	/// segment -- i.e. it has already survived a crossing check. Exposed for
	/// tests, exactly as on the baseline.
	pub fn is_in_slow_tail(&self, key: HashedKey) -> bool {
		self.queues.payload(key).map(|payload| payload.queue) == Some(Queue::SlowTail as u8)
	}

	/// `new_resident` refreshes the entry's DRAM-resident remainder: a re-set
	/// can add or drop a TTL, which changes it by the `Expiries` entry's cost.
	fn resize_key(&mut self, key: HashedKey, new_size: ObjectSize, new_resident: u8) {
		let Some(payload) = self.queues.payload_mut(key) else { return };

		let old_migrating = payload.migrating();
		payload.size = new_size;
		payload.dram_resident = new_resident;
		let delta = payload.migrating() as i64 - old_migrating as i64;
		let queue = Queue::from_u8(payload.queue);

		let counter = match queue {
			Queue::OneAccess => &mut self.one_access_used,
			Queue::Fast => &mut self.fast_used,
			Queue::SlowHead => &mut self.slow_head_used,
			Queue::SlowTail => &mut self.slow_tail_used,
		};

		*counter = (*counter as i64 + delta).max(0) as CacheSize;
	}

	fn touch(&mut self, key: HashedKey) {
		match self.queues.payload(key).map(|p| Queue::from_u8(p.queue)) {
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
			p.freq = 1;
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
			p.queue = Queue::Fast as u8;
			p.tier = Some(Queue::Fast.tier());
			p.phys = Some(Queue::Fast.tier());
			p.freq = 0;
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
		let was_slow = Queue::from_u8(payload.queue).is_slow();

		match Queue::from_u8(payload.queue) {
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
			p.queue = Queue::Fast as u8;
			p.tier = Some(Queue::Fast.tier());
			p.phys = Some(Queue::Fast.tier());
			p.freq = 0;
		}

		self.settle_fast_tier();

		// Only record a migration when the object genuinely crossed tiers AND
		// survived the settle above (which can demote it straight back out, in
		// which case that call already pushed the correct `Tier::Slow`
		// migration itself). A key that was already Fast needs no migration at
		// all -- a redundant Fast->Fast entry would make `PolicyWorker` rebuild
		// an identical buffer for nothing, and `give_second_chance` fires far
		// more often in this variant (every crossing).
		if was_slow && self.queues.payload(key).map(|p| p.queue) == Some(Queue::Fast as u8) {
			self.migrations.push((key, Tier::Fast));
		}
	}

	/// Demotes oldest-first out of the fast list into `slow_head` while the
	/// fast list exceeds its budget, reprieving any key whose bit is set instead.
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

		while self.fast_used > effective_capacity {
			let Some(candidate) = self.queues.back(Q_FAST) else { break };

			let accessed = self.queues.payload(candidate).map(|p| p.freq != 0).unwrap_or(false);

			if accessed {
				self.queues.move_front(Q_FAST, candidate);

				if let Some(p) = self.queues.payload_mut(candidate) {
					p.freq = 0;
				}

				continue;
			}

			let size = self.queues.payload(candidate).map(|p| p.migrating()).unwrap_or(0);

			self.queues.move_to_front_of(Q_FAST, Q_SLOW_HEAD, candidate);

			if let Some(p) = self.queues.payload_mut(candidate) {
				p.queue = Queue::SlowHead as u8;
				p.tier = Some(Queue::SlowHead.tier());
				p.phys = Some(Queue::SlowHead.tier());
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

			let accessed = self.queues.payload(candidate).map(|p| p.freq != 0).unwrap_or(false);

			if accessed {
				self.give_second_chance(candidate);
				continue;
			}

			let size = self.queues.payload(candidate).map(|p| p.migrating()).unwrap_or(0);

			self.queues.move_to_front_of(Q_SLOW_HEAD, Q_SLOW_TAIL, candidate);

			if let Some(p) = self.queues.payload_mut(candidate) {
				p.queue = Queue::SlowTail as u8;
				p.tier = Some(Queue::SlowTail.tier());
				p.phys = Some(Queue::SlowTail.tier());
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
	/// This segment relieves pressure synchronously rather than through a
	/// `PolicyWorker` migration batch.
	fn settle_one_access(&mut self) {
		let effective_capacity = self.effective_one_access_capacity();

		while self.one_access_used > effective_capacity {
			let Some(key) = self.queues.back(Q_ONE_ACCESS) else { break };
			let Some(payload) = self.queues.payload(key) else { break };
			let size = payload.migrating();

			self.one_access_used = self.one_access_used.saturating_sub(size);

			self.queues.move_to_front_of(Q_ONE_ACCESS, Q_SLOW_HEAD, key);

			if let Some(p) = self.queues.payload_mut(key) {
				p.queue = Queue::SlowHead as u8;
				p.tier = Some(Queue::SlowHead.tier());
				p.phys = Some(Queue::SlowHead.tier());
				p.freq = 0;
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
			NodePayload {
				size,
				freq: 0,
				ts: 0,
				queue: Queue::OneAccess as u8,
				tier: Some(Queue::OneAccess.tier()),
				phys: Some(Queue::OneAccess.tier()),
				dram_resident,
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

		match Queue::from_u8(payload.queue) {
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

			let accessed = self.queues.payload(key).map(|p| p.freq != 0).unwrap_or(false);

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
