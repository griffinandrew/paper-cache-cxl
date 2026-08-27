/*
 * Copyright (c) Kia Shakiba
 *
 * This source code is licensed under the GNU AGPLv3 license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! Slab-backed S3-FIFO lazy-demotion + fast-admission + reprieve hybrid:
//! behaviourally identical to
//! [`S3FifoLazyDemotionFastAdmissionReprieveHybridStack`], with one structure
//! where that has four.
//!
//! That stack keeps THREE `kwik::HashList`s -- `one_access_queue`,
//! `main_fast`, `main_slow` -- each owning its OWN key-to-node index, plus a
//! separate `entries` map holding the 8-byte payload. A key is in exactly one
//! of the three queues at any moment, so a single [`CompactQueueSet`] holds
//! all three orders over one slab, with the payload carried in that set's
//! single index value.
//!
//! This family is the one where the index-value layout earns its keep:
//! `mark_accessed` is the hottest per-get operation -- every hit on a
//! main-queue key does nothing but flip a reference bit, touching no queue
//! order at all -- so with the payload in the slab it would cost a
//! dereference on every such get for nothing. In the index value it is a
//! single probe.
//!
//! # What separates this from [`S3FifoCompactHybridStack`]
//!
//! Everything below is preserved byte for byte from the stack this compacts.
//!
//! 1. **The one-access queue is FAST.** `tier_of` reports `Tier::Fast` for a
//!    one-access resident, `fast_bytes_used()`/`fast_object_count()` count it,
//!    and `slow_bytes_used()`/`slow_object_count()` no longer do. Admission is
//!    a cheap DRAM write rather than a synchronous PMEM allocation on the
//!    calling thread (`hybrid_policy::admission_tier` returns `Fast` for a
//!    brand-new key under this policy). One consequence the stack must carry:
//!    `promote_from_one_access` emits NO `Tier::Fast` migration, because the
//!    key's bytes are already physically DRAM. `give_second_chance` keeps its
//!    push -- a key reaching it really can be in PMEM, so that move is real.
//!
//! 2. **The two fast segments share one budget.** `one_access_capacity` is a
//!    fixed carve-out of `fast_capacity` (`main_fast_capacity()`), and the
//!    shared-metadata reservation is split *proportionally* between the two
//!    (`reserved_shares`, following `LruSizedHybridStack`) so that
//!    `effective_one_access_capacity() + effective_main_fast_capacity() +
//!    reserved_overhead() == fast_capacity`. `main_slow` carries no capacity
//!    of its own, so it has nothing to reserve against.
//!
//! 3. **Demotion is lazy.** `settle_fast_tier` gives a `main_fast` tail whose
//!    reference bit is set a reprieve -- move it to the front of `main_fast`
//!    with the bit cleared, and try the next candidate -- instead of demoting
//!    it.
//!
//! 4. **The one-access tail is reprieved, not evicted.** Once
//!    `one_access_used` exceeds its effective capacity, `settle_one_access`
//!    moves the tail into the FRONT of `main_slow` -- a full life there,
//!    promotable through the ordinary `touch()`/tail-second-chance machinery
//!    -- instead of removing it from the cache. That relief runs
//!    *synchronously* from `insert()`/`resize()`, never through
//!    `evict_one()`/`needs_capacity_eviction()`: `apply_evictions`
//!    unconditionally erases whatever key `evict_one()` returns from the
//!    entire cache, and a reprieve is not an eviction.
//!    `needs_capacity_eviction()` therefore stays at the trait default
//!    `false`, and `evict_one()` is purely the main queue's tail loop.
//!
//! # Three queues, and why the boundary cursor is gone
//!
//! [`S3FifoCompactHybridStack`] keeps the main queue as ONE order with a
//! `main_boundary: Option<HashedKey>` cursor marking the oldest still-fast
//! key; the fast tier is the contiguous prefix up to that cursor and demotion
//! is a pure relabel. That works only while demotion is the sole thing that
//! ever crosses the boundary.
//!
//! A reprieve breaks that premise: it has to insert a node AT the boundary.
//! The stack this compacts solves it by splitting main into two physically
//! separate orders, and the same split is what this stack holds in its slab:
//!
//! * `Q_MAIN_FAST` -- front = newest, back = oldest fast key (the demotion
//!   candidate, previously `main_boundary`).
//! * `Q_MAIN_SLOW` -- front = *exactly* the fast/slow boundary position,
//!   back = the eviction candidate.
//!
//! So every boundary crossing is O(1) and needs no cursor to maintain:
//! demotion is a `Q_MAIN_FAST` -> front of `Q_MAIN_SLOW` move, promotion is
//! the reverse, a one-access reprieve is a move to the front of
//! `Q_MAIN_SLOW`, and eviction is the back of `Q_MAIN_SLOW` (falling back to
//! the back of `Q_MAIN_FAST` only when nothing has ever been demoted). The
//! per-tier counters `fast_count`/`main_count` go with the cursor: with
//! homogeneous lists, `queue_len` IS the count.
//!
//! One deliberate limitation carried over unchanged: `insert()` of a
//! brand-new key grows the tracked-key count (and so the reservation) but
//! only calls `settle_one_access()`, never `settle_fast_tier()`. `main_fast`
//! can therefore sit briefly above its freshly-shrunk effective budget. That
//! is bounded and self-correcting -- `fast_used` only ever grows via
//! `promote_from_one_access()` and `give_second_chance()`, both of which
//! settle immediately.

use crate::{
	object::ObjectSize,
	worker::policy::policy_stack::{
		compact_queue_set::CompactQueueSet, narrow_resident, watermarks, CacheSize, HashedKey,
		PolicyStack, Tier,
	},
	PaperPolicy,
};

const Q_ONE_ACCESS: usize = 0;
const Q_MAIN_FAST: usize = 1;
const Q_MAIN_SLOW: usize = 2;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Queue {
	OneAccess,
	Main,
}

/// Combined per-key bookkeeping, carried in the index value.
///
/// `tier` and `accessed` are only meaningful while `queue == Main`: the
/// one-access queue is entirely fast-tier in this variant and its promotion
/// is eager, so a key there needs neither. `tier` also names WHICH main list
/// the key is in -- `Some(Tier::Fast)` is `Q_MAIN_FAST`, `Some(Tier::Slow)`
/// is `Q_MAIN_SLOW` -- so the two are never allowed to disagree.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct S3FifoLazyDemotionFastAdmissionReprievePayload {
	queue: Queue,
	tier: Option<Tier>,
	/// The part of `size` that stays in DRAM in either tier; see `migrating`.
	dram_resident: u8,
	accessed: bool,
	size: ObjectSize,
}

/// Pinned, exactly as `S3FifoEntry` is in the stack this replaces.
const _: () = assert!(
	std::mem::size_of::<S3FifoLazyDemotionFastAdmissionReprievePayload>() == 8,
	"S3FifoLazyDemotionFastAdmissionReprievePayload grew past 8 bytes",
);

impl S3FifoLazyDemotionFastAdmissionReprievePayload {
	fn migrating(&self) -> CacheSize {
		(self.size as CacheSize).saturating_sub(self.dram_resident as CacheSize)
	}
}

pub struct S3FifoLazyDemotionFastAdmissionReprieveCompactHybridStack {
	queues: CompactQueueSet<S3FifoLazyDemotionFastAdmissionReprievePayload>,

	one_access_ratio: f64,
	one_access_capacity: CacheSize,
	one_access_used: CacheSize,

	/// The configured total fast-tier (DRAM) budget, shared between the
	/// one-access queue and the main queue's fast segment. There is no
	/// `main_capacity` in this variant: nothing is sized from `1 - ratio` and
	/// nothing gates eviction on main fullness.
	fast_capacity: CacheSize,
	fast_used: CacheSize,
	slow_used: CacheSize,

	shared_overhead: CacheSize,

	migrations: Vec<(HashedKey, Tier)>,
}

impl S3FifoLazyDemotionFastAdmissionReprieveCompactHybridStack {
	pub fn new(one_access_ratio: f64, max_size: CacheSize, fast_capacity: CacheSize) -> Self {
		S3FifoLazyDemotionFastAdmissionReprieveCompactHybridStack {
			queues: CompactQueueSet::default(),
			one_access_ratio,
			one_access_capacity: (one_access_ratio * max_size as f64) as CacheSize,
			one_access_used: 0,
			fast_capacity,
			fast_used: 0,
			slow_used: 0,
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

	fn reserved_overhead(&self) -> CacheSize {
		self.queues.len() as CacheSize * self.shared_overhead
	}

	/// The main queue's fast-segment budget *before* the shared-metadata
	/// reservation -- `fast_capacity` minus the one-access queue's fixed
	/// carve-out. Kept separate from `effective_main_fast_capacity` so
	/// `reserved_shares` has a reservation-free capacity to proportion
	/// against (using the effective one would be circular).
	fn main_fast_capacity(&self) -> CacheSize {
		self.fast_capacity.saturating_sub(self.one_access_capacity)
	}

	/// Splits `reserved_overhead()` proportionally between this stack's two
	/// independently-capacitied FAST segments -- the one-access queue and the
	/// main queue's fast portion -- returned as `(one_access_share,
	/// main_fast_share)`. `u128` intermediate so the product cannot overflow;
	/// remainder handed to the main segment so the two shares always re-sum
	/// exactly. `(0, 0)` if both capacities are zero.
	fn reserved_shares(&self) -> (CacheSize, CacheSize) {
		let reserved = self.reserved_overhead();

		let one_access_capacity = self.one_access_capacity;
		let main_fast_capacity = self.main_fast_capacity();
		let total_capacity = one_access_capacity + main_fast_capacity;

		if total_capacity == 0 {
			return (0, 0);
		}

		let one_access_share =
			((reserved as u128 * one_access_capacity as u128) / total_capacity as u128) as CacheSize;
		let main_fast_share = reserved.saturating_sub(one_access_share);

		(one_access_share, main_fast_share)
	}

	/// The one-access queue's own byte cap after giving up its share of the
	/// shared-metadata reservation. With no reservation wired in this is the
	/// raw cap.
	fn effective_one_access_capacity(&self) -> CacheSize {
		self.one_access_capacity.saturating_sub(self.reserved_shares().0)
	}

	/// The budget actually available to the main queue's fast segment: raw
	/// `fast_capacity`, minus the one-access carve-out, minus this segment's
	/// share of the shared-metadata reservation. The watermarks sit on top of
	/// this number, never in place of any part of it.
	fn effective_main_fast_capacity(&self) -> CacheSize {
		self.main_fast_capacity().saturating_sub(self.reserved_shares().1)
	}

	pub fn tier_of(&self, key: HashedKey) -> Option<Tier> {
		let payload = self.queues.payload(key)?;

		match payload.queue {
			// The one-access queue is DRAM-resident in this variant.
			Queue::OneAccess => Some(Tier::Fast),
			Queue::Main => payload.tier,
		}
	}

	fn resize_key(&mut self, key: HashedKey, new_size: ObjectSize, new_resident: u8) {
		let Some(payload) = self.queues.payload_mut(key) else { return };

		let old_migrating = payload.migrating();
		payload.size = new_size;
		payload.dram_resident = new_resident;
		let delta = payload.migrating() as i64 - old_migrating as i64;
		let (queue, tier) = (payload.queue, payload.tier);

		match (queue, tier) {
			(Queue::OneAccess, _) => {
				self.one_access_used = (self.one_access_used as i64 + delta).max(0) as CacheSize;
			},

			(Queue::Main, Some(Tier::Fast)) => {
				self.fast_used = (self.fast_used as i64 + delta).max(0) as CacheSize;
			},

			(Queue::Main, Some(Tier::Slow)) => {
				self.slow_used = (self.slow_used as i64 + delta).max(0) as CacheSize;
			},

			(Queue::Main, None) => {},
		}
	}

	fn touch(&mut self, key: HashedKey) {
		match self.queues.payload(key).map(|p| p.queue) {
			Some(Queue::OneAccess) => self.promote_from_one_access(key),
			Some(Queue::Main) => self.mark_accessed(key),
			None => {},
		}
	}

	/// The hottest per-get operation in this family, and the reason the
	/// payload lives in the index value: one probe, no slab access, no queue
	/// movement.
	fn mark_accessed(&mut self, key: HashedKey) {
		if let Some(p) = self.queues.payload_mut(key) {
			p.accessed = true;
		}
	}

	/// Moves a re-accessed one-access-queue key to the front of `main_fast`.
	/// Emits no migration for the promotion itself -- the key's bytes are
	/// already physically Fast in this variant.
	fn promote_from_one_access(&mut self, key: HashedKey) {
		let Some(payload) = self.queues.payload(key) else { return };
		let size_bytes = payload.migrating();

		self.queues.move_to_front_of(Q_ONE_ACCESS, Q_MAIN_FAST, key);
		self.one_access_used = self.one_access_used.saturating_sub(size_bytes);

		if let Some(p) = self.queues.payload_mut(key) {
			p.queue = Queue::Main;
			p.tier = Some(Tier::Fast);
			p.accessed = false;
		}

		self.fast_used += size_bytes;

		self.settle_fast_tier();
	}

	/// An accessed key at the main tail is reinserted at the front of
	/// `main_fast` with its reference bit cleared, rather than evicted.
	///
	/// Tier-aware, because the two main tiers are now two physical lists: a
	/// still-fast key only moves within `main_fast`, while a slow key leaves
	/// `main_slow` for the front of `main_fast` and its bytes move with it.
	///
	/// This is the one promotion path that STILL pushes a migration: a key
	/// reaching it can genuinely be in PMEM, so moving it back to Fast is a
	/// physical move, not a relabeling.
	fn give_second_chance(&mut self, key: HashedKey) {
		let Some(payload) = self.queues.payload(key) else { return };
		let size = payload.migrating();

		match payload.tier {
			Some(Tier::Fast) => {
				self.queues.move_front(Q_MAIN_FAST, key);

				if let Some(p) = self.queues.payload_mut(key) {
					p.accessed = false;
				}
			},

			Some(Tier::Slow) => {
				self.queues.move_to_front_of(Q_MAIN_SLOW, Q_MAIN_FAST, key);

				if let Some(p) = self.queues.payload_mut(key) {
					p.tier = Some(Tier::Fast);
					p.accessed = false;
				}

				self.slow_used = self.slow_used.saturating_sub(size);
				self.fast_used += size;
			},

			None => return,
		}

		self.settle_fast_tier();

		if self.queues.payload(key).and_then(|p| p.tier) == Some(Tier::Fast) {
			self.migrations.push((key, Tier::Fast));
		}
	}

	/// Demotes the `main_fast` tail once `fast_used` crosses the HIGH
	/// watermark of `effective_main_fast_capacity()`, then keeps going until
	/// it is back at or below the LOW watermark -- reference-bit gated.
	///
	/// The reference-bit gate is the "lazy demotion": a candidate whose bit
	/// is set is moved to the FRONT of `main_fast` with the bit cleared and
	/// the pass tries the next one. Each reprieve clears a bit, so a pass can
	/// reprieve at most `queue_len(Q_MAIN_FAST)` times before it must demote.
	///
	/// `effective_capacity` is read once, before the loop: a demotion only
	/// moves a key between two lists, so the tracked-key count -- and hence
	/// the reservation and the target -- cannot move underneath the pass.
	fn settle_fast_tier(&mut self) {
		let effective_capacity = self.effective_main_fast_capacity();

		if self.fast_used <= watermarks::high_bytes(effective_capacity) {
			return;
		}

		let drain_target = watermarks::low_bytes(effective_capacity);

		while self.fast_used > drain_target {
			let Some(candidate) = self.queues.back(Q_MAIN_FAST) else { break };

			let accessed = self.queues.payload(candidate).map(|p| p.accessed).unwrap_or(false);

			if accessed {
				// Reprieve: fresh start at the front instead of demotion.
				self.queues.move_front(Q_MAIN_FAST, candidate);

				if let Some(p) = self.queues.payload_mut(candidate) {
					p.accessed = false;
				}

				continue;
			}

			let size = self.queues.payload(candidate).map(|p| p.migrating()).unwrap_or(0);

			// The front of `main_slow` IS the fast/slow boundary position.
			self.queues.move_to_front_of(Q_MAIN_FAST, Q_MAIN_SLOW, candidate);

			if let Some(p) = self.queues.payload_mut(candidate) {
				p.tier = Some(Tier::Slow);
			}

			self.fast_used = self.fast_used.saturating_sub(size);
			self.slow_used += size;

			self.migrations.push((candidate, Tier::Slow));
		}
	}

	/// Relieves one-access-queue pressure by moving its tail(s) to the front
	/// of `main_slow` -- the fast/slow boundary position -- so the key gets a
	/// full life in the main queue instead of leaving the cache.
	///
	/// Called synchronously from `insert()`/`resize()`, exactly mirroring
	/// `settle_fast_tier()`'s relationship to the fast/slow boundary. A pure
	/// internal migration: nothing is ever removed from the cache here, so
	/// this must never be routed through
	/// `evict_one()`/`needs_capacity_eviction()`.
	///
	/// Budget hoisted out of the loop for the same reason as in
	/// `settle_fast_tier`: a reprieve moves a key between lists, it never
	/// adds or removes one, so the reservation is fixed for the pass. No
	/// watermarks here -- this boundary was never given a high/low pair.
	fn settle_one_access(&mut self) {
		let effective_capacity = self.effective_one_access_capacity();

		while self.one_access_used > effective_capacity {
			let Some(key) = self.queues.back(Q_ONE_ACCESS) else { break };
			let size = self.queues.payload(key).map(|p| p.migrating()).unwrap_or(0);

			self.queues.move_to_front_of(Q_ONE_ACCESS, Q_MAIN_SLOW, key);
			self.one_access_used = self.one_access_used.saturating_sub(size);

			if let Some(p) = self.queues.payload_mut(key) {
				p.queue = Queue::Main;
				p.tier = Some(Tier::Slow);
				p.accessed = false;
			}

			self.slow_used += size;

			self.migrations.push((key, Tier::Slow));
		}
	}
}

impl PolicyStack for S3FifoLazyDemotionFastAdmissionReprieveCompactHybridStack {
	fn is_policy(&self, policy: &PaperPolicy) -> bool {
		matches!(policy, PaperPolicy::S3FifoLazyDemotionFastAdmissionReprieveCompactHybrid(r) if *r == self.one_access_ratio)
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
			return;
		}

		self.queues.push_front(
			Q_ONE_ACCESS,
			key,
			S3FifoLazyDemotionFastAdmissionReprievePayload {
				queue: Queue::OneAccess,
				tier: None,
				dram_resident,
				accessed: false,
				size,
			},
		);
		self.one_access_used += (size as CacheSize).saturating_sub(dram_resident as CacheSize);

		self.settle_one_access();
	}

	fn update(&mut self, key: HashedKey) {
		if self.queues.contains(key) {
			self.touch(key);
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

			// `tier` names which main list holds the key, so it selects the
			// queue to unlink from. `None` there is unreachable -- a payload
			// only reaches `Queue::Main` through `promote_from_one_access`
			// (Fast) or `settle_one_access` (Slow) -- and is a no-op for the
			// same reason it is in the stack this compacts.
			Queue::Main => match payload.tier {
				Some(Tier::Fast) => {
					self.queues.remove(Q_MAIN_FAST, key);
					self.fast_used = self.fast_used.saturating_sub(size);
				},

				Some(Tier::Slow) => {
					self.queues.remove(Q_MAIN_SLOW, key);
					self.slow_used = self.slow_used.saturating_sub(size);
				},

				None => {},
			},
		}
	}

	fn resize(&mut self, max_size: CacheSize) {
		self.one_access_capacity = (self.one_access_ratio * max_size as f64) as CacheSize;

		// Both boundaries move: the one-access cap directly, and the main
		// queue's fast segment because it is what is LEFT of `fast_capacity`
		// after the carve-out.
		self.settle_one_access();
		self.settle_fast_tier();
	}

	fn clear(&mut self) {
		self.queues.clear();

		self.one_access_used = 0;
		self.fast_used = 0;
		self.slow_used = 0;
		self.migrations.clear();
	}

	/// Purely the main queue's tail loop. The one-access queue is never
	/// evicted from -- `settle_one_access` drains it into `main_slow`
	/// instead -- so a cache holding nothing but one-access keys correctly
	/// reports `None` here.
	fn evict_one(&mut self) -> Option<HashedKey> {
		loop {
			let (key, from_slow) = match self.queues.back(Q_MAIN_SLOW) {
				Some(key) => (key, true),
				None => (self.queues.back(Q_MAIN_FAST)?, false),
			};

			let accessed = self.queues.payload(key).map(|p| p.accessed).unwrap_or(false);

			if accessed {
				self.give_second_chance(key);
				continue;
			}

			let queue = if from_slow { Q_MAIN_SLOW } else { Q_MAIN_FAST };
			let payload = self.queues.remove(queue, key);
			let size = payload.map(|p| p.migrating()).unwrap_or(0);

			if from_slow {
				self.slow_used = self.slow_used.saturating_sub(size);
			} else {
				self.fast_used = self.fast_used.saturating_sub(size);
			}

			return Some(key);
		}
	}

	fn resize_fast_tier(&mut self, size: CacheSize) {
		self.fast_capacity = size;
		self.settle_fast_tier();
	}

	fn drain_tier_migrations(&mut self) -> Vec<(HashedKey, Tier)> {
		std::mem::take(&mut self.migrations)
	}

	fn dram_reserved_bytes(&self) -> CacheSize {
		self.reserved_overhead()
	}

	fn fast_bytes_used(&self) -> CacheSize {
		// Total DRAM: the main queue's fast segment plus the one-access
		// queue, both physically Fast in this variant.
		self.fast_used + self.one_access_used
	}

	fn slow_bytes_used(&self) -> CacheSize {
		// The one-access queue no longer touches Slow/PMEM at all.
		self.slow_used
	}

	fn fast_object_count(&self) -> usize {
		self.queues.queue_len(Q_MAIN_FAST) + self.queues.queue_len(Q_ONE_ACCESS)
	}

	fn slow_object_count(&self) -> usize {
		self.queues.queue_len(Q_MAIN_SLOW)
	}
}

/// Fidelity against `S3FifoLazyDemotionFastAdmissionReprieveHybridStack`,
/// which this stack is a compaction of.
#[cfg(all(test, feature = "s3_fifo_lazy_demotion_fast_admission_reprieve_hybrid_cache"))]
mod fidelity_tests {
	use super::*;
	use crate::worker::policy::policy_stack::s3_fifo_lazy_demotion_fast_admission_reprieve_hybrid_stack::S3FifoLazyDemotionFastAdmissionReprieveHybridStack;

	/// Deliberately small. `one_access_capacity` is a carve-out of
	/// `fast_capacity` here, so a `MAX` of a million against the fast-tier
	/// sizes this family is tested at would leave `main_fast_capacity()`
	/// saturating to zero at every ratio and the main fast segment would
	/// never hold anything.
	const MAX: CacheSize = 100_000;

	/// Repeats are essential here: a key only leaves the one-access queue on
	/// a SECOND access, and the reference bit only matters on a third, so a
	/// workload without reuse would exercise neither the main queue, the
	/// demotion-boundary reprieve, nor the second-chance path.
	///
	/// The tail of fresh, never-repeated keys is equally load-bearing in the
	/// other direction: 20_000 skewed ops over 200 keys touch essentially all
	/// of them twice, so WITHOUT that tail the one-access queue is empty at
	/// the end and a `tier_of` snapshot never observes a one-access resident
	/// -- which is exactly the population this variant reports `Fast` and
	/// plain s3-fifo reports `Slow`.
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
		for i in 0..64u64 {
			ops.push((10_000 + i, 1024));
		}
		ops
	}

	type Snapshot = (Vec<Option<Tier>>, CacheSize, CacheSize, usize, usize, usize);

	fn snapshot(stack: &dyn PolicyStack, tiers: Vec<Option<Tier>>) -> Snapshot {
		(
			tiers,
			stack.fast_bytes_used(),
			stack.slow_bytes_used(),
			stack.fast_object_count(),
			stack.slow_object_count(),
			stack.len(),
		)
	}

	fn replay(
		ratio: f64,
		fast: CacheSize,
		overhead: CacheSize,
		ops: &[(HashedKey, ObjectSize)],
	) -> (Vec<(HashedKey, Tier)>, Vec<(HashedKey, Tier)>, Snapshot, Snapshot) {
		let mut a = S3FifoLazyDemotionFastAdmissionReprieveHybridStack::new(ratio, MAX, fast)
			.with_shared_overhead(overhead);
		let mut b = S3FifoLazyDemotionFastAdmissionReprieveCompactHybridStack::new(ratio, MAX, fast)
			.with_shared_overhead(overhead);
		let (mut ma, mut mb) = (Vec::new(), Vec::new());

		for (k, size) in ops {
			if a.contains(*k) { a.update(*k); } else { a.insert(*k, *size); }
			if b.contains(*k) { b.update(*k); } else { b.insert(*k, *size); }
			ma.extend(a.drain_tier_migrations());
			mb.extend(b.drain_tier_migrations());
		}

		let keys: Vec<HashedKey> = ops.iter().map(|(k, _)| *k).collect();
		let ta = keys.iter().map(|k| a.tier_of(*k)).collect();
		let tb = keys.iter().map(|k| b.tier_of(*k)).collect();
		(ma, mb, snapshot(&a, ta), snapshot(&b, tb))
	}

	#[test]
	fn matches_baseline_migration_for_migration() {
		let ops = skewed_ops();
		// Some of the grid is degenerate on purpose -- at ratio 0.5 with a
		// 16 KiB fast tier the one-access carve-out swallows the whole
		// budget and `main_fast_capacity()` saturates to zero -- so "some
		// configuration ended with a live fast tier" is asserted across the
		// grid rather than inside it.
		let mut saw_fast = false;
		let mut saw_slow = false;

		for ratio in [0.1f64, 0.25, 0.5] {
			for fast in [16_384u64, 65_536, 262_144] {
				for overhead in [0u64, 112] {
					let (ma, mb, sa, sb) = replay(ratio, fast, overhead, &ops);
					assert_eq!(sa, sb, "state diverges at ratio {ratio} fast {fast} overhead {overhead}");
					assert_eq!(ma, mb, "migrations diverge at ratio {ratio} fast {fast} overhead {overhead}");
					saw_fast |= sa.3 > 0;
					saw_slow |= sa.4 > 0;
				}
			}
		}

		assert!(saw_fast && saw_slow, "the grid never populated both tiers; the snapshots prove nothing");
	}

	/// Eviction is where the reference bit is acted on at the tail: an
	/// accessed key there is reinserted at the front of `main_fast` instead
	/// of evicted, which both reorders the queue mid-eviction AND moves the
	/// key between the two main lists. Nothing above exercises that.
	///
	/// `evict_one` never touches the one-access queue in this variant, so
	/// the drain ends with keys still resident -- the assertion is that both
	/// stacks stop at the same point with the same population.
	#[test]
	fn evicts_in_the_same_order_including_second_chances() {
		let ops = skewed_ops();
		let mut a = S3FifoLazyDemotionFastAdmissionReprieveHybridStack::new(0.25, MAX, 65_536)
			.with_shared_overhead(112);
		let mut b = S3FifoLazyDemotionFastAdmissionReprieveCompactHybridStack::new(0.25, MAX, 65_536)
			.with_shared_overhead(112);
		for (k, size) in &ops {
			if a.contains(*k) { a.update(*k); } else { a.insert(*k, *size); }
			if b.contains(*k) { b.update(*k); } else { b.insert(*k, *size); }
			a.drain_tier_migrations();
			b.drain_tier_migrations();
		}
		assert_eq!(a.needs_capacity_eviction(), b.needs_capacity_eviction());

		let mut ea = Vec::new();
		let mut eb = Vec::new();
		while let Some(k) = a.evict_one() { ea.push(k); }
		while let Some(k) = b.evict_one() { eb.push(k); }
		assert!(!ea.is_empty(), "the workload evicted nothing; the test proves nothing");
		assert_eq!(ea, eb, "eviction order diverges");
		assert_eq!(a.len(), b.len());
		assert_eq!(a.drain_tier_migrations(), b.drain_tier_migrations());
	}

	#[test]
	fn removal_matches_across_all_three_queues() {
		let ops = skewed_ops();
		let mut a = S3FifoLazyDemotionFastAdmissionReprieveHybridStack::new(0.25, MAX, 65_536)
			.with_shared_overhead(112);
		let mut b = S3FifoLazyDemotionFastAdmissionReprieveCompactHybridStack::new(0.25, MAX, 65_536)
			.with_shared_overhead(112);
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
		assert_eq!(a.len(), b.len());
		assert_eq!(a.fast_object_count(), b.fast_object_count());
		assert_eq!(a.slow_object_count(), b.slow_object_count());
		assert_eq!(a.fast_bytes_used(), b.fast_bytes_used());
		assert_eq!(a.slow_bytes_used(), b.slow_bytes_used());
	}

	/// Resize in both directions with BRAND-NEW keys afterwards. The shape is
	/// load-bearing: the equivalent test on the LFU conversion passed with a
	/// real bug present because the workload had no new keys after the
	/// resize. `resize` here rescales `one_access_capacity` and then settles
	/// BOTH boundaries, since the main fast segment is what is left of
	/// `fast_capacity` after the carve-out.
	#[test]
	fn resizes_like_the_baseline() {
		for (start, resized) in [(65_536u64, 65_536u64), (262_144, 32_768), (32_768, 262_144)] {
			let mut a = S3FifoLazyDemotionFastAdmissionReprieveHybridStack::new(0.25, MAX, start)
				.with_shared_overhead(112);
			let mut b = S3FifoLazyDemotionFastAdmissionReprieveCompactHybridStack::new(0.25, MAX, start)
				.with_shared_overhead(112);
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
			assert_eq!(a.needs_capacity_eviction(), b.needs_capacity_eviction());

			for i in 0..2_000u64 {
				let k = 10_000 + i;
				a.insert(k, 1024);
				b.insert(k, 1024);
				ma.extend(a.drain_tier_migrations());
				mb.extend(b.drain_tier_migrations());
			}

			assert_eq!(ma, mb, "migrations diverge resizing {start} -> {resized}");
			assert_eq!(a.fast_bytes_used(), b.fast_bytes_used());
			assert_eq!(a.slow_bytes_used(), b.slow_bytes_used());
			for i in 0..2_000u64 {
				assert_eq!(a.tier_of(10_000 + i), b.tier_of(10_000 + i));
			}
		}
	}

	/// DELTA 9 + DELTA 12: `resize` settles BOTH internal boundaries
	/// synchronously, and it does so in that order.
	///
	/// `resizes_like_the_baseline` above cannot see this: its workload
	/// promotes every key out of the one-access queue before the resize, and
	/// its resize only ever GROWS the main fast segment, so both settles are
	/// no-ops there and dropping them changes nothing. This test leaves both
	/// boundaries with real work to do and reads the migrations before any
	/// later insert can settle them instead.
	#[test]
	fn resize_settles_both_boundaries_synchronously() {
		let mut a = S3FifoLazyDemotionFastAdmissionReprieveHybridStack::new(0.25, MAX, 65_536)
			.with_shared_overhead(0);
		let mut b = S3FifoLazyDemotionFastAdmissionReprieveCompactHybridStack::new(0.25, MAX, 65_536)
			.with_shared_overhead(0);

		// Twenty promoted keys in `main_fast`, twenty-one still one-access.
		// Both sit inside their budgets, so nothing has settled yet.
		for k in 1..=20u64 {
			a.insert(k, 1024);
			b.insert(k, 1024);
			a.update(k);
			b.update(k);
		}
		for k in 100..=120u64 {
			a.insert(k, 1024);
			b.insert(k, 1024);
		}
		assert_eq!(a.drain_tier_migrations(), b.drain_tier_migrations());
		assert!(b.queues.queue_len(Q_ONE_ACCESS) > 0, "no one-access residents to settle");
		assert!(b.queues.queue_len(Q_MAIN_FAST) > 0, "no fast main residents to settle");
		assert_eq!(b.slow_object_count(), 0, "nothing should have settled yet");

		// Shrink: `one_access_capacity` 25_000 -> 2_500. `settle_one_access`
		// must spill the one-access tail into `main_slow` right now.
		a.resize(10_000);
		b.resize(10_000);
		let (ma, mb) = (a.drain_tier_migrations(), b.drain_tier_migrations());
		assert_eq!(ma, mb, "one-access spill on resize diverges");
		assert!(!ma.is_empty(), "resize did not settle the one-access boundary");
		assert!(ma.iter().all(|(_, t)| *t == Tier::Slow));
		assert_eq!(a.slow_object_count(), b.slow_object_count());
		assert!(b.slow_object_count() > 0);
		assert_eq!(a.fast_bytes_used(), b.fast_bytes_used());
		assert_eq!(a.slow_bytes_used(), b.slow_bytes_used());

		// Grow: `one_access_capacity` 25_000 -> 65_000 leaves the main fast
		// segment 536 bytes of the 65_536 fast budget, so `settle_fast_tier`
		// must demote `main_fast` on the spot.
		let fast_before = b.queues.queue_len(Q_MAIN_FAST);
		a.resize(260_000);
		b.resize(260_000);
		let (ma, mb) = (a.drain_tier_migrations(), b.drain_tier_migrations());
		assert_eq!(ma, mb, "fast-tier demotion on resize diverges");
		assert!(!ma.is_empty(), "resize did not settle the fast-tier boundary");
		assert!(
			b.queues.queue_len(Q_MAIN_FAST) < fast_before,
			"the main fast segment did not shrink when its carve-out grew",
		);
		assert_eq!(a.fast_object_count(), b.fast_object_count());
		assert_eq!(a.slow_object_count(), b.slow_object_count());
		assert_eq!(a.fast_bytes_used(), b.fast_bytes_used());
		assert_eq!(a.slow_bytes_used(), b.slow_bytes_used());
	}

	/// DELTA 1 + DELTA 4, the two things that make this variant what it is.
	///
	/// A brand-new key is admitted FAST (not Slow, as plain s3-fifo does),
	/// and when the one-access queue overflows its tail is REPRIEVED into
	/// the slow tier of the main queue rather than evicted: the key stays in
	/// the cache, `len()` never drops, and `needs_capacity_eviction()` stays
	/// at the trait default `false`.
	///
	/// This is the test that fails outright if the delta was not applied:
	/// `S3FifoCompactHybridStack`'s shape would report the fresh key `Slow`,
	/// keep it in the one-access queue, and raise `needs_capacity_eviction`.
	#[test]
	fn one_access_is_fast_and_its_tail_is_reprieved_not_evicted() {
		// ratio 0.1 of 100_000 => a 10_000-byte one-access carve-out, so the
		// tenth 1 KiB object overflows it.
		let mut a = S3FifoLazyDemotionFastAdmissionReprieveHybridStack::new(0.1, MAX, 65_536)
			.with_shared_overhead(0);
		let mut b = S3FifoLazyDemotionFastAdmissionReprieveCompactHybridStack::new(0.1, MAX, 65_536)
			.with_shared_overhead(0);

		a.insert(1, 1024);
		b.insert(1, 1024);
		assert_eq!(a.tier_of(1), Some(Tier::Fast), "admission must be FAST in this variant");
		assert_eq!(b.tier_of(1), a.tier_of(1));
		assert_eq!(b.fast_bytes_used(), 1024, "a one-access resident counts as fast bytes");
		assert_eq!(b.slow_bytes_used(), 0);
		assert_eq!((b.fast_object_count(), b.slow_object_count()), (1, 0));
		assert_eq!(a.drain_tier_migrations(), b.drain_tier_migrations());

		for k in 2..=40u64 {
			a.insert(k, 1024);
			b.insert(k, 1024);
		}

		// Nothing left the cache: every overflowed key was reprieved into
		// `main_slow`, not evicted.
		assert_eq!(a.len(), 40, "the baseline must not evict on one-access overflow");
		assert_eq!(b.len(), a.len());
		assert!(!b.needs_capacity_eviction(), "relief must not run through evict_one");
		assert_eq!(a.needs_capacity_eviction(), b.needs_capacity_eviction());

		// The oldest keys spilled to the slow tier of main; the newest are
		// still one-access, and therefore still fast.
		assert_eq!(a.tier_of(1), Some(Tier::Slow));
		assert_eq!(b.tier_of(1), a.tier_of(1));
		assert_eq!(a.tier_of(40), Some(Tier::Fast));
		assert_eq!(b.tier_of(40), a.tier_of(40));

		let (ma, mb) = (a.drain_tier_migrations(), b.drain_tier_migrations());
		assert!(ma.contains(&(1, Tier::Slow)), "the spill must be reported as a migration");
		assert_eq!(ma, mb);
		assert!(b.slow_object_count() > 0);
		assert_eq!(a.slow_object_count(), b.slow_object_count());
		assert_eq!(a.fast_object_count(), b.fast_object_count());
		assert_eq!(a.fast_bytes_used(), b.fast_bytes_used());
		assert_eq!(a.slow_bytes_used(), b.slow_bytes_used());
	}

	/// DELTA 3: lazy demotion. An accessed key at the `main_fast` TAIL --
	/// the oldest fast key, the one [`S3FifoCompactHybridStack`] demotes
	/// unconditionally as `main_boundary` -- is moved back to the front with
	/// its bit cleared, and the pass demotes the next candidate instead.
	///
	/// The tail is read out of the slab rather than assumed, so the test does
	/// not depend on the exact watermark fractions.
	#[test]
	fn accessed_fast_tail_is_reprieved_instead_of_demoted() {
		// ratio 0.1 => 10_000-byte one-access carve-out; the main fast
		// segment gets the remaining 10_480 bytes, about ten 1 KiB objects.
		let mut a = S3FifoLazyDemotionFastAdmissionReprieveHybridStack::new(0.1, MAX, 20_480)
			.with_shared_overhead(0);
		let mut b = S3FifoLazyDemotionFastAdmissionReprieveCompactHybridStack::new(0.1, MAX, 20_480)
			.with_shared_overhead(0);

		// Insert-then-update promotes each key straight into `main_fast`, so
		// the one-access queue never overflows and every migration below is
		// a genuine demotion.
		for k in 1..=40u64 {
			a.insert(k, 1024);
			b.insert(k, 1024);
			a.update(k);
			b.update(k);
		}
		a.drain_tier_migrations();
		b.drain_tier_migrations();

		let tail = b.queues.back(Q_MAIN_FAST).expect("main_fast should be populated");
		assert_eq!(b.tier_of(tail), Some(Tier::Fast));
		assert_eq!(a.tier_of(tail), b.tier_of(tail));

		// Set the reference bit on that tail, then push the segment back over
		// its high watermark so a demotion pass runs.
		a.update(tail);
		b.update(tail);
		for k in 41..=44u64 {
			a.insert(k, 1024);
			b.insert(k, 1024);
			a.update(k);
			b.update(k);
		}

		let (ma, mb) = (a.drain_tier_migrations(), b.drain_tier_migrations());
		assert_eq!(ma, mb, "demotion migrations diverge");
		assert!(
			ma.iter().any(|(_, t)| *t == Tier::Slow),
			"no demotion pass ran; the reprieve was never exercised",
		);
		assert!(
			!ma.contains(&(tail, Tier::Slow)),
			"the accessed tail was demoted instead of reprieved",
		);
		assert_eq!(
			b.tier_of(tail),
			Some(Tier::Fast),
			"the accessed oldest fast key must survive the pass in DRAM",
		);
		assert_eq!(a.tier_of(tail), b.tier_of(tail));
		assert_eq!(a.fast_object_count(), b.fast_object_count());
		assert_eq!(a.slow_object_count(), b.slow_object_count());
		assert_eq!(a.fast_bytes_used(), b.fast_bytes_used());
		assert_eq!(a.slow_bytes_used(), b.slow_bytes_used());
	}

	/// DELTA 5 + DELTA 11: the two-list main queue, and the eviction order
	/// that follows from it. `main_slow`'s tail is the victim whenever
	/// `main_slow` is non-empty, even though `main_fast` holds strictly
	/// older-by-promotion keys; only an empty `main_slow` falls through to
	/// the `main_fast` tail.
	#[test]
	fn eviction_prefers_the_slow_tail_and_falls_back_to_the_fast_tail() {
		let mut a = S3FifoLazyDemotionFastAdmissionReprieveHybridStack::new(0.1, MAX, 20_480)
			.with_shared_overhead(0);
		let mut b = S3FifoLazyDemotionFastAdmissionReprieveCompactHybridStack::new(0.1, MAX, 20_480)
			.with_shared_overhead(0);

		// Nothing has been demoted or spilled yet: `main_slow` is empty and
		// the only eviction candidate is the `main_fast` tail.
		for k in 1..=3u64 {
			a.insert(k, 1024);
			b.insert(k, 1024);
			a.update(k);
			b.update(k);
		}
		a.drain_tier_migrations();
		b.drain_tier_migrations();
		assert_eq!(b.slow_object_count(), 0);
		assert_eq!(a.evict_one(), Some(1));
		assert_eq!(b.evict_one(), Some(1));

		// Now drive keys into `main_slow` and check the victim comes from
		// there while `main_fast` still holds promoted keys.
		for k in 4..=40u64 {
			a.insert(k, 1024);
			b.insert(k, 1024);
			a.update(k);
			b.update(k);
		}
		a.drain_tier_migrations();
		b.drain_tier_migrations();
		assert!(b.slow_object_count() > 0, "nothing reached main_slow");
		assert!(b.queues.queue_len(Q_MAIN_FAST) > 0, "main_fast emptied; the test proves nothing");

		let mut ea = Vec::new();
		let mut eb = Vec::new();
		for _ in 0..10 {
			ea.push(a.evict_one());
			eb.push(b.evict_one());
		}
		assert_eq!(ea, eb, "eviction order diverges once both main lists are populated");
		assert_eq!(a.len(), b.len());
		assert_eq!(a.drain_tier_migrations(), b.drain_tier_migrations());
	}
}
