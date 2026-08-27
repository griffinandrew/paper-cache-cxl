/*
 * Copyright (c) Kia Shakiba
 *
 * This source code is licensed under the GNU AGPLv3 license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! Slab-backed 2Q fast-admission REPRIEVE hybrid: the compact form of
//! `TwoQFastAdmissionReprieveHybridStack`, one structure where that has three.
//!
//! Everything `TwoQFastAdmissionCompactHybridStack` does, this does too: the
//! admission FIFO is DRAM-resident, so `tier_of` reports `Fast` for a key in
//! it, its reservation is carved OUT of the fast tier, a promotion out of it
//! emits no migration, and its bytes and objects count toward fast.
//!
//! What differs is what happens when the FIFO runs over budget. The
//! non-reprieve stack lets it grow and asks the caller to evict its tail
//! (`needs_capacity_eviction`). This one REPRIEVES the overflow instead:
//! `settle_fifo_queue` splices the FIFO tail onto the BACK of the main queue
//! as `Tier::Slow`, emitting a migration, so an aged-out one-access key gets a
//! second chance in PMEM rather than being dropped. Four consequences, each
//! carried over from the baseline:
//!
//! - `settle_fifo_queue` runs after every admission and after either resize --
//!   exactly the three call sites the baseline uses. It deliberately does NOT
//!   run on the re-set path of `insert_resident`, which returns early, nor on
//!   promotion out of the FIFO, which only ever lowers `fifo_used`.
//! - `needs_capacity_eviction` is NOT overridden. The FIFO polices itself, so
//!   the trait default (`false`) is the answer; the non-reprieve stack's
//!   `fifo_used > fifo_capacity` override would ask the caller to evict a
//!   queue that has already settled.
//! - `evict_one` drains the MAIN queue first and reaches the FIFO tail only
//!   when main is empty. The non-reprieve stack has that order reversed.
//! - The `shared_overhead` reservation is SPLIT between the two queues in
//!   proportion to their fast-tier capacities (`reserved_shares`), because
//!   both now settle against a budget and each has to pay its own share. The
//!   non-reprieve stack charges the whole reservation to the main queue.

use crate::{
	object::ObjectSize,
	worker::policy::policy_stack::{
		compact_queue_set::CompactQueueSet, narrow_resident, watermarks, CacheSize, HashedKey,
		PolicyStack, Tier,
	},
	PaperPolicy,
};

/// Queue slots in the shared set. The FIFO admission queue is 0, the LRU main
/// queue is 1; a key is in exactly one of them.
const Q_FIFO: usize = 0;
const Q_MAIN: usize = 1;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Queue {
	Fifo,
	Main,
}

/// Combined per-key bookkeeping, carried in the index value.
///
/// `tier` is `None` while `queue == Fifo`: the FIFO is entirely slow-tier, so a
/// key there has no tier of its own to record.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct TwoQFaReprievePayload {
	queue: Queue,
	tier: Option<Tier>,
	/// The part of `size` that stays in DRAM in either tier; see `migrating`.
	dram_resident: u8,
	size: ObjectSize,
}

/// Pinned, exactly as `TwoQEntry` is in the stack this replaces. The payload
/// rides in the index bucket, so growth here costs bytes on every tracked key.
const _: () = assert!(
	std::mem::size_of::<TwoQFaReprievePayload>() == 8,
	"TwoQFaReprievePayload grew past 8 bytes",
);

impl TwoQFaReprievePayload {
	/// Bytes that actually move between tiers.
	fn migrating(&self) -> CacheSize {
		(self.size as CacheSize).saturating_sub(self.dram_resident as CacheSize)
	}
}

pub struct TwoQFastAdmissionReprieveCompactHybridStack {
	queues: CompactQueueSet<TwoQFaReprievePayload>,

	k_in: f64,

	fifo_capacity: CacheSize,
	fifo_used: CacheSize,

	fast_capacity: CacheSize,
	fast_used: CacheSize,
	slow_used: CacheSize,

	shared_overhead: CacheSize,

	fast_count: usize,
	main_count: usize,

	/// The least-recently-used FAST key in the main queue.
	main_boundary: Option<HashedKey>,

	migrations: Vec<(HashedKey, Tier)>,
}

impl TwoQFastAdmissionReprieveCompactHybridStack {
	pub fn new(k_in: f64, max_size: CacheSize, fast_capacity: CacheSize) -> Self {
		TwoQFastAdmissionReprieveCompactHybridStack {
			queues: CompactQueueSet::default(),
			k_in,
			fifo_capacity: (k_in * max_size as f64) as CacheSize,
			fifo_used: 0,
			fast_capacity,
			fast_used: 0,
			slow_used: 0,
			shared_overhead: 0,
			fast_count: 0,
			main_count: 0,
			main_boundary: None,
			migrations: Vec::new(),
		}
	}

	pub fn with_shared_overhead(mut self, overhead: CacheSize) -> Self {
		self.shared_overhead = overhead;


		self
	}

	pub fn fast_capacity(&self) -> CacheSize {
		self.fast_capacity
	}

	/// The main queue's share of the fast tier. The FIFO is DRAM-resident here,
	/// so its reservation is carved out of the same budget the main queue
	/// settles against -- the two compete, where in plain 2Q the FIFO is in
	/// PMEM and does not.
	///
	/// Only the MAIN queue's SHARE of `reserved_overhead` is subtracted, not
	/// the whole of it: in this variant the FIFO settles against a budget of
	/// its own and pays the remainder. The non-reprieve stack, whose FIFO is
	/// unpoliced, charges the entire reservation here.
	fn effective_main_fast_capacity(&self) -> CacheSize {
		self.fast_capacity
			.saturating_sub(self.fifo_capacity)
			.saturating_sub(self.reserved_shares().1)
	}

	/// The FIFO's budget net of its share of the metadata reservation. What
	/// `settle_fifo_queue` settles against.
	fn effective_fifo_capacity(&self) -> CacheSize {
		self.fifo_capacity.saturating_sub(self.reserved_shares().0)
	}

	/// Splits `reserved_overhead` between the two queues in proportion to
	/// their fast-tier capacities: `(fifo_share, main_share)`.
	///
	/// `fifo_capacity` is clamped to `fast_capacity` first, so a FIFO
	/// reservation larger than the whole fast tier takes ALL of the overhead
	/// and leaves main none, rather than producing a share above 1. Widened to
	/// `u128` for the multiply: `reserved * fifo_capacity` overflows `u64` at
	/// realistic entry counts.
	fn reserved_shares(&self) -> (CacheSize, CacheSize) {
		let reserved = self.reserved_overhead();

		if self.fast_capacity == 0 {
			return (0, 0);
		}

		let fifo_capacity = self.fifo_capacity.min(self.fast_capacity);
		let fifo_share =
			((reserved as u128 * fifo_capacity as u128) / self.fast_capacity as u128) as CacheSize;
		let main_share = reserved.saturating_sub(fifo_share);

		(fifo_share, main_share)
	}

	fn reserved_overhead(&self) -> CacheSize {
		self.queues.len() as CacheSize * self.shared_overhead
	}

	pub fn tier_of(&self, key: HashedKey) -> Option<Tier> {
		let payload = self.queues.payload(key)?;
		match payload.queue {
			Queue::Fifo => Some(Tier::Fast),
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
			(Queue::Fifo, _) => {
				self.fifo_used = (self.fifo_used as i64 + delta).max(0) as CacheSize;
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
			Some(Queue::Fifo) => self.promote_from_fifo(key),
			Some(Queue::Main) => self.touch_main_fast(key),
			None => {},
		}
	}

	/// A hit in the FIFO promotes to the front of main, and to fast.
	///
	/// The slot does not move: this is an unlink from one queue and a relink
	/// into the other, where the stack this replaces removed the key from one
	/// hash-indexed list and inserted it into another.
	fn promote_from_fifo(&mut self, key: HashedKey) {
		let Some(payload) = self.queues.payload(key) else { return };
		let size_bytes = payload.migrating();

		self.queues.move_to_front_of(Q_FIFO, Q_MAIN, key);
		self.fifo_used = self.fifo_used.saturating_sub(size_bytes);

		if let Some(p) = self.queues.payload_mut(key) {
			p.queue = Queue::Main;
			p.tier = Some(Tier::Fast);
		}

		self.fast_used += size_bytes;
		self.fast_count += 1;
		self.main_count += 1;

		if self.main_boundary.is_none() {
			self.main_boundary = Some(key);
		}

		// No migration emitted: the FIFO is already DRAM, so promotion moves
		// bookkeeping rather than bytes.
		self.settle_fast_tier();
	}

	/// Faithful port of `TwoQFastAdmissionReprieveHybridStack::touch_main_fast`,
	/// which is itself unchanged from the non-reprieve stack's.
	fn touch_main_fast(&mut self, key: HashedKey) {
		let previous_tier = self.queues.payload(key).and_then(|p| p.tier);

		let already_at_front = self.queues.front(Q_MAIN) == Some(key);
		let is_boundary = self.main_boundary == Some(key);

		// Read the neighbour BEFORE moving: once the key is at the front its
		// predecessor is gone, and the boundary must step back to whatever was
		// in front of it.
		let new_boundary_if_moved = if is_boundary && !already_at_front {
			self.queues.before(key)
		} else {
			None
		};

		self.queues.move_front(Q_MAIN, key);

		if is_boundary && !already_at_front {
			self.main_boundary = new_boundary_if_moved;
		}

		let mut promoted = false;

		if previous_tier != Some(Tier::Fast) {
			if previous_tier == Some(Tier::Slow) {
				let size = self.queues.payload(key).map(|p| p.migrating()).unwrap_or(0);
				self.slow_used = self.slow_used.saturating_sub(size);
				self.fast_used += size;
				self.fast_count += 1;
				promoted = true;
			}

			if let Some(p) = self.queues.payload_mut(key) {
				p.tier = Some(Tier::Fast);
			}

			if self.main_boundary.is_none() {
				self.main_boundary = Some(key);
			}
		}

		self.settle_fast_tier();

		if promoted && self.queues.payload(key).and_then(|p| p.tier) == Some(Tier::Fast) {
			self.migrations.push((key, Tier::Fast));
		}
	}

	/// Demotes from the tier boundary until `fast_used` is back under the low
	/// watermark. The victim is always `main_boundary`, so nothing is searched.
	fn settle_fast_tier(&mut self) {
		let effective = self.effective_main_fast_capacity();

		if self.fast_used <= watermarks::high_bytes(effective) {
			return;
		}

		let low_water = watermarks::low_bytes(effective);

		while self.fast_used > low_water {
			let Some(demote_key) = self.main_boundary else { break };
			let size = self.queues.payload(demote_key).map(|p| p.migrating()).unwrap_or(0);
			let new_boundary = self.queues.before(demote_key);

			if let Some(p) = self.queues.payload_mut(demote_key) {
				p.tier = Some(Tier::Slow);
			}

			self.fast_used = self.fast_used.saturating_sub(size);
			self.fast_count = self.fast_count.saturating_sub(1);
			self.slow_used += size;
			self.main_boundary = new_boundary;

			self.migrations.push((demote_key, Tier::Slow));
		}
	}

	/// The reprieve, and the whole point of this variant. Splices the FIFO
	/// tail onto the BACK of the main queue as `Tier::Slow` until the FIFO is
	/// back inside its effective budget.
	///
	/// The slot does not move: this is an unlink from `Q_FIFO` and a relink at
	/// the tail of `Q_MAIN`, where the stack this replaces popped one
	/// `HashList` and pushed the other.
	///
	/// `main_boundary` is deliberately untouched. It tracks the least-recently
	/// used FAST key in main, and everything arriving here is slow and lands
	/// behind it, so the boundary is still where it was.
	fn settle_fifo_queue(&mut self) {
		let effective = self.effective_fifo_capacity();

		while self.fifo_used > effective {
			let Some(key) = self.queues.back(Q_FIFO) else { break };

			// Unreachable -- `back` returned the key, so it is indexed. The
			// baseline's `continue` on a missing entry is kept in shape here,
			// dropping the link so the loop cannot spin.
			let Some(payload) = self.queues.payload(key) else {
				self.queues.remove(Q_FIFO, key);
				continue;
			};

			let size = payload.migrating();

			self.queues.move_to_back_of(Q_FIFO, Q_MAIN, key);
			self.fifo_used = self.fifo_used.saturating_sub(size);

			if let Some(p) = self.queues.payload_mut(key) {
				p.queue = Queue::Main;
				p.tier = Some(Tier::Slow);
			}

			self.slow_used += size;
			self.main_count += 1;

			self.migrations.push((key, Tier::Slow));
		}
	}

	fn evict_fifo_tail(&mut self) -> Option<HashedKey> {
		let (key, payload) = self.queues.pop_back(Q_FIFO)?;
		self.fifo_used = self.fifo_used.saturating_sub(payload.migrating());
		Some(key)
	}
}

impl PolicyStack for TwoQFastAdmissionReprieveCompactHybridStack {
	fn is_policy(&self, policy: &PaperPolicy) -> bool {
		matches!(policy, PaperPolicy::TwoQFastAdmissionReprieveCompactHybrid(k_in) if *k_in == self.k_in)
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
			Q_FIFO,
			key,
			TwoQFaReprievePayload { queue: Queue::Fifo, tier: None, dram_resident, size },
		);
		self.fifo_used += (size as CacheSize).saturating_sub(dram_resident as CacheSize);

		// The reprieve: an admission that pushes the FIFO over budget spills
		// its tail into main here, rather than leaving it for an eviction.
		self.settle_fifo_queue();
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
			Queue::Fifo => {
				self.queues.remove(Q_FIFO, key);
				self.fifo_used = self.fifo_used.saturating_sub(size);
			},

			Queue::Main => {
				let new_boundary_if_needed =
					if payload.tier == Some(Tier::Fast) && self.main_boundary == Some(key) {
						self.queues.before(key)
					} else {
						None
					};

				self.queues.remove(Q_MAIN, key);
				self.main_count = self.main_count.saturating_sub(1);

				match payload.tier {
					Some(Tier::Fast) => {
						self.fast_used = self.fast_used.saturating_sub(size);
						self.fast_count = self.fast_count.saturating_sub(1);

						if self.main_boundary == Some(key) {
							self.main_boundary = new_boundary_if_needed;
						}
					},

					Some(Tier::Slow) => {
						self.slow_used = self.slow_used.saturating_sub(size);
					},

					None => {},
				}
			},
		}
	}

	fn resize(&mut self, max_size: CacheSize) {
		self.fifo_capacity = (self.k_in * max_size as f64) as CacheSize;

		// The FIFO reservation is carved out of the fast tier, so moving it
		// changes the main queue's budget. Plain 2Q does not need this.
		self.settle_fast_tier();

		// ... and it moved the FIFO's own budget too, which only this variant
		// settles against.
		self.settle_fifo_queue();
	}

	fn clear(&mut self) {
		self.queues.clear();

		self.fifo_used = 0;
		self.fast_used = 0;
		self.slow_used = 0;
		self.fast_count = 0;
		self.main_count = 0;
		self.main_boundary = None;
		self.migrations.clear();
	}

	fn evict_one(&mut self) -> Option<HashedKey> {
		// Main first; the FIFO tail only once main is empty -- the reverse of
		// the non-reprieve stack. The FIFO is policed by `settle_fifo_queue`,
		// so its tail is not eviction's first choice here.
		if self.queues.queue_len(Q_MAIN) == 0 {
			return self.evict_fifo_tail();
		}

		let (key, payload) = self.queues.pop_back(Q_MAIN)?;
		let size = payload.migrating();
		self.main_count = self.main_count.saturating_sub(1);

		match payload.tier {
			Some(Tier::Fast) => {
				self.fast_used = self.fast_used.saturating_sub(size);
				self.fast_count = self.fast_count.saturating_sub(1);

				if self.main_boundary == Some(key) {
					self.main_boundary = self.queues.back(Q_MAIN);
				}
			},

			Some(Tier::Slow) => {
				self.slow_used = self.slow_used.saturating_sub(size);
			},

			None => {},
		}

		Some(key)
	}

	fn resize_fast_tier(&mut self, size: CacheSize) {
		self.fast_capacity = size;
		self.settle_fast_tier();

		// `reserved_shares` is a function of `fast_capacity`, so the FIFO's
		// effective budget moved as well.
		self.settle_fifo_queue();
	}

	fn drain_tier_migrations(&mut self) -> Vec<(HashedKey, Tier)> {
		std::mem::take(&mut self.migrations)
	}

	fn dram_reserved_bytes(&self) -> CacheSize {
		self.reserved_overhead()
	}

	fn fast_bytes_used(&self) -> CacheSize {
		self.fifo_used + self.fast_used
	}

	fn slow_bytes_used(&self) -> CacheSize {
		self.slow_used
	}

	fn fast_object_count(&self) -> usize {
		self.queues.queue_len(Q_FIFO) + self.fast_count
	}

	fn slow_object_count(&self) -> usize {
		self.main_count - self.fast_count
	}

	// NO `needs_capacity_eviction` override, matching the baseline: the FIFO
	// settles itself, so the trait default (`false`) is correct. The
	// non-reprieve stack overrides it with `fifo_used > fifo_capacity`.
}


/// Fidelity against `TwoQFastAdmissionReprieveHybridStack`.
///
/// Same shape as the module this was derived from, plus the cases unique to
/// the reprieve: the FIFO spilling into main instead of being evicted, the
/// split metadata reservation, and the reversed eviction order.
///
/// SATURATION IS LOAD-BEARING. The skewed workload alone leaves the FIFO
/// EMPTY -- 200 distinct keys over 20,000 accesses are every one of them
/// promoted into main -- and against an empty FIFO the eviction order, the
/// `resize` re-settle and the `resize_fast_tier` re-settle are all
/// indistinguishable from their absence. An earlier version of this module
/// passed unchanged against a stack with `evict_one` reversed and both resize
/// call sites deleted. The tests below that matter first fill the FIFO with
/// keys that are never re-accessed, and assert the FIFO is still holding the
/// newest of them before they measure anything.
#[cfg(all(test, feature = "two_q_fast_admission_reprieve_hybrid_cache"))]
mod fidelity_tests {
	use super::*;
	use crate::worker::policy::policy_stack::two_q_fast_admission_reprieve_hybrid_stack::TwoQFastAdmissionReprieveHybridStack;

	const MAX: CacheSize = 1_000_000;

	/// Keys used to saturate the FIFO. Disjoint from `skewed_ops`'s 1..=200.
	const FRESH: HashedKey = 1_000_000;
	const FRESH_COUNT: u64 = 400;

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

	#[test]
	fn matches_the_baseline_migration_for_migration() {
		let ops = skewed_ops();
		for k_in in [0.1f64, 0.25, 0.5] {
			for fast in [8_192u64, 32_768, 131_072] {
				for overhead in [0u64, 112] {
					let mut a = TwoQFastAdmissionReprieveHybridStack::new(k_in, MAX, fast)
						.with_shared_overhead(overhead);
					let mut b = TwoQFastAdmissionReprieveCompactHybridStack::new(k_in, MAX, fast)
						.with_shared_overhead(overhead);
					let (mut ma, mut mb) = (Vec::new(), Vec::new());

					for (k, size) in &ops {
						if a.contains(*k) { a.update(*k); } else { a.insert(*k, *size); }
						if b.contains(*k) { b.update(*k); } else { b.insert(*k, *size); }
						ma.extend(a.drain_tier_migrations());
						mb.extend(b.drain_tier_migrations());
					}

					assert_eq!(ma, mb, "migrations diverge k_in {k_in} fast {fast} oh {overhead}");
					for (k, _) in &ops {
						assert_eq!(a.tier_of(*k), b.tier_of(*k), "tier of {k} diverges");
					}
					assert_eq!(a.len(), b.len(), "len diverges");
					assert_eq!(a.fast_bytes_used(), b.fast_bytes_used(), "fast bytes");
					assert_eq!(a.slow_bytes_used(), b.slow_bytes_used(), "slow bytes");
					assert_eq!(a.fast_object_count(), b.fast_object_count(), "fast objects");
					assert_eq!(a.slow_object_count(), b.slow_object_count(), "slow objects");
					assert_eq!(a.dram_reserved_bytes(), b.dram_reserved_bytes(), "reserved");
					assert_eq!(
						a.needs_capacity_eviction(),
						b.needs_capacity_eviction(),
						"needs_capacity_eviction diverges",
					);
				}
			}
		}
	}

	/// Admission lands in DRAM, as in the non-reprieve variant.
	#[test]
	fn admission_is_fast_and_promotion_matches_the_baseline() {
		const BIG_FAST: CacheSize = 900_000;
		let mut a =
			TwoQFastAdmissionReprieveHybridStack::new(0.1, MAX, BIG_FAST).with_shared_overhead(0);
		let mut b = TwoQFastAdmissionReprieveCompactHybridStack::new(0.1, MAX, BIG_FAST)
			.with_shared_overhead(0);

		a.insert(1, 1024);
		b.insert(1, 1024);
		assert_eq!(a.tier_of(1), Some(Tier::Fast), "admission should land fast");
		assert_eq!(b.tier_of(1), a.tier_of(1));
		assert_eq!(a.drain_tier_migrations(), b.drain_tier_migrations());

		a.update(1);
		b.update(1);
		let (ma, mb) = (a.drain_tier_migrations(), b.drain_tier_migrations());
		assert_eq!(ma, mb, "promotion migrations diverge");
		assert!(ma.is_empty(), "with a real main budget, promotion moves no bytes");
		assert_eq!(a.tier_of(1), b.tier_of(1));
		assert_eq!(a.fast_object_count(), b.fast_object_count());
	}

	/// The saturating case: a FIFO reservation larger than the fast tier drives
	/// the main budget to zero, so everything promoted is demoted straight back.
	#[test]
	fn a_fifo_reservation_larger_than_the_fast_tier_matches_too() {
		let mut a =
			TwoQFastAdmissionReprieveHybridStack::new(0.25, MAX, 131_072).with_shared_overhead(0);
		let mut b = TwoQFastAdmissionReprieveCompactHybridStack::new(0.25, MAX, 131_072)
			.with_shared_overhead(0);

		a.insert(1, 1024);
		b.insert(1, 1024);
		a.update(1);
		b.update(1);

		let (ma, mb) = (a.drain_tier_migrations(), b.drain_tier_migrations());
		assert_eq!(ma, mb, "migrations diverge under a saturated main budget");
		assert!(!ma.is_empty(), "expected the saturated budget to force a demotion");
		assert_eq!(a.tier_of(1), b.tier_of(1));
	}

	/// THE reprieve: overflow off the FIFO tail is spliced into main as slow,
	/// not evicted. Nothing is lost, and the baseline agrees key for key.
	#[test]
	fn fifo_overflow_is_reprieved_into_main_rather_than_dropped() {
		// fifo_capacity = 8_192 bytes = 8 objects of 1 KiB.
		let mut a = TwoQFastAdmissionReprieveHybridStack::new(0.008_192, MAX, 500_000)
			.with_shared_overhead(0);
		let mut b = TwoQFastAdmissionReprieveCompactHybridStack::new(0.008_192, MAX, 500_000)
			.with_shared_overhead(0);
		let (mut ma, mut mb) = (Vec::new(), Vec::new());

		for k in 1..=40u64 {
			a.insert(k, 1024);
			b.insert(k, 1024);
			ma.extend(a.drain_tier_migrations());
			mb.extend(b.drain_tier_migrations());
		}

		assert_eq!(ma, mb, "reprieve migrations diverge");
		assert!(!ma.is_empty(), "expected the FIFO to overflow and reprieve");
		assert_eq!(a.len(), 40, "reprieve must not drop keys");
		assert_eq!(b.len(), a.len(), "len diverges");

		for k in 1..=40u64 {
			assert_eq!(a.tier_of(k), b.tier_of(k), "tier of {k} diverges");
		}
		assert_eq!(a.tier_of(1), Some(Tier::Slow), "the oldest key should be reprieved to slow");
		assert_eq!(a.tier_of(40), Some(Tier::Fast), "the newest key should still be in the FIFO");
		assert_eq!(a.slow_object_count(), b.slow_object_count());
		assert_eq!(a.fast_object_count(), b.fast_object_count());
	}

	/// A reprieved key that is hit again comes back to the front of main and to
	/// fast, exactly as the baseline does it.
	#[test]
	fn a_reprieved_key_can_be_promoted_back_like_the_baseline() {
		let mut a = TwoQFastAdmissionReprieveHybridStack::new(0.008_192, MAX, 500_000)
			.with_shared_overhead(0);
		let mut b = TwoQFastAdmissionReprieveCompactHybridStack::new(0.008_192, MAX, 500_000)
			.with_shared_overhead(0);

		for k in 1..=40u64 {
			a.insert(k, 1024);
			b.insert(k, 1024);
		}
		a.drain_tier_migrations();
		b.drain_tier_migrations();
		assert_eq!(a.tier_of(1), Some(Tier::Slow));

		a.update(1);
		b.update(1);
		let (ma, mb) = (a.drain_tier_migrations(), b.drain_tier_migrations());
		assert_eq!(ma, mb, "promotion out of a reprieve diverges");
		assert_eq!(a.tier_of(1), b.tier_of(1));
		assert_eq!(a.tier_of(1), Some(Tier::Fast), "a hit should bring it back to fast");
		assert_eq!(a.fast_bytes_used(), b.fast_bytes_used());
		assert_eq!(a.slow_bytes_used(), b.slow_bytes_used());
	}

	/// The split reservation. With a nonzero `shared_overhead` both queues pay
	/// a share, and the shares move as entries arrive -- so the settle points
	/// of BOTH queues drift together.
	#[test]
	fn the_split_metadata_reservation_matches_the_baseline() {
		let ops = skewed_ops();
		for (k_in, fast) in [(0.05f64, 65_536u64), (0.25, 65_536), (0.5, 262_144)] {
			let mut a =
				TwoQFastAdmissionReprieveHybridStack::new(k_in, MAX, fast).with_shared_overhead(112);
			let mut b = TwoQFastAdmissionReprieveCompactHybridStack::new(k_in, MAX, fast)
				.with_shared_overhead(112);
			let (mut ma, mut mb) = (Vec::new(), Vec::new());

			for (k, size) in ops.iter().take(6_000) {
				if a.contains(*k) { a.update(*k); } else { a.insert(*k, *size); }
				if b.contains(*k) { b.update(*k); } else { b.insert(*k, *size); }
				ma.extend(a.drain_tier_migrations());
				mb.extend(b.drain_tier_migrations());
			}

			assert_eq!(ma, mb, "migrations diverge k_in {k_in} fast {fast}");
			assert_eq!(a.dram_reserved_bytes(), b.dram_reserved_bytes(), "reserved bytes");
			assert_eq!(a.fast_bytes_used(), b.fast_bytes_used(), "fast bytes");
			assert_eq!(a.slow_bytes_used(), b.slow_bytes_used(), "slow bytes");
			assert_eq!(a.fast_object_count(), b.fast_object_count(), "fast objects");
			assert_eq!(a.slow_object_count(), b.slow_object_count(), "slow objects");
		}
	}

	/// `resize` rescales `fifo_capacity`, so it moves the FIFO's own settle
	/// point as well as the main queue's budget.
	///
	/// The FIFO is saturated first, deliberately: with slack in the FIFO the
	/// re-settle is a no-op, and this test cannot tell the `settle_fifo_queue()`
	/// call in `resize` from its absence. It could not, before.
	#[test]
	fn resize_resettles_a_saturated_fifo_like_the_baseline() {
		let mut a =
			TwoQFastAdmissionReprieveHybridStack::new(0.25, MAX, 1_000_000).with_shared_overhead(112);
		let mut b = TwoQFastAdmissionReprieveCompactHybridStack::new(0.25, MAX, 1_000_000)
			.with_shared_overhead(112);
		let (mut ma, mut mb) = (Vec::new(), Vec::new());

		for i in 0..FRESH_COUNT {
			a.insert(FRESH + i, 1024);
			b.insert(FRESH + i, 1024);
			ma.extend(a.drain_tier_migrations());
			mb.extend(b.drain_tier_migrations());
		}
		assert_eq!(ma, mb, "saturation migrations diverge");
		assert_eq!(
			a.tier_of(FRESH + FRESH_COUNT - 1),
			Some(Tier::Fast),
			"the FIFO must still hold the newest keys, or this test proves nothing",
		);

		a.resize(MAX / 4);
		b.resize(MAX / 4);
		let (ra, rb) = (a.drain_tier_migrations(), b.drain_tier_migrations());

		assert_eq!(ra, rb, "the resize re-settle diverges");
		assert!(!ra.is_empty(), "shrinking fifo_capacity must spill the FIFO into main");
		for i in 0..FRESH_COUNT {
			assert_eq!(a.tier_of(FRESH + i), b.tier_of(FRESH + i), "tier of {i} diverges");
		}
		assert_eq!(a.len(), b.len(), "len diverges");
		assert_eq!(a.fast_bytes_used(), b.fast_bytes_used(), "fast bytes");
		assert_eq!(a.slow_bytes_used(), b.slow_bytes_used(), "slow bytes");
		assert_eq!(a.fast_object_count(), b.fast_object_count(), "fast objects");
		assert_eq!(a.slow_object_count(), b.slow_object_count(), "slow objects");
	}

	/// `resize_fast_tier` moves `reserved_shares`, and therefore the FIFO's
	/// effective budget, even though `fifo_capacity` itself does not change.
	///
	/// Shrinking the fast tier to the FIFO's own size hands the FIFO the WHOLE
	/// metadata reservation where it previously paid a quarter of it, so a
	/// saturated FIFO must spill. Same load-bearing saturation as above.
	#[test]
	fn resize_fast_tier_resettles_a_saturated_fifo_like_the_baseline() {
		let mut a =
			TwoQFastAdmissionReprieveHybridStack::new(0.25, MAX, 1_000_000).with_shared_overhead(112);
		let mut b = TwoQFastAdmissionReprieveCompactHybridStack::new(0.25, MAX, 1_000_000)
			.with_shared_overhead(112);
		let (mut ma, mut mb) = (Vec::new(), Vec::new());

		for i in 0..FRESH_COUNT {
			a.insert(FRESH + i, 1024);
			b.insert(FRESH + i, 1024);
			ma.extend(a.drain_tier_migrations());
			mb.extend(b.drain_tier_migrations());
		}
		assert_eq!(ma, mb, "saturation migrations diverge");
		assert_eq!(
			a.tier_of(FRESH + FRESH_COUNT - 1),
			Some(Tier::Fast),
			"the FIFO must still hold the newest keys, or this test proves nothing",
		);

		a.resize_fast_tier(250_000);
		b.resize_fast_tier(250_000);
		let (ra, rb) = (a.drain_tier_migrations(), b.drain_tier_migrations());

		assert_eq!(ra, rb, "the resize_fast_tier re-settle diverges");
		assert!(
			!ra.is_empty(),
			"the FIFO's share of the reservation grew, so a saturated FIFO must spill",
		);
		for i in 0..FRESH_COUNT {
			assert_eq!(a.tier_of(FRESH + i), b.tier_of(FRESH + i), "tier of {i} diverges");
		}
		assert_eq!(a.fast_bytes_used(), b.fast_bytes_used(), "fast bytes");
		assert_eq!(a.slow_bytes_used(), b.slow_bytes_used(), "slow bytes");
		assert_eq!(a.fast_object_count(), b.fast_object_count(), "fast objects");
		assert_eq!(a.slow_object_count(), b.slow_object_count(), "slow objects");
	}

	/// The same two resize call sites over the skewed workload, where the FIFO
	/// carries slack -- the re-settle is a no-op there, and both stacks must
	/// agree that it is.
	#[test]
	fn resize_and_resize_fast_tier_over_a_slack_fifo_match_too() {
		let ops = skewed_ops();
		let mut a =
			TwoQFastAdmissionReprieveHybridStack::new(0.25, MAX, 65_536).with_shared_overhead(112);
		let mut b = TwoQFastAdmissionReprieveCompactHybridStack::new(0.25, MAX, 65_536)
			.with_shared_overhead(112);
		let (mut ma, mut mb) = (Vec::new(), Vec::new());

		for (k, size) in ops.iter().take(8_000) {
			if a.contains(*k) { a.update(*k); } else { a.insert(*k, *size); }
			if b.contains(*k) { b.update(*k); } else { b.insert(*k, *size); }
			ma.extend(a.drain_tier_migrations());
			mb.extend(b.drain_tier_migrations());
		}

		a.resize(MAX / 4);
		b.resize(MAX / 4);
		ma.extend(a.drain_tier_migrations());
		mb.extend(b.drain_tier_migrations());

		for size in [16_384u64, 1_048_576, 8_192, 0] {
			a.resize_fast_tier(size);
			b.resize_fast_tier(size);
			ma.extend(a.drain_tier_migrations());
			mb.extend(b.drain_tier_migrations());
			assert_eq!(ma, mb, "migrations diverge after resize_fast_tier({size})");
			assert_eq!(a.fast_bytes_used(), b.fast_bytes_used(), "fast bytes at {size}");
			assert_eq!(a.slow_bytes_used(), b.slow_bytes_used(), "slow bytes at {size}");
			assert_eq!(a.fast_object_count(), b.fast_object_count(), "fast objects at {size}");
			assert_eq!(a.slow_object_count(), b.slow_object_count(), "slow objects at {size}");
		}

		for i in 0..2_000u64 {
			a.insert(10_000 + i, 1024);
			b.insert(10_000 + i, 1024);
			ma.extend(a.drain_tier_migrations());
			mb.extend(b.drain_tier_migrations());
		}
		assert_eq!(ma, mb, "migrations diverge across the post-resize refill");
		assert_eq!(a.needs_capacity_eviction(), b.needs_capacity_eviction());
	}

	/// Eviction order, which this variant reverses: main's tail first, and the
	/// FIFO's only once main is empty.
	///
	/// The fresh-key tail is load-bearing for the same reason as above. Driving
	/// the skewed workload alone leaves the FIFO empty -- every one of its 200
	/// keys is promoted into main -- and with an empty FIFO the two orders are
	/// identical, so the test passed against a stack that drained the FIFO
	/// first.
	#[test]
	fn evicts_in_the_same_order() {
		let ops = skewed_ops();
		for (k_in, fast) in [(0.25f64, 32_768u64), (0.05, 500_000)] {
			let mut a =
				TwoQFastAdmissionReprieveHybridStack::new(k_in, MAX, fast).with_shared_overhead(112);
			let mut b = TwoQFastAdmissionReprieveCompactHybridStack::new(k_in, MAX, fast)
				.with_shared_overhead(112);

			for (k, size) in &ops {
				if a.contains(*k) { a.update(*k); } else { a.insert(*k, *size); }
				if b.contains(*k) { b.update(*k); } else { b.insert(*k, *size); }
				a.drain_tier_migrations();
				b.drain_tier_migrations();
			}

			// A tail of keys that are never touched again, so the FIFO is still
			// occupied when the drain starts and both queues are non-empty.
			for i in 0..FRESH_COUNT {
				a.insert(FRESH + i, 1024);
				b.insert(FRESH + i, 1024);
			}
			a.drain_tier_migrations();
			b.drain_tier_migrations();

			assert_eq!(
				a.tier_of(FRESH + FRESH_COUNT - 1),
				Some(Tier::Fast),
				"the FIFO must be non-empty at drain time, k_in {k_in} fast {fast}",
			);
			assert!(
				a.slow_object_count() > 0,
				"main must be non-empty at drain time, k_in {k_in} fast {fast}",
			);

			let mut ea = Vec::new();
			let mut eb = Vec::new();
			while let Some(k) = a.evict_one() { ea.push(k); }
			while let Some(k) = b.evict_one() { eb.push(k); }

			assert_eq!(ea, eb, "eviction order diverges k_in {k_in} fast {fast}");
			assert_eq!(a.len(), 0, "eviction should drain both queues");
			assert_eq!(b.len(), 0, "eviction should drain both queues");
		}
	}

	/// The dropped `needs_capacity_eviction` override, pinned against the
	/// baseline in the state where the non-reprieve stack's
	/// `fifo_used > fifo_capacity` would fire.
	///
	/// It cannot fire here, in either stack: `settle_fifo_queue` has already
	/// brought `fifo_used` under `effective_fifo_capacity`, which is itself at
	/// most `fifo_capacity`. That is the whole reason the baseline drops the
	/// override, and it is why this assertion is an equality against the
	/// baseline rather than a claim that the two answers ever differ.
	#[test]
	fn a_saturated_fifo_still_asks_for_no_capacity_eviction() {
		for overhead in [0u64, 112] {
			let mut a = TwoQFastAdmissionReprieveHybridStack::new(0.25, MAX, 1_000_000)
				.with_shared_overhead(overhead);
			let mut b = TwoQFastAdmissionReprieveCompactHybridStack::new(0.25, MAX, 1_000_000)
				.with_shared_overhead(overhead);

			for i in 0..FRESH_COUNT {
				a.insert(FRESH + i, 1024);
				b.insert(FRESH + i, 1024);
			}
			a.drain_tier_migrations();
			b.drain_tier_migrations();

			assert_eq!(
				a.tier_of(FRESH + FRESH_COUNT - 1),
				Some(Tier::Fast),
				"the FIFO must be saturated, or this test proves nothing",
			);
			assert_eq!(
				a.needs_capacity_eviction(),
				b.needs_capacity_eviction(),
				"needs_capacity_eviction diverges at overhead {overhead}",
			);
			assert!(
				!b.needs_capacity_eviction(),
				"a settled FIFO asks for no eviction at overhead {overhead}",
			);
		}
	}

	/// Removal from either queue, interleaved with the reprieve, keeps every
	/// counter in step.
	#[test]
	fn removal_matches_the_baseline() {
		let ops = skewed_ops();
		let mut a =
			TwoQFastAdmissionReprieveHybridStack::new(0.05, MAX, 65_536).with_shared_overhead(112);
		let mut b = TwoQFastAdmissionReprieveCompactHybridStack::new(0.05, MAX, 65_536)
			.with_shared_overhead(112);
		let (mut ma, mut mb) = (Vec::new(), Vec::new());

		for (i, (k, size)) in ops.iter().take(10_000).enumerate() {
			if a.contains(*k) { a.update(*k); } else { a.insert(*k, *size); }
			if b.contains(*k) { b.update(*k); } else { b.insert(*k, *size); }
			if i % 37 == 0 {
				a.remove(*k);
				b.remove(*k);
			}
			ma.extend(a.drain_tier_migrations());
			mb.extend(b.drain_tier_migrations());
		}

		assert_eq!(ma, mb, "migrations diverge across removals");
		assert_eq!(a.len(), b.len(), "len diverges");
		assert_eq!(a.fast_bytes_used(), b.fast_bytes_used(), "fast bytes");
		assert_eq!(a.slow_bytes_used(), b.slow_bytes_used(), "slow bytes");
		assert_eq!(a.fast_object_count(), b.fast_object_count(), "fast objects");
		assert_eq!(a.slow_object_count(), b.slow_object_count(), "slow objects");
	}

	/// `clear` and re-fill, since the compact stack has a slab and a free list
	/// to reset where the baseline has three collections.
	#[test]
	fn clear_then_refill_matches_the_baseline() {
		let ops = skewed_ops();
		let mut a =
			TwoQFastAdmissionReprieveHybridStack::new(0.05, MAX, 65_536).with_shared_overhead(112);
		let mut b = TwoQFastAdmissionReprieveCompactHybridStack::new(0.05, MAX, 65_536)
			.with_shared_overhead(112);

		for (k, size) in ops.iter().take(5_000) {
			if a.contains(*k) { a.update(*k); } else { a.insert(*k, *size); }
			if b.contains(*k) { b.update(*k); } else { b.insert(*k, *size); }
		}

		a.clear();
		b.clear();
		assert_eq!(a.len(), 0);
		assert_eq!(b.len(), 0);

		let (mut ma, mut mb) = (Vec::new(), Vec::new());
		for (k, size) in ops.iter().take(5_000) {
			if a.contains(*k) { a.update(*k); } else { a.insert(*k, *size); }
			if b.contains(*k) { b.update(*k); } else { b.insert(*k, *size); }
			ma.extend(a.drain_tier_migrations());
			mb.extend(b.drain_tier_migrations());
		}

		assert_eq!(ma, mb, "migrations diverge after clear");
		assert_eq!(a.len(), b.len());
		assert_eq!(a.fast_bytes_used(), b.fast_bytes_used());
		assert_eq!(a.slow_bytes_used(), b.slow_bytes_used());
	}

	/// Re-set of a live key (the early-return path of `insert_resident`, which
	/// deliberately does NOT settle the FIFO), with sizes that change.
	#[test]
	fn resetting_a_live_key_matches_the_baseline() {
		let mut a =
			TwoQFastAdmissionReprieveHybridStack::new(0.05, MAX, 65_536).with_shared_overhead(112);
		let mut b = TwoQFastAdmissionReprieveCompactHybridStack::new(0.05, MAX, 65_536)
			.with_shared_overhead(112);
		let (mut ma, mut mb) = (Vec::new(), Vec::new());

		for k in 1..=200u64 {
			a.insert(k, 512);
			b.insert(k, 512);
			ma.extend(a.drain_tier_migrations());
			mb.extend(b.drain_tier_migrations());
		}
		for k in 1..=200u64 {
			let size = (k as ObjectSize % 7 + 1) * 700;
			a.insert(k, size);
			b.insert(k, size);
			ma.extend(a.drain_tier_migrations());
			mb.extend(b.drain_tier_migrations());
		}

		assert_eq!(ma, mb, "migrations diverge across re-sets");
		assert_eq!(a.len(), b.len());
		for k in 1..=200u64 {
			assert_eq!(a.tier_of(k), b.tier_of(k), "tier of {k} diverges");
		}
		assert_eq!(a.fast_bytes_used(), b.fast_bytes_used());
		assert_eq!(a.slow_bytes_used(), b.slow_bytes_used());
	}
}
