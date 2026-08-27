/*
 * Copyright (c) Kia Shakiba
 *
 * This source code is licensed under the GNU AGPLv3 license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! Slab-backed S3-FIFO lazy-demotion + fast-admission + midpoint + reprieve
//! hybrid: behaviourally identical to
//! `S3FifoLazyDemotionFastAdmissionMidpointReprieveHybridStack`, with one
//! structure where that has four.
//!
//! That stack keeps THREE `HashList`s -- `one_access_queue`, `main_fast`,
//! `main_slow` -- each owning its OWN key-to-node index, plus a separate
//! `entries` map holding the 8-byte payload. A key is in exactly one of the
//! three queues at any moment, so a single [`CompactQueueSet`] holds all three
//! orders over one slab, with the payload riding in the index value.
//!
//! ```text
//! Q_ONE_ACCESS (0)  admission FIFO, DRAM-resident (fast admission)
//! Q_MAIN_FAST  (1)  front = newest fast key, back = the demotion candidate
//! Q_MAIN_SLOW  (2)  front = the fast/slow boundary, back = eviction candidate
//! ```
//!
//! Every boundary crossing that the single-list-plus-cursor stacks in this
//! family express as a relabel plus a `before()` step is here an unlink and a
//! relink between two of those queues -- and, because `CompactQueueSet` keeps
//! all three over one slab, it is a handful of `u32` writes with the slot and
//! the payload staying exactly where they were.
//!
//! # What separates this from [`S3FifoCompactHybridStack`]
//!
//! 1. **Fast admission.** The one-access queue is DRAM. `tier_of` reports
//!    `Tier::Fast` for a key in it, `fast_bytes_used`/`fast_object_count`
//!    count it and `slow_bytes_used`/`slow_object_count` no longer do, and a
//!    promotion out of it emits NO `Tier::Fast` migration -- those bytes are
//!    already physically DRAM, so a migration would copy correct DRAM bytes
//!    into a fresh DRAM buffer for nothing.
//!
//! 2. **A split, proportionally-charged fast budget.** `one_access_capacity`
//!    is a fixed carve-out of `fast_capacity`, and the shared-metadata
//!    reservation is split *proportionally* between the two fast segments
//!    (`reserved_shares`, the scheme `LruSizedHybridStack` uses), so
//!    `effective_one_access_capacity() + effective_main_fast_capacity() +
//!    reserved_overhead() == fast_capacity` at every settled point. Both
//!    resize entry points therefore settle BOTH segments.
//!
//! 3. **Lazy demotion as two physical lists.** Demotion is
//!    `Q_MAIN_FAST` tail -> front of `Q_MAIN_SLOW`; promotion is the reverse;
//!    eviction takes the `Q_MAIN_SLOW` tail, falling back to the
//!    `Q_MAIN_FAST` tail only when nothing has ever been demoted. There is no
//!    `main_boundary` cursor to maintain, and no `main_capacity`.
//!
//! 4. **A reprieve at demotion time.** `settle_fast_tier` moves an accessed
//!    `Q_MAIN_FAST` tail back to the front of its own queue with the
//!    reference bit cleared, instead of demoting it, and tries the next
//!    candidate.
//!
//! 5. **A reprieve out of the one-access queue.** An aged-out one-access key
//!    is spliced into the front of `Q_MAIN_SLOW` as `Tier::Slow` rather than
//!    evicted (`settle_one_access`), and that runs SYNCHRONOUSLY from
//!    `insert`/`resize`/`resize_fast_tier` -- never through
//!    `evict_one`/`needs_capacity_eviction`, which would ask
//!    `apply_evictions` to erase the key from the whole cache. So
//!    `needs_capacity_eviction` stays at the trait default `false`, and
//!    `evict_one` never touches the one-access queue.
//!
//! 6. **A mid-slow-segment checkpoint.** `slow_midpoint` tracks
//!    (approximately) the middle of `Q_MAIN_SLOW` via a drift counter --
//!    every second qualifying mutation steps the cursor one position toward
//!    the front -- and `evict_one` checks its reference bit before walking the
//!    tail, promoting a re-accessed midpoint key early. Because `Q_MAIN_SLOW`
//!    is homogeneous, a `before()` walk inside it can never wander into
//!    fast-tagged territory, so no "is it still Slow?" filter is needed at any
//!    redirect site. See the design notes below the `PolicyStack` impl in the
//!    stack this replaces for the drift derivation.

use crate::{
	object::ObjectSize,
	worker::policy::policy_stack::{
		compact_queue_set::CompactQueueSet, narrow_resident, watermarks, CacheSize, HashedKey,
		PolicyStack, Tier,
	},
	PaperPolicy,
};

/// Queue slots in the shared set. A key is in exactly one of the three.
const Q_ONE_ACCESS: usize = 0;
const Q_MAIN_FAST: usize = 1;
const Q_MAIN_SLOW: usize = 2;

/// Which live queue a key currently belongs to. `Main` covers both physical
/// main lists; `tier` says which one.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Queue {
	OneAccess,
	Main,
}

/// Combined per-key bookkeeping, carried in the index value.
///
/// `tier`/`accessed` are only meaningful while `queue == Main`. `tier` is
/// redundant with which of the two main lists the key is physically in, but
/// kept because `tier_of()` and the `PolicyWorker` migration path both want it
/// as a cheap single-probe lookup rather than a pair of `contains()` probes --
/// which is exactly what the index-value layout makes it here.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct S3FifoMidpointReprievePayload {
	queue: Queue,
	tier: Option<Tier>,
	/// The part of `size` that stays in DRAM in either tier; see `migrating`.
	dram_resident: u8,
	accessed: bool,
	size: ObjectSize,
}

/// Pinned, exactly as `S3FifoEntry` is in the stack this replaces. The payload
/// rides in the index bucket, so growth here costs bytes on every tracked key.
const _: () = assert!(
	std::mem::size_of::<S3FifoMidpointReprievePayload>() == 8,
	"S3FifoMidpointReprievePayload grew past 8 bytes",
);

impl S3FifoMidpointReprievePayload {
	/// The bytes that actually move between tiers when this object migrates.
	fn migrating(&self) -> CacheSize {
		(self.size as CacheSize).saturating_sub(self.dram_resident as CacheSize)
	}
}

pub struct S3FifoLazyDemotionFastAdmissionMidpointReprieveCompactHybridStack {
	queues: CompactQueueSet<S3FifoMidpointReprievePayload>,

	one_access_ratio: f64,
	one_access_capacity: CacheSize,
	one_access_used: CacheSize,

	/// The configured total fast-tier (DRAM) budget, shared between the
	/// one-access queue and the main queue's fast segment. There is
	/// deliberately no `main_capacity`: this variant derives no budget from
	/// `1 - one_access_ratio` and never gates eviction on main fullness.
	fast_capacity: CacheSize,
	fast_used: CacheSize,
	slow_used: CacheSize,

	shared_overhead: CacheSize,

	/// Cursor at (approximately) the middle of `Q_MAIN_SLOW`.
	slow_midpoint: Option<HashedKey>,
	midpoint_drift: u8,

	migrations: Vec<(HashedKey, Tier)>,
}

impl S3FifoLazyDemotionFastAdmissionMidpointReprieveCompactHybridStack {
	pub fn new(one_access_ratio: f64, max_size: CacheSize, fast_capacity: CacheSize) -> Self {
		S3FifoLazyDemotionFastAdmissionMidpointReprieveCompactHybridStack {
			queues: CompactQueueSet::default(),
			one_access_ratio,
			one_access_capacity: (one_access_ratio * max_size as f64) as CacheSize,
			one_access_used: 0,
			fast_capacity,
			fast_used: 0,
			slow_used: 0,
			shared_overhead: 0,
			slow_midpoint: None,
			midpoint_drift: 0,
			migrations: Vec::new(),
		}
	}

	pub fn with_shared_overhead(mut self, overhead: CacheSize) -> Self {
		self.shared_overhead = overhead;


		self
	}

	fn reserved_overhead(&self) -> CacheSize {
		self.queues.len() as CacheSize * self.shared_overhead
	}

	/// Splits `reserved_overhead()` proportionally between this stack's two
	/// independently-capacitied FAST segments -- the one-access queue and the
	/// main queue's fast portion -- returned as `(one_access_share,
	/// main_share)`. `u128` intermediate so the product cannot overflow;
	/// remainder handed to the main segment so the two shares always re-sum
	/// exactly. `(0, 0)` if both capacities are zero.
	fn reserved_shares(&self) -> (CacheSize, CacheSize) {
		let reserved = self.reserved_overhead();

		let main_capacity = self.fast_capacity.saturating_sub(self.one_access_capacity);
		let total_capacity = self.one_access_capacity + main_capacity;

		if total_capacity == 0 {
			return (0, 0);
		}

		let one_access_share =
			((reserved as u128 * self.one_access_capacity as u128) / total_capacity as u128) as CacheSize;
		let main_share = reserved.saturating_sub(one_access_share);

		(one_access_share, main_share)
	}

	/// The one-access queue's own byte cap after giving up its share of the
	/// shared-metadata reservation. What `settle_one_access` settles against.
	fn effective_one_access_capacity(&self) -> CacheSize {
		self.one_access_capacity.saturating_sub(self.reserved_shares().0)
	}

	/// The budget actually available to the main queue's fast segment: raw
	/// `fast_capacity`, minus the one-access queue's fixed carve-out, minus
	/// this segment's share of the shared-metadata reservation. The watermarks
	/// sit on top of this number, never in place of any part of it.
	fn effective_main_fast_capacity(&self) -> CacheSize {
		self.fast_capacity
			.saturating_sub(self.one_access_capacity)
			.saturating_sub(self.reserved_shares().1)
	}

	pub fn tier_of(&self, key: HashedKey) -> Option<Tier> {
		let payload = self.queues.payload(key)?;

		match payload.queue {
			// The one-access queue is DRAM-resident in this variant.
			Queue::OneAccess => Some(Tier::Fast),
			Queue::Main => payload.tier,
		}
	}

	pub fn is_midpoint(&self, key: HashedKey) -> bool {
		self.slow_midpoint == Some(key)
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

	/// The hottest per-get operation in this family, and the reason the payload
	/// lives in the index value: one probe, no slab access, no queue movement.
	fn mark_accessed(&mut self, key: HashedKey) {
		if let Some(p) = self.queues.payload_mut(key) {
			p.accessed = true;
		}
	}

	/// Moves a re-accessed one-access-queue key into `Q_MAIN_FAST`. Emits no
	/// migration for the promotion itself -- the key's bytes are already
	/// physically Fast in this variant.
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

	/// One step of the midpoint cursor toward the front of `Q_MAIN_SLOW`.
	fn nudge_midpoint_toward_front(&mut self) {
		let Some(current) = self.slow_midpoint else { return };

		if let Some(candidate) = self.queues.before(current) {
			self.slow_midpoint = Some(candidate);
		}
	}

	/// One unit of accumulated drift; every second one is worth a full
	/// position, so the cursor steps then.
	fn bump_midpoint_drift(&mut self) {
		self.midpoint_drift += 1;

		if self.midpoint_drift >= 2 {
			self.midpoint_drift = 0;
			self.nudge_midpoint_toward_front();
		}
	}

	/// Steps the cursor off `key` before it leaves `Q_MAIN_SLOW`. Must run
	/// BEFORE the unlink, while `before(key)` still names its neighbour.
	fn redirect_midpoint_before_removing(&mut self, key: HashedKey) {
		if self.slow_midpoint != Some(key) {
			return;
		}

		self.slow_midpoint = self.queues.before(key);
	}

	/// The mid-slow-segment checkpoint: a re-accessed midpoint key is promoted
	/// early rather than waiting to reach the tail.
	fn check_slow_midpoint(&mut self) {
		let Some(candidate) = self.slow_midpoint else { return };

		let accessed = self.queues.payload(candidate).map(|p| p.accessed).unwrap_or(false);

		if accessed {
			self.give_second_chance(candidate);
		}
	}

	/// An accessed key gets a fresh start instead of being evicted: at the
	/// front of `Q_MAIN_FAST` if it is already fast, or lifted out of
	/// `Q_MAIN_SLOW` into it if it is not.
	///
	/// The Slow branch is a real physical move -- the key genuinely was in
	/// PMEM -- so it pushes a `Tier::Fast` migration, unlike
	/// `promote_from_one_access`.
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
				self.redirect_midpoint_before_removing(key);
				self.queues.move_to_front_of(Q_MAIN_SLOW, Q_MAIN_FAST, key);

				if let Some(p) = self.queues.payload_mut(key) {
					p.tier = Some(Tier::Fast);
					p.accessed = false;
				}

				self.slow_used = self.slow_used.saturating_sub(size);
				self.fast_used += size;

				self.bump_midpoint_drift();
			},

			None => return,
		}

		self.settle_fast_tier();

		if self.queues.payload(key).and_then(|p| p.tier) == Some(Tier::Fast) {
			self.migrations.push((key, Tier::Fast));
		}
	}

	/// Demotes the `Q_MAIN_FAST` tail into the front of `Q_MAIN_SLOW` once
	/// `fast_used` crosses the HIGH watermark of
	/// `effective_main_fast_capacity()`, and keeps going until it is back at or
	/// below the LOW watermark -- reference-bit gated, so an accessed candidate
	/// is reprieved to the front of its own queue instead.
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

			self.queues.move_to_front_of(Q_MAIN_FAST, Q_MAIN_SLOW, candidate);

			if let Some(p) = self.queues.payload_mut(candidate) {
				p.tier = Some(Tier::Slow);
			}

			self.fast_used = self.fast_used.saturating_sub(size);
			self.slow_used += size;

			self.migrations.push((candidate, Tier::Slow));

			if self.slow_midpoint.is_none() {
				self.slow_midpoint = Some(candidate);
			} else {
				self.bump_midpoint_drift();
			}
		}
	}

	/// The one-access reprieve. Splices the one-access tail into the FRONT of
	/// `Q_MAIN_SLOW` as `Tier::Slow` until the queue is back inside its
	/// effective budget. Deliberately not watermarked: `one_access_capacity`
	/// is a queue-length rule of the S3-FIFO design, not a tier-pressure
	/// threshold.
	fn settle_one_access(&mut self) {
		let effective_capacity = self.effective_one_access_capacity();

		while self.one_access_used > effective_capacity {
			let Some(key) = self.queues.back(Q_ONE_ACCESS) else { break };

			// Unreachable -- `back` returned the key, so it is indexed. The
			// baseline's `continue` on a missing entry is kept in shape here,
			// dropping the link so the loop cannot spin.
			let Some(payload) = self.queues.payload(key) else {
				self.queues.remove(Q_ONE_ACCESS, key);
				continue;
			};

			let size = payload.migrating();

			self.one_access_used = self.one_access_used.saturating_sub(size);
			self.queues.move_to_front_of(Q_ONE_ACCESS, Q_MAIN_SLOW, key);

			if let Some(p) = self.queues.payload_mut(key) {
				p.queue = Queue::Main;
				p.tier = Some(Tier::Slow);
				p.accessed = false;
			}

			self.slow_used += size;

			self.migrations.push((key, Tier::Slow));

			if self.slow_midpoint.is_none() {
				self.slow_midpoint = Some(key);
			} else {
				self.bump_midpoint_drift();
			}
		}
	}
}

impl PolicyStack for S3FifoLazyDemotionFastAdmissionMidpointReprieveCompactHybridStack {
	fn is_policy(&self, policy: &PaperPolicy) -> bool {
		matches!(policy, PaperPolicy::S3FifoLazyDemotionFastAdmissionMidpointReprieveCompactHybrid(r) if *r == self.one_access_ratio)
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
			S3FifoMidpointReprievePayload {
				queue: Queue::OneAccess,
				tier: None,
				dram_resident,
				accessed: false,
				size,
			},
		);
		self.one_access_used += (size as CacheSize).saturating_sub(dram_resident as CacheSize);

		// The reprieve, synchronously: an admission that pushes the one-access
		// queue over budget spills its tail into the main queue's slow segment
		// here, never through `evict_one`.
		self.settle_one_access();

		// The metadata reservation scales with the tracked key count, so an
		// admission tightens the main fast segment's budget too.
		self.settle_fast_tier();
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

			Queue::Main => match payload.tier {
				Some(Tier::Fast) => {
					self.queues.remove(Q_MAIN_FAST, key);
					self.fast_used = self.fast_used.saturating_sub(size);
				},

				Some(Tier::Slow) => {
					self.redirect_midpoint_before_removing(key);
					self.queues.remove(Q_MAIN_SLOW, key);
					self.slow_used = self.slow_used.saturating_sub(size);
					self.bump_midpoint_drift();
				},

				// Unreachable: `tier` is `None` only while
				// `queue == Queue::OneAccess`. The baseline leaves the queue
				// lists alone in this arm too.
				None => {},
			},
		}
	}

	fn resize(&mut self, max_size: CacheSize) {
		self.one_access_capacity = (self.one_access_ratio * max_size as f64) as CacheSize;

		// Both budgets moved: the one-access cap directly, and the main fast
		// segment's because it is `fast_capacity` minus that cap.
		self.settle_one_access();
		self.settle_fast_tier();
	}

	fn clear(&mut self) {
		self.queues.clear();

		self.one_access_used = 0;
		self.fast_used = 0;
		self.slow_used = 0;
		self.slow_midpoint = None;
		self.midpoint_drift = 0;
		self.migrations.clear();
	}

	fn evict_one(&mut self) -> Option<HashedKey> {
		// The midpoint checkpoint runs first, and only once per call.
		self.check_slow_midpoint();

		loop {
			// The one-access queue is never an eviction candidate here: it is
			// drained synchronously by `settle_one_access` instead.
			let (key, from_slow) = match self.queues.back(Q_MAIN_SLOW) {
				Some(key) => (key, true),
				None => (self.queues.back(Q_MAIN_FAST)?, false),
			};

			let accessed = self.queues.payload(key).map(|p| p.accessed).unwrap_or(false);

			if accessed {
				self.give_second_chance(key);
				continue;
			}

			let payload = if from_slow {
				self.redirect_midpoint_before_removing(key);
				self.queues.remove(Q_MAIN_SLOW, key)
			} else {
				self.queues.remove(Q_MAIN_FAST, key)
			};

			let size = payload.map(|p| p.migrating()).unwrap_or(0);

			if from_slow {
				self.slow_used = self.slow_used.saturating_sub(size);
				self.bump_midpoint_drift();
			} else {
				self.fast_used = self.fast_used.saturating_sub(size);
			}

			return Some(key);
		}
	}

	fn resize_fast_tier(&mut self, size: CacheSize) {
		self.fast_capacity = size;

		// `fast_capacity` is one of the two inputs to the proportional split of
		// the metadata reservation, so changing it re-proportions the
		// one-access queue's share as well -- settle that segment first (a
		// reprieve out of it only adds slow-tier bytes, so it can never make
		// the fast/slow settle below harder).
		self.settle_one_access();
		self.settle_fast_tier();
	}

	fn drain_tier_migrations(&mut self) -> Vec<(HashedKey, Tier)> {
		std::mem::take(&mut self.migrations)
	}

	fn dram_reserved_bytes(&self) -> CacheSize {
		self.reserved_overhead()
	}

	fn fast_bytes_used(&self) -> CacheSize {
		// Both fast segments: the main queue's, and the one-access queue's.
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

	// NO `needs_capacity_eviction` override, matching the baseline: the
	// one-access queue settles itself synchronously, so the trait default
	// (`false`) is the answer. Routing the reprieve through `evict_one` would
	// have `apply_evictions` erase the key from the entire cache.
}

/// Fidelity against `S3FifoLazyDemotionFastAdmissionMidpointReprieveHybridStack`,
/// which this stack is a compaction of.
///
/// SATURATION IS LOAD-BEARING, in two directions. A purely skewed workload over
/// a small key space leaves the one-access queue EMPTY -- every key is promoted
/// on its second access -- so `settle_one_access`, the midpoint cursor's drift
/// under reprieves, and the whole Slow segment are barely exercised. The
/// `cold_tail_ops` workload below therefore mixes a hot set with a stream of
/// keys seen exactly once, and the tests that matter assert the reprieve
/// actually fired before they measure anything.
#[cfg(all(test, feature = "s3_fifo_lazy_demotion_fast_admission_midpoint_reprieve_hybrid_cache"))]
mod fidelity_tests {
	use super::*;
	use crate::worker::policy::policy_stack::s3_fifo_lazy_demotion_fast_admission_midpoint_reprieve_hybrid_stack::S3FifoLazyDemotionFastAdmissionMidpointReprieveHybridStack as Baseline;

	type Compact = S3FifoLazyDemotionFastAdmissionMidpointReprieveCompactHybridStack;

	const MAX: CacheSize = 1_000_000;

	/// Repeats are essential: a key only leaves the one-access queue on a
	/// SECOND access, and the reference bit only matters on a third.
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

	/// A hot set plus a long cold tail of keys seen exactly once. This is what
	/// makes the one-access queue overflow, so `settle_one_access` -- and the
	/// midpoint drift it drives -- actually run.
	fn cold_tail_ops() -> Vec<(HashedKey, ObjectSize)> {
		let mut ops = Vec::new();
		let mut x: u64 = 0x1357_9BDF_2468_ACE0;
		for i in 0..20_000u64 {
			x ^= x << 13;
			x ^= x >> 7;
			x ^= x << 17;
			if i % 3 == 0 {
				let u = (x >> 11) as f64 / (1u64 << 53) as f64;
				ops.push((((u * u * 50.0) as u64) + 1, 1024));
			} else {
				ops.push((1_000_000 + i, 1024));
			}
		}
		ops
	}

	fn step(a: &mut Baseline, b: &mut Compact, k: HashedKey, size: ObjectSize) {
		if a.contains(k) { a.update(k); } else { a.insert(k, size); }
		if b.contains(k) { b.update(k); } else { b.insert(k, size); }
	}

	struct Replay {
		ma: Vec<(HashedKey, Tier)>,
		mb: Vec<(HashedKey, Tier)>,
		ta: Vec<Option<Tier>>,
		tb: Vec<Option<Tier>>,
		/// Per-op `is_midpoint` of the key just touched, from each stack.
		pa: Vec<bool>,
		pb: Vec<bool>,
		/// `is_midpoint` over every distinct key at the end.
		fa: Vec<bool>,
		fb: Vec<bool>,
		slow_migrations: usize,
	}

	fn replay(
		ratio: f64,
		fast: CacheSize,
		overhead: CacheSize,
		ops: &[(HashedKey, ObjectSize)],
	) -> Replay {
		let mut a = Baseline::new(ratio, MAX, fast).with_shared_overhead(overhead);
		let mut b = Compact::new(ratio, MAX, fast).with_shared_overhead(overhead);
		let mut r = Replay {
			ma: Vec::new(), mb: Vec::new(),
			ta: Vec::new(), tb: Vec::new(),
			pa: Vec::new(), pb: Vec::new(),
			fa: Vec::new(), fb: Vec::new(),
			slow_migrations: 0,
		};

		for (k, size) in ops {
			step(&mut a, &mut b, *k, *size);
			r.pa.push(a.is_midpoint(*k));
			r.pb.push(b.is_midpoint(*k));
			r.ma.extend(a.drain_tier_migrations());
			r.mb.extend(b.drain_tier_migrations());
		}

		let mut keys: Vec<HashedKey> = ops.iter().map(|(k, _)| *k).collect();
		r.ta = keys.iter().map(|k| a.tier_of(*k)).collect();
		r.tb = keys.iter().map(|k| b.tier_of(*k)).collect();

		keys.sort_unstable();
		keys.dedup();
		r.fa = keys.iter().map(|k| a.is_midpoint(*k)).collect();
		r.fb = keys.iter().map(|k| b.is_midpoint(*k)).collect();

		r.slow_migrations = r.ma.iter().filter(|(_, t)| *t == Tier::Slow).count();

		assert_eq!(a.len(), b.len(), "tracked counts diverge");
		assert_eq!(a.fast_bytes_used(), b.fast_bytes_used(), "fast bytes diverge");
		assert_eq!(a.slow_bytes_used(), b.slow_bytes_used(), "slow bytes diverge");
		assert_eq!(a.fast_object_count(), b.fast_object_count(), "fast counts diverge");
		assert_eq!(a.slow_object_count(), b.slow_object_count(), "slow counts diverge");
		assert_eq!(a.dram_reserved_bytes(), b.dram_reserved_bytes(), "reservations diverge");
		assert_eq!(a.needs_capacity_eviction(), b.needs_capacity_eviction());

		r
	}

	fn assert_same(r: &Replay, what: &str) {
		assert_eq!(r.ta, r.tb, "tiers diverge {what}");
		assert_eq!(r.ma, r.mb, "migrations diverge {what}");
		assert_eq!(r.pa, r.pb, "midpoint cursor diverges {what}");
		assert_eq!(r.fa, r.fb, "final midpoint diverges {what}");
	}

	#[test]
	fn matches_the_baseline_migration_for_migration() {
		let ops = skewed_ops();
		for ratio in [0.05f64, 0.25, 0.5] {
			for fast in [8_192u64, 32_768, 131_072] {
				for overhead in [0u64, 112] {
					let r = replay(ratio, fast, overhead, &ops);
					assert_same(&r, &format!("at ratio {ratio} fast {fast} overhead {overhead}"));
				}
			}
		}
	}

	/// The one-access reprieve and the midpoint drift it drives. Asserts the
	/// reprieve actually fired, so the test cannot pass vacuously.
	#[test]
	fn matches_the_baseline_under_a_cold_tail_workload() {
		let ops = cold_tail_ops();
		for ratio in [0.001f64, 0.01, 0.1] {
			for fast in [8_192u64, 65_536] {
				for overhead in [0u64, 112] {
					let r = replay(ratio, fast, overhead, &ops);
					assert_same(&r, &format!("at ratio {ratio} fast {fast} overhead {overhead}"));
					assert!(
						r.slow_migrations > 100,
						"cold-tail workload produced only {} Slow migrations at ratio {ratio} fast {fast}: \
						 the one-access reprieve is not being exercised",
						r.slow_migrations,
					);
				}
			}
		}
	}

	/// Eviction is where the reference bit is acted on -- at the tail AND at
	/// the midpoint checkpoint -- and both reorder the queues mid-eviction.
	#[test]
	fn evicts_in_the_same_order_including_second_chances_and_the_midpoint_check() {
		for ops in [skewed_ops(), cold_tail_ops()] {
			let mut a = Baseline::new(0.01, MAX, 32_768).with_shared_overhead(112);
			let mut b = Compact::new(0.01, MAX, 32_768).with_shared_overhead(112);

			for (k, size) in &ops {
				step(&mut a, &mut b, *k, *size);
				a.drain_tier_migrations();
				b.drain_tier_migrations();
			}

			// Set reference bits on a slice of the live set so the tail loop
			// and the midpoint check both have something to act on.
			for k in 1..=50u64 {
				if a.contains(k) { a.update(k); }
				if b.contains(k) { b.update(k); }
			}
			a.drain_tier_migrations();
			b.drain_tier_migrations();

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

			assert_eq!(ea, eb, "eviction order diverges");
			assert_eq!(ma, mb, "mid-eviction migrations diverge");

			// The cursor must have been walked off every key eviction removed,
			// not left dangling on a freed one: a dangling cursor is invisible
			// in the eviction order and permanently stops tracking the middle.
			let mut keys: Vec<HashedKey> = ops.iter().map(|(k, _)| *k).collect();
			keys.sort_unstable();
			keys.dedup();
			let ia: Vec<bool> = keys.iter().map(|k| a.is_midpoint(*k)).collect();
			let ib: Vec<bool> = keys.iter().map(|k| b.is_midpoint(*k)).collect();
			assert_eq!(ia, ib, "midpoint cursor diverges after a full drain");
			assert!(!ib.iter().any(|m| *m), "cursor left dangling on an evicted key");

			assert_eq!(a.len(), b.len());
			// `evict_one` never drains the one-access queue in this design, so
			// both stacks stop with exactly the same residue.
			assert_eq!(a.fast_object_count(), b.fast_object_count());
			assert_eq!(b.slow_object_count(), 0);
		}
	}

	#[test]
	fn removal_matches_including_midpoint_redirect() {
		let ops = cold_tail_ops();
		let mut a = Baseline::new(0.01, MAX, 32_768).with_shared_overhead(112);
		let mut b = Compact::new(0.01, MAX, 32_768).with_shared_overhead(112);
		let (mut ma, mut mb) = (Vec::new(), Vec::new());
		let mut midpoint_removals = 0usize;

		for (i, (k, size)) in ops.iter().enumerate() {
			step(&mut a, &mut b, *k, *size);

			if i % 37 == 0 {
				let victim = (i as u64 % 50) + 1;
				a.remove(victim);
				b.remove(victim);
			}

			// Periodically remove the midpoint key ITSELF -- the arm the
			// cursor redirect exists for, which a blind victim choice
			// essentially never hits in a slow segment thousands of keys long.
			if i % 1_000 == 999 {
				let found = ops[..=i].iter().map(|(kk, _)| *kk).find(|kk| a.is_midpoint(*kk));

				if let Some(m) = found {
					assert!(b.is_midpoint(m), "midpoint cursor diverges at op {i}");
					a.remove(m);
					b.remove(m);
					assert_eq!(a.is_midpoint(m), b.is_midpoint(m));
					midpoint_removals += 1;
				}
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
		assert!(midpoint_removals > 0, "no removal ever hit the midpoint cursor");
	}

	/// Resize in both directions with BRAND-NEW keys afterwards. `resize` here
	/// rescales `one_access_capacity` only -- there is no `main_capacity` --
	/// and both entry points settle BOTH segments.
	#[test]
	fn resizes_like_the_baseline() {
		for (start, resized) in [(65_536u64, 65_536u64), (131_072, 32_768), (32_768, 131_072)] {
			let mut a = Baseline::new(0.25, MAX, start).with_shared_overhead(112);
			let mut b = Compact::new(0.25, MAX, start).with_shared_overhead(112);
			let (mut ma, mut mb) = (Vec::new(), Vec::new());

			for i in 0..4_000u64 {
				let k = (i % 200) + 1;
				step(&mut a, &mut b, k, 1024);
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

			// Shrink the one-access budget to nothing: every untouched
			// one-access key must be reprieved into main_slow, in order.
			a.resize(0);
			b.resize(0);
			ma.extend(a.drain_tier_migrations());
			mb.extend(b.drain_tier_migrations());

			for i in 0..2_000u64 {
				let k = 10_000 + i;
				a.insert(k, 1024);
				b.insert(k, 1024);
				ma.extend(a.drain_tier_migrations());
				mb.extend(b.drain_tier_migrations());
			}

			assert_eq!(ma, mb, "migrations diverge resizing {start} -> {resized}");
			for i in 0..2_000u64 {
				assert_eq!(a.tier_of(10_000 + i), b.tier_of(10_000 + i));
			}
			assert_eq!(a.fast_bytes_used(), b.fast_bytes_used());
			assert_eq!(a.slow_bytes_used(), b.slow_bytes_used());
		}
	}

	// ── the deltas unique to this variant ──────────────────────────────────
	//
	// Each of the tests below fails if the corresponding piece of the
	// behavioural delta was not ported. They are run against BOTH stacks, so
	// they double as fidelity checks and as pins on the delta itself.

	/// Smallest *effective* main-fast budget that holds a fast segment of
	/// exactly `bytes` across a settled pass. Same helper the baseline's own
	/// unit tests use, for the same reason: the watermarks are process-global
	/// `OnceLock`s, so a fixture cannot pin them.
	fn effective_capacity_holding(bytes: CacheSize, next: CacheSize) -> CacheSize {
		let mut capacity = (bytes as f64 / watermarks::low()).ceil() as CacheSize;

		while watermarks::low_bytes(capacity) < bytes {
			capacity += 1;
		}

		assert!(
			watermarks::high_bytes(capacity) < bytes + next,
			"watermark config leaves no room for this fixture",
		);

		capacity
	}

	/// FAST ADMISSION: a brand-new key is Fast, not Slow, and its bytes and
	/// object count land on the fast side of the gauges. Fails against the
	/// unmodified `S3FifoCompactHybridStack` behaviour.
	#[test]
	fn admission_lands_in_the_one_access_queue_at_fast() {
		let mut a = Baseline::new(1.0, 1_000, 1_000);
		let mut b = Compact::new(1.0, 1_000, 1_000);

		for k in 1..=2u64 {
			a.insert(k, 10);
			b.insert(k, 10);
		}

		assert_eq!(b.tier_of(1), Some(Tier::Fast));
		assert_eq!(b.tier_of(2), Some(Tier::Fast));
		assert_eq!(a.tier_of(1), b.tier_of(1));
		assert_eq!(b.drain_tier_migrations(), Vec::new());
		assert_eq!(a.drain_tier_migrations(), Vec::new());

		assert_eq!(b.fast_bytes_used(), 20);
		assert_eq!(b.slow_bytes_used(), 0);
		assert_eq!(b.fast_object_count(), 2);
		assert_eq!(b.slow_object_count(), 0);
		assert_eq!(a.fast_bytes_used(), b.fast_bytes_used());
		assert_eq!(a.fast_object_count(), b.fast_object_count());
	}

	/// THE ONE-ACCESS REPRIEVE: an aged-out one-access key is spliced into
	/// main_slow, still tracked, with a real Fast->Slow migration -- not
	/// evicted, and not left for `needs_capacity_eviction`.
	#[test]
	fn a_key_aging_out_is_reprieved_into_main_slow_not_evicted() {
		// one_access_capacity = 0.01 * 1_000 = 10: exactly one 10-byte key.
		let mut a = Baseline::new(0.01, 1_000, 1_000);
		let mut b = Compact::new(0.01, 1_000, 1_000);

		a.insert(1, 10);
		b.insert(1, 10);
		a.drain_tier_migrations();
		b.drain_tier_migrations();
		assert_eq!(b.tier_of(1), Some(Tier::Fast));

		a.insert(2, 10);
		b.insert(2, 10);

		assert!(b.contains(1), "the key must still be tracked, not gone");
		assert_eq!(b.tier_of(1), Some(Tier::Slow));
		assert_eq!(b.tier_of(2), Some(Tier::Fast));
		assert_eq!(b.drain_tier_migrations(), vec![(1, Tier::Slow)]);
		assert_eq!(a.tier_of(1), b.tier_of(1));
		assert!(!b.needs_capacity_eviction(), "the reprieve must not ask the caller to evict");
		assert!(!a.needs_capacity_eviction());

		// And it can still be promoted by a later access.
		a.insert(3, 10);
		b.insert(3, 10);
		a.drain_tier_migrations();
		b.drain_tier_migrations();

		a.update(1);
		b.update(1);
		assert_eq!(b.tier_of(1), Some(Tier::Slow), "a mere access must not migrate it");
		assert_eq!(b.drain_tier_migrations(), Vec::new());

		// main_slow is [2, 1] here, so the midpoint check promotes key 1 and
		// redirects the cursor onto key 2 -- which the same call then evicts.
		// The cursor has to step off it, or it dangles on a freed key forever.
		assert_eq!(a.evict_one(), b.evict_one());
		assert_eq!(b.tier_of(1), Some(Tier::Fast), "second chance at the tail");
		assert_eq!(a.tier_of(1), b.tier_of(1));
		assert!(!b.is_midpoint(2), "cursor left dangling on the evicted key");
		assert_eq!(a.is_midpoint(2), b.is_midpoint(2));
		assert_eq!(a.is_midpoint(1), b.is_midpoint(1));
		assert_eq!(a.is_midpoint(3), b.is_midpoint(3));
	}

	/// `resize_fast_tier` must settle the ONE-ACCESS queue too, not just the
	/// main fast segment: `fast_capacity` is an input to the proportional split
	/// of the metadata reservation, so shrinking it enlarges the one-access
	/// segment's share and therefore shrinks its effective budget.
	///
	/// The other resize test cannot see this -- it calls `resize` afterwards,
	/// which settles the one-access queue anyway and masks the omission.
	#[test]
	fn resize_fast_tier_alone_settles_the_one_access_queue() {
		// ratio 0.1 of 1_000_000 -> one_access_capacity 100_000. Starting at a
		// fast_capacity of 1_000_000 the one-access segment carries only a
		// tenth of the reservation; dropping to 100_000 leaves the main fast
		// segment nothing, so it carries ALL of it.
		let mut a = Baseline::new(0.1, MAX, 1_000_000).with_shared_overhead(112);
		let mut b = Compact::new(0.1, MAX, 1_000_000).with_shared_overhead(112);

		// 300 cold keys, never re-accessed, so they stay in the one-access
		// queue until a budget pushes them out.
		for k in 1..=300u64 {
			a.insert(k, 1024);
			b.insert(k, 1024);
		}
		a.drain_tier_migrations();
		b.drain_tier_migrations();

		let held_before = b.queues.queue_len(Q_ONE_ACCESS);
		assert!(held_before > 0, "the one-access queue should be holding keys");

		a.resize_fast_tier(100_000);
		b.resize_fast_tier(100_000);

		let ma = a.drain_tier_migrations();
		let mb = b.drain_tier_migrations();

		assert!(
			!mb.is_empty(),
			"shrinking fast_capacity must reprieve one-access keys via the reservation split",
		);
		assert_eq!(ma, mb, "resize_fast_tier migrations diverge");
		assert!(b.queues.queue_len(Q_ONE_ACCESS) < held_before, "the one-access queue did not shrink");

		for k in 1..=300u64 {
			assert_eq!(a.tier_of(k), b.tier_of(k), "key {k}");
			assert_eq!(a.is_midpoint(k), b.is_midpoint(k), "key {k}");
		}
		assert_eq!(a.fast_bytes_used(), b.fast_bytes_used());
		assert_eq!(a.slow_bytes_used(), b.slow_bytes_used());
		assert_eq!(a.fast_object_count(), b.fast_object_count());
		assert_eq!(a.slow_object_count(), b.slow_object_count());
	}

	/// THE DEMOTION-TIME REPRIEVE: an accessed fast tail is moved back to the
	/// front of main_fast rather than demoted, and the next candidate takes
	/// its place.
	#[test]
	fn an_accessed_fast_tail_is_reprieved_at_demotion_time() {
		let fast = 1_000 + effective_capacity_holding(10, 10);
		let mut a = Baseline::new(1.0, 1_000, fast);
		let mut b = Compact::new(1.0, 1_000, fast);

		a.insert(1, 10); a.update(1);
		b.insert(1, 10); b.update(1);
		a.drain_tier_migrations();
		b.drain_tier_migrations();

		// Sets key 1's reference bit -- no movement, no migration.
		a.update(1);
		b.update(1);
		assert_eq!(a.drain_tier_migrations(), Vec::new());
		assert_eq!(b.drain_tier_migrations(), Vec::new());

		a.insert(2, 10); a.update(2);
		b.insert(2, 10); b.update(2);
		let ma = a.drain_tier_migrations();
		let mb = b.drain_tier_migrations();

		// Without the reprieve, key 1 (the tail) would be the one demoted.
		assert_eq!(b.tier_of(1), Some(Tier::Fast));
		assert_eq!(b.tier_of(2), Some(Tier::Slow));
		assert_eq!(mb, vec![(2, Tier::Slow)]);
		assert_eq!(ma, mb);
		assert_eq!(a.tier_of(1), b.tier_of(1));
		assert_eq!(a.tier_of(2), b.tier_of(2));
	}

	/// Five keys promoted in order against a budget holding two of them:
	/// 1, 2, 3 demote oldest-first, leaving main_slow = [3, 2, 1], and the
	/// drift cursor settles on the middle element, key 2.
	fn build_five_key_stacks() -> (Baseline, Compact) {
		let fast = 1_000 + effective_capacity_holding(20, 10);
		let mut a = Baseline::new(1.0, 1_000, fast);
		let mut b = Compact::new(1.0, 1_000, fast);

		for key in 1..=5u64 {
			a.insert(key, 10); a.update(key);
			b.insert(key, 10); b.update(key);
		}

		a.drain_tier_migrations();
		b.drain_tier_migrations();
		(a, b)
	}

	/// THE MIDPOINT CURSOR: it names the middle of the slow segment.
	#[test]
	fn slow_midpoint_tracks_the_middle_of_the_slow_segment() {
		let (a, b) = build_five_key_stacks();

		for (k, t) in [(1u64, Tier::Slow), (2, Tier::Slow), (3, Tier::Slow), (4, Tier::Fast), (5, Tier::Fast)] {
			assert_eq!(b.tier_of(k), Some(t), "key {k}");
			assert_eq!(a.tier_of(k), b.tier_of(k), "key {k}");
		}

		assert!(b.is_midpoint(2), "expected key 2, the middle of slow segment [3, 2, 1]");
		for k in 1..=5u64 {
			assert_eq!(a.is_midpoint(k), b.is_midpoint(k), "key {k}");
		}
	}

	/// THE MIDPOINT CHECKPOINT: a re-accessed midpoint key is promoted early,
	/// from `evict_one`, instead of waiting to reach the tail. Without
	/// `check_slow_midpoint` key 2 would still be Slow after this call.
	#[test]
	fn a_reaccessed_midpoint_key_is_promoted_early() {
		let (mut a, mut b) = build_five_key_stacks();
		assert!(b.is_midpoint(2));

		a.update(2);
		b.update(2);
		assert_eq!(b.tier_of(2), Some(Tier::Slow), "a mere access must not migrate or reorder");

		let ea = a.evict_one();
		let eb = b.evict_one();

		assert_eq!(b.tier_of(2), Some(Tier::Fast), "the midpoint key should have been promoted early");
		assert_eq!(b.tier_of(4), Some(Tier::Slow), "cascading demotion after the midpoint promotion");
		assert_eq!(eb, Some(1), "the tail is still evicted normally in the same call");
		assert!(!b.contains(1));

		assert_eq!(ea, eb);
		assert_eq!(a.tier_of(2), b.tier_of(2));
		assert_eq!(a.tier_of(4), b.tier_of(4));
		assert_eq!(a.drain_tier_migrations(), b.drain_tier_migrations());
	}

	#[test]
	fn an_unaccessed_midpoint_key_is_left_alone() {
		let (mut a, mut b) = build_five_key_stacks();

		let ea = a.evict_one();
		let eb = b.evict_one();

		assert_eq!(b.tier_of(2), Some(Tier::Slow), "an unaccessed midpoint key must not be promoted");
		assert_eq!(eb, Some(1));
		assert_eq!(ea, eb);
		assert_eq!(a.tier_of(2), b.tier_of(2));
	}

	#[test]
	fn removing_the_midpoint_key_directly_redirects_the_cursor() {
		let (mut a, mut b) = build_five_key_stacks();
		assert!(b.is_midpoint(2));

		a.remove(2);
		b.remove(2);

		assert!(!b.is_midpoint(2));
		assert!(b.is_midpoint(3), "cursor should redirect to the before()-neighbour in main_slow");
		for k in 1..=5u64 {
			assert_eq!(a.is_midpoint(k), b.is_midpoint(k), "key {k}");
		}
	}

	/// `evict_one` reaches the fast tail only when nothing has ever been
	/// demoted -- and never touches the one-access queue at all.
	#[test]
	fn evict_one_falls_back_to_the_fast_tail_and_never_drains_one_access() {
		let mut a = Baseline::new(1.0, 1_000, 10_000);
		let mut b = Compact::new(1.0, 1_000, 10_000);

		for key in 1..=3u64 {
			a.insert(key, 10); a.update(key);
			b.insert(key, 10); b.update(key);
		}
		// A key that stays in the one-access queue, untouched.
		a.insert(9, 10);
		b.insert(9, 10);
		a.drain_tier_migrations();
		b.drain_tier_migrations();

		assert_eq!(b.slow_object_count(), 0, "nothing should have been demoted yet");

		assert_eq!(a.evict_one(), Some(1));
		assert_eq!(b.evict_one(), Some(1));
		assert_eq!(a.evict_one(), b.evict_one());
		assert_eq!(a.evict_one(), b.evict_one());

		// main_fast is empty now; the one-access resident is NOT a candidate.
		assert_eq!(b.evict_one(), None);
		assert_eq!(a.evict_one(), None);
		assert!(b.contains(9));
		assert_eq!(b.tier_of(9), Some(Tier::Fast));
		assert_eq!(a.contains(9), b.contains(9));
	}

	#[test]
	fn clear_resets_the_midpoint_and_the_drift() {
		let (mut a, mut b) = build_five_key_stacks();

		a.clear();
		b.clear();

		assert_eq!(b.len(), 0);
		assert_eq!(b.tier_of(2), None);
		assert_eq!(b.evict_one(), None);
		assert!(!b.is_midpoint(2));
		assert_eq!(b.fast_bytes_used(), 0);
		assert_eq!(b.slow_bytes_used(), 0);
		assert_eq!(a.len(), b.len());
		assert_eq!(a.is_midpoint(2), b.is_midpoint(2));
	}

	/// THE HIGH WATERMARK IS A CEILING, NOT A TRIGGER-AT-EQUAL: usage resting
	/// exactly ON `high_bytes(effective)` is still inside the budget, so
	/// `settle_fast_tier` must return without demoting anything. One byte past
	/// it is what trips the pass.
	#[test]
	fn usage_resting_exactly_on_the_high_watermark_does_not_demote() {
		// ratio 1.0 of 1_000 carves the whole 1_000 out for the one-access
		// queue, and no shared overhead leaves the reservation split at
		// (0, 0) -- so `effective_main_fast_capacity()` is exactly
		// `fast - 1_000`, with no rounding anywhere near the boundary.
		const EFFECTIVE: CacheSize = 1_000;
		let fast = 1_000 + EFFECTIVE;

		// Derived from the watermarks, never a hard-coded ratio: they are
		// process-global `OnceLock`s, so a reconfigured pair has to move this
		// fixture with it.
		let resting = watermarks::high_bytes(EFFECTIVE);
		assert!(resting > 0, "watermark config leaves no room for this fixture");

		let mut a = Baseline::new(1.0, 1_000, fast);
		let mut b = Compact::new(1.0, 1_000, fast);

		// Exactly `resting` ONE-BYTE keys promoted into main_fast: the insert
		// lands in the one-access queue, the update promotes it out.
		for k in 1..=resting {
			a.insert(k, 1); a.update(k);
			b.insert(k, 1); b.update(k);
		}

		assert_eq!(b.effective_main_fast_capacity(), EFFECTIVE);
		assert_eq!(
			b.fast_used,
			watermarks::high_bytes(b.effective_main_fast_capacity()),
			"the fixture must come to rest exactly ON the high watermark",
		);

		assert_eq!(
			b.drain_tier_migrations(),
			Vec::new(),
			"usage resting exactly on the high watermark must not demote",
		);
		assert_eq!(a.drain_tier_migrations(), Vec::new());
		assert_eq!(b.slow_object_count(), 0);
		assert_eq!(b.slow_bytes_used(), 0);
		assert_eq!(a.slow_object_count(), b.slow_object_count());
		assert_eq!(a.slow_bytes_used(), b.slow_bytes_used());
		assert_eq!(a.fast_object_count(), b.fast_object_count());
		assert_eq!(a.fast_bytes_used(), b.fast_bytes_used());

		// ...and one byte PAST it does demote, so the fixture is sitting on the
		// real boundary rather than somewhere harmlessly below it.
		a.insert(resting + 1, 1); a.update(resting + 1);
		b.insert(resting + 1, 1); b.update(resting + 1);
		let ma = a.drain_tier_migrations();
		let mb = b.drain_tier_migrations();

		assert!(!mb.is_empty(), "one byte past the high watermark must demote");
		assert_eq!(ma, mb, "migrations diverge one byte past the watermark");
		assert_eq!(a.slow_object_count(), b.slow_object_count());
		assert_eq!(a.slow_bytes_used(), b.slow_bytes_used());
	}

	/// THE RESERVATION SPLIT IS EXACT: `reserved_shares` floors the ONE-ACCESS
	/// share only and hands the remainder to the MAIN share, so the pair
	/// re-sums to `reserved_overhead()` and the identity
	/// `effective_one_access + effective_main_fast + reserved == fast_capacity`
	/// holds at every tracked count. Flooring both sides independently drops
	/// the odd byte and silently widens the main fast segment by one.
	#[test]
	fn the_reservation_split_hands_its_remainder_to_the_main_share() {
		// ratio 0.5 of 1_000 gives a one-access carve-out of 500 against a
		// fast_capacity of 1_000, so the split is `reserved * 500 / 1_000` --
		// inexact at every odd `reserved`, which a 3-byte-per-key overhead
		// produces at every odd tracked count.
		const FAST: CacheSize = 1_000;

		let mut a = Baseline::new(0.5, 1_000, FAST).with_shared_overhead(3);
		let mut b = Compact::new(0.5, 1_000, FAST).with_shared_overhead(3);
		let (mut ma, mut mb) = (Vec::new(), Vec::new());
		let mut inexact = 0usize;

		for k in 1..=280u64 {
			a.insert(k, 1); a.update(k);
			b.insert(k, 1); b.update(k);
			ma.extend(a.drain_tier_migrations());
			mb.extend(b.drain_tier_migrations());

			let reserved = b.reserved_overhead();
			let (one_share, main_share) = b.reserved_shares();

			assert_eq!(
				one_share + main_share,
				reserved,
				"the split lost a byte at {k} tracked keys",
			);
			assert_eq!(
				b.effective_one_access_capacity() + b.effective_main_fast_capacity() + reserved,
				FAST,
				"the two effective budgets plus the reservation must re-sum to \
				 fast_capacity at {k} tracked keys",
			);

			if reserved % 2 == 1 {
				assert_eq!(
					main_share,
					one_share + 1,
					"the odd byte must land on the MAIN share at {k} tracked keys",
				);
				inexact += 1;
			}
		}

		assert!(inexact > 100, "the fixture never really exercised an inexact split ({inexact} of 280)");
		assert!(!mb.is_empty(), "the tightening main-fast budget should have demoted something");
		assert_eq!(ma, mb, "migrations diverge under the proportional split");

		for k in 1..=280u64 {
			assert_eq!(a.tier_of(k), b.tier_of(k), "key {k}");
			assert_eq!(a.is_midpoint(k), b.is_midpoint(k), "key {k}");
		}

		assert_eq!(a.fast_bytes_used(), b.fast_bytes_used());
		assert_eq!(a.slow_bytes_used(), b.slow_bytes_used());
		assert_eq!(a.fast_object_count(), b.fast_object_count());
		assert_eq!(a.slow_object_count(), b.slow_object_count());
		assert_eq!(a.dram_reserved_bytes(), b.dram_reserved_bytes());
	}

	/// `resize_fast_tier` settles the ONE-ACCESS queue FIRST and the main fast
	/// segment SECOND. Both settles append to the same migration log and both
	/// drive the midpoint cursor, so the order is observable -- but only when
	/// both actually have work to do, which shrinking `fast_capacity` is the
	/// only event to arrange: it tightens both fast segments at once, the
	/// one-access one through its re-proportioned share of the reservation.
	///
	/// `resize_fast_tier_alone_settles_the_one_access_queue` above cannot see
	/// this: its 300 keys are never re-accessed, so main_fast is empty and
	/// `settle_fast_tier` is a no-op -- only one of the two calls ever emits.
	#[test]
	fn resize_fast_tier_settles_the_one_access_queue_before_the_main_segment() {
		// ratio 0.01 of 1_000_000 -> one_access_capacity 10_000.
		let mut a = Baseline::new(0.01, MAX, 1_000_000).with_shared_overhead(112);
		let mut b = Compact::new(0.01, MAX, 1_000_000).with_shared_overhead(112);

		// 300 hot keys, promoted straight out of the one-access queue into
		// main_fast...
		for k in 1..=300u64 {
			a.insert(k, 100); a.update(k);
			b.insert(k, 100); b.update(k);
		}

		// ...and 50 cold ones left sitting in the one-access queue.
		for k in 1_001..=1_050u64 {
			a.insert(k, 100);
			b.insert(k, 100);
		}

		a.drain_tier_migrations();
		b.drain_tier_migrations();

		assert_eq!(b.queues.queue_len(Q_MAIN_FAST), 300, "the hot set must be in main_fast");
		assert_eq!(b.queues.queue_len(Q_ONE_ACCESS), 50, "the cold set must still be in the one-access queue");
		assert_eq!(b.slow_object_count(), 0, "nothing should have settled yet");

		a.resize_fast_tier(60_000);
		b.resize_fast_tier(60_000);

		let ma = a.drain_tier_migrations();
		let mb = b.drain_tier_migrations();

		// Both segments must have emitted, or the ordering assertion below is
		// vacuous -- which is exactly the state the existing test is in.
		let first_hot = mb
			.iter()
			.position(|(k, _)| *k <= 300)
			.expect("the main fast segment must have demoted");
		let last_cold = mb
			.iter()
			.rposition(|(k, _)| *k > 1_000)
			.expect("the one-access queue must have been reprieved");

		assert!(
			last_cold < first_hot,
			"every one-access reprieve must be logged before the first main-fast demotion \
			 (last cold at {last_cold}, first hot at {first_hot})",
		);

		assert_eq!(ma, mb, "resize_fast_tier migrations diverge");

		for k in (1..=300u64).chain(1_001..=1_050u64) {
			assert_eq!(a.tier_of(k), b.tier_of(k), "key {k}");
			assert_eq!(a.is_midpoint(k), b.is_midpoint(k), "key {k}");
		}

		assert_eq!(a.fast_bytes_used(), b.fast_bytes_used());
		assert_eq!(a.slow_bytes_used(), b.slow_bytes_used());
		assert_eq!(a.fast_object_count(), b.fast_object_count());
		assert_eq!(a.slow_object_count(), b.slow_object_count());
	}

	/// The split's numerator is `reserved_overhead() * one_access_capacity`,
	/// and both factors are FAST-tier byte counts -- a CXL-sized fast tier with
	/// a per-object metadata reservation pushes their product past u64 long
	/// before either factor gets anywhere near it. The `u128` intermediate is
	/// load-bearing, not decorative.
	#[test]
	fn the_reservation_split_survives_a_product_that_overflows_u64() {
		// one_access_capacity 2^33 out of a fast_capacity of 2^34, at 2^30 of
		// shared overhead per tracked key. At TWO tracked keys
		// `reserved_overhead()` is 2^31 and the numerator is exactly 2^64 --
		// one step past what u64 can hold.
		const FAST: CacheSize = 1 << 34;
		const ONE_ACCESS: CacheSize = 1 << 33;
		const OVERHEAD: CacheSize = 1 << 30;
		const SIZE: ObjectSize = 4_000_000_000;

		let mut a = Baseline::new(0.5, FAST, FAST).with_shared_overhead(OVERHEAD);
		let mut b = Compact::new(0.5, FAST, FAST).with_shared_overhead(OVERHEAD);

		a.insert(1, SIZE);
		b.insert(1, SIZE);
		a.drain_tier_migrations();
		b.drain_tier_migrations();

		assert_eq!(b.tier_of(1), Some(Tier::Fast), "one key alone fits the one-access budget");
		assert_eq!(a.tier_of(1), b.tier_of(1));

		a.insert(2, SIZE);
		b.insert(2, SIZE);

		// Widened: 2^31 * 2^33 / 2^34 == 2^30, leaving the queue an effective
		// budget of 2^33 - 2^30 == 7_516_192_768 -- under the 8_000_000_000
		// bytes now sitting in it, so its tail is reprieved into main_slow.
		// Taken in u64 the product wraps to zero (or traps outright under
		// debug overflow checks), the share is 0, the queue keeps its whole
		// 2^33, and nothing moves.
		assert_eq!(b.reserved_overhead(), 1 << 31);
		assert_eq!(b.reserved_shares(), (1 << 30, 1 << 30));
		assert_eq!(b.effective_one_access_capacity(), ONE_ACCESS - (1 << 30));

		let ma = a.drain_tier_migrations();
		let mb = b.drain_tier_migrations();

		assert_eq!(mb, vec![(1, Tier::Slow)], "the one-access tail must be reprieved");
		assert_eq!(ma, mb);
		assert_eq!(b.tier_of(1), Some(Tier::Slow));
		assert_eq!(b.tier_of(2), Some(Tier::Fast));
		assert_eq!(a.tier_of(1), b.tier_of(1));
		assert_eq!(a.tier_of(2), b.tier_of(2));
		assert!(b.contains(1), "reprieved, not evicted");

		assert_eq!(b.slow_object_count(), 1);
		assert_eq!(b.fast_object_count(), 1);
		assert_eq!(a.slow_object_count(), b.slow_object_count());
		assert_eq!(a.fast_object_count(), b.fast_object_count());
		assert_eq!(a.fast_bytes_used(), b.fast_bytes_used());
		assert_eq!(a.slow_bytes_used(), b.slow_bytes_used());
	}
}
