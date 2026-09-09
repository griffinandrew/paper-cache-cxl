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

/// Queue slots in the shared set. A key is in exactly one of the three.
const Q_ONE_ACCESS: usize = 0;
const Q_MAIN_FAST: usize = 1;
const Q_MAIN_SLOW: usize = 2;

/// Which live queue a key currently belongs to. `Main` covers both physical
/// main lists; `tier` says which one.
///
/// The shared node stores this as a plain `u8`, so the enum is kept purely for
/// readability and converted at that one boundary: `Queue as u8` on the way in,
/// [`Queue::from_u8`] on the way out. Every match below still reads as the enum.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
enum Queue {
	OneAccess = 0,
	Main = 1,
}

impl Queue {
	/// The inverse of `as u8`. `NodePayload::queue` is only ever written here
	/// from a `Queue as u8`, so the catch-all arm is unreachable and `Main` is
	/// the only value it could stand for.
	fn from_u8(tag: u8) -> Queue {
		match tag {
			0 => Queue::OneAccess,
			_ => Queue::Main,
		}
	}
}

/// Per-key bookkeeping is [`NodePayload`], the one node every policy shares.
///
/// This stack reads `queue` (as [`Queue`]), `tier`, `freq`, `size` and
/// `dram_resident`. `freq` carries the S3-FIFO REFERENCE BIT: set is `freq = 1`
/// and tested as `freq != 0`, since a reference bit is a one-bit frequency
/// counter. `tier`/`freq` are only meaningful while `queue == Main`; a
/// one-access key leaves `tier` at `None`, which is one of the reasons the
/// shared node made that field an `Option`. `tier` is redundant with which of
/// the two main lists the key is physically in, but kept because `tier_of()`
/// and the `PolicyWorker` migration path both want it as a cheap single-probe
/// lookup rather than a pair of `contains()` probes.
///
/// `ts` and `phys` belong to other policies; `phys` is set once at admission to
/// match `tier` and never read here.
pub struct S3FifoLazyDemotionFastAdmissionMidpointReprieveCompactHybridStack {
	queues: ArenaQueueSet<NodePayload>,

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
			queues: ArenaQueueSet::default(),
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
	/// this segment's share of the shared-metadata reservation. The settle
	/// drains to this number, never to any part of it alone.
	fn effective_main_fast_capacity(&self) -> CacheSize {
		self.fast_capacity
			.saturating_sub(self.one_access_capacity)
			.saturating_sub(self.reserved_shares().1)
	}

	pub fn tier_of(&self, key: HashedKey) -> Option<Tier> {
		let payload = self.queues.payload(key)?;

		match Queue::from_u8(payload.queue) {
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
		let (queue, tier) = (Queue::from_u8(payload.queue), payload.tier);

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

			// A one-access key really does carry `tier == None` here, but a
			// MAIN-queue key never does: it reaches `Queue::Main` only through
			// `promote_from_one_access` (Fast) or `settle_one_access` (Slow),
			// each of which sets a tier. Unreachable, and spelled out rather
			// than folded into the `_` above.
			(Queue::Main, None) => {},
		}
	}

	fn touch(&mut self, key: HashedKey) {
		match self.queues.payload(key).map(|p| Queue::from_u8(p.queue)) {
			Some(Queue::OneAccess) => self.promote_from_one_access(key),
			Some(Queue::Main) => self.mark_accessed(key),
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

	/// Moves a re-accessed one-access-queue key into `Q_MAIN_FAST`. Emits no
	/// migration for the promotion itself -- the key's bytes are already
	/// physically Fast in this variant.
	fn promote_from_one_access(&mut self, key: HashedKey) {
		let Some(payload) = self.queues.payload(key) else { return };
		let size_bytes = payload.migrating();

		self.queues.move_to_front_of(Q_ONE_ACCESS, Q_MAIN_FAST, key);
		self.one_access_used = self.one_access_used.saturating_sub(size_bytes);

		if let Some(p) = self.queues.payload_mut(key) {
			p.queue = Queue::Main as u8;
			p.tier = Some(Tier::Fast);
			p.freq = 0;
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

		let accessed = self.queues.payload(candidate).map(|p| p.freq != 0).unwrap_or(false);

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
					p.freq = 0;
				}
			},

			Some(Tier::Slow) => {
				self.redirect_midpoint_before_removing(key);
				self.queues.move_to_front_of(Q_MAIN_SLOW, Q_MAIN_FAST, key);

				if let Some(p) = self.queues.payload_mut(key) {
					p.tier = Some(Tier::Fast);
					p.freq = 0;
				}

				self.slow_used = self.slow_used.saturating_sub(size);
				self.fast_used += size;

				self.bump_midpoint_drift();
			},

			// Only a one-access key carries `tier == None`, and this is only
			// ever called on a main-queue key, so it is unreachable. The
			// baseline returns here without settling or pushing a migration,
			// so this does too.
			None => return,
		}

		self.settle_fast_tier();

		if self.queues.payload(key).and_then(|p| p.tier) == Some(Tier::Fast) {
			self.migrations.push((key, Tier::Fast));
		}
	}

	/// Demotes the `Q_MAIN_FAST` tail into the front of `Q_MAIN_SLOW` while
	/// `fast_used` exceeds `effective_main_fast_capacity()` -- reference-bit
	/// gated, so an accessed candidate is reprieved to the front of its own
	/// queue instead.
	fn settle_fast_tier(&mut self) {
		let effective_capacity = self.effective_main_fast_capacity();

		while self.fast_used > effective_capacity {
			let Some(candidate) = self.queues.back(Q_MAIN_FAST) else { break };

			let accessed = self.queues.payload(candidate).map(|p| p.freq != 0).unwrap_or(false);

			if accessed {
				// Reprieve: fresh start at the front instead of demotion.
				self.queues.move_front(Q_MAIN_FAST, candidate);

				if let Some(p) = self.queues.payload_mut(candidate) {
					p.freq = 0;
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
	/// effective budget. `one_access_capacity` is a queue-length rule of the
	/// S3-FIFO design, not a tier-pressure threshold.
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
				p.queue = Queue::Main as u8;
				p.tier = Some(Tier::Slow);
				p.freq = 0;
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
			NodePayload {
				size,
				freq: 0,
				ts: 0,
				queue: Queue::OneAccess as u8,
				tier: None,
				phys: None,
				dram_resident,
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

		match Queue::from_u8(payload.queue) {
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

			let accessed = self.queues.payload(key).map(|p| p.freq != 0).unwrap_or(false);

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
