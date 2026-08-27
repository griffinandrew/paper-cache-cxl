/*
 * Copyright (c) Kia Shakiba
 *
 * This source code is licensed under the GNU AGPLv3 license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! Slab-backed full 2Q with fast admission: behaviourally identical to
//! [`TwoQFullFastAdmissionHybridStack`], with one structure where that has
//! four.
//!
//! `TwoQFullFastAdmissionHybridStack` keeps THREE `kwik::HashList`s -- `a1_in`,
//! `a1_out` and `am`, each owning its OWN key-to-node index -- plus a separate
//! `entries` map holding the combined payload. Four indexes, for a population
//! where every key is in exactly one of the three queues.
//!
//! Here a single [`CompactQueueSet`] holds all three orders over one slab, with
//! the payload in the index value. The two transitions this design is built
//! around -- `a1_in -> a1_out` (a DRAM->PMEM demotion) and `a1_out -> am` (a
//! PMEM->DRAM promotion) -- become an unlink and a relink of the SAME slot
//! rather than a hash-indexed removal from one list and an insertion into
//! another. That is the hottest structural path in the policy, so it is the one
//! worth compacting.
//!
//! The queue algorithm is unchanged and deliberately so; see
//! [`TwoQFullFastAdmissionHybridStack`]'s module doc for the design argument.
//! Restated only far enough to read this file:
//!
//! | queue | role | tier |
//! |---|---|---|
//! | `a1_in` | probation FIFO for brand-new keys, capped at `k_in * max_size` | **FAST**, structurally |
//! | `a1_out` | overflow FIFO of keys aged out of `a1_in`, capped at `k_out * max_size` | **SLOW**, structurally |
//! | `am` | main LRU of proven keys | tier-**segmented** at `am_boundary` |
//!
//! * `a1_out` holds REAL RESIDENT OBJECTS, not ghosts: it counts toward
//!   `len()`/`contains()`, its bytes count toward `slow_bytes_used()`, and a hit
//!   there is a genuine PMEM->DRAM promotion.
//! * An `a1_in` hit is a COMPLETE no-op -- no list move, no tier change, no
//!   migration, no counter. This is the inversion from `TwoQCompactHybridStack`,
//!   where the same event is *the* promotion trigger.
//! * `a1_in` overflow DEMOTES into `a1_out` ([`Self::settle_a1_in`]);
//!   `a1_out` overflow is what [`Self::needs_capacity_eviction`] reports. A
//!   `PolicyStack` never self-evicts.
//! * Eviction order is `a1_out` tail, then `a1_in` tail, then `am`'s LRU tail.
//! * `a1_in`'s FIXED capacity (never its live usage) is carved out of the DRAM
//!   budget alongside the shared per-object metadata:
//!   `effective_am_fast_capacity() = fast_capacity - a1_in_capacity -
//!   reserved_overhead()`. So `resize` must re-settle BOTH invariants, and
//!   admission must re-settle NEITHER.
//!
//! `reserved_overhead()` stays a single term, exactly as in the stack this
//! replaces: an `a1_out`-resident key is a tracked key like any other, so its
//! index bucket and its slab slot are already counted by `queues.len()`.

use crate::{
	object::ObjectSize,
	worker::policy::policy_stack::{
		compact_queue_set::CompactQueueSet, narrow_resident, watermarks, CacheSize, HashedKey,
		PolicyStack, Tier,
	},
	PaperPolicy,
};

/// Queue slots in the shared set. Probation FIFO is 0, overflow FIFO is 1, the
/// main LRU is 2; a key is in exactly one of the three.
const Q_A1_IN: usize = 0;
const Q_A1_OUT: usize = 1;
const Q_AM: usize = 2;

/// Which of the three live queues a key currently belongs to.
///
/// The tag doubles as the tier for two of the three: `A1In` is Fast and
/// `A1Out` is Slow *structurally*, so neither stores a tier. Only `Am` is
/// segmented and therefore carries one.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Queue {
	A1In,
	A1Out,
	Am,
}

/// Combined per-key bookkeeping, carried in the index value.
///
/// Invariant: `tier.is_some()` iff `queue == Queue::Am`. A key is resident in
/// exactly one of the three queues, which is what keeps the four byte counters
/// and two object counters honest.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct TwoQFullFaPayload {
	queue: Queue,
	tier: Option<Tier>,
	/// The part of `size` that stays in DRAM in either tier; see `migrating`.
	dram_resident: u8,
	size: ObjectSize,
}

/// Pinned, exactly as `TwoQEntry` is in the stack this replaces. The payload
/// rides in the index bucket, so growth here costs bytes on every tracked key.
/// A third `Queue` variant is free: the tag was already a byte.
const _: () = assert!(
	std::mem::size_of::<TwoQFullFaPayload>() == 8,
	"TwoQFullFaPayload grew past 8 bytes",
);

impl TwoQFullFaPayload {
	/// Bytes that actually move between tiers.
	fn migrating(&self) -> CacheSize {
		(self.size as CacheSize).saturating_sub(self.dram_resident as CacheSize)
	}
}

pub struct TwoQFullFastAdmissionCompactHybridStack {
	queues: CompactQueueSet<TwoQFullFaPayload>,

	k_in: f64,
	k_out: f64,

	/// `k_in * max_size`. A reservation carved out of `fast_capacity` (these
	/// bytes are DRAM) -- see [`Self::effective_am_fast_capacity`].
	a1_in_capacity: CacheSize,
	a1_in_used: CacheSize,

	/// `k_out * max_size`. A PMEM budget, carved out of nothing; overrunning it
	/// is what [`Self::needs_capacity_eviction`] reports.
	a1_out_capacity: CacheSize,
	a1_out_used: CacheSize,

	/// Total fast-tier (DRAM) budget, covering BOTH `a1_in` and `am`'s fast
	/// segment.
	fast_capacity: CacheSize,

	shared_overhead: CacheSize,

	/// Bytes held by `am` keys tagged `Tier::Fast`. Does NOT include
	/// `a1_in_used`; [`Self::fast_bytes_used`] sums them.
	am_fast_used: CacheSize,

	/// Bytes held by `am` keys tagged `Tier::Slow`. Does NOT include
	/// `a1_out_used`; [`Self::slow_bytes_used`] sums them.
	am_slow_used: CacheSize,

	am_count: usize,
	am_fast_count: usize,

	/// The least-recently-used FAST key in the main queue.
	am_boundary: Option<HashedKey>,

	migrations: Vec<(HashedKey, Tier)>,
}

impl TwoQFullFastAdmissionCompactHybridStack {
	pub fn new(
		k_in: f64,
		k_out: f64,
		max_size: CacheSize,
		fast_capacity: CacheSize,
	) -> Self {
		TwoQFullFastAdmissionCompactHybridStack {
			queues: CompactQueueSet::default(),
			k_in,
			k_out,
			a1_in_capacity: (k_in * max_size as f64) as CacheSize,
			a1_in_used: 0,
			a1_out_capacity: (k_out * max_size as f64) as CacheSize,
			a1_out_used: 0,
			fast_capacity,
			shared_overhead: 0,
			am_fast_used: 0,
			am_slow_used: 0,
			am_count: 0,
			am_fast_count: 0,
			am_boundary: None,
			migrations: Vec::new(),
		}
	}

	pub fn with_shared_overhead(mut self, overhead: CacheSize) -> Self {
		self.shared_overhead = overhead;


		self
	}

	/// One term, not two: every tracked key of every queue and tier holds an
	/// index bucket and a slab slot, and `queues.len()` counts all three
	/// queues -- including `a1_out`, whose members are ordinary resident keys.
	fn reserved_overhead(&self) -> CacheSize {
		self.queues.len() as CacheSize * self.shared_overhead
	}

	/// How much of `fast_capacity` `am`'s fast segment may use, after `a1_in`'s
	/// FIXED reservation and the shared per-object metadata reservation are
	/// both carved out.
	///
	/// Saturating rather than panicking when the two carve-outs meet or exceed
	/// `fast_capacity`: that is a legitimate (if degenerate) configuration and
	/// means "`am` gets no fast segment", not an error.
	///
	/// `a1_out_capacity` is deliberately absent: those bytes are PMEM.
	fn effective_am_fast_capacity(&self) -> CacheSize {
		self.fast_capacity
			.saturating_sub(self.a1_in_capacity)
			.saturating_sub(self.reserved_overhead())
	}

	/// Two of the three queues answer structurally: `a1_in` is DRAM by the
	/// fast-admission rule, `a1_out` is PMEM by the demotion rule. Only `am`
	/// stores a tier.
	pub fn tier_of(&self, key: HashedKey) -> Option<Tier> {
		let payload = self.queues.payload(key)?;

		match payload.queue {
			Queue::A1In => Some(Tier::Fast),
			Queue::A1Out => Some(Tier::Slow),
			Queue::Am => payload.tier,
		}
	}

	/// Records a size change without altering the key's queue or tier.
	///
	/// No re-settle afterwards for an `a1_in`-resident key that grew: the
	/// reservation `am`'s budget is computed against is the *fixed*
	/// `a1_in_capacity`, so live `a1_in_used` does not move it.
	fn resize_key(&mut self, key: HashedKey, new_size: ObjectSize, new_resident: u8) {
		let Some(payload) = self.queues.payload_mut(key) else { return };

		let old_migrating = payload.migrating();
		payload.size = new_size;
		payload.dram_resident = new_resident;
		let delta = payload.migrating() as i64 - old_migrating as i64;
		let (queue, tier) = (payload.queue, payload.tier);

		match (queue, tier) {
			(Queue::A1In, _) => {
				self.a1_in_used = (self.a1_in_used as i64 + delta).max(0) as CacheSize;
			},

			(Queue::A1Out, _) => {
				self.a1_out_used = (self.a1_out_used as i64 + delta).max(0) as CacheSize;
			},

			(Queue::Am, Some(Tier::Fast)) => {
				self.am_fast_used = (self.am_fast_used as i64 + delta).max(0) as CacheSize;
			},

			(Queue::Am, Some(Tier::Slow)) => {
				self.am_slow_used = (self.am_slow_used as i64 + delta).max(0) as CacheSize;
			},

			(Queue::Am, None) => {},
		}
	}

	/// Treats an already-tracked key as accessed, dispatching on its queue.
	///
	/// The `A1In` arm is the fidelity point of the whole design and is
	/// deliberately empty: a hit on a probation key does NOTHING. Faithful to
	/// `TwoQStack`, where `a1_out.remove` misses and `am.move_front` is a
	/// silent no-op; and the key is already Fast, so there is nothing to
	/// migrate either.
	fn touch(&mut self, key: HashedKey) {
		match self.queues.payload(key).map(|p| p.queue) {
			Some(Queue::A1In) => {},
			Some(Queue::A1Out) => self.promote_from_a1_out(key),
			Some(Queue::Am) => self.touch_am(key),
			None => {},
		}
	}

	/// `TwoQStack::restructure_to_fit`, with the transition it performs being a
	/// real DRAM->PMEM tier migration.
	///
	/// Drains the `a1_in` tail into `a1_out`'s head until `incoming_size` fits.
	/// A **demotion**, never an eviction: the key keeps its slot, keeps its
	/// payload, and stays visible to `contains()`. The slot does not move --
	/// this is an unlink and a relink, where the stack this replaces removed
	/// the key from one hash-indexed list and inserted it into another.
	///
	/// The `else break` mirrors `restructure_to_fit`'s `else return`: an object
	/// larger than the whole of `a1_in` empties the queue and is then admitted
	/// anyway rather than looping forever.
	fn settle_a1_in(&mut self, incoming_size: ObjectSize) {
		let incoming = incoming_size as CacheSize;

		while self.a1_in_used + incoming > self.a1_in_capacity {
			let Some(key) = self.queues.back(Q_A1_IN) else { break };
			let size = self.queues.payload(key).map(|p| p.migrating()).unwrap_or(0);

			self.queues.move_to_front_of(Q_A1_IN, Q_A1_OUT, key);

			if let Some(p) = self.queues.payload_mut(key) {
				p.queue = Queue::A1Out;
				p.tier = None;
			}

			self.a1_in_used = self.a1_in_used.saturating_sub(size);
			self.a1_out_used += size;

			self.migrations.push((key, Tier::Slow));
		}
	}

	/// The 2Q promotion: an `a1_out` hit moves the live key to `am`'s MRU end
	/// at `Tier::Fast`.
	///
	/// Emits a genuine `(key, Tier::Fast)` migration -- unlike
	/// `TwoQFastAdmissionCompactHybridStack`'s promotion, which is a Fast->Fast
	/// bookkeeping move. Here the bytes really do live in PMEM beforehand,
	/// because `a1_out` is the slow tier.
	fn promote_from_a1_out(&mut self, key: HashedKey) {
		let Some(payload) = self.queues.payload(key) else { return };
		let size_bytes = payload.migrating();

		self.queues.move_to_front_of(Q_A1_OUT, Q_AM, key);
		self.a1_out_used = self.a1_out_used.saturating_sub(size_bytes);

		if let Some(p) = self.queues.payload_mut(key) {
			p.queue = Queue::Am;
			p.tier = Some(Tier::Fast);
		}

		self.am_fast_used += size_bytes;
		self.am_fast_count += 1;
		self.am_count += 1;

		if self.am_boundary.is_none() {
			self.am_boundary = Some(key);
		}

		self.settle_fast_tier();

		// Pushed *after* `settle_fast_tier`, so any demotion this promotion
		// itself triggered is applied (and its DRAM freed) first. Guarded on
		// the key still being Fast: a tight budget can demote it straight back
		// out within that same call, in which case the correct final
		// `(key, Tier::Slow)` has already been pushed.
		if self.queues.payload(key).and_then(|p| p.tier) == Some(Tier::Fast) {
			self.migrations.push((key, Tier::Fast));
		}
	}

	/// Faithful port of `TwoQFullFastAdmissionHybridStack::touch_am`: the LRU
	/// reorder composed additively with the tier promotion.
	fn touch_am(&mut self, key: HashedKey) {
		let previous_tier = self.queues.payload(key).and_then(|p| p.tier);

		let already_at_front = self.queues.front(Q_AM) == Some(key);
		let is_boundary = self.am_boundary == Some(key);

		// Read the neighbour BEFORE moving: once the key is at the front its
		// predecessor is gone, and the boundary must step back to whatever was
		// in front of it.
		let new_boundary_if_moved = if is_boundary && !already_at_front {
			self.queues.before(key)
		} else {
			None
		};

		self.queues.move_front(Q_AM, key);

		if is_boundary && !already_at_front {
			self.am_boundary = new_boundary_if_moved;
		}

		let mut promoted = false;

		if previous_tier != Some(Tier::Fast) {
			if previous_tier == Some(Tier::Slow) {
				let size = self.queues.payload(key).map(|p| p.migrating()).unwrap_or(0);
				self.am_slow_used = self.am_slow_used.saturating_sub(size);
				self.am_fast_used += size;
				self.am_fast_count += 1;
				promoted = true;
			}

			if let Some(p) = self.queues.payload_mut(key) {
				p.tier = Some(Tier::Fast);
			}

			if self.am_boundary.is_none() {
				self.am_boundary = Some(key);
			}
		}

		self.settle_fast_tier();

		// Same ordering and same guard as `promote_from_a1_out` above.
		if promoted && self.queues.payload(key).and_then(|p| p.tier) == Some(Tier::Fast) {
			self.migrations.push((key, Tier::Fast));
		}
	}

	/// Demotes from the tier boundary until `am_fast_used` is back under the
	/// low watermark. The victim is always `am_boundary`, so nothing is
	/// searched.
	///
	/// The ceiling is [`Self::effective_am_fast_capacity`] -- reservations
	/// first, watermarks on the remainder.
	fn settle_fast_tier(&mut self) {
		let effective = self.effective_am_fast_capacity();

		// Trigger only once usage is past the high watermark...
		if self.am_fast_used <= watermarks::high_bytes(effective) {
			return;
		}

		// ...but once triggered, drain all the way down to the low one.
		let drain_target = watermarks::low_bytes(effective);

		while self.am_fast_used > drain_target {
			let Some(demote_key) = self.am_boundary else { break };
			let size = self.queues.payload(demote_key).map(|p| p.migrating()).unwrap_or(0);
			let new_boundary = self.queues.before(demote_key);

			if let Some(p) = self.queues.payload_mut(demote_key) {
				p.tier = Some(Tier::Slow);
			}

			self.am_fast_used = self.am_fast_used.saturating_sub(size);
			self.am_fast_count = self.am_fast_count.saturating_sub(1);
			self.am_slow_used += size;
			self.am_boundary = new_boundary;

			self.migrations.push((demote_key, Tier::Slow));
		}
	}

	/// The FIRST eviction victim, per `TwoQStack::evict_one`.
	fn evict_a1_out_tail(&mut self) -> Option<HashedKey> {
		let (key, payload) = self.queues.pop_back(Q_A1_OUT)?;
		self.a1_out_used = self.a1_out_used.saturating_sub(payload.migrating());
		Some(key)
	}

	/// Reached only once `a1_out` is empty -- under normal operation `a1_in`'s
	/// tail is *demoted* into `a1_out` by [`Self::settle_a1_in`] long before it
	/// can be evicted here.
	fn evict_a1_in_tail(&mut self) -> Option<HashedKey> {
		let (key, payload) = self.queues.pop_back(Q_A1_IN)?;
		self.a1_in_used = self.a1_in_used.saturating_sub(payload.migrating());
		Some(key)
	}

	/// The last resort, per `TwoQStack::evict_one`.
	fn evict_am_tail(&mut self) -> Option<HashedKey> {
		let (key, payload) = self.queues.pop_back(Q_AM)?;
		let size = payload.migrating();

		self.am_count = self.am_count.saturating_sub(1);

		match payload.tier {
			Some(Tier::Fast) => {
				self.am_fast_used = self.am_fast_used.saturating_sub(size);
				self.am_fast_count = self.am_fast_count.saturating_sub(1);

				// The tail of `am` can only be Fast-tagged if every tracked
				// `am` key is still Fast (fast keys are a contiguous prefix),
				// in which case the boundary equalled this key. Re-point it at
				// the new tail, unless that tail is Slow or `am` is now empty.
				if self.am_boundary == Some(key) {
					self.am_boundary = match self.queues.back(Q_AM) {
						Some(back)
							if self.queues.payload(back).and_then(|p| p.tier)
								== Some(Tier::Fast) => Some(back),
						_ => None,
					};
				}
			},

			Some(Tier::Slow) => {
				self.am_slow_used = self.am_slow_used.saturating_sub(size);
			},

			None => {},
		}

		Some(key)
	}
}

impl PolicyStack for TwoQFullFastAdmissionCompactHybridStack {
	fn is_policy(&self, policy: &PaperPolicy) -> bool {
		matches!(
			policy,
			PaperPolicy::TwoQFullFastAdmissionCompactHybrid(k_in, k_out)
				if *k_in == self.k_in && *k_out == self.k_out
		)
	}

	fn len(&self) -> usize {
		self.queues.len()
	}

	/// `a1_out` members count: they are resident objects, not ghosts.
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

		// Brand-new key: `a1_in` first, which is FAST here.
		self.settle_a1_in(size);

		self.queues.push_front(
			Q_A1_IN,
			key,
			TwoQFullFaPayload { queue: Queue::A1In, tier: None, dram_resident, size },
		);
		self.a1_in_used += (size as CacheSize).saturating_sub(dram_resident as CacheSize);

		// Deliberately does NOT re-settle the fast tier: the reservation carved
		// out of `fast_capacity` is the fixed `a1_in_capacity`, not live
		// `a1_in_used`, so admission cannot move `am`'s budget. And it
		// deliberately does not evict: `settle_a1_in` demoted instead, and any
		// resulting `a1_out` overrun is reported via `needs_capacity_eviction`.
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
			Queue::A1In => {
				self.queues.remove(Q_A1_IN, key);
				self.a1_in_used = self.a1_in_used.saturating_sub(size);
			},

			Queue::A1Out => {
				self.queues.remove(Q_A1_OUT, key);
				self.a1_out_used = self.a1_out_used.saturating_sub(size);
			},

			Queue::Am => {
				let new_boundary_if_needed =
					if payload.tier == Some(Tier::Fast) && self.am_boundary == Some(key) {
						self.queues.before(key)
					} else {
						None
					};

				self.queues.remove(Q_AM, key);
				self.am_count = self.am_count.saturating_sub(1);

				match payload.tier {
					Some(Tier::Fast) => {
						self.am_fast_used = self.am_fast_used.saturating_sub(size);
						self.am_fast_count = self.am_fast_count.saturating_sub(1);

						if self.am_boundary == Some(key) {
							self.am_boundary = new_boundary_if_needed;
						}
					},

					Some(Tier::Slow) => {
						self.am_slow_used = self.am_slow_used.saturating_sub(size);
					},

					None => {},
				}
			},
		}
	}

	/// Rescales BOTH budgets and re-establishes BOTH invariants eagerly.
	///
	/// `a1_in_capacity` feeds [`Self::effective_am_fast_capacity`], so a stale
	/// one distorts `am`'s DRAM budget until some unrelated access happens to
	/// notice. `TwoQCompactHybridStack::resize` need not re-settle at all,
	/// because its FIFO queue is PMEM and competes for nothing.
	fn resize(&mut self, max_size: CacheSize) {
		self.a1_in_capacity = (self.k_in * max_size as f64) as CacheSize;
		self.a1_out_capacity = (self.k_out * max_size as f64) as CacheSize;

		// Drain `a1_in` down to its new budget (demoting, never evicting)...
		self.settle_a1_in(0);

		// ...then re-settle `am`, whose effective fast budget just moved with
		// `a1_in_capacity`.
		self.settle_fast_tier();
	}

	fn clear(&mut self) {
		self.queues.clear();

		self.a1_in_used = 0;
		self.a1_out_used = 0;
		self.am_fast_used = 0;
		self.am_slow_used = 0;
		self.am_count = 0;
		self.am_fast_count = 0;
		self.am_boundary = None;
		self.migrations.clear();

		// Capacities are configuration, not state: kept.
	}

	/// `TwoQStack::evict_one`, verbatim: `a1_out` tail, then `a1_in` tail, then
	/// `am`'s LRU tail. Emits no migrations -- an evicted object is gone, not
	/// moved.
	fn evict_one(&mut self) -> Option<HashedKey> {
		if let Some(key) = self.evict_a1_out_tail() {
			return Some(key);
		}

		if let Some(key) = self.evict_a1_in_tail() {
			return Some(key);
		}

		self.evict_am_tail()
	}

	fn resize_fast_tier(&mut self, size: CacheSize) {
		self.fast_capacity = size;

		// `fast_capacity` only arrives here, so this is the earliest point at
		// which the sizing constraint can be checked at all.
		if size > 0 && self.effective_am_fast_capacity() == 0 {
			log::warn!(
				"2q-full-fast-admission-compact-hybrid: a1_in's DRAM reservation ({} bytes) plus per-key metadata ({} bytes) meets or exceeds the fast-tier budget ({size} bytes); `am` gets no fast segment and every promotion will demote straight back out. Lower k_in or raise fast_tier_size.",
				self.a1_in_capacity,
				self.reserved_overhead(),
			);
		}

		self.settle_fast_tier();
	}

	fn drain_tier_migrations(&mut self) -> Vec<(HashedKey, Tier)> {
		std::mem::take(&mut self.migrations)
	}

	fn dram_reserved_bytes(&self) -> CacheSize {
		self.reserved_overhead()
	}

	/// Both DRAM-resident structures, summed: the probation FIFO plus `am`'s
	/// fast segment. `a1_out` is PMEM and is excluded.
	fn fast_bytes_used(&self) -> CacheSize {
		self.a1_in_used + self.am_fast_used
	}

	/// `a1_out` plus `am`'s slow segment. `a1_out` counts here because it holds
	/// real resident objects.
	fn slow_bytes_used(&self) -> CacheSize {
		self.a1_out_used + self.am_slow_used
	}

	fn fast_object_count(&self) -> usize {
		self.queues.queue_len(Q_A1_IN) + self.am_fast_count
	}

	fn slow_object_count(&self) -> usize {
		self.queues.queue_len(Q_A1_OUT) + (self.am_count - self.am_fast_count)
	}

	/// `a1_out` ONLY. `a1_in` overflow is a demotion handled internally by
	/// [`Self::settle_a1_in`]; reporting it here would evict where the
	/// algorithm demotes.
	fn needs_capacity_eviction(&self) -> bool {
		self.a1_out_used > self.a1_out_capacity
	}
}

/// Fidelity against `TwoQFullFastAdmissionHybridStack`, which this stack is a
/// compaction of.
///
/// The two must be indistinguishable: same queue for every key, same tier, same
/// migration sequence in the same order, same eviction order. Agreeing on a
/// miss ratio is necessary but not sufficient -- it would not catch a counter
/// firing on the wrong path, which is the class of defect that produced a
/// doubled demotion count on the LFU conversion.
#[cfg(all(test, feature = "two_q_full_fast_admission_hybrid_cache"))]
mod fidelity_tests {
	use super::*;
	use crate::worker::policy::policy_stack::two_q_full_fast_admission_hybrid_stack::TwoQFullFastAdmissionHybridStack;

	const MAX: CacheSize = 1_000_000;

	/// 200 keys biased toward low ids. Repeats matter especially here: a key is
	/// only promoted once it has aged out of `a1_in` into `a1_out` AND is then
	/// re-accessed, so a workload without reuse would never exercise `am` at
	/// all.
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

	fn replay(
		k_in: f64,
		k_out: f64,
		fast: CacheSize,
		overhead: CacheSize,
		ops: &[(HashedKey, ObjectSize)],
	) -> (Vec<(HashedKey, Tier)>, Vec<(HashedKey, Tier)>, Vec<Option<Tier>>, Vec<Option<Tier>>) {
		let mut a = TwoQFullFastAdmissionHybridStack::new(k_in, k_out, MAX, fast)
			.with_shared_overhead(overhead);
		let mut b = TwoQFullFastAdmissionCompactHybridStack::new(k_in, k_out, MAX, fast)
			.with_shared_overhead(overhead);
		let (mut ma, mut mb) = (Vec::new(), Vec::new());

		for (k, size) in ops {
			if a.contains(*k) { a.update(*k); } else { a.insert(*k, *size); }
			if b.contains(*k) { b.update(*k); } else { b.insert(*k, *size); }
			ma.extend(a.drain_tier_migrations());
			mb.extend(b.drain_tier_migrations());
		}

		// `k_out` reaches nothing else in this replay -- it sizes `a1_out`, and
		// that budget is only ever READ here.
		assert_eq!(
			a.needs_capacity_eviction(),
			b.needs_capacity_eviction(),
			"a1_out capacity pressure diverges at k_in {k_in} k_out {k_out} fast {fast} overhead {overhead}",
		);

		let keys: Vec<HashedKey> = ops.iter().map(|(k, _)| *k).collect();
		let ta = keys.iter().map(|k| a.tier_of(*k)).collect();
		let tb = keys.iter().map(|k| b.tier_of(*k)).collect();
		(ma, mb, ta, tb)
	}

	#[test]
	fn matches_two_q_full_fast_admission_hybrid_migration_for_migration() {
		let ops = skewed_ops();
		for k_in in [0.001f64, 0.01, 0.1] {
			for k_out in [0.05f64, 0.25] {
				for fast in [8_192u64, 32_768, 131_072] {
					for overhead in [0u64, 112] {
						let (ma, mb, ta, tb) = replay(k_in, k_out, fast, overhead, &ops);
						// Guards against a vacuous pass: two empty migration
						// vectors compare equal. Every configuration here moves
						// 11k-37k objects between tiers.
						assert!(
							ma.len() > 1_000,
							"only {} migrations at k_in {k_in} k_out {k_out} fast {fast} overhead {overhead} -- the workload stopped exercising the tier paths",
							ma.len(),
						);
						assert_eq!(
							ta, tb,
							"tiers diverge at k_in {k_in} k_out {k_out} fast {fast} overhead {overhead}"
						);
						assert_eq!(
							ma, mb,
							"migrations diverge at k_in {k_in} k_out {k_out} fast {fast} overhead {overhead}"
						);
					}
				}
			}
		}
	}

	/// Eviction order is separate from migration order: `evict_one` drains the
	/// `a1_out` tail first, then `a1_in`'s, and only then `am`'s.
	#[test]
	fn evicts_in_the_same_order() {
		let ops = skewed_ops();
		let mut a = TwoQFullFastAdmissionHybridStack::new(0.01, 0.25, MAX, 32_768)
			.with_shared_overhead(112);
		let mut b = TwoQFullFastAdmissionCompactHybridStack::new(0.01, 0.25, MAX, 32_768)
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
		assert_eq!(ea, eb, "eviction order diverges");
		assert_eq!(b.len(), 0);
	}

	/// Eviction INTERLEAVED with continued access, which the drain-to-empty
	/// test above cannot see.
	///
	/// `evict_am_tail` re-points `am_boundary` under a rule this variant does
	/// NOT share with `TwoQCompactHybridStack`: the plain 2Q stack hands the
	/// boundary the new tail unconditionally, while this one hands it the new
	/// tail ONLY if that tail is still Fast, and `None` otherwise. Nothing
	/// observes that difference until the NEXT `settle_fast_tier`, so a test
	/// that drains to empty and stops passes either way.
	///
	/// `k_out` is small and three keys are evicted every five operations, so
	/// `a1_out` is frequently empty and eviction really does reach `a1_in`'s
	/// and `am`'s tails rather than stopping at the first queue.
	#[test]
	fn eviction_interleaved_with_access_agrees() {
		let ops = skewed_ops();
		let mut a = TwoQFullFastAdmissionHybridStack::new(0.01, 0.05, MAX, 32_768)
			.with_shared_overhead(112);
		let mut b = TwoQFullFastAdmissionCompactHybridStack::new(0.01, 0.05, MAX, 32_768)
			.with_shared_overhead(112);
		let (mut ma, mut mb) = (Vec::new(), Vec::new());
		let (mut ea, mut eb) = (Vec::new(), Vec::new());

		for (i, (k, size)) in ops.iter().enumerate() {
			if a.contains(*k) { a.update(*k); } else { a.insert(*k, *size); }
			if b.contains(*k) { b.update(*k); } else { b.insert(*k, *size); }

			if i % 5 == 0 {
				for _ in 0..3 {
					ea.extend(a.evict_one());
					eb.extend(b.evict_one());
				}
			}

			ma.extend(a.drain_tier_migrations());
			mb.extend(b.drain_tier_migrations());
		}

		assert_eq!(ea, eb, "eviction order diverges under interleaving");
		assert_eq!(ma, mb, "migrations diverge under interleaving");
		assert!(ma.len() > 1_000, "only {} migrations -- paths not exercised", ma.len());
		assert_eq!(a.len(), b.len(), "lengths diverge");
		assert_eq!(a.fast_object_count(), b.fast_object_count(), "fast counts diverge");
		assert_eq!(a.slow_object_count(), b.slow_object_count(), "slow counts diverge");
		assert_eq!(a.fast_bytes_used(), b.fast_bytes_used(), "fast bytes diverge");
		assert_eq!(a.slow_bytes_used(), b.slow_bytes_used(), "slow bytes diverge");

		for k in 1..=200u64 {
			assert_eq!(b.tier_of(k), a.tier_of(k), "tier of {k} diverges");
		}
	}

	/// Removal maintains the `am` tier boundary and must handle all THREE
	/// queues; nothing above removes.
	#[test]
	fn removal_matches_including_boundary_maintenance() {
		let ops = skewed_ops();
		let mut a = TwoQFullFastAdmissionHybridStack::new(0.01, 0.25, MAX, 32_768)
			.with_shared_overhead(112);
		let mut b = TwoQFullFastAdmissionCompactHybridStack::new(0.01, 0.25, MAX, 32_768)
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
		assert_eq!(a.len(), b.len(), "lengths diverge");
		assert_eq!(a.fast_object_count(), b.fast_object_count(), "fast counts diverge");
		assert_eq!(a.slow_object_count(), b.slow_object_count(), "slow counts diverge");
		assert_eq!(a.fast_bytes_used(), b.fast_bytes_used(), "fast bytes diverge");
		assert_eq!(a.slow_bytes_used(), b.slow_bytes_used(), "slow bytes diverge");
		assert_eq!(a.dram_reserved_bytes(), b.dram_reserved_bytes(), "reservations diverge");
	}

	/// Resizing, both directions, with BRAND-NEW keys arriving afterwards.
	///
	/// The shape is load-bearing: on the LFU conversion the equivalent test
	/// passed with a real bug present because the workload drew from a fixed
	/// key set, so by the resize point nothing a resize affects was observable.
	/// `resize` here rescales BOTH `a1_in_capacity` and `a1_out_capacity` and
	/// then re-settles both invariants, which `resize_fast_tier` does not.
	#[test]
	fn resizes_like_two_q_full_fast_admission_hybrid() {
		for (start, resized) in [(65_536u64, 65_536u64), (131_072, 32_768), (32_768, 131_072)] {
			let mut a = TwoQFullFastAdmissionHybridStack::new(0.01, 0.25, MAX, start)
				.with_shared_overhead(112);
			let mut b = TwoQFullFastAdmissionCompactHybridStack::new(0.01, 0.25, MAX, start)
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
			assert_eq!(
				a.needs_capacity_eviction(),
				b.needs_capacity_eviction(),
				"a1_out pressure diverges after resize {start} -> {resized}"
			);

			for i in 0..2_000u64 {
				let k = 10_000 + i;
				a.insert(k, 1024);
				b.insert(k, 1024);
				ma.extend(a.drain_tier_migrations());
				mb.extend(b.drain_tier_migrations());
			}

			assert_eq!(ma, mb, "migrations diverge resizing {start} -> {resized}");
			for i in 0..2_000u64 {
				assert_eq!(
					a.tier_of(10_000 + i),
					b.tier_of(10_000 + i),
					"tier of new key {} diverges",
					10_000 + i
				);
			}
		}
	}

	/// The behaviour unique to THIS variant, and the exact inversion of
	/// `TwoQCompactHybridStack`'s: a first access admits to `a1_in` at
	/// **Fast**, a second access while still in `a1_in` is a COMPLETE no-op,
	/// and promotion to `am` requires the key to have aged into `a1_out` first.
	#[test]
	fn an_a1_in_hit_is_a_complete_no_op() {
		let mut a = TwoQFullFastAdmissionHybridStack::new(0.25, 0.25, MAX, 1_000_000)
			.with_shared_overhead(0);
		let mut b = TwoQFullFastAdmissionCompactHybridStack::new(0.25, 0.25, MAX, 1_000_000)
			.with_shared_overhead(0);

		a.insert(1, 1024);
		b.insert(1, 1024);
		assert_eq!(a.tier_of(1), Some(Tier::Fast));
		assert_eq!(b.tier_of(1), a.tier_of(1));
		assert_eq!(a.drain_tier_migrations(), b.drain_tier_migrations());

		// Second access, still in `a1_in`: nothing at all happens.
		a.update(1);
		b.update(1);
		assert_eq!(b.tier_of(1), a.tier_of(1));
		assert_eq!(a.tier_of(1), Some(Tier::Fast));
		assert!(b.drain_tier_migrations().is_empty(), "an a1_in hit emitted a migration");
		assert!(a.drain_tier_migrations().is_empty());
		assert_eq!(a.fast_object_count(), b.fast_object_count());
		assert_eq!(a.slow_object_count(), b.slow_object_count());
	}

	/// The `a1_in -> a1_out` demotion and the subsequent `a1_out -> am`
	/// promotion, step by step, against the baseline.
	///
	/// `a1_in_capacity` is deliberately tiny so admission of the second key
	/// pushes the first out of probation.
	#[test]
	fn ages_out_of_probation_then_promotes_on_the_next_hit() {
		// k_in * MAX = 2048: room for exactly two 1 KiB objects.
		let mut a = TwoQFullFastAdmissionHybridStack::new(0.002048, 0.5, MAX, 1_000_000)
			.with_shared_overhead(0);
		let mut b = TwoQFullFastAdmissionCompactHybridStack::new(0.002048, 0.5, MAX, 1_000_000)
			.with_shared_overhead(0);
		let (mut ma, mut mb) = (Vec::new(), Vec::new());

		for k in 1..=4u64 {
			a.insert(k, 1024);
			b.insert(k, 1024);
			ma.extend(a.drain_tier_migrations());
			mb.extend(b.drain_tier_migrations());
		}

		assert_eq!(ma, mb, "demotion migrations diverge");
		assert!(!ma.is_empty(), "nothing aged out of a1_in -- test is not exercising the path");
		for k in 1..=4u64 {
			assert_eq!(b.tier_of(k), a.tier_of(k), "tier of {k} diverges after ageing");
		}
		assert_eq!(a.fast_bytes_used(), b.fast_bytes_used());
		assert_eq!(a.slow_bytes_used(), b.slow_bytes_used());

		// Key 1 is now in `a1_out`; a hit there is a real PMEM->DRAM promotion.
		assert_eq!(a.tier_of(1), Some(Tier::Slow));
		a.update(1);
		b.update(1);
		let (pa, pb) = (a.drain_tier_migrations(), b.drain_tier_migrations());
		assert_eq!(pa, pb, "promotion migrations diverge");
		assert_eq!(pa, vec![(1u64, Tier::Fast)], "an a1_out hit must promote to Fast");
		assert_eq!(a.tier_of(1), Some(Tier::Fast));
		assert_eq!(b.tier_of(1), a.tier_of(1));
		assert_eq!(a.fast_object_count(), b.fast_object_count());
		assert_eq!(a.slow_object_count(), b.slow_object_count());
	}

	/// `needs_capacity_eviction` tracks `a1_out` ONLY: an `a1_in` overrun must
	/// not report, because the algorithm demotes there rather than evicting.
	#[test]
	fn capacity_pressure_tracks_a1_out_only() {
		let mut a = TwoQFullFastAdmissionHybridStack::new(0.002048, 0.004096, MAX, 1_000_000)
			.with_shared_overhead(0);
		let mut b = TwoQFullFastAdmissionCompactHybridStack::new(0.002048, 0.004096, MAX, 1_000_000)
			.with_shared_overhead(0);

		let mut seen_pressure = false;
		for k in 1..=32u64 {
			a.insert(k, 1024);
			b.insert(k, 1024);
			a.drain_tier_migrations();
			b.drain_tier_migrations();
			assert_eq!(
				a.needs_capacity_eviction(),
				b.needs_capacity_eviction(),
				"capacity pressure diverges after inserting {k}"
			);
			seen_pressure |= b.needs_capacity_eviction();
		}
		assert!(seen_pressure, "a1_out never overran -- test is not exercising the path");
	}
}
