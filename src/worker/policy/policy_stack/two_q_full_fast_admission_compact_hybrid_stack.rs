/*
 * Copyright (c) Kia Shakiba
 *
 * This source code is licensed under the GNU AGPLv3 license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! Slab-backed full 2Q with fast admission: behaviourally identical to
//! `TwoQFullFastAdmissionHybridStack`, with one structure where that has
//! four.
//!
//! `TwoQFullFastAdmissionHybridStack` keeps THREE `kwik::HashList`s -- `a1_in`,
//! `a1_out` and `am`, each owning its OWN key-to-node index -- plus a separate
//! `entries` map holding the combined payload. Four indexes, for a population
//! where every key is in exactly one of the three queues.
//!
//! Here a single [`ArenaQueueSet`] holds all three orders over one slab, with
//! the payload in the slot itself. The two transitions this design is built
//! around -- `a1_in -> a1_out` (a DRAM->PMEM demotion) and `a1_out -> am` (a
//! PMEM->DRAM promotion) -- become an unlink and a relink of the SAME slot
//! rather than a hash-indexed removal from one list and an insertion into
//! another. That is the hottest structural path in the policy, so it is the one
//! worth compacting.
//!
//! The queue algorithm is unchanged and deliberately so; see
//! `TwoQFullFastAdmissionHybridStack`'s module doc for the design argument.
//! Restated only far enough to read this file:
//!
//! | queue | role | tier |
//! |---|---|---|
//! | `a1_in` | probation FIFO for brand-new keys, capped at `k_in * max_size` clamped to the fast tier | **FAST**, structurally |
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
//!   budget -- clamped to what the tier can pay for (`a1_in_carve_out()`),
//!   since `k_in * max_size` is a fraction of the CACHE -- and the shared
//!   per-object metadata reservation is charged MAIN-FIRST
//!   (`reserved_shares`): `am`'s fast segment pays it out of `fast_capacity
//!   - a1_in_carve_out()`, as it always did, and `a1_in` pays only the part
//!   that does not fit there. `effective_a1_in_capacity() +
//!   effective_am_fast_capacity() + reserved_overhead() == fast_capacity`
//!   while the reservation fits in the tier, and wherever the carve-out and
//!   the reservation fit it together both budgets are exactly the pre-clamp
//!   ones. Deliberately NOT the proportional split of
//!   `TwoQFastAdmissionReprieveCompactHybridStack` and the S3-FIFO
//!   fast-admission stacks; see `reserved_shares`. So `resize` and
//!   `resize_fast_tier` must re-settle BOTH invariants. Admission settles
//!   `a1_in` only, before the push (`restructure_to_fit`), and so does a
//!   re-set that grows an `a1_in` key.
//!
//! `reserved_overhead()` stays a single term, exactly as in the stack this
//! replaces: an `a1_out`-resident key is a tracked key like any other, so its
//! index bucket and its slab slot are already counted by `queues.len()`. It is
//! that one term that `reserved_shares` apportions.
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
		arena_queue_set::{ArenaQueueSet, NodePayload}, narrow_resident, drain_target, CacheSize,
		HashedKey, PolicyStack, Tier, Placement, SetEvent, fast_at_or_before, placed, prev_fast,
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
#[repr(u8)]
enum Queue {
	A1In = 0,
	A1Out = 1,
	Am = 2,
}

impl Queue {
	/// The shared node stores `queue` as a bare `u8`, so this is the one
	/// place the tag becomes an enum again. Every write goes the other way
	/// through `Queue as u8`, which is why the last arm cannot be reached.
	#[inline]
	fn from_u8(tag: u8) -> Queue {
		match tag {
			0 => Queue::A1In,
			1 => Queue::A1Out,
			2 => Queue::Am,
			_ => unreachable!("2q-full-fast-admission-compact-hybrid queue tag out of range: {tag}"),
		}
	}
}

/// Per-key bookkeeping is [`NodePayload`], the one node every policy shares.
/// This stack reads `queue`, `tier`, `size` and `dram_resident`; `freq` and `ts`
/// belong to other policies and stay at their defaults here.
///
/// Invariant: `tier.is_some()` iff `queue == Queue::Am`. A key is resident in
/// exactly one of the three queues, which is what keeps the four byte counters
/// and two object counters honest. `queue` is a bare `u8` in the shared node,
/// so [`Queue`] converts at the boundary.
pub struct TwoQFullFastAdmissionCompactHybridStack {
	queues: ArenaQueueSet<NodePayload>,

	k_in: f64,
	k_out: f64,

	/// `k_in * max_size`. A reservation carved out of `fast_capacity` (these
	/// bytes are DRAM), but only through [`Self::a1_in_carve_out`], which
	/// clamps it to the tier -- see [`Self::effective_am_fast_capacity`].
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

	/// Whether the last check found `a1_in_capacity >= fast_capacity`, i.e.
	/// the one warning for that crossing has been emitted. See
	/// [`Self::warn_if_carve_out_fills_fast_tier`].
	carve_out_fills_fast_tier: bool,

	migrations: Vec<(HashedKey, Tier)>,

	/// S5: the measured M the policy worker pushed (`set_dram_metadata`),
	/// reserved instead of the per-object reservation; `None` under the
	/// per-object model.
	measured: Option<CacheSize>,
}

impl TwoQFullFastAdmissionCompactHybridStack {
	pub fn new(
		k_in: f64,
		k_out: f64,
		max_size: CacheSize,
		fast_capacity: CacheSize,
	) -> Self {
		TwoQFullFastAdmissionCompactHybridStack {
			queues: ArenaQueueSet::default(),
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
			carve_out_fills_fast_tier: false,
			migrations: Vec::new(),
			measured: None,
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
		self.measured.unwrap_or(self.queues.len() as CacheSize * self.shared_overhead)
	}

	/// This stack's eff (S5): the whole fast tier's budget for values -- the
	/// tier's, not a segment's -- before the drain target.
	fn own_eff(&self) -> CacheSize {
		self.fast_capacity.saturating_sub(self.reserved_overhead())
	}

	/// Whether a value of `migrating` bytes is STRUCTURAL (S5): larger than an
	/// empty fast tier. Such a key is placed slow, keeps its place in the
	/// policy's order, and is never promoted while it stays that large. The
	/// stack's own check beside the client's flag, so a key the client placed
	/// normally just before eff moved is placed as the stack's own promotions
	/// would place it.
	fn structural(&self, migrating: CacheSize) -> bool {
		migrating > self.own_eff()
	}

	/// A key the stack places FAST in `a1_in` whose new value is STRUCTURAL
	/// (S5) goes where a structural new key goes: `a1_out`'s front, as
	/// `settle_a1_in` demotes one -- pushed `(key, Slow)`, its placement
	/// changed.
	fn a1_in_to_a1_out(&mut self, key: HashedKey) {
		let Some(payload) = self.queues.payload(key) else { return };
		let size = payload.migrating();

		self.queues.move_to_front_of(Q_A1_IN, Q_A1_OUT, key);

		if let Some(p) = self.queues.payload_mut(key) {
			p.queue = Queue::A1Out as u8;
			p.tier = None;
		}

		self.a1_in_used = self.a1_in_used.saturating_sub(size);
		self.a1_out_used += size;

		self.migrations.push((key, Tier::Slow));
	}

	/// A `Set`, with the client's placement (S5). An existing key: its size
	/// tracked, then an access -- an `a1_in` hit is a no-op (a key that grew
	/// there re-polices `a1_in`), and a STRUCTURAL `a1_in` key moves to
	/// `a1_out`; an `a1_out` or `am` hit as `touch`. A new key: `a1_in`, fast --
	/// or, STRUCTURAL, `a1_out`'s front, slow, where `a1_in`'s overflow goes:
	/// built slow, nothing pushed. Returns the placement applied.
	fn insert_with(&mut self, key: HashedKey, size: ObjectSize, dram_resident: ObjectSize, placement: Placement) -> Placement {
		let dram_resident = narrow_resident(dram_resident);
		let migrating = (size as CacheSize).saturating_sub(dram_resident as CacheSize);
		let structural = placement == Placement::Structural || self.structural(migrating);

		if self.queues.contains(key) {
			let grew_in_a1_in = self.resize_key(key, size, dram_resident);
			let in_a1_in = self.queues.payload(key).map(|p| Queue::from_u8(p.queue)) == Some(Queue::A1In);

			if structural && in_a1_in {
				self.a1_in_to_a1_out(key);
				return Placement::Structural;
			}

			self.touch(key, structural);

			// The hit was a no-op, so a key that grew in `a1_in` is still there
			// and may have pushed `a1_in` past its budget: demote its tail as
			// an admission would, rather than leave the overrun to the next one.
			if grew_in_a1_in {
				self.settle_a1_in(0);
			}

			return placed(structural);
		}

		if structural {
			self.queues.push_front(Q_A1_OUT, key, NodePayload {
				size,
				dram_resident,
				tier: None,
				freq: 0,
				ts: 0,
				queue: Queue::A1Out as u8,
			});
			self.a1_out_used += migrating;

			return Placement::Structural;
		}

		// Brand-new key: `a1_in` first, which is FAST here.
		self.settle_a1_in(size);

		self.queues.push_front(Q_A1_IN, key, NodePayload {
			size,
			dram_resident,
			tier: None,
			freq: 0,
			ts: 0,
			queue: Queue::A1In as u8,
		});
		self.a1_in_used += migrating;

		// Deliberately does NOT re-settle the fast tier: the carve-out taken
		// out of `fast_capacity` is the fixed `a1_in_carve_out()`, not live
		// `a1_in_used`, so admission moves `am`'s budget only by the new key's
		// metadata, which `am` pays first -- as before the clamp. And it
		// deliberately does not evict: `settle_a1_in` demoted instead, and any
		// resulting `a1_out` overrun is reported via `needs_capacity_eviction`.
		Placement::Normal
	}

	/// `a1_in`'s carve-out from the fast tier: its configured capacity, but
	/// never more of the tier than the tier itself holds. The 2Q family's
	/// clamp, as `TwoQFastAdmissionReprieveCompactHybridStack::fifo_carve_out`.
	///
	/// `a1_in_capacity` is `k_in * max_size`, a fraction of the CACHE that
	/// nothing ties to `fast_capacity`. `a1_in` is DRAM, and `settle_a1_in` is
	/// the only thing that bounds it, so without this clamp `a1_in` alone
	/// could hold more DRAM than the whole tier: `effective_am_fast_capacity`
	/// saturates to 0 and the real ceiling becomes `max(fast_capacity, k_in *
	/// max_size)`. Measured before this clamp: `(0.25, 0.5)` held 125 MiB in a
	/// 32 MiB tier.
	///
	/// An accessor rather than a value fixed at either assignment site because
	/// `resize` rewrites `a1_in_capacity` and `resize_fast_tier` rewrites
	/// `fast_capacity`, each without touching the other. A no-op whenever
	/// `a1_in_capacity <= fast_capacity`.
	fn a1_in_carve_out(&self) -> CacheSize {
		self.a1_in_capacity.min(self.fast_capacity)
	}

	/// Splits `reserved_overhead` between the two DRAM segments MAIN-FIRST:
	/// `(a1_in_share, am_share)`.
	///
	/// `am` pays for the reservation out of the part of the tier the carve-out
	/// leaves it, `fast_capacity - a1_in_carve_out()`; `a1_in` pays only what
	/// does not fit there. The shares always re-sum to the reservation. With
	/// `carve = a1_in_carve_out()` and `R = reserved_overhead()`:
	///
	/// - `R <= fast_capacity - carve`: `(0, R)`. `a1_in` drains against its
	///   whole carve-out and `am` settles against `fast_capacity - carve - R`
	///   -- the budgets this stack had before the clamp, bit for bit, whenever
	///   the carve-out itself fits the tier (above it `R` must be 0 here, and
	///   the only change is the clamp).
	/// - `fast_capacity - carve < R <= fast_capacity`: `am`'s budget is 0 and
	///   `a1_in`'s is `fast_capacity - R`.
	/// - `R > fast_capacity`: both budgets are 0. The tier is metadata-bound:
	///   every admission demotes the one before it into `a1_out`, so `a1_in`
	///   holds only the newest key.
	///
	/// DELIBERATELY not the proportional split of
	/// `TwoQFastAdmissionReprieveCompactHybridStack` and the four S3-FIFO
	/// fast-admission stacks. They split in proportion before the clamp
	/// reached them, so for them the clamp changed nothing that fits. This
	/// stack charged the whole reservation to `am`, and a proportional split
	/// would take `R * carve / fast_capacity` of `a1_in`'s budget in every
	/// configuration with a reservation -- every production run -- where only
	/// the clamp was wanted. Main-first moves this stack's budgets only where
	/// the old ones over-subscribed the tier (`a1_in_capacity + R >
	/// fast_capacity`).
	fn reserved_shares(&self) -> (CacheSize, CacheSize) {
		let reserved = self.reserved_overhead();
		let am_share = reserved.min(self.fast_capacity.saturating_sub(self.a1_in_carve_out()));

		(reserved - am_share, am_share)
	}

	/// `a1_in`'s budget net of its share of the metadata reservation: what
	/// [`Self::settle_a1_in`] drains against. With the main-first split that
	/// is `min(a1_in_carve_out(), fast_capacity - reserved_overhead())`,
	/// saturating at 0.
	///
	/// At most `a1_in_carve_out()`, while `am`'s is at most `fast_capacity -
	/// a1_in_carve_out()`, so the two can never sum past the tier. Overflow is
	/// a DEMOTION into `a1_out` here, never an eviction, so the clamp costs
	/// DRAM, not data.
	fn effective_a1_in_capacity(&self) -> CacheSize {
		self.a1_in_carve_out().saturating_sub(self.reserved_shares().0)
	}

	/// How much of `fast_capacity` `am`'s fast segment may use, after `a1_in`'s
	/// FIXED, clamped carve-out and `am`'s share of the shared per-object
	/// metadata reservation are both taken out. `am` pays first, so this is
	/// `fast_capacity - a1_in_carve_out() - reserved_overhead()`, saturating
	/// -- the pre-clamp formula, which it equals for every input because
	/// `fast_capacity - a1_in_carve_out()` and `fast_capacity -
	/// a1_in_capacity` saturate to the same value.
	///
	/// Saturating rather than panicking when the two meet or exceed
	/// `fast_capacity`: that is a legitimate (if degenerate) configuration and
	/// means "`am` gets no fast segment", not an error.
	///
	/// `a1_out_capacity` is deliberately absent: those bytes are PMEM.
	fn effective_am_fast_capacity(&self) -> CacheSize {
		self.fast_capacity
			.saturating_sub(self.a1_in_carve_out())
			.saturating_sub(self.reserved_shares().1)
	}

	/// Prints ONE warning to stderr when the configured `a1_in` (`k_in *
	/// max_size`) is at least the whole fast tier -- the configuration
	/// `a1_in_carve_out()` clamps, in which `a1_in` takes all of the tier and
	/// `am` gets no fast segment -- and again only when a later resize makes
	/// that NEWLY true. Returns whether it warned.
	///
	/// `eprintln!`, not `log::warn!`: this crate installs no logger (see
	/// `merged_stack.rs`), so a `log` warning would print nowhere. Stderr is
	/// where the crate's other diagnostics go.
	///
	/// Checked from `resize_fast_tier` and `resize`, not `new`:
	/// `init_policy_stack` builds this stack against a 20%-of-`max_size`
	/// placeholder and `new_hybrid` sends the real budget through
	/// `resize_fast_tier` straight away, so that is where it first arrives.
	fn warn_if_carve_out_fills_fast_tier(&mut self) -> bool {
		let fills = self.fast_capacity > 0 && self.a1_in_capacity >= self.fast_capacity;
		let newly = fills && !self.carve_out_fills_fast_tier;
		self.carve_out_fills_fast_tier = fills;

		if newly {
			eprintln!(
				"2q-full-fast-admission-compact-hybrid: a1_in's configured capacity (k_in * max_size = {} bytes) meets or exceeds the fast-tier budget ({} bytes); a1_in is clamped to the whole fast tier, so `am` gets no fast segment and every promotion will demote straight back out. Lower k_in or raise fast_tier_size.",
				self.a1_in_capacity,
				self.fast_capacity,
			);
		}

		newly
	}

	/// Two of the three queues answer structurally: `a1_in` is DRAM by the
	/// fast-admission rule, `a1_out` is PMEM by the demotion rule. Only `am`
	/// stores a tier.
	pub fn tier_of(&self, key: HashedKey) -> Option<Tier> {
		let payload = self.queues.payload(key)?;

		match Queue::from_u8(payload.queue) {
			Queue::A1In => Some(Tier::Fast),
			Queue::A1Out => Some(Tier::Slow),
			Queue::Am => payload.tier,
		}
	}

	/// Records a size change without altering the key's queue or tier, and
	/// returns whether it GREW a key in `a1_in`.
	///
	/// `am`'s budget does not move with it: the carve-out `am` settles against
	/// is the *fixed* `a1_in_carve_out()`, not live `a1_in_used`. But `a1_in`'s
	/// own budget can now be exceeded, and a hit on an `a1_in` key is a no-op
	/// that will not move it out, so the caller re-settles `a1_in` when this
	/// returns `true`.
	fn resize_key(&mut self, key: HashedKey, new_size: ObjectSize, new_resident: u8) -> bool {
		let Some(payload) = self.queues.payload_mut(key) else { return false };

		let old_migrating = payload.migrating();
		payload.size = new_size;
		payload.dram_resident = new_resident;
		let delta = payload.migrating() as i64 - old_migrating as i64;
		let (queue, tier) = (Queue::from_u8(payload.queue), payload.tier);
		let grew_in_a1_in = queue == Queue::A1In && delta > 0;

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

		grew_in_a1_in
	}

	/// Treats an already-tracked key as accessed, dispatching on its queue.
	///
	/// The `A1In` arm is the fidelity point of the whole design and is
	/// deliberately empty: a hit on a probation key does NOTHING. Faithful to
	/// `TwoQCompactStack`, where `a1_out.remove` misses and `am.move_front` is a
	/// silent no-op; and the key is already Fast, so there is nothing to
	/// migrate either.
	fn touch(&mut self, key: HashedKey, structural: bool) {
		match self.queues.payload(key).map(|p| Queue::from_u8(p.queue)) {
			Some(Queue::A1In) => {},
			Some(Queue::A1Out) => self.promote_from_a1_out(key, structural),
			Some(Queue::Am) => self.touch_am(key, structural),
			None => {},
		}
	}

	/// `TwoQCompactStack::restructure_to_fit`, with the transition it performs being a
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
	///
	/// Drains against [`Self::effective_a1_in_capacity`] -- the CLAMPED
	/// carve-out net of `a1_in`'s share of the reservation -- never the
	/// cache-sized `a1_in_capacity` field. Read once: a demotion moves a key
	/// between queues but keeps it tracked, so the reservation cannot move
	/// underneath the pass. Run before the push, it cannot see the incoming
	/// key's own metadata. While `am` can absorb that metadata (the carve-out
	/// and the reservation fit the tier together) `a1_in`'s budget does not
	/// depend on it; past that point `a1_in` pays it, and can sit over its
	/// budget by at most one `shared_overhead` until the next settle.
	fn settle_a1_in(&mut self, incoming_size: ObjectSize) {
		let incoming = incoming_size as CacheSize;

		// At the DRAIN TARGET of `a1_in`'s budget (S5), as `am` rests at the
		// drain target of its own.
		let budget = drain_target::bytes(self.effective_a1_in_capacity());

		while self.a1_in_used + incoming > budget {
			let Some(key) = self.queues.back(Q_A1_IN) else { break };
			let size = self.queues.payload(key).map(|p| p.migrating()).unwrap_or(0);

			self.queues.move_to_front_of(Q_A1_IN, Q_A1_OUT, key);

			if let Some(p) = self.queues.payload_mut(key) {
				p.queue = Queue::A1Out as u8;
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
	/// Emits a genuine `(key, Tier::Fast)` migration -- unlike a promotion out
	/// of a DRAM-resident admission FIFO
	/// (`TwoQFastAdmissionReprieveCompactHybridStack`), which is a Fast->Fast
	/// bookkeeping move. Here the bytes really do live in PMEM beforehand,
	/// because `a1_out` is the slow tier.
	fn promote_from_a1_out(&mut self, key: HashedKey, structural: bool) {
		let Some(payload) = self.queues.payload(key) else { return };
		let size_bytes = payload.migrating();

		// S5: a STRUCTURAL key moves to `am`'s front all the same -- its place
		// in the order -- with tier slow; its bytes stay in PMEM.
		if structural {
			self.queues.move_to_front_of(Q_A1_OUT, Q_AM, key);
			self.a1_out_used = self.a1_out_used.saturating_sub(size_bytes);

			if let Some(p) = self.queues.payload_mut(key) {
				p.queue = Queue::Am as u8;
				p.tier = Some(Tier::Slow);
			}

			self.am_slow_used += size_bytes;
			self.am_count += 1;

			self.settle_fast_tier();
			return;
		}

		self.queues.move_to_front_of(Q_A1_OUT, Q_AM, key);
		self.a1_out_used = self.a1_out_used.saturating_sub(size_bytes);

		if let Some(p) = self.queues.payload_mut(key) {
			p.queue = Queue::Am as u8;
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
	fn touch_am(&mut self, key: HashedKey, structural: bool) {
		let previous_tier = self.queues.payload(key).and_then(|p| p.tier);

		let already_at_front = self.queues.front(Q_AM) == Some(key);
		let is_boundary = self.am_boundary == Some(key);

		// Read the neighbour BEFORE moving: once the key is at the front its
		// predecessor is gone, and the boundary must step back to whatever fast
		// key was in front of it (S5: past any structural ones).
		let new_boundary_if_moved = if is_boundary && !already_at_front {
			prev_fast(&self.queues, key)
		} else {
			None
		};

		self.queues.move_front(Q_AM, key);

		if is_boundary && !already_at_front {
			self.am_boundary = new_boundary_if_moved;
		}

		// S5: a STRUCTURAL key moves to the front all the same -- its place in
		// the order -- with tier slow: a slow one is not promoted, a fast one
		// leaves the fast set, pushed `(key, Slow)` (its placement changed).
		if structural {
			if previous_tier == Some(Tier::Fast) {
				let size = self.queues.payload(key).map(|p| p.migrating()).unwrap_or(0);
				self.am_fast_used = self.am_fast_used.saturating_sub(size);
				self.am_fast_count = self.am_fast_count.saturating_sub(1);
				self.am_slow_used += size;

				if let Some(p) = self.queues.payload_mut(key) {
					p.tier = Some(Tier::Slow);
				}

				if self.am_boundary == Some(key) {
					self.am_boundary = None;
				}

				self.migrations.push((key, Tier::Slow));
			}

			self.settle_fast_tier();
			return;
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
		}

		// Fast, at the front; with no fast key in front of it, the boundary.
		if self.am_boundary.is_none() {
			self.am_boundary = Some(key);
		}

		self.settle_fast_tier();

		// Same ordering and same guard as `promote_from_a1_out` above.
		if promoted && self.queues.payload(key).and_then(|p| p.tier) == Some(Tier::Fast) {
			self.migrations.push((key, Tier::Fast));
		}
	}

	/// Demotes from the tier boundary until `am_fast_used` is back within the
	/// budget. The victim is always `am_boundary`, so nothing is searched.
	///
	/// The ceiling is [`Self::effective_am_fast_capacity`] -- reservations
	/// come off first, and the drain runs against the remainder.
	fn settle_fast_tier(&mut self) {
		let effective = self.effective_am_fast_capacity();
		let target = drain_target::bytes(effective);

		while self.am_fast_used > target {
			let Some(demote_key) = self.am_boundary else { break };
			let size = self.queues.payload(demote_key).map(|p| p.migrating()).unwrap_or(0);
			let new_boundary = prev_fast(&self.queues, demote_key);

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

	/// The FIRST eviction victim, per `TwoQCompactStack::evict_one`.
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

	/// The last resort, per `TwoQCompactStack::evict_one`.
	fn evict_am_tail(&mut self) -> Option<HashedKey> {
		let (key, payload) = self.queues.pop_back(Q_AM)?;
		let size = payload.migrating();

		self.am_count = self.am_count.saturating_sub(1);

		match payload.tier {
			Some(Tier::Fast) => {
				self.am_fast_used = self.am_fast_used.saturating_sub(size);
				self.am_fast_count = self.am_fast_count.saturating_sub(1);

				// The tail of `am` can only be Fast-tagged if every `am` key
				// behind the boundary is gone, in which case the boundary
				// equalled this key. Re-point it at the nearest fast key from
				// the new tail (S5: past any structural ones), or none.
				if self.am_boundary == Some(key) {
					self.am_boundary = fast_at_or_before(&self.queues, self.queues.back(Q_AM));
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
		self.insert_with(key, size, dram_resident, Placement::Normal);
	}

	fn insert_placed(
		&mut self,
		key: HashedKey,
		size: ObjectSize,
		dram_resident: ObjectSize,
		_event: SetEvent,
		placement: Placement,
	) -> Placement {
		self.insert_with(key, size, dram_resident, placement)
	}

	fn set_dram_metadata(&mut self, measured: Option<CacheSize>) {
		self.measured = measured;
	}

	/// S5: every settle, against the current budget (the policy worker's
	/// end-of-pass step).
	fn resettle(&mut self) {
		// `a1_in` first -- its budget moves with the reservation and it is
		// policed only on an insert, a growth or a resize -- then `am`.
		self.settle_a1_in(0);
		self.settle_fast_tier();
	}

	fn update(&mut self, key: HashedKey) {
		if let Some(payload) = self.queues.payload(key) {
			let structural = self.structural(payload.migrating());
			self.touch(key, structural);
		}
	}

	fn remove(&mut self, key: HashedKey) {
		let Some(payload) = self.queues.payload(key) else { return };
		let size = payload.migrating();

		match Queue::from_u8(payload.queue) {
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
						prev_fast(&self.queues, key)
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
		self.warn_if_carve_out_fills_fast_tier();

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

	/// `TwoQCompactStack::evict_one`, verbatim: `a1_out` tail, then `a1_in` tail, then
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
		self.warn_if_carve_out_fills_fast_tier();

		// Shrinking the tier shrinks `a1_in_carve_out()` and moves the point at
		// which `a1_in` starts paying for the reservation: demote `a1_in`'s
		// tail into `a1_out` down to its new budget first (never evicting),
		// then settle `am` against what is left.
		self.settle_a1_in(0);
		self.settle_fast_tier();
	}

	/// `tier_of`: `a1_in` is DRAM, `a1_out` slow, `am` placed by its tier,
	/// and every crossing is pushed.
	/// See `PolicyStack::placement_of`.
	fn placement_of(&self, key: HashedKey) -> Option<Tier> {
		self.tier_of(key)
	}

	fn drain_tier_migrations(&mut self) -> Vec<(HashedKey, Tier)> {
		std::mem::take(&mut self.migrations)
	}

	fn structure_bytes(&self) -> Option<crate::meta::NodeBytes> {
		Some(crate::meta::NodeBytes::stack(self.queues.allocated_bytes()))
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

/// The DRAM ceiling, ported with the clamp from
/// `TwoQFastAdmissionReprieveCompactHybridStack`'s module of the same name.
/// Both DRAM segments -- `a1_in` and the main queue's fast portion --
/// are budgeted out of one `fast_capacity`, but `a1_in`'s own capacity
/// is `k_in * max_size`, a fraction of the CACHE.
///
/// `RATIOS` put that at the 4_000 B tier and above it. Only 0.6 and 1.0
/// detect a missing clamp: 0.4 is the equality case, where `min` returns the
/// raw capacity and the clamp is numerically inert -- it is there to show a
/// carve-out that exactly fills the tier does not wedge. `FITTING` (0.25)
/// covers the regime where the carve-out fits and the reservation lands on
/// `am` first, and pins those budgets to the pre-clamp ones.
#[cfg(test)]
mod dram_ceiling_tests {
	use super::*;

	const MAX_SIZE: CacheSize = 10_000;
	const FAST: CacheSize = 4_000;
	const SIZE: ObjectSize = 100;
	const OVERHEAD: CacheSize = 8;
	const KEYS: HashedKey = 120;

	/// `settle_a1_in` runs BEFORE the push (`restructure_to_fit`), so it
	/// cannot see the incoming key's own metadata. At the clamped `RATIOS`
	/// `am` has no room for any of the reservation, `a1_in` pays all of it,
	/// and right after an admission `a1_in` may sit over its budget by that
	/// key's one `OVERHEAD`. The settle at the end of the drive must be exact.
	const SLACK: CacheSize = OVERHEAD;

	/// `k_in * MAX_SIZE` = 4_000 B (exactly the tier), 6_000 B and
	/// 10_000 B.
	const RATIOS: [f64; 3] = [0.4, 0.6, 1.0];

	/// `k_in * MAX_SIZE` = 2_500 B: the carve-out fits the 4_000 B tier and
	/// leaves `am` 1_500 B, which the drive's reservation (at most 100 keys at
	/// 8 B) never outgrows.
	const FITTING: f64 = 0.25;

	fn stack(ratio: f64) -> TwoQFullFastAdmissionCompactHybridStack {
		TwoQFullFastAdmissionCompactHybridStack::new(ratio, 0.5, MAX_SIZE, FAST).with_shared_overhead(OVERHEAD)
	}

	/// What `PolicyWorker::apply_evictions` does after every event: evict
	/// while the stack asks for it or the cache is over `max_size`.
	fn evict_while_asked(stack: &mut TwoQFullFastAdmissionCompactHybridStack) {
		while (stack.needs_capacity_eviction()
			|| stack.fast_bytes_used() + stack.slow_bytes_used() > MAX_SIZE)
			&& stack.evict_one().is_some()
		{}
	}

	/// All the DRAM this stack holds against the tier it was given: both
	/// segments' values (`fast_bytes_used`) plus the metadata reservation it
	/// reports.
	fn assert_within_the_fast_tier(stack: &TwoQFullFastAdmissionCompactHybridStack, slack: CacheSize, context: &str) {
		let values = stack.fast_bytes_used();
		let reserved = stack.dram_reserved_bytes();

		assert!(
			values + reserved <= stack.fast_capacity + slack,
			"{context}: {} B of DRAM ({values} B of values + {reserved} B reserved) on a {} B fast tier",
			values + reserved,
			stack.fast_capacity,
		);
	}

	/// The budget identity: both DRAM segments' budgets plus ONE reservation
	/// are the fast tier exactly, however far `k_in * max_size`
	/// overshoots it. Unclamped, 0.6 gave `a1_in` a 6_000 B budget of its
	/// own on a 4_000 B tier while the main queue's budget saturated to 0.
	#[test]
	fn dram_budgets_never_over_subscribe_the_fast_tier() {
		for ratio in RATIOS {
			let mut stack = stack(ratio);

			for key in 1..=5 {
				stack.insert(key, SIZE);
			}

			let total = stack.effective_a1_in_capacity()
				+ stack.effective_am_fast_capacity()
				+ stack.reserved_overhead();

			assert_eq!(
				total, FAST,
				"ratio {ratio}: the two DRAM budgets plus the reservation come to {total} B, not the {FAST} B fast tier",
			);
		}
	}

	/// The ceiling on live bytes, driven the way the worker drives the stack:
	/// admissions, hits on the newest keys (no-ops in `a1_in`) and on older ones
	/// (promotions out of `a1_out` into an `am` with no fast segment left),
	/// re-admissions of early keys, and every eviction the stack asks for.
	/// Includes a carve-out exactly the size of the tier, which must not wedge
	/// or panic.
	///
	/// Per-step checks allow `SLACK`; the final re-settle allows nothing.
	#[test]
	fn admissions_and_hits_never_hold_more_dram_than_the_fast_tier() {
		for ratio in RATIOS {
			let mut stack = stack(ratio);

			for key in 1..=KEYS {
				stack.insert(key, SIZE);
				evict_while_asked(&mut stack);
				assert_within_the_fast_tier(&stack, SLACK, &format!("ratio {ratio}, admitted {key}"));

				if key % 3 == 0 {
					for hit in [key - 1, key / 2] {
						if stack.contains(hit) {
							stack.update(hit);
							evict_while_asked(&mut stack);
							assert_within_the_fast_tier(&stack, SLACK, &format!("ratio {ratio}, hit {hit}"));
						}
					}
				}
			}

			for key in 1..=KEYS / 3 {
				stack.insert(key, SIZE);
				evict_while_asked(&mut stack);
				assert_within_the_fast_tier(&stack, SLACK, &format!("ratio {ratio}, re-admitted {key}"));
			}

			// A settle with every key's metadata in view: exact.
			stack.resize_fast_tier(FAST);
			assert_within_the_fast_tier(&stack, 0, &format!("ratio {ratio}, re-settled"));

			assert!(stack.len() > 0, "ratio {ratio}: the stack must still hold keys");
		}
	}

	/// Why the clamp is an accessor and not a value fixed at construction: only
	/// `fast_capacity` moves here, and `a1_in`'s budget has to move with
	/// it. `resize_fast_tier` re-runs `settle_a1_in(0)`, which
	/// demotes the excess into `a1_out` -- nothing is lost.
	#[test]
	fn shrinking_the_fast_tier_settles_a1_in_to_the_new_clamp() {
		// 0.25 * 10_000 = 2_500 B of admission queue: fits the 4_000 B tier.
		let mut stack = stack(0.25);

		for key in 1..=5 {
			stack.insert(key, 400);
			evict_while_asked(&mut stack);
		}

		assert_eq!(stack.fast_bytes_used(), 2_000, "all five admitted straight to DRAM");

		stack.resize_fast_tier(1_000);
		evict_while_asked(&mut stack);

		assert_within_the_fast_tier(&stack, 0, "after shrinking the tier to 1_000 B");

		assert!(
			stack.a1_in_used <= stack.effective_a1_in_capacity(),
			"a1_in holds {} B over its {} B budget",
			stack.a1_in_used,
			stack.effective_a1_in_capacity(),
		);

		assert_eq!(stack.len(), 5, "nothing is evicted: the excess is demoted");
		assert_eq!(stack.slow_object_count(), 3, "the three oldest went to PMEM");
	}

	/// The budget identity where the carve-out FITS, with a reservation: `am`
	/// pays it first, and once it outgrows `am`'s 1_500 B `a1_in` pays the
	/// rest, up to a reservation that is the whole tier.
	#[test]
	fn a_fitting_carve_out_keeps_the_budget_identity() {
		// Five keys: 0, 40, 500, 2_000 and 4_000 B reserved.
		for overhead in [0, OVERHEAD, 100, 400, 800] {
			let mut stack =
				TwoQFullFastAdmissionCompactHybridStack::new(FITTING, 0.5, MAX_SIZE, FAST).with_shared_overhead(overhead);

			for key in 1..=5 {
				stack.insert(key, SIZE);
			}

			let total = stack.effective_a1_in_capacity()
				+ stack.effective_am_fast_capacity()
				+ stack.reserved_overhead();

			assert_eq!(
				total, FAST,
				"{} B reserved: the two DRAM budgets plus the reservation come to {total} B, not the {FAST} B fast tier",
				stack.reserved_overhead(),
			);
		}
	}

	/// The ceiling on live bytes where the carve-out fits, driven as the
	/// clamped ratios are above.
	///
	/// `a1_in`'s budget here is its whole carve-out whatever the reservation,
	/// so its own settle is exact. `am` pays the reservation, and `am` settles
	/// on an `a1_out` promotion, an `am` hit and a resize -- not on an
	/// admission, and not on an `a1_in` hit (a no-op), exactly as before the
	/// clamp -- so each admission since `am` last settled can add one
	/// `OVERHEAD` it has not yet settled against. The bound is therefore exact
	/// after every event that settles `am` and over by at most the metadata of
	/// the keys admitted since; the final re-settle is exact. (In this drive
	/// the headroom `drain_target` leaves absorbs that metadata and no event
	/// goes over the tier at all; the bound is what the code guarantees.)
	#[test]
	fn a_fitting_carve_out_holds_the_ceiling_through_admissions_and_hits() {
		let mut stack = stack(FITTING);
		let mut unsettled: CacheSize = 0;

		let settles_am = |stack: &TwoQFullFastAdmissionCompactHybridStack, key: HashedKey| {
			matches!(
				stack.queues.payload(key).map(|p| Queue::from_u8(p.queue)),
				Some(Queue::A1Out | Queue::Am),
			)
		};

		for key in 1..=KEYS {
			stack.insert(key, SIZE);
			unsettled += 1;
			evict_while_asked(&mut stack);
			assert_within_the_fast_tier(
				&stack,
				unsettled * OVERHEAD,
				&format!("admitted {key}, {unsettled} admission(s) since am settled"),
			);

			if key % 3 == 0 {
				for hit in [key - 1, key / 2] {
					if stack.contains(hit) {
						if settles_am(&stack, hit) {
							unsettled = 0;
						}

						stack.update(hit);
						evict_while_asked(&mut stack);
						assert_within_the_fast_tier(&stack, unsettled * OVERHEAD, &format!("hit {hit}"));
					}
				}
			}
		}

		for key in 1..=KEYS / 3 {
			if !stack.contains(key) {
				unsettled += 1;
			} else if settles_am(&stack, key) {
				unsettled = 0;
			}

			stack.insert(key, SIZE);
			evict_while_asked(&mut stack);
			assert_within_the_fast_tier(&stack, unsettled * OVERHEAD, &format!("re-admitted {key}"));
		}

		stack.resize_fast_tier(FAST);
		assert_within_the_fast_tier(&stack, 0, "re-settled");

		assert!(
			stack.reserved_overhead() <= FAST - stack.a1_in_carve_out(),
			"the drive left the fitting regime: {} B reserved",
			stack.reserved_overhead(),
		);
		assert!(stack.len() > 0, "the stack must still hold keys");
	}

	/// The main-first split itself, pinned in every regime (five keys each).
	#[test]
	fn main_first_shares_pin_every_regime() {
		// (ratio, overhead per key, (a1_in_share, am_share), a1_in budget, am budget)
		let cases: [(f64, CacheSize, (CacheSize, CacheSize), CacheSize, CacheSize); 4] = [
			// Carve-out 2_500 B, 40 B reserved: `am` pays all of it, and
			// `a1_in` keeps its whole carve-out.
			(FITTING, OVERHEAD, (0, 40), 2_500, 1_460),
			// 2_000 B reserved: `am` can pay 1_500 of it, `a1_in` the other 500.
			(FITTING, 400, (500, 1_500), 2_000, 0),
			// 5_000 B reserved, more than the tier: no value budget left.
			(FITTING, 1_000, (3_500, 1_500), 0, 0),
			// Clamped carve-out (6_000 B -> the 4_000 B tier), 40 B reserved:
			// `am` has no room, so `a1_in` pays it all.
			(0.6, OVERHEAD, (40, 0), 3_960, 0),
		];

		for (ratio, overhead, shares, a1_in, am) in cases {
			let mut stack =
				TwoQFullFastAdmissionCompactHybridStack::new(ratio, 0.5, MAX_SIZE, FAST).with_shared_overhead(overhead);

			for key in 1..=5 {
				stack.insert(key, SIZE);
			}

			let context = format!("ratio {ratio}, {} B reserved", stack.reserved_overhead());

			assert_eq!(stack.reserved_shares(), shares, "{context}: (a1_in_share, am_share)");
			assert_eq!(stack.effective_a1_in_capacity(), a1_in, "{context}: a1_in budget");
			assert_eq!(stack.effective_am_fast_capacity(), am, "{context}: am budget");
		}
	}

	/// The point of main-first: wherever the reservation fits beside the
	/// carve-out, both budgets are 34c6a4e's. That commit settled `am` against
	/// `fast_capacity.saturating_sub(a1_in_capacity).saturating_sub(
	/// reserved_overhead())` and drained `a1_in` against the raw
	/// `a1_in_capacity`; both are restated here verbatim. `am`'s formula holds
	/// for EVERY input; `a1_in`'s wherever `reserved <= fast - carve`, except
	/// that above the tier (where that forces `reserved == 0`) `a1_in`'s
	/// budget is the clamp's `fast` rather than the raw capacity.
	#[test]
	fn budgets_equal_the_pre_clamp_formulas_wherever_the_reservation_fits_beside_the_carve_out() {
		let mut checked = 0;

		for ratio in [0.0, 0.1, FITTING, 0.39, 0.4, 0.41, 0.6, 1.0] {
			for fast in [0, 1, 1_000, 2_500, FAST, MAX_SIZE] {
				for overhead in [0, 1, OVERHEAD, 100, 1_000] {
					for keys in [0, 1, 5, 20] {
						let mut stack = TwoQFullFastAdmissionCompactHybridStack::new(ratio, 0.5, MAX_SIZE, fast)
							.with_shared_overhead(overhead);

						for key in 1..=keys {
							stack.insert(key, 1);
						}

						let a1_in_capacity = stack.a1_in_capacity;
						let reserved = stack.reserved_overhead();
						let carve = a1_in_capacity.min(fast);
						let context = format!("ratio {ratio}, fast {fast}, {reserved} B reserved");

						assert_eq!(
							stack.effective_am_fast_capacity(),
							fast.saturating_sub(a1_in_capacity).saturating_sub(reserved),
							"{context}: am's budget moved from 34c6a4e's",
						);

						if reserved > fast - carve {
							continue;
						}

						let pre_clamp_a1_in = if a1_in_capacity <= fast { a1_in_capacity } else { fast };

						assert_eq!(
							stack.effective_a1_in_capacity(),
							pre_clamp_a1_in,
							"{context}: a1_in's budget moved from 34c6a4e's",
						);

						checked += 1;
					}
				}
			}
		}

		assert!(checked >= 300, "only {checked} fitting configurations were checked");
	}

	/// A reservation at or over the whole tier. `am`'s budget and `a1_in`'s
	/// are both 0, so eff is 0 and every new key is STRUCTURAL (S5): it goes
	/// where `a1_in`'s overflow goes, `a1_out`'s front, slow -- built slow,
	/// nothing pushed -- and no value is left in DRAM. (Before S5 each
	/// admission entered `a1_in`, demoting the one before it, so `a1_in` held
	/// the newest key in DRAM on a tier the metadata had filled.) Nothing is
	/// evicted for it. Before the clamp and the split `a1_in` kept its whole
	/// raw capacity in DRAM on top of the reservation.
	#[test]
	fn a_metadata_bound_tier_places_each_new_key_in_a1_out() {
		// k_in 0.01: a 100 B `a1_in`, so each admission demotes the key before
		// it into `a1_out`, where a hit proves it into `am`.
		let mut stack =
			TwoQFullFastAdmissionCompactHybridStack::new(0.01, 0.5, MAX_SIZE, FAST).with_shared_overhead(100);

		for key in 1..=31 {
			stack.insert(key, SIZE);

			if key > 1 {
				stack.update(key - 1);
			}

			evict_while_asked(&mut stack);
		}

		assert_eq!(stack.len(), 31, "every key is still tracked");
		assert_eq!(stack.reserved_overhead(), 3_100, "31 keys at 100 B, which still fits the 4_000 B tier");

		// The tier shrinks under the reservation.
		stack.resize_fast_tier(2_000);
		evict_while_asked(&mut stack);

		assert_eq!(stack.reserved_shares(), (1_200, 1_900));
		assert_eq!(stack.effective_a1_in_capacity(), 0, "no a1_in budget left");
		assert_eq!(stack.effective_am_fast_capacity(), 0, "no am budget left");
		assert_eq!(stack.fast_bytes_used(), 0, "a1_in and am demoted every value");

		stack.drain_tier_migrations();

		for key in 100..110 {
			stack.insert(key, SIZE);
			evict_while_asked(&mut stack);

			assert_eq!(stack.fast_bytes_used(), 0, "key {key}: no value is in DRAM");
			assert_eq!(stack.tier_of(key), Some(Tier::Slow), "key {key} was placed in a1_out");
			assert!(stack.drain_tier_migrations().is_empty(), "key {key}: built slow, nothing pushed");
			assert_eq!(stack.len(), 31 + (key - 99) as usize, "key {key}: nothing is evicted");
		}
	}

	/// A re-set is a hit, and an `a1_in` hit is a no-op, so a key that GROWS
	/// in `a1_in` stays there. `insert_resident` then re-settles `a1_in`
	/// rather than leaving the overrun to the next admission.
	#[test]
	fn re_setting_a_key_larger_in_a1_in_re_settles_a1_in() {
		let mut stack = stack(FITTING);

		// Forty admissions: `a1_in` holds the 24 newest (2_400 B: it rests at
		// the drain target of its 2_500 B budget since S5), `a1_out` the 16
		// oldest.
		for key in 1..=40 {
			stack.insert(key, SIZE);
			evict_while_asked(&mut stack);
		}

		// Proving 15 of those fills `am`'s fast segment to its budget.
		for key in 1..=15 {
			stack.update(key);
			evict_while_asked(&mut stack);
		}

		assert_eq!(stack.a1_in_used, 2_400);
		assert_within_the_fast_tier(&stack, 0, "before the re-set");

		// Key 40 is `a1_in`'s newest; re-set it 300 B larger.
		stack.insert(40, 400);
		evict_while_asked(&mut stack);

		assert!(
			stack.a1_in_used <= stack.effective_a1_in_capacity(),
			"a1_in holds {} B over its {} B budget",
			stack.a1_in_used,
			stack.effective_a1_in_capacity(),
		);
		assert_within_the_fast_tier(&stack, 0, "after re-setting key 40 to 400 B");

		assert_eq!(stack.tier_of(40), Some(Tier::Fast), "the re-set key is still in a1_in");

		for key in 17..=19 {
			assert_eq!(stack.tier_of(key), Some(Tier::Slow), "key {key}, a1_in's oldest, was demoted");
		}

		assert_eq!(stack.len(), 40, "nothing is evicted: the excess is demoted");
	}
}

/// The carve-out warning: one stderr line per crossing of
/// `a1_in_capacity >= fast_capacity`, checked from both resize entry points.
#[cfg(test)]
mod carve_out_warning_tests {
	use super::*;

	#[test]
	fn the_carve_out_warning_fires_once_per_crossing() {
		// 0.6 * 1_000 = 600 B of admission queue against a 1_000 B tier: fits.
		let mut stack = TwoQFullFastAdmissionCompactHybridStack::new(0.6, 0.5, 1_000, 1_000);
		assert!(!stack.warn_if_carve_out_fills_fast_tier(), "a queue that fits the tier must not warn");

		stack.fast_capacity = 600;
		assert!(stack.warn_if_carve_out_fills_fast_tier(), "a 600 B queue on a 600 B tier covers it and must warn");
		assert!(!stack.warn_if_carve_out_fills_fast_tier(), "once per crossing, not once per check");

		stack.resize_fast_tier(1_000);
		assert!(!stack.carve_out_fills_fast_tier, "resize_fast_tier re-checks: 600 B fits 1_000 B again");

		stack.resize_fast_tier(400);
		assert!(stack.carve_out_fills_fast_tier, "resize_fast_tier re-checks: 600 B covers 400 B");

		stack.resize(500);
		assert!(!stack.carve_out_fills_fast_tier, "resize re-checks: 0.6 * 500 = 300 B fits 400 B");
	}
}
