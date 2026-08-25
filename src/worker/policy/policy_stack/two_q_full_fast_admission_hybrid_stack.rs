/*
 * Copyright (c) Kia Shakiba
 *
 * This source code is licensed under the GNU AGPLv3 license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! `TwoQFullFastAdmissionHybridStack` — the **full**, three-queue 2Q of
//! [`TwoQStack`](super::two_q_stack) with fast-tier admission laid over it.
//! For `PaperPolicy::TwoQFullFastAdmissionHybrid`.
//!
//! This is the only hybrid design in the tree whose queue algorithm matches
//! `PaperPolicy::TwoQ`'s. `two_q_hybrid_cache` and its fast-admission /
//! reprieve / ghost siblings implement **Simplified 2Q** — a one-hit FIFO
//! admission filter in front of an LRU — which differs from the three-queue
//! algorithm in queue count, in what a FIFO hit does, in where the promotion
//! signal comes from, in eviction order, and in parameter count. "full" is
//! the paper's own word for the three-queue form, and the two-parameter
//! policy string (`2q-full-fast-admission-hybrid-<k_in>-<k_out>`, the only
//! two-parameter hybrid here) makes the distinction visible at a glance.
//!
//! ## The three queues, and which tier each lives in
//!
//! | queue | role | tier |
//! |---|---|---|
//! | `a1_in` | probation FIFO for brand-new keys, capped at `k_in * max_size` | **FAST**, structurally |
//! | `a1_out` | overflow FIFO holding keys aged out of `a1_in`, capped at `k_out * max_size` | **SLOW**, structurally |
//! | `am` | the main LRU of proven keys | tier-**segmented** at `am_boundary`, exactly like `LruHybridStack` |
//!
//! The queue hierarchy and the tier hierarchy are the *same* hierarchy here,
//! which is the argument for building this particular variant: the
//! `a1_in -> a1_out` transition, free in the single-tier `TwoQStack`, is
//! precisely a DRAM->PMEM demotion, and the `a1_out -> am` promotion is
//! precisely a PMEM->DRAM migration. Nothing is contrived to make them line
//! up.
//!
//! ## `a1_out` holds REAL RESIDENT OBJECTS, not ghosts
//!
//! This is the decision that governs everything else, and the single most
//! likely thing to be "fixed" in review by someone who has read the paper
//! rather than [`TwoQStack`]. In this repository's regular 2Q,
//! `a1_out` holds live `Object { key, size }` values, counts toward
//! `contains()`/`len()`, and is the *first* eviction victim; a hit in
//! `a1_out` is a real cache hit served from the resident copy, not a miss
//! plus a reload. This stack is faithful to that:
//!
//! * `contains()`/`len()` count `a1_out` members. They are resident.
//! * An `a1_out` hit is a hit: [`Self::promote_from_a1_out`] moves the live
//!   key to `am`'s MRU end at `Tier::Fast` and emits one `(key, Tier::Fast)`
//!   migration (a genuine PMEM->DRAM data move).
//! * `a1_out`'s bytes are counted in `slow_bytes_used()`.
//!
//! There is deliberately **no ghost queue**. `a1_out` supersedes one — it
//! carries strictly more information (the identity *and* the data) — and the
//! ghost design point is already occupied by `TwoQGhostHybridStack`. A
//! consequence worth noting: this stack's `reserved_overhead` is a single
//! term (`entries.len() * shared_overhead`) rather than the ghost variant's
//! two, because an `a1_out`-resident key is a tracked key like any other, so
//! its `entries` row, list node and hashtable slot are already covered.
//!
//! ## An `a1_in` hit is a COMPLETE no-op
//!
//! Per the regular stack: a hit in `a1_out` promotes, a hit in `a1_in` is
//! deliberately ignored (`TwoQStack::update` looks in `a1_out` first, then
//! calls `am.move_front`, which is a silent no-op for an `a1_in`-resident
//! key). 2Q ignores `a1_in` hits precisely so that a burst of references to a
//! just-loaded page — a scan touching a page twice — cannot buy promotion.
//!
//! So [`Self::update`] on an `a1_in` key mutates nothing: no list move, no
//! tier change, no migration, no counter. It is already Fast by
//! construction, so there is nothing to migrate either. The hottest path in
//! the policy does zero work.
//!
//! Contrast `TwoQHybridStack`, where this same event is *the* promotion
//! trigger (and where a test, `reaccessing_a_fifo_key_promotes_it_to_fast`,
//! pins that behaviour in place). That inversion is the main reason this
//! variant exists.
//!
//! ## `a1_in` overflow DEMOTES; `a1_out` overflow EVICTS
//!
//! [`Self::settle_a1_in`] is `TwoQStack::restructure_to_fit`, with the
//! transition it performs now being a real tier migration: the `a1_in` tail
//! is spliced onto `a1_out`'s head and a `(key, Tier::Slow)` migration is
//! emitted. **No object leaves the cache.** It runs synchronously inside
//! `insert` and touches only this stack's own bookkeeping plus the migration
//! queue — the same shape as `TwoQFastAdmissionReprieveHybridStack::settle_fifo_queue`.
//!
//! A `PolicyStack` must never self-evict: it holds no reference to the shared
//! object map or to `status`, so dropping a key from its own bookkeeping
//! would silently desync the two permanently (this bug was found the hard way
//! in `TwoQHybridStack` — see `CLAUDE.md`). Real removal always goes through
//! `PolicyWorker::apply_evictions`'s `evict_one()` + `erase()` pairing, which
//! is why over-capacity is *reported* rather than acted on:
//!
//! ```text
//! needs_capacity_eviction() == (a1_out_used > a1_out_capacity)
//! ```
//!
//! Deliberately `a1_out` only. `a1_in` overflow must not set it — that would
//! evict where the algorithm demotes. Because [`Self::evict_one`] drains
//! `a1_out` first, `apply_evictions`'s loop is guaranteed to make progress on
//! exactly the over-budget queue. This is what makes `k_out` a live
//! parameter for the first time in this codebase (`TwoQStack` writes
//! `a1_out.max_size` and never reads it).
//!
//! ## Eviction order: `a1_out` tail, then `a1_in` tail, then `am` LRU tail
//!
//! Verbatim from `TwoQStack::evict_one`. In a tiered cache this means
//! terminal eviction frees the SLOW tier first and DRAM last, which is both
//! coherent — the objects with the weakest evidence of value are the ones
//! already sitting in PMEM — and desirable, since PMEM is the large cheap
//! tier you want recycled.
//!
//! It does mean a recently demoted object can be evicted ahead of an ancient
//! `am` object. That is exactly what `TwoQStack` does and is therefore
//! correct by definition here; it is called out because it *looks* like a bug
//! in review. `eviction_order_matches_plain_two_q_stack` locks it in by
//! replaying the regular stack's own test trace.
//!
//! ## Accounting: `a1_in` competes for the same DRAM budget as `am`'s fast segment
//!
//! `a1_in` is DRAM now, so its byte budget is a reservation carved out of
//! `fast_capacity` rather than an independent PMEM budget — same treatment,
//! and same reasoning, as `TwoQFastAdmissionHybridStack`:
//!
//! ```text
//! effective_am_fast_capacity() = fast_capacity - a1_in_capacity - reserved_overhead()
//! ```
//!
//! The reservation is the **fixed `a1_in_capacity`, not the live
//! `a1_in_used`**: charging live usage would make `am`'s budget breathe with
//! probation occupancy and churn demotions as `a1_in` fills and drains.
//! `a1_out` is *not* carved out of it — those bytes are PMEM.
//! `reserved_overhead()` is charged for every tracked key of every queue and
//! tier: a demoted key's values move to PMEM, but its `entries` slot and its
//! list node stay in DRAM regardless.
//!
//! Two consequences, both inherited:
//!
//! * **Admission never demotes anyone by itself.** `insert`ing a brand-new
//!   key moves no capacity (the reservation is fixed), so it does not
//!   re-settle the fast tier.
//! * **`resize()` MUST re-settle, twice.** `a1_in_capacity` and
//!   `a1_out_capacity` are both derived from `max_size`, and the first of
//!   them feeds `effective_am_fast_capacity`. So `resize` re-runs
//!   [`Self::settle_a1_in`] (re-establishing the `a1_in` invariant eagerly,
//!   unlike `TwoQStack`, which fixes it lazily at the next new-key insert)
//!   and then [`Self::settle_fast_tier`]. Skipping either leaves `am`'s
//!   budget distorted until some unrelated access happens to notice.
//!
//! ### Sizing constraint: `k_in * max_size` must stay below `fast_tier_size`
//!
//! `a1_in_capacity` scales with `max_size` while the budget it is carved out
//! of is `fast_tier_size` — typically a small fraction of `max_size`. At the
//! 20%-of-`max_size` default fast tier, a conventional `k_in = 0.25`
//! saturates `effective_am_fast_capacity()` to **zero**: `am` gets no fast
//! segment, DRAM holds nothing but unproven objects, and the design inverts
//! its own purpose. That is legitimate accounting for that configuration
//! rather than an error (see
//! `am_fast_segment_survives_default_fast_capacity`), and it cannot be
//! validated at construction because `fast_capacity` arrives later via
//! `resize_fast_tier` — which is why that method logs a warning when the
//! effective capacity saturates. Pick `k_in` against `fast_tier_size`, not
//! against `max_size`.
//!
//! ## Migration ordering
//!
//! Within a single `insert`/`update`, `settle_a1_in`'s `(k, Slow)` demotions
//! and `settle_fast_tier`'s `(k, Slow)` demotions are always pushed **before**
//! any `(k, Tier::Fast)` promotion: `apply_tier_migrations` applies in push
//! order, and a promotion pushed first would allocate DRAM before the
//! demotion released it. Both promotion pushes are additionally guarded on
//! the key still being `Fast`, since a tight enough budget can demote it
//! straight back out inside the same `settle_fast_tier` call — in which case
//! that call has already pushed the correct final `(key, Tier::Slow)`.

#[cfg(not(feature = "eviction_stacks_pmem"))]
use std::collections::HashMap;
#[cfg(feature = "eviction_stacks_pmem")]
use hashbrown::HashMap;

#[cfg(not(feature = "eviction_stacks_pmem"))]
use kwik::collections::HashList;
#[cfg(feature = "eviction_stacks_pmem")]
use super::pmem_collections::PmemHashList;

// Eviction-stack metadata is allocated through the same crate-wide `Hybrid`
// alias (`SlowObjects`, node-1 jemalloc arenas) that `BufferPMEM`/other PMEM
// features already use.
#[cfg(feature = "eviction_stacks_pmem")]
use crate::Hybrid;

use crate::{
	CacheSize,
	HashedKey,
	NoHasher,
	policy::PaperPolicy,
	object::ObjectSize,
	worker::policy::policy_stack::{PolicyStack, Tier, narrow_resident, watermarks},
};

/// Which of the three live queues a key currently belongs to.
///
/// The tag doubles as the tier for two of the three: `A1In` is Fast and
/// `A1Out` is Slow *structurally*, so neither stores a tier. Only `Am` is
/// segmented and therefore carries one — see [`TwoQEntry::tier`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Queue {
	A1In,
	A1Out,
	Am,
}

/// Combined per-key bookkeeping: which queue, which tier, and the size.
///
/// Invariant: `tier.is_some()` iff `queue == Queue::Am`. A key is resident in
/// exactly one of the three lists, which is what keeps the six byte counters
/// and three object counters honest.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct TwoQEntry {queue: Queue,
	tier: Option<Tier>,
	/// Part of `size` that stays in DRAM in either tier; see `migrating`.
	dram_resident: u8,

	size: ObjectSize,
}

impl TwoQEntry {
/// The bytes that actually move between tiers when this object migrates.
	///
	/// `size` is `base_size`, which also counts the DRAM-resident remainder --
	/// the key and expiry field (inline in the object map) plus the `Expiries`
	/// entry when a TTL is set. `Object::set_data` replaces the value buffer
	/// alone, so none of that moves, and the key and expiry are already inside
	/// `shared_overhead`. Charging them to the tier counters double-counted
	/// every fast-tier object and made demotion appear to free DRAM it did not.
	#[inline]
	fn migrating(&self) -> CacheSize {
		(self.size as CacheSize).saturating_sub(self.dram_resident as CacheSize)
	}
}

/// `dram_resident` was meant to occupy padding the entry already had.
/// If this ever fails, the field is costing 4 more bytes on *every* tracked
/// object in both tiers, which defeats the point of storing it per entry.
const _: () = assert!(
	std::mem::size_of::<TwoQEntry>() == 8,
	"TwoQEntry grew past 8 bytes",
);


#[cfg(not(feature = "eviction_stacks_pmem"))]
type QueueList = HashList<HashedKey, NoHasher>;
#[cfg(feature = "eviction_stacks_pmem")]
type QueueList = PmemHashList<HashedKey, NoHasher>;

#[cfg(not(feature = "eviction_stacks_pmem"))]
type EntryMap = HashMap<HashedKey, TwoQEntry, NoHasher>;
#[cfg(feature = "eviction_stacks_pmem")]
type EntryMap = HashMap<HashedKey, TwoQEntry, NoHasher, Hybrid>;

pub struct TwoQFullFastAdmissionHybridStack {
	/// Probation FIFO, front = newest. FAST tier, structurally.
	a1_in: QueueList,

	/// Overflow FIFO, front = newest. SLOW tier, structurally. Holds REAL
	/// RESIDENT OBJECTS, not ghosts — see the module doc.
	a1_out: QueueList,

	/// Main LRU, front = MRU. Tier-segmented at [`Self::am_boundary`].
	am: QueueList,

	entries: EntryMap,

	k_in: f64,
	k_out: f64,

	/// `k_in * max_size`. A reservation carved out of `fast_capacity` (these
	/// bytes are DRAM) — see [`Self::effective_am_fast_capacity`].
	a1_in_capacity: CacheSize,
	a1_in_used: CacheSize,

	/// `k_out * max_size`. A PMEM budget, carved out of nothing; overrunning
	/// it is what [`Self::needs_capacity_eviction`] reports. Live for the
	/// first time in this codebase — `TwoQStack` writes its equivalent and
	/// never reads it.
	a1_out_capacity: CacheSize,
	a1_out_used: CacheSize,

	/// Total fast-tier (DRAM) budget, covering BOTH `a1_in` and `am`'s fast
	/// segment. Set at runtime by `resize_fast_tier`.
	fast_capacity: CacheSize,

	/// Approximate per-object DRAM cost of the shared structures (object
	/// hashtable + eviction stacks) that hold an entry for every tracked key
	/// of every queue and tier. Reserved out of `am`'s share of
	/// `fast_capacity`. `0` unless set via [`Self::with_shared_overhead`], so
	/// unit tests exercising the pure value-budget behaviour are unaffected.
	shared_overhead: CacheSize,

	/// Bytes held by `am` keys tagged `Tier::Fast`. Does NOT include
	/// `a1_in_used`, even though both are physically DRAM — see
	/// [`Self::fast_bytes_used`], which sums them for reporting.
	am_fast_used: CacheSize,

	/// Bytes held by `am` keys tagged `Tier::Slow`. Does NOT include
	/// `a1_out_used`; [`Self::slow_bytes_used`] sums them.
	am_slow_used: CacheSize,

	/// Number of keys currently in `am` (Fast or Slow).
	am_count: usize,

	/// Number of keys currently tagged `Tier::Fast` within `am`.
	am_fast_count: usize,

	/// The least-recently-used key currently tagged `Tier::Fast` within `am`
	/// — i.e. the next demotion candidate. `None` iff no key in `am` is
	/// currently Fast. Fast keys are a contiguous prefix of `am` from the
	/// front, so the boundary walks backwards via `am.before`.
	am_boundary: Option<HashedKey>,

	/// (key, new tier) pairs recorded since the last `drain_tier_migrations`.
	migrations: Vec<(HashedKey, Tier)>,
}

impl TwoQFullFastAdmissionHybridStack {
	#[cfg(not(feature = "eviction_stacks_pmem"))]
	fn new_collections() -> (QueueList, QueueList, QueueList, EntryMap) {
		(
			HashList::default(),
			HashList::default(),
			HashList::default(),
			HashMap::default(),
		)
	}

	#[cfg(feature = "eviction_stacks_pmem")]
	fn new_collections() -> (QueueList, QueueList, QueueList, EntryMap) {
		(
			PmemHashList::with_hasher(NoHasher::default()),
			PmemHashList::with_hasher(NoHasher::default()),
			PmemHashList::with_hasher(NoHasher::default()),
			HashMap::with_hasher_in(NoHasher::default(), Hybrid),
		)
	}

	pub fn new(
		k_in: f64,
		k_out: f64,
		max_size: CacheSize,
		fast_capacity: CacheSize,
	) -> Self {
		let (a1_in, a1_out, am, entries) = Self::new_collections();

		TwoQFullFastAdmissionHybridStack {
			a1_in,
			a1_out,
			am,

			entries,

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

	/// Sets the approximate per-object shared-structure DRAM overhead (object
	/// hashtable + eviction stacks) reserved out of the fast-tier budget. See
	/// `crate::object::overhead::get_hybrid_dram_shared_overhead`.
	/// Builder-style so `init_policy_stack` can wire it in without disturbing
	/// `new`'s signature (unit tests keep the default `0`).
	pub fn with_shared_overhead(mut self, overhead: CacheSize) -> Self {
		self.shared_overhead = overhead;
		self
	}

	/// Total DRAM currently reserved for shared per-object metadata across
	/// every tracked key (`entries.len() × shared_overhead`), whichever queue
	/// and whichever tier each of them is in — the hashtable slot and the list
	/// node are DRAM even for an `a1_out`-resident or demoted key.
	///
	/// A single term, unlike `TwoQGhostHybridStack`'s two: there is no ghost
	/// list here, and `a1_out` members are ordinary tracked keys already
	/// counted by `entries.len()`.
	fn reserved_overhead(&self) -> CacheSize {
		self.entries.len() as CacheSize * self.shared_overhead
	}

	/// How much of `fast_capacity` `am`'s fast segment may use, after
	/// `a1_in`'s fixed reservation and the shared per-object metadata
	/// reservation are both carved out.
	///
	/// Saturating rather than panicking when the two carve-outs meet or
	/// exceed `fast_capacity`: that is a legitimate (if degenerate)
	/// configuration — see the module doc's sizing constraint — and it means
	/// "`am` gets no fast segment", not an error.
	///
	/// `a1_out_capacity` is deliberately absent: those bytes are PMEM.
	///
	/// The watermarks in [`Self::settle_fast_tier`] are applied *on top of*
	/// this value, never in place of it.
	fn effective_am_fast_capacity(&self) -> CacheSize {
		self.fast_capacity
			.saturating_sub(self.a1_in_capacity)
			.saturating_sub(self.reserved_overhead())
	}

	/// Returns which tier the given (currently tracked) key is in, or `None`
	/// if the key isn't tracked. Exposed for tests/diagnostics.
	///
	/// Two of the three queues answer structurally: `a1_in` is DRAM by the
	/// fast-admission rule, `a1_out` is PMEM by the demotion rule. Only `am`
	/// stores a tier.
	pub fn tier_of(&self, key: HashedKey) -> Option<Tier> {
		let entry = self.entries.get(&key)?;

		match entry.queue {
			Queue::A1In => Some(Tier::Fast),
			Queue::A1Out => Some(Tier::Slow),
			Queue::Am => entry.tier,
		}
	}

	/// Records a size change for an already-tracked key without altering its
	/// queue/tier, adjusting whichever one of the four byte counters currently
	/// owns it.
	///
	/// No re-settle is needed afterwards for an `a1_in`-resident key that
	/// grew, unlike `TwoQFastAdmissionHybridStack::resize_key`: the
	/// reservation `am`'s budget is computed against is the *fixed*
	/// `a1_in_capacity`, so live `a1_in_used` does not move it.
	/// `new_resident` refreshes the entry's DRAM-resident remainder: a re-set
	/// can add or drop a TTL, which changes it by the `Expiries` entry's cost.
	/// Without this the entry keeps its old remainder and every later
	/// migration moves the wrong number of bytes.
	fn resize_key(&mut self, key: HashedKey, new_size: ObjectSize, new_resident: u8) {
		let Some(entry) = self.entries.get_mut(&key) else { return };

		let old_migrating = entry.migrating();
		entry.size = new_size;
		entry.dram_resident = new_resident;
		let delta = entry.migrating() as i64 - old_migrating as i64;

		match (entry.queue, entry.tier) {
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
	/// deliberately empty — see the module doc.
	fn touch(&mut self, key: HashedKey) {
		match self.entries.get(&key).map(|entry| entry.queue) {
			// A hit on a probation key does NOTHING: no list move, no tier
			// change, no migration, no counter. Faithful to `TwoQStack`,
			// where `a1_out.remove` misses and `am.move_front` is a silent
			// no-op; and it is already Fast, so there is nothing to migrate.
			Some(Queue::A1In) => {},

			Some(Queue::A1Out) => self.promote_from_a1_out(key),
			Some(Queue::Am) => self.touch_am(key),

			None => {},
		}
	}

	/// `TwoQStack::restructure_to_fit`, with the transition it performs now
	/// being a real DRAM->PMEM tier migration.
	///
	/// Drains the `a1_in` tail into `a1_out`'s head until `incoming_size`
	/// fits. This is a **demotion**, never an eviction: the key stays
	/// resident, stays in `entries`, and stays visible to `contains()`. The
	/// resulting `a1_out` overrun (if any) is reported by
	/// [`Self::needs_capacity_eviction`] and drained by `apply_evictions`
	/// through the real `evict_one()` + `erase()` path.
	///
	/// The `else break` mirrors `restructure_to_fit`'s `else return`: an
	/// object larger than the whole of `a1_in` empties the queue and is then
	/// admitted anyway rather than looping forever.
	fn settle_a1_in(&mut self, incoming_size: ObjectSize) {
		let incoming = incoming_size as CacheSize;

		while self.a1_in_used + incoming > self.a1_in_capacity {
			let Some(key) = self.a1_in.pop_back() else { break };
			let Some(size) = self.entries.get(&key).map(|entry| entry.migrating()) else { continue };

			if let Some(entry) = self.entries.get_mut(&key) {
				entry.queue = Queue::A1Out;
				entry.tier = None;
			}

			self.a1_in_used = self.a1_in_used.saturating_sub(size);

			self.a1_out.push_front(key);
			self.a1_out_used += size;

			self.migrations.push((key, Tier::Slow));
		}
	}

	/// The 2Q promotion: an `a1_out` hit moves the live key to `am`'s MRU end
	/// at `Tier::Fast`. Equivalent to `TwoQStack::update`'s
	/// `a1_out.remove -> am.insert`.
	///
	/// Emits a genuine `(key, Tier::Fast)` migration — unlike
	/// `TwoQFastAdmissionHybridStack::promote_from_fifo`, which is a
	/// Fast->Fast bookkeeping move. Here the bytes really do live in PMEM
	/// beforehand, because `a1_out` is the slow tier.
	fn promote_from_a1_out(&mut self, key: HashedKey) {
		let Some((size, dram_resident)) = self.entries
			.get(&key)
			.map(|entry| (entry.size, entry.dram_resident))
		else { return };
		// Tier arithmetic moves only what migrates; `size` still rebuilds the entry.
		let size_bytes = (size as CacheSize).saturating_sub(dram_resident as CacheSize);

		self.a1_out.remove(&key);
		self.a1_out_used = self.a1_out_used.saturating_sub(size_bytes);

		self.am.push_front(key);
		self.entries.insert(key, TwoQEntry { dram_resident, queue: Queue::Am, tier: Some(Tier::Fast), size });

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
		if self.entries.get(&key).and_then(|entry| entry.tier) == Some(Tier::Fast) {
			self.migrations.push((key, Tier::Fast));
		}
	}

	/// An `am` hit: the 2Q LRU reorder (`am.move_front`, unchanged from
	/// `TwoQStack`) composed additively with the tier promotion, which acts on
	/// orthogonal state — `am` is one physical list and the tier split is a
	/// boundary marker over it. Identical in shape to
	/// `TwoQHybridStack::touch_main_fast` / `LruHybridStack::touch_fast_key`.
	fn touch_am(&mut self, key: HashedKey) {
		let previous_tier = self.entries.get(&key).and_then(|entry| entry.tier);

		let already_at_front = self.am.front() == Some(&key);
		let is_boundary = self.am_boundary == Some(key);

		let new_boundary_if_moved = if is_boundary && !already_at_front {
			self.am.before(&key).copied()
		} else {
			None
		};

		self.am.move_front(&key);

		if is_boundary && !already_at_front {
			self.am_boundary = new_boundary_if_moved;
		}

		let mut promoted = false;

		if previous_tier != Some(Tier::Fast) {
			if previous_tier == Some(Tier::Slow) {
				let size = self.entries.get(&key).map(|entry| entry.migrating()).unwrap_or(0);

				self.am_slow_used = self.am_slow_used.saturating_sub(size);
				self.am_fast_used += size;
				self.am_fast_count += 1;

				promoted = true;
			}

			if let Some(entry) = self.entries.get_mut(&key) {
				entry.tier = Some(Tier::Fast);
			}

			if self.am_boundary.is_none() {
				self.am_boundary = Some(key);
			}
		}

		self.settle_fast_tier();

		// Same ordering and same guard as `promote_from_a1_out` above.
		if promoted && self.entries.get(&key).and_then(|entry| entry.tier) == Some(Tier::Fast) {
			self.migrations.push((key, Tier::Fast));
		}
	}

	/// Demotes the least-recently-used fast key(s) within `am` once
	/// `am_fast_used` exceeds the HIGH watermark of
	/// [`Self::effective_am_fast_capacity`], then drains down to the LOW
	/// watermark rather than merely back under the ceiling.
	///
	/// The ONLY demotion mechanism for `am`, and it never evicts: the DRAM
	/// budget sheds bytes by moving them to PMEM, never by dropping user data.
	/// Terminal eviction stays governed solely by `max_size` /
	/// `a1_out_capacity`, via [`Self::evict_one`].
	///
	/// Composition order is reservations first, watermarks on the remainder;
	/// see [`watermarks`] for why draining below the ceiling produces larger,
	/// less frequent migration batches.
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

			let size = self.entries.get(&demote_key).map(|entry| entry.migrating()).unwrap_or(0);
			let new_boundary = self.am.before(&demote_key).copied();

			if let Some(entry) = self.entries.get_mut(&demote_key) {
				entry.tier = Some(Tier::Slow);
			}

			self.am_fast_used = self.am_fast_used.saturating_sub(size);
			self.am_fast_count = self.am_fast_count.saturating_sub(1);
			self.am_slow_used += size;
			self.am_boundary = new_boundary;

			self.migrations.push((demote_key, Tier::Slow));
		}
	}

	/// Pops and fully removes `a1_out`'s tail from this stack's own
	/// bookkeeping. The FIRST eviction victim, per `TwoQStack::evict_one`.
	fn evict_a1_out_tail(&mut self) -> Option<HashedKey> {
		let key = self.a1_out.pop_back()?;
		let size = self.entries.remove(&key).map(|entry| entry.migrating()).unwrap_or(0);

		self.a1_out_used = self.a1_out_used.saturating_sub(size);

		Some(key)
	}

	/// Pops and fully removes `a1_in`'s tail. Reached only once `a1_out` is
	/// empty — under normal operation `a1_in`'s tail is *demoted* into
	/// `a1_out` by [`Self::settle_a1_in`] long before it can be evicted here.
	fn evict_a1_in_tail(&mut self) -> Option<HashedKey> {
		let key = self.a1_in.pop_back()?;
		let size = self.entries.remove(&key).map(|entry| entry.migrating()).unwrap_or(0);

		self.a1_in_used = self.a1_in_used.saturating_sub(size);

		Some(key)
	}

	/// Pops and fully removes `am`'s LRU tail. The last resort, per
	/// `TwoQStack::evict_one`.
	fn evict_am_tail(&mut self) -> Option<HashedKey> {
		let key = self.am.pop_back()?;
		let removed = self.entries.remove(&key);
		let size = removed.map(|entry| entry.migrating()).unwrap_or(0);
		let tier = removed.and_then(|entry| entry.tier);

		self.am_count = self.am_count.saturating_sub(1);

		match tier {
			Some(Tier::Fast) => {
				self.am_fast_used = self.am_fast_used.saturating_sub(size);
				self.am_fast_count = self.am_fast_count.saturating_sub(1);

				// The tail of `am` can only be Fast-tagged if every tracked
				// `am` key is still Fast (fast keys are a contiguous prefix),
				// in which case the boundary equalled this key. Re-point it at
				// the new tail, unless that tail is Slow or `am` is now empty.
				if self.am_boundary == Some(key) {
					self.am_boundary = match self.am.back().copied() {
						Some(back) if self.entries.get(&back).and_then(|entry| entry.tier) == Some(Tier::Fast) => Some(back),
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

impl PolicyStack for TwoQFullFastAdmissionHybridStack {
	fn is_policy(&self, policy: &PaperPolicy) -> bool {
		matches!(
			policy,
			PaperPolicy::TwoQFullFastAdmissionHybrid(k_in, k_out)
				if *k_in == self.k_in && *k_out == self.k_out
		)
	}

	fn len(&self) -> usize {
		self.entries.len()
	}

	/// `a1_out` members count: they are resident objects, not ghosts. See the
	/// module doc — excluding them would under-report to the worker's capacity
	/// logic and desynchronise the object map.
	fn contains(&self, key: HashedKey) -> bool {
		self.entries.contains_key(&key)
	}

	fn insert(&mut self, key: HashedKey, size: ObjectSize) {
		self.insert_resident(key, size, 0);
	}

	fn insert_resident(&mut self, key: HashedKey, size: ObjectSize, dram_resident: ObjectSize) {
		let dram_resident = narrow_resident(dram_resident);
		if self.entries.contains_key(&key) {
			// Existing key: track any size change, then treat as an access —
			// exactly what `TwoQStack::insert` does (three `Stack::update`
			// calls, then falling through to `update`).
			self.resize_key(key, size, dram_resident);
			self.touch(key);
			return;
		}

		// Brand-new key: `a1_in` first, which is FAST here — a cheap DRAM
		// write on the calling thread instead of a synchronous PMEM alloc.
		self.settle_a1_in(size);

		self.a1_in.push_front(key);
		self.entries.insert(key, TwoQEntry { dram_resident, queue: Queue::A1In, tier: None, size });
		self.a1_in_used += (size as CacheSize).saturating_sub(dram_resident as CacheSize);

		// Deliberately does NOT re-settle the fast tier: the reservation
		// carved out of `fast_capacity` is the fixed `a1_in_capacity`, not
		// live `a1_in_used`, so admission cannot move `am`'s budget. And it
		// deliberately does not evict: a `PolicyStack` never self-evicts (see
		// the module doc); `settle_a1_in` demoted instead, and any resulting
		// `a1_out` overrun is reported via `needs_capacity_eviction`.
	}

	fn update(&mut self, key: HashedKey) {
		if self.entries.contains_key(&key) {
			self.touch(key);
		}
	}

	fn remove(&mut self, key: HashedKey) {
		let Some(entry) = self.entries.remove(&key) else { return };
		let size = entry.migrating();

		match entry.queue {
			Queue::A1In => {
				self.a1_in.remove(&key);
				self.a1_in_used = self.a1_in_used.saturating_sub(size);
			},

			Queue::A1Out => {
				self.a1_out.remove(&key);
				self.a1_out_used = self.a1_out_used.saturating_sub(size);
			},

			Queue::Am => {
				let new_boundary_if_needed = if entry.tier == Some(Tier::Fast) && self.am_boundary == Some(key) {
					self.am.before(&key).copied()
				} else {
					None
				};

				self.am.remove(&key);
				self.am_count = self.am_count.saturating_sub(1);

				match entry.tier {
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
	/// `TwoQStack::resize` only writes the two `max_size` fields and lets the
	/// next new-key insert fix `a1_in` lazily. That is harmless there and
	/// harmful here: `a1_in_capacity` feeds
	/// [`Self::effective_am_fast_capacity`], so a stale one distorts `am`'s
	/// DRAM budget until some unrelated access happens to notice. And
	/// `TwoQHybridStack::resize` need not re-settle the fast tier at all,
	/// because its FIFO queue is PMEM and competes for nothing.
	fn resize(&mut self, max_size: CacheSize) {
		self.a1_in_capacity = (self.k_in * max_size as f64) as CacheSize;
		self.a1_out_capacity = (self.k_out * max_size as f64) as CacheSize;

		// Drain `a1_in` down to its new budget (demoting, never evicting)...
		self.settle_a1_in(0);

		// ...then re-settle `am`, whose effective fast budget just moved with
		// `a1_in_capacity`.
		self.settle_fast_tier();

		// A shrink may also push `a1_out_used` over the new, smaller
		// `a1_out_capacity`; `needs_capacity_eviction` reports it and
		// `apply_evictions` drains it.
	}

	fn clear(&mut self) {
		self.a1_in.clear();
		self.a1_out.clear();
		self.am.clear();
		self.entries.clear();

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

	/// `TwoQStack::evict_one`, verbatim: `a1_out` tail, then `a1_in` tail,
	/// then `am`'s LRU tail. Emits no migrations — an evicted object is gone,
	/// not moved.
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
		// which the module doc's sizing constraint can be checked at all.
		if size > 0 && self.effective_am_fast_capacity() == 0 {
			log::warn!(
				"2q-full-fast-admission-hybrid: a1_in's DRAM reservation ({} bytes) plus per-key metadata ({} bytes) meets or exceeds the fast-tier budget ({size} bytes); `am` gets no fast segment and every promotion will demote straight back out. Lower k_in or raise fast_tier_size.",
				self.a1_in_capacity,
				self.reserved_overhead(),
			);
		}

		self.settle_fast_tier();
	}

	fn drain_tier_migrations(&mut self) -> Vec<(HashedKey, Tier)> {
		std::mem::take(&mut self.migrations)
	}

	/// Both DRAM-resident structures, summed: the probation FIFO plus `am`'s
	/// fast segment. `a1_out` is PMEM and is excluded.
	fn fast_bytes_used(&self) -> CacheSize {
		self.a1_in_used + self.am_fast_used
	}

	/// `a1_out` plus `am`'s slow segment. `a1_out` counts here because it
	/// holds real resident objects.
	fn slow_bytes_used(&self) -> CacheSize {
		self.a1_out_used + self.am_slow_used
	}

	fn fast_object_count(&self) -> usize {
		self.a1_in.len() + self.am_fast_count
	}

	fn slow_object_count(&self) -> usize {
		self.a1_out.len() + (self.am_count - self.am_fast_count)
	}

	/// `a1_out` ONLY. `a1_in` overflow is a demotion handled internally by
	/// [`Self::settle_a1_in`]; reporting it here would evict where the
	/// algorithm demotes. Because [`Self::evict_one`] pops `a1_out` first,
	/// `apply_evictions`'s loop is guaranteed to make progress on exactly the
	/// over-budget queue.
	fn needs_capacity_eviction(&self) -> bool {
		self.a1_out_used > self.a1_out_capacity
	}
}

#[cfg(test)]
mod tests {
	use super::*;

	fn drain(stack: &mut TwoQFullFastAdmissionHybridStack) -> Vec<(HashedKey, Tier)> {
		stack.drain_tier_migrations()
	}

	// ── the fidelity proof ────────────────────────────────────────────────

	/// **THE** test for this variant. Replays `two_q_stack.rs`'s own
	/// `eviction_order_is_correct` trace — `TwoQStack::new(0.25, 0.5, 4)`,
	/// `insert(k, 1)` over `[0, 1, 0, 2, 1, 3, 0, 4, 2, 5, 0]`, expecting
	/// eviction order `[3, 4, 5, 1, 2, 0]` — and asserts this stack produces
	/// the identical sequence.
	///
	/// If this fails, the variant is not the full three-queue 2Q and the whole
	/// point of it is gone. It is deliberately run across several
	/// `fast_capacity` values: tier bookkeeping is layered on *top of* the
	/// queue logic and must never perturb queue order, however hard the DRAM
	/// budget squeezes.
	#[test]
	fn eviction_order_matches_plain_two_q_stack() {
		for fast_capacity in [1, 2, 4, 64, 1_024] {
			let mut stack = TwoQFullFastAdmissionHybridStack::new(0.25, 0.5, 4, fast_capacity);

			for access in [0, 1, 0, 2, 1, 3, 0, 4, 2, 5, 0] {
				stack.insert(access, 1);
			}

			for eviction in [3, 4, 5, 1, 2, 0] {
				assert_eq!(
					stack.evict_one(),
					Some(eviction),
					"fast_capacity {fast_capacity}",
				);
			}

			assert_eq!(stack.evict_one(), None, "fast_capacity {fast_capacity}");
			assert_eq!(stack.len(), 0, "fast_capacity {fast_capacity}");
		}
	}

	// ── admission ─────────────────────────────────────────────────────────

	#[test]
	fn admission_lands_in_a1_in_at_fast_tier() {
		let mut stack = TwoQFullFastAdmissionHybridStack::new(0.5, 0.25, 1_000, 1_000);

		stack.insert(1, 10);
		stack.insert(2, 30);

		assert_eq!(stack.tier_of(1), Some(Tier::Fast));
		assert_eq!(stack.tier_of(2), Some(Tier::Fast));

		assert_eq!(stack.a1_in.len(), 2);
		assert_eq!(stack.a1_out.len(), 0);
		assert_eq!(stack.am.len(), 0);

		assert_eq!(stack.fast_bytes_used(), 40);
		assert_eq!(stack.slow_bytes_used(), 0);
		assert_eq!(stack.fast_object_count(), 2);
		assert_eq!(stack.slow_object_count(), 0);

		// Admission is already physically correct, so nothing to migrate.
		assert!(drain(&mut stack).is_empty());
	}

	// ── the fidelity point: an a1_in hit does nothing at all ──────────────

	/// A hit on a probation key must not reorder, not re-tier, not migrate and
	/// not touch a counter. This is exactly where `TwoQHybridStack` diverges
	/// from `TwoQStack` (it promotes), and the reason this variant exists.
	#[test]
	fn a1_in_hit_is_a_complete_no_op() {
		let mut stack = TwoQFullFastAdmissionHybridStack::new(0.5, 0.25, 1_000, 1_000);

		for key in 1..=3 {
			stack.insert(key, 10);
		}

		drain(&mut stack);

		let a1_in_used = stack.a1_in_used;
		let a1_out_used = stack.a1_out_used;
		let am_fast_used = stack.am_fast_used;
		let am_slow_used = stack.am_slow_used;
		let am_count = stack.am_count;
		let am_fast_count = stack.am_fast_count;
		let am_boundary = stack.am_boundary;
		let len = stack.len();
		let front = stack.a1_in.front().copied();
		let back = stack.a1_in.back().copied();

		// Hit the middle of the FIFO — the position a reorder would be most
		// visible at.
		stack.update(2);

		assert_eq!(stack.a1_in_used, a1_in_used);
		assert_eq!(stack.a1_out_used, a1_out_used);
		assert_eq!(stack.am_fast_used, am_fast_used);
		assert_eq!(stack.am_slow_used, am_slow_used);
		assert_eq!(stack.am_count, am_count);
		assert_eq!(stack.am_fast_count, am_fast_count);
		assert_eq!(stack.am_boundary, am_boundary);
		assert_eq!(stack.len(), len);

		assert_eq!(stack.a1_in.front().copied(), front);
		assert_eq!(stack.a1_in.back().copied(), back);
		assert_eq!(stack.am.len(), 0, "an a1_in hit must NOT promote into am");
		assert_eq!(stack.tier_of(2), Some(Tier::Fast));

		assert!(drain(&mut stack).is_empty(), "an a1_in hit must emit no migration");

		// And the FIFO order itself is untouched: still oldest-first.
		assert_eq!(stack.evict_one(), Some(1));
		assert_eq!(stack.evict_one(), Some(2));
		assert_eq!(stack.evict_one(), Some(3));
	}

	// ── a1_in overflow: demotion, not eviction ────────────────────────────

	#[test]
	fn a1_in_overflow_demotes_to_a1_out_not_evicts() {
		// a1_in_capacity = 20, a1_out_capacity = 500.
		let mut stack = TwoQFullFastAdmissionHybridStack::new(0.02, 0.5, 1_000, 1_000);

		stack.insert(1, 10);
		stack.insert(2, 10);

		assert!(drain(&mut stack).is_empty(), "20 bytes still fits in a 20-byte a1_in");

		let len_before = stack.len();

		stack.insert(3, 10);

		// One demotion, of the FIFO tail.
		assert_eq!(drain(&mut stack), vec![(1, Tier::Slow)]);

		// The key is still in the cache — it moved tier, it did not leave.
		assert!(stack.contains(1));
		assert_eq!(stack.len(), len_before + 1);
		assert_eq!(stack.tier_of(1), Some(Tier::Slow));

		assert_eq!(stack.a1_in.len(), 2);
		assert_eq!(stack.a1_out.len(), 1);
		assert_eq!(stack.a1_in_used, 20);
		assert_eq!(stack.a1_out_used, 10);
		assert_eq!(stack.fast_bytes_used(), 20);
		assert_eq!(stack.slow_bytes_used(), 10);
	}

	/// R8: `a1_out` holds resident objects, so it must count toward
	/// `contains()`, `len()` and the slow-tier gauges. Excluding it (the
	/// natural mistake when porting from the ghost variant, where exclusion is
	/// correct) would desynchronise the object map.
	#[test]
	fn a1_out_members_are_reported_resident() {
		let mut stack = TwoQFullFastAdmissionHybridStack::new(0.02, 0.5, 1_000, 1_000);

		stack.insert(1, 10);
		stack.insert(2, 10);

		let len_before = stack.len();

		stack.insert(3, 10);
		stack.remove(3);

		assert_eq!(stack.len(), len_before, "the demoted key is still tracked");
		assert!(stack.contains(1));
		assert_eq!(stack.slow_object_count(), 1);
		assert_eq!(stack.slow_bytes_used(), 10);
	}

	// ── a1_out hit: the 2Q promotion ──────────────────────────────────────

	#[test]
	fn a1_out_hit_promotes_to_am_at_fast() {
		let mut stack = TwoQFullFastAdmissionHybridStack::new(0.02, 0.5, 1_000, 1_000);

		stack.insert(1, 10);
		stack.insert(2, 10);
		stack.insert(3, 10);

		assert_eq!(stack.tier_of(1), Some(Tier::Slow));

		drain(&mut stack);

		stack.update(1);

		// A real PMEM -> DRAM move, so a real migration.
		assert_eq!(drain(&mut stack), vec![(1, Tier::Fast)]);

		assert_eq!(stack.tier_of(1), Some(Tier::Fast));
		assert_eq!(stack.am.front().copied(), Some(1), "promotion lands at am's MRU end");
		assert_eq!(stack.a1_out.len(), 0);
		assert_eq!(stack.am.len(), 1);
		assert_eq!(stack.am_count, 1);
		assert_eq!(stack.am_fast_count, 1);
		assert_eq!(stack.am_fast_used, 10);
		assert_eq!(stack.am_boundary, Some(1));
		assert_eq!(stack.a1_out_used, 0);
	}

	// ── eviction order and priority ───────────────────────────────────────

	#[test]
	fn evict_one_drains_a1_out_then_a1_in_then_am() {
		// a1_in_capacity = 20, so every third 10-byte insert demotes one key.
		let mut stack = TwoQFullFastAdmissionHybridStack::new(0.02, 0.5, 1_000, 1_000);

		stack.insert(1, 10);
		stack.insert(2, 10);
		stack.insert(3, 10); // demotes 1 -> a1_out
		stack.update(1);     // promotes 1 -> am
		stack.insert(4, 10); // demotes 2 -> a1_out

		drain(&mut stack);

		// a1_in = [4, 3], a1_out = [2], am = [1].
		assert_eq!(stack.a1_in.len(), 2);
		assert_eq!(stack.a1_out.len(), 1);
		assert_eq!(stack.am.len(), 1);

		assert_eq!(stack.evict_one(), Some(2), "a1_out tail goes first");
		assert_eq!(stack.evict_one(), Some(3), "then the a1_in tail");
		assert_eq!(stack.evict_one(), Some(4), "then the rest of a1_in");
		assert_eq!(stack.evict_one(), Some(1), "am's LRU tail is the last resort");
		assert_eq!(stack.evict_one(), None);

		assert_eq!(stack.len(), 0);
		assert_eq!(stack.a1_in_used, 0);
		assert_eq!(stack.a1_out_used, 0);
		assert_eq!(stack.am_fast_used, 0);
		assert_eq!(stack.am_slow_used, 0);
		assert_eq!(stack.am_boundary, None);

		// Eviction moves nothing, it drops things.
		assert!(drain(&mut stack).is_empty());
	}

	// ── capacity signalling ───────────────────────────────────────────────

	/// Only an `a1_out` overrun asks `apply_evictions` for help. An `a1_in`
	/// overrun must not — that is what `settle_a1_in` demotes for.
	#[test]
	fn needs_capacity_eviction_tracks_a1_out_only() {
		// a1_in_capacity = 20, a1_out_capacity = 50.
		let mut stack = TwoQFullFastAdmissionHybridStack::new(0.02, 0.05, 1_000, 1_000);

		// An object larger than the whole of a1_in overruns it outright (the
		// `else break` bail), with a1_out still empty.
		stack.insert(1, 100);

		assert!(stack.a1_in_used > stack.a1_in_capacity, "a1_in is over budget");
		assert_eq!(stack.a1_out_used, 0);
		assert!(
			!stack.needs_capacity_eviction(),
			"an a1_in overrun must be demoted, not evicted",
		);

		// The next admission drains that oversized key into a1_out, which now
		// IS over its own budget.
		stack.insert(2, 10);

		assert_eq!(stack.a1_out_used, 100);
		assert!(stack.a1_out_used > stack.a1_out_capacity);
		assert!(stack.needs_capacity_eviction());

		// And evict_one pops a1_out first, so the loop makes progress on
		// exactly the over-budget queue.
		assert_eq!(stack.evict_one(), Some(1));
		assert!(!stack.needs_capacity_eviction());
	}

	/// `restructure_to_fit`'s `else return`, ported: an object bigger than the
	/// whole of `a1_in` empties the queue and is admitted anyway rather than
	/// looping forever.
	#[test]
	fn oversized_object_bypasses_a1_in_budget() {
		// a1_in_capacity = 10.
		let mut stack = TwoQFullFastAdmissionHybridStack::new(0.01, 0.5, 1_000, 1_000);

		stack.insert(1, 500);

		assert!(stack.contains(1));
		assert_eq!(stack.tier_of(1), Some(Tier::Fast));
		assert_eq!(stack.a1_in_used, 500);
		assert_eq!(stack.a1_in.len(), 1);
		assert_eq!(stack.a1_out.len(), 0);
		assert!(drain(&mut stack).is_empty());
	}

	// ── migration ordering ────────────────────────────────────────────────

	/// `apply_tier_migrations` applies in push order, so every `Tier::Slow`
	/// demotion a promotion triggers has to be pushed before the
	/// `Tier::Fast` promotion itself — otherwise DRAM is allocated before it
	/// is freed.
	#[test]
	fn demotions_precede_promotion_in_one_batch() {
		// a1_in_capacity = 120; fast_capacity 220 => effective am budget 100.
		let mut stack = TwoQFullFastAdmissionHybridStack::new(0.12, 0.5, 1_000, 220);

		assert_eq!(stack.effective_am_fast_capacity(), 100);

		stack.insert(1, 60);
		stack.insert(2, 60);
		stack.insert(3, 60); // demotes 1 -> a1_out
		stack.update(1);     // promotes 1 -> am (60 bytes, still under budget)
		stack.insert(4, 60); // demotes 2 -> a1_out

		drain(&mut stack);

		assert_eq!(stack.tier_of(1), Some(Tier::Fast));

		// Promoting 2 puts 120 bytes into a 100-byte budget, so a demotion is
		// forced at any watermark pair.
		stack.update(2);

		let migrations = drain(&mut stack);

		assert!(
			migrations.iter().any(|(_, tier)| *tier == Tier::Slow),
			"an over-budget promotion must force a demotion: {migrations:?}",
		);

		let last_slow = migrations.iter().rposition(|(_, tier)| *tier == Tier::Slow);
		let first_fast = migrations.iter().position(|(_, tier)| *tier == Tier::Fast);

		if let (Some(last_slow), Some(first_fast)) = (last_slow, first_fast) {
			assert!(
				last_slow < first_fast,
				"every demotion must precede every promotion: {migrations:?}",
			);
		}

		// Total DRAM stays inside the configured fast tier.
		assert!(stack.fast_bytes_used() <= 220);
	}

	// ── resize ────────────────────────────────────────────────────────────

	/// R6: `resize` rescales both budgets and must re-establish the `a1_in`
	/// invariant immediately, with no intervening insert.
	#[test]
	fn resize_reestablishes_both_budgets_and_resettles() {
		let mut stack = TwoQFullFastAdmissionHybridStack::new(0.5, 0.5, 1_000, 1_000);

		stack.insert(1, 100);
		stack.insert(2, 100);
		stack.insert(3, 100);

		assert_eq!(stack.a1_in_used, 300);
		assert_eq!(stack.a1_out.len(), 0);

		drain(&mut stack);

		stack.resize(200);

		assert_eq!(stack.a1_in_capacity, 100);
		assert_eq!(stack.a1_out_capacity, 100);

		// Drained immediately, not lazily at the next insert.
		assert!(stack.a1_in_used <= stack.a1_in_capacity);
		assert_eq!(stack.a1_in_used, 100);
		assert_eq!(stack.a1_out_used, 200);

		// Demotions in FIFO-tail order, and nothing was evicted.
		assert_eq!(drain(&mut stack), vec![(1, Tier::Slow), (2, Tier::Slow)]);
		assert_eq!(stack.len(), 3);

		// The a1_out overrun is reported rather than acted on.
		assert!(stack.needs_capacity_eviction());
	}

	/// The other half of R6: `a1_in_capacity` feeds
	/// `effective_am_fast_capacity`, so growing `max_size` shrinks `am`'s DRAM
	/// budget and must demote right away.
	#[test]
	fn resize_growth_shrinks_the_am_fast_budget_and_demotes() {
		// a1_in_capacity = 100 out of a 200-byte fast tier => am gets 100.
		let mut stack = TwoQFullFastAdmissionHybridStack::new(0.5, 0.5, 200, 200);

		stack.insert(1, 50);
		stack.insert(2, 50);
		stack.insert(3, 50); // demotes 1 -> a1_out
		stack.update(1);     // promotes 1 -> am at Fast

		drain(&mut stack);

		assert_eq!(stack.effective_am_fast_capacity(), 100);
		assert_eq!(stack.tier_of(1), Some(Tier::Fast));

		// Doubling max_size doubles a1_in's reservation, which eats the whole
		// DRAM budget.
		stack.resize(400);

		assert_eq!(stack.effective_am_fast_capacity(), 0);
		assert_eq!(stack.tier_of(1), Some(Tier::Slow));
		assert_eq!(drain(&mut stack), vec![(1, Tier::Slow)]);
		assert_eq!(stack.am_fast_count, 0);
		assert_eq!(stack.am_boundary, None);
	}

	// ── degenerate configurations ─────────────────────────────────────────

	/// R1: at the default 20%-of-`max_size` fast tier, a conventional
	/// `k_in = 0.25` saturates `am`'s fast segment to zero. Legitimate, if
	/// degenerate — it must not panic, wedge, or thrash migrations.
	#[test]
	fn am_fast_segment_survives_default_fast_capacity() {
		let max_size: CacheSize = 1_000;
		let mut stack = TwoQFullFastAdmissionHybridStack::new(
			0.25,
			0.5,
			max_size,
			(max_size as f64 * 0.2) as CacheSize,
		);

		// a1_in_capacity 250 against a 200-byte fast tier.
		assert_eq!(stack.effective_am_fast_capacity(), 0);

		for key in 1..=20 {
			stack.insert(key, 30);
		}

		for key in 1..=20 {
			stack.update(key);
		}

		// Every key that reached am was demoted straight back out.
		assert_eq!(stack.am_fast_used, 0);
		assert_eq!(stack.am_fast_count, 0);
		assert_eq!(stack.am_boundary, None);

		// Nothing was evicted behind the caller's back.
		assert_eq!(stack.len(), 20);
		assert_eq!(stack.fast_bytes_used(), stack.a1_in_used);
	}

	/// R3: `k_out = 0` makes every drained object immediately
	/// eviction-eligible. A documented degenerate configuration, not a bug.
	#[test]
	fn k_out_zero_makes_every_demotion_eviction_eligible() {
		let mut stack = TwoQFullFastAdmissionHybridStack::new(0.02, 0.0, 1_000, 1_000);

		assert_eq!(stack.a1_out_capacity, 0);

		stack.insert(1, 10);
		stack.insert(2, 10);

		assert!(!stack.needs_capacity_eviction());

		stack.insert(3, 10); // demotes 1

		assert!(stack.needs_capacity_eviction());
		assert_eq!(stack.evict_one(), Some(1));
		assert!(!stack.needs_capacity_eviction());
	}

	// ── counter integrity ─────────────────────────────────────────────────

	/// R5 and R9. Three queues, six byte counters, keys migrating
	/// `a1_in -> a1_out -> am` and back via re-set: the single-owner invariant
	/// is the only thing keeping them honest, and `HashList::push_front`
	/// silently drops duplicates, so a violation corrupts accounting rather
	/// than panicking.
	///
	/// Deterministic xorshift rather than a real property-test dependency, so
	/// a failure is reproducible from the seed alone.
	#[test]
	fn mixed_operations_keep_every_counter_consistent() {
		let mut stack = TwoQFullFastAdmissionHybridStack::new(0.2, 0.3, 1_000, 400);
		let mut state: u64 = 0x2545_F491_4F6C_DD1D;

		for step in 0..4_000u32 {
			state ^= state << 13;
			state ^= state >> 7;
			state ^= state << 17;

			let key = state % 48;

			match (state >> 8) % 6 {
				0 | 1 => stack.insert(key, ((state >> 16) % 40 + 1) as ObjectSize),
				2 => stack.update(key),
				3 => stack.remove(key),

				4 => {
					let len_before = stack.len();
					let evicted = stack.evict_one();

					// R9: `apply_evictions` answers a `None` with a *random*
					// eviction, so `None` while non-empty is a real bug.
					assert_eq!(
						evicted.is_some(),
						len_before > 0,
						"step {step}: evict_one disagreed with len()",
					);
				},

				_ => stack.resize(200 + (state >> 24) % 2_000),
			}

			stack.drain_tier_migrations();

			let total: CacheSize = stack.entries.values().map(|entry| entry.size as CacheSize).sum();

			assert_eq!(
				stack.a1_in_used + stack.a1_out_used + stack.am_fast_used + stack.am_slow_used,
				total,
				"step {step}: byte counters drifted from entries",
			);

			assert_eq!(
				stack.fast_bytes_used() + stack.slow_bytes_used(),
				total,
				"step {step}: tier gauges drifted from entries",
			);

			assert_eq!(
				stack.a1_in.len() + stack.a1_out.len() + stack.am.len(),
				stack.entries.len(),
				"step {step}: a key is in zero or two queues",
			);

			assert_eq!(stack.am_count, stack.am.len(), "step {step}: am_count drifted");

			assert!(
				stack.am_fast_count <= stack.am_count,
				"step {step}: more fast keys than am keys",
			);

			assert_eq!(
				stack.am_boundary.is_none(),
				stack.am_fast_count == 0,
				"step {step}: boundary disagrees with am_fast_count",
			);

			assert_eq!(
				stack.fast_object_count() + stack.slow_object_count(),
				stack.len(),
				"step {step}: object-count gauges drifted",
			);
		}
	}

	// ── policy identity ───────────────────────────────────────────────────

	#[test]
	fn is_policy_matches_on_both_ratios() {
		let stack = TwoQFullFastAdmissionHybridStack::new(0.25, 0.5, 1_000, 200);

		assert!(stack.is_policy(&PaperPolicy::TwoQFullFastAdmissionHybrid(0.25, 0.5)));
		assert!(!stack.is_policy(&PaperPolicy::TwoQFullFastAdmissionHybrid(0.25, 0.4)));
		assert!(!stack.is_policy(&PaperPolicy::TwoQFullFastAdmissionHybrid(0.3, 0.5)));
		assert!(!stack.is_policy(&PaperPolicy::TwoQFastAdmissionHybrid(0.25)));
		assert!(!stack.is_policy(&PaperPolicy::TwoQ(0.25, 0.5)));
	}

	#[test]
	fn clear_resets_every_counter_but_keeps_the_budgets() {
		let mut stack = TwoQFullFastAdmissionHybridStack::new(0.02, 0.5, 1_000, 1_000);

		stack.insert(1, 10);
		stack.insert(2, 10);
		stack.insert(3, 10);
		stack.update(1);

		stack.clear();

		assert_eq!(stack.len(), 0);
		assert_eq!(stack.a1_in.len(), 0);
		assert_eq!(stack.a1_out.len(), 0);
		assert_eq!(stack.am.len(), 0);
		assert_eq!(stack.a1_in_used, 0);
		assert_eq!(stack.a1_out_used, 0);
		assert_eq!(stack.am_fast_used, 0);
		assert_eq!(stack.am_slow_used, 0);
		assert_eq!(stack.am_count, 0);
		assert_eq!(stack.am_fast_count, 0);
		assert_eq!(stack.am_boundary, None);
		assert!(drain(&mut stack).is_empty());

		assert_eq!(stack.a1_in_capacity, 20);
		assert_eq!(stack.a1_out_capacity, 500);
	}
}
