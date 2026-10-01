/*
 * Copyright (c) Kia Shakiba
 *
 * This source code is licensed under the GNU AGPLv3 license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! The tiering layer the hybrid designs share (R4): [`TieredStack<P>`] owns
//! what a tier-segmented eviction stack does whatever its policy, and a
//! [`TierPolicy`] supplies what it decides.
//!
//! Every hybrid stack used to be a whole file: an [`ArenaQueueSet`], a pair of
//! byte and object counts per tier, a cursor at the tier boundary, the
//! metadata reservation, the settle loop, the migration log, and the
//! `PolicyStack` methods that forward to them -- 300 to 500 lines, of which
//! about 70 were the design. R3 took the first three (LRU, FIFO, CLOCK) into
//! one stack with a three-hook `ArenaOrder`; this is the same layer for any
//! number of queues.
//!
//! # The model
//!
//! A stack is up to [`MAX_QUEUES`] LANES, the queues of one
//! [`ArenaQueueSet`], and every tracked key is in exactly one. A [`Layout`]
//! says which lanes are SPLIT: a split lane holds a fast prefix and a slow
//! suffix of one order, with a cursor at its OLDEST fast key, so a demotion is
//! one step of the cursor and nothing is searched. Every other lane is
//! entirely slow (a probation queue whose bytes were built in the slow tier)
//! or, if the layout says so, entirely FAST: the DRAM admission queue of the
//! fast-admission designs, a carve-out of the fast tier (see `carve`).
//! The cursor steps over STRUCTURAL keys (S5): a key whose value is larger
//! than an empty fast tier keeps its place in the order with tier slow, and
//! the walk (`prev_fast`) skips it. The payload's `queue` is the lane and its
//! `tier` the placement, always recorded; `freq` is the reference bit.
//!
//! The layer keeps one [`Book`] per lane -- bytes and objects, per tier -- and
//! every gauge is a sum of them; the migration log, in emission order; the
//! reservation (measured M, else `len x omega` and the ghost's entries), eff
//! and the structural test; and the settle loop, which demotes the cursor's
//! key while the fast bytes of a lane are over the drain target of eff (or,
//! for a design with lazy demotion, reprieves it if its reference bit is
//! set).
//!
//! # The push rule
//!
//! A migration is pushed in the call that changes a key's tier, so the log
//! always ends in the key's placement (`PolicyStack::placement_of`): a
//! demotion pushes `(key, Slow)`; a key that leaves the fast set because it
//! is structural pushes it before the settle; a promotion pushes `(key, Fast)`
//! AFTER the settle that may undo it, guarded on the key still being fast. A
//! new key pushes nothing -- unless the ghost remembers it: built slow and
//! admitted fast, it is pushed `(key, Fast)` after the settle. One accident
//! of history is a knob rather than a rule: the S3-FIFO family's second
//! chance pushes `(key, Fast)` for a key that was already fast
//! ([`Push::IfEndsFast`]), where everything else pushes only for a real
//! promotion ([`Push::IfPromoted`]). Both are kept.
//!
//! # A policy
//!
//! [`TierPolicy`] is R3's three hooks -- `hit`, `overwrite` and the victim --
//! with the same defaults (LRU's rule is what the defaults of `hit` and
//! `overwrite` come to, given a `touch`), and the constants that say where a
//! new key goes and which lanes each entry point settles -- they differ per
//! design and are copied from each design's own bodies, not derived. The
//! hooks get the stack and read what they need from it: `update` is
//! `P::hit(self, key)`, so a FIFO hit costs nothing and a CLOCK hit one
//! payload write.
//!
//! # Cost
//!
//! Every key-addressed operation on the queues is one probe of the keyed
//! index (a hash and, at scale, a cache miss), and the layer makes no more of
//! them than the stacks it replaced: a payload is written only when it
//! changes (an LRU hit on a fast key writes nothing), a tail eviction is one
//! removal that hands back the payload, a settle hands its candidate's
//! payload to the demotion, and a second chance reads whether the key is
//! structural from the payload the relocation reads anyway. `tier_probes.rs`
//! counts them.

use std::marker::PhantomData;

use crate::{object::ObjectSize, PaperPolicy};

use super::{
	arena_queue_set::{ArenaQueueSet, MAX_QUEUES, NodePayload},
	ghost_filter::GhostFilter,
	drain_target, fast_at_or_before, narrow_resident, placed, prev_fast, CacheSize, HashedKey, Placement, PolicyStack, SetEvent, Tier,
};

pub(super) mod carve;

pub use carve::newly_fills;

/// A lane: a queue of the shared [`ArenaQueueSet`].
pub type Lane = usize;

/// The shape of a stack's lanes, a type so that the layer is compiled once per
/// shape rather than once per design, and the test of a lane's kind folds.
pub trait Layout: 'static {
	/// Lanes in use.
	const LANES: usize;

	/// Bit `i` set: lane `i` is SPLIT (a fast prefix, a cursor, a slow suffix);
	/// clear: lane `i` is entirely slow, or entirely fast if its bit of `FAST`
	/// is set.
	const SPLIT: u8;

	/// Bit `i` set: lane `i` is FAST, a DRAM admission queue carved out of the
	/// fast tier (see `carve`): every key in it fast, and no cursor.
	const FAST: u8 = 0;
}

/// One split lane: LRU, FIFO and CLOCK.
pub struct Single;

impl Layout for Single {
	const LANES: usize = 1;
	const SPLIT: u8 = 0b1;
}

/// Lane 0 an entirely slow probation queue (2Q's FIFO, S3-FIFO's one-access
/// queue), lane 1 a split main queue.
pub struct SlowSplit;

impl Layout for SlowSplit {
	const LANES: usize = 2;
	const SPLIT: u8 = 0b10;
}

/// Lane 0 an entirely fast DRAM admission queue, lane 1 a split main queue:
/// the fast-admission 2Q and S3-FIFO designs.
pub struct FastSplit;

impl Layout for FastSplit {
	const LANES: usize = 2;
	const SPLIT: u8 = 0b10;
	const FAST: u8 = 0b01;
}

/// When a key moved to the front of a split lane is pushed `(key, Fast)`.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Push {
	/// Only if it was slow and ends fast: a promotion.
	IfPromoted,

	/// Whenever it ends fast, even if it already was: the S3-FIFO family's
	/// second chance.
	IfEndsFast,
}

/// An end of a lane.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum End {
	Front,
	Back,
}

/// What a fast lane does with the bytes over its budget: its tail is spliced
/// onto `end` of lane `to` as slow, a demotion pushed `(key, Slow)`.
#[derive(Clone, Copy)]
pub struct Spill {
	pub to: Lane,
	pub end: End,
}

/// What a `Set` says about its value, for the hooks.
#[derive(Clone, Copy)]
pub struct Meta {
	pub size: ObjectSize,
	pub resident: u8,

	/// Larger than an empty fast tier (S5): the client's flag or this stack's
	/// own check.
	pub structural: bool,
}

impl Meta {
	/// The bytes that would move if the object migrated.
	pub fn migrating(&self) -> CacheSize {
		(self.size as CacheSize).saturating_sub(self.resident as CacheSize)
	}
}

/// What a design remembers of the keys it evicted: the S3-FIFO family's
/// fingerprint table, or nothing ([`NoGhost`]). The layer owns its lifecycle --
/// a `remove` drops a key's fingerprint before it asks whether the key is
/// tracked (after a probation eviction a key lives ONLY in the ghost), a
/// `clear` empties it, and its bytes join the reservation and the structure
/// -- and the policy decides what goes in and when it is trimmed.
pub trait Ghost: Send + 'static {
	/// Whether the design has a ghost at all.
	const PRESENT: bool;

	/// A ghost for a cache of `max_size` bytes.
	fn sized_for(max_size: CacheSize) -> Self;

	fn contains(&self, key: HashedKey) -> bool;
	fn insert(&mut self, key: HashedKey);
	fn remove(&mut self, key: HashedKey);
	fn clear(&mut self);

	/// Ages out what was inserted more than `entries` insertions ago.
	fn set_window(&mut self, entries: usize);

	/// DRAM its entries are charged, in the per-object reservation.
	fn dram_bytes(&self) -> CacheSize;

	/// Usable bytes its table holds (S5a).
	// Read only through `PolicyStack::structure_bytes`, which only a tiered
	// cache's worker calls.
	#[cfg_attr(not(feature = "hybrid_cache_common"), allow(dead_code))]
	fn allocated_bytes(&self) -> u64;
}

/// No ghost.
pub struct NoGhost;

impl Ghost for NoGhost {
	const PRESENT: bool = false;

	fn sized_for(_max_size: CacheSize) -> Self {
		NoGhost
	}

	fn contains(&self, _key: HashedKey) -> bool {
		false
	}

	fn insert(&mut self, _key: HashedKey) {}
	fn remove(&mut self, _key: HashedKey) {}
	fn clear(&mut self) {}
	fn set_window(&mut self, _entries: usize) {}

	fn dram_bytes(&self) -> CacheSize {
		0
	}

	fn allocated_bytes(&self) -> u64 {
		0
	}
}

impl Ghost for GhostFilter {
	const PRESENT: bool = true;

	/// Sized from the cache's own capacity assuming a 512-byte nominal object,
	/// capped at 8 Mi slots. Under-sizing only costs ghost hits.
	fn sized_for(max_size: CacheSize) -> Self {
		GhostFilter::with_capacity(((max_size / 512) as usize).min(8 << 20))
	}

	fn contains(&self, key: HashedKey) -> bool {
		GhostFilter::contains(self, key)
	}

	fn insert(&mut self, key: HashedKey) {
		GhostFilter::insert(self, key);
	}

	fn remove(&mut self, key: HashedKey) {
		GhostFilter::remove(self, key);
	}

	fn clear(&mut self) {
		GhostFilter::clear(self);
	}

	fn set_window(&mut self, entries: usize) {
		GhostFilter::set_window(self, entries);
	}

	fn dram_bytes(&self) -> CacheSize {
		GhostFilter::dram_bytes(self)
	}

	fn allocated_bytes(&self) -> u64 {
		GhostFilter::allocated_bytes(self)
	}
}

/// The rules a design supplies to the layer.
pub trait TierPolicy: Sized + Send + 'static {
	type Layout: Layout;
	type Ghost: Ghost;

	/// The lane a brand-new key enters.
	const ADMIT: Lane = 0;

	/// The lanes `resettle`, `resize_fast_tier` and `resize` settle, in order:
	/// they differ per design. A fast-tier resize settles what a resettle does
	/// unless the design says otherwise.
	const RESETTLE: &'static [Lane] = &[0];
	const ON_TIER_RESIZE: &'static [Lane] = Self::RESETTLE;
	const ON_RESIZE: &'static [Lane] = &[];

	/// The push rule of [`TieredStack::second_chance`].
	const SECOND_CHANCE: Push = Push::IfPromoted;

	/// Lazy demotion: the settle gives a candidate whose reference bit is set
	/// a fresh start at the front instead of demoting it.
	const LAZY_DEMOTION: bool = false;

	/// What a fast lane does with the bytes over its budget; `None`: nothing,
	/// the design evicts from it (`wants_eviction`).
	const SPILL: Option<Spill> = None;

	/// Whether `policy` names this design.
	fn is_policy(&self, policy: &PaperPolicy) -> bool;

	/// What touching a tracked key does, given whether its value is
	/// structural: the one rule a hit and an overwrite share in most designs.
	fn touch(_s: &mut TieredStack<Self>, _key: HashedKey, _structural: bool) {}

	/// A hit on `key`. The default reads whether it is structural and touches
	/// it; FIFO's rule (nothing) overrides it, so that costs no probe.
	fn hit(s: &mut TieredStack<Self>, key: HashedKey) {
		if let Some(structural) = s.structural_of(key) {
			Self::touch(s, key, structural);
		}
	}

	/// A `Set` of a key the stack tracks: resized, then touched.
	fn overwrite(s: &mut TieredStack<Self>, key: HashedKey, m: Meta) {
		s.resize_key(key, m);
		Self::touch(s, key, m.structural);
	}

	/// A `Set` of a key it does not: the front of the admission lane.
	fn admit(s: &mut TieredStack<Self>, key: HashedKey, m: Meta) {
		s.admit(key, m, Self::ADMIT, false);
	}

	/// Removes and returns the key this design evicts next: the tail of lane 0.
	fn victim(s: &mut TieredStack<Self>) -> Option<HashedKey> {
		s.evict_tail(0)
	}

	/// A key was evicted from `lane` (not removed, not moved): the ghost's to
	/// hear of.
	fn evicted(_s: &mut TieredStack<Self>, _lane: Lane, _key: HashedKey) {}

	/// `needs_capacity_eviction`: a sub-queue over its own capacity.
	fn wants_eviction(_s: &TieredStack<Self>) -> bool {
		false
	}

	/// `resize`: the cache's size changed.
	fn resized(_s: &mut TieredStack<Self>, _max_size: CacheSize) {}

	/// `resize_fast_tier`: the fast tier's size changed, before the lanes
	/// `ON_TIER_RESIZE` lists are settled.
	fn tier_resized(_s: &mut TieredStack<Self>) {}

	/// What `lane` may hold of fast bytes before the drain target: eff, unless
	/// the design carves the fast tier up between its lanes
	/// (`TieredStack::carve_budgets`).
	fn budget(s: &TieredStack<Self>, _lane: Lane) -> CacheSize {
		s.eff()
	}
}

/// Bytes and objects a lane holds, per tier (index 0 fast, 1 slow).
#[derive(Clone, Copy, Default)]
struct Book {
	bytes: [CacheSize; 2],
	count: [usize; 2],
}

fn idx(tier: Tier) -> usize {
	match tier {
		Tier::Fast => 0,
		Tier::Slow => 1,
	}
}

/// The placement a payload records. Always there: a tiered stack writes one
/// for every key, the keys of a slow lane (a probation queue) included.
fn placed_in(payload: &NodePayload) -> Tier {
	payload.tier.expect("a tiered stack records a tier for every key")
}

/// Whether a value of `migrating` bytes is STRUCTURAL (S5) in a fast tier of
/// `eff` bytes: larger than the tier empty.
fn structural_in(eff: CacheSize, migrating: CacheSize) -> bool {
	migrating > eff
}

/// The queues, their cursors and books, and the log: everything that depends
/// on the layout but not on the policy.
struct Lanes<L: Layout> {
	set: ArenaQueueSet<NodePayload>,

	/// Per split lane: its OLDEST fast key; everything from the front up to it
	/// is fast (but for structural keys), everything after is slow.
	cursor: [Option<HashedKey>; MAX_QUEUES],

	books: [Book; MAX_QUEUES],

	/// Every migration the stack queued since the last drain, in order.
	log: Vec<(HashedKey, Tier)>,

	layout: PhantomData<fn() -> L>,
}

impl<L: Layout> Lanes<L> {
	fn new() -> Self {
		Lanes { set: ArenaQueueSet::default(), cursor: [None; MAX_QUEUES], books: [Book::default(); MAX_QUEUES], log: Vec::new(), layout: PhantomData }
	}

	fn split(lane: Lane) -> bool {
		L::SPLIT >> lane & 1 == 1
	}

	fn fast(lane: Lane) -> bool {
		L::FAST >> lane & 1 == 1
	}

	/// The tier a brand-new key is built in at an end of `lane`: fast in a fast
	/// lane, and in a split lane unless its value is structural; else slow.
	fn tier_for(lane: Lane, structural: bool) -> Tier {
		match Self::fast(lane) || Self::split(lane) && !structural {
			true => Tier::Fast,
			false => Tier::Slow,
		}
	}

	fn credit(&mut self, lane: Lane, tier: Tier, bytes: CacheSize) {
		let book = &mut self.books[lane];

		book.bytes[idx(tier)] += bytes;
		book.count[idx(tier)] += 1;
	}

	fn debit(&mut self, lane: Lane, tier: Tier, bytes: CacheSize) {
		let book = &mut self.books[lane];

		book.bytes[idx(tier)] = book.bytes[idx(tier)].saturating_sub(bytes);
		book.count[idx(tier)] = book.count[idx(tier)].saturating_sub(1);
	}

	fn sum<T: Copy + std::iter::Sum>(&self, of: impl Fn(&Book) -> T) -> T {
		self.books[..L::LANES].iter().map(of).sum()
	}

	/// Applies a re-`set`'s size to a tracked key and its lane's books.
	fn resize_key(&mut self, key: HashedKey, m: Meta) {
		let Some(payload) = self.set.payload_mut(key) else { return };

		let old = payload.migrating();

		payload.size = m.size;
		payload.dram_resident = m.resident;

		let delta = payload.migrating() as i64 - old as i64;
		let bytes = &mut self.books[payload.queue as usize].bytes[idx(placed_in(payload))];

		*bytes = (*bytes as i64 + delta).max(0) as CacheSize;
	}

	/// A brand-new key at `end` of `lane`, in `tier` (fast only in a fast lane,
	/// or at the front of a split one).
	fn add(&mut self, lane: Lane, key: HashedKey, m: Meta, tier: Tier, end: End) {
		let payload = NodePayload {
			size: m.size,
			dram_resident: m.resident,
			tier: Some(tier),
			freq: 0,
			ts: 0,
			queue: lane as u8,
		};

		match end {
			End::Front => self.set.push_front(lane, key, payload),
			End::Back => self.set.push_back(lane, key, payload),
		}

		self.credit(lane, tier, m.migrating());

		// A fast key with no fast key in front of it is the cursor.
		if tier == Tier::Fast && Self::split(lane) && self.cursor[lane].is_none() {
			self.cursor[lane] = Some(key);
		}
	}

	/// Takes a tracked key out of its lane, off the books, and steps the cursor
	/// off it if it was the cursor (past any structural keys).
	fn unlink(&mut self, key: HashedKey) -> Option<NodePayload> {
		let payload = self.set.payload(key)?;
		let lane = payload.queue as usize;
		let stepped = (placed_in(&payload) == Tier::Fast && self.cursor[lane] == Some(key)).then(|| prev_fast(&self.set, key));

		self.set.remove(lane, key);
		self.debit(lane, placed_in(&payload), payload.migrating());

		if let Some(next) = stepped {
			self.cursor[lane] = next;
		}

		Some(payload)
	}

	/// Takes the tail of `lane` out, off the books: one probe, since the lane is
	/// known and the removal hands back the payload. If it was the cursor, the
	/// cursor steps to the fast key now at or before the new tail (S5: past any
	/// structural ones) -- the nearest fast key in front of the one that left.
	fn unlink_tail(&mut self, lane: Lane) -> Option<HashedKey> {
		let key = self.set.back(lane)?;
		let was_cursor = self.cursor[lane] == Some(key);
		let payload = self.set.remove(lane, key)?;

		self.debit(lane, placed_in(&payload), payload.migrating());

		if was_cursor && placed_in(&payload) == Tier::Fast {
			self.cursor[lane] = fast_at_or_before(&self.set, self.set.back(lane));
		}

		Some(key)
	}

	/// Moves a tracked key to the front of `to`, re-placing it: slow if
	/// `structural`, else fast. `structural` is the caller's when it has it;
	/// otherwise it is read from the payload this reads anyway, against `eff`.
	/// Books, cursors and the bit follow; a fast key that turns slow is pushed
	/// now, before the caller settles. Returns whether it was a promotion (slow
	/// to fast) and whether it is structural.
	fn relocate(&mut self, key: HashedKey, to: Lane, structural: Option<bool>, eff: CacheSize) -> Option<(bool, bool)> {
		let payload = self.set.payload(key)?;
		let (from, was, size) = (payload.queue as usize, placed_in(&payload), payload.migrating());
		let structural = structural.unwrap_or_else(|| structural_in(eff, size));
		let now = if structural { Tier::Slow } else { Tier::Fast };
		let at_front = from == to && self.set.front(to) == Some(key);

		// The cursor steps back off the key before it moves, to the fast key in
		// front of it (S5: past any structural ones).
		let stepped = (self.cursor[from] == Some(key) && !at_front).then(|| prev_fast(&self.set, key));

		match from == to {
			true => self.set.move_front(to, key),
			false => self.set.move_to_front_of(from, to, key),
		}

		if let Some(next) = stepped {
			self.cursor[from] = next;
		}

		if was != now || from != to {
			self.debit(from, was, size);
			self.credit(to, now, size);
		}

		// Not written when it would write what is there -- the same lane, the same
		// tier, a clear bit: an LRU hit on a fast key, the common one.
		if from != to || was != now || payload.freq != 0 {
			if let Some(slot) = self.set.payload_mut(key) {
				slot.queue = to as u8;
				slot.tier = Some(now);
				slot.freq = 0;
			}
		}

		match (structural, was) {
			// Leaves the fast set: still the cursor only if it was already at
			// the front, the one fast key, and then none is left.
			(true, Tier::Fast) => {
				if self.cursor[to] == Some(key) {
					self.cursor[to] = None;
				}

				self.log.push((key, Tier::Slow));
			},

			(true, Tier::Slow) => {},

			// Fast, at the front; with no fast key in front of it the cursor.
			(false, _) => {
				if Self::split(to) && self.cursor[to].is_none() {
					self.cursor[to] = Some(key);
				}
			},
		}

		Some((was == Tier::Slow && now == Tier::Fast, structural))
	}

	/// Books a fast key as slow, in place, and steps the cursor off it if it
	/// was the cursor. `payload` is what the caller read of it. The caller
	/// pushes `(key, Slow)` where the demotion is not a settle's.
	fn demote(&mut self, key: HashedKey, payload: NodePayload) {
		let lane = payload.queue as usize;

		if self.cursor[lane] == Some(key) {
			self.cursor[lane] = prev_fast(&self.set, key);
		}

		if let Some(slot) = self.set.payload_mut(key) {
			slot.tier = Some(Tier::Slow);
		}

		self.debit(lane, Tier::Fast, payload.migrating());
		self.credit(lane, Tier::Slow, payload.migrating());
	}

	/// `demote`, for a caller that has not read the key.
	fn demote_key(&mut self, key: HashedKey) {
		if let Some(payload) = self.set.payload(key) {
			self.demote(key, payload);
		}
	}

	/// A reprieve (lazy demotion): the settle's candidate, still fast, goes to
	/// the front with its bit cleared -- no books, no migration. The cursor
	/// steps to the fast key in front of it; with none, the candidate, now at
	/// the front, is the cursor again, and its cleared bit demotes it at the
	/// next step.
	fn reprieve(&mut self, lane: Lane, key: HashedKey) {
		let next = prev_fast(&self.set, key);

		self.set.move_front(lane, key);
		self.cursor[lane] = next.or(Some(key));

		if let Some(slot) = self.set.payload_mut(key) {
			slot.freq = 0;
		}
	}

	/// Demotes from the cursor of `lane` until its fast bytes are within
	/// `target`. The victim is always the cursor, so nothing is searched. With
	/// `lazy`, a candidate whose bit is set is reprieved instead and the sweep
	/// goes on to the next.
	fn settle(&mut self, lane: Lane, target: CacheSize, lazy: bool) {
		while self.books[lane].bytes[idx(Tier::Fast)] > target {
			let Some(candidate) = self.cursor[lane] else { break };
			let payload = self.set.payload(candidate);

			if lazy && payload.is_some_and(|p| p.freq != 0) {
				self.reprieve(lane, candidate);

				continue;
			}

			match payload {
				Some(payload) => self.demote(candidate, payload),
				None => self.cursor[lane] = None,
			}

			self.log.push((candidate, Tier::Slow));
		}
	}

	fn clear(&mut self) {
		self.set.clear();
		self.cursor = [None; MAX_QUEUES];
		self.books = [Book::default(); MAX_QUEUES];
		self.log.clear();
	}
}

/// One tier-segmented eviction stack: [`Lanes`] plus the reservation, driven
/// by a [`TierPolicy`]. Implements `PolicyStack` once, for every design.
pub struct TieredStack<P: TierPolicy> {
	/// The design's own state: its ratios, capacities, counters.
	pub policy: P,

	/// What it remembers of the keys it evicted.
	pub ghost: P::Ghost,

	lanes: Lanes<P::Layout>,

	fast_capacity: CacheSize,
	shared_overhead: CacheSize,

	/// S5: the measured M the policy worker pushed (`set_dram_metadata`),
	/// reserved instead of `len x shared_overhead`; `None` under the
	/// per-object model.
	measured: Option<CacheSize>,
}

impl<P: TierPolicy<Ghost = NoGhost>> TieredStack<P> {
	pub fn with(policy: P, fast_capacity: CacheSize) -> Self {
		Self::with_ghost(policy, fast_capacity, NoGhost)
	}
}

impl<P: TierPolicy> TieredStack<P> {
	pub fn with_ghost(policy: P, fast_capacity: CacheSize, ghost: P::Ghost) -> Self {
		TieredStack { policy, ghost, lanes: Lanes::new(), fast_capacity, shared_overhead: 0, measured: None }
	}

	/// Per-object DRAM reserved from the fast tier for shared metadata.
	// Called only under `hybrid_cache_common`, by `init_policy_stack`.
	#[cfg_attr(not(feature = "hybrid_cache_common"), allow(dead_code))]
	pub fn with_shared_overhead(mut self, overhead: CacheSize) -> Self {
		self.shared_overhead = overhead;

		self
	}

	/// Metadata reservation for EVERY tracked key, fast or slow: a demotion
	/// moves the value and leaves the key's row, stack node and header in
	/// DRAM -- plus the ghost's entries, which are DRAM as well (under the
	/// measured model its table is inside the pushed M, so its own term applies
	/// only to the per-object reservation). See
	/// `PolicyStack::dram_reserved_bytes` for the rule, and for why a
	/// reservation at or over `fast_capacity` is left to saturate.
	fn reserved(&self) -> CacheSize {
		self.measured.unwrap_or(self.lanes.set.len() as CacheSize * self.shared_overhead + self.ghost.dram_bytes())
	}

	/// This stack's eff (S5): the whole fast tier's budget for values, its
	/// settle's figure before the drain target.
	pub fn eff(&self) -> CacheSize {
		self.fast_capacity.saturating_sub(self.reserved())
	}

	/// Whether a value of `migrating` bytes is STRUCTURAL (S5): larger than an
	/// empty fast tier. Such a key is placed slow and never promoted while it
	/// stays that large. The stack's own check beside the client's flag, so a
	/// key the client placed normally just before eff moved is placed as the
	/// stack's own promotions would place it.
	fn structural(&self, migrating: CacheSize) -> bool {
		structural_in(self.eff(), migrating)
	}

	/// Whether the tracked `key`'s value is structural now.
	pub fn structural_of(&self, key: HashedKey) -> Option<bool> {
		self.lanes.set.payload(key).map(|p| self.structural(p.migrating()))
	}

	pub fn tier_of(&self, key: HashedKey) -> Option<Tier> {
		self.lanes.set.payload(key).map(|p| placed_in(&p))
	}

	pub fn tail(&self, lane: Lane) -> Option<HashedKey> {
		self.lanes.set.back(lane)
	}

	/// The lane `key` is in.
	pub fn lane_of(&self, key: HashedKey) -> Option<Lane> {
		self.lanes.set.payload(key).map(|p| p.queue as Lane)
	}

	/// Keys in `lane`.
	pub fn lane_len(&self, lane: Lane) -> usize {
		self.lanes.set.queue_len(lane)
	}

	/// Bytes `lane` holds, in both tiers.
	pub fn lane_bytes(&self, lane: Lane) -> CacheSize {
		self.lanes.books[lane].bytes.iter().sum()
	}

	/// The reference bit (`freq != 0`).
	pub fn bit(&self, key: HashedKey) -> bool {
		self.lanes.set.payload(key).is_some_and(|p| p.freq != 0)
	}

	pub fn set_bit(&mut self, key: HashedKey, on: bool) {
		if let Some(slot) = self.lanes.set.payload_mut(key) {
			slot.freq = u32::from(on);
		}
	}

	/// Applies a re-`set`'s size to a tracked key's books.
	pub fn resize_key(&mut self, key: HashedKey, m: Meta) {
		self.lanes.resize_key(key, m);
	}

	pub fn fast_capacity(&self) -> CacheSize {
		self.fast_capacity
	}

	/// Settles `lane` against its budget's drain target: a split lane demotes
	/// from its cursor, a fast lane spills its tail (if the design says
	/// where), and a slow lane has nothing to settle.
	pub fn settle(&mut self, lane: Lane) {
		let target = drain_target::bytes(P::budget(self, lane));

		match (Lanes::<P::Layout>::split(lane), P::SPILL) {
			(true, _) => self.lanes.settle(lane, target, P::LAZY_DEMOTION),
			(false, Some(spill)) if Lanes::<P::Layout>::fast(lane) => self.lanes.spill(lane, target, spill),
			_ => {},
		}
	}

	/// A brand-new key at the front of `lane`: fast, settled, if the lane is
	/// fast, or split and the value not structural; else slow (built slow,
	/// charged slow, nothing settled, nothing pushed). `push`: the key was
	/// built slow and goes to fast (a ghost hit), so it is pushed `(key, Fast)`
	/// after the settle, guarded on still being fast.
	pub fn admit(&mut self, key: HashedKey, m: Meta, lane: Lane, push: bool) {
		if self.place(key, m, lane, End::Front) == Tier::Fast {
			self.settle(lane);

			if push && self.tier_of(key) == Some(Tier::Fast) {
				self.lanes.log.push((key, Tier::Fast));
			}
		}
	}

	/// A brand-new key at `end` of `lane`, built in the tier the lane gives it
	/// (see `admit`) and booked there: nothing settled, nothing pushed. Returns
	/// the tier. A key at the back is a slow one: a fast key there would break
	/// the cursor.
	pub fn place(&mut self, key: HashedKey, m: Meta, lane: Lane, end: End) -> Tier {
		let tier = Lanes::<P::Layout>::tier_for(lane, m.structural);

		debug_assert!(end == End::Front || tier == Tier::Slow, "a fast key is placed at the front");

		self.lanes.add(lane, key, m, tier, end);

		tier
	}

	/// Moves `key` to the front of `to` and re-places it -- LRU's hit, and a
	/// second chance -- and, since S5, the structural rule: a STRUCTURAL key
	/// moves to the front all the same, its place in the order, but with tier
	/// slow; a fast one leaves the fast set, pushed `(key, Slow)` (its
	/// placement changed, and a promotion of its old value may still be in
	/// flight, which this entry, behind it on the key's FIFO consumer, undoes).
	///
	/// Otherwise it is fast at the front, the settle runs, and the promotion is
	/// pushed after it, guarded on the key still being fast (a tight budget can
	/// demote it straight back out, and that settle pushed the right final
	/// entry). Pushed even when the bytes are already fast, as they are on a
	/// re-`set` of a slow key: the API thread built the new value in DRAM
	/// before this worker saw the event, so the consumer will decline the
	/// entry, but it still has to be queued -- queued migrations carry no
	/// identity, so a demotion decided for the old object and still queued
	/// when the new one replaced it demotes the new one, and this entry behind
	/// it restores it.
	pub fn to_front(&mut self, key: HashedKey, to: Lane, structural: bool, push: Push) {
		self.bring_to_front(key, to, Some(structural), push);
	}

	/// `to_front`, with `structural` the caller's, or `None` for the layer to
	/// read it from the payload it reads anyway.
	fn bring_to_front(&mut self, key: HashedKey, to: Lane, structural: Option<bool>, push: Push) {
		let eff = if structural.is_none() { self.eff() } else { 0 };
		let Some((promoted, structural)) = self.lanes.relocate(key, to, structural, eff) else { return };

		self.settle(to);

		let pushed = promoted || push == Push::IfEndsFast;

		if !structural && pushed && self.tier_of(key) == Some(Tier::Fast) {
			self.lanes.log.push((key, Tier::Fast));
		}
	}

	/// A second chance: `key` to the front of `lane` by the rule of
	/// [`TierPolicy::SECOND_CHANCE`].
	pub fn second_chance(&mut self, key: HashedKey, lane: Lane) {
		self.bring_to_front(key, lane, None, P::SECOND_CHANCE);
	}

	/// Takes a FAST key out of the fast set IN PLACE (S5): an overwrite with a
	/// value larger than an empty fast tier, in an order that keeps an
	/// overwritten key where it is. Pushed `(key, Slow)`: its placement changed.
	pub fn demote_in_place(&mut self, key: HashedKey) {
		self.lanes.demote_key(key);
		self.lanes.log.push((key, Tier::Slow));
	}

	/// An overwrite that leaves the key WHERE IT IS in its order -- FIFO's and
	/// CLOCK's: it is resized, taken out of the fast set in place when its new
	/// value is STRUCTURAL and it was fast, and re-settled only if it was fast
	/// and resized, since only then can the resize have pushed the fast tier
	/// over its budget.
	pub fn overwrite_in_place(&mut self, key: HashedKey, m: Meta) {
		let Some(payload) = self.lanes.set.payload(key) else { return };

		let (lane, fast) = (payload.queue as Lane, placed_in(&payload) == Tier::Fast);
		let resized = payload.size != m.size;

		if resized {
			self.resize_key(key, m);
		}

		if m.structural && fast {
			self.demote_in_place(key);
		}

		if resized && fast {
			self.settle(lane);
		}
	}

	/// Removes the tail of `lane` from the lane and the books, and returns it:
	/// the victim.
	pub fn evict_tail(&mut self, lane: Lane) -> Option<HashedKey> {
		let key = self.lanes.unlink_tail(lane)?;

		P::evicted(self, lane, key);

		Some(key)
	}

	/// `insert`'s body: a `Set`, with the client's placement (S5). A tracked
	/// key is the policy's to overwrite, a new one its to admit; either way the
	/// answer is the placement applied.
	fn insert_with(&mut self, key: HashedKey, size: ObjectSize, dram_resident: ObjectSize, placement: Placement) -> Placement {
		let resident = narrow_resident(dram_resident);
		let migrating = (size as CacheSize).saturating_sub(resident as CacheSize);
		let structural = placement == Placement::Structural || self.structural(migrating);
		let meta = Meta { size, resident, structural };

		match self.lanes.set.contains(key) {
			true => P::overwrite(self, key, meta),
			false => P::admit(self, key, meta),
		}

		placed(structural)
	}
}

impl<P: TierPolicy> PolicyStack for TieredStack<P> {
	fn is_policy(&self, policy: &PaperPolicy) -> bool {
		self.policy.is_policy(policy)
	}

	fn len(&self) -> usize {
		self.lanes.set.len()
	}

	fn contains(&self, key: HashedKey) -> bool {
		self.lanes.set.contains(key)
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

	fn resettle(&mut self) {
		for &lane in P::RESETTLE {
			self.settle(lane);
		}
	}

	fn update(&mut self, key: HashedKey) {
		P::hit(self, key);
	}

	fn remove(&mut self, key: HashedKey) {
		// BEFORE the tracked check: after a probation eviction a key lives only
		// in the ghost, with no entry row to find.
		self.ghost.remove(key);
		self.lanes.unlink(key);
	}

	fn resize(&mut self, max_size: CacheSize) {
		P::resized(self, max_size);

		for &lane in P::ON_RESIZE {
			self.settle(lane);
		}
	}

	fn clear(&mut self) {
		self.lanes.clear();
		self.ghost.clear();
	}

	fn evict_one(&mut self) -> Option<HashedKey> {
		P::victim(self)
	}

	fn resize_fast_tier(&mut self, size: CacheSize) {
		self.fast_capacity = size;

		P::tier_resized(self);

		for &lane in P::ON_TIER_RESIZE {
			self.settle(lane);
		}
	}

	/// `tier_of`: every demotion is pushed by the settle or by a structural
	/// re-`set`, and every promotion after its settle, guarded on the key still
	/// being fast -- so the log ends in the placement.
	/// See `PolicyStack::placement_of`.
	fn placement_of(&self, key: HashedKey) -> Option<Tier> {
		self.tier_of(key)
	}

	fn drain_tier_migrations(&mut self) -> Vec<(HashedKey, Tier)> {
		std::mem::take(&mut self.lanes.log)
	}

	fn structure_bytes(&self) -> Option<crate::meta::NodeBytes> {
		Some(crate::meta::NodeBytes::stack(self.lanes.set.allocated_bytes() + self.ghost.allocated_bytes()))
	}

	fn dram_reserved_bytes(&self) -> CacheSize {
		self.reserved()
	}

	fn needs_capacity_eviction(&self) -> bool {
		P::wants_eviction(self)
	}

	fn fast_bytes_used(&self) -> CacheSize {
		self.lanes.sum(|b| b.bytes[idx(Tier::Fast)])
	}

	fn slow_bytes_used(&self) -> CacheSize {
		self.lanes.sum(|b| b.bytes[idx(Tier::Slow)])
	}

	fn fast_object_count(&self) -> usize {
		self.lanes.sum(|b| b.count[idx(Tier::Fast)])
	}

	fn slow_object_count(&self) -> usize {
		self.lanes.sum(|b| b.count[idx(Tier::Slow)])
	}
}

/// Helpers the designs' tests share: what a test may read of a stack's
/// internals, and the audit that holds its books to its queues.
#[cfg(test)]
pub(super) mod testing {
	use super::*;

	impl<P: TierPolicy> TieredStack<P> {
		/// Slab slots the committed chunks hold (for the construction tests).
		pub fn slab_capacity(&self) -> usize {
			self.lanes.set.slab_capacity()
		}

		/// The newest key of `lane`.
		pub fn front(&self, lane: Lane) -> Option<HashedKey> {
			self.lanes.set.front(lane)
		}
	}

	/// Walks every lane from its newest end and holds the books to it: the
	/// object and byte counts of each tier are what walking finds, the cursor
	/// of a split lane is its oldest FAST key with nothing but slow keys
	/// behind it, a slow lane holds no fast key and a fast lane no slow one,
	/// and every gauge is the sum of the lanes'.
	pub fn audit<P: TierPolicy>(stack: &TieredStack<P>, step: usize) {
		let lanes = &stack.lanes;
		let mut walked = 0;

		for lane in 0..<P::Layout as Layout>::LANES {
			let (mut count, mut bytes) = ([0usize; 2], [0 as CacheSize; 2]);
			let mut oldest_fast = None;
			let mut next = lanes.set.front(lane);

			while let Some(key) = next {
				let payload = lanes.set.payload(key).expect("a queued key has a payload");

				assert_eq!(payload.queue as usize, lane, "step {step}: key {key} is queued in lane {lane} and says otherwise");

				let t = idx(placed_in(&payload));

				count[t] += 1;
				bytes[t] += payload.migrating();

				if t == idx(Tier::Fast) {
					oldest_fast = Some(key);
				}

				walked += 1;
				next = lanes.set.after(key);
			}

			let book = lanes.books[lane];

			assert_eq!((count, bytes), (book.count, book.bytes), "step {step}: lane {lane}'s books");

			match (Lanes::<P::Layout>::split(lane), Lanes::<P::Layout>::fast(lane)) {
				(true, _) => assert_eq!(lanes.cursor[lane], oldest_fast, "step {step}: lane {lane}'s cursor is the oldest fast key"),

				(false, true) => {
					assert_eq!(count[idx(Tier::Slow)], 0, "step {step}: lane {lane} is fast and holds a slow key");
					assert_eq!(lanes.cursor[lane], None, "step {step}: lane {lane} has no cursor");
				},

				(false, false) => {
					assert_eq!(count[idx(Tier::Fast)], 0, "step {step}: lane {lane} is slow and holds a fast key");
					assert_eq!(lanes.cursor[lane], None, "step {step}: lane {lane} has no cursor");
				},
			}
		}

		assert_eq!(walked, stack.len(), "step {step}: the queues' length");
		assert_eq!(stack.fast_bytes_used() + stack.slow_bytes_used(), (0..P::Layout::LANES).map(|l| stack.lane_bytes(l)).sum::<CacheSize>(), "step {step}: byte gauges");
	}

	/// The stack's books against its queues after EVERY operation of a long
	/// random sequence: sets of new and tracked keys (some larger than an empty
	/// fast tier, so structural), hits, removals, evictions, a shrinking and
	/// growing fast tier, a pushed measured M -- against a tight tier, so
	/// demotions, promotions and second chances all happen. What the
	/// eviction-order tests cannot see: a count or a byte total that drifts
	/// while the order stays right. Returns how many demotions and promotions
	/// the sequence queued.
	pub fn books_match_the_queue_after_every_operation<P: TierPolicy>(mut stack: TieredStack<P>) -> (usize, usize) {
		let mut x: u64 = 0x9E37_79B9_7F4A_7C15;
		let (mut demoted, mut promoted) = (0, 0);

		for step in 0..20_000usize {
			x ^= x << 13;
			x ^= x >> 7;
			x ^= x << 17;

			let key = x % 300;
			// Mostly small; every so often larger than the whole tier.
			let size = match (x >> 32) % 20 {
				0 => 30_000,
				1..=6 => 1_500,
				_ => 200 + ((x >> 40) % 1_000) as ObjectSize,
			};

			match (x >> 8) % 16 {
				0..=5 => { stack.insert_resident(key, size, 12); },
				6..=9 => stack.update(key),
				10 | 11 => stack.remove(key),
				12 | 13 => { stack.evict_one(); },
				14 => stack.resize_fast_tier(if (x >> 50) % 2 == 0 { 8_000 } else { 24_000 }),
				_ => stack.set_dram_metadata(if (x >> 50) % 2 == 0 { None } else { Some(((x >> 52) % 4_000) as CacheSize) }),
			}

			for (_, tier) in stack.drain_tier_migrations() {
				match tier {
					Tier::Slow => demoted += 1,
					Tier::Fast => promoted += 1,
				}
			}

			audit(&stack, step);
		}

		(demoted, promoted)
	}

	/// Eight 1,000-byte keys, each set and then hit (a design with a probation
	/// queue promotes it), all fast in a tier of 10,000; then the measured metadata
	/// takes the whole tier: the stack is over its budget, and nothing has settled
	/// it.
	pub fn over_its_budget<P: TierPolicy>(stack: &mut TieredStack<P>) {
		for key in 1..=8 {
			stack.insert(key, 1_000);
			stack.update(key);
		}

		assert_eq!(stack.fast_bytes_used(), 8_000, "the fixture must leave every key fast");

		drop(stack.drain_tier_migrations());

		stack.set_dram_metadata(Some(20_000));
	}

	/// A resize of the cache settles the lanes the policy lists for it, and the
	/// designs on this layer list none: on a stack over its budget, `resize` queues
	/// nothing, and the demotions are the next settle's. A recorded run shows the
	/// difference only for a cache of hundreds of keys against a tier of a few,
	/// which the checked-in grid does not have: this is what pins it.
	pub fn a_resize_settles_nothing<P: TierPolicy>(mut stack: TieredStack<P>) {
		over_its_budget(&mut stack);
		stack.resize(1 << 20);

		assert_eq!(stack.drain_tier_migrations(), vec![], "a resize settled the stack");

		stack.resettle();

		assert!(!stack.drain_tier_migrations().is_empty(), "the stack was not over its budget: this shows nothing");
	}
}

/// The layer on a policy of its own: two lanes, a slow probation queue and a
/// split main, with every verb the one-lane designs do not use -- a key moving
/// between lanes, a slow lane, a second chance that pushes for a key already
/// fast -- so that the machinery is pinned whatever the designs on it are.
#[cfg(test)]
mod tests {
	use super::*;
	use super::testing::{a_resize_settles_nothing, books_match_the_queue_after_every_operation, over_its_budget};

	const PROBATION: Lane = 0;
	const MAIN: Lane = 1;

	/// A hit promotes a probation key to main's front; a hit in main is a
	/// second chance, by the S3-FIFO rule (`ENDS_FAST`) or the common one; the
	/// victim is probation's tail, else main's; a resize of the cache settles
	/// main or nothing (`RESIZE_SETTLES`).
	struct Toy<const ENDS_FAST: bool, const RESIZE_SETTLES: bool = false>;

	impl<const ENDS_FAST: bool, const RESIZE_SETTLES: bool> TierPolicy for Toy<ENDS_FAST, RESIZE_SETTLES> {
		type Layout = SlowSplit;
		type Ghost = NoGhost;

		const ADMIT: Lane = PROBATION;
		const RESETTLE: &'static [Lane] = &[MAIN];
		const ON_TIER_RESIZE: &'static [Lane] = &[MAIN];
		const ON_RESIZE: &'static [Lane] = if RESIZE_SETTLES { &[MAIN] } else { &[] };
		const SECOND_CHANCE: Push = if ENDS_FAST { Push::IfEndsFast } else { Push::IfPromoted };

		fn is_policy(&self, _policy: &PaperPolicy) -> bool {
			false
		}

		fn touch(s: &mut TieredStack<Self>, key: HashedKey, structural: bool) {
			match s.lane_of(key) {
				Some(PROBATION) => s.to_front(key, MAIN, structural, Push::IfPromoted),
				_ => s.second_chance(key, MAIN),
			}
		}

		fn victim(s: &mut TieredStack<Self>) -> Option<HashedKey> {
			s.evict_tail(PROBATION).or_else(|| s.evict_tail(MAIN))
		}
	}

	#[test]
	fn the_books_of_a_two_lane_stack_match_its_queues_after_every_operation() {
		let (demoted, promoted) = books_match_the_queue_after_every_operation(TieredStack::with(Toy::<true>, 24_000).with_shared_overhead(40));

		assert!(demoted > 100, "the sequence never demoted ({demoted})");
		assert!(promoted > 100, "the sequence never promoted ({promoted})");
	}

	/// A clear discards the migrations still queued, as every stack this replaces
	/// did, and every tier of books. (The worker drains after every event, so
	/// nothing is pending when it wipes, and no recorded run clears with entries
	/// pending either: nothing else pins it.)
	#[test]
	fn a_clear_discards_the_migrations_still_queued() {
		let build = || {
			let mut stack = TieredStack::with(Toy::<true>, 250);

			for key in 1..=5 {
				stack.insert(key, 100);
				stack.update(key);
			}

			stack
		};

		assert!(!build().drain_tier_migrations().is_empty(), "the fixture must leave migrations queued");

		let mut stack = build();

		stack.clear();

		assert_eq!(stack.drain_tier_migrations(), vec![]);
		assert_eq!((stack.len(), stack.fast_bytes_used(), stack.slow_bytes_used()), (0, 0, 0));
		assert_eq!((stack.fast_object_count(), stack.slow_object_count()), (0, 0));
	}

	/// The push rule's one exception: a second chance of a key that is already
	/// fast queues `(key, Fast)` under `IfEndsFast` and nothing under
	/// `IfPromoted` -- through `to_front`, and through `second_chance`, which
	/// takes the policy's rule.
	#[test]
	fn a_second_chance_pushes_for_a_fast_key_only_under_the_s3_rule() {
		fn check<const ENDS_FAST: bool>(pushed: bool) {
			for by_policy in [false, true] {
				let mut stack = TieredStack::with(Toy::<ENDS_FAST>, 1 << 30);

				for key in 1..=3 {
					stack.insert(key, 100);
					stack.update(key);
				}

				assert_eq!(stack.tier_of(1), Some(Tier::Fast));
				drop(stack.drain_tier_migrations());

				match by_policy {
					true => stack.second_chance(1, MAIN),
					false => stack.to_front(1, MAIN, false, if ENDS_FAST { Push::IfEndsFast } else { Push::IfPromoted }),
				}

				assert_eq!(
					stack.drain_tier_migrations(),
					if pushed { vec![(1, Tier::Fast)] } else { vec![] },
					"IfEndsFast: {ENDS_FAST}, by the policy's rule: {by_policy}",
				);
			}
		}

		check::<true>(true);
		check::<false>(false);
	}

	/// The ghost is sized from the cache -- a slot per 512 bytes, rounded up to a
	/// power of two -- and capped at 8 Mi slots, so a cache of any size gets a
	/// table of 64 MiB at most.
	#[test]
	fn the_ghost_grows_with_the_cache_up_to_8_mi_slots() {
		let table = |max_size: CacheSize| <GhostFilter as Ghost>::sized_for(max_size).allocated_bytes();

		assert!(table(2 << 30) > table(1 << 30), "the table does not grow with the cache");
		assert!(table(4 << 30) > table(2 << 30), "the table stops growing before 8 Mi slots");
		assert_eq!(table(8 << 30), table(4 << 30), "the table is not capped at 8 Mi slots");
	}

	/// A resize settles nothing unless the policy lists a lane for it...
	#[test]
	fn a_resize_settles_nothing_by_default() {
		a_resize_settles_nothing(TieredStack::with(Toy::<true>, 10_000));
	}

	/// ... and settles exactly that lane when it does.
	#[test]
	fn a_resize_settles_the_lane_the_policy_lists() {
		let mut stack = TieredStack::with(Toy::<true, true>, 10_000);

		over_its_budget(&mut stack);
		stack.resize(1 << 20);

		assert!(!stack.drain_tier_migrations().is_empty(), "the resize did not settle the lane the policy lists");
	}

	/// A fast lane (`PROBATION`, here) in front of a split main: a new key
	/// enters the fast lane, a hit moves it to main's front, the fast lane's
	/// overflow is spliced onto main's BACK as slow, and the victim is main's
	/// tail, else the fast lane's. The fast lane's budget is a carve-out of
	/// 6,000 B of the tier, main's what is left.
	struct ToyCarve;

	impl TierPolicy for ToyCarve {
		type Layout = FastSplit;
		type Ghost = NoGhost;

		const ADMIT: Lane = PROBATION;
		const RESETTLE: &'static [Lane] = &[MAIN, PROBATION];
		const SPILL: Option<Spill> = Some(Spill { to: MAIN, end: End::Back });

		fn is_policy(&self, _policy: &PaperPolicy) -> bool {
			false
		}

		fn touch(s: &mut TieredStack<Self>, key: HashedKey, structural: bool) {
			s.to_front(key, MAIN, structural, Push::IfPromoted);
		}

		fn victim(s: &mut TieredStack<Self>) -> Option<HashedKey> {
			s.evict_tail(MAIN).or_else(|| s.evict_tail(PROBATION))
		}

		fn budget(s: &TieredStack<Self>, lane: Lane) -> CacheSize {
			let (fast, main) = s.carve_budgets(6_000);

			match lane {
				PROBATION => fast,
				_ => main,
			}
		}
	}

	#[test]
	fn the_books_of_a_fast_lane_stack_match_its_queues_after_every_operation() {
		let (demoted, promoted) = books_match_the_queue_after_every_operation(TieredStack::with(ToyCarve, 24_000).with_shared_overhead(40));

		assert!(demoted > 100, "the sequence never demoted ({demoted})");
		assert!(promoted > 0, "the sequence never promoted ({promoted})");
	}

	#[test]
	fn a_fast_lane_spills_its_tail_to_the_back_of_the_lane_the_policy_names() {
		let mut stack = TieredStack::with(ToyCarve, 24_000);

		// Six keys of 1,000 B fill the fast lane past 0.95 x 6,000 B: the oldest spills.
		for key in 1..=6 {
			stack.insert(key, 1_000);
		}

		assert_eq!(stack.drain_tier_migrations(), vec![(1, Tier::Slow)], "the overflow is a demotion, pushed");
		assert_eq!((stack.lane_of(1), stack.tier_of(1)), (Some(MAIN), Some(Tier::Slow)));
		assert_eq!((stack.fast_object_count(), stack.slow_object_count()), (5, 1));

		// A hit moves key 2 to main's front, ahead of the spilled key 1; the next
		// overflow lands behind both.
		stack.update(2);
		stack.insert(7, 1_000);
		stack.insert(8, 1_000);

		assert_eq!(stack.front(MAIN), Some(2));
		assert_eq!(stack.tail(MAIN), Some(3), "a spilled key goes to main's back");
		assert_eq!(stack.fast_bytes_used() + stack.slow_bytes_used(), 8_000);
	}
}
