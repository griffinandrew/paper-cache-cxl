/*
 * Copyright (c) Kia Shakiba
 *
 * This source code is licensed under the GNU AGPLv3 license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! One tier-segmented stack for the three designs that differ only in what a
//! hit and an eviction do: LRU, FIFO and CLOCK.
//!
//! [`ArenaHybridStack<O>`] keeps every tracked key in ONE queue of an
//! [`ArenaQueueSet`], spanning both tiers with the newest end fast, and a
//! `fast_boundary` naming the OLDEST fast key. Everything from the newest end
//! up to and including the boundary is fast -- but for STRUCTURAL keys (S5),
//! which keep their place in the order with tier slow and which every step of
//! the boundary walks over (`prev_fast`) -- so a demotion is one step of a
//! cursor and nothing is searched. The queue set, the boundary, the settle,
//! the byte and object accounting, the metadata reservation and the migration
//! log are the same for the three; an [`ArenaOrder`] supplies what is not:
//!
//! ```text
//!             a hit                         an overwrite             the victim
//!   LRU       move to the front             a hit, after a resize    the tail
//!   FIFO      nothing (the trait default)   resize in place          the tail
//!   CLOCK     set the reference bit         FIFO's, and a hit        the hand: see below
//! ```
//!
//! FIFO is LRU without the reorder, and that is the whole of it: insertion
//! order IS eviction order, so `hit` is the trait's no-op -- overriding it
//! would silently turn the design into LRU -- and an overwrite of a tracked
//! key only resizes, re-settling if the key is fast.
//!
//! CLOCK is FIFO plus a reference bit, and the whole of CLOCK is what the
//! eviction path does when it finds that bit set. The semantics are the flat
//! [`ClockCompactStack`](super::clock_compact_stack::ClockCompactStack)'s,
//! verbatim, not a textbook's:
//!
//! ```text
//!   insert, new key       push_front, bit CLEAR
//!   insert, existing key  a hit: the bit is SET, and nothing moves
//!   hit                   the bit is SET, and nothing moves
//!   evict                 pop_back; bit clear -> evict it; bit set -> clear
//!                         the bit and PUSH IT TO THE FRONT, then look again
//! ```
//!
//! That last line is the choice that matters and the easy one to get wrong.
//! Recycling the second-chance entry to the FRONT is CLOCK: in the
//! circular-buffer rendering the hand passes the entry and will not reach it
//! again until a full revolution, which in list terms is the head. Leaving the
//! entry WHERE IT IS and moving a separate hand past it is SIEVE -- a
//! different policy with a different eviction order, and not what this tree's
//! flat clock stack implements. The second chance is LRU's `touch_to_front`:
//! relink to the front, promote, settle. Promotion is not optional: the fast
//! set has to stay a contiguous prefix of the queue, so a key moved into it
//! must become fast in the same operation. CLOCK differs from LRU not in what
//! that operation does but in WHEN it happens -- LRU runs it on every hit,
//! CLOCK only for the keys a hit saved from the hand.
//!
//! The merged object store is the other statement of the same three policies
//! (`MergedOrder::{Lru, Fifo, Clock}`), a sharded slab with its own boundary
//! and settle; `merged_stack`'s order-fidelity tests replay both against each
//! other, key for key, and this stack is their reference.
//!
//! # Cost
//!
//! One [`NodePayload`] per key in the arena: 40 B/object of eviction stack, the
//! same for all three -- the reference bit rides in `freq`, as `freq != 0`
//! (the idiom the S3-FIFO family uses), so CLOCK costs not a byte more than
//! FIFO. `overhead.rs` holds the measurement.

use std::marker::PhantomData;

use crate::{
	object::ObjectSize,
	worker::policy::policy_stack::{
		arena_queue_set::{ArenaQueueSet, NodePayload}, clock_hand_budget, narrow_resident,
		drain_target, CacheSize, HashedKey, PolicyStack, Tier, Placement, SetEvent,
		fast_at_or_before, placed, prev_fast,
	},
	PaperPolicy,
};

/// The single queue, in the shared queue set's slot 0.
const Q: usize = 0;

/// A queue rule: the parts of a tiered arena stack the order decides.
///
/// Implemented by the zero-sized [`LruOrder`], [`FifoOrder`] and
/// [`ClockOrder`]; everything else is [`ArenaHybridStack`]'s.
pub trait ArenaOrder: Sized + Send + 'static {
	/// The policy this order implements.
	fn is_policy(policy: &PaperPolicy) -> bool;

	/// A `Set` of a key the stack already tracks. `structural`: the new value
	/// is larger than an empty fast tier (the client's flag or the stack's own
	/// check). The stack has already narrowed `dram_resident`; the placement it
	/// answers with is `placed(structural)`, whatever this does.
	fn overwrite(
		stack: &mut ArenaHybridStack<Self>,
		key: HashedKey,
		size: ObjectSize,
		dram_resident: u8,
		structural: bool,
	);

	/// A hit on `key`. The default does nothing, which is FIFO's rule.
	fn hit(_stack: &mut ArenaHybridStack<Self>, _key: HashedKey) {}

	/// Removes and returns the key this order evicts next. The default takes
	/// the queue's tail.
	fn evict(stack: &mut ArenaHybridStack<Self>) -> Option<HashedKey> {
		let key = stack.list.back(Q)?;

		stack.evict_at_tail(key)
	}
}

/// LRU: the queue is a recency order.
pub struct LruOrder;

/// FIFO: the queue is an insertion order, and a hit does not reorder.
pub struct FifoOrder;

/// CLOCK: FIFO plus a reference bit, the second chance recycling to the front.
pub struct ClockOrder;

/// `PaperPolicy::LruCompactHybrid`.
pub type LruCompactHybridStack = ArenaHybridStack<LruOrder>;

/// `PaperPolicy::FifoCompactHybrid`.
pub type FifoCompactHybridStack = ArenaHybridStack<FifoOrder>;

/// `PaperPolicy::ClockCompactHybrid`.
pub type ClockCompactHybridStack = ArenaHybridStack<ClockOrder>;

/// Per-key bookkeeping is [`NodePayload`], the one node every policy shares.
/// This stack reads `tier`, `size` and `dram_resident` (CLOCK also `freq`, as
/// its reference bit); `ts` and `queue` belong to other policies and stay at
/// their defaults here.
pub struct ArenaHybridStack<O: ArenaOrder> {
	list: ArenaQueueSet<NodePayload>,

	fast_capacity: CacheSize,
	fast_used: CacheSize,
	slow_used: CacheSize,

	shared_overhead: CacheSize,

	fast_count: usize,

	/// The oldest FAST key: everything from the newest end up to it is fast
	/// (S5: but for structural keys, which keep their place with tier slow and
	/// which every step of this cursor walks over, `prev_fast`), everything
	/// after it is slow.
	fast_boundary: Option<HashedKey>,

	migrations: Vec<(HashedKey, Tier)>,

	/// S5: the measured M the policy worker pushed (`set_dram_metadata`),
	/// reserved instead of `len x shared_overhead`; `None` under the
	/// per-object model.
	measured: Option<CacheSize>,

	order: PhantomData<O>,
}

impl<O: ArenaOrder> ArenaHybridStack<O> {
	pub fn new(fast_capacity: CacheSize) -> Self {
		ArenaHybridStack {
			list: ArenaQueueSet::default(),
			fast_capacity,
			fast_used: 0,
			slow_used: 0,
			shared_overhead: 0,
			fast_count: 0,
			fast_boundary: None,
			migrations: Vec::new(),
			measured: None,
			order: PhantomData,
		}
	}

	/// Per-object DRAM reserved from the fast tier for shared metadata.
	pub fn with_shared_overhead(mut self, overhead: CacheSize) -> Self {
		self.shared_overhead = overhead;

		self
	}

	pub fn tier_of(&self, key: HashedKey) -> Option<Tier> {
		self.list.payload(key).and_then(|p| p.tier)
	}

	/// Metadata reservation for EVERY tracked key, fast or slow: a demotion
	/// moves the value and leaves the key's row, stack node and header in
	/// DRAM. See `PolicyStack::dram_reserved_bytes` for the rule, and for why
	/// a reservation at or over `fast_capacity` is left to saturate.
	fn reserved_overhead(&self) -> CacheSize {
		self.measured.unwrap_or(self.list.len() as CacheSize * self.shared_overhead)
	}

	/// This stack's eff (S5): the whole fast tier's budget for values, its
	/// settle's figure before the drain target.
	fn own_eff(&self) -> CacheSize {
		self.fast_capacity.saturating_sub(self.reserved_overhead())
	}

	/// Whether a value of `migrating` bytes is STRUCTURAL (S5): larger than an
	/// empty fast tier. Such a key is placed slow and never promoted while it
	/// stays that large. The stack's own check beside the client's flag, so a
	/// key the client placed normally just before eff moved is placed as the
	/// stack's own promotions would place it.
	fn structural(&self, migrating: CacheSize) -> bool {
		migrating > self.own_eff()
	}

	fn resize_key(&mut self, key: HashedKey, new_size: ObjectSize, new_resident: u8) {
		let Some(slot) = self.list.payload_mut(key) else { return };

		let old_migrating = slot.migrating();
		slot.size = new_size;
		slot.dram_resident = new_resident;
		let delta = slot.migrating() as i64 - old_migrating as i64;
		let tier = slot.tier;

		match tier {
			Some(Tier::Fast) => {
				self.fast_used = (self.fast_used as i64 + delta).max(0) as CacheSize;
			},

			Some(Tier::Slow) => {
				self.slow_used = (self.slow_used as i64 + delta).max(0) as CacheSize;
			},

			// The shared node makes `tier` optional because the 2Q and S3-FIFO
			// families legitimately have a queue with no tier of its own. This
			// stack always records one, so this arm is unreachable -- and it is
			// spelled out rather than papered over with `unwrap_or`, which
			// would silently pick a tier if that ever stopped being true.
			None => {},
		}
	}

	/// Takes a departing key's `size` bytes off the books of the tier it was
	/// in.
	fn forget(&mut self, tier: Option<Tier>, size: CacheSize) {
		match tier {
			Some(Tier::Fast) => {
				self.fast_used = self.fast_used.saturating_sub(size);
				self.fast_count = self.fast_count.saturating_sub(1);
			},

			Some(Tier::Slow) => {
				self.slow_used = self.slow_used.saturating_sub(size);
			},

			// Unreachable: see `resize_key`.
			None => {},
		}
	}

	/// Books a fast key as slow: its `size` bytes move from the fast books to
	/// the slow ones. The callers step the boundary and queue the migration.
	fn demote_books(&mut self, key: HashedKey, size: CacheSize) {
		if let Some(slot) = self.list.payload_mut(key) {
			slot.tier = Some(Tier::Slow);
		}

		self.fast_used = self.fast_used.saturating_sub(size);
		self.fast_count = self.fast_count.saturating_sub(1);
		self.slow_used += size;
	}

	/// A `Set`, with the client's placement (S5). An existing key is the
	/// order's to handle (`ArenaOrder::overwrite`); a new one is admitted at
	/// the front, fast -- or, when STRUCTURAL (the client's flag, or this
	/// stack's own check), slow: built slow, charged slow, never the boundary,
	/// nothing pushed (the reconcile's new-key rule covers a stale entry of the
	/// key), nothing settled (no fast byte moved). Returns the placement
	/// applied.
	fn insert_with(&mut self, key: HashedKey, size: ObjectSize, dram_resident: ObjectSize, placement: Placement) -> Placement {
		let dram_resident = narrow_resident(dram_resident);
		let migrating = (size as CacheSize).saturating_sub(dram_resident as CacheSize);
		let structural = placement == Placement::Structural || self.structural(migrating);

		if self.list.contains(key) {
			O::overwrite(self, key, size, dram_resident, structural);

			return placed(structural);
		}

		let tier = match structural {
			true => Tier::Slow,
			false => Tier::Fast,
		};

		// A brand-new key enters UNREFERENCED (`freq` 0), so one pass of a
		// CLOCK hand can evict it.
		self.list.push_front(Q, key, NodePayload {
			size,
			dram_resident,
			tier: Some(tier),
			freq: 0,
			ts: 0,
			queue: 0,
		});

		if structural {
			self.slow_used += migrating;

			return Placement::Structural;
		}

		self.fast_used += migrating;
		self.fast_count += 1;

		if self.fast_boundary.is_none() {
			self.fast_boundary = Some(key);
		}

		self.settle_fast_tier();

		Placement::Normal
	}

	/// Moves `key` to the front and makes it fast -- LRU's hit, and CLOCK's
	/// second chance -- and, since S5, the structural rule: a STRUCTURAL key
	/// (its value larger than an empty fast tier) moves to the front all the
	/// same -- its place in the order -- but with tier slow. A slow one is not
	/// promoted; a fast one (an overwrite with a value too large, or an eff
	/// that shrank) leaves the fast set, pushed `(key, Slow)`: its placement
	/// changed, and a promotion of its old value may still be in flight, which
	/// this entry, behind it on the key's FIFO consumer, undoes.
	fn touch_to_front(&mut self, key: HashedKey, structural: bool) {
		let previous_tier = self.list.payload(key).and_then(|p| p.tier);

		let already_at_front = self.list.front(Q) == Some(key);
		let is_boundary = self.fast_boundary == Some(key);

		// Read the neighbour BEFORE moving: once the key is at the front its
		// predecessor is gone, and the boundary has to step back to whatever
		// fast key was in front of it.
		let new_boundary_if_moved = if is_boundary && !already_at_front {
			prev_fast(&self.list, key)
		} else {
			None
		};

		self.list.move_front(Q, key);

		if is_boundary && !already_at_front {
			self.fast_boundary = new_boundary_if_moved;
		}

		if structural {
			if previous_tier == Some(Tier::Fast) {
				let size = self.list.payload(key).map(|p| p.migrating()).unwrap_or(0);

				self.demote_books(key, size);

				// Still the boundary only if it was already at the front: then
				// it was the one fast key, and none is left.
				if self.fast_boundary == Some(key) {
					self.fast_boundary = None;
				}

				self.migrations.push((key, Tier::Slow));
			}

			self.settle_fast_tier();
			return;
		}

		let mut promoted = false;

		if previous_tier != Some(Tier::Fast) {
			if previous_tier == Some(Tier::Slow) {
				let size = self.list.payload(key).map(|p| p.migrating()).unwrap_or(0);
				self.slow_used = self.slow_used.saturating_sub(size);
				self.fast_used += size;
				self.fast_count += 1;
				promoted = true;
			}

			if let Some(slot) = self.list.payload_mut(key) {
				slot.tier = Some(Tier::Fast);
			}
		}

		// The key is fast, at the front; with no fast key in front of it --
		// none at all, or (S5) only structural ones -- it is the boundary.
		if self.fast_boundary.is_none() {
			self.fast_boundary = Some(key);
		}

		self.settle_fast_tier();

		// Pushed after settling and guarded on the key still being fast: a
		// tight budget can demote it straight back out within the same settle,
		// in which case that call already pushed the correct final entry.
		//
		// Pushed even when the bytes are already fast, as they are on a re-`set`
		// of a slow key: the API thread built the new value in DRAM before this
		// worker saw the event, so the consumer will decline the entry. It still
		// has to be queued. Queued migrations carry no identity --
		// `apply_migration` acts on whatever object holds the key when it
		// dequeues -- so a demotion this stack decided for the old object, still
		// queued when the new one replaced it, demotes the new one; this entry,
		// behind it on the key's FIFO consumer, restores it. Pinned by
		// `overwrite_tests::an_overwrite_is_repromoted_after_a_stale_demotion`.
		if promoted && self.list.payload(key).and_then(|p| p.tier) == Some(Tier::Fast) {
			self.migrations.push((key, Tier::Fast));
		}
	}

	/// An overwrite that leaves the key WHERE IT IS in the queue -- FIFO's and
	/// CLOCK's: it is resized, taken out of the fast set in place when its new
	/// value is STRUCTURAL and it was fast (`demote_in_place`), and re-settled
	/// only if it was fast and resized, since only then can the resize have
	/// pushed the fast tier over its budget.
	fn overwrite_in_place(&mut self, key: HashedKey, size: ObjectSize, dram_resident: u8, structural: bool) {
		let Some(payload) = self.list.payload(key) else { return };

		let tier = payload.tier;
		let resized = payload.size != size;

		if resized {
			self.resize_key(key, size, dram_resident);
		}

		if structural && tier == Some(Tier::Fast) {
			self.demote_in_place(key);
		}

		if resized && tier == Some(Tier::Fast) {
			self.settle_fast_tier();
		}
	}

	/// Takes a FAST key out of the fast set IN PLACE (S5): an overwrite with a
	/// value larger than an empty fast tier, in an order that keeps an
	/// overwritten key where it is. Pushed `(key, Slow)`: its placement
	/// changed.
	fn demote_in_place(&mut self, key: HashedKey) {
		let Some(payload) = self.list.payload(key) else { return };
		let size = payload.migrating();

		if self.fast_boundary == Some(key) {
			self.fast_boundary = prev_fast(&self.list, key);
		}

		self.demote_books(key, size);

		self.migrations.push((key, Tier::Slow));
	}

	/// Demotes from the tier boundary until `fast_used` is back within the
	/// effective budget. The victim is always `fast_boundary` -- the oldest
	/// fast key -- so nothing is searched.
	fn settle_fast_tier(&mut self) {
		let effective = self.fast_capacity.saturating_sub(self.reserved_overhead());
		let target = drain_target::bytes(effective);

		while self.fast_used > target {
			let Some(demote_key) = self.fast_boundary else { break };
			let size = self.list.payload(demote_key).map(|p| p.migrating()).unwrap_or(0);
			let new_boundary = prev_fast(&self.list, demote_key);

			self.demote_books(demote_key, size);
			self.fast_boundary = new_boundary;

			self.migrations.push((demote_key, Tier::Slow));
		}
	}

	/// Evicts `key`, the queue's tail: unlinks it and takes its bytes off its
	/// tier's books.
	fn evict_at_tail(&mut self, key: HashedKey) -> Option<HashedKey> {
		let slot = self.list.remove(Q, key)?;
		let size = slot.migrating();

		self.forget(slot.tier, size);

		// The boundary was the tail: the nearest fast key from the new tail
		// (S5: past any structural ones).
		if slot.tier == Some(Tier::Fast) && self.fast_boundary == Some(key) {
			self.fast_boundary = fast_at_or_before(&self.list, self.list.back(Q));
		}

		Some(key)
	}
}

/// CLOCK's reference bit, as `freq != 0`.
impl ArenaHybridStack<ClockOrder> {
	fn referenced(&self, key: HashedKey) -> bool {
		self.list.payload(key).map(|p| p.freq != 0).unwrap_or(false)
	}

	fn set_referenced(&mut self, key: HashedKey, on: bool) {
		if let Some(slot) = self.list.payload_mut(key) {
			slot.freq = u32::from(on);
		}
	}
}

impl ArenaOrder for LruOrder {
	fn is_policy(policy: &PaperPolicy) -> bool {
		matches!(policy, PaperPolicy::LruCompactHybrid)
	}

	/// An existing key is resized and then hit: moved to the front.
	fn overwrite(stack: &mut ArenaHybridStack<Self>, key: HashedKey, size: ObjectSize, dram_resident: u8, structural: bool) {
		stack.resize_key(key, size, dram_resident);
		stack.touch_to_front(key, structural);
	}

	fn hit(stack: &mut ArenaHybridStack<Self>, key: HashedKey) {
		if let Some(payload) = stack.list.payload(key) {
			let structural = stack.structural(payload.migrating());

			stack.touch_to_front(key, structural);
		}
	}
}

impl ArenaOrder for FifoOrder {
	fn is_policy(policy: &PaperPolicy) -> bool {
		matches!(policy, PaperPolicy::FifoCompactHybrid)
	}

	/// An existing key is resized in place and NOT moved -- insertion order is
	/// eviction order.
	fn overwrite(stack: &mut ArenaHybridStack<Self>, key: HashedKey, size: ObjectSize, dram_resident: u8, structural: bool) {
		stack.overwrite_in_place(key, size, dram_resident, structural);
	}
}

impl ArenaOrder for ClockOrder {
	fn is_policy(policy: &PaperPolicy) -> bool {
		matches!(policy, PaperPolicy::ClockCompactHybrid)
	}

	/// An existing key is resized in place and NOT moved -- insertion order is
	/// the queue order -- but it IS referenced: `ClockCompactStack`'s `insert`
	/// forwards an existing key straight to `update`, so writing a key earns it
	/// a second chance exactly as reading it does.
	fn overwrite(stack: &mut ArenaHybridStack<Self>, key: HashedKey, size: ObjectSize, dram_resident: u8, structural: bool) {
		stack.set_referenced(key, true);
		stack.overwrite_in_place(key, size, dram_resident, structural);
	}

	/// A cache hit: set the reference bit and nothing else.
	///
	/// The queue is not touched. That is the half of CLOCK that makes it cheap,
	/// and the half `MergedOrder::Clock` turns into a relaxed atomic store
	/// under a read lock.
	fn hit(stack: &mut ArenaHybridStack<Self>, key: HashedKey) {
		stack.set_referenced(key, true);
	}

	/// The hand. Walks from the oldest end, giving a second chance to every
	/// referenced key it passes, and evicts the first unreferenced one.
	///
	/// `ClockCompactStack::evict_one`'s loop, with the tier accounting the flat
	/// stack has nothing to do. Terminates in at most `len()` second chances:
	/// each one clears a bit, and nothing in the loop sets one.
	///
	/// Capped all the same, at `clock_hand_budget(len)` second chances, after
	/// which the tail is evicted whatever its bit -- the cap the merged store's
	/// hand has, so the two stop at the same point. By the argument above it
	/// cannot fire; it changes no eviction.
	fn evict(stack: &mut ArenaHybridStack<Self>) -> Option<HashedKey> {
		let mut budget = clock_hand_budget(stack.list.len());

		loop {
			let key = stack.list.back(Q)?;

			if budget > 0 && stack.referenced(key) {
				budget -= 1;

				// Cleared and recycled to the FRONT -- CLOCK, not SIEVE. See
				// the module doc. Reached only here, and only for a key whose
				// bit the hand has just cleared: that is the whole difference
				// between this policy and LRU, the same relink from the
				// eviction path rather than from every hit.
				stack.set_referenced(key, false);

				let structural = stack.list.payload(key).is_some_and(|p| stack.structural(p.migrating()));

				stack.touch_to_front(key, structural);

				continue;
			}

			return stack.evict_at_tail(key);
		}
	}
}

impl<O: ArenaOrder> PolicyStack for ArenaHybridStack<O> {
	fn is_policy(&self, policy: &PaperPolicy) -> bool {
		O::is_policy(policy)
	}

	fn len(&self) -> usize {
		self.list.len()
	}

	fn contains(&self, key: HashedKey) -> bool {
		self.list.contains(key)
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
		self.settle_fast_tier();
	}

	fn update(&mut self, key: HashedKey) {
		O::hit(self, key);
	}

	fn remove(&mut self, key: HashedKey) {
		let Some(slot) = self.list.payload(key) else { return };
		let size = slot.migrating();
		let tier = slot.tier;

		let new_boundary_if_needed = if tier == Some(Tier::Fast) && self.fast_boundary == Some(key) {
			prev_fast(&self.list, key)
		} else {
			None
		};

		self.list.remove(Q, key);
		self.forget(tier, size);

		if tier == Some(Tier::Fast) && self.fast_boundary == Some(key) {
			self.fast_boundary = new_boundary_if_needed;
		}
	}

	fn clear(&mut self) {
		self.list.clear();

		self.fast_used = 0;
		self.slow_used = 0;
		self.fast_count = 0;
		self.fast_boundary = None;
		self.migrations.clear();
	}

	fn evict_one(&mut self) -> Option<HashedKey> {
		O::evict(self)
	}

	fn resize_fast_tier(&mut self, size: CacheSize) {
		self.fast_capacity = size;
		self.settle_fast_tier();
	}

	/// `tier_of`: every demotion is pushed by the settle or by a structural
	/// overwrite, and every promotion (LRU's hit, CLOCK's second chance) after
	/// its settle, guarded on the key still being fast -- also on an LRU re-set,
	/// where the bytes are already fast, to follow a stale queued demotion. A
	/// FIFO or CLOCK re-set moves nothing.
	/// See `PolicyStack::placement_of`.
	fn placement_of(&self, key: HashedKey) -> Option<Tier> {
		self.tier_of(key)
	}

	fn drain_tier_migrations(&mut self) -> Vec<(HashedKey, Tier)> {
		std::mem::take(&mut self.migrations)
	}

	fn structure_bytes(&self) -> Option<crate::meta::NodeBytes> {
		Some(crate::meta::NodeBytes::stack(self.list.allocated_bytes()))
	}

	fn dram_reserved_bytes(&self) -> CacheSize {
		self.reserved_overhead()
	}

	fn fast_bytes_used(&self) -> CacheSize {
		self.fast_used
	}

	fn slow_bytes_used(&self) -> CacheSize {
		self.slow_used
	}

	fn fast_object_count(&self) -> usize {
		self.fast_count
	}

	fn slow_object_count(&self) -> usize {
		self.list.len().saturating_sub(self.fast_count)
	}
}

/// The re-promotion `touch_to_front` queues on a re-`set` of a slow key looks
/// redundant -- the bytes are already in DRAM -- and is not. This replays the
/// one order in which it is load-bearing, a migration at a time, the way the
/// key's FIFO consumer would apply them.
///
/// Gated on `hybrid_cache_common` for `new_hybrid_object_map`, so it runs over
/// whichever object map the build selects; the stack is this one in every
/// build.
#[cfg(all(test, feature = "hybrid_cache_common"))]
mod overwrite_tests {
	use super::*;
	use crate::{object::Object, worker::policy::migration_queue::apply_migration};

	// `MergedStore` has `insert` and `get_ref` of its own; the other maps take
	// them from the trait.
	#[cfg(not(feature = "merged_object_store"))]
	use crate::object_store::ObjectStore;

	const K: HashedKey = 7;
	const A: HashedKey = 8;
	const SIZE: ObjectSize = 1_000;

	fn fresh(key: HashedKey) -> Object<u32, crate::TieredBuffer> {
		Object::new_in(key as u32, &[0xA5; 64], Tier::Fast, None)
	}

	/// K is demoted as the LRU tail, and that demotion is decided but still
	/// queued when a re-`set` replaces K with a value built in DRAM. The
	/// queued demotion then lands on the NEW value, because migrations carry
	/// no identity; the re-promotion queued behind it must bring it back, or
	/// K's fresh value sits in the slow tier while the stack counts it fast.
	#[test]
	fn an_overwrite_is_repromoted_after_a_stale_demotion() {
		// `apply_migration` bumps the process-wide migration counters that the
		// queue tests assert exact deltas on, so this runs under their lock.
		let _serialised = crate::worker::policy::migration_test_lock::lock();

		let objects: crate::ObjectMapRef<u32, crate::TieredBuffer> = crate::new_hybrid_object_map();

		// Room for one SIZE-byte object: the second admission demotes the first.
		let mut stack = LruCompactHybridStack::new(2 * SIZE as CacheSize - 1).with_shared_overhead(0);

		for key in [K, A] {
			objects.insert(key, fresh(key)); // `set`, on the API thread
			stack.insert_resident(key, SIZE, 0); // its `Set` event, on the worker
		}

		assert_eq!(stack.tier_of(K), Some(Tier::Slow), "K should be the demoted LRU tail");

		let mut queue = stack.drain_tier_migrations();
		assert_eq!(queue, vec![(K, Tier::Slow)], "the demotion is decided, not yet applied");

		// The overwrite: an LRU `set` builds the new value in DRAM, then the
		// worker handles its `Set`.
		objects.insert(K, fresh(K));
		stack.insert_resident(K, SIZE, 0);
		queue.extend(stack.drain_tier_migrations());

		for (key, tier) in queue {
			apply_migration(&objects, key, tier);
		}

		let physical = objects.get_ref(&K).map(|object| object.value().tier());

		assert_eq!(stack.tier_of(K), Some(Tier::Fast), "an overwrite makes K the most recent key");
		assert_eq!(
			physical,
			Some(Tier::Fast),
			"K's new value was left in the slow tier while the stack counts it fast",
		);
	}
}

/// These stacks grow dynamically and must not allocate from the cache budget
/// at construction. An eager reservation sized from capacity was removed: at
/// the standing 4 GiB fast tier it reserved 4.19M slots, which is 16x what the
/// 16.5 KB-object eval trace can hold there, while on the real Twitter traces
/// (~180 B objects, ~11.4M resident) it was too small to prevent doubling
/// anyway. It also meant the shipped stack allocated something other than the
/// per-object cost that was measured and reported, since the measurement runs
/// with the reservation disabled.
///
/// The old tests asserted only `len() == 0`, which is true either way; neither
/// observed capacity.
#[cfg(test)]
mod growth_tests {
	use super::*;

	fn construction_does_not_allocate_from_the_budget<O: ArenaOrder>() {
		for budget in [u64::MAX / 4, 4 * 1024 * 1024 * 1024, 1_024] {
			let stack = ArenaHybridStack::<O>::new(budget).with_shared_overhead(224);
			assert_eq!(stack.len(), 0, "budget {budget}: nothing is tracked yet");
			assert_eq!(
				stack.list.slab_capacity(),
				0,
				"budget {budget}: the slab must be empty at construction -- a stack \
				 that pre-sizes from capacity allocates for objects that may never \
				 arrive, and reserves a different amount than it was measured at",
			);
		}
	}

	/// The counterpart: growth still happens, it is just demand-driven.
	fn the_slab_grows_on_demand<O: ArenaOrder>() {
		let mut stack = ArenaHybridStack::<O>::new(u64::MAX / 4).with_shared_overhead(224);
		for i in 0..1_000u64 {
			stack.insert(i, 64);
		}
		assert_eq!(stack.len(), 1_000);
		assert!(
			stack.list.slab_capacity() >= 1_000,
			"the slab must have grown to hold what was inserted",
		);
	}

	macro_rules! per_order {
		($($order:ident: $O:ty),*) => {$(
			mod $order {
				use super::*;

				#[test]
				fn construction_does_not_allocate_from_the_budget() {
					super::construction_does_not_allocate_from_the_budget::<$O>();
				}

				#[test]
				fn the_slab_grows_on_demand() {
					super::the_slab_grows_on_demand::<$O>();
				}
			}
		)*};
	}

	per_order!(lru: LruOrder, fifo: FifoOrder, clock: ClockOrder);
}

/// The fast tier is charged the metadata of EVERY tracked object, not only the
/// fast ones: a demotion moves the value and leaves the row, the stack node and
/// the header in DRAM. A reservation of `fast_object_count() x shared_overhead`
/// understates DRAM by the whole slow tier's metadata, and both tests fail
/// under it.
#[cfg(test)]
mod reservation_tests {
	use super::*;

	const FAST_CAPACITY: CacheSize = 10_000;
	const OVERHEAD: CacheSize = 200;
	const SIZE: ObjectSize = 1_000;
	const N: HashedKey = 20;

	/// 20 x 200 B of metadata leaves 6_000 B for values: five of the twenty
	/// objects stay fast. Charging the fast ones alone would keep eight.
	fn slow_objects_are_charged_against_the_fast_tier<O: ArenaOrder>() {
		let mut stack = ArenaHybridStack::<O>::new(FAST_CAPACITY).with_shared_overhead(OVERHEAD);

		for key in 1..=N {
			stack.insert(key, SIZE);
		}

		assert!(
			stack.slow_object_count() > 0 && stack.fast_object_count() > 0,
			"the fixture must leave objects in both tiers to tell the rules apart",
		);
		assert_eq!(
			stack.dram_reserved_bytes(),
			N * OVERHEAD,
			"all {N} tracked objects keep their metadata in DRAM, but the reservation \
			 covers {} of them ({} fast, {} slow)",
			stack.dram_reserved_bytes() / OVERHEAD,
			stack.fast_object_count(),
			stack.slow_object_count(),
		);

		let effective = FAST_CAPACITY - N * OVERHEAD;

		assert!(
			stack.fast_bytes_used() <= drain_target::bytes(effective),
			"{} B of values are fast against {effective} B left once all {N} objects' \
			 metadata is reserved",
			stack.fast_bytes_used(),
		);
	}

	/// Metadata at, then over, the whole fast tier: the effective budget
	/// saturates at 0, every value is slow, and the stack keeps working --
	/// hits, removal and eviction included. The reservation is reported as it
	/// is, above `fast_capacity`, rather than clipped to it.
	///
	/// Key 2 is the victim after key 1's removal in all three orders: LRU's
	/// hits (in key order) leave 2 at the tail, FIFO's tail is insertion order,
	/// and CLOCK's hand recycles every referenced key to the front in turn.
	fn metadata_at_or_over_the_fast_tier_leaves_no_room_for_values<O: ArenaOrder>() {
		const OVERHEAD: CacheSize = 1_000;
		const SIZE: ObjectSize = 100;

		for n in [10, 12] {
			let mut stack = ArenaHybridStack::<O>::new(FAST_CAPACITY).with_shared_overhead(OVERHEAD);

			for key in 1..=n {
				stack.insert(key, SIZE);
			}

			assert_eq!(
				stack.dram_reserved_bytes(),
				n * OVERHEAD,
				"{n} objects: the reservation must be every object's metadata, \
				 even past the fast tier",
			);
			assert_eq!(
				stack.fast_object_count(),
				0,
				"{n} objects: {} B of metadata on a {FAST_CAPACITY} B fast tier leaves \
				 no room for a value, yet {} are fast",
				n * OVERHEAD,
				stack.fast_object_count(),
			);
			assert_eq!(stack.fast_bytes_used(), 0);
			assert_eq!(stack.slow_object_count(), n as usize);

			for key in 1..=n {
				stack.update(key);
			}

			assert_eq!(stack.fast_object_count(), 0, "a hit cannot find fast room that is not there");

			stack.remove(1);
			assert_eq!(stack.evict_one(), Some(2), "the order still holds");
			assert_eq!(stack.len(), n as usize - 2);
			assert_eq!(stack.dram_reserved_bytes(), (n - 2) * OVERHEAD);
		}
	}

	macro_rules! per_order {
		($($order:ident: $O:ty),*) => {$(
			mod $order {
				use super::*;

				#[test]
				fn slow_objects_are_charged_against_the_fast_tier() {
					super::slow_objects_are_charged_against_the_fast_tier::<$O>();
				}

				#[test]
				fn metadata_at_or_over_the_fast_tier_leaves_no_room_for_values() {
					super::metadata_at_or_over_the_fast_tier_leaves_no_room_for_values::<$O>();
				}
			}
		)*};
	}

	per_order!(lru: LruOrder, fifo: FifoOrder, clock: ClockOrder);
}

/// The stack's books against its queue, after EVERY operation of a long random
/// sequence: the object and byte counts of each tier are what walking the
/// queue finds, and the boundary is the oldest FAST key, with nothing but slow
/// keys behind it. The sequence mixes every way a key enters, moves and leaves
/// -- sets of new and tracked keys (some larger than an empty fast tier, so
/// structural), hits, removals, evictions, a shrinking and growing fast tier,
/// a pushed measured M -- against a tight tier, so demotions, promotions and
/// second chances all happen. What the eviction-order tests cannot see: a
/// count or a byte total that drifts while the order stays right.
#[cfg(test)]
mod invariant_tests {
	use super::*;

	/// Walks the queue from the newest end and holds the books to it.
	fn audit<O: ArenaOrder>(stack: &ArenaHybridStack<O>, step: usize) {
		let (mut fast, mut slow) = (0usize, 0usize);
		let (mut fast_bytes, mut slow_bytes) = (0 as CacheSize, 0 as CacheSize);
		let mut oldest_fast = None;
		let mut next = stack.list.front(Q);
		let mut walked = 0usize;

		while let Some(key) = next {
			let payload = stack.list.payload(key).expect("a queued key has a payload");

			match payload.tier {
				Some(Tier::Fast) => {
					fast += 1;
					fast_bytes += payload.migrating();
					oldest_fast = Some(key);
				},

				Some(Tier::Slow) => {
					slow += 1;
					slow_bytes += payload.migrating();
				},

				None => panic!("step {step}: key {key} has no tier"),
			}

			walked += 1;
			next = stack.list.after(key);
		}

		assert_eq!(walked, stack.len(), "step {step}: the queue's length");
		assert_eq!((fast, slow), (stack.fast_object_count(), stack.slow_object_count()), "step {step}: object counts");
		assert_eq!((fast_bytes, slow_bytes), (stack.fast_bytes_used(), stack.slow_bytes_used()), "step {step}: byte totals");
		assert_eq!(stack.fast_boundary, oldest_fast, "step {step}: the boundary is the oldest fast key");
	}

	fn books_match_the_queue_after_every_operation<O: ArenaOrder>() {
		let mut stack = ArenaHybridStack::<O>::new(24_000).with_shared_overhead(40);
		let mut x: u64 = 0x9E37_79B9_7F4A_7C15;
		let mut demoted = 0usize;
		let mut promoted = 0usize;

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

		assert!(demoted > 100, "the sequence never demoted ({demoted})");

		// FIFO never promotes: a hit moves nothing.
		if O::is_policy(&PaperPolicy::LruCompactHybrid) || O::is_policy(&PaperPolicy::ClockCompactHybrid) {
			assert!(promoted > 10, "the sequence never promoted ({promoted})");
		}
	}

	macro_rules! per_order {
		($($order:ident: $O:ty),*) => {$(
			#[test]
			fn $order() {
				books_match_the_queue_after_every_operation::<$O>();
			}
		)*};
	}

	per_order!(lru: LruOrder, fifo: FifoOrder, clock: ClockOrder);
}

/// Fidelity against the FLAT stacks, which are the references the three orders
/// are defined by.
///
/// Tiering cannot reorder the queue -- demotion only steps a cursor along it --
/// so a tier-segmented stack must evict in exactly the order the untiered one
/// does, given the same accesses. `(key, is_update)` is the shape the flat
/// stacks' own fidelity tests use, and `is_update == false` on a key already
/// present is an OVERWRITE, which each order must treat as its flat stack does:
/// a hit for LRU and CLOCK, nothing for FIFO.
#[cfg(test)]
mod flat_fidelity {
	use super::*;

	/// Big enough that nothing is ever demoted, so a test with it isolates the
	/// queue order from the tier boundary.
	pub const HUGE: CacheSize = 1 << 40;

	/// The fast tier biting: 40_000 B holds 78 of the 500 keys.
	pub const TIGHT: CacheSize = 40_000;

	pub fn drain(stack: &mut dyn PolicyStack) -> Vec<HashedKey> {
		let mut out = Vec::new();

		while let Some(k) = stack.evict_one() {
			out.push(k);
		}

		out
	}

	pub fn replay(
		flat: &mut dyn PolicyStack,
		hybrid: &mut dyn PolicyStack,
		ops: &[(HashedKey, bool)],
	) -> (Vec<HashedKey>, Vec<HashedKey>) {
		for &(key, is_update) in ops {
			if is_update {
				flat.update(key);
				hybrid.update(key);
			} else {
				flat.insert(key, 512);
				hybrid.insert(key, 512);
			}

			assert_eq!(flat.len(), hybrid.len(), "len diverged at key {key}");
		}

		(drain(flat), drain(hybrid))
	}

	pub fn skewed_ops() -> Vec<(HashedKey, bool)> {
		let mut ops = Vec::new();
		let mut x: u64 = 0x243F_6A88_85A3_08D3;

		for i in 0..40_000u64 {
			x ^= x << 13;
			x ^= x >> 7;
			x ^= x << 17;
			let u = (x >> 11) as f64 / (1u64 << 53) as f64;
			let key = ((u * u * 500.0) as u64) + 1;
			ops.push((key, i % 3 == 0));
		}

		ops
	}
}

#[cfg(test)]
mod flat_lru_fidelity {
	use super::*;
	use super::flat_fidelity::*;
	use super::super::lru_compact_stack::LruCompactStack;

	fn replay_at(capacity: CacheSize) -> (Vec<HashedKey>, Vec<HashedKey>) {
		replay(&mut LruCompactStack::default(), &mut LruCompactHybridStack::new(capacity), &skewed_ops())
	}

	#[test]
	fn evicts_in_the_same_order_as_the_flat_lru_stack() {
		let (flat, hybrid) = replay_at(HUGE);

		assert_eq!(flat, hybrid, "the tiered LRU evicts in a different order");
		assert!(!flat.is_empty());
	}

	#[test]
	fn a_tight_fast_tier_does_not_reorder_the_queue() {
		let (flat, hybrid) = replay_at(TIGHT);

		assert_eq!(
			flat, hybrid,
			"a tight fast-tier budget reordered the LRU queue -- demotion is \
			 supposed to move a cursor, not the list",
		);
	}
}

#[cfg(test)]
mod flat_fifo_fidelity {
	use super::*;
	use super::flat_fidelity::*;
	use super::super::fifo_compact_stack::FifoCompactStack;

	fn replay_at(capacity: CacheSize) -> (Vec<HashedKey>, Vec<HashedKey>) {
		replay(&mut FifoCompactStack::default(), &mut FifoCompactHybridStack::new(capacity), &skewed_ops())
	}

	#[test]
	fn evicts_in_the_same_order_as_the_flat_fifo_stack() {
		let (flat, hybrid) = replay_at(HUGE);

		assert_eq!(flat, hybrid, "the tiered FIFO evicts in a different order");
		assert!(!flat.is_empty());
	}

	#[test]
	fn a_tight_fast_tier_does_not_reorder_the_queue() {
		let (flat, hybrid) = replay_at(TIGHT);

		assert_eq!(
			flat, hybrid,
			"a tight fast-tier budget reordered the FIFO queue -- demotion is \
			 supposed to move a cursor, not the list",
		);
	}
}

/// The CLOCK order's own claims: the second chance is CLOCK's, not SIEVE's,
/// and an overwrite sets the bit.
#[cfg(test)]
mod flat_clock_fidelity {
	use super::*;
	use super::flat_fidelity::*;
	use super::super::clock_compact_stack::ClockCompactStack;

	fn replay_at(capacity: CacheSize) -> (Vec<HashedKey>, Vec<HashedKey>) {
		replay(&mut ClockCompactStack::default(), &mut ClockCompactHybridStack::new(capacity), &skewed_ops())
	}

	#[test]
	fn evicts_in_the_same_order_as_the_flat_clock_stack() {
		let (flat, hybrid) = replay_at(HUGE);

		assert_eq!(flat, hybrid, "the tiered CLOCK evicts in a different order");
		assert!(!flat.is_empty());
	}

	/// The same claim with the fast tier actually biting, which is the case
	/// that would catch a demotion or a promotion perturbing the queue.
	#[test]
	fn a_tight_fast_tier_does_not_reorder_the_queue() {
		let (flat, hybrid) = replay_at(TIGHT);

		assert_eq!(
			flat, hybrid,
			"a tight fast-tier budget reordered the CLOCK queue -- demotion is \
			 supposed to move a cursor, not the list",
		);
	}

	/// The SIEVE tripwire, stated as the structural fact rather than as an
	/// eviction order.
	///
	/// CLOCK and SIEVE agree on the eviction order of many short sequences --
	/// a queue that only drains gives the same answer either way -- so an
	/// order assertion is a weak test of this particular choice. The choice
	/// itself is not ambiguous at all: after a second chance the key must be at
	/// the FRONT of the queue, because `ClockCompactStack::evict_one` does
	/// `push_front`. SIEVE would leave it where it is.
	#[test]
	fn a_second_chance_moves_the_key_to_the_front() {
		let mut stack = ClockCompactHybridStack::new(HUGE);

		for key in [1u64, 2, 3] {
			stack.insert(key, 512);
		}

		assert_eq!(stack.list.front(Q), Some(3), "the newest key is not at the front");

		stack.update(1);

		assert_eq!(stack.evict_one(), Some(2), "the hand did not pass the referenced key");

		assert_eq!(
			stack.list.front(Q),
			Some(1),
			"the second-chance key was not recycled to the FRONT -- leaving it \
			 in place is SIEVE, and the flat ClockCompactStack this must match \
			 does `push_front`",
		);

		assert_eq!(stack.evict_one(), Some(3));
		assert_eq!(stack.evict_one(), Some(1));
		assert_eq!(stack.evict_one(), None);
	}

	/// An overwrite is a hit: `ClockCompactStack::insert` forwards an existing
	/// key to `update`.
	#[test]
	fn an_overwrite_sets_the_reference_bit() {
		let mut stack = ClockCompactHybridStack::new(HUGE);

		for key in [1u64, 2, 3] {
			stack.insert(key, 512);
		}

		// Same key, a DIFFERENT size, so this is a real resize as well.
		stack.insert(1, 1_024);

		assert_eq!(stack.evict_one(), Some(2), "the overwrite did not spare key 1");
		assert_eq!(stack.evict_one(), Some(3));
		assert_eq!(stack.evict_one(), Some(1));
	}
}
