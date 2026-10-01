/*
 * Copyright (c) Kia Shakiba
 *
 * This source code is licensed under the GNU AGPLv3 license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! The three designs that differ only in what a hit and an eviction do: LRU,
//! FIFO and CLOCK, each a [`TierPolicy`] over the one-lane [`TieredStack`].
//!
//! The stack keeps every tracked key in ONE queue spanning both tiers, with
//! the newest end fast and a cursor at the OLDEST fast key, so a demotion is
//! one step of the cursor and nothing is searched; the layer owns all of that.
//! What is left is three rules:
//!
//! ```text
//!             a hit                         an overwrite             the victim
//!   LRU       move to the front             a hit, after a resize    the tail
//!   FIFO      nothing (the trait default)   resize in place          the tail
//!   CLOCK     set the reference bit         FIFO's, and a hit        the hand: see below
//! ```
//!
//! FIFO is LRU without the reorder, and that is the whole of it: insertion
//! order IS eviction order, so `hit` does nothing -- overriding it with
//! LRU's would silently turn the design into LRU -- and an overwrite of a
//! tracked key only resizes, re-settling if the key is fast.
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
//! flat clock stack implements. The second chance is LRU's move to the front:
//! relink to the front, promote, settle ([`TieredStack::second_chance`]).
//! Promotion is not optional: the fast set has to stay a contiguous prefix of
//! the queue, so a key moved into it must become fast in the same operation.
//! CLOCK differs from LRU not in what that operation does but in WHEN it
//! happens -- LRU runs it on every hit, CLOCK only for the keys a hit saved
//! from the hand.
//!
//! The merged object store is the other statement of the same three policies
//! (`MergedOrder::{Lru, Fifo, Clock}`), a sharded slab with its own boundary
//! and settle; `merged_stack`'s order-fidelity tests replay both against each
//! other, key for key, and this stack is their reference.
//!
//! # Cost
//!
//! One [`NodePayload`](super::arena_queue_set::NodePayload) per key in the
//! arena: 40 B/object of eviction stack, the same for all three -- the
//! reference bit rides in `freq`, as `freq != 0` (the idiom the S3-FIFO family
//! uses), so CLOCK costs not a byte more than FIFO. `overhead.rs` holds the
//! measurement.

use crate::PaperPolicy;

use super::{
	clock_hand_budget,
	tiered_stack::{Lane, Meta, NoGhost, Push, Single, TierPolicy, TieredStack},
	CacheSize, HashedKey, PolicyStack,
};

/// The single lane.
const MAIN: Lane = 0;

/// LRU: the queue is a recency order.
pub struct Lru;

/// FIFO: the queue is an insertion order, and a hit does not reorder.
pub struct Fifo;

/// CLOCK: FIFO plus a reference bit, the second chance recycling to the front.
pub struct Clock;

/// `PaperPolicy::LruCompactHybrid`.
pub type LruCompactHybridStack = TieredStack<Lru>;

/// `PaperPolicy::FifoCompactHybrid`.
pub type FifoCompactHybridStack = TieredStack<Fifo>;

/// `PaperPolicy::ClockCompactHybrid`.
pub type ClockCompactHybridStack = TieredStack<Clock>;

macro_rules! constructors {
	($($stack:ident = $policy:ident),*) => {$(
		impl $stack {
			pub fn new(fast_capacity: CacheSize) -> Self {
				TieredStack::with($policy, fast_capacity)
			}
		}
	)*};
}

constructors!(LruCompactHybridStack = Lru, FifoCompactHybridStack = Fifo, ClockCompactHybridStack = Clock);

impl TierPolicy for Lru {
	type Layout = Single;
	type Ghost = NoGhost;

	fn is_policy(&self, policy: &PaperPolicy) -> bool {
		matches!(policy, PaperPolicy::LruCompactHybrid)
	}

	/// A tracked key is moved to the front (the default `hit` and `overwrite`
	/// both come here: an overwrite resizes first).
	fn touch(s: &mut TieredStack<Self>, key: HashedKey, structural: bool) {
		s.to_front(key, MAIN, structural, Push::IfPromoted);
	}
}

impl TierPolicy for Fifo {
	type Layout = Single;
	type Ghost = NoGhost;

	fn is_policy(&self, policy: &PaperPolicy) -> bool {
		matches!(policy, PaperPolicy::FifoCompactHybrid)
	}

	/// Nothing: a hit moves nothing, and is not even read.
	fn hit(_s: &mut TieredStack<Self>, _key: HashedKey) {}

	/// An existing key is resized in place and NOT moved -- insertion order is
	/// eviction order.
	fn overwrite(s: &mut TieredStack<Self>, key: HashedKey, m: Meta) {
		s.overwrite_in_place(key, m);
	}
}

impl TierPolicy for Clock {
	type Layout = Single;
	type Ghost = NoGhost;

	fn is_policy(&self, policy: &PaperPolicy) -> bool {
		matches!(policy, PaperPolicy::ClockCompactHybrid)
	}

	/// A cache hit: set the reference bit and nothing else.
	///
	/// The queue is not touched. That is the half of CLOCK that makes it cheap,
	/// and the half `MergedOrder::Clock` turns into a relaxed atomic store
	/// under a read lock.
	fn hit(s: &mut TieredStack<Self>, key: HashedKey) {
		s.set_bit(key, true);
	}

	/// An existing key is resized in place and NOT moved -- insertion order is
	/// the queue order -- but it IS referenced: `ClockCompactStack`'s `insert`
	/// forwards an existing key straight to `update`, so writing a key earns it
	/// a second chance exactly as reading it does.
	fn overwrite(s: &mut TieredStack<Self>, key: HashedKey, m: Meta) {
		s.set_bit(key, true);
		s.overwrite_in_place(key, m);
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
	fn victim(s: &mut TieredStack<Self>) -> Option<HashedKey> {
		let mut budget = clock_hand_budget(s.len());

		loop {
			let key = s.tail(MAIN)?;

			if budget > 0 && s.bit(key) {
				budget -= 1;

				// Cleared and recycled to the FRONT -- CLOCK, not SIEVE. See
				// the module doc. Reached only here, and only for a key whose
				// bit the hand has just cleared: that is the whole difference
				// between this policy and LRU, the same relink from the
				// eviction path rather than from every hit.
				s.second_chance(key, MAIN);

				continue;
			}

			return s.evict_tail(MAIN);
		}
	}
}
/// The re-promotion the LRU hit queues on a re-`set` of a slow key looks
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
	use super::super::Tier;
	use crate::object::ObjectSize;
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
		// A migrating test: it runs under the migrating tests' lock (see
		// `migration_test_lock`).
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

		let stats = crate::worker::MigStats::default();

		for (key, tier) in queue {
			apply_migration(&objects, key, tier, &stats);
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

	fn construction_does_not_allocate_from_the_budget<P: TierPolicy>(make: fn(CacheSize) -> TieredStack<P>) {
		for budget in [u64::MAX / 4, 4 * 1024 * 1024 * 1024, 1_024] {
			let stack = make(budget).with_shared_overhead(224);
			assert_eq!(stack.len(), 0, "budget {budget}: nothing is tracked yet");
			assert_eq!(
				stack.slab_capacity(),
				0,
				"budget {budget}: the slab must be empty at construction -- a stack \
				 that pre-sizes from capacity allocates for objects that may never \
				 arrive, and reserves a different amount than it was measured at",
			);
		}
	}

	/// The counterpart: growth still happens, it is just demand-driven.
	fn the_slab_grows_on_demand<P: TierPolicy>(make: fn(CacheSize) -> TieredStack<P>) {
		let mut stack = make(u64::MAX / 4).with_shared_overhead(224);
		for i in 0..1_000u64 {
			stack.insert(i, 64);
		}
		assert_eq!(stack.len(), 1_000);
		assert!(
			stack.slab_capacity() >= 1_000,
			"the slab must have grown to hold what was inserted",
		);
	}

	macro_rules! per_order {
		($($order:ident: $new:expr),*) => {$(
			mod $order {
				use super::*;

				#[test]
				fn construction_does_not_allocate_from_the_budget() {
					super::construction_does_not_allocate_from_the_budget($new);
				}

				#[test]
				fn the_slab_grows_on_demand() {
					super::the_slab_grows_on_demand($new);
				}
			}
		)*};
	}

	per_order!(lru: LruCompactHybridStack::new, fifo: FifoCompactHybridStack::new, clock: ClockCompactHybridStack::new);
}

/// The fast tier is charged the metadata of EVERY tracked object, not only the
/// fast ones: a demotion moves the value and leaves the row, the stack node and
/// the header in DRAM. A reservation of `fast_object_count() x shared_overhead`
/// understates DRAM by the whole slow tier's metadata, and both tests fail
/// under it.
#[cfg(test)]
mod reservation_tests {
	use super::*;
	use super::super::drain_target;
	use crate::object::ObjectSize;

	const FAST_CAPACITY: CacheSize = 10_000;
	const OVERHEAD: CacheSize = 200;
	const SIZE: ObjectSize = 1_000;
	const N: HashedKey = 20;

	/// 20 x 200 B of metadata leaves 6_000 B for values: five of the twenty
	/// objects stay fast. Charging the fast ones alone would keep eight.
	fn slow_objects_are_charged_against_the_fast_tier<P: TierPolicy>(make: fn(CacheSize) -> TieredStack<P>) {
		let mut stack = make(FAST_CAPACITY).with_shared_overhead(OVERHEAD);

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
	fn metadata_at_or_over_the_fast_tier_leaves_no_room_for_values<P: TierPolicy>(make: fn(CacheSize) -> TieredStack<P>) {
		const OVERHEAD: CacheSize = 1_000;
		const SIZE: ObjectSize = 100;

		for n in [10, 12] {
			let mut stack = make(FAST_CAPACITY).with_shared_overhead(OVERHEAD);

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
		($($order:ident: $new:expr),*) => {$(
			mod $order {
				use super::*;

				#[test]
				fn slow_objects_are_charged_against_the_fast_tier() {
					super::slow_objects_are_charged_against_the_fast_tier($new);
				}

				#[test]
				fn metadata_at_or_over_the_fast_tier_leaves_no_room_for_values() {
					super::metadata_at_or_over_the_fast_tier_leaves_no_room_for_values($new);
				}
			}
		)*};
	}

	per_order!(lru: LruCompactHybridStack::new, fifo: FifoCompactHybridStack::new, clock: ClockCompactHybridStack::new);
}

/// The stack's books against its queue, after EVERY operation of a long random
/// sequence (`tiered_stack::testing`): the object and byte counts of each tier
/// are what walking the queue finds, and the boundary is the oldest FAST key,
/// with nothing but slow keys behind it. The sequence mixes every way a key
/// enters, moves and leaves -- sets of new and tracked keys (some larger than
/// an empty fast tier, so structural), hits, removals, evictions, a shrinking
/// and growing fast tier, a pushed measured M -- against a tight tier, so
/// demotions, promotions and second chances all happen. What the
/// eviction-order tests cannot see: a count or a byte total that drifts while
/// the order stays right.
#[cfg(test)]
mod invariant_tests {
	use super::*;
	use super::super::tiered_stack::testing::books_match_the_queue_after_every_operation;

	/// FIFO never promotes: a hit moves nothing.
	fn books_match<P: TierPolicy>(make: fn(CacheSize) -> TieredStack<P>, promotes: bool) {
		let (demoted, promoted) = books_match_the_queue_after_every_operation(make(24_000).with_shared_overhead(40));

		assert!(demoted > 100, "the sequence never demoted ({demoted})");

		if promotes {
			assert!(promoted > 10, "the sequence never promoted ({promoted})");
		}
	}

	#[test]
	fn lru() {
		books_match(LruCompactHybridStack::new, true);
	}

	#[test]
	fn fifo() {
		books_match(FifoCompactHybridStack::new, false);
	}

	#[test]
	fn clock() {
		books_match(ClockCompactHybridStack::new, true);
	}
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

		assert_eq!(stack.front(MAIN), Some(3), "the newest key is not at the front");

		stack.update(1);

		assert_eq!(stack.evict_one(), Some(2), "the hand did not pass the referenced key");

		assert_eq!(
			stack.front(MAIN),
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

/// A resize of the cache queues nothing in LRU, FIFO or CLOCK, even on a stack
/// the metadata push has left over its budget: only `resettle` and a resize of
/// the fast tier settle it (`tiered_stack::testing::a_resize_settles_nothing`).
#[cfg(test)]
mod resize_tests {
	use super::*;
	use super::super::tiered_stack::testing::a_resize_settles_nothing;

	#[test]
	fn lru() {
		a_resize_settles_nothing(LruCompactHybridStack::new(10_000));
	}

	#[test]
	fn fifo() {
		a_resize_settles_nothing(FifoCompactHybridStack::new(10_000));
	}

	#[test]
	fn clock() {
		a_resize_settles_nothing(ClockCompactHybridStack::new(10_000));
	}
}
