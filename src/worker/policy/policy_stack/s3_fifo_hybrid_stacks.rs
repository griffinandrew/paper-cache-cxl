/*
 * Copyright (c) Kia Shakiba
 *
 * This source code is licensed under the GNU AGPLv3 license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! S3-FIFO, tier-segmented: a [`TierPolicy`] over a slow one-access queue and
//! a main queue whose fast prefix is the tier (R4; it was a whole stack of
//! its own, `S3FifoCompactHybridStack`, in a file of its own).
//!
//! Admission lands at the front of the one-access queue and is entirely
//! slow-tier: a new key is built slow, and nothing is pushed or settled. A
//! hit there promotes the key to the front of main and to fast, eagerly; a
//! hit in main only sets the reference bit -- this is the hottest per-get
//! operation in the family, and it touches no queue order at all -- and it is
//! eviction that acts on it: an accessed key at the main tail is moved to the
//! front with the bit cleared instead of being evicted (a SECOND CHANCE).
//! The main lane is split: the layer's cursor names its least-recently-used
//! fast key.
//!
//! Eviction prefers the one-access tail while main is not full (`fast + slow`
//! bytes of main under `(1 - ratio) x max_size`), and otherwise walks main's
//! tail. So a full main evicts from main even when the one-access queue is
//! what is over its budget.
//!
//! `needs_capacity_eviction` is the one-access queue's own budget
//! (`ratio x max_size`), raw.
//!
//! The second chance pushes `(key, Fast)` whenever the key ends fast, even if
//! it already was -- [`Push::IfEndsFast`], this family's rule, where a
//! promotion elsewhere pushes only for a key that was slow. A structural key
//! (S5) gets its second chance at the front with tier slow, never promoted,
//! and a fast one leaves the fast set, pushed.
//!
//! **The baseline named here before no longer exists in this crate.** Every
//! non-compact hybrid stack was removed once its compact twin was shown
//! behaviourally identical and cheaper: 72 B/object of eviction stack instead
//! of 112 then, and 40 since the arena conversion
//! (`ARENA_STACK_DRAM_OVERHEAD`). Git history holds the baseline and the
//! differential tests that proved the two agreed, and `tier_goldens.txt` holds
//! what this design did, fingerprinted, before it was ported.

use crate::PaperPolicy;

use super::{
	tiered_stack::{Lane, Meta, NoGhost, Push, SlowSplit, TierPolicy, TieredStack},
	CacheSize, HashedKey, Tier,
};

/// The slow one-access queue and the main queue.
const ONE: Lane = 0;
const MAIN: Lane = 1;

/// S3-FIFO: `ratio` is the one-access queue's share of the cache, with the
/// byte budgets it and main are held to.
pub struct S3 {
	ratio: f64,
	one_capacity: CacheSize,
	main_capacity: CacheSize,
}

/// `PaperPolicy::S3FifoCompactHybrid`.
pub type S3FifoCompactHybridStack = TieredStack<S3>;

impl S3 {
	fn sized(ratio: f64, max_size: CacheSize) -> Self {
		S3 {
			ratio,
			one_capacity: (ratio * max_size as f64) as CacheSize,
			main_capacity: ((1.0 - ratio) * max_size as f64) as CacheSize,
		}
	}
}

impl S3FifoCompactHybridStack {
	pub fn new(ratio: f64, max_size: CacheSize, fast_capacity: CacheSize) -> Self {
		TieredStack::with(S3::sized(ratio, max_size), fast_capacity)
	}
}

impl TierPolicy for S3 {
	type Layout = SlowSplit;
	type Ghost = NoGhost;

	const ADMIT: Lane = ONE;
	const RESETTLE: &'static [Lane] = &[MAIN];
	const ON_TIER_RESIZE: &'static [Lane] = &[MAIN];
	const SECOND_CHANCE: Push = Push::IfEndsFast;

	fn is_policy(&self, policy: &PaperPolicy) -> bool {
		matches!(policy, PaperPolicy::S3FifoCompactHybrid(ratio) if *ratio == self.ratio)
	}

	/// A one-access key is promoted to main's front, fast; a main key's bit is
	/// set, and nothing moves.
	fn touch(s: &mut TieredStack<Self>, key: HashedKey, structural: bool) {
		match s.lane_of(key) {
			Some(ONE) => s.to_front(key, MAIN, structural, Push::IfPromoted),
			Some(_) => s.set_bit(key, true),
			None => {},
		}
	}

	/// An access never reorders main, so a FAST main key overwritten with a
	/// structural value leaves the fast set in place (S5), then is touched.
	fn overwrite(s: &mut TieredStack<Self>, key: HashedKey, m: Meta) {
		s.resize_key(key, m);

		if m.structural && s.lane_of(key) == Some(MAIN) && s.tier_of(key) == Some(Tier::Fast) {
			s.demote_in_place(key);
		}

		Self::touch(s, key, m.structural);
	}

	/// The one-access tail while main is not full, else main's tail: a key
	/// with its bit set goes to the front (a second chance) and is looked past.
	fn victim(s: &mut TieredStack<Self>) -> Option<HashedKey> {
		if s.lane_bytes(MAIN) < s.policy.main_capacity {
			if let Some(key) = s.evict_tail(ONE) {
				return Some(key);
			}
		}

		loop {
			let key = s.tail(MAIN)?;

			if !s.bit(key) {
				return s.evict(key);
			}

			s.second_chance(key, MAIN);
		}
	}

	fn wants_eviction(s: &TieredStack<Self>) -> bool {
		s.lane_bytes(ONE) > s.policy.one_capacity
	}

	fn resized(s: &mut TieredStack<Self>, max_size: CacheSize) {
		s.policy = S3::sized(s.policy.ratio, max_size);
	}
}

/// S3-FIFO's two byte budgets are thresholds, and at exactly the threshold they
/// fall on different sides: the one-access queue is over its capacity only
/// ABOVE it, main is full AT it. (No golden reaches a byte total that equals a
/// capacity.)
#[cfg(test)]
mod capacity_tests {
	use super::*;
	use super::super::PolicyStack;

	#[test]
	fn the_one_access_queue_at_exactly_its_capacity_asks_for_nothing() {
		let mut stack = S3FifoCompactHybridStack::new(0.5, 2_000, 1 << 30);

		stack.insert(1, 1_000);

		assert!(!stack.needs_capacity_eviction(), "1,000 B in a queue of 1,000 B is not over it");

		stack.insert(2, 1);

		assert!(stack.needs_capacity_eviction(), "1,001 B is");
	}

	#[test]
	fn main_at_exactly_its_capacity_is_full() {
		let mut stack = S3FifoCompactHybridStack::new(0.5, 2_000, 1 << 30);

		stack.insert(1, 1_000);
		stack.update(1);
		stack.insert(2, 10);

		assert_eq!(
			stack.evict_one(),
			Some(1),
			"a main holding exactly its 1,000 B is full: the victim comes from main, not from the one-access queue",
		);
		assert_eq!(stack.evict_one(), Some(2), "main is empty now, so the one-access queue is next");
	}
}

/// A resize of the cache queues nothing in S3-FIFO, even on a stack the
/// metadata push has left over its budget: only `resettle` and a resize of the
/// fast tier settle it.
#[cfg(test)]
mod resize_tests {
	use super::*;
	use super::super::tiered_stack::testing::a_resize_settles_nothing;

	#[test]
	fn s3_fifo() {
		a_resize_settles_nothing(S3FifoCompactHybridStack::new(0.1, 1_000_000, 10_000));
	}
}
