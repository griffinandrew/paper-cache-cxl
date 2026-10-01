/*
 * Copyright (c) Kia Shakiba
 *
 * This source code is licensed under the GNU AGPLv3 license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! Simplified 2Q, tier-segmented: a [`TierPolicy`] over a slow admission FIFO
//! and a main queue whose fast prefix is the tier (R4; it was a whole stack
//! of its own, `TwoQCompactHybridStack`, in a file of its own).
//!
//! Admission lands at the front of the FIFO and is entirely slow-tier: a
//! new key is built slow, and nothing is pushed or settled. A hit there
//! promotes the key to the front of main and to fast; a hit in main is an LRU
//! touch (and promotes a key that was demoted). The main lane is split: the
//! layer's cursor names its least-recently-used fast key, and demotion steps
//! it one place toward the MRU end per victim. Terminal eviction prefers the
//! FIFO tail, falling back to the main tail. Nothing is searched for.
//!
//! `needs_capacity_eviction` is the FIFO's own budget: its bytes over
//! `k_in x max_size`, raw (not scaled by the drain target, and not clamped to
//! the fast tier -- the FIFO is in the slow tier, so its capacity bounds PMEM).
//! `resize` only recomputes it; the settle sets are main alone, at `resettle`
//! and at a fast-tier resize.
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
	tiered_stack::{Lane, Push, SlowSplit, TierPolicy, TieredStack},
	CacheSize, HashedKey,
};

/// The slow admission FIFO and the main queue.
const FIFO: Lane = 0;
const MAIN: Lane = 1;

/// 2Q: `k_in` is the FIFO's share of the cache, `fifo_capacity` its bytes.
pub struct TwoQ {
	k_in: f64,
	fifo_capacity: CacheSize,
}

/// `PaperPolicy::TwoQCompactHybrid`.
pub type TwoQCompactHybridStack = TieredStack<TwoQ>;

impl TwoQCompactHybridStack {
	pub fn new(k_in: f64, max_size: CacheSize, fast_capacity: CacheSize) -> Self {
		TieredStack::with(TwoQ { k_in, fifo_capacity: (k_in * max_size as f64) as CacheSize }, fast_capacity)
	}
}

impl TierPolicy for TwoQ {
	type Layout = SlowSplit;

	const ADMIT: Lane = FIFO;
	const RESETTLE: &'static [Lane] = &[MAIN];
	const ON_TIER_RESIZE: &'static [Lane] = &[MAIN];

	fn is_policy(&self, policy: &PaperPolicy) -> bool {
		matches!(policy, PaperPolicy::TwoQCompactHybrid(k_in) if *k_in == self.k_in)
	}

	/// A FIFO key goes to the front of main, fast; a main key is moved to the
	/// front, and promoted if it was demoted.
	fn touch(s: &mut TieredStack<Self>, key: HashedKey, structural: bool) {
		s.to_front(key, MAIN, structural, Push::IfPromoted);
	}

	/// The FIFO tail first, else the main tail.
	fn victim(s: &mut TieredStack<Self>) -> Option<HashedKey> {
		s.evict_tail(FIFO).or_else(|| s.evict_tail(MAIN))
	}

	fn wants_eviction(s: &TieredStack<Self>) -> bool {
		s.lane_bytes(FIFO) > s.policy.fifo_capacity
	}

	fn resized(s: &mut TieredStack<Self>, max_size: CacheSize) {
		s.policy.fifo_capacity = (s.policy.k_in * max_size as f64) as CacheSize;
	}
}

/// The fast tier is charged the metadata of EVERY tracked key, the slow FIFO
/// and slow main keys as much as the fast ones. A reservation of
/// `fast_object_count() x shared_overhead` understates DRAM by every slow
/// key's metadata and fails this test.
#[cfg(test)]
mod reservation_tests {
	use super::*;
	use super::super::{drain_target, PolicyStack};
	use crate::object::ObjectSize;

	const FAST_CAPACITY: CacheSize = 10_000;
	const OVERHEAD: CacheSize = 200;
	const SIZE: ObjectSize = 1_000;

	/// 20 admissions and 12 hits: twenty keys x 200 B leave 6_000 B for
	/// values, so five main keys stay fast, seven go slow and eight are still
	/// in the FIFO. Charging the fast ones alone would keep eight fast.
	#[test]
	fn fifo_and_slow_main_keys_are_charged_against_the_fast_tier() {
		let mut stack = TwoQCompactHybridStack::new(0.5, 1_000_000, FAST_CAPACITY)
			.with_shared_overhead(OVERHEAD);

		for key in 1..=20 {
			stack.insert(key, SIZE);
		}

		for key in 1..=12 {
			stack.update(key);
		}

		let tracked = stack.len() as CacheSize;
		let fifo = stack.lane_len(FIFO);
		let main_slow = stack.slow_object_count() - fifo;

		assert_eq!(tracked, 20);
		assert!(
			fifo > 0 && main_slow > 0 && stack.fast_object_count() > 0,
			"the fixture must populate every state: {fifo} in the FIFO, {} fast, \
			 {main_slow} slow in main",
			stack.fast_object_count(),
		);
		assert_eq!(
			stack.dram_reserved_bytes(),
			tracked * OVERHEAD,
			"all {tracked} tracked keys keep their metadata in DRAM, but the reservation \
			 covers {} of them ({} fast, {main_slow} slow in main, {fifo} in the FIFO)",
			stack.dram_reserved_bytes() / OVERHEAD,
			stack.fast_object_count(),
		);

		let effective = FAST_CAPACITY - tracked * OVERHEAD;

		assert!(
			stack.fast_bytes_used() <= drain_target::bytes(effective),
			"{} B of values are fast against {effective} B left once all {tracked} keys' \
			 metadata is reserved",
			stack.fast_bytes_used(),
		);
	}
}

/// The FIFO's budget is a ceiling: at exactly its capacity it asks for nothing,
/// over it, for an eviction. (No golden reaches a byte total that equals a
/// capacity.)
#[cfg(test)]
mod capacity_tests {
	use super::*;
	use super::super::PolicyStack;

	#[test]
	fn the_fifo_at_exactly_its_capacity_asks_for_nothing() {
		let mut stack = TwoQCompactHybridStack::new(0.5, 2_000, 1 << 30);

		stack.insert(1, 1_000);

		assert!(!stack.needs_capacity_eviction(), "1,000 B in a FIFO of 1,000 B is not over it");

		stack.insert(2, 1);

		assert!(stack.needs_capacity_eviction(), "1,001 B is");

		stack.resize(4_000);

		assert!(!stack.needs_capacity_eviction(), "a resize moves the budget");
	}
}
