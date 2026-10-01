/*
 * Copyright (c) Kia Shakiba
 *
 * This source code is licensed under the GNU AGPLv3 license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! The index probes of the tiering layer's hot operations (R4).
//!
//! A probe is one keyed operation on the queues' index -- a `get`, an `insert`
//! or a `remove`: a hash and, in a cache of millions of keys, a cache miss, the
//! cost of an operation that grows with the cache. The flat stacks the layer
//! replaced paid for theirs by hand: a hit on a key already in place wrote
//! nothing back, a tail eviction removed the key once and mended the cursor
//! from the new tail, a settle read its candidate once. A layer over every
//! design is where a probe too many hides -- a write of what is already there,
//! a read a helper repeats because it was not handed the first -- and no order
//! and no fingerprint sees it. These hold each design's hot operations to the
//! figure of the stack it replaced, measured on that stack before it was
//! ported, or to the lower one the layer has reached since (the flat stack's
//! is beside the rows where the two differ).
//!
//! The counter is the index's own, in test builds (`arena_index::probes`), and
//! per thread, so the tests do not disturb one another.

use std::ops::RangeInclusive;

use super::{
	arena_index::probes,
	ClockCompactHybridStack, FifoCompactHybridStack, HashedKey, LruCompactHybridStack, PolicyStack,
	S3FifoCompactHybridStack, S3FifoGhostLazyDemotionCompactHybridStack, TwoQCompactHybridStack,
	TwoQFastAdmissionReprieveCompactHybridStack,
};

/// Runs `op` and holds the probes it makes to `most`.
fn within<R>(what: &str, most: u64, op: impl FnOnce() -> R) {
	let before = probes();

	op();

	let made = probes() - before;

	assert!(made <= most, "{what}: {made} index probes, and {most} are the most it may make");
}

/// Sets `keys`, 1,000 bytes each.
fn fill(stack: &mut dyn PolicyStack, keys: RangeInclusive<HashedKey>) {
	for key in keys {
		stack.insert(key, 1_000);
	}
}

/// Sets 1..=8 and hits 1..=4, which promotes them to main: a tier of 4,500
/// bytes holds four of the 1,000-byte keys, so the next promotion demotes one.
fn four_in_main(stack: &mut dyn PolicyStack) {
	fill(stack, 1..=8);

	for key in 1..=4 {
		stack.update(key);
	}
}

/// LRU: the whole cache fast, and then a tier that holds four of eight keys.
#[test]
fn lru() {
	let mut stack = LruCompactHybridStack::new(100_000);

	fill(&mut stack, 1..=8);

	within("a hit on a fast key", 3, || stack.update(4));
	within("a hit on the newest key", 3, || stack.update(4));
	within("a new key", 2, || stack.insert(9, 1_000));
	within("an overwrite of the same size", 4, || stack.insert(5, 1_000));
	within("a removal", 2, || stack.remove(6));
	within("an eviction of the fast tail", 2, || stack.evict_one());

	let mut stack = LruCompactHybridStack::new(4_500);

	fill(&mut stack, 1..=8);

	within("an eviction of a slow tail", 1, || stack.evict_one());
	within("a new key that demotes one", 6, || stack.insert(9, 1_000));
	// The flat stack: 10.
	within("a hit on a slow key: a promotion and a demotion", 9, || stack.update(3));
	within("a hit on a fast key in a tight tier", 3, || stack.update(8));
}

#[test]
fn fifo() {
	let mut stack = FifoCompactHybridStack::new(100_000);

	fill(&mut stack, 1..=8);

	within("a hit", 0, || stack.update(4));
	within("an eviction of the fast tail", 2, || stack.evict_one());
}

#[test]
fn clock() {
	let mut stack = ClockCompactHybridStack::new(100_000);

	fill(&mut stack, 1..=8);

	within("a hit", 1, || stack.update(4));

	stack.update(1);

	// The flat stack: 10.
	within("an eviction past one second chance", 9, || stack.evict_one());
	within("an eviction of a key with no bit", 3, || stack.evict_one());
}

/// 2Q: a slow FIFO in front of a main queue whose head is the tier.
#[test]
fn two_q() {
	let mut stack = TwoQCompactHybridStack::new(0.5, 1_000_000, 100_000);

	fill(&mut stack, 1..=8);

	// The flat stack: 6.
	within("a hit on a FIFO key: a promotion to main", 5, || stack.update(3));
	// The flat stack: 4.
	within("a hit on a main key", 3, || stack.update(3));
	within("an eviction of the FIFO's tail", 1, || stack.evict_one());
	within("a new key", 2, || stack.insert(9, 1_000));

	let mut stack = TwoQCompactHybridStack::new(0.5, 1_000_000, 4_500);

	four_in_main(&mut stack);

	// The flat stack: 10.
	within("a hit on a FIFO key that demotes one", 9, || stack.update(5));
}

/// S3-FIFO: a slow one-access queue in front of a main queue whose head is the
/// tier; a hit in main sets a bit, and eviction acts on it.
#[test]
fn s3_fifo() {
	let mut stack = S3FifoCompactHybridStack::new(0.1, 1_000_000, 100_000);

	fill(&mut stack, 1..=8);

	within("a hit on a one-access key: a promotion to main", 6, || stack.update(3));
	within("a hit on a main key: the reference bit", 3, || stack.update(3));
	within("an eviction of the one-access tail", 1, || stack.evict_one());
	within("a new key", 2, || stack.insert(9, 1_000));

	let mut stack = S3FifoCompactHybridStack::new(0.1, 1_000_000, 4_500);

	four_in_main(&mut stack);

	within("a hit on a one-access key that demotes one", 10, || stack.update(5));

	// A full main: the eviction walks it, and a key with its bit set goes to
	// the front first.
	let mut stack = S3FifoCompactHybridStack::new(0.9, 20_000, 100_000);

	fill(&mut stack, 1..=4);

	for key in 1..=3 {
		stack.update(key);
	}

	stack.update(1);

	within("an eviction past one second chance", 10, || stack.evict_one());
}

/// S3-FIFO with lazy demotion: the settle gives a candidate whose bit is set a
/// fresh start instead of demoting it.
#[test]
fn s3_fifo_with_lazy_demotion() {
	let mut stack = S3FifoGhostLazyDemotionCompactHybridStack::new(0.1, 1_000_000, 4_500);

	four_in_main(&mut stack);
	stack.update(1);

	// The flat stack: 16.
	within("a promotion that reprieves one key and demotes another", 15, || stack.update(5));

	let mut stack = S3FifoGhostLazyDemotionCompactHybridStack::new(0.1, 1_000_000, 4_500);

	four_in_main(&mut stack);

	// The flat stack: 11.
	within("a promotion that demotes one key", 10, || stack.update(5));
}

/// 2Q with a DRAM admission FIFO and a reprieve: the FIFO's overflow is spliced
/// onto the back of main, and a hit moves a key into main. First a tier the
/// FIFO's carve-out covers (so main has no budget and every promotion falls
/// straight back out), then a FIFO of 4,500 B in a tier of 10,000 B.
#[test]
fn two_q_fast_admission_reprieve() {
	let mut stack = TwoQFastAdmissionReprieveCompactHybridStack::new(0.5, 1_000_000, 100_000);

	fill(&mut stack, 1..=8);

	// The flat stack: 8.
	within("a hit on a FIFO key: a promotion to main", 7, || stack.update(3));
	// The flat stack: 10.
	within("a hit on a main key", 8, || stack.update(3));
	within("a new key", 2, || stack.insert(9, 1_000));
	// The flat stack: 9.
	within("an overwrite of a FIFO key", 8, || stack.insert(8, 1_000));
	within("a removal", 2, || stack.remove(7));
	within("an eviction of main's tail", 1, || stack.evict_one());

	let mut stack = TwoQFastAdmissionReprieveCompactHybridStack::new(0.0045, 1_000_000, 10_000);

	fill(&mut stack, 1..=4);

	within("a new key that spills one", 5, || stack.insert(5, 1_000));

	fill(&mut stack, 6..=12);

	for key in 1..=5 {
		stack.update(key);
	}

	// The flat stack: 11.
	within("a hit on a slow main key: a promotion and a demotion", 9, || stack.update(6));
	// The flat stack: 9.
	within("a hit on a FIFO key that demotes one", 8, || stack.update(12));
	within("a structural new key", 2, || stack.insert(40, 50_000));
	within("an eviction of a slow main tail", 1, || stack.evict_one());

	let mut stack = TwoQFastAdmissionReprieveCompactHybridStack::new(0.0045, 1_000_000, 10_000);

	fill(&mut stack, 1..=4);

	for _ in 0..4 {
		stack.evict_one();
	}

	fill(&mut stack, 5..=6);

	within("an eviction of the FIFO's tail, main empty", 1, || stack.evict_one());
}
