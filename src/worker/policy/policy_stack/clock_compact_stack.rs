//! `ClockCompactStack` — `ClockStack`'s policy over the slab design.
//!
//! The `HashList`-based `ClockStack` this re-lays-out was removed in R2 with the other
//! original flat stacks, so the mentions of it below are historical. Its eviction
//! orders for the op sequences in `fidelity_tests` are recorded in `golden.rs` and
//! asserted there.
//!
//! CLOCK's second-chance rule: a hit sets the visited bit; eviction walks from
//! the back, and a visited entry has its bit cleared and is recycled to the
//! front instead of being evicted. The payload is therefore a single `bool`,
//! which `CompactQueueSet` stores in the INDEX value rather than the slab
//! slot — so the slot stays 16 bytes and the flag costs no alignment padding.
//!
//! Unlike the `HashList`-based original this honours `eviction_stacks_pmem`.

use crate::{
	HashedKey,
	ObjectSize,
	PaperPolicy,
};

use super::{
	PolicyStack,
	compact_queue_set::CompactQueueSet,
};

/// The single queue. `CompactQueueSet` supports up to `MAX_QUEUES`; CLOCK
/// needs exactly one.
const Q: usize = 0;

pub struct ClockCompactStack {
	/// Payload is the visited bit.
	list: CompactQueueSet<bool>,
}

impl Default for ClockCompactStack {
	fn default() -> Self {
		ClockCompactStack { list: CompactQueueSet::default() }
	}
}

impl PolicyStack for ClockCompactStack {
	fn is_policy(&self, policy: &PaperPolicy) -> bool {
		matches!(policy, PaperPolicy::ClockCompact)
	}

	fn len(&self) -> usize {
		self.list.len()
	}

	fn contains(&self, key: HashedKey) -> bool {
		self.list.contains(key)
	}

	fn insert(&mut self, key: HashedKey, _: ObjectSize) {
		if self.list.contains(key) {
			return self.update(key);
		}

		self.list.push_front(Q, key, false);
	}

	fn update(&mut self, key: HashedKey) {
		if let Some(visited) = self.list.payload_mut(key) {
			*visited = true;
		}
	}

	fn remove(&mut self, key: HashedKey) {
		self.list.remove(Q, key);
	}

	fn clear(&mut self) {
		self.list.clear();
	}

	/// Pops from the back; a visited entry is cleared and recycled to the
	/// front, exactly as `ClockStack::evict_one` does.
	fn evict_one(&mut self) -> Option<HashedKey> {
		loop {
			let (key, visited) = self.list.pop_back(Q)?;

			if !visited {
				return Some(key);
			}

			self.list.push_front(Q, key, false);
		}
	}
}

/// Fidelity to `ClockStack`, whose policy this re-lays-out and which was removed in
/// R2: the eviction order it gave for the two op sequences below is recorded in
/// `golden` and asserted here. Same access sequence, same eviction order: this
/// changes how the queue is STORED, not what the policy means.
#[cfg(test)]
mod fidelity_tests {
	use super::*;
	use super::super::golden;

	/// 40,000 skewed ops, a third of them hits.
	#[test]
	fn evicts_in_the_recorded_order() {
		assert_eq!(
			golden::single_queue_skewed(&mut ClockCompactStack::default()),
			golden::CLOCK_SKEWED,
		);
	}

	/// Removal must not disturb the order of what remains.
	#[test]
	fn removal_leaves_the_recorded_order() {
		assert_eq!(
			golden::removal(&mut ClockCompactStack::default(), 2_000, 512, true, 5),
			golden::CLOCK_REMOVAL,
		);
	}
}
