//! `FifoCompactStack` — `FifoStack`'s policy over the slab design.
//!
//! The `HashList`-based `FifoStack` this re-lays-out was removed in R2 with the other
//! original flat stacks, so the mentions of it below are historical. Its eviction
//! orders for the op sequences in `fidelity_tests` are recorded in `golden.rs` and
//! asserted there.
//!
//! Insertion order only: a hit does NOT reorder, which is the whole of what
//! separates FIFO from LRU. `PolicyStack::update` is left at its default
//! no-op for exactly that reason, matching `FifoStack`, which also does not
//! override it.
//!
//! Carries no payload — `CompactQueueSet<()>`, so the slab holds 16-byte
//! link-only slots and the index value is a bare slot number, against
//! `FifoStack`'s 48-byte `HashList` node plus its own key-to-node index.
//!
//! Unlike the `HashList`-based original this honours `eviction_stacks_pmem`,
//! because `CompactQueueSet` is allocator-parameterised.

use crate::{
	HashedKey,
	ObjectSize,
	PaperPolicy,
};

use super::{
	PolicyStack,
	compact_queue_set::CompactQueueSet,
};

/// The single queue. `CompactQueueSet` supports up to `MAX_QUEUES`; a FIFO
/// needs exactly one.
const Q: usize = 0;

pub struct FifoCompactStack {
	list: CompactQueueSet<()>,
}

impl Default for FifoCompactStack {
	fn default() -> Self {
		FifoCompactStack { list: CompactQueueSet::default() }
	}
}

impl PolicyStack for FifoCompactStack {
	fn is_policy(&self, policy: &PaperPolicy) -> bool {
		matches!(policy, PaperPolicy::FifoCompact)
	}

	fn len(&self) -> usize {
		self.list.len()
	}

	fn contains(&self, key: HashedKey) -> bool {
		self.list.contains(key)
	}

	/// Re-inserting a present key defers to `update`, which is the trait's
	/// no-op — so insertion order is preserved, as in `FifoStack`.
	fn insert(&mut self, key: HashedKey, _: ObjectSize) {
		if self.list.contains(key) {
			return self.update(key);
		}

		self.list.push_front(Q, key, ());
	}

	fn remove(&mut self, key: HashedKey) {
		self.list.remove(Q, key);
	}

	fn clear(&mut self) {
		self.list.clear();
	}

	fn evict_one(&mut self) -> Option<HashedKey> {
		self.list.pop_back(Q).map(|(key, ())| key)
	}
}

/// Fidelity to `FifoStack`, whose policy this re-lays-out and which was removed in
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
			golden::single_queue_skewed(&mut FifoCompactStack::default()),
			golden::FIFO_SKEWED,
		);
	}

	/// Removal must not disturb the order of what remains.
	#[test]
	fn removal_leaves_the_recorded_order() {
		assert_eq!(
			golden::removal(&mut FifoCompactStack::default(), 2_000, 512, true, 5),
			golden::FIFO_REMOVAL,
		);
	}
}
