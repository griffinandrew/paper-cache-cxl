//! `SieveCompactStack` — `SieveStack`'s policy over the slab design.
//!
//! The `HashList`-based `SieveStack` this re-lays-out was removed in R2 with the other
//! original flat stacks, so the mentions of it below are historical. Its eviction
//! orders for the op sequences in `fidelity_tests` are recorded in `golden.rs` and
//! asserted there.
//!
//! SIEVE differs from CLOCK in that the hand does NOT move entries. It scans
//! from its current position toward the front, clearing visited bits in place,
//! and evicts the first unvisited entry it meets; survivors keep their
//! position. The hand is therefore a key, not an index, and it is re-seated to
//! the entry BEFORE whatever it lands on — including on `remove`, matching
//! `SieveStack::remove`, which does the same so a removal cannot strand it.
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

/// The single queue. `CompactQueueSet` supports up to `MAX_QUEUES`; SIEVE
/// needs exactly one.
const Q: usize = 0;

pub struct SieveCompactStack {
	/// Payload is the visited bit.
	list: CompactQueueSet<bool>,

	/// The scan position, as a key. `None` restarts from the back.
	hand: Option<HashedKey>,
}

impl Default for SieveCompactStack {
	fn default() -> Self {
		SieveCompactStack {
			list: CompactQueueSet::default(),
			hand: None,
		}
	}
}

impl PolicyStack for SieveCompactStack {
	fn is_policy(&self, policy: &PaperPolicy) -> bool {
		matches!(policy, PaperPolicy::SieveCompact)
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

	/// Re-seats the hand before dropping the key, so removing the entry the
	/// hand points at cannot strand it. Mirrors `SieveStack::remove`.
	fn remove(&mut self, key: HashedKey) {
		self.hand = self.list.before(key);
		self.list.remove(Q, key);
	}

	fn clear(&mut self) {
		self.list.clear();
		self.hand = None;
	}

	fn evict_one(&mut self) -> Option<HashedKey> {
		loop {
			let key = match self.hand {
				Some(key) => key,
				None => self.list.back(Q)?,
			};

			self.hand = self.list.before(key);

			let visited = self.list.payload(key)?;

			if !visited {
				self.list.remove(Q, key);
				return Some(key);
			}

			if let Some(v) = self.list.payload_mut(key) {
				*v = false;
			}
		}
	}
}

/// Fidelity to `SieveStack`, whose policy this re-lays-out and which was removed in
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
			golden::single_queue_skewed(&mut SieveCompactStack::default()),
			golden::SIEVE_SKEWED,
		);
	}

	/// Removal must not disturb the order of what remains.
	#[test]
	fn removal_leaves_the_recorded_order() {
		assert_eq!(
			golden::removal(&mut SieveCompactStack::default(), 2_000, 512, true, 5),
			golden::SIEVE_REMOVAL,
		);
	}
}
