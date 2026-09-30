//! `MruCompactStack` — `MruStack`'s policy over the slab design.
//!
//! The `HashList`-based `MruStack` this re-lays-out was removed in R2 with the other
//! original flat stacks, so the mentions of it below are historical. Its eviction
//! orders for the op sequences in `fidelity_tests` are recorded in `golden.rs` and
//! asserted there.
//!
//! MRU evicts the most recently used object, so the MRU key is held OUTSIDE
//! the queue in its own slot and the queue holds everything else in recency
//! order. Eviction takes the queue front, falling back to the held key when
//! the queue is empty — the same structure `MruStack` uses, and the reason
//! `len` is `queue + 1` whenever a key is held.
//!
//! Note `insert` tests membership of the QUEUE only, not the held key, which
//! is faithful to `MruStack::insert`. Re-inserting the currently-held key
//! therefore takes the same path in both implementations.
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

/// The single queue. `CompactQueueSet` supports up to `MAX_QUEUES`; MRU
/// needs exactly one.
const Q: usize = 0;

pub struct MruCompactStack {
	/// The most recently used key, held outside the queue.
	maybe_mru_key: Option<HashedKey>,
	list: CompactQueueSet<()>,
}

impl Default for MruCompactStack {
	fn default() -> Self {
		MruCompactStack {
			maybe_mru_key: None,
			list: CompactQueueSet::default(),
		}
	}
}

impl MruCompactStack {
	/// Push `key` to the queue front, moving it if it is already queued.
	///
	/// MRU can transiently hold a key in BOTH the queue and the MRU slot --
	/// re-inserting the held key pushes it into the queue while it stays held.
	/// A later push then targets an already-queued key. `kwik::HashList` is
	/// key-indexed and de-duplicates that; `CompactQueueSet::push_front`
	/// appends unconditionally, which would leave two nodes for one key and
	/// inflate `len`. Matching the original's behaviour here keeps the two
	/// implementations bit-identical.
	fn requeue_front(&mut self, key: HashedKey) {
		if self.list.contains(key) {
			self.list.move_front(Q, key);
		} else {
			self.list.push_front(Q, key, ());
		}
	}
}

impl PolicyStack for MruCompactStack {
	fn is_policy(&self, policy: &PaperPolicy) -> bool {
		matches!(policy, PaperPolicy::MruCompact)
	}

	fn len(&self) -> usize {
		if self.maybe_mru_key.is_none() {
			return 0;
		}

		self.list.len() + 1
	}

	fn contains(&self, key: HashedKey) -> bool {
		if self.maybe_mru_key.is_some_and(|mru_key| mru_key == key) {
			return true;
		}

		self.list.contains(key)
	}

	fn insert(&mut self, key: HashedKey, _: ObjectSize) {
		if self.list.contains(key) {
			return self.update(key);
		}

		if let Some(mru_key) = self.maybe_mru_key {
			self.requeue_front(mru_key);
		}

		self.maybe_mru_key = Some(key);
	}

	fn update(&mut self, key: HashedKey) {
		if self.maybe_mru_key.is_some_and(|mru_key| mru_key == key) {
			return;
		}

		self.list.remove(Q, key);

		if let Some(old_mru_key) = self.maybe_mru_key.take() {
			self.requeue_front(old_mru_key);
		}

		self.maybe_mru_key = Some(key);
	}

	fn remove(&mut self, key: HashedKey) {
		if self.maybe_mru_key.is_some_and(|mru_key| mru_key == key) {
			self.maybe_mru_key = self.list.pop_front(Q).map(|(k, ())| k);
			return;
		}

		self.list.remove(Q, key);
	}

	fn clear(&mut self) {
		self.maybe_mru_key = None;
		self.list.clear();
	}

	fn evict_one(&mut self) -> Option<HashedKey> {
		self.list
			.pop_front(Q)
			.map(|(key, ())| key)
			.or_else(|| self.maybe_mru_key.take())
	}
}

/// Fidelity to `MruStack`, whose policy this re-lays-out and which was removed in
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
			golden::single_queue_skewed(&mut MruCompactStack::default()),
			golden::MRU_SKEWED,
		);
	}

	/// Removal must not disturb the order of what remains.
	#[test]
	fn removal_leaves_the_recorded_order() {
		assert_eq!(
			golden::removal(&mut MruCompactStack::default(), 2_000, 512, true, 5),
			golden::MRU_REMOVAL,
		);
	}
}
