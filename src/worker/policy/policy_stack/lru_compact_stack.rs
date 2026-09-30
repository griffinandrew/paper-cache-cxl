//! `LruCompactStack` — `LruStack`'s policy over the slab design.
//!
//! The `HashList`-based `LruStack` this re-lays-out was removed in R2 with the other
//! original flat stacks, so the mentions of it below are historical. Its eviction
//! orders for the op sequences in `fidelity_tests` are recorded in `golden.rs` and
//! asserted there.
//!
//! Exists to separate two effects the existing matrix confounds. Comparing
//! `Lru` (all-DRAM, `HashList`) against a tiered `HashList` stack measures
//! TIERING; comparing that against `LruCompactHybrid` measures LAYOUT. But
//! comparing all-DRAM against a tiered compact stack measures both at once,
//! which is how a 23% throughput gap on cluster13 LRU and a *reversed* 24%
//! gap on cluster13 LFU both appeared without either being attributable.
//!
//! This is the missing cell: the compact LAYOUT with no tiering at all.
//!
//! ```text
//!                    HashList layout        slab layout
//!   no tiering       LruStack               LruCompactStack   <- this
//!   tiered           (removed)              LruCompactHybridStack
//! ```
//!
//! The `HashList`-tiered cell is empty because `LruHybridStack` has been
//! removed from the crate: it was behaviourally identical to
//! `LruCompactHybridStack` and cost 112 B/object of eviction stack against
//! its 72 then (40 since the arena conversion). The layout comparison it anchored lives in git history.
//!
//! Deliberately carries NO payload. `LruStack` ignores the size argument
//! entirely (`fn insert(&mut self, key, _: ObjectSize)`) because a non-tiered
//! LRU needs only recency order — byte accounting belongs to the cache, not
//! the stack. So this uses `CompactQueueSet<()>`: the slab holds 16-byte
//! link-only slots and the index value is a bare `u32` slot number. No tier
//! tag, no size, no `dram_resident` — none of which a non-tiered design has
//! any use for.
//!
//! Per object that is a 16-byte slot plus one index entry, against
//! `LruStack`'s 48-byte `HashList` node, 8-byte key, and the `HashList`'s own
//! separate key-to-node index.

use crate::{
	HashedKey,
	ObjectSize,
	PaperPolicy,
};

use super::{
	PolicyStack,
	compact_queue_set::CompactQueueSet,
};

/// The single recency queue. `CompactQueueSet` supports up to `MAX_QUEUES`;
/// a non-tiered LRU needs exactly one.
const Q_LRU: usize = 0;

pub struct LruCompactStack {
	list: CompactQueueSet<()>,
}

impl Default for LruCompactStack {
	fn default() -> Self {
		LruCompactStack {
			list: CompactQueueSet::default(),
		}
	}
}

impl PolicyStack for LruCompactStack {
	fn is_policy(&self, policy: &PaperPolicy) -> bool {
		matches!(policy, PaperPolicy::LruCompact)
	}

	fn len(&self) -> usize {
		self.list.len()
	}

	fn contains(&self, key: HashedKey) -> bool {
		self.list.contains(key)
	}

	/// Size is ignored, exactly as `LruStack::insert` ignores it: recency
	/// order is the whole of the policy here.
	fn insert(&mut self, key: HashedKey, _: ObjectSize) {
		if self.list.contains(key) {
			return self.update(key);
		}

		self.list.push_front(Q_LRU, key, ());
	}

	fn update(&mut self, key: HashedKey) {
		self.list.move_front(Q_LRU, key);
	}

	fn remove(&mut self, key: HashedKey) {
		self.list.remove(Q_LRU, key);
	}

	fn clear(&mut self) {
		self.list.clear();
	}

	fn evict_one(&mut self) -> Option<HashedKey> {
		self.list.pop_back(Q_LRU).map(|(key, ())| key)
	}
}

/// Fidelity to `LruStack`, whose policy this re-lays-out and which was removed in
/// R2: the eviction order it gave for the two op sequences below is recorded in
/// `golden` and asserted here. Same access sequence, same eviction order: this
/// changes how the queue is STORED, not what the policy means.
#[cfg(test)]
mod fidelity_tests {
	use super::*;
	use super::super::golden;

	/// Skewed access with reuse, so keys are repeatedly moved to the front --
	/// the operation the two layouts implement differently.
	#[test]
	fn evicts_in_the_recorded_order() {
		assert_eq!(
			golden::single_queue_skewed(&mut LruCompactStack::default()),
			golden::LRU_SKEWED,
		);
	}

	/// Removal must not disturb the order of what remains.
	#[test]
	fn removal_leaves_the_recorded_order() {
		assert_eq!(
			golden::removal(&mut LruCompactStack::default(), 1_000, 512, false, 3),
			golden::LRU_REMOVAL,
		);
	}
}
