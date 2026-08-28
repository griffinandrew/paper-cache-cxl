//! `FifoCompactStack` — `FifoStack`'s policy over the slab design.
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

/// Fidelity against `FifoStack`, whose policy this re-lays-out. Same access
/// sequence must give the same eviction order: this changes how the queue is
/// STORED, not what the policy means.
#[cfg(test)]
mod fidelity_tests {
	use super::*;
	use super::super::fifo_stack::FifoStack;

	fn replay(ops: &[(HashedKey, bool)]) -> (Vec<HashedKey>, Vec<HashedKey>) {
		let mut a = FifoStack::default();
		let mut b = FifoCompactStack::default();

		for &(key, is_update) in ops {
			let sa: &mut dyn PolicyStack = &mut a;
			let sb: &mut dyn PolicyStack = &mut b;
			if is_update {
				sa.update(key);
				sb.update(key);
			} else {
				sa.insert(key, 1_024);
				sb.insert(key, 1_024);
			}
			assert_eq!(sa.len(), sb.len(), "len diverged at key {key}");
			assert_eq!(sa.contains(key), sb.contains(key), "contains diverged at key {key}");
		}

		let drain = |s: &mut dyn PolicyStack| {
			let mut out = Vec::new();
			while let Some(k) = s.evict_one() {
				out.push(k);
			}
			out
		};
		(drain(&mut a), drain(&mut b))
	}

	#[test]
	fn evicts_in_the_same_order_as_the_original() {
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
		let (oa, ob) = replay(&ops);
		assert_eq!(oa, ob, "eviction order diverged from FifoStack");
		assert!(!oa.is_empty());
	}

	#[test]
	fn removal_matches_the_original() {
		let mut a = FifoStack::default();
		let mut b = FifoCompactStack::default();
		for key in 0..2_000u64 {
			let sa: &mut dyn PolicyStack = &mut a;
			let sb: &mut dyn PolicyStack = &mut b;
			sa.insert(key, 512);
			sb.insert(key, 512);
			if key % 3 == 0 { sa.update(key); sb.update(key); }
		}
		for key in (0..2_000u64).step_by(5) {
			let sa: &mut dyn PolicyStack = &mut a;
			let sb: &mut dyn PolicyStack = &mut b;
			sa.remove(key);
			sb.remove(key);
		}
		let drain = |s: &mut dyn PolicyStack| {
			let mut out = Vec::new();
			while let Some(k) = s.evict_one() { out.push(k); }
			out
		};
		assert_eq!(drain(&mut a), drain(&mut b), "eviction order after removals diverged");
	}
}
