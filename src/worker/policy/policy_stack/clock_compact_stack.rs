//! `ClockCompactStack` — `ClockStack`'s policy over the slab design.
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

/// Fidelity against `ClockStack`, whose policy this re-lays-out. Same access
/// sequence must give the same eviction order: this changes how the queue is
/// STORED, not what the policy means.
#[cfg(test)]
mod fidelity_tests {
	use super::*;
	use super::super::clock_stack::ClockStack;

	fn replay(ops: &[(HashedKey, bool)]) -> (Vec<HashedKey>, Vec<HashedKey>) {
		let mut a = ClockStack::default();
		let mut b = ClockCompactStack::default();

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
		assert_eq!(oa, ob, "eviction order diverged from ClockStack");
		assert!(!oa.is_empty());
	}

	#[test]
	fn removal_matches_the_original() {
		let mut a = ClockStack::default();
		let mut b = ClockCompactStack::default();
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
