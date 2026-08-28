//! `SieveCompactStack` — `SieveStack`'s policy over the slab design.
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

/// Fidelity against `SieveStack`, whose policy this re-lays-out. Same access
/// sequence must give the same eviction order: this changes how the queue is
/// STORED, not what the policy means.
#[cfg(test)]
mod fidelity_tests {
	use super::*;
	use super::super::sieve_stack::SieveStack;

	fn replay(ops: &[(HashedKey, bool)]) -> (Vec<HashedKey>, Vec<HashedKey>) {
		let mut a = SieveStack::default();
		let mut b = SieveCompactStack::default();

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
		assert_eq!(oa, ob, "eviction order diverged from SieveStack");
		assert!(!oa.is_empty());
	}

	#[test]
	fn removal_matches_the_original() {
		let mut a = SieveStack::default();
		let mut b = SieveCompactStack::default();
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
