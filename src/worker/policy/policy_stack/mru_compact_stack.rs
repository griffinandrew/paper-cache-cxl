//! `MruCompactStack` — `MruStack`'s policy over the slab design.
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

/// Fidelity against `MruStack`, whose policy this re-lays-out. Same access
/// sequence must give the same eviction order: this changes how the queue is
/// STORED, not what the policy means.
#[cfg(test)]
mod fidelity_tests {
	use super::*;
	use super::super::mru_stack::MruStack;

	fn replay(ops: &[(HashedKey, bool)]) -> (Vec<HashedKey>, Vec<HashedKey>) {
		let mut a = MruStack::default();
		let mut b = MruCompactStack::default();

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
		assert_eq!(oa, ob, "eviction order diverged from MruStack");
		assert!(!oa.is_empty());
	}

	#[test]
	fn removal_matches_the_original() {
		let mut a = MruStack::default();
		let mut b = MruCompactStack::default();
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
