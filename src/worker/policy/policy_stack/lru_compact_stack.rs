//! `LruCompactStack` — `LruStack`'s policy over the slab design.
//!
//! Exists to separate two effects the existing matrix confounds. Comparing
//! `Lru` (all-DRAM, `HashList`) against `LruHybrid` (tiered, `HashList`)
//! measures TIERING. Comparing `LruHybrid` against `LruCompactHybrid`
//! measures LAYOUT. But comparing all-DRAM against a tiered compact stack
//! measures both at once, which is how a 23% throughput gap on cluster13 LRU
//! and a *reversed* 24% gap on cluster13 LFU both appeared without either
//! being attributable.
//!
//! This is the missing cell: the compact LAYOUT with no tiering at all.
//!
//! ```text
//!                    HashList layout        slab layout
//!   no tiering       LruStack               LruCompactStack   <- this
//!   tiered           LruHybridStack         LruCompactHybridStack
//! ```
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
	CacheSize,
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

/// Fidelity against `LruStack`, whose policy this is a re-layout of.
///
/// The two must produce the same eviction order for the same access
/// sequence: this changes how recency is STORED, not what recency means.
#[cfg(test)]
mod fidelity_tests {
	use super::*;
	use super::super::lru_stack::LruStack;

	fn replay(ops: &[(HashedKey, bool)]) -> (Vec<HashedKey>, Vec<HashedKey>) {
		let mut a = LruStack::default();
		let mut b = LruCompactStack::default();

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

	/// Skewed access with reuse, so keys are repeatedly moved to the front —
	/// the operation the two layouts implement differently.
	#[test]
	fn evicts_in_the_same_order_as_lru_stack() {
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

		let (base, compact) = replay(&ops);
		assert_eq!(base, compact, "eviction order diverged from LruStack");
		assert!(!base.is_empty(), "the replay must actually evict something");
	}

	/// Removal must not disturb the order of what remains.
	#[test]
	fn removal_matches_lru_stack() {
		let mut a = LruStack::default();
		let mut b = LruCompactStack::default();

		for key in 0..1_000u64 {
			let sa: &mut dyn PolicyStack = &mut a;
			let sb: &mut dyn PolicyStack = &mut b;
			sa.insert(key, 512);
			sb.insert(key, 512);
		}
		for key in (0..1_000u64).step_by(3) {
			let sa: &mut dyn PolicyStack = &mut a;
			let sb: &mut dyn PolicyStack = &mut b;
			sa.remove(key);
			sb.remove(key);
		}

		let sa: &mut dyn PolicyStack = &mut a;
		let sb: &mut dyn PolicyStack = &mut b;
		assert_eq!(sa.len(), sb.len());
		let mut order_a = Vec::new();
		let mut order_b = Vec::new();
		while let Some(k) = sa.evict_one() { order_a.push(k); }
		while let Some(k) = sb.evict_one() { order_b.push(k); }
		assert_eq!(order_a, order_b, "eviction order after removals diverged");
	}
}
