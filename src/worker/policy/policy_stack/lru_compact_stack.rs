//! `LruCompactStack` — the flat (untiered) LRU stack: one queue of an
//! [`ArenaQueueSet`] with no payload.
//!
//! The `HashList`-based `LruStack` whose policy this keeps was removed in R2
//! with the other original flat stacks, so the mentions of it below are
//! historical. Its eviction orders for the op sequences in `fidelity_tests` are
//! recorded in `golden.rs` and asserted there.
//!
//! ```text
//!                    eviction stack
//!   no tiering       LruCompactStack         ArenaQueueSet<()>, 16-byte node   <- this
//!   tiered           LruCompactHybridStack   ArenaQueueSet, 32-byte node (NodePayload)
//! ```
//!
//! # The policy
//!
//! `LruStack`'s, unchanged: a new key enters at the front, a hit or an
//! overwrite (`insert` of a tracked key forwards to `update`) moves it to the
//! front, and the victim is the back. Size is ignored, exactly as
//! `LruStack::insert` ignored it: recency order is the whole of the policy, and
//! byte accounting belongs to the cache.
//!
//! # The layout
//!
//! This stack used to sit on `CompactQueueSet<()>`: the same 16-byte link-only
//! slot, found through a `HashMap<HashedKey, (u32, ())>` that stored every key
//! a second time so a probe could compare it. That measured 55.99 B/object and
//! was charged 56. `ArenaQueueSet` keeps the slot -- `ArenaSlot<()>` is key 8,
//! prev 4, next 4 -- and finds it through a keyless table of bare `u32` slot
//! numbers verified against the slot's own key, 8 B/object at the half load it
//! grows to: 24 in all, charged `FLAT_LRU_STACK_DRAM_OVERHEAD`.
//!
//! The tiered LRU is the same arena with the 16-byte `NodePayload` (size, tier,
//! residency, ...) in every node, charged 40 (`ARENA_STACK_DRAM_OVERHEAD`). The
//! two now differ by exactly that payload -- the tiering bookkeeping -- and no
//! longer by an index layout one had and the other did not.
//!
//! MEASURED with `measure_one_point` (jemalloc `stats.allocated`, release, one
//! process per point), B/object = (bytes at 2^22 - bytes at 2^20) / (2^22 - 2^20):
//!
//! ```text
//!   lru-compact, CompactQueueSet<()> (before)      55.9931
//!   lru-compact, this                              23.9920
//!   lru-compact-hybrid                             40.0059
//! ```
//!
//! Under `eviction_stacks_pmem` the slab, index and free list are allocated
//! through `Hybrid`, as `CompactQueueSet`'s were.
//!
//! `differential_tests` holds it to a naive statement of the policy, victim by
//! victim, over random histories.

use crate::{
	HashedKey,
	ObjectSize,
	PaperPolicy,
};

use super::{
	PolicyStack,
	arena_queue_set::{ArenaQueueSet, ArenaSlot},
};

/// The single recency queue. `ArenaQueueSet` supports up to `MAX_QUEUES`; a
/// non-tiered LRU needs exactly one.
const Q_LRU: usize = 0;

const _: () = assert!(
	std::mem::size_of::<ArenaSlot<()>>() == 16,
	"the flat LRU node grew past 16 bytes -- re-measure FLAT_LRU_STACK_DRAM_OVERHEAD",
);

#[derive(Default)]
pub struct LruCompactStack {
	list: ArenaQueueSet<()>,
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

	/// Size is ignored, exactly as `LruStack::insert` ignored it: recency
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

/// Fidelity to `LruStack`, whose policy this keeps and which was removed in R2:
/// the eviction order it gave for the two op sequences below is recorded in
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

/// The stack against the policy written out as plainly as it can be -- one
/// `VecDeque`, most recent at the front -- over random histories of inserts,
/// hits, removals and evictions. The goldens above pin two fixed sequences;
/// this pins the rule, with a reference that shares no structure with the
/// arena.
#[cfg(test)]
mod differential_tests {
	use std::collections::VecDeque;

	use super::*;

	#[derive(Default)]
	struct Reference {
		queue: VecDeque<HashedKey>,
	}

	impl Reference {
		fn position(&self, key: HashedKey) -> Option<usize> {
			self.queue.iter().position(|&k| k == key)
		}

		fn contains(&self, key: HashedKey) -> bool {
			self.position(key).is_some()
		}

		/// A new key enters at the front. A tracked one is touched, as a hit.
		fn insert(&mut self, key: HashedKey) {
			match self.position(key) {
				Some(_) => self.update(key),
				None => self.queue.push_front(key),
			}
		}

		fn update(&mut self, key: HashedKey) {
			if let Some(i) = self.position(key) {
				self.queue.remove(i);
				self.queue.push_front(key);
			}
		}

		fn remove(&mut self, key: HashedKey) {
			if let Some(i) = self.position(key) {
				self.queue.remove(i);
			}
		}

		fn evict_one(&mut self) -> Option<HashedKey> {
			self.queue.pop_back()
		}
	}

	/// Twenty seeds, each over its own key universe (64 to 463 keys): 40%
	/// inserts, 30% hits, 10% removals and 20% evictions, interleaved. Every
	/// victim, every length and the membership of every key touched must
	/// match, and so must the final drain.
	#[test]
	fn every_victim_matches_the_reference() {
		for seed in 1..=20u64 {
			let mut rng = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
			let mut next = || {
				rng ^= rng << 13;
				rng ^= rng >> 7;
				rng ^= rng << 17;
				rng
			};

			let mut stack = LruCompactStack::default();
			let mut reference = Reference::default();
			let keys = 64 + (seed * 37) % 400;

			for step in 0..20_000 {
				let key = (next() % keys).wrapping_mul(0x9E37_79B9_7F4A_7C15);

				match next() % 10 {
					0..=3 => {
						stack.insert(key, 100);
						reference.insert(key);
					},

					4..=6 => {
						stack.update(key);
						reference.update(key);
					},

					7 => {
						stack.remove(key);
						reference.remove(key);
					},

					_ => assert_eq!(stack.evict_one(), reference.evict_one(), "seed {seed} step {step}"),
				}

				assert_eq!(stack.len(), reference.queue.len(), "seed {seed} step {step}");
				assert_eq!(stack.contains(key), reference.contains(key), "seed {seed} step {step}");
			}

			while let Some(victim) = reference.evict_one() {
				assert_eq!(stack.evict_one(), Some(victim), "seed {seed} drain");
			}

			assert_eq!(stack.evict_one(), None, "seed {seed}: the stack outlived the reference");
			assert_eq!(stack.len(), 0);
		}
	}
}
