//! `LfuCompactStack` — the flat (untiered) LFU stack: `LfuCompactHybridStack`'s
//! [`ArenaFrequencyChain`] with every key in one tier.
//!
//! The `HashList`-based `LfuStack` whose policy this keeps was removed in R2
//! with the other original flat stacks, so the mentions of it below are
//! historical. Its eviction orders for the op sequences in `fidelity_tests` are
//! recorded in `golden.rs` and asserted there.
//!
//! ```text
//!                    eviction stack
//!   no tiering       LfuCompactStack         ArenaFrequencyChain, one tier   <- this
//!   tiered           LfuCompactHybridStack   ArenaFrequencyChain, two tiers
//! ```
//!
//! # The policy
//!
//! `LfuStack`'s, unchanged. A new key enters at count 1; a hit, or an overwrite
//! (`insert` of a tracked key forwards to `update`), raises its count by one;
//! the victim is the least-frequent key, ties broken toward the key that
//! reached its count EARLIEST. Every touch moves a key up one count, so that is
//! also the least recently touched. The chain gives this order by construction:
//! `link` appends a key at the TAIL of its bucket and `min_key` reads the HEAD.
//! The old slab stated the same order from the other end (link at the head,
//! evict the tail).
//!
//! Size is recorded in the node and never read: `LfuStack` ignored it, and
//! byte accounting belongs to the cache.
//!
//! # The layout, and why it is the tiered one
//!
//! This stack used to own its structures: a slab of 16-byte link-only slots, a
//! `HashMap<HashedKey, (u32 slot, u32 freq)>` index that stored every key a
//! second time, and a `BTreeMap` of per-frequency bucket lists. That measured
//! 55.99 B/object and was charged 56, against the 40 every tiered design is
//! charged for its arena (`ARENA_STACK_DRAM_OVERHEAD`). At equal `max_size` a
//! flat LFU cache therefore held fewer objects than a tiered one, for a reason
//! that has nothing to do with tiering.
//!
//! Over the tiered LFU's own chain the two pay the same per-object stack: a
//! 32-byte node (key, prev, next and the shared 16-byte `NodePayload`) plus the
//! keyless index, bare `u32` slot numbers verified against the node's own key,
//! 8 B/object at the half load it grows to. Only `freq` of the payload is read
//! here; the tier, size and residency fields are carried because
//! `ArenaFrequencyChain` is not generic over its payload. (Flat LRU, over the
//! generic `ArenaQueueSet`, drops the payload and costs 24.) The slow bucket map
//! and the recency list stay empty.
//!
//! MEASURED with `measure_one_point` (jemalloc `stats.allocated`, release, one
//! process per point), B/object = (bytes at 2^22 - bytes at 2^20) / (2^22 - 2^20):
//!
//! ```text
//!   lfu-compact, slab + keyed HashMap (before)     55.9931
//!   lfu-compact, this                              40.0059
//!   lfu-compact-hybrid                             40.0059
//! ```
//!
//! Under `eviction_stacks_pmem` the chain allocates its slab, index, free list
//! and bucket maps through `Hybrid`, as the old stack's collections did.
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
	Tier,
	arena_frequency_chain::ArenaFrequencyChain,
};

/// Every key lives in this tier; the chain's other one is never touched.
const TIER: Tier = Tier::Fast;

#[derive(Default)]
pub struct LfuCompactStack {
	chain: ArenaFrequencyChain,
}

impl PolicyStack for LfuCompactStack {
	fn is_policy(&self, policy: &PaperPolicy) -> bool {
		matches!(policy, PaperPolicy::LfuCompact)
	}

	fn len(&self) -> usize {
		self.chain.len()
	}

	fn contains(&self, key: HashedKey) -> bool {
		self.chain.contains(key)
	}

	fn insert(&mut self, key: HashedKey, size: ObjectSize) {
		if self.chain.contains(key) {
			return self.update(key);
		}

		self.chain.insert(key, size, 0, TIER);
	}

	fn update(&mut self, key: HashedKey) {
		self.chain.bump(key);
	}

	fn remove(&mut self, key: HashedKey) {
		self.chain.remove(key);
	}

	fn clear(&mut self) {
		self.chain.clear();
	}

	/// Evicts the least-frequent key, breaking ties toward the one that reached
	/// its count earliest -- the head of the lowest bucket.
	fn evict_one(&mut self) -> Option<HashedKey> {
		let key = self.chain.min_key(TIER)?;
		self.chain.remove(key);

		Some(key)
	}
}

/// Fidelity to `LfuStack`, whose policy this keeps and which was removed in R2:
/// the eviction order it gave for the two op sequences below is recorded in
/// `golden` and asserted here. Same access sequence, same eviction order: this
/// changes how the queue is STORED, not what the policy means.
#[cfg(test)]
mod fidelity_tests {
	use super::*;
	use super::super::golden;

	/// Same access sequence, same eviction order: this changes how frequency is
	/// STORED, not what LFU means.
	#[test]
	fn evicts_in_the_recorded_order() {
		assert_eq!(
			golden::lfu_skewed(&mut LfuCompactStack::default()),
			golden::LFU_SKEWED,
		);
	}

	/// Removal must not disturb the order of what remains.
	#[test]
	fn removal_leaves_the_recorded_order() {
		assert_eq!(
			golden::removal(&mut LfuCompactStack::default(), 2_000, 512, true, 5),
			golden::LFU_REMOVAL,
		);
	}
}

/// The stack against the policy written out as plainly as it can be, over
/// random histories of inserts, hits, removals and evictions.
///
/// The goldens above pin two fixed sequences. This pins the RULE, and it is
/// the check that matters for the tie order: the stack shares its chain with
/// `LfuCompactHybridStack`, so agreeing with that stack would prove nothing
/// about either. The reference shares nothing with it -- no buckets, no links,
/// no index -- so a chain that broke ties toward the wrong end of a bucket, or
/// lost a key's count across a removal and a re-insert, fails here. (Checked
/// when this was written: with the reference's tie-break turned newest-first,
/// every one of the twenty seeds diverged.)
#[cfg(test)]
mod differential_tests {
	use super::*;

	/// Every tracked key with its count and the tick it was last touched; the
	/// victim is the minimum `(count, last touch)`, found by a scan.
	#[derive(Default)]
	struct Reference {
		keys: Vec<(HashedKey, u32, u64)>,
		clock: u64,
	}

	impl Reference {
		fn position(&self, key: HashedKey) -> Option<usize> {
			self.keys.iter().position(|&(k, _, _)| k == key)
		}

		fn contains(&self, key: HashedKey) -> bool {
			self.position(key).is_some()
		}

		/// A new key enters at count 1. A tracked one is touched, as a hit.
		fn insert(&mut self, key: HashedKey) {
			match self.position(key) {
				Some(_) => self.update(key),

				None => {
					self.clock += 1;
					self.keys.push((key, 1, self.clock));
				},
			}
		}

		fn update(&mut self, key: HashedKey) {
			let Some(i) = self.position(key) else { return };

			self.clock += 1;
			self.keys[i].1 += 1;
			self.keys[i].2 = self.clock;
		}

		fn remove(&mut self, key: HashedKey) {
			if let Some(i) = self.position(key) {
				self.keys.swap_remove(i);
			}
		}

		fn evict_one(&mut self) -> Option<HashedKey> {
			let i = (0..self.keys.len()).min_by_key(|&i| (self.keys[i].1, self.keys[i].2))?;

			Some(self.keys.swap_remove(i).0)
		}
	}

	/// Twenty seeds, each over its own key universe (64 to 463 keys) so the
	/// counts spread differently: 40% inserts, 30% hits, 10% removals and 20%
	/// evictions, interleaved. Every victim, every length and the membership of
	/// every key touched must match, and so must the final drain.
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

			let mut stack = LfuCompactStack::default();
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

				assert_eq!(stack.len(), reference.keys.len(), "seed {seed} step {step}");
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
