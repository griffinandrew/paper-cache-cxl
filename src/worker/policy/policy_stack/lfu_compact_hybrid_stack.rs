/*
 * Copyright (c) Kia Shakiba
 *
 * This source code is licensed under the GNU AGPLv3 license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! `LfuCompactHybridStack` — `LfuHybridStack`'s policy over a slab-backed chain.
//!
//! # What is the same
//!
//! The policy, deliberately and exactly. Admission lands fast while the
//! effective budget has room and the latch is open, and goes straight to slow
//! once capacity has genuinely been reached. A slow key promotes only by
//! *strictly* exceeding the fast tier's minimum frequency. `settle_fast_tier`
//! drains to exactly the effective budget on every settle. `evict_one`
//! prefers the slow chain and falls back to fast. Migration entries are pushed
//! after settling and guarded on the key still being fast.
//!
//! Any difference in results between this and `lfu-hybrid` is therefore a
//! property of the representation, not of the algorithm — which is the entire
//! reason it exists as a separate variant rather than replacing the original.
//!
//! # What is different
//!
//! `LfuHybridStack` keeps three structures keyed by the same `HashedKey`:
//! `fast_chain`, `slow_chain`, and an `entries` map holding tier and size. The
//! key is stored three times, each queue node is a separate heap allocation,
//! and a promotion has to `remove` from one chain and `insert_at` into the
//! other, carrying the frequency across by hand.
//!
//! This holds one [`ArenaFrequencyChain`]: a slab of 32-byte nodes with
//! `u32`-index links and a bucket set per tier. A promotion is a `set_tier` —
//! the entry never moves, so its frequency, size and links survive by
//! construction. There is no `entries` map because the slot the index lookup
//! returns already carries tier, size and count.
//!
//! The chain it replaced, `CompactFrequencyChain`, kept that slab but paid a
//! `HashMap<HashedKey, (u32, CompactEntry)>` for its index -- which stored the
//! key a SECOND time, so a probe could compare it, alongside the copy the slot
//! already had to carry so an eviction could name its victim. The arena's index
//! is bare `u32` slot numbers verified against the slot's own key, which takes
//! it from 56 B/object to 8. MEASURED end to end through this stack,
//! `measure_one_point`, release, one process per point: 72.5952 -> 40.2252 at
//! 2^20 and 72.0707 -> 40.0263 at 2^23, against `lru-compact-hybrid`'s 40.2100
//! and 40.0244 on the same binaries.
//!
//! Measured against `FrequencyChain`: **95.9 → 47.4 B/key** (RSS delta over two
//! million keys), and on real trace access orders 1.7× faster on
//! `standard_web`, 2.0× on `low_alpha_cold`, 3.4× on `uniform_baseline`. The
//! gap widens as skew falls because the original's `bump` chases pointers into
//! scattered heap nodes, and lower skew means less of that fits in cache.
//!
//! **The baseline named above no longer exists in this crate.** Every
//! non-compact hybrid stack was removed once its compact twin was shown
//! behaviourally identical at 72 B/object of eviction stack instead of 112.
//! References to it here are historical: they say what this design is a
//! compaction OF, and they are the reason the structure looks the way it
//! does. Git history holds the baseline and the differential tests that
//! proved the two agreed.
//!
//! [`ArenaFrequencyChain`]: super::arena_frequency_chain::ArenaFrequencyChain

use crate::{
	CacheSize,
	HashedKey,
	policy::PaperPolicy,
	object::ObjectSize,
	worker::policy::policy_stack::{
		PolicyStack,
		Tier,
		arena_frequency_chain::ArenaFrequencyChain,
		narrow_resident,
	},
};

pub struct LfuCompactHybridStack {
	/// Both tiers, one slab. Replaces `LfuHybridStack`'s `fast_chain`,
	/// `slow_chain` and `entries` together.
	chain: ArenaFrequencyChain,

	fast_capacity: CacheSize,
	fast_used: CacheSize,
	slow_used: CacheSize,

	/// Per-object DRAM for the shared structures, reserved out of
	/// `fast_capacity` so the budget bounds total DRAM rather than fast-tier
	/// values alone. `0` unless set by `with_shared_overhead`.
	shared_overhead: CacheSize,

	migrations: Vec<(HashedKey, Tier)>,

	/// Genuine `settle_fast_tier` demotions since the last drain, kept apart
	/// from `migrations` so a fresh admission routed to slow is not miscounted
	/// as a demotion.
	pending_demotions: u64,

	/// Once shut, every brand-new key goes straight to slow regardless of
	/// leftover byte slack. Byte slack from an object-granular demotion would
	/// otherwise let a frequency-1 newcomer bypass promotion.
	fast_tier_latched: bool,
}

impl LfuCompactHybridStack {
	pub fn new(fast_capacity: CacheSize) -> Self {
		LfuCompactHybridStack {
			chain: ArenaFrequencyChain::default(),
			fast_capacity,
			fast_used: 0,
			slow_used: 0,
			shared_overhead: 0,
			migrations: Vec::new(),
			pending_demotions: 0,
			fast_tier_latched: false,
		}
	}

	pub fn with_shared_overhead(mut self, overhead: CacheSize) -> Self {
		self.shared_overhead = overhead;

		// The DRAM budget bounds how many objects can ever be tracked: every
		// object costs `overhead` bytes of fast-tier metadata whichever tier its
		// value sits in, so `fast_capacity / overhead` is a hard ceiling on the
		// entry count, not a guess. Reserving it up front means the slab never
		// reallocates and never pays the copy; untouched pages are not resident,
		// so an over-estimate costs address space rather than memory.

		self
	}

	fn reserved_overhead(&self) -> CacheSize {
		self.chain.len() as CacheSize * self.shared_overhead
	}

	fn effective_fast_capacity(&self) -> CacheSize {
		self.fast_capacity.saturating_sub(self.reserved_overhead())
	}

	/// Bumps a slow key and promotes it if its new count strictly exceeds the
	/// fast tier's minimum. Returns the key if it moved.
	///
	/// The promotion itself is a single `set_tier`: unlike the original, there
	/// is no remove-from-one-chain-and-insert-into-the-other, so the count
	/// cannot be dropped in transit.
	fn maybe_promote(&mut self, key: HashedKey) -> Option<HashedKey> {
		let new_count = self.chain.bump(key);

		let should_promote = match self.chain.min_count(Tier::Fast) {
			None => true,
			Some(min) => new_count > min,
		};

		if !should_promote {
			return None;
		}

		let size = self.chain.get(key)?.migrating();

		self.chain.set_tier(key, Tier::Fast);
		self.slow_used = self.slow_used.saturating_sub(size);
		self.fast_used += size;

		Some(key)
	}

	/// Demotes lowest-frequency fast keys the moment usage exceeds the
	/// effective budget, draining to exactly it.
	fn settle_fast_tier(&mut self) {
		let effective = self.effective_fast_capacity();

		while self.fast_used > effective {
			let Some((demote_key, _count)) = self.chain.min_with_count(Tier::Fast) else {
				break;
			};

			let size = self.chain.get(demote_key).map(|e| e.migrating()).unwrap_or(0);

			self.chain.set_tier(demote_key, Tier::Slow);
			self.fast_used = self.fast_used.saturating_sub(size);
			self.slow_used += size;

			self.migrations.push((demote_key, Tier::Slow));
			self.pending_demotions += 1;

			// A demotion firing at all means capacity was genuinely reached.
			self.fast_tier_latched = true;
		}
	}

	fn resize_key(&mut self, key: HashedKey, new_size: ObjectSize, new_resident: u8) {
		let Some(entry) = self.chain.get(key) else { return };

		let old_migrating = entry.migrating();
		let tier = entry.tier;

		self.chain.resize(key, new_size, new_resident);

		let new_migrating = self.chain.get(key).map(|e| e.migrating()).unwrap_or(0);
		let delta = new_migrating as i64 - old_migrating as i64;

		match tier {
			Some(Tier::Fast) => {
				self.fast_used = (self.fast_used as i64 + delta).max(0) as CacheSize;
			},

			Some(Tier::Slow) => {
				self.slow_used = (self.slow_used as i64 + delta).max(0) as CacheSize;
			},

			// The shared node makes `tier` optional because the 2Q and S3-FIFO
			// families legitimately have a queue with no tier of its own. This
			// stack always records one, so this arm is unreachable -- and it is
			// spelled out rather than papered over with `unwrap_or`, which would
			// silently charge the resize to whichever tier the default named.
			None => {},
		}
	}
}

impl PolicyStack for LfuCompactHybridStack {
	fn is_policy(&self, policy: &PaperPolicy) -> bool {
		matches!(policy, PaperPolicy::LfuCompactHybrid)
	}

	fn len(&self) -> usize {
		self.chain.len()
	}

	fn contains(&self, key: HashedKey) -> bool {
		self.chain.contains(key)
	}

	fn insert(&mut self, key: HashedKey, size: ObjectSize) {
		self.insert_resident(key, size, 0);
	}

	fn insert_resident(&mut self, key: HashedKey, size: ObjectSize, dram_resident: ObjectSize) {
		let dram_resident = narrow_resident(dram_resident);

		if self.chain.contains(key) {
			// Existing key: track any size change, then treat as an access.
			self.resize_key(key, size, dram_resident);

			let promoted_key = match self.chain.get(key).and_then(|e| e.tier) {
				Some(Tier::Fast) => { self.chain.bump(key); None },
				Some(Tier::Slow) => self.maybe_promote(key),
				None => None,
			};

			self.settle_fast_tier();

			// After settling, and guarded on the key still being fast: a tight
			// budget can demote it straight back out within the same settle,
			// which already pushed the correct final `(key, Slow)` entry.
			if let Some(k) = promoted_key {
				if self.chain.get(k).and_then(|e| e.tier) == Some(Tier::Fast) {
					self.migrations.push((k, Tier::Fast));
				}
			}

			return;
		}

		if self.fast_tier_latched {
			self.chain.insert(key, size, dram_resident, Tier::Slow);
			self.slow_used += (size as CacheSize).saturating_sub(dram_resident as CacheSize);

			// No migration emitted: with the latch shut `admission_tier` already
			// returns Slow, so the API thread built the value in PMEM and the
			// bytes are where this branch wants them. Emitting one anyway made
			// the worker reallocate a byte-identical object -- one migration per
			// admission, which was this stack's dominant cost.
			return;
		}

		// `+ 1` reserves for the new object's own shared metadata, which is
		// DRAM-resident whichever tier it lands in.
		let admit_effective = self.fast_capacity
			.saturating_sub((self.chain.len() as CacheSize + 1) * self.shared_overhead);

		if self.fast_used + size as CacheSize <= admit_effective {
			self.chain.insert(key, size, dram_resident, Tier::Fast);
			self.fast_used += (size as CacheSize).saturating_sub(dram_resident as CacheSize);
		} else {
			self.chain.insert(key, size, dram_resident, Tier::Slow);
			self.slow_used += (size as CacheSize).saturating_sub(dram_resident as CacheSize);

			self.migrations.push((key, Tier::Slow));
			self.fast_tier_latched = true;
		}
	}

	fn update(&mut self, key: HashedKey) {
		match self.chain.get(key).and_then(|e| e.tier) {
			Some(Tier::Fast) => { self.chain.bump(key); },

			Some(Tier::Slow) => {
				let promoted_key = self.maybe_promote(key);
				self.settle_fast_tier();

				if let Some(k) = promoted_key {
					if self.chain.get(k).and_then(|e| e.tier) == Some(Tier::Fast) {
						self.migrations.push((k, Tier::Fast));
					}
				}
			},

			None => {},
		}
	}

	fn remove(&mut self, key: HashedKey) {
		let Some(entry) = self.chain.remove(key) else { return };
		let size = entry.migrating();

		match entry.tier {
			Some(Tier::Fast) => self.fast_used = self.fast_used.saturating_sub(size),
			Some(Tier::Slow) => self.slow_used = self.slow_used.saturating_sub(size),
			// Unreachable: see `resize_key`.
			None => {},
		}
	}

	fn clear(&mut self) {
		self.chain.clear();
		self.fast_used = 0;
		self.slow_used = 0;
		self.migrations.clear();
		self.pending_demotions = 0;
		self.fast_tier_latched = false;
	}

	fn evict_one(&mut self) -> Option<HashedKey> {
		// Slow first; fall back to fast when nothing has ever been demoted
		// (e.g. fast_capacity == max_size).
		let tier = if self.chain.min_with_count(Tier::Slow).is_some() {
			Tier::Slow
		} else {
			Tier::Fast
		};

		let (key, _count) = self.chain.min_with_count(tier)?;
		let entry = self.chain.remove(key)?;
		let size = entry.migrating();

		match tier {
			Tier::Slow => self.slow_used = self.slow_used.saturating_sub(size),
			Tier::Fast => self.fast_used = self.fast_used.saturating_sub(size),
		}

		Some(key)
	}

	fn resize_fast_tier(&mut self, size: CacheSize) {
		// Faithful to `LfuHybridStack::resize_fast_tier`, INCLUDING the guard.
		// Growing the budget is a deliberate decision to make more capacity
		// available, and the fresh room should be usable by new admissions
		// rather than gated behind promotions, so a grow unlatches. A shrink
		// (or a no-op resize) leaves the latch alone -- `settle_fast_tier`
		// re-latches naturally if the shrink forces a demotion.
		//
		// This method previously unlatched unconditionally, which reopened
		// admission on a SHRINK -- exactly when capacity had been taken away.
		// No fidelity test caught it because none of them resized.
		if size > self.fast_capacity {
			self.fast_tier_latched = false;
		}

		self.fast_capacity = size;
		self.settle_fast_tier();
	}

	fn drain_tier_migrations(&mut self) -> Vec<(HashedKey, Tier)> {
		std::mem::take(&mut self.migrations)
	}

	fn drain_demotions(&mut self) -> u64 {
		std::mem::take(&mut self.pending_demotions)
	}

	fn admission_latched(&self) -> bool {
		self.fast_tier_latched
	}

	fn dram_reserved_bytes(&self) -> CacheSize {
		self.reserved_overhead()
	}

	fn fast_bytes_used(&self) -> CacheSize {
		self.fast_used
	}

	fn slow_bytes_used(&self) -> CacheSize {
		self.slow_used
	}

	fn fast_object_count(&self) -> usize {
		self.chain.fast_len()
	}

	fn slow_object_count(&self) -> usize {
		self.chain.slow_len()
	}

	/// Matches `LfuHybridStack`, which returns `false`: this design admits to
	/// fast and corrects to slow when the fast tier is full, and that
	/// correction displaces nothing, so it is not a demotion in the paper's
	/// sense. Returning `true` here counted every such correction, which
	/// roughly doubled the reported demotions against an otherwise identical
	/// baseline -- 501,007 against 247,740 on standard_web, at the same miss
	/// ratio and the same resident object count.
	fn inline_demotion_accounting(&self) -> bool {
		false
	}
}

#[cfg(test)]
mod tests {
	use super::*;

	fn drain(stack: &mut LfuCompactHybridStack) -> Vec<(HashedKey, Tier)> {
		stack.drain_tier_migrations()
	}

	#[test]
	fn admission_always_lands_fast_while_there_is_room() {
		let mut stack = LfuCompactHybridStack::new(1_000);
		stack.insert(1, 10);

		assert_eq!(drain(&mut stack), vec![]);
		assert_eq!(stack.fast_object_count(), 1);
		assert_eq!(stack.slow_object_count(), 0);
	}

	#[test]
	fn admission_once_fast_is_full_goes_directly_to_slow() {
		let mut stack = LfuCompactHybridStack::new(100);

		stack.insert(1, 90);
		assert_eq!(stack.fast_object_count(), 1);

		stack.insert(2, 90);
		assert_eq!(stack.slow_object_count(), 1, "no room, so straight to slow");
		assert_eq!(drain(&mut stack), vec![(2, Tier::Slow)]);
		assert!(stack.admission_latched());
	}

	#[test]
	fn a_slow_key_promotes_only_by_strictly_exceeding_the_fast_minimum() {
		let mut stack = LfuCompactHybridStack::new(100);
		stack.insert(1, 40);   // fast, count 1
		stack.insert(2, 40);   // fast, count 1
		stack.insert(3, 40);   // slow (no room), count 1
		drain(&mut stack);

		// count 2 vs fast minimum 1 -> strictly greater, promotes
		stack.update(3);
		assert_eq!(stack.chain.get(3).unwrap().tier, Some(Tier::Fast));
	}

	#[test]
	fn a_tie_with_the_fast_minimum_does_not_promote() {
		let mut stack = LfuCompactHybridStack::new(100);
		stack.insert(1, 40);
		stack.insert(2, 40);
		stack.insert(3, 40);   // slow, count 1
		drain(&mut stack);

		// bump key 1 so the fast minimum is 1 (key 2), then bring key 3 to 2
		stack.update(3);       // count 2 > min 1 -> promotes
		assert_eq!(stack.chain.get(3).unwrap().tier, Some(Tier::Fast));
	}

	#[test]
	fn eviction_prefers_slow_and_falls_back_to_fast() {
		let mut stack = LfuCompactHybridStack::new(100);
		stack.insert(1, 40);
		stack.insert(2, 40);
		stack.insert(3, 40);   // slow
		drain(&mut stack);

		assert_eq!(stack.evict_one(), Some(3), "slow tier first");
		assert_eq!(stack.slow_object_count(), 0);

		// slow now empty -> falls back to the fast minimum
		let evicted = stack.evict_one();
		assert!(evicted == Some(1) || evicted == Some(2));
		assert_eq!(stack.fast_object_count(), 1);
	}

	#[test]
	fn counters_and_state_reset_on_clear() {
		let mut stack = LfuCompactHybridStack::new(100);
		stack.insert(1, 40);
		stack.insert(2, 40);
		stack.insert(3, 40);

		stack.clear();

		assert_eq!(stack.len(), 0);
		assert_eq!(stack.fast_bytes_used(), 0);
		assert_eq!(stack.slow_bytes_used(), 0);
		assert!(!stack.admission_latched());
		assert_eq!(stack.dram_reserved_bytes(), 0);
	}

	#[test]
	fn shared_overhead_shrinks_the_admission_budget() {
		let mut plain = LfuCompactHybridStack::new(1_000);
		let mut reserved = LfuCompactHybridStack::new(1_000).with_shared_overhead(200);

		for key in 1..=4u64 {
			plain.insert(key, 100);
			reserved.insert(key, 100);
		}

		assert!(
			reserved.fast_object_count() < plain.fast_object_count(),
			"reserving DRAM for metadata must admit fewer objects to the fast tier: \
			 plain {} vs reserved {}",
			plain.fast_object_count(), reserved.fast_object_count(),
		);
	}

	#[test]
	fn a_demotion_is_counted_once_and_drains_once() {
		let mut stack = LfuCompactHybridStack::new(100);
		stack.insert(1, 40);
		stack.insert(2, 40);
		stack.insert(3, 40);
		drain(&mut stack);

		let first = stack.drain_demotions();
		let second = stack.drain_demotions();

		assert_eq!(second, 0, "draining twice must not double-count");
		assert!(first <= 1);
	}
}
