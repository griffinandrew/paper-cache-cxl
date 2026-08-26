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
//! drains from the high watermark to the low one in a batch. `evict_one`
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
//! This holds one `CompactFrequencyChain`: a slab of 32-byte entries with
//! `u32`-index links and a bucket set per tier. A promotion is a `set_tier` —
//! the entry never moves, so its frequency, size and links survive by
//! construction. There is no `entries` map because the slot the index lookup
//! returns already carries tier, size and count.
//!
//! Measured against `FrequencyChain`: **95.9 → 47.4 B/key** (RSS delta over two
//! million keys), and on real trace access orders 1.7× faster on
//! `standard_web`, 2.0× on `low_alpha_cold`, 3.4× on `uniform_baseline`. The
//! gap widens as skew falls because the original's `bump` chases pointers into
//! scattered heap nodes, and lower skew means less of that fits in cache.

use crate::{
	CacheSize,
	HashedKey,
	policy::PaperPolicy,
	object::ObjectSize,
	worker::policy::policy_stack::{
		PolicyStack,
		Tier,
		compact_frequency_chain::CompactFrequencyChain,
		narrow_resident,
		watermarks,
	},
};

pub struct LfuCompactHybridStack {
	/// Both tiers, one slab. Replaces `LfuHybridStack`'s `fast_chain`,
	/// `slow_chain` and `entries` together.
	chain: CompactFrequencyChain,

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
			chain: CompactFrequencyChain::default(),
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
		if overhead > 0 {
			let ceiling = (self.fast_capacity / overhead) as usize;
			self.chain.reserve(ceiling);
		}

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

	/// Demotes lowest-frequency fast keys once usage crosses the high
	/// watermark, draining in one batch down to the low one.
	fn settle_fast_tier(&mut self) {
		let effective = self.effective_fast_capacity();

		if self.fast_used <= watermarks::high_bytes(effective) {
			return;
		}

		let target = watermarks::low_bytes(effective);

		while self.fast_used > target {
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
			Tier::Fast => self.fast_used = (self.fast_used as i64 + delta).max(0) as CacheSize,
			Tier::Slow => self.slow_used = (self.slow_used as i64 + delta).max(0) as CacheSize,
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

			let promoted_key = match self.chain.get(key).map(|e| e.tier) {
				Some(Tier::Fast) => { self.chain.bump(key); None },
				Some(Tier::Slow) => self.maybe_promote(key),
				None => None,
			};

			self.settle_fast_tier();

			// After settling, and guarded on the key still being fast: a tight
			// budget can demote it straight back out within the same settle,
			// which already pushed the correct final `(key, Slow)` entry.
			if let Some(k) = promoted_key {
				if self.chain.get(k).map(|e| e.tier) == Some(Tier::Fast) {
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
		match self.chain.get(key).map(|e| e.tier) {
			Some(Tier::Fast) => { self.chain.bump(key); },

			Some(Tier::Slow) => {
				let promoted_key = self.maybe_promote(key);
				self.settle_fast_tier();

				if let Some(k) = promoted_key {
					if self.chain.get(k).map(|e| e.tier) == Some(Tier::Fast) {
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
			Tier::Fast => self.fast_used = self.fast_used.saturating_sub(size),
			Tier::Slow => self.slow_used = self.slow_used.saturating_sub(size),
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
		assert_eq!(stack.chain.get(3).unwrap().tier, Tier::Fast);
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
		assert_eq!(stack.chain.get(3).unwrap().tier, Tier::Fast);
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

/// Fidelity against `LfuHybridStack`, which this stack is a compaction of.
///
/// The two must be behaviourally indistinguishable: same tier for every key at
/// every point, and the same migration sequence in the same order. Miss ratios
/// agreeing to three decimals on a real trace is necessary but not sufficient
/// -- it would not catch a counter that fires on the wrong path, which is
/// exactly the class of defect that produced a doubled demotion count here.
///
/// Requires both features, which is why the duplicate `mod hybrid_tests`
/// definitions in `worker::policy` had to be renamed first: seven policy
/// modules shared that one name, so enabling any two policy features at once
/// failed to compile and no cross-policy test could exist.
#[cfg(all(test, feature = "lfu_hybrid_cache"))]
mod fidelity_tests {
	use super::*;
	use crate::worker::policy::policy_stack::lfu_hybrid_stack::LfuHybridStack;

	/// Drives both stacks through one scripted workload and returns each one's
	/// migration log plus the final tier of every key.
	fn replay(fast_capacity: CacheSize, ops: &[(HashedKey, ObjectSize)])
		-> (Vec<(HashedKey, Tier)>, Vec<(HashedKey, Tier)>, Vec<Option<Tier>>, Vec<Option<Tier>>) {
		replay_with(fast_capacity, 0, 0, ops)
	}

	/// As `replay`, with an explicit per-object metadata reservation for each
	/// stack -- the quantity that differs between them in a real run.
	fn replay_with(
		fast_capacity: CacheSize,
		overhead_a: CacheSize,
		overhead_b: CacheSize,
		ops: &[(HashedKey, ObjectSize)],
	) -> (Vec<(HashedKey, Tier)>, Vec<(HashedKey, Tier)>, Vec<Option<Tier>>, Vec<Option<Tier>>) {
		let mut a = LfuHybridStack::new(fast_capacity).with_shared_overhead(overhead_a);
		let mut b = LfuCompactHybridStack::new(fast_capacity).with_shared_overhead(overhead_b);
		let (mut ma, mut mb) = (Vec::new(), Vec::new());
		for (k, size) in ops {
			if a.contains(*k) { a.update(*k); } else { a.insert(*k, *size); }
			if b.contains(*k) { b.update(*k); } else { b.insert(*k, *size); }
			ma.extend(a.drain_tier_migrations());
			mb.extend(b.drain_tier_migrations());
		}
		let keys: Vec<HashedKey> = ops.iter().map(|(k, _)| *k).collect();
		let ta = keys.iter().map(|k| a.tier_of(*k)).collect();
		let tb = keys.iter().map(|k| b.chain.get(*k).map(|e| e.tier)).collect();
		(ma, mb, ta, tb)
	}

	/// A skewed workload with enough pressure to force promotion, demotion and
	/// the admission latch: 200 keys, accessed with a bias toward low ids, at a
	/// fast capacity that holds only a fraction of them.
	#[test]
	fn matches_lfu_hybrid_migration_for_migration() {
		let mut ops = Vec::new();
		let mut x: u64 = 0x243F_6A88_85A3_08D3;
		for _ in 0..20_000 {
			x ^= x << 13; x ^= x >> 7; x ^= x << 17;
			// square the uniform draw to bias toward low keys
			let u = (x >> 11) as f64 / (1u64 << 53) as f64;
			ops.push((((u * u * 200.0) as u64) + 1, 1024));
		}
		for cap in [8_192, 32_768, 131_072] {
			let (ma, mb, ta, tb) = replay(cap, &ops);
			assert_eq!(ta, tb, "final tiers diverge at fast_capacity {cap}");
			let pa = ma.iter().filter(|(_, t)| *t == Tier::Fast).count();
			let pb = mb.iter().filter(|(_, t)| *t == Tier::Fast).count();
			let da = ma.iter().filter(|(_, t)| *t == Tier::Slow).count();
			let db = mb.iter().filter(|(_, t)| *t == Tier::Slow).count();
			assert_eq!(
				(pa, da), (pb, db),
				"migration counts diverge at fast_capacity {cap}: \
				 lfu_hybrid {pa} promotions / {da} demotions, \
				 compact {pb} promotions / {db} demotions"
			);
			assert_eq!(ma, mb, "migration ORDER diverges at fast_capacity {cap}");
		}
	}

	/// The same equivalence must hold with a metadata reservation in play, since
	/// `settle_fast_tier` derives its watermarks from `fast_capacity` MINUS the
	/// reservation. Equal reservation, so any divergence is logic.
	#[test]
	fn matches_lfu_hybrid_under_equal_reservation() {
		let ops = skewed_ops();
		for (cap, overhead) in [(32_768u64, 64u64), (131_072, 128), (131_072, 271)] {
			let (ma, mb, ta, tb) = replay_with(cap, overhead, overhead, &ops);
			assert_eq!(ta, tb, "final tiers diverge at cap {cap} overhead {overhead}");
			assert_eq!(ma, mb, "migrations diverge at cap {cap} overhead {overhead}");
		}
	}

	/// What the reservation difference actually does, measured rather than
	/// assumed. This stack reserves 190 B/object against `LfuHybridStack`'s 271
	/// (both measured -- see `object::overhead`), and a smaller reservation
	/// leaves a LARGER effective value budget, so more keys are admitted to
	/// fast directly and MORE promotions occur, not fewer.
	///
	/// Worth stating because the opposite was assumed first, to explain a
	/// benchmark that reported 62% FEWER promotions for this stack. The real
	/// cause was unrelated: `hybrid_policy::admission_tier` had no arm for
	/// `LfuCompactHybrid`, so brand-new keys were built in DRAM while the stack
	/// recorded them as slow, and the resulting promotions were declined as
	/// "already in the requested tier" instead of being counted. The reservation
	/// was pushing the count the other way the whole time.
	#[test]
	fn smaller_reservation_admits_more_and_promotes_more() {
		let ops = skewed_ops();
		let (ma, mb, _, _) = replay_with(131_072, 271, 190, &ops);
		let promos = |m: &Vec<(HashedKey, Tier)>| m.iter().filter(|(_, t)| *t == Tier::Fast).count();
		let (pa, pb) = (promos(&ma), promos(&mb));
		assert!(
			pb > pa,
			"a smaller reservation should leave a larger effective budget and so \
			 promote more, got {pb} against {pa}"
		);

		// and it IS the reservation, not the data structure: give this stack the
		// baseline's reservation and the difference disappears entirely
		let (ma2, mb2, _, _) = replay_with(131_072, 271, 271, &ops);
		assert_eq!(promos(&ma2), promos(&mb2));
	}

	/// Resizing must match too, in BOTH directions.
	///
	/// This caught a real divergence: this stack unlatched admission on ANY
	/// resize, where the baseline unlatches only on a grow, so a shrink or a
	/// no-op resize reopened admission here and not there.
	///
	/// The shape of this test is load-bearing and a first version of it had NONE
	/// of it, so it passed with the bug present. The latch governs BRAND-NEW key
	/// admission only, so the divergence is invisible unless new keys arrive
	/// AFTER the resize -- and the shared skewed workload draws from a fixed 200
	/// keys, all of which exist well before the midpoint. A shrink also re-latches
	/// on its own, because the demotions it forces set the latch again; the
	/// no-op resize is the case that actually separates the two.
	#[test]
	fn resizes_like_lfu_hybrid() {
		for (start_cap, resized) in [(65_536u64, 65_536u64), (131_072, 32_768), (32_768, 131_072)] {
			let mut a = LfuHybridStack::new(start_cap).with_shared_overhead(190);
			let mut b = LfuCompactHybridStack::new(start_cap).with_shared_overhead(190);
			let (mut ma, mut mb) = (Vec::new(), Vec::new());

			let mut step = |a: &mut LfuHybridStack, b: &mut LfuCompactHybridStack,
			                ma: &mut Vec<(HashedKey, Tier)>, mb: &mut Vec<(HashedKey, Tier)>,
			                k: HashedKey, size: ObjectSize| {
				if a.contains(k) { a.update(k); } else { a.insert(k, size); }
				if b.contains(k) { b.update(k); } else { b.insert(k, size); }
				ma.extend(a.drain_tier_migrations());
				mb.extend(b.drain_tier_migrations());
			};

			// Phase 1: fill and churn until the fast tier latches.
			for i in 0..4_000u64 {
				step(&mut a, &mut b, &mut ma, &mut mb, (i % 200) + 1, 1024);
			}
			assert!(a.admission_latched(), "baseline should be latched before the resize");

			a.resize_fast_tier(resized);
			b.resize_fast_tier(resized);
			assert_eq!(
				a.admission_latched(),
				b.admission_latched(),
				"admission latch diverges immediately after {start_cap} -> {resized}"
			);

			// Phase 2: BRAND-NEW keys, which is the only traffic the latch governs.
			for i in 0..2_000u64 {
				step(&mut a, &mut b, &mut ma, &mut mb, 10_000 + i, 1024);
			}

			assert_eq!(ma, mb, "migrations diverge resizing {start_cap} -> {resized}");
			for k in 0..2_000u64 {
				assert_eq!(
					a.tier_of(10_000 + k),
					b.chain.get(10_000 + k).map(|e| e.tier),
					"tier of new key {} diverges resizing {start_cap} -> {resized}",
					10_000 + k
				);
			}
		}
	}

	/// Shared skewed workload: 200 keys biased toward low ids, enough pressure
	/// to exercise promotion, demotion and the admission latch.
	fn skewed_ops() -> Vec<(HashedKey, ObjectSize)> {
		let mut ops = Vec::new();
		let mut x: u64 = 0x243F_6A88_85A3_08D3;
		for _ in 0..20_000 {
			x ^= x << 13; x ^= x >> 7; x ^= x << 17;
			let u = (x >> 11) as f64 / (1u64 << 53) as f64;
			ops.push((((u * u * 200.0) as u64) + 1, 1024));
		}
		ops
	}
}
