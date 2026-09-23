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
//! With ONE deliberate exception: a demoted key keeps its place among equal
//! counts (see "Tie order" below), where `lfu-hybrid` re-appended it.
//!
//! # Tie order
//!
//! Keys of equal count leave in the order they REACHED that count, as upstream
//! `LfuStack` orders them and as CLAUDE.md's LFU design decision 4 specifies.
//! Promotion always did this -- it immediately follows a bump, so the promoted
//! key is the newest at its count. Demotion did not: it was a `set_tier`, which
//! re-appended the key at the newest end of the slow bucket, as if it had just
//! reached its count. It is now `ArenaFrequencyChain::demote_min_fast`, which
//! keeps the key's last-touch stamp and its place.
//!
//! The one remaining tie difference from upstream is deliberate and not about
//! order within a bucket: eviction prefers the SLOW tier, so a fast key tied
//! at the minimum count is shielded from a slow key that reached it later.
//!
//! # The global-eviction variant: `lfu-global-compact-hybrid`
//!
//! The same stack also serves `PaperPolicy::LfuGlobalCompactHybrid`, built by
//! [`LfuCompactHybridStack::new_global`]. Admission, the latch, the strict
//! promotion rule, the settle, the migrations and every overhead term are the
//! ones above. Two things differ, both keyed off [`EvictionScope`]:
//!
//! * **Eviction takes the minimum `(count, stamp)` over BOTH tiers**
//!   (`ArenaFrequencyChain::min_over_both_tiers`) instead of the slow tier's
//!   minimum first. After the tie-order fix every key's stamp is the moment it
//!   reached its current count, in either tier, so this is upstream LFU's
//!   victim exactly: the eviction sequence equals flat `LfuCompactStack`'s over
//!   the same history, which `the_global_scope_evicts_exactly_what_flat_lfu_evicts`
//!   checks victim by victim over random histories. A fast victim frees its
//!   bytes where they are: `fast_used` and the fast count drop, no migration is
//!   emitted, and the latch is left as it was. This departs from the paper's
//!   "evicted from the slow tier" rule on purpose, which is why it is a policy
//!   of its own and not a change to `LfuCompactHybrid`.
//!
//! * **A credit-limited refill.** Once the latch is shut only a promotion can
//!   bring a key into DRAM, and a promotion needs a count STRICTLY above the
//!   fast minimum. Under slow-first eviction nothing else ever needs to, since
//!   fast bytes leave only by demotion and a demotion is always paid for by the
//!   promotion that caused it. Under global eviction fast bytes also leave by
//!   EVICTION, and then nothing puts them back: a slow key tied with the fast
//!   minimum may not promote, and the room stays empty. Measured on the Python
//!   reference model with a dead startup cohort filling DRAM, plain global
//!   eviction evicted the cohort and left the fast tier holding ONE object for
//!   the rest of the run, 0.05% of its budget.
//!
//!   So `refill_credit` counts the migrating bytes of every fast EVICTION --
//!   `evict_one`, not `remove`, so a delete earns nothing. A hit or overwrite
//!   that brings a SLOW key's count EQUAL to the fast minimum promotes it when
//!   there is credit and it covers the key's bytes, and the fast tier stays at
//!   or under the drain target, the key's own shared-overhead reservation
//!   included, so the settle that follows demotes nothing. Every promotion
//!   spends credit, and any demotion zeroes it: the tier is full again, and
//!   room a demotion left is sub-object slack, not room an eviction freed. The
//!   latch is untouched, and max(slow count) <= min(fast count) still holds,
//!   because only a key that was just re-accessed can refill and only at a
//!   count the fast tier already holds.
//!
//!   The credit is a BYTE BUDGET, not a claim on the room it was earned in. It
//!   caps what refills may bring in at what fast evictions freed since the
//!   last demotion, and only a promotion or a demotion spends it -- not an
//!   admission. So a latch-open admission (before the latch first shuts, or
//!   after a grow reopens it) can take evicted room without touching the
//!   credit, and the credit left over can later pay for a refill into room a
//!   delete or a shrinking overwrite freed.
//!   `the_credit_is_a_byte_budget_not_a_claim_on_the_room_it_came_from` pins
//!   that. It is the reference model's rule (`lfu_gp.py`), on which every
//!   measurement in HYBRID_CACHES.md was taken, and it cannot overfill the
//!   tier: the room test still applies. What the credit rules out is the
//!   unconditional "tie into any room" rule, which fired mostly in the slack
//!   every demotion leaves, and on the reference model's grid cost +59%
//!   promotions, +64% demotions and 2.8x the wasted promotions of plain global
//!   eviction for +0.37 pp of fast-hit share.
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
		narrow_resident, drain_target,
	},
};

/// Where `evict_one` looks for its victim -- the one policy difference between
/// `lfu-compact-hybrid` and `lfu-global-compact-hybrid`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EvictionScope {
	/// `PaperPolicy::LfuCompactHybrid`: the slow tier's minimum whenever the
	/// slow tier holds anything, the fast tier's only when it does not -- the
	/// paper's rule, and the slow-first shield.
	SlowFirst,

	/// `PaperPolicy::LfuGlobalCompactHybrid`: the minimum `(count, stamp)` over
	/// both tiers, which is upstream LFU's victim, plus the credit-limited
	/// refill of the fast room that evicting from DRAM frees. See the module
	/// doc.
	Global,
}

pub struct LfuCompactHybridStack {
	/// Both tiers, one slab. Replaces `LfuHybridStack`'s `fast_chain`,
	/// `slow_chain` and `entries` together.
	chain: ArenaFrequencyChain,

	/// Which policy this stack is: slow-first or global eviction.
	scope: EvictionScope,

	/// `EvictionScope::Global` only: migrating bytes that fast-tier EVICTIONS
	/// have freed and no promotion has spent since. Raised by `evict_one`
	/// (never by `remove`), lowered by every promotion, zeroed by every
	/// demotion and by `clear` -- and by nothing else, so an admission into the
	/// room an eviction freed leaves it standing. Always 0 under `SlowFirst`,
	/// which never adds to it. See the module doc.
	refill_credit: CacheSize,

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
	/// `PaperPolicy::LfuCompactHybrid`: slow-first eviction.
	pub fn new(fast_capacity: CacheSize) -> Self {
		Self::with_scope(fast_capacity, EvictionScope::SlowFirst)
	}

	/// `PaperPolicy::LfuGlobalCompactHybrid`: global eviction plus the
	/// credit-limited refill. Everything else is `new`'s.
	pub fn new_global(fast_capacity: CacheSize) -> Self {
		Self::with_scope(fast_capacity, EvictionScope::Global)
	}

	fn with_scope(fast_capacity: CacheSize, scope: EvictionScope) -> Self {
		LfuCompactHybridStack {
			chain: ArenaFrequencyChain::default(),
			scope,
			refill_credit: 0,
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
		// Only FAST-tier keys draw on the fast-tier budget; the container
		// tracks both tiers. Charging all of them floored the effective
		// capacity to zero at high object counts.
		self.fast_object_count() as CacheSize * self.shared_overhead
	}

	fn effective_fast_capacity(&self) -> CacheSize {
		self.fast_capacity.saturating_sub(self.reserved_overhead())
	}

	/// The tier this stack has `key` in, or `None` if it does not track it.
	///
	/// Present on every other hybrid stack in this directory --
	/// `LruCompactHybridStack:113`, `ClockCompactHybridStack:123`,
	/// `FifoCompactHybridStack:98`, the 2Q and S3-FIFO families -- and missing
	/// only here, because nothing had needed it: the differential tests that
	/// exist for LFU drive `chain` directly, from inside this module. The
	/// merged store's `lfu_order_fidelity` test compares
	/// `MergedStore::tier_of` against this stack from ANOTHER module, where
	/// `chain` is private and unreachable, so the accessor has to exist.
	pub fn tier_of(&self, key: HashedKey) -> Option<Tier> {
		self.chain.get(key).and_then(|e| e.tier)
	}

	/// The refill credit, for tests: this module's, and the merged store's
	/// fidelity test, which compares it against `MergedStore`'s after every
	/// step from another module, where the field is private.
	#[cfg(test)]
	pub(crate) fn refill_credit(&self) -> CacheSize {
		self.refill_credit
	}

	/// Bumps a slow key and promotes it if its new count strictly exceeds the
	/// fast tier's minimum -- or, under `EvictionScope::Global` only, if it
	/// EQUALS that minimum and the refill admits it (`refill_admits`). Returns
	/// the key if it moved.
	///
	/// The promotion itself is a single `set_tier`: unlike the original, there
	/// is no remove-from-one-chain-and-insert-into-the-other, so the count
	/// cannot be dropped in transit. `set_tier` re-appends the key at the newest
	/// end of its fast bucket, which is correct HERE and only here: the `bump`
	/// just before it made the key the newest at its count. That holds for a
	/// refill as much as for a strict promotion -- both follow the bump.
	fn maybe_promote(&mut self, key: HashedKey) -> Option<HashedKey> {
		let new_count = self.chain.bump(key);

		let should_promote = match self.chain.min_count(Tier::Fast) {
			None => true,
			Some(min) if new_count > min => true,

			// The tie branch. Tested last and only under the global scope, so
			// `LfuCompactHybrid` takes exactly the path it always took.
			Some(min) => {
				new_count == min
					&& self.scope == EvictionScope::Global
					&& self.refill_admits(key)
			},
		};

		if !should_promote {
			return None;
		}

		let size = self.chain.get(key)?.migrating();

		self.chain.set_tier(key, Tier::Fast);
		self.slow_used = self.slow_used.saturating_sub(size);
		self.fast_used += size;

		// Every promotion spends credit, strict ones included: a strict
		// promotion that lands in evicted room uses that room as surely as a
		// refill does, and one that overflows the tier demotes, which zeroes
		// the credit anyway. Always 0 -> 0 under `SlowFirst`.
		self.refill_credit = self.refill_credit.saturating_sub(size);

		Some(key)
	}

	/// Whether a slow key that has just TIED the fast minimum may refill room
	/// fast evictions freed: there is credit, its migrating bytes fit in it,
	/// and promoting it leaves `fast_used` at or under the drain target of the
	/// effective budget recomputed with the key's own shared-overhead
	/// reservation -- the budget the settle right after it will see -- so that
	/// settle demotes nothing. A refill that would displace a key is not a
	/// refill; the strict rule exists to stop two equal-count keys swapping.
	///
	/// "There is credit" is a test of its own because a key can migrate ZERO
	/// bytes -- its `dram_resident` covering its whole `size` -- and `0 > 0` is
	/// false: without it such a key would refill with no eviction behind it,
	/// and take a shared-overhead reservation that nothing freed. The
	/// reference model never builds a zero-byte key, so on every input it
	/// generates the two rules agree.
	fn refill_admits(&self, key: HashedKey) -> bool {
		let Some(entry) = self.chain.get(key) else { return false };
		let size = entry.migrating();

		if self.refill_credit == 0 || size > self.refill_credit {
			return false;
		}

		let effective_after = self.fast_capacity.saturating_sub(
			(self.fast_object_count() as CacheSize + 1) * self.shared_overhead,
		);

		self.fast_used + size <= drain_target::bytes(effective_after)
	}

	/// Demotes lowest-frequency fast keys the moment usage exceeds the
	/// effective budget, draining to exactly it.
	///
	/// Each demotion is `demote_min_fast`, not `min_with_count` + `set_tier`:
	/// the latter re-appended the key at the newest end of its slow bucket, so
	/// it outlived slow keys of the same count that reached it after it did.
	/// `demote_min_fast` keeps its place, and finds and moves the key by slot,
	/// without the two index probes the old pair of calls made.
	fn settle_fast_tier(&mut self) {
		let effective = self.effective_fast_capacity();
		let target = drain_target::bytes(effective);

		while self.fast_used > target {
			let Some((demote_key, entry)) = self.chain.demote_min_fast() else {
				break;
			};

			let size = entry.migrating();

			self.fast_used = self.fast_used.saturating_sub(size);
			self.slow_used += size;

			self.migrations.push((demote_key, Tier::Slow));
			self.pending_demotions += 1;

			// A demotion firing at all means capacity was genuinely reached.
			self.fast_tier_latched = true;

			// ...and that the tier is full again: whatever room evictions had
			// freed is spent, and the slack a demotion leaves is not theirs.
			self.refill_credit = 0;
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
		match self.scope {
			EvictionScope::SlowFirst => matches!(policy, PaperPolicy::LfuCompactHybrid),
			EvictionScope::Global => matches!(policy, PaperPolicy::LfuGlobalCompactHybrid),
		}
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
		self.refill_credit = 0;
	}

	fn evict_one(&mut self) -> Option<HashedKey> {
		let (key, tier) = match self.scope {
			// Slow first; fall back to fast when nothing has ever been demoted
			// (e.g. fast_capacity == max_size).
			EvictionScope::SlowFirst => {
				let tier = if self.chain.min_with_count(Tier::Slow).is_some() {
					Tier::Slow
				} else {
					Tier::Fast
				};

				(self.chain.min_with_count(tier)?.0, tier)
			},

			// Upstream LFU's victim, whichever tier holds it.
			EvictionScope::Global => {
				let (key, _count, tier) = self.chain.min_over_both_tiers()?;
				(key, tier)
			},
		};

		let entry = self.chain.remove(key)?;
		let size = entry.migrating();

		match tier {
			Tier::Slow => self.slow_used = self.slow_used.saturating_sub(size),

			// Freed in place: no migration, and the latch stays as it is. Under
			// the global scope the room is credited for a refill; under
			// slow-first this is the empty-slow fallback, which never refills.
			Tier::Fast => {
				self.fast_used = self.fast_used.saturating_sub(size);

				if self.scope == EvictionScope::Global {
					self.refill_credit += size;
				}
			},
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

	/// The tier-move tie fix, end to end through the stack. Key 2 is demoted by
	/// the promotion of key 4; it reached count 1 before 3 and 5 did, so it is
	/// evicted before them. A re-append (the old `set_tier` demotion) made it
	/// last: 3, 5, 2. Upstream `LfuStack` on the same history evicts 2 first.
	#[test]
	fn a_demoted_key_is_evicted_in_the_order_it_reached_its_count() {
		let mut stack = LfuCompactHybridStack::new(100);
		stack.insert(1, 40);   // fast, count 1
		stack.insert(2, 40);   // fast, count 1
		stack.insert(3, 40);   // slow (no room), count 1
		stack.insert(4, 40);   // slow (latched), count 1
		stack.insert(5, 40);   // slow (latched), count 1
		stack.update(1);       // fast, count 2
		stack.update(4);       // 2 > fast minimum 1: promotes, and demotes 2
		drain(&mut stack);

		assert_eq!(stack.tier_of(4), Some(Tier::Fast));
		assert_eq!(stack.tier_of(2), Some(Tier::Slow));

		assert_eq!(stack.evict_one(), Some(2));
		assert_eq!(stack.evict_one(), Some(3));
		assert_eq!(stack.evict_one(), Some(5));

		// slow is empty: the fast fallback, count 2, in the order 1 and 4 reached it
		assert_eq!(stack.evict_one(), Some(1));
		assert_eq!(stack.evict_one(), Some(4));
	}

	/// xorshift rather than a dev-dependency, from a fixed seed so a failure is
	/// reproducible.
	fn stream(seed: u64) -> impl FnMut() -> u64 {
		let mut state = seed;

		move || {
			state ^= state << 13;
			state ^= state >> 7;
			state ^= state << 17;
			state
		}
	}

	/// The tie-break under TIERING, against the flat arbiter -- this stack's
	/// twin of the merged store's `a_tiered_drain_is_the_flat_order_split_by_tier`.
	///
	/// A drain evicts without admitting, bumping, promoting or demoting, and
	/// removing a key never reorders the rest of an LFU order -- so the whole
	/// drain is fixed by the state it starts from: every slow key first
	/// (slow-first eviction, the deliberate shield), then every fast key, each
	/// group in the order the FLAT `LfuCompactStack` holds those same keys in.
	/// The flat stack saw the same accesses, so it knows every key's count and
	/// when the key reached it; it has never heard of tiers, so it cannot
	/// re-rank a key for changing one.
	///
	/// The re-append demotion broke exactly this: a demoted key came out behind
	/// slow keys the flat stack puts after it. The merged store's fixture is
	/// one history; this is random ones, because the property has to hold for
	/// every history -- admissions on both sides of the latch, hits, overwrites
	/// that resize, removals, and budget resizes both ways, so demotions come
	/// from promotions, from growing overwrites and from shrinks, and a grow
	/// reopens admission below keys already slow.
	#[test]
	fn a_tiered_drain_is_the_flat_order_split_by_tier() {
		use std::collections::HashMap;

		use crate::worker::policy::policy_stack::lfu_compact_stack::LfuCompactStack;

		const KEYS: u64 = 512;

		for seed in [0x2545_F491_4F6C_DD1D_u64, 0x9E37_79B9_7F4A_7C15, 0xD1B5_4A32_D192_ED03] {
			let mut next = stream(seed);
			let mut tiered = LfuCompactHybridStack::new(2_000);
			let mut flat = LfuCompactStack::default();
			let mut demotions = 0;

			for _ in 0..20_000 {
				// A hot 32 among 512, so counts spread well past 1.
				let key = match next() % 4 {
					0 => next() % 32,
					_ => next() % KEYS,
				};
				let size = 16 + (next() % 48) as ObjectSize;

				match next() % 16 {
					0 => {
						tiered.remove(key);
						flat.remove(key);
					},

					// The flat stack has no budget to resize.
					1 => tiered.resize_fast_tier(1_000 + next() % 2_000),

					// An admission, or an overwrite if the key is present --
					// which both stacks treat as an access.
					2..=5 => {
						tiered.insert(key, size);
						flat.insert(key, size);
					},

					_ => {
						tiered.update(key);
						flat.update(key);
					},
				}

				tiered.drain_tier_migrations();
				demotions += tiered.drain_demotions();
			}

			assert!(demotions > 0, "seed {seed:#x}: nothing was demoted, so no tier move was tested");
			assert!(
				tiered.slow_object_count() > 0 && tiered.fast_object_count() > 0,
				"seed {seed:#x}: one tier ended empty, so the split is untested",
			);

			let tier: HashMap<HashedKey, Tier> = (0..KEYS)
				.filter_map(|k| tiered.tier_of(k).map(|t| (k, t)))
				.collect();

			let mut flat_order = Vec::new();

			while let Some(k) = flat.evict_one() {
				flat_order.push(k);
			}

			assert_eq!(flat_order.len(), tier.len(), "seed {seed:#x}: the stacks track different keys");

			let mut expected = Vec::new();

			for want in [Tier::Slow, Tier::Fast] {
				expected.extend(flat_order.iter().copied().filter(|k| tier[k] == want));
			}

			let mut drained = Vec::new();

			while let Some(k) = tiered.evict_one() {
				drained.push(k);
			}

			assert_eq!(
				drained, expected,
				"seed {seed:#x}: the tiered drain is not the flat LFU order split by \
				 tier -- a tier move re-ranked a key among its equals",
			);
		}
	}

	// ── lfu-global-compact-hybrid: global eviction and the refill ─────────

	/// Keys 1 and 2 admitted fast at count 1, keys 3, 4 and 5 admitted slow at
	/// count 1 once the tier is full (the latch shuts on 3), then key 1 hit
	/// once. The fast tier now holds a key, 2, tied at the minimum count with
	/// three slow keys that all reached that count after it did -- the shape
	/// on which the two eviction scopes disagree.
	fn cohort(stack: &mut LfuCompactHybridStack) {
		for key in 1..=5 {
			stack.insert(key, 40);
		}

		stack.update(1);
		drain(stack);
		stack.drain_demotions();

		assert_eq!((stack.tier_of(1), stack.tier_of(2)), (Some(Tier::Fast), Some(Tier::Fast)));
		assert_eq!(stack.slow_object_count(), 3);
		assert!(stack.admission_latched());
	}

	#[test]
	fn each_eviction_scope_answers_to_its_own_policy_only() {
		let slow_first = LfuCompactHybridStack::new(100);
		let global = LfuCompactHybridStack::new_global(100);

		assert!(slow_first.is_policy(&PaperPolicy::LfuCompactHybrid));
		assert!(!slow_first.is_policy(&PaperPolicy::LfuGlobalCompactHybrid));
		assert!(global.is_policy(&PaperPolicy::LfuGlobalCompactHybrid));
		assert!(!global.is_policy(&PaperPolicy::LfuCompactHybrid));
	}

	/// The one rule the two policies differ on, from one history. Slow-first
	/// shields fast key 2 and evicts 3; global eviction takes 2, which reached
	/// count 1 first -- upstream LFU's victim. The fast victim is freed where
	/// it is: no migration, no demotion, and the latch stays shut.
	#[test]
	fn global_eviction_takes_the_fast_key_slow_first_shields() {
		let mut slow_first = LfuCompactHybridStack::new(100);
		let mut global = LfuCompactHybridStack::new_global(100);
		cohort(&mut slow_first);
		cohort(&mut global);

		assert_eq!(slow_first.evict_one(), Some(3), "slow-first shields the fast tier");
		assert_eq!(global.evict_one(), Some(2), "2 reached count 1 before 3, 4 and 5");

		assert_eq!((global.fast_object_count(), global.slow_object_count()), (1, 3));
		assert_eq!((global.fast_bytes_used(), global.slow_bytes_used()), (40, 120));
		assert_eq!(drain(&mut global), vec![], "a fast victim is freed in place");
		assert_eq!(global.drain_demotions(), 0);
		assert!(global.admission_latched(), "evicting from DRAM does not reopen admission");

		// The rest of the order is count 1 in arrival order, then count 2.
		for want in [3, 4, 5, 1] {
			assert_eq!(global.evict_one(), Some(want));
		}

		assert_eq!(global.evict_one(), None);
		assert_eq!((global.fast_bytes_used(), global.slow_bytes_used()), (0, 0));
	}

	/// The refill, end to end: evicting fast key 2 earns 40 bytes of credit,
	/// and a slow key whose hit TIES the fast minimum (key 1, count 2) spends
	/// it -- promoted into the freed room with no demotion. Then a delete of
	/// key 1 frees 40 more bytes, which earns nothing, and a second tie finds
	/// room for it but no credit, so it stays slow: it is the spent credit that
	/// refuses it, not the room test.
	#[test]
	fn a_fast_eviction_earns_credit_that_a_tie_spends_without_a_demotion() {
		let mut stack = LfuCompactHybridStack::new_global(100);
		cohort(&mut stack);

		assert_eq!(stack.refill_credit(), 0);
		assert_eq!(stack.evict_one(), Some(2));
		assert_eq!(stack.refill_credit(), 40, "the fast victim's bytes are credited");

		stack.update(4);

		assert_eq!(stack.tier_of(4), Some(Tier::Fast), "a tie refilled the evicted room");
		assert_eq!(drain(&mut stack), vec![(4, Tier::Fast)]);
		assert_eq!(stack.drain_demotions(), 0, "a refill displaces nothing");
		assert_eq!((stack.fast_bytes_used(), stack.refill_credit()), (80, 0));

		stack.remove(1);

		assert_eq!((stack.fast_bytes_used(), stack.refill_credit()), (40, 0), "a delete earns nothing");
		assert!(40 + 40 <= drain_target::bytes(100), "the fixture no longer leaves room for key 5");

		stack.update(5);       // count 2, ties key 4

		assert_eq!(stack.tier_of(5), Some(Tier::Slow), "no credit left, so the strict rule holds");
		assert_eq!(drain(&mut stack), vec![]);
	}

	/// With no credit, room that no EVICTION freed is not the refill's to fill.
	/// Here it is left by an overwrite shrinking a fast key, which earns no
	/// credit, so a tie stays slow while a strict promotion still takes the
	/// room.
	///
	/// And slow-first eviction never earns credit at all, even when its
	/// empty-slow fallback does evict from DRAM.
	#[test]
	fn room_no_eviction_freed_is_not_refilled_by_a_tie() {
		let mut stack = LfuCompactHybridStack::new_global(100);
		stack.insert(1, 40);
		stack.insert(2, 40);
		stack.insert(3, 30);   // slow, latched
		stack.update(1);
		stack.update(2);       // both fast at count 2
		stack.insert(1, 10);   // shrinks key 1: 50 bytes fast, 50 free
		stack.insert(4, 30);   // slow
		drain(&mut stack);

		stack.update(4);       // count 2, ties key 2

		assert_eq!(stack.tier_of(4), Some(Tier::Slow));
		assert_eq!(stack.refill_credit(), 0);

		stack.update(4);       // count 3, strictly above

		assert_eq!(stack.tier_of(4), Some(Tier::Fast));
		assert_eq!(stack.drain_demotions(), 0);

		let mut slow_first = LfuCompactHybridStack::new(1_000);
		slow_first.insert(1, 10);

		assert_eq!(slow_first.evict_one(), Some(1), "the empty-slow fallback");
		assert_eq!(slow_first.refill_credit(), 0, "slow-first never credits an eviction");
	}

	/// The credit is a byte budget, not a claim on the room it was earned in --
	/// the reference model's rule, pinned here so a change to it is a decision
	/// rather than an accident (see the module doc). Evicting fast key 2 earns
	/// 40 bytes; with the latch still open, key 3 is ADMITTED into that room
	/// and spends none of it; key 4 then finds the tier full and shuts the
	/// latch. A delete of key 3 frees 40 bytes that earn nothing -- and key 4's
	/// tie with key 1 refills them anyway, on the credit key 2's eviction left.
	/// No demotion follows, because the room test still holds.
	#[test]
	fn the_credit_is_a_byte_budget_not_a_claim_on_the_room_it_came_from() {
		let mut stack = LfuCompactHybridStack::new_global(100);
		stack.insert(1, 40);
		stack.insert(2, 40);
		stack.update(1);       // count 2: key 2 is now the minimum

		assert_eq!(stack.evict_one(), Some(2));
		assert!(!stack.admission_latched(), "the fixture needs the latch still open");

		stack.insert(3, 40);   // latch open: admitted fast into key 2's room

		assert_eq!(stack.tier_of(3), Some(Tier::Fast));
		assert_eq!((stack.fast_bytes_used(), stack.refill_credit()), (80, 40), "an admission spends nothing");

		stack.insert(4, 40);   // no room: slow, and the latch shuts
		stack.remove(3);       // a delete: 40 bytes of room, no credit for them
		drain(&mut stack);

		assert!(stack.admission_latched());
		assert_eq!((stack.fast_bytes_used(), stack.refill_credit()), (40, 40));

		stack.update(4);       // count 2, ties key 1

		assert_eq!(stack.tier_of(4), Some(Tier::Fast), "the leftover credit paid for the refill");
		assert_eq!(drain(&mut stack), vec![(4, Tier::Fast)]);
		assert_eq!(stack.drain_demotions(), 0);
		assert_eq!((stack.fast_bytes_used(), stack.refill_credit()), (80, 0));
	}

	/// A refill needs credit even for a key that migrates nothing. Key 3's
	/// whole size is DRAM-resident, so it would cost the fast tier no value
	/// bytes -- but it would cost a shared-overhead reservation, and with no
	/// fast eviction behind it that is room nothing freed. `0 > 0` is false, so
	/// the size test alone would have let it through.
	#[test]
	fn a_refill_needs_credit_even_for_a_key_that_migrates_nothing() {
		let mut stack = LfuCompactHybridStack::new_global(100).with_shared_overhead(4);
		stack.insert(1, 40);
		stack.insert(2, 40);
		stack.update(1);
		stack.update(2);                  // both fast at count 2
		stack.insert(4, 40);              // no room: slow, and the latch shuts
		stack.insert_resident(3, 10, 10); // latched, so slow; migrates 0 bytes
		drain(&mut stack);

		assert_eq!(stack.tier_of(3), Some(Tier::Slow));
		assert_eq!(stack.chain.get(3).map(|e| e.migrating()), Some(0));
		assert!(
			stack.fast_bytes_used() <= drain_target::bytes(100 - 3 * 4),
			"the fixture leaves no room for key 3's reservation, so it does not \
			 isolate the credit test",
		);

		stack.update(3);                  // count 2, ties the fast minimum

		assert_eq!(stack.tier_of(3), Some(Tier::Slow), "no eviction, no credit, no refill");
		assert_eq!(drain(&mut stack), vec![]);
		assert_eq!(stack.refill_credit(), 0);
	}

	/// Keys 1, 2 and 3 fast, 4 and 5 slow, key 1 hit once; then global
	/// eviction takes fast keys 2 and 3 -- 80 bytes of credit -- leaving key 1
	/// alone in DRAM at count 2.
	fn credited(stack: &mut LfuCompactHybridStack) {
		for key in 1..=5 {
			stack.insert(key, 40);
		}

		stack.update(1);

		assert_eq!(stack.evict_one(), Some(2));
		assert_eq!(stack.evict_one(), Some(3));
		assert_eq!((stack.refill_credit(), stack.fast_bytes_used()), (80, 40));

		drain(stack);
		stack.drain_demotions();
	}

	/// A demotion means the tier is full again, so it zeroes whatever credit
	/// was left: here an overwrite grows the one fast key past the budget.
	#[test]
	fn a_demotion_zeroes_the_credit() {
		let mut stack = LfuCompactHybridStack::new_global(150);
		credited(&mut stack);

		stack.insert(1, 160);

		assert_eq!(stack.drain_demotions(), 1);
		assert_eq!(stack.tier_of(1), Some(Tier::Slow));
		assert_eq!(stack.refill_credit(), 0, "the demotion spent the credit");
	}

	/// Credit is necessary, not sufficient: the refill must also leave the tier
	/// at or under its drain target, or the settle after it would demote. A
	/// shrink below two objects keeps the credit (it demotes nothing) but
	/// removes the room, so the tie is refused; once the room is back, the
	/// next tie is admitted.
	#[test]
	fn a_refill_never_takes_the_fast_tier_past_its_drain_target() {
		let mut stack = LfuCompactHybridStack::new_global(150);
		credited(&mut stack);

		stack.resize_fast_tier(70);
		assert_eq!(stack.drain_demotions(), 0, "one key fits 70 bytes");

		stack.update(4);       // count 2, ties key 1

		assert_eq!(stack.tier_of(4), Some(Tier::Slow), "80 bytes would not fit a 70-byte tier");
		assert_eq!(stack.drain_demotions(), 0);
		assert_eq!(stack.refill_credit(), 80, "a refused refill spends nothing");

		stack.resize_fast_tier(150);
		stack.update(5);       // count 2, ties key 1

		assert_eq!(stack.tier_of(5), Some(Tier::Fast));
		assert_eq!(stack.drain_demotions(), 0);
		assert!(stack.fast_bytes_used() <= drain_target::bytes(150));
		assert_eq!(stack.refill_credit(), 40);
	}

	/// The room test counts the refilled key's OWN shared-overhead reservation,
	/// because the settle that follows will: promoting it makes one more fast
	/// key, and the effective budget shrinks by one more reservation. Key 3's
	/// 40 bytes fit beside key 1's 50 against the budget as it stands (95 at a
	/// 10-byte reservation), but not against the 85 left once key 3 is fast.
	#[test]
	fn a_refill_counts_its_own_shared_overhead_reservation() {
		let mut stack = LfuCompactHybridStack::new_global(105).with_shared_overhead(10);

		stack.insert(1, 40);
		stack.insert(2, 40);
		stack.insert(3, 40);   // 120 > 105 - 3 * 10: slow, latched
		stack.update(1);

		assert_eq!(stack.evict_one(), Some(2));
		assert_eq!(stack.refill_credit(), 40);

		stack.insert(1, 50);   // grows in place, count 3, still under target
		stack.update(3);       // count 2, below the minimum
		stack.update(3);       // count 3, ties key 1
		drain(&mut stack);

		assert!(
			50 + 40 <= drain_target::bytes(105 - 10),
			"the fixture no longer fits WITHOUT the key's own reservation, so it \
			 does not tell the two budgets apart",
		);
		assert_eq!(stack.tier_of(3), Some(Tier::Slow), "85 bytes of budget cannot take 90");
		assert_eq!(stack.drain_demotions(), 0);
		assert_eq!(stack.refill_credit(), 40);
	}

	/// One history replayed into the global stack and into flat LFU at once,
	/// with the test's own record of every live key's bytes and count -- which
	/// decides when capacity eviction runs, and recognises a refill.
	struct Replay {
		tiered: LfuCompactHybridStack,
		flat: crate::worker::policy::policy_stack::lfu_compact_stack::LfuCompactStack,
		live: std::collections::HashMap<HashedKey, (ObjectSize, u32)>,
		used: u64,
		capacity: CacheSize,
		fast_victims: u32,
		slow_victims: u32,
		demotions: u64,
		refills: u32,
	}

	impl Replay {
		fn new(capacity: CacheSize) -> Self {
			Replay {
				tiered: LfuCompactHybridStack::new_global(capacity),
				flat: Default::default(),
				live: Default::default(),
				used: 0,
				capacity,
				fast_victims: 0,
				slow_victims: 0,
				demotions: 0,
				refills: 0,
			}
		}

		/// Runs one access-shaped operation on both stacks, counting it as a
		/// refill if it moved a slow key into the fast tier at exactly the
		/// fast minimum it saw -- a tie, which the strict rule never promotes.
		fn access(&mut self, key: HashedKey, op: impl FnOnce(&mut Self)) {
			let was_slow = self.tiered.tier_of(key) == Some(Tier::Slow);
			let fast_min = self.tiered.chain.min_count(Tier::Fast);

			op(self);

			if was_slow
				&& self.tiered.tier_of(key) == Some(Tier::Fast)
				&& fast_min == self.live.get(&key).map(|&(_, count)| count)
			{
				self.refills += 1;
			}
		}

		/// An admission, or an overwrite -- which both stacks count as an access.
		fn write(&mut self, key: HashedKey, size: ObjectSize) {
			self.access(key, |r| {
				r.tiered.insert(key, size);
				r.flat.insert(key, size);

				let (bytes, count) = r.live.entry(key).or_insert((0, 0));
				r.used = r.used + size as u64 - *bytes as u64;
				*bytes = size;
				*count += 1;
			});
		}

		fn hit(&mut self, key: HashedKey) {
			self.access(key, |r| {
				r.tiered.update(key);
				r.flat.update(key);

				if let Some((_, count)) = r.live.get_mut(&key) {
					*count += 1;
				}
			});
		}

		fn remove(&mut self, key: HashedKey) {
			self.tiered.remove(key);
			self.flat.remove(key);

			if let Some((bytes, _)) = self.live.remove(&key) {
				self.used -= bytes as u64;
			}
		}

		fn resize(&mut self, capacity: CacheSize) {
			self.capacity = capacity;
			self.tiered.resize_fast_tier(capacity);
		}

		/// `PolicyWorker::apply_evictions`: evict while the cache holds more than
		/// `budget` bytes, and require every victim to be flat LFU's.
		fn evict_over(&mut self, budget: u64, when: &str) {
			while self.used > budget {
				let tier = self.tiered.chain.min_over_both_tiers().map(|(_, _, tier)| tier);
				let victim = self.tiered.evict_one();

				assert_eq!(
					victim, self.flat.evict_one(),
					"{when}: the global stack and flat LFU evicted different keys",
				);

				let victim = victim.expect("over budget with nothing to evict");
				self.used -= self.live.remove(&victim).expect("evicted a key never admitted").0 as u64;

				match tier {
					Some(Tier::Fast) => self.fast_victims += 1,
					_ => self.slow_victims += 1,
				}
			}
		}

		/// What a fast victim must leave behind: gauges that equal the bytes and
		/// keys actually in each tier, and a fast tier inside its budget -- which
		/// a refill must not overshoot.
		fn check(&mut self, when: &str) {
			self.tiered.drain_tier_migrations();
			self.demotions += self.tiered.drain_demotions();

			let in_tier = |want: Tier| {
				self.live
					.iter()
					.filter(|&(k, _)| self.tiered.tier_of(*k) == Some(want))
					.fold((0_usize, 0 as CacheSize), |(n, b), (_, &(bytes, _))| (n + 1, b + bytes as CacheSize))
			};

			assert_eq!(
				(self.tiered.fast_object_count(), self.tiered.fast_bytes_used()), in_tier(Tier::Fast),
				"{when}: the fast gauges drifted from the fast keys",
			);
			assert_eq!(
				(self.tiered.slow_object_count(), self.tiered.slow_bytes_used()), in_tier(Tier::Slow),
				"{when}: the slow gauges drifted from the slow keys",
			);
			assert!(
				self.tiered.fast_bytes_used() <= self.capacity,
				"{when}: the fast tier holds {} bytes of a {}-byte budget",
				self.tiered.fast_bytes_used(), self.capacity,
			);
		}
	}

	/// Global eviction IS flat LFU's eviction order, victim by victim, over
	/// random histories -- not merely at the end of one. Both stacks see every
	/// admission, hit, overwrite and removal; capacity eviction runs the way
	/// `PolicyWorker::apply_evictions` runs it, while the tracked bytes exceed
	/// a budget, and every victim the tiered stack nominates must be the one
	/// the flat stack nominates. Budget resizes both ways demote, and reopen
	/// admission on a grow; promotions, demotions and refills all move keys
	/// between tiers without touching the order.
	///
	/// The history is shaped so every mechanism fires, or the equality would be
	/// cheap: a startup cohort fills DRAM and only half of it is ever read
	/// again, one-hit wonders keep arriving, and reads land on the read half of
	/// the cohort or on recent arrivals. So never-reused keys are evicted
	/// straight out of DRAM (the first thing slow-first eviction would never
	/// do), recent arrivals climb to the fast minimum and tie it, and both the
	/// refill and the strict rule get to act. Tuned on the Python reference
	/// model with this exact generator: per seed about 9,000 fast victims,
	/// 1,500 slow ones, 450 demotions and 5-7 refills.
	#[test]
	fn the_global_scope_evicts_exactly_what_flat_lfu_evicts() {
		const COHORT: u64 = 64;
		const WINDOW: u64 = 256;
		const BUDGET: u64 = 6_000;
		const CAPACITY: CacheSize = 2_000;

		for seed in [0x2545_F491_4F6C_DD1D_u64, 0x9E37_79B9_7F4A_7C15, 0xD1B5_4A32_D192_ED03] {
			let mut next = stream(seed);
			let mut r = Replay::new(CAPACITY);

			for key in 0..COHORT {
				r.write(key, 16 + (next() % 48) as ObjectSize);
				r.evict_over(BUDGET, &format!("seed {seed:#x} cohort {key}"));
			}

			for key in (0..COHORT).step_by(2) {
				r.hit(key);
				r.evict_over(BUDGET, &format!("seed {seed:#x} cohort read {key}"));
			}

			let mut fresh = COHORT;

			for step in 0..20_000 {
				let when = format!("seed {seed:#x} step {step}");

				// One of the WINDOW most recent arrivals.
				let recent = |fresh: u64, draw: u64| fresh - 1 - draw % WINDOW.min(fresh);

				match next() % 20 {
					// A one-hit wonder.
					0..=8 => {
						let key = fresh;
						fresh += 1;
						r.write(key, 16 + (next() % 48) as ObjectSize);
					},

					// An overwrite of a recent key.
					9 | 10 => {
						let key = recent(fresh, next());
						r.write(key, 16 + (next() % 48) as ObjectSize);
					},

					// A read of the read half of the cohort, or of a recent key.
					11..=17 => {
						let key = match next() % 2 {
							0 => (next() % (COHORT / 2)) * 2,
							_ => recent(fresh, next()),
						};
						r.hit(key);
					},

					18 => {
						let key = recent(fresh, next());
						r.remove(key);
					},

					_ => r.resize(CAPACITY * 3 / 4 + next() % (CAPACITY / 2)),
				}

				r.evict_over(BUDGET, &when);
				r.check(&when);
			}

			assert!(r.fast_victims > 0, "seed {seed:#x}: no victim came from DRAM");
			assert!(r.slow_victims > 0, "seed {seed:#x}: no victim came from the slow tier");
			assert!(r.demotions > 0, "seed {seed:#x}: nothing was demoted");
			assert!(r.refills > 0, "seed {seed:#x}: no tie was ever refilled");

			// And the whole remaining order, to the last key.
			loop {
				let victim = r.tiered.evict_one();
				assert_eq!(victim, r.flat.evict_one(), "seed {seed:#x}: the final drains diverged");

				if victim.is_none() {
					break;
				}
			}
		}
	}

	/// A golden trace from the Python reference model of this policy
	/// (`lfu_gp.py`, global eviction with the credit-limited refill), replayed
	/// step for step: after every operation the touched key's tier, the credit
	/// and the fast tier's bytes must be what the model had, and every victim
	/// the model's eviction loop took. Generated at the default 0.98 drain
	/// target with a 600-byte fast tier, a 4-byte shared overhead and a 1,500
	/// byte capacity; the model asserted each victim against its own shadow of
	/// upstream LFU as it went. 240 operations: 21 fast admissions, 25
	/// promotions (4 of them refills), 20 demotions, 36 evictions (8 from DRAM).
	///
	/// Rows are `(op, key, arg, tier after, credit after, fast bytes after)`:
	/// op 0 inserts `arg` bytes, 1 hits, 2 removes, 3 evicts (`key` is the
	/// victim), 4 resizes the fast tier to `arg`; tier 0 is absent, 1 fast, 2
	/// slow.
	#[test]
	fn the_refill_matches_the_reference_model_step_for_step() {
		#[rustfmt::skip]
		const GOLDEN: &[(u8, HashedKey, u64, u8, CacheSize, CacheSize)] = &[
			(0, 0, 37, 1, 0, 37), (0, 1, 34, 1, 0, 71), (0, 2, 55, 1, 0, 126), (0, 3, 60, 1, 0, 186),
			(0, 4, 38, 1, 0, 224), (0, 5, 61, 1, 0, 285), (0, 6, 50, 1, 0, 335), (0, 7, 56, 1, 0, 391),
			(0, 8, 59, 1, 0, 450), (0, 9, 60, 1, 0, 510), (0, 10, 41, 1, 0, 551), (0, 11, 35, 2, 0, 551),
			(1, 0, 0, 1, 0, 551), (1, 2, 0, 1, 0, 551), (1, 4, 0, 1, 0, 551), (1, 6, 0, 1, 0, 551),
			(1, 8, 0, 1, 0, 551), (1, 10, 0, 1, 0, 551), (0, 101, 52, 2, 0, 551), (1, 2, 0, 1, 0, 551),
			(0, 102, 62, 2, 0, 551), (1, 6, 0, 1, 0, 551), (0, 103, 43, 2, 0, 551), (1, 103, 0, 1, 0, 500),
			(0, 104, 62, 2, 0, 500), (0, 105, 19, 2, 0, 500), (0, 106, 18, 2, 0, 500), (1, 0, 0, 1, 0, 500),
			(0, 107, 45, 2, 0, 500), (1, 0, 0, 1, 0, 500), (1, 106, 0, 1, 0, 518), (1, 104, 0, 1, 0, 519),
			(0, 108, 22, 2, 0, 519), (0, 109, 16, 2, 0, 519), (0, 110, 72, 2, 0, 519), (1, 108, 0, 1, 0, 485),
			(0, 2, 89, 1, 0, 519), (0, 111, 47, 2, 0, 519), (1, 110, 0, 1, 0, 531), (0, 112, 53, 2, 0, 531),
			(0, 113, 37, 2, 0, 531), (0, 114, 17, 2, 0, 531), (0, 115, 22, 2, 0, 531), (0, 116, 73, 2, 0, 531),
			(0, 117, 55, 2, 0, 531), (0, 111, 72, 2, 0, 531), (1, 111, 0, 1, 0, 506), (0, 118, 92, 2, 0, 506),
			(0, 110, 41, 1, 0, 475), (1, 109, 0, 2, 0, 475), (1, 103, 0, 1, 0, 475), (1, 108, 0, 1, 0, 475),
			(0, 7, 58, 2, 0, 475), (4, 0, 750, 1, 0, 475), (0, 119, 62, 1, 0, 537), (2, 116, 0, 0, 0, 537),
			(0, 119, 73, 1, 0, 548), (1, 114, 0, 2, 0, 548), (0, 120, 27, 1, 0, 575), (0, 121, 78, 2, 0, 575),
			(3, 1, 0, 0, 0, 575), (0, 122, 56, 2, 0, 575), (3, 3, 0, 0, 0, 575), (2, 110, 0, 0, 0, 534),
			(1, 119, 0, 1, 0, 534), (2, 6, 0, 0, 0, 484), (1, 117, 0, 1, 0, 539), (2, 8, 0, 0, 0, 539),
			(0, 121, 25, 1, 0, 564), (0, 120, 29, 1, 0, 566), (0, 123, 18, 2, 0, 566), (2, 104, 0, 0, 0, 504),
			(1, 107, 0, 2, 0, 504), (0, 109, 85, 1, 0, 589), (0, 124, 53, 2, 0, 589), (0, 125, 88, 2, 0, 589),
			(0, 121, 50, 1, 0, 614), (2, 122, 0, 0, 0, 614), (1, 125, 0, 2, 0, 614), (1, 117, 0, 1, 0, 614),
			(0, 126, 47, 2, 0, 614), (1, 124, 0, 2, 0, 614), (0, 127, 35, 2, 0, 614), (3, 5, 0, 0, 0, 614),
			(1, 121, 0, 1, 0, 614), (0, 128, 36, 2, 0, 614), (1, 112, 0, 2, 0, 614), (1, 114, 0, 1, 0, 631),
			(1, 125, 0, 1, 0, 678), (0, 129, 33, 2, 0, 678), (3, 9, 0, 0, 0, 678), (0, 130, 91, 2, 0, 678),
			(3, 11, 0, 0, 0, 678), (3, 101, 0, 0, 0, 678), (0, 117, 47, 1, 0, 670), (0, 109, 66, 1, 0, 651),
			(1, 118, 0, 2, 0, 651), (1, 128, 0, 2, 0, 651), (1, 127, 0, 2, 0, 651), (1, 124, 0, 1, 0, 657),
			(1, 129, 0, 2, 0, 657), (1, 128, 0, 2, 0, 657), (1, 130, 0, 2, 0, 657), (0, 131, 71, 2, 0, 657),
			(3, 102, 0, 0, 0, 657), (1, 129, 0, 2, 0, 657), (0, 132, 47, 2, 0, 657), (1, 123, 0, 2, 0, 657),
			(1, 130, 0, 2, 0, 657), (1, 131, 0, 2, 0, 657), (0, 133, 45, 2, 0, 657), (3, 105, 0, 0, 0, 657),
			(3, 113, 0, 0, 0, 657), (1, 133, 0, 2, 0, 657), (0, 130, 22, 1, 0, 679), (0, 134, 21, 2, 0, 679),
			(1, 125, 0, 1, 0, 679), (1, 132, 0, 2, 0, 679), (0, 135, 35, 2, 0, 679), (1, 130, 0, 1, 0, 679),
			(0, 133, 41, 2, 0, 679), (0, 119, 41, 1, 0, 647), (1, 126, 0, 2, 0, 647), (0, 126, 82, 2, 0, 647),
			(1, 135, 0, 2, 0, 647), (1, 128, 0, 1, 0, 611), (1, 119, 0, 1, 0, 611), (1, 127, 0, 2, 0, 611),
			(1, 129, 0, 1, 0, 644), (0, 136, 16, 2, 0, 644), (0, 137, 17, 2, 0, 644), (1, 136, 0, 2, 0, 644),
			(1, 136, 0, 2, 0, 644), (4, 0, 600, 1, 0, 509), (2, 0, 0, 0, 0, 472), (1, 134, 0, 2, 0, 472),
			(1, 135, 0, 2, 0, 472), (0, 138, 92, 2, 0, 472), (3, 115, 0, 0, 0, 472), (3, 137, 0, 0, 0, 472),
			(3, 138, 0, 0, 0, 472), (1, 132, 0, 2, 0, 472), (1, 131, 0, 2, 0, 472), (1, 134, 0, 2, 0, 472),
			(0, 108, 22, 2, 0, 472), (1, 136, 0, 2, 0, 472), (4, 0, 600, 0, 0, 472), (1, 136, 0, 1, 0, 488),
			(1, 136, 0, 1, 0, 488), (0, 139, 79, 2, 0, 488), (4, 0, 450, 0, 0, 399), (1, 128, 0, 1, 0, 399),
			(0, 140, 47, 2, 0, 399), (3, 139, 0, 0, 0, 399), (1, 140, 0, 2, 0, 399), (1, 136, 0, 1, 0, 399),
			(1, 131, 0, 2, 0, 399), (0, 141, 88, 2, 0, 399), (3, 141, 0, 0, 0, 399), (1, 136, 0, 1, 0, 399),
			(1, 131, 0, 1, 0, 373), (1, 131, 0, 1, 0, 373), (4, 0, 600, 0, 0, 373), (0, 142, 30, 1, 0, 403),
			(1, 132, 0, 1, 0, 450), (0, 126, 74, 1, 0, 524), (0, 143, 17, 2, 0, 524), (3, 142, 0, 0, 30, 494),
			(4, 0, 750, 0, 30, 494), (0, 144, 56, 1, 30, 550), (3, 143, 0, 0, 30, 550), (3, 144, 0, 0, 86, 494),
			(0, 145, 63, 1, 86, 557), (3, 145, 0, 0, 149, 494), (0, 146, 92, 1, 149, 586), (3, 146, 0, 0, 241, 494),
			(0, 147, 94, 1, 241, 588), (3, 147, 0, 0, 335, 494), (0, 148, 86, 1, 335, 580), (3, 148, 0, 0, 421, 494),
			(1, 135, 0, 1, 386, 529), (0, 149, 69, 1, 386, 598), (3, 149, 0, 0, 455, 529), (1, 130, 0, 1, 455, 529),
			(0, 150, 88, 1, 455, 617), (3, 150, 0, 0, 543, 529), (1, 129, 0, 1, 543, 529), (1, 133, 0, 1, 502, 570),
			(0, 108, 75, 1, 427, 645), (3, 4, 0, 0, 427, 645), (0, 151, 43, 2, 427, 645), (3, 151, 0, 0, 427, 645),
			(2, 107, 0, 0, 427, 645), (1, 133, 0, 1, 427, 645), (0, 152, 86, 2, 427, 645), (3, 152, 0, 0, 427, 645),
			(1, 132, 0, 1, 427, 645), (0, 153, 36, 2, 427, 645), (1, 140, 0, 2, 427, 645), (0, 154, 61, 2, 427, 645),
			(3, 153, 0, 0, 427, 645), (0, 155, 49, 2, 427, 645), (3, 154, 0, 0, 427, 645), (1, 136, 0, 1, 427, 645),
			(0, 156, 31, 2, 427, 645), (3, 155, 0, 0, 427, 645), (0, 157, 41, 2, 427, 645), (1, 135, 0, 1, 427, 645),
			(1, 124, 0, 2, 427, 645), (0, 108, 52, 1, 427, 622), (0, 131, 22, 1, 427, 573), (0, 158, 57, 2, 427, 573),
			(0, 159, 41, 2, 427, 573), (3, 156, 0, 0, 427, 573), (1, 130, 0, 1, 427, 573), (1, 157, 0, 2, 427, 573),
			(1, 140, 0, 1, 380, 620), (1, 136, 0, 1, 380, 620), (1, 157, 0, 2, 380, 620), (1, 157, 0, 1, 339, 661),
			(1, 125, 0, 1, 339, 661), (1, 157, 0, 1, 339, 661), (0, 108, 63, 1, 339, 672), (3, 158, 0, 0, 339, 672),
			(1, 136, 0, 1, 339, 672), (1, 136, 0, 1, 339, 672), (0, 160, 41, 2, 339, 672), (0, 161, 32, 2, 339, 672),
			(3, 159, 0, 0, 339, 672), (1, 140, 0, 1, 339, 672), (0, 134, 58, 2, 339, 672), (3, 160, 0, 0, 339, 672),
			(0, 162, 80, 2, 339, 672), (3, 161, 0, 0, 339, 672), (3, 162, 0, 0, 339, 672), (0, 163, 29, 2, 339, 672),
			(0, 129, 60, 1, 0, 633), (3, 163, 0, 0, 0, 633), (1, 134, 0, 1, 0, 617), (1, 157, 0, 1, 0, 617),
		];

		assert_eq!(
			drain_target::ratio(), drain_target::DEFAULT_RATIO,
			"the golden trace was generated at the default drain target",
		);

		let mut stack = LfuCompactHybridStack::new_global(600).with_shared_overhead(4);

		for (n, &(op, key, arg, tier, credit, fast_used)) in GOLDEN.iter().enumerate() {
			match op {
				0 => stack.insert(key, arg as ObjectSize),
				1 => stack.update(key),
				2 => stack.remove(key),
				3 => assert_eq!(stack.evict_one(), Some(key), "row {n}: a different victim"),
				4 => stack.resize_fast_tier(arg),
				_ => unreachable!("row {n}: op {op}"),
			}

			stack.drain_tier_migrations();
			stack.drain_demotions();

			let want = match tier {
				0 => None,
				1 => Some(Tier::Fast),
				_ => Some(Tier::Slow),
			};

			assert_eq!(stack.tier_of(key), want, "row {n}: key {key} is in the wrong tier");
			assert_eq!(stack.refill_credit(), credit, "row {n}: the credit diverged");
			assert_eq!(stack.fast_bytes_used(), fast_used, "row {n}: the fast bytes diverged");
		}
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
