/*
 * Copyright (c) Kia Shakiba
 *
 * This source code is licensed under the GNU AGPLv3 license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! `LruLfuCompactHybridStack` — `LruLfuHybridStack`'s policy over one slab.
//!
//! # What is the same
//!
//! The policy, deliberately and exactly. Admission lands at the fast tier's
//! recency head at frequency 1 and emits no migration. An access to a fast key
//! is a recency splice plus a carried counter bump and nothing else. An access
//! to a slow key bumps its counter and promotes it — resetting that counter to
//! 1 — the moment it *reaches* `promote_k`, which is an ABSOLUTE frequency and
//! not a count of accesses since demotion. `settle_fast_tier` demotes the LRU
//! tail from the high watermark down to the low one in a batch, each demoted
//! key entering the slow tier at the frequency it accumulated. `evict_one`
//! takes the slow tier's minimum-frequency key, ties broken
//! least-recently-touched, and falls back to the fast tier's LRU tail only when
//! nothing has ever been demoted. There is no admission latch, so
//! `resize_fast_tier` is a capacity write and a settle, in both directions.
//!
//! Any difference in results between this and `lru-lfu-hybrid` is therefore a
//! property of the representation, not of the algorithm — which is the entire
//! reason it exists as a separate variant rather than replacing the original.
//! `fidelity_tests` at the bottom of this file pins that down migration for
//! migration, tier for tier and eviction for eviction.
//!
//! # What is different
//!
//! `LruLfuHybridStack` keeps THREE key-indexed containers for a population
//! where every key is in exactly one place:
//!
//! ```text
//! fast_stack:  RecencyList     a kwik::HashList, owning its own key->node index
//! slow_chain:  FrequencyChain  a per-bucket HashList each owning an index,
//!                              plus the chain's own key->bucket index_map
//! entries:     EntryMap        the { tier, size, freq } payload, keyed again
//! ```
//!
//! The two orders are disjoint by construction — promotion is
//! `slow_chain.remove` then `fast_stack.push_front`, demotion is
//! `fast_stack.pop_back` then `slow_chain.insert_at`, and removal touches
//! exactly one of the two — which is precisely what lets both collapse into
//! one structure.
//!
//! This holds a single [`CompactFrequencyChain`]: one slab of 16-byte
//! link-only slots, one index whose VALUE carries the payload, the slow tier's
//! frequency buckets, and a distinguished recency list over the SAME slots for
//! the fast tier. A key occupies one slot whichever tier it is in; a demotion
//! is an unlink from the recency list and a link into a bucket, with the
//! frequency carried across by construction rather than passed by hand.
//!
//! # Why extend the LFU primitive rather than fork it
//!
//! `CompactFrequencyChain` already had everything the slow tier needs — slab,
//! free list, layout-B index, ordered bucket map, full `eviction_stacks_pmem`
//! gating. What it lacked was a list that is not a frequency bucket. Adding
//! `recency_head`/`recency_tail` and the `recency_*` operations makes the fast
//! tier one more list over the same storage and leaves every existing method
//! untouched: `LfuCompactHybridStack` calls none of the new methods and leaves
//! the two new fields `NIL` for the structure's whole life. One tested
//! primitive now serves both designs.
//!
//! # Why the 12-byte payload, and why the wider counter is safe
//!
//! `LruLfuEntry` is 8 bytes because its counter is a `u16` that packs into the
//! padding `{ size: u32, tier: u8 }` already had; widening it to `u32` would
//! have pushed the ORIGINAL design's `(HashedKey, LruLfuEntry)` pair from 16
//! bytes to 24, on every object in both tiers. That constraint does not carry
//! over. Here the payload sits in the index value beside a `u32` slot number,
//! where `CompactEntry`'s 12 bytes are what the tested primitive already
//! stores and what LFU already measured; reusing it unchanged keeps one
//! payload type for one primitive.
//!
//! The widening cannot change ranking. [`FREQUENCY_CAP`] is applied on every
//! bump, so a counter here takes values in `1..=16` and only those — the
//! identical set the `u16` takes, reached on the identical accesses.
//! `promote_k` is clamped into the same `1..=FREQUENCY_CAP` range.
//!
//! Saturation is where a widening would normally show, and it cannot show
//! here. The reason is worth stating because it is not obvious: `promote_k` is
//! at most the cap, so an access that would leave a slow key's count *pinned*
//! has by definition already met the threshold, and promotes rather than
//! reordering. Neither design ever reorders a slow key at an unchanged count.
//! [`CompactFrequencyChain::slow_relink_at`] relinks unconditionally anyway,
//! because that is `FrequencyChain::move_to`'s contract and the primitive is
//! shared — the case is unreachable from this stack, not merely unused.
//!
//! Like every stack here, this one tracks *order and tier membership only*.
//! It moves no bytes; `PolicyWorker` drains `drain_tier_migrations` and
//! performs the real `TieredBuffer` reallocation.
//!
//! **The baseline named above no longer exists in this crate.** Every
//! non-compact hybrid stack was removed once its compact twin was shown
//! behaviourally identical at 72 B/object of eviction stack instead of 112.
//! References to it here are historical: they say what this design is a
//! compaction OF, and they are the reason the structure looks the way it
//! does. Git history holds the baseline and the differential tests that
//! proved the two agreed.

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

/// Maximum value the per-object frequency counter saturates at. See the
/// module doc's "Why the counter is capped" section: this bounds both the
/// entry size (so the `(HashedKey, LruLfuEntry)` pair stays 16 bytes) and
/// `insert_at`'s linear scan (so the slow chain can never hold more than this
/// many buckets).
///
/// 16 is generous for the discrimination LFU actually needs -- S3-FIFO caps
/// its equivalent counter at 3 -- while leaving room for a `promote_k` well
/// above the useful range.
///
/// This used to be imported from `lru_lfu_hybrid_stack`, the non-compact
/// baseline, so the two designs could not drift apart. That baseline has been
/// removed from the crate, so the constant now lives here, which is the only
/// design that reads it.
pub(super) const FREQUENCY_CAP: u16 = 16;

/// [`FREQUENCY_CAP`] in the width [`CompactFrequencyChain`] stores counts at.
const CAP: u32 = FREQUENCY_CAP as u32;

pub struct LruLfuCompactHybridStack {
	/// Both tiers, one slab. Replaces `LruLfuHybridStack`'s `fast_stack`,
	/// `slow_chain` and `entries` together: the recency list is the fast tier,
	/// the frequency buckets are the slow tier, and the index value is the
	/// payload.
	chain: CompactFrequencyChain,

	/// Absolute frequency a slow-tier key must reach to earn the fast tier —
	/// *not* a count of accesses since it was demoted. Values below 3 do not
	/// filter a never-accessed key at all; see the baseline's module doc.
	promote_k: u32,

	fast_capacity: CacheSize,
	fast_used: CacheSize,
	slow_used: CacheSize,

	/// Per-object DRAM for the shared structures, reserved out of
	/// `fast_capacity` so the budget bounds total DRAM rather than fast-tier
	/// values alone. `0` unless set by `with_shared_overhead`.
	shared_overhead: CacheSize,

	/// (key, new tier) pairs recorded since the last `drain_tier_migrations`.
	migrations: Vec<(HashedKey, Tier)>,
}

impl LruLfuCompactHybridStack {
	/// `promote_k` is the absolute frequency a slow object must reach to be
	/// promoted. Clamped to at least 1 (0 would make every slow object
	/// instantly promotable before it was ever accessed) and at most
	/// [`FREQUENCY_CAP`] (a threshold above the cap could never be reached,
	/// silently disabling promotion entirely) — the baseline's clamp, applied
	/// to the same input.
	pub fn new(fast_capacity: CacheSize, promote_k: u16) -> Self {
		LruLfuCompactHybridStack {
			chain: CompactFrequencyChain::default(),

			promote_k: promote_k.clamp(1, FREQUENCY_CAP) as u32,

			fast_capacity,
			fast_used: 0,
			slow_used: 0,
			shared_overhead: 0,

			migrations: Vec::new(),
		}
	}

	/// Per-object DRAM reserved from the fast tier for shared metadata.
	///
	pub fn with_shared_overhead(mut self, overhead: CacheSize) -> Self {
		self.shared_overhead = overhead;


		self
	}

	/// The configured fast-tier byte budget.
	pub fn fast_capacity(&self) -> CacheSize {
		self.fast_capacity
	}

	/// The configured promotion threshold, after clamping.
	pub fn promote_k(&self) -> u16 {
		self.promote_k as u16
	}

	fn reserved_overhead(&self) -> CacheSize {
		self.chain.len() as CacheSize * self.shared_overhead
	}

	/// Returns the tier the given (currently tracked) key is in, or `None`
	/// if the key isn't tracked. Exposed for tests/diagnostics.
	pub fn tier_of(&self, key: HashedKey) -> Option<Tier> {
		self.chain.get(key).map(|entry| entry.tier)
	}

	/// Returns the key's current frequency counter, in the baseline's width.
	/// Exact, not a truncation: the counter is capped at [`FREQUENCY_CAP`].
	pub fn frequency_of(&self, key: HashedKey) -> Option<u16> {
		self.chain.get(key).map(|entry| entry.freq as u16)
	}

	/// Records a size change for an already-tracked key without altering its
	/// tier, adjusting whichever tier's used-bytes counter applies.
	/// `new_resident` refreshes the entry's DRAM-resident remainder: a re-set
	/// can add or drop a TTL, which changes it by the `Expiries` entry's cost.
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

	/// What a saturating counter bump would produce, without writing it.
	///
	/// `LruLfuHybridStack::bump_frequency` computes and stores in one step
	/// because its counter lives in a map it can hand out a `&mut` to. Here
	/// where the value is stored depends on the tier — a fast key's counter
	/// ranks nothing and is written on its own, a slow key's counter is a
	/// bucket key and has to be written by a relink — so the arithmetic is
	/// separated from the write. `0` for an untracked key, exactly as the
	/// baseline returns.
	fn next_frequency(&self, key: HashedKey) -> u32 {
		match self.chain.get(key) {
			Some(entry) => entry.freq.saturating_add(1).min(CAP),
			None => 0,
		}
	}

	/// Handles an access (read or overwrite) to a key already in the fast
	/// tier: pure recency, plus the carried counter bump. Never migrates.
	fn touch_fast(&mut self, key: HashedKey) {
		self.chain.recency_move_front(key);

		let freq = self.next_frequency(key);
		self.chain.set_freq(key, freq);
	}

	/// Handles an access to a key in the slow tier: bump its counter, keep
	/// the frequency buckets ordered, and promote if that crossed
	/// `promote_k`.
	fn touch_slow(&mut self, key: HashedKey) {
		let freq = self.next_frequency(key);

		if freq < self.promote_k {
			// Still earning its way in — reorder within the slow tier only.
			// `freq` has genuinely risen on this branch: `promote_k` is at
			// most the cap, so a count pinned by the cap would have met the
			// threshold and taken the other branch.
			self.chain.slow_relink_at(key, freq);
			return;
		}

		self.promote(key);
	}

	/// Moves a slow-tier key to the fast tier's recency head, resetting its
	/// counter (it spent that credit earning DRAM).
	fn promote(&mut self, key: HashedKey) {
		let Some(entry) = self.chain.get(key) else { return };
		let size = entry.migrating();

		// One call where the baseline needs four steps across three
		// containers: unlink from the slow bucket, set tier and counter, link
		// at the recency head. The entry never moves in the slab, so its size
		// and links survive by construction.
		if self.chain.promote_to_recency_front(key, 1).is_none() {
			return;
		}

		self.slow_used = self.slow_used.saturating_sub(size);
		self.fast_used += size;

		self.settle_fast_tier();

		// Pushed *after* `settle_fast_tier`, matching the baseline: an
		// extremely tight budget can demote this very key straight back out
		// in the same settle, in which case `settle_fast_tier` has already
		// pushed the correct final `(key, Tier::Slow)` entry and no `Fast`
		// entry should follow it.
		if self.chain.get(key).map(|e| e.tier) == Some(Tier::Fast) {
			self.migrations.push((key, Tier::Fast));
		}
	}

	/// Demotes the least-recently-used fast key(s) into the slow tier,
	/// *triggered* once `fast_used` exceeds [`watermarks::high_bytes`] of the
	/// effective value budget (`fast_capacity` minus the DRAM reserved for
	/// shared per-object metadata across both tiers) but *drained* all the way
	/// down to [`watermarks::low_bytes`] of it.
	///
	/// Each demoted key enters the slow tier at its accumulated frequency, not
	/// at 1 — that carry is the whole reason the fast tier counts. Demotion is
	/// the only response; the DRAM budget never evicts (terminal eviction
	/// stays governed solely by `max_size`).
	fn settle_fast_tier(&mut self) {
		let effective = self.fast_capacity.saturating_sub(self.reserved_overhead());

		if self.fast_used <= watermarks::high_bytes(effective) {
			return;
		}

		let drain_target = watermarks::low_bytes(effective);

		while self.fast_used > drain_target {
			// The baseline pops the recency tail, then looks the key up in
			// `entries` and `continue`s past it if it is missing -- a state it
			// documents as impossible. It is not merely impossible here but
			// unrepresentable: the recency list is threaded through the same
			// slots the index points at, so a listed key IS an indexed key.
			let Some((demote_key, entry)) = self.chain.demote_recency_back() else { break };
			let size = entry.migrating();

			self.fast_used = self.fast_used.saturating_sub(size);
			self.slow_used += size;

			self.migrations.push((demote_key, Tier::Slow));
		}
	}
}

impl PolicyStack for LruLfuCompactHybridStack {
	/// Payload-blind, exactly as `LruLfuHybridStack::is_policy` is: this
	/// design is the crate's one documented exception, matching on the variant
	/// and ignoring `promote_k`. Tightening it here alone would make a retune
	/// rebuild the stack for `lru-lfu-compact-hybrid` and not for
	/// `lru-lfu-hybrid`, which is a behavioural divergence in the one place a
	/// conversion is supposed to have none. See
	/// `is_policy_discriminates_on_the_payload_of_a_parameterised_policy`.
	fn is_policy(&self, policy: &PaperPolicy) -> bool {
		matches!(policy, PaperPolicy::LruLfuCompactHybrid(_))
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
			// An overwrite is an access, not an automatic promotion — see the
			// baseline's "A `set()` is an access" section.
			self.resize_key(key, size, dram_resident);
			self.update(key);
			return;
		}

		self.chain.recency_push_front(key, size, dram_resident, 1);
		self.fast_used += (size as CacheSize).saturating_sub(dram_resident as CacheSize);

		self.settle_fast_tier();
	}

	fn update(&mut self, key: HashedKey) {
		match self.chain.get(key).map(|entry| entry.tier) {
			Some(Tier::Fast) => self.touch_fast(key),
			Some(Tier::Slow) => self.touch_slow(key),
			None => {},
		}
	}

	fn remove(&mut self, key: HashedKey) {
		let Some(tier) = self.chain.get(key).map(|entry| entry.tier) else { return };

		// Which door the key leaves by is the tier: the recency list for a
		// fast key, its frequency bucket for a slow one. Exactly the
		// baseline's two-armed `match`, over one structure instead of two.
		let entry = match tier {
			Tier::Fast => self.chain.recency_remove(key),
			Tier::Slow => self.chain.remove(key),
		};

		let Some(entry) = entry else { return };
		let size = entry.migrating();

		match tier {
			Tier::Fast => self.fast_used = self.fast_used.saturating_sub(size),
			Tier::Slow => self.slow_used = self.slow_used.saturating_sub(size),
		}
	}

	fn clear(&mut self) {
		self.chain.clear();

		self.fast_used = 0;
		self.slow_used = 0;
		self.migrations.clear();
	}

	fn evict_one(&mut self) -> Option<HashedKey> {
		// Slow tier first, by minimum frequency (ties broken
		// least-recently-touched). The fast-tier fallback only applies when
		// nothing has ever been demoted — the same last-resort path every
		// hybrid stack here keeps.
		let (key, tier) = match self.chain.min_key(Tier::Slow) {
			Some(key) => (key, Tier::Slow),
			None => (self.chain.recency_back()?, Tier::Fast),
		};

		let entry = match tier {
			Tier::Slow => self.chain.remove(key),
			Tier::Fast => self.chain.recency_remove(key),
		};

		if let Some(entry) = entry {
			let size = entry.migrating();

			match tier {
				Tier::Fast => self.fast_used = self.fast_used.saturating_sub(size),
				Tier::Slow => self.slow_used = self.slow_used.saturating_sub(size),
			}
		}

		Some(key)
	}

	fn resize_fast_tier(&mut self, size: CacheSize) {
		// Byte-for-byte `LruLfuHybridStack::resize_fast_tier`. There is no
		// admission latch in this design -- admission is unconditionally to
		// the fast tier, so there is nothing that could be shut and nothing to
		// reopen -- and so none of the grow/shrink asymmetry the latched
		// stacks have to get right here.
		self.fast_capacity = size;
		self.settle_fast_tier();
	}

	fn drain_tier_migrations(&mut self) -> Vec<(HashedKey, Tier)> {
		std::mem::take(&mut self.migrations)
	}

	// `drain_demotions`, `admission_latched` and `inline_demotion_accounting`
	// are deliberately NOT overridden: `LruLfuHybridStack` overrides none of
	// them either, so the trait defaults (0 / false / true) are what the
	// baseline reports and therefore what this must report. Only the LFU pair
	// override `inline_demotion_accounting`, and only because their admission
	// corrects a brand-new key to slow without displacing anything; nothing
	// here ever does that.

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
}

#[cfg(test)]
mod tests {
	use super::*;

	const K: u16 = 2;

	fn drain(stack: &mut LruLfuCompactHybridStack) -> Vec<(HashedKey, Tier)> {
		stack.drain_tier_migrations()
	}

	/// A fast-tier capacity that leaves `target` bytes sitting comfortably
	/// *above* `settle_fast_tier`'s low-water drain floor. Same helper, and
	/// same reason, as the baseline's: sizing a test's capacity to exactly
	/// what should survive cascades an extra demotion the test never meant to
	/// exercise, because a pass triggers at the high watermark and drains to
	/// the low one.
	fn low_water_safe(target: CacheSize) -> CacheSize {
		(target as f64 / watermarks::low()).ceil() as CacheSize + 1
	}

	// ── admission ─────────────────────────────────────────────────────────

	#[test]
	fn admission_always_lands_fast_at_frequency_one() {
		let mut stack = LruLfuCompactHybridStack::new(1_000, K);

		stack.insert(1, 10);

		assert_eq!(stack.tier_of(1), Some(Tier::Fast));
		assert_eq!(stack.frequency_of(1), Some(1));
		assert!(drain(&mut stack).is_empty(), "admission needs no migration");
	}

	// ── demotion ──────────────────────────────────────────────────────────

	#[test]
	fn fast_pressure_demotes_the_lru_tail() {
		let mut stack = LruLfuCompactHybridStack::new(low_water_safe(100), K);

		stack.insert(1, 50);
		stack.insert(2, 50);
		drain(&mut stack);

		stack.insert(3, 50);

		assert_eq!(stack.tier_of(1), Some(Tier::Slow));
		assert_eq!(stack.tier_of(3), Some(Tier::Fast));

		let migrations = drain(&mut stack);
		assert!(migrations.contains(&(1, Tier::Slow)), "got {migrations:?}");
	}

	#[test]
	fn demotion_carries_the_accumulated_frequency() {
		let mut stack = LruLfuCompactHybridStack::new(100, 99);

		stack.insert(1, 50);
		stack.update(1);
		stack.update(1);
		assert_eq!(stack.frequency_of(1), Some(3));

		stack.insert(2, 50);
		stack.insert(3, 50);
		drain(&mut stack);

		assert_eq!(stack.tier_of(1), Some(Tier::Slow));
		assert_eq!(stack.frequency_of(1), Some(3));
	}

	#[test]
	fn a_demoted_object_outranks_a_one_hit_wonder_in_the_slow_tier() {
		let mut stack = LruLfuCompactHybridStack::new(100, 99);

		stack.insert(1, 50);
		stack.update(1);
		stack.update(1); // freq 3 — genuinely hot while fast

		stack.insert(2, 50); // freq 1 — one-hit wonder
		stack.insert(3, 50);
		stack.insert(4, 50);
		drain(&mut stack);

		assert_eq!(stack.tier_of(1), Some(Tier::Slow));
		assert_eq!(stack.tier_of(2), Some(Tier::Slow));

		assert_eq!(stack.evict_one(), Some(2));
	}

	// ── promotion ─────────────────────────────────────────────────────────

	#[test]
	fn a_slow_access_below_the_absolute_threshold_does_not_promote() {
		// k = 2 is NOT a filter for a never-accessed key: it demotes at
		// frequency 1, so a single access already reaches the threshold. The
		// documented off-by-one that makes 3 the first meaningful value.
		let mut lenient = LruLfuCompactHybridStack::new(100, 2);

		lenient.insert(1, 50);
		lenient.insert(2, 50);
		lenient.insert(3, 50);
		drain(&mut lenient);
		assert_eq!(lenient.tier_of(1), Some(Tier::Slow));

		lenient.update(1);
		assert_eq!(
			lenient.tier_of(1),
			Some(Tier::Fast),
			"at k = 2 a single slow access suffices — same as lru_hybrid_cache",
		);

		let mut strict = LruLfuCompactHybridStack::new(100, 3);
		strict.insert(1, 50);
		strict.insert(2, 50);
		strict.insert(3, 50);
		drain(&mut strict);
		assert_eq!(strict.tier_of(1), Some(Tier::Slow));

		strict.update(1); // freq 1 -> 2, still below k=3
		assert_eq!(strict.tier_of(1), Some(Tier::Slow));
		assert!(drain(&mut strict).is_empty(), "no migration below threshold");

		strict.update(1); // freq 2 -> 3, meets k
		assert_eq!(strict.tier_of(1), Some(Tier::Fast));
	}

	#[test]
	fn crossing_the_threshold_promotes_and_resets_the_counter() {
		let mut stack = LruLfuCompactHybridStack::new(100, 2);

		stack.insert(1, 50);
		stack.insert(2, 50);
		stack.insert(3, 50);
		drain(&mut stack);
		assert_eq!(stack.tier_of(1), Some(Tier::Slow));

		stack.update(1);

		assert_eq!(stack.tier_of(1), Some(Tier::Fast));
		assert_eq!(stack.frequency_of(1), Some(1), "counter resets on promotion");

		let migrations = drain(&mut stack);
		assert!(migrations.contains(&(1, Tier::Fast)), "got {migrations:?}");
	}

	#[test]
	fn promotion_can_cascade_a_demotion() {
		let mut stack = LruLfuCompactHybridStack::new(low_water_safe(100), 2);

		stack.insert(1, 50);
		stack.insert(2, 50);
		stack.insert(3, 50);
		drain(&mut stack);

		assert_eq!(stack.tier_of(1), Some(Tier::Slow));
		assert_eq!(stack.tier_of(2), Some(Tier::Fast));
		assert_eq!(stack.tier_of(3), Some(Tier::Fast));

		stack.update(1);

		let migrations = drain(&mut stack);
		assert!(migrations.contains(&(1, Tier::Fast)), "got {migrations:?}");
		assert!(
			migrations.iter().any(|(_, tier)| *tier == Tier::Slow),
			"promotion should have cascaded a demotion; got {migrations:?}",
		);
	}

	#[test]
	fn an_overwrite_goes_through_the_same_frequency_gate_as_a_read() {
		let mut stack = LruLfuCompactHybridStack::new(100, 3);

		stack.insert(1, 50);
		stack.insert(2, 50);
		stack.insert(3, 50);
		drain(&mut stack);
		assert_eq!(stack.tier_of(1), Some(Tier::Slow));

		stack.insert(1, 50); // an overwrite, freq 1 -> 2, still below k=3

		assert_eq!(stack.tier_of(1), Some(Tier::Slow), "a set must not bypass the gate");
		assert_eq!(stack.frequency_of(1), Some(2));

		stack.insert(1, 50); // freq 2 -> 3, meets k
		assert_eq!(stack.tier_of(1), Some(Tier::Fast));
	}

	// ── eviction ──────────────────────────────────────────────────────────

	#[test]
	fn evict_one_prefers_the_slow_minimum_frequency() {
		let mut stack = LruLfuCompactHybridStack::new(100, 99);

		stack.insert(1, 50);
		stack.insert(2, 50);
		stack.insert(3, 50); // demotes 1
		stack.insert(4, 50); // demotes 2
		drain(&mut stack);

		// Both slow at freq 1; ties break least-recently-touched, so the
		// earlier-demoted key goes first.
		assert_eq!(stack.evict_one(), Some(1));
		assert_eq!(stack.evict_one(), Some(2));
	}

	#[test]
	fn evict_one_falls_back_to_the_fast_tail_when_slow_is_empty() {
		let mut stack = LruLfuCompactHybridStack::new(1_000, K);

		stack.insert(1, 10);
		stack.insert(2, 10);
		drain(&mut stack);

		assert_eq!(stack.slow_object_count(), 0);
		assert_eq!(stack.evict_one(), Some(1), "LRU tail of the fast tier");
	}

	#[test]
	fn evict_one_on_an_empty_stack_is_none() {
		let mut stack = LruLfuCompactHybridStack::new(1_000, K);
		assert_eq!(stack.evict_one(), None);
	}

	// ── bookkeeping ───────────────────────────────────────────────────────

	#[test]
	fn remove_updates_the_right_tier_counters() {
		let mut stack = LruLfuCompactHybridStack::new(low_water_safe(100), 99);

		stack.insert(1, 50);
		stack.insert(2, 50);
		stack.insert(3, 50); // demotes exactly 1
		drain(&mut stack);

		assert_eq!(stack.slow_bytes_used(), 50);
		stack.remove(1);
		assert_eq!(stack.slow_bytes_used(), 0);
		assert_eq!(stack.slow_object_count(), 0);

		let fast_before = stack.fast_bytes_used();
		stack.remove(2);
		assert_eq!(stack.fast_bytes_used(), fast_before - 50);
		assert!(!stack.contains(2));
	}

	/// Ported from the baseline, where it reads "every tracked key is in
	/// exactly one structure". Over the slab that becomes a stronger claim:
	/// the two counts must partition `len()`, AND `len()` is the size of the
	/// one index, so a key double-linked into both the recency list and a
	/// frequency bucket would be counted twice here and once there.
	#[test]
	fn gauges_track_one_slab_and_one_index() {
		let mut stack = LruLfuCompactHybridStack::new(100, 99);

		stack.insert(1, 50);
		stack.insert(2, 50);
		stack.insert(3, 50);
		drain(&mut stack);

		assert_eq!(stack.len(), 3);
		assert_eq!(
			stack.fast_object_count() + stack.slow_object_count(),
			3,
			"every tracked key is in exactly one structure",
		);
		assert_eq!(stack.fast_bytes_used() + stack.slow_bytes_used(), 150);
	}

	#[test]
	fn clear_resets_everything() {
		let mut stack = LruLfuCompactHybridStack::new(100, K);

		stack.insert(1, 50);
		stack.insert(2, 50);
		stack.insert(3, 50);

		stack.clear();

		assert_eq!(stack.len(), 0);
		assert_eq!(stack.fast_bytes_used(), 0);
		assert_eq!(stack.slow_bytes_used(), 0);
		assert_eq!(stack.fast_object_count(), 0);
		assert_eq!(stack.slow_object_count(), 0);
		assert!(drain(&mut stack).is_empty());
	}

	#[test]
	fn resize_fast_tier_shrink_demotes_and_grow_creates_headroom() {
		let mut stack = LruLfuCompactHybridStack::new(1_000, 99);

		stack.insert(1, 50);
		stack.insert(2, 50);
		drain(&mut stack);
		assert_eq!(stack.slow_object_count(), 0);

		stack.resize_fast_tier(50);
		let migrations = drain(&mut stack);
		assert!(!migrations.is_empty(), "shrinking should demote");
		assert!(stack.slow_object_count() > 0);

		stack.resize_fast_tier(1_000);
		assert!(drain(&mut stack).is_empty(), "growing demotes nothing");
	}

	#[test]
	fn frequency_saturates_at_the_cap() {
		let mut stack = LruLfuCompactHybridStack::new(1_000, 99);

		stack.insert(1, 10);
		for _ in 0..(FREQUENCY_CAP as usize * 3) {
			stack.update(1);
		}

		assert_eq!(stack.frequency_of(1), Some(FREQUENCY_CAP));
	}

	/// The widening the module doc argues is safe, exercised at the cap.
	///
	/// A key that was genuinely hot before it demoted arrives in the slow tier
	/// at the cap and promotes on its very next access whatever `promote_k`
	/// is. That is the baseline's stated intent, and it is also why the
	/// "relink at an unchanged count" case is unreachable from this stack:
	/// `promote_k` is clamped to at most the cap, so an access that would
	/// leave the count pinned has already met the threshold. Run at the
	/// strictest threshold there is, where promotion is hardest to earn.
	#[test]
	fn a_saturated_key_stops_counting_and_promotes_on_its_next_access() {
		let mut stack = LruLfuCompactHybridStack::new(low_water_safe(100), FREQUENCY_CAP);

		stack.insert(1, 50);
		for _ in 0..(FREQUENCY_CAP as usize * 2) { stack.update(1); }
		assert_eq!(stack.frequency_of(1), Some(FREQUENCY_CAP), "the count is pinned");

		stack.insert(2, 50);
		stack.insert(3, 50); // pressure demotes the LRU tail, which is 1
		drain(&mut stack);

		assert_eq!(stack.tier_of(1), Some(Tier::Slow));
		assert_eq!(
			stack.frequency_of(1), Some(FREQUENCY_CAP),
			"the cap survived the demotion",
		);

		stack.update(1);

		assert_eq!(
			stack.tier_of(1), Some(Tier::Fast),
			"a saturated slow key meets any threshold on its next access",
		);
		assert_eq!(stack.frequency_of(1), Some(1), "and spends the credit");
	}

	#[test]
	fn promote_k_is_clamped_into_range() {
		assert_eq!(LruLfuCompactHybridStack::new(100, 0).promote_k(), 1);
		assert_eq!(LruLfuCompactHybridStack::new(100, 5).promote_k(), 5);
		assert_eq!(
			LruLfuCompactHybridStack::new(100, FREQUENCY_CAP + 10).promote_k(),
			FREQUENCY_CAP,
			"a threshold above the cap would disable promotion entirely",
		);
	}

	#[test]
	fn shared_overhead_reserves_dram_and_demotes_earlier() {
		let mut plain = LruLfuCompactHybridStack::new(100, 99);
		plain.insert(1, 40);
		plain.insert(2, 40);
		drain(&mut plain);
		assert_eq!(plain.tier_of(1), Some(Tier::Fast));
		assert_eq!(plain.tier_of(2), Some(Tier::Fast));

		let mut reserved = LruLfuCompactHybridStack::new(100, 99).with_shared_overhead(30);
		reserved.insert(1, 40);
		reserved.insert(2, 40);
		drain(&mut reserved);
		assert!(
			reserved.slow_object_count() > 0,
			"the shared-metadata reservation should have forced a demotion",
		);
	}

	// ── watermarks ────────────────────────────────────────────────────────

	const WM_CAPACITY: CacheSize = 10_000;
	const WM_UNIT: ObjectSize = 10;

	#[test]
	fn a_triggered_pass_drains_to_the_low_watermark() {
		let mut stack = LruLfuCompactHybridStack::new(WM_CAPACITY, K);

		let high = watermarks::high_bytes(WM_CAPACITY);
		let low = watermarks::low_bytes(WM_CAPACITY);
		let unit = WM_UNIT as CacheSize;

		let mut key: HashedKey = 0;

		while stack.fast_bytes_used() + unit <= high {
			key += 1;
			stack.insert(key, WM_UNIT);
		}

		assert_eq!(stack.slow_object_count(), 0, "nothing has crossed the trigger yet");
		assert!(drain(&mut stack).is_empty());

		let before = stack.fast_bytes_used();

		key += 1;
		stack.insert(key, WM_UNIT); // crosses the high watermark

		assert!(
			stack.fast_bytes_used() <= low,
			"a triggered pass must drain to the low watermark ({low}), not merely back to the ceiling ({high}); fast_used = {}",
			stack.fast_bytes_used(),
		);
		assert!(
			stack.fast_bytes_used() + unit > low,
			"and must stop the moment it reaches it rather than over-draining",
		);

		let expected = (before + unit - low).div_ceil(unit);

		let migrations = drain(&mut stack);
		let demoted = migrations
			.iter()
			.filter(|(_, tier)| *tier == Tier::Slow)
			.count() as CacheSize;

		assert_eq!(demoted, expected, "got {} migrations", migrations.len());
		assert_eq!(stack.slow_object_count() as CacheSize, expected);
	}

	/// The baseline's second "every tracked key is in exactly one structure"
	/// assertion, ported. Several watermark passes run, so the invariant is
	/// checked after real churn rather than after three inserts.
	#[test]
	fn counters_stay_consistent_across_a_watermark_pass() {
		let count: HashedKey = WM_CAPACITY / WM_UNIT as CacheSize + 200;

		let mut stack = LruLfuCompactHybridStack::new(WM_CAPACITY, K);

		for key in 1..=count {
			stack.insert(key, WM_UNIT);
		}
		drain(&mut stack);

		let unit = WM_UNIT as CacheSize;

		assert!(stack.slow_object_count() > 0, "the run must have triggered a pass");

		assert_eq!(stack.len(), count as usize);
		assert_eq!(
			stack.fast_object_count() + stack.slow_object_count(),
			count as usize,
			"every tracked key is in exactly one structure",
		);
		assert_eq!(
			stack.fast_bytes_used() + stack.slow_bytes_used(),
			count * unit,
			"no bytes lost or double-counted across the demotions",
		);
		assert_eq!(
			stack.fast_bytes_used(),
			stack.fast_object_count() as CacheSize * unit,
			"fast bytes must equal the fast object count times the unit size",
		);
		assert_eq!(
			stack.slow_bytes_used(),
			stack.slow_object_count() as CacheSize * unit,
			"slow bytes must equal the slow object count times the unit size",
		);
		assert!(
			stack.fast_bytes_used() <= watermarks::high_bytes(WM_CAPACITY),
			"the tier must be settled below the trigger after the last insert",
		);
	}
}
