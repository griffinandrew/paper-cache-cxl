//! `LfuCompactStack` — `LfuStack`'s policy over the slab design.
//!
//! The `HashList`-based `LfuStack` this re-lays-out was removed in R2 with the other
//! original flat stacks, so the mentions of it below are historical. Its eviction
//! orders for the op sequences in `fidelity_tests` are recorded in `golden.rs` and
//! asserted there.
//!
//! The non-tiered counterpart to `LfuCompactHybridStack`, and the fourth cell
//! of the layout/tiering matrix:
//!
//! ```text
//!                    multi-map layout       slab layout
//!   no tiering       LfuStack               LfuCompactStack   <- this
//!   tiered           (removed)              LfuCompactHybridStack
//! ```
//!
//! The multi-map-tiered cell is empty because `LfuHybridStack` has been
//! removed from the crate: it was behaviourally identical to
//! `LfuCompactHybridStack` and cost 112 B/object of eviction stack against
//! its 72 then (40 since the arena conversion). The layout comparison it anchored lives in git history.
//!
//! Without it, comparing all-DRAM LFU against a tiered compact LFU moves two
//! variables at once — which is how cluster13 produced an all-DRAM LFU that
//! was SLOWER on GET (1045 ns) than either tiered variant (845 / 793 ns)
//! with no way to attribute it.
//!
//! Deliberately NOT `CompactFrequencyChain`: that carries two bucket maps
//! (fast and slow) and a 12-byte `CompactEntry` with `tier` and
//! `dram_resident`, none of which a non-tiered design has any use for. This
//! keeps one bucket map and a bare `u32` frequency, so the index value is
//! `(u32 slot, u32 freq)` — 8 bytes against `LfuStack`'s separate
//! `index_map` + `VecList<CountStack>` + a `HashList` per bucket, each with
//! its own key-to-node index.
//!
//! `LfuStack` ignores the size argument, so this does too: byte accounting
//! belongs to the cache.

#[cfg(not(feature = "eviction_stacks_pmem"))]
use std::collections::HashMap;
#[cfg(feature = "eviction_stacks_pmem")]
use hashbrown::HashMap;
use std::collections::BTreeMap;

use crate::{
	HashedKey,
	ObjectSize,
	PaperPolicy,
};

use super::PolicyStack;

const NIL: u32 = u32::MAX;

/// Link-only slab slot. 16 bytes, asserted below.
#[derive(Clone, Copy)]
struct Slot {
	key: HashedKey,
	prev: u32,
	next: u32,
}

const _: () = assert!(
	std::mem::size_of::<Slot>() == 16,
	"Slot must stay 16 bytes: it is the per-object cost this design exists to minimise",
);

// Under `eviction_stacks_pmem` every structure here is allocated through the
// crate-wide `Hybrid` allocator (the far CXL/PMEM node), matching
// `CompactQueueSet` and `CompactFrequencyChain`.
//
// Without this the feature was a SILENT NO-OP for this stack. `LruCompactStack`
// is a thin wrapper over `CompactQueueSet` and inherited the relocation for
// free; this one owns its collections directly and kept them in DRAM, so a
// `lfu-compact` run built with the feature measured an unchanged stack while
// `get_policy_overhead` was (separately) still charging its bytes to the DRAM
// budget.
#[cfg(not(feature = "eviction_stacks_pmem"))]
type SlotVec = Vec<Slot>;
#[cfg(feature = "eviction_stacks_pmem")]
type SlotVec = Vec<Slot, crate::Hybrid>;

#[cfg(not(feature = "eviction_stacks_pmem"))]
type FreeVec = Vec<u32>;
#[cfg(feature = "eviction_stacks_pmem")]
type FreeVec = Vec<u32, crate::Hybrid>;

#[cfg(not(feature = "eviction_stacks_pmem"))]
type Index = HashMap<HashedKey, (u32, u32), crate::NoHasher>;
#[cfg(feature = "eviction_stacks_pmem")]
type Index = HashMap<HashedKey, (u32, u32), crate::NoHasher, crate::Hybrid>;

#[cfg(not(feature = "eviction_stacks_pmem"))]
type BucketMap = BTreeMap<u32, (u32, u32)>;
#[cfg(feature = "eviction_stacks_pmem")]
type BucketMap = BTreeMap<u32, (u32, u32), crate::Hybrid>;

/// The four empty collections, built in whichever allocator the feature selects.
#[cfg(not(feature = "eviction_stacks_pmem"))]
fn empty_collections() -> (SlotVec, FreeVec, Index, BucketMap) {
	(
		Vec::new(),
		Vec::new(),
		HashMap::with_hasher(crate::NoHasher::default()),
		BTreeMap::new(),
	)
}

#[cfg(feature = "eviction_stacks_pmem")]
fn empty_collections() -> (SlotVec, FreeVec, Index, BucketMap) {
	(
		Vec::new_in(crate::Hybrid),
		Vec::new_in(crate::Hybrid),
		HashMap::with_hasher_in(crate::NoHasher::default(), crate::Hybrid),
		BTreeMap::new_in(crate::Hybrid),
	)
}

pub struct LfuCompactStack {
	slots: SlotVec,
	free: FreeVec,
	/// key -> (slot, frequency). One probe returns both, which is the whole
	/// point: `LfuStack` needs index_map -> count_stacks -> node.
	index: Index,
	/// frequency -> (head, tail) of that bucket's intrusive list. Ordered, so
	/// the minimum frequency is the first entry — that is the eviction victim.
	buckets: BucketMap,
}

impl Default for LfuCompactStack {
	fn default() -> Self {
		let (slots, free, index, buckets) = empty_collections();
		LfuCompactStack { slots, free, index, buckets }
	}
}

impl LfuCompactStack {
	fn alloc(&mut self, key: HashedKey) -> u32 {
		let slot = Slot { key, prev: NIL, next: NIL };
		match self.free.pop() {
			Some(i) => { self.slots[i as usize] = slot; i },
			None => { self.slots.push(slot); (self.slots.len() - 1) as u32 },
		}
	}

	/// Unlinks `i` from bucket `freq`, dropping the bucket if it empties.
	fn unlink(&mut self, freq: u32, i: u32) {
		let (prev, next) = {
			let s = &self.slots[i as usize];
			(s.prev, s.next)
		};
		if prev != NIL { self.slots[prev as usize].next = next; }
		if next != NIL { self.slots[next as usize].prev = prev; }

		if let Some((head, tail)) = self.buckets.get_mut(&freq) {
			if *head == i { *head = next; }
			if *tail == i { *tail = prev; }
			if *head == NIL { self.buckets.remove(&freq); }
		}
	}

	/// Links `i` at the FRONT of bucket `freq`, so the bucket is
	/// recency-ordered within a frequency and its tail is the LRU victim —
	/// matching `CountStack::push`/`pop` in `LfuStack`.
	fn link_front(&mut self, freq: u32, i: u32) {
		let entry = self.buckets.entry(freq).or_insert((NIL, NIL));
		let old_head = entry.0;
		entry.0 = i;
		if entry.1 == NIL { entry.1 = i; }

		self.slots[i as usize].prev = NIL;
		self.slots[i as usize].next = old_head;
		if old_head != NIL { self.slots[old_head as usize].prev = i; }
	}
}

impl PolicyStack for LfuCompactStack {
	fn is_policy(&self, policy: &PaperPolicy) -> bool {
		matches!(policy, PaperPolicy::LfuCompact)
	}

	fn len(&self) -> usize {
		self.index.len()
	}

	fn contains(&self, key: HashedKey) -> bool {
		self.index.contains_key(&key)
	}

	fn insert(&mut self, key: HashedKey, _: ObjectSize) {
		if self.index.contains_key(&key) {
			return self.update(key);
		}
		let i = self.alloc(key);
		self.link_front(1, i);
		self.index.insert(key, (i, 1));
	}

	fn update(&mut self, key: HashedKey) {
		let Some(&(i, freq)) = self.index.get(&key) else { return };
		self.unlink(freq, i);

		// get_mut, NOT insert. This runs on every GET hit, and an insert of
		// an already-present key re-hashes, probes, writes, then returns and
		// drops the old value, plus checks growth -- where a get_mut only
		// hashes and probes. The tiered CompactFrequencyChain::bump mutates
		// in place for exactly this reason; using insert here handicapped the
		// flat stack against the tiered one on the very path being compared,
		// which is how a flat all-DRAM LFU first measured SLOWER than a
		// tiered one doing migrations.
		if let Some(e) = self.index.get_mut(&key) {
			e.1 = freq + 1;
		}

		self.link_front(freq + 1, i);
	}

	fn remove(&mut self, key: HashedKey) {
		let Some((i, freq)) = self.index.remove(&key) else { return };
		self.unlink(freq, i);
		self.free.push(i);
	}

	fn clear(&mut self) {
		self.slots.clear();
		self.free.clear();
		self.index.clear();
		self.buckets.clear();
	}

	/// Evicts the least-frequent key, breaking ties by least-recently-used —
	/// the bucket tail, matching `LfuStack::evict_one`'s `CountStack::pop`.
	fn evict_one(&mut self) -> Option<HashedKey> {
		let (&freq, &(_, tail)) = self.buckets.iter().next()?;
		if tail == NIL { return None }
		let key = self.slots[tail as usize].key;
		self.unlink(freq, tail);
		self.index.remove(&key);
		self.free.push(tail);
		Some(key)
	}
}

/// Fidelity to `LfuStack`, whose policy this re-lays-out and which was removed in
/// R2: the eviction order it gave for the two op sequences below is recorded in
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
