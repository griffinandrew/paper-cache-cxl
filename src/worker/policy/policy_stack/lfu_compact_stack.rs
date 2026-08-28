//! `LfuCompactStack` — `LfuStack`'s policy over the slab design.
//!
//! The non-tiered counterpart to `LfuCompactHybridStack`, and the fourth cell
//! of the layout/tiering matrix:
//!
//! ```text
//!                    multi-map layout       slab layout
//!   no tiering       LfuStack               LfuCompactStack   <- this
//!   tiered           LfuHybridStack         LfuCompactHybridStack
//! ```
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

pub struct LfuCompactStack {
	slots: Vec<Slot>,
	free: Vec<u32>,
	/// key -> (slot, frequency). One probe returns both, which is the whole
	/// point: `LfuStack` needs index_map -> count_stacks -> node.
	index: std::collections::HashMap<HashedKey, (u32, u32), crate::NoHasher>,
	/// frequency -> (head, tail) of that bucket's intrusive list. Ordered, so
	/// the minimum frequency is the first entry — that is the eviction victim.
	buckets: BTreeMap<u32, (u32, u32)>,
}

impl Default for LfuCompactStack {
	fn default() -> Self {
		LfuCompactStack {
			slots: Vec::new(),
			free: Vec::new(),
			index: std::collections::HashMap::with_hasher(crate::NoHasher::default()),
			buckets: BTreeMap::new(),
		}
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

/// Fidelity against `LfuStack`, whose policy this re-lays-out.
#[cfg(test)]
mod fidelity_tests {
	use super::*;
	use super::super::lfu_stack::LfuStack;

	/// Same access sequence must give the same eviction order: this changes
	/// how frequency is STORED, not what LFU means.
	#[test]
	fn evicts_in_the_same_order_as_lfu_stack() {
		let mut a = LfuStack::default();
		let mut b = LfuCompactStack::default();
		let mut x: u64 = 0x243F_6A88_85A3_08D3;

		for i in 0..40_000u64 {
			x ^= x << 13;
			x ^= x >> 7;
			x ^= x << 17;
			let u = (x >> 11) as f64 / (1u64 << 53) as f64;
			let key = ((u * u * 400.0) as u64) + 1;
			let sa: &mut dyn PolicyStack = &mut a;
			let sb: &mut dyn PolicyStack = &mut b;
			if i % 4 == 3 && sa.contains(key) {
				sa.update(key);
				sb.update(key);
			} else {
				sa.insert(key, 1_024);
				sb.insert(key, 1_024);
			}
			assert_eq!(sa.len(), sb.len(), "len diverged at op {i}");
		}

		let mut oa = Vec::new();
		let mut ob = Vec::new();
		let sa: &mut dyn PolicyStack = &mut a;
		let sb: &mut dyn PolicyStack = &mut b;
		while let Some(k) = sa.evict_one() { oa.push(k); }
		while let Some(k) = sb.evict_one() { ob.push(k); }
		assert_eq!(oa, ob, "eviction order diverged from LfuStack");
		assert!(!oa.is_empty());
	}

	#[test]
	fn removal_matches_lfu_stack() {
		let mut a = LfuStack::default();
		let mut b = LfuCompactStack::default();
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
		let mut oa = Vec::new();
		let mut ob = Vec::new();
		let sa: &mut dyn PolicyStack = &mut a;
		let sb: &mut dyn PolicyStack = &mut b;
		while let Some(k) = sa.evict_one() { oa.push(k); }
		while let Some(k) = sb.evict_one() { ob.push(k); }
		assert_eq!(oa, ob, "eviction order after removals diverged");
	}
}
