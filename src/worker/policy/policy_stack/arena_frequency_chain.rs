/*
 * Copyright (c) Kia Shakiba
 *
 * This source code is licensed under the GNU AGPLv3 license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! [`CompactFrequencyChain`] with the key stored ONCE instead of twice.
//!
//! # What this is
//!
//! The frequency-ordered half of the arena conversion. Nineteen of the
//! twenty-one hybrid compact stacks moved from `CompactQueueSet` to
//! [`ArenaQueueSet`] by changing a type name, because their orders are queues
//! and `ArenaQueueSet` holds queues. The two LFU-ranked stacks could not:
//!
//! * `LfuCompactHybridStack` needs one ordered bucket per DISTINCT FREQUENCY,
//!   per tier, because eviction has to find the MINIMUM frequency. That is a
//!   `BTreeMap<u32, Bucket>` -- unbounded and data-dependent -- and a
//!   fixed `[u32; MAX_QUEUES]` tag cannot express it.
//!
//! * `LruLfuCompactHybridStack` additionally threads a THIRD, recency-ordered
//!   list over the same slab, for a fast tier that ranks by recency beside a
//!   slow tier that ranks by frequency. That works because a key is in exactly
//!   one tier at a time, so one `prev`/`next` pair per slot serves whichever
//!   list the key is currently in.
//!
//! So the ORDERS are kept from `CompactFrequencyChain` and the STORAGE is taken
//! from the arena. The bucket maps stay: they are O(distinct frequencies), not
//! O(objects), and were never the cost.
//!
//! # What changes, and what it is worth
//!
//! Only the index. `CompactFrequencyChain` is a 16-byte slot plus a
//! `HashMap<HashedKey, (u32, CompactEntry), NoHasher>`, and that map stores the
//! key a second time so a probe can compare it -- the same 8 bytes already in
//! the slot, in the same structure. [`KeylessIndex`] stores bare `u32` slot
//! numbers and verifies a probe against `slots[i].key`, so the index falls from
//! 56 B/object to 8.
//!
//! ```text
//! CompactFrequencyChain   slot 16 (key,prev,next)  + index 56           = 72
//! ArenaFrequencyChain     node 32 (key,prev,next,NodePayload) + index 8 = 40
//! ```
//!
//! The node grows 16 bytes because the payload moves INTO it and because it is
//! the shared [`NodePayload`] rather than a 12-byte `CompactEntry` -- the same
//! node all nineteen converted stacks now carry, so an LFU key and an LRU key
//! cost the same and no stack needs a payload of its own. Against a 56-byte
//! index that trade is 32 B/object.
//!
//! MEASURED with `measure_one_point`, release, ONE PROCESS PER POINT, at powers
//! of two, `MEASURE_POLICY=lfu-compact-hybrid`:
//!
//! ```text
//!   n        before (CompactFrequencyChain)   after (this)
//!   2^20              72.5952                   40.2100
//!   2^21              72.2962                   40.1050
//!   2^22              72.1451                   40.0518
//!   2^23              72.0707                   40.0244
//! ```
//!
//! Forty is predicted rather than fitted: the node is 32 bytes and the keyless
//! bucket array is 8 B/object at the doubling slack it holds. The residue above
//! it is a fixed intercept, not a per-object term, which is why it shrinks with
//! n. The figures are identical to `lru-compact-hybrid`'s, which is the
//! prediction the shared node makes and the check that the bucket maps really
//! are O(distinct frequencies): if they scaled with the object count they would
//! show up here as a term the queue stacks do not have.
//!
//! # What does not change
//!
//! The algorithm, method for method. Buckets are still one-per-distinct-
//! frequency, `bump` is still an O(1) unlink and relink into the adjacent
//! bucket, the minimum is still the first entry of an ordered map, and the
//! recency list is still invisible to the frequency-only path. Two differential
//! tests in this file drive this structure and `CompactFrequencyChain` through
//! the same random operation streams -- one per FACE, since the frequency stack
//! and the recency stack use the chain under different contracts -- and require
//! identical observable state at every step. That is the evidence that
//! "representation change only" is a fact about this commit rather than an
//! intention.
//!
//! # Tie order across a tier move
//!
//! Upstream `LfuStack` breaks ties among equal counts by the order keys
//! ENTERED the count -- `push_front` on reaching it, `pop_back` on eviction --
//! and CLAUDE.md's LFU design decision 4 says the tiered stack means the same
//! thing. A bucket list gets that for free as long as every key appended to it
//! is the newest key at that count. Promotion keeps that true: it always
//! immediately follows a `bump` of the same key. Demotion did not: `set_tier`
//! re-appended the demoted key at the NEWEST end of the slow bucket, so a key
//! that reached its count long ago ranked as if it had just arrived, behind
//! slow keys that arrived after it.
//!
//! The fix keeps two runs per bucket. The NATIVE run (`head`/`tail`) is what
//! there always was: every append is stamped with the chain's clock, so it is
//! stamp-ascending. The DEMOTED run (`demoted_head`/`demoted_tail`) is appended
//! only by [`ArenaFrequencyChain::demote_min_fast`], which keeps the key's own
//! stamp -- the moment it reached its count. It is stamp-ascending too, by
//! construction: fast buckets only ever take native appends, so they are
//! stamp-ascending, and a demotion always takes the HEAD (oldest) of the fast
//! minimum bucket, so every key still in that fast bucket, and every key that
//! enters it later, is at least as new as the one demoted. A bucket's oldest
//! key is therefore the older of its two run heads: O(1), no walk, no sort.
//!
//! A side effect worth naming: with every key's stamp meaning "when it reached
//! its current count" in BOTH tiers, the two tiers' minima are comparable with
//! each other, and [`ArenaFrequencyChain::min_over_both_tiers`] is exactly
//! upstream LFU's victim. `lfu-global-compact-hybrid` evicts through it.
//!
//! The stamp is 40 bits wide and costs no bytes, so the node stays 32. Its low
//! 32 bits are `NodePayload::ts`, which no other stack reads. Its high 8 are
//! `NodePayload::queue`, which only the multi-queue stacks use, each on its own
//! `ArenaQueueSet` slab: this chain has no queues, and before the stamp neither
//! of its faces read or wrote the byte except to zero it in [`node`]. The
//! clock ticks once per `insert` or `bump` -- at most once per trace record --
//! and wraps at 2^40, and two stamps are compared by AGE, `(clock - stamp) mod
//! 2^40`, which is exact while both keys' ages are below 2^40 (about 1.1e12)
//! ticks: some 900 times the 1.2e9 records of the largest trace. Past that
//! window a stale key's age aliases small and it ranks as newer than it is --
//! the pre-fix behaviour, for that key only. The runs themselves are
//! structural and cannot be corrupted by a wrap: the stamp only ever chooses
//! between two heads -- a bucket's two run heads, or, in
//! [`ArenaFrequencyChain::min_over_both_tiers`], the two tiers' minimum
//! heads. So past the window a misranked key can also take the eviction from
//! the wrong TIER under `lfu-global-compact-hybrid`; that changes which key
//! goes, never what the chain holds.
//!
//! `ts` alone would have been a 2^32 window, and that one a long-lived cache
//! CAN outrun -- 4.3e9 touches is under four passes of the largest trace
//! without a wipe -- which is why the spare byte is borrowed rather than the
//! narrower window accepted.
//!
//! The bucket maps grow with the fix from 8 to 16 bytes of value per entry.
//! Still per DISTINCT count present, not per object -- and a bound, not a
//! hope: a tier can only hold `D` distinct counts if its keys were touched at
//! least `1 + 2 + ... + D` times, so `D <= sqrt(2N)` after `N` touches. At the
//! 1.2e9 records of the largest trace that is under 49,000 buckets per tier,
//! about 1 MB of keys and values per tier, before a `BTreeMap`'s node overhead
//! (about 2.5x at worst, with every node at its minimum fill). Real traces sit
//! far below the bound: it needs every count from 1 to `D` present at once.
//!
//! `LruLfuCompactHybridStack` never demotes through `demote_min_fast` -- its
//! demotions come off the recency list -- so its buckets never hold a demoted
//! run and every method it calls behaves exactly as before.
//!
//! [`ArenaQueueSet`]: super::arena_queue_set::ArenaQueueSet
//! [`CompactFrequencyChain`]: super::compact_frequency_chain::CompactFrequencyChain

use std::collections::BTreeMap;

use crate::{
	HashedKey,
	object::ObjectSize,
	worker::policy::policy_stack::{
		Tier,
		arena_index::{
			ArenaSlot,
			KeylessIndex,
			NIL,
			SlotVec,
			U32Vec,
			new_slot_vec,
			new_u32_vec,
		},
		arena_queue_set::NodePayload,
	},
};

// The bucket maps carry the same allocator gating as everything else here, and
// for the same reason: `get_hybrid_dram_shared_overhead` drops the
// eviction-stack DRAM charge to ZERO under `eviction_stacks_pmem`, on the
// premise the stack is not in DRAM. A map that ignored the gate would sit in
// DRAM and be charged nothing.
//
// One entry per DISTINCT frequency rather than per object, so this stays small
// -- about 1 MB of entries per tier even at the bound for a 1.2e9-record
// trace; see "Tie order across a tier move".
#[cfg(not(feature = "eviction_stacks_pmem"))]
type BucketMap = BTreeMap<u32, Bucket>;
#[cfg(feature = "eviction_stacks_pmem")]
type BucketMap = BTreeMap<u32, Bucket, crate::Hybrid>;

/// One frequency's bucket: two intrusive runs over the slab, each `NIL`-ended.
///
/// Sixteen bytes per DISTINCT frequency per tier, up from eight -- not per
/// object, so the per-object figure does not move. O(distinct counts present),
/// which is at most `sqrt(2N)` after `N` touches: about 1 MB of entries per
/// tier at the largest trace's 1.2e9 records (see the module doc).
#[derive(Clone, Copy, Debug)]
struct Bucket {
	/// Keys that reached this count IN this tier: admission, `bump`, and the
	/// promotion `set_tier` that follows a bump. Stamp-ascending, head oldest.
	head: u32,
	tail: u32,

	/// Keys DEMOTED into this count by `demote_min_fast`, carrying their
	/// original stamps. Stamp-ascending by the argument in the module doc.
	/// Always `NIL` in a fast bucket, and in every bucket of the recency face.
	demoted_head: u32,
	demoted_tail: u32,
}

impl Bucket {
	fn native(slot: u32) -> Self {
		Bucket { head: slot, tail: slot, demoted_head: NIL, demoted_tail: NIL }
	}

	fn demoted(slot: u32) -> Self {
		Bucket { head: NIL, tail: NIL, demoted_head: slot, demoted_tail: slot }
	}

	fn is_empty(&self) -> bool {
		self.head == NIL && self.demoted_head == NIL
	}
}

#[cfg(not(feature = "eviction_stacks_pmem"))]
fn new_bucket_maps() -> (BucketMap, BucketMap) {
	(BTreeMap::new(), BTreeMap::new())
}

#[cfg(feature = "eviction_stacks_pmem")]
fn new_bucket_maps() -> (BucketMap, BucketMap) {
	(BTreeMap::new_in(crate::Hybrid), BTreeMap::new_in(crate::Hybrid))
}

/// Frequency-ordered buckets per tier, plus a recency-ordered list, over one
/// arena slab addressed by one keyless index.
pub struct ArenaFrequencyChain {
	slots: SlotVec<NodePayload>,

	/// Bare slot numbers, verified against the slot's own key. This is the
	/// whole of the saving over `CompactFrequencyChain`; see
	/// [`arena_index`](super::arena_index).
	index: KeylessIndex,

	/// Freed slab slots, reused before the slab grows.
	free: U32Vec,

	/// frequency -> that bucket's two intrusive runs (see [`Bucket`]), one map
	/// per tier. Ordered, so a tier's minimum frequency is its first entry.
	///
	/// Two bucket sets over *one* slab is what lets this replace both of
	/// `FrequencyChain`'s chains **and** the `entries` map they were paired
	/// with: a key is located in a single probe, and the slot that probe
	/// returns already carries its tier, size and frequency.
	fast_buckets: BucketMap,
	slow_buckets: BucketMap,

	fast_len: usize,
	slow_len: usize,

	/// Last-touch clock: ticks once per `insert` and per `bump`, and is written
	/// into the touched node's 40-bit stamp (see [`stamp_of`]). Wraps at 2^40,
	/// so it never exceeds [`STAMP_MASK`]; only ever compared as an AGE
	/// (`clock - stamp`, mod 2^40), and only between two heads: the two run
	/// heads of one bucket, or the two tiers' minimum heads in
	/// `min_over_both_tiers`. Owned by the chain and touched only by the
	/// policy worker that owns the stack, so it is a plain field: no atomic, no
	/// lock.
	clock: u64,

	/// Head and tail of the DISTINGUISHED RECENCY LIST: a third intrusive list
	/// over the SAME slab, ordered by recency rather than by frequency.
	///
	/// `LruLfuCompactHybridStack` needs a recency-ordered fast tier beside a
	/// frequency-ordered slow one. Those two populations are disjoint -- a key
	/// is in the fast tier or the slow tier, never both -- so one `prev`/`next`
	/// pair per slot serves either, and `fast_len`/`slow_len` keep counting
	/// tier membership exactly as they do for LFU. That is what lets one slab
	/// and one index carry a policy whose tiers rank by different metrics.
	///
	/// The frequency-bucket stack (`LfuCompactHybridStack`) never calls a
	/// `recency_*` method, so for it these stay `NIL` for the structure's whole
	/// life and every other method behaves exactly as it would without them.
	recency_head: u32,
	recency_tail: u32,
}

impl Default for ArenaFrequencyChain {
	fn default() -> Self {
		let (fast_buckets, slow_buckets) = new_bucket_maps();

		ArenaFrequencyChain {
			slots: new_slot_vec(),
			index: KeylessIndex::default(),
			free: new_u32_vec(),
			fast_buckets,
			slow_buckets,
			fast_len: 0,
			slow_len: 0,
			clock: 0,
			recency_head: NIL,
			recency_tail: NIL,
		}
	}
}

/// A brand-new node for this chain.
///
/// `phys` is set equal to `tier` and kept there by every tier move below.
/// Nothing in either LFU stack reads it -- only the lazy-copy design does, and
/// only because it promotes logically and defers the byte copy -- but the
/// node's contract is that the two are equal for everyone else, and a `phys`
/// left behind at admission tier would quietly make that false.
///
/// `ts` and `queue` start at zero here. There are no queues, so neither field
/// means what it means on an `ArenaQueueSet` node: together they are the
/// 40-bit last-touch stamp ([`stamp_of`]), which the frequency face overwrites
/// with the chain's clock on every `insert` and `bump` (see "Tie order across a
/// tier move"). The recency face never reads or writes either -- recency there
/// is `prev`/`next`.
fn node(size: ObjectSize, freq: u32, dram_resident: u8, tier: Tier) -> NodePayload {
	NodePayload {
		size,
		freq,
		ts: 0,
		queue: 0,
		tier: Some(tier),
		phys: Some(tier),
		dram_resident,
	}
}

/// Bits in the last-touch stamp: `NodePayload::ts` (32) + `NodePayload::queue`
/// (8). The clock wraps at this width.
const STAMP_BITS: u32 = 40;

/// The clock's range, and the modulus two stamps' ages are taken in.
const STAMP_MASK: u64 = (1 << STAMP_BITS) - 1;

/// A node's 40-bit last-touch stamp: `ts` is the low 32 bits and `queue`, a
/// byte this chain has no other use for, the high 8.
#[inline]
fn stamp_of(payload: &NodePayload) -> u64 {
	((payload.queue as u64) << 32) | payload.ts as u64
}

/// Writes a 40-bit stamp into `ts` and `queue`. `stamp` is a clock value, so
/// it is already within [`STAMP_MASK`] and the high byte cannot truncate.
#[inline]
fn set_stamp(payload: &mut NodePayload, stamp: u64) {
	debug_assert!(stamp <= STAMP_MASK, "stamp {stamp:#x} is wider than {STAMP_BITS} bits");

	payload.ts = stamp as u32;
	payload.queue = (stamp >> 32) as u8;
}

impl ArenaFrequencyChain {
	/// Slab slots currently allocated. Exposed so a test can assert that
	/// construction does NOT allocate from the cache budget: these stacks grow
	/// dynamically, and an eager reservation sized from capacity was removed
	/// because it reserved far more than the eval workload can hold while still
	/// not preventing doubling on the real traces.
	pub fn slab_capacity(&self) -> usize {
		self.slots.capacity()
	}

	/// Buckets in the keyless index. Exposed for the per-object measurement,
	/// which has to know the table size to reason about its cost.
	pub fn index_capacity(&self) -> usize {
		self.index.capacity()
	}

	/// Pre-sizes the slab and the index for `objects` entries.
	///
	/// The slab is a `Vec`, so growth is never in place: every doubling
	/// reallocates and COPIES every entry. At eval-trace scale that is one
	/// multi-hundred-millisecond stall on the policy worker -- measured at
	/// 827 ms -- and it would never have surfaced as a regression, because the
	/// policy stack runs behind an unbounded channel on its own thread and the
	/// client latency columns structurally cannot observe it.
	///
	/// Reserving costs no resident memory: the pages are not touched until
	/// entries occupy them.
	pub fn reserve(&mut self, objects: usize) {
		self.slots.reserve(objects);
		self.index.reserve(&self.slots, objects);
	}

	pub fn len(&self) -> usize { self.fast_len + self.slow_len }
	pub fn is_empty(&self) -> bool { self.len() == 0 }
	pub fn fast_len(&self) -> usize { self.fast_len }
	pub fn slow_len(&self) -> usize { self.slow_len }

	pub fn contains(&self, key: HashedKey) -> bool {
		self.index.get(&self.slots, key) != NIL
	}

	/// One index probe, then one slab dereference.
	///
	/// `CompactFrequencyChain` returned this in a single probe, because its
	/// payload rode in the hash bucket. What is not the same as that trade is
	/// the size of the thing probed first: the index here is 8 B/object against
	/// 56 there, so the first touch is into a table seven times likelier to be
	/// cache-resident.
	pub fn get(&self, key: HashedKey) -> Option<NodePayload> {
		self.slot_of(key).map(|slot| self.slots[slot as usize].payload)
	}

	/// Admits a key at frequency 1, stamped as the newest key there is.
	pub fn insert(&mut self, key: HashedKey, size: ObjectSize, dram_resident: u8, tier: Tier) {
		if self.contains(key) {
			return;
		}

		let stamp = self.tick();
		let slot = self.alloc_slot(key, node(size, 1, dram_resident, tier));
		set_stamp(&mut self.slots[slot as usize].payload, stamp);

		self.index.insert(&self.slots, slot);
		self.link(slot, 1, tier);

		match tier {
			Tier::Fast => self.fast_len += 1,
			Tier::Slow => self.slow_len += 1,
		}
	}

	/// Moves a key to the next frequency bucket. O(1): unlink, relink. The key
	/// is restamped: it has just reached its new count, so it is the newest key
	/// at that count -- which is why a native append is always in order. At the
	/// `u32` cap the count stays put but the key is still restamped and moved
	/// to the newest end, as before.
	pub fn bump(&mut self, key: HashedKey) -> u32 {
		let Some(slot) = self.slot_of(key) else { return 0 };
		let payload = self.slots[slot as usize].payload;
		let Some(tier) = payload.tier else { return 0 };

		self.unlink(slot, payload.freq, tier);

		let next_freq = payload.freq.saturating_add(1);
		let stamp = self.tick();
		self.slots[slot as usize].payload.freq = next_freq;
		set_stamp(&mut self.slots[slot as usize].payload, stamp);
		self.link(slot, next_freq, tier);

		next_freq
	}

	/// The least-frequently-used key in a tier: the older of the two run heads
	/// of its lowest-frequency bucket. O(log D) in the number of distinct
	/// frequencies present, plus one stamp comparison.
	pub fn min_key(&self, tier: Tier) -> Option<HashedKey> {
		let (_, bucket) = self.buckets(tier).iter().next()?;
		Some(self.slots[self.bucket_head(bucket) as usize].key)
	}

	/// The lowest frequency present in a tier, or `None` if it is empty.
	///
	/// The promotion rule compares a slow key's new count against this: a slow
	/// key overtakes the fast tier only by *strictly* exceeding its minimum.
	pub fn min_count(&self, tier: Tier) -> Option<u32> {
		self.buckets(tier).keys().next().copied()
	}

	/// The least-frequently-used key in a tier together with its count.
	pub fn min_with_count(&self, tier: Tier) -> Option<(HashedKey, u32)> {
		let (&freq, bucket) = self.buckets(tier).iter().next()?;
		Some((self.slots[self.bucket_head(bucket) as usize].key, freq))
	}

	/// The least-frequently-used key across BOTH tiers, with its count and the
	/// tier it is in: the lower of the two tier minima's counts, and between
	/// equal counts the key that reached the count first -- the older of the
	/// two bucket heads by stamp age, the same comparison `bucket_head` makes
	/// between a bucket's two runs. Identical stamps go to the slow tier.
	///
	/// This is upstream `LfuStack`'s victim, and it is only that because of
	/// the tie-order fix: every live key's stamp is now the moment it reached
	/// its CURRENT count, whichever tier it is in -- a bump stamps it, a
	/// promotion restamps it with that same bump's clock value, and a demotion
	/// (`demote_min_fast`) keeps it -- so stamps from the two tiers are one
	/// comparable sequence. Before the fix a demoted key carried its demotion
	/// time, and a cross-tier comparison would have ranked it by that.
	///
	/// Stamps are distinct between live keys (each tick is written into one
	/// key), so the slow-on-a-tie rule is a determinism rule, not a policy.
	/// Two `BTreeMap` first-entry lookups and at most four stamp reads: two to
	/// pick the slow bucket's head between its runs (a fast bucket never has a
	/// demoted run, so its head costs none), and two to compare across tiers.
	///
	/// Exact while every live key's stamp is within the 2^40 window (see the
	/// module doc). Past it, the cross-tier comparison can hand the eviction
	/// to the wrong tier's head -- a different victim, no corruption.
	pub fn min_over_both_tiers(&self) -> Option<(HashedKey, u32, Tier)> {
		let head = |(&freq, bucket): (&u32, &Bucket)| (self.bucket_head(bucket), freq);

		let (slot, freq, tier) = match (
			self.slow_buckets.iter().next().map(head),
			self.fast_buckets.iter().next().map(head),
		) {
			(None, None) => return None,
			(Some((slot, freq)), None) => (slot, freq, Tier::Slow),
			(None, Some((slot, freq))) => (slot, freq, Tier::Fast),

			(Some((slow, slow_freq)), Some((fast, fast_freq))) => {
				let slow_first = match slow_freq.cmp(&fast_freq) {
					std::cmp::Ordering::Less => true,
					std::cmp::Ordering::Greater => false,
					std::cmp::Ordering::Equal => self.not_newer(
						stamp_of(&self.slots[slow as usize].payload),
						stamp_of(&self.slots[fast as usize].payload),
					),
				};

				match slow_first {
					true => (slow, slow_freq, Tier::Slow),
					false => (fast, fast_freq, Tier::Fast),
				}
			},
		};

		Some((self.slots[slot as usize].key, freq, tier))
	}

	pub fn remove(&mut self, key: HashedKey) -> Option<NodePayload> {
		let slot = self.index.remove(&self.slots, key)?;
		let payload = self.slots[slot as usize].payload;

		match payload.tier {
			Some(Tier::Fast) => {
				self.unlink(slot, payload.freq, Tier::Fast);
				self.fast_len -= 1;
			},

			Some(Tier::Slow) => {
				self.unlink(slot, payload.freq, Tier::Slow);
				self.slow_len -= 1;
			},

			// The shared node makes `tier` optional because the 2Q and S3-FIFO
			// families legitimately have a queue with no tier of its own. Every
			// path into this chain records one, so this arm is unreachable --
			// and it is spelled out rather than papered over with `unwrap_or`,
			// which would silently unlink the slot from the wrong bucket set
			// and leave the other one holding a freed slot.
			None => {},
		}

		self.free.push(slot);

		Some(payload)
	}

	/// Moves a key between tiers, preserving its frequency. Relinks it from one
	/// bucket set into the other -- the key never moves in the slab, so its
	/// index entry and every link to it stay valid.
	///
	/// A RE-APPEND: the key goes to the newest end of the destination bucket's
	/// native run and is restamped with the current clock (without ticking), so
	/// it ranks as if it had just reached its count. That is exactly right for
	/// a promotion, which always immediately follows a `bump` of the same key --
	/// the restamp writes the value the bump just wrote. It is WRONG for a
	/// demotion, which must keep the key's place among equal counts: demote
	/// with [`demote_min_fast`](Self::demote_min_fast).
	pub fn set_tier(&mut self, key: HashedKey, tier: Tier) {
		let Some(slot) = self.slot_of(key) else { return };
		let payload = self.slots[slot as usize].payload;
		let Some(old_tier) = payload.tier else { return };

		if old_tier == tier {
			return;
		}

		self.unlink(slot, payload.freq, old_tier);
		self.set_tier_fields(slot, tier);
		set_stamp(&mut self.slots[slot as usize].payload, self.clock);
		self.link(slot, payload.freq, tier);

		match tier {
			Tier::Fast => { self.fast_len += 1; self.slow_len -= 1; },
			Tier::Slow => { self.slow_len += 1; self.fast_len -= 1; },
		}
	}

	/// Demotes the fast tier's least-frequently-used key -- the head of its
	/// lowest-frequency bucket -- into the slow bucket of the same count,
	/// KEEPING its stamp, so it ranks among equal-count slow keys by when it
	/// reached the count rather than by when it was demoted. Returns the key and
	/// its entry as it now stands, or `None` if the fast tier is empty.
	///
	/// No index probe: the victim is found through the bucket map and moved by
	/// slot, where `min_with_count` + `get` + `set_tier` probed twice.
	///
	/// Taking the head is not an optimisation but the precondition that keeps
	/// the destination's demoted run stamp-ascending (see the module doc), which
	/// is why there is no `demote(key)`.
	pub fn demote_min_fast(&mut self) -> Option<(HashedKey, NodePayload)> {
		let (freq, slot) = {
			let (&freq, bucket) = self.fast_buckets.iter().next()?;
			debug_assert_eq!(bucket.demoted_head, NIL, "a fast bucket never holds a demoted run");
			(freq, bucket.head)
		};

		let key = self.slots[slot as usize].key;

		self.unlink(slot, freq, Tier::Fast);
		self.set_tier_fields(slot, Tier::Slow);
		self.link_demoted(slot, freq);

		self.fast_len -= 1;
		self.slow_len += 1;

		Some((key, self.slots[slot as usize].payload))
	}

	pub fn resize(&mut self, key: HashedKey, size: ObjectSize, dram_resident: u8) {
		let Some(slot) = self.slot_of(key) else { return };
		let payload = &mut self.slots[slot as usize].payload;

		payload.size = size;
		payload.dram_resident = dram_resident;
	}

	pub fn clear(&mut self) {
		self.slots.clear();
		self.index.clear();
		self.free.clear();
		self.fast_buckets.clear();
		self.slow_buckets.clear();
		self.fast_len = 0;
		self.slow_len = 0;
		self.clock = 0;
		self.recency_head = NIL;
		self.recency_tail = NIL;
	}

	/// Advances the last-touch clock, wrapping at 2^40, and returns the new
	/// stamp.
	#[inline]
	fn tick(&mut self) -> u64 {
		self.clock = (self.clock + 1) & STAMP_MASK;
		self.clock
	}

	/// How many ticks ago `stamp` was taken, modulo 2^40.
	#[inline]
	fn age(&self, stamp: u64) -> u64 {
		self.clock.wrapping_sub(stamp) & STAMP_MASK
	}

	/// Whether the key stamped `a` reached its count no later than the key
	/// stamped `b`.
	///
	/// Compares AGES rather than the stamps: both keys are in the past, so each
	/// age is exact while below 2^40 ticks, twice the window of a signed
	/// serial-number comparison. Stamps must never be compared directly -- see
	/// `NodePayload::ts`.
	#[inline]
	fn not_newer(&self, a: u64, b: u64) -> bool {
		self.age(a) >= self.age(b)
	}

	/// The oldest key of a bucket: the older of its two run heads. A tie --
	/// possible only between two `set_tier` restamps -- goes to the demoted run.
	#[inline]
	fn bucket_head(&self, bucket: &Bucket) -> u32 {
		match (bucket.head, bucket.demoted_head) {
			(head, NIL) => head,
			(NIL, demoted) => demoted,

			(head, demoted) => {
				let head_stamp = stamp_of(&self.slots[head as usize].payload);
				let demoted_stamp = stamp_of(&self.slots[demoted as usize].payload);

				if self.not_newer(demoted_stamp, head_stamp) { demoted } else { head }
			},
		}
	}

	/// The slot holding `key`, or `None`. One probe into the keyless index.
	#[inline]
	fn slot_of(&self, key: HashedKey) -> Option<u32> {
		match self.index.get(&self.slots, key) {
			NIL => None,
			slot => Some(slot),
		}
	}

	fn buckets(&self, tier: Tier) -> &BucketMap {
		match tier {
			Tier::Fast => &self.fast_buckets,
			Tier::Slow => &self.slow_buckets,
		}
	}

	/// Records a tier on a slot, keeping `phys` equal to it. See [`node`].
	fn set_tier_fields(&mut self, slot: u32, tier: Tier) {
		let payload = &mut self.slots[slot as usize].payload;

		payload.tier = Some(tier);
		payload.phys = Some(tier);
	}

	/// Takes a free slab slot, or grows the slab by one. Shared by the
	/// frequency admissions and the recency ones below -- one slab, one free
	/// list, whichever list the key joins.
	fn alloc_slot(&mut self, key: HashedKey, payload: NodePayload) -> u32 {
		let node = ArenaSlot { key, prev: NIL, next: NIL, payload };

		match self.free.pop() {
			Some(slot) => {
				self.slots[slot as usize] = node;
				slot
			},

			None => {
				let slot = self.slots.len() as u32;
				assert!(slot != NIL, "ArenaFrequencyChain exceeded u32::MAX - 1 slots");
				self.slots.push(node);
				slot
			},
		}
	}

	// ── the distinguished recency list ────────────────────────────────────
	//
	// Everything below is additive: it maintains `recency_head`/`recency_tail`
	// over the same slots the frequency buckets use, and touches
	// `fast_buckets` never. `LfuCompactHybridStack` calls none of it.

	fn recency_link_front(&mut self, slot: u32) {
		let old = self.recency_head;

		{
			let s = &mut self.slots[slot as usize];
			s.prev = NIL;
			s.next = old;
		}

		match old {
			NIL => self.recency_tail = slot,
			o => self.slots[o as usize].prev = slot,
		}

		self.recency_head = slot;
	}

	fn recency_unlink(&mut self, slot: u32) {
		let (prev, next) = {
			let s = &self.slots[slot as usize];
			(s.prev, s.next)
		};

		match prev {
			NIL => self.recency_head = next,
			p => self.slots[p as usize].next = next,
		}

		match next {
			NIL => self.recency_tail = prev,
			n => self.slots[n as usize].prev = prev,
		}
	}

	/// Admits a NEW key at the recency head, in the fast tier, at `freq`.
	///
	/// The frequency is carried metadata here, not a ranking key: nothing in
	/// the recency list is ordered by it. It exists so a later demotion can
	/// enter the slow tier at the count the key actually earned.
	pub fn recency_push_front(
		&mut self,
		key: HashedKey,
		size: ObjectSize,
		dram_resident: u8,
		freq: u32,
	) {
		if self.contains(key) {
			return;
		}

		let slot = self.alloc_slot(key, node(size, freq, dram_resident, Tier::Fast));

		self.index.insert(&self.slots, slot);
		self.recency_link_front(slot);

		self.fast_len += 1;
	}

	/// Moves an existing recency-list key to the head. O(1).
	pub fn recency_move_front(&mut self, key: HashedKey) {
		let Some(slot) = self.slot_of(key) else { return };

		if self.recency_head == slot {
			return;
		}

		self.recency_unlink(slot);
		self.recency_link_front(slot);
	}

	/// The LRU end of the recency list: the demotion (and last-resort
	/// eviction) candidate.
	pub fn recency_back(&self) -> Option<HashedKey> {
		(self.recency_tail != NIL).then(|| self.slots[self.recency_tail as usize].key)
	}

	/// Removes a recency-list key outright, freeing its slot.
	pub fn recency_remove(&mut self, key: HashedKey) -> Option<NodePayload> {
		let slot = self.index.remove(&self.slots, key)?;
		let payload = self.slots[slot as usize].payload;

		self.recency_unlink(slot);
		self.free.push(slot);
		self.fast_len -= 1;

		Some(payload)
	}

	/// Moves the recency tail into the slow tier, into the bucket for the
	/// frequency it already carries. Returns the demoted key and its entry as
	/// it now stands.
	///
	/// This is the whole of a demotion. `FrequencyChain` needs a `pop_back`
	/// from one structure and an `insert_at` into another with the count passed
	/// across by hand; here the entry never moves in the slab, so the count is
	/// carried by construction and there is nothing to drop.
	///
	/// `CompactFrequencyChain` looked the tail's key back up in its index and
	/// bailed out if it was missing, a state it documented as impossible. It is
	/// not merely impossible but unrepresentable -- the recency list is
	/// threaded through the slots the index points at, so a listed slot IS an
	/// indexed slot -- so the tail's payload is read straight out of the slab
	/// and there is no bail-out arm to be wrong about.
	pub fn demote_recency_back(&mut self) -> Option<(HashedKey, NodePayload)> {
		if self.recency_tail == NIL {
			return None;
		}

		let slot = self.recency_tail;
		let key = self.slots[slot as usize].key;

		self.set_tier_fields(slot, Tier::Slow);
		let payload = self.slots[slot as usize].payload;

		self.recency_unlink(slot);
		self.link(slot, payload.freq, Tier::Slow);

		self.fast_len -= 1;
		self.slow_len += 1;

		Some((key, payload))
	}

	/// Moves a slow-tier key to the recency head, setting its frequency to
	/// `freq`. The whole of a promotion; `None` if the key is untracked or is
	/// not in the slow tier.
	pub fn promote_to_recency_front(&mut self, key: HashedKey, freq: u32) -> Option<NodePayload> {
		let slot = self.slot_of(key)?;
		let payload = self.slots[slot as usize].payload;

		if payload.tier != Some(Tier::Slow) {
			return None;
		}

		self.unlink(slot, payload.freq, Tier::Slow);

		self.set_tier_fields(slot, Tier::Fast);
		self.slots[slot as usize].payload.freq = freq;
		let payload = self.slots[slot as usize].payload;

		self.recency_link_front(slot);

		self.slow_len -= 1;
		self.fast_len += 1;

		Some(payload)
	}

	/// Sets a key's frequency and touches no list.
	///
	/// For a recency-list key only: its counter is carried metadata that ranks
	/// nothing, so there is no bucket to move it between. Calling this on a
	/// bucketed key would leave the buckets keyed on a stale frequency.
	pub fn set_freq(&mut self, key: HashedKey, freq: u32) {
		let Some(slot) = self.slot_of(key) else { return };
		self.slots[slot as usize].payload.freq = freq;
	}

	/// Sets a SLOW key's frequency and relinks it into that bucket.
	///
	/// Relinks even when `freq` is unchanged, which moves the key to the newest
	/// position within its bucket. That is deliberate and matches
	/// `FrequencyChain::move_to`'s unconditional remove-then-insert: at the
	/// frequency cap a further access cannot raise the count, but it still
	/// refreshes the key's standing against its equally-frequent peers.
	pub fn slow_relink_at(&mut self, key: HashedKey, freq: u32) {
		let Some(slot) = self.slot_of(key) else { return };
		let payload = self.slots[slot as usize].payload;

		if payload.tier != Some(Tier::Slow) {
			return;
		}

		self.unlink(slot, payload.freq, Tier::Slow);
		self.slots[slot as usize].payload.freq = freq;
		self.link(slot, freq, Tier::Slow);
	}

	/// Appends `slot` at the newest end of the NATIVE run of bucket
	/// `(tier, freq)`. The caller has stamped it (or, on the recency face, does
	/// not use stamps at all).
	fn link(&mut self, slot: u32, freq: u32, tier: Tier) {
		let buckets = match tier {
			Tier::Fast => &mut self.fast_buckets,
			Tier::Slow => &mut self.slow_buckets,
		};

		self.slots[slot as usize].next = NIL;

		match buckets.get_mut(&freq) {
			// A bucket can exist with an EMPTY native run when only demoted
			// keys hold it, so the run's own tail decides, not the bucket.
			Some(bucket) if bucket.tail != NIL => {
				let old_tail = bucket.tail;
				bucket.tail = slot;
				self.slots[old_tail as usize].next = slot;
				self.slots[slot as usize].prev = old_tail;
			},

			Some(bucket) => {
				bucket.head = slot;
				bucket.tail = slot;
				self.slots[slot as usize].prev = NIL;
			},

			None => {
				buckets.insert(freq, Bucket::native(slot));
				self.slots[slot as usize].prev = NIL;
			},
		}
	}

	/// Appends `slot` at the newest end of the DEMOTED run of slow bucket
	/// `freq`, keeping its stamp. Only `demote_min_fast` calls this, and only
	/// with the head of the fast minimum bucket, which is what keeps the run
	/// stamp-ascending; the debug assertion checks exactly that.
	fn link_demoted(&mut self, slot: u32, freq: u32) {
		self.slots[slot as usize].next = NIL;

		match self.slow_buckets.get(&freq).map(|b| b.demoted_tail) {
			Some(old_tail) if old_tail != NIL => {
				debug_assert!(
					self.not_newer(
						stamp_of(&self.slots[old_tail as usize].payload),
						stamp_of(&self.slots[slot as usize].payload),
					),
					"demoted run of bucket {freq} would go out of stamp order",
				);

				self.slots[old_tail as usize].next = slot;
				self.slots[slot as usize].prev = old_tail;
				self.slow_buckets.get_mut(&freq).expect("just read").demoted_tail = slot;
			},

			Some(_) => {
				let bucket = self.slow_buckets.get_mut(&freq).expect("just read");
				bucket.demoted_head = slot;
				bucket.demoted_tail = slot;
				self.slots[slot as usize].prev = NIL;
			},

			None => {
				self.slow_buckets.insert(freq, Bucket::demoted(slot));
				self.slots[slot as usize].prev = NIL;
			},
		}
	}

	/// Unlinks `slot` from whichever run of bucket `(tier, freq)` holds it.
	///
	/// No tag says which: a slot is in exactly one run, the runs are separately
	/// `NIL`-ended, and a slot is never `NIL`, so comparing it against all four
	/// endpoints fixes up exactly the ones it occupies.
	fn unlink(&mut self, slot: u32, freq: u32, tier: Tier) {
		let (prev, next) = {
			let e = &self.slots[slot as usize];
			(e.prev, e.next)
		};

		if prev != NIL { self.slots[prev as usize].next = next; }
		if next != NIL { self.slots[next as usize].prev = prev; }

		let buckets = match tier {
			Tier::Fast => &mut self.fast_buckets,
			Tier::Slow => &mut self.slow_buckets,
		};

		if let Some(bucket) = buckets.get_mut(&freq) {
			if bucket.head == slot { bucket.head = next; }
			if bucket.tail == slot { bucket.tail = prev; }
			if bucket.demoted_head == slot { bucket.demoted_head = next; }
			if bucket.demoted_tail == slot { bucket.demoted_tail = prev; }

			if bucket.is_empty() {
				buckets.remove(&freq);
			}
		}
	}
}

#[cfg(test)]
mod tests {
	use super::*;
	use crate::worker::policy::policy_stack::{
		arena_index::GOLDEN,
		compact_frequency_chain::CompactFrequencyChain,
	};

	/// The premise, in one assertion. The node carries the payload, so a
	/// tracked key costs 32 bytes of slab; the index carries no key, so it
	/// costs four bytes a bucket. Growth in either is paid on EVERY tracked
	/// object in both tiers.
	#[test]
	fn the_node_is_thirty_two_bytes() {
		assert_eq!(
			std::mem::size_of::<ArenaSlot<NodePayload>>(),
			32,
			"key 8 + prev 4 + next 4 + NodePayload 16 = 32",
		);
	}

	/// The index holds slot numbers and nothing else, so its whole cost is four
	/// bytes a bucket at half load. At a power-of-two population that is
	/// exactly the 8 B/object the design is built around, and it is the half of
	/// the 40 that `CompactFrequencyChain`'s 56-byte hashbrown index used to
	/// be.
	#[test]
	fn the_index_costs_eight_bytes_per_object_at_a_power_of_two_population() {
		for exponent in 8..14u32 {
			let n = 1u64 << exponent;
			let mut chain = ArenaFrequencyChain::default();

			for k in 0..n {
				chain.insert(k.wrapping_mul(GOLDEN), 100, 0, Tier::Fast);
			}

			let bytes = chain.index_capacity() * std::mem::size_of::<u32>();
			assert_eq!(
				bytes as u64 / n,
				8,
				"{n} keys sat in a {}-bucket table",
				chain.index_capacity(),
			);
		}
	}

	/// The bucket maps are the reason this structure exists rather than being
	/// an `ArenaQueueSet`, and the reason it can afford them is that they are
	/// O(DISTINCT FREQUENCIES) and not O(objects). Asserted, because if that
	/// ever stopped being true the measured 40 B/object would drift with n and
	/// nothing else here would notice.
	#[test]
	fn the_bucket_maps_are_sized_by_distinct_frequency_not_by_object_count() {
		let mut chain = ArenaFrequencyChain::default();

		for key in 0..10_000u64 {
			chain.insert(key, 100, 0, Tier::Fast);
			// three distinct counts across ten thousand keys
			for _ in 0..(key % 3) { chain.bump(key); }
		}

		assert_eq!(chain.len(), 10_000);
		assert_eq!(chain.fast_buckets.len(), 3, "one bucket per distinct frequency");
		assert!(chain.slow_buckets.is_empty());
	}

	#[test]
	fn least_frequently_used_comes_out_first() {
		let mut c = ArenaFrequencyChain::default();
		for key in 1..=3u64 { c.insert(key, 100, 24, Tier::Fast); }

		// key 1 accessed twice, key 2 once, key 3 not at all
		c.bump(1); c.bump(1); c.bump(2);

		assert_eq!(c.min_key(Tier::Fast), Some(3), "key 3 is the least frequently used");

		c.remove(3);
		assert_eq!(c.min_key(Tier::Fast), Some(2));

		c.remove(2);
		assert_eq!(c.min_key(Tier::Fast), Some(1));
	}

	#[test]
	fn keys_at_the_same_frequency_come_out_in_insertion_order() {
		let mut c = ArenaFrequencyChain::default();
		for key in 1..=3u64 { c.insert(key, 10, 0, Tier::Fast); }

		assert_eq!(c.min_key(Tier::Fast), Some(1), "all at frequency 1, so oldest first");
		c.remove(1);
		assert_eq!(c.min_key(Tier::Fast), Some(2));
	}

	#[test]
	fn bump_moves_between_buckets_and_keeps_the_chain_intact() {
		let mut c = ArenaFrequencyChain::default();
		for key in 1..=5u64 { c.insert(key, 10, 0, Tier::Fast); }

		assert_eq!(c.bump(3), 2);
		assert_eq!(c.get(3).unwrap().freq, 2);
		assert_eq!(c.len(), 5, "bumping must not lose or duplicate a key");

		// everything still reachable, and 3 is no longer the minimum
		for key in 1..=5u64 { assert!(c.contains(key)); }
		assert_ne!(c.min_key(Tier::Fast), Some(3));
	}

	#[test]
	fn freed_slots_are_reused_so_the_slab_does_not_grow_forever() {
		let mut c = ArenaFrequencyChain::default();
		for key in 1..=100u64 { c.insert(key, 10, 0, Tier::Fast); }
		for key in 1..=100u64 { c.remove(key); }

		let before = c.slots.len();
		for key in 101..=200u64 { c.insert(key, 10, 0, Tier::Fast); }

		assert_eq!(
			c.slots.len(), before,
			"a hundred inserts after a hundred removes must reuse the slab",
		);
		assert_eq!(c.len(), 100);
	}

	#[test]
	fn removing_from_the_middle_of_a_bucket_relinks_neighbours() {
		let mut c = ArenaFrequencyChain::default();
		for key in 1..=5u64 { c.insert(key, 10, 0, Tier::Fast); }

		c.remove(3);

		assert_eq!(c.len(), 4);
		let mut seen = Vec::new();
		while let Some(k) = c.min_key(Tier::Fast) { seen.push(k); c.remove(k); }

		assert_eq!(seen, vec![1, 2, 4, 5], "the chain must survive a middle removal");
	}

	/// A tier move is the *only* thing that happens on promotion or demotion.
	///
	/// `FrequencyChain` has to `remove` from one chain and `insert_at` into the
	/// other, carrying the count across by hand. Here the entry never moves in
	/// the slab, so its frequency, size and links are all preserved by
	/// construction -- there is no count to carry and nothing to get wrong.
	#[test]
	fn promotion_is_a_tier_move_and_nothing_else() {
		let mut c = ArenaFrequencyChain::default();
		c.insert(1, 100, 24, Tier::Fast);
		c.insert(2, 200, 24, Tier::Slow);
		for _ in 0..5 { c.bump(2); }

		assert_eq!(c.min_count(Tier::Fast), Some(1));
		assert_eq!(c.min_count(Tier::Slow), Some(6));

		// key 2 strictly exceeds the fast minimum, so it promotes
		c.set_tier(2, Tier::Fast);

		assert_eq!(c.get(2).unwrap().freq, 6, "count survives the move");
		assert_eq!(c.get(2).unwrap().size, 200, "size survives the move");
		assert_eq!(c.min_count(Tier::Slow), None);
		assert_eq!(c.min_with_count(Tier::Fast), Some((1, 1)),
			"key 1 is still the fast minimum at count 1");
	}

	#[test]
	fn moving_a_key_between_tiers_preserves_its_frequency_and_position() {
		let mut c = ArenaFrequencyChain::default();
		for key in 1..=3u64 { c.insert(key, 100, 24, Tier::Fast); }
		c.bump(2); c.bump(2);

		assert_eq!(c.fast_len(), 3);
		assert_eq!(c.slow_len(), 0);
		assert_eq!(c.min_key(Tier::Fast), Some(1));

		c.set_tier(2, Tier::Slow);

		assert_eq!(c.fast_len(), 2);
		assert_eq!(c.slow_len(), 1);
		assert_eq!(c.get(2).unwrap().freq, 3, "frequency must survive the move");
		assert_eq!(c.get(2).unwrap().tier, Some(Tier::Slow));
		assert_eq!(c.min_key(Tier::Slow), Some(2));
		assert_eq!(c.min_key(Tier::Fast), Some(1), "the fast chain is intact");

		// and back again
		c.set_tier(2, Tier::Fast);
		assert_eq!(c.fast_len(), 3);
		assert_eq!(c.slow_len(), 0);
		assert_eq!(c.min_key(Tier::Slow), None);
	}

	#[test]
	fn tier_and_size_travel_with_the_entry() {
		let mut c = ArenaFrequencyChain::default();
		c.insert(9, 500, 24, Tier::Fast);

		assert_eq!(c.get(9).unwrap().tier, Some(Tier::Fast));
		assert_eq!(c.get(9).unwrap().migrating(), 476);

		c.set_tier(9, Tier::Slow);
		c.resize(9, 800, 88);

		assert_eq!(c.get(9).unwrap().tier, Some(Tier::Slow));
		assert_eq!(c.get(9).unwrap().migrating(), 712);
	}

	/// `phys` is the one node field this chain writes that neither LFU stack
	/// reads, and the node's contract is that it equals `tier` for every design
	/// but the lazy-copy one. A tier move that left it behind would make that
	/// contract false for a key that had ever been demoted or promoted.
	#[test]
	fn a_tier_move_carries_the_physical_tier_with_it() {
		let mut c = ArenaFrequencyChain::default();
		c.insert(1, 100, 0, Tier::Fast);
		assert_eq!(c.get(1).unwrap().phys, Some(Tier::Fast));

		c.set_tier(1, Tier::Slow);
		assert_eq!(c.get(1).unwrap().phys, Some(Tier::Slow));

		c.recency_push_front(2, 100, 0, 1);
		c.demote_recency_back();
		assert_eq!(c.get(2).unwrap().phys, Some(Tier::Slow));

		c.promote_to_recency_front(2, 1);
		assert_eq!(c.get(2).unwrap().phys, Some(Tier::Fast));
	}

	// ── the distinguished recency list ────────────────────────────────────

	#[test]
	fn the_recency_list_orders_by_recency_not_frequency() {
		let mut c = ArenaFrequencyChain::default();
		for key in 1..=3u64 { c.recency_push_front(key, 10, 0, 1); }

		// 3 was pushed last, so 1 is the LRU tail regardless of counts.
		assert_eq!(c.recency_back(), Some(1));
		assert_eq!(c.fast_len(), 3);
		assert_eq!(c.slow_len(), 0);

		c.set_freq(1, 9);
		assert_eq!(c.recency_back(), Some(1), "frequency must not reorder the recency list");

		c.recency_move_front(1);
		assert_eq!(c.recency_back(), Some(2), "a touch moves the key off the tail");
	}

	#[test]
	fn moving_the_recency_head_to_the_front_is_a_no_op() {
		let mut c = ArenaFrequencyChain::default();
		for key in 1..=3u64 { c.recency_push_front(key, 10, 0, 1); }

		c.recency_move_front(3);

		assert_eq!(c.recency_back(), Some(1));
		assert_eq!(c.fast_len(), 3);
	}

	#[test]
	fn demotion_carries_the_count_into_the_slow_buckets() {
		let mut c = ArenaFrequencyChain::default();
		c.recency_push_front(1, 100, 24, 1);
		c.recency_push_front(2, 200, 24, 1);
		c.set_freq(2, 7);
		c.recency_move_front(2);

		// 1 is the tail; demoting it must land it at ITS count, not 2's.
		let (key, entry) = c.demote_recency_back().unwrap();

		assert_eq!(key, 1);
		assert_eq!(entry.tier, Some(Tier::Slow));
		assert_eq!(entry.freq, 1);
		assert_eq!(entry.size, 100, "size survives the move");
		assert_eq!(c.min_with_count(Tier::Slow), Some((1, 1)));
		assert_eq!(c.fast_len(), 1);
		assert_eq!(c.slow_len(), 1);
		assert_eq!(c.recency_back(), Some(2), "the recency list closed over the gap");

		// and a hot key demotes into a HIGHER bucket than a cold one
		let (key, entry) = c.demote_recency_back().unwrap();
		assert_eq!(key, 2);
		assert_eq!(entry.freq, 7);
		assert_eq!(c.min_with_count(Tier::Slow), Some((1, 1)), "the cold key still ranks lowest");
		assert_eq!(c.recency_back(), None);
		assert_eq!(c.fast_len(), 0);
		assert_eq!(c.slow_len(), 2);
	}

	#[test]
	fn promotion_leaves_the_slow_buckets_and_enters_the_recency_head() {
		let mut c = ArenaFrequencyChain::default();
		c.recency_push_front(1, 10, 0, 1);
		c.recency_push_front(2, 10, 0, 1);
		c.demote_recency_back().unwrap(); // 1 -> slow

		let entry = c.promote_to_recency_front(1, 1).unwrap();

		assert_eq!(entry.tier, Some(Tier::Fast));
		assert_eq!(entry.freq, 1, "the counter resets on the way in");
		assert_eq!(c.min_key(Tier::Slow), None, "the slow bucket is gone");
		assert_eq!(c.recency_back(), Some(2), "1 entered at the head, so 2 is now the tail");
		assert_eq!(c.fast_len(), 2);
		assert_eq!(c.slow_len(), 0);

		// promoting something that is not slow is a no-op
		assert_eq!(c.promote_to_recency_front(1, 1), None);
		assert_eq!(c.promote_to_recency_front(999, 1), None);
	}

	#[test]
	fn relinking_a_slow_key_at_an_unchanged_count_still_refreshes_it() {
		let mut c = ArenaFrequencyChain::default();
		for key in 1..=3u64 { c.recency_push_front(key, 10, 0, 4); }
		for _ in 0..3 { c.demote_recency_back().unwrap(); }

		// all three sit in bucket 4, oldest-demoted first
		assert_eq!(c.min_with_count(Tier::Slow), Some((1, 4)));

		c.slow_relink_at(1, 4);

		assert_eq!(
			c.min_key(Tier::Slow), Some(2),
			"an unchanged relink must still move the key to the back of its bucket",
		);

		c.slow_relink_at(2, 9);
		assert_eq!(c.get(2).unwrap().freq, 9);
		assert_eq!(c.min_key(Tier::Slow), Some(3), "2 left the minimum bucket");
	}

	#[test]
	fn the_recency_list_and_the_slow_buckets_share_one_slab_and_one_index() {
		let mut c = ArenaFrequencyChain::default();
		for key in 1..=100u64 { c.recency_push_front(key, 10, 0, 1); }
		for _ in 0..50 { c.demote_recency_back().unwrap(); }

		assert_eq!(c.len(), 100, "one index, so one count");
		assert_eq!(c.fast_len() + c.slow_len(), 100);
		assert_eq!(c.slots.len(), 100, "one slab, one slot per key");

		// every key is reachable through the single index
		for key in 1..=100u64 { assert!(c.contains(key)); }

		// removing through the right door frees the slot for reuse
		let before = c.slots.len();
		for key in 1..=50u64 { c.remove(key); }             // slow half
		for key in 51..=100u64 { c.recency_remove(key); }   // fast half

		assert_eq!(c.len(), 0);
		assert_eq!(c.fast_len(), 0);
		assert_eq!(c.slow_len(), 0);
		assert_eq!(c.recency_back(), None);

		for key in 201..=300u64 { c.recency_push_front(key, 10, 0, 1); }
		assert_eq!(c.slots.len(), before, "the freed slots must be reused");
	}

	#[test]
	fn removing_from_the_middle_of_the_recency_list_relinks_neighbours() {
		let mut c = ArenaFrequencyChain::default();
		for key in 1..=5u64 { c.recency_push_front(key, 10, 0, 1); }

		c.recency_remove(3);

		let mut seen = Vec::new();
		while let Some(k) = c.recency_back() { seen.push(k); c.recency_remove(k); }

		assert_eq!(seen, vec![1, 2, 4, 5], "the recency list must survive a middle removal");
	}

	#[test]
	fn clear_resets_the_recency_list_too() {
		let mut c = ArenaFrequencyChain::default();
		for key in 1..=5u64 { c.recency_push_front(key, 10, 0, 1); }
		c.demote_recency_back().unwrap();

		c.clear();

		assert_eq!(c.len(), 0);
		assert_eq!(c.recency_back(), None);

		// and it is usable again afterwards
		c.recency_push_front(9, 10, 0, 1);
		assert_eq!(c.recency_back(), Some(9));
		assert_eq!(c.fast_len(), 1);
	}

	/// The recency list must be invisible to the frequency-bucket stacks: a
	/// chain driven only through `insert`/`bump`/`set_tier`/`remove` -- which
	/// is exactly what `LfuCompactHybridStack` does -- must leave it empty.
	#[test]
	fn the_frequency_only_path_never_touches_the_recency_list() {
		let mut c = ArenaFrequencyChain::default();
		for key in 1..=10u64 { c.insert(key, 10, 0, Tier::Fast); }
		for key in 1..=5u64 { c.bump(key); }
		c.set_tier(3, Tier::Slow);
		c.remove(7);

		assert_eq!(c.recency_back(), None, "no recency link may exist on the LFU path");
		assert_eq!(c.recency_head, NIL);
		assert_eq!(c.recency_tail, NIL);
		assert_eq!(c.fast_len() + c.slow_len(), 9);
	}

	// ── tie order across a tier move ──────────────────────────────────────

	/// Every key left in a slow bucket, in the order `min_key` serves them.
	fn drain_slow(c: &mut ArenaFrequencyChain) -> Vec<HashedKey> {
		let mut order = Vec::new();
		while let Some(k) = c.min_key(Tier::Slow) { order.push(k); c.remove(k); }
		order
	}

	/// The defect this fixes, in its smallest form: a key demoted into a slow
	/// bucket used to be RE-APPENDED at the newest end, so it ranked behind
	/// slow keys that reached the same count after it did.
	#[test]
	fn a_demoted_key_keeps_its_place_among_equal_counts() {
		let mut c = ArenaFrequencyChain::default();
		c.insert(1, 10, 0, Tier::Fast);
		c.insert(2, 10, 0, Tier::Slow);
		c.insert(3, 10, 0, Tier::Slow);

		let (key, entry) = c.demote_min_fast().unwrap();
		assert_eq!((key, entry.freq, entry.tier, entry.phys), (1, 1, Some(Tier::Slow), Some(Tier::Slow)));
		assert_eq!((c.fast_len(), c.slow_len()), (0, 3));

		assert_eq!(
			drain_slow(&mut c), vec![1, 2, 3],
			"1 reached count 1 before 2 and 3 did, so it leaves first -- a \
			 re-append would have made it last",
		);
	}

	/// ...and it is an ORDER, not "demoted keys first": a key that reached its
	/// count after some slow keys did waits behind them.
	#[test]
	fn a_demoted_key_newer_than_the_slow_residents_waits_behind_them() {
		let mut c = ArenaFrequencyChain::default();
		c.insert(1, 10, 0, Tier::Slow);
		c.insert(2, 10, 0, Tier::Fast);
		c.insert(3, 10, 0, Tier::Fast);
		c.insert(4, 10, 0, Tier::Slow);

		assert_eq!(c.demote_min_fast().map(|(k, _)| k), Some(2));
		assert_eq!(c.demote_min_fast().map(|(k, _)| k), Some(3));
		assert_eq!(c.demote_min_fast(), None, "the fast tier is empty");

		assert_eq!(drain_slow(&mut c), vec![1, 2, 3, 4]);
	}

	/// The promotion path still re-appends, and that is correct: it always
	/// follows a `bump` of the same key, so the key IS the newest at its count.
	#[test]
	fn a_promotion_after_a_bump_is_the_newest_key_at_its_count() {
		let mut c = ArenaFrequencyChain::default();
		c.insert(1, 10, 0, Tier::Fast);
		c.insert(2, 10, 0, Tier::Slow);
		c.insert(3, 10, 0, Tier::Fast);

		c.bump(1);                      // fast, 1 -> 2
		c.bump(2);                      // slow, 1 -> 2 ...
		c.set_tier(2, Tier::Fast);      // ... and promoted, as maybe_promote does
		c.bump(3);                      // fast, 1 -> 2

		let mut order = Vec::new();
		while let Some((k, _)) = c.demote_min_fast() { order.push(k); }
		assert_eq!(order, vec![1, 2, 3], "fast bucket 2 in the order its keys reached it");
		assert_eq!(drain_slow(&mut c), vec![1, 2, 3], "and demoted in that order");
	}

	/// The stamp wraps at 2^40; the comparison is by AGE, so a wrap between two
	/// stamps does not reorder them.
	#[test]
	fn the_order_survives_the_clock_wrapping() {
		let mut c = ArenaFrequencyChain { clock: STAMP_MASK - 1, ..Default::default() };

		c.insert(1, 10, 0, Tier::Slow);   // stamp 2^40 - 1
		c.insert(2, 10, 0, Tier::Fast);   // stamp 0
		c.insert(3, 10, 0, Tier::Fast);   // stamp 1
		c.insert(4, 10, 0, Tier::Slow);   // stamp 2

		assert_eq!(stamp_of(&c.get(1).unwrap()), STAMP_MASK);
		assert_eq!(stamp_of(&c.get(4).unwrap()), 2, "the clock did wrap");

		c.demote_min_fast();
		c.demote_min_fast();

		assert_eq!(drain_slow(&mut c), vec![1, 2, 3, 4]);
	}

	/// The stamp's high byte is `queue`, and the clock carries into it instead
	/// of wrapping at 2^32 -- which is the whole point of borrowing it.
	#[test]
	fn the_stamp_carries_past_thirty_two_bits_into_the_queue_byte() {
		let mut c = ArenaFrequencyChain { clock: u32::MAX as u64 - 1, ..Default::default() };

		c.insert(1, 10, 0, Tier::Fast);   // stamp 2^32 - 1
		c.insert(2, 10, 0, Tier::Fast);   // stamp 2^32

		let (one, two) = (c.get(1).unwrap(), c.get(2).unwrap());
		assert_eq!((one.queue, one.ts), (0, u32::MAX));
		assert_eq!((two.queue, two.ts), (1, 0), "bit 32 of the stamp is bit 0 of queue");

		// Nothing else about the node moved: the stamp is invisible to every
		// field a stack reads.
		assert_eq!((two.freq, two.size, two.tier, two.phys), (1, 10, Some(Tier::Fast), Some(Tier::Fast)));

		c.bump(1);                        // stamp 2^32 + 1, restamped over the old high byte
		assert_eq!(stamp_of(&c.get(1).unwrap()), (1 << 32) + 1);
	}

	/// What the 40-bit window buys over the 32-bit one `ts` alone would give.
	///
	/// Key 1 is demoted with a stamp taken 2^32 + 49 ticks ago -- the clock is
	/// moved directly, standing in for four billion touches of other keys. With
	/// `u32` ages it would alias to 49 ticks old and rank as NEWER than key 2,
	/// which really is younger (2^32 - 50 ticks); with 40 bits both ages are
	/// exact.
	#[test]
	fn an_age_past_two_to_the_thirty_two_still_orders_exactly() {
		let mut c = ArenaFrequencyChain::default();

		c.insert(1, 10, 0, Tier::Fast);   // stamp 1
		c.clock = 99;
		c.insert(2, 10, 0, Tier::Slow);   // stamp 100
		c.clock = (1 << 32) + 50;

		assert_eq!(c.demote_min_fast().map(|(k, _)| k), Some(1));

		let (age_1, age_2) = (c.age(1), c.age(100));
		assert!(age_1 > u32::MAX as u64 && age_2 < u32::MAX as u64);
		assert!(
			(age_1 as u32) < age_2 as u32,
			"the fixture no longer aliases at 32 bits, so it tests nothing",
		);

		assert_eq!(drain_slow(&mut c), vec![1, 2]);
	}

	#[test]
	fn unlinking_from_either_run_keeps_both_runs_intact() {
		let mut c = ArenaFrequencyChain::default();
		c.insert(1, 10, 0, Tier::Slow);
		c.insert(2, 10, 0, Tier::Fast);
		c.insert(3, 10, 0, Tier::Fast);
		c.insert(4, 10, 0, Tier::Slow);
		c.insert(5, 10, 0, Tier::Slow);
		c.insert(6, 10, 0, Tier::Fast);
		for _ in 0..3 { c.demote_min_fast().unwrap(); }

		// native run [1, 4, 5], demoted run [2, 3, 6]
		c.remove(3);                       // middle of the demoted run
		c.remove(1);                       // head of the native run
		c.remove(6);                       // tail of the demoted run
		assert_eq!(c.min_with_count(Tier::Slow), Some((2, 1)));

		c.bump(2);                         // leaves the demoted run for bucket 2
		assert_eq!(c.min_with_count(Tier::Slow), Some((4, 1)));
		assert_eq!(drain_slow(&mut c), vec![4, 5, 2]);
		assert!(c.slow_buckets.is_empty(), "no bucket survives its last key");
	}

	/// Random streams over the contract `LfuCompactHybridStack` uses -- insert,
	/// bump, promote (bump then `set_tier(Fast)`), `demote_min_fast`, remove --
	/// against an oracle that ranks by (count, order of reaching it), which is
	/// upstream `LfuStack`'s rule within a tier. Three times: from a zero
	/// clock, from one that carries past 2^32 into the stamp's high byte a few
	/// thousand steps in, and from one that wraps at 2^40 as soon.
	#[test]
	fn every_tier_minimum_matches_an_oracle_ranked_by_when_the_count_was_reached() {
		use std::collections::{HashMap, hash_map::Entry};

		for start in [0, u32::MAX as u64 - 5_000, STAMP_MASK - 5_000] {
			let mut c = ArenaFrequencyChain { clock: start, ..Default::default() };

			// key -> (count, tier, sequence number of reaching that count)
			let mut oracle: HashMap<HashedKey, (u32, Tier, u64)> = HashMap::new();
			let mut seq = 0u64;
			let mut next = stream(0xD1B5_4A32_D192_ED03 ^ start);

			let oracle_min = |o: &HashMap<HashedKey, (u32, Tier, u64)>, tier: Tier| -> Option<(HashedKey, u32)> {
				o.iter()
					.filter(|entry| (entry.1).1 == tier)
					.min_by_key(|entry| ((entry.1).0, (entry.1).2))
					.map(|(&k, &(f, _, _))| (k, f))
			};

			for step in 0..40_000u32 {
				let key = next() % KEYS;

				match next() % 8 {
					0 | 1 => if let Entry::Vacant(vacant) = oracle.entry(key) {
						let tier = if next().is_multiple_of(2) { Tier::Fast } else { Tier::Slow };
						c.insert(key, 64, 0, tier);
						seq += 1;
						vacant.insert((1, tier, seq));
					},

					2 | 3 => if let Some(entry) = oracle.get_mut(&key) {
						c.bump(key);
						seq += 1;
						*entry = (entry.0 + 1, entry.1, seq);
					},

					4 => if let Some(entry) = oracle.get_mut(&key) && entry.1 == Tier::Slow {
						c.bump(key);
						c.set_tier(key, Tier::Fast);
						seq += 1;
						*entry = (entry.0 + 1, Tier::Fast, seq);
					},

					5 | 6 => {
						let want = oracle_min(&oracle, Tier::Fast);
						assert_eq!(
							c.demote_min_fast().map(|(k, e)| (k, e.freq)), want,
							"demotion picked a different key at step {step}",
						);
						if let Some((k, _)) = want { oracle.get_mut(&k).unwrap().1 = Tier::Slow; }
					},

					_ => {
						assert_eq!(c.remove(key).is_some(), oracle.remove(&key).is_some());
					},
				}

				for tier in [Tier::Fast, Tier::Slow] {
					assert_eq!(
						c.min_with_count(tier), oracle_min(&oracle, tier),
						"{tier:?} minimum diverged from the oracle at step {step} (clock start {start})",
					);
				}

				// The cross-tier minimum `lfu-global-compact-hybrid` evicts:
				// one ranking over both tiers, which is only meaningful because
				// a demotion keeps the stamp.
				let global = oracle
					.iter()
					.min_by_key(|entry| ((entry.1).0, (entry.1).2))
					.map(|(&k, &(f, t, _))| (k, f, t));

				assert_eq!(
					c.min_over_both_tiers(), global,
					"the cross-tier minimum diverged from the oracle at step {step} (clock start {start})",
				);
			}
		}
	}

	// ── the differential tests ────────────────────────────────────────────
	//
	// The claim this whole module rests on is that it is a REPRESENTATION
	// change and not an algorithm change. These two drive it and the chain it
	// replaces through the same pseudo-random operation stream and compare
	// every observable at every step. `CompactFrequencyChain` is kept in the
	// tree, test-gated and with no production caller, precisely so they can
	// exist.
	//
	// There are two of them because the chain has two FACES with two different
	// contracts, and the stacks use one each. `LfuCompactHybridStack` buckets
	// every key by frequency and never calls a `recency_*` method;
	// `LruLfuCompactHybridStack` keeps its fast tier in the recency list and
	// its slow tier in the buckets, and routes every operation by the key's
	// current tier. Driving one structure through both contracts at once is not
	// a harder test, it is an undefined one: `recency_move_front` on a bucketed
	// key splices it out of its bucket, and both implementations then corrupt
	// the same bucket in the same way. The first draft of this test did exactly
	// that and panicked identically in either -- which measures nothing.

	/// Every observable the two structures share, compared.
	fn agree(
		arena: &ArenaFrequencyChain,
		compact: &CompactFrequencyChain,
		step: u32,
		keys: &[HashedKey],
		face: &str,
	) {
		assert_eq!(arena.len(), compact.len(), "{face}: len diverged at {step}");
		assert_eq!(
			arena.fast_len(), compact.fast_len(),
			"{face}: fast_len diverged at {step}",
		);
		assert_eq!(
			arena.slow_len(), compact.slow_len(),
			"{face}: slow_len diverged at {step}",
		);
		assert_eq!(
			arena.recency_back(), compact.recency_back(),
			"{face}: recency tail diverged at {step}",
		);

		for tier in [Tier::Fast, Tier::Slow] {
			assert_eq!(
				arena.min_key(tier), compact.min_key(tier),
				"{face}: {tier:?} minimum key diverged at {step}",
			);
			assert_eq!(
				arena.min_count(tier), compact.min_count(tier),
				"{face}: {tier:?} minimum count diverged at {step}",
			);
			assert_eq!(
				arena.min_with_count(tier), compact.min_with_count(tier),
				"{face}: {tier:?} minimum pair diverged at {step}",
			);
		}

		for &k in keys {
			assert_eq!(
				arena.contains(k), compact.contains(k),
				"{face}: membership of {k} diverged at {step}",
			);

			match (arena.get(k), compact.get(k)) {
				(None, None) => {},

				(Some(a), Some(b)) => {
					assert_eq!(a.freq, b.freq, "{face}: freq of {k} diverged at {step}");
					assert_eq!(a.size, b.size, "{face}: size of {k} diverged at {step}");
					assert_eq!(
						a.tier, Some(b.tier),
						"{face}: tier of {k} diverged at {step}",
					);
					assert_eq!(
						a.dram_resident, b.dram_resident,
						"{face}: resident of {k} diverged at {step}",
					);
					assert_eq!(
						a.migrating(), b.migrating(),
						"{face}: migrating bytes of {k} diverged at {step}",
					);
				},

				_ => panic!("{face}: tracking of {k} diverged at {step}"),
			}
		}
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

	/// Small enough that collisions, re-insertions and removals of absent keys
	/// all happen constantly.
	const KEYS: u64 = 64;

	/// Sweeping all 64 keys on all 40,000 steps is 2.6M debug-mode comparisons
	/// for no extra coverage: a divergence the sweep catches and the touched-key
	/// check does not still had to be created by some step, and 200 operations
	/// is not long enough for one to be created and undone.
	fn sweep(step: u32, key: HashedKey) -> Vec<HashedKey> {
		if step % 200 == 0 { (0..KEYS).collect() } else { vec![key] }
	}

	/// The face `LfuCompactHybridStack` drives: every key in a frequency
	/// bucket, ranked by count, in one of two tiers. No recency list exists on
	/// this path and both structures must leave it empty.
	#[test]
	fn the_frequency_face_agrees_with_the_compact_chain_it_replaces() {
		let mut arena = ArenaFrequencyChain::default();
		let mut compact = CompactFrequencyChain::default();
		let mut next = stream(0x2545_F491_4F6C_DD1D);

		for step in 0..40_000u32 {
			let key = next() % KEYS;
			let size = 64 + (next() % 512) as ObjectSize;
			let resident = (next() % 32) as u8;
			let tier = if next() % 2 == 0 { Tier::Fast } else { Tier::Slow };

			match next() % 6 {
				0 | 1 => {
					arena.insert(key, size, resident, tier);
					compact.insert(key, size, resident, tier);
				},

				2 | 3 => {
					assert_eq!(
						arena.bump(key), compact.bump(key),
						"bump returned a different count at {step}",
					);
				},

				4 => {
					arena.set_tier(key, tier);
					compact.set_tier(key, tier);
				},

				5 if next() % 3 == 0 => {
					arena.resize(key, size, resident);
					compact.resize(key, size, resident);
				},

				_ => {
					assert_eq!(
						arena.remove(key).is_some(),
						compact.remove(key).is_some(),
						"remove disagreed on membership at {step}",
					);
				},
			}

			agree(&arena, &compact, step, &sweep(step, key), "frequency");
			assert_eq!(arena.recency_back(), None, "the LFU path grew a recency list");
		}

		arena.clear();
		compact.clear();
		agree(&arena, &compact, u32::MAX, &(0..KEYS).collect::<Vec<_>>(), "frequency");
	}

	/// The face `LruLfuCompactHybridStack` drives: a recency-ordered fast tier
	/// and a frequency-ordered slow tier over the same slab, with every
	/// operation routed by the key's current tier exactly as that stack routes
	/// it.
	///
	/// Routing reads the ARENA's answer, which is safe because the tier of
	/// every key is compared against the compact chain's on the step before: a
	/// disagreement fails there rather than being laundered into a divergent
	/// operation stream here.
	#[test]
	fn the_recency_face_agrees_with_the_compact_chain_it_replaces() {
		let mut arena = ArenaFrequencyChain::default();
		let mut compact = CompactFrequencyChain::default();
		let mut next = stream(0x9E37_79B9_7F4A_7C15);

		for step in 0..40_000u32 {
			let key = next() % KEYS;
			let size = 64 + (next() % 512) as ObjectSize;
			let resident = (next() % 32) as u8;
			let freq = 1 + (next() % 8) as u32;

			let tier = arena.get(key).and_then(|entry| entry.tier);
			assert_eq!(
				tier, compact.get(key).map(|entry| entry.tier),
				"tier of {key} diverged before step {step} could route on it",
			);

			match next() % 8 {
				0 | 1 => {
					arena.recency_push_front(key, size, resident, freq);
					compact.recency_push_front(key, size, resident, freq);
				},

				2 if tier == Some(Tier::Fast) => {
					arena.recency_move_front(key);
					compact.recency_move_front(key);
					arena.set_freq(key, freq);
					compact.set_freq(key, freq);
				},

				3 => {
					assert_eq!(
						arena.demote_recency_back().map(|(k, e)| (k, e.freq)),
						compact.demote_recency_back().map(|(k, e)| (k, e.freq)),
						"demotion picked a different key or count at {step}",
					);
				},

				4 if tier == Some(Tier::Slow) => {
					assert_eq!(
						arena.promote_to_recency_front(key, freq).map(|e| e.freq),
						compact.promote_to_recency_front(key, freq).map(|e| e.freq),
						"promotion disagreed at {step}",
					);
				},

				5 if tier == Some(Tier::Slow) => {
					arena.slow_relink_at(key, freq);
					compact.slow_relink_at(key, freq);
				},

				6 => {
					arena.resize(key, size, resident);
					compact.resize(key, size, resident);
				},

				// Which door a key leaves by is its tier, exactly as
				// `LruLfuCompactHybridStack::remove` decides it.
				7 => match tier {
					Some(Tier::Fast) => assert_eq!(
						arena.recency_remove(key).is_some(),
						compact.recency_remove(key).is_some(),
						"recency removal disagreed at {step}",
					),

					Some(Tier::Slow) => assert_eq!(
						arena.remove(key).is_some(),
						compact.remove(key).is_some(),
						"slow removal disagreed at {step}",
					),

					None => {},
				},

				// The guarded arms above fall through when their tier
				// condition does not hold, which is itself worth exercising:
				// an untracked key must be a no-op in both. `set_freq` is for
				// recency-list keys only -- on a bucketed key it would leave
				// the bucket keyed on a stale count -- so the fallthrough only
				// takes it for a fast key.
				_ => {
					if tier == Some(Tier::Fast) {
						arena.set_freq(key, freq);
						compact.set_freq(key, freq);
					}
				},
			}

			agree(&arena, &compact, step, &sweep(step, key), "recency");
		}

		arena.clear();
		compact.clear();
		agree(&arena, &compact, u32::MAX, &(0..KEYS).collect::<Vec<_>>(), "recency");
	}
}
