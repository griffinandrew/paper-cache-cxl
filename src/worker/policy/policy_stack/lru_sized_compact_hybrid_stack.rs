/*
 * Copyright (c) Kia Shakiba
 *
 * This source code is licensed under the GNU AGPLv3 license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! Slab-backed size-split LRU hybrid: behaviourally identical to
//! [`LruSizedHybridStack`], with one structure where that has five.
//!
//! `LruSizedHybridStack` keeps FOUR `HashList`s -- `small_fast`, `large_fast`,
//! `small_slow`, `large_slow`, each owning its own key-to-node index -- plus a
//! separate `entries` map holding `queue`, `size` and `dram_resident`. A key is
//! resident in exactly one of those four lists at a time, which is precisely
//! the condition [`CompactQueueSet`] exists for: four intrusive orders over one
//! slab of 16-byte link-only slots, with ONE index whose value carries the
//! payload. `MAX_QUEUES` is 4, and this is the design that uses all four.
//!
//! The payload lives in the INDEX MAP'S VALUE, not in the slab slot, so a
//! metadata read is one probe with the payload already in the bucket rather
//! than a probe plus a dereference into the slab. See `compact_queue_set`'s
//! module doc for the measurements behind that choice.
//!
//! ## What the four queues buy, and why there is no boundary cursor
//!
//! `LruHybridStack`/`LruCompactHybridStack` track fast and slow membership as
//! one recency order plus a `fast_boundary` cursor marking where the fast
//! prefix ends. That trick works because there is exactly one fast segment
//! feeding exactly one slow segment. Here there are two independent fast
//! sources each feeding its own independent slow destination, so every list is
//! fully homogeneous and each list's own tail is directly its own
//! demotion/eviction candidate. There is no cursor anywhere -- which also means
//! this conversion, unlike the plain LRU one, has no boundary maintenance to
//! port.
//!
//! ## What the queue tag replaces
//!
//! The baseline's `SizedEntry` carries a 4-variant `SizeQueue` tag and NO
//! `Tier` field: the tier is derivable from which queue a key is in
//! (`Small/LargeFast` -> `Tier::Fast`, `Small/LargeSlow` -> `Tier::Slow`), so
//! `tier_of` reads the tag. That is unchanged here; the tag doubles as the
//! `CompactQueueSet` slot number through `SizeQueue::slot`.
//!
//! The baseline also keeps four `usize` object counters alongside the four
//! lists. Those are dropped: `CompactQueueSet::queue_len` is the same number by
//! construction, because a key is in exactly one queue and every push/pop the
//! baseline pairs with a counter update is the same push/pop here. The four
//! BYTE counters are kept -- they are sums, not cardinalities.
//!
//! ## Everything else is `LruSizedHybridStack`, verbatim
//!
//! Size classification against `ObjectSize` (`classify`), admission/promotion/
//! reclassification funnelled through `touch_fast`, per-segment high/low
//! watermark settling against a capacity net of that segment's PROPORTIONAL
//! share of the shared metadata reservation (`reserved_shares`), slow-tier
//! eviction preferring whichever slow list holds more objects, and the
//! ratio-ranked fast fallback for when nothing has ever been demoted. The
//! `fidelity_tests` module below replays both stacks against each other and
//! asserts they are indistinguishable.

use crate::{
	object::ObjectSize,
	worker::policy::policy_stack::{
		compact_queue_set::CompactQueueSet, narrow_resident, watermarks, CacheSize,
		HashedKey, PolicyStack, Tier,
	},
	PaperPolicy,
};

/// The four recency orders, in the shared queue set's slots 0..=3.
const Q_SMALL_FAST: usize = 0;
const Q_LARGE_FAST: usize = 1;
const Q_SMALL_SLOW: usize = 2;
const Q_LARGE_SLOW: usize = 3;

/// Which of the four queues a key is currently tracked in. Also the key's
/// tier: the two `*Fast` variants are `Tier::Fast`, the two `*Slow` variants
/// are `Tier::Slow`, so no separate tier field is stored.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum SizeQueue {
	SmallFast,
	LargeFast,
	SmallSlow,
	LargeSlow,
}

impl SizeQueue {
	/// The `CompactQueueSet` slot this queue occupies.
	#[inline]
	fn slot(self) -> usize {
		match self {
			SizeQueue::SmallFast => Q_SMALL_FAST,
			SizeQueue::LargeFast => Q_LARGE_FAST,
			SizeQueue::SmallSlow => Q_SMALL_SLOW,
			SizeQueue::LargeSlow => Q_LARGE_SLOW,
		}
	}

	#[inline]
	fn is_slow(self) -> bool {
		matches!(self, SizeQueue::SmallSlow | SizeQueue::LargeSlow)
	}
}

/// Per-key bookkeeping, carried in the index value.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct SizedPayload {
	queue: SizeQueue,

	/// The part of `size` that stays in DRAM in either tier; see `migrating`.
	dram_resident: u8,

	size: ObjectSize,
}

const _: () = assert!(
	std::mem::size_of::<SizedPayload>() == 8,
	"SizedPayload grew past 8 bytes",
);

impl SizedPayload {
	/// The bytes that actually move between tiers when this object migrates.
	///
	/// Deliberately distinct from `size` (`base_size`), which remains the input
	/// to `classify`: the small/large split is a property of the whole object
	/// as the cache accounts for it, not of its value alone. Only the byte
	/// counters use this.
	#[inline]
	fn migrating(&self) -> CacheSize {
		(self.size as CacheSize).saturating_sub(self.dram_resident as CacheSize)
	}
}

pub struct LruSizedCompactHybridStack {
	queues: CompactQueueSet<SizedPayload>,

	small_capacity: CacheSize,
	large_capacity: CacheSize,
	size_threshold: CacheSize,

	small_fast_used: CacheSize,
	large_fast_used: CacheSize,
	small_slow_used: CacheSize,
	large_slow_used: CacheSize,

	/// Approximate per-object DRAM cost of the shared structures, reserved
	/// proportionally between the two fast segments' capacities -- see
	/// `reserved_shares`. `0` unless set via `with_shared_overhead`.
	shared_overhead: CacheSize,

	/// (key, new tier) pairs recorded since the last `drain_tier_migrations`.
	migrations: Vec<(HashedKey, Tier)>,
}

impl LruSizedCompactHybridStack {
	pub fn new(
		small_capacity: CacheSize,
		large_capacity: CacheSize,
		size_threshold: CacheSize,
	) -> Self {
		LruSizedCompactHybridStack {
			queues: CompactQueueSet::default(),

			small_capacity,
			large_capacity,
			size_threshold,

			small_fast_used: 0,
			large_fast_used: 0,
			small_slow_used: 0,
			large_slow_used: 0,

			shared_overhead: 0,
			migrations: Vec::new(),
		}
	}

	/// Per-object DRAM reserved from the fast segments for shared metadata.
	///
	pub fn with_shared_overhead(mut self, overhead: CacheSize) -> Self {
		self.shared_overhead = overhead;


		self
	}

	/// The configured SMALL fast segment's byte budget.
	pub fn small_capacity(&self) -> CacheSize {
		self.small_capacity
	}

	/// The configured LARGE fast segment's byte budget.
	pub fn large_capacity(&self) -> CacheSize {
		self.large_capacity
	}

	/// The current small/large size-classification threshold.
	pub fn size_threshold(&self) -> CacheSize {
		self.size_threshold
	}

	/// The tier the given (currently tracked) key is in, or `None` if the key
	/// isn't tracked. Derived from the queue tag; there is no tier field.
	pub fn tier_of(&self, key: HashedKey) -> Option<Tier> {
		self.queue_of(key).map(|queue| match queue {
			SizeQueue::SmallFast | SizeQueue::LargeFast => Tier::Fast,
			SizeQueue::SmallSlow | SizeQueue::LargeSlow => Tier::Slow,
		})
	}

	/// Which of the four queues the given (currently tracked) key is in.
	fn queue_of(&self, key: HashedKey) -> Option<SizeQueue> {
		self.queues.payload(key).map(|p| p.queue)
	}

	/// `true` if `size` classifies as the SMALL segment (`size <
	/// size_threshold`), `false` for LARGE.
	fn classify(&self, size: ObjectSize) -> bool {
		(size as CacheSize) < self.size_threshold
	}

	/// Splits the total reserved shared-structure DRAM cost (`tracked object
	/// count x shared_overhead`, across all four queues -- shared metadata
	/// scales with everything tracked, not just one segment) proportionally
	/// between the two fast segments' capacities. `(0, 0)` if both capacities
	/// are zero (nothing to proportion against).
	fn reserved_shares(&self) -> (CacheSize, CacheSize) {
		let reserved = self.queues.len() as CacheSize * self.shared_overhead;
		let total_capacity = self.small_capacity + self.large_capacity;

		if total_capacity == 0 {
			return (0, 0);
		}

		let small_share =
			((reserved as u128 * self.small_capacity as u128) / total_capacity as u128) as CacheSize;
		let large_share = reserved.saturating_sub(small_share);

		(small_share, large_share)
	}

	fn effective_small(&self) -> CacheSize {
		self.small_capacity.saturating_sub(self.reserved_shares().0)
	}

	fn effective_large(&self) -> CacheSize {
		self.large_capacity.saturating_sub(self.reserved_shares().1)
	}

	/// Subtracts `size` from whichever byte counter `queue` owns. Saturating,
	/// matching the baseline's four `remove_from_*` helpers.
	fn sub_used(&mut self, queue: SizeQueue, size: CacheSize) {
		match queue {
			SizeQueue::SmallFast => {
				self.small_fast_used = self.small_fast_used.saturating_sub(size);
			},

			SizeQueue::LargeFast => {
				self.large_fast_used = self.large_fast_used.saturating_sub(size);
			},

			SizeQueue::SmallSlow => {
				self.small_slow_used = self.small_slow_used.saturating_sub(size);
			},

			SizeQueue::LargeSlow => {
				self.large_slow_used = self.large_slow_used.saturating_sub(size);
			},
		}
	}

	/// Adds `size` to whichever byte counter `queue` owns.
	fn add_used(&mut self, queue: SizeQueue, size: CacheSize) {
		match queue {
			SizeQueue::SmallFast => self.small_fast_used += size,
			SizeQueue::LargeFast => self.large_fast_used += size,
			SizeQueue::SmallSlow => self.small_slow_used += size,
			SizeQueue::LargeSlow => self.large_slow_used += size,
		}
	}

	/// Records a size change for an already-tracked key without altering its
	/// queue, adjusting whichever counter currently applies. `new_resident`
	/// refreshes the entry's DRAM-resident remainder: a re-set can add or drop
	/// a TTL, which changes it by the `Expiries` entry's cost. Without this the
	/// entry keeps its old remainder and every later migration moves the wrong
	/// number of bytes.
	fn resize_key(&mut self, key: HashedKey, new_size: ObjectSize, new_resident: u8) {
		let Some(payload) = self.queues.payload_mut(key) else { return };

		let old_migrating = payload.migrating();
		payload.size = new_size;
		payload.dram_resident = new_resident;
		let delta = payload.migrating() as i64 - old_migrating as i64;
		let queue = payload.queue;

		match queue {
			SizeQueue::SmallFast => {
				self.small_fast_used = (self.small_fast_used as i64 + delta).max(0) as CacheSize;
			},

			SizeQueue::LargeFast => {
				self.large_fast_used = (self.large_fast_used as i64 + delta).max(0) as CacheSize;
			},

			SizeQueue::SmallSlow => {
				self.small_slow_used = (self.small_slow_used as i64 + delta).max(0) as CacheSize;
			},

			SizeQueue::LargeSlow => {
				self.large_slow_used = (self.large_slow_used as i64 + delta).max(0) as CacheSize;
			},
		}
	}

	/// Faithful port of `LruSizedHybridStack::touch_fast`.
	///
	/// Moves an already-tracked key to the front of whichever fast segment its
	/// CURRENT size classifies as: a plain `move_front` when it is already in
	/// that segment, otherwise an unlink-and-relink out of whichever of the
	/// four queues it was in. Promotion from either slow queue and
	/// reclassification between the two fast queues are the same code path;
	/// only the former emits a migration, because a fast->fast move never
	/// crosses the `Tier` boundary.
	fn touch_fast(&mut self, key: HashedKey) {
		let Some(payload) = self.queues.payload(key) else { return };

		let target_small = self.classify(payload.size);
		let was_slow = payload.queue.is_slow();
		let migrating = payload.migrating();

		match (payload.queue, target_small) {
			(SizeQueue::SmallFast, true) => {
				self.queues.move_front(Q_SMALL_FAST, key);
				self.settle_small_fast();
				return;
			},

			(SizeQueue::LargeFast, false) => {
				self.queues.move_front(Q_LARGE_FAST, key);
				self.settle_large_fast();
				return;
			},

			_ => {},
		}

		let target_queue = if target_small { SizeQueue::SmallFast } else { SizeQueue::LargeFast };

		self.sub_used(payload.queue, migrating);
		self.queues.move_to_front_of(payload.queue.slot(), target_queue.slot(), key);
		self.add_used(target_queue, migrating);

		if let Some(slot) = self.queues.payload_mut(key) {
			slot.queue = target_queue;
		}

		if target_small {
			self.settle_small_fast();
		} else {
			self.settle_large_fast();
		}

		// Only a genuine slow->fast promotion needs a migration -- a fast<->fast
		// reclassification never crosses the Tier boundary (both segments are
		// physically TieredBuffer::Fast). Pushed after `settle_*` (which may
		// push demotions this same promotion triggered) and guarded on the key
		// still being in the target queue: an extremely tight target segment
		// can demote this same key straight back out within the settle call
		// above, in which case that call already pushed the correct final
		// entry.
		if was_slow && self.queues.payload(key).map(|p| p.queue) == Some(target_queue) {
			self.migrations.push((key, Tier::Fast));
		}
	}

	/// Demotes the SMALL fast queue's LRU tail(s) into `small_slow`, triggered
	/// only once `small_fast_used` crosses the shared HIGH watermark of its
	/// effective budget, then drained in one pass down to the shared LOW
	/// watermark of that same budget.
	///
	/// `effective_small()` -- the configured capacity minus this segment's
	/// proportional share of the reserved shared-structure overhead -- remains
	/// the budget in play, and is loop-invariant: `reserved_shares()` counts
	/// TRACKED entries, and a demotion only changes which queue an entry is in,
	/// never whether it is tracked (`CompactQueueSet::len` is the sum over all
	/// four queues, so a cross-queue move leaves it alone).
	fn settle_small_fast(&mut self) {
		let effective = self.effective_small();

		if self.small_fast_used <= watermarks::high_bytes(effective) {
			return;
		}

		let drain_target = watermarks::low_bytes(effective);

		while self.small_fast_used > drain_target {
			let Some(demote_key) = self.queues.back(Q_SMALL_FAST) else { break };
			let size = self.queues.payload(demote_key).map(|p| p.migrating()).unwrap_or(0);

			self.small_fast_used = self.small_fast_used.saturating_sub(size);
			self.queues.move_to_front_of(Q_SMALL_FAST, Q_SMALL_SLOW, demote_key);
			self.small_slow_used += size;

			if let Some(slot) = self.queues.payload_mut(demote_key) {
				slot.queue = SizeQueue::SmallSlow;
			}

			self.migrations.push((demote_key, Tier::Slow));
		}
	}

	/// LARGE-segment counterpart of `settle_small_fast`, demoting into
	/// `large_slow`. Same shared high/low watermark pair, taken against
	/// `effective_large()` instead.
	fn settle_large_fast(&mut self) {
		let effective = self.effective_large();

		if self.large_fast_used <= watermarks::high_bytes(effective) {
			return;
		}

		let drain_target = watermarks::low_bytes(effective);

		while self.large_fast_used > drain_target {
			let Some(demote_key) = self.queues.back(Q_LARGE_FAST) else { break };
			let size = self.queues.payload(demote_key).map(|p| p.migrating()).unwrap_or(0);

			self.large_fast_used = self.large_fast_used.saturating_sub(size);
			self.queues.move_to_front_of(Q_LARGE_FAST, Q_LARGE_SLOW, demote_key);
			self.large_slow_used += size;

			if let Some(slot) = self.queues.payload_mut(demote_key) {
				slot.queue = SizeQueue::LargeSlow;
			}

			self.migrations.push((demote_key, Tier::Slow));
		}
	}

	/// Last-resort eviction fallback, only reachable when both slow queues are
	/// empty (nothing has ever been demoted). Evicts from whichever fast
	/// segment is furthest over its own budget by ratio (`used / capacity`,
	/// treating a zero-capacity segment with any usage as infinitely over),
	/// ties going to small.
	fn evict_fast_fallback(&mut self) -> Option<HashedKey> {
		let small_count = self.queues.queue_len(Q_SMALL_FAST);
		let large_count = self.queues.queue_len(Q_LARGE_FAST);

		if small_count == 0 && large_count == 0 {
			return None;
		}

		let ratio = |used: CacheSize, capacity: CacheSize| -> f64 {
			if capacity == 0 {
				if used > 0 { f64::INFINITY } else { 0.0 }
			} else {
				used as f64 / capacity as f64
			}
		};

		let pick_small = if small_count == 0 {
			false
		} else if large_count == 0 {
			true
		} else {
			ratio(self.small_fast_used, self.small_capacity)
				>= ratio(self.large_fast_used, self.large_capacity)
		};

		if pick_small {
			let (key, payload) = self.queues.pop_back(Q_SMALL_FAST)?;
			self.small_fast_used = self.small_fast_used.saturating_sub(payload.migrating());
			Some(key)
		} else {
			let (key, payload) = self.queues.pop_back(Q_LARGE_FAST)?;
			self.large_fast_used = self.large_fast_used.saturating_sub(payload.migrating());
			Some(key)
		}
	}
}

impl PolicyStack for LruSizedCompactHybridStack {
	fn is_policy(&self, policy: &PaperPolicy) -> bool {
		matches!(policy, PaperPolicy::LruSizedCompactHybrid)
	}

	fn len(&self) -> usize {
		self.queues.len()
	}

	fn contains(&self, key: HashedKey) -> bool {
		self.queues.contains(key)
	}

	fn insert(&mut self, key: HashedKey, size: ObjectSize) {
		self.insert_resident(key, size, 0);
	}

	fn insert_resident(&mut self, key: HashedKey, size: ObjectSize, dram_resident: ObjectSize) {
		let dram_resident = narrow_resident(dram_resident);
		let migrating = (size as CacheSize).saturating_sub(dram_resident as CacheSize);

		if self.queues.contains(key) {
			// Existing key: track any size change, then treat as an access --
			// a `set()` always re-admits to fast, reclassifying between
			// segments if the new size crosses the threshold.
			self.resize_key(key, size, dram_resident);
			self.touch_fast(key);
			return;
		}

		if self.classify(size) {
			self.queues.push_front(
				Q_SMALL_FAST,
				key,
				SizedPayload { queue: SizeQueue::SmallFast, dram_resident, size },
			);
			self.small_fast_used += migrating;
			self.settle_small_fast();
		} else {
			self.queues.push_front(
				Q_LARGE_FAST,
				key,
				SizedPayload { queue: SizeQueue::LargeFast, dram_resident, size },
			);
			self.large_fast_used += migrating;
			self.settle_large_fast();
		}
	}

	fn update(&mut self, key: HashedKey) {
		if self.queues.contains(key) {
			self.touch_fast(key);
		}
	}

	fn remove(&mut self, key: HashedKey) {
		let Some(payload) = self.queues.payload(key) else { return };
		let queue = payload.queue;
		let size = payload.migrating();

		self.queues.remove(queue.slot(), key);
		self.sub_used(queue, size);
	}

	fn clear(&mut self) {
		self.queues.clear();

		self.small_fast_used = 0;
		self.large_fast_used = 0;
		self.small_slow_used = 0;
		self.large_slow_used = 0;

		self.migrations.clear();
	}

	fn evict_one(&mut self) -> Option<HashedKey> {
		let small_count = self.queues.queue_len(Q_SMALL_SLOW);
		let large_count = self.queues.queue_len(Q_LARGE_SLOW);

		if small_count == 0 && large_count == 0 {
			return self.evict_fast_fallback();
		}

		let pick_small = if small_count == 0 {
			false
		} else if large_count == 0 {
			true
		} else {
			small_count >= large_count
		};

		if pick_small {
			let (key, payload) = self.queues.pop_back(Q_SMALL_SLOW)?;
			self.small_slow_used = self.small_slow_used.saturating_sub(payload.migrating());
			Some(key)
		} else {
			let (key, payload) = self.queues.pop_back(Q_LARGE_SLOW)?;
			self.large_slow_used = self.large_slow_used.saturating_sub(payload.migrating());
			Some(key)
		}
	}

	/// Resizes the SMALL fast segment. The LARGE segment uses
	/// `resize_large_fast_tier` instead.
	fn resize_fast_tier(&mut self, size: CacheSize) {
		self.small_capacity = size;
		self.settle_small_fast();
	}

	fn resize_large_fast_tier(&mut self, size: CacheSize) {
		self.large_capacity = size;
		self.settle_large_fast();
	}

	fn resize_size_threshold(&mut self, size: CacheSize) {
		self.size_threshold = size;
	}

	fn drain_tier_migrations(&mut self) -> Vec<(HashedKey, Tier)> {
		std::mem::take(&mut self.migrations)
	}

	fn dram_reserved_bytes(&self) -> CacheSize {
		// The undivided total `reserved_shares` proportions between the two
		// fast segments; shared metadata scales with everything tracked.
		self.queues.len() as CacheSize * self.shared_overhead
	}

	fn fast_bytes_used(&self) -> CacheSize {
		self.small_fast_used + self.large_fast_used
	}

	fn slow_bytes_used(&self) -> CacheSize {
		self.small_slow_used + self.large_slow_used
	}

	fn fast_object_count(&self) -> usize {
		self.queues.queue_len(Q_SMALL_FAST) + self.queues.queue_len(Q_LARGE_FAST)
	}

	fn slow_object_count(&self) -> usize {
		self.queues.queue_len(Q_SMALL_SLOW) + self.queues.queue_len(Q_LARGE_SLOW)
	}

	fn small_fast_bytes_used(&self) -> CacheSize {
		self.small_fast_used
	}

	fn large_fast_bytes_used(&self) -> CacheSize {
		self.large_fast_used
	}

	fn small_fast_object_count(&self) -> usize {
		self.queues.queue_len(Q_SMALL_FAST)
	}

	fn large_fast_object_count(&self) -> usize {
		self.queues.queue_len(Q_LARGE_FAST)
	}

	fn small_slow_bytes_used(&self) -> CacheSize {
		self.small_slow_used
	}

	fn large_slow_bytes_used(&self) -> CacheSize {
		self.large_slow_used
	}

	fn small_slow_object_count(&self) -> usize {
		self.queues.queue_len(Q_SMALL_SLOW)
	}

	fn large_slow_object_count(&self) -> usize {
		self.queues.queue_len(Q_LARGE_SLOW)
	}
}


/// Fidelity against `LruSizedHybridStack`, which this stack is a compaction of.
///
/// The two must be indistinguishable: same queue for every key, same migration
/// sequence in the same order, same eviction order. A miss ratio matching on a
/// trace is necessary but not sufficient -- it would not catch a counter firing
/// on the wrong path, which is the class of defect that produced a doubled
/// demotion count on the LFU conversion.
#[cfg(all(test, feature = "lru_sized_hybrid_cache"))]
mod fidelity_tests {
	use super::*;
	use crate::worker::policy::policy_stack::lru_sized_hybrid_stack::LruSizedHybridStack;

	/// The eight size-split gauges plus the four aggregate ones, as one
	/// comparable tuple. `queue_of` is private to each stack's own module, so
	/// these public gauges are how queue membership is compared across the two.
	type Gauges = (CacheSize, CacheSize, CacheSize, CacheSize, usize, usize, usize, usize, CacheSize, CacheSize, usize, usize);

	fn gauges(stack: &dyn PolicyStack) -> Gauges {
		(
			stack.small_fast_bytes_used(),
			stack.large_fast_bytes_used(),
			stack.small_slow_bytes_used(),
			stack.large_slow_bytes_used(),
			stack.small_fast_object_count(),
			stack.large_fast_object_count(),
			stack.small_slow_object_count(),
			stack.large_slow_object_count(),
			stack.fast_bytes_used(),
			stack.slow_bytes_used(),
			stack.fast_object_count(),
			stack.slow_object_count(),
		)
	}

	/// 200 keys biased toward low ids, with sizes straddling the 1024-byte
	/// threshold the replays below use: enough pressure to exercise admission
	/// into both segments, promotion, demotion, and reclassification when a
	/// re-set changes a key's size class.
	fn skewed_ops() -> Vec<(HashedKey, ObjectSize)> {
		let mut ops = Vec::new();
		let mut x: u64 = 0x243F_6A88_85A3_08D3;

		for _ in 0..20_000 {
			x ^= x << 13;
			x ^= x >> 7;
			x ^= x << 17;
			let u = (x >> 11) as f64 / (1u64 << 53) as f64;
			let key = ((u * u * 200.0) as u64) + 1;
			// Size class is a function of the key MOST of the time, so a key
			// keeps its segment across repeats -- but one key in eight flips
			// class on a re-set, which is the reclassification path.
			let large = (key % 2 == 0) != (x % 8 == 0);
			// One large object in four sits EXACTLY on the 1_024-byte
			// threshold the replays use. `classify` decides that boundary
			// with `<`, not `<=`, so a size equal to the threshold is LARGE
			// -- and nothing else in this workload would notice if that
			// comparison were widened.
			let size: ObjectSize = match (large, x % 4 == 0) {
				(false, _) => 256,
				(true, true) => 1_024,
				(true, false) => 4_096,
			};
			ops.push((key, size));
		}

		ops
	}

	/// One op against both stacks. One key in three arrives as a `set()` rather
	/// than a read, so an existing key sometimes comes back with a size on the
	/// other side of the threshold. That is the only route into `resize_key` +
	/// fast<->fast reclassification, and a replay of pure reads would never
	/// reach it.
	fn apply(
		a: &mut LruSizedHybridStack,
		b: &mut LruSizedCompactHybridStack,
		key: HashedKey,
		size: ObjectSize,
	) {
		if key % 3 == 0 || !a.contains(key) {
			a.insert(key, size);
			b.insert(key, size);
		} else {
			a.update(key);
			b.update(key);
		}
	}

	fn replay(
		small: CacheSize,
		large: CacheSize,
		threshold: CacheSize,
		overhead: CacheSize,
		ops: &[(HashedKey, ObjectSize)],
	) -> (Vec<(HashedKey, Tier)>, Vec<(HashedKey, Tier)>, Vec<Option<Tier>>, Vec<Option<Tier>>, Gauges, Gauges) {
		let mut a = LruSizedHybridStack::new(small, large, threshold).with_shared_overhead(overhead);
		let mut b =
			LruSizedCompactHybridStack::new(small, large, threshold).with_shared_overhead(overhead);
		let (mut ma, mut mb) = (Vec::new(), Vec::new());

		for (k, size) in ops {
			apply(&mut a, &mut b, *k, *size);
			ma.extend(a.drain_tier_migrations());
			mb.extend(b.drain_tier_migrations());
		}

		let keys: Vec<HashedKey> = ops.iter().map(|(k, _)| *k).collect();
		let ta = keys.iter().map(|k| a.tier_of(*k)).collect();
		let tb = keys.iter().map(|k| b.tier_of(*k)).collect();
		(ma, mb, ta, tb, gauges(&a), gauges(&b))
	}

	#[test]
	fn matches_lru_sized_hybrid_migration_for_migration() {
		let ops = skewed_ops();

		for (small, large) in [(8_192u64, 8_192u64), (32_768, 8_192), (8_192, 131_072), (131_072, 131_072)] {
			for overhead in [0u64, 64, 224] {
				let (ma, mb, ta, tb, ga, gb) = replay(small, large, 1_024, overhead, &ops);
				assert_eq!(ta, tb, "final tiers diverge at {small}/{large} overhead {overhead}");
				assert_eq!(ma, mb, "migrations diverge at {small}/{large} overhead {overhead}");
				assert_eq!(ga, gb, "gauges diverge at {small}/{large} overhead {overhead}");
			}
		}
	}

	/// The size-split delta itself: with a 1024-byte threshold the replay above
	/// must actually populate BOTH segments, in both stacks. Without this the
	/// fidelity replay could pass trivially by both stacks routing everything
	/// one way.
	#[test]
	fn the_replay_actually_exercises_both_segments() {
		let ops = skewed_ops();
		// Budgets chosen so BOTH segments overflow (populating both slow
		// queues) while neither is fully drained (leaving both fast queues
		// non-empty): 100 small keys x 256 B against 8 KiB, 101 large keys
		// x 4 KiB against 128 KiB.
		let (_, _, _, _, ga, gb) = replay(8_192, 131_072, 1_024, 0, &ops);

		assert!(ga.4 > 0 && ga.5 > 0, "baseline did not populate both fast segments: {ga:?}");
		assert!(ga.6 > 0 && ga.7 > 0, "baseline did not populate both slow segments: {ga:?}");
		assert_eq!(ga, gb);
	}

	/// Eviction order is separate from migration order and nothing above
	/// exercises it: `evict_one` prefers whichever slow queue holds more
	/// objects and falls back to the ratio-ranked fast queues.
	#[test]
	fn evicts_in_the_same_order() {
		let ops = skewed_ops();
		let mut a = LruSizedHybridStack::new(8_192, 8_192, 1_024).with_shared_overhead(224);
		let mut b = LruSizedCompactHybridStack::new(8_192, 8_192, 1_024).with_shared_overhead(224);

		for (k, size) in &ops {
			apply(&mut a, &mut b, *k, *size);
			a.drain_tier_migrations();
			b.drain_tier_migrations();
		}

		let mut ea = Vec::new();
		let mut eb = Vec::new();
		while let Some(k) = a.evict_one() { ea.push(k); }
		while let Some(k) = b.evict_one() { eb.push(k); }

		assert_eq!(ea, eb, "eviction order diverges");
		assert_eq!(a.len(), b.len());
		assert_eq!(b.len(), 0);
	}

	/// The fast fallback, reached only when nothing has ever been demoted:
	/// capacities generous enough that no settle fires, then drain. Ranked by
	/// `used / capacity`, so an asymmetric pair of budgets picks a specific
	/// interleaving of the two fast queues -- the thing a naive "small first"
	/// fallback would get wrong.
	#[test]
	fn fast_fallback_evicts_in_the_same_order() {
		let mut a = LruSizedHybridStack::new(4_000_000, 400_000, 1_024);
		let mut b = LruSizedCompactHybridStack::new(4_000_000, 400_000, 1_024);

		for i in 0..64u64 {
			let size = if i % 3 == 0 { 4_096 } else { 256 };
			a.insert(i + 1, size);
			b.insert(i + 1, size);
		}

		assert_eq!(a.slow_object_count(), 0, "nothing should have been demoted");
		assert_eq!(b.slow_object_count(), 0, "nothing should have been demoted");

		let mut ea = Vec::new();
		let mut eb = Vec::new();
		while let Some(k) = a.evict_one() { ea.push(k); }
		while let Some(k) = b.evict_one() { eb.push(k); }

		assert_eq!(ea, eb, "fast-fallback eviction order diverges");
		assert_eq!(eb.len(), 64);
	}

	/// The fast fallback's tie-break. Equal capacities and equal usage make the
	/// two ratios exactly equal, and the baseline breaks that tie toward SMALL
	/// (`>=`, not `>`). Nothing else in this suite produces an exact tie, so
	/// without this the direction is unpinned.
	#[test]
	fn a_fast_fallback_ratio_tie_goes_to_the_small_segment() {
		let mut a = LruSizedHybridStack::new(1_000_000, 1_000_000, 1_024);
		let mut b = LruSizedCompactHybridStack::new(1_000_000, 1_000_000, 1_024);

		// Two 512 B small objects against one 1_024 B large one (1_024 is AT
		// the threshold, so it is large): 1_024 bytes on each side.
		for (k, size) in [(1u64, 512u32), (2, 512), (3, 1_024)] {
			a.insert(k, size);
			b.insert(k, size);
		}

		assert_eq!(a.slow_object_count(), 0, "the fallback must be the path under test");
		assert_eq!(
			a.small_fast_bytes_used(),
			a.large_fast_bytes_used(),
			"the two ratios must tie for this test to mean anything"
		);

		let ea = a.evict_one();
		let eb = b.evict_one();

		assert_eq!(ea, Some(1), "a tie must evict the SMALL segment's LRU tail");
		assert_eq!(ea, eb, "fast-fallback tie-break diverges");
		assert_eq!(gauges(&a), gauges(&b));
	}

	/// Resizing, in all THREE directions this design has -- small segment,
	/// large segment, and the classification threshold -- with BRAND-NEW keys
	/// arriving afterwards.
	///
	/// The shape matters. On the LFU conversion the equivalent test passed with
	/// a real bug present, because the workload drew from a fixed key set and
	/// every key already existed by the time the resize happened -- so nothing
	/// a resize changes was observable.
	#[test]
	fn resizes_like_lru_sized_hybrid() {
		for (small, large, threshold) in [
			(65_536u64, 65_536u64, 1_024u64),
			(16_384, 131_072, 1_024),
			(131_072, 16_384, 8_192),
			(16_384, 16_384, 128),
		] {
			let mut a = LruSizedHybridStack::new(65_536, 65_536, 1_024).with_shared_overhead(224);
			let mut b =
				LruSizedCompactHybridStack::new(65_536, 65_536, 1_024).with_shared_overhead(224);
			let (mut ma, mut mb) = (Vec::new(), Vec::new());

			for i in 0..4_000u64 {
				let k = (i % 200) + 1;
				let size = if k % 2 == 0 { 4_096 } else { 256 };
				apply(&mut a, &mut b, k, size);
				ma.extend(a.drain_tier_migrations());
				mb.extend(b.drain_tier_migrations());
			}

			a.resize_fast_tier(small);
			b.resize_fast_tier(small);
			a.resize_large_fast_tier(large);
			b.resize_large_fast_tier(large);
			a.resize_size_threshold(threshold);
			b.resize_size_threshold(threshold);
			ma.extend(a.drain_tier_migrations());
			mb.extend(b.drain_tier_migrations());

			for i in 0..2_000u64 {
				let k = 10_000 + i;
				let size = if i % 2 == 0 { 4_096 } else { 256 };
				a.insert(k, size);
				b.insert(k, size);
				ma.extend(a.drain_tier_migrations());
				mb.extend(b.drain_tier_migrations());
			}

			assert_eq!(ma, mb, "migrations diverge resizing to {small}/{large}/{threshold}");
			assert_eq!(
				gauges(&a),
				gauges(&b),
				"gauges diverge resizing to {small}/{large}/{threshold}"
			);

			for i in 0..2_000u64 {
				assert_eq!(
					a.tier_of(10_000 + i),
					b.tier_of(10_000 + i),
					"tier of new key {} diverges resizing to {small}/{large}/{threshold}",
					10_000 + i
				);
			}
		}
	}

	/// Removal from any of the four queues, which nothing above exercises.
	#[test]
	fn removal_matches_across_all_four_queues() {
		let ops = skewed_ops();
		let mut a = LruSizedHybridStack::new(8_192, 8_192, 1_024).with_shared_overhead(224);
		let mut b = LruSizedCompactHybridStack::new(8_192, 8_192, 1_024).with_shared_overhead(224);
		let (mut ma, mut mb) = (Vec::new(), Vec::new());

		for (i, (k, size)) in ops.iter().enumerate() {
			apply(&mut a, &mut b, *k, *size);

			// Remove a key periodically -- across a long enough run this hits
			// residents of all four queues.
			if i % 97 == 0 {
				let victim = (i as u64 % 200) + 1;
				a.remove(victim);
				b.remove(victim);
			}

			ma.extend(a.drain_tier_migrations());
			mb.extend(b.drain_tier_migrations());
		}

		assert_eq!(ma, mb, "migrations diverge under removal");
		assert_eq!(a.len(), b.len(), "lengths diverge under removal");
		assert_eq!(gauges(&a), gauges(&b), "gauges diverge under removal");
		// The undivided reservation total, which `reserved_shares` proportions
		// and which `gauges` has no room for.
		assert_eq!(
			a.dram_reserved_bytes(),
			b.dram_reserved_bytes(),
			"reserved DRAM diverges under removal",
		);
	}

	/// Reclassification: an overwrite whose new size crosses the threshold
	/// moves the key between the two FAST segments and must emit NO migration,
	/// because both segments are physically `Tier::Fast`. A promotion out of a
	/// slow queue in the same call must still emit one.
	///
	/// This is the delta that does not exist in `LruCompactHybridStack` at all
	/// -- with a single fast queue there is nothing to reclassify between --
	/// so it fails outright if the size routing was not ported.
	#[test]
	fn reclassification_moves_segments_without_a_migration() {
		let mut a = LruSizedHybridStack::new(1_000_000, 1_000_000, 1_024);
		let mut b = LruSizedCompactHybridStack::new(1_000_000, 1_000_000, 1_024);

		a.insert(1, 256);
		b.insert(1, 256);
		assert_eq!(a.small_fast_object_count(), 1);
		assert_eq!(gauges(&a), gauges(&b));
		assert!(a.drain_tier_migrations().is_empty());
		assert!(b.drain_tier_migrations().is_empty());

		// Same key, now above the threshold: SmallFast -> LargeFast.
		a.insert(1, 4_096);
		b.insert(1, 4_096);

		assert_eq!(a.small_fast_object_count(), 0, "baseline did not reclassify");
		assert_eq!(a.large_fast_object_count(), 1, "baseline did not reclassify");
		assert_eq!(gauges(&a), gauges(&b), "reclassified gauges diverge");
		assert_eq!(
			a.drain_tier_migrations(),
			b.drain_tier_migrations(),
			"reclassification migrations diverge"
		);
		assert_eq!(a.tier_of(1), Some(Tier::Fast));
		assert_eq!(b.tier_of(1), Some(Tier::Fast));
	}

	/// The classification boundary itself. `classify` is `size <
	/// size_threshold`, so a size EQUAL to the threshold is LARGE. Widening
	/// that to `<=` is invisible to any workload whose sizes all sit strictly
	/// on one side or the other, which is what this test exists to prevent.
	#[test]
	fn a_size_exactly_at_the_threshold_classifies_large_like_the_baseline() {
		let mut a = LruSizedHybridStack::new(1_000_000, 1_000_000, 1_024);
		let mut b = LruSizedCompactHybridStack::new(1_000_000, 1_000_000, 1_024);

		for (k, size) in [(1u64, 1_023u32), (2, 1_024), (3, 1_025)] {
			a.insert(k, size);
			b.insert(k, size);
		}

		assert_eq!(a.small_fast_object_count(), 1, "only 1_023 is below the threshold");
		assert_eq!(a.large_fast_object_count(), 2, "1_024 and 1_025 are both LARGE");
		assert_eq!(gauges(&a), gauges(&b), "threshold-boundary routing diverges");
		assert_eq!(a.tier_of(2), b.tier_of(2));
	}

	/// Segment independence: pressure on the SMALL segment demotes only small
	/// keys and leaves a comfortably-sized LARGE segment entirely alone. A
	/// single-fast-queue stack cannot express this, so it is the second test
	/// that fails if the four-queue routing was not ported.
	#[test]
	fn segment_pressure_is_independent() {
		let mut a = LruSizedHybridStack::new(2_048, 1_000_000, 1_024);
		let mut b = LruSizedCompactHybridStack::new(2_048, 1_000_000, 1_024);
		let (mut ma, mut mb) = (Vec::new(), Vec::new());

		for i in 0..200u64 {
			// Interleaved: even keys large, odd keys small.
			let size = if i % 2 == 0 { 4_096 } else { 256 };
			a.insert(i + 1, size);
			b.insert(i + 1, size);
			ma.extend(a.drain_tier_migrations());
			mb.extend(b.drain_tier_migrations());
		}

		assert_eq!(ma, mb, "migrations diverge under small-segment pressure");
		assert_eq!(gauges(&a), gauges(&b));

		assert!(!ma.is_empty(), "the small segment should have demoted something");
		assert_eq!(a.large_slow_object_count(), 0, "the large segment must not have demoted");
		assert_eq!(b.large_slow_object_count(), 0, "the large segment must not have demoted");
		assert!(a.small_slow_object_count() > 0);
		assert!(b.small_slow_object_count() > 0);
		assert_eq!(a.large_fast_object_count(), 100);
		assert_eq!(b.large_fast_object_count(), 100);
	}
}
