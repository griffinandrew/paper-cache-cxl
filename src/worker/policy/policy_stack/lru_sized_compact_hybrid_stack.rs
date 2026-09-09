/*
 * Copyright (c) Kia Shakiba
 *
 * This source code is licensed under the GNU AGPLv3 license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! Slab-backed size-split LRU hybrid: behaviourally identical to
//! `LruSizedHybridStack`, with one structure where that has five.
//!
//! `LruSizedHybridStack` keeps FOUR `HashList`s -- `small_fast`, `large_fast`,
//! `small_slow`, `large_slow`, each owning its own key-to-node index -- plus a
//! separate `entries` map holding `queue`, `size` and `dram_resident`. A key is
//! resident in exactly one of those four lists at a time, which is precisely
//! the condition [`ArenaQueueSet`] exists for: four intrusive orders over one
//! slab of slots, with ONE index. `MAX_QUEUES` is 4, and this is the design
//! that uses all four.
//!
//! The payload lives in the SLAB SLOT, beside the links and the key, and the
//! index is a bare table of slot numbers holding neither keys nor payloads.
//! A metadata read is therefore one probe into that table plus one dereference
//! into the slab -- a probe into a table small enough to be worth the
//! dereference. See `arena_queue_set`'s module doc for the measurements behind
//! that choice.
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
//! `tier_of` reads the tag. That is unchanged here: the shared [`NodePayload`]
//! stores the tag as a `u8`, so the enum survives as the readable form and is
//! converted at the boundary by `SizeQueue::tag` and `SizeQueue::from_u8`, and
//! the tag still doubles as the `ArenaQueueSet` slot number through
//! `SizeQueue::slot`. `NodePayload::tier` is kept in step with the tag for the
//! policies that read it, but the TAG REMAINS THE AUTHORITY here -- `tier_of`
//! still derives from it rather than reading the field.
//!
//! The baseline also keeps four `usize` object counters alongside the four
//! lists. Those are dropped: `ArenaQueueSet::queue_len` is the same number by
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
//!
//! **The baseline named above no longer exists in this crate.** Every
//! non-compact hybrid stack was removed once its compact twin was shown
//! behaviourally identical at 72 B/object of eviction stack instead of 112.
//! References to it here are historical: they say what this design is a
//! compaction OF, and they are the reason the structure looks the way it
//! does. Git history holds the baseline and the differential tests that
//! proved the two agreed.

use crate::{
	object::ObjectSize,
	worker::policy::policy_stack::{
		arena_queue_set::{ArenaQueueSet, NodePayload}, narrow_resident, watermarks, CacheSize,
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
	/// The `ArenaQueueSet` slot this queue occupies.
	#[inline]
	fn slot(self) -> usize {
		match self {
			SizeQueue::SmallFast => Q_SMALL_FAST,
			SizeQueue::LargeFast => Q_LARGE_FAST,
			SizeQueue::SmallSlow => Q_SMALL_SLOW,
			SizeQueue::LargeSlow => Q_LARGE_SLOW,
		}
	}

	/// The tag as [`NodePayload`] stores it. Deliberately `slot()` narrowed
	/// rather than an independent discriminant, so the stored tag and the
	/// queue slot cannot drift apart.
	#[inline]
	fn tag(self) -> u8 {
		self.slot() as u8
	}

	/// Inverse of [`SizeQueue::tag`]. Only this stack ever writes the tag, and
	/// it writes nothing but `tag()`, so any other value means the node was
	/// corrupted -- which is worth a panic rather than a silent fourth queue.
	#[inline]
	fn from_u8(tag: u8) -> SizeQueue {
		match tag as usize {
			Q_SMALL_FAST => SizeQueue::SmallFast,
			Q_LARGE_FAST => SizeQueue::LargeFast,
			Q_SMALL_SLOW => SizeQueue::SmallSlow,
			Q_LARGE_SLOW => SizeQueue::LargeSlow,
			other => unreachable!("queue tag {other} is not one of the four size queues"),
		}
	}

	/// The tier this queue IS. The tier lives in the tag, which is why the
	/// baseline stored no tier field, and why this is the one place
	/// `NodePayload::tier` is derived from.
	#[inline]
	fn tier(self) -> Tier {
		match self {
			SizeQueue::SmallFast | SizeQueue::LargeFast => Tier::Fast,
			SizeQueue::SmallSlow | SizeQueue::LargeSlow => Tier::Slow,
		}
	}

	#[inline]
	fn is_slow(self) -> bool {
		matches!(self, SizeQueue::SmallSlow | SizeQueue::LargeSlow)
	}
}

/// Per-key bookkeeping is [`NodePayload`], the one node every policy shares.
/// This stack reads `queue` -- its `SizeQueue` tag, narrowed to `u8` -- along
/// with `size` and `dram_resident`, and writes `tier` as a mirror of the tag
/// for the policies that read it. `freq`, `ts` and `phys` belong to other
/// policies and keep the values they were constructed with, which is what
/// `LruCompactHybridStack` does with them too.
///
/// `NodePayload::migrating` is the bytes that actually move between tiers when
/// an object migrates, deliberately distinct from `size` (`base_size`), which
/// remains the input to `classify`: the small/large split is a property of the
/// whole object as the cache accounts for it, not of its value alone. Only the
/// byte counters use `migrating`.
pub struct LruSizedCompactHybridStack {
	queues: ArenaQueueSet<NodePayload>,

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
			queues: ArenaQueueSet::default(),

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
	/// isn't tracked. Derived from the queue tag, which is where the tier
	/// actually lives here: the shared node HAS a `tier` field and this stack
	/// keeps it in step, but the tag is what it is kept in step WITH.
	pub fn tier_of(&self, key: HashedKey) -> Option<Tier> {
		self.queue_of(key).map(SizeQueue::tier)
	}

	/// Which of the four queues the given (currently tracked) key is in.
	fn queue_of(&self, key: HashedKey) -> Option<SizeQueue> {
		self.queues.payload(key).map(|p| SizeQueue::from_u8(p.queue))
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
		let queue = SizeQueue::from_u8(payload.queue);

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

		let queue = SizeQueue::from_u8(payload.queue);
		let target_small = self.classify(payload.size);
		let was_slow = queue.is_slow();
		let migrating = payload.migrating();

		match (queue, target_small) {
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

		self.sub_used(queue, migrating);
		self.queues.move_to_front_of(queue.slot(), target_queue.slot(), key);
		self.add_used(target_queue, migrating);

		if let Some(slot) = self.queues.payload_mut(key) {
			slot.queue = target_queue.tag();
			slot.tier = Some(target_queue.tier());
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
		if was_slow && self.queues.payload(key).map(|p| p.queue) == Some(target_queue.tag()) {
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
	/// never whether it is tracked (`ArenaQueueSet::len` is the sum over all
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
				slot.queue = SizeQueue::SmallSlow.tag();
				slot.tier = Some(Tier::Slow);
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
				slot.queue = SizeQueue::LargeSlow.tag();
				slot.tier = Some(Tier::Slow);
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
				NodePayload {
					size,
					dram_resident,
					queue: SizeQueue::SmallFast.tag(),
					tier: Some(Tier::Fast),
					phys: Some(Tier::Fast),
					freq: 0,
					ts: 0,
				},
			);
			self.small_fast_used += migrating;
			self.settle_small_fast();
		} else {
			self.queues.push_front(
				Q_LARGE_FAST,
				key,
				NodePayload {
					size,
					dram_resident,
					queue: SizeQueue::LargeFast.tag(),
					tier: Some(Tier::Fast),
					phys: Some(Tier::Fast),
					freq: 0,
					ts: 0,
				},
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
		let queue = SizeQueue::from_u8(payload.queue);
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
