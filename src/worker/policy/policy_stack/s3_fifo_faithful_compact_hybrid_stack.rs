//! Faithful tier-segmented S3-FIFO, in four variants.
//!
//! Every other S3-FIFO hybrid in this tree diverges from the flat algorithm in
//! `s_three_fifo_stack.rs` in three ways at once, which is why `s3-fifo-0.1`
//! and `s3-fifo-hybrid-0.1` are not a fair pair and never were:
//!
//!   1. they carry an `accessed: bool` reference bit where flat carries a
//!      0..=3 SATURATING COUNTER;
//!   2. they promote EAGERLY -- a hit in the one-access queue moves the key to
//!      main immediately -- where flat's hit does nothing but bump the counter,
//!      and promotion happens only later, at eviction, gated on `freq > 1`;
//!   3. their main eviction gives one second chance on a bit, where flat
//!      DECREMENTS the counter and re-queues, so a key with `freq == 3`
//!      survives three sweeps.
//!
//! These four restore all three, so `s3-fifo-<r>` and
//! `s3-fifo-faithful-compact-hybrid-<r>` differ only by tiering.
//!
//! # Why three physical queues
//!
//! Main is split into `Q_MAIN_FAST` and `Q_MAIN_SLOW`, and main's logical order
//! is the concatenation `[MAIN_FAST front..back] ++ [MAIN_SLOW front..back]`.
//! Every tiering operation moves exactly one key across the seam between
//! fast's BACK and slow's FRONT, which is a no-op on that concatenation.
//!
//! **Therefore tier placement never reorders and never evicts, and eviction
//! order is identical to flat S3-FIFO at ANY fast-tier size** -- including
//! zero. That is a stronger guarantee than the rest of the family can make,
//! and it is what `evicts_in_the_same_order_as_flat_at_every_fast_size` pins.
//!
//! It also removes `main_boundary`, the cursor the other hybrids maintain in
//! six places: the boundary IS `Q_MAIN_FAST`'s back. And it makes the reprieve
//! splice an O(1) `move_to_front_of` rather than the O(number of fast keys)
//! walk that made an earlier reprieve design unrunnable (see
//! `s3_fifo_lazy_demotion_fast_admission_reprieve_hybrid_stack`'s module doc:
//! ~18 minutes of CPU without completing a run).
//!
//! # The four variants
//!
//! | type alias | small queue | key aged out of small |
//! |---|---|---|
//! | `S3FifoFaithfulCompactHybridStack` | slow | ghosted, evicted |
//! | `S3FifoFaithfulFastAdmissionCompactHybridStack` | **fast** | ghosted, evicted |
//! | `S3FifoFaithfulReprieveCompactHybridStack` | slow | **spliced into main-slow** |
//! | `S3FifoFaithfulFastAdmissionReprieveCompactHybridStack` | **fast** | **spliced into main-slow** |
//!
//! One generic core rather than four files, against this tree's usual
//! convention. The two axes are mechanical and provably orthogonal, while the
//! part that is subtle -- the counter, lazy promotion, lazy eviction -- is
//! identical in all four. Four copies of that would create a divergence no test
//! in the tree could detect, in a family whose type names already differ by one
//! word inside sixty characters. Const generics monomorphise, so
//! `if SMALL_IS_FAST` folds at compile time.
//!
//! # The ghost queue, and why the reprieve variants have none
//!
//! Flat's ghost is written at exactly one site: the arm of `evict_small` that
//! THROWS A KEY OUT of small. Its job is to remember a key we just discarded so
//! that if it returns we admit it to main instead of making it re-earn its way
//! through small.
//!
//! Under the reprieve, that arm keeps the key resident, so nothing populates a
//! ghost. Building one from main-slow-tail evictions instead would invert the
//! signal -- "keys that failed a LONG main-queue test get admitted straight to
//! main" -- which is close to admitting everything to main and forfeits the
//! scan resistance S3-FIFO exists for. So variants 1 and 2 carry an exact
//! ghost and variants 3 and 4 carry none, deliberately.
//!
//! The ghost is a SEPARATE `CompactQueueSet`, not a fourth queue in the same
//! slab, because flat's `insert` does not retire a ghost entry when it admits
//! through it: a key can be in the ghost AND in main at once, which one
//! key-indexed slab cannot represent. Exact bare keys, not the fingerprint
//! `GhostFilter` the other hybrids use, because flat's is exact.

use crate::{
	object::ObjectSize,
	worker::policy::policy_stack::{
		compact_queue_set::CompactQueueSet, narrow_resident, watermarks, CacheSize, HashedKey,
		PolicyStack, Tier,
	},
	PaperPolicy,
};

const Q_SMALL: usize = 0;
const Q_MAIN_FAST: usize = 1;
const Q_MAIN_SLOW: usize = 2;

/// Index into the SEPARATE ghost set, which has one queue of its own.
const Q_GHOST: usize = 0;

/// S3-FIFO's frequency counter saturates at 3, matching `Object::incr_freq`
/// in `s_three_fifo_stack.rs`.
const MAX_FREQ: u8 = 3;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Queue {
	Small,
	Main,
}

/// Per-key bookkeeping, carried in the index value rather than the slab slot.
///
/// `freq` REPLACES the `accessed: bool` every other stack in this family
/// carries -- one byte for one byte. That is the only reason this still fits in
/// eight bytes, and there is no slack left: adding a field trips the assert
/// below and invalidates the 72 B/object eviction-stack constant the whole
/// compact family shares.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct S3FifoFaithfulPayload {
	queue: Queue,
	/// Which physical main list holds the key. `None` only while
	/// `queue == Small` and the small queue is slow-resident.
	tier: Option<Tier>,
	/// The part of `size` that stays in DRAM in either tier; see `migrating`.
	dram_resident: u8,
	/// 0..=3, saturating. Bumped by a hit, read by both eviction paths.
	freq: u8,
	size: ObjectSize,
}

const _: () = assert!(
	std::mem::size_of::<S3FifoFaithfulPayload>() == 8,
	"S3FifoFaithfulPayload grew past 8 bytes",
);

impl S3FifoFaithfulPayload {
	fn migrating(&self) -> CacheSize {
		(self.size as CacheSize).saturating_sub(self.dram_resident as CacheSize)
	}
}

pub struct S3FifoFaithfulCore<const SMALL_IS_FAST: bool, const REPRIEVE: bool> {
	ratio: f64,

	queues: CompactQueueSet<S3FifoFaithfulPayload>,

	/// Exact, FIFO-ordered bare keys. Always empty when `REPRIEVE`.
	ghost: CompactQueueSet<()>,

	small_used: CacheSize,
	fast_used: CacheSize,
	slow_used: CacheSize,

	/// Reservation only. Flat writes `small.max_size` and never reads it, so
	/// the small queue has no ceiling and `ratio`'s whole behavioural effect is
	/// through `main_capacity`. Kept because the fast-admission variants carve
	/// their share of the fast tier from it.
	small_capacity: CacheSize,
	main_capacity: CacheSize,

	fast_capacity: CacheSize,
	shared_overhead: CacheSize,

	migrations: Vec<(HashedKey, Tier)>,
}

pub type S3FifoFaithfulCompactHybridStack = S3FifoFaithfulCore<false, false>;
pub type S3FifoFaithfulFastAdmissionCompactHybridStack = S3FifoFaithfulCore<true, false>;
pub type S3FifoFaithfulReprieveCompactHybridStack = S3FifoFaithfulCore<false, true>;
pub type S3FifoFaithfulFastAdmissionReprieveCompactHybridStack = S3FifoFaithfulCore<true, true>;

impl<const SMALL_IS_FAST: bool, const REPRIEVE: bool> S3FifoFaithfulCore<SMALL_IS_FAST, REPRIEVE> {
	pub fn new(ratio: f64, max_size: CacheSize, fast_capacity: CacheSize) -> Self {
		S3FifoFaithfulCore {
			ratio,
			queues: CompactQueueSet::default(),
			ghost: CompactQueueSet::default(),
			small_used: 0,
			fast_used: 0,
			slow_used: 0,
			small_capacity: (ratio * max_size as f64) as CacheSize,
			main_capacity: ((1.0 - ratio) * max_size as f64) as CacheSize,
			fast_capacity,
			shared_overhead: 0,
			migrations: Vec::new(),
		}
	}

	pub fn with_shared_overhead(mut self, overhead: CacheSize) -> Self {
		self.shared_overhead = overhead;
		self
	}

	pub fn fast_capacity(&self) -> CacheSize {
		self.fast_capacity
	}

	/// Exposed for the fidelity tests: flat signals ghost membership through
	/// `record_access`'s `GhostHit`, which this stack deliberately does not
	/// override (see the module doc on `record_access` in the risks).
	pub fn is_ghost(&self, key: HashedKey) -> bool {
		self.ghost.contains(key)
	}

	pub fn tier_of(&self, key: HashedKey) -> Option<Tier> {
		let payload = self.queues.payload(key)?;

		match payload.queue {
			Queue::Small if SMALL_IS_FAST => Some(Tier::Fast),
			Queue::Small => Some(Tier::Slow),
			Queue::Main => payload.tier,
		}
	}

	fn small_tier() -> Option<Tier> {
		if SMALL_IS_FAST {
			Some(Tier::Fast)
		} else {
			Some(Tier::Slow)
		}
	}

	/// Resident objects pay the shared per-object metadata; ghost entries pay
	/// only their own structural cost.
	///
	/// Charging `shared_overhead` per ghost entry -- which an earlier draft of
	/// this file did -- overstates it by roughly 2.6x: a ghost entry is a bare
	/// key with no object behind it, so there is no `Arc` header, no object-map
	/// entry and no value to account for.
	fn reserved_overhead(&self) -> CacheSize {
		let ghost_entry =
			crate::object::overhead::EXACT_GHOST_ENTRY_DRAM_OVERHEAD as CacheSize;

		self.queues.len() as CacheSize * self.shared_overhead
			+ self.ghost.queue_len(Q_GHOST) as CacheSize * ghost_entry
	}

	/// The fast-tier budget available to `Q_MAIN_FAST`.
	///
	/// When the small queue is fast-resident it is carved out of the same
	/// DRAM, so main's share is what is left after small's reservation and
	/// after the shared per-object metadata both queues reserve.
	fn effective_main_fast_capacity(&self) -> CacheSize {
		let raw = if SMALL_IS_FAST {
			self.fast_capacity.saturating_sub(self.small_capacity)
		} else {
			self.fast_capacity
		};

		raw.saturating_sub(self.reserved_overhead())
	}

	fn queue_index(payload: &S3FifoFaithfulPayload) -> usize {
		match payload.queue {
			Queue::Small => Q_SMALL,
			Queue::Main => match payload.tier {
				Some(Tier::Fast) => Q_MAIN_FAST,
				_ => Q_MAIN_SLOW,
			},
		}
	}

	fn main_len(&self) -> usize {
		self.queues.queue_len(Q_MAIN_FAST) + self.queues.queue_len(Q_MAIN_SLOW)
	}

	/// Mirrors flat's `Stack::is_full` on the MAIN stack only. Small has no
	/// ceiling in flat, so it has none here.
	fn main_is_full(&self) -> bool {
		self.fast_used + self.slow_used >= self.main_capacity
	}

	/// `HashList::push_front` de-duplicates because it is key-indexed;
	/// `CompactQueueSet::push_front` appends unconditionally. Re-queue instead,
	/// which is the same fix MRU needed in 01d3e62.
	fn ghost_push(&mut self, key: HashedKey) {
		if self.ghost.contains(key) {
			self.ghost.move_front(Q_GHOST, key);
		} else {
			self.ghost.push_front(Q_GHOST, key, ());
		}
	}

	/// Flat trims the ghost to main's length, and only on a genuine main
	/// eviction -- never on a second chance.
	fn trim_ghost(&mut self) {
		while self.ghost.queue_len(Q_GHOST) > self.main_len() {
			self.ghost.pop_back(Q_GHOST);
		}
	}

	fn add_used(&mut self, q: usize, bytes: CacheSize) {
		match q {
			Q_SMALL => self.small_used += bytes,
			Q_MAIN_FAST => self.fast_used += bytes,
			_ => self.slow_used += bytes,
		}
	}

	fn sub_used(&mut self, q: usize, bytes: CacheSize) {
		match q {
			Q_SMALL => self.small_used = self.small_used.saturating_sub(bytes),
			Q_MAIN_FAST => self.fast_used = self.fast_used.saturating_sub(bytes),
			_ => self.slow_used = self.slow_used.saturating_sub(bytes),
		}
	}

	/// Enters main at the front, in the fast segment. Used by both the
	/// ghost-hit admission and the small -> main promotion.
	fn place_at_main_front(&mut self, key: HashedKey, from: Option<usize>, bytes: CacheSize) {
		match from {
			Some(q) => {
				self.queues.move_to_front_of(q, Q_MAIN_FAST, key);
				self.sub_used(q, bytes);
			},
			None => {},
		}

		if let Some(p) = self.queues.payload_mut(key) {
			p.queue = Queue::Main;
			p.tier = Some(Tier::Fast);
		}

		self.fast_used += bytes;
		self.settle_fast_tier();
	}

	/// A key that survives lazy eviction moves to main's front, which is
	/// `Q_MAIN_FAST`'s front.
	fn requeue_to_main_front(&mut self, key: HashedKey, from: usize, bytes: CacheSize) {
		let was_slow = from == Q_MAIN_SLOW;

		if from == Q_MAIN_FAST {
			self.queues.move_front(Q_MAIN_FAST, key);
		} else {
			self.queues.move_to_front_of(Q_MAIN_SLOW, Q_MAIN_FAST, key);
			self.slow_used = self.slow_used.saturating_sub(bytes);
			self.fast_used += bytes;

			if let Some(p) = self.queues.payload_mut(key) {
				p.tier = Some(Tier::Fast);
			}
		}

		self.settle_fast_tier();

		// Only a key that was actually in the slow tier needs its bytes moved.
		if was_slow && self.queues.payload(key).and_then(|p| p.tier) == Some(Tier::Fast) {
			self.migrations.push((key, Tier::Fast));
		}
	}

	/// Demotes from `Q_MAIN_FAST`'s back to `Q_MAIN_SLOW`'s front -- across the
	/// seam, so the concatenated main order is unchanged.
	fn settle_fast_tier(&mut self) {
		let effective = self.effective_main_fast_capacity();

		if self.fast_used <= watermarks::high_bytes(effective) {
			return;
		}

		let drain_target = watermarks::low_bytes(effective);

		while self.fast_used > drain_target {
			let Some(key) = self.queues.back(Q_MAIN_FAST) else { break };
			let bytes = self.queues.payload(key).map(|p| p.migrating()).unwrap_or(0);

			self.queues.move_to_front_of(Q_MAIN_FAST, Q_MAIN_SLOW, key);

			if let Some(p) = self.queues.payload_mut(key) {
				p.tier = Some(Tier::Slow);
			}

			self.fast_used = self.fast_used.saturating_sub(bytes);
			self.slow_used += bytes;

			self.migrations.push((key, Tier::Slow));
		}
	}

	fn resize_key(&mut self, key: HashedKey, new_size: ObjectSize, new_resident: u8) {
		let Some(payload) = self.queues.payload(key) else { return };
		let q = Self::queue_index(&payload);
		let old = payload.migrating();

		if let Some(p) = self.queues.payload_mut(key) {
			p.size = new_size;
			p.dram_resident = new_resident;
		}

		let new = self.queues.payload(key).map(|p| p.migrating()).unwrap_or(0);
		self.sub_used(q, old);
		self.add_used(q, new);
	}

	/// The whole of S3-FIFO's hit path: bump a saturating counter. No queue
	/// movement, no promotion, no migration.
	///
	/// This is the single most important difference from every other stack in
	/// this family, and the one a copied `touch` would silently undo.
	fn bump_freq(&mut self, key: HashedKey) {
		if let Some(p) = self.queues.payload_mut(key) {
			p.freq = (p.freq + 1).min(MAX_FREQ);
		}
	}

	/// Flat's `evict_small`: drain small's tail, promoting anything seen more
	/// than once and disposing of the rest.
	fn evict_small(&mut self) -> Option<HashedKey> {
		loop {
			let key = self.queues.back(Q_SMALL)?;
			let payload = self.queues.payload(key)?;
			let bytes = payload.migrating();

			if payload.freq > 1 {
				self.place_at_main_front(key, Some(Q_SMALL), bytes);

				// Promotion into a full main is resolved by main, exactly as
				// flat does it.
				if self.main_is_full() {
					return self.evict_main();
				}

				// Bytes already in DRAM need no migration; a slow-resident
				// small queue does.
				if !SMALL_IS_FAST
					&& self.queues.payload(key).and_then(|p| p.tier) == Some(Tier::Fast)
				{
					self.migrations.push((key, Tier::Fast));
				}

				continue;
			}

			if REPRIEVE {
				// Spliced into main's slow segment rather than evicted: still
				// resident, so nothing is reported and nothing is ghosted.
				self.queues.move_to_front_of(Q_SMALL, Q_MAIN_SLOW, key);
				self.small_used = self.small_used.saturating_sub(bytes);
				self.slow_used += bytes;

				if let Some(p) = self.queues.payload_mut(key) {
					p.queue = Queue::Main;
					p.tier = Some(Tier::Slow);
				}

				// DRAM -> PMEM is a real byte move; PMEM -> PMEM is not.
				if SMALL_IS_FAST {
					self.migrations.push((key, Tier::Slow));
				}

				if self.main_is_full() {
					return self.evict_main();
				}

				continue;
			}

			let payload = self.queues.remove(Q_SMALL, key);
			self.small_used = self
				.small_used
				.saturating_sub(payload.map(|p| p.migrating()).unwrap_or(0));
			self.ghost_push(key);

			return Some(key);
		}
	}

	/// Flat's `evict_main`: a lazy sweep that decrements and re-queues until it
	/// finds a key at zero. Main's tail is `Q_MAIN_SLOW`'s back, falling back
	/// to `Q_MAIN_FAST`'s back when slow is empty.
	fn evict_main(&mut self) -> Option<HashedKey> {
		loop {
			let (key, q) = match self.queues.back(Q_MAIN_SLOW) {
				Some(key) => (key, Q_MAIN_SLOW),
				None => (self.queues.back(Q_MAIN_FAST)?, Q_MAIN_FAST),
			};

			let payload = self.queues.payload(key)?;
			let bytes = payload.migrating();

			if payload.freq > 0 {
				if let Some(p) = self.queues.payload_mut(key) {
					p.freq -= 1;
				}

				self.requeue_to_main_front(key, q, bytes);
				continue;
			}

			self.queues.remove(q, key);
			self.sub_used(q, bytes);
			self.trim_ghost();

			return Some(key);
		}
	}
}

impl<const SMALL_IS_FAST: bool, const REPRIEVE: bool> PolicyStack
	for S3FifoFaithfulCore<SMALL_IS_FAST, REPRIEVE>
{
	fn is_policy(&self, policy: &PaperPolicy) -> bool {
		let ratio = match (SMALL_IS_FAST, REPRIEVE) {
			(false, false) => match policy {
				PaperPolicy::S3FifoFaithfulCompactHybrid(r) => r,
				_ => return false,
			},
			(true, false) => match policy {
				PaperPolicy::S3FifoFaithfulFastAdmissionCompactHybrid(r) => r,
				_ => return false,
			},
			(false, true) => match policy {
				PaperPolicy::S3FifoFaithfulReprieveCompactHybrid(r) => r,
				_ => return false,
			},
			(true, true) => match policy {
				PaperPolicy::S3FifoFaithfulFastAdmissionReprieveCompactHybrid(r) => r,
				_ => return false,
			},
		};

		*ratio == self.ratio
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

		if self.queues.contains(key) {
			self.resize_key(key, size, dram_resident);
			self.bump_freq(key);
			return;
		}

		let bytes = (size as CacheSize).saturating_sub(dram_resident as CacheSize);

		// A ghost hit skips probation and enters main directly, at freq 0.
		// The ghost entry is deliberately NOT retired -- flat leaves it, which
		// is why a key can be in both structures at once.
		if !REPRIEVE && self.ghost.contains(key) {
			self.queues.push_front(
				Q_MAIN_FAST,
				key,
				S3FifoFaithfulPayload {
					queue: Queue::Main,
					tier: Some(Tier::Fast),
					dram_resident,
					freq: 0,
					size,
				},
			);
			self.fast_used += bytes;
			self.settle_fast_tier();

			if self.queues.payload(key).and_then(|p| p.tier) == Some(Tier::Fast) {
				self.migrations.push((key, Tier::Fast));
			}

			return;
		}

		self.queues.push_front(
			Q_SMALL,
			key,
			S3FifoFaithfulPayload {
				queue: Queue::Small,
				tier: Self::small_tier(),
				dram_resident,
				freq: 0,
				size,
			},
		);

		self.small_used += bytes;

		if SMALL_IS_FAST {
			self.settle_fast_tier();
		}
	}

	/// Deliberately NOT overriding `record_access`: returning `GhostHit` would
	/// reach a `debug_assert!(promotion_tx.is_some())` that these designs do
	/// not satisfy -- a debug panic and a silent drop in release.
	fn update(&mut self, key: HashedKey) {
		if self.queues.contains(key) {
			self.bump_freq(key);
		}
	}

	fn remove(&mut self, key: HashedKey) {
		// Before the early return: a ghosted key has no resident entry.
		self.ghost.remove(Q_GHOST, key);

		let Some(payload) = self.queues.payload(key) else { return };
		let q = Self::queue_index(&payload);
		let bytes = payload.migrating();

		self.queues.remove(q, key);
		self.sub_used(q, bytes);
	}

	fn resize(&mut self, max_size: CacheSize) {
		self.small_capacity = (self.ratio * max_size as f64) as CacheSize;
		self.main_capacity = ((1.0 - self.ratio) * max_size as f64) as CacheSize;

		// Growing the small reservation shrinks main's share of the same DRAM.
		if SMALL_IS_FAST {
			self.settle_fast_tier();
		}
	}

	fn clear(&mut self) {
		self.queues.clear();
		self.ghost.clear();

		self.small_used = 0;
		self.fast_used = 0;
		self.slow_used = 0;
		self.migrations.clear();
	}

	fn evict_one(&mut self) -> Option<HashedKey> {
		if !self.main_is_full() {
			if let Some(key) = self.evict_small() {
				return Some(key);
			}
		}

		self.evict_main()
	}

	fn resize_fast_tier(&mut self, size: CacheSize) {
		self.fast_capacity = size;
		self.settle_fast_tier();
	}

	fn drain_tier_migrations(&mut self) -> Vec<(HashedKey, Tier)> {
		std::mem::take(&mut self.migrations)
	}

	fn dram_reserved_bytes(&self) -> CacheSize {
		self.reserved_overhead()
	}

	fn fast_bytes_used(&self) -> CacheSize {
		if SMALL_IS_FAST {
			self.fast_used + self.small_used
		} else {
			self.fast_used
		}
	}

	fn slow_bytes_used(&self) -> CacheSize {
		if SMALL_IS_FAST {
			self.slow_used
		} else {
			self.small_used + self.slow_used
		}
	}

	fn fast_object_count(&self) -> usize {
		if SMALL_IS_FAST {
			self.queues.queue_len(Q_MAIN_FAST) + self.queues.queue_len(Q_SMALL)
		} else {
			self.queues.queue_len(Q_MAIN_FAST)
		}
	}

	fn slow_object_count(&self) -> usize {
		if SMALL_IS_FAST {
			self.queues.queue_len(Q_MAIN_SLOW)
		} else {
			self.queues.queue_len(Q_MAIN_SLOW) + self.queues.queue_len(Q_SMALL)
		}
	}
}

/// Fidelity against flat `SThreeFifoStack`, plus the behavioural tests that
/// catch the three divergences this design exists to remove.
///
/// Gated on this design's OWN feature. The habit elsewhere in the family is to
/// gate a fidelity module on the baseline twin's feature; these have no
/// baseline twin, so that would leave the module silently uncompiled.
#[cfg(all(test, feature = "s3_fifo_faithful_compact_hybrid_cache"))]
mod fidelity_tests {
	use super::*;
	use crate::worker::policy::policy_stack::s_three_fifo_stack::SThreeFifoStack;

	const MAX: CacheSize = 100_000;
	const RATIO: f64 = 0.1;

	type Faithful = S3FifoFaithfulCompactHybridStack;
	type FaithfulFast = S3FifoFaithfulFastAdmissionCompactHybridStack;
	type Reprieve = S3FifoFaithfulReprieveCompactHybridStack;
	type ReprieveFast = S3FifoFaithfulFastAdmissionReprieveCompactHybridStack;

	fn drain(s: &mut dyn PolicyStack) -> Vec<HashedKey> {
		let mut out = Vec::new();
		while let Some(k) = s.evict_one() {
			out.push(k);
		}
		out
	}

	/// Skewed keys with varied sizes, so the main budget actually binds and the
	/// promotion, requeue and ghost paths all run.
	fn workload(n: u64) -> Vec<(HashedKey, ObjectSize, bool)> {
		let mut x: u64 = 0x243F_6A88_85A3_08D3;
		let mut ops = Vec::new();

		for i in 0..n {
			x ^= x << 13;
			x ^= x >> 7;
			x ^= x << 17;
			let u = (x >> 11) as f64 / (1u64 << 53) as f64;
			let key = ((u * u * 400.0) as u64) + 1;
			let size = (1_024 + (x % 3_072)) as ObjectSize;
			ops.push((key, size, i % 4 == 3));
		}

		ops
	}

	/// THE central guarantee: tier placement never reorders, so eviction order
	/// is identical to flat at ANY fast-tier size -- including zero.
	#[test]
	fn evicts_in_the_same_order_as_flat_at_every_fast_size() {
		let ops = workload(40_000);

		let mut flat = SThreeFifoStack::new(RATIO, MAX);
		for &(key, size, upd) in &ops {
			let s: &mut dyn PolicyStack = &mut flat;
			if upd && s.contains(key) {
				s.update(key);
			} else {
				s.insert(key, size);
			}
		}
		let expected = drain(&mut flat);

		for fast in [MAX * 4, MAX / 5, 0] {
			let mut hyb = Faithful::new(RATIO, MAX, fast);

			for &(key, size, upd) in &ops {
				let s: &mut dyn PolicyStack = &mut hyb;
				if upd && s.contains(key) {
					s.update(key);
				} else {
					s.insert(key, size);
				}
			}

			assert_eq!(
				drain(&mut hyb),
				expected,
				"eviction order diverged from flat at fast_capacity={fast}",
			);
		}
	}

	/// The single most important behavioural test. Every other stack in this
	/// family promotes on access; flat does not. A copied `touch` would pass
	/// every gauge and migration test and silently restore the old design.
	#[test]
	fn a_hit_neither_reorders_nor_promotes() {
		let mut s = Faithful::new(RATIO, MAX, MAX * 4);

		for key in 0..8u64 {
			s.insert(key, 1_024);
		}

		// Key 0 is small's tail. Touch it repeatedly.
		for _ in 0..10 {
			s.update(0);
		}

		assert_eq!(s.tier_of(0), Some(Tier::Slow), "a hit moved the key out of small");
		assert_eq!(s.queues.back(Q_SMALL), Some(0), "a hit reordered the small queue");
		assert_eq!(s.queues.queue_len(Q_MAIN_FAST) + s.queues.queue_len(Q_MAIN_SLOW), 0);
		assert_eq!(s.queues.payload(0).unwrap().freq, MAX_FREQ, "counter did not saturate at 3");

		// Now it should be PROMOTED at eviction, not evicted, and arrive with
		// its counter intact.
		let evicted = s.evict_one();
		assert_ne!(evicted, Some(0), "a twice-seen key was evicted rather than promoted");
		assert!(s.contains(0));
		assert_eq!(s.queues.payload(0).unwrap().queue, Queue::Main);
		assert_eq!(s.queues.payload(0).unwrap().freq, MAX_FREQ, "freq lost across promotion");
	}

	/// Flat decrements and re-queues to the FRONT, so a key at freq 3 survives
	/// three full laps. A CLOCK-style one-slot-forward implementation fails.
	#[test]
	fn main_requeue_goes_to_the_front_and_decrements() {
		let mut s = Faithful::new(RATIO, MAX, MAX * 4);

		// Place two keys in main deterministically, via ghost re-admission:
		// insert, let it age out of small (ghosted), then insert again -- a
		// ghost hit enters main directly at freq 0.
		//
		// Building the state with a drain loop instead does not work: the same
		// `evict_one` that promotes out of small also sweeps main, so keys are
		// gone before the test can set their frequencies.
		for key in [11u64, 10u64] {
			s.insert(key, 1_024);
			assert_eq!(s.evict_one(), Some(key));
			assert!(s.is_ghost(key));
			s.insert(key, 1_024);
			assert_eq!(s.queues.payload(key).unwrap().queue, Queue::Main);
		}

		// Main front-to-back is now [10, 11], so 11 is the tail and is swept
		// first. Give the TAIL the high frequency, so surviving the sweep is
		// what has to be demonstrated.
		let (hot, cold) = (11u64, 10u64);
		s.queues.payload_mut(hot).unwrap().freq = MAX_FREQ;
		s.queues.payload_mut(cold).unwrap().freq = 0;
		assert_eq!(
			s.queues.back(Q_MAIN_SLOW).or_else(|| s.queues.back(Q_MAIN_FAST)),
			Some(hot),
			"expected the high-frequency key at main's tail",
		);

		s.main_capacity = 0;
		let evicted = s.evict_one();

		// The tail was swept, decremented and re-queued to the FRONT, so the
		// zero-frequency key behind it became the tail and was evicted. A
		// CLOCK-style one-slot-forward move would not produce this.
		assert_eq!(evicted, Some(cold), "the zero-frequency key should be evicted");
		assert!(s.contains(hot), "a freq-3 key must survive a sweep");
		assert!(
			s.queues.payload(hot).unwrap().freq < MAX_FREQ,
			"a swept key must be decremented, not left untouched",
		);
	}

	/// A ghost re-admission enters main at freq 0 -- it has not been seen
	/// twice, it has merely been seen before.
	#[test]
	fn ghost_readmission_enters_main_at_zero() {
		let mut s = Faithful::new(RATIO, MAX, MAX * 4);

		s.insert(7, 1_024);
		let evicted = s.evict_one();
		assert_eq!(evicted, Some(7), "an unseen small key should be evicted");
		assert!(s.is_ghost(7), "an evicted small key should be ghosted");

		s.insert(7, 1_024);
		let p = s.queues.payload(7).unwrap();
		assert_eq!(p.queue, Queue::Main, "a ghost hit should skip probation");
		assert_eq!(p.freq, 0, "a ghost hit should not confer frequency");
		assert!(s.is_ghost(7), "flat does not retire the ghost entry on admission");
	}

	/// `evict_one` drains small only while main has room -- the gate is on
	/// MAIN's budget, which is the most tempting thing to mis-port.
	#[test]
	fn the_evict_one_gate_is_on_main_not_small() {
		let mut s = Faithful::new(RATIO, MAX, MAX * 4);

		for key in 0..4u64 {
			s.insert(key, 1_024);
			s.update(key);
			s.update(key);
		}
		while s.queues.queue_len(Q_SMALL) > 0 {
			s.evict_one();
		}

		// Small non-empty, main over budget.
		for key in 100..104u64 {
			s.insert(key, 1_024);
		}
		let small_before = s.queues.queue_len(Q_SMALL);
		s.main_capacity = 0;

		let evicted = s.evict_one().unwrap();
		assert!(evicted < 100, "with main full, eviction must come from main");
		assert_eq!(
			s.queues.queue_len(Q_SMALL),
			small_before,
			"small was drained despite main being full",
		);
	}

	/// Under the reprieve a key aged out of small stays resident, at main's
	/// slow front, and nothing is ever ghosted.
	#[test]
	fn aged_out_of_small_is_spliced_not_evicted() {
		let mut plain = Faithful::new(RATIO, MAX, MAX * 4);
		let mut rep = Reprieve::new(RATIO, MAX, MAX * 4);

		for key in 0..6u64 {
			plain.insert(key, 1_024);
			rep.insert(key, 1_024);
		}

		// Plain: small's tail is unseen, so it is ghosted and reported.
		assert_eq!(plain.evict_one(), Some(0));
		assert!(plain.is_ghost(0), "plain must ghost what it evicts from small");

		// Reprieve: the same key is relocated instead. Like flat's promotion
		// branch, the splice CONTINUES the drain loop rather than returning, so
		// one call moves every small key that fits before main fills -- and the
		// eviction it finally reports comes from main.
		let evicted = rep.evict_one();
		assert!(rep.ghost.is_empty(), "the reprieve variants must never ghost");
		assert_eq!(rep.queues.queue_len(Q_SMALL), 0, "small should have drained into main");

		for key in 0..6u64 {
			if Some(key) == evicted {
				continue;
			}
			assert!(rep.contains(key), "key {key} was lost rather than reprieved");
			assert_eq!(
				rep.queues.payload(key).unwrap().queue,
				Queue::Main,
				"key {key} should have been spliced into main",
			);
		}
	}

	/// The two axes are orthogonal: placement changes tier labels, not order.
	#[test]
	fn the_two_axes_do_not_change_eviction_order() {
		let ops = workload(20_000);

		fn run<S: PolicyStack>(mut s: S, ops: &[(HashedKey, ObjectSize, bool)]) -> Vec<HashedKey> {
			for &(key, size, upd) in ops {
				let d: &mut dyn PolicyStack = &mut s;
				if upd && d.contains(key) {
					d.update(key);
				} else {
					d.insert(key, size);
				}
			}
			drain(&mut s)
		}

		assert_eq!(
			run(Faithful::new(RATIO, MAX, MAX / 5), &ops),
			run(FaithfulFast::new(RATIO, MAX, MAX / 5), &ops),
			"small-queue placement changed the eviction order",
		);

		assert_eq!(
			run(Reprieve::new(RATIO, MAX, MAX / 5), &ops),
			run(ReprieveFast::new(RATIO, MAX, MAX / 5), &ops),
			"small-queue placement changed the eviction order under reprieve",
		);
	}

	/// Outside the exact regime the order may legitimately differ, but the
	/// gauges must stay consistent with the queues.
	#[test]
	fn gauges_stay_consistent_under_pressure() {
		let ops = workload(20_000);
		let mut s = FaithfulFast::new(RATIO, MAX, MAX / 5);

		for &(key, size, upd) in &ops {
			let d: &mut dyn PolicyStack = &mut s;
			if upd && d.contains(key) {
				d.update(key);
			} else {
				d.insert(key, size);
			}

			assert_eq!(
				d.fast_object_count() + d.slow_object_count(),
				d.len(),
				"object gauges disagree with len",
			);
		}
	}
}
