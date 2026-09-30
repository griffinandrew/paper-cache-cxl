//! `SThreeFifoCompactStack` — `SThreeFifoStack`'s policy over the slab design.
//!
//! The `HashList`-based `SThreeFifoStack` this re-lays-out was removed in R2 with the other
//! original flat stacks, so the mentions of it below are historical. Its eviction
//! orders for the op sequences in `fidelity_tests` are recorded in `golden.rs` and
//! asserted there.
//!
//! Faithful S3-FIFO: a byte-budgeted `small` FIFO, a byte-budgeted `main`
//! FIFO, and a ghost queue of bare keys. A miss whose key is in the ghost
//! queue is admitted straight to `main`; everything else lands in `small`.
//! Eviction drains `small` while `main` has room, promoting anything with
//! `freq > 1` and ghosting the rest; `main` eviction is a clock sweep that
//! decrements `freq` and re-queues until it finds a zero.
//!
//! The ghost queue stays a SEPARATE structure rather than a third queue in
//! the same `CompactQueueSet`, because S3-FIFO lets a key sit in the ghost
//! queue and in `main` at the same time: `insert` routes a ghost-hit key into
//! `main` without retiring its ghost entry, and only the `evict_main` trim
//! ever removes it. One key-indexed slab cannot represent a key twice, so
//! folding the ghost queue in would silently drop one of the two memberships
//! and change which keys get admitted to `main`.
//!
//! Unlike the `HashList`-based original this honours `eviction_stacks_pmem`,
//! because `CompactQueueSet` is allocator-parameterised.

use std::cmp;

use crate::{
	CacheSize,
	HashedKey,
	ObjectSize,
	PaperPolicy,
};

use super::{
	PolicyStack,
	compact_queue_set::CompactQueueSet,
};

const Q_SMALL: usize = 0;
const Q_MAIN: usize = 1;

/// The ghost queue is its own single-queue `CompactQueueSet`; see the module
/// doc for why it cannot share the resident slab.
const Q_GHOST: usize = 0;

/// S3-FIFO's frequency counter saturates at 3 (`Object::incr_freq`).
const MAX_FREQ: u8 = 3;

/// Which of the two resident queues a key currently occupies.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Queue {
	Small,
	Main,
}

/// Per-key bookkeeping, carried in the index value. This is the compact
/// counterpart of the original's `Object { key, size, freq }` — the key is
/// already the index's key, so only the size, the queue tag and the frequency
/// ride here.
#[derive(Clone, Copy)]
struct SThreeFifoPayload {
	size: ObjectSize,
	queue: Queue,
	freq: u8,
}

/// Pinned. The payload rides in the index bucket, so growth here costs bytes
/// on every tracked key; the tag and the counter are free inside the `u32`'s
/// alignment padding.
const _: () = assert!(
	std::mem::size_of::<SThreeFifoPayload>() == 8,
	"SThreeFifoPayload grew past 8 bytes",
);

pub struct SThreeFifoCompactStack {
	ratio: f64,

	queues: CompactQueueSet<SThreeFifoPayload>,
	ghost: CompactQueueSet<()>,

	small_used: CacheSize,
	main_used: CacheSize,

	small_max: Option<CacheSize>,
	main_max: Option<CacheSize>,
}

impl SThreeFifoCompactStack {
	pub fn new(ratio: f64, max_size: CacheSize) -> Self {
		SThreeFifoCompactStack {
			ratio,

			queues: CompactQueueSet::default(),
			ghost: CompactQueueSet::default(),

			small_used: 0,
			main_used: 0,

			small_max: Some((ratio * max_size as f64) as CacheSize),
			main_max: Some(((1.0 - ratio) * max_size as f64) as CacheSize),
		}
	}

	fn queue_index(queue: Queue) -> usize {
		match queue {
			Queue::Small => Q_SMALL,
			Queue::Main => Q_MAIN,
		}
	}

	/// Mirrors `Stack::is_full`: an unbounded queue is never full.
	fn main_is_full(&self) -> bool {
		match self.main_max {
			Some(max) => self.main_used >= max,
			None => false,
		}
	}

	fn add_used(&mut self, queue: Queue, size: ObjectSize) {
		let size = size as CacheSize;

		match queue {
			Queue::Small => self.small_used += size,
			Queue::Main => self.main_used += size,
		}
	}

	fn sub_used(&mut self, queue: Queue, size: ObjectSize) {
		let size = size as CacheSize;

		match queue {
			Queue::Small => self.small_used = self.small_used.saturating_sub(size),
			Queue::Main => self.main_used = self.main_used.saturating_sub(size),
		}
	}

	/// `Stack::insert` — push to the front and charge the queue.
	fn insert_into(&mut self, queue: Queue, key: HashedKey, size: ObjectSize, freq: u8) {
		self.queues.push_front(
			Self::queue_index(queue),
			key,
			SThreeFifoPayload { size, queue, freq },
		);

		self.add_used(queue, size);
	}

	/// `Stack::pop` — take the tail and discharge the queue.
	fn pop_from(&mut self, queue: Queue) -> Option<(HashedKey, SThreeFifoPayload)> {
		let (key, payload) = self.queues.pop_back(Self::queue_index(queue))?;
		self.sub_used(queue, payload.size);

		Some((key, payload))
	}

	/// `Stack::update`'s used-size adjustment, applied to whichever queue
	/// holds the key.
	fn resize_resident(&mut self, key: HashedKey, size: ObjectSize) {
		let Some(payload) = self.queues.payload(key) else {
			return;
		};

		self.sub_used(payload.queue, payload.size);
		self.add_used(payload.queue, size);

		if let Some(p) = self.queues.payload_mut(key) {
			p.size = size;
		}
	}

	/// `HashList::push_front` is key-indexed and so de-duplicates; the slab's
	/// `push_front` appends unconditionally. The two only differ when the key
	/// is already queued, so re-queue it instead of double-linking it.
	fn ghost_push(&mut self, key: HashedKey) {
		if self.ghost.contains(key) {
			self.ghost.move_front(Q_GHOST, key);
		} else {
			self.ghost.push_front(Q_GHOST, key, ());
		}
	}

	/// Drains `small`'s tail. A key seen more than once graduates to `main`
	/// (and, if that fills `main`, the eviction is served from `main`
	/// instead); anything else is ghosted and reported as the eviction.
	fn evict_small(&mut self) -> Option<HashedKey> {
		loop {
			let (key, payload) = self.pop_from(Queue::Small)?;

			if payload.freq > 1 {
				self.insert_into(Queue::Main, key, payload.size, payload.freq);

				if self.main_is_full() {
					return self.evict_main();
				}
			} else {
				self.ghost_push(key);
				return Some(key);
			}
		}
	}

	/// Clock sweep over `main`: a key with any frequency left is re-queued at
	/// the front one decrement poorer, and the first key that reaches zero is
	/// evicted. The ghost queue is then trimmed to `main`'s length, which is
	/// what bounds it.
	fn evict_main(&mut self) -> Option<HashedKey> {
		loop {
			let (key, payload) = self.pop_from(Queue::Main)?;

			if payload.freq > 0 {
				self.insert_into(Queue::Main, key, payload.size, payload.freq - 1);
			} else {
				while self.ghost.queue_len(Q_GHOST) > self.queues.queue_len(Q_MAIN) {
					self.ghost.pop_back(Q_GHOST);
				}

				return Some(key);
			}
		}
	}
}

impl PolicyStack for SThreeFifoCompactStack {
	fn is_policy(&self, policy: &PaperPolicy) -> bool {
		let PaperPolicy::SThreeFifoCompact(ratio) = policy else {
			return false;
		};

		self.ratio == *ratio
	}

	fn len(&self) -> usize {
		self.queues.len()
	}

	fn contains(&self, key: HashedKey) -> bool {
		self.queues.contains(key)
	}

	/// A ghost hit is admitted straight to `main`; the ghost entry is
	/// deliberately left in place, matching the original.
	fn insert(&mut self, key: HashedKey, size: ObjectSize) {
		if self.queues.contains(key) {
			self.resize_resident(key, size);
			return self.update(key);
		}

		if self.ghost.contains(key) {
			self.insert_into(Queue::Main, key, size, 0);
		} else {
			self.insert_into(Queue::Small, key, size, 0);
		}
	}

	/// S3-FIFO never re-orders on a hit; it only bumps the saturating
	/// frequency counter that the two eviction sweeps read.
	fn update(&mut self, key: HashedKey) {
		if let Some(payload) = self.queues.payload_mut(key) {
			payload.freq = cmp::min(payload.freq + 1, MAX_FREQ);
		}
	}

	fn remove(&mut self, key: HashedKey) {
		if let Some(payload) = self.queues.payload(key) {
			self.queues.remove(Self::queue_index(payload.queue), key);
			self.sub_used(payload.queue, payload.size);
		}

		self.ghost.remove(Q_GHOST, key);
	}

	fn resize(&mut self, max_size: CacheSize) {
		self.small_max = Some((self.ratio * max_size as f64) as CacheSize);
		self.main_max = Some(((1.0 - self.ratio) * max_size as f64) as CacheSize);
	}

	fn clear(&mut self) {
		self.queues.clear();
		self.ghost.clear();

		self.small_used = 0;
		self.main_used = 0;
	}

	fn evict_one(&mut self) -> Option<HashedKey> {
		if !self.main_is_full() {
			// prioritize evicting from the small stack when possible
			if let Some(key) = self.evict_small() {
				return Some(key);
			}
		}

		self.evict_main()
	}
}

/// Fidelity to `SThreeFifoStack`, whose policy this re-lays-out and which was
/// removed in R2: the eviction order it gave for the op sequences below is
/// recorded in `golden` and asserted here.
#[cfg(test)]
mod fidelity_tests {
	use super::*;
	use super::super::golden;

	const MAX: CacheSize = 100_000;
	const RATIO: f64 = 0.1;

	/// Sizes are varied so both byte budgets actually bind, and the key
	/// distribution is skewed so keys accumulate frequency and exercise the
	/// small-to-main promotion and the `main` clock sweep rather than only the
	/// ghost path.
	#[test]
	fn evicts_in_the_recorded_order() {
		assert_eq!(
			golden::budgeted_skewed(&mut SThreeFifoCompactStack::new(RATIO, MAX)),
			golden::S3_SKEWED,
		);
	}

	/// Interleaves evictions with inserts so the ghost queue fills, keys get
	/// re-admitted through it into `main`, and the ghost trim in `evict_main`
	/// runs -- the paths a drain-at-the-end test never reaches.
	#[test]
	fn ghost_readmission_keeps_the_recorded_order() {
		assert_eq!(
			golden::ghost_readmission(&mut SThreeFifoCompactStack::new(RATIO, MAX)),
			golden::S3_GHOST_READMISSION,
		);
	}

	#[test]
	fn removal_leaves_the_recorded_order() {
		assert_eq!(
			golden::removal(&mut SThreeFifoCompactStack::new(RATIO, MAX), 3_000, 1_024, true, 7),
			golden::S3_REMOVAL,
		);
	}

	/// `resize` re-derives both budgets, which moves the `main`-is-full test
	/// that decides whether `evict_one` drains `small` at all.
	#[test]
	fn resize_keeps_the_recorded_order() {
		assert_eq!(
			golden::resize(&mut SThreeFifoCompactStack::new(RATIO, MAX), MAX),
			golden::S3_RESIZE,
		);
	}

	/// `record_access` is the only place the ghost queue is observable from
	/// outside, so it gets its own sequence.
	#[test]
	fn ghost_hits_keep_the_recorded_order() {
		assert_eq!(
			golden::ghost_hits(&mut SThreeFifoCompactStack::new(RATIO, MAX)),
			golden::S3_GHOST_HITS,
		);
	}
}
