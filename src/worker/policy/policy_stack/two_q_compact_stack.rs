//! `TwoQCompactStack` — `TwoQStack`'s policy over the slab design.
//!
//! The `HashList`-based `TwoQStack` this re-lays-out was removed in R2 with the other
//! original flat stacks, so the mentions of it below are historical. Its eviction
//! orders for the op sequences in `fidelity_tests` are recorded in `golden.rs` and
//! asserted there.
//!
//! Faithful three-queue 2Q: `a1_in` (byte-budgeted FIFO), `a1_out` (byte-
//! budgeted, holding what `a1_in` overflowed) and `am` (unbounded LRU). A hit
//! in `a1_out` promotes to `am`; a hit in `a1_in` is deliberately ignored,
//! which is 2Q's central mechanism — a scan touching a page twice must not buy
//! promotion. Eviction drains `a1_out`, then `a1_in`, then `am`.
//!
//! NOT derived from `TwoQCompactHybridStack`: that carries two queues and
//! implements *Simplified* 2Q, a different algorithm. The faithful three-queue
//! shape is `TwoQFullFastAdmissionCompactHybridStack`, and this is its
//! non-tiered counterpart.
//!
//! `CompactQueueSet::remove`/`move_to_*` take the source queue, so the payload
//! carries a queue tag alongside the size. That tag rides in the INDEX value,
//! not the slab slot, so the slot stays 16 bytes.
//!
//! Unlike the `HashList`-based original this honours `eviction_stacks_pmem`,
//! because `CompactQueueSet` is allocator-parameterised.

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

const Q_A1_IN: usize = 0;
const Q_A1_OUT: usize = 1;
const Q_AM: usize = 2;

/// Which of the three queues a key currently occupies.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Queue {
	A1In,
	A1Out,
	Am,
}

/// Per-key bookkeeping, carried in the index value.
#[derive(Clone, Copy)]
struct TwoQPayload {
	queue: Queue,
	size: ObjectSize,
}

/// Pinned, matching both hybrid 2Q compacts. The payload rides in the index
/// bucket, so growth here costs bytes on every tracked key. `Queue` is a byte
/// and `ObjectSize` a `u32`, so the tag is free inside the padding.
const _: () = assert!(
	std::mem::size_of::<TwoQPayload>() == 8,
	"TwoQPayload grew past 8 bytes",
);

pub struct TwoQCompactStack {
	queues: CompactQueueSet<TwoQPayload>,

	k_in: f64,
	k_out: f64,

	a1_in_used: CacheSize,
	a1_out_used: CacheSize,
	am_used: CacheSize,

	a1_in_max: Option<CacheSize>,
	a1_out_max: Option<CacheSize>,
}

impl TwoQCompactStack {
	pub fn new(k_in: f64, k_out: f64, max_size: CacheSize) -> Self {
		TwoQCompactStack {
			queues: CompactQueueSet::default(),
			k_in,
			k_out,
			a1_in_used: 0,
			a1_out_used: 0,
			am_used: 0,
			a1_in_max: Some((k_in * max_size as f64) as CacheSize),
			a1_out_max: Some((k_out * max_size as f64) as CacheSize),
		}
	}

	fn queue_index(queue: Queue) -> usize {
		match queue {
			Queue::A1In => Q_A1_IN,
			Queue::A1Out => Q_A1_OUT,
			Queue::Am => Q_AM,
		}
	}

	fn add_used(&mut self, queue: Queue, size: ObjectSize) {
		let size = size as CacheSize;
		match queue {
			Queue::A1In => self.a1_in_used += size,
			Queue::A1Out => self.a1_out_used += size,
			Queue::Am => self.am_used += size,
		}
	}

	fn sub_used(&mut self, queue: Queue, size: ObjectSize) {
		let size = size as CacheSize;
		match queue {
			Queue::A1In => self.a1_in_used = self.a1_in_used.saturating_sub(size),
			Queue::A1Out => self.a1_out_used = self.a1_out_used.saturating_sub(size),
			Queue::Am => self.am_used = self.am_used.saturating_sub(size),
		}
	}

	/// Mirrors `Stack::can_fit`: an unbounded queue always fits.
	fn a1_in_can_fit(&self, size: ObjectSize) -> bool {
		match self.a1_in_max {
			Some(max) => self.a1_in_used + size as CacheSize <= max,
			None => true,
		}
	}

	/// Drains `a1_in`'s tail into `a1_out`'s front until the incoming object
	/// fits, exactly as `TwoQStack::restructure_to_fit` does. `a1_out` has a
	/// budget but no admission check — overflow there is resolved by eviction,
	/// which drains `a1_out` first.
	fn restructure_to_fit(&mut self, size: ObjectSize) {
		while !self.a1_in_can_fit(size) {
			let Some(key) = self.queues.back(Q_A1_IN) else {
				return;
			};

			let Some(payload) = self.queues.payload(key) else {
				return;
			};

			self.sub_used(Queue::A1In, payload.size);
			self.queues.move_to_front_of(Q_A1_IN, Q_A1_OUT, key);

			if let Some(p) = self.queues.payload_mut(key) {
				p.queue = Queue::A1Out;
			}

			self.add_used(Queue::A1Out, payload.size);
		}
	}

	/// Re-sizes a resident key in place, matching `Stack::update`'s
	/// used-size adjustment on whichever queue holds it.
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
}

impl PolicyStack for TwoQCompactStack {
	fn is_policy(&self, policy: &PaperPolicy) -> bool {
		let PaperPolicy::TwoQCompact(k_in, k_out) = policy else {
			return false;
		};

		self.k_in == *k_in && self.k_out == *k_out
	}

	fn len(&self) -> usize {
		self.queues.len()
	}

	fn contains(&self, key: HashedKey) -> bool {
		self.queues.contains(key)
	}

	fn insert(&mut self, key: HashedKey, size: ObjectSize) {
		if self.queues.contains(key) {
			self.resize_resident(key, size);
			return self.update(key);
		}

		self.restructure_to_fit(size);
		self.queues.push_front(Q_A1_IN, key, TwoQPayload { queue: Queue::A1In, size });
		self.add_used(Queue::A1In, size);
	}

	/// A hit in `a1_out` promotes to `am`'s front; a hit already in `am` is a
	/// recency move; a hit in `a1_in` does nothing at all — that last case is
	/// 2Q's defining rule, and `TwoQStack` gets it by `move_front` silently
	/// missing on a key that is not in `am`.
	fn update(&mut self, key: HashedKey) {
		let Some(payload) = self.queues.payload(key) else {
			return;
		};

		match payload.queue {
			Queue::A1Out => {
				self.sub_used(Queue::A1Out, payload.size);
				self.queues.move_to_front_of(Q_A1_OUT, Q_AM, key);

				if let Some(p) = self.queues.payload_mut(key) {
					p.queue = Queue::Am;
				}

				self.add_used(Queue::Am, payload.size);
			},

			Queue::Am => self.queues.move_front(Q_AM, key),

			Queue::A1In => {},
		}
	}

	fn remove(&mut self, key: HashedKey) {
		let Some(payload) = self.queues.payload(key) else {
			return;
		};

		self.queues.remove(Self::queue_index(payload.queue), key);
		self.sub_used(payload.queue, payload.size);
	}

	fn resize(&mut self, max_size: CacheSize) {
		self.a1_in_max = Some((self.k_in * max_size as f64) as CacheSize);
		self.a1_out_max = Some((self.k_out * max_size as f64) as CacheSize);
	}

	fn clear(&mut self) {
		self.queues.clear();
		self.a1_in_used = 0;
		self.a1_out_used = 0;
		self.am_used = 0;
	}

	/// `a1_out` first, then `a1_in`, then `am` — matching
	/// `TwoQStack::evict_one`.
	fn evict_one(&mut self) -> Option<HashedKey> {
		if let Some((key, payload)) = self.queues.pop_back(Q_A1_OUT) {
			self.sub_used(Queue::A1Out, payload.size);
			return Some(key);
		}

		if let Some((key, payload)) = self.queues.pop_back(Q_A1_IN) {
			self.sub_used(Queue::A1In, payload.size);
			return Some(key);
		}

		let (key, payload) = self.queues.pop_back(Q_AM)?;
		self.sub_used(Queue::Am, payload.size);
		Some(key)
	}
}

/// Fidelity to `TwoQStack`, whose policy this re-lays-out and which was removed
/// in R2: the eviction order it gave for the op sequences below is recorded in
/// `golden` and asserted here.
#[cfg(test)]
mod fidelity_tests {
	use super::*;
	use super::super::golden;

	const MAX: CacheSize = 100_000;
	const K_IN: f64 = 0.25;
	const K_OUT: f64 = 0.5;

	/// Sizes are varied so the byte budgets on `a1_in`/`a1_out` actually bind
	/// and the spill path runs -- a fixed size would exercise one branch only.
	#[test]
	fn evicts_in_the_recorded_order() {
		assert_eq!(
			golden::budgeted_skewed(&mut TwoQCompactStack::new(K_IN, K_OUT, MAX)),
			golden::TWO_Q_SKEWED,
		);
	}

	#[test]
	fn removal_leaves_the_recorded_order() {
		assert_eq!(
			golden::removal(&mut TwoQCompactStack::new(K_IN, K_OUT, MAX), 3_000, 1_024, true, 7),
			golden::TWO_Q_REMOVAL,
		);
	}

	/// `resize` re-derives both budgets, which changes where the spill point
	/// falls; the order after it is recorded too.
	#[test]
	fn resize_keeps_the_recorded_order() {
		assert_eq!(
			golden::resize(&mut TwoQCompactStack::new(K_IN, K_OUT, MAX), MAX),
			golden::TWO_Q_RESIZE,
		);
	}
}
