/*
 * Copyright (c) Kia Shakiba
 *
 * This source code is licensed under the GNU AGPLv3 license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! Slab-backed 2Q fast-admission REPRIEVE hybrid: the compact form of
//! `TwoQFastAdmissionReprieveHybridStack`, one structure where that has three.
//!
//! Everything `TwoQFastAdmissionCompactHybridStack` does, this does too: the
//! admission FIFO is DRAM-resident, so `tier_of` reports `Fast` for a key in
//! it, its reservation is carved OUT of the fast tier, a promotion out of it
//! emits no migration, and its bytes and objects count toward fast.
//!
//! What differs is what happens when the FIFO runs over budget. The
//! non-reprieve stack lets it grow and asks the caller to evict its tail
//! (`needs_capacity_eviction`). This one REPRIEVES the overflow instead:
//! `settle_fifo_queue` splices the FIFO tail onto the BACK of the main queue
//! as `Tier::Slow`, emitting a migration, so an aged-out one-access key gets a
//! second chance in PMEM rather than being dropped. Four consequences, each
//! carried over from the baseline:
//!
//! - `settle_fifo_queue` runs after every admission and after either resize --
//!   exactly the three call sites the baseline uses. It deliberately does NOT
//!   run on the re-set path of `insert_resident`, which returns early, nor on
//!   promotion out of the FIFO, which only ever lowers `fifo_used`.
//! - `needs_capacity_eviction` is NOT overridden. The FIFO polices itself, so
//!   the trait default (`false`) is the answer; the non-reprieve stack's
//!   `fifo_used > fifo_capacity` override would ask the caller to evict a
//!   queue that has already settled.
//! - `evict_one` drains the MAIN queue first and reaches the FIFO tail only
//!   when main is empty. The non-reprieve stack has that order reversed.
//! - The `shared_overhead` reservation is SPLIT between the two queues in
//!   proportion to their fast-tier capacities (`reserved_shares`), because
//!   both now settle against a budget and each has to pay its own share. The
//!   non-reprieve stack charges the whole reservation to the main queue.
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
		arena_queue_set::{ArenaQueueSet, NodePayload}, narrow_resident, drain_target, CacheSize,
		HashedKey, PolicyStack, Tier,
	},
	PaperPolicy,
};

/// Queue slots in the shared set. The FIFO admission queue is 0, the LRU main
/// queue is 1; a key is in exactly one of them.
const Q_FIFO: usize = 0;
const Q_MAIN: usize = 1;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
enum Queue {
	Fifo = 0,
	Main = 1,
}

impl Queue {
	/// The shared node stores `queue` as a bare `u8`, so this is the one
	/// place the tag becomes an enum again. Every write goes the other way
	/// through `Queue as u8`, which is why the last arm cannot be reached.
	#[inline]
	fn from_u8(tag: u8) -> Queue {
		match tag {
			0 => Queue::Fifo,
			1 => Queue::Main,
			_ => unreachable!("2q-fast-admission-reprieve-compact-hybrid queue tag out of range: {tag}"),
		}
	}
}

/// Per-key bookkeeping is [`NodePayload`], the one node every policy shares.
/// This stack reads `queue`, `tier`, `size` and `dram_resident`; `freq`, `ts`
/// and `phys` belong to other policies and stay at their defaults here.
///
/// `tier` is `None` while `queue == Fifo`: the FIFO is entirely slow-tier, so a
/// key there has no tier of its own to record. `queue` is a bare `u8` in the
/// shared node, so [`Queue`] converts at the boundary.
pub struct TwoQFastAdmissionReprieveCompactHybridStack {
	queues: ArenaQueueSet<NodePayload>,

	k_in: f64,

	fifo_capacity: CacheSize,
	fifo_used: CacheSize,

	fast_capacity: CacheSize,
	fast_used: CacheSize,
	slow_used: CacheSize,

	shared_overhead: CacheSize,

	fast_count: usize,
	main_count: usize,

	/// The least-recently-used FAST key in the main queue.
	main_boundary: Option<HashedKey>,

	migrations: Vec<(HashedKey, Tier)>,
}

impl TwoQFastAdmissionReprieveCompactHybridStack {
	pub fn new(k_in: f64, max_size: CacheSize, fast_capacity: CacheSize) -> Self {
		TwoQFastAdmissionReprieveCompactHybridStack {
			queues: ArenaQueueSet::default(),
			k_in,
			fifo_capacity: (k_in * max_size as f64) as CacheSize,
			fifo_used: 0,
			fast_capacity,
			fast_used: 0,
			slow_used: 0,
			shared_overhead: 0,
			fast_count: 0,
			main_count: 0,
			main_boundary: None,
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

	/// The main queue's share of the fast tier. The FIFO is DRAM-resident here,
	/// so its reservation is carved out of the same budget the main queue
	/// settles against -- the two compete, where in plain 2Q the FIFO is in
	/// PMEM and does not.
	///
	/// Only the MAIN queue's SHARE of `reserved_overhead` is subtracted, not
	/// the whole of it: in this variant the FIFO settles against a budget of
	/// its own and pays the remainder. The non-reprieve stack, whose FIFO is
	/// unpoliced, charges the entire reservation here.
	fn effective_main_fast_capacity(&self) -> CacheSize {
		self.fast_capacity
			.saturating_sub(self.fifo_capacity)
			.saturating_sub(self.reserved_shares().1)
	}

	/// The FIFO's budget net of its share of the metadata reservation. What
	/// `settle_fifo_queue` settles against.
	fn effective_fifo_capacity(&self) -> CacheSize {
		self.fifo_capacity.saturating_sub(self.reserved_shares().0)
	}

	/// Splits `reserved_overhead` between the two queues in proportion to
	/// their fast-tier capacities: `(fifo_share, main_share)`.
	///
	/// `fifo_capacity` is clamped to `fast_capacity` first, so a FIFO
	/// reservation larger than the whole fast tier takes ALL of the overhead
	/// and leaves main none, rather than producing a share above 1. Widened to
	/// `u128` for the multiply: `reserved * fifo_capacity` overflows `u64` at
	/// realistic entry counts.
	fn reserved_shares(&self) -> (CacheSize, CacheSize) {
		let reserved = self.reserved_overhead();

		if self.fast_capacity == 0 {
			return (0, 0);
		}

		let fifo_capacity = self.fifo_capacity.min(self.fast_capacity);
		let fifo_share =
			((reserved as u128 * fifo_capacity as u128) / self.fast_capacity as u128) as CacheSize;
		let main_share = reserved.saturating_sub(fifo_share);

		(fifo_share, main_share)
	}

	fn reserved_overhead(&self) -> CacheSize {
		self.queues.len() as CacheSize * self.shared_overhead
	}

	pub fn tier_of(&self, key: HashedKey) -> Option<Tier> {
		let payload = self.queues.payload(key)?;
		match Queue::from_u8(payload.queue) {
			Queue::Fifo => Some(Tier::Fast),
			Queue::Main => payload.tier,
		}
	}

	fn resize_key(&mut self, key: HashedKey, new_size: ObjectSize, new_resident: u8) {
		let Some(payload) = self.queues.payload_mut(key) else { return };

		let old_migrating = payload.migrating();
		payload.size = new_size;
		payload.dram_resident = new_resident;
		let delta = payload.migrating() as i64 - old_migrating as i64;
		let (queue, tier) = (Queue::from_u8(payload.queue), payload.tier);

		match (queue, tier) {
			(Queue::Fifo, _) => {
				self.fifo_used = (self.fifo_used as i64 + delta).max(0) as CacheSize;
			},

			(Queue::Main, Some(Tier::Fast)) => {
				self.fast_used = (self.fast_used as i64 + delta).max(0) as CacheSize;
			},

			(Queue::Main, Some(Tier::Slow)) => {
				self.slow_used = (self.slow_used as i64 + delta).max(0) as CacheSize;
			},

			(Queue::Main, None) => {},
		}
	}

	fn touch(&mut self, key: HashedKey) {
		match self.queues.payload(key).map(|p| Queue::from_u8(p.queue)) {
			Some(Queue::Fifo) => self.promote_from_fifo(key),
			Some(Queue::Main) => self.touch_main_fast(key),
			None => {},
		}
	}

	/// A hit in the FIFO promotes to the front of main, and to fast.
	///
	/// The slot does not move: this is an unlink from one queue and a relink
	/// into the other, where the stack this replaces removed the key from one
	/// hash-indexed list and inserted it into another.
	fn promote_from_fifo(&mut self, key: HashedKey) {
		let Some(payload) = self.queues.payload(key) else { return };
		let size_bytes = payload.migrating();

		self.queues.move_to_front_of(Q_FIFO, Q_MAIN, key);
		self.fifo_used = self.fifo_used.saturating_sub(size_bytes);

		if let Some(p) = self.queues.payload_mut(key) {
			p.queue = Queue::Main as u8;
			p.tier = Some(Tier::Fast);
		}

		self.fast_used += size_bytes;
		self.fast_count += 1;
		self.main_count += 1;

		if self.main_boundary.is_none() {
			self.main_boundary = Some(key);
		}

		// No migration emitted: the FIFO is already DRAM, so promotion moves
		// bookkeeping rather than bytes.
		self.settle_fast_tier();
	}

	/// Faithful port of `TwoQFastAdmissionReprieveHybridStack::touch_main_fast`,
	/// which is itself unchanged from the non-reprieve stack's.
	fn touch_main_fast(&mut self, key: HashedKey) {
		let previous_tier = self.queues.payload(key).and_then(|p| p.tier);

		let already_at_front = self.queues.front(Q_MAIN) == Some(key);
		let is_boundary = self.main_boundary == Some(key);

		// Read the neighbour BEFORE moving: once the key is at the front its
		// predecessor is gone, and the boundary must step back to whatever was
		// in front of it.
		let new_boundary_if_moved = if is_boundary && !already_at_front {
			self.queues.before(key)
		} else {
			None
		};

		self.queues.move_front(Q_MAIN, key);

		if is_boundary && !already_at_front {
			self.main_boundary = new_boundary_if_moved;
		}

		let mut promoted = false;

		if previous_tier != Some(Tier::Fast) {
			if previous_tier == Some(Tier::Slow) {
				let size = self.queues.payload(key).map(|p| p.migrating()).unwrap_or(0);
				self.slow_used = self.slow_used.saturating_sub(size);
				self.fast_used += size;
				self.fast_count += 1;
				promoted = true;
			}

			if let Some(p) = self.queues.payload_mut(key) {
				p.tier = Some(Tier::Fast);
			}

			if self.main_boundary.is_none() {
				self.main_boundary = Some(key);
			}
		}

		self.settle_fast_tier();

		if promoted && self.queues.payload(key).and_then(|p| p.tier) == Some(Tier::Fast) {
			self.migrations.push((key, Tier::Fast));
		}
	}

	/// Demotes from the tier boundary until `fast_used` is back within the
	/// effective budget. The victim is always `main_boundary`, so nothing is searched.
	fn settle_fast_tier(&mut self) {
		let effective = self.effective_main_fast_capacity();
		let target = drain_target::bytes(effective);

		while self.fast_used > target {
			let Some(demote_key) = self.main_boundary else { break };
			let size = self.queues.payload(demote_key).map(|p| p.migrating()).unwrap_or(0);
			let new_boundary = self.queues.before(demote_key);

			if let Some(p) = self.queues.payload_mut(demote_key) {
				p.tier = Some(Tier::Slow);
			}

			self.fast_used = self.fast_used.saturating_sub(size);
			self.fast_count = self.fast_count.saturating_sub(1);
			self.slow_used += size;
			self.main_boundary = new_boundary;

			self.migrations.push((demote_key, Tier::Slow));
		}
	}

	/// The reprieve, and the whole point of this variant. Splices the FIFO
	/// tail onto the BACK of the main queue as `Tier::Slow` until the FIFO is
	/// back inside its effective budget.
	///
	/// The slot does not move: this is an unlink from `Q_FIFO` and a relink at
	/// the tail of `Q_MAIN`, where the stack this replaces popped one
	/// `HashList` and pushed the other.
	///
	/// `main_boundary` is deliberately untouched. It tracks the least-recently
	/// used FAST key in main, and everything arriving here is slow and lands
	/// behind it, so the boundary is still where it was.
	fn settle_fifo_queue(&mut self) {
		let effective = self.effective_fifo_capacity();

		while self.fifo_used > effective {
			let Some(key) = self.queues.back(Q_FIFO) else { break };

			// Unreachable -- `back` returned the key, so it is indexed. The
			// baseline's `continue` on a missing entry is kept in shape here,
			// dropping the link so the loop cannot spin.
			let Some(payload) = self.queues.payload(key) else {
				self.queues.remove(Q_FIFO, key);
				continue;
			};

			let size = payload.migrating();

			self.queues.move_to_back_of(Q_FIFO, Q_MAIN, key);
			self.fifo_used = self.fifo_used.saturating_sub(size);

			if let Some(p) = self.queues.payload_mut(key) {
				p.queue = Queue::Main as u8;
				p.tier = Some(Tier::Slow);
			}

			self.slow_used += size;
			self.main_count += 1;

			self.migrations.push((key, Tier::Slow));
		}
	}

	fn evict_fifo_tail(&mut self) -> Option<HashedKey> {
		let (key, payload) = self.queues.pop_back(Q_FIFO)?;
		self.fifo_used = self.fifo_used.saturating_sub(payload.migrating());
		Some(key)
	}
}

impl PolicyStack for TwoQFastAdmissionReprieveCompactHybridStack {
	fn is_policy(&self, policy: &PaperPolicy) -> bool {
		matches!(policy, PaperPolicy::TwoQFastAdmissionReprieveCompactHybrid(k_in) if *k_in == self.k_in)
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
			self.touch(key);
			return;
		}

		self.queues.push_front(Q_FIFO, key, NodePayload {
			size,
			dram_resident,
			tier: None,
			phys: None,
			freq: 0,
			ts: 0,
			queue: Queue::Fifo as u8,
		});
		self.fifo_used += (size as CacheSize).saturating_sub(dram_resident as CacheSize);

		// The reprieve: an admission that pushes the FIFO over budget spills
		// its tail into main here, rather than leaving it for an eviction.
		self.settle_fifo_queue();
	}

	fn update(&mut self, key: HashedKey) {
		if self.queues.contains(key) {
			self.touch(key);
		}
	}

	fn remove(&mut self, key: HashedKey) {
		let Some(payload) = self.queues.payload(key) else { return };
		let size = payload.migrating();

		match Queue::from_u8(payload.queue) {
			Queue::Fifo => {
				self.queues.remove(Q_FIFO, key);
				self.fifo_used = self.fifo_used.saturating_sub(size);
			},

			Queue::Main => {
				let new_boundary_if_needed =
					if payload.tier == Some(Tier::Fast) && self.main_boundary == Some(key) {
						self.queues.before(key)
					} else {
						None
					};

				self.queues.remove(Q_MAIN, key);
				self.main_count = self.main_count.saturating_sub(1);

				match payload.tier {
					Some(Tier::Fast) => {
						self.fast_used = self.fast_used.saturating_sub(size);
						self.fast_count = self.fast_count.saturating_sub(1);

						if self.main_boundary == Some(key) {
							self.main_boundary = new_boundary_if_needed;
						}
					},

					Some(Tier::Slow) => {
						self.slow_used = self.slow_used.saturating_sub(size);
					},

					None => {},
				}
			},
		}
	}

	fn resize(&mut self, max_size: CacheSize) {
		self.fifo_capacity = (self.k_in * max_size as f64) as CacheSize;

		// The FIFO reservation is carved out of the fast tier, so moving it
		// changes the main queue's budget. Plain 2Q does not need this.
		self.settle_fast_tier();

		// ... and it moved the FIFO's own budget too, which only this variant
		// settles against.
		self.settle_fifo_queue();
	}

	fn clear(&mut self) {
		self.queues.clear();

		self.fifo_used = 0;
		self.fast_used = 0;
		self.slow_used = 0;
		self.fast_count = 0;
		self.main_count = 0;
		self.main_boundary = None;
		self.migrations.clear();
	}

	fn evict_one(&mut self) -> Option<HashedKey> {
		// Main first; the FIFO tail only once main is empty -- the reverse of
		// the non-reprieve stack. The FIFO is policed by `settle_fifo_queue`,
		// so its tail is not eviction's first choice here.
		if self.queues.queue_len(Q_MAIN) == 0 {
			return self.evict_fifo_tail();
		}

		let (key, payload) = self.queues.pop_back(Q_MAIN)?;
		let size = payload.migrating();
		self.main_count = self.main_count.saturating_sub(1);

		match payload.tier {
			Some(Tier::Fast) => {
				self.fast_used = self.fast_used.saturating_sub(size);
				self.fast_count = self.fast_count.saturating_sub(1);

				if self.main_boundary == Some(key) {
					self.main_boundary = self.queues.back(Q_MAIN);
				}
			},

			Some(Tier::Slow) => {
				self.slow_used = self.slow_used.saturating_sub(size);
			},

			None => {},
		}

		Some(key)
	}

	fn resize_fast_tier(&mut self, size: CacheSize) {
		self.fast_capacity = size;
		self.settle_fast_tier();

		// `reserved_shares` is a function of `fast_capacity`, so the FIFO's
		// effective budget moved as well.
		self.settle_fifo_queue();
	}

	fn drain_tier_migrations(&mut self) -> Vec<(HashedKey, Tier)> {
		std::mem::take(&mut self.migrations)
	}

	fn dram_reserved_bytes(&self) -> CacheSize {
		self.reserved_overhead()
	}

	fn fast_bytes_used(&self) -> CacheSize {
		self.fifo_used + self.fast_used
	}

	fn slow_bytes_used(&self) -> CacheSize {
		self.slow_used
	}

	fn fast_object_count(&self) -> usize {
		self.queues.queue_len(Q_FIFO) + self.fast_count
	}

	fn slow_object_count(&self) -> usize {
		self.main_count - self.fast_count
	}

	// NO `needs_capacity_eviction` override, matching the baseline: the FIFO
	// settles itself, so the trait default (`false`) is correct. The
	// non-reprieve stack overrides it with `fifo_used > fifo_capacity`.
}
