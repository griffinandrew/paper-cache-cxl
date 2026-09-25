/*
 * Copyright (c) Kia Shakiba
 *
 * This source code is licensed under the GNU AGPLv3 license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! Slab-backed 2Q fast-admission hybrid: `TwoQFastAdmissionHybridStack` with
//! one structure where that has three.
//!
//! Identical to [`TwoQCompactHybridStack`] except that the admission queue is
//! DRAM-resident rather than PMEM-resident. That single placement change
//! propagates:
//!
//! - `tier_of` reports `Fast` for a key in the FIFO, not `Slow`.
//! - The FIFO is carved OUT of the fast tier. Its `k_in * max_size` capacity
//!   is a fraction of the CACHE, so the carve-out is that capacity clamped to
//!   what the tier can pay for (`fifo_carve_out()`). The `shared_overhead`
//!   reservation is charged MAIN-FIRST (`reserved_shares`): the main queue
//!   pays it out of `fast_capacity - fifo_carve_out()`, as it always did, and
//!   the FIFO pays only the part that does not fit there. The main queue
//!   settles against `fast_capacity - fifo_carve_out() - main_share`, the
//!   FIFO is policed (through `needs_capacity_eviction`) against
//!   `fifo_carve_out() - fifo_share`, and the two budgets plus the
//!   reservation are `fast_capacity` while the reservation fits in the tier.
//!   Wherever the carve-out and the reservation fit the tier together, both
//!   budgets are exactly the pre-clamp ones. This is deliberately NOT the
//!   proportional split `TwoQFastAdmissionReprieveCompactHybridStack` and the
//!   S3-FIFO fast-admission stacks use; see `reserved_shares`.
//! - A promotion out of the FIFO emits NO migration: the bytes are already in
//!   DRAM, so only the bookkeeping moves.
//! - `resize` must re-settle, because `fifo_capacity` scales with `max_size`
//!   and therefore changes the main queue's budget. Plain 2Q's `resize` does
//!   not need to.
//! - The byte and object counters swap sides: the FIFO counts toward fast.
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
			_ => unreachable!("2q-fast-admission-compact-hybrid queue tag out of range: {tag}"),
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
pub struct TwoQFastAdmissionCompactHybridStack {
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

	/// Whether the last check found `fifo_capacity >= fast_capacity`, i.e.
	/// the one warning for that crossing has been emitted. See
	/// [`Self::warn_if_carve_out_fills_fast_tier`].
	carve_out_fills_fast_tier: bool,

	migrations: Vec<(HashedKey, Tier)>,
}

impl TwoQFastAdmissionCompactHybridStack {
	pub fn new(k_in: f64, max_size: CacheSize, fast_capacity: CacheSize) -> Self {
		TwoQFastAdmissionCompactHybridStack {
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
			carve_out_fills_fast_tier: false,
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

	/// The FIFO's carve-out from the fast tier: its configured capacity, but
	/// never more of the tier than the tier itself holds. The same accessor,
	/// for the same reason, as
	/// `TwoQFastAdmissionReprieveCompactHybridStack::fifo_carve_out`.
	///
	/// `fifo_capacity` is a fraction of the CACHE size (`k_in * max_size`) and
	/// nothing ties it to `fast_capacity` -- the factory hands this stack
	/// `0.2 * max_size` of fast tier, which every `k_in` above 0.2
	/// over-subscribes on its own. The FIFO is DRAM-resident here and its only
	/// limit is `needs_capacity_eviction`, so without this clamp the admission
	/// queue alone could hold more DRAM than the whole fast tier:
	/// `effective_main_fast_capacity` saturates to 0, `settle_fast_tier` --
	/// which governs only the main queue -- has nothing left to demote, and
	/// the real ceiling becomes `max(fast_capacity, k_in * max_size)`.
	/// Measured before this clamp: `k_in` 0.25 held 125 MiB in a 32 MiB tier.
	///
	/// An accessor rather than a value fixed at either assignment site because
	/// `resize` rewrites `fifo_capacity` and `resize_fast_tier` rewrites
	/// `fast_capacity`, each without touching the other.
	fn fifo_carve_out(&self) -> CacheSize {
		self.fifo_capacity.min(self.fast_capacity)
	}

	/// The main queue's share of the fast tier. The FIFO is DRAM-resident here,
	/// so its carve-out comes out of the same budget the main queue settles
	/// against -- the two compete, where in plain 2Q the FIFO is in PMEM and
	/// does not.
	///
	/// Only the MAIN queue's share of `reserved_overhead` is subtracted, and
	/// main pays first: its share is the whole reservation up to everything
	/// the carve-out leaves it, so this is `fast_capacity - fifo_carve_out() -
	/// reserved_overhead()`, saturating at 0 -- the pre-clamp formula, which
	/// it equals for every input because `fast_capacity - fifo_carve_out()`
	/// and `fast_capacity - fifo_capacity` saturate to the same value. The
	/// FIFO pays whatever main could not (`effective_fifo_capacity`).
	fn effective_main_fast_capacity(&self) -> CacheSize {
		self.fast_capacity
			.saturating_sub(self.fifo_carve_out())
			.saturating_sub(self.reserved_shares().1)
	}

	/// The FIFO's budget net of its share of the metadata reservation: what
	/// `needs_capacity_eviction` polices the FIFO against. With the main-first
	/// split that is `min(fifo_carve_out(), fast_capacity -
	/// reserved_overhead())`, saturating at 0: the whole carve-out until the
	/// reservation outgrows main's part of the tier, then what the reservation
	/// leaves of the tier, then nothing once the reservation alone fills it.
	///
	/// At most `fifo_carve_out()`, while the main segment's is at most
	/// `fast_capacity - fifo_carve_out()`, so the two can never sum past the
	/// tier -- and while the reservation fits in it,
	/// `effective_fifo_capacity() + effective_main_fast_capacity() +
	/// reserved_overhead() == fast_capacity` exactly.
	///
	/// A FIFO over this budget is EVICTED from its tail by the worker -- this
	/// variant has no reprieve -- so a carve-out that fills the tier trades the
	/// DRAM overrun for data loss, exactly as the non-reprieve S3-FIFO
	/// fast-admission stacks do.
	fn effective_fifo_capacity(&self) -> CacheSize {
		self.fifo_carve_out().saturating_sub(self.reserved_shares().0)
	}

	/// Splits `reserved_overhead` between the two DRAM segments MAIN-FIRST:
	/// `(fifo_share, main_share)`.
	///
	/// Main pays for the reservation out of the part of the tier the carve-out
	/// leaves it, `fast_capacity - fifo_carve_out()`; the FIFO pays only what
	/// does not fit there. The shares always re-sum to the reservation. Three
	/// regimes, with `carve = fifo_carve_out()` and `R = reserved_overhead()`:
	///
	/// - `R <= fast_capacity - carve`: `(0, R)`. The FIFO's budget is the
	///   whole carve-out and main's is `fast_capacity - carve - R` -- the
	///   budgets this stack had before the clamp, bit for bit, whenever the
	///   carve-out itself fits the tier (above it `R` must be 0 here, and the
	///   only change is the clamp).
	/// - `fast_capacity - carve < R <= fast_capacity`: main's budget is 0 and
	///   the FIFO's is `fast_capacity - R`.
	/// - `R > fast_capacity`: both budgets are 0. The tier is metadata-bound
	///   and the worker evicts each new key from the FIFO on arrival. With no
	///   ghost, a key that comes back is new again, and main is never the
	///   victim, so the stack admits nothing until a delete, an expiry or a
	///   resize lowers `R`.
	///
	/// DELIBERATELY not the proportional split of
	/// `TwoQFastAdmissionReprieveCompactHybridStack` and the four S3-FIFO
	/// fast-admission stacks. They split in proportion before the clamp
	/// reached them, so for them the clamp changed nothing that fits. This
	/// stack charged the whole reservation to main, and a proportional split
	/// would take `R * carve / fast_capacity` of the FIFO's budget in every
	/// configuration with a reservation -- every production run -- where only
	/// the clamp was wanted. Main-first moves this stack's budgets only where
	/// the old ones over-subscribed the tier (`fifo_capacity + R >
	/// fast_capacity`).
	fn reserved_shares(&self) -> (CacheSize, CacheSize) {
		let reserved = self.reserved_overhead();
		let main_share = reserved.min(self.fast_capacity.saturating_sub(self.fifo_carve_out()));

		(reserved - main_share, main_share)
	}

	/// Prints ONE warning to stderr when the configured FIFO (`k_in *
	/// max_size`) is at least the whole fast tier -- the configuration
	/// `fifo_carve_out()` clamps, in which the FIFO takes all of the tier and
	/// the main queue gets no fast segment -- and again only when a later
	/// resize makes that NEWLY true. Returns whether it warned.
	///
	/// `eprintln!`, not `log::warn!`: this crate installs no logger (see
	/// `merged_stack.rs`), so a `log` warning would print nowhere. Stderr is
	/// where the crate's other diagnostics go.
	///
	/// Checked from `resize_fast_tier` and `resize`, not `new`, for the reason
	/// `TwoQFullFastAdmissionCompactHybridStack` gives for its own warning:
	/// `init_policy_stack` builds this stack against a 20%-of-`max_size`
	/// placeholder and `new_hybrid` sends the real budget through
	/// `resize_fast_tier` straight away, so that is where it first arrives.
	fn warn_if_carve_out_fills_fast_tier(&mut self) -> bool {
		let fills = self.fast_capacity > 0 && self.fifo_capacity >= self.fast_capacity;
		let newly = fills && !self.carve_out_fills_fast_tier;
		self.carve_out_fills_fast_tier = fills;

		if newly {
			eprintln!(
				"2q-fast-admission-compact-hybrid: the admission FIFO's configured capacity (k_in * max_size = {} bytes) meets or exceeds the fast-tier budget ({} bytes); the FIFO is clamped to the whole fast tier, so the main queue gets no fast segment and every promotion will demote straight back out. Lower k_in or raise fast_tier_size.",
				self.fifo_capacity,
				self.fast_capacity,
			);
		}

		newly
	}

	/// Metadata reservation for EVERY tracked key, fast or slow: a demotion
	/// moves the value and leaves the key's row, stack node and header in
	/// DRAM. See `PolicyStack::dram_reserved_bytes` for the rule, and for why
	/// a reservation at or over `fast_capacity` is left to saturate.
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

	/// Faithful port of `TwoQFastAdmissionHybridStack::touch_main_fast`.
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

	fn evict_fifo_tail(&mut self) -> Option<HashedKey> {
		let (key, payload) = self.queues.pop_back(Q_FIFO)?;
		self.fifo_used = self.fifo_used.saturating_sub(payload.migrating());
		Some(key)
	}
}

impl PolicyStack for TwoQFastAdmissionCompactHybridStack {
	fn is_policy(&self, policy: &PaperPolicy) -> bool {
		matches!(policy, PaperPolicy::TwoQFastAdmissionCompactHybrid(k_in) if *k_in == self.k_in)
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
		self.warn_if_carve_out_fills_fast_tier();

		// The FIFO reservation is carved out of the fast tier, so moving it
		// changes the main queue's budget. Plain 2Q does not need this. The
		// FIFO's own budget moved too; `needs_capacity_eviction` polices it.
		self.settle_fast_tier();
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
		if let Some(key) = self.evict_fifo_tail() {
			return Some(key);
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
		self.warn_if_carve_out_fills_fast_tier();

		// Both segments' budgets moved: main's is settled here, and the FIFO's
		// -- which only an eviction can shrink in this variant -- is reported
		// over budget through `needs_capacity_eviction` on the worker's next
		// eviction pass.
		self.settle_fast_tier();
	}

	/// `tier_of`: the admission FIFO is DRAM (a promotion out of it moves no
	/// bytes and pushes none), main is placed by its tier.
	/// See `PolicyStack::placement_of`.
	fn placement_of(&self, key: HashedKey) -> Option<Tier> {
		self.tier_of(key)
	}

	fn drain_tier_migrations(&mut self) -> Vec<(HashedKey, Tier)> {
		std::mem::take(&mut self.migrations)
	}

	fn structure_bytes(&self) -> Option<crate::meta::NodeBytes> {
		Some(crate::meta::NodeBytes::stack(self.queues.allocated_bytes()))
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

	/// Against `effective_fifo_capacity()` -- the CLAMPED carve-out net of the
	/// FIFO's share of the reservation -- never the cache-sized
	/// `fifo_capacity` field, which would let the FIFO outgrow the fast tier.
	fn needs_capacity_eviction(&self) -> bool {
		self.fifo_used > self.effective_fifo_capacity()
	}
}

/// The DRAM ceiling, ported with the clamp from
/// `TwoQFastAdmissionReprieveCompactHybridStack`'s module of the same name.
/// Both DRAM segments -- the FIFO and the main queue's fast portion --
/// are budgeted out of one `fast_capacity`, but the FIFO's own capacity
/// is `k_in * max_size`, a fraction of the CACHE.
///
/// `RATIOS` put that at the 4_000 B tier and above it. Only 0.6 and 1.0
/// detect a missing clamp: 0.4 is the equality case, where `min` returns the
/// raw capacity and the clamp is numerically inert -- it is there to show a
/// carve-out that exactly fills the tier does not wedge. `FITTING` (0.25)
/// covers the regime where the carve-out fits and the reservation lands on
/// main first, and pins those budgets to the pre-clamp ones.
#[cfg(test)]
mod dram_ceiling_tests {
	use super::*;

	const MAX_SIZE: CacheSize = 10_000;
	const FAST: CacheSize = 4_000;
	const SIZE: ObjectSize = 100;
	const OVERHEAD: CacheSize = 8;
	const KEYS: HashedKey = 120;

	const SLACK: CacheSize = 0;

	/// `k_in * MAX_SIZE` = 4_000 B (exactly the tier), 6_000 B and
	/// 10_000 B.
	const RATIOS: [f64; 3] = [0.4, 0.6, 1.0];

	/// `k_in * MAX_SIZE` = 2_500 B: the carve-out fits the 4_000 B tier and
	/// leaves main 1_500 B, which the drive's reservation (at most 100 keys at
	/// 8 B) never outgrows.
	const FITTING: f64 = 0.25;

	fn stack(ratio: f64) -> TwoQFastAdmissionCompactHybridStack {
		TwoQFastAdmissionCompactHybridStack::new(ratio, MAX_SIZE, FAST).with_shared_overhead(OVERHEAD)
	}

	/// What `PolicyWorker::apply_evictions` does after every event: evict
	/// while the stack asks for it or the cache is over `max_size`.
	fn evict_while_asked(stack: &mut TwoQFastAdmissionCompactHybridStack) {
		while (stack.needs_capacity_eviction()
			|| stack.fast_bytes_used() + stack.slow_bytes_used() > MAX_SIZE)
			&& stack.evict_one().is_some()
		{}
	}

	/// All the DRAM this stack holds against the tier it was given: both
	/// segments' values (`fast_bytes_used`) plus the metadata reservation it
	/// reports.
	fn assert_within_the_fast_tier(stack: &TwoQFastAdmissionCompactHybridStack, slack: CacheSize, context: &str) {
		let values = stack.fast_bytes_used();
		let reserved = stack.dram_reserved_bytes();

		assert!(
			values + reserved <= stack.fast_capacity + slack,
			"{context}: {} B of DRAM ({values} B of values + {reserved} B reserved) on a {} B fast tier",
			values + reserved,
			stack.fast_capacity,
		);
	}

	/// The budget identity: both DRAM segments' budgets plus ONE reservation
	/// are the fast tier exactly, however far `k_in * max_size`
	/// overshoots it. Unclamped, 0.6 gave the FIFO a 6_000 B budget of its
	/// own on a 4_000 B tier while the main queue's budget saturated to 0.
	#[test]
	fn dram_budgets_never_over_subscribe_the_fast_tier() {
		for ratio in RATIOS {
			let mut stack = stack(ratio);

			for key in 1..=5 {
				stack.insert(key, SIZE);
			}

			let total = stack.effective_fifo_capacity()
				+ stack.effective_main_fast_capacity()
				+ stack.reserved_overhead();

			assert_eq!(
				total, FAST,
				"ratio {ratio}: the two DRAM budgets plus the reservation come to {total} B, not the {FAST} B fast tier",
			);
		}
	}

	/// The ceiling on live bytes, driven the way the worker drives the stack:
	/// admissions, hits on the newest keys (promotions out of the FIFO, into a
	/// main queue with no fast segment left) and on older ones,
	/// re-admissions of early keys, and every eviction the stack asks for.
	/// Includes a carve-out exactly the size of the tier, which must not wedge
	/// or panic.
	#[test]
	fn admissions_and_hits_never_hold_more_dram_than_the_fast_tier() {
		for ratio in RATIOS {
			let mut stack = stack(ratio);

			for key in 1..=KEYS {
				stack.insert(key, SIZE);
				evict_while_asked(&mut stack);
				assert_within_the_fast_tier(&stack, SLACK, &format!("ratio {ratio}, admitted {key}"));

				if key % 3 == 0 {
					for hit in [key - 1, key / 2] {
						if stack.contains(hit) {
							stack.update(hit);
							evict_while_asked(&mut stack);
							assert_within_the_fast_tier(&stack, SLACK, &format!("ratio {ratio}, hit {hit}"));
						}
					}
				}
			}

			for key in 1..=KEYS / 3 {
				stack.insert(key, SIZE);
				evict_while_asked(&mut stack);
				assert_within_the_fast_tier(&stack, SLACK, &format!("ratio {ratio}, re-admitted {key}"));
			}

			assert!(stack.len() > 0, "ratio {ratio}: the stack must still hold keys");
		}
	}

	/// Why the clamp is an accessor and not a value fixed at construction: only
	/// `fast_capacity` moves here, and the FIFO's budget has to move with
	/// it. This variant has no reprieve, so the worker
	/// evicts the excess from the FIFO tail.
	#[test]
	fn shrinking_the_fast_tier_shrinks_the_admission_queue() {
		// 0.25 * 10_000 = 2_500 B of admission queue: fits the 4_000 B tier.
		let mut stack = stack(0.25);

		for key in 1..=5 {
			stack.insert(key, 400);
			evict_while_asked(&mut stack);
		}

		assert_eq!(stack.fast_bytes_used(), 2_000, "all five admitted straight to DRAM");

		stack.resize_fast_tier(1_000);
		evict_while_asked(&mut stack);

		assert_within_the_fast_tier(&stack, 0, "after shrinking the tier to 1_000 B");

		assert_eq!(stack.len(), 2, "the admission queue's excess was evicted from its tail");
	}

	/// The budget identity where the carve-out FITS, with a reservation: main
	/// pays it first, and once it outgrows main's 1_500 B the FIFO pays the
	/// rest, up to a reservation that is the whole tier.
	#[test]
	fn a_fitting_carve_out_keeps_the_budget_identity() {
		// Five keys: 0, 40, 500, 2_000 and 4_000 B reserved.
		for overhead in [0, OVERHEAD, 100, 400, 800] {
			let mut stack = TwoQFastAdmissionCompactHybridStack::new(FITTING, MAX_SIZE, FAST).with_shared_overhead(overhead);

			for key in 1..=5 {
				stack.insert(key, SIZE);
			}

			let total = stack.effective_fifo_capacity()
				+ stack.effective_main_fast_capacity()
				+ stack.reserved_overhead();

			assert_eq!(
				total, FAST,
				"{} B reserved: the two DRAM budgets plus the reservation come to {total} B, not the {FAST} B fast tier",
				stack.reserved_overhead(),
			);
		}
	}

	/// The ceiling on live bytes where the carve-out fits, driven as the
	/// clamped ratios are above.
	///
	/// Main pays the reservation here, and main settles on a promotion, a hit
	/// and a resize -- not on an admission, exactly as before the clamp -- so
	/// an admission the FIFO takes without an eviction adds one `OVERHEAD` to
	/// a total main has not yet settled against. The bound is therefore exact
	/// after every event that settles main and over by at most the metadata of
	/// the keys admitted since; the final re-settle is exact. (In this drive
	/// the headroom `drain_target` leaves absorbs that metadata and no event
	/// goes over the tier at all; the bound is what the code guarantees.)
	#[test]
	fn a_fitting_carve_out_holds_the_ceiling_through_admissions_and_hits() {
		let mut stack = stack(FITTING);
		let mut unsettled: CacheSize = 0;

		for key in 1..=KEYS {
			stack.insert(key, SIZE);
			unsettled += 1;
			evict_while_asked(&mut stack);
			assert_within_the_fast_tier(
				&stack,
				unsettled * OVERHEAD,
				&format!("admitted {key}, {unsettled} admission(s) since main settled"),
			);

			if key % 3 == 0 {
				for hit in [key - 1, key / 2] {
					if stack.contains(hit) {
						// A FIFO hit promotes and a main hit touches: both settle main.
						stack.update(hit);
						unsettled = 0;
						evict_while_asked(&mut stack);
						assert_within_the_fast_tier(&stack, SLACK, &format!("hit {hit}"));
					}
				}
			}
		}

		for key in 1..=KEYS / 3 {
			if stack.contains(key) {
				unsettled = 0;
			} else {
				unsettled += 1;
			}

			stack.insert(key, SIZE);
			evict_while_asked(&mut stack);
			assert_within_the_fast_tier(&stack, unsettled * OVERHEAD, &format!("re-admitted {key}"));
		}

		stack.resize_fast_tier(FAST);
		assert_within_the_fast_tier(&stack, 0, "re-settled");

		assert!(
			stack.reserved_overhead() <= FAST - stack.fifo_carve_out(),
			"the drive left the fitting regime: {} B reserved",
			stack.reserved_overhead(),
		);
		assert!(stack.len() > 0, "the stack must still hold keys");
	}

	/// The main-first split itself, pinned in every regime (five keys each).
	#[test]
	fn main_first_shares_pin_every_regime() {
		// (ratio, overhead per key, (fifo_share, main_share), FIFO budget, main budget)
		let cases: [(f64, CacheSize, (CacheSize, CacheSize), CacheSize, CacheSize); 4] = [
			// Carve-out 2_500 B, 40 B reserved: main pays all of it, and the
			// FIFO keeps its whole carve-out.
			(FITTING, OVERHEAD, (0, 40), 2_500, 1_460),
			// 2_000 B reserved: main can pay 1_500 of it, the FIFO the other 500.
			(FITTING, 400, (500, 1_500), 2_000, 0),
			// 5_000 B reserved, more than the tier: no value budget left.
			(FITTING, 1_000, (3_500, 1_500), 0, 0),
			// Clamped carve-out (6_000 B -> the 4_000 B tier), 40 B reserved:
			// main has no room, so the FIFO pays it all.
			(0.6, OVERHEAD, (40, 0), 3_960, 0),
		];

		for (ratio, overhead, shares, fifo, main) in cases {
			let mut stack = TwoQFastAdmissionCompactHybridStack::new(ratio, MAX_SIZE, FAST).with_shared_overhead(overhead);

			for key in 1..=5 {
				stack.insert(key, SIZE);
			}

			let context = format!("ratio {ratio}, {} B reserved", stack.reserved_overhead());

			assert_eq!(stack.reserved_shares(), shares, "{context}: (fifo_share, main_share)");
			assert_eq!(stack.effective_fifo_capacity(), fifo, "{context}: FIFO budget");
			assert_eq!(stack.effective_main_fast_capacity(), main, "{context}: main budget");
		}
	}

	/// The point of main-first: wherever the reservation fits beside the
	/// carve-out, both budgets are 34c6a4e's. That commit settled main against
	/// `fast_capacity.saturating_sub(fifo_capacity).saturating_sub(
	/// reserved_overhead())` and policed the FIFO against the raw
	/// `fifo_capacity`; both are restated here verbatim. Main's formula holds
	/// for EVERY input; the FIFO's wherever `reserved <= fast - carve`, except
	/// that above the tier (where that forces `reserved == 0`) the FIFO's
	/// budget is the clamp's `fast` rather than the raw capacity.
	#[test]
	fn budgets_equal_the_pre_clamp_formulas_wherever_the_reservation_fits_beside_the_carve_out() {
		let mut checked = 0;

		for ratio in [0.0, 0.1, FITTING, 0.39, 0.4, 0.41, 0.6, 1.0] {
			for fast in [0, 1, 1_000, 2_500, FAST, MAX_SIZE] {
				for overhead in [0, 1, OVERHEAD, 100, 1_000] {
					for keys in [0, 1, 5, 20] {
						let mut stack =
							TwoQFastAdmissionCompactHybridStack::new(ratio, MAX_SIZE, fast).with_shared_overhead(overhead);

						for key in 1..=keys {
							stack.insert(key, 1);
						}

						let fifo_capacity = stack.fifo_capacity;
						let reserved = stack.reserved_overhead();
						let carve = fifo_capacity.min(fast);
						let context = format!("ratio {ratio}, fast {fast}, {reserved} B reserved");

						assert_eq!(
							stack.effective_main_fast_capacity(),
							fast.saturating_sub(fifo_capacity).saturating_sub(reserved),
							"{context}: main's budget moved from 34c6a4e's",
						);

						if reserved > fast - carve {
							continue;
						}

						let pre_clamp_fifo = if fifo_capacity <= fast { fifo_capacity } else { fast };

						assert_eq!(
							stack.effective_fifo_capacity(),
							pre_clamp_fifo,
							"{context}: the FIFO's budget moved from 34c6a4e's",
						);

						checked += 1;
					}
				}
			}
		}

		assert!(checked >= 300, "only {checked} fitting configurations were checked");
	}

	/// A reservation at or over the whole tier. Main's budget and the FIFO's
	/// are both 0, so the worker evicts each new key from the FIFO the moment
	/// it arrives -- it is the FIFO's only member -- while the proven keys in
	/// main keep their place (their values demoted, their metadata still the
	/// reservation). Nothing in main is evicted either, so the stack stays
	/// closed until a delete, an expiry or a resize lowers the reservation.
	/// Before the clamp and the split the FIFO kept its whole raw capacity in
	/// DRAM on top of the reservation.
	#[test]
	fn a_metadata_bound_tier_evicts_each_new_key_on_arrival() {
		let mut stack = TwoQFastAdmissionCompactHybridStack::new(FITTING, MAX_SIZE, FAST).with_shared_overhead(100);

		// Thirty proven keys, promoted into main: 3_000 B reserved, which
		// still fits the 4_000 B tier.
		for key in 1..=30 {
			stack.insert(key, SIZE);
			evict_while_asked(&mut stack);
			stack.update(key);
			evict_while_asked(&mut stack);
		}

		assert_eq!(stack.len(), 30, "every proven key is still tracked");

		// The tier shrinks under the reservation.
		stack.resize_fast_tier(2_000);
		evict_while_asked(&mut stack);

		assert_eq!(stack.reserved_overhead(), 3_000);
		assert_eq!(stack.reserved_shares(), (3_000, 0));
		assert_eq!(stack.effective_fifo_capacity(), 0, "no FIFO budget left");
		assert_eq!(stack.effective_main_fast_capacity(), 0, "no main budget left");
		assert_eq!(stack.fast_bytes_used(), 0, "main demoted every value");

		for key in 100..110 {
			stack.insert(key, SIZE);

			assert!(stack.needs_capacity_eviction(), "key {key}: the FIFO is over its 0 B budget");
			assert_eq!(stack.evict_one(), Some(key), "key {key} is the FIFO's only member, so it goes first");

			evict_while_asked(&mut stack);

			assert_eq!(stack.len(), 30, "key {key}: the proven keys keep their place");
			assert_eq!(stack.fast_bytes_used(), 0, "key {key}: no value is left in DRAM");
		}
	}
}

/// The carve-out warning: one stderr line per crossing of
/// `fifo_capacity >= fast_capacity`, checked from both resize entry points.
#[cfg(test)]
mod carve_out_warning_tests {
	use super::*;

	#[test]
	fn the_carve_out_warning_fires_once_per_crossing() {
		// 0.6 * 1_000 = 600 B of admission queue against a 1_000 B tier: fits.
		let mut stack = TwoQFastAdmissionCompactHybridStack::new(0.6, 1_000, 1_000);
		assert!(!stack.warn_if_carve_out_fills_fast_tier(), "a queue that fits the tier must not warn");

		stack.fast_capacity = 600;
		assert!(stack.warn_if_carve_out_fills_fast_tier(), "a 600 B queue on a 600 B tier covers it and must warn");
		assert!(!stack.warn_if_carve_out_fills_fast_tier(), "once per crossing, not once per check");

		stack.resize_fast_tier(1_000);
		assert!(!stack.carve_out_fills_fast_tier, "resize_fast_tier re-checks: 600 B fits 1_000 B again");

		stack.resize_fast_tier(400);
		assert!(stack.carve_out_fills_fast_tier, "resize_fast_tier re-checks: 600 B covers 400 B");

		stack.resize(500);
		assert!(!stack.carve_out_fills_fast_tier, "resize re-checks: 0.6 * 500 = 300 B fits 400 B");
	}
}
