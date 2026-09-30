/*
 * Copyright (c) Kia Shakiba
 *
 * This source code is licensed under the GNU AGPLv3 license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! Slab-backed 2Q fast-admission REPRIEVE hybrid: the compact form of
//! `TwoQFastAdmissionReprieveHybridStack`, one structure where that has three.
//!
//! The admission FIFO is DRAM-resident, so `tier_of` reports `Fast` for a key
//! in it, its reservation is carved OUT of the fast tier, a promotion out of
//! it emits no migration, and its bytes and objects count toward fast.
//!
//! When the FIFO runs over budget this stack REPRIEVES the overflow rather
//! than dropping it: `settle_fifo_queue` splices the FIFO tail onto the BACK of
//! the main queue as `Tier::Slow`, emitting a migration, so an aged-out
//! one-access key gets a second chance in PMEM. (The plain fast-admission 2Q,
//! which let the FIFO grow and asked the caller to evict its tail
//! (`needs_capacity_eviction`), was removed in R2; git history holds it.)
//! Four consequences:
//!
//! - `settle_fifo_queue` runs after every admission and after either resize.
//!   It deliberately does NOT run on the re-set path of `insert_resident`,
//!   which returns early, nor on promotion out of the FIFO, which only ever
//!   lowers `fifo_used`.
//! - `needs_capacity_eviction` is NOT overridden. The FIFO polices itself, so
//!   the trait default (`false`) is the answer: an override asking the caller
//!   to evict from a queue that has already settled would be wrong.
//! - `evict_one` drains the MAIN queue first and reaches the FIFO tail only
//!   when main is empty.
//! - The `shared_overhead` reservation is SPLIT between the two queues in
//!   proportion to their fast-tier capacities (`reserved_shares`), because
//!   both settle against a budget and each has to pay its own share.
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
		HashedKey, PolicyStack, Tier, Placement, SetEvent, fast_at_or_before, placed, prev_fast,
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
/// This stack reads `queue`, `tier`, `size` and `dram_resident`; `freq` and `ts`
/// belong to other policies and stay at their defaults here.
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

	/// Whether the last check found `fifo_capacity >= fast_capacity`, i.e.
	/// the one warning for that crossing has been emitted. See
	/// [`Self::warn_if_carve_out_fills_fast_tier`].
	carve_out_fills_fast_tier: bool,

	migrations: Vec<(HashedKey, Tier)>,

	/// S5: the measured M the policy worker pushed (`set_dram_metadata`),
	/// reserved instead of the per-object reservation; `None` under the
	/// per-object model.
	measured: Option<CacheSize>,
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
			carve_out_fills_fast_tier: false,
			migrations: Vec::new(),
			measured: None,
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
	/// never more of the tier than the tier itself holds.
	///
	/// `fifo_capacity` is a fraction of the CACHE size (`k_in * max_size`) and
	/// nothing ties it to `fast_capacity` -- the factory hands this stack
	/// `0.2 * max_size` of fast tier, which every `k_in` above 0.2
	/// over-subscribes on its own. The FIFO is DRAM-resident here, so without
	/// this clamp the admission queue alone would be entitled to more DRAM than
	/// the whole fast tier: `effective_main_fast_capacity` saturates to 0 and
	/// the real ceiling becomes `max(fast_capacity, k_in * max_size)`.
	///
	/// The clamp lives in an accessor rather than at either assignment site
	/// because the two sides move independently: `resize` rewrites
	/// `fifo_capacity` and `resize_fast_tier` rewrites `fast_capacity`, each
	/// without touching the other, so a value fixed at assignment time goes
	/// stale the moment the other one runs.
	///
	/// A no-op for every configuration that already fits: when
	/// `fifo_capacity <= fast_capacity` this IS `fifo_capacity`.
	fn fifo_carve_out(&self) -> CacheSize {
		self.fifo_capacity.min(self.fast_capacity)
	}

	/// The main queue's share of the fast tier. The FIFO is DRAM-resident here,
	/// so its reservation is carved out of the same budget the main queue
	/// settles against -- the two compete, where in plain 2Q the FIFO is in
	/// PMEM and does not.
	///
	/// Only the MAIN queue's SHARE of `reserved_overhead` is subtracted, not
	/// the whole of it: in this variant the FIFO settles against a budget of
	/// its own and pays the remainder. The non-reprieve stack also subtracts
	/// only main's share, but charges main first (see its `reserved_shares`),
	/// and polices its FIFO by eviction rather than settling it.
	///
	/// Subtracting the CLAMPED carve-out is arithmetically identical to
	/// subtracting the raw one -- `saturating_sub` already floors at zero --
	/// and is written this way so both DRAM segments name one carve-out.
	fn effective_main_fast_capacity(&self) -> CacheSize {
		self.fast_capacity
			.saturating_sub(self.fifo_carve_out())
			.saturating_sub(self.reserved_shares().1)
	}

	/// The FIFO's budget net of its share of the metadata reservation. What
	/// `settle_fifo_queue` settles against.
	///
	/// Built on the CLAMPED carve-out, and that is what holds the two
	/// DRAM-resident segments inside one budget whatever `k_in * max_size`
	/// is: this one is at most `fifo_carve_out()`, the main segment's is at
	/// most `fast_capacity - fifo_carve_out()`, so their sum can never exceed
	/// `fast_capacity` -- and while the reservation itself fits in the tier,
	/// `effective_fifo_capacity() + effective_main_fast_capacity() +
	/// reserved_overhead() == fast_capacity` exactly.
	///
	/// A FIFO over that budget is not stranded: `settle_fifo_queue` reprieves
	/// the excess into main as `Tier::Slow`. That is why the clamp is
	/// actionable in THIS variant, where a fast-admission stack that only
	/// settles its main segment would have nothing left to demote.
	fn effective_fifo_capacity(&self) -> CacheSize {
		self.fifo_carve_out().saturating_sub(self.reserved_shares().0)
	}

	/// Splits `reserved_overhead` between the two queues in proportion to
	/// their fast-tier capacities: `(fifo_share, main_share)`.
	///
	/// It proportions against [`Self::fifo_carve_out`] -- the same clamped
	/// carve-out both effective capacities are built on -- so a FIFO
	/// reservation larger than the whole fast tier takes ALL of the overhead
	/// and leaves main none, rather than producing a share above 1. One clamp
	/// in one place: the split and the budgets cannot disagree about how big
	/// the carve-out is, which is what makes the two shares re-sum to exactly
	/// the reservation the segments then subtract one apiece. Widened to
	/// `u128` for the multiply: `reserved * fifo_capacity` overflows `u64` at
	/// realistic entry counts.
	fn reserved_shares(&self) -> (CacheSize, CacheSize) {
		let reserved = self.reserved_overhead();

		if self.fast_capacity == 0 {
			return (0, 0);
		}

		let fifo_capacity = self.fifo_carve_out();
		let fifo_share =
			((reserved as u128 * fifo_capacity as u128) / self.fast_capacity as u128) as CacheSize;
		let main_share = reserved.saturating_sub(fifo_share);

		(fifo_share, main_share)
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
	/// Checked from `resize_fast_tier` and `resize`, not `new`:
	/// `init_policy_stack` builds this stack against a 20%-of-`max_size`
	/// placeholder and `new_hybrid` sends the real budget through
	/// `resize_fast_tier` straight away, so that is where it first arrives.
	fn warn_if_carve_out_fills_fast_tier(&mut self) -> bool {
		let fills = self.fast_capacity > 0 && self.fifo_capacity >= self.fast_capacity;
		let newly = fills && !self.carve_out_fills_fast_tier;
		self.carve_out_fills_fast_tier = fills;

		if newly {
			eprintln!(
				"2q-fast-admission-reprieve-compact-hybrid: the admission FIFO's configured capacity (k_in * max_size = {} bytes) meets or exceeds the fast-tier budget ({} bytes); the FIFO is clamped to the whole fast tier, so the main queue gets no fast segment and every promotion will demote straight back out. Lower k_in or raise fast_tier_size.",
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
		self.measured.unwrap_or(self.queues.len() as CacheSize * self.shared_overhead)
	}

	/// This stack's eff (S5): the whole fast tier's budget for values -- the
	/// tier's, not a segment's -- before the drain target.
	fn own_eff(&self) -> CacheSize {
		self.fast_capacity.saturating_sub(self.reserved_overhead())
	}

	/// Whether a value of `migrating` bytes is STRUCTURAL (S5): larger than an
	/// empty fast tier. Such a key is placed slow, keeps its place in the
	/// policy's order, and is never promoted while it stays that large. The
	/// stack's own check beside the client's flag, so a key the client placed
	/// normally just before eff moved is placed as the stack's own promotions
	/// would place it.
	fn structural(&self, migrating: CacheSize) -> bool {
		migrating > self.own_eff()
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

	fn touch(&mut self, key: HashedKey, structural: bool) {
		match self.queues.payload(key).map(|p| Queue::from_u8(p.queue)) {
			Some(Queue::Fifo) => self.promote_from_fifo(key, structural),
			Some(Queue::Main) => self.touch_main_fast(key, structural),
			None => {},
		}
	}

	/// A hit in the FIFO promotes to the front of main, and to fast.
	///
	/// The slot does not move: this is an unlink from one queue and a relink
	/// into the other, where the stack this replaces removed the key from one
	/// hash-indexed list and inserted it into another.
	fn promote_from_fifo(&mut self, key: HashedKey, structural: bool) {
		let Some(payload) = self.queues.payload(key) else { return };
		let size_bytes = payload.migrating();

		// S5: a STRUCTURAL key moves to main's front all the same -- its place
		// in the order -- with tier slow, and is not promoted.
		if structural {
			self.queues.move_to_front_of(Q_FIFO, Q_MAIN, key);
			self.fifo_used = self.fifo_used.saturating_sub(size_bytes);

			if let Some(p) = self.queues.payload_mut(key) {
				p.queue = Queue::Main as u8;
				p.tier = Some(Tier::Slow);
			}

			self.slow_used += size_bytes;
			self.main_count += 1;

			// The FIFO is DRAM: the key's placement changed.
			self.migrations.push((key, Tier::Slow));

			self.settle_fast_tier();
			return;
		}

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
	fn touch_main_fast(&mut self, key: HashedKey, structural: bool) {
		let previous_tier = self.queues.payload(key).and_then(|p| p.tier);

		let already_at_front = self.queues.front(Q_MAIN) == Some(key);
		let is_boundary = self.main_boundary == Some(key);

		// Read the neighbour BEFORE moving: once the key is at the front its
		// predecessor is gone, and the boundary must step back to whatever fast
		// key was in front of it (S5: past any structural ones).
		let new_boundary_if_moved = if is_boundary && !already_at_front {
			prev_fast(&self.queues, key)
		} else {
			None
		};

		self.queues.move_front(Q_MAIN, key);

		if is_boundary && !already_at_front {
			self.main_boundary = new_boundary_if_moved;
		}

		// S5: a STRUCTURAL key moves to the front all the same -- its place in
		// the order -- with tier slow: a slow one is not promoted, a fast one
		// leaves the fast set, pushed `(key, Slow)` (its placement changed).
		if structural {
			if previous_tier == Some(Tier::Fast) {
				let size = self.queues.payload(key).map(|p| p.migrating()).unwrap_or(0);
				self.fast_used = self.fast_used.saturating_sub(size);
				self.fast_count = self.fast_count.saturating_sub(1);
				self.slow_used += size;

				if let Some(p) = self.queues.payload_mut(key) {
					p.tier = Some(Tier::Slow);
				}

				if self.main_boundary == Some(key) {
					self.main_boundary = None;
				}

				self.migrations.push((key, Tier::Slow));
			}

			self.settle_fast_tier();
			return;
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
		}

		// Fast, at the front; with no fast key in front of it, the boundary.
		if self.main_boundary.is_none() {
			self.main_boundary = Some(key);
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
			let new_boundary = prev_fast(&self.queues, demote_key);

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
		// At the DRAIN TARGET of the FIFO's budget (S5), as main rests at the
		// drain target of its own.
		let effective = drain_target::bytes(self.effective_fifo_capacity());

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

	/// A `Set`, with the client's placement (S5): an existing key is an access
	/// (`touch`); a new one enters the DRAM FIFO, whose admission spills its
	/// tail into main -- or, when STRUCTURAL, goes straight where that spill
	/// sends a key the FIFO cannot hold: main's back, slow. Returns the
	/// placement applied.
	fn insert_with(&mut self, key: HashedKey, size: ObjectSize, dram_resident: ObjectSize, placement: Placement) -> Placement {
		let dram_resident = narrow_resident(dram_resident);
		let migrating = (size as CacheSize).saturating_sub(dram_resident as CacheSize);
		let structural = placement == Placement::Structural || self.structural(migrating);

		if self.queues.contains(key) {
			self.resize_key(key, size, dram_resident);
			self.touch(key, structural);
			return placed(structural);
		}

		// S5: a STRUCTURAL new key takes a slow place in main instead of the
		// DRAM FIFO: its back, where the reprieve variant sends a key its FIFO
		// cannot hold -- built slow, nothing pushed or settled.
		if structural {
			self.queues.push_back(Q_MAIN, key, NodePayload {
				size,
				dram_resident,
				tier: Some(Tier::Slow),
				freq: 0,
				ts: 0,
				queue: Queue::Main as u8,
			});
			self.slow_used += migrating;
			self.main_count += 1;

			return Placement::Structural;
		}

		self.queues.push_front(Q_FIFO, key, NodePayload {
			size,
			dram_resident,
			tier: None,
			freq: 0,
			ts: 0,
			queue: Queue::Fifo as u8,
		});
		self.fifo_used += migrating;

		// The reprieve: an admission that pushes the FIFO over budget spills
		// its tail into main here, rather than leaving it for an eviction.
		self.settle_fifo_queue();

		Placement::Normal
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
		self.insert_with(key, size, dram_resident, Placement::Normal);
	}

	fn insert_placed(
		&mut self,
		key: HashedKey,
		size: ObjectSize,
		dram_resident: ObjectSize,
		_event: SetEvent,
		placement: Placement,
	) -> Placement {
		self.insert_with(key, size, dram_resident, placement)
	}

	fn set_dram_metadata(&mut self, measured: Option<CacheSize>) {
		self.measured = measured;
	}

	/// S5: every settle, against the current budget (the policy worker's
	/// end-of-pass step).
	fn resettle(&mut self) {
		self.settle_fast_tier();
		self.settle_fifo_queue();
	}

	fn update(&mut self, key: HashedKey) {
		if let Some(payload) = self.queues.payload(key) {
			let structural = self.structural(payload.migrating());
			self.touch(key, structural);
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
						prev_fast(&self.queues, key)
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

				// The boundary was main's tail: the nearest fast key from the
				// new tail (S5: past any structural ones).
				if self.main_boundary == Some(key) {
					self.main_boundary = fast_at_or_before(&self.queues, self.queues.back(Q_MAIN));
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
		self.settle_fast_tier();

		// `reserved_shares` is a function of `fast_capacity`, so the FIFO's
		// effective budget moved as well.
		self.settle_fifo_queue();
	}

	/// `tier_of`: the DRAM admission FIFO as in fast admission; a key
	/// reprieved out of it into main's slow segment is pushed slow at once.
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

	// NO `needs_capacity_eviction` override, matching the baseline: the FIFO
	// settles itself, so the trait default (`false`) is correct. The
	// non-reprieve stack overrides it with
	// `fifo_used > effective_fifo_capacity()`.
}

/// The DRAM ceiling. Both of this stack's fast segments -- the admission FIFO,
/// which `tier_of` reports as `Fast` structurally, and the main queue's fast
/// portion -- are budgeted out of one `fast_capacity`, but the FIFO's own cap
/// is `k_in * max_size`, a fraction of the CACHE that nothing ties to the DRAM
/// budget. These pin the clamp that reconciles them, including that it is
/// invisible to configurations that already fit.
#[cfg(test)]
mod dram_ceiling_tests {
	use super::*;

	/// The invariant itself, on the config the clamp exists for: 0.6 * 1_000 =
	/// 600 B of FIFO against a 400 B fast tier. Before the clamp the FIFO's
	/// budget was 600 while main saturated to 0, for a real ceiling of 600.
	#[test]
	fn dram_segments_never_over_subscribe_the_fast_tier() {
		let stack = TwoQFastAdmissionReprieveCompactHybridStack::new(0.6, 1_000, 400);

		assert_eq!(
			stack.effective_fifo_capacity() + stack.effective_main_fast_capacity(),
			stack.fast_capacity(),
			"the two DRAM segments must fill the fast tier exactly, never exceed it",
		);
	}

	/// The clamp must not double-charge the metadata reservation: each segment
	/// subtracts only its OWN share and the two shares re-sum to
	/// `reserved_overhead()`. Checked in the clamped regime, where the FIFO's
	/// share is the whole reservation and main's is nothing.
	#[test]
	fn clamped_segments_still_split_one_reservation() {
		let mut stack = TwoQFastAdmissionReprieveCompactHybridStack::new(0.6, 1_000, 400)
			.with_shared_overhead(10);

		for key in 1..=5 {
			stack.insert(key, 20);
		}

		assert_eq!(stack.reserved_overhead(), 50, "five tracked keys at 10 B each");
		assert_eq!(
			stack.effective_fifo_capacity()
				+ stack.effective_main_fast_capacity()
				+ stack.reserved_overhead(),
			stack.fast_capacity(),
			"both data budgets plus ONE reservation, never the reservation twice",
		);
	}

	/// The clamp must be invisible to every configuration that already fits --
	/// which is every published sweep, and those runs have to stay identical.
	/// 0.25 * 1_000 = 250 sits inside a 400 B tier, so both budgets are the raw
	/// pre-clamp arithmetic.
	#[test]
	fn a_fitting_carve_out_is_untouched() {
		let stack = TwoQFastAdmissionReprieveCompactHybridStack::new(0.25, 1_000, 400);

		assert_eq!(stack.effective_fifo_capacity(), 250);
		assert_eq!(stack.effective_main_fast_capacity(), 150);
	}

	/// Why the clamp is an accessor and not a value fixed at construction: only
	/// `fast_capacity` moves here, and the FIFO's budget has to move with it.
	/// The reprieve is what makes that enforceable -- the overflow splices into
	/// main as `Tier::Slow` rather than waiting on an eviction that this stack
	/// never asks for.
	///
	/// The FIFO rests at the DRAIN TARGET of its budget since S5, as main does:
	/// four 50 B keys (200 B) fit 0.98 x 250 = 245 B, a fifth would not (it
	/// was five, filling the 250 B budget exactly, before S5).
	#[test]
	fn shrinking_the_fast_tier_spills_the_admission_queue() {
		let mut stack = TwoQFastAdmissionReprieveCompactHybridStack::new(0.25, 1_000, 400);

		for key in 1..=4 {
			stack.insert(key, 50);
		}

		assert_eq!(stack.fast_bytes_used(), 200, "all four admitted straight to DRAM");

		stack.resize_fast_tier(100);

		assert!(
			stack.fast_bytes_used() <= stack.fast_capacity(),
			"fast tier holds {} B on a {} B budget",
			stack.fast_bytes_used(),
			stack.fast_capacity(),
		);
		assert_eq!(stack.slow_object_count(), 3, "the excess is reprieved into PMEM: one 50 B key fits 0.98 x 100 B");
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
		let mut stack = TwoQFastAdmissionReprieveCompactHybridStack::new(0.6, 1_000, 1_000);
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
