/*
 * Copyright (c) Kia Shakiba
 *
 * This source code is licensed under the GNU AGPLv3 license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! Slab-backed S3-FIFO lazy-demotion + fast-admission + reprieve hybrid:
//! behaviourally identical to
//! `S3FifoLazyDemotionFastAdmissionReprieveHybridStack`, with one structure
//! where that has four.
//!
//! That stack keeps THREE `kwik::HashList`s -- `one_access_queue`,
//! `main_fast`, `main_slow` -- each owning its OWN key-to-node index, plus a
//! separate `entries` map holding the 8-byte payload. A key is in exactly one
//! of the three queues at any moment, so a single [`CompactQueueSet`] holds
//! all three orders over one slab, with the payload carried in that set's
//! single index value.
//!
//! This family is the one where the index-value layout earns its keep:
//! `mark_accessed` is the hottest per-get operation -- every hit on a
//! main-queue key does nothing but flip a reference bit, touching no queue
//! order at all -- so with the payload in the slab it would cost a
//! dereference on every such get for nothing. In the index value it is a
//! single probe.
//!
//! # What separates this from [`S3FifoCompactHybridStack`]
//!
//! Everything below is preserved byte for byte from the stack this compacts.
//!
//! 1. **The one-access queue is FAST.** `tier_of` reports `Tier::Fast` for a
//!    one-access resident, `fast_bytes_used()`/`fast_object_count()` count it,
//!    and `slow_bytes_used()`/`slow_object_count()` no longer do. Admission is
//!    a cheap DRAM write rather than a synchronous PMEM allocation on the
//!    calling thread (`hybrid_policy::admission_tier` returns `Fast` for a
//!    brand-new key under this policy). One consequence the stack must carry:
//!    `promote_from_one_access` emits NO `Tier::Fast` migration, because the
//!    key's bytes are already physically DRAM. `give_second_chance` keeps its
//!    push -- a key reaching it really can be in PMEM, so that move is real.
//!
//! 2. **The two fast segments share one budget.** `one_access_capacity` is
//!    sized from `max_size` -- the CACHE budget -- so it is a carve-out of
//!    `fast_capacity` only up to what the tier can pay for
//!    (`raw_one_access_capacity()`, the clamp
//!    `S3FifoGhostLazyDemotionFastAdmissionCompactHybridStack` carries);
//!    what is left is the main queue's fast segment (`main_fast_capacity()`).
//!    The shared-metadata reservation is split *proportionally* between the
//!    two (`reserved_shares`, following `LruSizedHybridStack`) so that
//!    `effective_one_access_capacity() + effective_main_fast_capacity() +
//!    reserved_overhead() == fast_capacity` for every ratio, while the
//!    reservation fits in the tier. `main_slow` carries no capacity of its
//!    own, so it has nothing to reserve against.
//!
//! 3. **Demotion is lazy.** `settle_fast_tier` gives a `main_fast` tail whose
//!    reference bit is set a reprieve -- move it to the front of `main_fast`
//!    with the bit cleared, and try the next candidate -- instead of demoting
//!    it.
//!
//! 4. **The one-access tail is reprieved, not evicted.** Once
//!    `one_access_used` exceeds its effective capacity, `settle_one_access`
//!    moves the tail into the FRONT of `main_slow` -- a full life there,
//!    promotable through the ordinary `touch()`/tail-second-chance machinery
//!    -- instead of removing it from the cache. That relief runs
//!    *synchronously* from `insert()`/`resize()`/`resize_fast_tier()`, never
//!    through `evict_one()`/`needs_capacity_eviction()`: `apply_evictions`
//!    unconditionally erases whatever key `evict_one()` returns from the
//!    entire cache, and a reprieve is not an eviction.
//!    `needs_capacity_eviction()` therefore stays at the trait default
//!    `false`, and `evict_one()` is purely the main queue's tail loop.
//!
//! # Three queues, and why the boundary cursor is gone
//!
//! [`S3FifoCompactHybridStack`] keeps the main queue as ONE order with a
//! `main_boundary: Option<HashedKey>` cursor marking the oldest still-fast
//! key; the fast tier is the contiguous prefix up to that cursor and demotion
//! is a pure relabel. That works only while demotion is the sole thing that
//! ever crosses the boundary.
//!
//! A reprieve breaks that premise: it has to insert a node AT the boundary.
//! The stack this compacts solves it by splitting main into two physically
//! separate orders, and the same split is what this stack holds in its slab:
//!
//! * `Q_MAIN_FAST` -- front = newest, back = oldest fast key (the demotion
//!   candidate, previously `main_boundary`).
//! * `Q_MAIN_SLOW` -- front = *exactly* the fast/slow boundary position,
//!   back = the eviction candidate.
//!
//! So every boundary crossing is O(1) and needs no cursor to maintain:
//! demotion is a `Q_MAIN_FAST` -> front of `Q_MAIN_SLOW` move, promotion is
//! the reverse, a one-access reprieve is a move to the front of
//! `Q_MAIN_SLOW`, and eviction is the back of `Q_MAIN_SLOW` (falling back to
//! the back of `Q_MAIN_FAST` only when nothing has ever been demoted). The
//! per-tier counters `fast_count`/`main_count` go with the cursor: with
//! homogeneous lists, `queue_len` IS the count.
//!
//! One deliberate limitation carried over unchanged: `insert()` of a
//! brand-new key grows the tracked-key count (and so the reservation) but
//! only calls `settle_one_access()`, never `settle_fast_tier()`. `main_fast`
//! can therefore sit briefly above its freshly-shrunk effective budget. That
//! is bounded and self-correcting -- `fast_used` only ever grows via
//! `promote_from_one_access()` and `give_second_chance()`, both of which
//! settle immediately.
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
		HashedKey, PolicyStack, Tier, Placement, SetEvent, placed,
	},
	PaperPolicy,
};

const Q_ONE_ACCESS: usize = 0;
const Q_MAIN_FAST: usize = 1;
const Q_MAIN_SLOW: usize = 2;

/// Which live queue a key currently belongs to. `Main` covers both physical
/// main lists; the payload's `tier` says which.
///
/// The shared node stores this as a plain `u8`, so the enum is kept purely for
/// readability and converted at that one boundary: `Queue as u8` on the way in,
/// [`Queue::from_u8`] on the way out. Every match below still reads as the enum.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
enum Queue {
	OneAccess = 0,
	Main = 1,
}

impl Queue {
	/// The inverse of `as u8`. `NodePayload::queue` is only ever written here
	/// from a `Queue as u8`, so the catch-all arm is unreachable and `Main` is
	/// the only value it could stand for.
	fn from_u8(tag: u8) -> Queue {
		match tag {
			0 => Queue::OneAccess,
			_ => Queue::Main,
		}
	}
}

/// Per-key bookkeeping is [`NodePayload`], the one node every policy shares.
///
/// This stack reads `queue` (as [`Queue`]), `tier`, `freq`, `size` and
/// `dram_resident`. `freq` carries the S3-FIFO REFERENCE BIT: set is `freq = 1`
/// and tested as `freq != 0`, since a reference bit is a one-bit frequency
/// counter. `tier` and `freq` are only meaningful while `queue == Main`: the
/// one-access queue is entirely fast-tier in this variant and its promotion is
/// eager, so a key there needs neither and leaves `tier` at `None`, which is
/// one of the reasons the shared node made that field an `Option`. `tier` also
/// names WHICH main list the key is in -- `Some(Tier::Fast)` is `Q_MAIN_FAST`,
/// `Some(Tier::Slow)` is `Q_MAIN_SLOW` -- so the two are never allowed to
/// disagree.
///
/// `ts` and `phys` belong to other policies; `phys` is set once at admission to
/// match `tier` and never read here.
pub struct S3FifoLazyDemotionFastAdmissionReprieveCompactHybridStack {
	queues: ArenaQueueSet<NodePayload>,

	one_access_ratio: f64,
	one_access_capacity: CacheSize,
	one_access_used: CacheSize,

	/// The configured total fast-tier (DRAM) budget, shared between the
	/// one-access queue and the main queue's fast segment. There is no
	/// `main_capacity` in this variant: nothing is sized from `1 - ratio` and
	/// nothing gates eviction on main fullness.
	fast_capacity: CacheSize,
	fast_used: CacheSize,
	slow_used: CacheSize,

	shared_overhead: CacheSize,

	/// Whether the last check found `one_access_capacity >= fast_capacity`,
	/// i.e. the one warning for that crossing has been emitted. See
	/// [`Self::warn_if_carve_out_fills_fast_tier`].
	carve_out_fills_fast_tier: bool,

	migrations: Vec<(HashedKey, Tier)>,

	/// S5: the measured M the policy worker pushed (`set_dram_metadata`),
	/// reserved instead of the per-object reservation; `None` under the
	/// per-object model.
	measured: Option<CacheSize>,
}

impl S3FifoLazyDemotionFastAdmissionReprieveCompactHybridStack {
	pub fn new(one_access_ratio: f64, max_size: CacheSize, fast_capacity: CacheSize) -> Self {
		S3FifoLazyDemotionFastAdmissionReprieveCompactHybridStack {
			queues: ArenaQueueSet::default(),
			one_access_ratio,
			one_access_capacity: (one_access_ratio * max_size as f64) as CacheSize,
			one_access_used: 0,
			fast_capacity,
			fast_used: 0,
			slow_used: 0,
			shared_overhead: 0,
			carve_out_fills_fast_tier: false,
			migrations: Vec::new(),
			measured: None,
		}
	}

	pub fn with_shared_overhead(mut self, overhead: CacheSize) -> Self {
		self.shared_overhead = overhead;


		self
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

	/// S5: moves `key` -- a one-access, main-fast or main-slow key -- to the
	/// FRONT OF THE SLOW LIST with tier slow, its reference bit cleared: where
	/// a STRUCTURAL key (its value larger than an empty fast tier) goes when
	/// the policy would put it at main's front. Main's order is two physical
	/// lists here and a slow key cannot sit in the fast one, so the front of
	/// the slow list -- the seam -- is the nearest place; this design's one
	/// departure from "keeps its place in the order". Returns whether the key
	/// left a FAST placement, which the caller pushes `(key, Slow)` for.
	fn to_slow_front(&mut self, key: HashedKey) -> bool {
		let Some(payload) = self.queues.payload(key) else { return false };
		let size = payload.migrating();

		let from = match (Queue::from_u8(payload.queue), payload.tier) {
			(Queue::OneAccess, _) => Q_ONE_ACCESS,
			(Queue::Main, Some(Tier::Fast)) => Q_MAIN_FAST,
			_ => Q_MAIN_SLOW,
		};

		match from {
			Q_MAIN_SLOW => self.queues.move_front(Q_MAIN_SLOW, key),
			from => self.queues.move_to_front_of(from, Q_MAIN_SLOW, key),
		}

		if let Some(p) = self.queues.payload_mut(key) {
			p.queue = Queue::Main as u8;
			p.tier = Some(Tier::Slow);
			p.freq = 0;
		}

		match from {
			Q_ONE_ACCESS => {
				self.one_access_used = self.one_access_used.saturating_sub(size);
				self.slow_used += size;
			},

			Q_MAIN_FAST => {
				self.fast_used = self.fast_used.saturating_sub(size);
				self.slow_used += size;
			},

			_ => {},
		}

		from == Q_MAIN_FAST || from == Q_ONE_ACCESS
	}

	/// The one-access queue's carve-out as the FAST TIER can pay for it, before
	/// the shared-metadata reservation. The S3-FIFO family's clamp, as
	/// `S3FifoGhostLazyDemotionFastAdmissionCompactHybridStack::raw_one_access_capacity`.
	///
	/// The `one_access_capacity` field is `one_access_ratio * max_size`, a
	/// slice of the CACHE budget. The one-access queue is DRAM here and
	/// `settle_one_access` is the only thing that bounds it, so without the
	/// clamp any ratio above `fast_capacity / max_size` (0.2 at the factory's
	/// default) let it converge on more DRAM than the whole tier, while
	/// `main_fast_capacity()` saturated to 0.
	///
	/// An accessor rather than a value fixed where the field is assigned,
	/// because `resize` moves the field and `resize_fast_tier` moves the
	/// budget, independently. A pure no-op whenever the carve-out already
	/// fits -- `min` returns the raw field unchanged, equality included.
	fn raw_one_access_capacity(&self) -> CacheSize {
		self.one_access_capacity.min(self.fast_capacity)
	}

	/// The main queue's fast-segment budget *before* the shared-metadata
	/// reservation -- `fast_capacity` minus the one-access queue's clamped
	/// carve-out. Kept separate from `effective_main_fast_capacity` so
	/// `reserved_shares` has a reservation-free capacity to proportion
	/// against (using the effective one would be circular).
	fn main_fast_capacity(&self) -> CacheSize {
		self.fast_capacity.saturating_sub(self.raw_one_access_capacity())
	}

	/// Splits `reserved_overhead()` proportionally between this stack's two
	/// independently-capacitied FAST segments -- the one-access queue and the
	/// main queue's fast portion -- returned as `(one_access_share,
	/// main_fast_share)`. `u128` intermediate so the product cannot overflow;
	/// remainder handed to the main segment so the two shares always re-sum
	/// exactly. `(0, 0)` if both capacities are zero.
	fn reserved_shares(&self) -> (CacheSize, CacheSize) {
		let reserved = self.reserved_overhead();

		// Both terms are slices of `fast_capacity` -- the first clamped to it,
		// the second its remainder -- so `total_capacity` IS the fast tier.
		let one_access_capacity = self.raw_one_access_capacity();
		let main_fast_capacity = self.main_fast_capacity();
		let total_capacity = one_access_capacity + main_fast_capacity;

		if total_capacity == 0 {
			return (0, 0);
		}

		let one_access_share =
			((reserved as u128 * one_access_capacity as u128) / total_capacity as u128) as CacheSize;
		let main_fast_share = reserved.saturating_sub(one_access_share);

		(one_access_share, main_fast_share)
	}

	/// The one-access queue's own byte cap after giving up its share of the
	/// shared-metadata reservation: what `settle_one_access` settles a
	/// DRAM-resident queue against. Built on `raw_one_access_capacity()`,
	/// never the cache-sized `one_access_capacity` field.
	fn effective_one_access_capacity(&self) -> CacheSize {
		self.raw_one_access_capacity().saturating_sub(self.reserved_shares().0)
	}

	/// The budget actually available to the main queue's fast segment: raw
	/// `fast_capacity`, minus the one-access carve-out, minus this segment's
	/// share of the shared-metadata reservation. The settle drains to this
	/// number, never to any part of it alone.
	fn effective_main_fast_capacity(&self) -> CacheSize {
		self.main_fast_capacity().saturating_sub(self.reserved_shares().1)
	}

	/// Prints ONE warning to stderr when the configured one-access queue
	/// (`one_access_ratio * max_size`) is at least the whole fast tier -- the
	/// configuration `raw_one_access_capacity()` clamps, in which that queue
	/// takes all of the tier and the main queue gets no fast segment -- and
	/// again only when a later resize makes that NEWLY true. Returns whether it
	/// warned.
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
		let fills = self.fast_capacity > 0 && self.one_access_capacity >= self.fast_capacity;
		let newly = fills && !self.carve_out_fills_fast_tier;
		self.carve_out_fills_fast_tier = fills;

		if newly {
			eprintln!(
				"s3-fifo-lazy-demotion-fast-admission-reprieve-compact-hybrid: the one-access queue's configured capacity (one_access_ratio * max_size = {} bytes) meets or exceeds the fast-tier budget ({} bytes); the queue is clamped to the whole fast tier, so the main queue gets no fast segment and every promotion will demote straight back out. Lower the ratio or raise fast_tier_size.",
				self.one_access_capacity,
				self.fast_capacity,
			);
		}

		newly
	}

	pub fn tier_of(&self, key: HashedKey) -> Option<Tier> {
		let payload = self.queues.payload(key)?;

		match Queue::from_u8(payload.queue) {
			// The one-access queue is DRAM-resident in this variant.
			Queue::OneAccess => Some(Tier::Fast),
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
			(Queue::OneAccess, _) => {
				self.one_access_used = (self.one_access_used as i64 + delta).max(0) as CacheSize;
			},

			(Queue::Main, Some(Tier::Fast)) => {
				self.fast_used = (self.fast_used as i64 + delta).max(0) as CacheSize;
			},

			(Queue::Main, Some(Tier::Slow)) => {
				self.slow_used = (self.slow_used as i64 + delta).max(0) as CacheSize;
			},

			// A one-access key really does carry `tier == None` here, but a
			// MAIN-queue key never does: it reaches `Queue::Main` only through
			// `promote_from_one_access` (Fast) or `settle_one_access` (Slow),
			// each of which sets a tier. Unreachable, and spelled out rather
			// than folded into the `_` above.
			(Queue::Main, None) => {},
		}
	}

	fn touch(&mut self, key: HashedKey, structural: bool) {
		match self.queues.payload(key).map(|p| Queue::from_u8(p.queue)) {
			Some(Queue::OneAccess) => self.promote_from_one_access(key, structural),
			Some(Queue::Main) => self.mark_accessed(key),
			None => {},
		}
	}

	/// The hottest per-get operation in this family, and the reason the
	/// payload lives in the index value: one probe, no slab access, no queue
	/// movement.
	fn mark_accessed(&mut self, key: HashedKey) {
		if let Some(p) = self.queues.payload_mut(key) {
			p.freq = 1;
		}
	}

	/// Moves a re-accessed one-access-queue key to the front of `main_fast`.
	/// Emits no migration for the promotion itself -- the key's bytes are
	/// already physically Fast in this variant.
	fn promote_from_one_access(&mut self, key: HashedKey, structural: bool) {
		let Some(payload) = self.queues.payload(key) else { return };
		let size_bytes = payload.migrating();

		// S5: a STRUCTURAL key goes to the front of the slow list, slow, and is
		// not promoted; if it leaves a fast placement, pushed.
		if structural {
			if self.to_slow_front(key) {
				self.migrations.push((key, Tier::Slow));
			}

			self.settle_fast_tier();
			return;
		}

		self.queues.move_to_front_of(Q_ONE_ACCESS, Q_MAIN_FAST, key);
		self.one_access_used = self.one_access_used.saturating_sub(size_bytes);

		if let Some(p) = self.queues.payload_mut(key) {
			p.queue = Queue::Main as u8;
			p.tier = Some(Tier::Fast);
			p.freq = 0;
		}

		self.fast_used += size_bytes;

		self.settle_fast_tier();
	}

	/// An accessed key at the main tail is reinserted at the front of
	/// `main_fast` with its reference bit cleared, rather than evicted.
	///
	/// Tier-aware, because the two main tiers are now two physical lists: a
	/// still-fast key only moves within `main_fast`, while a slow key leaves
	/// `main_slow` for the front of `main_fast` and its bytes move with it.
	///
	/// This is the one promotion path that STILL pushes a migration: a key
	/// reaching it can genuinely be in PMEM, so moving it back to Fast is a
	/// physical move, not a relabeling.
	fn give_second_chance(&mut self, key: HashedKey) {
		let Some(payload) = self.queues.payload(key) else { return };
		let size = payload.migrating();

		// S5: a STRUCTURAL key's second chance keeps it slow, at the front of
		// the slow list; a fast one (the fast-tail fallback) moves there,
		// pushed.
		if self.structural(size) {
			if self.to_slow_front(key) {
				self.migrations.push((key, Tier::Slow));
			}

			self.settle_fast_tier();
			return;
		}

		match payload.tier {
			Some(Tier::Fast) => {
				self.queues.move_front(Q_MAIN_FAST, key);

				if let Some(p) = self.queues.payload_mut(key) {
					p.freq = 0;
				}
			},

			Some(Tier::Slow) => {
				self.queues.move_to_front_of(Q_MAIN_SLOW, Q_MAIN_FAST, key);

				if let Some(p) = self.queues.payload_mut(key) {
					p.tier = Some(Tier::Fast);
					p.freq = 0;
				}

				self.slow_used = self.slow_used.saturating_sub(size);
				self.fast_used += size;
			},

			// Only a one-access key carries `tier == None`, and this is only
			// ever called on a main-queue key, so it is unreachable. The
			// baseline returns here WITHOUT settling or pushing a migration,
			// so this does too.
			None => return,
		}

		self.settle_fast_tier();

		if self.queues.payload(key).and_then(|p| p.tier) == Some(Tier::Fast) {
			self.migrations.push((key, Tier::Fast));
		}
	}

	/// Demotes the `main_fast` tail while `fast_used` exceeds
	/// `effective_main_fast_capacity()` -- reference-bit gated.
	///
	/// The reference-bit gate is the "lazy demotion": a candidate whose bit
	/// is set is moved to the FRONT of `main_fast` with the bit cleared and
	/// the pass tries the next one. Each reprieve clears a bit, so a pass can
	/// reprieve at most `queue_len(Q_MAIN_FAST)` times before it must demote.
	///
	/// `effective_capacity` is read once, before the loop: a demotion only
	/// moves a key between two lists, so the tracked-key count -- and hence
	/// the reservation and the target -- cannot move underneath the pass.
	/// A `Set`, with the client's placement (S5): the design's insert, with a
	/// STRUCTURAL value placed slow -- see `to_slow_front`. Returns the
	/// placement applied.
	fn insert_with(&mut self, key: HashedKey, size: ObjectSize, dram_resident: ObjectSize, placement: Placement) -> Placement {
		let dram_resident = narrow_resident(dram_resident);
		let migrating = (size as CacheSize).saturating_sub(dram_resident as CacheSize);
		let structural = placement == Placement::Structural || self.structural(migrating);

		if self.queues.contains(key) {
			self.resize_key(key, size, dram_resident);

			// S5: a FAST main key overwritten with a structural value leaves the
			// fast set, to the front of the slow list, pushed.
			if structural && self.queues.payload(key).is_some_and(|p| Queue::from_u8(p.queue) == Queue::Main && p.tier == Some(Tier::Fast)) {
				if self.to_slow_front(key) {
					self.migrations.push((key, Tier::Slow));
				}
			}

			self.touch(key, structural);
			return placed(structural);
		}

		// S5: a STRUCTURAL new key goes where this design sends a key its DRAM
		// one-access queue cannot hold: the front of the slow list -- built
		// slow, nothing pushed, nothing to police.
		if structural {
			self.queues.push_front(
				Q_MAIN_SLOW,
				key,
				NodePayload {
					size,
					freq: 0,
					ts: 0,
					queue: Queue::Main as u8,
					tier: Some(Tier::Slow),
					phys: Some(Tier::Slow),
					dram_resident,
				},
			);
			self.slow_used += migrating;

			return Placement::Structural;
		}

		self.queues.push_front(
			Q_ONE_ACCESS,
			key,
			NodePayload {
				size,
				freq: 0,
				ts: 0,
				queue: Queue::OneAccess as u8,
				tier: None,
				phys: None,
				dram_resident,
			},
		);
		self.one_access_used += (size as CacheSize).saturating_sub(dram_resident as CacheSize);

		self.settle_one_access();

		Placement::Normal
	}

	fn settle_fast_tier(&mut self) {
		let effective_capacity = self.effective_main_fast_capacity();
		let target = drain_target::bytes(effective_capacity);

		while self.fast_used > target {
			let Some(candidate) = self.queues.back(Q_MAIN_FAST) else { break };

			let accessed = self.queues.payload(candidate).map(|p| p.freq != 0).unwrap_or(false);

			if accessed {
				// Reprieve: fresh start at the front instead of demotion.
				self.queues.move_front(Q_MAIN_FAST, candidate);

				if let Some(p) = self.queues.payload_mut(candidate) {
					p.freq = 0;
				}

				continue;
			}

			let size = self.queues.payload(candidate).map(|p| p.migrating()).unwrap_or(0);

			// The front of `main_slow` IS the fast/slow boundary position.
			self.queues.move_to_front_of(Q_MAIN_FAST, Q_MAIN_SLOW, candidate);

			if let Some(p) = self.queues.payload_mut(candidate) {
				p.tier = Some(Tier::Slow);
			}

			self.fast_used = self.fast_used.saturating_sub(size);
			self.slow_used += size;

			self.migrations.push((candidate, Tier::Slow));
		}
	}

	/// Relieves one-access-queue pressure by moving its tail(s) to the front
	/// of `main_slow` -- the fast/slow boundary position -- so the key gets a
	/// full life in the main queue instead of leaving the cache.
	///
	/// Called synchronously from `insert()`/`resize()`/`resize_fast_tier()`,
	/// exactly mirroring `settle_fast_tier()`'s relationship to the fast/slow
	/// boundary. A pure internal migration: nothing is ever removed from the
	/// cache here, so this must never be routed through
	/// `evict_one()`/`needs_capacity_eviction()`.
	///
	/// Budget hoisted out of the loop for the same reason as in
	/// `settle_fast_tier`: a reprieve moves a key between lists, it never
	/// adds or removes one, so the reservation is fixed for the pass.
	fn settle_one_access(&mut self) {
		// At the DRAIN TARGET of the queue's budget (S5), as main rests at
		// the drain target of its own.
		let effective_capacity = drain_target::bytes(self.effective_one_access_capacity());

		while self.one_access_used > effective_capacity {
			let Some(key) = self.queues.back(Q_ONE_ACCESS) else { break };
			let size = self.queues.payload(key).map(|p| p.migrating()).unwrap_or(0);

			self.queues.move_to_front_of(Q_ONE_ACCESS, Q_MAIN_SLOW, key);
			self.one_access_used = self.one_access_used.saturating_sub(size);

			if let Some(p) = self.queues.payload_mut(key) {
				p.queue = Queue::Main as u8;
				p.tier = Some(Tier::Slow);
				p.freq = 0;
			}

			self.slow_used += size;

			self.migrations.push((key, Tier::Slow));
		}
	}
}

impl PolicyStack for S3FifoLazyDemotionFastAdmissionReprieveCompactHybridStack {
	fn is_policy(&self, policy: &PaperPolicy) -> bool {
		matches!(policy, PaperPolicy::S3FifoLazyDemotionFastAdmissionReprieveCompactHybrid(r) if *r == self.one_access_ratio)
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
		self.settle_one_access();
		self.settle_fast_tier();
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
			Queue::OneAccess => {
				self.queues.remove(Q_ONE_ACCESS, key);
				self.one_access_used = self.one_access_used.saturating_sub(size);
			},

			// `tier` names which main list holds the key, so it selects the
			// queue to unlink from. `None` there is unreachable -- a payload
			// only reaches `Queue::Main` through `promote_from_one_access`
			// (Fast) or `settle_one_access` (Slow) -- and is a no-op for the
			// same reason it is in the stack this compacts.
			Queue::Main => match payload.tier {
				Some(Tier::Fast) => {
					self.queues.remove(Q_MAIN_FAST, key);
					self.fast_used = self.fast_used.saturating_sub(size);
				},

				Some(Tier::Slow) => {
					self.queues.remove(Q_MAIN_SLOW, key);
					self.slow_used = self.slow_used.saturating_sub(size);
				},

				None => {},
			},
		}
	}

	fn resize(&mut self, max_size: CacheSize) {
		self.one_access_capacity = (self.one_access_ratio * max_size as f64) as CacheSize;
		self.warn_if_carve_out_fills_fast_tier();

		// Both boundaries move: the one-access cap directly, and the main
		// queue's fast segment because it is what is LEFT of `fast_capacity`
		// after the carve-out.
		self.settle_one_access();
		self.settle_fast_tier();
	}

	fn clear(&mut self) {
		self.queues.clear();

		self.one_access_used = 0;
		self.fast_used = 0;
		self.slow_used = 0;
		self.migrations.clear();
	}

	/// Purely the main queue's tail loop. The one-access queue is never
	/// evicted from -- `settle_one_access` drains it into `main_slow`
	/// instead -- so a cache holding nothing but one-access keys correctly
	/// reports `None` here.
	fn evict_one(&mut self) -> Option<HashedKey> {
		loop {
			let (key, from_slow) = match self.queues.back(Q_MAIN_SLOW) {
				Some(key) => (key, true),
				None => (self.queues.back(Q_MAIN_FAST)?, false),
			};

			let accessed = self.queues.payload(key).map(|p| p.freq != 0).unwrap_or(false);

			if accessed {
				self.give_second_chance(key);
				continue;
			}

			let queue = if from_slow { Q_MAIN_SLOW } else { Q_MAIN_FAST };
			let payload = self.queues.remove(queue, key);
			let size = payload.map(|p| p.migrating()).unwrap_or(0);

			if from_slow {
				self.slow_used = self.slow_used.saturating_sub(size);
			} else {
				self.fast_used = self.fast_used.saturating_sub(size);
			}

			return Some(key);
		}
	}

	fn resize_fast_tier(&mut self, size: CacheSize) {
		self.fast_capacity = size;
		self.warn_if_carve_out_fills_fast_tier();

		// `fast_capacity` bounds the one-access carve-out and re-proportions
		// the reservation, so the one-access queue's budget moved too: settle
		// it first (a reprieve out of it only adds slow-tier bytes, so it can
		// never make the fast/slow settle below harder), exactly as the
		// midpoint variant does.
		self.settle_one_access();
		self.settle_fast_tier();
	}

	/// `tier_of`: the one-access queue is DRAM; a reprieve splices a key into
	/// main's slow segment and pushes it slow at once.
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
		// Total DRAM: the main queue's fast segment plus the one-access
		// queue, both physically Fast in this variant.
		self.fast_used + self.one_access_used
	}

	fn slow_bytes_used(&self) -> CacheSize {
		// The one-access queue no longer touches Slow/PMEM at all.
		self.slow_used
	}

	fn fast_object_count(&self) -> usize {
		self.queues.queue_len(Q_MAIN_FAST) + self.queues.queue_len(Q_ONE_ACCESS)
	}

	fn slow_object_count(&self) -> usize {
		self.queues.queue_len(Q_MAIN_SLOW)
	}
}

/// The DRAM ceiling, ported with the clamp from
/// `TwoQFastAdmissionReprieveCompactHybridStack`'s module of the same name.
/// Both DRAM segments -- the one-access queue and the main queue's fast portion --
/// are budgeted out of one `fast_capacity`, but the one-access queue's own capacity
/// is `one_access_ratio * max_size`, a fraction of the CACHE.
///
/// `RATIOS` put that at the 4_000 B tier and above it. Only 0.6 and 0.9
/// detect a missing clamp: 0.4 is the equality case, where `min` returns the
/// raw capacity and the clamp is numerically inert -- it is there to show a
/// carve-out that exactly fills the tier does not wedge. The shrink test
/// starts inside the tier (0.25) and binds only once the tier shrinks.
#[cfg(test)]
mod dram_ceiling_tests {
	use super::*;

	const MAX_SIZE: CacheSize = 10_000;
	const FAST: CacheSize = 4_000;
	const SIZE: ObjectSize = 100;
	const OVERHEAD: CacheSize = 8;
	const KEYS: HashedKey = 120;

	const SLACK: CacheSize = 0;

	/// `one_access_ratio * MAX_SIZE` = 4_000 B (exactly the tier), 6_000 B and
	/// 9_000 B.
	const RATIOS: [f64; 3] = [0.4, 0.6, 0.9];

	fn stack(ratio: f64) -> S3FifoLazyDemotionFastAdmissionReprieveCompactHybridStack {
		S3FifoLazyDemotionFastAdmissionReprieveCompactHybridStack::new(ratio, MAX_SIZE, FAST).with_shared_overhead(OVERHEAD)
	}

	/// What `PolicyWorker::apply_evictions` does after every event: evict
	/// while the stack asks for it or the cache is over `max_size`.
	fn evict_while_asked(stack: &mut S3FifoLazyDemotionFastAdmissionReprieveCompactHybridStack) {
		while (stack.needs_capacity_eviction()
			|| stack.fast_bytes_used() + stack.slow_bytes_used() > MAX_SIZE)
			&& stack.evict_one().is_some()
		{}
	}

	/// All the DRAM this stack holds against the tier it was given: both
	/// segments' values (`fast_bytes_used`) plus the metadata reservation it
	/// reports.
	fn assert_within_the_fast_tier(stack: &S3FifoLazyDemotionFastAdmissionReprieveCompactHybridStack, slack: CacheSize, context: &str) {
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
	/// are the fast tier exactly, however far `one_access_ratio * max_size`
	/// overshoots it. Unclamped, 0.6 gave the one-access queue a 6_000 B budget of its
	/// own on a 4_000 B tier while the main queue's budget saturated to 0.
	#[test]
	fn dram_budgets_never_over_subscribe_the_fast_tier() {
		for ratio in RATIOS {
			let mut stack = stack(ratio);

			for key in 1..=5 {
				stack.insert(key, SIZE);
			}

			let total = stack.effective_one_access_capacity()
				+ stack.effective_main_fast_capacity()
				+ stack.reserved_overhead();

			assert_eq!(
				total, FAST,
				"ratio {ratio}: the two DRAM budgets plus the reservation come to {total} B, not the {FAST} B fast tier",
			);
		}
	}

	/// The ceiling on live bytes, driven the way the worker drives the stack:
	/// admissions, hits on the newest keys (promotions out of the one-access queue, into a
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
	/// `fast_capacity` moves here, and the one-access queue's budget has to move with
	/// it. `settle_one_access` runs from `resize_fast_tier`
	/// and spills the excess into the slow tier, so nothing is lost.
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

		assert_eq!(stack.len(), 5, "nothing is evicted: the excess is demoted");
		assert_eq!(stack.slow_object_count(), 3, "the three oldest went to PMEM");
	}
}

/// The carve-out warning: one stderr line per crossing of
/// `one_access_capacity >= fast_capacity`, checked from both resize entry points.
#[cfg(test)]
mod carve_out_warning_tests {
	use super::*;

	#[test]
	fn the_carve_out_warning_fires_once_per_crossing() {
		// 0.6 * 1_000 = 600 B of admission queue against a 1_000 B tier: fits.
		let mut stack = S3FifoLazyDemotionFastAdmissionReprieveCompactHybridStack::new(0.6, 1_000, 1_000);
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
