/*
 * Copyright (c) Kia Shakiba
 *
 * This source code is licensed under the GNU AGPLv3 license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! Slab-backed S3-FIFO hybrid with a slow-tier one-access queue whose aged-out
//! keys are *reprieved* into the main queue rather than evicted, and with
//! reference-bit-gated (lazy) demotion at the fast/slow boundary:
//! behaviourally identical to `S3FifoLazyDemotionReprieveHybridStack`, with one
//! structure where that has four.
//!
//! `S3FifoLazyDemotionReprieveHybridStack` keeps THREE `kwik::HashList`s --
//! `one_access_queue`, `main_fast`, `main_slow` -- each owning its OWN
//! key-to-node index, plus a separate `entries` map holding the 8-byte payload.
//! A key is in exactly one of the three lists at any instant (every transition
//! removes before it inserts), so a single [`CompactQueueSet`] holds all three
//! orders over one slab of 16-byte link-only slots, with the payload carried in
//! the one index's value.
//!
//! ## Three queues, not two
//!
//! Every other converted S3-FIFO stack uses two slots of the queue set
//! (`Q_ONE_ACCESS`, `Q_MAIN`) and names the fast/slow split inside the single
//! main queue with a `main_boundary` cursor. This variant's baseline uses two
//! *physical* main lists instead, so the conversion uses three of
//! [`compact_queue_set::MAX_QUEUES`]:
//!
//! ```text
//! Q_ONE_ACCESS  admission queue, entirely slow-tier (PMEM)
//! Q_MAIN_FAST   main queue, fast portion. front = newest, back = demotion candidate
//! Q_MAIN_SLOW   main queue, slow portion. front = the fast/slow boundary,
//!               back = eviction candidate
//! ```
//!
//! There is deliberately NO `main_boundary`, `fast_count` or `main_count`
//! field: the boundary IS `Q_MAIN_SLOW`'s front, and the two counts are
//! `queue_len(Q_MAIN_FAST)` and `queue_len(Q_MAIN_SLOW)`. That is what the
//! baseline does too -- the cursor and counters exist only in the
//! single-main-list designs.
//!
//! ## The three deltas from [`S3FifoCompactHybridStack`]
//!
//! 1. **Reprieve instead of one-access eviction.** `settle_one_access()` runs
//!    synchronously from `insert_resident`/`resize` and splices the one-access
//!    tail to the FRONT of `Q_MAIN_SLOW` -- which is exactly the boundary
//!    position -- instead of dropping it. Nothing is ever evicted from the
//!    one-access queue, so `evict_one` is purely the main queue's tail loop and
//!    `needs_capacity_eviction()` keeps the trait default `false`. Both
//!    structures are PMEM here, so the splice emits **no migration at all**:
//!    the bytes never move.
//! 2. **Lazy demotion.** `settle_fast_tier` is reference-bit gated. A candidate
//!    at `Q_MAIN_FAST`'s back whose `accessed` bit is set is given a fresh start
//!    at the front with the bit cleared (no tier change, no byte movement, no
//!    migration) and the sweep re-evaluates; only a clear bit demotes.
//!    Terminates because each reprieve clears exactly one bit.
//! 3. **No `main_capacity`.** This design derives no budget from
//!    `1 - one_access_ratio` and never gates eviction on main fullness, so
//!    `main_is_full()` and `evict_one_access_tail()` are gone. Its real budgets
//!    are `one_access_capacity` (PMEM, enforced by `settle_one_access`) and
//!    `fast_capacity` (DRAM, enforced by `settle_fast_tier`).
//!
//! ## Promotion is a real move
//!
//! A one-access key's bytes genuinely are in PMEM here, so promoting it to
//! `Q_MAIN_FAST` is a genuine PMEM->DRAM move and must emit the migration --
//! guarded, because the `settle_fast_tier()` that follows may demote the key
//! straight back out in the same call, in which case that call has already
//! pushed the correct `Tier::Slow` migration and `apply_tier_migrations` runs
//! every demotion before any promotion.
//!
//! ## Where the index-value layout earns its keep
//!
//! Same as the rest of this family: `mark_accessed` is the hottest per-get
//! operation -- every hit on a main-queue key does nothing but flip a reference
//! bit -- and it touches no queue order at all. With the payload in the slab it
//! would cost a dereference on every such get for nothing; in the index value
//! it is a single probe. Measured, 59.9 ns against 97.4 ns.
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
		arena_queue_set::{ArenaQueueSet, NodePayload}, narrow_resident, CacheSize,
		HashedKey, PolicyStack, Tier,
	},
	PaperPolicy,
};

const Q_ONE_ACCESS: usize = 0;
const Q_MAIN_FAST: usize = 1;
const Q_MAIN_SLOW: usize = 2;

/// Which live queue a key currently belongs to. `Main` covers both physical
/// main lists; which one is named by the payload's `tier`.
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
/// counter. `tier` and `freq` are only meaningful while `queue == Main` -- the
/// one-access queue is entirely slow-tier and its promotion is eager, so a key
/// there needs no reference bit, and it leaves `tier` at `None`, which is one
/// of the reasons the shared node made that field an `Option`.
///
/// `tier` is redundant with which of the two main lists the key is physically
/// in, but kept for exactly the reason the baseline keeps it: `tier_of()` and
/// the `PolicyWorker` migration path both want it as a cheap single-probe
/// lookup rather than a pair of `contains()` probes.
///
/// `ts` and `phys` belong to other policies; `phys` is set once at admission to
/// match `tier` and never read here.
pub struct S3FifoLazyDemotionReprieveCompactHybridStack {
	/// One slab holding all three orders: `Q_ONE_ACCESS`, `Q_MAIN_FAST`,
	/// `Q_MAIN_SLOW`.
	queues: ArenaQueueSet<NodePayload>,

	one_access_ratio: f64,
	one_access_capacity: CacheSize,
	one_access_used: CacheSize,

	fast_capacity: CacheSize,
	fast_used: CacheSize,
	slow_used: CacheSize,

	/// Approximate per-object DRAM cost of the shared structures (object
	/// hashtable + eviction stack) that hold an entry for every *tracked* key
	/// of both tiers. Reserved out of `fast_capacity` by
	/// `effective_main_fast_capacity()`. `0` unless set via
	/// `with_shared_overhead`.
	shared_overhead: CacheSize,

	migrations: Vec<(HashedKey, Tier)>,
}

impl S3FifoLazyDemotionReprieveCompactHybridStack {
	pub fn new(one_access_ratio: f64, max_size: CacheSize, fast_capacity: CacheSize) -> Self {
		S3FifoLazyDemotionReprieveCompactHybridStack {
			queues: ArenaQueueSet::default(),
			one_access_ratio,
			one_access_capacity: (one_access_ratio * max_size as f64) as CacheSize,
			one_access_used: 0,
			fast_capacity,
			fast_used: 0,
			slow_used: 0,
			shared_overhead: 0,
			migrations: Vec::new(),
		}
	}

	pub fn with_shared_overhead(mut self, overhead: CacheSize) -> Self {
		self.shared_overhead = overhead;


		self
	}

	/// Total DRAM currently reserved for shared per-object metadata:
	/// `tracked key count * shared_overhead`.
	///
	/// `queues.len()` counts *every* tracked key -- the one-access queue and
	/// both main lists alike -- not just the fast ones. A one-access key's
	/// value is PMEM, but its object-hashtable entry and its slab slot and
	/// index bucket are DRAM exactly like a `Q_MAIN_FAST` key's, so it is
	/// charged too. There is no ghost-queue term: this variant has no ghost
	/// queue at all, since no key is ever evicted from the one-access queue.
	fn reserved_overhead(&self) -> CacheSize {
		self.queues.len() as CacheSize * self.shared_overhead
	}

	/// The whole `fast_capacity`, less the shared per-object metadata
	/// reservation, is available to the main queue's fast portion.
	///
	/// The fast-admission variants subtract `one_access_capacity` here, because
	/// there the one-access queue is DRAM-resident and both budgets draw on the
	/// same physical pool. Here the one-access queue lives in PMEM, so it
	/// competes for nothing `Q_MAIN_FAST` wants. `one_access_capacity` still
	/// bounds the one-access queue's own (PMEM) footprint via
	/// `settle_one_access`.
	fn effective_main_fast_capacity(&self) -> CacheSize {
		self.fast_capacity.saturating_sub(self.reserved_overhead())
	}

	pub fn tier_of(&self, key: HashedKey) -> Option<Tier> {
		let payload = self.queues.payload(key)?;

		match Queue::from_u8(payload.queue) {
			Queue::OneAccess => Some(Tier::Slow),
			Queue::Main => payload.tier,
		}
	}

	/// `new_resident` refreshes the payload's DRAM-resident remainder: a re-set
	/// can add or drop a TTL, which changes it by the `Expiries` entry's cost.
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

			// A one-access key really does carry `tier == None` in this stack,
			// but a MAIN-queue key never does: it reaches `Queue::Main` only
			// through `promote_from_one_access` (Fast) or `settle_one_access`
			// (Slow), each of which sets a tier. Unreachable, and spelled out
			// rather than folded into the `_` above.
			(Queue::Main, None) => {},
		}
	}

	fn touch(&mut self, key: HashedKey) {
		match self.queues.payload(key).map(|p| Queue::from_u8(p.queue)) {
			Some(Queue::OneAccess) => self.promote_from_one_access(key),
			Some(Queue::Main) => self.mark_accessed(key),
			None => {},
		}
	}

	/// The hottest per-get operation in this family, and the reason the payload
	/// lives in the index value: one probe, no slab access, no queue movement.
	fn mark_accessed(&mut self, key: HashedKey) {
		if let Some(p) = self.queues.payload_mut(key) {
			p.freq = 1;
		}
	}

	fn promote_from_one_access(&mut self, key: HashedKey) {
		let Some(payload) = self.queues.payload(key) else { return };
		let size_bytes = payload.migrating();

		self.queues.move_to_front_of(Q_ONE_ACCESS, Q_MAIN_FAST, key);
		self.one_access_used = self.one_access_used.saturating_sub(size_bytes);

		if let Some(p) = self.queues.payload_mut(key) {
			p.queue = Queue::Main as u8;
			p.tier = Some(Tier::Fast);
			p.freq = 0;
		}

		self.fast_used += size_bytes;

		self.settle_fast_tier();

		// Unlike the fast-admission variants -- where a one-access entry's
		// bytes are already in DRAM, so promoting it moved nothing -- the
		// bytes genuinely are in PMEM here, so this needs a real promotion
		// migration. Guarded: `settle_fast_tier` above may have demoted this
		// very key straight back out, in which case it already pushed the
		// correct `Tier::Slow` migration and adding `Tier::Fast` after it
		// would leave the bytes in DRAM while the stack believes otherwise.
		if self.queues.payload(key).and_then(|p| p.tier) == Some(Tier::Fast) {
			self.migrations.push((key, Tier::Fast));
		}
	}

	/// The eviction-time second chance: an accessed key at the main tail is
	/// moved to the front of the FAST list with its reference bit cleared,
	/// rather than evicted.
	fn give_second_chance(&mut self, key: HashedKey) {
		let Some(payload) = self.queues.payload(key) else { return };
		let size = payload.migrating();

		match payload.tier {
			// Already fast -- only reachable from `evict_one`'s fast-tail
			// fallback, i.e. nothing has ever been demoted. Just reorder
			// within `Q_MAIN_FAST`: no tier change and no byte movement.
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

			// A main-queue key always carries a tier; this is unreachable.
			// The baseline returns here WITHOUT settling or pushing a
			// migration, so this does too.
			None => return,
		}

		self.settle_fast_tier();

		// Only record a migration if the key actually ended up Fast -- the
		// `settle_fast_tier` above can immediately demote it right back out
		// when the fast tier is at capacity, in which case that call has
		// already pushed the correct `Tier::Slow` migration itself.
		if self.queues.payload(key).and_then(|p| p.tier) == Some(Tier::Fast) {
			self.migrations.push((key, Tier::Fast));
		}
	}

	/// Demotes oldest-first from `Q_MAIN_FAST` into the front of `Q_MAIN_SLOW`
	/// until the fast tier is back within its effective budget.
	///
	/// Lazy demotion: a candidate whose reference bit is set is given a
	/// reprieve (moved to the front of `Q_MAIN_FAST`, bit cleared) instead of
	/// being demoted, and the sweep continues to the next candidate. It changes
	/// no tier, no byte counter and emits no migration. Terminates even when
	/// every fast key's bit is set, since each reprieve clears exactly one bit.
	///
	/// Per-demotion bookkeeping is deliberately untouched: each demoted object
	/// still retags its payload, still moves between the two physical lists,
	/// still moves `fast_used`/`slow_used` by its own size, and still emits
	/// exactly one `Tier::Slow` migration.
	fn settle_fast_tier(&mut self) {
		let effective_capacity = self.effective_main_fast_capacity();

		while self.fast_used > effective_capacity {
			let Some(candidate) = self.queues.back(Q_MAIN_FAST) else { break };

			let accessed = self.queues.payload(candidate).map(|p| p.freq != 0).unwrap_or(false);

			if accessed {
				self.queues.move_front(Q_MAIN_FAST, candidate);

				if let Some(p) = self.queues.payload_mut(candidate) {
					p.freq = 0;
				}

				continue;
			}

			let size = self.queues.payload(candidate).map(|p| p.migrating()).unwrap_or(0);

			// `push_front` of the slow list IS the fast/slow boundary
			// position, so demotion is a plain unlink + relink.
			self.queues.move_to_front_of(Q_MAIN_FAST, Q_MAIN_SLOW, candidate);

			if let Some(p) = self.queues.payload_mut(candidate) {
				p.tier = Some(Tier::Slow);
			}

			self.fast_used = self.fast_used.saturating_sub(size);
			self.slow_used += size;

			self.migrations.push((candidate, Tier::Slow));
		}
	}

	/// Relieves one-access-queue pressure by splicing its tail(s) to the front
	/// of `Q_MAIN_SLOW` -- the boundary position -- rather than evicting them.
	/// Called synchronously from `insert_resident()`/`resize()`, exactly
	/// mirroring `settle_fast_tier()`'s relationship to the fast/slow boundary.
	///
	/// A pure internal migration: nothing is ever removed from the cache here,
	/// so this must never be routed through `evict_one()` /
	/// `needs_capacity_eviction()`.
	///
	/// No migration is pushed. Both the one-access queue and the main queue's
	/// slow segment live in PMEM, so this moves the key between two orders
	/// without moving a single byte -- the whole point of pairing a slow-tier
	/// one-access queue with the reprieve.
	fn settle_one_access(&mut self) {
		while self.one_access_used > self.one_access_capacity {
			let Some(key) = self.queues.back(Q_ONE_ACCESS) else { break };

			let Some(payload) = self.queues.payload(key) else {
				// Unreachable -- `back` returns a key the index holds by
				// construction. Detach it rather than spin, which is what the
				// baseline's `pop_back`-then-`continue` amounts to.
				self.queues.remove(Q_ONE_ACCESS, key);
				continue;
			};

			let size = payload.migrating();

			self.one_access_used = self.one_access_used.saturating_sub(size);

			self.queues.move_to_front_of(Q_ONE_ACCESS, Q_MAIN_SLOW, key);

			if let Some(p) = self.queues.payload_mut(key) {
				p.queue = Queue::Main as u8;
				p.tier = Some(Tier::Slow);
				p.freq = 0;
			}

			self.slow_used += size;
		}
	}
}

impl PolicyStack for S3FifoLazyDemotionReprieveCompactHybridStack {
	fn is_policy(&self, policy: &PaperPolicy) -> bool {
		matches!(policy, PaperPolicy::S3FifoLazyDemotionReprieveCompactHybrid(r) if *r == self.one_access_ratio)
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
			Queue::OneAccess => {
				self.queues.remove(Q_ONE_ACCESS, key);
				self.one_access_used = self.one_access_used.saturating_sub(size);
			},

			Queue::Main => match payload.tier {
				Some(Tier::Fast) => {
					self.queues.remove(Q_MAIN_FAST, key);
					self.fast_used = self.fast_used.saturating_sub(size);
				},

				Some(Tier::Slow) => {
					self.queues.remove(Q_MAIN_SLOW, key);
					self.slow_used = self.slow_used.saturating_sub(size);
				},

				// Unreachable: a main-queue key always carries a tier, and
				// without one there is no list to unlink it from. Leaving the
				// row in place is the only safe response -- `remove(q, key)`
				// on the wrong `q` would corrupt that queue's length.
				None => {},
			},
		}
	}

	fn resize(&mut self, max_size: CacheSize) {
		self.one_access_capacity = (self.one_access_ratio * max_size as f64) as CacheSize;
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

	fn evict_one(&mut self) -> Option<HashedKey> {
		// The one-access queue never reaches here -- its own capacity pressure
		// is relieved synchronously by `settle_one_access()`, the same way the
		// fast/slow boundary is settled by `settle_fast_tier()` rather than
		// through eviction. This is purely the main queue's tail loop.
		loop {
			// The slow tail is the real eviction candidate; fall back to the
			// fast tail only when nothing has ever been demoted.
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
		self.settle_fast_tier();
	}

	fn drain_tier_migrations(&mut self) -> Vec<(HashedKey, Tier)> {
		std::mem::take(&mut self.migrations)
	}

	fn dram_reserved_bytes(&self) -> CacheSize {
		self.reserved_overhead()
	}

	// The one-access queue counts toward the SLOW gauges here, not the fast
	// ones. The fast-admission variants add `one_access_used` to
	// `fast_bytes_used` because their one-access queue really is DRAM; this
	// variant's is PMEM, so attributing it to the fast tier would over-report
	// DRAM by the whole one-access budget. `tier_of` already reports
	// `Tier::Slow` for these keys -- these gauges must agree with it.
	fn fast_bytes_used(&self) -> CacheSize {
		self.fast_used
	}

	fn slow_bytes_used(&self) -> CacheSize {
		self.slow_used + self.one_access_used
	}

	fn fast_object_count(&self) -> usize {
		self.queues.queue_len(Q_MAIN_FAST)
	}

	fn slow_object_count(&self) -> usize {
		self.queues.queue_len(Q_MAIN_SLOW) + self.queues.queue_len(Q_ONE_ACCESS)
	}

	// No `needs_capacity_eviction` override, deliberately: this stack's
	// one-access budget is enforced by `settle_one_access()` (an internal
	// splice), never by eviction. Returning `true` would make `apply_evictions`
	// spin on `evict_one()` -- which only ever drains the MAIN queue -- until
	// the cache emptied. The baseline dropped this override for the same
	// reason.
}

/// The policy string this stack answers to. The parse function slices the
/// prefix off by a HARD-CODED byte offset, and a wrong offset makes every such
/// policy string unparseable at runtime without producing a single compile
/// error -- so it is pinned here rather than assumed.
#[cfg(test)]
mod policy_string_tests {
	use super::*;
	use core::str::FromStr;

	#[test]
	fn policy_string_round_trips() {
		for ratio in [0.0f64, 0.1, 0.25, 0.5, 1.0] {
			let policy = PaperPolicy::S3FifoLazyDemotionReprieveCompactHybrid(ratio);
			let text = policy.to_string();
			assert!(
				text.starts_with("s3-fifo-lazy-demotion-reprieve-compact-hybrid-"),
				"unexpected policy string {text}",
			);
			assert_eq!(
				PaperPolicy::from_str(&text).unwrap(),
				policy,
				"round trip failed for {text}",
			);
		}

		// Not swallowed by the plain `s3-fifo-` guard, which is the prefix
		// this one extends. (This used to also assert the non-compact
		// `s3-fifo-lazy-demotion-reprieve-hybrid-` prefix still parsed to its
		// own variant; that design has been removed from the crate.)
		assert!(matches!(
			PaperPolicy::from_str("s3-fifo-lazy-demotion-reprieve-compact-hybrid-0.25"),
			Ok(PaperPolicy::S3FifoLazyDemotionReprieveCompactHybrid(_)),
		));

		// The reprieve family's bound is INCLUSIVE at both ends, and anything
		// outside it is rejected.
		assert!(PaperPolicy::from_str("s3-fifo-lazy-demotion-reprieve-compact-hybrid-1").is_ok());
		assert!(PaperPolicy::from_str("s3-fifo-lazy-demotion-reprieve-compact-hybrid-1.5").is_err());
		assert!(PaperPolicy::from_str("s3-fifo-lazy-demotion-reprieve-compact-hybrid--0.1").is_err());
		assert!(PaperPolicy::from_str("s3-fifo-lazy-demotion-reprieve-compact-hybrid-abc").is_err());
	}

	/// Every tiered design must report itself as hybrid: `is_hybrid` is a
	/// hand-written `matches!` that cannot produce a compile error when a
	/// variant is missing.
	#[test]
	fn is_reported_as_hybrid() {
		assert!(PaperPolicy::S3FifoLazyDemotionReprieveCompactHybrid(0.1).is_hybrid());
	}
}
