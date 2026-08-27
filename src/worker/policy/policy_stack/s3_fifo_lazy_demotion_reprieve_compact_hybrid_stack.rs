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

use crate::{
	object::ObjectSize,
	worker::policy::policy_stack::{
		compact_queue_set::CompactQueueSet, narrow_resident, watermarks, CacheSize, HashedKey,
		PolicyStack, Tier,
	},
	PaperPolicy,
};

const Q_ONE_ACCESS: usize = 0;
const Q_MAIN_FAST: usize = 1;
const Q_MAIN_SLOW: usize = 2;

/// Which live queue a key currently belongs to. `Main` covers both physical
/// main lists; which one is named by the payload's `tier`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Queue {
	OneAccess,
	Main,
}

/// Combined per-key bookkeeping, carried in the index value.
///
/// `tier` and `accessed` are only meaningful while `queue == Main`: the
/// one-access queue is entirely slow-tier and its promotion is eager, so a key
/// there needs no reference bit.
///
/// `tier` is redundant with which of the two main lists the key is physically
/// in, but kept for exactly the reason the baseline keeps it: `tier_of()` and
/// the `PolicyWorker` migration path both want it as a cheap map lookup rather
/// than a pair of `contains()` probes -- and here it is free, riding in the
/// same index bucket the lookup already touched.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct S3FifoLazyDemotionReprievePayload {
	queue: Queue,
	tier: Option<Tier>,
	/// The part of `size` that stays in DRAM in either tier; see `migrating`.
	dram_resident: u8,
	accessed: bool,
	size: ObjectSize,
}

/// Pinned, exactly as `S3FifoEntry` is in the stack this replaces.
const _: () = assert!(
	std::mem::size_of::<S3FifoLazyDemotionReprievePayload>() == 8,
	"S3FifoLazyDemotionReprievePayload grew past 8 bytes",
);

impl S3FifoLazyDemotionReprievePayload {
	fn migrating(&self) -> CacheSize {
		(self.size as CacheSize).saturating_sub(self.dram_resident as CacheSize)
	}
}

pub struct S3FifoLazyDemotionReprieveCompactHybridStack {
	/// One slab holding all three orders: `Q_ONE_ACCESS`, `Q_MAIN_FAST`,
	/// `Q_MAIN_SLOW`.
	queues: CompactQueueSet<S3FifoLazyDemotionReprievePayload>,

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
			queues: CompactQueueSet::default(),
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

	/// Also pre-sizes the slab from the DRAM-budget ceiling, capped -- see
	/// `MAX_PREALLOC_ENTRIES` for why the ceiling alone is not safe.
	pub fn with_shared_overhead(mut self, overhead: CacheSize) -> Self {
		self.shared_overhead = overhead;

		if overhead > 0 {
			let ceiling = (self.fast_capacity / overhead) as usize;
			self.queues.reserve(ceiling.min(super::MAX_PREALLOC_ENTRIES));
		}

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

		match payload.queue {
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
		let (queue, tier) = (payload.queue, payload.tier);

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

			(Queue::Main, None) => {},
		}
	}

	fn touch(&mut self, key: HashedKey) {
		match self.queues.payload(key).map(|p| p.queue) {
			Some(Queue::OneAccess) => self.promote_from_one_access(key),
			Some(Queue::Main) => self.mark_accessed(key),
			None => {},
		}
	}

	/// The hottest per-get operation in this family, and the reason the payload
	/// lives in the index value: one probe, no slab access, no queue movement.
	fn mark_accessed(&mut self, key: HashedKey) {
		if let Some(p) = self.queues.payload_mut(key) {
			p.accessed = true;
		}
	}

	fn promote_from_one_access(&mut self, key: HashedKey) {
		let Some(payload) = self.queues.payload(key) else { return };
		let size_bytes = payload.migrating();

		self.queues.move_to_front_of(Q_ONE_ACCESS, Q_MAIN_FAST, key);
		self.one_access_used = self.one_access_used.saturating_sub(size_bytes);

		if let Some(p) = self.queues.payload_mut(key) {
			p.queue = Queue::Main;
			p.tier = Some(Tier::Fast);
			p.accessed = false;
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
					p.accessed = false;
				}
			},

			Some(Tier::Slow) => {
				self.queues.move_to_front_of(Q_MAIN_SLOW, Q_MAIN_FAST, key);

				if let Some(p) = self.queues.payload_mut(key) {
					p.tier = Some(Tier::Fast);
					p.accessed = false;
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
	/// until the fast tier is back under the shared *low* watermark -- but only
	/// once usage has crossed the shared *high* watermark in the first place.
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

		if self.fast_used <= watermarks::high_bytes(effective_capacity) {
			return;
		}

		let drain_target = watermarks::low_bytes(effective_capacity);

		while self.fast_used > drain_target {
			let Some(candidate) = self.queues.back(Q_MAIN_FAST) else { break };

			let accessed = self.queues.payload(candidate).map(|p| p.accessed).unwrap_or(false);

			if accessed {
				self.queues.move_front(Q_MAIN_FAST, candidate);

				if let Some(p) = self.queues.payload_mut(candidate) {
					p.accessed = false;
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
				p.queue = Queue::Main;
				p.tier = Some(Tier::Slow);
				p.accessed = false;
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
			S3FifoLazyDemotionReprievePayload {
				queue: Queue::OneAccess,
				tier: None,
				dram_resident,
				accessed: false,
				size,
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

		match payload.queue {
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

			let accessed = self.queues.payload(key).map(|p| p.accessed).unwrap_or(false);

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

		// Not swallowed by the plain `s3-fifo-` guard, nor by the baseline's
		// `s3-fifo-lazy-demotion-reprieve-hybrid-` one.
		assert!(matches!(
			PaperPolicy::from_str("s3-fifo-lazy-demotion-reprieve-compact-hybrid-0.25"),
			Ok(PaperPolicy::S3FifoLazyDemotionReprieveCompactHybrid(_)),
		));
		assert!(matches!(
			PaperPolicy::from_str("s3-fifo-lazy-demotion-reprieve-hybrid-0.25"),
			Ok(PaperPolicy::S3FifoLazyDemotionReprieveHybrid(_)),
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

/// Fidelity against `S3FifoLazyDemotionReprieveHybridStack`, which this stack
/// is a compaction of.
#[cfg(all(test, feature = "s3_fifo_lazy_demotion_reprieve_hybrid_cache"))]
mod fidelity_tests {
	use super::*;
	use crate::worker::policy::policy_stack::s3_fifo_lazy_demotion_reprieve_hybrid_stack::S3FifoLazyDemotionReprieveHybridStack;

	const MAX: CacheSize = 1_000_000;

	/// Repeats are essential here: a key only leaves the one-access queue by
	/// promotion on a SECOND access, and the reference bit only matters on a
	/// third, so a workload without reuse would exercise neither the main
	/// queue's fast list nor either reprieve path.
	fn skewed_ops() -> Vec<(HashedKey, ObjectSize)> {
		let mut ops = Vec::new();
		let mut x: u64 = 0x243F_6A88_85A3_08D3;
		for _ in 0..20_000 {
			x ^= x << 13;
			x ^= x >> 7;
			x ^= x << 17;
			let u = (x >> 11) as f64 / (1u64 << 53) as f64;
			ops.push((((u * u * 200.0) as u64) + 1, 1024));
		}
		ops
	}

	fn replay(
		ratio: f64,
		fast: CacheSize,
		overhead: CacheSize,
		ops: &[(HashedKey, ObjectSize)],
	) -> (Vec<(HashedKey, Tier)>, Vec<(HashedKey, Tier)>, Vec<Option<Tier>>, Vec<Option<Tier>>) {
		let mut a =
			S3FifoLazyDemotionReprieveHybridStack::new(ratio, MAX, fast).with_shared_overhead(overhead);
		let mut b = S3FifoLazyDemotionReprieveCompactHybridStack::new(ratio, MAX, fast)
			.with_shared_overhead(overhead);
		let (mut ma, mut mb) = (Vec::new(), Vec::new());

		for (k, size) in ops {
			if a.contains(*k) { a.update(*k); } else { a.insert(*k, *size); }
			if b.contains(*k) { b.update(*k); } else { b.insert(*k, *size); }
			ma.extend(a.drain_tier_migrations());
			mb.extend(b.drain_tier_migrations());
		}

		let keys: Vec<HashedKey> = ops.iter().map(|(k, _)| *k).collect();
		let ta = keys.iter().map(|k| a.tier_of(*k)).collect();
		let tb = keys.iter().map(|k| b.tier_of(*k)).collect();
		(ma, mb, ta, tb)
	}

	#[test]
	fn matches_baseline_migration_for_migration() {
		let ops = skewed_ops();
		// 0.01 is in the grid deliberately: at 0.1 and above the one-access
		// budget is never exceeded by this workload, so the replay alone would
		// pass with `settle_one_access` missing entirely. At 0.01 the reprieve
		// fires constantly.
		for ratio in [0.01f64, 0.1, 0.25, 0.5] {
			for fast in [8_192u64, 32_768, 131_072] {
				for overhead in [0u64, 112] {
					let (ma, mb, ta, tb) = replay(ratio, fast, overhead, &ops);
					assert_eq!(ta, tb, "tiers diverge at ratio {ratio} fast {fast} overhead {overhead}");
					assert_eq!(ma, mb, "migrations diverge at ratio {ratio} fast {fast} overhead {overhead}");
				}
			}
		}
	}

	/// Eviction is where the reference bit is acted on at the SLOW tail: an
	/// accessed key there is spliced back into the fast list instead of being
	/// evicted, which reorders both lists mid-eviction. Nothing above
	/// exercises that.
	#[test]
	fn evicts_in_the_same_order_including_second_chances() {
		let ops = skewed_ops();
		let mut a =
			S3FifoLazyDemotionReprieveHybridStack::new(0.25, MAX, 32_768).with_shared_overhead(112);
		let mut b = S3FifoLazyDemotionReprieveCompactHybridStack::new(0.25, MAX, 32_768)
			.with_shared_overhead(112);
		for (k, size) in &ops {
			if a.contains(*k) { a.update(*k); } else { a.insert(*k, *size); }
			if b.contains(*k) { b.update(*k); } else { b.insert(*k, *size); }
			a.drain_tier_migrations();
			b.drain_tier_migrations();
		}
		assert_eq!(a.needs_capacity_eviction(), b.needs_capacity_eviction());
		assert!(!b.needs_capacity_eviction(), "this design never evicts for capacity");

		let mut ea = Vec::new();
		let mut eb = Vec::new();
		while let Some(k) = a.evict_one() { ea.push(k); }
		while let Some(k) = b.evict_one() { eb.push(k); }
		assert_eq!(ea, eb, "eviction order diverges");
		// Whatever the baseline leaves behind (the one-access queue is not
		// drained by `evict_one` in this design), the compact stack must leave
		// exactly the same.
		assert_eq!(a.len(), b.len(), "residual population diverges");
		assert_eq!(a.slow_object_count(), b.slow_object_count());
	}

	#[test]
	fn removal_matches_across_all_three_queues() {
		let ops = skewed_ops();
		let mut a =
			S3FifoLazyDemotionReprieveHybridStack::new(0.25, MAX, 32_768).with_shared_overhead(112);
		let mut b = S3FifoLazyDemotionReprieveCompactHybridStack::new(0.25, MAX, 32_768)
			.with_shared_overhead(112);
		let (mut ma, mut mb) = (Vec::new(), Vec::new());

		for (i, (k, size)) in ops.iter().enumerate() {
			if a.contains(*k) { a.update(*k); } else { a.insert(*k, *size); }
			if b.contains(*k) { b.update(*k); } else { b.insert(*k, *size); }
			if i % 97 == 0 {
				let victim = (i as u64 % 200) + 1;
				a.remove(victim);
				b.remove(victim);
			}
			ma.extend(a.drain_tier_migrations());
			mb.extend(b.drain_tier_migrations());
		}

		assert_eq!(ma, mb, "migrations diverge under removal");
		assert_eq!(a.len(), b.len());
		assert_eq!(a.fast_object_count(), b.fast_object_count());
		assert_eq!(a.slow_object_count(), b.slow_object_count());
		assert_eq!(a.fast_bytes_used(), b.fast_bytes_used());
		assert_eq!(a.slow_bytes_used(), b.slow_bytes_used());
		assert_eq!(a.dram_reserved_bytes(), b.dram_reserved_bytes());
	}

	/// Resize in both directions with BRAND-NEW keys afterwards. The shape is
	/// load-bearing: the equivalent test on the LFU conversion passed with a
	/// real bug present because the workload had no new keys after the resize.
	/// `resize` here rescales `one_access_capacity` and then runs BOTH settle
	/// passes, which `resize_fast_tier` does not.
	#[test]
	fn resizes_like_the_baseline() {
		for (start, resized) in [(65_536u64, 65_536u64), (131_072, 32_768), (32_768, 131_072)] {
			let mut a = S3FifoLazyDemotionReprieveHybridStack::new(0.25, MAX, start)
				.with_shared_overhead(112);
			let mut b = S3FifoLazyDemotionReprieveCompactHybridStack::new(0.25, MAX, start)
				.with_shared_overhead(112);
			let (mut ma, mut mb) = (Vec::new(), Vec::new());

			for i in 0..4_000u64 {
				let k = (i % 200) + 1;
				if a.contains(k) { a.update(k); } else { a.insert(k, 1024); }
				if b.contains(k) { b.update(k); } else { b.insert(k, 1024); }
				ma.extend(a.drain_tier_migrations());
				mb.extend(b.drain_tier_migrations());
			}

			a.resize_fast_tier(resized);
			b.resize_fast_tier(resized);
			a.resize(MAX / 2);
			b.resize(MAX / 2);
			ma.extend(a.drain_tier_migrations());
			mb.extend(b.drain_tier_migrations());
			assert_eq!(a.needs_capacity_eviction(), b.needs_capacity_eviction());

			for i in 0..2_000u64 {
				let k = 10_000 + i;
				a.insert(k, 1024);
				b.insert(k, 1024);
				ma.extend(a.drain_tier_migrations());
				mb.extend(b.drain_tier_migrations());
			}

			assert_eq!(ma, mb, "migrations diverge resizing {start} -> {resized}");
			for i in 0..2_000u64 {
				assert_eq!(a.tier_of(10_000 + i), b.tier_of(10_000 + i));
			}
			assert_eq!(a.fast_object_count(), b.fast_object_count());
			assert_eq!(a.slow_object_count(), b.slow_object_count());
		}
	}

	/// The defining S3-FIFO behaviours: a first access admits to the one-access
	/// queue (slow), a second promotes to main and fast, and a third only sets
	/// the reference bit -- it must NOT reorder or migrate.
	#[test]
	fn admission_promotion_and_reference_bit() {
		let mut a = S3FifoLazyDemotionReprieveHybridStack::new(0.25, MAX, 131_072)
			.with_shared_overhead(0);
		let mut b = S3FifoLazyDemotionReprieveCompactHybridStack::new(0.25, MAX, 131_072)
			.with_shared_overhead(0);

		a.insert(1, 1024);
		b.insert(1, 1024);
		assert_eq!(a.tier_of(1), Some(Tier::Slow));
		assert_eq!(b.tier_of(1), a.tier_of(1));

		a.update(1);
		b.update(1);
		assert_eq!(a.tier_of(1), Some(Tier::Fast));
		assert_eq!(b.tier_of(1), a.tier_of(1));
		assert_eq!(a.drain_tier_migrations(), b.drain_tier_migrations());

		// third access: reference bit only
		a.update(1);
		b.update(1);
		assert_eq!(a.drain_tier_migrations(), b.drain_tier_migrations());
		assert!(a.drain_tier_migrations().is_empty());
		assert_eq!(a.fast_object_count(), b.fast_object_count());
	}

	// ---------------------------------------------------------------------
	// The two deltas that make this variant what it is. Each of these fails
	// if the corresponding behaviour were pattern-matched from
	// `S3FifoCompactHybridStack` instead of ported from this baseline.
	// ---------------------------------------------------------------------

	/// DELTA 1 -- the reprieve. A key that ages out of the one-access queue is
	/// spliced into the main queue's SLOW segment, not evicted, and the splice
	/// emits NO migration (both structures are PMEM, so no byte moves).
	///
	/// `S3FifoCompactHybridStack` would have dropped these keys from
	/// `evict_one`'s one-access tail instead, so `contains` would be false for
	/// every aged-out key and `len()` would be bounded by the one-access
	/// capacity.
	#[test]
	fn one_access_overflow_is_reprieved_not_evicted() {
		// ratio 0.01 of 1_000_000 = a 10_000-byte one-access budget: ~9 keys
		// of 1024 bytes fit, and everything after that must be reprieved.
		let mut a = S3FifoLazyDemotionReprieveHybridStack::new(0.01, MAX, 1_000_000)
			.with_shared_overhead(0);
		let mut b = S3FifoLazyDemotionReprieveCompactHybridStack::new(0.01, MAX, 1_000_000)
			.with_shared_overhead(0);

		for k in 1..=100u64 {
			a.insert(k, 1024);
			b.insert(k, 1024);
		}

		// Nothing was evicted: every key inserted is still tracked.
		assert_eq!(a.len(), 100, "baseline dropped keys it should have reprieved");
		assert_eq!(b.len(), a.len(), "population diverges after one-access overflow");

		for k in 1..=100u64 {
			assert!(b.contains(k), "key {k} was evicted instead of reprieved");
			assert_eq!(b.tier_of(k), a.tier_of(k), "tier diverges for key {k}");
			// Nothing ever reached the fast tier: promotion needs a second
			// access, and the reprieve lands in the SLOW segment.
			assert_eq!(b.tier_of(k), Some(Tier::Slow));
		}

		// The splice moved no bytes, so it pushed no migration at all.
		let ma = a.drain_tier_migrations();
		let mb = b.drain_tier_migrations();
		assert_eq!(ma, mb);
		assert!(mb.is_empty(), "the one-access reprieve must emit no migration");

		// The reprieved keys are in the main queue's slow list, and the
		// one-access queue is back under its budget.
		assert_eq!(a.fast_object_count(), 0);
		assert_eq!(b.fast_object_count(), a.fast_object_count());
		assert_eq!(b.slow_object_count(), 100);
		assert_eq!(b.slow_object_count(), a.slow_object_count());
		assert_eq!(b.slow_bytes_used(), a.slow_bytes_used());

		// And `evict_one` drains the MAIN queue oldest-first -- the reprieved
		// keys -- in exactly the baseline's order. This is the assertion that
		// makes the test non-vacuous: without `settle_one_access` the main
		// queue would be empty and every `evict_one` would return `None`.
		let mut ea = Vec::new();
		let mut eb = Vec::new();
		for _ in 0..20 {
			ea.push(a.evict_one());
			eb.push(b.evict_one());
		}
		assert_eq!(ea, eb, "eviction order over reprieved keys diverges");
		assert_eq!(
			eb,
			(1..=20u64).map(Some).collect::<Vec<_>>(),
			"the reprieved keys must evict oldest-first out of the main queue",
		);
		assert_eq!(b.len(), 80);
		assert_eq!(a.len(), b.len());
	}

	/// DELTA 2 -- lazy demotion. A fast key whose reference bit is set is NOT
	/// demoted when the fast tier overflows: it is reprieved to the front of
	/// the fast list with the bit cleared, and an unaccessed key is demoted
	/// instead. Under `S3FifoCompactHybridStack`'s unconditional
	/// `main_boundary` demotion the accessed key would be the one to go.
	#[test]
	fn demotion_is_reference_bit_gated() {
		const SIZE: ObjectSize = 1024;
		const FAST: CacheSize = 10 * 1024;

		// How many objects fit under the HIGH watermark, computed rather than
		// assumed: the watermarks are env-tunable, and `FAST_TIER_*=1.0`
		// legitimately fits one more object than the defaults do.
		let fits = (watermarks::high_bytes(FAST) / SIZE as CacheSize) as u64;
		assert!(fits >= 2, "test needs at least two fast keys");

		// ratio 0.9 of MAX is a 900_000-byte one-access budget: nothing is
		// reprieved out of it here, so the only pressure is the fast tier's.
		let mut a =
			S3FifoLazyDemotionReprieveHybridStack::new(0.9, MAX, FAST).with_shared_overhead(0);
		let mut b =
			S3FifoLazyDemotionReprieveCompactHybridStack::new(0.9, MAX, FAST).with_shared_overhead(0);

		// Promote `fits` keys into the fast tier, oldest first, staying under
		// the high watermark so nothing has demoted yet.
		for k in 1..=fits {
			a.insert(k, SIZE);
			b.insert(k, SIZE);
			a.update(k);
			b.update(k);
		}
		a.drain_tier_migrations();
		b.drain_tier_migrations();
		assert_eq!(a.fast_object_count(), fits as usize, "no demotion should have fired yet");
		assert_eq!(b.fast_object_count(), a.fast_object_count());

		// Set the reference bit on the OLDEST fast key -- the one an
		// unconditional (non-lazy) demotion would take.
		a.update(1);
		b.update(1);

		// Force a demotion pass by promoting one more key.
		let extra = fits + 1;
		a.insert(extra, SIZE);
		b.insert(extra, SIZE);
		a.update(extra);
		b.update(extra);

		let ma = a.drain_tier_migrations();
		let mb = b.drain_tier_migrations();
		assert_eq!(ma, mb, "migrations diverge across the gated demotion");

		// Key 1 was spared by its reference bit; key 2 -- the next candidate,
		// bit clear -- went instead.
		assert_eq!(a.tier_of(1), Some(Tier::Fast), "baseline demoted the accessed key");
		assert_eq!(b.tier_of(1), Some(Tier::Fast), "accessed key was demoted anyway");
		assert_eq!(b.tier_of(2), Some(Tier::Slow), "the unaccessed key should have gone");
		assert_eq!(a.tier_of(2), b.tier_of(2));
		assert!(
			mb.contains(&(2, Tier::Slow)),
			"the demotion migration must name the unaccessed key",
		);
		assert!(
			!mb.contains(&(1, Tier::Slow)),
			"a reprieve must emit no demotion migration",
		);

		for k in 1..=extra {
			assert_eq!(a.tier_of(k), b.tier_of(k), "tier diverges for key {k}");
		}
		assert_eq!(a.fast_bytes_used(), b.fast_bytes_used());
		assert_eq!(a.slow_bytes_used(), b.slow_bytes_used());
		assert_eq!(a.fast_object_count(), b.fast_object_count());
		assert_eq!(a.slow_object_count(), b.slow_object_count());
	}

	// ---------------------------------------------------------------------
	// Gap closures. Each of these pins a behaviour that a real mutation
	// changed without a single test above noticing -- the mutation it kills
	// is named in its doc comment.
	// ---------------------------------------------------------------------

	/// `settle_one_access`'s trigger is a strict `>`: a one-access queue
	/// filled to EXACTLY its budget is not over it, so nothing may be
	/// reprieved. Relaxing that loop to `>=` splices the tail into the main
	/// queue anyway, and `one_access_overflow_is_reprieved_not_evicted` above
	/// only ever runs the queue far past its budget, so it cannot see the
	/// difference.
	#[test]
	fn nothing_is_reprieved_at_exactly_the_one_access_budget() {
		// ratio 0.01 of 1_000_000 is a 10_000-byte one-access budget, and ten
		// 1_000-byte objects fill it to the byte.
		const SIZE: ObjectSize = 1_000;
		const COUNT: u64 = 10;

		let mut a =
			S3FifoLazyDemotionReprieveHybridStack::new(0.01, MAX, 1_000_000).with_shared_overhead(0);
		let mut b = S3FifoLazyDemotionReprieveCompactHybridStack::new(0.01, MAX, 1_000_000)
			.with_shared_overhead(0);

		assert_eq!(
			b.one_access_capacity,
			COUNT * SIZE as CacheSize,
			"the construction depends on filling the budget to the byte",
		);

		for k in 1..=COUNT {
			a.insert(k, SIZE);
			b.insert(k, SIZE);
		}

		// Sitting exactly ON the budget: everything is still in the one-access
		// queue, the main queue is empty, and the whole charge is on the slow
		// side (the one-access queue is PMEM in this design).
		assert_eq!(a.slow_bytes_used(), COUNT * SIZE as CacheSize);
		assert_eq!(b.slow_bytes_used(), a.slow_bytes_used());
		assert_eq!(b.one_access_used, b.one_access_capacity);
		assert_eq!(a.fast_object_count(), 0);
		assert_eq!(b.fast_object_count(), a.fast_object_count());

		let ma = a.drain_tier_migrations();
		let mb = b.drain_tier_migrations();
		assert_eq!(ma, mb);
		assert!(mb.is_empty(), "nothing moved, so nothing to migrate");

		// `evict_one` drains the MAIN queue only, so with nothing reprieved
		// there is nothing for it to find at all.
		assert_eq!(a.evict_one(), None, "baseline reprieved at the budget rather than past it");
		assert_eq!(
			b.evict_one(),
			None,
			"a key was reprieved into the main queue while the one-access queue sat exactly on its budget",
		);
		assert_eq!(a.len(), COUNT as usize);
		assert_eq!(b.len(), a.len());

		// One byte more IS over the budget: the oldest key is reprieved and
		// becomes the main queue's eviction candidate. Without this the
		// assertions above would hold just as well against a stack that never
		// reprieved anything.
		a.insert(COUNT + 1, 1);
		b.insert(COUNT + 1, 1);
		assert_eq!(a.evict_one(), Some(1), "one byte past the budget must reprieve the tail");
		assert_eq!(b.evict_one(), Some(1), "one byte past the budget must reprieve the tail");
		assert_eq!(a.slow_bytes_used(), b.slow_bytes_used());
		assert_eq!(a.len(), b.len());
	}

	/// `settle_fast_tier`'s guard is `fast_used <= high_bytes(...)`: usage
	/// sitting exactly ON the high watermark must fire no pass. Tightening it
	/// to `<` makes the boundary itself trigger a full drain to the low
	/// watermark. `demotion_is_reference_bit_gated` above stays strictly under
	/// the watermark and then jumps clear past it, so it never lands on it.
	#[test]
	fn no_demotion_pass_at_exactly_the_high_watermark() {
		const FAST: CacheSize = 1_000;

		// Computed, not assumed: the watermarks are env-tunable, so the number
		// of one-byte objects that lands usage exactly on the threshold is
		// whatever `high_bytes` says it is at the configured ratio.
		let high = watermarks::high_bytes(FAST);
		assert!(high >= 2, "test needs room for at least two fast objects");

		// ratio 1.0 keeps the one-access budget out of the way entirely --
		// nothing is ever reprieved, so the only pressure is the fast tier's.
		let mut a = S3FifoLazyDemotionReprieveHybridStack::new(1.0, MAX, FAST).with_shared_overhead(0);
		let mut b =
			S3FifoLazyDemotionReprieveCompactHybridStack::new(1.0, MAX, FAST).with_shared_overhead(0);

		// `insert` then `update` is this variant's only route into the fast
		// tier, and the promotion clears the reference bit, so every key built
		// this way is genuinely demotable.
		for k in 1..=high {
			a.insert(k, 1);
			b.insert(k, 1);
			a.update(k);
			b.update(k);
		}

		assert_eq!(b.fast_bytes_used(), high, "usage must land exactly on the high watermark");
		assert_eq!(a.fast_bytes_used(), b.fast_bytes_used());

		let ma = a.drain_tier_migrations();
		let mb = b.drain_tier_migrations();
		assert_eq!(ma, mb, "migrations diverge at the high watermark");
		assert!(
			!mb.iter().any(|(_, tier)| *tier == Tier::Slow),
			"usage at the high watermark must not trigger a demotion pass, got {mb:?}",
		);
		assert_eq!(a.fast_object_count(), high as usize);
		assert_eq!(b.fast_object_count(), a.fast_object_count());
		assert_eq!(a.slow_object_count(), 0, "`promote` empties the one-access queue");
		assert_eq!(b.slow_object_count(), a.slow_object_count());
		assert_eq!(b.slow_bytes_used(), a.slow_bytes_used());

		// One byte past it does fire, so the assertion above is about the
		// boundary rather than about a tier that never demotes at all.
		a.insert(high + 1, 1);
		b.insert(high + 1, 1);
		a.update(high + 1);
		b.update(high + 1);

		let ma = a.drain_tier_migrations();
		let mb = b.drain_tier_migrations();
		assert_eq!(ma, mb, "migrations diverge one byte past the high watermark");
		assert!(
			mb.iter().any(|(_, tier)| *tier == Tier::Slow),
			"one byte past the high watermark must fire a pass, got {mb:?}",
		);
		assert!(b.fast_bytes_used() <= watermarks::low_bytes(FAST));
		assert_eq!(a.fast_bytes_used(), b.fast_bytes_used());
		assert_eq!(a.slow_bytes_used(), b.slow_bytes_used());
	}

	/// `give_second_chance` pushes its `Tier::Fast` migration only if the key
	/// actually ENDED UP fast: the `settle_fast_tier()` inside it can demote
	/// the just-reprieved key straight back out in the same call, and that
	/// call has already pushed the correct `Tier::Slow`. Dropping the guard
	/// leaves the stack claiming a DRAM promotion that never happened -- and
	/// `apply_tier_migrations` runs every demotion before any promotion, so
	/// the spurious `Tier::Fast` would land last and strand the bytes.
	///
	/// The guard only arms when the effective fast budget is at rock bottom,
	/// and both existing `evict_one` tests run with room to spare. So starve
	/// it deliberately: `shared_overhead` reserving more DRAM than
	/// `fast_capacity` holds drives `effective_main_fast_capacity()` to 0,
	/// which puts every watermark computed on top of it at 0 too, whatever
	/// ratios are configured.
	#[test]
	fn a_second_chance_undone_in_the_same_call_emits_no_promotion_migration() {
		const SIZE: ObjectSize = 112;
		const OVERHEAD: CacheSize = 112;
		const FAST: CacheSize = 2_048;
		const COUNT: u64 = 30;

		// ratio 0.0 is a zero-byte one-access budget, so every admitted key is
		// reprieved into `Q_MAIN_SLOW` on the spot -- the shortest route to a
		// populated main queue with an empty fast list.
		let mut a =
			S3FifoLazyDemotionReprieveHybridStack::new(0.0, MAX, FAST).with_shared_overhead(OVERHEAD);
		let mut b = S3FifoLazyDemotionReprieveCompactHybridStack::new(0.0, MAX, FAST)
			.with_shared_overhead(OVERHEAD);

		for k in 1..=COUNT {
			a.insert(k, SIZE);
			b.insert(k, SIZE);
		}
		a.drain_tier_migrations();
		b.drain_tier_migrations();

		assert_eq!(
			b.effective_main_fast_capacity(),
			0,
			"{COUNT} keys of {OVERHEAD}-byte shared overhead must swallow the whole {FAST}-byte budget",
		);
		assert_eq!(b.slow_object_count(), COUNT as usize);
		assert_eq!(b.slow_object_count(), a.slow_object_count());
		assert_eq!(b.fast_object_count(), 0);
		assert_eq!(b.fast_object_count(), a.fast_object_count());

		// Key 1 is the oldest reprieved key, so it is the slow tail --
		// `evict_one`'s candidate. Setting its bit sends it into
		// `give_second_chance`, whose `settle_fast_tier` then finds a
		// zero-byte budget and demotes it straight back out.
		a.update(1);
		b.update(1);

		let ea = a.evict_one();
		let eb = b.evict_one();
		let ma = a.drain_tier_migrations();
		let mb = b.drain_tier_migrations();

		assert_eq!(ea, eb, "eviction diverges across an undone second chance");
		assert_eq!(ea, Some(2), "key 1 was spared, so the next-oldest goes instead");

		assert_eq!(ma, mb, "migrations diverge across an undone second chance");
		assert_eq!(
			mb,
			vec![(1, Tier::Slow)],
			"an undone second chance emits the demotion and nothing else, got {mb:?}",
		);
		assert!(
			!mb.contains(&(1, Tier::Fast)),
			"the key never stayed in DRAM, so no promotion migration may be recorded",
		);

		assert_eq!(b.tier_of(1), Some(Tier::Slow));
		assert_eq!(b.tier_of(1), a.tier_of(1));
		assert_eq!(b.fast_bytes_used(), 0, "the fast list must be empty again");
		assert_eq!(b.fast_bytes_used(), a.fast_bytes_used());
		assert_eq!(b.slow_bytes_used(), a.slow_bytes_used());
		assert_eq!(b.len(), a.len());
	}

	/// `resize` settles the fast tier on the way out, and `resize_key` --
	/// reached by RE-SETTING an already-tracked key -- is the one path that
	/// raises `fast_used` without settling behind it. Every other route into
	/// the fast tier (`promote_from_one_access`, `give_second_chance`) settles
	/// itself, which is why dropping `resize`'s trailing `settle_fast_tier()`
	/// went unnoticed: `resizes_like_the_baseline` above re-sets keys only at
	/// one fixed size, so its tier is always already settled by the time
	/// `resize` is reached.
	#[test]
	fn resize_settles_a_fast_tier_grown_since_the_last_pass() {
		const FAST: CacheSize = 1_000;

		let mut a = S3FifoLazyDemotionReprieveHybridStack::new(1.0, MAX, FAST).with_shared_overhead(0);
		let mut b =
			S3FifoLazyDemotionReprieveCompactHybridStack::new(1.0, MAX, FAST).with_shared_overhead(0);

		// A tiny fast key, well under the high watermark.
		a.insert(1, 8);
		b.insert(1, 8);
		a.update(1);
		b.update(1);
		assert_eq!(a.tier_of(1), Some(Tier::Fast));
		assert_eq!(b.tier_of(1), a.tier_of(1));
		assert_eq!(a.drain_tier_migrations(), b.drain_tier_migrations());

		// Re-set it to the whole budget. That runs `resize_key`, which moves
		// `fast_used` and returns WITHOUT settling -- so the tier is now over
		// the high watermark with no pass having fired.
		let grown = FAST as ObjectSize;
		a.insert(1, grown);
		b.insert(1, grown);

		let ma = a.drain_tier_migrations();
		let mb = b.drain_tier_migrations();
		assert_eq!(ma, mb);
		assert!(mb.is_empty(), "a re-set settles nothing by itself");
		assert_eq!(b.fast_bytes_used(), FAST);
		assert_eq!(a.fast_bytes_used(), b.fast_bytes_used());
		assert!(
			b.fast_bytes_used() > watermarks::high_bytes(FAST),
			"the re-set must leave the tier genuinely over the high watermark",
		);
		assert_eq!(b.tier_of(1), Some(Tier::Fast));

		// `resize` must run the pass. (`resize_fast_tier` would too -- this is
		// specifically `resize`'s trailing settle, which the replay only ever
		// reaches with an already-settled tier.)
		a.resize(MAX);
		b.resize(MAX);

		let ma = a.drain_tier_migrations();
		let mb = b.drain_tier_migrations();
		assert_eq!(ma, mb, "migrations diverge across the settling resize");
		assert_eq!(
			mb,
			vec![(1, Tier::Slow)],
			"resize must settle a fast tier grown since the last pass, got {mb:?}",
		);
		assert_eq!(a.tier_of(1), Some(Tier::Slow));
		assert_eq!(b.tier_of(1), a.tier_of(1));
		assert!(b.fast_bytes_used() <= watermarks::low_bytes(FAST));
		assert_eq!(a.fast_bytes_used(), b.fast_bytes_used());
		assert_eq!(a.slow_bytes_used(), b.slow_bytes_used());
		assert_eq!(a.fast_object_count(), b.fast_object_count());
		assert_eq!(a.slow_object_count(), b.slow_object_count());
	}

	/// Two byte-accounting paths the replay above cannot reach. It admits at
	/// one fixed size with a ZERO DRAM-resident remainder, which makes
	/// `insert_resident`'s `size - dram_resident` charge indistinguishable
	/// from a plain `size`, and it only ever calls `update` on a key it
	/// already holds -- so `resize_key` is never called at all and
	/// `migrating()` is always just `size`.
	///
	/// This walks every charging arm in one go: admission carrying a real
	/// remainder, then a re-set of a one-access key (whose charge is handed on
	/// by the promoting touch that follows it), of a fast key, and of a
	/// reprieved -- main-queue SLOW -- key.
	#[test]
	fn resident_remainder_and_re_sets_are_charged_like_the_baseline() {
		// ratio 0.01 of 1_000_000 is a 10_000-byte one-access budget: small
		// enough that a handful of admissions produces a reprieved key to
		// re-set, large enough that the earlier steps are undisturbed.
		let mut a =
			S3FifoLazyDemotionReprieveHybridStack::new(0.01, MAX, 131_072).with_shared_overhead(0);
		let mut b = S3FifoLazyDemotionReprieveCompactHybridStack::new(0.01, MAX, 131_072)
			.with_shared_overhead(0);

		// (1) Admission carrying a DRAM-resident remainder. Only the migrating
		// part -- `size - dram_resident` -- is charged, to the one-access
		// queue, which this design gauges on the SLOW side.
		a.insert_resident(1, 1_024, 64);
		b.insert_resident(1, 1_024, 64);
		assert_eq!(a.slow_bytes_used(), 1_024 - 64, "only the migrating remainder is charged");
		assert_eq!(b.slow_bytes_used(), a.slow_bytes_used(), "admission charge diverges");
		assert_eq!(a.fast_bytes_used(), 0);
		assert_eq!(b.fast_bytes_used(), a.fast_bytes_used());

		// (2) `resize_key`'s one-access arm. The re-set adjusts the one-access
		// charge and the touch that follows it promotes the key, handing
		// exactly that charge to the fast tier -- so a wrong delta here shows
		// in BOTH gauges.
		a.insert_resident(1, 2_048, 32);
		b.insert_resident(1, 2_048, 32);
		assert_eq!(a.tier_of(1), Some(Tier::Fast), "the re-set's touch promotes a one-access key");
		assert_eq!(b.tier_of(1), a.tier_of(1));
		assert_eq!(a.fast_bytes_used(), 2_048 - 32);
		assert_eq!(b.fast_bytes_used(), a.fast_bytes_used(), "one-access re-set charge diverges");
		assert_eq!(a.slow_bytes_used(), 0);
		assert_eq!(b.slow_bytes_used(), a.slow_bytes_used());

		// (3) `resize_key`'s fast arm, shrinking: a negative delta off
		// `fast_used`.
		a.insert_resident(1, 512, 12);
		b.insert_resident(1, 512, 12);
		assert_eq!(a.fast_bytes_used(), 512 - 12);
		assert_eq!(b.fast_bytes_used(), a.fast_bytes_used(), "fast re-set charge diverges");
		assert_eq!(a.slow_bytes_used(), 0);
		assert_eq!(b.slow_bytes_used(), a.slow_bytes_used());

		// Fill the one-access queue past its 10_000-byte budget so that key 2
		// -- the oldest -- is reprieved into the main queue's slow segment.
		for k in 2..=12u64 {
			a.insert_resident(k, 1_024, 64);
			b.insert_resident(k, 1_024, 64);
		}
		assert_eq!(a.tier_of(2), Some(Tier::Slow));
		assert_eq!(b.tier_of(2), a.tier_of(2));
		assert_eq!(a.fast_object_count(), 1);
		assert_eq!(b.fast_object_count(), a.fast_object_count());
		assert_eq!(b.slow_bytes_used(), a.slow_bytes_used());

		let fast_before = a.fast_bytes_used();

		// (4) `resize_key`'s slow arm. A reprieved key is in the main queue,
		// but its bytes never left PMEM: growing it must move the SLOW gauge
		// and leave the DRAM gauge exactly where it was.
		a.insert_resident(2, 4_096, 16);
		b.insert_resident(2, 4_096, 16);
		assert_eq!(
			a.fast_bytes_used(),
			fast_before,
			"a slow key's re-set must not touch the DRAM gauge",
		);
		assert_eq!(
			b.fast_bytes_used(),
			a.fast_bytes_used(),
			"a slow key's re-set charge landed on the wrong gauge",
		);
		assert_eq!(b.slow_bytes_used(), a.slow_bytes_used(), "slow re-set charge diverges");
		assert_eq!(b.tier_of(2), a.tier_of(2));

		assert_eq!(a.drain_tier_migrations(), b.drain_tier_migrations());
		assert_eq!(a.len(), b.len());
		assert_eq!(a.fast_object_count(), b.fast_object_count());
		assert_eq!(a.slow_object_count(), b.slow_object_count());
		assert_eq!(a.dram_reserved_bytes(), b.dram_reserved_bytes());
	}
}
