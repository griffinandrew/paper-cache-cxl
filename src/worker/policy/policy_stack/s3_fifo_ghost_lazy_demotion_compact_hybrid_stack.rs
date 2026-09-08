/*
 * Copyright (c) Kia Shakiba
 *
 * This source code is licensed under the GNU AGPLv3 license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! Slab-backed S3-FIFO ghost hybrid with lazy demotion:
//! `S3FifoGhostLazyDemotionHybridStack` with one structure where that has
//! three.
//!
//! Identical to [`S3FifoGhostCompactHybridStack`] in every respect but one --
//! same ghost lifecycle, same admission/promotion/eviction rules, same
//! "contiguous front run" invariant, same shared-metadata reservation -- and
//! the one difference is `settle_fast_tier`.
//!
//! ## Lazy demotion: the whole point of this variant
//!
//! The base S3-FIFO design is classic "quick demotion, lazy promotion":
//! `settle_fast_tier` demotes the key anchoring `main_boundary`
//! *unconditionally*, consulting the reference bit only at eviction time.
//! This variant gates demotion on the bit too:
//!
//! * **Bit set** -- the key was touched since it was promoted. It is given a
//!   fresh start instead of being demoted: moved to the front of the main
//!   queue, bit cleared, `Tier` and all fast/slow accounting left alone (it
//!   was already `Tier::Fast` and stays `Tier::Fast`, so this is a reprieve,
//!   not a promotion, and produces no migration). The sweep continues to the
//!   next-oldest fast key and re-evaluates.
//! * **Bit clear** -- demoted for real, exactly as the base design does.
//!
//! The eviction-time `give_second_chance` (protecting a *slow* key touched
//! again before it reaches the tail) is completely unchanged. The two
//! mechanisms protect different things -- an unfairly-demoted fast key here,
//! an unfairly-evicted slow key there -- and compose naturally.
//!
//! **Termination.** A reprieve moves the candidate to the front and clears its
//! bit, and the sweep only ever walks toward the back via
//! `main_boundary`/`before`, so a reprieved key cannot be re-examined until
//! every other currently-fast key has had its turn. Bounded by `fast_count`
//! reprieves per call before either a real demotion happens or the boundary
//! runs out.
//!
//! **Deliberately not implemented via `give_second_chance`.** That method
//! calls `settle_fast_tier` at its own end, which its caller `evict_one`
//! needs; reusing it here would recurse for every reprieved key. The reprieve
//! arm below is the trimmed-down inline copy: no `was_fast` accounting branch
//! (the candidate is always already fast) and no trailing migration push (no
//! tier changed).
//!
//! The ghost stays OUTSIDE the slab, for the same reason it does in
//! [`S3FifoGhostCompactHybridStack`]: it holds no keys and has no index -- a
//! fixed power-of-two table of `{fingerprint: u32, inserted_at: u32}` with an
//! insertion-count window -- so it cannot live in a structure keyed by slot.
//! It is charged as a separate term alongside the per-object one and therefore
//! does not enter this stack's per-object figure at all.
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
		compact_queue_set::CompactQueueSet, ghost_filter::GhostFilter, narrow_resident,
		watermarks, CacheSize, HashedKey,
		PolicyStack, Tier,
	},
	PaperPolicy,
};

const Q_ONE_ACCESS: usize = 0;
const Q_MAIN: usize = 1;

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
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct S3FifoGhostLazyDemotionPayload {
	queue: Queue,
	tier: Option<Tier>,
	/// The part of `size` that stays in DRAM in either tier; see `migrating`.
	dram_resident: u8,
	accessed: bool,
	size: ObjectSize,
}

/// Pinned, exactly as `S3FifoEntry` is in the stack this replaces.
const _: () = assert!(
	std::mem::size_of::<S3FifoGhostLazyDemotionPayload>() == 8,
	"S3FifoGhostLazyDemotionPayload grew past 8 bytes",
);

impl S3FifoGhostLazyDemotionPayload {
	fn migrating(&self) -> CacheSize {
		(self.size as CacheSize).saturating_sub(self.dram_resident as CacheSize)
	}
}

pub struct S3FifoGhostLazyDemotionCompactHybridStack {
	queues: CompactQueueSet<S3FifoGhostLazyDemotionPayload>,

	/// Fingerprints of keys evicted from the one-access tail. Holds no keys
	/// and no slots, so it stays outside the slab.
	ghost: GhostFilter,

	one_access_ratio: f64,
	one_access_capacity: CacheSize,
	one_access_used: CacheSize,

	main_capacity: CacheSize,

	fast_capacity: CacheSize,
	fast_used: CacheSize,
	slow_used: CacheSize,

	shared_overhead: CacheSize,

	fast_count: usize,
	main_count: usize,

	main_boundary: Option<HashedKey>,

	migrations: Vec<(HashedKey, Tier)>,
}

impl S3FifoGhostLazyDemotionCompactHybridStack {
	pub fn new(one_access_ratio: f64, max_size: CacheSize, fast_capacity: CacheSize) -> Self {
		// Sized from the cache's own capacity assuming a 512-byte nominal
		// object, capped at 8 Mi slots. Under-sizing only costs ghost hits.
		let ghost = GhostFilter::with_capacity(((max_size / 512) as usize).min(8 << 20));

		S3FifoGhostLazyDemotionCompactHybridStack {
			queues: CompactQueueSet::default(),
			ghost,
			one_access_ratio,
			one_access_capacity: (one_access_ratio * max_size as f64) as CacheSize,
			one_access_used: 0,
			main_capacity: ((1.0 - one_access_ratio) * max_size as f64) as CacheSize,
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

	fn reserved_overhead(&self) -> CacheSize {
		self.queues.len() as CacheSize * self.shared_overhead + self.ghost.dram_bytes()
	}

	/// The fast-tier *value*-byte budget actually available: `fast_capacity`
	/// minus [`Self::reserved_overhead`], saturating at 0. This is the value
	/// `settle_fast_tier` applies the watermarks to. Exposed for tests.
	pub fn effective_fast_capacity(&self) -> CacheSize {
		self.fast_capacity.saturating_sub(self.reserved_overhead())
	}

	pub fn is_ghost(&self, key: HashedKey) -> bool {
		self.ghost.contains(key)
	}

	/// A brand-new key whose fingerprint is in the ghost skips the one-access
	/// queue and enters main directly, in the fast tier.
	fn admit_via_ghost_hit(&mut self, key: HashedKey, size: ObjectSize, dram_resident: u8) {
		self.queues.push_front(
			Q_MAIN,
			key,
			S3FifoGhostLazyDemotionPayload {
				queue: Queue::Main,
				tier: Some(Tier::Fast),
				dram_resident,
				accessed: false,
				size,
			},
		);
		self.fast_used += (size as CacheSize).saturating_sub(dram_resident as CacheSize);
		self.fast_count += 1;
		self.main_count += 1;

		if self.main_boundary.is_none() {
			self.main_boundary = Some(key);
		}

		self.settle_fast_tier();

		if self.queues.payload(key).and_then(|p| p.tier) == Some(Tier::Fast) {
			self.migrations.push((key, Tier::Fast));
		}
	}

	/// The ghost window tracks the main queue's population. It runs only on a
	/// genuine main-queue eviction, not on a second chance.
	fn trim_ghost(&mut self) {
		self.ghost.set_window(self.main_count);
	}

	pub fn tier_of(&self, key: HashedKey) -> Option<Tier> {
		let payload = self.queues.payload(key)?;
		match payload.queue {
			Queue::OneAccess => Some(Tier::Slow),
			Queue::Main => payload.tier,
		}
	}

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

		self.queues.move_to_front_of(Q_ONE_ACCESS, Q_MAIN, key);
		self.one_access_used = self.one_access_used.saturating_sub(size_bytes);

		if let Some(p) = self.queues.payload_mut(key) {
			p.queue = Queue::Main;
			p.tier = Some(Tier::Fast);
			p.accessed = false;
		}

		self.fast_used += size_bytes;
		self.fast_count += 1;
		self.main_count += 1;

		if self.main_boundary.is_none() {
			self.main_boundary = Some(key);
		}

		self.settle_fast_tier();

		if self.queues.payload(key).and_then(|p| p.tier) == Some(Tier::Fast) {
			self.migrations.push((key, Tier::Fast));
		}
	}

	/// An accessed key at the main tail is reinserted at the front with its
	/// reference bit cleared, rather than evicted. Unchanged by this variant:
	/// it protects a *slow* key from eviction, where the demotion-time
	/// reprieve in `settle_fast_tier` protects a *fast* key from demotion.
	fn give_second_chance(&mut self, key: HashedKey) {
		let Some(payload) = self.queues.payload(key) else { return };
		let size = payload.migrating();
		let was_fast = payload.tier == Some(Tier::Fast);
		let was_boundary = was_fast && self.main_boundary == Some(key);

		let new_boundary_if_moved = if was_boundary {
			self.queues.before(key)
		} else {
			None
		};

		self.queues.move_front(Q_MAIN, key);

		if was_boundary {
			self.main_boundary = new_boundary_if_moved;
		}

		if let Some(p) = self.queues.payload_mut(key) {
			p.tier = Some(Tier::Fast);
			p.accessed = false;
		}

		if !was_fast {
			self.slow_used = self.slow_used.saturating_sub(size);
			self.fast_used += size;
			self.fast_count += 1;
		}

		if self.main_boundary.is_none() {
			self.main_boundary = Some(key);
		}

		self.settle_fast_tier();

		if self.queues.payload(key).and_then(|p| p.tier) == Some(Tier::Fast) {
			self.migrations.push((key, Tier::Fast));
		}
	}

	/// Demotes key(s) anchoring `main_boundary` until the fast tier is back
	/// under the shared *low* watermark -- but only once usage has crossed the
	/// shared *high* watermark, and still reference-bit gated per candidate
	/// rather than unconditional. That gate is the one mechanic that differs
	/// from [`S3FifoGhostCompactHybridStack`]; see the module doc.
	///
	/// Per-demotion bookkeeping is untouched: a demoted object still retags its
	/// payload, still moves `fast_used`/`fast_count`/`slow_used` by its own
	/// size, still walks `main_boundary` one step toward the front, and still
	/// emits exactly one `Tier::Slow` migration -- and a reprieved candidate
	/// still changes none of them. A pass simply walks further before it stops.
	fn settle_fast_tier(&mut self) {
		let effective_capacity = self.effective_fast_capacity();

		if self.fast_used <= watermarks::high_bytes(effective_capacity) {
			return;
		}

		let drain_target = watermarks::low_bytes(effective_capacity);

		while self.fast_used > drain_target {
			let Some(candidate) = self.main_boundary else { break };
			let accessed = self.queues.payload(candidate).map(|p| p.accessed).unwrap_or(false);

			if accessed {
				// Reprieve: fresh start at the front instead of demotion. Same
				// before-then-move ordering `give_second_chance` uses. No
				// fast/slow accounting change -- the key was already Fast and
				// stays Fast -- and no migration, since no tier changed.
				let new_boundary = self.queues.before(candidate);

				self.queues.move_front(Q_MAIN, candidate);
				self.main_boundary = new_boundary;

				if let Some(p) = self.queues.payload_mut(candidate) {
					p.accessed = false;
				}

				continue;
			}

			let size = self.queues.payload(candidate).map(|p| p.migrating()).unwrap_or(0);
			let new_boundary = self.queues.before(candidate);

			if let Some(p) = self.queues.payload_mut(candidate) {
				p.tier = Some(Tier::Slow);
			}

			self.fast_used = self.fast_used.saturating_sub(size);
			self.fast_count = self.fast_count.saturating_sub(1);
			self.slow_used += size;
			self.main_boundary = new_boundary;

			self.migrations.push((candidate, Tier::Slow));
		}
	}

	fn main_is_full(&self) -> bool {
		self.fast_used + self.slow_used >= self.main_capacity
	}

	fn evict_one_access_tail(&mut self) -> Option<HashedKey> {
		let (key, payload) = self.queues.pop_back(Q_ONE_ACCESS)?;
		self.one_access_used = self.one_access_used.saturating_sub(payload.migrating());
		self.ghost.insert(key);
		Some(key)
	}
}

impl PolicyStack for S3FifoGhostLazyDemotionCompactHybridStack {
	fn is_policy(&self, policy: &PaperPolicy) -> bool {
		matches!(policy, PaperPolicy::S3FifoGhostLazyDemotionCompactHybrid(r) if *r == self.one_access_ratio)
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

		if self.ghost.contains(key) {
			self.admit_via_ghost_hit(key, size, dram_resident);
			return;
		}

		self.queues.push_front(
			Q_ONE_ACCESS,
			key,
			S3FifoGhostLazyDemotionPayload {
				queue: Queue::OneAccess,
				tier: None,
				dram_resident,
				accessed: false,
				size,
			},
		);
		self.one_access_used += (size as CacheSize).saturating_sub(dram_resident as CacheSize);
	}

	fn update(&mut self, key: HashedKey) {
		if self.queues.contains(key) {
			self.touch(key);
		}
	}

	fn remove(&mut self, key: HashedKey) {
		// BEFORE the early return: after a one-access eviction a key lives only
		// in the ghost, with no entry row to find.
		self.ghost.remove(key);

		let Some(payload) = self.queues.payload(key) else { return };
		let size = payload.migrating();

		match payload.queue {
			Queue::OneAccess => {
				self.queues.remove(Q_ONE_ACCESS, key);
				self.one_access_used = self.one_access_used.saturating_sub(size);
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
		self.one_access_capacity = (self.one_access_ratio * max_size as f64) as CacheSize;
		self.main_capacity = ((1.0 - self.one_access_ratio) * max_size as f64) as CacheSize;
	}

	fn clear(&mut self) {
		self.queues.clear();
		self.ghost.clear();

		self.one_access_used = 0;
		self.fast_used = 0;
		self.slow_used = 0;
		self.fast_count = 0;
		self.main_count = 0;
		self.main_boundary = None;
		self.migrations.clear();
	}

	fn evict_one(&mut self) -> Option<HashedKey> {
		if !self.main_is_full() {
			if let Some(key) = self.evict_one_access_tail() {
				return Some(key);
			}
		}

		loop {
			let key = self.queues.back(Q_MAIN)?;
			let accessed = self.queues.payload(key).map(|p| p.accessed).unwrap_or(false);

			if accessed {
				self.give_second_chance(key);
				continue;
			}

			let payload = self.queues.remove(Q_MAIN, key);
			let size = payload.map(|p| p.migrating()).unwrap_or(0);
			let tier = payload.and_then(|p| p.tier);
			self.main_count = self.main_count.saturating_sub(1);

			match tier {
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

			self.trim_ghost();

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

	fn fast_bytes_used(&self) -> CacheSize {
		self.fast_used
	}

	fn slow_bytes_used(&self) -> CacheSize {
		self.one_access_used + self.slow_used
	}

	fn fast_object_count(&self) -> usize {
		self.fast_count
	}

	fn slow_object_count(&self) -> usize {
		self.queues.queue_len(Q_ONE_ACCESS) + (self.main_count - self.fast_count)
	}

	fn needs_capacity_eviction(&self) -> bool {
		self.one_access_used > self.one_access_capacity
	}
}


/// What survives of this stack's original `fidelity_tests` module.
///
/// That module replayed an op stream through this stack and through
/// `S3FifoGhostLazyDemotionHybridStack`, the non-compact baseline it is a
/// compaction of, and asserted the two were indistinguishable. The baseline
/// has since been removed from the crate, so the oracle is gone and the
/// differential cases went with it -- git history keeps them.
///
/// These two do not need the baseline. The first pins the registration
/// surface, which is easy for a copied parser to get subtly wrong; the second
/// compares this stack against the NON-lazy compact ghost stack, which is very
/// much alive, and is what proves the lazy-demotion delta was actually applied
/// rather than copied across unchanged.
#[cfg(all(test, feature = "s3_fifo_ghost_lazy_demotion_compact_hybrid_cache"))]
mod compact_tests {
	use super::*;
	use crate::worker::policy::policy_stack::s3_fifo_ghost_compact_hybrid_stack::S3FifoGhostCompactHybridStack;

	/// The policy string round-trips and rejects the ratio that would starve
	/// the main queue.
	///
	/// `policy.rs`'s own `S3_FIFO_MAIN_SIZED_PREFIXES` would be the natural
	/// home for this, but its companion test asserts the two prefix lists
	/// account for exactly ten s3-fifo parsers, and none of the earlier
	/// compact conversions added themselves to it. Pinned here instead so the
	/// guarantee is tested somewhere rather than nowhere.
	#[test]
	fn the_policy_string_round_trips_and_rejects_a_starving_ratio() {
		let parsed = "s3-fifo-ghost-lazy-demotion-compact-hybrid-0.25"
			.parse::<PaperPolicy>()
			.expect("should parse");

		assert_eq!(parsed, PaperPolicy::S3FifoGhostLazyDemotionCompactHybrid(0.25));
		assert_eq!(parsed.to_string(), "s3-fifo-ghost-lazy-demotion-compact-hybrid-0.25");
		assert!(parsed.is_hybrid(), "the compact variant is still a tiered design");

		// This stack sizes `main_capacity` at `(1 - ratio) * max_size` and
		// gates `evict_one` on `main_is_full`, so a ratio of exactly 1 leaves
		// the main queue zero bytes and the eviction loop spins.
		assert!(
			"s3-fifo-ghost-lazy-demotion-compact-hybrid-1.0".parse::<PaperPolicy>().is_err(),
			"a ratio of 1 leaves the main queue zero bytes and must be rejected",
		);
		assert!(
			"s3-fifo-ghost-lazy-demotion-compact-hybrid-0.999".parse::<PaperPolicy>().is_ok(),
			"the exclusion must be an endpoint exclusion and nothing more",
		);
		assert!(
			"s3-fifo-ghost-lazy-demotion-compact-hybrid-0.0".parse::<PaperPolicy>().is_ok(),
			"zero means no one-access queue, which starves nothing",
		);
	}

	/// The lazy-demotion delta really was applied. Under a workload that leaves
	/// the demotion candidates' reference bits SET, the non-lazy compact ghost
	/// stack demotes them anyway; this stack must reprieve them. If
	/// `settle_fast_tier` had been copied across from the non-lazy stack
	/// unchanged, this is the assertion that would catch it.
	#[test]
	fn lazy_demotion_actually_diverges_from_the_non_lazy_compact_stack() {
		let fast_capacity: CacheSize = 1_000;
		let size: ObjectSize = 10;
		let bytes = size as CacheSize;
		let count = watermarks::high_bytes(fast_capacity) / bytes + 1;

		let mut lazy = S3FifoGhostLazyDemotionCompactHybridStack::new(1.0, 100_000, fast_capacity);
		let mut eager = S3FifoGhostCompactHybridStack::new(1.0, 100_000, fast_capacity);

		for key in 1..count {
			lazy.insert(key, size);
			lazy.update(key);
			eager.insert(key, size);
			eager.update(key);
		}
		lazy.drain_tier_migrations();
		eager.drain_tier_migrations();

		for key in 1..=3 {
			lazy.update(key);
			eager.update(key);
		}

		lazy.insert(count, size);
		lazy.update(count);
		eager.insert(count, size);
		eager.update(count);

		let m_lazy = lazy.drain_tier_migrations();
		let m_eager = eager.drain_tier_migrations();

		assert!(
			m_eager.contains(&(1, Tier::Slow)),
			"the non-lazy stack demotes unconditionally; got {m_eager:?}",
		);
		assert!(
			!m_lazy.contains(&(1, Tier::Slow)),
			"the lazy stack must reprieve an accessed candidate; got {m_lazy:?}",
		);
		assert_ne!(m_lazy, m_eager, "the lazy-demotion delta was not applied");
		assert_eq!(lazy.tier_of(1), Some(Tier::Fast));
		assert_eq!(eager.tier_of(1), Some(Tier::Slow));
	}
}
