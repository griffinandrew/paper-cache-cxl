/*
 * Copyright (c) Kia Shakiba
 *
 * This source code is licensed under the GNU AGPLv3 license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! S3-FIFO, tier-segmented: a [`TierPolicy`] over a slow one-access queue and
//! a main queue whose fast prefix is the tier (R4; it was a whole stack of
//! its own, `S3FifoCompactHybridStack`, in a file of its own), with and
//! without a ghost (`S3FifoGhostCompactHybridStack`), and with lazy demotion
//! (`S3FifoGhostLazyDemotionCompactHybridStack`).
//!
//! Admission lands at the front of the one-access queue and is entirely
//! slow-tier: a new key is built slow, and nothing is pushed or settled. A
//! hit there promotes the key to the front of main and to fast, eagerly; a
//! hit in main only sets the reference bit -- this is the hottest per-get
//! operation in the family, and it touches no queue order at all -- and it is
//! eviction that acts on it: an accessed key at the main tail is moved to the
//! front with the bit cleared instead of being evicted (a SECOND CHANCE).
//! The main lane is split: the layer's cursor names its least-recently-used
//! fast key.
//!
//! Eviction prefers the one-access tail while main is not full (`fast + slow`
//! bytes of main under `(1 - ratio) x max_size`), and otherwise walks main's
//! tail. So a full main evicts from main even when the one-access queue is
//! what is over its budget.
//!
//! The ghost: a key evicted from the one-access tail leaves a fingerprint
//! behind ([`GhostFilter`](super::ghost_filter::GhostFilter): a fixed table
//! with no keys and no index, outside the slab -- so it is charged as a
//! separate term beside the per-object reservation), and a later admission
//! that hits it skips the one-access queue entirely and enters the front of
//! main, fast: a real promotion, since the value was built slow, so it is
//! pushed `(key, Fast)` (slow, and pushed nothing, when the value is
//! structural). A hit does not retire the entry; the ghost's window follows
//! main's population and is trimmed only by a genuine main eviction, not by
//! a second chance or a one-access eviction; and the layer clears a key's
//! fingerprint in `remove` before it asks whether the key is tracked.
//!
//! Lazy demotion (`LAZY`) is the one change to the settle. The base design is
//! classic "quick demotion, lazy promotion": the settle demotes the cursor's
//! key unconditionally and the reference bit is consulted only at eviction.
//! This variant gates demotion on the bit too: a candidate whose bit is set was
//! touched since it was promoted, and is given a fresh start instead -- moved
//! to the front of main, bit cleared, tier and accounting left alone (it was
//! fast and stays fast: a reprieve, not a promotion, so no migration) -- and
//! the sweep goes on to the next-oldest fast key. A candidate whose bit is
//! clear is demoted as before. It terminates: a reprieve clears the bit and
//! moves the key to the front, and the cursor only walks toward the back, so
//! a reprieved key is not examined again until every other fast key has had
//! its turn. The eviction-time second chance protects a SLOW key touched
//! again before it reaches the tail; the two mechanisms compose. With no
//! other fast key in front of it a reprieved candidate is the cursor again,
//! and its cleared bit demotes it at the next step.
//!
//! `needs_capacity_eviction` is the one-access queue's own budget
//! (`ratio x max_size`), raw.
//!
//! The second chance pushes `(key, Fast)` whenever the key ends fast, even if
//! it already was -- [`Push::IfEndsFast`], this family's rule, where a
//! promotion elsewhere pushes only for a key that was slow. A structural key
//! (S5) gets its second chance at the front with tier slow, never promoted,
//! and a fast one leaves the fast set, pushed.
//!
//! **The baseline named here before no longer exists in this crate.** Every
//! non-compact hybrid stack was removed once its compact twin was shown
//! behaviourally identical and cheaper: 72 B/object of eviction stack instead
//! of 112 then, and 40 since the arena conversion
//! (`ARENA_STACK_DRAM_OVERHEAD`). Git history holds the baseline and the
//! differential tests that proved the two agreed, and `tier_goldens.txt` holds
//! what this design did, fingerprinted, before it was ported.

use std::marker::PhantomData;

use crate::PaperPolicy;

use super::{
	ghost_filter::GhostFilter,
	tiered_stack::{Ghost, Lane, Meta, NoGhost, Push, SlowSplit, TierPolicy, TieredStack},
	CacheSize, HashedKey, Tier,
};

/// The slow one-access queue and the main queue.
const ONE: Lane = 0;
const MAIN: Lane = 1;

/// S3-FIFO: `ratio` is the one-access queue's share of the cache, with the
/// byte budgets it and main are held to; `G` is the ghost, if any; `LAZY`
/// gates the settle's demotion on the reference bit.
pub struct S3<G: Ghost, const LAZY: bool> {
	ratio: f64,
	one_capacity: CacheSize,
	main_capacity: CacheSize,
	ghost: PhantomData<fn() -> G>,
}

/// `PaperPolicy::S3FifoCompactHybrid`.
pub type S3FifoCompactHybridStack = TieredStack<S3<NoGhost, false>>;

/// `PaperPolicy::S3FifoGhostCompactHybrid`.
pub type S3FifoGhostCompactHybridStack = TieredStack<S3<GhostFilter, false>>;

/// `PaperPolicy::S3FifoGhostLazyDemotionCompactHybrid`.
pub type S3FifoGhostLazyDemotionCompactHybridStack = TieredStack<S3<GhostFilter, true>>;

impl<G: Ghost, const LAZY: bool> S3<G, LAZY> {
	fn sized(ratio: f64, max_size: CacheSize) -> Self {
		S3 {
			ratio,
			one_capacity: (ratio * max_size as f64) as CacheSize,
			main_capacity: ((1.0 - ratio) * max_size as f64) as CacheSize,
			ghost: PhantomData,
		}
	}
}

impl<G: Ghost, const LAZY: bool> TieredStack<S3<G, LAZY>> {
	pub fn new(ratio: f64, max_size: CacheSize, fast_capacity: CacheSize) -> Self {
		TieredStack::with_ghost(S3::sized(ratio, max_size), fast_capacity, G::sized_for(max_size))
	}
}

impl<G: Ghost, const LAZY: bool> TierPolicy for S3<G, LAZY> {
	type Layout = SlowSplit;
	type Ghost = G;

	const LAZY_DEMOTION: bool = LAZY;

	const ADMIT: Lane = ONE;
	const RESETTLE: &'static [Lane] = &[MAIN];
	const ON_TIER_RESIZE: &'static [Lane] = &[MAIN];
	const SECOND_CHANCE: Push = Push::IfEndsFast;

	fn is_policy(&self, policy: &PaperPolicy) -> bool {
		match policy {
			PaperPolicy::S3FifoCompactHybrid(ratio) => !G::PRESENT && !LAZY && *ratio == self.ratio,
			PaperPolicy::S3FifoGhostCompactHybrid(ratio) => G::PRESENT && !LAZY && *ratio == self.ratio,
			PaperPolicy::S3FifoGhostLazyDemotionCompactHybrid(ratio) => G::PRESENT && LAZY && *ratio == self.ratio,
			_ => false,
		}
	}

	/// A brand-new key enters the one-access queue; one the ghost remembers
	/// enters main.
	fn admit(s: &mut TieredStack<Self>, key: HashedKey, m: Meta) {
		match s.ghost.contains(key) {
			true => s.admit(key, m, MAIN, true),
			false => s.admit(key, m, ONE, false),
		}
	}

	/// The one-access queue's evictions populate the ghost; a main eviction
	/// trims it to main's population (a second chance does not).
	fn evicted(s: &mut TieredStack<Self>, lane: Lane, key: HashedKey) {
		match lane {
			ONE => s.ghost.insert(key),
			_ => s.ghost.set_window(s.lane_len(MAIN)),
		}
	}

	/// A one-access key is promoted to main's front, fast; a main key's bit is
	/// set, and nothing moves.
	fn touch(s: &mut TieredStack<Self>, key: HashedKey, structural: bool) {
		match s.lane_of(key) {
			Some(ONE) => s.to_front(key, MAIN, structural, Push::IfPromoted),
			Some(_) => s.set_bit(key, true),
			None => {},
		}
	}

	/// An access never reorders main, so a FAST main key overwritten with a
	/// structural value leaves the fast set in place (S5), then is touched.
	fn overwrite(s: &mut TieredStack<Self>, key: HashedKey, m: Meta) {
		s.resize_key(key, m);

		if m.structural && s.lane_of(key) == Some(MAIN) && s.tier_of(key) == Some(Tier::Fast) {
			s.demote_in_place(key);
		}

		Self::touch(s, key, m.structural);
	}

	/// The one-access tail while main is not full, else main's tail: a key
	/// with its bit set goes to the front (a second chance) and is looked past.
	fn victim(s: &mut TieredStack<Self>) -> Option<HashedKey> {
		if s.lane_bytes(MAIN) < s.policy.main_capacity {
			if let Some(key) = s.evict_tail(ONE) {
				return Some(key);
			}
		}

		loop {
			let key = s.tail(MAIN)?;

			if !s.bit(key) {
				return s.evict_tail(MAIN);
			}

			s.second_chance(key, MAIN);
		}
	}

	fn wants_eviction(s: &TieredStack<Self>) -> bool {
		s.lane_bytes(ONE) > s.policy.one_capacity
	}

	fn resized(s: &mut TieredStack<Self>, max_size: CacheSize) {
		s.policy = S3::sized(s.policy.ratio, max_size);
	}
}

/// The fast tier is charged the metadata of EVERY tracked key -- one-access,
/// fast main and slow main alike -- and the ghost entries on top. A
/// reservation of `fast_object_count() x shared_overhead` understates DRAM by
/// every slow key's metadata and fails this test.
#[cfg(test)]
mod reservation_tests {
	use super::*;
	use super::super::{drain_target, PolicyStack};
	use crate::object::ObjectSize;

	const FAST_CAPACITY: CacheSize = 10_000;
	const OVERHEAD: CacheSize = 200;
	const SIZE: ObjectSize = 1_000;
	/// Large enough that main is never full, so an eviction always takes the
	/// one-access tail and leaves a ghost.
	const MAX_SIZE: CacheSize = 1_000_000;

	/// 24 admissions, four one-access evictions (four ghosts), twelve hits
	/// promoting into main. Twenty keys x 200 B plus the ghost leave ~6_000 B
	/// for values: five main keys stay fast, seven go slow, eight are still
	/// one-access. Charging the fast ones alone would keep eight fast.
	#[test]
	fn every_tracked_key_is_charged_and_the_ghost_on_top() {
		let mut stack = S3FifoGhostCompactHybridStack::new(0.1, MAX_SIZE, FAST_CAPACITY)
			.with_shared_overhead(OVERHEAD);

		for key in 1..=24 {
			stack.insert(key, SIZE);
		}

		for oldest in 1..=4 {
			assert_eq!(stack.evict_one(), Some(oldest));
		}

		for key in 5..=16 {
			stack.update(key);
		}

		let tracked = stack.len() as CacheSize;
		let one_access = stack.lane_len(ONE);
		let main_slow = stack.slow_object_count() - one_access;

		assert_eq!(tracked, 20);
		assert!(
			one_access > 0 && main_slow > 0 && stack.fast_object_count() > 0,
			"the fixture must populate every state: {one_access} one-access, {} fast, \
			 {main_slow} slow in main",
			stack.fast_object_count(),
		);
		assert!((1..=4).all(|k| stack.ghost.contains(k)), "each eviction must leave a ghost");

		let ghost = stack.ghost.dram_bytes();

		assert_eq!(
			ghost,
			4 * crate::object::overhead::GHOST_ENTRY_DRAM_OVERHEAD as CacheSize,
			"four live ghost entries",
		);
		assert_eq!(
			stack.dram_reserved_bytes(),
			tracked * OVERHEAD + ghost,
			"all {tracked} tracked keys keep their metadata in DRAM and the ghost is \
			 charged on top, but the reservation covers {} keys' worth ({} fast, \
			 {main_slow} slow in main, {one_access} one-access)",
			stack.dram_reserved_bytes().saturating_sub(ghost) / OVERHEAD,
			stack.fast_object_count(),
		);

		let effective = FAST_CAPACITY - tracked * OVERHEAD - ghost;

		assert!(
			stack.fast_bytes_used() <= drain_target::bytes(effective),
			"{} B of values are fast against {effective} B left once every key's \
			 metadata and the ghost are reserved",
			stack.fast_bytes_used(),
		);
	}
}

/// What survives of this design's original `fidelity_tests` module.
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
	use super::super::{drain_target, PolicyStack, Tier};
	use crate::object::ObjectSize;

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
	/// the settle had been copied across from the non-lazy stack unchanged,
	/// this is the assertion that would catch it.
	#[test]
	fn lazy_demotion_actually_diverges_from_the_non_lazy_compact_stack() {
		let fast_capacity: CacheSize = 1_000;
		let size: ObjectSize = 10;
		let bytes = size as CacheSize;
		let count = drain_target::bytes(fast_capacity) / bytes + 1;

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

/// S3-FIFO's two byte budgets are thresholds, and at exactly the threshold they
/// fall on different sides: the one-access queue is over its capacity only
/// ABOVE it, main is full AT it. (No golden reaches a byte total that equals a
/// capacity.)
#[cfg(test)]
mod capacity_tests {
	use super::*;
	use super::super::PolicyStack;

	#[test]
	fn the_one_access_queue_at_exactly_its_capacity_asks_for_nothing() {
		let mut stack = S3FifoCompactHybridStack::new(0.5, 2_000, 1 << 30);

		stack.insert(1, 1_000);

		assert!(!stack.needs_capacity_eviction(), "1,000 B in a queue of 1,000 B is not over it");

		stack.insert(2, 1);

		assert!(stack.needs_capacity_eviction(), "1,001 B is");
	}

	#[test]
	fn main_at_exactly_its_capacity_is_full() {
		let mut stack = S3FifoCompactHybridStack::new(0.5, 2_000, 1 << 30);

		stack.insert(1, 1_000);
		stack.update(1);
		stack.insert(2, 10);

		assert_eq!(
			stack.evict_one(),
			Some(1),
			"a main holding exactly its 1,000 B is full: the victim comes from main, not from the one-access queue",
		);
		assert_eq!(stack.evict_one(), Some(2), "main is empty now, so the one-access queue is next");
	}
}

/// A resize of the cache queues nothing in S3-FIFO, with or without its ghost,
/// or with lazy demotion, even on a stack the metadata push has left over its
/// budget: only `resettle` and a resize of the fast tier settle it.
#[cfg(test)]
mod resize_tests {
	use super::*;
	use super::super::tiered_stack::testing::a_resize_settles_nothing;

	#[test]
	fn s3_fifo() {
		a_resize_settles_nothing(S3FifoCompactHybridStack::new(0.1, 1_000_000, 10_000));
	}

	#[test]
	fn s3_fifo_ghost() {
		a_resize_settles_nothing(S3FifoGhostCompactHybridStack::new(0.1, 1_000_000, 10_000));
	}

	#[test]
	fn s3_fifo_ghost_lazy_demotion() {
		a_resize_settles_nothing(S3FifoGhostLazyDemotionCompactHybridStack::new(0.1, 1_000_000, 10_000));
	}
}
