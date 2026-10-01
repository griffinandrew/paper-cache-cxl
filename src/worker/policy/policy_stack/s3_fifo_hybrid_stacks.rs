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
//! Fast admission (`FastAdmission`, and `S3FifoGhostLazyDemotionFastAdmissionCompactHybridStack`)
//! makes the one-access queue DRAM: a FAST lane of the layer, carved out of the
//! fast tier (`tiered_stack::carve`). `placement_of` reports `Fast` for a key in
//! it and `fast_bytes_used` and `fast_object_count` count it, so admission is a
//! plain DRAM write rather than a synchronous PMEM allocation on the calling
//! thread. Three things follow. The two fast segments share one budget: the
//! queue's capacity is sized from the CACHE, so it is carved out of the tier
//! only up to what the tier can pay for, main's fast segment is what is left,
//! and the reservation is split between them in proportion; the queue's
//! eviction trigger reads its own budget, main's demotion trigger reads main's,
//! and a resize of the cache settles main, since it moves the carve-out. A
//! promotion out of the queue, and a ghost-hit admission, emit NO `(key, Fast)`
//! migration: those keys' bytes are already DRAM (the API layer built them
//! fast), and a migration would copy correct DRAM bytes into a fresh DRAM
//! buffer for nothing. The second chance keeps its push: a key reaching it can
//! really be in PMEM. And a STRUCTURAL new key takes a slow place at main's
//! front instead of the queue, whose overflow is an eviction.
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
	tiered_stack::{newly_fills, FastSplit, Ghost, Lane, Layout, Meta, NoGhost, Push, SlowSplit, TierPolicy, TieredStack},
	drain_target, CacheSize, HashedKey, Tier,
};

/// The one-access queue and the main queue.
const ONE: Lane = 0;
const MAIN: Lane = 1;

/// Where a brand-new key is admitted: a slow one-access queue, the base
/// design's, or a DRAM one.
pub trait Admission: 'static {
	type Layout: Layout;

	/// Whether the one-access queue is DRAM: a FAST lane, carved out of the
	/// fast tier.
	const FAST: bool;
}

/// The one-access queue is entirely slow.
pub struct SlowAdmission;

/// The one-access queue is DRAM.
pub struct FastAdmission;

impl Admission for SlowAdmission {
	type Layout = SlowSplit;

	const FAST: bool = false;
}

impl Admission for FastAdmission {
	type Layout = FastSplit;

	const FAST: bool = true;
}

/// S3-FIFO: `ratio` is the one-access queue's share of the cache, with the
/// byte budgets it and main are held to; `G` is the ghost, if any; `LAZY`
/// gates the settle's demotion on the reference bit; `A` says where a new key
/// is admitted.
pub struct S3<G: Ghost, const LAZY: bool, A: Admission = SlowAdmission> {
	ratio: f64,
	one_capacity: CacheSize,
	main_capacity: CacheSize,

	/// Whether the last check found `one_capacity` at least the whole fast
	/// tier (`FastAdmission`): the warning for that crossing has been given.
	filled: bool,

	ghost: PhantomData<fn() -> (G, A)>,
}

/// `PaperPolicy::S3FifoCompactHybrid`.
pub type S3FifoCompactHybridStack = TieredStack<S3<NoGhost, false>>;

/// `PaperPolicy::S3FifoGhostCompactHybrid`.
pub type S3FifoGhostCompactHybridStack = TieredStack<S3<GhostFilter, false>>;

/// `PaperPolicy::S3FifoGhostLazyDemotionCompactHybrid`.
pub type S3FifoGhostLazyDemotionCompactHybridStack = TieredStack<S3<GhostFilter, true>>;

/// `PaperPolicy::S3FifoGhostLazyDemotionFastAdmissionCompactHybrid`.
pub type S3FifoGhostLazyDemotionFastAdmissionCompactHybridStack = TieredStack<S3<GhostFilter, true, FastAdmission>>;

impl<G: Ghost, const LAZY: bool, A: Admission> S3<G, LAZY, A> {
	/// The byte capacities of the one-access queue and of main, for a cache of
	/// `max_size`.
	fn capacities(ratio: f64, max_size: CacheSize) -> (CacheSize, CacheSize) {
		((ratio * max_size as f64) as CacheSize, ((1.0 - ratio) * max_size as f64) as CacheSize)
	}

	fn sized(ratio: f64, max_size: CacheSize) -> Self {
		let (one_capacity, main_capacity) = Self::capacities(ratio, max_size);

		S3 { ratio, one_capacity, main_capacity, filled: false, ghost: PhantomData }
	}

	/// The one-access queue's budget and main's (`FastAdmission`): the
	/// carve-out clamped to the tier, the reservation split in proportion.
	fn budgets(s: &TieredStack<Self>) -> (CacheSize, CacheSize) {
		s.carve_budgets(s.policy.one_capacity)
	}

	/// ONE warning to stderr when the configured one-access queue
	/// (`ratio * max_size`) is at least the whole fast tier -- the
	/// configuration the carve-out clamps, in which that queue takes all of the
	/// tier and the main queue gets no fast segment -- and again only when a
	/// later resize makes that NEWLY true. Returns whether it warned.
	///
	/// `eprintln!`, not `log::warn!`: this crate installs no logger (see
	/// `merged_stack.rs`), so a `log` warning would print nowhere. Stderr is
	/// where the crate's other diagnostics go.
	///
	/// Checked from `resize_fast_tier` and `resize`, not from the constructor:
	/// `init_policy_stack` builds this stack against a 20%-of-`max_size`
	/// placeholder and `new_hybrid` sends the real budget through
	/// `resize_fast_tier` straight away, so that is where it first arrives.
	fn warn(s: &mut TieredStack<Self>) -> bool {
		let fast = s.fast_capacity();
		let newly = newly_fills(s.policy.one_capacity, fast, &mut s.policy.filled);

		if newly {
			eprintln!(
				"s3-fifo-ghost-lazy-demotion-fast-admission-compact-hybrid: the one-access queue's configured capacity (one_access_ratio * max_size = {} bytes) meets or exceeds the fast-tier budget ({} bytes); the queue is clamped to the whole fast tier, so the main queue gets no fast segment and every promotion will demote straight back out. Lower the ratio or raise fast_tier_size.",
				s.policy.one_capacity,
				fast,
			);
		}

		newly
	}
}

impl<G: Ghost, const LAZY: bool, A: Admission> TieredStack<S3<G, LAZY, A>> {
	pub fn new(ratio: f64, max_size: CacheSize, fast_capacity: CacheSize) -> Self {
		TieredStack::with_ghost(S3::sized(ratio, max_size), fast_capacity, G::sized_for(max_size))
	}
}

impl<G: Ghost, const LAZY: bool, A: Admission> TierPolicy for S3<G, LAZY, A> {
	type Layout = A::Layout;
	type Ghost = G;

	const LAZY_DEMOTION: bool = LAZY;

	const ADMIT: Lane = ONE;
	const RESETTLE: &'static [Lane] = &[MAIN];
	const ON_TIER_RESIZE: &'static [Lane] = &[MAIN];

	/// With a DRAM one-access queue a resize of the cache moves its carve-out,
	/// and so main's budget.
	const ON_RESIZE: &'static [Lane] = if A::FAST { &[MAIN] } else { &[] };
	const SECOND_CHANCE: Push = Push::IfEndsFast;

	fn is_policy(&self, policy: &PaperPolicy) -> bool {
		match policy {
			PaperPolicy::S3FifoCompactHybrid(ratio) => !G::PRESENT && !LAZY && !A::FAST && *ratio == self.ratio,
			PaperPolicy::S3FifoGhostCompactHybrid(ratio) => G::PRESENT && !LAZY && !A::FAST && *ratio == self.ratio,
			PaperPolicy::S3FifoGhostLazyDemotionCompactHybrid(ratio) => G::PRESENT && LAZY && !A::FAST && *ratio == self.ratio,
			PaperPolicy::S3FifoGhostLazyDemotionFastAdmissionCompactHybrid(ratio) => G::PRESENT && LAZY && A::FAST && *ratio == self.ratio,
			_ => false,
		}
	}

	/// A brand-new key enters the one-access queue; one the ghost remembers
	/// enters main. With a DRAM one-access queue the value is already where it
	/// goes, so a ghost hit is pushed nothing, and a STRUCTURAL key takes a slow
	/// place at main's front instead of the queue.
	fn admit(s: &mut TieredStack<Self>, key: HashedKey, m: Meta) {
		match s.ghost.contains(key) {
			true => s.admit(key, m, MAIN, !A::FAST),
			false if A::FAST && m.structural => s.admit(key, m, MAIN, false),
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

	/// The one-access queue's own budget: raw (`ratio x max_size`), or, when it
	/// is DRAM, its budget net of its share of the reservation, at the drain
	/// target as main rests at the drain target of its own.
	fn wants_eviction(s: &TieredStack<Self>) -> bool {
		match A::FAST {
			true => s.lane_bytes(ONE) > drain_target::bytes(Self::budgets(s).0),
			false => s.lane_bytes(ONE) > s.policy.one_capacity,
		}
	}

	fn resized(s: &mut TieredStack<Self>, max_size: CacheSize) {
		(s.policy.one_capacity, s.policy.main_capacity) = Self::capacities(s.policy.ratio, max_size);

		if A::FAST {
			Self::warn(s);
		}
	}

	fn tier_resized(s: &mut TieredStack<Self>) {
		if A::FAST {
			Self::warn(s);
		}
	}

	/// With a DRAM one-access queue each lane settles against its budget net
	/// of its share of the reservation; else main's is eff.
	fn budget(s: &TieredStack<Self>, lane: Lane) -> CacheSize {
		match (A::FAST, lane) {
			(false, _) => s.eff(),
			(true, ONE) => Self::budgets(s).0,
			(true, _) => Self::budgets(s).1,
		}
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

/// Fast admission's two DRAM segments share one budget. The one-access queue's
/// capacity is sized from the CACHE (`ratio * max_size`), so nothing ties it to
/// the DRAM the fast tier holds: it is a carve-out of the tier only up to what
/// the tier can pay for, and what is left is main's fast segment. The
/// reservation is split in proportion, so that the two budgets plus ONE
/// reservation are the fast tier exactly, for every `(ratio, max_size,
/// fast_capacity)` -- the ones where `ratio * max_size` alone exceeds the tier
/// included, where an unclamped queue would let the stack hold `max(fast_capacity,
/// ratio * max_size)` bytes of DRAM.
#[cfg(test)]
mod fast_budget_tests {
	use super::*;
	use super::super::{tiered_stack::carve::shares, PolicyStack};

	type Stack = S3FifoGhostLazyDemotionFastAdmissionCompactHybridStack;
	type Policy = S3<GhostFilter, true, FastAdmission>;

	/// Both of this design's segments are DRAM (the one-access queue is fast),
	/// so their budgets plus the reservation they were jointly charged for ARE
	/// the fast tier. An admission into a DRAM-resident queue has to draw down
	/// the DRAM budget, not the cache budget.
	///
	/// The configurations that matter are the ones where `ratio * max_size`
	/// alone exceeds `fast_capacity`: without the clamp the queue is capped
	/// ABOVE the whole tier while main's budget saturates to 0, and the settle,
	/// which governs only main, cannot pull any of it back.
	#[test]
	fn the_two_fast_segments_never_exceed_the_fast_budget() {
		const MAX_SIZE: CacheSize = 12_000_000_000;
		const FAST_CAPACITY: CacheSize = 4 * 1024 * 1024 * 1024;

		for ratio in [0.0f64, 0.1, 0.25, 0.3, 0.5, 1.0] {
			let mut stack = Stack::new(ratio, MAX_SIZE, FAST_CAPACITY).with_shared_overhead(224);

			// A populated stack, so the reservation is a real number and the
			// proportional split is actually exercised rather than trivially zero.
			for i in 0..1_000u64 {
				stack.insert(i, 4_096);
			}

			let (one, main) = Policy::budgets(&stack);
			let total = one + main + stack.dram_reserved_bytes();

			assert!(total <= FAST_CAPACITY, "ratio {ratio}: the DRAM-resident caps sum to {total}, over the {FAST_CAPACITY} byte fast tier");

			// And exactly, not merely under: the split hands the remainder to main,
			// so the two shares re-sum to the reservation.
			assert_eq!(total, FAST_CAPACITY, "ratio {ratio}: the split must account for the budget exactly");
		}
	}

	/// The clamp must be invisible to every configuration that already fits --
	/// which is every published sweep -- so those results stay bit-identical.
	#[test]
	fn a_carve_out_that_fits_is_untouched() {
		const MAX_SIZE: CacheSize = 12_000_000_000;
		const FAST_CAPACITY: CacheSize = 4 * 1024 * 1024 * 1024;

		// 0.1 * MAX_SIZE = 1.2e9, comfortably inside the 4 GiB budget.
		let stack = Stack::new(0.1, MAX_SIZE, FAST_CAPACITY).with_shared_overhead(224);
		let one_capacity = stack.policy.one_capacity;
		let (_, main_share) = shares(one_capacity, stack.dram_reserved_bytes(), FAST_CAPACITY);

		assert_eq!(one_capacity.min(FAST_CAPACITY), one_capacity, "a carve-out under the budget must pass through unchanged");
		assert_eq!(Policy::budgets(&stack).1, FAST_CAPACITY - one_capacity - main_share, "and main must still get the plain remainder");
	}
}

/// The carve-out warning: one stderr line per crossing of
/// `one_capacity >= fast_capacity`, checked from both resize entry points.
#[cfg(test)]
mod carve_out_warning_tests {
	use super::*;
	use super::super::PolicyStack;

	type Stack = S3FifoGhostLazyDemotionFastAdmissionCompactHybridStack;
	type Policy = S3<GhostFilter, true, FastAdmission>;

	#[test]
	fn the_carve_out_warning_fires_once_per_crossing() {
		// 0.6 * 1_000 = 600 B of admission queue against a 1_000 B tier: fits.
		let mut stack = Stack::new(0.6, 1_000, 1_000);

		assert!(!Policy::warn(&mut stack), "a queue that fits the tier must not warn");

		stack.resize_fast_tier(600);
		assert!(stack.policy.filled, "resize_fast_tier checks: a 600 B queue on a 600 B tier covers it");
		assert!(!Policy::warn(&mut stack), "once per crossing, not once per check");

		stack.resize_fast_tier(1_000);
		assert!(!stack.policy.filled, "resize_fast_tier re-checks: 600 B fits 1_000 B again");

		stack.resize_fast_tier(400);
		assert!(stack.policy.filled, "resize_fast_tier re-checks: 600 B covers 400 B");

		stack.resize(500);
		assert!(!stack.policy.filled, "resize re-checks: 0.6 * 500 = 300 B fits 400 B");
	}

	/// The slow-admission designs have no carve-out to warn about.
	#[test]
	fn a_slow_one_access_queue_never_warns() {
		let mut stack = S3FifoGhostLazyDemotionCompactHybridStack::new(1.0, 1_000, 1_000);

		stack.resize_fast_tier(400);
		stack.resize(500);

		assert!(!stack.policy.filled);
	}
}

/// The books against the queues after EVERY operation of a long random
/// sequence (`tiered_stack::testing`), the DRAM one-access queue's included.
#[cfg(test)]
mod invariant_tests {
	use super::*;
	use super::super::tiered_stack::testing::books_match_the_queue_after_every_operation;

	#[test]
	fn the_books_match_the_queues_after_every_operation() {
		let stack = S3FifoGhostLazyDemotionFastAdmissionCompactHybridStack::new(0.04, 240_000, 24_000).with_shared_overhead(40);
		let (demoted, promoted) = books_match_the_queue_after_every_operation(stack);

		assert!(demoted > 100, "the sequence never demoted ({demoted})");
		assert!(promoted > 0, "the sequence never promoted ({promoted})");
	}
}

/// With a DRAM one-access queue a resize of the cache settles main, as
/// `resettle` and a resize of the fast tier do: the queue's capacity moves with
/// the cache, and main's budget with that. (With a slow one it settles
/// nothing.)
#[cfg(test)]
mod fast_resize_tests {
	use super::*;
	use super::super::PolicyStack;

	/// A tier of 10_000 B with a 2_000 B one-access queue. Six keys of 1_000 B
	/// are admitted and five of them hit, which promotes them to main, all
	/// fast; then the measured metadata takes 7_000 B of the tier, so main's
	/// budget is 2_400 B and nothing has settled it.
	fn over_its_budget() -> S3FifoGhostLazyDemotionFastAdmissionCompactHybridStack {
		let mut stack = S3FifoGhostLazyDemotionFastAdmissionCompactHybridStack::new(0.02, 100_000, 10_000);

		for key in 1..=6 {
			stack.insert(key, 1_000);
		}

		for key in 1..=5 {
			stack.update(key);
		}

		assert_eq!(stack.fast_bytes_used(), 6_000, "the fixture must leave five keys fast in main and one in the queue");

		drop(stack.drain_tier_migrations());
		stack.set_dram_metadata(Some(7_000));

		stack
	}

	#[test]
	fn a_resize_settles_main() {
		let mut stack = over_its_budget();

		stack.resize(100_000);

		assert_eq!(
			stack.drain_tier_migrations(),
			vec![(1, Tier::Slow), (2, Tier::Slow), (3, Tier::Slow)],
			"main drains to 0.95 x 2_400 B: three of its five keys go",
		);
	}

	#[test]
	fn a_resize_of_the_fast_tier_and_a_resettle_settle_the_same_lane() {
		let (mut by_tier, mut by_resettle) = (over_its_budget(), over_its_budget());

		by_tier.resize_fast_tier(10_000);
		by_resettle.resettle();

		let migrations = by_tier.drain_tier_migrations();

		assert_eq!(migrations.len(), 3, "the fixture must leave main over its budget");
		assert_eq!(migrations, by_resettle.drain_tier_migrations());
	}

	/// The one-access queue's own budget is the carve-out net of its share of
	/// the reservation, at the drain target: 1_400 B of the 2_000 B are the
	/// queue's share, so 600 B x 0.95 = 570 B, and one 1_000 B key is over it.
	#[test]
	fn the_queue_asks_for_an_eviction_over_its_budget_at_the_drain_target() {
		let mut stack = over_its_budget();

		assert!(stack.needs_capacity_eviction(), "1_000 B in a queue whose budget is 570 B");

		stack.set_dram_metadata(None);

		assert!(!stack.needs_capacity_eviction(), "1_000 B against a 2_000 B budget at its drain target of 1_900 B");
	}
}
