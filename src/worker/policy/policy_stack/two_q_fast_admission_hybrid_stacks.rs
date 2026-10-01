/*
 * Copyright (c) Kia Shakiba
 *
 * This source code is licensed under the GNU AGPLv3 license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! Fast-admission 2Q, tier-segmented: a [`TierPolicy`] over a DRAM admission
//! FIFO and a main queue whose fast prefix is the tier (R4; it was a whole
//! stack of its own, `TwoQFastAdmissionReprieveCompactHybridStack`, in a file
//! of its own).
//!
//! The admission FIFO is DRAM-resident, so `placement_of` reports `Fast` for a
//! key in it, its capacity is carved OUT of the fast tier (a FAST lane of the
//! layer, see `tiered_stack::carve`), a promotion out of it emits no migration,
//! and its bytes and objects count toward fast.
//!
//! When the FIFO runs over budget this design REPRIEVES the overflow rather
//! than dropping it: the layer splices the FIFO tail onto the BACK of the main
//! queue as slow, emitting a migration, so an aged-out one-access key gets a
//! second chance in PMEM. (The plain fast-admission 2Q, which let the FIFO
//! grow and asked the caller to evict its tail (`needs_capacity_eviction`),
//! was removed in R2; git history holds it.) Four consequences:
//!
//! - The FIFO is settled after every admission and after either resize, and
//!   `resettle` does. It deliberately is NOT on the re-set path of a tracked
//!   key, nor on a promotion out of the FIFO, which only ever lowers its
//!   bytes.
//! - `needs_capacity_eviction` is NOT overridden. The FIFO polices itself, so
//!   the trait default (`false`) is the answer: an override asking the caller
//!   to evict from a queue that has already settled would be wrong.
//! - `evict_one` drains the MAIN queue first and reaches the FIFO tail only
//!   when main is empty.
//! - The `shared_overhead` reservation is SPLIT between the two queues in
//!   proportion to their fast-tier capacities (`carve::shares`), because both
//!   settle against a budget and each has to pay its own share.
//!
//! A STRUCTURAL new key (S5) takes a slow place at main's BACK instead of the
//! DRAM FIFO, where the reprieve sends a key the FIFO cannot hold: built slow,
//! nothing pushed or settled. A hit, in the FIFO or in main, is `touch`: the
//! key goes to the front of main, and is promoted if it was not fast.
//!
//! **The baseline named above no longer exists in this crate.** Every
//! non-compact hybrid stack was removed once its compact twin was shown
//! behaviourally identical and cheaper: 72 B/object of eviction stack instead
//! of 112 then, and 40 since the arena conversion
//! (`ARENA_STACK_DRAM_OVERHEAD`). Git history holds the baseline and the
//! differential tests that proved the two agreed, and `tier_goldens.txt` holds
//! what this design did, fingerprinted, before it was ported.

use crate::PaperPolicy;

use super::{
	tiered_stack::{newly_fills, End, FastSplit, Lane, Meta, NoGhost, Push, Spill, TierPolicy, TieredStack},
	CacheSize, HashedKey,
};

/// The DRAM admission FIFO and the main queue.
const FIFO: Lane = 0;
const MAIN: Lane = 1;

/// Fast-admission 2Q with a reprieve: `k_in` is the FIFO's share of the cache,
/// `fifo_capacity` its bytes (a carve-out of the fast tier, clamped to it).
pub struct TwoQFar {
	k_in: f64,
	fifo_capacity: CacheSize,

	/// Whether the last check found `fifo_capacity` at least the whole fast
	/// tier: the warning for that crossing has been given.
	filled: bool,
}

/// `PaperPolicy::TwoQFastAdmissionReprieveCompactHybrid`.
pub type TwoQFastAdmissionReprieveCompactHybridStack = TieredStack<TwoQFar>;

impl TieredStack<TwoQFar> {
	pub fn new(k_in: f64, max_size: CacheSize, fast_capacity: CacheSize) -> Self {
		TieredStack::with(TwoQFar { k_in, fifo_capacity: (k_in * max_size as f64) as CacheSize, filled: false }, fast_capacity)
	}
}

impl TwoQFar {
	/// The FIFO's budget and main's: the carve-out clamped to the tier, the
	/// reservation split in proportion.
	fn budgets(s: &TieredStack<Self>) -> (CacheSize, CacheSize) {
		s.carve_budgets(s.policy.fifo_capacity)
	}

	/// ONE warning to stderr when the configured FIFO (`k_in * max_size`) is at
	/// least the whole fast tier -- the configuration the carve-out clamps, in
	/// which the FIFO takes all of the tier and the main queue gets no fast
	/// segment -- and again only when a later resize makes that NEWLY true.
	/// Returns whether it warned.
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
		let newly = newly_fills(s.policy.fifo_capacity, fast, &mut s.policy.filled);

		if newly {
			eprintln!(
				"2q-fast-admission-reprieve-compact-hybrid: the admission FIFO's configured capacity (k_in * max_size = {} bytes) meets or exceeds the fast-tier budget ({} bytes); the FIFO is clamped to the whole fast tier, so the main queue gets no fast segment and every promotion will demote straight back out. Lower k_in or raise fast_tier_size.",
				s.policy.fifo_capacity,
				fast,
			);
		}

		newly
	}
}

impl TierPolicy for TwoQFar {
	type Layout = FastSplit;
	type Ghost = NoGhost;

	const RESETTLE: &'static [Lane] = &[MAIN, FIFO];
	const ON_RESIZE: &'static [Lane] = &[MAIN, FIFO];

	/// The reprieve: the FIFO's overflow goes to the back of main, slow.
	const SPILL: Option<Spill> = Some(Spill { to: MAIN, end: End::Back });

	fn is_policy(&self, policy: &PaperPolicy) -> bool {
		matches!(policy, PaperPolicy::TwoQFastAdmissionReprieveCompactHybrid(k_in) if *k_in == self.k_in)
	}

	/// A brand-new key enters the DRAM FIFO, whose admission spills its tail
	/// into main -- or, STRUCTURAL, goes straight where that spill sends a key
	/// the FIFO cannot hold.
	fn admit(s: &mut TieredStack<Self>, key: HashedKey, m: Meta) {
		match m.structural {
			true => {
				s.place(key, m, MAIN, End::Back);
			},
			false => s.admit(key, m, FIFO, false),
		}
	}

	/// A FIFO key goes to the front of main, and a main key is moved to the
	/// front, promoted if it was demoted.
	fn touch(s: &mut TieredStack<Self>, key: HashedKey, structural: bool) {
		s.to_front(key, MAIN, structural, Push::IfPromoted);
	}

	/// Main first; the FIFO tail only once main is empty -- the reverse of the
	/// non-reprieve stack. The FIFO is policed by its own settle, so its tail
	/// is not eviction's first choice here.
	fn victim(s: &mut TieredStack<Self>) -> Option<HashedKey> {
		s.evict_tail(MAIN).or_else(|| s.evict_tail(FIFO))
	}

	/// Each lane settles against its budget net of its share of the
	/// reservation.
	fn budget(s: &TieredStack<Self>, lane: Lane) -> CacheSize {
		let (fifo, main) = Self::budgets(s);

		match lane {
			FIFO => fifo,
			_ => main,
		}
	}

	fn resized(s: &mut TieredStack<Self>, max_size: CacheSize) {
		s.policy.fifo_capacity = (s.policy.k_in * max_size as f64) as CacheSize;

		Self::warn(s);
	}

	fn tier_resized(s: &mut TieredStack<Self>) {
		Self::warn(s);
	}
}

/// The DRAM ceiling. Both of this design's fast segments -- the admission
/// FIFO, which `placement_of` reports as `Fast` structurally, and the main
/// queue's fast portion -- are budgeted out of one `fast_capacity`, but the
/// FIFO's own cap is `k_in * max_size`, a fraction of the CACHE that nothing
/// ties to the DRAM budget. These pin the clamp that reconciles them,
/// including that it is invisible to configurations that already fit.
#[cfg(test)]
mod dram_ceiling_tests {
	use super::*;
	use super::super::PolicyStack;

	/// The invariant itself, on the config the clamp exists for: 0.6 * 1_000 =
	/// 600 B of FIFO against a 400 B fast tier. Before the clamp the FIFO's
	/// budget was 600 while main saturated to 0, for a real ceiling of 600.
	#[test]
	fn dram_segments_never_over_subscribe_the_fast_tier() {
		let stack = TwoQFastAdmissionReprieveCompactHybridStack::new(0.6, 1_000, 400);
		let (fifo, main) = TwoQFar::budgets(&stack);

		assert_eq!(
			fifo + main,
			stack.fast_capacity(),
			"the two DRAM segments must fill the fast tier exactly, never exceed it",
		);
	}

	/// The clamp must not double-charge the metadata reservation: each segment
	/// subtracts only its OWN share and the two shares re-sum to the
	/// reservation. Checked in the clamped regime, where the FIFO's share is
	/// the whole reservation and main's is nothing.
	#[test]
	fn clamped_segments_still_split_one_reservation() {
		let mut stack = TwoQFastAdmissionReprieveCompactHybridStack::new(0.6, 1_000, 400).with_shared_overhead(10);

		for key in 1..=5 {
			stack.insert(key, 20);
		}

		let (fifo, main) = TwoQFar::budgets(&stack);

		assert_eq!(stack.dram_reserved_bytes(), 50, "five tracked keys at 10 B each");
		assert_eq!(
			fifo + main + stack.dram_reserved_bytes(),
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

		assert_eq!(TwoQFar::budgets(&stack), (250, 150));
	}

	/// Why the clamp is computed and not a value fixed at construction: only
	/// `fast_capacity` moves here, and the FIFO's budget has to move with it.
	/// The reprieve is what makes that enforceable -- the overflow splices into
	/// main as slow rather than waiting on an eviction that this design never
	/// asks for.
	///
	/// The FIFO rests at the DRAIN TARGET of its budget since S5, as main does:
	/// four 50 B keys (200 B) fit 0.95 x 250 = 237 B, a fifth would not (it
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
		assert_eq!(stack.slow_object_count(), 3, "the excess is reprieved into PMEM: one 50 B key fits 0.95 x 100 B");
	}
}

/// The carve-out warning: one stderr line per crossing of
/// `fifo_capacity >= fast_capacity`, checked from both resize entry points.
#[cfg(test)]
mod carve_out_warning_tests {
	use super::*;
	use super::super::PolicyStack;

	#[test]
	fn the_carve_out_warning_fires_once_per_crossing() {
		// 0.6 * 1_000 = 600 B of admission queue against a 1_000 B tier: fits.
		let mut stack = TwoQFastAdmissionReprieveCompactHybridStack::new(0.6, 1_000, 1_000);

		assert!(!TwoQFar::warn(&mut stack), "a queue that fits the tier must not warn");

		stack.resize_fast_tier(600);
		assert!(stack.policy.filled, "resize_fast_tier checks: a 600 B queue on a 600 B tier covers it");
		assert!(!TwoQFar::warn(&mut stack), "once per crossing, not once per check");

		stack.resize_fast_tier(1_000);
		assert!(!stack.policy.filled, "resize_fast_tier re-checks: 600 B fits 1_000 B again");

		stack.resize_fast_tier(400);
		assert!(stack.policy.filled, "resize_fast_tier re-checks: 600 B covers 400 B");

		stack.resize(500);
		assert!(!stack.policy.filled, "resize re-checks: 0.6 * 500 = 300 B fits 400 B");
	}
}

/// The books against the queues after EVERY operation of a long random
/// sequence (`tiered_stack::testing`), the FIFO's included: a count or a byte
/// total that drifts while the order stays right.
#[cfg(test)]
mod invariant_tests {
	use super::*;
	use super::super::tiered_stack::testing::books_match_the_queue_after_every_operation;

	#[test]
	fn the_books_match_the_queues_after_every_operation() {
		let stack = TwoQFastAdmissionReprieveCompactHybridStack::new(0.04, 240_000, 24_000).with_shared_overhead(40);
		let (demoted, _) = books_match_the_queue_after_every_operation(stack);

		assert!(demoted > 100, "the sequence never demoted ({demoted})");
	}
}

/// A resize of the cache settles BOTH lanes, main and then the FIFO, as
/// `resettle` and a resize of the fast tier do: the FIFO's capacity moves with
/// the cache, and main's budget with that. (The plain 2Q's resize settles
/// nothing.) What no recorded run of a few dozen keys shows, and the wider
/// universes of the recorder do.
#[cfg(test)]
mod resize_tests {
	use super::*;
	use super::super::{PolicyStack, Tier};

	/// A tier of 10_000 B with a 2_000 B FIFO. Six keys of 1_000 B are admitted
	/// (each spills the one before it into main, slow) and the five spilled
	/// ones promoted, all fast; then the measured metadata takes 7_000 B of the
	/// tier, so main's budget is 2_400 B and the FIFO's 600 B, and nothing has
	/// settled either.
	fn over_both_budgets() -> TwoQFastAdmissionReprieveCompactHybridStack {
		let mut stack = TwoQFastAdmissionReprieveCompactHybridStack::new(0.02, 100_000, 10_000);

		for key in 1..=6 {
			stack.insert(key, 1_000);
		}

		for key in 1..=5 {
			stack.update(key);
		}

		assert_eq!(stack.fast_bytes_used(), 6_000, "the fixture must leave five keys fast in main and one in the FIFO");

		drop(stack.drain_tier_migrations());
		stack.set_dram_metadata(Some(7_000));

		stack
	}

	#[test]
	fn a_resize_settles_main_and_then_the_fifo() {
		let mut stack = over_both_budgets();

		stack.resize(100_000);

		assert_eq!(
			stack.drain_tier_migrations(),
			vec![(1, Tier::Slow), (2, Tier::Slow), (3, Tier::Slow), (6, Tier::Slow)],
			"main drains to 0.95 x 2_400 B (three keys), then the FIFO to 0.95 x 600 B (its one key)",
		);
	}

	#[test]
	fn a_resize_of_the_fast_tier_and_a_resettle_settle_the_same_lanes() {
		let (mut by_tier, mut by_resettle) = (over_both_budgets(), over_both_budgets());

		by_tier.resize_fast_tier(10_000);
		by_resettle.resettle();

		let migrations = by_tier.drain_tier_migrations();

		assert_eq!(migrations.len(), 4, "the fixture must leave both lanes over their budgets");
		assert_eq!(migrations, by_resettle.drain_tier_migrations());
	}
}
