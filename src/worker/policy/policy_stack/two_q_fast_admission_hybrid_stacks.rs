/*
 * Copyright (c) Kia Shakiba
 *
 * This source code is licensed under the GNU AGPLv3 license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! Fast-admission 2Q, tier-segmented: a [`TierPolicy`] over a DRAM admission
//! FIFO and a main queue whose fast prefix is the tier (R4; it was a whole
//! stack of its own, `TwoQFastAdmissionReprieveCompactHybridStack`, in a file
//! of its own); the full 2Q with fast admission is the last section.
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
//! # The full 2Q
//!
//! `TwoQFull` (`TwoQFullFastAdmissionCompactHybridStack`) is 2Q with all
//! three of its queues and the admission queue in DRAM:
//!
//! | queue | role | tier |
//! |---|---|---|
//! | `a1_in` | probation FIFO for brand-new keys, capped at `k_in * max_size` clamped to the fast tier | **FAST**, structurally: a fast lane |
//! | `a1_out` | overflow FIFO of keys aged out of `a1_in`, capped at `k_out * max_size` | **SLOW**, structurally: a slow lane |
//! | `am` | main LRU of proven keys | tier-**segmented**: a split lane |
//!
//! * `a1_out` holds REAL RESIDENT OBJECTS, not ghosts: it counts toward
//!   `len()` and `contains()`, its bytes toward `slow_bytes_used()`, and a hit
//!   there is a genuine PMEM->DRAM promotion.
//! * An `a1_in` hit is a COMPLETE no-op -- no list move, no tier change, no
//!   migration, no counter. This is the inversion from the plain 2Q, where the
//!   same event is *the* promotion trigger.
//! * `a1_in` overflow DEMOTES into `a1_out`'s front (a `Spill`, pushed
//!   `(key, Slow)`); `a1_out` overflow is what `needs_capacity_eviction`
//!   reports. A `PolicyStack` never self-evicts.
//! * Eviction order is `a1_out`'s tail, then `a1_in`'s, then `am`'s LRU tail.
//! * `a1_in`'s FIXED capacity (never its live usage) is carved out of the
//!   DRAM budget, clamped to what the tier can pay for, and the shared
//!   per-object metadata reservation is charged MAIN FIRST
//!   (`Shares::MainFirst`): `am`'s fast segment pays it out of `fast_capacity -
//!   carve-out`, as it always did, and `a1_in` pays only the part that does
//!   not fit there. Deliberately NOT the proportional split of the 2Q with a
//!   reprieve and the S3-FIFO fast-admission design. They split in proportion
//!   before the clamp reached them, so for them the clamp changed nothing that
//!   fits; this design charged the whole reservation to `am`, and a
//!   proportional split would take `reserved x carve-out / tier` of `a1_in`'s
//!   budget in every configuration with a reservation -- every production run
//!   -- where only the clamp was wanted.
//! * Admission settles `a1_in` BEFORE the push, counting the incoming key
//!   (`settle_with`), and deliberately not after: the carve-out is fixed, so
//!   admission moves `am`'s budget only by the new key's metadata, which `am`
//!   pays first. A re-set that grows an `a1_in` key re-settles `a1_in`; a
//!   resize of the cache or of the fast tier settles both lanes, `a1_in` first.
//! * STRUCTURAL (S5): a new key goes to `a1_out`'s front, slow -- where
//!   `a1_in`'s overflow goes -- built slow and nothing pushed; a key in `a1_in`
//!   that grows structural moves there, pushed `(key, Slow)`.
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
	tiered_stack::{newly_fills, End, FastSlowSplit, FastSplit, Lane, Meta, NoGhost, Push, Shares, Spill, TierPolicy, TieredStack},
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
		s.carve_budgets(s.policy.fifo_capacity, Shares::Proportional)
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

/// The probation FIFO (DRAM), the overflow FIFO (PMEM) and the main LRU.
const A1_IN: Lane = 0;
const A1_OUT: Lane = 1;
const AM: Lane = 2;

/// Full 2Q with fast admission: `k_in` is `a1_in`'s share of the cache and
/// `k_out` `a1_out`'s, with the byte capacities they are held to.
pub struct TwoQFull {
	k_in: f64,
	k_out: f64,

	/// `k_in * max_size`. A reservation carved out of the fast tier (these
	/// bytes are DRAM), but only through [`TieredStack::carve_budgets`], which
	/// clamps it to the tier.
	a1_in_capacity: CacheSize,

	/// `k_out * max_size`. A PMEM budget, carved out of nothing; overrunning it
	/// is what `wants_eviction` reports.
	a1_out_capacity: CacheSize,

	/// Whether the last check found `a1_in_capacity` at least the whole fast
	/// tier: the warning for that crossing has been given.
	filled: bool,
}

/// `PaperPolicy::TwoQFullFastAdmissionCompactHybrid`.
pub type TwoQFullFastAdmissionCompactHybridStack = TieredStack<TwoQFull>;

impl TieredStack<TwoQFull> {
	pub fn new(k_in: f64, k_out: f64, max_size: CacheSize, fast_capacity: CacheSize) -> Self {
		let policy = TwoQFull {
			k_in,
			k_out,
			a1_in_capacity: (k_in * max_size as f64) as CacheSize,
			a1_out_capacity: (k_out * max_size as f64) as CacheSize,
			filled: false,
		};

		TieredStack::with(policy, fast_capacity)
	}
}

impl TwoQFull {
	/// `a1_in`'s budget and `am`'s: the carve-out clamped to the tier, the
	/// reservation charged main first.
	fn budgets(s: &TieredStack<Self>) -> (CacheSize, CacheSize) {
		s.carve_budgets(s.policy.a1_in_capacity, Shares::MainFirst)
	}

	/// ONE warning to stderr when the configured `a1_in` (`k_in * max_size`) is
	/// at least the whole fast tier -- the configuration the carve-out clamps,
	/// in which `a1_in` takes all of the tier and `am` gets no fast segment --
	/// and again only when a later resize makes that NEWLY true. Returns
	/// whether it warned.
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
		let newly = newly_fills(s.policy.a1_in_capacity, fast, &mut s.policy.filled);

		if newly {
			eprintln!(
				"2q-full-fast-admission-compact-hybrid: a1_in's configured capacity (k_in * max_size = {} bytes) meets or exceeds the fast-tier budget ({} bytes); a1_in is clamped to the whole fast tier, so `am` gets no fast segment and every promotion will demote straight back out. Lower k_in or raise fast_tier_size.",
				s.policy.a1_in_capacity,
				fast,
			);
		}

		newly
	}
}

impl TierPolicy for TwoQFull {
	type Layout = FastSlowSplit;
	type Ghost = NoGhost;

	/// `a1_in` first -- its budget moves with the reservation and it is policed
	/// only on an insert, a growth or a resize -- then `am`.
	const RESETTLE: &'static [Lane] = &[A1_IN, AM];
	const ON_RESIZE: &'static [Lane] = &[A1_IN, AM];

	/// `a1_in`'s overflow DEMOTES into the front of `a1_out`, never evicts.
	const SPILL: Option<Spill> = Some(Spill { to: A1_OUT, end: End::Front });

	fn is_policy(&self, policy: &PaperPolicy) -> bool {
		matches!(
			policy,
			PaperPolicy::TwoQFullFastAdmissionCompactHybrid(k_in, k_out)
				if *k_in == self.k_in && *k_out == self.k_out
		)
	}

	/// A brand-new key enters `a1_in`, DRAM: room is made first, counting the
	/// incoming key, and nothing is settled after. A STRUCTURAL one goes where
	/// `a1_in`'s overflow goes, `a1_out`'s front, slow: built slow, nothing
	/// pushed.
	fn admit(s: &mut TieredStack<Self>, key: HashedKey, m: Meta) {
		match m.structural {
			true => {
				s.place(key, m, A1_OUT, End::Front);
			},
			false => {
				s.settle_with(A1_IN, m.size as CacheSize);
				s.place(key, m, A1_IN, End::Front);
			},
		}
	}

	/// A re-set is an access, and an `a1_in` hit is a no-op, so a key that GROWS
	/// there stays there and re-settles `a1_in` rather than leaving the overrun
	/// to the next admission; a STRUCTURAL key in `a1_in` moves to `a1_out`'s
	/// front, pushed `(key, Slow)` (its placement changed).
	fn overwrite(s: &mut TieredStack<Self>, key: HashedKey, m: Meta) {
		let Some((lane, grew)) = s.resize_key(key, m) else { return };

		if m.structural && lane == A1_IN {
			s.to_front(key, A1_OUT, true, Push::IfPromoted);

			return;
		}

		Self::touch(s, key, m.structural);

		if grew && lane == A1_IN {
			s.settle(A1_IN);
		}
	}

	/// An `a1_in` hit does NOTHING: no move, no tier change, no migration. An
	/// `a1_out` or `am` hit moves the key to `am`'s front, promoted if it was
	/// not fast -- a real PMEM to DRAM copy for a key out of `a1_out`.
	fn touch(s: &mut TieredStack<Self>, key: HashedKey, structural: bool) {
		match s.lane_of(key) {
			Some(A1_IN) | None => {},
			Some(_) => s.to_front(key, AM, structural, Push::IfPromoted),
		}
	}

	/// `a1_out`'s tail, then `a1_in`'s, then `am`'s LRU tail.
	fn victim(s: &mut TieredStack<Self>) -> Option<HashedKey> {
		s.evict_tail(A1_OUT).or_else(|| s.evict_tail(A1_IN)).or_else(|| s.evict_tail(AM))
	}

	/// `a1_out` ONLY, raw. `a1_in` overflow is a demotion, handled by its own
	/// settle; reporting it here would evict where the algorithm demotes.
	fn wants_eviction(s: &TieredStack<Self>) -> bool {
		s.lane_bytes(A1_OUT) > s.policy.a1_out_capacity
	}

	fn resized(s: &mut TieredStack<Self>, max_size: CacheSize) {
		s.policy.a1_in_capacity = (s.policy.k_in * max_size as f64) as CacheSize;
		s.policy.a1_out_capacity = (s.policy.k_out * max_size as f64) as CacheSize;

		Self::warn(s);
	}

	fn tier_resized(s: &mut TieredStack<Self>) {
		Self::warn(s);
	}

	/// `a1_in` and `am` each settle against their budget net of their share of
	/// the reservation; `a1_out` is PMEM and has none.
	fn budget(s: &TieredStack<Self>, lane: Lane) -> CacheSize {
		let (a1_in, am) = Self::budgets(s);

		match lane {
			A1_IN => a1_in,
			_ => am,
		}
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

/// The DRAM ceiling of the full 2Q, ported with the clamp from the
/// reprieve design's `dram_ceiling_tests`.
/// Both DRAM segments -- `a1_in` and the main queue's fast portion --
/// are budgeted out of one `fast_capacity`, but `a1_in`'s own capacity
/// is `k_in * max_size`, a fraction of the CACHE.
///
/// `RATIOS` put that at the 4_000 B tier and above it. Only 0.6 and 1.0
/// detect a missing clamp: 0.4 is the equality case, where `min` returns the
/// raw capacity and the clamp is numerically inert -- it is there to show a
/// carve-out that exactly fills the tier does not wedge. `FITTING` (0.25)
/// covers the regime where the carve-out fits and the reservation lands on
/// `am` first, and pins those budgets to the pre-clamp ones.
#[cfg(test)]
mod full_dram_ceiling_tests {
	use super::*;
	use super::super::{tiered_stack::carve::shares, PolicyStack, Tier};
	use crate::object::ObjectSize;

	const MAX_SIZE: CacheSize = 10_000;
	const FAST: CacheSize = 4_000;
	const SIZE: ObjectSize = 100;
	const OVERHEAD: CacheSize = 8;
	const KEYS: HashedKey = 120;

	/// The settle of `a1_in` runs BEFORE the push (`settle_with`), so it
	/// cannot see the incoming key's own metadata. At the clamped `RATIOS`
	/// `am` has no room for any of the reservation, `a1_in` pays all of it,
	/// and right after an admission `a1_in` may sit over its budget by that
	/// key's one `OVERHEAD`. The settle at the end of the drive must be exact.
	const SLACK: CacheSize = OVERHEAD;

	/// `k_in * MAX_SIZE` = 4_000 B (exactly the tier), 6_000 B and
	/// 10_000 B.
	const RATIOS: [f64; 3] = [0.4, 0.6, 1.0];

	/// `k_in * MAX_SIZE` = 2_500 B: the carve-out fits the 4_000 B tier and
	/// leaves `am` 1_500 B, which the drive's reservation (at most 100 keys at
	/// 8 B) never outgrows.
	const FITTING: f64 = 0.25;

	fn stack(ratio: f64) -> TwoQFullFastAdmissionCompactHybridStack {
		TwoQFullFastAdmissionCompactHybridStack::new(ratio, 0.5, MAX_SIZE, FAST).with_shared_overhead(OVERHEAD)
	}

	/// What `PolicyWorker::apply_evictions` does after every event: evict
	/// while the stack asks for it or the cache is over `max_size`.
	fn evict_while_asked(stack: &mut TwoQFullFastAdmissionCompactHybridStack) {
		while (stack.needs_capacity_eviction()
			|| stack.fast_bytes_used() + stack.slow_bytes_used() > MAX_SIZE)
			&& stack.evict_one().is_some()
		{}
	}

	/// All the DRAM this stack holds against the tier it was given: both
	/// segments' values (`fast_bytes_used`) plus the metadata reservation it
	/// reports.
	fn assert_within_the_fast_tier(stack: &TwoQFullFastAdmissionCompactHybridStack, slack: CacheSize, context: &str) {
		let values = stack.fast_bytes_used();
		let reserved = stack.dram_reserved_bytes();

		assert!(
			values + reserved <= stack.fast_capacity() + slack,
			"{context}: {} B of DRAM ({values} B of values + {reserved} B reserved) on a {} B fast tier",
			values + reserved,
			stack.fast_capacity(),
		);
	}

	/// The budget identity: both DRAM segments' budgets plus ONE reservation
	/// are the fast tier exactly, however far `k_in * max_size`
	/// overshoots it. Unclamped, 0.6 gave `a1_in` a 6_000 B budget of its
	/// own on a 4_000 B tier while the main queue's budget saturated to 0.
	#[test]
	fn dram_budgets_never_over_subscribe_the_fast_tier() {
		for ratio in RATIOS {
			let mut stack = stack(ratio);

			for key in 1..=5 {
				stack.insert(key, SIZE);
			}

			let total = TwoQFull::budgets(&stack).0
				+ TwoQFull::budgets(&stack).1
				+ stack.dram_reserved_bytes();

			assert_eq!(
				total, FAST,
				"ratio {ratio}: the two DRAM budgets plus the reservation come to {total} B, not the {FAST} B fast tier",
			);
		}
	}

	/// The ceiling on live bytes, driven the way the worker drives the stack:
	/// admissions, hits on the newest keys (no-ops in `a1_in`) and on older ones
	/// (promotions out of `a1_out` into an `am` with no fast segment left),
	/// re-admissions of early keys, and every eviction the stack asks for.
	/// Includes a carve-out exactly the size of the tier, which must not wedge
	/// or panic.
	///
	/// Per-step checks allow `SLACK`; the final re-settle allows nothing.
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

			// A settle with every key's metadata in view: exact.
			stack.resize_fast_tier(FAST);
			assert_within_the_fast_tier(&stack, 0, &format!("ratio {ratio}, re-settled"));

			assert!(stack.len() > 0, "ratio {ratio}: the stack must still hold keys");
		}
	}

	/// Why the clamp is an accessor and not a value fixed at construction: only
	/// `fast_capacity` moves here, and `a1_in`'s budget has to move with
	/// it. `resize_fast_tier` re-settles `a1_in`, which
	/// demotes the excess into `a1_out` -- nothing is lost.
	#[test]
	fn shrinking_the_fast_tier_settles_a1_in_to_the_new_clamp() {
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

		assert!(
			stack.lane_bytes(A1_IN) <= TwoQFull::budgets(&stack).0,
			"a1_in holds {} B over its {} B budget",
			stack.lane_bytes(A1_IN),
			TwoQFull::budgets(&stack).0,
		);

		assert_eq!(stack.len(), 5, "nothing is evicted: the excess is demoted");
		assert_eq!(stack.slow_object_count(), 3, "the three oldest went to PMEM");
	}

	/// The budget identity where the carve-out FITS, with a reservation: `am`
	/// pays it first, and once it outgrows `am`'s 1_500 B `a1_in` pays the
	/// rest, up to a reservation that is the whole tier.
	#[test]
	fn a_fitting_carve_out_keeps_the_budget_identity() {
		// Five keys: 0, 40, 500, 2_000 and 4_000 B reserved.
		for overhead in [0, OVERHEAD, 100, 400, 800] {
			let mut stack =
				TwoQFullFastAdmissionCompactHybridStack::new(FITTING, 0.5, MAX_SIZE, FAST).with_shared_overhead(overhead);

			for key in 1..=5 {
				stack.insert(key, SIZE);
			}

			let total = TwoQFull::budgets(&stack).0
				+ TwoQFull::budgets(&stack).1
				+ stack.dram_reserved_bytes();

			assert_eq!(
				total, FAST,
				"{} B reserved: the two DRAM budgets plus the reservation come to {total} B, not the {FAST} B fast tier",
				stack.dram_reserved_bytes(),
			);
		}
	}

	/// The ceiling on live bytes where the carve-out fits, driven as the
	/// clamped ratios are above.
	///
	/// `a1_in`'s budget here is its whole carve-out whatever the reservation,
	/// so its own settle is exact. `am` pays the reservation, and `am` settles
	/// on an `a1_out` promotion, an `am` hit and a resize -- not on an
	/// admission, and not on an `a1_in` hit (a no-op), exactly as before the
	/// clamp -- so each admission since `am` last settled can add one
	/// `OVERHEAD` it has not yet settled against. The bound is therefore exact
	/// after every event that settles `am` and over by at most the metadata of
	/// the keys admitted since; the final re-settle is exact. (In this drive
	/// the headroom `drain_target` leaves absorbs that metadata and no event
	/// goes over the tier at all; the bound is what the code guarantees.)
	#[test]
	fn a_fitting_carve_out_holds_the_ceiling_through_admissions_and_hits() {
		let mut stack = stack(FITTING);
		let mut unsettled: CacheSize = 0;

		let settles_am = |stack: &TwoQFullFastAdmissionCompactHybridStack, key: HashedKey| {
			matches!(stack.lane_of(key), Some(A1_OUT | AM))
		};

		for key in 1..=KEYS {
			stack.insert(key, SIZE);
			unsettled += 1;
			evict_while_asked(&mut stack);
			assert_within_the_fast_tier(
				&stack,
				unsettled * OVERHEAD,
				&format!("admitted {key}, {unsettled} admission(s) since am settled"),
			);

			if key % 3 == 0 {
				for hit in [key - 1, key / 2] {
					if stack.contains(hit) {
						if settles_am(&stack, hit) {
							unsettled = 0;
						}

						stack.update(hit);
						evict_while_asked(&mut stack);
						assert_within_the_fast_tier(&stack, unsettled * OVERHEAD, &format!("hit {hit}"));
					}
				}
			}
		}

		for key in 1..=KEYS / 3 {
			if !stack.contains(key) {
				unsettled += 1;
			} else if settles_am(&stack, key) {
				unsettled = 0;
			}

			stack.insert(key, SIZE);
			evict_while_asked(&mut stack);
			assert_within_the_fast_tier(&stack, unsettled * OVERHEAD, &format!("re-admitted {key}"));
		}

		stack.resize_fast_tier(FAST);
		assert_within_the_fast_tier(&stack, 0, "re-settled");

		assert!(
			stack.dram_reserved_bytes() <= FAST - stack.policy.a1_in_capacity.min(stack.fast_capacity()),
			"the drive left the fitting regime: {} B reserved",
			stack.dram_reserved_bytes(),
		);
		assert!(stack.len() > 0, "the stack must still hold keys");
	}

	/// The main-first split itself, pinned in every regime (five keys each).
	#[test]
	fn main_first_shares_pin_every_regime() {
		// (ratio, overhead per key, (a1_in_share, am_share), a1_in budget, am budget)
		let cases: [(f64, CacheSize, (CacheSize, CacheSize), CacheSize, CacheSize); 4] = [
			// Carve-out 2_500 B, 40 B reserved: `am` pays all of it, and
			// `a1_in` keeps its whole carve-out.
			(FITTING, OVERHEAD, (0, 40), 2_500, 1_460),
			// 2_000 B reserved: `am` can pay 1_500 of it, `a1_in` the other 500.
			(FITTING, 400, (500, 1_500), 2_000, 0),
			// 5_000 B reserved, more than the tier: no value budget left.
			(FITTING, 1_000, (3_500, 1_500), 0, 0),
			// Clamped carve-out (6_000 B -> the 4_000 B tier), 40 B reserved:
			// `am` has no room, so `a1_in` pays it all.
			(0.6, OVERHEAD, (40, 0), 3_960, 0),
		];

		for (ratio, overhead, split, a1_in, am) in cases {
			let mut stack =
				TwoQFullFastAdmissionCompactHybridStack::new(ratio, 0.5, MAX_SIZE, FAST).with_shared_overhead(overhead);

			for key in 1..=5 {
				stack.insert(key, SIZE);
			}

			let context = format!("ratio {ratio}, {} B reserved", stack.dram_reserved_bytes());

			assert_eq!(shares(stack.policy.a1_in_capacity, Shares::MainFirst, stack.dram_reserved_bytes(), stack.fast_capacity()), split, "{context}: (a1_in_share, am_share)");
			assert_eq!(TwoQFull::budgets(&stack).0, a1_in, "{context}: a1_in budget");
			assert_eq!(TwoQFull::budgets(&stack).1, am, "{context}: am budget");
		}
	}

	/// The point of main-first: wherever the reservation fits beside the
	/// carve-out, both budgets are 34c6a4e's. That commit settled `am` against
	/// `fast_capacity.saturating_sub(a1_in_capacity).saturating_sub(
	/// reserved_overhead())` and drained `a1_in` against the raw
	/// `a1_in_capacity`; both are restated here verbatim. `am`'s formula holds
	/// for EVERY input; `a1_in`'s wherever `reserved <= fast - carve`, except
	/// that above the tier (where that forces `reserved == 0`) `a1_in`'s
	/// budget is the clamp's `fast` rather than the raw capacity.
	#[test]
	fn budgets_equal_the_pre_clamp_formulas_wherever_the_reservation_fits_beside_the_carve_out() {
		let mut checked = 0;

		for ratio in [0.0, 0.1, FITTING, 0.39, 0.4, 0.41, 0.6, 1.0] {
			for fast in [0, 1, 1_000, 2_500, FAST, MAX_SIZE] {
				for overhead in [0, 1, OVERHEAD, 100, 1_000] {
					for keys in [0, 1, 5, 20] {
						let mut stack = TwoQFullFastAdmissionCompactHybridStack::new(ratio, 0.5, MAX_SIZE, fast)
							.with_shared_overhead(overhead);

						for key in 1..=keys {
							stack.insert(key, 1);
						}

						let a1_in_capacity = stack.policy.a1_in_capacity;
						let reserved = stack.dram_reserved_bytes();
						let carve = a1_in_capacity.min(fast);
						let context = format!("ratio {ratio}, fast {fast}, {reserved} B reserved");

						assert_eq!(
							TwoQFull::budgets(&stack).1,
							fast.saturating_sub(a1_in_capacity).saturating_sub(reserved),
							"{context}: am's budget moved from 34c6a4e's",
						);

						if reserved > fast - carve {
							continue;
						}

						let pre_clamp_a1_in = if a1_in_capacity <= fast { a1_in_capacity } else { fast };

						assert_eq!(
							TwoQFull::budgets(&stack).0,
							pre_clamp_a1_in,
							"{context}: a1_in's budget moved from 34c6a4e's",
						);

						checked += 1;
					}
				}
			}
		}

		assert!(checked >= 300, "only {checked} fitting configurations were checked");
	}

	/// A reservation at or over the whole tier. `am`'s budget and `a1_in`'s
	/// are both 0, so eff is 0 and every new key is STRUCTURAL (S5): it goes
	/// where `a1_in`'s overflow goes, `a1_out`'s front, slow -- built slow,
	/// nothing pushed -- and no value is left in DRAM. (Before S5 each
	/// admission entered `a1_in`, demoting the one before it, so `a1_in` held
	/// the newest key in DRAM on a tier the metadata had filled.) Nothing is
	/// evicted for it. Before the clamp and the split `a1_in` kept its whole
	/// raw capacity in DRAM on top of the reservation.
	#[test]
	fn a_metadata_bound_tier_places_each_new_key_in_a1_out() {
		// k_in 0.01: a 100 B `a1_in`, so each admission demotes the key before
		// it into `a1_out`, where a hit proves it into `am`.
		let mut stack =
			TwoQFullFastAdmissionCompactHybridStack::new(0.01, 0.5, MAX_SIZE, FAST).with_shared_overhead(100);

		for key in 1..=31 {
			stack.insert(key, SIZE);

			if key > 1 {
				stack.update(key - 1);
			}

			evict_while_asked(&mut stack);
		}

		assert_eq!(stack.len(), 31, "every key is still tracked");
		assert_eq!(stack.dram_reserved_bytes(), 3_100, "31 keys at 100 B, which still fits the 4_000 B tier");

		// The tier shrinks under the reservation.
		stack.resize_fast_tier(2_000);
		evict_while_asked(&mut stack);

		assert_eq!(shares(stack.policy.a1_in_capacity, Shares::MainFirst, stack.dram_reserved_bytes(), stack.fast_capacity()), (1_200, 1_900));
		assert_eq!(TwoQFull::budgets(&stack).0, 0, "no a1_in budget left");
		assert_eq!(TwoQFull::budgets(&stack).1, 0, "no am budget left");
		assert_eq!(stack.fast_bytes_used(), 0, "a1_in and am demoted every value");

		stack.drain_tier_migrations();

		for key in 100..110 {
			stack.insert(key, SIZE);
			evict_while_asked(&mut stack);

			assert_eq!(stack.fast_bytes_used(), 0, "key {key}: no value is in DRAM");
			assert_eq!(stack.tier_of(key), Some(Tier::Slow), "key {key} was placed in a1_out");
			assert!(stack.drain_tier_migrations().is_empty(), "key {key}: built slow, nothing pushed");
			assert_eq!(stack.len(), 31 + (key - 99) as usize, "key {key}: nothing is evicted");
		}
	}

	/// A re-set is a hit, and an `a1_in` hit is a no-op, so a key that GROWS
	/// in `a1_in` stays there. `insert_resident` then re-settles `a1_in`
	/// rather than leaving the overrun to the next admission.
	#[test]
	fn re_setting_a_key_larger_in_a1_in_re_settles_a1_in() {
		let mut stack = stack(FITTING);

		// Forty admissions: `a1_in` holds the 23 newest (2_300 B: it rests at
		// the drain target of its 2_500 B budget since S5, 2_375 B at 0.95 and
		// 2_450 B, 24 keys, at the 0.98 it was until E1b), `a1_out` the 17
		// oldest.
		for key in 1..=40 {
			stack.insert(key, SIZE);
			evict_while_asked(&mut stack);
		}

		// Proving 15 of those fills `am`'s fast segment to its budget.
		for key in 1..=15 {
			stack.update(key);
			evict_while_asked(&mut stack);
		}

		assert_eq!(stack.lane_bytes(A1_IN), 2_300);
		assert_within_the_fast_tier(&stack, 0, "before the re-set");

		// Key 40 is `a1_in`'s newest; re-set it 300 B larger.
		stack.insert(40, 400);
		evict_while_asked(&mut stack);

		assert!(
			stack.lane_bytes(A1_IN) <= TwoQFull::budgets(&stack).0,
			"a1_in holds {} B over its {} B budget",
			stack.lane_bytes(A1_IN),
			TwoQFull::budgets(&stack).0,
		);
		assert_within_the_fast_tier(&stack, 0, "after re-setting key 40 to 400 B");

		assert_eq!(stack.tier_of(40), Some(Tier::Fast), "the re-set key is still in a1_in");

		for key in 18..=20 {
			assert_eq!(stack.tier_of(key), Some(Tier::Slow), "key {key}, a1_in's oldest, was demoted");
		}

		assert_eq!(stack.len(), 40, "nothing is evicted: the excess is demoted");
	}

	/// Only a GROWTH re-settles `a1_in`: a re-set at the same size, with
	/// `a1_in` over a budget the reservation has since squeezed, moves nothing
	/// -- the next admission or resize polices it.
	#[test]
	fn re_setting_a_key_at_the_same_size_in_a1_in_settles_nothing() {
		let mut stack = stack(FITTING);

		for key in 1..=40 {
			stack.insert(key, SIZE);
			evict_while_asked(&mut stack);
		}

		for key in 1..=15 {
			stack.update(key);
			evict_while_asked(&mut stack);
		}

		// A tenfold reservation: `am` pays 1_500 B of it, `a1_in` the rest, and
		// its 2_300 B are now over its budget.
		let mut stack = stack.with_shared_overhead(OVERHEAD * 10);
		let held = stack.lane_bytes(A1_IN);

		assert!(held > TwoQFull::budgets(&stack).0, "a1_in is not over its budget ({held} B)");

		stack.drain_tier_migrations();
		stack.insert(40, SIZE);

		assert!(stack.drain_tier_migrations().is_empty(), "a re-set at the same size demoted");
		assert_eq!(stack.lane_bytes(A1_IN), held, "a re-set at the same size settled a1_in");

		// One byte larger is a growth: it settles.
		stack.insert(40, SIZE + 1);

		assert!(stack.lane_bytes(A1_IN) < held, "a growth did not settle a1_in");
	}
}

/// The carve-out warning: one stderr line per crossing of
/// `a1_in_capacity >= fast_capacity`, checked from both resize entry points.
#[cfg(test)]
mod full_carve_out_warning_tests {
	use super::*;
	use super::super::PolicyStack;

	#[test]
	fn the_carve_out_warning_fires_once_per_crossing() {
		// 0.6 * 1_000 = 600 B of admission queue against a 1_000 B tier: fits.
		let mut stack = TwoQFullFastAdmissionCompactHybridStack::new(0.6, 0.5, 1_000, 1_000);

		assert!(!TwoQFull::warn(&mut stack), "a queue that fits the tier must not warn");

		stack.resize_fast_tier(600);
		assert!(stack.policy.filled, "resize_fast_tier checks: a 600 B queue on a 600 B tier covers it");
		assert!(!TwoQFull::warn(&mut stack), "once per crossing, not once per check");

		stack.resize_fast_tier(1_000);
		assert!(!stack.policy.filled, "resize_fast_tier re-checks: 600 B fits 1_000 B again");

		stack.resize_fast_tier(400);
		assert!(stack.policy.filled, "resize_fast_tier re-checks: 600 B covers 400 B");

		stack.resize(500);
		assert!(!stack.policy.filled, "resize re-checks: 0.6 * 500 = 300 B fits 400 B");
	}
}

/// A stack answers to its own policy and to no other: both ratios count, each
/// alone.
#[cfg(test)]
mod full_is_policy_tests {
	use super::*;
	use super::super::PolicyStack;

	#[test]
	fn the_policy_is_both_ratios() {
		let stack = TwoQFullFastAdmissionCompactHybridStack::new(0.25, 0.5, 10_000, 4_000);
		let full = PaperPolicy::TwoQFullFastAdmissionCompactHybrid;

		assert!(stack.is_policy(&full(0.25, 0.5)));
		assert!(!stack.is_policy(&full(0.5, 0.5)), "k_in is not compared");
		assert!(!stack.is_policy(&full(0.25, 0.4)), "k_out is not compared");
		assert!(!stack.is_policy(&PaperPolicy::TwoQFastAdmissionReprieveCompactHybrid(0.25)));
	}
}

/// The books against the queues after EVERY operation of a long random
/// sequence (`tiered_stack::testing`): the fast lane, the slow lane and the
/// split one.
#[cfg(test)]
mod full_invariant_tests {
	use super::*;
	use super::super::tiered_stack::testing::books_match_the_queue_after_every_operation;

	#[test]
	fn the_books_match_the_queues_after_every_operation() {
		let stack = TwoQFullFastAdmissionCompactHybridStack::new(0.04, 0.5, 240_000, 24_000).with_shared_overhead(40);
		let (demoted, _) = books_match_the_queue_after_every_operation(stack);

		assert!(demoted > 100, "the sequence never demoted ({demoted})");
	}
}

/// A resize of the cache settles BOTH lanes, `a1_in` and then `am`, as
/// `resettle` and a resize of the fast tier do.
#[cfg(test)]
mod full_resize_tests {
	use super::*;
	use super::super::{PolicyStack, Tier};

	/// A tier of 10_000 B with a 2_000 B `a1_in`. Six keys of 1_000 B are
	/// admitted (each demotes the one before it into `a1_out`) and the five
	/// demoted ones promoted into `am`, all fast; then the measured metadata
	/// takes 9_500 B of the tier, so `am` can pay 8_000 B of it, `a1_in` the
	/// other 1_500 B and nothing has settled either.
	fn over_both_budgets() -> TwoQFullFastAdmissionCompactHybridStack {
		let mut stack = TwoQFullFastAdmissionCompactHybridStack::new(0.02, 0.5, 100_000, 10_000);

		for key in 1..=6 {
			stack.insert(key, 1_000);
		}

		for key in 1..=5 {
			stack.update(key);
		}

		assert_eq!(stack.fast_bytes_used(), 6_000, "the fixture must leave five keys fast in am and one in a1_in");

		drop(stack.drain_tier_migrations());
		stack.set_dram_metadata(Some(9_500));

		stack
	}

	#[test]
	fn a_resize_settles_a1_in_and_then_am() {
		let mut stack = over_both_budgets();

		stack.resize(100_000);

		assert_eq!(
			stack.drain_tier_migrations(),
			vec![(6, Tier::Slow), (1, Tier::Slow), (2, Tier::Slow), (3, Tier::Slow), (4, Tier::Slow), (5, Tier::Slow)],
			"a1_in drains to 0.95 x 500 B (its one key goes), then am to nothing (its five)",
		);
	}

	#[test]
	fn a_resize_of_the_fast_tier_and_a_resettle_settle_the_same_lanes() {
		let (mut by_tier, mut by_resettle) = (over_both_budgets(), over_both_budgets());

		by_tier.resize_fast_tier(10_000);
		by_resettle.resettle();

		let migrations = by_tier.drain_tier_migrations();

		assert_eq!(migrations.len(), 6, "the fixture must leave both lanes over their budgets");
		assert_eq!(migrations, by_resettle.drain_tier_migrations());
	}
}
