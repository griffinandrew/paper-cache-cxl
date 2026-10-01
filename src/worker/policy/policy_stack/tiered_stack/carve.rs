/*
 * Copyright (c) Kia Shakiba
 *
 * This source code is licensed under the GNU AGPLv3 license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! The fast lane of the tiering layer (R4): the DRAM admission queue of the
//! fast-admission designs, entirely fast and a carve-out of the fast tier.
//!
//! The queue's capacity is a fraction of the CACHE (`k x max_size`), and
//! nothing ties that to the DRAM the fast tier holds, so the carve-out is the
//! capacity up to what the tier can pay for -- `min(capacity, fast tier)` --
//! and what is left is the main lane's. The metadata reservation is paid out
//! of the same tier and split between the two lanes ([`Shares`]); what each
//! is left is its BUDGET, the figure its settle drains from ([`budgets`]), and
//! while the reservation fits the tier the two budgets and the reservation are
//! the fast tier exactly. It is the capacity that is carved out, not the
//! queue's live bytes, so an admission moves the main lane's budget only by
//! the metadata of the new key.
//!
//! What the lane does with the bytes over its budget is the design's: evict
//! them (`TierPolicy::wants_eviction`), or splice its tail into another lane
//! as slow ([`Spill`]) -- a demotion, an unlink and a relink of the same
//! slot, pushed `(key, Slow)`.
//!
//! A configured carve-out at least the whole tier takes all of it and leaves
//! the main lane none, so every promotion demotes straight back out: the
//! design warns once, when that is first true, and again only when a resize
//! makes it newly true ([`newly_fills`]).

use super::{idx, CacheSize, End, Lane, Lanes, Layout, Spill, Tier, TierPolicy, TieredStack};

/// How the metadata reservation is split between the carve-out lane and the
/// main lane.
#[derive(Clone, Copy)]
pub enum Shares {
	/// In proportion to their shares of the tier: the carve-out pays
	/// `reserved x carve-out / tier`, main the rest.
	Proportional,

	/// Main first: it pays out of what the carve-out leaves it, and the
	/// carve-out only what does not fit there. Where the carve-out fits the
	/// tier beside the reservation, the carve-out's budget is its whole
	/// capacity whatever the reservation.
	MainFirst,
}

/// The reservation's split `(carve-out's share, main's share)`, for a lane of
/// `capacity` bytes configured on a fast tier of `fast`. The two re-sum to
/// `reserved` (but for a proportional split, to nothing on no tier at all).
pub fn shares(capacity: CacheSize, how: Shares, reserved: CacheSize, fast: CacheSize) -> (CacheSize, CacheSize) {
	let carve = capacity.min(fast);

	match how {
		Shares::Proportional => {
			if fast == 0 {
				return (0, 0);
			}

			// Widened to u128: `reserved x carve` overflows u64 at realistic entry
			// counts.
			let share = ((reserved as u128 * carve as u128) / fast as u128) as CacheSize;

			(share, reserved.saturating_sub(share))
		},

		Shares::MainFirst => {
			let main = reserved.min(fast.saturating_sub(carve));

			(reserved - main, main)
		},
	}
}

/// The carve-out lane's budget and the main lane's, before the drain target:
/// what each may hold of values once the carve-out is taken and the
/// reservation paid. Saturating, so a carve-out and a reservation that meet
/// or exceed the tier leave a lane none rather than fail.
pub fn budgets(capacity: CacheSize, how: Shares, reserved: CacheSize, fast: CacheSize) -> (CacheSize, CacheSize) {
	let carve = capacity.min(fast);
	let (carve_share, main_share) = shares(capacity, how, reserved, fast);

	(carve.saturating_sub(carve_share), fast.saturating_sub(carve).saturating_sub(main_share))
}

/// Whether the configured carve-out is at least the whole fast tier -- the
/// configuration `budgets` clamps -- and this is the first check to find it
/// so: `filled` remembers the last answer, so a design warns once per crossing.
pub fn newly_fills(capacity: CacheSize, fast: CacheSize, filled: &mut bool) -> bool {
	let fills = fast > 0 && capacity >= fast;
	let newly = fills && !*filled;

	*filled = fills;

	newly
}

impl<L: Layout> Lanes<L> {
	/// Splices the tail of the fast lane `from` onto an end of another lane,
	/// slow, while `incoming` bytes beside what `from` holds are over `target`
	/// (a design that makes room before it places a key): each key a demotion,
	/// pushed `(key, Slow)`.
	pub(super) fn spill(&mut self, from: Lane, target: CacheSize, incoming: CacheSize, spill: Spill) {
		while self.books[from].bytes[idx(Tier::Fast)] + incoming > target {
			let Some(key) = self.set.back(from) else { break };
			let Some(payload) = self.set.payload(key) else { break };
			let size = payload.migrating();

			match spill.end {
				End::Front => self.set.move_to_front_of(from, spill.to, key),
				End::Back => self.set.move_to_back_of(from, spill.to, key),
			}

			self.debit(from, Tier::Fast, size);
			self.credit(spill.to, Tier::Slow, size);

			if let Some(slot) = self.set.payload_mut(key) {
				slot.queue = spill.to as u8;
				slot.tier = Some(Tier::Slow);
			}

			self.log.push((key, Tier::Slow));
		}
	}
}

impl<P: TierPolicy> TieredStack<P> {
	/// The budgets of a carve-out lane configured at `capacity` and of the
	/// main lane, against this stack's tier and reservation.
	pub fn carve_budgets(&self, capacity: CacheSize, how: Shares) -> (CacheSize, CacheSize) {
		budgets(capacity, how, self.reserved(), self.fast_capacity)
	}
}

#[cfg(test)]
mod tests {
	use super::*;

	const HOW: [Shares; 2] = [Shares::Proportional, Shares::MainFirst];

	/// The split re-sums to the reservation, whatever the carve-out is, and the
	/// two budgets and the reservation are the fast tier exactly while the
	/// reservation fits in it -- clamped carve-outs included.
	#[test]
	fn the_budgets_and_the_reservation_are_the_fast_tier_while_it_fits() {
		let mut checked = 0;

		for how in HOW {
			for capacity in [0, 250, 1_000, 4_000, 6_000, 100_000] {
				for fast in [0, 1, 400, 1_000, 4_000] {
					for reserved in [0, 1, 10, 50, 600, 3_999, 4_000, 5_000] {
						let (carve_share, main_share) = shares(capacity, how, reserved, fast);

						if fast > 0 || matches!(how, Shares::MainFirst) {
							assert_eq!(carve_share + main_share, reserved, "{capacity} B on {fast} B, {reserved} B reserved");
						}

						if reserved <= fast {
							let (carve, main) = budgets(capacity, how, reserved, fast);

							assert_eq!(carve + main + reserved, fast, "{capacity} B on {fast} B, {reserved} B reserved");

							checked += 1;
						}
					}
				}
			}
		}

		assert!(checked > 100, "only {checked} fitting configurations were checked");
	}

	/// Proportional: the carve-out pays its share of the tier, rounded down; one
	/// larger than the tier is clamped to it, and leaves main nothing to pay
	/// with. Main first: main pays out of what the carve-out leaves it, and the
	/// carve-out only the rest.
	#[test]
	fn the_two_splits_charge_the_reservation_as_they_say() {
		// A 2_500 B carve-out on a 4_000 B tier.
		assert_eq!(shares(2_500, Shares::Proportional, 40, 4_000), (25, 15));
		assert_eq!(shares(2_500, Shares::Proportional, 2_000, 4_000), (1_250, 750));
		assert_eq!(budgets(2_500, Shares::Proportional, 40, 4_000), (2_475, 1_485));

		assert_eq!(shares(2_500, Shares::MainFirst, 40, 4_000), (0, 40));
		assert_eq!(shares(2_500, Shares::MainFirst, 2_000, 4_000), (500, 1_500));
		assert_eq!(budgets(2_500, Shares::MainFirst, 40, 4_000), (2_500, 1_460));
		assert_eq!(budgets(2_500, Shares::MainFirst, 2_000, 4_000), (2_000, 0));

		// Clamped to the tier: main has nothing to pay with.
		assert_eq!(shares(6_000, Shares::Proportional, 40, 4_000), (40, 0));
		assert_eq!(shares(6_000, Shares::MainFirst, 40, 4_000), (40, 0));
		assert_eq!(budgets(6_000, Shares::MainFirst, 40, 4_000), (3_960, 0));

		// Past the tier: no value budget left in either lane.
		assert_eq!(budgets(2_500, Shares::MainFirst, 5_000, 4_000), (0, 0));
		assert_eq!(budgets(2_500, Shares::Proportional, 5_000, 4_000), (0, 0));

		// No tier at all.
		assert_eq!(shares(2_500, Shares::Proportional, 40, 0), (0, 0));
		assert_eq!(budgets(2_500, Shares::Proportional, 40, 0), (0, 0));
		assert_eq!(budgets(2_500, Shares::MainFirst, 40, 0), (0, 0));
	}

	/// A carve-out that exactly fills the tier is the clamp's equality case: it
	/// warns, and the main lane's budget is none.
	#[test]
	fn the_warning_is_given_once_per_crossing() {
		let mut filled = false;

		assert!(!newly_fills(600, 1_000, &mut filled), "a queue that fits the tier must not warn");
		assert!(newly_fills(600, 600, &mut filled), "a 600 B queue on a 600 B tier covers it and must warn");
		assert!(!newly_fills(600, 600, &mut filled), "once per crossing, not once per check");
		assert!(!newly_fills(600, 1_000, &mut filled) && !filled, "it fits again");
		assert!(newly_fills(600, 400, &mut filled), "and warns when it covers the tier anew");
		assert!(!newly_fills(600, 0, &mut filled) && !filled, "a tier of nothing is not a crossing");

		for how in HOW {
			assert_eq!(budgets(600, how, 0, 600).1, 0, "a carve-out the size of the tier leaves main none");
		}
	}
}
