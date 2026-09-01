/*
 * Copyright (c) Kia Shakiba
 *
 * This source code is licensed under the GNU AGPLv3 license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::{
	time::Instant,
	collections::BTreeSet,
};

use crate::{
	HashedKey,
	object::{ExpireTime, get_expiry_from_ttl},
};

/// `true` when the expiry index is slow-tier allocated: a hybrid build that
/// has PMEM available and has not opted back out via `ttl_index_dram`.
#[cfg(all(
	feature = "hybrid_cache_common",
	feature = "key_value_pmem",
	not(feature = "ttl_index_dram"),
))]
type ExpirySet = BTreeSet<(u32, HashedKey), crate::Hybrid>;

#[cfg(not(all(
	feature = "hybrid_cache_common",
	feature = "key_value_pmem",
	not(feature = "ttl_index_dram"),
)))]
type ExpirySet = BTreeSet<(u32, HashedKey)>;

#[cfg(all(
	feature = "hybrid_cache_common",
	feature = "key_value_pmem",
	not(feature = "ttl_index_dram"),
))]
fn new_expiry_set() -> ExpirySet {
	BTreeSet::new_in(crate::Hybrid)
}

#[cfg(not(all(
	feature = "hybrid_cache_common",
	feature = "key_value_pmem",
	not(feature = "ttl_index_dram"),
)))]
fn new_expiry_set() -> ExpirySet {
	BTreeSet::new()
}

impl Default for Expiries {
	fn default() -> Self {
		Expiries {
			set: new_expiry_set(),
		}
	}
}

pub struct Expiries {
	/// Pending deadlines, ordered by expiry and then by key.
	///
	/// The key is part of the *sort key* rather than a payload so that two
	/// objects sharing a deadline occupy two entries. Keyed by the `Instant`
	/// alone -- as this was until the accompanying test was written -- the
	/// second insert overwrote the first, and the displaced object was never
	/// returned by `pop_expired`. It then outlived its own deadline: invisible
	/// to reads, which filter on `Object::is_expired`, but still holding its
	/// object-map slot, its `base_size` in `used_size` and its eviction-stack
	/// entry until capacity eviction happened to pick it.
	///
	/// This costs nothing. A `BTreeMap<Instant, HashedKey>` node slot already
	/// stored a 16-byte key beside an 8-byte value; the tuple is the same 24
	/// bytes, just arranged so the ordering is total.
	/// Slow-tier allocated for the hybrid designs (see `ExpirySet`). `TtlWorker`
	/// is its only user and never runs on a client get/set path, so the extra
	/// access latency is off the critical path; `ttl_index_dram` forces DRAM.
	set: ExpirySet,
}

impl Expiries {
	pub fn has_within(&self, ttl: u32) -> bool {
		let Some((nearest_expiry, _)) = self.set.first() else {
			return false;
		};

		*nearest_expiry <= get_expiry_from_ttl(ttl).get()
	}

	pub fn insert(&mut self, key: HashedKey, expiry: ExpireTime) {
		let Some(expiry) = expiry else {
			return;
		};

		self.set.insert((expiry.get(), key));
	}

	/// Removes this key's deadline, if it is still pending.
	///
	/// Exact by construction: the pair identifies one entry, so this can no
	/// longer touch another key's. The previous implementation needed a guard
	/// against removing a different key that happened to occupy the same
	/// instant -- with a total ordering there is nothing to guard against.
	pub fn remove(&mut self, key: HashedKey, expiry: ExpireTime) {
		let Some(expiry) = expiry else {
			return;
		};

		self.set.remove(&(expiry.get(), key));
	}

	pub fn pop_expired(&mut self, now: u32) -> Option<HashedKey> {
		let (first_expiry, _) = self.set.first()?;

		if *first_expiry > now {
			return None;
		}

		self.set.pop_first().map(|(_, key)| key)
	}

	pub fn clear(&mut self) {
		self.set.clear();
	}
}

#[cfg(test)]
mod tests {
	use std::num::NonZeroU32;

	use super::Expiries;
	use crate::object::now_ticks;

	/// Two live objects that happen to share an expiry `Instant` must both be
	/// reaped.
	///
	/// The index is keyed by the instant alone, so it holds at most one key per
	/// distinct deadline: the second `insert` overwrites the first and the
	/// displaced `HashedKey` is discarded. That object is then never returned by
	/// `pop_expired`, so the TTL worker never erases it. It outlives its own
	/// deadline as a zombie -- invisible to reads, which filter on
	/// `Object::is_expired`, but still holding its object-map slot, its
	/// `base_size` in `used_size` and its eviction-stack entry, until capacity
	/// eviction happens to pick it.
	///
	/// This used to be nearly unreachable: expiry was an `Instant`, so two
	/// objects had to be set in the same NANOSECOND with the same integer TTL.
	/// Expiry is now a tick of one SECOND, so sharing a deadline is the common
	/// case rather than a race -- any two objects set in the same second with
	/// the same TTL collide. The index keys on `(tick, key)` precisely so the
	/// key disambiguates them; a `BTreeMap<tick, HashedKey>` would now lose
	/// keys constantly rather than rarely.
	#[test]
	fn both_keys_sharing_an_expiry_instant_are_reaped() {
		let mut expiries = Expiries::default();
		let deadline = NonZeroU32::new(now_ticks() + 60).unwrap();

		expiries.insert(1, Some(deadline));
		expiries.insert(2, Some(deadline));

		let after_deadline = deadline.get() + 1;

		let mut reaped = Vec::new();

		while let Some(key) = expiries.pop_expired(after_deadline) {
			reaped.push(key);
		}

		reaped.sort_unstable();

		assert_eq!(
			reaped,
			vec![1, 2],
			"both keys share a deadline and must both be reaped; the index \
			 dropped one, leaving it to outlive its TTL",
		);
	}

	/// Distinct deadlines are unaffected -- the guard against a regression that
	/// "fixes" the above by breaking ordering.
	#[test]
	fn distinct_deadlines_are_reaped_in_chronological_order() {
		let mut expiries = Expiries::default();
		let base = now_ticks() + 60;

		expiries.insert(30, NonZeroU32::new(base + 30));
		expiries.insert(10, NonZeroU32::new(base + 10));
		expiries.insert(20, NonZeroU32::new(base + 20));

		let after_all = base + 120;

		let mut reaped = Vec::new();

		while let Some(key) = expiries.pop_expired(after_all) {
			reaped.push(key);
		}

		assert_eq!(reaped, vec![10, 20, 30], "reaping must stay in deadline order");
	}

	/// Guards the *placement*, not the cfg logic: it reports what this build's
	/// feature set actually resolved to. If a Cargo.toml change ever stopped
	/// `hybrid_cache_common` implying `key_value_pmem`, the index would quietly
	/// fall back to DRAM and every hybrid run would carry ~24 B/TTL'd object of
	/// unaccounted DRAM again. This fires instead.
	#[test]
	fn hybrid_builds_allocate_the_expiry_index_in_the_slow_tier() {
		let slow_tier = cfg!(all(
			feature = "hybrid_cache_common",
			feature = "key_value_pmem",
			not(feature = "ttl_index_dram"),
		));

		if cfg!(feature = "hybrid_cache_common") && !cfg!(feature = "ttl_index_dram") {
			assert!(
				slow_tier,
				"a hybrid build must allocate the expiry index in the slow tier; \
				 it fell back to DRAM, so `key_value_pmem` is no longer implied",
			);
		}

		if cfg!(feature = "ttl_index_dram") {
			assert!(!slow_tier, "`ttl_index_dram` must force the index back to DRAM");
		}
	}
}
