/*
 * Copyright (c) Kia Shakiba
 *
 * This source code is licensed under the GNU AGPLv3 license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! Shared tier-size unit type.
//!
//! Used by `lru_hybrid_cache`/`lfu_hybrid_cache`/`two_q_hybrid_cache`/
//! `fifo_hybrid_cache` (fast-tier capacity, including the runtime-adjustable
//! `PaperCache::set_fast_tier_size`) so that tier capacities can be specified in
//! bytes, mebibytes, or gibibytes without any of them depending on the others.

/// A size specification for a cache tier, in bytes, mebibytes, or gibibytes.
///
/// Units are **binary**: `Mib(64)` is `64 * 2^20`, not `64 * 10^6`. This
/// matches how every other byte budget in the cache is reasoned about --
/// `max_size`, the fast-tier capacity, and the policy queue budgets derived
/// from them are all powers of two in practice, and mixing a decimal tier size
/// into that arithmetic silently produced tier splits ~7% off the intended
/// fraction.
///
/// # Examples
///
/// ```ignore
/// use paper_cache::CacheTierSize;
///
/// let size = CacheTierSize::Mib(64);
/// assert_eq!(size.to_bytes(), 67_108_864);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CacheTierSize {
	/// Exact capacity in bytes.
	Bytes(u64),
	/// Capacity in binary mebibytes (1 MiB = 1,048,576 bytes).
	Mib(u64),
	/// Capacity in binary gibibytes (1 GiB = 1,073,741,824 bytes).
	Gib(u64),
}

impl CacheTierSize {
	/// Returns the size converted to bytes.
	pub fn to_bytes(self) -> u64 {
		match self {
			CacheTierSize::Bytes(b) => b,
			CacheTierSize::Mib(mib) => mib * 1_048_576,
			CacheTierSize::Gib(gib) => gib * 1_073_741_824,
		}
	}
}

#[cfg(test)]
mod tests {
	use super::*;

	/// The whole point of the type is this conversion, and the failure mode
	/// it exists to prevent is a decimal constant surviving the switch to
	/// binary units -- which would still compile and still look right.
	#[test]
	fn units_are_binary_not_decimal() {
		assert_eq!(CacheTierSize::Bytes(1).to_bytes(), 1);

		assert_eq!(CacheTierSize::Mib(1).to_bytes(), 1_048_576);
		assert_eq!(CacheTierSize::Gib(1).to_bytes(), 1_073_741_824);

		// Explicitly NOT the SI values these replaced.
		assert_ne!(CacheTierSize::Mib(1).to_bytes(), 1_000_000);
		assert_ne!(CacheTierSize::Gib(1).to_bytes(), 1_000_000_000);
	}

	/// 1 GiB is exactly 1024 MiB. A unit pair that disagrees on this is a
	/// typo in one of the two constants.
	#[test]
	fn the_units_agree_with_each_other() {
		assert_eq!(
			CacheTierSize::Gib(1).to_bytes(),
			CacheTierSize::Mib(1_024).to_bytes(),
		);

		assert_eq!(CacheTierSize::Gib(4).to_bytes(), CacheTierSize::Mib(4_096).to_bytes());
	}
}
