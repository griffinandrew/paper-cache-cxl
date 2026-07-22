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
//! bytes, megabytes, or gigabytes without any of them depending on the others.

/// A size specification for a cache tier, in bytes, megabytes, or gigabytes.
///
/// # Examples
///
/// ```ignore
/// use paper_cache::CacheTierSize;
///
/// let size = CacheTierSize::Mb(64);
/// assert_eq!(size.to_bytes(), 64_000_000);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CacheTierSize {
	/// Exact capacity in bytes.
	Bytes(u64),
	/// Capacity in decimal megabytes (1 MB = 1,000,000 bytes, SI standard).
	Mb(u64),
	/// Capacity in decimal gigabytes (1 GB = 1,000,000,000 bytes, SI standard).
	Gb(u64),
}

impl CacheTierSize {
	/// Returns the size converted to bytes.
	pub fn to_bytes(self) -> u64 {
		match self {
			CacheTierSize::Bytes(b) => b,
			CacheTierSize::Mb(mb) => mb * 1_000_000,
			CacheTierSize::Gb(gb) => gb * 1_000_000_000,
		}
	}
}
