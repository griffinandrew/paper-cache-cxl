/*
 * Copyright (c) Kia Shakiba
 *
 * This source code is licensed under the GNU AGPLv3 license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! `TieredBuffer` — the value type stored in `lru_hybrid_cache`'s single,
//! unified object table.
//!
//! Unlike `hybridcache`, which gets two tiers by running two independent
//! `PaperCache` instances (one `BufferDRAM`, one `BufferPMEM`) side by side,
//! `lru_hybrid_cache` stores every object in one `PaperCache<K, TieredBuffer>`.
//! `TieredBuffer` is a tagged union recording *where this particular object's
//! bytes currently live*. Promotion and demotion replace an `Object`'s
//! `TieredBuffer` in place (via `Object::set_data`) rather than copying bytes
//! into a second map, so a live object's bytes exist in exactly one tier's
//! allocation at any given time.

use typesize::TypeSize;

use crate::Hybrid;

/// A value buffer that is physically stored in exactly one tier at a time.
#[derive(Clone)]
pub enum TieredBuffer {
	/// Fast tier: plain DRAM allocation (goes through the crate's global allocator).
	Fast(Box<[u8]>),
	/// Slow tier: PMEM allocation via the `Hybrid` allocator.
	Slow(Box<[u8], Hybrid>),
}

impl TieredBuffer {
	/// Creates a new fast-tier (DRAM) buffer by copying the given bytes.
	pub fn new_fast(bytes: &[u8]) -> Self {
		TieredBuffer::Fast(bytes.to_vec().into_boxed_slice())
	}

	/// Creates a new slow-tier (PMEM) buffer by copying the given bytes into
	/// a `Hybrid`-allocated buffer.
	pub fn new_slow(bytes: &[u8]) -> Self {
		TieredBuffer::Slow(bytes.to_vec_in(Hybrid).into_boxed_slice())
	}

	/// Returns `true` if this buffer currently lives in the fast (DRAM) tier.
	pub fn is_fast(&self) -> bool {
		matches!(self, TieredBuffer::Fast(_))
	}

	/// Returns `true` if this buffer currently lives in the slow (PMEM) tier.
	pub fn is_slow(&self) -> bool {
		matches!(self, TieredBuffer::Slow(_))
	}
}

impl AsRef<[u8]> for TieredBuffer {
	fn as_ref(&self) -> &[u8] {
		match self {
			TieredBuffer::Fast(bytes) => bytes.as_ref(),
			TieredBuffer::Slow(bytes) => bytes.as_ref(),
		}
	}
}

impl TypeSize for TieredBuffer {
	fn get_size(&self) -> usize {
		self.as_ref().len()
	}
}

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn fast_and_slow_round_trip_bytes() {
		let fast = TieredBuffer::new_fast(b"hello");
		assert!(fast.is_fast());
		assert!(!fast.is_slow());
		assert_eq!(fast.as_ref(), b"hello");
		assert_eq!(fast.get_size(), 5);

		let slow = TieredBuffer::new_slow(b"world!");
		assert!(slow.is_slow());
		assert!(!slow.is_fast());
		assert_eq!(slow.as_ref(), b"world!");
		assert_eq!(slow.get_size(), 6);
	}

	#[test]
	fn clone_preserves_tier_and_bytes() {
		let slow = TieredBuffer::new_slow(b"abc");
		let cloned = slow.clone();

		assert!(cloned.is_slow());
		assert_eq!(cloned.as_ref(), b"abc");
	}
}
