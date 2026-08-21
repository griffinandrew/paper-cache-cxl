/*
 * Copyright (c) Kia Shakiba
 *
 * This source code is licensed under the GNU AGPLv3 license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! `TieredBuffer` — the value type stored in all four hybrid-cache
//! features' (`lru_hybrid_cache`, `lfu_hybrid_cache`, `two_q_hybrid_cache`,
//! `fifo_hybrid_cache`) single, unified object table.
//!
//! Unlike a design that gets two tiers by running two independent
//! `PaperCache` instances (one `BufferDRAM`, one `BufferPMEM`) side by side,
//! the hybrid-cache features built around `TieredBuffer` store every object
//! in one `PaperCache<K, TieredBuffer>`. `TieredBuffer` is a tagged union
//! recording *where this particular object's bytes currently live*.
//! Promotion and demotion replace an `Object`'s `TieredBuffer` in place (via
//! `Object::set_data`) rather than copying bytes into a second map, so a
//! live object's bytes exist in exactly one tier's allocation at any given
//! time.
//!
//! Shared (rather than duplicated per feature) because each of the four
//! hybrid-cache features defines its own inherent-method `impl<K, S>
//! PaperCache<K, TieredBuffer, S>` block — two such blocks for the same
//! concrete type cannot coexist, so the four features are mutually
//! exclusive (see the `compile_error!` guards in `lib.rs`) and share this
//! one buffer type rather than each defining their own.
//!
//! ## Fast tier: ordinary heap allocation; slow tier: node-1 arenas
//!
//! `Fast` is a plain `Box<[u8]>` -- an ordinary heap allocation through this
//! crate's `#[global_allocator]` (`numa_alloc::FastAlloc`, node-0-bound
//! jemalloc arenas). `Fast` needing nothing beyond `Box::from` follows from
//! that: "ordinary heap allocation" and "the fast tier" are the same thing
//! once the global allocator is installed.
//!
//! The slow tier is `Box<[u8], Hybrid>` -- node-1-bound jemalloc arenas
//! (`numa_alloc::SlowObjects`). Earlier revisions selected between a jemalloc
//! pool and a custom-extent-hooks arena here; both are gone.
//!
//! `new_fast`/`new_slow`/`is_fast`/`is_slow`/`AsRef<[u8]>`/`TypeSize`/
//! `Clone` keep the same signatures regardless of which slow-tier backend
//! is selected, so no other file in this crate needs to change either way.

use typesize::TypeSize;

use crate::Hybrid;

/// A value buffer that is physically stored in exactly one tier at a time.
pub enum TieredBuffer {
	/// Fast tier: an ordinary DRAM allocation through this crate's active
	/// global allocator.
	Fast(Box<[u8]>),

	/// Slow tier: PMEM/CXL allocation. Backend selected at compile time --
	/// see the module doc comment above.
	Slow(Box<[u8], Hybrid>),
}

impl TieredBuffer {
	/// Creates a new fast-tier (DRAM) buffer by copying the given bytes.
	pub fn new_fast(bytes: &[u8]) -> Self {
		TieredBuffer::Fast(Box::from(bytes))
	}

	/// Creates a new slow-tier (PMEM/CXL) buffer by copying the given bytes.
	pub fn new_slow(bytes: &[u8]) -> Self {
		TieredBuffer::Slow(Box::clone_from_ref_in(bytes, Hybrid))
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

impl Clone for TieredBuffer {
	fn clone(&self) -> Self {
		match self {
			TieredBuffer::Fast(buffer) => TieredBuffer::Fast(buffer.clone()),
			TieredBuffer::Slow(buffer) => TieredBuffer::Slow(buffer.clone()),
		}
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

	#[test]
	fn clone_preserves_fast_tier_and_bytes() {
		let fast = TieredBuffer::new_fast(b"xyz");
		let cloned = fast.clone();

		assert!(cloned.is_fast());
		assert_eq!(cloned.as_ref(), b"xyz");
	}
}
