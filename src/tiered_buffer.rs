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
//! Unlike `hybridcache`, which gets two tiers by running two independent
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
//! ## Backed by `tier_allocator`, not the crate's own `Hybrid` allocator
//!
//! Both variants are backed by the standalone `tier_allocator` crate's
//! `TierBuffer` (see `umf_tier_allocator/`) rather than `Box<[u8]>`/
//! `Box<[u8], Hybrid>` as in earlier versions of this file. This is a
//! deliberate architectural swap, not a behavioral change: `tier_allocator`
//! is a runtime-parameterized NUMA-tier allocator (construct a
//! `TierAllocator` bound to any NUMA node id at runtime, rather than
//! picking a hardcoded allocator type at compile time via Cargo features
//! the way `Hybrid`/`HybridObjects`/`DRAMObjects` do). It builds on stable
//! Rust (no `std::alloc::Allocator`/`#![feature(allocator_api)]` involved),
//! and its `TierAllocator` deliberately has no teardown — pools live for
//! the process, matching this crate's own `HybridObjects`/`DRAMObjects`
//! precedent — so a `TierBuffer` allocated from one of the two static
//! allocators below stays valid regardless of anything else in the process.
//! See `tier_allocator`'s own crate-level doc comment for the full
//! reasoning (including why `std::alloc::Allocator` was rejected in favor
//! of this hand-rolled buffer type).
//!
//! `new_fast`/`new_slow`/`is_fast`/`is_slow`/`AsRef<[u8]>`/`TypeSize`/
//! `Clone` are unchanged from before this swap, so no other file in this
//! crate needed to change to adopt it.

use std::sync::OnceLock;

use typesize::TypeSize;

use tier_allocator::{TierAllocator, TierBuffer};

/// NUMA node backing the fast (DRAM) tier, matching this crate's own
/// `DRAMObjects::NODE_DRAM` convention (`src/allocator.rs`).
const FAST_TIER_NODE: i32 = 0;

/// NUMA node backing the slow (PMEM/CXL) tier, matching this crate's own
/// `HybridObjects::NODE` convention (`src/allocator.rs`).
const SLOW_TIER_NODE: i32 = 1;

/// The fast-tier `TierAllocator`, constructed once and kept alive for the
/// life of the process (see `tier_allocator::TierAllocator`'s "no teardown"
/// doc comment for why that's safe).
fn fast_allocator() -> &'static TierAllocator {
	static FAST: OnceLock<TierAllocator> = OnceLock::new();
	FAST.get_or_init(|| {
		TierAllocator::new_numa(FAST_TIER_NODE)
			.expect("TierAllocator::new_numa(FAST_TIER_NODE) should succeed")
	})
}

/// The slow-tier `TierAllocator`, constructed once and kept alive for the
/// life of the process.
fn slow_allocator() -> &'static TierAllocator {
	static SLOW: OnceLock<TierAllocator> = OnceLock::new();
	SLOW.get_or_init(|| {
		TierAllocator::new_numa(SLOW_TIER_NODE)
			.expect("TierAllocator::new_numa(SLOW_TIER_NODE) should succeed")
	})
}

/// A value buffer that is physically stored in exactly one tier at a time.
pub enum TieredBuffer {
	/// Fast tier: DRAM allocation via a dedicated `tier_allocator::TierAllocator`.
	Fast(TierBuffer),
	/// Slow tier: PMEM/CXL allocation via a dedicated `tier_allocator::TierAllocator`.
	Slow(TierBuffer),
}

impl TieredBuffer {
	/// Creates a new fast-tier (DRAM) buffer by copying the given bytes.
	pub fn new_fast(bytes: &[u8]) -> Self {
		let mut buffer = fast_allocator()
			.alloc(bytes.len())
			.expect("fast-tier allocation should succeed");

		buffer.copy_from_slice(bytes);
		TieredBuffer::Fast(buffer)
	}

	/// Creates a new slow-tier (PMEM/CXL) buffer by copying the given bytes.
	pub fn new_slow(bytes: &[u8]) -> Self {
		let mut buffer = slow_allocator()
			.alloc(bytes.len())
			.expect("slow-tier allocation should succeed");

		buffer.copy_from_slice(bytes);
		TieredBuffer::Slow(buffer)
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
			TieredBuffer::Fast(buffer) => TieredBuffer::Fast(
				buffer.duplicate(fast_allocator()).expect("fast-tier duplicate should succeed"),
			),

			TieredBuffer::Slow(buffer) => TieredBuffer::Slow(
				buffer.duplicate(slow_allocator()).expect("slow-tier duplicate should succeed"),
			),
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
}
