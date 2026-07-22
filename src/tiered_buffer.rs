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
//! ## Fast tier: ordinary heap allocation; slow tier: one of two backends
//!
//! `Fast` is a plain `Box<[u8]>` — an ordinary heap allocation through
//! whatever this crate's `#[global_allocator]` currently is. `Fast` needing
//! nothing beyond `Box::from` is a direct consequence of that: "ordinary
//! heap allocation" and "the fast tier" are the same thing once the global
//! allocator is installed -- which global allocator that is depends on the
//! same feature switch that picks `Slow`'s backend, below.
//!
//! `Slow` has two selectable backends, chosen at compile time, and each one
//! also determines what `Fast`'s global allocator is (see `lib.rs`'s
//! `#[global_allocator]` cfg split):
//!
//! - **Default**: `tier_allocator::TierBuffer`, via `tier_allocator::alloc_on`
//!   (UMF + Intel TBB's scalable pool). `Fast`'s global allocator is
//!   `tier_allocator::NumaAllocator` bound to node 0 -- the same shared
//!   per-node registry `Slow` draws from for node 1, so both tiers share one
//!   real UMF pool per node. See `tier_allocator`'s own crate-level doc
//!   comment for the full reasoning.
//! - **`jemalloc_cxl_slow_tier` feature**: `Box<[u8], allocator::
//!   SlowTierJemallocAllocator>` — jemalloc_cxl's custom-extent-hooks
//!   NUMA/CXL arena mechanism. This feature also switches `Fast`'s global
//!   allocator to plain jemalloc (via jemalloc_cxl's own `#[global_allocator]`
//!   declaration, enabled through its `own_global_allocator` feature -- see
//!   `lib.rs`) -- so **both** tiers go through jemalloc in this
//!   configuration, and `tier_allocator`/UMF is never actually called at
//!   all, deliberately, not just for the slow tier. Measured (single-threaded
//!   and under real concurrent stress, both against a real UMF/tier_allocator
//!   comparison) to be slower per-call single-threaded but faster in
//!   aggregate under concurrent multi-thread load, and to place data
//!   correctly on the target NUMA node under that same concurrent load
//!   (verified via `/proc/self/numa_maps`, not merely assumed). See
//!   `jemalloc_cxl/README.md` for the underlying mechanism.
//!
//! `new_fast`/`new_slow`/`is_fast`/`is_slow`/`AsRef<[u8]>`/`TypeSize`/
//! `Clone` keep the same signatures regardless of which slow-tier backend
//! is selected, so no other file in this crate needs to change either way.

use typesize::TypeSize;

#[cfg(not(feature = "jemalloc_cxl_slow_tier"))]
use tier_allocator::TierBuffer;

#[cfg(feature = "jemalloc_cxl_slow_tier")]
use crate::allocator::SlowTierJemallocAllocator;

/// NUMA node backing the slow (PMEM/CXL) tier, matching this crate's own
/// `HybridObjects::NODE` convention (`src/allocator.rs`). Only meaningful
/// for the default (`tier_allocator`) backend; the `jemalloc_cxl_slow_tier`
/// backend has its own copy of this same node id in
/// `allocator::slow_tier_jemalloc_allocator`, since that module must
/// compile standalone.
#[cfg(not(feature = "jemalloc_cxl_slow_tier"))]
const SLOW_TIER_NODE: i32 = 1;

/// A value buffer that is physically stored in exactly one tier at a time.
pub enum TieredBuffer {
	/// Fast tier: an ordinary DRAM allocation through this crate's active
	/// global allocator (`tier_allocator::NumaAllocator`, node 0, for the
	/// four hybrid-cache features).
	Fast(Box<[u8]>),

	/// Slow tier: PMEM/CXL allocation. Backend selected at compile time --
	/// see the module doc comment above.
	#[cfg(not(feature = "jemalloc_cxl_slow_tier"))]
	Slow(TierBuffer),
	#[cfg(feature = "jemalloc_cxl_slow_tier")]
	Slow(Box<[u8], SlowTierJemallocAllocator>),
}

impl TieredBuffer {
	/// Creates a new fast-tier (DRAM) buffer by copying the given bytes.
	pub fn new_fast(bytes: &[u8]) -> Self {
		TieredBuffer::Fast(Box::from(bytes))
	}

	/// Creates a new slow-tier (PMEM/CXL) buffer by copying the given bytes.
	#[cfg(not(feature = "jemalloc_cxl_slow_tier"))]
	pub fn new_slow(bytes: &[u8]) -> Self {
		let mut buffer = tier_allocator::alloc_on(SLOW_TIER_NODE, bytes.len())
			.expect("slow-tier allocation should succeed");

		buffer.copy_from_slice(bytes);
		TieredBuffer::Slow(buffer)
	}

	/// Creates a new slow-tier (PMEM/CXL) buffer by copying the given bytes,
	/// via jemalloc_cxl's custom-extent-hooks arena instead of
	/// `tier_allocator`.
	#[cfg(feature = "jemalloc_cxl_slow_tier")]
	pub fn new_slow(bytes: &[u8]) -> Self {
		let mut buffer = Box::<[u8], _>::new_uninit_slice_in(bytes.len(), SlowTierJemallocAllocator);

		// SAFETY: `buffer` was just allocated with exactly `bytes.len()`
		// elements; `MaybeUninit<u8>` has the same layout as `u8`, so
		// writing `bytes.len()` bytes into it via a raw copy and then
		// calling `assume_init()` is exactly equivalent to initializing
		// each element individually.
		unsafe {
			std::ptr::copy_nonoverlapping(bytes.as_ptr(), buffer.as_mut_ptr().cast::<u8>(), bytes.len());
			TieredBuffer::Slow(buffer.assume_init())
		}
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
	#[cfg(not(feature = "jemalloc_cxl_slow_tier"))]
	fn clone(&self) -> Self {
		match self {
			TieredBuffer::Fast(buffer) => TieredBuffer::Fast(buffer.clone()),

			TieredBuffer::Slow(buffer) => {
				let allocator = tier_allocator::allocator_for(SLOW_TIER_NODE)
					.expect("slow-tier allocator should be available");

				TieredBuffer::Slow(
					buffer.duplicate(allocator).expect("slow-tier duplicate should succeed"),
				)
			}
		}
	}

	#[cfg(feature = "jemalloc_cxl_slow_tier")]
	fn clone(&self) -> Self {
		match self {
			TieredBuffer::Fast(buffer) => TieredBuffer::Fast(buffer.clone()),

			TieredBuffer::Slow(buffer) => {
				let mut new_buffer =
					Box::<[u8], _>::new_uninit_slice_in(buffer.len(), SlowTierJemallocAllocator);

				// SAFETY: same reasoning as `new_slow` above -- `new_buffer`
				// has exactly `buffer.len()` elements, freshly allocated.
				unsafe {
					std::ptr::copy_nonoverlapping(
						buffer.as_ptr(),
						new_buffer.as_mut_ptr().cast::<u8>(),
						buffer.len(),
					);
					TieredBuffer::Slow(new_buffer.assume_init())
				}
			}
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
