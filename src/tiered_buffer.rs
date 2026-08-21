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
//! ## Fast tier: ordinary heap allocation; slow tier: one of two backends
//!
//! `Fast` is a plain `Box<[u8]>` — an ordinary heap allocation through
//! whatever this crate's `#[global_allocator]` currently is (`DRAMObjects`,
//! `src/allocator.rs`, NUMA node 0). `Fast` needing nothing beyond `Box::from`
//! is a direct consequence of that: "ordinary heap allocation" and "the fast
//! tier" are the same thing once the global allocator is installed.
//!
//! `Slow` has two selectable backends, chosen at compile time:
//!
//! - **Default**: `Box<[u8], Hybrid>` — the crate-wide PMEM allocator alias
//!   (`Hybrid` = `HybridObjects`, `src/allocator.rs`), the same UMF +
//!   Intel TBB scalable-pool allocator every other PMEM feature
//!   (`key_value_pmem`, `global_hashtable_pmem`, ...) already uses. A prior
//!   session tried replacing this with a standalone `tier_allocator` crate
//!   (a from-scratch, runtime-parameterized reimplementation of "one UMF/TBB
//!   pool per NUMA node") on the theory that sharing one registry between the
//!   fast and slow tier's pools would be an improvement over two separate,
//!   independent allocator types. It wasn't: `tier_allocator`'s own registry
//!   still constructed one pool per node (`DRAMObjects`-equivalent for node 0,
//!   `HybridObjects`-equivalent for node 1) -- structurally identical to what
//!   `DRAMObjects`+`HybridObjects` already provided, just reimplemented in a
//!   second crate. Removed entirely; `Slow` reverted to plain `Box<[u8],
//!   Hybrid>`.
//! - **`jemalloc_cxl_slow_tier` feature**: `Box<[u8], allocator::
//!   SlowTierJemallocAllocator>` — jemalloc_cxl's custom-extent-hooks
//!   NUMA/CXL arena mechanism. This feature also switches `Fast`'s global
//!   allocator to plain jemalloc (via jemalloc_cxl's own `#[global_allocator]`
//!   declaration, enabled through its `own_global_allocator` feature -- see
//!   `lib.rs`) -- so **both** tiers go through jemalloc in this
//!   configuration, and `Hybrid`/UMF is never actually called at all,
//!   deliberately, not just for the slow tier. Measured (single-threaded and
//!   under real concurrent stress) to be slower per-call single-threaded but
//!   faster in aggregate under concurrent multi-thread load, and to place
//!   data correctly on the target NUMA node under that same concurrent load
//!   (verified via `/proc/self/numa_maps`, not merely assumed). See
//!   `jemalloc_cxl/README.md` for the underlying mechanism.
//!
//! `new_fast`/`new_slow`/`is_fast`/`is_slow`/`AsRef<[u8]>`/`TypeSize`/
//! `Clone` keep the same signatures regardless of which slow-tier backend
//! is selected, so no other file in this crate needs to change either way.

use typesize::TypeSize;

use crate::Hybrid;

#[cfg(feature = "jemalloc_cxl_slow_tier")]
use crate::allocator::SlowTierJemallocAllocator;

/// A value buffer that is physically stored in exactly one tier at a time.
pub enum TieredBuffer {
	/// Fast tier: an ordinary DRAM allocation through this crate's active
	/// global allocator.
	Fast(Box<[u8]>),

	/// Slow tier: PMEM/CXL allocation. Backend selected at compile time --
	/// see the module doc comment above.
	#[cfg(all(not(feature = "jemalloc_cxl_slow_tier"), not(feature = "numa_jemalloc")))]
	Slow(Box<[u8], Hybrid>),
	/// Slow tier on node-1-bound jemalloc arenas, replacing the UMF/TBB pool.
	#[cfg(all(not(feature = "jemalloc_cxl_slow_tier"), feature = "numa_jemalloc"))]
	Slow(Box<[u8], crate::numa_alloc::SlowAlloc>),
	#[cfg(feature = "jemalloc_cxl_slow_tier")]
	Slow(Box<[u8], SlowTierJemallocAllocator>),
}

impl TieredBuffer {
	/// Creates a new fast-tier (DRAM) buffer by copying the given bytes.
	pub fn new_fast(bytes: &[u8]) -> Self {
		TieredBuffer::Fast(Box::from(bytes))
	}

	/// Creates a new slow-tier (PMEM/CXL) buffer by copying the given bytes.
	#[cfg(all(not(feature = "jemalloc_cxl_slow_tier"), not(feature = "numa_jemalloc")))]
	pub fn new_slow(bytes: &[u8]) -> Self {
		TieredBuffer::Slow(Box::clone_from_ref_in(bytes, Hybrid))
	}

	/// Slow-tier copy onto node-1-bound jemalloc arenas.
	#[cfg(all(not(feature = "jemalloc_cxl_slow_tier"), feature = "numa_jemalloc"))]
	pub fn new_slow(bytes: &[u8]) -> Self {
		TieredBuffer::Slow(Box::clone_from_ref_in(bytes, crate::numa_alloc::NumaAlloc))
	}

	/// Creates a new slow-tier (PMEM/CXL) buffer by copying the given bytes,
	/// via jemalloc_cxl's custom-extent-hooks arena instead of `Hybrid`.
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
			TieredBuffer::Slow(buffer) => TieredBuffer::Slow(buffer.clone()),
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
