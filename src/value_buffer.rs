/*
 * Copyright (c) Kia Shakiba
 *
 * This source code is licensed under the GNU AGPLv3 license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! Abstracts over the two value-buffer types (`BufferDRAM` = `Box<[u8]>`,
//! `BufferPMEM` = `Box<[u8], Hybrid>`) so `PaperCache`'s generic impl
//! blocks can build a fresh value from `set()`'s `&[u8]` argument without
//! knowing which allocator backs it.

use typesize::TypeSize;

use crate::BufferDRAM;

#[cfg(feature = "key_value_pmem")]
use crate::{BufferPMEM, Hybrid};

/// A value type a [`crate::PaperCache`] can store: a byte buffer backed by
/// some allocator (plain heap for `BufferDRAM`, the crate's `Hybrid`
/// PMEM/CXL allocator for `BufferPMEM`).
pub trait ValueBuffer: TypeSize + AsRef<[u8]> + Clone + Send + Sync + 'static {
	/// Builds a new value by copying `bytes`, allocated through whichever
	/// backend this buffer type uses.
	fn from_bytes(bytes: &[u8]) -> Self;
}

#[cfg(not(feature = "segregated_value_arena"))]
impl ValueBuffer for BufferDRAM {
	fn from_bytes(bytes: &[u8]) -> Self {
		Box::clone_from_ref(bytes)
	}
}

/// Builds the value in the segregated DRAM pool rather than the process-wide
/// default one, so this long-lived allocation never shares bins or a tcache
/// with the transient `to_vec()` destination `get()` hands back.
#[cfg(feature = "segregated_value_arena")]
impl ValueBuffer for BufferDRAM {
	fn from_bytes(bytes: &[u8]) -> Self {
		Box::clone_from_ref_in(bytes, crate::numa_alloc::FastValues)
	}
}

#[cfg(feature = "key_value_pmem")]
impl ValueBuffer for BufferPMEM {
	fn from_bytes(bytes: &[u8]) -> Self {
		Box::clone_from_ref_in(bytes, Hybrid)
	}
}
