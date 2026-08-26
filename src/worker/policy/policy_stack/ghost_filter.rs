/*
 * Copyright (c) Kia Shakiba
 *
 * This source code is licensed under the GNU AGPLv3 license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! S3-FIFO's ghost queue, implemented the way the paper specifies it.
//!
//! # What the paper says
//!
//! > "The ghost FIFO queue G can be implemented as part of the indexing
//! > structure." Each entry is a 4-byte fingerprint of the object plus an
//! > eviction timestamp measured in request count, and "when an entry is
//! > evicted from the ghost queue, it is not immediately removed from the hash
//! > table; instead, the hash table entry is removed during hash collision --
//! > when the slot is needed to store other entries."
//!
//! So there is no queue: membership is *fingerprint matches* **and** *the
//! eviction is within the last `window` insertions*, and reclamation happens
//! by being overwritten. G holds the same number of entries as M.
//!
//! # Why this replaced a `HashList`
//!
//! The previous ghost was a `HashList<HashedKey>` -- a doubly-linked list with
//! a hash index over it -- costing 44 bytes to hold an 8-byte key: a heap
//! `Entry { key, prev, next }` (24) plus an index slot pointing at that node
//! (20). It bought O(1) removal from the middle, which the ghost barely uses.
//!
//! Worse, its capacity bound was unreachable on the workloads that need it.
//! `trim_ghost` ran only from a genuine main-queue eviction, never from the
//! one-access eviction that *populates* the ghost -- so on a trace with no
//! reuse, where nothing is ever promoted to main, the ghost grew without
//! limit. Measured on Twitter cluster38: 1.94 GB, **45% of a 4 GiB fast
//! tier**, which is why those variants held no object data at all.
//!
//! This design cannot grow past its allocation, so that class of bug is gone
//! by construction rather than by remembering to call a trim.
//!
//! # Cost
//!
//! 8 bytes per slot (`u32` fingerprint + `u32` timestamp) against 44, and
//! bounded rather than unbounded: ~39x less on the measured trace.
//!
//! # What is given up
//!
//! Membership becomes approximate, which the algorithm tolerates by design. A
//! fingerprint collision (~1 in 2^32 per probe) reports a key as recently
//! evicted when it was not, routing it to the main queue instead of the small
//! one -- a heuristic miss, not a correctness failure. A slot collision does
//! the reverse, forgetting a real ghost early; that is exactly the paper's
//! "removed during hash collision" reclamation.

use crate::{CacheSize, HashedKey};

// The ghost table follows the eviction stacks: under `eviction_stacks_pmem` it
// is allocated through the crate-wide `Hybrid` allocator on the far
// (CXL/PMEM) node. Without this it stayed in DRAM while
// `GHOST_ENTRY_DRAM_OVERHEAD` was gated to 0 on the premise it had moved --
// physically resident in DRAM and charged nothing.
#[cfg(not(feature = "eviction_stacks_pmem"))]
type SlotVec = Vec<GhostSlot>;
#[cfg(feature = "eviction_stacks_pmem")]
type SlotVec = Vec<GhostSlot, crate::Hybrid>;

#[cfg(not(feature = "eviction_stacks_pmem"))]
fn new_slots(capacity: usize) -> SlotVec {
	vec![GhostSlot::default(); capacity]
}

#[cfg(feature = "eviction_stacks_pmem")]
fn new_slots(capacity: usize) -> SlotVec {
	let mut v = Vec::with_capacity_in(capacity, crate::Hybrid);
	v.resize(capacity, GhostSlot::default());
	v
}

/// One ghost slot: a fingerprint and the insertion counter value at which the
/// key entered the ghost. Eight bytes, no allocation, no pointers.
#[derive(Clone, Copy, Default)]
struct GhostSlot {
	/// Fingerprint of the key. `0` means empty -- `fingerprint()` never
	/// returns `0`, so there is no need for a separate occupancy bit.
	fingerprint: u32,

	/// Value of `inserted` when this key was pushed. Compared against the
	/// current counter to decide whether the entry is still within the
	/// window; wrapping arithmetic makes the `u32` rollover harmless for any
	/// window below 2^31.
	inserted_at: u32,
}

pub struct GhostFilter {
	slots: SlotVec,

	/// `slots.len() - 1`; `slots.len()` is always a power of two.
	mask: usize,

	/// Monotonic count of insertions. The FIFO bound is arithmetic on this
	/// rather than a list traversal: an entry is live iff fewer than `window`
	/// insertions have happened since it was written.
	inserted: u32,

	/// How many of the most recent insertions count as "in the ghost queue" --
	/// the paper's |G| = |M|. Updated as the main queue grows or shrinks.
	window: u32,

	/// Live entries, maintained incrementally so `len` stays O(1) on the
	/// demotion path. Capped at `window`, and decremented on an explicit
	/// `remove`; entries that simply age out of the window are covered by
	/// that cap rather than tracked individually.
	live: u32,
}

/// Non-zero 32-bit fingerprint of a key.
///
/// Takes the *high* bits: `HashedKey` is already a hash, and the low bits are
/// consumed by the slot index, so using the high bits keeps the two
/// independent. Forced non-zero so `0` can mean "empty".
#[inline]
fn fingerprint(key: HashedKey) -> u32 {
	let fp = (key >> 32) as u32;
	if fp == 0 { 1 } else { fp }
}

impl GhostFilter {
	/// A filter with room for `hint` entries, rounded up to a power of two and
	/// floored at 1024 so a small cache still gets a usable table.
	pub fn with_capacity(hint: usize) -> Self {
		let n = hint.max(1024).next_power_of_two();

		GhostFilter {
			slots: new_slots(n),
			mask: n - 1,
			inserted: 0,
			window: n as u32,
			live: 0,
		}
	}

	/// Sets the FIFO window -- the number of most-recent insertions that count
	/// as resident, i.e. |M|. Growing the table beyond its allocation would
	/// need the original keys, which are deliberately not stored, so the table
	/// itself is fixed and only the window tracks the main queue.
	#[inline]
	pub fn set_window(&mut self, entries: usize) {
		self.window = entries.min(self.slots.len()).max(1) as u32;
		self.live = self.live.min(self.window);
	}

	#[inline]
	fn index(&self, key: HashedKey) -> usize {
		(key as usize) & self.mask
	}

	/// True if this key was evicted within the last `window` insertions.
	pub fn contains(&self, key: HashedKey) -> bool {
		let slot = self.slots[self.index(key)];

		slot.fingerprint == fingerprint(key)
			&& self.inserted.wrapping_sub(slot.inserted_at) < self.window
	}

	/// Records a key as recently evicted, overwriting whatever occupied its
	/// slot. That overwrite *is* the paper's reclamation-on-collision: no
	/// separate eviction pass exists or is needed.
	pub fn insert(&mut self, key: HashedKey) {
		let index = self.index(key);
		self.inserted = self.inserted.wrapping_add(1);

		let replacing_live = self.slots[index].fingerprint != 0;

		self.slots[index] = GhostSlot {
			fingerprint: fingerprint(key),
			inserted_at: self.inserted,
		};

		// Overwriting an occupied slot swaps one entry for another; only a
		// fresh slot adds to the population.
		if !replacing_live {
			self.live = self.live.saturating_add(1).min(self.window);
		}
	}

	/// Drops this key's entry if it holds the slot. Cheap and exact for the
	/// key that is present; a key whose slot was already taken by another is
	/// already absent.
	pub fn remove(&mut self, key: HashedKey) {
		let index = self.index(key);

		if self.slots[index].fingerprint == fingerprint(key) {
			self.slots[index] = GhostSlot::default();
			self.live = self.live.saturating_sub(1);
		}
	}

	pub fn clear(&mut self) {
		self.slots.iter_mut().for_each(|slot| *slot = GhostSlot::default());
		self.inserted = 0;
		self.live = 0;
	}

	/// Live entries: the FIFO window, or everything inserted so far if that is
	/// smaller.
	///
	/// O(1) rather than a scan, because `reserved_overhead` consults it inside
	/// `settle_fast_tier` -- on the demotion path. Slot collisions can drop an
	/// entry early, so this is an upper bound, which is the right direction for
	/// a reservation.
	#[inline]
	pub fn len(&self) -> usize {
		self.live.min(self.window) as usize
	}

	/// DRAM charged for the ghost: 8 bytes per *live* entry.
	///
	/// Per live entry, not per allocated slot, because the paper places the
	/// ghost "as part of the indexing structure" -- entries occupy slots the
	/// index already owns, and dead ones are reclaimed on collision when
	/// something else needs the slot. The backing `Vec` here is an
	/// implementation shortcut for not being inside the object map; charging
	/// its whole allocation would bill DRAM the paper's design never spends.
	///
	/// Bounded either way: `len()` cannot exceed the window, so this cannot
	/// run away the way the `HashList` did.
	#[inline]
	pub fn dram_bytes(&self) -> CacheSize {
		// Charges `GHOST_ENTRY_DRAM_OVERHEAD`, not `size_of::<GhostSlot>()`. The
		// two differ under `eviction_stacks_pmem`: the constant becomes 0 because
		// the table is no longer in DRAM, while the struct is still 8 bytes wide.
		// This previously hardcoded the width, so a PMEM build charged the fast
		// tier for a table that was not in it -- the opposite of the placement
		// bug fixed above, and in the same function.
		(self.len() as CacheSize)
			* (crate::object::overhead::GHOST_ENTRY_DRAM_OVERHEAD as CacheSize)
	}
}

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn a_slot_is_eight_bytes() {
		assert_eq!(
			core::mem::size_of::<GhostSlot>(),
			8,
			"the point of this type is 8 bytes against the old HashList's 44",
		);
	}

	#[test]
	fn recently_inserted_keys_are_members() {
		let mut g = GhostFilter::with_capacity(1024);
		g.set_window(1024);

		for key in 0..100u64 {
			g.insert(key);
		}

		for key in 0..100u64 {
			assert!(g.contains(key), "key {key} was just inserted");
		}
	}

	#[test]
	fn entries_fall_out_of_the_window_without_any_trim() {
		let mut g = GhostFilter::with_capacity(4096);
		g.set_window(16);

		g.insert(1);
		assert!(g.contains(1));

		// 16 further insertions push key 1 out of the window. Nothing is
		// trimmed, popped or freed -- the bound is arithmetic.
		for key in 100..116u64 {
			g.insert(key);
		}

		assert!(!g.contains(1), "key 1 left the window on its own");
		assert!(g.contains(115), "the newest key is still inside it");
	}

	#[test]
	fn memory_is_fixed_regardless_of_how_much_is_inserted() {
		let mut g = GhostFilter::with_capacity(1024);
		let before = g.dram_bytes();

		for key in 0..1_000_000u64 {
			g.insert(key);
		}

		assert!(
			g.dram_bytes() <= (g.window as usize * 8) as CacheSize,
			"a million insertions must stay inside the window -- this is the bug \
			 the old HashList had, where a ghost reached 1.94 GB on cluster38",
		);
		assert_eq!(before, 0, "an empty ghost costs nothing");
	}

	#[test]
	fn remove_drops_the_key() {
		let mut g = GhostFilter::with_capacity(1024);
		g.set_window(1024);

		g.insert(7);
		assert!(g.contains(7));

		g.remove(7);
		assert!(!g.contains(7));
	}

	#[test]
	fn clear_empties_it() {
		let mut g = GhostFilter::with_capacity(1024);
		g.set_window(1024);
		(0..50u64).for_each(|key| g.insert(key));

		g.clear();

		assert_eq!(g.len(), 0);
		(0..50u64).for_each(|key| assert!(!g.contains(key)));
	}

	/// The counter is `u32`; a long trace overflows it. Wrapping subtraction
	/// keeps the window comparison correct across the rollover.
	#[test]
	fn the_window_survives_counter_wraparound() {
		let mut g = GhostFilter::with_capacity(1024);
		g.set_window(8);
		g.inserted = u32::MAX - 2;

		g.insert(42);
		assert!(g.contains(42));

		for key in 200..205u64 {
			g.insert(key);
		}

		// wrapped past zero, and 42 is still within 8 insertions
		assert!(g.inserted < 16, "counter should have wrapped");
		assert!(g.contains(42), "wrapping must not evict a live entry");
	}
}
