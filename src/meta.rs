/*
 * Copyright (c) Kia Shakiba
 *
 * This source code is licensed under the GNU AGPLv3 license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! M: the bytes the cache's OWN DRAM metadata structures hold right now,
//! counted from the structures rather than modelled per object -- step S5a of
//! the fast-tier backpressure plan. Reporting only: nothing decides anything
//! on M yet (S5's gate will budget `P + M <= F`).
//!
//! # The unit
//!
//! Every figure is an allocation's USABLE size: jemalloc's size class for the
//! request, `nallocx` with the alignment flag this crate's allocators pass
//! (`numa_alloc::flags_for`: `MALLOCX_LG_ALIGN` whenever `align > 1`). That is
//! what `thread.allocated`, `numa_alloc::measured` and Redis's `used_memory`
//! count, and the unit the value charges (`resident_object_bytes`) are in.
//! Like Redis's `used_memory` it counts allocations, not residency, but only
//! the cache's own: nothing outside the three parts below.
//!
//! # The four parts
//!
//!   * the MAP -- the object map's own structures and the `Arc` it lives in.
//!     DashMap: every shard's hashbrown table and the shard array. The
//!     merged store: its bucket arrays, slab chunks and their tables, free
//!     lists, LFU frequency maps, the shard array and the two tail-mirror
//!     arrays;
//!   * the STACK -- the policy stack's structures
//!     (`PolicyStack::structure_bytes`: slab chunks and their table, the
//!     keyless index, the free list, the LFU bucket maps, the ghost) and the
//!     box it lives in;
//!   * the HEADERS -- one DRAM value header per live object: `live x` the
//!     header allocation's usable size in the build's layout
//!     (`value::dram_header_bytes`: 32 split, 16 `thin_header`);
//!   * the KEYS -- under the default layout, the allocation behind each
//!     `String`, `Vec<u8>` or `Box<[u8]>` key, which the header's inline `K`
//!     points at: `nallocx(len)`, charged when the object enters the map and
//!     refunded when it leaves (`AtomicStatus::key_heap_bytes`, kept beside the
//!     object count). Nothing for an inline key, nor under `thin_header`.
//!
//! Structures on the SLOW node are not in M -- the eviction stacks under
//! `eviction_stacks_pmem` -- and are reported apart (`DramMetadata::slow`).
//!
//! NOT attributed, and not in M: the channels (one crossbeam block per
//! channel once it has been used), the policy worker's reusable buffers and
//! each stack's transient `migrations` vector, the migration pipeline's
//! in-flight table (128 KiB, built on the first set), the status, the TTL
//! index (on the slow node), and anything outside the cache. `tests/dram_metadata_identity.rs` holds the DRAM pool to M plus P
//! with those taken out by a warm-up; the per-structure tests hold each
//! structure to `thread.allocated`.
//!
//! # How each part is kept
//!
//! The stack's structures and the merged store count their own allocations
//! where they make them -- a slab chunk committed, an index rehashed, a free
//! list's buffer grown, a bucket array pushed past its capacity -- so reading
//! them is a handful of loads. The LFU bucket maps allocate B-tree nodes
//! inside `BTreeMap`, so they allocate through [`Metered`], which charges a
//! [`Meter`]. The DashMap has no growth hook: the POLICY WORKER re-reads a
//! shard's table layout (`RawTable::allocation_info`, under the shard's read
//! lock, `try_read` so it never waits) at the end of any pass whose `Set`s
//! could have made that table reallocate -- see [`ShardState`]. The worker
//! publishes M into the status each pass, after a wipe, and after any event
//! that grew the stack (`PolicyWorker::publish_metadata`), where one load
//! reads it (`AtomicStatus::dram_metadata_bytes`).

use std::alloc::Layout;

/// What jemalloc hands out for a request of `size` bytes aligned to `align`:
/// its usable size. The alignment flag is the one the crate's allocators pass
/// (`MALLOCX_LG_ALIGN(log2 align)` whenever `align > 1`), which is also what
/// `numa_alloc::measured` charges. 0 for a zero-sized request, which allocates
/// nothing.
pub fn usable(size: usize, align: usize) -> u64 {
	if size == 0 {
		return 0;
	}

	// `MALLOCX_LG_ALIGN(la)` is `la` itself.
	let flags = if align > 1 { align.trailing_zeros() as core::ffi::c_int } else { 0 };

	// SAFETY: a pure size-class computation on a non-zero size.
	unsafe { tikv_jemalloc_sys::nallocx(size, flags) as u64 }
}

/// [`usable`] of a `Layout`.
pub fn layout_bytes(layout: Layout) -> u64 {
	usable(layout.size(), layout.align())
}

/// A `Vec<T>` holding `capacity` elements: `RawVec` allocates exactly
/// `Layout::array::<T>(capacity)`, whichever allocator it is in. 0 for a
/// zero capacity or a zero-sized `T`.
pub fn vec_bytes<T>(capacity: usize) -> u64 {
	usable(capacity.saturating_mul(core::mem::size_of::<T>()), core::mem::align_of::<T>())
}

/// The allocation `Arc::new` makes for a `T`: std's `ArcInner` is
/// `#[repr(C)] { strong, weak, data }`.
pub fn arc_bytes<T>() -> u64 {
	let (inner, _) = Layout::new::<[usize; 2]>()
		.extend(Layout::new::<T>())
		.expect("an ArcInner layout");

	layout_bytes(inner.pad_to_align())
}

/// The allocation a `Box` of `value` holds.
pub fn box_bytes_of_val<T: ?Sized>(value: &T) -> u64 {
	usable(core::mem::size_of_val(value), core::mem::align_of_val(value))
}

/// Bytes, by the node they are on.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct NodeBytes {
	pub dram: u64,
	pub slow: u64,
}

impl NodeBytes {
	pub fn dram(bytes: u64) -> Self {
		NodeBytes { dram: bytes, slow: 0 }
	}

	pub fn slow(bytes: u64) -> Self {
		NodeBytes { dram: 0, slow: bytes }
	}

	/// An eviction stack's own structures: in DRAM, or on the slow node under
	/// `eviction_stacks_pmem`, which allocates every one of them -- slab chunks
	/// and their table, index buckets, free lists, bucket maps, ghosts --
	/// through `crate::Hybrid`.
	pub fn stack(bytes: u64) -> Self {
		match cfg!(feature = "eviction_stacks_pmem") {
			true => NodeBytes::slow(bytes),
			false => NodeBytes::dram(bytes),
		}
	}

	pub fn total(&self) -> u64 {
		self.dram + self.slow
	}
}

impl core::ops::Add for NodeBytes {
	type Output = NodeBytes;

	fn add(self, other: NodeBytes) -> NodeBytes {
		NodeBytes { dram: self.dram + other.dram, slow: self.slow + other.slow }
	}
}

#[cfg(any(feature = "hybrid_cache_common", feature = "merged_object_store"))]
pub use metered::{Meter, Metered};

#[cfg(any(feature = "hybrid_cache_common", feature = "merged_object_store"))]
mod metered {
	use std::{
		alloc::{AllocError, Allocator, Global, Layout},
		ptr::NonNull,
		sync::{
			Arc,
			atomic::{AtomicI64, Ordering},
		},
	};

	use super::{arc_bytes, layout_bytes, vec_bytes};

	/// A byte count that a structure's allocations are charged to, shared by
	/// every part of it that allocates and read with one load.
	///
	/// Signed, because a part can free on one path what another allocated;
	/// only the sum means anything, and it never goes below zero.
	#[derive(Clone, Default)]
	pub struct Meter(Arc<AtomicI64>);

	impl Meter {
		pub fn new() -> Self {
			Meter::default()
		}

		/// Adds `bytes`, or takes them off when negative.
		#[inline]
		pub fn charge(&self, bytes: i64) {
			if bytes != 0 {
				self.0.fetch_add(bytes, Ordering::Relaxed);
			}
		}

		/// The bytes charged so far.
		#[inline]
		pub fn bytes(&self) -> u64 {
			self.0.load(Ordering::Relaxed).max(0) as u64
		}

		/// Records that a `Vec<T>`'s buffer went from `before` to `after`
		/// elements of capacity: nothing when it did not move.
		#[inline]
		pub fn vec_resized<T>(&self, before: usize, after: usize) {
			if before != after {
				self.charge(vec_bytes::<T>(after) as i64 - vec_bytes::<T>(before) as i64);
			}
		}

		/// The meter's own allocation, the `Arc` it counts in: an owner that
		/// reports its bytes to the allocator's last byte adds this once.
		/// Asked of the allocator once per process: the policy worker reads
		/// the LFU chain's bytes after every event.
		pub fn own_bytes() -> u64 {
			static OWN: std::sync::OnceLock<u64> = std::sync::OnceLock::new();

			*OWN.get_or_init(arc_bytes::<AtomicI64>)
		}
	}

	/// An allocator that charges every allocation's usable size to a [`Meter`]
	/// and takes it off again at the free, delegating the memory itself to `A`.
	///
	/// For structures that allocate out of sight -- a `BTreeMap`'s nodes -- so
	/// that their bytes are counted where they are made, like the rest of M.
	/// `grow` and `shrink` are the trait's defaults, which allocate, copy and
	/// free through the two methods below, so a resize is charged as what it
	/// is: the new block on, the old one off.
	#[derive(Clone)]
	pub struct Metered<A = Global> {
		inner: A,
		meter: Meter,
	}

	impl Metered {
		/// Over the global allocator: DRAM.
		pub fn new(meter: &Meter) -> Self {
			Metered { inner: Global, meter: meter.clone() }
		}
	}

	impl<A> Metered<A> {
		pub fn new_in(inner: A, meter: &Meter) -> Self {
			Metered { inner, meter: meter.clone() }
		}
	}

	// SAFETY: every block comes from `inner` and goes back to it with the
	// layout it was allocated with; the meter only counts.
	unsafe impl<A: Allocator> Allocator for Metered<A> {
		fn allocate(&self, layout: Layout) -> Result<NonNull<[u8]>, AllocError> {
			let block = self.inner.allocate(layout)?;
			self.meter.charge(layout_bytes(layout) as i64);

			Ok(block)
		}

		fn allocate_zeroed(&self, layout: Layout) -> Result<NonNull<[u8]>, AllocError> {
			let block = self.inner.allocate_zeroed(layout)?;
			self.meter.charge(layout_bytes(layout) as i64);

			Ok(block)
		}

		unsafe fn deallocate(&self, ptr: NonNull<u8>, layout: Layout) {
			self.meter.charge(-(layout_bytes(layout) as i64));

			// SAFETY: forwarded as the caller gave it.
			unsafe { self.inner.deallocate(ptr, layout) }
		}
	}

	// SAFETY: a clone allocates from the same `inner` (itself interchangeable
	// with its clones) and charges the same meter, so any clone can free what
	// another allocated.
	unsafe impl<A: std::alloc::AllocatorClone> std::alloc::AllocatorClone for Metered<A> {}
}

#[cfg(feature = "hybrid_cache_common")]
pub use published::*;

#[cfg(feature = "hybrid_cache_common")]
mod published {
	use std::hash::BuildHasher;

	use dashmap::DashMap;

	use super::{layout_bytes, usable, NodeBytes};
	use crate::{HashedKey, ObjectMapRef};

	/// M and its parts, as the policy worker last published them.
	#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
	pub struct DramMetadata {
		/// The object map's own structures, and the `Arc` it lives in.
		pub map: u64,
		/// The policy stack's own structures, and the box it lives in.
		pub stack: u64,
		/// One DRAM value header per live object.
		pub headers: u64,
		/// The heap bytes behind the live objects' keys: `nallocx(len)` per
		/// `String`, `Vec<u8>` or `Box<[u8]>` key under the default layout (the
		/// header counts the key's inline handle only), 0 for an inline key and
		/// under `thin_header`, where the key is inside the item.
		pub keys: u64,
		/// NOT in M: the cache's structures on the slow node (the eviction
		/// stacks under `eviction_stacks_pmem`), reported apart.
		pub slow: u64,
	}

	impl DramMetadata {
		/// M: the three DRAM parts.
		pub fn total(&self) -> u64 {
			self.map + self.stack + self.headers + self.keys
		}
	}

	/// The policy worker's reading of a DashMap's tables.
	///
	/// A shard's table reallocates only when a write reserves room in it at
	/// its load limit (hashbrown's `growth_left` at 0). DashMap reserves one
	/// slot BEFORE it looks for the key, in `insert` and in `entry` alike --
	/// so an overwrite reserves, and so does `erase`'s lookup, which goes
	/// through `entry`: a `del`, a TTL reap or an eviction into a shard at its
	/// load limit doubles it. `remove`, `retain` and `clear` never free a
	/// table. After a read, `growth_left` falls by at most one per write, so
	/// the table cannot reallocate before `growth_left + 1` writes.
	///
	/// So the worker counts every write it can see against the key's shard --
	/// each `Set`, `Del` and `Expire` it handles and each eviction it makes
	/// (`map_write`) -- and re-reads a shard once its count exceeds the
	/// `growth_left` of its last read: exact, conservative (a write that
	/// preceded the read is counted after it; an insert into a tombstone
	/// costs nothing), and cheap -- most passes re-read no shard at all. Each
	/// such write reaches the worker after it happened, and a pass drains its
	/// events before it reads, so a write the read did not see is counted after
	/// it.
	///
	/// One write the worker cannot see: a `del` of a key that is not there
	/// (and a collision's), which looks the key up with `entry` and returns an
	/// error, sending nothing. It is caught at the shard's next counted write
	/// -- a shard read at `growth_left == 0` is re-read at its first -- and in
	/// any case by the full re-read the worker makes every
	/// `FULL_REFRESH_INTERVAL` (`PolicyWorker::publish_metadata`).
	///
	/// A shard whose lock is held when the worker tries it stays queued for
	/// the next pass rather than blocking the worker.
	#[derive(Default)]
	pub struct ShardState {
		/// Per shard: its table's usable bytes at its last read.
		bytes: Vec<u64>,
		/// Per shard: `capacity - len` -- hashbrown's `growth_left` -- at its
		/// last read.
		headroom: Vec<usize>,
		/// Per shard: writes counted since its last read.
		writes: Vec<usize>,
		/// Per shard: queued for the next refresh.
		stale: Vec<bool>,
		/// The shards with `stale` set, so a refresh visits only those. Sized
		/// for every shard by the first refresh, so it never reallocates.
		queue: Vec<u32>,
		/// The sum of `bytes`.
		tables: u64,
		/// The shard array, fixed at construction.
		shard_array: u64,
	}

	/// A write into `key`'s shard (see [`ShardState`]): counts it, and queues
	/// the shard once it could have reallocated.
	pub fn dashmap_write<V, S>(map: &DashMap<HashedKey, V, S>, state: &mut ShardState, key: HashedKey)
	where
		S: BuildHasher + Clone,
	{
		// Not sized yet: the first refresh reads every shard anyway.
		if state.bytes.is_empty() {
			return;
		}

		let s = map.determine_map(&key);
		state.writes[s] += 1;

		if state.writes[s] > state.headroom[s] && !state.stale[s] {
			state.stale[s] = true;
			state.queue.push(s as u32);
		}
	}

	/// The DashMap's own bytes -- its shard array and every shard's table --
	/// re-reading the queued shards (all of them when `all`, and on the first
	/// call).
	pub fn dashmap_bytes<V, S>(map: &DashMap<HashedKey, V, S>, state: &mut ShardState, mut all: bool) -> u64
	where
		S: BuildHasher + Clone,
	{
		let shards = map.shards();

		if state.bytes.len() != shards.len() {
			let n = shards.len();

			*state = ShardState {
				bytes: vec![0; n],
				headroom: vec![0; n],
				writes: vec![0; n],
				stale: vec![false; n],
				queue: Vec::with_capacity(n),
				tables: 0,
				shard_array: usable(core::mem::size_of_val(shards), core::mem::align_of_val(&shards[0])),
			};

			all = true;
		}

		if all {
			state.queue.clear();

			for s in 0..shards.len() {
				state.stale[s] = true;
				state.queue.push(s as u32);
			}
		}

		let mut i = 0;

		while i < state.queue.len() {
			let s = state.queue[i] as usize;

			let Some(table) = shards[s].try_read() else {
				i += 1;
				continue;
			};

			let (_, layout) = table.allocation_info();
			let bytes = layout_bytes(layout);

			state.tables = state.tables - state.bytes[s] + bytes;
			state.bytes[s] = bytes;
			state.headroom[s] = table.capacity() - table.len();
			state.writes[s] = 0;
			state.stale[s] = false;

			drop(table);
			state.queue.swap_remove(i);
		}

		state.shard_array + state.tables
	}

	/// This build's object map's worker-side state.
	#[cfg(not(feature = "merged_object_store"))]
	pub type MapState = ShardState;

	#[cfg(feature = "merged_object_store")]
	pub type MapState = ();

	/// A write into the map at `key` that the policy worker saw: a `Set`,
	/// `Del` or `Expire` it handled, or an eviction it made. See
	/// [`ShardState`]. Nothing for the merged store, which counts itself.
	#[allow(unused_variables)]
	pub fn map_write<K, V>(objects: &ObjectMapRef<K, V>, state: &mut MapState, key: HashedKey) {
		#[cfg(not(feature = "merged_object_store"))]
		dashmap_write(&**objects, state, key);
	}

	/// The object map's own bytes, by node: its structures and the `Arc` it
	/// lives in (always DRAM), re-reading what could have grown since the last
	/// call (everything when `all`).
	#[allow(unused_variables)]
	pub fn map_bytes<K, V>(objects: &ObjectMapRef<K, V>, state: &mut MapState, all: bool) -> NodeBytes {
		#[cfg(not(feature = "merged_object_store"))]
		return NodeBytes::dram(super::arc_bytes_of_ref(objects) + dashmap_bytes(&**objects, state, all));

		#[cfg(feature = "merged_object_store")]
		return NodeBytes::dram(super::arc_bytes_of_ref(objects) + objects.structure_bytes());
	}
}

/// [`arc_bytes`] of the `T` an `Arc<T>` points at.
#[cfg(feature = "hybrid_cache_common")]
fn arc_bytes_of_ref<T>(_arc: &std::sync::Arc<T>) -> u64 {
	arc_bytes::<T>()
}

/// Test support: this thread's jemalloc `thread.allocated` less its
/// `thread.deallocated` -- the usable bytes it has allocated and not freed,
/// counted per thread, so a delta is independent of whatever the tests
/// running beside it do. The per-structure tests hold M's parts to it
/// exactly (the precedent: `object::overhead`'s and `arena_index`'s
/// allocator tests).
#[cfg(test)]
pub(crate) fn thread_live_bytes() -> i64 {
	fn counter(name: &core::ffi::CStr) -> u64 {
		let mut value: u64 = 0;
		let mut len = core::mem::size_of::<u64>();

		// SAFETY: reads one u64 statistic into a u64 of the size passed.
		let rc = unsafe {
			tikv_jemalloc_sys::mallctl(
				name.as_ptr(),
				&mut value as *mut u64 as *mut core::ffi::c_void,
				&mut len,
				core::ptr::null_mut(),
				0,
			)
		};

		assert_eq!(rc, 0, "{name:?} unavailable");
		value
	}

	counter(c"thread.allocated") as i64 - counter(c"thread.deallocated") as i64
}

#[cfg(test)]
mod tests {
	use std::alloc::Layout;

	#[cfg(feature = "hybrid_cache_common")]
	use dashmap::DashMap;

	use super::*;
	#[cfg(feature = "hybrid_cache_common")]
	use crate::HashedKey;

	/// `usable` is what the allocator hands out -- the crate's global
	/// allocator, which passes the alignment flag -- for small, large and
	/// over-aligned requests alike.
	#[test]
	fn usable_is_what_the_allocator_hands_out() {
		for (size, align) in [
			(1, 1), (12, 4), (16, 16), (20, 16), (84, 16), (100, 8), (152, 16), (1_104, 16),
			(4_097, 8), (14_337, 8), (131_072, 8), (163_840, 8), (32_768, 128), (1 << 22, 16),
		] {
			let layout = Layout::from_size_align(size, align).expect("a layout");
			let base = thread_live_bytes();

			// SAFETY: a non-zero layout, freed with the same layout below.
			// `black_box`, or the optimizer drops an allocation nothing reads.
			let block = std::hint::black_box(unsafe { std::alloc::alloc(layout) });
			assert!(!block.is_null());

			let got = (thread_live_bytes() - base) as u64;

			// SAFETY: allocated just above with `layout`.
			unsafe { std::alloc::dealloc(block, layout) };

			assert_eq!(got, usable(size, align), "{size} B aligned to {align}");
			assert_eq!(got, layout_bytes(layout));
		}

		assert_eq!(usable(0, 8), 0, "a zero-sized request allocates nothing");
		assert_eq!(vec_bytes::<u64>(0), 0);
		assert_eq!(vec_bytes::<()>(1_000), 0, "a zero-sized element allocates nothing");
	}

	/// `vec_bytes`, `arc_bytes` and `box_bytes_of_val` are what the allocator
	/// holds for the `Vec`, the `Arc` and the `Box` they describe.
	#[test]
	fn the_shape_helpers_are_what_the_allocator_holds() {
		for capacity in [1usize, 3, 4, 17, 1_000, 40_000] {
			let base = thread_live_bytes();
			let held: Vec<u32> = Vec::with_capacity(capacity);

			assert_eq!((thread_live_bytes() - base) as u64, vec_bytes::<u32>(held.capacity()), "Vec<u32> of {capacity}");
			drop(held);
		}

		let base = thread_live_bytes();
		let arc = std::sync::Arc::new([7u64; 5]);
		assert_eq!((thread_live_bytes() - base) as u64, arc_bytes::<[u64; 5]>());
		drop(arc);

		let base = thread_live_bytes();
		let boxed: Box<[u16]> = vec![1u16; 999].into_boxed_slice();
		assert_eq!((thread_live_bytes() - base) as u64, box_bytes_of_val(&*boxed));
		drop(boxed);
	}

	/// The node split: an eviction stack's structures are DRAM, or the slow
	/// node's under `eviction_stacks_pmem` -- and `total` is the same either
	/// way.
	#[test]
	fn a_stack_s_structures_are_on_the_node_its_build_puts_them() {
		let bytes = NodeBytes::stack(4_096);

		assert_eq!(bytes.total(), 4_096);

		match cfg!(feature = "eviction_stacks_pmem") {
			true => assert_eq!(bytes, NodeBytes::slow(4_096)),
			false => assert_eq!(bytes, NodeBytes::dram(4_096)),
		}
	}

	/// A `Metered` allocator counts exactly the usable bytes of the blocks a
	/// `BTreeMap` makes through it -- leaves and internal nodes -- as the map
	/// grows and shrinks, and gives them all back when it drops.
	///
	/// Not cloned: `BTreeMap::clone` over an allocator that owns a handle
	/// (this nightly's) leaks allocator clones -- the meter's `Arc` count
	/// never returns, so its 32 bytes are never freed (seen in a standalone
	/// probe: strong count 180 after the clone, 178 once both maps dropped).
	/// Nothing in the crate clones a metered map.
	#[cfg(feature = "hybrid_cache_common")]
	#[test]
	fn metered_counts_exactly_what_a_btree_allocates_through_it() {
		let base = thread_live_bytes();
		let live = || (thread_live_bytes() - base) as u64;

		let meter = Meter::new();
		let mut map: std::collections::BTreeMap<u32, (u32, u32), Metered> =
			std::collections::BTreeMap::new_in(Metered::new(&meter));

		assert_eq!(live(), meter.bytes() + Meter::own_bytes(), "an empty map allocates nothing");

		for k in 0..2_000u32 {
			map.insert(k.wrapping_mul(2_654_435_761), (k, k));
			assert_eq!(live(), meter.bytes() + Meter::own_bytes(), "after {} inserts", k + 1);
		}

		for k in (0..2_000u32).step_by(3) {
			map.remove(&k.wrapping_mul(2_654_435_761));
			assert_eq!(live(), meter.bytes() + Meter::own_bytes(), "after removing {k}");
		}

		drop(map);

		assert_eq!(meter.bytes(), 0, "every node was freed through the meter");
		drop(meter);
		assert_eq!(live(), 0);
	}

	/// The DashMap reading, exact after every write: a few shards so each one
	/// grows many times, objects built before the measurement so only the
	/// tables move, and a shard driven to its load limit and then OVERWRITTEN
	/// -- DashMap reserves before it looks, so the overwrite doubles it, and
	/// the reading has to catch a growth no fresh insert made.
	#[cfg(feature = "hybrid_cache_common")]
	#[test]
	fn the_dashmap_reading_is_exact_after_every_write() {
		type Map = DashMap<HashedKey, crate::object::Object<u64, crate::BufferDRAM>, crate::NoHasher>;

		const N: u64 = 20_000;

		let mut objects: Vec<_> = (0..N + 64).map(|i| crate::object::Object::new(i, &[1u8; 8], None)).collect();

		let base = thread_live_bytes();
		let map = Map::with_hasher_and_shard_amount(crate::NoHasher::default(), 4);
		let built = (thread_live_bytes() - base) as u64;

		// The first reading sizes the reading's own state -- the worker's, not
		// the map's -- so the baseline is taken again past it.
		let mut state = ShardState::default();
		let initial = dashmap_bytes(&map, &mut state, false);

		assert_eq!(built, initial, "an empty map: the shard array");

		let base = thread_live_bytes() - initial as i64;
		let live = || (thread_live_bytes() - base) as u64;

		let mut steps = 0;
		let mut last = dashmap_bytes(&map, &mut state, false);

		for i in 0..N {
			let key = i.wrapping_mul(0x9E37_79B9_7F4A_7C15);

			map.insert(key, objects.pop().expect("an object"));
			dashmap_write(&map, &mut state, key);

			let now = dashmap_bytes(&map, &mut state, false);
			assert_eq!(live(), now, "after insert {i}");

			if now != last {
				steps += 1;
				last = now;
			}
		}

		assert!(steps >= 4 * 12, "every shard grew a dozen times: {steps} steps");

		// Shard 0 of 4 is the keys whose bits 57 and 56 are 0: fill a fresh
		// map's shard 0 to its load limit, then overwrite.
		let map = Map::with_hasher_and_shard_amount(crate::NoHasher::default(), 4);
		let mut state = ShardState::default();
		let initial = dashmap_bytes(&map, &mut state, false);
		let base = thread_live_bytes() - initial as i64;
		let live = || (thread_live_bytes() - base) as u64;
		let shard0 = |i: u64| i;

		let mut i = 0;
		loop {
			map.insert(shard0(i), objects.pop().expect("an object"));
			dashmap_write(&map, &mut state, shard0(i));
			assert_eq!(live(), dashmap_bytes(&map, &mut state, false), "shard 0, insert {i}");

			let table = map.shards()[0].read();
			let full = table.capacity() == table.len() && table.len() >= 7;
			drop(table);

			if full {
				break;
			}

			i += 1;
		}

		let before = dashmap_bytes(&map, &mut state, false);
		let dropped = map.insert(shard0(0), objects.pop().expect("an object"));
		drop(dropped);

		// The overwrite's old value is freed; the table grew under it.
		let freed = crate::object::overhead::resident_object_bytes::<u64>(8) as u64 + crate::value::dram_header_bytes::<u64>();
		dashmap_write(&map, &mut state, shard0(0));
		let after = dashmap_bytes(&map, &mut state, false);

		assert!(after > before, "the overwrite at the load limit doubled shard 0: {before} -> {after}");
		assert_eq!(live() + freed, after, "and the reading saw it");
	}

	/// The worker's publication on a real tiered cache: at quiescence M's map
	/// part is what a fresh full reading of the map gives (so the reading
	/// kept up with every table's growth), the header part is one header per
	/// live object, and the total is `dram_metadata_bytes`, one load. Then a
	/// wipe, which keeps the tables and drops the headers.
	#[cfg(feature = "hybrid_cache_common")]
	#[test]
	fn a_tiered_cache_publishes_what_a_fresh_reading_of_it_gives() {
		use crate::{CacheTierSize, PaperCache, PaperPolicy, TieredBuffer};

		// The per-object metadata model (S5): the test is about M's
		// publication, which the model does not enter, and this 1 MiB tier is
		// smaller than the merged store's own empty structures -- under the
		// measured model's key ceiling it would refuse every key.
		let _per_object = crate::object::overhead::test_overheads::per_object();

		let cache = PaperCache::<u64, TieredBuffer>::new(
			64 << 20,
			CacheTierSize::Bytes(1 << 20),
			PaperPolicy::LruCompactHybrid,
		)
		.expect("a tiered cache");

		let value = [3u8; 200];

		for key in 0..3_000u64 {
			cache.set(key, &value, None).expect("set");
		}

		for key in (0..3_000u64).step_by(7) {
			cache.del(&key).expect("del");
		}

		let settled = |cache: &PaperCache<u64, TieredBuffer>, live: u64| {
			let deadline = std::time::Instant::now() + std::time::Duration::from_secs(20);

			loop {
				let m = cache.dram_metadata();
				let mut fresh = MapState::default();
				let map = map_bytes(&cache.objects, &mut fresh, true);
				let s = cache.hybrid_stats();

				if s.fast_objects + s.slow_objects == live
					&& m.map == map.dram
					&& m.headers == live * crate::value::dram_header_bytes::<u64>()
				{
					return (m, map);
				}

				assert!(
					std::time::Instant::now() < deadline,
					"the published map part never matched a fresh reading: {m:?} vs {map:?}, {} live",
					live,
				);

				std::thread::sleep(std::time::Duration::from_millis(2));
			}
		};

		let live = 3_000 - 3_000u64.div_ceil(7);
		let (m, map) = settled(&cache, live);

		assert_eq!(cache.dram_metadata_bytes(), m.total());
		assert!(m.stack > 0, "the stack's box at least");

		match cfg!(feature = "eviction_stacks_pmem") {
			true => assert!(m.slow > map.slow, "the stack's structures are on the slow node"),
			false => assert_eq!(m.slow, map.slow, "only a slow-node table is reported apart"),
		}
		assert_eq!(
			cache.effective_fast_capacity_measured(),
			(1u64 << 20).saturating_sub(m.total()),
		);

		cache.wipe().expect("wipe");

		let (wiped, _) = settled(&cache, 0);
		assert_eq!(wiped.headers, 0);

		match cfg!(feature = "merged_object_store") {
			true => assert!(wiped.map < m.map, "the merged slab freed its chunks"),
			false => assert_eq!(wiped.map, m.map, "a wipe keeps every table's capacity"),
		}
	}
}
