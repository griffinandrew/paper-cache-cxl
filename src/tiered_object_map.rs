/*
 * Copyright (c) Kia Shakiba
 *
 * This source code is licensed under the GNU AGPLv3 license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! A per-tier object map (`tiered_object_map`): the object map's ROWS live in
//! the tier their values live in.
//!
//! # Shape
//!
//! `N` shards (a power of two, DashMap's default count: four per CPU, rounded
//! up), each one lock (in DRAM) over TWO `hashbrown::HashTable`s of
//! `(HashedKey, Object)` rows:
//!
//!   * the FAST table, on the global allocator (DRAM, node 0), holding every key
//!     whose value's bytes are in the fast tier;
//!   * the SLOW table, on the slow-tier allocator (`crate::Hybrid`, the node-2
//!     arenas), holding every key whose value's bytes are in the slow tier.
//!
//! A key is in exactly one of the two, chosen by its value's tag
//! (`TieredValue::tier`, which a value never changes: a migration installs a
//! NEW value). The shard is picked from the hash's bits below the top seven, as
//! DashMap picks it, and both tables index on the low bits (the `NoHasher`'d key
//! is the hash).
//!
//! # Keeping the invariant
//!
//! Every way an object's value can change goes through this map under the
//! shard's write lock, and each one re-files the row:
//!
//!   * `insert` places the row in the new value's table and removes it from the
//!     other one if it was there;
//!   * `swap_value` (the migration's check-and-act, `crate::removal::SwapValue`)
//!     checks `ptr_eq`, swaps, and moves the row;
//!   * `get_mut_ref` hands out a guard ([`TieredRefMut`]) that, when it drops --
//!     still under the write lock -- moves the row if the value it guards now
//!     has a different tier. So `object.set_data(..)` through ANY mutable handle
//!     (the tests' and anything added later) cannot leave a row misfiled. The
//!     production `get_mut_ref` caller (`ttl`) does not change the tier.
//!
//! [`TieredObjectMap::audit`] checks the invariant (a diagnostic: it read-locks
//! every shard in turn).
//!
//! # Reads
//!
//! The shard's read lock, then the fast table, then the slow table: a fast hit
//! is one probe, a slow hit or a miss two. The slow probe touches CXL (the
//! slow table's control bytes and the row).
//!
//! # Accounting
//!
//! Each table allocates through a counting wrapper ([`Counted`]) that charges
//! the allocation's usable size (`crate::meta::layout_bytes`, jemalloc's size
//! class with the crate's alignment flags) to its tier's counter at the
//! allocation and takes it off at the free. So the map knows its own bytes per
//! tier exactly, with no shard re-reading (`crate::meta::map_bytes` under this
//! feature): the fast tables and the shard array are DRAM metadata (in M), the
//! slow tables are reported apart, on the slow node (`DramMetadata::slow`).

use std::{
	ops::{Deref, DerefMut},
	ptr::NonNull,
	sync::{
		Arc,
		atomic::{AtomicI64, AtomicU64, Ordering},
	},
};

use allocator_api2::alloc::{AllocError, Allocator, Global, Layout};
use hashbrown::HashTable;
use parking_lot::{MappedRwLockReadGuard, RwLock, RwLockReadGuard, RwLockWriteGuard};

use crate::{CacheError, CacheSize, HashedKey, Tier, TieredValue};
use crate::object::{Object, ObjectSize};
use crate::status::Cleared;

type Row<K, V> = (HashedKey, Object<K, V>);

/// An allocator that charges every allocation's usable size to a shared
/// counter and takes it off at the free, delegating the memory to `A`. The
/// `allocator_api2` twin of `crate::meta::Metered`, which `hashbrown` takes.
#[derive(Clone)]
pub struct Counted<A> {
	inner: A,
	bytes: Arc<AtomicI64>,
}

// SAFETY: every block comes from `inner` and goes back to it with the layout it
// was allocated with; the counter only counts.
unsafe impl<A: Allocator> Allocator for Counted<A> {
	fn allocate(&self, layout: Layout) -> Result<NonNull<[u8]>, AllocError> {
		let block = self.inner.allocate(layout)?;
		self.bytes.fetch_add(crate::meta::layout_bytes(layout) as i64, Ordering::Relaxed);

		Ok(block)
	}

	unsafe fn deallocate(&self, ptr: NonNull<u8>, layout: Layout) {
		self.bytes.fetch_sub(crate::meta::layout_bytes(layout) as i64, Ordering::Relaxed);

		// SAFETY: forwarded as the caller gave it.
		unsafe { self.inner.deallocate(ptr, layout) }
	}
}

type FastTable<K, V> = HashTable<Row<K, V>, Counted<Global>>;
type SlowTable<K, V> = HashTable<Row<K, V>, Counted<crate::Hybrid>>;

/// One shard: the lock (the `RwLock` around this) and the two tables.
pub struct Shard<K, V> {
	fast: FastTable<K, V>,
	slow: SlowTable<K, V>,
}

#[inline]
fn eq<K, V>(key: HashedKey) -> impl Fn(&Row<K, V>) -> bool {
	move |row| row.0 == key
}

#[inline]
fn rehash<K, V>(row: &Row<K, V>) -> u64 {
	row.0
}

/// A table this small is never shrunk.
const SHRINK_FLOOR: usize = 128;

/// Shrinks `table` to fit its rows when they fill under 3/8 of its capacity.
///
/// The one place this map departs from DashMap, which never shrinks a shard.
/// Rows FLOW between the two tables here -- a burst of new keys is admitted
/// fast and then demoted -- so a table that kept its high-water allocation
/// would hold DRAM for rows that now live in the slow table: measured before
/// this existed, lru-compact-hybrid with half of 4M objects demoted kept fast
/// tables as large as a DashMap holding all 4M (168 MB) next to 84 MB of slow
/// tables, and saved no DRAM at all.
///
/// The hysteresis: a table grows when full (hashbrown's 7/8 load) to half
/// full, and shrinks below 3/8 full to between 3/8 and 3/4 full -- so a
/// table must lose a quarter of its rows after a growth, or gain a third
/// after a shrink, before it resizes again. A shrink rehashes the table's rows
/// under the shard's write lock.
#[inline]
fn shrink_if_sparse<K, V, A: Allocator>(table: &mut HashTable<Row<K, V>, A>) {
	let capacity = table.capacity();

	if capacity > SHRINK_FLOOR && table.len() * 8 < capacity * 3 {
		table.shrink_to(table.len(), rehash);
	}
}

impl<K, V> Shard<K, V> {
	/// The table `key` is in, and its row.
	#[inline]
	fn find(&self, key: HashedKey) -> Option<(Tier, &Row<K, V>)> {
		if let Some(row) = self.fast.find(key, eq(key)) {
			return Some((Tier::Fast, row));
		}

		self.slow.find(key, eq(key)).map(|row| (Tier::Slow, row))
	}

	/// The table `key` is in, and a pointer to its object, valid while the
	/// caller holds this shard exclusively and changes neither table.
	#[inline]
	fn find_ptr(&mut self, key: HashedKey) -> Option<(Tier, NonNull<Object<K, V>>)> {
		if let Some(row) = self.fast.find_mut(key, eq(key)) {
			return Some((Tier::Fast, NonNull::from(&mut row.1)));
		}

		self.slow.find_mut(key, eq(key)).map(|row| (Tier::Slow, NonNull::from(&mut row.1)))
	}

	/// Removes `key`'s row from the `tier` table, and shrinks that table if
	/// the removal left it under a quarter full (see [`shrink_if_sparse`]).
	#[inline]
	fn remove_from(&mut self, tier: Tier, key: HashedKey) -> Option<Row<K, V>> {
		match tier {
			Tier::Fast => {
				let row = self.fast.find_entry(key, eq(key)).ok().map(|entry| entry.remove().0);
				shrink_if_sparse(&mut self.fast);
				row
			},
			Tier::Slow => {
				let row = self.slow.find_entry(key, eq(key)).ok().map(|entry| entry.remove().0);
				shrink_if_sparse(&mut self.slow);
				row
			},
		}
	}

	/// Files a row the shard does not hold in the table of its value's tier.
	#[inline]
	fn place(&mut self, row: Row<K, V>) {
		match row.1.value().tier() {
			Tier::Fast => {
				self.fast.insert_unique(row.0, row, rehash);
			},
			Tier::Slow => {
				self.slow.insert_unique(row.0, row, rehash);
			},
		}
	}

	/// Moves `key`'s row out of the `from` table into its value's table.
	fn relocate(&mut self, key: HashedKey, from: Tier) {
		let row = self.remove_from(from, key).expect("the row the guard names is in the table it was found in");

		self.place(row);
	}

	/// The row's own removal, from whichever table holds it, if `pred` holds
	/// of it: the check and the removal under the caller's write lock.
	fn take_if(&mut self, key: HashedKey, pred: impl FnOnce(&Object<K, V>) -> bool) -> Option<Object<K, V>> {
		let (tier, row) = self.find(key)?;

		if !pred(&row.1) {
			return None;
		}

		self.remove_from(tier, key).map(|row| row.1)
	}
}

/// Cache-line padded, as DashMap pads its shards.
#[repr(align(128))]
struct Padded<T>(T);

/// The per-tier object map. See the module doc.
pub struct TieredObjectMap<K, V> {
	shards: Box<[Padded<RwLock<Shard<K, V>>>]>,
	shift: u32,

	/// Usable bytes of every fast table, and of every slow table.
	fast_bytes: Arc<AtomicI64>,
	slow_bytes: Arc<AtomicI64>,

	/// Rows moved between the tables (a tier-changing insert, swap or guard):
	/// a diagnostic.
	moves: AtomicU64,
}

/// What [`TieredObjectMap::audit`] found: rows per table, and the rows filed in
/// the table of the OTHER tier (`misplaced_*`: a slow value in the fast table,
/// a fast value in the slow table). Both zero always: every change re-files
/// under the shard's write lock, so this holds at every instant, not only at
/// quiescence.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct MapAudit {
	pub fast_rows: u64,
	pub slow_rows: u64,
	pub misplaced_in_fast: u64,
	pub misplaced_in_slow: u64,
}

/// DashMap's default shard count: four per CPU, rounded up to a power of two
/// (at least 2, so `shift` stays under 64).
fn default_shard_amount() -> usize {
	static AMOUNT: std::sync::OnceLock<usize> = std::sync::OnceLock::new();

	*AMOUNT.get_or_init(|| (std::thread::available_parallelism().map_or(1, usize::from) * 4).next_power_of_two().max(2))
}

impl<K, V> Default for TieredObjectMap<K, V> {
	fn default() -> Self {
		Self::new()
	}
}

impl<K, V> TieredObjectMap<K, V> {
	pub fn new() -> Self {
		Self::with_shard_amount(default_shard_amount())
	}

	/// `amount` shards: a power of two, at least 2.
	pub fn with_shard_amount(amount: usize) -> Self {
		assert!(amount >= 2 && amount.is_power_of_two(), "shard amount must be a power of two >= 2");

		let fast_bytes = Arc::new(AtomicI64::new(0));
		let slow_bytes = Arc::new(AtomicI64::new(0));

		let shards = (0..amount)
			.map(|_| {
				Padded(RwLock::new(Shard {
					fast: HashTable::new_in(Counted { inner: Global, bytes: fast_bytes.clone() }),
					slow: HashTable::new_in(Counted { inner: crate::Hybrid::default(), bytes: slow_bytes.clone() }),
				}))
			})
			.collect::<Vec<_>>()
			.into_boxed_slice();

		TieredObjectMap {
			shards,
			shift: 64 - amount.trailing_zeros(),
			fast_bytes,
			slow_bytes,
			moves: AtomicU64::new(0),
		}
	}

	/// The shard index: the hash's bits below the top seven (hashbrown's tag),
	/// as DashMap picks it.
	#[inline]
	fn shard_index(&self, key: HashedKey) -> usize {
		((key << 7) >> self.shift) as usize
	}

	#[inline]
	fn shard(&self, key: HashedKey) -> &RwLock<Shard<K, V>> {
		&self.shards[self.shard_index(key)].0
	}

	pub fn shard_amount(&self) -> usize {
		self.shards.len()
	}

	/// A read-only handle to `key`'s object, under the shard's read lock.
	#[inline]
	pub fn get_ref(&self, key: &HashedKey) -> Option<MappedRwLockReadGuard<'_, Object<K, V>>> {
		let key = *key;

		RwLockReadGuard::try_map(self.shard(key).read(), |shard| shard.find(key).map(|(_, row)| &row.1)).ok()
	}

	/// A mutable handle to `key`'s object, under the shard's write lock. The
	/// handle re-files the row if the object's tier changed when it drops.
	#[inline]
	pub fn get_mut_ref(&self, key: &HashedKey) -> Option<TieredRefMut<'_, K, V>> {
		let key = *key;
		let mut guard = self.shard(key).write();
		let (table, object) = guard.find_ptr(key)?;

		Some(TieredRefMut { guard, key, table, object, moves: &self.moves })
	}

	pub fn contains_key(&self, key: &HashedKey) -> bool {
		self.shard(*key).read().find(*key).is_some()
	}

	/// The tier whose table holds `key`, if either does.
	pub fn table_of(&self, key: &HashedKey) -> Option<Tier> {
		self.shard(*key).read().find(*key).map(|(tier, _)| tier)
	}

	/// Inserts `object` at `key` in the table of its value's tier, returning
	/// the object it replaced; a replaced object in the OTHER table is removed
	/// from it. One write lock.
	pub fn insert(&self, key: HashedKey, object: Object<K, V>) -> Option<Object<K, V>> {
		let tier = object.value().tier();
		let mut shard = self.shard(key).write();

		match tier {
			Tier::Fast => {
				if let Some(row) = shard.fast.find_mut(key, eq(key)) {
					return Some(std::mem::replace(&mut row.1, object));
				}
			},
			Tier::Slow => {
				if let Some(row) = shard.slow.find_mut(key, eq(key)) {
					return Some(std::mem::replace(&mut row.1, object));
				}
			},
		}

		let other = match tier {
			Tier::Fast => Tier::Slow,
			Tier::Slow => Tier::Fast,
		};

		let old = shard.remove_from(other, key).map(|row| row.1);

		if old.is_some() {
			self.moves.fetch_add(1, Ordering::Relaxed);
		}

		shard.place((key, object));

		old
	}

	/// The migration's check-and-act (`crate::removal::SwapValue`): under the
	/// shard's write lock, if `key`'s value is still `expected` (the same
	/// header, `TieredValue::ptr_eq`), `swap` replaces it and the row moves to
	/// the table of the new value's tier.
	pub fn swap_value<R>(
		&self,
		key: &HashedKey,
		expected: &TieredValue<K>,
		swap: impl FnOnce(&mut Object<K, V>) -> R,
	) -> crate::removal::Swap<R> {
		let Some(mut object) = self.get_mut_ref(key) else {
			return crate::removal::Swap::Gone;
		};

		if !TieredValue::ptr_eq(object.value(), expected) {
			return crate::removal::Swap::Superseded;
		}

		// The guard re-files the row as it drops, under the same lock.
		crate::removal::Swap::Swapped(swap(&mut *object))
	}

	/// Removes and returns `key`'s object if `pred` holds of it, the check and
	/// the removal under one write lock.
	pub fn take_if(&self, key: &HashedKey, pred: impl FnOnce(&Object<K, V>) -> bool) -> Option<Object<K, V>> {
		self.shard(*key).write().take_if(*key, pred)
	}

	/// Removes and returns `key`'s object, live or not.
	pub fn take_evict(&self, key: &HashedKey) -> Option<Object<K, V>> {
		self.shard(*key).write().take_if(*key, |_| true)
	}

	/// The first key the map iterates (shards in order, fast table first): the
	/// eviction fallback's victim, as DashMap's `iter().next()` names one.
	pub fn first_key(&self) -> Option<HashedKey> {
		for shard in self.shards.iter() {
			let shard = shard.0.read();

			if let Some(row) = shard.fast.iter().next().or_else(|| shard.slow.iter().next()) {
				return Some(row.0);
			}
		}

		None
	}

	/// Rows in both tables, read-locking each shard in turn (DashMap's `len`).
	pub fn len(&self) -> usize {
		self.shards.iter().map(|shard| {
			let shard = shard.0.read();
			shard.fast.len() + shard.slow.len()
		}).sum()
	}

	pub fn is_empty(&self) -> bool {
		self.len() == 0
	}

	/// Rows per table, `(fast, slow)`.
	pub fn tier_lens(&self) -> (usize, usize) {
		self.shards.iter().fold((0, 0), |(fast, slow), shard| {
			let shard = shard.0.read();
			(fast + shard.fast.len(), slow + shard.slow.len())
		})
	}

	/// Every row against its table. See [`MapAudit`].
	pub fn audit(&self) -> MapAudit {
		let mut audit = MapAudit::default();

		for shard in self.shards.iter() {
			let shard = shard.0.read();

			audit.fast_rows += shard.fast.len() as u64;
			audit.slow_rows += shard.slow.len() as u64;
			audit.misplaced_in_fast += shard.fast.iter().filter(|row| row.1.value().tier() != Tier::Fast).count() as u64;
			audit.misplaced_in_slow += shard.slow.iter().filter(|row| row.1.value().tier() != Tier::Slow).count() as u64;
		}

		audit
	}

	/// Rows moved between the tables so far.
	pub fn moves(&self) -> u64 {
		self.moves.load(Ordering::Relaxed)
	}

	/// Removes every object, shard by shard under each write lock, counting
	/// each as it goes (`ObjectStore::clear_counted`). The tables keep their
	/// allocations, as DashMap's `retain` does (a wipe keeps every table's
	/// capacity: `meta`'s published-reading test holds it); the next removal
	/// from an emptied table shrinks it (`shrink_if_sparse`).
	pub fn clear_counted(&self, base_size: impl Fn(&Object<K, V>) -> ObjectSize) -> Cleared {
		let mut cleared = Cleared::default();

		for shard in self.shards.iter() {
			let mut shard = shard.0.write();

			let mut count = |row: &mut Row<K, V>| {
				cleared.objects += 1;
				cleared.base_bytes += base_size(&row.1) as CacheSize;

				false
			};

			shard.fast.retain(&mut count);
			shard.slow.retain(&mut count);
		}

		cleared
	}

	/// `f(key, tier, charge)` for every object, under each shard's read lock.
	pub fn for_each_value(&self, mut f: impl FnMut(HashedKey, Tier, ObjectSize)) {
		for shard in self.shards.iter() {
			let shard = shard.0.read();

			for row in shard.fast.iter().chain(shard.slow.iter()) {
				f(row.0, row.1.value().tier(), crate::object::overhead::resident_item_bytes(&row.1));
			}
		}
	}

	/// The usable bytes of every fast table (DRAM) and every slow table (the
	/// slow node), as their allocators counted them.
	pub fn table_bytes(&self) -> (u64, u64) {
		(
			self.fast_bytes.load(Ordering::Relaxed).max(0) as u64,
			self.slow_bytes.load(Ordering::Relaxed).max(0) as u64,
		)
	}

	/// The shard array's own allocation (DRAM).
	pub fn shard_array_bytes(&self) -> u64 {
		crate::meta::usable(core::mem::size_of_val(&*self.shards), core::mem::align_of::<Padded<RwLock<Shard<K, V>>>>())
	}

	/// The map's own bytes by node: the shard array and the fast tables in
	/// DRAM, the slow tables on the slow node. Not the `Arc` it lives in.
	pub fn structure_bytes(&self) -> crate::meta::NodeBytes {
		let (fast, slow) = self.table_bytes();

		crate::meta::NodeBytes { dram: self.shard_array_bytes() + fast, slow }
	}

	/// Test support: the bytes each table's allocation holds, asked of the
	/// tables (`HashTable::allocation_size`, rounded the way the allocator
	/// rounds it), for the accounting test to hold the counters to.
	#[cfg(test)]
	pub(crate) fn table_bytes_by_layout(&self) -> (u64, u64) {
		// hashbrown aligns a table's allocation to max(align_of::<T>(), the
		// SIMD group width): 16 with SSE2.
		let align = core::mem::align_of::<Row<K, V>>().max(16);

		self.shards.iter().fold((0, 0), |(fast, slow), shard| {
			let shard = shard.0.read();
			(
				fast + crate::meta::usable(shard.fast.allocation_size(), align),
				slow + crate::meta::usable(shard.slow.allocation_size(), align),
			)
		})
	}
}

/// [`TieredObjectMap::get_mut_ref`]'s handle: the shard's write guard and the
/// object. Dropping it re-files the row if the object's tier changed.
pub struct TieredRefMut<'a, K, V> {
	guard: RwLockWriteGuard<'a, Shard<K, V>>,
	key: HashedKey,
	/// The table the row was found in.
	table: Tier,
	/// The row's object: valid while `guard` is held and neither table
	/// changes, which nothing but this handle's own drop can do.
	object: NonNull<Object<K, V>>,
	moves: &'a AtomicU64,
}

impl<K, V> Deref for TieredRefMut<'_, K, V> {
	type Target = Object<K, V>;

	#[inline]
	fn deref(&self) -> &Object<K, V> {
		// SAFETY: see the field's doc.
		unsafe { self.object.as_ref() }
	}
}

impl<K, V> DerefMut for TieredRefMut<'_, K, V> {
	#[inline]
	fn deref_mut(&mut self) -> &mut Object<K, V> {
		// SAFETY: see the field's doc; `&mut self` makes this the only borrow.
		unsafe { self.object.as_mut() }
	}
}

impl<K, V> Drop for TieredRefMut<'_, K, V> {
	fn drop(&mut self) {
		// SAFETY: see the field's doc; the pointer is not used after this.
		let tier = unsafe { self.object.as_ref() }.value().tier();

		if tier != self.table {
			self.guard.relocate(self.key, self.table);
			self.moves.fetch_add(1, Ordering::Relaxed);
		}
	}
}

// ---------------------------------------------------------------------
// The object-map traits.
// ---------------------------------------------------------------------

#[cfg(any(feature = "all_dram", feature = "key_value_pmem"))]
impl<K, V> crate::object_store::ObjectStore<K, V> for TieredObjectMap<K, V> {
	fn get_ref(&self, key: &HashedKey) -> Option<impl Deref<Target = Object<K, V>> + '_> {
		TieredObjectMap::get_ref(self, key)
	}

	fn get_mut_ref(&self, key: &HashedKey) -> Option<impl DerefMut<Target = Object<K, V>> + '_> {
		TieredObjectMap::get_mut_ref(self, key)
	}

	fn insert(&self, key: HashedKey, object: Object<K, V>) -> Option<Object<K, V>> {
		TieredObjectMap::insert(self, key, object)
	}

	fn clear_counted(&self, base_size: impl Fn(&Object<K, V>) -> ObjectSize) -> Cleared {
		TieredObjectMap::clear_counted(self, base_size)
	}

	fn len(&self) -> usize {
		TieredObjectMap::len(self)
	}

	fn for_each_value(&self, f: impl FnMut(HashedKey, Tier, ObjectSize)) {
		TieredObjectMap::for_each_value(self, f)
	}
}

impl<K, V> crate::removal::Removal<K, V> for TieredObjectMap<K, V> {
	fn take_if(&self, key: &HashedKey, pred: impl FnOnce(&Object<K, V>) -> bool) -> Option<Object<K, V>> {
		TieredObjectMap::take_if(self, key, pred)
	}

	fn take_evict(&self, key: &HashedKey) -> Option<Object<K, V>> {
		TieredObjectMap::take_evict(self, key)
	}

	/// The first key the map iterates, counted like DashMap's fallback.
	fn fallback_victim(&self, fallbacks: &AtomicU64) -> Result<HashedKey, CacheError> {
		crate::ERASE_FALLBACK.fetch_add(1, Ordering::Relaxed);
		fallbacks.fetch_add(1, Ordering::Relaxed);

		self.first_key().ok_or_else(|| {
			log::error!("Object store is empty with non-zero used size");
			CacheError::Internal
		})
	}
}

impl<K, V> crate::removal::SwapValue<K, V> for TieredObjectMap<K, V> {
	fn swap_value<R>(
		&self,
		key: &HashedKey,
		expected: &TieredValue<K>,
		swap: impl FnOnce(&mut Object<K, V>) -> R,
	) -> crate::removal::Swap<R> {
		TieredObjectMap::swap_value(self, key, expected, swap)
	}
}

#[cfg(test)]
mod tests;
