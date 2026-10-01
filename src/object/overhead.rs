/*
 * Copyright (c) Kia Shakiba
 *
 * This source code is licensed under the GNU AGPLv3 license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::mem;

use typesize::TypeSize;

use crate::{
	StatusRef,
	policy::PaperPolicy,
	object::{Object, ObjectSize},
};

pub struct OverheadManager {
	status: StatusRef,
}

impl OverheadManager {
	pub fn new(status: &StatusRef) -> Self {
		OverheadManager {
			status: status.clone(),
		}
	}

	/// Returns the size of the object including non-policy-related overheads.
	/// The part of `base_size` that stays in DRAM whichever tier the object is in.
	///
	/// `Object::set_data` replaces only the value buffer, so a migration moves the
	/// value and nothing else. The key and the expiry live inline in the object map
	/// -- which is DRAM -- and a set TTL additionally owns an entry in `Expiries`,
	/// also DRAM. None of that moves, so none of it belongs in `fast_used` /
	/// `slow_used`.
	///
	/// The key and expiry are moreover already inside `shared_overhead`: the
	/// empirically fitted `OBJECT_MAP_ENTRY_OVERHEAD` was derived from the 40-byte
	/// `(HashedKey, Object)` pair, and `Object` is `{key, Arc ptr, expiry}`. Before
	/// this existed they were charged twice -- once here, once there.
	///
	/// Equals `base_size(object) - value bytes`, computed without touching the
	/// value so the set path does not pay an `Arc` clone.
	pub fn dram_resident_size<K, V>(&self, object: &Object<K, V>) -> ObjectSize
	where
		K: TypeSize,
	{
		// Under `thin_header` the key and the expiry are NOT in DRAM -- they are
		// in the item and tier with it -- and this figure is right anyway, because
		// what callers do with the figure is subtract it. `base_size` counts
		// them twice there: once as `key_size + 4`, and again inside
		// `resident_item_bytes`, which rounds the item's prefix plus `len`.
		// (A key held as bytes is no exception: its `key_size` is the 8-byte
		// hash, and its characters are only in the item.)
		// Taking the first copy off leaves `size - dram_resident` equal to the
		// item's own allocation, which is exactly what a migration moves and
		// what `fast_used`/`slow_used` charge. What the object keeps in DRAM
		// is the row, the 16-byte header and the stack node -- the terms the
		// per-object reservation names. Every design reserves them for every
		// live object, whichever tier its value is in, since none of them
		// moves on a demotion: the merged store in `settle_tier`, the DashMap
		// stacks in `reserved_overhead`.
		let mut resident =
			object.key_size() + mem::size_of::<crate::object::ExpireTime>() as ObjectSize;

		if object.expiry().is_some() {
			resident += get_ttl_overhead();
		}

		resident
	}

	pub fn base_size<K, V>(&self, object: &Object<K, V>) -> ObjectSize
	where
		K: TypeSize,
	{
		// The value is counted as the bytes jemalloc actually commits for the
		// object's OWN allocation, asked of the allocator rather than estimated
		// -- see `resident_item_bytes`. Before this it was counted as the
		// bytes *requested*, so a tier sized to its accounted bytes overran;
		// and under `thin_header` the allocation is `nallocx(bytes_offset::<K>()
		// + len)`, not `nallocx(len)`.
		//
		// The key and expiry are deliberately NOT scaled here: they are inside
		// `shared_overhead`, which applies its own factor to them already, and
		// scaling them twice is exactly the double-charge this module has been
		// untangling elsewhere. Under `thin_header` they are ALSO inside the
		// item this line now rounds, and `DOUBLE_COUNTED_IN_BASE_SIZE` takes
		// them back off in `get_policy_overhead` -- the two meet and cancel
		// exactly.
		let value = resident_item_bytes(object);
		let mut total_size = object.key_size()
			+ value
			+ mem::size_of::<crate::object::ExpireTime>() as ObjectSize;

		if object.expiry().is_some() {
			total_size += get_ttl_overhead();
		}

		total_size
	}

	/// `base_size` of the object a set WILL build -- its key, `len` value bytes
	/// and `ttl` -- computed before anything is allocated (S5): the size checks
	/// run from it, so an oversize value is refused without being built. The
	/// same terms as `base_size`, from the inputs the object is made of:
	/// `key_size()` is the key's accounted size (`key_accounted_size_for`),
	/// the item is `resident_item_bytes_for` the key and `len`, and an expiry
	/// exists exactly when the ttl is `Some` and not 0 (`expiry_from_ttl`).
	/// `base_size_for_equals_base_size` holds the two equal. `None` for a
	/// length no object can carry (over `u32::MAX`).
	pub fn base_size_for<K>(&self, key: &K, len: usize, ttl: Option<u32>) -> Option<ObjectSize>
	where
		K: TypeSize + 'static,
	{
		let len = ObjectSize::try_from(len).ok()?;
		let value = resident_item_bytes_for(key, len);
		let mut total_size = (crate::value::TieredValue::<K>::key_accounted_size_for(key) as ObjectSize)
			.checked_add(value)?
			.checked_add(mem::size_of::<crate::object::ExpireTime>() as ObjectSize)?;

		if ttl.is_some_and(|ttl| ttl != 0) {
			total_size = total_size.checked_add(get_ttl_overhead())?;
		}

		Some(total_size)
	}

	/// `dram_resident_size` of the object a set will build (see
	/// `base_size_for`).
	pub fn dram_resident_size_for<K>(&self, key: &K, ttl: Option<u32>) -> ObjectSize
	where
		K: TypeSize + 'static,
	{
		let mut resident = crate::value::TieredValue::<K>::key_accounted_size_for(key) as ObjectSize
			+ mem::size_of::<crate::object::ExpireTime>() as ObjectSize;

		if ttl.is_some_and(|ttl| ttl != 0) {
			resident += get_ttl_overhead();
		}

		resident
	}

	/// Returns the size of the object including base and policy-related overheads.
	pub fn total_size<K, V>(&self, object: &Object<K, V>) -> ObjectSize
	where
		K: TypeSize,
	{
		let policy = self.status.policy();
		self.base_size(object) + get_policy_overhead(&policy)
	}
}

/// Returns the per-object policy overhead.
/// Per-object cost every design pays regardless of policy or tiering: one
/// object-map row, and since v5 nothing else -- the value's separate
/// refcounted allocation is gone.
///
/// MEASURED at 80.0 B/object, R2 = 1.000000, and identical at 16-, 32-, 64-
/// and 128-byte values. The hybrid designs charge this against the fast tier
/// via `get_hybrid_dram_shared_overhead`; a non-tiered design has no fast
/// tier, so without this it went uncharged entirely.
/// Bytes `get_policy_overhead` adds ON TOP of `base_size`.
///
/// NOT simply the row. `OBJECT_MAP_ENTRY_OVERHEAD` is an ALLOCATION figure for
/// the whole `(HashedKey, Object)` row, and `Object` holds the key and the
/// expiry INLINE -- so the row already contains the two things `base_size`
/// counts separately. Adding the whole row on top charged them twice.
///
/// The error scaled inversely with object size -- ~0.4% on cluster13's 5.6 KB
/// objects, ~8.8% on cluster19's 100 B ones, where the cache held ~9% fewer
/// objects than the budget intended.
///
/// The value's `len: u32` is inside the row too and is NOT subtracted, because
/// `base_size` does not count it separately: it is bookkeeping the row pays
/// for, like the control byte.
const DOUBLE_COUNTED_IN_BASE_SIZE: ObjectSize =
	core::mem::size_of::<crate::HashedKey>() as ObjectSize
		+ core::mem::size_of::<crate::object::ExpireTime>() as ObjectSize;

/// 40 + 32 - 12 = **60**, and 40 + 16 - 12 = 44 under `thin_header`. Was
/// 80 + 0 - 12 = 68, and 96 + 32 - 12 = 116 before that.
const OBJECT_MAP_ROW_OVERHEAD: ObjectSize =
	OBJECT_MAP_ENTRY_OVERHEAD + VALUE_ALLOCATION_OVERHEAD - DOUBLE_COUNTED_IN_BASE_SIZE;

/// The merged store's structural cost per object, replacing BOTH the map row
/// and the eviction stack.
///
/// RE-MEASURED for the 40-byte slot -- jemalloc `stats.allocated`,
/// `measure_merged_store_point`, ONE PROCESS per point, `MSTORE_VALUE=64`,
/// least squares over 2^20..2^23. Run TWICE, once under the fused value layout
/// (a `fused_value` feature, since removed) and once under the split value
/// layout, because the slot is 40 bytes in both and the two fits are the check
/// on each other:
///
/// ```text
///                        slope B/object     R^2      less value   structural
///   merged + fused value     141.2966    0.999999        96         45.2966
///   merged + split value     141.2966    0.999999        96         45.2966
/// ```
///
/// The 96 is not assumed. The DashMap control ran in the same processes over
/// the same objects and fits 136.0008 (fused) and 136.0003 (split) B/object at
/// R^2 = 1.000000, and its row is `OBJECT_MAP_ENTRY_OVERHEAD` = 40 -- so the
/// value item is 136 - 40 = 96 from the other side, in both layouts. Fused, it
/// was ONE allocation, `nallocx(24 + 64)` = `nallocx(88)` = 96; under the
/// split layout it is TWO, `nallocx(64)` = 64 plus the 32-byte
/// `Arc<ValueHeader>`. That the two structural fits agree to four decimal
/// places is the evidence that this constant is a property of the SLOT and not
/// of the value layout.
///
/// It was **62**, and 62 was fitted when the merged slot was 56 bytes. The slot
/// is 40 bytes now, pinned by two const asserts in `merged_store.rs`, and the
/// same derivation the old figure used gives:
///
/// ```text
///   slab   40 B/slot   x 1.005 fill = 40.20   (was 56 x 1.005 = 56.27)
///   index   4 B/bucket x 1.313 fill =  5.25   (unchanged)
///                                    -----
///                                    45.45
/// ```
///
/// The slab fill is 1.005 because a chunked slab wastes at most one partly
/// filled 4096-slot chunk per shard -- bounded, where a growth factor is
/// proportional. The index fill is the bucket `Vec`'s own doubling: linear
/// hashing keeps `buckets.len()` equal to the live count, so the only slack
/// left is the `Vec` sitting mid-double, 1x to 2x, and the shards straddle a
/// power of two at every scale.
///
/// The derivation is quoted because the MEASUREMENT agrees with it, not in
/// place of it: 45.2966 fitted against 45.45 derived. The 16 bytes the slot
/// lost do not simply come off the old 62 -- both fill factors multiply the
/// slot, so the arithmetic had to be redone and then checked.
///
/// Set to **46** rather than 45, for the reason the old figure was 62 rather
/// than 61: the slope is approached from above, and every measured point past
/// 2^20 sits between 45.45 and 45.86 B/object structural -- 45.63 at 2^21,
/// 45.48 at 2^22, 45.45 at 2^23, and 45.86 / 45.57 at the mid-round points
/// 3 x 2^20 and 6 x 2^20 where the bucket `Vec` is furthest from full. 2^20
/// alone sits at 46.82, above this figure: a million objects over 32 shards
/// still leaves a partly filled 4096-slot chunk mattering. Rounding up keeps
/// the charge conservative -- the cache holds slightly fewer objects than the
/// budget allows, rather than overrunning it.
///
/// Against this, the split design costs 40 (the DashMap row, re-measured here
/// at 40.0003) + 40 (the measured compact hybrid eviction stack) = 80 B/object
/// of structure, against 45.3 here.
#[cfg(feature = "merged_object_store")]
const MERGED_STORE_STRUCTURE_OVERHEAD: ObjectSize = 46;

/// Under `merged_object_store` the object map IS the eviction stack, so the
/// per-policy stack terms below do not apply at all -- there is no second
/// structure to charge for. Charging them anyway is what left the measured
/// saving entirely unrealized: `used_size` kept billing every object for a
/// stack row that no longer exists, so the cache held fewer objects than its
/// budget allowed and the saving showed up nowhere.
///
/// Same shape as `OBJECT_MAP_ROW_OVERHEAD`: the slot embeds the `Object`,
/// hence reaches the key and expiry that `base_size` counts separately, so
/// those 12 bytes come back off. The value-allocation term is now layout
/// dependent, so this is too:
///
/// ```text
///   split value    46 + 32 - 12 = 66     (was 62 + 32 - 12 = 82)
///   thin_header    46 + 16 - 12 = 50
/// ```
///
/// Under `thin_header` the 12 that comes off here is exactly the 12 that
/// `resident_object_bytes` puts back inside `base_size`, because the key and
/// expiry are inside the item's `bytes_offset::<K>()` header. The two halves
/// of the correction meet here and cancel, which is why `total_size` under
/// `thin_header` is `nallocx(bytes_offset + len) + 16 + 46`: the item, the
/// DRAM header and the structure, and nothing else.
///
/// The two functions agree here in a way they do not for the split designs:
/// both name `MERGED_STORE_STRUCTURE_OVERHEAD`, and they differ by exactly
/// `DOUBLE_COUNTED_IN_BASE_SIZE` and nothing else.
#[cfg(feature = "merged_object_store")]
pub fn get_policy_overhead(_policy: &PaperPolicy) -> ObjectSize {
	#[cfg(all(test, feature = "hybrid_cache_common"))]
	if let Some(overhead) = test_overheads::policy() {
		return overhead;
	}

	MERGED_STORE_STRUCTURE_OVERHEAD + VALUE_ALLOCATION_OVERHEAD
		- DOUBLE_COUNTED_IN_BASE_SIZE
}

/// Test support: the two per-object constants that differ between the object
/// stores -- the fast-tier reservation, omega (`get_hybrid_dram_shared_overhead`)
/// and the per-object overhead `used_size` adds (`get_policy_overhead`) --
/// overridden for the calling thread while a `Guard` lives. T14 sets both, so
/// the stores' decisions are compared with neither a data-structure cost that
/// differs by design nor an eviction trigger that fires at a different op. A
/// thread-local rather than the environment (`PAPER_DISABLE_SHARED_OVERHEAD`):
/// the lib's tests run in parallel, on threads of one process.
#[cfg(all(test, feature = "hybrid_cache_common"))]
pub(crate) mod test_overheads {
	use std::cell::Cell;

	use super::ObjectSize;

	thread_local! {
		static OMEGA: Cell<Option<ObjectSize>> = const { Cell::new(None) };
		static POLICY: Cell<Option<ObjectSize>> = const { Cell::new(None) };
		static PER_OBJECT: Cell<bool> = const { Cell::new(false) };
	}

	pub(crate) fn omega() -> Option<ObjectSize> {
		OMEGA.with(Cell::get)
	}

	pub(crate) fn policy() -> Option<ObjectSize> {
		POLICY.with(Cell::get)
	}

	/// Whether a cache built on this thread is pinned to the per-object
	/// metadata model (S5): `set` and `per_object` pin it.
	pub(crate) fn per_object_pinned() -> bool {
		PER_OBJECT.with(Cell::get)
	}

	/// Both overrides, for this thread, until the guard drops -- and the
	/// per-object metadata model (S5), whose arithmetic they are: T14's
	/// constants are omega's.
	#[must_use]
	pub(crate) fn set(omega: ObjectSize, policy: ObjectSize) -> Guard {
		OMEGA.with(|cell| cell.set(Some(omega)));
		POLICY.with(|cell| cell.set(Some(policy)));
		PER_OBJECT.with(|cell| cell.set(true));

		Guard
	}

	/// The per-object metadata model alone, for this thread, until the guard
	/// drops (S5): the real-cache lib tests at toy scales, whose fast tiers
	/// the measured M's fixed first allocations would fill.
	#[must_use]
	pub(crate) fn per_object() -> Guard {
		PER_OBJECT.with(|cell| cell.set(true));

		Guard
	}

	pub(crate) struct Guard;

	impl Drop for Guard {
		fn drop(&mut self) {
			OMEGA.with(|cell| cell.set(None));
			POLICY.with(|cell| cell.set(None));
			PER_OBJECT.with(|cell| cell.set(false));
		}
	}
}

/// Every arm charges `OBJECT_MAP_ROW_OVERHEAD`, the tiered designs' too: they
/// did not once, so a tiered design was charged ~40 B/object against a real
/// cost near 200, `max_size` did not bound memory for any of them, and the
/// flat arms -- which always included the row -- were not comparable with them.
///
/// The two functions of this module now name the same three quantities -- the
/// stack ([`stack_dram_overhead`]), the map row and the value allocation --
/// and differ in one deliberate way: `get_hybrid_dram_shared_overhead` does
/// NOT subtract `DOUBLE_COUNTED_IN_BASE_SIZE`, because it is a fast-tier
/// RESERVATION rather than an addition on top of `base_size`, so it has nothing
/// to double-count against.
#[cfg(not(feature = "merged_object_store"))]
pub fn get_policy_overhead(policy: &PaperPolicy) -> ObjectSize {
	#[cfg(all(test, feature = "hybrid_cache_common"))]
	if let Some(overhead) = test_overheads::policy() {
		return overhead;
	}

	stack_dram_overhead(policy) + OBJECT_MAP_ROW_OVERHEAD
}

pub fn get_ttl_overhead() -> ObjectSize {
	// The index stores `(tick, HashedKey)` -- 16 bytes with alignment -- plus
	// 48 for the BTree node slot. Was `(Instant, HashedKey)` at 24 before
	// expiry shrank to a 4-byte tick; the tuple pads back to 16 either way, so
	// the total is unchanged, but the expression now names the real type.
	mem::size_of::<(u32, crate::HashedKey)>() as ObjectSize + 48
}

// ── Measured building blocks for the hybrid-cache DRAM reservation below ───
//
// Unlike `get_policy_overhead`'s eyeballed round numbers (explicitly
// documented above as "just rough estimates"), the constants in this section
// were derived from `std::mem::size_of` measurements of the *actual*
// concrete types involved (`HashedKey = u64`, `ObjectSize = u32`, the 1-byte
// `Tier` tag, `dlv_list::Index<T>` — 16 bytes regardless of `T`, since it's
// just `{ generation: u64, index: NonMaxUsize }` — and `kwik::HashList`'s
// heap-allocated `Entry<T>` node — `size_of::<T>() + 16` for its two
// intrusive-list pointers), combined with `hashbrown`'s documented ~7/8
// maximum load factor to get a real amortized per-entry cost rather than a
// flat guess. This matters here specifically because `get_policy_overhead`'s
// "48 bytes for the HashList entry" turned out — on inspection — to be
// `size_of::<HashList<..>>()` (the *container's* fixed struct size: a 32-byte
// HashMap header + 2 pointers), not a per-entry cost at all; reusing it
// verbatim (as an earlier version of this function did) inherited that
// mismatch on top of a redundant separate "+8 for the key" charge (the key is
// already stored once, inside the list's heap node).
//
// `hashbrown_entry_cost(raw_pair_size)` — the per-entry cost (amortized over
// load factor, control byte included) of a `hashbrown`-based map (backs
// `std::collections::HashMap`, `dashmap`, and `hashbrown::HashMap` alike)
// storing a pair of `raw_pair_size` bytes — is, at the worst-case point right
// before a table resize (capacity C satisfies `entries <= (7/8) * C`, so
// `C ≈ (8/7) * entries`, each bucket costing `raw_pair_size + 1` bytes: the
// pair plus one control byte): `ceil((8/7) * (raw_pair_size + 1))`. Applied
// below with pair sizes taken from real `size_of::<(HashedKey, V)>()`
// measurements (which include Rust's own alignment padding, e.g. a 9-byte
// `(u64, u8)` logical pair actually occupies 16 bytes).
//
//   HashedKey entry alone (map's own key, e.g. for the shared object
//     hashtable):                     cost(8)  = ceil(9*8/7)   = ceil(10.29) = 11
//   HashMap<HashedKey, Tier>:         cost(16) = ceil(17*8/7)  = ceil(19.43) = 20
//   HashMap<HashedKey, ObjectSize>:   cost(16) = ceil(17*8/7)  = ceil(19.43) = 20
//   HashMap<HashedKey, Index<_>>:     cost(24) = ceil(25*8/7)  = ceil(28.57) = 29
//   kwik HashList<HashedKey> entry:   24 (heap Entry<HashedKey> node:
//                                     8 data + 8 prev + 8 next) + cost(16)
//                                     (internal map slot: two 8-byte
//                                     pointers) = 24 + 20 = 44

// HASHTABLE_ENTRY_OVERHEAD (11) was deleted: already dead, superseded by
// OBJECT_MAP_ENTRY_OVERHEAD, which is an ALLOCATION figure for the whole row
// rather than a hand-derived slot estimate. Its derivation notes went with it;
// `numa_alloc::measured` counts the row directly under `measured_accounting`.

/// Per-object DRAM cost of an ARENA eviction stack, **40 B/object**: the one term
/// every tiered design's stack is charged ([`stack_dram_overhead`]).
///
/// One 32-byte arena node -- key 8, prev 4, next 4 and the 16-byte
/// `NodePayload` (size, frequency, aging epoch, queue, tier, resident bytes) --
/// plus the KEYLESS index that finds it: bare `u32` slot numbers verified
/// against the slot's own key, four bytes a bucket at the half load it grows
/// to, so 8 B/object. There is no `entries` map and no per-key list node: the
/// slot the index returns already carries tier, size and, where a policy has
/// one, its count or reference bit. A key is in exactly one queue at a time, so
/// the number of queues -- one for LRU, FIFO and CLOCK, two or three for the 2Q
/// and S3-FIFO families, four for the size-split LRU -- does not change it; nor
/// does the policy: the LFU-ranked stacks keep two ordered bucket maps over the
/// same node and index (`ArenaFrequencyChain`), O(distinct frequencies) rather
/// than O(objects). CLOCK's reference bit rides in `freq`, so it costs not a
/// byte more than FIFO; a ghost queue's memory is not per tracked object and is
/// charged apart ([`GHOST_ENTRY_DRAM_OVERHEAD`]).
///
/// MEASURED, and predicted rather than merely fitted (32 + 8): jemalloc
/// `stats.allocated`, `measure_one_point` (`policy_stack::measure_overhead`),
/// release, ONE PROCESS PER POINT, at powers of two, so every point sits at the
/// same phase of the doubling cycle:
///
/// ```text
///   policy                              2^20      2^21      2^22      2^23
///   lru-compact-hybrid               40.2100   40.1050   40.0518   40.0244
///   fifo-compact-hybrid              40.2100   40.1050   40.0518   40.0244
///   lru-sized-compact-hybrid         40.2100   40.1050   40.0518   40.0244
///   lfu-compact-hybrid               40.2252   40.1126   40.0556   40.0263
///   lru-lfu-compact-hybrid-2         40.2274   40.1137   40.0561   40.0266
/// ```
///
/// The residue above 40 is a fixed intercept, not a per-object term, which is
/// why it shrinks with n; LFU sits ~0.002 B/object above LRU at every point
/// for the two bucket maps, a fixed ~15 KB. The 2Q and S3-FIFO stacks share
/// the node and the index and carry the same term by construction, not by a
/// measurement of their own.
///
/// Forty is the 2^k figure. Off 2^k, DERIVED from the growth rules and not
/// measured: the node is 32 B/object at every population (plus at most one
/// partly filled 128 KiB chunk and 24 B of chunk table per chunk), and only the
/// index swings, 8-16 B/object -- so the stack costs 40-48 B/object across a
/// growth cycle. Under `thin_header` the split design is this stack, the map
/// row (40 B/object at 2^k, 23-46 across its 7/16..7/8 load) and the 16 B
/// header: ~80-104 B/object against the 96 charged.
///
/// Before the arena conversion every one of these was 72 (a 16-byte slot and a
/// 56-byte index that stored each key a second time, so a probe could compare
/// it): 72.5952 / 72.2962 / 72.1451 / 72.0707 for LFU and 72.5760 / 72.2866 /
/// 72.1403 / 72.0698 for LRU-LFU at 2^20..2^23, measured from a `git archive`
/// of the pre-conversion commit, built and run the same way. Statements of 72
/// or 112 B/object elsewhere in the tree describe those retired layouts.
///
/// ALLOCATED, not resident -- size-class-rounded usable bytes, the quantity
/// `malloc_usable_size` returns and Redis reports as `used_memory` -- so no
/// resident factor scales it. A field-by-field derivation understated this
/// stack by roughly a third: it counted struct fields and not size-class
/// rounding, index load factor or the growth slack of every doubling
/// structure.
#[cfg(any(feature = "hybrid_cache_common", not(feature = "merged_object_store")))]
pub const ARENA_STACK_DRAM_OVERHEAD: ObjectSize = 40;

/// The eviction stack's own per-object DRAM cost under `policy`: what
/// `get_policy_overhead` adds to the map row and, for a tiered design only,
/// what `get_hybrid_dram_shared_overhead` reserves out of the fast tier.
///
/// Every TIERED design is one arena stack, [`ARENA_STACK_DRAM_OVERHEAD`]. The
/// FLAT stacks are the older slab layout, and were re-measured with
/// `measure_one_point`: each fits its term to within 0.01 B/object at
/// R^2 = 1.000000. 56 for the one-queue stacks -- a 16-byte link-only slot plus
/// one 12-byte index entry (8-byte key, 4-byte slot number; the CLOCK/SIEVE
/// visited bit and MRU's held key live in the index value) -- and 72 for the
/// stacks whose index value carries a payload (2Q's queue tag and size, ARC,
/// S3-FIFO's tag and frequency counter). Neither charge covers a bare-key ghost
/// queue.
///
/// The match is exhaustive deliberately: adding a policy without giving it an
/// overhead term is a compile error rather than a silent zero.
#[cfg(any(feature = "hybrid_cache_common", not(feature = "merged_object_store")))]
fn stack_dram_overhead(policy: &PaperPolicy) -> ObjectSize {
	match policy {
		PaperPolicy::LfuCompact
		| PaperPolicy::FifoCompact
		| PaperPolicy::ClockCompact
		| PaperPolicy::SieveCompact
		| PaperPolicy::MruCompact
		| PaperPolicy::LruCompact => 56,

		PaperPolicy::TwoQCompact(..)
		| PaperPolicy::Arc
		| PaperPolicy::SThreeFifoCompact(..) => 72,

		PaperPolicy::LruCompactHybrid
		| PaperPolicy::LfuCompactHybrid
		| PaperPolicy::LruSizedCompactHybrid
		| PaperPolicy::LruLfuCompactHybrid(..)
		| PaperPolicy::FifoCompactHybrid
		| PaperPolicy::ClockCompactHybrid
		| PaperPolicy::TwoQCompactHybrid(..)
		| PaperPolicy::TwoQFastAdmissionReprieveCompactHybrid(..)
		| PaperPolicy::TwoQFullFastAdmissionCompactHybrid(..)
		| PaperPolicy::TwoQGhostCompactHybrid(..)
		| PaperPolicy::S3FifoCompactHybrid(..)
		| PaperPolicy::S3FifoFaithfulCompactHybrid(..)
		| PaperPolicy::S3FifoFaithfulFastAdmissionCompactHybrid(..)
		| PaperPolicy::S3FifoFaithfulReprieveCompactHybrid(..)
		| PaperPolicy::S3FifoFaithfulFastAdmissionReprieveCompactHybrid(..)
		| PaperPolicy::S3FifoGhostCompactHybrid(..)
		| PaperPolicy::S3FifoGhostLazyDemotionCompactHybrid(..)
		| PaperPolicy::S3FifoGhostLazyDemotionFastAdmissionCompactHybrid(..)
		| PaperPolicy::S3FifoGhostLazyDemotionFastAdmissionMidpointCompactHybrid(..)
		| PaperPolicy::S3FifoLazyDemotionReprieveCompactHybrid(..)
		| PaperPolicy::S3FifoLazyDemotionFastAdmissionReprieveCompactHybrid(..)
		| PaperPolicy::S3FifoLazyDemotionFastAdmissionMidpointReprieveCompactHybrid(..)
		| PaperPolicy::S3FifoLazyDemotionFastAdmissionSplitSlowReprieveCompactHybrid(..) => ARENA_STACK_DRAM_OVERHEAD,
	}
}

/// Per-*ghost-entry* DRAM cost shared by every hybrid design that keeps a
/// bare-key ghost queue (`TwoQGhostCompactHybrid`, and the `S3Fifo*Ghost*`
/// variants).
///
/// One `HashList<HashedKey>` node: 24-byte heap `Entry<HashedKey>` (8 data +
/// 8 prev + 8 next) plus the list's internal key->node map slot,
/// `cost(16) = 20`. Same 44 every other list-entry term in this module uses.
///
/// Deliberately *not* a per-tracked-object charge, and so deliberately not
/// part of [`get_hybrid_dram_shared_overhead`]'s return value: a ghost entry
/// exists precisely for a key that is *no longer in the cache*, so there is
/// no tracked-object count to multiply it by and — critically — it has **no
/// object-hashtable slot**, which is why it must never carry the
/// object-map row term. The owning stacks multiply it by
/// `ghost.len()` inside their own `reserved_overhead`.
///
/// **8 bytes**, not 44: the ghost is a `GhostFilter` — a 4-byte fingerprint
/// plus a 4-byte insertion timestamp, per S3-FIFO's own description of G as
/// "part of the indexing structure". It was a `HashList<HashedKey>`, whose
/// heap `Entry { key, prev, next }` (24) plus index slot (20) cost 44 bytes to
/// hold an 8-byte key, and whose capacity bound was unreachable from the path
/// that populated it: on a no-reuse trace the ghost grew without limit, to
/// 1.94 GB — 45% of a 4 GiB fast tier — on Twitter cluster38.
///
/// Gated on `eviction_stacks_pmem` **only**: when that feature moves the
/// eviction stacks — ghost list included — to PMEM, the ghost costs the
/// fast/DRAM tier nothing and the term drops to 0.
///
/// Unlike the per-policy constants below this is *not* gated on
/// `hybrid_cache_common`: the split policy stacks compile -- and reference
/// this -- under every feature combination, including none, except in a
/// merged build's lib, whose stack is the object map (see
/// `worker::policy::policy_stack`); gated as they are.
#[cfg(not(feature = "eviction_stacks_pmem"))]
#[cfg(any(test, not(feature = "merged_object_store")))]
pub const GHOST_ENTRY_DRAM_OVERHEAD: ObjectSize = 8;

/// Per-entry DRAM cost of an EXACT ghost queue -- a `CompactQueueSet<()>` of
/// bare keys, as the faithful S3-FIFO family carries.
///
/// Distinct from [`GHOST_ENTRY_DRAM_OVERHEAD`] above, which sizes a `GhostSlot`
/// FINGERPRINT (8 bytes, approximate, fixed-capacity). An exact ghost costs a
/// 16-byte `QueueSlot` plus the 12-byte index entry that finds it -- the same
/// 16 + 12 shape charged for `LruCompact` -- because flat S3-FIFO's
/// ghost is exact and a faithful port cannot substitute an approximate filter
/// without changing which keys get admitted to main.
///
/// 3.5x the fingerprint's cost per entry, and that is the real price of
/// fidelity here; it is bounded by the main queue's length, which the ghost is
/// trimmed against.
#[cfg(not(feature = "eviction_stacks_pmem"))]
#[cfg(any(test, not(feature = "merged_object_store")))]
pub const EXACT_GHOST_ENTRY_DRAM_OVERHEAD: ObjectSize = 16 + 12;

/// PMEM-resident ghost list: costs the fast/DRAM tier nothing. See the
/// `not(eviction_stacks_pmem)` arm above for the derivation and rationale.
#[cfg(feature = "eviction_stacks_pmem")]
#[cfg(any(test, not(feature = "merged_object_store")))]
pub const GHOST_ENTRY_DRAM_OVERHEAD: ObjectSize = 0;

/// Zero under `eviction_stacks_pmem` for the same reason as
/// [`GHOST_ENTRY_DRAM_OVERHEAD`]: `CompactQueueSet` is allocator-parameterised,
/// so the exact ghost follows the eviction stacks to the far node and stops
/// occupying fast-tier DRAM.
#[cfg(feature = "eviction_stacks_pmem")]
#[cfg(any(test, not(feature = "merged_object_store")))]
pub const EXACT_GHOST_ENTRY_DRAM_OVERHEAD: ObjectSize = 0;

/// Approximate per-object DRAM cost of the *shared* structures (the object
/// hashtable + the eviction stacks) that hold an entry for every object of both
/// tiers. Used by the LRU/LFU/LRU-sized hybrid stacks to reserve room in the
/// fast-tier (DRAM) budget so demotion bounds total DRAM, not just fast-tier
/// values.
///
/// The MODEL of those structures. Since S5a `crate::meta` also counts them
/// from their allocations (M, `AtomicStatus::dram_metadata_bytes`), and this
/// per-object figure stays what every settle reserves until S5 switches the
/// budget onto M -- and after that, a fallback and the sanity check M is
/// reported beside.
///
/// Unlike [`get_policy_overhead`] — which `used_size` charges unconditionally
/// because the eviction-stack bytes count toward the overall DRAM+PMEM budget
/// regardless of which tier they physically live in — this counts only the
/// terms that are actually DRAM-resident: the eviction-stack term is dropped
/// when `eviction_stacks_pmem` moves those stacks to PMEM, and the hashtable
/// entry is dropped when a hashtable-PMEM feature moves the object map to PMEM.
/// Per-object DRAM cost of the value's own allocation, over and above its
/// bytes. **Zero since v5**, and measured to be zero.
///
/// It was 48, then 32, and it is now nothing at all. `Arc`'s inner allocation
/// carried a strong AND a weak count around the 24-byte `TieredBuffer` enum --
/// 40 bytes, rounded to jemalloc's 48-byte class. `shared::Shared` dropped the
/// weak count nothing in this tree ever used: 8 + 24 = 32, one class down.
/// v5 removes the refcount entirely: a value IS its bytes, addressed by an
/// eight-byte word that lives inside the `Object`, so there is no second
/// allocation to charge for and nothing that could be called a header.
///
/// MEASURED, not asserted by inspection. `measure_object_map_point` at
/// MEASURE_VALUE = 64, 2^20..2^23, one process per point:
///
/// ```text
///   before (Shared<TieredBuffer>)   176.0000 B/object   R2 = 1.000000
///   after  (TieredValue)            144.0000 B/object   R2 = 1.000000
/// ```
///
/// and sweeping the value size at n = 2^22 puts the whole remainder in the map
/// row rather than leaving any third term behind:
///
/// ```text
///   value  16 ->  95.998 B/object   non-value remainder 79.998
///   value  32 -> 111.998                                79.998
///   value  64 -> 143.998                                79.998
///   value 128 -> 207.998                                79.998
/// ```
///
/// A constant that is zero is kept rather than deleted so the arithmetic below
/// still NAMES the term: `used_size` charging a value-allocation header is a
/// decision, and a build that reintroduces one (variant A's four-byte length
/// prefix, say) has a single place to say so.
// Not cfg-gated: the term applies to every design, tiered or not.
/// The value's own separate allocation header, which EXISTS AGAIN.
///
/// Zero was correct for v5, where the value was a bare tagged pointer with no
/// header at all. The arc-value-header work reintroduced one: a
/// `triomphe::Arc<ValueHeader<K>>` is an 8-byte strong count in front of a
/// 24-byte header, and jemalloc's 32-byte class holds it exactly. Measured as
/// the residue of 136.00 B/object less the 40-byte row and the 64-byte value.
///
/// Leaving it at zero under-charged every object by 32 bytes. Note the row
/// constant above was simultaneously 40 too high, so the two errors largely
/// cancelled and the net was an 8 B/object OVER-charge -- which is exactly why
/// neither showed up as a crash or an obvious mis-sizing.
///
/// ## Why this is gated
///
/// It names a SEPARATE DRAM allocation holding the value header, and what that
/// allocation holds depends on the layout: under `thin_header` it is half the
/// size (the constant below). A third layout, the fused one, kept the count,
/// the key, the length and the expiry inside the ONE item that tiers, with no
/// separate header, so the term was zero there; that layout has been removed.
///
/// MEASURED, in the harness rather than inferred: the DashMap control fit
/// 136.0 B/object in BOTH the split and the fused layouts, of which 40 is the
/// row and 96 the value item. Split, that 96 is `nallocx(64) = 64` plus this
/// 32. Fused, it was a single `nallocx(24 + 64) = nallocx(88) = 96` with no
/// second allocation anywhere in it.
#[cfg(not(feature = "thin_header"))]
const VALUE_ALLOCATION_OVERHEAD: ObjectSize = 32;

/// Under `thin_header` the separate DRAM allocation exists and is half the
/// split layout's: `triomphe::Arc<ValueHeader<K>>` is an 8-byte strong count
/// in front of ONE 8-byte tagged item pointer, 16 bytes, jemalloc's 16-byte
/// class exactly. The length, the expiry and the key moved into the item and
/// are charged through `resident_object_bytes` instead. Held to the allocator
/// by `an_object_costs_what_the_accounting_says_it_costs`.
#[cfg(feature = "thin_header")]
const VALUE_ALLOCATION_OVERHEAD: ObjectSize = 16;

/// Per-object DRAM cost of the object map (`DashMap<HashedKey, Object>`):
/// the `(u64, Object{key, value word, len, expiry})` pair -- 8 + 24 = 32 bytes
/// -- plus hashbrown's control byte and load-factor slack.
///
/// **80**, MEASURED. `measure_object_map_point`, jemalloc `stats.allocated`,
/// one process per point, 2^20..2^23 at MEASURE_VALUE = 64: 144.0000 B/object
/// at R2 = 1.000000, of which 64 is the size-class-rounded value. Sweeping the
/// value size at n = 2^22 pins it independently -- the non-value remainder is
/// 79.998 at 16, 32, 64 AND 128 byte values, i.e. a container cost rather than
/// a mis-attributed value cost.
///
/// Was 96, and 96 was already 16 too high BEFORE v5: the same harness measured
/// 80 on the base commit. So the drop from 96 to 80 is a correction, not a
/// saving -- the v5 saving is the separate 32-byte value allocation
/// disappearing, which is `VALUE_ALLOCATION_OVERHEAD` above. Both errors ran in
/// the same direction (over-charging, so the cache held fewer objects than its
/// budget allowed), which is why neither showed up as a crash.
///
/// The row did not change size in v5: `Object` was 24 bytes with a `Shared`
/// handle and is 24 bytes with a `TieredValue` and a `len`. That the measured
/// row is unchanged at 80 is therefore a check on the layout claim, not a
/// coincidence.
// Not cfg-gated: every design allocates one object-map row per object, tiered
// or not, so this term is charged in get_policy_overhead for non-hybrid
// policies as well.
/// MEASURED on this branch: `measure_object_map_point`, release, one process
/// per point, value 64, converging 135.9963 / 135.9985 / 135.9992 B/object at
/// 2^21..2^23. That total decomposes as this row plus a 64-byte value plus a
/// 32-byte header allocation, so the row is 40.
///
/// It was 80, measured against a THIRTY-THREE byte row: v5 stored a 24-byte
/// `Object` inline in the map, so the pair was 8 + 24 plus a control byte. The
/// entry is now 8 (hashed key) + 8 (one handle) + control, and the measured
/// allocation halved with it -- which is also the evidence that the 80 was
/// tracking content rather than shard-doubling slack.
const OBJECT_MAP_ENTRY_OVERHEAD: ObjectSize = 40;


/// Bytes jemalloc will actually commit for a value of this requested size.
///
/// Asked of the allocator rather than estimated. This is what Redis does --
/// `used_memory` is the sum of `malloc_usable_size` per allocation, never a
/// scaled request -- and `nallocx` answers the same question without
/// allocating, so it costs a size-class lookup.
///
/// A flat factor was tried first and was wrong in both directions. Measured
/// against jemalloc's real classes: 8 -> 8 (1.000x), 24 -> 32 (1.333x),
/// 194 -> 224 (1.155x), 1024 -> 1024 (1.000x), 4096 -> 4096 (1.000x). The
/// ratio is 1.0 for anything landing on a class and up to 1.33 just above one;
/// no constant models that.
///
/// It also mis-attributed process-level waste. A live run showed the slow tier
/// accounting 8.24 GiB against 9.90 GiB resident on node1 -- 20% -- so 1.20 was
/// adopted. But size-class rounding over these corpora's object-size mix is
/// only **1.081x**; the rest is jemalloc's retained dirty pages and arena
/// fragmentation, which are properties of the process, not of an object.
/// Charging them per object over-reserved every small value by ~11%.
///
/// Process-level waste belongs in a reported ratio, the way Redis reports
/// `mem_fragmentation_ratio`, not inside a per-object budget -- and that is how
/// it is reported: the benchmark's `mem_fragmentation_ratio` column is RSS over
/// `used_size`, and `paper_cache::jemalloc_stats()` samples the allocator's own
/// process-wide figures (`stats.allocated`, `active`, `resident`, `mapped`,
/// `retained`, through `mallctl`) for a diagnosis. Neither feeds a decision.
#[cfg(feature = "numa_jemalloc")]
pub(crate) fn resident_value_bytes(requested: ObjectSize) -> ObjectSize {
	// SAFETY: `nallocx` is a pure size-class computation. It allocates
	// nothing, dereferences nothing, and cannot fail for a non-zero size.
	match requested {
		0 => 0,
		n => unsafe { tikv_jemalloc_sys::nallocx(n as usize, 0) as ObjectSize },
	}
}

/// Without jemalloc there is no allocator to ask, so the request stands.
/// Estimating here would reintroduce exactly the error described above.
#[cfg(not(feature = "numa_jemalloc"))]
pub(crate) fn resident_value_bytes(requested: ObjectSize) -> ObjectSize {
	requested
}

/// The header bytes that share the VALUE'S OWN allocation: `bytes_offset::<K>()`
/// under `thin_header`, nothing under the split layout.
///
/// Under `thin_header` that prefix is the length, the expiry and the key; the
/// count and the item pointer are the separate 16-byte DRAM allocation charged
/// as `VALUE_ALLOCATION_OVERHEAD`. So both terms are non-zero there, and each
/// names a different allocation: no byte is counted twice or not at all.
///
/// Under the split layout the header is its own `Arc` allocation, charged as
/// `VALUE_ALLOCATION_OVERHEAD`, and this is zero.
///
/// Only where something asks by type -- `phys::value_charge` (the tiered
/// builds) and the tests' fixtures; the accounting asks the object or its key.
#[cfg(all(feature = "thin_header", any(test, feature = "hybrid_cache_common")))]
#[inline]
pub(crate) fn value_header_bytes<K>() -> ObjectSize {
	crate::value::bytes_offset::<K>() as ObjectSize
}

#[cfg(all(not(feature = "thin_header"), any(test, feature = "hybrid_cache_common")))]
#[inline]
pub(crate) fn value_header_bytes<K>() -> ObjectSize {
	0
}

/// What jemalloc would commit for the item of a `value_len`-byte value whose
/// key the item holds as a `K`: the value behind `value_header_bytes::<K>()`.
///
/// A per-TYPE figure, so it is exact only for that shape of key. Under
/// `thin_header` a key held as bytes (`String`, `Vec<u8>`, `Box<[u8]>`) makes
/// the prefix as long as the key, and the accounting asks the object
/// (`resident_item_bytes`) or, before there is one, the key
/// (`resident_item_bytes_for`). This survives for what has no key to ask --
/// `phys::value_charge` -- and for the fixtures that build an object of a
/// given cost from a length, all of which key by `u64`.
#[cfg(any(test, feature = "hybrid_cache_common"))]
pub(crate) fn resident_object_bytes<K>(value_len: ObjectSize) -> ObjectSize {
	resident_value_bytes(value_len.saturating_add(value_header_bytes::<K>()))
}

/// Bytes jemalloc commits for an object's OWN allocation -- the one that tiers.
///
/// THE ONLY WAY to ask that question. `base_size` and `Slot::migrating` both
/// call this rather than rounding a length themselves, because they were
/// rounding DIFFERENT things from the same `data_size()` and a third caller
/// would have got it wrong the same way.
///
/// `Object::data_size()` is the value's LENGTH and nothing else. Under the
/// split layout that is also the whole of the value's own allocation, so
/// `nallocx(len)` is right. Under `thin_header` it is not: the length, expiry
/// and key sit in front of the bytes in the SAME allocation, so the request is
/// the item's prefix plus `len` -- 16 bytes more for a `u64` key, and a WHOLE
/// SIZE CLASS more whenever the value alone was landing on a class boundary.
/// `nallocx(4096)` is 4096 and `nallocx(4112)` is 5120: rounding the length
/// alone would under-charge by 16 bytes in the ordinary case and by 1024 in the
/// common one, because powers of two are common in both traces and synthetic
/// workloads.
///
/// It is asked of the OBJECT, not of a length and a key type, because under
/// `thin_header` a key held as bytes is in the item, so the item's prefix is
/// as long as the key and differs object to object. The value reports its own
/// prefix (`item_prefix_bytes`): nothing for the split layout,
/// `bytes_offset::<K>()` for a key held as a `K`, and the offset past the
/// key's bytes for a key held as bytes.
pub(crate) fn resident_item_bytes<K, V>(object: &Object<K, V>) -> ObjectSize {
	resident_value_bytes(
		object.data_size().saturating_add(prefix_size(object.value().item_prefix_bytes())),
	)
}

/// `resident_item_bytes` of the object a set WILL build, from its key and its
/// value's length, before there is an object: how the admission path sizes a
/// set (`OverheadManager::base_size_for`, `phys::value_charge_for`). The
/// prefix comes from the key's type the way the built value will report it
/// (`TieredValue::item_prefix_bytes_for`), so the two are equal --
/// `base_size_for_equals_base_size` holds it.
pub(crate) fn resident_item_bytes_for<K: 'static>(key: &K, value_len: ObjectSize) -> ObjectSize {
	resident_value_bytes(
		value_len.saturating_add(prefix_size(crate::value::TieredValue::<K>::item_prefix_bytes_for(key))),
	)
}

/// A prefix length as an `ObjectSize`, saturating: a key so long its prefix
/// does not fit makes an item no set can hold, which the size checks refuse.
#[inline]
fn prefix_size(prefix: usize) -> ObjectSize {
	ObjectSize::try_from(prefix).unwrap_or(ObjectSize::MAX)
}

// resident_factor() / DRAM_OVERHEAD_RESIDENT_FACTOR were deleted: INERT, by
// their own doc. Every term they scaled now comes from size-class-rounded
// jemalloc figures, so applying the factor would charge the rounding twice.

#[cfg(feature = "hybrid_cache_common")]
pub fn get_hybrid_dram_shared_overhead(policy: &PaperPolicy) -> ObjectSize {
	#[cfg(all(test, feature = "hybrid_cache_common"))]
	if let Some(omega) = test_overheads::omega() {
		return omega;
	}

	// Test support: the tier-mechanics integration tests choreograph
	// promotions and demotions with fast-tier budgets of tens of bytes,
	// where the ~75 B/object metadata reservation below exceeds the whole
	// budget and no promotion can ever fit. Those tests predate the
	// reservation and test policy mechanics, not DRAM accounting, so
	// `PAPER_DISABLE_SHARED_OVERHEAD=1` restores their value-only
	// semantics, and those binaries set it from their own
	// `ensure_pmem_allocator_warm()`.
	//
	// The reservation itself is therefore covered by separate test binaries
	// -- the tests/*_shared_overhead.rs family -- which
	// never set the variable, so every cache they build gets the production
	// default. They are separate PROCESSES on purpose: this is read at every
	// cache construction, so a test flipping the variable back would race
	// every sibling test constructing a cache on another thread.
	if std::env::var_os("PAPER_DISABLE_SHARED_OVERHEAD").is_some_and(|v| v == "1") {
		return 0;
	}

	// Under `merged_object_store` there is no eviction stack to reserve DRAM
	// for, and the map row is the merged store's slot -- so the whole
	// per-policy match below names structures that do not exist in this build.
	// The merged store reserves its OWN structure instead: the measured
	// slot+bucket cost plus the (now zero) value-allocation term, all of which
	// is DRAM-resident in either tier because only the value bytes migrate.
	//
	// Without this the fast tier is charged the split design's ~192 B/object
	// for metadata it does not have, so it demotes far earlier than it should
	// and the merged build would be measured on a smaller effective fast tier
	// than the baseline it is being compared against.
	//
	// 46 for the 40-byte slot, measured at 45.2966 B/object structural, the
	// same in both layouts it was measured in -- see
	// `MERGED_STORE_STRUCTURE_OVERHEAD`. The reservation is per LIVE object and
	// `MergedStore::settle_tier` takes it off the fast budget before it drains,
	// so this number directly sets how many objects fit in DRAM. It was
	// 62 + 32 = 94 in every build; it is now 46 + 32 = 78 split. Under
	// `thin_header` it is 46 + 16 = 62: the header holds only the count and the
	// item pointer.
	#[cfg(feature = "merged_object_store")]
	{
		let _ = policy;
		return MERGED_STORE_STRUCTURE_OVERHEAD + VALUE_ALLOCATION_OVERHEAD;
	}

	#[cfg(feature = "merged_object_store")]
	#[allow(unreachable_code)]
	{
		unreachable!()
	}

	#[allow(unused_mut)]
	let mut overhead: ObjectSize = 0;

	// Eviction stacks live in DRAM unless `eviction_stacks_pmem` relocates them.
	//
	// Selected by a runtime call, not by `cfg`: nothing about a policy's constant
	// requires its stack module to be compiled, and gating them meant a build
	// without that feature silently contributed 0 -- no error, no warning, no
	// failing test. A binary built with only one hybrid feature once charged every
	// other policy the value and map terms alone, handing each a larger effective
	// fast tier than it should have had, for a whole sweep, before anyone noticed.
	// A flat policy has no tiers and reserves nothing.
	#[cfg(not(feature = "eviction_stacks_pmem"))]
	if policy.is_hybrid() {
		overhead += stack_dram_overhead(policy);
	}

	// The value's own allocation costs a DRAM-resident refcounted header
	// regardless of which tier the bytes themselves occupy -- 32 bytes of
	// `Arc<ValueHeader<K>>` under the split layout, 16 under `thin_header`,
	// whose header holds only the count and the item pointer (the length, the
	// expiry and the key are in the item, charged through
	// `resident_object_bytes` against the tier the item is actually in). Kept
	// as a named term so that this reservation and `get_policy_overhead` can
	// be compared line for line.
	overhead += VALUE_ALLOCATION_OVERHEAD;

	// The object map lives in DRAM.
	//
	// MEASURED at 80.0 B/object for the `DashMap` shape: the whole map
	// measures 144.0 at a 64-byte value (R2 = 1.000000), and the non-value
	// remainder is 79.998 at 16-, 32-, 64- AND 128-byte values, so it is a
	// container cost rather than a mis-attributed value cost.
	overhead += OBJECT_MAP_ENTRY_OVERHEAD;

	// No resident factor. Every term above is now MEASURED from jemalloc
	// `stats.allocated`, which is the size-class-rounded usable figure -- the
	// same quantity Redis reports as `used_memory`. The factor modelled
	// "requested -> resident", so applying it to an already-rounded
	// measurement charges the rounding twice.
	//
	// (The comment this replaces claimed the eviction-stack term "comes from
	// RSS". It did once; the measurement was moved to `stats.allocated`, and
	// the comment was not.)
	overhead
}

#[cfg(all(test, feature = "hybrid_cache_common"))]
mod shared_overhead_is_feature_independent {
	use super::*;

	/// A policy's DRAM term must not depend on which *other* stack modules were
	/// compiled.
	///
	/// A regression test for a silent, whole-sweep measurement error: the terms
	/// were once `cfg`-gated per policy, so a binary built with a single hybrid
	/// feature -- how the benchmark was configured -- charged every other
	/// policy the value and map terms alone. Runs under whatever feature set the
	/// build has, so it fails if anyone reintroduces the gating.
	#[test]
	fn every_hybrid_policy_keeps_its_own_term() {
		if std::env::var_os("PAPER_DISABLE_SHARED_OVERHEAD").is_some() {
			return; // the escape hatch zeroes everything by design
		}

		let reserved = [
			("lru", get_hybrid_dram_shared_overhead(&PaperPolicy::LruCompactHybrid)),
			("lfu", get_hybrid_dram_shared_overhead(&PaperPolicy::LfuCompactHybrid)),
			("fifo", get_hybrid_dram_shared_overhead(&PaperPolicy::FifoCompactHybrid)),
			("s3-fifo", get_hybrid_dram_shared_overhead(&PaperPolicy::S3FifoCompactHybrid(0.1))),
		];

		// The value with NO eviction-stack term -- what a gated-out policy
		// collapses to.
		#[allow(unused_variables)] // named by two of the three arms below
		let no_stack_term = VALUE_ALLOCATION_OVERHEAD + OBJECT_MAP_ENTRY_OVERHEAD;

		for (name, got) in reserved {
			// Under `eviction_stacks_pmem` the stacks live in CXL, so they are
			// deliberately absent from the FAST-TIER reservation while still
			// counting toward `get_policy_overhead`: every policy collapses to
			// `no_stack_term` ON PURPOSE.
			#[cfg(all(feature = "eviction_stacks_pmem", not(feature = "merged_object_store")))]
			assert_eq!(
				got, no_stack_term,
				"{name} still reserves fast-tier DRAM for an eviction stack that lives in CXL",
			);

			// Under `merged_object_store` the object map IS the eviction
			// structure: every policy reserves the merged store's own measured
			// structural cost and nothing else.
			#[cfg(feature = "merged_object_store")]
			assert_eq!(
				got,
				MERGED_STORE_STRUCTURE_OVERHEAD + VALUE_ALLOCATION_OVERHEAD,
				"{name} reserves fast-tier DRAM for a split-design eviction stack this build does not have",
			);

			#[cfg(not(any(feature = "eviction_stacks_pmem", feature = "merged_object_store")))]
			assert_eq!(
				got,
				no_stack_term + ARENA_STACK_DRAM_OVERHEAD,
				"{name} lost its eviction-stack term -- the per-policy cfg gating is back",
			);
		}

		// The term is the arena node and the index, each pinned where it is
		// built -- 32: `const _: () = assert!(size_of::<ArenaSlot<NodePayload>>()
		// == 32)` in `arena_queue_set`; 8: `the_index_costs_eight_bytes_per_
		// object_at_a_power_of_two_population` in `arena_index`'s consumers. A
		// structure that moves breaks its own check first and this one second:
		// the fix is to re-run `measure_one_point`, not to adjust one side to fit
		// the other.
		const ARENA_NODE: ObjectSize = 32;
		const KEYLESS_INDEX_PER_OBJECT: ObjectSize = 8;

		assert_eq!(ARENA_STACK_DRAM_OVERHEAD, ARENA_NODE + KEYLESS_INDEX_PER_OBJECT);
	}
}

#[cfg(all(test, feature = "hybrid_cache_common"))]
mod value_counted_at_its_allocated_size {
	use std::sync::Arc;

	use super::*;
	use crate::{policy::PaperPolicy, status::AtomicStatus};

	/// `base_size` must report what the allocator holds, not what was asked
	/// for. Before this, only metadata carried a resident factor; a live run
	/// showed the slow tier accounting 8.24 GiB while node1 held 9.90 GiB.
	///
	/// Every existing test derives its expectations from `base_size` itself,
	/// so all of them stayed green through this change -- good design on their
	/// part, but it means not one of them would have caught the omission.
	#[test]
	fn the_value_is_counted_at_its_allocated_size() {
		let status: crate::StatusRef = Arc::new(
			AtomicStatus::new(
				1_000_000,
				&[PaperPolicy::LruCompactHybrid],
				PaperPolicy::LruCompactHybrid,
			)
			.expect("status"),
		);
		let manager = OverheadManager::new(&status);
		let object = Object::<u32, crate::BufferDRAM>::new(0u32, &vec![0u8; 1000], None);

		let got = manager.base_size(&object);
		let key = object.key_size();
		let expiry = mem::size_of::<crate::object::ExpireTime>() as ObjectSize;
		// `data_size` is now the value's length and nothing else -- there is no
		// fat pointer left to count, in any shape.
		let raw = object.data_size();
		// The object's OWN allocation, which under `thin_header` is the item's
		// header and the bytes together. Asserting `resident_value_bytes(raw)`
		// here would re-state the bug: it is the value BYTES, not the allocation.
		let scaled = resident_object_bytes::<u32>(raw);

		assert_eq!(
			got, key + scaled + expiry,
			"the value is counted at jemalloc's committed size for the whole \
			 item; the key and expiry are left alone, since shared_overhead \
			 covers them",
		);

		assert!(
			scaled >= resident_value_bytes(raw),
			"the item's allocation cannot be smaller than the bytes inside it: \
			 {scaled} < {}",
			resident_value_bytes(raw),
		);

		assert!(
			got > key + raw + expiry,
			"a {raw}-byte value must account for more than {raw} bytes: got {got}, \
			 unscaled would be {}",
			key + raw + expiry,
		);
	}
}

/// What jemalloc would actually commit for a given request, versus the flat
/// factor this module estimates with.
///
/// Redis answers this question by calling `malloc_usable_size` per allocation;
/// memcached sidesteps it by budgeting slab pages, so its rounding waste is
/// explicit. This crate estimates instead, which is why the value side was out
/// by 20% until it was measured. `nallocx` gives the exact size class for a
/// request without allocating -- the same information Redis uses.
///
///   cargo +nightly test --release --features lru_compact_hybrid_cache --lib \
///       what_jemalloc -- --ignored --nocapture
#[cfg(all(test, feature = "numa_jemalloc"))]
mod what_jemalloc_actually_rounds_to {
	#[test]
	#[ignore]
	fn what_jemalloc_rounds_to() {
		use tikv_jemalloc_sys::nallocx;

		println!("NX  requested   jemalloc    ratio");
		let mut worst: f64 = 0.0;
		for s in [8usize, 13, 24, 29, 64, 100, 130, 194, 225, 500, 1000,
		          1024, 1497, 1500, 4096, 5607, 8392, 60103] {
			let a = unsafe { nallocx(s, 0) };
			let r = a as f64 / s as f64;
			if r > worst { worst = r; }
			println!("NX  {:>9} {:>10}   {:.3}x", s, a, r);
		}
		println!("NX  worst single-size ratio: {:.3}x", worst);

		// Weighted by the object-size mix actually measured in these corpora:
		// Meta kvcache medians are 13-132 B, Twitter clusters 77-8392 B.
		let mix: [(usize, f64); 7] = [
			(13, 25.0), (29, 20.0), (77, 15.0), (132, 15.0),
			(194, 10.0), (1497, 10.0), (5607, 5.0),
		];
		let (mut req, mut act) = (0.0, 0.0);
		for (s, w) in mix {
			req += s as f64 * w;
			act += unsafe { nallocx(s, 0) } as f64 * w;
		}
		println!("NX  trace-weighted mix: {:.3}x  (the flat estimate in use: 1.20)",
			act / req);
	}
}

/// What the cache charges for one object's own memory, against what jemalloc
/// actually handed out for it.
///
/// Every other test in this module derives its expectation from the same
/// arithmetic it is checking, so none of them could catch a term that names the
/// wrong allocation. This one asks the allocator.
///
/// The claim is ONE identity, and it holds in both value layouts:
///
/// ```text
///   allocated per object  ==  resident_object_bytes::<K>(len)
///                             + VALUE_ALLOCATION_OVERHEAD
/// ```
///
/// Split, the right-hand side is `nallocx(len) + 32`: the bytes, plus the
/// `Arc<ValueHeader<K>>` that owns them. Thin (`thin_header`), it is
/// `nallocx(bytes_offset::<K>() + len) + 16`: the item, plus the
/// count-and-pointer header in DRAM.
/// A build that gates either term wrongly fails here, and so does one that
/// rounds the value's LENGTH when the allocation is the length plus a header --
/// which is what both sides of this identity were doing before.
///
/// The sizes are chosen so the header crosses a size class in some and not in
/// others: `nallocx(4096)` is 4096 but `nallocx(4112)` is 5120, so under
/// `thin_header` a 4 KiB value's item is a whole KiB larger than its bytes.
#[cfg(all(test, feature = "numa_jemalloc"))]
mod the_charge_matches_the_allocator {
	use super::*;
	use crate::object::Object;

	/// This THREAD's cumulative allocated and deallocated bytes, as jemalloc
	/// counts them -- `thread.allocated` / `thread.deallocated`.
	///
	/// Deliberately not `stats.allocated`. That is process-wide, and the test
	/// harness runs this beside two hundred other tests on other threads, whose
	/// allocations and frees land in the same number: the first version of this
	/// test read a delta of ZERO because sibling threads happened to free more
	/// than this one allocated. Per-thread counters make the measurement
	/// independent of what else the binary is doing, so the test needs no
	/// `--test-threads=1` and cannot become flaky when one is added.
	fn thread_counter(name: &core::ffi::CStr) -> u64 {
		unsafe {
			let mut v: u64 = 0;
			let mut len = core::mem::size_of::<u64>();

			let rc = tikv_jemalloc_sys::mallctl(
				name.as_ptr(),
				&mut v as *mut u64 as *mut core::ffi::c_void,
				&mut len,
				core::ptr::null_mut(),
				0,
			);

			assert_eq!(rc, 0, "{name:?} unavailable");
			v
		}
	}

	/// Bytes this thread has allocated and not yet freed.
	fn thread_live() -> i64 {
		thread_counter(c"thread.allocated") as i64
			- thread_counter(c"thread.deallocated") as i64
	}

	#[test]
	fn an_object_costs_what_the_accounting_says_it_costs() {
		const N: usize = 4096;

		for len in [64u32, 100, 1000, 1024, 4096, 8192] {
			let payload = vec![0u8; len as usize];

			// Reserved BEFORE the baseline so the handles' own `Vec` is not in
			// the delta, and a fresh `Vec` per size so no growth lands inside.
			let mut held: Vec<Object<u64, crate::BufferDRAM>> = Vec::with_capacity(N);

			let base = thread_live();

			for i in 0..N {
				held.push(Object::new(i as u64, &payload, None));
			}

			let delta = (thread_live() - base).max(0) as u64;
			core::hint::black_box(&held);

			let charged = resident_object_bytes::<u64>(len) as u64
				+ VALUE_ALLOCATION_OVERHEAD as u64;
			let measured = delta / N as u64;

			assert_eq!(
				measured, charged,
				"a {len}-byte value costs {measured} B/object from the \
				 allocator but is charged {charged}: \
				 resident_object_bytes = {}, VALUE_ALLOCATION_OVERHEAD = {}",
				resident_object_bytes::<u64>(len),
				VALUE_ALLOCATION_OVERHEAD,
			);

			drop(held);
		}
	}

	/// S5a: the DRAM value header M counts per live object
	/// (`value::dram_header_bytes`) is what the allocator hands out for one,
	/// in every layout -- an object costs its item and exactly that -- and it
	/// is the size class the per-object model names
	/// (`VALUE_ALLOCATION_OVERHEAD`: 32 split, 16 `thin_header`).
	#[test]
	fn the_dram_header_m_counts_is_what_the_allocator_holds_per_object() {
		const N: usize = 2048;

		let header = crate::value::dram_header_bytes::<u64>();

		assert_eq!(
			header,
			VALUE_ALLOCATION_OVERHEAD as u64,
			"M's header ({header} B) and the model's constant disagree",
		);

		for len in [64u32, 100, 1000, 4096] {
			let payload = vec![0u8; len as usize];
			let mut held: Vec<Object<u64, crate::BufferDRAM>> = Vec::with_capacity(N);

			let base = thread_live();

			for i in 0..N {
				held.push(Object::new(i as u64, &payload, None));
			}

			let per_object = (thread_live() - base).max(0) as u64 / N as u64;
			core::hint::black_box(&held);

			assert_eq!(
				per_object - resident_object_bytes::<u64>(len) as u64,
				header,
				"a {len}-byte value: {per_object} B/object from the allocator, of which the \
				 item is {} -- the rest is the header M counts as {header}",
				resident_object_bytes::<u64>(len),
			);

			drop(held);
		}
	}

	/// The same identity for a `String`-keyed object under `thin_header`, where
	/// the key's bytes are IN the item: one allocation of
	/// `nallocx(align8(12 + key_len) + len)` plus the 16-byte DRAM header, and
	/// nothing for the key anywhere else. Each key is built inside the window,
	/// so its own buffer -- allocated, then freed once the item holds a copy --
	/// nets to zero, as a server's request buffer would. The key lengths are
	/// the ones the prefix's rounding and the size classes treat differently.
	#[test]
	#[cfg(feature = "thin_header")]
	fn a_string_keyed_object_costs_what_the_accounting_says_it_costs() {
		const N: usize = 4096;

		for key_len in [0usize, 1, 3, 4, 5, 11, 12, 13, 19, 43, 44, 250] {
			for len in [64u32, 100, 1000, 4096] {
				let payload = vec![0u8; len as usize];
				let mut held: Vec<Object<String, crate::BufferDRAM>> = Vec::with_capacity(N);

				let base = thread_live();

				for _ in 0..N {
					held.push(Object::new("k".repeat(key_len), &payload, None));
				}

				let delta = (thread_live() - base).max(0) as u64;
				core::hint::black_box(&held);

				let item = resident_item_bytes(&held[0]);
				let prefix = (12 + key_len).next_multiple_of(8) as ObjectSize;

				assert_eq!(
					item,
					resident_value_bytes(prefix + len),
					"key {key_len} B, value {len} B: the item is its header, key and value",
				);
				assert_eq!(
					item,
					resident_item_bytes_for(&"k".repeat(key_len), len),
					"key {key_len} B, value {len} B: the object and the key it is built from agree",
				);

				let charged = item as u64 + VALUE_ALLOCATION_OVERHEAD as u64;
				let measured = delta / N as u64;

				assert_eq!(
					measured, charged,
					"a {key_len}-byte key and a {len}-byte value cost {measured} B/object \
					 from the allocator but are charged {charged}",
				);

				drop(held);
			}
		}
	}

	/// The accessor must never report LESS than the bytes it contains, in any
	/// build. Cheap, and it is the invariant that would have failed loudest had
	/// the header been subtracted instead of added.
	#[test]
	fn the_item_is_never_smaller_than_its_value() {
		for len in [0u32, 1, 8, 63, 64, 65, 4095, 4096, 4097, 65536] {
			assert!(
				resident_object_bytes::<u64>(len) >= resident_value_bytes(len),
				"len {len}: item {} < bytes {}",
				resident_object_bytes::<u64>(len),
				resident_value_bytes(len),
			);
		}
	}
}

