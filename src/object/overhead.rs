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
		// The value is counted as the bytes jemalloc actually commits, asked of
		// the allocator rather than estimated -- see `resident_value_bytes`.
		// Before this it was counted as the bytes *requested*, so a tier sized
		// to its accounted bytes overran.
		//
		// The key and expiry are deliberately NOT scaled here: they are inside
		// `shared_overhead`, which applies its own factor to them already, and
		// scaling them twice is exactly the double-charge this module has been
		// untangling elsewhere.
		let value = resident_value_bytes(object.data_size());
		let mut total_size = object.key_size()
			+ value
			+ mem::size_of::<crate::object::ExpireTime>() as ObjectSize;

		if object.expiry().is_some() {
			total_size += get_ttl_overhead();
		}

		total_size
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

/// 40 + 32 - 12 = **60**. Was 80 + 0 - 12 = 68, and 96 + 32 - 12 = 116 before that.
const OBJECT_MAP_ROW_OVERHEAD: ObjectSize =
	OBJECT_MAP_ENTRY_OVERHEAD + VALUE_ALLOCATION_OVERHEAD - DOUBLE_COUNTED_IN_BASE_SIZE;

/// The merged store's structural cost per object, replacing BOTH the map row
/// and the eviction stack.
///
/// RE-MEASURED for the v3 slot -- jemalloc `stats.allocated`, ONE point per
/// process, `MSTORE_VALUE=64` (a jemalloc class exactly, so no rounding
/// correction), least squares over 2^20..2^23:
///
/// ```text
///   MergedStore  125.3041 B/object   R^2 = 0.999999
///   less the value                    64
///                                   ------
///   structural                       61.30
/// ```
///
/// It was 73, measured at 165.35 - 64 = 69.35 on a slab that grew by `Vec`
/// reallocation and an index that doubled. Where the 8 B/object went:
///
/// ```text
///   slab   56 B/slot   x 1.005 fill = 56.27   (was 56 x 1.101 = 61.7)
///   index   4 B/bucket x 1.313 fill =  5.25   (was  4 x 1.398 =  5.6)
///                                    -----
///                                    61.52
/// ```
///
/// The slab fill is 1.005 because a chunked slab wastes at most one partly
/// filled 4096-slot chunk per shard -- bounded, where a growth factor is
/// proportional. The index fill is the bucket `Vec`'s own doubling: linear
/// hashing keeps `buckets.len()` equal to the live count, so the only slack
/// left is the `Vec` sitting mid-double, 1x to 2x, and the shards straddle a
/// power of two at every scale.
///
/// Set to **62** rather than 61: the slope is asymptotic, and every measured
/// point above 2^20 sits between 61.5 and 62.1 B/object structural (the
/// mid-round points 3 x 2^20 and 6 x 2^20 included, where the bucket `Vec` is
/// furthest from full). Rounding up keeps the charge on the conservative side
/// -- the cache holds slightly fewer objects than the budget allows, rather
/// than overrunning it.
///
/// Against this, the split design costs 144.00 (DashMap alone, re-measured on
/// this same tree, R^2 = 1.000000, less the same 64-byte value: 80.00) + 72
/// (measured `LruCompactHybridStack`) = 152 B/object of structure, against
/// 61.3 here.
#[cfg(feature = "merged_object_store")]
const MERGED_STORE_STRUCTURE_OVERHEAD: ObjectSize = 62;

/// Under `merged_object_store` the object map IS the eviction stack, so the
/// per-policy stack terms below do not apply at all -- there is no second
/// structure to charge for. Charging them anyway is what left the measured
/// saving entirely unrealized: `used_size` kept billing every object for a
/// stack row that no longer exists, so the cache held fewer objects than its
/// budget allowed and the saving showed up nowhere.
///
/// Same shape as `OBJECT_MAP_ROW_OVERHEAD`: the slot embeds the `Object`,
/// hence contains the key and expiry that `base_size` counts separately, so
/// those 12 bytes come back off. 62 + 0 - 12 = **50**, from 73 + 0 - 12 = 61.
///
/// The two functions agree here in a way they do not for the split designs:
/// both name `MERGED_STORE_STRUCTURE_OVERHEAD`, and they differ by exactly
/// `DOUBLE_COUNTED_IN_BASE_SIZE` and nothing else.
#[cfg(feature = "merged_object_store")]
pub fn get_policy_overhead(_policy: &PaperPolicy) -> ObjectSize {
	MERGED_STORE_STRUCTURE_OVERHEAD + VALUE_ALLOCATION_OVERHEAD
		- DOUBLE_COUNTED_IN_BASE_SIZE
}

/// MEASURED_STACK marker: the per-policy terms below are measured, not
/// hand-counted -- jemalloc `stats.allocated`, ONE process per point,
/// 2^20..2^23, least squares. The previous values were, per this module's own
/// harness comment, "just rough estimates", and every compact variant carried a
/// flat `16 + 24` "written by the registration helper and never checked against
/// anything". Every one understated its stack by 18-100%.
///
/// RE-MEASURED with `measure_one_point` for all 43 hybrid policies of that
/// sweep, not a sample of them: every compact family fits 71.995-71.998
/// B/object, R^2 = 1.000000 on all 43. The split families fit 112.0003 (168.0003
/// for the split LFU) and have since been REMOVED -- each was proven
/// behaviourally identical to its compact twin by a differential test, so the
/// tree keeps only the 72 B/object shape, and the 24 policies that remain here
/// are all compact. The
/// constants below stand unchanged; what was wrong was the hand count in
/// `get_policy_overhead`, which is why that function now names these constants
/// instead of repeating a number.
///
/// The FLAT arms were re-measured in the same pass and are all correct: lru,
/// fifo, clock, sieve, mru, 2q, arc and s3-fifo (compact and not) each fit
/// their hand-written term to within 0.01 B/object, R^2 = 1.000000 on all
/// seventeen. Only the hybrid arms were wrong, which is why only they change.
///
/// (The note this replaces cited results/measured_stack_allocation.txt for the
/// seventeen. That file is not in the tree -- there is no results/ directory --
/// so the provenance now lives here, where it cannot go missing.)

/// EVERY arm carries `OBJECT_MAP_ROW_OVERHEAD`, including the hybrid ones.
///
/// The hybrid arms did not, until now: they returned their eviction-stack term
/// and nothing else, so a hybrid design was charged ~40 B/object against a real
/// cost near 200. `used_size` is what `max_size` bounds, so the effect was that
/// `max_size` did not bound memory for any tiered design -- the cache admitted
/// objects until its accounted total hit the cap while its actual DRAM footprint
/// ran ~4x that per object of metadata. The flat arms have always included the
/// term; this only makes the hybrids agree with them.
///
/// The hybrid arms name the MEASURED eviction-stack constant. They used to
/// carry the registration helper's flat `16 + 24`, and THAT is where the two
/// functions in this module disagreed: for every compact hybrid
/// `get_hybrid_dram_shared_overhead` reserved the measured 72 while this
/// function charged 40, so `used_size` under-billed every compact hybrid by
/// 32 B/object and `max_size` let in ~30% more metadata than it meant to. The
/// split hybrids, since removed, were out by 25-27, and the split LFU by 55.
///
/// The constants were not the problem -- re-measured with `measure_one_point`
/// across all 43 hybrid policies of that sweep, they are right to four decimal
/// places (see
/// the MEASURED_STACK note above). The hand counts were. Naming the constant
/// rather than restating its value is the fix that lasts: the two tables can no
/// longer drift, and `the_two_overhead_tables_agree_on_every_hybrid` asserts
/// that what is left between them is exactly `DOUBLE_COUNTED_IN_BASE_SIZE`.
///
/// The two functions now name the same three quantities -- the stack, the map
/// row, and the (zero) value allocation -- and they differ in one deliberate
/// way: `get_hybrid_dram_shared_overhead` does NOT subtract
/// `DOUBLE_COUNTED_IN_BASE_SIZE`, because it is a fast-tier RESERVATION rather
/// than an addition on top of `base_size`, so it has nothing to double-count
/// against.
#[cfg(not(feature = "merged_object_store"))]
pub fn get_policy_overhead(policy: &PaperPolicy) -> ObjectSize {
	// Each arm is <this policy's eviction-stack cost> + the object-map row.
	// The stack terms are measured (see the MEASURED_STACK note above); the row
	// term is measured too (see `OBJECT_MAP_ENTRY_OVERHEAD`).
	//
	// READ THE HYBRID COMMENTS AS STRUCTURE, NOT AS ARITHMETIC. Each one argues
	// which structures a key occupies -- one list node, one combined entry, one
	// slab slot -- and that argument is still what makes two policies share a
	// constant. The field-by-field byte counts inside them are the SUPERSEDED
	// hand derivation, kept only because the structural argument is written
	// around them; the number an arm returns comes from the named measured
	// constant, and every one of those hand counts was low.

	match policy {
		PaperPolicy::Auto => 0,

		// 24 bytes for the HashMap entry 48 bytes for the HashList entry,
		// 8 bytes for the HashedKey, 4 bytes for the count
		// Slab layout: 16-byte link-only slot plus one index entry
		// (8-byte key + 4-byte slot + 4-byte frequency), against `Lfu`s
		// index_map entry + HashList node + key + count, each bucket
		// carrying its own key-to-node index.
		PaperPolicy::LfuCompact => 56 + OBJECT_MAP_ROW_OVERHEAD,
		PaperPolicy::Lfu => 128 + OBJECT_MAP_ROW_OVERHEAD,

		// 48 bytes for the HashList entry, 8 bytes for the HashedKey
		// Slab layout: a 16-byte link-only slot plus one index entry, against
		// the original's 48-byte HashList node, key, and separate index. The
		// CLOCK/SIEVE visited bit and MRU's held key live in the index value,
		// so they cost nothing beyond it.
		PaperPolicy::FifoCompact => 56 + OBJECT_MAP_ROW_OVERHEAD,
		PaperPolicy::ClockCompact => 56 + OBJECT_MAP_ROW_OVERHEAD,
		PaperPolicy::SieveCompact => 56 + OBJECT_MAP_ROW_OVERHEAD,
		PaperPolicy::MruCompact => 56 + OBJECT_MAP_ROW_OVERHEAD,

		PaperPolicy::Fifo => 72 + OBJECT_MAP_ROW_OVERHEAD,

		// 48 bytes for the HashList entry, 8 bytes for the HashedKey,
		// 1 byte for the visited flag
		PaperPolicy::Clock => 72 + OBJECT_MAP_ROW_OVERHEAD,

		// 48 bytes for the HashList entry, 8 bytes for the HashedKey,
		// 1 byte for the visited flag
		PaperPolicy::Sieve => 72 + OBJECT_MAP_ROW_OVERHEAD,

		// 48 bytes for the HashList entry, 8 bytes for the HashedKey
		// Slab layout: a 16-byte link-only slot plus one index entry
		// (8-byte key + 4-byte slot number, no payload), against
		// `Lru`s 48-byte HashList node + 8-byte key + the HashLists own
		// separate key-to-node index.
		PaperPolicy::LruCompact => 56 + OBJECT_MAP_ROW_OVERHEAD,
		PaperPolicy::Lru => 72 + OBJECT_MAP_ROW_OVERHEAD,

		// 48 bytes for the HashList entry, 8 bytes for the HashedKey
		PaperPolicy::Mru => 72 + OBJECT_MAP_ROW_OVERHEAD,

		// 48 bytes for the HashList entry, 8 bytes for the HashedKey,
		// 4 bytes for the object size
		// Slab layout: one 16-byte `QueueSlot` plus the index entry that
		// finds it (8-byte key + 4-byte slot index + the 8-byte payload
		// carrying the queue tag and the object size), against the
		// original's 48-byte `HashList` node + 8-byte key + 4-byte size.
		PaperPolicy::TwoQCompact(_, _) => 72 + OBJECT_MAP_ROW_OVERHEAD,
		PaperPolicy::TwoQ(_, _) => 72 + OBJECT_MAP_ROW_OVERHEAD,

		// 48 bytes for the HashList entry, 8 bytes for the HashedKey,
		// 4 bytes for the object size
		PaperPolicy::Arc => 72 + OBJECT_MAP_ROW_OVERHEAD,

		// 48 bytes for the HashList entry, 8 bytes for the HashedKey,
		// 4 bytes for the object size, 1 byte for the frequency count
		// Slab layout: one 16-byte `QueueSlot` plus the index entry that
		// finds it (8-byte key + 4-byte slot index + the 8-byte payload
		// carrying the size, the queue tag and the frequency counter),
		// against the original's 48-byte `HashList` node + 8-byte key +
		// 4-byte size + 1-byte freq. Like `SThreeFifo` above, neither
		// charge covers the bare-key ghost queue, so the two stay
		// directly comparable.
		PaperPolicy::SThreeFifoCompact(_) => 72 + OBJECT_MAP_ROW_OVERHEAD,
		PaperPolicy::SThreeFifo(_) => 72 + OBJECT_MAP_ROW_OVERHEAD,

		// One 32-byte slab slot plus a 16-byte index entry. No `entries` map
		// and no per-key list node: the slot the index returns already carries
		// tier, size and frequency. Measured 47.4 B/key against this 48.
		PaperPolicy::LruCompactHybrid => LRU_COMPACT_HYBRID_EVICTION_STACK_DRAM_OVERHEAD + OBJECT_MAP_ROW_OVERHEAD,

		// Same 8-byte payload as `LruCompactHybrid`: `phys` was paid for out
		// of padding `LruPayload` already carried, so the layout is unchanged.
		PaperPolicy::LruLazyCopyCompactHybrid => LRU_COMPACT_HYBRID_EVICTION_STACK_DRAM_OVERHEAD + OBJECT_MAP_ROW_OVERHEAD,
		PaperPolicy::LfuCompactHybrid => LFU_COMPACT_HYBRID_EVICTION_STACK_DRAM_OVERHEAD + OBJECT_MAP_ROW_OVERHEAD,

		// Structurally identical to `LruCompactHybrid`, and deliberately so:
		// one slab slot plus one index row either way, since a key is in
		// exactly one tier's structure at a time and is never charged twice.
		// The frequency counter rides inside the 16-byte `NodePayload` the
		// arena node already carries, which every converted hybrid carries
		// whether or not it reads the field. See
		// `lru_lfu_compact_hybrid_stack.rs`'s module doc.
		PaperPolicy::LruLfuCompactHybrid(_) => LRU_LFU_COMPACT_HYBRID_EVICTION_STACK_DRAM_OVERHEAD + OBJECT_MAP_ROW_OVERHEAD,

		// Worst-case charge for a key resident in main_stack as Fast: one
		// 16-byte `QueueSlot` plus the index entry that finds it (8-byte key
		// + 4-byte slot index + the 8-byte payload carrying the queue tag,
		// the tier and the object size). One structure, not three -- see
		// `two_q_compact_hybrid_stack.rs`'s module doc.
		PaperPolicy::TwoQCompactHybrid(_) => TWO_Q_COMPACT_HYBRID_EVICTION_STACK_DRAM_OVERHEAD + OBJECT_MAP_ROW_OVERHEAD,

		// Structurally identical to `TwoQCompactHybrid`: the same
		// one-slot/one-index-row shape, differing only in which physical tier
		// the one-access FIFO queue's bytes live in (fast rather than slow) —
		// a placement decision that costs no extra per-key metadata.
		PaperPolicy::TwoQFastAdmissionCompactHybrid(_) => TWO_Q_FAST_ADMISSION_COMPACT_HYBRID_EVICTION_STACK_DRAM_OVERHEAD + OBJECT_MAP_ROW_OVERHEAD,

		// Structurally identical again: the reprieve variant changes where an
		// aged-out one-access key goes, not what is tracked per key.
		PaperPolicy::TwoQFastAdmissionReprieveCompactHybrid(_) => TWO_Q_FAST_ADMISSION_REPRIEVE_COMPACT_HYBRID_EVICTION_STACK_DRAM_OVERHEAD + OBJECT_MAP_ROW_OVERHEAD,

		// Structurally identical again, despite the third queue: a key is
		// resident in exactly one of `a1_in`/`a1_out`/`am` at any moment, so
		// it still costs one HashList entry plus one combined `entries` row
		// (queue tag + Option<Tier> tag + size). No reference bit, and no
		// ghost list -- `a1_out` holds the real objects.
		PaperPolicy::TwoQFullFastAdmissionCompactHybrid(_, _) => TWO_Q_FULL_FAST_ADMISSION_COMPACT_HYBRID_EVICTION_STACK_DRAM_OVERHEAD + OBJECT_MAP_ROW_OVERHEAD,

		// Structurally identical to `LruCompactHybrid`: one slab slot plus one
		// index row, the payload carrying tier and size — see
		// `fifo_compact_hybrid_stack.rs`'s module doc.
		PaperPolicy::FifoCompactHybrid => FIFO_COMPACT_HYBRID_EVICTION_STACK_DRAM_OVERHEAD + OBJECT_MAP_ROW_OVERHEAD,

		// Structurally identical to `LruCompactHybrid` despite having 4
		// recency lists instead of 1: a key is only ever resident in exactly
		// ONE of {small_fast, large_fast, small_slow, large_slow} at a time,
		// so only one slab slot is ever charged, and the 4-variant
		// `SizeQueue` tag still fits in the same 1 byte `Tier`'s 2-variant
		// tag did.
		PaperPolicy::LruSizedCompactHybrid => LRU_SIZED_COMPACT_HYBRID_EVICTION_STACK_DRAM_OVERHEAD + OBJECT_MAP_ROW_OVERHEAD,

		// Structurally identical to `TwoQCompactHybrid`'s charge (same shape:
		// a one-access queue + a segmented main FIFO queue, one slab slot and
		// one index row — see `s3_fifo_compact_hybrid_stack.rs`'s module
		// doc). The `accessed: bool` reference bit rides inside the 8-byte
		// payload the index row already carries, so it costs nothing beyond
		// it (only meaningful for keys currently in Main — see that field's
		// doc).
		PaperPolicy::S3FifoCompactHybrid(_) => S3_FIFO_COMPACT_HYBRID_EVICTION_STACK_DRAM_OVERHEAD + OBJECT_MAP_ROW_OVERHEAD,
		// The faithful family: same 8-byte payload, since `freq: u8`
		// replaces `accessed: bool` one-for-one. Like every other ghost
		// design here, the ghost queue's own memory is not charged.
		PaperPolicy::S3FifoFaithfulCompactHybrid(_) => S3_FIFO_COMPACT_HYBRID_EVICTION_STACK_DRAM_OVERHEAD + OBJECT_MAP_ROW_OVERHEAD,
		PaperPolicy::S3FifoFaithfulFastAdmissionCompactHybrid(_) => S3_FIFO_COMPACT_HYBRID_EVICTION_STACK_DRAM_OVERHEAD + OBJECT_MAP_ROW_OVERHEAD,
		PaperPolicy::S3FifoFaithfulReprieveCompactHybrid(_) => S3_FIFO_COMPACT_HYBRID_EVICTION_STACK_DRAM_OVERHEAD + OBJECT_MAP_ROW_OVERHEAD,
		PaperPolicy::S3FifoFaithfulFastAdmissionReprieveCompactHybrid(_) => S3_FIFO_COMPACT_HYBRID_EVICTION_STACK_DRAM_OVERHEAD + OBJECT_MAP_ROW_OVERHEAD,

		// Ghost-hybrid variants: identical per-*tracked*-object charge to
		// their non-ghost counterparts. The ghost list's own memory isn't
		// charged here at all, matching this crate's existing precedent for
		// `SThreeFifo`'s plain (non-hybrid) ghost queue above -- a ghost
		// entry only ever exists for a key that has already been evicted
		// (no longer counted in `num_objects`, which is what this whole
		// function's result gets multiplied by), so it isn't a *tracked*
		// object's overhead to add to in the first place.
		PaperPolicy::TwoQGhostCompactHybrid(_) => TWO_Q_GHOST_COMPACT_HYBRID_EVICTION_STACK_DRAM_OVERHEAD + OBJECT_MAP_ROW_OVERHEAD,
		PaperPolicy::S3FifoGhostCompactHybrid(_) => S3_FIFO_GHOST_COMPACT_HYBRID_EVICTION_STACK_DRAM_OVERHEAD + OBJECT_MAP_ROW_OVERHEAD,

		// Identical entry shape to `S3FifoGhostCompactHybrid` (same
		// `S3FifoEntry` fields: queue, tier, size, accessed) -- the
		// reference-bit gate this variant adds only changes when the bit is
		// read, not anything about the per-entry bookkeeping shape.
		PaperPolicy::S3FifoGhostLazyDemotionCompactHybrid(_) => S3_FIFO_GHOST_LAZY_DEMOTION_COMPACT_HYBRID_EVICTION_STACK_DRAM_OVERHEAD + OBJECT_MAP_ROW_OVERHEAD,

		// Identical entry shape again -- moving the one-access queue into the
		// fast tier is a placement/accounting change, not a bookkeeping-shape
		// change.
		PaperPolicy::S3FifoGhostLazyDemotionFastAdmissionCompactHybrid(_) => S3_FIFO_GHOST_LAZY_DEMOTION_FAST_ADMISSION_COMPACT_HYBRID_EVICTION_STACK_DRAM_OVERHEAD + OBJECT_MAP_ROW_OVERHEAD,

		// Identical entry shape again -- the midpoint cursor is a
		// stack-level field (like main_boundary), not a per-object one, so
		// it doesn't change this per-tracked-object charge.
		PaperPolicy::S3FifoGhostLazyDemotionFastAdmissionMidpointCompactHybrid(_) => S3_FIFO_GHOST_LAZY_DEMOTION_FAST_ADMISSION_MIDPOINT_COMPACT_HYBRID_EVICTION_STACK_DRAM_OVERHEAD + OBJECT_MAP_ROW_OVERHEAD,

		// Same `S3FifoEntry` shape as the midpoint variant above, minus the
		// ghost list -- this variant removes it entirely (a one-access key
		// that ages out is spliced into the slow tier of the main queue
		// instead of being evicted, so there's no longer any event that ever
		// populates a ghost entry). No per-tracked-object charge changes
		// either way (the ghost list was never charged per-object to begin
		// with -- see the ghost-hybrid comment above), so the number is
		// identical; only the removed list's fixed struct-level cost
		// (irrelevant here, this function is purely per-object) is gone.
		PaperPolicy::S3FifoLazyDemotionFastAdmissionMidpointReprieveCompactHybrid(_) => S3_FIFO_LAZY_DEMOTION_FAST_ADMISSION_MIDPOINT_REPRIEVE_COMPACT_HYBRID_EVICTION_STACK_DRAM_OVERHEAD + OBJECT_MAP_ROW_OVERHEAD,

		// Same per-object charge as the midpoint variant -- dropping the
		// mid-slow checkpoint removes stack-level fields (a cursor and a
		// drift counter), not per-object ones.
		PaperPolicy::S3FifoLazyDemotionFastAdmissionReprieveCompactHybrid(_) => S3_FIFO_LAZY_DEMOTION_FAST_ADMISSION_REPRIEVE_COMPACT_HYBRID_EVICTION_STACK_DRAM_OVERHEAD + OBJECT_MAP_ROW_OVERHEAD,

		// Identical per-object bookkeeping to the fast-admission reprieve
		// variant above: same `S3FifoEntry { queue, tier, size, accessed }`,
		// same two-list main queue. Moving the one-access queue to the slow
		// tier changes which allocator backs an object's bytes, not what the
		// stack records per key.
		PaperPolicy::S3FifoLazyDemotionReprieveCompactHybrid(_) => S3_FIFO_LAZY_DEMOTION_REPRIEVE_COMPACT_HYBRID_EVICTION_STACK_DRAM_OVERHEAD + OBJECT_MAP_ROW_OVERHEAD,

		// Same per-object charge as the predecessor. The slow tier being
		// two physical lists instead of one doesn't change what a tracked
		// object costs -- it's still one list node plus one combined
		// entry -- and this variant actually drops the separate
		// `Option<Tier>` field (the queue tag now carries the tier), so
		// if anything this is a slight over-estimate rather than under.
		PaperPolicy::S3FifoLazyDemotionFastAdmissionSplitSlowReprieveCompactHybrid(_) => S3_FIFO_LAZY_DEMOTION_FAST_ADMISSION_SPLIT_SLOW_REPRIEVE_COMPACT_HYBRID_EVICTION_STACK_DRAM_OVERHEAD + OBJECT_MAP_ROW_OVERHEAD,
	}
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

/// Approximate per-entry structural overhead of the shared object hashtable
/// (`DashMap`), *beyond* the stored `Object` itself (which `base_size`
/// already accounts for, including any key `K` the `Object` stores
/// internally — see `object/mod.rs`). The map's own key is a *separate*
/// `HashedKey` (the hash of `K`, not `K` itself), so this charges that
/// 8-byte key's own storage plus its amortized hashbrown overhead: `cost(8)
/// = 11` (see the derivation above).
///
/// This deliberately does **not** add load-factor slack proportional to the
/// stored `Object<K, V>`'s own size: this function isn't generic over `K`/
/// `V`, so that size is unknowable here. This makes the hashtable term an
/// under-estimate for large objects (their slack is real DRAM cost this
/// reservation doesn't see), not a safety margin — acceptable given the
/// fast-tier budget is a demotion target, not a hard data-dropping ceiling
/// (see the `lru_compact_hybrid_cache`/`lfu_compact_hybrid_cache` design
/// notes). TODO: if an exact DRAM ceiling is ever needed, thread a
/// `size_of::<Object<K,V>>()`
/// hint through from the generic `PaperCache::new` call site instead.
#[cfg(feature = "hybrid_cache_common")]
#[allow(dead_code)] // superseded by OBJECT_MAP_ENTRY_OVERHEAD; kept for the derivation notes above
pub const HASHTABLE_ENTRY_OVERHEAD: ObjectSize = 11;

/// Per-object DRAM cost of `LruCompactHybridStack`'s eviction stack.
///
/// MEASURED: jemalloc `stats.allocated`, one point per process at 2^20..2^23
/// objects, R^2 = 1.0000. See `policy_stack::measure_overhead`.
///
/// 72 B against the 112 of the split LRU hybrid this replaced -- a 35.7%
/// reduction, and the reason that design was removed rather than kept beside
/// this one. It kept a `kwik::HashList`, which owns its own key-to-node index,
/// PLUS a separate `entries` map for the 8-byte payload: two indexes, one row
/// each per object. This keeps one.
///
/// It was 64 while the payload lived in the slab slot. Moving it into the index
/// value costs 8 B/object and buys 12% on `move_front` -- LRU's hot path -- and
/// 47% on metadata reads, measured on an idle machine. The list operation gets
/// faster because the slab is denser without the payload (16-byte slots against
/// 24), so the pointer chase touches fewer cache lines. Equal to
/// `TwoQCompactHybridStack` and `S3FifoCompactHybridStack`, which is expected:
/// all three now share `CompactQueueSet` and all three payloads are 8 bytes.
#[cfg(any(feature = "hybrid_cache_common", not(feature = "merged_object_store")))]
/// MEASURED after the arena conversion, `measure_one_point`, release, one
/// process per point, powers of two:
///
/// ```text
///   policy                          2^20      2^21      2^22      2^23
///   lru-compact-hybrid           40.2100   40.1050   40.0518   40.0244
///   fifo-compact-hybrid          40.2100   40.1050   40.0518   40.0244
///   lru-sized-compact-hybrid     40.2100   40.1050   40.0518   40.0244
///   lru-lazy-copy-compact-hybrid 40.2295   40.1147   40.0567   40.0268
///   lfu-compact-hybrid (control) 72.5952   72.2962   72.1451   72.0707
/// ```
///
/// Forty is PREDICTED, not merely fitted: the arena node is 32 bytes and its
/// keyless bucket array is 8 B/object at the doubling slack it holds. The
/// residue above 40 is a fixed intercept, not a per-object term, which is why
/// it shrinks with n.
///
/// `lfu-compact-hybrid` was the control in THAT run and did not move, because
/// it had not been converted. It has been converted since. `CompactQueueSet`
/// could not hold it -- LFU needs one ordered bucket per DISTINCT FREQUENCY,
/// because eviction has to find the minimum, which a fixed four-queue tag
/// cannot express -- so it moved to `ArenaFrequencyChain` instead, which keeps
/// the ordered bucket maps and puts the arena's 32-byte node and keyless index
/// underneath them. Re-measured the same way:
///
/// ```text
///   policy                            2^20      2^21      2^22      2^23
///   lfu-compact-hybrid             40.2252   40.1126   40.0556   40.0263
///   lru-lfu-compact-hybrid-2       40.2274   40.1137   40.0561   40.0266
///   lru-compact-hybrid (control)   40.2100   40.1050   40.0518   40.0244
/// ```
///
/// The control role passed to this constant's own policy, and it reproduced
/// 40.2100 / 40.1050 / 40.0518 / 40.0244 to four decimal places on the
/// converted tree -- which is the evidence the harness did not move underneath
/// LFU. The same binaries put the unconverted LFU at 72.5952 / 72.2962 /
/// 72.1451 / 72.0707, measured from a `git archive` of the pre-conversion
/// commit.
///
/// LFU sits ~0.002 B/object above LRU at every point because
/// `ArenaFrequencyChain` also carries two ordered bucket maps. Those are
/// O(DISTINCT FREQUENCIES), not O(objects): the gap is a fixed ~15 KB at every
/// population measured, which is why it shrinks with n rather than holding.
const LRU_COMPACT_HYBRID_EVICTION_STACK_DRAM_OVERHEAD: ObjectSize = 40;

/// Per-object DRAM cost of `LfuCompactHybridStack`'s eviction-stack
/// bookkeeping.
///
/// One `ArenaFrequencyChain` node -- key 8, prev 4, next 4, `NodePayload` 16 --
/// plus the keyless index that finds it, four bytes a bucket at the half load
/// it grows to, so 8 B/object. There is no third structure and no `entries`
/// map: the slot the index returns already carries tier, size and count.
///
/// It was 72 while the chain kept a `HashMap<HashedKey, (u32, CompactEntry)>`
/// index, which stored every key a SECOND time so a probe could compare it --
/// the same eight bytes the slot already carried so an eviction could name its
/// victim. That index was 56 B/object. Replacing it with bare `u32` slot
/// numbers verified against the slot's own key costs 16 bytes of node (the
/// payload moves in, and it is the shared 16-byte `NodePayload` rather than a
/// 12-byte `CompactEntry`) and saves 48 of index.
///
/// MEASURED, not derived: jemalloc `stats.allocated`, ONE PROCESS PER POINT, at
/// powers of two, `MEASURE_POLICY=lfu-compact-hybrid`. Before is a `git
/// archive` of the pre-conversion commit, built and run the same way:
///
/// ```text
///   n         before    after
///   2^20     72.5952  40.2252
///   2^21     72.2962  40.1126
///   2^22     72.1451  40.0556
///   2^23     72.0707  40.0263
/// ```
///
/// Forty is PREDICTED and not merely fitted -- 32 of node and 8 of index -- and
/// the residue above it is a fixed intercept, which is why it shrinks with n.
/// See `policy_stack::measure_overhead`.
///
/// ALLOCATED, not resident -- size-class-rounded usable bytes, the quantity
/// `malloc_usable_size` returns and therefore the same quantity Redis reports
/// as `used_memory`. An earlier revision measured RSS instead, which counts
/// retained-but-freed pages that belong in a fragmentation ratio rather than in
/// a per-object cost, and which disagreed with itself by 20% depending on where
/// the sample points fell.
///
/// The field-by-field derivation that used to sit here understated this stack
/// by roughly a third: it counted struct fields and not size-class rounding,
/// index-map load factor, or the growth slack of every doubling structure.
/// Being a measured allocation figure it is NOT multiplied by
/// `resident_factor()` -- see the split in `get_hybrid_dram_shared_overhead`.
#[cfg(any(feature = "hybrid_cache_common", not(feature = "merged_object_store")))]
const LFU_COMPACT_HYBRID_EVICTION_STACK_DRAM_OVERHEAD: ObjectSize = 40;

/// Per-object DRAM cost of `LruSizedCompactHybridStack`.
///
/// MEASURED, not derived: jemalloc `stats.allocated`, one point per
/// process, sampled at powers of two. 72 B/object, R2 = 1.0000.
#[cfg(any(feature = "hybrid_cache_common", not(feature = "merged_object_store")))]
const LRU_SIZED_COMPACT_HYBRID_EVICTION_STACK_DRAM_OVERHEAD: ObjectSize = 40;

/// Per-object DRAM cost of `LruLfuCompactHybridStack`.
///
/// The same `ArenaFrequencyChain` as `LfuCompactHybridStack`, so the same term:
/// this design puts its recency-ordered fast tier in the chain's distinguished
/// recency list and its frequency-ordered slow tier in the chain's buckets, and
/// a key is in exactly one of them at a time, so one node per key covers both.
///
/// MEASURED separately rather than inferred from that argument -- jemalloc
/// `stats.allocated`, one process per point, powers of two,
/// `MEASURE_POLICY=lru-lfu-compact-hybrid-2`, against a `git archive` of the
/// pre-conversion commit:
///
/// ```text
///   n         before    after
///   2^20     72.5760  40.2274
///   2^21     72.2866  40.1137
///   2^22     72.1403  40.0561
///   2^23     72.0698  40.0266
/// ```
#[cfg(any(feature = "hybrid_cache_common", not(feature = "merged_object_store")))]
const LRU_LFU_COMPACT_HYBRID_EVICTION_STACK_DRAM_OVERHEAD: ObjectSize = 40;

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
/// object-hashtable slot**, which is why it must never carry
/// [`HASHTABLE_ENTRY_OVERHEAD`]. The owning stacks multiply it by
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
/// Gated on `eviction_stacks_pmem` **only** (never `global_hashtable_pmem`,
/// per the no-hashtable-slot point above): when that feature moves the
/// eviction stacks — ghost list included — to PMEM, the ghost costs the
/// fast/DRAM tier nothing and the term drops to 0.
///
/// Unlike the per-policy constants below this is *not* gated on
/// `hybrid_cache_common`: the policy-stack modules are declared
/// unconditionally (see `worker::policy::policy_stack`), so they compile —
/// and reference this — under every feature combination, including none.
#[cfg(not(feature = "eviction_stacks_pmem"))]
pub const GHOST_ENTRY_DRAM_OVERHEAD: ObjectSize = 8;

/// Per-entry DRAM cost of an EXACT ghost queue -- a `CompactQueueSet<()>` of
/// bare keys, as the faithful S3-FIFO family carries.
///
/// Distinct from [`GHOST_ENTRY_DRAM_OVERHEAD`] above, which sizes a `GhostSlot`
/// FINGERPRINT (8 bytes, approximate, fixed-capacity). An exact ghost costs a
/// 16-byte `QueueSlot` plus the 12-byte index entry that finds it -- the same
/// 16 + 12 shape charged for `LruCompact` -- because flat `SThreeFifoStack`'s
/// ghost is exact and a faithful port cannot substitute an approximate filter
/// without changing which keys get admitted to main.
///
/// 3.5x the fingerprint's cost per entry, and that is the real price of
/// fidelity here; it is bounded by the main queue's length, which the ghost is
/// trimmed against.
#[cfg(not(feature = "eviction_stacks_pmem"))]
pub const EXACT_GHOST_ENTRY_DRAM_OVERHEAD: ObjectSize = 16 + 12;

/// PMEM-resident ghost list: costs the fast/DRAM tier nothing. See the
/// `not(eviction_stacks_pmem)` arm above for the derivation and rationale.
#[cfg(feature = "eviction_stacks_pmem")]
pub const GHOST_ENTRY_DRAM_OVERHEAD: ObjectSize = 0;

/// Zero under `eviction_stacks_pmem` for the same reason as
/// [`GHOST_ENTRY_DRAM_OVERHEAD`]: `CompactQueueSet` is allocator-parameterised,
/// so the exact ghost follows the eviction stacks to the far node and stops
/// occupying fast-tier DRAM.
#[cfg(feature = "eviction_stacks_pmem")]
pub const EXACT_GHOST_ENTRY_DRAM_OVERHEAD: ObjectSize = 0;

/// Per-object DRAM cost of `FifoCompactHybridStack`.
///
/// PLACEHOLDER pending measurement: shares `CompactQueueSet` and an 8-byte
/// payload with the other converted queue stacks, all MEASURED at 72.
#[cfg(any(feature = "hybrid_cache_common", not(feature = "merged_object_store")))]
const FIFO_COMPACT_HYBRID_EVICTION_STACK_DRAM_OVERHEAD: ObjectSize = 40;

/// Per-object DRAM cost of `TwoQCompactHybridStack`'s eviction stack.
///
/// MEASURED: jemalloc `stats.allocated`, one point per process at 2^20..2^23
/// objects, R^2 = 1.0000. See `policy_stack::measure_overhead`.
///
/// 72 B against the 112 of the split 2Q hybrid this replaced -- a 35.7%
/// reduction. That design kept THREE indexes for a population where every key
/// is in exactly one of its two queues: a FIFO `HashList` and an LRU
/// `HashList`, each owning its own key-to-node map, plus the separate
/// `entries` map. This keeps one.
///
/// 8 B above `LruCompactHybridStack`'s 64, and that gap is the layout choice
/// rather than the policy: this stack carries the payload in the index value
/// (layout B) where the LRU list carries it in the slab slot (layout A). The
/// standalone comparison measured layout B at +8.01 B/object, so the two
/// results agree to within a byte. B is right here because `mark_accessed` and
/// the queue-dispatch read in `touch` are hot AND touch no queue order.
#[cfg(any(feature = "hybrid_cache_common", not(feature = "merged_object_store")))]
const TWO_Q_COMPACT_HYBRID_EVICTION_STACK_DRAM_OVERHEAD: ObjectSize = 40;

/// Per-object DRAM cost of `TwoQFastAdmissionCompactHybridStack`.
///
/// MEASURED: jemalloc `stats.allocated`, one point per process at 2^20..2^23
/// objects, R^2 = 1.0000. See `policy_stack::measure_overhead`.
///
/// 72 B against the split fast-admission 2Q's 112 -- a 35.7% reduction, and
/// equal to the compact 2Q, S3-FIFO and LRU stacks. All four share
/// `CompactQueueSet` and an 8-byte payload, so equality was the prediction and
/// the measurement confirms it.
#[cfg(any(feature = "hybrid_cache_common", not(feature = "merged_object_store")))]
const TWO_Q_FAST_ADMISSION_COMPACT_HYBRID_EVICTION_STACK_DRAM_OVERHEAD: ObjectSize = 40;

/// Per-object DRAM cost of `TwoQFastAdmissionReprieveCompactHybridStack`.
///
/// PLACEHOLDER pending measurement: it shares `CompactQueueSet` and an 8-byte
/// payload with the other converted queue stacks, all MEASURED at 72.
#[cfg(any(feature = "hybrid_cache_common", not(feature = "merged_object_store")))]
const TWO_Q_FAST_ADMISSION_REPRIEVE_COMPACT_HYBRID_EVICTION_STACK_DRAM_OVERHEAD: ObjectSize = 40;

/// Per-object DRAM cost of `TwoQFullFastAdmissionCompactHybridStack`.
///
/// MEASURED at 72 by the converting agent, matching every other stack sharing
/// `CompactQueueSet` and an 8-byte payload. Three queues rather than two makes
/// no difference: a key is in exactly one of them at a time.
#[cfg(any(feature = "hybrid_cache_common", not(feature = "merged_object_store")))]
const TWO_Q_FULL_FAST_ADMISSION_COMPACT_HYBRID_EVICTION_STACK_DRAM_OVERHEAD: ObjectSize = 40;

/// Per-object DRAM cost of `TwoQGhostCompactHybridStack`.
///
/// PLACEHOLDER pending measurement: it shares `CompactQueueSet` and an 8-byte
/// payload with the other converted queue stacks, all MEASURED at 72.
#[cfg(any(feature = "hybrid_cache_common", not(feature = "merged_object_store")))]
const TWO_Q_GHOST_COMPACT_HYBRID_EVICTION_STACK_DRAM_OVERHEAD: ObjectSize = 40;

/// Per-object DRAM cost of `S3FifoCompactHybridStack`'s eviction stack.
///
/// MEASURED: jemalloc `stats.allocated`, one point per process at 2^20..2^23
/// objects, R^2 = 1.0000. See `policy_stack::measure_overhead`.
///
/// 72 B against the split S3-FIFO hybrid's 112 -- a 35.7% reduction -- and identical
/// to the measured `TwoQCompactHybridStack`, which is the expected result:
/// the two share the primitive and both payloads are 8 bytes. Predicted before
/// the run and confirmed by it.
#[cfg(any(feature = "hybrid_cache_common", not(feature = "merged_object_store")))]
const S3_FIFO_COMPACT_HYBRID_EVICTION_STACK_DRAM_OVERHEAD: ObjectSize = 40;

/// Per-object DRAM cost of `S3FifoGhostCompactHybridStack`.
///
/// PLACEHOLDER pending measurement: it shares `CompactQueueSet` and an 8-byte
/// payload with the other converted queue stacks, all MEASURED at 72.
#[cfg(any(feature = "hybrid_cache_common", not(feature = "merged_object_store")))]
const S3_FIFO_GHOST_COMPACT_HYBRID_EVICTION_STACK_DRAM_OVERHEAD: ObjectSize = 40;

/// Per-object DRAM cost of `S3FifoGhostLazyDemotionCompactHybridStack`.
///
/// PLACEHOLDER pending measurement: it shares `CompactQueueSet` and an 8-byte
/// payload with the other converted queue stacks, all MEASURED at 72.
#[cfg(any(feature = "hybrid_cache_common", not(feature = "merged_object_store")))]
const S3_FIFO_GHOST_LAZY_DEMOTION_COMPACT_HYBRID_EVICTION_STACK_DRAM_OVERHEAD: ObjectSize = 40;

/// Per-object DRAM cost of `S3FifoGhostLazyDemotionFastAdmissionCompactHybridStack`.
///
/// PLACEHOLDER pending measurement: it shares `CompactQueueSet` and an 8-byte
/// payload with the other converted queue stacks, all MEASURED at 72.
#[cfg(any(feature = "hybrid_cache_common", not(feature = "merged_object_store")))]
const S3_FIFO_GHOST_LAZY_DEMOTION_FAST_ADMISSION_COMPACT_HYBRID_EVICTION_STACK_DRAM_OVERHEAD: ObjectSize = 40;

/// Per-object DRAM cost of `S3FifoGhostLazyDemotionFastAdmissionMidpointCompactHybridStack`.
///
/// PLACEHOLDER pending measurement: it shares `CompactQueueSet` and an 8-byte
/// payload with the other converted queue stacks, all MEASURED at 72.
#[cfg(any(feature = "hybrid_cache_common", not(feature = "merged_object_store")))]
const S3_FIFO_GHOST_LAZY_DEMOTION_FAST_ADMISSION_MIDPOINT_COMPACT_HYBRID_EVICTION_STACK_DRAM_OVERHEAD: ObjectSize = 40;

/// Per-object DRAM cost of `S3FifoLazyDemotionReprieveCompactHybridStack`.
///
/// PLACEHOLDER pending measurement: it shares `CompactQueueSet` and an 8-byte
/// payload with the other converted queue stacks, all MEASURED at 72.
#[cfg(any(feature = "hybrid_cache_common", not(feature = "merged_object_store")))]
const S3_FIFO_LAZY_DEMOTION_REPRIEVE_COMPACT_HYBRID_EVICTION_STACK_DRAM_OVERHEAD: ObjectSize = 40;

/// Per-object DRAM cost of `S3FifoLazyDemotionFastAdmissionReprieveCompactHybridStack`.
///
/// PLACEHOLDER pending measurement: it shares `CompactQueueSet` and an 8-byte
/// payload with the other converted queue stacks, all MEASURED at 72.
#[cfg(any(feature = "hybrid_cache_common", not(feature = "merged_object_store")))]
const S3_FIFO_LAZY_DEMOTION_FAST_ADMISSION_REPRIEVE_COMPACT_HYBRID_EVICTION_STACK_DRAM_OVERHEAD: ObjectSize = 40;

/// Per-object DRAM cost of `S3FifoLazyDemotionFastAdmissionMidpointReprieveCompactHybridStack`.
///
/// PLACEHOLDER pending measurement: it shares `CompactQueueSet` and an 8-byte
/// payload with the other converted queue stacks, all MEASURED at 72.
#[cfg(any(feature = "hybrid_cache_common", not(feature = "merged_object_store")))]
const S3_FIFO_LAZY_DEMOTION_FAST_ADMISSION_MIDPOINT_REPRIEVE_COMPACT_HYBRID_EVICTION_STACK_DRAM_OVERHEAD: ObjectSize = 40;

/// Per-object DRAM cost of `S3FifoLazyDemotionFastAdmissionSplitSlowReprieveCompactHybridStack`.
///
/// PLACEHOLDER pending measurement: it shares `CompactQueueSet` and an 8-byte
/// payload with the other converted queue stacks, all MEASURED at 72.
#[cfg(any(feature = "hybrid_cache_common", not(feature = "merged_object_store")))]
const S3_FIFO_LAZY_DEMOTION_FAST_ADMISSION_SPLIT_SLOW_REPRIEVE_COMPACT_HYBRID_EVICTION_STACK_DRAM_OVERHEAD: ObjectSize = 40;


/// Approximate per-object DRAM cost of the *shared* structures (the object
/// hashtable + the eviction stacks) that hold an entry for every object of both
/// tiers. Used by the LRU/LFU/LRU-sized hybrid stacks to reserve room in the
/// fast-tier (DRAM) budget so demotion bounds total DRAM, not just fast-tier
/// values.
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
const VALUE_ALLOCATION_OVERHEAD: ObjectSize = 32;

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

/// Requested-to-resident multiplier for the DRAM metadata reserved above.
///
/// The terms above count bytes the cache *requests*; the fast-tier budget is
/// meant to bound bytes actually *resident*, and an allocator holds more than
/// was asked for -- size-class rounding plus whatever it retains rather than
/// returning to the OS. Measured at peak on cluster12:
///
/// | allocator                    | rounding | retention | total |
/// |------------------------------|----------|-----------|-------|
/// | jemalloc (default)            | 1.064    | 1.29      | ~1.37 |
/// | jemalloc (`numa_jemalloc`) | 1.061 | 1.017-1.056 | 1.08-1.12 |
///
/// Rounding is near-identical between them; the entire difference is
/// retention, because TBB's per-thread and large-object caches have no purge
/// discipline while jemalloc decays dirty pages back.
///
/// # This is an allocator property, not a workload constant
///
/// It is a *ratio*, so unlike the per-object terms it is not inflated by the
/// harness's own allocations -- those bytes pay the same multiplier. What it
/// does depend on is churn, which depends on the fast-tier size, which this
/// number helps determine: measured values on TBB ranged 1.29-2.75 across
/// configurations for exactly that reason. Treat the constants as starting
/// points for the shipped configurations and recalibrate with
/// `DRAM_OVERHEAD_RESIDENT_FACTOR` when the workload or allocator changes;
/// `jemalloc_stats()` reports the inputs.
///
/// A second-order caveat: the ratio is measured process-wide, so a workload
/// whose non-cache allocations have a very different size profile from the
/// cache's metadata will skew it slightly.
/// Measured 1.08-1.12 resident/allocated on the NUMA-bound jemalloc arenas
/// that back every build. The jemalloc pairing this replaced needed 1.37, and
/// carrying TBB's number into a jemalloc build over-reserved by ~22%,
/// shrinking the effective fast tier for no reason.
#[cfg(feature = "hybrid_cache_common")]
const DEFAULT_RESIDENT_FACTOR: f64 = 1.12;

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
/// `mem_fragmentation_ratio`, not inside a per-object budget. See
/// `AtomicStatus::fragmentation_ratio`.
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

#[cfg(feature = "hybrid_cache_common")]
/// NOTE: nothing applies this any more. `get_hybrid_dram_shared_overhead`
/// dropped it when its terms moved to measured jemalloc `stats.allocated`
/// figures, which are already size-class-rounded (see that function's closing
/// comment); the per-policy doc comments above still reference the concept.
/// Retained rather than deleted because `DRAM_OVERHEAD_RESIDENT_FACTOR` is a
/// documented knob -- but it is currently INERT, and setting it changes
/// nothing.
#[allow(dead_code)]
fn resident_factor() -> f64 {
	use std::sync::OnceLock;
	static FACTOR: OnceLock<f64> = OnceLock::new();

	*FACTOR.get_or_init(|| {
		std::env::var("DRAM_OVERHEAD_RESIDENT_FACTOR")
			.ok()
			.and_then(|value| value.parse::<f64>().ok())
			.filter(|factor| (1.0..=4.0).contains(factor))
			.unwrap_or(DEFAULT_RESIDENT_FACTOR)
	})
}

#[cfg(feature = "hybrid_cache_common")]
pub fn get_hybrid_dram_shared_overhead(policy: &PaperPolicy) -> ObjectSize {
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
	// 62 since the v3 slot, re-measured at 61.30 B/object structural -- see
	// `MERGED_STORE_STRUCTURE_OVERHEAD`. The reservation is per LIVE object and
	// `MergedStore::settle_tier` takes it off the fast budget before the
	// watermarks, so this number directly sets how many objects fit in DRAM.
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
	// Selected by a runtime `match`, not by `cfg`. These are integers: nothing
	// about a policy's constant requires its stack module to be compiled, and
	// gating them meant a build without that feature silently contributed 0 --
	// no error, no warning, no failing test. That is not hypothetical: a binary
	// built with only one hybrid feature charged every other policy
	// Arc(48) + map(63) = 111 -> 124 B/object instead of its real 196 or 228,
	// so each non-LRU policy was handed a larger effective fast tier than it
	// should have had, for a whole sweep, before anyone noticed.
	//
	// The match is exhaustive deliberately: adding a policy without giving it an
	// overhead term is now a compile error rather than a silent zero.
	// Measured resident, kept separate from the derived terms below.
	#[allow(unused_mut)]
	let mut stack_resident: ObjectSize = 0;

	#[cfg(not(feature = "eviction_stacks_pmem"))]
	{
		stack_resident = match policy {
			PaperPolicy::LruCompactHybrid => LRU_COMPACT_HYBRID_EVICTION_STACK_DRAM_OVERHEAD,
			PaperPolicy::LruLazyCopyCompactHybrid => LRU_COMPACT_HYBRID_EVICTION_STACK_DRAM_OVERHEAD,
			PaperPolicy::LfuCompactHybrid => LFU_COMPACT_HYBRID_EVICTION_STACK_DRAM_OVERHEAD,
			PaperPolicy::LruSizedCompactHybrid => LRU_SIZED_COMPACT_HYBRID_EVICTION_STACK_DRAM_OVERHEAD,
			PaperPolicy::LruLfuCompactHybrid(..) => LRU_LFU_COMPACT_HYBRID_EVICTION_STACK_DRAM_OVERHEAD,
			PaperPolicy::FifoCompactHybrid => FIFO_COMPACT_HYBRID_EVICTION_STACK_DRAM_OVERHEAD,
			PaperPolicy::TwoQCompactHybrid(..) => TWO_Q_COMPACT_HYBRID_EVICTION_STACK_DRAM_OVERHEAD,
			PaperPolicy::TwoQFastAdmissionCompactHybrid(..) => TWO_Q_FAST_ADMISSION_COMPACT_HYBRID_EVICTION_STACK_DRAM_OVERHEAD,
			PaperPolicy::TwoQFastAdmissionReprieveCompactHybrid(..) => TWO_Q_FAST_ADMISSION_REPRIEVE_COMPACT_HYBRID_EVICTION_STACK_DRAM_OVERHEAD,
			PaperPolicy::TwoQFullFastAdmissionCompactHybrid(..) => TWO_Q_FULL_FAST_ADMISSION_COMPACT_HYBRID_EVICTION_STACK_DRAM_OVERHEAD,
			PaperPolicy::TwoQGhostCompactHybrid(..) => TWO_Q_GHOST_COMPACT_HYBRID_EVICTION_STACK_DRAM_OVERHEAD,
			PaperPolicy::S3FifoCompactHybrid(..) => S3_FIFO_COMPACT_HYBRID_EVICTION_STACK_DRAM_OVERHEAD,
			PaperPolicy::S3FifoFaithfulCompactHybrid(..) => S3_FIFO_COMPACT_HYBRID_EVICTION_STACK_DRAM_OVERHEAD,
			PaperPolicy::S3FifoFaithfulFastAdmissionCompactHybrid(..) => S3_FIFO_COMPACT_HYBRID_EVICTION_STACK_DRAM_OVERHEAD,
			PaperPolicy::S3FifoFaithfulReprieveCompactHybrid(..) => S3_FIFO_COMPACT_HYBRID_EVICTION_STACK_DRAM_OVERHEAD,
			PaperPolicy::S3FifoFaithfulFastAdmissionReprieveCompactHybrid(..) => S3_FIFO_COMPACT_HYBRID_EVICTION_STACK_DRAM_OVERHEAD,
			PaperPolicy::S3FifoGhostCompactHybrid(..) => S3_FIFO_GHOST_COMPACT_HYBRID_EVICTION_STACK_DRAM_OVERHEAD,
			PaperPolicy::S3FifoGhostLazyDemotionCompactHybrid(..) => S3_FIFO_GHOST_LAZY_DEMOTION_COMPACT_HYBRID_EVICTION_STACK_DRAM_OVERHEAD,
			PaperPolicy::S3FifoGhostLazyDemotionFastAdmissionCompactHybrid(..) => S3_FIFO_GHOST_LAZY_DEMOTION_FAST_ADMISSION_COMPACT_HYBRID_EVICTION_STACK_DRAM_OVERHEAD,
			PaperPolicy::S3FifoGhostLazyDemotionFastAdmissionMidpointCompactHybrid(..) => S3_FIFO_GHOST_LAZY_DEMOTION_FAST_ADMISSION_MIDPOINT_COMPACT_HYBRID_EVICTION_STACK_DRAM_OVERHEAD,
			PaperPolicy::S3FifoLazyDemotionReprieveCompactHybrid(..) => S3_FIFO_LAZY_DEMOTION_REPRIEVE_COMPACT_HYBRID_EVICTION_STACK_DRAM_OVERHEAD,
			PaperPolicy::S3FifoLazyDemotionFastAdmissionReprieveCompactHybrid(..) => S3_FIFO_LAZY_DEMOTION_FAST_ADMISSION_REPRIEVE_COMPACT_HYBRID_EVICTION_STACK_DRAM_OVERHEAD,
			PaperPolicy::S3FifoLazyDemotionFastAdmissionMidpointReprieveCompactHybrid(..) => S3_FIFO_LAZY_DEMOTION_FAST_ADMISSION_MIDPOINT_REPRIEVE_COMPACT_HYBRID_EVICTION_STACK_DRAM_OVERHEAD,
			PaperPolicy::S3FifoLazyDemotionFastAdmissionSplitSlowReprieveCompactHybrid(..) => S3_FIFO_LAZY_DEMOTION_FAST_ADMISSION_SPLIT_SLOW_REPRIEVE_COMPACT_HYBRID_EVICTION_STACK_DRAM_OVERHEAD,

			// All-DRAM policies have no tiers and reserve no fast-tier metadata.
			PaperPolicy::Auto
			| PaperPolicy::Lfu
			| PaperPolicy::Fifo
			| PaperPolicy::Clock
			| PaperPolicy::Sieve
			| PaperPolicy::Lru
			| PaperPolicy::LruCompact
			| PaperPolicy::LfuCompact
			| PaperPolicy::FifoCompact
			| PaperPolicy::ClockCompact
			| PaperPolicy::SieveCompact
			| PaperPolicy::MruCompact
			| PaperPolicy::Mru
			| PaperPolicy::TwoQ(..)
			| PaperPolicy::TwoQCompact(..)
			| PaperPolicy::Arc
			| PaperPolicy::SThreeFifo(..)
			| PaperPolicy::SThreeFifoCompact(..) => 0,
		};
		overhead += stack_resident;
	}

	// The value's own allocation used to cost a DRAM-resident refcounted
	// header regardless of which tier the bytes themselves occupied. Since v5
	// there is no such allocation, so this term is ZERO -- kept, rather than
	// deleted, so that this reservation and `get_policy_overhead` name the
	// same terms and can be compared line for line. MEASURED zero: see the
	// constant's own doc for the before/after fits.
	overhead += VALUE_ALLOCATION_OVERHEAD;

	// The object map lives in DRAM unless a hashtable-PMEM feature
	// relocates it (`global_hashtable_pmem`).
	//
	// MEASURED at 80.0 B/object for the default `DashMap` shape: the whole
	// map measures 144.0 at a 64-byte value (R2 = 1.000000), and the non-value
	// remainder is 79.998 at 16-, 32-, 64- AND 128-byte values, so it is a
	// container cost rather than a mis-attributed value cost.
	#[cfg(not(feature = "global_hashtable_pmem"))]
	{
		overhead += OBJECT_MAP_ENTRY_OVERHEAD;
	}

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
	/// This is a regression test for a silent, whole-sweep measurement error.
	/// The terms used to be `cfg`-gated per policy, so a binary built with only
	/// a single hybrid feature -- which is how the benchmark was configured --
	/// charged every non-LRU policy `Arc(48) + map(63) = 111 -> 124` B/object
	/// instead of its real 196 or 228. Each of those policies was therefore
	/// handed a larger effective fast tier than it should have had, and nothing
	/// failed: no error, no warning, no test.
	///
	/// This test runs under whatever feature set the build has, so it fails if
	/// anyone reintroduces the gating.
	#[test]
	fn every_hybrid_policy_keeps_its_own_term() {
		if std::env::var_os("PAPER_DISABLE_SHARED_OVERHEAD").is_some() {
			return; // the escape hatch zeroes everything by design
		}

		let lru = get_hybrid_dram_shared_overhead(&PaperPolicy::LruCompactHybrid);
		let lfu = get_hybrid_dram_shared_overhead(&PaperPolicy::LfuCompactHybrid);
		let fifo = get_hybrid_dram_shared_overhead(&PaperPolicy::FifoCompactHybrid);
		let s3 = get_hybrid_dram_shared_overhead(&PaperPolicy::S3FifoCompactHybrid(0.1));

		// The value with NO eviction-stack term -- what a gated-out policy
		// collapses to.
		//
		// This was `* resident_factor()`, which made the guard DEAD: the
		// function applies no resident factor (see its closing comment), so a
		// collapsed policy returns 144 while this expression produced 161, and
		// `assert_ne!` could never fire. The one thing this test exists to
		// catch was the one thing it could not catch.
		#[allow(unused_variables)] // named by two of the three arms below
		let no_stack_term = VALUE_ALLOCATION_OVERHEAD + OBJECT_MAP_ENTRY_OVERHEAD;

		// Under `eviction_stacks_pmem` the stacks live in CXL, so they are
		// deliberately absent from the FAST-TIER reservation -- while still
		// counting toward the aggregate budget in `get_policy_overhead`. Every
		// policy therefore collapses to exactly `no_stack_term` ON PURPOSE, and
		// the assertions below invert. Same property, checked from the other
		// side: the split is what makes both directions meaningful.
		#[cfg(all(feature = "eviction_stacks_pmem", not(feature = "merged_object_store")))]
		for (name, got) in [("lru", lru), ("lfu", lfu), ("fifo", fifo), ("s3-fifo", s3)] {
			assert_eq!(
				got, no_stack_term,
				"{name} still reserves fast-tier DRAM for an eviction stack that \
				 lives in CXL -- the pmem accounting split is broken",
			);
		}

		// Under `merged_object_store` there is no per-policy eviction stack to
		// carry a term: the object map IS the eviction structure, so every
		// policy reserves the merged store's own measured structural cost and
		// nothing else. The assertions invert here for the same reason they do
		// under `eviction_stacks_pmem` -- the term is deliberately absent, not
		// lost to cfg gating -- so the check becomes that they all collapse to
		// exactly that one figure.
		#[cfg(feature = "merged_object_store")]
		for (name, got) in [("lru", lru), ("lfu", lfu), ("fifo", fifo), ("s3-fifo", s3)] {
			assert_eq!(
				got,
				MERGED_STORE_STRUCTURE_OVERHEAD + VALUE_ALLOCATION_OVERHEAD,
				"{name} reserves fast-tier DRAM for a split-design eviction stack 				 this build does not have",
			);
		}

		#[cfg(not(any(feature = "eviction_stacks_pmem", feature = "merged_object_store")))]
		{
			for (name, got) in [("lru", lru), ("lfu", lfu), ("fifo", fifo), ("s3-fifo", s3)] {
				assert_ne!(
					got, no_stack_term,
					"{name} lost its eviction-stack term and collapsed to {no_stack_term} \
					 B/object -- the per-policy cfg gating is back",
				);
			}

			// Every stack here is on the arena node now -- a 32-byte node plus
			// a KEYLESS 8-byte bucket array, measured at 40 B/object -- LFU
			// included. It could not use `ArenaQueueSet`, whose orders are a
			// fixed `[u32; MAX_QUEUES]`, because LFU needs one ordered bucket
			// per DISTINCT frequency; it uses `ArenaFrequencyChain`, which
			// keeps those bucket maps over the arena's node and index.
			//
			// So LFU is INSIDE the group now rather than the control outside
			// it, and equality is the claim across all four.
			//
			// This assertion used to read `lfu - lru == 32`, and it was true
			// only BECAUSE lfu was unconverted. Changing that 32 to 0 would
			// have said nothing at all: four values that must be equal are
			// already asserted equal below, and a gap of zero adds no claim.
			assert_eq!(fifo, lru, "fifo and lru share the arena node");
			assert_eq!(s3, lru, "s3-fifo and lru share the arena node");
			assert_eq!(
				lfu, lru,
				"lfu moved to `ArenaFrequencyChain` and must now carry the \
				 same arena node as the queue stacks",
			);

			// Equality alone would survive all four regressing TOGETHER, which
			// is exactly what the old assertion's cross-design gap guarded
			// against. There is no unconverted design left to measure a gap
			// to, so the term is pinned to the STRUCTURE it charges for
			// instead, and stated as a DECOMPOSITION rather than as a total:
			// 40 is the node and the index, and each half is pinned where it
			// is built.
			//
			// The two halves cannot be `size_of`d here -- `worker::policy` is
			// a private module and this file is in `object` -- but they are
			// not free-floating either, and their own checks fire FIRST:
			//
			//   32  `const _: () = assert!(size_of::<ArenaSlot<NodePayload>>()
			//       == 32)`, beside `NodePayload` in `arena_queue_set`. A
			//       compile error, not a test failure.
			//    8  `the_index_costs_eight_bytes_per_object_at_a_power_of_two_
			//       population`, in both of `arena_index`'s consumers. Four
			//       bytes a bucket at the half load `KeylessIndex` grows to.
			//
			// So adding a field to `NodePayload`, widening the index's bucket
			// or loosening its load factor breaks the structure's own check
			// first and this one second, which is the order that reads
			// correctly: the structure moved, so the CHARGE is now unmeasured.
			// The fix is to re-run `measure_one_point`, not to adjust one side
			// of this assertion to fit the other.
			const ARENA_NODE: ObjectSize = 32;
			const KEYLESS_INDEX_PER_OBJECT: ObjectSize = 8;

			assert_eq!(
				LRU_COMPACT_HYBRID_EVICTION_STACK_DRAM_OVERHEAD,
				ARENA_NODE + KEYLESS_INDEX_PER_OBJECT,
				"the arena term stopped matching the node and index it charges \
				 for",
			);

			// `LruLfuCompactHybrid` is not one of the four queried above and
			// would otherwise be pinned by nothing. It shares
			// `ArenaFrequencyChain` with `LfuCompactHybrid`, so it shares the
			// cost: one node per key whichever tier the key is in, because the
			// recency list and the frequency buckets are threaded through the
			// same slots and a key is in exactly one of them.
			assert_eq!(
				LRU_LFU_COMPACT_HYBRID_EVICTION_STACK_DRAM_OVERHEAD,
				LFU_COMPACT_HYBRID_EVICTION_STACK_DRAM_OVERHEAD,
				"both LFU-ranked stacks are on `ArenaFrequencyChain` and must \
				 carry one term",
			);
		}
	}
}

#[cfg(all(test, feature = "hybrid_cache_common"))]
mod value_resident_factor_applies {
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
		let scaled = resident_value_bytes(raw);

		assert_eq!(
			got, key + scaled + expiry,
			"the value is counted at jemalloc's committed size; the key and \
			 expiry are left alone, since shared_overhead covers them",
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

/// The two tables in this module describe the same structures from two sides:
/// `get_policy_overhead` is added to `base_size` to give what an object is
/// CHARGED against `max_size`, and `get_hybrid_dram_shared_overhead` is what the
/// fast tier RESERVES for that object's metadata. Nothing keeps them in step
/// except this test.
///
/// They were out of step. Every compact hybrid reserved the measured 72 B/object
/// for its eviction stack and charged the hand-counted 40, and `used_size` is
/// what drives eviction, so `max_size` bounded 32 B/object less metadata than
/// the cache actually holds -- 30% of a compact hybrid's per-object metadata,
/// silently.
#[cfg(all(
	test,
	feature = "hybrid_cache_common",
	not(feature = "merged_object_store"),
	not(feature = "eviction_stacks_pmem"),
	not(feature = "global_hashtable_pmem")
))]
mod the_two_overhead_tables_agree {
	use super::*;

	/// Every hybrid policy, parameterised the way the sweep runs them. Written
	/// out rather than derived: a policy added to `PaperPolicy` without being
	/// added here would silently escape the check, so the list is meant to be
	/// read against the two matches above.
	fn every_hybrid_policy() -> Vec<PaperPolicy> {
		vec![
			PaperPolicy::LruCompactHybrid,
			PaperPolicy::LruLazyCopyCompactHybrid,
			PaperPolicy::LfuCompactHybrid,
			PaperPolicy::LruSizedCompactHybrid,
			PaperPolicy::LruLfuCompactHybrid(2),
			PaperPolicy::FifoCompactHybrid,
			PaperPolicy::TwoQCompactHybrid(0.25),
			PaperPolicy::TwoQFastAdmissionCompactHybrid(0.25),
			PaperPolicy::TwoQFastAdmissionReprieveCompactHybrid(0.25),
			PaperPolicy::TwoQFullFastAdmissionCompactHybrid(0.25, 0.5),
			PaperPolicy::TwoQGhostCompactHybrid(0.25),
			PaperPolicy::S3FifoCompactHybrid(0.1),
			PaperPolicy::S3FifoFaithfulCompactHybrid(0.1),
			PaperPolicy::S3FifoFaithfulFastAdmissionCompactHybrid(0.1),
			PaperPolicy::S3FifoFaithfulReprieveCompactHybrid(0.1),
			PaperPolicy::S3FifoFaithfulFastAdmissionReprieveCompactHybrid(0.1),
			PaperPolicy::S3FifoGhostCompactHybrid(0.1),
			PaperPolicy::S3FifoGhostLazyDemotionCompactHybrid(0.1),
			PaperPolicy::S3FifoGhostLazyDemotionFastAdmissionCompactHybrid(0.1),
			PaperPolicy::S3FifoGhostLazyDemotionFastAdmissionMidpointCompactHybrid(0.1),
			PaperPolicy::S3FifoLazyDemotionReprieveCompactHybrid(0.1),
			PaperPolicy::S3FifoLazyDemotionFastAdmissionReprieveCompactHybrid(0.1),
			PaperPolicy::S3FifoLazyDemotionFastAdmissionMidpointReprieveCompactHybrid(0.1),
			PaperPolicy::S3FifoLazyDemotionFastAdmissionSplitSlowReprieveCompactHybrid(0.1),
		]
	}

	/// The charge and the reservation must differ by exactly
	/// `DOUBLE_COUNTED_IN_BASE_SIZE` -- the key and the expiry, which live
	/// inside the map row but which `base_size` already counts separately, so
	/// the charge takes them back off and the reservation does not.
	///
	/// Anything else means the two tables name different structures for the
	/// same policy, which is the failure this exists to catch.
	#[test]
	fn the_two_overhead_tables_agree_on_every_hybrid() {
		if std::env::var_os("PAPER_DISABLE_SHARED_OVERHEAD").is_some() {
			return; // the escape hatch zeroes the reservation by design
		}

		for policy in every_hybrid_policy() {
			let charged = get_policy_overhead(&policy);
			let reserved = get_hybrid_dram_shared_overhead(&policy);

			assert_eq!(
				charged,
				reserved - DOUBLE_COUNTED_IN_BASE_SIZE,
				"{policy}: used_size charges {charged} B/object while the fast \
				 tier reserves {reserved} for the same structures",
			);
		}
	}

	/// The whole list, not just the four the older feature-independence test
	/// samples: a policy left out of `every_hybrid_policy` would make the check
	/// above vacuous for that policy without failing anything.
	#[test]
	fn every_hybrid_policy_is_actually_covered() {
		assert_eq!(
			every_hybrid_policy().len(),
			24,
			"the hybrid policy list has drifted from the 24 arms the two \
			 overhead tables carry",
		);
	}
}
