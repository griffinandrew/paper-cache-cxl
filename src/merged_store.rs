/*
 * Copyright (c) Kia Shakiba
 *
 * This source code is licensed under the GNU AGPLv3 license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! `MergedStore` -- object map, recency order and tier placement in ONE
//! structure.
//!
//! A fourth `ObjectMapRef` shape behind `merged_object_store`. A FEATURE and
//! not a `PaperPolicy` variant because the object store is a compile-time
//! choice here: `ObjectStore::get_ref` returns `impl Deref`, so the trait is
//! not object-safe and `objects` cannot be `dyn`.
//!
//! # Why
//!
//! Measured with `stats.allocated`, R^2 = 1.000000, a tiered LRU object costs
//! three independently-keyed pieces:
//!
//! ```text
//!   DashMap<HashedKey, Object>   96 B   key -> object
//!   Arc<TieredBuffer>            48 B   refcount + buffer handle
//!   LruCompactHybridStack        72 B   key -> slot -> links + payload
//! ```
//!
//! Most of that 72 is two more copies of the key -- one in the stack's slab,
//! one in its index -- present only to answer "where in the recency order is
//! this key, and which tier is it in?". Merging answers both for free: finding
//! the object IS finding its position and its tier.
//!
//! # The index is chained, not a second HashMap
//!
//! The first sharded revision kept a `HashMap<HashedKey, u32, NoHasher>` per
//! shard and measured 251.8 B/object against a DashMap control's 179.1 -- it
//! removed a 72 B/object eviction stack and added 72.7, which is not a saving.
//! The reason was structural: a `HashMap` index does not remove a hash table,
//! it MOVES one. DashMap stores the object in the bucket (41 B/bucket); that
//! design stored a `u32` in a hashbrown bucket (17 B) AND the object in a slab
//! slot, so it paid for both.
//!
//! So the index is now CacheLib's `ChainedHashTable`: a flat `Vec<u32>` of slot
//! ids, one per bucket, with the chain threaded through a `hash_next` field in
//! the slot itself -- CacheLib's `hashHook_`. 4 bytes per bucket instead of 17,
//! no second copy of the key, and no second allocation to keep sized.
//!
//! Chaining rather than open addressing is what makes this composable at all:
//! the chain link lives in a slot that already exists, so the table's marginal
//! cost is the bucket array alone. Load factor 1.0 -- mean chain length 1 --
//! rather than hashbrown's 7/8, because a chain does not degrade at high load
//! the way a probe sequence does.
//!
//! # Sharding, and memcached's two tricks
//!
//! The first revision used one `RwLock` over the whole store, on the reasoning
//! that sharding forces per-shard recency lists and so approximate order. That
//! measured +3.6% GET and +25.4% SET against DashMap on standard_web, and the
//! reasoning was only half right.
//!
//! memcached does not solve this with better locking -- it avoids the work.
//! `item_lock(hv)` is an array of mutexes indexed by hash, and
//! `ITEM_UPDATE_INTERVAL` makes `item_update` early-return unless the item is
//! older than the interval, so a key hit a million times takes the LRU lock
//! ONCE. The lock is rare because it is skipped, not because it is cheap.
//!
//! Both are here. `SHARDS` independent `{index, slab, list}` regions, and
//! `MERGED_UPDATE_INTERVAL` counted in ACCESSES rather than seconds --
//! memcached's 60 s is calibrated to wall clock, and a trace replaying 300M
//! records as fast as it can would barely reorder at all under it. The default
//! is 0, exact behaviour, so the effect is measured rather than assumed.
//!
//! ## Sharding on the HIGH bits
//!
//! The key IS the hash -- the store is `NoHasher`d -- and a bucket is selected
//! with the LOW bits of it. Sharding on the low bits too would hold them
//! constant inside a shard and drive every key in it to one bucket. Shard on
//! the high bits, which is what DashMap does and why DashMap and `NoHasher`
//! compose in the first place.
//!
//! ## Sharding without losing the order
//!
//! Each shard's list is recency-ordered within itself, so the globally
//! least-recently-used object is the MINIMUM over the shard tails. Every slot
//! carries the access counter it was last relinked at, and each shard mirrors
//! its tail's counter in a padded `AtomicU64` -- so choosing an eviction victim
//! is `SHARDS` atomic loads and then ONE shard lock, never a global lock.
//!
//! At interval 0 that is exact LRU across shards. Above 0 the order is
//! quantised by the interval, which is exactly the trade memcached makes and
//! the only approximation here.
//!
//! # More than one order
//!
//! The list above is a RECENCY list only because of what a hit does to it. Make
//! a hit do nothing and the identical structure is a FIFO queue: `last_access`
//! stops being "the stamp of the last relink" and becomes "the stamp of the
//! insert", which is precisely the ordering key a FIFO victim is chosen by. So
//! [`MergedOrder`] costs no bytes -- not one per slot, not one per shard -- and
//! `tail_key`, `fast_boundary`, the mirrors and the settle loop are all
//! untouched by it. See [`MergedOrder`] for why the choice is a runtime field.
//!
//! CLOCK is that same FIFO queue plus a second chance, and it is the order this
//! design is actually FOR. Under LRU a hit relinks, which means the policy
//! worker takes a shard WRITE lock on every hit -- measured as the merged
//! store degrading to 1.24x the DashMap arm's service time at sixteen clients.
//! A CLOCK hit takes the READ lock and does one relaxed store into a reference
//! bit, and the second chance that bit buys is paid for later, by the eviction
//! path, which was holding the write lock anyway. The bit is one byte in the
//! slot's existing TAIL PADDING, so CLOCK costs no bytes either: the slot is
//! still exactly 40.
//!
//! # Tiering
//!
//! Ported from [`LruCompactHybridStack`], whose structure this can reproduce
//! exactly: ONE recency list plus a `fast_boundary` cursor at the
//! least-recently-used FAST slot. Everything from the head up to and including
//! the boundary is fast; everything after it is slow. So a demotion is a
//! one-step walk of the cursor and never touches the list, which is why tier
//! placement here can never reorder and never evict.
//!
//! ## The tier boundary is global, by the same trick eviction uses
//!
//! An earlier revision split `fast_capacity` EVENLY across the shards and let
//! each settle its own boundary. That is not a global order: a shard holding
//! large recent objects demoted things MORE RECENT than fast objects idling in
//! another shard, and the fast tier went under-used whenever recency skewed
//! across shards.
//!
//! So the boundary is chosen the way the eviction victim is. Each shard
//! mirrors the stamp of its `fast_boundary` slot into a second padded array,
//! `fast_tails`, and the store keeps one `fast_used` total. Settling runs with
//! NO shard lock held: while the total is over the high watermark, `SHARDS`
//! relaxed loads name the shard whose boundary is oldest, that ONE shard's
//! write lock is taken, its boundary steps back one slot, the mirror is
//! republished and the lock released -- repeat until under the low watermark.
//!
//! That is exact, not approximate: every shard's fast set is a contiguous MRU
//! PREFIX of its own list, so the oldest boundary across shards IS the globally
//! least-recently-used fast object. (The prefix property is what the boundary
//! needs, and it does not depend on the order: under `MergedOrder::Fifo` the
//! fast set is the newest-INSERTED prefix, held by the same cursor, and the
//! oldest boundary is the global FIFO demotion victim.) It cannot deadlock,
//! because a toucher releases its own shard before settling and no thread ever
//! holds two shard locks. Concurrent settlers may each demote one extra object,
//! which the 0.98/0.95 hysteresis absorbs.
//!
//! Migrations accumulate per shard and `drain_tier_migrations` concatenates
//! them, so `PolicyWorker::apply_tier_migrations` performs the physical
//! `Object::set_data` moves exactly as it does for every other hybrid stack.
//!
//! # Slot recycling
//!
//! A freed slot goes on `free` and is reused whole. The `generation` counter an
//! earlier revision carried is gone: it guarded against a stale `u32` outliving
//! its slot, and with the index chained there is no `u32` handed out to anyone
//! -- `MergedRef` holds one only for as long as it holds the shard guard, which
//! is exactly as long as the slot cannot be recycled.
//!
//! # Nothing under the shard lock is O(n)
//!
//! Two paths used to be, and both stalled the API THREAD, since in this design
//! the API thread is what inserts:
//!
//!   * the slab was one `Vec<Slot>` per shard, so a growth `realloc`ed and
//!     copied every live slot;
//!   * `grow_buckets` doubled the index and re-threaded every chain by walking
//!     the WHOLE recency list.
//!
//! The slab is now a `Vec` of fixed 4096-slot CHUNKS. Growth appends a chunk;
//! nothing is copied, no slot ever moves, and a slot id is just
//! `chunk << 12 | offset`. At 40 B a slot that is 160 KiB per chunk, so an
//! empty 32-shard store costs ~5 MiB rather than the 80 MiB a 64K chunk would.
//!
//! The index grows by LINEAR HASHING: a `split` cursor and two masks, one
//! bucket rehashed per insert past the load factor. The cost of a growth step
//! is one bucket's chain -- mean length 1 -- instead of the entire list, so
//! there is no stall to move off the API thread in the first place.

use std::{
	collections::HashMap,
	ops::{Deref, DerefMut},
	sync::{
		atomic::{AtomicBool, AtomicU8, AtomicU32, AtomicU64, AtomicUsize, Ordering},
		RwLock, RwLockReadGuard, RwLockWriteGuard,
	},
};

use crate::{
	error::CacheError,
	object::{Object, ObjectSize},
	worker::Tier,
	CacheSize, HashedKey, NoHasher, PaperPolicy,
};

/// `frequency -> (head, tail)` of that bucket's intrusive list, one map per
/// tier per shard, under [`MergedOrder::Lfu`] only.
///
/// Ordered, so a tier's minimum frequency is its first entry -- the same
/// structure, and the same reason, as `ArenaFrequencyChain`'s `fast_buckets` /
/// `slow_buckets`. One entry per DISTINCT frequency rather than per object, so
/// it does not scale with the cache: `BTreeMap::new()` does not allocate, so
/// under the other three orders a shard carries two empty maps and nothing
/// else.
///
/// NOT allocator-gated on `eviction_stacks_pmem`, unlike the split path's
/// identical maps. That gate exists because
/// `get_hybrid_dram_shared_overhead` drops the eviction-stack DRAM charge to
/// zero when the stack is supposed to be on the far node -- but under
/// `merged_object_store` that function ignores the feature entirely and
/// charges `MERGED_STORE_STRUCTURE_OVERHEAD`, and this store's slab, bucket
/// array and free list are all plain DRAM allocations. Gating only these two
/// maps would put one structure of the merged store on the far node while the
/// slot it points into stayed in DRAM, and would be charged as neither.
type FreqBuckets = std::collections::BTreeMap<u16, (u32, u32)>;

const NIL: u32 = u32::MAX;

/// Which eviction order the store imposes on its OWN link structure.
///
/// The store is the eviction stack, so "which policy" is not a second
/// structure to swap out -- it is a rule about what a HIT does to the list, and
/// nothing else. LRU relinks to the front and promotes; FIFO does nothing at
/// all. That is the entire difference, and it is why FIFO needs no extra byte
/// per slot and no extra mirror: `last_access` is already stamped at insert, so
/// under FIFO it simply IS the insertion stamp, and `tail_key`'s
/// minimum-over-shard-tails already yields the exact global FIFO victim.
///
/// # Why a runtime field and not a `cfg`
///
/// The server picks its policy from `--policy` at startup, so one binary has to
/// be able to run any of them; a `cfg` would mean one binary per policy and an
/// A/B between two policies would then also be an A/B between two builds --
/// which this project has already measured as a ~1.7% shift on binary layout
/// alone. The field is written once, before the cache serves a request, and
/// read on every hit: a branch on a value that never changes is predicted
/// perfectly, and it sits next to a shard lock acquisition in any case.
///
/// FIFO was the first of several wanted here -- SIEVE, 2Q and S3-FIFO are still
/// on the list -- so adding one is adding a variant and an arm at the sites
/// below. CLOCK was the second, and it is the one that pays for the seam: it is
/// the only order here whose hit path does not take the shard WRITE lock.
///
/// LFU was the third, and it is the one that shows the seam is not free. The
/// first three orders are all the SAME structure -- one list per shard, ordered
/// by one monotonic stamp -- so each was a variant and a handful of arms. LFU
/// is not: a frequency count is not monotonic and is not unique, so "the
/// minimum over 32 shard tails" does not name its victim. It brings two
/// `BTreeMap` bucket sets per shard (empty, and therefore free, under the other
/// three orders), a second word in each shard's mirror, and its own meaning for
/// `prev`/`next` -- which are bucket chains under `Lfu` and a recency list
/// under everything else. See [`MergedOrder::Lfu`].
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
#[repr(u8)]
pub enum MergedOrder {
	/// A hit moves the slot to the MRU end, restamps `last_access` and
	/// promotes it to the fast tier if it was slow.
	Lru = 0,

	/// A hit does NOTHING: insertion order is eviction order. `last_access` is
	/// written once, at insert, and never again -- including on an overwrite,
	/// which resizes in place and keeps the object's original queue position.
	Fifo = 1,

	/// FIFO plus a second chance: a hit sets the slot's REFERENCE BIT and does
	/// nothing else, and the eviction hand clears that bit and recycles the
	/// slot to the head instead of evicting it.
	///
	/// # Where the hand is
	///
	/// There is no hand cursor, because the list already is one. `tail_key`
	/// nominates the globally oldest slot, which is exactly where a circular
	/// CLOCK's hand would be pointing; a slot that gets its second chance is
	/// relinked to the head, which is exactly where a circular CLOCK's hand
	/// would next reach it -- one full revolution away. That is the standard
	/// linked-list rendering of CLOCK, and it is what the flat
	/// `ClockCompactStack` and `ClockStack` in this tree both implement:
	/// `pop_back`, and on a set bit `push_front` with the bit cleared.
	///
	/// It is NOT SIEVE. SIEVE leaves the second-chance object where it is and
	/// advances a separate hand, so the object keeps its place in insertion
	/// order; CLOCK moves it to the front. The two are different policies and
	/// the flat stack this must match implements the second.
	///
	/// # Why the second chance promotes
	///
	/// `fast_boundary` requires the fast set to be a contiguous PREFIX of its
	/// shard's list. A second chance moves the slot to the head, which IS in
	/// that prefix, so the slot has to become fast in the same step -- leaving
	/// it slow at the head would put a slow object in front of a fast one and
	/// break the cursor. So the second chance is `touch_slot`, the same relink
	/// -restamp-promote an LRU hit performs, moved off the hit path and onto
	/// the eviction path where the write lock is already held.
	Clock = 2,

	/// Least-frequently-used, EXACTLY -- and the only order here that is not
	/// the one recency list wearing a different rule.
	///
	/// # What a hit does
	///
	/// Unlinks the slot from its frequency bucket, increments `Slot::freq`,
	/// restamps `last_access` and relinks it at the TAIL of the next bucket. A
	/// slow key whose new count STRICTLY exceeds the fast tier's global minimum
	/// is promoted in the same step, which is
	/// `LfuCompactHybridStack::maybe_promote`'s rule verbatim.
	///
	/// So the hit path takes the shard WRITE lock. There is no read-lock
	/// formulation that leaves the buckets truthful, and stale buckets mean a
	/// wrong minimum, which is to say not LFU -- so this order does NOT inherit
	/// CLOCK's win, and should be expected to measure like `Lru` on service
	/// time rather than like `Clock`. A "lazy" variant that incremented an
	/// atomic under the read lock and re-bucketed at eviction time would be a
	/// different policy, with stale minima, and must not be run under this
	/// label.
	///
	/// # Where the order lives
	///
	/// NOT in `head`/`tail`, which stay `NIL` for a shard's whole life under
	/// this order. `prev`/`next` are instead the chains of the per-tier
	/// frequency buckets (`Inner::fast_buckets` / `slow_buckets`), which is
	/// sound because a key is in exactly one bucket of one tier at a time --
	/// the same argument `ArenaFrequencyChain` makes for sharing one link pair
	/// between its recency list and its slow buckets. So LFU costs no link
	/// bytes, and `Slot::freq` fits the slot's remaining tail padding: the
	/// whole order is free per slot, exactly as FIFO and CLOCK were.
	///
	/// # How 32 shards still give the EXACT global order
	///
	/// By the same argument as the other three, one level down. The
	/// load-bearing part of "the global LRU object is some shard's tail" is not
	/// the stamp -- it is that each shard is ordered WITHIN ITSELF, so a global
	/// extremum is necessarily one of 32 per-shard extrema. Keep each shard's
	/// keys frequency-bucketed per tier and the globally least-frequent object
	/// is necessarily some shard's own least-frequent object. So the victim is a
	/// lexicographic minimum over 32 mirrored `(freq, stamp)` pairs and then ONE
	/// shard lock -- the same shape as today, with no global frequency
	/// structure and no cross-shard lock. Per-shard approximation was the
	/// alternative and it is not taken: it would change which key is evicted,
	/// hence the miss ratio, so the merged arm would no longer differ from the
	/// DashMap arm by the object store alone -- and the confound would be
	/// invisible in the results, which is the same failure the deleted LRU
	/// fallback produced.
	///
	/// # The tie-break, which is where exactness is won or lost
	///
	/// Many keys share a count, so the secondary key is `last_access`, read
	/// under this order as "the stamp at which this key entered its CURRENT
	/// bucket" -- it is restamped on every bump and on every tier move, which
	/// is exactly when the bucket changes. Earliest entrant leaves first, and
	/// both references agree on that from opposite ends: `ArenaFrequencyChain`
	/// appends at the bucket tail and takes `min_key` from its head, while the
	/// flat `LfuCompactStack` pushes at the head and evicts the tail.
	///
	/// # Eviction prefers SLOW, and that is not free here
	///
	/// Under the other three orders "globally oldest" IS "oldest slow object"
	/// whenever anything is slow, because the fast set is the newest prefix --
	/// so `tail_key` got slow-preference without anyone having to write it.
	/// Under LFU it does not: a freshly admitted key sits at frequency 1 while
	/// demoted keys sit higher, so the global minimum is very often a FAST
	/// newcomer. `LfuCompactHybridStack::evict_one` checks the slow chain first
	/// and falls back to fast, and `lfu_victim` has to do the same
	/// deliberately. Missing it would quietly evict hot new keys and report a
	/// worse miss ratio that looked like a property of the merged store.
	Lfu = 3,
}

impl MergedOrder {
	/// The order `policy` asks for, or [`CacheError::PolicyNotImplemented`]
	/// when the merged store does not implement that policy's order at all.
	///
	/// **The error is the contract.** It does not mean LRU and it does not mean
	/// "warn and carry on": it means the caller was asked for something this
	/// build cannot do, and the cache must fail to construct rather than run a
	/// different policy under the requested name. This function used to return
	/// `Option` and its own doc already said as much -- "`None` does not mean
	/// LRU. It means the caller is about to run something other than what it
	/// was asked for and has to say so" -- but saying so was left to a
	/// `eprintln!` at the one caller, which then ran LRU anyway. Now it is
	/// enforced, so there is no path on which a merged build serves a policy it
	/// is not honouring.
	///
	/// A policy's flat and hybrid spellings map to the same order: the tier
	/// boundary is settled separately, by byte budget, and does not change what
	/// a hit does to the queue.
	///
	/// `PaperPolicy::Auto` is deliberately NOT here. A merged build cannot
	/// honour the *auto* part at all -- `handle_policy` refuses every switch
	/// under `merged_object_store`, before it ever reaches the `is_auto` check
	/// -- so resolving it to some fixed order would be exactly the silent
	/// mislabelling this error exists to delete.
	pub fn from_policy(policy: &PaperPolicy) -> Result<MergedOrder, CacheError> {
		match policy {
			PaperPolicy::Lru
			| PaperPolicy::LruCompact
			| PaperPolicy::LruCompactHybrid => Ok(MergedOrder::Lru),

			PaperPolicy::Fifo
			| PaperPolicy::FifoCompact
			| PaperPolicy::FifoCompactHybrid => Ok(MergedOrder::Fifo),

			PaperPolicy::Clock
			| PaperPolicy::ClockCompact
			| PaperPolicy::ClockCompactHybrid => Ok(MergedOrder::Clock),

			PaperPolicy::Lfu
			| PaperPolicy::LfuCompact
			| PaperPolicy::LfuCompactHybrid => Ok(MergedOrder::Lfu),

			other => Err(CacheError::PolicyNotImplemented(*other)),
		}
	}

	/// Decodes the `AtomicU8` the store keeps the order in.
	///
	/// Exhaustive over the reprs that exist, with `unreachable!` rather than a
	/// catch-all. The catch-all this replaces was `_ => MergedOrder::Lru`, and
	/// it was the one site in the whole change that a new variant could reach
	/// WITHOUT a compile error: a stored repr of 3 decoded as LRU, on the hit
	/// path, via `order()`. A fourth order that silently ran as the first is
	/// the precise failure this commit exists to remove, so the fallback is
	/// gone from here too.
	#[inline]
	fn from_repr(v: u8) -> MergedOrder {
		match v {
			0 => MergedOrder::Lru,
			1 => MergedOrder::Fifo,
			2 => MergedOrder::Clock,
			3 => MergedOrder::Lfu,
			other => unreachable!("MergedOrder repr {other} was never stored"),
		}
	}
}

/// 32 shards on an 8-core box matches DashMap's own `4 * ncpus` default, so
/// the comparison against it is like-for-like.
const SHARD_BITS: u32 = 5;
const SHARDS: usize = 1 << SHARD_BITS;

/// `MergedStore::dirty_migrations` carries one bit per shard in a `u32`, so the
/// shard count has to fit in one. At `SHARD_BITS = 5` it is exactly 32.
const _: () = assert!(
	SHARDS <= u32::BITS as usize,
	"SHARDS no longer fits the u32 dirty-shard mask -- widen dirty_migrations \
	 along with SHARD_BITS",
);

/// Sharded on the HIGH bits -- see the module doc. The store is `NoHasher`d, so
/// low-bit sharding would collapse each shard onto one hashbrown bucket.
#[inline]
fn shard_of(key: HashedKey) -> usize {
	(key >> (64 - SHARD_BITS)) as usize
}

/// A shard tail's access counter, on its own cache line.
///
/// Unpadded these 32 counters share four lines, and every relink of any shard's
/// tail would invalidate the line under seven other shards -- false sharing on
/// precisely the value that exists to be read without a lock.
#[repr(align(64))]
struct TailSeq {
	/// `Lru`/`Fifo`/`Clock`: the shard tail's `last_access`. `Lfu`: the
	/// `last_access` of the HEAD of this tier's lowest-frequency bucket, which
	/// is the stamp that separates two shards tied at the same frequency.
	seq: AtomicU64,

	/// `Lfu` only: this tier's minimum frequency, `EMPTY_TAIL` when the tier
	/// holds nothing in this shard. Never read under the other three orders.
	///
	/// In the SAME cache line as `seq`, deliberately: the victim is a
	/// lexicographic minimum over `(freq, seq)`, so widening the mirror rather
	/// than adding a second array keeps the whole choice to the same 32 lines
	/// this already touched, and a shard still occupies exactly one line.
	///
	/// Not packed into ONE `u64` with the stamp, though it would make the
	/// choice a single pass: that means truncating the stamp to 48 bits, and
	/// `last_access` was deliberately widened from `u32` for precisely this
	/// class of bug -- "exact only while no live slot goes un-relinked for 2^32
	/// accesses ... silently wrong when reached". 2^48 is out of reach here,
	/// but reintroducing a bounded-but-silent truncation is the wrong trade to
	/// make for one pass over warm cache lines.
	freq: AtomicU64,
}

/// A shard with nothing in it, distinguishable from a real `last_access` of 0.
const EMPTY_TAIL: u64 = u64::MAX;

/// Buckets a fresh shard starts with. 32 shards x 16 x 4 B = 2 KB of baseline,
/// and the table doubles from there.
const INITIAL_BUCKETS: usize = 16;

/// Slots per slab chunk, and the width of a slot id's offset field.
///
/// The slab is a `Vec` of these chunks, so growth APPENDS one and nothing is
/// ever copied or reallocated -- the `Vec<Slot>` this replaced `realloc`ed
/// under the shard lock, on the API thread, which is the stall the chunking
/// exists to remove. A slot id is `chunk << SLAB_CHUNK_BITS | offset`, which
/// for a linearly allocated slab is just the slot's ordinal.
///
/// 4096 is picked from both ends. At 40 B a slot that is 160 KiB per chunk, so
/// 32 shards start at ~5 MiB of committed slab -- a 64K-slot chunk would make
/// an EMPTY store cost 80 MiB. And the worst-case waste is one partly-filled
/// chunk per shard, 4095 slots, bounded and independent of how large the cache
/// grows, where `Vec` doubling measured 1.40x the live count and 25% growth
/// steps 1.10x -- both proportional, both unbounded.
const SLAB_CHUNK_BITS: u32 = 12;
const SLAB_CHUNK: usize = 1 << SLAB_CHUNK_BITS;
const SLAB_OFFSET_MASK: usize = SLAB_CHUNK - 1;

/// Live entries per bucket before the table doubles. 1.0, not hashbrown's 7/8:
/// a chain degrades linearly with load where a probe sequence degrades sharply,
/// so chaining can be run full and spend the difference on memory instead.
const MAX_LOAD_NUMER: usize = 1;
const MAX_LOAD_DENOM: usize = 1;

struct Slot<K, V> {
	/// `Option` so `take` can move the object out without disturbing the slab.
	/// Free: the `NonNull` inside the object's `TieredValue` is the niche that
	/// absorbs the discriminant, so `Option<Object>` is the same size as
	/// `Object` -- 24 bytes. `object/mod.rs::layout` asserts it.
	object: Option<Object<K, V>>,
	/// The store is keyed by hash; `Object::key()` is the real key, which
	/// `key_matches` needs to make collisions safe.
	hashed: HashedKey,
	prev: u32,
	next: u32,
	/// Next slot in this key's hash-bucket chain, `NIL` at the end --
	/// CacheLib's `hashHook_`. Threading the chain through the slot is what
	/// lets the index cost 4 bytes a bucket instead of a whole hashbrown row.
	hash_next: u32,
	/// The store clock at the last relink, FULL WIDTH. Double duty: the
	/// update-interval check, and the cross-shard comparison that keeps
	/// eviction and the tier boundary picking the true LRU slot.
	///
	/// It was a truncated `u32` and compared as a wrapping difference, which
	/// is exact only while no live slot goes un-relinked for 2^32 accesses.
	/// That is four billion -- reachable by a long replay of a 306M-record
	/// trace, and silently wrong when reached. At 64 bits, one relaxed
	/// `fetch_add` per relink, a billion accesses a second would take 584
	/// years to wrap, so the stamps are simply ordered and compared as such.
	last_access: u64,
	tier: Tier,

	/// CLOCK's reference bit -- see [`MergedOrder::Clock`]. 1 after a hit,
	/// cleared by the hand when it grants the second chance.
	///
	/// FREE, and that is the point. Bytes 37-39 of this slot were pure tail
	/// padding, so this field costs nothing: the slot still measures exactly
	/// 40 and both asserts below still hold. Meaningless under `Lru` and
	/// `Fifo`, which never read it.
	///
	/// An `AtomicU8` rather than a `bool` because the whole reason CLOCK is
	/// here is that its hit path must not take the shard WRITE lock: the hit
	/// writes this field through a `&Slot` obtained under the READ lock, which
	/// only an atomic makes sound. Relaxed on both sides -- the bit is a hint,
	/// a lost update costs one object one second chance, and there is no other
	/// datum whose visibility is being ordered against it.
	referenced: AtomicU8,

	/// LFU's access count -- see [`MergedOrder::Lfu`]. 1 at admission, then
	/// `saturating_add(1)` per hit. Meaningless under the other three orders,
	/// which never read it.
	///
	/// FREE, like `referenced` before it and for the same reason: `tier` sits
	/// at offset 36 and `referenced` at 37, so bytes 38-39 were the last of the
	/// slot's tail padding and a 2-byte-aligned `u16` fits them exactly. The
	/// fields become 40, `size_of` stays 40, and both asserts below still hold.
	///
	/// # Why `u16` and not `u8`, and not `u32`
	///
	/// A `u8` caps at 255, which is reachable by ordinary warm keys on these
	/// traces, and saturation there would collapse the fast set's internal
	/// ordering -- promotion compares against the fast tier's MINIMUM and
	/// demotion takes it, so a pile-up at the ceiling changes tier placement.
	/// The S3-FIFO family's 0..=3 counter is not a precedent for that: it is a
	/// reinsertion counter, not a ranking key.
	///
	/// A `u32` -- the split path's width (`NodePayload::freq`) -- would take
	/// the fields to 42 and the slot to 48, breaking both asserts by design,
	/// costing 8 B per slot (56.8 MB at 7.1M objects, 582.4 MB at 72.8M), and
	/// invalidating `MERGED_STORE_STRUCTURE_OVERHEAD`. That last one is the
	/// real objection: under `merged_object_store`
	/// `get_hybrid_dram_shared_overhead` ignores the policy, so a re-measured
	/// constant would shrink the effective fast tier for LRU, FIFO and CLOCK
	/// too and make the already-completed comparison matrix incomparable.
	///
	/// At `u16` the cap is 65,535 and it cannot change a victim choice:
	/// victims are drawn from the MINIMUM, so keys pinned at the ceiling are
	/// the last things ever evicted and saturation reorders only the hottest
	/// keys relative to each other, after everything colder has gone. That is
	/// a documented divergence from the reference's `u32`, not a hidden one.
	///
	/// A plain `u16` rather than an atomic: unlike CLOCK's bit, every write to
	/// this happens under the shard WRITE lock, so there is no
	/// `&Slot`-under-a-read-lock soundness requirement to satisfy.
	freq: u16,
}

/// 8 object + 8 hashed + 4 prev + 4 next + 4 hash_next + 8 last_access +
/// 1 tier + 1 referenced + 2 freq = 40, exactly filling the padding the
/// object's 8-byte alignment imposes. `size_of` is unchanged at 40.
///
/// `referenced` and `freq` both went into that padding: the slot measured 40
/// with 37 bytes of fields before either existed and measures 40 with both.
/// CLOCK's reference bit and LFU's counter are genuinely free, which is the
/// claim the EXACT assert below is guarding -- and the padding is now FULL, so
/// the next order to want per-slot state will grow the slot and trip it.
///
/// It was 56, with a 24-byte `Object` holding the key, the value pointer, the
/// length and the expiry inline. Those three moved into the refcounted value
/// header, leaving the `Object` as one pointer -- see `crate::value`. `size`
/// (u32) and `dram_resident` (u8) used to sit here too and are gone: both are
/// derived from the object, which reaches the value's length already -- see
/// `Slot::migrating`.
///
/// The `<= 40` bound is the claim being made against the split design; the
/// EXACT assert is what catches a field silently landing in the padding and
/// then, later, pushing the slot over.
///
/// `MERGED_STORE_STRUCTURE_OVERHEAD` was re-measured against THIS slot with
/// `merged_store::measure::measure_merged_store_point` and is 46, from a fitted
/// 45.2966 B/object structural (R^2 = 0.999999, and identical under both value
/// layouts). It was 62, fitted against the 56-byte slot. The 16 bytes this slot
/// lost did not simply come off that figure -- both fill factors multiply the
/// slot -- which is why it was re-measured rather than adjusted.
///
/// The exact assert below is what makes that number falsifiable: change the
/// slot and the build stops, rather than quietly charging a stale constant.
const _: () = assert!(
	core::mem::size_of::<Slot<u64, std::sync::Arc<[u8]>>>() <= 40,
	"Slot grew past 40 bytes -- the whole point is that it is smaller than a \
	 DashMap row plus an eviction-stack row",
);

const _: () = assert!(
	core::mem::size_of::<Slot<u64, std::sync::Arc<[u8]>>>() == 40,
	"Slot is no longer exactly 40 bytes -- re-measure \
	 MERGED_STORE_STRUCTURE_OVERHEAD before changing this number",
);

impl<K, V> Slot<K, V> {
	/// A recycled or never-used slot. Linked nowhere, holding nothing.
	fn empty() -> Self {
		Slot {
			object: None,
			hashed: 0,
			prev: NIL,
			next: NIL,
			hash_next: NIL,
			last_access: 0,
			tier: Tier::Fast,
			referenced: AtomicU8::new(0),
			freq: 0,
		}
	}

	/// Bytes that actually move between tiers. `Object::set_data` migrates the
	/// value buffer alone, so the key, the expiry field and the `Expiries` row
	/// stay in DRAM in either tier and must not be charged to a tier's budget.
	///
	/// DERIVED, not stored. It used to be `size - dram_resident`, two fields
	/// filled in by the worker one event after the API thread inserted the
	/// object -- so a freshly inserted object was accounted as ZERO bytes until
	/// the worker caught up. The object carries its value's length, and
	/// `size - dram_resident` was by construction the object's own allocation:
	/// `base_size` is `key + value + expiry (+ ttl)` and `dram_resident_size`
	/// is the same sum without the value.
	///
	/// It calls `resident_object_bytes` -- the SAME accessor `base_size` calls,
	/// and deliberately not a second rounding of `data_size()`. The two used to
	/// round `nallocx(len)` independently, which was right under the split
	/// layout and wrong under `fused_value` and `thin_header`, where the item
	/// is `bytes_offset::<K>() + len` and the whole of it travels. One accessor is
	/// what stops a third caller repeating the mistake.
	fn migrating(&self) -> CacheSize {
		match &self.object {
			Some(object) => {
				crate::object::overhead::resident_object_bytes::<K>(object.data_size())
					as CacheSize
			},

			None => 0,
		}
	}
}

/// The chunked slab: a `Vec` of fixed-size chunks, indexed as one flat array.
///
/// `Index`/`IndexMut` on the slot id keep every call site reading like the
/// `Vec<Slot>` this replaced, which is the point -- the change is where the
/// memory comes from, not how slots are reached. Growth appends a chunk, so a
/// slot's address is stable for as long as the slot lives and no `realloc`
/// ever runs under the shard lock.
struct Slab<K, V> {
	chunks: Vec<Box<[Slot<K, V>; SLAB_CHUNK]>>,

	/// Slot ids ever handed out. Ids are allocated linearly, so this is both
	/// the next id and the high-water mark; recycled ids come off `Inner::free`
	/// and never move it.
	allocated: usize,
}

impl<K, V> Slab<K, V> {
	fn new() -> Self {
		Slab {
			chunks: Vec::new(),
			allocated: 0,
		}
	}

	/// Slots handed out so far.
	#[cfg(test)]
	fn len(&self) -> usize {
		self.allocated
	}

	/// Slots the committed chunks can hold. What `capacities()` reports as the
	/// slab's cost, since a chunk is committed whole.
	fn capacity(&self) -> usize {
		self.chunks.len() * SLAB_CHUNK
	}

	/// Takes the next id, appending a chunk when the last one is full.
	fn alloc(&mut self, fresh: Slot<K, V>) -> u32 {
		if self.allocated == self.capacity() {
			self.push_chunk();
		}

		let i = self.allocated as u32;
		self.allocated += 1;
		self[i as usize] = fresh;

		i
	}

	/// Built through a `Vec` rather than as an array literal: a
	/// `Box::new([Slot::empty(); N])` would materialise 224 KiB on the STACK
	/// first and only then move it to the heap, which is a stack overflow
	/// waiting for a thread with a small stack.
	fn push_chunk(&mut self) {
		let mut slots = Vec::with_capacity(SLAB_CHUNK);

		for _ in 0..SLAB_CHUNK {
			slots.push(Slot::empty());
		}

		let chunk: Box<[Slot<K, V>; SLAB_CHUNK]> = slots
			.into_boxed_slice()
			.try_into()
			.unwrap_or_else(|_| unreachable!("built with exactly SLAB_CHUNK slots"));

		self.chunks.push(chunk);
	}

	fn clear(&mut self) {
		self.chunks.clear();
		self.allocated = 0;
	}
}

impl<K, V> std::ops::Index<usize> for Slab<K, V> {
	type Output = Slot<K, V>;

	#[inline]
	fn index(&self, i: usize) -> &Slot<K, V> {
		debug_assert!(i < self.allocated, "slot id {i} was never handed out");

		&self.chunks[i >> SLAB_CHUNK_BITS][i & SLAB_OFFSET_MASK]
	}
}

impl<K, V> std::ops::IndexMut<usize> for Slab<K, V> {
	#[inline]
	fn index_mut(&mut self, i: usize) -> &mut Slot<K, V> {
		debug_assert!(i < self.allocated, "slot id {i} was never handed out");

		&mut self.chunks[i >> SLAB_CHUNK_BITS][i & SLAB_OFFSET_MASK]
	}
}

struct Inner<K, V> {
	/// One slot id per bucket, `NIL` when empty; the rest of the chain is in
	/// each slot's `hash_next`. `base + split` long -- NOT a power of two, see
	/// `bucket_of`.
	buckets: Vec<u32>,
	/// The power of two the low mask is taken against. `buckets.len()` sits in
	/// `[base, 2 * base)`.
	base: usize,
	/// The next bucket to split. Buckets below it have already been split this
	/// round and so are addressed with the WIDE mask.
	split: usize,
	/// Live entries, which `buckets.len()` is grown to keep up with. Not
	/// derivable from `slots.len()`, which counts recycled slots too.
	live: usize,
	slots: Slab<K, V>,
	free: Vec<u32>,

	/// Test-only: chain links walked. The linear-hashing claim -- an insert
	/// walks its own bucket's chain plus, at most, the one bucket it splits --
	/// is a cost claim, so it is asserted against a count rather than argued
	/// from the code shape.
	#[cfg(test)]
	walk: AtomicUsize,

	/// MRU end.
	head: u32,
	/// LRU end.
	tail: u32,

	/// The least-recently-used FAST slot: head..=fast_boundary is the fast
	/// tier, everything after it is slow. `NIL` when nothing is fast.
	fast_boundary: u32,

	fast_used: CacheSize,
	slow_used: CacheSize,
	fast_count: usize,

	/// `MergedOrder::Lfu` only: `frequency -> (head, tail)` per tier, with the
	/// chains threaded through the slots' own `prev`/`next`. See
	/// [`FreqBuckets`] and [`MergedOrder::Lfu`].
	///
	/// Deliberately NOT `cfg`'d. The order is a runtime field so that a policy
	/// A/B is not also a binary A/B -- this project has measured binary layout
	/// alone as a ~1.7% shift -- and `BTreeMap::new()` does not allocate, so
	/// under the other three orders these are two empty maps per shard, 64 in
	/// total, and nothing else. Under `Lfu` they are O(distinct frequencies per
	/// shard per tier), not O(objects), which is the property the split path
	/// measured when `lfu-compact-hybrid` came out at the same per-object cost
	/// as `lru-compact-hybrid` (40.2100 both, at 2^20).
	fast_buckets: FreqBuckets,
	slow_buckets: FreqBuckets,

	migrations: Vec<(HashedKey, Tier)>,
}

impl<K, V> Inner<K, V> {
	fn new() -> Self {
		Inner {
			buckets: vec![NIL; INITIAL_BUCKETS],
			base: INITIAL_BUCKETS,
			split: 0,
			live: 0,
			slots: Slab::new(),
			free: Vec::new(),

			#[cfg(test)]
			walk: AtomicUsize::new(0),
			head: NIL,
			tail: NIL,
			fast_boundary: NIL,
			fast_used: 0,
			slow_used: 0,
			fast_count: 0,
			fast_buckets: FreqBuckets::new(),
			slow_buckets: FreqBuckets::new(),
			migrations: Vec::new(),
		}
	}

	/// The LOW bits pick the bucket; the shard already took the high ones, so
	/// the two selections are independent and every bucket stays reachable.
	///
	/// Two masks, because the table grows one bucket at a time: buckets below
	/// `split` have already been re-partitioned this round and answer to the
	/// WIDE mask, the rest still answer to the narrow one. Litwin's linear
	/// hashing, and the reason the table can be a length that is not a power
	/// of two.
	#[inline]
	fn bucket_of(&self, key: HashedKey) -> usize {
		let b = (key as usize) & (self.base - 1);

		match b < self.split {
			true => (key as usize) & (self.base * 2 - 1),
			false => b,
		}
	}

	/// Walk one bucket chain. Mean length 1 at load factor 1.0.
	#[inline]
	fn find(&self, key: HashedKey) -> Option<u32> {
		let mut i = self.buckets[self.bucket_of(key)];

		while i != NIL {
			#[cfg(test)]
			self.walk.fetch_add(1, Ordering::Relaxed);

			let slot = &self.slots[i as usize];

			if slot.hashed == key {
				return Some(i);
			}

			i = slot.hash_next;
		}

		None
	}

	/// Links on one bucket's chain, for the tests that bound what an insert is
	/// allowed to walk.
	#[cfg(test)]
	fn chain_len(&self, b: usize) -> usize {
		let mut i = self.buckets[b];
		let mut n = 0;

		while i != NIL {
			n += 1;
			i = self.slots[i as usize].hash_next;
		}

		n
	}

	/// Push onto the front of the key's chain, and split ONE bucket if that
	/// took the table past the load factor.
	///
	/// Front insertion is deliberate: a freshly inserted key is the one most
	/// likely to be looked up next, so it should be the first link walked.
	fn bucket_link(&mut self, i: u32) {
		let b = self.bucket_of(self.slots[i as usize].hashed);

		self.slots[i as usize].hash_next = self.buckets[b];
		self.buckets[b] = i;
		self.live += 1;

		if self.live * MAX_LOAD_DENOM > self.buckets.len() * MAX_LOAD_NUMER {
			self.split_one_bucket();
		}
	}

	/// Unlink by key, repairing the predecessor's `hash_next`.
	fn bucket_unlink(&mut self, key: HashedKey) -> Option<u32> {
		let b = self.bucket_of(key);
		let mut i = self.buckets[b];
		let mut prev = NIL;

		while i != NIL {
			let (hashed, next) = {
				let slot = &self.slots[i as usize];
				(slot.hashed, slot.hash_next)
			};

			if hashed == key {
				match prev {
					NIL => self.buckets[b] = next,
					prev => self.slots[prev as usize].hash_next = next,
				}

				self.slots[i as usize].hash_next = NIL;
				self.live -= 1;

				return Some(i);
			}

			prev = i;
			i = next;
		}

		None
	}

	/// Grow the table by ONE bucket, re-partitioning one chain.
	///
	/// This replaces a `grow_buckets` that doubled the table and re-threaded
	/// every chain by walking the whole recency list -- under the shard write
	/// lock, on the API thread, so a shard holding a million objects stalled
	/// every request to it for the length of a million-node pointer chase.
	/// Here the work is one bucket's chain, mean length 1, and it is paid by
	/// the insert that crossed the load factor.
	///
	/// Splitting bucket `split` under the wide mask sends each of its keys to
	/// either `split` or `split + base` and nowhere else, which is the whole
	/// trick: no other bucket's contents can be affected, so no other bucket
	/// has to be visited. Once every bucket of the round has been split the
	/// wide mask becomes the narrow one and the round starts again.
	///
	/// Recycled slots cannot be dragged in: only the chain is walked, and a
	/// recycled slot is off every chain -- `bucket_unlink` takes it off before
	/// it reaches the free list.
	fn split_one_bucket(&mut self) {
		let from = self.split;
		let wide = self.base * 2 - 1;

		// `push` rather than a resize: the table grows by one bucket, and the
		// `Vec`'s own amortised doubling copies a flat array of `u32`s, which
		// is a memcpy and not a walk of anything.
		self.buckets.push(NIL);

		let mut i = self.buckets[from];
		let mut stay = NIL;
		let mut moved = NIL;

		while i != NIL {
			#[cfg(test)]
			self.walk.fetch_add(1, Ordering::Relaxed);

			let (key, next) = {
				let slot = &self.slots[i as usize];
				(slot.hashed, slot.hash_next)
			};

			match (key as usize) & wide == from {
				true => {
					self.slots[i as usize].hash_next = stay;
					stay = i;
				},

				false => {
					self.slots[i as usize].hash_next = moved;
					moved = i;
				},
			}

			i = next;
		}

		self.buckets[from] = stay;
		self.buckets[from + self.base] = moved;

		self.split += 1;

		if self.split == self.base {
			self.split = 0;
			self.base *= 2;
		}
	}

	fn unlink(&mut self, i: u32) {
		let (p, n) = {
			let s = &self.slots[i as usize];
			(s.prev, s.next)
		};

		match p {
			NIL => self.head = n,
			p => self.slots[p as usize].next = n,
		}

		match n {
			NIL => self.tail = p,
			n => self.slots[n as usize].prev = p,
		}

		let s = &mut self.slots[i as usize];
		s.prev = NIL;
		s.next = NIL;
	}

	fn link_front(&mut self, i: u32) {
		let old = self.head;

		{
			let s = &mut self.slots[i as usize];
			s.prev = NIL;
			s.next = old;
		}

		match old {
			NIL => self.tail = i,
			old => self.slots[old as usize].prev = i,
		}

		self.head = i;
	}

	/// Reverses this slot's contribution to the tier accounting and steps the
	/// boundary back off it. Must run BEFORE `unlink`, which clears `prev`.
	fn detach_tier(&mut self, i: u32, lfu: bool) {
		let (tier, migrating, prev, freq) = {
			let s = &self.slots[i as usize];
			(s.tier, s.migrating(), s.prev, s.freq)
		};

		match lfu {
			// Under `Lfu` there is no prefix cursor to step: the slot leaves
			// the frequency bucket it is in, and that ALSO repairs its
			// neighbours' `prev`/`next`, because those are bucket chains here.
			// So the recency-list `unlink` the callers run under the other
			// three orders must not run -- it would walk `head`/`tail`, which
			// are `NIL`, and corrupt the bucket.
			true => self.freq_unlink(i, freq, tier),

			// If the departing slot was the boundary, the new least-recently-
			// used fast slot is the one in front of it. When the boundary was
			// also the list tail -- every slot fast -- that is the new tail,
			// which is the same answer.
			false => {
				if self.fast_boundary == i {
					self.fast_boundary = prev;
				}
			},
		}

		match tier {
			Tier::Fast => {
				self.fast_used = self.fast_used.saturating_sub(migrating);
				self.fast_count = self.fast_count.saturating_sub(1);
			},

			Tier::Slow => {
				self.slow_used = self.slow_used.saturating_sub(migrating);
			},
		}
	}

	/// Unlink, drop the object and return the slot to the free list.
	///
	/// Dropping the object is what RETIRES its value: `Object::drop` defers the
	/// free under an epoch pin rather than performing it, so a reader that
	/// lifted this value's pointer out from under the shard guard a moment ago
	/// and is still copying its bytes is safe. Nothing extra is needed here --
	/// and deliberately so, since this runs on the policy worker, `take` runs
	/// on the API thread, and the TTL reaper runs on a third; a per-site rule
	/// would have to be repeated at all of them.
	fn retire(&mut self, i: u32, lfu: bool) {
		self.detach_tier(i, lfu);

		// `detach_tier` already took the slot off its frequency bucket under
		// `Lfu`, and that is the only list it was on.
		if !lfu {
			self.unlink(i);
		}

		self.slots[i as usize].object = None;
		self.free.push(i);
	}

	/// Move to the MRU end and make fast, promoting from slow if needed.
	///
	/// Faithful port of `LruCompactHybridStack::touch_fast_key`, minus the
	/// settle: the tier boundary is now settled globally, with no shard lock
	/// held, so the caller drops this shard's guard and then calls
	/// `MergedStore::settle_tier`. A promotion that a tight budget immediately
	/// undoes therefore reports BOTH transitions, in order, rather than
	/// suppressing the first -- per-key order is preserved, so the consumer
	/// applies promote-then-demote and lands on the same final placement.
	fn touch_slot(&mut self, i: u32, now: u64) {
		let previous_tier = self.slots[i as usize].tier;
		let already_at_front = self.head == i;
		let is_boundary = self.fast_boundary == i;

		// Read the neighbour BEFORE moving: once the slot is at the front its
		// predecessor is gone, and the boundary has to step back to whatever
		// was in front of it.
		let new_boundary_if_moved = match is_boundary && !already_at_front {
			true => self.slots[i as usize].prev,
			false => NIL,
		};

		if !already_at_front {
			self.unlink(i);
			self.link_front(i);

			if is_boundary {
				self.fast_boundary = new_boundary_if_moved;
			}
		}

		self.slots[i as usize].last_access = now;

		if previous_tier != Tier::Fast {
			let migrating = self.slots[i as usize].migrating();

			self.slow_used = self.slow_used.saturating_sub(migrating);
			self.fast_used += migrating;
			self.fast_count += 1;
			self.slots[i as usize].tier = Tier::Fast;

			if self.fast_boundary == NIL {
				self.fast_boundary = i;
			}

			// Queued even when the bytes are already fast, as they are on an
			// overwrite: `set` builds an LRU value in DRAM before `insert` gets
			// here, so the consumer will decline this entry. The no-op is the
			// price of a guarantee. Queued migrations carry no identity --
			// `apply_migration` acts on whatever object holds the key when it
			// dequeues -- so a demotion decided for the OLD object and still
			// queued when the overwrite lands demotes the NEW one, and this
			// entry, behind it on the key's FIFO consumer, is what restores it.
			// Skipping it for a physically fast value strands a fresh value in
			// the slow tier while this slot counts it fast. Pinned by
			// `merged_overwrite_tests::an_overwrite_is_repromoted_after_a_stale_demotion`.
			let key = self.slots[i as usize].hashed;
			self.migrations.push((key, Tier::Fast));
		}
	}

	/// Demotes exactly ONE slot -- the boundary, the least-recently-used fast
	/// slot in this shard -- and steps the boundary back off it.
	///
	/// Nothing is searched, and because the boundary only walks along a list
	/// this never reorders anything. Returns the bytes that left the fast tier,
	/// or `None` when the shard holds nothing fast.
	///
	/// One step per call, rather than a drain loop, because the loop now lives
	/// in `MergedStore::settle_tier` and re-chooses the shard after every step:
	/// the next victim is whichever shard's boundary is now oldest, which is
	/// what makes the demotion order global rather than per shard.
	fn demote_boundary(&mut self) -> Option<CacheSize> {
		let d = self.fast_boundary;

		if d == NIL {
			return None;
		}

		let (key, migrating, prev) = {
			let s = &self.slots[d as usize];
			(s.hashed, s.migrating(), s.prev)
		};

		self.slots[d as usize].tier = Tier::Slow;

		self.fast_used = self.fast_used.saturating_sub(migrating);
		self.fast_count = self.fast_count.saturating_sub(1);
		self.slow_used += migrating;
		self.fast_boundary = prev;

		self.migrations.push((key, Tier::Slow));

		Some(migrating)
	}

	/// This shard's three tier totals, read as a group.
	///
	/// Taken together rather than one accessor per counter so that a
	/// before/after bracket cannot cover one of them and silently miss the
	/// others: every locked section that moves ANY of the three was already
	/// bracketed for `fast_used`, and reading all three through one accessor is
	/// what extends those proven brackets to the other two.
	#[inline]
	fn totals(&self) -> ShardTotals {
		ShardTotals {
			fast_used: self.fast_used,
			slow_used: self.slow_used,
			fast_count: self.fast_count as CacheSize,
		}
	}

	fn tail_seq(&self) -> u64 {
		match self.tail {
			NIL => EMPTY_TAIL,
			t => self.slots[t as usize].last_access,
		}
	}

	/// The stamp of the least-recently-used FAST slot, for the second mirror.
	/// `EMPTY_TAIL` when the shard holds nothing fast, which reads as "never a
	/// candidate" in the settle loop's minimum.
	fn fast_tail_seq(&self) -> u64 {
		match self.fast_boundary {
			NIL => EMPTY_TAIL,
			b => self.slots[b as usize].last_access,
		}
	}

	// ── MergedOrder::Lfu ─────────────────────────────────────────────────
	//
	// Everything below maintains the two frequency bucket sets. Ported method
	// for method from `ArenaFrequencyChain`, which is the structure
	// `LfuCompactHybridStack` drives, so the two agree on bucket order by
	// construction rather than by intention.
	//
	// Named `freq_link`/`freq_unlink` and NOT `bucket_link`/`bucket_unlink`:
	// those names are taken by the HASH chain a dozen lines up, and confusing
	// the two would corrupt the index.

	/// The frequency buckets for `tier`.
	#[inline]
	fn freq_buckets(&self, tier: Tier) -> &FreqBuckets {
		match tier {
			Tier::Fast => &self.fast_buckets,
			Tier::Slow => &self.slow_buckets,
		}
	}

	/// Appends slot `i` at the TAIL of bucket `freq` in `tier`'s bucket set.
	///
	/// The tail rather than the head, because that is where
	/// `ArenaFrequencyChain::link` puts it while `min_key` reads the head -- so
	/// the key that entered a bucket EARLIEST leaves first. The flat
	/// `LfuCompactStack` states the same order from the other end (`link_front`
	/// at the head, `evict_one` from the tail), and matching both is what makes
	/// the differential tests passable at all.
	fn freq_link(&mut self, i: u32, freq: u16, tier: Tier) {
		// The tripwire for the one mistake this order can make silently.
		// `head`/`tail` and `link_front`/`unlink` are the RECENCY list, shared
		// by the other three orders; under `Lfu` `prev`/`next` are bucket
		// chains instead, so a shared path that relinked the recency list would
		// corrupt a bucket and produce a plausible-but-wrong victim with no
		// compile error anywhere. Nothing links the recency list under `Lfu`,
		// so it must still be empty every time a bucket is touched.
		debug_assert!(
			self.head == NIL && self.tail == NIL,
			"the recency list is not empty under Lfu -- some shared path called \
			 link_front, and prev/next are bucket chains here",
		);

		let buckets = match tier {
			Tier::Fast => &mut self.fast_buckets,
			Tier::Slow => &mut self.slow_buckets,
		};

		match buckets.get_mut(&freq) {
			Some((_, tail)) => {
				let old_tail = *tail;
				*tail = i;

				self.slots[old_tail as usize].next = i;
				self.slots[i as usize].prev = old_tail;
				self.slots[i as usize].next = NIL;
			},

			None => {
				buckets.insert(freq, (i, i));

				self.slots[i as usize].prev = NIL;
				self.slots[i as usize].next = NIL;
			},
		}
	}

	/// Unlinks slot `i` from bucket `freq` in `tier`, dropping the bucket when
	/// it empties -- so the maps stay O(distinct frequencies PRESENT) rather
	/// than accumulating an entry per frequency ever seen.
	fn freq_unlink(&mut self, i: u32, freq: u16, tier: Tier) {
		let (prev, next) = {
			let s = &self.slots[i as usize];
			(s.prev, s.next)
		};

		if prev != NIL {
			self.slots[prev as usize].next = next;
		}

		if next != NIL {
			self.slots[next as usize].prev = prev;
		}

		let buckets = match tier {
			Tier::Fast => &mut self.fast_buckets,
			Tier::Slow => &mut self.slow_buckets,
		};

		if let Some((head, tail)) = buckets.get_mut(&freq) {
			if *head == i {
				*head = next;
			}

			if *tail == i {
				*tail = prev;
			}

			if *head == NIL {
				buckets.remove(&freq);
			}
		}

		let s = &mut self.slots[i as usize];
		s.prev = NIL;
		s.next = NIL;
	}

	/// `(minimum frequency, that bucket head's stamp)` for `tier`, or `None`
	/// when this shard holds nothing in it.
	///
	/// The PAIR is what the cross-shard minimum is taken over: frequency
	/// first, then the stamp at which that bucket's head entered the bucket, so
	/// two shards tied at a frequency are separated exactly the way two keys
	/// inside one bucket are. O(log D) in the distinct frequencies present.
	fn min_freq(&self, tier: Tier) -> Option<(u16, u64)> {
		let (&freq, &(head, _)) = self.freq_buckets(tier).iter().next()?;

		Some((freq, self.slots[head as usize].last_access))
	}

	/// The slot holding this tier's least-frequently-used key: the head of its
	/// lowest-frequency bucket.
	fn freq_min_slot(&self, tier: Tier) -> Option<u32> {
		let (_, &(head, _)) = self.freq_buckets(tier).iter().next()?;

		Some(head)
	}

	/// A frequency bump: unlink, increment, restamp, relink at the tail of the
	/// next bucket. Returns the new count.
	///
	/// The restamp is not cosmetic. Under this order `last_access` means "when
	/// this key entered its current bucket", which is what orders two keys of
	/// equal frequency -- so a bump that moved the key without restamping it
	/// would leave it ranked against its new peers by a stale position.
	fn bump_slot(&mut self, i: u32, now: u64) -> u16 {
		let (freq, tier) = {
			let s = &self.slots[i as usize];
			(s.freq, s.tier)
		};

		// Saturating at 65,535 -- see `Slot::freq` for why that cap cannot
		// change a victim choice.
		let next = freq.saturating_add(1);

		self.freq_unlink(i, freq, tier);

		{
			let s = &mut self.slots[i as usize];
			s.freq = next;
			s.last_access = now;
		}

		self.freq_link(i, next, tier);

		next
	}

	/// Promotes a SLOW slot into the fast tier at its current frequency.
	///
	/// `ArenaFrequencyChain::set_tier`: the key keeps its count, and because
	/// `set_tier` relinks by APPENDING at the destination bucket's tail, the
	/// stamp is refreshed here too.
	fn promote_freq(&mut self, i: u32, now: u64) {
		let (key, freq, migrating) = {
			let s = &self.slots[i as usize];
			(s.hashed, s.freq, s.migrating())
		};

		self.freq_unlink(i, freq, Tier::Slow);

		{
			let s = &mut self.slots[i as usize];
			s.tier = Tier::Fast;
			s.last_access = now;
		}

		self.freq_link(i, freq, Tier::Fast);

		self.slow_used = self.slow_used.saturating_sub(migrating);
		self.fast_used += migrating;
		self.fast_count += 1;

		// Pushed unconditionally, exactly as `touch_slot`'s promotion is, and
		// NOT guarded on the key still being fast after the settle the way
		// `LfuCompactHybridStack` guards its own. This store's convention is
		// already the other one: "a promotion that a tight budget immediately
		// undoes reports BOTH transitions, in order, rather than suppressing
		// the first -- per-key order is preserved, so the consumer applies
		// promote-then-demote and lands on the same final placement." The
		// placement the two reach is identical; only the record stream differs.
		self.migrations.push((key, Tier::Fast));
	}

	/// Demotes this shard's least-frequently-used FAST key, at its own count.
	///
	/// `LfuCompactHybridStack::settle_fast_tier` demotes `min_with_count(Fast)`
	/// and carries the count across, which is what
	/// `ArenaFrequencyChain::set_tier` does for free. Returns the bytes that
	/// left the fast tier, or `None` when this shard holds nothing fast.
	fn demote_freq_min(&mut self, now: u64) -> Option<CacheSize> {
		let d = self.freq_min_slot(Tier::Fast)?;

		let (key, freq, migrating) = {
			let s = &self.slots[d as usize];
			(s.hashed, s.freq, s.migrating())
		};

		self.freq_unlink(d, freq, Tier::Fast);

		{
			let s = &mut self.slots[d as usize];
			s.tier = Tier::Slow;
			s.last_access = now;
		}

		self.freq_link(d, freq, Tier::Slow);

		self.fast_used = self.fast_used.saturating_sub(migrating);
		self.fast_count = self.fast_count.saturating_sub(1);
		self.slow_used += migrating;

		self.migrations.push((key, Tier::Slow));

		Some(migrating)
	}
}

/// The tiering configuration the settle loop needs, resolved once so the loop
/// reads the store's atomics once rather than per demotion.
///
/// `per_shard_capacity` is GONE. It was `fast_capacity / SHARDS`, and settling
/// each shard against its own share is precisely what made the demotion order
/// per-shard rather than global: a shard whose recent objects are large demoted
/// objects more recent than fast objects idling in another shard, and the fast
/// tier went under-used whenever recency skewed across shards. The capacity is
/// now the whole store's, checked against one `fast_used` total.
#[derive(Clone, Copy)]
struct TierBudget {
	capacity: CacheSize,
	shared_overhead: CacheSize,
	high_ppm: u64,
	low_ppm: u64,
}

/// Watermark fractions are carried as parts-per-million so they fit an
/// `AtomicU64` without a float-atomic dance.
#[inline]
fn scale(bytes: CacheSize, ppm: u64) -> CacheSize {
	(bytes as f64 * (ppm as f64 / 1_000_000.0)) as CacheSize
}

const DEFAULT_HIGH_PPM: u64 = 980_000;
const DEFAULT_LOW_PPM: u64 = 950_000;

/// A snapshot of one shard's tier totals, as the before/after bracket pair
/// that every locked section reports its mutations through.
#[derive(Clone, Copy)]
struct ShardTotals {
	fast_used: CacheSize,
	slow_used: CacheSize,
	fast_count: CacheSize,
}

/// Applies one counter's before/after change to the store-level total that
/// mirrors it.
///
/// Saturating on the way down: a concurrent settler may have subtracted the
/// same bytes a moment earlier, and an underflow here would wrap to `u64::MAX`
/// -- which on `fast_used` would demote the entire fast tier.
#[inline]
fn apply_delta(total: &AtomicU64, before: CacheSize, after: CacheSize) {
	match after.cmp(&before) {
		std::cmp::Ordering::Greater => {
			total.fetch_add(after - before, Ordering::Relaxed);
		},

		std::cmp::Ordering::Less => {
			let _ = total.fetch_update(
				Ordering::Relaxed,
				Ordering::Relaxed,
				|v| Some(v.saturating_sub(before - after)),
			);
		},

		std::cmp::Ordering::Equal => {},
	}
}

pub struct MergedStore<K, V> {
	shards: Box<[RwLock<Inner<K, V>>]>,

	/// Mirrors each shard's tail `last_access`, readable without that shard's
	/// lock so an eviction victim can be chosen with atomic loads alone.
	tails: Box<[TailSeq]>,

	/// The same trick for the TIER boundary: each shard mirrors the stamp of
	/// its `fast_boundary` slot, so the globally least-recently-used FAST
	/// object is found with `SHARDS` relaxed loads and no lock at all.
	///
	/// Exact, not approximate: a shard's fast set is a contiguous MRU prefix
	/// of its own list, so the oldest boundary across shards is the global LRU
	/// fast object.
	fast_tails: Box<[TailSeq]>,

	clock: AtomicU64,
	tracked: AtomicUsize,

	/// `sum(shard.fast_used)`, maintained by every locked section that changes
	/// one, so the settle loop can test the budget without taking a lock.
	/// `fast_used_matches_the_shards` pins the two together.
	fast_used: AtomicU64,

	/// `sum(shard.slow_used)` and `sum(shard.fast_count)`, on exactly the same
	/// footing as `fast_used` above and maintained by the same brackets.
	///
	/// These exist because `refresh_tier_gauges` reads all three gauges once per
	/// worker pass, and reading them by sweeping cost 96 shard read locks a pass
	/// (32 per gauge, plus 32 more recomputing `slow_object_count` from a
	/// `fast_object_count` taken one line earlier). A gauge feeds the fast-tier
	/// budget, so the price of mirroring them is that a path which moves a shard
	/// counter without reporting it makes the cache admit the wrong number of
	/// objects, silently -- `gauges_match_the_shards` and `verify_gauges` are
	/// what turn that into a failing test instead.
	slow_used: AtomicU64,
	fast_count: AtomicU64,

	/// One bit per shard, set while that shard's write lock is held whenever it
	/// has migration records waiting and cleared only by `drain_migrations`.
	///
	/// A single global flag meant that one migrated shard cost the drain a WRITE
	/// lock on all 32, and `apply_tier_migrations` calls it once per worker
	/// EVENT. With the mask the drain locks only the shards that actually
	/// migrated -- normally the one the settle loop just demoted from -- and an
	/// event with nothing pending costs a single relaxed load.
	dirty_migrations: AtomicU32,

	/// Accesses that must elapse before a key is relinked again. 0 relinks
	/// every time, which is exact LRU. memcached's equivalent is 60 seconds.
	///
	/// Meaningless under `MergedOrder::Fifo`, which never relinks at all.
	update_interval: u64,

	/// The eviction order -- see [`MergedOrder`].
	///
	/// An atomic only because the store is built before the policy worker
	/// builds its `PolicyStack` over the same `Arc`, so the order arrives
	/// through a `&self` exactly as the tiering configuration does. It is
	/// written once, by `MergedStackHandle::new`, before the cache has served
	/// anything, and is a relaxed load on the read side.
	order: AtomicU8,

	/// `MergedOrder::Lfu` only: once shut, every brand-new key is admitted
	/// straight to the SLOW tier regardless of byte slack.
	///
	/// `LfuCompactHybridStack::fast_tier_latched`, for the same reason it
	/// exists there: byte slack freed by an object-granular demotion would
	/// otherwise let a frequency-1 newcomer take the room back and bypass the
	/// promotion rule entirely.
	///
	/// Deliberately the store's OWN field, and deliberately not published into
	/// `status`. `status`'s `hybrid_admission_latched` gauge is written once
	/// per worker pass by `refresh_tier_gauges`, and it is what
	/// `hybrid_policy::admission_tier` misreads under a burst; this flag is
	/// written under the shard lock of the settle that demoted and read on the
	/// thread that admits, so it cannot lag. `admission_latched()` on the
	/// handle therefore stays FALSE -- publishing this would recreate exactly
	/// the lagging mirror the split stack is bitten by.
	lfu_latched: AtomicBool,

	/// Fast-tier byte budget across ALL shards, settled against globally.
	fast_capacity: AtomicU64,
	shared_overhead: AtomicU64,
	high_ppm: AtomicU64,
	low_ppm: AtomicU64,
}

impl<K, V> Default for MergedStore<K, V> {
	fn default() -> Self {
		let shards = (0..SHARDS)
			.map(|_| RwLock::new(Inner::new()))
			.collect::<Vec<_>>()
			.into_boxed_slice();

		let new_mirrors = || {
			(0..SHARDS)
				.map(|_| TailSeq {
					seq: AtomicU64::new(EMPTY_TAIL),
					freq: AtomicU64::new(EMPTY_TAIL),
				})
				.collect::<Vec<_>>()
				.into_boxed_slice()
		};

		let tails = new_mirrors();
		let fast_tails = new_mirrors();

		MergedStore {
			shards,
			tails,
			fast_tails,
			clock: AtomicU64::new(0),
			tracked: AtomicUsize::new(0),
			fast_used: AtomicU64::new(0),
			slow_used: AtomicU64::new(0),
			fast_count: AtomicU64::new(0),
			dirty_migrations: AtomicU32::new(0),
			update_interval: std::env::var("MERGED_UPDATE_INTERVAL")
				.ok()
				.and_then(|v| v.parse().ok())
				.unwrap_or(0),

			// Recency until told otherwise, which is what every caller that
			// never mentions an order was already getting.
			order: AtomicU8::new(MergedOrder::Lru as u8),

			lfu_latched: AtomicBool::new(false),

			// Untiered until the worker configures it: nothing can ever exceed
			// this, so `settle_tier` returns at its first comparison -- one
			// relaxed load -- and a flat build pays no tiering cost at all.
			fast_capacity: AtomicU64::new(CacheSize::MAX),
			shared_overhead: AtomicU64::new(0),
			high_ppm: AtomicU64::new(DEFAULT_HIGH_PPM),
			low_ppm: AtomicU64::new(DEFAULT_LOW_PPM),
		}
	}
}

impl<K, V> MergedStore<K, V> {
	pub fn new() -> Self {
		Self::default()
	}

	/// Installs the eviction order.
	///
	/// Called once, by `MergedStackHandle::new`, at the moment the policy
	/// worker builds its stack over this same `Arc` -- the same point and the
	/// same reason as `configure_tiering`. Not called at all by a caller that
	/// wants recency, which is the default.
	pub fn set_order(&self, order: MergedOrder) {
		self.order.store(order as u8, Ordering::Relaxed);
	}

	#[inline]
	pub fn order(&self) -> MergedOrder {
		MergedOrder::from_repr(self.order.load(Ordering::Relaxed))
	}

	/// Installs the fast-tier budget and the per-object DRAM reservation, and
	/// settles every shard against them.
	///
	/// Called once by the policy worker when it builds its `PolicyStack` over
	/// this same `Arc`, with the values `init_policy_stack` hands the split
	/// hybrid stacks -- so the merged store is tiered on exactly the terms
	/// `LruCompactHybridStack` is.
	pub fn configure_tiering(
		&self,
		fast_capacity: CacheSize,
		shared_overhead: CacheSize,
		high_ppm: u64,
		low_ppm: u64,
	) {
		self.fast_capacity.store(fast_capacity, Ordering::Relaxed);
		self.shared_overhead.store(shared_overhead, Ordering::Relaxed);
		self.high_ppm.store(high_ppm, Ordering::Relaxed);
		self.low_ppm.store(low_ppm.min(high_ppm), Ordering::Relaxed);

		self.settle_all();
	}

	fn budget(&self) -> TierBudget {
		TierBudget {
			capacity: self.fast_capacity.load(Ordering::Relaxed),
			shared_overhead: self.shared_overhead.load(Ordering::Relaxed),
			high_ppm: self.high_ppm.load(Ordering::Relaxed),
			low_ppm: self.low_ppm.load(Ordering::Relaxed),
		}
	}

	#[inline]
	fn publish_tail(&self, shard: usize, inner: &Inner<K, V>) {
		match self.order() {
			// Under `Lfu` the recency list is unused, so there is no tail to
			// mirror: `tails` carries the SLOW tier's
			// `(minimum frequency, head stamp)` instead -- the pair
			// `lfu_victim` takes its lexicographic minimum over, and the slow
			// tier because that is the one eviction prefers.
			MergedOrder::Lfu => Self::publish_freq(&self.tails[shard], inner, Tier::Slow),
			_ => self.tails[shard].seq.store(inner.tail_seq(), Ordering::Relaxed),
		}
	}

	#[inline]
	fn publish_fast_tail(&self, shard: usize, inner: &Inner<K, V>) {
		match self.order() {
			// The FAST tier's minimum, which under this order serves three
			// callers: the demotion victim, the promotion rule's `min_count`,
			// and `lfu_victim`'s fallback when nothing is slow.
			MergedOrder::Lfu => Self::publish_freq(&self.fast_tails[shard], inner, Tier::Fast),
			_ => self.fast_tails[shard].seq.store(inner.fast_tail_seq(), Ordering::Relaxed),
		}
	}

	/// Publishes one tier's `(minimum frequency, head stamp)` into one mirror.
	#[inline]
	fn publish_freq(mirror: &TailSeq, inner: &Inner<K, V>, tier: Tier) {
		let (freq, seq) = match inner.min_freq(tier) {
			Some((freq, seq)) => (freq as u64, seq),
			None => (EMPTY_TAIL, EMPTY_TAIL),
		};

		// Stamp first, frequency second. Every reader tests `freq` against
		// `EMPTY_TAIL` and only then reads `seq`, so publishing in this order
		// means a reader can never pair a live frequency with a stamp from the
		// shard's previous state.
		mirror.seq.store(seq, Ordering::Relaxed);
		mirror.freq.store(freq, Ordering::Relaxed);
	}

	/// Both mirrors at once. Every locked section that can move a shard's tail
	/// or its tier boundary ends with this, so the two lock-free choices --
	/// which key to evict and which shard to demote from -- are always reading
	/// the shard's real state.
	#[inline]
	fn publish_mirrors(&self, shard: usize, inner: &Inner<K, V>) {
		self.publish_tail(shard, inner);
		self.publish_fast_tail(shard, inner);
	}

	#[inline]
	fn note_migrations(&self, shard: usize, inner: &Inner<K, V>) {
		if inner.migrations.is_empty() {
			return;
		}

		let bit = 1u32 << shard;

		// Load before the read-modify-write. A shard under a steady demotion
		// stream sets its bit once and then only reads it until the next drain,
		// so the common case does not bounce the line between API threads.
		if self.dirty_migrations.load(Ordering::Relaxed) & bit == 0 {
			self.dirty_migrations.fetch_or(bit, Ordering::Relaxed);
		}
	}

	/// Applies a shard's change in ALL THREE tier totals to the store-level
	/// totals that mirror them.
	///
	/// Taken as a before/after pair rather than as a delta computed by each
	/// call site: the shard's own counters are the authority, so a total
	/// cannot drift from them by anyone forgetting which way a particular
	/// operation moved the bytes -- or, now, by remembering for the bytes and
	/// forgetting for the count. `insert`'s overwrite branch is exactly that
	/// case: a resize of a slow object moves `slow_used` and touches neither
	/// of the other two, and it reports through this one call like every other
	/// path because the bracket reads the group.
	#[inline]
	fn apply_totals_delta(&self, before: ShardTotals, after: ShardTotals) {
		apply_delta(&self.fast_used, before.fast_used, after.fast_used);
		apply_delta(&self.slow_used, before.slow_used, after.slow_used);
		apply_delta(&self.fast_count, before.fast_count, after.fast_count);
	}

	/// The shard whose fast boundary is oldest -- the globally least-recently-
	/// used fast object -- from `SHARDS` relaxed loads and no lock.
	fn oldest_fast_shard(&self) -> Option<usize> {
		let mut best = None;
		let mut best_seq = EMPTY_TAIL;

		for (s, t) in self.fast_tails.iter().enumerate() {
			let seq = t.seq.load(Ordering::Relaxed);

			if seq == EMPTY_TAIL {
				continue;
			}

			// A plain minimum: the clock is 64 bits and never wraps, so an
			// older slot's stamp is simply the smaller number.
			if best.is_none() || seq < best_seq {
				best_seq = seq;
				best = Some(s);
			}
		}

		best
	}

	/// Demote, globally, until the fast tier is back under the low watermark.
	///
	/// Runs with NO shard lock held. Each step is: 32 relaxed loads to name the
	/// shard holding the oldest fast object, that ONE shard's write lock, one
	/// boundary step, republish, unlock. So no thread ever holds two shard
	/// locks and no lock-order cycle can form -- a toucher releases its own
	/// shard before calling this.
	///
	/// Two settlers running at once may each demote one extra object, which is
	/// what the 0.98/0.95 hysteresis is for. The cost is paid by the API thread
	/// that caused the overshoot.
	fn settle_tier(&self) {
		let budget = self.budget();

		// The reservation is per LIVE object and applies across both tiers, so
		// it comes off the budget before the watermarks are taken.
		let effective = budget
			.capacity
			.saturating_sub(self.len() as CacheSize * budget.shared_overhead);

		if self.fast_used.load(Ordering::Relaxed) <= scale(effective, budget.high_ppm) {
			return;
		}

		let target = scale(effective, budget.low_ppm);

		let order = self.order();

		while self.fast_used.load(Ordering::Relaxed) > target {
			// Under `Lfu` the demotion victim is the fast tier's LOWEST
			// FREQUENCY rather than its oldest boundary, chosen globally by the
			// same trick: a lexicographic minimum over the 32 `(freq, stamp)`
			// mirrors, with no lock held. That is
			// `LfuCompactHybridStack::settle_fast_tier`'s
			// `min_with_count(Tier::Fast)`, taken across shards.
			let chosen = match order {
				MergedOrder::Lfu => self.lfu_min_shard(&self.fast_tails),
				_ => self.oldest_fast_shard(),
			};

			let Some(s) = chosen else {
				// Nothing anywhere is fast. Whatever is left over the target is
				// the shared-overhead reservation, not value bytes.
				break;
			};

			// A demotion carries the key into a slow bucket at its own count,
			// as the newest entrant of that frequency, so it needs a stamp --
			// see `Inner::demote_freq_min`.
			let now = match order {
				MergedOrder::Lfu => self.clock.fetch_add(1, Ordering::Relaxed),
				_ => 0,
			};

			let mut g = self.shards[s].write().unwrap();
			let before = g.totals();

			// `None` when that shard's fast set went away between the load and
			// the lock -- another settler took it. Republish and re-choose.
			let demoted = match order {
				MergedOrder::Lfu => g.demote_freq_min(now),
				_ => g.demote_boundary(),
			};

			// A demotion firing at all means fast-tier capacity was genuinely
			// reached, which is what shuts admission -- the same rule, and the
			// same reason, as `settle_fast_tier`'s `fast_tier_latched = true`.
			if order == MergedOrder::Lfu && demoted.is_some() {
				self.lfu_latched.store(true, Ordering::Relaxed);
			}

			self.apply_totals_delta(before, g.totals());
			self.note_migrations(s, &g);

			// BOTH mirrors, not just the fast one: under `Lfu` a demotion takes
			// the key out of a fast bucket AND puts it into a slow one, so the
			// victim mirror moved too. Under the other three orders the list
			// tail is untouched and republishing it stores the value it already
			// held.
			self.publish_mirrors(s, &g);
		}
	}

	/// The worker's per-pass settle, and what `configure_tiering` and
	/// `resize_fast_tier` call. The SAME loop as an API thread's -- there is
	/// only one, so there is only one demotion order.
	fn settle_all(&self) {
		self.settle_tier();
	}

	/// Move `key` to the MRU end and make it fast.
	///
	/// The operation the design exists for: one lookup reaches the object, its
	/// position AND its tier, where the split design needs a second keyed
	/// lookup into a separate eviction stack for the last two.
	///
	/// Under `MergedOrder::Fifo` this is the whole of the policy difference and
	/// it does NOTHING -- not even advance the clock, since the clock exists to
	/// stamp relinks and FIFO has none. The guard is here rather than only at
	/// the caller so that no present or future caller of `touch` can smuggle
	/// recency into a FIFO run: a hit that restamped `last_access` would make
	/// `tail_key` nominate the wrong victim, silently, and the result would
	/// still look like a plausible miss ratio.
	pub fn touch(&self, key: HashedKey) {
		match self.order() {
			MergedOrder::Fifo => return,

			// The whole of an LFU hit, split out for the same reason CLOCK's
			// is: it shares nothing with the body below. In particular it must
			// NOT reach the `update_interval` probe -- skipping a relink is a
			// recency approximation memcached makes deliberately, but skipping
			// a BUMP loses a count, which changes the policy rather than
			// quantising it.
			MergedOrder::Lfu => return self.bump(key),

			// The whole of a CLOCK hit. It is split out rather than written
			// here because it shares NOTHING with the body below -- no clock
			// tick, no write lock, no relink, no promotion and no settle -- and
			// running any of that would be the bug. See `mark_referenced`.
			MergedOrder::Clock => return self.mark_referenced(key),

			MergedOrder::Lru => {},
		}

		let now = self.clock.fetch_add(1, Ordering::Relaxed);
		let s = shard_of(key);

		// memcached's `ITEM_UPDATE_INTERVAL`: a key hit repeatedly inside the
		// interval is already near enough the MRU end that relinking it buys
		// nothing, so skip under a READ lock and never block this shard's
		// readers at all.
		if self.update_interval > 0 {
			let g = self.shards[s].read().unwrap();

			let Some(i) = g.find(key) else { return };

			// A plain subtraction: the clock is 64 bits and monotonic, so the
			// stamp can only be at or behind it.
			let age = now.saturating_sub(g.slots[i as usize].last_access);

			if age < self.update_interval {
				return;
			}
		}

		{
			let mut g = self.shards[s].write().unwrap();

			let Some(i) = g.find(key) else { return };

			let before = g.totals();

			g.touch_slot(i, now);

			self.apply_totals_delta(before, g.totals());
			self.note_migrations(s, &g);
			self.publish_mirrors(s, &g);
		}

		// AFTER the guard is dropped -- see `settle_tier`. Holding it here
		// would let the settle take a second shard lock while holding this one.
		self.settle_tier();
	}

	/// A CLOCK hit: set the slot's reference bit, and do nothing else.
	///
	/// **This is where the hit path ends under `MergedOrder::Clock`.** One
	/// shard READ lock, one chain walk, one relaxed `store`, return. No
	/// `clock.fetch_add`, no relink, no `last_access`, no tier change, no
	/// mirror republish and no `settle_tier` -- and, the point of the whole
	/// exercise, NO SHARD WRITE LOCK. Under `Lru` the same hit runs
	/// `touch_slot` under `shards[s].write()`, which serialises every reader of
	/// that shard behind one relink and is the measured cause of the merged
	/// store's service time reaching 1.24x the DashMap arm's at sixteen
	/// clients. Here concurrent hits to the same shard proceed in parallel,
	/// and two hits racing on the SAME slot both write 1.
	///
	/// The work that relink represented has not vanished, it has MOVED: the
	/// hand pays for it in `clock_victim`, under a write lock the eviction path
	/// was taking anyway, and only for the slots that actually reach the tail.
	///
	/// `pub` so `MergedStackHandle::update` can reach it directly instead of
	/// going through `touch` and re-testing the order.
	pub fn mark_referenced(&self, key: HashedKey) {
		let g = self.shards[shard_of(key)].read().unwrap();

		let Some(i) = g.find(key) else { return };

		g.slots[i as usize].referenced.store(1, Ordering::Relaxed);
	}

	/// An LFU hit: bump the frequency, and promote out of the slow tier if the
	/// new count STRICTLY exceeds the fast tier's global minimum.
	///
	/// `LfuCompactHybridStack::maybe_promote`, with `min_count(Tier::Fast)`
	/// taken across the 32 shards instead of out of one chain.
	///
	/// Takes the shard WRITE lock, and that is inherent rather than lazy: the
	/// bump moves the slot between two bucket chains and mutates the shard's
	/// bucket map. This is the cost `MergedOrder::Clock` exists to avoid and
	/// LFU cannot avoid -- see [`MergedOrder::Lfu`].
	///
	/// `pub` so `MergedStackHandle::update` can reach it directly instead of
	/// going through `touch` and re-testing the order.
	pub fn bump(&self, key: HashedKey) {
		let now = self.clock.fetch_add(1, Ordering::Relaxed);
		let s = shard_of(key);

		// Read with NO lock held, before this shard's is taken: the fast
		// tier's minimum frequency is `SHARDS` relaxed loads over the mirror,
		// and every locked section republishes before it releases, so on entry
		// the mirror agrees with the shards it mirrors.
		let fast_min = self.lfu_min_freq(&self.fast_tails);

		let was_slow = {
			let mut g = self.shards[s].write().unwrap();

			let Some(i) = g.find(key) else { return };

			let before = g.totals();

			// Read before anything moves: a promotion below makes the slot
			// fast, and it is the tier the hit ARRIVED at that decides whether
			// this hit settles.
			let was_slow = g.slots[i as usize].tier == Tier::Slow;

			let new_freq = g.bump_slot(i, now);

			// STRICTLY greater, and an empty fast tier promotes: the reference
			// rule verbatim, so a TIE with the fast minimum does not promote.
			let promote = match fast_min {
				None => true,
				Some(min) => new_freq > min,
			};

			if promote && was_slow {
				let stamp = self.clock.fetch_add(1, Ordering::Relaxed);
				g.promote_freq(i, stamp);
			}

			self.apply_totals_delta(before, g.totals());
			self.note_migrations(s, &g);
			self.publish_mirrors(s, &g);

			was_slow
		};

		// Only a hit that arrived on a SLOW key settles, and that is the
		// reference's asymmetry rather than a shortcut.
		// `LfuCompactHybridStack::update` settles in its `Some(Tier::Slow)` arm
		// alone -- unconditionally there, whether or not the promotion actually
		// fired -- while a hit on a fast key is a bare `chain.bump` that
		// returns without settling.
		//
		// Same underlying reason as `insert`'s guard: admission is byte-gated
		// at the full effective capacity while the drain target sits at
		// `drain_target::ratio()` of it, so a tier legitimately rests in the
		// band between the two. Settling on a FAST hit would drain it out of
		// that band and demote keys the reference keeps fast -- which is
		// exactly how the differential test caught this, one step after the
		// admission guard fixed the same mistake on the insert path.
		//
		// AFTER the guard is dropped, as `touch` does it: the settle takes one
		// shard lock at a time and must not find this thread holding another.
		if was_slow {
			self.settle_tier();
		}
	}

	/// The shard holding the lexicographic minimum of `(freq, stamp)` over a
	/// mirror array. `SHARDS` relaxed loads, no lock.
	///
	/// This is the `Lfu` counterpart of `oldest_tail_shard` and
	/// `oldest_fast_shard`, and it is exact for the same reason: each shard is
	/// frequency-ordered within itself, so the global least-frequent object is
	/// one of 32 per-shard minima. The stamp is the tie-break, without which
	/// two shards holding equally-frequent keys would be separated arbitrarily
	/// and the differential test could not pass.
	fn lfu_min_shard(&self, mirrors: &[TailSeq]) -> Option<usize> {
		let mut best: Option<(u64, u64, usize)> = None;

		for (s, t) in mirrors.iter().enumerate() {
			let freq = t.freq.load(Ordering::Relaxed);

			if freq == EMPTY_TAIL {
				continue;
			}

			let seq = t.seq.load(Ordering::Relaxed);

			let better = match best {
				None => true,
				Some((best_freq, best_seq, _)) => (freq, seq) < (best_freq, best_seq),
			};

			if better {
				best = Some((freq, seq, s));
			}
		}

		best.map(|(_, _, s)| s)
	}

	/// The lowest frequency present in a tier across every shard, or `None`
	/// when no shard holds anything in it. The promotion rule's `min_count`.
	fn lfu_min_freq(&self, mirrors: &[TailSeq]) -> Option<u16> {
		let mut best: Option<u64> = None;

		for t in mirrors.iter() {
			let freq = t.freq.load(Ordering::Relaxed);

			if freq == EMPTY_TAIL {
				continue;
			}

			let better = match best {
				None => true,
				Some(best_freq) => freq < best_freq,
			};

			if better {
				best = Some(freq);
			}
		}

		// Lossless: only `publish_freq` writes this field, and it writes a
		// widened `u16`.
		best.map(|freq| freq as u16)
	}

	/// The globally least-frequently-used key, PREFERRING THE SLOW TIER.
	///
	/// The slow preference is `LfuCompactHybridStack::evict_one`'s -- "slow
	/// first; fall back to fast when nothing has ever been demoted" -- and
	/// under this order it has to be written out. The other three get it for
	/// free because their fast set is the newest prefix, so the globally oldest
	/// object is already a slow one whenever anything is slow. Under LFU a
	/// freshly admitted key sits at frequency 1 while demoted keys sit higher,
	/// so the unqualified global minimum is very often a FAST newcomer, and
	/// evicting it would be a different policy that still reported a plausible
	/// miss ratio.
	fn lfu_victim(&self) -> Option<HashedKey> {
		for tier in [Tier::Slow, Tier::Fast] {
			let mirrors: &[TailSeq] = match tier {
				Tier::Slow => &self.tails,
				Tier::Fast => &self.fast_tails,
			};

			let Some(s) = self.lfu_min_shard(mirrors) else { continue };

			let g = self.shards[s].read().unwrap();

			if let Some(i) = g.freq_min_slot(tier) {
				return Some(g.slots[i as usize].hashed);
			}
		}

		None
	}

	/// Where a brand-new key is admitted under `Lfu`.
	///
	/// `LfuCompactHybridStack::insert_resident`'s rule, in this store's byte
	/// terms: FAST while the effective budget has room and the latch is open,
	/// SLOW once it does not -- and the first refusal LATCHES, so every later
	/// newcomer goes straight to slow whatever slack an object-granular
	/// demotion has since freed.
	///
	/// This store's other three orders admit unconditionally fast and let the
	/// settle sort it out, and under LFU that would be WRONG rather than
	/// merely different: the settle demotes the lowest frequency, which is some
	/// older frequency-1 key, not the newcomer that caused the overflow. The
	/// reference would have left that older key alone and put the newcomer in
	/// the slow tier.
	///
	/// `+ 1` on the object count reserves the new object's own shared metadata,
	/// which is DRAM-resident whichever tier its value lands in.
	fn lfu_admission_tier(&self, migrating: CacheSize) -> Tier {
		if self.lfu_latched.load(Ordering::Relaxed) {
			return Tier::Slow;
		}

		let budget = self.budget();

		let admit_effective = budget.capacity.saturating_sub(
			(self.len() as CacheSize + 1) * budget.shared_overhead,
		);

		match self.fast_used.load(Ordering::Relaxed) + migrating <= admit_effective {
			true => Tier::Fast,

			false => {
				self.lfu_latched.store(true, Ordering::Relaxed);
				Tier::Slow
			},
		}
	}

	/// The worker has finished processing the `Set` event for `key`: settle the
	/// tier against the bytes the insert already accounted.
	///
	/// This used to do three more things, and each was wrong once the slot
	/// stopped storing a size:
	///
	///   * it wrote `size` and `dram_resident` into the slot. Both are gone --
	///     `Slot::migrating` derives them from the object, which has held the
	///     value's length since the value became one word, so the bytes are
	///     accounted by `insert` at the moment they become reachable rather
	///     than one worker event later;
	///   * it RELINKED the slot to the MRU end, which `insert` had already
	///     done a moment earlier. Two relinks per set, the second of them
	///     redundant, both taking the shard write lock;
	///   * it bumped the clock a SECOND time, so one set consumed two stamps
	///     and a set looked, to the recency order, more recent than a get of
	///     the same age.
	///
	/// The parameters stay to match `PolicyStack::insert_resident`, whose other
	/// implementations do keep a size of their own. A key evicted between the
	/// insert and this call is simply gone, and settling is still correct.
	pub fn record_size(&self, key: HashedKey, _size: ObjectSize, _dram_resident: ObjectSize) {
		let _ = key;

		// Under `Lfu` this settles NOTHING, for the reason spelled out at the
		// end of `insert`: admission is byte-gated there, so the settle points
		// are an overwrite, a hit and a resize -- exactly the three
		// `LfuCompactHybridStack` has. Leaving the settle here would undo
		// `insert`'s restraint one call later, since the worker reaches this
		// through `MergedStackHandle::insert_resident` for every Set.
		//
		// The budget is still bounded without it: the admission gate is the
		// whole effective capacity, so `fast_used` can reach that capacity and
		// never exceed it. What CAN drift is the per-object DRAM reservation --
		// `effective` shrinks as objects are admitted, so a tier admitted when
		// the reservation was smaller may sit above the current target until
		// the next hit or resize settles it. The reference has precisely the
		// same property, and matching it is the point.
		if self.order() == MergedOrder::Lfu {
			return;
		}

		self.settle_tier();
	}

	/// The globally least-recently-used key: the minimum over the shard tails.
	///
	/// `SHARDS` relaxed atomic loads and no lock. Each shard's list is ordered
	/// within itself, so the global LRU object is necessarily some shard's
	/// tail, and the oldest of those tails is it.
	pub fn tail_key(&self) -> Option<HashedKey> {
		match self.order() {
			// LRU and FIFO read the oldest tail and are done. CLOCK may have to
			// walk past a run of referenced slots first, and that walk MUTATES
			// -- see `clock_victim`.
			MergedOrder::Lru | MergedOrder::Fifo => self.oldest_tail_key(),
			MergedOrder::Clock => self.clock_victim(),

			// The minimum over 32 `(freq, stamp)` mirrors, slow tier first.
			MergedOrder::Lfu => self.lfu_victim(),
		}
	}

	/// The shard whose list tail is oldest, from `SHARDS` relaxed loads and no
	/// lock. `None` when every shard is empty.
	fn oldest_tail_shard(&self) -> Option<usize> {
		// A plain minimum over the stamps. This was a wrapping-difference AGE
		// comparison because `last_access` was the clock truncated to 32 bits,
		// where raw values stop being ordered once the clock wraps; at 64 bits
		// the clock does not wrap, so the older stamp is simply the smaller
		// number and the comparison is exact by construction.
		let mut best_seq = EMPTY_TAIL;
		let mut best_shard = usize::MAX;

		for (s, t) in self.tails.iter().enumerate() {
			let raw = t.seq.load(Ordering::Relaxed);

			if raw == EMPTY_TAIL {
				continue;
			}

			if best_shard == usize::MAX || raw < best_seq {
				best_seq = raw;
				best_shard = s;
			}
		}

		match best_shard {
			usize::MAX => None,
			s => Some(s),
		}
	}

	fn oldest_tail_key(&self) -> Option<HashedKey> {
		let s = self.oldest_tail_shard()?;
		let g = self.shards[s].read().unwrap();

		match g.tail {
			NIL => None,
			t => Some(g.slots[t as usize].hashed),
		}
	}

	/// CLOCK's hand: the oldest slot whose reference bit is CLEAR, granting a
	/// second chance to every referenced slot it passes.
	///
	/// The hand is the existing nomination path and not a cursor of its own.
	/// `oldest_tail_shard` already names the globally oldest slot, which is
	/// where a circular CLOCK's hand would be pointing; a second chance is
	/// `touch_slot`, which relinks that slot to its shard's head and restamps
	/// it as the newest thing in the store, which is where a circular CLOCK
	/// would next reach it -- one full revolution later. So the 32 per-shard
	/// lists still reconstruct one global CLOCK order out of the stamps, for
	/// exactly the reason they reconstruct one global FIFO order: the stamp is
	/// the position, and a second chance is a re-insertion.
	///
	/// Unlike the other two orders this WRITES, so it takes the shard's write
	/// lock rather than its read lock. That costs nothing that was not already
	/// being paid: `evict_one` nominates and `erase`'s `take` immediately takes
	/// the same shard's write lock to remove the victim. The hit path is what
	/// CLOCK keeps clean -- see `mark_referenced`.
	///
	/// # The budget
	///
	/// Sequentially this terminates in at most `len()` chances, since each one
	/// clears a bit and nothing else sets one. Concurrently an API thread can
	/// set a bit the hand just cleared, so a hot enough shard could in
	/// principle keep the hand spinning; `budget` bounds that and then evicts
	/// whatever is at the tail. It cannot fire on a quiesced store, which is
	/// what the differential test replays, so it changes no compared behaviour
	/// -- it is a liveness guard for the server, not a policy.
	fn clock_victim(&self) -> Option<HashedKey> {
		enum Step {
			Victim(HashedKey),
			Chance,
			Retry,
		}

		let mut budget = self.len().saturating_mul(2).saturating_add(8);

		loop {
			let s = self.oldest_tail_shard()?;

			let step = {
				let mut g = self.shards[s].write().unwrap();

				match g.tail {
					// The shard emptied between the relaxed load and the lock.
					// Republish so the next choice cannot pick it again.
					NIL => {
						self.publish_mirrors(s, &g);
						Step::Retry
					},

					t if budget == 0
						|| g.slots[t as usize].referenced.load(Ordering::Relaxed) == 0 =>
					{
						Step::Victim(g.slots[t as usize].hashed)
					},

					t => {
						let now = self.clock.fetch_add(1, Ordering::Relaxed);
						let before = g.totals();

						// Clear THEN relink, so a hit racing this one is
						// recorded against the slot's new position rather than
						// being wiped by the clear.
						g.slots[t as usize].referenced.store(0, Ordering::Relaxed);
						g.touch_slot(t, now);

						self.apply_totals_delta(before, g.totals());
						self.note_migrations(s, &g);
						self.publish_mirrors(s, &g);

						Step::Chance
					},
				}
			};

			match step {
				Step::Victim(key) => return Some(key),

				// Outside the guard -- `settle_tier` takes one shard lock at a
				// time and must not find this thread holding another.
				Step::Chance => {
					self.settle_tier();
					budget = budget.saturating_sub(1);
				},

				Step::Retry => budget = budget.saturating_sub(1),
			}
		}
	}

	pub fn contains_key(&self, key: &HashedKey) -> bool {
		self.shards[shard_of(*key)].read().unwrap().find(*key).is_some()
	}

	pub fn contains(&self, key: HashedKey) -> bool {
		self.contains_key(&key)
	}

	/// Remove and RETURN the object, unlinking it from the recency order and
	/// reversing its tier accounting in the same operation.
	///
	/// This is the merge paying off directly: the split design removes from the
	/// map and separately tells the stack, and when the second half is skipped
	/// the two diverge -- the failure `ERASE_FALLBACK` in `lib.rs::erase`
	/// exists to count. Here there is one structure, so the divergence has no
	/// way to occur.
	pub fn take(&self, key: &HashedKey) -> Option<Object<K, V>> {
		let s = shard_of(*key);
		let mut g = self.shards[s].write().unwrap();
		let i = g.bucket_unlink(*key)?;

		self.take_unlinked(s, &mut g, i)
	}

	/// `take`, but only if `pred` holds for the object stored under `key`.
	///
	/// The test and the removal run under one shard write guard, so no `set`
	/// can replace the object in between -- which is the point: `erase` checks
	/// a key match (hash collisions) and an expiry (the TTL reaper) against the
	/// object it then removes, not against whatever was there a moment ago.
	///
	/// Walks the bucket chain twice, once to test and once to unlink. `take`
	/// does not delegate here for that reason: capacity eviction has nothing to
	/// test and keeps the single walk.
	pub fn take_if(
		&self,
		key: &HashedKey,
		pred: impl FnOnce(&Object<K, V>) -> bool,
	) -> Option<Object<K, V>> {
		let s = shard_of(*key);
		let mut g = self.shards[s].write().unwrap();
		let found = g.find(*key)?;

		if !g.slots[found as usize].object.as_ref().is_some_and(pred) {
			return None;
		}

		let i = g.bucket_unlink(*key)?;
		debug_assert_eq!(i, found, "the write guard is held, so the slot cannot move");

		self.take_unlinked(s, &mut g, i)
	}

	/// The tail of `take` and `take_if`, once slot `i` is out of its bucket:
	/// detach it from its tier and the recency order, hand back its object and
	/// free the slot.
	fn take_unlinked(&self, s: usize, g: &mut Inner<K, V>, i: u32) -> Option<Object<K, V>> {
		let before = g.totals();
		let lfu = self.order() == MergedOrder::Lfu;

		g.detach_tier(i, lfu);

		if !lfu {
			g.unlink(i);
		}

		// Handed to the caller rather than dropped here, so the value's
		// retirement happens wherever the caller drops it -- still under a pin,
		// via `Object::drop`.
		let taken = g.slots[i as usize].object.take();
		g.free.push(i);

		self.apply_totals_delta(before, g.totals());
		self.publish_mirrors(s, g);
		self.tracked.fetch_sub(1, Ordering::Relaxed);

		taken
	}

	pub fn remove_key(&self, key: HashedKey) -> bool {
		let s = shard_of(key);
		let mut g = self.shards[s].write().unwrap();

		let Some(i) = g.bucket_unlink(key) else { return false };

		let before = g.totals();

		g.retire(i, self.order() == MergedOrder::Lfu);

		self.apply_totals_delta(before, g.totals());
		self.publish_mirrors(s, &g);
		self.tracked.fetch_sub(1, Ordering::Relaxed);

		true
	}

	pub fn get_ref(&self, key: &HashedKey) -> Option<MergedRef<'_, K, V>> {
		let guard = self.shards[shard_of(*key)].read().unwrap();
		let slot = guard.find(*key)?;

		Some(MergedRef { guard, slot })
	}

	pub fn get_mut_ref(&self, key: &HashedKey) -> Option<MergedRefMut<'_, K, V>> {
		let guard = self.shards[shard_of(*key)].write().unwrap();
		let slot = guard.find(*key)?;

		Some(MergedRefMut { guard, slot })
	}

	/// Insert at the MRU end, replacing any existing object for `key`.
	///
	/// Admission is unconditionally fast, matching `LruCompactHybridStack` --
	/// and matching what `PaperCache::set` physically built, since
	/// `admission_latched` is false for this store.
	pub fn insert(&self, key: HashedKey, object: Object<K, V>) -> Option<Object<K, V>> {
		let now = self.clock.fetch_add(1, Ordering::Relaxed);
		let s = shard_of(key);
		let order = self.order();

		// `Lfu` only, and read before the shard lock for the same reason `bump`
		// reads it there: it is `SHARDS` relaxed loads over the fast-tier
		// mirror, and an overwrite is an ACCESS under this order, so it can
		// promote. Never consulted under the other three.
		let lfu_fast_min = match order {
			MergedOrder::Lfu => self.lfu_min_freq(&self.fast_tails),
			_ => None,
		};

		let old = {
			let mut g = self.shards[s].write().unwrap();
			let before = g.totals();

			let old = match g.find(key) {
				Some(i) => {
					// An overwrite can change the value's length, so the tier
					// accounting moves by the DIFFERENCE, charged to whichever
					// tier the slot is in at this instant. Under LRU
					// `touch_slot` then moves the new figure to the fast tier
					// if it was slow.
					let was = g.slots[i as usize].migrating();
					let old = g.slots[i as usize].object.replace(object);
					let now_bytes = g.slots[i as usize].migrating();

					match g.slots[i as usize].tier {
						Tier::Fast => {
							g.fast_used = (g.fast_used + now_bytes).saturating_sub(was)
						},

						Tier::Slow => {
							g.slow_used = (g.slow_used + now_bytes).saturating_sub(was)
						},
					}

					// The accounting above runs under BOTH orders -- the bytes
					// really did change and somebody has to be charged for
					// them. Only the relink is conditional.
					//
					// Under FIFO an overwrite is a resize in place and nothing
					// else: no move to the front, no promotion out of the slow
					// tier, and above all no new `last_access`, since that
					// stamp is this object's position in the queue and the
					// object has not been re-inserted. This is
					// `FifoCompactHybridStack::insert_resident`'s "an existing
					// key is resized in place and NOT moved", reached from the
					// other side -- there the API thread's write never touches
					// the stack at all; here the map IS the stack, so the
					// restraint has to be spelled out.
					//
					// Under CLOCK it is FIFO's restraint PLUS the reference
					// bit, because the flat stack this must match treats a
					// re-insert as a hit: `ClockCompactStack::insert` forwards
					// an existing key straight to `update`, which sets the bit.
					// So an overwrite earns the object a second chance without
					// moving it, and forgetting the bit here would make a
					// written-and-then-evicted key leave in the wrong place.
					//
					// Under LFU an overwrite IS an access, and both references
					// agree: `LfuCompactStack::insert` forwards an existing key
					// to `update`, and
					// `LfuCompactHybridStack::insert_resident` records the size
					// change and then bumps it -- promoting it if the bump
					// carries it past the fast tier's minimum. So the count
					// moves and the position within the new bucket is refreshed,
					// exactly as on a GET hit.
					match order {
						MergedOrder::Lru => g.touch_slot(i, now),

						MergedOrder::Clock => {
							g.slots[i as usize].referenced.store(1, Ordering::Relaxed);
						},

						MergedOrder::Lfu => {
							let new_freq = g.bump_slot(i, now);

							let promote = match lfu_fast_min {
								None => true,
								Some(min) => new_freq > min,
							};

							if promote && g.slots[i as usize].tier == Tier::Slow {
								let stamp = self.clock.fetch_add(1, Ordering::Relaxed);
								g.promote_freq(i, stamp);
							}
						},

						MergedOrder::Fifo => {},
					}

					old
				},

				None => {
					// What this object will charge to whichever tier it lands
					// in. Computed from the object rather than from the slot,
					// because the admission decision below needs it BEFORE
					// there is a slot -- it is the same accessor
					// `Slot::migrating` uses, so the two cannot disagree.
					let migrating = crate::object::overhead::resident_object_bytes::<K>(
						object.data_size(),
					) as CacheSize;

					// The tier the POLICY decides. Unconditionally fast under
					// the three queue orders, which is what
					// `LruCompactHybridStack` does; under `Lfu` it is the
					// reference stack's admission rule, which can genuinely
					// choose slow -- see `lfu_admission_tier`.
					let decided = match order {
						MergedOrder::Lfu => self.lfu_admission_tier(migrating),
						_ => Tier::Fast,
					};

					// The tier the bytes are ACTUALLY in, read off the value
					// this store was just handed.
					//
					// This is the whole defence against the class of bug the
					// split LFU stack's latched branch has: that branch records
					// a tier and emits NO migration, trusting
					// `hybrid_policy::admission_tier` to have built the bytes
					// there -- and `admission_tier` consults a mirror that
					// `refresh_tier_gauges` publishes once per worker pass, so
					// under a burst it is stale and nothing ever repairs the
					// placement. Measured, on the split path: 7,999 of 8,000
					// objects physically in DRAM while the stack reported 5,966
					// slow and `fast_bytes_used` reported a compliant 31.78
					// MiB.
					//
					// Here the comparison is against the object itself, not
					// against a prediction, so a stale `admission_tier` costs
					// one corrective migration and never a wrong placement.
					// It is symmetric, so it also covers the mirror-image case
					// a grow produces -- bytes built slow, policy decides fast.
					// And it needs no new trait bound: `Object::value` and
					// `TieredValue::tier` are both inherent on the UNBOUNDED
					// impls, under both value layouts.
					let built = object.value().tier();

					let fresh = Slot {
						object: Some(object),
						hashed: key,
						prev: NIL,
						next: NIL,
						hash_next: NIL,
						last_access: now,
						tier: decided,
						referenced: AtomicU8::new(0),
						// Frequency 1, matching `ArenaFrequencyChain::insert`
						// ("admits a key at frequency 1") and
						// `LfuCompactStack::insert`'s `link_front(1, i)`.
						freq: 1,
					};

					let i = match g.free.pop() {
						Some(i) => {
							g.slots[i as usize] = fresh;
							i
						},

						// Appends a chunk when the last one is full. Nothing is
						// copied and no slot moves, so this cannot stall the
						// shard the way the old `reserve_exact` growth did.
						None => g.slots.alloc(fresh),
					};

					// Under `Lfu` the slot joins a frequency BUCKET, not the
					// recency list -- `prev`/`next` are bucket chains there, so
					// `link_front` must not run at all.
					match order {
						MergedOrder::Lfu => g.freq_link(i, 1, decided),
						_ => g.link_front(i),
					}

					g.bucket_link(i);

					// The bytes are accounted HERE, not one worker event later:
					// the object carries its own length, so admitting it and
					// charging it are the same moment. Charged to the tier the
					// policy decided, which under `Lfu` may be the slow one.
					match decided {
						Tier::Fast => {
							g.fast_count += 1;
							g.fast_used += g.slots[i as usize].migrating();
						},

						Tier::Slow => {
							g.slow_used += g.slots[i as usize].migrating();
						},
					}

					// `fast_boundary` is the prefix cursor the other three
					// orders' tiering runs on. Under `Lfu` tier membership is
					// the slot's own `tier` field and the two bucket sets are
					// already per-tier, so the cursor has no meaning and is
					// left at `NIL` for the shard's whole life.
					if order != MergedOrder::Lfu && g.fast_boundary == NIL {
						g.fast_boundary = i;
					}

					// A corrective migration exactly when the policy's tier and
					// the bytes' tier disagree, in either direction. Under the
					// queue orders they never do, so this costs nothing there.
					if built != decided {
						g.migrations.push((key, decided));
					}

					self.tracked.fetch_add(1, Ordering::Relaxed);

					None
				},
			};

			self.apply_totals_delta(before, g.totals());
			self.note_migrations(s, &g);
			self.publish_mirrors(s, &g);

			old
		};

		// Outside the guard: the settle takes one shard lock at a time and
		// this thread must not be holding another one.
		//
		// Under `Lfu` a brand-new key does NOT settle, and that asymmetry is
		// the reference's, not an optimisation. `LfuCompactHybridStack::
		// insert_resident`'s new-key path returns from BOTH of its branches
		// without calling `settle_fast_tier` at all, because admission there is
		// byte-gated and so cannot overshoot: it admits fast only while
		// `fast_used + size <= admit_effective`.
		//
		// That gate is the FULL effective capacity, while the drain target is
		// `drain_target::ratio()` of it -- 0.98 by default -- so there is a
		// legitimate band between them. Settling after an admission would drain
		// the tier down into that band and demote keys the reference keeps
		// fast: at a 60,000-byte budget and 512-byte items the gate admits 117
		// objects (59,904 B) while the target is 58,800, so an unconditional
		// settle here demoted three keys per fill and the differential test
		// diverged on the first admission past 58,800 -- which is exactly how
		// this was found.
		//
		// An overwrite DOES settle, because the reference's existing-key path
		// does: the bytes can grow in place, and nothing gates that.
		let settle = match order {
			MergedOrder::Lfu => old.is_some(),
			_ => true,
		};

		if settle {
			self.settle_tier();
		}

		old
	}

	pub fn clear(&self) {
		for (s, lock) in self.shards.iter().enumerate() {
			let mut g = lock.write().unwrap();

			g.buckets.clear();
			g.buckets.resize(INITIAL_BUCKETS, NIL);
			g.base = INITIAL_BUCKETS;
			g.split = 0;
			g.live = 0;
			g.slots.clear();
			g.free.clear();
			g.head = NIL;
			g.tail = NIL;
			g.fast_boundary = NIL;
			g.fast_used = 0;
			g.slow_used = 0;
			g.fast_count = 0;
			g.fast_buckets.clear();
			g.slow_buckets.clear();
			g.migrations.clear();

			self.publish_mirrors(s, &g);
		}

		// Every slot vector dropped above retired its objects' values into this
		// thread's epoch bag. Push them out now: a `clear` is the one moment
		// the whole cache's worth of garbage appears at once, and leaving it in
		// a local bag would keep it resident until this thread happened to pin
		// enough more times to fill it.

		self.tracked.store(0, Ordering::Relaxed);
		self.dirty_migrations.store(0, Ordering::Relaxed);

		// An empty cache has reached no capacity, so admission reopens --
		// `LfuCompactHybridStack::clear` resets `fast_tier_latched` for the
		// same reason.
		self.lfu_latched.store(false, Ordering::Relaxed);
		self.fast_used.store(0, Ordering::Relaxed);
		self.slow_used.store(0, Ordering::Relaxed);
		self.fast_count.store(0, Ordering::Relaxed);
	}

	pub fn len(&self) -> usize {
		self.tracked.load(Ordering::Relaxed)
	}

	pub fn is_empty(&self) -> bool {
		self.len() == 0
	}

	/// Drains every (key, new tier) pair that crossed the fast/slow boundary
	/// since the last call, across all shards.
	pub fn drain_migrations(&self) -> Vec<(HashedKey, Tier)> {
		// A plain load first, and the read-modify-write only once there is
		// something to collect. `PolicyWorker::apply_tier_migrations` calls this
		// once per EVENT and the overwhelmingly common answer is "nothing", so an
		// unconditional `swap` would write a shared line on every event and
		// invalidate it under every API thread trying to set its own bit.
		if self.dirty_migrations.load(Ordering::Relaxed) == 0 {
			return Vec::new();
		}

		let mut dirty = self.dirty_migrations.swap(0, Ordering::Relaxed);
		let mut out = Vec::new();

		// Only the shards that actually migrated, not all 32.
		//
		// Nothing is lost to the race with a concurrent push: a shard sets its
		// bit while holding its own write lock, and the bit is cleared only by
		// the `swap` above. A push that lands before this loop reaches that
		// shard is taken by this call (the lock orders them); one that lands
		// after leaves the bit set for the next call. The bit outliving an
		// already-drained shard costs one wasted lock and nothing else.
		while dirty != 0 {
			let s = dirty.trailing_zeros() as usize;
			dirty &= dirty - 1;

			let mut g = self.shards[s].write().unwrap();

			if !g.migrations.is_empty() {
				out.append(&mut g.migrations);
			}
		}

		out
	}

	pub fn resize_fast_tier(&self, size: CacheSize) {
		// A GROW reopens LFU admission; a shrink, or a no-op resize, does not.
		//
		// Faithful to `LfuCompactHybridStack::resize_fast_tier` INCLUDING the
		// guard, and the guard is the part with history: growing the budget is
		// a deliberate decision to make more capacity available and the fresh
		// room should be usable by new admissions rather than gated behind
		// promotions, while unlatching on a SHRINK would reopen admission at
		// exactly the moment capacity was taken away. The reference did that
		// unconditionally once, and no fidelity test caught it because none of
		// them resized.
		if size > self.fast_capacity.load(Ordering::Relaxed) {
			self.lfu_latched.store(false, Ordering::Relaxed);
		}

		self.fast_capacity.store(size, Ordering::Relaxed);
		self.settle_all();
	}

	/// DRAM reserved out of the fast tier for shared per-object metadata across
	/// both tiers, so demotion bounds total DRAM and not just fast-tier values.
	pub fn dram_reserved_bytes(&self) -> CacheSize {
		self.len() as CacheSize * self.shared_overhead.load(Ordering::Relaxed)
	}

	/// The store-level total the settle loop tests, rather than a sum over the
	/// shards: the two are the same number -- `fast_used_matches_the_shards`
	/// asserts it -- and this one costs a relaxed load instead of 32 locks.
	pub fn fast_bytes_used(&self) -> CacheSize {
		self.fast_used.load(Ordering::Relaxed)
	}

	/// Mirrored, like `fast_bytes_used` above: one relaxed load rather than 32
	/// shard read locks.
	pub fn slow_bytes_used(&self) -> CacheSize {
		self.slow_used.load(Ordering::Relaxed)
	}

	pub fn fast_object_count(&self) -> usize {
		self.fast_count.load(Ordering::Relaxed) as usize
	}

	/// Every live object is in exactly one tier, so the slow count is the
	/// tracked total less the fast one -- two relaxed loads, and no third
	/// counter that could drift on its own.
	pub fn slow_object_count(&self) -> usize {
		self.len().saturating_sub(self.fast_object_count())
	}

	/// Re-derives all three mirrored gauges from the shards and asserts the
	/// store-level atomics agree.
	///
	/// The gauges feed the fast-tier budget, so a path that moves a shard
	/// counter without reporting it does not announce itself -- the cache just
	/// admits the wrong number of objects. This is what makes that fail loudly:
	/// it is the sweep the accessors used to do, kept as the oracle the
	/// counters are checked against.
	#[cfg(test)]
	pub(crate) fn verify_gauges(&self) {
		assert_eq!(
			self.fast_bytes_used(),
			self.sum_shards(|g| g.fast_used),
			"the store-level fast_used drifted from the shards it mirrors",
		);

		assert_eq!(
			self.slow_bytes_used(),
			self.sum_shards(|g| g.slow_used),
			"the store-level slow_used drifted from the shards it mirrors",
		);

		assert_eq!(
			self.fast_object_count() as CacheSize,
			self.sum_shards(|g| g.fast_count as CacheSize),
			"the store-level fast_count drifted from the shards it mirrors",
		);
	}

	/// The sweep the three gauges above used to be, kept as the oracle
	/// `verify_gauges` checks the mirrored counters against.
	#[cfg(test)]
	fn sum_shards<F>(&self, f: F) -> CacheSize
	where
		F: Fn(&Inner<K, V>) -> CacheSize,
	{
		self.shards.iter().map(|lock| f(&lock.read().unwrap())).sum()
	}

	/// Slab, index and free-list CAPACITIES summed across shards, for
	/// attributing a measured allocation to the three things that hold it.
	///
	/// Sharding makes this worth reporting rather than deriving: 32 slabs and
	/// 32 bucket arrays each round up to their own size class independently, so
	/// the slack is real and is not visible from the object count alone.
	///
	/// The slab figure is chunks x `SLAB_CHUNK` -- a chunk is committed whole,
	/// so that is what it costs. The index figure is the bucket `Vec`'s
	/// CAPACITY, not its length: linear hashing pushes one bucket at a time, so
	/// the `Vec`'s own doubling leaves it holding between 1x and 2x the buckets
	/// in use, and that spare is allocated memory like any other. Measuring at
	/// powers of two samples the 1x end of that range.
	pub fn capacities(&self) -> (usize, usize, usize) {
		self.shards
			.iter()
			.map(|lock| {
				let g = lock.read().unwrap();
				(g.slots.capacity(), g.buckets.capacity(), g.free.capacity())
			})
			.fold((0, 0, 0), |a, b| (a.0 + b.0, a.1 + b.1, a.2 + b.2))
	}

	/// The tier `key` is currently placed in, for tests and fidelity checks.
	pub fn tier_of(&self, key: HashedKey) -> Option<Tier> {
		let g = self.shards[shard_of(key)].read().unwrap();
		let i = g.find(key)?;

		Some(g.slots[i as usize].tier)
	}
}

/// Read handle. Holds the shard guard and re-indexes on `Deref`, matching what
/// DashMap's `Ref` gives the rest of the crate.
pub struct MergedRef<'a, K, V> {
	guard: RwLockReadGuard<'a, Inner<K, V>>,
	slot: u32,
}

impl<K, V> Deref for MergedRef<'_, K, V> {
	type Target = Object<K, V>;

	fn deref(&self) -> &Object<K, V> {
		self.guard.slots[self.slot as usize].object.as_ref().expect("live slot")
	}
}

pub struct MergedRefMut<'a, K, V> {
	guard: RwLockWriteGuard<'a, Inner<K, V>>,
	slot: u32,
}

impl<K, V> Deref for MergedRefMut<'_, K, V> {
	type Target = Object<K, V>;

	fn deref(&self) -> &Object<K, V> {
		self.guard.slots[self.slot as usize].object.as_ref().expect("live slot")
	}
}

impl<K, V> DerefMut for MergedRefMut<'_, K, V> {
	fn deref_mut(&mut self) -> &mut Object<K, V> {
		self.guard.slots[self.slot as usize].object.as_mut().expect("live slot")
	}
}

impl<K, V> MergedStore<K, V> {
	/// Overrides `MERGED_UPDATE_INTERVAL` for a single store. Builder-shaped
	/// because the interval is read once and then never written -- it is the
	/// one piece of configuration on the `touch` fast path, and keeping it a
	/// plain field rather than an atomic keeps that path to a compare.
	pub fn with_update_interval(mut self, accesses: u64) -> Self {
		self.update_interval = accesses;
		self
	}
}

#[cfg(test)]
mod tests {
	use super::*;

	type Store = MergedStore<u64, crate::BufferDRAM>;

	/// Spreads the HIGH bits, which is what selects a shard.
	fn mix(i: u64) -> HashedKey {
		i.wrapping_mul(0x9E37_79B9_7F4A_7C15)
	}

	fn tiered(fast_capacity: CacheSize) -> Store {
		let s = Store::new();
		s.configure_tiering(fast_capacity, 0, DEFAULT_HIGH_PPM, DEFAULT_LOW_PPM);
		s
	}

	/// The value is `size` bytes of real allocation, because the tier
	/// accounting is now DERIVED from the object rather than reported
	/// separately -- an object built with an empty value migrates zero bytes
	/// whatever `record_size` is told.
	fn put(s: &Store, key: HashedKey, size: ObjectSize) {
		s.insert(key, Object::new(key, &vec![0u8; size as usize], None));
		s.record_size(key, size, 0);
	}

	/// What a slot of `size` bytes of value contributes to a tier, as the store
	/// counts it: the allocator's rounded figure for the object's whole
	/// allocation, not the request and not the value bytes alone.
	///
	/// Routed through the same accessor `Slot::migrating` uses, so a test
	/// cannot pass by agreeing with a formula the store no longer applies --
	/// which is exactly what would have happened here under `fused_value`.
	fn migrating_bytes(size: ObjectSize) -> CacheSize {
		crate::object::overhead::resident_object_bytes::<u64>(size) as CacheSize
	}

	/// A live key's queue-position stamp, straight out of the slot. The CLOCK
	/// tests assert on this because "did not relink" and "did not restamp" are
	/// the same claim: the stamp IS the position.
	fn stamp(s: &Store, key: HashedKey) -> u64 {
		let g = s.shards[shard_of(key)].read().unwrap();
		let i = g.find(key).expect("live key");

		g.slots[i as usize].last_access
	}

	/// A live key's CLOCK reference bit.
	fn referenced(s: &Store, key: HashedKey) -> bool {
		let g = s.shards[shard_of(key)].read().unwrap();
		let i = g.find(key).expect("live key");

		g.slots[i as usize].referenced.load(Ordering::Relaxed) != 0
	}

	/// **The point of `MergedOrder::Clock`.** A hit takes the shard's READ
	/// lock, so it cannot be blocked by -- and cannot block -- a concurrent
	/// reader of the same shard.
	///
	/// Under `Lru` the same hit runs `touch_slot` under `shards[s].write()`,
	/// which serialises every reader of that shard behind one relink; that is
	/// the measured cause of the merged store reaching 1.24x the DashMap arm's
	/// service time at sixteen clients. This test is that difference, made
	/// observable: a `MergedRef` -- the guard a concurrent GET holds -- is kept
	/// alive on this thread while another thread performs the hit. A read lock
	/// joins it immediately; a write lock waits for it.
	///
	/// The hit runs on its OWN thread rather than this one, because taking a
	/// second read guard on a thread that already holds one is not something
	/// `std::sync::RwLock` promises to allow. And it is bounded by
	/// `recv_timeout` rather than by `join`, so a regression FAILS here instead
	/// of hanging the suite.
	#[test]
	fn a_clock_hit_does_not_take_the_shard_write_lock() {
		use std::{sync::{mpsc, Arc}, time::Duration};

		let s = Arc::new(Store::new());
		s.set_order(MergedOrder::Clock);

		// Two keys in ONE shard, so the queue order is unambiguous: `a` is
		// older, `b` is newer.
		let shard = shard_of(mix(1));
		let mut keys = (1u64..).map(mix).filter(|&k| shard_of(k) == shard);
		let a = keys.next().unwrap();
		let b = keys.next().unwrap();

		put(&s, a, 128);
		put(&s, b, 128);

		assert_eq!(s.tail_key(), Some(a), "the older key is not the victim");

		// The guard a concurrent GET of `a` would be holding: `get_ref` takes
		// `shards[shard_of(key)].read()`, which is this exact lock.
		let reader = s.get_ref(&a).expect("the key was just inserted");

		let (tx, rx) = mpsc::channel();

		let hit = {
			let s = Arc::clone(&s);

			std::thread::spawn(move || {
				s.touch(a);
				let _ = tx.send(());
			})
		};

		let finished = rx.recv_timeout(Duration::from_secs(10)).is_ok();

		// Released before the assert so the worker can finish either way and
		// `join` cannot hang on a failure.
		drop(reader);
		hit.join().expect("the hit thread panicked");

		assert!(
			finished,
			"a CLOCK hit blocked behind a live reader of its own shard, so it \
			 took the shard WRITE lock -- which is the whole cost this order \
			 exists to avoid",
		);

		assert!(referenced(&s, a), "the hit did not set the reference bit");

		// And the second chance that bit bought: the hand clears it, recycles
		// `a` to the front, and evicts `b` instead.
		assert_eq!(s.tail_key(), Some(b), "the hand did not spare the referenced key");
		assert!(!referenced(&s, a), "the hand did not clear the bit it passed");
	}

	/// A CLOCK hit moves nothing. Not the links, not the stamp, not the tier --
	/// the bit, and only the bit.
	///
	/// This is the guard against the easy mistake, which is to reach for
	/// `touch_slot` on the hit path because it is right there. A hit that
	/// restamped would make `tail_key` nominate the wrong victim silently and
	/// the run would still report a plausible miss ratio; a hit that promoted
	/// would make CLOCK cost a PMEM->DRAM copy per access, which is exactly the
	/// cost it exists to avoid.
	#[test]
	fn a_clock_hit_does_not_relink_restamp_or_promote() {
		// A tight fast tier, so the older keys really are demoted and a
		// promotion would be visible.
		let s = tiered(4 * migrating_bytes(128));
		s.set_order(MergedOrder::Clock);

		// ONE shard, so the fast prefix and the demotion boundary are both
		// unambiguous and the assertions below cannot depend on how 16 keys
		// happened to scatter over 32 shards.
		let shard = shard_of(mix(1));
		let keys: Vec<HashedKey> =
			(1u64..).map(mix).filter(|&k| shard_of(k) == shard).take(16).collect();

		for &k in &keys {
			put(&s, k, 128);
		}

		let victim = s.tail_key().expect("something must be evictable");
		assert_eq!(s.tier_of(victim), Some(Tier::Slow), "the budget never bit");

		let before: Vec<u64> = keys.iter().map(|&k| stamp(&s, k)).collect();
		let clock_before = s.clock.load(Ordering::Relaxed);

		// Hit the oldest key, which is the one a relink or a promotion would
		// move the furthest.
		s.touch(victim);

		let after: Vec<u64> = keys.iter().map(|&k| stamp(&s, k)).collect();

		assert_eq!(before, after, "a CLOCK hit restamped a slot");
		assert_eq!(
			s.clock.load(Ordering::Relaxed),
			clock_before,
			"a CLOCK hit consumed a stamp, so it went through the LRU path",
		);
		assert_eq!(
			s.tier_of(victim),
			Some(Tier::Slow),
			"a CLOCK hit promoted a slow key -- promotion is the hand's job, \
			 not the hit's",
		);
		assert!(referenced(&s, victim), "the hit did not set the reference bit");

		// The hand is where the work happens: it clears the bit, recycles the
		// key to the front -- which under this store means restamping it as
		// the newest thing there is -- and promotes it.
		let spared = s.tail_key().expect("something must still be evictable");

		assert_ne!(spared, victim, "the hand did not spare the referenced key");
		assert!(
			stamp(&s, victim) > *before.iter().max().unwrap(),
			"the second chance did not restamp the key as the newest, so the \
			 32 shard lists no longer reconstruct one global order",
		);
		assert_eq!(
			s.tier_of(victim),
			Some(Tier::Fast),
			"the second chance moved the key to the head without promoting it, \
			 which leaves a slow key inside the fast prefix",
		);
	}

	/// The slot did not grow. Both const asserts at the top of this file are
	/// compile-time, so this is here to say WHY the number is 40 and to fail
	/// readably if someone reads the asserts as a formality.
	#[test]
	fn the_reference_bit_cost_no_bytes() {
		assert_eq!(
			core::mem::size_of::<Slot<u64, std::sync::Arc<[u8]>>>(),
			40,
			"the CLOCK reference bit was supposed to fit in the slot's tail \
			 padding",
		);
	}

	/// Sum of what every live slot claims to be migrating, walked directly.
	fn live_migrating(s: &Store) -> CacheSize {
		s.shards
			.iter()
			.map(|lock| {
				let g = lock.read().unwrap();
				let mut total = 0;
				let mut i = g.head;

				while i != NIL {
					total += g.slots[i as usize].migrating();
					i = g.slots[i as usize].next;
				}

				total
			})
			.sum()
	}

	/// The index is `NoHasher`d, so hashbrown's bucket index IS the key's low
	/// bits. Sharding on those would put every key in a shard into one bucket
	/// and turn each shard's map into a linked list.
	#[test]
	fn shards_on_the_high_bits() {
		// Keys differing ONLY in their low bits must spread across shards...
		let low_bit_family: Vec<usize> = (0..SHARDS as u64)
			.map(|i| shard_of(0xABCD_0000_0000_0000 | i))
			.collect();

		assert!(
			low_bit_family.iter().all(|s| *s == low_bit_family[0]),
			"low bits must NOT select the shard, or the shard choice and the \
			 bucket choice would be the same bits",
		);

		// ...and the high bits must select every shard exactly once.
		let mut seen = vec![false; SHARDS];

		for i in 0..SHARDS as u64 {
			seen[shard_of(i << (64 - SHARD_BITS))] = true;
		}

		assert!(seen.into_iter().all(|b| b), "every shard must be reachable");
	}

	/// The claim sharding has to earn: per-shard lists still yield the exact
	/// global LRU order, because each shard is ordered within itself and the
	/// oldest tail is the global tail.
	#[test]
	fn eviction_order_is_exact_lru_across_shards() {
		let s = tiered(CacheSize::MAX);
		let keys: Vec<HashedKey> = (1..=500u64).map(mix).collect();

		for &k in &keys {
			put(&s, k, 128);
		}

		// Re-touch in an order unrelated to insertion, so the expected answer
		// is the touch order and not an artefact of how the slabs filled.
		let mut order = keys.clone();
		order.rotate_left(137);

		for &k in &order {
			s.touch(k);
		}

		let mut evicted = Vec::new();

		while let Some(k) = s.tail_key() {
			assert!(s.take(&k).is_some(), "nominated victim must be present");
			evicted.push(k);
		}

		assert_eq!(evicted, order, "eviction order is not exact LRU");
		assert_eq!(s.len(), 0);
	}

	/// Tier placement walks a cursor along the list and never moves a node, so
	/// a tight fast budget must not perturb the eviction order by one position.
	#[test]
	fn tiering_never_reorders() {
		let keys: Vec<HashedKey> = (1..=800u64).map(mix).collect();

		let drain = |cap: CacheSize| {
			let s = tiered(cap);

			for (n, &k) in keys.iter().enumerate() {
				put(&s, k, 256);

				// Interleave re-touches so promotions happen too, not just the
				// demotions a monotonic fill would produce.
				if n % 3 == 0 {
					s.touch(keys[n / 3]);
				}
			}

			let mut out = Vec::new();

			while let Some(k) = s.tail_key() {
				s.take(&k);
				out.push(k);
			}

			out
		};

		let untiered = drain(CacheSize::MAX);

		for cap in [4_096u64, 65_536, 1 << 20] {
			assert_eq!(drain(cap), untiered, "fast budget {cap} reordered the list");
		}
	}

	/// Every byte is in exactly one tier, and every object is counted once.
	#[test]
	fn tier_accounting_is_conserved() {
		let per_shard = 8_192u64;
		let s = tiered(per_shard * SHARDS as CacheSize);

		for i in 1..=2_000u64 {
			put(&s, mix(i), 512);

			if i % 7 == 0 {
				s.touch(mix(i / 7));
			}

			if i % 23 == 0 {
				s.remove_key(mix(i / 23));
			}
		}

		assert_eq!(
			s.fast_bytes_used() + s.slow_bytes_used(),
			live_migrating(&s),
			"tier byte totals do not add up to what the live slots hold",
		);

		// `slow_object_count` is `len - fast_object_count`, so this identity
		// holds by construction. `verify_gauges` is the one with teeth: it
		// re-derives the fast count from the shards themselves.
		assert_eq!(
			s.fast_object_count() + s.slow_object_count(),
			s.len(),
			"tier object counts do not add up to the tracked total",
		);

		s.verify_gauges();

		// The budget is GLOBAL -- there is no per-shard share to overrun -- so
		// the bound is on the store's total.
		let capacity = per_shard * SHARDS as CacheSize;

		assert!(
			s.fast_bytes_used() <= scale(capacity, DEFAULT_HIGH_PPM),
			"the store overran its fast budget: {} > {}",
			s.fast_bytes_used(),
			scale(capacity, DEFAULT_HIGH_PPM),
		);
	}

	/// The store-level `fast_used` is what the settle loop tests without a
	/// lock, so it has to be the sum of what the shards actually hold. Every
	/// locked section that moves a shard's counter reports the difference; this
	/// is what catches one that forgets.
	#[test]
	fn fast_used_matches_the_shards() {
		let s = tiered(4_096 * SHARDS as CacheSize);

		for i in 1..=3_000u64 {
			put(&s, mix(i), 256);

			if i % 6 == 0 {
				s.touch(mix(i / 6));
			}

			if i % 11 == 0 {
				s.remove_key(mix(i / 11));
			}

			if i % 17 == 0 {
				// An overwrite with a DIFFERENT length, which is the case the
				// delta accounting in `insert` exists for.
				put(&s, mix(i / 17), 1_024);
			}

			if i % 29 == 0 {
				s.take(&mix(i / 29));
			}
		}

		let summed: CacheSize = s.sum_shards(|g| g.fast_used);

		assert_eq!(
			s.fast_bytes_used(),
			summed,
			"the store-level fast_used has drifted from the shards it mirrors",
		);

		// `slow_used` and `fast_count` are mirrored on exactly the same terms.
		s.verify_gauges();
	}

	/// The drift test for the two gauges `refresh_tier_gauges` stopped sweeping
	/// for: `slow_bytes_used` and `fast_object_count`.
	///
	/// These feed the fast-tier budget, so a counter that misses an update path
	/// does not announce itself -- the cache simply admits the wrong number of
	/// objects. So the workload below is written from the enumerated list of
	/// every path that moves a shard's `fast_used`, `slow_used` or `fast_count`,
	/// and `verify_gauges` re-derives all three from the shards after each one:
	///
	///   * insert of a NEW key                    (`insert`, the `None` arm)
	///   * overwrite with a DIFFERENT size, while the slot is FAST
	///   * overwrite with a different size while the slot is SLOW -- the one
	///     path that moves `slow_used` and neither of the other two
	///   * promotion                              (`touch_slot`)
	///   * demotion                               (`demote_boundary`)
	///   * removal                                (`remove_key` -> `retire`)
	///   * removal returning the object           (`take` -> `detach_tier`)
	///   * a fast-tier resize, which demotes in bulk
	///   * `clear`, which zeroes every shard and every mirror at once
	#[test]
	fn gauges_match_the_shards() {
		// Tight enough that inserting pushes objects over the boundary, so the
		// slow tier is populated and the slow-side paths are actually reached.
		let s = tiered(2_048 * SHARDS as CacheSize);

		// TWO key families, so the two removal paths cannot cannibalise each
		// other's targets. The first draft of this test walked one family with
		// `remove_key(mix(i / 11))` and `take(&mix(i / 13))`, and `remove_key`
		// reached every key first -- 11k always lands before 13k -- so `take`
		// returned `None` three thousand times and the test was blind to that
		// path completely. Deleting `take`'s call to `apply_totals_delta` still
		// passed. The reached-counters below are what stop that recurring.
		let a = |i: u64| mix(i);
		let b = |i: u64| mix(i + 10_000_000);

		let mut removed = 0usize;
		let mut taken = 0usize;
		let mut resized = 0usize;
		let mut promotions = 0usize;
		let mut demotions = 0usize;

		for i in 1..=3_000u64 {
			put(&s, a(i), 256);
			put(&s, b(i), 256);

			// Promotion out of the slow tier, and a relink.
			if i % 5 == 0 {
				s.touch(a(i / 5));
			}

			// An overwrite with a DIFFERENT length. Whichever tier the slot is
			// in at this instant is charged the difference, so over 3,000
			// iterations this reaches both the fast branch and the slow one --
			// and the slow branch is the only path in the store that moves
			// `slow_used` while touching neither of the other two counters.
			if i % 7 == 0 {
				put(&s, a(i / 7), 1_024);
			}

			if i % 11 == 0 && s.remove_key(a(i / 11)) {
				removed += 1;
			}

			if i % 13 == 0 && s.take(&b(i / 13)).is_some() {
				taken += 1;
			}

			// Bulk demotion, then bulk headroom.
			if i % 199 == 0 {
				s.resize_fast_tier(1_024 * SHARDS as CacheSize);
				s.verify_gauges();
				s.resize_fast_tier(2_048 * SHARDS as CacheSize);
				resized += 1;
			}

			// Which transitions actually fired, straight from the records the
			// store emitted rather than inferred from the totals.
			for (_, tier) in s.drain_migrations() {
				match tier {
					Tier::Fast => promotions += 1,
					Tier::Slow => demotions += 1,
				}
			}

			// Every iteration, not just at the end: a drift that a later
			// operation happens to cancel out would otherwise pass.
			s.verify_gauges();
		}

		// Every enumerated path was actually REACHED. Without these the
		// workload can stop exercising one and the sweep above still agrees
		// with itself, which is exactly how the first draft went blind.
		assert!(removed > 0, "remove_key never removed a live key");
		assert!(taken > 0, "take never took a live key -- that path went unchecked");
		assert!(resized > 0, "the fast tier was never resized");
		assert!(promotions > 0, "nothing was ever promoted");
		assert!(demotions > 0, "nothing was ever demoted");

		// The slow tier really was exercised -- otherwise this whole test would
		// be checking `slow_used == 0` three thousand times.
		assert!(
			s.slow_bytes_used() > 0,
			"the budget was not tight enough to demote anything, so the slow-side \
			 paths went unchecked",
		);

		assert!(s.slow_object_count() > 0, "nothing ended up in the slow tier");

		// And `clear` zeroes the mirrors along with the shards.
		s.clear();
		s.verify_gauges();

		assert_eq!(s.slow_bytes_used(), 0, "clear left slow bytes behind");
		assert_eq!(s.fast_bytes_used(), 0, "clear left fast bytes behind");
		assert_eq!(s.fast_object_count(), 0, "clear left fast objects behind");
		assert_eq!(s.slow_object_count(), 0, "clear left slow objects behind");

		// Still correct after a clear -- the mirrors and the shards resumed from
		// zero together.
		for i in 1..=500u64 {
			put(&s, mix(i), 256);
		}

		s.verify_gauges();
	}

	/// The fast region must stay a PREFIX of the list: head..=fast_boundary
	/// fast, everything after it slow. If that ever breaks, demotion picks the
	/// wrong victim and the tier split silently stops meaning recency.
	#[test]
	fn the_fast_region_stays_a_prefix() {
		let s = tiered(4_096 * SHARDS as CacheSize);

		for i in 1..=1_500u64 {
			put(&s, mix(i), 256);

			if i % 5 == 0 {
				s.touch(mix(i / 5));
			}
		}

		for lock in s.shards.iter() {
			let g = lock.read().unwrap();
			let mut i = g.head;
			let mut seen_slow = false;
			let mut counted_fast = 0usize;

			while i != NIL {
				match g.slots[i as usize].tier {
					Tier::Fast => {
						assert!(!seen_slow, "a fast slot sits behind a slow one");
						counted_fast += 1;
						assert!(
							!seen_slow && (g.fast_boundary != NIL),
							"fast slots exist but the boundary is unset",
						);
					},

					Tier::Slow => {
						if !seen_slow {
							// The slot just before the first slow one must be
							// the boundary.
							assert_eq!(
								g.slots[i as usize].prev,
								g.fast_boundary,
								"the boundary is not the last fast slot",
							);
						}

						seen_slow = true;
					},
				}

				i = g.slots[i as usize].next;
			}

			assert_eq!(counted_fast, g.fast_count, "fast_count disagrees with the list");
		}
	}

	/// `drain_migrations` visits only the shards whose dirty bit is set, so
	/// "every migration is eventually drained" is now a claim about that mask
	/// rather than about a loop over all 32 shards.
	///
	/// `migrations_agree_with_final_placement` below CANNOT check this, and
	/// that is not a guess: making shard 0 skip setting its bit leaves it
	/// stranding records forever and that test still passes, because a key
	/// whose migrations are never drained simply never enters its comparison.
	/// An absence is invisible to a test that only inspects what it was given,
	/// so this one inspects the shards directly.
	#[test]
	fn the_drain_leaves_no_shard_holding_migrations() {
		let s = tiered(2_048 * SHARDS as CacheSize);
		let mut shards_seen = [false; SHARDS];

		let mut note = |drained: Vec<(HashedKey, Tier)>| {
			for (k, _) in drained {
				shards_seen[shard_of(k)] = true;
			}
		};

		for i in 1..=4_000u64 {
			put(&s, mix(i), 256);

			if i % 3 == 0 {
				s.touch(mix(i / 3));
			}

			// Interleaved, so the mask is cleared and re-set many times over
			// rather than accumulating into one final sweep.
			if i % 50 == 0 {
				note(s.drain_migrations());
			}
		}

		note(s.drain_migrations());

		// Multi-shard, or the sweep below proves nothing.
		assert!(
			shards_seen.iter().filter(|seen| **seen).count() > SHARDS / 2,
			"the workload migrated across too few shards to be a test of the mask",
		);

		// The claim itself: nothing is left stranded behind a bit that never
		// got set.
		for (i, lock) in s.shards.iter().enumerate() {
			assert!(
				lock.read().unwrap().migrations.is_empty(),
				"shard {i} still holds migrations after a drain -- its dirty bit \
				 was never set, so the drain never visited it",
			);
		}

		assert_eq!(
			s.dirty_migrations.load(Ordering::Relaxed),
			0,
			"the dirty mask outlived the records it points at",
		);
	}

	/// Every drained migration must name a real transition, and the last
	/// migration for a key must agree with where that key actually ended up.
	#[test]
	fn migrations_agree_with_final_placement() {
		let s = tiered(8_192 * SHARDS as CacheSize);
		let mut last: HashMap<HashedKey, Tier, NoHasher> =
			HashMap::with_hasher(NoHasher::default());

		for i in 1..=3_000u64 {
			put(&s, mix(i), 256);

			if i % 4 == 0 {
				s.touch(mix(i / 4));
			}

			for (k, t) in s.drain_migrations() {
				last.insert(k, t);
			}
		}

		for (k, t) in s.drain_migrations() {
			last.insert(k, t);
		}

		assert!(!last.is_empty(), "a budget this tight must have produced migrations");

		// This workload is the most migration-heavy in the file, so it is also
		// worth asking the gauges here. Whether the DRAIN stranded anything is a
		// different question and this test cannot answer it -- see
		// `the_drain_leaves_no_shard_holding_migrations`.
		s.verify_gauges();

		let mut checked = 0usize;

		for (k, t) in &last {
			if let Some(actual) = s.tier_of(*k) {
				assert_eq!(actual, *t, "key {k:#x} was reported as {t:?} but is {actual:?}");
				checked += 1;
			}
		}

		assert!(checked > 0, "no migrated key survived to be checked");
	}

	/// memcached's `ITEM_UPDATE_INTERVAL`: a key hit repeatedly inside the
	/// interval must not be relinked, so a hot key does NOT keep taking its
	/// shard's write lock.
	#[test]
	fn the_update_interval_skips_relinks() {
		let s = Store::new().with_update_interval(1_000);
		let hot = mix(1);
		let cold = mix(2);

		put(&s, hot, 64);
		put(&s, cold, 64);

		// `cold` was inserted last and so is currently the MRU end.
		for _ in 0..100 {
			s.touch(hot);
		}

		assert_eq!(
			s.tail_key(),
			Some(hot),
			"the interval must have suppressed every relink, leaving `hot` at the LRU end",
		);

		// The same store with the interval off relinks on the first touch.
		let exact = Store::new();
		put(&exact, hot, 64);
		put(&exact, cold, 64);
		exact.touch(hot);

		assert_eq!(exact.tail_key(), Some(cold), "exact mode must relink");
	}

	/// The slab's waste is ONE PARTLY-FILLED CHUNK per shard, and no more --
	/// bounded, rather than proportional to the cache like the 1.40x `Vec`
	/// doubling gave and the 1.10x a 25% growth factor gave.
	///
	/// This was `the_slab_does_not_double`, which asserted a RATIO. A ratio is
	/// the wrong shape for a chunked slab: it is dominated by the fixed 4095
	/// slots per shard, so it is loose at 200k objects and vanishes at 20M,
	/// while the real guarantee -- bounded absolute slack -- holds at both.
	#[test]
	fn the_slab_wastes_at_most_one_chunk_per_shard() {
		let s = tiered(CacheSize::MAX);

		for i in 1..=200_000u64 {
			put(&s, mix(i), 64);
		}

		let (slab_cap, _, _) = s.capacities();
		let live = s.len();
		let bound = live + SHARDS * SLAB_CHUNK;

		assert!(
			slab_cap <= bound,
			"slab capacity {slab_cap} exceeds live {live} by more than one \
			 chunk per shard ({bound})",
		);

		// The bound above is the store-wide consequence. The per-shard
		// statement is the exact one, and it is what makes the slab's cost
		// predictable rather than merely bounded: a shard commits its live
		// count ROUNDED UP to a whole chunk, and not one slot more.
		//
		// Asserted as an equality, not an inequality. A `<=` would still hold
		// if growth started over-committing -- appending two chunks at a time,
		// say -- which is the `Vec` doubling this replaced, wearing a
		// different constant.
		let mut committed = 0usize;

		for (i, lock) in s.shards.iter().enumerate() {
			let g = lock.read().unwrap();
			let live = g.slots.len();
			let want = live.div_ceil(SLAB_CHUNK) * SLAB_CHUNK;

			assert_eq!(
				g.slots.capacity(),
				want,
				"shard {i} holds {live} slots but committed {} -- a shard commits \
				 its live count rounded up to one chunk, and nothing else",
				g.slots.capacity(),
			);

			committed += g.slots.capacity();
		}

		// And the harness reads the same figure the shards do, so a slope
		// measured through `capacities()` is measuring these chunks.
		assert_eq!(committed, slab_cap, "capacities() disagrees with the shards");
	}

	/// Growth must never move a slot: a chunk is boxed, so its address is fixed
	/// for as long as it lives, and appending more chunks cannot disturb it.
	/// The `Vec<Slot>` this replaced `realloc`ed -- under the shard write lock,
	/// on the API thread -- which is the stall being removed.
	#[test]
	fn appending_a_chunk_moves_no_slot() {
		let s = tiered(CacheSize::MAX);

		// Keys with a zero top five bits all land in shard 0, so one chunk
		// boundary is crossed after 4096 inserts rather than after 4096 x 32.
		let one_shard = |i: u64| mix(i) >> SHARD_BITS;
		let first = one_shard(1);

		put(&s, first, 64);
		assert_eq!(shard_of(first), 0);

		let address = {
			let g = s.shards[0].read().unwrap();
			let i = g.find(first).expect("just inserted");
			&g.slots[i as usize] as *const Slot<u64, crate::BufferDRAM> as usize
		};

		for i in 2..=(SLAB_CHUNK as u64 * 2 + 8) {
			put(&s, one_shard(i), 8);
		}

		let g = s.shards[0].read().unwrap();

		assert!(g.slots.capacity() >= SLAB_CHUNK * 2, "the shard never grew");

		let i = g.find(first).expect("still present");

		assert_eq!(
			&g.slots[i as usize] as *const Slot<u64, crate::BufferDRAM> as usize,
			address,
			"a slot moved when the slab grew",
		);
	}

	/// The cost claim linear hashing is here to make: an insert walks its OWN
	/// bucket's chain, plus -- when it crosses the load factor -- the one
	/// bucket it splits. Nothing else. The `grow_buckets` this replaced walked
	/// the whole recency list under the shard write lock, so a shard holding
	/// half a million keys stalled every request to it for half a million
	/// pointer chases.
	#[test]
	fn an_insert_walks_one_chain_plus_the_split_bucket() {
		let s = tiered(CacheSize::MAX);

		let mut splits = 0usize;
		let mut worst = 0usize;

		// `insert` alone, not `put`: `record_size` no longer looks the key up,
		// but keeping the measurement to the one call makes what is counted
		// unambiguous.
		for i in 1..=20_000u64 {
			let key = mix(i);
			let sh = shard_of(key);

			// The two chains this insert is ALLOWED to walk, measured before it
			// runs: its own bucket, and whichever bucket is next to split.
			let (allowed, before, buckets_before) = {
				let g = s.shards[sh].read().unwrap();
				let bucket = g.bucket_of(key);
				let own = g.chain_len(bucket);

				// The bucket a split would take, plus the new slot itself when
				// the insert links it onto that same chain before splitting it.
				let split = g.chain_len(g.split) + usize::from(bucket == g.split);

				(own + split, g.walk.load(Ordering::Relaxed), g.buckets.len())
			};

			s.insert(key, Object::new(key, &[0u8; 8], None));

			let g = s.shards[sh].read().unwrap();
			let walked = g.walk.load(Ordering::Relaxed) - before;

			assert!(
				walked <= allowed,
				"an insert walked {walked} links with only {allowed} available \
				 in its own bucket and the split bucket",
			);

			if g.buckets.len() > buckets_before {
				splits += 1;
			}

			worst = worst.max(walked);
		}

		assert!(splits > 100, "the table barely grew ({splits} splits); the bound is untested");

		// The absolute claim: the per-insert cost does not scale with the
		// table. Mean chain length is 1 at load factor 1.0, so both chains are
		// short no matter how many keys the shard holds.
		assert!(
			worst < 32,
			"the worst insert walked {worst} links -- that is a stall, not a chain",
		);
	}

	/// Every live slot must be reachable from its own bucket, exactly once,
	/// and no recycled slot may be reachable at all.
	///
	/// The invariant a chained table lives or dies on. A dangling `hash_next`
	/// into a recycled slot would resurrect a dead key -- `find` would return a
	/// slot whose `hashed` happens to still match, handing the caller an object
	/// that was deleted.
	#[test]
	fn bucket_chains_hold_exactly_the_live_slots() {
		let s = tiered(CacheSize::MAX);
		let mut alive: Vec<HashedKey> = Vec::new();

		// Churn hard enough to recycle slots and to force several doublings.
		for i in 1..=4_000u64 {
			let k = mix(i);
			put(&s, k, 128);
			alive.push(k);

			if i % 3 == 0 {
				let victim = alive.remove(i as usize % alive.len());
				assert!(s.remove_key(victim), "remove of a live key failed");
			}
		}

		for &k in &alive {
			assert!(s.contains(k), "live key {k:#x} is not reachable from its bucket");
		}

		let mut chained = 0usize;

		for lock in s.shards.iter() {
			let g = lock.read().unwrap();

			// Everything on a chain must also be on the recency list.
			let mut on_list = std::collections::HashSet::new();
			let mut i = g.head;

			while i != NIL {
				assert!(on_list.insert(i), "the recency list contains a cycle");
				i = g.slots[i as usize].next;
			}

			for b in 0..g.buckets.len() {
				let mut i = g.buckets[b];
				let mut walked = 0usize;

				while i != NIL {
					assert!(
						on_list.contains(&i),
						"slot {i} is on a chain but not on the recency list -- it \
						 is recycled and would resurrect a deleted key",
					);
					assert_eq!(
						g.bucket_of(g.slots[i as usize].hashed),
						b,
						"slot {i} is in the wrong bucket",
					);

					chained += 1;
					walked += 1;
					assert!(walked <= g.live + 1, "bucket {b} chain does not terminate");
					i = g.slots[i as usize].hash_next;
				}
			}

			assert_eq!(on_list.len(), g.live, "live count disagrees with the list");
		}

		assert_eq!(chained, s.len(), "chain membership disagrees with the tracked total");
		assert_eq!(chained, alive.len(), "chain membership disagrees with what was kept");
	}

	/// The table must actually double, and carry every key across each rehash.
	#[test]
	fn growth_rehashes_without_losing_a_key() {
		let s = tiered(CacheSize::MAX);

		let start: usize = {
			let g = s.shards[0].read().unwrap();
			g.buckets.len()
		};

		let keys: Vec<HashedKey> = (1..=20_000u64).map(mix).collect();

		for &k in &keys {
			put(&s, k, 64);
		}

		let grown: usize = {
			let g = s.shards[0].read().unwrap();
			g.buckets.len()
		};

		assert!(grown > start, "the table never grew: {start} -> {grown}");

		for &k in &keys {
			assert!(s.contains(k), "key {k:#x} was lost across a rehash");
		}

		// Load factor 1.0 means buckets never exceed 2x the live count -- one
		// doubling past the trigger -- which is the memory claim being made.
		for lock in s.shards.iter() {
			let g = lock.read().unwrap();
			assert!(
				g.buckets.len() <= 2 * g.live.max(INITIAL_BUCKETS),
				"a shard over-provisioned its table: {} buckets for {} entries",
				g.buckets.len(),
				g.live,
			);
		}
	}

	/// A taken key must leave no trace on its chain.
	#[test]
	fn taking_a_key_clears_its_chain_entry() {
		let s = tiered(CacheSize::MAX);

		// Three keys deliberately sharing one bucket, so removal has to repair
		// a predecessor link rather than just a bucket head.
		let g0 = s.shards[0].read().unwrap();
		let nb = g0.buckets.len() as u64;
		drop(g0);

		let collide: Vec<HashedKey> = (0..3u64).map(|i| (i + 1) * nb).collect();

		for &k in &collide {
			assert_eq!(shard_of(k), shard_of(collide[0]), "keys must share a shard");
			put(&s, k, 64);
		}

		{
			let g = s.shards[shard_of(collide[0])].read().unwrap();
			let b = g.bucket_of(collide[0]);
			assert_eq!(
				collide.iter().filter(|k| g.bucket_of(**k) == b).count(),
				3,
				"the keys did not actually collide",
			);
		}

		// Remove the MIDDLE of the chain.
		assert!(s.take(&collide[1]).is_some());
		assert!(!s.contains(collide[1]), "a taken key is still reachable");
		assert!(s.contains(collide[0]), "removing the middle broke the chain head");
		assert!(s.contains(collide[2]), "removing the middle orphaned the tail");

		// And reinserting it must not double-link.
		put(&s, collide[1], 64);

		let g = s.shards[shard_of(collide[0])].read().unwrap();
		let b = g.bucket_of(collide[1]);
		let mut i = g.buckets[b];
		let mut hits = 0;

		while i != NIL {
			if g.slots[i as usize].hashed == collide[1] {
				hits += 1;
			}

			i = g.slots[i as usize].hash_next;
		}

		assert_eq!(hits, 1, "the reinserted key appears {hits} times on its chain");
	}

	/// `last_access` is the FULL clock, so cross-shard ordering is a plain
	/// comparison of stamps -- and stays exact across the 2^32 boundary that
	/// used to be the truncation's wrap point.
	///
	/// Two claims, because widening the field is only half of it:
	///
	///   1. the clock does not wrap within the test, and cannot wrap in
	///      practice -- one `fetch_add` per relink, so a billion accesses a
	///      second would take 584 years;
	///   2. the order the store reports is exactly the order keys were touched
	///      in, across shards, with stamps straddling 2^32.
	///
	/// This replaces `cross_shard_ordering_survives_the_32_bit_wrap`, which
	/// hand-wrote truncated stamps either side of the wrap and asserted the
	/// wrapping difference picked the older one. There is no wrap left to
	/// survive; what has to be shown now is that there is none.
	#[test]
	fn cross_shard_ordering_is_exact_and_the_clock_never_wraps() {
		let s = tiered(CacheSize::MAX);

		// Start just below the old truncation boundary, so the run crosses it.
		let start = (1u64 << 32) - 64;
		s.clock.store(start, Ordering::Relaxed);

		let keys: Vec<HashedKey> = (1..=400u64).map(mix).collect();

		for &k in &keys {
			put(&s, k, 64);
		}

		// Re-touch everything EXCEPT the first 32, in an order unrelated to
		// insertion. The untouched ones keep stamps from below 2^32 while the
		// rest are above it, so the final order the store has to report spans
		// the old truncation boundary.
		let (kept, rest) = keys.split_at(32);
		let mut order = rest.to_vec();
		order.rotate_left(197);

		for &k in &order {
			s.touch(k);
		}

		let spanned: std::collections::HashSet<usize> =
			order.iter().map(|k| shard_of(*k)).collect();

		assert!(spanned.len() > 1, "the keys must span several shards for this to say anything");

		// The stamps really do straddle the old 32-bit boundary...
		let stamps: Vec<u64> = {
			let mut v = Vec::new();

			for lock in s.shards.iter() {
				let g = lock.read().unwrap();
				let mut i = g.head;

				while i != NIL {
					v.push(g.slots[i as usize].last_access);
					i = g.slots[i as usize].next;
				}
			}

			v
		};

		assert!(
			stamps.iter().any(|v| *v < (1 << 32)) && stamps.iter().any(|v| *v >= (1 << 32)),
			"the run did not cross 2^32, so the old truncation would not have \
			 been exercised either",
		);

		// ...and the clock is nowhere near wrapping.
		let clock = s.clock.load(Ordering::Relaxed);

		assert!(clock > start, "the clock did not advance");
		assert!(
			clock < u64::MAX / 2,
			"the clock is within a factor of two of wrapping, which the stamp \
			 comparison assumes cannot happen",
		);

		// Exact global order: drain by the store's own victim choice and get
		// back the never-touched keys in insert order, then the rest in touch
		// order -- across shards, and across 2^32.
		let mut expected = kept.to_vec();
		expected.extend_from_slice(&order);

		let mut evicted = Vec::new();

		while let Some(k) = s.tail_key() {
			assert!(s.take(&k).is_some(), "nominated victim must be present");
			evicted.push(k);
		}

		assert_eq!(evicted, expected, "cross-shard order is not exact");
	}

	/// The settle loop holds NO shard lock while it chooses, and takes exactly
	/// one while it demotes -- so touchers, inserters and settlers running at
	/// once cannot form a lock-order cycle. A regression would show up here as
	/// a hang rather than a failure, which is why the shape of this test is
	/// "everything finishes".
	///
	/// It also exercises the one thing the single-threaded tests cannot: the
	/// store-level `fast_used` being maintained correctly under contention,
	/// where two settlers may each demote an object the other has already
	/// accounted for.
	#[test]
	fn concurrent_touches_and_settles_do_not_deadlock() {
		use std::sync::Arc;

		let s = Arc::new(Store::new());
		s.configure_tiering(64 * 1_024, 0, DEFAULT_HIGH_PPM, DEFAULT_LOW_PPM);

		let threads: Vec<_> = (0..8u64)
			.map(|t| {
				let s = Arc::clone(&s);

				std::thread::spawn(move || {
					for i in 1..=2_000u64 {
						let key = mix(t * 1_000_000 + i);

						s.insert(key, Object::new(key, &[0u8; 128], None));
						s.record_size(key, 128, 0);
						s.touch(mix(t * 1_000_000 + (i / 2).max(1)));

						if i % 13 == 0 {
							s.remove_key(mix(t * 1_000_000 + i / 13));
						}

						if i % 101 == 0 {
							s.resize_fast_tier(32 * 1_024 * (1 + i % 3));
						}
					}
				})
			})
			.collect();

		for t in threads {
			t.join().expect("a worker panicked -- or the settle loop deadlocked");
		}

		assert_eq!(
			s.fast_bytes_used(),
			s.sum_shards(|g| g.fast_used),
			"the store-level fast_used drifted from the shards under contention",
		);

		// The gauges `refresh_tier_gauges` reads are maintained by the same
		// brackets, so they have to survive the same contention.
		s.verify_gauges();

		assert_eq!(
			s.fast_object_count() + s.slow_object_count(),
			s.len(),
			"the tier counts lost track of objects under contention",
		);

		// Every shard's fast region is still a prefix of its own list, which is
		// what makes the boundary mirror mean anything.
		for lock in s.shards.iter() {
			let g = lock.read().unwrap();
			let mut i = g.head;
			let mut seen_slow = false;

			while i != NIL {
				match g.slots[i as usize].tier {
					Tier::Fast => assert!(!seen_slow, "a fast slot sits behind a slow one"),
					Tier::Slow => seen_slow = true,
				}

				i = g.slots[i as usize].next;
			}
		}
	}

	/// The tier boundary is GLOBAL: the fast tier holds the globally most
	/// recently used objects, not each shard's own most recent.
	///
	/// The property is stated as a separation -- every fast stamp is newer than
	/// every slow stamp -- because that is what "global" means here and it is
	/// checkable without reconstructing the whole order. Under the per-shard
	/// budget this fails outright: each shard demotes to its own share, so a
	/// quiet shard keeps old fast objects while a busy one demotes new ones.
	#[test]
	fn the_tier_boundary_is_global_across_shards() {
		let s = tiered(CacheSize::MAX);

		// Deliberately skewed: a third of the keys go to one shard, so the
		// per-shard split would give that shard far too little room and the
		// others far too much.
		let mut keys: Vec<HashedKey> = Vec::new();

		for i in 1..=1_200u64 {
			keys.push(match i % 3 {
				0 => mix(i) >> SHARD_BITS,
				_ => mix(i),
			});
		}

		for &k in &keys {
			put(&s, k, 256);
		}

		// Room for about a quarter of them.
		s.resize_fast_tier(migrating_bytes(256) * 300);

		let mut newest_slow = 0u64;
		let mut oldest_fast = u64::MAX;
		let mut fast = 0usize;

		for lock in s.shards.iter() {
			let g = lock.read().unwrap();
			let mut i = g.head;

			while i != NIL {
				let slot = &g.slots[i as usize];

				match slot.tier {
					Tier::Fast => {
						oldest_fast = oldest_fast.min(slot.last_access);
						fast += 1;
					},

					Tier::Slow => newest_slow = newest_slow.max(slot.last_access),
				}

				i = g.slots[i as usize].next;
			}
		}

		assert!(fast > 0 && fast < keys.len(), "the budget demoted everything or nothing");

		assert!(
			newest_slow < oldest_fast,
			"a slow object (stamp {newest_slow}) is more recent than a fast one \
			 (stamp {oldest_fast}) -- the boundary is per shard, not global",
		);
	}

	/// Shrinking the fast tier at runtime must demote immediately, the way
	/// `PaperCache::set_fast_tier_size` expects of every hybrid stack.
	#[test]
	fn shrinking_the_fast_tier_demotes() {
		let s = tiered(CacheSize::MAX);

		for i in 1..=1_000u64 {
			put(&s, mix(i), 512);
		}

		s.drain_migrations();
		assert_eq!(s.slow_object_count(), 0, "an unbounded fast tier demotes nothing");

		s.resize_fast_tier(64 * SHARDS as CacheSize);

		assert!(s.slow_object_count() > 0, "shrinking the budget demoted nothing");
		assert!(
			!s.drain_migrations().is_empty(),
			"demotions happened but nothing was reported for physical migration",
		);
		assert_eq!(
			s.fast_object_count() + s.slow_object_count(),
			s.len(),
			"the resize lost track of objects",
		);
	}

	// ── MergedOrder::Lfu ─────────────────────────────────────────────────

	/// A store whose order is LFU, set BEFORE anything is inserted.
	///
	/// The order before the data, deliberately: under `Lfu` `prev`/`next` are
	/// bucket chains, so a store that admitted keys under `Lru` and was then
	/// switched would be holding a recency list the bucket code has no business
	/// touching. `MergedStackHandle::new` installs the order once at startup
	/// for exactly this reason, and `freq_link`'s `debug_assert!` is what
	/// catches the mistake.
	fn lfu(fast_capacity: CacheSize) -> Store {
		let s = Store::new();

		s.set_order(MergedOrder::Lfu);
		s.configure_tiering(fast_capacity, 0, DEFAULT_HIGH_PPM, DEFAULT_LOW_PPM);

		s
	}

	/// A live key's frequency count, straight out of the slot.
	fn freq(s: &Store, key: HashedKey) -> u16 {
		let g = s.shards[shard_of(key)].read().unwrap();
		let i = g.find(key).expect("live key");

		g.slots[i as usize].freq
	}

	/// A key that lands in shard `sh`: the mixed value shifted clear of the
	/// shard field, then the shard written into it.
	fn in_test_shard(sh: u64, i: u64) -> HashedKey {
		(mix(i) >> SHARD_BITS) | (sh << (64 - SHARD_BITS))
	}

	/// Sum of what every live slot claims to be migrating, walked through the
	/// FREQUENCY BUCKETS.
	///
	/// `live_migrating` walks the recency list, which is empty under `Lfu` --
	/// so it would report zero here, and that is the invariant rather than a
	/// limitation.
	fn live_migrating_lfu(s: &Store) -> CacheSize {
		s.shards
			.iter()
			.map(|lock| {
				let g = lock.read().unwrap();
				let mut total = 0;

				for tier in [Tier::Fast, Tier::Slow] {
					for (_, &(head, _)) in g.freq_buckets(tier).iter() {
						let mut i = head;

						while i != NIL {
							total += g.slots[i as usize].migrating();
							i = g.slots[i as usize].next;
						}
					}
				}

				total
			})
			.sum()
	}

	/// The slot did not grow. Both const asserts at the top of this file are
	/// compile-time, so this is here to say WHY the number is still 40 and to
	/// fail readably if someone reads the asserts as a formality.
	///
	/// This is the assertion the `u32` frequency counter the split path uses
	/// would have failed: 38 + 4 = 42 rounds the slot to 48.
	#[test]
	fn the_lfu_counter_costs_no_bytes() {
		assert_eq!(
			core::mem::size_of::<Slot<u64, std::sync::Arc<[u8]>>>(),
			40,
			"LFU's frequency counter was supposed to fit the slot's remaining \
			 tail padding",
		);
	}

	/// An LFU hit bumps the count and refreshes the key's position within its
	/// new bucket. It does NOT touch the recency list, which is what the other
	/// three orders live on and what this order leaves empty.
	#[test]
	fn an_lfu_hit_bumps_the_count_and_nothing_else() {
		let s = lfu(CacheSize::MAX);
		let a = mix(1);
		let b = mix(2);

		put(&s, a, 128);
		put(&s, b, 128);

		assert_eq!(freq(&s, a), 1, "admission is at frequency 1");
		assert_eq!(freq(&s, b), 1, "admission is at frequency 1");

		// Both sit at frequency 1 and `a` entered that bucket first, so `a` is
		// the victim -- the tie-break, on the one case where it is the only
		// thing deciding.
		assert_eq!(s.tail_key(), Some(a), "the earliest entrant at a frequency leaves first");

		s.touch(a);

		assert_eq!(freq(&s, a), 2, "a hit did not bump the count");
		assert_eq!(freq(&s, b), 1, "a hit bumped the wrong key");

		// `b` is now the only key at frequency 1, so the victim switched -- and
		// it switched because something COUNTED, not because something moved.
		assert_eq!(s.tail_key(), Some(b), "the bumped key is still the victim");

		for lock in s.shards.iter() {
			let g = lock.read().unwrap();

			assert_eq!(g.head, NIL, "Lfu linked the recency list");
			assert_eq!(g.tail, NIL, "Lfu linked the recency list");
			assert_eq!(g.fast_boundary, NIL, "Lfu moved the tier prefix cursor");
		}

		s.verify_gauges();
	}

	/// An overwrite is an ACCESS under this order, which is what both
	/// references do: `LfuCompactStack::insert` forwards an existing key to
	/// `update`, and `LfuCompactHybridStack::insert_resident` bumps it.
	#[test]
	fn an_lfu_overwrite_counts_as_an_access() {
		let s = lfu(CacheSize::MAX);
		let a = mix(1);
		let b = mix(2);

		put(&s, a, 128);
		put(&s, b, 128);

		assert_eq!(s.tail_key(), Some(a), "`a` entered frequency 1 first");

		put(&s, a, 128);

		assert_eq!(freq(&s, a), 2, "an overwrite did not count as an access");
		assert_eq!(s.tail_key(), Some(b), "the overwritten key is still the victim");

		s.verify_gauges();
	}

	/// The claim sharding has to earn for THIS order: per-shard frequency
	/// buckets still yield the exact global LFU order, because each shard is
	/// frequency-ordered within itself and the lowest of the 32 minima is the
	/// global minimum.
	///
	/// The expected order is written out rather than read back off the store,
	/// so this compares the store against LFU and not against itself: keys
	/// still at frequency 1 leave first in INSERTION order, then the once-hit
	/// keys in the order they were hit, then the twice-hit keys in the order of
	/// their SECOND hit -- because a bump restamps, so a key's position inside
	/// its new bucket is when it arrived there.
	#[test]
	fn eviction_order_is_exact_lfu_across_shards() {
		let s = lfu(CacheSize::MAX);
		let keys: Vec<HashedKey> = (1..=300u64).map(mix).collect();

		for &k in &keys {
			put(&s, k, 128);
		}

		let once: Vec<HashedKey> = keys.iter().copied().skip(10).step_by(7).collect();
		let twice: Vec<HashedKey> = once.iter().copied().step_by(3).collect();

		for &k in &once {
			s.touch(k);
		}

		for &k in &twice {
			s.touch(k);
		}

		let spanned: std::collections::HashSet<usize> =
			keys.iter().map(|k| shard_of(*k)).collect();

		assert!(spanned.len() > 1, "the keys must span several shards to say anything");

		let untouched: Vec<HashedKey> =
			keys.iter().copied().filter(|k| !once.contains(k)).collect();

		let hit_once: Vec<HashedKey> =
			once.iter().copied().filter(|k| !twice.contains(k)).collect();

		let mut expected = untouched;
		expected.extend_from_slice(&hit_once);
		expected.extend_from_slice(&twice);

		let mut evicted = Vec::new();

		while let Some(k) = s.tail_key() {
			assert!(s.take(&k).is_some(), "nominated victim must be present");
			evicted.push(k);
		}

		assert_eq!(evicted, expected, "eviction order is not exact LFU");

		// And it is NOT insertion order, which is what this whole sequence
		// would collapse to if `freq` were never read.
		assert_ne!(evicted, keys, "eviction order is exactly insertion order");
		assert_eq!(s.len(), 0);
	}

	/// The tier boundary is GLOBAL under this order too: demotion takes the
	/// least-frequent FAST keys across the whole store, not each shard's own.
	///
	/// This is the LFU analogue of `the_tier_boundary_is_global_across_shards`.
	/// The separation it asserts is the honest one -- every fast key's
	/// frequency is at least every slow key's -- and it holds here because
	/// nothing is admitted after the shrink. It is NOT a general invariant of
	/// the order: once admission latches, newcomers enter the slow tier at
	/// frequency 1 while fast keys sit above them.
	#[test]
	fn demotion_picks_the_lowest_frequencies_globally() {
		let s = lfu(CacheSize::MAX);

		// Two keys in each of two shards, so the demotion order is a statement
		// about the GLOBAL minimum and not about either shard's own: the two
		// lowest frequencies are in a different shard from the two highest, so
		// a per-shard rule cannot agree by accident.
		let a0 = in_test_shard(0, 1);
		let a1 = in_test_shard(0, 2);
		let b0 = in_test_shard(1, 3);
		let b1 = in_test_shard(1, 4);

		for &k in &[a0, a1, b0, b1] {
			put(&s, k, 128);
		}

		for _ in 0..4 {
			s.touch(a0);
		}

		for _ in 0..3 {
			s.touch(a1);
		}

		for _ in 0..2 {
			s.touch(b0);
		}

		s.touch(b1);

		assert_eq!(
			(freq(&s, a0), freq(&s, a1), freq(&s, b0), freq(&s, b1)),
			(5, 4, 3, 2),
			"the fixture did not produce four distinct frequencies",
		);

		s.resize_fast_tier(migrating_bytes(128) * 2);

		let counted = |want: Tier| -> Vec<(HashedKey, u16)> {
			[a0, a1, b0, b1]
				.into_iter()
				.filter(|&k| s.tier_of(k) == Some(want))
				.map(|k| (k, freq(&s, k)))
				.collect()
		};

		let fast = counted(Tier::Fast);
		let slow = counted(Tier::Slow);

		assert!(
			!fast.is_empty() && !slow.is_empty(),
			"the resize demoted everything or nothing, so the order is untested",
		);

		for &(fast_key, fast_freq) in &fast {
			for &(slow_key, slow_freq) in &slow {
				assert!(
					fast_freq >= slow_freq,
					"fast key {fast_key:#x} sits at frequency {fast_freq} while \
					 slow key {slow_key:#x} sits at {slow_freq} -- demotion did \
					 not take the globally least-frequent fast key",
				);
			}
		}

		s.verify_gauges();

		// A genuine demotion shuts admission, exactly as
		// `LfuCompactHybridStack::settle_fast_tier` latches.
		assert!(
			s.lfu_latched.load(Ordering::Relaxed),
			"a demotion did not latch admission, so a frequency-1 newcomer can \
			 take back the room it freed",
		);

		// A grow reopens it. A shrink must not -- that guard is the one the
		// reference got wrong once.
		s.resize_fast_tier(migrating_bytes(128) * 8);
		assert!(!s.lfu_latched.load(Ordering::Relaxed), "a grow did not unlatch admission");

		s.resize_fast_tier(migrating_bytes(128));
		assert!(s.lfu_latched.load(Ordering::Relaxed), "a shrink reopened admission");
	}

	/// The gauge drift test, for this order's own mutation sites.
	///
	/// Every path below goes through the same `totals()` bracket the other
	/// three orders use, so `verify_gauges` covers LFU with no changes -- this
	/// is what proves a new site did not skip the bracket. Written from the
	/// enumerated list: admission (fast AND slow), overwrite-as-access,
	/// promotion, demotion, `remove_key` and `take`.
	#[test]
	fn lfu_gauges_match_the_shards_under_churn() {
		let s = lfu(2_048 * SHARDS as CacheSize);

		let a = |i: u64| mix(i);
		let b = |i: u64| mix(i + 10_000_000);

		let mut removed = 0usize;
		let mut taken = 0usize;
		let mut promotions = 0usize;
		let mut demotions = 0usize;

		for i in 1..=1_200u64 {
			put(&s, a(i), 256);
			put(&s, b(i), 256);

			if i % 5 == 0 {
				s.touch(a(i / 5));
			}

			// An overwrite with a DIFFERENT length: the byte delta is charged
			// to whichever tier the slot is in, and the access is counted.
			if i % 7 == 0 {
				put(&s, a(i / 7), 1_024);
			}

			if i % 11 == 0 && s.remove_key(a(i / 11)) {
				removed += 1;
			}

			if i % 13 == 0 && s.take(&b(i / 13)).is_some() {
				taken += 1;
			}

			for (_, tier) in s.drain_migrations() {
				match tier {
					Tier::Fast => promotions += 1,
					Tier::Slow => demotions += 1,
				}
			}

			// Every iteration, not just at the end: a drift a later operation
			// happens to cancel out would otherwise pass.
			s.verify_gauges();
		}

		assert!(removed > 0, "remove_key never removed a live key");
		assert!(taken > 0, "take never took a live key -- that path went unchecked");
		assert!(demotions > 0, "nothing was ever demoted");
		assert!(promotions > 0, "nothing was ever promoted");

		assert!(s.slow_object_count() > 0, "nothing ended up in the slow tier");
		assert!(s.fast_object_count() > 0, "everything ended up in the slow tier");

		assert_eq!(
			s.fast_bytes_used() + s.slow_bytes_used(),
			live_migrating_lfu(&s),
			"tier byte totals do not add up to what the live slots hold",
		);

		// The recency list stayed empty through all of that, which is the
		// invariant every shared removal path could have broken silently.
		for (n, lock) in s.shards.iter().enumerate() {
			let g = lock.read().unwrap();

			assert_eq!(g.head, NIL, "shard {n} linked the recency list under Lfu");
			assert_eq!(g.tail, NIL, "shard {n} linked the recency list under Lfu");
		}

		s.clear();
		s.verify_gauges();

		assert!(!s.lfu_latched.load(Ordering::Relaxed), "clear left admission latched");
	}

	/// Concurrent bumps, admissions and settles must not deadlock, and the
	/// mirrored gauges must survive the contention -- the LFU counterpart of
	/// `concurrent_touches_and_settles_do_not_deadlock`, and the only test that
	/// exercises two settlers racing on the new `(freq, stamp)` mirror.
	#[test]
	fn concurrent_lfu_bumps_and_settles_do_not_deadlock() {
		use std::sync::Arc;

		let s = Arc::new(Store::new());
		s.set_order(MergedOrder::Lfu);
		s.configure_tiering(64 * 1_024, 0, DEFAULT_HIGH_PPM, DEFAULT_LOW_PPM);

		let threads: Vec<_> = (0..8u64)
			.map(|t| {
				let s = Arc::clone(&s);

				std::thread::spawn(move || {
					for i in 1..=1_000u64 {
						let key = mix(t * 1_000_000 + i);

						s.insert(key, Object::new(key, &[0u8; 128], None));
						s.record_size(key, 128, 0);
						s.touch(mix(t * 1_000_000 + (i / 2).max(1)));

						if i % 13 == 0 {
							s.remove_key(mix(t * 1_000_000 + i / 13));
						}

						if i % 101 == 0 {
							s.resize_fast_tier(32 * 1_024 * (1 + i % 3));
						}
					}
				})
			})
			.collect();

		for t in threads {
			t.join().expect("a worker panicked -- or the settle loop deadlocked");
		}

		s.verify_gauges();

		assert_eq!(
			s.fast_object_count() + s.slow_object_count(),
			s.len(),
			"the tier counts lost track of objects under contention",
		);

		// Every live slot is on exactly one frequency bucket of its own tier.
		// The hash-chain analogue of this is
		// `bucket_chains_hold_exactly_the_live_slots`; the frequency chains
		// need their own, because a corrupted bucket would still answer
		// `find` correctly and only produce a wrong victim.
		let mut bucketed = 0usize;

		for lock in s.shards.iter() {
			let g = lock.read().unwrap();
			let mut seen = std::collections::HashSet::new();

			for tier in [Tier::Fast, Tier::Slow] {
				for (&bucket_freq, &(head, tail)) in g.freq_buckets(tier).iter() {
					let mut i = head;
					let mut last = NIL;

					while i != NIL {
						assert!(seen.insert(i), "slot {i} is on two frequency buckets");
						assert_eq!(
							g.slots[i as usize].freq, bucket_freq,
							"slot {i} is in the bucket for a frequency it does not hold",
						);
						assert_eq!(
							g.slots[i as usize].tier, tier,
							"slot {i} is in the wrong tier's bucket set",
						);
						assert!(
							g.slots[i as usize].object.is_some(),
							"slot {i} is bucketed but holds no object -- it is recycled",
						);

						bucketed += 1;
						last = i;
						i = g.slots[i as usize].next;

						assert!(bucketed <= s.len() + 1, "a bucket chain does not terminate");
					}

					assert_eq!(tail, last, "a bucket's tail is not the end of its chain");
				}
			}
		}

		assert_eq!(bucketed, s.len(), "bucket membership disagrees with the tracked total");
	}
}

/// Measured against the same baselines, same harness: jemalloc
/// `stats.allocated`, ONE point per process, powers of two.
#[cfg(all(test, feature = "numa_jemalloc"))]
mod measure {
	use super::*;

	/// Same reader as `policy_stack::measure_overhead`, duplicated because that
	/// module is private to `worker::policy`. The `epoch` write is required:
	/// jemalloc caches these statistics per epoch.
	fn allocated_bytes() -> u64 {
		unsafe {
			let mut e: u64 = 1;
			let mut sz = core::mem::size_of::<u64>();

			tikv_jemalloc_sys::mallctl(
				c"epoch".as_ptr(),
				&mut e as *mut u64 as *mut core::ffi::c_void,
				&mut sz,
				&mut e as *mut u64 as *mut core::ffi::c_void,
				sz,
			);

			let mut allocated: usize = 0;
			let mut len = core::mem::size_of::<usize>();

			let rc = tikv_jemalloc_sys::mallctl(
				c"stats.allocated".as_ptr(),
				&mut allocated as *mut usize as *mut core::ffi::c_void,
				&mut len,
				core::ptr::null_mut(),
				0,
			);

			assert_eq!(rc, 0, "stats.allocated unavailable");
			allocated as u64
		}
	}

	/// One point per process: the slab grows a chunk at a time and the bucket
	/// array a bucket at a time, so a second point in the same process would be
	/// read off a different point on a step function. `MSTORE_N` powers of two,
	/// least-squares slope outside.
	///
	/// The slope is what `MERGED_STORE_STRUCTURE_OVERHEAD` is set from, with
	/// the size-class-rounded value cost subtracted.
	#[test]
	#[ignore]
	fn measure_merged_store_point() {
		let n: u64 = match std::env::var("MSTORE_N") {
			Ok(v) => v.parse().expect("MSTORE_N"),
			Err(_) => return,
		};

		let vsize: usize = std::env::var("MSTORE_VALUE")
			.map(|v| v.parse().expect("MSTORE_VALUE"))
			.unwrap_or(64);

		let base = allocated_bytes();
		let store: MergedStore<u64, crate::BufferDRAM> = MergedStore::new();

		for i in 0..n {
			let k = i.wrapping_mul(0x9E37_79B9_7F4A_7C15);
			store.insert(k, Object::new(k, &vec![0u8; vsize], None));
			store.record_size(k, (vsize + 16) as ObjectSize, 16);
		}

		let after = allocated_bytes();
		let held = store.len();
		core::hint::black_box(&store);

		let (slab, index, free) = store.capacities();

		println!(
			"MSTORE {} {} {} {} slab_cap={} slab_b={} index_cap={} free_cap={} slot={}",
			n,
			vsize,
			after.saturating_sub(base),
			held,
			slab,
			slab * core::mem::size_of::<Slot<u64, crate::BufferDRAM>>(),
			index,
			free,
			core::mem::size_of::<Slot<u64, crate::BufferDRAM>>(),
		);
	}

	/// The control the merged point is differenced against: the SAME objects,
	/// the same value type, the same allocator, in the map this store replaces.
	///
	/// Comparing slopes rather than absolutes is what makes the number mean
	/// something -- the value buffer and its `Arc` are identical on both sides
	/// and cancel, leaving only the structural difference. The split design's
	/// third piece, the eviction stack, is not in this control: it is the
	/// separately measured 72 B/object of `LruCompactHybridStack`, which has to
	/// be added back to get the split design's true total.
	#[test]
	#[ignore]
	fn measure_dashmap_point() {
		let n: u64 = match std::env::var("MSTORE_N") {
			Ok(v) => v.parse().expect("MSTORE_N"),
			Err(_) => return,
		};

		let vsize: usize = std::env::var("MSTORE_VALUE")
			.map(|v| v.parse().expect("MSTORE_VALUE"))
			.unwrap_or(64);

		let base = allocated_bytes();
		let map: dashmap::DashMap<HashedKey, Object<u64, crate::BufferDRAM>, NoHasher> =
			dashmap::DashMap::with_hasher(NoHasher::default());

		for i in 0..n {
			let k = i.wrapping_mul(0x9E37_79B9_7F4A_7C15);
			map.insert(k, Object::new(k, &vec![0u8; vsize], None));
		}

		let after = allocated_bytes();
		let held = map.len();
		core::hint::black_box(&map);

		println!("DMAP {} {} {} {}", n, vsize, after.saturating_sub(base), held);
	}

	/// The layout claim the saving rests on, asserted rather than asserted-in-
	/// prose: `Option<Object>` must be free, because the `NonNull` inside the
	/// object's `TieredValue` is a niche the discriminant fits in.
	#[test]
	fn the_option_in_a_slot_is_free() {
		assert_eq!(
			core::mem::size_of::<Option<Object<u64, crate::BufferDRAM>>>(),
			core::mem::size_of::<Object<u64, crate::BufferDRAM>>(),
			"Option<Object> grew: the Arc niche is no longer absorbing the discriminant",
		);
	}

	/// Everything a slot adds on top of the object it already had to store.
	#[test]
	fn print_slot_layout() {
		println!(
			"SLOTLAYOUT slot={} object={} overhead={}",
			core::mem::size_of::<Slot<u64, crate::BufferDRAM>>(),
			core::mem::size_of::<Object<u64, crate::BufferDRAM>>(),
			core::mem::size_of::<Slot<u64, crate::BufferDRAM>>()
				- core::mem::size_of::<Object<u64, crate::BufferDRAM>>(),
		);
	}
}
