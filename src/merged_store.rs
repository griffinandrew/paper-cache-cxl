/*
 * Copyright (c) Kia Shakiba
 *
 * This source code is licensed under the GNU AGPLv3 license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! `MergedStore` -- object map, recency order and tier placement in ONE
//! structure.
//!
//! The other `ObjectMapRef` shape (the default is a `DashMap`), behind
//! `merged_object_store`. A FEATURE and
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
//! because a caller releases its own shard before settling and no thread ever
//! holds two shard locks.
//!
//! Every tier change is decided on the policy worker and appended to ONE log
//! it owns (`MigrationLog`), in decision order, which `drain_tier_migrations`
//! takes whole -- so `PolicyWorker::apply_tier_migrations` performs the
//! physical `Object::set_data` moves exactly as it does for every other hybrid
//! stack, and in the order a DashMap stack's drain would have.
//!
//! # Who does what: the client publishes, the policy worker decides
//!
//! The split designs divide a `set` in two: the client writes the object map
//! and sends `Set`; the policy worker, handling the event, inserts the key into
//! its stack, charges it, places it and settles. This store is the map AND the
//! stack, and it used to do all of that on the client, under the shard lock,
//! at the insert. It now divides the work the same way (backpressure plan
//! S4), so the two stores make the same decisions at the same points and a
//! comparison between them measures the structure, not a different division
//! of work (T14 is the evidence: a scripted sequence gives identical drains,
//! victims, placements and stats in both):
//!
//! ```text
//!                 client (API thread, TTL reaper)    policy worker
//!   set, new key  publish an UNLINKED slot           link, stamp, charge, place,
//!                 (`insert`)                         settle (`worker_set`)
//!   set, existing swap the object; record the        per order: relink/promote,
//!                 byte change UNFOLDED               bit, bump/promote; settle
//!   get           read                               relink/bit/bump (`touch` ...)
//!   del, reap     take the object; a linked slot     retire the DEAD slot
//!                 goes DEAD (`take_if`)              (`retire_dead`)
//!   eviction      --                                 nominate, remove (`take_evict`)
//!   wipe          wait for the worker                clear everything, answer
//! ```
//!
//! What the client still does is the object map's own work -- a chain walk, a
//! slot write, a chain push -- which is what the DashMap map does. Every piece
//! of policy state (the lists and buckets, stamps, tiers, tier totals, the
//! boundary, the mirrors, the latch, the link count, the log) has one writer:
//! the policy worker. So a shard's list order is the order in which the
//! worker handled the events, as it is for a DashMap stack.
//!
//! Three slot states follow (see `UNLINKED`): a value the client published
//! and the worker has not linked is readable, overwritable and deletable, and
//! invisible to the policy -- not charged, not placed (`tier_of` is `None`, as
//! a DashMap stack's is for a key whose `Set` it has not handled), never
//! evicted, never settled. A value a client deleted after the link leaves a
//! DEAD slot on its list until the worker retires it at the key's `Del` or
//! `Expire` (or reaches it first in a settle or an eviction, and retires it
//! there without queueing anything) -- as a DashMap stack keeps a deleted key
//! until its `Del`. And the bytes a client's overwrite or delete changes in a
//! linked slot are UNFOLDED -- recorded beside the tier totals, and folded into
//! them at the start of every worker section that can move a slot
//! (`write_folded`), so every move is exact.
//!
//! The cost of the division: every `Set` takes the policy worker one shard
//! write lock, which a client reading the same shard queues behind (the LRU and
//! LFU hits already did), and the worker does serially what clients used to do
//! in parallel -- visible as its event backlog. While a backlog lasts the
//! values in it are unlinked: uncharged and unevictable, as a DashMap stack's
//! are untracked. And the merged store no longer has a real-time bound on the
//! fast tier of its own: a worker parked on its idle poll (up to 1 s after 5 s
//! without a set) links nothing until it wakes, exactly as a DashMap stack
//! places nothing. Only a set-path kick of the worker (S5's gate) closes that
//! window, in both stores.
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
//! the API thread is what inserts into the map:
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
		atomic::{AtomicBool, AtomicU8, AtomicU64, AtomicUsize, Ordering},
		RwLock, RwLockReadGuard, RwLockWriteGuard,
	},
};

use crate::{
	error::CacheError,
	object::{Object, ObjectSize},
	status::Cleared,
	worker::{MigrationOrigin, Placement, SetEvent, TaggedMigration, Tier},
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
///
/// Allocated through `crate::meta::Metered`, charging the store's `Meter`
/// (S5a): its nodes are counted where `BTreeMap` makes them, as the rest of
/// the store counts its structures where it grows them.
type FreqBuckets = std::collections::BTreeMap<u16, (u32, u32), crate::meta::Metered>;

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
	/// `ClockCompactStack` in this tree implements: `pop_back`, and on a set
	/// bit `push_front` with the bit cleared.
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
	pub fn from_policy(policy: &PaperPolicy) -> Result<MergedOrder, CacheError> {
		match policy {
			PaperPolicy::LruCompact
			| PaperPolicy::LruCompactHybrid => Ok(MergedOrder::Lru),

			PaperPolicy::FifoCompact
			| PaperPolicy::FifoCompactHybrid => Ok(MergedOrder::Fifo),

			PaperPolicy::ClockCompact
			| PaperPolicy::ClockCompactHybrid => Ok(MergedOrder::Clock),

			PaperPolicy::LfuCompact
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

/// `Slot::last_access` of a PUBLISHED but UNLINKED slot: in the object map --
/// findable, readable, overwritable, deletable, migratable -- but on no list
/// and in no frequency bucket, charged to no tier and counted in no `linked`.
/// A value a client has inserted whose `Set` the policy worker has not
/// handled yet; the worker's `worker_set` links it. See "Who does what" in
/// the module doc.
///
/// A sentinel, because the 40-byte slot has no spare bit. Safe because the
/// clock is a monotonic `fetch_add` from 0 that cannot reach 2^64 (see
/// `Slot::last_access`), and every reader of a stamp reaches the slot through
/// a list or a bucket, which an unlinked slot is on neither of -- except
/// `touch`'s update-interval probe, which checks. It equals `EMPTY_TAIL`, so a
/// leak into a mirror would read "empty shard", never "oldest".
///
/// The other state a slot on a hash chain can be in is DEAD: `object` is
/// `None` (a client deleted or reaped a LINKED value) while it is still on its
/// list or bucket and still counted in `linked`, until the policy worker
/// retires it at the key's `Del`/`Expire` (`retire_dead`), or reaches it first
/// in a settle or an eviction and retires it there. `find` never returns a
/// DEAD slot; `find_dead` returns nothing else.
const UNLINKED: u64 = u64::MAX;

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
	/// Whether the policy worker has linked this slot (see [`UNLINKED`]).
	/// Meaningful for a slot on a hash chain: a DEAD slot is linked.
	#[inline]
	fn is_linked(&self) -> bool {
		self.last_access != UNLINKED
	}

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

	/// The store's meter (S5a), charged a chunk and any growth of `chunks`
	/// when one is committed, and every chunk when `clear` frees them.
	meter: crate::meta::Meter,
}

impl<K, V> Slab<K, V> {
	fn new(meter: &crate::meta::Meter) -> Self {
		Slab {
			chunks: Vec::new(),
			allocated: 0,
			meter: meter.clone(),
		}
	}

	/// Usable bytes of one chunk: `Vec::with_capacity(SLAB_CHUNK)`'s buffer,
	/// which `push_chunk` boxes without reallocating.
	fn chunk_bytes() -> u64 {
		crate::meta::vec_bytes::<Slot<K, V>>(SLAB_CHUNK)
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

		let table = self.chunks.capacity();

		self.chunks.push(chunk);

		self.meter.charge(Self::chunk_bytes() as i64);
		self.meter.vec_resized::<Box<[Slot<K, V>; SLAB_CHUNK]>>(table, self.chunks.capacity());
	}

	/// Frees every chunk; the table keeps its capacity.
	fn clear(&mut self) {
		self.meter.charge(-((self.chunks.len() as u64 * Self::chunk_bytes()) as i64));
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

	/// Bytes a CLIENT moved in this shard that the tier totals above do not
	/// hold yet: an overwrite of a LINKED slot (`insert`: the new object's
	/// `migrating()` less the old one's, to the slot's tier) and a delete or
	/// reap of one (`take_if`: less its `migrating()`). Signed, since either
	/// can be negative. The policy worker FOLDS them into `fast_used` /
	/// `slow_used` at the start of every section of its that can move a slot
	/// (`MergedStore::write_folded`), so every move it makes by `migrating()`
	/// -- which reads the slot's CURRENT object -- is exact, and nothing
	/// saturates.
	///
	/// Invariant I, whenever the shard lock is free: `fast_used +
	/// unfolded_fast` is the sum of `migrating()` over the linked slots whose
	/// tier is fast (a DEAD slot's is 0), and `slow_used + unfolded_slow` the
	/// same over the slow ones. `verify_charges` checks it.
	///
	/// Not policy state: the client's bookkeeping, as the DashMap map's new
	/// value size is until the worker's `resize_key` reads it off the event.
	/// The one difference is when the worker sees it -- at its first write
	/// lock of the shard rather than at the change's own event -- which only
	/// concurrency can show.
	unfolded_fast: i64,
	unfolded_slow: i64,

	/// Slots on this shard's list or frequency buckets: LINKED ones and DEAD
	/// ones. Changed only by the policy worker, and mirrored into
	/// `MergedStore::linked` through the `totals()` bracket like the three
	/// tier totals.
	linked: usize,

	/// The store's meter (S5a), which every structure of this shard charges
	/// where it grows: the bucket array and the free list at their pushes, the
	/// slab at its chunks, the frequency maps through their allocator.
	meter: crate::meta::Meter,
}

impl<K, V> Inner<K, V> {
	fn new(meter: &crate::meta::Meter) -> Self {
		// The one allocation a new shard makes: its first buckets.
		meter.charge(crate::meta::vec_bytes::<u32>(INITIAL_BUCKETS) as i64);

		Inner {
			buckets: vec![NIL; INITIAL_BUCKETS],
			base: INITIAL_BUCKETS,
			split: 0,
			live: 0,
			slots: Slab::new(meter),
			free: Vec::new(),

			#[cfg(test)]
			walk: AtomicUsize::new(0),
			head: NIL,
			tail: NIL,
			fast_boundary: NIL,
			fast_used: 0,
			slow_used: 0,
			fast_count: 0,
			fast_buckets: FreqBuckets::new_in(crate::meta::Metered::new(meter)),
			slow_buckets: FreqBuckets::new_in(crate::meta::Metered::new(meter)),
			unfolded_fast: 0,
			unfolded_slow: 0,
			linked: 0,
			meter: meter.clone(),
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

	/// The key's LIVE slot -- linked or unlinked -- on its bucket chain.
	/// Walks past a DEAD slot of the key: after a delete and a re-set one or
	/// more can sit beside the live one until the policy worker retires them.
	/// Mean chain length 1 at load factor 1.0.
	#[inline]
	fn find(&self, key: HashedKey) -> Option<u32> {
		let mut i = self.buckets[self.bucket_of(key)];

		while i != NIL {
			#[cfg(test)]
			self.walk.fetch_add(1, Ordering::Relaxed);

			let slot = &self.slots[i as usize];

			if slot.hashed == key && slot.object.is_some() {
				return Some(i);
			}

			i = slot.hash_next;
		}

		None
	}

	/// The first DEAD slot of `key` on its chain: what `retire_dead` retires.
	/// Never a live slot, and never a free one, which is on no chain.
	fn find_dead(&self, key: HashedKey) -> Option<u32> {
		let mut i = self.buckets[self.bucket_of(key)];

		while i != NIL {
			let slot = &self.slots[i as usize];

			if slot.hashed == key && slot.object.is_none() {
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

	/// Takes slot `i` off its bucket chain, repairing the predecessor's
	/// `hash_next`. By SLOT, not by key: a DEAD slot and a live one of the
	/// same key can share a chain, and only the one being removed may go.
	fn bucket_unlink_slot(&mut self, i: u32) {
		let b = self.bucket_of(self.slots[i as usize].hashed);
		let next = self.slots[i as usize].hash_next;

		match self.buckets[b] == i {
			true => self.buckets[b] = next,

			false => {
				let mut p = self.buckets[b];

				// Slot `i` is on this chain -- the caller found it there under
				// the same guard -- so this walk ends at its predecessor.
				while self.slots[p as usize].hash_next != i {
					p = self.slots[p as usize].hash_next;
				}

				self.slots[p as usize].hash_next = next;
			},
		}

		self.slots[i as usize].hash_next = NIL;
		self.live -= 1;
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
		let capacity = self.buckets.capacity();
		self.buckets.push(NIL);
		self.meter.vec_resized::<u32>(capacity, self.buckets.capacity());

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
		self.assert_folded();

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
			// used fast slot is the nearest FAST one in front of it (S5: a
			// structural slot keeps its place with tier slow, and the cursor
			// steps over it). When the boundary was also the list tail -- every
			// slot behind it gone -- that is the same walk from the new tail.
			false => {
				if self.fast_boundary == i {
					self.fast_boundary = self.fast_at_or_before(prev);
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

	/// Moves the clients' pending byte changes into the tier totals (see
	/// `unfolded_fast`). Called only by `MergedStore::write_folded`.
	fn fold(&mut self) {
		let fast = self.fast_used as i128 + self.unfolded_fast as i128;
		let slow = self.slow_used as i128 + self.unfolded_slow as i128;

		// Invariant I: each is a sum of object sizes.
		#[cfg(any(test, debug_assertions))]
		assert!(fast >= 0 && slow >= 0, "a fold took a tier below zero: {fast} / {slow}");

		self.fast_used = fast.max(0) as CacheSize;
		self.slow_used = slow.max(0) as CacheSize;
		self.unfolded_fast = 0;
		self.unfolded_slow = 0;
	}

	/// Every policy-worker move checks it runs on a folded shard: a move by
	/// `migrating()` against totals that miss a client's change is off by
	/// that change, silently. Test and debug builds only.
	#[inline]
	fn assert_folded(&self) {
		#[cfg(any(test, debug_assertions))]
		assert!(
			self.unfolded_fast == 0 && self.unfolded_slow == 0,
			"a policy-worker move ran on a shard holding unfolded client bytes \
			 ({} / {}) -- its section must begin with write_folded",
			self.unfolded_fast,
			self.unfolded_slow,
		);
	}

	/// Takes a LINKED or DEAD slot off its list (or frequency bucket) and out
	/// of the tier accounting -- `detach_tier`, then `unlink` -- and out of
	/// `linked`. It stays on its hash chain and keeps its object.
	fn unlink_linked(&mut self, i: u32, lfu: bool) {
		self.detach_tier(i, lfu);

		// `detach_tier` already took the slot off its frequency bucket under
		// `Lfu`, and that is the only list it was on.
		if !lfu {
			self.unlink(i);
		}

		self.linked -= 1;
	}

	/// Takes slot `i` off its hash chain and returns it to the free list,
	/// handing back its object (`None` for a DEAD slot).
	///
	/// Handed back rather than dropped here, so the value's retirement happens
	/// wherever the caller drops it -- still under a pin, via `Object::drop`,
	/// which defers the free: a reader that lifted this value's pointer out
	/// from under the shard guard a moment ago and is still copying its bytes
	/// is safe. That holds on every thread that removes -- the client's
	/// delete, the TTL reaper, the policy worker's eviction.
	fn free_slot(&mut self, i: u32) -> Option<Object<K, V>> {
		self.bucket_unlink_slot(i);

		let taken = self.slots[i as usize].object.take();
		let capacity = self.free.capacity();
		self.free.push(i);
		self.meter.vec_resized::<u32>(capacity, self.free.capacity());

		taken
	}

	/// Retires a DEAD slot: off its list and its chain, and freed. The tier
	/// totals do not move -- its bytes left them when the client's delete was
	/// folded, and `detach_tier` charges a DEAD slot's `migrating()` of 0 --
	/// but `fast_count`, the boundary (or its bucket) and `linked` do. Queues
	/// nothing: a retire is an un-tracking, like the DashMap stacks' `remove`.
	fn retire_dead_slot(&mut self, i: u32, lfu: bool) {
		debug_assert!(self.slots[i as usize].object.is_none(), "retiring a live slot");

		self.unlink_linked(i, lfu);
		self.free_slot(i);
	}

	/// The nearest slot at or before `start`, toward the list's front, whose
	/// tier is FAST (`NIL` if none): the boundary's walk since S5. A structural
	/// slot keeps its place in the order with tier slow, so the cursor -- the
	/// least-recently-used fast slot -- steps over it; a DEAD slot keeps the
	/// tier it had, as before. Amortized O(1) per structural slot: the cursor
	/// only moves toward the front, and a slot it passed stays behind it until
	/// a hit relinks it at the front. The DashMap stacks' `prev_fast`.
	fn fast_at_or_before(&self, mut start: u32) -> u32 {
		while start != NIL && self.slots[start as usize].tier != Tier::Fast {
			start = self.slots[start as usize].prev;
		}

		start
	}

	/// Links slot `i` at the MRU end as a new STRUCTURAL key (S5): a value
	/// larger than an empty fast tier, built slow and placed slow -- in its
	/// place in the order, stamped `now`, unreferenced, charged to the slow
	/// tier; never the boundary. The caller counts it in `linked`.
	fn link_front_slow(&mut self, i: u32, now: u64) {
		self.assert_folded();

		{
			let s = &mut self.slots[i as usize];
			s.tier = Tier::Slow;
			s.last_access = now;
			s.referenced.store(0, Ordering::Relaxed);
		}

		self.link_front(i);

		self.slow_used += self.slots[i as usize].migrating();
	}

	/// Takes a FAST slot out of the fast set in place (S5): an overwrite whose
	/// value is larger than an empty fast tier under `Fifo` or `Clock` (the
	/// orders that keep an overwritten key where it is). Its bytes move to
	/// the slow total and the boundary steps off it. The caller queues
	/// `(key, Slow)`: the key's placement changed.
	fn demote_in_place(&mut self, i: u32) {
		self.assert_folded();

		let (migrating, prev) = {
			let s = &self.slots[i as usize];
			(s.migrating(), s.prev)
		};

		self.slots[i as usize].tier = Tier::Slow;
		self.fast_used = self.fast_used.saturating_sub(migrating);
		self.fast_count = self.fast_count.saturating_sub(1);
		self.slow_used += migrating;

		if self.fast_boundary == i {
			self.fast_boundary = self.fast_at_or_before(prev);
		}
	}

	/// `demote_in_place` under `Lfu` (S5): a fast slot overwritten with a value
	/// larger than an empty fast tier moves to the slow bucket of its own
	/// frequency, appended at its tail with a fresh stamp --
	/// `ArenaFrequencyChain::set_tier`, as `demote_freq_min` does it. The
	/// caller queues `(key, Slow)`.
	fn demote_freq_in_place(&mut self, i: u32, now: u64) {
		self.assert_folded();

		let (freq, migrating) = {
			let s = &self.slots[i as usize];
			(s.freq, s.migrating())
		};

		self.freq_unlink(i, freq, Tier::Fast);

		{
			let s = &mut self.slots[i as usize];
			s.tier = Tier::Slow;
			s.last_access = now;
		}

		self.freq_link(i, freq, Tier::Slow);

		self.fast_used = self.fast_used.saturating_sub(migrating);
		self.fast_count = self.fast_count.saturating_sub(1);
		self.slow_used += migrating;
	}

	/// Links slot `i` at the MRU end as a new FAST key: the DashMap stacks'
	/// `push_front` of a new key under `Lru`, `Fifo` and `Clock` --
	/// unreferenced, stamped `now`, charged its CURRENT object's bytes, and
	/// the boundary if nothing was fast. The caller counts it in `linked`
	/// when it was unlinked, and settles.
	fn link_front_fast(&mut self, i: u32, now: u64) {
		self.assert_folded();

		{
			let s = &mut self.slots[i as usize];
			s.tier = Tier::Fast;
			s.last_access = now;
			s.referenced.store(0, Ordering::Relaxed);
		}

		self.link_front(i);

		self.fast_used += self.slots[i as usize].migrating();
		self.fast_count += 1;

		if self.fast_boundary == NIL {
			self.fast_boundary = i;
		}
	}

	/// Links slot `i` into `tier`'s frequency-1 bucket as a new key under
	/// `Lfu`: `ArenaFrequencyChain::insert`'s "admits a key at frequency 1",
	/// appended at the bucket's tail, stamped `now`, charged to `tier`.
	fn link_freq(&mut self, i: u32, now: u64, tier: Tier) {
		self.assert_folded();

		{
			let s = &mut self.slots[i as usize];
			s.freq = 1;
			s.last_access = now;
			s.tier = tier;
		}

		self.freq_link(i, 1, tier);

		let migrating = self.slots[i as usize].migrating();

		match tier {
			Tier::Fast => {
				self.fast_used += migrating;
				self.fast_count += 1;
			},

			Tier::Slow => self.slow_used += migrating,
		}
	}

	/// Move to the MRU end and make fast, promoting from slow if needed.
	/// Returns whether it promoted.
	///
	/// Faithful port of `LruCompactHybridStack::touch_fast_key`, minus the
	/// settle and the push: the tier boundary is settled globally, with no
	/// shard lock held, so the caller drops this shard's guard and then calls
	/// `MergedStore::settle_tier` -- and only after that queues `(key, Fast)`
	/// for a promotion, and only if that settle did not demote the key again
	/// (`MigrationLog::demoted_since`). That is the DashMap stacks' rule --
	/// "pushed after settling and guarded on the key still being fast" -- and
	/// on the policy worker it costs no lock: the only thing between the
	/// promotion and the check is this thread's own settle, which records
	/// every demotion it makes in the same log. So a promotion its own settle
	/// undoes queues only the settle's `(key, Slow)`, never a `(key, Fast),
	/// (key, Slow)` pair.
	///
	/// The push is made even when the bytes are already fast, as they are on
	/// an overwrite: `set` builds an LRU value in DRAM before the policy
	/// worker handles its `Set`, so the consumer declines the entry. The
	/// no-op is the price of a guarantee. Queued migrations carry no identity
	/// -- `apply_migration` acts on whatever object holds the key when it
	/// dequeues -- so a demotion decided for the OLD object and still queued
	/// when the overwrite lands demotes the NEW one, and this entry, behind it
	/// on the key's FIFO consumer, is what restores it. Pinned by
	/// `merged_overwrite_tests::an_overwrite_is_repromoted_after_a_stale_demotion`.
	///
	/// A STRUCTURAL slot (S5: its value larger than an empty fast tier) is
	/// moved to the front all the same -- it keeps its place in the order --
	/// but with tier slow: a slow one is not promoted, and a fast one (an
	/// overwrite with a value too large, or an eff that shrank) leaves the fast
	/// set, which the caller queues as `(key, Slow)`. `LruCompactHybridStack::
	/// touch_fast_key`'s rule, move for move.
	fn touch_slot(&mut self, i: u32, now: u64, structural: bool) -> Touched {
		self.assert_folded();

		let previous_tier = self.slots[i as usize].tier;
		let already_at_front = self.head == i;
		let is_boundary = self.fast_boundary == i;

		// Read the neighbour BEFORE moving: once the slot is at the front its
		// predecessor is gone, and the boundary has to step back to whatever
		// fast slot was in front of it.
		let new_boundary_if_moved = match is_boundary && !already_at_front {
			true => self.fast_at_or_before(self.slots[i as usize].prev),
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

		if structural {
			if previous_tier != Tier::Fast {
				return Touched::default();
			}

			let migrating = self.slots[i as usize].migrating();

			self.fast_used = self.fast_used.saturating_sub(migrating);
			self.fast_count = self.fast_count.saturating_sub(1);
			self.slow_used += migrating;
			self.slots[i as usize].tier = Tier::Slow;

			// Still the boundary only if it was already at the front: then it
			// was the one fast slot, and none is left.
			if self.fast_boundary == i {
				self.fast_boundary = NIL;
			}

			return Touched { promoted: false, demoted: true };
		}

		if previous_tier != Tier::Fast {
			let migrating = self.slots[i as usize].migrating();

			self.slow_used = self.slow_used.saturating_sub(migrating);
			self.fast_used += migrating;
			self.fast_count += 1;
			self.slots[i as usize].tier = Tier::Fast;

			if self.fast_boundary == NIL {
				self.fast_boundary = i;
			}

			return Touched { promoted: true, demoted: false };
		}

		// The boundary relinked to the front with no fast slot in front of it
		// (S5: only structural ones): it is the one fast slot, and still the
		// boundary.
		if self.fast_boundary == NIL {
			self.fast_boundary = i;
		}

		Touched::default()
	}

	/// Demotes exactly ONE slot -- the boundary, the least-recently-used fast
	/// slot in this shard -- and steps the boundary back off it.
	///
	/// Nothing is searched, and because the boundary only walks along a list
	/// this never reorders anything. One step per call, rather than a drain
	/// loop, because the loop lives in `MergedStore::settle_tier` and
	/// re-chooses the shard after every step: the next victim is whichever
	/// shard's boundary is now oldest, which is what makes the demotion order
	/// global rather than per shard.
	///
	/// A DEAD boundary is RETIRED instead: its bytes already left the tier
	/// with the client's delete, and demoting it would queue a `(key, Slow)`
	/// that lands on whatever value holds the key next.
	fn demote_boundary(&mut self) -> Demote {
		self.assert_folded();

		let d = self.fast_boundary;

		if d == NIL {
			return Demote::Empty;
		}

		if self.slots[d as usize].object.is_none() {
			// `detach_tier` steps the boundary back off it, as below.
			self.retire_dead_slot(d, false);
			return Demote::Retired;
		}

		let (key, migrating, prev) = {
			let s = &self.slots[d as usize];
			(s.hashed, s.migrating(), s.prev)
		};

		self.slots[d as usize].tier = Tier::Slow;

		self.fast_used = self.fast_used.saturating_sub(migrating);
		self.fast_count = self.fast_count.saturating_sub(1);
		self.slow_used += migrating;
		self.fast_boundary = self.fast_at_or_before(prev);

		Demote::Demoted(key)
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
			linked: self.linked as CacheSize,
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
		self.assert_folded();

		let (freq, migrating) = {
			let s = &self.slots[i as usize];
			(s.freq, s.migrating())
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

		// Nothing queued here: the caller queues `(key, Fast)` after the
		// settle that follows, and only if that settle left the key fast --
		// `LfuCompactHybridStack`'s own guard, and `touch_slot`'s.
	}

	/// Demotes this shard's least-frequently-used FAST key, at its own count.
	///
	/// `LfuCompactHybridStack::settle_fast_tier` demotes `min_with_count(Fast)`
	/// and carries the count across, which is what
	/// `ArenaFrequencyChain::set_tier` does for free. A DEAD minimum is
	/// retired instead, as in `demote_boundary`.
	fn demote_freq_min(&mut self, now: u64) -> Demote {
		self.assert_folded();

		let Some(d) = self.freq_min_slot(Tier::Fast) else { return Demote::Empty };

		if self.slots[d as usize].object.is_none() {
			self.retire_dead_slot(d, true);
			return Demote::Retired;
		}

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

		Demote::Demoted(key)
	}
}

/// What one settle step did in the shard it chose.
enum Demote {
	/// A live slot left the fast tier: queue `(key, Slow)`.
	Demoted(HashedKey),

	/// The boundary (or the tier's minimum) was a DEAD slot, retired in place.
	/// Nothing queued.
	Retired,

	/// Nothing fast in the shard: the mirror named it before another section
	/// emptied it. Republish and re-choose.
	Empty,
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
	linked: CacheSize,
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

/// The policy worker's record of the tier changes it decides in this store,
/// in decision order: what the handle's `drain_tagged_migrations` hands the
/// worker, and what every store method that can decide a tier change appends
/// to.
///
/// One ordered log, owned by the one thread that decides, rather than a list
/// per shard: the per-shard lists this replaced lost the order across shards
/// (a settle that demoted k1 in shard 9 and then k2 in shard 3 drained as
/// [k2, k1]), where a DashMap stack drains in decision order. With every tier
/// change decided on the policy worker, the log is that order, and draining
/// it is a `mem::take` -- no lock, no dirty mask. A test that drives the
/// store directly passes its own.
///
/// Every entry is the store's own decision (`MigrationOrigin::Stack`): the
/// client queues nothing any more, so the only correctives are the worker's
/// reconcile's, as in the DashMap stores.
#[derive(Default)]
pub struct MigrationLog {
	entries: Vec<TaggedMigration>,

	/// `Lfu` only: settle demotions decided since the last drain -- the
	/// merged store's `drain_demotions`, `LfuCompactHybridStack`'s
	/// `pending_demotions`. Under `Lfu` a slow landing is not always a
	/// demotion (an admission refused to slow queues one too), so demotions
	/// are counted where they are decided.
	lfu_demotions: u64,
}

impl MigrationLog {
	#[inline]
	fn push(&mut self, key: HashedKey, tier: Tier) {
		self.entries.push((key, tier, MigrationOrigin::Stack));
	}

	#[inline]
	fn len(&self) -> usize {
		self.entries.len()
	}

	/// Whether `key` was queued slow since `mark` (a `len()` taken earlier):
	/// the push-after rule's "still fast" check, exact because the settle only
	/// ever demotes and records every demotion here.
	fn demoted_since(&self, mark: usize, key: HashedKey) -> bool {
		self.entries[mark..].iter().any(|&(k, tier, _)| k == key && tier == Tier::Slow)
	}

	/// Everything queued since the last take, in order.
	pub fn take_entries(&mut self) -> Vec<TaggedMigration> {
		std::mem::take(&mut self.entries)
	}

	/// `take_entries` without the origin tags (all `Stack`).
	pub fn take_untagged(&mut self) -> Vec<(HashedKey, Tier)> {
		self.take_entries().into_iter().map(|(key, tier, _)| (key, tier)).collect()
	}

	/// The `Lfu` settle demotions counted since the last take.
	pub fn take_demotions(&mut self) -> u64 {
		std::mem::take(&mut self.lfu_demotions)
	}
}

/// What `Inner::touch_slot` did to the slot's tier: promoted it (queue
/// `(key, Fast)` after the settle, if the settle leaves it fast), or -- a
/// structural slot that was fast (S5) -- took it out of the fast set (queue
/// `(key, Slow)` at once: its placement changed).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct Touched {
	promoted: bool,
	demoted: bool,
}

/// What a policy-worker section leaves to do once its shard guard is dropped:
/// queue `(key, Slow)` for a structural move out of the fast set (S5) -- first,
/// as the DashMap stacks push it where they make it, before their settle --
/// then, if asked, a settle, after which a promotion the section made is
/// queued `(key, Fast)` unless that settle demoted the key again.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct After {
	demoted: bool,
	settle: bool,
	promoted: bool,
}

impl After {
	/// Nothing to do.
	const NOTHING: After = After { demoted: false, settle: false, promoted: false };

	/// A settle, and the push of a promotion it leaves standing.
	fn settle(promoted: bool) -> After {
		After { demoted: false, settle: true, promoted }
	}

	/// After a touch: its structural move's push, the settle, its promotion's.
	fn touched(touched: Touched) -> After {
		After { demoted: touched.demoted, settle: true, promoted: touched.promoted }
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

	/// `sum(shard.linked)`: the slots the policy worker has linked (and not
	/// yet retired), on the same footing as the three totals above. What the
	/// per-object reservation, the CLOCK hand's budget, `slow_object_count`
	/// and the handle's `len` count -- the DashMap stacks' `len()`, i.e. the
	/// keys whose `Set` the worker has handled. `tracked` is the object map's
	/// count (`len()`); the two agree at quiescence.
	linked: AtomicU64,

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
	/// Written only on the policy worker -- by the settle that demotes, by an
	/// admission it refuses, by `clear_counted` and by a grow -- which admits
	/// here too, so the decision never lags it. The handle's `admission_latched()`
	/// returns it, and the worker publishes it into `status` right after the
	/// stack call that moved it, where `hybrid_policy::admission_tier` reads it
	/// to decide which tier a client builds a NEW key in -- exactly as for
	/// `LfuCompactHybridStack`. A key built before the worker reached the Set
	/// that latched is built fast and placed slow, and the reconcile of its
	/// own Set queues its corrective.
	lfu_latched: AtomicBool,

	/// Fast-tier byte budget across ALL shards, settled against globally.
	fast_capacity: AtomicU64,
	shared_overhead: AtomicU64,
	high_ppm: AtomicU64,
	low_ppm: AtomicU64,

	/// The usable bytes of the store's own structures (S5a;
	/// `structure_bytes`): charged by every shard where it grows a structure,
	/// by the frequency maps' allocator, and here for the fixed arrays.
	meter: crate::meta::Meter,

	/// S5: the measured M the policy worker pushed under the measured model
	/// (`set_dram_metadata`), which the reservation is then instead of
	/// `linked x shared_overhead`; `u64::MAX` for none (the per-object model).
	measured_metadata: AtomicU64,
}

impl<K, V> Default for MergedStore<K, V> {
	fn default() -> Self {
		let meter = crate::meta::Meter::new();

		let shards = (0..SHARDS)
			.map(|_| RwLock::new(Inner::new(&meter)))
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

		// The fixed allocations, once: the shard array, the two mirror arrays
		// and the meter's own `Arc`.
		meter.charge(
			(crate::meta::box_bytes_of_val(&*shards)
				+ crate::meta::box_bytes_of_val(&*tails)
				+ crate::meta::box_bytes_of_val(&*fast_tails)
				+ crate::meta::Meter::own_bytes()) as i64,
		);

		MergedStore {
			shards,
			tails,
			fast_tails,
			clock: AtomicU64::new(0),
			tracked: AtomicUsize::new(0),
			fast_used: AtomicU64::new(0),
			slow_used: AtomicU64::new(0),
			fast_count: AtomicU64::new(0),
			linked: AtomicU64::new(0),
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
			meter,
			measured_metadata: AtomicU64::new(u64::MAX),
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

	/// Installs the fast-tier budget and the per-object DRAM reservation.
	///
	/// Called once, when the policy worker builds its `PolicyStack` over this
	/// same `Arc` (on the constructing thread), with the values
	/// `init_policy_stack` hands the split hybrid stacks -- so the merged store
	/// is tiered on exactly the terms `LruCompactHybridStack` is. The store is
	/// empty then, so there is nothing to settle; the settle runs with the
	/// policy worker's next link. (It used to settle here, into the per-shard
	/// lists; the log it would need now is the worker's.)
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
	}

	/// S5: the measured M the policy worker pushes under the measured model;
	/// `None` restores the per-object reservation. The handle's
	/// `set_dram_metadata`.
	pub fn set_dram_metadata(&self, measured: Option<CacheSize>) {
		self.measured_metadata.store(measured.unwrap_or(u64::MAX), Ordering::Relaxed);
	}

	/// The DRAM metadata reserved out of the fast tier: the pushed M under the
	/// measured model, `linked x shared_overhead` under the per-object one.
	fn reservation(&self, shared_overhead: CacheSize) -> CacheSize {
		match self.measured_metadata.load(Ordering::Relaxed) {
			u64::MAX => self.linked() as CacheSize * shared_overhead,
			measured => measured,
		}
	}

	/// This store's eff (S5): the whole fast tier's budget for values, the
	/// settle's figure before its drain target. Untiered (a flat store), it is
	/// the sentinel capacity, and nothing is ever structural.
	fn own_eff(&self) -> CacheSize {
		let budget = self.budget();

		budget.capacity.saturating_sub(self.reservation(budget.shared_overhead))
	}

	/// Whether a value of `migrating` bytes is STRUCTURAL: larger than an
	/// empty fast tier (S5). `LruCompactHybridStack::structural`'s rule.
	fn structural(&self, migrating: CacheSize) -> bool {
		migrating > self.own_eff()
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

	/// The shard's WRITE guard, with the clients' pending byte changes
	/// (`Inner::unfolded_fast`) folded into its tier totals AND into the store
	/// totals -- how every policy-worker section that can move a slot begins,
	/// which is what keeps invariant I (on `Inner`) and every move by
	/// `migrating()` exact. The store totals follow at once, before any
	/// decision in the section, so a budget read inside it (the LFU gate, a
	/// settle step's loop test) sees this shard's newest bytes.
	///
	/// The one place a fold happens: a section that moves a slot without it
	/// trips `Inner::assert_folded` in test builds.
	fn write_folded(&self, s: usize) -> RwLockWriteGuard<'_, Inner<K, V>> {
		let mut g = self.shards[s].write().unwrap();

		if g.unfolded_fast != 0 || g.unfolded_slow != 0 {
			let before = g.totals();
			g.fold();
			self.apply_totals_delta(before, g.totals());
		}

		g
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
		apply_delta(&self.linked, before.linked, after.linked);
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
	/// Runs with NO shard lock held, on the policy worker. Each step is: 32
	/// relaxed loads to name the shard holding the oldest fast object, that
	/// ONE shard's write lock (folded), one boundary step, republish, unlock.
	/// So no thread ever holds two shard locks and no lock-order cycle can
	/// form -- a caller releases its own shard before calling this.
	///
	/// Every demotion is appended to `log`, in the order decided. A step that
	/// finds a DEAD slot where the victim would be retires it and queues
	/// nothing (`Demote::Retired`); under `Lfu` such a step latches admission
	/// as a demotion does -- the loop runs only while the tier is over its
	/// target, which is what "capacity was reached" means, and the DashMap
	/// stack at the same point demotes the deleted-but-unhandled key and
	/// latches. A step whose fold brings the tier under the target stops
	/// there, before it demotes.
	fn settle_tier(&self, log: &mut MigrationLog) {
		let budget = self.budget();

		// The reservation is per LINKED object -- or, under the measured model,
		// the pushed M (S5) -- and applies across both tiers, so it comes off
		// the budget before the watermarks are taken.
		let effective = budget
			.capacity
			.saturating_sub(self.reservation(budget.shared_overhead));

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

			let mut g = self.write_folded(s);

			// The fold can bring the tier back under the target by itself:
			// the clients' pending changes in this shard -- a delete, a
			// shrinking overwrite of a fast value -- were in `fast_used` until
			// now. A step past this point would demote one key more than the
			// target asks for. Nothing but the totals moved, and the fold
			// published those, so there is nothing to republish.
			if self.fast_used.load(Ordering::Relaxed) <= target {
				break;
			}

			let before = g.totals();

			let step = match order {
				// A demotion carries the key into a slow bucket at its own
				// count, as the newest entrant of that frequency, so it needs a
				// stamp -- see `Inner::demote_freq_min`.
				MergedOrder::Lfu => g.demote_freq_min(self.clock.fetch_add(1, Ordering::Relaxed)),
				_ => g.demote_boundary(),
			};

			match step {
				Demote::Demoted(key) => {
					log.push(key, Tier::Slow);

					// A demotion firing at all means fast-tier capacity was
					// genuinely reached, which is what shuts admission -- the
					// same rule, and the same reason, as `settle_fast_tier`'s
					// `fast_tier_latched = true`; and it is counted where it is
					// decided, as `pending_demotions` is there.
					if order == MergedOrder::Lfu {
						self.lfu_latched.store(true, Ordering::Relaxed);
						log.lfu_demotions += 1;
					}
				},

				Demote::Retired => {
					if order == MergedOrder::Lfu {
						self.lfu_latched.store(true, Ordering::Relaxed);
					}
				},

				// That shard's fast set went away between the load and the
				// lock. Republish and re-choose.
				Demote::Empty => {},
			}

			self.apply_totals_delta(before, g.totals());

			// BOTH mirrors, not just the fast one: under `Lfu` a demotion takes
			// the key out of a fast bucket AND puts it into a slow one, so the
			// victim mirror moved too. Under the other three orders a retire can
			// move the list tail as well.
			self.publish_mirrors(s, &g);
		}
	}

	/// After a section: the settle it asked for, then the push of a promotion
	/// that settle left standing (`After`).
	fn finish(&self, key: HashedKey, after: After, log: &mut MigrationLog) {
		if after.demoted {
			log.push(key, Tier::Slow);
		}

		if after.settle {
			let mark = log.len();

			self.settle_tier(log);

			if after.promoted && !log.demoted_since(mark, key) {
				log.push(key, Tier::Fast);
			}
		}
	}

	/// A hit under `Lru`: move `key` to the MRU end and make it fast; the
	/// policy worker's `update` for this store. `log` records the promotion
	/// (queued after the settle, if the settle left it fast) and whatever the
	/// settle demotes.
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
	///
	/// A hit on an UNLINKED slot -- its `Get` handled before its `Set`, which
	/// two client threads can arrange -- moves nothing, as the DashMap stacks'
	/// `update` of a key they do not track moves nothing.
	pub fn touch(&self, key: HashedKey, log: &mut MigrationLog) {
		match self.order() {
			MergedOrder::Fifo => return,

			// The whole of an LFU hit, split out for the same reason CLOCK's
			// is: it shares nothing with the body below. In particular it must
			// NOT reach the `update_interval` probe -- skipping a relink is a
			// recency approximation memcached makes deliberately, but skipping
			// a BUMP loses a count, which changes the policy rather than
			// quantising it.
			MergedOrder::Lfu => return self.bump(key, log),

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

			// The one reader of a stamp that can reach an unlinked slot (see
			// `UNLINKED`): its sentinel is not an age.
			if !g.slots[i as usize].is_linked() {
				return;
			}

			// A plain subtraction: the clock is 64 bits and monotonic, so the
			// stamp can only be at or behind it.
			let age = now.saturating_sub(g.slots[i as usize].last_access);

			if age < self.update_interval {
				return;
			}
		}

		let touched = {
			let mut g = self.write_folded(s);

			let Some(i) = g.find(key) else { return };

			if !g.slots[i as usize].is_linked() {
				return;
			}

			let before = g.totals();

			let structural = self.structural(g.slots[i as usize].migrating());
			let touched = g.touch_slot(i, now, structural);

			self.apply_totals_delta(before, g.totals());
			self.publish_mirrors(s, &g);

			touched
		};

		// AFTER the guard is dropped -- see `settle_tier`. Holding it here
		// would let the settle take a second shard lock while holding this one.
		self.finish(key, After::touched(touched), log);
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
	/// clients. Here concurrent hits to the same shard proceed in parallel.
	///
	/// The work that relink represented has not vanished, it has MOVED: the
	/// hand pays for it in `clock_victim`, under a write lock the eviction path
	/// was taking anyway, and only for the slots that actually reach the tail.
	///
	/// Called by the policy worker only (a hit's `update`, and `worker_set`
	/// for an overwrite), so the hand, which also runs there, is the only
	/// other writer of the bit. An unlinked slot is not referenced: the
	/// DashMap stack's `set_referenced` of a key it does not track does
	/// nothing either.
	pub fn mark_referenced(&self, key: HashedKey) {
		let g = self.shards[shard_of(key)].read().unwrap();

		let Some(i) = g.find(key) else { return };

		if g.slots[i as usize].is_linked() {
			g.slots[i as usize].referenced.store(1, Ordering::Relaxed);
		}
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
	/// LFU cannot avoid -- see [`MergedOrder::Lfu`]. An unlinked slot is not
	/// bumped (see `touch`).
	pub fn bump(&self, key: HashedKey, log: &mut MigrationLog) {
		let now = self.clock.fetch_add(1, Ordering::Relaxed);
		let s = shard_of(key);

		// Read with NO lock held, before this shard's is taken: the fast
		// tier's minimum frequency is `SHARDS` relaxed loads over the mirror,
		// and every locked section republishes before it releases, so on entry
		// the mirror agrees with the shards it mirrors.
		let fast_min = self.lfu_min_freq(&self.fast_tails);

		let (was_slow, promoted) = {
			let mut g = self.write_folded(s);

			let Some(i) = g.find(key) else { return };

			if !g.slots[i as usize].is_linked() {
				return;
			}

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

			// S5: never a structural slot -- its value is larger than an empty
			// fast tier; the bump counts all the same.
			let promoted = promote
				&& was_slow
				&& !self.structural(g.slots[i as usize].migrating());

			if promoted {
				let stamp = self.clock.fetch_add(1, Ordering::Relaxed);
				g.promote_freq(i, stamp);
			}

			self.apply_totals_delta(before, g.totals());
			self.publish_mirrors(s, &g);

			(was_slow, promoted)
		};

		// Only a hit that arrived on a SLOW key settles, and that is the
		// reference's asymmetry rather than a shortcut.
		// `LfuCompactHybridStack::update` settles in its `Some(Tier::Slow)` arm
		// alone -- unconditionally there, whether or not the promotion actually
		// fired -- while a hit on a fast key is a bare `chain.bump` that
		// returns without settling.
		//
		// Same underlying reason as the admission's: admission is byte-gated
		// at the full effective capacity while the drain target sits at
		// `drain_target::ratio()` of it, so a tier legitimately rests in the
		// band between the two. Settling on a FAST hit would drain it out of
		// that band and demote keys the reference keeps fast.
		//
		// AFTER the guard is dropped, as `touch` does it: the settle takes one
		// shard lock at a time and must not find this thread holding another.
		if was_slow {
			self.finish(key, After::settle(promoted), log);
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
		'choose: loop {
			for tier in [Tier::Slow, Tier::Fast] {
				let mirrors: &[TailSeq] = match tier {
					Tier::Slow => &self.tails,
					Tier::Fast => &self.fast_tails,
				};

				let Some(s) = self.lfu_min_shard(mirrors) else { continue };

				{
					let g = self.shards[s].read().unwrap();

					match g.freq_min_slot(tier) {
						Some(i) if g.slots[i as usize].object.is_some() => {
							return Some(g.slots[i as usize].hashed);
						},

						// Emptied since the mirror was read: the other tier.
						None => continue,

						// DEAD: retired below, then chosen again.
						Some(_) => {},
					}
				}

				// The tier's minimum is a DEAD slot -- deleted, its `Del` not
				// handled yet. It is not a victim (its object is gone); it is
				// retired in place, under the write lock the read lock could not
				// be upgraded to, and the choice is made again.
				let mut g = self.write_folded(s);
				let before = g.totals();

				if let Some(i) = g.freq_min_slot(tier) {
					if g.slots[i as usize].object.is_none() {
						g.retire_dead_slot(i, true);
					}
				}

				self.apply_totals_delta(before, g.totals());
				self.publish_mirrors(s, &g);

				continue 'choose;
			}

			return None;
		}
	}

	/// Where a brand-new key is admitted under `Lfu`, decided on the policy
	/// worker at its `Set`.
	///
	/// `LfuCompactHybridStack::insert_resident`'s rule, verbatim: FAST while
	/// the effective budget has room and the latch is open, SLOW once it does
	/// not -- and the first refusal LATCHES, so every later newcomer goes
	/// straight to slow whatever slack an object-granular demotion has since
	/// freed, and queues `(key, Slow)` (the one entry the DashMap stack queues
	/// for an admission; a latched one queues nothing, trusting the client to
	/// have built the value slow, and the reconcile corrects it when it did
	/// not).
	///
	/// This store's other three orders admit unconditionally fast and let the
	/// settle sort it out, and under LFU that would be WRONG rather than
	/// merely different: the settle demotes the lowest frequency, which is some
	/// older frequency-1 key, not the newcomer that caused the overflow.
	///
	/// The gate (S5, both stores at once): `fast_used + migrating <=
	/// drain_target(F - reservation)`, the reservation counting the new key
	/// (`(others + 1) x shared_overhead`, or the pushed M under the measured
	/// model). In the stacks' unit -- `migrating`, the bytes `fast_used` is
	/// kept in, where it added the `Set`'s BASE size until S5 -- and up to the
	/// SETTLE TARGET rather than eff: an admission above it would be demoted by
	/// the next settle, and the newcomer, at frequency 1, is the very minimum
	/// it would pick. `LfuCompactHybridStack`'s rule, term for term.
	///
	/// `fast_used` and `others` are the tier's fast bytes and the linked keys
	/// OTHER than this one, as the caller has them.
	fn lfu_admission(
		&self,
		key: HashedKey,
		migrating: CacheSize,
		fast_used: CacheSize,
		others: usize,
		log: &mut MigrationLog,
	) -> Tier {
		if self.lfu_latched.load(Ordering::Relaxed) {
			return Tier::Slow;
		}

		let budget = self.budget();

		let reservation = match self.measured_metadata.load(Ordering::Relaxed) {
			u64::MAX => (others as CacheSize + 1) * budget.shared_overhead,
			measured => measured,
		};

		let target = scale(budget.capacity.saturating_sub(reservation), budget.high_ppm);

		if fast_used + migrating <= target {
			return Tier::Fast;
		}

		self.lfu_latched.store(true, Ordering::Relaxed);
		log.push(key, Tier::Slow);

		Tier::Slow
	}

	/// The policy worker's handling of a `Set` of `key` -- this store's
	/// `PolicyStack::insert_set` -- with `size` the event's base size and
	/// `event` whether the map insert replaced a value (and if so whether the
	/// base size changed).
	///
	/// The client's `insert` only published the value; everything the DashMap
	/// stacks do for a `Set` on the policy worker happens here, on the policy
	/// worker:
	///
	///   * the slot is UNLINKED -- the value's first `Set` (or an earlier `Set`
	///     of the key whose value this one replaced before the worker got
	///     there): LINK it, charging its CURRENT object -- `Lru`, `Fifo` and
	///     `Clock` fast at the MRU end and settle; `Lfu` by `lfu_admission`,
	///     at frequency 1, with no settle (the reference's new-key path
	///     returns unsettled from both branches);
	///   * the slot is LINKED and the event is `Fresh` -- the map insert
	///     replaced nothing, yet an earlier `Set` of the key already linked the
	///     slot on its behalf (`set v1; del; set v2` all published before the
	///     worker took `Set(v1)`): RE-ADMIT it as a new key, as the DashMap
	///     stack does at `Set(v2)` after `Del` removed it -- relinked at the
	///     head with a fresh stamp, unreferenced, fast (`Lfu`: back to
	///     frequency 1 through `lfu_admission`, the slot itself not counted
	///     among the others);
	///   * the slot is LINKED and the event is `Replaced` -- an overwrite, per
	///     order: `Lru` relinks, restamps and promotes (`touch_slot`) and
	///     settles; `Fifo` settles if it resized a fast slot; `Clock` sets the
	///     reference bit, and settles if it resized a fast slot; `Lfu` bumps,
	///     promotes past the fast minimum, and always settles.
	///
	/// A promotion is queued after the settle, and only if the settle left
	/// the key fast (`finish`). A key with no live slot -- deleted, reaped,
	/// evicted or wiped before the worker got here -- is left alone.
	///
	/// The clients' byte changes in the shard are folded first
	/// (`write_folded`), so an overwrite's bytes are in the tier the slot is
	/// in by the time the section decides anything.
	pub fn worker_set(&self, key: HashedKey, size: ObjectSize, event: SetEvent, log: &mut MigrationLog) {
		self.worker_set_placed(key, size, event, Placement::Normal, log);
	}

	/// `worker_set`, told where the client placed the value (S5; the handle's
	/// `insert_placed`), and returning the placement APPLIED: a `Structural`
	/// set -- its value larger than an empty fast tier -- or one this store's
	/// own check finds structural (`structural`: eff moved since the client
	/// decided) is linked, relinked or overwritten slow and never promoted,
	/// in its place in the order: the DashMap stacks' rules, move for move.
	pub fn worker_set_placed(
		&self,
		key: HashedKey,
		size: ObjectSize,
		event: SetEvent,
		placement: Placement,
		log: &mut MigrationLog,
	) -> Placement {
		let order = self.order();
		let s = shard_of(key);
		let flagged = placement == Placement::Structural;

		// A same-size overwrite under `Fifo` or `Clock` moves nothing -- no
		// bytes, no link -- so a linked slot is handled under the READ lock:
		// nothing at all under `Fifo`, the reference bit under `Clock`, which a
		// read guard may set (`mark_referenced`). Not taking the write lock for
		// work that moves nothing is what `Clock` is for. An unlinked slot goes
		// on to be linked -- and so does a FAST slot the set makes structural
		// (S5), which leaves the fast set.
		if let (SetEvent::Replaced { resized: false }, MergedOrder::Fifo | MergedOrder::Clock) = (event, order) {
			let g = self.shards[s].read().unwrap();

			match g.find(key) {
				None => return placement,

				Some(i) if g.slots[i as usize].is_linked() => {
					let structural = flagged || self.structural(g.slots[i as usize].migrating());

					if !(structural && g.slots[i as usize].tier == Tier::Fast) {
						if order == MergedOrder::Clock {
							g.slots[i as usize].referenced.store(1, Ordering::Relaxed);
						}

						return match structural {
							true => Placement::Structural,
							false => placement,
						};
					}
				},

				Some(_) => {},
			}
		}

		// `Lfu` only, read with no lock held, as `bump` reads it.
		let fast_min = match order {
			MergedOrder::Lfu => self.lfu_min_freq(&self.fast_tails),
			_ => None,
		};

		let (after, structural) = {
			let mut g = self.write_folded(s);

			let Some(i) = g.find(key) else { return placement };

			let before = g.totals();

			// The slot's CURRENT object, folded: what it would cost the tier.
			let structural = flagged || self.structural(g.slots[i as usize].migrating());

			let after = match (g.slots[i as usize].is_linked(), event) {
				(false, _) => self.link(&mut g, i, key, structural, log),
				(true, SetEvent::Fresh) => self.readmit(&mut g, i, key, structural, log),
				(true, SetEvent::Replaced { resized }) => self.overwrite(&mut g, i, resized, fast_min, structural),
			};

			self.apply_totals_delta(before, g.totals());
			self.publish_mirrors(s, &g);

			(after, structural)
		};

		let _ = size;

		// Outside the guard: the settle takes one shard lock at a time and
		// this thread must not be holding another one.
		self.finish(key, after, log);

		match structural {
			true => Placement::Structural,
			false => placement,
		}
	}

	/// `worker_set` for an UNLINKED slot: its link, under the shard's write
	/// lock. The stamp is taken under the lock, so a shard's list order and
	/// stamp order agree even with several threads linking (tests do).
	///
	/// A STRUCTURAL slot (S5) is linked slow in its place -- `Lfu`: the slow
	/// bucket at frequency 1, with no latch and nothing queued (a value too
	/// large for the tier says nothing about its capacity); the others: the
	/// MRU end, tier slow -- and nothing settles (no fast byte moved).
	fn link(&self, g: &mut Inner<K, V>, i: u32, key: HashedKey, structural: bool, log: &mut MigrationLog) -> After {
		let now = self.clock.fetch_add(1, Ordering::Relaxed);

		g.linked += 1;

		match self.order() {
			MergedOrder::Lfu if structural => {
				g.link_freq(i, now, Tier::Slow);

				After::NOTHING
			},

			MergedOrder::Lfu => {
				// The totals bracket has not published this section yet, so the
				// atomics are the shard's state before the link: the others.
				let tier = self.lfu_admission(
					key,
					g.slots[i as usize].migrating(),
					self.fast_used.load(Ordering::Relaxed),
					self.linked(),
					log,
				);

				g.link_freq(i, now, tier);

				After::NOTHING
			},

			_ if structural => {
				g.link_front_slow(i, now);

				After::NOTHING
			},

			_ => {
				g.link_front_fast(i, now);

				After::settle(false)
			},
		}
	}

	/// `worker_set` for a LINKED slot whose `Set` is `Fresh`: a re-admission
	/// -- see `worker_set`. Nothing is queued but an `Lfu` refusal's
	/// `(key, Slow)`; where the value was built is the reconcile's business, as
	/// for any new key.
	fn readmit(&self, g: &mut Inner<K, V>, i: u32, key: HashedKey, structural: bool, log: &mut MigrationLog) -> After {
		let now = self.clock.fetch_add(1, Ordering::Relaxed);

		match self.order() {
			// S5: re-admitted structural, as `link` admits one.
			MergedOrder::Lfu if structural => {
				g.detach_tier(i, true);
				g.link_freq(i, now, Tier::Slow);

				After::NOTHING
			},

			MergedOrder::Lfu => {
				// The slot leaves its bucket and its tier's totals first, so the
				// gate sees the tier without it -- as the DashMap stack's does,
				// having removed the key at its `Del`.
				let (tier, migrating) = {
					let slot = &g.slots[i as usize];
					(slot.tier, slot.migrating())
				};

				g.detach_tier(i, true);

				let fast_used = match tier {
					Tier::Fast => self.fast_used.load(Ordering::Relaxed).saturating_sub(migrating),
					Tier::Slow => self.fast_used.load(Ordering::Relaxed),
				};

				let tier = self.lfu_admission(key, migrating, fast_used, self.linked() - 1, log);

				g.link_freq(i, now, tier);

				After::NOTHING
			},

			_ if structural => {
				g.detach_tier(i, false);
				g.unlink(i);
				g.link_front_slow(i, now);

				After::NOTHING
			},

			_ => {
				g.detach_tier(i, false);
				g.unlink(i);
				g.link_front_fast(i, now);

				After::settle(false)
			},
		}
	}

	/// `worker_set` for a LINKED slot whose `Set` replaced a value: the
	/// overwrite, per order -- see `worker_set`. The client already swapped
	/// the object and the fold charged its bytes to the slot's tier; `resized`
	/// is the DashMap FIFO and CLOCK stacks' criterion for settling (their
	/// stored size against the event's).
	///
	/// A STRUCTURAL overwrite (S5: the new value larger than an empty fast
	/// tier) of a FAST slot takes it out of the fast set -- in place under
	/// `Fifo` and `Clock`, at the front under `Lru` (an overwrite is a touch),
	/// in its frequency's slow bucket under `Lfu` -- and queues `(key, Slow)`:
	/// the key's placement changed, and a promotion of the old value may still
	/// be in flight. A structural slot is never promoted. The settle runs
	/// where it ran before: `Fifo`/`Clock` when a fast slot was resized, the
	/// others always.
	fn overwrite(&self, g: &mut Inner<K, V>, i: u32, resized: bool, fast_min: Option<u16>, structural: bool) -> After {
		let fast = g.slots[i as usize].tier == Tier::Fast;
		let demoted = structural && fast;

		match self.order() {
			MergedOrder::Lru => {
				let now = self.clock.fetch_add(1, Ordering::Relaxed);

				After::touched(g.touch_slot(i, now, structural))
			},

			// A resize in place and nothing else: no move to the front, no
			// promotion, no new stamp -- `FifoCompactHybridStack`'s "an existing
			// key is resized in place and NOT moved". Re-settling matters only
			// if it is fast, since only then can the resize have pushed the
			// fast tier over its budget.
			MergedOrder::Fifo => {
				if demoted {
					g.demote_in_place(i);
				}

				After { demoted, settle: resized && fast, promoted: false }
			},

			// FIFO's restraint PLUS the reference bit: `ClockCompactStack::
			// insert` forwards an existing key to `update`, which sets it, so a
			// write earns the key a second chance without moving it.
			MergedOrder::Clock => {
				g.slots[i as usize].referenced.store(1, Ordering::Relaxed);

				if demoted {
					g.demote_in_place(i);
				}

				After { demoted, settle: resized && fast, promoted: false }
			},

			// An ACCESS, as in both references: `LfuCompactStack::insert`
			// forwards an existing key to `update`, and `LfuCompactHybridStack::
			// insert_resident` bumps it (promoting a slow key past the fast
			// minimum) and then settles, whatever the tier.
			MergedOrder::Lfu => {
				let now = self.clock.fetch_add(1, Ordering::Relaxed);
				let new_freq = g.bump_slot(i, now);

				if demoted {
					let stamp = self.clock.fetch_add(1, Ordering::Relaxed);
					g.demote_freq_in_place(i, stamp);
				}

				let promoted = !fast && !structural && fast_min.is_none_or(|min| new_freq > min);

				if promoted {
					let stamp = self.clock.fetch_add(1, Ordering::Relaxed);
					g.promote_freq(i, stamp);
				}

				After { demoted, settle: true, promoted }
			},
		}
	}

	/// The eviction victim by the store's order, NOMINATED, not removed: the
	/// policy worker's `evict_one`, which `apply_evictions` pairs with
	/// `erase` -- whose `take_evict` removes it.
	///
	/// `SHARDS` relaxed atomic loads and one shard lock. Each shard's list is
	/// ordered within itself, so the global LRU object is necessarily some
	/// shard's tail, and the oldest of those tails is it. Only linked slots
	/// are on a list, so an unlinked value -- published, its `Set` not yet
	/// handled -- is never nominated, as the DashMap stack cannot pop a key
	/// it has not inserted. A DEAD slot where the victim would be is retired
	/// in place and the choice made again. `log` records a CLOCK second
	/// chance's promotion and its settle's demotions.
	pub fn tail_key(&self, log: &mut MigrationLog) -> Option<HashedKey> {
		match self.order() {
			// LRU and FIFO read the oldest tail and are done. CLOCK may have to
			// walk past a run of referenced slots first, and that walk MUTATES
			// -- see `clock_victim`.
			MergedOrder::Lru | MergedOrder::Fifo => self.oldest_tail_key(),
			MergedOrder::Clock => self.clock_victim(log),

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
		loop {
			let s = self.oldest_tail_shard()?;

			{
				let g = self.shards[s].read().unwrap();

				if g.tail != NIL && g.slots[g.tail as usize].object.is_some() {
					return Some(g.slots[g.tail as usize].hashed);
				}
			}

			// The tail is DEAD (retired here, see `lfu_victim`), or the shard
			// emptied since the mirror was read (republished, so it is not
			// chosen again).
			let mut g = self.write_folded(s);
			let before = g.totals();

			if g.tail != NIL && g.slots[g.tail as usize].object.is_none() {
				let tail = g.tail;
				g.retire_dead_slot(tail, false);
			}

			self.apply_totals_delta(before, g.totals());
			self.publish_mirrors(s, &g);
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
	/// being paid: `evict_one` nominates and `erase`'s `take_evict`
	/// immediately takes the same shard's write lock to remove the victim.
	/// The hit path is what CLOCK keeps clean -- see `mark_referenced`. A
	/// second chance that promotes queues its `(key, Fast)` after the settle
	/// behind it, and only if that settle left the key fast
	/// (`ClockCompactHybridStack::recycle_to_front`'s rule). A DEAD tail is
	/// retired, not passed.
	///
	/// # The budget
	///
	/// `clock_hand_budget(linked)` second chances, then whatever is at the
	/// tail is evicted -- the same cap `ClockCompactHybridStack::evict_one`
	/// has. It cannot fire here: each second chance clears a bit, and bits
	/// are set only on the policy worker (a hit's `update`, an overwrite's
	/// `worker_set`), which is also the thread running this loop -- so one
	/// call makes at most as many second chances as there are set bits. It
	/// used to be reachable, when the client's overwrite set bits
	/// concurrently; it stays as a liveness guard. A retired DEAD tail does
	/// not count against it.
	fn clock_victim(&self, log: &mut MigrationLog) -> Option<HashedKey> {
		enum Step {
			Victim(HashedKey),
			Chance(HashedKey, Touched),
			Retry,
			Retired,
		}

		let mut budget = crate::worker::clock_hand_budget(self.linked());

		loop {
			let s = self.oldest_tail_shard()?;

			let step = {
				let mut g = self.write_folded(s);

				match g.tail {
					// The shard emptied between the relaxed load and the lock.
					// Republish so the next choice cannot pick it again.
					NIL => {
						self.publish_mirrors(s, &g);
						Step::Retry
					},

					t if g.slots[t as usize].object.is_none() => {
						let before = g.totals();

						g.retire_dead_slot(t, false);

						self.apply_totals_delta(before, g.totals());
						self.publish_mirrors(s, &g);

						Step::Retired
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
						// being wiped by the clear. A structural slot (S5) is
						// relinked slow, and never promoted.
						g.slots[t as usize].referenced.store(0, Ordering::Relaxed);
						let structural = self.structural(g.slots[t as usize].migrating());
						let touched = g.touch_slot(t, now, structural);

						self.apply_totals_delta(before, g.totals());
						self.publish_mirrors(s, &g);

						Step::Chance(g.slots[t as usize].hashed, touched)
					},
				}
			};

			match step {
				Step::Victim(key) => return Some(key),

				// Outside the guard -- `settle_tier` takes one shard lock at a
				// time and must not find this thread holding another.
				Step::Chance(key, touched) => {
					self.finish(key, After::touched(touched), log);
					budget = budget.saturating_sub(1);
				},

				Step::Retry => budget = budget.saturating_sub(1),

				Step::Retired => {},
			}
		}
	}

	pub fn contains_key(&self, key: &HashedKey) -> bool {
		self.shards[shard_of(*key)].read().unwrap().find(*key).is_some()
	}

	pub fn contains(&self, key: HashedKey) -> bool {
		self.contains_key(&key)
	}

	/// The policy worker's EVICTION of `key`: remove and RETURN its object,
	/// taking the slot off its list and its chain and out of the tier
	/// accounting in the same operation -- the `Hashed` arm of `erase`, which
	/// `apply_evictions` pairs with `evict_one`'s nomination (and the no-key
	/// fallback).
	///
	/// This is the merge paying off directly: the split design removes from the
	/// map and separately tells the stack, and when the second half is skipped
	/// the two diverge -- the failure `ERASE_FALLBACK` in `lib.rs::erase`
	/// exists to count. Here there is one structure.
	///
	/// `None` for an UNLINKED slot: a value the worker has not linked is never
	/// evicted, as the DashMap stack cannot pop a key it has not inserted.
	/// That covers the window between a nomination and this call, too -- a
	/// client's delete and re-set of the nominated key leaves an unlinked
	/// value under it, and `erase` then answers `KeyNotFound` and the loop
	/// nominates again.
	pub fn take_evict(&self, key: &HashedKey) -> Option<Object<K, V>> {
		let s = shard_of(*key);
		let lfu = self.order() == MergedOrder::Lfu;

		let mut g = self.write_folded(s);

		let i = g.find(*key)?;

		if !g.slots[i as usize].is_linked() {
			return None;
		}

		let before = g.totals();

		g.unlink_linked(i, lfu);
		let taken = g.free_slot(i);

		self.apply_totals_delta(before, g.totals());
		self.publish_mirrors(s, &g);
		self.tracked.fetch_sub(1, Ordering::Relaxed);

		taken
	}

	/// A CLIENT's removal of `key` -- `del`, and the TTL reaper -- if `pred`
	/// holds for its object: returns the object, and changes no policy state.
	///
	/// The test and the removal run under one shard write guard, so no `set`
	/// can replace the object in between -- which is the point: `erase` checks
	/// a key match (hash collisions) and an expiry (the TTL reaper) against the
	/// object it then removes, not against whatever was there a moment ago.
	///
	///   * an UNLINKED slot -- nothing charged, nothing linked -- is freed at
	///     once: off its chain, onto the free list;
	///   * a LINKED slot goes DEAD: its object is taken, its bytes leave the
	///     tier as an unfolded change (`Inner::unfolded_fast`), and the slot
	///     stays on its chain and its list, counted in `linked` and in
	///     `fast_count`, until the policy worker retires it at the key's `Del`
	///     or `Expire` (`retire_dead`) -- as the DashMap stacks keep a deleted
	///     key until `handle_del` removes it.
	///
	/// No mirror moves, so nothing is republished: a DEAD slot keeps its
	/// stamp, and whichever nominator or settle reaches it first retires it.
	pub fn take_if(
		&self,
		key: &HashedKey,
		pred: impl FnOnce(&Object<K, V>) -> bool,
	) -> Option<Object<K, V>> {
		let mut g = self.shards[shard_of(*key)].write().unwrap();
		let i = g.find(*key)?;

		if !g.slots[i as usize].object.as_ref().is_some_and(pred) {
			return None;
		}

		let taken = match g.slots[i as usize].is_linked() {
			true => {
				let migrating = g.slots[i as usize].migrating() as i64;

				match g.slots[i as usize].tier {
					Tier::Fast => g.unfolded_fast -= migrating,
					Tier::Slow => g.unfolded_slow -= migrating,
				}

				g.slots[i as usize].object.take()
			},

			false => g.free_slot(i),
		};

		self.tracked.fetch_sub(1, Ordering::Relaxed);

		taken
	}

	/// The policy worker's retire of one DEAD slot of `key` -- its handling of
	/// the key's `Del` or `Expire`, `MergedStackHandle::remove`. `false` when
	/// there is none: the value was unlinked when the client deleted it, or a
	/// settle or a nominator reached the slot first, or the `Expire` was a
	/// reap that found the value live.
	///
	/// Never a live slot, so it is safe whatever the key holds now -- unlike
	/// the DashMap stacks' `remove`, which `handle_expire` guards on the map
	/// no longer holding the key. Each `Del` or `Expire` retires at most one:
	/// every DEAD slot is created by a delete or a reap, each of which is
	/// followed by its own event (a `del` whose `erase` found the value
	/// expired sends no `Del`, but the value's due TTL entry sends its
	/// `Expire`); a spurious `Expire`, or a settle or a nominator retiring one
	/// first, only removes one without its event. So the DEAD slots of a key
	/// never outnumber its outstanding events, and every one is retired by the
	/// time they are handled.
	pub fn retire_dead(&self, key: HashedKey) -> bool {
		let s = shard_of(key);
		let lfu = self.order() == MergedOrder::Lfu;

		let mut g = self.write_folded(s);

		let Some(j) = g.find_dead(key) else { return false };

		let before = g.totals();

		g.retire_dead_slot(j, lfu);

		self.apply_totals_delta(before, g.totals());
		self.publish_mirrors(s, &g);

		true
	}

	/// Test support: the store's structure allocations as `capacities` reads
	/// them, for the S5a test.
	#[cfg(test)]
	fn shards_chunks(&self, shard: usize) -> (usize, usize) {
		let g = self.shards[shard].read().unwrap();
		(g.slots.chunks.len(), g.slots.chunks.capacity())
	}

	/// Test support: `take_evict`, as a bool.
	#[cfg(test)]
	pub fn remove_key(&self, key: HashedKey) -> bool {
		self.take_evict(&key).is_some()
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

	/// A CLIENT's insert of `key`: publishes the object, replacing any live
	/// one, and changes no policy state.
	///
	///   * no live slot: a new UNLINKED slot is put first on the key's hash
	///     chain (a free one or a fresh one) and counted in `tracked` -- no
	///     stamp, no link, no charge, no boundary, no mirror, no settle, no
	///     latch read. The policy worker links it at the `Set` this insert is
	///     followed by (`worker_set`), deciding its tier there;
	///   * a live slot: the object is swapped, and -- for a LINKED slot -- the
	///     change in its `migrating()` is recorded as unfolded bytes of the
	///     slot's tier, which the worker folds before it moves anything in the
	///     shard. An unlinked slot records nothing: its link charges whatever
	///     object it then holds, exactly once. No relink, no reference bit, no
	///     bump, no promotion, no settle: the `Set`'s `worker_set` does those.
	///
	/// So a client holds the shard's write lock for a chain walk, a slot write
	/// and a chain push (or a swap), the DashMap map's own work, and the bytes
	/// of a value it builds are the value's; the policy work is the worker's,
	/// on the terms of the DashMap stacks. The old object is returned, and its
	/// bytes are freed wherever the caller drops it, as the DashMap insert
	/// frees them.
	pub fn insert(&self, key: HashedKey, object: Object<K, V>) -> Option<Object<K, V>> {
		let mut g = self.shards[shard_of(key)].write().unwrap();

		match g.find(key) {
			Some(i) => {
				let was = g.slots[i as usize].migrating() as i64;
				let old = g.slots[i as usize].object.replace(object);

				if g.slots[i as usize].is_linked() {
					let delta = g.slots[i as usize].migrating() as i64 - was;

					match g.slots[i as usize].tier {
						Tier::Fast => g.unfolded_fast += delta,
						Tier::Slow => g.unfolded_slow += delta,
					}
				}

				old
			},

			None => {
				let fresh = Slot {
					object: Some(object),
					hashed: key,
					prev: NIL,
					next: NIL,
					hash_next: NIL,
					last_access: UNLINKED,
					// Meaningless until the link, which writes all three.
					tier: Tier::Fast,
					referenced: AtomicU8::new(0),
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

				// First on its chain, so it is found before any DEAD slot of
				// the same key left by an earlier delete.
				g.bucket_link(i);

				self.tracked.fetch_add(1, Ordering::Relaxed);

				None
			},
		}
	}

	/// Empties the store -- the policy worker's handling of a `Wipe`
	/// (`PolicyWorker::handle_wipe`), which acknowledges it to the client
	/// waiting in `PaperCache::wipe` -- and returns what it removed: the live
	/// objects, and the base bytes `base_size` gives each, for
	/// `AtomicStatus::clear` to take off (`ObjectStore::clear_counted`'s
	/// twin).
	///
	/// Shard by shard, each under its own lock and each reporting its change
	/// to the store totals through the same before/after bracket as every
	/// other section -- not by storing 0 into them at the end: a client's
	/// insert into a shard already cleared is counted in `tracked` and stays
	/// counted, and the worker's link of it (its `Set` follows the `Wipe` in
	/// the channel) lands in `linked` and the tier totals the same way; and
	/// it is not in what this returns, so the status keeps counting it too.
	/// A live object is an unlinked or a linked slot's; a DEAD slot's was
	/// taken by the client's delete, which took it off the status then.
	pub fn clear_counted(&self, base_size: impl Fn(&Object<K, V>) -> ObjectSize) -> Cleared {
		let mut cleared = Cleared::default();

		for (s, lock) in self.shards.iter().enumerate() {
			let mut g = lock.write().unwrap();
			let before = g.totals();

			let (objects, base_bytes) = (0..g.slots.allocated)
				.filter_map(|i| g.slots[i].object.as_ref())
				.fold((0usize, 0 as CacheSize), |(n, bytes), object| (n + 1, bytes + base_size(object) as CacheSize));

			// Keeps its capacity, which is at least INITIAL_BUCKETS; counted
			// all the same, so the meter stays right if that ever changes.
			let buckets = g.buckets.capacity();
			g.buckets.clear();
			g.buckets.resize(INITIAL_BUCKETS, NIL);
			g.meter.vec_resized::<u32>(buckets, g.buckets.capacity());
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
			g.unfolded_fast = 0;
			g.unfolded_slow = 0;
			g.linked = 0;

			self.apply_totals_delta(before, g.totals());
			self.publish_mirrors(s, &g);
			self.tracked.fetch_sub(objects, Ordering::Relaxed);

			cleared.objects += objects as u64;
			cleared.base_bytes += base_bytes;
		}

		// Every slot vector dropped above retired its objects' values into this
		// thread's epoch bag, which the policy worker's pass flushes.

		// An empty cache has reached no capacity, so admission reopens --
		// `LfuCompactHybridStack::clear` resets `fast_tier_latched` for the
		// same reason.
		self.lfu_latched.store(false, Ordering::Relaxed);

		cleared
	}

	/// Live objects in the object map: published, linked or not. The DashMap
	/// map's `len()`.
	pub fn len(&self) -> usize {
		self.tracked.load(Ordering::Relaxed)
	}

	/// Usable bytes the store's own structures hold (S5a), one load: every
	/// shard's bucket array, slab chunks and chunk table, free list and LFU
	/// frequency maps, the shard array, the two tail-mirror arrays and the
	/// meter's own allocation -- each charged where it is allocated or freed,
	/// on whichever thread that is (a client's insert commits chunks and
	/// splits buckets, the policy worker's retirements fill free lists). The
	/// store's part of M: in this build the object map and the eviction stack
	/// are this one structure. Not the `Arc` it lives in (the policy worker
	/// adds that) and not the value headers (M counts one per live object).
	pub fn structure_bytes(&self) -> u64 {
		self.meter.bytes()
	}

	/// Slots the policy worker has linked and not yet retired -- the DashMap
	/// stacks' `len()` (see the field).
	pub fn linked(&self) -> usize {
		self.linked.load(Ordering::Relaxed) as usize
	}

	/// The `Lfu` admission latch (see the field): the handle's
	/// `admission_latched()`.
	pub fn lfu_latched(&self) -> bool {
		self.lfu_latched.load(Ordering::Relaxed)
	}

	/// Calls `f(key, tier, len)` for every live object: its key, the tier its
	/// value's bytes are in (the value's tag) and the value's length -- the
	/// placement audit's walk, `ObjectStore::for_each_value`'s twin. One
	/// difference: each shard is read under its lock into a buffer and `f`
	/// runs after the lock is released, since the merged handle's
	/// `placement_of` IS a lookup under that lock (`tier_of`).
	pub fn for_each_value(&self, mut f: impl FnMut(HashedKey, Tier, crate::object::ObjectSize)) {
		let mut shard = Vec::new();

		for lock in self.shards.iter() {
			{
				let g = lock.read().unwrap();

				for i in 0..g.slots.allocated {
					if let Some(object) = &g.slots[i].object {
						shard.push((g.slots[i].hashed, object.value().tier(), object.data_size()));
					}
				}
			}

			for (key, tier, len) in shard.drain(..) {
				f(key, tier, len);
			}
		}
	}

	pub fn is_empty(&self) -> bool {
		self.len() == 0
	}

	pub fn resize_fast_tier(&self, size: CacheSize, log: &mut MigrationLog) {
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
		self.settle_tier(log);
	}

	/// DRAM reserved out of the fast tier for shared per-object metadata across
	/// both tiers, so demotion bounds total DRAM and not just fast-tier values.
	pub fn dram_reserved_bytes(&self) -> CacheSize {
		self.reservation(self.shared_overhead.load(Ordering::Relaxed))
	}

	/// S5: the settle against the current budget -- the handle's `resettle`,
	/// the policy worker's end-of-pass step.
	pub fn resettle(&self, log: &mut MigrationLog) {
		self.settle_tier(log);
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

	/// Every linked slot is in exactly one tier, so the slow count is the
	/// linked total less the fast one -- two relaxed loads, and no third
	/// counter that could drift on its own.
	pub fn slow_object_count(&self) -> usize {
		self.linked().saturating_sub(self.fast_object_count())
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

		assert_eq!(
			self.linked() as CacheSize,
			self.sum_shards(|g| g.linked as CacheSize),
			"the store-level linked count drifted from the shards it mirrors",
		);
	}

	/// Invariant I of `Inner`, and the counts the policy worker keeps, walked
	/// out of every shard's slots: the tier totals plus the unfolded client
	/// bytes are the linked slots' `migrating()` per tier, `linked` is the
	/// slots on lists or buckets, `fast_count` the fast ones among them, and
	/// every slot on a list is on a chain. With `quiescent` (every event
	/// handled), also: no DEAD slot, nothing unfolded, `linked == len()`, and
	/// no unlinked slot. Plus `verify_gauges`.
	#[cfg(test)]
	pub(crate) fn verify_charges(&self, quiescent: bool) {
		let lfu = self.order() == MergedOrder::Lfu;

		for (s, lock) in self.shards.iter().enumerate() {
			let g = lock.read().unwrap();

			let mut listed = Vec::new();

			match lfu {
				true => {
					for tier in [Tier::Fast, Tier::Slow] {
						for (_, &(head, _)) in g.freq_buckets(tier).iter() {
							let mut i = head;

							while i != NIL {
								listed.push(i);
								i = g.slots[i as usize].next;
							}
						}
					}
				},

				false => {
					let mut i = g.head;

					while i != NIL {
						listed.push(i);
						i = g.slots[i as usize].next;
					}
				},
			}

			let (mut fast, mut slow, mut fast_count, mut dead) = (0i128, 0i128, 0usize, 0usize);

			for &i in &listed {
				let slot = &g.slots[i as usize];

				assert!(slot.is_linked(), "shard {s}: an unlinked slot is on a list");

				match slot.tier {
					Tier::Fast => {
						fast += slot.migrating() as i128;
						fast_count += 1;
					},

					Tier::Slow => slow += slot.migrating() as i128,
				}

				dead += slot.object.is_none() as usize;

				assert_eq!(g.find(slot.hashed).filter(|_| slot.object.is_some()), slot.object.is_some().then_some(i),
					"shard {s}: a live listed slot is not what find returns for its key");
			}

			assert_eq!(fast, g.fast_used as i128 + g.unfolded_fast as i128, "shard {s}: fast_used + unfolded_fast != the fast slots' bytes");
			assert_eq!(slow, g.slow_used as i128 + g.unfolded_slow as i128, "shard {s}: slow_used + unfolded_slow != the slow slots' bytes");
			assert_eq!(listed.len(), g.linked, "shard {s}: linked != the slots on lists");
			assert_eq!(fast_count, g.fast_count, "shard {s}: fast_count != the fast slots on lists");

			if quiescent {
				assert_eq!(dead, 0, "shard {s}: {dead} DEAD slots at quiescence");
				assert_eq!((g.unfolded_fast, g.unfolded_slow), (0, 0), "shard {s}: unfolded bytes at quiescence");

				let unlinked = (0..g.slots.allocated)
					.filter(|&i| g.slots[i].object.is_some() && !g.slots[i].is_linked())
					.count();

				assert_eq!(unlinked, 0, "shard {s}: {unlinked} unlinked values at quiescence");
			}
		}

		if quiescent {
			assert_eq!(self.linked(), self.len(), "linked != the object map's count at quiescence");
		}

		self.verify_gauges();
	}

	/// T15's snapshot of everything that is POLICY state in the store (see
	/// "Who does what" in the module doc): per shard the list ends, the
	/// boundary, the tier totals, `fast_count`, `linked` and both bucket maps,
	/// and every listed slot's `(hashed, prev, next, last_access, tier,
	/// referenced, freq)`; store-wide the clock, the four totals, both mirror
	/// arrays and the latch. NOT the object map's: chains, slab, free list,
	/// `tracked`, nor the clients' unfolded bytes.
	#[cfg(test)]
	pub(crate) fn policy_snapshot(&self) -> Vec<String> {
		let mut out = Vec::new();

		out.push(format!(
			"store clock={} fast_used={} slow_used={} fast_count={} linked={} latched={}",
			self.clock.load(Ordering::Relaxed),
			self.fast_used.load(Ordering::Relaxed),
			self.slow_used.load(Ordering::Relaxed),
			self.fast_count.load(Ordering::Relaxed),
			self.linked(),
			self.lfu_latched(),
		));

		for (s, (t, f)) in self.tails.iter().zip(self.fast_tails.iter()).enumerate() {
			out.push(format!(
				"mirror {s} tail=({},{}) fast=({},{})",
				t.seq.load(Ordering::Relaxed),
				t.freq.load(Ordering::Relaxed),
				f.seq.load(Ordering::Relaxed),
				f.freq.load(Ordering::Relaxed),
			));
		}

		for (s, lock) in self.shards.iter().enumerate() {
			let g = lock.read().unwrap();

			out.push(format!(
				"shard {s} head={} tail={} boundary={} fast_used={} slow_used={} fast_count={} linked={} fast_buckets={:?} slow_buckets={:?}",
				g.head, g.tail, g.fast_boundary, g.fast_used, g.slow_used, g.fast_count, g.linked,
				g.fast_buckets, g.slow_buckets,
			));

			let mut listed = Vec::new();
			let mut i = g.head;

			while i != NIL {
				listed.push(i);
				i = g.slots[i as usize].next;
			}

			for tier in [Tier::Fast, Tier::Slow] {
				for (_, &(head, _)) in g.freq_buckets(tier).iter() {
					let mut i = head;

					while i != NIL {
						listed.push(i);
						i = g.slots[i as usize].next;
					}
				}
			}

			for i in listed {
				let slot = &g.slots[i as usize];

				out.push(format!(
					"  slot {i} hashed={:#x} prev={} next={} last_access={} tier={:?} referenced={} freq={}",
					slot.hashed, slot.prev, slot.next, slot.last_access, slot.tier,
					slot.referenced.load(Ordering::Relaxed), slot.freq,
				));
			}
		}

		out
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

	/// The tier `key` is placed in: its LINKED slot's tier, `None` for a key
	/// with no live slot or an UNLINKED one -- a value the policy worker has
	/// not placed yet, as the DashMap stacks' `tier_of` is `None` for a key
	/// whose `Set` they have not handled. The merged handle's
	/// `PolicyStack::placement_of`, and what tests and fidelity checks read.
	/// One shard read lock.
	pub fn tier_of(&self, key: HashedKey) -> Option<Tier> {
		let g = self.shards[shard_of(key)].read().unwrap();
		let slot = &g.slots[g.find(key)? as usize];

		slot.is_linked().then_some(slot.tier)
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

	/// A store driven the way the policy worker drives it: its calls go
	/// through here with ONE migration log, which `drain_migrations` hands
	/// back -- the handle's part, for the tests that use the store directly.
	/// Everything else is the store's, through `Deref`.
	struct Driven {
		store: Store,
		log: std::cell::RefCell<MigrationLog>,
	}

	impl Deref for Driven {
		type Target = Store;

		fn deref(&self) -> &Store {
			&self.store
		}
	}

	impl Driven {
		fn new(store: Store) -> Self {
			Driven { store, log: Default::default() }
		}

		fn touch(&self, key: HashedKey) {
			self.store.touch(key, &mut self.log.borrow_mut());
		}

		fn tail_key(&self) -> Option<HashedKey> {
			self.store.tail_key(&mut self.log.borrow_mut())
		}

		fn resize_fast_tier(&self, size: CacheSize) {
			self.store.resize_fast_tier(size, &mut self.log.borrow_mut());
		}

		fn drain_migrations(&self) -> Vec<(HashedKey, Tier)> {
			self.log.borrow_mut().take_untagged()
		}

		/// The policy worker's removal of a nominated victim.
		fn take(&self, key: &HashedKey) -> Option<Object<u64, crate::BufferDRAM>> {
			self.store.take_evict(key)
		}
	}

	fn tiered(fast_capacity: CacheSize) -> Driven {
		let s = Store::new();
		s.configure_tiering(fast_capacity, 0, DEFAULT_HIGH_PPM, DEFAULT_LOW_PPM);
		Driven::new(s)
	}

	/// A whole set: the client's `insert`, then the policy worker's
	/// `worker_set` for its `Set`, told what the insert did.
	///
	/// The value is `size` bytes of real allocation, because the tier
	/// accounting is DERIVED from the object rather than reported separately
	/// -- an object built with an empty value migrates zero bytes. The event's
	/// size, which only the `Lfu` admission gate reads, is the object's
	/// `migrating()`: what these tests' gate compared before it took the
	/// event's base size.
	fn put_with(s: &Store, log: &mut MigrationLog, key: HashedKey, size: ObjectSize) {
		let old = s.insert(key, Object::new(key, &vec![0u8; size as usize], None));

		let event = match old {
			None => SetEvent::Fresh,
			Some(old) => SetEvent::Replaced { resized: old.data_size() != size },
		};

		s.worker_set(key, migrating_bytes(size) as ObjectSize, event, log);
	}

	fn put(s: &Driven, key: HashedKey, size: ObjectSize) {
		put_with(&s.store, &mut s.log.borrow_mut(), key, size);
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

	/// S5a: the store's own count of its structures is what the allocator
	/// holds for them, from construction (the fixed arrays and every shard's
	/// first buckets) on: inserts across the shards (a slab chunk per shard,
	/// bucket splits), one shard driven to six chunks (its chunk table grows
	/// twice), LFU links and bumps (the frequency maps' nodes), evictions into
	/// the free lists, a wipe that frees every chunk, and a refill. Checked
	/// after every operation. The objects are built first and what the store
	/// hands back is kept, so only the store's own allocations are measured;
	/// the migration log is drained (and its buffer freed) before each check.
	#[test]
	fn the_store_counts_exactly_what_its_structures_allocated() {
		const SPREAD: u64 = 3_000;
		const ONE_SHARD: u64 = 6 * SLAB_CHUNK as u64 - 100;

		let len = 48u32;
		let mut objects: Vec<Object<u64, crate::BufferDRAM>> = (0..SPREAD + ONE_SHARD)
			.map(|i| Object::new(i, &[7u8; 48], None))
			.collect();
		let mut kept: Vec<Object<u64, crate::BufferDRAM>> = Vec::with_capacity(objects.len());
		let spread = |i: u64| mix(i + 1);
		let packed = |i: u64| (3u64 << (64 - SHARD_BITS)) | (i + 1);

		let base = crate::meta::thread_live_bytes();
		let live = || (crate::meta::thread_live_bytes() - base) as u64;

		let store = Store::new();
		store.set_order(MergedOrder::Lfu);
		store.configure_tiering(1 << 20, 0, DEFAULT_HIGH_PPM, DEFAULT_LOW_PPM);
		let mut log = MigrationLog::default();

		assert_eq!(live(), store.structure_bytes(), "a new store: its fixed arrays and first buckets");

		let set = |store: &Store, log: &mut MigrationLog, key: HashedKey, object: Object<u64, crate::BufferDRAM>| {
			assert!(store.insert(key, object).is_none());
			store.worker_set(key, migrating_bytes(len) as ObjectSize, SetEvent::Fresh, log);
			drop(log.take_entries());
		};

		for i in 0..SPREAD {
			set(&store, &mut log, spread(i), objects.pop().unwrap());
			assert_eq!(live(), store.structure_bytes(), "after spread insert {i}");
		}

		for i in 0..ONE_SHARD {
			set(&store, &mut log, packed(i), objects.pop().unwrap());
			assert_eq!(live(), store.structure_bytes(), "after packed insert {i}");
		}

		assert_eq!(store.shards_chunks(3).0, 6, "the packed shard holds six chunks");

		// Distinct frequencies: key i touched i times.
		for i in 0..300 {
			for _ in 0..i {
				store.touch(spread(i), &mut log);
			}

			drop(log.take_entries());
			assert_eq!(live(), store.structure_bytes(), "after spread key {i}'s {i} touches");
		}

		for i in (0..SPREAD).step_by(2) {
			kept.push(store.take_evict(&spread(i)).expect("a live key"));
			assert_eq!(live(), store.structure_bytes(), "after evicting spread key {i}");
		}

		for i in (0..ONE_SHARD).step_by(3) {
			kept.push(store.take_evict(&packed(i)).expect("a live key"));
			assert_eq!(live(), store.structure_bytes(), "after evicting packed key {i}");
		}

		// Everything out, so the wipe frees structures and nothing else.
		for i in (0..SPREAD).filter(|i| i % 2 != 0) {
			kept.push(store.take_evict(&spread(i)).expect("a live key"));
		}

		for i in (0..ONE_SHARD).filter(|i| i % 3 != 0) {
			kept.push(store.take_evict(&packed(i)).expect("a live key"));
		}

		assert_eq!(live(), store.structure_bytes(), "after emptying");

		let held = store.structure_bytes();
		store.clear_counted(|_| 0);
		assert_eq!(live(), store.structure_bytes(), "after the wipe");
		assert!(store.structure_bytes() < held, "the wipe freed the chunks and the maps' nodes");

		for (i, object) in kept.drain(..).take(2_000).enumerate() {
			set(&store, &mut log, spread(i as u64), object);
			assert_eq!(live(), store.structure_bytes(), "after refill insert {i}");
		}

		drop(store);
		drop(log);
		drop(kept);
		drop(objects);
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

		let mut log = MigrationLog::default();

		// Two keys in ONE shard, so the queue order is unambiguous: `a` is
		// older, `b` is newer.
		let shard = shard_of(mix(1));
		let mut keys = (1u64..).map(mix).filter(|&k| shard_of(k) == shard);
		let a = keys.next().unwrap();
		let b = keys.next().unwrap();

		put_with(&s, &mut log, a, 128);
		put_with(&s, &mut log, b, 128);

		assert_eq!(s.tail_key(&mut log), Some(a), "the older key is not the victim");

		// The guard a concurrent GET of `a` would be holding: `get_ref` takes
		// `shards[shard_of(key)].read()`, which is this exact lock.
		let reader = s.get_ref(&a).expect("the key was just inserted");

		let (tx, rx) = mpsc::channel();

		let hit = {
			let s = Arc::clone(&s);

			std::thread::spawn(move || {
				s.touch(a, &mut MigrationLog::default());
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
		assert_eq!(s.tail_key(&mut log), Some(b), "the hand did not spare the referenced key");
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
	///   * a NEW key's link                        (`worker_set` -> `link`)
	///   * overwrite with a DIFFERENT size, while the slot is FAST
	///   * overwrite with a different size while the slot is SLOW -- the one
	///     path that moves `slow_used` and neither of the other two
	///   * promotion                              (`touch_slot`)
	///   * demotion                               (`demote_boundary`)
	///   * removal                                (`remove_key` -> `take_evict`)
	///   * removal returning the object           (`take` -> `detach_tier`)
	///   * a fast-tier resize, which demotes in bulk
	///   * `clear_counted`, which zeroes every shard and every mirror at once
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

		// And `clear_counted` zeroes the mirrors along with the shards.
		s.clear_counted(|_| 0);
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

	/// A settle's demotions drain in the order the settle DECIDED them, across
	/// shards -- the order a DashMap stack's drain has. The per-shard lists
	/// this store used to keep concatenated in shard order, so a settle that
	/// demoted a key in a high shard and then one in a low shard drained them
	/// the other way round.
	#[test]
	fn a_settle_drains_its_demotions_in_decision_order_across_shards() {
		let s = tiered(CacheSize::MAX);

		// Oldest first: shard 9, then shard 3, then two more to keep fast.
		let keys = [
			(mix(1) >> SHARD_BITS) | (9 << (64 - SHARD_BITS)),
			(mix(2) >> SHARD_BITS) | (3 << (64 - SHARD_BITS)),
			(mix(3) >> SHARD_BITS) | (5 << (64 - SHARD_BITS)),
			(mix(4) >> SHARD_BITS) | (1 << (64 - SHARD_BITS)),
		];

		for &k in &keys {
			put(&s, k, 256);
		}

		assert!(s.drain_migrations().is_empty(), "an unbounded tier demotes nothing");

		// The settle drains to 0.95 of the budget: three objects' worth of
		// budget keeps two of the four fast, so it demotes the two oldest,
		// shard 9's first.
		s.resize_fast_tier(migrating_bytes(256) * 3);

		assert_eq!(
			s.drain_migrations(),
			vec![(keys[0], Tier::Slow), (keys[1], Tier::Slow)],
			"the demotions did not drain in the order the settle decided them",
		);
	}

	/// A settle step folds its shard before it demotes, and the fold can bring
	/// the tier under its target by itself: a client's shrinking overwrite of a
	/// fast value, pending in the shard, was in `fast_used` until then. The
	/// step then stops -- a demotion past that point takes a key the target
	/// does not ask for. Here `k`, the oldest fast key, is shrunk by a client
	/// from 8 KiB to 256 B, and a resize puts the tier over its watermark on
	/// the stale total; the step chooses `k`'s shard, folds it, is under the
	/// target, and demotes nothing. Red without the re-check (`norecheck`:
	/// `k` goes slow).
	#[test]
	fn a_settle_step_whose_fold_brings_the_tier_under_its_target_demotes_nothing() {
		let s = tiered(CacheSize::MAX);
		let k = mix(1);
		let j = mix(2);

		put(&s, k, 8_192);
		put(&s, j, 256);
		assert!(s.drain_migrations().is_empty(), "an unbounded tier demotes nothing");
		assert_ne!(shard_of(k), shard_of(j), "two shards");

		// The client's overwrite: pending in `k`'s shard, not in `fast_used`.
		s.insert(k, Object::new(k, &[0u8; 256], None));
		assert_eq!(s.fast_bytes_used(), migrating_bytes(8_192) + migrating_bytes(256), "stale until folded");

		// Over the watermark on the stale total, under it once folded.
		let capacity = 4 * 1_024;
		assert!(s.fast_bytes_used() > scale(capacity, DEFAULT_HIGH_PPM));
		assert!(2 * migrating_bytes(256) <= scale(capacity, DEFAULT_LOW_PPM));

		s.resize_fast_tier(capacity);

		assert_eq!(s.drain_migrations(), vec![], "the fold brought the tier under: nothing to demote");
		assert_eq!((s.tier_of(k), s.tier_of(j)), (Some(Tier::Fast), Some(Tier::Fast)));
		assert_eq!(s.fast_bytes_used(), 2 * migrating_bytes(256), "folded");

		// The overwrite's `Set`.
		s.store.worker_set(k, migrating_bytes(256) as ObjectSize, SetEvent::Replaced { resized: true }, &mut s.log.borrow_mut());
		assert_eq!(s.drain_migrations(), vec![]);
		s.verify_charges(true);
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
		// worth asking the gauges here. Whether the drain stranded anything is
		// no longer a question: the migrations are one log the policy worker
		// takes whole (`MigrationLog`, a `mem::take`), not per-shard lists
		// behind a dirty mask.
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
		let s = Driven::new(Store::new().with_update_interval(1_000));
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
		let exact = Driven::new(Store::new());
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

		// `insert` alone, not `put`: the worker's `worker_set` looks the key up,
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
					// Each thread both publishes and links -- a client and a
					// worker in one -- with a log of its own.
					let mut log = MigrationLog::default();

					for i in 1..=2_000u64 {
						let key = mix(t * 1_000_000 + i);

						put_with(&s, &mut log, key, 128);
						s.touch(mix(t * 1_000_000 + (i / 2).max(1)), &mut log);

						if i % 13 == 0 {
							s.remove_key(mix(t * 1_000_000 + i / 13));
						}

						if i % 101 == 0 {
							s.resize_fast_tier(32 * 1_024 * (1 + i % 3), &mut log);
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

	// ── T15: a client changes no policy state ─────────────────────────────

	/// A store of `order` filled past its budget through the policy worker's
	/// path -- 64 keys of 256 B against room for 16 -- so it holds fast and
	/// slow keys, and under `Lfu` has latched.
	fn t15_store(order: MergedOrder) -> Driven {
		let s = Store::new();
		s.set_order(order);
		s.configure_tiering(16 * migrating_bytes(256), 0, DEFAULT_HIGH_PPM, DEFAULT_LOW_PPM);

		let s = Driven::new(s);

		for i in 1..=64u64 {
			put(&s, mix(i), 256);
		}

		s.drain_migrations();

		assert!(s.fast_object_count() > 0 && s.slow_object_count() > 0, "{order:?}: fast and slow");

		if order == MergedOrder::Lfu {
			assert!(s.lfu_latched(), "Lfu latched");
		}

		s
	}

	/// Fails with the first line of the policy snapshot that changed.
	fn assert_same_policy_state(before: &[String], after: &[String], what: &str) {
		for (b, a) in before.iter().zip(after) {
			assert_eq!(b, a, "{what}: the policy state changed");
		}

		assert_eq!(before.len(), after.len(), "{what}: the policy state changed (slots listed)");
	}

	/// A key's shard's unfolded client bytes, `(fast, slow)`.
	fn unfolded(s: &Store, key: HashedKey) -> (i64, i64) {
		let g = s.shards[shard_of(key)].read().unwrap();
		(g.unfolded_fast, g.unfolded_slow)
	}

	const T15_ORDERS: [MergedOrder; 4] = [MergedOrder::Lru, MergedOrder::Fifo, MergedOrder::Clock, MergedOrder::Lfu];

	/// T15: with the policy worker not running, a CLIENT's insert of a new key
	/// changes no policy state -- no stamp, link, charge, boundary, mirror,
	/// settle or latch (`policy_snapshot` identical) -- in every order, at the
	/// tier's target and over it. The key is published: readable, counted in
	/// the map's `len`, not placed, not linked. Before S4 the client's insert
	/// linked, stamped, charged and settled. Red with the client linking
	/// (`clientlink`) and settling (`clientsettle`).
	#[test]
	fn t15_a_client_set_of_a_new_key_changes_no_policy_state() {
		for order in T15_ORDERS {
			let s = t15_store(order);
			let k = mix(1_000);

			let before = s.policy_snapshot();
			let (len, linked) = (s.len(), s.linked());

			s.insert(k, Object::new(k, &[0u8; 256], None));

			assert_same_policy_state(&before, &s.policy_snapshot(), &format!("{order:?} new key"));
			assert!(s.get_ref(&k).is_some(), "{order:?}: the value is published");
			assert_eq!(s.tier_of(k), None, "{order:?}: placed before its Set");
			assert_eq!((s.len(), s.linked()), (len + 1, linked), "{order:?}: map and link counts");

			// And with the tier over its watermark -- the budget cut under it
			// and no settle run yet, as a worker's link leaves it until its
			// settle -- the insert still settles nothing: a store at its
			// target cannot show a settle, so this is the half a client-side
			// settle fails (`clientsettle`).
			s.configure_tiering(s.fast_bytes_used() / 2, 0, DEFAULT_HIGH_PPM, DEFAULT_LOW_PPM);

			let before = s.policy_snapshot();
			let k2 = mix(1_001);

			s.insert(k2, Object::new(k2, &[0u8; 256], None));

			assert_same_policy_state(&before, &s.policy_snapshot(), &format!("{order:?} new key, the tier over its watermark"));
		}
	}

	/// T15: a CLIENT's overwrite of a linked key -- fast or slow, the same
	/// size or resized -- changes no policy state: no relink, restamp,
	/// reference bit, bump, promotion or settle, and no tier total. Its bytes
	/// are recorded as unfolded bytes of the slot's tier, exactly the change,
	/// for the worker to fold. Before S4 the client charged them, relinked
	/// (LRU), set the bit (CLOCK), bumped (LFU) and settled.
	#[test]
	fn t15_a_client_overwrite_changes_no_policy_state() {
		for order in T15_ORDERS {
			let s = t15_store(order);

			let fast = (1..=64u64).map(mix).find(|&k| s.tier_of(k) == Some(Tier::Fast)).expect("a fast key");
			let slow = (1..=64u64).map(mix).find(|&k| s.tier_of(k) == Some(Tier::Slow)).expect("a slow key");

			for (key, len) in [(fast, 256), (fast, 1_024), (slow, 256), (slow, 1_024)] {
				let tier = s.tier_of(key).expect("linked");
				let was = s.get_ref(&key).map(|o| migrating_bytes(o.data_size())).expect("live");

				let before = s.policy_snapshot();
				let (fast_before, slow_before) = unfolded(&s, key);

				s.insert(key, Object::new(key, &vec![0u8; len as usize], None));

				let what = format!("{order:?} {tier:?} overwrite to {len}");
				assert_same_policy_state(&before, &s.policy_snapshot(), &what);

				let delta = migrating_bytes(len) as i64 - was as i64;
				let (fast_after, slow_after) = unfolded(&s, key);

				match tier {
					Tier::Fast => assert_eq!((fast_after - fast_before, slow_after - slow_before), (delta, 0), "{what}"),
					Tier::Slow => assert_eq!((fast_after - fast_before, slow_after - slow_before), (0, delta), "{what}"),
				}

				// The worker's `Set`, so the next case starts folded.
				s.store.worker_set(
					key,
					migrating_bytes(len) as ObjectSize,
					SetEvent::Replaced { resized: was != migrating_bytes(len) },
					&mut s.log.borrow_mut(),
				);
				s.drain_migrations();
				s.verify_charges(true);
			}
		}
	}

	/// T15: a CLIENT's delete -- `del`'s key-matched `take_if` -- and a TTL
	/// reap -- the reaper's expired-only `take_if` -- of a linked key change no
	/// policy state. The value is gone to every reader, its slot is DEAD on
	/// its list (still counted in `linked`, for the worker's `Del` or `Expire`
	/// to retire), and its bytes are recorded unfolded, negative. Before S4 the
	/// client unlinked it, uncharged it and republished the mirrors.
	#[test]
	fn t15_a_client_delete_or_ttl_reap_changes_no_policy_state() {
		for reap in [false, true] {
			for order in T15_ORDERS {
				let s = t15_store(order);

				let k = mix(1_000);

				let object = match reap {
					false => Object::new(k, &[0u8; 256], None),
					true => Object::with_expiry(k, &[0u8; 256], std::num::NonZeroU32::new(1)),
				};

				s.insert(k, object);
				s.store.worker_set(k, migrating_bytes(256) as ObjectSize, SetEvent::Fresh, &mut s.log.borrow_mut());
				s.drain_migrations();

				let tier = s.tier_of(k).expect("linked");
				let before = s.policy_snapshot();
				let (len, linked) = (s.len(), s.linked());

				let taken = match reap {
					false => s.take_if(&k, |_| true),
					true => s.take_if(&k, |object| object.is_expired()),
				};

				let what = format!("{order:?} {tier:?} {}", if reap { "reap" } else { "delete" });

				assert!(taken.is_some(), "{what}: nothing taken");
				assert_same_policy_state(&before, &s.policy_snapshot(), &what);
				assert!(s.get_ref(&k).is_none(), "{what}: the value is still readable");
				assert_eq!((s.len(), s.linked()), (len - 1, linked), "{what}: map and link counts");

				{
					let g = s.shards[shard_of(k)].read().unwrap();
					assert!(g.find_dead(k).is_some(), "{what}: no DEAD slot on the list");
				}

				let (fast, slow) = unfolded(&s, k);
				let m = migrating_bytes(256) as i64;

				match tier {
					Tier::Fast => assert_eq!((fast, slow), (-m, 0), "{what}"),
					Tier::Slow => assert_eq!((fast, slow), (0, -m), "{what}"),
				}

				s.verify_charges(false);

				// The worker's `Del` / `Expire` retires it.
				assert!(s.retire_dead(k), "{what}: nothing to retire");
				s.verify_charges(true);
			}
		}
	}

	/// `clear_counted` returns what it removed: the LIVE objects -- linked and
	/// unlinked -- and their base bytes as its caller's `base_size` gives them,
	/// not a DEAD slot's, whose object the client's delete already took (and
	/// took off the status then). The worker's wipe subtracts exactly that
	/// from the status (`AtomicStatus::clear`).
	#[test]
	fn clear_counted_returns_the_live_objects_and_their_base_bytes() {
		let s = tiered(CacheSize::MAX);

		for i in 1..=10u64 {
			put(&s, mix(i), 256 * i as ObjectSize);
		}

		// Published, not linked; and a linked value deleted by a client.
		s.insert(mix(11), Object::new(mix(11), &[0u8; 100], None));
		assert!(s.take_if(&mix(3), |_| true).is_some());
		assert_eq!((s.len(), s.linked()), (10, 10), "ten live, ten on lists (one DEAD)");

		let cleared = s.clear_counted(|object| object.data_size());

		let bytes = (1..=10u64).filter(|&i| i != 3).map(|i| 256 * i).sum::<u64>() + 100;
		assert_eq!(cleared, Cleared { objects: 10, base_bytes: bytes as CacheSize });
		assert_eq!((s.len(), s.linked(), s.fast_bytes_used()), (0, 0, 0));
		s.verify_charges(true);
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
	fn lfu(fast_capacity: CacheSize) -> Driven {
		let s = Store::new();

		s.set_order(MergedOrder::Lfu);
		s.configure_tiering(fast_capacity, 0, DEFAULT_HIGH_PPM, DEFAULT_LOW_PPM);

		Driven::new(s)
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

		s.clear_counted(|_| 0);
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
					// Each thread both publishes and links -- a client and a
					// worker in one -- with a log of its own.
					let mut log = MigrationLog::default();

					for i in 1..=1_000u64 {
						let key = mix(t * 1_000_000 + i);

						put_with(&s, &mut log, key, 128);
						s.touch(mix(t * 1_000_000 + (i / 2).max(1)), &mut log);

						if i % 13 == 0 {
							s.remove_key(mix(t * 1_000_000 + i / 13));
						}

						if i % 101 == 0 {
							s.resize_fast_tier(32 * 1_024 * (1 + i % 3), &mut log);
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
		let mut log = MigrationLog::default();

		for i in 0..n {
			let k = i.wrapping_mul(0x9E37_79B9_7F4A_7C15);
			store.insert(k, Object::new(k, &vec![0u8; vsize], None));
			store.worker_set(k, (vsize + 16) as ObjectSize, SetEvent::Fresh, &mut log);
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
