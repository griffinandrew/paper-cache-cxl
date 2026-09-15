/*
 * Copyright (c) Kia Shakiba
 *
 * This source code is licensed under the GNU AGPLv3 license found in the
 * LICENSE file in the root directory of this source tree.
 */

mod lfu_compact_stack;
mod lfu_stack;
mod fifo_stack;
mod clock_stack;
mod sieve_stack;
mod lru_compact_stack;
mod lru_stack;
mod mru_stack;
mod two_q_stack;
mod arc_stack;
mod s_three_fifo_stack;
pub(crate) mod ghost_filter;

pub(crate) mod compact_queue_set;
pub mod arena_index;
pub mod arena_queue_set;
pub(crate) mod arena_frequency_chain;

/// The frequency chain `ArenaFrequencyChain` replaces, kept ONLY as the
/// reference implementation that pins it.
///
/// It has no production caller left: both LFU-ranked stacks moved to the arena,
/// which stores the key once instead of twice and measures 40 B/object against
/// this one's 72. It stays in the tree, test-gated, because
/// `arena_frequency_chain`'s two differential tests --
/// `the_frequency_face_agrees_with_the_compact_chain_it_replaces` and
/// `the_recency_face_agrees_with_the_compact_chain_it_replaces` -- drive the
/// two through the same random operation streams and require identical
/// observable state at every step. That is the evidence that the conversion
/// was a representation change and not an algorithm change, and it is worth
/// more in the tree than the module is worth out of it.
#[cfg(test)]
pub(crate) mod compact_frequency_chain;
#[cfg(test)]
mod measure_overhead;
mod lru_lfu_compact_hybrid_stack;
mod lru_compact_hybrid_stack;
mod lru_lazy_copy_compact_hybrid_stack;
mod lfu_compact_hybrid_stack;
mod two_q_compact_hybrid_stack;
mod two_q_fast_admission_compact_hybrid_stack;
mod two_q_fast_admission_reprieve_compact_hybrid_stack;
mod two_q_full_fast_admission_compact_hybrid_stack;
mod fifo_compact_hybrid_stack;
mod clock_compact_hybrid_stack;
mod lru_sized_compact_hybrid_stack;
mod s3_fifo_compact_hybrid_stack;
mod s3_fifo_faithful_compact_hybrid_stack;
mod two_q_ghost_compact_hybrid_stack;
mod s3_fifo_ghost_compact_hybrid_stack;
mod s3_fifo_ghost_lazy_demotion_compact_hybrid_stack;
mod s3_fifo_ghost_lazy_demotion_fast_admission_compact_hybrid_stack;
mod s3_fifo_ghost_lazy_demotion_fast_admission_midpoint_compact_hybrid_stack;
mod s3_fifo_lazy_demotion_fast_admission_midpoint_reprieve_compact_hybrid_stack;
mod s3_fifo_lazy_demotion_fast_admission_reprieve_compact_hybrid_stack;
mod s3_fifo_lazy_demotion_reprieve_compact_hybrid_stack;
mod two_q_compact_stack;
#[cfg(all(test, feature = "hybrid_cache_common"))]
mod merged_prototype;

/// `PolicyStack` over the merged object store -- the store IS the
/// eviction stack, so this forwards rather than owning anything.
#[cfg(feature = "merged_object_store")]
pub(crate) mod merged_stack;
mod s_three_fifo_compact_stack;
mod fifo_compact_stack;
mod clock_compact_stack;
mod sieve_compact_stack;
mod mru_compact_stack;
mod s3_fifo_lazy_demotion_fast_admission_split_slow_reprieve_compact_hybrid_stack;

#[cfg(feature = "eviction_stacks_pmem")] mod pmem_collections;

use crate::{
	CacheSize,
	HashedKey,
	policy::PaperPolicy,
	object::ObjectSize,
	worker::policy::policy_stack::{
		lfu_compact_stack::LfuCompactStack,
		fifo_compact_stack::FifoCompactStack,
		clock_compact_stack::ClockCompactStack,
		sieve_compact_stack::SieveCompactStack,
		mru_compact_stack::MruCompactStack,
		two_q_compact_stack::TwoQCompactStack,
		s_three_fifo_compact_stack::SThreeFifoCompactStack,
		s3_fifo_faithful_compact_hybrid_stack::S3FifoFaithfulCompactHybridStack,
		s3_fifo_faithful_compact_hybrid_stack::S3FifoFaithfulFastAdmissionCompactHybridStack,
		s3_fifo_faithful_compact_hybrid_stack::S3FifoFaithfulReprieveCompactHybridStack,
		s3_fifo_faithful_compact_hybrid_stack::S3FifoFaithfulFastAdmissionReprieveCompactHybridStack,
		lfu_stack::LfuStack,
		fifo_stack::FifoStack,
		clock_stack::ClockStack,
		sieve_stack::SieveStack,
		lru_compact_stack::LruCompactStack,
		lru_stack::LruStack,
		mru_stack::MruStack,
		two_q_stack::TwoQStack,
		arc_stack::ArcStack,
		s_three_fifo_stack::SThreeFifoStack,
		lru_lfu_compact_hybrid_stack::LruLfuCompactHybridStack,
		lru_compact_hybrid_stack::LruCompactHybridStack,
		lru_lazy_copy_compact_hybrid_stack::LruLazyCopyCompactHybridStack,
		lfu_compact_hybrid_stack::LfuCompactHybridStack,
		two_q_compact_hybrid_stack::TwoQCompactHybridStack,
		two_q_fast_admission_compact_hybrid_stack::TwoQFastAdmissionCompactHybridStack,
		two_q_fast_admission_reprieve_compact_hybrid_stack::TwoQFastAdmissionReprieveCompactHybridStack,
		two_q_full_fast_admission_compact_hybrid_stack::TwoQFullFastAdmissionCompactHybridStack,
		fifo_compact_hybrid_stack::FifoCompactHybridStack,
		clock_compact_hybrid_stack::ClockCompactHybridStack,
		lru_sized_compact_hybrid_stack::LruSizedCompactHybridStack,
		s3_fifo_compact_hybrid_stack::S3FifoCompactHybridStack,
		two_q_ghost_compact_hybrid_stack::TwoQGhostCompactHybridStack,
		s3_fifo_ghost_compact_hybrid_stack::S3FifoGhostCompactHybridStack,
		s3_fifo_ghost_lazy_demotion_compact_hybrid_stack::S3FifoGhostLazyDemotionCompactHybridStack,
		s3_fifo_ghost_lazy_demotion_fast_admission_compact_hybrid_stack::S3FifoGhostLazyDemotionFastAdmissionCompactHybridStack,
		s3_fifo_ghost_lazy_demotion_fast_admission_midpoint_compact_hybrid_stack::S3FifoGhostLazyDemotionFastAdmissionMidpointCompactHybridStack,
		s3_fifo_lazy_demotion_fast_admission_midpoint_reprieve_compact_hybrid_stack::S3FifoLazyDemotionFastAdmissionMidpointReprieveCompactHybridStack,
		s3_fifo_lazy_demotion_fast_admission_reprieve_compact_hybrid_stack::S3FifoLazyDemotionFastAdmissionReprieveCompactHybridStack,
		s3_fifo_lazy_demotion_reprieve_compact_hybrid_stack::S3FifoLazyDemotionReprieveCompactHybridStack,
		s3_fifo_lazy_demotion_fast_admission_split_slow_reprieve_compact_hybrid_stack::S3FifoLazyDemotionFastAdmissionSplitSlowReprieveCompactHybridStack,
	},
};

/// Outcome of a policy stack access that may carry extra routing signals.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AccessOutcome {
	None,
	GhostHit,
}

/// The level the fast tier is continuously held at, as a fraction of its
/// effective budget.
///
/// `settle_fast_tier` demotes whenever `fast_used` is above
/// `ratio * effective_capacity` and stops the moment it is back at it. ONE
/// threshold, not a band: there is no arm-here / drain-to-there gap, so a
/// settle moves only what the admission or promotion that triggered it
/// displaced, and the tier steady-states just under its budget rather than
/// sawtoothing between two marks.
///
/// The margin is burst headroom, and it is the whole reason the ratio is not
/// 1.0. `PaperCache::set()` writes a new object's bytes to DRAM synchronously
/// at the API layer, before the event reaches `PolicyWorker` at all; and a
/// demotion this function decides is not physically applied until a
/// `migration_queue` consumer runs it. Real DRAM therefore sits above what the
/// stack believes for as long as those two windows last. Held at exactly its
/// ceiling, a tier has nowhere for that overshoot to go but outside the
/// budget; held at 0.98, it lands inside.
///
/// 0.98 is where this tree's own burst margin sat before a 0.98/0.95 high/low
/// pair replaced it (`FAST_TIER_LOW_WATER_RATIO`; see `LRU_HYBRID_CACHE.md`).
/// It is the band that is gone, not the margin. The band ARMED at 0.98 and
/// only then drained to 0.95, which cost 5% of the allocation at the bottom of
/// every sawtooth and emitted the drop as one burst of demotions -- ~3% of the
/// budget in a single `apply_tier_migrations` batch. This keeps the headroom
/// and pays for it once, not per pass.
///
/// `FAST_TIER_DRAIN_TARGET=1.0` holds the tier at exactly its ceiling, which is
/// the no-headroom behaviour.
pub mod drain_target {
	use std::sync::OnceLock;

	pub const DEFAULT_RATIO: f64 = 0.98;

	static RATIO: OnceLock<f64> = OnceLock::new();

	/// Read once through a `OnceLock`, so the env var is startup configuration
	/// rather than runtime adjustable. A value that fails to parse or falls
	/// outside `(0.0, 1.0]` is silently replaced by the default.
	pub fn ratio() -> f64 {
		*RATIO.get_or_init(|| {
			std::env::var("FAST_TIER_DRAIN_TARGET")
				.ok()
				.and_then(|v| v.parse::<f64>().ok())
				.filter(|v| *v > 0.0 && *v <= 1.0)
				.unwrap_or(DEFAULT_RATIO)
		})
	}

	/// The byte level a settle holds the tier at: both the point above which it
	/// demotes and the point it stops at.
	pub fn bytes(effective_capacity: u64) -> u64 {
		(effective_capacity as f64 * ratio()) as u64
	}
}

/// Which tier an object currently lives in, for policy stacks that track a
/// segmented (fast/slow) queue. Used by `LruCompactHybridStack`
/// (`PaperPolicy::LruCompactHybrid`, recency-segmented),
/// `LfuCompactHybridStack` (`PaperPolicy::LfuCompactHybrid`,
/// frequency-segmented), `TwoQCompactHybridStack`
/// (`PaperPolicy::TwoQCompactHybrid`, 2Q-segmented), and
/// `FifoCompactHybridStack` (`PaperPolicy::FifoCompactHybrid`,
/// insertion-order-segmented); every other stack's default
/// `drain_tier_migrations` never produces one.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Tier {
	Fast,
	Slow,
}

/// Narrows a DRAM-resident remainder so it fits an entry's spare padding byte.
///
/// The remainder is `key + expiry field (16) + Expiries entry (64 with a TTL)`,
/// so `u8` covers every key up to 175 bytes -- and the benchmark's keys are
/// pre-hashed `u64`s. Saturating is safe rather than merely convenient: any
/// excess is then treated as migrating, which is exactly the behaviour before
/// this accounting existed, so it degrades toward the old over-charge instead
/// of going wrong in a new way.
#[inline]
pub(crate) fn narrow_resident(resident: ObjectSize) -> u8 {
	resident.min(u8::MAX as ObjectSize) as u8
}

pub trait PolicyStack
where
	Self: Send,
{
	fn is_policy(&self, policy: &PaperPolicy) -> bool;
	fn len(&self) -> usize;

	fn contains(&self, key: HashedKey) -> bool;
	fn insert(&mut self, key: HashedKey, size: ObjectSize);

	/// `insert`, plus the part of `size` that stays in DRAM whichever tier the
	/// object lands in (key, expiry field, and the `Expiries` entry when a TTL
	/// is set -- see `OverheadManager::dram_resident_size`).
	///
	/// Only the hybrid stacks care: they must keep `fast_used` / `slow_used` to
	/// bytes that actually migrate, since `Object::set_data` moves the value
	/// buffer alone. All-DRAM stacks have no tiers and ignore it.
	fn insert_resident(&mut self, key: HashedKey, size: ObjectSize, dram_resident: ObjectSize) {
		let _ = dram_resident;
		self.insert(key, size);
	}
	fn update(&mut self, _key: HashedKey) {}
	fn record_access(&mut self, key: HashedKey, hit: bool) -> AccessOutcome {
		if hit {
			self.update(key);
		}

		AccessOutcome::None
	}
	fn remove(&mut self, key: HashedKey);

	fn resize(&mut self, _size: CacheSize) {}
	fn clear(&mut self);

	fn evict_one(&mut self) -> Option<HashedKey>;

	/// Runtime-adjusts the fast-tier byte budget. No-op for every stack
	/// except the hybrid stacks, which shrinking may trigger immediate
	/// demotions for (see `drain_tier_migrations`).
	fn resize_fast_tier(&mut self, _size: CacheSize) {}

	/// Drains and returns every (key, new tier) pair that crossed the
	/// fast/slow boundary since the last call. Only the hybrid stacks
	/// (`LruCompactHybridStack`, `LfuCompactHybridStack`,
	/// `TwoQCompactHybridStack`, ...) ever produce entries; every other stack
	/// keeps the default empty `Vec`. The caller
	/// (`PolicyWorker`) is responsible for physically migrating each
	/// returned key's object bytes to `new_tier`.
	fn drain_tier_migrations(&mut self) -> Vec<(HashedKey, Tier)> {
		Vec::new()
	}

	/// Current bytes accounted to the fast tier. `0` for every stack except
	/// the hybrid stacks.
	/// DRAM reserved for shared per-object metadata across *both* tiers
	/// (`tracked objects x shared_overhead`). `fast_bytes_used` counts object
	/// bytes only, so the fast tier's true DRAM footprint is the two summed.
	/// `0` on all-DRAM stacks, which have no tiers and reserve nothing.
	fn dram_reserved_bytes(&self) -> CacheSize {
		0
	}

	fn fast_bytes_used(&self) -> CacheSize {
		0
	}

	/// Current bytes accounted to the slow tier. `0` for every stack except
	/// the hybrid stacks.
	fn slow_bytes_used(&self) -> CacheSize {
		0
	}

	/// Current number of objects in the fast tier. `0` for every stack
	/// except the hybrid stacks.
	fn fast_object_count(&self) -> usize {
		0
	}

	/// Current number of objects in the slow tier. `0` for every stack
	/// except the hybrid stacks.
	fn slow_object_count(&self) -> usize {
		0
	}

	/// Returns `true` if this stack has an internal sub-structure over its
	/// own capacity budget and wants `apply_evictions` to keep calling
	/// `evict_one()` even though overall `status.used_size()` is still
	/// within `max_size`. Only the 2Q/S3-FIFO hybrid stacks override this
	/// (`TwoQCompactHybridStack` and its siblings -- their `fifo_queue` has
	/// its own `k_in`-derived byte budget, independent of
	/// — and often much tighter than — the overall cache capacity); every
	/// other stack keeps the default `false`. Unlike `drain_tier_migrations`,
	/// which the stack can safely apply to its own bookkeeping in-place, an
	/// eviction needs the caller to also remove the object from the shared
	/// object map and adjust `status`, which only `apply_evictions`'s
	/// `evict_one()` + `erase()` pairing does correctly — a stack must never
	/// silently drop a key from its own bookkeeping without going through
	/// that path, or the object map and the stack's view of the world
	/// desync permanently.
	fn needs_capacity_eviction(&self) -> bool {
		false
	}

	/// Drains and returns the number of genuine demotions (fast-tier objects
	/// moved to slow due to capacity pressure) recorded since the last call.
	/// Distinct from `drain_tier_migrations`'s `Tier::Slow` entries: for
	/// `LfuCompactHybridStack`, a `Tier::Slow` migration can *also* be a fresh
	/// admission routed directly to slow because the fast tier was already
	/// full — that still needs the same physical `Object::set_data`
	/// correction (the object was initially built as `Fast` by the API
	/// layer), but it isn't a demotion in the paper's sense (no existing
	/// fast-tier object was displaced). `LruCompactHybridStack` /
	/// `TwoQCompactHybridStack` never produce that ambiguity (their admission
	/// never lands fast
	/// unconditionally then needs correcting), so they keep the default `0`
	/// and their callers keep counting every `Tier::Slow` migration as a
	/// demotion directly.
	/// Whether `PolicyWorker` should count each applied `Tier::Slow`
	/// migration as a demotion. The LFU-style design returns `false`: its
	/// `Tier::Slow` entries are not always genuine demotions, and the true
	/// count comes from [`Self::drain_demotions`] instead.
	fn inline_demotion_accounting(&self) -> bool {
		true
	}

	fn drain_demotions(&mut self) -> u64 {
		0
	}

	/// Returns `true` if this stack has permanently closed brand-new-key
	/// admission to the fast tier (see `LfuCompactHybridStack`'s module doc
	/// for why a one-time latch is needed on top of a raw byte-capacity
	/// check).
	/// Every other stack keeps the default `false` — `LruCompactHybridStack`
	/// and `TwoQCompactHybridStack`'s admission rules don't have this
	/// ambiguity (LRU always lands fast; 2Q-hybrid always lands slow), so
	/// there's nothing
	/// to latch. `PolicyWorker` mirrors this onto `AtomicStatus` so the
	/// API-calling thread — which has no access to the stack itself, owned
	/// exclusively by the worker thread — can decide a brand-new key's
	/// physical tier placement (`TieredBuffer::new_fast` vs. `new_slow`) up
	/// front in `PaperCache::set()`, instead of always guessing fast and
	/// relying on an async correction.
	fn admission_latched(&self) -> bool {
		false
	}

	/// Runtime-adjusts the LARGE fast segment's byte budget. Only
	/// `LruSizedCompactHybridStack` overrides this -- the SMALL segment reuses
	/// `resize_fast_tier` above; every other stack keeps the default no-op.
	fn resize_large_fast_tier(&mut self, _size: CacheSize) {}

	/// Runtime-adjusts the small/large size-classification threshold. Only
	/// `LruSizedCompactHybridStack` overrides this; every other stack keeps the
	/// default no-op.
	fn resize_size_threshold(&mut self, _size: CacheSize) {}

	/// Current bytes accounted to the SMALL fast segment. `0` for every
	/// stack except `LruSizedCompactHybridStack`.
	fn small_fast_bytes_used(&self) -> CacheSize {
		0
	}

	/// Current bytes accounted to the LARGE fast segment. `0` for every
	/// stack except `LruSizedCompactHybridStack`.
	fn large_fast_bytes_used(&self) -> CacheSize {
		0
	}

	/// Current number of objects in the SMALL fast segment. `0` for every
	/// stack except `LruSizedCompactHybridStack`.
	fn small_fast_object_count(&self) -> usize {
		0
	}

	/// Current number of objects in the LARGE fast segment. `0` for every
	/// stack except `LruSizedCompactHybridStack`.
	fn large_fast_object_count(&self) -> usize {
		0
	}

	/// Current bytes accounted to the SMALL slow list. `0` for every stack
	/// except `LruSizedCompactHybridStack`.
	fn small_slow_bytes_used(&self) -> CacheSize {
		0
	}

	/// Current bytes accounted to the LARGE slow list. `0` for every stack
	/// except `LruSizedCompactHybridStack`.
	fn large_slow_bytes_used(&self) -> CacheSize {
		0
	}

	/// Current number of objects in the SMALL slow list. `0` for every
	/// stack except `LruSizedCompactHybridStack`.
	fn small_slow_object_count(&self) -> usize {
		0
	}

	/// Current number of objects in the LARGE slow list. `0` for every
	/// stack except `LruSizedCompactHybridStack`.
	fn large_slow_object_count(&self) -> usize {
		0
	}
}

pub fn init_policy_stack(policy: PaperPolicy, max_size: CacheSize) -> Box<dyn PolicyStack> {
	match policy {
		PaperPolicy::Auto => Box::new(LfuStack::default()),
		PaperPolicy::LfuCompact => Box::new(LfuCompactStack::default()),
		PaperPolicy::FifoCompact => Box::new(FifoCompactStack::default()),
		PaperPolicy::ClockCompact => Box::new(ClockCompactStack::default()),
		PaperPolicy::SieveCompact => Box::new(SieveCompactStack::default()),
		PaperPolicy::MruCompact => Box::new(MruCompactStack::default()),
		PaperPolicy::Lfu => Box::new(LfuStack::default()),
		PaperPolicy::Fifo => Box::new(FifoStack::default()),
		PaperPolicy::Clock => Box::new(ClockStack::default()),
		PaperPolicy::Sieve => Box::new(SieveStack::default()),
		PaperPolicy::LruCompact => Box::new(LruCompactStack::default()),
		PaperPolicy::Lru => Box::new(LruStack::default()),
		PaperPolicy::Mru => Box::new(MruStack::default()),
		PaperPolicy::TwoQ(k_in, k_out) => Box::new(TwoQStack::new(k_in, k_out, max_size)),
		PaperPolicy::TwoQCompact(k_in, k_out) => Box::new(TwoQCompactStack::new(k_in, k_out, max_size)),
		PaperPolicy::Arc => Box::new(ArcStack::new(max_size)),
		PaperPolicy::SThreeFifo(ratio) => Box::new(SThreeFifoStack::new(ratio, max_size)),
		PaperPolicy::SThreeFifoCompact(ratio) => Box::new(SThreeFifoCompactStack::new(ratio, max_size)),

		// Default fast-tier budget is 20% of the overall cache size, matching
		// the tiering manager's default `dram_threshold` ratio (see
		// `TieringManager::new` in lib.rs). Runtime-adjustable afterward via
		// `resize_fast_tier` / `PaperCache::set_fast_tier_size` (step 10).
		// `with_shared_overhead` reserves the DRAM cost of the shared object
		// hashtable + eviction stacks out of that budget so demotion bounds
		// total DRAM, not just fast-tier values. Every hybrid arm below is built
		// the same way.
		//
		// `promote_k` comes from the policy value itself rather than a default
		// here, since it is carried in the policy string.
		#[cfg(feature = "hybrid_cache_common")]
		PaperPolicy::LruLfuCompactHybrid(promote_k) => Box::new(
			LruLfuCompactHybridStack::new((max_size as f64 * 0.2) as CacheSize, promote_k)
				.with_shared_overhead(
					crate::object::overhead::get_hybrid_dram_shared_overhead(&policy) as CacheSize,
				),
		),

		// Same default fast-tier budget/override mechanism as above. The
		// reservation is not optional: without it the stack gets a larger
		// effective fast tier than every policy it is compared against, which is
		// exactly how the first run of this variant produced a flattering and
		// meaningless result.
		#[cfg(feature = "hybrid_cache_common")]
		PaperPolicy::LruCompactHybrid => Box::new(
			LruCompactHybridStack::new((max_size as f64 * 0.2) as CacheSize)
				.with_shared_overhead(
					crate::object::overhead::get_hybrid_dram_shared_overhead(&policy) as CacheSize,
				),
		),

		#[cfg(not(feature = "hybrid_cache_common"))]
		PaperPolicy::LruCompactHybrid =>
			Box::new(LruCompactHybridStack::new((max_size as f64 * 0.2) as CacheSize)),

		// The budget handed here is the DRAM allowance; this stack derives its
		// smaller LOGICAL fast capacity from it, holding back `LAZY_COPY_WINDOW`
		// as room for candidates.
		#[cfg(feature = "hybrid_cache_common")]
		PaperPolicy::LruLazyCopyCompactHybrid => Box::new(
			LruLazyCopyCompactHybridStack::new((max_size as f64 * 0.2) as CacheSize)
				.with_shared_overhead(
					crate::object::overhead::get_hybrid_dram_shared_overhead(&policy) as CacheSize,
				),
		),

		#[cfg(not(feature = "hybrid_cache_common"))]
		PaperPolicy::LruLazyCopyCompactHybrid =>
			Box::new(LruLazyCopyCompactHybridStack::new((max_size as f64 * 0.2) as CacheSize)),

		#[cfg(feature = "hybrid_cache_common")]
		PaperPolicy::LfuCompactHybrid => Box::new(
			LfuCompactHybridStack::new((max_size as f64 * 0.2) as CacheSize)
				.with_shared_overhead(
					crate::object::overhead::get_hybrid_dram_shared_overhead(&policy) as CacheSize,
				),
		),

		// k_in comes from the policy string itself (same as plain `TwoQ`);
		// the fast-tier budget still defaults to 20% of max_size, same
		// override mechanism as the other hybrids.
		#[cfg(feature = "hybrid_cache_common")]
		PaperPolicy::TwoQCompactHybrid(k_in) => Box::new(
			TwoQCompactHybridStack::new(k_in, max_size, (max_size as f64 * 0.2) as CacheSize)
				.with_shared_overhead(
					crate::object::overhead::get_hybrid_dram_shared_overhead(&policy) as CacheSize,
				),
		),

		#[cfg(not(feature = "hybrid_cache_common"))]
		PaperPolicy::TwoQCompactHybrid(k_in) => Box::new(
			TwoQCompactHybridStack::new(k_in, max_size, (max_size as f64 * 0.2) as CacheSize),
		),

		// Same construction shape as `TwoQCompactHybrid` above. Note the default
		// fast-tier budget matters more here: `fifo_capacity` (k_in *
		// max_size) is carved *out of* it rather than being an independent
		// PMEM budget, so at the 20% default a k_in above 0.2 would leave
		// the main queue no fast segment at all. That is a legitimate
		// configuration (see the stack's module doc), and callers override
		// the budget via `ResizeFastTier` immediately after construction
		// anyway, but it is worth knowing when picking k_in.
		#[cfg(feature = "hybrid_cache_common")]
		PaperPolicy::TwoQFastAdmissionCompactHybrid(k_in) => Box::new(
			TwoQFastAdmissionCompactHybridStack::new(k_in, max_size, (max_size as f64 * 0.2) as CacheSize)
				.with_shared_overhead(
					crate::object::overhead::get_hybrid_dram_shared_overhead(&policy) as CacheSize,
				),
		),

		// Same construction shape and the same k_in-vs-fast-tier caveat as
		// `TwoQFastAdmissionCompactHybrid` above.
		#[cfg(feature = "hybrid_cache_common")]
		PaperPolicy::TwoQFastAdmissionReprieveCompactHybrid(k_in) => Box::new(
			TwoQFastAdmissionReprieveCompactHybridStack::new(k_in, max_size, (max_size as f64 * 0.2) as CacheSize).with_shared_overhead(
				crate::object::overhead::get_hybrid_dram_shared_overhead(&policy) as CacheSize,
			),
		),

		// The full three-queue 2Q -- the only design here whose queue
		// algorithm matches `PaperPolicy::TwoQ`'s (the other 2Q hybrids are
		// Simplified 2Q). Two parameters, not one: `k_out` sizes the live
		// `a1_out` overflow queue and is a real, read parameter here, unlike
		// in `TwoQStack`. Same default fast-tier budget and the same
		// k_in-vs-fast-tier caveat as `TwoQFastAdmissionCompactHybrid` above --
		// more acutely so, since `a1_in`'s reservation is carved out of the same
		// DRAM budget `am`'s fast segment draws on.
		#[cfg(feature = "hybrid_cache_common")]
		PaperPolicy::TwoQFullFastAdmissionCompactHybrid(k_in, k_out) => Box::new(
			TwoQFullFastAdmissionCompactHybridStack::new(k_in, k_out, max_size, (max_size as f64 * 0.2) as CacheSize).with_shared_overhead(
				crate::object::overhead::get_hybrid_dram_shared_overhead(&policy) as CacheSize,
			),
		),

		// Now carries the same `with_shared_overhead` reservation and the same
		// continuous drain-target settle as `LruCompactHybrid`/`LfuCompactHybrid`,
		// in the same two-arm with/without-feature shape this comment used to
		// ask for: a follow-up DRAM-usage measurement did show the same issue
		// (metadata is DRAM-resident but is not counted in `fast_used`, so
		// the fast tier overshot its budget).
		#[cfg(feature = "hybrid_cache_common")]
		PaperPolicy::FifoCompactHybrid => Box::new(
			FifoCompactHybridStack::new((max_size as f64 * 0.2) as CacheSize).with_shared_overhead(
				crate::object::overhead::get_hybrid_dram_shared_overhead(&policy) as CacheSize,
			),
		),

		// Same construction as `FifoCompactHybrid` above, which is the design
		// this is a second chance bolted onto -- see
		// `clock_compact_hybrid_stack.rs`'s module doc.
		#[cfg(feature = "hybrid_cache_common")]
		PaperPolicy::ClockCompactHybrid => Box::new(
			ClockCompactHybridStack::new((max_size as f64 * 0.2) as CacheSize).with_shared_overhead(
				crate::object::overhead::get_hybrid_dram_shared_overhead(&policy) as CacheSize,
			),
		),

		// Default small/large fast-segment budgets: 10% of max_size each
		// (totaling the same 20% aggregate default the other four hybrids
		// use for their single fast tier), immediately overridden by the
		// real constructor-supplied values right after construction (see
		// `PaperCache::new_sized_hybrid`'s three broadcasts). The
		// 4096-byte (4 KiB) default size threshold is a fixed constant
		// rather than max_size-scaled -- there's no principled way to scale
		// a *classification* threshold with overall cache size the way a
		// capacity budget scales -- also immediately overridden by the real
		// constructor-supplied value.
		#[cfg(feature = "hybrid_cache_common")]
		PaperPolicy::LruSizedCompactHybrid => Box::new(
			LruSizedCompactHybridStack::new(
				(max_size as f64 * 0.1) as CacheSize,
				(max_size as f64 * 0.1) as CacheSize,
				4_096,
			).with_shared_overhead(
				crate::object::overhead::get_hybrid_dram_shared_overhead(&policy) as CacheSize,
			),
		),

		// `ratio` comes from the policy string itself (same as plain
		// `SThreeFifo`/`TwoQCompactHybrid`); the fast-tier budget still
		// defaults to 20% of max_size, same override mechanism as the
		// other hybrids -- immediately overridden by the caller's real
		// CacheTierSize via `new_hybrid`'s `ResizeFastTier` broadcast, same
		// as every other hybrid design. Carries the same
		// `with_shared_overhead` reservation as `TwoQCompactHybrid`, which now has
		// one too: admission being always-slow does not avoid the cost, since
		// the hashtable and eviction-stack entries are DRAM-resident for
		// slow-tier objects just as much as for fast-tier ones.
		#[cfg(feature = "hybrid_cache_common")]
		PaperPolicy::S3FifoCompactHybrid(ratio) => Box::new(
			S3FifoCompactHybridStack::new(ratio, max_size, (max_size as f64 * 0.2) as CacheSize)
				.with_shared_overhead(
					crate::object::overhead::get_hybrid_dram_shared_overhead(&policy) as CacheSize,
				),
		),

		// Faithful tier-segmented S3-FIFO: 0..=3 counter, lazy promotion, lazy
		// eviction. Same construction as `S3FifoCompactHybrid` above.
		#[cfg(feature = "hybrid_cache_common")]
		PaperPolicy::S3FifoFaithfulCompactHybrid(ratio) => Box::new(
			S3FifoFaithfulCompactHybridStack::new(ratio, max_size, (max_size as f64 * 0.2) as CacheSize)
				.with_shared_overhead(
					crate::object::overhead::get_hybrid_dram_shared_overhead(&policy) as CacheSize,
				),
		),

		// Faithful tier-segmented S3-FIFO: 0..=3 counter, lazy promotion, lazy
		// eviction. Same construction as `S3FifoCompactHybrid` above.
		#[cfg(feature = "hybrid_cache_common")]
		PaperPolicy::S3FifoFaithfulFastAdmissionCompactHybrid(ratio) => Box::new(
			S3FifoFaithfulFastAdmissionCompactHybridStack::new(ratio, max_size, (max_size as f64 * 0.2) as CacheSize)
				.with_shared_overhead(
					crate::object::overhead::get_hybrid_dram_shared_overhead(&policy) as CacheSize,
				),
		),

		// Faithful tier-segmented S3-FIFO: 0..=3 counter, lazy promotion, lazy
		// eviction. Same construction as `S3FifoCompactHybrid` above.
		#[cfg(feature = "hybrid_cache_common")]
		PaperPolicy::S3FifoFaithfulReprieveCompactHybrid(ratio) => Box::new(
			S3FifoFaithfulReprieveCompactHybridStack::new(ratio, max_size, (max_size as f64 * 0.2) as CacheSize)
				.with_shared_overhead(
					crate::object::overhead::get_hybrid_dram_shared_overhead(&policy) as CacheSize,
				),
		),

		// Faithful tier-segmented S3-FIFO: 0..=3 counter, lazy promotion, lazy
		// eviction. Same construction as `S3FifoCompactHybrid` above.
		#[cfg(feature = "hybrid_cache_common")]
		PaperPolicy::S3FifoFaithfulFastAdmissionReprieveCompactHybrid(ratio) => Box::new(
			S3FifoFaithfulFastAdmissionReprieveCompactHybridStack::new(ratio, max_size, (max_size as f64 * 0.2) as CacheSize)
				.with_shared_overhead(
					crate::object::overhead::get_hybrid_dram_shared_overhead(&policy) as CacheSize,
				),
		),

		// Same construction/default-fast-tier-budget shape as
		// TwoQCompactHybrid/S3FifoCompactHybrid above -- see
		// two_q_ghost_compact_hybrid_stack.rs's module doc for the ghost-queue
		// mechanics these add on top.
		#[cfg(feature = "hybrid_cache_common")]
		PaperPolicy::TwoQGhostCompactHybrid(k_in) => Box::new(
			TwoQGhostCompactHybridStack::new(k_in, max_size, (max_size as f64 * 0.2) as CacheSize).with_shared_overhead(
				crate::object::overhead::get_hybrid_dram_shared_overhead(&policy) as CacheSize,
			),
		),

		// Same shape again, with the ghost queue keyed on the S3-FIFO
		// one-access queue -- see s3_fifo_ghost_compact_hybrid_stack.rs.
		#[cfg(feature = "hybrid_cache_common")]
		PaperPolicy::S3FifoGhostCompactHybrid(ratio) => Box::new(
			S3FifoGhostCompactHybridStack::new(ratio, max_size, (max_size as f64 * 0.2) as CacheSize).with_shared_overhead(
				crate::object::overhead::get_hybrid_dram_shared_overhead(&policy) as CacheSize,
			),
		),

		// Same construction/default-fast-tier-budget shape as
		// S3FifoGhostCompactHybrid above -- see
		// s3_fifo_ghost_lazy_demotion_compact_hybrid_stack.rs's module doc for
		// the demotion-time reference-bit gate this adds on top.
		#[cfg(feature = "hybrid_cache_common")]
		PaperPolicy::S3FifoGhostLazyDemotionCompactHybrid(ratio) => Box::new(
			S3FifoGhostLazyDemotionCompactHybridStack::new(ratio, max_size, (max_size as f64 * 0.2) as CacheSize).with_shared_overhead(
				crate::object::overhead::get_hybrid_dram_shared_overhead(&policy) as CacheSize,
			),
		),

		// Same construction/default-fast-tier-budget shape as
		// S3FifoGhostLazyDemotionCompactHybrid above -- see
		// s3_fifo_ghost_lazy_demotion_fast_admission_compact_hybrid_stack.rs's
		// module doc for the shared-DRAM-budget accounting this adds (the
		// one-access queue now competes with the main queue's fast segment
		// for the same fast_capacity).
		#[cfg(feature = "hybrid_cache_common")]
		PaperPolicy::S3FifoGhostLazyDemotionFastAdmissionCompactHybrid(ratio) => Box::new(
			S3FifoGhostLazyDemotionFastAdmissionCompactHybridStack::new(ratio, max_size, (max_size as f64 * 0.2) as CacheSize).with_shared_overhead(
				crate::object::overhead::get_hybrid_dram_shared_overhead(&policy) as CacheSize,
			),
		),

		// Same construction/default-fast-tier-budget shape as
		// S3FifoGhostLazyDemotionFastAdmissionCompactHybrid above -- see
		// s3_fifo_ghost_lazy_demotion_fast_admission_midpoint_compact_hybrid_stack.rs's
		// module doc for the mid-slow-segment reference-bit checkpoint this
		// adds on top.
		#[cfg(feature = "hybrid_cache_common")]
		PaperPolicy::S3FifoGhostLazyDemotionFastAdmissionMidpointCompactHybrid(ratio) => Box::new(
			S3FifoGhostLazyDemotionFastAdmissionMidpointCompactHybridStack::new(ratio, max_size, (max_size as f64 * 0.2) as CacheSize).with_shared_overhead(
				crate::object::overhead::get_hybrid_dram_shared_overhead(&policy) as CacheSize,
			),
		),

		// Same construction/default-fast-tier-budget shape as
		// S3FifoGhostLazyDemotionFastAdmissionMidpointCompactHybrid above -- see
		// s3_fifo_lazy_demotion_fast_admission_midpoint_reprieve_compact_hybrid_stack.rs's
		// module doc: no ghost queue (removed entirely), and a one-access
		// key that ages out is spliced into the slow tier of the main
		// queue instead of being evicted.
		#[cfg(feature = "hybrid_cache_common")]
		PaperPolicy::S3FifoLazyDemotionFastAdmissionMidpointReprieveCompactHybrid(ratio) => Box::new(
			S3FifoLazyDemotionFastAdmissionMidpointReprieveCompactHybridStack::new(ratio, max_size, (max_size as f64 * 0.2) as CacheSize).with_shared_overhead(
				crate::object::overhead::get_hybrid_dram_shared_overhead(&policy) as CacheSize,
			),
		),

		// Same construction shape as the midpoint variant above, minus the
		// mid-slow checkpoint -- see
		// s3_fifo_lazy_demotion_fast_admission_reprieve_compact_hybrid_stack.rs's
		// module doc.
		#[cfg(feature = "hybrid_cache_common")]
		PaperPolicy::S3FifoLazyDemotionFastAdmissionReprieveCompactHybrid(ratio) => Box::new(
			S3FifoLazyDemotionFastAdmissionReprieveCompactHybridStack::new(ratio, max_size, (max_size as f64 * 0.2) as CacheSize).with_shared_overhead(
				crate::object::overhead::get_hybrid_dram_shared_overhead(&policy) as CacheSize,
			),
		),

		// Same construction shape as its fast-admission sibling above. The
		// one-access queue is slow-tier here, so its `one_access_capacity`
		// bounds PMEM rather than being carved out of the DRAM budget -- see
		// s3_fifo_lazy_demotion_reprieve_compact_hybrid_stack.rs's
		// `effective_main_fast_capacity`.
		#[cfg(feature = "hybrid_cache_common")]
		PaperPolicy::S3FifoLazyDemotionReprieveCompactHybrid(ratio) => Box::new(
			S3FifoLazyDemotionReprieveCompactHybridStack::new(ratio, max_size, (max_size as f64 * 0.2) as CacheSize).with_shared_overhead(
				crate::object::overhead::get_hybrid_dram_shared_overhead(&policy) as CacheSize,
			),
		),

		// Same construction/default-fast-tier-budget shape as its
		// predecessor above -- see
		// s3_fifo_lazy_demotion_fast_admission_split_slow_reprieve_compact_hybrid_stack.rs's
		// module doc: the slow tier is split into two physical FIFO
		// segments, and every object's reference bit is checked as it
		// crosses between them.
		#[cfg(feature = "hybrid_cache_common")]
		PaperPolicy::S3FifoLazyDemotionFastAdmissionSplitSlowReprieveCompactHybrid(ratio) => Box::new(
			S3FifoLazyDemotionFastAdmissionSplitSlowReprieveCompactHybridStack::new(ratio, max_size, (max_size as f64 * 0.2) as CacheSize).with_shared_overhead(
				crate::object::overhead::get_hybrid_dram_shared_overhead(&policy) as CacheSize,
			),
		),
		// Hybrid stacks are compiled in every build; without the hybrid
		// feature there is no shared-overhead reservation to wire in, so
		// the bare construction the per-feature fallbacks used is kept.
		#[cfg(not(feature = "hybrid_cache_common"))]
		PaperPolicy::LruLfuCompactHybrid(promote_k) => Box::new(
			LruLfuCompactHybridStack::new((max_size as f64 * 0.2) as CacheSize, promote_k),
		),

		#[cfg(not(feature = "hybrid_cache_common"))]
		PaperPolicy::LfuCompactHybrid =>
			Box::new(LfuCompactHybridStack::new((max_size as f64 * 0.2) as CacheSize)),
		#[cfg(not(feature = "hybrid_cache_common"))]
		PaperPolicy::TwoQFastAdmissionCompactHybrid(k_in) => Box::new(TwoQFastAdmissionCompactHybridStack::new(k_in, max_size, (max_size as f64 * 0.2) as CacheSize)),
		#[cfg(not(feature = "hybrid_cache_common"))]
		PaperPolicy::TwoQFastAdmissionReprieveCompactHybrid(k_in) => Box::new(TwoQFastAdmissionReprieveCompactHybridStack::new(k_in, max_size, (max_size as f64 * 0.2) as CacheSize)),
		#[cfg(not(feature = "hybrid_cache_common"))]
		PaperPolicy::TwoQFullFastAdmissionCompactHybrid(k_in, k_out) => Box::new(TwoQFullFastAdmissionCompactHybridStack::new(k_in, k_out, max_size, (max_size as f64 * 0.2) as CacheSize)),
		#[cfg(not(feature = "hybrid_cache_common"))]
		PaperPolicy::FifoCompactHybrid => Box::new(FifoCompactHybridStack::new((max_size as f64 * 0.2) as CacheSize)),
		#[cfg(not(feature = "hybrid_cache_common"))]
		PaperPolicy::ClockCompactHybrid => Box::new(ClockCompactHybridStack::new((max_size as f64 * 0.2) as CacheSize)),
		#[cfg(not(feature = "hybrid_cache_common"))]
		PaperPolicy::LruSizedCompactHybrid => Box::new(LruSizedCompactHybridStack::new(
			(max_size as f64 * 0.1) as CacheSize,
			(max_size as f64 * 0.1) as CacheSize,
			4_096,
		)),
		#[cfg(not(feature = "hybrid_cache_common"))]
		PaperPolicy::S3FifoCompactHybrid(ratio) => Box::new(S3FifoCompactHybridStack::new(ratio, max_size, (max_size as f64 * 0.2) as CacheSize)),
		#[cfg(not(feature = "hybrid_cache_common"))]
		PaperPolicy::S3FifoFaithfulCompactHybrid(ratio) => Box::new(S3FifoFaithfulCompactHybridStack::new(ratio, max_size, (max_size as f64 * 0.2) as CacheSize)),
		#[cfg(not(feature = "hybrid_cache_common"))]
		PaperPolicy::S3FifoFaithfulFastAdmissionCompactHybrid(ratio) => Box::new(S3FifoFaithfulFastAdmissionCompactHybridStack::new(ratio, max_size, (max_size as f64 * 0.2) as CacheSize)),
		#[cfg(not(feature = "hybrid_cache_common"))]
		PaperPolicy::S3FifoFaithfulReprieveCompactHybrid(ratio) => Box::new(S3FifoFaithfulReprieveCompactHybridStack::new(ratio, max_size, (max_size as f64 * 0.2) as CacheSize)),
		#[cfg(not(feature = "hybrid_cache_common"))]
		PaperPolicy::S3FifoFaithfulFastAdmissionReprieveCompactHybrid(ratio) => Box::new(S3FifoFaithfulFastAdmissionReprieveCompactHybridStack::new(ratio, max_size, (max_size as f64 * 0.2) as CacheSize)),
		#[cfg(not(feature = "hybrid_cache_common"))]
		PaperPolicy::TwoQGhostCompactHybrid(k_in) => Box::new(TwoQGhostCompactHybridStack::new(k_in, max_size, (max_size as f64 * 0.2) as CacheSize)),
		#[cfg(not(feature = "hybrid_cache_common"))]
		PaperPolicy::S3FifoGhostCompactHybrid(ratio) => Box::new(S3FifoGhostCompactHybridStack::new(ratio, max_size, (max_size as f64 * 0.2) as CacheSize)),
		#[cfg(not(feature = "hybrid_cache_common"))]
		PaperPolicy::S3FifoGhostLazyDemotionCompactHybrid(ratio) => Box::new(S3FifoGhostLazyDemotionCompactHybridStack::new(ratio, max_size, (max_size as f64 * 0.2) as CacheSize)),
		#[cfg(not(feature = "hybrid_cache_common"))]
		PaperPolicy::S3FifoGhostLazyDemotionFastAdmissionCompactHybrid(ratio) => Box::new(S3FifoGhostLazyDemotionFastAdmissionCompactHybridStack::new(ratio, max_size, (max_size as f64 * 0.2) as CacheSize)),
		#[cfg(not(feature = "hybrid_cache_common"))]
		PaperPolicy::S3FifoGhostLazyDemotionFastAdmissionMidpointCompactHybrid(ratio) => Box::new(S3FifoGhostLazyDemotionFastAdmissionMidpointCompactHybridStack::new(ratio, max_size, (max_size as f64 * 0.2) as CacheSize)),
		#[cfg(not(feature = "hybrid_cache_common"))]
		PaperPolicy::S3FifoLazyDemotionFastAdmissionMidpointReprieveCompactHybrid(ratio) => Box::new(S3FifoLazyDemotionFastAdmissionMidpointReprieveCompactHybridStack::new(ratio, max_size, (max_size as f64 * 0.2) as CacheSize)),
		#[cfg(not(feature = "hybrid_cache_common"))]
		PaperPolicy::S3FifoLazyDemotionFastAdmissionReprieveCompactHybrid(ratio) => Box::new(S3FifoLazyDemotionFastAdmissionReprieveCompactHybridStack::new(ratio, max_size, (max_size as f64 * 0.2) as CacheSize)),
		#[cfg(not(feature = "hybrid_cache_common"))]
		PaperPolicy::S3FifoLazyDemotionReprieveCompactHybrid(ratio) => Box::new(S3FifoLazyDemotionReprieveCompactHybridStack::new(ratio, max_size, (max_size as f64 * 0.2) as CacheSize)),
		#[cfg(not(feature = "hybrid_cache_common"))]
		PaperPolicy::S3FifoLazyDemotionFastAdmissionSplitSlowReprieveCompactHybrid(ratio) => Box::new(S3FifoLazyDemotionFastAdmissionSplitSlowReprieveCompactHybridStack::new(ratio, max_size, (max_size as f64 * 0.2) as CacheSize)),

	}
}

#[cfg(test)]
mod init_policy_stack_tests {
	//! Runtime-dispatch tests for `init_policy_stack`.
	//!
	//! The "every design in every build, chosen at runtime" architecture rests
	//! entirely on the match above: all `POLICY_VARIANT_COUNT` policies
	//! (`HYBRID_DESIGN_COUNT` of them tiered) are compiled into every binary,
	//! and the design that actually runs is picked by the `PaperPolicy` value
	//! handed to `PaperCache::new`. Nothing else in the crate checks that the
	//! match returns the stack it was asked for. The arms are near-identical
	//! one-expression lines, several of which differ by a single word inside a
	//! 60-character type name
	//! (`S3FifoGhostLazyDemotionCompactHybridStack` vs.
	//! `S3FifoGhostLazyDemotionFastAdmissionCompactHybridStack`), so a
	//! copy-paste slip between two of them would silently run a different
	//! eviction design for
	//! the lifetime of the process and misattribute every number measured from
	//! it.
	//!
	//! One suite covers both cfg branches. The
	//! `feature = "hybrid_cache_common"` arms and the
	//! `not(feature = "hybrid_cache_common")` fallbacks construct the *same*
	//! stack types -- the feature only adds the `with_shared_overhead`
	//! reservation, which changes a fast-tier budget, not a stack's identity --
	//! and every one of the hybrid policies has an arm in both sets, so no
	//! variant is unreachable in either build and no gate is needed here. (A
	//! hybrid design added to only one of the two sets makes the match
	//! non-exhaustive in the other build, which is a compile error rather than
	//! something a test could observe.)
	//!
	//! Everything here is construction-only: no key is ever inserted, so
	//! nothing allocates through the `Hybrid` allocator and these tests need no
	//! warmed PMEM pool under `eviction_stacks_pmem`.

	use std::collections::HashSet;

	use super::*;

	/// Every arm derives its sub-budgets from `max_size`: `max_size * 0.2` for
	/// the hybrids' fast tier, `max_size * 0.1` per segment for
	/// `LruSizedCompactHybrid`, `k_in * max_size` for the 2Q family and
	/// `ratio * max_size` for the S3-FIFO family. 1 MB is the scale the hybrid
	/// integration suites build their caches at, and it keeps every one of
	/// those derived budgets comfortably non-zero -- below ~5 bytes the 20%
	/// fast-tier budget truncates to 0, which is a degenerate stack rather than
	/// a dispatch question.
	const TEST_MAX_SIZE: CacheSize = 1_000_000;

	/// Number of `PaperPolicy` variants, and therefore the number of rows the
	/// table below must have. Kept as a named constant so a mismatch reads as
	/// "a design is missing from the table", not as an off-by-one.
	const POLICY_VARIANT_COUNT: usize = 43;

	/// Number of variants for which `PaperPolicy::is_hybrid` must hold: the
	/// tiered designs this crate exists to compare.
	const HYBRID_DESIGN_COUNT: usize = 25;

	/// Every `PaperPolicy` variant, listed explicitly, in declaration order.
	///
	/// Column 0 is the value handed to `init_policy_stack`. Column 1 is a
	/// *different value of the same variant* -- a different `k_in`, `ratio` or
	/// `promote_k` -- used to pin down that `is_policy` discriminates on the
	/// variant and not on the payload. For the payload-free variants the two
	/// columns are necessarily the same value.
	///
	/// The list is written out rather than derived: `variant_name` below is an
	/// exhaustive match with no `_` arm, so adding a variant to `PaperPolicy`
	/// stops this file compiling until the new design is added here too.
	const POLICY_DISPATCH_TABLE: [(PaperPolicy, PaperPolicy); POLICY_VARIANT_COUNT] = [
		(PaperPolicy::Auto, PaperPolicy::Auto),
		(PaperPolicy::LfuCompact, PaperPolicy::LfuCompact),
		(PaperPolicy::FifoCompact, PaperPolicy::FifoCompact),
		(PaperPolicy::ClockCompact, PaperPolicy::ClockCompact),
		(PaperPolicy::SieveCompact, PaperPolicy::SieveCompact),
		(PaperPolicy::MruCompact, PaperPolicy::MruCompact),
		(PaperPolicy::TwoQCompact(0.25, 0.5), PaperPolicy::TwoQCompact(0.25, 0.5)),
		(PaperPolicy::Lfu, PaperPolicy::Lfu),
		(PaperPolicy::Fifo, PaperPolicy::Fifo),
		(PaperPolicy::Clock, PaperPolicy::Clock),
		(PaperPolicy::Sieve, PaperPolicy::Sieve),
		(PaperPolicy::LruCompact, PaperPolicy::LruCompact),
		(PaperPolicy::Lru, PaperPolicy::Lru),
		(PaperPolicy::Mru, PaperPolicy::Mru),
		(PaperPolicy::TwoQ(0.25, 0.25), PaperPolicy::TwoQ(0.5, 0.4)),
		(PaperPolicy::Arc, PaperPolicy::Arc),
		(PaperPolicy::SThreeFifo(0.1), PaperPolicy::SThreeFifo(0.9)),
		(PaperPolicy::SThreeFifoCompact(0.1), PaperPolicy::SThreeFifoCompact(0.9)),
		(PaperPolicy::S3FifoFaithfulCompactHybrid(0.1), PaperPolicy::S3FifoFaithfulCompactHybrid(0.9)),
		(PaperPolicy::S3FifoFaithfulFastAdmissionCompactHybrid(0.1), PaperPolicy::S3FifoFaithfulFastAdmissionCompactHybrid(0.9)),
		(PaperPolicy::S3FifoFaithfulReprieveCompactHybrid(0.1), PaperPolicy::S3FifoFaithfulReprieveCompactHybrid(0.9)),
		(PaperPolicy::S3FifoFaithfulFastAdmissionReprieveCompactHybrid(0.1), PaperPolicy::S3FifoFaithfulFastAdmissionReprieveCompactHybrid(0.9)),
		(PaperPolicy::LruCompactHybrid, PaperPolicy::LruCompactHybrid),
		(PaperPolicy::LruLazyCopyCompactHybrid, PaperPolicy::LruLazyCopyCompactHybrid),
		(PaperPolicy::LfuCompactHybrid, PaperPolicy::LfuCompactHybrid),
		(PaperPolicy::TwoQCompactHybrid(0.1), PaperPolicy::TwoQCompactHybrid(0.9)),
		(PaperPolicy::TwoQFastAdmissionCompactHybrid(0.1), PaperPolicy::TwoQFastAdmissionCompactHybrid(0.9)),
		(PaperPolicy::TwoQFastAdmissionReprieveCompactHybrid(0.1), PaperPolicy::TwoQFastAdmissionReprieveCompactHybrid(0.9)),
		(PaperPolicy::TwoQFullFastAdmissionCompactHybrid(0.25, 0.25), PaperPolicy::TwoQFullFastAdmissionCompactHybrid(0.5, 0.4)),
		(PaperPolicy::FifoCompactHybrid, PaperPolicy::FifoCompactHybrid),
		(PaperPolicy::ClockCompactHybrid, PaperPolicy::ClockCompactHybrid),
		(PaperPolicy::LruSizedCompactHybrid, PaperPolicy::LruSizedCompactHybrid),
		(PaperPolicy::LruLfuCompactHybrid(3), PaperPolicy::LruLfuCompactHybrid(7)),
		(PaperPolicy::S3FifoCompactHybrid(0.1), PaperPolicy::S3FifoCompactHybrid(0.9)),
		(PaperPolicy::TwoQGhostCompactHybrid(0.1), PaperPolicy::TwoQGhostCompactHybrid(0.9)),
		(PaperPolicy::S3FifoGhostCompactHybrid(0.1), PaperPolicy::S3FifoGhostCompactHybrid(0.9)),
		(PaperPolicy::S3FifoGhostLazyDemotionCompactHybrid(0.1), PaperPolicy::S3FifoGhostLazyDemotionCompactHybrid(0.9)),
		(PaperPolicy::S3FifoGhostLazyDemotionFastAdmissionCompactHybrid(0.1), PaperPolicy::S3FifoGhostLazyDemotionFastAdmissionCompactHybrid(0.9)),
		(PaperPolicy::S3FifoGhostLazyDemotionFastAdmissionMidpointCompactHybrid(0.1), PaperPolicy::S3FifoGhostLazyDemotionFastAdmissionMidpointCompactHybrid(0.9)),
		(PaperPolicy::S3FifoLazyDemotionFastAdmissionMidpointReprieveCompactHybrid(0.1), PaperPolicy::S3FifoLazyDemotionFastAdmissionMidpointReprieveCompactHybrid(0.9)),
		(PaperPolicy::S3FifoLazyDemotionFastAdmissionReprieveCompactHybrid(0.1), PaperPolicy::S3FifoLazyDemotionFastAdmissionReprieveCompactHybrid(0.9)),
		(PaperPolicy::S3FifoLazyDemotionReprieveCompactHybrid(0.1), PaperPolicy::S3FifoLazyDemotionReprieveCompactHybrid(0.9)),
		(PaperPolicy::S3FifoLazyDemotionFastAdmissionSplitSlowReprieveCompactHybrid(0.1), PaperPolicy::S3FifoLazyDemotionFastAdmissionSplitSlowReprieveCompactHybrid(0.9)),
	];

	/// The variant's name, ignoring any payload.
	///
	/// Deliberately exhaustive with no `_` arm: adding a variant to
	/// `PaperPolicy` breaks this match at compile time, and that is the signal
	/// to add the new design to `POLICY_DISPATCH_TABLE` (and to bump
	/// `POLICY_VARIANT_COUNT`) so it is dispatch-tested like every other one.
	fn variant_name(policy: &PaperPolicy) -> &'static str {
		match policy {
			PaperPolicy::Auto => "Auto",
			PaperPolicy::LfuCompact => "LfuCompact",
			PaperPolicy::FifoCompact => "FifoCompact",
			PaperPolicy::ClockCompact => "ClockCompact",
			PaperPolicy::SieveCompact => "SieveCompact",
			PaperPolicy::MruCompact => "MruCompact",
			PaperPolicy::TwoQCompact(..) => "TwoQCompact",
			PaperPolicy::Lfu => "Lfu",
			PaperPolicy::Fifo => "Fifo",
			PaperPolicy::Clock => "Clock",
			PaperPolicy::Sieve => "Sieve",
			PaperPolicy::LruCompact => "LruCompact",
			PaperPolicy::Lru => "Lru",
			PaperPolicy::Mru => "Mru",
			PaperPolicy::TwoQ(..) => "TwoQ",
			PaperPolicy::Arc => "Arc",
			PaperPolicy::SThreeFifo(_) => "SThreeFifo",
			PaperPolicy::SThreeFifoCompact(_) => "SThreeFifoCompact",
			PaperPolicy::S3FifoFaithfulCompactHybrid(_) => "S3FifoFaithfulCompactHybrid",
			PaperPolicy::S3FifoFaithfulFastAdmissionCompactHybrid(_) => "S3FifoFaithfulFastAdmissionCompactHybrid",
			PaperPolicy::S3FifoFaithfulReprieveCompactHybrid(_) => "S3FifoFaithfulReprieveCompactHybrid",
			PaperPolicy::S3FifoFaithfulFastAdmissionReprieveCompactHybrid(_) => "S3FifoFaithfulFastAdmissionReprieveCompactHybrid",
			PaperPolicy::LruCompactHybrid => "LruCompactHybrid",
			PaperPolicy::LruLazyCopyCompactHybrid => "LruLazyCopyCompactHybrid",
			PaperPolicy::LfuCompactHybrid => "LfuCompactHybrid",
			PaperPolicy::TwoQCompactHybrid(_) => "TwoQCompactHybrid",
			PaperPolicy::TwoQFastAdmissionCompactHybrid(_) => "TwoQFastAdmissionCompactHybrid",
			PaperPolicy::TwoQFastAdmissionReprieveCompactHybrid(_) => "TwoQFastAdmissionReprieveCompactHybrid",
			PaperPolicy::TwoQFullFastAdmissionCompactHybrid(..) => "TwoQFullFastAdmissionCompactHybrid",
			PaperPolicy::FifoCompactHybrid => "FifoCompactHybrid",
			PaperPolicy::ClockCompactHybrid => "ClockCompactHybrid",
			PaperPolicy::LruSizedCompactHybrid => "LruSizedCompactHybrid",
			PaperPolicy::LruLfuCompactHybrid(_) => "LruLfuCompactHybrid",
			PaperPolicy::S3FifoCompactHybrid(_) => "S3FifoCompactHybrid",
			PaperPolicy::TwoQGhostCompactHybrid(_) => "TwoQGhostCompactHybrid",
			PaperPolicy::S3FifoGhostCompactHybrid(_) => "S3FifoGhostCompactHybrid",
			PaperPolicy::S3FifoGhostLazyDemotionCompactHybrid(_) => "S3FifoGhostLazyDemotionCompactHybrid",
			PaperPolicy::S3FifoGhostLazyDemotionFastAdmissionCompactHybrid(_) => "S3FifoGhostLazyDemotionFastAdmissionCompactHybrid",
			PaperPolicy::S3FifoGhostLazyDemotionFastAdmissionMidpointCompactHybrid(_) => "S3FifoGhostLazyDemotionFastAdmissionMidpointCompactHybrid",
			PaperPolicy::S3FifoLazyDemotionFastAdmissionMidpointReprieveCompactHybrid(_) => "S3FifoLazyDemotionFastAdmissionMidpointReprieveCompactHybrid",
			PaperPolicy::S3FifoLazyDemotionFastAdmissionReprieveCompactHybrid(_) => "S3FifoLazyDemotionFastAdmissionReprieveCompactHybrid",
			PaperPolicy::S3FifoLazyDemotionReprieveCompactHybrid(_) => "S3FifoLazyDemotionReprieveCompactHybrid",
			PaperPolicy::S3FifoLazyDemotionFastAdmissionSplitSlowReprieveCompactHybrid(_) => "S3FifoLazyDemotionFastAdmissionSplitSlowReprieveCompactHybrid",
		}
	}

	/// The group of policy values that share one constructed stack.
	///
	/// The match above has exactly one such collision: `Auto` and `Lfu` both
	/// build a bare `LfuStack` (`Auto` means "let the cache choose", and it is
	/// resolved to LFU right there in the dispatch). A single `LfuStack` cannot
	/// report two different identities, so those two are one family and the
	/// too-loose-`is_policy` cross-check below skips that pair -- and only that
	/// pair. Every other variant is its own family, so every other pairing is
	/// checked.
	fn dispatch_family(policy: &PaperPolicy) -> &'static str {
		match policy {
			PaperPolicy::Auto | PaperPolicy::Lfu => "Lfu",
			other => variant_name(other),
		}
	}

	/// The premise of the architecture: asking for a design gets you that
	/// design, for every one of them.
	#[test]
	fn every_policy_variant_dispatches_to_a_stack_that_claims_it() {
		for (policy, _) in POLICY_DISPATCH_TABLE {
			let stack = init_policy_stack(policy, TEST_MAX_SIZE);

			if policy.is_auto() {
				// `Auto` is resolved to LFU by the dispatch itself, so the
				// stack it hands back is a plain `LfuStack` and may answer to
				// either name; which one it answers to is not this test's
				// business. That the two arms stay in lockstep is pinned by
				// `auto_dispatches_to_the_same_stack_as_lfu` below.
				assert!(
					stack.is_policy(&PaperPolicy::Auto) || stack.is_policy(&PaperPolicy::Lfu),
					"the stack built for `{policy}` claims to be neither `auto` nor the LFU design `auto` resolves to",
				);

				continue;
			}

			assert!(
				stack.is_policy(&policy),
				"`init_policy_stack` built a stack for `{policy}` that does not report itself as `{policy}`: that arm constructs some other design",
			);
		}
	}

	/// The other half of the premise, and the half a positive-only test cannot
	/// see: a stack that says yes to everything would pass the test above while
	/// making every runtime policy decision meaningless.
	///
	/// Foils are compared by *variant*, never by value, because `is_policy` is
	/// deliberately payload-blind (see
	/// `is_policy_matches_on_the_variant_not_its_payload`) -- asking a
	/// `TwoQCompactHybrid(0.1)` stack about `TwoQCompactHybrid(0.9)` is not a
	/// cross-check.
	#[test]
	fn no_stack_claims_a_policy_it_was_not_built_for() {
		for (policy, _) in POLICY_DISPATCH_TABLE {
			let stack = init_policy_stack(policy, TEST_MAX_SIZE);

			for (foil, _) in POLICY_DISPATCH_TABLE {
				if dispatch_family(&foil) == dispatch_family(&policy) {
					continue;
				}

				assert!(
					!stack.is_policy(&foil),
					"the stack built for `{policy}` also claims to be `{foil}`: `is_policy` is too loose, so a switch between those two designs would be treated as a no-op and the old design would keep running",
				);
			}
		}
	}

	/// `is_policy` discriminates on the variant, not on the payload. Every
	/// stack that carries a `k_in`/`ratio`/`promote_k` keeps it as its own
	/// field; the identity question it answers is "which design am I", not
	/// "which parameterisation am I".
	///
	/// This is the contract the cross-check above depends on, and it is what
	/// stops a `k_in` that round-tripped through the policy string from making
	/// a stack disown itself.
	#[test]
	fn is_policy_discriminates_on_the_payload_of_a_parameterised_policy() {
		// A parameterised policy names both a design AND its tuning, so a
		// stack built for one payload must not answer to another: that is
		// what makes `MiniStackManager` rebuild rather than silently keep a
		// stack tuned to the old value (mini_stack/manager.rs:50, :125).
		// `TwoQStack::is_policy` is the clearest statement of the rule --
		// `self.k_in == *k_in && self.k_out == *k_out`.
		//
		// `LruLfuCompactHybrid` is the exception in the crate: it matches its
		// own variant with `(_)` and ignores `promote_k`, so a change to that
		// knob alone does not read as a different policy. Pinned here rather
		// than papered over, so that if it is ever brought in line this test
		// fails and says so.
		for (policy, same_variant_other_payload) in POLICY_DISPATCH_TABLE {
			if policy == same_variant_other_payload {
				continue; // payload-free variant: nothing to discriminate
			}

			let stack = init_policy_stack(policy, TEST_MAX_SIZE);

			if matches!(policy, PaperPolicy::LruLfuCompactHybrid(_)) {
				assert!(
					stack.is_policy(&same_variant_other_payload),
					"`{policy}` is documented as the one payload-insensitive \
					 design, but its stack now rejects \
					 `{same_variant_other_payload}` -- if `is_policy` was \
					 deliberately tightened to compare `promote_k`, move it \
					 in with the others and delete this branch",
				);

				continue;
			}

			assert!(
				!stack.is_policy(&same_variant_other_payload),
				"the stack built for `{policy}` also claims to be \
				 `{same_variant_other_payload}`: `is_policy` ignores the \
				 payload, so a retune would not rebuild the stack",
			);
		}
	}

	/// Guards the three tests above from silently shrinking: they are only
	/// exhaustive if the table is.
	#[test]
	fn the_dispatch_table_covers_every_policy_variant_exactly_once() {
		let names = POLICY_DISPATCH_TABLE
			.into_iter()
			.map(|(policy, _)| variant_name(&policy))
			.collect::<HashSet<&'static str>>();

		assert_eq!(
			names.len(),
			POLICY_VARIANT_COUNT,
			"the dispatch table has {} rows but only {} distinct variants: a row is duplicated, so some design is not dispatch-tested at all",
			POLICY_DISPATCH_TABLE.len(),
			names.len(),
		);

		for (policy, same_variant_other_payload) in POLICY_DISPATCH_TABLE {
			assert_eq!(
				variant_name(&policy),
				variant_name(&same_variant_other_payload),
				"the second column of the `{policy}` row is a different variant (`{same_variant_other_payload}`), which would turn the payload check into an accidental cross-variant check",
			);
		}
	}

	/// `PaperPolicy::is_hybrid` is a hand-written `matches!` over the tiered
	/// variants with no exhaustiveness check of its own, and it is what
	/// decides whether
	/// a cache gets a fast tier at all. Anchor it to the one list that *is*
	/// compiler-checked against the enum.
	#[test]
	fn every_tiered_design_is_reported_as_hybrid() {
		let hybrids = POLICY_DISPATCH_TABLE
			.into_iter()
			.filter(|(policy, _)| policy.is_hybrid())
			.count();

		assert_eq!(
			hybrids, HYBRID_DESIGN_COUNT,
			"`is_hybrid` recognises {hybrids} of the {POLICY_VARIANT_COUNT} designs, expected {HYBRID_DESIGN_COUNT}",
		);

		for (policy, _) in POLICY_DISPATCH_TABLE {
			assert_eq!(
				policy.is_hybrid(),
				variant_name(&policy).ends_with("Hybrid"),
				"`{policy}`: `is_hybrid` disagrees with the variant's own name",
			);
		}
	}

	/// A freshly dispatched stack tracks nothing, for every design. The object
	/// map it will be paired with is empty at that moment, and `apply_evictions`
	/// trusts the stack's view: a stack that arrives holding keys of its own
	/// would have `evict_one` hand back a key the map has never heard of.
	#[test]
	fn every_freshly_built_stack_is_empty() {
		for (policy, _) in POLICY_DISPATCH_TABLE {
			let stack = init_policy_stack(policy, TEST_MAX_SIZE);
			let tracked = stack.len();

			assert_eq!(
				tracked, 0,
				"the stack built for `{policy}` reports {tracked} tracked keys before anything was inserted",
			);

			assert!(
				!stack.contains(1),
				"the stack built for `{policy}` claims to contain a key that was never inserted",
			);
		}
	}

	/// `Auto` and `Lfu` are the only two policies that share a stack. Pin that
	/// down directly, rather than leaving it as an unstated exception in the
	/// cross-check above: `Auto` is what a caller passes when it does not want
	/// to name a design, so if its arm ever drifted, those callers would
	/// quietly get different eviction behaviour with nothing else to notice.
	#[test]
	fn auto_dispatches_to_the_same_stack_as_lfu() {
		let auto = init_policy_stack(PaperPolicy::Auto, TEST_MAX_SIZE);
		let lfu = init_policy_stack(PaperPolicy::Lfu, TEST_MAX_SIZE);

		for (candidate, _) in POLICY_DISPATCH_TABLE {
			assert_eq!(
				auto.is_policy(&candidate),
				lfu.is_policy(&candidate),
				"the stacks built for `auto` and `lfu` disagree about `{candidate}`, so `auto` is no longer resolving to the LFU design",
			);
		}
	}
}
