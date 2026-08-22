/*
 * Copyright (c) Kia Shakiba
 *
 * This source code is licensed under the GNU AGPLv3 license found in the
 * LICENSE file in the root directory of this source tree.
 */

mod lfu_stack;
mod fifo_stack;
mod clock_stack;
mod sieve_stack;
mod lru_stack;
mod mru_stack;
mod two_q_stack;
mod arc_stack;
mod s_three_fifo_stack;
mod lru_hybrid_stack;
mod lru_lfu_hybrid_stack;
mod lfu_hybrid_stack;
mod two_q_hybrid_stack;
mod two_q_fast_admission_hybrid_stack;
mod two_q_fast_admission_reprieve_hybrid_stack;
mod fifo_hybrid_stack;
mod lru_sized_hybrid_stack;
mod s3_fifo_hybrid_stack;
mod two_q_ghost_hybrid_stack;
mod s3_fifo_ghost_hybrid_stack;
mod s3_fifo_ghost_lazy_demotion_hybrid_stack;
mod s3_fifo_ghost_lazy_demotion_fast_admission_hybrid_stack;
mod s3_fifo_ghost_lazy_demotion_fast_admission_midpoint_hybrid_stack;
mod s3_fifo_lazy_demotion_fast_admission_midpoint_reprieve_hybrid_stack;
mod s3_fifo_lazy_demotion_fast_admission_reprieve_hybrid_stack;
mod s3_fifo_lazy_demotion_reprieve_hybrid_stack;
mod s3_fifo_lazy_demotion_fast_admission_split_slow_reprieve_hybrid_stack;

#[cfg(feature = "eviction_stacks_pmem")] mod pmem_collections;

use crate::{
	CacheSize,
	HashedKey,
	policy::PaperPolicy,
	object::ObjectSize,
	worker::policy::policy_stack::{
		lfu_stack::LfuStack,
		fifo_stack::FifoStack,
		clock_stack::ClockStack,
		sieve_stack::SieveStack,
		lru_stack::LruStack,
		mru_stack::MruStack,
		two_q_stack::TwoQStack,
		arc_stack::ArcStack,
		s_three_fifo_stack::SThreeFifoStack,
		lru_hybrid_stack::LruHybridStack,
		lru_lfu_hybrid_stack::LruLfuHybridStack,
		lfu_hybrid_stack::LfuHybridStack,
		two_q_hybrid_stack::TwoQHybridStack,
		two_q_fast_admission_hybrid_stack::TwoQFastAdmissionHybridStack,
		two_q_fast_admission_reprieve_hybrid_stack::TwoQFastAdmissionReprieveHybridStack,
		fifo_hybrid_stack::FifoHybridStack,
		lru_sized_hybrid_stack::LruSizedHybridStack,
		s3_fifo_hybrid_stack::S3FifoHybridStack,
		two_q_ghost_hybrid_stack::TwoQGhostHybridStack,
		s3_fifo_ghost_hybrid_stack::S3FifoGhostHybridStack,
		s3_fifo_ghost_lazy_demotion_hybrid_stack::S3FifoGhostLazyDemotionHybridStack,
		s3_fifo_ghost_lazy_demotion_fast_admission_hybrid_stack::S3FifoGhostLazyDemotionFastAdmissionHybridStack,
		s3_fifo_ghost_lazy_demotion_fast_admission_midpoint_hybrid_stack::S3FifoGhostLazyDemotionFastAdmissionMidpointHybridStack,
		s3_fifo_lazy_demotion_fast_admission_midpoint_reprieve_hybrid_stack::S3FifoLazyDemotionFastAdmissionMidpointReprieveHybridStack,
		s3_fifo_lazy_demotion_fast_admission_reprieve_hybrid_stack::S3FifoLazyDemotionFastAdmissionReprieveHybridStack,
		s3_fifo_lazy_demotion_reprieve_hybrid_stack::S3FifoLazyDemotionReprieveHybridStack,
		s3_fifo_lazy_demotion_fast_admission_split_slow_reprieve_hybrid_stack::S3FifoLazyDemotionFastAdmissionSplitSlowReprieveHybridStack,
	},
};

/// Outcome of a policy stack access that may carry extra routing signals.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AccessOutcome {
	None,
	GhostHit,
}

/// Fast-tier watermarks, shared by every hybrid stack's `settle_fast_tier`.
///
/// Historically each stack drained back to exactly its fast-tier ceiling, so
/// the tier sat pinned at 100% utilisation and *every* admission triggered a
/// demotion of exactly one object. That produces migration batches of one,
/// which maximises per-batch worker overhead and makes the copies impossible
/// to parallelise (measured: >99% of `apply_tier_migrations` calls carried
/// 0-1 entries).
///
/// With watermarks, `settle_fast_tier` only triggers once usage exceeds
/// `high * capacity`, then drains to `low * capacity` in one pass -- trading
/// a slice of resident fast-tier capacity for larger, less frequent demotion
/// batches.
///
/// NOTE: a 90% low-water floor previously existed in `LruHybridStack` and was
/// removed at the user's explicit request for hurting performance; a 2%
/// version (`FAST_TIER_LOW_WATER_RATIO = 0.98`) replaced it as a burst
/// margin. These defaults are deliberately more aggressive and are tunable at
/// runtime so the tradeoff can be measured rather than assumed. Set
/// `FAST_TIER_HIGH_WATERMARK=1.0` and `FAST_TIER_LOW_WATERMARK=1.0` to
/// restore the original drain-to-ceiling behaviour exactly.
pub mod watermarks {
	use std::sync::OnceLock;

	pub const DEFAULT_HIGH: f64 = 0.98;
	pub const DEFAULT_LOW: f64 = 0.95;

	static HIGH: OnceLock<f64> = OnceLock::new();
	static LOW: OnceLock<f64> = OnceLock::new();

	fn read(var: &str, default: f64) -> f64 {
		std::env::var(var)
			.ok()
			.and_then(|v| v.parse::<f64>().ok())
			.filter(|v| *v > 0.0 && *v <= 1.0)
			.unwrap_or(default)
	}

	/// Fraction of the effective fast-tier budget above which a demotion pass
	/// is triggered. `1.0` restores trigger-at-ceiling.
	pub fn high() -> f64 {
		*HIGH.get_or_init(|| read("FAST_TIER_HIGH_WATERMARK", DEFAULT_HIGH))
	}

	/// Fraction of the effective fast-tier budget a triggered pass drains down
	/// to. Clamped to at most `high()` so a misconfiguration cannot invert the
	/// pair (which would make every pass a no-op and let the tier overrun).
	pub fn low() -> f64 {
		*LOW.get_or_init(|| {
			let l = read("FAST_TIER_LOW_WATERMARK", DEFAULT_LOW);
			if l > high() { high() } else { l }
		})
	}

	/// The byte threshold at which a demotion pass triggers.
	pub fn high_bytes(effective_capacity: u64) -> u64 {
		(effective_capacity as f64 * high()) as u64
	}

	/// The byte target a triggered demotion pass drains down to.
	pub fn low_bytes(effective_capacity: u64) -> u64 {
		(effective_capacity as f64 * low()) as u64
	}
}


/// Which tier an object currently lives in, for policy stacks that track a
/// segmented (fast/slow) queue. Used by `LruHybridStack`
/// (`PaperPolicy::LruHybrid`, recency-segmented), `LfuHybridStack`
/// (`PaperPolicy::LfuHybrid`, frequency-segmented), `TwoQHybridStack`
/// (`PaperPolicy::TwoQHybrid`, 2Q-segmented), and `FifoHybridStack`
/// (`PaperPolicy::FifoHybrid`, insertion-order-segmented); every other
/// stack's default `drain_tier_migrations` never produces one.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Tier {
	Fast,
	Slow,
}

pub trait PolicyStack
where
	Self: Send,
{
	fn is_policy(&self, policy: &PaperPolicy) -> bool;
	fn len(&self) -> usize;

	fn contains(&self, key: HashedKey) -> bool;
	fn insert(&mut self, key: HashedKey, size: ObjectSize);
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
	/// (`LruHybridStack`, `LfuHybridStack`, `TwoQHybridStack`) ever produce
	/// entries; every other stack keeps the default empty `Vec`. The caller
	/// (`PolicyWorker`) is responsible for physically migrating each
	/// returned key's object bytes to `new_tier`.
	fn drain_tier_migrations(&mut self) -> Vec<(HashedKey, Tier)> {
		Vec::new()
	}

	/// Current bytes accounted to the fast tier. `0` for every stack except
	/// the hybrid stacks.
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
	/// within `max_size`. Only `TwoQHybridStack` overrides this (its
	/// `fifo_queue` has its own `k_in`-derived byte budget, independent of
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
	/// `LfuHybridStack`, a `Tier::Slow` migration can *also* be a fresh
	/// admission routed directly to slow because the fast tier was already
	/// full — that still needs the same physical `Object::set_data`
	/// correction (the object was initially built as `Fast` by the API
	/// layer), but it isn't a demotion in the paper's sense (no existing
	/// fast-tier object was displaced). `LruHybridStack`/`TwoQHybridStack`
	/// never produce that ambiguity (their admission never lands fast
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
	/// admission to the fast tier (see `LfuHybridStack`'s module doc for why
	/// a one-time latch is needed on top of a raw byte-capacity check).
	/// Every other stack keeps the default `false` — `LruHybridStack` and
	/// `TwoQHybridStack`'s admission rules don't have this ambiguity (LRU
	/// always lands fast; 2Q-hybrid always lands slow), so there's nothing
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
	/// `LruSizedHybridStack` overrides this -- the SMALL segment reuses
	/// `resize_fast_tier` above; every other stack keeps the default no-op.
	fn resize_large_fast_tier(&mut self, _size: CacheSize) {}

	/// Runtime-adjusts the small/large size-classification threshold. Only
	/// `LruSizedHybridStack` overrides this; every other stack keeps the
	/// default no-op.
	fn resize_size_threshold(&mut self, _size: CacheSize) {}

	/// Current bytes accounted to the SMALL fast segment. `0` for every
	/// stack except `LruSizedHybridStack`.
	fn small_fast_bytes_used(&self) -> CacheSize {
		0
	}

	/// Current bytes accounted to the LARGE fast segment. `0` for every
	/// stack except `LruSizedHybridStack`.
	fn large_fast_bytes_used(&self) -> CacheSize {
		0
	}

	/// Current number of objects in the SMALL fast segment. `0` for every
	/// stack except `LruSizedHybridStack`.
	fn small_fast_object_count(&self) -> usize {
		0
	}

	/// Current number of objects in the LARGE fast segment. `0` for every
	/// stack except `LruSizedHybridStack`.
	fn large_fast_object_count(&self) -> usize {
		0
	}

	/// Current bytes accounted to the SMALL slow list. `0` for every stack
	/// except `LruSizedHybridStack`.
	fn small_slow_bytes_used(&self) -> CacheSize {
		0
	}

	/// Current bytes accounted to the LARGE slow list. `0` for every stack
	/// except `LruSizedHybridStack`.
	fn large_slow_bytes_used(&self) -> CacheSize {
		0
	}

	/// Current number of objects in the SMALL slow list. `0` for every
	/// stack except `LruSizedHybridStack`.
	fn small_slow_object_count(&self) -> usize {
		0
	}

	/// Current number of objects in the LARGE slow list. `0` for every
	/// stack except `LruSizedHybridStack`.
	fn large_slow_object_count(&self) -> usize {
		0
	}
}

pub fn init_policy_stack(policy: PaperPolicy, max_size: CacheSize) -> Box<dyn PolicyStack> {
	match policy {
		PaperPolicy::Auto => Box::new(LfuStack::default()),
		PaperPolicy::Lfu => Box::new(LfuStack::default()),
		PaperPolicy::Fifo => Box::new(FifoStack::default()),
		PaperPolicy::Clock => Box::new(ClockStack::default()),
		PaperPolicy::Sieve => Box::new(SieveStack::default()),
		PaperPolicy::Lru => Box::new(LruStack::default()),
		PaperPolicy::Mru => Box::new(MruStack::default()),
		PaperPolicy::TwoQ(k_in, k_out) => Box::new(TwoQStack::new(k_in, k_out, max_size)),
		PaperPolicy::Arc => Box::new(ArcStack::new(max_size)),
		PaperPolicy::SThreeFifo(ratio) => Box::new(SThreeFifoStack::new(ratio, max_size)),

		// Default fast-tier budget is 20% of the overall cache size, matching
		// the tiering manager's default `dram_threshold` ratio (see
		// `TieringManager::new` in lib.rs). Runtime-adjustable afterward via
		// `resize_fast_tier` / `PaperCache::set_fast_tier_size` (step 10).
		// `with_shared_overhead` reserves the DRAM cost of the shared object
		// hashtable + eviction stacks out of that budget so demotion bounds
		// total DRAM, not just fast-tier values.
		#[cfg(feature = "hybrid_cache_common")]
		PaperPolicy::LruHybrid => Box::new(
			LruHybridStack::new((max_size as f64 * 0.2) as CacheSize)
				.with_shared_overhead(
					crate::object::overhead::get_hybrid_dram_shared_overhead(&policy) as CacheSize,
				),
		),

		// Same default fast-tier budget/override mechanism as `LruHybrid`.
		// `promote_k` comes from the policy value itself rather than a default
		// here, since it is carried in the policy string.
		#[cfg(feature = "hybrid_cache_common")]
		PaperPolicy::LruLfuHybrid(promote_k) => Box::new(
			LruLfuHybridStack::new((max_size as f64 * 0.2) as CacheSize, promote_k)
				.with_shared_overhead(
					crate::object::overhead::get_hybrid_dram_shared_overhead(&policy) as CacheSize,
				),
		),

		// Same default fast-tier budget/override mechanism as `LruHybrid`.
		#[cfg(feature = "hybrid_cache_common")]
		PaperPolicy::LfuHybrid => Box::new(
			LfuHybridStack::new((max_size as f64 * 0.2) as CacheSize)
				.with_shared_overhead(
					crate::object::overhead::get_hybrid_dram_shared_overhead(&policy) as CacheSize,
				),
		),

		// k_in comes from the policy string itself (same as plain `TwoQ`);
		// the fast-tier budget still defaults to 20% of max_size, same
		// override mechanism as the other two hybrids.
		#[cfg(feature = "hybrid_cache_common")]
		PaperPolicy::TwoQHybrid(k_in) => Box::new(
			TwoQHybridStack::new(k_in, max_size, (max_size as f64 * 0.2) as CacheSize).with_shared_overhead(
				crate::object::overhead::get_hybrid_dram_shared_overhead(&policy) as CacheSize,
			),
		),

		// Same construction shape as `TwoQHybrid` above. Note the default
		// fast-tier budget matters more here: `fifo_capacity` (k_in *
		// max_size) is carved *out of* it rather than being an independent
		// PMEM budget, so at the 20% default a k_in above 0.2 would leave
		// the main queue no fast segment at all. That is a legitimate
		// configuration (see the stack's module doc), and callers override
		// the budget via `ResizeFastTier` immediately after construction
		// anyway, but it is worth knowing when picking k_in.
		#[cfg(feature = "hybrid_cache_common")]
		PaperPolicy::TwoQFastAdmissionHybrid(k_in) => Box::new(
			TwoQFastAdmissionHybridStack::new(k_in, max_size, (max_size as f64 * 0.2) as CacheSize).with_shared_overhead(
				crate::object::overhead::get_hybrid_dram_shared_overhead(&policy) as CacheSize,
			),
		),

		// Same construction shape and the same k_in-vs-fast-tier caveat as
		// `TwoQFastAdmissionHybrid` above.
		#[cfg(feature = "hybrid_cache_common")]
		PaperPolicy::TwoQFastAdmissionReprieveHybrid(k_in) => Box::new(
			TwoQFastAdmissionReprieveHybridStack::new(k_in, max_size, (max_size as f64 * 0.2) as CacheSize).with_shared_overhead(
				crate::object::overhead::get_hybrid_dram_shared_overhead(&policy) as CacheSize,
			),
		),

		// Now carries the same `with_shared_overhead` reservation and the same
		// high/low fast-tier watermarks as `LruHybrid`/`LfuHybrid`, in the
		// same two-arm with/without-feature shape this comment used to ask
		// for: a follow-up DRAM-usage measurement did show the same issue
		// (metadata is DRAM-resident but is not counted in `fast_used`, so
		// the fast tier overshot its budget).
		#[cfg(feature = "hybrid_cache_common")]
		PaperPolicy::FifoHybrid => Box::new(
			FifoHybridStack::new((max_size as f64 * 0.2) as CacheSize).with_shared_overhead(
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
		PaperPolicy::LruSizedHybrid => Box::new(
			LruSizedHybridStack::new(
				(max_size as f64 * 0.1) as CacheSize,
				(max_size as f64 * 0.1) as CacheSize,
				4_096,
			).with_shared_overhead(
				crate::object::overhead::get_hybrid_dram_shared_overhead(&policy) as CacheSize,
			),
		),

		// `ratio` comes from the policy string itself (same as plain
		// `SThreeFifo`/`TwoQHybrid`); the fast-tier budget still
		// defaults to 20% of max_size, same override mechanism as the
		// other hybrids -- immediately overridden by the caller's real
		// CacheTierSize via `new_hybrid`'s `ResizeFastTier` broadcast, same
		// as every other hybrid design. Carries the same
		// `with_shared_overhead` reservation as `TwoQHybrid`, which now has
		// one too: admission being always-slow does not avoid the cost, since
		// the hashtable and eviction-stack entries are DRAM-resident for
		// slow-tier objects just as much as for fast-tier ones.
		#[cfg(feature = "hybrid_cache_common")]
		PaperPolicy::S3FifoHybrid(ratio) => Box::new(
			S3FifoHybridStack::new(ratio, max_size, (max_size as f64 * 0.2) as CacheSize).with_shared_overhead(
				crate::object::overhead::get_hybrid_dram_shared_overhead(&policy) as CacheSize,
			),
		),

		// Same construction/default-fast-tier-budget shape as TwoQHybrid/
		// S3FifoHybrid above -- see two_q_ghost_hybrid_stack.rs's module doc
		// for the ghost-queue mechanics these add on top.
		#[cfg(feature = "hybrid_cache_common")]
		PaperPolicy::TwoQGhostHybrid(k_in) => Box::new(
			TwoQGhostHybridStack::new(k_in, max_size, (max_size as f64 * 0.2) as CacheSize).with_shared_overhead(
				crate::object::overhead::get_hybrid_dram_shared_overhead(&policy) as CacheSize,
			),
		),
		#[cfg(feature = "hybrid_cache_common")]
		PaperPolicy::S3FifoGhostHybrid(ratio) => Box::new(
			S3FifoGhostHybridStack::new(ratio, max_size, (max_size as f64 * 0.2) as CacheSize).with_shared_overhead(
				crate::object::overhead::get_hybrid_dram_shared_overhead(&policy) as CacheSize,
			),
		),

		// Same construction/default-fast-tier-budget shape as
		// S3FifoGhostHybrid above -- see
		// s3_fifo_ghost_lazy_demotion_hybrid_stack.rs's module doc for the
		// demotion-time reference-bit gate this adds on top.
		#[cfg(feature = "hybrid_cache_common")]
		PaperPolicy::S3FifoGhostLazyDemotionHybrid(ratio) => Box::new(
			S3FifoGhostLazyDemotionHybridStack::new(ratio, max_size, (max_size as f64 * 0.2) as CacheSize).with_shared_overhead(
				crate::object::overhead::get_hybrid_dram_shared_overhead(&policy) as CacheSize,
			),
		),

		// Same construction/default-fast-tier-budget shape as
		// S3FifoGhostLazyDemotionHybrid above -- see
		// s3_fifo_ghost_lazy_demotion_fast_admission_hybrid_stack.rs's
		// module doc for the shared-DRAM-budget accounting this adds (the
		// one-access queue now competes with the main queue's fast segment
		// for the same fast_capacity).
		#[cfg(feature = "hybrid_cache_common")]
		PaperPolicy::S3FifoGhostLazyDemotionFastAdmissionHybrid(ratio) => Box::new(
			S3FifoGhostLazyDemotionFastAdmissionHybridStack::new(ratio, max_size, (max_size as f64 * 0.2) as CacheSize).with_shared_overhead(
				crate::object::overhead::get_hybrid_dram_shared_overhead(&policy) as CacheSize,
			),
		),

		// Same construction/default-fast-tier-budget shape as
		// S3FifoGhostLazyDemotionFastAdmissionHybrid above -- see
		// s3_fifo_ghost_lazy_demotion_fast_admission_midpoint_hybrid_stack.rs's
		// module doc for the mid-slow-segment reference-bit checkpoint this
		// adds on top.
		#[cfg(feature = "hybrid_cache_common")]
		PaperPolicy::S3FifoGhostLazyDemotionFastAdmissionMidpointHybrid(ratio) => Box::new(
			S3FifoGhostLazyDemotionFastAdmissionMidpointHybridStack::new(ratio, max_size, (max_size as f64 * 0.2) as CacheSize).with_shared_overhead(
				crate::object::overhead::get_hybrid_dram_shared_overhead(&policy) as CacheSize,
			),
		),

		// Same construction/default-fast-tier-budget shape as
		// S3FifoGhostLazyDemotionFastAdmissionMidpointHybrid above -- see
		// s3_fifo_lazy_demotion_fast_admission_midpoint_reprieve_hybrid_stack.rs's
		// module doc: no ghost queue (removed entirely), and a one-access
		// key that ages out is spliced into the slow tier of the main
		// queue instead of being evicted.
		#[cfg(feature = "hybrid_cache_common")]
		PaperPolicy::S3FifoLazyDemotionFastAdmissionMidpointReprieveHybrid(ratio) => Box::new(
			S3FifoLazyDemotionFastAdmissionMidpointReprieveHybridStack::new(ratio, max_size, (max_size as f64 * 0.2) as CacheSize).with_shared_overhead(
				crate::object::overhead::get_hybrid_dram_shared_overhead(&policy) as CacheSize,
			),
		),

		// Same construction shape as the midpoint variant above, minus the
		// mid-slow checkpoint -- see
		// s3_fifo_lazy_demotion_fast_admission_reprieve_hybrid_stack.rs's module doc.
		#[cfg(feature = "hybrid_cache_common")]
		PaperPolicy::S3FifoLazyDemotionFastAdmissionReprieveHybrid(ratio) => Box::new(
			S3FifoLazyDemotionFastAdmissionReprieveHybridStack::new(ratio, max_size, (max_size as f64 * 0.2) as CacheSize).with_shared_overhead(
				crate::object::overhead::get_hybrid_dram_shared_overhead(&policy) as CacheSize,
			),
		),

		// Same construction shape as its fast-admission sibling above. The
		// one-access queue is slow-tier here, so its `one_access_capacity`
		// bounds PMEM rather than being carved out of the DRAM budget -- see
		// s3_fifo_lazy_demotion_reprieve_hybrid_stack.rs's
		// `effective_main_fast_capacity`.
		#[cfg(feature = "hybrid_cache_common")]
		PaperPolicy::S3FifoLazyDemotionReprieveHybrid(ratio) => Box::new(
			S3FifoLazyDemotionReprieveHybridStack::new(ratio, max_size, (max_size as f64 * 0.2) as CacheSize).with_shared_overhead(
				crate::object::overhead::get_hybrid_dram_shared_overhead(&policy) as CacheSize,
			),
		),

		// Same construction/default-fast-tier-budget shape as its
		// predecessor above -- see
		// s3_fifo_lazy_demotion_fast_admission_split_slow_reprieve_hybrid_stack.rs's
		// module doc: the slow tier is split into two physical FIFO
		// segments, and every object's reference bit is checked as it
		// crosses between them.
		#[cfg(feature = "hybrid_cache_common")]
		PaperPolicy::S3FifoLazyDemotionFastAdmissionSplitSlowReprieveHybrid(ratio) => Box::new(
			S3FifoLazyDemotionFastAdmissionSplitSlowReprieveHybridStack::new(ratio, max_size, (max_size as f64 * 0.2) as CacheSize).with_shared_overhead(
				crate::object::overhead::get_hybrid_dram_shared_overhead(&policy) as CacheSize,
			),
		),
		// Hybrid stacks are compiled in every build; without the hybrid
		// feature there is no shared-overhead reservation to wire in, so
		// the bare construction the per-feature fallbacks used is kept.
		#[cfg(not(feature = "hybrid_cache_common"))]
		PaperPolicy::LruHybrid => Box::new(LruHybridStack::new((max_size as f64 * 0.2) as CacheSize)),
		#[cfg(not(feature = "hybrid_cache_common"))]
		PaperPolicy::LruLfuHybrid(promote_k) => Box::new(
			LruLfuHybridStack::new((max_size as f64 * 0.2) as CacheSize, promote_k),
		),
		#[cfg(not(feature = "hybrid_cache_common"))]
		PaperPolicy::LfuHybrid => Box::new(LfuHybridStack::new((max_size as f64 * 0.2) as CacheSize)),
		#[cfg(not(feature = "hybrid_cache_common"))]
		PaperPolicy::TwoQHybrid(k_in) => Box::new(TwoQHybridStack::new(k_in, max_size, (max_size as f64 * 0.2) as CacheSize)),
		#[cfg(not(feature = "hybrid_cache_common"))]
		PaperPolicy::TwoQFastAdmissionHybrid(k_in) => Box::new(TwoQFastAdmissionHybridStack::new(k_in, max_size, (max_size as f64 * 0.2) as CacheSize)),
		#[cfg(not(feature = "hybrid_cache_common"))]
		PaperPolicy::TwoQFastAdmissionReprieveHybrid(k_in) => Box::new(TwoQFastAdmissionReprieveHybridStack::new(k_in, max_size, (max_size as f64 * 0.2) as CacheSize)),
		#[cfg(not(feature = "hybrid_cache_common"))]
		PaperPolicy::FifoHybrid => Box::new(FifoHybridStack::new((max_size as f64 * 0.2) as CacheSize)),
		#[cfg(not(feature = "hybrid_cache_common"))]
		PaperPolicy::LruSizedHybrid => Box::new(LruSizedHybridStack::new(
			(max_size as f64 * 0.1) as CacheSize,
			(max_size as f64 * 0.1) as CacheSize,
			4_096,
		)),
		#[cfg(not(feature = "hybrid_cache_common"))]
		PaperPolicy::S3FifoHybrid(ratio) => Box::new(S3FifoHybridStack::new(ratio, max_size, (max_size as f64 * 0.2) as CacheSize)),
		#[cfg(not(feature = "hybrid_cache_common"))]
		PaperPolicy::TwoQGhostHybrid(k_in) => Box::new(TwoQGhostHybridStack::new(k_in, max_size, (max_size as f64 * 0.2) as CacheSize)),
		#[cfg(not(feature = "hybrid_cache_common"))]
		PaperPolicy::S3FifoGhostHybrid(ratio) => Box::new(S3FifoGhostHybridStack::new(ratio, max_size, (max_size as f64 * 0.2) as CacheSize)),
		#[cfg(not(feature = "hybrid_cache_common"))]
		PaperPolicy::S3FifoGhostLazyDemotionHybrid(ratio) => Box::new(S3FifoGhostLazyDemotionHybridStack::new(ratio, max_size, (max_size as f64 * 0.2) as CacheSize)),
		#[cfg(not(feature = "hybrid_cache_common"))]
		PaperPolicy::S3FifoGhostLazyDemotionFastAdmissionHybrid(ratio) => Box::new(S3FifoGhostLazyDemotionFastAdmissionHybridStack::new(ratio, max_size, (max_size as f64 * 0.2) as CacheSize)),
		#[cfg(not(feature = "hybrid_cache_common"))]
		PaperPolicy::S3FifoGhostLazyDemotionFastAdmissionMidpointHybrid(ratio) => Box::new(S3FifoGhostLazyDemotionFastAdmissionMidpointHybridStack::new(ratio, max_size, (max_size as f64 * 0.2) as CacheSize)),
		#[cfg(not(feature = "hybrid_cache_common"))]
		PaperPolicy::S3FifoLazyDemotionFastAdmissionMidpointReprieveHybrid(ratio) => Box::new(S3FifoLazyDemotionFastAdmissionMidpointReprieveHybridStack::new(ratio, max_size, (max_size as f64 * 0.2) as CacheSize)),
		#[cfg(not(feature = "hybrid_cache_common"))]
		PaperPolicy::S3FifoLazyDemotionFastAdmissionReprieveHybrid(ratio) => Box::new(S3FifoLazyDemotionFastAdmissionReprieveHybridStack::new(ratio, max_size, (max_size as f64 * 0.2) as CacheSize)),
		#[cfg(not(feature = "hybrid_cache_common"))]
		PaperPolicy::S3FifoLazyDemotionReprieveHybrid(ratio) => Box::new(S3FifoLazyDemotionReprieveHybridStack::new(ratio, max_size, (max_size as f64 * 0.2) as CacheSize)),
		#[cfg(not(feature = "hybrid_cache_common"))]
		PaperPolicy::S3FifoLazyDemotionFastAdmissionSplitSlowReprieveHybrid(ratio) => Box::new(S3FifoLazyDemotionFastAdmissionSplitSlowReprieveHybridStack::new(ratio, max_size, (max_size as f64 * 0.2) as CacheSize)),

	}
}
