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
mod lfu_hybrid_stack;

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
		lfu_hybrid_stack::LfuHybridStack,
	},
};

/// Outcome of a policy stack access that may carry extra routing signals.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AccessOutcome {
	None,
	GhostHit,
}

/// Which tier an object currently lives in, for policy stacks that track a
/// segmented (fast/slow) queue. Used by `LruHybridStack`
/// (`PaperPolicy::LruHybrid`, recency-segmented) and `LfuHybridStack`
/// (`PaperPolicy::LfuHybrid`, frequency-segmented); every other stack's
/// default `drain_tier_migrations` never produces one.
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
	/// except `LruHybridStack`, which shrinking may trigger immediate
	/// demotions for (see `drain_tier_migrations`).
	fn resize_fast_tier(&mut self, _size: CacheSize) {}

	/// Drains and returns every (key, new tier) pair that crossed the
	/// fast/slow boundary since the last call. Only `LruHybridStack` and
	/// `LfuHybridStack` ever produce entries; every other stack keeps the
	/// default empty `Vec`. The caller (`PolicyWorker`) is responsible for
	/// physically migrating each returned key's object bytes to `new_tier`.
	fn drain_tier_migrations(&mut self) -> Vec<(HashedKey, Tier)> {
		Vec::new()
	}

	/// Current bytes accounted to the fast tier. `0` for every stack except
	/// `LruHybridStack`/`LfuHybridStack`.
	fn fast_bytes_used(&self) -> CacheSize {
		0
	}

	/// Current bytes accounted to the slow tier. `0` for every stack except
	/// `LruHybridStack`/`LfuHybridStack`.
	fn slow_bytes_used(&self) -> CacheSize {
		0
	}

	/// Current number of objects in the fast tier. `0` for every stack
	/// except `LruHybridStack`/`LfuHybridStack`.
	fn fast_object_count(&self) -> usize {
		0
	}

	/// Current number of objects in the slow tier. `0` for every stack
	/// except `LruHybridStack`/`LfuHybridStack`.
	fn slow_object_count(&self) -> usize {
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
		PaperPolicy::LruHybrid => Box::new(LruHybridStack::new((max_size as f64 * 0.2) as CacheSize)),

		// Same default fast-tier budget/override mechanism as `LruHybrid`.
		PaperPolicy::LfuHybrid => Box::new(LfuHybridStack::new((max_size as f64 * 0.2) as CacheSize)),
	}
}
