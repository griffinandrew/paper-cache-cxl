/*
 * Copyright (c) Kia Shakiba
 *
 * This source code is licensed under the GNU AGPLv3 license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! Two-tier DRAM-first cache with S3-FIFO-inspired promotion logic.
//!
//! [`S3FifoHybridCache`] wraps two [`crate::PaperCache`] instances:
//!
//! - **Small tier** (configurable fraction of total capacity, default 10%):
//!   receives all newly inserted items, mirroring S3-FIFO's small queue.
//! - **Main tier** (remaining capacity, default 90%):
//!   holds items that have accumulated enough small-tier accesses to be
//!   "promoted", mirroring S3-FIFO's main queue.
//!
//! # Promotion logic
//!
//! Each access to an item resident in the small tier increments an internal
//! per-key frequency counter.  When the counter reaches
//! [`HybridCacheConfig::freq_threshold`] (default: 2) the item is
//! **promoted**: its value is copied into the main tier and removed from
//! the small tier.  This mirrors S3-FIFO's rule of promoting objects whose
//! `freq > 1`.
//!
//! # Example
//!
//! ```
//! use paper_cache::hybridcache::{S3FifoHybridCache, HybridCacheConfig};
//!
//! let config = HybridCacheConfig::default();
//! let cache = S3FifoHybridCache::<u32>::new(config).unwrap();
//!
//! // Insert a value – it starts in the small tier.
//! cache.set(1u32, &[0u8; 128], None).unwrap();
//!
//! // First access: still in small tier.
//! let val = cache.get(&1u32).unwrap();
//! assert_eq!(val.len(), 128);
//!
//! // Second access reaches freq_threshold=2, triggering promotion.
//! let _val = cache.get(&1u32).unwrap();
//!
//! let stats = cache.stats();
//! assert_eq!(stats.promotions, 1);
//! ```

use std::{
	hash::{Hash, BuildHasher, RandomState},
	sync::{
		Arc,
		atomic::{AtomicU64, Ordering},
	},
};

use dashmap::DashMap;
use typesize::TypeSize;

use crate::{PaperCache, PaperPolicy, CacheError};

// ── Configuration ────────────────────────────────────────────────────────────

/// Configuration for [`S3FifoHybridCache`].
///
/// Use [`HybridCacheConfig::default`] to obtain sensible defaults (10 MB
/// total, 10 % small tier, frequency threshold of 2).
#[derive(Debug, Clone)]
pub struct HybridCacheConfig {
	/// Total cache capacity in bytes.
	pub total_size: u64,

	/// Fraction of [`total_size`](Self::total_size) reserved for the small
	/// tier.  Must be in `0.0..=1.0`; clamped silently otherwise.
	///
	/// The main tier receives the remainder `(1.0 - small_ratio) * total_size`.
	/// Defaults to `0.1` (10 %).
	pub small_ratio: f64,

	/// Number of small-tier hits required to promote an item to the main tier.
	///
	/// Mirrors S3-FIFO's `freq > 1` rule.  Defaults to `2`.
	pub freq_threshold: u64,

	/// Eviction policy used by the small tier.
	///
	/// Defaults to [`PaperPolicy::Fifo`], matching S3-FIFO's small-queue
	/// semantics (first-in, first-out among cold items).
	pub small_policy: PaperPolicy,

	/// Eviction policy used by the main tier.
	///
	/// Defaults to [`PaperPolicy::SThreeFifo`]`(0.1)` to preserve
	/// frequency-aware eviction for promoted hot items.
	pub main_policy: PaperPolicy,
}

impl Default for HybridCacheConfig {
	fn default() -> Self {
		HybridCacheConfig {
			total_size: 10_000_000, // 10 MB
			small_ratio: 0.1,
			freq_threshold: 2,
			small_policy: PaperPolicy::Fifo,
			main_policy: PaperPolicy::SThreeFifo(0.1),
		}
	}
}

// ── Statistics ───────────────────────────────────────────────────────────────

/// A point-in-time snapshot of [`S3FifoHybridCache`] runtime statistics.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct HybridCacheStats {
	/// Hits served from the **main** (hot) tier.
	pub main_hits: u64,
	/// Hits served from the **small** (new) tier.
	pub small_hits: u64,
	/// Lookups that found the key in neither tier.
	pub misses: u64,
	/// Items moved from the small tier to the main tier.
	pub promotions: u64,
}

struct AtomicHybridStats {
	main_hits: AtomicU64,
	small_hits: AtomicU64,
	misses: AtomicU64,
	promotions: AtomicU64,
}

impl AtomicHybridStats {
	fn new() -> Self {
		AtomicHybridStats {
			main_hits: AtomicU64::new(0),
			small_hits: AtomicU64::new(0),
			misses: AtomicU64::new(0),
			promotions: AtomicU64::new(0),
		}
	}

	fn snapshot(&self) -> HybridCacheStats {
		HybridCacheStats {
			main_hits: self.main_hits.load(Ordering::Relaxed),
			small_hits: self.small_hits.load(Ordering::Relaxed),
			misses: self.misses.load(Ordering::Relaxed),
			promotions: self.promotions.load(Ordering::Relaxed),
		}
	}
}

// ── S3FifoHybridCache ────────────────────────────────────────────────────────

/// A two-tier DRAM-first cache with S3-FIFO-inspired promotion logic.
///
/// See the [module documentation](self) for architecture details.
///
/// # Type parameter
///
/// `K` is the key type.  It must satisfy:
/// `'static + Eq + Hash + TypeSize + Debug + Clone + Send + Sync`.
///
/// Values are always stored as raw byte slices (`&[u8]` / `Vec<u8>`),
/// consistent with the `all_dram` storage tier used by the underlying
/// [`PaperCache`] instances.
pub struct S3FifoHybridCache<K>
where
	K: 'static + Eq + Hash + TypeSize + std::fmt::Debug + Clone + Send + Sync,
{
	/// Small tier – receives all newly inserted items.
	small: PaperCache<K, crate::BufferDRAM>,
	/// Main tier – receives items promoted from the small tier.
	main: PaperCache<K, crate::BufferDRAM>,
	/// Per-key access frequency in the small tier.
	///
	/// Keyed by the hashed form of K to avoid requiring K to be Hash for
	/// DashMap independently of the PaperCache hasher.
	access_counts: DashMap<crate::HashedKey, u64>,
	/// Minimum small-tier hit count before promoting an item.
	freq_threshold: u64,
	/// Shared atomic counters for [`HybridCacheStats`].
	stats: Arc<AtomicHybridStats>,
	/// Hasher used to derive [`crate::HashedKey`] from K, consistent with the
	/// internal hasher used by the two [`PaperCache`] instances.
	hasher: RandomState,
}

impl<K> S3FifoHybridCache<K>
where
	K: 'static + Eq + Hash + TypeSize + std::fmt::Debug + Clone + Send + Sync,
{
	/// Creates a new [`S3FifoHybridCache`] from the provided `config`.
	///
	/// # Errors
	///
	/// Returns [`CacheError::ZeroCacheSize`] if `total_size` is zero or if
	/// either derived tier size would be zero after applying `small_ratio`.
	pub fn new(config: HybridCacheConfig) -> Result<Self, CacheError> {
		if config.total_size == 0 {
			return Err(CacheError::ZeroCacheSize);
		}

		let ratio = config.small_ratio.clamp(0.0, 1.0);
		let small_size = ((ratio * config.total_size as f64) as u64).max(1);
		let main_size = config.total_size.saturating_sub(small_size).max(1);

		let small = PaperCache::<K, crate::BufferDRAM>::new(
			small_size,
			&[config.small_policy],
			config.small_policy,
		)?;

		let main = PaperCache::<K, crate::BufferDRAM>::new(
			main_size,
			&[config.main_policy],
			config.main_policy,
		)?;

		Ok(S3FifoHybridCache {
			small,
			main,
			access_counts: DashMap::new(),
			freq_threshold: config.freq_threshold,
			stats: Arc::new(AtomicHybridStats::new()),
			hasher: RandomState::default(),
		})
	}

	/// Inserts or updates a key-value pair.
	///
	/// - If the key already exists in the **main tier**, the value is updated
	///   there, preserving its hot-tier residence.
	/// - Otherwise the item enters the **small tier**.
	///
	/// # Errors
	///
	/// Propagates any [`CacheError`] returned by the underlying
	/// [`PaperCache::set`] call (e.g. [`CacheError::ExceedingValueSize`]).
	pub fn set(&self, key: K, value: &[u8], ttl: Option<u32>) -> Result<(), CacheError> {
		// Hot-tier update: preserve residence in main if already there.
		if self.main.has(&key) {
			return self.main.set(key, value, ttl);
		}
		// New / cold items always enter the small tier.
		self.small.set(key, value, ttl)
	}

	/// Retrieves the value associated with `key`.
	///
	/// Lookup order: **main tier** (hot) → **small tier** (new).
	///
	/// Each successful small-tier lookup increments the item's frequency
	/// counter; when the counter reaches `freq_threshold` the item is
	/// automatically **promoted** to the main tier.
	///
	/// # Errors
	///
	/// Returns [`CacheError::KeyNotFound`] when the key is absent from both
	/// tiers.
	pub fn get(&self, key: &K) -> Result<Vec<u8>, CacheError> {
		// Hot path: main tier.
		if let Ok(val) = self.main.get(key) {
			self.stats.main_hits.fetch_add(1, Ordering::Relaxed);
			return Ok(val);
		}

		// Cold path: small tier.
		match self.small.get(key) {
			Ok(val) => {
				self.stats.small_hits.fetch_add(1, Ordering::Relaxed);

				let hashed_key = self.hash_key(key);
				let freq = self.incr_access_count(hashed_key);

				if freq >= self.freq_threshold {
					self.promote(key, hashed_key, &val);
				}

				Ok(val)
			}
			Err(_) => {
				self.stats.misses.fetch_add(1, Ordering::Relaxed);
				Err(CacheError::KeyNotFound)
			}
		}
	}

	/// Removes the key from whichever tier contains it.
	///
	/// Cleans up the associated frequency counter.
	///
	/// # Errors
	///
	/// Returns [`CacheError::KeyNotFound`] if the key is present in neither
	/// tier.
	pub fn del(&self, key: &K) -> Result<(), CacheError> {
		let hashed_key = self.hash_key(key);
		self.access_counts.remove(&hashed_key);

		let in_main = self.main.del(key).is_ok();
		let in_small = self.small.del(key).is_ok();

		if in_main || in_small {
			Ok(())
		} else {
			Err(CacheError::KeyNotFound)
		}
	}

	/// Returns `true` if the key exists (and has not expired) in either tier.
	pub fn has(&self, key: &K) -> bool {
		self.main.has(key) || self.small.has(key)
	}

	/// Clears **both** tiers and resets all per-key frequency counters.
	///
	/// # Errors
	///
	/// Propagates any [`CacheError`] returned by the underlying wipe calls.
	pub fn wipe(&self) -> Result<(), CacheError> {
		self.access_counts.clear();
		self.small.wipe()?;
		self.main.wipe()?;
		Ok(())
	}

	/// Returns a point-in-time snapshot of the cache statistics.
	pub fn stats(&self) -> HybridCacheStats {
		self.stats.snapshot()
	}

	// ── private helpers ──────────────────────────────────────────────────────

	/// Increments and returns the access count for `hashed_key`.
	///
	/// Uses DashMap's entry API so the read-modify-write is performed while
	/// holding the shard lock, avoiding lost-update races.
	fn incr_access_count(&self, hashed_key: crate::HashedKey) -> u64 {
		let entry = self.access_counts
			.entry(hashed_key)
			.and_modify(|v| *v += 1)
			.or_insert(1);
		*entry
	}

	/// Promotes `key` from the small tier to the main tier.
	///
	/// On success the item is removed from the small tier and its frequency
	/// counter is cleared.  The promotion counter is incremented.
	///
	/// **TTL limitation**: the original TTL of the small-tier item cannot be
	/// retrieved from `PaperCache`'s public API, so promoted items are
	/// inserted into the main tier without a TTL (`None`).  This means a
	/// promoted item's expiry resets; callers that require TTL preservation
	/// should track expiry independently.
	///
	/// **Small-tier deletion**: if `small.del` returns `KeyNotFound` after a
	/// successful `main.set`, it means the item was concurrently evicted from
	/// the small tier by its own eviction policy.  This is benign: the item
	/// now lives exclusively in the main tier, which is the desired outcome.
	fn promote(&self, key: &K, hashed_key: crate::HashedKey, value: &[u8]) {
		if self.main.set(key.clone(), value, None).is_ok() {
			// KeyNotFound is expected when the item was naturally evicted from
			// the small tier between the get() call and this promotion attempt.
			let _ = self.small.del(key);
			self.access_counts.remove(&hashed_key);
			self.stats.promotions.fetch_add(1, Ordering::Relaxed);
		}
	}

	/// Computes the [`crate::HashedKey`] for `key` using the shared hasher.
	fn hash_key(&self, key: &K) -> crate::HashedKey {
		self.hasher.hash_one(key)
	}
}
