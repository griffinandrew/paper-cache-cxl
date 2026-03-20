/*
 * Copyright (c) Kia Shakiba
 *
 * This source code is licensed under the GNU AGPLv3 license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! Two-tier cache: small DRAM tier (S3-FIFO) backed by a far LRU tier.
//!
//! [`S3FifoHybridCache`] wraps two [`crate::PaperCache`] instances:
//!
//! - **Small tier** (DRAM, S3-FIFO, configurable fraction – default 10 %):
//!   receives **all** newly inserted items.  When the small tier evicts an
//!   item the PolicyWorker eviction callback automatically writes it to the
//!   main tier.
//!
//! - **Main tier** (LRU, far memory – default 90 %):
//!   acts as the backing store.  In production with nightly Rust +
//!   `key_value_pmem`, this tier uses `BufferPMEM` (persistent / CXL memory).
//!   On stable Rust it uses `BufferDRAM` to remain compilable without nightly
//!   feature flags.
//!
//!   A `get` that misses the small tier but hits the main tier returns the
//!   value immediately and schedules a background re-insertion into the small
//!   tier so that S3-FIFO's ghost queue can decide the admission tier on the
//!   next reference.
//!
//! # Data-flow
//!
//! ```text
//!  set(k, v)  ──▶  small tier (S3-FIFO, DRAM)
//!                       │  eviction
//!                       ▼  (PolicyWorker eviction callback)
//!                  main tier  (LRU, far/PMEM memory)
//!
//!  get(k) ──▶  small hit?  ──yes──▶  return value
//!                  │no
//!                  ▼
//!             main hit?   ──yes──▶  return value
//!                  │                + schedule background re-insert into small
//!                  │no
//!                  ▼
//!               miss
//! ```
//!
//! # Example
//!
//! ```
//! use paper_cache::hybridcache::{S3FifoHybridCache, HybridCacheConfig};
//!
//! let config = HybridCacheConfig::default();
//! let cache = S3FifoHybridCache::<u32>::new(config).unwrap();
//!
//! // Insert a value – it starts in the small DRAM tier.
//! cache.set(1u32, &[0u8; 128], None).unwrap();
//!
//! let val = cache.get(&1u32).unwrap();
//! assert_eq!(val.len(), 128);
//! ```

use std::{
hash::Hash,
sync::{
Arc,
atomic::{AtomicU64, Ordering},
},
thread,
};

use crossbeam_channel::{Sender, unbounded};
use typesize::TypeSize;

use crate::{PaperCache, PaperPolicy, CacheError, BufferDRAM};

// ── Configuration ─────────────────────────────────────────────────────────────

/// Configuration for [`S3FifoHybridCache`].
///
/// Use [`HybridCacheConfig::default`] to obtain sensible defaults (10 MB
/// total, 10 % small DRAM tier, S3-FIFO small policy, LRU main policy).
#[derive(Debug, Clone)]
pub struct HybridCacheConfig {
/// Total cache capacity in bytes.
pub total_size: u64,

/// Fraction of [`total_size`](Self::total_size) reserved for the small
/// DRAM tier.  Must be in `0.0..=1.0`; clamped silently otherwise.
///
/// The main tier receives the remainder `(1.0 - small_ratio) * total_size`.
/// Defaults to `0.1` (10 %).
pub small_ratio: f64,

/// Eviction policy used by the small DRAM tier.
///
/// Defaults to [`PaperPolicy::SThreeFifo`]`(0.1)`, implementing the full
/// S3-FIFO policy with its internal small-queue ratio at 10 %.
pub small_policy: PaperPolicy,

/// Eviction policy used by the main (far) tier.
///
/// Defaults to [`PaperPolicy::Lru`].
pub main_policy: PaperPolicy,
}

impl Default for HybridCacheConfig {
fn default() -> Self {
HybridCacheConfig {
total_size: 10_000_000, // 10 MB
small_ratio: 0.1,
small_policy: PaperPolicy::SThreeFifo(0.1),
main_policy: PaperPolicy::Lru,
}
}
}

// ── Statistics ────────────────────────────────────────────────────────────────

/// A point-in-time snapshot of [`S3FifoHybridCache`] runtime statistics.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct HybridCacheStats {
/// Hits served from the **small** DRAM tier.
pub small_hits: u64,
/// Hits served from the **main** (far) tier.
pub main_hits: u64,
/// Lookups that found the key in neither tier.
pub misses: u64,
/// Items re-inserted into the small tier after a main-tier hit.
pub promotions: u64,
}

struct AtomicHybridStats {
small_hits: AtomicU64,
main_hits: AtomicU64,
misses: AtomicU64,
promotions: AtomicU64,
}

impl AtomicHybridStats {
fn new() -> Self {
AtomicHybridStats {
small_hits: AtomicU64::new(0),
main_hits: AtomicU64::new(0),
misses: AtomicU64::new(0),
promotions: AtomicU64::new(0),
}
}

fn snapshot(&self) -> HybridCacheStats {
HybridCacheStats {
small_hits: self.small_hits.load(Ordering::Relaxed),
main_hits: self.main_hits.load(Ordering::Relaxed),
misses: self.misses.load(Ordering::Relaxed),
promotions: self.promotions.load(Ordering::Relaxed),
}
}
}

// ── S3FifoHybridCache ─────────────────────────────────────────────────────────

/// A two-tier cache: small DRAM tier (S3-FIFO) backed by a far LRU tier.
///
/// See the [module documentation](self) for full architecture details.
///
/// # Type parameter
///
/// `K` is the key type.  It must satisfy:
/// `'static + Eq + Hash + TypeSize + Debug + Clone + Send + Sync`.
///
/// Values are stored as raw byte slices (`&[u8]` on write, `Vec<u8>` on read).
pub struct S3FifoHybridCache<K>
where
K: 'static + Eq + Hash + TypeSize + std::fmt::Debug + Clone + Send + Sync,
{
/// Small DRAM tier (S3-FIFO) – all new insertions land here.
small: Arc<PaperCache<K, BufferDRAM>>,
/// Main far-memory tier (LRU) – backing store populated by small evictions.
///
/// Note: in production with nightly + `key_value_pmem` this would be
/// `PaperCache<K, BufferPMEM>` (CXL / persistent memory).
main: Arc<PaperCache<K, BufferDRAM>>,
/// Channel used to request re-insertion of a main-tier hit into the small
/// tier.  The background reinsertion worker reads from the other end.
reinsertion_tx: Sender<(K, Vec<u8>)>,
/// Shared atomic counters for [`HybridCacheStats`].
stats: Arc<AtomicHybridStats>,
}

impl<K> S3FifoHybridCache<K>
where
K: 'static + Eq + Hash + TypeSize + std::fmt::Debug + Clone + Send + Sync,
{
/// Creates a new [`S3FifoHybridCache`] from the provided `config`.
///
/// Spawns a background worker thread that re-inserts main-tier hits back
/// into the small tier.
///
/// # Errors
///
/// Returns [`CacheError::ZeroCacheSize`] if `total_size` is zero.
pub fn new(config: HybridCacheConfig) -> Result<Self, CacheError> {
if config.total_size == 0 {
return Err(CacheError::ZeroCacheSize);
}

let ratio = config.small_ratio.clamp(0.0, 1.0);
let small_size = ((ratio * config.total_size as f64) as u64).max(1);
let main_size = config.total_size.saturating_sub(small_size).max(1);

// ── Main (far) tier – LRU ────────────────────────────────────────────
// Receives items evicted from the small tier via the eviction callback.
let main = Arc::new(PaperCache::<K, BufferDRAM>::new(
main_size,
&[config.main_policy],
config.main_policy,
)?);

// ── Eviction callback: small eviction → write to main ────────────────
// Called synchronously from the PolicyWorker background thread when
// the small tier evicts an item.  Writes the evicted item to the main
// tier so no data is lost.
let main_evict = Arc::clone(&main);
let eviction_callback: Box<dyn for<'a> Fn(crate::HashedKey, Arc<BufferDRAM>, &'a K) + Send + Sync> =
Box::new(move |_, val, k| {
let _ = main_evict.set(k.clone(), &**val, None);
});

// ── Small DRAM tier – S3-FIFO, with eviction callback ────────────────
let small = Arc::new(PaperCache::<K, BufferDRAM>::new_with_eviction_callback(
small_size,
&[config.small_policy],
config.small_policy,
eviction_callback,
)?);

let stats = Arc::new(AtomicHybridStats::new());

// ── Reinsertion worker: main-tier hit → re-insert into small ─────────
// When `get` finds an item in the main tier, it sends (key, value) here.
// The worker re-inserts the item into the small tier so that S3-FIFO's
// ghost queue can control its admission tier on the next reference:
//   – ghost hit  → item enters S3-FIFO's M (main) queue
//   – ghost miss → item enters S3-FIFO's S (small) queue
let (reinsertion_tx, reinsertion_rx) = unbounded::<(K, Vec<u8>)>();
let small_reinsert = Arc::clone(&small);
let stats_reinsert = Arc::clone(&stats);
thread::spawn(move || {
while let Ok((k, val)) = reinsertion_rx.recv() {
let _ = small_reinsert.set(k, &val, None);
stats_reinsert.promotions.fetch_add(1, Ordering::Relaxed);
}
});

Ok(S3FifoHybridCache {
small,
main,
reinsertion_tx,
stats,
})
}

/// Inserts or updates a key-value pair in the **small DRAM tier** only.
///
/// All writes land in the small tier.  When the small tier fills up and
/// evicts an item, the PolicyWorker eviction callback automatically writes
/// it to the main tier.
///
/// # Errors
///
/// Propagates any [`CacheError`] returned by the underlying
/// [`PaperCache::set`] call.
pub fn set(&self, key: K, value: &[u8], ttl: Option<u32>) -> Result<(), CacheError> {
self.small.set(key, value, ttl)
}

/// Retrieves the value associated with `key`.
///
/// Lookup order: **small DRAM tier** (fast path) → **main (far) tier**
/// (slow path).
///
/// A main-tier hit schedules a background re-insertion into the small tier
/// so that S3-FIFO's ghost queue drives the admission tier on the next
/// reference.
///
/// # Errors
///
/// Returns [`CacheError::KeyNotFound`] when the key is absent from both
/// tiers.
pub fn get(&self, key: &K) -> Result<Vec<u8>, CacheError> {
// Fast path: small DRAM tier.
if let Ok(val) = self.small.get(key) {
self.stats.small_hits.fetch_add(1, Ordering::Relaxed);
return Ok(val);
}

// Slow path: main (far) tier.
match self.main.get(key) {
Ok(val) => {
self.stats.main_hits.fetch_add(1, Ordering::Relaxed);
// Schedule background re-insertion into the small tier.
let _ = self.reinsertion_tx.send((key.clone(), val.clone()));
Ok(val)
}
Err(_) => {
self.stats.misses.fetch_add(1, Ordering::Relaxed);
Err(CacheError::KeyNotFound)
}
}
}

/// Removes the key from whichever tier(s) contain it.
///
/// # Errors
///
/// Returns [`CacheError::KeyNotFound`] if the key is absent from both tiers.
pub fn del(&self, key: &K) -> Result<(), CacheError> {
let in_small = self.small.del(key).is_ok();
let in_main = self.main.del(key).is_ok();

if in_small || in_main {
Ok(())
} else {
Err(CacheError::KeyNotFound)
}
}

/// Returns `true` if the key exists (and has not expired) in either tier.
pub fn has(&self, key: &K) -> bool {
self.small.has(key) || self.main.has(key)
}

/// Clears **both** tiers.
///
/// # Errors
///
/// Propagates any [`CacheError`] returned by the underlying wipe calls.
pub fn wipe(&self) -> Result<(), CacheError> {
self.small.wipe()?;
self.main.wipe()?;
Ok(())
}

/// Returns a point-in-time snapshot of the cache statistics.
pub fn stats(&self) -> HybridCacheStats {
self.stats.snapshot()
}
}
