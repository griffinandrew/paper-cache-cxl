/*
 * Copyright (c) Kia Shakiba
 *
 * This source code is licensed under the GNU AGPLv3 license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! Two-tier cache: small DRAM tier (S3-FIFO) backed by a far PMEM tier (LRU).
//!
//! [`S3FifoHybridCache`] wraps two [`crate::PaperCache`] instances:
//!
//! - **Small tier** (DRAM, S3-FIFO, configurable size – default 1 MB):
//!   backed by [`crate::BufferDRAM`].  Receives **all** newly inserted items.
//!   When the small tier evicts an item the PolicyWorker eviction callback
//!   automatically writes it to the far PMEM tier.
//!
//! - **Far / main tier** (PMEM, LRU, configurable size – default 9 MB):
//!   backed by [`crate::BufferPMEM`] (Hybrid allocator → persistent / CXL
//!   memory).  Acts as the backing store populated exclusively by evictions
//!   from the small tier.
//!
//!   A `get` that misses the small tier but hits the far tier returns the
//!   value immediately and schedules a background re-insertion into the small
//!   tier so that S3-FIFO's ghost queue can decide the admission tier on the
//!   next reference.
//!
//! # Memory layout
//!
//! | Tier  | Buffer     | Allocator       | Location   |
//! |-------|-----------|-----------------|------------|
//! | Small | BufferDRAM | jemalloc        | DRAM       |
//! | Far   | BufferPMEM | Hybrid (UMF)    | PMEM / CXL |
//!
//! # Data-flow
//!
//! ```text
//!  set(k, v)  ──▶  small tier (S3-FIFO, DRAM / BufferDRAM)
//!                       │  eviction
//!                       ▼  (PolicyWorker eviction callback)
//!                  far tier  (LRU, PMEM / BufferPMEM)
//!
//!  get(k) ──▶  small hit?  ──yes──▶  return value
//!                  │no
//!                  ▼
//!             far hit?    ──yes──▶  return value
//!                  │                + schedule background re-insert into small
//!                  │no
//!                  ▼
//!               miss
//! ```
//!
//! # Requirements
//!
//! This module requires nightly Rust because the PMEM far tier uses the
//! `allocator_api` nightly feature (via `key_pmem_value_pmem` +
//! `eviction_stacks_pmem`).
//!
//! # Example
//!
//! ```ignore
//! use paper_cache::hybridcache::{S3FifoHybridCache, HybridCacheConfig, CacheTierSize};
//!
//! let config = HybridCacheConfig {
//!     small_size: CacheTierSize::Mb(1),
//!     main_size: CacheTierSize::Gb(2),
//!     ..Default::default()
//! };
//! let cache = S3FifoHybridCache::<u32>::new(config).unwrap();
//!
//! // Insert a string value – it starts in the small DRAM tier.
//! cache.set(1u32, "hello world").unwrap();
//!
//! let val = cache.get(&1u32).unwrap();
//! assert_eq!(val, "hello world");
//!
//! // Insert with an explicit TTL of 60 seconds.
//! cache.set_with_ttl(2u32, "expires soon", 60).unwrap();
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

use crate::{PaperCache, PaperPolicy, CacheError, BufferDRAM, BufferPMEM};

// ── Tier Size ─────────────────────────────────────────────────────────────────

/// A size specification for a cache tier, in bytes, megabytes, or gigabytes.
///
/// Used in [`HybridCacheConfig`] to set the capacity of each tier independently.
///
/// # Examples
///
/// ```ignore
/// use paper_cache::hybridcache::{HybridCacheConfig, CacheTierSize};
///
/// let config = HybridCacheConfig {
///     small_size: CacheTierSize::Mb(1),
///     main_size: CacheTierSize::Gb(2),
///     ..Default::default()
/// };
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CacheTierSize {
    /// Exact capacity in bytes.
    Bytes(u64),
    /// Capacity in decimal megabytes (1 MB = 1,000,000 bytes, SI standard).
    Mb(u64),
    /// Capacity in decimal gigabytes (1 GB = 1,000,000,000 bytes, SI standard).
    Gb(u64),
}

impl CacheTierSize {
    /// Returns the size converted to bytes.
    pub fn to_bytes(self) -> u64 {
        match self {
            CacheTierSize::Bytes(b) => b,
            CacheTierSize::Mb(mb) => mb * 1_000_000,
            CacheTierSize::Gb(gb) => gb * 1_000_000_000,
        }
    }
}

// ── Configuration ─────────────────────────────────────────────────────────────

/// Configuration for [`S3FifoHybridCache`].
///
/// Use [`HybridCacheConfig::default`] to obtain sensible defaults (1 MB small
/// DRAM tier, 9 MB far PMEM tier, S3-FIFO small policy, LRU far policy).
///
/// Each tier's capacity is specified independently via [`CacheTierSize`],
/// accepting bytes, megabytes, or gigabytes.
///
/// # Examples
///
/// ```ignore
/// use paper_cache::hybridcache::{HybridCacheConfig, CacheTierSize};
///
/// let config = HybridCacheConfig {
///     small_size: CacheTierSize::Mb(2),
///     main_size: CacheTierSize::Gb(1),
///     ..Default::default()
/// };
/// ```
#[derive(Debug, Clone)]
pub struct HybridCacheConfig {
    /// Capacity of the small (DRAM) tier.
    ///
    /// Accepts [`CacheTierSize::Bytes`], [`CacheTierSize::Mb`], or
    /// [`CacheTierSize::Gb`].  Defaults to 1 MB.
    pub small_size: CacheTierSize,

    /// Capacity of the far (PMEM) tier.
    ///
    /// Accepts [`CacheTierSize::Bytes`], [`CacheTierSize::Mb`], or
    /// [`CacheTierSize::Gb`].  Defaults to 9 MB.
    pub main_size: CacheTierSize,

    /// Eviction policy used by the small DRAM tier.
    ///
    /// Defaults to [`PaperPolicy::SThreeFifo`]`(0.1)`, implementing the full
    /// S3-FIFO policy with its internal small-queue ratio at 10 %.
    pub small_policy: PaperPolicy,

    /// Eviction policy used by the far PMEM tier.
    ///
    /// Defaults to [`PaperPolicy::Lru`].
    pub main_policy: PaperPolicy,
}

impl Default for HybridCacheConfig {
    fn default() -> Self {
        HybridCacheConfig {
            small_size: CacheTierSize::Mb(1), // 1 MB DRAM
            main_size: CacheTierSize::Mb(9),  // 9 MB PMEM
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
/// Hits served from the **far** PMEM tier.
pub main_hits: u64,
/// Lookups that found the key in neither tier.
pub misses: u64,
/// Items re-inserted into the small DRAM tier after a far-tier hit.
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

/// A two-tier cache: small DRAM tier (S3-FIFO) backed by a far PMEM tier (LRU).
///
/// See the [module documentation](self) for full architecture details.
///
/// # Type parameter
///
/// `K` is the key type.  It must satisfy:
/// `'static + Eq + Hash + TypeSize + Debug + Clone + Send + Sync`.
///
/// Values are stored and retrieved as UTF-8 strings (`&str` on write,
/// `String` on read).
pub struct S3FifoHybridCache<K>
where
K: 'static + Eq + Hash + TypeSize + std::fmt::Debug + Clone + Send + Sync,
{
/// Small DRAM tier (S3-FIFO, `BufferDRAM`) – all new insertions land here.
small: Arc<PaperCache<K, BufferDRAM>>,
/// Far PMEM tier (LRU, `BufferPMEM`) – backing store populated by small
/// evictions.  Uses the Hybrid allocator to place values in persistent /
/// CXL memory.
main: Arc<PaperCache<K, BufferPMEM>>,
/// Channel used to request re-insertion of a far-tier hit into the small
/// DRAM tier.  The background reinsertion worker reads from the other end.
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
/// Spawns a background worker thread that re-inserts far-tier hits back
/// into the small DRAM tier.
///
/// # Errors
///
/// Returns [`CacheError::ZeroCacheSize`] if both `small_size` and
/// `main_size` resolve to zero bytes.
pub fn new(config: HybridCacheConfig) -> Result<Self, CacheError> {
let small_bytes = config.small_size.to_bytes();
let main_bytes = config.main_size.to_bytes();

if small_bytes == 0 && main_bytes == 0 {
return Err(CacheError::ZeroCacheSize);
}

let small_size = small_bytes.max(1);
let main_size = main_bytes.max(1);

// ── Far PMEM tier – LRU, BufferPMEM ──────────────────────────────────
// Receives items evicted from the small DRAM tier via the eviction
// callback.  Values are allocated via the Hybrid (UMF) allocator into
// persistent / CXL memory.
let main = Arc::new(PaperCache::<K, BufferPMEM>::new(
main_size,
&[config.main_policy],
config.main_policy,
)?);

// ── Eviction callback: small DRAM eviction → write to far PMEM ───────
// Called synchronously from the PolicyWorker background thread when
// the small tier evicts an item.  Writes the evicted DRAM item to the
// far PMEM tier using the PMEM PaperCache's `set` method, which
// allocates the value via the Hybrid allocator into PMEM.
let main_evict = Arc::clone(&main);
let eviction_callback: Box<dyn for<'a> Fn(crate::HashedKey, Arc<BufferDRAM>, &'a K) + Send + Sync> =
Box::new(move |_, val, k| {
// `val` is Arc<BufferDRAM> = Arc<Box<[u8]>>.
// `&**val` gives &[u8], which PMEM set() allocates via Hybrid → PMEM.
let _ = main_evict.set(k.clone(), &**val, None);
});

// ── Small DRAM tier – S3-FIFO, BufferDRAM, with eviction callback ─────
let small = Arc::new(PaperCache::<K, BufferDRAM>::new_with_eviction_callback(
small_size,
&[config.small_policy],
config.small_policy,
eviction_callback,
)?);

let stats = Arc::new(AtomicHybridStats::new());

// ── Reinsertion worker: far PMEM hit → re-insert into small DRAM ─────
// When `get` finds an item in the far PMEM tier, it sends (key, value)
// here.  The worker re-inserts the item into the small DRAM tier so
// that S3-FIFO's ghost queue can control its admission tier on the next
// reference:
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
/// All writes land in the small S3-FIFO DRAM tier.  When the small tier
/// fills up and evicts an item, the PolicyWorker eviction callback
/// automatically writes it to the far PMEM tier via
/// `PaperCache<K, BufferPMEM>::set()`.
///
/// No TTL is applied.  Use [`set_with_ttl`](Self::set_with_ttl) when an
/// expiry time is required.
///
/// # Errors
///
/// Propagates any [`CacheError`] returned by the underlying
/// [`PaperCache::set`] call.
pub fn set(&self, key: K, value: &str) -> Result<(), CacheError> {
self.small.set(key, value.as_bytes(), None)
}

/// Inserts or updates a key-value pair in the **small DRAM tier** with an
/// explicit TTL.
///
/// Behaves identically to [`set`](Self::set) except that the entry expires
/// after `ttl` seconds.
///
/// # Errors
///
/// Propagates any [`CacheError`] returned by the underlying
/// [`PaperCache::set`] call.
pub fn set_with_ttl(&self, key: K, value: &str, ttl: u32) -> Result<(), CacheError> {
self.small.set(key, value.as_bytes(), Some(ttl))
}

/// Retrieves the value associated with `key`.
///
/// Lookup order: **small DRAM tier** (fast path) → **far PMEM tier**
/// (slow path).
///
/// A far-tier hit schedules a background re-insertion into the small DRAM
/// tier so that S3-FIFO's ghost queue drives the admission tier on the next
/// reference.
///
/// # UTF-8 encoding
///
/// Since [`set`](Self::set) and [`set_with_ttl`](Self::set_with_ttl) accept
/// only `&str`, stored bytes are always valid UTF-8.  Any invalid byte
/// sequences encountered during retrieval are replaced with the Unicode
/// replacement character (`�`).
///
/// # Errors
///
/// Returns [`CacheError::KeyNotFound`] when the key is absent from both
/// tiers.
pub fn get(&self, key: &K) -> Result<String, CacheError> {
// Fast path: small DRAM tier.
if let Ok(val) = self.small.get(key) {
self.stats.small_hits.fetch_add(1, Ordering::Relaxed);
return Ok(String::from_utf8_lossy(&val).into_owned());
}

// Slow path: far PMEM tier.
match self.main.get(key) {
Ok(val) => {
self.stats.main_hits.fetch_add(1, Ordering::Relaxed);
#[cfg(debug_assertions)]println!("Far tier hit for key {:?} (scheduling reinsertion into small tier)", key);
// Schedule background re-insertion into the small DRAM tier.
let _ = self.reinsertion_tx.send((key.clone(), val.clone()));
Ok(String::from_utf8_lossy(&val).into_owned())
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

/// Returns `true` if `key` is currently in the **small DRAM tier**.
///
/// Useful for testing and diagnostics to confirm that a key has been
/// admitted (or re-admitted) to the DRAM tier.
pub fn has_in_dram(&self, key: &K) -> bool {
self.small.has(key)
}

/// Returns `true` if `key` is currently in the **far PMEM tier**.
///
/// Because the hybrid cache uses copy-on-read semantics — a far-tier hit
/// schedules an asynchronous re-insertion into the DRAM tier without
/// removing the PMEM copy — a key can be present in both tiers
/// simultaneously.
pub fn has_in_pmem(&self, key: &K) -> bool {
self.main.has(key)
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
