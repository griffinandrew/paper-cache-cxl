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
//!                       │  eviction (PolicyWorker eviction callback)
//!                       ▼  try_send on bounded demotion channel (non-blocking)
//!                  demotion worker thread  (pre-started, already in recv)
//!                       │  PMEM write + insert key into demoted_keys
//!                       ▼
//!                  far tier  (LRU, PMEM / BufferPMEM)
//!
//!  get(k) ──▶  small hit?   ──yes──▶  return value
//!                  │no
//!                  ▼
//!             far hit?      ──yes──▶  return value
//!                  │                + if k in demoted_keys (ghost hit):
//!                  │                    remove from demoted_keys
//!                  │                    try_send re-insert into small (bounded)
//!                  │                + else (ghost miss): serve from PMEM only
//!                  │no
//!                  ▼
//!             in-flight?    ──yes──▶  return miss (caller retries)
//!                  │no
//!                  ▼
//!               miss
//! ```
//!
//! # Requirements
//!
//! This module requires nightly Rust because the PMEM far tier uses the
//! `allocator_api` nightly feature (via `key_pmem_value_pmem`).
//!
//! # Segfault avoidance
//!
//! Only far-tier **object bytes** (key + value) are placed in PMEM via the
//! Hybrid allocator.  Eviction stacks and hashtables remain in DRAM.
//! Enable `far_tier_pmem_evst_hash` to also move those structures to PMEM
//! (useful for CXL metadata profiling, but requires stable
//! `eviction_stacks_pmem` support).
//!
//! # Copy-on-read and the in-flight demotion window
//!
//! Promotions (far-tier hit → re-insert into small) use **copy-on-read**:
//! the PMEM copy is never deleted.  This means that when the DRAM tier
//! evicts a previously-promoted item, the eviction callback finds the key
//! already in the far tier and skips the write entirely — no in-flight
//! window and no race.  The in-flight set only matters for items being
//! demoted to PMEM for the **first time**.
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
    Arc, Barrier,
    atomic::{AtomicU64, Ordering},
},
thread,
};

use crossbeam_channel::{Sender, bounded};
use dashmap::DashSet;
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

    /// Capacity of the bounded demotion channel (DRAM→PMEM migration queue).
    ///
    /// The eviction callback enqueues `(key, value_bytes)` onto this channel
    /// without blocking.  A dedicated demotion worker thread drains it and
    /// writes each item to the far PMEM tier.  If the channel is full, the
    /// demotion is silently dropped (the item becomes a miss; acceptable).
    ///
    /// Defaults to `512`.
    pub demotion_channel_capacity: usize,

    /// Capacity of the bounded reinsertion channel (PMEM→DRAM promotion queue).
    ///
    /// A far-tier hit enqueues `(key, value_bytes)` here with a non-blocking
    /// send.  A background worker re-inserts the item into the small DRAM tier.
    /// If the channel is full, the promotion is skipped; the next access will
    /// re-trigger it.
    ///
    /// Defaults to `2048`.
    pub reinsertion_channel_capacity: usize,
}

impl Default for HybridCacheConfig {
    fn default() -> Self {
        HybridCacheConfig {
            small_size: CacheTierSize::Mb(1), // 1 MB DRAM
            main_size: CacheTierSize::Mb(9),  // 9 MB PMEM
            small_policy: PaperPolicy::SThreeFifo(0.1),
            main_policy: PaperPolicy::Lru,
            demotion_channel_capacity: 512,
            reinsertion_channel_capacity: 2048,
        }
    }
}

// ── Statistics ────────────────────────────────────────────────────────────────

/// A point-in-time snapshot of [`S3FifoHybridCache`] runtime statistics.
///
/// `dram_items` and `pmem_items` are live counts queried from each tier's
/// internal status at the moment [`S3FifoHybridCache::stats`] is called.
/// All other fields are cumulative counters that only ever increase.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct HybridCacheStats {
    /// Hits served from the **small** DRAM tier.
    pub small_hits: u64,
    /// Hits served from the **far** PMEM tier.
    pub main_hits: u64,
    /// Lookups that found the key in neither tier (and not in-flight).
    pub misses: u64,
    /// Items re-inserted into the small DRAM tier after a far-tier hit.
    ///
    /// Every far-tier hit attempts a re-insertion so that S3-FIFO's ghost
    /// queue can decide the admission tier on the next reference.  The
    /// reinsertion channel is bounded; promotions are dropped when it is full
    /// (counted in `dropped_promotions`).
    pub promotions: u64,
    /// Items evicted from the small DRAM tier and written to the far PMEM tier.
    pub demotions: u64,
    /// Current number of items in the small DRAM tier (live snapshot).
    pub dram_items: u64,
    /// Current number of items in the far PMEM tier (live snapshot).
    pub pmem_items: u64,
    /// Demotions dropped because the demotion channel was full.
    ///
    /// A dropped demotion means the evicted item is lost (a miss on next access).
    /// This is acceptable; tune `demotion_channel_capacity` to reduce drops.
    pub dropped_demotions: u64,
    /// Promotions dropped because the reinsertion channel was full.
    ///
    /// A dropped promotion means the PMEM item is not re-inserted into DRAM.
    /// The next access will re-trigger the promotion attempt.
    pub dropped_promotions: u64,
}

struct AtomicHybridStats {
    small_hits: AtomicU64,
    main_hits: AtomicU64,
    misses: AtomicU64,
    promotions: AtomicU64,
    demotions: AtomicU64,
    dropped_demotions: AtomicU64,
    dropped_promotions: AtomicU64,
}

impl AtomicHybridStats {
    fn new() -> Self {
        AtomicHybridStats {
            small_hits: AtomicU64::new(0),
            main_hits: AtomicU64::new(0),
            misses: AtomicU64::new(0),
            promotions: AtomicU64::new(0),
            demotions: AtomicU64::new(0),
            dropped_demotions: AtomicU64::new(0),
            dropped_promotions: AtomicU64::new(0),
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
    /// Bounded channel used to request re-insertion of a far-tier hit into
    /// the small DRAM tier.  The background reinsertion worker reads from the
    /// other end.
    ///
    /// The channel capacity is configured via
    /// [`HybridCacheConfig::reinsertion_channel_capacity`] (default 2048).
    /// When the channel is full, the promotion is silently dropped
    /// (`dropped_promotions` counter is incremented) and the next access will
    /// re-trigger it.
    reinsertion_tx: Sender<(K, Vec<u8>)>,
    /// Keys currently being migrated from the small DRAM tier to the far PMEM
    /// tier by the demotion worker thread.  During this window the key is
    /// absent from both tier hashtables; `get` treats an in-flight key as a
    /// cache miss rather than spinning, keeping latency predictable.
    in_flight_demotions: Arc<DashSet<K>>,
    /// Keys that have been successfully written to the far PMEM tier and are
    /// therefore eligible for promotion back to DRAM on the next far-tier hit.
    ///
    /// When a PMEM hit occurs, `get` checks this set.  If the key is present
    /// it is removed and a re-insertion into the small DRAM tier is enqueued
    /// (ghost-hit path).  If the key is absent (it was written to PMEM in a
    /// previous cycle and has already been promoted once, or was never tracked),
    /// no promotion is triggered — the value is served directly from PMEM.
    ///
    /// This prevents unbounded copy-on-read churn: items are only pulled back
    /// into DRAM when they were recently evicted from it (the proxy for an
    /// S3-FIFO ghost-queue hit), not on every subsequent PMEM access.
    demoted_keys: Arc<DashSet<K>>,
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

    let stats = Arc::new(AtomicHybridStats::new());
    let in_flight_demotions = Arc::new(DashSet::<K>::new());
    // Tracks keys that have been written to the far PMEM tier and are eligible
    // for one promotion back to DRAM on the next far-tier hit.
    let demoted_keys = Arc::new(DashSet::<K>::new());

    // ── Demotion channel + dedicated worker ──────────────────────────────────
    // The eviction callback (fired on the PolicyWorker thread) enqueues
    // (key, value_bytes) onto this bounded channel without blocking.  A
    // dedicated demotion worker thread drains the channel and performs the
    // actual PMEM write, keeping the PolicyWorker eviction loop non-blocking.
    //
    // Capacity is configurable; if the channel is full the demotion is dropped
    // (the item becomes a miss on next access — acceptable).
    let (demotion_tx, demotion_rx) = bounded::<(K, Vec<u8>)>(config.demotion_channel_capacity);

    // ── Demotion worker: drains channel → writes to far PMEM tier ─────────
    // The worker thread is started and confirmed running (via a Barrier)
    // BEFORE `small` is created, so it is already blocking on `recv` when
    // the PolicyWorker fires its first eviction.  This eliminates the
    // "thread not yet scheduled" race that can cause PMEM writes to be
    // delayed beyond the eviction window.
    //
    // After each successful write the key is added to `demoted_keys` so that
    // the next far-tier hit triggers a promotion (ghost-hit proxy).
    let startup_barrier = Arc::new(Barrier::new(2));
    let startup_barrier_clone = Arc::clone(&startup_barrier);
    let main_worker = Arc::clone(&main);
    let stats_worker = Arc::clone(&stats);
    let in_flight_worker = Arc::clone(&in_flight_demotions);
    let demoted_worker = Arc::clone(&demoted_keys);
    thread::Builder::new()
        .name("hybridcache-demotion".to_string())
        .spawn(move || {
            // Signal the parent thread that this worker is running.
            startup_barrier_clone.wait();
            while let Ok((k, val)) = demotion_rx.recv() {
                let _ = main_worker.set(k.clone(), &val, None);
                in_flight_worker.remove(&k);
                demoted_worker.insert(k);
                stats_worker.demotions.fetch_add(1, Ordering::Relaxed);
            }
        })
        .expect("failed to spawn demotion worker thread");
    // Block until the demotion worker is confirmed running and in recv().
    startup_barrier.wait();

    // ── Eviction callback: non-blocking enqueue onto demotion channel ─────
    // Fired by PolicyWorker when the small tier evicts an item.  Only
    // enqueues; no synchronous PMEM I/O on the PolicyWorker thread.
    //
    // If the key is already in the far tier (copy-on-read: the item was
    // promoted and re-evicted), skip entirely — the PMEM copy is still valid.
    //
    // The key is added to `in_flight_demotions` before the enqueue and
    // removed by the demotion worker after the PMEM write completes, so that
    // concurrent `get` calls can detect the migration window.
    let main_check = Arc::clone(&main);
    let stats_evict = Arc::clone(&stats);
    let in_flight_evict = Arc::clone(&in_flight_demotions);
    let eviction_callback: Box<dyn for<'a> Fn(crate::HashedKey, Arc<BufferDRAM>, &'a K) + Send + Sync> =
    Box::new(move |_, val, k| {
        // Skip if this key already has a live entry in the far PMEM tier.
        if main_check.has(k) {
            return;
        }
        // Mark the key as in-flight before enqueuing so `get` detects the
        // migration window.
        in_flight_evict.insert(k.clone());
        // Non-blocking send.  If the channel is full, drop the demotion,
        // remove the in-flight marker, and count the drop.
        if demotion_tx.try_send((k.clone(), (**val).to_vec())).is_err() {
            in_flight_evict.remove(k);
            stats_evict.dropped_demotions.fetch_add(1, Ordering::Relaxed);
        }
    });

    // ── Small DRAM tier – S3-FIFO, BufferDRAM, with eviction callback ─────
    let small = Arc::new(PaperCache::<K, BufferDRAM>::new_with_eviction_callback(
        small_size,
        &[config.small_policy],
        config.small_policy,
        eviction_callback,
    )?);

    // ── Reinsertion worker: far PMEM hit → re-insert into small DRAM ─────
    // When `get` finds an item in the far PMEM tier, it tries to send
    // (key, value) here.  The worker re-inserts the item into the small DRAM
    // tier so that S3-FIFO's ghost queue can control its admission tier on the
    // next reference:
    //   – ghost hit  → item enters S3-FIFO's M (main) queue (key was recently
    //                   evicted and is in demoted_keys)
    //   – ghost miss → value served from PMEM only; no reinsertion enqueued
    //
    // The channel is bounded; if it is full the promotion is dropped
    // (counted in dropped_promotions).  The next access will re-trigger it.
    let (reinsertion_tx, reinsertion_rx) = bounded::<(K, Vec<u8>)>(config.reinsertion_channel_capacity);
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
        in_flight_demotions,
        demoted_keys,
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
/// A far-tier hit only schedules a background re-insertion into the small
/// DRAM tier when the key is present in the internal `demoted_keys` set,
/// meaning it was recently evicted from DRAM to PMEM (the proxy for an
/// S3-FIFO ghost-queue hit).  If the key is not in `demoted_keys`, the
/// value is served directly from PMEM without a promotion, preventing
/// unbounded copy-on-read churn.
///
/// If the key is absent from both tiers but is currently being migrated from
/// DRAM to PMEM, this method returns [`CacheError::KeyNotFound`] rather than
/// spinning.  The caller can retry; with async demotion the window is brief.
///
/// # UTF-8 encoding
///
/// Since [`set`](Self::set) and [`set_with_ttl`](Self::set_with_ttl) accept
/// only `&str`, stored bytes are always valid UTF-8.  Any invalid byte
/// sequences encountered during retrieval are replaced with the Unicode
/// replacement character (`\u{FFFD}`).
///
/// # Errors
///
/// Returns [`CacheError::KeyNotFound`] when the key is absent from both
/// tiers (or is in-flight).
pub fn get(&self, key: &K) -> Result<String, CacheError> {
    // Fast path: small DRAM tier.
    if let Ok(val) = self.small.get(key) {
        #[cfg(debug_assertions)]println!("Small tier hit for key {:?}", key);
        self.stats.small_hits.fetch_add(1, Ordering::Relaxed);
        return Ok(String::from_utf8_lossy(&val).into_owned());
    }

    // Slow path: far PMEM tier.
    match self.main.get(key) {
        Ok(val) => {
            self.stats.main_hits.fetch_add(1, Ordering::Relaxed);
            // Only promote to DRAM if the key is in `demoted_keys` — i.e. it
            // was recently evicted from DRAM and this access is a ghost-queue
            // hit proxy.  Remove it from the set atomically to ensure each
            // demotion cycle grants exactly one promotion attempt.
            if self.demoted_keys.remove(key).is_some() {
                #[cfg(debug_assertions)]println!("Far tier hit for key {:?} (ghost hit — scheduling reinsertion)", key);
                if self.reinsertion_tx.try_send((key.clone(), val.clone())).is_err() {
                    self.stats.dropped_promotions.fetch_add(1, Ordering::Relaxed);
                }
            } else {
                #[cfg(debug_assertions)]println!("Far tier hit for key {:?} (ghost miss — serving from PMEM only)", key);
            }
            Ok(String::from_utf8_lossy(&val).into_owned())
        }
        Err(_) => {
            // If the key is currently being migrated to PMEM, treat it as a
            // miss rather than spinning.  With async demotion the in-flight
            // window is very short; the caller can retry.
            if self.in_flight_demotions.contains(key) {
                self.stats.misses.fetch_add(1, Ordering::Relaxed);
                return Err(CacheError::KeyNotFound);
            }
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
    self.demoted_keys.remove(key);

    if in_small || in_main {
        Ok(())
    } else {
        Err(CacheError::KeyNotFound)
    }
}

/// Returns `true` if the key exists (and has not expired) in either tier,
/// or is currently being migrated from the DRAM tier to the PMEM tier.
pub fn has(&self, key: &K) -> bool {
    self.small.has(key) || self.main.has(key) || self.in_flight_demotions.contains(key)
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

/// Returns `true` if `key` is currently being migrated from the small DRAM
/// tier to the far PMEM tier (i.e. the demotion worker has not yet written
/// it to PMEM).
///
/// Under normal conditions this window is very short.  This method is
/// primarily useful for diagnostics and tests.
pub fn has_in_flight_demotion(&self, key: &K) -> bool {
    self.in_flight_demotions.contains(key)
}

/// Clears **both** tiers and the internal ghost-tracking set.
///
/// # Errors
///
/// Propagates any [`CacheError`] returned by the underlying wipe calls.
pub fn wipe(&self) -> Result<(), CacheError> {
    self.small.wipe()?;
    self.main.wipe()?;
    self.demoted_keys.clear();
    Ok(())
}

/// Returns a point-in-time snapshot of the cache statistics.
///
/// `dram_items` and `pmem_items` are queried live from each tier's internal
/// status counter.  All other fields are cumulative atomics.
pub fn stats(&self) -> HybridCacheStats {
    HybridCacheStats {
        small_hits:        self.stats.small_hits.load(Ordering::Relaxed),
        main_hits:         self.stats.main_hits.load(Ordering::Relaxed),
        misses:            self.stats.misses.load(Ordering::Relaxed),
        promotions:        self.stats.promotions.load(Ordering::Relaxed),
        demotions:         self.stats.demotions.load(Ordering::Relaxed),
        dram_items:        self.small.status().map(|s| s.num_objects()).unwrap_or(0),
        pmem_items:        self.main.status().map(|s| s.num_objects()).unwrap_or(0),
        dropped_demotions: self.stats.dropped_demotions.load(Ordering::Relaxed),
        dropped_promotions:self.stats.dropped_promotions.load(Ordering::Relaxed),
    }
}
}
