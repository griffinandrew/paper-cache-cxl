/*
 * Copyright (c) Kia Shakiba
 *
 * This source code is licensed under the GNU AGPLv3 license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! Admission Tier Cache — a two-tier cache backed by inner PaperCache instances.
//!
//! Each tier is a `PaperCache<K, BufferDRAM, RandomState>` that owns both the
//! object store and the eviction-policy worker (LRU, 2Q, …).  When a `get` is
//! issued, it flows into the appropriate tier's PaperCache which fires a
//! `WorkerEvent::Get` to that tier's policy worker, updating the eviction stack
//! automatically.  The `AdmissionTierCache` model uses those stacks as a
//! reference to guide tier-promotion decisions rather than maintaining
//! independent data structures.
//!
//! # Object lifecycle
//!
//! ```text
//! set(key) ──► DRAM tier PaperCache only (2Q policy)
//!
//! DRAM eviction ──► eviction callback fires ──► written to far-memory tier
//!
//! get hit DRAM ──► DRAM 2Q stack refreshed via WorkerEvent::Get
//! get hit far  ──► far LRU stack refreshed via WorkerEvent::Get
//!                  copy promoted back to DRAM tier PaperCache
//! ```
//!
//! # Independence
//!
//! This module is enabled solely via the `admission_tiering` feature flag.
//! The method bodies that operate on PaperCache instances are additionally
//! gated on `#[cfg(all(feature = "all_dram", not(feature = "global_flatmap_dram")))]`
//! because they rely on the DRAM-backed PaperCache impl.

use std::{
    hash::{Hash, RandomState},
    sync::{
        Arc,
        atomic::{AtomicU64, Ordering},
    },
};

use typesize::TypeSize;

use crate::{CacheError, CacheSize, policy::PaperPolicy};

// ─── Configuration ────────────────────────────────────────────────────────────

/// Configuration for the admission tier cache.
#[derive(Clone, Debug)]
pub struct AdmissionTierConfig {
    /// Maximum byte capacity for the DRAM (hot) tier.
    pub dram_max_bytes: CacheSize,
    /// Maximum byte capacity for the far-memory (cold) tier.
    pub far_max_bytes: CacheSize,
    /// 2Q `k_in` parameter: fraction of tier capacity allocated to the
    /// "new arrivals" (a1_in) queue. Default: `0.25`.
    pub k_in: f64,
    /// 2Q `k_out` parameter: fraction of tier capacity allocated to the
    /// "overflow" (a1_out) queue. Default: `0.50`.
    pub k_out: f64,
}

impl Default for AdmissionTierConfig {
    fn default() -> Self {
        AdmissionTierConfig {
            dram_max_bytes: 16 * 1024 * 1024,  // 16 MiB
            far_max_bytes: 64 * 1024 * 1024,   // 64 MiB
            k_in: 0.25,
            k_out: 0.50,
        }
    }
}

// ─── Statistics ───────────────────────────────────────────────────────────────

/// Runtime statistics for the admission tier cache.
///
/// Objects in the far-memory tier only arrived there via DRAM eviction.
/// An object is never simultaneously present in both tiers.
#[derive(Clone, Debug, Default)]
pub struct AdmissionTierStats {
    /// Current number of physical entries in the DRAM tier.
    pub dram_objects: u64,
    /// Current byte usage in the DRAM tier.
    pub dram_bytes: u64,
    /// Current number of physical entries in the far-memory tier.
    pub far_objects: u64,
    /// Current byte usage in the far-memory tier.
    pub far_bytes: u64,
    /// Cumulative count of promotions: far-memory objects copied back to DRAM.
    pub promotions_to_dram: u64,
    /// Cumulative `get` hits served from the DRAM tier.
    pub dram_hits: u64,
    /// Cumulative `get` hits served from the far-memory tier.
    pub far_hits: u64,
    /// Cumulative `get` misses (not found in either tier).
    pub misses: u64,
}

// ─── Public cache struct ──────────────────────────────────────────────────────

/// A two-tier cache whose eviction ordering is driven by inner PaperCache
/// instances.
///
/// - `K`: key type. Must implement `Eq + Hash + Clone + TypeSize + Debug`.
///
/// Each PaperCache tier runs its own policy worker thread.  `get` events
/// propagate into those workers automatically — the admission tier model
/// decides *which* tier to consult and delegates ordering to the policy stacks.
///
/// Objects **only** enter the far-memory tier when they are evicted from the
/// DRAM tier — never on an initial `set`.
pub struct AdmissionTierCache<K> {
    /// Hot DRAM tier backed by a PaperCache with a 2Q eviction policy.
    /// Its `PolicyWorker` is wired with an eviction callback that writes
    /// evicted objects to `far_cache`.
    dram_cache: Arc<crate::PaperCache<K, crate::BufferDRAM, RandomState>>,
    /// Cold far-memory tier backed by a PaperCache with an LRU eviction policy.
    /// Only contains objects that were evicted from the DRAM tier.
    far_cache: Arc<crate::PaperCache<K, crate::BufferDRAM, RandomState>>,

    config: AdmissionTierConfig,

    // ── Cumulative counters (updated by each operation) ──────────────────────
    dram_hits: AtomicU64,
    far_hits: AtomicU64,
    misses: AtomicU64,
    promotions_to_dram: AtomicU64,
}

// ─── Constructors ─────────────────────────────────────────────────────────────

/// Create a new [`AdmissionTierCache`] from a config, constructing both tier
/// PaperCaches internally.
///
/// Requires the `all_dram` feature (and not `global_flatmap_dram`) so that
/// the DRAM-backed PaperCache impl is available.
#[cfg(all(
    feature = "all_dram",
    not(feature = "global_flatmap_dram"),
    not(all(feature = "key_value_pmem", feature = "enable_tiering_manager")),
))]
impl<K> AdmissionTierCache<K>
where
    K: 'static + Eq + Hash + TypeSize + std::fmt::Debug + Clone + Send + Sync,
{
    /// Create a new cache with the supplied configuration.
    ///
    /// Objects `set()` into this cache are written to the DRAM tier only.
    /// When the DRAM tier evicts an object, the eviction callback automatically
    /// writes it to the far-memory tier, so objects only ever enter far-memory
    /// via DRAM eviction.
    pub fn new(config: AdmissionTierConfig) -> Result<Self, CacheError> {
        // Build the far-memory tier first (no eviction callback needed).
        let far_cache = Arc::new(
            crate::PaperCache::<K, crate::BufferDRAM, RandomState>::with_hasher_tier(
                config.far_max_bytes,
                &[PaperPolicy::Lru],
                PaperPolicy::Lru,
                RandomState::default(),
            )?,
        );

        // Build the DRAM tier with an eviction callback that forwards evicted
        // objects to far_cache.  The closure captures an Arc so it does not
        // hold a borrow on Self.
        let far_for_cb = Arc::clone(&far_cache);
        let eviction_cb: Arc<dyn Fn(&crate::object::Object<K, crate::BufferDRAM>) + Send + Sync> =
            Arc::new(move |object| {
                let key = object.key().clone();
                let data = object.data();
                // Ignore errors: e.g. far tier is full or value is zero-size.
                let _ = far_for_cb.set(key, &data, None);
            });

        let dram_cache = Arc::new(
            crate::PaperCache::<K, crate::BufferDRAM, RandomState>::with_hasher_tier_eviction_cb(
                config.dram_max_bytes,
                &[PaperPolicy::TwoQ(config.k_in, config.k_out)],
                PaperPolicy::TwoQ(config.k_in, config.k_out),
                RandomState::default(),
                eviction_cb,
            )?,
        );

        Ok(Self::with_caches(dram_cache, far_cache, config))
    }

    /// Construct from pre-built tier PaperCaches.
    ///
    /// The caller is responsible for wiring the DRAM PaperCache's eviction
    /// callback to forward evicted items to `far_cache` if that behaviour is
    /// desired.
    pub(crate) fn with_caches(
        dram_cache: Arc<crate::PaperCache<K, crate::BufferDRAM, RandomState>>,
        far_cache: Arc<crate::PaperCache<K, crate::BufferDRAM, RandomState>>,
        config: AdmissionTierConfig,
    ) -> Self {
        AdmissionTierCache {
            dram_cache,
            far_cache,
            config,
            dram_hits: AtomicU64::new(0),
            far_hits: AtomicU64::new(0),
            misses: AtomicU64::new(0),
            promotions_to_dram: AtomicU64::new(0),
        }
    }
}

// ─── Public API ───────────────────────────────────────────────────────────────

/// Core get/set/del/has operations backed by the inner PaperCache tiers.
///
/// Gated on `all_dram` (and not `global_flatmap_dram`) because the method
/// bodies call PaperCache methods from that impl block.
#[cfg(all(feature = "all_dram", not(feature = "global_flatmap_dram")))]
impl<K> AdmissionTierCache<K>
where
    K: 'static + Eq + Hash + TypeSize + std::fmt::Debug + Clone + Send + Sync,
{
    /// Insert or update a key-value pair.
    ///
    /// The object is written to the **DRAM tier only**.  It will enter the
    /// far-memory tier automatically if and when the DRAM `PolicyWorker`
    /// evicts it (via the eviction callback registered at construction time).
    ///
    /// # Errors
    /// - [`CacheError::ZeroValueSize`] — `value` is empty.
    pub fn set(&self, key: K, value: &[u8], ttl: Option<u32>) -> Result<(), CacheError> {
        self.dram_cache.set(key, value, ttl)
    }

    /// Retrieve the value associated with `key`.
    ///
    /// Lookup order:
    /// 1. **DRAM tier** — `dram_cache.get(key)` fires `WorkerEvent::Get` into
    ///    the DRAM policy worker, refreshing that tier's 2Q eviction stack.
    /// 2. **Far-memory tier** — `far_cache.get(key)` fires `WorkerEvent::Get`
    ///    into the far-memory policy worker.  The retrieved value is then
    ///    promoted back to the DRAM tier via `dram_cache.set`.
    ///
    /// # Errors
    /// - [`CacheError::KeyNotFound`] — key is not in either tier (or is expired).
    pub fn get(&self, key: &K) -> Result<Vec<u8>, CacheError> {
        // ── DRAM lookup ──────────────────────────────────────────────────────
        // Calling dram_cache.get() fires WorkerEvent::Get into the DRAM policy
        // worker, updating the 2Q eviction stack for this key.
        match self.dram_cache.get(key) {
            Ok(data) => {
                self.dram_hits.fetch_add(1, Ordering::Relaxed);
                return Ok(data);
            }
            Err(_) => {}
        }

        // ── Far-memory lookup ────────────────────────────────────────────────
        // Calling far_cache.get() fires WorkerEvent::Get into the far-memory
        // policy worker, updating the LRU eviction stack for this key.
        match self.far_cache.get(key) {
            Ok(data) => {
                self.far_hits.fetch_add(1, Ordering::Relaxed);
                // Promote a copy back to the DRAM tier so subsequent accesses
                // are fast.  The object will naturally migrate to far memory
                // again if DRAM capacity is exceeded.
                let _ = self.dram_cache.set(key.clone(), &data, None);
                self.promotions_to_dram.fetch_add(1, Ordering::Relaxed);
                Ok(data)
            }
            Err(_) => {
                self.misses.fetch_add(1, Ordering::Relaxed);
                Err(CacheError::KeyNotFound)
            }
        }
    }

    /// Delete the entry associated with `key` from both tiers.
    ///
    /// # Errors
    /// - [`CacheError::KeyNotFound`] — key is not in either tier.
    pub fn del(&self, key: &K) -> Result<(), CacheError> {
        let dram_ok = self.dram_cache.del(key).is_ok();
        let far_ok = self.far_cache.del(key).is_ok();
        if dram_ok || far_ok {
            Ok(())
        } else {
            Err(CacheError::KeyNotFound)
        }
    }

    /// Returns `true` if `key` is present in either tier and is not expired.
    pub fn has(&self, key: &K) -> bool {
        self.dram_cache.has(key) || self.far_cache.has(key)
    }

    /// Return a snapshot of the current runtime statistics.
    ///
    /// Size metrics (`dram_objects`, `dram_bytes`, `far_objects`, `far_bytes`)
    /// are read live from the inner PaperCache status.  Hit/miss counters are
    /// accumulated atomically across all operations.
    pub fn stats(&self) -> AdmissionTierStats {
        let mut stats = AdmissionTierStats {
            dram_hits: self.dram_hits.load(Ordering::Relaxed),
            far_hits: self.far_hits.load(Ordering::Relaxed),
            misses: self.misses.load(Ordering::Relaxed),
            promotions_to_dram: self.promotions_to_dram.load(Ordering::Relaxed),
            ..Default::default()
        };

        if let Ok(s) = self.dram_cache.status() {
            stats.dram_objects = s.num_objects();
            stats.dram_bytes = s.used_size();
        }
        if let Ok(s) = self.far_cache.status() {
            stats.far_objects = s.num_objects();
            stats.far_bytes = s.used_size();
        }

        stats
    }

    /// Return the current configuration.
    pub fn config(&self) -> &AdmissionTierConfig {
        &self.config
    }
}
