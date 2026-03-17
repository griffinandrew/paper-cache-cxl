/*
 * Copyright (c) Kia Shakiba
 *
 * This source code is licensed under the GNU AGPLv3 license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! Admission Tier Cache — a standalone two-tier cache with a DRAM hot tier and
//! a far-memory cold tier, each governed by its own independent 2Q structure.
//!
//! # Overview
//!
//! The `AdmissionTierCache` implements an **admission policy** where:
//!
//! 1. **`set`** — New objects are admitted only to the **DRAM cache** (fast tier).
//! 2. **DRAM eviction** — When the DRAM tier is full, the 2Q policy selects the
//!    coldest object and moves it **down** to the far-memory (slow) tier.
//! 3. **`get` from far memory** — If the object is found in far memory, its
//!    far-memory 2Q entry is updated.  When the 2Q decides the object is hot
//!    (in `a1_out` — second chance — or already in `am`), it is **promoted**
//!    back up to the DRAM cache automatically.
//! 4. **Far-memory eviction** — When the far-memory tier is full, the coldest
//!    object is evicted entirely from the cache.
//!
//! # Independence
//!
//! This module is self-contained and is enabled solely via the `admission_tiering`
//! feature flag.  It does not depend on any existing tiering infrastructure
//! (`enable_tiering_manager`, `sets_dram`) and does not modify any existing code.
//!
//! # Example
//!
//! ```rust
//! use paper_cache::{AdmissionTierCache, AdmissionTierConfig};
//!
//! let config = AdmissionTierConfig {
//!     dram_max_bytes: 4_096,
//!     far_max_bytes: 16_384,
//!     ..Default::default()
//! };
//!
//! let cache = AdmissionTierCache::<u32>::new(config);
//!
//! // Set goes to DRAM only.
//! cache.set(1u32, &[0u8; 64], None).unwrap();
//!
//! // Get checks DRAM first, then far memory.
//! let val = cache.get(&1u32).unwrap();
//! assert_eq!(val.len(), 64);
//! ```

use std::{
    collections::HashMap,
    hash::{Hash, BuildHasher, RandomState},
    sync::{Arc, Mutex},
    time::{Instant, Duration},
};

use crate::{HashedKey, CacheError, CacheSize};
use super::two_q::AdmissionTwoQ;

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
#[derive(Clone, Debug, Default)]
pub struct AdmissionTierStats {
    /// Current number of objects in the DRAM tier.
    pub dram_objects: u64,
    /// Current byte usage in the DRAM tier.
    pub dram_bytes: u64,
    /// Current number of objects in the far-memory tier.
    pub far_objects: u64,
    /// Current byte usage in the far-memory tier.
    pub far_bytes: u64,
    /// Cumulative count of objects moved from DRAM to far memory (evictions).
    pub evictions_to_far: u64,
    /// Cumulative count of objects fully evicted from far memory.
    pub evictions_from_far: u64,
    /// Cumulative count of objects promoted from far memory back to DRAM.
    pub promotions_to_dram: u64,
    /// Cumulative `get` hits served from the DRAM tier.
    pub dram_hits: u64,
    /// Cumulative `get` hits served from the far-memory tier.
    pub far_hits: u64,
    /// Cumulative `get` misses (not found in either tier).
    pub misses: u64,
}

// ─── Internal storage entries ─────────────────────────────────────────────────

/// An object entry stored in the DRAM or far-memory tier.
struct TierEntry<K> {
    /// Original (unhashed) key — used for hash-collision resolution.
    key: K,
    /// Object data.
    data: Arc<Box<[u8]>>,
    /// Expiry timestamp, if any.
    expiry: Option<Instant>,
    /// Byte size of this entry (used for capacity accounting).
    size: u32,
}

impl<K> TierEntry<K> {
    fn new(key: K, data: Box<[u8]>, ttl: Option<u32>, size: u32) -> Self {
        let expiry = ttl
            .filter(|&t| t > 0)
            .map(|t| Instant::now() + Duration::from_secs(t as u64));

        TierEntry {
            key,
            data: Arc::new(data),
            expiry,
            size,
        }
    }

    fn is_expired(&self) -> bool {
        self.expiry.is_some_and(|e| e <= Instant::now())
    }

    fn key_matches(&self, key: &K) -> bool
    where
        K: Eq,
    {
        self.key == *key
    }
}

// ─── Inner state (held under a single Mutex) ──────────────────────────────────

struct Inner<K> {
    dram_store: HashMap<HashedKey, TierEntry<K>>,
    far_store: HashMap<HashedKey, TierEntry<K>>,
    dram_2q: AdmissionTwoQ,
    far_2q: AdmissionTwoQ,
    stats: AdmissionTierStats,
}

impl<K> Inner<K> {
    fn new(config: &AdmissionTierConfig) -> Self {
        Inner {
            dram_store: HashMap::new(),
            far_store: HashMap::new(),
            dram_2q: AdmissionTwoQ::new(config.dram_max_bytes, config.k_in, config.k_out),
            far_2q: AdmissionTwoQ::new(config.far_max_bytes, config.k_in, config.k_out),
            stats: AdmissionTierStats::default(),
        }
    }
}

// ─── Public cache struct ──────────────────────────────────────────────────────

/// A standalone two-tier cache with an admission policy.
///
/// - `K`: key type. Must implement `Eq + Hash + Clone`.
/// - `S`: hasher. Defaults to `RandomState`.
///
/// See the [module documentation](self) for a detailed description of the
/// eviction and promotion policies.
pub struct AdmissionTierCache<K, S = RandomState> {
    inner: Mutex<Inner<K>>,
    config: AdmissionTierConfig,
    hasher: S,
}

impl<K> AdmissionTierCache<K, RandomState>
where
    K: Eq + Hash + Clone,
{
    /// Create a new cache with the supplied configuration.
    pub fn new(config: AdmissionTierConfig) -> Self {
        Self::with_hasher(config, RandomState::default())
    }
}

impl<K, S> AdmissionTierCache<K, S>
where
    K: Eq + Hash + Clone,
    S: BuildHasher,
{
    /// Create a new cache with the supplied configuration and custom hasher.
    pub fn with_hasher(config: AdmissionTierConfig, hasher: S) -> Self {
        let inner = Inner::new(&config);
        AdmissionTierCache {
            inner: Mutex::new(inner),
            config,
            hasher,
        }
    }

    // ── Public API ────────────────────────────────────────────────────────────

    /// Insert or update a key-value pair.
    ///
    /// The object is placed **directly into the DRAM tier** (admission policy).
    /// If the DRAM tier exceeds its capacity after the insert, the coldest
    /// object (according to the DRAM 2Q) is moved to far memory.
    ///
    /// # Errors
    /// - [`CacheError::ZeroValueSize`] — `value` is empty.
    pub fn set(&self, key: K, value: &[u8], ttl: Option<u32>) -> Result<(), CacheError> {
        if value.is_empty() {
            return Err(CacheError::ZeroValueSize);
        }

        let hashed_key = self.hash(&key);
        let data: Box<[u8]> = value.to_vec().into_boxed_slice();
        let size = value.len() as u32;

        let mut inner = self.inner.lock().unwrap();

        // If the key exists in far memory, remove it from there first
        // (the new value will live in DRAM).
        if let Some(old) = inner.far_store.remove(&hashed_key) {
            if old.key_matches(&key) {
                inner.far_2q.remove(hashed_key);
                inner.stats.far_objects = inner.stats.far_objects.saturating_sub(1);
                inner.stats.far_bytes = inner.stats.far_bytes.saturating_sub(old.size as u64);
            } else {
                // Hash collision — put the old entry back.
                inner.far_store.insert(hashed_key, old);
            }
        }

        // Handle existing DRAM entry (update path).
        let old_size = inner.dram_store
            .get(&hashed_key)
            .filter(|e| e.key_matches(&key))
            .map(|e| e.size);

        if let Some(old_sz) = old_size {
            // Update existing entry in DRAM.
            inner.stats.dram_bytes = inner.stats.dram_bytes
                .saturating_sub(old_sz as u64)
                + size as u64;
        } else {
            inner.stats.dram_objects += 1;
            inner.stats.dram_bytes += size as u64;
        }

        let entry = TierEntry::new(key, data, ttl, size);
        inner.dram_store.insert(hashed_key, entry);
        inner.dram_2q.insert(hashed_key, size);

        // Evict from DRAM to far memory if over capacity.
        self.evict_dram_to_far(&mut inner);

        Ok(())
    }

    /// Retrieve the value associated with `key`.
    ///
    /// Lookup order:
    /// 1. DRAM tier — update DRAM 2Q position.
    /// 2. Far-memory tier — update far-memory 2Q; if the 2Q signals the object
    ///    is hot, **promote it to DRAM** before returning.
    ///
    /// # Errors
    /// - [`CacheError::KeyNotFound`] — key is not in either tier (or is expired).
    pub fn get(&self, key: &K) -> Result<Vec<u8>, CacheError> {
        let hashed_key = self.hash(key);
        let mut inner = self.inner.lock().unwrap();

        // ── DRAM lookup ──────────────────────────────────────────────────────
        // Extract all needed info while the borrow is active, then act on it.
        let dram_info = inner.dram_store
            .get(&hashed_key)
            .filter(|e| e.key_matches(key))
            .map(|e| (e.data.as_ref().to_vec(), e.size, e.is_expired()));

        if let Some((data, size, expired)) = dram_info {
            if !expired {
                inner.dram_2q.access(hashed_key);
                inner.stats.dram_hits += 1;
                return Ok(data);
            }
            // Expired DRAM entry — remove it.
            inner.dram_store.remove(&hashed_key);
            inner.dram_2q.remove(hashed_key);
            inner.stats.dram_objects = inner.stats.dram_objects.saturating_sub(1);
            inner.stats.dram_bytes = inner.stats.dram_bytes.saturating_sub(size as u64);
            inner.stats.misses += 1;
            return Err(CacheError::KeyNotFound);
        }

        // ── Far-memory lookup ────────────────────────────────────────────────
        let far_info = inner.far_store
            .get(&hashed_key)
            .filter(|e| e.key_matches(key))
            .map(|e| (e.data.as_ref().to_vec(), e.size, e.is_expired()));

        if let Some((data, size, expired)) = far_info {
            if !expired {
                // Update far-memory 2Q position for eviction ordering.
                inner.far_2q.access(hashed_key);
                inner.stats.far_hits += 1;

                // Any re-access to a far-memory object signals it is "warm".
                // Promote it back to the DRAM tier immediately.
                // Use if-let to safely handle the remove (the lock is held, but be
                // explicit rather than relying on unwrap).
                if let Some(entry) = inner.far_store.remove(&hashed_key) {
                    inner.far_2q.remove(hashed_key);
                    inner.stats.far_objects = inner.stats.far_objects.saturating_sub(1);
                    inner.stats.far_bytes = inner.stats.far_bytes.saturating_sub(size as u64);

                    inner.dram_store.insert(hashed_key, entry);
                    inner.dram_2q.insert(hashed_key, size);
                    inner.stats.dram_objects += 1;
                    inner.stats.dram_bytes += size as u64;
                    inner.stats.promotions_to_dram += 1;

                    self.evict_dram_to_far(&mut inner);
                }

                return Ok(data);
            }
            // Expired far entry — remove it.
            inner.far_store.remove(&hashed_key);
            inner.far_2q.remove(hashed_key);
            inner.stats.far_objects = inner.stats.far_objects.saturating_sub(1);
            inner.stats.far_bytes = inner.stats.far_bytes.saturating_sub(size as u64);
            inner.stats.misses += 1;
            return Err(CacheError::KeyNotFound);
        }

        inner.stats.misses += 1;
        Err(CacheError::KeyNotFound)
    }

    /// Delete the entry associated with `key` from both tiers.
    ///
    /// # Errors
    /// - [`CacheError::KeyNotFound`] — key is not in either tier.
    pub fn del(&self, key: &K) -> Result<(), CacheError> {
        let hashed_key = self.hash(key);
        let mut inner = self.inner.lock().unwrap();
        let mut found = false;

        // Read DRAM size before mutating.
        let dram_size = inner.dram_store
            .get(&hashed_key)
            .filter(|e| e.key_matches(key))
            .map(|e| e.size);

        if let Some(sz) = dram_size {
            inner.dram_store.remove(&hashed_key);
            inner.dram_2q.remove(hashed_key);
            inner.stats.dram_objects = inner.stats.dram_objects.saturating_sub(1);
            inner.stats.dram_bytes = inner.stats.dram_bytes.saturating_sub(sz as u64);
            found = true;
        }

        // Read far size before mutating.
        let far_size = inner.far_store
            .get(&hashed_key)
            .filter(|e| e.key_matches(key))
            .map(|e| e.size);

        if let Some(sz) = far_size {
            inner.far_store.remove(&hashed_key);
            inner.far_2q.remove(hashed_key);
            inner.stats.far_objects = inner.stats.far_objects.saturating_sub(1);
            inner.stats.far_bytes = inner.stats.far_bytes.saturating_sub(sz as u64);
            found = true;
        }

        if found {
            Ok(())
        } else {
            Err(CacheError::KeyNotFound)
        }
    }

    /// Returns `true` if `key` is present in either tier and is not expired.
    pub fn has(&self, key: &K) -> bool {
        let hashed_key = self.hash(key);
        let inner = self.inner.lock().unwrap();

        let in_dram = inner.dram_store
            .get(&hashed_key)
            .is_some_and(|e| e.key_matches(key) && !e.is_expired());

        let in_far = inner.far_store
            .get(&hashed_key)
            .is_some_and(|e| e.key_matches(key) && !e.is_expired());

        in_dram || in_far
    }

    /// Return a snapshot of the current runtime statistics.
    pub fn stats(&self) -> AdmissionTierStats {
        self.inner.lock().unwrap().stats.clone()
    }

    /// Return the current configuration.
    pub fn config(&self) -> &AdmissionTierConfig {
        &self.config
    }

    /// Update the DRAM tier capacity limit at runtime.
    pub fn set_dram_max_bytes(&mut self, bytes: CacheSize) {
        self.config.dram_max_bytes = bytes;
        let mut inner = self.inner.lock().unwrap();
        inner.dram_2q.resize(bytes);
    }

    /// Update the far-memory tier capacity limit at runtime.
    pub fn set_far_max_bytes(&mut self, bytes: CacheSize) {
        self.config.far_max_bytes = bytes;
        let mut inner = self.inner.lock().unwrap();
        inner.far_2q.resize(bytes);
    }

    // ── Internal helpers ──────────────────────────────────────────────────────

    /// Hash a key using the configured hasher.
    fn hash(&self, key: &K) -> HashedKey {
        use std::hash::Hasher;
        let mut hasher = self.hasher.build_hasher();
        key.hash(&mut hasher);
        hasher.finish()
    }

    /// Evict objects from the DRAM tier into far memory until the tier is
    /// within its configured byte limit.
    fn evict_dram_to_far(&self, inner: &mut Inner<K>) {
        let max = self.config.dram_max_bytes;
        while inner.stats.dram_bytes > max {
            let Some(victim_key) = inner.dram_2q.evict_one() else {
                break;
            };

            if let Some(entry) = inner.dram_store.remove(&victim_key) {
                let size = entry.size;
                inner.stats.dram_objects = inner.stats.dram_objects.saturating_sub(1);
                inner.stats.dram_bytes = inner.stats.dram_bytes.saturating_sub(size as u64);

                // Move to far memory.
                inner.far_2q.insert(victim_key, size);
                inner.far_store.insert(victim_key, entry);
                inner.stats.far_objects += 1;
                inner.stats.far_bytes += size as u64;
                inner.stats.evictions_to_far += 1;

                // If far memory is also over capacity, evict from far.
                self.evict_from_far(inner);
            }
        }
    }

    /// Evict objects from far memory entirely until the tier is within its limit.
    fn evict_from_far(&self, inner: &mut Inner<K>) {
        let max = self.config.far_max_bytes;
        while inner.stats.far_bytes > max {
            let Some(victim_key) = inner.far_2q.evict_one() else {
                break;
            };

            if let Some(entry) = inner.far_store.remove(&victim_key) {
                let size = entry.size;
                inner.stats.far_objects = inner.stats.far_objects.saturating_sub(1);
                inner.stats.far_bytes = inner.stats.far_bytes.saturating_sub(size as u64);
                inner.stats.evictions_from_far += 1;
            }
        }
    }
}
