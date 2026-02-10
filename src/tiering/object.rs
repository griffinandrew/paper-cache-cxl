/*
 * Copyright (c) Kia Shakiba
 *
 * This source code is licensed under the GNU AGPLv3 license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! TieringObject - Object type specifically for the tiering manager
//!
//! This object type ensures that both the key and value are always allocated in DRAM,
//! regardless of whether key_value_pmem is enabled. The tiering manager's DRAM cache
//! stores hot copies of objects for fast access, so they should always be in DRAM.

use std::{
    sync::Arc,
    time::Instant,
};

use crate::object::ExpireTime;

/// TieringData represents different data storage strategies for tiered objects
#[derive(Clone)]
pub enum TieringData<V> {
    /// Zero-copy pointer to CXL data (Warm tier)
    Reference(Arc<V>),
    /// Physical DRAM copy (Hot tier)
    Physical(Box<[u8]>),
}

/// TieringObject struct for the tiering manager's DRAM cache.
/// Both the key and value are always stored in DRAM (not using Hybrid allocator)
/// to ensure fast access to hot objects.
#[derive(Clone)]
pub struct TieringObject<K, V> {
    key: K,
    data: TieringData<V>,
    expiry: ExpireTime,
}

impl<K, V> TieringObject<K, V> {
    /// Create a new TieringObject with a physical copy (Hot tier)
    pub fn with_physical_copy(key: K, data: Box<[u8]>, expiry: ExpireTime) -> Self {
        TieringObject {
            key,
            data: TieringData::Physical(data),
            expiry,
        }
    }

    /// Create a new TieringObject with a reference (Warm tier)
    pub fn with_reference(key: K, arc_ref: Arc<V>, expiry: ExpireTime) -> Self {
        TieringObject {
            key,
            data: TieringData::Reference(arc_ref),
            expiry,
        }
    }

    /// Get a reference to the key
    pub fn key(&self) -> &K {
        &self.key
    }

    /// Get the tiering data (enum variant)
    pub fn tiering_data(&self) -> &TieringData<V> {
        &self.data
    }

    /// Get the expiry time
    pub fn expiry(&self) -> ExpireTime {
        self.expiry
    }

    /// Check if the key matches
    pub fn key_matches(&self, key: &K) -> bool
    where
        K: Eq,
    {
        self.key.eq(key)
    }

    /// Check if the object is expired
    pub fn is_expired(&self) -> bool {
        self.expiry.is_some_and(|expiry| expiry <= Instant::now())
    }

    /// Get the physical data if it's a hot tier object
    pub fn get_physical_data(&self) -> Option<&[u8]> {
        match &self.data {
            TieringData::Physical(bytes) => Some(bytes),
            _ => None,
        }
    }

    /// Get the reference if it's a warm tier object
    pub fn get_reference(&self) -> Option<&Arc<V>> {
        match &self.data {
            TieringData::Reference(arc) => Some(arc),
            _ => None,
        }
    }

    /// Get data as bytes - works for both Warm and Hot tiers
    pub fn get_data_bytes(&self) -> &[u8]
    where
        V: AsRef<[u8]>,
    {
        match &self.data {
            TieringData::Physical(bytes) => bytes,
            TieringData::Reference(arc) => arc.as_ref().as_ref(),
        }
    }
}

// Default implementation for TieringObject to support FlatMap
impl<K, V> Default for TieringObject<K, V>
where
    K: Default,
    V: Default,
{
    fn default() -> Self {
        TieringObject {
            key: K::default(),
            data: TieringData::Physical(Box::new([])),
            expiry: None,
        }
    }
}

// Backward compatibility - keep the old interface for existing code
impl<K> TieringObject<K, Box<[u8]>> {
    /// Create a new TieringObject with an explicit expiry time (legacy compatibility)
    pub fn with_expiry(key: K, data: Box<[u8]>, expiry: ExpireTime) -> Self {
        TieringObject::with_physical_copy(key, data, expiry)
    }

    /// Get a clone of the data Arc (legacy compatibility)
    pub fn data(&self) -> Arc<Box<[u8]>> {
        match &self.data {
            TieringData::Physical(bytes) => Arc::new(bytes.clone()),
            TieringData::Reference(_) => panic!("Cannot get physical data from warm tier object"),
        }
    }
}
