/*




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

/// TieringObject struct for the tiering manager's DRAM cache.
/// Both the key and value are always stored in DRAM (not using Hybrid allocator)
/// to ensure fast access to hot objects.
#[derive(Clone)]
pub struct TieringObject<K> {
    key: K,
    data: Arc<Box<[u8]>>,
    expiry: ExpireTime,
}

impl<K> TieringObject<K> {
    /// Create a new TieringObject with an explicit expiry time
    pub fn with_expiry(key: K, data: Box<[u8]>, expiry: ExpireTime) -> Self {
        TieringObject {
            key,
            data: Arc::new(data),
            expiry,
        }
    }

    /// Get a reference to the key
    pub fn key(&self) -> &K {
        &self.key
    }

    /// Get a clone of the data Arc
    pub fn data(&self) -> Arc<Box<[u8]>> {
        self.data.clone()
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
        self.expiry.is_some_and(|expiry| expiry.get() <= crate::object::now_ticks())
    }
}



*/



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

#[cfg(feature = "hashtable_tiering")]
/// Data storage mode for TieringObject (hashtable_tiering feature)
#[derive(Clone)]
pub enum TieringData<V> {
    /// Physical copy of data in DRAM
    PhysicalCopy(Arc<Box<[u8]>>),
    /// Reference to data in CXL/PMEM (zero-copy)
    CxlReference(Arc<V>),
}

/// TieringObject struct for the tiering manager's DRAM cache.
/// Both the key and value are always stored in DRAM (not using Hybrid allocator)
/// to ensure fast access to hot objects.
#[cfg(not(feature = "hashtable_tiering"))]
#[derive(Clone)]
pub struct TieringObject<K> {
    key: K,
    data: Arc<Box<[u8]>>,
    expiry: ExpireTime,
}

#[cfg(feature = "hashtable_tiering")]
#[derive(Clone)]
pub struct TieringObject<K, V = Box<[u8]>> {
    key: K,
    data: TieringData<V>,
    expiry: ExpireTime,
}

#[cfg(not(feature = "hashtable_tiering"))]
impl<K> TieringObject<K> {
    /// Create a new TieringObject with an explicit expiry time
    pub fn with_expiry(key: K, data: Box<[u8]>, expiry: ExpireTime) -> Self {
        TieringObject {
            key,
            data: Arc::new(data),
            expiry,
        }
    }

    /// Get a reference to the key
    pub fn key(&self) -> &K {
        &self.key
    }

    /// Get a clone of the data Arc
    pub fn data(&self) -> Arc<Box<[u8]>> {
        self.data.clone()
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
        self.expiry.is_some_and(|expiry| expiry.get() <= crate::object::now_ticks())
    }
}

#[cfg(feature = "hashtable_tiering")]
impl<K, V> TieringObject<K, V> {
    /// Create a new TieringObject with physical copy in DRAM
    pub fn with_expiry(key: K, data: Box<[u8]>, expiry: ExpireTime) -> Self {
        TieringObject {
            key,
            data: TieringData::PhysicalCopy(Arc::new(data)),
            expiry,
        }
    }

    /// Create a new TieringObject with CXL reference (zero-copy)
    pub fn with_cxl_reference(key: K, data: Arc<V>, expiry: ExpireTime) -> Self {
        TieringObject {
            key,
            data: TieringData::CxlReference(data),
            expiry,
        }
    }

    /// Get a reference to the key
    pub fn key(&self) -> &K {
        &self.key
    }

    /// Get a reference to the data
    pub fn data(&self) -> &TieringData<V> {
        &self.data
    }

    /// Get the data as bytes (either from physical copy or CXL reference)
    /// Returns a Vec for compatibility with the cache API
    /// Note: This always allocates due to API requirements
    pub fn data_as_bytes(&self) -> Vec<u8>
    where
        V: AsRef<[u8]>,
    {
        match &self.data {
            TieringData::PhysicalCopy(arc_data) => {
                // For physical copy in DRAM, clone the bytes
                arc_data.as_ref().to_vec()
            }
            TieringData::CxlReference(arc_val) => {
                // For CXL reference, read and clone the bytes from CXL memory
                arc_val.as_ref().as_ref().to_vec()
            }
        }
    }

    /// Check if this object holds a CXL reference (warm tier) vs physical copy (hot tier)
    pub fn is_warm_tier(&self) -> bool {
        matches!(&self.data, TieringData::CxlReference(_))
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
        self.expiry.is_some_and(|expiry| expiry.get() <= crate::object::now_ticks())
    }
}