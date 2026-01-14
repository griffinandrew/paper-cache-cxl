/*
 * Copyright (c) Kia Shakiba
 *
 * This source code is licensed under the GNU AGPLv3 license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! TieringObject - Object type specifically for the tiering manager
//!
//! This object type ensures that when allocator_api is enabled, both the key
//! and value are allocated on the same tier (PMEM) using the Hybrid allocator.

use std::{
    sync::Arc,
    time::Instant,
};

use typesize::TypeSize;

use crate::object::ExpireTime;

#[cfg(feature = "allocator_api")]
use crate::allocator::HybridObjects as Hybrid;

/// TieringObject struct for the allocator_api feature.
/// When allocator_api is enabled, both the key and value are allocated in the pmem tier
/// using the Hybrid allocator (Box<K, Hybrid> and Box<[u8], Hybrid>). This ensures that
/// all tiering data resides in persistent memory for CXL/pmem use cases.
#[cfg(feature = "allocator_api")]
pub struct TieringObject<K> {
    key: Box<K, Hybrid>,
    data: Arc<Box<[u8], Hybrid>>,
    expiry: ExpireTime,
}

/// TieringObject struct for builds without the allocator_api feature.
/// Keys and values are stored as plain types in DRAM.
#[cfg(not(feature = "allocator_api"))]
pub struct TieringObject<K> {
    key: K,
    data: Arc<Box<[u8]>>,
    expiry: ExpireTime,
}

#[cfg(feature = "allocator_api")]
impl<K> TieringObject<K> {
    /// Create a new TieringObject with an explicit expiry time
    pub fn with_expiry(key: K, data: Box<[u8]>, expiry: ExpireTime) -> Self {
        // Convert the regular Box<[u8]> to Box<[u8], Hybrid> by copying the data
        let len = data.len();
        let mut hybrid_data = Box::new_in([0u8; 0], Hybrid);
        
        // We need to allocate the correct size
        let mut vec = Vec::with_capacity_in(len, Hybrid);
        vec.extend_from_slice(&data);
        let hybrid_data = vec.into_boxed_slice();
        
        TieringObject {
            key: Box::new_in(key, Hybrid),
            data: Arc::new(hybrid_data),
            expiry,
        }
    }

    /// Get a reference to the key
    pub fn key(&self) -> &K {
        &*self.key
    }

    /// Get a clone of the data Arc
    pub fn data(&self) -> Arc<Box<[u8], Hybrid>> {
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
        self.key.as_ref().eq(key)
    }

    /// Check if the object is expired
    pub fn is_expired(&self) -> bool {
        self.expiry.is_some_and(|expiry| expiry <= Instant::now())
    }
}

#[cfg(not(feature = "allocator_api"))]
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
        self.expiry.is_some_and(|expiry| expiry <= Instant::now())
    }
}
