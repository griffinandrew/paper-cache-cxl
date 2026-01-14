/*
 * Copyright (c) Kia Shakiba
 *
 * This source code is licensed under the GNU AGPLv3 license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! TieringObject - Object type specifically for the tiering manager
//!
//! This object type ensures that both the key and value are always allocated in DRAM,
//! regardless of whether allocator_api is enabled. The tiering manager's DRAM cache
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
        self.expiry.is_some_and(|expiry| expiry <= Instant::now())
    }
}
