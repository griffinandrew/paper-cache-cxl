/*
 * Copyright (c) Kia Shakiba
 *
 * This source code is licensed under the GNU AGPLv3 license found in the
 * LICENSE file in the root directory of this source tree.
 */

use thiserror::Error;

#[derive(Debug, PartialEq, Error)]
pub enum CacheError {
	#[error("internal error")]
	Internal,

	#[error("the key was not found in the cache")]
	KeyNotFound,

	#[error("the value size cannot be zero")]
	ZeroValueSize,

	#[error("the value size cannot exceed the cache size")]
	ExceedingValueSize,

	#[error("the cache size cannot be zero")]
	ZeroCacheSize,

	#[error("must configure at least one eviction policy")]
	EmptyPolicies,

	#[error("cannot configure auto eviction policy")]
	ConfiguredAutoPolicy,

	#[error("cannot configure duplicate eviction policies")]
	DuplicatePolicies,

	#[error("unconfigured policy")]
	UnconfiguredPolicy,

	#[error("invalid policy")]
	InvalidPolicy,

	#[error("allocation failed")]
	AllocationFailed,

	#[error("the fast tier size must be greater than zero and cannot exceed the cache size")]
	InvalidFastTierSize,

	/// A NEW key's metadata would not fit the fast tier: the cache's DRAM
	/// metadata is at its ceiling (`gate::key_ceiling`) and the cache's
	/// `on_metadata_overflow` is `Error` (the default), or `EvictToFit`
	/// found nothing to evict or no progress within its window. Nothing was
	/// allocated or sent; overwrites, gets and deletes continue (S5).
	#[error("the metadata of a new key would not fit the fast tier")]
	MetadataOverflow,

	/// A `GateConfig` no set could run under (`GateConfig::validate`).
	#[error("invalid admission (gate) configuration")]
	InvalidGateConfig,

	/// This BUILD cannot honour the policy, which is not the same thing as the
	/// policy being unparseable -- `InvalidPolicy` already means that, and
	/// conflating the two would report a typo and an unimplemented eviction
	/// order identically.
	///
	/// Raised only by `merged_object_store`, whose object map IS its eviction
	/// stack: it implements the orders it has been taught and no others, and it
	/// must never substitute one for another. It carries the policy so the
	/// message can name what was asked for rather than leaving the operator to
	/// guess which of several configured names was refused.
	#[error(
		"the merged object store does not implement {0}; it implements lru, \
		 fifo, clock and lfu (each in its plain, -compact and -compact-hybrid \
		 spelling). Rebuild without --features merged_object_store to run this \
		 policy."
	)]
	PolicyNotImplemented(crate::PaperPolicy),
}
