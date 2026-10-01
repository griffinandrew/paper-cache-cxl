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

	/// The value could not be held by the cache: its accounted size -- the
	/// base size (key, value and expiry as the allocator rounds them) plus the
	/// per-object overhead the cache charges for every object -- is over the
	/// level capacity eviction holds the cache at, so accepting it would evict
	/// the whole cache and then the value itself.
	///
	/// That level is `EVICTION_HIGH_WATERMARK` (0.98 by default) of the cache's
	/// current `max_size`, so it moves with `resize`; a value whose size is
	/// close to `max_size` is refused, not only one larger than it. With
	/// `EVICTION_HIGH_WATERMARK=1.0` only a value whose base size exceeds
	/// `max_size` is (the check before the watermark existed). Refused before
	/// the value is allocated, in every cache: flat and tiered, over either
	/// object store.
	#[error("the value size cannot exceed the cache's eviction threshold")]
	ExceedingValueSize,

	/// A set by a borrowed key (`PaperCache::set_borrowed`,
	/// `PaperCache::reserve_set_borrowed`) whose bytes no key of the cache's key
	/// type can hold: only a `String` cache refuses any, those that are not
	/// UTF-8. Nothing was allocated or sent. A lookup of such bytes is not an
	/// error, it is a miss: no key is stored under them.
	#[error("the key's bytes are not a valid key of this cache")]
	InvalidKey,

	#[error("the cache size cannot be zero")]
	ZeroCacheSize,

	#[error("must configure at least one eviction policy")]
	EmptyPolicies,

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
	/// found nothing to evict or no progress within its window -- or, for
	/// `PaperCache::reserve_set` (S9), the deadline passed while it waited.
	/// Nothing was allocated or sent; overwrites, gets and deletes continue
	/// (S5).
	#[error("the metadata of a new key would not fit the fast tier")]
	MetadataOverflow,

	/// A `GateConfig` no set could run under (`GateConfig::validate`).
	#[error("invalid admission (gate) configuration")]
	InvalidGateConfig,

	/// A set waited for room in the fast tier and NOTHING was freed for the
	/// byte gate's `stall_window` -- no demotion landed, no byte was
	/// refunded -- once the policy worker had caught up with its channel, or
	/// for five windows of a worker showing no sign of life (hung): a stuck
	/// state (a dead migration consumer, bytes pinned by readers, a hung
	/// worker, a bug), not load (a worker behind its channel only lengthens
	/// the wait), and the
	/// cache's `on_stall` is `Error` (the default). Nothing was allocated or
	/// sent (S5).
	///
	/// Also what `PaperCache::reserve_set` returns when its DEADLINE passes
	/// while the set waits for room in the fast tier (S9): the tier stayed full
	/// -- whether or not anything was being freed -- for as long as the caller
	/// would wait, whatever `on_stall` is.
	#[error("the fast tier freed nothing for the gate's stall window")]
	FastTierStalled,

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
