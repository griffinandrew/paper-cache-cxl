/*
 * Copyright (c) Kia Shakiba
 *
 * This source code is licensed under the GNU AGPLv3 license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! Where each hybrid design's admission rule lives.
//!
//! All 18 `TieredBuffer`-based designs share the two
//! `impl<K, S> PaperCache<K, TieredBuffer, S>` blocks in `lib.rs`, gated only
//! on `hybrid_cache_common`. The one thing that still genuinely differs
//! between them on the `set()` path is which tier a value is built in, so
//! that is all this module holds: [`admission_tier`], a runtime `match` over
//! the cache's [`PaperPolicy`] with one arm per design.
//!
//! Dispatch is *runtime*, not compile-time. Every hybrid build compiles all 18
//! designs; the policy is chosen when the cache is constructed and stored in
//! `AtomicStatus`, so two caches in one process can run different designs.
//! An earlier revision dispatched through a `HybridPolicy` trait with one
//! marker type per feature, selected by an `ActiveHybridPolicy` alias and
//! kept unambiguous by `compile_error!` guards; all of that is gone.

use crate::{HashedKey, ObjectMapRef, Tier};
use crate::status::AtomicStatus;
use crate::policy::PaperPolicy;
use crate::tiered_buffer::TieredBuffer;

/// The object map type every hybrid-cache design uses. Resolves to whatever
/// `ObjectMapRef<K, TieredBuffer>` resolves to crate-wide -- a plain
/// `DashMap` by default, or the `hashbrown_dram`-selected
/// `RwLock<HashMap<..., Global>>` shape when that feature is enabled --
/// rather than always hardcoding `DashMap` regardless of the active
/// storage-backend feature. `admission_tier` implementations go through the
/// `ObjectStore` abstraction (see `crate::object_store`) so they work
/// unchanged against either shape.
pub type HybridObjectMap<K> = ObjectMapRef<K, TieredBuffer>;

/// Which tier a `set()` builds the value's bytes in, for the given hybrid
/// policy.
///
/// Runtime replacement for the per-design `HybridPolicy` marker types this
/// crate used to dispatch through at compile time: one match, with each arm
/// carrying its design's admission rule
/// verbatim. Non-hybrid policies never reach `set()` on a hybrid cache
/// (construction rejects them), so the catch-all admits fast, which is also
/// every design's default for a brand-new key except the ghost variants.
pub fn admission_tier<K>(
	policy: PaperPolicy,
	hashed_key: HashedKey,
	status: &AtomicStatus,
	objects: &HybridObjectMap<K>,
) -> Tier {
	use crate::object_store::ObjectStore;

	// Shared bindings the arms below use; harmless where unused.
	let _ = (&hashed_key, &status, &objects);

	match policy {
		PaperPolicy::FifoHybrid | PaperPolicy::LruLfuHybrid(..) => {
			let existing_tier = objects.get_ref(&hashed_key)
				.map(|object| if object.data().is_fast() { crate::Tier::Fast } else { crate::Tier::Slow });
			match existing_tier {
				Some(crate::Tier::Slow) => crate::Tier::Slow,
				Some(crate::Tier::Fast) | None => crate::Tier::Fast,
			}
		},
		PaperPolicy::LfuHybrid => {
			match objects.get_ref(&hashed_key) {
				Some(object) => match object.data().is_fast() {
					true => crate::Tier::Fast,
					false => crate::Tier::Slow,
				},
				None if status.hybrid_admission_latched() => crate::Tier::Slow,
				None => crate::Tier::Fast,
			}
		},
		PaperPolicy::LruHybrid | PaperPolicy::LruSizedHybrid | PaperPolicy::TwoQFastAdmissionHybrid(..) | PaperPolicy::TwoQFastAdmissionReprieveHybrid(..) => {
			crate::Tier::Fast
		},
		PaperPolicy::S3FifoGhostHybrid(..) | PaperPolicy::S3FifoGhostLazyDemotionHybrid(..) | PaperPolicy::S3FifoHybrid(..) => {
			match objects.get_ref(&hashed_key) {
				Some(object) => match object.data().is_fast() {
					true => crate::Tier::Fast,
					false => crate::Tier::Slow,
				},
				None => crate::Tier::Slow,
			}
		},
		PaperPolicy::S3FifoGhostLazyDemotionFastAdmissionHybrid(..) | PaperPolicy::S3FifoGhostLazyDemotionFastAdmissionMidpointHybrid(..) => {
			match objects.get_ref(&hashed_key) {
				Some(object) => match object.data().is_fast() {
					true => crate::Tier::Fast,
					false => crate::Tier::Slow,
				},
				None => crate::Tier::Fast,
			}
		},
		PaperPolicy::S3FifoLazyDemotionFastAdmissionMidpointReprieveHybrid(..) | PaperPolicy::S3FifoLazyDemotionFastAdmissionReprieveHybrid(..) | PaperPolicy::S3FifoLazyDemotionFastAdmissionSplitSlowReprieveHybrid(..) => {
			match objects.get_ref(&hashed_key) {
				Some(object) if object.data().is_slow() => crate::Tier::Slow,
				_ => crate::Tier::Fast,
			}
		},
		PaperPolicy::S3FifoLazyDemotionReprieveHybrid(..) => {
			match objects.get_ref(&hashed_key) {
				Some(object) if object.data().is_slow() => crate::Tier::Slow,
				Some(_) => crate::Tier::Fast,
				None => crate::Tier::Slow,
			}
		},
		PaperPolicy::TwoQGhostHybrid(..) | PaperPolicy::TwoQHybrid(..) => {
			match objects.get_ref(&hashed_key) {
				Some(_) => crate::Tier::Fast,
				None => crate::Tier::Slow,
			}
		},
		_ => crate::Tier::Fast,
	}
}
