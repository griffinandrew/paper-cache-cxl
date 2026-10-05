/*
 * Copyright (c) Kia Shakiba
 *
 * This source code is licensed under the GNU AGPLv3 license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! Backpressure plan S5, commit B1: the admission path of a tiered cache's
//! `set` (`crate::gate`) -- the size checks before anything is allocated, the
//! metadata cap, structural slow placement in every design, every settle on
//! the one figure eff, and the set-path kick.
//!
//! Every test here runs in every unit build over the build's own store,
//! unless it names the designs only the DashMap stores have (the merged store
//! implements four orders and refuses the rest at construction). The
//! worker-level ones drive the worker by hand with `reconcile_tests`' harness,
//! whose client half here goes through the real decision (`gate::decide`);
//! the rest build real caches. The T-numbers are the design's (section 7.1).

use std::time::{Duration, Instant};

use super::*;
use super::eviction_watermarks::{DEFAULT_HIGH, Watermarks};
use super::test_support::{alone_in_with_env, client_set, decide, each_alone, parked_on_the_long_poll, stack, wait_for};
use super::reconcile_tests::{
	Objects, Worker, FAST, LEN, assert_settled, bytes_tier, drain_and_apply, handle, make_worker, of,
	placement, publish, publish_del,
};

use crate::gate::{GateConfig, MetadataModel, MetadataOverflow, Verdict};
use crate::object::Object;
use crate::object::overhead::{OverheadManager, get_policy_overhead, test_overheads};
use crate::status::AtomicStatus;
use crate::{CacheTierSize, PaperCache, TieredBuffer};

use Tier::{Fast, Slow};

/// The designs the worker-level tests run: all 23 tiered designs where the
/// build's store has them, the merged store's four orders in its builds.
#[cfg(not(feature = "merged_object_store"))]
const DESIGNS: &[PaperPolicy] = &[
	PaperPolicy::LruCompactHybrid,
	PaperPolicy::LfuCompactHybrid,
	PaperPolicy::LruLfuCompactHybrid(3),
	PaperPolicy::LruSizedCompactHybrid,
	PaperPolicy::FifoCompactHybrid,
	PaperPolicy::ClockCompactHybrid,
	PaperPolicy::TwoQCompactHybrid(0.25),
	PaperPolicy::TwoQFastAdmissionReprieveCompactHybrid(0.25),
	PaperPolicy::TwoQFullFastAdmissionCompactHybrid(0.25, 0.5),
	PaperPolicy::TwoQGhostCompactHybrid(0.25),
	PaperPolicy::S3FifoCompactHybrid(0.1),
	PaperPolicy::S3FifoFaithfulCompactHybrid(0.1),
	PaperPolicy::S3FifoFaithfulFastAdmissionCompactHybrid(0.1),
	PaperPolicy::S3FifoFaithfulReprieveCompactHybrid(0.1),
	PaperPolicy::S3FifoFaithfulFastAdmissionReprieveCompactHybrid(0.1),
	PaperPolicy::S3FifoGhostCompactHybrid(0.1),
	PaperPolicy::S3FifoGhostLazyDemotionCompactHybrid(0.1),
	PaperPolicy::S3FifoGhostLazyDemotionFastAdmissionCompactHybrid(0.1),
	PaperPolicy::S3FifoGhostLazyDemotionFastAdmissionMidpointCompactHybrid(0.1),
	PaperPolicy::S3FifoLazyDemotionReprieveCompactHybrid(0.1),
	PaperPolicy::S3FifoLazyDemotionFastAdmissionReprieveCompactHybrid(0.1),
	PaperPolicy::S3FifoLazyDemotionFastAdmissionMidpointReprieveCompactHybrid(0.1),
	PaperPolicy::S3FifoLazyDemotionFastAdmissionSplitSlowReprieveCompactHybrid(0.1),
];

#[cfg(feature = "merged_object_store")]
const DESIGNS: &[PaperPolicy] = &[
	PaperPolicy::LruCompactHybrid,
	PaperPolicy::FifoCompactHybrid,
	PaperPolicy::ClockCompactHybrid,
	PaperPolicy::LfuCompactHybrid,
];

/// The designs whose order is ONE list with a cursor at the tier boundary,
/// where tier placement never reorders: a structural key keeps its place.
#[cfg(not(feature = "merged_object_store"))]
const ONE_LIST: &[PaperPolicy] = &[
	PaperPolicy::LruCompactHybrid,
	PaperPolicy::FifoCompactHybrid,
	PaperPolicy::ClockCompactHybrid,
];

#[cfg(feature = "merged_object_store")]
const ONE_LIST: &[PaperPolicy] = &[
	PaperPolicy::LruCompactHybrid,
	PaperPolicy::FifoCompactHybrid,
	PaperPolicy::ClockCompactHybrid,
];

/// The designs with a boundary cursor that a structural key can stand in
/// front of (T18b): the one-list designs, and the 2Q and S3-FIFO designs
/// whose main queue is one list with a cursor.
#[cfg(not(feature = "merged_object_store"))]
const CURSORS: &[PaperPolicy] = &[
	PaperPolicy::LruCompactHybrid,
	PaperPolicy::FifoCompactHybrid,
	PaperPolicy::ClockCompactHybrid,
	PaperPolicy::TwoQCompactHybrid(0.25),
	PaperPolicy::TwoQGhostCompactHybrid(0.25),
	PaperPolicy::S3FifoCompactHybrid(0.1),
	PaperPolicy::S3FifoGhostCompactHybrid(0.1),
	PaperPolicy::S3FifoGhostLazyDemotionCompactHybrid(0.1),
];

#[cfg(feature = "merged_object_store")]
const CURSORS: &[PaperPolicy] = ONE_LIST;

/// The designs with a DRAM admission queue (T18c): the three that police it
/// by EVICTION, full 2Q (whose `a1_in` overflow is a demotion to `a1_out`)
/// and the five REPRIEVE designs, which splice their overflow into main.
const FAST_ADMISSION: &[PaperPolicy] = &[
	PaperPolicy::S3FifoGhostLazyDemotionFastAdmissionCompactHybrid(0.1),
	PaperPolicy::S3FifoGhostLazyDemotionFastAdmissionMidpointCompactHybrid(0.1),
	PaperPolicy::S3FifoFaithfulFastAdmissionCompactHybrid(0.1),
	PaperPolicy::TwoQFullFastAdmissionCompactHybrid(0.25, 0.5),
	PaperPolicy::TwoQFastAdmissionReprieveCompactHybrid(0.25),
	PaperPolicy::S3FifoLazyDemotionFastAdmissionReprieveCompactHybrid(0.1),
	PaperPolicy::S3FifoLazyDemotionFastAdmissionMidpointReprieveCompactHybrid(0.1),
	PaperPolicy::S3FifoLazyDemotionFastAdmissionSplitSlowReprieveCompactHybrid(0.1),
	PaperPolicy::S3FifoFaithfulFastAdmissionReprieveCompactHybrid(0.1),
];

/// The seven DRAM admission queues S5 polices at the drain target (3.6.5);
/// the faithful fast-admission pair's small queue is left ungated.
const POLICED_QUEUES: &[PaperPolicy] = &[
	PaperPolicy::S3FifoGhostLazyDemotionFastAdmissionCompactHybrid(0.1),
	PaperPolicy::S3FifoGhostLazyDemotionFastAdmissionMidpointCompactHybrid(0.1),
	PaperPolicy::TwoQFastAdmissionReprieveCompactHybrid(0.25),
	PaperPolicy::S3FifoLazyDemotionFastAdmissionReprieveCompactHybrid(0.1),
	PaperPolicy::S3FifoLazyDemotionFastAdmissionMidpointReprieveCompactHybrid(0.1),
	PaperPolicy::S3FifoLazyDemotionFastAdmissionSplitSlowReprieveCompactHybrid(0.1),
	PaperPolicy::TwoQFullFastAdmissionCompactHybrid(0.25, 0.5),
];

/// What one `len`-byte value is charged to a tier: the stacks' migrating
/// bytes, and the structural check's `v`.
fn charge(len: usize) -> CacheSize {
	crate::phys::value_charge::<u64>(len as ObjectSize)
}

/// This thread's jemalloc `thread.allocated`: every byte it has allocated,
/// freed or not -- so an allocation made and freed inside a call still shows.
fn thread_allocated() -> u64 {
	let mut value: u64 = 0;
	let mut len = std::mem::size_of::<u64>();

	// SAFETY: reads one u64 statistic into a u64 of the size passed.
	let rc = unsafe {
		tikv_jemalloc_sys::mallctl(
			c"thread.allocated".as_ptr(),
			&mut value as *mut u64 as *mut core::ffi::c_void,
			&mut len,
			std::ptr::null_mut(),
			0,
		)
	};

	assert_eq!(rc, 0, "thread.allocated unavailable");
	value
}

/// The fast tier `f` everywhere the decision and the stack read it -- the
/// status, whose F `publish_gate` publishes eff from, and the stack -- then
/// the pass end's publication. The size-split design gets `f` in two equal
/// segments, and the status its threshold, so its class figures are the
/// stack's.
fn set_fast(worker: &mut Worker, f: CacheSize) {
	match worker.status.policy() {
		PaperPolicy::LruSizedCompactHybrid => {
			let small = f / 2;

			worker.status.set_fast_tier_capacity(small);
			worker.status.set_hybrid_large_fast_capacity(f - small);
			worker.status.set_hybrid_size_threshold(4_096);
			worker.handle_resize_fast_tier(small);
			worker.handle_resize_large_fast_tier(f - small);
			worker.handle_resize_size_threshold(4_096);
		},

		_ => {
			worker.status.set_fast_tier_capacity(f);
			worker.handle_resize_fast_tier(f);
		},
	}

	drain_and_apply(worker);
	worker.publish_gate();
	drain_and_apply(worker);
}

/// A hit, served from where the key's bytes are, and its drain.
fn hit(worker: &mut Worker, objects: &Objects, key: HashedKey) -> Vec<TaggedMigration> {
	worker.handle_get(key, Some(bytes_tier(objects, key)));
	drain_and_apply(worker)
}

/// Every key evicted by the worker's eviction pass, in the order it took
/// them.
fn evict_all(worker: &mut Worker) -> Vec<HashedKey> {
	evict_all_drained(worker).0
}

/// `evict_all`, and the pass's drain.
fn evict_all_drained(worker: &mut Worker) -> (Vec<HashedKey>, Vec<TaggedMigration>) {
	worker.evicted = Some(Vec::new());

	worker.status.set_max_size(1);
	worker.apply_evictions().expect("an eviction pass");
	worker.status.set_max_size(1 << 30);
	let drain = drain_and_apply(worker);

	(worker.evicted.take().expect("recording"), drain)
}

/// A real tiered cache with an admission configuration.
pub(super) fn cache_with(policy: PaperPolicy, fast: CacheSize, config: GateConfig) -> PaperCache<u64, TieredBuffer> {
	PaperCache::<u64, TieredBuffer>::new_with_gate(1 << 20, CacheTierSize::Bytes(fast), policy, config)
		.expect("a tiered cache")
}

pub(super) fn evict_to_fit() -> GateConfig {
	let mut config = GateConfig::default();
	config.on_metadata_overflow = MetadataOverflow::EvictToFit;
	config
}

// ---------------------------------------------------------------------------
// The size checks (3.3)

/// `OverheadManager::base_size_for` -- what `begin_set` checks and charges
/// before anything is allocated -- is `base_size` of the object `commit`
/// builds, and `dram_resident_size_for` its `dram_resident_size`: over key
/// types of every shape -- POD keys, and the byte-string keys (`String`,
/// `Vec<u8>`, `Box<[u8]>`) that `thin_header` holds as bytes inside the item,
/// where the item's size depends on the key's length -- lengths around the
/// value header's size classes, and a TTL of none, 0 (none) and 5 s. Red with
/// the length taken as the value's size (`layoutsize`).
#[test]
fn base_size_for_equals_base_size() {
	fn agree<K: typesize::TypeSize + Clone + std::fmt::Debug + 'static>(overhead_manager: &OverheadManager, key: K) {
		let mut lens: Vec<usize> = vec![0, 1, 7, 8, 9, 15, 16, 17, 31, 32, 33, 100, 1_000, 4_095, 4_096, 4_097, 16_384, 70_000, 1 << 20];
		lens.extend((0..40).map(|i| 8 * i + 3));

		for len in lens {
			for ttl in [None, Some(0), Some(5)] {
				let object = Object::<K, TieredBuffer>::new_in(key.clone(), &vec![1u8; len], Fast, ttl);

				assert_eq!(
					overhead_manager.base_size_for(&key, len, ttl),
					Some(overhead_manager.base_size(&object)),
					"{key:?}, {len} bytes, ttl {ttl:?}: base size",
				);
				assert_eq!(
					overhead_manager.dram_resident_size_for(&key, ttl),
					overhead_manager.dram_resident_size(&object),
					"{key:?}, {len} bytes, ttl {ttl:?}: DRAM-resident size",
				);
			}
		}
	}

	let status = Arc::new(AtomicStatus::new(1 << 30, &[PaperPolicy::LruCompactHybrid], PaperPolicy::LruCompactHybrid).unwrap());
	let overhead_manager = OverheadManager::new(&status);

	agree(&overhead_manager, 7u32);
	agree(&overhead_manager, 7u64);
	agree(&overhead_manager, String::from("a key of some length"));
	agree(&overhead_manager, String::new());
	agree(&overhead_manager, "k".repeat(250));
	agree(&overhead_manager, vec![0xFFu8, 0x00, 0x80, b'k', 0xC3]);
	agree(&overhead_manager, Box::<[u8]>::from(&b"a boxed key, Kia's server's key type"[..]));
}

/// What the byte gate reserves for a set before its value exists
/// (`Sizes::value`, from `phys::value_charge_for`) is what the value it then
/// builds is charged: the object's own `resident_item_bytes`, and for a key
/// held as bytes under `thin_header` that depends on the key. Over every key
/// shape and lengths around the size classes.
#[test]
fn the_gate_reserves_what_the_built_value_charges() {
	fn agree<K: 'static + Eq + std::hash::Hash + typesize::TypeSize + Clone + Send + Sync + std::fmt::Debug>(key: K) {
		let cache = PaperCache::<K, TieredBuffer>::new_with_gate(1 << 30, CacheTierSize::Bytes(1 << 20), PaperPolicy::LruCompactHybrid, GateConfig::default())
			.expect("a tiered cache");

		for len in [0usize, 1, 7, 8, 100, 4_080, 4_096, 5_000] {
			let permit = cache.begin_set(&key, len, None).expect("a permit");
			let object = Object::<K, TieredBuffer>::new_in(key.clone(), &vec![1u8; len], Fast, None);

			assert_eq!(
				permit.sizes.value,
				crate::object::overhead::resident_item_bytes(&object) as CacheSize,
				"{key:?}, {len} bytes: what the gate reserves against what the value is charged",
			);
		}
	}

	agree(7u64);
	agree(String::new());
	agree("k".repeat(43));
	agree("k".repeat(250));
	agree(vec![0xFFu8, 0x00, 0x80, b'k', 0xC3]);
	agree(Box::<[u8]>::from(&b"a boxed key, Kia's server's key type"[..]));
}

/// A value larger than the whole cache is refused -- `ExceedingValueSize`,
/// as always -- BEFORE it is built: the client thread allocates nothing for
/// it. Until S5 the value was built (in DRAM, charged to P) and then the
/// check ran.
#[test]
fn an_oversize_value_is_refused_before_it_is_allocated() {
	let _serialised = migration_test_lock::lock();
	let _per_object = test_overheads::per_object();

	let cache = cache_with(PaperPolicy::LruCompactHybrid, 64 << 10, GateConfig::default());
	let value = vec![5u8; 2 << 20];

	let before = thread_allocated();
	let result = cache.set(1, &value, None);
	let allocated = thread_allocated() - before;

	assert_eq!(result, Err(CacheError::ExceedingValueSize));
	assert!(allocated < 4_096, "the refused 2 MiB set allocated {allocated} bytes");
	assert_eq!(cache.status.live_num_objects(), 0);
}

// ---------------------------------------------------------------------------
// The eviction threshold and the size check (V)

/// A cache of either shape, for the tests of the size check below: FLAT
/// (`BufferDRAM`, LRU) and TIERED (`TieredBuffer`, LRU hybrid, the fast tier as
/// large as the cache, the per-object metadata model so that a cache of a few
/// tens of KB is not "metadata bound"). Both go through the client's real
/// `set`, over this build's object store.
trait Probe {
	fn set_value(&self, key: u64, len: usize) -> Result<(), CacheError>;
	fn resize_to(&self, max_size: CacheSize) -> Result<(), CacheError>;
	fn status(&self) -> &AtomicStatus;
}

/// The flat cache exists in the builds that select a flat object store.
#[cfg(any(feature = "all_dram", feature = "key_value_pmem"))]
impl Probe for PaperCache<u64, crate::BufferDRAM> {
	fn set_value(&self, key: u64, len: usize) -> Result<(), CacheError> {
		self.set(key, &vec![0xAB; len], None)
	}

	fn resize_to(&self, max_size: CacheSize) -> Result<(), CacheError> {
		self.resize(max_size)
	}

	fn status(&self) -> &AtomicStatus {
		&self.status
	}
}

impl Probe for PaperCache<u64, TieredBuffer> {
	fn set_value(&self, key: u64, len: usize) -> Result<(), CacheError> {
		self.set(key, &vec![0xAB; len], None)
	}

	fn resize_to(&self, max_size: CacheSize) -> Result<(), CacheError> {
		self.resize(max_size)
	}

	fn status(&self) -> &AtomicStatus {
		&self.status
	}
}

#[cfg(any(feature = "all_dram", feature = "key_value_pmem"))]
const FLAT: PaperPolicy = PaperPolicy::LruCompact;
const TIERED: PaperPolicy = PaperPolicy::LruCompactHybrid;

/// The shapes' names and policies: the tiered cache, and the flat one where the
/// build has it.
#[cfg(any(feature = "all_dram", feature = "key_value_pmem"))]
const SHAPES: [(&str, PaperPolicy); 2] = [("flat", FLAT), ("tiered", TIERED)];
#[cfg(not(any(feature = "all_dram", feature = "key_value_pmem")))]
const SHAPES: [(&str, PaperPolicy); 1] = [("tiered", TIERED)];

/// A cache of `policy`'s shape with a cap of `max_size` bytes, handed to `body`
/// and dropped (its workers joined) after it.
fn with_cache(policy: PaperPolicy, max_size: CacheSize, body: impl FnOnce(&dyn Probe)) {
	#[cfg(any(feature = "all_dram", feature = "key_value_pmem"))]
	if policy == FLAT {
		let cache = PaperCache::<u64, crate::BufferDRAM>::new(max_size, &[FLAT], FLAT).expect("a flat cache");

		return body(&cache);
	}

	let mut config = GateConfig::default();
	config.metadata_model = MetadataModel::PerObject;

	let cache = PaperCache::<u64, TieredBuffer>::new_with_gate(max_size, CacheTierSize::Bytes(max_size), policy, config)
		.expect("a tiered cache");

	body(&cache);
}

/// Base size and accounted size -- the base plus the per-object overhead
/// `used_size` charges, which is what the eviction loop counts -- of a `len`-byte
/// value of a `u64` key under `policy`.
fn base_and_accounted(policy: PaperPolicy, len: usize) -> (CacheSize, CacheSize) {
	let status = Arc::new(AtomicStatus::new(1 << 30, &[policy], policy).unwrap());
	let overhead_manager = OverheadManager::new(&status);

	let base = overhead_manager.base_size_for(&1u64, len, None).expect("a length in range") as CacheSize;

	(base, base + get_policy_overhead(&policy) as CacheSize)
}

/// The smallest cap whose arming level under `marks` is at least `accounted`
/// bytes: a value of that accounted size is held by it and refused by every
/// cap one byte smaller.
fn smallest_cap_holding(marks: Watermarks, accounted: CacheSize) -> CacheSize {
	let cap = (0..).map(|extra| accounted + extra).find(|cap| marks.bytes(*cap).0 >= accounted).unwrap();

	assert_eq!(marks.bytes(cap).0, accounted, "the level steps by at most one byte per byte of cap");
	assert_eq!(marks.bytes(cap - 1).0, accounted - 1);

	cap
}

/// Every event sent so far has been handled: two whole passes of the policy
/// worker after this call, the worker kicked off its idle poll.
fn quiesce(status: &AtomicStatus) {
	let passes = status.policy_worker_passes();

	wait_for("two more passes of the policy worker", Duration::from_secs(10), || {
		status.kick_policy_worker();
		status.policy_worker_passes() >= passes + 2
	});
}

/// A run that exports an eviction watermark override chose its own operating
/// point, and the tests of the DEFAULT one have nothing to say about it.
fn overridden() -> bool {
	std::env::var_os("EVICTION_HIGH_WATERMARK").is_some() || std::env::var_os("EVICTION_LOW_WATERMARK").is_some()
}

/// V: a value the eviction threshold cannot hold is REFUSED, not accepted and
/// then evicted with the whole cache. The E1 agent's reproduction: a
/// 30,000-byte value (32,876 accounted in some builds) alone in a cache a
/// hundredth over its accounted size -- inside the 2% window between the
/// threshold (98% of `max_size`) and the cap -- was kept before E1 and was lost
/// under it, because a set was refused only above `max_size` on its BASE size.
/// Now `ExceedingValueSize`, nothing built or counted. And the boundary is
/// exact, to the byte, on the ACCOUNTED size (base plus per-object overhead,
/// what the loop charges): a value whose accounted size equals the threshold
/// is held -- kept, after the worker has handled it -- and a cap one byte
/// smaller refuses it. Flat and tiered, over this build's store. Red with the
/// check on the base size against `max_size` (the code before V).
#[test]
fn a_value_the_eviction_threshold_cannot_hold_is_refused() {
	if overridden() {
		return;
	}

	let _serialised = migration_test_lock::lock();

	const LEN: usize = 30_000;

	let marks = Watermarks::new(DEFAULT_HIGH, DEFAULT_HIGH);

	for (shape, policy) in SHAPES {
		let (base, accounted) = base_and_accounted(policy, LEN);

		assert!(accounted > base, "{shape}: an object is charged an overhead beyond its base size");

		// Inside the window: the threshold is under the accounted size, the cap over it.
		let window = accounted + accounted / 100;

		assert!(marks.bytes(window).0 < accounted && accounted <= window, "{shape}: {accounted} accounted in a cache of {window}");
		assert!(base <= window, "{shape}: the base size fits, as the old check saw it");

		with_cache(policy, window, |cache| {
			assert_eq!(cache.set_value(1, LEN), Err(CacheError::ExceedingValueSize), "{shape}: refused in the window");
			assert_eq!(cache.status().live_num_objects(), 0, "{shape}: and nothing was built");

			quiesce(cache.status());
			assert_eq!(cache.status().live_num_objects(), 0, "{shape}");
		});

		// The boundary: held at the smallest cap whose threshold reaches the
		// accounted size, refused one byte under it.
		let cap = smallest_cap_holding(marks, accounted);

		with_cache(policy, cap, |cache| {
			assert_eq!(cache.set_value(1, LEN), Ok(()), "{shape}: held at a cap of {cap} ({accounted} accounted)");
			quiesce(cache.status());
			assert_eq!(cache.status().live_num_objects(), 1, "{shape}: and kept, not evicted with the cache");
		});

		with_cache(policy, cap - 1, |cache| {
			assert_eq!(
				cache.set_value(1, LEN),
				Err(CacheError::ExceedingValueSize),
				"{shape}: refused at a cap of {} ({accounted} accounted, threshold {})",
				cap - 1,
				marks.bytes(cap - 1).0,
			);
			assert_eq!(cache.status().live_num_objects(), 0);
		});
	}
}

/// V: the refusal is read against the CURRENT `max_size`, as the eviction loop
/// reads its threshold, so a `resize` moves it -- on the status and through a
/// real cache of either shape. A value held at a cap is refused after the
/// cache shrinks by a byte, and held again after it grows back.
#[test]
fn the_refusal_moves_with_a_resize() {
	if overridden() {
		return;
	}

	let _serialised = migration_test_lock::lock();

	const LEN: usize = 5_000;

	let marks = Watermarks::new(DEFAULT_HIGH, DEFAULT_HIGH);

	for (shape, policy) in SHAPES {
		let (base, accounted) = base_and_accounted(policy, LEN);
		let cap = smallest_cap_holding(marks, accounted);

		let status = AtomicStatus::new(cap, &[policy], policy).unwrap();

		assert_eq!(status.eviction_watermarks(), marks, "{shape}: the status runs the default watermarks");
		assert!(!status.exceeds_eviction_threshold(base), "{shape}: held at {cap}");

		status.set_max_size(cap - 1);
		assert!(status.exceeds_eviction_threshold(base), "{shape}: refused at {}", cap - 1);

		status.set_max_size(cap);
		assert!(!status.exceeds_eviction_threshold(base), "{shape}: held again at {cap}");

		with_cache(policy, 1 << 20, |cache| {
			assert_eq!(cache.set_value(1, LEN), Ok(()), "{shape}: held in a cache of 1 MiB");

			cache.resize_to(cap - 1).expect("resize");
			assert_eq!(cache.set_value(2, LEN), Err(CacheError::ExceedingValueSize), "{shape}: refused after a shrink to {}", cap - 1);

			cache.resize_to(cap).expect("resize");
			assert_eq!(cache.set_value(2, LEN), Ok(()), "{shape}: held after growing back to {cap}");
		});
	}
}

/// V, `EVICTION_HIGH_WATERMARK=1.0`: the arming level at the cap restores the
/// old edge EXACTLY -- a set is refused when its BASE size exceeds `max_size`,
/// not one byte earlier, and a value whose base size fits but whose accounted
/// size does not is accepted, as it always was. In a child process, where the
/// variable (memoised per process) is in place from the start.
#[test]
fn at_the_cap_the_refusal_is_the_old_edge_exactly() {
	alone_in_with_env(module_path!(), "at_the_cap_the_refusal_is_the_old_edge_exactly", &[("EVICTION_HIGH_WATERMARK", "1.0")], || {
		const LEN: usize = 30_000;

		for (shape, policy) in SHAPES {
			let (base, accounted) = base_and_accounted(policy, LEN);

			assert!(accounted >= base + 2, "{shape}: the window between the base and the accounted size has room");

			// The base size fits, the accounted size does not: accepted. Exactly
			// the base size: accepted. One byte under it: refused.
			for (cap, expected) in [
				(base + (accounted - base) / 2, Ok(())),
				(base, Ok(())),
				(base - 1, Err(CacheError::ExceedingValueSize)),
			] {
				with_cache(policy, cap, |cache| {
					assert!(cache.status().eviction_watermarks().at_cap(), "the child runs the override");
					assert_eq!(cache.status().exceeds_eviction_threshold(base), cap < base);
					assert_eq!(cache.set_value(1, LEN), expected, "{shape}: cap {cap}, base {base}, accounted {accounted}");
				});
			}
		}
	});
}

/// V, an opted-in band (`EVICTION_HIGH_WATERMARK=0.75`, `EVICTION_LOW_WATERMARK=
/// 0.5`): the refusal follows the ARMING level, the high mark, not the drain
/// target -- a value whose accounted size is over the high mark is refused and
/// one at it is held until a pass arms. In a child process (see above).
#[test]
fn a_band_refuses_above_its_arming_level() {
	alone_in_with_env(
		module_path!(),
		"a_band_refuses_above_its_arming_level",
		&[("EVICTION_HIGH_WATERMARK", "0.75"), ("EVICTION_LOW_WATERMARK", "0.5")],
		|| {
			const LEN: usize = 20_000;

			let marks = Watermarks::new(0.75, 0.5);

			for (shape, policy) in SHAPES {
				let (_, accounted) = base_and_accounted(policy, LEN);
				let cap = smallest_cap_holding(marks, accounted);

				// Between the drain target and the arming level: kept until a pass arms.
				assert!(marks.bytes(cap).1 < accounted);

				with_cache(policy, cap, |cache| {
					assert_eq!(cache.status().eviction_watermarks(), marks);
					assert_eq!(cache.set_value(1, LEN), Ok(()), "{shape}: held at the arming level");
					quiesce(cache.status());
					assert_eq!(cache.status().live_num_objects(), 1, "{shape}: and no pass armed");
				});

				with_cache(policy, cap - 1, |cache| {
					assert_eq!(cache.set_value(1, LEN), Err(CacheError::ExceedingValueSize), "{shape}: refused over it");
				});
			}
		},
	);
}

// ---------------------------------------------------------------------------
// The metadata cap (3.4)

/// T10a: at the key ceiling a NEW key's set fails with `MetadataOverflow`
/// (the default), and builds nothing -- the client allocates nothing, the
/// object count does not move; an overwrite of a live key succeeds (it adds
/// no metadata); after a delete a new key fits again. Per-object model with
/// omega 64 on a 4 KiB tier: the ceiling is 64 keys. Both orders the merged
/// store shares with LRU's family and LFU's. Red with the cap off (`nocap`).
#[test]
fn a_new_key_past_the_metadata_ceiling_errs_and_allocates_nothing() {
	let _serialised = migration_test_lock::lock();
	let _overheads = test_overheads::set(64, 100);

	each_alone!("a_new_key_past_the_metadata_ceiling_errs_and_allocates_nothing", [PaperPolicy::LruCompactHybrid, PaperPolicy::LfuCompactHybrid], |policy| {
		let cache = cache_with(policy, 4_096, GateConfig::default());
		let value = [7u8; 1_000];

		for key in 0..64u64 {
			cache.set(key, &value, None).unwrap_or_else(|e| panic!("{policy}: key {key} under the ceiling: {e:?}"));
		}

		assert_eq!(cache.status.gate().k_max(), 64, "{policy}: the ceiling");

		let (count, before) = (cache.status.live_num_objects(), thread_allocated());
		let result = cache.set(64, &value, None);
		let allocated = thread_allocated() - before;

		assert_eq!(result, Err(CacheError::MetadataOverflow), "{policy}: the 65th key");
		assert_eq!(cache.status.live_num_objects(), count, "{policy}: the refused key was counted");
		assert!(allocated < 512, "{policy}: the refused set allocated {allocated} bytes");
		assert!(cache.get(&64).is_err(), "{policy}: the refused key is in the cache");
		assert_eq!(cache.hybrid_stats().metadata_overflows, 1, "{policy}: counted");

		cache.set(3, &[9u8; 1_000], None).expect("an overwrite adds no metadata");
		cache.del(&5).expect("a delete");
		cache.set(64, &value, None).expect("room again after a delete");
	});
}

/// T10b, `EvictToFit`: at the ceiling a new key's set waits while the policy
/// worker evicts the policy's own victim -- exactly one, LRU's tail -- and
/// returns once it is gone: the moment the set returns, the victim is missing
/// and the count is back at the ceiling. Red with a victim other than the
/// policy's (`wrongvictim`) or the key admitted before its room is made
/// (`capafterinsert`).
#[test]
fn evict_to_fit_evicts_the_policys_victim_while_the_set_waits() {
	let _serialised = migration_test_lock::lock();
	let _overheads = test_overheads::set(64, 100);

	let cache = cache_with(PaperPolicy::LruCompactHybrid, 4_096, evict_to_fit());

	for key in 0..64u64 {
		cache.set(key, &[key as u8; 100], None).expect("under the ceiling");
	}

	// The worker has taken every set, in order: key 0 is LRU's tail.
	wait_for("the worker taking the sets", Duration::from_secs(10), || {
		let s = cache.hybrid_stats();
		s.fast_objects + s.slow_objects == 64
	});

	cache.set(64, &[64u8; 100], None).expect("room made");

	assert_eq!(cache.status.live_num_objects(), 64, "the count is back at the ceiling as the set returns");
	assert!(cache.peek(&0).is_err(), "LRU's tail was the victim, gone when the set returned");
	assert!((1..=64u64).all(|key| cache.peek(&key).is_ok()), "one victim only");

	let s = cache.hybrid_stats();
	assert_eq!((s.make_room_requests, s.make_room_evictions, s.make_room_failures), (1, 1, 0));
	assert_eq!(s.metadata_overflows, 0);
}

/// T10c: the measured model's ceiling after a table step. With M published
/// above what the tier can hold, `K_max` is the high-water mark `L_hw` --
/// refilling what the cache has held is free -- so after a delete the key it
/// freed can be reused without an eviction, and the ceiling does not fall
/// with the count (no eviction cascade). Red with the high-water mark
/// dropped (`nohw`).
#[test]
fn the_ceiling_allows_reuse_after_a_table_step() {
	let _serialised = migration_test_lock::lock();

	let (mut worker, objects) = make_worker(PaperPolicy::LruCompactHybrid);

	// The measured model, omega registered (the estimate's unit), and M held
	// at 0 -- as a small cache's structures, before the step.
	worker.status.register_tiered_cache(64);

	let mut config = worker.status.gate().config();
	config.metadata_model = MetadataModel::Measured;
	config.on_metadata_overflow = MetadataOverflow::EvictToFit;
	worker.status.gate().set_config(config);

	worker.status.set_dram_metadata(crate::meta::DramMetadata::default());
	set_fast(&mut worker, FAST);

	for key in 1..=10 {
		client_set(&mut worker, &objects, key, LEN);
		drain_and_apply(&mut worker);
	}

	worker.publish_gate();

	// The step: M above the whole tier.
	worker.status.set_dram_metadata(crate::meta::DramMetadata { map: 2 * FAST, ..Default::default() });
	worker.publish_gate();
	assert_eq!(worker.status.gate().k_max(), 10, "the ceiling is the high-water mark");
	assert!(matches!(decide(&worker, &objects, 11, LEN), Ok(Verdict::NeedsRoom)), "a new key past it needs room");

	// Reuse: a delete, then a new key in its place.
	publish_del(&worker.status, &worker.overhead_manager, &objects, 3);
	worker.handle_del(3);
	drain_and_apply(&mut worker);
	worker.publish_gate();

	assert_eq!(worker.status.gate().k_max(), 10, "the ceiling did not fall with the count");
	assert!(matches!(decide(&worker, &objects, 11, LEN), Ok(Verdict::Admit { .. })), "reuse is free");

	// Admitted -- structural, as every value is on a tier M fills.
	assert_eq!(client_set(&mut worker, &objects, 11, LEN), (Slow, Placement::Structural));
	drain_and_apply(&mut worker);
	assert_eq!(worker.status.gate().stats().make_room_requests, 0, "nothing evicted for it");
}

/// T10d: `EvictToFit` with nothing to evict -- a ceiling of 0 on an empty
/// cache (a 32 B tier, omega 64) -- fails with `MetadataOverflow` at once,
/// not after waiting out its stall window (2 s). Red with the head asking
/// again after a request that evicted nothing (`waitonzero`).
#[test]
fn evict_to_fit_with_nothing_to_evict_fails_at_once() {
	let _serialised = migration_test_lock::lock();
	let _overheads = test_overheads::set(64, 100);

	let cache = cache_with(PaperPolicy::LruCompactHybrid, 32, evict_to_fit());
	assert_eq!(cache.status.gate().k_max(), 0);

	let started = Instant::now();
	let result = cache.set(1, &[1u8; 100], None);
	let took = started.elapsed();

	assert_eq!(result, Err(CacheError::MetadataOverflow));
	assert!(took < Duration::from_millis(500), "it took {took:?}: the set waited on the worker's empty answer");

	let s = cache.hybrid_stats();
	assert_eq!((s.make_room_requests, s.make_room_evictions, s.make_room_failures), (1, 0, 1));
}

/// `EvictToFit` under `stall_window` 0 -- "never wait" -- fails a new key at
/// the ceiling at once and evicts nothing (a 256 B tier, omega 64: a ceiling
/// of 4 keys): a MakeRoom would take the policy's victim for a set that then
/// fails without waiting for it (the correctness review). Red with the
/// MakeRoom sent anyway (`zerowindowroom`).
#[test]
fn evict_to_fit_with_no_window_evicts_nothing() {
	let _serialised = migration_test_lock::lock();
	let _overheads = test_overheads::set(64, 100);

	let mut config = evict_to_fit();
	config.stall_window = Duration::ZERO;
	let cache = cache_with(PaperPolicy::LruCompactHybrid, 256, config);
	assert_eq!(cache.status.gate().k_max(), 4);

	for key in 0..4 {
		cache.set(key, &[1u8; 100], None).expect("under the ceiling");
	}

	assert_eq!(cache.set(4, &[1u8; 100], None), Err(CacheError::MetadataOverflow));

	let s = cache.hybrid_stats();
	assert_eq!((s.metadata_overflows, s.make_room_requests, s.make_room_evictions), (1, 0, 0));
	assert!((0..4).all(|key| cache.has(&key)), "a key was evicted for a set that failed");
}

/// A dead policy worker cannot make room: the waiting set returns
/// `CacheError::Internal` within a second -- the worker's thread, gone by
/// unwinding through a test-only panic at its `MakeRoom`, marks the gate
/// (`WorkerGoneGuard`). Red with the guard off (`noguard`): the set waits
/// out the stall window and reports an overflow.
#[test]
fn a_dead_worker_fails_evict_to_fit_with_internal() {
	let _serialised = migration_test_lock::lock();
	let _overheads = test_overheads::set(64, 100);

	let cache = cache_with(PaperPolicy::LruCompactHybrid, 4_096, evict_to_fit());

	for key in 0..64u64 {
		cache.set(key, &[key as u8; 100], None).expect("under the ceiling");
	}

	cache.status.gate().test_panic_on_make_room.store(true, std::sync::atomic::Ordering::Relaxed);

	let started = Instant::now();
	let result = cache.set(64, &[64u8; 100], None);
	let took = started.elapsed();

	assert_eq!(result, Err(CacheError::Internal));
	assert!(took < Duration::from_secs(1), "it took {took:?}");
	assert!(cache.status.gate().worker_gone());
}

// ---------------------------------------------------------------------------
// Structural slow placement (3.5)

/// T18: a value larger than an EMPTY fast tier -- eff 0 (the metadata fills
/// the tier) and 0 < eff < v -- in every design: the decision builds it slow
/// with a `Structural` set; the stack places it slow; nothing is queued for
/// it (its drain empty, nothing in flight); a hit and an overwrite as large
/// queue nothing and leave it slow, and so does a second hit; a set the
/// client decided NORMAL (built fast, as one decided before eff moved) of a
/// value as large is placed slow by the stack's own check, counted
/// (`structural_placements`), and corrected once toward where it is placed;
/// an eviction pass that takes everything -- a CLOCK, S3-FIFO or faithful
/// second chance or requeue of the referenced key on the way -- queues
/// nothing for it (a promotion its settle undid would leave the settle's
/// entry); and the audit is clean. Red at 0b2c41f (LRU built it fast and
/// demoted it), and per design with the stack's own check off (`noskip`)
/// or the insert's structural decision off (`noflag`).
#[test]
fn a_structural_value_is_built_and_placed_slow_in_every_design() {
	let _serialised = migration_test_lock::lock();

	const J: HashedKey = 1;
	const K: HashedKey = 2;
	const L: HashedKey = 3;

	let v = charge(LEN);

	each_alone!("a_structural_value_is_built_and_placed_slow_in_every_design", DESIGNS, |policy| {
		for eff_zero in [true, false] {
			let case = if eff_zero { "eff 0" } else { "0 < eff < v" };
			let (mut worker, objects) = make_worker(policy);

			// J, while the tier is large: its metadata is M, the stack's own
			// reservation under the per-object model.
			set_fast(&mut worker, FAST);
			client_set(&mut worker, &objects, J, LEN);
			drain_and_apply(&mut worker);

			let m = stack(&worker).dram_reserved_bytes();
			set_fast(&mut worker, if eff_zero { m } else { m + v / 2 });
			assert_eq!(worker.status.effective_fast_capacity(), if eff_zero { 0 } else { v / 2 }, "{policy}, {case}: eff");

			assert_eq!(
				decide(&worker, &objects, K, LEN).ok(),
				Some(Verdict::Admit { tier: Slow, placement: Placement::Structural }),
				"{policy}, {case}: the decision",
			);

			client_set(&mut worker, &objects, K, LEN);
			assert_eq!(bytes_tier(&objects, K), Slow, "{policy}, {case}: built slow");

			let drain = drain_and_apply(&mut worker);
			assert_eq!(of(&drain, K), vec![], "{policy}, {case}: something was queued for K");
			assert_eq!(worker.status.migration_in_flight().pending(K), 0, "{policy}, {case}: in flight");
			assert_eq!(placement(&worker, K), Some(Slow), "{policy}, {case}: placed");

			let drain = hit(&mut worker, &objects, K);
			assert_eq!(of(&drain, K), vec![], "{policy}, {case}: a hit queued something for K");
			assert_eq!(placement(&worker, K), Some(Slow), "{policy}, {case}: a hit placed K");

			client_set(&mut worker, &objects, K, LEN);
			let drain = drain_and_apply(&mut worker);
			assert_eq!(of(&drain, K), vec![], "{policy}, {case}: an overwrite queued something for K");
			assert_eq!((placement(&worker, K), bytes_tier(&objects, K)), (Some(Slow), Slow), "{policy}, {case}: an overwrite");

			let drain = hit(&mut worker, &objects, K);
			assert_eq!(of(&drain, K), vec![], "{policy}, {case}: a second hit queued something for K");

			assert_eq!(worker.status.gate().stats().structural_placements, 0, "{policy}, {case}: the stack disagreed with the client");

			// The stack's own check: a set the client decided NORMAL, built
			// fast, of a value too large for the tier.
			let mut published = publish(&worker.status, &worker.overhead_manager, &objects, L, LEN, Fast);
			published.placement = Placement::Normal;
			handle(&mut worker, L, published);

			let drain = drain_and_apply(&mut worker);
			assert_eq!(placement(&worker, L), Some(Slow), "{policy}, {case}: the stack's own check did not place L slow");
			assert_eq!(worker.status.gate().stats().structural_placements, 1, "{policy}, {case}: the stack's own check was not counted");
			assert_eq!((of(&drain, L), bytes_tier(&objects, L)), (vec![Slow], Slow), "{policy}, {case}: L's one corrective");

			let (_, drain) = evict_all_drained(&mut worker);
			assert_eq!(of(&drain, K), vec![], "{policy}, {case}: the eviction pass queued something for K");
			assert_eq!(of(&drain, L), vec![], "{policy}, {case}: the eviction pass queued something for L");

			assert_settled(&mut worker);
		}
	});
}

/// T18, the order: a structural key keeps its place in the policy's order --
/// in the designs whose order is one list, tier placement never reorders, so
/// with one key's value too large for the tier the eviction order is the
/// order with it small. Red with a structural key placed after the boundary
/// instead of at the front (`afterboundary`).
#[test]
fn a_structural_key_keeps_its_place_in_the_order() {
	let _serialised = migration_test_lock::lock();

	const K: HashedKey = 4;

	each_alone!("a_structural_key_keeps_its_place_in_the_order", ONE_LIST, |policy| {
		let order = |big: bool| {
			let (mut worker, objects) = make_worker(policy);
			set_fast(&mut worker, 4 << 10);

			for key in 1..=8 {
				let len = if big && key == K { 16 << 10 } else { 200 };
				let (_, placement) = client_set(&mut worker, &objects, key, len);
				drain_and_apply(&mut worker);

				assert_eq!(placement == Placement::Structural, big && key == K, "{policy}: key {key}'s placement");
			}

			// Hits on other keys only: they reorder the recency designs (and
			// reference keys for CLOCK) around K, which stays where it was put.
			for key in [2, 6, 1, 7] {
				hit(&mut worker, &objects, key);
			}

			evict_all(&mut worker)
		};

		assert_eq!(order(true), order(false), "{policy}: the structural key moved in the order");
	});
}

/// T18b: the tier-boundary cursors step over structural keys. Fast keys and
/// structural keys interleaved in a design's order (each set, then hit --
/// the hit is what makes a key fast in main in the designs that admit to a
/// probation queue); the tier shrunk to a byte: the settle demotes exactly
/// the fast keys, oldest first, and never a structural key; the gauges end
/// empty; the audit is clean. Red with a cursor stepping onto the previous
/// key whatever its tier (`noprevfast`): the next demotion "demotes" a slow
/// key, queueing it and taking its bytes off the fast gauge.
#[test]
fn boundaries_skip_structural_keys() {
	let _serialised = migration_test_lock::lock();

	const A: HashedKey = 1;
	const K1: HashedKey = 2;
	const B: HashedKey = 3;
	const K2: HashedKey = 4;
	const C: HashedKey = 5;

	each_alone!("boundaries_skip_structural_keys", CURSORS, |policy| {
		let (mut worker, objects) = make_worker(policy);
		set_fast(&mut worker, 8 << 10);

		for key in [A, K1, B, K2, C] {
			let len = if key == K1 || key == K2 { 16 << 10 } else { 200 };
			client_set(&mut worker, &objects, key, len);
			drain_and_apply(&mut worker);
			hit(&mut worker, &objects, key);
		}

		for key in [A, B, C] {
			assert_eq!(placement(&worker, key), Some(Fast), "{policy}: key {key} is fast in main");
		}

		for key in [K1, K2] {
			assert_eq!(placement(&worker, key), Some(Slow), "{policy}: key {key} is structural");
		}

		worker.status.set_fast_tier_capacity(1);
		worker.handle_resize_fast_tier(1);
		let drain = drain_and_apply(&mut worker);

		let demoted: Vec<HashedKey> = drain.iter().filter(|e| e.1 == Slow).map(|e| e.0).collect();
		assert_eq!(demoted, vec![A, B, C], "{policy}: the settle's demotions");
		assert!(drain.iter().all(|e| e.0 != K1 && e.0 != K2), "{policy}: a structural key was queued: {drain:?}");

		assert_eq!((stack(&worker).fast_bytes_used(), stack(&worker).fast_object_count()), (0, 0), "{policy}: the fast gauges");
		assert_settled(&mut worker);
	});
}

/// T18c: the designs with a DRAM admission queue place a structural NEW key
/// slow where the design sends a key its queue cannot keep -- the front of
/// main or `Q_MAIN_SLOW` (S3-FIFO), `a1_out` (full 2Q), the reprieve's
/// destination -- instead of the queue: it is not evicted on arrival, nothing is pushed, and no DRAM holds it. At the stack.
/// Red at 0b2c41f: the evicting designs evicted it on arrival, the rest
/// admitted it to DRAM and demoted it.
#[test]
fn fast_admission_designs_place_a_structural_key_slow_instead_of_evicting_it() {
	for &policy in FAST_ADMISSION {
		let mut stack = policy_stack::init_policy_stack(policy, 1 << 20);
		stack.resize_fast_tier(4_096);
		stack.drain_tier_migrations();

		for key in 1..=3 {
			stack.insert_resident(key, 8_192, 12);

			while stack.needs_capacity_eviction() {
				assert_ne!(stack.evict_one(), Some(key), "{policy}: key {key} was evicted on arrival");
			}

			assert_eq!(stack.placement_of(key), Some(Slow), "{policy}: key {key} placed");
			assert!(stack.drain_tier_migrations().is_empty(), "{policy}: key {key}: something was pushed");
			assert_eq!(stack.fast_bytes_used(), 0, "{policy}: key {key} is in DRAM");
		}

		assert_eq!(stack.len(), 3, "{policy}: every structural key is tracked");
	}
}

/// T18d: a structural admission does not latch LFU, in both stores -- the
/// latch means "the tier is full", and a value too large for an empty tier
/// says nothing about that -- and pushes nothing: the next key that fits is
/// admitted fast. Red with the latch shut by a structural admission
/// (`latchstructural`; `mergedlatchstructural` for the merged store).
#[test]
fn a_structural_admission_does_not_latch_lfu() {
	let _serialised = migration_test_lock::lock();

	let (mut worker, objects) = make_worker(PaperPolicy::LfuCompactHybrid);
	set_fast(&mut worker, 8 << 10);

	let decision = client_set(&mut worker, &objects, 1, 16 << 10);
	let drain = drain_and_apply(&mut worker);

	assert_eq!(decision, (Slow, Placement::Structural));
	assert_eq!(of(&drain, 1), vec![], "a structural admission pushed something");
	assert!(!worker.status.hybrid_admission_latched(), "a structural admission latched the tier");

	assert_eq!(client_set(&mut worker, &objects, 2, 200).0, Fast, "the next key was built slow");
	drain_and_apply(&mut worker);
	assert_eq!(placement(&worker, 2), Some(Fast), "the next key was placed slow");
	assert_settled(&mut worker);
}

// ---------------------------------------------------------------------------
// Every settle on eff (3.6)

/// Every DRAM admission queue rests at the DRAIN TARGET of its budget, as
/// main does (3.6.5): after each insert, its policing and the pass end's
/// resettle (a queue budget shrinks with the new key's reservation after the
/// insert policed it) the queue's DRAM is at most 0.95 of its budget. At the
/// stack, with the queue's carve-out clamped to the whole tier, so its budget
/// is the tier's eff (`F - M`) and its DRAM the stack's fast bytes (no key is
/// ever hit into main). Red with 3.6.5 reverted (`fullqueue`).
#[test]
fn fa_queues_rest_at_the_drain_target() {
	for &policy in POLICED_QUEUES {
		let mut stack = policy_stack::init_policy_stack(policy, 100_000);
		stack.resize_fast_tier(10_000);
		stack.drain_tier_migrations();

		for key in 1..=200 {
			stack.insert_resident(key, 100, 0);

			while stack.needs_capacity_eviction() {
				stack.evict_one();
			}

			stack.resettle();

			let budget = 10_000u64.saturating_sub(stack.dram_reserved_bytes());

			assert!(
				stack.fast_bytes_used() <= policy_stack::drain_target::bytes(budget),
				"{policy}: key {key}: the queue holds {} B of its {budget} B budget",
				stack.fast_bytes_used(),
			);
		}
	}
}

/// S5's LFU admission gate, in both stores: a new key is admitted fast while
/// `fast_used + migrating <= drain_target(F - (L + 1) * omega)` -- the
/// stacks' unit (the value's migrating bytes, not its base size) against the
/// settle target (not eff). Built so the fourth key's migrating bytes meet
/// the target exactly while its base size, 12 bytes more, would not; and,
/// with the tier two bytes smaller, the same key one byte or two past the
/// target, which eff would still admit. Red with the base size in the gate
/// (`baseunit`) or eff in place of the target (`admitoneff`).
#[test]
fn lfu_admits_in_the_stacks_unit_up_to_the_settle_target() {
	let _serialised = migration_test_lock::lock();

	let v = charge(LEN);

	for (shrink, fast) in [(0, Fast), (2, Slow)] {
		let (mut worker, objects) = make_worker(PaperPolicy::LfuCompactHybrid);
		set_fast(&mut worker, FAST);

		for key in 1..=3 {
			assert_eq!(client_set(&mut worker, &objects, key, LEN).0, Fast);
			drain_and_apply(&mut worker);
		}

		// The smallest F whose target, with the fourth key's reservation,
		// holds all four values.
		let omega = stack(&worker).dram_reserved_bytes() / 3;
		let reserved = 4 * omega;
		let mut f = 4 * v + reserved;

		while policy_stack::drain_target::bytes(f - reserved) < 4 * v {
			f += 1;
		}

		assert!(policy_stack::drain_target::bytes(f - reserved) < 4 * v + 12, "the base size would fit too");

		set_fast(&mut worker, f - shrink);

		client_set(&mut worker, &objects, 4, LEN);
		drain_and_apply(&mut worker);

		assert_eq!(placement(&worker, 4), Some(fast), "F = {} ({shrink} B under the fit)", f - shrink);
		assert_eq!(worker.status.hybrid_admission_latched(), fast == Slow, "the latch");
		assert_settled(&mut worker);
	}
}

/// The measured model: the stacks settle on the M the policy worker
/// published -- `0.95 x (F - M)`, with no ghost term on top (the ghost's
/// structures are inside the measured M) -- and reserve exactly M. With an M
/// held by hand: LRU in both stores, and in the DashMap builds a design with
/// a ghost, filled until the ghost holds keys. Red with a ghost's term added
/// under the measured model (`measuredghost`).
#[test]
fn the_measured_model_settles_on_the_published_m() {
	let _serialised = migration_test_lock::lock();

	#[cfg(not(feature = "merged_object_store"))]
	let designs = [PaperPolicy::LruCompactHybrid, PaperPolicy::S3FifoGhostCompactHybrid(0.1)];

	#[cfg(feature = "merged_object_store")]
	let designs = [PaperPolicy::LruCompactHybrid];

	const M: CacheSize = 6 << 10;

	each_alone!("the_measured_model_settles_on_the_published_m", designs, |policy| {
		let (mut worker, objects) = make_worker(policy);

		let mut config = worker.status.gate().config();
		config.metadata_model = MetadataModel::Measured;
		worker.status.gate().set_config(config);

		worker.status.set_dram_metadata(crate::meta::DramMetadata { map: M, ..Default::default() });
		worker.status.set_max_size(40 * charge(LEN));
		set_fast(&mut worker, FAST);

		for key in 1..=120 {
			client_set(&mut worker, &objects, key, LEN);
			drain_and_apply(&mut worker);
			worker.apply_evictions().expect("an eviction pass");
			drain_and_apply(&mut worker);

			// Hits, so the S3-FIFO design promotes into main.
			if key % 2 == 0 && objects.get_ref(&(key - 1)).is_some() {
				hit(&mut worker, &objects, key - 1);
			}

			worker.publish_gate();
			drain_and_apply(&mut worker);

			assert_eq!(stack(&worker).dram_reserved_bytes(), M, "{policy}, key {key}: the stack reserves the published M");
			assert!(
				stack(&worker).fast_bytes_used() <= policy_stack::drain_target::bytes(FAST - M),
				"{policy}, key {key}: {} fast bytes over the target {}",
				stack(&worker).fast_bytes_used(),
				policy_stack::drain_target::bytes(FAST - M),
			);
		}

		assert!(stack(&worker).fast_bytes_used() > policy_stack::drain_target::bytes(FAST - M) - 2 * charge(LEN), "{policy}: the tier filled");
		assert_eq!(worker.status.effective_fast_capacity(), FAST - M);
		assert_settled(&mut worker);
	});
}

/// The per-object model reproduces 0b2c41f's reservation -- `M_model` is the
/// stack's own `L x omega` (plus a ghost's DRAM), eff is `F` less it -- and
/// the measured model publishes the cache's M. (The environment variable's
/// half, `PAPER_DISABLE_SHARED_OVERHEAD=1` zeroing the term under either
/// model, is in `tests/lru_compact_hybrid_cache_integration.rs`, whose
/// process sets it.)
#[test]
fn models_and_the_env_var() {
	let _serialised = migration_test_lock::lock();

	let (mut worker, objects) = make_worker(PaperPolicy::LruCompactHybrid);
	set_fast(&mut worker, FAST);

	for key in 1..=5 {
		client_set(&mut worker, &objects, key, LEN);
		drain_and_apply(&mut worker);
	}

	worker.publish_gate();

	let omega = crate::object::overhead::get_hybrid_dram_shared_overhead(&PaperPolicy::LruCompactHybrid) as CacheSize;
	let s = worker.status.hybrid_stats();

	assert_eq!(s.metadata_model, MetadataModel::PerObject);
	assert_eq!(s.dram_metadata_bytes_model, 5 * omega, "L x omega");
	assert_eq!(s.effective_fast_capacity, FAST - 5 * omega, "F - L x omega, as at 0b2c41f");

	let mut config = worker.status.gate().config();
	config.metadata_model = MetadataModel::Measured;
	worker.status.gate().set_config(config);

	worker.status.set_dram_metadata(crate::meta::DramMetadata { map: 1_234, stack: 100, headers: 10, keys: 0, slow: 0 });
	worker.publish_gate();

	let s = worker.status.hybrid_stats();

	assert_eq!(s.metadata_model, MetadataModel::Measured);
	assert_eq!(s.dram_metadata_bytes_model, 1_344, "the published M");
	assert_eq!(s.effective_fast_capacity, FAST - 1_344);
}

// ---------------------------------------------------------------------------
// The set-path kick (3.7)

/// T17: the first set after an idle spell wakes the policy worker -- the
/// stack has it within 200 ms, where a worker left parked on its 1 s poll
/// would take up to a second -- in both stores. One kick. Red with the
/// set-path kick off (`nokick`).
#[test]
fn a_set_after_an_idle_spell_wakes_the_worker() {
	let _serialised = migration_test_lock::lock();
	let _per_object = test_overheads::per_object();

	let cache = cache_with(PaperPolicy::LruCompactHybrid, 256 << 10, GateConfig::default());
	parked_on_the_long_poll(&cache.status, Duration::from_millis(1));

	let started = Instant::now();
	cache.set(1, &[1u8; 100], None).expect("set");

	wait_for("the worker taking the set", Duration::from_secs(10), || {
		let s = cache.hybrid_stats();
		s.fast_objects + s.slow_objects == 1
	});

	let took = started.elapsed();

	assert!(took < Duration::from_millis(200), "the set's worker took {took:?}: it slept through the kick");
	assert_eq!(cache.hybrid_stats().idle_kicks, 1);
}

/// One kick per idle spell: 1,000 sets paced over two seconds kick the worker
/// at most once for each time it went idle -- the first set after an idle
/// spell does, and the rest find the idle bit clear (the worker polls short
/// while sets arrive). Red with a kick on every set (`kickalways`).
#[test]
fn one_kick_per_idle_spell() {
	let _serialised = migration_test_lock::lock();
	let _per_object = test_overheads::per_object();

	let cache = cache_with(PaperPolicy::LruCompactHybrid, 256 << 10, GateConfig::default());
	parked_on_the_long_poll(&cache.status, Duration::from_millis(1));

	for key in 0..1_000u64 {
		cache.set(key % 200, &[key as u8; 100], None).expect("set");
		std::thread::sleep(Duration::from_millis(2));
	}

	let kicks = cache.hybrid_stats().idle_kicks;
	let spells = cache.status.gate().test_idle_spells.load(std::sync::atomic::Ordering::Relaxed);

	assert!(kicks >= 1, "the first set after the idle spell kicked nothing");
	assert!(kicks <= spells, "{kicks} kicks for {spells} idle spells");
}

/// `MakeRoom` is for the policy worker alone, and the `Set` that grew a
/// `Placement` did not grow the event.
#[test]
fn make_room_goes_to_the_policy_worker_only() {
	use crate::worker::Events;

	let bit = WorkerEvent::MakeRoom(7).mask_bit();

	assert_eq!(bit, Events::MAKE_ROOM);
	assert_ne!(Events::POLICY_WORKER & bit, 0, "the policy worker takes it");
	assert_eq!(Events::TTL_WORKER & bit, 0, "the TTL worker does not");

	for other in [
		Events::GET, Events::SET, Events::DEL, Events::EXPIRE, Events::TTL, Events::WIPE,
		Events::RESIZE, Events::RESIZE_FAST_TIER, Events::RESIZE_LARGE_FAST_TIER, Events::RESIZE_SIZE_THRESHOLD,
		Events::SHUTDOWN, Events::AUDIT,
	] {
		assert_eq!(other & bit, 0, "its bit is its own");
	}
}

#[test]
fn worker_event_stays_40_bytes() {
	assert_eq!(std::mem::size_of::<WorkerEvent>(), 40);
}
