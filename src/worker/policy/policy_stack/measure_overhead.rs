//! Measured per-object DRAM cost of each eviction stack.
//!
//! Every `*_EVICTION_STACK_DRAM_OVERHEAD` constant in `object::overhead` was
//! derived on paper -- struct fields added up by hand, with a guess for the
//! index map's load factor. This measures them instead.
//!
//! A `PolicyStack` holds only metadata: `insert` takes a key and a size, never
//! an object, so the memory a stack grows by across a batch of inserts IS its
//! own footprint.
//!
//! Method, ALLOCATED bytes only:
//!
//! - **jemalloc `stats.allocated`**, never RSS. This is size-class-rounded
//!   usable bytes -- what `malloc_usable_size` returns, and therefore the same
//!   quantity Redis reports as `used_memory`. RSS counts retained-but-freed
//!   pages, which belong in a fragmentation ratio rather than in a per-object
//!   cost, and measuring it made this disagree with itself by 20% depending on
//!   where the sample points fell.
//! - **One point per process.** These structures grow by doubling and a
//!   doubling abandons its old buffer, so two points sampled inside one process
//!   are separated by however much abandoned buffer lies between them.
//! - **Sampled at powers of two**, so every point sits at the same phase of the
//!   doubling cycle. The same policies fit at R^2 0.89-0.96 at 1/2/3/4M objects
//!   and R^2 = 1.0000 at 2^20..2^23.
//!
//! The delta is taken immediately around the insert loop in a single-threaded
//! test process with nothing else allocating between the two reads, so it is
//! not contaminated by the rest of the binary. The evidence is in the output:
//! marginal cost converges to exact integers (72.0000, 112.0007, 168.0000)
//! across four independent processes each, which a contaminated delta cannot do.
//!
//! Ignored by default: allocates gigabytes and takes ~30 s.
//! Run with `cargo test --features <policies> -- --ignored --nocapture measure_`.

use crate::worker::policy::policy_stack::init_policy_stack;
use crate::PaperPolicy;

/// Bytes jemalloc has handed to the application, from `stats.allocated`.
///
/// This is the ALLOCATED figure, not the resident one: size-class-rounded
/// usable bytes, the same quantity `malloc_usable_size` returns and therefore
/// the same quantity Redis reports as `used_memory`. Retained-but-freed pages
/// are excluded -- they belong in a fragmentation ratio, not in a per-object
/// cost. An earlier version of this measurement read RSS instead, which
/// conflated the two and inflated every constant by the growth slack of every
/// doubling structure.
///
/// The `epoch` write is required: jemalloc caches these statistics per epoch,
/// and a read without advancing it returns whatever the previous read saw.
pub(crate) fn allocated_bytes() -> u64 {
	unsafe {
		let mut e: u64 = 1;
		let mut elen = core::mem::size_of::<u64>();
		tikv_jemalloc_sys::mallctl(
			c"epoch".as_ptr(),
			&mut e as *mut u64 as *mut core::ffi::c_void,
			&mut elen as *mut usize,
			&mut e as *mut u64 as *mut core::ffi::c_void,
			core::mem::size_of::<u64>(),
		);
		let mut allocated: usize = 0;
		let mut len = core::mem::size_of::<usize>();
		let rc = tikv_jemalloc_sys::mallctl(
			c"stats.allocated".as_ptr(),
			&mut allocated as *mut usize as *mut core::ffi::c_void,
			&mut len as *mut usize,
			core::ptr::null_mut(),
			0,
		);
		assert_eq!(
			rc, 0,
			"stats.allocated unavailable -- tikv-jemalloc-sys needs features = [\"stats\"]"
		);
		allocated as u64
	}
}

/// One measurement, one process.
///
/// Both stack families grow by doubling, and a doubling frees the old buffer.
/// jemalloc retains those pages rather than returning them to the OS, so any
/// two sample points taken inside ONE process are separated by however much
/// abandoned buffer happens to sit between them. Measured in-process this gave
/// 67.7 B/object one way and 60.3 B/object another, with an R^2 of 0.90 that
/// said plainly the growth was not linear.
///
/// So each process measures exactly one point and exits, and the caller fits
/// the line across processes: every process starts with a clean heap, so a
/// point at n carries no residue from any smaller n.
///
/// The caller must also sample at POWERS OF TWO. Every structure here resizes
/// at a fixed load factor, so per-object cost genuinely oscillates between
/// packed and just-doubled, and sampling at arbitrary n mixes phases -- the
/// same three policies fitted with R^2 0.89-0.96 at 1/2/3/4M and R^2 0.9997+
/// at 2^20..2^23. Same code, same machine; only the sample points differed.
///
/// LIMITATION: `max_size` is set far above the batch so nothing evicts, which
/// makes this the cost of HOLDING n objects. Ghost-queue policies populate
/// their ghosts on eviction, so for those this measures the resident-object
/// term only and the ghost term has to be accounted separately.
#[test]
#[ignore]
fn measure_one_point() {
	let n: u64 = match std::env::var("MEASURE_N") {
		Ok(v) => v.parse().expect("MEASURE_N"),
		Err(_) => return,
	};
	let want = std::env::var("MEASURE_POLICY").expect("MEASURE_POLICY");
	// Parsed from the same string the benchmark takes, so a measured policy is
	// literally the policy the sweep runs -- including its parameters.
	let policy: PaperPolicy = want.parse().expect("policy string");

	let base = allocated_bytes();
	let mut stack = init_policy_stack(policy, u64::MAX / 4);
	for i in 0..n {
		// sizes vary across a range: the LFU family keys its buckets on
		// frequency, and an all-identical input collapses the bucket map
		stack.insert(i.wrapping_mul(0x9E37_79B9_7F4A_7C15), 64 + (i % 512) as u32);
	}
	let after = allocated_bytes();
	core::hint::black_box(&stack);
	println!("MEASURED {} {} {}", want, n, after.saturating_sub(base));
}

/// One measurement of the WHOLE cache, one process.
///
/// `measure_one_point` above measures a bare eviction stack, which is what
/// `*_EVICTION_STACK_DRAM_OVERHEAD` needs. `get_policy_overhead` is a
/// different quantity: it is added to `Object::base_size` to give the bytes a
/// cached object is CHARGED against `max_size`, so it must account for
/// everything the cache allocates per object beyond the object's own bytes --
/// the object-map row and its hashtable slot, the eviction-stack entry, and
/// the expiry entry. Only a real `PaperCache` allocates all of those.
///
/// Its current values are, by the table's own admission, "just rough estimates
/// of the number of bytes per object": hand-counted struct sizes like 48 for a
/// `HashList` entry. Every compact variant carries a flat `16 + 24` written by
/// the registration helper and never checked against anything.
///
/// Same methodology as `measure_one_point`, for the same reasons: jemalloc
/// `stats.allocated` rather than RSS, ONE point per process, and the caller
/// samples at POWERS OF TWO so every point sits at the same phase of every
/// structure's resize cycle.
///
/// The fast tier is set to `max_size` and `max_size` far above the batch, so
/// nothing evicts and nothing demotes: this is the cost of HOLDING n objects
/// entirely in DRAM, which is what the charge against `max_size` describes.
///
/// The slope across n is `value_allocation + overhead`, so the caller
/// subtracts the size-class-rounded value cost -- NOT the nominal value size.
/// `nallocx(15)` is 16, and getting that wrong is what made a 15-byte value
/// look free against a 40-byte tier earlier in this work.
#[cfg(feature = "hybrid_cache_common")]
#[test]
#[ignore]
fn measure_cache_point() {
	let n: u64 = match std::env::var("MEASURE_CACHE_N") {
		Ok(v) => v.parse().expect("MEASURE_CACHE_N"),
		Err(_) => return,
	};
	let want = std::env::var("MEASURE_CACHE_POLICY").expect("MEASURE_CACHE_POLICY");
	let vsize: usize = std::env::var("MEASURE_VALUE")
		.map(|v| v.parse().expect("MEASURE_VALUE"))
		.unwrap_or(64);
	let policy: PaperPolicy = want.parse().expect("policy string");

	// Far above the batch: nothing may evict, or this measures a steady state
	// rather than the cost of holding n.
	let max_size: crate::CacheSize = 1 << 40;
	let value = vec![0u8; vsize];

	// Warm-up: the first cache built in a process is measurably cheaper than
	// every later one, so build and drop one BEFORE taking `base`.
	{
		let warm = crate::PaperCache::<u64, crate::TieredBuffer>::new(
			max_size,
			crate::CacheTierSize::Bytes(max_size),
			policy,
		)
		.expect("warm-up cache should construct");
		for i in 0..4_096u64 {
			let _ = warm.set(i, &value, None);
		}
		std::thread::sleep(std::time::Duration::from_millis(200));
	}
	std::thread::sleep(std::time::Duration::from_millis(200));

	let base = allocated_bytes();
	let cache = crate::PaperCache::<u64, crate::TieredBuffer>::new(
		max_size,
		crate::CacheTierSize::Bytes(max_size),
		policy,
	)
	.expect("cache should construct");

	for i in 0..n {
		let key = i.wrapping_mul(0x9E37_79B9_7F4A_7C15);
		cache.set(key, &value, None).expect("set should succeed");
	}

	// The policy worker consumes inserts asynchronously; measuring before it
	// has drained would count an arbitrary prefix of the eviction-stack cost.
	let deadline = std::time::Instant::now() + std::time::Duration::from_secs(120);
	while cache.status().map(|st| st.num_objects()).unwrap_or(0) < n
		&& std::time::Instant::now() < deadline
	{
		std::thread::sleep(std::time::Duration::from_millis(20));
	}
	std::thread::sleep(std::time::Duration::from_millis(500));

	let after = allocated_bytes();

	// `after` is taken the moment the object COUNT reaches n, which is the last
	// thing the policy worker updates for a `Set` but not the last thing it
	// frees: every `set` also broadcast a `WorkerEvent` down an unbounded
	// channel, and some may still be queued here. So take a SECOND reading once
	// `allocated` has stopped falling, bounded, on a cache nothing is touching.
	//
	// What that settling is actually worth, MEASURED (release, 64-byte values,
	// lru-compact-hybrid, merged store): nothing at 2^20, 2^22, 2^23 and 2^24
	// -- under 0.1% each -- and 26 MB at 2^21. So the queue is NOT why this
	// harness reads high at small n, and the settled reading must not be
	// presented as though it were.
	//
	// The real shape is a FIXED term. Fitting the settled points 2^20..2^24
	// gives 127.43 B/object with an intercept of 70 MB (R^2 = 0.9996), against
	// `charged`, which is exactly 126.00 B/object with no intercept at all. The
	// SLOPES agree to 1.1%, and to 0.04% over 2^22..2^24 where the fixed term
	// is small. Per point the gap runs -25.7% at 2^20 down to -4.0% at 2^24,
	// purely because that ~70-98 MB is spread over more objects.
	//
	// So compare SLOPES here, never a single point. A single point at 2^20 is a
	// measurement of the fixed term, not of the per-object accounting. What the
	// fixed term IS remains unidentified: it is not the event queue, it does not
	// scale with n, and it is outside the merged store, whose own harness fits
	// 125.30 B/object with a 1.7 MB intercept over the same range.
	//
	// Every figure in the four paragraphs above was taken against the PRE-ARC
	// value representation -- a 24-byte `Object` inline in the map row, with
	// the value reclaimed by epoch. The key, length and expiry have since
	// moved into a separate refcounted header (`crate::value`), so both sides
	// of that comparison moved: `charged` changed because
	// `OBJECT_MAP_ENTRY_OVERHEAD` was fitted to the old row, and `allocated`
	// changed because there is a second allocation per object again. The
	// METHOD stands -- compare slopes, never a single point -- but re-run the
	// fit before quoting 127.43 or 126.00 as this build's numbers.
	//
	// Both readings are printed, so the original line keeps its meaning.
	let settled = {
		let deadline = std::time::Instant::now() + std::time::Duration::from_secs(120);
		let mut last = after;
		let mut stable = 0;

		loop {
			std::thread::sleep(std::time::Duration::from_millis(250));

			let now = allocated_bytes();

			// Only a FALL counts as draining. A rise is the worker's own
			// bookkeeping and must not restart the clock forever.
			stable = if now < last { 0 } else { stable + 1 };
			last = now;

			if stable >= 8 || std::time::Instant::now() >= deadline {
				break now.min(last);
			}
		}
	};

	let st = cache.status().expect("status");
	let held = st.num_objects();
	let charged = st.used_size();
	core::hint::black_box(&cache);
	// `charged` is what the cache BELIEVES it is using -- base_size plus
	// get_policy_overhead, summed over every object. `allocated` is what
	// jemalloc actually handed out. The gap between them IS the error in
	// the table, reported per point rather than inferred afterwards.
	println!(
		"MEASURED_CACHE {} {} {} {} {} {}",
		want, n, vsize, after.saturating_sub(base), held, charged,
	);
	println!(
		"MEASURED_CACHE_SETTLED {} {} {} {} {} {}",
		want, n, vsize, settled.saturating_sub(base), held, charged,
	);
}

/// The size-class-rounded cost of one value allocation, which the caller must
/// subtract from the measured slope. Printed rather than derived so the driver
/// never has to guess a jemalloc size class.
#[test]
#[ignore]
fn measure_value_class() {
	for v in [15usize, 16, 32, 64, 100, 128, 512, 1024] {
		let rounded = unsafe { tikv_jemalloc_sys::nallocx(v, 0) };
		println!("VALUE_CLASS {} {}", v, rounded);
	}
}

/// The object map alone, one point per process.
///
/// This is the missing half of the decomposition. `measure_one_point` gives the
/// eviction stack cleanly (R2 = 1.0000, a single structure with deterministic
/// growth). `measure_cache_point` gives the whole cache but is noisy enough
/// (R2 0.99, a systematic dip at 2^20) that it cannot resolve the 40 B/object
/// the compaction is supposed to save, and it reports compact and baseline as
/// costing the same 342 B/object -- which cannot be right.
///
/// Measuring the map on its own turns that into an arithmetic check:
///
///     whole cache  ==  object map  +  eviction stack
///
/// The map is identical for every policy, so if the identity holds then the
/// whole-cache figures must differ by exactly the stack difference (40 B), and
/// if it does not hold then the whole-cache measurement is what is wrong.
///
/// The map is the DEFAULT `ObjectMapRef` shape -- `DashMap` on the GLOBAL
/// allocator.
///
/// Same rules as every other measurement in this module: jemalloc
/// `stats.allocated` rather than RSS, ONE point per process, and the caller
/// samples at POWERS OF TWO.
///
/// NOT compiled under `merged_object_store`: there `ObjectMapRef` is
/// `Arc<MergedStore>`, so the `Arc<DashMap>` this builds does not typecheck at
/// all -- one E0308, and it was the only thing keeping the
/// `lru_compact_hybrid_cache,merged_object_store` pair from building its test
/// target. The merged store's own equivalent is
/// `merged_store::measure::measure_merged_store_point`, and its DashMap
/// control is `measure_dashmap_point` in the same module.
#[cfg(not(feature = "merged_object_store"))]
#[cfg(feature = "hybrid_cache_common")]
#[test]
#[ignore]
fn measure_object_map_point() {
	let n: u64 = match std::env::var("MEASURE_MAP_N") {
		Ok(v) => v.parse().expect("MEASURE_MAP_N"),
		Err(_) => return,
	};
	let vsize: usize = std::env::var("MEASURE_VALUE")
		.map(|v| v.parse().expect("MEASURE_VALUE"))
		.unwrap_or(64);

	let value = vec![0u8; vsize];

	let base = allocated_bytes();
	let map: crate::ObjectMapRef<u64, crate::TieredBuffer> = std::sync::Arc::new(
		dashmap::DashMap::with_hasher(crate::NoHasher::default()),
	);
	for i in 0..n {
		let key = i.wrapping_mul(0x9E37_79B9_7F4A_7C15);
		map.insert(key, crate::object::Object::new(key, &value, None));
	}
	let after = allocated_bytes();
	let held = map.len() as u64;
	core::hint::black_box(&map);
	println!("MEASURED_MAP {} {} {} {}", n, vsize, after.saturating_sub(base), held);
}
#[cfg(feature = "hybrid_cache_common")]

/// Exact struct layout behind the measured object-map row.
///
/// The measured 96 B/object for the DashMap row is an ALLOCATION figure; this
/// prints the sizes it is built from, so the container overhead proper can be
/// separated from what the row genuinely stores.
///
/// It used to exist to separate out the row's INLINE key and expiry -- both of
/// which `base_size` already counts, so adding the whole 96 on top of
/// `base_size` would have double-charged them. Neither is in the row any more:
/// since `crate::value` the row holds one eight-byte `TieredValue` handle, and
/// the key, the length and the expiry live in a `ValueHeader` that is its own
/// allocation. So the double-charge question moved with them, and this prints
/// the header's size too -- the row and the header are now two numbers, and
/// the accounting has to name which one it is charging.
///
/// The header size printed here is the STRUCT, not the allocation: `triomphe
/// ::Arc` puts an eight-byte strong count in front of it, and jemalloc then
/// rounds. Take the allocation figure from `measure_value_class`.
#[test]
#[ignore]
fn print_row_layout() {
	use core::mem::size_of;
	type Obj = crate::object::Object<u64, crate::TieredBuffer>;
	println!("LAYOUT HashedKey                {}", size_of::<crate::HashedKey>());
	println!("LAYOUT key u64                  {}", size_of::<u64>());
	println!("LAYOUT ExpireTime               {}", size_of::<crate::object::ExpireTime>());
	println!("LAYOUT TieredValue<u64>         {}", size_of::<crate::TieredValue<u64>>());
	println!("LAYOUT ValueHeader<u64>         {}", size_of::<crate::value::ValueHeader<u64>>());
	#[cfg(feature = "thin_header")]
	println!("LAYOUT ItemHeader<u64>          {}", size_of::<crate::value::ItemHeader<u64>>());
	println!("LAYOUT TieredBuffer (ZST shape) {}", size_of::<crate::TieredBuffer>());
	println!("LAYOUT Object<u64,TieredBuffer> {}", size_of::<Obj>());
	println!("LAYOUT Option<Object>           {}", size_of::<Option<Obj>>());
	println!("LAYOUT (HashedKey, Object) pair {}", size_of::<(crate::HashedKey, Obj)>());
}

