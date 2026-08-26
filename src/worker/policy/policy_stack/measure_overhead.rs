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
//! Method: build ONE stack and sample RSS at n and at 2n while inserting
//! monotonically, then take `(rss(2n) - rss(n)) / n`. Two properties matter.
//! Nothing is ever freed, so jemalloc never has retained-but-unreturned pages
//! to confound the delta -- which is why this is one growing stack rather than
//! two separately-measured ones. And differencing cancels both the process
//! baseline and the stack's fixed allocations, leaving the marginal per-object
//! cost, which is the quantity the constants are supposed to hold.
//!
//! RSS rather than jemalloc's `stats.allocated`: this jemalloc is built without
//! `--enable-stats`, and resident pages are the more honest quantity anyway --
//! they include the size-class rounding and table slack a request-side counter
//! misses. Page granularity over a million objects is well under 0.1 B/object.
//!
//! Ignored by default: allocates gigabytes and takes ~30 s.
//! Run with `cargo test --features <policies> -- --ignored --nocapture measure_`.

use crate::worker::policy::policy_stack::init_policy_stack;
use crate::PaperPolicy;
use core::str::FromStr;

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
fn allocated_bytes() -> u64 {
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
