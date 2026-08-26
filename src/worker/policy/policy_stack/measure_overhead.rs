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

/// Layout A vs layout B for the slab designs, measured rather than derived.
///
/// Both keep a slab of list nodes and one index map. The question is which of
/// the two holds the 8-byte metadata payload (`size`, `tier`, `dram_resident`):
///
///   A -- payload in the SLAB slot; the map value is a bare `u32` slot index.
///        A metadata read is two hops (map bucket, then the slab).
///   B -- payload in the map VALUE alongside the slot index; the slab holds
///        links only. A metadata read is one hop.
///
/// B is the faster of the two on metadata-only reads and never worse on list
/// operations, so the only question is whether it costs more memory. A prior
/// standalone measurement reported the two as identical to 0.01 B/object;
/// deriving it by hand suggests B should cost slightly MORE, because it moves 8
/// bytes out of the densely-packed slab and into hash buckets that are only
/// 50-87.5% occupied, so those bytes get multiplied by 1/load.
///
/// Sampled across the load cycle, not just at powers of two: hashbrown resizes
/// at 7/8, so `n = 2^k` is immediately AFTER a doubling (~50% full) and
/// `n = 7/8 * 2^k` is immediately before one (maximum density). Any comparison
/// taken at one phase only can be an artifact of that phase.
#[cfg(test)]
#[allow(dead_code)]
mod layout_ab {
	use super::allocated_bytes;
	use crate::{HashedKey, NoHasher};
	use std::collections::HashMap;

	#[derive(Clone, Copy)]
	struct Payload {
		size: u32,
		tier: u8,
		dram_resident: u8,
	}

	/// Payload lives in the slot.
	#[derive(Clone, Copy)]
	struct SlotA {
		key: HashedKey,
		prev: u32,
		next: u32,
		size: u32,
		tier: u8,
		dram_resident: u8,
	}

	/// Payload lives in the map value; the slot holds links only.
	#[derive(Clone, Copy)]
	struct SlotB {
		key: HashedKey,
		prev: u32,
		next: u32,
	}

	#[test]
	#[ignore]
	fn measure_layout_point() {
		let n: u64 = match std::env::var("LAYOUT_N") {
			Ok(v) => v.parse().expect("LAYOUT_N"),
			Err(_) => return,
		};
		let which = std::env::var("LAYOUT").expect("LAYOUT");

		let base = allocated_bytes();
		let total = match which.as_str() {
			"a" => {
				let mut slots: Vec<SlotA> = Vec::new();
				let mut index: HashMap<HashedKey, u32, NoHasher> = HashMap::default();
				for i in 0..n {
					let k = i.wrapping_mul(0x9E37_79B9_7F4A_7C15);
					slots.push(SlotA {
						key: k,
						prev: u32::MAX,
						next: u32::MAX,
						size: 1024,
						tier: 0,
						dram_resident: 24,
					});
					index.insert(k, i as u32);
				}
				let after = allocated_bytes();
				core::hint::black_box((&slots, &index));
				after.saturating_sub(base)
			},
			"b" => {
				let mut slots: Vec<SlotB> = Vec::new();
				let mut index: HashMap<HashedKey, (u32, Payload), NoHasher> = HashMap::default();
				for i in 0..n {
					let k = i.wrapping_mul(0x9E37_79B9_7F4A_7C15);
					slots.push(SlotB { key: k, prev: u32::MAX, next: u32::MAX });
					index.insert(k, (i as u32, Payload { size: 1024, tier: 0, dram_resident: 24 }));
				}
				let after = allocated_bytes();
				core::hint::black_box((&slots, &index));
				after.saturating_sub(base)
			},
			other => panic!("LAYOUT must be a or b, got {other}"),
		};

		println!(
			"LAYOUT {} {} {} {:.3}",
			which,
			n,
			total,
			total as f64 / n as f64
		);
	}

	#[test]
	fn struct_sizes_are_what_the_comparison_assumes() {
		assert_eq!(core::mem::size_of::<SlotA>(), 24);
		assert_eq!(core::mem::size_of::<SlotB>(), 16);
		assert_eq!(core::mem::size_of::<Payload>(), 8);
		// what each map stores per live entry, before load factor
		assert_eq!(core::mem::size_of::<(HashedKey, u32)>(), 16);
		assert_eq!(core::mem::size_of::<(HashedKey, (u32, Payload))>(), 24);
	}
}
