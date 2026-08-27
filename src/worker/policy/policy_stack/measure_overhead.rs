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
/// allocator. Not the `global_hashtable_pmem` variant: copying the wrong cfg
/// arm here once put the map in PMEM and measured a different structure
/// entirely.
///
/// Same rules as every other measurement in this module: jemalloc
/// `stats.allocated` rather than RSS, ONE point per process, and the caller
/// samples at POWERS OF TWO.
#[cfg(not(any(feature = "global_hashtable_pmem", feature = "key_pmem_value_pmem")))]
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
		let buf = crate::TieredBuffer::new_fast(&value);
		map.insert(key, crate::object::Object::new(key, buf, None));
	}
	let after = allocated_bytes();
	let held = map.len() as u64;
	core::hint::black_box(&map);
	println!("MEASURED_MAP {} {} {} {}", n, vsize, after.saturating_sub(base), held);
}

/// Exact struct layout behind the measured object-map row.
///
/// The measured 96 B/object for the DashMap row is an ALLOCATION figure; this
/// prints the sizes it is built from, so the share that is the inline key and
/// expiry -- both of which `base_size` ALREADY counts -- can be separated from
/// the container overhead proper. Adding the whole 96 on top of `base_size`
/// would double-charge whatever part of it `base_size` covers.
#[test]
#[ignore]
fn print_row_layout() {
	use core::mem::size_of;
	type Obj = crate::object::Object<u64, crate::TieredBuffer>;
	println!("LAYOUT HashedKey                {}", size_of::<crate::HashedKey>());
	println!("LAYOUT key u64                  {}", size_of::<u64>());
	println!("LAYOUT ExpireTime               {}", size_of::<crate::object::ExpireTime>());
	println!("LAYOUT Arc<TieredBuffer> ptr    {}", size_of::<std::sync::Arc<crate::TieredBuffer>>());
	println!("LAYOUT TieredBuffer             {}", size_of::<crate::TieredBuffer>());
	println!("LAYOUT Object<u64,TieredBuffer> {}", size_of::<Obj>());
	println!("LAYOUT (HashedKey, Object) pair {}", size_of::<(crate::HashedKey, Obj)>());
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

	/// The LFU payload, which is LARGER than the 2Q/S3-FIFO one: it carries a
	/// frequency as well. That changes the sign of the comparison, because the
	/// slab saving grows with payload size while the per-bucket cost does not.
	#[derive(Clone, Copy)]
	struct LfuPayload {
		freq: u32,
		size: u32,
		tier: u8,
		dram_resident: u8,
	}

	/// LFU payload in the slot.
	#[derive(Clone, Copy)]
	struct SlotLfuA {
		key: HashedKey,
		prev: u32,
		next: u32,
		freq: u32,
		size: u32,
		tier: u8,
		dram_resident: u8,
	}

	#[test]
	#[ignore]
	fn measure_lfu_layout_point() {
		let n: u64 = match std::env::var("LAYOUT_N") {
			Ok(v) => v.parse().expect("LAYOUT_N"),
			Err(_) => return,
		};
		let which = std::env::var("LAYOUT").expect("LAYOUT");

		let base = allocated_bytes();
		let total = match which.as_str() {
			"a" => {
				let mut slots: Vec<SlotLfuA> = Vec::new();
				let mut index: HashMap<HashedKey, u32, NoHasher> = HashMap::default();
				for i in 0..n {
					let k = i.wrapping_mul(0x9E37_79B9_7F4A_7C15);
					slots.push(SlotLfuA {
						key: k, prev: u32::MAX, next: u32::MAX,
						freq: 1, size: 1024, tier: 0, dram_resident: 24,
					});
					index.insert(k, i as u32);
				}
				let after = allocated_bytes();
				core::hint::black_box((&slots, &index));
				after.saturating_sub(base)
			},
			"b" => {
				let mut slots: Vec<SlotB> = Vec::new();
				let mut index: HashMap<HashedKey, (u32, LfuPayload), NoHasher> = HashMap::default();
				for i in 0..n {
					let k = i.wrapping_mul(0x9E37_79B9_7F4A_7C15);
					slots.push(SlotB { key: k, prev: u32::MAX, next: u32::MAX });
					index.insert(
						k,
						(i as u32, LfuPayload { freq: 1, size: 1024, tier: 0, dram_resident: 24 }),
					);
				}
				let after = allocated_bytes();
				core::hint::black_box((&slots, &index));
				after.saturating_sub(base)
			},
			other => panic!("LAYOUT must be a or b, got {other}"),
		};

		println!("LAYOUT {} {} {} {:.3}", which, n, total, total as f64 / n as f64);
	}

	#[test]
	fn lfu_struct_sizes_are_what_the_comparison_assumes() {
		assert_eq!(core::mem::size_of::<SlotLfuA>(), 32);
		assert_eq!(core::mem::size_of::<LfuPayload>(), 12);
		assert_eq!(core::mem::size_of::<(HashedKey, u32)>(), 16);
		assert_eq!(core::mem::size_of::<(HashedKey, (u32, LfuPayload))>(), 24);
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

/// Layout A vs layout B on TIME, not bytes.
///
/// The byte comparison is settled. What was never established is whether B
/// costs anything on LIST-TOUCHING operations, which is the question that
/// decides LRU: `move_front` is its hot path, and unlike `mark_accessed` it
/// cannot be served from the index alone.
///
/// A prior standalone attempt reported `cur 446-541, a 429-446, b 434-572 ns`
/// and concluded there was no separation above noise, on a machine with a load
/// average of 3.83. This runs on a quiet box, interleaves the two layouts
/// within each repeat so any drift hits both equally, and reports every repeat
/// rather than a mean, so the spread is visible.
///
/// Both layouts here are real doubly-linked lists over a slab, differing only
/// in where the payload sits. Timing a structure that does not maintain the
/// links would measure nothing.
#[cfg(test)]
#[allow(dead_code)]
mod layout_timing {
	use crate::{HashedKey, NoHasher};
	use std::collections::HashMap;
	use std::time::Instant;

	const NIL: u32 = u32::MAX;

	#[derive(Clone, Copy, Default)]
	struct Payload {
		size: u32,
		tier: u8,
		dram_resident: u8,
	}

	#[derive(Clone, Copy)]
	struct SlotA {
		key: HashedKey,
		prev: u32,
		next: u32,
		payload: Payload,
	}

	#[derive(Clone, Copy)]
	struct SlotB {
		key: HashedKey,
		prev: u32,
		next: u32,
	}

	struct ListA {
		slots: Vec<SlotA>,
		index: HashMap<HashedKey, u32, NoHasher>,
		head: u32,
		tail: u32,
	}

	struct ListB {
		slots: Vec<SlotB>,
		index: HashMap<HashedKey, (u32, Payload), NoHasher>,
		head: u32,
		tail: u32,
	}

	macro_rules! impl_list {
		($name:ident, $slot:ident, $get_slot:expr) => {
			impl $name {
				fn unlink(&mut self, i: u32) {
					let (p, n) = (self.slots[i as usize].prev, self.slots[i as usize].next);
					match p {
						NIL => self.head = n,
						p => self.slots[p as usize].next = n,
					}
					match n {
						NIL => self.tail = p,
						n => self.slots[n as usize].prev = p,
					}
				}

				fn link_front(&mut self, i: u32) {
					let old = self.head;
					self.slots[i as usize].prev = NIL;
					self.slots[i as usize].next = old;
					match old {
						NIL => self.tail = i,
						o => self.slots[o as usize].prev = i,
					}
					self.head = i;
				}
			}
		};
	}

	impl_list!(ListA, SlotA, ());
	impl_list!(ListB, SlotB, ());

	impl ListA {
		fn build(n: u64) -> Self {
			let mut l = ListA {
				slots: Vec::with_capacity(n as usize),
				index: HashMap::default(),
				head: NIL,
				tail: NIL,
			};
			l.index.reserve(n as usize);
			for i in 0..n {
				let k = i.wrapping_mul(0x9E37_79B9_7F4A_7C15);
				l.slots.push(SlotA {
					key: k,
					prev: NIL,
					next: NIL,
					payload: Payload { size: 1024, tier: 0, dram_resident: 24 },
				});
				l.index.insert(k, i as u32);
				l.link_front(i as u32);
			}
			l
		}

		/// Metadata read: probe, then dereference into the slab.
		fn read_meta(&self, k: HashedKey) -> u32 {
			match self.index.get(&k) {
				Some(&i) => self.slots[i as usize].payload.size,
				None => 0,
			}
		}

		fn move_front(&mut self, k: HashedKey) {
			let Some(&i) = self.index.get(&k) else { return };
			if self.head == i {
				return;
			}
			self.unlink(i);
			self.link_front(i);
		}
	}

	impl ListB {
		fn build(n: u64) -> Self {
			let mut l = ListB {
				slots: Vec::with_capacity(n as usize),
				index: HashMap::default(),
				head: NIL,
				tail: NIL,
			};
			l.index.reserve(n as usize);
			for i in 0..n {
				let k = i.wrapping_mul(0x9E37_79B9_7F4A_7C15);
				l.slots.push(SlotB { key: k, prev: NIL, next: NIL });
				l.index.insert(k, (i as u32, Payload { size: 1024, tier: 0, dram_resident: 24 }));
				l.link_front(i as u32);
			}
			l
		}

		/// Metadata read: one probe, payload already in the bucket.
		fn read_meta(&self, k: HashedKey) -> u32 {
			match self.index.get(&k) {
				Some(&(_, p)) => p.size,
				None => 0,
			}
		}

		fn move_front(&mut self, k: HashedKey) {
			let Some(&(i, _)) = self.index.get(&k) else { return };
			if self.head == i {
				return;
			}
			self.unlink(i);
			self.link_front(i);
		}
	}

	/// Fixed pseudo-random probe order, identical for both layouts.
	fn probe_order(n: u64, probes: usize) -> Vec<HashedKey> {
		let mut out = Vec::with_capacity(probes);
		let mut x: u64 = 0x9E37_79B9_7F4A_7C15;
		for _ in 0..probes {
			x ^= x << 13;
			x ^= x >> 7;
			x ^= x << 17;
			out.push((x % n).wrapping_mul(0x9E37_79B9_7F4A_7C15));
		}
		out
	}

	#[test]
	#[ignore]
	fn measure_layout_timing() {
		let n: u64 = std::env::var("TIME_N").map(|v| v.parse().unwrap()).unwrap_or(4_194_304);
		let probes: usize =
			std::env::var("TIME_PROBES").map(|v| v.parse().unwrap()).unwrap_or(4_000_000);
		let repeats: usize = std::env::var("TIME_REPEATS").map(|v| v.parse().unwrap()).unwrap_or(3);

		let order = probe_order(n, probes);
		let mut a = ListA::build(n);
		let mut b = ListB::build(n);

		// warm both before timing either
		let mut sink = 0u64;
		for k in order.iter().take(200_000) {
			sink += a.read_meta(*k) as u64 + b.read_meta(*k) as u64;
		}
		core::hint::black_box(sink);

		println!("\nn={n} probes={probes}  (ns/op; A and B interleaved within each repeat)");
		println!("{:<10} {:>12} {:>12} {:>10}", "op", "A", "B", "B vs A");

		for r in 0..repeats {
			// --- metadata-only read ---
			let t = Instant::now();
			let mut s = 0u64;
			for k in &order {
				s += a.read_meta(*k) as u64;
			}
			let ra = t.elapsed().as_nanos() as f64 / probes as f64;
			core::hint::black_box(s);

			let t = Instant::now();
			let mut s = 0u64;
			for k in &order {
				s += b.read_meta(*k) as u64;
			}
			let rb = t.elapsed().as_nanos() as f64 / probes as f64;
			core::hint::black_box(s);

			// --- move_front: the list-touching path, LRU's hot one ---
			let t = Instant::now();
			for k in &order {
				a.move_front(*k);
			}
			let ma = t.elapsed().as_nanos() as f64 / probes as f64;

			let t = Instant::now();
			for k in &order {
				b.move_front(*k);
			}
			let mb = t.elapsed().as_nanos() as f64 / probes as f64;

			println!(
				"{:<10} {:>12.2} {:>12.2} {:>9.1}%",
				format!("read[{r}]"),
				ra,
				rb,
				(rb - ra) / ra * 100.0
			);
			println!(
				"{:<10} {:>12.2} {:>12.2} {:>9.1}%",
				format!("move[{r}]"),
				ma,
				mb,
				(mb - ma) / ma * 100.0
			);
		}
		println!();
	}
}
