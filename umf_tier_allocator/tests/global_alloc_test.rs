//! Integration test for `NumaAllocator` as an actual `#[global_allocator]`.
//!
//! This is a separate file from `tier_alloc_test.rs` because Rust allows at
//! most one `#[global_allocator]` per compiled binary, and installing one
//! here means this *entire test binary's* heap traffic -- including the
//! test harness's own internals, not just the bytes this file allocates
//! directly -- routes through UMF node 0. That's a meaningfully more
//! invasive test than the rest of this crate's suite, hence its own binary.
//!
//! Run with: `cargo test` (stable Rust, same as `tier_alloc_test.rs`).

use tier_allocator::NumaAllocator;

#[global_allocator]
static GLOBAL: NumaAllocator = NumaAllocator::new(0);

fn numa_node_1_available() -> bool {
    std::path::Path::new("/sys/devices/system/node/node1").exists()
}

macro_rules! skip_if_no_node1 {
    () => {
        if !numa_node_1_available() {
            eprintln!("skipping: this machine has no NUMA node 1 (single-node system)");
            return;
        }
    };
}

#[test]
fn ordinary_vec_and_box_allocations_succeed_via_numa_allocator() {
    let mut v: Vec<u64> = Vec::new();
    for i in 0..256u64 {
        v.push(i);
    }
    assert_eq!(v.len(), 256);
    assert_eq!(v[255], 255);

    let b: Box<[u8]> = Box::from(&b"hello from node 0"[..]);
    assert_eq!(&*b, b"hello from node 0");

    // Dropping both here must free correctly through NumaAllocator::dealloc.
}

#[test]
fn large_vec_growth_reallocates_correctly() {
    // Force multiple grow/realloc cycles well past any small-buffer
    // optimization, exercising NumaAllocator::alloc/dealloc repeatedly as
    // the Vec's backing allocation is replaced.
    let mut v: Vec<u64> = Vec::with_capacity(1);
    for i in 0..100_000u64 {
        v.push(i);
    }

    assert_eq!(v.len(), 100_000);
    assert_eq!(v[0], 0);
    assert_eq!(v[99_999], 99_999);
    assert_eq!(v.iter().sum::<u64>(), (0..100_000u64).sum());
}

#[test]
fn explicit_alloc_on_other_node_still_coexists_with_the_global_allocator() {
    skip_if_no_node1!();

    // The global allocator (node 0) is already in heavy use by the test
    // harness itself at this point. Confirm the explicit alloc_on(1, ...)
    // path still works independently -- proving both access patterns share
    // the registry without interfering with each other.
    let mut buffer = tier_allocator::alloc_on(1, 4096).expect("alloc_on(1, 4096) should succeed");
    buffer.fill(0x7A);
    assert!(buffer.iter().all(|&b| b == 0x7A));
    assert_eq!(buffer.len(), 4096);

    // Meanwhile ordinary node-0 allocations (via the global allocator)
    // still work fine alongside the explicit node-1 buffer.
    let v: Vec<u8> = vec![1, 2, 3, 4];
    assert_eq!(v, vec![1, 2, 3, 4]);
}
