//! Integration tests for `tier_allocator`.
//!
//! Run with: `cargo test` (stable Rust -- no `+nightly` needed for this
//! crate, unlike the sibling `paper-cache-cxl` crate).

use tier_allocator::TierAllocator;

/// Returns true if NUMA node 1 exists and has memory (i.e. is a plausible
/// CXL/far-tier stand-in), without assuming anything about node count on
/// whatever machine runs this test. Reads `/sys/devices/system/node/`
/// rather than shelling out to `numactl`, avoiding an extra runtime
/// dependency. Confirmed on this development machine via `numactl
/// --hardware`: node 0 (50 GB, 8 CPUs), node 1 (124 GB, 0 CPUs --
/// memory-only, a genuine CXL/far-tier stand-in).
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
fn alloc_and_readback_bytes_on_each_tier() {
    skip_if_no_node1!();

    let dram = TierAllocator::new_numa(0).expect("TierAllocator::new_numa(0)");
    let cxl = TierAllocator::new_numa(1).expect("TierAllocator::new_numa(1)");

    let mut hot = dram.alloc(1024).expect("alloc on node 0");
    hot[0] = 42;
    assert_eq!(hot[0], 42);
    assert_eq!(hot.len(), 1024);

    let mut cold = cxl.alloc(1024).expect("alloc on node 1");
    cold[0] = 100;
    assert_eq!(cold[0], 100);
    assert_eq!(cold.len(), 1024);

    // Confirm both buffers are independent -- writing to one never
    // affected the other.
    assert_eq!(hot[0], 42);
}

#[test]
fn zero_length_allocation() {
    skip_if_no_node1!();

    let tier = TierAllocator::new_numa(1).expect("TierAllocator::new_numa(1)");
    let buf = tier.alloc(0).expect("zero-length alloc should succeed");

    assert_eq!(buf.len(), 0);
    assert!(buf.is_empty());
    // Dropping `buf` here must not call into UMF at all (see
    // `TierBuffer::empty`) -- nothing to assert directly, but this
    // completing without panicking/aborting is the point.
}

#[test]
fn multiple_buffers_share_one_allocator_instance() {
    skip_if_no_node1!();

    // TierAllocator is Copy -- this is the realistic usage pattern the
    // TierBuffer redesign is actually meant for: one long-lived allocator
    // instance backing many independently-owned, independently-freed
    // buffers.
    let tier = TierAllocator::new_numa(1).expect("TierAllocator::new_numa(1)");

    let mut buffers = Vec::new();
    for i in 0..16u8 {
        let mut buf = tier.alloc(64).expect("alloc should succeed");
        buf[0] = i;
        buffers.push(buf);
    }

    for (i, buf) in buffers.iter().enumerate() {
        assert_eq!(buf[0], i as u8);
        assert_eq!(buf.len(), 64);
    }

    // Dropping all 16 buffers here must each independently free correctly.
}

#[test]
fn duplicate_produces_an_independent_copy() {
    skip_if_no_node1!();

    let tier = TierAllocator::new_numa(1).expect("TierAllocator::new_numa(1)");

    let mut original = tier.alloc(32).expect("alloc should succeed");
    original.fill(0xAB);

    let mut copy = original.duplicate(&tier).expect("duplicate should succeed");
    assert_eq!(&copy[..], &original[..]);

    // Mutating the copy must not affect the original.
    copy.fill(0xCD);
    assert!(original.iter().all(|&b| b == 0xAB));
    assert!(copy.iter().all(|&b| b == 0xCD));
}

#[test]
fn alloc_on_invalid_node_fails_gracefully() {
    // Discovered empirically (not assumed): a wildly out-of-range NUMA node
    // id does NOT make `new_numa` itself fail -- UMF's OS memory provider
    // only stores the node list/mode at construction time; it doesn't
    // validate that the node actually exists until an allocation is
    // actually attempted (the underlying `mbind()`/`mmap()` call is what
    // fails, lazily, at `alloc()` time). So the real error-path contract to
    // test is: construction succeeds, but the first `alloc()` surfaces a
    // `TierAllocError`, not a panic or crash. Doesn't need a real second
    // NUMA node, so isn't gated by the skip helper.
    let tier = TierAllocator::new_numa(9999).expect("construction itself does not validate the node");
    assert!(tier.alloc(64).is_err());
}
