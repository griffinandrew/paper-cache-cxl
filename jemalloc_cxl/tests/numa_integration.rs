//! Integration tests that need a real second NUMA node to say anything
//! meaningful (as opposed to `src/*.rs`'s own unit tests, which only need
//! *a* working jemalloc instance and always run against node 0). Each test
//! checks for node 1 at runtime and skips gracefully if it's absent, rather
//! than using `#[ignore]` -- so the suite still exercises real placement
//! end-to-end on any machine that actually has a second node, while still
//! being safe to run unconditionally in CI on a single-node box.
#![feature(allocator_api)]

use jemalloc_cxl::{create_cxl_arena, CxlAllocator, CxlArenaConfig, NumaPolicy};

const TARGET_NODE: u32 = 1;

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

/// Reads `/proc/self/numa_maps`, returning the resident-page count reported
/// for `node` on whichever line's address range contains `addr` (or `None`
/// if no matching `N<node>=` field is present in that line).
fn resident_pages_on_node(addr: usize, node: u32) -> Option<u64> {
    let contents = std::fs::read_to_string("/proc/self/numa_maps").ok()?;

    let mut best: Option<&str> = None;
    for line in contents.lines() {
        let start_hex = line.split_whitespace().next()?;
        let Ok(start) = usize::from_str_radix(start_hex, 16) else {
            continue;
        };
        if start <= addr {
            best = Some(line);
        } else {
            break;
        }
    }

    let line = best?;
    let field = format!("N{node}=");
    let value = line.split_whitespace().find_map(|tok| tok.strip_prefix(&field))?;
    value.parse().ok()
}

#[test]
fn bind_strict_places_touched_pages_on_the_target_node() {
    skip_if_no_node1!();

    let arena = create_cxl_arena(CxlArenaConfig::new(TARGET_NODE, NumaPolicy::BindStrict))
        .expect("arena creation on node 1 should succeed");
    let alloc = CxlAllocator::new(arena);

    const LEN: usize = 4 * 1024 * 1024; // 4M u64 = 32 MiB, several pages
    let mut v: Vec<u64, CxlAllocator> = Vec::with_capacity_in(LEN, alloc);
    for i in 0..LEN {
        v.push(i as u64);
    }
    assert_eq!(v.iter().sum::<u64>(), (0..LEN as u64).sum());

    let addr = v.as_ptr() as usize;
    let resident = resident_pages_on_node(addr, TARGET_NODE)
        .expect("numa_maps should report resident pages on the target node");
    assert!(resident > 0, "expected some pages resident on node {TARGET_NODE}, found none");
}

#[test]
fn preferred_places_touched_pages_on_the_target_node_when_available() {
    skip_if_no_node1!();

    let arena = create_cxl_arena(CxlArenaConfig::new(TARGET_NODE, NumaPolicy::Preferred))
        .expect("arena creation on node 1 should succeed");
    let alloc = CxlAllocator::new(arena);

    const LEN: usize = 1024 * 1024; // 1M u64 = 8 MiB
    let mut v: Vec<u64, CxlAllocator> = Vec::with_capacity_in(LEN, alloc);
    for i in 0..LEN {
        v.push(i as u64);
    }

    let addr = v.as_ptr() as usize;
    let resident = resident_pages_on_node(addr, TARGET_NODE)
        .expect("numa_maps should report resident pages on the target node");
    assert!(resident > 0, "expected some pages resident on node {TARGET_NODE}, found none");
}

#[test]
fn thread_arena_guard_routes_implicit_allocations_to_the_target_node() {
    skip_if_no_node1!();

    use jemalloc_cxl::ThreadArenaGuard;

    let arena = create_cxl_arena(CxlArenaConfig::new(TARGET_NODE, NumaPolicy::BindStrict))
        .expect("arena creation on node 1 should succeed");

    let _guard = ThreadArenaGuard::enter(arena).expect("enter should succeed");

    // An ordinary, implicit allocation -- no CxlAllocator involved -- should
    // still land on the guarded arena/node.
    let mut v: Vec<u64> = Vec::with_capacity(1024 * 1024);
    for i in 0..(1024 * 1024u64) {
        v.push(i);
    }
    assert_eq!(v.len(), 1024 * 1024);

    // Checked while `v` is still alive and mapped -- once dropped, jemalloc
    // may or may not have returned the extent to the OS yet (depends on its
    // own internal extent-retention/decay heuristics for this arena), so
    // asserting on numa_maps after the drop would be flaky.
    let addr = v.as_ptr() as usize;
    let resident = resident_pages_on_node(addr, TARGET_NODE);
    assert!(
        resident.is_some_and(|n| n > 0),
        "expected the guarded thread's implicit Vec allocation to be resident on node {TARGET_NODE}"
    );
}
