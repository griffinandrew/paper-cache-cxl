//! Demo: create a CXL/NUMA-tier jemalloc arena for a user-provided node,
//! allocate a large `Vec` through it, fault in every page, and report
//! where the memory actually landed.
//!
//! Usage: `cargo +nightly run --example cxl_vec -- [numa_node] [bind|preferred]`
//! (defaults: node 1, preferred)
#![feature(allocator_api)]

use std::env;

use jemalloc_cxl::{create_cxl_arena, CxlAllocator, CxlArenaConfig, NumaPolicy};

const LEN: usize = 64 * 1024 * 1024; // 64 MiB of u64 (512 MiB total)

fn main() {
    let mut args = env::args().skip(1);

    let node: u32 = args
        .next()
        .map(|s| s.parse().expect("numa_node must be a non-negative integer"))
        .unwrap_or(1);

    let policy = match args.next().as_deref() {
        Some("bind") => NumaPolicy::BindStrict,
        Some("preferred") | None => NumaPolicy::Preferred,
        Some(other) => panic!("unknown policy {other:?}, expected \"bind\" or \"preferred\""),
    };

    println!("Creating a CXL arena bound to NUMA node {node} ({policy:?})...");
    let arena = create_cxl_arena(CxlArenaConfig::new(node, policy))
        .expect("arena creation failed -- does this node exist? (see /sys/devices/system/node/)");
    println!("Arena created: index = {}", arena.index());

    let alloc = CxlAllocator::new(arena);

    println!("Allocating a {} MiB Vec<u64> in arena {}...", (LEN * size_of::<u64>()) / (1024 * 1024), arena.index());
    let mut v: Vec<u64, CxlAllocator> = Vec::with_capacity_in(LEN, alloc);

    // Touching every element forces every page to actually fault in now,
    // rather than leaving most of the allocation as never-faulted virtual
    // address space. This matters because mbind(2) (called inside this
    // crate's extent alloc hook -- see src/extent.rs) sets the *policy*
    // governing which NUMA node a page is placed on **at the moment it is
    // first faulted in** -- it does not itself force immediate physical
    // allocation. An untouched mmap region has a NUMA policy but no actual
    // physical pages yet; only after this loop will /proc/self/numa_maps
    // (below) show real, resident pages attributed to the target node.
    for i in 0..LEN {
        v.push(i as u64);
    }

    let ptr = v.as_ptr();
    let byte_len = v.len() * size_of::<u64>();
    println!();
    println!("Vec pointer:  {ptr:p}");
    println!("Byte length:  {byte_len} ({} MiB)", byte_len / (1024 * 1024));
    println!("Arena index:  {}", arena.index());
    println!("Checksum:     {} (sanity check the memory is really readable/writable)",
        v.iter().fold(0u64, |acc, &x| acc.wrapping_add(x)));

    report_numa_maps_placement(ptr as usize, node);
}

/// Reads `/proc/self/numa_maps` and prints the one mapping line whose
/// address range contains `addr` -- a rough, best-effort placement check,
/// not a substitute for real profiling. `numa_maps` reports node placement
/// of pages that have actually been faulted in (see the comment above the
/// touch loop in `main`), aggregated per-VMA, so this line's `N<node>=`
/// counters are what confirm the `mbind` policy actually took effect.
fn report_numa_maps_placement(addr: usize, target_node: u32) {
    let Ok(contents) = std::fs::read_to_string("/proc/self/numa_maps") else {
        println!("\n(could not read /proc/self/numa_maps -- skipping placement report)");
        return;
    };

    // numa_maps lines don't carry an explicit end address, only a start --
    // each VMA's line covers [start, next_line's_start). Find the last
    // line whose start is still <= addr; that's the containing VMA.
    let mut best: Option<&str> = None;
    for line in contents.lines() {
        let Some(start_hex) = line.split_whitespace().next() else {
            continue;
        };
        let Ok(start) = usize::from_str_radix(start_hex, 16) else {
            continue;
        };
        if start <= addr {
            best = Some(line);
        } else {
            break;
        }
    }

    match best {
        Some(line) => {
            println!("\n/proc/self/numa_maps entry for this allocation:\n  {line}");
            println!(
                "\n(look for \"N{target_node}=\" above -- that's the page count actually resident on node {target_node})"
            );
        }
        None => println!("\n(no matching /proc/self/numa_maps entry found for {addr:#x})"),
    }
}
