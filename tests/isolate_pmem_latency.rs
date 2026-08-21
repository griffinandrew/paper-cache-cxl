#![feature(allocator_api)]

// Isolates the slow-tier allocator (`SlowObjects`, node-1-bound jemalloc arenas)
// from all cache logic -- no PaperCache, no worker threads, no policy stack,
// no LRU/demotion decisions. Just raw allocate/write/free cycles at the
// crate's real demotion size (~16 KB) and volume, matching how real objects
// are physically written to the slow tier. If the same multi-millisecond
// stalls show up here, the anomaly is in the allocator/NUMA path itself, not
// in `lru_hybrid_cache`'s logic sitting on top of it.
//
// Run with: cargo +nightly test --release --test isolate_pmem_latency
//   --features key_value_pmem -- --ignored --nocapture

use std::time::Instant;
use paper_cache::numa_alloc::SlowObjects;

const OBJECT_SIZE: usize = 16 * 1024; // matches the crate's real trace avg (~16 KB)
const TOTAL_ADMISSIONS: usize = 1_000_000; // sized so each process launch takes ~1s, for repeated-process testing
const SURVIVAL_RATE: f64 = 0.10; // ~10% survive, matching real demotion-survivor pattern
const STALL_THRESHOLD_NS: u128 = 1_000_000; // 1ms -- anything above this gets logged

#[test]
#[ignore]
fn isolate_pmem_latency() {
    assert!(paper_cache::numa_alloc::init(), "arena pool must build");

    let layout = std::alloc::Layout::from_size_align(OBJECT_SIZE, 8).unwrap();

    let mut live: Vec<Box<[u8], SlowObjects>> = Vec::new();
    let mut latencies_ns: Vec<u128> = Vec::with_capacity(TOTAL_ADMISSIONS);
    let mut stalls: Vec<(usize, u128)> = Vec::new();

    let start = Instant::now();

    for i in 0..TOTAL_ADMISSIONS {
        let op_start = Instant::now();

        // Allocate + touch every page (matches a real write, not just a
        // reservation -- TBB/UMF only pay real cost once pages are dirtied).
        let mut buf: Box<[u8], SlowObjects> = Box::new_in([0u8; OBJECT_SIZE], SlowObjects);
        for page in buf.chunks_mut(4096) {
            page[0] = (i % 256) as u8;
        }
        live.push(buf);

        // Scattered ~90% eviction once the live set exceeds a small working
        // window, mirroring real demotion-survivor scatter rather than a
        // clean FIFO drain.
        if live.len() > 200 {
            let mut idx = 0;
            live.retain(|_| {
                idx += 1;
                // Keep roughly SURVIVAL_RATE of entries, spread out rather
                // than clustered at the head or tail.
                (idx * 7 + i) % 100 < (SURVIVAL_RATE * 100.0) as usize
            });
        }

        let elapsed = op_start.elapsed().as_nanos();
        latencies_ns.push(elapsed);
        if elapsed > STALL_THRESHOLD_NS {
            stalls.push((i, elapsed));
        }

    }

    latencies_ns.sort_unstable();
    let n = latencies_ns.len();
    let p50 = latencies_ns[n / 2];
    let p99 = latencies_ns[(n as f64 * 0.99) as usize];
    let p999 = latencies_ns[(n as f64 * 0.999) as usize];
    let p100 = latencies_ns[n - 1];
    let avg = latencies_ns.iter().sum::<u128>() / n as u128;

    eprintln!("\n=== isolate_pmem_latency results ===");
    eprintln!("total ops: {}", n);
    eprintln!("avg: {} ns", avg);
    eprintln!("p50: {} ns", p50);
    eprintln!("p99: {} ns", p99);
    eprintln!("p999: {} ns", p999);
    eprintln!("p100 (max): {} ns", p100);
    eprintln!("stalls over {}ns: {}", STALL_THRESHOLD_NS, stalls.len());
    if !stalls.is_empty() {
        let worst = stalls.iter().max_by_key(|(_, ns)| *ns).unwrap();
        eprintln!("worst stall: op #{}, {} ns ({:.1} ms)", worst.0, worst.1, worst.1 as f64 / 1_000_000.0);
        eprintln!("first 10 stalls: {:?}", &stalls[..stalls.len().min(10)]);
    }
    eprintln!("wall clock: {:.1}s", start.elapsed().as_secs_f64());
}
