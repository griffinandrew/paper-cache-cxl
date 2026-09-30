/*
 * Copyright (c) Kia Shakiba
 *
 * This source code is licensed under the GNU AGPLv3 license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! S5a: M -- the cache's own DRAM metadata as its policy worker publishes it
//! (`PaperCache::dram_metadata`: the object map's structures, the policy
//! stack's, one value header per live object) -- against the allocator, for
//! a whole cache.
//!
//! Under `measured_accounting` the crate's global allocator counts the live
//! bytes it hands out per pool (`numa_alloc::measured`), in the unit M is in
//! (jemalloc's usable size). NODE_FAST is every Rust allocation in DRAM,
//! process-wide. So between two QUIESCENT readings of one cache that pool
//! moves by
//!
//! ```text
//!   delta M                 the map's structures, the stack's, the headers
//! + delta P                 the fast tier's values (phys), unless
//!                           segregated_value_arena gives them a pool of their
//!                           own (NODE_FAST_VALUES), where they are checked
//!                           against P instead
//! + everything else the process allocated in DRAM in between
//! ```
//!
//! and this test holds the third line to ZERO between the two drained
//! endpoints (the warmed-up cache and the filled one), and to a whole number
//! of one named term's blocks at every step between them. Each term that
//! could make it non-zero, and why it is not:
//!
//!   * CHANNEL BLOCKS. A crossbeam list channel holds exactly its tail block
//!     once everything sent has been read (it frees a block when its last
//!     slot is read, and allocates the next when its last slot is written).
//!     The warm-up sends on every channel -- the policy worker's, the TTL
//!     worker's, and both migration consumers', which are chosen by key hash:
//!     it shrinks the fast tier to one byte, which demotes every fast value
//!     (at least 20, and none left fast, both checked, so both consumers get
//!     one but for a 2^-19 chance), and restores it -- so none is first
//!     allocated during the measurement. (A run whose warm-up migrated only a few keys once
//!     showed the other consumer's first block, 768 B, mid-measurement.) The policy worker and the migration consumers
//!     read at once, so a quiescent point has read everything of theirs. The
//!     TTL WORKER does not: with no expiry due it drains its channel once a
//!     second, so between two steps its channel can hold more or fewer blocks
//!     than it did, and a step's pool moves by that many `TTL_BLOCK`s more
//!     than M and P explain. So at every step the difference is held to a
//!     whole number of `TTL_BLOCK`s, and at the two endpoints -- each taken
//!     after a wait the TTL worker drains in -- to exactly zero.
//!   * THE POLICY WORKER'S REUSABLE BUFFERS -- its event batch and its
//!     observation list -- grow only with the largest batch a pass drains.
//!     Every step here is one operation followed by quiescence, so no pass
//!     drains more than the one event the warm-up already made them hold.
//!     Each stack's `migrations` vector is taken whole by every drain, empty
//!     at quiescence.
//!   * THE MIGRATION PIPELINE'S IN-FLIGHT TABLE (128 KiB) is built by the
//!     first set, in the warm-up. The worker's per-shard DashMap reading is
//!     sized when the worker is built.
//!   * PARKING_LOT'S TABLE OF PARKED THREADS. `parking_lot_core` keeps one
//!     process-global hash table of parked threads, created by the first park
//!     on (or unpark from) any lock built on it -- a contended DashMap shard, a
//!     parking_lot mutex -- at 16 buckets of 64 B plus its 32-byte header,
//!     1,056 B, and grown (the old one kept, never freed) once more than a
//!     third as many threads have parked as it has buckets. Whether the
//!     process's first contended lock fell in the warm-up or in the middle of
//!     the measurement was timing: with S5's gate it fell mid-measurement in
//!     about half the DashMap-build runs (a step 1,056 B over M). Built here
//!     before the first cache, and grown past what the cache's threads need:
//!     sixteen threads parked on one condvar at once.
//!   * OUTPUT. A test's output is captured into a heap buffer that every
//!     spawned thread inherits, and the policy worker prints (DIVERGE,
//!     MIGSTATS); `set_output_capture(None)` sends this test's and its
//!     cache's threads' output straight to stderr, which allocates nothing.
//!   * THE TEST ITSELF allocates nothing between two readings: values are
//!     slices of one buffer, a `get`'s copy is dropped before the reading, and
//!     the step log is preallocated. One `#[test]` in the binary, so libtest
//!     starts and reports no other test meanwhile.
//!
//! Every step is checked: a set of a new key, and on some steps a get (LRU
//! and LFU promote, CLOCK references) or a delete, so the run fills the fast
//! tier, demotes, evicts and frees. The steps that moved M are logged by
//! part -- a DashMap shard's table doubling, a slab chunk, an index doubling,
//! a merged bucket array's growth -- and printed at the end; in the DashMap
//! builds every step of the map part is checked to be made of a hashbrown
//! table's own steps.
//!
//! A binary of its own: the pool is process-global. Run with
//!   cargo +nightly test --release --features \
//!     server,lru_compact_hybrid_cache,measured_accounting --test dram_metadata_identity
//! and with merged_object_store, thin_header or segregated_value_arena added.
#![cfg(all(feature = "hybrid_cache_common", feature = "measured_accounting"))]
#![feature(internal_output_capture)]

mod common;

use std::time::{Duration, Instant};

use paper_cache::{
    numa_alloc::{measured, NODE_FAST},
    phys, CacheTierSize, DramMetadata, GateConfig, MetadataModel, PaperCache, PaperPolicy, Tier,
    TieredBuffer,
};

type Cache = PaperCache<u64, TieredBuffer>;

const QUIESCE_TIMEOUT: Duration = Duration::from_secs(20);

/// Keys the warm-up sets: past the fast tier, so both migration consumers
/// have demoted something before the measurement starts.
const WARM: u64 = 250;

/// New keys per design, after the warm-up's.
const KEYS: u64 = 700;

/// 200 KB of cache over values of 100..700 B: it starts evicting some 390
/// keys in. The fast tier holds a third of it, so it demotes from key ~150.
/// (400 KB until S5: its ~750 keys would pass the per-object key ceiling,
/// 64 KiB / omega -- 585 keys at the split build's 112 B.)
const MAX_SIZE: u64 = 200 * 1024;
const FAST: u64 = 64 * 1024;

/// Whether the fast values are in NODE_FAST (and so in the pool this test
/// reads) or in a pool of their own.
const VALUES_IN_THE_POOL: bool = !cfg!(feature = "segregated_value_arena");

/// One block of a crossbeam list channel of `WorkerEvent`s (40 bytes):
/// a `next` pointer and 31 slots of the message and an 8-byte state,
/// `8 + 31 * 48 = 1496` bytes, jemalloc's 1536 class. What the TTL worker's
/// channel holds one more or one fewer of whenever a step's events crossed a
/// block boundary it has not drained yet.
const TTL_BLOCK: i64 = 1536;

/// Longer than the TTL worker's idle sleep (1 s): after it, the TTL worker has
/// drained everything sent before it.
const TTL_DRAIN: Duration = Duration::from_millis(1_200);

fn len_of(key: u64) -> usize {
    100 + (key * 7_919 % 600) as usize
}

/// One quiescent reading.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Reading {
    /// `measured::allocated(NODE_FAST)`: every live DRAM byte in the process.
    pool: i64,
    /// P, exact.
    p: i64,
    m: DramMetadata,
    /// The live keys, walked in the map.
    live: u64,
    /// The modelled reservation, `L * omega`.
    model: u64,
}

impl Reading {
    /// What the pool must have moved by since `from`.
    fn explained_since(&self, from: &Reading) -> i64 {
        let m = self.m.total() as i64 - from.m.total() as i64;
        let p = if VALUES_IN_THE_POOL { self.p - from.p } else { 0 };

        m + p
    }
}

/// Waits until the worker has handled everything -- its stack tracks exactly
/// the keys the map holds of `0..keys` -- nothing is migrating, and the pool,
/// P and M hold still for five polls; then returns the reading. Reads only
/// atomics and the map: nothing here allocates.
fn quiesce(cache: &Cache, keys: u64) -> Reading {
    let deadline = Instant::now() + QUIESCE_TIMEOUT;
    let mut last = None;
    let mut stable = 0;

    loop {
        let pool = measured::allocated(NODE_FAST) as i64;
        let s = cache.hybrid_stats();
        let live = (0..keys).filter(|key| cache.tier_of(key).is_some()).count() as u64;
        let r = Reading {
            pool,
            p: phys::fast_bytes_signed(),
            m: cache.dram_metadata(),
            live,
            model: s.fast_metadata_bytes,
        };

        let settled = s.fast_objects + s.slow_objects == live
            && phys::pending_migrations() == (0, 0)
            && cache.migrations_in_flight() == 0
            && s.dram_metadata_bytes == r.m.total();

        stable = if settled && last == Some(r) { stable + 1 } else { 0 };
        last = Some(r);

        if stable >= 5 {
            return r;
        }

        assert!(
            Instant::now() < deadline,
            "never quiesced over keys 0..{keys}: {r:?}, stack {}, pending {:?}",
            s.fast_objects + s.slow_objects,
            phys::pending_migrations(),
        );

        std::thread::sleep(Duration::from_micros(300));
    }
}

/// A step whose M moved.
#[derive(Clone, Copy, Debug)]
struct Step {
    at: u64,
    map: i64,
    stack: i64,
    headers: i64,
}

/// One step: quiesce, and hold the pool's move to M's (and P's), up to whole
/// TTL-channel blocks. Returns how many blocks the TTL channel moved by.
fn step(cache: &Cache, keys: u64, policy: PaperPolicy, previous: &mut Reading, log: &mut Vec<Step>) -> i64 {
    let r = quiesce(cache, keys);
    let unexplained = (r.pool - previous.pool) - r.explained_since(previous);

    assert_eq!(
        unexplained % TTL_BLOCK,
        0,
        "{policy} at key {}: the DRAM pool moved by {} B, where M moved by {} B ({:?} -> {:?}){} \
         -- {unexplained} B unexplained, not a whole number of TTL-channel blocks",
        keys - 1,
        r.pool - previous.pool,
        r.m.total() as i64 - previous.m.total() as i64,
        previous.m,
        r.m,
        match VALUES_IN_THE_POOL {
            true => format!(" and P by {} B", r.p - previous.p),
            false => String::new(),
        },
    );

    if r.m != previous.m && log.len() < log.capacity() {
        log.push(Step {
            at: keys - 1,
            map: r.m.map as i64 - previous.m.map as i64,
            stack: r.m.stack as i64 - previous.m.stack as i64,
            headers: r.m.headers as i64 - previous.m.headers as i64,
        });
    }

    *previous = r;

    unexplained / TTL_BLOCK
}

/// The steps a hashbrown table of `(u64, Object)` buckets -- a DashMap shard's
/// table -- can take: its first table (4 buckets), and every doubling after
/// it. Its layout is `16 * b` of buckets and `b + 16` control bytes, aligned
/// to 16 (hashbrown's `calculate_layout_for`, a 16-byte bucket).
#[cfg(not(feature = "merged_object_store"))]
fn hashbrown_steps() -> Vec<i64> {
    let table = |b: usize| paper_cache::meta::usable(17 * b + 16, 16) as i64;
    let mut steps = vec![table(4)];

    for shift in 2..24 {
        steps.push(table(2 << shift) - table(1 << shift));
    }

    steps
}

/// One design's run: the warmed-up cache's reading, the filled one's, and the
/// steps checked.
fn run(policy: PaperPolicy, value: &[u8], log: &mut Vec<Step>) -> (Reading, Reading, u64) {
    log.clear();

    // The per-object metadata model (S5): what this test holds M to is the
    // allocator, not the model, and a 64 KiB tier is smaller than the cache's
    // own structures -- under the measured model's key ceiling it would refuse
    // every key.
    let mut gate = GateConfig::default();
    gate.metadata_model = MetadataModel::PerObject;

    let cache = Cache::new_with_gate(MAX_SIZE, CacheTierSize::Bytes(FAST), policy, gate).expect("a tiered cache");

    // The warm-up, one operation at a time as the measurement will be: every
    // channel used, the worker's buffers and the in-flight table made, both
    // migration consumers used (the sets overflow the fast tier, and the gets
    // promote in every design that promotes, the ones that admit slow
    // included), a miss, a delete.
    let mut keys = 0u64;

    while keys < WARM {
        cache.set(keys, &value[..len_of(keys)], None).expect("set");
        keys += 1;
        quiesce(&cache, keys);
    }

    for key in (0..WARM).step_by(3) {
        drop(cache.get(&key));
        quiesce(&cache, keys);
    }

    drop(cache.get(&u64::MAX));

    // Every fast value demoted: a migration on both consumers.
    let fast = (0..WARM).filter(|key| cache.tier_of(key) == Some(Tier::Fast)).count();
    assert!(fast >= 20, "{policy}: only {fast} fast values to demote in the warm-up");

    cache.set_fast_tier_size(CacheTierSize::Bytes(1)).expect("shrink");
    quiesce(&cache, keys);

    let left = (0..WARM).filter(|key| cache.tier_of(key) == Some(Tier::Fast)).count();
    assert_eq!(left, 0, "{policy}: the shrink left {left} values fast, so the warm-up cannot vouch for the consumers");

    cache.set_fast_tier_size(CacheTierSize::Bytes(FAST)).expect("restore");
    quiesce(&cache, keys);

    // A live key: the 2Q and S3-FIFO designs evict from their admission
    // queue early, so an early key may be gone.
    let victim = (0..WARM).find(|key| cache.tier_of(key).is_some()).expect("a live key");
    cache.del(&victim).expect("del of a live key");

    // The first endpoint: the TTL worker has drained everything so far.
    std::thread::sleep(TTL_DRAIN);
    let empty = quiesce(&cache, keys);
    let mut previous = empty;
    let mut deleted = 0u64;
    let mut checked = 0u64;
    let mut ttl_blocks = 0i64;

    for _ in 0..KEYS {
        let key = keys;

        cache.set(key, &value[..len_of(key)], None).expect("set");
        keys += 1;
        ttl_blocks += step(&cache, keys, policy, &mut previous, log);
        checked += 1;

        if key % 5 == 0 {
            drop(cache.get(&(key / 2)));
            ttl_blocks += step(&cache, keys, policy, &mut previous, log);
            checked += 1;
        }

        // Only a key known live: a delete of an absent key reserves in its
        // shard and tells the worker nothing (`meta::ShardState`), which the
        // worker's full re-read catches within 100 ms, not within a step.
        if key % 11 == 0 && cache.tier_of(&(key / 3)).is_some() {
            cache.del(&(key / 3)).expect("del of a live key");
            deleted += 1;
            ttl_blocks += step(&cache, keys, policy, &mut previous, log);
            checked += 1;
        }
    }

    // The second endpoint, drained the same way; whatever TTL-channel blocks
    // the steps counted are gone again.
    std::thread::sleep(TTL_DRAIN);
    ttl_blocks += step(&cache, keys, policy, &mut previous, log);

    let filled = previous;
    let stats = cache.hybrid_stats();

    assert_eq!(ttl_blocks, 0, "{policy}: the TTL channel's blocks did not return to one");

    assert!(stats.demotions > 0, "{policy}: the run never demoted");
    assert!(stats.evictions > 0, "{policy}: the run never evicted");
    assert!(deleted > 0, "{policy}: the run never deleted");

    // Warmed-up to filled, in one comparison, EXACTLY: at both endpoints every
    // channel holds its tail block and nothing else.
    assert_eq!(
        filled.pool - empty.pool,
        filled.explained_since(&empty),
        "{policy}: from the warmed-up cache to the filled one",
    );

    drop(cache);

    (empty, filled, checked)
}

/// Parks sixteen threads on one parking_lot condvar at once: creates
/// `parking_lot_core`'s table of parked threads and grows it to 64 buckets,
/// which the cache's own threads never outgrow. The table is kept for the
/// process's life; everything else here is freed when the threads are joined.
fn warm_parking_lot() {
    const THREADS: usize = 16;

    let pair = std::sync::Arc::new((parking_lot::Mutex::new(0usize), parking_lot::Condvar::new()));

    let threads: Vec<_> = (0..THREADS)
        .map(|_| {
            let pair = pair.clone();

            std::thread::spawn(move || {
                let (lock, arrived) = &*pair;
                let mut count = lock.lock();

                *count += 1;
                arrived.notify_all();

                while *count < THREADS {
                    arrived.wait(&mut count);
                }
            })
        })
        .collect();

    for thread in threads {
        thread.join().expect("a parking thread");
    }
}

/// Whether `step` is a sum of at most `terms` of `legal`: one step can hold a
/// set's own shard's growth and those of the shards its evictions looked a
/// victim up in (`erase`'s `entry` reserves too).
#[cfg(not(feature = "merged_object_store"))]
fn decomposes(step: i64, legal: &[i64], terms: usize) -> bool {
    step == 0 || (terms > 0 && legal.iter().any(|&l| l <= step && decomposes(step - l, legal, terms - 1)))
}

#[test]
fn the_dram_pool_moves_by_exactly_m_between_quiescent_points() {
    // See the module doc: the cache's threads inherit this.
    let _ = std::io::set_output_capture(None);

    let mut designs = vec![
        PaperPolicy::LruCompactHybrid,
        PaperPolicy::FifoCompactHybrid,
        PaperPolicy::ClockCompactHybrid,
        PaperPolicy::LfuCompactHybrid,
    ];

    // The DashMap builds host every design; a ghost filter and
    // the recency-plus-frequency chain besides. Not the faithful S3-FIFO's
    // exact ghost (the stack test holds it to the allocator): its fast-small
    // variant's DRAM queue has no ceiling, so a shrink moves nothing and its
    // first migrations -- a consumer channel's first 768 B block each -- come
    // mid-measurement, and its slow-small variant has nothing fast to warm
    // the consumers with.
    if !cfg!(feature = "merged_object_store") {
        designs.extend([
            PaperPolicy::TwoQGhostCompactHybrid(0.25),
            PaperPolicy::LruLfuCompactHybrid(3),
        ]);
    }

    let value = vec![0xA5u8; 1024];
    let mut log: Vec<Step> = Vec::with_capacity(4 * KEYS as usize);

    // Each design in a child process of its own.
    common::each_alone(module_path!(), "the_dram_pool_moves_by_exactly_m_between_quiescent_points", designs, |policy| {
        // See the module doc: parking_lot's table of parked threads, before any reading.
        warm_parking_lot();

        let (empty, filled, checked) = run(policy, &value, &mut log);

        #[cfg(not(feature = "merged_object_store"))]
        {
            let legal = hashbrown_steps();

            for step in log.iter().filter(|s| s.map != 0) {
                assert!(
                    decomposes(step.map, &legal, 3),
                    "{policy} at key {}: the map part moved by {} B, which is no sum of up to three \
                     hashbrown table steps ({legal:?})",
                    step.at,
                    step.map,
                );
            }

            assert!(
                log.iter().any(|s| s.map > legal[0]),
                "{policy}: no shard table doubled past its first 4 buckets: {log:?}",
            );
        }

        let steps = |part: fn(&Step) -> i64| {
            let mut sizes: Vec<i64> = log.iter().map(part).filter(|&d| d != 0).collect();
            sizes.sort_unstable();
            sizes.dedup();
            sizes
        };

        // The largest single step of the map part, and the key it came at.
        let largest = log.iter().filter(|s| s.map != 0).max_by_key(|s| s.map).map(|s| (s.map, s.at));

        eprintln!(
            "S5a {policy}: {checked} steps held exactly; empty {:?} -> filled {:?} at {} live; \
             M {:.1} B/object against the model's {:.1}; M's steps by part: map {:?} (largest \
             {:?} as (bytes, key)), stack {:?}, headers {:?}",
            empty.m,
            filled.m,
            filled.live,
            filled.m.total() as f64 / filled.live as f64,
            filled.model as f64 / filled.live as f64,
            steps(|s| s.map),
            largest,
            steps(|s| s.stack),
            steps(|s| s.headers),
        );
    });
}
