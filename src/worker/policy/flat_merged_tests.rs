/*
 * Copyright (c) Kia Shakiba
 *
 * This source code is licensed under the GNU AGPLv3 license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! A FLAT cache over the merged store, end to end -- the paper's flat-merged
//! arm (`merged_object_store` with no hybrid feature), and in the hybrid
//! merged builds the same flat `PaperCache` over the same store. Since S4 its
//! sets are linked on the policy worker as a tiered merged cache's are
//! (`PolicyWorker::new` builds the same handle), and its deletes, reaps and
//! wipes go through the same DEAD slots and the same round trip. The
//! hand-driven tests (`s4_tests`) reach none of that through a real cache;
//! this does: a worker thread, the TTL reaper, and `wipe()`.

use std::time::Duration;

use super::test_support::each_alone;
use crate::{BufferDRAM, PaperCache, PaperPolicy};

type Cache = PaperCache<u64, BufferDRAM>;

/// Waits for `done`, up to ten seconds, kicking the worker off its idle poll.
fn wait_for(cache: &Cache, what: &str, mut done: impl FnMut() -> bool) {
	super::test_support::wait_for(what, Duration::from_secs(10), || {
		cache.status.kick_policy_worker();
		done()
	});
}

/// Every event sent before this call has been handled: two whole passes of
/// the policy worker after it.
fn quiesce(cache: &Cache) {
	let passes = cache.status.policy_worker_passes();

	wait_for(cache, "two more passes of the policy worker", || cache.status.policy_worker_passes() >= passes + 2);
}

/// Every flat order the merged store implements, through a real cache: sets
/// past the cache's size (evictions), overwrites, deletes, hits, and sets with
/// a 1 s TTL that the reaper takes; then, quiet, every charge exact and every
/// value linked (`verify_charges`: no DEAD slot left by a delete or a reap, no
/// unlinked value, `linked == len`), the status counting the store's objects,
/// and every expired value gone; then `wipe()` returns with the worker's clear
/// done -- the store, its links and the status empty at once -- and the cache
/// works after it. Red with the handle's retire a no-op (`noretire`: the
/// deletes' and reaps' DEAD slots stay) and with `wipe()` not waiting for the
/// worker (`asyncwipe`).
#[test]
fn a_flat_merged_cache_links_reaps_and_wipes_on_its_worker() {
	const MAX: u64 = 256 * 1024;
	const KEYS: u64 = 600;

	each_alone!("a_flat_merged_cache_links_reaps_and_wipes_on_its_worker", [PaperPolicy::LruCompact, PaperPolicy::FifoCompact, PaperPolicy::ClockCompact, PaperPolicy::LfuCompact], |policy| {
		let cache = Cache::new(MAX, &[policy], policy).expect("a flat cache");

		for key in 0..KEYS {
			let ttl = (key % 5 == 0).then_some(1);

			cache.set(key, &vec![key as u8; 200 + (key as usize * 37) % 700], ttl).expect("set");

			if key % 7 == 0 {
				let _ = cache.del(&(key / 2));
			}

			if key % 3 == 0 {
				let _ = cache.get(&(key / 3));
			}

			if key % 11 == 0 {
				let _ = cache.set(key / 4, &vec![0u8; 64], None);
			}
		}

		// Past every TTL: the reaper has taken the values and sent their
		// `Expire`s.
		std::thread::sleep(Duration::from_millis(2_200));
		quiesce(&cache);

		cache.objects.verify_charges(true);

		let live = cache.objects.len();

		assert!(live > 0, "{policy}: nothing left to check");
		assert_eq!(cache.status.live_num_objects(), live as u64, "{policy}: the status counts the store's objects");
		assert!(cache.status.used_size(&policy) <= MAX, "{policy}: over the cache's size when quiet");

		// The TTL keys no later set replaced with a value that has none.
		let replaced: std::collections::HashSet<u64> = (0..KEYS).filter(|key| key % 11 == 0).map(|key| key / 4).collect();

		let expired = (0..KEYS)
			.step_by(5)
			.filter(|key| !replaced.contains(key))
			.filter(|key| cache.objects.get_ref(&cache.hash_key(key)).is_some())
			.count();

		assert_eq!(expired, 0, "{policy}: values set with a 1 s TTL outlived the reaper");

		cache.wipe().expect("wipe");

		assert_eq!((cache.objects.len(), cache.objects.linked()), (0, 0), "{policy}: the store right after wipe()");
		assert_eq!(
			(cache.status.live_num_objects(), cache.status.used_size(&policy)),
			(0, 0),
			"{policy}: the status right after wipe()",
		);

		cache.set(7, &[7; 100], None).expect("a set after the wipe");
		assert_eq!(cache.get(&7).expect("a get after the wipe"), vec![7; 100]);

		quiesce(&cache);
		cache.objects.verify_charges(true);
		assert_eq!(cache.objects.linked(), 1, "{policy}: the set after the wipe is linked");
	});
}
