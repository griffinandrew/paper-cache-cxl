/*
 * Copyright (c) Kia Shakiba
 *
 * This source code is licensed under the GNU AGPLv3 license found in the
 * LICENSE file in the root directory of this source tree.
 */

mod expiries;

use std::{
	thread,
	time::Duration,
};

use typesize::TypeSize;
use log::debug;

use crate::{
	HashedKey,
	ObjectMapRef,
	StatusRef,
	OverheadManagerRef,
	EraseKey,
	erase,
	error::CacheError,
	worker::{
		Worker,
		WorkerEvent,
		WorkerReceiver,
		WorkerSender,
		ttl::expiries::Expiries,
	},
};

pub struct TtlWorker<K, V> {
	listener: WorkerReceiver,

	/// Point-to-point channel to `PolicyWorker`, used to report every reap as
	/// a `WorkerEvent::Expire` so the policy stack drops the key too.
	///
	/// Sent directly rather than through `WorkerFanout` for two reasons: this
	/// worker has no handle on the fanout (the fanout is built *from* the
	/// sub-workers it constructs, so handing it back in would need an
	/// `Arc::new_cyclic`-style dance), and a fanned-out `Expire` would be
	/// delivered back to this worker as well. `PolicyWorker` passing the
	/// tiering worker's sender straight into its own constructor
	/// (`promotion_tx`) is the same pattern.
	policy_tx: WorkerSender,

	objects: ObjectMapRef<K, V>,
	status: StatusRef,
	overhead_manager: OverheadManagerRef,

	expiries: Expiries,
}

impl<K, V> Worker for TtlWorker<K, V>
where
	Self: 'static + Send,
	K: Eq + TypeSize,
{
	fn run(&mut self) -> Result<(), CacheError> {
		loop {
			let now = crate::object::now_ticks();

			for event in self.listener.try_iter() {
				match event {
					WorkerEvent::Set(key, _, _, expiry, old_info, _, _) => {
						if let Some((_, old_expiry)) = old_info {
							self.expiries.remove(key, old_expiry);
						}

						self.expiries.insert(key, expiry);
					},

					WorkerEvent::Del(key, expiry) => self.expiries.remove(key, expiry),

					WorkerEvent::Ttl(key, old_expiry, new_expiry) => {
						self.expiries.remove(key, old_expiry);
						self.expiries.insert(key, new_expiry);
					},

					WorkerEvent::Wipe(_) => self.expiries.clear(),

					WorkerEvent::Shutdown => return Ok(()),

					_ => {},
				}
			}

			self.reap_due(now);

			let delay_ms = match self.expiries.has_within(2) {
				true => 1,
				false => 1000,
			};

			thread::sleep(Duration::from_millis(delay_ms));
		}
	}
}

impl<K, V> TtlWorker<K, V>
where
	K: Eq + TypeSize,
{
	/// Erases every object whose index entry is due by `now`, and reports each
	/// popped key to the policy worker.
	///
	/// Split out of `run` so a test can drive exactly one pass against an index
	/// and object map it arranged, with no sleeping thread in the way.
	fn reap_due(&mut self, now: u32) {
		while let Some(key) = self.expiries.pop_expired(now) {
			// `Expired`, not `Hashed`: the popped entry can be stale. A `set`
			// or `ttl` that landed after this pass drained the channel has
			// already put a live object at this hash, while the event that
			// would retire this entry is still queued. `Hashed` removes
			// whatever sits at the hash; `Expired` removes it only if it has
			// expired, tested under the lock the removal holds.
			erase(
				&self.objects,
				&self.status,
				&self.overhead_manager,
				Some(EraseKey::Expired(key)),
			).ok();

			// Unconditional, and deliberately not gated on `erase`'s result:
			// `erase` answers `KeyNotFound` for a key that was already gone,
			// for one it left in place because it is live, *and* for one it
			// just removed (a removed object that had expired is reported as
			// not found). The result cannot tell a removal apart, so there is
			// nothing useful to branch on.
			//
			// Notifying when nothing was removed is harmless in both
			// directions: `handle_expire` re-checks the object map and leaves a
			// key that is still there alone, and `PolicyStack::remove` on a key
			// the stack doesn't track is a no-op for every stack.
			self.notify_expired(key);
		}
	}
}

impl<K, V> TtlWorker<K, V> {
	pub fn new(
		listener: WorkerReceiver,
		policy_tx: WorkerSender,
		objects: ObjectMapRef<K, V>,
		status: StatusRef,
		overhead_manager: OverheadManagerRef,
	) -> Self {
		TtlWorker {
			listener,
			policy_tx,

			objects,
			status,
			overhead_manager,

			expiries: Expiries::default(),
		}
	}

	/// Reports a reaped key to `PolicyWorker` so it drops the key from the
	/// active policy stack and the mini stacks.
	///
	/// Best-effort by design. The channel is unbounded, so the only way
	/// `try_send` fails is a disconnected receiver -- which means the policy
	/// worker has already returned on `WorkerEvent::Shutdown` and this cache
	/// is being dropped, so there is no longer any stack state to keep in
	/// sync. Logged at debug rather than error for that reason: during
	/// teardown the two workers receive `Shutdown` in an arbitrary order, so
	/// losing a late notification here is expected, not a fault.
	fn notify_expired(&self, key: HashedKey) {
		if self.policy_tx.try_send(WorkerEvent::Expire(key)).is_err() {
			debug!("Policy worker unavailable; dropping expiry notification for {key}");
		}
	}
}

unsafe impl<K, V> Send for TtlWorker<K, V> {}

/// The reaper against a stale index entry, driven one pass at a time.
///
/// A stale entry needs no exotic timing: `set` writes the object map before it
/// sends the event that retires the old entry, so a pass that drains the
/// channel just before a `set` lands pops the old deadline with the new object
/// already in the map. These tests build that state directly -- a due index
/// entry over an object that is not due -- instead of racing threads for it.
///
/// Gated on `hybrid_cache_common` because `new_hybrid_object_map` is, which is
/// also what runs them against whichever object map the build selects: DashMap,
/// `hashbrown_dram` or `merged_object_store`, each with its own `erase`.
#[cfg(all(test, feature = "hybrid_cache_common"))]
mod tests {
	use std::{num::NonZeroU32, sync::Arc};

	use crossbeam_channel::unbounded;

	use super::TtlWorker;
	use crate::{
		HashedKey,
		ObjectMapRef,
		PaperPolicy,
		StatusRef,
		object::{ExpireTime, Object, get_expiry_from_ttl, now_ticks, overhead::OverheadManager},
		status::AtomicStatus,
		worker::{Tier, WorkerEvent, WorkerReceiver},
	};

	// `MergedStore` has `insert` and `get_ref` of its own; the other maps take
	// them from the trait.
	#[cfg(not(feature = "merged_object_store"))]
	use crate::object_store::ObjectStore;

	type TestBuffer = crate::TieredBuffer;

	const KEY: HashedKey = 7;
	const VALUE_LEN: usize = 64;

	struct Rig {
		worker: TtlWorker<u32, TestBuffer>,
		objects: ObjectMapRef<u32, TestBuffer>,
		status: StatusRef,
		policy_rx: WorkerReceiver,
	}

	fn rig() -> Rig {
		let (_events_tx, events_rx) = unbounded::<WorkerEvent>();
		let (policy_tx, policy_rx) = unbounded::<WorkerEvent>();

		let objects: ObjectMapRef<u32, TestBuffer> = crate::new_hybrid_object_map();
		let status: StatusRef = Arc::new(
			AtomicStatus::new(1 << 20, &[PaperPolicy::Lru], PaperPolicy::Lru).unwrap(),
		);
		let overhead_manager = Arc::new(OverheadManager::new(&status));

		let worker = TtlWorker::new(
			events_rx,
			policy_tx,
			objects.clone(),
			status.clone(),
			overhead_manager,
		);

		Rig { worker, objects, status, policy_rx }
	}

	/// Puts an object expiring at `expiry` at `KEY`, and accounts for it the
	/// way `set` does. Takes the tick rather than a TTL because a TTL cannot
	/// express "due now": `expiry_from_ttl` reads a zero TTL as no TTL.
	fn admit(rig: &Rig, expiry: ExpireTime) {
		let object = Object::with_expiry_in(KEY as u32, &[0xA5; VALUE_LEN], Tier::Fast, expiry);
		let base_size = rig.worker.overhead_manager.base_size(&object);

		rig.objects.insert(KEY, object);
		rig.status.incr_num_objects();
		rig.status.update_base_used_size(base_size as i64);
	}

	/// Files a due entry for `KEY`: the previous incarnation's deadline, as the
	/// index still holds it before the event that retires it has been drained.
	fn file_stale_entry(rig: &mut Rig) {
		rig.worker.expiries.insert(KEY, NonZeroU32::new(now_ticks()));
	}

	#[test]
	fn a_stale_entry_does_not_reap_an_object_re_set_without_a_ttl() {
		let mut rig = rig();
		admit(&rig, None);
		let used = rig.status.used_size(&PaperPolicy::Lru);
		file_stale_entry(&mut rig);

		rig.worker.reap_due(now_ticks());

		assert!(
			rig.objects.get_ref(&KEY).is_some(),
			"the reaper removed a live object on a stale index entry",
		);
		assert_eq!(rig.status.live_num_objects(), 1);
		assert_eq!(rig.status.used_size(&PaperPolicy::Lru), used);
	}

	#[test]
	fn a_stale_entry_does_not_reap_an_object_whose_ttl_was_extended() {
		let mut rig = rig();
		admit(&rig, Some(get_expiry_from_ttl(3_600)));
		let used = rig.status.used_size(&PaperPolicy::Lru);
		file_stale_entry(&mut rig);

		rig.worker.reap_due(now_ticks());

		assert!(
			rig.objects.get_ref(&KEY).is_some(),
			"the reaper removed an object whose TTL runs another hour",
		);
		assert_eq!(rig.status.live_num_objects(), 1);
		assert_eq!(rig.status.used_size(&PaperPolicy::Lru), used);
	}

	#[test]
	fn an_expired_object_is_still_reaped_and_reported() {
		let mut rig = rig();
		let used_empty = rig.status.used_size(&PaperPolicy::Lru);

		// Due at the current tick: `is_expired` is `expiry <= now_ticks()`.
		admit(&rig, NonZeroU32::new(now_ticks()));
		let expiry = rig.objects.get_ref(&KEY).unwrap().expiry();
		assert!(rig.objects.get_ref(&KEY).unwrap().is_expired());
		rig.worker.expiries.insert(KEY, expiry);

		rig.worker.reap_due(now_ticks());

		assert!(rig.objects.get_ref(&KEY).is_none(), "an expired object survived its reap");
		assert_eq!(rig.status.live_num_objects(), 0);
		assert_eq!(rig.status.used_size(&PaperPolicy::Lru), used_empty);
		assert!(
			matches!(rig.policy_rx.try_recv(), Ok(WorkerEvent::Expire(KEY))),
			"the reap was not reported to the policy worker",
		);
	}
}
