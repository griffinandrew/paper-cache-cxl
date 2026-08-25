/*
 * Copyright (c) Kia Shakiba
 *
 * This source code is licensed under the GNU AGPLv3 license found in the
 * LICENSE file in the root directory of this source tree.
 */

mod expiries;

use std::{
	thread,
	time::{Instant, Duration},
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
	V: TypeSize,
{
	fn run(&mut self) -> Result<(), CacheError> {
		loop {
			let now = Instant::now();

			for event in self.listener.try_iter() {
				match event {
					WorkerEvent::Set(key, _, _, expiry, old_info) => {
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

					WorkerEvent::Wipe => self.expiries.clear(),

					WorkerEvent::Shutdown => return Ok(()),

					_ => {},
				}
			}

			while let Some(key) = self.expiries.pop_expired(now) {
				erase(
					&self.objects,
					&self.status,
					&self.overhead_manager,
					Some(EraseKey::Hashed(key)),
				).ok();

				// Unconditional, and deliberately not gated on `erase`'s
				// result: `erase` answers `KeyNotFound` both for a key that
				// was already gone *and* for one it just successfully removed
				// that turned out to be expired -- which is every reap here,
				// by construction (see its trailing `match !object
				// .is_expired()`). So the result cannot distinguish the two
				// cases and there is nothing useful to branch on.
				//
				// Notifying in the already-gone case is harmless in both
				// directions: `handle_expire` re-checks the object map before
				// touching the stack, and `PolicyStack::remove` on a key the
				// stack doesn't track is a no-op for every stack.
				self.notify_expired(key);
			}

			let delay_ms = match self.expiries.has_within(2) {
				true => 1,
				false => 1000,
			};

			thread::sleep(Duration::from_millis(delay_ms));
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
