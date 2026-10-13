/*
 * Copyright (c) Kia Shakiba
 *
 * This source code is licensed under the GNU AGPLv3 license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! The MOVE LEDGER: a transient, exact, per-key record of what the migration
//! pipeline is doing to a key, so that a queued tier move can carry the
//! identity of the VALUE it was decided for. Three knobs, each read once per
//! process from the environment (an integer, 0 = off; unset or unparsable =
//! the default):
//!
//!   * `MIG_TAGGED` (default ON): TAGGED MOVES. Every queued move carries the
//!     identity of the newest value published under its key when it was handed
//!     off, and a move whose key has been given a newer value since is skipped
//!     by its consumer (`stale_skipped`). The new-key rule's bucket-wide fence
//!     and the client's `InFlight` mark are replaced by an exact, per-key
//!     fence (`pending_fences`, below), and the heal runs only where bytes can
//!     lag their placement for a reason other than a stale move. Off, the
//!     cache behaves exactly as it did before the ledger existed: the mark,
//!     the fence and the heal of `Observed`'s doc in `super`.
//!   * `MIG_LATEST` (default off): every queued move carries a hand-off
//!     sequence number, and a move that a NEWER queued move of the same key
//!     supersedes is skipped (`latest_skipped`).
//!   * `MIG_REVCOUNT` (default off): counts, whatever the other two say,
//!     `reversal_copies` (a move that copied while a later move of the same
//!     key naming the OTHER tier was already queued -- a copy that later move
//!     undoes, unless it is itself skipped or declined) and `dup_same_tier` (a
//!     move handed off while the key's newest in-flight move named the same
//!     tier -- one of the two declines unless something lands between them).
//!     Pairs inside one drain never reach the queue (`split_tier_migrations`,
//!     `coalesced_tot`) and are not counted here. Diagnostic only: it decides
//!     nothing.
//!
//! With all three off the hand-off and the finish return at their first
//! comparison, the check is not made, and a queued move carries a zero `Tag`;
//! the ledger itself (its empty shards, 64 KiB) is still built on first use.
//! With only `MIG_REVCOUNT` it is touched by the worker's hand-off and the
//! consumer's finish (two short critical sections per queued move) and changes
//! no decision.
//!
//! # Zero bytes per object
//!
//! Nothing is added to a value, a map row or a stack node. The value header
//! (`value.rs`: count 8 | key 8 | tagged bytes word 8 | len 4 | expiry 4, the
//! 32-byte jemalloc class exactly; `thin_header`'s is 16 bytes, count and
//! tagged item pointer) has no padding, and its only spare bits -- the low
//! alignment bits and the top bits of a pointer -- are too few for an identity
//! that cannot wrap. The header's ADDRESS is not an identity either: the
//! allocator recycles it (ABA), and a tier migration builds a NEW header
//! around the copied bytes, so one value has as many addresses as it has had
//! migrations.
//!
//! The identity is instead `(key, epoch)`, where `epoch` is the number of
//! values PUBLISHED under the key since the key's ledger record was created.
//! A record exists only while the key has a move in flight or a `Set` the
//! policy worker has not handled yet, and is removed the moment it has
//! neither. It holds two publication counts (`published`, `seen`), the moves
//! handed off while a `Set` of the key was unhandled, the newest hand-off's
//! sequence and tier, and the in-flight count per tier (`Rec`, 40 bytes, 48
//! with its key in the table). The memory is transient -- bounded by the
//! migration queue's depth plus the policy worker's event backlog -- and
//! `ledger_max` reports its sampled high-water mark in records.
//!
//! Why `(key, epoch)` is exact, with no ABA: an epoch is only ever compared
//! against the record it was read from. A record is removed only when no move
//! of the key is in flight -- so no queued move holds an epoch of it -- and no
//! `Set` of the key is unhandled. A re-created record starts again at 0, and
//! every move that will ever be compared against it is handed off after it was
//! created. Within one record both counts only grow.
//!
//! # The protocol (MIG_TAGGED)
//!
//!   1. CLIENT, `PaperCache::publish_set`: `published += 1` under the record's
//!      shard lock, BEFORE the map insert publishes the value; the value's
//!      `Set` then carries `LEDGERED` in its mark field, which is otherwise
//!      only read for the bucket-wide fence this mode never issues
//!      (`client_mark`).
//!   2. WORKER, `PolicyWorker::handle_set` of a `LEDGERED` `Set`: `seen += 1`,
//!      first thing, whatever the event then finds (`seen`). While
//!      `published != seen` the key is in a PENDING WINDOW: the map may hold a
//!      value whose `Set` the worker has not handled, so the stack's state for
//!      the key is older than the key's value. The `Set` that closes the window
//!      (`seen` reaching `published`) FENCES if any move of the key was handed
//!      off inside it: `(k, placement)` is appended to that `Set`'s drain,
//!      behind every such move, unless the drain's last entry for the key
//!      already names it (`Observed::Built`'s `fence`, `pending_fences`).
//!   3. WORKER, hand-off (`MigrationQueue::push` and the inline path; the
//!      promotion retry and the settle go through them): the move is stamped
//!      `epoch = published`, the publication of the newest value. Inside a
//!      pending window it is counted (`Rec::pending_handoffs`, the
//!      `pending_handoffs` statistic) -- and still queued: it names where the
//!      stack places the key, which the newest value needs as much as the one
//!      it was decided for, and a settle demotion dropped here would leave its
//!      bytes in DRAM for as long as the key's window stays open (under a
//!      worker backlog, every hot key's, indefinitely).
//!   4. CONSUMER (or the inline path), after the snapshot and before the copy
//!      (`apply_migration_guarded`): `published != epoch` -> SKIPPED, counted
//!      `stale_skipped`. The snapshot is taken first: a client that published
//!      the snapshotted value bumped `published` before its insert, and the
//!      map's shard lock orders that insert before the snapshot, so the check
//!      sees the bump. A value published AFTER the snapshot makes the swap
//!      lose (`ptr_eq`), as it always did (`superseded`).
//!
//! # Correctness (MIG_TAGGED)
//!
//! THE INVARIANT, which replaces "for a key the stack places, the last entry
//! queued for it that is still in flight names its placement" (`Observed`):
//!
//!   A VALUE IS ONLY EVER MOVED BY A MOVE HANDED OFF AFTER THE VALUE WAS
//!   PUBLISHED; AND A MOVE HANDED OFF BEFORE THE WORKER HANDLED ITS KEY'S LAST
//!   `Set` IS FOLLOWED, ON THE KEY'S CONSUMER, BY THE FENCE OF THE `Set` THAT
//!   CLOSES THE WINDOW.
//!
//! Proof of the first half. Take a move M of key k, handed off with epoch e,
//! and a value v of k that M's consumer snapshots. M copies only if its check
//! then reads `published == e`. M holds the record from its hand-off to its
//! finish, so every publication of k in between bumps that record, whose
//! counts only grow: none happened between M's hand-off and the check. v's own
//! bump precedes its insert, which precedes the snapshot (step 4), which
//! precedes the check -- so v was published before M's hand-off, and, no
//! later publication having happened, v is the value that was newest at the
//! hand-off. M's swap then lands only if v is still the key's value
//! (`ptr_eq`). A move handed off before v's publication carries an older
//! epoch and copies nothing onto v: it reads v's bump, or it snapshotted the
//! older value and loses the swap. The second half is steps 2 and 3: a
//! hand-off inside a window is counted in the record, the record lives until
//! the window closes, and the closing `Set` reads the count and queues its
//! fence in its own drain -- after every hand-off of the window, on the key's
//! one FIFO consumer.
//!
//! From the invariant, the key's newest value v ends at the key's placement.
//! Let S be the `Set` that closes the last window (v's own, or -- the two
//! publish in one order and send in the other -- an older value's handled
//! after it). The moves of k fall in three groups:
//!
//!   * handed off before v's publication: never land on v (first half);
//!   * handed off after v's publication and before S, inside the window:
//!     whatever they do to v, S's fence lands after them and puts v at the
//!     placement S reads (or declines when it is already there). v's own
//!     `Set`, if it was handled inside the window, compared v's built tier
//!     with the placement then, and its corrective is one of these moves;
//!   * handed off after S: decided from a stack that has handled every `Set`
//!     of k, so the base contract holds unchanged -- every placement change of
//!     a tracked key pushes its own move (`placement_of`; un-trackings move
//!     nothing, and they take the value out of the map), FIFO on the key's
//!     consumer, the last naming the placement.
//!
//! And with no window at all (S is v's own `Set`, handled with nothing pending
//! and no hand-off since v's publication), v's bytes are where its client
//! built them when S is handled, and the plain rule compares that built tier
//! with the placement, as it always did. Every `Set` is reconciled with its
//! own built tier and its own placement byte; a corrective always names the
//! placement, so one queued for an older value and landing on a newer one
//! moves it to where the key belongs.
//!
//! The races the new-key rule fenced (`Observed`, review M1) all reduce to the
//! invariant -- a stale move cannot land on the fresh value:
//!
//!   (i)   2Q: a heal `(k, Fast)` decided for v1, then `del(k)` and
//!         `set(k, v2)`, built slow and admitted to the slow FIFO with no push.
//!         Handed off before v2's publication, the heal carries v1's epoch and
//!         is skipped. Handed off after it (v2's `Set` still unhandled), it is
//!         inside v2's window: it may promote v2, and v2's `Set` closes the
//!         window and fences `(k, Slow)` behind it.
//!   (ii)  LFU: the corrective `(k, Fast)` queued for v1; `k` is evicted; v2
//!         is built slow and admitted slow (latched): the same.
//!   (iii) a stale STACK `(k, Fast)` still queued when `k` is deleted or
//!         evicted and re-set into a slow admission queue: the same.
//!
//! What remains of the new-key rule is that fence, exact: per key, only when
//! a move of the key itself was handed off inside its window -- where the old
//! rule fenced for any move of the key's 1/16,384th BUCKET in flight or landed
//! since the client's mark, and under a backlog that was most fresh sets. The
//! `InFlight` mark (one client load per set) and the worker's
//! `moved_since`/`placement_of` probe are not issued. The plain rule stays: it
//! is what corrects a value BUILT in the wrong tier (the LFU stale latch, race
//! (a)), which no tag can prevent.
//!
//! # Which configurations keep the heal (MIG_TAGGED)
//!
//! The heal promotes a value served slow whose key is placed fast. Under the
//! invariant a stale move can no longer leave that state behind; what still
//! can:
//!
//!   * KEPT, byte-gate `OnStall::Divert`: a diverted set is built slow, placed
//!     fast, and its stack promotion is dropped from the drain ON PURPOSE
//!     (`drain_reconciled`) -- the heal is how such a key gets promoted, on
//!     its first slow-served hit.
//!   * KEPT, `PromotionGate::On`: a promotion the consumer declines for room
//!     is retried by `PromotionRetry`, which DROPS keys past
//!     `PROMO_DECLINED_CAP`: such a key is placed fast with its bytes slow
//!     and nothing queued.
//!   * DROPPED, 2Q's and the S3-FIFO designs' slow admission: a new key is
//!     admitted where it was built, or the plain rule corrects it at its
//!     `Set`; their lagging promotions were the stale-move races (i), now
//!     skipped or fenced.
//!   * DROPPED, the LFU latch (both LFU designs): a stale mirror builds a new
//!     key in the wrong tier, and the plain rule at its `Set` corrects it, as
//!     it always did; the heal covered only race (ii).
//!   * DROPPED, a migration that loses its swap, or a move skipped: the value
//!     it was not for is reconciled at its own `Set`.
//!
//! Every other configuration (the defaults: `OnStall::Error`,
//! `PromotionGate::Off`) needs none, and a slow-served hit there is counted
//! `heal_off` and not healed. One in `HEAL_PROBE_EVERY` of those hits is still
//! looked up, for the record only: a hit the heal WOULD have promoted is
//! counted `heal_off_lagging` -- a lag nothing in this argument explains, so
//! a design breaking the `placement_of` contract shows there. The decision is
//! read per slow-served hit from the gate's live configuration
//! (`PolicyWorker::heals_needed`), so a runtime change takes effect at once.
//! A diverted key that lags when the configuration leaves `Divert` stays
//! lagging until its next placement change moves it.
//!
//! # Both object stores
//!
//! The protocol touches only what both stores share: the client's
//! `publish_set` (the merged store's `MergedStore::insert` publishes exactly
//! as the DashMap insert does), the worker's `handle_set`, the hand-offs of
//! `apply_migration_batches`, and `apply_migration_guarded`'s snapshot,
//! check and `ptr_eq` swap through `ObjectStore`. So tagged moves apply, with
//! the same argument, in the merged store. Its race R3 (`s4_tests`): the
//! settle demotion of K handed off after the clients re-set K twice lands on
//! the newest value inside the window, and the `Set` that closes the window
//! fences it back.
//!
//! # MIG_LATEST
//!
//! Every hand-off gets the next value of one per-cache counter (`seq`), and
//! the key's record keeps the newest `seq` handed off (`latest_seq`). The
//! consumer skips a move whose `seq` is not the key's newest. Correctness:
//!
//!   * PER-KEY FIFO: all moves of a key go to one consumer channel (`key %
//!     N`) and are applied in hand-off order; skipping removes moves from the
//!     FRONT of that order and never reorders the rest.
//!   * FINAL TIER = LAST DECISION: a move is skipped only if a newer move of
//!     the same key is already queued; that one is applied later (or skipped
//!     in turn for a still newer one), so the newest move is never skipped,
//!     and the key's bytes end where its last move sends them -- exactly where
//!     the unskipped sequence would leave them (each move is idempotent: "be
//!     in tier T", declined when already there). Demote-then-promote across
//!     drains ends with zero copies (the demotion is skipped; the promotion
//!     finds the value fast and declines); promote-then-demote ends with one
//!     demotion if the value was fast, none if it was slow.
//!   * Under MIG_TAGGED as well: the newest move may carry an OLDER epoch than
//!     the current value and be skipped as stale -- then every move of the
//!     key was handed off before the current value's publication, and the
//!     current value is reconciled at its own `Set` (above). The tag is
//!     checked first. A fence is the newest move of its key when queued.
//!   * ACCOUNTING: a skipped move is FINISHED like any other (pending gauges,
//!     `InFlight`, `processed` all count it) and LANDS nothing, so it is never
//!     a promotion, a demotion or an applied corrective -- and a test or a
//!     reader that waits for a promotion count to reach the number of hits
//!     that decided one can wait forever under MIG_LATEST (a promotion
//!     superseded by a demotion of the same key is skipped). The stacks'
//!     fast/slow byte counts are decision-time (intent) and are unaffected;
//!     so are `demo_tot`/`promo_tot` and the reconcile's queued counts, which
//!     count intents at the hand-off. Under LFU (`inline_demotion_accounting
//!     == false`) demotions are counted at DECISION time by the stack
//!     (`drain_demotions`), so a skipped LFU demotion is still counted as a
//!     demotion -- `latest_skipped` and `stale_skipped` say how many intents
//!     moved nothing. The byte gate is woken only by LANDED demotions
//!     (`note_demotion`), as before.
//!
//! # Cost
//!
//! MIG_TAGGED, per `set`: on the client, one shard lock and a probe-or-insert
//! of a 48-byte table entry (no allocation in the steady state: a shard keeps
//! room for its records) instead of the mark's one load; no counter shared
//! between clients is written -- but the record's line is usually last
//! written by the policy worker (`seen`), so a busy worker makes it a
//! cross-core transfer. On the policy worker, one shard lock and a
//! probe-and-remove (`seen`) instead of the `moved_since` load, and no
//! `placement_of` probe for the fence. Per queued move: a lock and an
//! insert-or-probe at the hand-off, a lock and a probe at the check, a lock
//! and a probe-or-remove at the finish -- next to the copy they guard. Per
//! slow-served hit with the heal off: the mode's load, `heals_needed`'s
//! relaxed load and read lock, and one relaxed increment -- no `InFlight`
//! load, and a `placement_of` for one hit in `HEAL_PROBE_EVERY`. `ledger_max`
//! is sampled by the policy worker (`MoveLedger::sample`): one pass over the
//! shards every `SAMPLE_EVERY` worker passes (idle or not), and at shutdown.
//!
//! Memory: about 0.9 MiB of tables per cache, plus 48 bytes (and hashbrown's
//! slack) per live record -- at most one per key with a `Set` still in the
//! policy worker's channel or a move still in the migration queue, each of
//! which holds a larger entry there already. Not in the byte gate's metadata
//! figure `M` (`crate::meta`'s "not attributed" list, with the `InFlight`
//! table and the channels). A shard's table grows with a burst and the
//! worker's samples shrink it back.

use std::collections::HashMap;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicU8, AtomicU64, Ordering};

use parking_lot::Mutex;

use crate::{HashedKey, NoHasher};
use crate::worker::policy::migstats::Stats as MigStats;
use crate::worker::policy::policy_stack::Tier;

/// The value a `LEDGERED` `Set` carries in its mark field (`WorkerEvent::Set`):
/// the client bumped the key's `published` count before inserting
/// (`client_mark`). `u32::MAX`, which an `InFlight` mark -- a bucket's landed
/// count, wrapping at 2^32 -- reaches only after four billion landings in one
/// bucket; and the worker only looks for it when the cache runs tagged moves,
/// where every `Set` from `publish_set` carries it.
pub const LEDGERED: u32 = u32::MAX;

/// An on/off knob: an integer, 0 = off, anything else = on; unset or
/// unparsable -> `default`.
pub(crate) fn parse_switch(value: Option<&str>, default: bool) -> bool {
	value
		.and_then(|value| value.trim().parse::<u64>().ok())
		.map_or(default, |value| value != 0)
}

fn knob(name: &str, default: bool) -> bool {
	parse_switch(std::env::var(name).ok().as_deref(), default)
}

/// `MIG_TAGGED`, read once (default ON).
pub fn tagged_env() -> bool {
	static ON: OnceLock<bool> = OnceLock::new();
	*ON.get_or_init(|| knob("MIG_TAGGED", true))
}

/// `MIG_LATEST`, read once (default off).
pub fn latest_env() -> bool {
	static ON: OnceLock<bool> = OnceLock::new();
	*ON.get_or_init(|| knob("MIG_LATEST", false))
}

/// `MIG_REVCOUNT`, read once (default off).
pub fn revcount_env() -> bool {
	static ON: OnceLock<bool> = OnceLock::new();
	*ON.get_or_init(|| knob("MIG_REVCOUNT", false))
}

/// What a ledger does: the three knobs.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub struct Mode {
	pub tagged: bool,
	pub latest: bool,
	pub count: bool,
}

impl Mode {
	pub fn from_env() -> Self {
		Mode { tagged: tagged_env(), latest: latest_env(), count: revcount_env() }
	}

	fn bits(self) -> u8 {
		self.tagged as u8 | (self.latest as u8) << 1 | (self.count as u8) << 2
	}

	fn from_bits(bits: u8) -> Self {
		Mode { tagged: bits & 1 != 0, latest: bits & 2 != 0, count: bits & 4 != 0 }
	}

	/// Whether a hand-off or a finish touches the table at all.
	pub fn tracks(self) -> bool {
		self.tagged || self.latest || self.count
	}

	/// Whether a consumer checks a move before copying it.
	pub fn checks(self) -> bool {
		self.tagged || self.latest
	}
}

/// What a queued move carries besides `(key, tier, origin)`: its hand-off
/// sequence and the epoch of the newest value at its hand-off. `(0, 0)` when
/// the ledger does not track.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Tag {
	pub seq: u64,
	pub epoch: u64,
}

/// A consumer's verdict on a move, after its snapshot.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Verdict {
	Apply,
	/// The key holds a value published after the move was handed off.
	Stale,
	/// A newer move of the key was handed off after this one.
	Superseded,
}

/// What the worker's handling of one `LEDGERED` `Set` learns (`seen`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Seen {
	/// A NEWER publication of the key is still unhandled: the key's pending
	/// window stays open.
	pub pending: bool,
	/// This `Set` closes the window, and moves of the key were handed off
	/// inside it: the `Set` fences (protocol step 2).
	pub fence: bool,
}

const NO_TIER: u8 = 0;

fn tier_code(tier: Tier) -> u8 {
	match tier {
		Tier::Fast => 1,
		Tier::Slow => 2,
	}
}

/// One key's record. See the module doc.
#[derive(Default, Debug, Clone, Copy)]
struct Rec {
	/// Values of the key published since the record was created.
	published: u64,
	/// Their `Set`s the policy worker has handled.
	seen: u64,
	/// Moves of the key handed off while `published != seen`, since the key
	/// last had every `Set` handled: the next closing `Set` fences if any.
	pending_handoffs: u32,
	/// The newest hand-off's sequence and tier (`NO_TIER` once nothing of
	/// the key is in flight).
	latest_seq: u64,
	latest_tier: u8,
	/// Moves of the key in flight, by destination.
	in_fast: u32,
	in_slow: u32,
}

impl Rec {
	fn in_flight(&self) -> u32 {
		self.in_fast + self.in_slow
	}

	fn quiescent(&self) -> bool {
		self.in_flight() == 0 && self.published == self.seen
	}

	fn count_mut(&mut self, tier: Tier) -> &mut u32 {
		match tier {
			Tier::Fast => &mut self.in_fast,
			Tier::Slow => &mut self.in_slow,
		}
	}

	fn other(&self, tier: Tier) -> u32 {
		match tier {
			Tier::Fast => self.in_slow,
			Tier::Slow => self.in_fast,
		}
	}
}

const SHARDS: usize = 1024;

/// Records each shard has room for from the start (`MoveLedger::new`).
const SHARD_ROOM: usize = 8;

/// With the heal off (tagged moves, `PolicyWorker::heals_needed`), one
/// slow-served hit in this many is still looked up -- for `heal_off_lagging`
/// only, never healed.
pub const HEAL_PROBE_EVERY: u64 = 64;

/// `ledger_max`'s sampling period, in policy-worker passes (`sample_due`,
/// counted by `evict_calls`): the cadence of the worker's other per-pass
/// sample (`DIVERGE`).
pub const SAMPLE_EVERY: u64 = 4096;

/// One shard, alone on its cache line(s): the clients' publications of
/// unrelated keys do not share a line.
#[repr(align(64))]
struct Shard(Mutex<HashMap<HashedKey, Rec, NoHasher>>);

/// The ledger of one cache. See the module doc.
pub struct MoveLedger {
	shards: Box<[Shard]>,
	mode: AtomicU8,
	/// The next hand-off's sequence. Written only by the policy worker, the
	/// one thread that hands off.
	next_seq: AtomicU64,
	/// The highest live-record count `sample` has seen.
	live_max: AtomicU64,
}

impl Default for MoveLedger {
	fn default() -> Self {
		Self::new(Mode::from_env())
	}
}

impl MoveLedger {
	pub fn new(mode: Mode) -> Self {
		// Each shard starts with room for a few records, so a publication does
		// not allocate in the steady state (a shard grows only past its
		// high-water mark, until `sample` shrinks it). Empty tables when the
		// ledger is off (64 KiB of shard headers); ~0.9 MiB per cache when on
		// (16 buckets of 48 bytes and their control bytes per shard).
		let room = if mode.tracks() { SHARD_ROOM } else { 0 };

		MoveLedger {
			shards: (0..SHARDS)
				.map(|_| Shard(Mutex::new(HashMap::with_capacity_and_hasher(room, NoHasher::default()))))
				.collect(),
			mode: AtomicU8::new(mode.bits()),
			next_seq: AtomicU64::new(1),
			live_max: AtomicU64::new(0),
		}
	}

	#[inline]
	pub fn mode(&self) -> Mode {
		Mode::from_bits(self.mode.load(Ordering::Relaxed))
	}

	/// Tests: switches the mode of a QUIESCENT cache (nothing in flight, no
	/// unhandled `Set`). Production reads the environment once.
	#[cfg(test)]
	pub fn set_mode(&self, mode: Mode) {
		self.mode.store(mode.bits(), Ordering::Relaxed);
	}

	#[inline]
	fn shard(&self, key: HashedKey) -> &Mutex<HashMap<HashedKey, Rec, NoHasher>> {
		// The high bits: the consumer routing and `InFlight` use the low ones,
		// and the table inside a shard indexes by the low ones too.
		&self.shards[(key >> 40) as usize & (SHARDS - 1)].0
	}

	/// Live records now (0 at quiescence): one lock per shard. Tests.
	#[cfg(test)]
	pub fn live(&self) -> u64 {
		self.shards.iter().map(|shard| shard.0.lock().len() as u64).sum()
	}

	/// POLICY WORKER: samples the live-record count into the high-water mark
	/// and returns the mark (`ledger_max`). One pass over the shards, which
	/// also gives back what a burst left: a shard whose table has room for
	/// more than four times its records (and more than `SHARD_ROOM`) is
	/// shrunk to twice its records, so the ledger's memory follows its live
	/// records rather than staying at its peak.
	pub fn sample(&self) -> u64 {
		let mut live = 0;

		for shard in self.shards.iter() {
			let mut table = shard.0.lock();
			let len = table.len();

			if table.capacity() > 4 * len.max(SHARD_ROOM) {
				table.shrink_to((2 * len).max(SHARD_ROOM));
			}

			live += len as u64;
		}

		self.live_max.fetch_max(live, Ordering::Relaxed).max(live)
	}

	/// Whether the policy worker's `pass`th pass samples (`SAMPLE_EVERY`).
	#[inline]
	pub fn sample_due(&self, pass: u64) -> bool {
		self.mode().tracks() && pass % SAMPLE_EVERY == 0
	}

	/// The highest live-record count sampled so far.
	#[cfg(test)]
	pub fn live_max(&self) -> u64 {
		self.live_max.load(Ordering::Relaxed)
	}

	/// CLIENT (tagged), protocol step 1: one value of `key` is about to be
	/// published. Returns the mark its `Set` carries: `LEDGERED`.
	#[inline]
	pub fn publish(&self, key: HashedKey) -> u32 {
		self.shard(key).lock().entry(key).or_default().published += 1;

		LEDGERED
	}

	/// WORKER (tagged), protocol step 2: the `Set` of one `LEDGERED`
	/// publication of `key` is being handled. Says whether the key's pending
	/// window stays open, or closes with a fence. A `LEDGERED` `Set` with no
	/// publication left to account for changes nothing.
	pub fn seen(&self, key: HashedKey) -> Seen {
		let mut shard = self.shard(key).lock();

		let Some(rec) = shard.get_mut(&key) else {
			debug_assert!(false, "a LEDGERED Set with no record: never published");
			return Seen::default();
		};

		debug_assert!(rec.seen < rec.published, "a LEDGERED Set handled twice, or never published");

		if rec.seen < rec.published {
			rec.seen += 1;
		}

		let pending = rec.published != rec.seen;
		let fence = !pending && rec.pending_handoffs > 0;

		if !pending {
			rec.pending_handoffs = 0;
		}

		if rec.quiescent() {
			shard.remove(&key);
		}

		Seen { pending, fence }
	}

	/// WORKER, protocol step 3, the hand-off: stamps one move of `key` to
	/// `tier` with the newest publication. Under tagged moves a hand-off while
	/// a `Set` of the key is unhandled is counted, in the record (for the
	/// closing `Set`'s fence) and in `pending_handoffs`. Counts
	/// `dup_same_tier`.
	pub fn hand_off(&self, key: HashedKey, tier: Tier, stats: &MigStats) -> Tag {
		let mode = self.mode();

		if !mode.tracks() {
			return Tag::default();
		}

		let mut shard = self.shard(key).lock();
		let rec = shard.entry(key).or_default();

		if mode.tagged && rec.published != rec.seen {
			rec.pending_handoffs = rec.pending_handoffs.saturating_add(1);
			stats.pending_handoffs.fetch_add(1, Ordering::Relaxed);
		}

		if mode.count && rec.in_flight() > 0 && rec.latest_tier == tier_code(tier) {
			stats.dup_same_tier.fetch_add(1, Ordering::Relaxed);
		}

		let seq = self.next_seq.fetch_add(1, Ordering::Relaxed);

		rec.latest_seq = seq;
		rec.latest_tier = tier_code(tier);
		*rec.count_mut(tier) += 1;

		Tag { seq, epoch: rec.published }
	}

	/// CONSUMER (or the inline path), protocol step 4, after the snapshot:
	/// whether to copy. The tag is checked before the sequence.
	pub fn check(&self, key: HashedKey, tag: Tag) -> Verdict {
		let mode = self.mode();

		if !mode.checks() {
			return Verdict::Apply;
		}

		let shard = self.shard(key).lock();

		let Some(rec) = shard.get(&key) else {
			// Every move handed off holds its record until its finish: this is
			// a test's raw entry, pushed past the hand-off.
			return Verdict::Apply;
		};

		if mode.tagged && rec.published != tag.epoch {
			return Verdict::Stale;
		}

		if mode.latest && rec.latest_seq != tag.seq {
			return Verdict::Superseded;
		}

		Verdict::Apply
	}

	/// CONSUMER (or the inline path, or a refused send): one move of `key` to
	/// `tier` is finished; `landed`: it copied. Counts `reversal_copies`, and
	/// removes the record once the key is quiescent.
	pub fn finish(&self, key: HashedKey, tier: Tier, landed: bool, stats: Option<&MigStats>) {
		let mode = self.mode();

		if !mode.tracks() {
			return;
		}

		let mut shard = self.shard(key).lock();

		let Some(rec) = shard.get_mut(&key) else {
			return;
		};

		if landed && mode.count && rec.other(tier) > 0 {
			if let Some(stats) = stats {
				stats.reversal_copies.fetch_add(1, Ordering::Relaxed);
			}
		}

		let count = rec.count_mut(tier);
		*count = count.saturating_sub(1);

		if rec.in_flight() == 0 {
			rec.latest_tier = NO_TIER;
		}

		if rec.quiescent() {
			shard.remove(&key);
		}
	}
}

/// What the client sends as a `Set`'s mark, BEFORE the map insert: under
/// tagged moves `LEDGERED`, after bumping the key's publication count
/// (protocol step 1); else the `InFlight` landed count the bucket-wide
/// new-key fence reads (`InFlight::mark`), as it always was.
#[inline]
pub fn client_mark(status: &crate::status::AtomicStatus, key: HashedKey) -> u32 {
	let ledger = status.move_ledger();

	match ledger.mode().tagged {
		true => ledger.publish(key),
		false => status.migration_in_flight().mark(key),
	}
}

#[cfg(test)]
mod tests {
	use super::*;

	fn tracked(tagged: bool, latest: bool) -> MoveLedger {
		MoveLedger::new(Mode { tagged, latest, count: true })
	}

	#[test]
	fn knobs_parse() {
		assert!(!parse_switch(None, false));
		assert!(parse_switch(None, true));
		assert!(parse_switch(Some("1"), false));
		assert!(!parse_switch(Some("0"), true));
		assert!(parse_switch(Some(" 2 "), false));
		assert!(parse_switch(Some("x"), true));
		assert!(!parse_switch(Some(""), false));
	}

	/// The defaults: tagged moves on, MIG_LATEST and MIG_REVCOUNT off -- as
	/// this process reads them, so only when none of the three is set.
	#[test]
	fn the_defaults_are_tagged_only() {
		if ["MIG_TAGGED", "MIG_LATEST", "MIG_REVCOUNT"].iter().any(|name| std::env::var_os(name).is_some()) {
			eprintln!("SKIPPED the_defaults_are_tagged_only: a ledger knob is set");
			return;
		}

		assert_eq!(Mode::from_env(), Mode { tagged: true, latest: false, count: false });
		assert!(Mode::from_env().tracks() && Mode::from_env().checks());
	}

	#[test]
	fn a_record_lives_exactly_while_something_of_its_key_is_pending() {
		let stats = MigStats::default();
		let ledger = tracked(true, false);

		assert_eq!(ledger.publish(7), LEDGERED);
		assert_eq!(ledger.live(), 1);

		// Handed off inside the window: queued, stamped with the newest
		// publication, and counted for the closing `Set`'s fence.
		let early = ledger.hand_off(7, Tier::Slow, &stats);
		assert_eq!(early.epoch, 1);
		assert_eq!(stats.pending_handoffs.load(Ordering::Relaxed), 1);
		assert_eq!(ledger.check(7, early), Verdict::Apply, "it may land on the newest value");

		assert_eq!(ledger.seen(7), Seen { pending: false, fence: true }, "the window closes with a fence");
		ledger.finish(7, Tier::Slow, true, Some(&stats));
		assert_eq!(ledger.live(), 0, "quiescent: removed");

		let tag = ledger.hand_off(7, Tier::Slow, &stats);
		assert_eq!(tag.epoch, 0, "a fresh record");
		assert_eq!(stats.pending_handoffs.load(Ordering::Relaxed), 1, "no window open: not counted");
		assert_eq!(ledger.check(7, tag), Verdict::Apply);
		assert_eq!(ledger.sample(), 1);

		ledger.publish(7);
		assert_eq!(ledger.check(7, tag), Verdict::Stale, "a newer value published");
		ledger.finish(7, Tier::Slow, false, Some(&stats));
		assert_eq!(ledger.live(), 1, "its Set is still unhandled");
		assert_eq!(ledger.seen(7), Seen { pending: false, fence: false }, "nothing was handed off inside this window");
		assert_eq!(ledger.live(), 0);
		assert_eq!(ledger.sample(), 1, "the high-water mark stays");
		assert_eq!(ledger.live_max(), 1);
	}

	/// Two publications, their `Set`s handled in either order: the first
	/// handled keeps the window open, the second closes it -- with a fence
	/// only if a move was handed off in between -- and the counts start again
	/// for the next window.
	#[test]
	fn sets_handled_out_of_order_close_the_window_at_the_last() {
		let stats = MigStats::default();
		let ledger = tracked(true, false);

		ledger.publish(3);
		ledger.publish(3);

		assert_eq!(ledger.seen(3), Seen { pending: true, fence: false }, "one more publication is unhandled");
		let inside = ledger.hand_off(3, Tier::Fast, &stats);
		assert_eq!(inside.epoch, 2);
		assert_eq!(ledger.seen(3), Seen { pending: false, fence: true });
		ledger.finish(3, Tier::Fast, true, Some(&stats));
		assert_eq!(ledger.live(), 0, "quiescent once both are handled and the move is done");

		// A window with no hand-off in it closes without a fence, even right
		// after one that had one.
		ledger.publish(3);
		ledger.publish(3);
		assert!(ledger.seen(3).pending);
		assert_eq!(ledger.seen(3), Seen { pending: false, fence: false });

		// Handed off before a third publication, which then makes it stale.
		let tag = ledger.hand_off(3, Tier::Fast, &stats);
		assert_eq!(tag.epoch, 0, "a record re-created at 0");
		ledger.publish(3);
		assert_eq!(ledger.check(3, tag), Verdict::Stale);
		ledger.finish(3, Tier::Fast, false, Some(&stats));
		assert_eq!(ledger.seen(3), Seen { pending: false, fence: false }, "handed off before the window opened");
		assert_eq!(ledger.live(), 0);
	}

	/// A `LEDGERED` `Set` with nothing to account for (in release builds,
	/// where the debug assertion is off; the suites run release) leaves the
	/// counts balanced: the key's next publication is still pending until ITS
	/// `Set`.
	#[cfg(not(debug_assertions))]
	#[test]
	fn a_stray_ledgered_set_does_not_unbalance_the_counts() {
		let stats = MigStats::default();
		let ledger = tracked(true, false);

		let tag = ledger.hand_off(5, Tier::Fast, &stats);
		assert_eq!(ledger.seen(5), Seen::default(), "nothing published: nothing pending");

		ledger.publish(5);
		ledger.hand_off(5, Tier::Slow, &stats);
		assert_eq!(stats.pending_handoffs.load(Ordering::Relaxed), 1, "the publication's Set is unhandled");
		assert_eq!(ledger.check(5, tag), Verdict::Stale);
		ledger.finish(5, Tier::Fast, false, Some(&stats));
		assert_eq!(ledger.seen(5), Seen { pending: false, fence: true });
		ledger.finish(5, Tier::Slow, true, Some(&stats));
		assert_eq!(ledger.live(), 0);
	}

	#[test]
	fn latest_skips_all_but_the_newest_and_counts_reversals_and_duplicates() {
		let stats = MigStats::default();
		let ledger = tracked(false, true);

		let a = ledger.hand_off(9, Tier::Slow, &stats);
		let b = ledger.hand_off(9, Tier::Fast, &stats);
		let c = ledger.hand_off(9, Tier::Fast, &stats);
		assert_eq!(stats.dup_same_tier.load(Ordering::Relaxed), 1, "c after b, both fast");

		assert_eq!(ledger.check(9, a), Verdict::Superseded);
		ledger.finish(9, Tier::Slow, false, Some(&stats));
		assert_eq!(ledger.check(9, b), Verdict::Superseded);
		ledger.finish(9, Tier::Fast, false, Some(&stats));
		assert_eq!(ledger.check(9, c), Verdict::Apply);
		ledger.finish(9, Tier::Fast, true, Some(&stats));
		assert_eq!(stats.reversal_copies.load(Ordering::Relaxed), 0);
		assert_eq!(ledger.live(), 0);

		// Count only: a copy with a later other-tier move queued is a reversal.
		let ledger = tracked(false, false);
		let a = ledger.hand_off(9, Tier::Slow, &stats);
		ledger.hand_off(9, Tier::Fast, &stats);
		assert_eq!(ledger.check(9, a), Verdict::Apply, "nothing is skipped when only counting");
		ledger.finish(9, Tier::Slow, true, Some(&stats));
		assert_eq!(stats.reversal_copies.load(Ordering::Relaxed), 1);
		ledger.finish(9, Tier::Fast, true, Some(&stats));
		assert_eq!(stats.reversal_copies.load(Ordering::Relaxed), 1);
		assert_eq!(ledger.live(), 0);
	}

	#[test]
	fn off_touches_nothing() {
		let stats = MigStats::default();
		let ledger = MoveLedger::new(Mode::default());

		assert_eq!(ledger.hand_off(1, Tier::Fast, &stats), Tag::default());
		assert_eq!(ledger.check(1, Tag::default()), Verdict::Apply);
		ledger.finish(1, Tier::Fast, true, Some(&stats));
		assert_eq!(ledger.live(), 0);
		assert!(!ledger.sample_due(0), "nothing to sample");
	}
}
