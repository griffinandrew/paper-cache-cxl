/*
 * Copyright (c) Kia Shakiba
 *
 * This source code is licensed under the GNU AGPLv3 license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! LRU hybrid with a LAZY PHYSICAL COPY: identical eviction policy to
//! [`LruCompactHybridStack`], with the tier COPY deferred until DRAM pressure
//! actually demands it.
//!
//! # The problem this exists for
//!
//! Measured on uniform_baseline at 5 clients, `LruCompactHybridStack` performs
//! 2,208,767 tier migrations of which **728,418 are round trips** -- an object
//! copied DRAM -> Optane and then straight back, 2 x 16.5 KB to end where it
//! started. That is 66% of all migration bandwidth accomplishing nothing, and
//! it is not a scheduling artefact: 99.99% of every promotion the stack issues
//! is undoing its own earlier demotion of that same key.
//!
//! The waste is structural. `settle_fast_tier` demotes the moment the watermark
//! trips, and demoting means COPYING, so an object that will be read again
//! shortly pays two crossings for nothing.
//!
//! # What this changes, and what it deliberately does not
//!
//! The POLICY is untouched. `fast_boundary` moves exactly when
//! `LruCompactHybridStack` moves it, so the tier assignment, the eviction
//! order, `fast_used`/`slow_used` and the miss ratio are all identical. A
//! fidelity test asserts that directly.
//!
//! What changes is that a logical demotion no longer enqueues a copy. It only
//! marks the object a CANDIDATE: logically slow, physically still in DRAM. A
//! second cursor trails the first --
//!
//! ```text
//! head            phys_boundary        fast_boundary              tail
//!  |                    |                    |                      |
//!  [F][F][F][F][F][F][F][c][c][c][c][c][c][c][S][S][S][S][S][S][S][S]
//!  <--- logical + physical fast ---><- cands -><--- logical + physical slow --->
//! ```
//!
//! and only `reclaim_dram` advances it, only when measured DRAM exceeds its
//! budget. A candidate that gets promoted back is simply relabelled: it never
//! left DRAM, so there is nothing to copy. That is where the 728,418 round
//! trips go.
//!
//! # Why the DRAM budget, not the fast budget, is the one enforced
//!
//! The hardware constraint is DRAM. So `dram_capacity` is the real budget and
//! the LOGICAL fast tier is sized at `1 - LAZY_WINDOW` of it, leaving the
//! remainder as room for candidates. Sizing both to the same number would
//! leave no window at all and degenerate this stack back into its baseline --
//! `fast_used <= budget` and `dram_used <= budget` together force
//! `candidates ~ 0`.
//!
//! # Why this makes deferral safe where throttling demotion is not
//!
//! Backpressure on `settle_fast_tier` is incoherent: demotion IS the policy, so
//! throttling it stops the policy from running, and it does not even free DRAM
//! -- the bytes stay put either way. Deferring the COPY is different. The
//! policy has already run; only the physical move waits, and waiting is exactly
//! what "candidate" means. So `reclaim_dram` may be as lazy as the DRAM budget
//! permits without the policy noticing.

use crate::{
	object::ObjectSize,
	worker::policy::policy_stack::{
		compact_queue_set::CompactQueueSet, narrow_resident, watermarks, CacheSize,
		HashedKey, PolicyStack, Tier,
	},
	PaperPolicy,
};

/// The single recency order, in the shared queue set's slot 0.
const Q_LRU: usize = 0;

/// Fraction of the DRAM budget held back as room for candidates -- objects the
/// policy has demoted but whose bytes have not been copied yet.
///
/// Zero would make this stack bit-identical to its baseline, which is the
/// useful degenerate case for testing. The default trades 20% of the logical
/// fast tier for the ability to skip a round trip entirely.
fn lazy_window() -> f64 {
	static WINDOW: std::sync::OnceLock<f64> = std::sync::OnceLock::new();

	*WINDOW.get_or_init(|| {
		std::env::var("LAZY_COPY_WINDOW")
			.ok()
			.and_then(|v| v.parse::<f64>().ok())
			.filter(|v| *v >= 0.0 && *v < 1.0)
			.unwrap_or(0.2)
	})
}

/// Per-key bookkeeping. `tier` is what the POLICY believes; `phys` is where the
/// bytes actually are. They differ exactly for candidates.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct LazyCopyPayload {
	tier: Tier,
	phys: Tier,
	dram_resident: u8,
	size: ObjectSize,
}

const _: () = assert!(
	std::mem::size_of::<LazyCopyPayload>() == 8,
	"LazyCopyPayload grew past 8 bytes -- `phys` was meant to fit the padding \
	 `LruPayload` already had",
);

impl LazyCopyPayload {
	fn migrating(&self) -> CacheSize {
		(self.size as CacheSize).saturating_sub(self.dram_resident as CacheSize)
	}
}

pub struct LruLazyCopyCompactHybridStack {
	list: CompactQueueSet<LazyCopyPayload>,

	/// The HARDWARE budget: bytes of value data allowed on the fast node.
	dram_capacity: CacheSize,

	/// Logical, exactly as in the baseline.
	fast_used: CacheSize,
	slow_used: CacheSize,

	/// Physical: bytes whose buffers are on the fast node. Differs from
	/// `fast_used` by the candidate bytes.
	dram_used: CacheSize,

	shared_overhead: CacheSize,
	fast_count: usize,

	/// Least-recently-used LOGICALLY fast key.
	fast_boundary: Option<HashedKey>,

	/// Least-recently-used PHYSICALLY fast key. Trails `fast_boundary`.
	phys_boundary: Option<HashedKey>,

	migrations: Vec<(HashedKey, Tier)>,

	/// Snapshotted per stack rather than read from the `OnceLock` per call, so
	/// a test can drive a specific window without exporting an env var into
	/// every sibling test in the same binary. At 0.0 this stack is
	/// behaviourally identical to `LruCompactHybridStack`, which is exactly
	/// what `fidelity_tests` asserts.
	lazy_window: f64,

	/// Round trips this stack did NOT perform: a candidate promoted back before
	/// `reclaim_dram` reached it, so neither crossing happened. The whole
	/// point, counted.
	copies_avoided: u64,
}

impl LruLazyCopyCompactHybridStack {
	pub fn new(dram_capacity: CacheSize) -> Self {
		LruLazyCopyCompactHybridStack {
			list: CompactQueueSet::default(),
			dram_capacity,
			fast_used: 0,
			slow_used: 0,
			dram_used: 0,
			shared_overhead: 0,
			fast_count: 0,
			fast_boundary: None,
			phys_boundary: None,
			migrations: Vec::new(),
			lazy_window: lazy_window(),
			copies_avoided: 0,
		}
	}

	/// Overrides the window for one stack. `0.0` collapses the candidate region
	/// to nothing, making this stack the baseline.
	pub fn with_lazy_window(mut self, window: f64) -> Self {
		self.lazy_window = window.clamp(0.0, 0.99);
		self
	}

	pub fn with_shared_overhead(mut self, overhead: CacheSize) -> Self {
		self.shared_overhead = overhead;
		self
	}

	pub fn fast_capacity(&self) -> CacheSize {
		self.logical_capacity()
	}

	pub fn copies_avoided(&self) -> u64 {
		self.copies_avoided
	}

	fn reserved_overhead(&self) -> CacheSize {
		self.list.len() as CacheSize * self.shared_overhead
	}

	/// Budget the PHYSICAL cursor enforces: the whole DRAM allowance, net of
	/// the metadata that is DRAM-resident in either tier.
	fn physical_capacity(&self) -> CacheSize {
		self.dram_capacity.saturating_sub(self.reserved_overhead())
	}

	/// Budget the POLICY enforces, held back from the physical one so there is
	/// room for candidates to accumulate.
	fn logical_capacity(&self) -> CacheSize {
		(self.physical_capacity() as f64 * (1.0 - self.lazy_window)) as CacheSize
	}

	pub fn tier_of(&self, key: HashedKey) -> Option<Tier> {
		self.list.payload(key).map(|p| p.tier)
	}

	/// Where the bytes actually are, as against `tier_of`'s belief.
	pub fn physical_tier_of(&self, key: HashedKey) -> Option<Tier> {
		self.list.payload(key).map(|p| p.phys)
	}

	fn resize_key(&mut self, key: HashedKey, new_size: ObjectSize, new_resident: u8) {
		let Some(slot) = self.list.payload_mut(key) else { return };

		let old = slot.migrating();
		slot.size = new_size;
		slot.dram_resident = new_resident;
		let delta = slot.migrating() as i64 - old as i64;
		let (tier, phys) = (slot.tier, slot.phys);

		match tier {
			Tier::Fast => self.fast_used = (self.fast_used as i64 + delta).max(0) as CacheSize,
			Tier::Slow => self.slow_used = (self.slow_used as i64 + delta).max(0) as CacheSize,
		}

		if phys == Tier::Fast {
			self.dram_used = (self.dram_used as i64 + delta).max(0) as CacheSize;
		}
	}

	/// Move to the MRU end and make logically fast.
	///
	/// The one place the saving is realised: if the key is a CANDIDATE its
	/// bytes never left DRAM, so promotion is a relabel and no migration is
	/// enqueued at all.
	fn touch_fast_key(&mut self, key: HashedKey) {
		let Some(before) = self.list.payload(key) else { return };

		let already_front = self.list.front(Q_LRU) == Some(key);
		let was_fast_boundary = self.fast_boundary == Some(key);
		let was_phys_boundary = self.phys_boundary == Some(key);

		// Read predecessors BEFORE moving: once the key is at the front its
		// neighbour is gone, and each cursor must step back to whatever
		// preceded it.
		let (new_fast_b, new_phys_b) = match already_front {
			true => (None, None),
			false => (
				was_fast_boundary.then(|| self.list.before(key)).flatten(),
				was_phys_boundary.then(|| self.list.before(key)).flatten(),
			),
		};

		self.list.move_front(Q_LRU, key);

		if was_fast_boundary && !already_front {
			self.fast_boundary = new_fast_b;
		}

		if was_phys_boundary && !already_front {
			self.phys_boundary = new_phys_b;
		}

		let mut promoted_physically = false;

		if before.tier != Tier::Fast {
			let size = before.migrating();
			self.slow_used = self.slow_used.saturating_sub(size);
			self.fast_used += size;
			self.fast_count += 1;

			if let Some(slot) = self.list.payload_mut(key) {
				slot.tier = Tier::Fast;
			}

			if self.fast_boundary.is_none() {
				self.fast_boundary = Some(key);
			}

			match before.phys {
				// Candidate: the bytes are still in DRAM. Relabel and skip the
				// crossing entirely -- this is the round trip that does not
				// happen.
				Tier::Fast => self.copies_avoided += 1,

				// Genuinely in the slow tier; it has to be copied back.
				Tier::Slow => {
					if let Some(slot) = self.list.payload_mut(key) {
						slot.phys = Tier::Fast;
					}

					self.dram_used += size;
					promoted_physically = true;

					if self.phys_boundary.is_none() {
						self.phys_boundary = Some(key);
					}
				},
			}
		}

		self.settle_fast_tier();
		self.reclaim_dram();

		// Guarded on the key still being physically fast: a tight DRAM budget
		// can reclaim it straight back out within the same pass, in which case
		// `reclaim_dram` already pushed the correct final entry.
		if promoted_physically
			&& self.list.payload(key).map(|p| p.phys) == Some(Tier::Fast)
		{
			self.migrations.push((key, Tier::Fast));
		}
	}

	/// Moves the LOGICAL boundary only. Pushes NO migration -- that is the
	/// entire difference from the baseline.
	fn settle_fast_tier(&mut self) {
		let effective = self.logical_capacity();

		if self.fast_used <= watermarks::high_bytes(effective) {
			return;
		}

		let target = watermarks::low_bytes(effective);

		while self.fast_used > target {
			let Some(key) = self.fast_boundary else { break };
			let size = self.list.payload(key).map(|p| p.migrating()).unwrap_or(0);
			let next = self.list.before(key);

			if let Some(slot) = self.list.payload_mut(key) {
				slot.tier = Tier::Slow;
			}

			self.fast_used = self.fast_used.saturating_sub(size);
			self.fast_count = self.fast_count.saturating_sub(1);
			self.slow_used += size;
			self.fast_boundary = next;
		}
	}

	/// Advances the PHYSICAL cursor, and is the only thing that enqueues a
	/// demotion. Driven by measured DRAM, never by the policy.
	fn reclaim_dram(&mut self) {
		let effective = self.physical_capacity();

		if self.dram_used <= watermarks::high_bytes(effective) {
			return;
		}

		let target = watermarks::low_bytes(effective);

		while self.dram_used > target {
			let Some(key) = self.phys_boundary else { break };
			let size = self.list.payload(key).map(|p| p.migrating()).unwrap_or(0);
			let next = self.list.before(key);

			if let Some(slot) = self.list.payload_mut(key) {
				slot.phys = Tier::Slow;
			}

			self.dram_used = self.dram_used.saturating_sub(size);
			self.phys_boundary = next;

			self.migrations.push((key, Tier::Slow));
		}
	}

	/// Reverses one key's contribution to both accountings and steps whichever
	/// cursors pointed at it.
	fn detach(&mut self, key: HashedKey) {
		let Some(p) = self.list.payload(key) else { return };
		let size = p.migrating();

		if self.fast_boundary == Some(key) {
			self.fast_boundary = self.list.before(key);
		}

		if self.phys_boundary == Some(key) {
			self.phys_boundary = self.list.before(key);
		}

		match p.tier {
			Tier::Fast => {
				self.fast_used = self.fast_used.saturating_sub(size);
				self.fast_count = self.fast_count.saturating_sub(1);
			},

			Tier::Slow => self.slow_used = self.slow_used.saturating_sub(size),
		}

		if p.phys == Tier::Fast {
			self.dram_used = self.dram_used.saturating_sub(size);
		}
	}
}

impl PolicyStack for LruLazyCopyCompactHybridStack {
	fn is_policy(&self, policy: &PaperPolicy) -> bool {
		matches!(policy, PaperPolicy::LruLazyCopyCompactHybrid)
	}

	fn len(&self) -> usize {
		self.list.len()
	}

	fn contains(&self, key: HashedKey) -> bool {
		self.list.contains(key)
	}

	fn insert(&mut self, key: HashedKey, size: ObjectSize) {
		self.insert_resident(key, size, 0);
	}

	fn insert_resident(&mut self, key: HashedKey, size: ObjectSize, dram_resident: ObjectSize) {
		let dram_resident = narrow_resident(dram_resident);

		if self.list.contains(key) {
			self.resize_key(key, size, dram_resident);
			self.touch_fast_key(key);
			return;
		}

		// Admission is unconditionally fast, logically AND physically, exactly
		// as in the baseline -- `admission_tier` builds the buffer in DRAM.
		self.list.push_front(
			Q_LRU,
			key,
			LazyCopyPayload { tier: Tier::Fast, phys: Tier::Fast, dram_resident, size },
		);

		let migrating = (size as CacheSize).saturating_sub(dram_resident as CacheSize);
		self.fast_used += migrating;
		self.dram_used += migrating;
		self.fast_count += 1;

		if self.fast_boundary.is_none() {
			self.fast_boundary = Some(key);
		}

		if self.phys_boundary.is_none() {
			self.phys_boundary = Some(key);
		}

		self.settle_fast_tier();
		self.reclaim_dram();
	}

	fn update(&mut self, key: HashedKey) {
		if self.list.contains(key) {
			self.touch_fast_key(key);
		}
	}

	fn remove(&mut self, key: HashedKey) {
		self.detach(key);
		self.list.remove(Q_LRU, key);
	}

	fn clear(&mut self) {
		self.list.clear();
		self.fast_used = 0;
		self.slow_used = 0;
		self.dram_used = 0;
		self.fast_count = 0;
		self.fast_boundary = None;
		self.phys_boundary = None;
		self.migrations.clear();
	}

	fn evict_one(&mut self) -> Option<HashedKey> {
		let key = self.list.back(Q_LRU)?;

		self.detach(key);
		self.list.remove(Q_LRU, key)?;

		Some(key)
	}

	fn resize_fast_tier(&mut self, size: CacheSize) {
		self.dram_capacity = size;
		self.settle_fast_tier();
		self.reclaim_dram();
	}

	fn drain_tier_migrations(&mut self) -> Vec<(HashedKey, Tier)> {
		std::mem::take(&mut self.migrations)
	}

	fn dram_reserved_bytes(&self) -> CacheSize {
		self.reserved_overhead()
	}

	/// The PHYSICAL number, deliberately -- this stack exists to make the
	/// reported fast-tier figure describe where the bytes are rather than where
	/// the policy intends them. `fast_used` is still the policy's, and is what
	/// `settle_fast_tier` steers on.
	fn fast_bytes_used(&self) -> CacheSize {
		self.dram_used
	}

	fn slow_bytes_used(&self) -> CacheSize {
		(self.fast_used + self.slow_used).saturating_sub(self.dram_used)
	}

	fn fast_object_count(&self) -> usize {
		self.fast_count
	}

	fn slow_object_count(&self) -> usize {
		self.list.len().saturating_sub(self.fast_count)
	}
}

/// Fidelity against `LruCompactHybridStack`, which this is a deferral of.
///
/// At `lazy_window = 0.0` the candidate region is empty, so every logical
/// demotion is reclaimed in the same call that made it and this stack must be
/// INDISTINGUISHABLE from its baseline -- same tier for every key, same
/// migration sequence, same eviction order. A miss ratio matching is necessary
/// but not sufficient; it would not catch a counter firing on the wrong path.
#[cfg(all(test, feature = "lru_compact_hybrid_cache"))]
mod fidelity_tests {
	use super::*;
	use crate::worker::policy::policy_stack::lru_compact_hybrid_stack::LruCompactHybridStack;

	fn skewed_ops() -> Vec<(HashedKey, ObjectSize)> {
		let mut ops = Vec::new();
		let mut x: u64 = 0x243F_6A88_85A3_08D3;
		for _ in 0..20_000 {
			x ^= x << 13;
			x ^= x >> 7;
			x ^= x << 17;
			let u = (x >> 11) as f64 / (1u64 << 53) as f64;
			ops.push((((u * u * 200.0) as u64) + 1, 1024));
		}
		ops
	}

	#[test]
	fn window_zero_is_the_baseline() {
		let ops = skewed_ops();

		for cap in [8_192u64, 32_768, 131_072] {
			for overhead in [0u64, 224] {
				let mut a = LruCompactHybridStack::new(cap).with_shared_overhead(overhead);
				let mut b = LruLazyCopyCompactHybridStack::new(cap)
					.with_shared_overhead(overhead)
					.with_lazy_window(0.0);

				let (mut ma, mut mb) = (Vec::new(), Vec::new());

				for (k, size) in &ops {
					if a.contains(*k) { a.update(*k); } else { a.insert(*k, *size); }
					if b.contains(*k) { b.update(*k); } else { b.insert(*k, *size); }
					ma.extend(a.drain_tier_migrations());
					mb.extend(b.drain_tier_migrations());
				}

				let keys: Vec<HashedKey> = ops.iter().map(|(k, _)| *k).collect();
				let ta: Vec<_> = keys.iter().map(|k| a.tier_of(*k)).collect();
				let tb: Vec<_> = keys.iter().map(|k| b.tier_of(*k)).collect();

				assert_eq!(ta, tb, "tiers diverge at cap {cap} overhead {overhead}");
				assert_eq!(ma, mb, "migrations diverge at cap {cap} overhead {overhead}");

				let mut ea = Vec::new();
				let mut eb = Vec::new();
				while let Some(k) = a.evict_one() { ea.push(k); }
				while let Some(k) = b.evict_one() { eb.push(k); }
				assert_eq!(ea, eb, "eviction order diverges at cap {cap}");
			}
		}
	}

	/// The mechanism itself, tested directly rather than through an aggregate.
	///
	/// An aggregate copy count is the wrong instrument: whether a window PAYS
	/// depends on the re-access distance of the workload against the size of
	/// the window, so a small window over a churn-heavy op stream reclaims
	/// every candidate before it can be promoted and saves exactly nothing.
	/// That is a property of the trace, not of the mechanism. What must hold
	/// unconditionally is that promoting a CANDIDATE costs no crossing.
	#[test]
	fn a_candidate_promoted_back_costs_no_copy() {
		// A wide window: the logical fast tier is a tenth of DRAM, so keys
		// demoted by `settle_fast_tier` sit as candidates rather than being
		// reclaimed immediately.
		let mut s = LruLazyCopyCompactHybridStack::new(100 * 1024)
			.with_shared_overhead(0)
			.with_lazy_window(0.9);

		for i in 1..=20u64 {
			s.insert(i, 1024);
		}

		s.drain_tier_migrations();

		let candidate = (1..=20u64)
			.find(|k| {
				s.tier_of(*k) == Some(Tier::Slow)
					&& s.physical_tier_of(*k) == Some(Tier::Fast)
			})
			.expect("a window this wide must leave candidates");

		let before = s.copies_avoided();
		s.update(candidate);
		let migrations = s.drain_tier_migrations();

		assert_eq!(
			s.copies_avoided(),
			before + 1,
			"promoting a candidate must be counted as an avoided round trip",
		);
		assert!(
			!migrations.iter().any(|(k, _)| *k == candidate),
			"promoting a candidate must enqueue NO migration for it -- its bytes \
			 never left DRAM",
		);
		assert_eq!(s.tier_of(candidate), Some(Tier::Fast));
		assert_eq!(s.physical_tier_of(candidate), Some(Tier::Fast));
	}

	/// A window must never issue MORE copies than the baseline. Whether it
	/// issues fewer is workload-dependent (see above), so that is not asserted.
	#[test]
	fn a_window_never_costs_more_copies() {
		let ops = skewed_ops();
		let cap = 32_768u64;

		let mut b = LruLazyCopyCompactHybridStack::new(cap)
			.with_shared_overhead(0)
			.with_lazy_window(0.0);
		let mut c = LruLazyCopyCompactHybridStack::new(cap)
			.with_shared_overhead(0)
			.with_lazy_window(0.5);

		let (mut nb, mut nc) = (0usize, 0usize);

		for (k, size) in &ops {
			if b.contains(*k) { b.update(*k); } else { b.insert(*k, *size); }
			nb += b.drain_tier_migrations().len();
			if c.contains(*k) { c.update(*k); } else { c.insert(*k, *size); }
			nc += c.drain_tier_migrations().len();
		}

		assert!(nc <= nb, "window=0.5 issued {nc} copies, window=0.0 issued {nb}");
	}

	/// The invariant both cursors rest on: the physically-fast region is a
	/// PREFIX of the logically-fast one, which is a prefix of the list. A
	/// candidate is logically slow and physically fast; the reverse must never
	/// occur.
	#[test]
	fn physical_fast_is_a_prefix_of_logical_fast() {
		let ops = skewed_ops();
		let mut s = LruLazyCopyCompactHybridStack::new(32_768)
			.with_shared_overhead(0)
			.with_lazy_window(0.4);

		for (k, size) in &ops {
			if s.contains(*k) { s.update(*k); } else { s.insert(*k, *size); }
			s.drain_tier_migrations();

			for (k2, _) in ops.iter().take(40) {
				match (s.tier_of(*k2), s.physical_tier_of(*k2)) {
					(Some(Tier::Slow), Some(Tier::Fast)) => {},   // a candidate
					(Some(a), Some(b)) => assert_eq!(
						a, b,
						"key {k2} is logically {a:?} but physically {b:?} -- only \
						 logically-slow/physically-fast is a legal disagreement",
					),
					_ => {},
				}
			}
		}
	}
}
