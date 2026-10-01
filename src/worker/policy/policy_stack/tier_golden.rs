/*
 * Copyright (c) Kia Shakiba
 *
 * This source code is licensed under the GNU AGPLv3 license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! Golden fingerprints of all 23 tiered designs, on `dyn PolicyStack` (R4,
//! step 0).
//!
//! The tiering bookkeeping every hybrid stack replicates is moving into one
//! layer, and a design is to become a layout plus a few hooks. What must not
//! move is what each design DOES, including the accidents: the redundant +F on
//! an already-fast key's second chance, the settle set each entry point runs,
//! where a structural key is homed, which budget is raw and which is drained,
//! what a ghost remembers. Only four designs have a byte-level oracle (T14, the
//! merged store); the rest were pinned by per-design invariants, which do not
//! see an eviction order or a migration that came out different. So this is
//! the oracle for all 23, recorded from the stacks as they were before any of
//! them was ported (`tier_goldens.txt`, checked in) and required of every one
//! that is.
//!
//! # What is fingerprinted
//!
//! Each run builds one design and applies a few thousand operations drawn from
//! a seeded stream: sets of new and tracked keys (some larger than an empty
//! fast tier, some flagged structural by the client), hits, removals of
//! tracked and untracked keys, `evict_one`, the worker's eviction pass
//! (`needs_capacity_eviction` or over `max_size`), `resize`,
//! `resize_fast_tier`, `set_dram_metadata(Some | None)`, `resettle`, `clear`
//! (and LruSized's two resizes). After EVERY operation it folds into an FNV-1a
//! fingerprint what a caller can observe: the operation's answer (the
//! placement applied, the victim), the ORDERED migrations the stack queued,
//! `len`, `contains` and `placement_of` of the key touched, the twelve gauges,
//! `dram_reserved_bytes`, `structure_bytes`, `needs_capacity_eviction`, and the
//! LFU latch trio. Every sixteenth operation it folds `contains` and
//! `placement_of` of EVERY key of the run's universe. A stack that evicts one
//! key differently, queues one migration in a different order, charges one byte
//! differently or forgets one ghost changes the fingerprint.
//!
//! # The grid
//!
//! Per design: two seeds (60 and 250 keys), three fast tiers (8 KiB: one or two
//! values fit; 48 KiB: eff crosses the item sizes as the keys and the
//! reservation grow; 1 MiB: nothing demotes), each with its own `max_size`
//! (8x the tier), three per-object reservations (0, 64, 400 B: the last
//! leaves no room for values at a few hundred keys), and, for the designs that
//! have one, three ratios (the 2Q and S3-FIFO probation share of `max_size`:
//! small enough that a carve-out fits the tier, about equal to it, and
//! larger; promote_k for LRU-LFU; the size threshold for LruSized). The drain
//! target (`FAST_TIER_DRAIN_TARGET`) is read once per process, so the 0.95
//! default and 1.0 are two tests, the second in a child process.
//!
//! Wider universes follow: `w3` (600 keys) and `w4` (2,000), each at three
//! fast tiers (8, 24 and 64 KiB) and the same reservations and ratios, and
//! `w0` (40 keys) at 24 KiB alone. A cache of hundreds of keys against a tier
//! of a few is where a rule that matters only while the stack is over its
//! budget, and nothing has settled it yet, shows: a `resize` of the cache that
//! also settles, for one, which the two seeds above show in the S3-FIFO
//! designs alone, and these show in 2Q, 2Q-ghost, LRU, FIFO and CLOCK as well
//! (by one or two runs each; LRU's only in `w0`, at the default target). They
//! are drawn the way the widened differential against the legacy stacks drew
//! its seeds (an xorshift from `0x9E37_79B9_7F4A_7C15`; the number in the name
//! is the place in that list), so each of them was recorded on the stacks the
//! layer replaced as well, and a design ported later is held to that too.
//!
//! # Using it
//!
//!   * A mismatch names every (design, configuration) that moved. Re-run with
//!     `PAPER_TIER_GOLDEN_ONLY=<design>/<configuration>` (substrings) and
//!     `PAPER_TIER_GOLDEN_DUMP=<dir>` for a line per operation -- the op, its
//!     answer, the migrations, the gauges -- in `<dir>/<design>/<cfg>.txt`,
//!     from a build of each tree, and diff the two: the first differing line is
//!     the first operation at which they disagree.
//!   * `PAPER_TIER_GOLDEN_OUT=<dir>` records instead of checking: it writes
//!     `<dir>/drain-<ratio>.txt` (the fingerprints, and at the default target
//!     the `seen` lines below), runs every configuration twice to show the
//!     recording is deterministic, and T14's scripts (`t14x_*` in
//!     `s4_tests.rs`, which ask [`check_script`] for their hashes) each write
//!     `<dir>/t14-<script>.txt`. `tier_goldens.txt` is those files together.
//!   * The coverage table: each design's runs count what they observed of the
//!     stack's paths from outside (see [`Event`]), and the events a design
//!     reached when this was recorded are listed in `tier_goldens.txt` as
//!     `seen <design> <event>`. A design that stops reaching one fails the
//!     default-target test, so a change to the stream cannot quietly turn a
//!     golden into a test of less. That a golden is SENSITIVE to what happens
//!     inside a path is shown by mutating the stacks, not by this table.
//!   * `admission::the_clients_build_tier_is_what_every_design_does_to_a_set`:
//!     `hybrid_policy::admission_tier` repeats each design's rule for where a
//!     set's value is built, and the worker's reconcile assumes it says what
//!     the design's `Set` then does to the key's placement; here the real
//!     function is asked and the two are held together, for every design.

use std::{collections::BTreeMap, fmt::Write as _, path::PathBuf};

use super::{
	ClockCompactHybridStack, FifoCompactHybridStack, LfuCompactHybridStack, LruCompactHybridStack,
	LruLfuCompactHybridStack, LruSizedCompactHybridStack, PolicyStack, S3FifoCompactHybridStack,
	S3FifoFaithfulCompactHybridStack, S3FifoFaithfulFastAdmissionCompactHybridStack,
	S3FifoFaithfulFastAdmissionReprieveCompactHybridStack, S3FifoFaithfulReprieveCompactHybridStack,
	S3FifoGhostCompactHybridStack, S3FifoGhostLazyDemotionCompactHybridStack,
	S3FifoGhostLazyDemotionFastAdmissionCompactHybridStack,
	S3FifoGhostLazyDemotionFastAdmissionMidpointCompactHybridStack,
	S3FifoLazyDemotionFastAdmissionMidpointReprieveCompactHybridStack,
	S3FifoLazyDemotionFastAdmissionReprieveCompactHybridStack,
	S3FifoLazyDemotionFastAdmissionSplitSlowReprieveCompactHybridStack,
	S3FifoLazyDemotionReprieveCompactHybridStack, SetEvent, Tier, TwoQCompactHybridStack,
	TwoQFastAdmissionReprieveCompactHybridStack, TwoQFullFastAdmissionCompactHybridStack,
	TwoQGhostCompactHybridStack, Placement, drain_target,
};
use crate::{object::ObjectSize, CacheSize, HashedKey, PaperPolicy};

/// Every fingerprint recorded before any design was ported, one per line:
/// `<drain target> <design> <configuration> <fingerprint>`.
const RECORDED: &str = include_str!("tier_goldens.txt");

/// Operations per run.
const OPS: usize = 6_000;

/// How often every key of the universe is folded in.
const AUDIT_EVERY: usize = 16;

// ---------------------------------------------------------------------------
// the designs
// ---------------------------------------------------------------------------

/// One point of the grid.
#[derive(Clone, Copy)]
struct Cfg {
	fast: CacheSize,
	max: CacheSize,
	omega: CacheSize,
	ratio: f64,
	seed: u64,
	keys: u64,
}

struct Design {
	name: &'static str,
	/// The `PaperPolicy` it implements (its payload is not meaningful here),
	/// for the admission test, which the merged builds do not have.
	#[cfg_attr(feature = "merged_object_store", allow(dead_code))]
	policy: PaperPolicy,
	/// The per-design knob, where it has one: 2Q's `k_in`, S3-FIFO's
	/// one-access share, LRU-LFU's `promote_k`, LruSized's size threshold.
	ratios: &'static [f64],
	build: fn(&Cfg) -> Box<dyn PolicyStack>,
}

const NONE: &[f64] = &[0.0];
const SHARES: &[f64] = &[0.05, 0.12, 0.4];

macro_rules! built {
	($stack:expr, $cfg:ident) => {
		Box::new($stack.with_shared_overhead($cfg.omega))
	};
}

/// All 23, in the order the recorded file lists them.
const DESIGNS: [Design; 23] = [
	Design { name: "lru", policy: PaperPolicy::LruCompactHybrid, ratios: NONE, build: |c| built!(LruCompactHybridStack::new(c.fast), c) },
	Design { name: "fifo", policy: PaperPolicy::FifoCompactHybrid, ratios: NONE, build: |c| built!(FifoCompactHybridStack::new(c.fast), c) },
	Design { name: "clock", policy: PaperPolicy::ClockCompactHybrid, ratios: NONE, build: |c| built!(ClockCompactHybridStack::new(c.fast), c) },
	Design { name: "lfu", policy: PaperPolicy::LfuCompactHybrid, ratios: NONE, build: |c| built!(LfuCompactHybridStack::new(c.fast), c) },
	Design {
		name: "lru-lfu", policy: PaperPolicy::LruLfuCompactHybrid(3),
		ratios: &[2.0, 4.0, 9.0],
		build: |c| built!(LruLfuCompactHybridStack::new(c.fast, c.ratio as u16), c),
	},
	Design {
		name: "lru-sized", policy: PaperPolicy::LruSizedCompactHybrid,
		ratios: &[700.0, 1_500.0, 3_000.0],
		build: |c| built!(LruSizedCompactHybridStack::new(c.fast / 2, c.fast - c.fast / 2, c.ratio as CacheSize), c),
	},
	Design { name: "q2", policy: PaperPolicy::TwoQCompactHybrid(0.1), ratios: SHARES, build: |c| built!(TwoQCompactHybridStack::new(c.ratio, c.max, c.fast), c) },
	Design { name: "q2gh", policy: PaperPolicy::TwoQGhostCompactHybrid(0.1), ratios: SHARES, build: |c| built!(TwoQGhostCompactHybridStack::new(c.ratio, c.max, c.fast), c) },
	Design {
		name: "q2far", policy: PaperPolicy::TwoQFastAdmissionReprieveCompactHybrid(0.1),
		ratios: SHARES,
		build: |c| built!(TwoQFastAdmissionReprieveCompactHybridStack::new(c.ratio, c.max, c.fast), c),
	},
	Design {
		name: "q2full", policy: PaperPolicy::TwoQFullFastAdmissionCompactHybrid(0.1, 0.5),
		ratios: SHARES,
		build: |c| built!(TwoQFullFastAdmissionCompactHybridStack::new(c.ratio, 0.5, c.max, c.fast), c),
	},
	Design { name: "s3", policy: PaperPolicy::S3FifoCompactHybrid(0.1), ratios: SHARES, build: |c| built!(S3FifoCompactHybridStack::new(c.ratio, c.max, c.fast), c) },
	Design { name: "s3gh", policy: PaperPolicy::S3FifoGhostCompactHybrid(0.1), ratios: SHARES, build: |c| built!(S3FifoGhostCompactHybridStack::new(c.ratio, c.max, c.fast), c) },
	Design {
		name: "gld", policy: PaperPolicy::S3FifoGhostLazyDemotionCompactHybrid(0.1),
		ratios: SHARES,
		build: |c| built!(S3FifoGhostLazyDemotionCompactHybridStack::new(c.ratio, c.max, c.fast), c),
	},
	Design {
		name: "gldfa", policy: PaperPolicy::S3FifoGhostLazyDemotionFastAdmissionCompactHybrid(0.1),
		ratios: SHARES,
		build: |c| built!(S3FifoGhostLazyDemotionFastAdmissionCompactHybridStack::new(c.ratio, c.max, c.fast), c),
	},
	Design {
		name: "mid21", policy: PaperPolicy::S3FifoGhostLazyDemotionFastAdmissionMidpointCompactHybrid(0.1),
		ratios: SHARES,
		build: |c| built!(S3FifoGhostLazyDemotionFastAdmissionMidpointCompactHybridStack::new(c.ratio, c.max, c.fast), c),
	},
	Design {
		name: "ldr", policy: PaperPolicy::S3FifoLazyDemotionReprieveCompactHybrid(0.1),
		ratios: SHARES,
		build: |c| built!(S3FifoLazyDemotionReprieveCompactHybridStack::new(c.ratio, c.max, c.fast), c),
	},
	Design {
		name: "ldfar", policy: PaperPolicy::S3FifoLazyDemotionFastAdmissionReprieveCompactHybrid(0.1),
		ratios: SHARES,
		build: |c| built!(S3FifoLazyDemotionFastAdmissionReprieveCompactHybridStack::new(c.ratio, c.max, c.fast), c),
	},
	Design {
		name: "mid22", policy: PaperPolicy::S3FifoLazyDemotionFastAdmissionMidpointReprieveCompactHybrid(0.1),
		ratios: SHARES,
		build: |c| built!(S3FifoLazyDemotionFastAdmissionMidpointReprieveCompactHybridStack::new(c.ratio, c.max, c.fast), c),
	},
	Design {
		name: "split25", policy: PaperPolicy::S3FifoLazyDemotionFastAdmissionSplitSlowReprieveCompactHybrid(0.1),
		ratios: SHARES,
		build: |c| built!(S3FifoLazyDemotionFastAdmissionSplitSlowReprieveCompactHybridStack::new(c.ratio, c.max, c.fast), c),
	},
	Design { name: "faith", policy: PaperPolicy::S3FifoFaithfulCompactHybrid(0.1), ratios: SHARES, build: |c| built!(S3FifoFaithfulCompactHybridStack::new(c.ratio, c.max, c.fast), c) },
	Design {
		name: "faith-fa", policy: PaperPolicy::S3FifoFaithfulFastAdmissionCompactHybrid(0.1),
		ratios: SHARES,
		build: |c| built!(S3FifoFaithfulFastAdmissionCompactHybridStack::new(c.ratio, c.max, c.fast), c),
	},
	Design {
		name: "faith-r", policy: PaperPolicy::S3FifoFaithfulReprieveCompactHybrid(0.1),
		ratios: SHARES,
		build: |c| built!(S3FifoFaithfulReprieveCompactHybridStack::new(c.ratio, c.max, c.fast), c),
	},
	Design {
		name: "faith-fa-r", policy: PaperPolicy::S3FifoFaithfulFastAdmissionReprieveCompactHybrid(0.1),
		ratios: SHARES,
		build: |c| built!(S3FifoFaithfulFastAdmissionReprieveCompactHybridStack::new(c.ratio, c.max, c.fast), c),
	},
];

/// The configurations of one design, in the order they are recorded.
fn grid(design: &Design) -> Vec<(String, Cfg)> {
	let mut out = Vec::new();

	for (s, (seed, keys)) in [(0x243F_6A88_85A3_08D3u64, 60u64), (0x1357_9BDF_2468_ACE0, 250)].into_iter().enumerate() {
		for (f, fast) in [8 * 1024u64, 48 * 1024, 1024 * 1024].into_iter().enumerate() {
			for omega in [0u64, 64, 400] {
				for (r, &ratio) in design.ratios.iter().enumerate() {
					let cfg = Cfg { fast, max: fast * 8, omega, ratio, seed: seed ^ (f as u64) << 8, keys };

					out.push((format!("s{s}-f{f}-o{omega}-r{r}"), cfg));
				}
			}
		}
	}

	for (w, seed, keys, tiers) in WIDE {
		for &f in tiers {
			for omega in [0u64, 64, 400] {
				for (r, &ratio) in design.ratios.iter().enumerate() {
					let fast = WIDE_TIERS[f];
					let cfg = Cfg { fast, max: fast * 8, omega, ratio, seed: seed ^ (f as u64) << 8, keys };

					out.push((format!("w{w}-f{f}-o{omega}-r{r}"), cfg));
				}
			}
		}
	}

	out
}

/// The wide universes: their place in the differential's list of seeds, the
/// seed, the keys, and the tiers (indices into `WIDE_TIERS`) they run at.
const WIDE: [(usize, u64, u64, &[usize]); 3] = [
	(0, 0xDC1B_77AE_0BF3_4DAD, 40, &[1]),
	(3, 0x305F_050C_368D_CC74, 600, &[0, 1, 2]),
	(4, 0x2CEB_16E0_A1C5_4AEC, 2_000, &[0, 1, 2]),
];

/// Their fast tiers: a few values, then the reservation crossing the item sizes.
const WIDE_TIERS: [u64; 3] = [8 * 1024, 24 * 1024, 64 * 1024];

// ---------------------------------------------------------------------------
// the stream
// ---------------------------------------------------------------------------

struct Rng(u64);

impl Rng {
	fn next(&mut self) -> u64 {
		self.0 ^= self.0 << 13;
		self.0 ^= self.0 >> 7;
		self.0 ^= self.0 << 17;
		self.0.wrapping_mul(0x2545_F491_4F6C_DD1D)
	}

	fn below(&mut self, n: u64) -> u64 {
		(self.next() >> 11) % n.max(1)
	}

	fn chance(&mut self, percent: u64) -> bool {
		self.below(100) < percent
	}
}

/// FNV-1a over 64-bit words, as `golden.rs`.
#[derive(Clone, Copy)]
struct Fnv(u64);

impl Fnv {
	fn new() -> Self {
		Fnv(0xCBF2_9CE4_8422_2325)
	}

	fn feed(&mut self, word: u64) {
		for byte in word.to_le_bytes() {
			self.0 = (self.0 ^ byte as u64).wrapping_mul(0x0000_0100_0000_01B3);
		}
	}
}

fn tier_code(tier: Option<Tier>) -> u64 {
	match tier {
		None => 0,
		Some(Tier::Fast) => 1,
		Some(Tier::Slow) => 2,
	}
}

fn tier_char(tier: Option<Tier>) -> char {
	match tier {
		None => '-',
		Some(Tier::Fast) => 'F',
		Some(Tier::Slow) => 'S',
	}
}

fn placement_code(placement: Placement) -> u64 {
	match placement {
		Placement::Normal => 0,
		Placement::Structural => 1,
		Placement::Diverted => 2,
	}
}

// ---------------------------------------------------------------------------
// what a run saw of the paths
// ---------------------------------------------------------------------------

/// Paths seen from OUTSIDE the stack: an operation, the tier `placement_of`
/// gave the key before and after, and the migrations it queued. They prove the
/// stream reaches a class of behaviour at all; that a golden is sensitive to
/// what happens inside one is shown by mutating the stack (the ports' red-first
/// runs), not by this table.
macro_rules! events {
	($($name:ident),* $(,)?) => {
		#[derive(Clone, Copy, PartialEq, Eq, Debug)]
		enum Event { $($name),* }

		const EVENT_NAMES: &[&str] = &[$(stringify!($name)),*];
		const EVENTS: usize = EVENT_NAMES.len();
	};
}

events! {
	SetNewFast,
	SetNewSlow,
	SetNewStructuralFlagged,
	SetNewStructuralOwnCheck,
	SetNewPromoted,
	SetOverFastFast,
	SetOverFastSlow,
	SetOverSlowFast,
	SetOverSlowSlow,
	SetOverStructural,
	SetOverPromoted,
	SetDemoted,
	SetOverDemoted,
	HitFastFast,
	HitFastSlow,
	HitSlowFast,
	HitSlowSlow,
	HitPromoted,
	HitDemoted,
	RemoveFast,
	RemoveSlow,
	RemoveAbsent,
	EvictFast,
	EvictSlow,
	EvictNone,
	EvictPromotedAnother,
	EvictPromotedAFastKey,
	EvictDemotedAnother,
	EvictedKeyReadmitted,
	NeedsEviction,
	ResizeFastDemoted,
	ResizeFastGrown,
	ResettleDemoted,
	MeasuredDemoted,
	ResizeMax,
	Cleared,
	LatchSet,
	LatchCleared,
	DemotionsCounted,
	SmallFast,
	LargeFast,
	SmallSlow,
	LargeSlow,
	TwoOrMoreMigrations,
	BothDirections,
}

// ---------------------------------------------------------------------------
// one run
// ---------------------------------------------------------------------------

/// What one run produced.
struct Outcome {
	fingerprint: u64,
	seen: [u64; EVENTS],
	/// One line per operation, when a dump was asked for.
	dump: Option<String>,
}

struct Run<'a> {
	stack: Box<dyn PolicyStack>,
	cfg: &'a Cfg,
	sized: bool,
	fp: Fnv,
	seen: [u64; EVENTS],
	dump: Option<String>,
	/// Keys the stack tracks, by `contains` after the operations that can
	/// change it (not a model of the stack's business).
	tracked: Vec<HashedKey>,
	/// The last keys `evict_one` returned, to be re-admitted (ghost hits).
	evicted: Vec<HashedKey>,
	latched: bool,
}

impl<'a> Run<'a> {
	fn see(&mut self, event: Event) {
		self.seen[event as usize] += 1;
	}

	fn key(&self, index: u64) -> HashedKey {
		index + 1
	}

	fn note_tracked(&mut self, key: HashedKey) {
		let is = self.stack.contains(key);
		let at = self.tracked.iter().position(|&k| k == key);

		match (is, at) {
			(true, None) => self.tracked.push(key),
			(false, Some(i)) => { self.tracked.swap_remove(i); },
			_ => {},
		}
	}

	/// A key the stack tracks most of the time, else any key of the universe.
	fn pick(&self, rng: &mut Rng) -> HashedKey {
		if !self.tracked.is_empty() && rng.chance(88) {
			return self.tracked[rng.below(self.tracked.len() as u64) as usize];
		}

		self.key(rng.below(self.cfg.keys))
	}

	fn size(&self, rng: &mut Rng) -> ObjectSize {
		match rng.below(100) {
			0..3 => 16 + rng.below(112) as ObjectSize,
			3..58 => 200 + rng.below(1_300) as ObjectSize,
			58..85 => 1_500 + rng.below(2_500) as ObjectSize,
			85..95 => (self.cfg.fast / 4 + rng.below(self.cfg.fast / 4 + 1)) as ObjectSize,
			// Larger than an empty fast tier: structural by the stack's own check.
			_ => (self.cfg.fast + rng.below(self.cfg.fast / 2 + 1)) as ObjectSize,
		}
	}

	fn feed(&mut self, word: u64) {
		self.fp.feed(word);
	}

	/// Everything the caller can see of the stack, after an operation on `key`.
	fn snapshot(&mut self, key: HashedKey) -> String {
		let s = &self.stack;
		let gauges = [
			s.fast_bytes_used(),
			s.slow_bytes_used(),
			s.fast_object_count() as u64,
			s.slow_object_count() as u64,
			s.small_fast_bytes_used(),
			s.large_fast_bytes_used(),
			s.small_fast_object_count() as u64,
			s.large_fast_object_count() as u64,
			s.small_slow_bytes_used(),
			s.large_slow_bytes_used(),
			s.small_slow_object_count() as u64,
			s.large_slow_object_count() as u64,
		];
		let bytes = s.structure_bytes();
		let (dram, slow) = bytes.map_or((u64::MAX, u64::MAX), |b| (b.dram, b.slow));
		let parts = [
			s.len() as u64,
			s.contains(key) as u64,
			tier_code(s.placement_of(key)),
			s.dram_reserved_bytes(),
			dram,
			slow,
			s.needs_capacity_eviction() as u64,
			s.admission_latched() as u64,
			s.inline_demotion_accounting() as u64,
		];

		let counted = self.stack.drain_demotions();

		for word in gauges.iter().chain(parts.iter()) {
			self.fp.feed(*word);
		}

		self.fp.feed(counted);

		if counted > 0 {
			self.see(Event::DemotionsCounted);
		}

		let latched = parts[7] == 1;

		match (self.latched, latched) {
			(false, true) => self.see(Event::LatchSet),
			(true, false) => self.see(Event::LatchCleared),
			_ => {},
		}

		self.latched = latched;

		for (class, event) in [(4, Event::SmallFast), (5, Event::LargeFast), (8, Event::SmallSlow), (9, Event::LargeSlow)] {
			if gauges[class] > 0 {
				self.seen[event as usize] += 1;
			}
		}

		if parts[6] == 1 {
			self.see(Event::NeedsEviction);
		}

		match self.dump {
			Some(_) => format!(
				"len={} has={} at={} gauges={:?} reserved={} struct={}/{} need={} latched={} lfu_demotions={}",
				parts[0], parts[1], tier_char(self.stack.placement_of(key)), gauges, parts[3], dram, slow, parts[6], parts[7], counted,
			),
			None => String::new(),
		}
	}

	/// Folds the migrations the operation queued, in order, and returns them.
	fn drain(&mut self) -> Vec<(HashedKey, Tier)> {
		let migrations = self.stack.drain_tier_migrations();

		self.feed(migrations.len() as u64);

		if migrations.len() >= 2 {
			self.see(Event::TwoOrMoreMigrations);
		}

		if migrations.iter().any(|&(_, t)| t == Tier::Fast) && migrations.iter().any(|&(_, t)| t == Tier::Slow) {
			self.see(Event::BothDirections);
		}

		for &(key, tier) in &migrations {
			self.feed(key);
			self.feed(tier_code(Some(tier)));
		}

		migrations
	}

	fn line(&mut self, n: usize, op: &str, answer: &str, migrations: &[(HashedKey, Tier)], state: &str) {
		if let Some(dump) = &mut self.dump {
			let migrations = migrations
				.iter()
				.map(|&(k, t)| format!("{k}{}", tier_char(Some(t))))
				.collect::<Vec<_>>()
				.join(",");

			let _ = writeln!(dump, "{n} {op} => {answer} | migrations [{migrations}] | {state}");
		}
	}

	/// The worker's eviction pass: evict while a sub-queue is over its own
	/// capacity or the cache is over `max_size`.
	fn pass(&mut self) -> Vec<HashedKey> {
		let mut victims = Vec::new();

		while (self.stack.needs_capacity_eviction() || self.stack.fast_bytes_used() + self.stack.slow_bytes_used() > self.cfg.max)
			&& victims.len() < 4_096
		{
			match self.stack.evict_one() {
				Some(key) => victims.push(key),
				None => break,
			}
		}

		victims
	}

	fn step(&mut self, n: usize, rng: &mut Rng) {
		let op = rng.below(1_000);

		match op {
			// A set: a new key, a tracked one, or a key that was just evicted.
			0..340 | 340..370 => self.set(n, rng, op >= 340),
			370..600 => self.hit(n, rng),
			600..640 => self.remove(n, rng),
			640..700 => self.evict(n),
			700..840 => self.eviction_pass(n),
			840..846 => self.resize_max(n, rng),
			846..862 => self.resize_fast(n, rng),
			862..880 => self.measured(n, rng),
			880..896 => self.resettle(n),
			896..898 => self.clear(n),
			898..910 if self.sized => self.resize_sized(n, rng),
			_ => self.hit(n, rng),
		}
	}

	fn set(&mut self, n: usize, rng: &mut Rng, plain: bool) {
		let readmit = !self.evicted.is_empty() && rng.chance(45);
		let key = match readmit {
			true => self.evicted.swap_remove(rng.below(self.evicted.len() as u64) as usize),
			false => self.pick(rng),
		};
		let size = self.size(rng);
		let resident = match rng.below(4) {
			0 => 0,
			1 => 16 + rng.below(48) as ObjectSize,
			2 => 64,
			_ => 300,
		};
		let resident = resident.min(size);
		let placement = match rng.below(100) {
			0..6 => Placement::Structural,
			6..9 => Placement::Diverted,
			_ => Placement::Normal,
		};
		let event = match rng.below(3) {
			0 => SetEvent::Fresh,
			1 => SetEvent::Replaced { resized: true },
			_ => SetEvent::Replaced { resized: false },
		};

		let tracked = self.stack.contains(key);
		let before = self.stack.placement_of(key);

		let answer = match plain {
			true => {
				match rng.below(2) {
					0 => self.stack.insert(key, size),
					_ => self.stack.insert_resident(key, size, resident),
				}

				Placement::Normal
			},

			false => self.stack.insert_placed(key, size, resident, event, placement),
		};

		let after = self.stack.placement_of(key);
		let migrations = self.drain();

		self.feed(0x5E7);
		self.feed(key);
		self.feed(size as u64);
		self.feed(resident as u64);
		self.feed(placement_code(answer));

		let own = migrations.iter().filter(|&&(k, _)| k == key).map(|&(_, t)| t).last();
		let others_demoted = migrations.iter().any(|&(k, t)| k != key && t == Tier::Slow);

		if readmit {
			self.see(Event::EvictedKeyReadmitted);
		}

		match (tracked, before, after) {
			(false, _, _) => {
				match answer {
					Placement::Structural if placement == Placement::Structural => self.see(Event::SetNewStructuralFlagged),
					Placement::Structural => self.see(Event::SetNewStructuralOwnCheck),
					_ => {},
				}

				match after {
					Some(Tier::Fast) => self.see(Event::SetNewFast),
					Some(Tier::Slow) => self.see(Event::SetNewSlow),
					None => {},
				}

				if own == Some(Tier::Fast) {
					self.see(Event::SetNewPromoted);
				}

				if others_demoted {
					self.see(Event::SetDemoted);
				}
			},

			(true, b, a) => {
				match (b, a) {
					(Some(Tier::Fast), Some(Tier::Fast)) => self.see(Event::SetOverFastFast),
					(Some(Tier::Fast), Some(Tier::Slow)) => self.see(Event::SetOverFastSlow),
					(Some(Tier::Slow), Some(Tier::Fast)) => self.see(Event::SetOverSlowFast),
					(Some(Tier::Slow), Some(Tier::Slow)) => self.see(Event::SetOverSlowSlow),
					_ => {},
				}

				if answer == Placement::Structural {
					self.see(Event::SetOverStructural);
				}

				if own == Some(Tier::Fast) {
					self.see(Event::SetOverPromoted);
				}

				if others_demoted || own == Some(Tier::Slow) {
					self.see(Event::SetOverDemoted);
				}
			},
		}

		self.note_tracked(key);

		let state = self.snapshot(key);

		self.line(n, &format!("set {key} size={size} res={resident} {placement:?} plain={plain}"), &format!("{answer:?}"), &migrations, &state);
	}

	fn hit(&mut self, n: usize, rng: &mut Rng) {
		let key = self.pick(rng);
		let before = self.stack.placement_of(key);

		let tracked = self.stack.contains(key);

		match rng.below(4) {
			0 => self.stack.record_access(key, tracked),
			_ => self.stack.update(key),
		}

		let after = self.stack.placement_of(key);
		let migrations = self.drain();

		self.feed(0x417);
		self.feed(key);

		match (before, after) {
			(Some(Tier::Fast), Some(Tier::Fast)) => self.see(Event::HitFastFast),
			(Some(Tier::Fast), Some(Tier::Slow)) => self.see(Event::HitFastSlow),
			(Some(Tier::Slow), Some(Tier::Fast)) => self.see(Event::HitSlowFast),
			(Some(Tier::Slow), Some(Tier::Slow)) => self.see(Event::HitSlowSlow),
			_ => {},
		}

		if migrations.iter().any(|&(k, t)| k == key && t == Tier::Fast) {
			self.see(Event::HitPromoted);
		}

		if migrations.iter().any(|&(_, t)| t == Tier::Slow) {
			self.see(Event::HitDemoted);
		}

		let state = self.snapshot(key);

		self.line(n, &format!("hit {key}"), "", &migrations, &state);
	}

	fn remove(&mut self, n: usize, rng: &mut Rng) {
		// A key the stack has forgotten too: a ghost design must still drop its
		// fingerprint.
		let key = match !self.evicted.is_empty() && rng.chance(30) {
			true => self.evicted[rng.below(self.evicted.len() as u64) as usize],
			false => self.pick(rng),
		};
		let before = self.stack.placement_of(key);

		self.stack.remove(key);
		self.note_tracked(key);

		let migrations = self.drain();

		self.feed(0x2E3);
		self.feed(key);

		match before {
			Some(Tier::Fast) => self.see(Event::RemoveFast),
			Some(Tier::Slow) => self.see(Event::RemoveSlow),
			None => self.see(Event::RemoveAbsent),
		}

		let state = self.snapshot(key);

		self.line(n, &format!("remove {key}"), "", &migrations, &state);
	}

	fn evict(&mut self, n: usize) {
		let placements = self.tracked.iter().map(|&k| (k, self.stack.placement_of(k))).collect::<Vec<_>>();
		let victim = self.stack.evict_one();
		let migrations = self.drain();

		self.feed(0xE71);
		self.feed(victim.unwrap_or(u64::MAX));

		let key = victim.unwrap_or(0);

		match victim {
			Some(key) => {
				match placements.iter().find(|&&(k, _)| k == key).and_then(|&(_, p)| p) {
					Some(Tier::Fast) => self.see(Event::EvictFast),
					_ => self.see(Event::EvictSlow),
				}

				self.evicted.push(key);

				if self.evicted.len() > 48 {
					self.evicted.remove(0);
				}

				self.note_tracked(key);
			},

			None => self.see(Event::EvictNone),
		}

		for &(k, tier) in &migrations {
			let was = placements.iter().find(|&&(other, _)| other == k).and_then(|&(_, p)| p);

			match tier {
				Tier::Fast => {
					self.see(Event::EvictPromotedAnother);

					if was == Some(Tier::Fast) {
						self.see(Event::EvictPromotedAFastKey);
					}
				},

				Tier::Slow => self.see(Event::EvictDemotedAnother),
			}
		}

		let state = self.snapshot(key);

		self.line(n, "evict", &format!("{victim:?}"), &migrations, &state);
	}

	fn eviction_pass(&mut self, n: usize) {
		let victims = self.pass();
		let migrations = self.drain();

		self.feed(0xE77);

		for &victim in &victims {
			self.feed(victim);
			self.evicted.push(victim);
			self.note_tracked(victim);
		}

		if self.evicted.len() > 48 {
			let excess = self.evicted.len() - 48;

			self.evicted.drain(..excess);
		}

		let state = self.snapshot(victims.first().copied().unwrap_or(0));

		self.line(n, "pass", &format!("{victims:?}"), &migrations, &state);
	}

	fn resize_max(&mut self, n: usize, rng: &mut Rng) {
		let max = [self.cfg.max / 2, self.cfg.max, self.cfg.max * 2][rng.below(3) as usize];

		self.stack.resize(max);
		self.see(Event::ResizeMax);

		let migrations = self.drain();

		self.feed(0x5E1);
		self.feed(max);

		let state = self.snapshot(0);

		self.line(n, &format!("resize {max}"), "", &migrations, &state);
	}

	fn resize_fast(&mut self, n: usize, rng: &mut Rng) {
		let fast = [1, self.cfg.fast / 4, self.cfg.fast / 2, self.cfg.fast, self.cfg.fast * 2][rng.below(5) as usize];
		let before = self.stack.fast_bytes_used();

		self.stack.resize_fast_tier(fast);

		let migrations = self.drain();

		self.feed(0x5F1);
		self.feed(fast);

		if migrations.iter().any(|&(_, t)| t == Tier::Slow) {
			self.see(Event::ResizeFastDemoted);
		}

		if fast >= self.cfg.fast && before > 0 {
			self.see(Event::ResizeFastGrown);
		}

		let state = self.snapshot(0);

		self.line(n, &format!("resize_fast_tier {fast}"), "", &migrations, &state);
	}

	fn measured(&mut self, n: usize, rng: &mut Rng) {
		let measured = match rng.below(5) {
			0 | 1 => None,
			_ => Some(rng.below(self.cfg.fast + self.cfg.fast / 4 + 1)),
		};

		self.stack.set_dram_metadata(measured);

		let migrations = self.drain();

		self.feed(0x3E5);
		self.feed(measured.map_or(u64::MAX, |m| m));

		let state = self.snapshot(0);

		self.line(n, &format!("set_dram_metadata {measured:?}"), "", &migrations, &state);

		// `resettle` is the worker's step after the reservation moves.
		self.resettle(n);
	}

	fn resettle(&mut self, n: usize) {
		self.stack.resettle();

		let migrations = self.drain();

		self.feed(0x2E5);

		if migrations.iter().any(|&(_, t)| t == Tier::Slow) {
			self.see(Event::ResettleDemoted);
			self.see(Event::MeasuredDemoted);
		}

		let state = self.snapshot(0);

		self.line(n, "resettle", "", &migrations, &state);
	}

	fn clear(&mut self, n: usize) {
		self.stack.clear();
		self.tracked.clear();
		self.see(Event::Cleared);

		let migrations = self.drain();

		self.feed(0xC1E);

		let state = self.snapshot(0);

		self.line(n, "clear", "", &migrations, &state);
	}

	fn resize_sized(&mut self, n: usize, rng: &mut Rng) {
		let migrations = match rng.below(2) {
			0 => {
				let large = [self.cfg.fast / 4, self.cfg.fast / 2, self.cfg.fast][rng.below(3) as usize];

				self.stack.resize_large_fast_tier(large);
				self.feed(0x51A);
				self.feed(large);

				self.drain()
			},

			_ => {
				let threshold = [400, 1_000, 2_500][rng.below(3) as usize];

				self.stack.resize_size_threshold(threshold);
				self.feed(0x51B);
				self.feed(threshold);

				self.drain()
			},
		};

		let state = self.snapshot(0);

		self.line(n, "resize_sized", "", &migrations, &state);
	}

	/// Every key of the universe: membership and placement.
	fn audit(&mut self) {
		self.feed(0xA0D);

		for index in 0..self.cfg.keys {
			let key = self.key(index);

			self.feed(self.stack.contains(key) as u64);
			self.fp.feed(tier_code(self.stack.placement_of(key)));
		}
	}
}

/// One run: `design` at `cfg`, `OPS` operations.
fn run(design: &Design, cfg: &Cfg, dump: bool) -> Outcome {
	let mut run = Run {
		stack: (design.build)(cfg),
		cfg,
		sized: design.name == "lru-sized",
		fp: Fnv::new(),
		seen: [0; EVENTS],
		dump: dump.then(String::new),
		tracked: Vec::new(),
		evicted: Vec::new(),
		latched: false,
	};
	let mut rng = Rng(cfg.seed | 1);

	for n in 0..OPS {
		run.step(n, &mut rng);

		if n % AUDIT_EVERY == 0 {
			run.audit();
		}
	}

	// Everything the stack holds, in eviction order.
	run.feed(0xD8A);

	while let Some(victim) = run.stack.evict_one() {
		run.feed(victim);

		let migrations = run.drain();

		run.feed(migrations.len() as u64);
	}

	Outcome { fingerprint: run.fp.0, seen: run.seen, dump: run.dump }
}

// ---------------------------------------------------------------------------
// recording and checking
// ---------------------------------------------------------------------------

fn label(ratio: f64) -> String {
	format!("{ratio:.2}")
}

fn selected(design: &str, config: &str) -> bool {
	match std::env::var("PAPER_TIER_GOLDEN_ONLY") {
		Ok(only) => only.split(',').any(|want| format!("{design}/{config}").contains(want)),
		Err(_) => true,
	}
}

/// Runs the whole grid at the process's drain target. Returns
/// `(design, configuration, fingerprint)` in recording order, and each
/// design's coverage.
fn record() -> (Vec<(String, String, u64)>, BTreeMap<&'static str, [u64; EVENTS]>) {
	let dump_dir = std::env::var_os("PAPER_TIER_GOLDEN_DUMP").map(PathBuf::from);
	let mut out = Vec::new();
	let mut coverage = BTreeMap::new();

	for design in &DESIGNS {
		let mut seen = [0u64; EVENTS];

		for (config, cfg) in grid(design) {
			if !selected(design.name, &config) {
				continue;
			}

			let outcome = run(design, &cfg, dump_dir.is_some());

			if let (Some(dir), Some(lines)) = (&dump_dir, &outcome.dump) {
				let dir = dir.join(design.name);

				std::fs::create_dir_all(&dir).expect("the dump directory");
				std::fs::write(dir.join(format!("{config}.txt")), lines).expect("the dump");
			}

			for (total, n) in seen.iter_mut().zip(outcome.seen) {
				*total += n;
			}

			out.push((design.name.to_string(), config, outcome.fingerprint));
		}

		coverage.insert(design.name, seen);
	}

	(out, coverage)
}

/// The recorded lines for one drain target.
fn recorded(ratio: f64) -> BTreeMap<(String, String), u64> {
	let want = label(ratio);

	RECORDED
		.lines()
		.filter(|line| !line.starts_with('#') && !line.trim().is_empty())
		.filter_map(|line| {
			let mut words = line.split_whitespace();
			let (drain, design, config, fingerprint) = (words.next()?, words.next()?, words.next()?, words.next()?);

			(drain == want).then(|| ((design.to_string(), config.to_string()), u64::from_str_radix(fingerprint, 16).expect("a hex fingerprint")))
		})
		.collect()
}

fn check_at(ratio: f64) {
	assert_eq!(
		drain_target::ratio(),
		ratio,
		"these fingerprints are for FAST_TIER_DRAIN_TARGET {ratio}; unset it (or run the `of_one` test, which sets it)",
	);

	let (results, coverage) = record();

	if let Some(dir) = recording() {
		// Recording: twice, to show it is deterministic.
		let (again, _) = record();

		assert!(results == again, "the recording is not deterministic: two runs of the same grid differ");

		let mut text = String::new();

		for (design, config, fingerprint) in &results {
			let _ = writeln!(text, "{} {design} {config} {fingerprint:016x}", label(ratio));
		}

		// What each design's runs reached (once, at the default target).
		if ratio == drain_target::DEFAULT_RATIO {
			for (design, seen) in &coverage {
				for event in (0..EVENTS).filter(|&i| seen[i] > 0) {
					let _ = writeln!(text, "seen {design} {}", EVENT_NAMES[event]);
				}
			}
		}

		let path = dir.join(format!("drain-{}.txt", label(ratio)));

		std::fs::write(&path, text).unwrap_or_else(|e| panic!("writing {path:?}: {e}"));
		report_coverage(&coverage);

		return;
	}

	if std::env::var_os("PAPER_TIER_GOLDEN_DUMP").is_some() {
		return;
	}

	let want = recorded(ratio);
	let mut moved = Vec::new();

	for (design, config, fingerprint) in &results {
		match want.get(&(design.clone(), config.clone())) {
			Some(expected) if expected == fingerprint => {},
			Some(expected) => moved.push(format!("{design}/{config}: recorded {expected:016x}, now {fingerprint:016x}")),
			None => moved.push(format!("{design}/{config}: not recorded")),
		}
	}

	if std::env::var_os("PAPER_TIER_GOLDEN_ONLY").is_none() {
		assert_eq!(want.len(), results.len(), "the recorded file and the grid list a different number of runs");
	}

	// Every path a design's runs reached when this was recorded is still
	// reached: the stream cannot be edited into a test of less.
	if ratio == drain_target::DEFAULT_RATIO && std::env::var_os("PAPER_TIER_GOLDEN_ONLY").is_none() {
		let mut lost = Vec::new();

		for line in RECORDED.lines().filter(|line| line.starts_with("seen ")) {
			let mut words = line.split_whitespace().skip(1);
			let (design, event) = (words.next().expect("a design"), words.next().expect("an event"));
			let at = EVENT_NAMES.iter().position(|&name| name == event).unwrap_or_else(|| panic!("no event `{event}`"));

			if coverage[design][at] == 0 {
				lost.push(format!("{design} no longer reaches {event}"));
			}
		}

		assert!(lost.is_empty(), "the stream stopped reaching paths it reached when this was recorded:\n{}", lost.join("\n"));
	}

	assert!(
		moved.is_empty(),
		"{} of {} runs no longer reproduce what the stacks did when this was recorded (PAPER_TIER_GOLDEN_ONLY and \
		 PAPER_TIER_GOLDEN_DUMP bisect; the module doc says how):\n{}",
		moved.len(),
		results.len(),
		moved.join("\n"),
	);
}

fn report_coverage(coverage: &BTreeMap<&'static str, [u64; EVENTS]>) {
	for (design, seen) in coverage {
		let line = (0..EVENTS)
			.filter(|&i| seen[i] == 0)
			.map(|i| EVENT_NAMES[i].to_string())
			.collect::<Vec<_>>()
			.join(" ");

		println!("coverage {design}: never saw [{line}]");
	}
}


/// Whether this run records (`PAPER_TIER_GOLDEN_OUT`) instead of checking.
pub(crate) fn recording() -> Option<PathBuf> {
	std::env::var_os("PAPER_TIER_GOLDEN_OUT").map(PathBuf::from)
}

/// T14's scripts for the designs it does not cover (`s4_tests::t14`, which the
/// merged builds do not have): the hash of one script's record against the one
/// recorded for it, or, recording, written to `<dir>/t14-<name>.txt`.
#[cfg(not(feature = "merged_object_store"))]
pub(crate) fn check_script(name: &str, hash: u64) {
	if let Some(dir) = recording() {
		let path = dir.join(format!("t14-{name}.txt"));

		std::fs::write(&path, format!("t14 {name} {hash:016x}\n")).unwrap_or_else(|e| panic!("writing {path:?}: {e}"));

		return;
	}

	let recorded = RECORDED
		.lines()
		.filter_map(|line| {
			let mut words = line.split_whitespace();

			(words.next()? == "t14" && words.next()? == name).then(|| words.next()).flatten()
		})
		.next()
		.unwrap_or_else(|| panic!("T14 script `{name}` has no recorded hash"));

	assert_eq!(
		format!("{hash:016x}"),
		recorded,
		"T14's script `{name}` no longer reproduces what the stack did when this was recorded",
	);
}

#[cfg(test)]
mod tests {
	use super::*;
	use crate::worker::policy::test_support::alone_in_with_env;

	#[test]
	fn every_design_reproduces_its_recorded_fingerprints() {
		check_at(0.95);
	}

	/// `FAST_TIER_DRAIN_TARGET` is read once per process, so the second target
	/// runs in a child that has it from the start.
	#[test]
	fn every_design_reproduces_its_recorded_fingerprints_at_a_drain_target_of_one() {
		alone_in_with_env(
			module_path!(),
			"every_design_reproduces_its_recorded_fingerprints_at_a_drain_target_of_one",
			&[("FAST_TIER_DRAIN_TARGET", "1.0")],
			|| check_at(1.0),
		);
	}

	/// `hybrid_policy::admission_tier` repeats each design's rule for where the
	/// client builds a set's value, for a key the stack tracks and for one it
	/// does not -- and the worker's reconcile (`corrective`) takes the stack's
	/// last queued entry for the key, else the tier the client built in, as
	/// where the bytes are going to be, and queues a corrective migration if
	/// that is not `placement_of`. So the rule and what a design's `Set` does to
	/// its placement must say the same thing, or every set of that key pays a
	/// copy: nothing else ties the two together.
	///
	/// Each set here asks the REAL `admission_tier`, with the key's object in a
	/// map in the tier the stack places it (or no object, for a key it does
	/// not track, and the LFU latch mirrored as the worker publishes it), then
	/// applies the set and requires `queued.unwrap_or(built) == placement_of`,
	/// as `corrective` does. Sets the stack makes structural are left out: the
	/// client decides those by `gate::decide`, not by this function.
	#[cfg(not(feature = "merged_object_store"))]
	mod admission {
		use super::*;
		use crate::{hybrid_policy::admission_tier, object::Object, status::AtomicStatus, TieredBuffer};

		type Objects = crate::ObjectMapRef<u32, TieredBuffer>;

		/// What one design's sets showed: how many of each kind were checked and
		/// which, if any, needed a corrective.
		#[derive(Default)]
		struct Tally {
			checked_new: u64,
			checked_over: u64,
			needing: Vec<String>,
		}

		fn tally(design: &Design, cfg: &Cfg) -> Tally {
			let status = AtomicStatus::new(cfg.max, &[design.policy], design.policy).expect("a status");
			let empty: Objects = crate::new_hybrid_object_map();
			let held: Objects = crate::new_hybrid_object_map();
			let mut stack = (design.build)(cfg);
			let mut rng = Rng(cfg.seed | 1);
			let mut tracked: Vec<HashedKey> = Vec::new();
			let mut out = Tally::default();

			for n in 0..3_000 {
				match rng.below(100) {
					0..55 => {
						let key = match !tracked.is_empty() && rng.chance(50) {
							true => tracked[rng.below(tracked.len() as u64) as usize],
							false => 1 + rng.below(cfg.keys),
						};
						let size = 200 + rng.below(1_800) as ObjectSize;
						let resident = rng.below(3) as ObjectSize * 24;
						let physical = stack.placement_of(key);

						let objects = match physical {
							Some(tier) => {
								held.insert(key, Object::new_in(key as u32, &[0xA5; 64], tier, None));
								&held
							},

							None => &empty,
						};

						status.set_hybrid_admission_latched(stack.admission_latched());

						let built = admission_tier::<u32>(design.policy, key, &status, objects);
						let answer = stack.insert_placed(key, size, resident, SetEvent::Fresh, Placement::Normal);
						let queued = stack
							.drain_tier_migrations()
							.iter()
							.rev()
							.find(|&&(k, _)| k == key)
							.map(|&(_, tier)| tier);
						let placement = stack.placement_of(key);

						if answer != Placement::Structural {
							match physical {
								None => out.checked_new += 1,
								Some(_) => out.checked_over += 1,
							}

							if Some(queued.unwrap_or(built)) != placement {
								out.needing.push(format!(
									"op {n}: key {key} was {:?}, built {built:?}, the stack queued {queued:?} and places it {placement:?}",
									physical,
								));
							}
						}

						if !tracked.contains(&key) && stack.contains(key) {
							tracked.push(key);
						}
					},

					55..80 => {
						if !tracked.is_empty() {
							stack.update(tracked[rng.below(tracked.len() as u64) as usize]);
							drop(stack.drain_tier_migrations());
						}
					},

					80..90 => {
						while (stack.needs_capacity_eviction() || stack.fast_bytes_used() + stack.slow_bytes_used() > cfg.max)
							&& stack.evict_one().is_some()
						{}

						drop(stack.drain_tier_migrations());
						tracked.retain(|&k| stack.contains(k));
					},

					90..95 => {
						if !tracked.is_empty() {
							stack.remove(tracked.swap_remove(rng.below(tracked.len() as u64) as usize));
						}
					},

					_ => {
						stack.resize_fast_tier([cfg.fast / 2, cfg.fast, cfg.fast * 2][rng.below(3) as usize]);
						drop(stack.drain_tier_migrations());
					},
				}
			}

			out
		}

		#[test]
		fn the_clients_build_tier_is_what_every_design_does_to_a_set() {
			let mut failures = Vec::new();

			for design in &DESIGNS {
				let mut checked = (0u64, 0u64);

				for (config, cfg) in grid(design).into_iter().filter(|(c, _)| c.starts_with("s1-f1-o0") || c.starts_with("s1-f0-o64")) {
					let out = tally(design, &cfg);

					checked.0 += out.checked_new;
					checked.1 += out.checked_over;

					if let Some(first) = out.needing.first() {
						failures.push(format!("{}/{config}: {} of {} sets needed a corrective, the first: {first}", design.name, out.needing.len(), out.checked_new + out.checked_over));
					}
				}

				assert!(
					checked.0 > 100 && checked.1 > 100,
					"{}: only {} new and {} tracked keys were checked",
					design.name,
					checked.0,
					checked.1,
				);
			}

			assert!(failures.is_empty(), "admission_tier and a design's own placement of a set disagree:\n{}", failures.join("\n"));
		}
	}
}
