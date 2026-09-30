/*
 * Copyright (c) Kia Shakiba
 *
 * This source code is licensed under the GNU AGPLv3 license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! Golden sequences for the flat `*CompactStack` fidelity tests.
//!
//! Each Compact stack re-lays-out one of the original `HashList` stacks
//! (`LruStack`, `FifoStack`, ...), and its tests used to replay a deterministic
//! op sequence through both and require every observable to agree at every
//! step. The originals are gone (R2), so what those tests compared was
//! recorded from them first and is asserted here as fixed data: the scenarios
//! below are the SAME op sequences, run on one stack, and reduced to a
//! [`Golden`] -- a fingerprint of every observation made along the way (the
//! length and membership after each op, each `evict_one` an op interleaves),
//! then the number of keys the final drain evicted, its first few, and a
//! fingerprint of the whole eviction order. A stack that evicts one key
//! differently anywhere changes the order's fingerprint; one that reports a
//! different `len` or `contains` at any step changes the steps'.
//!
//! The constants live with the stack they pin (each `fidelity_tests`), and
//! were recorded from the original stack and checked equal to the Compact
//! stack's before the original was deleted.

use super::PolicyStack;
use crate::{CacheSize, HashedKey, ObjectSize};

/// What a scenario observed: see the module doc.
#[derive(Debug, PartialEq, Eq)]
pub(super) struct Golden {
	pub steps: u64,
	pub drained: usize,
	pub head: [HashedKey; 4],
	pub order: u64,
}

/// FNV-1a over 64-bit words.
struct Fingerprint(u64);

impl Fingerprint {
	fn new() -> Self {
		Fingerprint(0xCBF2_9CE4_8422_2325)
	}

	fn feed(&mut self, word: u64) {
		for byte in word.to_le_bytes() {
			self.0 = (self.0 ^ byte as u64).wrapping_mul(0x0000_0100_0000_01B3);
		}
	}
}

/// Runs a scenario against one stack, folding what it observes into a [`Golden`].
struct Run<'a> {
	stack: &'a mut dyn PolicyStack,
	steps: Fingerprint,
}

impl<'a> Run<'a> {
	fn new(stack: &'a mut dyn PolicyStack) -> Self {
		Run { stack, steps: Fingerprint::new() }
	}

	/// `len` and `contains(key)` after an op on `key`.
	fn observe(&mut self, key: HashedKey) {
		self.steps.feed(self.stack.len() as u64);
		self.steps.feed(self.stack.contains(key) as u64);
	}

	/// An `evict_one` the scenario interleaves: its answer is an observation.
	fn evict(&mut self) {
		self.steps.feed(self.stack.evict_one().map_or(u64::MAX, |key| key));
	}

	/// Evicts everything and reduces the whole run.
	fn finish(mut self) -> Golden {
		self.steps.feed(self.stack.len() as u64);

		let mut order = Fingerprint::new();
		let mut head = [0; 4];
		let mut drained = 0;

		while let Some(key) = self.stack.evict_one() {
			if drained < head.len() {
				head[drained] = key;
			}

			order.feed(key);
			drained += 1;
		}

		Golden { steps: self.steps.0, drained, head, order: order.0 }
	}
}

/// The xorshift stream every scenario draws its keys from.
struct Xorshift(u64);

impl Xorshift {
	fn next(&mut self) -> u64 {
		self.0 ^= self.0 << 13;
		self.0 ^= self.0 >> 7;
		self.0 ^= self.0 << 17;
		self.0
	}
}

/// A key in `1..=span` from a draw `x`, skewed towards the low keys.
fn skewed_key(x: u64, span: f64) -> HashedKey {
	let u = (x >> 11) as f64 / (1u64 << 53) as f64;
	((u * u * span) as u64) + 1
}

/// The fingerprint of an eviction order, as [`Golden::order`].
#[cfg(feature = "s3_fifo_faithful_compact_hybrid_cache")]
pub(super) fn order_fingerprint(order: &[HashedKey]) -> u64 {
	let mut fingerprint = Fingerprint::new();

	for &key in order {
		fingerprint.feed(key);
	}

	fingerprint.0
}

const SEED: u64 = 0x243F_6A88_85A3_08D3;

/// The single-queue policies (LRU, FIFO, CLOCK, MRU, SIEVE): 40,000 ops on a
/// skewed key set, every third a hit and the rest inserts, so keys are
/// repeatedly moved, marked or passed over.
pub(super) fn single_queue_skewed(stack: &mut dyn PolicyStack) -> Golden {
	let mut run = Run::new(stack);
	let mut rng = Xorshift(SEED);

	for i in 0..40_000u64 {
		let x = rng.next();
		let key = skewed_key(x, 500.0);

		if i % 3 == 0 {
			run.stack.update(key);
		} else {
			run.stack.insert(key, 1_024);
		}

		run.observe(key);
	}

	run.finish()
}

/// LFU: 40,000 ops on 400 skewed keys, every fourth a hit on a resident key
/// (an insert otherwise), so counts build up and the buckets churn.
pub(super) fn lfu_skewed(stack: &mut dyn PolicyStack) -> Golden {
	let mut run = Run::new(stack);
	let mut rng = Xorshift(SEED);

	for i in 0..40_000u64 {
		let x = rng.next();
		let key = skewed_key(x, 400.0);

		if i % 4 == 3 && run.stack.contains(key) {
			run.stack.update(key);
		} else {
			run.stack.insert(key, 1_024);
		}

		run.observe(key);
	}

	run.finish()
}

/// The queue policies with byte budgets (2Q, S3-FIFO; and the faithful S3-FIFO
/// hybrid's oracle): the same 400 skewed keys with sizes varied so the budgets
/// bind, every fourth op a hit on a resident key.
pub(super) fn budgeted_skewed(stack: &mut dyn PolicyStack) -> Golden {
	let mut run = Run::new(stack);
	let mut rng = Xorshift(SEED);

	for i in 0..40_000u64 {
		let x = rng.next();
		let key = skewed_key(x, 400.0);
		let size = (1_024 + (x % 3_072)) as ObjectSize;

		if i % 4 == 3 && run.stack.contains(key) {
			run.stack.update(key);
		} else {
			run.stack.insert(key, size);
		}

		run.observe(key);
	}

	run.finish()
}

/// Inserts `0..keys` (a hit on every third), removes every `step`-th, and
/// drains: removal must not disturb the order of what remains.
pub(super) fn removal(stack: &mut dyn PolicyStack, keys: u64, size: ObjectSize, hits: bool, step: usize) -> Golden {
	let mut run = Run::new(stack);

	for key in 0..keys {
		run.stack.insert(key, size);

		if hits && key % 3 == 0 {
			run.stack.update(key);
		}
	}

	for key in (0..keys).step_by(step) {
		run.stack.remove(key);
		run.observe(key);
	}

	run.finish()
}

/// 2Q and S3-FIFO: 1,500 inserts, a `resize` to four times the budget (which
/// re-derives both queue budgets and moves the spill point), 1,000 more.
pub(super) fn resize(stack: &mut dyn PolicyStack, max: CacheSize) -> Golden {
	let mut run = Run::new(stack);

	for key in 0..1_500u64 {
		run.stack.insert(key, 2_048);
		run.observe(key);
	}

	run.stack.resize(max * 4);

	for key in 1_500..2_500u64 {
		run.stack.insert(key, 2_048);
		run.observe(key);
	}

	run.finish()
}

/// S3-FIFO: 40,000 inserts over 600 keys with an eviction after every third,
/// so the ghost queue fills, keys are re-admitted through it into `main`, and
/// the ghost trim in `evict_main` runs -- the paths a drain-at-the-end test
/// never reaches.
pub(super) fn ghost_readmission(stack: &mut dyn PolicyStack) -> Golden {
	let mut run = Run::new(stack);
	let mut rng = Xorshift(0x1357_9BDF_2468_ACE0);

	for i in 0..40_000u64 {
		let x = rng.next();
		let key = (x % 600) + 1;
		let size = (512 + (x % 2_048)) as ObjectSize;

		run.stack.insert(key, size);

		if i % 3 == 0 {
			run.evict();
		}

		run.observe(key);
	}

	run.finish()
}

/// S3-FIFO: 20,000 accesses over 400 keys, each recorded (`record_access`) and
/// inserted on a miss, with an eviction after every third -- the ghost queue's
/// only view from outside.
pub(super) fn ghost_hits(stack: &mut dyn PolicyStack) -> Golden {
	let mut run = Run::new(stack);
	let mut rng = Xorshift(0x0BAD_C0DE_DEAD_BEEF);

	for i in 0..20_000u64 {
		let x = rng.next();
		let key = (x % 400) + 1;
		let size = (512 + (x % 1_024)) as ObjectSize;

		let hit = run.stack.contains(key);
		run.steps.feed(hit as u64);
		run.stack.record_access(key, hit);

		if !hit {
			run.stack.insert(key, size);
		}

		if i % 3 == 0 {
			run.evict();
		}

		run.observe(key);
	}

	run.finish()
}

// ---------------------------------------------------------------------------
// Recorded from the original stacks (see the module doc), one constant per scenario and stack.

// single-queue policies
pub(super) const LRU_SKEWED: Golden = Golden {
	steps: 5201350894172037447,
	drained: 500,
	head: [450, 388, 401, 490],
	order: 6255129187431067550,
};

pub(super) const FIFO_SKEWED: Golden = Golden {
	steps: 5201350894172037447,
	drained: 500,
	head: [139, 292, 6, 29],
	order: 2512103167002506034,
};

pub(super) const CLOCK_SKEWED: Golden = Golden {
	steps: 5201350894172037447,
	drained: 500,
	head: [139, 292, 6, 29],
	order: 2512103167002506034,
};

pub(super) const MRU_SKEWED: Golden = Golden {
	steps: 2271070854664265730,
	drained: 500,
	head: [394, 292, 420, 274],
	order: 7852835337410773082,
};

pub(super) const SIEVE_SKEWED: Golden = Golden {
	steps: 5201350894172037447,
	drained: 500,
	head: [139, 292, 6, 29],
	order: 2512103167002506034,
};

pub(super) const LFU_SKEWED: Golden = Golden {
	steps: 12314135448676023636,
	drained: 400,
	head: [353, 321, 400, 296],
	order: 16907019944003786334,
};

pub(super) const LRU_REMOVAL: Golden = Golden {
	steps: 13521343689783939712,
	drained: 666,
	head: [1, 2, 4, 5],
	order: 3565997819715641559,
};

pub(super) const FIFO_REMOVAL: Golden = Golden {
	steps: 10578632713951802915,
	drained: 1600,
	head: [1, 2, 3, 4],
	order: 15242242348355386474,
};

pub(super) const CLOCK_REMOVAL: Golden = Golden {
	steps: 10578632713951802915,
	drained: 1600,
	head: [1, 2, 4, 7],
	order: 5934249419155351162,
};

pub(super) const MRU_REMOVAL: Golden = Golden {
	steps: 10578632713951802915,
	drained: 1600,
	head: [1998, 1997, 1996, 1994],
	order: 15451636121754225690,
};

pub(super) const SIEVE_REMOVAL: Golden = Golden {
	steps: 10578632713951802915,
	drained: 1600,
	head: [1996, 1997, 1999, 1],
	order: 1798035554494654902,
};

pub(super) const LFU_REMOVAL: Golden = Golden {
	steps: 10578632713951802915,
	drained: 1600,
	head: [1, 2, 4, 7],
	order: 5934249419155351162,
};


// 2Q
pub(super) const TWO_Q_SKEWED: Golden = Golden {
	steps: 12314135448676023636,
	drained: 400,
	head: [363, 88, 217, 281],
	order: 9206843083803802122,
};

pub(super) const TWO_Q_REMOVAL: Golden = Golden {
	steps: 6398147522828692969,
	drained: 2571,
	head: [1, 2, 3, 4],
	order: 13436726472474433855,
};

pub(super) const TWO_Q_RESIZE: Golden = Golden {
	steps: 14605397659231675693,
	drained: 2500,
	head: [0, 1, 2, 3],
	order: 4818587527063352425,
};


// S3-FIFO
pub(super) const S3_SKEWED: Golden = Golden {
	steps: 12314135448676023636,
	drained: 400,
	head: [109, 111, 234, 260],
	order: 16795970193589313914,
};

pub(super) const S3_GHOST_READMISSION: Golden = Golden {
	steps: 14294581546570166168,
	drained: 393,
	head: [466, 158, 254, 144],
	order: 11363478864956951160,
};

pub(super) const S3_REMOVAL: Golden = Golden {
	steps: 6398147522828692969,
	drained: 2571,
	head: [1, 2, 3, 4],
	order: 13436726472474433855,
};

pub(super) const S3_RESIZE: Golden = Golden {
	steps: 14605397659231675693,
	drained: 2500,
	head: [0, 1, 2, 3],
	order: 4818587527063352425,
};

pub(super) const S3_GHOST_HITS: Golden = Golden {
	steps: 8685336452599374901,
	drained: 270,
	head: [371, 295, 336, 374],
	order: 14877799966561747648,
};
