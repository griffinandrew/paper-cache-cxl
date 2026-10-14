/*
 * Copyright (c) Kia Shakiba
 *
 * This source code is licensed under the GNU AGPLv3 license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! The read path's prefetch: a `get` that hits issues `prefetcht0` for the
//! first cache lines of what it is about to read as soon as the map hit has
//! produced the value's DRAM header, so the fetch overlaps the refcount bump,
//! the shard guard's release and the output allocation instead of following
//! them.
//!
//! What it fetches depends on the layout (`TieredValue::prefetch`):
//!
//!   * `thin_header`: the ITEM. On a slow object the key check, `is_expired`
//!     and `len` all read the item's first line, on the far node, before the
//!     copy can start; without the prefetch that line is a blocking remote
//!     load under the shard guard, and the copy's lines follow it one miss at
//!     a time. The number of lines the item spans is carried in the DRAM word
//!     (bits 58-63, set when the item is built), so `auto` knows it without
//!     reading the item.
//!   * the default layout: the BYTES. Their length is in the DRAM header, so
//!     the count is exact there too. The header's own fields are already DRAM,
//!     and measured, the prefetch changes nothing in this layout (see below).
//!
//! Two knobs, read once from the environment and packed into one word, so a
//! `get` pays one relaxed load for both:
//!
//!   * `THIN_PREFETCH_LINES` -- `auto` (the lines the item or the bytes span),
//!     `auto:CAP` (the same, at most CAP lines), a fixed count `N` (the first N
//!     lines, never more than the bytes span under the default layout; under
//!     `thin_header` the count cannot be capped by a length that is in the
//!     item being fetched), or `0` (off). CAP and N are clamped to 1..=64; the
//!     word's line count saturates at 63. DEFAULT: `auto:64` under
//!     `thin_header`, `0` (off) otherwise.
//!   * `THIN_PREFETCH_FAST` -- `1` prefetches fast-tier objects too, `0` only
//!     slow-tier ones. DEFAULT: `1` under `thin_header`, `0` otherwise.
//!
//! Measured (`value_bench`'s `get_bench`, one thread pinned to node 0, uniform
//! keys over a working set far above the LLC, p50 ns, slow / fast tier):
//!
//! ```text
//!   thin_header   value    off           auto:16       auto:64
//!                   128    1540 / 1026   1172 /  929   1172 /  924
//!                   256    1619 / 1064   1228 /  960   1227 /  962
//!                  1024    1830 / 1053   1448 /  939   1420 /  933
//!                  4096    2366 / 1202   2141 / 1137   1962 / 1091
//!                 16384    4906 / 2174   4715 / 2011   4553 / 2023
//!   default         128    1163 /  871   1163 /  883   1159 /  860
//!                  1024    1387 /  953   1338 /  942   1334 /  939
//! ```
//!
//! Under `thin_header` the slow hit's extra cost over the default layout fell
//! from +350-450 ns to +10-40 ns up to 1 KiB; `auto:64` was never worse than
//! `auto:16` and better from 4 KiB, hence the default cap. A fixed count was
//! worse than `auto` on both sides of its own size (2-4 lines did nothing
//! for 256 B, 8 lines cost 50 ns on 64 B values). Two related experiments
//! were not kept: prefetching the replaced item's first line on a set (on
//! the Meta kvcache_202401 trace SET mean 1560 -> 2223 ns and more byte-gate
//! waits) and moving the key and expiry checks after the copy, off the guard
//! (no clear gain there).

use std::sync::atomic::{AtomicU32, Ordering};

const INIT: u32 = 1 << 31;
const FAST: u32 = 1 << 8;
const AUTO: u32 = 1 << 11;
const LINES: u32 = 0xff;

/// The most lines a prefetch issues.
const MAX_LINES: u32 = 64;

static WORD: AtomicU32 = AtomicU32::new(0);

/// The read path's knobs as one word.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct ReadKnobs(u32);

impl ReadKnobs {
	/// Lines to prefetch (`auto`: the cap), 0 for none.
	#[inline]
	pub(crate) fn lines(self) -> u32 {
		self.0 & LINES
	}

	/// `auto[:CAP]`: prefetch the lines the item or the bytes span, known
	/// from DRAM, at most [`ReadKnobs::lines`].
	#[inline]
	pub(crate) fn auto(self) -> bool {
		self.0 & AUTO != 0
	}

	/// Whether fast-tier objects are prefetched too.
	#[inline]
	pub(crate) fn fast_too(self) -> bool {
		self.0 & FAST != 0
	}

	/// The lines to fetch for a value that spans `spans` lines, under these
	/// knobs: all of them up to the cap under `auto`, else the fixed count.
	#[inline]
	pub(crate) fn lines_for(self, spans: u32) -> u32 {
		match self.auto() {
			true => spans.min(self.lines()),
			false => self.lines(),
		}
	}
}

/// The knobs, read from the environment on first use.
#[inline]
pub(crate) fn knobs() -> ReadKnobs {
	let word = WORD.load(Ordering::Relaxed);

	if word & INIT != 0 {
		return ReadKnobs(word);
	}

	init()
}

/// The defaults: on, `auto:64`, both tiers, under `thin_header`; off otherwise.
const DEFAULT_LINES: &str = if cfg!(feature = "thin_header") { "auto:64" } else { "0" };
const DEFAULT_FAST: bool = cfg!(feature = "thin_header");

#[cold]
fn init() -> ReadKnobs {
	let lines = std::env::var("THIN_PREFETCH_LINES").unwrap_or_else(|_| DEFAULT_LINES.to_string());
	let fast = std::env::var("THIN_PREFETCH_FAST")
		.ok()
		.and_then(|v| v.trim().parse::<u32>().ok())
		.map_or(DEFAULT_FAST, |v| v != 0);

	let word = parse_lines(&lines) | if fast { FAST } else { 0 } | INIT;

	// A racing first use computes the same word; a test override that got
	// there first wins.
	let _ = WORD.compare_exchange(0, word, Ordering::Relaxed, Ordering::Relaxed);

	ReadKnobs(WORD.load(Ordering::Relaxed))
}

/// `THIN_PREFETCH_LINES` as the word's lines and `AUTO` bits. Anything that
/// does not parse is off.
fn parse_lines(value: &str) -> u32 {
	let value = value.trim();

	match value.strip_prefix("auto") {
		Some(rest) => {
			let cap = match rest.strip_prefix(':') {
				Some(cap) => cap.trim().parse::<u32>().unwrap_or(MAX_LINES),
				None => MAX_LINES,
			};

			AUTO | cap.clamp(1, MAX_LINES)
		},

		None => value.parse::<u32>().unwrap_or(0).min(MAX_LINES),
	}
}

/// Overrides the knobs for this process: `auto` capped at `lines` when
/// `auto`, else a fixed count. Tests and benchmarks only.
#[cfg(test)]
#[allow(dead_code)]
pub(crate) fn set_knobs_for_test(lines: u32, auto: bool, fast_too: bool) {
	let lines = match auto {
		true => AUTO | lines.clamp(1, MAX_LINES),
		false => lines.min(MAX_LINES),
	};

	WORD.store(INIT | lines | if fast_too { FAST } else { 0 }, Ordering::Relaxed);
}

/// `prefetcht0` of `lines` consecutive 64-byte lines starting at the line that
/// holds `addr`. A hint: it never faults, so a line past the allocation is
/// harmless (a wasted fetch), and it is not ordered by locked instructions or
/// fences, so it proceeds while the reader bumps the refcount and drops the
/// shard guard.
#[inline(always)]
pub(crate) fn prefetch_lines(addr: usize, lines: u32) {
	#[cfg(target_arch = "x86_64")]
	{
		let base = addr & !63;

		for i in 0..lines as usize {
			// SAFETY: a prefetch is a hint and never dereferences in the
			// architectural sense; any address is allowed.
			unsafe {
				core::arch::x86_64::_mm_prefetch::<{ core::arch::x86_64::_MM_HINT_T0 }>(
					(base + i * 64) as *const i8,
				);
			}
		}
	}

	#[cfg(not(target_arch = "x86_64"))]
	{
		let _ = (addr, lines);
	}
}

#[cfg(test)]
mod tests {
	use super::*;

	/// Every setting, on values of every size in both tiers, in this build's
	/// layout: a prefetch past an item's end or of a fixed count larger than
	/// it is a hint, and the value reads back whole after it.
	#[cfg(feature = "hybrid_cache_common")]
	#[test]
	fn a_prefetch_of_any_count_leaves_the_value_intact() {
		use crate::{Tier, value::TieredValue};

		let _guard = crate::global_counter_lock();

		let settings = [
			ReadKnobs(INIT),
			ReadKnobs(INIT | 1),
			ReadKnobs(INIT | MAX_LINES | FAST),
			ReadKnobs(INIT | AUTO | 1 | FAST),
			ReadKnobs(INIT | AUTO | MAX_LINES | FAST),
			ReadKnobs(INIT | AUTO | MAX_LINES),
		];

		for len in [0usize, 1, 63, 64, 65, 1000, 4096, 1 << 16] {
			let bytes: Vec<u8> = (0..len).map(|i| (i % 253) as u8).collect();

			for tier in [Tier::Fast, Tier::Slow] {
				let value = TieredValue::new_in(7u64, &bytes, tier, None);

				for knobs in settings {
					value.prefetch(knobs);
					assert_eq!(value.bytes(), &bytes[..], "len {len} in {tier:?} under {knobs:?}");
				}
			}
		}
	}

	#[test]
	fn the_lines_knob_parses() {
		assert_eq!(parse_lines("0"), 0);
		assert_eq!(parse_lines(""), 0);
		assert_eq!(parse_lines("junk"), 0);
		assert_eq!(parse_lines("8"), 8);
		assert_eq!(parse_lines("500"), MAX_LINES);
		assert_eq!(parse_lines("auto"), AUTO | MAX_LINES);
		assert_eq!(parse_lines(" auto:16 "), AUTO | 16);
		assert_eq!(parse_lines("auto:0"), AUTO | 1);
		assert_eq!(parse_lines("auto:999"), AUTO | MAX_LINES);
		assert_eq!(parse_lines("auto:x"), AUTO | MAX_LINES);
	}

	#[test]
	fn auto_takes_the_span_up_to_the_cap() {
		let auto = ReadKnobs(INIT | AUTO | 16);
		assert_eq!(auto.lines_for(3), 3);
		assert_eq!(auto.lines_for(40), 16);
		assert_eq!(auto.lines_for(0), 0);

		let fixed = ReadKnobs(INIT | 4);
		assert_eq!(fixed.lines_for(1), 4);
		assert_eq!(fixed.lines_for(40), 4);
	}
}
