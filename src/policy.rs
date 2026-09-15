/*
 * Copyright (c) Kia Shakiba
 *
 * This source code is licensed under the GNU AGPLv3 license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::{
	fmt::{self, Display},
	str::FromStr,
};

use serde::{
	Deserialize,
	de::{self, Deserializer, Visitor},
};

use crate::error::CacheError;

#[derive(PartialEq, Clone, Copy, Debug)]
pub enum PaperPolicy {
	Auto,
	LfuCompact,

	/// Slab-layout counterparts of the single-queue designs. Same policy,
	/// different storage -- and unlike the `HashList` originals these honour
	/// `eviction_stacks_pmem`, because `CompactQueueSet` is
	/// allocator-parameterised.
	FifoCompact,
	ClockCompact,
	SieveCompact,
	MruCompact,

	/// Faithful three-queue 2Q over the compact slab layout. Takes the
	/// same two ratios as [`PaperPolicy::TwoQ`] and evicts in exactly the
	/// same order; only the per-object bookkeeping is cheaper.
	/// `2q-compact-<k_in>-<k_out>`.
	TwoQCompact(f64, f64),
	Lfu,
	Fifo,
	Clock,
	Sieve,
	LruCompact,
	Lru,
	Mru,
	TwoQ(f64, f64),
	Arc,
	SThreeFifo(f64),

	/// S3-FIFO over the compact slab layout. Same ratio as
	/// [`PaperPolicy::SThreeFifo`] and the same eviction order; only the
	/// per-object bookkeeping is cheaper. `s3-fifo-compact-<ratio>`.
	SThreeFifoCompact(f64),

	/// The tier-segmented LRU policy over a slab-backed recency list --
	/// same algorithm, one structure instead of two. See
	/// `LruCompactHybridStack`.
	LruCompactHybrid,

	/// Same policy as `LruCompactHybrid`, with the tier copy deferred --
	/// see `LruLazyCopyCompactHybridStack`.
	LruLazyCopyCompactHybrid,
	LfuCompactHybrid,
	TwoQCompactHybrid(f64),
	TwoQFastAdmissionCompactHybrid(f64),
	TwoQFastAdmissionReprieveCompactHybrid(f64),
	/// The full (three-queue) 2Q with fast-tier admission -- the only
	/// hybrid design whose queue algorithm matches [`PaperPolicy::TwoQ`]'s,
	/// and the only hybrid carrying TWO parameters: `k_in` sizes the
	/// fast-tier probation FIFO and `k_out` sizes the slow-tier `a1_out`
	/// overflow FIFO, which holds real resident objects rather than ghosts.
	/// `k_out` is a live parameter here; `TwoQ` writes its equivalent and
	/// never reads it. See
	/// `worker::policy::policy_stack::two_q_full_fast_admission_hybrid_stack`.
	TwoQFullFastAdmissionCompactHybrid(f64, f64),
	FifoCompactHybrid,

	/// [`PaperPolicy::ClockCompact`]'s policy, tier-segmented: the same
	/// insertion-ordered queue `FifoCompactHybrid` uses, plus a reference bit
	/// that buys an object one pass of the hand. `clock-compact-hybrid`.
	///
	/// The reason it exists is the merged object store. There the map IS the
	/// eviction stack, so an LRU hit relinks under a shard WRITE lock; a CLOCK
	/// hit is one relaxed store into a byte of the slot's tail padding, under
	/// the READ lock. See `merged_store::MergedOrder::Clock`.
	ClockCompactHybrid,
	LruSizedCompactHybrid,
	/// Recency (LRU) in the fast tier, frequency (LFU) in the slow tier.
	/// The parameter is `promote_k`: how many accesses a slow-tier object
	/// must accumulate to earn promotion into the fast tier. Carried in the
	/// policy string (like `TwoQCompactHybrid`'s `k_in`) rather than being
	/// runtime-configurable, because it is a policy parameter, not a size.
	LruLfuCompactHybrid(u16),
	S3FifoCompactHybrid(f64),

	/// Faithful tier-segmented S3-FIFO: 0..=3 counter, lazy promotion,
	/// lazy eviction. `s3-fifo-faithful-compact-hybrid-<ratio>`.
	S3FifoFaithfulCompactHybrid(f64),
	/// Faithful tier-segmented S3-FIFO: 0..=3 counter, lazy promotion,
	/// lazy eviction. `s3-fifo-faithful-fast-admission-compact-hybrid-<ratio>`.
	S3FifoFaithfulFastAdmissionCompactHybrid(f64),
	/// Faithful tier-segmented S3-FIFO: 0..=3 counter, lazy promotion,
	/// lazy eviction. `s3-fifo-faithful-reprieve-compact-hybrid-<ratio>`.
	S3FifoFaithfulReprieveCompactHybrid(f64),
	/// Faithful tier-segmented S3-FIFO: 0..=3 counter, lazy promotion,
	/// lazy eviction. `s3-fifo-faithful-fast-admission-reprieve-compact-hybrid-<ratio>`.
	S3FifoFaithfulFastAdmissionReprieveCompactHybrid(f64),
	TwoQGhostCompactHybrid(f64),
	S3FifoGhostCompactHybrid(f64),
	S3FifoGhostLazyDemotionCompactHybrid(f64),
	S3FifoGhostLazyDemotionFastAdmissionCompactHybrid(f64),
	S3FifoGhostLazyDemotionFastAdmissionMidpointCompactHybrid(f64),
	S3FifoLazyDemotionFastAdmissionMidpointReprieveCompactHybrid(f64),
	S3FifoLazyDemotionFastAdmissionReprieveCompactHybrid(f64),
	S3FifoLazyDemotionReprieveCompactHybrid(f64),
	S3FifoLazyDemotionFastAdmissionSplitSlowReprieveCompactHybrid(f64),
}

impl PaperPolicy {
	/// Whether this policy is one of the tiered (hybrid) designs.
	#[must_use]
	pub fn is_hybrid(&self) -> bool {
		matches!(self, PaperPolicy::FifoCompactHybrid { .. } | PaperPolicy::ClockCompactHybrid { .. } | PaperPolicy::LfuCompactHybrid { .. } | PaperPolicy::LruCompactHybrid { .. } | PaperPolicy::LruLazyCopyCompactHybrid { .. } | PaperPolicy::LruLfuCompactHybrid { .. } | PaperPolicy::LruSizedCompactHybrid { .. } | PaperPolicy::S3FifoGhostCompactHybrid { .. } | PaperPolicy::S3FifoGhostLazyDemotionFastAdmissionCompactHybrid { .. } | PaperPolicy::S3FifoGhostLazyDemotionFastAdmissionMidpointCompactHybrid { .. } | PaperPolicy::S3FifoGhostLazyDemotionCompactHybrid { .. } | PaperPolicy::S3FifoCompactHybrid { .. } | PaperPolicy::S3FifoLazyDemotionFastAdmissionMidpointReprieveCompactHybrid { .. } | PaperPolicy::S3FifoLazyDemotionFastAdmissionReprieveCompactHybrid { .. } | PaperPolicy::S3FifoLazyDemotionFastAdmissionSplitSlowReprieveCompactHybrid { .. } | PaperPolicy::S3FifoLazyDemotionReprieveCompactHybrid { .. } | PaperPolicy::TwoQFastAdmissionCompactHybrid { .. } | PaperPolicy::TwoQFastAdmissionReprieveCompactHybrid { .. } | PaperPolicy::TwoQFullFastAdmissionCompactHybrid { .. } | PaperPolicy::TwoQGhostCompactHybrid { .. } | PaperPolicy::S3FifoFaithfulCompactHybrid { .. } | PaperPolicy::S3FifoFaithfulFastAdmissionCompactHybrid { .. } | PaperPolicy::S3FifoFaithfulReprieveCompactHybrid { .. } | PaperPolicy::S3FifoFaithfulFastAdmissionReprieveCompactHybrid { .. } | PaperPolicy::TwoQCompactHybrid { .. })
	}

	pub fn is_auto(&self) -> bool {
		matches!(self, PaperPolicy::Auto)
	}
}

impl Display for PaperPolicy {
	fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
		match self {
			PaperPolicy::Auto => write!(f, "auto"),
			PaperPolicy::LfuCompact => write!(f, "lfu-compact"),
			PaperPolicy::FifoCompact => write!(f, "fifo-compact"),
			PaperPolicy::ClockCompact => write!(f, "clock-compact"),
			PaperPolicy::SieveCompact => write!(f, "sieve-compact"),
			PaperPolicy::MruCompact => write!(f, "mru-compact"),
			PaperPolicy::Lfu => write!(f, "lfu"),
			PaperPolicy::Fifo => write!(f, "fifo"),
			PaperPolicy::Clock => write!(f, "clock"),
			PaperPolicy::Sieve => write!(f, "sieve"),
			PaperPolicy::LruCompact => write!(f, "lru-compact"),
			PaperPolicy::Lru => write!(f, "lru"),
			PaperPolicy::Mru => write!(f, "mru"),
			PaperPolicy::TwoQ(k_in, k_out) => write!(f, "2q-{k_in}-{k_out}"),
			PaperPolicy::TwoQCompact(k_in, k_out) => write!(f, "2q-compact-{k_in}-{k_out}"),
			PaperPolicy::Arc => write!(f, "arc"),
			PaperPolicy::SThreeFifo(ratio) => write!(f, "s3-fifo-{ratio}"),
			PaperPolicy::SThreeFifoCompact(ratio) => write!(f, "s3-fifo-compact-{ratio}"),
			PaperPolicy::TwoQCompactHybrid(k_in) => write!(f, "2q-compact-hybrid-{k_in}"),
			PaperPolicy::TwoQFastAdmissionCompactHybrid(k_in) => write!(f, "2q-fast-admission-compact-hybrid-{k_in}"),
			PaperPolicy::TwoQFastAdmissionReprieveCompactHybrid(k_in) => write!(f, "2q-fast-admission-reprieve-compact-hybrid-{k_in}"),
			PaperPolicy::TwoQFullFastAdmissionCompactHybrid(k_in, k_out) => write!(f, "2q-full-fast-admission-compact-hybrid-{k_in}-{k_out}"),
			PaperPolicy::FifoCompactHybrid => write!(f, "fifo-compact-hybrid"),
			PaperPolicy::ClockCompactHybrid => write!(f, "clock-compact-hybrid"),
			PaperPolicy::LruSizedCompactHybrid => write!(f, "lru-sized-compact-hybrid"),
			PaperPolicy::LruCompactHybrid => write!(f, "lru-compact-hybrid"),
			PaperPolicy::LruLazyCopyCompactHybrid => write!(f, "lru-lazy-copy-compact-hybrid"),
			PaperPolicy::LfuCompactHybrid => write!(f, "lfu-compact-hybrid"),
			PaperPolicy::LruLfuCompactHybrid(promote_k) => write!(f, "lru-lfu-compact-hybrid-{promote_k}"),
			PaperPolicy::S3FifoCompactHybrid(ratio) => write!(f, "s3-fifo-compact-hybrid-{ratio}"),
			PaperPolicy::S3FifoFaithfulCompactHybrid(ratio) => write!(f, "s3-fifo-faithful-compact-hybrid-{ratio}"),
			PaperPolicy::S3FifoFaithfulFastAdmissionCompactHybrid(ratio) => write!(f, "s3-fifo-faithful-fast-admission-compact-hybrid-{ratio}"),
			PaperPolicy::S3FifoFaithfulReprieveCompactHybrid(ratio) => write!(f, "s3-fifo-faithful-reprieve-compact-hybrid-{ratio}"),
			PaperPolicy::S3FifoFaithfulFastAdmissionReprieveCompactHybrid(ratio) => write!(f, "s3-fifo-faithful-fast-admission-reprieve-compact-hybrid-{ratio}"),
			PaperPolicy::TwoQGhostCompactHybrid(k_in) => write!(f, "2q-ghost-compact-hybrid-{k_in}"),
			PaperPolicy::S3FifoGhostCompactHybrid(ratio) => write!(f, "s3-fifo-ghost-compact-hybrid-{ratio}"),
			PaperPolicy::S3FifoGhostLazyDemotionCompactHybrid(ratio) => write!(f, "s3-fifo-ghost-lazy-demotion-compact-hybrid-{ratio}"),
			PaperPolicy::S3FifoGhostLazyDemotionFastAdmissionCompactHybrid(ratio) => write!(f, "s3-fifo-ghost-lazy-demotion-fast-admission-compact-hybrid-{ratio}"),
			PaperPolicy::S3FifoGhostLazyDemotionFastAdmissionMidpointCompactHybrid(ratio) => write!(f, "s3-fifo-ghost-lazy-demotion-fast-admission-midpoint-compact-hybrid-{ratio}"),
			PaperPolicy::S3FifoLazyDemotionFastAdmissionMidpointReprieveCompactHybrid(ratio) => write!(f, "s3-fifo-lazy-demotion-fast-admission-midpoint-reprieve-compact-hybrid-{ratio}"),
			PaperPolicy::S3FifoLazyDemotionFastAdmissionReprieveCompactHybrid(ratio) => write!(f, "s3-fifo-lazy-demotion-fast-admission-reprieve-compact-hybrid-{ratio}"),
			PaperPolicy::S3FifoLazyDemotionReprieveCompactHybrid(ratio) => write!(f, "s3-fifo-lazy-demotion-reprieve-compact-hybrid-{ratio}"),
			PaperPolicy::S3FifoLazyDemotionFastAdmissionSplitSlowReprieveCompactHybrid(ratio) => write!(f, "s3-fifo-lazy-demotion-fast-admission-split-slow-reprieve-compact-hybrid-{ratio}"),
		}
	}
}

impl FromStr for PaperPolicy {
	type Err = CacheError;

	fn from_str(value: &str) -> Result<Self, Self::Err> {
		let policy = match value {
			"auto" => PaperPolicy::Auto,
			"lfu-compact" => PaperPolicy::LfuCompact,
			"fifo-compact" => PaperPolicy::FifoCompact,
			"clock-compact" => PaperPolicy::ClockCompact,
			"sieve-compact" => PaperPolicy::SieveCompact,
			"mru-compact" => PaperPolicy::MruCompact,
			"lfu" => PaperPolicy::Lfu,
			"fifo" => PaperPolicy::Fifo,
			"clock" => PaperPolicy::Clock,
			"sieve" => PaperPolicy::Sieve,
			"lru-compact" => PaperPolicy::LruCompact,
			"lru" => PaperPolicy::Lru,
			"mru" => PaperPolicy::Mru,
			// Order matters and is load-bearing: every guard below also starts
			// with a prefix of the ones above it ("2q-fast-admission-compact-
			// hybrid-" starts with "2q-", and so does "2q-compact-hybrid-"), so
			// the most specific prefix has to be tested first or a more general
			// guard silently swallows it. See
			// `compact_does_not_collide_with_other_2q_forms`.
			value if value.starts_with("2q-full-fast-admission-compact-hybrid-") => parse_two_q_full_fast_admission_compact_hybrid(value)?,
			value if value.starts_with("2q-fast-admission-reprieve-compact-hybrid-") => parse_two_q_fast_admission_reprieve_compact_hybrid(value)?,
			value if value.starts_with("2q-fast-admission-compact-hybrid-") => parse_two_q_fast_admission_compact_hybrid(value)?,
			value if value.starts_with("2q-ghost-compact-hybrid-") => parse_two_q_ghost_compact_hybrid(value)?,
			value if value.starts_with("2q-compact-hybrid-") => parse_two_q_compact_hybrid(value)?,
			// Must follow "2q-compact-hybrid-", which it is a prefix of.
			value if value.starts_with("2q-compact-") => parse_two_q_compact(value)?,
			value if value.starts_with("2q-") => parse_two_q(value)?,
			"arc" => PaperPolicy::Arc,
			value if value.starts_with("s3-fifo-ghost-lazy-demotion-fast-admission-midpoint-compact-hybrid-") => parse_s_three_fifo_ghost_lazy_demotion_fast_admission_midpoint_compact_hybrid(value)?,
			value if value.starts_with("s3-fifo-ghost-lazy-demotion-fast-admission-compact-hybrid-") => parse_s_three_fifo_ghost_lazy_demotion_fast_admission_compact_hybrid(value)?,
			value if value.starts_with("s3-fifo-ghost-lazy-demotion-compact-hybrid-") => parse_s_three_fifo_ghost_lazy_demotion_compact_hybrid(value)?,
			value if value.starts_with("s3-fifo-ghost-compact-hybrid-") => parse_s_three_fifo_ghost_compact_hybrid(value)?,
			value if value.starts_with("s3-fifo-compact-hybrid-") => parse_s_three_fifo_compact_hybrid(value)?,
			// The faithful family, longest stem first. All four must precede
			// the bare "s3-fifo-" guard, which would otherwise swallow them.
			value if value.starts_with("s3-fifo-faithful-fast-admission-reprieve-compact-hybrid-") => parse_s3_fifo_faithful_fast_admission_reprieve_compact_hybrid(value)?,
			value if value.starts_with("s3-fifo-faithful-fast-admission-compact-hybrid-") => parse_s3_fifo_faithful_fast_admission_compact_hybrid(value)?,
			value if value.starts_with("s3-fifo-faithful-reprieve-compact-hybrid-") => parse_s3_fifo_faithful_reprieve_compact_hybrid(value)?,
			value if value.starts_with("s3-fifo-faithful-compact-hybrid-") => parse_s3_fifo_faithful_compact_hybrid(value)?,
			// Must follow "s3-fifo-compact-hybrid-", which it is a prefix of.
			value if value.starts_with("s3-fifo-compact-") => parse_s_three_fifo_compact(value)?,
			value if value.starts_with("s3-fifo-lazy-demotion-fast-admission-midpoint-reprieve-compact-hybrid-") => parse_s_three_fifo_lazy_demotion_fast_admission_midpoint_reprieve_compact_hybrid(value)?,
			value if value.starts_with("s3-fifo-lazy-demotion-fast-admission-reprieve-compact-hybrid-") => parse_s_three_fifo_lazy_demotion_fast_admission_reprieve_compact_hybrid(value)?,
			value if value.starts_with("s3-fifo-lazy-demotion-reprieve-compact-hybrid-") => parse_s_three_fifo_lazy_demotion_reprieve_compact_hybrid(value)?,
			value if value.starts_with("s3-fifo-lazy-demotion-fast-admission-split-slow-reprieve-compact-hybrid-") => parse_s_three_fifo_lazy_demotion_fast_admission_split_slow_reprieve_compact_hybrid(value)?,
			value if value.starts_with("s3-fifo-") => parse_s_three_fifo(value)?,
			// Prefix guard, so it must be tested before any *exact* arm it
			// could be confused with is irrelevant (exact arms cannot swallow a
			// longer string) -- but it does have to precede nothing else here,
			// since no other guard starts with "lru-lfu-compact-hybrid-". Kept
			// beside the other lru forms for readability.
			value if value.starts_with("lru-lfu-compact-hybrid-") => parse_lru_lfu_compact_hybrid(value)?,
			"lru-compact-hybrid" => PaperPolicy::LruCompactHybrid,
			"lru-lazy-copy-compact-hybrid" => PaperPolicy::LruLazyCopyCompactHybrid,
			"lfu-compact-hybrid" => PaperPolicy::LfuCompactHybrid,
			"fifo-compact-hybrid" => PaperPolicy::FifoCompactHybrid,
			// Both this and "clock-compact" above are EXACT arms, so neither
			// can swallow the other however they are ordered -- unlike the
			// `starts_with` guards further up, where order is load-bearing.
			"clock-compact-hybrid" => PaperPolicy::ClockCompactHybrid,
			"lru-sized-compact-hybrid" => PaperPolicy::LruSizedCompactHybrid,

			_ => return Err(CacheError::InvalidPolicy),
		};

		Ok(policy)
	}
}

impl<'a> Deserialize<'a> for PaperPolicy {
	fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
	where
		D: Deserializer<'a>,
	{
		deserializer.deserialize_str(PaperPolicyVisitor)
	}
}

struct PaperPolicyVisitor;

impl Visitor<'_> for PaperPolicyVisitor {
	type Value = PaperPolicy;

	fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
		formatter.write_str("a PaperPolicy config")
	}

	fn visit_str<E>(self, value: &str) -> Result<Self::Value, E>
	where
		E: de::Error,
	{
		PaperPolicy::from_str(value)
			.map_err(|err| E::custom(err.to_string()))
	}
}

fn parse_two_q_compact(value: &str) -> Result<PaperPolicy, CacheError> {
	// skip the "2q-compact-"
	let tokens = value[11..]
		.split('-')
		.collect::<Vec<&str>>();

	if tokens.len() != 2 {
		return Err(CacheError::InvalidPolicy);
	}

	let Ok(k_in) = tokens[0].parse::<f64>() else {
		return Err(CacheError::InvalidPolicy);
	};

	let Ok(k_out) = tokens[1].parse::<f64>() else {
		return Err(CacheError::InvalidPolicy);
	};

	if k_in + k_out > 1.0
		|| !(0.0..=1.0).contains(&k_in)
		|| !(0.0..=1.0).contains(&k_out)
	{
		return Err(CacheError::InvalidPolicy);
	}

	Ok(PaperPolicy::TwoQCompact(k_in, k_out))
}

fn parse_two_q(value: &str) -> Result<PaperPolicy, CacheError> {
	// skip the "2q-"
	let tokens = value[3..]
		.split('-')
		.collect::<Vec<&str>>();

	if tokens.len() != 2 {
		return Err(CacheError::InvalidPolicy);
	}

	let Ok(k_in) = tokens[0].parse::<f64>() else {
		return Err(CacheError::InvalidPolicy);
	};

	let Ok(k_out) = tokens[1].parse::<f64>() else {
		return Err(CacheError::InvalidPolicy);
	};

	if k_in + k_out > 1.0
		|| !(0.0..=1.0).contains(&k_in)
		|| !(0.0..=1.0).contains(&k_out)
	{
		return Err(CacheError::InvalidPolicy);
	}

	Ok(PaperPolicy::TwoQ(k_in, k_out))
}

fn parse_lru_lfu_compact_hybrid(value: &str) -> Result<PaperPolicy, CacheError> {
	// skip the "lru-lfu-compact-hybrid-"
	let tokens = value[23..]
		.split('-')
		.collect::<Vec<&str>>();

	if tokens.len() != 1 {
		return Err(CacheError::InvalidPolicy);
	}

	let Ok(promote_k) = tokens[0].parse::<u16>() else {
		return Err(CacheError::InvalidPolicy);
	};

	// 0 would make every slow object promotable before it was ever accessed.
	// The upper bound is enforced by the stack itself (clamped to its
	// frequency cap), not here, so the policy string stays a faithful record
	// of what was asked for.
	if promote_k == 0 {
		return Err(CacheError::InvalidPolicy);
	}

	Ok(PaperPolicy::LruLfuCompactHybrid(promote_k))
}

fn parse_two_q_compact_hybrid(value: &str) -> Result<PaperPolicy, CacheError> {
	// skip the "2q-compact-hybrid-"
	let tokens = value[18..]
		.split('-')
		.collect::<Vec<&str>>();

	if tokens.len() != 1 {
		return Err(CacheError::InvalidPolicy);
	}

	let Ok(k_in) = tokens[0].parse::<f64>() else {
		return Err(CacheError::InvalidPolicy);
	};

	if !(0.0..=1.0).contains(&k_in) {
		return Err(CacheError::InvalidPolicy);
	}

	Ok(PaperPolicy::TwoQCompactHybrid(k_in))
}

fn parse_two_q_fast_admission_compact_hybrid(value: &str) -> Result<PaperPolicy, CacheError> {
	// skip the "2q-fast-admission-compact-hybrid-"
	let tokens = value[33..].split('-').collect::<Vec<&str>>();

	if tokens.len() != 1 {
		return Err(CacheError::InvalidPolicy);
	}

	let Ok(k_in) = tokens[0].parse::<f64>() else {
		return Err(CacheError::InvalidPolicy);
	};

	if !(0.0..=1.0).contains(&k_in) {
		return Err(CacheError::InvalidPolicy);
	}

	Ok(PaperPolicy::TwoQFastAdmissionCompactHybrid(k_in))
}

fn parse_two_q_fast_admission_reprieve_compact_hybrid(value: &str) -> Result<PaperPolicy, CacheError> {
	// skip the "2q-fast-admission-reprieve-compact-hybrid-"
	let tokens = value[42..]
		.split('-')
		.collect::<Vec<&str>>();

	if tokens.len() != 1 {
		return Err(CacheError::InvalidPolicy);
	}

	let Ok(k_in) = tokens[0].parse::<f64>() else {
		return Err(CacheError::InvalidPolicy);
	};

	if !(0.0..=1.0).contains(&k_in) {
		return Err(CacheError::InvalidPolicy);
	}

	Ok(PaperPolicy::TwoQFastAdmissionReprieveCompactHybrid(k_in))
}

/// The only two-token hybrid parser. Modelled on [`parse_two_q`] rather
/// than on the one-token hybrid parsers: `k_in` sizes the fast-tier
/// probation FIFO, `k_out` the slow-tier `a1_out` overflow FIFO.
///
/// Unlike `parse_two_q` there is no `k_in + k_out <= 1.0` constraint: the
/// two budgets are denominated against different physical tiers here
/// (`k_in` against DRAM, `k_out` against PMEM), so their sum is not a
/// fraction of any one thing.
fn parse_two_q_full_fast_admission_compact_hybrid(value: &str) -> Result<PaperPolicy, CacheError> {
	// skip the "2q-full-fast-admission-compact-hybrid-"
	let tokens = value[38..]
		.split('-')
		.collect::<Vec<&str>>();

	if tokens.len() != 2 {
		return Err(CacheError::InvalidPolicy);
	}

	let Ok(k_in) = tokens[0].parse::<f64>() else {
		return Err(CacheError::InvalidPolicy);
	};

	let Ok(k_out) = tokens[1].parse::<f64>() else {
		return Err(CacheError::InvalidPolicy);
	};

	if !(0.0..=1.0).contains(&k_in) || !(0.0..=1.0).contains(&k_out) {
		return Err(CacheError::InvalidPolicy);
	}

	Ok(PaperPolicy::TwoQFullFastAdmissionCompactHybrid(k_in, k_out))
}

fn parse_s_three_fifo_compact(value: &str) -> Result<PaperPolicy, CacheError> {
	// skip the "s3-fifo-compact-"
	let tokens = value[16..]
		.split('-')
		.collect::<Vec<&str>>();

	if tokens.len() != 1 {
		return Err(CacheError::InvalidPolicy);
	}

	let Ok(ratio) = tokens[0].parse::<f64>() else {
		return Err(CacheError::InvalidPolicy);
	};

	if !(0.0..1.0).contains(&ratio) {
		return Err(CacheError::InvalidPolicy);
	}

	Ok(PaperPolicy::SThreeFifoCompact(ratio))
}

fn parse_s_three_fifo(value: &str) -> Result<PaperPolicy, CacheError> {
	// skip the "s3-fifo-"
	let tokens = value[8..]
		.split('-')
		.collect::<Vec<&str>>();

	if tokens.len() != 1 {
		return Err(CacheError::InvalidPolicy);
	}

	let Ok(ratio) = tokens[0].parse::<f64>() else {
		return Err(CacheError::InvalidPolicy);
	};

	if !(0.0..1.0).contains(&ratio) {
		return Err(CacheError::InvalidPolicy);
	}

	Ok(PaperPolicy::SThreeFifo(ratio))
}

fn parse_s3_fifo_faithful_compact_hybrid(value: &str) -> Result<PaperPolicy, CacheError> {
	// skip the "s3-fifo-faithful-compact-hybrid-"
	let tokens = value[32..].split('-').collect::<Vec<&str>>();

	if tokens.len() != 1 {
		return Err(CacheError::InvalidPolicy);
	}

	let Ok(ratio) = tokens[0].parse::<f64>() else {
		return Err(CacheError::InvalidPolicy);
	};

	// Excludes 1.0: main is sized at `(1 - ratio) * max_size`, so 1.0 leaves it
	// zero bytes and an empty main reports itself full forever.
	if !(0.0..1.0).contains(&ratio) {
		return Err(CacheError::InvalidPolicy);
	}

	Ok(PaperPolicy::S3FifoFaithfulCompactHybrid(ratio))
}

fn parse_s3_fifo_faithful_fast_admission_compact_hybrid(value: &str) -> Result<PaperPolicy, CacheError> {
	// skip the "s3-fifo-faithful-fast-admission-compact-hybrid-"
	let tokens = value[47..].split('-').collect::<Vec<&str>>();

	if tokens.len() != 1 {
		return Err(CacheError::InvalidPolicy);
	}

	let Ok(ratio) = tokens[0].parse::<f64>() else {
		return Err(CacheError::InvalidPolicy);
	};

	// Excludes 1.0: main is sized at `(1 - ratio) * max_size`, so 1.0 leaves it
	// zero bytes and an empty main reports itself full forever.
	if !(0.0..1.0).contains(&ratio) {
		return Err(CacheError::InvalidPolicy);
	}

	Ok(PaperPolicy::S3FifoFaithfulFastAdmissionCompactHybrid(ratio))
}

fn parse_s3_fifo_faithful_reprieve_compact_hybrid(value: &str) -> Result<PaperPolicy, CacheError> {
	// skip the "s3-fifo-faithful-reprieve-compact-hybrid-"
	let tokens = value[41..].split('-').collect::<Vec<&str>>();

	if tokens.len() != 1 {
		return Err(CacheError::InvalidPolicy);
	}

	let Ok(ratio) = tokens[0].parse::<f64>() else {
		return Err(CacheError::InvalidPolicy);
	};

	// Excludes 1.0: main is sized at `(1 - ratio) * max_size`, so 1.0 leaves it
	// zero bytes and an empty main reports itself full forever.
	if !(0.0..1.0).contains(&ratio) {
		return Err(CacheError::InvalidPolicy);
	}

	Ok(PaperPolicy::S3FifoFaithfulReprieveCompactHybrid(ratio))
}

fn parse_s3_fifo_faithful_fast_admission_reprieve_compact_hybrid(value: &str) -> Result<PaperPolicy, CacheError> {
	// skip the "s3-fifo-faithful-fast-admission-reprieve-compact-hybrid-"
	let tokens = value[56..].split('-').collect::<Vec<&str>>();

	if tokens.len() != 1 {
		return Err(CacheError::InvalidPolicy);
	}

	let Ok(ratio) = tokens[0].parse::<f64>() else {
		return Err(CacheError::InvalidPolicy);
	};

	// Excludes 1.0: main is sized at `(1 - ratio) * max_size`, so 1.0 leaves it
	// zero bytes and an empty main reports itself full forever.
	if !(0.0..1.0).contains(&ratio) {
		return Err(CacheError::InvalidPolicy);
	}

	Ok(PaperPolicy::S3FifoFaithfulFastAdmissionReprieveCompactHybrid(ratio))
}

fn parse_s_three_fifo_compact_hybrid(value: &str) -> Result<PaperPolicy, CacheError> {
	// skip the "s3-fifo-compact-hybrid-"
	let tokens = value[23..]
		.split('-')
		.collect::<Vec<&str>>();

	if tokens.len() != 1 {
		return Err(CacheError::InvalidPolicy);
	}

	let Ok(ratio) = tokens[0].parse::<f64>() else {
		return Err(CacheError::InvalidPolicy);
	};

	// EXCLUSIVE of 1.0, matching `parse_s_three_fifo_hybrid`. This stack
	// sizes its main queue at `(1 - ratio) * max_size`, so a ratio of
	// exactly 1 leaves it zero bytes; `Stack::is_full` is `used >= max`,
	// so an EMPTY main queue then reports itself full, `evict_main` pops
	// nothing while the cache is still over budget, and `apply_evictions`
	// spins. The baseline parser was tightened for this; the clone kept
	// the old inclusive bound.
	if !(0.0..1.0).contains(&ratio) {
		return Err(CacheError::InvalidPolicy);
	}

	Ok(PaperPolicy::S3FifoCompactHybrid(ratio))
}

fn parse_two_q_ghost_compact_hybrid(value: &str) -> Result<PaperPolicy, CacheError> {
	// skip the "2q-ghost-compact-hybrid-"
	let tokens = value[24..]
		.split('-')
		.collect::<Vec<&str>>();

	if tokens.len() != 1 {
		return Err(CacheError::InvalidPolicy);
	}

	let Ok(k_in) = tokens[0].parse::<f64>() else {
		return Err(CacheError::InvalidPolicy);
	};

	if !(0.0..=1.0).contains(&k_in) {
		return Err(CacheError::InvalidPolicy);
	}

	Ok(PaperPolicy::TwoQGhostCompactHybrid(k_in))
}

fn parse_s_three_fifo_ghost_compact_hybrid(value: &str) -> Result<PaperPolicy, CacheError> {
	// skip the "s3-fifo-ghost-compact-hybrid-"
	let tokens = value[29..]
		.split('-')
		.collect::<Vec<&str>>();

	if tokens.len() != 1 {
		return Err(CacheError::InvalidPolicy);
	}

	let Ok(ratio) = tokens[0].parse::<f64>() else {
		return Err(CacheError::InvalidPolicy);
	};

	if !(0.0..1.0).contains(&ratio) {
		return Err(CacheError::InvalidPolicy);
	}

	Ok(PaperPolicy::S3FifoGhostCompactHybrid(ratio))
}

fn parse_s_three_fifo_ghost_lazy_demotion_compact_hybrid(value: &str) -> Result<PaperPolicy, CacheError> {
	// skip the "s3-fifo-ghost-lazy-demotion-compact-hybrid-"
	let tokens = value[43..]
		.split('-')
		.collect::<Vec<&str>>();

	if tokens.len() != 1 {
		return Err(CacheError::InvalidPolicy);
	}

	let Ok(ratio) = tokens[0].parse::<f64>() else {
		return Err(CacheError::InvalidPolicy);
	};

	if !(0.0..1.0).contains(&ratio) {
		return Err(CacheError::InvalidPolicy);
	}

	Ok(PaperPolicy::S3FifoGhostLazyDemotionCompactHybrid(ratio))
}

fn parse_s_three_fifo_ghost_lazy_demotion_fast_admission_compact_hybrid(value: &str) -> Result<PaperPolicy, CacheError> {
	// skip the "s3-fifo-ghost-lazy-demotion-fast-admission-compact-hybrid-"
	let tokens = value[58..]
		.split('-')
		.collect::<Vec<&str>>();

	if tokens.len() != 1 {
		return Err(CacheError::InvalidPolicy);
	}

	let Ok(ratio) = tokens[0].parse::<f64>() else {
		return Err(CacheError::InvalidPolicy);
	};

	if !(0.0..1.0).contains(&ratio) {
		return Err(CacheError::InvalidPolicy);
	}

	Ok(PaperPolicy::S3FifoGhostLazyDemotionFastAdmissionCompactHybrid(ratio))
}

fn parse_s_three_fifo_ghost_lazy_demotion_fast_admission_midpoint_compact_hybrid(value: &str) -> Result<PaperPolicy, CacheError> {
	// skip the "s3-fifo-ghost-lazy-demotion-fast-admission-midpoint-compact-hybrid-"
	let tokens = value[67..]
		.split('-')
		.collect::<Vec<&str>>();

	if tokens.len() != 1 {
		return Err(CacheError::InvalidPolicy);
	}

	let Ok(ratio) = tokens[0].parse::<f64>() else {
		return Err(CacheError::InvalidPolicy);
	};

	if !(0.0..1.0).contains(&ratio) {
		return Err(CacheError::InvalidPolicy);
	}

	Ok(PaperPolicy::S3FifoGhostLazyDemotionFastAdmissionMidpointCompactHybrid(ratio))
}

fn parse_s_three_fifo_lazy_demotion_fast_admission_midpoint_reprieve_compact_hybrid(value: &str) -> Result<PaperPolicy, CacheError> {
	// skip the "s3-fifo-lazy-demotion-fast-admission-midpoint-reprieve-compact-hybrid-"
	let tokens = value[70..]
		.split('-')
		.collect::<Vec<&str>>();

	if tokens.len() != 1 {
		return Err(CacheError::InvalidPolicy);
	}

	let Ok(ratio) = tokens[0].parse::<f64>() else {
		return Err(CacheError::InvalidPolicy);
	};

	if !(0.0..=1.0).contains(&ratio) {
		return Err(CacheError::InvalidPolicy);
	}

	Ok(PaperPolicy::S3FifoLazyDemotionFastAdmissionMidpointReprieveCompactHybrid(ratio))
}

fn parse_s_three_fifo_lazy_demotion_fast_admission_reprieve_compact_hybrid(value: &str) -> Result<PaperPolicy, CacheError> {
	// skip the "s3-fifo-lazy-demotion-fast-admission-reprieve-compact-hybrid-"
	let tokens = value[61..]
		.split('-')
		.collect::<Vec<&str>>();

	if tokens.len() != 1 {
		return Err(CacheError::InvalidPolicy);
	}

	let Ok(ratio) = tokens[0].parse::<f64>() else {
		return Err(CacheError::InvalidPolicy);
	};

	if !(0.0..=1.0).contains(&ratio) {
		return Err(CacheError::InvalidPolicy);
	}

	Ok(PaperPolicy::S3FifoLazyDemotionFastAdmissionReprieveCompactHybrid(ratio))
}

fn parse_s_three_fifo_lazy_demotion_reprieve_compact_hybrid(value: &str) -> Result<PaperPolicy, CacheError> {
	// skip the "s3-fifo-lazy-demotion-reprieve-compact-hybrid-"
	let tokens = value[46..]
		.split('-')
		.collect::<Vec<&str>>();

	if tokens.len() != 1 {
		return Err(CacheError::InvalidPolicy);
	}

	let Ok(ratio) = tokens[0].parse::<f64>() else {
		return Err(CacheError::InvalidPolicy);
	};

	if !(0.0..=1.0).contains(&ratio) {
		return Err(CacheError::InvalidPolicy);
	}

	Ok(PaperPolicy::S3FifoLazyDemotionReprieveCompactHybrid(ratio))
}

fn parse_s_three_fifo_lazy_demotion_fast_admission_split_slow_reprieve_compact_hybrid(value: &str) -> Result<PaperPolicy, CacheError> {
	// skip the "s3-fifo-lazy-demotion-fast-admission-split-slow-reprieve-compact-hybrid-"
	let tokens = value[72..]
		.split('-')
		.collect::<Vec<&str>>();

	if tokens.len() != 1 {
		return Err(CacheError::InvalidPolicy);
	}

	let Ok(ratio) = tokens[0].parse::<f64>() else {
		return Err(CacheError::InvalidPolicy);
	};

	if !(0.0..=1.0).contains(&ratio) {
		return Err(CacheError::InvalidPolicy);
	}

	Ok(PaperPolicy::S3FifoLazyDemotionFastAdmissionSplitSlowReprieveCompactHybrid(ratio))
}

#[cfg(test)]
mod tests {
	use super::*;

	/// Locks in `FromStr`'s guard ordering for the compact forms. Both new
	/// strings start with a stem an existing guard already claims:
	/// `"2q-compact-0.25-0.5"` starts with `"2q-"`, and
	/// `"2q-compact-hybrid-0.2"` starts with `"2q-compact-"`. Get the order
	/// wrong in either direction and one of the two parses as the other
	/// policy with no error, which a run would report under the wrong name.
	#[test]
	fn compact_does_not_collide_with_other_2q_forms() {
		assert_eq!(
			"2q-compact-0.25-0.5".parse::<PaperPolicy>(),
			Ok(PaperPolicy::TwoQCompact(0.25, 0.5)),
		);

		// The hybrid is the longer form and must still win its own string.
		assert_eq!(
			"2q-compact-hybrid-0.2".parse::<PaperPolicy>(),
			Ok(PaperPolicy::TwoQCompactHybrid(0.2)),
		);

		// Unchanged by the new guard.
		assert_eq!("2q-0.25-0.5".parse::<PaperPolicy>(), Ok(PaperPolicy::TwoQ(0.25, 0.5)));

		assert_eq!(PaperPolicy::TwoQCompact(0.25, 0.5).to_string(), "2q-compact-0.25-0.5");

		assert_eq!(
			PaperPolicy::TwoQCompact(0.25, 0.5).to_string().parse::<PaperPolicy>(),
			Ok(PaperPolicy::TwoQCompact(0.25, 0.5)),
		);

		// The compact stack is all-DRAM, not a tiered design.
		assert!(!PaperPolicy::TwoQCompact(0.25, 0.5).is_hybrid());

		// A one-token argument is 2Q's other arity and must be rejected, not
		// silently accepted with a defaulted second ratio.
		assert!("2q-compact-0.25".parse::<PaperPolicy>().is_err());
	}

	/// Same guard-ordering hazard as `compact_does_not_collide_with_other_2q_forms`,
	/// for S3-FIFO: `"s3-fifo-compact-0.1"` starts with `"s3-fifo-"`, and
	/// `"s3-fifo-compact-hybrid-0.1"` starts with `"s3-fifo-compact-"`.
	#[test]
	fn compact_does_not_collide_with_other_s3_fifo_forms() {
		assert_eq!(
			"s3-fifo-compact-0.1".parse::<PaperPolicy>(),
			Ok(PaperPolicy::SThreeFifoCompact(0.1)),
		);

		assert_eq!(
			"s3-fifo-compact-hybrid-0.1".parse::<PaperPolicy>(),
			Ok(PaperPolicy::S3FifoCompactHybrid(0.1)),
		);

		// Unchanged by the new guard.
		assert_eq!("s3-fifo-0.1".parse::<PaperPolicy>(), Ok(PaperPolicy::SThreeFifo(0.1)));

		assert_eq!(PaperPolicy::SThreeFifoCompact(0.1).to_string(), "s3-fifo-compact-0.1");

		assert_eq!(
			PaperPolicy::SThreeFifoCompact(0.1).to_string().parse::<PaperPolicy>(),
			Ok(PaperPolicy::SThreeFifoCompact(0.1)),
		);

		assert!(!PaperPolicy::SThreeFifoCompact(0.1).is_hybrid());

		// `ratio` is a fraction of the cache; 1.0 would leave `main` empty.
		assert!("s3-fifo-compact-1.0".parse::<PaperPolicy>().is_err());
	}

	/// The prefixes whose designs size a main queue at `(1 - ratio) * max_size`
	/// -- the plain stack and the five corrected hybrids. These EXCLUDE 1.0.
	///
	/// Enumerated rather than spot-checked because the bound lives in ten
	/// separately hand-written parsers; the realistic mistake is tightening
	/// five of six, or tightening one of the reprieve four by copy-paste.
	#[cfg(test)]
	const S3_FIFO_MAIN_SIZED_PREFIXES: &[&str] = &[
		"s3-fifo-",
		"s3-fifo-compact-hybrid-",
		"s3-fifo-ghost-compact-hybrid-",
		"s3-fifo-ghost-lazy-demotion-compact-hybrid-",
		"s3-fifo-ghost-lazy-demotion-fast-admission-compact-hybrid-",
		"s3-fifo-ghost-lazy-demotion-fast-admission-midpoint-compact-hybrid-",
	];

	/// The four reprieve designs, which derive no budget from `1 - ratio` and
	/// so keep the INCLUSIVE bound. Their `evict_one` is purely the main
	/// queue's tail loop; the one-access queue is drained by
	/// `settle_one_access()` and never reaches eviction, so the
	/// `!main.is_full()` dispatch gate `main_capacity` serves is absent.
	#[cfg(test)]
	const S3_FIFO_REPRIEVE_PREFIXES: &[&str] = &[
		"s3-fifo-lazy-demotion-fast-admission-midpoint-reprieve-compact-hybrid-",
		"s3-fifo-lazy-demotion-fast-admission-reprieve-compact-hybrid-",
		"s3-fifo-lazy-demotion-reprieve-compact-hybrid-",
		"s3-fifo-lazy-demotion-fast-admission-split-slow-reprieve-compact-hybrid-",
	];

	/// A ratio of exactly 1 gives the main queue `(1 - 1) * max_size == 0`
	/// bytes. `Stack::is_full` is `used >= max`, so an *empty* main queue
	/// reports itself full, `evict_one` declines to touch the one-access
	/// queue, `evict_main` pops nothing, and the eviction loop spins on a
	/// cache it can never bring under budget. Rejecting the endpoint at parse
	/// time makes that state unreachable.
	#[test]
	fn s3_fifo_family_rejects_a_ratio_of_exactly_one() {
		for prefix in S3_FIFO_MAIN_SIZED_PREFIXES {
			let policy = format!("{prefix}1.0");

			assert_eq!(
				policy.parse::<PaperPolicy>(),
				Err(CacheError::InvalidPolicy),
				"{policy} should be rejected: it leaves the main queue zero bytes",
			);
		}
	}

	/// The exclusion has to be an endpoint exclusion and nothing more -- a
	/// `<` accidentally written where `<=` was meant elsewhere in the guard
	/// would also reject everything below 1, and the test above would still
	/// pass.
	#[test]
	fn s3_fifo_family_still_accepts_ratios_just_below_one() {
		for prefix in S3_FIFO_MAIN_SIZED_PREFIXES {
			let policy = format!("{prefix}0.999");

			assert!(
				policy.parse::<PaperPolicy>().is_ok(),
				"{policy} should parse: 0.999 leaves both queues a real budget",
			);
		}
	}

	/// 0 stays legal. It means "no one-access queue", which is a coherent
	/// request -- every insert goes straight to main -- and unlike 1 it
	/// starves no queue that eviction depends on.
	#[test]
	fn s3_fifo_family_still_accepts_a_ratio_of_zero() {
		for prefix in S3_FIFO_MAIN_SIZED_PREFIXES.iter().chain(S3_FIFO_REPRIEVE_PREFIXES) {
			let policy = format!("{prefix}0.0");

			assert!(policy.parse::<PaperPolicy>().is_ok(), "{policy} should parse");
		}
	}

	/// The reprieve designs accept the upper endpoint the other six refuse.
	/// Nothing in them computes `1 - ratio`, so a ratio of 1 starves no queue
	/// -- `settle_one_access()` still drains the one-access queue against its
	/// own capacity, and `evict_one` still drains the main tail.
	#[test]
	fn reprieve_designs_accept_a_ratio_of_exactly_one() {
		for prefix in S3_FIFO_REPRIEVE_PREFIXES {
			let policy = format!("{prefix}1.0");

			assert!(
				policy.parse::<PaperPolicy>().is_ok(),
				"{policy} should parse: this design sizes no queue at (1 - ratio)",
			);
		}
	}

	/// ...and the split is exactly where it should be: no design appears on
	/// both lists, and between them they cover all ten parsers.
	#[test]
	fn every_s3_fifo_prefix_is_on_exactly_one_side_of_the_split() {
		for prefix in S3_FIFO_REPRIEVE_PREFIXES {
			assert!(
				!S3_FIFO_MAIN_SIZED_PREFIXES.contains(prefix),
				"{prefix} is on both sides of the bound split",
			);
		}

		assert_eq!(
			S3_FIFO_MAIN_SIZED_PREFIXES.len() + S3_FIFO_REPRIEVE_PREFIXES.len(),
			10,
			"the two lists should account for all ten s3-fifo parsers",
		);
	}

	/// The 2Q family deliberately keeps the INCLUSIVE bound. No 2Q stack
	/// derives a budget from `1 - k_in`: `fifo_capacity` is `k_in * max_size`
	/// and the main queue is bounded by the cache's overall `max_size`, so
	/// `k_in == 1.0` hands the FIFO queue the whole cache -- extreme, but
	/// every queue still has capacity and nothing spins. Tightening these to
	/// match s3-fifo would break working call sites to fix nothing.
	#[test]
	fn two_q_family_still_accepts_a_ratio_of_exactly_one() {
		for policy in [
			// Plain `2q-` takes both k_in and k_out, and separately
			// requires they sum to at most 1 -- so k_out is 0 here to
			// isolate k_in at its upper bound.
			"2q-1.0-0.0",
			"2q-compact-hybrid-1.0",
			"2q-fast-admission-compact-hybrid-1.0",
			"2q-fast-admission-reprieve-compact-hybrid-1.0",
			"2q-ghost-compact-hybrid-1.0",
			"2q-full-fast-admission-compact-hybrid-1.0-1.0",
		] {
			assert!(
				policy.parse::<PaperPolicy>().is_ok(),
				"{policy} should still parse: 2Q sizes no queue at (1 - k_in)",
			);
		}
	}
}
