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
	Lfu,
	Fifo,
	Clock,
	Sieve,
	Lru,
	Mru,
	TwoQ(f64, f64),
	Arc,
	SThreeFifo(f64),
	LruHybrid,
	LfuHybrid,

	/// `LfuHybrid`'s policy over a slab-backed frequency chain -- same
	/// algorithm, one structure instead of three. See
	/// `LfuCompactHybridStack`.
	LruCompactHybrid,
	LfuCompactHybrid,
	TwoQHybrid(f64),
	TwoQFastAdmissionHybrid(f64),
	TwoQFastAdmissionReprieveHybrid(f64),
	/// The full (three-queue) 2Q with fast-tier admission -- the only
	/// hybrid design whose queue algorithm matches [`PaperPolicy::TwoQ`]'s,
	/// and the only hybrid carrying TWO parameters: `k_in` sizes the
	/// fast-tier probation FIFO and `k_out` sizes the slow-tier `a1_out`
	/// overflow FIFO, which holds real resident objects rather than ghosts.
	/// `k_out` is a live parameter here; `TwoQ` writes its equivalent and
	/// never reads it. See
	/// `worker::policy::policy_stack::two_q_full_fast_admission_hybrid_stack`.
	TwoQFullFastAdmissionHybrid(f64, f64),
	FifoHybrid,
	LruSizedHybrid,
	/// Recency (LRU) in the fast tier, frequency (LFU) in the slow tier.
	/// The parameter is `promote_k`: how many accesses a slow-tier object
	/// must accumulate to earn promotion into the fast tier. Carried in the
	/// policy string (like `TwoQHybrid`'s `k_in`) rather than being runtime-
	/// configurable, because it is a policy parameter, not a size.
	LruLfuHybrid(u16),
	S3FifoHybrid(f64),
	TwoQGhostHybrid(f64),
	S3FifoGhostHybrid(f64),
	S3FifoGhostLazyDemotionHybrid(f64),
	S3FifoGhostLazyDemotionFastAdmissionHybrid(f64),
	S3FifoGhostLazyDemotionFastAdmissionMidpointHybrid(f64),
	S3FifoLazyDemotionFastAdmissionMidpointReprieveHybrid(f64),
	S3FifoLazyDemotionFastAdmissionReprieveHybrid(f64),
	S3FifoLazyDemotionReprieveHybrid(f64),
	S3FifoLazyDemotionFastAdmissionSplitSlowReprieveHybrid(f64),
}

impl PaperPolicy {
	/// Whether this policy is one of the tiered (hybrid) designs.
	#[must_use]
	pub fn is_hybrid(&self) -> bool {
		matches!(self, PaperPolicy::FifoHybrid { .. } | PaperPolicy::LfuHybrid { .. } | PaperPolicy::LfuCompactHybrid { .. } | PaperPolicy::LruCompactHybrid { .. } | PaperPolicy::LruHybrid { .. } | PaperPolicy::LruLfuHybrid { .. } | PaperPolicy::LruSizedHybrid { .. } | PaperPolicy::S3FifoGhostHybrid { .. } | PaperPolicy::S3FifoGhostLazyDemotionFastAdmissionHybrid { .. } | PaperPolicy::S3FifoGhostLazyDemotionFastAdmissionMidpointHybrid { .. } | PaperPolicy::S3FifoGhostLazyDemotionHybrid { .. } | PaperPolicy::S3FifoHybrid { .. } | PaperPolicy::S3FifoLazyDemotionFastAdmissionMidpointReprieveHybrid { .. } | PaperPolicy::S3FifoLazyDemotionFastAdmissionReprieveHybrid { .. } | PaperPolicy::S3FifoLazyDemotionFastAdmissionSplitSlowReprieveHybrid { .. } | PaperPolicy::S3FifoLazyDemotionReprieveHybrid { .. } | PaperPolicy::TwoQFastAdmissionHybrid { .. } | PaperPolicy::TwoQFastAdmissionReprieveHybrid { .. } | PaperPolicy::TwoQFullFastAdmissionHybrid { .. } | PaperPolicy::TwoQGhostHybrid { .. } | PaperPolicy::TwoQHybrid { .. })
	}

	pub fn is_auto(&self) -> bool {
		matches!(self, PaperPolicy::Auto)
	}
}

impl Display for PaperPolicy {
	fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
		match self {
			PaperPolicy::Auto => write!(f, "auto"),
			PaperPolicy::Lfu => write!(f, "lfu"),
			PaperPolicy::Fifo => write!(f, "fifo"),
			PaperPolicy::Clock => write!(f, "clock"),
			PaperPolicy::Sieve => write!(f, "sieve"),
			PaperPolicy::Lru => write!(f, "lru"),
			PaperPolicy::Mru => write!(f, "mru"),
			PaperPolicy::TwoQ(k_in, k_out) => write!(f, "2q-{k_in}-{k_out}"),
			PaperPolicy::Arc => write!(f, "arc"),
			PaperPolicy::SThreeFifo(ratio) => write!(f, "s3-fifo-{ratio}"),
			PaperPolicy::LruHybrid => write!(f, "lru-hybrid"),
			PaperPolicy::LfuHybrid => write!(f, "lfu-hybrid"),
			PaperPolicy::TwoQHybrid(k_in) => write!(f, "2q-hybrid-{k_in}"),
			PaperPolicy::TwoQFastAdmissionHybrid(k_in) => write!(f, "2q-fast-admission-hybrid-{k_in}"),
			PaperPolicy::TwoQFastAdmissionReprieveHybrid(k_in) => write!(f, "2q-fast-admission-reprieve-hybrid-{k_in}"),
			PaperPolicy::TwoQFullFastAdmissionHybrid(k_in, k_out) => write!(f, "2q-full-fast-admission-hybrid-{k_in}-{k_out}"),
			PaperPolicy::FifoHybrid => write!(f, "fifo-hybrid"),
			PaperPolicy::LruSizedHybrid => write!(f, "lru-sized-hybrid"),
			PaperPolicy::LruCompactHybrid => write!(f, "lru-compact-hybrid"),
			PaperPolicy::LfuCompactHybrid => write!(f, "lfu-compact-hybrid"),
			PaperPolicy::LruLfuHybrid(promote_k) => write!(f, "lru-lfu-hybrid-{promote_k}"),
			PaperPolicy::S3FifoHybrid(ratio) => write!(f, "s3-fifo-hybrid-{ratio}"),
			PaperPolicy::TwoQGhostHybrid(k_in) => write!(f, "2q-ghost-hybrid-{k_in}"),
			PaperPolicy::S3FifoGhostHybrid(ratio) => write!(f, "s3-fifo-ghost-hybrid-{ratio}"),
			PaperPolicy::S3FifoGhostLazyDemotionHybrid(ratio) => write!(f, "s3-fifo-ghost-lazy-demotion-hybrid-{ratio}"),
			PaperPolicy::S3FifoGhostLazyDemotionFastAdmissionHybrid(ratio) => write!(f, "s3-fifo-ghost-lazy-demotion-fast-admission-hybrid-{ratio}"),
			PaperPolicy::S3FifoGhostLazyDemotionFastAdmissionMidpointHybrid(ratio) => write!(f, "s3-fifo-ghost-lazy-demotion-fast-admission-midpoint-hybrid-{ratio}"),
			PaperPolicy::S3FifoLazyDemotionFastAdmissionMidpointReprieveHybrid(ratio) => write!(f, "s3-fifo-lazy-demotion-fast-admission-midpoint-reprieve-hybrid-{ratio}"),
			PaperPolicy::S3FifoLazyDemotionFastAdmissionReprieveHybrid(ratio) => write!(f, "s3-fifo-lazy-demotion-fast-admission-reprieve-hybrid-{ratio}"),
			PaperPolicy::S3FifoLazyDemotionReprieveHybrid(ratio) => write!(f, "s3-fifo-lazy-demotion-reprieve-hybrid-{ratio}"),
			PaperPolicy::S3FifoLazyDemotionFastAdmissionSplitSlowReprieveHybrid(ratio) => write!(f, "s3-fifo-lazy-demotion-fast-admission-split-slow-reprieve-hybrid-{ratio}"),
		}
	}
}

impl FromStr for PaperPolicy {
	type Err = CacheError;

	fn from_str(value: &str) -> Result<Self, Self::Err> {
		let policy = match value {
			"auto" => PaperPolicy::Auto,
			"lfu" => PaperPolicy::Lfu,
			"fifo" => PaperPolicy::Fifo,
			"clock" => PaperPolicy::Clock,
			"sieve" => PaperPolicy::Sieve,
			"lru" => PaperPolicy::Lru,
			"mru" => PaperPolicy::Mru,
			// Order matters and is load-bearing: every guard below also starts
			// with a prefix of the ones above it ("2q-fast-admission-hybrid-"
			// starts with "2q-", and so does "2q-hybrid-"), so the most
			// specific prefix has to be tested first or a more general guard
			// silently swallows it. See
			// `hybrid_does_not_collide_with_other_2q_forms`.
			value if value.starts_with("2q-full-fast-admission-hybrid-") => parse_two_q_full_fast_admission_hybrid(value)?,
			value if value.starts_with("2q-fast-admission-reprieve-hybrid-") => parse_two_q_fast_admission_reprieve_hybrid(value)?,
			value if value.starts_with("2q-fast-admission-hybrid-") => parse_two_q_fast_admission_hybrid(value)?,
			value if value.starts_with("2q-ghost-hybrid-") => parse_two_q_ghost_hybrid(value)?,
			value if value.starts_with("2q-hybrid-") => parse_two_q_hybrid(value)?,
			value if value.starts_with("2q-") => parse_two_q(value)?,
			"arc" => PaperPolicy::Arc,
			value if value.starts_with("s3-fifo-ghost-lazy-demotion-fast-admission-midpoint-hybrid-") => parse_s_three_fifo_ghost_lazy_demotion_fast_admission_midpoint_hybrid(value)?,
			value if value.starts_with("s3-fifo-ghost-lazy-demotion-fast-admission-hybrid-") => parse_s_three_fifo_ghost_lazy_demotion_fast_admission_hybrid(value)?,
			value if value.starts_with("s3-fifo-ghost-lazy-demotion-hybrid-") => parse_s_three_fifo_ghost_lazy_demotion_hybrid(value)?,
			value if value.starts_with("s3-fifo-ghost-hybrid-") => parse_s_three_fifo_ghost_hybrid(value)?,
			value if value.starts_with("s3-fifo-hybrid-") => parse_s_three_fifo_hybrid(value)?,
			value if value.starts_with("s3-fifo-lazy-demotion-fast-admission-midpoint-reprieve-hybrid-") => parse_s_three_fifo_lazy_demotion_fast_admission_midpoint_reprieve_hybrid(value)?,
			value if value.starts_with("s3-fifo-lazy-demotion-fast-admission-reprieve-hybrid-") => parse_s_three_fifo_lazy_demotion_fast_admission_reprieve_hybrid(value)?,
			value if value.starts_with("s3-fifo-lazy-demotion-reprieve-hybrid-") => parse_s_three_fifo_lazy_demotion_reprieve_hybrid(value)?,
			value if value.starts_with("s3-fifo-lazy-demotion-fast-admission-split-slow-reprieve-hybrid-") => parse_s_three_fifo_lazy_demotion_fast_admission_split_slow_reprieve_hybrid(value)?,
			value if value.starts_with("s3-fifo-") => parse_s_three_fifo(value)?,
			// Prefix guard, so it must be tested before any *exact* arm it
			// could be confused with is irrelevant (exact arms cannot swallow a
			// longer string) -- but it does have to precede nothing else here,
			// since no other guard starts with "lru-lfu-hybrid-". Kept beside
			// the other lru forms for readability. See
			// `hybrid_does_not_collide_with_other_lru_forms`.
			value if value.starts_with("lru-lfu-hybrid-") => parse_lru_lfu_hybrid(value)?,
			"lru-hybrid" => PaperPolicy::LruHybrid,
			"lfu-hybrid" => PaperPolicy::LfuHybrid,
			"lru-compact-hybrid" => PaperPolicy::LruCompactHybrid,
			"lfu-compact-hybrid" => PaperPolicy::LfuCompactHybrid,
			"fifo-hybrid" => PaperPolicy::FifoHybrid,
			"lru-sized-hybrid" => PaperPolicy::LruSizedHybrid,

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

fn parse_lru_lfu_hybrid(value: &str) -> Result<PaperPolicy, CacheError> {
	// skip the "lru-lfu-hybrid-"
	let tokens = value[15..]
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

	Ok(PaperPolicy::LruLfuHybrid(promote_k))
}

fn parse_two_q_hybrid(value: &str) -> Result<PaperPolicy, CacheError> {
	// skip the "2q-hybrid-"
	let tokens = value[10..]
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

	Ok(PaperPolicy::TwoQHybrid(k_in))
}

fn parse_two_q_fast_admission_hybrid(value: &str) -> Result<PaperPolicy, CacheError> {
	// skip the "2q-fast-admission-hybrid-"
	let tokens = value[25..]
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

	Ok(PaperPolicy::TwoQFastAdmissionHybrid(k_in))
}

fn parse_two_q_fast_admission_reprieve_hybrid(value: &str) -> Result<PaperPolicy, CacheError> {
	// skip the "2q-fast-admission-reprieve-hybrid-"
	let tokens = value[34..]
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

	Ok(PaperPolicy::TwoQFastAdmissionReprieveHybrid(k_in))
}

/// The only two-token hybrid parser. Modelled on [`parse_two_q`] rather
/// than on the one-token hybrid parsers: `k_in` sizes the fast-tier
/// probation FIFO, `k_out` the slow-tier `a1_out` overflow FIFO.
///
/// Unlike `parse_two_q` there is no `k_in + k_out <= 1.0` constraint: the
/// two budgets are denominated against different physical tiers here
/// (`k_in` against DRAM, `k_out` against PMEM), so their sum is not a
/// fraction of any one thing.
fn parse_two_q_full_fast_admission_hybrid(value: &str) -> Result<PaperPolicy, CacheError> {
	// skip the "2q-full-fast-admission-hybrid-"
	let tokens = value[30..]
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

	Ok(PaperPolicy::TwoQFullFastAdmissionHybrid(k_in, k_out))
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

fn parse_s_three_fifo_hybrid(value: &str) -> Result<PaperPolicy, CacheError> {
	// skip the "s3-fifo-hybrid-"
	let tokens = value[15..]
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

	Ok(PaperPolicy::S3FifoHybrid(ratio))
}

fn parse_two_q_ghost_hybrid(value: &str) -> Result<PaperPolicy, CacheError> {
	// skip the "2q-ghost-hybrid-"
	let tokens = value[16..]
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

	Ok(PaperPolicy::TwoQGhostHybrid(k_in))
}

fn parse_s_three_fifo_ghost_hybrid(value: &str) -> Result<PaperPolicy, CacheError> {
	// skip the "s3-fifo-ghost-hybrid-"
	let tokens = value[21..]
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

	Ok(PaperPolicy::S3FifoGhostHybrid(ratio))
}

fn parse_s_three_fifo_ghost_lazy_demotion_hybrid(value: &str) -> Result<PaperPolicy, CacheError> {
	// skip the "s3-fifo-ghost-lazy-demotion-hybrid-"
	let tokens = value[35..]
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

	Ok(PaperPolicy::S3FifoGhostLazyDemotionHybrid(ratio))
}

fn parse_s_three_fifo_ghost_lazy_demotion_fast_admission_hybrid(value: &str) -> Result<PaperPolicy, CacheError> {
	// skip the "s3-fifo-ghost-lazy-demotion-fast-admission-hybrid-"
	let tokens = value[50..]
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

	Ok(PaperPolicy::S3FifoGhostLazyDemotionFastAdmissionHybrid(ratio))
}

fn parse_s_three_fifo_ghost_lazy_demotion_fast_admission_midpoint_hybrid(value: &str) -> Result<PaperPolicy, CacheError> {
	// skip the "s3-fifo-ghost-lazy-demotion-fast-admission-midpoint-hybrid-"
	let tokens = value[59..]
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

	Ok(PaperPolicy::S3FifoGhostLazyDemotionFastAdmissionMidpointHybrid(ratio))
}

fn parse_s_three_fifo_lazy_demotion_fast_admission_midpoint_reprieve_hybrid(value: &str) -> Result<PaperPolicy, CacheError> {
	// skip the "s3-fifo-lazy-demotion-fast-admission-midpoint-reprieve-hybrid-"
	let tokens = value[62..]
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

	Ok(PaperPolicy::S3FifoLazyDemotionFastAdmissionMidpointReprieveHybrid(ratio))
}

fn parse_s_three_fifo_lazy_demotion_fast_admission_reprieve_hybrid(value: &str) -> Result<PaperPolicy, CacheError> {
	// skip the "s3-fifo-lazy-demotion-fast-admission-reprieve-hybrid-"
	let tokens = value[53..]
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

	Ok(PaperPolicy::S3FifoLazyDemotionFastAdmissionReprieveHybrid(ratio))
}

fn parse_s_three_fifo_lazy_demotion_reprieve_hybrid(value: &str) -> Result<PaperPolicy, CacheError> {
	// skip the "s3-fifo-lazy-demotion-reprieve-hybrid-"
	let tokens = value[38..]
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

	Ok(PaperPolicy::S3FifoLazyDemotionReprieveHybrid(ratio))
}

fn parse_s_three_fifo_lazy_demotion_fast_admission_split_slow_reprieve_hybrid(value: &str) -> Result<PaperPolicy, CacheError> {
	// skip the "s3-fifo-lazy-demotion-fast-admission-split-slow-reprieve-hybrid-"
	let tokens = value[64..]
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

	Ok(PaperPolicy::S3FifoLazyDemotionFastAdmissionSplitSlowReprieveHybrid(ratio))
}

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn lru_hybrid_round_trips_through_display_and_from_str() {
		assert_eq!(PaperPolicy::LruHybrid.to_string(), "lru-hybrid");
		assert_eq!("lru-hybrid".parse::<PaperPolicy>(), Ok(PaperPolicy::LruHybrid));
	}

	#[test]
	fn hybrid_does_not_collide_with_plain_lru() {
		assert_eq!("lru".parse::<PaperPolicy>(), Ok(PaperPolicy::Lru));
		assert_ne!(
			"lru".parse::<PaperPolicy>().unwrap(),
			"lru-hybrid".parse::<PaperPolicy>().unwrap(),
		);
	}

	#[test]
	fn lfu_hybrid_round_trips_through_display_and_from_str() {
		assert_eq!(PaperPolicy::LfuHybrid.to_string(), "lfu-hybrid");
		assert_eq!("lfu-hybrid".parse::<PaperPolicy>(), Ok(PaperPolicy::LfuHybrid));
	}

	#[test]
	fn hybrid_does_not_collide_with_plain_lfu() {
		assert_eq!("lfu".parse::<PaperPolicy>(), Ok(PaperPolicy::Lfu));
		assert_ne!(
			"lfu".parse::<PaperPolicy>().unwrap(),
			"lfu-hybrid".parse::<PaperPolicy>().unwrap(),
		);
	}

	#[test]
	fn two_q_hybrid_round_trips_through_display_and_from_str() {
		assert_eq!(PaperPolicy::TwoQHybrid(0.2).to_string(), "2q-hybrid-0.2");
		assert_eq!("2q-hybrid-0.2".parse::<PaperPolicy>(), Ok(PaperPolicy::TwoQHybrid(0.2)));
	}

	#[test]
	fn hybrid_does_not_collide_with_parameterized_2q() {
		assert_eq!("2q-0.2-0.2".parse::<PaperPolicy>(), Ok(PaperPolicy::TwoQ(0.2, 0.2)));
		assert_eq!("2q-hybrid-0.2".parse::<PaperPolicy>(), Ok(PaperPolicy::TwoQHybrid(0.2)));
		assert_ne!(
			"2q-0.2-0.2".parse::<PaperPolicy>().unwrap(),
			"2q-hybrid-0.2".parse::<PaperPolicy>().unwrap(),
		);
	}

	#[test]
	fn fifo_hybrid_rejects_out_of_range_ratio() {
		assert_eq!("2q-hybrid-1.5".parse::<PaperPolicy>(), Err(CacheError::InvalidPolicy));
	}

	#[test]
	fn fifo_hybrid_round_trips_through_display_and_from_str() {
		assert_eq!(PaperPolicy::FifoHybrid.to_string(), "fifo-hybrid");
		assert_eq!("fifo-hybrid".parse::<PaperPolicy>(), Ok(PaperPolicy::FifoHybrid));
	}

	#[test]
	fn hybrid_does_not_collide_with_plain_fifo() {
		assert_eq!("fifo".parse::<PaperPolicy>(), Ok(PaperPolicy::Fifo));
		assert_ne!(
			"fifo".parse::<PaperPolicy>().unwrap(),
			"fifo-hybrid".parse::<PaperPolicy>().unwrap(),
		);
	}

	#[test]
	fn lru_lfu_hybrid_round_trips_through_display_and_from_str() {
		assert_eq!(PaperPolicy::LruLfuHybrid(3).to_string(), "lru-lfu-hybrid-3");
		assert_eq!(
			"lru-lfu-hybrid-3".parse::<PaperPolicy>(),
			Ok(PaperPolicy::LruLfuHybrid(3)),
		);
	}

	#[test]
	fn hybrid_does_not_collide_with_other_lru_forms() {
		// "lru-lfu-hybrid-3" is matched by a `starts_with` guard while
		// "lru"/"lru-hybrid"/"lru-sized-hybrid" are exact arms, so they
		// cannot swallow it -- but a future guard added as
		// `starts_with("lru-")` could, which is what this pins down.
		assert_eq!("lru".parse::<PaperPolicy>(), Ok(PaperPolicy::Lru));
		assert_eq!("lru-hybrid".parse::<PaperPolicy>(), Ok(PaperPolicy::LruHybrid));
		assert_eq!("lru-sized-hybrid".parse::<PaperPolicy>(), Ok(PaperPolicy::LruSizedHybrid));
		assert_eq!(
			"lru-lfu-hybrid-4".parse::<PaperPolicy>(),
			Ok(PaperPolicy::LruLfuHybrid(4)),
		);
	}

	#[test]
	fn hybrid_rejects_malformed_and_zero_thresholds() {
		// 0 would make every slow object promotable before it was ever
		// accessed, which is not the same policy at any threshold.
		assert!("lru-lfu-hybrid-0".parse::<PaperPolicy>().is_err());
		assert!("lru-lfu-hybrid-".parse::<PaperPolicy>().is_err());
		assert!("lru-lfu-hybrid-abc".parse::<PaperPolicy>().is_err());
		assert!("lru-lfu-hybrid-1-2".parse::<PaperPolicy>().is_err());
		assert!("lru-lfu-hybrid".parse::<PaperPolicy>().is_err());
	}

	#[test]
	fn lru_sized_hybrid_round_trips_through_display_and_from_str() {
		assert_eq!(PaperPolicy::LruSizedHybrid.to_string(), "lru-sized-hybrid");
		assert_eq!("lru-sized-hybrid".parse::<PaperPolicy>(), Ok(PaperPolicy::LruSizedHybrid));
	}

	#[test]
	fn hybrid_does_not_collide_with_lru_hybrid() {
		assert_eq!("lru-hybrid".parse::<PaperPolicy>(), Ok(PaperPolicy::LruHybrid));
		assert_ne!(
			"lru-hybrid".parse::<PaperPolicy>().unwrap(),
			"lru-sized-hybrid".parse::<PaperPolicy>().unwrap(),
		);
	}

	#[test]
	fn two_q_fast_admission_hybrid_round_trips_through_display_and_from_str() {
		assert_eq!(PaperPolicy::TwoQFastAdmissionHybrid(0.2).to_string(), "2q-fast-admission-hybrid-0.2");

		assert_eq!(
			"2q-fast-admission-hybrid-0.2".parse::<PaperPolicy>(),
			Ok(PaperPolicy::TwoQFastAdmissionHybrid(0.2)),
		);
	}

	/// Locks in `FromStr`'s guard ordering. Every one of these strings also
	/// starts with `"2q-"`, and two of them also start with `"2q-hybrid-"`'s
	/// stem, so a less specific guard placed first would silently swallow the
	/// more specific form and parse it as the wrong policy.
	#[test]
	fn hybrid_does_not_collide_with_other_2q_forms() {
		assert_eq!("2q-0.2-0.2".parse::<PaperPolicy>(), Ok(PaperPolicy::TwoQ(0.2, 0.2)));
		assert_eq!("2q-hybrid-0.2".parse::<PaperPolicy>(), Ok(PaperPolicy::TwoQHybrid(0.2)));
		assert_eq!("2q-ghost-hybrid-0.2".parse::<PaperPolicy>(), Ok(PaperPolicy::TwoQGhostHybrid(0.2)));

		assert_eq!(
			"2q-fast-admission-hybrid-0.2".parse::<PaperPolicy>(),
			Ok(PaperPolicy::TwoQFastAdmissionHybrid(0.2)),
		);

		// The longest form of all, and the only two-token one. It shares
		// nothing with the others past "2q-f", but it is tested first in
		// `FromStr` and pinned here so a future, shorter guard cannot
		// swallow it.
		assert_eq!(
			"2q-full-fast-admission-hybrid-0.2-0.5".parse::<PaperPolicy>(),
			Ok(PaperPolicy::TwoQFullFastAdmissionHybrid(0.2, 0.5)),
		);

		assert_ne!(
			"2q-hybrid-0.2".parse::<PaperPolicy>().unwrap(),
			"2q-fast-admission-hybrid-0.2".parse::<PaperPolicy>().unwrap(),
		);

		assert_ne!(
			"2q-fast-admission-hybrid-0.2".parse::<PaperPolicy>().unwrap(),
			"2q-full-fast-admission-hybrid-0.2-0.5".parse::<PaperPolicy>().unwrap(),
		);
	}

	/// The only TWO-parameter policy string in the tree, hybrid or not.
	#[test]
	fn two_q_full_fast_admission_hybrid_round_trips_through_display_and_from_str() {
		assert_eq!(
			PaperPolicy::TwoQFullFastAdmissionHybrid(0.25, 0.5).to_string(),
			"2q-full-fast-admission-hybrid-0.25-0.5",
		);

		assert_eq!(
			"2q-full-fast-admission-hybrid-0.25-0.5".parse::<PaperPolicy>(),
			Ok(PaperPolicy::TwoQFullFastAdmissionHybrid(0.25, 0.5)),
		);
	}

	/// Two ratios means two range checks. A parser that validated only
	/// `k_in` -- the shape every other hybrid parser here has -- would pass
	/// the second of these.
	#[test]
	fn two_q_full_fast_admission_hybrid_rejects_out_of_range_ratios() {
		assert_eq!(
			"2q-full-fast-admission-hybrid-1.5-0.5".parse::<PaperPolicy>(),
			Err(CacheError::InvalidPolicy),
		);

		assert_eq!(
			"2q-full-fast-admission-hybrid-0.25-1.5".parse::<PaperPolicy>(),
			Err(CacheError::InvalidPolicy),
		);
	}

	/// Exactly two tokens, no more and no fewer -- the failure mode of
	/// having copied a one-token parser and only changed the byte offset.
	/// The double-dash cases are how a negative ratio arrives: the format
	/// is dash-separated, so it lands here as a token-count error rather
	/// than a range error. Either way it must not parse.
	#[test]
	fn two_q_full_fast_admission_hybrid_rejects_the_wrong_token_count() {
		assert!("2q-full-fast-admission-hybrid-0.25".parse::<PaperPolicy>().is_err());
		assert!("2q-full-fast-admission-hybrid-0.25-0.5-0.75".parse::<PaperPolicy>().is_err());
		assert!("2q-full-fast-admission-hybrid-".parse::<PaperPolicy>().is_err());
		assert!("2q-full-fast-admission-hybrid".parse::<PaperPolicy>().is_err());
		assert!("2q-full-fast-admission-hybrid-abc-0.5".parse::<PaperPolicy>().is_err());
		assert!("2q-full-fast-admission-hybrid-0.25-abc".parse::<PaperPolicy>().is_err());
		assert!("2q-full-fast-admission-hybrid--0.1-0.5".parse::<PaperPolicy>().is_err());
		assert!("2q-full-fast-admission-hybrid-0.25--0.1".parse::<PaperPolicy>().is_err());
	}

	/// Unlike `parse_two_q`, the two ratios are budgets against different
	/// physical tiers (`k_in` DRAM, `k_out` PMEM), so their sum is not a
	/// fraction of anything and is deliberately NOT constrained to <= 1.
	#[test]
	fn two_q_full_fast_admission_hybrid_allows_ratios_summing_past_one() {
		assert_eq!(
			"2q-full-fast-admission-hybrid-0.8-0.8".parse::<PaperPolicy>(),
			Ok(PaperPolicy::TwoQFullFastAdmissionHybrid(0.8, 0.8)),
		);

		// ...whereas plain `2q-` still is.
		assert_eq!("2q-0.8-0.8".parse::<PaperPolicy>(), Err(CacheError::InvalidPolicy));
	}

	/// It is a hybrid, so `new_hybrid` must accept it rather than
	/// rejecting it with `InvalidPolicy` at the `is_hybrid()` gate.
	#[test]
	fn two_q_full_fast_admission_hybrid_is_reported_as_a_hybrid() {
		assert!(PaperPolicy::TwoQFullFastAdmissionHybrid(0.25, 0.5).is_hybrid());
		assert!(!PaperPolicy::TwoQ(0.25, 0.5).is_hybrid());
	}

	#[test]
	fn two_q_fast_admission_reprieve_hybrid_round_trips_through_display_and_from_str() {
		assert_eq!(
			PaperPolicy::TwoQFastAdmissionReprieveHybrid(0.2).to_string(),
			"2q-fast-admission-reprieve-hybrid-0.2",
		);

		assert_eq!(
			"2q-fast-admission-reprieve-hybrid-0.2".parse::<PaperPolicy>(),
			Ok(PaperPolicy::TwoQFastAdmissionReprieveHybrid(0.2)),
		);
	}

	/// Every 2Q form must stay distinguishable. These two in particular
	/// share the whole `"2q-fast-admission-"` stem, so a less specific guard
	/// placed first would swallow the reprieve variant.
	#[test]
	fn two_q_fast_admission_reprieve_does_not_collide_with_the_non_reprieve_form() {
		assert_eq!(
			"2q-fast-admission-hybrid-0.2".parse::<PaperPolicy>(),
			Ok(PaperPolicy::TwoQFastAdmissionHybrid(0.2)),
		);

		assert_eq!(
			"2q-fast-admission-reprieve-hybrid-0.2".parse::<PaperPolicy>(),
			Ok(PaperPolicy::TwoQFastAdmissionReprieveHybrid(0.2)),
		);

		assert_ne!(
			"2q-fast-admission-hybrid-0.2".parse::<PaperPolicy>().unwrap(),
			"2q-fast-admission-reprieve-hybrid-0.2".parse::<PaperPolicy>().unwrap(),
		);
	}

	#[test]
	fn hybrid_rejects_out_of_range_k_in() {
		assert_eq!(
			"2q-fast-admission-reprieve-hybrid-1.5".parse::<PaperPolicy>(),
			Err(CacheError::InvalidPolicy),
		);
	}

	#[test]
	fn s3_fifo_hybrid_rejects_out_of_range_k_in() {
		assert_eq!("2q-fast-admission-hybrid-1.5".parse::<PaperPolicy>(), Err(CacheError::InvalidPolicy));
	}

	#[test]
	fn s3_fifo_hybrid_round_trips_through_display_and_from_str() {
		assert_eq!(PaperPolicy::S3FifoHybrid(0.1).to_string(), "s3-fifo-hybrid-0.1");
		assert_eq!("s3-fifo-hybrid-0.1".parse::<PaperPolicy>(), Ok(PaperPolicy::S3FifoHybrid(0.1)));
	}

	#[test]
	fn hybrid_does_not_collide_with_parameterized_s3_fifo() {
		assert_eq!("s3-fifo-0.1".parse::<PaperPolicy>(), Ok(PaperPolicy::SThreeFifo(0.1)));
		assert_eq!("s3-fifo-hybrid-0.1".parse::<PaperPolicy>(), Ok(PaperPolicy::S3FifoHybrid(0.1)));
		assert_ne!(
			"s3-fifo-0.1".parse::<PaperPolicy>().unwrap(),
			"s3-fifo-hybrid-0.1".parse::<PaperPolicy>().unwrap(),
		);
	}

	#[test]
	fn two_q_ghost_hybrid_rejects_out_of_range_ratio() {
		assert_eq!("s3-fifo-hybrid-1.5".parse::<PaperPolicy>(), Err(CacheError::InvalidPolicy));
	}

	#[test]
	fn two_q_ghost_hybrid_round_trips_through_display_and_from_str() {
		assert_eq!(PaperPolicy::TwoQGhostHybrid(0.2).to_string(), "2q-ghost-hybrid-0.2");
		assert_eq!("2q-ghost-hybrid-0.2".parse::<PaperPolicy>(), Ok(PaperPolicy::TwoQGhostHybrid(0.2)));
	}

	#[test]
	fn hybrid_does_not_collide_with_2q_hybrid_or_parameterized_2q() {
		assert_eq!("2q-0.2-0.2".parse::<PaperPolicy>(), Ok(PaperPolicy::TwoQ(0.2, 0.2)));
		assert_eq!("2q-hybrid-0.2".parse::<PaperPolicy>(), Ok(PaperPolicy::TwoQHybrid(0.2)));
		assert_eq!("2q-ghost-hybrid-0.2".parse::<PaperPolicy>(), Ok(PaperPolicy::TwoQGhostHybrid(0.2)));
		assert_ne!(
			"2q-hybrid-0.2".parse::<PaperPolicy>().unwrap(),
			"2q-ghost-hybrid-0.2".parse::<PaperPolicy>().unwrap(),
		);
	}

	#[test]
	fn s3_fifo_ghost_hybrid_rejects_out_of_range_ratio() {
		assert_eq!("2q-ghost-hybrid-1.5".parse::<PaperPolicy>(), Err(CacheError::InvalidPolicy));
	}

	#[test]
	fn s3_fifo_ghost_hybrid_round_trips_through_display_and_from_str() {
		assert_eq!(PaperPolicy::S3FifoGhostHybrid(0.1).to_string(), "s3-fifo-ghost-hybrid-0.1");
		assert_eq!("s3-fifo-ghost-hybrid-0.1".parse::<PaperPolicy>(), Ok(PaperPolicy::S3FifoGhostHybrid(0.1)));
	}

	#[test]
	fn hybrid_does_not_collide_with_s3_fifo_hybrid_or_parameterized_s3_fifo() {
		assert_eq!("s3-fifo-0.1".parse::<PaperPolicy>(), Ok(PaperPolicy::SThreeFifo(0.1)));
		assert_eq!("s3-fifo-hybrid-0.1".parse::<PaperPolicy>(), Ok(PaperPolicy::S3FifoHybrid(0.1)));
		assert_eq!("s3-fifo-ghost-hybrid-0.1".parse::<PaperPolicy>(), Ok(PaperPolicy::S3FifoGhostHybrid(0.1)));
		assert_ne!(
			"s3-fifo-hybrid-0.1".parse::<PaperPolicy>().unwrap(),
			"s3-fifo-ghost-hybrid-0.1".parse::<PaperPolicy>().unwrap(),
		);
	}

	#[test]
	fn s3_fifo_ghost_lazy_demotion_hybrid_rejects_out_of_range_ratio() {
		assert_eq!("s3-fifo-ghost-hybrid-1.5".parse::<PaperPolicy>(), Err(CacheError::InvalidPolicy));
	}

	#[test]
	fn s3_fifo_ghost_lazy_demotion_hybrid_round_trips_through_display_and_from_str() {
		assert_eq!(
			PaperPolicy::S3FifoGhostLazyDemotionHybrid(0.1).to_string(),
			"s3-fifo-ghost-lazy-demotion-hybrid-0.1",
		);
		assert_eq!(
			"s3-fifo-ghost-lazy-demotion-hybrid-0.1".parse::<PaperPolicy>(),
			Ok(PaperPolicy::S3FifoGhostLazyDemotionHybrid(0.1)),
		);
	}

	#[test]
	fn hybrid_does_not_collide_with_s3_fifo_ghost_hybrid_or_others() {
		assert_eq!("s3-fifo-0.1".parse::<PaperPolicy>(), Ok(PaperPolicy::SThreeFifo(0.1)));
		assert_eq!("s3-fifo-hybrid-0.1".parse::<PaperPolicy>(), Ok(PaperPolicy::S3FifoHybrid(0.1)));
		assert_eq!("s3-fifo-ghost-hybrid-0.1".parse::<PaperPolicy>(), Ok(PaperPolicy::S3FifoGhostHybrid(0.1)));
		assert_eq!(
			"s3-fifo-ghost-lazy-demotion-hybrid-0.1".parse::<PaperPolicy>(),
			Ok(PaperPolicy::S3FifoGhostLazyDemotionHybrid(0.1)),
		);
		assert_ne!(
			"s3-fifo-ghost-hybrid-0.1".parse::<PaperPolicy>().unwrap(),
			"s3-fifo-ghost-lazy-demotion-hybrid-0.1".parse::<PaperPolicy>().unwrap(),
		);
	}

	#[test]
	fn s3_fifo_ghost_lazy_demotion_fast_admission_hybrid_rejects_out_of_range_ratio() {
		assert_eq!(
			"s3-fifo-ghost-lazy-demotion-hybrid-1.5".parse::<PaperPolicy>(),
			Err(CacheError::InvalidPolicy),
		);
	}

	#[test]
	fn s3_fifo_ghost_lazy_demotion_fast_admission_hybrid_round_trips_through_display_and_from_str() {
		assert_eq!(
			PaperPolicy::S3FifoGhostLazyDemotionFastAdmissionHybrid(0.1).to_string(),
			"s3-fifo-ghost-lazy-demotion-fast-admission-hybrid-0.1",
		);
		assert_eq!(
			"s3-fifo-ghost-lazy-demotion-fast-admission-hybrid-0.1".parse::<PaperPolicy>(),
			Ok(PaperPolicy::S3FifoGhostLazyDemotionFastAdmissionHybrid(0.1)),
		);
	}

	#[test]
	fn hybrid_does_not_collide_with_s3_fifo_ghost_lazy_demotion_hybrid_or_others() {
		assert_eq!("s3-fifo-0.1".parse::<PaperPolicy>(), Ok(PaperPolicy::SThreeFifo(0.1)));
		assert_eq!("s3-fifo-hybrid-0.1".parse::<PaperPolicy>(), Ok(PaperPolicy::S3FifoHybrid(0.1)));
		assert_eq!("s3-fifo-ghost-hybrid-0.1".parse::<PaperPolicy>(), Ok(PaperPolicy::S3FifoGhostHybrid(0.1)));
		assert_eq!(
			"s3-fifo-ghost-lazy-demotion-hybrid-0.1".parse::<PaperPolicy>(),
			Ok(PaperPolicy::S3FifoGhostLazyDemotionHybrid(0.1)),
		);
		assert_eq!(
			"s3-fifo-ghost-lazy-demotion-fast-admission-hybrid-0.1".parse::<PaperPolicy>(),
			Ok(PaperPolicy::S3FifoGhostLazyDemotionFastAdmissionHybrid(0.1)),
		);
		assert_ne!(
			"s3-fifo-ghost-lazy-demotion-hybrid-0.1".parse::<PaperPolicy>().unwrap(),
			"s3-fifo-ghost-lazy-demotion-fast-admission-hybrid-0.1".parse::<PaperPolicy>().unwrap(),
		);
	}

	#[test]
	fn s3_fifo_ghost_lazy_demotion_fast_admission_midpoint_hybrid_rejects_out_of_range_ratio() {
		assert_eq!(
			"s3-fifo-ghost-lazy-demotion-fast-admission-hybrid-1.5".parse::<PaperPolicy>(),
			Err(CacheError::InvalidPolicy),
		);
	}

	#[test]
	fn s3_fifo_ghost_lazy_demotion_fast_admission_midpoint_hybrid_round_trips_through_display_and_from_str() {
		assert_eq!(
			PaperPolicy::S3FifoGhostLazyDemotionFastAdmissionMidpointHybrid(0.1).to_string(),
			"s3-fifo-ghost-lazy-demotion-fast-admission-midpoint-hybrid-0.1",
		);
		assert_eq!(
			"s3-fifo-ghost-lazy-demotion-fast-admission-midpoint-hybrid-0.1".parse::<PaperPolicy>(),
			Ok(PaperPolicy::S3FifoGhostLazyDemotionFastAdmissionMidpointHybrid(0.1)),
		);
	}

	#[test]
	fn hybrid_does_not_collide_with_fast_admission_hybrid_or_others() {
		assert_eq!("s3-fifo-0.1".parse::<PaperPolicy>(), Ok(PaperPolicy::SThreeFifo(0.1)));
		assert_eq!("s3-fifo-hybrid-0.1".parse::<PaperPolicy>(), Ok(PaperPolicy::S3FifoHybrid(0.1)));
		assert_eq!(
			"s3-fifo-ghost-lazy-demotion-fast-admission-hybrid-0.1".parse::<PaperPolicy>(),
			Ok(PaperPolicy::S3FifoGhostLazyDemotionFastAdmissionHybrid(0.1)),
		);
		assert_eq!(
			"s3-fifo-ghost-lazy-demotion-fast-admission-midpoint-hybrid-0.1".parse::<PaperPolicy>(),
			Ok(PaperPolicy::S3FifoGhostLazyDemotionFastAdmissionMidpointHybrid(0.1)),
		);
		assert_ne!(
			"s3-fifo-ghost-lazy-demotion-fast-admission-hybrid-0.1".parse::<PaperPolicy>().unwrap(),
			"s3-fifo-ghost-lazy-demotion-fast-admission-midpoint-hybrid-0.1".parse::<PaperPolicy>().unwrap(),
		);
	}

	#[test]
	fn s3_fifo_lazy_demotion_fast_admission_midpoint_reprieve_hybrid_rejects_out_of_range_ratio() {
		assert_eq!(
			"s3-fifo-ghost-lazy-demotion-fast-admission-midpoint-hybrid-1.5".parse::<PaperPolicy>(),
			Err(CacheError::InvalidPolicy),
		);
	}

	#[test]
	fn s3_fifo_lazy_demotion_fast_admission_midpoint_reprieve_hybrid_round_trips_through_display_and_from_str() {
		assert_eq!(
			PaperPolicy::S3FifoLazyDemotionFastAdmissionMidpointReprieveHybrid(0.1).to_string(),
			"s3-fifo-lazy-demotion-fast-admission-midpoint-reprieve-hybrid-0.1",
		);
		assert_eq!(
			"s3-fifo-lazy-demotion-fast-admission-midpoint-reprieve-hybrid-0.1".parse::<PaperPolicy>(),
			Ok(PaperPolicy::S3FifoLazyDemotionFastAdmissionMidpointReprieveHybrid(0.1)),
		);
	}

	#[test]
	fn s3_fifo_lazy_demotion_fast_admission_midpoint_reprieve_hybrid_does_not_collide_with_others() {
		assert_eq!("s3-fifo-0.1".parse::<PaperPolicy>(), Ok(PaperPolicy::SThreeFifo(0.1)));
		assert_eq!("s3-fifo-hybrid-0.1".parse::<PaperPolicy>(), Ok(PaperPolicy::S3FifoHybrid(0.1)));
		assert_eq!(
			"s3-fifo-ghost-lazy-demotion-fast-admission-midpoint-hybrid-0.1".parse::<PaperPolicy>(),
			Ok(PaperPolicy::S3FifoGhostLazyDemotionFastAdmissionMidpointHybrid(0.1)),
		);
		assert_eq!(
			"s3-fifo-lazy-demotion-fast-admission-midpoint-reprieve-hybrid-0.1".parse::<PaperPolicy>(),
			Ok(PaperPolicy::S3FifoLazyDemotionFastAdmissionMidpointReprieveHybrid(0.1)),
		);
		assert_ne!(
			"s3-fifo-ghost-lazy-demotion-fast-admission-midpoint-hybrid-0.1".parse::<PaperPolicy>().unwrap(),
			"s3-fifo-lazy-demotion-fast-admission-midpoint-reprieve-hybrid-0.1".parse::<PaperPolicy>().unwrap(),
		);
	}

	#[test]
	fn s3_fifo_lazy_demotion_fast_admission_reprieve_hybrid_rejects_out_of_range_ratio() {
		assert_eq!(
			"s3-fifo-lazy-demotion-fast-admission-midpoint-reprieve-hybrid-1.5".parse::<PaperPolicy>(),
			Err(CacheError::InvalidPolicy),
		);
	}

	#[test]
	fn s3_fifo_lazy_demotion_fast_admission_reprieve_hybrid_round_trips_through_display_and_from_str() {
		assert_eq!(
			PaperPolicy::S3FifoLazyDemotionFastAdmissionReprieveHybrid(0.1).to_string(),
			"s3-fifo-lazy-demotion-fast-admission-reprieve-hybrid-0.1",
		);
		assert_eq!(
			"s3-fifo-lazy-demotion-fast-admission-reprieve-hybrid-0.1".parse::<PaperPolicy>(),
			Ok(PaperPolicy::S3FifoLazyDemotionFastAdmissionReprieveHybrid(0.1)),
		);
	}

	#[test]
	fn hybrid_does_not_collide_with_the_midpoint_variant() {
		assert_eq!("s3-fifo-0.1".parse::<PaperPolicy>(), Ok(PaperPolicy::SThreeFifo(0.1)));
		assert_eq!(
			"s3-fifo-lazy-demotion-fast-admission-midpoint-reprieve-hybrid-0.1".parse::<PaperPolicy>(),
			Ok(PaperPolicy::S3FifoLazyDemotionFastAdmissionMidpointReprieveHybrid(0.1)),
		);
		assert_eq!(
			"s3-fifo-lazy-demotion-fast-admission-reprieve-hybrid-0.1".parse::<PaperPolicy>(),
			Ok(PaperPolicy::S3FifoLazyDemotionFastAdmissionReprieveHybrid(0.1)),
		);
		assert_ne!(
			"s3-fifo-lazy-demotion-fast-admission-midpoint-reprieve-hybrid-0.1".parse::<PaperPolicy>().unwrap(),
			"s3-fifo-lazy-demotion-fast-admission-reprieve-hybrid-0.1".parse::<PaperPolicy>().unwrap(),
		);
	}

	#[test]
	fn split_slow_module_reprieve_prefix_rejects_out_of_range_ratio() {
		assert_eq!(
			"s3-fifo-lazy-demotion-fast-admission-reprieve-hybrid-1.5".parse::<PaperPolicy>(),
			Err(CacheError::InvalidPolicy),
		);
	}

	#[test]
	fn s3_fifo_lazy_demotion_fast_admission_split_slow_reprieve_hybrid_round_trips_through_display_and_from_str() {
		assert_eq!(
			PaperPolicy::S3FifoLazyDemotionFastAdmissionSplitSlowReprieveHybrid(0.1).to_string(),
			"s3-fifo-lazy-demotion-fast-admission-split-slow-reprieve-hybrid-0.1",
		);
		assert_eq!(
			"s3-fifo-lazy-demotion-fast-admission-split-slow-reprieve-hybrid-0.1".parse::<PaperPolicy>(),
			Ok(PaperPolicy::S3FifoLazyDemotionFastAdmissionSplitSlowReprieveHybrid(0.1)),
		);
	}

	#[test]
	fn s3_fifo_lazy_demotion_fast_admission_split_slow_reprieve_hybrid_does_not_collide_with_others() {
		assert_eq!("s3-fifo-0.1".parse::<PaperPolicy>(), Ok(PaperPolicy::SThreeFifo(0.1)));
		assert_eq!("s3-fifo-hybrid-0.1".parse::<PaperPolicy>(), Ok(PaperPolicy::S3FifoHybrid(0.1)));
		assert_eq!(
			"s3-fifo-lazy-demotion-fast-admission-midpoint-reprieve-hybrid-0.1".parse::<PaperPolicy>(),
			Ok(PaperPolicy::S3FifoLazyDemotionFastAdmissionMidpointReprieveHybrid(0.1)),
		);
		assert_eq!(
			"s3-fifo-lazy-demotion-fast-admission-split-slow-reprieve-hybrid-0.1".parse::<PaperPolicy>(),
			Ok(PaperPolicy::S3FifoLazyDemotionFastAdmissionSplitSlowReprieveHybrid(0.1)),
		);
		assert_ne!(
			"s3-fifo-lazy-demotion-fast-admission-midpoint-reprieve-hybrid-0.1".parse::<PaperPolicy>().unwrap(),
			"s3-fifo-lazy-demotion-fast-admission-split-slow-reprieve-hybrid-0.1".parse::<PaperPolicy>().unwrap(),
		);
	}

	#[test]
	fn s3_fifo_lazy_demotion_fast_admission_split_slow_reprieve_hybrid_rejects_out_of_range_ratio() {
		assert_eq!(
			"s3-fifo-lazy-demotion-fast-admission-split-slow-reprieve-hybrid-1.5".parse::<PaperPolicy>(),
			Err(CacheError::InvalidPolicy),
		);
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
		"s3-fifo-hybrid-",
		"s3-fifo-ghost-hybrid-",
		"s3-fifo-ghost-lazy-demotion-hybrid-",
		"s3-fifo-ghost-lazy-demotion-fast-admission-hybrid-",
		"s3-fifo-ghost-lazy-demotion-fast-admission-midpoint-hybrid-",
	];

	/// The four reprieve designs, which derive no budget from `1 - ratio` and
	/// so keep the INCLUSIVE bound. Their `evict_one` is purely the main
	/// queue's tail loop; the one-access queue is drained by
	/// `settle_one_access()` and never reaches eviction, so the
	/// `!main.is_full()` dispatch gate `main_capacity` serves is absent.
	#[cfg(test)]
	const S3_FIFO_REPRIEVE_PREFIXES: &[&str] = &[
		"s3-fifo-lazy-demotion-fast-admission-midpoint-reprieve-hybrid-",
		"s3-fifo-lazy-demotion-fast-admission-reprieve-hybrid-",
		"s3-fifo-lazy-demotion-reprieve-hybrid-",
		"s3-fifo-lazy-demotion-fast-admission-split-slow-reprieve-hybrid-",
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
			"2q-hybrid-1.0",
			"2q-fast-admission-hybrid-1.0",
			"2q-fast-admission-reprieve-hybrid-1.0",
			"2q-ghost-hybrid-1.0",
			"2q-full-fast-admission-hybrid-1.0-1.0",
		] {
			assert!(
				policy.parse::<PaperPolicy>().is_ok(),
				"{policy} should still parse: 2Q sizes no queue at (1 - k_in)",
			);
		}
	}

}
