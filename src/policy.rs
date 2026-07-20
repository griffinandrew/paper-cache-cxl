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
	TwoQHybrid(f64),
	FifoHybrid,
}

impl PaperPolicy {
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
			PaperPolicy::FifoHybrid => write!(f, "fifo-hybrid"),
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
			value if value.starts_with("2q-hybrid-") => parse_two_q_hybrid(value)?,
			value if value.starts_with("2q-") => parse_two_q(value)?,
			"arc" => PaperPolicy::Arc,
			value if value.starts_with("s3-fifo-") => parse_s_three_fifo(value)?,
			"lru-hybrid" => PaperPolicy::LruHybrid,
			"lfu-hybrid" => PaperPolicy::LfuHybrid,
			"fifo-hybrid" => PaperPolicy::FifoHybrid,

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

	if !(0.0..=1.0).contains(&ratio) {
		return Err(CacheError::InvalidPolicy);
	}

	Ok(PaperPolicy::SThreeFifo(ratio))
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
	fn lru_hybrid_does_not_collide_with_plain_lru() {
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
	fn lfu_hybrid_does_not_collide_with_plain_lfu() {
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
	fn two_q_hybrid_does_not_collide_with_parameterized_2q() {
		assert_eq!("2q-0.2-0.2".parse::<PaperPolicy>(), Ok(PaperPolicy::TwoQ(0.2, 0.2)));
		assert_eq!("2q-hybrid-0.2".parse::<PaperPolicy>(), Ok(PaperPolicy::TwoQHybrid(0.2)));
		assert_ne!(
			"2q-0.2-0.2".parse::<PaperPolicy>().unwrap(),
			"2q-hybrid-0.2".parse::<PaperPolicy>().unwrap(),
		);
	}

	#[test]
	fn two_q_hybrid_rejects_out_of_range_ratio() {
		assert_eq!("2q-hybrid-1.5".parse::<PaperPolicy>(), Err(CacheError::InvalidPolicy));
	}

	#[test]
	fn fifo_hybrid_round_trips_through_display_and_from_str() {
		assert_eq!(PaperPolicy::FifoHybrid.to_string(), "fifo-hybrid");
		assert_eq!("fifo-hybrid".parse::<PaperPolicy>(), Ok(PaperPolicy::FifoHybrid));
	}

	#[test]
	fn fifo_hybrid_does_not_collide_with_plain_fifo() {
		assert_eq!("fifo".parse::<PaperPolicy>(), Ok(PaperPolicy::Fifo));
		assert_ne!(
			"fifo".parse::<PaperPolicy>().unwrap(),
			"fifo-hybrid".parse::<PaperPolicy>().unwrap(),
		);
	}
}
