/*
 * Copyright (c) Kia Shakiba
 *
 * This source code is licensed under the GNU AGPLv3 license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! Kept as a name for source compatibility: every hybrid design now shares
//! one stats shape, `crate::hybrid_stats::HybridStats`, filled from one set
//! of counters on `AtomicStatus`. The four-segment gauges the size-split
//! design adds are fields on that shared struct, zero for everyone else.

pub type S3FifoLazyDemotionFastAdmissionMidpointReprieveHybridStats = crate::hybrid_stats::HybridStats;
