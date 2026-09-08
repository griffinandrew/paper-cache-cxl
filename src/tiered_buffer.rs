/*
 * Copyright (c) Kia Shakiba
 *
 * This source code is licensed under the GNU AGPLv3 license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! `TieredBuffer` -- the SHAPE marker every hybrid-cache design is spelled
//! with. Once a value type; since the Arc rewrite, a zero-sized marker.
//!
//! ## What used to be here
//!
//! A two-variant enum, `Fast(Box<[u8]>) | Slow(Box<[u8], Hybrid>)`, boxed
//! behind a `Shared` refcount. Measured on the base commit with
//! `-Zprint-type-sizes`, that cost 64 bytes of DRAM per object before a single
//! value byte:
//!
//! ```text
//!   Shared<TieredBuffer> handle in the Object      8
//!   Shared inner: strong count                     8
//!   TieredBuffer discriminant (2 variants, 8 B!)   8
//!   Box<[u8]> fat-pointer length                   8
//!   the Inner allocation's own size class         32
//! ```
//!
//! `TieredValue` replaces all of it with one eight-byte word: the address of
//! the bytes, with the tier in bit 0. The discriminant becomes a tag bit that
//! is free because every value is 8-aligned; the fat-pointer length moves into
//! `Object::len`, which fits in padding the struct already had; and the strong
//! count and its allocation go away entirely, replaced by crossbeam-epoch
//! reclamation (see [`crate::value::defer_free`]).
//!
//! ## Why the name survives
//!
//! `PaperCache<K, TieredBuffer>` is the type every one of the ~20 hybrid design
//! modules names, re-exports, and documents. Keeping the alias makes the
//! representation change invisible to all of them, and to their tests, which is
//! why the enum could be deleted in one step instead of ~20.
//!
//! It is also load-bearing for the impl blocks: `TieredBuffer` is what tells
//! the hybrid `impl<K, S> PaperCache<K, TieredBuffer, S>` apart from the flat
//! `impl<K, V, S> ... where V: ValueShape`. See `ValueShape`'s documentation
//! for why those two must stay disjoint.

/// The hybrid designs' cache SHAPE. See the module documentation.
///
/// Zero-sized, and deliberately not a value type: since the value became a
/// refcounted header keyed by `K` ([`crate::value::TieredValue<K>`]), a shape
/// marker cannot be an alias for it without dragging `K` into every
/// `PaperCache<K, TieredBuffer, S>` spelling in ~20 design modules, their
/// tests, `paper-server` and `paper-benchmark-cxl`.
///
/// It deliberately does NOT implement [`crate::value::ValueShape`]. That is
/// what keeps the hybrid `impl<K, S> PaperCache<K, TieredBuffer, S>` block
/// disjoint from the flat `impl<K, V, S> ... where V: ValueShape` one -- both
/// are compiled together in any build with a hybrid feature, since every
/// hybrid feature enables `key_value_pmem`. A `ValueShape` impl here would
/// make them overlap and produce a duplicate-inherent-impl error on `get`,
/// `set`, `peek` and every other method. The hybrid admission tier is chosen
/// at runtime by `hybrid_policy::admission_tier`, not by a `TIER` constant,
/// which is why this marker needs no trait at all.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct TieredBuffer;
