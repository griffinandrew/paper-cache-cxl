#![allow(non_camel_case_types, non_snake_case, non_upper_case_globals, dead_code)]

// Entirely bindgen output -- generated from wrapper.h (which #includes the
// real UMF public headers) at build time. Not hand-maintained. Kept private
// (no `pub` on `mod ffi;` in lib.rs) so raw bindings never leak into this
// crate's public API.
include!(concat!(env!("OUT_DIR"), "/umf_bindings.rs"));
