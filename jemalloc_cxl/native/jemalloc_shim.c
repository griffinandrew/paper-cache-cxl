/*
 * Tiny shim exposing jemalloc's MALLOCX_* flag macros as callable functions.
 *
 * jemalloc.h defines MALLOCX_ARENA/MALLOCX_ALIGN/MALLOCX_TCACHE(_NONE) as
 * preprocessor macros (some involving `ffs()` on the alignment value), not
 * exported symbols -- there is nothing to link against from Rust. Rather
 * than reimplementing that bit-twiddling in Rust (and risking it drifting
 * from jemalloc's own definition on a future upgrade), this shim includes
 * the exact jemalloc.h this crate is built against (located via
 * DEP_JEMALLOC_ROOT, see build.rs) and re-exposes the macros as plain
 * functions with a stable, unprefixed ABI that src/ffi.rs declares directly.
 *
 * This file does NOT call mallocx/dallocx/mallctl itself -- those are
 * declared directly in Rust (src/ffi.rs) against the real, prefixed
 * (`_rjem_`) symbols that tikv-jemalloc-sys links into the final binary,
 * so there is exactly one jemalloc instance in play, not two.
 */

#include <jemalloc/jemalloc.h>

int32_t jemalloc_cxl_shim_mallocx_arena(int32_t arena_ind) {
	return MALLOCX_ARENA((unsigned)arena_ind);
}

int32_t jemalloc_cxl_shim_mallocx_align(size_t alignment) {
	return MALLOCX_ALIGN(alignment);
}

int32_t jemalloc_cxl_shim_mallocx_tcache_none(void) {
	return MALLOCX_TCACHE_NONE;
}

int32_t jemalloc_cxl_shim_mallocx_tcache(int32_t tcache_ind) {
	return MALLOCX_TCACHE(tcache_ind);
}

int32_t jemalloc_cxl_shim_mallocx_zero(void) {
	return MALLOCX_ZERO;
}
