//! Custom jemalloc extent hooks: the mechanism that lets one arena's memory
//! be `mmap`'d and NUMA-bound differently from every other arena in the
//! same jemalloc instance.
//!
//! jemalloc calls back into these functions whenever it needs to grow,
//! shrink, or otherwise manage the raw virtual-memory "extents" backing a
//! specific arena. A single, static [`ExtentHooks`] value (`CXL_HOOKS`) is
//! shared by every CXL arena this crate ever creates; the hooks look up
//! which NUMA node/policy a given call's `arena_ind` maps to via
//! [`register_arena_numa`]'s registry, rather than each arena getting its
//! own hook closures (jemalloc's C ABI requires plain function pointers,
//! not closures, so per-arena state has to live in a side table keyed by
//! arena index instead of being captured directly).

use std::collections::HashMap;
use std::ffi::c_void;
use std::os::raw::c_uint;
use std::ptr;
use std::sync::{Mutex, OnceLock};

/// Which NUMA placement policy an arena's memory should be bound with.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NumaPolicy {
    /// `MPOL_BIND` -- allocations on the bound node(s) only. Strict: if the
    /// node is under memory pressure, allocation/page-fault can fail rather
    /// than silently falling back to another node.
    BindStrict,
    /// `MPOL_PREFERRED` -- prefer the given node, but the kernel is free to
    /// fall back to another node under pressure rather than failing.
    Preferred,
}

impl NumaPolicy {
    fn mpol_mode(self) -> libc::c_int {
        match self {
            NumaPolicy::BindStrict => libc::MPOL_BIND,
            NumaPolicy::Preferred => libc::MPOL_PREFERRED,
        }
    }
}

/// Per-arena NUMA target, recorded by [`crate::arena::create_cxl_arena`] and
/// consulted by the shared `alloc` hook on every extent it maps for that
/// arena.
#[derive(Debug, Clone, Copy)]
pub(crate) struct ArenaNumaConfig {
    pub node: u32,
    pub policy: NumaPolicy,
}

fn registry() -> &'static Mutex<HashMap<c_uint, ArenaNumaConfig>> {
    static REGISTRY: OnceLock<Mutex<HashMap<c_uint, ArenaNumaConfig>>> = OnceLock::new();
    REGISTRY.get_or_init(|| Mutex::new(HashMap::new()))
}

/// Registers the NUMA target for `arena_ind`, consulted by the shared
/// extent `alloc` hook for every future extent mapped into that arena.
/// Called once by [`crate::arena::create_cxl_arena`] right after the arena
/// is created (the arena index isn't known before that call returns).
pub(crate) fn register_arena_numa(arena_ind: c_uint, config: ArenaNumaConfig) {
    registry()
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
        .insert(arena_ind, config);
}

// ---------------------------------------------------------------------
// NUMA nodemask + mbind
// ---------------------------------------------------------------------

/// Linux's `mbind(2)` nodemask is a bitmap `maxnode` bits wide. 1024 bits
/// (16 `u64` words) comfortably covers every NUMA node id in practice
/// (Linux's own default `CONFIG_NODES_SHIFT` ceiling) while staying a small,
/// fixed-size stack array -- no need to query the live topology just to
/// size the mask.
const NODEMASK_BITS: usize = 1024;
const NODEMASK_WORDS: usize = NODEMASK_BITS / 64;

/// Builds an `mbind(2)` nodemask with exactly `node`'s bit set. Works for
/// any node id in `0..NODEMASK_BITS`, not just small/known ids -- the word
/// and bit position are both computed from `node`, never hardcoded.
fn build_nodemask(node: u32) -> Result<[u64; NODEMASK_WORDS], MbindError> {
    let word = (node as usize) / 64;
    let bit = (node as usize) % 64;

    if word >= NODEMASK_WORDS {
        return Err(MbindError::NodeOutOfRange { node });
    }

    let mut mask = [0u64; NODEMASK_WORDS];
    mask[word] = 1u64 << bit;
    Ok(mask)
}

#[derive(Debug, thiserror::Error)]
pub enum MbindError {
    #[error("NUMA node {node} exceeds this crate's supported nodemask range (0..{NODEMASK_BITS})")]
    NodeOutOfRange { node: u32 },

    #[error("mbind(2) failed: {0}")]
    MbindFailed(#[source] std::io::Error),
}

/// Binds the already-mapped `[addr, addr+len)` region to `node` under
/// `policy` via the raw `mbind(2)` syscall (not wrapped by the `libc` crate
/// directly -- NUMA policy calls aren't part of its portable surface, so
/// this goes through `libc::syscall` with the platform's `SYS_mbind`
/// number, which `libc` does expose).
///
/// Always OR's `MPOL_F_STATIC_NODES` into the *mode* argument (per this
/// crate's own design decision, documented in the README): without it,
/// node ids in the mask are relative to the calling thread's cpuset/cgroup
/// allowed-node list rather than absolute physical node ids, which would
/// silently rebind to the wrong node under a restrictive cgroup.
/// `MPOL_F_STATIC_NODES` makes the node id in `nodemask` always mean the
/// literal physical node number.
///
/// **Confirmed by direct testing against this environment's kernel**:
/// `MPOL_F_STATIC_NODES` is a *mode flag* -- it must be OR'd into `mbind`'s
/// third (`mode`) argument, not passed via its separate sixth (`flags`)
/// argument (that argument is for a different flag namespace entirely --
/// `MPOL_MF_STRICT`/`MPOL_MF_MOVE`/`MPOL_MF_MOVE_ALL`, which govern what
/// happens to *already-resident* pages, irrelevant here since every call
/// site is a freshly mapped, never-yet-faulted region). Passing it as the
/// `flags` argument instead compiles fine but fails at runtime with EINVAL
/// unconditionally, for both `MPOL_BIND` and `MPOL_PREFERRED` -- verified
/// with a standalone C reproduction against this same kernel before fixing
/// this function. The `flags` argument itself is passed as `0`.
///
/// Note: `mbind` sets the *policy* governing future page faults in this
/// range -- it does not itself force immediate physical placement. Pages
/// are placed on first fault (see `examples/cxl_vec.rs`'s "touch every
/// page" step, and the README's "mbind is a policy, not a placement
/// guarantee" section).
fn mbind(addr: *mut c_void, len: usize, node: u32, policy: NumaPolicy) -> Result<(), MbindError> {
    let mask = build_nodemask(node)?;
    let mode = policy.mpol_mode() | libc::MPOL_F_STATIC_NODES;

    // SAFETY: `addr`/`len` describe a region this process just mmap'd
    // (caller's responsibility, enforced by this function's only caller,
    // `cxl_extent_alloc`, immediately after a successful `mmap`); `mask` is
    // a valid, fully-initialized on-stack nodemask sized exactly to
    // `NODEMASK_BITS`, matching the `maxnode` argument passed alongside it.
    let ret = unsafe {
        libc::syscall(
            libc::SYS_mbind,
            addr,
            len as libc::c_ulong,
            mode,
            mask.as_ptr(),
            NODEMASK_BITS as libc::c_ulong,
            0 as libc::c_uint,
        )
    };

    if ret != 0 {
        return Err(MbindError::MbindFailed(std::io::Error::last_os_error()));
    }

    Ok(())
}

// ---------------------------------------------------------------------
// extent_hooks_t: a Rust-compatible mirror of jemalloc's C struct
// ---------------------------------------------------------------------

pub type ExtentAllocFn = unsafe extern "C" fn(
    extent_hooks: *mut ExtentHooks,
    new_addr: *mut c_void,
    size: usize,
    alignment: usize,
    zero: *mut bool,
    commit: *mut bool,
    arena_ind: c_uint,
) -> *mut c_void;

pub type ExtentDallocFn = unsafe extern "C" fn(
    extent_hooks: *mut ExtentHooks,
    addr: *mut c_void,
    size: usize,
    committed: bool,
    arena_ind: c_uint,
) -> bool;

pub type ExtentDestroyFn = unsafe extern "C" fn(
    extent_hooks: *mut ExtentHooks,
    addr: *mut c_void,
    size: usize,
    committed: bool,
    arena_ind: c_uint,
);

pub type ExtentCommitFn = unsafe extern "C" fn(
    extent_hooks: *mut ExtentHooks,
    addr: *mut c_void,
    size: usize,
    offset: usize,
    length: usize,
    arena_ind: c_uint,
) -> bool;

pub type ExtentDecommitFn = ExtentCommitFn;
pub type ExtentPurgeFn = ExtentCommitFn;

pub type ExtentSplitFn = unsafe extern "C" fn(
    extent_hooks: *mut ExtentHooks,
    addr: *mut c_void,
    size: usize,
    size_a: usize,
    size_b: usize,
    committed: bool,
    arena_ind: c_uint,
) -> bool;

pub type ExtentMergeFn = unsafe extern "C" fn(
    extent_hooks: *mut ExtentHooks,
    addr_a: *mut c_void,
    size_a: usize,
    addr_b: *mut c_void,
    size_b: usize,
    committed: bool,
    arena_ind: c_uint,
) -> bool;

/// Field order and function-pointer signatures mirror jemalloc.h's
/// `struct extent_hooks_s` exactly (verified directly against the
/// jemalloc.h this crate links against -- see build.rs). `#[repr(C)]` is
/// load-bearing: jemalloc reads this struct from C, with no knowledge of
/// Rust layout rules.
#[repr(C)]
pub struct ExtentHooks {
    pub alloc: Option<ExtentAllocFn>,
    pub dalloc: Option<ExtentDallocFn>,
    pub destroy: Option<ExtentDestroyFn>,
    pub commit: Option<ExtentCommitFn>,
    pub decommit: Option<ExtentDecommitFn>,
    pub purge_lazy: Option<ExtentPurgeFn>,
    pub purge_forced: Option<ExtentPurgeFn>,
    pub split: Option<ExtentSplitFn>,
    pub merge: Option<ExtentMergeFn>,
}

// SAFETY: contains only function pointers (`Option<fn(...)>`, all `'static`
// -- these are the module's own `extern "C" fn` items, never closures) --
// no interior mutability, no thread-affinity.
unsafe impl Send for ExtentHooks {}
unsafe impl Sync for ExtentHooks {}

/// The one, process-lifetime `ExtentHooks` value every CXL arena is created
/// with. Leaked deliberately (`Box::leak`, see [`crate::arena`]) --
/// jemalloc retains the pointer passed to `arenas.create` for as long as
/// the arena exists, which in this crate's process-lifetime-arena design
/// is "forever," so there is no sound place to free it from.
pub(crate) static CXL_HOOKS: ExtentHooks = ExtentHooks {
    alloc: Some(cxl_extent_alloc),
    dalloc: Some(cxl_extent_dalloc),
    destroy: Some(cxl_extent_destroy),
    commit: Some(cxl_extent_commit),
    decommit: Some(cxl_extent_decommit),
    purge_lazy: Some(cxl_extent_purge_lazy),
    purge_forced: Some(cxl_extent_purge_forced),
    split: Some(cxl_extent_split),
    merge: Some(cxl_extent_merge),
};

fn page_size() -> usize {
    unsafe extern "C" {
        fn getpagesize() -> libc::c_int;
    }
    // SAFETY: `getpagesize` takes no arguments and has no failure mode --
    // it's a thin, universally-available libc wrapper around the kernel's
    // fixed page size for this process.
    unsafe { getpagesize() as usize }
}

/// Maps `size` bytes, anonymous and private, aligned to `alignment`.
///
/// For `alignment <= page_size` this is a single `mmap` (every mmap
/// returned address is already page-aligned, so nothing further is
/// needed). For larger alignments, over-maps `size + alignment` bytes,
/// then trims the unaligned prefix and trailing suffix back down to
/// exactly `size` bytes at an aligned address -- the standard technique
/// for aligned anonymous mappings (mmap itself has no alignment
/// parameter).
fn map_aligned(size: usize, alignment: usize) -> Option<ptr::NonNull<u8>> {
    let page = page_size();

    // SAFETY: standard anonymous/private mapping request; `addr` hint is
    // null (kernel chooses the address), fd/offset are the conventional
    // zero values MAP_ANONYMOUS requires.
    let raw_map = |len: usize| unsafe {
        libc::mmap(
            ptr::null_mut(),
            len,
            libc::PROT_READ | libc::PROT_WRITE,
            libc::MAP_PRIVATE | libc::MAP_ANONYMOUS,
            -1,
            0,
        )
    };

    if alignment <= page {
        let addr = raw_map(size);
        if addr == libc::MAP_FAILED {
            return None;
        }
        return ptr::NonNull::new(addr as *mut u8);
    }

    let over_size = size + alignment;
    let addr = raw_map(over_size);
    if addr == libc::MAP_FAILED {
        return None;
    }
    let base = addr as usize;
    let aligned = (base + alignment - 1) & !(alignment - 1);

    let prefix_len = aligned - base;
    let suffix_start = aligned + size;
    let suffix_len = (base + over_size) - suffix_start;

    // SAFETY: `prefix_len`/`suffix_len` describe the unused head/tail of
    // the mapping just created above, both strictly within its bounds
    // (arithmetic above guarantees `base <= aligned` and
    // `aligned + size <= base + over_size`); trimming them is the standard
    // overmap-then-shrink pattern for aligned anonymous mmaps.
    unsafe {
        if prefix_len > 0 {
            libc::munmap(base as *mut c_void, prefix_len);
        }
        if suffix_len > 0 {
            libc::munmap(suffix_start as *mut c_void, suffix_len);
        }
    }

    ptr::NonNull::new(aligned as *mut u8)
}

unsafe extern "C" fn cxl_extent_alloc(
    _extent_hooks: *mut ExtentHooks,
    new_addr: *mut c_void,
    size: usize,
    alignment: usize,
    zero: *mut bool,
    commit: *mut bool,
    arena_ind: c_uint,
) -> *mut c_void {
    // jemalloc uses a non-null `new_addr` to ask for a specific address
    // (e.g. while trying to grow an extent in place). Honoring that would
    // require MAP_FIXED_NOREPLACE plus careful handling of the "someone
    // else already has that range" case -- not confidently supported here,
    // so per this crate's documented policy we simply decline (return
    // null) rather than risk silently mapping the wrong thing.
    if !new_addr.is_null() {
        return ptr::null_mut();
    }

    let Some(mapped) = map_aligned(size, alignment) else {
        return ptr::null_mut();
    };

    if let Some(cfg) = registry()
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
        .get(&arena_ind)
        .copied()
    {
        if let Err(e) = mbind(mapped.as_ptr() as *mut c_void, size, cfg.node, cfg.policy) {
            eprintln!(
                "jemalloc_cxl: mbind failed for arena {arena_ind} (node {}, {:?}): {e} -- failing this allocation",
                cfg.node, cfg.policy
            );
            // SAFETY: `mapped`/`size` are exactly the region just mapped
            // above by this same call, not yet returned to jemalloc or
            // touched by anyone else.
            unsafe {
                libc::munmap(mapped.as_ptr() as *mut c_void, size);
            }
            return ptr::null_mut();
        }
    }
    // No registered config (shouldn't normally happen -- every CXL arena
    // registers one at creation time in `create_cxl_arena`): fall back to
    // ordinary, unbound anonymous memory rather than failing outright.

    // SAFETY: `zero`/`commit` are valid out-parameters per jemalloc's
    // extent_alloc_t contract for the duration of this call.
    unsafe {
        // MAP_ANONYMOUS pages are always kernel-zeroed on first fault, so
        // this is true regardless of what the caller requested.
        *zero = true;
        // This hook never does two-phase commit/decommit -- everything it
        // returns is immediately usable (backed by a real, already-mapped
        // region), matching `cxl_extent_commit`'s "always already
        // committed" behavior below.
        *commit = true;
    }

    mapped.as_ptr() as *mut c_void
}

unsafe extern "C" fn cxl_extent_dalloc(
    _extent_hooks: *mut ExtentHooks,
    addr: *mut c_void,
    size: usize,
    _committed: bool,
    _arena_ind: c_uint,
) -> bool {
    // SAFETY: jemalloc guarantees `addr`/`size` describe an extent this
    // same hook set previously returned from `alloc`, not otherwise in use.
    let rc = unsafe { libc::munmap(addr, size) };
    // bool return convention here (per jemalloc's extent_dalloc_t): `false`
    // = successfully deallocated, `true` = failed/opted out (jemalloc will
    // retain the extent mapped and may call `destroy` on it later instead).
    rc != 0
}

unsafe extern "C" fn cxl_extent_destroy(
    _extent_hooks: *mut ExtentHooks,
    addr: *mut c_void,
    size: usize,
    _committed: bool,
    _arena_ind: c_uint,
) {
    // SAFETY: same guarantee as `cxl_extent_dalloc` -- `destroy` is
    // jemalloc's unconditional "actually give this back" call, with no
    // return value to signal failure (there's nothing left to fall back
    // to), so `munmap`'s result is intentionally not checked.
    unsafe {
        libc::munmap(addr, size);
    }
}

unsafe extern "C" fn cxl_extent_commit(
    _extent_hooks: *mut ExtentHooks,
    _addr: *mut c_void,
    _size: usize,
    _offset: usize,
    _length: usize,
    _arena_ind: c_uint,
) -> bool {
    // This hook's memory is always already fully committed (a plain,
    // fully-backed anonymous mapping from `cxl_extent_alloc`, never a
    // reserved-but-uncommitted region) -- `false` means "success."
    false
}

unsafe extern "C" fn cxl_extent_decommit(
    _extent_hooks: *mut ExtentHooks,
    _addr: *mut c_void,
    _size: usize,
    _offset: usize,
    _length: usize,
    _arena_ind: c_uint,
) -> bool {
    // Documented choice: opt out of decommit entirely (`true`) rather than
    // implementing it via madvise. This hook's `commit` above always
    // reports "already committed," so honoring a decommit request would
    // require this hook to then also handle being asked to re-commit that
    // same range later -- opting out keeps the commit/decommit contract
    // simple and matches this prototype's "always-committed" model. The
    // memory is still reclaimable via `purge_forced` (MADV_DONTNEED) below.
    true
}

unsafe extern "C" fn cxl_extent_purge_lazy(
    _extent_hooks: *mut ExtentHooks,
    addr: *mut c_void,
    _size: usize,
    offset: usize,
    length: usize,
    _arena_ind: c_uint,
) -> bool {
    // SAFETY: `addr + offset` for `length` bytes is within the extent this
    // hook set previously allocated (jemalloc's contract for extent_purge_t).
    let target = unsafe { (addr as *mut u8).add(offset) as *mut c_void };
    // SAFETY: `target`/`length` as computed above; MADV_FREE is a hint the
    // kernel may reclaim lazily -- it never changes the mapping's validity.
    let rc = unsafe { libc::madvise(target, length, libc::MADV_FREE) };
    // `false` = purge succeeded/was accepted; `true` = failed/unsupported.
    rc != 0
}

unsafe extern "C" fn cxl_extent_purge_forced(
    _extent_hooks: *mut ExtentHooks,
    addr: *mut c_void,
    _size: usize,
    offset: usize,
    length: usize,
    _arena_ind: c_uint,
) -> bool {
    // SAFETY: see `cxl_extent_purge_lazy` -- same contract.
    let target = unsafe { (addr as *mut u8).add(offset) as *mut c_void };
    // SAFETY: `target`/`length` as computed above; MADV_DONTNEED
    // immediately discards the pages' contents (this is the "forced"
    // variant -- callers must not assume the data survives).
    let rc = unsafe { libc::madvise(target, length, libc::MADV_DONTNEED) };
    rc != 0
}

unsafe extern "C" fn cxl_extent_split(
    _extent_hooks: *mut ExtentHooks,
    _addr: *mut c_void,
    _size: usize,
    _size_a: usize,
    _size_b: usize,
    _committed: bool,
    _arena_ind: c_uint,
) -> bool {
    // Accept (`false` = "split succeeded"). Splitting one mmap'd region
    // into two independently-managed extents needs no bookkeeping of our
    // own: `cxl_extent_dalloc`/`cxl_extent_destroy` already just `munmap`
    // whatever `[addr, addr+size)` jemalloc hands them, and `munmap`
    // operates on virtual address ranges, not on the original `mmap` call
    // that created them -- it's perfectly valid to `munmap` half of a
    // larger mapping while the other half stays live. jemalloc's own
    // extent tracker (not this hook) is what remembers the two halves are
    // now separate; nothing here needs to record that split.
    //
    // Declining this (the prior behavior) meant every extent, once
    // allocated, could only ever be freed or destroyed as a single whole
    // unit -- jemalloc could never carve a smaller allocation out of a
    // larger extent's leftover space, which is a prerequisite for
    // `cxl_extent_merge` below ever mattering (there's nothing to
    // re-merge if nothing was ever split).
    false
}

unsafe extern "C" fn cxl_extent_merge(
    _extent_hooks: *mut ExtentHooks,
    _addr_a: *mut c_void,
    _size_a: usize,
    _addr_b: *mut c_void,
    _size_b: usize,
    _committed: bool,
    _arena_ind: c_uint,
) -> bool {
    // Accept (`false` = "merge succeeded"). jemalloc only ever calls this
    // for two extents it has already verified are adjacent
    // (`addr_b == addr_a + size_a`) and that belong to the same arena --
    // this hook doesn't need to re-verify either property itself, just
    // honor the request. Two same-arena extents are always safe to treat
    // as one from here on:
    //
    //  - NUMA policy can't mismatch: every extent in a given arena was
    //    `mbind`'d with that arena's single registered `ArenaNumaConfig`
    //    (see `cxl_extent_alloc`), so both halves already share the same
    //    node/policy.
    //  - Commit state can't mismatch: `cxl_extent_commit` always reports
    //    "already committed" and `cxl_extent_decommit` always declines, so
    //    every extent this hook set ever produces is uniformly committed.
    //  - Freeing the merged range later is already safe: same reasoning as
    //    `cxl_extent_split` above -- `munmap` doesn't care whether
    //    `[addr_a, addr_a+size_a+size_b)` was one original mapping, two
    //    adjacent ones, or a re-split/re-merged combination of prior
    //    splits.
    //
    // This is what actually lets jemalloc's own decay/coalescing logic
    // reclaim fragmented free space in these arenas (previously declining
    // both split and merge meant every freed extent stayed a permanently
    // separate, never-reunited island -- see the DramMultiArenaObjects
    // investigation in the parent crate for the retention cost this had).
    false
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nodemask_sets_only_the_requested_bit() {
        let mask = build_nodemask(0).unwrap();
        assert_eq!(mask[0], 1);
        assert!(mask[1..].iter().all(|&w| w == 0));

        let mask = build_nodemask(65).unwrap();
        assert_eq!(mask[0], 0);
        assert_eq!(mask[1], 1 << 1);

        let mask = build_nodemask(1023).unwrap();
        assert_eq!(mask[15], 1u64 << 63);
    }

    #[test]
    fn nodemask_rejects_out_of_range_node() {
        assert!(matches!(
            build_nodemask(NODEMASK_BITS as u32),
            Err(MbindError::NodeOutOfRange { .. })
        ));
    }

    #[test]
    fn map_aligned_respects_large_alignment() {
        let alignment = 1 << 21; // 2 MiB, larger than the page size
        let mapped = map_aligned(4096, alignment).expect("map_aligned should succeed");
        assert_eq!(mapped.as_ptr() as usize % alignment, 0);

        // SAFETY: freeing exactly the region map_aligned returned, sized to
        // what was requested (matches this test's own allocation).
        unsafe {
            libc::munmap(mapped.as_ptr() as *mut c_void, 4096);
        }
    }

    #[test]
    fn map_aligned_page_granularity_roundtrips() {
        let page = page_size();
        let mapped = map_aligned(page * 4, page).expect("map_aligned should succeed");
        assert_eq!(mapped.as_ptr() as usize % page, 0);

        // SAFETY: touch every mapped byte to confirm the region is really
        // usable memory, then free it.
        unsafe {
            std::ptr::write_bytes(mapped.as_ptr(), 0xAB, page * 4);
            assert_eq!(*mapped.as_ptr(), 0xAB);
            libc::munmap(mapped.as_ptr() as *mut c_void, page * 4);
        }
    }
}
