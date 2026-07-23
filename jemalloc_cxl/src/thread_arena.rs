//! [`ThreadArenaGuard`]: scoped, whole-thread arena routing via
//! `"thread.arena"`.
//!
//! Unlike [`crate::allocator::CxlAllocator`] (which only affects containers
//! explicitly parameterized with it), this changes where the *calling
//! thread's ordinary, implicit* allocations go -- every `Box::new`,
//! `Vec::push`-triggered growth, `String` allocation, etc. on this thread
//! while the guard is alive lands in the target arena instead of jemalloc's
//! normal automatic-arena selection. That is powerful and risky: it's easy
//! to accidentally route something you didn't mean to (a logging string, an
//! error `Box<dyn Error>`, ...) into a NUMA-bound/CXL arena. Prefer
//! `CxlAllocator` for anything you can name explicitly; reach for this only
//! when you specifically want *everything* a thread does, implicit
//! allocations included, on one arena for a bounded scope.
//!
//! This only affects the thread that creates the guard -- other threads'
//! arena selection is untouched.

use std::os::raw::c_uint;
use std::ptr;

use crate::arena::CxlArena;
use crate::ffi;

#[derive(Debug, thiserror::Error)]
pub enum ThreadArenaError {
    #[error("mallctl(\"thread.arena\") failed with jemalloc error code {0}")]
    MallctlFailed(i32),
}

/// While alive, routes the current thread's implicit allocations into a
/// [`CxlArena`]; restores the thread's previous arena on drop.
#[must_use = "the arena switch is undone as soon as this guard is dropped"]
pub struct ThreadArenaGuard {
    previous: c_uint,
}

impl ThreadArenaGuard {
    /// Switches the current thread to `arena`, returning a guard that
    /// restores the thread's previous arena when dropped.
    pub fn enter(arena: CxlArena) -> Result<Self, ThreadArenaError> {
        let mut previous: c_uint = 0;
        let mut previous_size = size_of::<c_uint>();
        let mut new_ind: c_uint = arena.index();

        // SAFETY: `oldp`/`oldlenp` point at valid, correctly-sized local
        // variables for jemalloc to write the previous arena index into;
        // `newp` points at `new_ind`, a valid `c_uint` for the duration of
        // this call, with `newlen` matching its size exactly.
        let rc = unsafe {
            ffi::mallctl(
                c"thread.arena".as_ptr(),
                (&raw mut previous).cast(),
                &raw mut previous_size,
                (&raw mut new_ind).cast(),
                size_of::<c_uint>(),
            )
        };

        if rc != 0 {
            return Err(ThreadArenaError::MallctlFailed(rc));
        }

        Ok(ThreadArenaGuard { previous })
    }
}

impl Drop for ThreadArenaGuard {
    fn drop(&mut self) {
        let mut restore = self.previous;

        // SAFETY: `newp` points at `restore`, a valid `c_uint` for the
        // duration of this call, with `newlen` matching its size exactly;
        // `oldp`/`oldlenp` are null, which `mallctl` treats as "caller does
        // not want the previous value back" -- fine here, we already have
        // it (`self.previous` was captured before this call overwrites it).
        let rc = unsafe {
            ffi::mallctl(
                c"thread.arena".as_ptr(),
                ptr::null_mut(),
                ptr::null_mut(),
                (&raw mut restore).cast(),
                size_of::<c_uint>(),
            )
        };

        if rc != 0 {
            // Never panic in Drop: log and move on. A failure here means
            // the thread stays on the CXL arena for its remaining implicit
            // allocations, which is surprising but not unsound -- the arena
            // is still a perfectly valid jemalloc arena, just not the one
            // the caller expected to be back on.
            eprintln!(
                "ThreadArenaGuard: failed to restore previous arena {} on drop: mallctl errno {rc}",
                self.previous
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::arena::{create_cxl_arena, CxlArenaConfig};
    use crate::extent::NumaPolicy;

    #[test]
    fn guard_switches_and_restores_thread_arena() {
        let arena = create_cxl_arena(CxlArenaConfig::new(0, NumaPolicy::Preferred))
            .expect("arena creation on node 0 should succeed");

        let before = current_thread_arena();

        {
            let _guard = ThreadArenaGuard::enter(arena).expect("enter should succeed");
            assert_eq!(current_thread_arena(), arena.index());

            // Ordinary, implicit allocations while the guard is held --
            // this is exactly the "whole thread" routing this guard exists
            // for, as opposed to CxlAllocator's explicit, per-container
            // routing.
            let v: Vec<u8> = vec![1, 2, 3, 4, 5];
            assert_eq!(v.iter().sum::<u8>(), 15);
        }

        assert_eq!(current_thread_arena(), before);
    }

    fn current_thread_arena() -> c_uint {
        let mut ind: c_uint = 0;
        let mut ind_size = size_of::<c_uint>();
        let rc = unsafe {
            ffi::mallctl(
                c"thread.arena".as_ptr(),
                (&raw mut ind).cast(),
                &raw mut ind_size,
                ptr::null_mut(),
                0,
            )
        };
        assert_eq!(rc, 0, "reading thread.arena should succeed");
        ind
    }
}
