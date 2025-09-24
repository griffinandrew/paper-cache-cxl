use std::alloc::{GlobalAlloc, Layout};
use std::ptr::null_mut;

mod memkind_bindings {
    include!("memkind_bindings.rs");
}

use memkind_bindings::*;

pub struct FarTierAllocator;

// export MEMKIND_DAX_KMEM_NODES=0

unsafe impl GlobalAlloc for FarTierAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let ptr = memkind_malloc(MEMKIND_DAX_KMEM, layout.size());
        if ptr.is_null() {
            null_mut()
        } else {
            ptr as *mut u8
        }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, _layout: Layout) {
        // not sure about this second argument
        memkind_free(MEMKIND_DAX_KMEM, ptr as *mut ::std::os::raw::c_void);
    }
}
