// Stub bindings for CI environment
use std::ffi::c_void;

#[allow(dead_code)]
pub unsafe fn umf_allocator_init(_dax_path: *const i8, _dax_size: u64) {}

#[allow(dead_code)]
pub unsafe fn umf_alloc(_size: usize, _align: usize) -> *mut c_void {
    std::ptr::null_mut()
}

#[allow(dead_code)]
pub unsafe fn umf_free(_ptr: *mut c_void, _size: usize) {}

#[allow(dead_code)]
pub unsafe fn check_tier(_ptr: *mut c_void) -> i32 {
    0
}
