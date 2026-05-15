
/* 

extern crate bindgen;

use std::path::{Path, PathBuf};

fn main() {
    let wrapper_path = "/home/griff/work/wrapper.h";

    let umf_lib_dir = "/home/griff/umf-install/lib64";
    let umf_include_dir = "/home/griff/umf-install/include";

    if Path::new(wrapper_path).exists() {
        println!("cargo:rerun-if-changed=wrapper.h");

        // Tell Rust where UMF lives
        println!("cargo:rustc-link-search=native={}", umf_lib_dir);
        println!("cargo:rustc-link-lib=dylib=umf"); // use shared library

        // Compile the real UMF wrapper C code
        cc::Build::new()
            .file("umf_allocator/umf_allocator_wrapper.c")
            .include("umf_allocator")
            .include(umf_include_dir)
            .compile("umf_allocator_wrapper");

    } else {
        // UMF hardware / headers not available (CI, developer machines).
        // Compile the stub implementation so that the UMF C symbols are
        // defined and the binary links without errors.  All allocations in
        // the stub fall back to standard malloc/free, which is correct for
        // testing: the far-tier PMEM path is exercised functionally even
        // though no actual persistent memory is used.
        println!("cargo:warning=UMF wrapper.h not found; compiling stub allocator for testing");

        cc::Build::new()
            .file("umf_allocator/umf_stub.c")
            .include("umf_allocator")
            .compile("umf_allocator_stub");

        // Keep umf_bindings.rs stub (auto-generated declarations are not
        // needed here; the real Rust-side bindings are in
        // src/umf_allocator_bindings.rs which is always included).
        let umf_bindings_path = PathBuf::from("src/umf_bindings.rs");
        if !umf_bindings_path.exists() {
            std::fs::write(&umf_bindings_path, "// Stub UMF bindings\n")
                .expect("Could not write stub umf_bindings.rs");
        }
    }
}



*/





extern crate bindgen;

use std::path::{Path, PathBuf};

fn main() {
    //let wrapper_path = "/home/griff/work/wrapper.h";
    //let wrapper_path = "/home/griffin/cxl_baseline/paper-cache-cxl/wrapper.h";

    // Detect the wrapper header in either canonical location.
    let candidates = [
        "/home/griff/work/wrapper.h",
        "/home/griffin/cxl_baseline/paper-cache-cxl/wrapper.h",
        "wrapper.h",  // project-local fallback
    ];
    let wrapper_path = candidates.iter().find(|p| Path::new(p).exists());

    if let Some(path) = wrapper_path {
        println!("cargo:warning=Building real allocator with wrapper at {}", path);
        println!("cargo:rerun-if-changed={}", path);
        println!("cargo:rerun-if-changed=umf_allocator/jemalloc_extent_hooks.c");

        // jemalloc + libnuma. No UMF, no TBB.
        println!("cargo:rustc-link-lib=dylib=jemalloc");
        println!("cargo:rustc-link-lib=dylib=numa");

        cc::Build::new()
            .file("umf_allocator/jemalloc_extent_hooks.c")
            .include("umf_allocator")
            .flag("-D_GNU_SOURCE")
            .compile("umf_allocator_wrapper");
    } else {
        println!("cargo:warning=No wrapper.h found at any candidate path; compiling stub");
        for p in &candidates {
            println!("cargo:warning=  checked: {}", p);
        }

        cc::Build::new()
            .file("umf_allocator/umf_stub.c")
            .include("umf_allocator")
            .compile("umf_allocator_stub");

        let umf_bindings_path = PathBuf::from("src/umf_bindings.rs");
        if !umf_bindings_path.exists() {
            std::fs::write(&umf_bindings_path, "// Stub UMF bindings\n")
                .expect("Could not write stub umf_bindings.rs");
        }
    }
}




