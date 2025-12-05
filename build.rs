extern crate bindgen;

use std::path::PathBuf;
use std::env;
use std::fs;

fn main() {
    println!("cargo:rerun-if-changed=wrapper.h");
    println!("cargo:rustc-link-lib=umf");
    println!("cargo:rustc-link-search=native=/home/griffin/libs/unified-memory-framework/lib");
    println!("cargo:rustc-link-lib=umf_allocator");

    let manifest_dir = env::var("CARGO_MANIFEST_DIR").unwrap();
    let src_path = PathBuf::from(&manifest_dir).join("src");

    // Check if UMF library is available
    let wrapper_path = PathBuf::from(&manifest_dir).join("wrapper.h");
    
    // Try to generate bindings, but fall back to stubs if it fails
    let bindings_result = bindgen::Builder::default()
        .header(wrapper_path.to_str().unwrap())
        .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()))
        .generate();

    match bindings_result {
        Ok(bindings) => {
            println!("Generated bindings");
            bindings
                .write_to_file(src_path.join("umf_bindings.rs"))
                .expect("Couldn't write bindings!");
        }
        Err(_) => {
            println!("cargo:warning=UMF library not available, using stub bindings");
            // Stub bindings are already created in src/umf_bindings.rs
        }
    }

    println!("DONE");

    // Try to compile the wrapper, but don't fail if it doesn't work
    let _ = cc::Build::new()
        .file("umf_allocator/umf_allocator_wrapper.c")
        .include("umf_allocator")
        .try_compile("umf_allocator");

    println!("Build script completed");
}