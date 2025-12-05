extern crate bindgen;

use std::path::{Path, PathBuf};

fn main() {
    let wrapper_path = "/home/griffin/cxl_baseline/paper-server-cxl/wrapper.h";
    
    // Only run bindgen if the wrapper exists
    if Path::new(wrapper_path).exists() {
        println!("cargo:rerun-if-changed=wrapper.h");
        println!("cargo:rustc-link-lib=umf");
        println!("cargo:rustc-link-search=native=/home/griffin/libs/unified-memory-framework/lib");
        println!("cargo:rustc-link-lib=umf_allocator");

        // Generate bindings
        let bindings = bindgen::Builder::default()
            .header(wrapper_path) // Ensure this header includes memkind headers
            .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()))
            .generate()
            .expect("Unable to generate bindings");

        println!("Generated bindings");

        let out_path = PathBuf::from("/home/griffin/cxl_baseline/paper-server-cxl/src/");
        bindings
            .write_to_file(out_path.join("umf_bindings.rs"))
            .expect("Couldn't write bindings!");

        println!("DONE");

        cc::Build::new()
            .file("umf_allocator/umf_allocator_wrapper.c")
            .include("umf_allocator")
            .compile("umf_allocator"); 

        println!("Compiled umf_allocator.c");
    } else {
        println!("cargo:warning=UMF wrapper.h not found; skipping bindgen");
        // Create a minimal stub if umf_bindings.rs doesn't exist
        let umf_bindings_path = PathBuf::from("src/umf_bindings.rs");
        if !umf_bindings_path.exists() {
            std::fs::write(&umf_bindings_path, "// Stub UMF bindings\n")
                .expect("Could not write stub umf_bindings.rs");
        }
    }
}