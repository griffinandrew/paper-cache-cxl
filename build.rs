extern crate bindgen;

use std::path::PathBuf;

fn main() {
    println!("cargo:rerun-if-changed=wrapper.h");
    
    // Only compile and link UMF-related code if allocator_api feature is enabled
    if std::env::var("CARGO_FEATURE_ALLOCATOR_API").is_ok() {
        println!("cargo:rustc-link-lib=umf");
        // Use environment variable for UMF lib path, with a default fallback
        if let Ok(umf_lib_path) = std::env::var("UMF_LIB_PATH") {
            println!("cargo:rustc-link-search=native={}", umf_lib_path);
        }
        println!("cargo:rustc-link-lib=umf_allocator");
        
        // Generate bindings only if UMF is available
        let bindings = bindgen::Builder::default()
            .header("wrapper.h")
            .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()))
            .generate()
            .expect("Unable to generate bindings");

        let out_path = PathBuf::from("src/");
        bindings
            .write_to_file(out_path.join("umf_bindings.rs"))
            .expect("Couldn't write bindings!");

        cc::Build::new()
            .file("umf_allocator/umf_allocator_wrapper.c")
            .include("umf_allocator")
            .compile("umf_allocator");
    } else {
        // Create dummy bindings file when feature is not enabled
        let out_path = PathBuf::from("src/");
        std::fs::write(out_path.join("umf_bindings.rs"), "// UMF bindings disabled\n")
            .expect("Couldn't write dummy bindings!");
    }
}