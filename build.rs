extern crate bindgen;

use std::path::PathBuf;
use std::env;

fn main() {
    println!("cargo:rerun-if-changed=wrapper.h");
    println!("cargo:rustc-link-lib=umf");
    println!("cargo:rustc-link-search=native=/home/griffin/libs/unified-memory-framework/lib");
    println!("cargo:rustc-link-lib=umf_allocator");


    //these outputs should be different.... all that matters is the linking with the umf repo thooooooooo
    
    // Generate bindings using the local wrapper.h
    let manifest_dir = env::var("CARGO_MANIFEST_DIR").unwrap();
    let wrapper_path = PathBuf::from(&manifest_dir).join("wrapper.h");
    let bindings_path = PathBuf::from(&manifest_dir).join("src").join("umf_bindings.rs");
    
    // Only generate bindings if they don't exist
    if !bindings_path.exists() {
        let bindings = bindgen::Builder::default()
            .header(wrapper_path.to_str().unwrap()) // Use local wrapper.h
            .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()))
            .generate()
            .expect("Unable to generate bindings");

        println!("Generated bindings");

        let out_path = PathBuf::from(&manifest_dir).join("src");
        bindings
            .write_to_file(out_path.join("umf_bindings.rs"))
            .expect("Couldn't write bindings!");
    } else {
        println!("Using existing bindings");
    }

    println!("DONE");

    // Only try to compile the C code if UMF headers are available
    // Check if we can find the UMF header
    let umf_header_path = PathBuf::from("/home/griffin/libs/unified-memory-framework/include/umf/providers/provider_devdax_memory.h");
    if umf_header_path.exists() {
        cc::Build::new()
            .file("umf_allocator/umf_allocator_wrapper.c")
            .include("umf_allocator")
            .include("/home/griffin/libs/unified-memory-framework/include")
            .compile("umf_allocator"); 

        println!("Compiled umf_allocator.c");
    } else {
        println!("cargo:warning=UMF headers not found, skipping C compilation. This is expected in CI/test environments.");
        println!("Skipping C compilation");
    }
}