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

    cc::Build::new()
        .file("umf_allocator/umf_allocator_wrapper.c")
        .include("umf_allocator")
        .compile("umf_allocator"); 

    println!("Compiled umf_allocator.c");
}