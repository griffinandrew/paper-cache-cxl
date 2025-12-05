extern crate bindgen;

use std::path::PathBuf;
use std::env;

fn main() {
    println!("cargo:rerun-if-changed=wrapper.h");
    println!("cargo:rustc-link-lib=umf");
    
    // Allow UMF library path to be configured via environment variable
    let umf_lib_path = env::var("UMF_LIB_PATH")
        .unwrap_or_else(|_| "/home/griffin/libs/unified-memory-framework/lib".to_string());
    println!("cargo:rustc-link-search=native={}", umf_lib_path);
    
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
    // Allow UMF include path to be configured via environment variable
    let umf_include_path = env::var("UMF_INCLUDE_PATH")
        .unwrap_or_else(|_| "/home/griffin/libs/unified-memory-framework/include".to_string());
    let umf_header_path = PathBuf::from(&umf_include_path)
        .join("umf/providers/provider_devdax_memory.h");
    
    if umf_header_path.exists() {
        cc::Build::new()
            .file("umf_allocator/umf_allocator_wrapper.c")
            .include("umf_allocator")
            .include(&umf_include_path)
            .compile("umf_allocator"); 

        println!("Compiled umf_allocator.c");
    } else {
        println!("cargo:warning=UMF headers not found at {}, skipping C compilation. This is expected in CI/test environments.", umf_include_path);
        println!("cargo:warning=Set UMF_INCLUDE_PATH environment variable to specify UMF header location.");
        println!("Skipping C compilation");
    }
}