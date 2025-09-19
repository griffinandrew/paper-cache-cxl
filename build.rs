extern crate bindgen;

use std::env;
use std::path::PathBuf;

fn main() {
    println!("cargo:rerun-if-changed=wrapper.h");    
    println!("cargo:rustc-link-lib=pmemobj");

    // Generate bindings
    let bindings = bindgen::Builder::default()
        .header("/home/griffin/code/cxl_paper_cache/paper-cache-cxl/wrapper.h")
        .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()))
        .generate()
        .expect("Unable to generate bindings");

    
    println!("Generated bindings");
    //let out_path = PathBuf::from(env::var("OUT_DIR").unwrap()); if want more dynamic
    //leave hardcoded as so is the path to wrapper.h `above so when moving, change both anyway
    
    let out_path = PathBuf::from("/home/griffin/code/cxl_paper_cache/paper-cache-cxl/src/");
    bindings
        .write_to_file(out_path.join("pmemobj_bindings.rs"))
        .expect("Couldn't write bindings!");

    println!("DONE");
}