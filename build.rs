extern crate bindgen;

use std::env;
use std::path::PathBuf;

fn main() {
    println!("cargo:rerun-if-changed=wrapper.h");
    
    println!("cargo:rustc-link-lib=numa");

    // Generate bindings
    let bindings = bindgen::Builder::default()
        .header("/home/griffin/code/cxl/paper-cache-cxl/wrapper.h")
        .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()))
        .generate()
        .expect("Unable to generate bindings");

    // Write the bindings to the $OUT_DIR/bindings.rs
    //let out_path = PathBuf::from(env::var("OUT_DIR").unwrap());
    
    let out_path = PathBuf::from("/home/griffin/code/cxl/paper-cache-cxl");
    bindings
        .write_to_file(out_path.join("bindings.rs"))
        .expect("Couldn't write bindings!");

    println!("DONE");
}
