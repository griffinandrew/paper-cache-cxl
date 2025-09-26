extern crate bindgen;

use std::path::PathBuf;

fn main() {
    println!("cargo:rerun-if-changed=wrapper.h");    
    println!("cargo:rustc-link-lib=memkind");

    // Generate bindings
    let bindings = bindgen::Builder::default()
        .header("/home/griffin/cxl_baseline/paper-cache-cxl/wrapper.h") // Ensure this header includes memkind headers
        .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()))
        .generate()
        .expect("Unable to generate bindings");

    println!("Generated bindings");

    let out_path = PathBuf::from("/home/griffin/cxl_baseline/paper-cache-cxl/src/");
    bindings
        .write_to_file(out_path.join("memkind_bindings.rs"))
        .expect("Couldn't write bindings!");

    println!("DONE");
}
