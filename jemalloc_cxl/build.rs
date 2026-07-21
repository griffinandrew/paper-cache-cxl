use std::env;
use std::path::PathBuf;

fn main() {
    // tikv-jemalloc-sys declares `links = "jemalloc"` in its Cargo.toml, so
    // Cargo forwards every `cargo:key=value` line its build script emits to
    // us as `DEP_JEMALLOC_<KEY>` -- including `cargo:root=<OUT_DIR>`, the
    // directory containing the exact jemalloc.h this build was configured
    // with (prefix, size classes, etc. all match the linked library).
    // Compiling our shim against any other jemalloc.h (e.g. a system-wide
    // /usr/include/jemalloc/jemalloc.h) risks a header/library ABI mismatch
    // -- this is what guarantees "one jemalloc instance" end to end.
    let jemalloc_root = env::var("DEP_JEMALLOC_ROOT")
        .expect("DEP_JEMALLOC_ROOT not set -- is tikv-jemalloc-sys a direct dependency?");

    let include_dir = PathBuf::from(&jemalloc_root).join("include");

    println!("cargo:rerun-if-changed=native/jemalloc_shim.c");
    println!("cargo:rerun-if-env-changed=DEP_JEMALLOC_ROOT");

    cc::Build::new()
        .file("native/jemalloc_shim.c")
        .include(&include_dir)
        .warnings(true)
        .compile("jemalloc_cxl_shim");
}
