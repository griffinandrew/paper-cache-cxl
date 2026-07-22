use std::env;
use std::path::PathBuf;

fn main() {
    // Env-var-with-fallback: this crate has no existing callers to break by
    // making these paths overridable (unlike the parent paper-cache crate's
    // build.rs, which hardcodes them). Falls back to the known-working
    // install on this machine if unset.
    let umf_include_dir = env::var("UMF_INCLUDE_DIR")
        .unwrap_or_else(|_| "/home/griff/umf-install/include".to_string());
    let umf_lib_dir = env::var("UMF_LIB_DIR")
        .unwrap_or_else(|_| "/home/griff/umf-install/lib64".to_string());

    println!("cargo:rustc-link-search=native={umf_lib_dir}");
    println!("cargo:rustc-link-lib=dylib=umf");
    println!("cargo:rustc-link-lib=dylib=numa");

    // Cargo features are visible to build.rs via CARGO_FEATURE_<NAME> env
    // vars, not #[cfg(...)] -- build.rs is a separate compilation unit from
    // the crate it builds.
    let jemalloc_pool = env::var_os("CARGO_FEATURE_JEMALLOC_POOL").is_some();

    let mut builder = bindgen::Builder::default()
        .header("wrapper.h")
        .clang_arg(format!("-I{umf_include_dir}"));

    // The pip-installed `libclang` shared library bindgen loads (via
    // LIBCLANG_PATH) doesn't know where this machine's system C standard
    // headers (stddef.h, etc.) live -- unlike a full clang/LLVM install,
    // it ships no bundled resource-dir headers of its own. Point it at
    // gcc's own builtin include directory (which does have them), the same
    // compiler this build already relies on via the `cc` crate elsewhere in
    // this codebase's build scripts.
    if let Ok(output) = std::process::Command::new("gcc")
        .arg("-print-file-name=include")
        .output()
    {
        if output.status.success() {
            let gcc_include = String::from_utf8_lossy(&output.stdout).trim().to_string();
            if !gcc_include.is_empty() {
                builder = builder.clang_arg(format!("-I{gcc_include}"));
            }
        }
    }

    builder = builder
        .allowlist_function("umf(Memory(Provider|Pool).*|OsMemoryProvider.*|Pool.*|ScalablePool.*|Jemalloc.*|Disjoint.*|Free)")
        .allowlist_type("umf_.*")
        .allowlist_var("UMF_.*")
        .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()));

    if jemalloc_pool {
        builder = builder.clang_arg("-DTIER_ALLOCATOR_WITH_JEMALLOC_POOL");
    }

    let bindings = builder
        .generate()
        .expect("failed to generate UMF bindings via bindgen");

    let out_path = PathBuf::from(env::var("OUT_DIR").unwrap());
    bindings
        .write_to_file(out_path.join("umf_bindings.rs"))
        .expect("failed to write UMF bindings to $OUT_DIR");

    println!("cargo:rerun-if-changed=wrapper.h");
    println!("cargo:rerun-if-env-changed=UMF_INCLUDE_DIR");
    println!("cargo:rerun-if-env-changed=UMF_LIB_DIR");
}
