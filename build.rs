


extern crate bindgen;

use std::path::{Path, PathBuf};

fn main() {
    let wrapper_path = "/home/griff/work/wrapper.h";

    let umf_lib_dir = "/home/griff/umf-install/lib64";
    let umf_include_dir = "/home/griff/umf-install/include";

    if Path::new(wrapper_path).exists() {
        println!("cargo:rerun-if-changed=wrapper.h");

        // Tell Rust where UMF lives
        println!("cargo:rustc-link-search=native={}", umf_lib_dir);
        println!("cargo:rustc-link-lib=dylib=umf"); // use shared library

        // Compile the real UMF wrapper C code
        cc::Build::new()
            .file("umf_allocator/umf_allocator_wrapper.c")
            .file("umf_allocator/prefault_provider.c")   // <-- add this
            .include("umf_allocator")
            .include(umf_include_dir)
            .compile("umf_allocator_wrapper");

    } else {
        // UMF hardware / headers not available (CI, developer machines).
        // Compile the stub implementation so that the UMF C symbols are
        // defined and the binary links without errors.  All allocations in
        // the stub fall back to standard malloc/free, which is correct for
        // testing: the far-tier PMEM path is exercised functionally even
        // though no actual persistent memory is used.
        println!("cargo:warning=UMF wrapper.h not found; compiling stub allocator for testing");

        cc::Build::new()
            .file("umf_allocator/umf_stub.c")
            .include("umf_allocator")
            .compile("umf_allocator_stub");

        // Keep umf_bindings.rs stub (auto-generated declarations are not
        // needed here; the real Rust-side bindings are in
        // src/umf_allocator_bindings.rs which is always included).
        let umf_bindings_path = PathBuf::from("src/umf_bindings.rs");
        if !umf_bindings_path.exists() {
            std::fs::write(&umf_bindings_path, "// Stub UMF bindings\n")
                .expect("Could not write stub umf_bindings.rs");
        }
    }
}



/* this is for jemallic extent hooks``

extern crate bindgen;

use std::path::{Path, PathBuf};

fn main() {
    // Detect the wrapper header in any of the canonical locations.
    let candidates = [
        "/home/griff/work/wrapper.h",
        "/home/griffin/cxl_baseline/paper-cache-cxl/wrapper.h",
        "wrapper.h",  // project-local fallback
    ];
    let wrapper_path = candidates.iter().find(|p| Path::new(p).exists());

    if let Some(path) = wrapper_path {
        println!("cargo:warning=Building real allocator with wrapper at {}", path);
        println!("cargo:rerun-if-changed={}", path);
        println!("cargo:rerun-if-changed=umf_allocator/jemalloc_extent_hooks.c");

        // Find the system libjemalloc. Both common locations are tried.
        let jemalloc_candidates = [
            "/usr/lib64/libjemalloc.so.2",
            "/usr/lib/x86_64-linux-gnu/libjemalloc.so.2",
        ];
        let jemalloc_path = jemalloc_candidates
            .iter()
            .find(|p| Path::new(p).exists())
            .expect("libjemalloc.so.2 not found in /usr/lib64 or /usr/lib/x86_64-linux-gnu — install jemalloc-devel (Fedora) or libjemalloc-dev (Debian/Ubuntu)");

        println!("cargo:warning=Using jemalloc at {}", jemalloc_path);

        // Detect whether the installed libjemalloc uses je_-prefixed symbols.
        // Fedora: unprefixed (mallctl).  Some Debian builds: prefixed (je_mallctl).
        // We compile the C file with -DJEMALLOC_USE_PREFIX accordingly so it
        // can route its calls through je_mallctl / je_mallocx / je_dallocx
        // when the library expects that.
        let jemalloc_uses_prefix = detect_jemalloc_prefix(jemalloc_path);
        println!("cargo:warning=jemalloc je_ prefix: {}", jemalloc_uses_prefix);

        // Link the system jemalloc by absolute path. Using a path instead of
        // -ljemalloc avoids any chance of the linker resolving to tikv-jemalloc-sys's
        // build directory (which gets added via -L by tikv-jemallocator).
        //println!("cargo:rustc-link-arg={}", jemalloc_path);
        //println!("cargo:rustc-link-search=native=/usr/lib64");
        //println!("cargo:rustc-link-lib=dylib=jemalloc");
        //println!("cargo:rustc-link-lib=dylib=numa");

        //println!("cargo:rustc-link-search=native=/usr/lib64");
        //println!("cargo:rustc-link-arg=-Wl,--no-as-needed");
        //println!("cargo:rustc-link-lib=dylib=jemalloc");
       // println!("cargo:rustc-link-arg=-l:libjemalloc.so.2");
        //println!("cargo:rustc-link-arg=-Wl,--as-needed");
        //println!("cargo:rustc-link-lib=dylib=numa");

        println!("cargo:rustc-link-search=native=/usr/lib64");
        println!("cargo:rustc-link-lib=dylib=numa");

        let mut build = cc::Build::new();
        build
            .file("umf_allocator/jemalloc_extent_hooks.c")
            .include("umf_allocator");

        if jemalloc_uses_prefix {
            build.define("JEMALLOC_USE_PREFIX", None);
        }

        build.compile("umf_allocator_wrapper");
    } else {
        println!("cargo:warning=No wrapper.h found at any candidate path; compiling stub");
        for p in &candidates {
            println!("cargo:warning=  checked: {}", p);
        }

        cc::Build::new()
            .file("umf_allocator/umf_stub.c")
            .include("umf_allocator")
            .compile("umf_allocator_stub");

        let umf_bindings_path = PathBuf::from("src/umf_bindings.rs");
        if !umf_bindings_path.exists() {
            std::fs::write(&umf_bindings_path, "// Stub UMF bindings\n")
                .expect("Could not write stub umf_bindings.rs");
        }
    }
}

/// Returns true if libjemalloc at `path` exports `je_mallctl` instead of
/// the unprefixed `mallctl`. Falls back to false (assume unprefixed) on any error.
fn detect_jemalloc_prefix(path: &str) -> bool {
    let output = match std::process::Command::new("nm").args(["-D", path]).output() {
        Ok(o) => o,
        Err(_) => return false,
    };

    let symbols = String::from_utf8_lossy(&output.stdout);
    let mut has_unprefixed = false;
    let mut has_prefixed = false;

    for line in symbols.lines() {
        // nm -D output format: "<addr> <type> <name>"; we want text symbols ('T').
        // We match "mallctl" / "je_mallctl" exactly to avoid partial matches like
        // "mallctlbymib" or "mallctlnametomib".
        if let Some(name) = line.split_whitespace().nth(2) {
            if !line.contains(" T ") && !line.contains(" W ") {
                continue;
            }
            if name == "mallctl" {
                has_unprefixed = true;
            } else if name == "je_mallctl" {
                has_prefixed = true;
            }
        }
    }

    // Prefer unprefixed when both exist (some builds export both, but the
    // C file's default is to call unprefixed names from jemalloc.h).
    if has_unprefixed {
        false
    } else {
        has_prefixed
    }
}

*/