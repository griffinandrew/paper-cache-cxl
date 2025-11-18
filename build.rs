
fn main() 
{
    println!("cargo:rerun-if-changed=wrapper.h");
    println!("cargo:rustc-link-lib=umf");
    println!("cargo:rustc-link-search=native=/home/griffin/libs/unified-memory-framework/lib");
    println!("cargo:rustc-link-lib=umf_allocator");
 
}