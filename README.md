# paper-cache

PaperCache is an in-memory cache which supports the dynamic switching between any eviction policy at runtime.

Note: this crate should not be used directly; please use the paper-server crate instead.

this branch adds both key and value to the far tier... in libcache it is just the value that is in the far tier... 