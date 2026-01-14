# paper-cache

PaperCache is an in-memory cache which supports the dynamic switching between any eviction policy at runtime.

Note: this crate should not be used directly; please use the paper-server crate instead.

## Tier Allocation (allocator_api feature)

When built with the `allocator_api` feature, this implementation stores **both keys and values** in the pmem (persistent memory) tier using the Hybrid allocator.

- **Keys**: Stored as `Box<K, Hybrid>` in pmem tier
- **Values**: Stored as `Box<[u8], Hybrid>` in pmem tier

This differs from the original libcache implementation where only values were in the far tier and keys remained in DRAM.

See [TIER_ALLOCATION_FINDINGS.md](TIER_ALLOCATION_FINDINGS.md) for detailed implementation documentation. 