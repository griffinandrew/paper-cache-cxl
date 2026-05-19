#[cfg(all(
    feature = "pmem_region_alloc",
    feature = "key_value_pmem",
    feature = "global_hashtable_pmem"
))]
mod pmem_region_global_hashtable_tests {
    use paper_cache::{BufferPMEM, PaperCache, PaperPolicy};
    use std::sync::Once;

    fn configure_test_region() {
        static INIT: Once = Once::new();
        INIT.call_once(|| unsafe {
            std::env::set_var("PAPER_CACHE_PMEM_REGION_SIZE", "67108864");
            std::env::set_var("PAPER_CACHE_PMEM_NUMA_NODE", "0");
            std::env::set_var("PAPER_CACHE_EVICTION_STACK_CAPACITY", "10000");
        });
    }

    #[test]
    fn global_hashtable_pmem_region_alloc_supports_get_set() {
        configure_test_region();
        let cache = PaperCache::<u64, BufferPMEM>::new(
            1_000_000,
            &[PaperPolicy::Lfu],
            PaperPolicy::Lfu,
        )
        .expect("cache init failed");

        cache.set(10, b"alpha", None).expect("set alpha failed");
        cache.set(20, b"beta", None).expect("set beta failed");

        assert_eq!(cache.get(&10).expect("get alpha failed"), b"alpha".to_vec());
        assert_eq!(cache.get(&20).expect("get beta failed"), b"beta".to_vec());

        cache.set(10, b"alpha-2", None).expect("update alpha failed");
        assert_eq!(cache.get(&10).expect("get alpha update failed"), b"alpha-2".to_vec());
    }
}

#[cfg(all(
    feature = "pmem_region_alloc",
    feature = "key_value_pmem",
    feature = "eviction_stacks_pmem"
))]
mod pmem_region_eviction_stacks_tests {
    use paper_cache::{BufferPMEM, PaperCache, PaperPolicy};
    use std::sync::Once;

    fn configure_test_region() {
        static INIT: Once = Once::new();
        INIT.call_once(|| unsafe {
            std::env::set_var("PAPER_CACHE_PMEM_REGION_SIZE", "67108864");
            std::env::set_var("PAPER_CACHE_PMEM_NUMA_NODE", "0");
            std::env::set_var("PAPER_CACHE_EVICTION_STACK_CAPACITY", "10000");
        });
    }

    #[test]
    fn eviction_stacks_pmem_region_alloc_supports_get_set() {
        configure_test_region();
        let cache = PaperCache::<u64, BufferPMEM>::new(
            2_000_000,
            &[PaperPolicy::Lfu],
            PaperPolicy::Lfu,
        )
        .expect("cache init failed");

        for i in 0..256u64 {
            let value = vec![i as u8; 64];
            cache.set(i, &value, None).expect("set failed");
        }

        for i in 0..256u64 {
            let value = cache.get(&i).expect("get failed");
            assert_eq!(value, vec![i as u8; 64]);
        }
    }
}
