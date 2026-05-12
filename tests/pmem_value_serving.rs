#[cfg(feature = "key_value_pmem")]
mod pmem_value_serving_tests {
    use paper_cache::{BufferPMEM, PaperCache, PaperPolicy};
    use std::sync::Arc;

    #[test]
    fn get_pmem_returns_same_arc_as_peek() {
        let cache = PaperCache::<u32, BufferPMEM>::new(10_000, &[PaperPolicy::Lfu], PaperPolicy::Lfu)
            .expect("failed to create cache");

        let key = 7u32;
        let expected: Vec<u8> = (0u8..64).collect();
        cache.set(key, &expected, None).expect("set failed");

        let peeked = cache.peek(&key).expect("peek failed");
        let got = cache.get_pmem(&key).expect("get_pmem failed");

        assert!(Arc::ptr_eq(&peeked, &got), "expected get_pmem to be zero-copy (same Arc)");
        assert_eq!(peeked.as_ref().as_ref(), expected.as_slice());
        assert_eq!(got.as_ref().as_ref(), expected.as_slice());
    }

    #[test]
    fn get_pmem_returns_key_not_found_for_missing_key() {
        let cache = PaperCache::<u32, BufferPMEM>::new(10_000, &[PaperPolicy::Lfu], PaperPolicy::Lfu)
            .expect("failed to create cache");

        assert!(cache.get_pmem(&123).is_err());
    }
}

#[cfg(all(feature = "key_value_pmem", feature = "enable_tiering_manager"))]
mod tiering_get_pmem_tests {
    use paper_cache::{BufferPMEM, PaperCache, PaperPolicy};
    use std::sync::Arc;
    use std::thread;
    use std::time::Duration;

    #[test]
    fn get_pmem_always_reads_from_pmem_source_of_truth() {
        let cache = PaperCache::<u32, BufferPMEM>::new(100_000, &[PaperPolicy::Lfu], PaperPolicy::Lfu)
            .expect("failed to create cache");

        cache.set_hotness_threshold(1);

        let key = 42u32;
        let expected = vec![9u8; 256];
        cache.set(key, &expected, None).expect("set failed");

        // Drive promotion so the tiering manager may cache a DRAM copy.
        let _ = cache.get(&key);
        let _ = cache.get(&key);
        thread::sleep(Duration::from_millis(150));

        let peeked = cache.peek(&key).expect("peek failed");
        let got = cache.get_pmem(&key).expect("get_pmem failed");

        assert!(Arc::ptr_eq(&peeked, &got), "expected get_pmem to return PMEM Arc even after promotion");
        assert_eq!(got.as_ref().as_ref(), expected.as_slice());
    }
}

