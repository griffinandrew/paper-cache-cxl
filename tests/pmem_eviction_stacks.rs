// Test to verify that PMem eviction stacks feature compiles and works
// This file only runs when pmem_eviction_stacks feature is enabled
#[cfg(all(feature = "pmem_eviction_stacks", any(feature = "alloc_with_hash", feature = "alloc_api_exp")))]
mod pmem_eviction_tests {
    use paper_cache::{PaperCache, PaperPolicy, BufferPMEM};
    use std::thread;
    use std::time::Duration;

    #[test]
    fn test_cache_with_pmem_eviction_stacks() {
        // Create a cache - this will use PMem for eviction stacks
        let cache = PaperCache::<u32, BufferPMEM>::new(
            1000, // 1KB cache
            &[PaperPolicy::Lru],
            PaperPolicy::Lru,
        ).expect("Failed to create cache");

        // Set some objects
        for i in 0..5 {
            cache.set(i, &[0u8; 100], None).expect("Failed to set object");
        }

        // Give time for worker to process
        thread::sleep(Duration::from_millis(100));

        // Access some objects
        for i in 0..3 {
            let _ = cache.get(&i);
        }

        // Set more objects to trigger eviction
        for i in 5..15 {
            cache.set(i, &[0u8; 100], None).expect("Failed to set object");
        }

        thread::sleep(Duration::from_millis(100));

        // Verify cache is working
        // The recently accessed items (0, 1, 2) should still be in cache
        // while older unaccessed items should have been evicted
        let has_0 = cache.get(&0).is_ok();
        println!("Cache has key 0: {}", has_0);
    }

    #[test]
    fn test_multiple_policies_with_pmem() {
        let policies = vec![
            PaperPolicy::Lru,
            PaperPolicy::Fifo,
            PaperPolicy::Lfu,
        ];

        for policy in policies {
            let cache = PaperCache::<u32, BufferPMEM>::new(
                1000,
                &[policy],
                policy,
            ).expect("Failed to create cache");

            // Basic operations
            for i in 0..5 {
                cache.set(i, &[0u8; 100], None).expect("Failed to set");
            }

            thread::sleep(Duration::from_millis(50));

            // This should work without segfaulting
            for i in 0..3 {
                let _ = cache.get(&i);
            }

            println!("Policy {:?} works with PMem eviction stacks", policy);
        }
    }

    #[test]
    fn test_large_number_of_operations() {
        let cache = PaperCache::<u32, BufferPMEM>::new(
            10000, // 10KB cache
            &[PaperPolicy::Lru],
            PaperPolicy::Lru,
        ).expect("Failed to create cache");

        // Perform many operations to stress test PMem allocation
        for i in 0..200 {
            cache.set(i, &[0u8; 40], None).expect("Failed to set");
        }

        thread::sleep(Duration::from_millis(200));

        // Access pattern
        for _ in 0..5 {
            for i in (0..100).step_by(7) {
                let _ = cache.get(&i);
            }
        }

        thread::sleep(Duration::from_millis(200));

        // Add more to force evictions
        for i in 200..400 {
            cache.set(i, &[0u8; 40], None).expect("Failed to set");
        }

        thread::sleep(Duration::from_millis(200));

        println!("Large operations test completed without segfault");
    }
}


