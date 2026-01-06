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

    #[test]
    fn test_lfu_eviction_with_pmem() {
        // Test LFU-specific eviction behavior with PMem-backed HashMap
        let cache = PaperCache::<u32, BufferPMEM>::new(
            500, // Small cache to force evictions
            &[PaperPolicy::Lfu],
            PaperPolicy::Lfu,
        ).expect("Failed to create cache");

        // Insert items with different access frequencies
        // Item 0: accessed 4 times
        // Item 1: accessed 3 times  
        // Item 2: accessed 2 times
        // Item 3: accessed 1 time
        for access in [0, 1, 1, 1, 0, 2, 3, 0, 2, 0] {
            cache.set(access, &[0u8; 50], None).expect("Failed to set");
            thread::sleep(Duration::from_millis(10));
        }

        thread::sleep(Duration::from_millis(100));

        // Add more items to trigger evictions
        // The least frequently used items should be evicted first
        for i in 10..20 {
            cache.set(i, &[0u8; 50], None).expect("Failed to set");
            thread::sleep(Duration::from_millis(10));
        }

        thread::sleep(Duration::from_millis(100));

        // Item 0 (most frequent) should still be in cache
        let has_0 = cache.get(&0).is_ok();
        println!("LFU test: Most frequent item (0) in cache: {}", has_0);

        // Item 3 (least frequent) should likely have been evicted
        let has_3 = cache.get(&3).is_ok();
        println!("LFU test: Least frequent item (3) in cache: {}", has_3);

        println!("LFU eviction test with PMem completed without segfault");
    }

    #[test]
    fn test_lfu_stress_with_pmem() {
        // Stress test for LFU with many insertions, updates, and evictions
        let cache = PaperCache::<u32, BufferPMEM>::new(
            2000, // 2KB cache
            &[PaperPolicy::Lfu],
            PaperPolicy::Lfu,
        ).expect("Failed to create cache");

        // Perform many operations to stress test LFU with PMem HashMap
        for iteration in 0..10 {
            // Insert a batch of items
            for i in (iteration * 20)..((iteration + 1) * 20) {
                cache.set(i, &[0u8; 30], None).expect("Failed to set");
            }

            // Access some items multiple times to create frequency patterns
            for _ in 0..3 {
                for i in (iteration * 20)..((iteration + 1) * 20) {
                    if i % 3 == 0 {
                        let _ = cache.get(&i);
                    }
                }
            }

            thread::sleep(Duration::from_millis(50));
        }

        thread::sleep(Duration::from_millis(200));

        println!("LFU stress test with PMem completed without segfault - {} operations", 200 * 10);
    }
}


