/*
 * Integration tests for the tiering manager
 */

use paper_cache::{TieringManager, TieringConfig, TieringStats};

#[test]
fn test_tiering_manager_basic_operations() {
    let manager = TieringManager::with_defaults();
    
    // Register an object
    manager.register_object(1, 100);
    let stats = manager.stats();
    assert_eq!(stats.pmem_only_objects, 1);
    assert_eq!(stats.dram_objects, 0);
    
    // Promote to DRAM
    assert!(manager.promote_to_dram(1));
    let stats = manager.stats();
    assert_eq!(stats.dram_objects, 1);
    assert_eq!(stats.dram_size, 100);
    assert_eq!(stats.promotions, 1);
    assert!(manager.is_in_dram(&1));
    
    // Demote from DRAM
    assert!(manager.demote_from_dram(1));
    let stats = manager.stats();
    assert_eq!(stats.dram_objects, 0);
    assert_eq!(stats.dram_size, 0);
    assert_eq!(stats.demotions, 1);
    assert!(!manager.is_in_dram(&1));
}

#[test]
fn test_tiering_threshold_behavior() {
    let config = TieringConfig {
        dram_threshold: 500,
        high_water_mark: 0.8,  // 400 bytes
        low_water_mark: 0.6,   // 300 bytes
    };
    let manager = TieringManager::new(config);
    
    // Add objects up to threshold
    for i in 0..5 {
        manager.register_object(i, 100);
        manager.promote_to_dram(i);
    }
    
    let stats = manager.stats();
    assert_eq!(stats.dram_size, 500);
    
    // Try to add one more (should fail as it exceeds threshold)
    manager.register_object(10, 100);
    assert!(!manager.promote_to_dram(10));
    
    let stats = manager.stats();
    assert_eq!(stats.dram_objects, 5);
    assert_eq!(stats.dram_size, 500);
}

#[test]
fn test_automatic_demotion() {
    let config = TieringConfig {
        dram_threshold: 500,
        high_water_mark: 0.8,  // 400 bytes
        low_water_mark: 0.6,   // 300 bytes
    };
    let manager = TieringManager::new(config);
    
    // Add objects exceeding high water mark
    for i in 0..5 {
        manager.register_object(i, 100);
        manager.promote_to_dram(i);
        std::thread::sleep(std::time::Duration::from_millis(10));
    }
    
    let stats = manager.stats();
    assert_eq!(stats.dram_size, 500);
    
    // Get keys to demote
    let keys_to_demote = manager.get_keys_to_demote();
    
    // Should suggest demoting to get below low water mark (300 bytes)
    // Need to demote at least 200 bytes worth
    assert!(keys_to_demote.len() >= 2);
    
    // Perform demotions
    for key in keys_to_demote {
        manager.demote_from_dram(key);
    }
    
    let stats = manager.stats();
    assert!(stats.dram_size <= 300);
}

#[test]
fn test_access_based_promotion() {
    let manager = TieringManager::with_defaults();
    
    manager.register_object(1, 100);
    
    // First access should not trigger promotion
    assert!(!manager.record_access(1));
    
    // Second access should trigger promotion
    assert!(manager.record_access(1));
}

#[test]
fn test_remove_object_from_dram() {
    let manager = TieringManager::with_defaults();
    
    manager.register_object(1, 100);
    manager.promote_to_dram(1);
    
    let stats = manager.stats();
    assert_eq!(stats.dram_objects, 1);
    
    manager.remove_object(1);
    
    let stats = manager.stats();
    assert_eq!(stats.dram_objects, 0);
    assert_eq!(stats.dram_size, 0);
}

#[test]
fn test_clear_all_tiering_info() {
    let manager = TieringManager::with_defaults();
    
    // Add several objects
    for i in 0..5 {
        manager.register_object(i, 100);
        manager.promote_to_dram(i);
    }
    
    let stats = manager.stats();
    assert_eq!(stats.dram_objects, 5);
    
    // Clear all
    manager.clear();
    
    let stats = manager.stats();
    assert_eq!(stats.dram_objects, 0);
    assert_eq!(stats.dram_size, 0);
    assert_eq!(stats.pmem_only_objects, 0);
}

#[test]
fn test_threshold_update() {
    let manager = TieringManager::with_defaults();
    
    assert_eq!(manager.dram_threshold(), 1_073_741_824); // Default 1GB
    
    manager.set_dram_threshold(2_000_000_000);
    assert_eq!(manager.dram_threshold(), 2_000_000_000);
}
