// Integration test to verify example config files work correctly
// These tests require the allocator_api feature
#![cfg(feature = "allocator_api")]

use std::path::PathBuf;

#[test]
fn test_example_json_config() {
    // Path to the example JSON config in the repo root
    let mut path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    path.push("tiering_config.json");
    
    // This test will only pass if the file exists
    if !path.exists() {
        panic!("Example config file not found at {:?}", path);
    }
    
    // Try to read and parse the file content
    let content = std::fs::read_to_string(&path).expect("Failed to read config file");
    let config: serde_json::Value = serde_json::from_str(&content)
        .expect("Failed to parse JSON config");
    
    // Verify all required fields exist
    assert!(config.get("dram_threshold").is_some(), "dram_threshold is missing");
    assert!(config.get("high_water_mark").is_some(), "high_water_mark is missing");
    assert!(config.get("low_water_mark").is_some(), "low_water_mark is missing");
    assert!(config.get("hotness_threshold").is_some(), "hotness_threshold is missing");
    
    // Verify values are reasonable
    assert_eq!(config["dram_threshold"].as_u64().unwrap(), 1073741824);
    assert_eq!(config["high_water_mark"].as_f64().unwrap(), 0.95);
    assert_eq!(config["low_water_mark"].as_f64().unwrap(), 0.7);
    assert_eq!(config["hotness_threshold"].as_u64().unwrap(), 2);
}

#[test]
fn test_example_toml_config() {
    // Path to the example TOML config in the repo root
    let mut path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    path.push("tiering_config.toml");
    
    // This test will only pass if the file exists
    if !path.exists() {
        panic!("Example config file not found at {:?}", path);
    }
    
    // Try to read and parse the file content
    let content = std::fs::read_to_string(&path).expect("Failed to read config file");
    let config: toml::Value = toml::from_str(&content)
        .expect("Failed to parse TOML config");
    
    // Verify all required fields exist
    assert!(config.get("dram_threshold").is_some(), "dram_threshold is missing");
    assert!(config.get("high_water_mark").is_some(), "high_water_mark is missing");
    assert!(config.get("low_water_mark").is_some(), "low_water_mark is missing");
    assert!(config.get("hotness_threshold").is_some(), "hotness_threshold is missing");
    
    // Verify values are reasonable
    assert_eq!(config["dram_threshold"].as_integer().unwrap(), 1073741824);
    assert_eq!(config["high_water_mark"].as_float().unwrap(), 0.95);
    assert_eq!(config["low_water_mark"].as_float().unwrap(), 0.7);
    assert_eq!(config["hotness_threshold"].as_integer().unwrap(), 2);
}
