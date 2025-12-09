// Test config file loading functionality
use std::io::Write;
use tempfile::NamedTempFile;

// Mock the TieringConfig to test serialization in isolation
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
struct TieringConfig {
    dram_threshold: u64,
    high_water_mark: f64,
    low_water_mark: f64,
    hotness_threshold: u64,
}

#[test]
fn test_json_serialization() {
    let config = TieringConfig {
        dram_threshold: 2_000_000_000,
        high_water_mark: 0.85,
        low_water_mark: 0.65,
        hotness_threshold: 5,
    };

    let json = serde_json::to_string(&config).unwrap();
    let deserialized: TieringConfig = serde_json::from_str(&json).unwrap();
    
    assert_eq!(deserialized, config);
}

#[test]
fn test_toml_serialization() {
    let config = TieringConfig {
        dram_threshold: 2_000_000_000,
        high_water_mark: 0.85,
        low_water_mark: 0.65,
        hotness_threshold: 5,
    };

    let toml_str = toml::to_string(&config).unwrap();
    let deserialized: TieringConfig = toml::from_str(&toml_str).unwrap();
    
    assert_eq!(deserialized, config);
}

#[test]
fn test_json_file_loading() {
    let mut file = NamedTempFile::new().unwrap();
    let json_content = r#"{
        "dram_threshold": 3000000000,
        "high_water_mark": 0.92,
        "low_water_mark": 0.68,
        "hotness_threshold": 4
    }"#;
    file.write_all(json_content.as_bytes()).unwrap();
    file.flush().unwrap();
    
    let contents = std::fs::read_to_string(file.path()).unwrap();
    let config: TieringConfig = serde_json::from_str(&contents).unwrap();
    
    assert_eq!(config.dram_threshold, 3_000_000_000);
    assert_eq!(config.high_water_mark, 0.92);
    assert_eq!(config.low_water_mark, 0.68);
    assert_eq!(config.hotness_threshold, 4);
}

#[test]
fn test_toml_file_loading() {
    let mut file = NamedTempFile::new().unwrap();
    let toml_content = r#"
dram_threshold = 4000000000
high_water_mark = 0.88
low_water_mark = 0.72
hotness_threshold = 6
"#;
    file.write_all(toml_content.as_bytes()).unwrap();
    file.flush().unwrap();
    
    let contents = std::fs::read_to_string(file.path()).unwrap();
    let config: TieringConfig = toml::from_str(&contents).unwrap();
    
    assert_eq!(config.dram_threshold, 4_000_000_000);
    assert_eq!(config.high_water_mark, 0.88);
    assert_eq!(config.low_water_mark, 0.72);
    assert_eq!(config.hotness_threshold, 6);
}
