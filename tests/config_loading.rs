// Test config file loading functionality using the real TieringConfig
// These tests require the allocator_api feature
#![cfg(feature = "allocator_api")]

use paper_cache::TieringConfig;
use std::io::Write;
use tempfile::NamedTempFile;

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
    
    assert_eq!(deserialized.dram_threshold, config.dram_threshold);
    assert_eq!(deserialized.high_water_mark, config.high_water_mark);
    assert_eq!(deserialized.low_water_mark, config.low_water_mark);
    assert_eq!(deserialized.hotness_threshold, config.hotness_threshold);
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
    
    assert_eq!(deserialized.dram_threshold, config.dram_threshold);
    assert_eq!(deserialized.high_water_mark, config.high_water_mark);
    assert_eq!(deserialized.low_water_mark, config.low_water_mark);
    assert_eq!(deserialized.hotness_threshold, config.hotness_threshold);
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
    
    let config = TieringConfig::from_json_file(file.path()).unwrap();
    
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
    
    let config = TieringConfig::from_toml_file(file.path()).unwrap();
    
    assert_eq!(config.dram_threshold, 4_000_000_000);
    assert_eq!(config.high_water_mark, 0.88);
    assert_eq!(config.low_water_mark, 0.72);
    assert_eq!(config.hotness_threshold, 6);
}
