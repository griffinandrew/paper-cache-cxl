/*
 * Hardware Performance Counters for Memory Access Tracking
 * 
 * This module uses Linux perf_event to track actual hardware-level memory accesses
 * for hashmap structures with both DRAM and PMEM configurations.
 * 
 * Uses perf_event_open system call to measure:
 * - CPU cycles and instructions (IPC)
 * - Cache references and misses
 * - Memory loads and stores (when available)
 */

use perf_event::{Builder, Group, Counter};
use perf_event::events::Hardware;
use std::sync::Mutex;
use std::io;

/// Hardware performance measurement for a specific operation
#[derive(Debug, Clone, Default)]
pub struct HwPerfMeasurement {
    pub cycles: u64,              // CPU cycles
    pub instructions: u64,        // Instructions executed
    pub cache_references: u64,    // Total cache references
    pub cache_misses: u64,        // Cache misses
    pub mem_loads: u64,           // Memory load operations (if available)
    pub mem_stores: u64,          // Memory store operations (if available)
}

impl HwPerfMeasurement {
    pub fn ipc(&self) -> f64 {
        if self.cycles == 0 {
            0.0
        } else {
            self.instructions as f64 / self.cycles as f64
        }
    }

    pub fn cache_miss_rate(&self) -> f64 {
        if self.cache_references == 0 {
            0.0
        } else {
            (self.cache_misses as f64 / self.cache_references as f64) * 100.0
        }
    }

    pub fn total_mem_accesses(&self) -> u64 {
        self.mem_loads + self.mem_stores
    }
}

/// Performance counter group for measuring operations
pub struct PerfCounterGroup {
    group: Option<Group>,
    cycles_counter: Option<Counter>,
    instructions_counter: Option<Counter>,
    cache_refs_counter: Option<Counter>,
    cache_miss_counter: Option<Counter>,
}

impl PerfCounterGroup {
    /// Create a new performance counter group
    /// Returns a group with available counters (may be limited based on permissions/platform)
    pub fn new() -> Self {
        match Self::try_create_counters() {
            Ok((group, cycles, instructions, cache_refs, cache_miss)) => {
                PerfCounterGroup {
                    group: Some(group),
                    cycles_counter: Some(cycles),
                    instructions_counter: Some(instructions),
                    cache_refs_counter: Some(cache_refs),
                    cache_miss_counter: Some(cache_miss),
                }
            }
            Err(_) => {
                // Counters not available (insufficient permissions, virtualized environment, etc.)
                PerfCounterGroup {
                    group: None,
                    cycles_counter: None,
                    instructions_counter: None,
                    cache_refs_counter: None,
                    cache_miss_counter: None,
                }
            }
        }
    }

    fn try_create_counters() -> io::Result<(Group, Counter, Counter, Counter, Counter)> {
        let mut group = Group::new()?;
        
        let cycles = Builder::new()
            .group(&mut group)
            .kind(Hardware::CPU_CYCLES)
            .build()?;
        
        let instructions = Builder::new()
            .group(&mut group)
            .kind(Hardware::INSTRUCTIONS)
            .build()?;
        
        let cache_refs = Builder::new()
            .group(&mut group)
            .kind(Hardware::CACHE_REFERENCES)
            .build()?;
        
        let cache_miss = Builder::new()
            .group(&mut group)
            .kind(Hardware::CACHE_MISSES)
            .build()?;
        
        Ok((group, cycles, instructions, cache_refs, cache_miss))
    }

    /// Start measuring performance counters
    pub fn start(&mut self) -> Result<(), String> {
        if let Some(ref mut group) = self.group {
            group.enable().map_err(|e| format!("Failed to enable counters: {}", e))
        } else {
            Err("Performance counters not available".to_string())
        }
    }

    /// Stop measuring and return the results
    pub fn stop(&mut self) -> Result<HwPerfMeasurement, String> {
        if let Some(ref mut group) = self.group {
            group.disable().map_err(|e| format!("Failed to disable counters: {}", e))?;
            
            // Read counter values
            let cycles = self.cycles_counter.as_mut()
                .and_then(|c| c.read().ok())
                .unwrap_or(0);
            
            let instructions = self.instructions_counter.as_mut()
                .and_then(|c| c.read().ok())
                .unwrap_or(0);
            
            let cache_refs = self.cache_refs_counter.as_mut()
                .and_then(|c| c.read().ok())
                .unwrap_or(0);
            
            let cache_miss = self.cache_miss_counter.as_mut()
                .and_then(|c| c.read().ok())
                .unwrap_or(0);
            
            Ok(HwPerfMeasurement {
                cycles,
                instructions,
                cache_references: cache_refs,
                cache_misses: cache_miss,
                mem_loads: 0,   // Would need architecture-specific events
                mem_stores: 0,  // Would need architecture-specific events
            })
        } else {
            Err("Performance counters not available".to_string())
        }
    }

    /// Reset counters to zero
    pub fn reset(&mut self) -> Result<(), String> {
        if let Some(ref mut group) = self.group {
            group.reset().map_err(|e| format!("Failed to reset counters: {}", e))
        } else {
            Err("Performance counters not available".to_string())
        }
    }

    pub fn is_available(&self) -> bool {
        self.group.is_some()
    }
}

/// Hardware performance counters for hashmap operations
#[derive(Debug)]
pub struct HwHashMapCounters {
    // Accumulated measurements for each operation type
    pub get_measurements: Mutex<Vec<HwPerfMeasurement>>,
    pub set_measurements: Mutex<Vec<HwPerfMeasurement>>,
    pub del_measurements: Mutex<Vec<HwPerfMeasurement>>,
    pub has_measurements: Mutex<Vec<HwPerfMeasurement>>,
}

impl HwHashMapCounters {
    pub fn new() -> Self {
        HwHashMapCounters {
            get_measurements: Mutex::new(Vec::new()),
            set_measurements: Mutex::new(Vec::new()),
            del_measurements: Mutex::new(Vec::new()),
            has_measurements: Mutex::new(Vec::new()),
        }
    }

    pub fn record_get(&self, measurement: HwPerfMeasurement) {
        if let Ok(mut measurements) = self.get_measurements.lock() {
            measurements.push(measurement);
        }
    }

    pub fn record_set(&self, measurement: HwPerfMeasurement) {
        if let Ok(mut measurements) = self.set_measurements.lock() {
            measurements.push(measurement);
        }
    }

    pub fn record_del(&self, measurement: HwPerfMeasurement) {
        if let Ok(mut measurements) = self.del_measurements.lock() {
            measurements.push(measurement);
        }
    }

    pub fn record_has(&self, measurement: HwPerfMeasurement) {
        if let Ok(mut measurements) = self.has_measurements.lock() {
            measurements.push(measurement);
        }
    }

    /// Get aggregated statistics for all operations
    pub fn get_stats(&self) -> HwHashMapStats {
        HwHashMapStats {
            get: self.aggregate_measurements(&self.get_measurements),
            set: self.aggregate_measurements(&self.set_measurements),
            del: self.aggregate_measurements(&self.del_measurements),
            has: self.aggregate_measurements(&self.has_measurements),
        }
    }

    fn aggregate_measurements(&self, measurements: &Mutex<Vec<HwPerfMeasurement>>) -> AggregatedMeasurement {
        if let Ok(measurements) = measurements.lock() {
            if measurements.is_empty() {
                return AggregatedMeasurement::default();
            }

            let count = measurements.len() as u64;
            let total_cycles: u64 = measurements.iter().map(|m| m.cycles).sum();
            let total_instructions: u64 = measurements.iter().map(|m| m.instructions).sum();
            let total_cache_refs: u64 = measurements.iter().map(|m| m.cache_references).sum();
            let total_cache_misses: u64 = measurements.iter().map(|m| m.cache_misses).sum();
            let total_mem_loads: u64 = measurements.iter().map(|m| m.mem_loads).sum();
            let total_mem_stores: u64 = measurements.iter().map(|m| m.mem_stores).sum();

            AggregatedMeasurement {
                count,
                total_cycles,
                total_instructions,
                total_cache_refs,
                total_cache_misses,
                total_mem_loads,
                total_mem_stores,
                avg_cycles: total_cycles / count,
                avg_instructions: total_instructions / count,
                avg_cache_refs: total_cache_refs / count,
                avg_cache_misses: total_cache_misses / count,
                avg_mem_loads: total_mem_loads / count,
                avg_mem_stores: total_mem_stores / count,
            }
        } else {
            AggregatedMeasurement::default()
        }
    }
}

impl Default for HwHashMapCounters {
    fn default() -> Self {
        Self::new()
    }
}

/// Aggregated measurement statistics
#[derive(Debug, Clone, Default)]
pub struct AggregatedMeasurement {
    pub count: u64,
    pub total_cycles: u64,
    pub total_instructions: u64,
    pub total_cache_refs: u64,
    pub total_cache_misses: u64,
    pub total_mem_loads: u64,
    pub total_mem_stores: u64,
    pub avg_cycles: u64,
    pub avg_instructions: u64,
    pub avg_cache_refs: u64,
    pub avg_cache_misses: u64,
    pub avg_mem_loads: u64,
    pub avg_mem_stores: u64,
}

impl AggregatedMeasurement {
    pub fn total_mem_accesses(&self) -> u64 {
        self.total_mem_loads + self.total_mem_stores
    }

    pub fn avg_mem_accesses(&self) -> u64 {
        self.avg_mem_loads + self.avg_mem_stores
    }

    pub fn cache_miss_rate(&self) -> f64 {
        if self.total_cache_refs == 0 {
            0.0
        } else {
            (self.total_cache_misses as f64 / self.total_cache_refs as f64) * 100.0
        }
    }

    pub fn avg_ipc(&self) -> f64 {
        if self.avg_cycles == 0 {
            0.0
        } else {
            self.avg_instructions as f64 / self.avg_cycles as f64
        }
    }
}

/// Complete statistics for all hashmap operations
#[derive(Debug, Clone)]
pub struct HwHashMapStats {
    pub get: AggregatedMeasurement,
    pub set: AggregatedMeasurement,
    pub del: AggregatedMeasurement,
    pub has: AggregatedMeasurement,
}

impl HwHashMapStats {
    pub fn total_operations(&self) -> u64 {
        self.get.count + self.set.count + self.del.count + self.has.count
    }

    pub fn total_cycles(&self) -> u64 {
        self.get.total_cycles + 
        self.set.total_cycles + 
        self.del.total_cycles + 
        self.has.total_cycles
    }

    pub fn total_cache_refs(&self) -> u64 {
        self.get.total_cache_refs + 
        self.set.total_cache_refs + 
        self.del.total_cache_refs + 
        self.has.total_cache_refs
    }

    pub fn total_cache_misses(&self) -> u64 {
        self.get.total_cache_misses + 
        self.set.total_cache_misses + 
        self.del.total_cache_misses + 
        self.has.total_cache_misses
    }
}

impl std::fmt::Display for HwHashMapStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Hardware Performance Statistics (HashMap):")?;
        writeln!(f, "  Total Operations: {}", self.total_operations())?;
        writeln!(f, "  Total Cycles: {}", self.total_cycles())?;
        writeln!(f, "  Total Cache References: {}", self.total_cache_refs())?;
        writeln!(f, "  Total Cache Misses: {} ({:.2}% miss rate)", 
                 self.total_cache_misses(),
                 if self.total_cache_refs() > 0 {
                     100.0 * self.total_cache_misses() as f64 / self.total_cache_refs() as f64
                 } else { 0.0 })?;
        writeln!(f)?;
        
        if self.get.count > 0 {
            writeln!(f, "GET Operations ({} calls):", self.get.count)?;
            writeln!(f, "    Avg Cycles: {}", self.get.avg_cycles)?;
            writeln!(f, "    Avg Instructions: {} (IPC: {:.2})", self.get.avg_instructions, self.get.avg_ipc())?;
            writeln!(f, "    Avg Cache References: {}", self.get.avg_cache_refs)?;
            writeln!(f, "    Avg Cache Misses: {} ({:.2}% miss rate)", 
                     self.get.avg_cache_misses, self.get.cache_miss_rate())?;
            writeln!(f)?;
        }
        
        if self.set.count > 0 {
            writeln!(f, "SET Operations ({} calls):", self.set.count)?;
            writeln!(f, "    Avg Cycles: {}", self.set.avg_cycles)?;
            writeln!(f, "    Avg Instructions: {} (IPC: {:.2})", self.set.avg_instructions, self.set.avg_ipc())?;
            writeln!(f, "    Avg Cache References: {}", self.set.avg_cache_refs)?;
            writeln!(f, "    Avg Cache Misses: {} ({:.2}% miss rate)", 
                     self.set.avg_cache_misses, self.set.cache_miss_rate())?;
            writeln!(f)?;
        }
        
        if self.del.count > 0 {
            writeln!(f, "DEL Operations ({} calls):", self.del.count)?;
            writeln!(f, "    Avg Cycles: {}", self.del.avg_cycles)?;
            writeln!(f, "    Avg Instructions: {} (IPC: {:.2})", self.del.avg_instructions, self.del.avg_ipc())?;
            writeln!(f, "    Avg Cache References: {}", self.del.avg_cache_refs)?;
            writeln!(f, "    Avg Cache Misses: {} ({:.2}% miss rate)", 
                     self.del.avg_cache_misses, self.del.cache_miss_rate())?;
            writeln!(f)?;
        }
        
        if self.has.count > 0 {
            writeln!(f, "HAS Operations ({} calls):", self.has.count)?;
            writeln!(f, "    Avg Cycles: {}", self.has.avg_cycles)?;
            writeln!(f, "    Avg Instructions: {} (IPC: {:.2})", self.has.avg_instructions, self.has.avg_ipc())?;
            writeln!(f, "    Avg Cache References: {}", self.has.avg_cache_refs)?;
            writeln!(f, "    Avg Cache Misses: {} ({:.2}% miss rate)", 
                     self.has.avg_cache_misses, self.has.cache_miss_rate())?;
        }
        
        Ok(())
    }
}

/// Global hardware performance counters for the cache
#[derive(Debug)]
pub struct GlobalHwPerfCounters {
    #[cfg(feature = "hashbrown_dram")]
    pub global_hashbrown_dram: HwHashMapCounters,
    
    #[cfg(feature = "global_hashtable_pmem")]
    pub global_hashbrown_pmem: HwHashMapCounters,
    
    #[cfg(feature = "global_flatmap_dram")]
    pub global_flatmap_dram: HwHashMapCounters,
    
    #[cfg(feature = "global_flatmap_pmem")]
    pub global_flatmap_pmem: HwHashMapCounters,
}

impl GlobalHwPerfCounters {
    pub fn new() -> Self {
        GlobalHwPerfCounters {
            #[cfg(feature = "hashbrown_dram")]
            global_hashbrown_dram: HwHashMapCounters::new(),
            
            #[cfg(feature = "global_hashtable_pmem")]
            global_hashbrown_pmem: HwHashMapCounters::new(),
            
            #[cfg(feature = "global_flatmap_dram")]
            global_flatmap_dram: HwHashMapCounters::new(),
            
            #[cfg(feature = "global_flatmap_pmem")]
            global_flatmap_pmem: HwHashMapCounters::new(),
        }
    }
}

impl Default for GlobalHwPerfCounters {
    fn default() -> Self {
        Self::new()
    }
}

/// Global instance of hardware performance counters
use std::sync::OnceLock;

static HW_PERF_COUNTERS: OnceLock<GlobalHwPerfCounters> = OnceLock::new();

/// Get the global hardware performance counters instance
pub fn get_hw_counters() -> &'static GlobalHwPerfCounters {
    HW_PERF_COUNTERS.get_or_init(GlobalHwPerfCounters::new)
}

/// Get hardware statistics for the active hashmap configuration
pub fn get_hw_hashmap_stats() -> Option<HwHashMapStats> {
    let counters = get_hw_counters();
    
    #[cfg(feature = "hashbrown_dram")]
    {
        return Some(counters.global_hashbrown_dram.get_stats());
    }
    
    #[cfg(all(feature = "global_hashtable_pmem", not(feature = "hashbrown_dram")))]
    {
        return Some(counters.global_hashbrown_pmem.get_stats());
    }
    
    #[cfg(all(feature = "global_flatmap_dram", not(feature = "hashbrown_dram"), not(feature = "global_hashtable_pmem")))]
    {
        return Some(counters.global_flatmap_dram.get_stats());
    }
    
    #[cfg(all(feature = "global_flatmap_pmem", not(feature = "hashbrown_dram"), not(feature = "global_hashtable_pmem"), not(feature = "global_flatmap_dram")))]
    {
        return Some(counters.global_flatmap_pmem.get_stats());
    }
    
    #[cfg(not(any(feature = "hashbrown_dram", feature = "global_hashtable_pmem", feature = "global_flatmap_dram", feature = "global_flatmap_pmem")))]
    {
        None
    }
}

/// Print hardware performance statistics
pub fn print_hw_perf_stats() {
    println!("\n=== Hardware Performance Counter Statistics ===\n");
    
    if let Some(stats) = get_hw_hashmap_stats() {
        #[cfg(feature = "hashbrown_dram")]
        println!("Global HashMap (hashbrown in DRAM):");
        
        #[cfg(all(feature = "global_hashtable_pmem", not(feature = "hashbrown_dram")))]
        println!("Global HashMap (hashbrown in PMEM):");
        
        #[cfg(all(feature = "global_flatmap_dram", not(feature = "hashbrown_dram"), not(feature = "global_hashtable_pmem")))]
        println!("Global HashMap (FlatMap in DRAM):");
        
        #[cfg(all(feature = "global_flatmap_pmem", not(feature = "hashbrown_dram"), not(feature = "global_hashtable_pmem"), not(feature = "global_flatmap_dram")))]
        println!("Global HashMap (FlatMap in PMEM):");
        
        print!("{}", stats);
    } else {
        println!("No hardware performance counters available");
    }
    
    println!("\n===============================================\n");
}

/// Thread-local performance counter for measuring individual operations
use std::cell::RefCell;

thread_local! {
    static PERF_COUNTER: RefCell<PerfCounterGroup> = RefCell::new(PerfCounterGroup::new());
}

/// Measure a hashmap operation with hardware performance counters
/// Returns the operation result and optionally the hardware measurements
pub fn measure_operation<F, R>(operation: F) -> (R, Option<HwPerfMeasurement>)
where
    F: FnOnce() -> R,
{
    PERF_COUNTER.with(|counter_cell| {
        let mut counter = counter_cell.borrow_mut();
        
        if !counter.is_available() {
            // Counters not available, just run the operation
            return (operation(), None);
        }
        
        // Reset and start counting
        let _ = counter.reset();
        if counter.start().is_err() {
            return (operation(), None);
        }
        
        // Run the operation
        let result = operation();
        
        // Stop counting and get measurements
        let measurement = counter.stop().ok();
        
        (result, measurement)
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_perf_measurement() {
        let measurement = HwPerfMeasurement {
            cycles: 1000,
            instructions: 2000,
            cache_references: 150,
            cache_misses: 15,
            mem_loads: 100,
            mem_stores: 50,
        };

        assert_eq!(measurement.total_mem_accesses(), 150);
        assert_eq!(measurement.cache_miss_rate(), 10.0);
        assert_eq!(measurement.ipc(), 2.0);
    }

    #[test]
    fn test_aggregated_measurement() {
        let agg = AggregatedMeasurement {
            count: 10,
            total_cycles: 10000,
            total_instructions: 20000,
            total_cache_refs: 1500,
            total_cache_misses: 150,
            total_mem_loads: 1000,
            total_mem_stores: 500,
            avg_cycles: 1000,
            avg_instructions: 2000,
            avg_cache_refs: 150,
            avg_cache_misses: 15,
            avg_mem_loads: 100,
            avg_mem_stores: 50,
        };

        assert_eq!(agg.total_mem_accesses(), 1500);
        assert_eq!(agg.avg_mem_accesses(), 150);
        assert_eq!(agg.cache_miss_rate(), 10.0);
        assert_eq!(agg.avg_ipc(), 2.0);
    }

    #[test]
    fn test_measure_operation() {
        let (result, _measurement) = measure_operation(|| {
            let mut sum = 0;
            for i in 0..1000 {
                sum += i;
            }
            sum
        });

        assert_eq!(result, 499500);
        // measurement may be None if perf counters not available
    }
}
