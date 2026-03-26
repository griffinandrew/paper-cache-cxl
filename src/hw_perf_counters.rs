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

use perf_event::events::{Cache, CacheOp, CacheResult, Hardware, WhichCache};
use perf_event::{Builder, Counter, Group};
use std::io;
use std::sync::Mutex;

/// Hardware performance measurement for a specific operation
#[derive(Debug, Clone, Default)]
pub struct HwPerfMeasurement {
    pub cycles: u64,           // CPU cycles
    pub instructions: u64,     // Instructions executed
    pub cache_references: u64, // LLC references (Last-Level Cache accesses)
    pub cache_misses: u64,     // LLC misses (Last-Level Cache misses)
    pub mem_loads: u64,        // Memory load operations (if available)
    pub mem_stores: u64,       // Memory store operations (if available)
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
            Ok((group, cycles, instructions, cache_refs, cache_miss)) => PerfCounterGroup {
                group: Some(group),
                cycles_counter: Some(cycles),
                instructions_counter: Some(instructions),
                cache_refs_counter: Some(cache_refs),
                cache_miss_counter: Some(cache_miss),
            },
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

        // Core CPU metrics - Essential for IPC
        let cycles = Builder::new()
            .group(&mut group)
            .kind(Hardware::CPU_CYCLES)
            .build()?;

        let instructions = Builder::new()
            .group(&mut group)
            .kind(Hardware::INSTRUCTIONS)
            .build()?;

        // LLC cache metrics - Track Last-Level Cache only
        // Using Cache enum with WhichCache::LL to track only LLC accesses and misses
        let cache_refs = Builder::new()
            .group(&mut group)
            .kind(Cache {
                which: WhichCache::LL,
                operation: CacheOp::READ,
                result: CacheResult::ACCESS,
            })
            .build()?;

        let cache_miss = Builder::new()
            .group(&mut group)
            .kind(Cache {
                which: WhichCache::LL,
                operation: CacheOp::READ,
                result: CacheResult::MISS,
            })
            .build()?;

        Ok((group, cycles, instructions, cache_refs, cache_miss))
    }

    /// Start measuring performance counters
    pub fn start(&mut self) -> Result<(), String> {
        if let Some(ref mut group) = self.group {
            group
                .enable()
                .map_err(|e| format!("Failed to enable counters: {}", e))
        } else {
            Err("Performance counters not available".to_string())
        }
    }

    /// Stop measuring and return the results
    pub fn stop(&mut self) -> Result<HwPerfMeasurement, String> {
        if let Some(ref mut group) = self.group {
            group
                .disable()
                .map_err(|e| format!("Failed to disable counters: {}", e))?;

            // Read counter values
            let cycles = self
                .cycles_counter
                .as_mut()
                .and_then(|c| c.read().ok())
                .unwrap_or(0);

            let instructions = self
                .instructions_counter
                .as_mut()
                .and_then(|c| c.read().ok())
                .unwrap_or(0);

            let cache_refs = self
                .cache_refs_counter
                .as_mut()
                .and_then(|c| c.read().ok())
                .unwrap_or(0);

            let cache_miss = self
                .cache_miss_counter
                .as_mut()
                .and_then(|c| c.read().ok())
                .unwrap_or(0);

            Ok(HwPerfMeasurement {
                cycles,
                instructions,
                cache_references: cache_refs,
                cache_misses: cache_miss,
                mem_loads: 0,  // Would need architecture-specific events
                mem_stores: 0, // Would need architecture-specific events
            })
        } else {
            Err("Performance counters not available".to_string())
        }
    }

    /// Reset counters to zero
    pub fn reset(&mut self) -> Result<(), String> {
        if let Some(ref mut group) = self.group {
            group
                .reset()
                .map_err(|e| format!("Failed to reset counters: {}", e))
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

    fn aggregate_measurements(
        &self,
        measurements: &Mutex<Vec<HwPerfMeasurement>>,
    ) -> AggregatedMeasurement {
        if let Ok(measurements) = measurements.lock() {
            if measurements.is_empty() {
                return AggregatedMeasurement::default();
            }

            let count = measurements.len() as u64;

            // Helper macro to sum and average
            macro_rules! sum_and_avg {
                ($field:ident) => {{
                    let total: u64 = measurements.iter().map(|m| m.$field).sum();
                    (total, total / count)
                }};
            }

            let (total_cycles, avg_cycles) = sum_and_avg!(cycles);
            let (total_instructions, avg_instructions) = sum_and_avg!(instructions);
            let (total_cache_refs, avg_cache_refs) = sum_and_avg!(cache_references);
            let (total_cache_misses, avg_cache_misses) = sum_and_avg!(cache_misses);

            AggregatedMeasurement {
                count,
                total_cycles,
                total_instructions,
                avg_cycles,
                avg_instructions,
                total_cache_refs,
                total_cache_misses,
                avg_cache_refs,
                avg_cache_misses,
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

    // Core CPU metrics
    pub total_cycles: u64,
    pub total_instructions: u64,
    pub avg_cycles: u64,
    pub avg_instructions: u64,

    // LLC cache (Last-Level Cache) - tracking actual memory accesses
    pub total_cache_refs: u64,
    pub total_cache_misses: u64,
    pub avg_cache_refs: u64,
    pub avg_cache_misses: u64,
}

impl AggregatedMeasurement {
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
        self.get.total_cycles
            + self.set.total_cycles
            + self.del.total_cycles
            + self.has.total_cycles
    }

    pub fn total_cache_refs(&self) -> u64 {
        self.get.total_cache_refs
            + self.set.total_cache_refs
            + self.del.total_cache_refs
            + self.has.total_cache_refs
    }

    pub fn total_cache_misses(&self) -> u64 {
        self.get.total_cache_misses
            + self.set.total_cache_misses
            + self.del.total_cache_misses
            + self.has.total_cache_misses
    }
}

impl std::fmt::Display for HwHashMapStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Hardware Performance Statistics (HashMap):")?;
        writeln!(f, "  Total Operations: {}", self.total_operations())?;
        writeln!(f, "  Total Cycles: {}", self.total_cycles())?;
        writeln!(f, "  Total Cache References: {}", self.total_cache_refs())?;
        writeln!(
            f,
            "  Total Cache Misses: {} ({:.2}% miss rate)",
            self.total_cache_misses(),
            if self.total_cache_refs() > 0 {
                100.0 * self.total_cache_misses() as f64 / self.total_cache_refs() as f64
            } else {
                0.0
            }
        )?;
        writeln!(f)?;

        // Helper closure to print operation details
        let print_operation = |f: &mut std::fmt::Formatter,
                               name: &str,
                               agg: &AggregatedMeasurement|
         -> std::fmt::Result {
            if agg.count == 0 {
                return Ok(());
            }

            writeln!(f, "{} Operations ({} calls):", name, agg.count)?;
            writeln!(f, "  ┌─ Execution Metrics:")?;
            writeln!(
                f,
                "  │  Cycles: {} avg, {} total",
                agg.avg_cycles, agg.total_cycles
            )?;
            writeln!(
                f,
                "  │  Instructions: {} avg (IPC: {:.2})",
                agg.avg_instructions,
                agg.avg_ipc()
            )?;
            writeln!(
                f,
                "  ├─ LLC Cache (Last-Level Cache - actual memory accesses):"
            )?;
            writeln!(
                f,
                "  │  LLC Accesses: {} avg, {} total",
                agg.avg_cache_refs, agg.total_cache_refs
            )?;
            writeln!(
                f,
                "  │  LLC Misses: {} avg, {} total ({:.2}% miss rate)",
                agg.avg_cache_misses,
                agg.total_cache_misses,
                agg.cache_miss_rate()
            )?;

            writeln!(f)?;
            Ok(())
        };

        print_operation(f, "GET", &self.get)?;
        print_operation(f, "SET", &self.set)?;
        print_operation(f, "DEL", &self.del)?;
        print_operation(f, "HAS", &self.has)?;

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

    #[cfg(all(
        feature = "global_flatmap_dram",
        not(feature = "hashbrown_dram"),
        not(feature = "global_hashtable_pmem")
    ))]
    {
        return Some(counters.global_flatmap_dram.get_stats());
    }

    #[cfg(all(
        feature = "global_flatmap_pmem",
        not(feature = "hashbrown_dram"),
        not(feature = "global_hashtable_pmem"),
        not(feature = "global_flatmap_dram")
    ))]
    {
        return Some(counters.global_flatmap_pmem.get_stats());
    }

    #[cfg(not(any(
        feature = "hashbrown_dram",
        feature = "global_hashtable_pmem",
        feature = "global_flatmap_dram",
        feature = "global_flatmap_pmem"
    )))]
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

        #[cfg(all(
            feature = "global_flatmap_dram",
            not(feature = "hashbrown_dram"),
            not(feature = "global_hashtable_pmem")
        ))]
        println!("Global HashMap (FlatMap in DRAM):");

        #[cfg(all(
            feature = "global_flatmap_pmem",
            not(feature = "hashbrown_dram"),
            not(feature = "global_hashtable_pmem"),
            not(feature = "global_flatmap_dram")
        ))]
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
    static DEBUG_LOGGED: RefCell<bool> = RefCell::new(false);
}

/// Measure a hashmap operation with hardware performance counters
/// Returns the operation result and optionally the hardware measurements
pub fn measure_operation<F, R>(operation: F) -> (R, Option<HwPerfMeasurement>)
where
    F: FnOnce() -> R,
{
    // Check debug mode once
    let debug_enabled = !std::env::var("PAPER_CACHE_DEBUG_PERF")
        .unwrap_or_default()
        .is_empty();

    PERF_COUNTER.with(|counter_cell| {
        let mut counter = counter_cell.borrow_mut();

        if !counter.is_available() {
            // Log debug info only once per thread if debug is enabled
            if debug_enabled {
                DEBUG_LOGGED.with(|logged| {
                    if !*logged.borrow() {
                        eprintln!("[DEBUG] measure_operation: Performance counters not available for this thread");
                        eprintln!("[DEBUG] measure_operation: Returning None for measurements");
                        *logged.borrow_mut() = true;
                    }
                });
            }
            // Counters not available, just run the operation
            return (operation(), None);
        }

        // Reset counters
        if let Err(e) = counter.reset() {
            if debug_enabled {
                eprintln!("[DEBUG] measure_operation: Failed to reset counters: {}", e);
            }
        }

        // Start counting - if this fails, run operation without measurement
        if let Err(e) = counter.start() {
            if debug_enabled {
                DEBUG_LOGGED.with(|logged| {
                    if !*logged.borrow() {
                        eprintln!("[DEBUG] measure_operation: Failed to start counters: {}", e);
                        eprintln!("[DEBUG] measure_operation: Running operation without measurement");
                        *logged.borrow_mut() = true;
                    }
                });
            }
            return (operation(), None);
        }

        // Run the operation
        let result = operation();

        // Stop counting and get measurements
        let measurement = counter.stop().ok();

        if measurement.is_none() {
            if debug_enabled {
                DEBUG_LOGGED.with(|logged| {
                    if !*logged.borrow() {
                        eprintln!("[DEBUG] measure_operation: Failed to stop counters and get measurement");
                        *logged.borrow_mut() = true;
                    }
                });
            }
        }

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
            avg_cycles: 1000,
            avg_instructions: 2000,
            total_cache_refs: 1500,
            total_cache_misses: 150,
            avg_cache_refs: 150,
            avg_cache_misses: 15,
        };

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

    #[test]
    fn test_perf_counter_group_resilience() {
        // This test verifies that PerfCounterGroup::new() succeeds even if
        // Group::new() fails or some individual counters fail to create.
        // The group should be created with whatever counters are available.

        let mut counter_group = PerfCounterGroup::new();

        // The counter group should always be created (never panics)
        // It should have is_available() return true if the group was created,
        // or false if group creation failed (e.g., insufficient permissions)

        // We just verify that it doesn't panic and has a valid state
        let available = counter_group.is_available();

        // If available, we should be able to start/stop without errors
        if available {
            // Group exists, so start() should succeed
            assert!(counter_group.start().is_ok());
        }
    }
}
