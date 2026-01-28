/*
 * Hardware Performance Counters for Memory Access Tracking
 * 
 * This module uses Linux perf_event to track actual hardware-level memory accesses
 * with focus on LLC (Last Level Cache) specific events.
 * 
 * Uses perf_event_open system call to measure:
 * - CPU cycles and instructions (IPC)
 * - LLC (Last Level Cache) loads and load misses
 * - General cache references and misses
 */

use perf_event::{Builder, Group, Counter};
use perf_event::events::{Hardware, Cache, WhichCache, CacheOp, CacheResult};
use std::sync::Mutex;
use std::io;
use std::time::{SystemTime, UNIX_EPOCH};

/// Hardware performance measurement for a specific operation
#[derive(Debug, Clone, Default)]
pub struct HwPerfMeasurement {
    pub duration_ns: u64,         // Duration in nanoseconds
    pub timestamp_ns: u128,       // Timestamp when measurement was taken
    
    // Core CPU metrics
    pub cycles: u64,              // CPU cycles
    pub instructions: u64,        // Instructions executed
    pub ref_cpu_cycles: u64,      // Reference CPU cycles (if available)
    
    // Branch metrics
    pub branch_instructions: u64, // Branch instructions
    pub branch_misses: u64,       // Branch prediction misses
    
    // Pipeline stalls
    pub stalled_cycles_frontend: u64,  // Frontend stall cycles
    pub stalled_cycles_backend: u64,   // Backend stall cycles
    
    // Cache metrics (general)
    pub cache_references: u64,    // Total cache references
    pub cache_misses: u64,        // Total cache misses
    
    // L1 D-cache
    pub l1_dcache_loads: u64,
    pub l1_dcache_load_misses: u64,
    pub l1_dcache_stores: u64,
    pub l1_dcache_store_misses: u64,
    
    // L1 I-cache
    pub l1_icache_loads: u64,
    pub l1_icache_load_misses: u64,
    
    // LLC (Last Level Cache) - This is what we focus on
    pub llc_loads: u64,
    pub llc_load_misses: u64,
    pub llc_stores: u64,
    pub llc_store_misses: u64,
    
    // TLB metrics
    pub dtlb_loads: u64,
    pub dtlb_load_misses: u64,
    pub dtlb_stores: u64,
    pub dtlb_store_misses: u64,
    pub itlb_loads: u64,
    pub itlb_load_misses: u64,
    
    // Page faults
    pub page_faults: u64,
    pub page_faults_min: u64,
    pub page_faults_maj: u64,
    
    // Context switches
    pub context_switches: u64,
    pub cpu_migrations: u64,
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
    
    pub fn llc_miss_rate(&self) -> f64 {
        if self.llc_loads == 0 {
            0.0
        } else {
            (self.llc_load_misses as f64 / self.llc_loads as f64) * 100.0
        }
    }
}

/// Performance counter group for measuring operations
/// Limited to 6-8 essential counters to avoid multiplexing
pub struct PerfCounterGroup {
    group: Option<Group>,
    
    // Essential counters (6 total - matching user's configuration)
    cycles_counter: Option<Counter>,
    instructions_counter: Option<Counter>,
    cache_refs_counter: Option<Counter>,
    cache_miss_counter: Option<Counter>,
    llc_loads_counter: Option<Counter>,
    llc_load_misses_counter: Option<Counter>,
    
    start_time: Option<std::time::Instant>,
}

impl PerfCounterGroup {
    /// Create a new performance counter group with essential counters only
    /// Returns a group with available counters (limited to avoid multiplexing)
    pub fn new() -> Self {
        eprintln!("[DEBUG] Attempting to create performance counter group...");
        
        match Self::try_create_essential_counters() {
            Ok((group, cycles, instructions, cache_refs, cache_miss, llc_loads, llc_load_misses)) => {
                eprintln!("[DEBUG] Successfully created performance counter group");
                eprintln!("[DEBUG] Performance counter creation summary:");
                eprintln!("[DEBUG] ");
                eprintln!("[DEBUG] === ESSENTIAL COUNTERS (6 total) ===");
                
                eprintln!("[DEBUG]   ✓ CPU_CYCLES");
                eprintln!("[DEBUG]   ✓ INSTRUCTIONS");
                eprintln!("[DEBUG]   ✓ CACHE_REFERENCES");
                eprintln!("[DEBUG]   ✓ CACHE_MISSES");
                eprintln!("[DEBUG]   ✓ LLC_LOADS (WhichCache::LL)");
                eprintln!("[DEBUG]   ✓ LLC_LOAD_MISSES (WhichCache::LL)");
                
                eprintln!("[DEBUG] ");
                eprintln!("[DEBUG] Counter creation complete: 6 enabled");
                eprintln!("[DEBUG] Note: Limited to 6 counters to stay within hardware limits and avoid multiplexing");
                
                PerfCounterGroup {
                    group: Some(group),
                    cycles_counter: Some(cycles),
                    instructions_counter: Some(instructions),
                    cache_refs_counter: Some(cache_refs),
                    cache_miss_counter: Some(cache_miss),
                    llc_loads_counter: Some(llc_loads),
                    llc_load_misses_counter: Some(llc_load_misses),
                    start_time: None,
                }
            }
            Err(e) => {
                eprintln!("[DEBUG] Failed to create performance counters: {:?}", e);
                eprintln!("[DEBUG] Counters not available (insufficient permissions, virtualized environment, etc.)");
                PerfCounterGroup {
                    group: None,
                    cycles_counter: None,
                    instructions_counter: None,
                    cache_refs_counter: None,
                    cache_miss_counter: None,
                    llc_loads_counter: None,
                    llc_load_misses_counter: None,
                    start_time: None,
                }
            }
        }
    }

    fn try_create_essential_counters() -> io::Result<(Group, Counter, Counter, Counter, Counter, Counter, Counter)> {
        // Create counter group - this must be done BEFORE creating any counters
        let mut group = Group::new()?;
        
        // Essential counter 1: CPU Cycles
        let cycles = Builder::new()
            .group(&mut group)
            .kind(Hardware::CPU_CYCLES)
            .build()?;
        
        // Essential counter 2: Instructions
        let instructions = Builder::new()
            .group(&mut group)
            .kind(Hardware::INSTRUCTIONS)
            .build()?;
        
        // Essential counter 3: Cache References (all levels)
        let cache_refs = Builder::new()
            .group(&mut group)
            .kind(Hardware::CACHE_REFERENCES)
            .build()?;
        
        // Essential counter 4: Cache Misses (all levels)
        let cache_miss = Builder::new()
            .group(&mut group)
            .kind(Hardware::CACHE_MISSES)
            .build()?;
        
        // Essential counter 5: LLC Loads (Last Level Cache - Read operations)
        // Use WhichCache::LL to specify Last Level Cache
        let llc_loads = Builder::new()
            .group(&mut group)
            .kind(Cache {
                which: WhichCache::LL,    // Last Level Cache
                operation: CacheOp::READ,  // Load operations
                result: CacheResult::ACCESS, // All accesses
            })
            .build()?;
        
        // Essential counter 6: LLC Load Misses (Last Level Cache - Read misses)
        let llc_load_misses = Builder::new()
            .group(&mut group)
            .kind(Cache {
                which: WhichCache::LL,    // Last Level Cache
                operation: CacheOp::READ,  // Load operations
                result: CacheResult::MISS, // Only misses
            })
            .build()?;
        
        Ok((group, cycles, instructions, cache_refs, cache_miss, llc_loads, llc_load_misses))
    }

    /// Start measuring performance counters
    pub fn start(&mut self) -> Result<(), String> {
        if let Some(ref mut group) = self.group {
            // Reset all counters to zero before starting
            group.reset().map_err(|e| format!("Failed to reset counters: {}", e))?;
            
            // Enable the counter group
            group.enable().map_err(|e| format!("Failed to enable counters: {}", e))?;
            
            // Record start time for duration calculation
            self.start_time = Some(std::time::Instant::now());
            
            eprintln!("[DEBUG] Performance counters started");
            Ok(())
        } else {
            Err("Performance counters not available".to_string())
        }
    }

    /// Stop measuring and return the results
    pub fn stop(&mut self) -> Result<HwPerfMeasurement, String> {
        let duration_ns = self.start_time
            .map(|start| start.elapsed().as_nanos() as u64)
            .unwrap_or(0);
        
        let timestamp_ns = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos();
        
        if let Some(ref mut group) = self.group {
            // Disable the counter group first to stop counting
            group.disable().map_err(|e| format!("Failed to disable counters: {}", e))?;
            
            eprintln!("[DEBUG] Performance counters stopped, reading values...");
            
            // Read counter values - IMPORTANT: Must read AFTER disabling
            let cycles = if let Some(ref mut c) = self.cycles_counter {
                match c.read() {
                    Ok(val) => {
                        eprintln!("[DEBUG] Read cycles: {}", val);
                        val
                    },
                    Err(e) => {
                        eprintln!("[DEBUG] Failed to read cycles counter: {:?}", e);
                        0
                    }
                }
            } else {
                0
            };
            
            let instructions = if let Some(ref mut c) = self.instructions_counter {
                match c.read() {
                    Ok(val) => {
                        eprintln!("[DEBUG] Read instructions: {}", val);
                        val
                    },
                    Err(e) => {
                        eprintln!("[DEBUG] Failed to read instructions counter: {:?}", e);
                        0
                    }
                }
            } else {
                0
            };
            
            let cache_refs = if let Some(ref mut c) = self.cache_refs_counter {
                match c.read() {
                    Ok(val) => {
                        eprintln!("[DEBUG] Read cache_refs: {}", val);
                        val
                    },
                    Err(e) => {
                        eprintln!("[DEBUG] Failed to read cache_refs counter: {:?}", e);
                        0
                    }
                }
            } else {
                0
            };
            
            let cache_miss = if let Some(ref mut c) = self.cache_miss_counter {
                match c.read() {
                    Ok(val) => {
                        eprintln!("[DEBUG] Read cache_miss: {}", val);
                        val
                    },
                    Err(e) => {
                        eprintln!("[DEBUG] Failed to read cache_miss counter: {:?}", e);
                        0
                    }
                }
            } else {
                0
            };
            
            let llc_loads = if let Some(ref mut c) = self.llc_loads_counter {
                match c.read() {
                    Ok(val) => {
                        eprintln!("[DEBUG] Read llc_loads: {}", val);
                        val
                    },
                    Err(e) => {
                        eprintln!("[DEBUG] Failed to read llc_loads counter: {:?}", e);
                        0
                    }
                }
            } else {
                0
            };
            
            let llc_load_misses = if let Some(ref mut c) = self.llc_load_misses_counter {
                match c.read() {
                    Ok(val) => {
                        eprintln!("[DEBUG] Read llc_load_misses: {}", val);
                        val
                    },
                    Err(e) => {
                        eprintln!("[DEBUG] Failed to read llc_load_misses counter: {:?}", e);
                        0
                    }
                }
            } else {
                0
            };
            
            eprintln!("[DEBUG] Summary: cycles={}, instructions={}, cache_refs={}, cache_miss={}, llc_loads={}, llc_load_misses={}", 
                     cycles, instructions, cache_refs, cache_miss, llc_loads, llc_load_misses);
            
            Ok(HwPerfMeasurement {
                duration_ns,
                timestamp_ns,
                cycles,
                instructions,
                cache_references: cache_refs,
                cache_misses: cache_miss,
                llc_loads,
                llc_load_misses,
                // All other fields default to 0 since we're only tracking essentials
                ..Default::default()
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

impl Default for PerfCounterGroup {
    fn default() -> Self {
        Self::new()
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
            let total_llc_loads: u64 = measurements.iter().map(|m| m.llc_loads).sum();
            let total_llc_load_misses: u64 = measurements.iter().map(|m| m.llc_load_misses).sum();

            AggregatedMeasurement {
                count,
                total_cycles,
                total_instructions,
                total_cache_refs,
                total_cache_misses,
                total_llc_loads,
                total_llc_load_misses,
                avg_cycles: total_cycles / count,
                avg_instructions: total_instructions / count,
                avg_cache_refs: total_cache_refs / count,
                avg_cache_misses: total_cache_misses / count,
                avg_llc_loads: total_llc_loads / count,
                avg_llc_load_misses: total_llc_load_misses / count,
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
    pub total_llc_loads: u64,
    pub total_llc_load_misses: u64,
    pub avg_cycles: u64,
    pub avg_instructions: u64,
    pub avg_cache_refs: u64,
    pub avg_cache_misses: u64,
    pub avg_llc_loads: u64,
    pub avg_llc_load_misses: u64,
}

impl AggregatedMeasurement {
    pub fn cache_miss_rate(&self) -> f64 {
        if self.total_cache_refs == 0 {
            0.0
        } else {
            (self.total_cache_misses as f64 / self.total_cache_refs as f64) * 100.0
        }
    }

    pub fn llc_miss_rate(&self) -> f64 {
        if self.total_llc_loads == 0 {
            0.0
        } else {
            (self.total_llc_load_misses as f64 / self.total_llc_loads as f64) * 100.0
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
    
    pub fn total_llc_loads(&self) -> u64 {
        self.get.total_llc_loads + 
        self.set.total_llc_loads + 
        self.del.total_llc_loads + 
        self.has.total_llc_loads
    }
    
    pub fn total_llc_load_misses(&self) -> u64 {
        self.get.total_llc_load_misses + 
        self.set.total_llc_load_misses + 
        self.del.total_llc_load_misses + 
        self.has.total_llc_load_misses
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
        writeln!(f, "  Total LLC Loads: {}", self.total_llc_loads())?;
        writeln!(f, "  Total LLC Load Misses: {} ({:.2}% LLC miss rate)", 
                 self.total_llc_load_misses(),
                 if self.total_llc_loads() > 0 {
                     100.0 * self.total_llc_load_misses() as f64 / self.total_llc_loads() as f64
                 } else { 0.0 })?;
        writeln!(f)?;
        
        if self.get.count > 0 {
            writeln!(f, "GET Operations ({} calls):", self.get.count)?;
            writeln!(f, "    Avg Cycles: {}", self.get.avg_cycles)?;
            writeln!(f, "    Avg Instructions: {} (IPC: {:.2})", self.get.avg_instructions, self.get.avg_ipc())?;
            writeln!(f, "    Avg Cache References: {}", self.get.avg_cache_refs)?;
            writeln!(f, "    Avg Cache Misses: {} ({:.2}% miss rate)", 
                     self.get.avg_cache_misses, self.get.cache_miss_rate())?;
            writeln!(f, "    Avg LLC Loads: {}", self.get.avg_llc_loads)?;
            writeln!(f, "    Avg LLC Load Misses: {} ({:.2}% LLC miss rate)", 
                     self.get.avg_llc_load_misses, self.get.llc_miss_rate())?;
            writeln!(f)?;
        }
        
        if self.set.count > 0 {
            writeln!(f, "SET Operations ({} calls):", self.set.count)?;
            writeln!(f, "    Avg Cycles: {}", self.set.avg_cycles)?;
            writeln!(f, "    Avg Instructions: {} (IPC: {:.2})", self.set.avg_instructions, self.set.avg_ipc())?;
            writeln!(f, "    Avg Cache References: {}", self.set.avg_cache_refs)?;
            writeln!(f, "    Avg Cache Misses: {} ({:.2}% miss rate)", 
                     self.set.avg_cache_misses, self.set.cache_miss_rate())?;
            writeln!(f, "    Avg LLC Loads: {}", self.set.avg_llc_loads)?;
            writeln!(f, "    Avg LLC Load Misses: {} ({:.2}% LLC miss rate)", 
                     self.set.avg_llc_load_misses, self.set.llc_miss_rate())?;
            writeln!(f)?;
        }
        
        if self.del.count > 0 {
            writeln!(f, "DEL Operations ({} calls):", self.del.count)?;
            writeln!(f, "    Avg Cycles: {}", self.del.avg_cycles)?;
            writeln!(f, "    Avg Instructions: {} (IPC: {:.2})", self.del.avg_instructions, self.del.avg_ipc())?;
            writeln!(f, "    Avg Cache References: {}", self.del.avg_cache_refs)?;
            writeln!(f, "    Avg Cache Misses: {} ({:.2}% miss rate)", 
                     self.del.avg_cache_misses, self.del.cache_miss_rate())?;
            writeln!(f, "    Avg LLC Loads: {}", self.del.avg_llc_loads)?;
            writeln!(f, "    Avg LLC Load Misses: {} ({:.2}% LLC miss rate)", 
                     self.del.avg_llc_load_misses, self.del.llc_miss_rate())?;
            writeln!(f)?;
        }
        
        if self.has.count > 0 {
            writeln!(f, "HAS Operations ({} calls):", self.has.count)?;
            writeln!(f, "    Avg Cycles: {}", self.has.avg_cycles)?;
            writeln!(f, "    Avg Instructions: {} (IPC: {:.2})", self.has.avg_instructions, self.has.avg_ipc())?;
            writeln!(f, "    Avg Cache References: {}", self.has.avg_cache_refs)?;
            writeln!(f, "    Avg Cache Misses: {} ({:.2}% miss rate)", 
                     self.has.avg_cache_misses, self.has.cache_miss_rate())?;
            writeln!(f, "    Avg LLC Loads: {}", self.has.avg_llc_loads)?;
            writeln!(f, "    Avg LLC Load Misses: {} ({:.2}% LLC miss rate)", 
                     self.has.avg_llc_load_misses, self.has.llc_miss_rate())?;
        }
        
        Ok(())
    }
}

/// Global hardware performance counters for the cache
#[derive(Debug)]
pub struct GlobalHwPerfCounters {
    pub global_hashbrown_dram: HwHashMapCounters,
}

impl GlobalHwPerfCounters {
    pub fn new() -> Self {
        GlobalHwPerfCounters {
            global_hashbrown_dram: HwHashMapCounters::new(),
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
/// 
/// Returns statistics even if hardware counters are unavailable (all values will be 0).
/// This allows code to uniformly handle the statistics without checking availability first.
pub fn get_hw_hashmap_stats() -> HwHashMapStats {
    let counters = get_hw_counters();
    counters.global_hashbrown_dram.get_stats()
}

/// Print hardware performance statistics
pub fn print_hw_perf_stats() {
    println!("\n=== Hardware Performance Counter Statistics ===\n");
    
    let stats = get_hw_hashmap_stats();
    println!("Global HashMap (hashbrown in DRAM):");
    print!("{}", stats);
    
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
            llc_loads: 100,
            llc_load_misses: 10,
            ..Default::default()
        };

        assert_eq!(measurement.cache_miss_rate(), 10.0);
        assert_eq!(measurement.ipc(), 2.0);
        assert_eq!(measurement.llc_miss_rate(), 10.0);
    }

    #[test]
    fn test_aggregated_measurement() {
        let agg = AggregatedMeasurement {
            count: 10,
            total_cycles: 10000,
            total_instructions: 20000,
            total_cache_refs: 1500,
            total_cache_misses: 150,
            total_llc_loads: 1000,
            total_llc_load_misses: 100,
            avg_cycles: 1000,
            avg_instructions: 2000,
            avg_cache_refs: 150,
            avg_cache_misses: 15,
            avg_llc_loads: 100,
            avg_llc_load_misses: 10,
        };

        assert_eq!(agg.cache_miss_rate(), 10.0);
        assert_eq!(agg.llc_miss_rate(), 10.0);
        assert_eq!(agg.avg_ipc(), 2.0);
    }
}
