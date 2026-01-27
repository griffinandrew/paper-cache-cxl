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
use perf_event::events::{Hardware, Software, Cache, CacheOp, CacheResult, WhichCache};
use std::sync::Mutex;
use std::io;
use std::time::{Instant, Duration};

/// Hardware performance measurement for a specific operation
/// 
/// Captures comprehensive microarchitectural and memory hierarchy statistics
/// including CPU execution metrics, cache behavior at all levels, TLB performance,
/// branch prediction, memory subsystem activity, and system-level events.
#[derive(Debug, Clone, Default)]
pub struct HwPerfMeasurement {
    // === Timing & Duration ===
    /// Wall-clock duration of the measurement in nanoseconds
    pub duration_ns: u64,
    /// Timestamp when measurement was taken (nanos since epoch)
    pub timestamp_ns: u64,

    // === CPU Execution Metrics ===
    /// CPU cycles consumed
    pub cycles: u64,
    /// Instructions retired (completed)
    pub instructions: u64,
    /// Reference CPU cycles (unaffected by frequency scaling)
    pub ref_cpu_cycles: u64,
    
    // === Branch Prediction ===
    /// Total branch instructions executed
    pub branch_instructions: u64,
    /// Branch mispredictions
    pub branch_misses: u64,
    
    // === Pipeline Stalls ===
    /// Cycles stalled on frontend (instruction fetch/decode)
    pub stalled_cycles_frontend: u64,
    /// Cycles stalled on backend (execution/memory)
    pub stalled_cycles_backend: u64,
    
    // === Generic Cache Metrics ===
    /// Total cache references (all levels)
    pub cache_references: u64,
    /// Total cache misses (all levels)
    pub cache_misses: u64,
    
    // === L1 Data Cache ===
    /// L1 D-cache load accesses
    pub l1_dcache_loads: u64,
    /// L1 D-cache load misses
    pub l1_dcache_load_misses: u64,
    /// L1 D-cache store accesses
    pub l1_dcache_stores: u64,
    /// L1 D-cache store misses
    pub l1_dcache_store_misses: u64,
    
    // === L1 Instruction Cache ===
    /// L1 I-cache load accesses
    pub l1_icache_loads: u64,
    /// L1 I-cache load misses
    pub l1_icache_load_misses: u64,
    
    // === Last-Level Cache (LLC) ===
    /// LLC load accesses
    pub llc_loads: u64,
    /// LLC load misses
    pub llc_load_misses: u64,
    /// LLC store accesses
    pub llc_stores: u64,
    /// LLC store misses
    pub llc_store_misses: u64,
    
    // === TLB (Translation Lookaside Buffer) ===
    /// Data TLB load accesses
    pub dtlb_loads: u64,
    /// Data TLB load misses
    pub dtlb_load_misses: u64,
    /// Data TLB store accesses
    pub dtlb_stores: u64,
    /// Data TLB store misses
    pub dtlb_store_misses: u64,
    /// Instruction TLB load accesses
    pub itlb_loads: u64,
    /// Instruction TLB load misses
    pub itlb_load_misses: u64,
    
    // === Software Events ===
    /// Page faults (minor + major)
    pub page_faults: u64,
    /// Minor page faults (no I/O required)
    pub page_faults_min: u64,
    /// Major page faults (I/O required)
    pub page_faults_maj: u64,
    /// Context switches
    pub context_switches: u64,
    /// CPU migrations (moved to different CPU core)
    pub cpu_migrations: u64,
}

impl HwPerfMeasurement {
    /// Instructions per cycle (IPC) - higher is better
    pub fn ipc(&self) -> f64 {
        if self.cycles == 0 {
            0.0
        } else {
            self.instructions as f64 / self.cycles as f64
        }
    }

    /// Overall cache miss rate (percentage)
    pub fn cache_miss_rate(&self) -> f64 {
        if self.cache_references == 0 {
            0.0
        } else {
            (self.cache_misses as f64 / self.cache_references as f64) * 100.0
        }
    }

    /// L1 D-cache load miss rate (percentage)
    pub fn l1_dcache_load_miss_rate(&self) -> f64 {
        if self.l1_dcache_loads == 0 {
            0.0
        } else {
            (self.l1_dcache_load_misses as f64 / self.l1_dcache_loads as f64) * 100.0
        }
    }

    /// L1 D-cache store miss rate (percentage)
    pub fn l1_dcache_store_miss_rate(&self) -> f64 {
        if self.l1_dcache_stores == 0 {
            0.0
        } else {
            (self.l1_dcache_store_misses as f64 / self.l1_dcache_stores as f64) * 100.0
        }
    }

    /// L1 I-cache miss rate (percentage)
    pub fn l1_icache_miss_rate(&self) -> f64 {
        if self.l1_icache_loads == 0 {
            0.0
        } else {
            (self.l1_icache_load_misses as f64 / self.l1_icache_loads as f64) * 100.0
        }
    }

    /// LLC load miss rate (percentage)
    pub fn llc_load_miss_rate(&self) -> f64 {
        if self.llc_loads == 0 {
            0.0
        } else {
            (self.llc_load_misses as f64 / self.llc_loads as f64) * 100.0
        }
    }

    /// LLC store miss rate (percentage)
    pub fn llc_store_miss_rate(&self) -> f64 {
        if self.llc_stores == 0 {
            0.0
        } else {
            (self.llc_store_misses as f64 / self.llc_stores as f64) * 100.0
        }
    }

    /// Overall LLC miss rate (percentage)
    pub fn llc_miss_rate(&self) -> f64 {
        let total_llc_accesses = self.llc_loads + self.llc_stores;
        let total_llc_misses = self.llc_load_misses + self.llc_store_misses;
        if total_llc_accesses == 0 {
            0.0
        } else {
            (total_llc_misses as f64 / total_llc_accesses as f64) * 100.0
        }
    }

    /// dTLB load miss rate (percentage)
    pub fn dtlb_load_miss_rate(&self) -> f64 {
        if self.dtlb_loads == 0 {
            0.0
        } else {
            (self.dtlb_load_misses as f64 / self.dtlb_loads as f64) * 100.0
        }
    }

    /// dTLB store miss rate (percentage)
    pub fn dtlb_store_miss_rate(&self) -> f64 {
        if self.dtlb_stores == 0 {
            0.0
        } else {
            (self.dtlb_store_misses as f64 / self.dtlb_stores as f64) * 100.0
        }
    }

    /// iTLB miss rate (percentage)
    pub fn itlb_miss_rate(&self) -> f64 {
        if self.itlb_loads == 0 {
            0.0
        } else {
            (self.itlb_load_misses as f64 / self.itlb_loads as f64) * 100.0
        }
    }

    /// Branch misprediction rate (percentage)
    pub fn branch_miss_rate(&self) -> f64 {
        if self.branch_instructions == 0 {
            0.0
        } else {
            (self.branch_misses as f64 / self.branch_instructions as f64) * 100.0
        }
    }

    /// Frontend stall percentage (% of total cycles)
    pub fn frontend_stall_percentage(&self) -> f64 {
        if self.cycles == 0 {
            0.0
        } else {
            (self.stalled_cycles_frontend as f64 / self.cycles as f64) * 100.0
        }
    }

    /// Backend stall percentage (% of total cycles)
    pub fn backend_stall_percentage(&self) -> f64 {
        if self.cycles == 0 {
            0.0
        } else {
            (self.stalled_cycles_backend as f64 / self.cycles as f64) * 100.0
        }
    }

    /// Total memory accesses (loads + stores across all levels)
    pub fn total_mem_accesses(&self) -> u64 {
        self.l1_dcache_loads + self.l1_dcache_stores
    }

    /// Duration in milliseconds
    pub fn duration_ms(&self) -> f64 {
        self.duration_ns as f64 / 1_000_000.0
    }

    /// Duration in microseconds
    pub fn duration_us(&self) -> f64 {
        self.duration_ns as f64 / 1_000.0
    }

    /// Cycles per instruction (CPI) - lower is better
    pub fn cpi(&self) -> f64 {
        if self.instructions == 0 {
            0.0
        } else {
            self.cycles as f64 / self.instructions as f64
        }
    }

    /// Total TLB misses
    pub fn total_tlb_misses(&self) -> u64 {
        self.dtlb_load_misses + self.dtlb_store_misses + self.itlb_load_misses
    }

    /// Total page faults
    pub fn total_page_faults(&self) -> u64 {
        self.page_faults
    }
}

/// Performance counter group for measuring operations
/// 
/// Attempts to create counters for all supported hardware events.
/// Some counters may not be available on all platforms or may require elevated permissions.
pub struct PerfCounterGroup {
    group: Option<Group>,
    // Core CPU metrics
    cycles_counter: Option<Counter>,
    instructions_counter: Option<Counter>,
    ref_cycles_counter: Option<Counter>,
    // Cache (generic)
    cache_refs_counter: Option<Counter>,
    cache_miss_counter: Option<Counter>,
    // Branch prediction
    branch_instructions_counter: Option<Counter>,
    branch_misses_counter: Option<Counter>,
    // Pipeline stalls
    stalled_frontend_counter: Option<Counter>,
    stalled_backend_counter: Option<Counter>,
    // L1 D-cache
    l1_dcache_loads_counter: Option<Counter>,
    l1_dcache_load_misses_counter: Option<Counter>,
    l1_dcache_stores_counter: Option<Counter>,
    l1_dcache_store_misses_counter: Option<Counter>,
    // L1 I-cache
    l1_icache_loads_counter: Option<Counter>,
    l1_icache_load_misses_counter: Option<Counter>,
    // LLC
    llc_loads_counter: Option<Counter>,
    llc_load_misses_counter: Option<Counter>,
    llc_stores_counter: Option<Counter>,
    llc_store_misses_counter: Option<Counter>,
    // TLB
    dtlb_loads_counter: Option<Counter>,
    dtlb_load_misses_counter: Option<Counter>,
    dtlb_stores_counter: Option<Counter>,
    dtlb_store_misses_counter: Option<Counter>,
    itlb_loads_counter: Option<Counter>,
    itlb_load_misses_counter: Option<Counter>,
    // Software events
    page_faults_counter: Option<Counter>,
    page_faults_min_counter: Option<Counter>,
    page_faults_maj_counter: Option<Counter>,
    context_switches_counter: Option<Counter>,
    cpu_migrations_counter: Option<Counter>,
}

impl PerfCounterGroup {
    /// Create a new performance counter group
    /// Returns a group with available counters (may be limited based on permissions/platform)
    /// 
    /// Note: Not all counters may be available on all systems. The implementation attempts
    /// to create all counters but will gracefully degrade if some are unavailable.
    pub fn new() -> Self {
        match Self::try_create_counters() {
            Ok(counters) => counters,
            Err(_) => {
                // Counters not available (insufficient permissions, virtualized environment, etc.)
                Self::empty()
            }
        }
    }

    /// Create an empty counter group (no counters available)
    fn empty() -> Self {
        PerfCounterGroup {
            group: None,
            cycles_counter: None,
            instructions_counter: None,
            ref_cycles_counter: None,
            cache_refs_counter: None,
            cache_miss_counter: None,
            branch_instructions_counter: None,
            branch_misses_counter: None,
            stalled_frontend_counter: None,
            stalled_backend_counter: None,
            l1_dcache_loads_counter: None,
            l1_dcache_load_misses_counter: None,
            l1_dcache_stores_counter: None,
            l1_dcache_store_misses_counter: None,
            l1_icache_loads_counter: None,
            l1_icache_load_misses_counter: None,
            llc_loads_counter: None,
            llc_load_misses_counter: None,
            llc_stores_counter: None,
            llc_store_misses_counter: None,
            dtlb_loads_counter: None,
            dtlb_load_misses_counter: None,
            dtlb_stores_counter: None,
            dtlb_store_misses_counter: None,
            itlb_loads_counter: None,
            itlb_load_misses_counter: None,
            page_faults_counter: None,
            page_faults_min_counter: None,
            page_faults_maj_counter: None,
            context_switches_counter: None,
            cpu_migrations_counter: None,
        }
    }

    fn try_create_counters() -> io::Result<PerfCounterGroup> {
        let mut group = Group::new()?;
        
        // Helper macro to create counter, returning None if it fails
        macro_rules! try_counter {
            ($group:expr, $kind:expr) => {
                Builder::new()
                    .group($group)
                    .kind($kind)
                    .build()
                    .ok()
            };
        }
        
        // Core CPU metrics (most likely to be available)
        let cycles = try_counter!(&mut group, Hardware::CPU_CYCLES);
        let instructions = try_counter!(&mut group, Hardware::INSTRUCTIONS);
        let ref_cycles = try_counter!(&mut group, Hardware::REF_CPU_CYCLES);
        
        // Generic cache metrics
        let cache_refs = try_counter!(&mut group, Hardware::CACHE_REFERENCES);
        let cache_miss = try_counter!(&mut group, Hardware::CACHE_MISSES);
        
        // Branch prediction
        let branch_instructions = try_counter!(&mut group, Hardware::BRANCH_INSTRUCTIONS);
        let branch_misses = try_counter!(&mut group, Hardware::BRANCH_MISSES);
        
        // Pipeline stalls
        let stalled_frontend = try_counter!(&mut group, Hardware::STALLED_CYCLES_FRONTEND);
        let stalled_backend = try_counter!(&mut group, Hardware::STALLED_CYCLES_BACKEND);
        
        // L1 D-cache
        let l1_dcache_loads = try_counter!(
            &mut group,
            Cache {
                which: WhichCache::L1D,
                operation: CacheOp::READ,
                result: CacheResult::ACCESS,
            }
        );
        let l1_dcache_load_misses = try_counter!(
            &mut group,
            Cache {
                which: WhichCache::L1D,
                operation: CacheOp::READ,
                result: CacheResult::MISS,
            }
        );
        let l1_dcache_stores = try_counter!(
            &mut group,
            Cache {
                which: WhichCache::L1D,
                operation: CacheOp::WRITE,
                result: CacheResult::ACCESS,
            }
        );
        let l1_dcache_store_misses = try_counter!(
            &mut group,
            Cache {
                which: WhichCache::L1D,
                operation: CacheOp::WRITE,
                result: CacheResult::MISS,
            }
        );
        
        // L1 I-cache
        let l1_icache_loads = try_counter!(
            &mut group,
            Cache {
                which: WhichCache::L1I,
                operation: CacheOp::READ,
                result: CacheResult::ACCESS,
            }
        );
        let l1_icache_load_misses = try_counter!(
            &mut group,
            Cache {
                which: WhichCache::L1I,
                operation: CacheOp::READ,
                result: CacheResult::MISS,
            }
        );
        
        // LLC (Last-Level Cache)
        let llc_loads = try_counter!(
            &mut group,
            Cache {
                which: WhichCache::LL,
                operation: CacheOp::READ,
                result: CacheResult::ACCESS,
            }
        );
        let llc_load_misses = try_counter!(
            &mut group,
            Cache {
                which: WhichCache::LL,
                operation: CacheOp::READ,
                result: CacheResult::MISS,
            }
        );
        let llc_stores = try_counter!(
            &mut group,
            Cache {
                which: WhichCache::LL,
                operation: CacheOp::WRITE,
                result: CacheResult::ACCESS,
            }
        );
        let llc_store_misses = try_counter!(
            &mut group,
            Cache {
                which: WhichCache::LL,
                operation: CacheOp::WRITE,
                result: CacheResult::MISS,
            }
        );
        
        // dTLB
        let dtlb_loads = try_counter!(
            &mut group,
            Cache {
                which: WhichCache::DTLB,
                operation: CacheOp::READ,
                result: CacheResult::ACCESS,
            }
        );
        let dtlb_load_misses = try_counter!(
            &mut group,
            Cache {
                which: WhichCache::DTLB,
                operation: CacheOp::READ,
                result: CacheResult::MISS,
            }
        );
        let dtlb_stores = try_counter!(
            &mut group,
            Cache {
                which: WhichCache::DTLB,
                operation: CacheOp::WRITE,
                result: CacheResult::ACCESS,
            }
        );
        let dtlb_store_misses = try_counter!(
            &mut group,
            Cache {
                which: WhichCache::DTLB,
                operation: CacheOp::WRITE,
                result: CacheResult::MISS,
            }
        );
        
        // iTLB
        let itlb_loads = try_counter!(
            &mut group,
            Cache {
                which: WhichCache::ITLB,
                operation: CacheOp::READ,
                result: CacheResult::ACCESS,
            }
        );
        let itlb_load_misses = try_counter!(
            &mut group,
            Cache {
                which: WhichCache::ITLB,
                operation: CacheOp::READ,
                result: CacheResult::MISS,
            }
        );
        
        // Software events
        let page_faults = try_counter!(&mut group, Software::PAGE_FAULTS);
        let page_faults_min = try_counter!(&mut group, Software::PAGE_FAULTS_MIN);
        let page_faults_maj = try_counter!(&mut group, Software::PAGE_FAULTS_MAJ);
        let context_switches = try_counter!(&mut group, Software::CONTEXT_SWITCHES);
        let cpu_migrations = try_counter!(&mut group, Software::CPU_MIGRATIONS);
        
        Ok(PerfCounterGroup {
            group: Some(group),
            cycles_counter: cycles,
            instructions_counter: instructions,
            ref_cycles_counter: ref_cycles,
            cache_refs_counter: cache_refs,
            cache_miss_counter: cache_miss,
            branch_instructions_counter: branch_instructions,
            branch_misses_counter: branch_misses,
            stalled_frontend_counter: stalled_frontend,
            stalled_backend_counter: stalled_backend,
            l1_dcache_loads_counter: l1_dcache_loads,
            l1_dcache_load_misses_counter: l1_dcache_load_misses,
            l1_dcache_stores_counter: l1_dcache_stores,
            l1_dcache_store_misses_counter: l1_dcache_store_misses,
            l1_icache_loads_counter: l1_icache_loads,
            l1_icache_load_misses_counter: l1_icache_load_misses,
            llc_loads_counter: llc_loads,
            llc_load_misses_counter: llc_load_misses,
            llc_stores_counter: llc_stores,
            llc_store_misses_counter: llc_store_misses,
            dtlb_loads_counter: dtlb_loads,
            dtlb_load_misses_counter: dtlb_load_misses,
            dtlb_stores_counter: dtlb_stores,
            dtlb_store_misses_counter: dtlb_store_misses,
            itlb_loads_counter: itlb_loads,
            itlb_load_misses_counter: itlb_load_misses,
            page_faults_counter: page_faults,
            page_faults_min_counter: page_faults_min,
            page_faults_maj_counter: page_faults_maj,
            context_switches_counter: context_switches,
            cpu_migrations_counter: cpu_migrations,
        })
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
    pub fn stop(&mut self, start_time: Instant) -> Result<HwPerfMeasurement, String> {
        if let Some(ref mut group) = self.group {
            group.disable().map_err(|e| format!("Failed to disable counters: {}", e))?;
            
            let duration = start_time.elapsed();
            let timestamp = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_nanos() as u64;
            
            // Helper macro to read counter value
            macro_rules! read_counter {
                ($counter:expr) => {
                    $counter.as_mut()
                        .and_then(|c| c.read().ok())
                        .unwrap_or(0)
                };
            }
            
            Ok(HwPerfMeasurement {
                // Timing
                duration_ns: duration.as_nanos() as u64,
                timestamp_ns: timestamp,
                
                // Core CPU metrics
                cycles: read_counter!(self.cycles_counter),
                instructions: read_counter!(self.instructions_counter),
                ref_cpu_cycles: read_counter!(self.ref_cycles_counter),
                
                // Branch prediction
                branch_instructions: read_counter!(self.branch_instructions_counter),
                branch_misses: read_counter!(self.branch_misses_counter),
                
                // Pipeline stalls
                stalled_cycles_frontend: read_counter!(self.stalled_frontend_counter),
                stalled_cycles_backend: read_counter!(self.stalled_backend_counter),
                
                // Generic cache
                cache_references: read_counter!(self.cache_refs_counter),
                cache_misses: read_counter!(self.cache_miss_counter),
                
                // L1 D-cache
                l1_dcache_loads: read_counter!(self.l1_dcache_loads_counter),
                l1_dcache_load_misses: read_counter!(self.l1_dcache_load_misses_counter),
                l1_dcache_stores: read_counter!(self.l1_dcache_stores_counter),
                l1_dcache_store_misses: read_counter!(self.l1_dcache_store_misses_counter),
                
                // L1 I-cache
                l1_icache_loads: read_counter!(self.l1_icache_loads_counter),
                l1_icache_load_misses: read_counter!(self.l1_icache_load_misses_counter),
                
                // LLC
                llc_loads: read_counter!(self.llc_loads_counter),
                llc_load_misses: read_counter!(self.llc_load_misses_counter),
                llc_stores: read_counter!(self.llc_stores_counter),
                llc_store_misses: read_counter!(self.llc_store_misses_counter),
                
                // TLB
                dtlb_loads: read_counter!(self.dtlb_loads_counter),
                dtlb_load_misses: read_counter!(self.dtlb_load_misses_counter),
                dtlb_stores: read_counter!(self.dtlb_stores_counter),
                dtlb_store_misses: read_counter!(self.dtlb_store_misses_counter),
                itlb_loads: read_counter!(self.itlb_loads_counter),
                itlb_load_misses: read_counter!(self.itlb_load_misses_counter),
                
                // Software events
                page_faults: read_counter!(self.page_faults_counter),
                page_faults_min: read_counter!(self.page_faults_min_counter),
                page_faults_maj: read_counter!(self.page_faults_maj_counter),
                context_switches: read_counter!(self.context_switches_counter),
                cpu_migrations: read_counter!(self.cpu_migrations_counter),
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
            
            // Helper macro to sum and average
            macro_rules! sum_and_avg {
                ($field:ident) => {
                    {
                        let total: u64 = measurements.iter().map(|m| m.$field).sum();
                        (total, total / count)
                    }
                };
            }
            
            let (total_duration_ns, avg_duration_ns) = sum_and_avg!(duration_ns);
            let (total_cycles, avg_cycles) = sum_and_avg!(cycles);
            let (total_instructions, avg_instructions) = sum_and_avg!(instructions);
            let (total_ref_cpu_cycles, avg_ref_cpu_cycles) = sum_and_avg!(ref_cpu_cycles);
            let (total_branch_instructions, avg_branch_instructions) = sum_and_avg!(branch_instructions);
            let (total_branch_misses, avg_branch_misses) = sum_and_avg!(branch_misses);
            let (total_stalled_cycles_frontend, avg_stalled_cycles_frontend) = sum_and_avg!(stalled_cycles_frontend);
            let (total_stalled_cycles_backend, avg_stalled_cycles_backend) = sum_and_avg!(stalled_cycles_backend);
            let (total_cache_refs, avg_cache_refs) = sum_and_avg!(cache_references);
            let (total_cache_misses, avg_cache_misses) = sum_and_avg!(cache_misses);
            let (total_l1_dcache_loads, avg_l1_dcache_loads) = sum_and_avg!(l1_dcache_loads);
            let (total_l1_dcache_load_misses, avg_l1_dcache_load_misses) = sum_and_avg!(l1_dcache_load_misses);
            let (total_l1_dcache_stores, avg_l1_dcache_stores) = sum_and_avg!(l1_dcache_stores);
            let (total_l1_dcache_store_misses, avg_l1_dcache_store_misses) = sum_and_avg!(l1_dcache_store_misses);
            let (total_l1_icache_loads, avg_l1_icache_loads) = sum_and_avg!(l1_icache_loads);
            let (total_l1_icache_load_misses, avg_l1_icache_load_misses) = sum_and_avg!(l1_icache_load_misses);
            let (total_llc_loads, avg_llc_loads) = sum_and_avg!(llc_loads);
            let (total_llc_load_misses, avg_llc_load_misses) = sum_and_avg!(llc_load_misses);
            let (total_llc_stores, avg_llc_stores) = sum_and_avg!(llc_stores);
            let (total_llc_store_misses, avg_llc_store_misses) = sum_and_avg!(llc_store_misses);
            let (total_dtlb_loads, avg_dtlb_loads) = sum_and_avg!(dtlb_loads);
            let (total_dtlb_load_misses, avg_dtlb_load_misses) = sum_and_avg!(dtlb_load_misses);
            let (total_dtlb_stores, avg_dtlb_stores) = sum_and_avg!(dtlb_stores);
            let (total_dtlb_store_misses, avg_dtlb_store_misses) = sum_and_avg!(dtlb_store_misses);
            let (total_itlb_loads, avg_itlb_loads) = sum_and_avg!(itlb_loads);
            let (total_itlb_load_misses, avg_itlb_load_misses) = sum_and_avg!(itlb_load_misses);
            let (total_page_faults, avg_page_faults) = sum_and_avg!(page_faults);
            let (total_page_faults_min, avg_page_faults_min) = sum_and_avg!(page_faults_min);
            let (total_page_faults_maj, avg_page_faults_maj) = sum_and_avg!(page_faults_maj);
            let (total_context_switches, avg_context_switches) = sum_and_avg!(context_switches);
            let (total_cpu_migrations, avg_cpu_migrations) = sum_and_avg!(cpu_migrations);

            AggregatedMeasurement {
                count,
                total_duration_ns,
                avg_duration_ns,
                total_cycles,
                total_instructions,
                total_ref_cpu_cycles,
                avg_cycles,
                avg_instructions,
                avg_ref_cpu_cycles,
                total_branch_instructions,
                total_branch_misses,
                avg_branch_instructions,
                avg_branch_misses,
                total_stalled_cycles_frontend,
                total_stalled_cycles_backend,
                avg_stalled_cycles_frontend,
                avg_stalled_cycles_backend,
                total_cache_refs,
                total_cache_misses,
                avg_cache_refs,
                avg_cache_misses,
                total_l1_dcache_loads,
                total_l1_dcache_load_misses,
                total_l1_dcache_stores,
                total_l1_dcache_store_misses,
                avg_l1_dcache_loads,
                avg_l1_dcache_load_misses,
                avg_l1_dcache_stores,
                avg_l1_dcache_store_misses,
                total_l1_icache_loads,
                total_l1_icache_load_misses,
                avg_l1_icache_loads,
                avg_l1_icache_load_misses,
                total_llc_loads,
                total_llc_load_misses,
                total_llc_stores,
                total_llc_store_misses,
                avg_llc_loads,
                avg_llc_load_misses,
                avg_llc_stores,
                avg_llc_store_misses,
                total_dtlb_loads,
                total_dtlb_load_misses,
                total_dtlb_stores,
                total_dtlb_store_misses,
                total_itlb_loads,
                total_itlb_load_misses,
                avg_dtlb_loads,
                avg_dtlb_load_misses,
                avg_dtlb_stores,
                avg_dtlb_store_misses,
                avg_itlb_loads,
                avg_itlb_load_misses,
                total_page_faults,
                total_page_faults_min,
                total_page_faults_maj,
                total_context_switches,
                total_cpu_migrations,
                avg_page_faults,
                avg_page_faults_min,
                avg_page_faults_maj,
                avg_context_switches,
                avg_cpu_migrations,
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
/// 
/// Provides totals and averages for all collected hardware performance metrics
/// across multiple operations of the same type (GET, SET, DEL, HAS).
#[derive(Debug, Clone, Default)]
pub struct AggregatedMeasurement {
    pub count: u64,
    
    // Timing
    pub total_duration_ns: u64,
    pub avg_duration_ns: u64,
    
    // Core CPU metrics
    pub total_cycles: u64,
    pub total_instructions: u64,
    pub total_ref_cpu_cycles: u64,
    pub avg_cycles: u64,
    pub avg_instructions: u64,
    pub avg_ref_cpu_cycles: u64,
    
    // Branch prediction
    pub total_branch_instructions: u64,
    pub total_branch_misses: u64,
    pub avg_branch_instructions: u64,
    pub avg_branch_misses: u64,
    
    // Pipeline stalls
    pub total_stalled_cycles_frontend: u64,
    pub total_stalled_cycles_backend: u64,
    pub avg_stalled_cycles_frontend: u64,
    pub avg_stalled_cycles_backend: u64,
    
    // Generic cache
    pub total_cache_refs: u64,
    pub total_cache_misses: u64,
    pub avg_cache_refs: u64,
    pub avg_cache_misses: u64,
    
    // L1 D-cache
    pub total_l1_dcache_loads: u64,
    pub total_l1_dcache_load_misses: u64,
    pub total_l1_dcache_stores: u64,
    pub total_l1_dcache_store_misses: u64,
    pub avg_l1_dcache_loads: u64,
    pub avg_l1_dcache_load_misses: u64,
    pub avg_l1_dcache_stores: u64,
    pub avg_l1_dcache_store_misses: u64,
    
    // L1 I-cache
    pub total_l1_icache_loads: u64,
    pub total_l1_icache_load_misses: u64,
    pub avg_l1_icache_loads: u64,
    pub avg_l1_icache_load_misses: u64,
    
    // LLC
    pub total_llc_loads: u64,
    pub total_llc_load_misses: u64,
    pub total_llc_stores: u64,
    pub total_llc_store_misses: u64,
    pub avg_llc_loads: u64,
    pub avg_llc_load_misses: u64,
    pub avg_llc_stores: u64,
    pub avg_llc_store_misses: u64,
    
    // TLB
    pub total_dtlb_loads: u64,
    pub total_dtlb_load_misses: u64,
    pub total_dtlb_stores: u64,
    pub total_dtlb_store_misses: u64,
    pub total_itlb_loads: u64,
    pub total_itlb_load_misses: u64,
    pub avg_dtlb_loads: u64,
    pub avg_dtlb_load_misses: u64,
    pub avg_dtlb_stores: u64,
    pub avg_dtlb_store_misses: u64,
    pub avg_itlb_loads: u64,
    pub avg_itlb_load_misses: u64,
    
    // Software events
    pub total_page_faults: u64,
    pub total_page_faults_min: u64,
    pub total_page_faults_maj: u64,
    pub total_context_switches: u64,
    pub total_cpu_migrations: u64,
    pub avg_page_faults: u64,
    pub avg_page_faults_min: u64,
    pub avg_page_faults_maj: u64,
    pub avg_context_switches: u64,
    pub avg_cpu_migrations: u64,
}

impl AggregatedMeasurement {
    pub fn total_mem_accesses(&self) -> u64 {
        self.total_l1_dcache_loads + self.total_l1_dcache_stores
    }

    pub fn avg_mem_accesses(&self) -> u64 {
        self.avg_l1_dcache_loads + self.avg_l1_dcache_stores
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

    pub fn l1_dcache_load_miss_rate(&self) -> f64 {
        if self.total_l1_dcache_loads == 0 {
            0.0
        } else {
            (self.total_l1_dcache_load_misses as f64 / self.total_l1_dcache_loads as f64) * 100.0
        }
    }

    pub fn l1_dcache_store_miss_rate(&self) -> f64 {
        if self.total_l1_dcache_stores == 0 {
            0.0
        } else {
            (self.total_l1_dcache_store_misses as f64 / self.total_l1_dcache_stores as f64) * 100.0
        }
    }

    pub fn l1_icache_miss_rate(&self) -> f64 {
        if self.total_l1_icache_loads == 0 {
            0.0
        } else {
            (self.total_l1_icache_load_misses as f64 / self.total_l1_icache_loads as f64) * 100.0
        }
    }

    pub fn llc_miss_rate(&self) -> f64 {
        let total_llc = self.total_llc_loads + self.total_llc_stores;
        let total_llc_misses = self.total_llc_load_misses + self.total_llc_store_misses;
        if total_llc == 0 {
            0.0
        } else {
            (total_llc_misses as f64 / total_llc as f64) * 100.0
        }
    }

    pub fn dtlb_miss_rate(&self) -> f64 {
        let total_dtlb = self.total_dtlb_loads + self.total_dtlb_stores;
        let total_dtlb_misses = self.total_dtlb_load_misses + self.total_dtlb_store_misses;
        if total_dtlb == 0 {
            0.0
        } else {
            (total_dtlb_misses as f64 / total_dtlb as f64) * 100.0
        }
    }

    pub fn itlb_miss_rate(&self) -> f64 {
        if self.total_itlb_loads == 0 {
            0.0
        } else {
            (self.total_itlb_load_misses as f64 / self.total_itlb_loads as f64) * 100.0
        }
    }

    pub fn branch_miss_rate(&self) -> f64 {
        if self.total_branch_instructions == 0 {
            0.0
        } else {
            (self.total_branch_misses as f64 / self.total_branch_instructions as f64) * 100.0
        }
    }

    pub fn frontend_stall_percentage(&self) -> f64 {
        if self.total_cycles == 0 {
            0.0
        } else {
            (self.total_stalled_cycles_frontend as f64 / self.total_cycles as f64) * 100.0
        }
    }

    pub fn backend_stall_percentage(&self) -> f64 {
        if self.total_cycles == 0 {
            0.0
        } else {
            (self.total_stalled_cycles_backend as f64 / self.total_cycles as f64) * 100.0
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
        
        // Helper closure to print operation details
        let print_operation = |f: &mut std::fmt::Formatter, name: &str, agg: &AggregatedMeasurement| -> std::fmt::Result {
            if agg.count == 0 {
                return Ok(());
            }
            
            writeln!(f, "{} Operations ({} calls):", name, agg.count)?;
            writeln!(f, "  ┌─ Execution Metrics:")?;
            writeln!(f, "  │  Duration: {:.2} µs avg", agg.avg_duration_ns as f64 / 1000.0)?;
            writeln!(f, "  │  Cycles: {} avg, {} total", agg.avg_cycles, agg.total_cycles)?;
            writeln!(f, "  │  Instructions: {} avg (IPC: {:.2})", agg.avg_instructions, agg.avg_ipc())?;
            
            if agg.total_branch_instructions > 0 {
                writeln!(f, "  │  Branches: {} avg, {} mispredictions ({:.2}% miss rate)", 
                         agg.avg_branch_instructions, 
                         agg.avg_branch_misses,
                         agg.branch_miss_rate())?;
            }
            
            if agg.total_stalled_cycles_frontend > 0 || agg.total_stalled_cycles_backend > 0 {
                writeln!(f, "  │  Stalls: Frontend {:.1}%, Backend {:.1}%", 
                         agg.frontend_stall_percentage(),
                         agg.backend_stall_percentage())?;
            }
            
            writeln!(f, "  ├─ Cache Hierarchy:")?;
            
            if agg.total_cache_refs > 0 {
                writeln!(f, "  │  Overall: {} refs, {} misses ({:.2}% miss rate)", 
                         agg.avg_cache_refs, agg.avg_cache_misses, agg.cache_miss_rate())?;
            }
            
            if agg.total_l1_dcache_loads > 0 || agg.total_l1_dcache_stores > 0 {
                writeln!(f, "  │  L1 D-cache:")?;
                if agg.total_l1_dcache_loads > 0 {
                    writeln!(f, "  │    Loads: {} avg, {} misses ({:.2}% miss rate)", 
                             agg.avg_l1_dcache_loads, agg.avg_l1_dcache_load_misses, 
                             agg.l1_dcache_load_miss_rate())?;
                }
                if agg.total_l1_dcache_stores > 0 {
                    writeln!(f, "  │    Stores: {} avg, {} misses ({:.2}% miss rate)", 
                             agg.avg_l1_dcache_stores, agg.avg_l1_dcache_store_misses,
                             agg.l1_dcache_store_miss_rate())?;
                }
            }
            
            if agg.total_l1_icache_loads > 0 {
                writeln!(f, "  │  L1 I-cache: {} loads, {} misses ({:.2}% miss rate)", 
                         agg.avg_l1_icache_loads, agg.avg_l1_icache_load_misses,
                         agg.l1_icache_miss_rate())?;
            }
            
            if agg.total_llc_loads > 0 || agg.total_llc_stores > 0 {
                writeln!(f, "  │  LLC:")?;
                if agg.total_llc_loads > 0 {
                    writeln!(f, "  │    Loads: {} avg, {} misses", 
                             agg.avg_llc_loads, agg.avg_llc_load_misses)?;
                }
                if agg.total_llc_stores > 0 {
                    writeln!(f, "  │    Stores: {} avg, {} misses", 
                             agg.avg_llc_stores, agg.avg_llc_store_misses)?;
                }
                if agg.total_llc_loads + agg.total_llc_stores > 0 {
                    writeln!(f, "  │    Overall: {:.2}% miss rate", agg.llc_miss_rate())?;
                }
            }
            
            writeln!(f, "  ├─ TLB Performance:")?;
            if agg.total_dtlb_loads > 0 || agg.total_dtlb_stores > 0 {
                let dtlb_accesses = agg.avg_dtlb_loads + agg.avg_dtlb_stores;
                let dtlb_misses = agg.avg_dtlb_load_misses + agg.avg_dtlb_store_misses;
                writeln!(f, "  │  dTLB: {} accesses, {} misses ({:.2}% miss rate)", 
                         dtlb_accesses, dtlb_misses, agg.dtlb_miss_rate())?;
            }
            if agg.total_itlb_loads > 0 {
                writeln!(f, "  │  iTLB: {} accesses, {} misses ({:.2}% miss rate)", 
                         agg.avg_itlb_loads, agg.avg_itlb_load_misses,
                         agg.itlb_miss_rate())?;
            }
            
            if agg.total_page_faults > 0 || agg.total_context_switches > 0 || agg.total_cpu_migrations > 0 {
                writeln!(f, "  └─ System Events:")?;
                if agg.total_page_faults > 0 {
                    writeln!(f, "     Page Faults: {} total ({} minor, {} major)", 
                             agg.avg_page_faults, agg.avg_page_faults_min, agg.avg_page_faults_maj)?;
                }
                if agg.total_context_switches > 0 {
                    writeln!(f, "     Context Switches: {} avg", agg.avg_context_switches)?;
                }
                if agg.total_cpu_migrations > 0 {
                    writeln!(f, "     CPU Migrations: {} avg", agg.avg_cpu_migrations)?;
                }
            }
            
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
        let start_time = Instant::now();
        if counter.start().is_err() {
            return (operation(), None);
        }
        
        // Run the operation
        let result = operation();
        
        // Stop counting and get measurements
        let measurement = counter.stop(start_time).ok();
        
        (result, measurement)
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_perf_measurement() {
        let measurement = HwPerfMeasurement {
            duration_ns: 1000,
            timestamp_ns: 0,
            cycles: 1000,
            instructions: 2000,
            ref_cpu_cycles: 1000,
            branch_instructions: 100,
            branch_misses: 5,
            stalled_cycles_frontend: 100,
            stalled_cycles_backend: 50,
            cache_references: 150,
            cache_misses: 15,
            l1_dcache_loads: 100,
            l1_dcache_load_misses: 10,
            l1_dcache_stores: 50,
            l1_dcache_store_misses: 5,
            l1_icache_loads: 200,
            l1_icache_load_misses: 10,
            llc_loads: 50,
            llc_load_misses: 5,
            llc_stores: 25,
            llc_store_misses: 2,
            dtlb_loads: 100,
            dtlb_load_misses: 1,
            dtlb_stores: 50,
            dtlb_store_misses: 0,
            itlb_loads: 200,
            itlb_load_misses: 2,
            page_faults: 0,
            page_faults_min: 0,
            page_faults_maj: 0,
            context_switches: 0,
            cpu_migrations: 0,
        };

        assert_eq!(measurement.total_mem_accesses(), 150);
        assert_eq!(measurement.cache_miss_rate(), 10.0);
        assert_eq!(measurement.ipc(), 2.0);
        assert_eq!(measurement.branch_miss_rate(), 5.0);
        assert_eq!(measurement.l1_dcache_load_miss_rate(), 10.0);
    }

    #[test]
    fn test_aggregated_measurement() {
        let agg = AggregatedMeasurement {
            count: 10,
            total_duration_ns: 10000,
            avg_duration_ns: 1000,
            total_cycles: 10000,
            total_instructions: 20000,
            total_ref_cpu_cycles: 10000,
            avg_cycles: 1000,
            avg_instructions: 2000,
            avg_ref_cpu_cycles: 1000,
            total_branch_instructions: 1000,
            total_branch_misses: 50,
            avg_branch_instructions: 100,
            avg_branch_misses: 5,
            total_stalled_cycles_frontend: 1000,
            total_stalled_cycles_backend: 500,
            avg_stalled_cycles_frontend: 100,
            avg_stalled_cycles_backend: 50,
            total_cache_refs: 1500,
            total_cache_misses: 150,
            avg_cache_refs: 150,
            avg_cache_misses: 15,
            total_l1_dcache_loads: 1000,
            total_l1_dcache_load_misses: 100,
            total_l1_dcache_stores: 500,
            total_l1_dcache_store_misses: 50,
            avg_l1_dcache_loads: 100,
            avg_l1_dcache_load_misses: 10,
            avg_l1_dcache_stores: 50,
            avg_l1_dcache_store_misses: 5,
            total_l1_icache_loads: 2000,
            total_l1_icache_load_misses: 100,
            avg_l1_icache_loads: 200,
            avg_l1_icache_load_misses: 10,
            total_llc_loads: 500,
            total_llc_load_misses: 50,
            total_llc_stores: 250,
            total_llc_store_misses: 25,
            avg_llc_loads: 50,
            avg_llc_load_misses: 5,
            avg_llc_stores: 25,
            avg_llc_store_misses: 2,
            total_dtlb_loads: 1000,
            total_dtlb_load_misses: 10,
            total_dtlb_stores: 500,
            total_dtlb_store_misses: 5,
            total_itlb_loads: 2000,
            total_itlb_load_misses: 20,
            avg_dtlb_loads: 100,
            avg_dtlb_load_misses: 1,
            avg_dtlb_stores: 50,
            avg_dtlb_store_misses: 0,
            avg_itlb_loads: 200,
            avg_itlb_load_misses: 2,
            total_page_faults: 0,
            total_page_faults_min: 0,
            total_page_faults_maj: 0,
            total_context_switches: 0,
            total_cpu_migrations: 0,
            avg_page_faults: 0,
            avg_page_faults_min: 0,
            avg_page_faults_maj: 0,
            avg_context_switches: 0,
            avg_cpu_migrations: 0,
        };

        assert_eq!(agg.total_mem_accesses(), 1500);
        assert_eq!(agg.avg_mem_accesses(), 150);
        assert_eq!(agg.cache_miss_rate(), 10.0);
        assert_eq!(agg.avg_ipc(), 2.0);
        assert_eq!(agg.l1_dcache_load_miss_rate(), 10.0);
        assert_eq!(agg.branch_miss_rate(), 5.0);
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
