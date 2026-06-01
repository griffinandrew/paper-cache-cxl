use core::sync::atomic::{AtomicU64, Ordering};

#[inline(always)]
pub fn rdtsc() -> u64 {
    #[cfg(target_arch = "x86_64")]
    unsafe {
        // _mm_lfence + rdtsc is the standard "serialize then read" pattern.
        // Without lfence the CPU can reorder rdtsc relative to surrounding work,
        // contaminating the measurement. lfence is cheap (~few cycles) and worth it.
        core::arch::x86_64::_mm_lfence();
        core::arch::x86_64::_rdtsc()
    }
    #[cfg(not(target_arch = "x86_64"))]
    { 0 }
}

// One bucket per phase of the SET path. Each accumulates total cycles and
// total samples; mean = total_cycles / samples. Atomic so the probes are
// safe even though your benchmark is single-threaded — costs ~nothing here.
pub struct PhaseStats {
    pub name: &'static str,
    pub total_cycles: AtomicU64,
    pub samples: AtomicU64,
}

impl PhaseStats {
    pub const fn new(name: &'static str) -> Self {
        Self {
            name,
            total_cycles: AtomicU64::new(0),
            samples: AtomicU64::new(0),
        }
    }

    #[inline(always)]
    pub fn record(&self, cycles: u64) {
        self.total_cycles.fetch_add(cycles, Ordering::Relaxed);
        self.samples.fetch_add(1, Ordering::Relaxed);
    }

    pub fn report(&self, tsc_hz: f64) {
        let s = self.samples.load(Ordering::Relaxed);
        let c = self.total_cycles.load(Ordering::Relaxed);
        if s == 0 {
            println!("{:<20} no samples", self.name);
            return;
        }
        let mean_cycles = c as f64 / s as f64;
        let mean_ns = mean_cycles * 1e9 / tsc_hz;
        println!("{:<20} samples={:>10}  mean={:>8.0} cycles  ({:>7.1} ns)",
                 self.name, s, mean_cycles, mean_ns);
    }
}

// One global instance per phase you want to time. Add or remove these to
// match the phases you instrument in the SET path.
pub static PHASE_PRE_ALLOC:  PhaseStats = PhaseStats::new("pre_alloc (index)");
pub static PHASE_ALLOC:      PhaseStats = PhaseStats::new("alloc (umf/bump)");
pub static PHASE_MEMCPY:     PhaseStats = PhaseStats::new("memcpy (write)");
pub static PHASE_POST:       PhaseStats = PhaseStats::new("post (bookkeeping)");
pub static PHASE_INSERT:     PhaseStats = PhaseStats::new("hashtable insert");


pub static PHASE_GET_HASH:      PhaseStats = PhaseStats::new("get: hash_key");
pub static PHASE_GET_LOCK:      PhaseStats = PhaseStats::new("get: rwlock read");
pub static PHASE_GET_LOOKUP:    PhaseStats = PhaseStats::new("get: map.get");
pub static PHASE_GET_VALIDATE:  PhaseStats = PhaseStats::new("get: key/expiry");
pub static PHASE_GET_COPY:      PhaseStats = PhaseStats::new("get: to_vec");
pub static PHASE_GET_BROADCAST: PhaseStats = PhaseStats::new("get: broadcast");
pub static PHASE_PROBE:         PhaseStats = PhaseStats::new("rdtsc probe pair");

pub fn report_set(tsc_hz: f64) {
    println!("\n=== SET path phase breakdown ===");
    PHASE_PRE_ALLOC.report(tsc_hz);
    PHASE_ALLOC.report(tsc_hz);
    PHASE_MEMCPY.report(tsc_hz);
    PHASE_POST.report(tsc_hz);
    PHASE_INSERT.report(tsc_hz);
    println!();
}

/// Calibrate TSC by sleeping a known wall-clock interval and measuring
/// elapsed cycles. Call this once at startup, before the benchmark loop.
pub fn calibrate_tsc_hz() -> f64 {
    let t0 = rdtsc();
    let wall0 = std::time::Instant::now();
    std::thread::sleep(std::time::Duration::from_millis(100));
    let elapsed = wall0.elapsed().as_secs_f64();
    let t1 = rdtsc();
    let hz = (t1 - t0) as f64 / elapsed;
    println!("TSC calibration: {:.3} GHz", hz / 1e9);
    hz
}


pub fn report_get(tsc_hz: f64) {
    println!("\n=== GET path phase breakdown ===");
    PHASE_GET_HASH.report(tsc_hz);
    PHASE_GET_LOCK.report(tsc_hz);
    PHASE_GET_LOOKUP.report(tsc_hz);
    PHASE_GET_VALIDATE.report(tsc_hz);
    PHASE_GET_COPY.report(tsc_hz);
    PHASE_GET_BROADCAST.report(tsc_hz);
    PHASE_PROBE.report(tsc_hz);  // subtract this floor from each phase above
    println!();
}

// Call once at startup, after calibrate_tsc_hz, before the bench loop.
pub fn calibrate_probe_overhead(iters: u64) {
    for _ in 0..iters {
        let a = rdtsc();
        let b = rdtsc();
        PHASE_PROBE.record(b - a);
    }
}


