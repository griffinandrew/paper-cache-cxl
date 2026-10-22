//! Can a sampled mini stack drive tiering?
//!
//! One read-through LRU cache of `C` bytes with a fast tier of `F` bytes, run over a trace, with
//! two tier placements side by side over the SAME residency (eviction is exact LRU for both, so
//! only placement differs):
//!
//! * EXACT -- the `lru-compact-hybrid` rule: the fast tier is the top `F` bytes of the recency
//!   order, kept by a boundary cursor. Every slow hit promotes, the cursor's key demotes.
//! * APPROX -- no ordering for placement. Each object keeps only its last-access stamp. A mini LRU
//!   over the keys whose hash falls in the sample (rate `R`, PaperCache's rule), sized `F * R`,
//!   estimates T*: the access age at the fast tier's edge (now - stamp of the mini's tail). A fast
//!   object older than T* is demoted by a CLOCK hand; a slow hit promotes (`promote=always`, which
//!   approximates EXACT) or promotes only if its previous inter-access gap was under T*
//!   (`promote=gap`, a reuse filter). If fast bytes exceed `F` anyway, the hand demotes the oldest
//!   of a small sample (the byte gate's job) and counts it as FORCED.
//!
//! Reported: fast-hit ratio of each, migrations of each, forced demotions, the byte overlap of the
//! two fast sets at checkpoints, and T*'s error against the true edge age (EXACT's cursor key).
//!
//! Usage: `approx_tier [key=value ...]`, keys listed in `Config::parse`.

use std::time::Instant;

const NIL: u32 = u32::MAX;

// ---- config -------------------------------------------------------------------------------

#[derive(Clone, Debug)]
struct Config {
	keys:        usize,
	requests:    u64,
	alpha:       f64,
	min_size:    u32,
	max_size:    u32,
	cache_frac:  f64, // C as a fraction of the total bytes of all keys
	fast_frac:   f64, // F as a fraction of C
	rate:        f64, // sampling rate R
	promote_gap: bool,
	gap_mult:    f64, // promote=gap promotes when the previous gap < gap_mult * T*
	sweep:       u32, // hand steps per request (0 = no threshold demotions: the backstop alone, a control)
	ewma:        f64, // smoothing of T* per sampled access (1.0 = raw)
	shift_every: u64, // 0 = stationary popularity
	shift_frac:  f64, // fraction of ranks reshuffled at each shift
	checkpoints: u64,
	seed:        u64,
	evict:       Evict,
	trace:       Option<String>, // a paper-benchmark trace (25-byte records); replaces the Zipf generator
	limit:       u64,  // trace: stop after this many records (0 = all)
	meta:        u32,  // bytes of per-object metadata charged with each value (cache and fast tier)
	cache_bytes: u64,  // absolute C (0 = cache_frac of all bytes)
	fast_bytes:  u64,  // absolute F (0 = fast_frac of C)
	evict_samples: usize, // keys sampled per eviction (evict=sampled|pool)
}

/// How the victim is chosen when the cache is full.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Evict {
	/// The LRU list's tail: exact LRU (needs the per-object list).
	Exact,
	/// The oldest stamp among `evict_samples` random residents (no list: Redis's original
	/// approximate LRU).
	Sampled,
	/// As `Sampled`, plus a 16-entry pool of the oldest candidates seen so far, kept across
	/// evictions (Redis 3.0's eviction pool).
	Pool,
}

impl Config {
	fn parse() -> Self {
		let mut c = Config {
			keys:        1_000_000,
			requests:    20_000_000,
			alpha:       0.9,
			min_size:    64,
			max_size:    16_384,
			cache_frac:  0.3,
			fast_frac:   0.25,
			rate:        0.01,
			promote_gap: false,
			gap_mult:    1.0,
			sweep:       4,
			ewma:        0.05,
			shift_every: 0,
			shift_frac:  0.2,
			checkpoints: 40,
			seed:        1,
			evict:       Evict::Exact,
			trace:       None,
			limit:       0,
			meta:        0,
			cache_bytes: 0,
			fast_bytes:  0,
			evict_samples: 8,
		};

		for arg in std::env::args().skip(1) {
			let (k, v) = arg.split_once('=').unwrap_or_else(|| panic!("expected key=value, got {arg}"));
			match k {
				"keys" => c.keys = v.parse().unwrap(),
				"requests" => c.requests = v.parse().unwrap(),
				"alpha" => c.alpha = v.parse().unwrap(),
				"min_size" => c.min_size = v.parse().unwrap(),
				"max_size" => c.max_size = v.parse().unwrap(),
				"cache_frac" => c.cache_frac = v.parse().unwrap(),
				"fast_frac" => c.fast_frac = v.parse().unwrap(),
				"rate" => c.rate = v.parse().unwrap(),
				"promote" => c.promote_gap = match v {
					"always" => false,
					"gap" => true,
					_ => panic!("promote=always|gap"),
				},
				"gap_mult" => c.gap_mult = v.parse().unwrap(),
				"sweep" => c.sweep = v.parse().unwrap(),
				"ewma" => c.ewma = v.parse().unwrap(),
				"shift_every" => c.shift_every = v.parse().unwrap(),
				"shift_frac" => c.shift_frac = v.parse().unwrap(),
				"checkpoints" => c.checkpoints = v.parse().unwrap(),
				"seed" => c.seed = v.parse().unwrap(),
				"evict" => c.evict = match v {
					"exact" => Evict::Exact,
					"sampled" => Evict::Sampled,
					"pool" => Evict::Pool,
					_ => panic!("evict=exact|sampled|pool"),
				},
				"trace" => c.trace = Some(v.to_string()),
				"limit" => c.limit = v.parse().unwrap(),
				"meta" => c.meta = v.parse().unwrap(),
				"cache_bytes" => c.cache_bytes = v.parse().unwrap(),
				"fast_bytes" => c.fast_bytes = v.parse().unwrap(),
				"evict_samples" => c.evict_samples = v.parse().unwrap(),
				_ => panic!("unknown key {k}"),
			}
		}

		c
	}
}

// ---- rng, hash ------------------------------------------------------------------------------

struct Rng(u64);

impl Rng {
	fn next(&mut self) -> u64 {
		// xorshift64*
		self.0 ^= self.0 >> 12;
		self.0 ^= self.0 << 25;
		self.0 ^= self.0 >> 27;
		self.0.wrapping_mul(0x2545_f491_4f6c_dd1d)
	}

	fn unit(&mut self) -> f64 {
		(self.next() >> 11) as f64 / (1u64 << 53) as f64
	}
}

fn splitmix(mut x: u64) -> u64 {
	x = x.wrapping_add(0x9e37_79b9_7f4a_7c15);
	x = (x ^ (x >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
	x = (x ^ (x >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
	x ^ (x >> 31)
}

// ---- intrusive LRU over key ids -----------------------------------------------------------

struct List {
	prev: Vec<u32>,
	next: Vec<u32>,
	head: u32, // most recent
	tail: u32, // least recent
}

impl List {
	fn new(n: usize) -> Self {
		List {
			prev: vec![NIL; n],
			next: vec![NIL; n],
			head: NIL,
			tail: NIL,
		}
	}

	fn unlink(&mut self, k: u32) {
		let (p, n) = (self.prev[k as usize], self.next[k as usize]);
		if p == NIL { self.head = n } else { self.next[p as usize] = n }
		if n == NIL { self.tail = p } else { self.prev[n as usize] = p }
		self.prev[k as usize] = NIL;
		self.next[k as usize] = NIL;
	}

	fn push_front(&mut self, k: u32) {
		self.prev[k as usize] = NIL;
		self.next[k as usize] = self.head;
		if self.head != NIL { self.prev[self.head as usize] = k } else { self.tail = k }
		self.head = k;
	}
}

// ---- the simulation -----------------------------------------------------------------------

#[derive(Default, Debug)]
struct Counts {
	reqs:            u64,
	sets:            u64,
	gets:            u64,
	hits:            u64,
	exact_fast_hits: u64,
	approx_fast_hits: u64,
	exact_promotions: u64,
	exact_demotions: u64,
	approx_promotions: u64,
	approx_demotions: u64,
	approx_forced:   u64,
	// policy-worker work, in key-addressed touches (each a likely cache miss)
	ops_list:        u64, // stack: list relinks on every access (exact)
	ops_cursor:      u64, // stack: cursor steps to demote (exact)
	ops_mini:        u64, // stackless: mini-stack relinks/evictions (sampled keys only)
	ops_sweep:       u64, // stackless: objects the clock hand reads
	ops_backstop:    u64, // stackless: objects the backstop samples
	ops_evict:       u64, // stackless: objects eviction samples (sampled/pool)
	// checkpoint sums
	cps:             u64,
	overlap:         f64, // bytes(approx & exact) / bytes(exact)
	extra:           f64, // bytes(approx & !exact) / F
	approx_fill:     f64, // approx fast bytes / F
	tstar_rel_err:   f64,
	tstar_cps:       u64,
}

struct Sim {
	cfg:  Config,
	size: Vec<u32>,
	cap:  u64,
	fast: u64,

	// residency (shared)
	lru:      List,
	resident: Vec<bool>,
	used:     u64,
	stamp:    Vec<u64>,

	// EXACT placement
	efast:       Vec<bool>,
	efast_bytes: u64,
	boundary:    u32, // oldest fast key; fast keys are the prefix head..=boundary

	// APPROX placement
	afast:       Vec<bool>,
	afast_bytes: u64,
	aset:        Vec<u32>, // the fast keys, for the hand
	apos:        Vec<u32>, // index of a fast key in `aset`
	hand:        usize,

	// mini stack over the sampled keys
	sampled:    Vec<bool>,
	mini:       List,
	in_mini:    Vec<bool>,
	mini_used:  u64,
	mini_cap:   u64,
	tstar:      f64,
	tstar_seen: bool,

	// residents as a vector, for random sampling (stands in for sampling the object map)
	rset:  Vec<u32>,
	rpos:  Vec<u32>,
	pool:  Vec<u32>,
	rng:   Rng,

	now: u64,
	c:   Counts,
}

impl Sim {
	fn new(cfg: Config, size: Vec<u32>) -> Self {
		let n = size.len();
		let total: u64 = size.iter().map(|&s| s as u64).sum();
		let cap = if cfg.cache_bytes > 0 { cfg.cache_bytes } else { (total as f64 * cfg.cache_frac) as u64 };
		let fast = if cfg.fast_bytes > 0 { cfg.fast_bytes } else { (cap as f64 * cfg.fast_frac) as u64 };
		let threshold = (cfg.rate * (1u64 << 24) as f64) as u64;
		let sampled = (0..n as u64)
			.map(|k| splitmix(k ^ cfg.seed.rotate_left(17)) & ((1 << 24) - 1) < threshold)
			.collect();
		let mini_cap = (fast as f64 * cfg.rate) as u64;
		let cfg_seed = cfg.seed.wrapping_mul(0xd6e8_feb8_6659_fd93) | 1;

		Sim {
			cfg,
			size,
			cap,
			fast,
			lru: List::new(n),
			resident: vec![false; n],
			used: 0,
			stamp: vec![0; n],
			efast: vec![false; n],
			efast_bytes: 0,
			boundary: NIL,
			afast: vec![false; n],
			afast_bytes: 0,
			aset: Vec::new(),
			apos: vec![NIL; n],
			hand: 0,
			sampled,
			mini: List::new(n),
			in_mini: vec![false; n],
			mini_used: 0,
			mini_cap,
			tstar: 0.0,
			tstar_seen: false,
			rset: Vec::new(),
			rpos: vec![NIL; n],
			pool: Vec::new(),
			rng: Rng(0x5eed_0f_e71c7 ^ cfg_seed),
			now: 0,
			c: Counts::default(),
		}
	}

	/// A request for key `k` whose value (with `meta`) is `size` bytes. A GET that misses is
	/// admitted (read-through); a SET is admitted or overwrites. Only GETs count toward the hit
	/// ratios.
	fn access(&mut self, k: u32, size: u32, is_get: bool) {
		self.now += 1;
		self.c.reqs += 1;
		if is_get { self.c.gets += 1 } else { self.c.sets += 1 }
		let ku = k as usize;
		if self.size[ku] != size { self.set_size(k, size) }
		let prev_stamp = self.stamp[ku];

		self.c.ops_list += 1;
		if self.sampled[ku] {
			self.c.ops_mini += 1;
			self.mini_access(k);
		}

		if self.resident[ku] {
			if is_get {
				self.c.hits += 1;
				if self.efast[ku] { self.c.exact_fast_hits += 1 }
				if self.afast[ku] { self.c.approx_fast_hits += 1 }
			}

			self.exact_hit(k);
			self.lru.unlink(k);
			self.lru.push_front(k);
			self.stamp[ku] = self.now;
			self.approx_hit(k, self.now - prev_stamp);
		} else {
			self.lru.push_front(k);
			self.resident[ku] = true;
			self.used += self.size[ku] as u64;
			self.rpos[ku] = self.rset.len() as u32;
			self.rset.push(k);
			self.stamp[ku] = self.now;

			// both placements admit to fast (the LRU hybrid's rule)
			self.efast[ku] = true;
			self.efast_bytes += self.size[ku] as u64;
			if self.boundary == NIL { self.boundary = k }
			self.approx_set_fast(k);

			self.evict();
		}

		// an overwrite that grew can put the cache over C
		if self.used > self.cap { self.evict() }
		self.exact_settle();
		self.approx_sweep();
	}

	fn set_size(&mut self, k: u32, new: u32) {
		let ku = k as usize;
		let old = self.size[ku] as u64;
		let new64 = new as u64;
		if self.resident[ku] { self.used = self.used - old + new64 }
		if self.efast[ku] { self.efast_bytes = self.efast_bytes - old + new64 }
		if self.afast[ku] { self.afast_bytes = self.afast_bytes - old + new64 }
		if self.in_mini[ku] { self.mini_used = self.mini_used - old + new64 }
		self.size[ku] = new;
	}

	// -- EXACT --

	fn exact_hit(&mut self, k: u32) {
		let ku = k as usize;
		if self.efast[ku] {
			// moving the cursor's own key to the front: the cursor steps to its predecessor
			if self.boundary == k && self.lru.prev[ku] != NIL {
				self.boundary = self.lru.prev[ku];
			}
		} else {
			self.efast[ku] = true;
			self.efast_bytes += self.size[ku] as u64;
			self.c.exact_promotions += 1;
			if self.boundary == NIL { self.boundary = k }
		}
	}

	fn exact_settle(&mut self) {
		while self.efast_bytes > self.fast && self.boundary != NIL {
			let b = self.boundary;
			self.efast[b as usize] = false;
			self.efast_bytes -= self.size[b as usize] as u64;
			self.c.exact_demotions += 1;
			self.c.ops_cursor += 1;
			self.boundary = self.lru.prev[b as usize];
		}
	}

	fn evict(&mut self) {
		while self.used > self.cap {
			let v = self.victim();
			let vu = v as usize;
			if self.boundary == v { self.boundary = self.lru.prev[vu] }
			if self.efast[vu] {
				self.efast[vu] = false;
				self.efast_bytes -= self.size[vu] as u64;
			}
			if self.afast[vu] { self.approx_unset_fast(v) }
			self.lru.unlink(v);
			self.resident[vu] = false;
			self.used -= self.size[vu] as u64;
			let i = self.rpos[vu] as usize;
			let last = self.rset.pop().unwrap();
			if last != v {
				self.rset[i] = last;
				self.rpos[last as usize] = i as u32;
			}
			self.rpos[vu] = NIL;
		}
	}

	fn victim(&mut self) -> u32 {
		match self.cfg.evict {
			Evict::Exact => self.lru.tail,
			Evict::Sampled => {
				let mut best = NIL;
				for _ in 0..self.cfg.evict_samples {
					let k = self.rset[(self.rng.next() % self.rset.len() as u64) as usize];
					self.c.ops_evict += 1;
					if best == NIL || self.stamp[k as usize] < self.stamp[best as usize] { best = k }
				}
				best
			},
			Evict::Pool => {
				// drop entries that left the cache; add fresh samples; keep the 16 oldest
				let resident = &self.resident;
				self.pool.retain(|&k| resident[k as usize]);
				for _ in 0..self.cfg.evict_samples {
					let k = self.rset[(self.rng.next() % self.rset.len() as u64) as usize];
					self.c.ops_evict += 1;
					if !self.pool.contains(&k) { self.pool.push(k) }
				}
				let stamp = &self.stamp;
				self.pool.sort_unstable_by_key(|&k| std::cmp::Reverse(stamp[k as usize]));
				let excess = self.pool.len().saturating_sub(16);
				self.pool.drain(..excess);
				self.pool.pop().unwrap() // the oldest stamp
			},
		}
	}

	// -- APPROX --

	fn approx_set_fast(&mut self, k: u32) {
		let ku = k as usize;
		self.afast[ku] = true;
		self.afast_bytes += self.size[ku] as u64;
		self.apos[ku] = self.aset.len() as u32;
		self.aset.push(k);
	}

	fn approx_unset_fast(&mut self, k: u32) {
		let ku = k as usize;
		self.afast[ku] = false;
		self.afast_bytes -= self.size[ku] as u64;
		let i = self.apos[ku] as usize;
		let last = self.aset.pop().unwrap();
		if last != k {
			self.aset[i] = last;
			self.apos[last as usize] = i as u32;
		}
		self.apos[ku] = NIL;
	}

	fn approx_hit(&mut self, k: u32, gap: u64) {
		if self.afast[k as usize] { return }
		let promote = !self.cfg.promote_gap || !self.tstar_seen || (gap as f64) < self.cfg.gap_mult * self.tstar;
		if promote {
			self.approx_set_fast(k);
			self.c.approx_promotions += 1;
		}
	}

	fn age(&self, k: u32) -> f64 {
		(self.now - self.stamp[k as usize]) as f64
	}

	fn approx_sweep(&mut self) {
		// threshold demotions: the hand ages fast keys out at T*
		if self.tstar_seen {
			for _ in 0..self.cfg.sweep {
				if self.aset.is_empty() { break }
				if self.hand >= self.aset.len() { self.hand = 0 }
				let k = self.aset[self.hand];
				self.c.ops_sweep += 1;
				if self.age(k) > self.tstar {
					self.approx_unset_fast(k); // the last key lands at `hand`: look at it next
					self.c.approx_demotions += 1;
				} else {
					self.hand += 1;
				}
			}
		}

		// the backstop: over budget anyway, demote the oldest of 8 the hand looks at
		while self.afast_bytes > self.fast && !self.aset.is_empty() {
			let mut best = NIL;
			let mut best_age = -1.0;
			for _ in 0..8 {
				if self.hand >= self.aset.len() { self.hand = 0 }
				let k = self.aset[self.hand];
				self.c.ops_backstop += 1;
				if self.age(k) > best_age { best = k; best_age = self.age(k) }
				self.hand += 1;
			}
			self.approx_unset_fast(best);
			self.c.approx_demotions += 1;
			self.c.approx_forced += 1;
		}
	}

	// -- the mini stack --

	fn mini_access(&mut self, k: u32) {
		let ku = k as usize;
		if self.in_mini[ku] {
			self.mini.unlink(k);
		} else {
			self.in_mini[ku] = true;
			self.mini_used += self.size[ku] as u64;
		}
		self.mini.push_front(k);

		let mut evicted = false;
		while self.mini_used > self.mini_cap && self.mini.tail != NIL {
			let t = self.mini.tail;
			self.mini.unlink(t);
			self.in_mini[t as usize] = false;
			self.mini_used -= self.size[t as usize] as u64;
			self.c.ops_mini += 1;
			evicted = true;
		}

		// T* only means something once the mini is full: the age at its tail, i.e. at byte F*R
		if evicted || self.tstar_seen {
			let t = self.mini.tail;
			if t != NIL {
				// stamps are the real clock (every request), so the age is in real requests: no
				// rescaling by R. The current key's stamp is written after this call, but it is
				// the head, not the tail, unless the mini holds one key.
				let raw = (self.now - self.stamp[t as usize]) as f64;
				self.tstar = if self.tstar_seen {
					self.tstar + self.cfg.ewma * (raw - self.tstar)
				} else {
					raw
				};
				self.tstar_seen = true;
			}
		}
	}


	// -- checkpoints --

	fn checkpoint(&mut self) {
		let (mut both, mut exact, mut extra) = (0u64, 0u64, 0u64);
		let mut k = self.lru.head;
		while k != NIL {
			let ku = k as usize;
			let s = self.size[ku] as u64;
			match (self.efast[ku], self.afast[ku]) {
				(true, true) => { both += s; exact += s }
				(true, false) => exact += s,
				(false, true) => extra += s,
				_ => {}
			}
			k = self.lru.next[ku];
		}
		if exact == 0 { return }
		self.c.cps += 1;
		self.c.overlap += both as f64 / exact as f64;
		self.c.extra += extra as f64 / self.fast as f64;
		self.c.approx_fill += self.afast_bytes as f64 / self.fast as f64;

		if self.tstar_seen && self.boundary != NIL {
			let truth = self.age(self.boundary);
			if truth > 0.0 {
				self.c.tstar_rel_err += ((self.tstar - truth) / truth).abs();
				self.c.tstar_cps += 1;
			}
		}
	}
}

// ---- trace ---------------------------------------------------------------------------------

struct Zipf {
	cdf: Vec<f64>,
}

impl Zipf {
	fn new(n: usize, alpha: f64) -> Self {
		let mut cdf = Vec::with_capacity(n);
		let mut sum = 0.0;
		for r in 1..=n {
			sum += 1.0 / (r as f64).powf(alpha);
			cdf.push(sum);
		}
		for x in &mut cdf { *x /= sum }
		Zipf { cdf }
	}

	fn rank(&self, u: f64) -> usize {
		self.cdf.partition_point(|&x| x < u).min(self.cdf.len() - 1)
	}
}

fn main() {
	let cfg = Config::parse();
	let mut rng = Rng(cfg.seed.wrapping_mul(0x9e37_79b9_7f4a_7c15) | 1);

	// per-key size, log-uniform in [min, max]
	let (lo, hi) = ((cfg.min_size as f64).ln(), (cfg.max_size as f64).ln());
	let size: Vec<u32> = (0..cfg.keys).map(|_| (lo + rng.unit() * (hi - lo)).exp() as u32).collect();

	// rank -> key, shuffled so popularity is independent of size and of the sampling hash
	let mut key_of: Vec<u32> = (0..cfg.keys as u32).collect();
	for i in (1..key_of.len()).rev() {
		let j = (rng.next() % (i as u64 + 1)) as usize;
		key_of.swap(i, j);
	}

	let start = Instant::now();

	if let Some(path) = cfg.trace.clone() {
		let (ids, sizes, gets, size0) = read_trace(&path, cfg.limit, cfg.meta);
		let mut cfg = cfg.clone();
		cfg.keys = size0.len();
		cfg.requests = ids.len() as u64;
		eprintln!("loaded {} records, {} keys in {:.1}s", ids.len(), size0.len(), start.elapsed().as_secs_f64());
		let mut sim = Sim::new(cfg.clone(), size0);
		let every = (cfg.requests / cfg.checkpoints.max(1)).max(1);
		let warm = cfg.requests / 10;
		for i in 0..ids.len() {
			sim.access(ids[i], sizes[i], gets[i]);
			if i as u64 >= warm && i as u64 % every == 0 {
				sim.checkpoint();
			}
		}
		report(&cfg, &sim, start);
		return;
	}

	let zipf = Zipf::new(cfg.keys, cfg.alpha);
	let size: Vec<u32> = size.into_iter().map(|s| s + cfg.meta).collect();
	let mut sim = Sim::new(cfg.clone(), size);
	let every = (cfg.requests / cfg.checkpoints.max(1)).max(1);
	let warm = cfg.requests / 10; // checkpoints start after a 10% warm-up

	for i in 0..cfg.requests {
		if cfg.shift_every > 0 && i > 0 && i % cfg.shift_every == 0 {
			let swaps = (cfg.keys as f64 * cfg.shift_frac) as usize;
			for _ in 0..swaps {
				let a = (rng.next() % cfg.keys as u64) as usize;
				let b = (rng.next() % cfg.keys as u64) as usize;
				key_of.swap(a, b);
			}
		}

		let k = key_of[zipf.rank(rng.unit())];
		let size = sim.size[k as usize];
		sim.access(k, size, true);

		if i >= warm && i % every == 0 {
			sim.checkpoint();
		}
	}

	report(&cfg, &sim, start);
}

/// Reads a paper-benchmark trace: 25-byte little-endian records of
/// `timestamp u64 | command u8 (0 GET, 1 SET) | key u64 | value size u32 | ttl u32`.
/// Keys are renumbered densely; each key's starting size is its first record's.
fn read_trace(path: &str, limit: u64, meta: u32) -> (Vec<u32>, Vec<u32>, Vec<bool>, Vec<u32>) {
	use std::collections::HashMap;
	use std::io::Read;

	let file = std::fs::File::open(path).unwrap_or_else(|e| panic!("open {path}: {e}"));
	let len = file.metadata().map(|m| m.len()).unwrap_or(0);
	let mut records = len / 25;
	if limit > 0 { records = records.min(limit) }
	let mut rdr = std::io::BufReader::with_capacity(1 << 22, file);

	// PBTRACE2 (the Meta traces): a 64-byte header, then variable-length records
	// `ts u64 | cmd u8 | key_len u8 | value_size u32 | ttl u32 | key bytes`. Otherwise the
	// benchmark's fixed 25-byte records with a u64 key.
	let mut magic = [0u8; 8];
	let pb = rdr.read_exact(&mut magic).is_ok() && &magic == b"PBTRACE2";
	if pb {
		let mut hdr = [0u8; 56];
		rdr.read_exact(&mut hdr).expect("PBTRACE2 header");
		records = if limit > 0 { limit } else { u64::MAX };
	} else {
		drop(rdr);
		rdr = std::io::BufReader::with_capacity(1 << 22, std::fs::File::open(path).unwrap());
	}

	let mut id_of: HashMap<u64, u32> = HashMap::with_capacity(if pb { 1 << 20 } else { (records / 4) as usize });
	let cap = if pb { 1 << 20 } else { records as usize };
	let (mut ids, mut sizes, mut gets) = (Vec::with_capacity(cap), Vec::with_capacity(cap), Vec::with_capacity(cap));
	let mut kbuf = [0u8; 255];
	let mut size0 = Vec::new();
	let mut buf = [0u8; 25];

	for _ in 0..records {
		let (cmd, key, raw) = if pb {
			if rdr.read_exact(&mut buf[..18]).is_err() { break }
			let klen = buf[9] as usize;
			if rdr.read_exact(&mut kbuf[..klen]).is_err() { break }
			// FNV-1a then splitmix: a 64-bit id per key string
			let mut h: u64 = 0xcbf2_9ce4_8422_2325;
			for &b in &kbuf[..klen] { h = (h ^ b as u64).wrapping_mul(0x100_0000_01b3) }
			(buf[8], splitmix(h), u32::from_le_bytes(buf[10..14].try_into().unwrap()))
		} else {
			if rdr.read_exact(&mut buf).is_err() { break }
			(buf[8], u64::from_le_bytes(buf[9..17].try_into().unwrap()), u32::from_le_bytes(buf[17..21].try_into().unwrap()))
		};
		let size = raw.max(1) + meta;
		let next = id_of.len() as u32;
		let id = *id_of.entry(key).or_insert(next);
		if id == next { size0.push(size) }
		ids.push(id);
		sizes.push(size);
		gets.push(cmd == 0);
	}

	(ids, sizes, gets, size0)
}

fn report(cfg: &Config, sim: &Sim, start: Instant) {
	let c = &sim.c;
	let per = |x: u64| x as f64 / c.gets.max(1) as f64;
	let per_req = |x: u64| x as f64 / c.reqs.max(1) as f64;
	let cps = c.cps.max(1) as f64;
	println!(
		"config evict={:?}/{} keys={} requests={} alpha={} cache={}B fast={}B rate={} mini_cap={}B promote={} gap_mult={} sweep={} ewma={} shift_every={} shift_frac={}",
		cfg.evict, cfg.evict_samples, cfg.keys, cfg.requests, cfg.alpha, sim.cap, sim.fast, cfg.rate, sim.mini_cap,
		if cfg.promote_gap { "gap" } else { "always" }, cfg.gap_mult, cfg.sweep, cfg.ewma, cfg.shift_every, cfg.shift_frac,
	);
	println!("requests             {} (gets {} sets {})  meta={}B", c.reqs, c.gets, c.sets, cfg.meta);
	println!("hit_ratio            {:.4}", per(c.hits));
	println!("fast_hit_ratio       exact {:.4}  approx {:.4}", per(c.exact_fast_hits), per(c.approx_fast_hits));
	println!(
		"migrations_per_kreq  exact {:.2} (promo {:.2} demo {:.2})  approx {:.2} (promo {:.2} demo {:.2}, forced {:.2})",
		1e3 * per_req(c.exact_promotions + c.exact_demotions), 1e3 * per_req(c.exact_promotions), 1e3 * per_req(c.exact_demotions),
		1e3 * per_req(c.approx_promotions + c.approx_demotions), 1e3 * per_req(c.approx_promotions),
		1e3 * per_req(c.approx_demotions), 1e3 * per_req(c.approx_forced),
	);
	println!(
		"fast_set             overlap {:.4} of exact bytes  extra {:.4} of F  approx_fill {:.4} of F",
		c.overlap / cps, c.extra / cps, c.approx_fill / cps,
	);
	println!(
		"tstar                final {:.0} reqs  mean_rel_err {:.4} (over {} checkpoints)",
		sim.tstar, c.tstar_rel_err / c.tstar_cps.max(1) as f64, c.tstar_cps,
	);
	let stackless = c.ops_mini + c.ops_sweep + c.ops_backstop + c.ops_evict;
	println!(
		"worker_ops_per_req   stack {:.2} (relink {:.2} cursor {:.2})  stackless {:.2} (mini {:.3} sweep {:.2} backstop {:.2} evict {:.2})",
		per_req(c.ops_list + c.ops_cursor), per_req(c.ops_list), per_req(c.ops_cursor),
		per_req(stackless), per_req(c.ops_mini), per_req(c.ops_sweep), per_req(c.ops_backstop), per_req(c.ops_evict),
	);
	println!("elapsed              {:.1}s", start.elapsed().as_secs_f64());
}
