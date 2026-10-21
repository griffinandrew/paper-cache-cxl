# approx_tier: can a sampled mini stack drive tiering?

This is a standalone simulator with no dependencies, and it is not part of the `paper-cache` build:

```
cd tools/approx_tier && cargo run --release -- [key=value ...]
```

It runs one read-through LRU cache of `C` bytes with a fast tier of `F` bytes. Two placements are compared over the **same** residency: eviction is exact LRU for both, so only fast/slow placement differs.

- **exact**: the `lru-compact-hybrid` rule. The fast tier is the top `F` bytes of the recency order, kept by a boundary cursor. Every slow hit promotes, and the cursor's key demotes.
- **approx**: placement keeps no ordering, only a per-object last-access stamp.
  - **T\* estimate:** a mini LRU over the hash-sampled keys (PaperCache's sampling rule, rate `R`), sized `F * R`, estimates T\*. T\* is the access age at the fast tier's edge, read as now minus the stamp of the mini's tail.
  - **Threshold demotion:** a CLOCK hand steps `sweep` fast keys per request and demotes any key older than T\*.
  - **Backstop:** if fast bytes still exceed `F`, the hand demotes the oldest of 8 keys it looks at. This stands in for the byte gate, and these demotions are counted as **forced**.
  - **`promote=always`:** every slow hit promotes, as in exact.
  - **`promote=gap`:** a slow hit promotes only if its previous inter-access gap was under `gap_mult * T*`. This is a reuse filter.

Output:
- the fast-hit ratio of each placement
- migrations per 1,000 requests (admissions land fast in both, so most demotions are admit-then-demote)
- the byte overlap of the two fast sets at 36 checkpoints after a 10% warm-up
- T\*'s mean relative error against the true edge age (the age of exact's cursor key)

## Results

Setup: 1M keys, 20M requests, Zipf, sizes log-uniform 64 B to 16 KiB, `C` = 30% of all bytes (883 MB), `F` = 25% of `C` (221 MB), seed 1, one run each.

### Does T\* track the real edge?

| R | mini size | T\* mean rel. error | approx fast-hit (exact 0.6007) | overlap |
|---|---|---|---|---|
| 0.001 | 2.2 MB | 14.1% | 0.5929 | 0.915 |
| 0.01 | 22 MB | 4.7% | 0.5985 | 0.966 |
| 0.1 | 221 MB | 1.2% | 0.6002 | 0.985 |

(These runs used alpha 0.9, `sweep=16`, `promote=always`.)

Yes. At 1% sampling the estimate is within about 5%, and placement driven by it loses about 0.2 points of fast-hit ratio.

### Is the threshold what makes it work? (`promote=always`)

| alpha | exact | approx, backstop only (`sweep=0`) | approx, T\* + backstop (`sweep=16`) | forced share of demotions, sweep 0 / 16 |
|---|---|---|---|---|
| 0.6 | 0.2133 | 0.2114 | 0.2111 | 100% / 22% |
| 0.9 | 0.6007 | 0.5987 | 0.5985 | 100% / 19% |
| 1.2 | 0.9410 | 0.9406 | 0.9402 | 100% / 13% |
| 0.9, 20% of ranks reshuffled every 2M requests | 0.5988 | 0.5969 | 0.5957 | 100% / 12% |

No. The backstop alone ("cap at F, evict the oldest of 8 sampled") is an approximate LRU in the Redis style, and it is already within 0.2 points of exact. So T\* adds nothing to the hit ratio when the goal is to mimic LRU.

What T\* does add is that most demotions become proactive: forced demotions drop from 100% to 12–22% of demotions, and the tier rests at about 97–99% of `F` instead of pinned at 100%. That is headroom the byte gate would otherwise have to create by making sets wait. That benefit is worth something on the real system, but this simulator does not model it.

### Where the mini changes behaviour: a promotion threshold (`promote=gap`, `sweep=16`)

| alpha | gap_mult | fast-hit (exact) | migrations/kreq (exact) | tier fill |
|---|---|---|---|---|
| 0.9 | 1 | 0.5483 (0.6007) | 317 (582) | 0.69 F |
| 0.9 | 2 | 0.5757 | 399 | 0.80 F |
| 0.9 | 4 | 0.5939 | 504 | 0.93 F |
| 0.6 | 4 | 0.2063 (0.2133) | 967 (1067) | 0.94 F |
| 1.2 | 4 | 0.9392 (0.9410) | 73 (84) | 0.90 F |

This is a real knob, and T\* sets it automatically for the workload. At `gap_mult=4` it removes 9–13% of migrations for a cost of 0.2–0.7 fast-hit points. At `gap_mult=1` it removes 46% of migrations for a cost of 5.2 points. Whether that trade pays off depends on what a migration costs on CXL against what a slow hit costs, which needs the real box.

### Dropping the stack entirely: approximate eviction (`evict=sampled|pool`)

With `evict=sampled`, the victim is the oldest stamp among `evict_samples` random residents. There is no list; the random pick stands in for sampling the object map. With `evict=pool`, the same samples also feed a 16-entry pool of the oldest candidates seen so far (Redis 3.0's eviction pool), and the victim is the oldest entry in the pool. Placement in all runs is T\* plus the backstop (`sweep=16`, `promote=always`).

Overall hit ratio (exact LRU, then the change in points):

| workload | exact | sampled, 5 | sampled, 8 | pool, 5 | pool, 8 |
|---|---|---|---|---|---|
| alpha 0.6 | 0.4974 | -0.45 | -0.21 | -0.19 | -0.08 |
| alpha 0.9 | 0.7870 | -0.41 | -0.19 | -0.17 | -0.08 |
| alpha 1.2 | 0.9700 | -0.05 | -0.02 | -0.02 | -0.01 |
| alpha 0.9, shifting | 0.7793 | -0.40 | -0.19 | -0.16 | -0.07 |

The fast-hit ratio is unchanged to four digits in every run. Evicted objects are old, so they are almost never fast.

Per-object state with no stack is a 4 B stamp, plus the mini at `R` × the arena node (about 0.4 B/object at R = 1%). That compares with `ARENA_STACK_DRAM_OVERHEAD` = 40 B/object today. Not modelled: the cost of sampling a real object map (cheap with the merged store's slab, while DashMap has no random access), and a wider stamp clock.

## Caveats

- The traces are synthetic (Zipf with a static size per key). PaperCache's trace files have not been run through this yet, and that is the next step. It needs no CXL hardware.
- Byte movement, latency and queueing are not modelled: a migration counts as one event.
- Approximate eviction (above) is tested on synthetic traces only.
- The CLOCK hand walks a swap-removed vector rather than a ring, which is fine for an estimate.
