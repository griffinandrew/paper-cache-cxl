#!/bin/bash
# Builds and runs paper-benchmark-cxl once per (paper-cache feature, benchmark
# feature) pair below, against every trace in $TRACES_DIR, capturing GET/SET
# latency stats per run to $OUT_DIR. Used to compare all_dram against each of
# the four hybrid-cache designs (lru/lfu/two_q/fifo) under identical
# conditions (single client, default 24GB cache / 4GB fast tier, TBB
# allocator). Assumes paper-benchmark-cxl's Cargo.toml has paper-cache
# path-pointed at a local checkout of this crate, with exactly one
# uncommented `features=[...]` line for the paper-cache dependency.
set -uo pipefail

export LIBCLANG_PATH=$(python3 -c "import clang, os; print(os.path.join(os.path.dirname(clang.__file__), 'native'))")
cd /home/griff/work/paper-benchmark-cxl

TRACES_DIR=/home/griff/final_traces
TRACES="standard_web low_alpha_cold uniform_baseline"
OUT_DIR=/tmp/full_matrix
mkdir -p "$OUT_DIR"

# name : paper-cache-feature : benchmark-build-feature (empty = none)
CONFIGS=(
  "all_dram:all_dram:"
  "lru:lru_hybrid_cache:hybrid"
  "lfu:lfu_hybrid_cache:hybrid_lfu"
  "two_q:two_q_hybrid_cache:hybrid_2q"
  "fifo:fifo_hybrid_cache:hybrid_fifo"
)

for entry in "${CONFIGS[@]}"; do
  IFS=':' read -r name pc_feat bench_feat <<< "$entry"
  echo "=================================================="
  echo "=== CONFIG: $name  (paper-cache=$pc_feat, benchmark=$bench_feat) ==="
  echo "=================================================="

  sed -i "s/^features=\[.*\]/features=[\"${pc_feat}\"]/" Cargo.toml
  grep -n "^features=" Cargo.toml

  if [ -n "$bench_feat" ]; then
    cargo +nightly build --release --features "$bench_feat" > "$OUT_DIR/build_${name}.log" 2>&1
  else
    cargo +nightly build --release > "$OUT_DIR/build_${name}.log" 2>&1
  fi
  build_rc=$?
  if [ $build_rc -ne 0 ]; then
    echo "!!! BUILD FAILED for $name, see $OUT_DIR/build_${name}.log"
    tail -40 "$OUT_DIR/build_${name}.log"
    continue
  fi
  echo "build ok for $name"

  for trace in $TRACES; do
    echo "--- running $name / $trace ---"
    LOG="$OUT_DIR/${name}_${trace}.log"
    ./target/release/paper-benchmark --trace-path "$TRACES_DIR/${trace}.bin" -c 1 > "$LOG" 2>&1
    run_rc=$?
    echo "exit code: $run_rc"
    if [ $run_rc -ne 0 ]; then
      echo "!!! RUN FAILED for $name/$trace"
      tail -40 "$LOG"
    fi
    echo "--- dmesg check ---"
    dmesg | tail -5 | grep -i "segfault\|oom" && echo "!!! POSSIBLE CRASH/OOM DETECTED (verify timestamp -- dmesg tail may be stale from an earlier run)"
  done
done

echo "=================================================="
echo "ALL RUNS COMPLETE"
echo "=================================================="
