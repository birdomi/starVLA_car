#!/bin/bash
# Run the past+future window sweep (H=8,16,32,64) sequentially on GPUs 4-7.
# Usage: run_pf_sweep.sh [wait_pid]   (waits for that PID to exit before starting)
set -u
ROOT=/mnt/DATA2/yyg/starVLA; TF=$ROOT/examples/realRobots/EgoDex/train_files
cd "$ROOT"
if [ -n "${1:-}" ]; then echo "[sweep] waiting for PID $1"; while kill -0 "$1" 2>/dev/null; do sleep 60; done; sleep 120; fi
for H in 8 16 32 64; do
  YAML=$TF/starvla_qwengr00t_regressor_pf${H}_part1_50k.yaml
  while ss -tlnp 2>/dev/null | grep -q ":29621"; do sleep 30; done   # wait for the port to free up
  $TF/launch_regressor_yaml.sh "$YAML" 29621
  sleep 90
  PT=$(pgrep -u user -x pt_elastic | head -1)
  echo "[sweep] H=$H running as torchrun $PT at $(date)"
  while [ -n "$PT" ] && kill -0 "$PT" 2>/dev/null; do sleep 120; done
  echo "[sweep] H=$H finished at $(date)"; sleep 60
done
echo "[sweep] ALL DONE $(date)"
