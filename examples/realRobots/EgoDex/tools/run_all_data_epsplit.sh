#!/bin/bash
# Stage per-episode parquets for all subsets (pilot extra first, verify gate), no live files touched.
set -u
PY=/home/user/miniconda3/envs/starVLA/bin/python
TOOL=/mnt/DATA2/yyg/starVLA/examples/realRobots/EgoDex/tools/convert_data_epsplit.py
ROOT=/mnt/DATA2/yyg/egodex
run_one() { echo "=== [$(date +%H:%M:%S)] $1 ==="; $PY "$TOOL" convert "$ROOT/$1" "$ROOT/$1_epsplit" --workers 32 && $PY "$TOOL" verify "$ROOT/$1" "$ROOT/$1_epsplit" --samples 30; }
run_one egodex_lerobot_extra || { echo "PILOT FAILED"; exit 1; }
echo "=== PILOT PASSED ==="
FAILED=""
for n in egodex_lerobot_part1 egodex_lerobot_part2 egodex_lerobot_part3 egodex_lerobot_part4 egodex_lerobot_part5 trex_lerobot; do run_one $n || FAILED="$FAILED $n"; done
[ -n "$FAILED" ] && { echo "FAILED:$FAILED"; exit 1; }
echo "=== ALL DATA STAGED AND VERIFIED at $(date) ==="
