#!/bin/bash
# Pilot (extra) -> verify gate -> convert all remaining EgoDex subsets.
set -u
PY=/home/user/miniconda3/envs/starVLA/bin/python
TOOL=/mnt/DATA2/yyg/starVLA/examples/realRobots/EgoDex/tools/convert_epsplit.py
ROOT=/mnt/DATA2/yyg/egodex

run_one() {
  local name=$1
  echo "=== [$(date +%H:%M:%S)] convert $name ==="
  $PY "$TOOL" convert "$ROOT/$name" "$ROOT/${name}_epsplit" --workers 60 || return 1
  echo "=== [$(date +%H:%M:%S)] verify $name ==="
  $PY "$TOOL" verify "$ROOT/$name" "$ROOT/${name}_epsplit" --samples 20 || return 1
}

# pilot first; abort everything if it fails
run_one egodex_lerobot_extra || { echo "PILOT FAILED - aborting"; exit 1; }
echo "=== PILOT PASSED, converting remaining subsets ==="

FAILED=""
for name in egodex_lerobot_part1 egodex_lerobot_part2 egodex_lerobot_part3 \
            egodex_lerobot_part4 egodex_lerobot_part5 trex_lerobot; do
  run_one "$name" || FAILED="$FAILED $name"
done

if [ -n "$FAILED" ]; then
  echo "FAILED subsets:$FAILED"
  exit 1
fi
echo "=== ALL SUBSETS CONVERTED AND VERIFIED at $(date) ==="
