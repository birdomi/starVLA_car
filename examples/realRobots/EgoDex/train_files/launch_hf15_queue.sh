#!/bin/bash
# Queue the two unfrozen-VLM two-frame experiments on the same 4 GPUs, one after the other:
#   1) WTA regressor (Muon head)   2) flow-matching DiT-B (starts when 1 exits)
# usage: launch_hf15_queue.sh [gpus=4,5,6,7] [port_base=29641] [order=regressor,fm]
set -u
ROOT=/mnt/DATA2/yyg/starVLA; D=$ROOT/examples/realRobots/EgoDex/train_files
GPUS=${1:-4,5,6,7}; PORT=${2:-29641}; ORDER=${3:-regressor,fm}
cd $ROOT
PREV=""
for name in ${ORDER//,/ }; do
  Y=$D/starvla_qwengr00t_${name}_hf15_rel_part1_unfreeze_50k.yaml
  OUT=$(bash $D/launch_regressor_yaml.sh "$Y" "$PORT" "$PREV" "$GPUS")
  echo "$OUT"
  PREV=$(echo "$OUT" | grep -o 'pid=[0-9]*' | cut -d= -f2)
  PORT=$((PORT + 1))
done
echo "[queue] done at $(date)"
