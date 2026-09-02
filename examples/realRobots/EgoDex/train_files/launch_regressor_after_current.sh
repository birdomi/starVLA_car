#!/bin/bash
# Waits for the current EgoDex stage2 run (PID given as $1) to finish,
# then launches QwenGR00T_Regressor training (VLM frozen) on GPUs 4-7.
set -u
WAIT_PID=${1:-2061735}
ROOT=/mnt/DATA2/yyg/starVLA
YAML=$ROOT/examples/realRobots/EgoDex/train_files/starvla_qwengr00t_regressor_egodex.yaml
LOG=$ROOT/regressor_train_$(date +%m%d_%H%M).log

cd "$ROOT"

echo "[watcher] waiting for PID $WAIT_PID to exit..."
while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 60; done
echo "[watcher] PID $WAIT_PID exited at $(date)"
sleep 120  # let GPU memory drain

setsid nohup env PYTHONPATH=$ROOT CUDA_VISIBLE_DEVICES=4,5,6,7 \
  /home/user/miniconda3/envs/starVLA/bin/torchrun --nproc_per_node=4 --master_port=29618 \
  starVLA/training/train_starvla.py \
  --config_yaml "$YAML" \
  >> "$LOG" 2>&1 < /dev/null &
echo "[watcher] regressor training launched, log: $LOG"
