#!/bin/bash
# Launches QwenGR00T_Regressor MoE+Muon training (VLM frozen) on GPUs 4-7.
# ZeRO stage 0 is REQUIRED: torch.optim.Muon needs full per-matrix grads.
# Optional $1: PID to wait for before launching (e.g. a run occupying the GPUs).
set -u
ROOT=/mnt/DATA2/yyg/starVLA
YAML=$ROOT/examples/realRobots/EgoDex/train_files/starvla_qwengr00t_regressor_muon_egodex.yaml
LOG=$ROOT/regressor_muon_train_$(date +%m%d_%H%M).log

cd "$ROOT"

if [ "${1:-}" != "" ]; then
  echo "[watcher] waiting for PID $1 to exit..."
  while kill -0 "$1" 2>/dev/null; do sleep 60; done
  echo "[watcher] PID $1 exited at $(date)"
  sleep 120  # let GPU memory drain
fi

setsid nohup env PYTHONPATH=$ROOT CUDA_VISIBLE_DEVICES=4,5,6,7 \
  ACCELERATE_DEEPSPEED_ZERO_STAGE=0 OMP_NUM_THREADS=1 STARVLA_MAIN_TORCH_THREADS=8 \
  /home/user/miniconda3/envs/starVLA/bin/torchrun --nproc_per_node=4 --master_port=29619 \
  starVLA/training/train_starvla.py \
  --config_yaml "$YAML" \
  >> "$LOG" 2>&1 < /dev/null &
echo "[watcher] Muon regressor training launched, log: $LOG"
