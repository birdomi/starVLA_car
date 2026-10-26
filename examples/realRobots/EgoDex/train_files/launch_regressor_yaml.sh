#!/bin/bash
# Generic launcher: launch_regressor_yaml.sh <yaml> [master_port=29621] [wait_pid] [gpus=4,5,6,7]
# Env: ZeRO-0 (required by Muon, fine for AdamW), single-thread workers, 8 torch threads per rank.
set -u
ROOT=/mnt/DATA2/yyg/starVLA
YAML=${1:?usage: $0 <yaml> [master_port] [wait_pid] [gpus]}
PORT=${2:-29621}
WAIT_PID=${3:-}
GPUS=${4:-4,5,6,7}
NPROC=$(echo "$GPUS" | tr ',' '\n' | grep -c .)
LOG=$ROOT/train_$(basename "$YAML" .yaml)_$(date +%m%d_%H%M).log
cd "$ROOT"
if [ -n "$WAIT_PID" ]; then
  echo "[launcher] waiting for PID $WAIT_PID to exit..."
  while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 60; done
  echo "[launcher] PID $WAIT_PID exited at $(date)"; sleep 120
fi
setsid nohup env PYTHONPATH=$ROOT CUDA_VISIBLE_DEVICES=$GPUS NCCL_P2P_DISABLE=1 \
  ACCELERATE_DEEPSPEED_ZERO_STAGE=0 OMP_NUM_THREADS=1 STARVLA_MAIN_TORCH_THREADS=8 \
  /home/user/miniconda3/envs/starVLA/bin/torchrun --nproc_per_node=$NPROC --master_port=$PORT \
  starVLA/training/train_starvla.py --config_yaml "$YAML" >> "$LOG" 2>&1 < /dev/null &
echo "[launcher] launched $(basename "$YAML") on GPUs $GPUS (port $PORT) pid=$! , log: $LOG"
