#!/bin/bash
# Launch an H1-2 car training yaml (run inside tmux so ranks get nice 0).
# Usage: run_car_yaml.sh <yaml> [gpus=0,1,2,3] [port=29751] [extra train_starvla.py args...]
# ZeRO-0 is required by the Muon optimizer; NCCL P2P is off on this box.
YAML=${1:?usage: $0 <yaml> [gpus] [port] [args...]}; GPUS=${2:-0,1,2,3}; PORT=${3:-29751}; shift 3 2>/dev/null || shift $#
NPROC=$(echo "$GPUS" | tr ',' '\n' | grep -c .)
cd /mnt/DATA2/yyg/starVLA
LOG=/mnt/DATA2/yyg/starVLA/train_$(basename "$YAML" .yaml)_$(date +%m%d_%H%M).log
echo "[launcher] $(basename $YAML) on GPUs $GPUS -> $LOG"
env PYTHONPATH=$PWD CUDA_VISIBLE_DEVICES=$GPUS NCCL_P2P_DISABLE=1 ACCELERATE_DEEPSPEED_ZERO_STAGE=0 \
  OMP_NUM_THREADS=1 STARVLA_MAIN_TORCH_THREADS=8 \
  /home/user/miniconda3/envs/starVLA/bin/torchrun --nproc_per_node=$NPROC --master_port=$PORT \
  starVLA/training/train_starvla.py --config_yaml "$YAML" "$@" > "$LOG" 2>&1
echo "EXIT $?" >> "$LOG"
