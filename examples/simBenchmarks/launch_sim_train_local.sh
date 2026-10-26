#!/usr/bin/env bash
# Local launcher for sim-benchmark training on this machine (accelerate + DeepSpeed ZeRO-2).
# Usage: bash examples/simBenchmarks/launch_sim_train_local.sh <yaml> [gpus=4,5,6,7] [master_port=29650] [extra train args...]
# Example:
#   bash examples/simBenchmarks/launch_sim_train_local.sh \
#       examples/simBenchmarks/Robocasa_tabletop/train_files/starvla_qwengr00t_robocasa_gr1_local.yaml 4,5,6,7 29650
set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
YAML=${1:?usage: $0 <yaml> [gpus] [master_port] [extra args]}
GPUS=${2:-4,5,6,7}
PORT=${3:-29650}
shift 3 2>/dev/null || shift $#
NPROC=$(echo "$GPUS" | tr ',' '\n' | wc -l)
LOG=$ROOT/sim_train_$(basename "$YAML" .yaml)_$(date +%m%d_%H%M).log
cd "$ROOT"
setsid nohup env PYTHONPATH=$ROOT CUDA_VISIBLE_DEVICES=$GPUS NCCL_P2P_DISABLE=${NCCL_P2P_DISABLE:-1} OMP_NUM_THREADS=1 STARVLA_MAIN_TORCH_THREADS=8 \
  "$HOME/miniconda3/envs/starVLA/bin/accelerate" launch \
    --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \
    --num_processes "$NPROC" --main_process_port "$PORT" \
    starVLA/training/train_starvla.py --config_yaml "$YAML" "$@" >> "$LOG" 2>&1 < /dev/null &
echo "[launcher] pid $!  yaml=$(basename "$YAML")  gpus=$GPUS  port=$PORT  log=$LOG"
