#!/usr/bin/env bash
# QwenGR00T_Regressor (Muon) launcher for sim benchmarks. Same env as EgoDex launch_regressor_yaml.sh:
# NCCL_P2P_DISABLE=1 is set by default: PCIe P2P allreduce hangs on this box (verified 2026-09-07); host-path NCCL works.
# ZeRO stage 0 (Muon needs full per-matrix grads), torchrun, single-thread workers.
# Usage: bash examples/simBenchmarks/launch_regressor_sim_local.sh <yaml> [gpus=0,1,2,3] [master_port=29630] [extra train args...]
set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
YAML=${1:?usage: $0 <yaml> [gpus] [master_port] [extra args]}
GPUS=${2:-0,1,2,3}
PORT=${3:-29630}
shift 3 2>/dev/null || shift $#
NPROC=$(echo "$GPUS" | tr ',' '\n' | wc -l)
LOG=$ROOT/regressor_$(basename "$YAML" .yaml)_$(date +%m%d_%H%M).log
cd "$ROOT"
setsid nohup env PYTHONPATH=$ROOT CUDA_VISIBLE_DEVICES=$GPUS \
  ACCELERATE_DEEPSPEED_ZERO_STAGE=0 NCCL_P2P_DISABLE=${NCCL_P2P_DISABLE:-1} OMP_NUM_THREADS=1 STARVLA_MAIN_TORCH_THREADS=8 \
  "$HOME/miniconda3/envs/starVLA/bin/torchrun" --nproc_per_node="$NPROC" --master_port="$PORT" \
  starVLA/training/train_starvla.py --config_yaml "$YAML" "$@" >> "$LOG" 2>&1 < /dev/null &
echo "[launcher] pid $!  yaml=$(basename "$YAML")  gpus=$GPUS  port=$PORT  log=$LOG"
