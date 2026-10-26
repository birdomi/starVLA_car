#!/bin/bash
# QwenGR00T_DualSystem on RoboCasa rc100: stage 1 (System 2 + Qwen) -> stage 2 (System 1), back to back on ONE GPU group.
#
#   GPUS=4,5,6,7 TRAIN_PORT=29760 setsid nohup bash examples/simBenchmarks/Robocasa_tabletop/train_files/dual_system/run_dualsystem_rc100.sh \
#     > dualsystem_rc100_chain.log 2>&1 < /dev/null &
#
# Stage 2 starts from stage 1's last checkpoint (stage2 yaml trainer.pretrained_checkpoint). SKIP_STAGE1=1 runs stage 2 only;
# STAGE2_YAML=<yaml> picks another stage-2 yaml (e.g. stage2_system1_noise_rc100.yaml).
set -u
cd "$(dirname "$0")/../../../../.."   # repo root
GPUS=${GPUS:-4,5,6,7}
TRAIN_PORT=${TRAIN_PORT:-29760}
NPROC=$(echo "$GPUS" | tr ',' '\n' | wc -l)
DIR=examples/simBenchmarks/Robocasa_tabletop/train_files/dual_system

free_port() {
  local p=$1
  while ss -ltn | grep -q ":$p "; do p=$((p + 1)); done
  echo "$p"
}

train() {   # $1 = yaml, $2 = tag
  local yaml=$1 tag=$2 log port rc
  log=$PWD/regressor_dualsystem_${tag}_$(date +%m%d_%H%M).log
  port=$(free_port "$TRAIN_PORT")
  echo "[dual] TRAIN START $tag ($(grep '^run_id:' "$yaml" | awk '{print $2}')) GPUs=$GPUS port=$port log=$log $(date)"
  env PYTHONPATH=$PWD CUDA_VISIBLE_DEVICES=$GPUS \
      ACCELERATE_DEEPSPEED_ZERO_STAGE=0 NCCL_P2P_DISABLE=1 OMP_NUM_THREADS=1 STARVLA_MAIN_TORCH_THREADS=8 \
      "$HOME/miniconda3/envs/starVLA/bin/torchrun" --nproc_per_node="$NPROC" --master_port="$port" \
      starVLA/training/train_starvla.py --config_yaml "$yaml" >> "$log" 2>&1
  rc=$?
  echo "[dual] TRAIN END $tag rc=$rc $(date)"
  return $rc
}

S1=$DIR/stage1_system2_rc100.yaml
S2=${STAGE2_YAML:-$DIR/stage2_system1_rc100.yaml}
CKPT=$(grep -E '^\s+pretrained_checkpoint:' "$S2" | awk '{print $2}')

if [ "${SKIP_STAGE1:-0}" != 1 ]; then
  train "$S1" stage1_s2 || { echo "[dual] stage 1 failed - stop"; exit 1; }
  sleep 30
fi
[ -f "$CKPT" ] || { echo "[dual] stage-1 checkpoint missing: $CKPT - stop"; exit 1; }
train "$S2" "$(basename "$S2" .yaml | sed 's/_rc100$//')" || { echo "[dual] stage 2 failed - stop"; exit 1; }
echo "[dual] ALL DONE $(date)"
