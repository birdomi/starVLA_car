#!/bin/bash
# QwenGR00T_DualSystem on the FULL RoboCasa GR1 set (1000 demos x 24 tasks): stage 1 (System 2 + Qwen, 40k) -> stage 2
# (System 1, 0.8 s = 16-action chunk, plan + state as conditions, 10k) -> seeded 24-task sim eval, back to back on ONE GPU group.
#
#   GPUS=4,5,6,7 TRAIN_PORT=29790 setsid nohup bash examples/simBenchmarks/Robocasa_tabletop/train_files/dual_system/run_dualsystem_rcfull.sh \
#     > dualsystem_rcfull_chain.log 2>&1 < /dev/null &
#
# SKIP_STAGE1=1 / SKIP_STAGE2=1 skip a stage; STAGE1_YAML / STAGE2_YAML / EVAL_TAG override the defaults.
set -u
cd "$(dirname "$0")/../../../../.."   # repo root
GPUS=${GPUS:-4,5,6,7}
TRAIN_PORT=${TRAIN_PORT:-29790}
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
  echo "[dual-full] TRAIN START $tag ($(grep '^run_id:' "$yaml" | awk '{print $2}')) GPUs=$GPUS port=$port log=$log $(date)"
  env PYTHONPATH=$PWD CUDA_VISIBLE_DEVICES=$GPUS \
      ACCELERATE_DEEPSPEED_ZERO_STAGE=0 NCCL_P2P_DISABLE=1 OMP_NUM_THREADS=1 STARVLA_MAIN_TORCH_THREADS=8 \
      "$HOME/miniconda3/envs/starVLA/bin/torchrun" --nproc_per_node="$NPROC" --master_port="$port" \
      starVLA/training/train_starvla.py --config_yaml "$yaml" >> "$log" 2>&1
  rc=$?
  echo "[dual-full] TRAIN END $tag rc=$rc $(date)"
  return $rc
}

S1=${STAGE1_YAML:-$DIR/stage1_system2_rcfull.yaml}
S2=${STAGE2_YAML:-$DIR/stage2_system1_plancond_rcfull.yaml}
CKPT1=$(grep -E '^\s+pretrained_checkpoint:' "$S2" | awk '{print $2}')
RID2=$(grep '^run_id:' "$S2" | awk '{print $2}')
STEPS2=$(grep -E '^\s+max_train_steps:' "$S2" | awk '{print $2}')
CKPT2=playground/Checkpoints/$RID2/checkpoints/steps_${STEPS2}_pytorch_model.pt

if [ "${SKIP_STAGE1:-0}" != 1 ]; then
  train "$S1" stage1_s2_rcfull || { echo "[dual-full] stage 1 failed - stop"; exit 1; }
  sleep 30
fi
[ -f "$CKPT1" ] || { echo "[dual-full] stage-1 checkpoint missing: $CKPT1 - stop"; exit 1; }
if [ "${SKIP_STAGE2:-0}" != 1 ]; then
  train "$S2" "$(basename "$S2" .yaml)" || { echo "[dual-full] stage 2 failed - stop"; exit 1; }
  sleep 30
fi
[ -f "$CKPT2" ] || { echo "[dual-full] stage-2 checkpoint missing: $CKPT2 - stop"; exit 1; }
GPUS="$(echo "$GPUS" | tr ',' ' ')" bash "$DIR/eval_dualsystem_rc100.sh" "$CKPT2" "${EVAL_TAG:-$(echo "${RID2#starvla_}" | sed "s/_robocasa_gr1$//")_${STEPS2}}"
echo "[dual-full] eval rc=$? $(date)"
echo "[dual-full] ALL DONE $(date)"
