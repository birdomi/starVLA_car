#!/bin/bash
# QwenGR00T_DualSystem on RoboTwin 2.0 (robotwin_clean50): stage 1 (System 2 + Qwen) -> stage 2 (System 1) -> 50-task sim
# eval (demo_clean, TEST_NUM episodes per task, state sent; run_robotwin_train_eval.sh rounds / resume / watchdog), back to
# back on ONE GPU group.
#
#   GPUS=4,5,6,7 TRAIN_PORT=29780 setsid nohup bash examples/simBenchmarks/Robotwin/train_files/dual_system/run_dualsystem_robotwin.sh \
#     > dualsystem_robotwin_chain.log 2>&1 < /dev/null &
#
# Stage 2 starts from stage 1's last checkpoint (stage2 yaml trainer.pretrained_checkpoint). SKIP_STAGE1=1 starts at
# stage 2, SKIP_STAGE2=1 goes straight to the eval, SKIP_EVAL=1 stops after training; STAGE2_YAML=<yaml> picks another
# stage-2 yaml. The eval's results: results/robotwin_<EVAL_TAG>_demo_clean_n<TEST_NUM>_results.csv
# (EVAL_TAG default dualsystem_s1).
set -u
cd "$(dirname "$0")/../../../../.."   # repo root
GPUS=${GPUS:-4,5,6,7}
TRAIN_PORT=${TRAIN_PORT:-29780}
NPROC=$(echo "$GPUS" | tr ',' '\n' | wc -l)
DIR=examples/simBenchmarks/Robotwin/train_files/dual_system

free_port() {
  local p=$1
  while ss -ltn | grep -q ":$p "; do p=$((p + 1)); done
  echo "$p"
}

train() {   # $1 = yaml, $2 = tag
  local yaml=$1 tag=$2 log port rc
  log=$PWD/regressor_dualsystem_robotwin_${tag}_$(date +%m%d_%H%M).log
  port=$(free_port "$TRAIN_PORT")
  echo "[dual-rt] TRAIN START $tag ($(grep '^run_id:' "$yaml" | awk '{print $2}')) GPUs=$GPUS port=$port log=$log $(date)"
  env PYTHONPATH=$PWD CUDA_VISIBLE_DEVICES=$GPUS \
      ACCELERATE_DEEPSPEED_ZERO_STAGE=0 NCCL_P2P_DISABLE=1 OMP_NUM_THREADS=1 STARVLA_MAIN_TORCH_THREADS=8 \
      "$HOME/miniconda3/envs/starVLA/bin/torchrun" --nproc_per_node="$NPROC" --master_port="$port" \
      starVLA/training/train_starvla.py --config_yaml "$yaml" >> "$log" 2>&1
  rc=$?
  echo "[dual-rt] TRAIN END $tag rc=$rc $(date)"
  return $rc
}

S1=$DIR/stage1_system2_robotwin.yaml
S2=${STAGE2_YAML:-$DIR/stage2_system1_robotwin.yaml}
CKPT=$(grep -E '^\s+pretrained_checkpoint:' "$S2" | awk '{print $2}')

if [ "${SKIP_STAGE1:-0}" != 1 ]; then
  train "$S1" stage1_s2 || { echo "[dual-rt] stage 1 failed - stop"; exit 1; }
  sleep 30
fi
if [ "${SKIP_STAGE2:-0}" != 1 ]; then
  [ -f "$CKPT" ] || { echo "[dual-rt] stage-1 checkpoint missing: $CKPT - stop"; exit 1; }
  train "$S2" "$(basename "$S2" .yaml | sed 's/_robotwin$//')" || { echo "[dual-rt] stage 2 failed - stop"; exit 1; }
  sleep 30
fi
if [ "${SKIP_EVAL:-0}" != 1 ]; then
  echo "[dual-rt] EVAL $S2 $(date)"
  GPUS=$GPUS SKIP_TRAIN=1 EVAL_TAG=${EVAL_TAG:-dualsystem_s1} \
    bash examples/simBenchmarks/Robotwin/train_files/run_robotwin_train_eval.sh "$S2" || { echo "[dual-rt] eval failed - stop"; exit 1; }
fi
echo "[dual-rt] ALL DONE $(date)"
