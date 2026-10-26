#!/bin/bash
# RoboCasa rc100: [train -> seeded 24-task eval -> CSV] for one or more yamls, run back to back on ONE GPU group.
#
#   GPUS=4,5,6,7 TRAIN_PORT=29750 EVAL_BASE_PORT=6900 \
#     bash examples/simBenchmarks/Robocasa_tabletop/train_files/run_rc100_train_eval.sh 50k_dit_v2 50k_actionmap_v3 ...
#
# A spec is the yaml name between "starvla_qwen35_gr00t_rc100_" and "_robocasa.yaml" (or a path to a yaml).
# The evaluated checkpoint is the LAST one (trainer.max_train_steps). Eval scenes are seeded (simulation_env.py,
# base seed 7, PYTHONHASHSEED pinned), so every run is scored on the same 24 x 50 episodes.
# Per-task eval logs are copied to results/eval_logs/<run_id>/ so results survive checkpoint deletion.
# Env: GPUS (4 ids), TRAIN_PORT, EVAL_BASE_PORT, SKIP_TRAIN=1 (eval an existing checkpoint only),
#      EVAL_SEED=<base> (another scene-seed set; results/logs/videos get a _seed<base> suffix, default set = 7),
#      SKIP_GPU_WAIT=1 (start even if other jobs are using the GPUs).
set -u
cd "$(dirname "$0")/../../../.."   # repo root
GPUS=${GPUS:-4,5,6,7}
TRAIN_PORT=${TRAIN_PORT:-29750}
EVAL_BASE_PORT=${EVAL_BASE_PORT:-6900}
NPROC=$(echo "$GPUS" | tr ',' '\n' | wc -l)
SFX="${EVAL_SEED:+_seed${EVAL_SEED}}"                 # result-name suffix for a non-default scene-seed set
EXTRA_EVAL="${EVAL_SEED:+--args.seed ${EVAL_SEED}}"    # forwarded to simulation_env.py
YAML_DIR=examples/simBenchmarks/Robocasa_tabletop/train_files

# first port >= $1 such that $2 consecutive ports are all free (other jobs on this box grab 29xxx / 6xxx ports too)
free_ports() {
  local p=$1 n=${2:-1} i busy
  while :; do
    busy=0
    for ((i = 0; i < n; i++)); do ss -ltn | grep -q ":$((p + i)) " && { busy=1; break; }; done
    [ $busy -eq 0 ] && { echo "$p"; return; }
    p=$((p + 1))
  done
}

# wait until no process is using the target GPUs
UUIDS="$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader | awk -F', ' -v g=",$GPUS," 'index(g, ","$1",") {print $2}' | paste -sd'|')"
while [ "${SKIP_GPU_WAIT:-0}" != 1 ] && nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader | grep -qE "$UUIDS"; do
  echo "[rc100] GPUs $GPUS busy, waiting... $(date)"; sleep 300
done

for spec in "$@"; do
  if [ -f "$spec" ]; then YAML=$spec
  elif [ -f "$YAML_DIR/starvla_qwen35_gr00t_rc100_${spec}_robocasa.yaml" ]; then YAML=$YAML_DIR/starvla_qwen35_gr00t_rc100_${spec}_robocasa.yaml
  else YAML=$YAML_DIR/starvla_${spec%%_*}_gr00t_rc100_${spec#*_}_robocasa.yaml; fi   # e.g. vlact_50k_dit_v2
  [ -f "$YAML" ] || { echo "[rc100] no yaml for spec '$spec' ($YAML)"; exit 1; }
  RID=$(grep '^run_id:' "$YAML" | awk '{print $2}')
  STEPS=$(grep -E '^\s+max_train_steps:' "$YAML" | awk '{print $2}')
  # non-Qwen3.5 backbones keep their prefix in TAG so result files never collide with the baselines
  TAG=$(basename "$YAML" .yaml | sed 's/^starvla_qwen35_gr00t_rc100_//;s/^starvla_\([a-z0-9]*\)_gr00t_rc100_/\1_/;s/_robocasa$//')
  # launch_zero_stage: 2 in the yaml -> accelerate + DeepSpeed ZeRO-2 bf16 (fp32 master weights); default torchrun ZeRO-0
  ZERO=$(grep -E '^launch_zero_stage:' "$YAML" | awk '{print $2}'); ZERO=${ZERO:-0}
  CKPT=playground/Checkpoints/$RID/checkpoints/steps_${STEPS}_pytorch_model.pt

  if [ "${SKIP_TRAIN:-0}" != 1 ]; then
    LOG=$PWD/regressor_rc100_${TAG}_$(date +%m%d_%H%M).log
    TRAIN_PORT=$(free_ports "$TRAIN_PORT" 1)
    echo "[rc100] TRAIN START $TAG ($RID, ${STEPS} steps) GPUs=$GPUS port=$TRAIN_PORT zero=$ZERO log=$LOG $(date)"
    if [ "$ZERO" = 2 ]; then
      env PYTHONPATH=$PWD CUDA_VISIBLE_DEVICES=$GPUS \
          NCCL_P2P_DISABLE=1 OMP_NUM_THREADS=1 STARVLA_MAIN_TORCH_THREADS=8 \
          "$HOME/miniconda3/envs/starVLA/bin/accelerate" launch \
          --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \
          --num_processes "$NPROC" --main_process_port "$TRAIN_PORT" \
          starVLA/training/train_starvla.py --config_yaml "$YAML" >> "$LOG" 2>&1
    else
      env PYTHONPATH=$PWD CUDA_VISIBLE_DEVICES=$GPUS \
          ACCELERATE_DEEPSPEED_ZERO_STAGE=0 NCCL_P2P_DISABLE=1 OMP_NUM_THREADS=1 STARVLA_MAIN_TORCH_THREADS=8 \
          "$HOME/miniconda3/envs/starVLA/bin/torchrun" --nproc_per_node=$NPROC --master_port=$TRAIN_PORT \
          starVLA/training/train_starvla.py --config_yaml "$YAML" >> "$LOG" 2>&1
    fi
    RC=$?
    echo "[rc100] TRAIN END $TAG rc=$RC $(date)"
    [ $RC -ne 0 ] && { echo "[rc100] $TAG training failed - stop"; exit 1; }
    sleep 30
  fi
  [ -f "$CKPT" ] || { echo "[rc100] checkpoint missing: $CKPT - stop"; exit 1; }

  EVAL_BASE_PORT=$(free_ports "$EVAL_BASE_PORT" "$NPROC")
  echo "[rc100] EVAL START $TAG ($CKPT, scene seeds ${EVAL_SEED:-7}+i) eval ports ${EVAL_BASE_PORT}+ $(date)"
  ROBOCASA_ACTION_MODE=rel ROBOCASA_ACTION_MODE_KEYS=left_arm,right_arm,waist \
  GPU_IDS="$(echo "$GPUS" | tr ',' ' ')" BASE_PORT=$EVAL_BASE_PORT VIDEO_SUFFIX="$SFX" \
    bash examples/simBenchmarks/Robocasa_tabletop/eval_files/batch_eval_args.sh "$CKPT" 1 720 12 "$EXTRA_EVAL" \
    > "results/robocasa_rc100_${TAG}${SFX}_full_eval.log" 2>&1
  L=$(ls -dt "$CKPT.log"/eval_2* 2>/dev/null | head -1)
  if [ -n "$L" ]; then
    mkdir -p "results/eval_logs/$RID$SFX"
    cp -r "$L" "results/eval_logs/$RID$SFX/"
    { echo "task,success_rate"
      for f in "$L"/eval_env_*.log; do
        echo "$(basename "$f" | sed 's/eval_env_gr1_unified_//;s/_GR1.*//'),$(grep -ah 'Success rate' "$f" | tail -1 | grep -oE '[0-9.]+$')"
      done | sort; } > "results/robocasa_rc100_${TAG}${SFX}_eval_results.csv"
    echo "[rc100] EVAL END $TAG -> results/robocasa_rc100_${TAG}${SFX}_eval_results.csv" \
         "(mean $(awk -F, 'NR>1 && $2!="" {s+=$2; n++} END {if (n) printf "%.1f%% over %d tasks", 100*s/n, n}' "results/robocasa_rc100_${TAG}${SFX}_eval_results.csv")) $(date)"
  else
    echo "[rc100] EVAL END $TAG - no eval log dir found under $CKPT.log $(date)"
  fi
  TRAIN_PORT=$((TRAIN_PORT + 1)); sleep 30
done
echo "[rc100] ALL DONE $(date)"
