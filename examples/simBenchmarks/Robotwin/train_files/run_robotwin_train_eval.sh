#!/bin/bash
# RoboTwin 2.0: [train -> 50-task eval (resumable rounds) -> CSV] for one or more yamls, back to back on ONE GPU group.
#
#   GPUS=4,5,6,7 bash examples/simBenchmarks/Robotwin/train_files/run_robotwin_train_eval.sh rt50k_dit_v2 rt50k_actionmap_v3
#
# A spec is the yaml name between "starvla_qwen35_gr00t_" and "_robotwin.yaml" (or a path to a yaml).
# Eval: start_eval.sh on all 50 tasks, TEST_NUM episodes each (RoboTwin seeds 100000+i, expert-failed seeds skipped, so every
# checkpoint sees the same episodes), state sent to the policy. RoboTwin sims hang now and then, so the eval runs in rounds:
# per-seed results are cached in results/robotwin_resume/<run_id>/ (ROBOTWIN_RESUME_DIR) and each round only re-runs tasks
# that have < TEST_NUM valid episodes; a watchdog SIGSEGVs sims whose log is idle too long.
# Env: GPUS, TRAIN_PORT, EVAL_BASE_PORT, TEST_NUM (20), MODE (demo_clean), MAX_ROUNDS (30), SKIP_TRAIN=1,
#      EVAL_TAG (result / resume name instead of the yaml-derived tag; single spec only).
set -u
cd "$(dirname "$0")/../../../.."   # repo root
GPUS=${GPUS:-4,5,6,7}
TRAIN_PORT=${TRAIN_PORT:-29770}
EVAL_BASE_PORT=${EVAL_BASE_PORT:-5800}
TEST_NUM=${TEST_NUM:-20}
MODE=${MODE:-demo_clean}
MAX_ROUNDS=${MAX_ROUNDS:-30}
NPROC=$(echo "$GPUS" | tr ',' '\n' | wc -l)
YAML_DIR=examples/simBenchmarks/Robotwin/train_files
TASKS_FILE=results/robotwin_all_tasks.txt      # the 50 RoboTwin 2.0 tasks, one per line
WATCHDOG=results/robotwin_stall_watchdog.sh
source examples/simBenchmarks/Robotwin/eval_files/env_local.sh

# first port >= $1 such that $2 consecutive ports are all free (other jobs on this box grab 29xxx / 5xxx ports too)
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
while nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader | grep -qE "$UUIDS"; do
  echo "[robotwin] GPUs $GPUS busy, waiting... $(date)"; sleep 300
done

for spec in "$@"; do
  if [ -f "$spec" ]; then YAML=$spec; else YAML=$YAML_DIR/starvla_qwen35_gr00t_${spec}_robotwin.yaml; fi
  [ -f "$YAML" ] || { echo "[robotwin] no yaml for spec '$spec' ($YAML)"; exit 1; }
  RID=$(grep '^run_id:' "$YAML" | awk '{print $2}')
  STEPS=$(grep -E '^\s+max_train_steps:' "$YAML" | awk '{print $2}')
  TAG=${EVAL_TAG:-$(basename "$YAML" .yaml | sed 's/^starvla_qwen35_gr00t_//;s/_robotwin$//')}
  CKPT=playground/Checkpoints/$RID/checkpoints/steps_${STEPS}_pytorch_model.pt

  if [ "${SKIP_TRAIN:-0}" != 1 ]; then
    LOG=$PWD/regressor_robotwin_${TAG}_$(date +%m%d_%H%M).log
    TRAIN_PORT=$(free_ports "$TRAIN_PORT" 1)
    echo "[robotwin] TRAIN START $TAG ($RID, ${STEPS} steps) GPUs=$GPUS port=$TRAIN_PORT log=$LOG $(date)"
    env PYTHONPATH=$PWD CUDA_VISIBLE_DEVICES=$GPUS \
        ACCELERATE_DEEPSPEED_ZERO_STAGE=0 NCCL_P2P_DISABLE=1 OMP_NUM_THREADS=1 STARVLA_MAIN_TORCH_THREADS=8 \
        "$HOME/miniconda3/envs/starVLA/bin/torchrun" --nproc_per_node=$NPROC --master_port=$TRAIN_PORT \
        starVLA/training/train_starvla.py --config_yaml "$YAML" >> "$LOG" 2>&1
    RC=$?
    echo "[robotwin] TRAIN END $TAG rc=$RC $(date)"
    [ $RC -ne 0 ] && { echo "[robotwin] $TAG training failed - stop"; exit 1; }
    sleep 30
  fi
  [ -f "$CKPT" ] || { echo "[robotwin] checkpoint missing: $CKPT - stop"; exit 1; }

  RESUME=results/robotwin_resume/$RID
  RES=results/robotwin_${TAG}_${MODE}_n${TEST_NUM}_results.csv
  mkdir -p "$RESUME"
  echo "[robotwin] EVAL START $TAG ($CKPT, $MODE, $TEST_NUM eps/task) $(date)"
  for round in $(seq 1 "$MAX_ROUNDS"); do
    # harvest: tasks with >= TEST_NUM valid cached episodes are done (first TEST_NUM valid seeds count)
    python3 - "$RESUME" "$MODE" "$TAG" "$TEST_NUM" "$RES" <<'PY'
import glob, json, os, sys
resume, mode, tag, n, out = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), sys.argv[5]
rows = []
for f in sorted(glob.glob(f"{resume}/*_{mode}_{tag}_st*.json")):
    d = json.load(open(f))
    v = [d[k] for k in sorted(d, key=int) if d[k]["valid"]][:n]
    if len(v) >= n:
        rows.append((os.path.basename(f).split(f"_{mode}_")[0], sum(x["succ"] for x in v), len(v)))
with open(out, "w") as fh:
    fh.write("task,success,episodes,success_rate\n")
    for t, s, k in sorted(rows): fh.write(f"{t},{s},{k},{s / k:.2f}\n")
PY
    TODO=$RESUME/todo_tasks.txt
    comm -23 <(sort "$TASKS_FILE") <(tail -n +2 "$RES" | cut -d, -f1 | sort) > "$TODO"
    left=$(wc -l < "$TODO")
    echo "[robotwin] $TAG round $round: $(( $(wc -l < "$RES") - 1 )) tasks done, $left to (re)run $(date)"
    [ "$left" -eq 0 ] && break
    LOG_ROOT=$PWD/results/robotwin_logs/$RID/round$round
    mkdir -p "$LOG_ROOT"
    PORT=$(free_ports "$EVAL_BASE_PORT" "$NPROC")
    env CUDA_VISIBLE_DEVICES=$GPUS ROBOTWIN_TEST_NUM=$TEST_NUM ROBOTWIN_SEND_STATE=1 ROBOTWIN_RESUME_DIR=$PWD/$RESUME \
        ROBOTWIN_LOG_ROOT=$LOG_ROOT PYTHONFAULTHANDLER=1 ROBOTWIN_RT_DENOISER=none \
        bash examples/simBenchmarks/Robotwin/eval_files/start_eval.sh -m "$MODE" -n "$TAG" -p "$PORT" -j 1 -c "$CKPT" "$TODO" \
        > "results/robotwin_logs/${RID}_round${round}_eval.log" 2>&1 &
    SE=$!
    bash "$WATCHDOG" "$LOG_ROOT" 1200 & WD=$!
    wait $SE; kill $WD 2>/dev/null
    # leftovers of THIS checkpoint/round only
    for p in $(pgrep -f "[s]erver_policy.py --ckpt_path $CKPT") $(pgrep -f "[s]cript/eval_policy.py.*--ckpt_setting $TAG "); do kill -9 "$p" 2>/dev/null; done
    sleep 5
  done
  echo "[robotwin] EVAL END $TAG -> $RES (mean $(awk -F, 'NR>1 {s+=$4; n++} END {if (n) printf "%.1f%% over %d tasks", 100*s/n, n}' "$RES")) $(date)"
  TRAIN_PORT=$((TRAIN_PORT + 1)); sleep 30
done
echo "[robotwin] ALL DONE $(date)"
