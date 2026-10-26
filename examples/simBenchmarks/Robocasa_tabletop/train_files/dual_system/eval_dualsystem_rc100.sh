#!/bin/bash
# Seeded 24-task RoboCasa sim eval of a QwenGR00T_DualSystem checkpoint (same scenes as the rc100 baselines: scene
# seed 7 + episode, 50 episodes per task, 720 steps, 12 of the 20 System 1 actions executed per call).
#
#   GPUS="4 5 6 7" bash examples/simBenchmarks/Robocasa_tabletop/train_files/dual_system/eval_dualsystem_rc100.sh <ckpt.pt> <tag>
#
# Dual-system specifics vs run_rc100_train_eval.sh: ABSOLUTE actions (ROBOCASA_ACTION_MODE unset = abs; the baselines
# use rel), state is sent (both systems are state-conditioned), frames resized to 256 x 256 (= frame_size /
# obs_image_size). The policy server runs predict_action: System 2 then System 1 synchronously (plan age 0).
# Writes results/robocasa_rc100_<tag>_eval_results.csv and copies the per-task logs to results/eval_logs/<run_id>/.
set -u
cd "$(dirname "$0")/../../../../.."   # repo root
CKPT=${1:?usage: $0 <ckpt.pt> <tag>}
TAG=${2:?usage: $0 <ckpt.pt> <tag>}
GPUS=${GPUS:-4 5 6 7}
NGPU=$(echo $GPUS | wc -w)
RID=$(basename "$(dirname "$(dirname "$CKPT")")")

p=${EVAL_BASE_PORT:-6900}   # first port with NGPU consecutive free ports
while :; do
  busy=0
  for ((i = 0; i < NGPU; i++)); do ss -ltn | grep -q ":$((p + i)) " && { busy=1; break; }; done
  [ $busy -eq 0 ] && break
  p=$((p + 1))
done

echo "[dual-eval] START $TAG ($CKPT) GPUs=$GPUS ports $p+ $(date)"
unset ROBOCASA_ACTION_MODE ROBOCASA_ACTION_MODE_KEYS
GPU_IDS="$GPUS" BASE_PORT=$p \
  bash examples/simBenchmarks/Robocasa_tabletop/eval_files/batch_eval_args.sh "$CKPT" 1 720 12 "--args.resize_size 256 256" \
  > "results/robocasa_rc100_${TAG}_full_eval.log" 2>&1
L=$(ls -dt "$CKPT.log"/eval_2* 2>/dev/null | head -1)
[ -n "$L" ] || { echo "[dual-eval] no eval log dir under $CKPT.log $(date)"; exit 1; }
mkdir -p "results/eval_logs/$RID"
cp -r "$L" "results/eval_logs/$RID/"
CSV=results/robocasa_rc100_${TAG}_eval_results.csv
{ echo "task,success_rate"
  for f in "$L"/eval_env_*.log; do
    echo "$(basename "$f" | sed 's/eval_env_gr1_unified_//;s/_GR1.*//'),$(grep -ah 'Success rate' "$f" | tail -1 | grep -oE '[0-9.]+$')"
  done | sort; } > "$CSV"
echo "[dual-eval] END $TAG -> $CSV (mean $(awk -F, 'NR>1 && $2!="" {s+=$2; n++} END {if (n) printf "%.1f%% over %d tasks", 100*s/n, n}' "$CSV")) $(date)"
