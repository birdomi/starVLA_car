#!/bin/bash
# RoboCasa GR1 24-task simulation eval (50 seeded episodes/task) of one checkpoint on KISTI Hangang (1 node, 4 GPU).
# Mirrors the eval half of run_rc100_train_eval.sh. Submit from the repo root, chained after a training job:
#   [submit] --dependency=afterok:<train_jobid> examples/simBenchmarks/Robocasa_tabletop/train_files/hangang/eval_rc100.sbatch <yaml>
# Evaluates playground/Checkpoints/<run_id>/checkpoints/steps_<max_train_steps>_pytorch_model.pt;
# results -> results/robocasa_rc100_<tag>_eval_results.csv, per-task logs -> results/eval_logs/<run_id>/.
#SBATCH -J rc100_eval
#SBATCH -p natStr
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=72
#SBATCH --exclusive
#SBATCH -t 24:00:00
#SBATCH --comment="field=robotics;appl=deepspeed"
#SBATCH -o /scratch/t509a04/workspace/logs/%x_%j.out

set -uo pipefail

YAML=${1:?usage: eval_rc100.sbatch <yaml>}
REPO=/scratch/t509a04/workspace/starVLA_car
cd ${REPO}

RID=$(grep '^run_id:' "$YAML" | awk '{print $2}')
STEPS=$(grep -E '^\s+max_train_steps:' "$YAML" | awk '{print $2}')
TAG=$(basename "$YAML" .yaml | sed 's/^starvla_//;s/_robocasa$//')
CKPT=${CKPT:-playground/Checkpoints/$RID/checkpoints/steps_${STEPS}_pytorch_model.pt}
[ -f "$CKPT" ] || { echo "checkpoint missing: $CKPT"; exit 1; }
# an intermediate checkpoint (CKPT=...steps_N...) gets a _stepN suffix so it never overwrites the final eval's CSV/logs
CKPT_STEP=$(basename "$CKPT" | grep -oE 'steps_[0-9]+' | cut -d_ -f2)
SFX=""; [ -n "$CKPT_STEP" ] && [ "$CKPT_STEP" != "$STEPS" ] && SFX=_step${CKPT_STEP}
TAG=${TAG}${SFX}

module load miniconda3 cuda/12.8
export STARVLA_PYTHON=/home01/t509a04/envs/starVLA/bin/python
export ROBOCASA_PYTHON=/home01/t509a04/envs/robocasa/bin/python
export HF_HUB_OFFLINE=1 RAYON_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
# ~28 sim/server processes share the node; multithreaded BLAS oversubscribes it
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
# training used relative arm/waist targets (yaml action_mode: rel), so the sim must integrate them the same way
export ROBOCASA_ACTION_MODE=rel ROBOCASA_ACTION_MODE_KEYS=left_arm,right_arm,waist
# episode-memory models: the server carries every episode's memory across calls (the client sends slot / time / reset)
grep -qE '^\s+name: QwenGR00T_EpisodeMemory' "$YAML" && export ROBOCASA_MEMORY=1

echo "[$(date)] $(hostname) eval ${TAG} ckpt=${CKPT}"
mkdir -p results
# "" as 5th arg: GR00T-style heads consume state, so keep sending it (the default would pass --args.no_send_state)
GPU_IDS="0 1 2 3" BASE_PORT=$((20000 + SLURM_JOB_ID % 20000)) \
  bash examples/simBenchmarks/Robocasa_tabletop/eval_files/batch_eval_args.sh "$CKPT" 1 720 12 "" \
  > "results/robocasa_rc100_${TAG}_full_eval.log" 2>&1

L=$(ls -dt "$CKPT.log"/eval_2* 2>/dev/null | head -1)
[ -n "$L" ] || { echo "no eval log dir under $CKPT.log"; exit 1; }
mkdir -p "results/eval_logs/$RID$SFX"
cp -r "$L" "results/eval_logs/$RID$SFX/"
CSV=results/robocasa_rc100_${TAG}_eval_results.csv
{ echo "task,success_rate"
  for f in "$L"/eval_env_*.log; do
    echo "$(basename "$f" | sed 's/eval_env_gr1_unified_//;s/_GR1.*//'),$(grep -ah 'Success rate' "$f" | tail -1 | grep -oE '[0-9.]+$')"
  done | sort; } > "$CSV"
echo "[$(date)] done -> $CSV (mean $(awk -F, 'NR>1 && $2!="" {s+=$2; n++} END {if (n) printf "%.1f%% over %d tasks", 100*s/n, n}' "$CSV"))"
