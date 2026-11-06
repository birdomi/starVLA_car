#!/bin/bash
# RoboCasa multi-node train -> 24-task sim eval in ONE job on KISTI Hangang (default 2 nodes = 8 GPU).
# Global batch = per_device_batch_size x 4 x nodes (e.g. 32 x 8 = 256 on 2 nodes).
#   training: one torchrun per node over Slingshot, same ZeRO-0 recipe as train_rc100.sbatch;
#   eval    : the 24 tasks are dealt round-robin to the nodes, each running 4 policy servers + its share of sims,
#             so 2 nodes evaluate in ~half the time; the per-task logs of all nodes are merged into one CSV.
# Separate from train_rc100.sbatch / eval_rc100.sbatch so that jobs already running those files are not disturbed.
# Submit from the repo root:
#   sbatch [--nodes N] [-J name] examples/simBenchmarks/Robocasa_tabletop/train_files/hangang/train_eval_rc_mn.sbatch <yaml>
# Resume after the 24h walltime: EXTRA_ARGS="--trainer.is_resume True" sbatch ... (same yaml)
#SBATCH -J rc_mn
#SBATCH -p natStr
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=72
#SBATCH --exclusive
#SBATCH -t 24:00:00
#SBATCH --comment="field=robotics;appl=deepspeed"
#SBATCH -o /scratch/t509a04/workspace/logs/%x_%j.out

# no -e: training ends FAILED at NCCL teardown even after saving its final checkpoint
set -uo pipefail

YAML=${1:?usage: sbatch train_eval_rc_mn.sbatch <yaml>}
SAVE_INTERVAL=${SAVE_INTERVAL:-5000}   # yaml says 25000; save more often so a 24h timeout loses little
EXTRA_ARGS=${EXTRA_ARGS:-}

REPO=/scratch/t509a04/workspace/starVLA_car
cd ${REPO}

RID=$(grep '^run_id:' "$YAML" | awk '{print $2}')
STEPS=$(grep -E '^\s+max_train_steps:' "$YAML" | awk '{print $2}')
TAG=$(basename "$YAML" .yaml | sed 's/^starvla_//;s/_robocasa$//')
CKPT=playground/Checkpoints/$RID/checkpoints/steps_${STEPS}_pytorch_model.pt

module load miniconda3 cuda/12.8 aws-ofi-nccl/1.20.0 libfabric/2.3.1
source activate /home01/t509a04/envs/starVLA

# Slingshot (CXI) for inter-node NCCL, per Hangang docs; a 1-node job gets no CXI domain, so NVLink only
if [ "${SLURM_NNODES}" -gt 1 ]; then
  export NCCL_NET="AWS Libfabric" FI_PROVIDER=cxi NCCL_IB_DISABLE=1 NCCL_PROTO=^LL128
fi
export NCCL_DEBUG=${NCCL_DEBUG:-INFO} NCCL_DEBUG_SUBSYS=INIT,NET   # check log for "Using network AWS Libfabric"
export HF_HUB_OFFLINE=1 RAYON_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WANDB_MODE=${WANDB_MODE:-disabled}
# torchvision_av decoding warns once per frame read (~5GB of log per 50k run)
export PYTHONWARNINGS="ignore:The video decoding and encoding capabilities of torchvision are deprecated"
export ACCELERATE_DEEPSPEED_ZERO_STAGE=0 OMP_NUM_THREADS=1 STARVLA_MAIN_TORCH_THREADS=8
export PYTHONPATH=${REPO}

export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
export MASTER_PORT=$((20000 + SLURM_JOB_ID % 20000))
export YAML SAVE_INTERVAL EXTRA_ARGS CKPT TAG

echo "[$(date)] ${SLURM_NNODES} nodes ($(scontrol show hostnames "$SLURM_JOB_NODELIST" | tr '\n' ' ')), master ${MASTER_ADDR}:${MASTER_PORT}"
echo "  yaml=${YAML} save_interval=${SAVE_INTERVAL} extra=${EXTRA_ARGS}"

# ---------------- train ----------------
# Triton cache per node on local /tmp: ranks of several nodes compiling into the shared ~/.triton on /home01 race and
# read half-written metadata (JSONDecodeError at the first step).
srun --kill-on-bad-exit=1 bash -c '
export TRITON_CACHE_DIR=/tmp/triton_${SLURM_JOB_ID}_${SLURM_NODEID}
torchrun --nnodes ${SLURM_NNODES} --nproc_per_node 4 --node_rank ${SLURM_NODEID} \
  --master_addr ${MASTER_ADDR} --master_port ${MASTER_PORT} \
  starVLA/training/train_starvla.py \
  --config_yaml ${YAML} \
  --trainer.save_interval ${SAVE_INTERVAL} \
  ${EXTRA_ARGS}
'
echo "[$(date)] training exited rc=$? -> eval"
sleep 30

# ---------------- eval ----------------
[ -f "$CKPT" ] || { echo "checkpoint missing: $CKPT"; exit 1; }
export STARVLA_PYTHON=/home01/t509a04/envs/starVLA/bin/python
export ROBOCASA_PYTHON=/home01/t509a04/envs/robocasa/bin/python
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
# ~28 sim/server processes share each node; multithreaded BLAS oversubscribes it
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
# training used relative arm/waist targets (yaml action_mode: rel), so the sim must integrate them the same way
export ROBOCASA_ACTION_MODE=rel ROBOCASA_ACTION_MODE_KEYS=left_arm,right_arm,waist

mkdir -p results "$CKPT.log"
MARK="$CKPT.log/.eval_start_${SLURM_JOB_ID}"
touch "$MARK"
echo "[$(date)] eval ${TAG} ckpt=${CKPT} on ${SLURM_NNODES} nodes"
# Each node: its round-robin share of the 24 tasks; ports offset per node; start staggered so every node gets its
# own timestamped eval_<date> log dir. "" as 5th arg: GR00T-style heads consume state, so keep sending it.
srun bash -c '
mapfile -t ALL < <(grep -oE "gr1_unified/[A-Za-z]+_GR1ArmsAndWaistFourierHands_Env" \
  examples/simBenchmarks/Robocasa_tabletop/eval_files/batch_eval_args.sh)
MINE=""
for i in "${!ALL[@]}"; do [ $((i % SLURM_NNODES)) -eq ${SLURM_NODEID} ] && MINE+=" ${ALL[$i]}"; done
sleep $((5 * SLURM_NODEID))
echo "[$(date)] $(hostname) node ${SLURM_NODEID}:${MINE}"
EVAL_ENVS="${MINE}" GPU_IDS="0 1 2 3" BASE_PORT=$((20000 + SLURM_JOB_ID % 20000 + 10 * SLURM_NODEID)) \
  bash examples/simBenchmarks/Robocasa_tabletop/eval_files/batch_eval_args.sh "${CKPT}" 1 720 12 "" \
  > "results/robocasa_rc100_${TAG}_full_eval_node${SLURM_NODEID}.log" 2>&1
'

DIRS=$(find "$CKPT.log" -mindepth 1 -maxdepth 1 -type d -name 'eval_2*' -newer "$MARK" | sort)
[ -n "$DIRS" ] || { echo "no eval log dir under $CKPT.log"; exit 1; }
mkdir -p "results/eval_logs/$RID"
cp -r $DIRS "results/eval_logs/$RID/"
CSV=results/robocasa_rc100_${TAG}_eval_results.csv
{ echo "task,success_rate"
  for f in $(for d in $DIRS; do ls "$d"/eval_env_*.log; done); do
    echo "$(basename "$f" | sed 's/eval_env_gr1_unified_//;s/_GR1.*//'),$(grep -ah 'Success rate' "$f" | tail -1 | grep -oE '[0-9.]+$')"
  done | sort; } > "$CSV"
echo "[$(date)] done -> $CSV (mean $(awk -F, 'NR>1 && $2!="" {s+=$2; n++} END {if (n) printf "%.1f%% over %d tasks", 100*s/n, n}' "$CSV"))"
