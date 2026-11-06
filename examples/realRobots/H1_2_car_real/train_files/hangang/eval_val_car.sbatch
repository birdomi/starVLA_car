#!/bin/bash
# Streamed val eval (eval_val_stream.py) of car-1015-v3-ep checkpoints on one Hangang node (4 GPU): every val episode
# (265) walked from frame 0, one query per STRIDE frames, memory carried for QwenGR00T_EpisodeMemory, every other
# framework queried statelessly at the same frames with the same noise (paired). Each model is split into 4 parts
# (--part i/4, one per GPU), all models at once, then merged into <out>/<name>/stream.json.
# Usage: [STRIDE=30] [DTYPE=bfloat16] [submit] eval_val_car.sbatch <name>=<ckpt.pt>=<config.yaml> [...]
#SBATCH -J eval_val_car
#SBATCH -p natStr
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=72
#SBATCH --exclusive
#SBATCH -t 6:00:00
#SBATCH --comment="field=robotics;appl=deepspeed"
#SBATCH -o /scratch/t509a04/workspace/logs/%x_%j.out

set -eo pipefail
[ $# -ge 1 ] || { echo "usage: eval_val_car.sbatch <name>=<ckpt>=<config> [...]"; exit 1; }

REPO=/scratch/t509a04/workspace/starVLA_car
OUT=${OUT:-${REPO}/playground/eval_results/car1015_v3_val}
STRIDE=${STRIDE:-30}
DTYPE=${DTYPE:-bfloat16}
BATCH=${BATCH:-16}
PARTS=4

module load miniconda3 cuda/12.8
source activate /home01/t509a04/envs/starVLA
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4 RANK=0
export PYTHONWARNINGS="ignore:The video decoding and encoding capabilities of torchvision are deprecated"
export PYTHONPATH=${REPO} TRITON_CACHE_DIR=/tmp/triton_${SLURM_JOB_ID}
cd ${REPO}
echo "[$(date)] stride ${STRIDE} dtype ${DTYPE} batch ${BATCH} -> ${OUT}"

pids=()
for spec in "$@"; do
  IFS='=' read -r name ckpt cfg <<< "${spec}"
  mkdir -p ${OUT}/${name}
  for i in $(seq 0 $((PARTS - 1))); do
    CUDA_VISIBLE_DEVICES=$i python examples/realRobots/H1_2_car_real/eval_val_stream.py --ckpt ${ckpt} --config ${cfg} \
      --dataset car-1015-v3-ep --robot h1_2_car_3cam_headstate --out ${OUT}/${name} --stride ${STRIDE} \
      --dtype ${DTYPE} --batch ${BATCH} --part ${i}/${PARTS} > ${OUT}/${name}/part${i}.log 2>&1 &
    pids+=($!)
  done
done
fail=0
for p in "${pids[@]}"; do wait $p || fail=1; done
echo "[$(date)] parts done (fail=${fail})"

for spec in "$@"; do
  IFS='=' read -r name ckpt cfg <<< "${spec}"
  echo "===== ${name}  (${ckpt})"
  python examples/realRobots/H1_2_car_real/eval_val_stream.py --ckpt ${ckpt} --config ${cfg} --robot h1_2_car_3cam_headstate \
    --out ${OUT}/${name} --merge_parts ${PARTS} 2>&1 | tail -n 25 || echo "merge failed: see ${OUT}/${name}/part*.log"
done
echo "[$(date)] done"
exit ${fail}
