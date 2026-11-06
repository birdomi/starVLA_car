#!/bin/bash
# car-v3 val (eval_val_car_v3.py) of several checkpoints on one node: each spec split into 4 parts (one per GPU), all
# specs at once, then merged into <OUT>/<name>/result.json.
# Usage: [STRIDE=30] [submit] eval_val_car_v3.sbatch <name>=<ckpt>[=reset] [...]   (reset: --memory_reset)
#SBATCH -J eval_val_car_v3
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
REPO=/scratch/t509a04/workspace/starVLA_car
OUT=${OUT:-${REPO}/playground/eval_results/car_v3_val}
STRIDE=${STRIDE:-30}
PARTS=4
module load miniconda3 cuda/12.8
source activate /home01/t509a04/envs/starVLA
ulimit -u $(ulimit -Hu) 2>/dev/null || true
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 RANK=0
export PYTHONWARNINGS="ignore:The video decoding and encoding capabilities of torchvision are deprecated"
export PYTHONPATH=${REPO} TRITON_CACHE_DIR=/tmp/triton_${SLURM_JOB_ID}
cd ${REPO}
echo "[$(date)] stride ${STRIDE} -> ${OUT}"
pids=()
for spec in "$@"; do
  IFS='=' read -r name ckpt reset <<< "${spec}"
  flag=""; [ "${reset}" = reset ] && flag="--memory_reset"
  mkdir -p ${OUT}/${name}
  for i in $(seq 0 $((PARTS - 1))); do
    CUDA_VISIBLE_DEVICES=$i python examples/realRobots/H1_2_car_real/eval_val_car_v3.py --ckpt ${ckpt} --out ${OUT}/${name} \
      --stride ${STRIDE} --part ${i}/${PARTS} ${flag} > ${OUT}/${name}/part${i}.log 2>&1 &
    pids+=($!)
  done
done
fail=0
for p in "${pids[@]}"; do wait $p || fail=1; done
echo "[$(date)] parts done (fail=${fail})"
for spec in "$@"; do
  IFS='=' read -r name ckpt reset <<< "${spec}"
  echo "===== ${name}"
  python examples/realRobots/H1_2_car_real/eval_val_car_v3.py --ckpt ${ckpt} --out ${OUT}/${name} --merge_parts ${PARTS} 2>&1 | tail -n 8 || echo "merge failed: see ${OUT}/${name}/part*.log"
done
echo "[$(date)] done"
exit ${fail}
