#!/bin/bash
# EgoDex raw zip -> 10 fps 320x192 per-episode frame files (tools/egodex_frames.py), one split per job, CPU-only work
# on an aarch64 GPU node (288 cores). Unzips <zips>/<split>.zip into <raw>/ first if needed, deletes the raw split after.
# usage: sbatch examples/realRobots/EgoDex/train_files/hangang/egodex_frames.sbatch <split> [keep_raw]
#SBATCH -J egodex_frames
#SBATCH -p natStr
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --exclusive
#SBATCH --cpus-per-task=288
#SBATCH -t 24:00:00
#SBATCH --comment="field=robotics;appl=pytorch"
#SBATCH -o /scratch/t509a04/workspace/logs/%x_%j.out
set -eo pipefail
SPLIT=${1:?split}
KEEP_RAW=${2:-}
E=/scratch/t509a04/data/EgoDex
module load miniconda3
source activate /home01/t509a04/envs/starVLA
cd /scratch/t509a04/workspace/starVLA_car
if [ ! -d ${E}/raw/${SPLIT} ]; then
  echo "[$(date)] unzip ${SPLIT}"; (cd ${E}/raw && unzip -q -n ${E}/zips/${SPLIT}.zip)
fi
echo "[$(date)] convert ${SPLIT}"
python examples/realRobots/EgoDex/tools/egodex_frames.py ${E}/raw ${E}/frames --splits ${SPLIT} --workers 256
[ -z "${KEEP_RAW}" ] && [ "${SPLIT}" != test ] && rm -rf ${E}/raw/${SPLIT} || true
echo "[$(date)] done"
