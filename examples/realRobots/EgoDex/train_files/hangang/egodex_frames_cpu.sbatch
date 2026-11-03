#!/bin/bash
# EgoDex raw zip -> 10 fps 320x192 per-episode frame files (tools/egodex_frames.py) on an x86 CPU node (256 cores).
# Submit from an x86 login node (clogin01..04), e.g. from glogin01:
#   ssh clogin01 "cd /scratch/t509a04/workspace/starVLA_car && sbatch examples/realRobots/EgoDex/train_files/hangang/egodex_frames_cpu.sbatch <split>"
# Unzips <zips>/<split>.zip into <raw>/ first if needed; deletes the raw split after (except test).
#SBATCH -J egodex_frames
#SBATCH -p normal_cpu
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH -t 24:00:00
#SBATCH --comment="field=robotics;appl=python"
#SBATCH -o /scratch/t509a04/workspace/logs/%x_%j.out
set -eo pipefail
SPLIT=${1:?split}
E=/scratch/t509a04/data/EgoDex
PY=/home01/t509a04/envs/egodex_x86/bin/python
cd /scratch/t509a04/workspace/starVLA_car
if [ ! -d ${E}/raw/${SPLIT} ]; then
  echo "[$(date)] unzip ${SPLIT}"; (cd ${E}/raw && unzip -q -n ${E}/zips/${SPLIT}.zip)
fi
echo "[$(date)] convert ${SPLIT}"
${PY} examples/realRobots/EgoDex/tools/egodex_frames.py ${E}/raw ${E}/frames --splits ${SPLIT} --workers 240
[ "${SPLIT}" != test ] && rm -rf ${E}/raw/${SPLIT} || true
echo "[$(date)] done"
