#!/bin/bash
# Convert the downloaded RoboTwin 2.0 aloha-agilex zips (HF TianxingChen/RoboTwin2.0, dataset/<task>/*.zip) into LeRobot v2.0
# under /scratch/t509a04/data/RoboTwin2.0/lerobot/{Clean,Randomized}/<task> (playground/Datasets/RoboTwin points there).
# Randomized zips that were not downloaded are skipped, so this is clean-only unless the randomized_500 zips are present.
#   [submit] examples/simBenchmarks/Robotwin/train_files/hangang/convert_robotwin.sbatch
#SBATCH -J rt_convert
#SBATCH -p natStr
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=72
#SBATCH --exclusive
#SBATCH -t 12:00:00
#SBATCH --comment="field=robotics;appl=deepspeed"
#SBATCH -o /scratch/t509a04/workspace/logs/%x_%j.out

set -uo pipefail
REPO=/scratch/t509a04/workspace/starVLA_car
DATA=/scratch/t509a04/data/RoboTwin2.0
RAW=$DATA/raw/dataset
OUT=$DATA/lerobot
PY=/home01/t509a04/envs/starVLA/bin/python
export PYTHONPATH=$DATA/pylib   # h5py lives here (pip --target), so the training env stays untouched
export ROBOTWIN_EXTRACT_TMP=$DATA/_extract
mkdir -p "$ROBOTWIN_EXTRACT_TMP" "$OUT/Clean" "$OUT/Randomized"
TASKS=$(cat $REPO/results/robotwin_all_tasks.txt)

for split in "Clean:aloha-agilex_clean_50" "Randomized:aloha-agilex_randomized_500"; do
  name="${split%%:*}"; zipbase="${split##*:}"
  for t in $TASKS; do
    zip="$RAW/$t/$zipbase.zip"; out="$OUT/$name/$t"
    [[ -f "$out/meta/info.json" ]] && { echo "[skip] $name/$t"; continue; }
    [[ -f "$zip" ]] || { echo "[missing] $zip"; continue; }
    echo "[convert] $name/$t  $(date)"
    "$PY" $REPO/examples/simBenchmarks/Robotwin/train_files/convert_robotwin_to_lerobot.py --zip "$zip" --out "$out" --workers 64 \
      || { echo "[FAIL] $name/$t"; rm -rf "$out"; }
  done
done
echo "ALL_CONVERT_DONE $(date)  clean=$(ls $OUT/Clean/*/meta/info.json 2>/dev/null | wc -l)/50"
