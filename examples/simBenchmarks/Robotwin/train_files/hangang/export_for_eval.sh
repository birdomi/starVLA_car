#!/bin/bash
# Hangang GH200 nodes are aarch64 and SAPIEN has no aarch64 build, so RoboTwin eval runs on the x86 box instead.
# This collects everything that box needs into one directory laid out like the repo root (hardlinks, no extra space):
#   playground/Checkpoints/<run_id>/{config.yaml,dataset_statistics.json,checkpoints/steps_<N>_pytorch_model.pt}
#   playground/Pretrained_models/Qwen3.5-2B      (base_vlm in the yaml; the model is built from it before loading weights)
#   examples/simBenchmarks/Robotwin/train_files/<yaml>, results/robotwin_all_tasks.txt
#   starvla_local.patch + COMMIT                  (uncommitted code the checkpoints were trained with)
# Usage (repo root, after training): bash examples/simBenchmarks/Robotwin/train_files/hangang/export_for_eval.sh <yaml>... [STEPS=N]
# Then on the x86 box: rsync -a <hangang>:$OUT/ <starVLA repo>/   (see EVAL_ON_X86.md written next to it)
set -euo pipefail
cd "$(dirname "$0")/../../../../.."   # repo root
OUT=${OUT:-/scratch/t509a04/workspace/robotwin_export}
mkdir -p "$OUT/results"

for YAML in "$@"; do
  RID=$(grep '^run_id:' "$YAML" | awk '{print $2}')
  N=${STEPS:-$(grep -E '^\s+max_train_steps:' "$YAML" | awk '{print $2}')}
  SRC=playground/Checkpoints/$RID
  CKPT=$SRC/checkpoints/steps_${N}_pytorch_model.pt
  [ -f "$CKPT" ] || { echo "checkpoint missing: $CKPT"; exit 1; }
  DST=$OUT/playground/Checkpoints/$RID
  mkdir -p "$DST/checkpoints" "$OUT/$(dirname "$YAML")"
  cp -f "$SRC/config.yaml" "$SRC/dataset_statistics.json" "$DST/"
  ln -f "$CKPT" "$DST/checkpoints/"
  cp -f "$YAML" "$OUT/$YAML"
  BASE=$(grep -E '^\s+base_vlm:' "$YAML" | awk '{print $2}')
  [ -d "$OUT/$BASE" ] || { mkdir -p "$OUT/$(dirname "$BASE")"; cp -rlL "$BASE" "$OUT/$BASE"; }
  echo "[export] $RID steps_$N"
done

cp -f results/robotwin_all_tasks.txt "$OUT/results/"
git rev-parse HEAD > "$OUT/COMMIT"
git diff HEAD -- starVLA deployment > "$OUT/starvla_local.patch"
cat > "$OUT/EVAL_ON_X86.md" <<EOF
# RoboTwin eval of the Hangang checkpoints on the x86 box

1. Pull the bundle into the starVLA repo root (paths inside mirror the repo):
   rsync -a --info=progress2 <hangang>:$OUT/ /path/to/starVLA_car/
2. Match the training code:
   git checkout \$(cat COMMIT) && git apply starvla_local.patch
3. Eval only (one yaml per call so EVAL_TAG stays unique):
   SKIP_TRAIN=1 GPUS=0,1,2,3 EVAL_TAG=2b_rt50k_dit_v2 \\
     bash examples/simBenchmarks/Robotwin/train_files/run_robotwin_train_eval.sh \\
     examples/simBenchmarks/Robotwin/train_files/starvla_qwen35_2b_gr00t_rt50k_dit_v2_robotwin.yaml
   SKIP_TRAIN=1 GPUS=0,1,2,3 EVAL_TAG=2b_rt50k_actionmap_v3 \\
     bash examples/simBenchmarks/Robotwin/train_files/run_robotwin_train_eval.sh \\
     examples/simBenchmarks/Robotwin/train_files/starvla_qwen35_2b_gr00t_rt50k_actionmap_v3_robotwin.yaml
   -> results/robotwin_<EVAL_TAG>_demo_clean_n20_results.csv
EOF
du -shL "$OUT"; echo "-> $OUT/EVAL_ON_X86.md"
