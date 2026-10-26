#!/bin/bash
# Keyframe-spacing ablation of the am5reg run (sim 49.8%): joint_system12_plancond_kf010_am5reg_rcfull_100k.yaml (0.1 s
# keyframes, else identical) via run_dualsystem_joint_rcfull.sh (train -> seeded 24-task sim eval, frame t only) ->
# paired comparisons on the same scenes vs kf 0.25 am5reg, joint plancond (0.1 s, flow S1, 50k) and staged plancond.
#
#   GPUS=4,5,6,7 setsid nohup bash examples/simBenchmarks/Robocasa_tabletop/train_files/dual_system/run_dualsystem_kf010_am5reg.sh \
#     > dualsystem_kf010_am5reg_chain.log 2>&1 < /dev/null &
set -u
cd "$(dirname "$0")/../../../../.."   # repo root
GPUS=${GPUS:-4,5,6,7}
DIR=examples/simBenchmarks/Robocasa_tabletop/train_files/dual_system
YAML=${YAML:-$DIR/joint_system12_plancond_kf010_am5reg_rcfull_100k.yaml}
RID=$(grep '^run_id:' "$YAML" | awk '{print $2}')
STEPS=$(grep -E '^\s+max_train_steps:' "$YAML" | awk '{print $2}')
CKPT=playground/Checkpoints/$RID/checkpoints/steps_${STEPS}_pytorch_model.pt
PY=$HOME/miniconda3/envs/starVLA/bin/python
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader -i "$GPUS"

unset ROBOCASA_VLM_PREV_FRAME ROBOCASA_VLM_PREV_FRAMES
JOINT_YAML=$YAML GPUS=$GPUS TRAIN_PORT=${TRAIN_PORT:-29797} bash "$DIR/run_dualsystem_joint_rcfull.sh"
rc=$?
echo "[kf010] train + eval chain rc=$rc $(date)"
N1=$(ls -dt "$CKPT.log"/eval_2* 2>/dev/null | head -1)
[ -n "$N1" ] || { echo "[kf010] no eval dir under $CKPT.log - stop"; exit 1; }
grep -l "vlm_prev_frame" "$N1"/eval_env_*.log > /dev/null 2>&1 && echo "[kf010] WARNING: clients sent previous-call frames (VLM lag)" \
    || echo "[kf010] clients send frame t only (no VLM lag)"
A=playground/Checkpoints/starvla_dualsystem_joint_plancond_kf025_am5reg_h16_rcfull_robocasa_gr1/checkpoints/steps_100000_pytorch_model.pt.log/eval_20261018_032315
J=results/eval_logs/starvla_dualsystem_joint_plancond_h16_rcfull_robocasa_gr1/eval_20261014_071215
S=results/eval_logs/starvla_dualsystem_s1plancond_h16_rcfull_robocasa_gr1/eval_20261013_030712
echo "===== am5reg kf 0.25 s vs kf 0.1 s (per task)"; $PY $DIR/paired_eval_compare.py "$A" "$N1" kf025 kf010
echo "===== joint plancond (kf 0.1 s, flow S1, 50k) vs kf010 am5reg"; $PY $DIR/paired_eval_compare.py "$J" "$N1" joint kf010 | tail -4
echo "===== staged plancond vs kf010 am5reg"; $PY $DIR/paired_eval_compare.py "$S" "$N1" staged kf010 | tail -4
echo "[kf010] ALL DONE $(date)"
