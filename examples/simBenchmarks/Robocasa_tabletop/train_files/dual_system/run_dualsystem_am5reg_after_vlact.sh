#!/bin/bash
# [optional: waits for a pid (e.g. the VLAct-4B memory chain) to exit and the GPUs to drain] then QwenGR00T_DualSystem with Qwen3.5-0.8B fully fine-tuned on the current frame (no VLM lag), 0.25 s absolute
# keyframes and an ActionMap System 1 trained by REGRESSION, no memory (joint_system12_plancond_kf025_am5reg_rcfull_100k.yaml)
# via run_dualsystem_joint_rcfull.sh (train -> seeded 24-task sim eval, client sends frame t only) -> paired comparisons
# on the same scenes.
#
#   GPUS=4,5,6,7 setsid nohup bash examples/simBenchmarks/Robocasa_tabletop/train_files/dual_system/run_dualsystem_am5reg_after_vlact.sh \
#     [pid to wait for] > dualsystem_am5reg_chain.log 2>&1 < /dev/null &
set -u
cd "$(dirname "$0")/../../../../.."   # repo root
WAIT_PID=${1:-}
GPUS=${GPUS:-4,5,6,7}
DIR=examples/simBenchmarks/Robocasa_tabletop/train_files/dual_system
YAML=${YAML:-$DIR/joint_system12_plancond_kf025_am5reg_rcfull_100k.yaml}
RID=$(grep '^run_id:' "$YAML" | awk '{print $2}')
STEPS=$(grep -E '^\s+max_train_steps:' "$YAML" | awk '{print $2}')
CKPT=playground/Checkpoints/$RID/checkpoints/steps_${STEPS}_pytorch_model.pt
PY=$HOME/miniconda3/envs/starVLA/bin/python
if [ -n "$WAIT_PID" ]; then
    echo "[am5reg] waiting for pid $WAIT_PID $(date)"
    while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 60; done
    # eval servers / sim workers of the previous chain must be gone: every GPU of the group under 3 GB (at most 30 min)
    for i in $(seq 30); do
        busy=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits -i "$GPUS" | awk -F', ' '$2 > 3000' | wc -l)
        [ "$busy" -eq 0 ] && break
        echo "[am5reg] $busy GPU(s) of $GPUS still above 3 GB $(date)"; sleep 60
    done
fi
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader -i "$GPUS"

unset ROBOCASA_VLM_PREV_FRAME ROBOCASA_VLM_PREV_FRAMES
JOINT_YAML=$YAML GPUS=$GPUS TRAIN_PORT=${TRAIN_PORT:-29797} bash "$DIR/run_dualsystem_joint_rcfull.sh"
rc=$?
echo "[am5reg] train + eval chain rc=$rc $(date)"
N1=$(ls -dt "$CKPT.log"/eval_2* 2>/dev/null | head -1)
[ -n "$N1" ] || { echo "[am5reg] no eval dir under $CKPT.log - stop"; exit 1; }
grep -l "vlm_prev_frame" "$N1"/eval_env_*.log > /dev/null 2>&1 && echo "[am5reg] WARNING: clients sent previous-call frames (VLM lag)" \
    || echo "[am5reg] clients send frame t only (no VLM lag)"
J=results/eval_logs/starvla_dualsystem_joint_plancond_h16_rcfull_robocasa_gr1/eval_20261014_071215
O=playground/Checkpoints/starvla_dualsystem_joint_plancond_vlm12_s2delta_s1res10_h16_rcfull_robocasa_gr1/checkpoints/steps_50000_pytorch_model.pt.log
O1=$O/eval_20261015_042524; O8=$O/eval_20261015_080006
V=$(ls -dt playground/Checkpoints/starvla_dualsystem_joint_vlact4b_l4_kf025_mem_am5_vlm12_h16_rcfull_robocasa_gr1/checkpoints/steps_50000_pytorch_model.pt.log/eval_2* | head -1)
S=results/eval_logs/starvla_dualsystem_s1plancond_h16_rcfull_robocasa_gr1/eval_20261013_030712
echo "===== joint plancond (no lag, 0.1 s keyframes, flow S1) vs am5reg"; $PY $DIR/paired_eval_compare.py "$J" "$N1" joint am5reg | tail -4
echo "===== s1res10 (VLM lag, delta plan + residual flow S1) 1 sample vs am5reg"; $PY $DIR/paired_eval_compare.py "$O1" "$N1" s1res10_x1 am5reg | tail -4
echo "===== s1res10 mean of 8 vs am5reg"; $PY $DIR/paired_eval_compare.py "$O8" "$N1" s1res10_mean8 am5reg | tail -4
echo "===== VLAct-4B memory (ActionMap flow S1) vs am5reg"; $PY $DIR/paired_eval_compare.py "$V" "$N1" vlact4b_mem am5reg | tail -4
echo "===== staged plancond (no lag) vs am5reg"; $PY $DIR/paired_eval_compare.py "$S" "$N1" staged am5reg | tail -4
echo "[am5reg] ALL DONE $(date)"
