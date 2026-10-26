#!/bin/bash
# [optional: waits for a pid to exit and the GPUs to drain] then QwenGR00T_RadioAdaLN (starvla_qwen35_radioadaln_allft_flow_rcfull_100k.yaml)
# via dual_system/run_dualsystem_joint_rcfull.sh (train -> seeded 24-task sim eval: absolute actions, state, 256 x 256,
# 12 of 16 actions per call) -> paired comparisons on the same scenes vs the am5reg DualSystem (49.8%), staged and joint.
#
#   GPUS=4,5,6,7 setsid nohup bash examples/simBenchmarks/Robocasa_tabletop/train_files/radio_adaln/run_radioadaln_rcfull.sh \
#     [pid to wait for] > radioadaln_chain.log 2>&1 < /dev/null &
set -u
cd "$(dirname "$0")/../../../../.."   # repo root
WAIT_PID=${1:-}
GPUS=${GPUS:-4,5,6,7}
DD=examples/simBenchmarks/Robocasa_tabletop/train_files/dual_system
YAML=${YAML:-examples/simBenchmarks/Robocasa_tabletop/train_files/radio_adaln/starvla_qwen35_radioadaln_allft_flow_rcfull_100k.yaml}
RID=$(grep '^run_id:' "$YAML" | awk '{print $2}')
STEPS=$(grep -E '^\s+max_train_steps:' "$YAML" | awk '{print $2}')
CKPT=playground/Checkpoints/$RID/checkpoints/steps_${STEPS}_pytorch_model.pt
PY=$HOME/miniconda3/envs/starVLA/bin/python
if [ -n "$WAIT_PID" ]; then
    echo "[radioadaln] waiting for pid $WAIT_PID $(date)"
    while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 60; done
fi
i=0
while :; do  # never start on top of another job: every GPU of the group under 3 GB, 3 checks in a row 1 min apart
    busy=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits -i "$GPUS" | awk -F', ' '$2 > 3000' | wc -l)
    if [ "$busy" -eq 0 ]; then ok=$((${ok:-0} + 1)); [ "$ok" -ge 3 ] && break; else ok=0; fi
    [ $((i % 10)) -eq 0 ] && echo "[radioadaln] waiting: $busy GPU(s) of $GPUS above 3 GB $(date)"
    i=$((i + 1)); sleep 60
done
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader -i "$GPUS"

unset ROBOCASA_VLM_PREV_FRAME ROBOCASA_VLM_PREV_FRAMES ROBOCASA_ACTION_MODE ROBOCASA_ACTION_MODE_KEYS
JOINT_YAML=$YAML GPUS=$GPUS TRAIN_PORT=${TRAIN_PORT:-29797} bash "$DD/run_dualsystem_joint_rcfull.sh"
echo "[radioadaln] train + eval chain rc=$? $(date)"
N1=$(ls -dt "$CKPT.log"/eval_2* 2>/dev/null | head -1)
[ -n "$N1" ] || { echo "[radioadaln] no eval dir under $CKPT.log - stop"; exit 1; }
A=playground/Checkpoints/starvla_dualsystem_joint_plancond_kf025_am5reg_h16_rcfull_robocasa_gr1/checkpoints/steps_100000_pytorch_model.pt.log/eval_20261018_032315
S=results/eval_logs/starvla_dualsystem_s1plancond_h16_rcfull_robocasa_gr1/eval_20261013_030712
J=results/eval_logs/starvla_dualsystem_joint_plancond_h16_rcfull_robocasa_gr1/eval_20261014_071215
echo "===== am5reg DualSystem (49.8%) vs radioadaln"; $PY $DD/paired_eval_compare.py "$A" "$N1" am5reg radioadaln | tail -4
echo "===== staged plancond vs radioadaln"; $PY $DD/paired_eval_compare.py "$S" "$N1" staged radioadaln | tail -4
echo "===== joint plancond vs radioadaln"; $PY $DD/paired_eval_compare.py "$J" "$N1" joint radioadaln | tail -4
echo "[radioadaln] ALL DONE $(date)"
