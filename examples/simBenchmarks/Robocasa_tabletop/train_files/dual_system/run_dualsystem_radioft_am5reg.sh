#!/bin/bash
# C-RADIO fine-tuning ablation of the am5reg run (sim 49.8%): joint_system12_plancond_kf025_am5reg_radioft_rcfull_100k.yaml
# (radio.trainable true, radio lr 1e-5, else identical) via run_dualsystem_joint_rcfull.sh (train -> seeded 24-task sim
# eval, frame t only) -> paired comparisons on the same scenes vs frozen-RADIO am5reg kf025, joint plancond and staged.
#
#   GPUS=0,1,2,3 TRAIN_PORT=29811 setsid nohup bash examples/simBenchmarks/Robocasa_tabletop/train_files/dual_system/run_dualsystem_radioft_am5reg.sh \
#     > dualsystem_radioft_am5reg_chain.log 2>&1 < /dev/null &
set -u
cd "$(dirname "$0")/../../../../.."   # repo root
GPUS=${GPUS:-0,1,2,3}
DIR=examples/simBenchmarks/Robocasa_tabletop/train_files/dual_system
YAML=${YAML:-$DIR/joint_system12_plancond_kf025_am5reg_radioft_rcfull_100k.yaml}
RID=$(grep '^run_id:' "$YAML" | awk '{print $2}')
STEPS=$(grep -E '^\s+max_train_steps:' "$YAML" | awk '{print $2}')
CKPT=playground/Checkpoints/$RID/checkpoints/steps_${STEPS}_pytorch_model.pt
PY=$HOME/miniconda3/envs/starVLA/bin/python
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader -i "$GPUS"

unset ROBOCASA_VLM_PREV_FRAME ROBOCASA_VLM_PREV_FRAMES STARVLA_DUAL_S1_ITERS STARVLA_DUAL_CORRECTOR STARVLA_DUAL_FASTS1
JOINT_YAML=$YAML GPUS=$GPUS TRAIN_PORT=${TRAIN_PORT:-29811} bash "$DIR/run_dualsystem_joint_rcfull.sh"
rc=$?
echo "[radioft] train + eval chain rc=$rc $(date)"
N1=$(ls -dt "$CKPT.log"/eval_2* 2>/dev/null | head -1)
[ -n "$N1" ] || { echo "[radioft] no eval dir under $CKPT.log - stop"; exit 1; }
grep -l "vlm_prev_frame" "$N1"/eval_env_*.log > /dev/null 2>&1 && echo "[radioft] WARNING: clients sent previous-call frames (VLM lag)" \
    || echo "[radioft] clients send frame t only (no VLM lag)"
A=playground/Checkpoints/starvla_dualsystem_joint_plancond_kf025_am5reg_h16_rcfull_robocasa_gr1/checkpoints/steps_100000_pytorch_model.pt.log/eval_20261018_032315
J=results/eval_logs/starvla_dualsystem_joint_plancond_h16_rcfull_robocasa_gr1/eval_20261014_071215
S=results/eval_logs/starvla_dualsystem_s1plancond_h16_rcfull_robocasa_gr1/eval_20261013_030712
echo "===== am5reg frozen RADIO vs fine-tuned RADIO (per task)"; $PY $DIR/paired_eval_compare.py "$A" "$N1" radio_frozen radio_ft
echo "===== joint plancond (kf 0.1 s, flow S1, 50k) vs radioft am5reg"; $PY $DIR/paired_eval_compare.py "$J" "$N1" joint radio_ft | tail -4
echo "===== staged plancond vs radioft am5reg"; $PY $DIR/paired_eval_compare.py "$S" "$N1" staged radio_ft | tail -4
echo "[radioft] ALL DONE $(date)"
