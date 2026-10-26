#!/bin/bash
# s1res10 50k checkpoint, seeded 24-task sim eval with the VLM lag where the actions are System 2's PLAN itself on the
# System 1 window (STARVLA_DUAL_ACTION_SOURCE=s2_plan -> inference.action_source, System 1 unused), then paired
# comparisons (paired_eval_compare.py) vs System 1 x1 / mean of 8 on the same checkpoint and vs joint / staged.
# Runs on a SYMLINKED copy of the run dir (<run>_s2plan) so its eval log dir cannot be confused with the concurrent
# mean-of-8 eval of the same checkpoint (eval_dualsystem_rc100.sh picks the newest <ckpt>.log/eval_* dir).
#
#   setsid nohup bash examples/simBenchmarks/Robocasa_tabletop/train_files/dual_system/run_eval_s2plan.sh \
#     <s1mean waiter pid> > dualsystem_s1res10_s2plan_eval_chain.log 2>&1 < /dev/null &
set -u
cd "$(dirname "$0")/../../../../.."   # repo root
WAIT_PID=${1:-}
DIR=examples/simBenchmarks/Robocasa_tabletop/train_files/dual_system
RID=starvla_dualsystem_joint_plancond_vlm12_s2delta_s1res10_h16_rcfull_robocasa_gr1
SRC=playground/Checkpoints/$RID
DST=playground/Checkpoints/${RID}_s2plan
mkdir -p "$DST/checkpoints"
for f in config.yaml config.full.yaml dataset_statistics.json; do ln -sfn "$PWD/$SRC/$f" "$DST/$f"; done
ln -sfn "$PWD/$SRC/checkpoints/steps_50000_pytorch_model.pt" "$DST/checkpoints/steps_50000_pytorch_model.pt"
CKPT=$DST/checkpoints/steps_50000_pytorch_model.pt
TAG=dualsystem_joint_plancond_vlm12_s2delta_s1res10_h16_rcfull_50000_s2plan
PY=$HOME/miniconda3/envs/starVLA/bin/python
STARVLA_DUAL_ACTION_SOURCE=s2_plan ROBOCASA_VLM_PREV_FRAME=1 GPUS="4 5 6 7" bash "$DIR/eval_dualsystem_rc100.sh" "$CKPT" "$TAG"
echo "[s2plan] eval rc=$? $(date)"
LP=$(ls -dt "$CKPT.log"/eval_2* | head -1)
grep -l "System 2's plan on the System 1 window" "$LP"/server_gpu*.log > /dev/null 2>&1 && echo "[s2plan] servers confirm action_source s2_plan" \
    || echo "[s2plan] WARNING: server logs do not show action_source s2_plan"
L=$SRC/checkpoints/steps_50000_pytorch_model.pt.log
L1=$L/eval_20261015_042524
LK=$L/eval_20261015_080006
J=results/eval_logs/starvla_dualsystem_joint_plancond_h16_rcfull_robocasa_gr1/eval_20261014_071215
S=results/eval_logs/starvla_dualsystem_s1plancond_h16_rcfull_robocasa_gr1/eval_20261013_030712
echo "===== System 1 x1 vs System 2 plan (same ckpt, same scenes)"; $PY $DIR/paired_eval_compare.py "$L1" "$LP" s1x1 s2plan
echo "===== vs joint plancond (no lag, abs plan)"; $PY $DIR/paired_eval_compare.py "$J" "$LP" joint s2plan | tail -4
echo "===== vs staged plancond"; $PY $DIR/paired_eval_compare.py "$S" "$LP" staged s2plan | tail -4
[ -n "$WAIT_PID" ] && while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 120; done
echo "===== System 1 mean of 8 vs System 2 plan"; $PY $DIR/paired_eval_compare.py "$LK" "$LP" s1mean8 s2plan
echo "[s2plan] ALL DONE $(date)"
